"""Dynamic-resolution JiT dehazing training.

This is a separate entry point and does not alter the fixed-resolution
training code.  Every training batch uses one rectangular crop size selected
from ``--train_sizes``.  Validation runs at each SOTS image's native size.
"""

import argparse
import datetime
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import util.lr_sched as lr_sched
import util.misc as misc
from data.dataset_dynamic import (
    DynamicRESIDEDataset,
    DynamicSOTSDataset,
    MultiScaleBatchSampler,
    parse_train_sizes,
)
from denoiser_rest_dynamic import DynamicRestorationDenoiser
from test_restoration import heatmap, region_metrics, tensor_rgb
from train_restoration import get_args_parser as get_fixed_parser


def get_args_parser():
    parser = get_fixed_parser()
    parser.description = "Dynamic-resolution JiT restoration"
    parser.add_argument(
        "--train_sizes",
        default="256x256,320x320,384x384,384x512,512x384,512x512",
        type=str,
        help="Comma-separated HxW crop buckets; each side must divide by 16",
    )
    parser.add_argument(
        "--init_checkpoint",
        default="",
        type=str,
        help="Optional fixed/dynamic JiT checkpoint for weight initialization",
    )
    parser.set_defaults(
        output_dir="./output/jit_reside_dynamic",
        img_size=512,
        patch_size=512,
    )
    return parser


def to_norm(tensor):
    return tensor * 2.0 - 1.0


def to_01(tensor):
    return (tensor.clamp(-1, 1) + 1.0) * 0.5


def batch_psnr(prediction, target):
    mse = (
        (prediction - target)
        .pow(2)
        .mean(dim=(1, 2, 3))
        .clamp_min(1e-10)
    )
    return -10.0 * torch.log10(mse)


def autocast_context(device):
    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def save_native_panels(panels, path, panel_height, max_columns=6):
    """Save labeled panels while preserving the source aspect ratio."""
    label_height = 28
    source_h, source_w = panels[0][1].shape[:2]
    panel_width = max(1, int(round(panel_height * source_w / source_h)))
    columns = min(max_columns, len(panels))
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new(
        "RGB",
        (
            columns * panel_width,
            rows * (panel_height + label_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(panels):
        row, column = divmod(index, columns)
        left = column * panel_width
        top = row * (panel_height + label_height)
        image = Image.fromarray(array).resize(
            (panel_width, panel_height),
            Image.Resampling.BILINEAR,
        )
        canvas.paste(image, (left, top + label_height))
        draw.text((left + 5, top + 7), label, fill="black")
    canvas.save(path)


@torch.no_grad()
def save_dynamic_diagnostics(
    model,
    degraded,
    clean,
    stem,
    device,
    args,
    sample_index,
    output_dir,
):
    steps = sorted(
        {
            int(value)
            for value in args.diagnostic_steps.split(",")
            if value.strip()
        }
    )
    if not steps or steps[0] < 1:
        raise ValueError(
            "--diagnostic_steps must contain positive integers"
        )

    outputs = {}
    with autocast_context(device):
        for step_count in steps:
            generator = torch.Generator(device=device).manual_seed(
                args.eval_seed + sample_index
            )
            outputs[step_count] = model.restore(
                degraded,
                generator=generator,
                steps=step_count,
                method=args.sampling_method,
            )
        generator = torch.Generator(device=device).manual_seed(
            args.eval_seed + sample_index
        )
        _, trajectory = model.restore_trajectory(
            degraded,
            generator=generator,
            steps=steps[-1],
            method=args.sampling_method,
        )

    degraded_01 = to_01(degraded)[0]
    clean_01 = clean[0]
    outputs_01 = {
        step_count: to_01(output)[0]
        for step_count, output in outputs.items()
    }
    trajectory_01 = [to_01(state)[0] for state in trajectory]
    height, width = degraded.shape[-2:]
    input_score = float(batch_psnr(
        degraded_01.unsqueeze(0),
        clean_01.unsqueeze(0),
    )[0])

    step_panels = [
        (
            f"LQ {height}x{width} | {input_score:.2f} dB",
            tensor_rgb(degraded_01),
        ),
        (
            f"Noisy start | strength={model.generation_strength:.2f}",
            tensor_rgb(trajectory_01[0]),
        ),
    ]
    for step_count in steps:
        output = outputs_01[step_count]
        score = float(batch_psnr(
            output.unsqueeze(0),
            clean_01.unsqueeze(0),
        )[0])
        step_panels.append(
            (
                f"{step_count} step | {score:.2f} dB",
                tensor_rgb(output),
            )
        )
    step_panels.append(("GT", tensor_rgb(clean_01)))
    save_native_panels(
        step_panels,
        output_dir / f"{stem}_steps.png",
        args.diagnostic_panel_size,
    )

    trajectory_panels = []
    t_start = 1.0 - model.generation_strength
    for trajectory_index, state in enumerate(trajectory_01):
        actual_t = (
            t_start
            + model.generation_strength
            * trajectory_index
            / steps[-1]
        )
        label = (
            f"start t={actual_t:.2f}"
            if trajectory_index == 0
            else f"t={actual_t:.2f}"
        )
        trajectory_panels.append((label, tensor_rgb(state)))
    trajectory_panels.append(("GT", tensor_rgb(clean_01)))
    save_native_panels(
        trajectory_panels,
        output_dir / f"{stem}_trajectory.png",
        args.diagnostic_panel_size,
    )

    restored = outputs_01[steps[-1]]
    _, maps = region_metrics(degraded_01, restored, clean_01)
    severity, correction, error_before, error_after = maps
    analysis_panels = [
        ("Original LQ", tensor_rgb(degraded_01)),
        ("Noisy start", tensor_rgb(trajectory_01[0])),
        ("GT", tensor_rgb(clean_01)),
        ("GT degradation", heatmap(severity)),
        ("Model correction", heatmap(correction)),
        ("Restored", tensor_rgb(restored)),
        ("Error before", heatmap(error_before)),
        ("Error after", heatmap(error_after)),
    ]
    save_native_panels(
        analysis_panels,
        output_dir / f"{stem}_pixel_analysis.png",
        args.diagnostic_panel_size,
    )


@torch.no_grad()
def evaluate(model, loader, device, args, epoch, log_writer=None):
    model.eval()
    scores = []
    saved_samples = 0
    saved_diagnostics = 0
    sample_dir = Path(args.output_dir) / "samples" / f"epoch{epoch:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for index, (meta, degraded, clean) in enumerate(loader):
        degraded = to_norm(degraded.to(device, non_blocking=True))
        clean = clean.to(device, non_blocking=True)
        generator = torch.Generator(device=device).manual_seed(
            args.eval_seed + index
        )
        with autocast_context(device):
            prediction = model.restore(degraded, generator=generator)
        prediction_01 = to_01(prediction)
        scores.extend(
            batch_psnr(prediction_01, clean).float().cpu().tolist()
        )
        if saved_samples < args.num_eval_images:
            panel = torch.cat(
                [to_01(degraded), prediction_01, clean],
                dim=0,
            )
            stem = Path(meta[0][0]).stem
            height, width = degraded.shape[-2:]
            save_image(
                panel,
                sample_dir / f"{stem}_{height}x{width}_native.png",
                nrow=3,
            )
            if saved_diagnostics < args.diagnostic_images:
                save_dynamic_diagnostics(
                    model,
                    degraded,
                    clean,
                    stem,
                    device,
                    args,
                    index,
                    sample_dir,
                )
                saved_diagnostics += degraded.size(0)
            saved_samples += degraded.size(0)

    mean_psnr = float(np.mean(scores)) if scores else 0.0
    print(f"[Dynamic Eval] epoch={epoch} PSNR={mean_psnr:.3f}")
    if log_writer is not None:
        log_writer.add_scalar("val/psnr_native", mean_psnr, epoch)
    model.train()
    return mean_psnr


def train_one_epoch(
    model,
    loader,
    batch_sampler,
    optimizer,
    device,
    epoch,
    args,
    log_writer=None,
):
    model.train(True)
    batch_sampler.set_epoch(epoch)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr",
        misc.SmoothedValue(window_size=1, fmt="{value:.6f}"),
    )
    metric_logger.add_meter(
        "height",
        misc.SmoothedValue(window_size=1, fmt="{value:.0f}"),
    )
    metric_logger.add_meter(
        "width",
        misc.SmoothedValue(window_size=1, fmt="{value:.0f}"),
    )
    header = f"Dynamic epoch: [{epoch}]"

    iterator = metric_logger.log_every(loader, args.log_freq, header)
    for step, (_, degraded, clean) in enumerate(iterator):
        lr_sched.adjust_learning_rate(
            optimizer,
            step / len(loader) + epoch,
            args,
        )
        degraded = to_norm(degraded.to(device, non_blocking=True))
        clean = to_norm(clean.to(device, non_blocking=True))
        with autocast_context(device):
            loss = model(clean, degraded)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss: {loss_value}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.update_ema()

        height, width = clean.shape[-2:]
        metric_logger.update(
            loss=loss_value,
            flow_loss=model.loss_terms["flow"].item(),
            l1_loss=model.loss_terms["l1"].item(),
            lr=optimizer.param_groups[0]["lr"],
            height=height,
            width=width,
        )
        if log_writer is not None and step % args.log_freq == 0:
            progress = int((step / len(loader) + epoch) * 1000)
            log_writer.add_scalar("train/loss", loss_value, progress)
            log_writer.add_scalar(
                "train/flow_loss",
                model.loss_terms["flow"].item(),
                progress,
            )
            log_writer.add_scalar(
                "train/l1_loss",
                model.loss_terms["l1"].item(),
                progress,
            )
            log_writer.add_scalar("train/height", height, progress)
            log_writer.add_scalar("train/width", width, progress)
            log_writer.add_scalar(
                "train/lr",
                optimizer.param_groups[0]["lr"],
                progress,
            )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {
        name: meter.global_avg
        for name, meter in metric_logger.meters.items()
    }


def checkpoint_path(path):
    if path and os.path.isdir(path):
        return os.path.join(path, "checkpoint-last.pth")
    return path


def load_initial_weights(model, path):
    path = checkpoint_path(path)
    if not path:
        return
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model.load_compatible_state_dict(state)
    print(f"Initialized dynamic JiT from {path}")


def run_dynamic_training(args, dataset_train, dataset_val):
    if args.timestamp_output and not args.resume and not args.evaluate_only:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(args.output_dir, timestamp)
    args.dynamic_resolution = True
    print(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.output_dir)

    train_sizes = parse_train_sizes(args.train_sizes, patch_size=16)
    print("Dynamic training sizes:", train_sizes)
    batch_sampler = MultiScaleBatchSampler(
        len(dataset_train),
        args.batch_size,
        train_sizes,
        shuffle=True,
        drop_last=True,
        seed=args.seed,
    )
    loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    model = DynamicRestorationDenoiser(args).to(device)
    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable params: {parameters / 1e6:.3f}M")
    optimizer = torch.optim.AdamW(
        misc.add_weight_decay(model, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    start_epoch = 0
    resume_path = checkpoint_path(args.resume)
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_compatible_state_dict(checkpoint["model"])
        if checkpoint.get("model_ema"):
            model.ema_params = [
                checkpoint["model_ema"][name].to(device)
                for name, _ in model.named_parameters()
            ]
        else:
            model.ema_params = [
                parameter.detach().clone()
                for parameter in model.parameters()
            ]
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(
            f"Resumed dynamic training from {resume_path}, "
            f"start_epoch={start_epoch}"
        )
    else:
        if args.init_checkpoint:
            load_initial_weights(model, args.init_checkpoint)
        model.ema_params = [
            parameter.detach().clone()
            for parameter in model.parameters()
        ]

    if args.evaluate_only:
        evaluate(
            model,
            loader_val,
            device,
            args,
            start_epoch,
            log_writer,
        )
        return

    print(f"Start dynamic training for {args.epochs} epochs")
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            model,
            loader_train,
            batch_sampler,
            optimizer,
            device,
            epoch,
            args,
            log_writer,
        )
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            ema_state = {
                name: parameter
                for (name, _), parameter in zip(
                    model.named_parameters(),
                    model.ema_params,
                )
            }
            state = {
                "model": model.state_dict(),
                "model_ema": ema_state,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            path = os.path.join(args.output_dir, "checkpoint-last.pth")
            temporary_path = path + ".tmp"
            torch.save(state, temporary_path)
            os.replace(temporary_path, path)
            print(f"Saved {path}")
        if epoch % args.eval_freq == 0 or epoch + 1 == args.epochs:
            evaluate(
                model,
                loader_val,
                device,
                args,
                epoch,
                log_writer,
            )
        log_writer.flush()

    elapsed = str(
        datetime.timedelta(seconds=int(time.time() - started))
    )
    print("Dynamic training time:", elapsed)


def main(args):
    dataset_train = DynamicRESIDEDataset(args)
    dataset_val = DynamicSOTSDataset(args)
    run_dynamic_training(args, dataset_train, dataset_val)


if __name__ == "__main__":
    parsed_args = get_args_parser().parse_args()
    Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)

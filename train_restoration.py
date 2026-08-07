"""
Train JiT as a rectified-flow dehazing model.

Protocol:
  - Train on RESIDE OTS.
  - Validate on the complete SOTS outdoor benchmark.

Key idea:
  z_t = t * x_clean + (1 - t) * noise
  predict clean x from cat([z_t, x_hazy])
  optimize velocity MSE plus explicit clean-image L1 supervision
  use an img2img generation strength to initialize sampling from noisy LQ

Recommended next experiments:
  1) Add a small stochastic bridge after the deterministic baseline works.
  2) Replace the dummy class embedding with task / text conditioning.

Example:
  python train_restoration.py \
      --data_file_dir ./datasets \
      --output_dir ./output/jit_reside_rectified \
      --model JiT-B/16 \
      --img_size 512 --patch_size 512 \
      --batch_size 2 --epochs 50 --lr 1e-4
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import util.lr_sched as lr_sched
import util.misc as misc
from data.dataset_utils import RESIDEDehazeDataset, SOTSDehazeDataset
from denoiser_rest import RestorationDenoiser
from test_restoration import (
    heatmap,
    region_metrics,
    restore_trajectory,
    save_panels,
    tensor_rgb,
)


def get_args_parser():
    parser = argparse.ArgumentParser("JiT restoration (RESIDE train / SOTS val)")

    # model
    parser.add_argument("--model", default="JiT-B/16", type=str)
    parser.add_argument("--img_size", default=512, type=int, help="Network input spatial size")
    parser.add_argument("--patch_size", default=512, type=int, help="Dataset crop size")
    parser.add_argument("--attn_dropout", default=0.0, type=float)
    parser.add_argument("--proj_dropout", default=0.0, type=float)
    parser.add_argument("--class_num", default=1, type=int, help="Dummy class table size")

    # rectified flow
    parser.add_argument("--P_mean", default=-0.8, type=float)
    parser.add_argument("--P_std", default=0.8, type=float)
    parser.add_argument("--noise_scale", default=1.0, type=float)
    parser.add_argument("--t_eps", default=5e-2, type=float)
    parser.add_argument("--cond_drop_prob", default=0.0, type=float)
    parser.add_argument("--lambda_flow", default=1.0, type=float)
    parser.add_argument("--lambda_l1", default=1.0, type=float)
    parser.add_argument("--prediction_type", default="conditional_x",
                        choices=["conditional_x"], type=str)
    parser.add_argument("--generation_strength", default=0.6, type=float,
                        help="Img2img noise strength: 0 copies LQ, 1 starts from pure noise")

    # sampling / eval
    parser.add_argument("--sampling_method", default="euler", type=str, choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", default=20, type=int)
    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--num_eval_images", default=4, type=int)
    parser.add_argument("--eval_seed", default=1234, type=int,
                        help="Fixed bridge-noise seed for comparable validation")
    parser.add_argument("--diagnostic_steps", default="1,4,20", type=str,
                        help="Step counts shown in periodic validation diagnostics")
    parser.add_argument("--diagnostic_images", default=2, type=int)
    parser.add_argument("--diagnostic_panel_size", default=256, type=int)

    # optim
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--blr", default=5e-5, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--lr_schedule", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--ema_decay", default=0.9999, type=float)

    # data
    parser.add_argument("--data_file_dir", default="./datasets", type=str)
    parser.add_argument("--reside_root", default=None, type=str,
                        help="Optional RESIDE root containing part1..part4 and clear")
    parser.add_argument("--sots_root", default=None, type=str,
                        help="Optional explicit SOTS outdoor root. Default: data_file_dir/dehazing/SOTS/outdoor")
    parser.add_argument("--reside_repeat", default=1, type=int,
                        help="Number of random haze variants sampled per clean scene each epoch")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.set_defaults(pin_mem=True)

    # runtime
    parser.add_argument("--output_dir", default="./output/jit_reside_rectified", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save_last_freq", default=1, type=int)
    parser.add_argument("--log_freq", default=20, type=int)
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--timestamp-output", dest="timestamp_output",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Append YYYYMMDD-HHMMSS to new training output directories")
    return parser


def to_norm(x):
    """[0,1] -> [-1,1]"""
    return x * 2.0 - 1.0


def to_01(x):
    """[-1,1] -> [0,1]"""
    return (x.clamp(-1, 1) + 1.0) * 0.5


def psnr(pred, target):
    mse = (pred - target).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return (-10.0 * torch.log10(mse)).mean().item()


@torch.no_grad()
def save_periodic_diagnostics(model, meta, lr, hr, device, args, epoch):
    """Save the same multi-step diagnostics used by test_restoration.py."""
    steps = sorted({int(value) for value in args.diagnostic_steps.split(",") if value.strip()})
    if not steps or steps[0] < 1:
        raise ValueError("--diagnostic_steps must contain positive integers")

    output_dir = Path(args.output_dir) / "samples" / f"epoch{epoch:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(args.diagnostic_images, lr.size(0))

    for sample_idx in range(count):
        degraded_01 = lr[sample_idx].to(device)
        clean = hr[sample_idx].to(device)
        degraded = to_norm(degraded_01.unsqueeze(0))
        generator = torch.Generator(device=device).manual_seed(
            args.eval_seed + sample_idx
        )
        initial_noise = torch.randn(
            degraded.shape,
            device=device,
            dtype=degraded.dtype,
            generator=generator,
        )

        step_outputs = {}
        longest_trajectory = None
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for step_count in steps:
                final, trajectory = restore_trajectory(
                    model,
                    degraded,
                    step_count,
                    args.sampling_method,
                    initial_noise,
                    record=step_count == steps[-1],
                )
                step_outputs[step_count] = to_01(final)[0]
                if trajectory is not None:
                    longest_trajectory = [to_01(value)[0] for value in trajectory]

        image_path = meta[0][sample_idx]
        stem = Path(image_path).stem
        input_score = psnr(degraded_01.unsqueeze(0), clean.unsqueeze(0))
        noisy_start = longest_trajectory[0]

        step_panels = [
            (f"Original LQ | {input_score:.2f} dB", tensor_rgb(degraded_01)),
            (
                f"Noisy start | strength={model.generation_strength:.2f}",
                tensor_rgb(noisy_start),
            ),
        ]
        for step_count in steps:
            output = step_outputs[step_count]
            output_score = psnr(output.unsqueeze(0), clean.unsqueeze(0))
            step_panels.append(
                (f"{step_count} step | {output_score:.2f} dB", tensor_rgb(output))
            )
        step_panels.append(("GT", tensor_rgb(clean)))
        save_panels(
            step_panels,
            output_dir / f"{stem}_steps.png",
            args.diagnostic_panel_size,
        )

        trajectory_panels = []
        t_start = 1.0 - model.generation_strength
        for trajectory_idx, value in enumerate(longest_trajectory):
            actual_t = (
                t_start
                + model.generation_strength * trajectory_idx / steps[-1]
            )
            label = (
                f"start t={actual_t:.2f}"
                if trajectory_idx == 0
                else f"t={actual_t:.2f}"
            )
            trajectory_panels.append((label, tensor_rgb(value)))
        trajectory_panels.append(("GT", tensor_rgb(clean)))
        save_panels(
            trajectory_panels,
            output_dir / f"{stem}_trajectory.png",
            args.diagnostic_panel_size,
        )

        restored = step_outputs[steps[-1]]
        _, maps = region_metrics(degraded_01, restored, clean)
        severity, correction, error_before, error_after = maps
        analysis_panels = [
            ("Original LQ", tensor_rgb(degraded_01)),
            ("Noisy start", tensor_rgb(noisy_start)),
            ("GT", tensor_rgb(clean)),
            ("GT degradation", heatmap(severity)),
            ("Model correction", heatmap(correction)),
            ("Restored", tensor_rgb(restored)),
            ("Error before", heatmap(error_before)),
            ("Error after", heatmap(error_after)),
        ]
        save_panels(
            analysis_panels,
            output_dir / f"{stem}_pixel_analysis.png",
            args.diagnostic_panel_size,
        )


@torch.no_grad()
def evaluate(model, loader, device, args, epoch, log_writer=None):
    model.eval()
    psnrs = []
    generator = torch.Generator(device=device).manual_seed(args.eval_seed)
    vis_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(vis_dir, exist_ok=True)

    for i, (meta, lr, hr) in enumerate(loader):
        y = to_norm(lr.to(device, non_blocking=True))
        x = to_norm(hr.to(device, non_blocking=True))
        pred = model.restore(y, generator=generator)
        psnrs.append(psnr(to_01(pred), to_01(x)))

        if i == 0:
            n = min(args.num_eval_images, y.size(0))
            grid = torch.cat([to_01(y[:n]), to_01(pred[:n]), to_01(x[:n])], dim=0)
            save_image(grid, os.path.join(vis_dir, f"epoch{epoch:04d}.png"), nrow=n)
            save_periodic_diagnostics(model, meta, lr, hr, device, args, epoch)

    mean_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    print(f"[Eval] epoch={epoch} PSNR={mean_psnr:.3f}")
    if log_writer is not None:
        log_writer.add_scalar("val/psnr", mean_psnr, epoch)
    model.train()
    return mean_psnr


def train_one_epoch(model, loader, optimizer, device, epoch, args, log_writer=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for data_iter_step, (meta, lr, hr) in enumerate(
        metric_logger.log_every(loader, args.log_freq, header)
    ):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(loader) + epoch, args)

        y = to_norm(lr.to(device, non_blocking=True))
        x = to_norm(hr.to(device, non_blocking=True))

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss: {loss_value}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.update_ema()

        metric_logger.update(loss=loss_value)
        metric_logger.update(flow_loss=model.loss_terms["flow"].item())
        metric_logger.update(l1_loss=model.loss_terms["l1"].item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if log_writer is not None and data_iter_step % args.log_freq == 0:
            epoch_1000x = int((data_iter_step / len(loader) + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value, epoch_1000x)
            log_writer.add_scalar("train/flow_loss", model.loss_terms["flow"].item(), epoch_1000x)
            log_writer.add_scalar("train/l1_loss", model.loss_terms["l1"].item(), epoch_1000x)
            log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def run_training(args, dataset_train, dataset_val):
    print("Job dir:", os.path.dirname(os.path.realpath(__file__)))
    if args.timestamp_output and not args.resume and not args.evaluate_only:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(args.output_dir, timestamp)
        print("Timestamped output dir:", args.output_dir)
    print(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.output_dir)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=min(args.batch_size, args.num_eval_images),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = RestorationDenoiser(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params / 1e6:.3f}M")

    eff_batch_size = args.batch_size
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"Base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"Actual lr: {args.lr:.2e}")
    print(f"Effective batch size: {eff_batch_size}")

    param_groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    start_epoch = 0
    ckpt_path = None
    if args.resume:
        ckpt_path = args.resume
        if os.path.isdir(args.resume):
            ckpt_path = os.path.join(args.resume, "checkpoint-last.pth")
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        checkpoint_prediction = getattr(ckpt.get("args"), "prediction_type", "v")
        if checkpoint_prediction != model.prediction_type:
            raise ValueError(
                "Checkpoint prediction type "
                f"{checkpoint_prediction!r} does not match requested "
                f"{model.prediction_type!r}; start a new output directory."
            )
        model.load_state_dict(ckpt["model"])
        if "model_ema" in ckpt:
            model.ema_params = [ckpt["model_ema"][name].to(device) for name, _ in model.named_parameters()]
        else:
            model.ema_params = [p.detach().clone() for p in model.parameters()]
        if "optimizer" in ckpt and "epoch" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {ckpt_path}, start_epoch={start_epoch}")
        del ckpt
    else:
        model.ema_params = [p.detach().clone() for p in model.parameters()]
        print("Training from scratch")

    if args.evaluate_only:
        evaluate(model, loader_val, device, args, start_epoch, log_writer)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, loader_train, optimizer, device, epoch, args, log_writer)

        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            ema_state = {name: p for (name, _), p in zip(model.named_parameters(), model.ema_params)}
            to_save = {
                "model": model.state_dict(),
                "model_ema": ema_state,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            path = os.path.join(args.output_dir, "checkpoint-last.pth")
            # Keep readers from observing a partially written multi-GB file.
            temporary_path = path + ".tmp"
            torch.save(to_save, temporary_path)
            os.replace(temporary_path, path)
            print(f"Saved {path}")

        if epoch % args.eval_freq == 0 or epoch + 1 == args.epochs:
            evaluate(model, loader_val, device, args, epoch, log_writer)

        log_writer.flush()

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print("Training time:", total_time)


def main(args):
    dataset_train = RESIDEDehazeDataset(args)
    dataset_val = SOTSDehazeDataset(args, split="test")
    run_training(args, dataset_train, dataset_val)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

"""Dynamic-resolution blind All-in-One JiT training with native-res eval."""

import argparse
import copy
import datetime
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import util.lr_sched as lr_sched
import util.misc as misc
from data.dataset_aio_dynamic import (
    DynamicAIOTrainDataset,
    make_balanced_multiscale_sampler,
    parse_train_sizes,
)
from data.dataset_utils import IRBenchmarks
from denoiser_all_in_one_dynamic import DynamicAllInOneRestorationDenoiser
from train_all_in_one import (
    BENCHMARKS,
    TASK_ORDER,
    replace_denoising_inputs,
)
from train_restoration import get_args_parser as get_fixed_parser
from train_restoration_dynamic import save_native_panels
from test_restoration import tensor_rgb


def get_args_parser():
    parser = get_fixed_parser()
    parser.description = "Dynamic-resolution JiT All-in-One restoration"
    for action in parser._actions:
        if action.dest == "sampling_method":
            action.choices = ["deterministic_bridge"]
    parser.add_argument(
        "--config",
        default="",
        type=str,
        help="YAML config; explicit command-line values override it",
    )
    parser.add_argument(
        "--de_type",
        nargs="+",
        default=TASK_ORDER,
        choices=TASK_ORDER,
        help="Keep the default order to preserve task-label semantics",
    )
    parser.add_argument(
        "--samples_per_epoch",
        default=6000,
        type=int,
        help="Balanced samples drawn with replacement each epoch",
    )
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
        help="Optional dynamic All-in-One checkpoint for weight init",
    )
    parser.add_argument(
        "--bsd68_root",
        default=None,
        type=str,
        help="Optional CBSD68 original_png directory",
    )
    parser.add_argument(
        "--bridge_steps",
        default=15,
        type=int,
        help="Number of canonical bridge intervals",
    )
    parser.add_argument(
        "--bridge_noise_shared",
        default=0.6,
        type=float,
        help="lambda_b for shared bridge noise lambda_b * t(1-t)",
    )
    parser.add_argument(
        "--bridge_noise_terminal",
        default=0.2,
        type=float,
        help="lambda_r for terminal relaxation lambda_r * t^2",
    )
    parser.add_argument(
        "--bridge_type",
        default="global_udbm_bridge",
        choices=["global_udbm_bridge"],
    )
    parser.add_argument(
        "--conditioning_type",
        default="state_and_degraded",
        choices=["state_and_degraded"],
    )
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
        default=-1,
        type=int,
    )
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.set_defaults(
        img_size=512,
        patch_size=512,
        output_dir="./output/jit_all_in_one_dynamic_expB_udbm_global",
        num_sampling_steps=15,
        sampling_method="deterministic_bridge",
    )
    return parser


def validate_args(args):
    if list(args.de_type) != TASK_ORDER:
        raise ValueError(
            "--de_type must keep this order: " + " ".join(TASK_ORDER)
        )
    if args.bridge_type != "global_udbm_bridge":
        raise ValueError(
            "Experiment B requires bridge_type=global_udbm_bridge"
        )
    if args.sampling_method != "deterministic_bridge":
        raise ValueError(
            "Experiment B requires sampling_method=deterministic_bridge"
        )
    if args.conditioning_type != "state_and_degraded":
        raise ValueError(
            "All-in-One training requires "
            "conditioning_type=state_and_degraded"
        )
    if args.bridge_steps < 1:
        raise ValueError("--bridge_steps must be positive")
    if args.num_sampling_steps > args.bridge_steps:
        raise ValueError(
            "--num_sampling_steps cannot exceed --bridge_steps "
            "(inference uses a subsequence of the trained schedule)"
        )


def to_norm(tensor):
    return tensor * 2.0 - 1.0


def to_01(tensor):
    return (tensor.clamp(-1, 1) + 1.0) * 0.5


def autocast_context(device):
    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def psnr_per_image(prediction, target):
    mse = (
        (prediction - target)
        .pow(2)
        .mean(dim=(1, 2, 3))
        .clamp_min(1e-10)
    )
    return -10.0 * torch.log10(mse)


def build_benchmark_loaders(args):
    loaders = {}
    for benchmark in BENCHMARKS:
        benchmark_args = copy.copy(args)
        benchmark_args.benchmarks = benchmark
        dataset = IRBenchmarks(benchmark_args)
        loaders[benchmark] = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )
    return loaders


@torch.no_grad()
def save_native_step_diagnostics(
    model,
    degraded,
    clean,
    benchmark,
    stem,
    sample_index,
    device,
    args,
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
    noise_generator = torch.Generator(device=device).manual_seed(
        args.eval_seed + sample_index
    )
    initial_noise = torch.randn(
        degraded.shape,
        device=device,
        dtype=degraded.dtype,
        generator=noise_generator,
    )
    outputs = {}
    trajectory = None
    trajectory_coeffs = None
    trajectory_x0 = None
    trajectory_steps = steps[-1]
    with autocast_context(device):
        for step_count in steps:
            reverse_generator = torch.Generator(device=device).manual_seed(
                args.eval_seed + sample_index + 100000
            )
            result = model.restore(
                degraded,
                generator=reverse_generator,
                initial_noise=initial_noise,
                steps=step_count,
                method=args.sampling_method,
                return_trajectory=step_count == trajectory_steps,
            )
            if step_count == trajectory_steps:
                (
                    outputs[step_count],
                    trajectory,
                    trajectory_coeffs,
                    trajectory_x0,
                ) = result
            else:
                outputs[step_count] = result

    degraded_01 = to_01(degraded)[0]
    clean_01 = clean[0]
    height, width = degraded.shape[-2:]
    input_score = float(
        psnr_per_image(
            degraded_01.unsqueeze(0),
            clean_01.unsqueeze(0),
        )[0]
    )
    noisy_start = model.make_initial_state(
        degraded,
        initial_noise,
        steps=steps[-1],
    )
    terminal_b = float(
        model._canonical_bridge_schedules(
            degraded.device,
            degraded.dtype,
        )[1][-1]
    )
    panels = [
        (
            f"{benchmark} LQ {height}x{width} | {input_score:.2f} dB",
            tensor_rgb(degraded_01),
        ),
        (
            f"LQ + noise | b_T={terminal_b:.2f}",
            tensor_rgb(to_01(noisy_start)[0]),
        ),
    ]
    for step_count in steps:
        output = to_01(outputs[step_count])[0]
        score = float(
            psnr_per_image(
                output.unsqueeze(0),
                clean_01.unsqueeze(0),
            )[0]
        )
        panels.append(
            (
                f"{step_count} step | {score:.2f} dB",
                tensor_rgb(output),
            )
        )
    panels.append(("GT", tensor_rgb(clean_01)))
    save_native_panels(
        panels,
        output_dir / f"{stem}_steps.png",
        args.diagnostic_panel_size,
    )

    trajectory_panels = [
        (
            f"{benchmark} LQ {height}x{width} | {input_score:.2f} dB",
            tensor_rgb(degraded_01),
        )
    ]
    for reverse_index, (state, coeffs) in enumerate(
        zip(trajectory, trajectory_coeffs)
    ):
        a_value, b_value = coeffs
        state_01 = to_01(state)[0]
        score = float(
            psnr_per_image(
                state_01.unsqueeze(0),
                clean_01.unsqueeze(0),
            )[0]
        )
        label = (
            f"start t={a_value:.3f} b={b_value:.3f} | {score:.2f} dB"
            if reverse_index == 0
            else (
                f"r{reverse_index:02d} t={a_value:.3f} b={b_value:.3f} | "
                f"{score:.2f} dB"
            )
        )
        trajectory_panels.append((label, tensor_rgb(state_01)))
    trajectory_panels.append(("GT", tensor_rgb(clean_01)))
    save_native_panels(
        trajectory_panels,
        output_dir / f"{stem}_trajectory.png",
        args.diagnostic_panel_size,
    )

    # Track whether multi-step x0 predictions improve along the bridge.
    x0_panels = [
        (
            f"{benchmark} LQ {height}x{width} | {input_score:.2f} dB",
            tensor_rgb(degraded_01),
        )
    ]
    for reverse_index, (clean_pred, coeffs) in enumerate(
        zip(trajectory_x0, trajectory_coeffs[:-1])
    ):
        a_value, b_value = coeffs
        pred_01 = to_01(clean_pred)[0]
        score = float(
            psnr_per_image(
                pred_01.unsqueeze(0),
                clean_01.unsqueeze(0),
            )[0]
        )
        x0_panels.append(
            (
                f"x0@{reverse_index + 1:02d} t={a_value:.3f} | {score:.2f} dB",
                tensor_rgb(pred_01),
            )
        )
    x0_panels.append(("GT", tensor_rgb(clean_01)))
    save_native_panels(
        x0_panels,
        output_dir / f"{stem}_x0_trajectory.png",
        args.diagnostic_panel_size,
    )


@torch.no_grad()
def evaluate_benchmark(
    model,
    loader,
    benchmark,
    device,
    args,
    epoch,
    output_root,
):
    """Evaluate at each image's native resolution (no fixed 256 crop)."""
    model.eval()
    scores = []
    maes = []
    benchmark_dir = Path(output_root) / benchmark
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    for index, (meta, degraded, clean) in enumerate(loader):
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        if benchmark.startswith("denoise_"):
            sigma = float(benchmark.rsplit("_", 1)[-1])
            noise_generator = torch.Generator(device=device).manual_seed(
                args.eval_seed + index + int(sigma) * 100000
            )
            degraded = (
                clean
                + torch.randn(
                    clean.shape,
                    device=device,
                    dtype=clean.dtype,
                    generator=noise_generator,
                )
                * sigma
                / 255.0
            ).clamp(0, 1)
        degraded = to_norm(degraded)
        generator = torch.Generator(device=device).manual_seed(
            args.eval_seed + index
        )
        with autocast_context(device):
            restored = model.restore(
                degraded,
                generator=generator,
                steps=args.num_sampling_steps,
                method=args.sampling_method,
            )
        restored = to_01(restored)
        scores.extend(psnr_per_image(restored, clean).cpu().tolist())
        maes.append(float((restored - clean).abs().mean()))
        if index < args.num_eval_images:
            stem = Path(meta[0][0]).stem
            height, width = degraded.shape[-2:]
            panel = torch.cat([to_01(degraded), restored, clean], dim=0)
            save_image(
                panel,
                benchmark_dir / f"{stem}_{height}x{width}_native.png",
                nrow=3,
            )
            if index < args.diagnostic_images:
                save_native_step_diagnostics(
                    model,
                    degraded,
                    clean,
                    benchmark,
                    stem,
                    index,
                    device,
                    args,
                    benchmark_dir,
                )
    result = {
        "psnr": float(np.mean(scores)),
        "mae": float(np.mean(maes)),
        "images": len(scores),
    }
    print(
        f"[{benchmark}] epoch={epoch} images={result['images']} "
        f"PSNR={result['psnr']:.3f} MAE={result['mae']:.5f} "
        f"(native resolution)"
    )
    return result


@torch.no_grad()
def evaluate_all(model, loaders, device, args, epoch, log_writer=None):
    output_root = Path(args.output_dir) / "samples" / f"epoch{epoch:04d}"
    results = {}
    for benchmark, loader in loaders.items():
        results[benchmark] = evaluate_benchmark(
            model,
            loader,
            benchmark,
            device,
            args,
            epoch,
            output_root,
        )
        if log_writer is not None:
            log_writer.add_scalar(
                f"val/{benchmark}_psnr",
                results[benchmark]["psnr"],
                epoch,
            )
            log_writer.add_scalar(
                f"val/{benchmark}_mae",
                results[benchmark]["mae"],
                epoch,
            )
    model.train()
    return results


def train_one_epoch(
    model,
    model_without_ddp,
    loader,
    optimizer,
    device,
    epoch,
    args,
    writer,
):
    model.train(True)
    logger = misc.MetricLogger(delimiter="  ")
    logger.add_meter(
        "lr",
        misc.SmoothedValue(window_size=1, fmt="{value:.6f}"),
    )
    logger.add_meter(
        "height",
        misc.SmoothedValue(window_size=1, fmt="{value:.0f}"),
    )
    logger.add_meter(
        "width",
        misc.SmoothedValue(window_size=1, fmt="{value:.0f}"),
    )
    degradation_totals = torch.zeros(
        len(TASK_ORDER),
        dtype=torch.long,
    )
    iterator = logger.log_every(
        loader,
        args.log_freq,
        f"Dynamic AIO epoch: [{epoch}]",
    )
    for step, (meta, degraded, clean) in enumerate(iterator):
        lr_sched.adjust_learning_rate(
            optimizer,
            step / len(loader) + epoch,
            args,
        )
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        degradation_ids = meta[1].to(device, non_blocking=True)
        degraded = replace_denoising_inputs(
            degraded,
            clean,
            degradation_ids,
        )
        degraded = to_norm(degraded)
        clean = to_norm(clean)
        with autocast_context(device):
            loss = model(clean, degraded)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss: {loss_value}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model_without_ddp.update_ema()

        height, width = clean.shape[-2:]
        degradation_totals += torch.bincount(
            degradation_ids.detach().cpu(),
            minlength=len(TASK_ORDER),
        )
        logger.update(
            loss=loss_value,
            flow_loss=model_without_ddp.loss_terms["flow"].item(),
            l1_loss=model_without_ddp.loss_terms["l1"].item(),
            t=model_without_ddp.loss_terms["t"].item(),
            b=model_without_ddp.loss_terms["b"].item(),
            lr=optimizer.param_groups[0]["lr"],
            height=height,
            width=width,
        )
        if writer is not None and step % args.log_freq == 0:
            progress = int((step / len(loader) + epoch) * 1000)
            writer.add_scalar("train/loss", loss_value, progress)
            writer.add_scalar(
                "train/flow_loss",
                model_without_ddp.loss_terms["flow"].item(),
                progress,
            )
            writer.add_scalar(
                "train/l1_loss",
                model_without_ddp.loss_terms["l1"].item(),
                progress,
            )
            writer.add_scalar(
                "train/t",
                model_without_ddp.loss_terms["t"].item(),
                progress,
            )
            writer.add_scalar(
                "train/b",
                model_without_ddp.loss_terms["b"].item(),
                progress,
            )
            writer.add_scalar("train/height", height, progress)
            writer.add_scalar("train/width", width, progress)
    logger.synchronize_between_processes()
    if args.distributed:
        totals = degradation_totals.to(device)
        dist.all_reduce(totals)
        degradation_totals = totals.cpu()
    print(
        "Sampled degradation counts:",
        dict(zip(TASK_ORDER, degradation_totals.tolist())),
    )
    print("Averaged stats:", logger)


def checkpoint_path(path):
    if path and os.path.isdir(path):
        return os.path.join(path, "checkpoint-last.pth")
    return path


def run_training(args):
    misc.init_distributed_mode(args)
    validate_args(args)
    args.dynamic_resolution = True
    if args.timestamp_output and not args.resume and not args.evaluate_only:
        timestamp = (
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            if misc.is_main_process()
            else None
        )
        if args.distributed:
            timestamp_holder = [timestamp]
            dist.broadcast_object_list(timestamp_holder, src=0)
            timestamp = timestamp_holder[0]
        args.output_dir = os.path.join(args.output_dir, timestamp)
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = (
        SummaryWriter(log_dir=args.output_dir)
        if misc.is_main_process()
        else None
    )
    device = torch.device(
        f"cuda:{args.gpu}" if args.distributed else args.device
    )
    rank = misc.get_rank()
    world_size = misc.get_world_size()
    print(
        f"DDP world size: {world_size}; batch per GPU: {args.batch_size}; "
        f"global batch: {args.batch_size * world_size}"
    )
    process_seed = args.seed + rank
    torch.manual_seed(process_seed)
    np.random.seed(process_seed)
    random.seed(process_seed)
    cudnn.benchmark = True

    train_sizes = parse_train_sizes(args.train_sizes, patch_size=16)
    print("Building multi-scale All-in-One train dataset...", flush=True)
    train_dataset = DynamicAIOTrainDataset(args)
    batch_sampler = make_balanced_multiscale_sampler(
        train_dataset,
        args.samples_per_epoch,
        args.batch_size,
        train_sizes,
        num_replicas=world_size,
        rank=rank,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    if misc.is_main_process():
        print("Building native-resolution benchmark loaders...", flush=True)
    benchmark_loaders = (
        build_benchmark_loaders(args) if misc.is_main_process() else {}
    )
    print("Creating DynamicAIO model on device...", flush=True)
    model_without_ddp = DynamicAllInOneRestorationDenoiser(args).to(device)
    print("Model ready.", flush=True)
    if misc.is_main_process():
        with torch.no_grad():
            a_schedule, b_schedule = (
                model_without_ddp._canonical_bridge_schedules(
                    device,
                    torch.float32,
                )
            )
        print("a_schedule =", a_schedule.detach().cpu().tolist(), flush=True)
        print("b_schedule =", b_schedule.detach().cpu().tolist(), flush=True)
        print(model_without_ddp.describe_bridge_schedule(), flush=True)
    optimizer = torch.optim.AdamW(
        misc.add_weight_decay(model_without_ddp, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    start_epoch = 0
    resume_path = checkpoint_path(args.resume)
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu")
        checkpoint_bridge = getattr(
            checkpoint.get("args"),
            "bridge_type",
            "noise_to_clean",
        )
        if checkpoint_bridge != args.bridge_type:
            raise ValueError(
                f"Cannot resume {checkpoint_bridge!r} checkpoint with "
                f"{args.bridge_type!r} training"
            )
        if not getattr(
            checkpoint.get("args"),
            "dynamic_resolution",
            False,
        ):
            raise ValueError(
                "Cannot resume a fixed-resolution checkpoint into "
                "dynamic All-in-One training; use --init_checkpoint only "
                "if weights are already DynamicAIO-compatible."
            )
        model_without_ddp.load_state_dict(checkpoint["model"])
        if checkpoint.get("model_ema"):
            model_without_ddp.ema_params = [
                checkpoint["model_ema"][name].to(device)
                for name, _ in model_without_ddp.named_parameters()
            ]
        else:
            model_without_ddp.ema_params = [
                parameter.detach().clone()
                for parameter in model_without_ddp.parameters()
            ]
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {resume_path}, start_epoch={start_epoch}")
    else:
        if args.init_checkpoint:
            checkpoint = torch.load(
                checkpoint_path(args.init_checkpoint),
                map_location="cpu",
            )
            model_without_ddp.load_state_dict(
                checkpoint.get("model", checkpoint),
                strict=False,
            )
            print(f"Initialized from {args.init_checkpoint}")
        model_without_ddp.ema_params = [
            parameter.detach().clone()
            for parameter in model_without_ddp.parameters()
        ]

    model = model_without_ddp
    if args.distributed:
        model = DistributedDataParallel(
            model_without_ddp,
            device_ids=[args.gpu],
            output_device=args.gpu,
        )

    if args.evaluate_only:
        if misc.is_main_process():
            evaluate_all(
                model_without_ddp,
                benchmark_loaders,
                device,
                args,
                start_epoch,
                writer,
            )
        if args.distributed:
            dist.barrier()
        if writer is not None:
            writer.close()
        if args.distributed:
            dist.destroy_process_group()
        return

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        batch_sampler.set_epoch(epoch)
        train_one_epoch(
            model,
            model_without_ddp,
            train_loader,
            optimizer,
            device,
            epoch,
            args,
            writer,
        )
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            if misc.is_main_process():
                ema_state = {
                    name: parameter
                    for (name, _), parameter in zip(
                        model_without_ddp.named_parameters(),
                        model_without_ddp.ema_params,
                    )
                }
                state = {
                    "model": model_without_ddp.state_dict(),
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
            if args.distributed:
                dist.barrier()
        if epoch % args.eval_freq == 0 or epoch + 1 == args.epochs:
            if misc.is_main_process():
                evaluate_all(
                    model_without_ddp,
                    benchmark_loaders,
                    device,
                    args,
                    epoch,
                    writer,
                )
            if args.distributed:
                dist.barrier()
        if writer is not None:
            writer.flush()
    elapsed = str(
        datetime.timedelta(seconds=int(time.time() - started))
    )
    print("Dynamic All-in-One training time:", elapsed)
    if writer is not None:
        writer.close()
    if args.distributed:
        dist.destroy_process_group()


def parse_args_with_config():
    parser = get_args_parser()
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        with open(preliminary.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            raise ValueError("YAML config root must be a mapping")
        valid_keys = {
            action.dest
            for action in parser._actions
            if action.dest != "help"
        }
        unknown = sorted(set(config) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown YAML config keys: {unknown}")
        parser.set_defaults(**config)
    return parser.parse_args()


if __name__ == "__main__":
    # Unbuffered progress so interactive shells show activity during slow startup.
    print("Loading config / building arg parser...", flush=True)
    parsed_args = parse_args_with_config()
    Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    print(
        f"Starting dynamic All-in-One training "
        f"(device={parsed_args.device}, "
        f"output_dir={parsed_args.output_dir})",
        flush=True,
    )
    run_training(parsed_args)

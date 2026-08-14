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
            action.choices = ["one_step"]
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
        "--diffusion_steps",
        default=1000,
        type=int,
        help="Official UDBM timeline length T with t in {0,...,T-1}",
    )
    parser.add_argument(
        "--bridge_noise_shared",
        default=20.0,
        type=float,
        help="Official UDBM lambda_b for u=0: 20 * tau * (1-tau)",
    )
    parser.add_argument(
        "--bridge_noise_terminal",
        default=1.0,
        type=float,
        help="Official UDBM lambda_r for u=0: 1 * tau^2 so beta_T=1",
    )
    parser.add_argument(
        "--bridge_version",
        default="udbm_exact_v1",
        type=str,
        help="Exact schedule identifier; rejects mismatched checkpoints",
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
    parser.add_argument(
        "--grad_clip",
        default=1.0,
        type=float,
        help="Max grad norm before optimizer step (0 disables clipping)",
    )
    parser.add_argument(
        "--lambda_ponder",
        default=1.0e-4,
        type=float,
        help="Weight of internal adaptive-depth ponder loss",
    )
    parser.add_argument(
        "--eval_use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use EMA weights for periodic / evaluate_only validation",
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
        num_sampling_steps=1,
        sampling_method="one_step",
        diagnostic_steps="1",
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
    if args.bridge_version != "udbm_exact_v1":
        raise ValueError(
            "Experiment B requires bridge_version=udbm_exact_v1"
        )
    if args.sampling_method != "one_step":
        raise ValueError(
            "Experiment B one-step inference requires "
            "sampling_method=one_step"
        )
    if args.conditioning_type != "state_and_degraded":
        raise ValueError(
            "All-in-One training requires "
            "conditioning_type=state_and_degraded"
        )
    if args.prediction_type != "conditional_x":
        raise ValueError("Experiment B requires prediction_type=conditional_x")
    if args.diffusion_steps < 2:
        raise ValueError("--diffusion_steps must be >= 2")
    if args.num_sampling_steps != 1:
        raise ValueError(
            "One-step inference requires num_sampling_steps=1"
        )
    if abs(args.bridge_noise_shared - 20.0) > 1e-8:
        print(
            "Warning: bridge_noise_shared != 20.0; "
            "official UDBM u=0 uses 20.0",
            flush=True,
        )
    if abs(args.bridge_noise_terminal - 1.0) > 1e-8:
        print(
            "Warning: bridge_noise_terminal != 1.0; "
            "official UDBM u=0 uses 1.0",
            flush=True,
        )


def validate_bridge_checkpoint(checkpoint_args, current_args):
    """Reject silent resume across incompatible bridge schedules."""
    required = {
        "bridge_type": current_args.bridge_type,
        "bridge_version": current_args.bridge_version,
        "sampling_method": current_args.sampling_method,
        "prediction_type": current_args.prediction_type,
        "conditioning_type": current_args.conditioning_type,
        "diffusion_steps": current_args.diffusion_steps,
        "bridge_noise_shared": current_args.bridge_noise_shared,
        "bridge_noise_terminal": current_args.bridge_noise_terminal,
    }
    for key, expected in required.items():
        actual = getattr(checkpoint_args, key, None)
        if actual is None:
            raise ValueError(
                f"Checkpoint missing {key!r}; refuse resume/init into "
                f"{current_args.bridge_version!r}"
            )
        if isinstance(expected, float):
            if abs(float(actual) - float(expected)) > 1e-8:
                raise ValueError(
                    f"Checkpoint {key}={actual!r} incompatible with "
                    f"current {expected!r}"
                )
        elif actual != expected:
            raise ValueError(
                f"Checkpoint {key}={actual!r} incompatible with "
                f"current {expected!r}"
            )

    checkpoint_parameterization = getattr(
        checkpoint_args,
        "output_parameterization",
        "direct_clean_v0",
    )
    if checkpoint_parameterization != current_args.output_parameterization:
        raise ValueError(
            "Checkpoint output parameterization mismatch: "
            f"{checkpoint_parameterization!r} vs "
            f"{current_args.output_parameterization!r}"
        )


@torch.no_grad()
def swap_ema_weights(model, enable):
    """Temporarily replace online weights with EMA (or restore backup)."""
    if not enable or model.ema_params is None:
        return None
    backup = [parameter.detach().clone() for parameter in model.parameters()]
    for parameter, ema_parameter in zip(model.parameters(), model.ema_params):
        parameter.data.copy_(ema_parameter)
    return backup


@torch.no_grad()
def restore_online_weights(model, backup):
    if backup is None:
        return
    for parameter, saved in zip(model.parameters(), backup):
        parameter.data.copy_(saved)


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
    """One-step UDBM diagnostics: LQ | x_T | x0_hat | GT."""
    noise_generator = torch.Generator(device=device).manual_seed(
        args.eval_seed + sample_index
    )
    padded, _ = model.pad_to_patch(degraded)
    initial_noise = torch.randn(
        padded.shape,
        device=device,
        dtype=degraded.dtype,
        generator=noise_generator,
    )
    reverse_generator = torch.Generator(device=device).manual_seed(
        args.eval_seed + sample_index + 100000
    )
    with autocast_context(device):
        restored, trajectory, trajectory_coeffs, trajectory_x0 = (
            model.restore(
                degraded,
                generator=reverse_generator,
                initial_noise=initial_noise,
                steps=1,
                method=args.sampling_method,
                return_trajectory=True,
            )
        )
    degraded_01 = to_01(degraded)[0]
    clean_01 = clean[0]
    restored_01 = to_01(restored)[0]
    height, width = degraded.shape[-2:]
    input_score = float(
        psnr_per_image(
            degraded_01.unsqueeze(0),
            clean_01.unsqueeze(0),
        )[0]
    )
    output_score = float(
        psnr_per_image(
            restored_01.unsqueeze(0),
            clean_01.unsqueeze(0),
        )[0]
    )
    terminal_b = float(model.bridge_noise_terminal)
    noisy_start = to_01(trajectory[0])[0]
    panels = [
        (
            f"{benchmark} LQ {height}x{width} | {input_score:.2f} dB",
            tensor_rgb(degraded_01),
        ),
        (
            f"x_T | tau=1.0 | beta_T={terminal_b:.2f}",
            tensor_rgb(noisy_start),
        ),
        (
            f"1-step x0 | {output_score:.2f} dB",
            tensor_rgb(restored_01),
        ),
        ("GT", tensor_rgb(clean_01)),
    ]
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
        t_value, b_value = coeffs
        state_01 = to_01(state)[0]
        score = float(
            psnr_per_image(
                state_01.unsqueeze(0),
                clean_01.unsqueeze(0),
            )[0]
        )
        label = (
            f"x_T t={t_value:.3f} b={b_value:.3f} | {score:.2f} dB"
            if reverse_index == 0
            else f"x0_hat t={t_value:.3f} | {score:.2f} dB"
        )
        trajectory_panels.append((label, tensor_rgb(state_01)))
    trajectory_panels.append(("GT", tensor_rgb(clean_01)))
    save_native_panels(
        trajectory_panels,
        output_dir / f"{stem}_trajectory.png",
        args.diagnostic_panel_size,
    )

    x0_01 = to_01(trajectory_x0[0])[0]
    x0_score = float(
        psnr_per_image(
            x0_01.unsqueeze(0),
            clean_01.unsqueeze(0),
        )[0]
    )
    save_native_panels(
        [
            (
                f"{benchmark} LQ {height}x{width} | {input_score:.2f} dB",
                tensor_rgb(degraded_01),
            ),
            (
                f"x0 from t=1 | {x0_score:.2f} dB",
                tensor_rgb(x0_01),
            ),
            ("GT", tensor_rgb(clean_01)),
        ],
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
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
        optimizer.step()
        model_without_ddp.update_ema()

        height, width = clean.shape[-2:]
        degradation_totals += torch.bincount(
            degradation_ids.detach().cpu(),
            minlength=len(TASK_ORDER),
        )
        logger.update(
            loss=loss_value,
            mse_loss=model_without_ddp.loss_terms["mse"].item(),
            l1_loss=model_without_ddp.loss_terms["l1"].item(),
            ponder_loss=(
                model_without_ddp.loss_terms["ponder_loss"].item()
            ),
            adaptive_mean_depth=(
                model_without_ddp
                .loss_terms["adaptive_mean_depth"]
                .item()
            ),
            difficulty_mean=(
                model_without_ddp
                .loss_terms["difficulty_mean"]
                .item()
            ),
            difficulty_std=(
                model_without_ddp
                .loss_terms["difficulty_std"]
                .item()
            ),
            reconstruction_mae=(
                model_without_ddp
                .loss_terms["reconstruction_mae"]
                .item()
            ),
            residual_target_abs=(
                model_without_ddp
                .loss_terms["residual_target_abs"]
                .item()
            ),
            residual_pred_abs=(
                model_without_ddp
                .loss_terms["residual_pred_abs"]
                .item()
            ),
            t=model_without_ddp.loss_terms["t"].item(),
            alpha=model_without_ddp.loss_terms["alpha"].item(),
            beta=model_without_ddp.loss_terms["beta"].item(),
            lr=optimizer.param_groups[0]["lr"],
            height=height,
            width=width,
        )
        if writer is not None and step % args.log_freq == 0:
            progress = int((step / len(loader) + epoch) * 1000)
            writer.add_scalar("train/loss", loss_value, progress)
            writer.add_scalar(
                "train/mse_loss",
                model_without_ddp.loss_terms["mse"].item(),
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
                "train/alpha",
                model_without_ddp.loss_terms["alpha"].item(),
                progress,
            )
            writer.add_scalar(
                "train/beta",
                model_without_ddp.loss_terms["beta"].item(),
                progress,
            )
            writer.add_scalar(
                "train/reconstruction_mae",
                model_without_ddp
                .loss_terms["reconstruction_mae"]
                .item(),
                progress,
            )
            writer.add_scalar(
                "train/residual_pred_abs",
                model_without_ddp
                .loss_terms["residual_pred_abs"]
                .item(),
                progress,
            )
            writer.add_scalar(
                "train/ponder_loss",
                model_without_ddp.loss_terms["ponder_loss"].item(),
                progress,
            )
            writer.add_scalar(
                "train/adaptive_mean_depth",
                model_without_ddp
                .loss_terms["adaptive_mean_depth"]
                .item(),
                progress,
            )
            writer.add_scalar(
                "train/difficulty_mean",
                model_without_ddp
                .loss_terms["difficulty_mean"]
                .item(),
                progress,
            )
            writer.add_scalar(
                "train/difficulty_std",
                model_without_ddp
                .loss_terms["difficulty_std"]
                .item(),
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
    args.output_parameterization = "observation_residual_v1"
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
        checkpoint_args = checkpoint.get("args")
        if checkpoint_args is None:
            raise ValueError(
                "Checkpoint missing args; refuse resume into udbm_exact_v1"
            )
        validate_bridge_checkpoint(checkpoint_args, args)
        if not getattr(checkpoint_args, "dynamic_resolution", False):
            raise ValueError(
                "Cannot resume a fixed-resolution checkpoint into "
                "dynamic All-in-One training"
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
            checkpoint_args = checkpoint.get("args")
            if checkpoint_args is not None:
                validate_bridge_checkpoint(checkpoint_args, args)
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
            ema_backup = swap_ema_weights(
                model_without_ddp,
                args.eval_use_ema,
            )
            try:
                evaluate_all(
                    model_without_ddp,
                    benchmark_loaders,
                    device,
                    args,
                    start_epoch,
                    writer,
                )
            finally:
                restore_online_weights(model_without_ddp, ema_backup)
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
                ema_backup = swap_ema_weights(
                    model_without_ddp,
                    args.eval_use_ema,
                )
                try:
                    evaluate_all(
                        model_without_ddp,
                        benchmark_loaders,
                        device,
                        args,
                        epoch,
                        writer,
                    )
                finally:
                    restore_online_weights(model_without_ddp, ema_backup)
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

"""Fixed-resolution JiT All-in-One restoration training.

Model task classes:
  0: dehazing
  1: deraining
  2: blind Gaussian denoising (sigma=15/25/50)

Training uses ``AIOTrainDataset`` from ``data/dataset_utils.py`` and sends its
256x256 crops directly into a fixed 256x256 JiT. Benchmark images are center
cropped (or padded when necessary) to 256x256 without interpolation.
"""

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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import util.lr_sched as lr_sched
import util.misc as misc
from data.dataset_utils import AIOTrainDataset, IRBenchmarks
from denoiser_all_in_one import AllInOneRestorationDenoiser
from train_restoration import get_args_parser as get_fixed_parser


TASK_ORDER = [
    "denoise_15",
    "denoise_25",
    "denoise_50",
    "derain",
    "dehaze",
]
BENCHMARKS = [
    "dehaze",
    "derain",
    "denoise_15",
    "denoise_25",
    "denoise_50",
]
MODEL_TASK_NAMES = ["dehaze", "derain", "denoise"]
DEGRADATION_TO_MODEL_TASK = torch.tensor([2, 2, 2, 1, 0])
BENCHMARK_TO_MODEL_TASK = {
    "dehaze": 0,
    "derain": 1,
    "denoise_15": 2,
    "denoise_25": 2,
    "denoise_50": 2,
}


def get_args_parser():
    parser = get_fixed_parser()
    parser.description = "Fixed-resolution JiT All-in-One restoration"
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
        "--init_checkpoint",
        default="",
        type=str,
        help="Optional All-in-One checkpoint used for weight initialization",
    )
    parser.add_argument(
        "--bsd68_root",
        default=None,
        type=str,
        help="Optional CBSD68 original_png directory",
    )
    parser.set_defaults(
        class_num=3,
        img_size=256,
        patch_size=256,
        output_dir="./output/jit_all_in_one",
        num_sampling_steps=1,
    )
    return parser


def validate_task_order(args):
    if list(args.de_type) != TASK_ORDER:
        raise ValueError(
            "--de_type must keep this order: " + " ".join(TASK_ORDER)
        )
    if args.class_num != len(MODEL_TASK_NAMES):
        raise ValueError(f"--class_num must be {len(MODEL_TASK_NAMES)}")
    if args.img_size != args.patch_size:
        raise ValueError(
            "Fixed All-in-One training requires img_size == patch_size"
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


def center_crop_or_pad(tensor, image_size):
    """Match the fixed model size without resampling image pixels."""
    height, width = tensor.shape[-2:]
    pad_h = max(image_size - height, 0)
    pad_w = max(image_size - width, 0)
    if pad_h or pad_w:
        mode = (
            "reflect"
            if height > pad_h and width > pad_w
            else "replicate"
        )
        tensor = F.pad(
            tensor,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
            mode=mode,
        )
    height, width = tensor.shape[-2:]
    top = (height - image_size) // 2
    left = (width - image_size) // 2
    return tensor[
        ...,
        top : top + image_size,
        left : left + image_size,
    ]


def replace_denoising_inputs(degraded, clean, degradation_ids):
    """Add the requested Gaussian noise to the clean training crops."""
    denoise_mask = degradation_ids < 3
    if not denoise_mask.any():
        return degraded
    degraded = degraded.clone()
    sigmas = torch.tensor(
        [15.0, 25.0, 50.0],
        device=clean.device,
        dtype=clean.dtype,
    )
    selected_sigmas = sigmas[degradation_ids[denoise_mask]].view(
        -1,
        1,
        1,
        1,
    )
    degraded[denoise_mask] = (
        clean[denoise_mask]
        + torch.randn_like(clean[denoise_mask]) * selected_sigmas / 255.0
    ).clamp(0, 1)
    return degraded


class FixedAIOTrainDataset(AIOTrainDataset):
    """Reuse AIO data discovery while supplying its paired crop transform."""

    def _crop_patch(self, image_1, image_2):
        height = min(image_1.shape[0], image_2.shape[0])
        width = min(image_1.shape[1], image_2.shape[1])
        size = self.args.patch_size
        image_1 = image_1[:height, :width]
        image_2 = image_2[:height, :width]
        pad_h = max(size - height, 0)
        pad_w = max(size - width, 0)
        if pad_h or pad_w:
            top_pad = pad_h // 2
            bottom_pad = pad_h - top_pad
            left_pad = pad_w // 2
            right_pad = pad_w - left_pad
            mode = (
                "reflect"
                if height > max(top_pad, bottom_pad)
                and width > max(left_pad, right_pad)
                else "edge"
            )
            padding = (
                (top_pad, bottom_pad),
                (left_pad, right_pad),
                (0, 0),
            )
            image_1 = np.pad(image_1, padding, mode=mode)
            image_2 = np.pad(image_2, padding, mode=mode)
            height, width = image_1.shape[:2]
        top = random.randint(0, height - size)
        left = random.randint(0, width - size)
        return (
            image_1[top : top + size, left : left + size],
            image_2[top : top + size, left : left + size],
        )


def make_balanced_sampler(dataset, samples_per_epoch):
    degradation_ids = [int(sample["de_type"]) for sample in dataset.lr]
    counts = np.bincount(degradation_ids, minlength=len(TASK_ORDER))
    if np.any(counts == 0):
        raise RuntimeError(f"Missing All-in-One task samples: {counts}")
    target_probabilities = np.array(
        [1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 3],
        dtype=np.float64,
    )
    weights = torch.tensor(
        [
            target_probabilities[degradation_id]
            / counts[degradation_id]
            for degradation_id in degradation_ids
        ],
        dtype=torch.double,
    )
    print("Raw All-in-One task counts:", dict(zip(TASK_ORDER, counts.tolist())))
    return WeightedRandomSampler(
        weights,
        num_samples=samples_per_epoch,
        replacement=True,
    )


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
def evaluate_benchmark(
    model,
    loader,
    benchmark,
    device,
    args,
    epoch,
    output_root,
):
    model.eval()
    scores = []
    maes = []
    benchmark_dir = Path(output_root) / benchmark
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    for index, (meta, degraded, clean) in enumerate(loader):
        degraded = center_crop_or_pad(
            degraded.to(device, non_blocking=True),
            args.img_size,
        )
        clean = center_crop_or_pad(
            clean.to(device, non_blocking=True),
            args.img_size,
        )
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
        task_ids = torch.full(
            (degraded.size(0),),
            BENCHMARK_TO_MODEL_TASK[benchmark],
            dtype=torch.long,
            device=device,
        )
        degraded = to_norm(degraded)
        generator = torch.Generator(device=device).manual_seed(
            args.eval_seed + index
        )
        with autocast_context(device):
            restored = model.restore(
                degraded,
                task_ids,
                generator=generator,
                steps=args.num_sampling_steps,
                method=args.sampling_method,
            )
        restored = to_01(restored)
        scores.extend(psnr_per_image(restored, clean).cpu().tolist())
        maes.append(float((restored - clean).abs().mean()))
        if index < args.num_eval_images:
            stem = Path(meta[0][0]).stem
            panel = torch.cat([to_01(degraded), restored, clean], dim=0)
            save_image(
                panel,
                benchmark_dir / f"{stem}_epoch{epoch:04d}.png",
                nrow=3,
            )
    result = {
        "psnr": float(np.mean(scores)),
        "mae": float(np.mean(maes)),
        "images": len(scores),
    }
    print(
        f"[{benchmark}] epoch={epoch} images={result['images']} "
        f"PSNR={result['psnr']:.3f} MAE={result['mae']:.5f}"
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


def train_one_epoch(model, loader, optimizer, device, epoch, args, writer):
    model.train(True)
    logger = misc.MetricLogger(delimiter="  ")
    logger.add_meter(
        "lr",
        misc.SmoothedValue(window_size=1, fmt="{value:.6f}"),
    )
    task_totals = torch.zeros(
        len(MODEL_TASK_NAMES),
        dtype=torch.long,
    )
    iterator = logger.log_every(
        loader,
        args.log_freq,
        f"All-in-One epoch: [{epoch}]",
    )
    for step, (meta, degraded, clean) in enumerate(iterator):
        lr_sched.adjust_learning_rate(
            optimizer,
            step / len(loader) + epoch,
            args,
        )
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        if degraded.shape[-2:] != (args.img_size, args.img_size):
            raise RuntimeError(
                f"Training crop is {degraded.shape[-2:]}, expected "
                f"{args.img_size}x{args.img_size}"
            )
        degradation_ids = meta[1].to(device, non_blocking=True)
        degraded = replace_denoising_inputs(
            degraded,
            clean,
            degradation_ids,
        )
        degraded = to_norm(degraded)
        clean = to_norm(clean)
        task_ids = DEGRADATION_TO_MODEL_TASK.to(device)[degradation_ids]
        with autocast_context(device):
            loss = model(clean, degraded, task_ids)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss: {loss_value}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.update_ema()

        task_totals += torch.bincount(
            task_ids.detach().cpu(),
            minlength=len(MODEL_TASK_NAMES),
        )
        logger.update(
            loss=loss_value,
            flow_loss=model.loss_terms["flow"].item(),
            l1_loss=model.loss_terms["l1"].item(),
            lr=optimizer.param_groups[0]["lr"],
        )
        if writer is not None and step % args.log_freq == 0:
            progress = int((step / len(loader) + epoch) * 1000)
            writer.add_scalar("train/loss", loss_value, progress)
            writer.add_scalar(
                "train/flow_loss",
                model.loss_terms["flow"].item(),
                progress,
            )
            writer.add_scalar(
                "train/l1_loss",
                model.loss_terms["l1"].item(),
                progress,
            )
    logger.synchronize_between_processes()
    print(
        "Sampled task counts:",
        dict(zip(MODEL_TASK_NAMES, task_totals.tolist())),
    )
    print("Averaged stats:", logger)


def checkpoint_path(path):
    if path and os.path.isdir(path):
        return os.path.join(path, "checkpoint-last.pth")
    return path


def run_training(args):
    validate_task_order(args)
    if args.timestamp_output and not args.resume and not args.evaluate_only:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(args.output_dir, timestamp)
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.output_dir)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    train_dataset = FixedAIOTrainDataset(args)
    train_sampler = make_balanced_sampler(
        train_dataset,
        args.samples_per_epoch,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    benchmark_loaders = build_benchmark_loaders(args)
    model = AllInOneRestorationDenoiser(args).to(device)
    optimizer = torch.optim.AdamW(
        misc.add_weight_decay(model, args.weight_decay),
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    start_epoch = 0
    resume_path = checkpoint_path(args.resume)
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
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
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {resume_path}, start_epoch={start_epoch}")
    else:
        if args.init_checkpoint:
            checkpoint = torch.load(
                checkpoint_path(args.init_checkpoint),
                map_location="cpu",
            )
            model.load_state_dict(checkpoint.get("model", checkpoint))
            print(f"Initialized from {args.init_checkpoint}")
        model.ema_params = [
            parameter.detach().clone()
            for parameter in model.parameters()
        ]

    if args.evaluate_only:
        evaluate_all(
            model,
            benchmark_loaders,
            device,
            args,
            start_epoch,
            writer,
        )
        return

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args,
            writer,
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
            evaluate_all(
                model,
                benchmark_loaders,
                device,
                args,
                epoch,
                writer,
            )
        writer.flush()
    elapsed = str(
        datetime.timedelta(seconds=int(time.time() - started))
    )
    print("All-in-One training time:", elapsed)


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
    parsed_args = parse_args_with_config()
    Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    run_training(parsed_args)

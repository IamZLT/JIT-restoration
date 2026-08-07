"""Native-resolution testing for dynamic JiT restoration checkpoints."""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from data.dataset_dynamic import DynamicRainDataset, DynamicSOTSDataset
from denoiser_rest_dynamic import DynamicRestorationDenoiser


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Dynamic JiT native-resolution restoration test"
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument(
        "--task",
        default="derain",
        choices=["dehaze", "derain"],
    )
    parser.add_argument("--data_file_dir", default="./datasets", type=str)
    parser.add_argument("--sots_root", default=None, type=str)
    parser.add_argument("--rain_test_root", default=None, type=str)
    parser.add_argument(
        "--output_dir",
        default="./output/dynamic_test",
        type=str,
    )
    parser.add_argument(
        "--steps",
        default="1,4,20",
        type=str,
        help="Comma-separated sampling step counts",
    )
    parser.add_argument(
        "--method",
        default="euler",
        choices=["euler", "heun"],
    )
    parser.add_argument("--generation-strength", default=None, type=float)
    parser.add_argument("--num_images", default=10, type=int)
    parser.add_argument(
        "--indices",
        default="",
        type=str,
        help="Optional comma-separated dataset indices",
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument(
        "--use-ema",
        dest="use_ema",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def to_norm(tensor):
    return tensor * 2.0 - 1.0


def to_01(tensor):
    return (tensor.clamp(-1, 1) + 1.0) * 0.5


def psnr(prediction, target):
    mse = (prediction - target).pow(2).mean().clamp_min(1e-10)
    return float(-10.0 * torch.log10(mse))


def autocast_context(device):
    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def resolve_checkpoint(path):
    if os.path.isdir(path):
        return os.path.join(path, "checkpoint-last.pth")
    return path


def load_model(args):
    path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(path, map_location="cpu")
    train_args = checkpoint["args"]
    model = DynamicRestorationDenoiser(train_args)
    state = dict(checkpoint["model"])
    weight_name = "online"
    if args.use_ema and checkpoint.get("model_ema"):
        state.update(checkpoint["model_ema"])
        weight_name = "EMA"
    model.load_compatible_state_dict(state)
    if args.generation_strength is not None:
        if not 0.0 <= args.generation_strength <= 1.0:
            raise ValueError("--generation-strength must be in [0, 1]")
        model.generation_strength = args.generation_strength
    model.to(args.device).eval()
    print(
        f"Loaded epoch {checkpoint.get('epoch', 'unknown')} "
        f"using {weight_name} weights"
    )
    return model, checkpoint.get("epoch", "unknown")


def build_dataset(args):
    if args.task == "dehaze":
        return DynamicSOTSDataset(args)
    return DynamicRainDataset(args, split="test")


def selected_indices(args, dataset):
    if args.indices:
        indices = [
            int(value)
            for value in args.indices.split(",")
            if value.strip()
        ]
    else:
        indices = list(range(min(args.num_images, len(dataset))))
    for index in indices:
        if not 0 <= index < len(dataset):
            raise IndexError(
                f"Dataset index {index} outside [0, {len(dataset)})"
            )
    return indices[: args.num_images]


def main(args):
    steps = sorted(
        {
            int(value)
            for value in args.steps.split(",")
            if value.strip()
        }
    )
    if not steps or steps[0] < 1:
        raise ValueError("--steps must contain positive integers")
    device = torch.device(args.device)
    model, epoch = load_model(args)
    dataset = build_dataset(args)
    indices = selected_indices(args, dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for index in indices:
        meta, degraded_cpu, clean_cpu = dataset[index]
        degraded = to_norm(degraded_cpu.unsqueeze(0).to(device))
        clean = clean_cpu.to(device)
        stem = Path(meta[0]).stem
        height, width = degraded.shape[-2:]
        input_score = psnr(degraded_cpu.to(device), clean)
        save_image(degraded_cpu, output_dir / f"{stem}_input.png")
        save_image(clean_cpu, output_dir / f"{stem}_gt.png")

        for step_count in steps:
            generator = torch.Generator(device=device).manual_seed(
                args.seed + index
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                baseline_memory = torch.cuda.memory_allocated(device)
            else:
                baseline_memory = 0
            started = time.perf_counter()
            with autocast_context(device):
                prediction = model.restore(
                    degraded,
                    generator=generator,
                    steps=step_count,
                    method=args.method,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory = torch.cuda.max_memory_allocated(device)
            else:
                peak_memory = 0
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            prediction = to_01(prediction)[0]
            score = psnr(prediction, clean)
            save_image(
                prediction,
                output_dir / f"{stem}_step{step_count}.png",
            )
            rows.append(
                {
                    "image": stem,
                    "index": index,
                    "epoch": epoch,
                    "height": height,
                    "width": width,
                    "padded_height": (
                        (height + model.patch_size - 1)
                        // model.patch_size
                        * model.patch_size
                    ),
                    "padded_width": (
                        (width + model.patch_size - 1)
                        // model.patch_size
                        * model.patch_size
                    ),
                    "steps": step_count,
                    "method": args.method,
                    "input_psnr_db": input_score,
                    "psnr_db": score,
                    "psnr_gain_db": score - input_score,
                    "mae": float((prediction - clean).abs().mean()),
                    "latency_ms": elapsed_ms,
                    "peak_memory_mib": peak_memory / (1024**2),
                    "inference_peak_extra_mib": (
                        peak_memory - baseline_memory
                    )
                    / (1024**2),
                }
            )
        print(
            f"[{stem}] native={height}x{width}, "
            f"{steps[-1]}-step PSNR={rows[-1]['psnr_db']:.2f} dB"
        )

    fieldnames = [
        "image",
        "index",
        "epoch",
        "height",
        "width",
        "padded_height",
        "padded_width",
        "steps",
        "method",
        "input_psnr_db",
        "psnr_db",
        "psnr_gain_db",
        "mae",
        "latency_ms",
        "peak_memory_mib",
        "inference_peak_extra_mib",
    ]
    with open(
        output_dir / "metrics.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for step_count in steps:
        selected = [
            row["psnr_db"]
            for row in rows
            if row["steps"] == step_count
        ]
        print(
            f"step={step_count}: mean PSNR={np.mean(selected):.3f} dB"
        )
    print(f"Saved native-resolution outputs to {output_dir}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())

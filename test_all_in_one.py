"""Evaluate a fixed-resolution All-in-One JiT checkpoint on five benchmarks."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset_utils import IRBenchmarks
from denoiser_all_in_one import AllInOneRestorationDenoiser
from train_all_in_one import (
    BENCHMARKS,
    TASK_ORDER,
    evaluate_benchmark,
)


def get_args_parser():
    parser = argparse.ArgumentParser("Test fixed JiT All-in-One restoration")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data_file_dir", default="./datasets", type=str)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=BENCHMARKS,
        choices=BENCHMARKS,
    )
    parser.add_argument("--output_dir", default="./output/aio_test", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--steps", default=15, type=int)
    parser.add_argument(
        "--method",
        default="resshift",
        choices=["resshift"],
    )
    parser.add_argument("--num_eval_images", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--eval_seed", default=1234, type=int)
    parser.add_argument(
        "--use-ema",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def resolve_checkpoint(path):
    if os.path.isdir(path):
        return os.path.join(path, "checkpoint-last.pth")
    return path


def main(args):
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint["args"]
    checkpoint_bridge = getattr(
        train_args,
        "bridge_type",
        "noise_to_clean",
    )
    if checkpoint_bridge != "resshift":
        raise ValueError(
            "This evaluator requires a ResShift-bridge checkpoint; "
            "old noise-to-clean checkpoints are not sampler-compatible."
        )
    checkpoint_conditioning = getattr(
        train_args,
        "conditioning_type",
        "degraded_and_task",
    )
    if checkpoint_conditioning != "state_only":
        raise ValueError(
            "This evaluator requires a state-only checkpoint without "
            "degraded-image or task-label network conditioning."
        )
    if list(train_args.de_type) != TASK_ORDER:
        raise ValueError(
            f"Checkpoint task order {train_args.de_type} does not match "
            f"{TASK_ORDER}"
        )
    model = AllInOneRestorationDenoiser(train_args)
    state = dict(checkpoint["model"])
    weight_name = "online"
    if args.use_ema and checkpoint.get("model_ema"):
        state.update(checkpoint["model_ema"])
        weight_name = "EMA"
    model.load_state_dict(state)
    device = torch.device(args.device)
    model.to(device).eval()

    eval_args = copy.copy(train_args)
    eval_args.data_file_dir = args.data_file_dir
    eval_args.output_dir = args.output_dir
    eval_args.device = args.device
    eval_args.num_sampling_steps = args.steps
    eval_args.sampling_method = args.method
    eval_args.num_eval_images = args.num_eval_images
    eval_args.num_workers = args.num_workers
    eval_args.eval_seed = args.eval_seed
    eval_args.pin_mem = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    epoch = checkpoint.get("epoch", -1)
    for benchmark in args.benchmarks:
        benchmark_args = copy.copy(eval_args)
        benchmark_args.benchmarks = benchmark
        dataset = IRBenchmarks(benchmark_args)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        results[benchmark] = evaluate_benchmark(
            model,
            loader,
            benchmark,
            device,
            eval_args,
            epoch,
            output_dir,
        )
    summary = {
        "checkpoint": checkpoint_path,
        "epoch": epoch,
        "weights": weight_name,
        "steps": args.steps,
        "method": args.method,
        "bridge_type": model.bridge_type,
        "conditioning_type": model.conditioning_type,
        "resshift_kappa": model.resshift_kappa,
        "evaluation_crop": f"{train_args.img_size}x{train_args.img_size}",
        "results": results,
    }
    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved All-in-One results to {output_dir}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())

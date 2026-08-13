"""Native-resolution testing for dynamic All-in-One JiT checkpoints."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset_utils import IRBenchmarks
from denoiser_all_in_one_dynamic import DynamicAllInOneRestorationDenoiser
from train_all_in_one_dynamic import (
    BENCHMARKS,
    TASK_ORDER,
    evaluate_benchmark,
)


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Test dynamic JiT All-in-One restoration at native resolution"
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data_file_dir", default="./datasets", type=str)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=BENCHMARKS,
        choices=BENCHMARKS,
    )
    parser.add_argument(
        "--output_dir",
        default="./output/aio_dynamic_expB_test",
        type=str,
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--steps", default=1, type=int)
    parser.add_argument(
        "--method",
        default="one_step",
        choices=["one_step"],
    )
    parser.add_argument("--num_eval_images", default=4, type=int)
    parser.add_argument("--diagnostic_images", default=2, type=int)
    parser.add_argument(
        "--diagnostic_steps",
        default="1",
        type=str,
    )
    parser.add_argument("--diagnostic_panel_size", default=256, type=int)
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
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    train_args = checkpoint["args"]
    if not getattr(train_args, "dynamic_resolution", False):
        raise ValueError(
            "This evaluator expects a dynamic-resolution All-in-One "
            "checkpoint (train_all_in_one_dynamic.py)."
        )
    checkpoint_bridge = getattr(
        train_args,
        "bridge_type",
        "noise_to_clean",
    )
    if checkpoint_bridge != "global_udbm_bridge":
        raise ValueError(
            "This evaluator requires an Experiment-B "
            "global_udbm_bridge checkpoint"
        )
    if args.steps != 1 or args.method != "one_step":
        raise ValueError(
            "UDBM one-step evaluation requires --steps 1 --method one_step"
        )
    checkpoint_conditioning = getattr(
        train_args,
        "conditioning_type",
        "degraded_and_task",
    )
    if checkpoint_conditioning != "state_and_degraded":
        raise ValueError(
            "This evaluator requires blind state-and-degraded conditioning"
        )
    if list(train_args.de_type) != TASK_ORDER:
        raise ValueError(
            f"Checkpoint task order {train_args.de_type} does not match "
            f"{TASK_ORDER}"
        )
    model = DynamicAllInOneRestorationDenoiser(train_args)
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
    eval_args.num_sampling_steps = 1
    eval_args.sampling_method = "one_step"
    eval_args.num_eval_images = args.num_eval_images
    eval_args.diagnostic_images = args.diagnostic_images
    eval_args.diagnostic_steps = "1"
    eval_args.diagnostic_panel_size = args.diagnostic_panel_size
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
        "steps": 1,
        "method": "one_step",
        "bridge_type": model.bridge_type,
        "conditioning_type": model.conditioning_type,
        "diffusion_steps": model.diffusion_steps,
        "bridge_noise_shared": model.bridge_noise_shared,
        "bridge_noise_terminal": model.bridge_noise_terminal,
        "evaluation": "native_resolution_one_step",
        "results": results,
    }
    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved dynamic All-in-One Exp-B results to {output_dir}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())

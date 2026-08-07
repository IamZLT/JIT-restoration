"""Diagnose a trained JiT restoration checkpoint on dehazing or deraining.

Outputs for each selected image:
  1. ``*_steps.png``: final restorations produced with different ODE step counts.
  2. ``*_trajectory.png``: intermediate states of the longest sampling run.
  3. ``*_pixel_analysis.png``: GT degradation, model correction, and error maps.
  4. ``metrics.csv``: PSNR/MAE and severe-vs-normal region statistics.

The degradation map uses paired GT and is therefore an analysis-only oracle.
The correction map uses only model output and input; agreement between these
two maps indicates whether JiT applies larger changes to more degraded pixels.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data.dataset_utils import RainDerainDataset, SOTSDehazeDataset
from denoiser_rest import RestorationDenoiser


def get_args_parser():
    parser = argparse.ArgumentParser("JiT restoration diagnostics")
    parser.add_argument(
        "--checkpoint",
        default="./output/jit_reside_rectified/checkpoint-last.pth",
        type=str,
    )
    parser.add_argument("--task", default="dehaze", choices=["dehaze", "derain"])
    parser.add_argument("--data_file_dir", default="./datasets", type=str)
    parser.add_argument("--sots_root", default=None, type=str)
    parser.add_argument("--rain_test_root", default=None, type=str)
    parser.add_argument("--output_dir", default="./output/jit_reside_rectified/test_diagnostics", type=str)
    parser.add_argument("--steps", default="1,2,4,8", type=str,
                        help="Comma-separated ODE step counts")
    parser.add_argument("--method", default="euler", choices=["euler", "heun"])
    parser.add_argument("--generation-strength", default=None, type=float,
                        help="Override checkpoint img2img strength in [0,1]")
    parser.add_argument("--num_images", default=3, type=int)
    parser.add_argument("--indices", default="", type=str,
                        help="Optional comma-separated test indices; otherwise auto-select")
    parser.add_argument("--panel_size", default=256, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=1234, type=int,
                        help="Bridge-noise seed; the same noise is used for all step counts")
    parser.add_argument(
        "--use-ema",
        dest="use_ema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use EMA weights. Keep disabled to match train_restoration.py validation.",
    )
    parser.add_argument("--checkpoint_retries", default=5, type=int)
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure FLOPs, latency, parameter size, and CUDA peak memory",
    )
    return parser


def to_norm(x):
    return x * 2.0 - 1.0


def to_01(x):
    return (x.clamp(-1, 1) + 1.0) * 0.5


def load_checkpoint_when_stable(path, retries):
    """Retry when training happens to be replacing the checkpoint."""
    error = None
    for attempt in range(retries):
        try:
            stat_before = os.stat(path)
            time.sleep(0.5)
            stat_after = os.stat(path)
            if (stat_before.st_size, stat_before.st_mtime_ns) != (
                stat_after.st_size,
                stat_after.st_mtime_ns,
            ):
                raise RuntimeError("checkpoint is currently being written")
            return torch.load(path, map_location="cpu")
        except (OSError, RuntimeError, EOFError) as exc:
            error = exc
            if attempt + 1 < retries:
                print(f"Checkpoint unavailable ({exc}); retrying...")
                time.sleep(2.0)
    raise RuntimeError(f"Could not load a stable checkpoint: {path}") from error


def load_model(args):
    checkpoint = load_checkpoint_when_stable(args.checkpoint, args.checkpoint_retries)
    train_args = checkpoint["args"]
    model = RestorationDenoiser(train_args)
    model.load_state_dict(checkpoint["model"])

    if args.use_ema and checkpoint.get("model_ema"):
        state = model.state_dict()
        state.update(checkpoint["model_ema"])
        model.load_state_dict(state)
        weight_name = "EMA"
    else:
        weight_name = "online"

    model.to(args.device).eval()
    if args.generation_strength is not None:
        if not 0.0 <= args.generation_strength <= 1.0:
            raise ValueError("--generation-strength must be in [0,1]")
        model.generation_strength = args.generation_strength
    epoch = checkpoint.get("epoch", "unknown")
    print(f"Loaded epoch {epoch} using {weight_name} weights")
    return model, train_args, epoch


@torch.no_grad()
def restore_trajectory(model, degraded, steps, method, initial_noise, record=False):
    t_start = 1.0 - model.generation_strength
    if model.generation_strength == 0.0:
        return degraded.clone(), [degraded.clone()] if record else None
    z = t_start * degraded + model.generation_strength * model.noise_scale * initial_noise
    trajectory = [z.clone()] if record else None
    timesteps = torch.linspace(t_start, 1.0, steps + 1, device=z.device)
    timesteps = timesteps.view(-1, 1, 1, 1, 1).expand(-1, z.size(0), -1, -1, -1)

    for i in range(steps):
        t, t_next = timesteps[i], timesteps[i + 1]
        # Match training-time restore(): use Euler at the final endpoint.
        if method == "heun" and i < steps - 1:
            z = model._heun_step(z, degraded, t, t_next)
        else:
            z = model._euler_step(z, degraded, t, t_next)
        if record:
            trajectory.append(z.clone())
    return z, trajectory


def model_statistics(model):
    parameters = list(model.parameters())
    count = sum(parameter.numel() for parameter in parameters)
    size_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    return {
        "parameters": count,
        "parameters_million": count / 1e6,
        "model_parameter_mib": size_bytes / (1024 ** 2),
    }


@torch.no_grad()
def profile_single_nfe(model, state, condition):
    """Profile one velocity evaluation; total sampling FLOPs scale with NFE."""
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if state.is_cuda:
        activities.append(ProfilerActivity.CUDA)
    t = torch.full(
        (state.size(0), 1, 1, 1),
        1.0 - model.generation_strength,
        device=state.device,
        dtype=state.dtype,
    )
    # Compile/warm up outside the profiler so tracing work is not counted as
    # inference FLOPs or latency.
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=state.is_cuda):
        model._forward_sample(state, condition, t)
    if state.is_cuda:
        torch.cuda.synchronize(state.device)
    with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=state.is_cuda):
            model._forward_sample(state, condition, t)
        if state.is_cuda:
            torch.cuda.synchronize(state.device)
    return int(sum(event.flops or 0 for event in prof.key_averages()))


def nfe_for_steps(steps, method):
    return steps if method == "euler" else 2 * steps - 1


def psnr(pred, target):
    mse = (pred - target).pow(2).mean().clamp_min(1e-10)
    return float(-10.0 * torch.log10(mse))


def tensor_rgb(image):
    array = image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (array * 255.0 + 0.5).astype(np.uint8)


def heatmap(values, value_max=None):
    values = values.detach().float().cpu().numpy()
    if value_max is None:
        value_max = max(float(np.quantile(values, 0.99)), 1e-6)
    x = np.clip(values / value_max, 0.0, 1.0)
    # Compact blue-cyan-yellow-red map without an extra plotting dependency.
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0 + 0.5).astype(np.uint8)


def save_panels(panels, path, panel_size):
    """Save one labeled row of RGB uint8 panels."""
    label_height = 28
    canvas = Image.new("RGB", (panel_size * len(panels), panel_size + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, array) in enumerate(panels):
        panel = Image.fromarray(array).resize((panel_size, panel_size), Image.Resampling.BILINEAR)
        canvas.paste(panel, (i * panel_size, label_height))
        draw.text((i * panel_size + 5, 7), label, fill="black")
    canvas.save(path)


def select_indices(dataset, count):
    """Select images containing both relatively mild and severe local haze."""
    ranked = []
    print(f"Ranking {len(dataset)} test pairs by spatial degradation contrast...")
    for idx in range(len(dataset)):
        _, degraded, clean = dataset[idx]
        severity = (degraded - clean).abs().mean(dim=0)
        contrast = float(torch.quantile(severity, 0.9) - torch.quantile(severity, 0.1))
        ranked.append((contrast, idx))
    return [idx for _, idx in sorted(ranked, reverse=True)[:count]]


def region_metrics(degraded, restored, clean):
    severity = (degraded - clean).abs().mean(dim=0)
    correction = (restored - degraded).abs().mean(dim=0)
    error_before = (degraded - clean).abs().mean(dim=0)
    error_after = (restored - clean).abs().mean(dim=0)

    low = torch.quantile(severity, 0.2)
    high = torch.quantile(severity, 0.8)
    normal = severity <= low
    severe = severity >= high

    severity_flat = severity.flatten()
    correction_flat = correction.flatten()
    correlation = torch.corrcoef(torch.stack([severity_flat, correction_flat]))[0, 1]

    metrics = {
        "correction_severe": float(correction[severe].mean()),
        "correction_normal": float(correction[normal].mean()),
        "error_reduction_severe": float((error_before[severe] - error_after[severe]).mean()),
        "error_reduction_normal": float((error_before[normal] - error_after[normal]).mean()),
        "severity_correction_corr": float(torch.nan_to_num(correlation)),
    }
    metrics["correction_ratio_severe_normal"] = metrics["correction_severe"] / max(
        metrics["correction_normal"], 1e-8
    )
    maps = severity, correction, error_before, error_after
    return metrics, maps


def main(args):
    steps = sorted({int(value) for value in args.steps.split(",") if value.strip()})
    if not steps or steps[0] < 1:
        raise ValueError("--steps must contain positive integers")

    model, train_args, epoch = load_model(args)
    # Dataset crop must match the fixed-size JiT positional embedding.
    if args.task == "dehaze":
        data_args = argparse.Namespace(
            data_file_dir=args.data_file_dir,
            sots_root=args.sots_root,
            patch_size=train_args.img_size,
        )
        dataset = SOTSDehazeDataset(data_args, split="test")
    else:
        data_args = argparse.Namespace(
            data_file_dir=args.data_file_dir,
            rain_test_root=args.rain_test_root,
            patch_size=train_args.img_size,
        )
        dataset = RainDerainDataset(data_args, split="test")

    if args.indices:
        indices = [int(value) for value in args.indices.split(",") if value.strip()]
    else:
        indices = select_indices(dataset, args.num_images)
    indices = indices[: args.num_images]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    profile_summary = model_statistics(model)
    flops_per_nfe = None

    for idx in indices:
        meta, degraded_cpu, clean_cpu = dataset[idx]
        degraded = to_norm(degraded_cpu.unsqueeze(0).to(args.device))
        clean = clean_cpu.to(args.device)
        generator = torch.Generator(device=args.device).manual_seed(args.seed + idx)
        initial_noise = torch.randn(
            degraded.shape,
            device=degraded.device,
            dtype=degraded.dtype,
            generator=generator,
        )

        t_start = 1.0 - model.generation_strength
        initial_state = (
            t_start * degraded
            + model.generation_strength * model.noise_scale * initial_noise
        )
        if args.profile and flops_per_nfe is None:
            measured_flops = profile_single_nfe(model, initial_state, degraded)
            flops_per_nfe = measured_flops if measured_flops > 0 else None

        step_outputs = {}
        run_metrics = {}
        for count in steps:
            if degraded.is_cuda:
                torch.cuda.synchronize(degraded.device)
                torch.cuda.reset_peak_memory_stats(degraded.device)
                baseline_memory = torch.cuda.memory_allocated(degraded.device)
            started = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                final_state, _ = restore_trajectory(
                    model, degraded, count, args.method, initial_noise, record=False
                )
            if degraded.is_cuda:
                torch.cuda.synchronize(degraded.device)
                peak_memory = torch.cuda.max_memory_allocated(degraded.device)
            else:
                peak_memory = baseline_memory = 0
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            nfe = nfe_for_steps(count, args.method)
            step_outputs[count] = to_01(final_state)[0]
            run_metrics[count] = {
                "nfe": nfe,
                "latency_ms": elapsed_ms,
                "peak_memory_mib": peak_memory / (1024 ** 2),
                "inference_peak_extra_mib": (peak_memory - baseline_memory) / (1024 ** 2),
                "estimated_gflops": (
                    flops_per_nfe * nfe / 1e9 if flops_per_nfe is not None else None
                ),
            }

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, trajectory = restore_trajectory(
                model, degraded, steps[-1], args.method, initial_noise, record=True
            )
        longest_trajectory = [to_01(value)[0] for value in trajectory]

        degraded_01 = degraded_cpu.to(args.device)
        stem = Path(meta[0]).stem
        input_psnr = psnr(degraded_01, clean)
        noisy_start = longest_trajectory[0]

        step_panels = [
            (f"Original LQ | {input_psnr:.2f} dB", tensor_rgb(degraded_01)),
            (
                f"Noisy start | strength={model.generation_strength:.2f}",
                tensor_rgb(noisy_start),
            ),
        ]
        for count in steps:
            output = step_outputs[count]
            score = psnr(output, clean)
            step_panels.append((f"{count} step | {score:.2f} dB", tensor_rgb(output)))
            rows.append(
                {
                    "image": stem,
                    "index": idx,
                    "epoch": epoch,
                    "steps": count,
                    "input_psnr_db": input_psnr,
                    "psnr_db": score,
                    "psnr_gain_db": score - input_psnr,
                    "mae": float((output - clean).abs().mean()),
                    **run_metrics[count],
                }
            )
        step_panels.append(("GT", tensor_rgb(clean)))
        save_panels(step_panels, output_dir / f"{stem}_steps.png", args.panel_size)

        trajectory_panels = []
        for i, value in enumerate(longest_trajectory):
            actual_t = t_start + model.generation_strength * i / steps[-1]
            label = f"start t={actual_t:.2f}" if i == 0 else f"t={actual_t:.2f}"
            trajectory_panels.append((label, tensor_rgb(value)))
        trajectory_panels.append(("GT", tensor_rgb(clean)))
        save_panels(trajectory_panels, output_dir / f"{stem}_trajectory.png", args.panel_size)

        restored = step_outputs[steps[-1]]
        diagnostics, maps = region_metrics(degraded_01, restored, clean)
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
        save_panels(analysis_panels, output_dir / f"{stem}_pixel_analysis.png", args.panel_size)

        for row in rows:
            if row["image"] == stem and row["steps"] == steps[-1]:
                row.update(diagnostics)
        print(
            f"[{stem}] PSNR {rows[-1]['psnr_db']:.2f} dB | "
            f"correction severe/normal {diagnostics['correction_ratio_severe_normal']:.2f} | "
            f"severity-correction corr {diagnostics['severity_correction_corr']:.3f}"
        )

    fieldnames = [
        "image", "index", "epoch", "steps",
        "input_psnr_db", "psnr_db", "psnr_gain_db", "mae",
        "nfe", "estimated_gflops", "latency_ms",
        "peak_memory_mib", "inference_peak_extra_mib",
        "correction_severe", "correction_normal",
        "correction_ratio_severe_normal",
        "error_reduction_severe", "error_reduction_normal",
        "severity_correction_corr",
    ]
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    profile_summary.update(
        {
            "epoch": epoch,
            "task": args.task,
            "image_size": train_args.img_size,
            "method": args.method,
            "generation_strength": model.generation_strength,
            "flops_per_nfe": flops_per_nfe,
            "gflops_per_nfe": flops_per_nfe / 1e9 if flops_per_nfe else None,
            "note": "torch.profiler FLOPs cover supported matmul/conv operators",
            "by_steps": {},
        }
    )
    for count in steps:
        selected = [row for row in rows if row["steps"] == count]
        profile_summary["by_steps"][str(count)] = {
            "nfe": nfe_for_steps(count, args.method),
            "estimated_gflops": selected[0]["estimated_gflops"],
            "mean_latency_ms": float(np.mean([row["latency_ms"] for row in selected])),
            "max_peak_memory_mib": max(row["peak_memory_mib"] for row in selected),
            "max_inference_peak_extra_mib": max(
                row["inference_peak_extra_mib"] for row in selected
            ),
        }
    with open(output_dir / "profile.json", "w", encoding="utf-8") as handle:
        json.dump(profile_summary, handle, indent=2)

    print(f"Saved diagnostics to {output_dir}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())

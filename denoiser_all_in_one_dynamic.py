"""Blind dual-branch dynamic JiT with a decoupled restoration bridge.

Experiment A: keep the ResShift-style path schedule a_t, but replace the
bound noise b_t=kappa*sqrt(a_t) with an independent schedule, and replace the
ResShift stochastic posterior with deterministic bridge transport.

Inference with fewer than the trained step count uses a subsequence of the
canonical training (a,b) schedule, so the network never sees unseen pairings.

Native-resolution inputs are padded to a multiple of the ViT patch size,
restored, then cropped back. Training crops may be any HxW divisible by 16.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_jit_aio_dynamic import DynamicAIOJiT_models


class DynamicAllInOneRestorationDenoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = DynamicAIOJiT_models[args.model](
            input_size=getattr(args, "img_size", 512),
            in_channels=3,
            out_channels=3,
            use_observation_branch=True,
            use_shallow_skip=True,
            bottleneck_dim=128,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.patch_size = self.net.patch_size
        self.bridge_type = getattr(args, "bridge_type", "decoupled_bridge")
        self.conditioning_type = getattr(
            args,
            "conditioning_type",
            "state_and_degraded",
        )
        self.lambda_flow = getattr(args, "lambda_flow", 1.0)
        self.lambda_l1 = getattr(args, "lambda_l1", 1.0)
        self.prediction_type = args.prediction_type
        self.resshift_steps = getattr(args, "resshift_steps", 15)
        # Deprecated for Experiment A path construction; kept only so old
        # argparse/configs do not break. Path start uses bridge_path_start.
        self.resshift_kappa = getattr(args, "resshift_kappa", 0.2)
        self.resshift_schedule_power = getattr(
            args,
            "resshift_schedule_power",
            0.3,
        )
        self.resshift_eta_end = getattr(args, "resshift_eta_end", 0.999)
        self.bridge_path_start = getattr(args, "bridge_path_start", 0.001)
        self.hard_eta_mix = getattr(args, "hard_eta_mix", 0.5)
        self.bridge_noise_shared = getattr(args, "bridge_noise_shared", 0.2)
        self.bridge_noise_terminal = getattr(
            args,
            "bridge_noise_terminal",
            0.2,
        )
        if self.prediction_type != "conditional_x":
            raise ValueError("Dynamic All-in-One JiT requires conditional_x")
        if self.bridge_type != "decoupled_bridge":
            raise ValueError(
                "Experiment A requires bridge_type=decoupled_bridge"
            )
        if self.conditioning_type != "state_and_degraded":
            raise ValueError(
                "Dynamic All-in-One JiT requires "
                "conditioning_type=state_and_degraded"
            )
        if self.resshift_steps < 1:
            raise ValueError("resshift_steps must be positive")
        if self.resshift_schedule_power <= 0:
            raise ValueError("resshift_schedule_power must be positive")
        if not 0.0 < self.resshift_eta_end < 1.0:
            raise ValueError("resshift_eta_end must be in (0, 1)")
        if not 0.0 < self.bridge_path_start < self.resshift_eta_end:
            raise ValueError(
                "bridge_path_start must satisfy "
                "0 < bridge_path_start < resshift_eta_end"
            )
        if not 0.0 <= self.hard_eta_mix <= 1.0:
            raise ValueError("hard_eta_mix must be in [0, 1]")
        if self.bridge_noise_shared < 0:
            raise ValueError("bridge_noise_shared must be non-negative")
        if self.bridge_noise_terminal < 0:
            raise ValueError("bridge_noise_terminal must be non-negative")

        self.ema_decay = args.ema_decay
        self.ema_params = None
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def _eta_schedule(self, steps, device, dtype):
        """ResShift-style path coefficients used as a_t (κ-independent)."""
        eta_start = self.bridge_path_start
        if steps == 1:
            return torch.tensor(
                [0.0, self.resshift_eta_end],
                device=device,
                dtype=dtype,
            )
        progress = torch.linspace(
            0.0,
            1.0,
            steps,
            device=device,
            dtype=torch.float64,
        )
        log_eta = (
            math.log(eta_start)
            + progress.pow(self.resshift_schedule_power)
            * (math.log(self.resshift_eta_end) - math.log(eta_start))
        )
        eta = log_eta.exp().to(dtype=dtype)
        return torch.cat([eta.new_zeros(1), eta])

    def _bridge_schedules(self, steps, device, dtype):
        """Canonical schedules of length steps+1 with independent a_t / b_t."""
        a_schedule = self._eta_schedule(steps, device, dtype).clone()
        # Exact degraded endpoint so train/test share x_T = y + b_T eps.
        a_schedule[-1] = a_schedule.new_tensor(1.0)

        tau = torch.linspace(
            0.0,
            1.0,
            steps + 1,
            device=device,
            dtype=dtype,
        )
        b_schedule = (
            self.bridge_noise_shared * tau * (1.0 - tau)
            + self.bridge_noise_terminal * tau.pow(2)
        )
        b_schedule[0] = b_schedule.new_tensor(0.0)
        return a_schedule, b_schedule

    def _canonical_bridge_schedules(self, device, dtype):
        """Training schedule of length T+1; source of all inference (a,b)."""
        return self._bridge_schedules(
            self.resshift_steps,
            device,
            dtype,
        )

    def _subsample_schedule_indices(self, steps):
        """Pick steps+1 indices from the trained 0..T grid, keeping endpoints."""
        train_steps = self.resshift_steps
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        if steps > train_steps:
            raise ValueError(
                f"Requested {steps} sampling steps exceeds the trained "
                f"canonical schedule ({train_steps}). Use a subsequence "
                "of the training (a,b) pairs instead of regenerating."
            )
        if steps == train_steps:
            return list(range(train_steps + 1))

        indices = (
            torch.linspace(0, train_steps, steps + 1)
            .round()
            .to(torch.long)
            .tolist()
        )
        indices[0] = 0
        indices[-1] = train_steps
        for index in range(1, len(indices)):
            if indices[index] <= indices[index - 1]:
                indices[index] = indices[index - 1] + 1
        if indices[-1] > train_steps:
            indices[-1] = train_steps
            for index in range(len(indices) - 2, -1, -1):
                if indices[index] >= indices[index + 1]:
                    indices[index] = indices[index + 1] - 1
        if indices[0] != 0 or indices[-1] != train_steps:
            raise RuntimeError(
                f"Failed to build subsequence endpoints: {indices}"
            )
        if len(set(indices)) != len(indices):
            raise RuntimeError(
                f"Non-unique subsequence indices: {indices}"
            )
        return indices

    def _inference_bridge_schedules(self, steps, device, dtype):
        """Subsequence of the trained (a,b) schedule for fair multi-step eval."""
        a_full, b_full = self._canonical_bridge_schedules(device, dtype)
        indices = self._subsample_schedule_indices(steps)
        index_tensor = torch.tensor(
            indices,
            device=device,
            dtype=torch.long,
        )
        return a_full[index_tensor], b_full[index_tensor]

    def _sample_step_indices(self, batch_size, device):
        """Keep the previous high-a mixture curriculum on the path schedule."""
        a_schedule, _ = self._canonical_bridge_schedules(
            device,
            torch.float32,
        )
        path = a_schedule[1:]
        uniform_prob = torch.ones_like(path)
        uniform_prob = uniform_prob / uniform_prob.sum()
        hard_prob = path / path.sum()
        sample_prob = (
            (1.0 - self.hard_eta_mix) * uniform_prob
            + self.hard_eta_mix * hard_prob
        )
        return (
            torch.multinomial(
                sample_prob,
                batch_size,
                replacement=True,
            )
            + 1
        )

    def _validate_training_shape(self, clean, degraded):
        if clean.shape != degraded.shape:
            raise ValueError(
                f"Clean/degraded shapes differ: {clean.shape} vs {degraded.shape}"
            )
        height, width = clean.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Training shape {height}x{width} must be divisible by "
                f"{self.patch_size}"
            )

    def forward(self, clean, degraded):
        self._validate_training_shape(clean, degraded)
        a_schedule, b_schedule = self._canonical_bridge_schedules(
            clean.device,
            clean.dtype,
        )
        step_indices = self._sample_step_indices(
            clean.size(0),
            clean.device,
        )
        a_t = a_schedule[step_indices].view(
            -1,
            *([1] * (clean.ndim - 1)),
        )
        b_t = b_schedule[step_indices].view(
            -1,
            *([1] * (clean.ndim - 1)),
        )
        noise = torch.randn_like(clean)
        state = (
            clean
            + a_t * (degraded - clean)
            + b_t * noise
        )
        # Condition on path coefficient a_t (b_t is a deterministic function of t).
        clean_pred = self.net(
            state,
            a_t.flatten(),
            observation=degraded,
        )
        flow_loss = (clean_pred - clean).pow(2).mean()
        l1_loss = (clean_pred - clean).abs().mean()
        self.loss_terms = {
            "flow": flow_loss.detach(),
            "l1": l1_loss.detach(),
            "a": a_t.mean().detach(),
            "b": b_t.mean().detach(),
        }
        return self.lambda_flow * flow_loss + self.lambda_l1 * l1_loss

    def pad_to_patch(self, tensor):
        height, width = tensor.shape[-2:]
        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return tensor, (height, width)
        mode = (
            "reflect"
            if height > pad_h and width > pad_w
            else "replicate"
        )
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode), (
            height,
            width,
        )

    @torch.no_grad()
    def restore(
        self,
        degraded,
        generator=None,
        initial_noise=None,
        steps=None,
        method=None,
        return_trajectory=False,
    ):
        padded, original_size = self.pad_to_patch(degraded)
        if initial_noise is not None:
            initial_noise, _ = self.pad_to_patch(initial_noise)
        result = self._restore_padded(
            padded,
            generator=generator,
            initial_noise=initial_noise,
            steps=steps,
            method=method,
            return_trajectory=return_trajectory,
        )
        height, width = original_size
        if return_trajectory:
            restored, trajectory, trajectory_coeffs, trajectory_x0 = result
            restored = restored[..., :height, :width]
            trajectory = [
                state[..., :height, :width] for state in trajectory
            ]
            trajectory_x0 = [
                prediction[..., :height, :width]
                for prediction in trajectory_x0
            ]
            return restored, trajectory, trajectory_coeffs, trajectory_x0
        return result[..., :height, :width]

    @torch.no_grad()
    def _restore_padded(
        self,
        degraded,
        generator=None,
        initial_noise=None,
        steps=None,
        method=None,
        return_trajectory=False,
    ):
        steps = self.steps if steps is None else steps
        method = self.method if method is None else method
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        if method != "deterministic_bridge":
            raise ValueError(
                "Decoupled bridge requires "
                "sampling_method='deterministic_bridge'"
            )
        if initial_noise is None:
            initial_noise = torch.randn(
                degraded.shape,
                device=degraded.device,
                dtype=degraded.dtype,
                generator=generator,
            )
        a_schedule, b_schedule = self._inference_bridge_schedules(
            steps,
            degraded.device,
            degraded.dtype,
        )
        state = self.make_initial_state(
            degraded,
            initial_noise,
            a_schedule=a_schedule,
            b_schedule=b_schedule,
        )
        trajectory = [state.clone()] if return_trajectory else None
        trajectory_coeffs = (
            [(float(a_schedule[-1]), float(b_schedule[-1]))]
            if return_trajectory
            else None
        )
        trajectory_x0 = [] if return_trajectory else None
        for index in range(steps, 0, -1):
            a_t = a_schedule[index]
            b_t = b_schedule[index]
            a_s = a_schedule[index - 1]
            b_s = b_schedule[index - 1]
            time_batch = a_t.expand(state.size(0))
            clean_pred = self._predict_clean(
                state,
                degraded,
                time_batch,
            )
            eps_hat = (
                state
                - a_t * degraded
                - (1.0 - a_t) * clean_pred
            ) / b_t.clamp_min(1e-6)
            state = (
                a_s * degraded
                + (1.0 - a_s) * clean_pred
                + b_s * eps_hat
            )
            if return_trajectory:
                trajectory.append(state.clone())
                trajectory_coeffs.append((float(a_s), float(b_s)))
                trajectory_x0.append(clean_pred.clone())
        if return_trajectory:
            return state, trajectory, trajectory_coeffs, trajectory_x0
        return state

    @torch.no_grad()
    def make_initial_state(
        self,
        degraded,
        initial_noise,
        a_schedule=None,
        b_schedule=None,
        steps=None,
    ):
        if a_schedule is None or b_schedule is None:
            steps = self.steps if steps is None else steps
            a_schedule, b_schedule = self._inference_bridge_schedules(
                steps,
                degraded.device,
                degraded.dtype,
            )
        return degraded + b_schedule[-1] * initial_noise

    @torch.no_grad()
    def _predict_clean(self, state, degraded, time):
        return self.net(
            state,
            time.flatten(),
            observation=degraded,
        ).clamp(-1, 1)

    @torch.no_grad()
    def update_ema(self):
        if self.ema_params is None:
            self.ema_params = copy.deepcopy(list(self.parameters()))
            return
        for target, source in zip(self.ema_params, self.parameters()):
            target.detach().mul_(self.ema_decay).add_(
                source,
                alpha=1.0 - self.ema_decay,
            )

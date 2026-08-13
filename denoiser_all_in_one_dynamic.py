"""Blind dual-branch dynamic JiT with a global UDBM-style bridge.

Experiment B: linear HQ→LQ path alpha_t = t, with UDBM-style noise
beta_t = lambda_b * t(1-t) + lambda_r * t^2, uniform t sampling, and
deterministic bridge transport.

Inference with fewer than the trained step count uses a subsequence of the
canonical training (t, beta) schedule, so the network never sees unseen
pairings.

Native-resolution inputs are padded to a multiple of the ViT patch size,
restored, then cropped back. Training crops may be any HxW divisible by 16.
"""

import copy

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
        self.bridge_type = getattr(
            args,
            "bridge_type",
            "global_udbm_bridge",
        )
        self.conditioning_type = getattr(
            args,
            "conditioning_type",
            "state_and_degraded",
        )
        self.lambda_flow = getattr(args, "lambda_flow", 1.0)
        self.lambda_l1 = getattr(args, "lambda_l1", 1.0)
        self.prediction_type = args.prediction_type
        self.bridge_steps = getattr(args, "bridge_steps", 15)
        self.bridge_noise_shared = getattr(
            args,
            "bridge_noise_shared",
            0.6,
        )
        self.bridge_noise_terminal = getattr(
            args,
            "bridge_noise_terminal",
            0.2,
        )
        if self.prediction_type != "conditional_x":
            raise ValueError("Dynamic All-in-One JiT requires conditional_x")
        if self.bridge_type != "global_udbm_bridge":
            raise ValueError(
                "Experiment B requires bridge_type=global_udbm_bridge"
            )
        if self.conditioning_type != "state_and_degraded":
            raise ValueError(
                "Dynamic All-in-One JiT requires "
                "conditioning_type=state_and_degraded"
            )
        if self.bridge_steps < 1:
            raise ValueError("bridge_steps must be positive")
        if self.bridge_noise_shared < 0:
            raise ValueError("bridge_noise_shared must be non-negative")
        if self.bridge_noise_terminal < 0:
            raise ValueError("bridge_noise_terminal must be non-negative")

        self.ema_decay = args.ema_decay
        self.ema_params = None
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def _bridge_schedules(self, steps, device, dtype):
        """Global UDBM-style linear path with decoupled bridge/terminal noise."""
        tau = torch.linspace(
            0.0,
            1.0,
            steps + 1,
            device=device,
            dtype=dtype,
        )
        # Linear HQ -> LQ mean path: mu_t = (1 - t) * x + t * y
        a_schedule = tau.clone()
        # UDBM-style stochastic schedule:
        # beta_t = lambda_b * t(1-t) + lambda_r * t^2
        b_schedule = (
            self.bridge_noise_shared * tau * (1.0 - tau)
            + self.bridge_noise_terminal * tau.pow(2)
        )
        return a_schedule, b_schedule

    def _canonical_bridge_schedules(self, device, dtype):
        """Training schedule of length T+1; source of all inference (t, beta)."""
        return self._bridge_schedules(
            self.bridge_steps,
            device,
            dtype,
        )

    def _subsample_schedule_indices(self, steps):
        """Pick steps+1 indices from the trained 0..T grid, keeping endpoints."""
        train_steps = self.bridge_steps
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        if steps > train_steps:
            raise ValueError(
                f"Requested {steps} sampling steps exceeds the trained "
                f"canonical schedule ({train_steps}). Use a subsequence "
                "of the training (t, beta) pairs instead of regenerating."
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
        """Subsequence of the trained (t, beta) schedule for fair multi-step eval."""
        a_full, b_full = self._canonical_bridge_schedules(device, dtype)
        indices = self._subsample_schedule_indices(steps)
        index_tensor = torch.tensor(
            indices,
            device=device,
            dtype=torch.long,
        )
        return a_full[index_tensor], b_full[index_tensor]

    def _sample_step_indices(self, batch_size, device):
        """Uniformly sample non-zero bridge states t in {1/T, ..., 1}."""
        return torch.randint(
            low=1,
            high=self.bridge_steps + 1,
            size=(batch_size,),
            device=device,
        )

    def describe_bridge_schedule(self):
        """Human-readable table of the canonical UDBM schedule."""
        lines = ["step | t     | bridge | relax | beta"]
        for index in range(self.bridge_steps + 1):
            time = index / self.bridge_steps
            bridge = self.bridge_noise_shared * time * (1.0 - time)
            relax = self.bridge_noise_terminal * time * time
            beta = bridge + relax
            lines.append(
                f"{index:02d}   | "
                f"{time:.3f} | "
                f"{bridge:.3f}  | "
                f"{relax:.3f} | "
                f"{beta:.3f}"
            )
        return "\n".join(lines)

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
        t_t = a_schedule[step_indices].view(
            -1,
            *([1] * (clean.ndim - 1)),
        )
        b_t = b_schedule[step_indices].view(
            -1,
            *([1] * (clean.ndim - 1)),
        )
        noise = torch.randn_like(clean)
        state = (
            (1.0 - t_t) * clean
            + t_t * degraded
            + b_t * noise
        )
        clean_pred = self.net(
            state,
            t_t.flatten(),
            observation=degraded,
        )
        flow_loss = (clean_pred - clean).pow(2).mean()
        l1_loss = (clean_pred - clean).abs().mean()
        self.loss_terms = {
            "flow": flow_loss.detach(),
            "l1": l1_loss.detach(),
            "t": t_t.mean().detach(),
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
                "Global UDBM bridge requires "
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

"""Blind dual-branch dynamic JiT with a global UDBM bridge (u=0).

Experiment B (one-step inference, no pixel uncertainty yet):
  Official UDBM schedule with u=0:
      tau = t / (T - 1),  t in {0,...,T-1}
      alpha = tau, gamma = 1 - tau
      beta  = 20 * tau * (1 - tau) + 1 * tau^2
  Train: sample t uniformly over {0,...,T-1}, predict observation residual
      r_target = y - x_0
      x_hat_0 = y - f_theta(x_t, tau, y)
  Infer: one forward from terminal state
      x_{T-1} = y + 1 * eps
      r_hat = f_theta(x_{T-1}, tau=1, y)
      x_hat_0 = y - r_hat

This is the global (u=0) special case of UDBM, not the full uncertainty-aware
model. bridge_version identifies the exact schedule for checkpoint safety.

Native-resolution inputs are padded to a multiple of the ViT patch size.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_jit_aio_dynamic import DynamicAIOJiT_models

BRIDGE_VERSION = "udbm_exact_v1"


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
        self.bridge_version = getattr(
            args,
            "bridge_version",
            BRIDGE_VERSION,
        )
        self.conditioning_type = getattr(
            args,
            "conditioning_type",
            "state_and_degraded",
        )
        self.lambda_flow = getattr(args, "lambda_flow", 1.0)
        self.lambda_l1 = getattr(args, "lambda_l1", 1.0)
        self.output_parameterization = "observation_residual_v1"
        self.prediction_type = args.prediction_type
        self.diffusion_steps = getattr(args, "diffusion_steps", 1000)
        # Official UDBM effective coefficients for u=0:
        # beta = 20*(1+u)*tau*(1-tau) + (1+u)*tau^2
        self.bridge_noise_shared = getattr(
            args,
            "bridge_noise_shared",
            20.0,
        )
        self.bridge_noise_terminal = getattr(
            args,
            "bridge_noise_terminal",
            1.0,
        )
        if self.prediction_type != "conditional_x":
            raise ValueError("Dynamic All-in-One JiT requires conditional_x")
        if self.bridge_type != "global_udbm_bridge":
            raise ValueError(
                "Experiment B requires bridge_type=global_udbm_bridge"
            )
        if self.bridge_version != BRIDGE_VERSION:
            raise ValueError(
                f"Expected bridge_version={BRIDGE_VERSION!r}, "
                f"got {self.bridge_version!r}"
            )
        if self.conditioning_type != "state_and_degraded":
            raise ValueError(
                "Dynamic All-in-One JiT requires "
                "conditioning_type=state_and_degraded"
            )
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be >= 2")
        if self.bridge_noise_shared < 0:
            raise ValueError("bridge_noise_shared must be non-negative")
        if self.bridge_noise_terminal < 0:
            raise ValueError("bridge_noise_terminal must be non-negative")

        self.ema_decay = args.ema_decay
        self.ema_params = None
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def _tau_from_timesteps(self, timesteps, dtype):
        # Official: t in {0,...,T-1}, tau = t / (T-1)
        return timesteps.to(dtype) / float(self.diffusion_steps - 1)

    def _bridge_coefficients(self, timesteps, ndim, dtype):
        """Return alpha, gamma, beta, tau for integer timesteps in [0, T-1]."""
        tau = self._tau_from_timesteps(timesteps, dtype)
        alpha = tau
        gamma = 1.0 - alpha
        beta = (
            self.bridge_noise_shared * tau * (1.0 - tau)
            + self.bridge_noise_terminal * tau.square()
        )
        shape = (timesteps.shape[0], *([1] * (ndim - 1)))
        return (
            alpha.view(shape),
            gamma.view(shape),
            beta.view(shape),
            tau,
        )

    def _sample_training_timesteps(self, batch_size, device):
        """Uniformly sample t in {0, 1, ..., T-1} (official UDBM)."""
        return torch.randint(
            low=0,
            high=self.diffusion_steps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def describe_bridge_schedule(self, max_rows=17):
        """Sparse human-readable table of the UDBM schedule."""
        last = self.diffusion_steps - 1
        lines = [
            f"bridge_version={self.bridge_version}",
            f"diffusion_steps={self.diffusion_steps} "
            f"(t=0..{last})",
            f"lambda_b={self.bridge_noise_shared} "
            f"lambda_r={self.bridge_noise_terminal}",
            "step | t     | bridge | relax | beta",
        ]
        if self.diffusion_steps <= max_rows:
            indices = list(range(self.diffusion_steps))
        else:
            indices = (
                torch.linspace(0, last, max_rows)
                .round()
                .to(torch.long)
                .tolist()
            )
            indices[0] = 0
            indices[-1] = last
        for index in indices:
            time = index / last
            bridge = self.bridge_noise_shared * time * (1.0 - time)
            relax = self.bridge_noise_terminal * time * time
            beta = bridge + relax
            lines.append(
                f"{index:04d} | "
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
        timesteps = self._sample_training_timesteps(
            clean.size(0),
            clean.device,
        )
        alpha_t, gamma_t, beta_t, tau = self._bridge_coefficients(
            timesteps,
            clean.ndim,
            clean.dtype,
        )
        noise = torch.randn_like(clean)
        state_t = (
            alpha_t * degraded
            + gamma_t * clean
            + beta_t * noise
        )

        # Observation residual used by UDBM:
        # degraded = clean + residual
        residual_target = degraded - clean

        residual_pred = self.net(
            state_t,
            tau,
            observation=degraded,
        )

        # Observation-anchored x0 prediction.
        clean_pred = degraded - residual_pred

        error = residual_pred - residual_target

        per_sample_mse = error.square().mean(
            dim=(1, 2, 3)
        )
        per_sample_l1 = error.abs().mean(
            dim=(1, 2, 3)
        )

        target_scale = (
            residual_target.detach()
            .abs()
            .mean(dim=(1, 2, 3))
            .clamp_min(2.0 / 255.0)
        )

        # Half normalization: avoids excessively amplifying denoise_15.
        sample_weight = target_scale.rsqrt()
        sample_weight = sample_weight / sample_weight.mean()

        mse_loss = (
            sample_weight * per_sample_mse
        ).mean()

        l1_loss = (
            sample_weight * per_sample_l1
        ).mean()

        reconstruction_mae = (
            clean_pred - clean
        ).abs().mean()

        self.loss_terms = {
            "mse": mse_loss.detach(),
            "l1": l1_loss.detach(),
            "reconstruction_mae": reconstruction_mae.detach(),
            "residual_target_abs": (
                residual_target.abs().mean().detach()
            ),
            "residual_pred_abs": (
                residual_pred.abs().mean().detach()
            ),
            "t": tau.mean().detach(),
            "alpha": alpha_t.mean().detach(),
            "beta": beta_t.mean().detach(),
        }

        return (
            self.lambda_flow * mse_loss
            + self.lambda_l1 * l1_loss
        )

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
    def make_initial_state(
        self,
        degraded,
        generator=None,
        initial_noise=None,
    ):
        """Terminal state at t=T-1: x = y + beta_T * eps with beta_T=1 (u=0)."""
        if initial_noise is None:
            initial_noise = torch.randn(
                degraded.shape,
                device=degraded.device,
                dtype=degraded.dtype,
                generator=generator,
            )
        if initial_noise.shape != degraded.shape:
            raise ValueError(
                "initial_noise must match degraded shape "
                f"{tuple(degraded.shape)}, got {tuple(initial_noise.shape)}"
            )
        timesteps = torch.full(
            (degraded.shape[0],),
            self.diffusion_steps - 1,
            device=degraded.device,
            dtype=torch.long,
        )
        alpha_t, _, beta_t, _ = self._bridge_coefficients(
            timesteps,
            degraded.ndim,
            degraded.dtype,
        )
        return alpha_t * degraded + beta_t * initial_noise

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
        # Generate / validate noise on the padded grid so eval and diagnostics
        # share the same x_T when the same padded noise is passed in.
        if initial_noise is None:
            initial_noise = torch.randn(
                padded.shape,
                device=padded.device,
                dtype=padded.dtype,
                generator=generator,
            )
        elif initial_noise.shape[-2:] != padded.shape[-2:]:
            raise ValueError(
                "initial_noise spatial size must match the patch-padded "
                f"input {tuple(padded.shape[-2:])}, got "
                f"{tuple(initial_noise.shape[-2:])}"
            )
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
        if steps != 1:
            raise ValueError(
                "UDBM one-step inference requires steps=1"
            )
        if method != "one_step":
            raise ValueError(
                "Global UDBM one-step inference requires "
                "sampling_method='one_step'"
            )
        state_t = self.make_initial_state(
            degraded,
            generator=generator,
            initial_noise=initial_noise,
        )
        tau_t = torch.ones(
            degraded.shape[0],
            device=degraded.device,
            dtype=degraded.dtype,
        )
        residual_pred = self.net(
            state_t,
            tau_t,
            observation=degraded,
        )
        clean_pred = degraded - residual_pred
        restored = clean_pred.clamp(-1.0, 1.0)
        if return_trajectory:
            return (
                restored,
                [state_t.clone(), restored.clone()],
                [
                    (1.0, float(self.bridge_noise_terminal)),
                    (0.0, 0.0),
                ],
                [restored.clone()],
            )
        return restored

    @torch.no_grad()
    def routing_difficulty_map(
        self,
        degraded,
        generator=None,
        initial_noise=None,
    ):
        """One-step internal routing map at native resolution.

        Returns a [B, 1, H, W] float map in [0, 1] indicating which pixels
        persistently trigger large internal feature updates across JiT blocks.
        This is not an externally predicted degradation map; it reflects JiT's
        own recovery dynamics.
        """
        padded, original_size = self.pad_to_patch(degraded)
        if initial_noise is None:
            initial_noise = torch.randn(
                padded.shape,
                device=padded.device,
                dtype=padded.dtype,
                generator=generator,
            )
        state_t = self.make_initial_state(
            padded,
            generator=generator,
            initial_noise=initial_noise,
        )
        tau_t = torch.ones(
            padded.shape[0],
            device=padded.device,
            dtype=padded.dtype,
        )
        _, adaptive_info = self.net(
            state_t,
            tau_t,
            observation=padded,
            return_adaptive_info=True,
        )
        patch_mask = adaptive_info["difficulty_map"]
        if patch_mask is None:
            return None
        height, width = original_size
        return F.interpolate(
            patch_mask.float(),
            size=(height, width),
            mode="nearest",
        )

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

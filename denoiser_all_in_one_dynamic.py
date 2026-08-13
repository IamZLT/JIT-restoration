"""Blind dual-branch dynamic JiT with a global UDBM-style bridge.

Experiment B (one-step inference):
  Train over the full discrete timeline t in {0,...,T}:
      x_t = alpha_t * y + gamma_t * x + beta_t * eps
      alpha_t = tau, gamma_t = 1 - tau, tau = t / T
      beta_t  = lambda_b * tau(1-tau) + lambda_r * tau^2
  Infer with a single network call from the terminal state:
      x_T = y + lambda_r * eps
      x_hat_0 = f_theta(x_T, tau=1, y)

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
        self.diffusion_steps = getattr(args, "diffusion_steps", 1000)
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
        if self.diffusion_steps < 1:
            raise ValueError("diffusion_steps must be positive")
        if self.bridge_noise_shared < 0:
            raise ValueError("bridge_noise_shared must be non-negative")
        if self.bridge_noise_terminal < 0:
            raise ValueError("bridge_noise_terminal must be non-negative")

        self.ema_decay = args.ema_decay
        self.ema_params = None
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def _bridge_coefficients(self, timesteps, ndim, dtype):
        """Return alpha, gamma, beta, tau for integer timesteps in [0, T]."""
        tau = timesteps.to(dtype) / float(self.diffusion_steps)
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
        """Uniformly sample t in {0, 1, ..., T}."""
        return torch.randint(
            low=0,
            high=self.diffusion_steps + 1,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def describe_bridge_schedule(self, max_rows=17):
        """Sparse human-readable table of the UDBM schedule."""
        lines = [
            f"diffusion_steps={self.diffusion_steps} "
            f"lambda_b={self.bridge_noise_shared} "
            f"lambda_r={self.bridge_noise_terminal}",
            "step | t     | bridge | relax | beta",
        ]
        if self.diffusion_steps + 1 <= max_rows:
            indices = list(range(self.diffusion_steps + 1))
        else:
            indices = (
                torch.linspace(0, self.diffusion_steps, max_rows)
                .round()
                .to(torch.long)
                .tolist()
            )
            indices[0] = 0
            indices[-1] = self.diffusion_steps
        for index in indices:
            time = index / self.diffusion_steps
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
        clean_pred = self.net(
            state_t,
            tau,
            observation=degraded,
        )
        mse_loss = (clean_pred - clean).square().mean()
        l1_loss = (clean_pred - clean).abs().mean()
        self.loss_terms = {
            "mse": mse_loss.detach(),
            "l1": l1_loss.detach(),
            "t": tau.mean().detach(),
            "alpha": alpha_t.mean().detach(),
            "beta": beta_t.mean().detach(),
        }
        return self.lambda_flow * mse_loss + self.lambda_l1 * l1_loss

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
        """Construct the terminal bridge state x_T = y + lambda_r * eps."""
        if initial_noise is None:
            initial_noise = torch.randn(
                degraded.shape,
                device=degraded.device,
                dtype=degraded.dtype,
                generator=generator,
            )
        timesteps = torch.full(
            (degraded.shape[0],),
            self.diffusion_steps,
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
        clean_pred = self.net(
            state_t,
            tau_t,
            observation=degraded,
        )
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
    def update_ema(self):
        if self.ema_params is None:
            self.ema_params = copy.deepcopy(list(self.parameters()))
            return
        for target, source in zip(self.ema_params, self.parameters()):
            target.detach().mul_(self.ema_decay).add_(
                source,
                alpha=1.0 - self.ema_decay,
            )

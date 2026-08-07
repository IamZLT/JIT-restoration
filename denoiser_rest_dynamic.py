"""Conditional restoration denoiser backed by dynamic-resolution JiT."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_jit_dynamic import DynamicJiT_models


class DynamicRestorationDenoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = DynamicJiT_models[args.model](
            input_size=getattr(args, "img_size", 512),
            in_channels=6,
            out_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.patch_size = self.net.patch_size
        self.num_classes = args.class_num
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.noise_scale = args.noise_scale
        self.cond_drop_prob = getattr(args, "cond_drop_prob", 0.0)
        self.lambda_flow = getattr(args, "lambda_flow", 1.0)
        self.lambda_l1 = getattr(args, "lambda_l1", 1.0)
        self.prediction_type = args.prediction_type
        self.generation_strength = args.generation_strength
        self.t_eps = args.t_eps
        if self.prediction_type != "conditional_x":
            raise ValueError("Dynamic JiT supports prediction_type='conditional_x'")
        if not 0.0 <= self.generation_strength <= 1.0:
            raise ValueError("generation_strength must be in [0, 1]")

        self.ema_decay = args.ema_decay
        self.ema_params = None
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def sample_t(self, count, device=None):
        logits = (
            torch.randn(count, device=device) * self.P_std + self.P_mean
        )
        return torch.sigmoid(logits)

    def _maybe_drop_cond(self, condition):
        if not self.training or self.cond_drop_prob <= 0:
            return condition
        drop = (
            torch.rand(condition.size(0), device=condition.device)
            < self.cond_drop_prob
        )
        if not drop.any():
            return condition
        condition = condition.clone()
        condition[drop] = 0
        return condition

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
        condition = self._maybe_drop_cond(degraded)
        t = self.sample_t(clean.size(0), clean.device).view(
            -1,
            *([1] * (clean.ndim - 1)),
        )
        noise = torch.randn_like(clean) * self.noise_scale
        state = t * clean + (1.0 - t) * noise
        velocity_target = (clean - state) / (1.0 - t).clamp_min(
            self.t_eps
        )
        labels = torch.zeros(
            clean.size(0),
            dtype=torch.long,
            device=clean.device,
        )
        clean_pred = self.net(
            torch.cat([state, condition], dim=1),
            t.flatten(),
            labels,
        )
        velocity_pred = (clean_pred - state) / (
            1.0 - t
        ).clamp_min(self.t_eps)
        flow_loss = (velocity_pred - velocity_target).pow(2).mean()
        l1_loss = (clean_pred - clean).abs().mean()
        self.loss_terms = {
            "flow": flow_loss.detach(),
            "l1": l1_loss.detach(),
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
        steps=None,
        method=None,
    ):
        padded, original_size = self.pad_to_patch(degraded)
        restored = self._restore_padded(
            padded,
            generator=generator,
            steps=steps,
            method=method,
        )
        height, width = original_size
        return restored[..., :height, :width]

    @torch.no_grad()
    def restore_trajectory(
        self,
        degraded,
        generator=None,
        steps=None,
        method=None,
    ):
        """Return the native-size final result and every sampling state."""
        padded, original_size = self.pad_to_patch(degraded)
        restored, trajectory = self._sample_padded(
            padded,
            generator=generator,
            steps=steps,
            method=method,
            record=True,
        )
        height, width = original_size
        restored = restored[..., :height, :width]
        trajectory = [
            state[..., :height, :width]
            for state in trajectory
        ]
        return restored, trajectory

    @torch.no_grad()
    def _restore_padded(
        self,
        degraded,
        generator=None,
        steps=None,
        method=None,
    ):
        restored, _ = self._sample_padded(
            degraded,
            generator=generator,
            steps=steps,
            method=method,
            record=False,
        )
        return restored

    @torch.no_grad()
    def _sample_padded(
        self,
        degraded,
        generator=None,
        steps=None,
        method=None,
        record=False,
    ):
        steps = self.steps if steps is None else steps
        method = self.method if method is None else method
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        if self.generation_strength == 0.0:
            restored = degraded.clone()
            return restored, [restored.clone()] if record else None
        noise = torch.randn(
            degraded.shape,
            device=degraded.device,
            dtype=degraded.dtype,
            generator=generator,
        )
        t_start = 1.0 - self.generation_strength
        state = (
            t_start * degraded
            + self.generation_strength * self.noise_scale * noise
        )
        trajectory = [state.clone()] if record else None
        timesteps = torch.linspace(
            t_start,
            1.0,
            steps + 1,
            device=degraded.device,
        )
        timesteps = timesteps.view(-1, 1, 1, 1, 1).expand(
            -1,
            degraded.size(0),
            -1,
            -1,
            -1,
        )
        for index in range(steps):
            t = timesteps[index]
            t_next = timesteps[index + 1]
            if method == "heun" and index < steps - 1:
                state = self._heun_step(
                    state,
                    degraded,
                    t,
                    t_next,
                )
            elif method in ("euler", "heun"):
                state = self._euler_step(
                    state,
                    degraded,
                    t,
                    t_next,
                )
            else:
                raise NotImplementedError(method)
            if record:
                trajectory.append(state.clone())
        return state, trajectory

    @torch.no_grad()
    def _forward_sample(self, state, degraded, t):
        labels = torch.zeros(
            state.size(0),
            dtype=torch.long,
            device=state.device,
        )
        clean_pred = self.net(
            torch.cat([state, degraded], dim=1),
            t.flatten(),
            labels,
        )
        return (clean_pred - state) / (1.0 - t).clamp_min(self.t_eps)

    @torch.no_grad()
    def _euler_step(self, state, degraded, t, t_next):
        velocity = self._forward_sample(state, degraded, t)
        return state + (t_next - t) * velocity

    @torch.no_grad()
    def _heun_step(self, state, degraded, t, t_next):
        velocity = self._forward_sample(state, degraded, t)
        euler = state + (t_next - t) * velocity
        velocity_next = self._forward_sample(
            euler,
            degraded,
            t_next,
        )
        return state + (t_next - t) * 0.5 * (
            velocity + velocity_next
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

    @torch.no_grad()
    def load_ema(self):
        if self.ema_params is None:
            raise RuntimeError("EMA parameters have not been initialized")
        for parameter, ema_parameter in zip(
            self.parameters(),
            self.ema_params,
        ):
            parameter.data.copy_(ema_parameter.data)

    def load_compatible_state_dict(self, state_dict):
        """Load dynamic or legacy fixed-resolution JiT restoration weights."""
        filtered = {
            name: value
            for name, value in state_dict.items()
            if name != "net.pos_embed"
        }
        incompatible = self.load_state_dict(filtered, strict=False)
        missing = [
            name
            for name in incompatible.missing_keys
            if name != "net.pos_embed"
        ]
        unexpected = [
            name
            for name in incompatible.unexpected_keys
            if name != "net.pos_embed"
        ]
        if missing or unexpected:
            raise RuntimeError(
                f"Incompatible checkpoint; missing={missing}, "
                f"unexpected={unexpected}"
            )

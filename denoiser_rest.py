"""Conditional pixel-space generative JiT for paired image restoration.

Training noises the clean target at arbitrary times while the uncorrupted LQ
image remains a condition:
    z_t = t * clean + (1 - t) * noise

Inference follows an img2img strategy:
    z_start = (1 - strength) * LQ + strength * noise

The model directly predicts clean x (x-pred), following JiT's high-dimensional
pixel-space parameterization.
"""

import copy

import torch
import torch.nn as nn

from model_jit import JiT_models


class RestorationDenoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=6,
            out_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
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
            raise ValueError(
                "Only prediction_type='conditional_x' is supported by this version"
            )
        if not 0.0 <= self.generation_strength <= 1.0:
            raise ValueError("generation_strength must be in [0, 1]")

        self.ema_decay = args.ema_decay
        self.ema_params = None

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def sample_t(self, n, device=None):
        # Match JiT's logit-normal time distribution.
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def _maybe_drop_cond(self, y):
        if (not self.training) or self.cond_drop_prob <= 0:
            return y
        drop = torch.rand(y.size(0), device=y.device) < self.cond_drop_prob
        if not drop.any():
            return y
        y = y.clone()
        y[drop] = 0
        return y

    def forward(self, x, y):
        """
        x: clean image in [-1, 1]
        y: degraded image in [-1, 1]
        """
        condition = self._maybe_drop_cond(y)
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        noise = torch.randn_like(x) * self.noise_scale
        z = t * x + (1.0 - t) * noise
        v_target = (x - z) / (1.0 - t).clamp_min(self.t_eps)

        labels = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_pred = self.net(torch.cat([z, condition], dim=1), t.flatten(), labels)
        v_pred = (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

        # Explicit GT reconstruction supervision complements velocity matching.
        loss_flow = (v_pred - v_target).pow(2).mean()
        loss_l1 = (x_pred - x).abs().mean()
        self.loss_terms = {
            "flow": loss_flow.detach(),
            "l1": loss_l1.detach(),
        }
        return self.lambda_flow * loss_flow + self.lambda_l1 * loss_l1

    @torch.no_grad()
    def restore(self, y, generator=None):
        """Restore a batch of degraded images y in [-1, 1]."""
        device = y.device
        bsz = y.size(0)
        noise = torch.randn(y.shape, device=device, dtype=y.dtype, generator=generator)
        t_start = 1.0 - self.generation_strength
        if self.generation_strength == 0.0:
            return y.clone()
        z = t_start * y + (1.0 - t_start) * self.noise_scale * noise
        timesteps = torch.linspace(t_start, 1.0, self.steps + 1, device=device)
        timesteps = timesteps.view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(self.method)

        for i in range(self.steps - 1):
            z = stepper(z, y, timesteps[i], timesteps[i + 1])
        z = self._euler_step(z, y, timesteps[-2], timesteps[-1])
        return z

    @torch.no_grad()
    def _forward_sample(self, z, y, t):
        labels = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        x_pred = self.net(torch.cat([z, y], dim=1), t.flatten(), labels)
        return (x_pred - z) / (1.0 - t).clamp_min(self.t_eps)

    @torch.no_grad()
    def _euler_step(self, z, y, t, t_next):
        v_pred = self._forward_sample(z, y, t)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, z, y, t, t_next):
        v_t = self._forward_sample(z, y, t)
        z_euler = z + (t_next - t) * v_t
        v_next = self._forward_sample(z_euler, y, t_next)
        return z + (t_next - t) * 0.5 * (v_t + v_next)

    @torch.no_grad()
    def update_ema(self):
        if self.ema_params is None:
            self.ema_params = copy.deepcopy(list(self.parameters()))
            return
        for targ, src in zip(self.ema_params, self.parameters()):
            targ.detach().mul_(self.ema_decay).add_(src, alpha=1.0 - self.ema_decay)

    @torch.no_grad()
    def load_ema(self):
        assert self.ema_params is not None
        for p, ema_p in zip(self.parameters(), self.ema_params):
            p.data.copy_(ema_p.data)

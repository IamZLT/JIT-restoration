"""Task-conditioned fixed-resolution JiT for All-in-One restoration."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_jit import JiT_models


class AllInOneRestorationDenoiser(nn.Module):
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
            raise ValueError("All-in-One JiT requires conditional_x")
        if self.num_classes < 3:
            raise ValueError("All-in-One JiT requires class_num >= 3")
        if not 0.0 <= self.generation_strength <= 1.0:
            raise ValueError("generation_strength must be in [0, 1]")

        self.ema_decay = args.ema_decay
        self.ema_params = None
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def sample_t(self, count, device):
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
        if drop.any():
            condition = condition.clone()
            condition[drop] = 0
        return condition

    @staticmethod
    def _labels(task_ids, batch_size, device):
        if task_ids is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        task_ids = torch.as_tensor(
            task_ids,
            dtype=torch.long,
            device=device,
        ).flatten()
        if task_ids.numel() == 1 and batch_size > 1:
            task_ids = task_ids.expand(batch_size)
        if task_ids.numel() != batch_size:
            raise ValueError(
                f"Expected {batch_size} task labels, got {task_ids.numel()}"
            )
        return task_ids

    def forward(self, clean, degraded, task_ids):
        if clean.shape != degraded.shape:
            raise ValueError("Clean and degraded tensors must have same shape")
        if clean.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Training patches must be {self.img_size}x{self.img_size}"
            )
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
        labels = self._labels(task_ids, clean.size(0), clean.device)
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

    @torch.no_grad()
    def restore(
        self,
        degraded,
        task_ids,
        generator=None,
        initial_noise=None,
        steps=None,
        method=None,
    ):
        if degraded.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"restore() expects {self.img_size}x{self.img_size}; "
                "use restore_tiled() for native-resolution inputs"
            )
        steps = self.steps if steps is None else steps
        method = self.method if method is None else method
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        if self.generation_strength == 0.0:
            return degraded.clone()
        if initial_noise is None:
            initial_noise = torch.randn(
                degraded.shape,
                device=degraded.device,
                dtype=degraded.dtype,
                generator=generator,
            )
        labels = self._labels(
            task_ids,
            degraded.size(0),
            degraded.device,
        )
        t_start = 1.0 - self.generation_strength
        state = (
            t_start * degraded
            + self.generation_strength * self.noise_scale * initial_noise
        )
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
                    labels,
                    t,
                    t_next,
                )
            elif method in ("euler", "heun"):
                state = self._euler_step(
                    state,
                    degraded,
                    labels,
                    t,
                    t_next,
                )
            else:
                raise NotImplementedError(method)
        return state

    @torch.no_grad()
    def _forward_sample(self, state, degraded, labels, t):
        clean_pred = self.net(
            torch.cat([state, degraded], dim=1),
            t.flatten(),
            labels,
        )
        return (clean_pred - state) / (1.0 - t).clamp_min(self.t_eps)

    @torch.no_grad()
    def _euler_step(self, state, degraded, labels, t, t_next):
        velocity = self._forward_sample(state, degraded, labels, t)
        return state + (t_next - t) * velocity

    @torch.no_grad()
    def _heun_step(self, state, degraded, labels, t, t_next):
        velocity = self._forward_sample(state, degraded, labels, t)
        euler = state + (t_next - t) * velocity
        velocity_next = self._forward_sample(
            euler,
            degraded,
            labels,
            t_next,
        )
        return state + (t_next - t) * 0.5 * (
            velocity + velocity_next
        )

    @staticmethod
    def _tile_positions(length, tile, stride):
        if length <= tile:
            return [0]
        positions = list(range(0, length - tile + 1, stride))
        if positions[-1] != length - tile:
            positions.append(length - tile)
        return positions

    @torch.no_grad()
    def restore_tiled(
        self,
        degraded,
        task_ids,
        generator=None,
        steps=None,
        method=None,
        overlap=64,
        tile_batch_size=1,
    ):
        """Restore native-resolution images using overlapping fixed-size tiles."""
        if degraded.size(0) != 1:
            raise ValueError("restore_tiled currently expects batch size 1")
        if not 0 <= overlap < self.img_size:
            raise ValueError("tile overlap must be in [0, img_size)")
        original_h, original_w = degraded.shape[-2:]
        pad_h = max(self.img_size - original_h, 0)
        pad_w = max(self.img_size - original_w, 0)
        mode = (
            "reflect"
            if original_h > pad_h and original_w > pad_w
            else "replicate"
        )
        padded = F.pad(
            degraded,
            (0, pad_w, 0, pad_h),
            mode=mode,
        )
        height, width = padded.shape[-2:]
        stride = self.img_size - overlap
        rows = self._tile_positions(height, self.img_size, stride)
        cols = self._tile_positions(width, self.img_size, stride)
        full_noise = torch.randn(
            padded.shape,
            device=padded.device,
            dtype=padded.dtype,
            generator=generator,
        )
        output = torch.zeros_like(padded)
        weight = torch.zeros_like(padded)
        coordinates = [(top, left) for top in rows for left in cols]
        labels = self._labels(task_ids, 1, padded.device)

        for start in range(0, len(coordinates), tile_batch_size):
            current = coordinates[start : start + tile_batch_size]
            tiles = torch.cat(
                [
                    padded[
                        :,
                        :,
                        top : top + self.img_size,
                        left : left + self.img_size,
                    ]
                    for top, left in current
                ],
                dim=0,
            )
            noise_tiles = torch.cat(
                [
                    full_noise[
                        :,
                        :,
                        top : top + self.img_size,
                        left : left + self.img_size,
                    ]
                    for top, left in current
                ],
                dim=0,
            )
            restored = self.restore(
                tiles,
                labels.expand(len(current)),
                initial_noise=noise_tiles,
                steps=steps,
                method=method,
            )
            for tile_index, (top, left) in enumerate(current):
                output[
                    :,
                    :,
                    top : top + self.img_size,
                    left : left + self.img_size,
                ] += restored[tile_index : tile_index + 1]
                weight[
                    :,
                    :,
                    top : top + self.img_size,
                    left : left + self.img_size,
                ] += 1
        output = output / weight.clamp_min(1)
        return output[..., :original_h, :original_w]

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

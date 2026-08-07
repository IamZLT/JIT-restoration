"""State-only JiT with a ResShift-style restoration bridge."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_jit import JiT_models


class AllInOneRestorationDenoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            out_channels=3,
            use_class_condition=False,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.bridge_type = getattr(args, "bridge_type", "resshift")
        self.conditioning_type = getattr(
            args,
            "conditioning_type",
            "state_only",
        )
        self.lambda_flow = getattr(args, "lambda_flow", 1.0)
        self.lambda_l1 = getattr(args, "lambda_l1", 1.0)
        self.prediction_type = args.prediction_type
        self.resshift_steps = getattr(args, "resshift_steps", 15)
        self.resshift_kappa = getattr(args, "resshift_kappa", 0.2)
        self.resshift_schedule_power = getattr(
            args,
            "resshift_schedule_power",
            0.3,
        )
        self.resshift_eta_end = getattr(args, "resshift_eta_end", 0.999)
        if self.prediction_type != "conditional_x":
            raise ValueError("All-in-One JiT requires conditional_x")
        if self.bridge_type != "resshift":
            raise ValueError("All-in-One JiT requires bridge_type=resshift")
        if self.conditioning_type != "state_only":
            raise ValueError(
                "All-in-One JiT requires conditioning_type=state_only"
            )
        if self.resshift_steps < 1:
            raise ValueError("resshift_steps must be positive")
        if self.resshift_kappa <= 0:
            raise ValueError("resshift_kappa must be positive")
        if self.resshift_schedule_power <= 0:
            raise ValueError("resshift_schedule_power must be positive")
        if not 0.0 < self.resshift_eta_end < 1.0:
            raise ValueError("resshift_eta_end must be in (0, 1)")

        self.ema_decay = args.ema_decay
        self.ema_params = None
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    def _eta_schedule(self, steps, device, dtype):
        """Return [eta_0=0, eta_1, ..., eta_T] from ResShift."""
        eta_start = min(
            (0.04 / self.resshift_kappa) ** 2,
            0.001,
        )
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

    def forward(self, clean, degraded):
        if clean.shape != degraded.shape:
            raise ValueError("Clean and degraded tensors must have same shape")
        if clean.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Training patches must be {self.img_size}x{self.img_size}"
            )
        eta_schedule = self._eta_schedule(
            self.resshift_steps,
            clean.device,
            clean.dtype,
        )
        step_indices = torch.randint(
            1,
            self.resshift_steps + 1,
            (clean.size(0),),
            device=clean.device,
        )
        eta = eta_schedule[step_indices].view(
            -1,
            *([1] * (clean.ndim - 1)),
        )
        noise = torch.randn_like(clean)
        state = (
            clean
            + eta * (degraded - clean)
            + self.resshift_kappa * eta.sqrt() * noise
        )
        clean_pred = self.net(
            state,
            eta.flatten(),
        )
        # Keep the existing unweighted x0 MSE + L1 objective.
        flow_loss = (clean_pred - clean).pow(2).mean()
        l1_loss = (clean_pred - clean).abs().mean()
        self.loss_terms = {
            "flow": flow_loss.detach(),
            "l1": l1_loss.detach(),
            "eta": eta.mean().detach(),
        }
        return self.lambda_flow * flow_loss + self.lambda_l1 * l1_loss

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
        if degraded.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"restore() expects {self.img_size}x{self.img_size}; "
                "use restore_tiled() for native-resolution inputs"
            )
        steps = self.steps if steps is None else steps
        method = self.method if method is None else method
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        if method != "resshift":
            raise ValueError("ResShift bridge requires sampling_method='resshift'")
        if initial_noise is None:
            initial_noise = torch.randn(
                degraded.shape,
                device=degraded.device,
                dtype=degraded.dtype,
                generator=generator,
            )
        eta_schedule = self._eta_schedule(
            steps,
            degraded.device,
            degraded.dtype,
        )
        state = self.make_initial_state(
            degraded,
            initial_noise,
            eta_schedule=eta_schedule,
        )
        trajectory = [state.clone()] if return_trajectory else None
        trajectory_etas = (
            [float(eta_schedule[-1])] if return_trajectory else None
        )
        for index in range(steps, 0, -1):
            eta_t = eta_schedule[index]
            eta_previous = eta_schedule[index - 1]
            eta_batch = eta_t.expand(state.size(0))
            clean_pred = self._predict_clean(
                state,
                eta_batch,
            )
            alpha = eta_t - eta_previous
            state = (
                (eta_previous / eta_t) * state
                + (alpha / eta_t) * clean_pred
            )
            if index > 1:
                posterior_variance = (
                    self.resshift_kappa**2
                    * (eta_previous / eta_t)
                    * alpha
                )
                posterior_noise = torch.randn(
                    state.shape,
                    device=state.device,
                    dtype=state.dtype,
                    generator=generator,
                )
                state = (
                    state
                    + posterior_variance.clamp_min(0).sqrt()
                    * posterior_noise
                )
            if return_trajectory:
                trajectory.append(state.clone())
                trajectory_etas.append(float(eta_previous))
        if return_trajectory:
            return state, trajectory, trajectory_etas
        return state

    @torch.no_grad()
    def make_initial_state(
        self,
        degraded,
        initial_noise,
        eta_schedule=None,
        steps=None,
    ):
        if eta_schedule is None:
            steps = self.steps if steps is None else steps
            eta_schedule = self._eta_schedule(
                steps,
                degraded.device,
                degraded.dtype,
            )
        return (
            degraded
            + self.resshift_kappa
            * eta_schedule[-1].sqrt()
            * initial_noise
        )

    @torch.no_grad()
    def _predict_clean(self, state, eta):
        return self.net(state, eta.flatten()).clamp(-1, 1)

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

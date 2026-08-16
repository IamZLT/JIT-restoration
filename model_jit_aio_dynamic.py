"""Dynamic-resolution JiT for blind All-in-One restoration.

Separate from ``model_jit_dynamic.py`` (single-task 6-ch) and fixed
``model_jit.py``. Supports dual RGB patch streams, generic registers,
shallow skip, and runtime 2-D RoPE / sin-cos positional encodings.
"""

import torch
import torch.nn as nn

from model_jit import TimestepEmbedder, modulate
from model_jit_dynamic import (
    DynamicBottleneckPatchEmbed,
    DynamicFinalLayer,
    DynamicJiTBlock,
    DynamicVisionRoPE,
)


class DynamicAIOJiT(nn.Module):
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        use_observation_branch=True,
        use_shallow_skip=True,
    ):
        super().__init__()
        del input_size
        if in_channels != 3:
            raise ValueError("DynamicAIOJiT expects RGB inputs (in_channels=3)")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.use_observation_branch = use_observation_branch
        self.use_shallow_skip = use_shallow_skip

        # Internal residual routing starts after register tokens are inserted.
        self.routing_start = (
            in_context_start if in_context_len > 0 else 0
        )

        # Maximum modulation is ±20%.
        self.max_routing_strength = 0.2

        # sigmoid(-3) * 0.2 ≈ 0.0095.
        # Therefore the initial model is almost identical to the original JiT.
        self.routing_logit = nn.Parameter(
            torch.tensor(-3.0)
        )

        self.t_embedder = TimestepEmbedder(hidden_size)
        if self.use_observation_branch:
            self.state_embedder = DynamicBottleneckPatchEmbed(
                patch_size,
                3,
                bottleneck_dim,
                hidden_size,
                bias=True,
            )
            self.obs_embedder = DynamicBottleneckPatchEmbed(
                patch_size,
                3,
                bottleneck_dim,
                hidden_size,
                bias=True,
            )
            self.input_fusion = nn.Linear(hidden_size * 2, hidden_size)
            self.x_embedder = self.state_embedder
        else:
            self.state_embedder = None
            self.obs_embedder = None
            self.input_fusion = None
            self.x_embedder = DynamicBottleneckPatchEmbed(
                patch_size,
                3,
                bottleneck_dim,
                hidden_size,
                bias=True,
            )

        if in_context_len > 0:
            self.register_tokens = nn.Parameter(
                torch.zeros(1, in_context_len, hidden_size)
            )
            nn.init.normal_(self.register_tokens, std=0.02)
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, in_context_len, hidden_size)
            )
            nn.init.normal_(self.in_context_posemb, std=0.02)
        else:
            self.register_tokens = None
            self.in_context_posemb = None

        if self.use_shallow_skip:
            self.shallow_skip = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.shallow_skip = None

        self.feat_rope = DynamicVisionRoPE(hidden_size // num_heads)
        self.blocks = nn.ModuleList(
            [
                DynamicJiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=(
                        attn_drop
                        if depth // 4 * 3 > index >= depth // 4
                        else 0.0
                    ),
                    proj_drop=(
                        proj_drop
                        if depth // 4 * 3 > index >= depth // 4
                        else 0.0
                    ),
                )
                for index in range(depth)
            ]
        )
        self.final_layer = DynamicFinalLayer(
            hidden_size,
            patch_size,
            self.out_channels,
        )
        self._position_cache = {}
        self.initialize_weights()

    @staticmethod
    def _axis_sincos(positions, dim):
        omega = torch.arange(
            dim // 2,
            device=positions.device,
            dtype=torch.float64,
        )
        omega = 1.0 / (10000 ** (omega / (dim / 2.0)))
        values = positions.reshape(-1).double()[:, None] * omega[None]
        return torch.cat([values.sin(), values.cos()], dim=1)

    def _position_embedding(self, grid_size, device, dtype):
        grid_h, grid_w = grid_size
        key = (grid_h, grid_w, device.type, device.index, dtype)
        cached = self._position_cache.get(key)
        if cached is not None:
            return cached
        rows, cols = torch.meshgrid(
            torch.arange(grid_h, device=device),
            torch.arange(grid_w, device=device),
            indexing="ij",
        )
        half = self.hidden_size // 2
        embedding = torch.cat(
            [
                self._axis_sincos(cols, half),
                self._axis_sincos(rows, half),
            ],
            dim=1,
        )
        embedding = embedding.to(dtype=dtype).unsqueeze(0)
        self._position_cache[key] = embedding
        return embedding

    def _init_patch_embed(self, embedder):
        weight = embedder.proj1.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        weight = embedder.proj2.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.constant_(embedder.proj2.bias, 0)

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        if self.use_observation_branch:
            self._init_patch_embed(self.state_embedder)
            self._init_patch_embed(self.obs_embedder)
            nn.init.xavier_uniform_(self.input_fusion.weight)
            nn.init.constant_(self.input_fusion.bias, 0)
        else:
            self._init_patch_embed(self.x_embedder)
        if self.shallow_skip is not None:
            nn.init.zeros_(self.shallow_skip.weight)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, tokens, grid_size):
        grid_h, grid_w = grid_size
        batch = tokens.shape[0]
        patch = self.patch_size
        channels = self.out_channels
        tokens = tokens.reshape(
            batch,
            grid_h,
            grid_w,
            patch,
            patch,
            channels,
        )
        tokens = torch.einsum("nhwpqc->nchpwq", tokens)
        return tokens.reshape(
            batch,
            channels,
            grid_h * patch,
            grid_w * patch,
        )

    def forward(
        self,
        state,
        t,
        observation=None,
        return_adaptive_info=False,
    ):
        condition = self.t_embedder(t)

        if self.use_observation_branch:
            if observation is None:
                raise ValueError(
                    "observation is required for dual-branch DynamicAIOJiT"
                )

            state_tokens, grid_size = self.state_embedder(state)
            obs_tokens, obs_grid = self.obs_embedder(observation)

            if obs_grid != grid_size:
                raise ValueError(
                    f"State grid {grid_size} != observation grid {obs_grid}"
                )

            tokens = self.input_fusion(
                torch.cat([state_tokens, obs_tokens], dim=-1)
            )
        else:
            tokens, grid_size = self.x_embedder(state)

        tokens = tokens + self._position_embedding(
            grid_size,
            tokens.device,
            tokens.dtype,
        )

        shallow = tokens if self.shallow_skip is not None else None

        prefix_tokens = 0
        routing_maps = []

        for index, block in enumerate(self.blocks):
            if (
                self.in_context_len > 0
                and index == self.in_context_start
            ):
                registers = self.register_tokens.expand(
                    tokens.size(0),
                    -1,
                    -1,
                )
                tokens = torch.cat(
                    [
                        registers + self.in_context_posemb,
                        tokens,
                    ],
                    dim=1,
                )
                prefix_tokens = self.in_context_len

            tokens_before = tokens

            candidate = block(
                tokens_before,
                condition,
                self.feat_rope,
                grid_size,
                prefix_tokens,
            )

            if index < self.routing_start:
                tokens = candidate
                continue

            old_patch = tokens_before[:, prefix_tokens:]
            candidate_patch = candidate[:, prefix_tokens:]

            # The actual update produced by this JiT block.
            patch_delta = candidate_patch - old_patch

            # Calculate in float32 for numerical stability under bfloat16.
            update_score = (
                patch_delta.float()
                .square()
                .mean(dim=-1, keepdim=True)
                .add(1.0e-8)
                .sqrt()
            )

            # Normalize inside each image.
            # This removes global scale differences between haze/noise/rain.
            score_mean = update_score.mean(
                dim=1,
                keepdim=True,
            )
            score_std = update_score.std(
                dim=1,
                keepdim=True,
                unbiased=False,
            ).clamp_min(1.0e-4)

            normalized_score = (
                update_score - score_mean
            ) / score_std

            internal_mask = torch.sigmoid(
                normalized_score
            ).to(dtype=patch_delta.dtype)

            # Prevent the network from artificially enlarging delta
            # solely to manipulate the mask.
            internal_mask = internal_mask.detach()

            routing_strength = (
                self.max_routing_strength
                * torch.sigmoid(self.routing_logit)
            ).to(dtype=patch_delta.dtype)

            routing_scale = 1.0 + routing_strength * (
                2.0 * internal_mask - 1.0
            )

            routed_patch = (
                old_patch
                + routing_scale * patch_delta
            )

            # Register tokens always receive the complete block update.
            if prefix_tokens > 0:
                tokens = torch.cat(
                    [
                        candidate[:, :prefix_tokens],
                        routed_patch,
                    ],
                    dim=1,
                )
            else:
                tokens = routed_patch

            routing_maps.append(internal_mask)

        if prefix_tokens > 0:
            tokens = tokens[:, prefix_tokens:]

        if shallow is not None:
            tokens = tokens + self.shallow_skip(shallow)

        if routing_maps:
            difficulty = torch.stack(
                routing_maps,
                dim=0,
            ).mean(dim=0)

            grid_h, grid_w = grid_size
            difficulty_map = difficulty.transpose(
                1,
                2,
            ).reshape(
                tokens.shape[0],
                1,
                grid_h,
                grid_w,
            )
        else:
            difficulty_map = None

        output = self.unpatchify(
            self.final_layer(tokens, condition),
            grid_size,
        )

        if not return_adaptive_info:
            return output

        return output, {
            "difficulty_map": difficulty_map,
            "routing_strength": routing_strength.detach(),
            "difficulty_mean": difficulty.mean().detach(),
            "difficulty_std": (
                difficulty.std(unbiased=False).detach()
            ),
        }


def DynamicAIOJiT_B_16(**kwargs):
    bottleneck_dim = kwargs.pop("bottleneck_dim", 128)
    return DynamicAIOJiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=bottleneck_dim,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def DynamicAIOJiT_L_16(**kwargs):
    bottleneck_dim = kwargs.pop("bottleneck_dim", 128)
    return DynamicAIOJiT(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=bottleneck_dim,
        in_context_len=32,
        in_context_start=8,
        patch_size=16,
        **kwargs,
    )


DynamicAIOJiT_models = {
    "JiT-B/16": DynamicAIOJiT_B_16,
    "JiT-L/16": DynamicAIOJiT_L_16,
}

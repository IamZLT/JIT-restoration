"""Dynamic-resolution JiT.

This module is intentionally separate from ``model_jit.py``.  It keeps the
same learnable parameter layout while generating positional encodings and
2-D RoPE from the runtime patch grid.  Inputs may be rectangular, but their
height and width must be divisible by the patch size.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_jit import LabelEmbedder, SwiGLUFFN, TimestepEmbedder, modulate
from util.model_util import RMSNorm, rotate_half


class DynamicBottleneckPatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, pca_dim, embed_dim, bias=True):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj1 = nn.Conv2d(
            in_chans,
            pca_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        height, width = x.shape[-2:]
        patch_h, patch_w = self.patch_size
        if height % patch_h or width % patch_w:
            raise ValueError(
                f"Input {height}x{width} must be divisible by "
                f"patch size {patch_h}x{patch_w}"
            )
        features = self.proj2(self.proj1(x))
        grid_size = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        return tokens, grid_size


class DynamicVisionRoPE(nn.Module):
    """Runtime 2-D rotary embedding with a small device/dtype cache."""

    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        if head_dim % 4:
            raise ValueError("Attention head dimension must be divisible by 4")
        axis_dim = head_dim // 2
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, axis_dim, 2, dtype=torch.float32)
                / axis_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = {}

    def _cos_sin(self, grid_h, grid_w, prefix_tokens, device, dtype):
        key = (grid_h, grid_w, prefix_tokens, device.type, device.index, dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        inv_freq = self.inv_freq.to(device=device)
        rows = torch.arange(grid_h, device=device, dtype=torch.float32)
        cols = torch.arange(grid_w, device=device, dtype=torch.float32)
        row_freq = torch.outer(rows, inv_freq).repeat_interleave(2, dim=-1)
        col_freq = torch.outer(cols, inv_freq).repeat_interleave(2, dim=-1)
        row_freq = row_freq[:, None, :].expand(grid_h, grid_w, -1)
        col_freq = col_freq[None, :, :].expand(grid_h, grid_w, -1)
        frequencies = torch.cat([row_freq, col_freq], dim=-1).reshape(
            grid_h * grid_w, -1
        )
        cos = frequencies.cos()
        sin = frequencies.sin()
        if prefix_tokens:
            cos = torch.cat(
                [torch.ones(prefix_tokens, cos.size(1), device=device), cos],
                dim=0,
            )
            sin = torch.cat(
                [torch.zeros(prefix_tokens, sin.size(1), device=device), sin],
                dim=0,
            )
        result = (cos.to(dtype=dtype), sin.to(dtype=dtype))
        self._cache[key] = result
        return result

    def forward(self, tensor, grid_size, prefix_tokens=0):
        cos, sin = self._cos_sin(
            grid_size[0],
            grid_size[1],
            prefix_tokens,
            tensor.device,
            tensor.dtype,
        )
        return tensor * cos + rotate_half(tensor) * sin


class DynamicAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope, grid_size, prefix_tokens):
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            length,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = rope(self.q_norm(query), grid_size, prefix_tokens)
        key = rope(self.k_norm(key), grid_size, prefix_tokens)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(output))


class DynamicJiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = DynamicAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(
            hidden_size,
            int(hidden_size * mlp_ratio),
            drop=proj_drop,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, condition, rope, grid_size, prefix_tokens):
        modulation = self.adaLN_modulation(condition).chunk(6, dim=-1)
        shift_attn, scale_attn, gate_attn = modulation[:3]
        shift_mlp, scale_mlp, gate_mlp = modulation[3:]
        attended = self.attn(
            modulate(self.norm1(x), shift_attn, scale_attn),
            rope,
            grid_size,
            prefix_tokens,
        )
        x = x + gate_attn.unsqueeze(1) * attended
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class DynamicFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(
            modulate(self.norm_final(x), shift, scale)
        )


class DynamicJiT(nn.Module):
    def __init__(
        self,
        input_size=512,
        patch_size=16,
        in_channels=3,
        out_channels=None,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
    ):
        super().__init__()
        del input_size  # Kept in the signature for checkpoint argument parity.
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = DynamicBottleneckPatchEmbed(
            patch_size,
            in_channels,
            bottleneck_dim,
            hidden_size,
            bias=True,
        )
        if in_context_len > 0:
            self.in_context_posemb = nn.Parameter(
                torch.zeros(1, in_context_len, hidden_size)
            )
            nn.init.normal_(self.in_context_posemb, std=0.02)

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

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        weight = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        weight = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
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

    def forward(self, x, t, y):
        time_embedding = self.t_embedder(t)
        label_embedding = self.y_embedder(y)
        condition = time_embedding + label_embedding

        x, grid_size = self.x_embedder(x)
        x = x + self._position_embedding(grid_size, x.device, x.dtype)
        prefix_tokens = 0
        for index, block in enumerate(self.blocks):
            if self.in_context_len > 0 and index == self.in_context_start:
                context = label_embedding.unsqueeze(1).expand(
                    -1,
                    self.in_context_len,
                    -1,
                )
                x = torch.cat(
                    [context + self.in_context_posemb, x],
                    dim=1,
                )
                prefix_tokens = self.in_context_len
            x = block(
                x,
                condition,
                self.feat_rope,
                grid_size,
                prefix_tokens,
            )

        if self.in_context_len > 0:
            x = x[:, self.in_context_len :]
        return self.unpatchify(self.final_layer(x, condition), grid_size)


def DynamicJiT_B_16(**kwargs):
    return DynamicJiT(
        depth=12,
        hidden_size=768,
        num_heads=12,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size=16,
        **kwargs,
    )


def DynamicJiT_L_16(**kwargs):
    return DynamicJiT(
        depth=24,
        hidden_size=1024,
        num_heads=16,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        patch_size=16,
        **kwargs,
    )


def DynamicJiT_H_16(**kwargs):
    return DynamicJiT(
        depth=32,
        hidden_size=1280,
        num_heads=16,
        bottleneck_dim=256,
        in_context_len=32,
        in_context_start=10,
        patch_size=16,
        **kwargs,
    )


DynamicJiT_models = {
    "JiT-B/16": DynamicJiT_B_16,
    "JiT-L/16": DynamicJiT_L_16,
    "JiT-H/16": DynamicJiT_H_16,
}

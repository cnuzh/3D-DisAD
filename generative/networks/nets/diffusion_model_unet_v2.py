# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =========================================================================
# Adapted from https://github.com/huggingface/diffusers
# which has the following license:
# https://github.com/huggingface/diffusers/blob/main/LICENSE
#
# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import annotations

import importlib.util
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from monai.networks.blocks import Convolution, MLPBlock
from monai.networks.layers.factories import Pool
from monai.utils import ensure_tuple_rep
from torch import nn
import contextlib
import numpy as np
from typing import NamedTuple
import itertools

# To install xformers, use pip install xformers==0.0.16rc401
if importlib.util.find_spec("xformers") is not None:
    import xformers
    import xformers.ops

    has_xformers = True
else:
    xformers = None
    has_xformers = False

# TODO: Use MONAI's optional_import
# from monai.utils import optional_import
# xformers, has_xformers = optional_import("xformers.ops", name="xformers")

__all__ = ["DiffusionModelUNet", "Return", "Return_grad", "Return_grad_full"]


class Return(NamedTuple):
    pred: torch.Tensor


class Return_grad(NamedTuple):
    pred: torch.Tensor
    out_grad: torch.Tensor


class Return_grad_full(NamedTuple):
    pred: torch.Tensor
    out_grad: torch.Tensor
    sub_grad: torch.Tensor


def return_wrap(inp, coef):
    if isinstance(inp, Return):
        return inp.pred
    elif isinstance(inp, Return_grad) or isinstance(inp, Return_grad_full):
        # return inp.out_grad
        return inp.pred + coef * inp.out_grad


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class CrossAttention(nn.Module):
    """
    A cross attention layer.

    Args:
        query_dim: number of channels in the query.
        cross_attention_dim: number of channels in the context.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each head.
        dropout: dropout probability to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
            self,
            query_dim: int,
            cross_attention_dim: int | None = None,
            num_attention_heads: int = 8,
            num_head_channels: int = 64,
            dropout: float = 0.0,
            upcast_attention: bool = False,
            use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        inner_dim = num_head_channels * num_attention_heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = 1 / math.sqrt(num_head_channels)
        self.num_heads = num_attention_heads

        self.upcast_attention = upcast_attention

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide hidden state dimension to the multiple attention heads and reshape their input as instances in the batch.
        """
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the output of the attention heads back into the hidden state dimension."""
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def _memory_efficient_attention_xformers(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype=dtype)

        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        query = self.to_q(x)
        context = context if context is not None else x
        key = self.to_k(context)
        value = self.to_v(context)

        # Multi-Head Attention
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)

        return self.to_out(x)


class BasicTransformerBlock(nn.Module):
    """
    A basic Transformer block.

    Args:
        num_channels: number of channels in the input and output.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each attention head.
        dropout: dropout probability to use.
        cross_attention_dim: size of the context vector for cross attention.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
            self,
            num_channels: int,
            num_attention_heads: int,
            num_head_channels: int,
            dropout: float = 0.0,
            cross_attention_dim: int | None = None,
            upcast_attention: bool = False,
            use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=num_channels,
            num_attention_heads=num_attention_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )  # is a self-attention
        self.ff = MLPBlock(hidden_size=num_channels, mlp_dim=num_channels * 4, act="GEGLU", dropout_rate=dropout)
        self.attn2 = CrossAttention(
            query_dim=num_channels,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )  # is a self-attention if context is None
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        self.norm3 = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # 1. Self-Attention
        x = self.attn1(self.norm1(x)) + x

        # 2. Cross-Attention
        x = self.attn2(self.norm2(x), context=context) + x

        # 3. Feed-forward
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of channels in the input and output.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each attention head.
        num_layers: number of layers of Transformer blocks to use.
        dropout: dropout probability to use.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            num_attention_heads: int,
            num_head_channels: int,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            cross_attention_dim: int | None = None,
            upcast_attention: bool = False,
            use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        inner_dim = num_attention_heads * num_head_channels

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)

        self.proj_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=inner_dim,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    num_channels=inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_head_channels=num_head_channels,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=inner_dim,
                out_channels=in_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # note: if no context is given, cross-attention defaults to self-attention
        batch = channel = height = width = depth = -1
        if self.spatial_dims == 1:
            batch, channel, length = x.shape
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        residual = x
        x = self.norm(x)
        x = self.proj_in(x)

        inner_dim = x.shape[1]
        if self.spatial_dims == 2:
            x = x.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        if self.spatial_dims == 3:
            x = x.permute(0, 2, 3, 4, 1).reshape(batch, height * width * depth, inner_dim)

        for block in self.transformer_blocks:
            x = block(x, context=context)

        if self.spatial_dims == 2:
            x = x.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        if self.spatial_dims == 3:
            x = x.reshape(batch, height, width, depth, inner_dim).permute(0, 4, 1, 2, 3).contiguous()

        x = self.proj_out(x)
        return x + residual


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other. Uses three q, k, v linear layers to
    compute attention.

    Args:
        spatial_dims: number of spatial dimensions.
        num_channels: number of input channels.
        num_head_channels: number of channels in each attention head.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon value to use for the normalisation.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
            self,
            spatial_dims: int,
            num_channels: int,
            num_head_channels: int | None = None,
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels

        self.num_heads = num_channels // num_head_channels if num_head_channels is not None else 1
        self.scale = 1 / math.sqrt(num_channels / self.num_heads)

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)

        self.to_q = nn.Linear(num_channels, num_channels)
        self.to_k = nn.Linear(num_channels, num_channels)
        self.to_v = nn.Linear(num_channels, num_channels)

        self.proj_attn = nn.Linear(num_channels, num_channels)

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def _memory_efficient_attention_xformers(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        # norm
        x = self.norm(x)

        if self.spatial_dims == 2:
            x = x.view(batch, channel, height * width).transpose(1, 2)
        if self.spatial_dims == 3:
            x = x.view(batch, channel, height * width * depth).transpose(1, 2)

        # proj to q, k, v
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Multi-Head Attention
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)

        if self.spatial_dims == 2:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width)
        if self.spatial_dims == 3:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width, depth)

        return x + residual


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings following the implementation in Ho et al. "Denoising Diffusion Probabilistic
    Models" https://arxiv.org/abs/2006.11239.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        embedding_dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    """
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32,
                                                    device=timesteps.device)
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding


class Downsample(nn.Module):
    """
    Downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions.
        num_channels: number of input channels.
        use_conv: if True uses Convolution instead of Pool average to perform downsampling. In case that use_conv is
            False, the number of output channels must be the same as the number of input channels.
        out_channels: number of output channels.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension.
    """

    def __init__(
            self, spatial_dims: int, num_channels: int, use_conv: bool, out_channels: int | None = None,
            padding: int = 1
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.op = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=2,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            if self.num_channels != self.out_channels:
                raise ValueError("num_channels and out_channels must be equal when use_conv=False")
            self.op = Pool[Pool.AVG, spatial_dims](kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        del emb
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input number of channels ({x.shape[1]}) is not equal to expected number of channels "
                f"({self.num_channels})"
            )
        return self.op(x)


class Upsample(nn.Module):
    """
    Upsampling layer with an optional convolution.

    Args:
        spatial_dims: number of spatial dimensions.
        num_channels: number of input channels.
        use_conv: if True uses Convolution instead of Pool average to perform downsampling.
        out_channels: number of output channels.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points for each
            dimension.
    """

    def __init__(
            self, spatial_dims: int, num_channels: int, use_conv: bool, out_channels: int | None = None,
            padding: int = 1
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels or num_channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.num_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=padding,
                conv_only=True,
            )
        else:
            self.conv = None

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        del emb
        if x.shape[1] != self.num_channels:
            raise ValueError("Input channels should be equal to num_channels")

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        if self.use_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        temb_channels: number of timestep embedding  channels.
        out_channels: number of output channels.
        up: if True, performs upsampling.
        down: if True, performs downsampling.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            temb_channels: int,
            out_channels: int | None = None,
            up: bool = False,
            down: bool = False,
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            use_scale_shift_norm: bool = False,
            with_resblock_cond: bool = False,
            cond_emb_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.channels = in_channels
        self.emb_channels = temb_channels
        self.out_channels = out_channels or in_channels
        self.up = up
        self.down = down
        self.use_scale_shift_norm = use_scale_shift_norm
        self.with_resblock_cond = with_resblock_cond

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.nonlinearity = nn.SiLU()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample(spatial_dims, in_channels, use_conv=False)
        elif down:
            self.downsample = Downsample(spatial_dims, in_channels, use_conv=False)

        self.time_emb_proj = nn.Linear(
            temb_channels,
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )

        if self.with_resblock_cond:
            self.cond_emb_proj = nn.Linear(
                cond_emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels
            )

        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=self.out_channels, eps=norm_eps, affine=True)
        self.conv2 = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            if h.shape[0] >= 64:
                x = x.contiguous()
                h = h.contiguous()
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        temb = self.time_emb_proj(self.nonlinearity(emb))
        while len(temb.shape) < len(h.shape):
            temb = temb[..., None]

        if self.with_resblock_cond:
            cemb = self.cond_emb_proj(self.nonlinearity(context))
            while len(cemb.shape) < len(h.shape):
                cemb = cemb[..., None]

        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(temb, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift

            if self.with_resblock_cond:
                cond_scale, cond_shift = torch.chunk(cemb, 2, dim=1)
                h = h * (1 + cond_scale) + cond_shift

            h = self.nonlinearity(h)
            h = self.conv2(h)
        else:
            if self.with_resblock_cond:
                h = h + temb + cemb
            else:
                h = h + temb

            h = self.norm2(h)
            h = self.nonlinearity(h)
            h = self.conv2(h)

        return self.skip_connection(x) + h


class DownBlock(nn.Module):
    """
    Unet's down block containing resnet and downsamplers blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            num_res_blocks: int = 1,
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            use_scale_shift_norm: bool = False,
            add_downsample: bool = True,
            resblock_updown: bool = False,
            downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                    down=True,
                )
            else:
                self.downsampler = Downsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
            self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnDownBlock(nn.Module):
    """
    Unet's down block containing resnet, downsamplers and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding  channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
        num_head_channels: number of channels in each attention head.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            num_res_blocks: int = 1,
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            use_scale_shift_norm: bool = False,
            add_downsample: bool = True,
            resblock_updown: bool = False,
            downsample_padding: int = 1,
            num_head_channels: int = 1,
            use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )
            attentions.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                    down=True,
                )
            else:
                self.downsampler = Downsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
            self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del context
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock(nn.Module):
    """
    Unet's down block containing resnet, downsamplers and cross-attention blocks.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_downsample: if True add downsample block.
        resblock_updown: if True use residual blocks for downsampling.
        downsample_padding: padding used in the downsampling block.
        num_head_channels: number of channels in each attention head.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        dropout_cattn: if different from zero, this will be the dropout value for the cross-attention layers
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            num_res_blocks: int = 1,
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            use_scale_shift_norm: bool = False,
            add_downsample: bool = True,
            resblock_updown: bool = False,
            downsample_padding: int = 1,
            num_head_channels: int = 1,
            transformer_num_layers: int = 1,
            cross_attention_dim: int | None = None,
            upcast_attention: bool = False,
            use_flash_attention: bool = False,
            dropout_cattn: float = 0.0,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )

            attentions.append(
                SpatialTransformer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    num_layers=transformer_num_layers,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                    dropout=dropout_cattn,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            if resblock_updown:
                self.downsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                    down=True,
                )
            else:
                self.downsampler = Downsample(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
        else:
            self.downsampler = None

    def forward(
            self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=context)
            output_states.append(hidden_states)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states, temb)
            output_states.append(hidden_states)

        return hidden_states, output_states


class AttnMidBlock(nn.Module):
    """
    Unet's mid block containing resnet and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        temb_channels: number of timestep embedding channels.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        num_head_channels: number of channels in each attention head.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            temb_channels: int,
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            use_scale_shift_norm: bool = False,
            num_head_channels: int = 1,
            use_flash_attention: bool = False,
            with_resblock_cond: bool = False,
            cond_emb_channels: int | None = None,

    ) -> None:
        super().__init__()
        self.attention = None
        self.with_resblock_cond = with_resblock_cond

        self.resnet_1 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            with_resblock_cond=with_resblock_cond,
            cond_emb_channels=cond_emb_channels,
        )
        self.attention = AttentionBlock(
            spatial_dims=spatial_dims,
            num_channels=in_channels,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_flash_attention=use_flash_attention,
        )

        self.resnet_2 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            with_resblock_cond=with_resblock_cond,
            cond_emb_channels=cond_emb_channels,
        )

    def forward(
            self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:

        if self.with_resblock_cond:
            hidden_states = self.resnet_1(hidden_states, temb, context=context)
            hidden_states = self.attention(hidden_states)
            hidden_states = self.resnet_2(hidden_states, temb, context=context)
        else:
            del context
            hidden_states = self.resnet_1(hidden_states, temb)
            hidden_states = self.attention(hidden_states)
            hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states


class CrossAttnMidBlock(nn.Module):
    """
    Unet's mid block containing resnet and cross-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        temb_channels: number of timestep embedding channels
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        num_head_channels: number of channels in each attention head.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            temb_channels: int,
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            use_scale_shift_norm: bool = False,
            num_head_channels: int = 1,
            transformer_num_layers: int = 1,
            cross_attention_dim: int | None = None,
            upcast_attention: bool = False,
            use_flash_attention: bool = False,
            dropout_cattn: float = 0.0,
            with_resblock_cond: bool = False,
            cond_emb_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.attention = None
        self.use_scale_shift_norm = use_scale_shift_norm
        self.with_resblock_cond = with_resblock_cond

        self.resnet_1 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            with_resblock_cond=with_resblock_cond,
            cond_emb_channels=cond_emb_channels,
        )
        self.attention = SpatialTransformer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_attention_heads=in_channels // num_head_channels,
            num_head_channels=num_head_channels,
            num_layers=transformer_num_layers,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout=dropout_cattn,
        )
        self.resnet_2 = ResnetBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            with_resblock_cond=with_resblock_cond,
            cond_emb_channels=cond_emb_channels,
        )

    def forward(
            self, hidden_states: torch.Tensor, temb: torch.Tensor, context: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if self.with_resblock_cond:
            hidden_states = self.resnet_1(hidden_states, temb, context=context)
            hidden_states = self.attention(hidden_states, context=context)
            hidden_states = self.resnet_2(hidden_states, temb, context=context)
        else:
            hidden_states = self.resnet_1(hidden_states, temb)
            hidden_states = self.attention(hidden_states, context=context)
            hidden_states = self.resnet_2(hidden_states, temb)

        return hidden_states


class UpBlock(nn.Module):
    """
    Unet's up block containing resnet and upsamplers blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            temb_channels: int,
            num_res_blocks: int = 1,
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            use_scale_shift_norm: bool = False,
            add_upsample: bool = True,
            resblock_updown: bool = False,
            with_resblock_cond: bool = False,
            cond_emb_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown
        self.with_resblock_cond = with_resblock_cond

        resnets = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                    with_resblock_cond=with_resblock_cond,
                    cond_emb_channels=cond_emb_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                    with_resblock_cond=with_resblock_cond,
                    cond_emb_channels=cond_emb_channels,
                    up=True,
                )
            else:
                self.upsampler = Upsample(
                    spatial_dims=spatial_dims, num_channels=out_channels, use_conv=True, out_channels=out_channels
                )
        else:
            self.upsampler = None

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_list: list[torch.Tensor],
            temb: torch.Tensor,
            context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.with_resblock_cond:
            del context

        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.with_resblock_cond:
                hidden_states = resnet(hidden_states, temb, context=context)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsampler is not None:
            if self.resblock_updown and self.with_resblock_cond:
                hidden_states = self.upsampler(hidden_states, temb, context=context)
            else:
                hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


class AttnUpBlock(nn.Module):
    """
    Unet's up block containing resnet, upsamplers, and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
        num_head_channels: number of channels in each attention head.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            temb_channels: int,
            num_res_blocks: int = 1,
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            use_scale_shift_norm: bool = False,
            add_upsample: bool = True,
            resblock_updown: bool = False,
            num_head_channels: int = 1,
            use_flash_attention: bool = False,
            with_resblock_cond: bool = False,
            cond_emb_channels: int | None = None,
    ) -> None:
        super().__init__()

        self.resblock_updown = resblock_updown
        self.with_resblock_cond = with_resblock_cond

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                    with_resblock_cond=with_resblock_cond,
                    cond_emb_channels=cond_emb_channels,
                )
            )
            attentions.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=out_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                    with_resblock_cond=with_resblock_cond,
                    cond_emb_channels=cond_emb_channels,
                    up=True,
                )
            else:
                self.upsampler = Upsample(
                    spatial_dims=spatial_dims, num_channels=out_channels, use_conv=True, out_channels=out_channels
                )
        else:
            self.upsampler = None

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_list: list[torch.Tensor],
            temb: torch.Tensor,
            context: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if not self.with_resblock_cond:
            del context

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.with_resblock_cond:
                hidden_states = resnet(hidden_states, temb, context=context)
            else:
                hidden_states = resnet(hidden_states, temb)

            hidden_states = attn(hidden_states)

        if self.upsampler is not None:
            if self.resblock_updown and self.with_resblock_cond:
                hidden_states = self.upsampler(hidden_states, temb, context=context)
            else:
                hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


class CrossAttnUpBlock(nn.Module):
    """
    Unet's up block containing resnet, upsamplers, and self-attention blocks.

    Args:
        spatial_dims: The number of spatial dimensions.
        in_channels: number of input channels.
        prev_output_channel: number of channels from residual connection.
        out_channels: number of output channels.
        temb_channels: number of timestep embedding channels.
        num_res_blocks: number of residual blocks.
        norm_num_groups: number of groups for the group normalization.
        norm_eps: epsilon for the group normalization.
        add_upsample: if True add downsample block.
        resblock_updown: if True use residual blocks for upsampling.
        num_head_channels: number of channels in each attention head.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        dropout_cattn: if different from zero, this will be the dropout value for the cross-attention layers
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            temb_channels: int,
            num_res_blocks: int = 1,
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            use_scale_shift_norm: bool = False,
            add_upsample: bool = True,
            resblock_updown: bool = False,
            num_head_channels: int = 1,
            transformer_num_layers: int = 1,
            cross_attention_dim: int | None = None,
            upcast_attention: bool = False,
            use_flash_attention: bool = False,
            dropout_cattn: float = 0.0,
            with_resblock_cond: bool = False,
            cond_emb_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.resblock_updown = resblock_updown
        self.with_resblock_cond = with_resblock_cond

        resnets = []
        attentions = []

        for i in range(num_res_blocks):
            res_skip_channels = in_channels if (i == num_res_blocks - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                    with_resblock_cond=with_resblock_cond,
                    cond_emb_channels=cond_emb_channels,
                )
            )
            attentions.append(
                SpatialTransformer(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    num_attention_heads=out_channels // num_head_channels,
                    num_head_channels=num_head_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    num_layers=transformer_num_layers,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                    dropout=dropout_cattn,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            if resblock_updown:
                self.upsampler = ResnetBlock(
                    spatial_dims=spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                    with_resblock_cond=with_resblock_cond,
                    cond_emb_channels=cond_emb_channels,
                    up=True,
                )
            else:
                self.upsampler = Upsample(
                    spatial_dims=spatial_dims, num_channels=out_channels, use_conv=True, out_channels=out_channels
                )
        else:
            self.upsampler = None

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_list: list[torch.Tensor],
            temb: torch.Tensor,
            context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_list[-1]
            res_hidden_states_list = res_hidden_states_list[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.with_resblock_cond:
                hidden_states = resnet(hidden_states, temb, context=context)
            else:
                hidden_states = resnet(hidden_states, temb)

            hidden_states = attn(hidden_states, context=context)

        if self.upsampler is not None:
            if self.resblock_updown and self.with_resblock_cond:
                hidden_states = self.upsampler(hidden_states, temb, context=context)
            else:
                hidden_states = self.upsampler(hidden_states, temb)

        return hidden_states


def get_down_block(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int,
        norm_num_groups: int,
        norm_eps: float,
        use_scale_shift_norm: bool,
        add_downsample: bool,
        resblock_updown: bool,
        with_attn: bool,
        with_cross_attn: bool,
        num_head_channels: int,
        transformer_num_layers: int,
        cross_attention_dim: int | None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
) -> nn.Module:
    if with_attn:
        return AttnDownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
        )
    elif with_cross_attn:
        return CrossAttnDownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )
    else:
        return DownBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            add_downsample=add_downsample,
            resblock_updown=resblock_updown,
        )


def get_mid_block(
        spatial_dims: int,
        in_channels: int,
        temb_channels: int,
        norm_num_groups: int,
        norm_eps: float,
        use_scale_shift_norm: bool,
        with_conditioning: bool,
        num_head_channels: int,
        transformer_num_layers: int,
        cross_attention_dim: int | None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
        with_resblock_cond: bool = False,
        cond_emb_channels: int | None = None,

) -> nn.Module:
    if with_conditioning:
        return CrossAttnMidBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
            with_resblock_cond=with_resblock_cond,
            cond_emb_channels=cond_emb_channels,
        )
    else:
        return AttnMidBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            temb_channels=temb_channels,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
            with_resblock_cond=with_resblock_cond,
            cond_emb_channels=cond_emb_channels,
        )


def get_up_block(
        spatial_dims: int,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_res_blocks: int,
        norm_num_groups: int,
        norm_eps: float,
        use_scale_shift_norm: bool,
        add_upsample: bool,
        resblock_updown: bool,
        with_attn: bool,
        with_cross_attn: bool,
        num_head_channels: int,
        transformer_num_layers: int,
        cross_attention_dim: int | None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
        with_resblock_cond: bool = False,
        cond_emb_channels: int | None = None,
) -> nn.Module:
    if with_attn:
        return AttnUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            use_flash_attention=use_flash_attention,
            with_resblock_cond=with_resblock_cond,
            cond_emb_channels=cond_emb_channels,
        )
    elif with_cross_attn:
        return CrossAttnUpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            num_head_channels=num_head_channels,
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
            with_resblock_cond=with_resblock_cond,
            cond_emb_channels=cond_emb_channels,
        )
    else:
        return UpBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            add_upsample=add_upsample,
            resblock_updown=resblock_updown,
            with_resblock_cond=with_resblock_cond,
            cond_emb_channels=cond_emb_channels,
        )


class DiffusionModelUNet(nn.Module):
    """
    Unet network with timestep embedding and attention mechanisms for conditioning based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResnetBlock) per level.
        num_channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        resblock_updown: if True use residual blocks for up/downsampling.
        num_head_channels: number of channels in each attention head.
        with_conditioning: if True add spatial transformers to perform conditioning.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        num_class_embeds: if specified (as an int), then this model will be class-conditional with `num_class_embeds`
        classes.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        dropout_cattn: if different from zero, this will be the dropout value for the cross-attention layers
        with_decoupling: if True, add decouple layers to the model.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
            num_channels: Sequence[int] = (32, 64, 64, 64),
            attention_levels: Sequence[bool] = (False, False, True, True),
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            use_scale_shift_norm: bool = False,
            resblock_updown: bool = False,
            num_head_channels: int | Sequence[int] = 8,
            with_conditioning: bool = False,
            transformer_num_layers: int = 1,
            cross_attention_dim: int | None = None,
            num_class_embeds: int | None = None,
            upcast_attention: bool = False,
            use_flash_attention: bool = False,
            dropout_cattn: float = 0.0,
            with_decoupling: bool = False,
            represent_dims: int | None = None,
            num_represent_embeds: int | None = None,
            separate_decoder: bool = False,
            orthogonal_emb: bool = False,
            ddpm_path: str | None = None,
    ) -> None:
        super().__init__()
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "DiffusionModelUNet expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "when using with_conditioning."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "DiffusionModelUNet expects with_conditioning=True when specifying the cross_attention_dim."
            )

        if with_decoupling is True and represent_dims is None:
            raise ValueError(
                "DiffusionModelUNet expects dimension of the representation conditioning (represent_dims) "
                "when using with_conditioning."
            )
        if with_decoupling is True and num_represent_embeds is None:
            raise ValueError(
                "DiffusionModelUNet expects number of the representation conditioning (num_represent_embeds) "
                "when using with_conditioning."
            )

        if dropout_cattn > 1.0 or dropout_cattn < 0.0:
            raise ValueError("Dropout cannot be negative or >1.0!")

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError("DiffusionModelUNet expects all num_channels being multiple of norm_num_groups")

        if len(num_channels) != len(attention_levels):
            raise ValueError("DiffusionModelUNet expects num_channels being same size of attention_levels")

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        if use_flash_attention and not has_xformers:
            raise ValueError("use_flash_attention is True but xformers is not installed.")

        if use_flash_attention is True and not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. Flash attention is only available for GPU."
            )

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning
        self.with_decoupling = with_decoupling
        self.represent_dims = represent_dims
        self.num_represent_embeds = num_represent_embeds
        self.separate_decoder = separate_decoder
        self.orthogonal_emb = orthogonal_emb
        self.ddpm_path = ddpm_path

        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # time
        time_embed_dim = num_channels[0] * 4
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels[0], time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )

        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        # decouple embedding
        if self.with_decoupling:
            if self.separate_decoder:
                self.represent_embed = nn.Linear(self.represent_dims, self.time_embed_dim)
                if self.orthogonal_emb:
                    emb_weight = torch.randn([self.time_embed_dim // 2, self.time_embed_dim // 2])
                    self.part_latents = nn.Embedding.from_pretrained(emb_weight, freeze=False)
                    self.part_emb = nn.Linear(self.time_embed_dim // 2, self.time_embed_dim)
                else:
                    self.part_latents = nn.Embedding(self.latent_unit, self.repre_emb_channels)
                    self.part_emb = nn.Linear(self.repre_emb_channels, self.time_embed_dim)
            else:
                self.represent_embed = nn.Linear(
                    self.represent_dims * self.num_represent_embeds, self.time_embed_dim
                )

        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                use_scale_shift_norm=use_scale_shift_norm,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )

            self.down_blocks.append(down_block)

        # mid
        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            temb_channels=time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            use_scale_shift_norm=use_scale_shift_norm,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )

        # shift mid
        if self.with_decoupling:
            self.shift_middle_block = get_mid_block(
                spatial_dims=spatial_dims,
                in_channels=num_channels[-1],
                temb_channels=time_embed_dim,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                use_scale_shift_norm=use_scale_shift_norm,
                with_conditioning=with_conditioning,
                num_head_channels=num_head_channels[-1],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
                with_resblock_cond=True,
                cond_emb_channels=time_embed_dim,
            )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(num_channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(num_channels) - 1)]

            is_final_block = i == len(num_channels) - 1

            up_block = get_up_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=reversed_num_res_blocks[i] + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                use_scale_shift_norm=use_scale_shift_norm,
                add_upsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(reversed_attention_levels[i] and not with_conditioning),
                with_cross_attn=(reversed_attention_levels[i] and with_conditioning),
                num_head_channels=reversed_num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )

            self.up_blocks.append(up_block)

        # shift up
        if self.with_decoupling:
            self.shift_up_blocks = nn.ModuleList([])
            output_channel = reversed_block_out_channels[0]
            for i in range(len(reversed_block_out_channels)):
                prev_output_channel = output_channel
                output_channel = reversed_block_out_channels[i]
                input_channel = reversed_block_out_channels[min(i + 1, len(num_channels) - 1)]

                is_final_block = i == len(num_channels) - 1

                shift_up_block = get_up_block(
                    spatial_dims=spatial_dims,
                    in_channels=input_channel,
                    prev_output_channel=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    num_res_blocks=reversed_num_res_blocks[i] + 1,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_scale_shift_norm=use_scale_shift_norm,
                    add_upsample=not is_final_block,
                    resblock_updown=resblock_updown,
                    with_attn=(reversed_attention_levels[i] and not with_conditioning),
                    with_cross_attn=(reversed_attention_levels[i] and with_conditioning),
                    num_head_channels=reversed_num_head_channels[i],
                    transformer_num_layers=transformer_num_layers,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                    use_flash_attention=use_flash_attention,
                    dropout_cattn=dropout_cattn,
                    with_resblock_cond=True,
                    cond_emb_channels=time_embed_dim,
                )

                self.shift_up_blocks.append(shift_up_block)

        # out
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

        # shift out
        if self.with_decoupling:
            self.shift_out = nn.Sequential(
                nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels[0], eps=norm_eps, affine=True),
                nn.SiLU(),
                # zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                ),
                # ),
            )
        self.init_model()

    def init_model(self):
        if self.with_decoupling and self.ddpm_path is not None:
            state_dict = torch.load(self.ddpm_path, map_location="cpu")["state_dict"]
            model_dict = {}
            for k, v in state_dict.items():
                model_dict[k.replace('net.', '')] = v
            self.load_state_dict(model_dict, strict=False)

            self.time_embed.eval()
            if self.num_class_embeds is not None:
                self.class_embedding.eval()
            self.conv_in.eval()
            self.down_blocks.eval()
            self.middle_block.eval()
            self.up_blocks.eval()
            self.out.eval()

            self.time_embed.requires_grad_(False)
            if self.num_class_embeds is not None:
                self.class_embedding.requires_grad_(False)
            self.conv_in.requires_grad_(False)
            self.down_blocks.requires_grad_(False)
            self.middle_block.requires_grad_(False)
            self.up_blocks.requires_grad_(False)
            self.out.requires_grad_(False)

    def train(self, mode=True):
        if self.with_decoupling:
            self.training = mode
            if self.separate_decoder:
                self.part_emb.train(mode)
                self.part_latents.train(mode)

            self.shift_middle_block.train(mode)
            self.shift_up_blocks.train(mode)
            self.shift_out.train(mode)
            self.represent_embed.train(mode)
        else:
            super().train(mode)

        return self

    def eval(self):
        if self.with_decoupling:
            self.training = False
            if self.separate_decoder:
                self.part_emb.eval()
                self.part_latents.eval()

            self.represent_embed.eval()

            self.shift_middle_block.eval()
            self.shift_up_blocks.eval()
            self.shift_out.eval()
        else:
            super().eval()

        return self

    def parameters(self, recurse: bool = True):
        if self.with_decoupling:
            params = list()
            params += [
                self.shift_middle_block.parameters(recurse),
                self.shift_up_blocks.parameters(recurse),
                self.shift_out.parameters(recurse),
                self.represent_embed.parameters(recurse),
            ]
            if self.separate_decoder:
                params.append(self.part_emb.parameters(recurse))
                params.append(self.part_latents.parameters(recurse))
            return itertools.chain(*params)

        return super().parameters(recurse)

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            context: torch.Tensor | None = None,
            represent: torch.Tensor | None = None,
            class_labels: torch.Tensor | None = None,
            down_block_additional_residuals: tuple[torch.Tensor] | None = None,
            mid_block_additional_residual: torch.Tensor | None = None,
            sampled_concept: np.ndarray | None = None,
            sampled_index: np.ndarray | None = None,
    ) -> torch.Tensor | Return_grad_full | Return_grad:
        """
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            context: context tensor (N, 1, ContextDim).
            represent: represent tensor (N, 1, RepresentDim).
            class_labels: context tensor (N, ).
            down_block_additional_residuals: additional residual tensors for down blocks (N, C, FeatureMapsDims).
            mid_block_additional_residual: additional residual tensor for mid block (N, C, FeatureMapsDims).
            sampled_index:
            sampled_concept:
        """
        with torch.no_grad() if self.with_decoupling else contextlib.nullcontext():
            # 1. time
            t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])

            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=x.dtype)
            emb = self.time_embed(t_emb)

            # 2. class
            if self.num_class_embeds is not None:
                if class_labels is None:
                    raise ValueError("class_labels should be provided when num_class_embeds > 0")
                class_emb = self.class_embedding(class_labels)
                class_emb = class_emb.to(dtype=x.dtype)
                emb = emb + class_emb

            # 3. initial convolution
            h = self.conv_in(x)

            # 4. down
            if context is not None and self.with_conditioning is False:
                raise ValueError("model should have with_conditioning = True if context is provided")
            down_block_res_samples: list[torch.Tensor] = [h]
            for downsample_block in self.down_blocks:
                h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
                for residual in res_samples:
                    down_block_res_samples.append(residual)

            # Additional residual conections for Controlnets
            if down_block_additional_residuals is not None:
                new_down_block_res_samples = ()
                for down_block_res_sample, down_block_additional_residual in zip(
                        down_block_res_samples, down_block_additional_residuals
                ):
                    down_block_res_sample = down_block_res_sample + down_block_additional_residual
                    new_down_block_res_samples += (down_block_res_sample,)

                down_block_res_samples = new_down_block_res_samples

        out_grad = None
        sub_grad = None
        if self.with_decoupling and represent is not None:
            # prepare represent
            if self.separate_decoder:
                if self.orthogonal_emb:
                    prt_emb = torch_expm(
                        (self.part_latents.weight - self.part_latents.weight.transpose(0, 1)).unsqueeze(0)
                    )
                z_parts = represent.chunk(self.num_represent_embeds, dim=1)
            else:
                z_parts = represent

            out_grad = 0
            h0 = h.clone()
            sub_grad = torch.zeros_like(x)
            for idx in range(self.num_represent_embeds):
                if self.separate_decoder:
                    cond = self.represent_embed(z_parts[idx])
                    prt_idx = torch.tensor([idx] * h0.shape[0]).to(h0.device)

                    if self.orthogonal_emb:
                        part_emb = prt_emb[prt_idx]
                    else:
                        part_emb = self.part_latents(prt_idx)
                    part_emb = self.part_emb(part_emb)

                    shift_t_emb = emb + part_emb
                else:
                    cond = self.represent_embed(z_parts)
                    shift_t_emb = emb

                # 5. shift mid
                shift_h = self.shift_middle_block(hidden_states=h0, temb=shift_t_emb, context=cond)

                # 6. shift up
                offset = 0
                for shift_upsample_block in self.shift_up_blocks:
                    if offset == 0:
                        res_samples = down_block_res_samples[-len(shift_upsample_block.resnets):]
                        offset = -len(shift_upsample_block.resnets)
                    else:
                        res_samples = down_block_res_samples[-len(shift_upsample_block.resnets) + offset:offset]
                        offset += -len(shift_upsample_block.resnets)
                    # down_block_res_samples = down_block_res_samples[: -len(shift_upsample_block.resnets)]
                    shift_h = shift_upsample_block(
                        hidden_states=shift_h, res_hidden_states_list=res_samples, temb=shift_t_emb, context=cond
                    )

                # 7. shift output
                shift_h = self.shift_out(shift_h)

                if self.separate_decoder:
                    if sampled_concept is not None:
                        indexes = torch.where(sampled_concept == idx)[0]
                        sub_grad[indexes] = shift_h[indexes]

                    if sampled_index is not None:
                        indexes = torch.where(sampled_index == idx)[0]
                        sub_grad[indexes[sampled_index == idx]] = shift_h[indexes[sampled_index == idx]]

                    out_grad += shift_h
                else:
                    out_grad = shift_h
                    break

        with torch.no_grad() if self.with_decoupling else contextlib.nullcontext():
            # 5. mid
            h = self.middle_block(hidden_states=h, temb=emb, context=context)

            # Additional residual conections for Controlnets
            if mid_block_additional_residual is not None:
                h = h + mid_block_additional_residual

            # 6. up
            for upsample_block in self.up_blocks:
                res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
                h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)

            # 7. output
            pred = self.out(h)

        if self.with_decoupling:
            if self.separate_decoder:
                if sampled_concept is not None:
                    return Return_grad_full(pred=pred, out_grad=out_grad, sub_grad=sub_grad)
                elif sampled_index is not None:
                    return Return_grad(pred=pred, out_grad=sub_grad)

            return Return_grad(pred=pred, out_grad=out_grad)

        return pred


class DiffusionModelEncoder(nn.Module):
    """
    Classification Network based on the Encoder of the Diffusion Model, followed by fully connected layers. This network is based on
    Wolleb et al. "Diffusion Models for Medical Anomaly Detection" (https://arxiv.org/abs/2203.04306).

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResnetBlock) per level.
        num_channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        resblock_updown: if True use residual blocks for downsampling.
        num_head_channels: number of channels in each attention head.
        with_conditioning: if True add spatial transformers to perform conditioning.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        num_class_embeds: if specified (as an int), then this model will be class-conditional with `num_class_embeds` classes.
        upcast_attention: if True, upcast attention operations to full precision.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
            num_channels: Sequence[int] = (32, 64, 64, 64),
            attention_levels: Sequence[bool] = (False, False, True, True),
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            use_scale_shift_norm: bool = False,
            resblock_updown: bool = False,
            num_head_channels: int | Sequence[int] = 8,
            with_conditioning: bool = False,
            transformer_num_layers: int = 1,
            cross_attention_dim: int | None = None,
            num_class_embeds: int | None = None,
            upcast_attention: bool = False,
    ) -> None:
        super().__init__()
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "DiffusionModelEncoder expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "when using with_conditioning."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "DiffusionModelEncoder expects with_conditioning=True when specifying the cross_attention_dim."
            )

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError("DiffusionModelEncoder expects all num_channels being multiple of norm_num_groups")
        if len(num_channels) != len(attention_levels):
            raise ValueError("DiffusionModelEncoder expects num_channels being same size of attention_levels")

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # time
        time_embed_dim = num_channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels[0], time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )

        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels)  # - 1

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                use_scale_shift_norm=use_scale_shift_norm,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
            )

            self.down_blocks.append(down_block)

        self.out = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, self.out_channels))

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            context: torch.Tensor | None = None,
            class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            context: context tensor (N, 1, ContextDim).
            class_labels: context tensor (N, ).
        """
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        # 2. class
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        # 3. initial convolution
        h = self.conv_in(x)

        # 4. down
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        for downsample_block in self.down_blocks:
            h, _ = downsample_block(hidden_states=h, temb=emb, context=context)

        h = h.reshape(h.shape[0], -1)
        output = self.out(h)

        return output


def torch_log2(x):
    return torch.log(x) / np.log(2.0)


def torch_pade13(A):
    b = torch.tensor(
        [
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ],
        dtype=A.dtype,
        device=A.device,
    )

    ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
    A2 = torch.matmul(A, A)
    A4 = torch.matmul(A2, A2)
    A6 = torch.matmul(A4, A2)
    U = torch.matmul(
        A,
        torch.matmul(A6, b[13] * A6 + b[11] * A4 + b[9] * A2)
        + b[7] * A6
        + b[5] * A4
        + b[3] * A2
        + b[1] * ident,
    )
    V = (
            torch.matmul(A6, b[12] * A6 + b[10] * A4 + b[8] * A2)
            + b[6] * A6
            + b[4] * A4
            + b[2] * A2
            + b[0] * ident
    )
    return U, V


def torch_expm(A):
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1, 2), keepdim=True))

    # Scaling step
    maxnorm = torch.tensor([5.371920351148152], dtype=A.dtype, device=A.device)
    zero = torch.tensor([0.0], dtype=A.dtype, device=A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    A_scaled = A / 2.0 ** n_squarings
    n_squarings = n_squarings.flatten().type(torch.int64)

    # Pade 13 approximation
    U, V = torch_pade13(A_scaled)
    P = U + V
    Q = -U + V
    R = torch.linalg.solve(P, Q)

    # Unsquaring step
    res = [R]
    for i in range(int(n_squarings.max())):
        res.append(res[-1].matmul(res[-1]))
    R = torch.stack(res)
    expmA = R[n_squarings, torch.arange(n_A)]
    return expmA[0]

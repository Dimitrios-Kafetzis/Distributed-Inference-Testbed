"""Per-head decomposition of a decoder layer: head units, projection, FFN.

A Hugging Face attention module computes all heads in one matrix product.
:class:`HeadwiseLayer` rebuilds the same computation from the same weights as
separate ``nn.Module`` units, one per attention head (query, key and value
slices, rotary embedding for Llama, causal softmax, context), one for the
output projection (the gather of the head outputs) and one for the
feed-forward block (with its normalisation), so that forward hooks can time
each unit. The output equals the stock layer's output.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn

from src.models.torch.decoder import FAMILIES, DecoderStageAdapter, causal_mask


def _project(linear: nn.Module, x: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Rows ``start:stop`` of a linear projection (one head's slice), bias included."""
    out = x @ linear.weight[start:stop].T
    bias = getattr(linear, "bias", None)
    if bias is not None:
        out = out + bias[start:stop]
    return cast(torch.Tensor, out)


def rotary_cos_sin(
    attention: nn.Module, v: torch.Tensor, positions: torch.Tensor, t: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rotary tables of the layer: Transformers 4.38 and later take the positions,
    4.36 takes the sequence length."""
    rotary = attention.rotary_emb
    try:
        cos, sin = rotary(v, positions)
    except TypeError:
        cos, sin = rotary(v, seq_len=t)
    return cast(torch.Tensor, cos), cast(torch.Tensor, sin)


class HeadUnit(nn.Module):
    """One attention head: q, k, v slices of the layer's projections, scaled causal attention."""

    def __init__(self, adapter: DecoderStageAdapter, attention: nn.Module, head: int) -> None:
        super().__init__()
        self.family = adapter.family
        self.spec = adapter.spec
        self.attention = attention
        self.head = head
        self.num_heads = adapter.num_heads()
        self.head_dim = adapter.hidden_size() // self.num_heads
        self.kv_head = head // max(1, self.num_heads // adapter.num_kv_heads())

    def forward(
        self, normed: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        t = normed.shape[1]
        hd, h, kvh = self.head_dim, self.head, self.kv_head
        if self.spec.fused_qkv:
            qkv = getattr(self.attention, self.spec.fused_qkv)(normed)  # [1, T, 3D]
            d = qkv.shape[-1] // 3
            q = qkv[..., h * hd : (h + 1) * hd]
            k = qkv[..., d + h * hd : d + (h + 1) * hd]
            v = qkv[..., 2 * d + h * hd : 2 * d + (h + 1) * hd]
        else:
            q = _project(self.attention.q_proj, normed, h * hd, (h + 1) * hd)
            k = _project(self.attention.k_proj, normed, kvh * hd, (kvh + 1) * hd)
            v = _project(self.attention.v_proj, normed, kvh * hd, (kvh + 1) * hd)
        if self.spec.rotary:
            cos, sin = rotary_cos_sin(self.attention, v, positions, t)
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

            q, k = apply_rotary_pos_emb(q[:, None], k[:, None], cos, sin, positions)
            q, k = q[:, 0], k[:, 0]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd) + mask[0, 0, :t, :t]
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return cast(torch.Tensor, weights @ v)  # [1, T, hd]


class ProjectionUnit(nn.Module):
    """The gather of the head outputs: concatenation and output projection."""

    def __init__(self, family: str, attention: nn.Module) -> None:
        super().__init__()
        self.family = family
        self.attention = attention
        self.out_proj = FAMILIES[family].out_proj

    def forward(self, head_outputs: list[torch.Tensor]) -> torch.Tensor:
        context = torch.cat(head_outputs, dim=-1)
        return cast(torch.Tensor, getattr(self.attention, self.out_proj)(context))


class FfnUnit(nn.Module):
    def __init__(self, norm: nn.Module, mlp: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.mlp = mlp

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.mlp(self.norm(hidden)))


class HeadwiseLayer(nn.Module):
    """A decoder layer as head units, a projection unit and an FFN unit, with residuals."""

    def __init__(self, adapter: DecoderStageAdapter, layer: nn.Module) -> None:
        super().__init__()
        attention = adapter.attention_module(layer)
        self.attention_norm = adapter.attention_norm(layer)
        self.heads = nn.ModuleList(
            [HeadUnit(adapter, attention, h) for h in range(adapter.num_heads())]
        )
        self.projection = ProjectionUnit(adapter.family, attention)
        self.ffn = FfnUnit(adapter.ffn_norm(layer), adapter.mlp_module(layer))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        t = hidden.shape[1]
        mask = causal_mask(t, hidden.dtype)
        positions = torch.arange(t)[None, :]
        normed = self.attention_norm(hidden)
        head_outputs = [head(normed, mask, positions) for head in self.heads]
        hidden = hidden + self.projection(head_outputs)
        return cast(torch.Tensor, hidden + self.ffn(hidden))


def headwise_layers(adapter: DecoderStageAdapter) -> list[HeadwiseLayer]:
    adapter._ensure_built()
    return [HeadwiseLayer(adapter, layer) for layer in adapter.layers()]


def block_names(adapter: DecoderStageAdapter) -> dict[str, Any]:
    """The names used in block profiles: embed, layer{l}.head{h}, layer{l}.projection,
    layer{l}.ffn, lm_head."""
    names: dict[str, Any] = {"embed": None, "lm_head": None}
    for layer in range(1, adapter.num_layers() + 1):
        for head in range(adapter.num_heads()):
            names[f"layer{layer}.head{head}"] = (layer, "head", head)
        names[f"layer{layer}.projection"] = (layer, "projection", None)
        names[f"layer{layer}.ffn"] = (layer, "ffn", None)
    return names

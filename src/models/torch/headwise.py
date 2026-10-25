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

from src.models.torch.decoder import DecoderStageAdapter, causal_mask


class HeadUnit(nn.Module):
    """One attention head: q, k, v slices of the layer's projections, scaled causal attention."""

    def __init__(self, adapter: DecoderStageAdapter, attention: nn.Module, head: int) -> None:
        super().__init__()
        self.family = adapter.family
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
        if self.family == "gpt2":
            qkv = self.attention.c_attn(normed)  # [1, T, 3D]
            d = qkv.shape[-1] // 3
            q = qkv[..., h * hd : (h + 1) * hd]
            k = qkv[..., d + h * hd : d + (h + 1) * hd]
            v = qkv[..., 2 * d + h * hd : 2 * d + (h + 1) * hd]
        else:
            wq = self.attention.q_proj.weight[h * hd : (h + 1) * hd]
            wk = self.attention.k_proj.weight[kvh * hd : (kvh + 1) * hd]
            wv = self.attention.v_proj.weight[kvh * hd : (kvh + 1) * hd]
            q = normed @ wq.T
            k = normed @ wk.T
            v = normed @ wv.T
            cos, sin = self.attention.rotary_emb(v, seq_len=t)
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

    def forward(self, head_outputs: list[torch.Tensor]) -> torch.Tensor:
        context = torch.cat(head_outputs, dim=-1)
        if self.family == "gpt2":
            return cast(torch.Tensor, self.attention.c_proj(context))
        return cast(torch.Tensor, self.attention.o_proj(context))


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

"""Decoder-only language models as stages: embedding, decoder layers, LM head.

The stage view runs a fixed-length prompt (prefill, no key-value cache): the
cut tensor between two stages is the hidden-state tensor [1, T, D], and the
last stage produces the logits [1, T, V]. Operation counts per stage are
computed analytically from the configuration for T tokens. Weights come from
a Hugging Face checkpoint directory (the local mirror) or, for tests and dry
runs, from a small random configuration.

Model families differ only in attribute names and a few structural facts
(fused or separate query, key and value projections; rotary or learned
positions; gated or plain feed-forward block; scaled embeddings). The
:data:`FAMILIES` table holds these per family, so that one adapter covers
GPT-2, Llama, Mistral, Qwen2 and Gemma, and a new family is one more row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from src.models.registry import Array, StageInfo, deserialize_array, serialize_array


@dataclass(frozen=True)
class FamilySpec:
    """Where a family keeps its blocks and what its configuration keys are called."""

    model_type: str
    layers: str  # attribute path of the decoder layer list
    attention: str  # attribute of the attention module inside a layer
    mlp: str
    attention_norm: str
    ffn_norm: str
    embed_tokens: str
    final_norm: str
    cfg_layers: str
    cfg_hidden: str
    cfg_heads: str
    cfg_inner: str
    cfg_kv_heads: str | None = None  # None: as many key-value heads as heads
    embed_positions: str | None = None  # learned positions (GPT-2)
    fused_qkv: str | None = None  # one projection for q, k and v (GPT-2's c_attn)
    out_proj: str = "o_proj"
    gated_ffn: bool = True  # gate, up and down projections (3 matrix products)
    rotary: bool = True
    embed_scale: bool = False  # embeddings multiplied by sqrt(hidden size) (Gemma)


def _llama_like(model_type: str, **overrides: Any) -> FamilySpec:
    return FamilySpec(
        model_type=model_type,
        layers="model.layers",
        attention="self_attn",
        mlp="mlp",
        attention_norm="input_layernorm",
        ffn_norm="post_attention_layernorm",
        embed_tokens="model.embed_tokens",
        final_norm="model.norm",
        cfg_layers="num_hidden_layers",
        cfg_hidden="hidden_size",
        cfg_heads="num_attention_heads",
        cfg_inner="intermediate_size",
        cfg_kv_heads="num_key_value_heads",
        **overrides,
    )


FAMILIES: dict[str, FamilySpec] = {
    "gpt2": FamilySpec(
        model_type="gpt2",
        layers="transformer.h",
        attention="attn",
        mlp="mlp",
        attention_norm="ln_1",
        ffn_norm="ln_2",
        embed_tokens="transformer.wte",
        embed_positions="transformer.wpe",
        final_norm="transformer.ln_f",
        cfg_layers="n_layer",
        cfg_hidden="n_embd",
        cfg_heads="n_head",
        cfg_inner="n_inner",
        fused_qkv="c_attn",
        out_proj="c_proj",
        gated_ffn=False,
        rotary=False,
    ),
    "llama": _llama_like("llama"),
    "mistral": _llama_like("mistral"),
    "qwen2": _llama_like("qwen2"),
    "gemma": _llama_like("gemma", embed_scale=True),
}
MODEL_TYPE_TO_FAMILY = {spec.model_type: name for name, spec in FAMILIES.items()}


def resolve(root: Any, path: str) -> Any:
    """``resolve(model, "transformer.h")`` -> ``model.transformer.h``."""
    node = root
    for part in path.split("."):
        node = getattr(node, part)
    return node


def causal_mask(seq_len: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Additive 4-D causal mask [1, 1, T, T] (0 on and below the diagonal, -inf above)."""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return cast(torch.Tensor, mask[None, None, :, :])


class DecoderStageAdapter:
    """Base adapter; subclasses set ``family`` and build the Hugging Face model."""

    name = "decoder"
    framework = "torch"
    family = "gpt2"
    PARAMETERS = {
        "type": "object",
        "properties": {
            "seq_len": {
                "type": "integer",
                "minimum": 1,
                "default": 32,
                "description": "prompt length in tokens (the prefill the stages run)",
            },
            "config": {
                "type": "object",
                "description": "Hugging Face configuration overrides (a smaller model for tests)",
            },
        },
    }

    def __init__(self, seq_len: int = 32, config: dict[str, Any] | None = None) -> None:
        self.seq_len = seq_len
        self.config_overrides = dict(config or {})
        self.model: nn.Module | None = None
        self.input_shape: tuple[int, ...] = (seq_len,)
        self._stages: list[StageInfo] | None = None

    # --- construction --------------------------------------------------------------------------

    @property
    def spec(self) -> FamilySpec:
        try:
            return FAMILIES[self.family]
        except KeyError as exc:
            raise ValueError(
                f"unknown model family {self.family!r}; known: {sorted(FAMILIES)}"
            ) from exc

    def hf_config(self) -> Any:
        raise NotImplementedError

    def build(self, seed: int) -> None:
        from transformers import AutoModelForCausalLM

        torch.manual_seed(seed)
        self.model = AutoModelForCausalLM.from_config(self.hf_config())
        self.model.eval()
        self._stages = None

    def load_pretrained(self, path: str | Path) -> None:
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32)
        self.model.eval()
        self._stages = None

    def _ensure_built(self) -> None:
        if self.model is None:
            self.build(0)

    # --- structure (from the family table) ----------------------------------------------------

    @property
    def cfg(self) -> Any:
        self._ensure_built()
        assert self.model is not None
        return self.model.config

    def num_layers(self) -> int:
        return int(getattr(self.cfg, self.spec.cfg_layers))

    def hidden_size(self) -> int:
        return int(getattr(self.cfg, self.spec.cfg_hidden))

    def num_heads(self) -> int:
        return int(getattr(self.cfg, self.spec.cfg_heads))

    def num_kv_heads(self) -> int:
        key = self.spec.cfg_kv_heads
        if key is None:
            return self.num_heads()
        return int(getattr(self.cfg, key, None) or self.num_heads())

    def intermediate_size(self) -> int:
        inner = getattr(self.cfg, self.spec.cfg_inner, None)
        return int(inner or 4 * self.hidden_size())

    def vocab_size(self) -> int:
        return int(self.cfg.vocab_size)

    def layers(self) -> list[nn.Module]:
        assert self.model is not None
        return list(resolve(self.model, self.spec.layers))

    def attention_module(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, getattr(layer, self.spec.attention))

    def mlp_module(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, getattr(layer, self.spec.mlp))

    def attention_norm(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, getattr(layer, self.spec.attention_norm))

    def ffn_norm(self, layer: nn.Module) -> nn.Module:
        return cast(nn.Module, getattr(layer, self.spec.ffn_norm))

    def embed_module(self) -> nn.Module:
        assert self.model is not None
        return cast(nn.Module, resolve(self.model, self.spec.embed_tokens))

    # --- operation counts ----------------------------------------------------------------------

    def layer_macs(self) -> tuple[int, int, int]:
        """(attention heads, projection, feed-forward) MACs for one layer and T tokens."""
        t, d, h, kv = self.seq_len, self.hidden_size(), self.num_heads(), self.num_kv_heads()
        head_dim = d // h
        # q for every head, k and v for the key-value heads, scores and context per head.
        heads = t * d * d + 2 * t * d * (kv * head_dim) + 2 * h * t * t * head_dim
        projection = t * d * d
        inner = self.intermediate_size()
        ffn = 3 * t * d * inner if self.spec.gated_ffn else 2 * t * d * inner
        return heads, projection, ffn

    @property
    def stages(self) -> list[StageInfo]:
        self._ensure_built()
        if self._stages is None:
            t, d, v = self.seq_len, self.hidden_size(), self.vocab_size()
            heads, projection, ffn = self.layer_macs()
            per_layer = heads + projection + ffn
            stages = [StageInfo(1, "embed", 0, 0, (t, d), t * d * 4)]
            for layer in range(1, self.num_layers() + 1):
                stages.append(
                    StageInfo(
                        layer + 1, f"layer{layer}", 2 * per_layer, per_layer, (t, d), t * d * 4
                    )
                )
            lm = t * d * v
            stages.append(
                StageInfo(self.num_layers() + 2, "lm_head", 2 * lm, lm, (t, v), t * v * 4)
            )
            self._stages = stages
        return self._stages

    @property
    def input_bytes(self) -> int:
        return self.seq_len * 8  # int64 token ids

    # --- execution -----------------------------------------------------------------------------

    def make_input(self, seed: int) -> Array:
        rng = np.random.default_rng(seed)
        return rng.integers(0, self.vocab_size(), size=(1, self.seq_len), dtype=np.int64)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        embedded = self.embed_module()(ids)
        if self.spec.embed_positions:
            positions = torch.arange(ids.shape[1])[None, :]
            embedded = embedded + resolve(self.model, self.spec.embed_positions)(positions)
        if self.spec.embed_scale:
            embedded = embedded * torch.tensor(self.hidden_size() ** 0.5, dtype=embedded.dtype)
        return cast(torch.Tensor, embedded)

    def run_layer(self, hidden: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        if not self.spec.rotary:
            return cast(torch.Tensor, layer(hidden)[0])
        t = hidden.shape[1]
        positions = torch.arange(t)[None, :]
        out = layer(hidden, attention_mask=causal_mask(t, hidden.dtype), position_ids=positions)
        return cast(torch.Tensor, out[0] if isinstance(out, tuple) else out)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        normed = resolve(self.model, self.spec.final_norm)(hidden)
        return cast(torch.Tensor, self.model.lm_head(normed))

    @torch.no_grad()
    def run_stages(self, x: Array, start: int, stop: int) -> Array:
        self._ensure_built()
        n = len(self.stages)
        if not 1 <= start <= stop <= n:
            raise ValueError(f"stage range {start}..{stop} outside 1..{n}")
        tensor = torch.from_numpy(np.ascontiguousarray(x))
        layers = self.layers()
        for index in range(start, stop + 1):
            if index == 1:
                tensor = self.embed(tensor.to(torch.int64))
            elif index == n:
                tensor = self.head(tensor.to(torch.float32))
            else:
                tensor = self.run_layer(tensor.to(torch.float32), layers[index - 2])
        return cast(Array, tensor.detach().cpu().numpy())

    @torch.no_grad()
    def full_logits(self, x: Array) -> Array:
        """Reference: the Hugging Face forward pass (for equivalence tests)."""
        self._ensure_built()
        assert self.model is not None
        ids = torch.from_numpy(np.ascontiguousarray(x)).to(torch.int64)
        return cast(Array, self.model(ids).logits.numpy())

    # --- weights and serialisation -------------------------------------------------------------

    def save_weights(self, path: Path) -> None:
        self._ensure_built()
        assert self.model is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()})

    def load_weights(self, path: Path) -> None:
        if Path(path).is_dir():
            self.load_pretrained(path)
            return
        self._ensure_built()
        assert self.model is not None
        data = np.load(path)
        state = {k: torch.from_numpy(data[k]) for k in data.files}
        self.model.load_state_dict(state, strict=True)

    def serialize(self, array: Array) -> bytes:
        return serialize_array(array)

    def deserialize(self, data: bytes) -> Array:
        return deserialize_array(data)

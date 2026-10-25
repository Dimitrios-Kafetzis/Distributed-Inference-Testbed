"""TinyLlama-1.1B (Llama architecture) as a stage adapter.

Without overrides the TinyLlama-1.1B configuration is built (22 layers, 32
attention heads, 4 key-value heads, 2048 dimensions, 5632 intermediate);
tests use a tiny Llama configuration. Weights are loaded from a checkpoint
directory with ``load_weights``.
"""

from __future__ import annotations

from typing import Any

from src.models.registry import register
from src.models.torch.decoder import DecoderStageAdapter

TINYLLAMA_1_1B = {
    "num_hidden_layers": 22,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "vocab_size": 32000,
    "max_position_embeddings": 2048,
}


class TinyLlamaAdapter(DecoderStageAdapter):
    name = "tinyllama"
    family = "llama"

    def hf_config(self) -> Any:
        from transformers import LlamaConfig

        settings = {**TINYLLAMA_1_1B, **self.config_overrides}
        settings.update({"use_cache": False, "attn_implementation": "eager"})
        return LlamaConfig(**settings)


ADAPTERS = {"tinyllama": TinyLlamaAdapter}
register("tinyllama", TinyLlamaAdapter)

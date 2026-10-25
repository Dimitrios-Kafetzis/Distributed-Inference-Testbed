"""Any Hugging Face decoder-only model as a stage adapter, and two named models.

``hf_decoder`` takes a checkpoint (a directory distributed to the devices, or a
model name in the Hugging Face cache) or a configuration and derives the family
from ``model_type`` through the family table of :mod:`src.models.torch.decoder`.
``llama32_1b`` (Llama 3.2 1B) and ``qwen25_05b`` (Qwen2.5 0.5B) are registered
with their published configurations, like ``gpt2`` and ``tinyllama``: without a
checkpoint they are built with random weights (tests use tiny overrides).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from src.models.registry import register
from src.models.torch.decoder import FAMILIES, MODEL_TYPE_TO_FAMILY, DecoderStageAdapter

LLAMA_3_2_1B = {
    "model_type": "llama",
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },
    "tie_word_embeddings": True,
}

QWEN2_5_0_5B = {
    "model_type": "qwen2",
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "hidden_size": 896,
    "intermediate_size": 4864,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
}


def config_from_dict(settings: dict[str, Any]) -> Any:
    """A Hugging Face configuration object from a plain mapping with ``model_type``."""
    from transformers import AutoConfig

    settings = copy.deepcopy(settings)
    model_type = settings.pop("model_type")
    settings.setdefault("use_cache", False)
    if model_type != "gpt2":
        settings.setdefault("attn_implementation", "eager")
    return AutoConfig.for_model(model_type, **settings)


class HFDecoderAdapter(DecoderStageAdapter):
    """A checkpoint directory or model name, or a configuration mapping; the family
    follows ``model_type``."""

    name = "hf_decoder"
    PARAMETERS = {
        "type": "object",
        "properties": {
            "checkpoint": {
                "type": ["string", "null"],
                "description": "checkpoint directory on the node, or a Hugging Face model name "
                "present in its cache; null builds random weights from `config`",
            },
            "config": {
                "type": "object",
                "description": "Hugging Face configuration with model_type (used without a "
                "checkpoint, or as overrides)",
            },
            "seq_len": {"type": "integer", "minimum": 1, "default": 32},
        },
    }

    def __init__(
        self,
        checkpoint: str | None = None,
        config: dict[str, Any] | None = None,
        seq_len: int = 32,
    ) -> None:
        super().__init__(seq_len=seq_len, config=config)
        self.checkpoint = checkpoint
        self.family = "llama"  # until the configuration is known
        self._config: Any = None

    def hf_config(self) -> Any:
        if self._config is None:
            if self.checkpoint:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(self.checkpoint)
                for key, value in self.config_overrides.items():
                    setattr(config, key, value)
            elif "model_type" in self.config_overrides:
                config = config_from_dict(self.config_overrides)
            else:
                raise ValueError("hf_decoder needs a checkpoint or a config with model_type")
            family = MODEL_TYPE_TO_FAMILY.get(getattr(config, "model_type", ""))
            if family is None:
                raise ValueError(
                    f"model_type {config.model_type!r} is not in the family table "
                    f"({sorted(FAMILIES)})"
                )
            self.family = family
            self._config = config
        return self._config

    def build(self, seed: int) -> None:
        if self.checkpoint and Path(self.checkpoint).is_dir():
            self.hf_config()
            self.load_pretrained(self.checkpoint)
            return
        super().build(seed)


class Llama32Adapter(DecoderStageAdapter):
    name = "llama32_1b"
    family = "llama"

    def hf_config(self) -> Any:
        return config_from_dict({**LLAMA_3_2_1B, **self.config_overrides})


class Qwen25Adapter(DecoderStageAdapter):
    name = "qwen25_05b"
    family = "qwen2"

    def hf_config(self) -> Any:
        return config_from_dict({**QWEN2_5_0_5B, **self.config_overrides})


ADAPTERS = {
    "hf_decoder": HFDecoderAdapter,
    "llama32_1b": Llama32Adapter,
    "qwen25_05b": Qwen25Adapter,
}
for _name, _factory in ADAPTERS.items():
    register(_name, _factory)

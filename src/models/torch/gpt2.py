"""GPT-2 (Small by default) as a stage adapter.

``config`` overrides build a smaller model (tests use ``n_layer=2``,
``n_embd=32``, ``n_head=4``); without overrides the GPT-2 Small configuration
(12 layers, 12 heads, 768 dimensions) is built with random weights, or loaded
from a checkpoint directory with ``load_weights``.
"""

from __future__ import annotations

from typing import Any

from src.models.registry import register
from src.models.torch.decoder import DecoderStageAdapter

GPT2_SMALL = {"n_layer": 12, "n_head": 12, "n_embd": 768, "vocab_size": 50257, "n_positions": 1024}


class Gpt2Adapter(DecoderStageAdapter):
    name = "gpt2"
    family = "gpt2"

    def hf_config(self) -> Any:
        from transformers import GPT2Config

        settings = {**GPT2_SMALL, **self.config_overrides}
        settings.setdefault("n_positions", max(1024, self.seq_len))
        settings.update(
            {"resid_pdrop": 0.0, "embd_pdrop": 0.0, "attn_pdrop": 0.0, "use_cache": False}
        )
        return GPT2Config(**settings)


ADAPTERS = {"gpt2": Gpt2Adapter}
register("gpt2", Gpt2Adapter)

"""PyTorch LLM adapters: stage view, head-wise decomposition, hook profiler.

Tiny random configurations only; nothing is downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.models import registry  # noqa: E402
from src.models.torch import hooks  # noqa: E402
from src.models.torch.decoder import DecoderStageAdapter  # noqa: E402
from src.models.torch.headwise import headwise_layers  # noqa: E402

GPT2_TINY = {"n_layer": 2, "n_embd": 32, "n_head": 4, "vocab_size": 101}
LLAMA_TINY = {
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "hidden_size": 32,
    "intermediate_size": 48,
    "vocab_size": 101,
    "max_position_embeddings": 64,
}


TINY_LLAMA_LIKE = {
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "hidden_size": 32,
    "intermediate_size": 48,
    "vocab_size": 101,
    "max_position_embeddings": 64,
}
# Registered name -> (keyword arguments); every row of the family table is covered.
TINY_MODELS: dict[str, dict[str, object]] = {
    "gpt2": {"config": GPT2_TINY},
    "tinyllama": {"config": LLAMA_TINY},
    "llama32_1b": {"config": TINY_LLAMA_LIKE},
    "qwen25_05b": {"config": TINY_LLAMA_LIKE},
    "mistral": {"config": {"model_type": "mistral", **TINY_LLAMA_LIKE}},
    "gemma": {"config": {"model_type": "gemma", "head_dim": 8, **TINY_LLAMA_LIKE}},
}


@pytest.fixture(scope="module", params=list(TINY_MODELS))
def adapter(request: pytest.FixtureRequest) -> DecoderStageAdapter:
    name = "hf_decoder" if request.param in ("mistral", "gemma") else request.param
    built = registry.get(name, seq_len=8, **TINY_MODELS[request.param])
    assert isinstance(built, DecoderStageAdapter)
    built.build(seed=1)
    return built


def test_stage_view_matches_the_hugging_face_forward_pass(adapter: DecoderStageAdapter) -> None:
    assert [s.name for s in adapter.stages] == ["embed", "layer1", "layer2", "lm_head"]
    x = adapter.make_input(3)
    assert x.shape == (1, 8) and x.dtype == np.int64
    logits = adapter.run_stages(x, 1, 4)
    assert logits.shape == (1, 8, 101)
    np.testing.assert_allclose(logits, adapter.full_logits(x), rtol=1e-4, atol=1e-5)
    for s in (1, 2, 3):
        cut = adapter.run_stages(x, 1, s)
        assert cut.shape == (1, 8, 32)
        np.testing.assert_allclose(adapter.run_stages(cut, s + 1, 4), logits, rtol=1e-4, atol=1e-5)
    assert adapter.stages[1].flops > 0 and adapter.stages[3].macs == 8 * 32 * 101
    assert adapter.input_bytes == 64 and adapter.stages[1].output_bytes == 8 * 32 * 4


def test_headwise_decomposition_equals_the_stock_layer(adapter: DecoderStageAdapter) -> None:
    x = adapter.make_input(5)
    hidden = torch.from_numpy(adapter.run_stages(x, 1, 1))
    layers = headwise_layers(adapter)
    with torch.no_grad():
        stock = adapter.run_layer(hidden, adapter.layers()[0])
        ours = layers[0](hidden)
    np.testing.assert_allclose(ours.detach().numpy(), stock.detach().numpy(), rtol=1e-4, atol=1e-5)
    full = hooks.forward_headwise(adapter, layers, torch.from_numpy(x))
    np.testing.assert_allclose(full.detach().numpy(), adapter.full_logits(x), rtol=1e-4, atol=1e-5)
    assert len(layers[0].heads) == 4


def test_hook_profiler_times_every_block_and_removes_hooks(adapter: DecoderStageAdapter) -> None:
    result = hooks.profile_blocks(adapter, adapter.make_input(1), repetitions=2, warmup=1)
    expected = {"embed", "lm_head"} | {
        f"layer{layer}.{b}" for layer in (1, 2) for b in ("projection", "ffn")
    }
    expected |= {f"layer{layer}.head{h}" for layer in (1, 2) for h in range(4)}
    assert set(result["blocks_ns"]) == expected
    assert all(len(v) == 2 and all(t > 0 for t in v) for v in result["blocks_ns"].values())
    assert (
        result["per_layer"]["layer1"]["attention_heads_ns"]
        >= result["per_layer"]["layer1"]["head_max_ns"]
    )
    _, modules = hooks.block_modules(adapter)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in modules.values())


def test_weights_round_trip(adapter: DecoderStageAdapter, tmp_path: Path) -> None:
    x = adapter.make_input(2)
    expected = adapter.run_stages(x, 1, 4)
    adapter.save_weights(tmp_path / "w.npz")
    # A second adapter of the same configuration: the generic adapter needs the
    # configuration mapping (model_type included), the named ones carry theirs.
    if adapter.name == "hf_decoder":
        key = next(
            k
            for k in ("mistral", "gemma")
            if TINY_MODELS[k]["config"]["model_type"] == adapter.family  # type: ignore[index]
        )
    else:
        key = adapter.name
    kwargs = TINY_MODELS[key]
    other = registry.get(adapter.name, seq_len=8, **kwargs)
    other.build(seed=9)
    assert not np.allclose(other.run_stages(x, 1, 4), expected)
    other.load_weights(tmp_path / "w.npz")
    np.testing.assert_allclose(other.run_stages(x, 1, 4), expected, rtol=1e-6)


def test_hooks_cli_writes_a_block_profile(tmp_path: Path) -> None:
    out = tmp_path / "blocks.json"
    hooks.main(
        [
            "--model",
            "gpt2",
            "--config",
            json.dumps(GPT2_TINY),
            "--seq-len",
            "4",
            "--repetitions",
            "1",
            "--warmup",
            "0",
            "--output",
            str(out),
            "--node-id",
            "d1",
        ]
    )
    data = json.loads(out.read_text())
    assert data["num_layers"] == 2 and data["node_id"] == "d1" and "layer2.ffn" in data["blocks_ns"]


def test_default_configurations_are_the_thesis_models() -> None:
    from src.models.torch.gpt2 import GPT2_SMALL
    from src.models.torch.tinyllama import TINYLLAMA_1_1B

    assert (
        GPT2_SMALL["n_layer"] == 12 and GPT2_SMALL["n_head"] == 12 and GPT2_SMALL["n_embd"] == 768
    )
    assert TINYLLAMA_1_1B["num_hidden_layers"] == 22 and TINYLLAMA_1_1B["num_attention_heads"] == 32
    assert TINYLLAMA_1_1B["hidden_size"] == 2048


def test_family_table_and_named_models() -> None:
    from src.models.torch.decoder import FAMILIES, MODEL_TYPE_TO_FAMILY
    from src.models.torch.hf_decoder import LLAMA_3_2_1B, QWEN2_5_0_5B, HFDecoderAdapter

    assert set(FAMILIES) == {"gpt2", "llama", "mistral", "qwen2", "gemma"}
    assert MODEL_TYPE_TO_FAMILY["qwen2"] == "qwen2" and FAMILIES["gemma"].embed_scale
    assert LLAMA_3_2_1B["num_hidden_layers"] == 16 and LLAMA_3_2_1B["num_key_value_heads"] == 8
    assert QWEN2_5_0_5B["num_hidden_layers"] == 24 and QWEN2_5_0_5B["hidden_size"] == 896
    generic = registry.get(
        "hf_decoder", seq_len=4, config={"model_type": "qwen2", **TINY_LLAMA_LIKE}
    )
    assert isinstance(generic, HFDecoderAdapter)
    generic.build(0)
    assert generic.family == "qwen2" and len(generic.stages) == 4
    with pytest.raises(ValueError, match="checkpoint or a config"):
        registry.get("hf_decoder").build(0)
    with pytest.raises(ValueError, match="family table"):
        registry.get("hf_decoder", config={"model_type": "bert", "vocab_size": 10}).build(0)

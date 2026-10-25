from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import registry
from src.models.fake import FakeStagesAdapter


def test_builtin_fake_is_registered() -> None:
    assert "fake" in registry.names()
    adapter = registry.get("fake")
    assert isinstance(adapter, FakeStagesAdapter)
    assert isinstance(adapter, registry.ModelAdapter)
    with pytest.raises(KeyError, match="unknown model"):
        registry.get("no-such-model")


def test_plugin_module_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(registry.MODULES_ENV, "tests.toy_model_plugin")
    registry.reset_for_tests()
    try:
        assert "toy" in registry.names()
        toy = registry.get("toy")
        toy.build(1)
        assert [s.name for s in toy.stages] == ["a", "b"]
        assert toy.run_stages(toy.make_input(0), 1, 2).shape == (1, 4)
    finally:
        monkeypatch.delenv(registry.MODULES_ENV)
        registry.reset_for_tests()


def test_duplicate_registration_rejected() -> None:
    registry.names()  # make sure the built-in adapters are loaded
    with pytest.raises(ValueError, match="already registered"):
        registry.register("fake", lambda: FakeStagesAdapter(input_elements=1))


def test_fake_adapter_partition_equivalence_and_weights(tmp_path: Path) -> None:
    adapter = FakeStagesAdapter(time_scale=0.0)
    adapter.build(seed=3)
    x = adapter.make_input(0)
    full = adapter.run_stages(x, 1, 5)
    for s in range(1, 5):
        cut = adapter.run_stages(x, 1, s)
        assert cut.shape == (1, adapter.stages[s - 1].output_shape[0])
        np.testing.assert_array_equal(adapter.run_stages(cut, s + 1, 5), full)
    adapter.save_weights(tmp_path / "w.npz")
    other = FakeStagesAdapter(time_scale=0.0)
    other.load_weights(tmp_path / "w.npz")
    np.testing.assert_array_equal(other.run_stages(x, 1, 5), full)
    assert adapter.deserialize(adapter.serialize(full)).tolist() == full.tolist()
    assert adapter.input_bytes == 2048 * 4
    with pytest.raises(ValueError):
        adapter.run_stages(x, 0, 5)
    with pytest.raises(RuntimeError):
        FakeStagesAdapter().run_stages(x, 1, 1)

"""TensorFlow adapters: shapes against the Chapter 4 tables, partition equivalence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytestmark = pytest.mark.slow

from src.models import registry  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CH4_TABLE = json.loads(
    (REPO / "src" / "models" / "reference" / "alexnet_ch4_table.json").read_text()
)


@pytest.fixture(scope="module")
def alexnet() -> registry.ModelAdapter:
    adapter = registry.get("alexnet")
    adapter.build(seed=1)
    return adapter


def test_alexnet_has_the_eleven_stages_of_chapter_4(alexnet: registry.ModelAdapter) -> None:
    assert alexnet.input_shape == (227, 227, 3)
    assert alexnet.input_bytes == CH4_TABLE["input"]["elements"] * 4
    assert len(alexnet.stages) == 11
    for stage, row in zip(alexnet.stages, CH4_TABLE["stages"], strict=True):
        assert stage.index == row["s"]
        assert list(stage.output_shape) == row["output_shape"], stage.name
        assert stage.output_bytes == row["elements"] * 4
    conv1 = alexnet.stages[0]
    assert conv1.macs == 105_415_200 and conv1.flops == 2 * conv1.macs
    assert alexnet.stages[8].macs == 9216 * 4096  # FC1
    assert alexnet.stages[1].macs == 0 and alexnet.stages[1].flops > 0  # pooling: comparisons only


@pytest.mark.parametrize("s", [1, 2, 4, 8, 9, 10])
def test_alexnet_partition_equivalence(alexnet: registry.ModelAdapter, s: int) -> None:
    x = alexnet.make_input(7)
    full = alexnet.run_stages(x, 1, 11)
    cut = alexnet.run_stages(x, 1, s)
    assert cut.shape[1:] == alexnet.stages[s - 1].output_shape
    np.testing.assert_allclose(alexnet.run_stages(cut, s + 1, 11), full, rtol=1e-5, atol=1e-6)
    assert full.shape == (1, 1000) and abs(float(full.sum()) - 1.0) < 1e-4


def test_alexnet_weights_round_trip(alexnet: registry.ModelAdapter, tmp_path: Path) -> None:
    x = alexnet.make_input(3)
    expected = alexnet.run_stages(x, 1, 11)
    alexnet.save_weights(tmp_path / "alexnet.npz")
    other = registry.get("alexnet")
    other.build(seed=99)
    assert not np.allclose(other.run_stages(x, 1, 11), expected)
    other.load_weights(tmp_path / "alexnet.npz")
    np.testing.assert_allclose(other.run_stages(x, 1, 11), expected, rtol=1e-6)
    restored = alexnet.deserialize(alexnet.serialize(expected))
    np.testing.assert_array_equal(restored, expected)


def test_vgg16_and_gru_shapes_and_equivalence() -> None:
    vgg = registry.get("vgg16")
    vgg.build(seed=0)
    assert [s.name for s in vgg.stages] == [
        "block1",
        "block2",
        "block3",
        "block4",
        "block5",
        "fc1",
        "fc2",
        "predictions",
    ]
    assert vgg.stages[4].output_shape == (7, 7, 512) and vgg.stages[7].output_shape == (1000,)
    x = vgg.make_input(1)
    full = vgg.run_stages(x, 1, 8)
    cut = vgg.run_stages(x, 1, 5)
    np.testing.assert_allclose(vgg.run_stages(cut, 6, 8), full, rtol=1e-5, atol=1e-6)

    gru = registry.get("gru_qoe")
    gru.build(seed=0)
    assert [s.output_shape for s in gru.stages] == [(32,), (128,), (1,)]
    assert gru.stages[0].output_bytes == 128  # the compact hidden state of Chapter 4
    x = gru.make_input(1)
    full = gru.run_stages(x, 1, 3)
    np.testing.assert_allclose(
        gru.run_stages(gru.run_stages(x, 1, 1), 2, 3), full, rtol=1e-5, atol=1e-6
    )


def test_resnet50_cuts_and_equivalence() -> None:
    resnet = registry.get("resnet50")
    resnet.build(seed=0)
    assert [s.output_shape for s in resnet.stages] == [
        (56, 56, 64),
        (56, 56, 256),
        (28, 28, 512),
        (14, 14, 1024),
        (7, 7, 2048),
        (1000,),
    ]
    assert sum(s.macs for s in resnet.stages) > 3.5e9  # about 3.9 GMACs for ResNet-50
    x = resnet.make_input(1)
    full = resnet.run_stages(x, 1, 6)
    for s in (1, 3, 5):
        cut = resnet.run_stages(x, 1, s)
        np.testing.assert_allclose(resnet.run_stages(cut, s + 1, 6), full, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", ["mobilenetv2", "efficientnetb0", "densenet121"])
def test_keras_applications_cut_automatically_and_partition_equivalence(name: str) -> None:
    from src.models.tf.keras_applications import find_single_tensor_cuts

    adapter = registry.get(name, input_size=64, max_stages=6)
    adapter.build(seed=0)
    stages = adapter.stages
    assert 2 <= len(stages) <= 6
    assert stages[-1].output_shape == (1000,) and stages[-1].name == "predictions"
    # Every stage boundary is a single-tensor cut of the functional graph.
    candidates = find_single_tensor_cuts(adapter._model)
    assert all(s.name in candidates for s in stages)
    assert len(candidates) > 6  # thinning was needed
    assert sum(s.macs for s in stages) > 0
    x = adapter.make_input(1)
    full = adapter.run_stages(x, 1, len(stages))
    for s in range(1, len(stages)):
        cut = adapter.run_stages(x, 1, s)
        assert cut.shape[1:] == stages[s - 1].output_shape
        np.testing.assert_allclose(
            adapter.run_stages(cut, s + 1, len(stages)), full, rtol=1e-4, atol=1e-5
        )


def test_keras_app_by_class_name_and_parameter_schema() -> None:
    from src.models.tf.keras_applications import KerasApplicationsAdapter, thin_cuts

    schema = registry.parameter_schema("keras_app")
    assert {"application", "num_classes", "input_size", "max_stages"} <= set(schema["properties"])
    adapter = registry.get(
        "keras_app", application="MobileNetV2", input_size=32, num_classes=10, max_stages=3
    )
    assert isinstance(adapter, KerasApplicationsAdapter)
    adapter.build(seed=2)
    assert len(adapter.stages) == 3 and adapter.stages[-1].output_shape == (10,)
    assert thin_cuts(adapter._model, adapter.candidate_cuts, 100) == adapter.candidate_cuts
    with pytest.raises(ValueError, match="has no"):
        registry.get("keras_app", application="NoSuchNet").build(0)

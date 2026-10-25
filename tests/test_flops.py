from __future__ import annotations

from typing import Any

from src.models import flops


def test_alexnet_conv1_and_fc_counts() -> None:
    conv1 = flops.conv_count((11, 11), 3, 96, (55, 55, 96))
    assert conv1.macs == 55 * 55 * 96 * 121 * 3 == 105_415_200
    assert conv1.flops == 2 * conv1.macs
    grouped = flops.conv_count((5, 5), 96, 256, (27, 27, 256), groups=2)
    assert grouped.macs == 27 * 27 * 256 * 25 * 48
    fc1 = flops.dense_count(9216, 4096)
    assert fc1.macs == 37_748_736
    pool = flops.pool_count((3, 3), (27, 27, 96))
    assert pool.macs == 0 and pool.flops == 27 * 27 * 96 * 9
    gru = flops.gru_count(5, 32, 5)
    assert gru.macs == 3 * (5 * 32 + 32 * 32) * 5


class _Layer:
    def __init__(self, cls: str, **config: Any) -> None:
        self._cls = cls
        self._config = config

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)


def make(cls: str, **config: Any) -> Any:
    layer = _Layer(cls, **config)
    layer.__class__ = type(cls, (_Layer,), {})
    return layer


def test_keras_layer_count_by_class_name() -> None:
    conv = flops.keras_layer_count(
        make("Conv2D", kernel_size=(3, 3), filters=64, groups=1), (224, 224, 3), (224, 224, 64)
    )
    assert conv.macs == 224 * 224 * 64 * 9 * 3
    dense = flops.keras_layer_count(make("Dense", units=10), (4096,), (10,))
    assert dense.flops == 2 * 4096 * 10
    pool = flops.keras_layer_count(
        make("MaxPooling2D", pool_size=(2, 2)), (224, 224, 64), (112, 112, 64)
    )
    assert pool.flops == 112 * 112 * 64 * 4
    assert flops.keras_layer_count(make("Flatten"), (6, 6, 256), (9216,)) == flops.ZERO
    gap = flops.keras_layer_count(make("GlobalAveragePooling2D"), (7, 7, 2048), (2048,))
    assert gap.flops == 7 * 7 * 2048
    sep = flops.keras_layer_count(
        make("SeparableConv2D", kernel_size=(3, 3), filters=16, depth_multiplier=1),
        (10, 10, 8),
        (10, 10, 16),
    )
    assert sep.macs == 10 * 10 * 8 * 9 + 10 * 10 * 8 * 16

"""VGG-16 as eight stages: five convolutional blocks and three dense layers.

Built from the Keras definition (``tf.keras.applications.VGG16`` layer
structure) with random weights and a 224 x 224 x 3 input. Partition points
after each block correspond to the pooled feature maps.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from src.models.registry import register
from src.models.tf.keras_stages import SequentialStageAdapter

L = tf.keras.layers


def _block(name: str, filters: int, convs: int) -> list[Any]:
    layers: list[Any] = [
        L.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv{i}")
        for i in range(1, convs + 1)
    ]
    layers.append(L.MaxPooling2D(2, strides=2, name=f"{name}_pool"))
    return layers


class VGG16Adapter(SequentialStageAdapter):
    name = "vgg16"
    input_shape: tuple[int, ...] = (224, 224, 3)
    PARAMETERS = {
        "type": "object",
        "properties": {
            "num_classes": {
                "type": "integer",
                "minimum": 2,
                "default": 1000,
                "description": "size of the classifier output",
            }
        },
    }

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.num_classes = num_classes

    def define_stages(self) -> list[tuple[str, list[Any]]]:
        return [
            ("block1", _block("block1", 64, 2)),
            ("block2", _block("block2", 128, 2)),
            ("block3", _block("block3", 256, 3)),
            ("block4", _block("block4", 512, 3)),
            ("block5", _block("block5", 512, 3)),
            ("fc1", [L.Flatten(name="flatten"), L.Dense(4096, activation="relu", name="fc1")]),
            ("fc2", [L.Dense(4096, activation="relu", name="fc2")]),
            ("predictions", [L.Dense(self.num_classes, activation="softmax", name="predictions")]),
        ]


ADAPTERS = {"vgg16": VGG16Adapter}
register("vgg16", VGG16Adapter)

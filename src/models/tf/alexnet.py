"""AlexNet with the 11 stages of Chapter 4.

Single-tower AlexNet on a 227 x 227 x 3 input with valid padding on the
first convolution, so that the stage outputs are exactly those of the Chapter 4
tables: Conv1 55x55x96, MaxPool1 27x27x96, Conv2 27x27x256, MaxPool2
13x13x256, Conv3 13x13x384, Conv4 13x13x384, Conv5 13x13x256, MaxPool3
6x6x256, FC1 4096, FC2 4096, FC3 1000 (softmax). ReLU is folded into the
convolution and dense stages; dropout acts only in training and is omitted.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from src.models.registry import register
from src.models.tf.keras_stages import SequentialStageAdapter

L = tf.keras.layers


class AlexNetAdapter(SequentialStageAdapter):
    name = "alexnet"
    input_shape: tuple[int, ...] = (227, 227, 3)

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.num_classes = num_classes

    def define_stages(self) -> list[tuple[str, list[Any]]]:
        return [
            (
                "conv1",
                [L.Conv2D(96, 11, strides=4, padding="valid", activation="relu", name="conv1")],
            ),
            ("pool1", [L.MaxPooling2D(3, strides=2, padding="valid", name="pool1")]),
            ("conv2", [L.Conv2D(256, 5, padding="same", activation="relu", name="conv2")]),
            ("pool2", [L.MaxPooling2D(3, strides=2, padding="valid", name="pool2")]),
            ("conv3", [L.Conv2D(384, 3, padding="same", activation="relu", name="conv3")]),
            ("conv4", [L.Conv2D(384, 3, padding="same", activation="relu", name="conv4")]),
            ("conv5", [L.Conv2D(256, 3, padding="same", activation="relu", name="conv5")]),
            ("pool3", [L.MaxPooling2D(3, strides=2, padding="valid", name="pool3")]),
            ("fc1", [L.Flatten(name="flatten"), L.Dense(4096, activation="relu", name="fc1")]),
            ("fc2", [L.Dense(4096, activation="relu", name="fc2")]),
            ("fc3", [L.Dense(self.num_classes, activation="softmax", name="fc3")]),
        ]


ADAPTERS = {"alexnet": AlexNetAdapter}
register("alexnet", AlexNetAdapter)

"""ResNet-50 as six stages, cut at the residual-stage boundaries.

The functional Keras model (``tf.keras.applications.ResNet50`` with random
weights, 224 x 224 x 3 input) is cut after the stem (``pool1_pool``), after
each of the four residual stages (``conv2_block3_out`` ... ``conv5_block3_out``)
and at the classifier (``predictions``). Residual connections never cross a
cut, so each stage is a self-contained sub-graph.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from src.models.registry import register
from src.models.tf.keras_stages import FunctionalCutAdapter


class ResNet50Adapter(FunctionalCutAdapter):
    name = "resnet50"
    input_shape: tuple[int, ...] = (224, 224, 3)
    cuts = (
        "pool1_pool",
        "conv2_block3_out",
        "conv3_block4_out",
        "conv4_block6_out",
        "conv5_block3_out",
        "predictions",
    )

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.num_classes = num_classes

    def define_model(self) -> Any:
        return tf.keras.applications.ResNet50(
            weights=None, input_shape=self.input_shape, classes=self.num_classes
        )


ADAPTERS = {"resnet50": ResNet50Adapter}
register("resnet50", ResNet50Adapter)

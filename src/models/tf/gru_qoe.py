"""The QoE predictor of the Chapter 4 case study: GRU, attention block, output.

Input: a window of 5 measurement intervals with 5 features each (throughput,
packets lost, loss rate, jitter, speed). Stage 1 is a 32-unit GRU whose final
hidden state (32 floats, 128 bytes) is the compact tensor transmitted when
the model is split after the recurrent layer; stage 2 is the 128-unit
attention block; stage 3 the output layer (the predicted VMAF score).
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from src.models.registry import register
from src.models.tf.keras_stages import SequentialStageAdapter

L = tf.keras.layers


class GruQoeAdapter(SequentialStageAdapter):
    name = "gru_qoe"
    input_shape: tuple[int, ...] = (5, 5)

    def __init__(self, units: int = 32, attention_units: int = 128) -> None:
        super().__init__()
        self.units = units
        self.attention_units = attention_units

    def define_stages(self) -> list[tuple[str, list[Any]]]:
        return [
            ("gru", [L.GRU(self.units, name="gru")]),
            (
                "attention",
                [
                    L.Dense(self.attention_units, activation="tanh", name="attention_dense"),
                    L.Dense(self.attention_units, activation="softmax", name="attention_weights"),
                ],
            ),
            ("output", [L.Dense(1, name="vmaf")]),
        ]


ADAPTERS = {"gru_qoe": GruQoeAdapter}
register("gru_qoe", GruQoeAdapter)

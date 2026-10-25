"""Analytical operation counts per layer, from layer configuration and shapes.

Convention: one multiply-accumulate (MAC) is two floating-point operations
(FLOPs = 2 x MACs). Convolutions count ``H_out x W_out x C_out x k_h x k_w x
C_in / groups`` MACs, dense layers ``in x out``, pooling ``output elements x
kernel elements`` FLOPs (comparisons or additions) and zero MACs; activations,
normalisation, flatten, dropout and additions of residual branches are not
counted. The Chapter 4 AlexNet table (``reference/alexnet_ch4_table.json``)
mixes conventions and is kept as reference data rather than reproduced.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpCount:
    macs: int
    flops: int

    def __add__(self, other: OpCount) -> OpCount:
        return OpCount(self.macs + other.macs, self.flops + other.flops)


ZERO = OpCount(0, 0)


def _spatial(shape: Sequence[int]) -> tuple[int, ...]:
    """Spatial dims of a channels-last shape without batch: (H, W, C) -> (H, W)."""
    return tuple(int(d) for d in shape[:-1])


def conv_count(
    kernel: Sequence[int],
    in_channels: int,
    out_channels: int,
    output_shape: Sequence[int],
    groups: int = 1,
) -> OpCount:
    spatial = math.prod(_spatial(output_shape))
    macs = spatial * out_channels * math.prod(kernel) * in_channels // groups
    return OpCount(macs, 2 * macs)


def depthwise_conv_count(
    kernel: Sequence[int], channels: int, output_shape: Sequence[int], multiplier: int = 1
) -> OpCount:
    spatial = math.prod(_spatial(output_shape))
    macs = spatial * channels * multiplier * math.prod(kernel)
    return OpCount(macs, 2 * macs)


def dense_count(in_features: int, out_features: int) -> OpCount:
    macs = in_features * out_features
    return OpCount(macs, 2 * macs)


def pool_count(kernel: Sequence[int], output_shape: Sequence[int]) -> OpCount:
    return OpCount(0, math.prod(int(d) for d in output_shape) * math.prod(kernel))


def gru_count(input_features: int, units: int, timesteps: int) -> OpCount:
    """Three gates, each a dense map of the input and of the state, per timestep."""
    per_step = 3 * (input_features * units + units * units)
    macs = per_step * timesteps
    return OpCount(macs, 2 * macs)


def keras_layer_count(
    layer: Any, input_shape: Sequence[int], output_shape: Sequence[int]
) -> OpCount:
    """Operation count of a Keras layer from its configuration and the shapes it sees
    (shapes without the batch dimension)."""
    cls = type(layer).__name__
    config = layer.get_config()
    if cls == "Conv2D":
        return conv_count(
            config["kernel_size"],
            int(input_shape[-1]),
            config["filters"],
            output_shape,
            config.get("groups", 1),
        )
    if cls == "DepthwiseConv2D":
        return depthwise_conv_count(
            config["kernel_size"],
            int(input_shape[-1]),
            output_shape,
            config.get("depth_multiplier", 1),
        )
    if cls == "SeparableConv2D":
        depthwise = depthwise_conv_count(
            config["kernel_size"],
            int(input_shape[-1]),
            output_shape,
            config.get("depth_multiplier", 1),
        )
        pointwise = conv_count(
            (1, 1),
            int(input_shape[-1]) * config.get("depth_multiplier", 1),
            config["filters"],
            output_shape,
        )
        return depthwise + pointwise
    if cls == "Dense":
        return dense_count(int(input_shape[-1]), config["units"])
    if cls in ("MaxPooling2D", "AveragePooling2D"):
        return pool_count(config["pool_size"], output_shape)
    if cls in ("GlobalAveragePooling2D", "GlobalMaxPooling2D"):
        return OpCount(0, math.prod(int(d) for d in input_shape))
    if cls == "GRU":
        return gru_count(int(input_shape[-1]), config["units"], int(input_shape[0]))
    return ZERO

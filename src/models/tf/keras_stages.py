"""Base classes for Keras models exposed as stages.

:class:`SequentialStageAdapter` holds an ordered list of named stages, each a
list of Keras layers applied in order (AlexNet, VGG-16, the GRU predictor).
:class:`FunctionalCutAdapter` wraps a functional Keras model (ResNet-50) and
cuts it at named layers; each stage is the sub-graph between two cuts.

Both run on the CPU only, count operations analytically from the layer
configuration and the shapes seen during a tracing pass, and store weights as
NumPy ``.npz`` files (framework-independent, deterministic).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # see src/models/tf/__init__.py
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only

import tensorflow as tf

from src.models.flops import OpCount, keras_layer_count
from src.models.registry import Array, StageInfo, deserialize_array, serialize_array

tf.config.set_visible_devices([], "GPU")


def _no_batch(shape: Any) -> tuple[int, ...]:
    dims = list(shape)
    return tuple(int(d) for d in dims[1:])


class _WeightsMixin:
    _layers_with_weights: list[Any]

    def save_weights(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for i, weight in enumerate(self._all_weights()):
            arrays[f"w{i}"] = weight.numpy()
        np.savez(path, **arrays)

    def load_weights(self, path: Path) -> None:
        self._ensure_built()
        data = np.load(path)
        weights = self._all_weights()
        if len(data.files) != len(weights):
            raise ValueError(f"{path}: {len(data.files)} arrays, model has {len(weights)} weights")
        for i, weight in enumerate(weights):
            weight.assign(data[f"w{i}"])

    def _all_weights(self) -> list[Any]:
        raise NotImplementedError

    def _ensure_built(self) -> None:
        raise NotImplementedError

    def serialize(self, array: Array) -> bytes:
        return serialize_array(array)

    def deserialize(self, data: bytes) -> Array:
        return deserialize_array(data)


class SequentialStageAdapter(_WeightsMixin):
    """Stages = named lists of Keras layers applied in sequence."""

    name = "sequential"
    framework = "tensorflow"
    input_shape: tuple[int, ...] = ()

    def __init__(self) -> None:
        self._stage_layers: list[tuple[str, list[Any]]] = []
        self._stages: list[StageInfo] | None = None

    def define_stages(self) -> list[tuple[str, list[Any]]]:
        raise NotImplementedError

    def _ensure_built(self) -> None:
        if not self._stage_layers:
            self.build(0)

    def build(self, seed: int) -> None:
        tf.keras.utils.set_random_seed(seed)
        self._stage_layers = self.define_stages()
        self._stages = None
        # Trace once to create the weights and record shapes and operation counts.
        x = tf.zeros((1, *self.input_shape), dtype=tf.float32)
        stages: list[StageInfo] = []
        for index, (stage_name, layers) in enumerate(self._stage_layers, start=1):
            count = OpCount(0, 0)
            for layer in layers:
                before = _no_batch(x.shape)
                x = layer(x)
                count = count + keras_layer_count(layer, before, _no_batch(x.shape))
            shape = _no_batch(x.shape)
            stages.append(
                StageInfo(
                    index, stage_name, count.flops, count.macs, shape, int(np.prod(shape)) * 4
                )
            )
        self._stages = stages

    @property
    def stages(self) -> list[StageInfo]:
        self._ensure_built()
        assert self._stages is not None
        return self._stages

    @property
    def input_bytes(self) -> int:
        return int(np.prod(self.input_shape)) * 4

    def make_input(self, seed: int) -> Array:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 1.0, size=(1, *self.input_shape)).astype(np.float32)

    def run_stages(self, x: Array, start: int, stop: int) -> Array:
        self._ensure_built()
        n = len(self._stage_layers)
        if not 1 <= start <= stop <= n:
            raise ValueError(f"stage range {start}..{stop} outside 1..{n}")
        tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        for index in range(start, stop + 1):
            for layer in self._stage_layers[index - 1][1]:
                tensor = layer(tensor, training=False)
        return np.asarray(tensor.numpy())

    def _all_weights(self) -> list[Any]:
        weights: list[Any] = []
        for _, layers in self._stage_layers:
            for layer in layers:
                weights.extend(layer.weights)
        return weights


class FunctionalCutAdapter(_WeightsMixin):
    """A functional Keras model cut at named layers; stage i ends at ``cuts[i-1]``."""

    name = "functional"
    framework = "tensorflow"
    input_shape: tuple[int, ...] = ()
    cuts: Sequence[str] = ()  # layer names, the last one is the model output

    def __init__(self) -> None:
        self._model: Any = None
        self._stages: list[StageInfo] | None = None
        self._submodels: dict[tuple[int, int], Any] = {}

    def define_model(self) -> Any:
        raise NotImplementedError

    def _ensure_built(self) -> None:
        if self._model is None:
            self.build(0)

    def build(self, seed: int) -> None:
        tf.keras.utils.set_random_seed(seed)
        self._model = self.define_model()
        self._submodels = {}
        self._model(tf.zeros((1, *self.input_shape), dtype=tf.float32), training=False)
        cut_index = {name: i for i, name in enumerate(self.cuts)}
        counts = [OpCount(0, 0) for _ in self.cuts]
        current = 0
        for layer in self._model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            in_shape = (
                layer.input_shape if isinstance(layer.input_shape, tuple) else layer.input_shape[0]
            )
            out_shape = (
                layer.output_shape
                if isinstance(layer.output_shape, tuple)
                else layer.output_shape[0]
            )
            counts[current] = counts[current] + keras_layer_count(
                layer, _no_batch(in_shape), _no_batch(out_shape)
            )
            if layer.name in cut_index:
                current = min(cut_index[layer.name] + 1, len(self.cuts) - 1)
        stages = []
        for index, name in enumerate(self.cuts, start=1):
            out_shape = self._model.get_layer(name).output_shape
            shape = _no_batch(out_shape if isinstance(out_shape, tuple) else out_shape[0])
            stages.append(
                StageInfo(
                    index,
                    name,
                    counts[index - 1].flops,
                    counts[index - 1].macs,
                    shape,
                    int(np.prod(shape)) * 4,
                )
            )
        self._stages = stages

    @property
    def stages(self) -> list[StageInfo]:
        self._ensure_built()
        assert self._stages is not None
        return self._stages

    @property
    def input_bytes(self) -> int:
        return int(np.prod(self.input_shape)) * 4

    def make_input(self, seed: int) -> Array:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 1.0, size=(1, *self.input_shape)).astype(np.float32)

    def _submodel(self, start: int, stop: int) -> Any:
        key = (start, stop)
        if key not in self._submodels:
            inputs = (
                self._model.input
                if start == 1
                else self._model.get_layer(self.cuts[start - 2]).output
            )
            outputs = self._model.get_layer(self.cuts[stop - 1]).output
            self._submodels[key] = tf.keras.Model(inputs=inputs, outputs=outputs)
        return self._submodels[key]

    def run_stages(self, x: Array, start: int, stop: int) -> Array:
        self._ensure_built()
        n = len(self.cuts)
        if not 1 <= start <= stop <= n:
            raise ValueError(f"stage range {start}..{stop} outside 1..{n}")
        tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        return np.asarray(self._submodel(start, stop)(tensor, training=False).numpy())

    def _all_weights(self) -> list[Any]:
        return list(self._model.weights)

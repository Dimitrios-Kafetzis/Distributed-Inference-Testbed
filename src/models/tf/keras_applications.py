"""Any ``tf.keras.applications`` classifier as a stage adapter, cut automatically.

A functional Keras graph with skip connections (MobileNetV2, EfficientNet,
DenseNet) can be cut only where a single tensor carries everything the rest of
the graph needs. :func:`find_single_tensor_cuts` walks the graph in its
topological order and keeps the layers whose output is the only tensor
crossing from the layers before them to the layers after them; the stages are
the sub-graphs between consecutive cuts. Because such models have dozens of
candidate cuts, the candidates are thinned to ``max_stages`` boundaries placed
at roughly equal shares of the model's multiply-accumulate operations, which
is what the partition search needs: a handful of stages of comparable work.

Registered names: ``mobilenetv2``, ``efficientnetb0``, ``densenet121``
(``resnet50`` keeps its explicit cuts at the residual-stage boundaries). Any
other application can be used through ``keras_app`` with the ``application``
parameter. Weights are random with the scenario's seed unless the ImageNet
weights were fetched into the model catalogue (``src/models/weights.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import tensorflow as tf

from src.models.flops import OpCount, keras_layer_count
from src.models.registry import register
from src.models.tf.keras_stages import FunctionalCutAdapter, _no_batch

# Registered name -> Keras application class name and its default input size.
APPLICATIONS: dict[str, tuple[str, int]] = {
    "mobilenetv2": ("MobileNetV2", 224),
    "efficientnetb0": ("EfficientNetB0", 224),
    "densenet121": ("DenseNet121", 224),
    "resnet50v2": ("ResNet50V2", 224),
    "inceptionv3": ("InceptionV3", 299),
    "nasnetmobile": ("NASNetMobile", 224),
    "xception": ("Xception", 299),
}
DEFAULT_MAX_STAGES = 12


def _inbound_layers(layer: Any) -> list[Any]:
    """The layers whose outputs feed ``layer`` (functional graph, one call per layer)."""
    producers: list[Any] = []
    for node in getattr(layer, "_inbound_nodes", []):
        inbound = node.inbound_layers
        inbound = inbound if isinstance(inbound, list | tuple) else [inbound]
        for producer in inbound:
            if producer not in producers:
                producers.append(producer)
    return producers


def find_single_tensor_cuts(model: Any) -> list[str]:
    """Layer names after which exactly one tensor, that layer's output, crosses to
    the rest of the graph. The last layer is always a cut (the model output)."""
    layers = [layer for layer in model.layers if not isinstance(layer, tf.keras.layers.InputLayer)]
    index = {id(layer): i for i, layer in enumerate(layers)}
    consumers: dict[int, list[int]] = {i: [] for i in range(len(layers))}
    for i, layer in enumerate(layers):
        for producer in _inbound_layers(layer):
            if id(producer) in index:
                consumers[index[id(producer)]].append(i)
    cuts: list[str] = []
    for i, layer in enumerate(layers[:-1]):
        if len(getattr(layer, "output_shape", None) or []) and isinstance(layer.output_shape, list):
            continue  # a layer with several outputs is never a cut
        crossing = {p for p in range(i + 1) if any(c > i for c in consumers[p])}
        if crossing == {i}:
            cuts.append(layer.name)
    cuts.append(layers[-1].name)
    return cuts


def layer_macs(model: Any) -> dict[str, int]:
    """Multiply-accumulates per layer name, from the shapes of the traced graph."""
    counts: dict[str, int] = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        in_shape = (
            layer.input_shape if isinstance(layer.input_shape, tuple) else layer.input_shape[0]
        )
        out_shape = (
            layer.output_shape if isinstance(layer.output_shape, tuple) else layer.output_shape[0]
        )
        count: OpCount = keras_layer_count(layer, _no_batch(in_shape), _no_batch(out_shape))
        counts[layer.name] = count.macs
    return counts


def thin_cuts(model: Any, candidates: Sequence[str], max_stages: int) -> list[str]:
    """Keep at most ``max_stages`` cuts: the last one, and the candidates closest to
    equal shares of the cumulative multiply-accumulate count."""
    if len(candidates) <= max_stages:
        return list(candidates)
    macs = layer_macs(model)
    order = [layer.name for layer in model.layers]
    cumulative: dict[str, int] = {}
    running = 0
    for name in order:
        running += macs.get(name, 0)
        cumulative[name] = running
    total = running or 1
    inner = [c for c in candidates[:-1]]
    chosen: list[str] = []
    for k in range(1, max_stages):
        target = total * k / max_stages
        best = min(inner, key=lambda name: abs(cumulative[name] - target))
        if best not in chosen:
            chosen.append(best)
    chosen.sort(key=order.index)
    return [*chosen, candidates[-1]]


class KerasApplicationsAdapter(FunctionalCutAdapter):
    """A ``tf.keras.applications`` model cut at automatically found boundaries."""

    name = "keras_app"
    application = "MobileNetV2"
    default_input_size = 224
    PARAMETERS = {
        "type": "object",
        "properties": {
            "application": {
                "type": "string",
                "description": "tf.keras.applications class name (MobileNetV2, EfficientNetB0, "
                "DenseNet121, InceptionV3, Xception, ...)",
            },
            "num_classes": {"type": "integer", "minimum": 2, "default": 1000},
            "input_size": {
                "type": "integer",
                "minimum": 32,
                "description": "side of the square input (the application's default size "
                "when absent)",
            },
            "max_stages": {
                "type": "integer",
                "minimum": 2,
                "default": DEFAULT_MAX_STAGES,
                "description": "at most this many stages of comparable work",
            },
        },
    }

    def __init__(
        self,
        application: str | None = None,
        num_classes: int = 1000,
        input_size: int | None = None,
        max_stages: int = DEFAULT_MAX_STAGES,
    ) -> None:
        super().__init__()
        if application:
            self.application = application
        self.num_classes = num_classes
        size = input_size or self.default_input_size
        self.input_shape = (size, size, 3)
        self.max_stages = max_stages
        self.candidate_cuts: list[str] = []

    def define_model(self) -> Any:
        factory = getattr(tf.keras.applications, self.application, None)
        if factory is None:
            raise ValueError(f"tf.keras.applications has no {self.application!r}")
        return factory(weights=None, input_shape=self.input_shape, classes=self.num_classes)

    def select_cuts(self, model: Any) -> Sequence[str]:
        self.candidate_cuts = find_single_tensor_cuts(model)
        return thin_cuts(model, self.candidate_cuts, self.max_stages)

    def make_input(self, seed: int) -> Any:
        rng = np.random.default_rng(seed)
        # Keras applications expect pixel values; their preprocessing is inside the graph
        # (EfficientNet) or expects [-1, 1] (MobileNetV2): uniform [0, 1) is a safe input.
        return rng.uniform(0.0, 1.0, size=(1, *self.input_shape)).astype(np.float32)


def _named(registered: str, class_name: str, size: int) -> type[KerasApplicationsAdapter]:
    return type(
        f"{class_name}Adapter",
        (KerasApplicationsAdapter,),
        {"name": registered, "application": class_name, "default_input_size": size},
    )


ADAPTERS: dict[str, Any] = {"keras_app": KerasApplicationsAdapter}
for _registered, (_class_name, _size) in APPLICATIONS.items():
    ADAPTERS[_registered] = _named(_registered, _class_name, _size)
for _registered, _factory in ADAPTERS.items():
    register(_registered, _factory)

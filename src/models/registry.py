"""Model adapters: the interface every model implements, and the name registry.

A model is a sequence of S stages. Partition point s in {0, ..., S} means that
stages 1..s run on the device and s+1..S on the edge server; s = 0 sends the
raw input, s = S runs everything locally. Each stage reports its FLOPs
(2 x multiply-accumulates), its MACs and the size of its output, so that the
delay model has per-stage FLOP-time pairs and transmitted data sizes.

Adapters register under a name. Third-party adapters can be added without
changing the package: list their modules in the environment variable
``TESTBED_MODEL_MODULES`` (comma-separated) or expose them through the
``testbed.models`` entry-point group. An adapter class may carry
``PARAMETERS``, a JSON Schema of its constructor arguments: a scenario passes
them as ``model_args`` and NNPM renders them as form fields.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]
ENTRY_POINT_GROUP = "testbed.models"
MODULES_ENV = "TESTBED_MODEL_MODULES"


@dataclass(frozen=True)
class StageInfo:
    index: int  # 1-based
    name: str
    flops: int
    macs: int
    output_shape: tuple[int, ...]  # without the batch dimension
    output_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class ModelAdapter(Protocol):
    name: str
    framework: str  # "tensorflow" | "torch" | "fake"
    input_shape: tuple[int, ...]  # without the batch dimension

    @property
    def stages(self) -> list[StageInfo]: ...

    @property
    def input_bytes(self) -> int: ...

    def build(self, seed: int) -> None:
        """Create the model with deterministic weights."""
        ...

    def load_weights(self, path: Path) -> None: ...

    def save_weights(self, path: Path) -> None: ...

    def make_input(self, seed: int) -> Array:
        """A deterministic synthetic input (batch of one)."""
        ...

    def run_stages(self, x: Array, start: int, stop: int) -> Array:
        """Run stages ``start``..``stop`` (1-based, inclusive) on ``x``."""
        ...

    def serialize(self, array: Array) -> bytes: ...

    def deserialize(self, data: bytes) -> Array: ...


def serialize_array(array: Array) -> bytes:
    """NumPy ``.npy`` bytes: shape and dtype travel with the data."""
    import io

    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def deserialize_array(data: bytes) -> Array:
    import io

    return np.asarray(np.load(io.BytesIO(data), allow_pickle=False))


Factory = Callable[..., ModelAdapter]
_REGISTRY: dict[str, Factory] = {}
_LOADED_PLUGINS = False


def register(name: str, factory: Factory) -> None:
    if name in _REGISTRY and _REGISTRY[name] is not factory:
        raise ValueError(f"model {name!r} is already registered")
    _REGISTRY[name] = factory


BUILTIN_MODULES = (
    "src.models.fake",
    "src.models.tf.alexnet",
    "src.models.tf.vgg16",
    "src.models.tf.resnet50",
    "src.models.tf.gru_qoe",
    "src.models.torch.gpt2",
    "src.models.torch.tinyllama",
)


BUILTIN_NAMES = {
    "fake": "src.models.fake",
    "alexnet": "src.models.tf.alexnet",
    "vgg16": "src.models.tf.vgg16",
    "resnet50": "src.models.tf.resnet50",
    "gru_qoe": "src.models.tf.gru_qoe",
    "gpt2": "src.models.torch.gpt2",
    "tinyllama": "src.models.torch.tinyllama",
}


def _load_module(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        # TensorFlow or PyTorch may be absent on this node (DNN-only or LLM-only device).
        return
    for name, factory in getattr(module, "ADAPTERS", {}).items():
        register(name, factory)


def _load_builtin() -> None:
    """Register the built-in adapters from each module's ``ADAPTERS`` table."""
    for module_name in BUILTIN_MODULES:
        _load_module(module_name)


def _load_plugins() -> None:
    global _LOADED_PLUGINS
    if _LOADED_PLUGINS:
        return
    _LOADED_PLUGINS = True
    _load_builtin()
    for module in filter(None, os.environ.get(MODULES_ENV, "").split(",")):
        importlib.import_module(module.strip())
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        factory = ep.load()
        register(ep.name, factory)


def names() -> list[str]:
    _load_plugins()
    return sorted(_REGISTRY)


def get(name: str, **kwargs: Any) -> ModelAdapter:
    """The adapter registered under ``name``; built-in modules are imported on demand,
    so asking for the fake model does not import TensorFlow."""
    if name not in _REGISTRY and name in BUILTIN_NAMES:
        _load_module(BUILTIN_NAMES[name])
    if name not in _REGISTRY:
        _load_plugins()
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown model {name!r}; registered: {names()}") from exc
    return factory(**kwargs)


def iter_registered() -> Iterator[tuple[str, Factory]]:
    _load_plugins()
    yield from sorted(_REGISTRY.items())


def parameter_schema(name: str) -> dict[str, Any]:
    """The JSON Schema of the adapter's constructor arguments (``PARAMETERS`` on the
    adapter class), or ``{}`` when the adapter takes none. A scenario's ``model_args``
    block is validated against it and NNPM renders a form from it."""
    if name not in _REGISTRY and name in BUILTIN_NAMES:
        _load_module(BUILTIN_NAMES[name])
    if name not in _REGISTRY:
        _load_plugins()
    factory = _REGISTRY.get(name)
    schema = getattr(factory, "PARAMETERS", None)
    return dict(schema) if isinstance(schema, dict) else {}


def parameter_schemas() -> dict[str, dict[str, Any]]:
    """``{model name: parameter schema}`` for every registered model with parameters."""
    out = {}
    for name, factory in iter_registered():
        schema = getattr(factory, "PARAMETERS", None)
        if isinstance(schema, dict) and schema.get("properties"):
            out[name] = dict(schema)
    return out


def validate_model_args(name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Check ``args`` against the adapter's parameter schema; unknown keys are refused
    even for adapters without a schema (the factory would reject them later, on the
    device, where the error is hard to see)."""
    import jsonschema

    args = dict(args or {})
    if not args:
        return args
    schema = parameter_schema(name)
    if not schema:
        raise ValueError(f"model {name!r} takes no model_args (given: {sorted(args)})")
    schema = {"additionalProperties": False, **schema}
    try:
        jsonschema.validate(args, schema)
    except jsonschema.ValidationError as exc:
        location = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise ValueError(f"model_args for {name!r} invalid at {location}: {exc.message}") from exc
    return args


def parse_model_args(text: str | None) -> dict[str, Any]:
    """The ``--model-args`` command-line value: a JSON object, or empty."""
    import json

    if not text or not text.strip():
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("--model-args must be a JSON object")
    return value


def reset_for_tests() -> None:
    """Forget plug-ins loaded so far (tests register toy adapters)."""
    global _LOADED_PLUGINS
    _LOADED_PLUGINS = False
    _REGISTRY.clear()

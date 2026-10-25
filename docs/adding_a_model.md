# Adding a model

A model is an **adapter**: a sequence of S stages with, per stage, its operation count and the size of its output, plus the means to run stages 1..s and s+1..S separately. Partition point s means that stages 1..s run on the device and s+1..S on the edge server (s = 0 sends the raw input, s = S runs everything locally). Everything else in the testbed, the profiler, the partitioned-inference service and client, the aggregation and the optimiser, works on that interface (`src/models/registry.py`, `ModelAdapter`).

Three ways to add a model, from cheapest to most general:

## 1. A name and a parameter

Many models need no code at all:

- **Any `tf.keras.applications` classifier**: `model: keras_app` with `model_args: {application: InceptionV3}`; or one of the registered names `mobilenetv2`, `efficientnetb0`, `densenet121`, `resnet50v2`, `inceptionv3`, `nasnetmobile`, `xception`. The adapter (`src/models/tf/keras_applications.py`) finds the cut points automatically: the layers whose output is the only tensor crossing from the layers before them to the layers after them, thinned to `max_stages` boundaries at roughly equal shares of the multiply-accumulate operations. `num_classes` and `input_size` are parameters.
- **Any Hugging Face decoder-only model** whose family is in the family table (GPT-2, Llama, Mistral, Qwen2, Gemma): `model: hf_decoder` with `model_args: {checkpoint: <directory on the device or model name in its cache>, seq_len: 64}`, or without a checkpoint `model_args: {config: {model_type: llama, num_hidden_layers: 4, ...}}` for random weights. `llama32_1b` and `qwen25_05b` are registered with their published configurations, like `gpt2` and `tinyllama`.

Model parameters are listed in the catalogue (`GET /api/catalog`, `model_parameters`) and rendered by the Create scenario dialog under the model drop-down. They are validated against the adapter's `PARAMETERS` schema when a run starts and passed to every device tool as `--model-args`.

## 2. A subclass of a base adapter

When the model is a sequence of Keras layer groups, subclass `SequentialStageAdapter` (`src/models/tf/keras_stages.py`) and return the stages:

```python
"""MyNet: four stages."""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from src.models.registry import register
from src.models.tf.keras_stages import SequentialStageAdapter

L = tf.keras.layers


class MyNetAdapter(SequentialStageAdapter):
    name = "mynet"
    input_shape: tuple[int, ...] = (64, 64, 3)
    PARAMETERS = {
        "type": "object",
        "properties": {"num_classes": {"type": "integer", "minimum": 2, "default": 10}},
    }

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes

    def define_stages(self) -> list[tuple[str, list[Any]]]:
        return [
            ("conv1", [L.Conv2D(16, 3, activation="relu", name="conv1")]),
            ("pool1", [L.MaxPooling2D(2, name="pool1")]),
            ("conv2", [L.Conv2D(32, 3, activation="relu", name="conv2")]),
            ("fc", [L.Flatten(), L.Dense(self.num_classes, activation="softmax", name="fc")]),
        ]


ADAPTERS = {"mynet": MyNetAdapter}
register("mynet", MyNetAdapter)
```

The base class traces the model once to record shapes and operation counts (`src/models/flops.py` knows convolutions, depthwise and separable convolutions, dense and recurrent layers, pooling, normalisation and the element-wise layers), stores weights as `.npz`, and implements `run_stages`.

- A functional graph with skip connections: subclass `FunctionalCutAdapter`, return the model from `define_model()` and name the cut layers in `cuts` (ResNet-50 does this), or override `select_cuts(model)` to compute them (`keras_applications.py` does this).
- A decoder-only PyTorch model of a known family: subclass `DecoderStageAdapter` (`src/models/torch/decoder.py`), set `family` and return the Hugging Face configuration from `hf_config()` (`gpt2.py`, `tinyllama.py`, `hf_decoder.py`). A new family is one more row of `FAMILIES`: attribute paths of the layers, the attention and feed-forward modules, the norms and the embeddings, and a few structural flags (fused projections, rotary positions, gated feed-forward block, scaled embeddings). The head-wise decomposition and the hook profiler follow the table.

## 3. An adapter from scratch

Implement the protocol directly when the model is neither (`src/models/fake.py` is the smallest complete example): `name`, `framework`, `input_shape`, `stages`, `input_bytes`, `build(seed)`, `load_weights`, `save_weights`, `make_input(seed)`, `run_stages(x, start, stop)`, `serialize`, `deserialize`. `run_stages` must be exact: stages 1..s then s+1..S must equal 1..S on the same input (float tolerance), which is what the profiler and the partitioned-inference measurements assume and what `python -m src.models.inference verify` checks.

## Where to put it and how it reaches the devices

- Inside the repository: a module under `src/models/tf/` or `src/models/torch/` listed in `BUILTIN_MODULES` of `src/models/registry.py`. The repository is synchronised to every device at provisioning, so the adapter is available there.
- Outside the repository: a module importable on the edge server and on the devices, named in `TESTBED_MODEL_MODULES` (comma-separated), or an installed package with a `testbed.models` entry point. The devices need the module too: the simplest way is a directory synchronised with the code.

## Weights and inputs

- Random weights with the scenario's seed are the default (`train_and_run_benchmark.sh train` prepares them on the device). Real weights go through the model catalogue (`<bs_models_dir>/models.yaml`, example in `examples/models.yaml`; `python -m src.models.weights fetch NAME --allow-upstream` obtains them, `verify` checks the checksums, `build_mirror.sh models` puts them in the mirror). Provisioning copies `bs_models_dir` to the devices; NNPM's Runtimes dialog distributes a checkpoint on demand.
- Inputs: the scenario's `input` block (`synthetic`, `numpy_file`, `image_file`, `image_folder`, `text_prompt`); see [scenario_reference.md](scenario_reference.md). Image sources resize to the model's input; an image folder with class sub-directories gives a label, so `verify` reports top-1 accuracy as well.

## Tests

Every adapter has a test that builds a small configuration (no download), asserts the stage shapes and the partition equivalence at every cut, and round-trips the weights (`tests/test_tf_models.py`, `tests/test_torch_models.py`). Mark tests that build real models `slow`; the fast suite must not import TensorFlow or PyTorch.

"""A fake model with configurable stage costs, for tests and local mode.

Each stage sleeps for a configured time, allocates a configured amount of
memory while it runs and produces an output of a configured size. The
"computation" is a deterministic affine map, so partition equivalence
(stages 1..s then s+1..S equals 1..S) holds exactly and can be tested.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.models.registry import Array, StageInfo, deserialize_array, register, serialize_array

DEFAULT_STAGES = [
    # (name, sleep ms, allocate MB, output elements, flops)
    ("conv1", 2.0, 2.0, 1024, 4_000_000),
    ("pool1", 0.5, 0.0, 256, 20_000),
    ("conv2", 3.0, 3.0, 512, 6_000_000),
    ("fc1", 1.0, 1.0, 64, 1_000_000),
    ("fc2", 0.5, 0.0, 10, 100_000),
]


class FakeStagesAdapter:
    name = "fake"
    framework = "fake"
    PARAMETERS = {
        "type": "object",
        "properties": {
            "input_elements": {
                "type": "integer",
                "minimum": 1,
                "default": 2048,
                "description": "size of the synthetic input vector",
            },
            "time_scale": {
                "type": "number",
                "minimum": 0,
                "default": 1.0,
                "description": "multiplier of every stage's configured sleep time",
            },
        },
    }

    def __init__(
        self,
        stage_specs: list[tuple[str, float, float, int, int]] | None = None,
        input_elements: int = 2048,
        time_scale: float = 1.0,
    ) -> None:
        self.specs = list(stage_specs or DEFAULT_STAGES)
        self.input_shape: tuple[int, ...] = (input_elements,)
        self.time_scale = time_scale
        self._weights: list[tuple[float, float]] = []
        self._held: list[bytearray] = []

    @property
    def stages(self) -> list[StageInfo]:
        return [
            StageInfo(i, name, flops, flops // 2, (elements,), elements * 4)
            for i, (name, _, _, elements, flops) in enumerate(self.specs, start=1)
        ]

    @property
    def input_bytes(self) -> int:
        return int(np.prod(self.input_shape)) * 4

    def build(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._weights = [
            (float(rng.uniform(0.5, 1.5)), float(rng.uniform(-1, 1))) for _ in self.specs
        ]

    def load_weights(self, path: Path) -> None:
        data = np.load(path)
        self._weights = [(float(a), float(b)) for a, b in data["weights"]]

    def save_weights(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=np.array(self._weights, dtype=np.float64))

    def make_input(self, seed: int) -> Array:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((1, *self.input_shape)).astype(np.float32)

    def _stage(self, x: Array, index: int) -> Array:
        _name, sleep_ms, alloc_mb, elements, _ = self.specs[index - 1]
        scale, shift = self._weights[index - 1]
        if alloc_mb > 0:
            self._held.append(bytearray(int(alloc_mb * 2**20)))
        if sleep_ms > 0:
            time.sleep(sleep_ms * self.time_scale / 1000.0)
        flat = x.reshape(x.shape[0], -1).astype(np.float32)
        # Resample to the stage's output size deterministically.
        idx = np.arange(elements) % flat.shape[1]
        out = flat[:, idx] * np.float32(scale) + np.float32(shift)
        self._held.clear()
        return out

    def run_stages(self, x: Array, start: int, stop: int) -> Array:
        if not self._weights:
            raise RuntimeError("build() or load_weights() first")
        if not 1 <= start <= stop <= len(self.specs):
            raise ValueError(f"stage range {start}..{stop} outside 1..{len(self.specs)}")
        for index in range(start, stop + 1):
            x = self._stage(x, index)
        return x

    def serialize(self, array: Array) -> bytes:
        return serialize_array(array)

    def deserialize(self, data: bytes) -> Array:
        return deserialize_array(data)


ADAPTERS = {"fake": FakeStagesAdapter}
register("fake", FakeStagesAdapter)

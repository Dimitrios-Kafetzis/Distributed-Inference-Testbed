"""Partitioning algorithms behind one interface, registered by name.

An algorithm takes the delay model, the objective ("total_delay" or
"variance"), an initial partition vector, optional per-device maximum total
delays, and a search log; it returns the best feasible vector it found. New
algorithms register through :func:`register_algorithm`, through the
environment variable ``TESTBED_ALGORITHM_MODULES`` (comma-separated module
names) or through the ``testbed.algorithms`` entry-point group.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from src.partitioning.delay_model import DelayModel

ENTRY_POINT_GROUP = "testbed.algorithms"
MODULES_ENV = "TESTBED_ALGORITHM_MODULES"
OBJECTIVES = ("total_delay", "variance")


@dataclass
class SearchLog:
    """One line per evaluated vector: vector, objective, per-device components, note."""

    path: Path | None = None
    lines: list[str] = field(default_factory=list)
    evaluations: int = 0

    def start(self, model: DelayModel, algorithm: str, objective: str) -> None:
        header = (
            f"Partition search log: algorithm={algorithm} objective={objective} "
            f"policy={model.config.policy}"
        )
        columns = "vector | objective | " + " | ".join(
            f"{d.node_id} (local, trans, remote) ms" for d in model.devices
        )
        self.lines = [header, "=" * len(header), columns, "-" * len(columns)]
        self._flush(truncate=True)

    def record(self, model: DelayModel, vector: list[int], value: float, note: str = "") -> None:
        self.evaluations += 1
        parts = [
            f"({d.local_s * 1e3:.2f}, {d.transmission_s * 1e3:.2f}, {d.remote_s * 1e3:.2f})"
            for d in model.per_device(vector)
        ]
        shown = "inf" if value == float("inf") else f"{value:.6f}"
        self.lines.append(
            f"{vector} | {shown} | " + " | ".join(parts) + (f" | {note}" if note else "")
        )
        if len(self.lines) % 200 == 0:
            self._flush()

    def note(self, text: str) -> None:
        self.lines.append(f"# {text}")

    def close(self) -> None:
        self._flush()

    def _flush(self, truncate: bool = False) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        del truncate  # the whole log is rewritten on every flush
        with self.path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")


@dataclass
class SearchResult:
    vector: list[int]
    value: float
    objective: str
    algorithm: str
    evaluations: int
    iterations: int
    extra: dict[str, Any] = field(default_factory=dict)


def evaluate(
    model: DelayModel,
    vector: list[int],
    objective: str,
    max_total_delays_s: list[float] | None,
    log: SearchLog | None = None,
    note: str = "",
) -> float:
    """Objective value, or +inf when the vector is infeasible (feasible set or constraints)."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if not model.is_feasible(vector):
        value = float("inf")
        tag = "infeasible: beyond a device's maximum partition point"
    else:
        value = model.objective(vector, objective)
        tag = note
        if max_total_delays_s is not None:
            totals = [d.total_s for d in model.per_device(vector)]
            if any(t > limit for t, limit in zip(totals, max_total_delays_s, strict=True)):
                value = float("inf")
                tag = "infeasible: exceeds a device's maximum total delay"
    if log is not None:
        log.record(model, vector, value, tag)
    return value


Algorithm = Callable[..., SearchResult]
_REGISTRY: dict[str, Algorithm] = {}
_LOADED = False


def register_algorithm(name: str, algorithm: Algorithm) -> None:
    if name in _REGISTRY and _REGISTRY[name] is not algorithm:
        raise ValueError(f"algorithm {name!r} is already registered")
    _REGISTRY[name] = algorithm


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for module_name in (
        "src.partitioning.algorithms.greedy",
        "src.partitioning.algorithms.exhaustive",
        "src.partitioning.algorithms.trend_based",
    ):
        module = importlib.import_module(module_name)
        register_algorithm(module.NAME, module.run)
    for module_name in filter(None, os.environ.get(MODULES_ENV, "").split(",")):
        module = importlib.import_module(module_name.strip())
        register_algorithm(module.NAME, module.run)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        register_algorithm(ep.name, ep.load())


def names() -> list[str]:
    _load()
    return sorted(_REGISTRY)


def get_algorithm(name: str) -> Algorithm:
    _load()
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown algorithm {name!r}; registered: {names()}") from exc


def iter_algorithms() -> Iterator[tuple[str, Algorithm]]:
    _load()
    yield from sorted(_REGISTRY.items())


def reset_for_tests() -> None:
    global _LOADED
    _LOADED = False
    _REGISTRY.clear()

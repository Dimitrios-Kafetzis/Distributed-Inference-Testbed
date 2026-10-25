"""Exhaustive enumeration: the global optimum over the feasible partition vectors.

Tractable for small instances only ((S + 1)^N vectors); used to measure the
optimality gap of the heuristics, as in the Chapter 4 evaluation.
"""

from __future__ import annotations

from itertools import product

from src.partitioning.algorithms import SearchLog, SearchResult, evaluate
from src.partitioning.delay_model import DelayModel

NAME = "exhaustive"
MAX_VECTORS = 5_000_000


def run(
    model: DelayModel,
    objective: str,
    initial: list[int] | None = None,
    max_total_delays_s: list[float] | None = None,
    log: SearchLog | None = None,
    **_: object,
) -> SearchResult:
    ranges = [device.feasible_points for device in model.devices]
    count = 1
    for r in ranges:
        count *= len(r)
    if count > MAX_VECTORS:
        raise ValueError(f"{count} vectors exceed the exhaustive-search limit of {MAX_VECTORS}")
    best_vector: list[int] | None = None
    best_value = float("inf")
    evaluations = 0
    for candidate in product(*ranges):
        vector = list(candidate)
        value = evaluate(model, vector, objective, max_total_delays_s, log)
        evaluations += 1
        if value < best_value:
            best_value = value
            best_vector = vector
    if best_vector is None:
        raise ValueError("no feasible partition vector")
    extra: dict[str, float] = {"vectors": count}
    if initial is not None:
        initial_value = evaluate(model, list(initial), objective, max_total_delays_s)
        extra["initial_value"] = initial_value
        extra["optimality_gap_percent"] = optimality_gap_percent(initial_value, best_value)
    return SearchResult(best_vector, best_value, objective, NAME, evaluations, 1, extra)


def optimality_gap_percent(value: float, optimum: float) -> float:
    """Relative difference of a solution from the global optimum, in percent."""
    if optimum == 0:
        return 0.0 if value == 0 else float("inf")
    return 100.0 * (value - optimum) / optimum

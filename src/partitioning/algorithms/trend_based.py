"""Trend-based greedy search.

For every device, ``num_solutions`` neighbouring vectors on each side of the
current partition point are evaluated and a linear trend of the objective is
fitted to each side; the device and direction with the steepest decrease are
followed when the first step in that direction improves the objective. The
search stops when no step improves, or when the time budget is exhausted.
"""

from __future__ import annotations

import time

import numpy as np

from src.partitioning.algorithms import SearchLog, SearchResult, evaluate
from src.partitioning.delay_model import DelayModel

NAME = "trend_based"


def linear_trend(values: list[float]) -> float:
    """Slope of a least-squares line through the values (0 for fewer than two points)."""
    finite = [v for v in values if v != float("inf")]
    if len(finite) < 2:
        return 0.0
    x = np.arange(len(finite), dtype=float)
    slope, _ = np.polyfit(x, np.array(finite, dtype=float), 1)
    return float(slope)


def run(
    model: DelayModel,
    objective: str,
    initial: list[int],
    max_total_delays_s: list[float] | None = None,
    log: SearchLog | None = None,
    num_solutions: int = 5,
    max_time_s: float = 300.0,
    **_: object,
) -> SearchResult:
    model.check_vector(initial)
    current = list(initial)
    current_value = evaluate(model, current, objective, max_total_delays_s, log, "initial")
    started = time.perf_counter()
    iterations = 0
    evaluations = 1
    while time.perf_counter() - started < max_time_s:
        best_slope = 0.0
        best_move: tuple[int, int, float] | None = None
        for i, device in enumerate(model.devices):
            for direction in (-1, +1):
                values = []
                first_value = float("inf")
                for k in range(1, num_solutions + 1):
                    s = current[i] + direction * k
                    if not 0 <= s <= device.max_partition_point:
                        break
                    candidate = list(current)
                    candidate[i] = s
                    value = evaluate(model, candidate, objective, max_total_delays_s, log)
                    evaluations += 1
                    if k == 1:
                        first_value = value
                    values.append(value)
                if not values or first_value >= current_value:
                    continue
                slope = linear_trend([current_value, *values])
                if slope < best_slope or (best_move is None and first_value < current_value):
                    best_slope = min(slope, 0.0)
                    best_move = (i, current[i] + direction, first_value)
        if best_move is None:
            break
        i, s, value = best_move
        current[i] = s
        current_value = value
        iterations += 1
        if log is not None:
            log.note(
                f"iteration {iterations}: device {model.devices[i].node_id} -> s={s}, "
                f"{objective}={current_value:.6f}"
            )
    return SearchResult(
        current,
        current_value,
        objective,
        NAME,
        evaluations,
        iterations,
        {"num_solutions": num_solutions},
    )

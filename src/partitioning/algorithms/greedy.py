"""Greedy partitioning (Chapter 4, Algorithm greedy-dnn).

At each iteration every single-step move is evaluated: shifting one device's
partition point one position to the left (more offloading) or to the right
(more local execution), with the coupling through the edge-server schedule
taken into account because the whole vector is re-evaluated. The move with the
largest improvement is committed; the search stops when no single-step move
improves the objective. The objective never increases, so the search
converges to a local optimum in a finite number of steps.
"""

from __future__ import annotations

from src.partitioning.algorithms import SearchLog, SearchResult, evaluate
from src.partitioning.delay_model import DelayModel

NAME = "greedy"


def run(
    model: DelayModel,
    objective: str,
    initial: list[int],
    max_total_delays_s: list[float] | None = None,
    log: SearchLog | None = None,
    **_: object,
) -> SearchResult:
    model.check_vector(initial)
    current = list(initial)
    current_value = evaluate(model, current, objective, max_total_delays_s, log, "initial")
    iterations = 0
    evaluations = 1
    while True:
        best_delta = 0.0
        best_move: tuple[int, int] | None = None
        for i, device in enumerate(model.devices):
            for direction in (+1, -1):
                candidate_s = current[i] + direction
                if not 0 <= candidate_s <= device.max_partition_point:
                    continue
                candidate = list(current)
                candidate[i] = candidate_s
                value = evaluate(model, candidate, objective, max_total_delays_s, log)
                evaluations += 1
                if value == float("inf"):
                    continue
                delta = current_value - value if current_value != float("inf") else float("inf")
                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, candidate_s)
        if best_move is None:
            break
        i, s = best_move
        current[i] = s
        current_value = model.objective(current, objective)
        iterations += 1
        if log is not None:
            log.note(
                f"iteration {iterations}: device {model.devices[i].node_id} -> s={s}, "
                f"{objective}={current_value:.6f}"
            )
    return SearchResult(current, current_value, objective, NAME, evaluations, iterations)

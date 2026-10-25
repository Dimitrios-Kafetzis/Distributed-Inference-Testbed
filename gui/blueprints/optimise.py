"""The Optimise dialog: run the partition optimiser on a run's profiling outputs."""

from __future__ import annotations

import json
import time
from typing import Any

from flask import Blueprint, request

from gui.blueprints.common import error, state
from src.common.runs import OUTPUT_FILES, write_json_atomic
from src.partitioning import algorithms, delay_optimizer
from src.partitioning.algorithms import SearchLog
from src.partitioning.delay_model import DelayModel, DelayModelConfig, profiles_from_files

bp = Blueprint("optimise", __name__, url_prefix="/api/optimise")


@bp.post("")
def optimise() -> Any:
    payload = request.get_json(silent=True) or {}
    run_id = payload.get("run_id")
    if not run_id:
        return error("run_id is required")
    run_dir = state().results_dir / run_id
    files = {name: run_dir / name for name in OUTPUT_FILES}
    missing = [
        name for name, path in files.items() if not path.exists() and name != OUTPUT_FILES[2]
    ]
    if missing:
        return error(f"run {run_id} lacks {', '.join(missing)}", 404)
    totals = json.loads(files[OUTPUT_FILES[0]].read_text(encoding="utf-8"))
    max_points = json.loads(files[OUTPUT_FILES[1]].read_text(encoding="utf-8"))
    latencies = (
        json.loads(files[OUTPUT_FILES[2]].read_text(encoding="utf-8"))
        if files[OUTPUT_FILES[2]].exists()
        else {}
    )
    capacity_path = run_dir / "edge_capacity.json"
    capacity = float(
        payload.get("edge_capacity_flops")
        or (
            json.loads(capacity_path.read_text())["flops_per_second"]
            if capacity_path.exists()
            else state().config.bs_computation_capacity
        )
    )
    try:
        devices = profiles_from_files(totals, max_points, latencies, None, payload.get("nodes"))
        model = DelayModel(
            devices,
            DelayModelConfig(
                payload.get("policy", "SJF"),
                capacity,
                transmission_mode=payload.get("transmission_mode", "rtt_only"),
            ),
        )
        scheme = payload.get("initial_scheme") or None
        mode = (
            "percentage" if isinstance(scheme, str) and scheme.strip().endswith("%") else "explicit"
        )
        initial = delay_optimizer.initial_vector(
            devices,
            scheme if scheme is None or isinstance(scheme, str) else json.dumps(scheme),
            mode,
            model.num_stages,
        )
        limits_raw = payload.get("max_total_delays_ms")
        limits = (
            delay_optimizer.max_delays_seconds(json.dumps(limits_raw), devices)
            if limits_raw
            else None
        )
        algorithm = payload.get("algorithm", "greedy")
        if algorithm not in algorithms.names():
            return error(f"unknown algorithm {algorithm}; registered: {algorithms.names()}")
        out_dir = run_dir / "optimisation" / f"{time.strftime('%Y%m%d-%H%M%S')}-nnpm"
        log = SearchLog(out_dir / "search_log.txt")
        result = delay_optimizer.optimise(
            model,
            algorithm,
            payload.get("objective", "total_delay"),
            initial,
            limits,
            log,
            num_solutions=int(payload.get("num_solutions", 5)),
            max_time_s=float(payload.get("max_time_s", 300)),
        )
    except (ValueError, KeyError) as exc:
        return error(str(exc))
    if result.value == float("inf"):
        return error("no feasible partition vector satisfies the constraints", 422)
    inputs = {"run_id": run_id, "edge_capacity_flops": capacity, "request": payload}
    scenario_path = run_dir / "scenario.json"
    if scenario_path.exists():
        write_json_atomic(
            out_dir / "scenario.json", json.loads(scenario_path.read_text(encoding="utf-8"))
        )
    path = delay_optimizer.write_result(
        out_dir, model, result, inputs, log.path or out_dir / "search_log.txt"
    )
    return {
        "result": json.loads(path.read_text(encoding="utf-8")),
        "output_dir": str(out_dir.relative_to(run_dir)),
    }

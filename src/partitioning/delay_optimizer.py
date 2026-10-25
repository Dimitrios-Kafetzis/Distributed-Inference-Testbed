"""delay_optimizer.py: the partition-optimisation entry point.

Builds the per-device delay model from the profiling outputs
(``total_averages_of_benchmark_results.json``,
``calculated_max_partition_points.json``, ``final_latencies_averages.json``
and, optionally, measured throughput), searches for the partition vector that
minimises the chosen objective under the chosen edge-server scheduling policy,
and writes the selected assignment as a JSON file together with a search log.

Usage::

    python -m src.partitioning.delay_optimizer MAX_POINTS.json TOTAL_AVERAGES.json LATENCIES.json
        --scheduling_algorithm SJF --mode optimize --objective total_delay
        [--algorithm greedy|exhaustive|trend_based]
        [--partition_scheme '[3,5,8,1]' --partition_scheme_mode explicit]
        [--partition_scheme 50% --partition_scheme_mode percentage]
        [--max_total_delays '[150,150,150,150]'] [--output-dir DIR] [--log_file FILE]
        [--edge-capacity FLOPS] [--transmission-mode size_over_rate|rtt_only]
        [--throughput-file FILE]

``--mode default`` only tabulates the delay of every partition point per
device; ``optimize`` runs the greedy search, ``exhaustive_search`` the
enumeration and ``trend_based_optimize`` the trend-based search (the names of
the earlier release); ``--algorithm`` selects any registered algorithm directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.runs import write_json_atomic
from src.partitioning import algorithms
from src.partitioning.algorithms import SearchLog, SearchResult
from src.partitioning.delay_model import (
    DelayModel,
    DelayModelConfig,
    DeviceProfile,
    load_json,
    profiles_from_files,
)

MODE_TO_ALGORITHM = {
    "optimize": "greedy",
    "exhaustive_search": "exhaustive",
    "trend_based_optimize": "trend_based",
}


def initial_vector(
    devices: list[DeviceProfile], scheme: str | None, mode: str | None, num_stages: int
) -> list[int]:
    """Explicit list (in device order, or a mapping by node id) or a percentage of the depth."""
    if scheme is None:
        return [1 if d.max_partition_point >= 1 else 0 for d in devices]
    if mode == "percentage":
        percentage = float(scheme.strip().rstrip("%")) / 100.0
        if not 0.0 <= percentage <= 1.0:
            raise ValueError("percentage must be between 0 and 100")
        return [min(d.max_partition_point, round(percentage * num_stages)) for d in devices]
    parsed = json.loads(scheme)
    if isinstance(parsed, dict):
        try:
            return [int(parsed[d.node_id]) for d in devices]
        except KeyError as exc:
            raise ValueError(f"partition scheme lacks node {exc}") from exc
    if not isinstance(parsed, list) or len(parsed) != len(devices):
        raise ValueError(
            f"partition scheme must list {len(devices)} values "
            f"(device order: {[d.node_id for d in devices]})"
        )
    return [int(v) for v in parsed]


def max_delays_seconds(raw: str | None, devices: list[DeviceProfile]) -> list[float] | None:
    """Per-device maximum total delays in milliseconds, as a list or a mapping by node id."""
    if raw is None:
        return None
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        return [float(parsed[d.node_id]) / 1e3 for d in devices]
    if len(parsed) != len(devices):
        raise ValueError(f"max_total_delays must list {len(devices)} values")
    return [float(v) / 1e3 for v in parsed]


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tabulate(model: DelayModel) -> dict[str, list[dict[str, Any]]]:
    """``default`` mode: each device's delay at every feasible s (the others at their start)."""
    table: dict[str, list[dict[str, Any]]] = {}
    base = [1 if d.max_partition_point >= 1 else 0 for d in model.devices]
    for i, device in enumerate(model.devices):
        rows = []
        for s in device.feasible_points:
            vector = list(base)
            vector[i] = s
            rows.append(model.per_device(vector)[i].to_dict())
        table[device.node_id] = rows
    return table


def optimise(
    model: DelayModel,
    algorithm: str,
    objective: str,
    initial: list[int],
    max_total_delays_s: list[float] | None,
    log: SearchLog,
    **params: Any,
) -> SearchResult:
    log.start(model, algorithm, objective)
    try:
        result = algorithms.get_algorithm(algorithm)(
            model, objective, initial, max_total_delays_s=max_total_delays_s, log=log, **params
        )
    finally:
        log.close()
    return result


def write_result(
    output_dir: Path,
    model: DelayModel,
    result: SearchResult,
    inputs: dict[str, Any],
    log_path: Path,
) -> Path:
    per_device = model.per_device(result.vector)
    baseline = model.all_local()
    payload = {
        "optimized_partition_scheme": result.vector,
        "by_node": {d.node_id: s for d, s in zip(model.devices, result.vector, strict=True)},
        "objective": result.objective,
        "value": result.value,
        "value_unit": "s" if result.objective == "total_delay" else "s^2",
        "algorithm": result.algorithm,
        "policy": model.config.policy,
        "evaluations": result.evaluations,
        "iterations": result.iterations,
        "per_device": [d.to_dict() for d in per_device],
        "all_local_baseline": {
            "vector": baseline,
            "value": model.objective(baseline, result.objective),
        },
        "extra": result.extra,
        "inputs": inputs,
        "search_log": str(log_path),
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    path = output_dir / "optimized_partition_scheme.json"
    write_json_atomic(path, payload)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimise the partition points of all devices.")
    parser.add_argument("max_partition_points_path")
    parser.add_argument(
        "local_execution_delays_path", help="total_averages_of_benchmark_results.json"
    )
    parser.add_argument("transmission_delays_path", help="final_latencies_averages.json")
    parser.add_argument("--scheduling_algorithm", default="SJF", choices=["SJF", "RR", "sjf", "rr"])
    parser.add_argument(
        "--mode",
        default="optimize",
        choices=["default", "optimize", "exhaustive_search", "trend_based_optimize"],
    )
    parser.add_argument("--algorithm", default=None, help="registered algorithm (overrides --mode)")
    parser.add_argument("--objective", default="total_delay", choices=list(algorithms.OBJECTIVES))
    parser.add_argument("--partition_scheme", default=None)
    parser.add_argument(
        "--partition_scheme_mode", default="explicit", choices=["explicit", "percentage"]
    )
    parser.add_argument("--max_total_delays", default=None, help="milliseconds, list or {node: ms}")
    parser.add_argument(
        "--log_file", default=None, help="search log (default: <output-dir>/search_log.txt)"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("optimisation"))
    parser.add_argument(
        "--edge-capacity", type=float, default=None, help="C^R in FLOP/s (default: config)"
    )
    parser.add_argument("--rr-quantum-ms", type=float, default=0.0)
    parser.add_argument("--transmission-mode", default=None, choices=["size_over_rate", "rtt_only"])
    parser.add_argument("--throughput-file", default=None, help="JSON {node: Mbit/s}")
    parser.add_argument(
        "--nodes", default=None, help="comma-separated device order (default: sorted)"
    )
    parser.add_argument("--num_solutions", type=int, default=5)
    parser.add_argument("--max_time", type=float, default=300.0)
    parser.add_argument("--scenario", default=None, help="scenario JSON to copy next to the result")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    max_points = load_json(args.max_partition_points_path)
    total_averages = load_json(args.local_execution_delays_path)
    latencies = load_json(args.transmission_delays_path)
    throughput = load_json(args.throughput_file) if args.throughput_file else None
    nodes = [n for n in args.nodes.split(",") if n] if args.nodes else None
    devices = profiles_from_files(total_averages, max_points, latencies, throughput, nodes)

    capacity = args.edge_capacity
    if capacity is None:
        from src.config import load_config

        capacity = load_config().bs_computation_capacity
    transmission_mode = args.transmission_mode or ("size_over_rate" if throughput else "rtt_only")
    config = DelayModelConfig(
        policy=args.scheduling_algorithm,
        edge_capacity_flops=capacity,
        rr_quantum_s=args.rr_quantum_ms / 1e3,
        transmission_mode=transmission_mode,
    )
    model = DelayModel(devices, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "max_partition_points": {
            "path": args.max_partition_points_path,
            "sha256": file_hash(args.max_partition_points_path),
        },
        "total_averages": {
            "path": args.local_execution_delays_path,
            "sha256": file_hash(args.local_execution_delays_path),
        },
        "latencies": {
            "path": args.transmission_delays_path,
            "sha256": file_hash(args.transmission_delays_path),
        },
        "throughput": args.throughput_file,
        "edge_capacity_flops": capacity,
        "transmission_mode": transmission_mode,
        "scenario": args.scenario,
    }
    write_json_atomic(args.output_dir / "inputs.json", inputs)
    if args.scenario:
        write_json_atomic(args.output_dir / "scenario.json", load_json(args.scenario))

    if args.mode == "default" and args.algorithm is None:
        table = tabulate(model)
        write_json_atomic(args.output_dir / "delays.json", table)
        for node_id, rows in table.items():
            print(node_id)
            for row in rows:
                print(
                    f"  s={row['s']:2d}: local {row['local_ms']:8.2f}  "
                    f"trans {row['transmission_ms']:8.2f}  remote {row['remote_ms']:8.2f}  "
                    f"total {row['total_ms']:8.2f} ms"
                )
        return

    algorithm = args.algorithm or MODE_TO_ALGORITHM[args.mode]
    initial = initial_vector(
        devices, args.partition_scheme, args.partition_scheme_mode, model.num_stages
    )
    limits = max_delays_seconds(args.max_total_delays, devices)
    log_path = Path(args.log_file) if args.log_file else args.output_dir / "search_log.txt"
    result = optimise(
        model,
        algorithm,
        args.objective,
        initial,
        limits,
        SearchLog(log_path),
        num_solutions=args.num_solutions,
        max_time_s=args.max_time,
    )
    if result.value == float("inf"):
        print("no feasible partition vector satisfies the constraints", file=sys.stderr)
        raise SystemExit(2)
    path = write_result(args.output_dir, model, result, inputs, log_path)
    print(
        f"Optimized Partition Scheme: {result.vector} "
        f"({result.objective}={result.value:.6f}, {result.algorithm}, {model.config.policy})"
    )
    for d in model.per_device(result.vector):
        print(
            f"  {d.node_id}: s={d.s} local {d.local_s * 1e3:.2f} ms, "
            f"transmission {d.transmission_s * 1e3:.2f} ms, remote {d.remote_s * 1e3:.2f} ms, "
            f"total {d.total_s * 1e3:.2f} ms"
        )
    print(f"written to {path}; search log {log_path}")


if __name__ == "__main__":
    main()

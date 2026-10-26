"""Result aggregation: per-device averages over repeated runs, and trace merging.

After benchmarking has run on all nodes, this module collects the per-node
result files, groups them by device, averages the repeated runs into
per-layer and per-partition-point statistics, and writes
``total_averages_of_benchmark_results.json``. It also merges each device's
timing records with its resource-monitoring and power traces: the records are
grouped by node and run, and every sample is converted to wall-clock time
through the time origin of its own record, producing one unified record per
device and run.

Usage::

    python -m src.benchmarking.result_aggregation <raw_dir> <output_file>
        [--monitor-dir DIR] [--power-dir DIR] [--merged-dir DIR]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.benchmarking.benchmark_results import (
    BenchmarkResults,
    PowerTrace,
    Record,
    ResourceTrace,
    load,
)
from src.benchmarking.stats import summarise
from src.common.runs import write_json_atomic

STAGE_METRICS = ("latency_ns", "cpu_peak_percent", "memory_peak_mb")
POINT_METRICS = ("cumulative_latency_ns", "cumulative_cpu_percent", "cumulative_memory_mb")
PHASE_METRICS = ("local_ns", "transmission_ns", "remote_ns", "queue_wait_ns")


def load_records(directory: str | Path) -> list[Record]:
    return [load(p) for p in sorted(Path(directory).glob("*.json"))]


def _benchmarks(records: list[Record]) -> list[BenchmarkResults]:
    return [r for r in records if isinstance(r, BenchmarkResults)]


def aggregate(benchmarks: list[BenchmarkResults]) -> dict[str, dict[str, Any]]:
    """Group by node, average over repetitions. Stage- and point-constant values
    (name, FLOPs, output bytes) are taken from the records and checked for agreement."""
    by_node: dict[str, list[BenchmarkResults]] = defaultdict(list)
    for record in benchmarks:
        by_node[record.node_id].append(record)

    result: dict[str, dict[str, Any]] = {}
    for node_id, records in sorted(by_node.items()):
        models = {r.model for r in records}
        if len(models) != 1:
            raise ValueError(f"node {node_id}: records of different models {sorted(models)}")
        stages: dict[int, dict[str, Any]] = {}
        for index in sorted({s.index for r in records for s in r.stages}):
            samples = [r.stage(index) for r in records if any(s.index == index for s in r.stages)]
            constants = {(s.name, s.flops, s.macs, s.output_bytes) for s in samples}
            if len(constants) != 1:
                raise ValueError(f"node {node_id}, stage {index}: inconsistent stage definition")
            name, flops, macs, output_bytes = constants.pop()
            stages[index] = {
                "name": name,
                "flops": flops,
                "macs": macs,
                "output_bytes": output_bytes,
                **{
                    metric: summarise([float(getattr(s, metric)) for s in samples]).to_dict()
                    for metric in STAGE_METRICS
                },
            }
        points: dict[int, dict[str, Any]] = {}
        for s_value in sorted({p.s for r in records for p in r.partition_points}):
            samples_p = [
                r.partition_point(s_value)
                for r in records
                if any(p.s == s_value for p in r.partition_points)
            ]
            transmitted = {p.transmitted_bytes for p in samples_p}
            if len(transmitted) != 1:
                raise ValueError(f"node {node_id}, s={s_value}: inconsistent transmitted bytes")
            points[s_value] = {
                "transmitted_bytes": transmitted.pop(),
                **{
                    metric: summarise([float(getattr(p, metric)) for p in samples_p]).to_dict()
                    for metric in POINT_METRICS
                },
            }
        phases: dict[int, dict[str, Any]] = {}
        for s_value in sorted({ph.s for r in records for ph in r.phases}):
            samples_ph = [ph for r in records for ph in r.phases if ph.s == s_value]
            phases[s_value] = {
                metric: summarise([float(getattr(ph, metric)) for ph in samples_ph]).to_dict()
                for metric in PHASE_METRICS
            }
        result[node_id] = {
            "hostname": records[-1].hostname,
            "model": models.pop(),
            "repetitions": len([r for r in records if r.stages]),
            "run_ids": sorted({r.run_id for r in records}),
            "stages": {str(k): v for k, v in stages.items()},
            "partition_points": {str(k): v for k, v in points.items()},
            "phases": {str(k): v for k, v in phases.items()},
        }
    return result


def merge_traces(records: list[Record]) -> dict[str, dict[str, Any]]:
    """One unified record per device and run (keyed by node id and run id): timing
    records plus resource and power samples re-aligned to the wall clock through the
    shared origin; the hostname is kept inside the record."""
    merged: dict[str, dict[str, Any]] = {}
    for record in records:
        key = f"{record.node_id}__{record.run_id}"
        entry = merged.setdefault(
            key,
            {
                "hostname": record.hostname,
                "node_id": record.node_id,
                "run_id": record.run_id,
                "origins": [],
                "benchmarks": [],
                "resource_samples": [],
                "power_samples": [],
            },
        )
        origin = record.origin
        entry["origins"].append({"kind": record.kind, **origin.__dict__})
        if isinstance(record, BenchmarkResults):
            entry["benchmarks"].append(
                {
                    "repetition": record.repetition,
                    "model": record.model,
                    "stages": [s.__dict__ for s in record.stages],
                    "partition_points": [p.__dict__ for p in record.partition_points],
                    "phases": [p.__dict__ for p in record.phases],
                }
            )
        elif isinstance(record, ResourceTrace):
            entry["resource_samples"].extend(
                {"wall_time_s": origin.to_wall_time(s.perf_counter_ns), **s.__dict__}
                for s in record.samples
            )
        elif isinstance(record, PowerTrace):
            entry["power_samples"].extend(
                {"wall_time_s": origin.to_wall_time(s.perf_counter_ns), **s.__dict__}
                for s in record.samples
            )
    for entry in merged.values():
        entry["resource_samples"].sort(key=lambda s: s["wall_time_s"])
        entry["power_samples"].sort(key=lambda s: s["wall_time_s"])
        entry["benchmarks"].sort(key=lambda b: b["repetition"])
    return merged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("raw_dir", help="directory with the collected BenchmarkResults files")
    parser.add_argument("output", help="total_averages_of_benchmark_results.json to write")
    parser.add_argument("--monitor-dir", help="resource traces to merge")
    parser.add_argument("--power-dir", help="power traces to merge")
    parser.add_argument("--merged-dir", help="where to write the unified per-device records")
    args = parser.parse_args(argv)

    records = load_records(args.raw_dir)
    benchmarks = _benchmarks(records)
    if not benchmarks:
        raise SystemExit(f"no benchmark records in {args.raw_dir}")
    write_json_atomic(Path(args.output), aggregate(benchmarks))
    print(f"averages for {len({b.node_id for b in benchmarks})} node(s) written to {args.output}")

    if args.merged_dir:
        for extra in (args.monitor_dir, args.power_dir):
            if extra:
                records.extend(load_records(extra))
        merged_dir = Path(args.merged_dir)
        for key, entry in merge_traces(records).items():
            write_json_atomic(merged_dir / f"{key}.json", entry)
        print(f"unified records written to {merged_dir}")


if __name__ == "__main__":
    main()

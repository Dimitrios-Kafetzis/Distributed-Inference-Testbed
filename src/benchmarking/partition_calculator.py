"""Partition feasibility: the maximum feasible partition point of each device.

Reads the aggregated profile (``total_averages_of_benchmark_results.json``) and
the system-state snapshot taken by the latency-measurement module
(``final_latencies_averages.json``) and applies two independent constraints:

* **Memory.** The largest s such that, for every s' <= s, the cumulative memory
  requirement of stages 1..s' does not exceed the device's available memory at
  the time of profiling.
* **CPU.** The largest s such that, for every stage up to s, the baseline CPU
  usage plus the stage's incremental CPU demand remains below 100 %.

The tighter constraint gives the maximum feasible partition point, written to
``calculated_max_partition_points.json`` as ``{node_id: s}``; the per-node
details (both limits and the binding constraint) go to a sibling
``calculated_max_partition_points.details.json``.

Usage::

    python -m src.benchmarking.partition_calculator <final_latencies_averages.json>
        <total_averages_of_benchmark_results.json> <output_file>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.common.runs import write_json_atomic

CPU_LIMIT_PERCENT = 100.0


@dataclass(frozen=True)
class FeasibilityDetail:
    node_id: str
    available_memory_mb: float
    baseline_cpu_percent: float
    memory_limited_s: int
    cpu_limited_s: int
    max_partition_point: int
    binding: str  # "memory", "cpu", "both" or "none"


def memory_limit(cumulative_memory_mb: dict[int, float], available_mb: float) -> int:
    """Largest s whose every prefix s' <= s fits in the available memory (s = 0 always fits)."""
    limit = 0
    for s in sorted(k for k in cumulative_memory_mb if k > 0):
        if cumulative_memory_mb[s] <= available_mb:
            limit = s
        else:
            break
    return limit


def cpu_limit(stage_cpu_percent: dict[int, float], baseline_percent: float) -> int:
    """Largest s such that baseline + increment stays below 100 % for every stage up to s."""
    limit = 0
    for index in sorted(stage_cpu_percent):
        if baseline_percent + stage_cpu_percent[index] < CPU_LIMIT_PERCENT:
            limit = index
        else:
            break
    return limit


def calculate(
    system_state: dict[str, dict[str, Any]], averages: dict[str, dict[str, Any]]
) -> tuple[dict[str, int], list[FeasibilityDetail]]:
    results: dict[str, int] = {}
    details: list[FeasibilityDetail] = []
    for node_id, profile in sorted(averages.items()):
        if node_id not in system_state:
            raise KeyError(f"no system-state snapshot for node {node_id}")
        state = system_state[node_id]
        available_mb = float(state["available_memory_mb"])
        baseline_cpu = float(state["cpu_usage_percent"])
        cumulative = {
            int(s): float(values["cumulative_memory_mb"]["mean"])
            for s, values in profile["partition_points"].items()
        }
        stage_cpu = {
            int(i): float(values["cpu_peak_percent"]["mean"])
            for i, values in profile["stages"].items()
        }
        mem_s = memory_limit(cumulative, available_mb)
        cpu_s = cpu_limit(stage_cpu, baseline_cpu)
        max_s = min(mem_s, cpu_s)
        num_stages = max(stage_cpu) if stage_cpu else 0
        if max_s >= num_stages:
            binding = "none"
        elif mem_s == cpu_s:
            binding = "both"
        else:
            binding = "memory" if mem_s < cpu_s else "cpu"
        results[node_id] = max_s
        details.append(
            FeasibilityDetail(node_id, available_mb, baseline_cpu, mem_s, cpu_s, max_s, binding)
        )
    return results, details


def details_path(output: str | Path) -> Path:
    out = Path(output)
    return out.with_name(out.stem + ".details.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("final_latencies_file", help="final_latencies_averages.json")
    parser.add_argument("total_averages_file", help="total_averages_of_benchmark_results.json")
    parser.add_argument("output", help="calculated_max_partition_points.json to write")
    args = parser.parse_args(argv)

    with Path(args.final_latencies_file).open(encoding="utf-8") as fh:
        system_state = json.load(fh)
    with Path(args.total_averages_file).open(encoding="utf-8") as fh:
        averages = json.load(fh)
    results, details = calculate(system_state, averages)
    write_json_atomic(Path(args.output), results)
    write_json_atomic(details_path(args.output), [asdict(d) for d in details])
    for detail in details:
        print(
            f"{detail.node_id}: max partition point {detail.max_partition_point} "
            f"(memory limit {detail.memory_limited_s}, CPU limit {detail.cpu_limited_s}, "
            f"binding: {detail.binding})"
        )


if __name__ == "__main__":
    main()

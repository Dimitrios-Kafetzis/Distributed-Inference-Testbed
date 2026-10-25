"""Regenerate data/sample/ (synthetic). Run from the repository root."""

from __future__ import annotations

import json
from pathlib import Path

from src.benchmarking import partition_calculator, result_aggregation
from src.common.runs import write_json_atomic
from src.network import node_statistics
from tests import synthetic

ROOT = Path(__file__).resolve().parents[1] / "data" / "sample"


def main() -> None:
    for sub in ("raw", "monitor", "power", "latency"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    synthetic.benchmark("device1", 0, seed=1, device_index=0, with_phases=True).save(
        ROOT / "raw" / "device1-host_rep0.json"
    )
    synthetic.benchmark("device1", 1, seed=1, device_index=0, with_phases=True).save(
        ROOT / "raw" / "device1-host_rep1.json"
    )
    synthetic.benchmark("device2", 0, seed=2, device_index=1, stage_scale=1.4).save(
        ROOT / "raw" / "device2-host_rep0.json"
    )
    synthetic.resource_trace("device1", seconds=12, throttle_at=9).save(
        ROOT / "monitor" / "device1.json"
    )
    synthetic.power_trace("device1", seconds=12, power_mw=3800.0).save(
        ROOT / "power" / "device1.json"
    )
    synthetic.latency_record(
        "device1", available_mb=3100.0, cpu=3.5, extra_targets={"device2": [1.9, 2.2, 2.0]}
    ).save(ROOT / "latency" / "device1.json")
    synthetic.latency_record(
        "device2", rtts=[7.9, 8.4, 8.1, None, 8.0], available_mb=420.0, cpu=35.0
    ).save(ROOT / "latency" / "device2.json")

    averages = node_statistics.compute_averages(node_statistics.load_records(ROOT / "latency"))
    write_json_atomic(ROOT / "final_latencies_averages.json", averages)
    benchmarks = [r for r in result_aggregation.load_records(ROOT / "raw") if r.kind == "benchmark"]
    totals = result_aggregation.aggregate(benchmarks)  # type: ignore[arg-type]
    write_json_atomic(ROOT / "total_averages_of_benchmark_results.json", totals)
    results, details = partition_calculator.calculate(averages, totals)
    write_json_atomic(ROOT / "calculated_max_partition_points.json", results)
    write_json_atomic(
        ROOT / "calculated_max_partition_points.details.json", [d.__dict__ for d in details]
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmarking import result_aggregation as agg
from tests import synthetic


def write_records(raw: Path, node_ids: list[str], repetitions: int) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for i, node in enumerate(node_ids):
        for rep in range(repetitions):
            synthetic.benchmark(node, rep, seed=i, device_index=i, with_phases=True).save(
                raw / f"{node}-host_rep{rep}.json"
            )


def test_aggregate_groups_by_node_and_averages_repetitions(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_records(raw, ["device1", "device2"], repetitions=3)
    records = agg.load_records(raw)
    result = agg.aggregate([r for r in records if r.kind == "benchmark"])
    assert sorted(result) == ["device1", "device2"]
    d1 = result["device1"]
    assert d1["repetitions"] == 3 and d1["model"] == "alexnet"
    reps = [synthetic.benchmark("device1", rep, seed=0) for rep in range(3)]
    expected_mean = sum(r.latency_ns(3) for r in reps) / 3
    assert d1["stages"]["3"]["latency_ns"]["mean"] == pytest.approx(expected_mean)
    assert d1["stages"]["3"]["latency_ns"]["count"] == 3
    assert d1["stages"]["3"]["output_bytes"] == synthetic.OUTPUT_BYTES[2]
    assert d1["partition_points"]["0"]["transmitted_bytes"] == synthetic.INPUT_BYTES
    expected_cum = sum(r.partition_point(5).cumulative_memory_mb for r in reps) / 3
    assert d1["partition_points"]["5"]["cumulative_memory_mb"]["mean"] == pytest.approx(
        expected_cum
    )
    assert set(d1["phases"]) == {"0", "4", "8", "11"}
    assert d1["phases"]["4"]["transmission_ns"]["count"] == 3


def test_aggregate_rejects_mixed_models() -> None:
    a = synthetic.benchmark("device1", 0)
    b = synthetic.benchmark("device1", 1)
    b.model = "vgg16"
    with pytest.raises(ValueError, match="different models"):
        agg.aggregate([a, b])


def test_merge_traces_aligns_on_origin() -> None:
    bench = synthetic.benchmark("device1", 0, device_index=0)
    monitor = synthetic.resource_trace("device1", seconds=3, device_index=0)
    power = synthetic.power_trace("device1", seconds=2, device_index=0)
    other = synthetic.resource_trace("device2", seconds=2, device_index=1)
    merged = agg.merge_traces([bench, monitor, power, other])
    assert sorted(merged) == [
        "device1-host__20261001-120000-synth1",
        "device2-host__20261001-120000-synth1",
    ]
    entry = merged["device1-host__20261001-120000-synth1"]
    assert len(entry["benchmarks"]) == 1 and len(entry["resource_samples"]) == 3
    # Monotonic offsets become wall-clock times through the shared origin.
    walls = [s["wall_time_s"] for s in entry["resource_samples"]]
    assert walls == [bench.origin.wall_time_s + t for t in range(3)]
    # Device 2 has a different monotonic counter but the same wall clock.
    walls2 = [
        s["wall_time_s"] for s in merged["device2-host__20261001-120000-synth1"]["resource_samples"]
    ]
    assert walls2 == [other.origin.wall_time_s, other.origin.wall_time_s + 1]


def test_cli_writes_outputs(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_records(raw, ["device1"], repetitions=2)
    monitor = tmp_path / "monitor"
    monitor.mkdir()
    synthetic.resource_trace("device1").save(monitor / "device1.json")
    out = tmp_path / "total_averages_of_benchmark_results.json"
    agg.main(
        [
            str(raw),
            str(out),
            "--monitor-dir",
            str(monitor),
            "--merged-dir",
            str(tmp_path / "merged"),
        ]
    )
    data = json.loads(out.read_text())
    assert data["device1"]["repetitions"] == 2
    merged_files = list((tmp_path / "merged").glob("*.json"))
    assert len(merged_files) == 1
    assert len(json.loads(merged_files[0].read_text())["resource_samples"]) == 10


def test_cli_fails_without_benchmarks(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    with pytest.raises(SystemExit, match="no benchmark records"):
        agg.main([str(tmp_path / "raw"), str(tmp_path / "out.json")])

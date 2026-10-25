from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmarking import benchmark_results as br
from src.benchmarking.benchmark_results import (
    BenchmarkResults,
    Origin,
    PowerTrace,
    RecordError,
    ResourceTrace,
    StageRecord,
)
from src.benchmarking.stats import percentile, summarise
from tests import synthetic


def test_summarise_and_percentile() -> None:
    s = summarise([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert s.mean == 5.0 and s.min == 2.0 and s.max == 9.0 and s.count == 8
    assert round(s.std, 4) == 2.1381  # sample standard deviation
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([10.0], 95) == 10.0
    assert summarise([3.0]).std == 0.0
    with pytest.raises(ValueError):
        summarise([])


def test_benchmark_round_trip(tmp_path: Path) -> None:
    record = synthetic.benchmark("device1", 0, with_phases=True)
    path = record.save(tmp_path / "device1_rep0.json")
    loaded = br.load(path)
    assert isinstance(loaded, BenchmarkResults)
    assert loaded == record
    assert loaded.latency_ns(1) == record.stages[0].latency_ns
    assert loaded.cpu_percent(3) == record.stages[2].cpu_peak_percent
    assert loaded.memory_mb(11) == record.stages[10].memory_peak_mb
    assert br.load_benchmark(path).num_stages == 11
    with pytest.raises(KeyError):
        loaded.stage(12)


def test_cumulative_from_stages_is_the_prefix_sum() -> None:
    record = synthetic.benchmark("device1", 0)
    points = record.partition_points
    assert points[0].s == 0 and points[0].transmitted_bytes == synthetic.INPUT_BYTES
    for s in range(1, 12):
        p = record.partition_point(s)
        assert p.cumulative_latency_ns == sum(st.latency_ns for st in record.stages[:s])
        assert p.cumulative_memory_mb == max(st.memory_peak_mb for st in record.stages[:s])
        assert p.transmitted_bytes == record.stages[s - 1].output_bytes


def test_resource_and_power_round_trip(tmp_path: Path) -> None:
    trace = synthetic.resource_trace("device2", seconds=6, device_index=1, throttle_at=4)
    assert [s.perf_counter_ns for s in trace.below_nominal()] == [
        trace.origin.perf_counter_ns + 4 * 10**9,
        trace.origin.perf_counter_ns + 5 * 10**9,
    ]
    loaded = br.load(trace.save(tmp_path / "monitor.json"))
    assert isinstance(loaded, ResourceTrace) and loaded == trace

    power = synthetic.power_trace("device2", seconds=3, power_mw=4200.0)
    loaded_p = br.load(power.save(tmp_path / "power.json"))
    assert isinstance(loaded_p, PowerTrace)
    assert loaded_p.averages() == {"current_ma": 840.0, "voltage_v": 5.0, "power_mw": 4200.0}
    assert PowerTrace(1, "power_trace", "r", "h", "n", Origin(0.0, 0)).averages()["power_mw"] == 0.0


def test_origin_converts_monotonic_to_wall_time() -> None:
    o = Origin(wall_time_s=1000.0, perf_counter_ns=5_000_000_000)
    assert o.to_wall_time(7_500_000_000) == 1002.5


def test_schema_rejects_bad_records(tmp_path: Path) -> None:
    good = synthetic.benchmark("device1", 0).to_dict()
    bad_kind = dict(good, kind="profile")
    with pytest.raises(RecordError, match="kind"):
        br.from_dict(bad_kind)
    bad_stage = json.loads(json.dumps(good))
    bad_stage["stages"][0]["latency_ns"] = -1
    with pytest.raises(RecordError, match="latency_ns"):
        br.from_dict(bad_stage)
    with pytest.raises(RecordError, match="JSON"):
        br.from_json("{not json")
    (tmp_path / "x.json").write_text(json.dumps(synthetic.resource_trace("d").to_dict()))
    with pytest.raises(RecordError, match="expected a benchmark"):
        br.load_benchmark(tmp_path / "x.json")


def test_save_validates_before_writing(tmp_path: Path) -> None:
    record = synthetic.benchmark("device1", 0)
    record.stages[0] = StageRecord(1, "conv1", -5, 0.0, 0.0, 0, 0, 0)
    with pytest.raises(RecordError):
        record.save(tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_cli_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = synthetic.benchmark("device1", 2).save(tmp_path / "r.json")
    monkeypatch.setattr("sys.argv", ["benchmark_results", str(path)])
    br.main()
    out = capsys.readouterr().out
    assert "repetition 2, 11 stages" in out and "stage 11" in out

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmarking import partition_calculator as pc


def test_memory_limit_prefix_rule_and_boundaries() -> None:
    cumulative = {0: 0.0, 1: 100.0, 2: 250.0, 3: 400.0, 4: 500.0}
    assert pc.memory_limit(cumulative, 1000.0) == 4
    assert pc.memory_limit(cumulative, 400.0) == 3  # exactly equal does not exceed
    assert pc.memory_limit(cumulative, 399.9) == 2
    assert pc.memory_limit(cumulative, 50.0) == 0
    # A later prefix that would fit does not count once an earlier one failed.
    assert pc.memory_limit({1: 100.0, 2: 900.0, 3: 200.0}, 500.0) == 1


def test_cpu_limit_below_100_rule() -> None:
    stages = {1: 30.0, 2: 60.0, 3: 20.0}
    assert pc.cpu_limit(stages, 10.0) == 3
    assert pc.cpu_limit(stages, 40.0) == 1  # 40 + 60 = 100 is not below 100
    assert pc.cpu_limit(stages, 39.9) == 3
    assert pc.cpu_limit(stages, 70.0) == 0


def profile(cumulative: dict[int, float], stage_cpu: dict[int, float]) -> dict[str, object]:
    return {
        "stages": {str(i): {"cpu_peak_percent": {"mean": v}} for i, v in stage_cpu.items()},
        "partition_points": {
            str(s): {"cumulative_memory_mb": {"mean": v}} for s, v in cumulative.items()
        },
    }


def test_calculate_takes_the_tighter_constraint() -> None:
    averages = {
        "device1": profile({0: 0, 1: 100, 2: 200, 3: 300}, {1: 10, 2: 10, 3: 10}),
        "device2": profile({0: 0, 1: 100, 2: 200, 3: 300}, {1: 10, 2: 80, 3: 10}),
        "device3": profile({0: 0, 1: 100, 2: 200, 3: 300}, {1: 10, 2: 10, 3: 10}),
    }
    state = {
        "device1": {"available_memory_mb": 250.0, "cpu_usage_percent": 5.0},
        "device2": {"available_memory_mb": 1000.0, "cpu_usage_percent": 20.0},
        "device3": {"available_memory_mb": 1000.0, "cpu_usage_percent": 5.0},
    }
    results, details = pc.calculate(state, averages)
    assert results == {"device1": 2, "device2": 1, "device3": 3}
    by_node = {d.node_id: d for d in details}
    assert by_node["device1"].binding == "memory"
    assert by_node["device2"].binding == "cpu"
    assert by_node["device3"].binding == "none"


def test_calculate_requires_system_state() -> None:
    with pytest.raises(KeyError, match="device9"):
        pc.calculate({}, {"device9": profile({0: 0, 1: 1}, {1: 1})})


def test_cli_writes_results_and_details(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    averages = {"device1": profile({0: 0, 1: 100, 2: 200}, {1: 10, 2: 95})}
    state = {"device1": {"available_memory_mb": 500.0, "cpu_usage_percent": 10.0}}
    (tmp_path / "final_latencies_averages.json").write_text(json.dumps(state))
    (tmp_path / "total.json").write_text(json.dumps(averages))
    out = tmp_path / "calculated_max_partition_points.json"
    pc.main(
        [str(tmp_path / "final_latencies_averages.json"), str(tmp_path / "total.json"), str(out)]
    )
    assert json.loads(out.read_text()) == {"device1": 1}
    details = json.loads(pc.details_path(out).read_text())
    assert details[0]["cpu_limited_s"] == 1 and details[0]["memory_limited_s"] == 2
    assert "binding: cpu" in capsys.readouterr().out

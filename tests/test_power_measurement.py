from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmarking import benchmark_results as br
from src.benchmarking.benchmark_results import PowerSample
from src.common.clock import FakeClock
from src.network import power_measurement as pm


def test_monitor_records_samples_and_energy(tmp_path: Path) -> None:
    sensor = pm.FakeSensor([(800.0, 5.0, 4000.0), (900.0, 5.0, 4500.0), (1000.0, 5.0, 5000.0)])
    clock = FakeClock(start_ns=0)
    monitor = pm.PowerMonitor(
        sensor,
        run_id="r",
        node_id="d",
        output=tmp_path / "p.json",
        interval_s=0.5,
        clock=clock,
        hostname="h",
    )
    trace = monitor.run(max_samples=3)
    assert [s.perf_counter_ns for s in trace.samples] == [0, 500_000_000, 1_000_000_000]
    assert trace.averages() == {"current_ma": 900.0, "voltage_v": 5.0, "power_mw": 4500.0}
    # Trapezoid: (4000+4500)/2*0.5 + (4500+5000)/2*0.5 mJ = 2125 + 2375 = 4500 mJ
    assert trace.metadata["energy_j"] == pytest.approx(4.5)
    loaded = br.load(tmp_path / "p.json")
    assert isinstance(loaded, br.PowerTrace) and loaded.metadata["averages"]["power_mw"] == 4500.0


def test_energy_integration_windows_and_metrics() -> None:
    samples = [PowerSample(t * 10**9, 0, 0, 4000.0) for t in range(5)]
    assert pm.integrate_energy_j(samples) == pytest.approx(16.0)
    assert pm.integrate_energy_j(samples, start_ns=10**9, end_ns=3 * 10**9) == pytest.approx(8.0)
    assert pm.integrate_energy_j(samples[:1]) == 0.0
    step = [
        PowerSample(0, 0, 0, 2000.0),
        PowerSample(10**9, 0, 0, 2000.0),
        PowerSample(2 * 10**9, 0, 0, 6000.0),
    ]
    assert pm.integrate_energy_j(step) == pytest.approx(2.0 + 4.0)
    assert pm.energy_per_unit_j(100.0, 50) == 2.0
    assert pm.energy_delay_product(1102.0, 40.0) == pytest.approx(44080.0)
    with pytest.raises(ValueError):
        pm.energy_per_unit_j(1.0, 0)


def test_interval_validation() -> None:
    with pytest.raises(ValueError):
        pm.PowerMonitor(pm.FakeSensor(), run_id="r", node_id="d", output=Path("x"), interval_s=0)


def test_cli_with_fake_sensor(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "power.json"
    pm.main(
        [
            "--fake",
            "--interval",
            "0.05",
            "--duration",
            "0.2",
            "--output",
            str(out),
            "--run-id",
            "r",
            "--node-id",
            "d",
        ]
    )
    data = json.loads(out.read_text())
    assert data["kind"] == "power_trace" and len(data["samples"]) >= 3
    assert "mW" in capsys.readouterr().err

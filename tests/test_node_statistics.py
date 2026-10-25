from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.nodes import load_topology
from src.network import node_statistics as ns
from src.network.records import LatencyRecord
from tests import synthetic


def test_latency_record_round_trip(tmp_path: Path) -> None:
    record = synthetic.latency_record("device1", extra_targets={"device2": [1.0, None]})
    path = record.save(tmp_path / "lat.json")
    loaded = LatencyRecord.load(path)
    assert loaded == record
    ping = loaded.ping_to("edge_server")
    assert ping is not None and ping.loss_percent == 20.0 and ping.rtts == [5.1, 5.3, 4.9, 5.0]
    assert loaded.ping_to("nobody") is None


def test_compute_averages_per_node_and_target() -> None:
    records = [
        synthetic.latency_record("device1", rtts=[10.0, 20.0], available_mb=3000.0, cpu=4.0),
        synthetic.latency_record(
            "device1",
            rtts=[30.0, None],
            available_mb=3200.0,
            cpu=6.0,
            extra_targets={"device2": [2.0, 4.0]},
        ),
        synthetic.latency_record("device2", rtts=[None, None], available_mb=1000.0, cpu=50.0),
    ]
    averages = ns.compute_averages(records)
    d1 = averages["device1"]
    assert d1["samples"] == 2
    assert d1["average_latency_ms"] == pytest.approx(20.0)
    assert d1["loss_percent"] == pytest.approx(25.0)
    assert d1["available_memory_mb"] == pytest.approx(3100.0)
    assert d1["cpu_usage_percent"] == pytest.approx(5.0)
    assert d1["targets"]["device2"]["average_latency_ms"] == pytest.approx(3.0)
    d2 = averages["device2"]
    assert d2["average_latency_ms"] is None and d2["loss_percent"] == 100.0


def test_cli_writes_averages_and_updates_topology(tmp_path: Path, repo_root: Path) -> None:
    lat = tmp_path / "latency"
    lat.mkdir()
    synthetic.latency_record("device1").save(lat / "device1.json")
    synthetic.latency_record("device3", available_mb=2000.0, cpu=12.5).save(lat / "device3.json")
    nodes = tmp_path / "nodes.json"
    nodes.write_text((repo_root / "nodes.example.json").read_text())
    out = lat / "final_latencies_averages.json"
    ns.main([str(lat), str(out), "--nodes", str(nodes), "--run-id", "run-7"])
    averages = json.loads(out.read_text())
    assert set(averages) == {"device1", "device3"}
    topology = load_topology(nodes)
    stats = topology.get("device3").statistics
    assert stats is not None and stats.run_id == "run-7" and stats.cpu_usage_percent == 12.5
    assert topology.get("device2").statistics is None  # untouched
    # The output file itself is skipped when the directory is read again.
    assert len(ns.load_records(lat)) == 2


def test_cli_without_records(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no latency records"):
        ns.main([str(tmp_path / "empty"), str(tmp_path / "out.json")])

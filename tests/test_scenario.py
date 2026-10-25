from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.common.clock import FakeClock
from src.common.start_signal import wait_until
from src.orchestration import geo
from src.orchestration import scenario as sc

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((REPO / "examples" / "scenarios").glob("*.yaml"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_scenarios_validate(path: Path) -> None:
    scenario = sc.load_scenario_file(path)
    assert scenario["name"] == path.stem
    assert scenario["pipeline"] in ("experiment", "benchmarking")
    assert scenario["edge_server"]["policy"] in ("RR", "SJF")  # defaults filled in


def test_defaults_and_validation_errors() -> None:
    minimal = sc.validate({"name": "m", "model": "fake"})
    assert minimal["repetitions"] == 1 and minimal["warmup"] == 1 and minimal["seed"] == 0
    assert minimal["edge_server"] == {
        "policy": "SJF",
        "nominal_frequency": True,
        "profile_capacity": True,
        "port": 5300,
    }
    assert minimal["monitoring"]["interval_s"] == 1.0
    with pytest.raises(sc.ScenarioError, match="name"):
        sc.validate({"name": "bad name", "model": "fake"})
    with pytest.raises(sc.ScenarioError, match="load"):
        sc.validate({"name": "m", "model": "fake", "load": {"default": {"mode": "random"}}})
    with pytest.raises(sc.ScenarioError, match="unexpected"):
        sc.validate({"name": "m", "model": "fake", "unexpected": 1})


def test_per_node_settings_overlay_default() -> None:
    section = {"default": {"delay_ms": 10, "loss_percent": 1}, "device2": {"delay_ms": 30}}
    assert sc.for_node(section, "device1") == {"delay_ms": 10, "loss_percent": 1}
    assert sc.for_node(section, "device2") == {"delay_ms": 30, "loss_percent": 1}
    assert sc.for_node(None, "device1") == {}


def test_sweep_expands_into_variants() -> None:
    base = sc.validate(
        {
            "name": "s",
            "model": "fake",
            "sweep": {"path": "network.default.added_rtt_ms", "values": [0, 10]},
        }
    )
    variants = sc.expand_sweep(base)
    assert [v["name"] for v in variants] == ["s__added_rtt_ms-0", "s__added_rtt_ms-10"]
    assert variants[1]["network"]["default"]["added_rtt_ms"] == 10
    assert all("sweep" not in v for v in variants)
    assert sc.expand_sweep(sc.validate({"name": "n", "model": "fake"}))[0]["name"] == "n"


def test_store_versions_and_replay(tmp_path: Path) -> None:
    store = sc.ScenarioStore(tmp_path / "scenarios")
    v1 = store.save({"name": "exp", "model": "fake", "repetitions": 2})
    again = store.save({"name": "exp", "model": "fake", "repetitions": 2})
    assert v1.version == 1 and again.version == 1 and again.sha256 == v1.sha256
    v2 = store.save({"name": "exp", "model": "fake", "repetitions": 3})
    assert v2.version == 2 and v2.sha256 != v1.sha256
    assert store.names() == ["exp"] and [v.version for v in store.versions("exp")] == [1, 2]
    loaded, version = store.load("exp", 1)
    assert loaded["repetitions"] == 2 and version.version == 1
    latest, version = store.load("exp")
    assert latest["repetitions"] == 3 and version.version == 2
    assert sc.content_hash(loaded) == v1.sha256
    with pytest.raises(sc.ScenarioError):
        store.load("missing")
    with pytest.raises(sc.ScenarioError):
        store.load("exp", 9)
    stored = json.loads(v2.path.read_text())
    assert stored["_version"] == 2 and stored["_sha256"] == v2.sha256


def test_yaml_and_json_files(tmp_path: Path) -> None:
    (tmp_path / "s.yaml").write_text(yaml.safe_dump({"name": "y", "model": "fake"}))
    (tmp_path / "s.json").write_text(json.dumps({"name": "j", "model": "fake"}))
    assert sc.load_scenario_file(tmp_path / "s.yaml")["name"] == "y"
    assert sc.load_scenario_file(tmp_path / "s.json")["name"] == "j"
    (tmp_path / "bad.yaml").write_text("- a list")
    with pytest.raises(sc.ScenarioError):
        sc.load_scenario_file(tmp_path / "bad.yaml")


def test_map_rule_distance_to_latency() -> None:
    assert geo.calculate_latency(0.0) == 5.0
    assert geo.calculate_latency(10_000.0) == 15.0
    assert geo.calculate_latency(45_000.0) == 50.0
    assert geo.calculate_latency(100_000.0) == 50.0
    # Athens centre to Piraeus, about 8 km
    d = geo.calculate_distance(37.9838, 23.7275, 37.9475, 23.6361)
    assert 8_500 < d < 9_500
    assert geo.calculate_distance(0, 0, 0, 0) == 0.0
    assert geo.calculate_distance(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)


def test_start_signal_waits_until_t0() -> None:
    clock = FakeClock(start_ns=0, start_time=100.0)
    origin = wait_until(102.5, clock)
    assert origin.wall_time_s >= 102.5 and origin.wall_time_s < 102.51
    assert origin.perf_counter_ns >= 2_500_000_000
    past = wait_until(50.0, clock)
    assert past.wall_time_s == clock.time()
    assert wait_until(None, clock).wall_time_s == clock.time()
    with pytest.raises(ValueError, match="refusing"):
        wait_until(clock.time() + 10_000, clock)

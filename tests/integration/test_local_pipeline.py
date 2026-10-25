"""The whole pipeline in local mode: every node simulated on this machine."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.benchmarking import benchmark_results as br
from src.benchmarking import result_aggregation
from src.common.runs import RunJournal
from src.orchestration import experiment, local_mode
from src.orchestration import scenario as sc
from src.orchestration.experiment import build_context, plan, run_pipeline

REPO = Path(__file__).resolve().parents[1].parent

SCENARIO: dict[str, Any] = {
    "name": "it_local",
    "pipeline": "experiment",
    "model": "fake",
    "devices": None,
    "repetitions": 2,
    "warmup": 0,
    "seed": 5,
    "settle_s": 0,
    "start_lead_s": 0.3,
    "edge_server": {"policy": "SJF", "nominal_frequency": True, "profile_capacity": True},
    "inference": {"enabled": True, "partition_points": [0, 2, 5], "repetitions": 1},
    "network": {
        "default": {"delay_ms": 10, "jitter_ms": 2, "loss_percent": 0.5},
        "device2": {"variation": {"mode": "uniform", "min_ms": 5, "max_ms": 15, "period_s": 0.2}},
    },
    "load": {"default": {"mode": "fixed", "cpu_percent": 10, "memory_percent": 1, "cycle_s": 0.5}},
    "heterogeneity": {"device2": {"max_freq_mhz": 1200, "memory_limit_mb": 512, "threads": 2}},
    "monitoring": {
        "enabled": True,
        "interval_s": 0.1,
        "power": True,
        "power_interval_s": 0.05,
        "latency_pings": 2,
    },
    "optimisation": {
        "enabled": True,
        "algorithm": "greedy",
        "objective": "total_delay",
        "initial_scheme": "50%",
    },
}


@pytest.fixture
def local_root(tmp_path: Path) -> Path:
    return tmp_path / "testbed"


def make_context(
    local_root: Path,
    overrides: dict[str, Any] | None = None,
    run_id: str = "run1",
    plan_only: bool = False,
) -> experiment.ExperimentContext:
    config, topology = local_mode.create_local_testbed(local_root, devices=2)
    scenario = sc.validate({**SCENARIO, **(overrides or {})})
    return build_context(
        scenario,
        config=config,
        topology=topology,
        local=True,
        run_id=run_id,
        nodes_path=local_root / "nodes.json",
        scenario_version={"source": "test"},
        plan_only=plan_only,
    )


def test_full_experiment_in_local_mode(local_root: Path) -> None:
    ctx = make_context(local_root)
    summary = run_pipeline(ctx)
    assert summary["status"] == "completed" and summary["teardown_failures"] == 0
    root = ctx.paths.root
    for name in (
        "metadata.json",
        "scenario.json",
        "journal.json",
        "summary.json",
        "edge_capacity.json",
        "total_averages_of_benchmark_results.json",
        "calculated_max_partition_points.json",
        "final_latencies_averages.json",
    ):
        assert (root / name).exists(), name
    metadata = json.loads((root / "metadata.json").read_text())
    assert set(metadata["nodes"]) == {"device1", "device2"} and metadata["seeds"] == {"scenario": 5}
    totals = json.loads((root / "total_averages_of_benchmark_results.json").read_text())
    assert set(totals) == {"device1", "device2"} and totals["device1"]["repetitions"] == 2
    assert set(totals["device1"]["phases"]) == {"0", "2", "5"}
    averages = json.loads((root / "final_latencies_averages.json").read_text())
    assert averages["device1"]["targets"]["device2"]["average_latency_ms"] is not None
    points = json.loads((root / "calculated_max_partition_points.json").read_text())
    assert set(points) == {"device1", "device2"}
    monitor = br.load(next((root / "monitor").glob("*.json")))
    assert isinstance(monitor, br.ResourceTrace) and len(monitor.samples) >= 3
    power = br.load(next((root / "power").glob("*.json")))
    assert isinstance(power, br.PowerTrace) and power.metadata["energy_j"] > 0
    assert (root / "merged").exists() and len(list((root / "merged").glob("*.json"))) == 2
    optimisation = next((root / "optimisation").glob("*/optimized_partition_scheme.json"))
    result = json.loads(optimisation.read_text())
    assert set(result["by_node"]) == {"device1", "device2"} and result["inputs"]["run_id"] == "run1"
    assert (optimisation.parent / "search_log.txt").exists()
    # The topology statistics were updated from the latency measurement.
    nodes = json.loads((local_root / "nodes.json").read_text())
    stats = {n["id"]: n["statistics"] for n in nodes["nodes"]}
    assert stats["device1"]["run_id"] == "run1"
    # Teardown undid every journal entry, and the dry-run tc commands pair up.
    journal = RunJournal.load(root / "journal.json")
    assert journal.entries and all(e.undone for e in journal.entries)
    log = (root / "logs" / "merged.log").read_text()
    # device1 is shaped once in the foreground; device2's variation process logs to its own file.
    assert log.count("[dry-run] sudo tc qdisc add") >= 1
    assert log.count("[dry-run] sudo tc qdisc del") == 2
    assert "freq --restore" in log and "cgroup remove" in log
    assert "teardown complete" in log
    # Nothing tagged is left running on the simulated devices.
    pgrep = subprocess.run(["pgrep", "-f", "[t]estbed-run1-"], capture_output=True, text=True)
    assert pgrep.stdout.strip() == ""
    # Device 2 ran with the configured thread count and frequency cap.
    device2_log = log
    assert "TESTBED_THREADS=2" in device2_log and "--max-mhz 1200" in device2_log
    assert (local_root / "results" / "latest").resolve() == root.resolve()


def test_benchmarking_pipeline_skips_load_and_shaping(local_root: Path) -> None:
    ctx = make_context(
        local_root,
        {
            "pipeline": "benchmarking",
            "inference": {"enabled": False},
            "monitoring": {"enabled": True, "interval_s": 0.1, "latency_pings": 1},
        },
        run_id="bench1",
    )
    summary = run_pipeline(ctx)
    assert summary["status"] == "completed"
    log = (ctx.paths.root / "logs" / "merged.log").read_text()
    assert (
        "latency_injector" not in log and "load_generator" not in log and "heterogeneity" not in log
    )
    assert "resource_monitor" in log and "latency_measurement" in log
    assert (ctx.paths.root / "total_averages_of_benchmark_results.json").exists()


def test_failure_mid_run_still_tears_down(local_root: Path) -> None:
    ctx = make_context(
        local_root, {"model": "no_such_model", "inference": {"enabled": False}}, run_id="fail1"
    )
    with pytest.raises(experiment.ExperimentAbortedError):
        run_pipeline(ctx)
    journal = RunJournal.load(ctx.paths.root / "journal.json")
    assert journal.entries and all(e.undone for e in journal.entries)
    summary = json.loads((ctx.paths.root / "summary.json").read_text())
    assert summary["status"] == "aborted" and summary["teardown_failures"] == 0
    log = (ctx.paths.root / "logs" / "merged.log").read_text()
    assert log.count("[dry-run] sudo tc qdisc del") == 2  # both devices' shaping cleared
    pgrep = subprocess.run(["pgrep", "-f", "[t]estbed-fail1-"], capture_output=True, text=True)
    assert pgrep.stdout.strip() == ""


def test_scenario_without_devices_is_refused(local_root: Path) -> None:
    ctx = make_context(local_root, {"devices": ["device9"]}, run_id="nodev")
    with pytest.raises(KeyError):
        run_pipeline(ctx)
    ctx2 = make_context(
        local_root, {"runtime": "headsplit", "inference": {"enabled": False}}, run_id="hs"
    )
    from src.orchestration.headsplit_runtime import HeadSplitError

    with pytest.raises(HeadSplitError, match="must name nodes"):
        run_pipeline(ctx2)


def test_plan_is_deterministic_for_a_scenario_version(local_root: Path) -> None:
    first = plan(make_context(local_root, run_id="planrun", plan_only=True))
    second = plan(make_context(local_root, run_id="planrun", plan_only=True))
    assert first == second
    text = "\n".join(first)
    assert "--model-args" not in text  # no model_args in the scenario: nothing is passed
    with_args = plan(
        make_context(
            local_root,
            {"model_args": {"input_elements": 64, "time_scale": 0.5}},
            run_id="planargs",
            plan_only=True,
        )
    )
    args_text = "\n".join(with_args)
    # The device profiler, the edge-server capacity tool, the inference service and the
    # client all receive the same model arguments.
    assert args_text.count("--model-args") >= 4 and "input_elements" in args_text
    with pytest.raises(experiment.ExperimentError, match="model_args"):
        make_context(local_root, {"model_args": {"bogus": 1}}, run_id="badargs", plan_only=True)
    assert (
        "train_and_run_benchmark.sh" in text
        and "latency_injector" in text
        and "load_generator" in text
    )
    assert "--start-at 0.000000" in text


def test_teardown_of_an_open_run_after_restart(local_root: Path) -> None:
    ctx = make_context(local_root, run_id="open1")
    ctx.paths.create()
    ctx.record(
        "device1", "shaping", "tc", {"kind": "tc_clear", "node_id": "device1", "interface": "wlan0"}
    )
    (ctx.paths.root / "scenario.json").write_text(json.dumps(ctx.scenario))
    handled = experiment.teardown_open_runs(ctx.config, ctx.topology, True, ctx.paths.results_dir)
    assert handled == ["open1"]
    assert all(e.undone for e in RunJournal.load(ctx.paths.journal).entries)


def test_run_full_benchmarking_script_in_local_mode(local_root: Path, tmp_path: Path) -> None:
    scenario = tmp_path / "bench.yaml"
    scenario.write_text(
        json.dumps(
            {
                **SCENARIO,
                "name": "script_bench",
                "inference": {"enabled": False},
                "monitoring": {"enabled": False},
            }
        )
    )
    env = dict(os.environ, TESTBED_PYTHON=sys.executable)
    result = subprocess.run(
        [
            "bash",
            str(REPO / "scripts" / "base_station" / "run_full_benchmarking.sh"),
            str(scenario),
            "--local",
            str(local_root),
            "--run-id",
            "script1",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    run_root = local_root / "results" / "script1"
    assert (run_root / "total_averages_of_benchmark_results.json").exists()
    assert (run_root / "calculated_max_partition_points.json").exists()
    assert (run_root / "final_latencies_averages.json").exists()
    summary = json.loads((run_root / "summary.json").read_text())
    assert summary["status"] == "completed"


def test_input_source_and_partition_check_in_local_mode(local_root: Path) -> None:
    """A numpy input on the devices and the partition-equivalence check: both reach the
    device tools, the verification file is collected, and a deviation aborts the run."""
    _config, topology = local_mode.create_local_testbed(local_root, devices=2)
    for node in topology.end_devices():
        data_dir = Path(node.workdir or ".") / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / "sample.npy", np.full(2048, 0.5, dtype=np.float32))
    overrides: dict[str, Any] = {
        "repetitions": 1,
        "inference": {"enabled": True, "partition_points": [0, 2], "repetitions": 1},
        "input": {"source": "numpy_file", "path": "sample.npy"},
        "checks": {"partition_equivalence": True, "tolerance": 1e-4},
        "monitoring": {"enabled": False},
        "load": {},
        "network": {},
        "heterogeneity": {},
        "optimisation": {"enabled": False},
    }
    ctx = make_context(local_root, overrides, run_id="inputrun")
    summary = run_pipeline(ctx)
    assert summary["status"] == "completed", summary
    verification = sorted((ctx.paths.root / "verification").glob("*_verification.json"))
    assert len(verification) == 2
    report = json.loads(verification[0].read_text())
    assert report["ok"] is True and report["input"]["source"] == "numpy_file"
    assert report["input"]["path"].endswith("data/sample.npy")
    # The profiling records (the client's phase records carry no stages and no input).
    records = [
        r
        for r in result_aggregation.load_records(ctx.paths.raw)
        if isinstance(r, br.BenchmarkResults) and r.stages
    ]
    assert records and all(r.metadata["input"]["source"] == "numpy_file" for r in records)
    log = (ctx.paths.root / "logs" / "merged.log").read_text()
    assert "partition equivalence check" in log and "--input" in log

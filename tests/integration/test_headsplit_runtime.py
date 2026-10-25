"""The HeadSplit runtime integration, against a stub of the HeadSplit scripts."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

from src.common.runs import RunJournal
from src.orchestration import headsplit_runtime as hs
from src.orchestration import local_mode
from src.orchestration import scenario as sc
from src.orchestration.experiment import build_context, run_pipeline

CONTROLLER_OUTPUT = """controller: RPC group of 5 (127.0.0.1:29500) ...
measuring device capabilities and links ...
speculating layers: [2, 5, 9]
running layer_based ...
layer_based   total   62400.0 ms  prefill   1200.5 ms  decode  1224.0 ms/token  payload   12.34 MB
  The history of Athens begins
running head_only ...
head_only     total   45100.0 ms  prefill   900.0 ms  decode   884.0 ms/token  payload   40.10 MB
running headsplit ...
headsplit     total   40000.0 ms  prefill   880.0 ms  decode   782.4 ms/token  payload   41.00 MB  \
speculation 44/50 accepted  re-placements 1

speedup vs layer-based:
  head_only    1.38 times
  headsplit    1.56 times
"""


def test_parse_controller_output() -> None:
    parsed = hs.parse_controller_output(CONTROLLER_OUTPUT)
    assert set(parsed["modes"]) == {"layer_based", "head_only", "headsplit"}
    assert parsed["modes"]["headsplit"]["total_ms"] == 40000.0
    assert parsed["modes"]["headsplit"]["speculation_accepted"] == 44
    assert parsed["modes"]["headsplit"]["replacements"] == 1
    assert parsed["modes"]["layer_based"]["speculation_accepted"] is None
    assert parsed["speedup_vs_layer_based"] == {"head_only": 1.38, "headsplit": 1.56}
    assert hs.parse_controller_output("nothing here")["modes"] == {}


def make_stub(root: Path) -> Path:
    """A fake HeadSplit checkout: the two scripts print what the real ones print."""
    scripts = root / "headsplit" / "scripts"
    scripts.mkdir(parents=True)
    worker = scripts / "run_worker.py"
    worker.write_text(
        "import argparse, sys, time\n"
        "p = argparse.ArgumentParser()\n"
        "for a in ('--name', '--rank', '--world-size', '--master-addr', '--master-port',\n"
        "          '--model'):\n"
        "    p.add_argument(a)\n"
        "a = p.parse_args()\n"
        "print(f'{a.name}: joining RPC group ({a.master_addr}:{a.master_port}) ...', flush=True)\n"
        "print(f'{a.name}: ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    controller = scripts / "run_controller.py"
    controller.write_text(
        "import argparse, sys\n"
        "p = argparse.ArgumentParser()\n"
        "for a in ('--config', '--profile', '--master-addr', '--master-port', '--prompt',\n"
        "          '--tokens', '--mode'):\n"
        "    p.add_argument(a)\n"
        "a = p.parse_args()\n"
        "assert open(a.config).read().startswith('model_name:'), 'config missing'\n"
        f"sys.stdout.write({CONTROLLER_OUTPUT!r})\n"
    )
    for script in (worker, controller):
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return root / "headsplit"


def headsplit_scenario(**overrides: Any) -> dict[str, Any]:
    base = {
        "name": "hs_local",
        "pipeline": "experiment",
        "runtime": "headsplit",
        "model": "tinyllama",
        "devices": None,
        "repetitions": 2,
        "seed": 1,
        "settle_s": 0,
        "edge_server": {"nominal_frequency": False, "profile_capacity": False},
        "headsplit": {
            "controller": "device1",
            "workers": ["device2", "device3"],
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "tokens": 50,
            "mode": "all",
        },
        "monitoring": {"enabled": True, "interval_s": 0.1, "latency_pings": 1},
        "optimisation": {"enabled": False},
    }
    base.update(overrides)
    return sc.validate(base)


def test_headsplit_runtime_in_local_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = make_stub(tmp_path)
    root = tmp_path / "testbed"
    monkeypatch.setenv("TESTBED_HEADSPLIT_DIR", str(stub))
    monkeypatch.setenv("TESTBED_HEADSPLIT_COMMIT", "abcdef1")
    config, topology = local_mode.create_local_testbed(root, devices=3)
    ctx = build_context(
        headsplit_scenario(),
        config=config,
        topology=topology,
        local=True,
        run_id="hs1",
        nodes_path=root / "nodes.json",
    )
    summary = run_pipeline(ctx)
    assert summary["status"] == "completed" and summary["teardown_failures"] == 0
    results = json.loads((ctx.paths.root / "headsplit" / "results.json").read_text())
    assert results["controller"] == "device1" and results["workers"] == ["device2", "device3"]
    assert len(results["repetitions"]) == 2
    assert results["mean_total_ms"]["headsplit"] == 40000.0
    assert (
        (ctx.paths.root / "headsplit" / "headsplit_config.yaml")
        .read_text()
        .startswith("model_name:")
    )
    assert (ctx.paths.root / "headsplit" / "controller_rep1.log").exists()
    log = (ctx.paths.root / "logs" / "merged.log").read_text()
    assert "run_worker.py --name worker1 --rank 1 --world-size 3" in log
    assert "run_controller.py --config" in log and "--tokens 50 --mode all" in log
    journal = RunJournal.load(ctx.paths.root / "journal.json")
    assert all(e.undone for e in journal.entries)
    # The per-layer profiling stages did not run: this is an external-runtime experiment.
    assert "train_and_run_benchmark.sh" not in log
    assert "latency_measurement" in log and "resource_monitor" in log


def test_headsplit_refused_without_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TESTBED_HEADSPLIT_COMMIT", raising=False)
    root = tmp_path / "testbed"
    config, topology = local_mode.create_local_testbed(root, devices=3)
    bad = headsplit_scenario(headsplit={"controller": "device1", "workers": ["device9"]})
    ctx = build_context(bad, config=config, topology=topology, local=True, run_id="hs2")
    with pytest.raises(hs.HeadSplitError, match="must name nodes"):
        run_pipeline(ctx)


def test_commands_and_config_text(tmp_path: Path) -> None:
    root = tmp_path / "testbed"
    config, topology = local_mode.create_local_testbed(root, devices=2)
    ctx = build_context(
        headsplit_scenario(
            headsplit={"controller": "device1", "workers": ["device2"], "worker_memory_bytes": 1024}
        ),
        config=config,
        topology=topology,
        local=True,
        run_id="hs3",
    )
    text = hs.headsplit_config(ctx, [topology.get("device2")], "M", "artifacts/profile.json")
    data = __import__("yaml").safe_load(text)
    assert data["devices"] == [{"name": "worker1", "host": "127.0.0.1", "memory_bytes": 1024}]
    assert data["speculation"] == {"epsilon": 0.05, "alpha_min": 0.7}
    command = hs.worker_command(
        ctx, topology.get("device2"), "worker1", 1, 2, "127.0.0.1", 29500, "M"
    )
    assert command.endswith(
        "scripts/run_worker.py --name worker1 --rank 1 --world-size 2 "
        "--master-addr 127.0.0.1 --master-port 29500 --model M"
    )
    controller = hs.controller_command(
        ctx,
        topology.get("device1"),
        "/c.yaml",
        "p.json",
        "127.0.0.1",
        29500,
        10,
        "headsplit",
        "hello",
        ["--x"],
    )
    assert "--prompt hello --x" in controller

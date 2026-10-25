"""Runtime plug-ins: the registry, and HeadSplit's commands, checks and actions
against a fake transport and a fake command runner (no device, no HeadSplit)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from src.common.nodes import load_topology
from src.common.runner import CommandResult, FakeRunner, substring
from src.config import load_config
from src.network.ssh_executor import Executor
from src.network.transport import FakeTransport, ScriptedOutcome
from src.orchestration import runtimes
from src.orchestration import scenario as sc
from src.orchestration.experiment import ExperimentError, build_context
from src.orchestration.runtimes import ActionSession
from src.orchestration.runtimes import headsplit as hs

REPO = Path(__file__).resolve().parents[1]
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def example_environment(tmp_path: Path, **extra: str) -> tuple[Any, Any]:
    environ = {
        "TESTBED_HEADSPLIT_COMMIT": COMMIT,
        "TESTBED_BS_ROOT": str(tmp_path / "bs"),
        "TESTBED_BS_MODELS_DIR": str(tmp_path / "bs" / "models"),
        "TESTBED_MIRROR_DIR": str(tmp_path / "bs" / "mirror"),
        **extra,
    }
    config = load_config(REPO / "config.example.yaml", environ=environ)
    topology = load_topology(REPO / "nodes.example.json")
    return config, topology


# --- registry ------------------------------------------------------------------------------------


def test_builtin_runtimes_and_catalogue_description() -> None:
    assert runtimes.names() == ["headsplit", "testbed"]
    testbed = runtimes.get_runtime("testbed")
    assert not testbed.external and testbed.schema == {} and testbed.actions() == []
    headsplit = runtimes.get_runtime("headsplit")
    assert headsplit.external and "controller" in headsplit.schema["properties"]
    assert [a.name for a in headsplit.actions()] == ["install", "profile", "checkpoint"]
    described = {d["name"]: d for d in runtimes.describe()}
    assert described["headsplit"]["actions"][0]["schema"]["properties"]["nodes"]["x-nnpm"] == {
        "widget": "nodes"
    }
    with pytest.raises(KeyError, match="unknown runtime"):
        runtimes.get_runtime("no-such-runtime")


def test_plugin_runtime_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(runtimes.MODULES_ENV, "tests.toy_runtime_plugin")
    runtimes.reset_for_tests()
    try:
        assert "toy" in runtimes.names()
        # The plug-in's block joins the scenario schema, so a scenario can carry it.
        assert "toy" in sc.load_schema()["properties"]
        scenario = sc.validate({"name": "t", "model": "fake", "runtime": "toy", "toy": {}})
        assert scenario["toy"] == {"knob": 1}
        config, topology = example_environment(tmp_path)
        ctx = build_context(scenario, config=config, topology=topology, local=False, plan_only=True)
        runtimes.get_runtime("toy").run(ctx)
        assert ctx.run_id in runtimes.get_runtime("toy").ran  # type: ignore[attr-defined]
    finally:
        monkeypatch.delenv(runtimes.MODULES_ENV)
        runtimes.reset_for_tests()


def test_unknown_runtime_is_refused_when_the_context_is_built(tmp_path: Path) -> None:
    config, topology = example_environment(tmp_path)
    scenario = sc.validate({"name": "t", "model": "fake", "runtime": "nothing"})
    with pytest.raises(ExperimentError, match="unknown runtime 'nothing'"):
        build_context(scenario, config=config, topology=topology, local=False, plan_only=True)


# --- HeadSplit commands ---------------------------------------------------------------------


def headsplit_scenario(**overrides: Any) -> dict[str, Any]:
    base = {
        "name": "hs",
        "runtime": "headsplit",
        "model": "tinyllama",
        "devices": None,
        "headsplit": {"controller": "device1", "workers": ["device2", "device3"]},
        "optimisation": {"enabled": False},
    }
    base.update(overrides)
    return sc.validate(base)


def test_real_devices_use_the_environment_provisioning_creates(tmp_path: Path) -> None:
    """Provisioning creates <headsplit_dir>/.venv; the worker and controller commands must
    start that interpreter (a wrong path fails on every device, and nowhere in local mode)."""
    config, topology = example_environment(tmp_path)
    ctx = build_context(
        headsplit_scenario(), config=config, topology=topology, local=False, plan_only=True
    )
    worker = topology.get("device2")
    command = hs.worker_command(ctx, worker, "worker1", 1, 3, "192.168.0.11", 29500, "M")
    assert command.startswith(
        "cd /home/pi/testbed/headsplit && /home/pi/testbed/headsplit/.venv/bin/python "
        "scripts/run_worker.py"
    )
    controller = hs.controller_command(
        ctx,
        topology.get("device1"),
        "/c.yaml",
        "p.json",
        "192.168.0.11",
        29500,
        10,
        "headsplit",
        None,
        [],
        json_path="/r/out.json",
    )
    assert "/.venv/bin/python scripts/run_controller.py" in controller
    assert controller.endswith("--tokens 10 --mode headsplit --json /r/out.json")
    assert hs.checkpoint_dirname("TinyLlama/TinyLlama-1.1B-Chat-v1.0") == (
        "TinyLlama__TinyLlama-1.1B-Chat-v1.0"
    )


def test_json_results_and_summary() -> None:
    data = {
        "modes": {
            "layer_based": {
                "total_ms": 60000,
                "prefill_ms": 1000,
                "decode_ms_per_token": 1180,
                "payload_mb": 12.0,
                "per_token_ms": [1180.0] * 3,
            },
            "headsplit": {
                "total_ms": 40000,
                "prefill_ms": 900,
                "decode_ms_per_token": 780,
                "payload_mb": 41.0,
                "speculation_accepted": 44,
                "speculation_attempts": 50,
                "replacements": 1,
            },
        },
        "speedup_vs_layer_based": {"headsplit": 1.5},
        "rtt_ms": {"worker1": 9.5},
    }
    parsed = hs.parse_controller_json(data)
    assert parsed["modes"]["headsplit"]["speculation_accepted"] == 44
    assert parsed["modes"]["layer_based"]["per_token_ms"] == [1180.0] * 3
    assert parsed["rtt_ms"] == {"worker1": 9.5}
    results = {
        "model": "M",
        "tokens": 50,
        "mode": "all",
        "controller": "device1",
        "workers": ["device2"],
        "json_results": True,
        "repetitions": [dict(parsed, repetition=0), dict(parsed, repetition=1)],
        "mean_total_ms": {"headsplit": 40000.0, "layer_based": 60000.0},
    }
    summary = hs.summarise_results(results)
    assert summary["kind"] == "headsplit" and summary["repetitions"] == 2
    rows = {row["mode"]: row for row in summary["modes"]}
    assert rows["headsplit"]["acceptance"] == pytest.approx(0.88)
    assert rows["headsplit"]["speedup_vs_layer_based"] == 1.5
    assert rows["layer_based"]["speedup_vs_layer_based"] is None
    assert [row["mode"] for row in summary["modes"]] == ["layer_based", "headsplit"]


# --- HeadSplit checks and actions against fakes ---------------------------------------------------


def make_session(
    tmp_path: Path, transport: FakeTransport, runner: FakeRunner, **extra: str
) -> tuple[ActionSession, list[str]]:
    config, topology = example_environment(tmp_path, **extra)
    lines: list[str] = []
    executor = Executor(transport, subscribers=[lines.append])
    session = ActionSession(
        config, topology, executor, local=False, runner=runner, log=lines.append
    )
    return session, lines


def test_checks_classify_the_device_state(tmp_path: Path) -> None:
    transport = (
        FakeTransport(default=ScriptedOutcome(1, []))
        .on("device1", "rev-parse", ScriptedOutcome(0, [COMMIT]))
        .on("device1", "-c", ScriptedOutcome(0, ["torch 2.1.2 transformers 4.46.3"]))
        .on("device1", "profile.json", ScriptedOutcome(0, []))
        .on("device2", "rev-parse", ScriptedOutcome(0, ["missing"]))
    )
    session, _ = make_session(tmp_path, transport, FakeRunner(strict=False))
    runtime = runtimes.get_runtime("headsplit")
    ready = {c.name: c for c in runtime.check_node(session, session.topology.get("device1"), {})}
    assert ready["checkout"].status == "ok" and ready["checkout"].detail == COMMIT[:12]
    assert ready["environment"].status == "ok" and "4.46.3" in ready["environment"].detail
    assert ready["profile"].status == "ok"
    assert (
        ready["checkpoint"].status == "warn"
        and "TinyLlama__TinyLlama" in ready["checkpoint"].detail
    )
    bare = {c.name: c for c in runtime.check_node(session, session.topology.get("device2"), {})}
    assert bare["checkout"].status == "fail" and bare["environment"].status == "fail"
    assert runtime.check_node(session, session.topology.get("edge-server"), {}) == []
    # Every check ran through the transport on the right node.
    assert all(node == "device1" or node == "device2" for node, _ in transport.calls)


def test_install_action_runs_device_setup_from_the_mirror(tmp_path: Path) -> None:
    transport = FakeTransport()
    session, lines = make_session(tmp_path, transport, FakeRunner(strict=False))
    runtimes.get_runtime("headsplit").perform("install", {"nodes": ["device2"]}, session)
    commands = [c for n, c in transport.calls if n == "device2"]
    assert len(commands) == 1
    assert commands[0].startswith("bash /home/pi/testbed/code/scripts/provision/device_setup.sh")
    assert (
        "--mirror http://10.0.0.1:8080 --headsplit-dir /home/pi/testbed/headsplit "
        f"--headsplit-commit {COMMIT} headsplit" in commands[0]
    )
    assert "--dry-run" not in commands[0]
    assert any("install HeadSplit" in line for line in lines)


def test_install_needs_a_pinned_commit(tmp_path: Path) -> None:
    config = load_config(
        REPO / "config.example.yaml",
        environ={"TESTBED_BS_ROOT": str(tmp_path), "TESTBED_HEADSPLIT_COMMIT": "null"},
    )
    topology = load_topology(REPO / "nodes.example.json")
    session = ActionSession(
        config, topology, Executor(FakeTransport()), runner=FakeRunner(strict=False)
    )
    with pytest.raises(hs.HeadSplitError, match="headsplit_commit"):
        runtimes.get_runtime("headsplit").perform("install", {}, session)


def test_profile_action_builds_on_the_edge_server_and_copies_to_the_controller(
    tmp_path: Path,
) -> None:
    (tmp_path / "bs" / "mirror" / "src" / "headsplit.git").mkdir(parents=True)
    checkout = tmp_path / "bs" / "headsplit"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "scripts" / "profile_offline.py").write_text("# stub\n")
    (checkout / ".venv" / "bin").mkdir(parents=True)
    (checkout / ".venv" / "bin" / "python").write_text("")

    def write_profile(argv: Sequence[str]) -> CommandResult:
        out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
        if out is None:  # the bash -c wrapper: the arguments are inside the command string
            text = argv[-1]
            out = Path(text.split("--out ")[1].split()[0])
        out.write_text(json.dumps({"alpha": [0.9, 0.8]}))
        return CommandResult(tuple(argv), 0, "profiled\n")

    runner = FakeRunner(strict=False).on(substring("profile_offline.py"), write_profile)
    transport = FakeTransport()
    session, lines = make_session(tmp_path, transport, runner)
    runtime = runtimes.get_runtime("headsplit")
    with pytest.raises(hs.HeadSplitError, match="allow_upstream"):
        runtime.perform("profile", {"controller": "device1"}, session)
    runtime.perform(
        "profile", {"controller": "device1", "allow_upstream": True, "chunks": 4}, session
    )
    commands = runner.commands()
    assert any("git -C" in c and "fetch" in c for c in commands) or any(
        "git clone" in c for c in commands
    )
    assert any("checkout --quiet " + COMMIT in c for c in commands)
    assert any(
        "profile_offline.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --chunks 4" in c
        for c in commands
    )
    assert ("device1", "/home/pi/testbed/headsplit/artifacts/profile.json") in transport.files
    assert json.loads(
        transport.files[("device1", "/home/pi/testbed/headsplit/artifacts/profile.json")]
    ) == {"alpha": [0.9, 0.8]}
    assert any("[edge-server][stdout] profiled" in line for line in lines)


def test_checkpoint_action_copies_and_verifies(tmp_path: Path) -> None:
    source = tmp_path / "bs" / "models" / "TinyLlama__TinyLlama-1.1B-Chat-v1.0"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "SHA256SUMS").write_text("x  config.json\n")
    runner = FakeRunner(strict=False)
    transport = FakeTransport()
    session, _ = make_session(tmp_path, transport, runner)
    runtime = runtimes.get_runtime("headsplit")
    runtime.perform("checkpoint", {"nodes": ["device3"]}, session)
    rsync = [c for c in runner.commands() if c.startswith("rsync")]
    assert len(rsync) == 1
    assert rsync[0].endswith(
        f"{source}/ pi@10.0.0.13:/home/pi/testbed/models/TinyLlama__TinyLlama-1.1B-Chat-v1.0/"
    )
    assert "-o BatchMode=yes" in rsync[0]
    remote = [c for n, c in transport.calls if n == "device3"]
    assert remote[0] == "mkdir -p /home/pi/testbed/models/TinyLlama__TinyLlama-1.1B-Chat-v1.0"
    assert remote[1].endswith("&& sha256sum -c --quiet SHA256SUMS")
    with pytest.raises(hs.HeadSplitError, match="SHA256SUMS"):
        runtime.perform("checkpoint", {"model": "other/model"}, session)

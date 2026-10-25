from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.nodes import EDGE_SERVER_ROLE, END_DEVICE_ROLE, Node, Topology, save_topology
from src.common.process_tags import TaggedProcess
from src.config import load_config
from src.network import ssh_executor as se
from src.network.transport import (
    AuthenticationError,
    FakeTransport,
    LocalTransport,
    ScriptedOutcome,
    UnreachableHostError,
)


def nodes(*ids: str) -> list[Node]:
    return [
        Node(id=i, role=END_DEVICE_ROLE, ip=f"192.0.2.{10 + k}", lat=0, lon=0)
        for k, i in enumerate(ids)
    ]


def test_success_merges_tagged_lines_into_log_and_subscribers(tmp_path: Path) -> None:
    transport = FakeTransport().on("a", "", ScriptedOutcome(0, ["a1", "a2"], ["warn"]))
    transport.on("b", "", ScriptedOutcome(0, ["b1"]))
    seen: list[str] = []
    executor = se.Executor(
        transport, log_path=tmp_path / "logs" / "merged.log", subscribers=[seen.append]
    )
    outcomes = executor.run_on_nodes(nodes("a", "b"), "do work")
    assert outcomes["a"].ok and outcomes["b"].ok
    log = (tmp_path / "logs" / "merged.log").read_text().splitlines()
    assert any(line.endswith("[a][stdout] a1") for line in log)
    assert any(line.endswith("[a][stderr] warn") for line in log)
    assert any(line.endswith("[b][stdout] b1") for line in log)
    assert any("[a][exec] do work" in line for line in log)
    assert any("[a][exit] status 0" in line for line in log)
    assert len(seen) == len(log)


@pytest.mark.parametrize(
    ("outcome", "kind"),
    [
        (ScriptedOutcome(raise_error=AuthenticationError), "authentication"),
        (ScriptedOutcome(raise_error=UnreachableHostError), "unreachable"),
        (ScriptedOutcome(2, stderr_lines=["no such file"]), "command"),
    ],
)
def test_failure_classes_abort_the_experiment(
    outcome: ScriptedOutcome, kind: str, tmp_path: Path
) -> None:
    transport = FakeTransport().on("bad", "", outcome)
    executor = se.Executor(transport, log_path=tmp_path / "merged.log")
    with pytest.raises(se.ExperimentAbortedError) as exc:
        executor.run_on_nodes(nodes("good", "bad"), "cmd")
    assert [f.kind for f in exc.value.failures] == [kind]
    if kind == "command":
        assert "exit status 2: no such file" in exc.value.failures[0].message
    assert exc.value.outcomes["good"].ok
    assert (tmp_path / "merged.log").exists()  # partial log preserved


def test_abort_cancels_pending_and_running_commands() -> None:
    transport = FakeTransport().on("bad", "", ScriptedOutcome(1, stderr_lines=["fail"]))
    transport.on("slow", "", ScriptedOutcome(0, delay_s=5))
    executor = se.Executor(transport, max_workers=2)
    with pytest.raises(se.ExperimentAbortedError) as exc:
        executor.run_on_nodes(nodes("bad", "slow", "later"), "cmd")
    outcomes = exc.value.outcomes
    assert outcomes["slow"].cancelled
    assert outcomes["later"].cancelled
    assert [f.kind for f in exc.value.failures] == ["command"]


def test_no_abort_returns_every_outcome() -> None:
    transport = FakeTransport().on("bad", "", ScriptedOutcome(raise_error=UnreachableHostError))
    executor = se.Executor(transport)
    outcomes = executor.run_on_nodes(nodes("ok", "bad"), "cmd", abort_on_failure=False)
    assert (
        outcomes["ok"].ok
        and outcomes["bad"].exit_status == 255
        and "scripted failure" in outcomes["bad"].stderr
    )


def test_run_on_node_check_and_per_node_command() -> None:
    transport = FakeTransport().on(None, "exit 4", ScriptedOutcome(4))
    executor = se.Executor(transport)
    with pytest.raises(se.ExperimentAbortedError):
        executor.run_on_node(nodes("a")[0], "exit 4")
    executor.run_on_nodes(nodes("a", "b"), "", per_node_command=lambda n: f"echo {n.id}")
    assert ("a", "echo a") in transport.calls and ("b", "echo b") in transport.calls


def test_tagged_process_helpers() -> None:
    process = TaggedProcess("run1", "monitor", "/tmp/run1")
    transport = FakeTransport().on(None, "setsid nohup", ScriptedOutcome(0, ["4242"]))
    transport.on(None, "echo running; else echo stopped", ScriptedOutcome(0, ["stopped"]))
    executor = se.Executor(transport)
    n = nodes("a")[0]
    assert (
        executor.start_tagged(n, process, "python3 -m src.testbed.resource_monitor", {"K": "v"})
        == 4242
    )
    assert executor.stop_tagged(n, process) is True
    assert executor.is_running(n, process) is False
    assert any("K=v" in c for _, c in transport.calls)
    broken = FakeTransport().on(None, "setsid nohup", ScriptedOutcome(0, ["not a pid"]))
    with pytest.raises(se.ExperimentAbortedError, match="no PID"):
        se.Executor(broken).start_tagged(n, process, "x")


def test_resolve_nodes_by_id_and_octet() -> None:
    config = load_config(environ={"TESTBED_BASE_IP_PREFIX": "192.0.2."})
    topology = Topology(
        [Node(id="edge", role=EDGE_SERVER_ROLE, ip="192.0.2.1", lat=0, lon=0), *nodes("d1", "d2")]
    )
    resolved = se.resolve_nodes(topology, config, ["d1", "11"])
    assert [n.id for n in resolved] == ["d1", "d2"]
    with pytest.raises(SystemExit):
        se.resolve_nodes(topology, config, ["99"])


def test_cli_local_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    topology = Topology(
        [
            Node(
                id="d1",
                role=END_DEVICE_ROLE,
                ip="192.0.2.11",
                lat=0,
                lon=0,
                transport="local",
                workdir=str(tmp_path / "d1"),
            ),
            Node(
                id="d2",
                role=END_DEVICE_ROLE,
                ip="192.0.2.12",
                lat=0,
                lon=0,
                transport="local",
                workdir=str(tmp_path / "d2"),
            ),
        ]
    )
    nodes_file = tmp_path / "nodes.json"
    save_topology(topology, nodes_file)
    config_file = tmp_path / "config.yaml"
    config_file.write_text("ssh_username: pi\n")
    with pytest.raises(SystemExit) as exc:
        se.main(
            [
                "echo hello $TESTBED_NODE_ID",
                "d1",
                "d2",
                "--nodes-file",
                str(nodes_file),
                "--config",
                str(config_file),
                "--log",
                str(tmp_path / "m.log"),
            ]
        )
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "[d1][stdout] hello d1" in out and "[d2][stdout] hello d2" in out
    with pytest.raises(SystemExit) as exc2:
        se.main(["exit 7", "d1", "--nodes-file", str(nodes_file), "--config", str(config_file)])
    assert exc2.value.code == 1


def test_make_transport_picks_local_for_local_nodes(tmp_path: Path) -> None:
    config = load_config(environ={})
    topology = Topology(
        [
            Node(
                id="d1",
                role=END_DEVICE_ROLE,
                ip="192.0.2.11",
                lat=0,
                lon=0,
                transport="local",
                workdir=str(tmp_path),
            )
        ]
    )
    assert isinstance(se.make_transport(config, topology), LocalTransport)
    ssh_topology = Topology(nodes("d1"))
    assert not isinstance(se.make_transport(config, ssh_topology), LocalTransport)
    json.dumps(topology.to_dict())

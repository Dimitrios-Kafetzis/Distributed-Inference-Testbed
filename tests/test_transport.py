from __future__ import annotations

import threading
import time
from pathlib import Path

import paramiko
import pytest

from src.common.nodes import END_DEVICE_ROLE, Node
from src.config import load_config
from src.network.transport import (
    AuthenticationError,
    FakeTransport,
    LocalTransport,
    ParamikoTransport,
    ScriptedOutcome,
    UnreachableHostError,
)


def node(node_id: str = "d1", workdir: str | None = None) -> Node:
    return Node(id=node_id, role=END_DEVICE_ROLE, ip="192.0.2.11", lat=0, lon=0, workdir=workdir)


def test_local_transport_streams_and_reports_status(tmp_path: Path) -> None:
    transport = LocalTransport({"d1": tmp_path / "d1"})
    lines: list[tuple[str, str]] = []
    outcome = transport.run(
        node(),
        "echo out1; echo err1 >&2; echo out2; echo $TESTBED_NODE_ID; echo $X; exit 3",
        on_line=lambda s, line: lines.append((s, line)),
        env={"X": "y z"},
    )
    assert outcome.exit_status == 3 and not outcome.ok and not outcome.cancelled
    assert outcome.stdout.splitlines() == ["out1", "out2", "d1", "y z"]
    assert outcome.stderr == "err1"
    assert ("stdout", "out1") in lines and ("stderr", "err1") in lines
    assert (tmp_path / "d1").is_dir()


def test_local_transport_cancel_and_timeout(tmp_path: Path) -> None:
    transport = LocalTransport({"d1": tmp_path / "d1"})
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.perf_counter()
    outcome = transport.run(node(), "sleep 10", cancel=cancel)
    assert outcome.cancelled and outcome.exit_status == 255
    assert time.perf_counter() - started < 5
    timed = transport.run(node(), "sleep 10", timeout=0.3)
    assert timed.cancelled and "timed out" in timed.stderr


def test_local_transport_fetch_and_workdir_from_node(tmp_path: Path) -> None:
    transport = LocalTransport({})
    n = node(workdir=str(tmp_path / "w"))
    transport.run(n, "echo data > result.json")
    local = transport.fetch(n, "result.json", tmp_path / "out" / "d1_result.json")
    assert local.read_text().strip() == "data"
    with pytest.raises(FileNotFoundError):
        transport.fetch(n, "missing.json", tmp_path / "x")
    with pytest.raises(UnreachableHostError):
        LocalTransport({}).run(node(), "true")


def test_fake_transport_scripts_and_records() -> None:
    transport = FakeTransport().on("d1", "probe", ScriptedOutcome(0, ['{"a": 1}']))
    transport.on(None, "fail", ScriptedOutcome(2, stderr_lines=["boom"]))
    transport.on("d2", "auth", ScriptedOutcome(raise_error=AuthenticationError))
    assert transport.run(node("d1"), "python -m probe").stdout == '{"a": 1}'
    assert transport.run(node("d9"), "fail now").exit_status == 2
    with pytest.raises(AuthenticationError):
        transport.run(node("d2"), "auth")
    assert transport.calls == [("d1", "python -m probe"), ("d9", "fail now"), ("d2", "auth")]
    cancel = threading.Event()
    cancel.set()
    slow = FakeTransport().on(None, "", ScriptedOutcome(delay_s=5))
    assert slow.run(node(), "x", cancel=cancel).cancelled


class FailingClient:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def connect(self, *args: object, **kwargs: object) -> None:
        raise self.error

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (paramiko.AuthenticationException("bad key"), AuthenticationError),
        (
            paramiko.ssh_exception.NoValidConnectionsError(
                {("192.0.2.11", 22): OSError("refused")}
            ),
            UnreachableHostError,
        ),
        (TimeoutError(), UnreachableHostError),
        (OSError("network unreachable"), UnreachableHostError),
        (paramiko.SSHException("banner"), UnreachableHostError),
    ],
)
def test_paramiko_transport_classifies_connection_errors(
    error: BaseException, expected: type[Exception]
) -> None:
    config = load_config(environ={})
    transport = ParamikoTransport(config, client_factory=lambda: FailingClient(error))
    with pytest.raises(expected):
        transport.run(node(), "true")

from __future__ import annotations

import io
import sys

import pytest

from src.common.runner import (
    CommandError,
    CommandResult,
    DryRunRunner,
    FakeRunner,
    SubprocessRunner,
    contains,
    fail,
    ok,
    prefix,
)


def test_subprocess_runner_runs_and_captures() -> None:
    result = SubprocessRunner().run([sys.executable, "-c", "print('hi'); raise SystemExit(3)"])
    assert result.returncode == 3
    assert result.stdout.strip() == "hi"
    assert not result.ok
    assert result.duration_s >= 0


def test_subprocess_runner_check_raises() -> None:
    with pytest.raises(CommandError, match="status 2"):
        SubprocessRunner().run([sys.executable, "-c", "raise SystemExit(2)"], check=True)


def test_subprocess_runner_missing_binary() -> None:
    result = SubprocessRunner().run(["/definitely/not/here"])
    assert result.returncode == 127
    assert "not found" in result.stderr


def test_subprocess_runner_timeout() -> None:
    result = SubprocessRunner().run(
        [sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2
    )
    assert result.returncode == 124
    assert "timed out" in result.stderr


def test_fake_runner_replays_and_records() -> None:
    runner = FakeRunner().on(prefix("tc", "qdisc", "show"), ok("qdisc noqueue 0: dev wlan0\n"))
    runner.on(contains("ping"), fail(1, "unreachable"))
    shown = runner.run(["tc", "qdisc", "show", "dev", "wlan0"])
    pinged = runner.run(["ping", "-c", "1", "10.0.0.5"])
    assert shown.ok and "noqueue" in shown.stdout
    assert pinged.returncode == 1
    assert runner.commands() == ["tc qdisc show dev wlan0", "ping -c 1 10.0.0.5"]


def test_fake_runner_strict_rejects_unexpected() -> None:
    with pytest.raises(AssertionError, match="unexpected command"):
        FakeRunner().run(["rm", "-rf", "/"])


def test_fake_runner_lenient_succeeds() -> None:
    assert FakeRunner(strict=False).run(["anything"]).ok


def test_fake_runner_check() -> None:
    runner = FakeRunner().on(prefix("false"), CommandResult(("false",), 1))
    with pytest.raises(CommandError):
        runner.run(["false"], check=True)


def test_dry_run_prints_and_succeeds() -> None:
    stream = io.StringIO()
    runner = DryRunRunner(stream=stream)
    result = runner.run(["tc", "qdisc", "add", "dev", "wlan0", "root", "netem", "delay", "10ms"])
    assert result.ok
    assert stream.getvalue() == "[dry-run] tc qdisc add dev wlan0 root netem delay 10ms\n"

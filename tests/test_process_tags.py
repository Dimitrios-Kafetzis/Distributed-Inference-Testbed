from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from src.common.process_tags import (
    TaggedProcess,
    is_running,
    make_tag,
    parse_pids,
    stop_all_command,
)


def sh(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=True)


def test_make_tag_and_validation() -> None:
    assert make_tag("20260101-120000-abc123", "monitor") == "testbed-20260101-120000-abc123-monitor"
    with pytest.raises(ValueError):
        make_tag("bad id", "monitor")


def test_commands_contain_tag_pid_and_log() -> None:
    proc = TaggedProcess("run1", "monitor", "/opt/testbed/run/run1")
    start = proc.start_command(
        "python3 -m src.testbed.resource_monitor --interval 1", {"TESTBED_NODE_ID": "d1"}
    )
    assert "setsid nohup sh -c" in start
    assert f"TESTBED_TAG={proc.tag}" in start
    assert "/opt/testbed/run/run1/monitor.pid" in start
    assert "TESTBED_NODE_ID=d1" in start
    stop = proc.stop_command()
    safe = "[t]" + proc.tag[1:]
    assert 'kill -s TERM -- "-$pgid"' in stop and f"pkill -KILL -f {safe}" in stop
    assert "echo running; else echo stopped" in proc.check_command()
    assert "[t]estbed-run1-" in stop_all_command("run1")


def test_parse_pids_and_is_running() -> None:
    assert parse_pids("123\n456\n\nnot a pid\n") == [123, 456]
    assert is_running("running\n")
    assert not is_running("stopped\n")
    assert not is_running("")


def test_start_check_and_stop_a_real_process_group(tmp_path: Path) -> None:
    """The generated shell detaches a process group and stops it, children included."""
    proc = TaggedProcess("t1", "sleeper", str(tmp_path))
    # A parent shell with a child: stopping must take both down.
    start = proc.start_command("sh -c 'sleep 30 & wait'")
    pgid = int(sh(start).stdout.strip())
    assert (tmp_path / "sleeper.pid").read_text().strip() == str(pgid)
    time.sleep(0.2)
    assert is_running(sh(proc.check_command()).stdout)
    children = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True)
    assert len(parse_pids(children.stdout)) >= 2  # the wrapper shell and sleep

    sh(proc.stop_command(grace_s=0.2))
    time.sleep(0.2)
    assert not is_running(sh(proc.check_command()).stdout)
    survivors = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True)
    assert parse_pids(survivors.stdout) == []
    assert not (tmp_path / "sleeper.pid").exists()

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from src.common.runner import CommandResult, FakeRunner, ok, prefix
from src.network import latency_measurement as lm
from src.network.records import LatencyRecord, SystemSnapshot

NORMAL = """PING 192.0.2.1 (192.0.2.1) 56(84) bytes of data.
64 bytes from 192.0.2.1: icmp_seq=1 ttl=64 time=24.26 ms
64 bytes from 192.0.2.1: icmp_seq=2 ttl=64 time=105 ms
64 bytes from 192.0.2.1: icmp_seq=3 ttl=64 time=5.1 ms

--- 192.0.2.1 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2003ms
rtt min/avg/max/mdev = 5.100/44.820/105.000/43.654 ms
"""
LOSSY = """PING 192.0.2.1 (192.0.2.1) 56(84) bytes of data.
64 bytes from 192.0.2.1: icmp_seq=1 ttl=64 time=9.8 ms
64 bytes from 192.0.2.1: icmp_seq=3 ttl=64 time=10.2 ms

--- 192.0.2.1 ping statistics ---
4 packets transmitted, 2 received, 50% packet loss, time 3050ms
"""
DEAD = """PING 192.0.2.9 (192.0.2.9) 56(84) bytes of data.

--- 192.0.2.9 ping statistics ---
3 packets transmitted, 0 received, 100% packet loss, time 2040ms
"""


def test_parse_normal_output_including_integer_milliseconds() -> None:
    sent, received, samples = lm.parse_ping_output(NORMAL, 3)
    assert (sent, received) == (3, 3)
    assert [s.rtt_ms for s in samples] == [24.26, 105.0, 5.1]


def test_parse_lossy_and_dead() -> None:
    sent, received, samples = lm.parse_ping_output(LOSSY, 4)
    assert (sent, received) == (4, 2)
    assert [s.rtt_ms for s in samples] == [9.8, None, 10.2, None]
    sent, received, samples = lm.parse_ping_output(DEAD, 3)
    assert (sent, received) == (3, 0) and all(s.rtt_ms is None for s in samples)


def test_ping_builds_argv_and_handles_exit_codes() -> None:
    runner = FakeRunner().on(prefix("ping"), ok(NORMAL))
    sent, received, _ = lm.ping(runner, "192.0.2.1", 3, interface="wlan0", interval_s=0.5)
    assert runner.commands() == ["ping -n -c 3 -i 0.5 -W 2 -I wlan0 192.0.2.1"]
    assert (sent, received) == (3, 3)
    lossy = FakeRunner().on(prefix("ping"), CommandResult(("ping",), 1, LOSSY))
    assert lm.ping(lossy, "192.0.2.1", 4)[1] == 2
    unknown_host = FakeRunner().on(
        prefix("ping"), CommandResult(("ping",), 2, "", "Name or service not known")
    )
    sent, received, samples = lm.ping(unknown_host, "nowhere", 2)
    assert (sent, received) == (2, 0) and len(samples) == 2


def test_measure_writes_a_record(tmp_path: Path) -> None:
    runner = FakeRunner().on(
        prefix("ping", "-n", "-c", "2", "-i", "0.2", "-W", "2", "192.0.2.1"), ok(NORMAL)
    )
    runner.on(prefix("ping"), ok(LOSSY))
    snapshot = SystemSnapshot(3800.0, 3000.0, 78.95, 4.0)
    record = lm.measure(
        {"edge_server": "192.0.2.1", "device2": "192.0.2.12"},
        run_id="r",
        node_id="device1",
        count=2,
        runner=runner,
        hostname="d1",
        snapshot=snapshot,
    )
    record.save(tmp_path / "lat.json")
    loaded = LatencyRecord.load(tmp_path / "lat.json")
    assert loaded.ping_to("edge_server") is not None and loaded.ping_to("device2") is not None
    assert loaded.system.cpu_usage_percent == 4.0 and loaded.metadata["count"] == 2


def test_cli_with_fake_ping_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ping = fake_bin / "ping"
    fake_ping.write_text("#!/bin/sh\ncat <<'EOF'\n" + NORMAL + "EOF\n")
    fake_ping.chmod(fake_ping.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(lm, "system_snapshot", lambda: SystemSnapshot(1.0, 1.0, 100.0, 0.0))
    out = tmp_path / "lat.json"
    lm.main(
        [
            "--target",
            "edge_server=192.0.2.1",
            "--count",
            "3",
            "--output",
            str(out),
            "--run-id",
            "r",
            "--node-id",
            "d",
        ]
    )
    data = json.loads(out.read_text())
    assert data["pings"][0]["received"] == 3
    with pytest.raises(SystemExit):
        lm.main(
            ["--target", "edge_server", "--output", str(out), "--run-id", "r", "--node-id", "d"]
        )

from __future__ import annotations

import json

import pytest

from src.common.runner import FakeRunner, ok, prefix
from src.network import throughput_measurement as tp
from src.testbed.background_traffic import BackgroundTraffic

TCP_JSON = {
    "start": {"test_start": {"protocol": "TCP", "duration": 10}},
    "end": {
        "sum_sent": {"bits_per_second": 95_400_000.0, "retransmits": 12},
        "sum_received": {"bits_per_second": 94_900_000.0},
    },
}
UDP_JSON = {
    "start": {"test_start": {"protocol": "UDP", "duration": 5}},
    "end": {"sum": {"bits_per_second": 19_600_000.0, "jitter_ms": 1.25, "lost_percent": 0.4}},
}


def test_parse_tcp_and_udp() -> None:
    tcp = tp.parse_iperf3_json(json.dumps(TCP_JSON), "192.0.2.1")
    assert tcp.protocol == "TCP" and tcp.sent_mbps == pytest.approx(95.4)
    assert (
        tcp.received_mbps == pytest.approx(94.9) and tcp.retransmits == 12 and tcp.duration_s == 10
    )
    udp = tp.parse_iperf3_json(json.dumps(UDP_JSON), "192.0.2.1")
    assert udp.protocol == "UDP" and udp.received_mbps == pytest.approx(19.6)
    assert udp.jitter_ms == 1.25 and udp.lost_percent == 0.4
    with pytest.raises(RuntimeError, match="unable to connect"):
        tp.parse_iperf3_json(json.dumps({"error": "unable to connect to server"}), "x")


def test_command_builders() -> None:
    assert tp.server_command() == ["iperf3", "-s", "-p", "5201", "-J", "-1"]
    assert tp.client_command("192.0.2.1", 10, parallel=2, reverse=True, bind="192.0.2.11") == [
        "iperf3",
        "-c",
        "192.0.2.1",
        "-p",
        "5201",
        "-t",
        "10",
        "-J",
        "-P",
        "2",
        "-R",
        "-B",
        "192.0.2.11",
    ]
    assert tp.client_command("192.0.2.1", 5, udp=True, bitrate="50M")[-3:] == ["-u", "-b", "50M"]


def test_measure_with_fake_runner() -> None:
    runner = FakeRunner().on(prefix("iperf3", "-c"), ok(json.dumps(TCP_JSON)))
    result = tp.measure("192.0.2.1", 10, runner=runner, parallel=1)
    assert result.sent_mbps == pytest.approx(95.4)
    assert runner.commands() == ["iperf3 -c 192.0.2.1 -p 5201 -t 10 -J"]


def test_background_traffic_commands() -> None:
    traffic = BackgroundTraffic("192.0.2.1", rate_mbps=20, duration_s=60, bind="192.0.2.11")
    assert traffic.client_argv() == [
        "iperf3",
        "-c",
        "192.0.2.1",
        "-p",
        "5202",
        "-t",
        "60",
        "-J",
        "-u",
        "-b",
        "20M",
        "-B",
        "192.0.2.11",
    ]
    assert traffic.server_argv() == ["iperf3", "-s", "-p", "5202", "-J"]
    with pytest.raises(ValueError):
        BackgroundTraffic("x", rate_mbps=0, duration_s=1)

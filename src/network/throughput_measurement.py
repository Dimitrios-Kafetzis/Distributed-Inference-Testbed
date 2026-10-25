"""Application-layer throughput between a device and the edge server (iperf3).

The wireless link's achievable throughput parameterises the transmission-delay
component of the delay model (data size over rate). An ``iperf3`` server is
started on the target only for the duration of the test; the client runs on
the device and reports JSON, which is parsed into a :class:`ThroughputResult`.
Concurrent tests (several devices at once) share the link and show the
per-device rate under contention.

Usage (device side)::

    python -m src.network.throughput_measurement --server 192.0.2.1 --duration 10 --output tp.json
        [--parallel 1] [--reverse] [--udp --bitrate 50M] [--port 5201]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.common.runner import CommandRunner, SubprocessRunner

DEFAULT_PORT = 5201


@dataclass(frozen=True)
class ThroughputResult:
    server: str
    duration_s: float
    protocol: str
    sent_mbps: float
    received_mbps: float
    retransmits: int | None
    jitter_ms: float | None
    lost_percent: float | None
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def server_command(port: int = DEFAULT_PORT, one_off: bool = True) -> list[str]:
    """``iperf3 -s``; ``-1`` makes the server exit after one client, so nothing lingers."""
    argv = ["iperf3", "-s", "-p", str(port), "-J"]
    if one_off:
        argv.append("-1")
    return argv


def client_command(
    server: str,
    duration_s: float,
    *,
    port: int = DEFAULT_PORT,
    parallel: int = 1,
    reverse: bool = False,
    udp: bool = False,
    bitrate: str | None = None,
    bind: str | None = None,
) -> list[str]:
    argv = ["iperf3", "-c", server, "-p", str(port), "-t", f"{duration_s:g}", "-J"]
    if parallel > 1:
        argv += ["-P", str(parallel)]
    if reverse:
        argv.append("-R")
    if udp:
        argv.append("-u")
    if bitrate:
        argv += ["-b", bitrate]
    if bind:
        argv += ["-B", bind]
    return argv


def parse_iperf3_json(text: str, server: str) -> ThroughputResult:
    data = json.loads(text)
    if "error" in data:
        raise RuntimeError(f"iperf3: {data['error']}")
    end = data["end"]
    protocol = data.get("start", {}).get("test_start", {}).get("protocol", "TCP")
    duration = float(data.get("start", {}).get("test_start", {}).get("duration", 0))
    if protocol.upper() == "UDP":
        summary = end.get("sum", {})
        bps = float(summary.get("bits_per_second", 0.0))
        return ThroughputResult(
            server,
            duration,
            "UDP",
            bps / 1e6,
            bps / 1e6,
            None,
            float(summary.get("jitter_ms", 0.0)),
            float(summary.get("lost_percent", 0.0)),
            data,
        )
    sent = end.get("sum_sent", {})
    received = end.get("sum_received", {})
    return ThroughputResult(
        server,
        duration,
        "TCP",
        float(sent.get("bits_per_second", 0.0)) / 1e6,
        float(received.get("bits_per_second", 0.0)) / 1e6,
        int(sent["retransmits"]) if "retransmits" in sent else None,
        None,
        None,
        data,
    )


def measure(
    server: str,
    duration_s: float,
    *,
    runner: CommandRunner | None = None,
    **options: Any,
) -> ThroughputResult:
    runner = runner or SubprocessRunner()
    argv = client_command(server, duration_s, **options)
    result = runner.run(argv, timeout=duration_s + 30, check=True)
    return parse_iperf3_json(result.stdout, server)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure throughput to an iperf3 server.")
    parser.add_argument("--server", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--reverse", action="store_true", help="server sends, device receives")
    parser.add_argument("--udp", action="store_true")
    parser.add_argument("--bitrate", help="target bitrate for UDP, e.g. 50M")
    parser.add_argument("--bind", help="local address of the data-plane interface")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", help="process tag (ignored; present for pkill -f)")
    args = parser.parse_args(argv)
    result = measure(
        args.server,
        args.duration,
        port=args.port,
        parallel=args.parallel,
        reverse=args.reverse,
        udp=args.udp,
        bitrate=args.bitrate,
        bind=args.bind,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(
        f"{result.protocol} to {result.server}: sent {result.sent_mbps:.1f} Mbit/s, "
        f"received {result.received_mbps:.1f} Mbit/s -> {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

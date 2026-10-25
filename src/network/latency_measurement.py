"""Network latency and system metrics, measured on each end device.

Sends ping probes to the edge server (and to any other registered node),
extracts the round-trip time of every reply with a regular expression on the
``time=`` field, takes a psutil snapshot of the device's total and available
memory (MB and percent) and a one-second average of its CPU utilisation, and
writes both to a JSON file on the device (a latency record,
``src/network/records.py``). The edge server aggregates the collected files
with ``node_statistics.py``.

Usage::

    python -m src.network.latency_measurement --target edge_server=192.0.2.1
        [--target device2=192.0.2.12 ...] --count 5 --output lat.json --run-id R --node-id device1
        [--interface wlan0] [--interval 0.2] [--timeout 2]
"""

from __future__ import annotations

import argparse
import re
import socket
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import psutil

from src.common.runner import CommandRunner, SubprocessRunner
from src.network.records import LatencyRecord, PingResult, PingSample, SystemSnapshot

# "64 bytes from 192.0.2.1: icmp_seq=1 ttl=64 time=24.26 ms" or "time=105 ms"
REPLY_RE = re.compile(r"icmp_seq=(\d+).*?time=(\d+(?:\.\d+)?)\s*ms")
SUMMARY_RE = re.compile(r"(\d+) packets transmitted, (\d+) (?:packets )?received")


def parse_ping_output(output: str, count: int) -> tuple[int, int, list[PingSample]]:
    """(sent, received, samples); missing sequence numbers are recorded as lost."""
    replies: dict[int, float] = {}
    for match in REPLY_RE.finditer(output):
        replies[int(match.group(1))] = float(match.group(2))
    summary = SUMMARY_RE.search(output)
    if summary:
        sent, received = int(summary.group(1)), int(summary.group(2))
    else:
        sent, received = count, len(replies)
    sent = max(sent, count)
    samples = [PingSample(seq, replies.get(seq)) for seq in range(1, sent + 1)]
    return sent, min(received, len(replies)) if replies else received, samples


def ping(
    runner: CommandRunner,
    target_ip: str,
    count: int,
    *,
    interval_s: float = 0.2,
    timeout_s: float = 2.0,
    interface: str | None = None,
) -> tuple[int, int, list[PingSample]]:
    argv = ["ping", "-n", "-c", str(count), "-i", f"{interval_s:g}", "-W", f"{timeout_s:g}"]
    if interface:
        argv += ["-I", interface]
    argv.append(target_ip)
    # ping exits 1 when some packets are lost; the output is still parsed.
    result = runner.run(argv, timeout=count * (interval_s + timeout_s) + 5)
    if result.returncode not in (0, 1):
        return count, 0, [PingSample(seq, None) for seq in range(1, count + 1)]
    return parse_ping_output(result.stdout, count)


def system_snapshot(
    cpu_window_s: float = 1.0, cpu_percent: Callable[[float], float] | None = None
) -> SystemSnapshot:
    vm = psutil.virtual_memory()
    measure = cpu_percent or (lambda window: float(psutil.cpu_percent(interval=window)))
    return SystemSnapshot(
        total_memory_mb=round(vm.total / 2**20, 2),
        available_memory_mb=round(vm.available / 2**20, 2),
        available_memory_percent=round(100.0 * vm.available / vm.total, 2),
        cpu_usage_percent=round(measure(cpu_window_s), 2),
    )


def measure(
    targets: dict[str, str],
    *,
    run_id: str,
    node_id: str,
    count: int = 5,
    interval_s: float = 0.2,
    timeout_s: float = 2.0,
    interface: str | None = None,
    runner: CommandRunner | None = None,
    hostname: str | None = None,
    snapshot: SystemSnapshot | None = None,
) -> LatencyRecord:
    runner = runner or SubprocessRunner()
    pings = []
    for target_id, ip in targets.items():
        sent, received, samples = ping(
            runner, ip, count, interval_s=interval_s, timeout_s=timeout_s, interface=interface
        )
        pings.append(PingResult(target_id, ip, sent, received, samples))
    return LatencyRecord(
        run_id=run_id,
        node_id=node_id,
        hostname=hostname or socket.gethostname(),
        measured_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        pings=pings,
        system=snapshot or system_snapshot(),
        metadata={"count": count, "interval_s": interval_s, "interface": interface},
    )


def parse_target(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("targets are given as id=ip")
    target_id, ip = value.split("=", 1)
    return target_id, ip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure ping latency and system metrics.")
    parser.add_argument(
        "--target", type=parse_target, action="append", required=True, metavar="ID=IP"
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--interface", default=None, help="data-plane interface to ping from")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--tag", help="process tag (ignored; present for pkill -f)")
    args = parser.parse_args(argv)
    record = measure(
        dict(args.target),
        run_id=args.run_id,
        node_id=args.node_id,
        count=args.count,
        interval_s=args.interval,
        timeout_s=args.timeout,
        interface=args.interface,
    )
    record.save(args.output)
    for result in record.pings:
        rtts = result.rtts
        mean = sum(rtts) / len(rtts) if rtts else float("nan")
        print(
            f"{result.target_id} ({result.target_ip}): {result.received}/{result.sent} replies, "
            f"mean {mean:.2f} ms",
            file=sys.stderr,
        )
    print(f"written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

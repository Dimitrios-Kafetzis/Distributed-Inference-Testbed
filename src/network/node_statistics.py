"""Node statistics: per-node averages of the latency and system records.

Reads the latency records that ``latency_measurement.py`` wrote on each device
(collected into the run's ``latency/`` directory), averages them per node and
writes ``final_latencies_averages.json``, which parameterises the
transmission-delay component of the delay model and gives the
partition-feasibility calculator the system-state snapshot (available memory
and baseline CPU usage). It also updates the statistics stored in
``nodes.json``.

Usage::

    python -m src.network.node_statistics <latency_dir> <output_file>
        [--nodes nodes.json] [--run-id ID]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.benchmarking.stats import summarise
from src.common.nodes import NodeStatistics, Topology, load_topology, save_topology
from src.common.runs import write_json_atomic
from src.network.records import LatencyRecord

EDGE_SERVER_TARGET = "edge_server"  # the ping target id for the edge server in every record


def load_records(directory: str | Path) -> list[LatencyRecord]:
    records = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name == "final_latencies_averages.json":
            continue
        records.append(LatencyRecord.load(path))
    return records


def compute_averages(records: list[LatencyRecord]) -> dict[str, dict[str, Any]]:
    """Per node: round-trip time to the edge server, loss, memory and CPU averages,
    and the per-target latency matrix (for device-to-device measurements)."""
    by_node: dict[str, list[LatencyRecord]] = defaultdict(list)
    for record in records:
        by_node[record.node_id].append(record)

    averages: dict[str, dict[str, Any]] = {}
    for node_id, node_records in sorted(by_node.items()):
        rtts: list[float] = []
        sent = received = 0
        per_target: dict[str, list[float]] = defaultdict(list)
        per_target_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for record in node_records:
            for ping in record.pings:
                per_target[ping.target_id].extend(ping.rtts)
                per_target_counts[ping.target_id][0] += ping.sent
                per_target_counts[ping.target_id][1] += ping.received
                if ping.target_id == EDGE_SERVER_TARGET:
                    rtts.extend(ping.rtts)
                    sent += ping.sent
                    received += ping.received
        systems = [r.system for r in node_records]
        rtt_summary = summarise(rtts) if rtts else None
        averages[node_id] = {
            "hostname": node_records[-1].hostname,
            "samples": len(node_records),
            "measured_at": max(r.measured_at for r in node_records),
            "average_latency_ms": rtt_summary.mean if rtt_summary else None,
            "latency_std_ms": rtt_summary.std if rtt_summary else None,
            "latency_p95_ms": rtt_summary.p95 if rtt_summary else None,
            "loss_percent": 100.0 * (sent - received) / sent if sent else None,
            "total_memory_mb": summarise([s.total_memory_mb for s in systems]).mean,
            "available_memory_mb": summarise([s.available_memory_mb for s in systems]).mean,
            "available_memory_percent": summarise(
                [s.available_memory_percent for s in systems]
            ).mean,
            "cpu_usage_percent": summarise([s.cpu_usage_percent for s in systems]).mean,
            "targets": {
                target: {
                    "average_latency_ms": summarise(values).mean if values else None,
                    "latency_std_ms": summarise(values).std if values else None,
                    "loss_percent": (
                        100.0 * (counts[0] - counts[1]) / counts[0] if counts[0] else None
                    ),
                }
                for target, values in sorted(per_target.items())
                for counts in [per_target_counts[target]]
            },
        }
    return averages


def update_topology_statistics(
    topology: Topology, averages: dict[str, dict[str, Any]], run_id: str
) -> None:
    for node_id, values in averages.items():
        try:
            node = topology.get(node_id)
        except KeyError:
            continue
        node.statistics = NodeStatistics(
            average_latency_ms=values["average_latency_ms"],
            available_memory_mb=values["available_memory_mb"],
            cpu_usage_percent=values["cpu_usage_percent"],
            measured_at=values["measured_at"],
            run_id=run_id,
        )


def write_averages(averages: dict[str, dict[str, Any]], output: str | Path) -> Path:
    write_json_atomic(Path(output), averages)
    return Path(output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("latency_dir", help="directory with the collected latency records")
    parser.add_argument("output", help="final_latencies_averages.json to write")
    parser.add_argument("--nodes", help="nodes.json to update with the statistics")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    records = load_records(args.latency_dir)
    if not records:
        raise SystemExit(f"no latency records in {args.latency_dir}")
    averages = compute_averages(records)
    write_averages(averages, args.output)
    if args.nodes:
        topology = load_topology(args.nodes)
        run_id = args.run_id or records[0].run_id
        update_topology_statistics(topology, averages, run_id)
        save_topology(topology, args.nodes)
    stamp = datetime.now(tz=UTC).isoformat(timespec="seconds")
    print(f"{stamp}: averages for {len(averages)} node(s) written to {args.output}")


if __name__ == "__main__":
    main()

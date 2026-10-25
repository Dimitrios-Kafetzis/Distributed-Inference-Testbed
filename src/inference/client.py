"""Device-side client for partitioned inference, with per-phase timing.

Each end-to-end inference is decomposed into three non-overlapping phases:

* **local computation**: stages 1..s on the device;
* **transmission**: from the first byte of the cut tensor sent to the
  server's acknowledgement of the last byte (the acknowledgement's own
  return trip is included, which is documented);
* **remote computation**: stages s+1..S on the edge server, in the order set
  by its scheduling policy; the time the request waited in the server queue
  is recorded separately.

Usage::

    python -m src.inference.client --model alexnet --weights W.npz --server 192.0.2.1:5300 --s 4
        --repetitions 10 --output phases.json --run-id R --node-id device1
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmarking.benchmark_results import BenchmarkResults, Origin, PhaseRecord, new_record
from src.inference import protocol
from src.models.registry import Array, ModelAdapter


@dataclass
class PartitionedResult:
    phase: PhaseRecord
    output: Array
    transmitted_bytes: int
    server_header: dict[str, Any]

    @property
    def total_ns(self) -> int:
        return (
            self.phase.local_ns
            + self.phase.transmission_ns
            + self.phase.queue_wait_ns
            + self.phase.remote_ns
        )


def run_partitioned(
    adapter: ModelAdapter,
    x: Array,
    s: int,
    server: tuple[str, int],
    *,
    run_id: str,
    node_id: str,
    repetition: int = 0,
    timeout_s: float = 120.0,
) -> PartitionedResult:
    num_stages = len(adapter.stages)
    if not 0 <= s <= num_stages:
        raise ValueError(f"s must be in 0..{num_stages}")
    t0 = time.perf_counter_ns()
    cut = adapter.run_stages(x, 1, s) if s > 0 else x
    local_ns = time.perf_counter_ns() - t0
    if s == num_stages:
        return PartitionedResult(PhaseRecord(s, local_ns, 0, 0, 0), cut, 0, {})

    payload = adapter.serialize(cut)
    header = {
        "kind": "request",
        "run_id": run_id,
        "node_id": node_id,
        "model": adapter.name,
        "s": s,
        "repetition": repetition,
    }
    with socket.create_connection(server, timeout=timeout_s) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        header["send_wall_s"] = time.time()
        send_start = time.perf_counter_ns()
        sent = protocol.send_message(sock, header, payload)
        ack, _ = protocol.recv_message(sock)
        transmission_ns = time.perf_counter_ns() - send_start
        if ack.get("kind") != "ack":
            raise RuntimeError(f"server error: {ack.get('message')}")
        result, output_bytes = protocol.recv_message(sock)
    if result.get("kind") != "result":
        raise RuntimeError(f"server error: {result.get('message')}")
    output = adapter.deserialize(output_bytes)
    phase = PhaseRecord(
        s=s,
        local_ns=local_ns,
        transmission_ns=transmission_ns,
        remote_ns=int(result["remote_ns"]),
        queue_wait_ns=int(result["queue_wait_ns"]),
        server_arrival_wall_s=result.get("arrival_wall_s"),
        server_service_start_wall_s=result.get("service_start_wall_s"),
        server_service_end_wall_s=result.get("service_end_wall_s"),
    )
    return PartitionedResult(phase, output, sent, result)


def phases_record(
    adapter: ModelAdapter,
    results: list[PartitionedResult],
    *,
    run_id: str,
    node_id: str,
    origin: Origin,
    hostname: str | None = None,
) -> BenchmarkResults:
    record = new_record(
        "benchmark",
        run_id=run_id,
        hostname=hostname or socket.gethostname(),
        node_id=node_id,
        origin=origin,
        model=adapter.name,
        repetition=0,
        phases=[r.phase for r in results],
        metadata={
            "transmitted_bytes": {str(r.phase.s): r.transmitted_bytes for r in results},
            "kind": "phases",
        },
    )
    assert isinstance(record, BenchmarkResults)
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run partitioned inference against the edge-server service."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-args", default=None, help="JSON object of adapter parameters")
    parser.add_argument("--input-source", default=None, help="JSON object: the input source")
    parser.add_argument("--server", required=True, help="host:port")
    parser.add_argument(
        "--s", type=int, required=True, action="append", help="partition point (repeatable)"
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--origin-wall", type=float)
    parser.add_argument("--origin-perf", type=int)
    parser.add_argument("--tag", help="process tag (ignored; present for pkill -f)")
    args = parser.parse_args(argv)

    from src.common.seeds import seed_everything
    from src.models import registry
    from src.models.inference import apply_thread_count, load_adapter, load_sample

    seed_everything(args.seed)
    apply_thread_count()
    adapter = load_adapter(
        args.model, args.weights, args.seed, **registry.parse_model_args(args.model_args)
    )
    x = load_sample(adapter, args).array
    host, port = args.server.rsplit(":", 1)
    origin = (
        Origin(args.origin_wall, args.origin_perf)
        if args.origin_wall is not None and args.origin_perf is not None
        else Origin(time.time(), time.perf_counter_ns())
    )
    results = []
    for s in args.s:
        for repetition in range(args.repetitions):
            result = run_partitioned(
                adapter,
                x,
                s,
                (host, int(port)),
                run_id=args.run_id,
                node_id=args.node_id,
                repetition=repetition,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "s": s,
                        "repetition": repetition,
                        "local_ms": result.phase.local_ns / 1e6,
                        "transmission_ms": result.phase.transmission_ns / 1e6,
                        "queue_ms": result.phase.queue_wait_ns / 1e6,
                        "remote_ms": result.phase.remote_ns / 1e6,
                        "total_ms": result.total_ns / 1e6,
                        "argmax": int(np.asarray(result.output).reshape(-1).argmax()),
                    }
                ),
                file=sys.stderr,
            )
    phases_record(adapter, results, run_id=args.run_id, node_id=args.node_id, origin=origin).save(
        args.output
    )
    print(f"{len(results)} inference(s) written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""The edge-server inference service.

A process on the edge server, started for the duration of an experiment,
receives offloaded inference requests from the devices, orders them
according to the chosen scheduling policy (Round-Robin or Shortest-Job-First),
executes the requested layers (stages s+1..S) and returns the results. Every
request is acknowledged as soon as its last byte has arrived, so that the
device can measure the transmission delay from first byte sent to last byte
acknowledged; the result carries the server-side timestamps (arrival,
service start, service end) for the remote-computation and queue-wait
phases.

Usage::

    python -m src.inference.server --model alexnet --weights W.npz --policy SJF --port 5300
        [--host 0.0.0.0] [--max-requests N] [--nodes node1,node2,...]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.inference import protocol
from src.inference.scheduler import Request, Scheduler, make_scheduler
from src.models.registry import ModelAdapter

DEFAULT_PORT = 5300


@dataclass
class _Pending:
    request: Request
    payload: bytes
    conn: socket.socket
    header: dict[str, Any]
    arrival_wall_s: float
    event: threading.Event


class InferenceServer:
    def __init__(
        self,
        adapter: ModelAdapter,
        policy: str = "SJF",
        host: str = "127.0.0.1",
        port: int = 0,
        node_order: list[str] | None = None,
        perf_counter_ns: Callable[[], int] = time.perf_counter_ns,
        wall_time: Callable[[], float] = time.time,
        request_timeout_s: float = 120.0,
    ) -> None:
        self.adapter = adapter
        self.scheduler: Scheduler = make_scheduler(policy)
        self.policy = self.scheduler.policy
        self.node_order = list(node_order or [])
        self.perf_counter_ns = perf_counter_ns
        self.wall_time = wall_time
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(64)
        self._sock.settimeout(0.2)
        self.host, self.port = self._sock.getsockname()[:2]
        self.request_timeout_s = request_timeout_s
        self._lock = threading.Lock()
        self._work = threading.Condition(self._lock)
        self._pending: dict[int, _Pending] = {}
        self._next_id = 1
        self._stop = threading.Event()
        # Cleared by tests (or a coordinated start) to hold the worker until requests are queued.
        self.release = threading.Event()
        self.release.set()
        self._threads: list[threading.Thread] = []
        self.served = 0
        self.max_requests: int | None = None

    # --- lifecycle ---------------------------------------------------------------------------

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._accept_loop, name="accept", daemon=True),
            threading.Thread(target=self._worker_loop, name="worker", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        with self._work:
            self._work.notify_all()
        for t in self._threads:
            t.join(timeout=5)
        self._sock.close()

    def pending_count(self) -> int:
        with self._work:
            return len(self._pending)

    def serve_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                if self.max_requests is not None and self.served >= self.max_requests:
                    break
                time.sleep(0.1)
        finally:
            self.stop()

    # --- connections ---------------------------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            # Accepted sockets inherit the listener's short timeout; give requests their own.
            conn.settimeout(self.request_timeout_s)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _node_index(self, node_id: str) -> int:
        return (
            self.node_order.index(node_id) if node_id in self.node_order else len(self.node_order)
        )

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                header, payload = protocol.recv_message(conn)
                arrival_ns = self.perf_counter_ns()
                arrival_wall = self.wall_time()
                if header.get("kind") != "request":
                    protocol.send_message(conn, {"kind": "error", "message": "expected a request"})
                    return
                s = int(header["s"])
                num_stages = len(self.adapter.stages)
                if not 0 <= s <= num_stages:
                    protocol.send_message(
                        conn, {"kind": "error", "message": f"s outside 0..{num_stages}"}
                    )
                    return
                if header.get("model") not in (None, self.adapter.name):
                    protocol.send_message(
                        conn, {"kind": "error", "message": f"server runs {self.adapter.name}"}
                    )
                    return
                protocol.send_message(
                    conn,
                    {
                        "kind": "ack",
                        "arrival_wall_s": arrival_wall,
                        "arrival_perf_ns": arrival_ns,
                        "bytes": len(payload),
                    },
                )
                remaining = list(range(s + 1, num_stages + 1))
                offloaded = sum(st.flops for st in self.adapter.stages[s:])
                with self._work:
                    request = Request(
                        request_id=self._next_id,
                        node_id=str(header.get("node_id", "?")),
                        node_index=self._node_index(str(header.get("node_id", "?"))),
                        s=s,
                        stages_remaining=remaining,
                        offloaded_flops=offloaded,
                        arrival_ns=arrival_ns,
                    )
                    self._next_id += 1
                    done = threading.Event()
                    self._pending[request.request_id] = _Pending(
                        request, payload, conn, header, arrival_wall, done
                    )
                    self.scheduler.add(request)
                    self._work.notify()
                done.wait()
            except (protocol.ProtocolError, OSError, ValueError, KeyError) as exc:
                with contextlib.suppress(OSError):
                    protocol.send_message(
                        conn, {"kind": "error", "message": f"{type(exc).__name__}: {exc}"}
                    )

    # --- execution -------------------------------------------------------------------------------

    def _worker_loop(self) -> None:
        outputs: dict[int, Any] = {}
        while not self._stop.is_set():
            if not self.release.wait(timeout=0.2):
                continue
            with self._work:
                step = self.scheduler.next_step()
                while step is None and not self._stop.is_set():
                    self._work.wait(timeout=0.2)
                    step = self.scheduler.next_step()
                if step is None:
                    continue
                request, stage = step
                pending = self._pending[request.request_id]
            if request.service_start_ns is None:
                request.service_start_ns = self.perf_counter_ns()
                pending.header["service_start_wall_s"] = self.wall_time()
                outputs[request.request_id] = self.adapter.deserialize(pending.payload)
            outputs[request.request_id] = self.adapter.run_stages(
                outputs[request.request_id], stage, stage
            )
            with self._work:
                self.scheduler.step_done(request, stage)
            if request.done:
                request.service_end_ns = self.perf_counter_ns()
                self._finish(pending, outputs.pop(request.request_id))

    def _finish(self, pending: _Pending, output: Any) -> None:
        request = pending.request
        assert request.service_start_ns is not None and request.service_end_ns is not None
        header = {
            "kind": "result",
            "request_id": request.request_id,
            "s": request.s,
            "policy": self.policy,
            "arrival_wall_s": pending.arrival_wall_s,
            "service_start_wall_s": pending.header.get("service_start_wall_s"),
            "service_end_wall_s": self.wall_time(),
            "queue_wait_ns": request.service_start_ns - request.arrival_ns,
            "remote_ns": request.service_end_ns - request.service_start_ns,
            "offloaded_flops": request.offloaded_flops,
            "shape": list(output.shape),
        }
        with contextlib.suppress(OSError):
            protocol.send_message(pending.conn, header, self.adapter.serialize(output))
        with self._work:
            self._pending.pop(request.request_id, None)
            self.served += 1
        pending.event.set()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Edge-server inference service (RR or SJF).")
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", default="SJF", choices=["RR", "SJF", "rr", "sjf"])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--nodes", default="", help="comma-separated device order for SJF tie-breaks"
    )
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--tag", help="process tag (ignored; present for pkill -f)")
    args = parser.parse_args(argv)

    from src.models.inference import apply_thread_count, load_adapter

    apply_thread_count()
    adapter = load_adapter(args.model, args.weights, args.seed)
    adapter.run_stages(adapter.make_input(0), 1, len(adapter.stages))  # warm up
    server = InferenceServer(
        adapter, args.policy, args.host, args.port, [n for n in args.nodes.split(",") if n]
    )
    server.max_requests = args.max_requests
    print(
        json.dumps(
            {
                "listening": [server.host, server.port],
                "policy": server.policy,
                "model": adapter.name,
            }
        ),
        flush=True,
    )

    def handle_signal(signum: int, _frame: Any) -> None:
        server._stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    server.serve_forever()
    print(f"served {server.served} request(s)", file=sys.stderr)


if __name__ == "__main__":
    main()

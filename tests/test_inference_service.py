from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from src.benchmarking import benchmark_results as br
from src.inference import client, protocol, scheduler
from src.inference.server import InferenceServer
from src.models.fake import FakeStagesAdapter

# Stage times (ms): five stages, offloaded work shrinks with s.
SPECS = [
    ("s1", 20.0, 0.0, 256, 5_000_000),
    ("s2", 20.0, 0.0, 256, 4_000_000),
    ("s3", 20.0, 0.0, 256, 3_000_000),
    ("s4", 20.0, 0.0, 256, 2_000_000),
    ("s5", 20.0, 0.0, 16, 1_000_000),
]


def make_adapter(time_scale: float = 1.0) -> FakeStagesAdapter:
    adapter = FakeStagesAdapter(SPECS, time_scale=time_scale)
    adapter.build(0)
    return adapter


def wait_until(predicate: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met in time")


@pytest.fixture
def server_factory() -> Iterator[type[InferenceServer]]:
    servers: list[InferenceServer] = []

    class Factory(InferenceServer):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            servers.append(self)

    yield Factory
    for s in servers:
        s.stop()


def test_protocol_round_trip() -> None:
    a, b = socket.socketpair()
    payload = b"\x00" * 100_000
    threading.Thread(
        target=protocol.send_message, args=(a, {"kind": "request", "s": 3}, payload)
    ).start()
    header, received = protocol.recv_message(b)
    assert header == {"kind": "request", "s": 3} and received == payload
    a.close()
    with pytest.raises(protocol.ProtocolError):
        protocol.recv_message(b)
    b.close()


def test_sjf_serves_shortest_offloaded_work_first_with_tie_break() -> None:
    requests = [
        scheduler.Request(1, "d1", 0, 2, [3, 4, 5], 6_000_000, 0),
        scheduler.Request(2, "d2", 1, 4, [5], 1_000_000, 0),
        scheduler.Request(3, "d3", 2, 2, [3, 4, 5], 6_000_000, 0),
    ]
    completion = scheduler.simulate("SJF", requests, dict.fromkeys(range(1, 6), 10))
    # d2 first (10), then d1 (ties with d3, lower device index), then d3.
    assert completion == {2: 10, 1: 40, 3: 70}
    assert requests[1].service_start_ns == 0 and requests[0].service_start_ns == 10


def test_round_robin_one_stage_per_turn() -> None:
    requests = [
        scheduler.Request(1, "d1", 0, 2, [3, 4, 5], 6_000_000, 0),
        scheduler.Request(2, "d2", 1, 4, [5], 1_000_000, 0),
    ]
    completion = scheduler.simulate("RR", requests, dict.fromkeys(range(1, 6), 10))
    # Round 1: d1 stage 3, d2 stage 5 (done at 20); then d1 stages 4, 5 -> 40.
    assert completion == {2: 20, 1: 40}
    with pytest.raises(ValueError):
        scheduler.make_scheduler("FCFS")


def test_chapter4_sjf_closed_form_matches_simulation() -> None:
    """delta_remote_i = (F_i^R + sum of smaller offloaded workloads) / C^R."""
    flops_per_stage = {
        1: 211,
        2: 0,
        3: 448,
        4: 0,
        5: 150,
        6: 112,
        7: 75,
        8: 0,
        9: 38,
        10: 17,
        11: 4,
    }
    capacity = 20.0  # "GFLOP/s" -> stage time in ms = flops / capacity

    def request(rid: int, node_index: int, s: int) -> scheduler.Request:
        remaining = list(range(s + 1, 12))
        return scheduler.Request(
            rid, f"d{rid}", node_index, s, remaining, sum(flops_per_stage[i] for i in remaining), 0
        )

    reqs = [request(1, 0, 2), request(2, 1, 4), request(3, 2, 2)]
    stage_ns = {i: int(f / capacity * 1e6) for i, f in flops_per_stage.items()}
    completion = scheduler.simulate("SJF", reqs, stage_ns)
    assert completion[2] / 1e6 == pytest.approx(19.8, abs=0.1)
    assert completion[1] / 1e6 == pytest.approx(62.0, abs=0.1)
    assert completion[3] / 1e6 == pytest.approx(104.2, abs=0.1)


def test_server_and_client_phases_over_loopback(server_factory: type[InferenceServer]) -> None:
    adapter = make_adapter()
    server = server_factory(adapter, "SJF", "127.0.0.1", 0, ["d1", "d2"])
    server.start()
    x = adapter.make_input(1)
    expected = adapter.run_stages(x, 1, 5)
    result = client.run_partitioned(
        adapter, x, 2, (server.host, server.port), run_id="r", node_id="d1"
    )
    np.testing.assert_allclose(result.output, expected)
    phase = result.phase
    assert phase.local_ns >= 40_000_000  # two stages of 20 ms
    assert phase.remote_ns >= 60_000_000  # three stages of 20 ms
    assert phase.transmission_ns > 0 and phase.queue_wait_ns >= 0
    assert result.transmitted_bytes > 256 * 4
    assert phase.server_arrival_wall_s is not None and phase.server_service_end_wall_s is not None
    local_only = client.run_partitioned(
        adapter, x, 5, (server.host, server.port), run_id="r", node_id="d1"
    )
    assert local_only.phase.transmission_ns == 0 and local_only.phase.remote_ns == 0
    np.testing.assert_allclose(local_only.output, expected)
    raw = client.run_partitioned(
        adapter, x, 0, (server.host, server.port), run_id="r", node_id="d2"
    )
    np.testing.assert_allclose(raw.output, expected)
    assert server.served == 2


def run_concurrent(
    server: InferenceServer,
    x: object,
    jobs: list[tuple[str, int]],
    *,
    in_order: bool = False,
) -> dict[str, client.PartitionedResult]:
    """Start one zero-cost client per job, hold the server until every request is
    queued (so the policy, not the arrival order, decides), then release it.

    With ``in_order`` the requests arrive in the order of ``jobs`` (each client waits
    until the previous request is queued), which matters for round robin.
    """
    results: dict[str, client.PartitionedResult] = {}
    barrier = threading.Barrier(len(jobs))

    def run(node: str, s: int) -> None:
        local = make_adapter(time_scale=0.0)
        if in_order:
            position = [j[0] for j in jobs].index(node)
            wait_until(lambda: server.pending_count() >= position)
        else:
            barrier.wait()
        results[node] = client.run_partitioned(
            local,
            x,
            s,
            (server.host, server.port),
            run_id="r",
            node_id=node,  # type: ignore[arg-type]
        )

    threads = [threading.Thread(target=run, args=job, daemon=True) for job in jobs]
    for t in threads:
        t.start()
    try:
        wait_until(lambda: server.pending_count() == len(jobs))
    finally:
        server.release.set()
    for t in threads:
        t.join(timeout=30)
    assert len(results) == len(jobs)
    return results


def test_sjf_order_and_waiting_times_with_concurrent_clients(
    server_factory: type[InferenceServer],
) -> None:
    adapter = make_adapter()
    server = server_factory(adapter, "SJF", "127.0.0.1", 0, ["d1", "d2", "d3"])
    server.release.clear()
    server.start()
    results = run_concurrent(server, adapter.make_input(1), [("d1", 1), ("d2", 4), ("d3", 1)])
    # Shortest offloaded work (d2: one stage) is served first, then d1 (index 0) before d3.
    starts = {n: r.phase.server_service_start_wall_s for n, r in results.items()}
    assert starts["d2"] < starts["d1"] < starts["d3"]
    waits = {n: r.phase.queue_wait_ns for n, r in results.items()}
    # Relative to the first served request: d1 waits for d2 (20 ms), d3 for d2 and d1 (100 ms).
    assert 15_000_000 <= waits["d1"] - waits["d2"] <= 60_000_000
    assert 90_000_000 <= waits["d3"] - waits["d2"] <= 200_000_000
    assert results["d3"].phase.remote_ns >= 80_000_000  # four stages of 20 ms, uninterrupted


def test_round_robin_server_shares_stages(server_factory: type[InferenceServer]) -> None:
    adapter = make_adapter()
    server = server_factory(adapter, "RR", "127.0.0.1", 0, ["d1", "d2"])
    server.release.clear()
    server.start()
    results = run_concurrent(server, adapter.make_input(1), [("d1", 1), ("d2", 4)], in_order=True)
    # d1 (four stages) arrives first. Under RR the one-stage request of d2 runs in the second
    # turn and finishes within two rounds; d1 has that stage interleaved, so its remote time
    # exceeds its own 80 ms of work.
    short, long = results["d2"], results["d1"]
    assert short.phase.server_service_end_wall_s < long.phase.server_service_end_wall_s
    assert long.phase.remote_ns >= 95_000_000
    assert short.phase.remote_ns <= 60_000_000


def test_server_rejects_bad_requests(server_factory: type[InferenceServer]) -> None:
    adapter = make_adapter()
    server = server_factory(adapter, "SJF", "127.0.0.1", 0)
    server.start()
    with socket.create_connection((server.host, server.port)) as sock:
        protocol.send_message(sock, {"kind": "request", "s": 99, "node_id": "x"}, b"")
        header, _ = protocol.recv_message(sock)
    assert header["kind"] == "error" and "outside" in header["message"]
    with pytest.raises(ValueError):
        client.run_partitioned(
            adapter, adapter.make_input(0), 9, (server.host, server.port), run_id="r", node_id="d"
        )


def test_client_cli_writes_a_phases_record(
    server_factory: type[InferenceServer], tmp_path: Path
) -> None:
    adapter = make_adapter()
    weights = tmp_path / "fake.npz"
    adapter.save_weights(weights)
    server = server_factory(adapter, "SJF", "127.0.0.1", 0)
    server.start()
    out = tmp_path / "phases.json"
    client.main(
        [
            "--model",
            "fake",
            "--weights",
            str(weights),
            "--server",
            f"{server.host}:{server.port}",
            "--s",
            "1",
            "--s",
            "3",
            "--repetitions",
            "2",
            "--output",
            str(out),
            "--run-id",
            "r",
            "--node-id",
            "d1",
        ]
    )
    record = br.load_benchmark(out)
    assert [p.s for p in record.phases] == [1, 1, 3, 3]
    assert record.metadata["kind"] == "phases"

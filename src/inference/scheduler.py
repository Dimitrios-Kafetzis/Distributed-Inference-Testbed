"""Scheduling policies of the edge server: Round-Robin and Shortest-Job-First.

The edge server executes the offloaded workloads of all devices one at a
time. Under **Round-Robin (RR)** the processor cycles through the active
requests in a fixed order, executing one stage (the scheduling quantum) of
each per round. Under **Shortest-Job-First (SJF)**, a non-preemptive policy,
requests are served in non-decreasing order of their offloaded workload
F^R(s) (FLOPs of stages s+1..S); ties are broken by device index, as in
Chapter 4. A request that is being served runs to completion before the
next one starts.

The schedulers are pure (no threads, no sockets): the service asks for the
next (request, stage) step and reports its completion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

POLICIES = ("RR", "SJF")


@dataclass
class Request:
    request_id: int
    node_id: str
    node_index: int  # position of the device in the topology (SJF tie-break)
    s: int
    stages_remaining: list[int]  # stage indices still to run, ascending
    offloaded_flops: int
    arrival_ns: int
    service_start_ns: int | None = None
    service_end_ns: int | None = None
    steps_done: int = 0

    @property
    def done(self) -> bool:
        return not self.stages_remaining


class Scheduler(Protocol):
    policy: str

    def add(self, request: Request) -> None: ...

    def next_step(self) -> tuple[Request, int] | None:
        """The next (request, stage index) to execute, or None when idle."""
        ...

    def step_done(self, request: Request, stage: int) -> None: ...

    def pending(self) -> list[Request]: ...


@dataclass
class RoundRobinScheduler:
    policy: str = "RR"
    _queue: deque[Request] = field(default_factory=deque)

    def add(self, request: Request) -> None:
        self._queue.append(request)

    def next_step(self) -> tuple[Request, int] | None:
        while self._queue:
            request = self._queue[0]
            if request.done:
                self._queue.popleft()
                continue
            return request, request.stages_remaining[0]
        return None

    def step_done(self, request: Request, stage: int) -> None:
        assert self._queue and self._queue[0] is request
        request.stages_remaining.remove(stage)
        request.steps_done += 1
        self._queue.popleft()
        if not request.done:
            self._queue.append(request)  # back of the rotation after its quantum

    def pending(self) -> list[Request]:
        return list(self._queue)


@dataclass
class ShortestJobFirstScheduler:
    policy: str = "SJF"
    _waiting: list[Request] = field(default_factory=list)
    _current: Request | None = None

    def add(self, request: Request) -> None:
        self._waiting.append(request)

    def _pick(self) -> Request | None:
        if not self._waiting:
            return None
        self._waiting.sort(
            key=lambda r: (r.offloaded_flops, r.node_index, r.arrival_ns, r.request_id)
        )
        return self._waiting.pop(0)

    def next_step(self) -> tuple[Request, int] | None:
        if self._current is None or self._current.done:
            self._current = self._pick()
        if self._current is None:
            return None
        return self._current, self._current.stages_remaining[0]

    def step_done(self, request: Request, stage: int) -> None:
        assert request is self._current
        request.stages_remaining.remove(stage)
        request.steps_done += 1
        if request.done:
            self._current = None

    def pending(self) -> list[Request]:
        current = [self._current] if self._current is not None and not self._current.done else []
        return current + list(self._waiting)


def make_scheduler(policy: str) -> Scheduler:
    policy = policy.upper()
    if policy == "RR":
        return RoundRobinScheduler()
    if policy == "SJF":
        return ShortestJobFirstScheduler()
    raise ValueError(f"unknown scheduling policy {policy!r}; expected RR or SJF")


def simulate(policy: str, requests: list[Request], stage_time_ns: dict[int, int]) -> dict[int, int]:
    """Completion time (ns) of every request when all arrive at time 0 and each
    stage takes ``stage_time_ns[stage]``. Used by tests against the Chapter 4
    closed forms."""
    scheduler = make_scheduler(policy)
    for request in requests:
        scheduler.add(request)
    now = 0
    completion: dict[int, int] = {}
    while (step := scheduler.next_step()) is not None:
        request, stage = step
        if request.service_start_ns is None:
            request.service_start_ns = now
        now += stage_time_ns[stage]
        scheduler.step_done(request, stage)
        if request.done:
            request.service_end_ns = now
            completion[request.request_id] = now
    return completion

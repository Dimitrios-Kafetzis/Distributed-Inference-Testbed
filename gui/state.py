"""Application state: topology, probes, scenarios, runs.

* :class:`Prober`: a background thread that issues a three-ping reachability
  probe to every registered node concurrently, one round every two seconds
  (``probe_period_s``); a node is green only when all three pings succeed.
  Rounds never overlap: a node whose probe has not finished keeps its last
  colour.
* :class:`RunManager`: one experiment at a time, executed in a thread through
  the orchestration pipeline; the merged log is kept in a ring buffer and
  streamed to subscribers (the terminal panel). A scenario with a sweep runs
  its variants one after the other, each as its own run.
* :class:`TaskManager`: one maintenance action of a runtime at a time (install
  HeadSplit, build a profile, distribute a checkpoint), never while an
  experiment runs, with the same log streaming.
"""

from __future__ import annotations

import contextlib
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.nodes import Node, Topology, load_topology, save_topology
from src.common.runner import CommandRunner, SubprocessRunner
from src.config import Config
from src.orchestration import experiment, local_mode
from src.orchestration import scenario as scenario_module
from src.orchestration.runtimes import ActionSession, Runtime

PING_COUNT = 3
REPLY_RE = re.compile(r"(\d+) packets transmitted, (\d+) (?:packets )?received")


def status_from_replies(sent: int, received: int) -> str:
    """The marker colour: green if all three pings succeed, red otherwise."""
    return "green" if sent == PING_COUNT and received == PING_COUNT else "red"


def parse_ping_summary(output: str) -> tuple[int, int]:
    match = REPLY_RE.search(output)
    if not match:
        return PING_COUNT, 0
    return int(match.group(1)), int(match.group(2))


@dataclass
class NodeStatus:
    node_id: str
    status: str
    received: int
    checked_at: float


class Prober:
    def __init__(
        self,
        topology_provider: Callable[[], Topology],
        period_s: float = 2.0,
        runner: CommandRunner | None = None,
        ping_interval_s: float = 0.2,
        ping_timeout_s: float = 1.0,
    ) -> None:
        self.topology_provider = topology_provider
        self.period_s = period_s
        self.runner = runner or SubprocessRunner()
        self.ping_interval_s = ping_interval_s
        self.ping_timeout_s = ping_timeout_s
        self.statuses: dict[str, NodeStatus] = {}
        self.round = 0
        self._inflight: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._listeners: list[queue.Queue[dict[str, Any]]] = []

    def probe_node(self, node: Node) -> NodeStatus:
        argv = [
            "ping",
            "-n",
            "-c",
            str(PING_COUNT),
            "-i",
            f"{self.ping_interval_s:g}",
            "-W",
            f"{self.ping_timeout_s:g}",
            node.data_address,
        ]
        result = self.runner.run(
            argv, timeout=PING_COUNT * (self.ping_interval_s + self.ping_timeout_s) + 2
        )
        sent, received = (
            parse_ping_summary(result.stdout) if result.returncode in (0, 1) else (PING_COUNT, 0)
        )
        return NodeStatus(node.id, status_from_replies(sent, received), received, time.time())

    def probe_round(self) -> dict[str, NodeStatus]:
        """Probe every node concurrently and publish a snapshot within one period.

        A probe that has not answered by the end of the period (a node that is down takes
        about 3.6 s with the default ping timeouts) keeps its last colour; its result is
        taken up by a later snapshot, and no second probe of that node is started while
        the first is still running. This keeps one round every ``period_s`` (App. B).
        """
        nodes = self.topology_provider().nodes
        deadline = time.monotonic() + self.period_s
        results: dict[str, NodeStatus] = {}
        threads = []

        def work(node: Node) -> None:
            try:
                status = self.probe_node(node)
            finally:
                with self._lock:
                    self._inflight.discard(node.id)
            with self._lock:
                self.statuses[node.id] = status
                results[node.id] = status

        with self._lock:
            pending = [n for n in nodes if n.id not in self._inflight]
            self._inflight.update(n.id for n in pending)
        for node in pending:
            thread = threading.Thread(target=work, args=(node,), daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._lock:
            known = {n.id for n in nodes}
            for stale in [k for k in self.statuses if k not in known]:
                del self.statuses[stale]
            self.round += 1
            snapshot = self.snapshot()
        for listener in list(self._listeners):
            listener.put(snapshot)
        return dict(results)

    def snapshot(self) -> dict[str, Any]:
        return {"round": self.round, "nodes": {k: v.__dict__ for k, v in self.statuses.items()}}

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        listener: queue.Queue[dict[str, Any]] = queue.Queue()
        self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: queue.Queue[dict[str, Any]]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            # A failing round (for example a ping binary that is missing) must not stop the
            # prober: the next round reports the nodes as unreachable.
            with contextlib.suppress(Exception):
                self.probe_round()
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.period_s - elapsed))

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="prober", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


@dataclass
class RunRecord:
    run_id: str
    scenario_name: str
    status: str = "running"  # queued | running | completed | aborted
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    summary: dict[str, Any] | None = None
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=5000))
    sweep_origin: dict[str, Any] | None = None


class RunManager:
    def __init__(self, state: AppState) -> None:
        self.state = state
        self.current: RunRecord | None = None
        self.history: dict[str, RunRecord] = {}
        self._ctx: experiment.ExperimentContext | None = None
        self._thread: threading.Thread | None = None
        self._listeners: dict[str, list[queue.Queue[str]]] = {}
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return any(r.status in ("running", "queued") for r in self.history.values())

    def _publish(self, run_id: str, line: str) -> None:
        record = self.history.get(run_id)
        if record is not None:
            record.lines.append(line)
        for listener in list(self._listeners.get(run_id, [])):
            listener.put(line)

    def subscribe(self, run_id: str) -> queue.Queue[str]:
        listener: queue.Queue[str] = queue.Queue()
        self._listeners.setdefault(run_id, []).append(listener)
        return listener

    def unsubscribe(self, run_id: str, listener: queue.Queue[str]) -> None:
        listeners = self._listeners.get(run_id, [])
        if listener in listeners:
            listeners.remove(listener)

    def _build(
        self, scenario: dict[str, Any], version: dict[str, Any], run_id: str | None
    ) -> tuple[experiment.ExperimentContext, RunRecord]:
        ctx = experiment.build_context(
            scenario,
            config=self.state.config,
            topology=self.state.topology,
            local=self.state.local,
            results_dir=self.state.results_dir,
            run_id=run_id,
            allow_dirty=self.state.allow_dirty,
            nodes_path=self.state.nodes_path,
            scenario_version=version,
        )
        record = RunRecord(
            ctx.run_id, scenario["name"], status="queued", sweep_origin=scenario.get("sweep_origin")
        )
        this_id = ctx.run_id

        def forward(line: str) -> None:
            self._publish(this_id, line)

        ctx.executor.subscribe(forward)
        return ctx, record

    def start(
        self,
        scenario: dict[str, Any],
        version: dict[str, Any],
        *,
        pipeline: str | None = None,
        run_id: str | None = None,
    ) -> RunRecord:
        """Start the scenario; a scenario with a sweep queues one run per value and runs
        them one after the other. The returned record is the first run."""
        with self._lock:
            if self.busy:
                raise RuntimeError("an experiment is already running")
            if pipeline:
                scenario = dict(scenario, pipeline=pipeline)
            variants = scenario_module.expand_sweep(scenario)
            group = [
                self._build(variant, version, run_id if len(variants) == 1 else None)
                for variant in variants
            ]
            for _, record in group:
                self.history[record.run_id] = record
            first_ctx, first_record = group[0]
            first_record.status = "running"
            self.current = first_record
            self._ctx = first_ctx
            self._thread = threading.Thread(
                target=self._run_group, args=(group,), name=f"run-{first_ctx.run_id}", daemon=True
            )
            self._thread.start()
            return first_record

    def _run_group(self, group: list[tuple[experiment.ExperimentContext, RunRecord]]) -> None:
        for index, (ctx, record) in enumerate(group):
            if record.status == "aborted":  # stopped while queued
                continue
            record.status = "running"
            record.started_at = time.time()
            self.current = record
            self._ctx = ctx
            if record.sweep_origin:
                origin = record.sweep_origin
                self._publish(
                    ctx.run_id,
                    f"[nnpm][info] sweep {origin['name']}: run {index + 1}/{len(group)}"
                    f" ({origin['path']} = {origin['value']})",
                )
            try:
                record.summary = experiment.run_pipeline(ctx)
                record.status = "completed"
            except Exception as exc:
                record.status = "aborted"
                record.error = str(exc)
                record.summary = {"run_id": ctx.run_id, "status": "aborted", "error": str(exc)}
            finally:
                record.finished_at = time.time()
                self.state.reload_topology()
            if ctx.cancel.is_set():
                # Stop: the queued variants are abandoned, not run.
                for _, pending in group[index + 1 :]:
                    if pending.status == "queued":
                        pending.status = "aborted"
                        pending.error = "stopped by the user"
                        pending.finished_at = time.time()
            # The next run is current before this one is announced as done, so that a
            # page following the group picks it up without a gap.
            following = next((r for _, r in group[index + 1 :] if r.status == "queued"), None)
            if following is not None:
                following.status = "running"
                self.current = following
            self._publish(ctx.run_id, f"[nnpm][done] {record.status}")

    def stop(self) -> bool:
        if not self.busy or self._ctx is None:
            return False
        self._ctx.cancel.set()
        self._publish(
            self._ctx.run_id, "[nnpm][info] stop requested: the run ends before its next stage"
        )
        return True

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)


@dataclass
class TaskRecord:
    """A maintenance action of a runtime (install, profile, checkpoint), run once."""

    task_id: str
    runtime: str
    action: str
    params: dict[str, Any]
    status: str = "running"  # running | completed | failed | stopped
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=5000))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "runtime": self.runtime,
            "action": self.action,
            "params": self.params,
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class TaskManager:
    """One maintenance task at a time, never while an experiment runs; the task's log is
    streamed to subscribers like a run's log. Device commands go through the same
    executor and transport as experiments."""

    def __init__(self, state: AppState, runner: CommandRunner | None = None) -> None:
        self.state = state
        self.runner = runner or SubprocessRunner()
        self.current: TaskRecord | None = None
        self.history: dict[str, TaskRecord] = {}
        self._cancel: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._listeners: dict[str, list[queue.Queue[str]]] = {}
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.current is not None and self.current.status == "running"

    def _publish(self, task_id: str, line: str) -> None:
        record = self.history.get(task_id)
        if record is not None:
            record.lines.append(line)
        for listener in list(self._listeners.get(task_id, [])):
            listener.put(line)

    def subscribe(self, task_id: str) -> queue.Queue[str]:
        listener: queue.Queue[str] = queue.Queue()
        self._listeners.setdefault(task_id, []).append(listener)
        return listener

    def session(self, log: Callable[[str], None]) -> ActionSession:
        """An action session on the live transport (SSH, or local processes in local mode)."""
        from src.config import get_ssh_password
        from src.network.ssh_executor import Executor, make_transport

        config = self.state.config
        topology = self.state.topology
        transport = make_transport(
            config, topology, local=self.state.local, password=get_ssh_password(config)
        )
        executor = Executor(transport, subscribers=[log])
        return ActionSession(
            config,
            topology,
            executor,
            local=self.state.local,
            runner=self.runner,
            log=log,
            cancel=threading.Event(),
        )

    def start(self, runtime: Runtime, action: str, params: dict[str, Any]) -> TaskRecord:
        with self._lock:
            if self.busy:
                raise RuntimeError("a task is already running")
            if self.state.runs.busy:
                raise RuntimeError("an experiment is running; wait for it to finish")
            task_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{runtime.name}-{action}"
            if task_id in self.history:
                task_id += f"-{len(self.history)}"
            record = TaskRecord(task_id, runtime.name, action, dict(params))
            self.history[task_id] = record
            self.current = record

            def log(line: str) -> None:
                self._publish(task_id, line)

            session = self.session(log)
            self._cancel = session.cancel
            self._thread = threading.Thread(
                target=self._run,
                args=(runtime, action, params, session, record),
                name=f"task-{task_id}",
                daemon=True,
            )
            self._thread.start()
            return record

    def _run(
        self,
        runtime: Runtime,
        action: str,
        params: dict[str, Any],
        session: ActionSession,
        record: TaskRecord,
    ) -> None:
        self._publish(record.task_id, f"[task][info] {runtime.name}: {action} started")
        try:
            runtime.perform(action, params, session)
            record.status = "stopped" if session.cancel.is_set() else "completed"
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            self._publish(record.task_id, f"[task][error] {exc}")
        finally:
            record.finished_at = time.time()
            self._publish(record.task_id, f"[task][done] {record.status}")

    def stop(self, task_id: str) -> bool:
        if not self.busy or self.current is None or self.current.task_id != task_id:
            return False
        if self._cancel is not None:
            self._cancel.set()
        self._publish(task_id, "[task][info] stop requested: the task ends after its current step")
        return True

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)


class AppState:
    def __init__(
        self,
        config: Config,
        nodes_path: Path,
        *,
        local: bool = False,
        results_dir: Path | None = None,
        scenario_root: Path | None = None,
        allow_dirty: bool = False,
        probe_runner: CommandRunner | None = None,
        task_runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.nodes_path = nodes_path
        self.local = local
        self.allow_dirty = allow_dirty or local
        self.results_dir = results_dir or config.bs_results_path
        self.scenarios = scenario_module.ScenarioStore(
            scenario_root or config.bs_root_path / "scenarios"
        )
        self._topology_lock = threading.Lock()
        self.topology = load_topology(nodes_path)
        self.prober = Prober(lambda: self.topology, config.probe_period_s, probe_runner)
        self.runs = RunManager(self)
        self.tasks = TaskManager(self, task_runner)

    def reload_topology(self) -> Topology:
        with self._topology_lock:
            self.topology = load_topology(self.nodes_path)
            return self.topology

    def save_topology(self) -> None:
        with self._topology_lock:
            save_topology(self.topology, self.nodes_path)

    @classmethod
    def local_testbed(cls, root: Path, devices: int = 2) -> AppState:
        config, _ = local_mode.create_local_testbed(root, devices=devices)
        return cls(
            config,
            root / "nodes.json",
            local=True,
            results_dir=root / "results",
            scenario_root=root / "scenarios",
        )

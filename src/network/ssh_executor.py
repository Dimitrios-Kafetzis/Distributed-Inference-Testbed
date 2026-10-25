"""SSH dispatch layer: concurrent remote commands with a merged, streamed log.

Remote commands are dispatched to all target devices concurrently through a
thread pool. Each device's stdout and stderr are streamed back line by line,
tagged with the node and the stream, merged into a single log file on the
edge server and forwarded to subscribers (the NNPM terminal panel). Three
failure classes are distinguished: authentication failures (raised before the
command is dispatched), unreachable hosts (no response within the SSH
connection timeout) and non-zero exit codes (command-level failures). On any
failure the experiment is aborted: pending commands are cancelled, the
partial logs are preserved, and :class:`ExperimentAbortedError` is raised.

Usage::

    python -m src.network.ssh_executor "<command>" <node id or last octet> ... [--log FILE]
        [--no-abort] [--timeout S] [--nodes nodes.json] [--local]
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.common.nodes import Node, Topology, load_topology
from src.common.process_tags import TaggedProcess, is_running
from src.config import Config, get_ssh_password, load_config
from src.network.transport import (
    AuthenticationError,
    CommandOutcome,
    FakeTransport,
    LocalTransport,
    ParamikoTransport,
    Transport,
    TransportError,
    UnreachableHostError,
)

LogSubscriber = Callable[[str], None]  # receives one formatted log line


@dataclass
class Failure:
    node_id: str
    kind: str  # "authentication" | "unreachable" | "command" | "cancelled"
    message: str
    exit_status: int | None = None


class ExperimentAbortedError(RuntimeError):
    def __init__(self, failures: list[Failure], outcomes: dict[str, CommandOutcome]) -> None:
        self.failures = failures
        self.outcomes = outcomes
        summary = "; ".join(f"{f.node_id}: {f.kind} ({f.message})" for f in failures)
        super().__init__(f"experiment aborted: {summary}")


@dataclass
class Executor:
    transport: Transport
    log_path: Path | None = None
    subscribers: list[LogSubscriber] = field(default_factory=list)
    max_workers: int | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # --- logging ----------------------------------------------------------------------------------

    def log(self, node_id: str, stream: str, line: str) -> None:
        stamp = datetime.now(tz=UTC).strftime("%H:%M:%S.%f")[:-3]
        formatted = f"{stamp} [{node_id}][{stream}] {line}"
        with self._lock:
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(formatted + "\n")
            for subscriber in list(self.subscribers):
                subscriber(formatted)

    def subscribe(self, subscriber: LogSubscriber) -> None:
        self.subscribers.append(subscriber)

    def unsubscribe(self, subscriber: LogSubscriber) -> None:
        with self._lock:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)

    # --- dispatch ------------------------------------------------------------------------------

    def run_on_node(
        self,
        node: Node,
        command: str,
        *,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
        env: Mapping[str, str] | None = None,
        check: bool = True,
    ) -> CommandOutcome:
        self.log(node.id, "exec", command)
        try:
            outcome = self.transport.run(
                node,
                command,
                on_line=lambda s, line: self.log(node.id, s, line),
                timeout=timeout,
                cancel=cancel,
                env=env,
            )
        except TransportError as exc:
            self.log(node.id, "error", str(exc))
            raise
        self.log(
            node.id,
            "exit",
            f"status {outcome.exit_status}" + (" (cancelled)" if outcome.cancelled else ""),
        )
        if check and not outcome.ok:
            raise ExperimentAbortedError([classify_outcome(outcome)], {node.id: outcome})
        return outcome

    def run_on_nodes(
        self,
        nodes: Sequence[Node],
        command: str,
        *,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        abort_on_failure: bool = True,
        per_node_command: Callable[[Node], str] | None = None,
    ) -> dict[str, CommandOutcome]:
        """Run ``command`` (or ``per_node_command(node)``) on every node concurrently.

        With ``abort_on_failure`` the first failure cancels the other commands and
        raises :class:`ExperimentAbortedError` with every failure collected so far.
        Without it, all commands run to the end and the outcomes are returned,
        failures included (unreachable or unauthenticated nodes get status 255).
        """
        if not nodes:
            return {}
        cancel = threading.Event()
        outcomes: dict[str, CommandOutcome] = {}
        failures: list[Failure] = []
        lock = threading.Lock()

        def task(node: Node) -> None:
            cmd = per_node_command(node) if per_node_command else command
            if cancel.is_set():
                with lock:
                    outcomes[node.id] = CommandOutcome(node.id, cmd, 255, cancelled=True)
                    failures.append(
                        Failure(node.id, "cancelled", "not started: experiment aborted")
                    )
                return
            try:
                outcome = self.run_on_node(
                    node, cmd, timeout=timeout, cancel=cancel, env=env, check=False
                )
            except AuthenticationError as exc:
                outcome = CommandOutcome(node.id, cmd, 255, stderr=str(exc))
                failure: Failure | None = Failure(node.id, "authentication", str(exc))
            except UnreachableHostError as exc:
                outcome = CommandOutcome(node.id, cmd, 255, stderr=str(exc))
                failure = Failure(node.id, "unreachable", str(exc))
            else:
                failure = None if outcome.ok else classify_outcome(outcome)
            with lock:
                outcomes[node.id] = outcome
                if failure is not None:
                    failures.append(failure)
                    if abort_on_failure:
                        cancel.set()

        workers = self.max_workers or len(nodes)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: list[Future[None]] = [pool.submit(task, node) for node in nodes]
            for future in futures:
                future.result()
        if failures and abort_on_failure:
            real = [f for f in failures if f.kind != "cancelled"] or failures
            raise ExperimentAbortedError(real, outcomes)
        return outcomes

    # --- tagged processes ----------------------------------------------------------------------

    def start_tagged(
        self, node: Node, process: TaggedProcess, command: str, env: Mapping[str, str] | None = None
    ) -> int:
        """Start a detached, tagged process on the node; return its process-group id."""
        outcome = self.run_on_node(node, process.start_command(command, dict(env or {})))
        try:
            return int(outcome.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise ExperimentAbortedError(
                [Failure(node.id, "command", f"no PID returned for {process.component}")],
                {node.id: outcome},
            ) from exc

    def stop_tagged(self, node: Node, process: TaggedProcess, grace_s: float = 2.0) -> bool:
        """Stop the process; return True when nothing with the tag is left."""
        self.run_on_node(node, process.stop_command(grace_s), check=False)
        outcome = self.run_on_node(node, process.check_command(), check=False)
        return not is_running(outcome.stdout)

    def is_running(self, node: Node, process: TaggedProcess) -> bool:
        outcome = self.run_on_node(node, process.check_command(), check=False)
        return is_running(outcome.stdout)

    # --- files ---------------------------------------------------------------------------------

    def fetch(self, node: Node, remote_path: str, local_path: Path) -> Path:
        self.log(node.id, "fetch", f"{remote_path} -> {local_path}")
        return self.transport.fetch(node, remote_path, local_path)


def classify_outcome(outcome: CommandOutcome) -> Failure:
    if outcome.cancelled:
        return Failure(outcome.node_id, "cancelled", "cancelled", outcome.exit_status)
    detail = (
        outcome.stderr.strip().splitlines()[-1] if outcome.stderr.strip() else "no error output"
    )
    return Failure(
        outcome.node_id,
        "command",
        f"exit status {outcome.exit_status}: {detail}",
        outcome.exit_status,
    )


# --- construction helpers ----------------------------------------------------------------------


def make_transport(
    config: Config, topology: Topology, *, local: bool = False, password: str | None = None
) -> Transport:
    if local or all(n.transport == "local" for n in topology.end_devices()):
        return LocalTransport({n.id: Path(n.workdir) for n in topology.nodes if n.workdir})
    return ParamikoTransport(config, password=password)


def resolve_nodes(topology: Topology, config: Config, identifiers: Iterable[str]) -> list[Node]:
    """Node ids, or the last octet of a node's data-plane or management address."""
    nodes = []
    for ident in identifiers:
        try:
            nodes.append(topology.get(ident))
            continue
        except KeyError:
            pass
        candidates = [
            n
            for n in topology.nodes
            if n.ip == f"{config.base_ip_prefix}{ident}"
            or n.management_address == f"{config.management_prefix}{ident}"
            or n.ip.rsplit(".", 1)[-1] == ident
        ]
        if len(candidates) != 1:
            raise SystemExit(
                f"cannot resolve node {ident!r} (matches: {[c.id for c in candidates]})"
            )
        nodes.append(candidates[0])
    return nodes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a command on several nodes concurrently.")
    parser.add_argument("command")
    parser.add_argument("nodes", nargs="+", help="node ids (nodes.json) or last address octets")
    parser.add_argument("--nodes-file", default=None, help="nodes.json (default: repository root)")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--log", type=Path, default=None, help="merged log file (default: stdout only)"
    )
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument(
        "--no-abort", action="store_true", help="run everything even after a failure"
    )
    parser.add_argument(
        "--local", action="store_true", help="local mode (per-node working directories)"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    topology = load_topology(args.nodes_file)
    nodes = resolve_nodes(topology, config, args.nodes)
    transport = make_transport(
        config, topology, local=args.local, password=get_ssh_password(config)
    )
    executor = Executor(
        transport, log_path=args.log, subscribers=[lambda line: print(line, flush=True)]
    )
    try:
        outcomes = executor.run_on_nodes(
            nodes, args.command, timeout=args.timeout, abort_on_failure=not args.no_abort
        )
    except ExperimentAbortedError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    worst = max((o.exit_status for o in outcomes.values()), default=0)
    print(
        f"{len(outcomes)} node(s): "
        + ", ".join(f"{k}={v.exit_status}" for k, v in sorted(outcomes.items()))
    )
    raise SystemExit(0 if worst == 0 else 1)


__all__ = [
    "Executor",
    "ExperimentAbortedError",
    "Failure",
    "FakeTransport",
    "classify_outcome",
    "main",
    "make_transport",
    "resolve_nodes",
]

if __name__ == "__main__":
    main()

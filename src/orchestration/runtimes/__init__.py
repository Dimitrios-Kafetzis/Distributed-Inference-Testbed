"""Inference runtimes behind one interface, registered by name.

A runtime is what runs in the profile stage of an experiment. The built-in
``testbed`` runtime is the per-layer profiling and partitioned-inference path
of the thesis; ``headsplit`` starts the author's external HeadSplit runtime on
the devices. A runtime also tells NNPM how to render its scenario block (a
JSON Schema), how to check that the devices are ready for it, and which
maintenance actions it offers (install, build a profile, distribute a
checkpoint), so that every runtime is configured and managed through the same
dialogs. New runtimes register through :func:`register_runtime`, through the
environment variable ``TESTBED_RUNTIME_MODULES`` (comma-separated module
names) or through the ``testbed.runtimes`` entry-point group.
"""

from __future__ import annotations

import importlib
import os
import shlex
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.common.nodes import Node, Topology
from src.common.runner import CommandError, CommandResult, CommandRunner, SubprocessRunner
from src.config import REPO_ROOT, Config
from src.network.ssh_executor import Executor

if TYPE_CHECKING:
    from src.orchestration.experiment import ExperimentContext

ENTRY_POINT_GROUP = "testbed.runtimes"
MODULES_ENV = "TESTBED_RUNTIME_MODULES"
DEFAULT_RUNTIME = "testbed"
BUILTIN_MODULES = ("src.orchestration.runtimes.testbed", "src.orchestration.runtimes.headsplit")

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"


class RuntimeActionError(RuntimeError):
    """A runtime refused to run, check or act (configuration or device state)."""


@dataclass(frozen=True)
class Check:
    """One readiness check of one node."""

    name: str
    status: str  # ok | warn | fail | skip
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class ActionSpec:
    """A maintenance action: its parameters are rendered by NNPM from ``schema``."""

    name: str
    label: str
    description: str
    schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "schema": self.schema,
        }


@dataclass
class ActionSession:
    """What a check or an action may use: the devices through the executor, the edge
    server through a command runner, a log sink and a cancel flag."""

    config: Config
    topology: Topology
    executor: Executor
    local: bool = False
    runner: CommandRunner = field(default_factory=SubprocessRunner)
    log: Callable[[str], None] = lambda line: None
    cancel: threading.Event = field(default_factory=threading.Event)
    repo_root: Path = REPO_ROOT

    def run_local(
        self, argv: Sequence[str], *, cwd: Path | None = None, check: bool = True
    ) -> CommandResult:
        """A command on the edge server, logged like a device command."""
        self.log(f"[edge-server][exec] {shlex.join(argv)}" + (f" (in {cwd})" if cwd else ""))
        argv = list(argv)
        if cwd is not None:
            argv = ["bash", "-c", f"cd {shlex.quote(str(cwd))} && exec {shlex.join(argv)}"]
        result = self.runner.run(argv, timeout=24 * 3600)
        for stream, text in (("stdout", result.stdout), ("stderr", result.stderr)):
            for line in text.splitlines():
                self.log(f"[edge-server][{stream}] {line}")
        if check and not result.ok:
            raise CommandError(result)
        return result

    def devices(self, node_ids: Sequence[str] | None) -> list[Node]:
        if not node_ids:
            return self.topology.end_devices()
        return [self.topology.get(node_id) for node_id in node_ids]

    def stopped(self) -> bool:
        return self.cancel.is_set()

    def ssh_options(self) -> list[str]:
        """The ssh options the provisioning scripts use, for rsync's ``-e``."""
        c = self.config
        options = ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={c.ssh_connect_timeout_s:g}"]
        if c.ssh_known_hosts_path:
            options += ["-o", f"UserKnownHostsFile={c.ssh_known_hosts_path}"]
        if c.ssh_key_path:
            options += ["-i", c.ssh_key_path]
        return options

    def copy_to_node(self, node: Node, local_dir: Path, remote_dir: str) -> None:
        """Copy a directory to a node: rsync over the management network, or a plain
        copy into the node's working directory in local mode."""
        if node.transport == "local":
            import shutil

            target = Path(node.workdir or ".") / remote_dir
            self.log(f"[{node.id}][exec] copy {local_dir} -> {target}")
            shutil.copytree(local_dir, target, dirs_exist_ok=True)
            return
        destination = f"{self.config.ssh_username}@{node.management_address}:{remote_dir}/"
        self.executor.run_on_node(node, f"mkdir -p {shlex.quote(remote_dir)}")
        ssh = shlex.join(["ssh", *self.ssh_options()])
        self.run_local(["rsync", "-az", "-e", ssh, f"{local_dir}/", destination])


@runtime_checkable
class Runtime(Protocol):
    name: str
    label: str
    external: bool  # no per-layer profile: aggregation and optimisation are skipped
    description: str

    @property
    def schema(self) -> dict[str, Any]:
        """JSON Schema of the scenario block ``scenario[name]`` ({} when there is none)."""
        ...

    def run(self, ctx: ExperimentContext) -> None:
        """The profile stage of the experiment pipeline."""
        ...

    def check_node(self, session: ActionSession, node: Node, params: dict[str, Any]) -> list[Check]:
        """Readiness checks of one node (an empty list when the runtime needs nothing)."""
        ...

    def actions(self) -> list[ActionSpec]: ...

    def perform(self, action: str, params: dict[str, Any], session: ActionSession) -> None: ...

    def results(self, run_dir: Path) -> dict[str, Any] | None:
        """The runtime's parsed results of a finished run, for the run summary page."""
        ...


_REGISTRY: dict[str, Runtime] = {}
_LOADED = False


def register_runtime(runtime: Runtime) -> None:
    existing = _REGISTRY.get(runtime.name)
    if existing is not None and existing is not runtime:
        raise ValueError(f"runtime {runtime.name!r} is already registered")
    _REGISTRY[runtime.name] = runtime


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for module_name in BUILTIN_MODULES:
        register_runtime(importlib.import_module(module_name).RUNTIME)
    for module_name in filter(None, os.environ.get(MODULES_ENV, "").split(",")):
        register_runtime(importlib.import_module(module_name.strip()).RUNTIME)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        register_runtime(ep.load())


def names() -> list[str]:
    _load()
    return sorted(_REGISTRY)


def get_runtime(name: str) -> Runtime:
    _load()
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown runtime {name!r}; registered: {names()}") from exc


def iter_runtimes() -> Iterator[Runtime]:
    _load()
    for name in sorted(_REGISTRY):
        yield _REGISTRY[name]


def describe() -> list[dict[str, Any]]:
    """The runtimes as NNPM's catalogue lists them."""
    return [
        {
            "name": r.name,
            "label": r.label,
            "description": r.description,
            "external": r.external,
            "schema": r.schema,
            "actions": [a.to_dict() for a in r.actions()],
        }
        for r in iter_runtimes()
    ]


def block_schemas() -> dict[str, dict[str, Any]]:
    """``{runtime name: schema}`` for every runtime that has a scenario block."""
    return {r.name: r.schema for r in iter_runtimes() if r.schema}


def reset_for_tests() -> None:
    global _LOADED
    _LOADED = False
    _REGISTRY.clear()

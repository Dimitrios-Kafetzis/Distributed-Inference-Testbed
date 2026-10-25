"""Transports: how a command reaches a node and how its output comes back.

The SSH executor talks to nodes through this interface. Three transports:

* :class:`ParamikoTransport`: SSH with Paramiko (production);
* :class:`LocalTransport`: runs the command as a local subprocess in a
  per-node working directory (local mode: every node simulated on one machine);
* :class:`FakeTransport`: scripted outcomes for tests.

Every transport streams stdout and stderr line by line as they arrive, reports
the exit status, classifies failures (authentication, unreachable, command)
and can fetch a file from the node.
"""

from __future__ import annotations

import select
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.common.nodes import Node
from src.config import Config

LineSink = Callable[[str, str], None]  # (stream name, line)


class TransportError(RuntimeError):
    """Base class of the failure classes."""

    def __init__(self, node_id: str, message: str) -> None:
        self.node_id = node_id
        super().__init__(f"{node_id}: {message}")


class AuthenticationError(TransportError):
    """Credentials rejected; raised before the command is dispatched."""


class UnreachableHostError(TransportError):
    """No response within the connection timeout, or a connection error."""


@dataclass
class CommandOutcome:
    node_id: str
    command: str
    exit_status: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_status == 0 and not self.cancelled


class Transport(Protocol):
    def run(
        self,
        node: Node,
        command: str,
        *,
        on_line: LineSink | None = None,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandOutcome: ...

    def fetch(self, node: Node, remote_path: str, local_path: Path) -> Path: ...


def _env_prefix(env: Mapping[str, str] | None) -> str:
    if not env:
        return ""
    import shlex

    return " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items())) + " "


# --- Paramiko -----------------------------------------------------------------------------------


class ParamikoTransport:
    def __init__(
        self, config: Config, password: str | None = None, client_factory: Any = None
    ) -> None:
        self.config = config
        self.password = password
        self._client_factory = client_factory

    def _client(self) -> Any:
        import paramiko

        if self._client_factory is not None:
            return self._client_factory()
        client = paramiko.SSHClient()
        known_hosts = self.config.ssh_known_hosts_path
        if known_hosts:
            client.load_host_keys(known_hosts)
        else:
            client.load_system_host_keys()
        policy = (
            paramiko.AutoAddPolicy() if self.config.ssh_trust_new_hosts else paramiko.RejectPolicy()
        )
        client.set_missing_host_key_policy(policy)
        return client

    def _connect(self, node: Node) -> Any:
        import paramiko

        client = self._client()
        try:
            client.connect(
                node.management_address,
                username=self.config.ssh_username,
                password=self.password,
                key_filename=self.config.ssh_key_path,
                timeout=self.config.ssh_connect_timeout_s,
                banner_timeout=self.config.ssh_connect_timeout_s,
                auth_timeout=self.config.ssh_connect_timeout_s,
                allow_agent=self.password is None,
                look_for_keys=self.password is None,
            )
        except paramiko.AuthenticationException as exc:
            raise AuthenticationError(node.id, f"authentication failed ({exc})") from exc
        except paramiko.ssh_exception.NoValidConnectionsError as exc:
            raise UnreachableHostError(
                node.id, f"no connection to {node.management_address} ({exc})"
            ) from exc
        except TimeoutError as exc:
            raise UnreachableHostError(
                node.id,
                f"no response from {node.management_address} within "
                f"{self.config.ssh_connect_timeout_s} s",
            ) from exc
        except (OSError, paramiko.SSHException) as exc:
            raise UnreachableHostError(
                node.id, f"connection to {node.management_address} failed ({exc})"
            ) from exc
        return client

    def run(
        self,
        node: Node,
        command: str,
        *,
        on_line: LineSink | None = None,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandOutcome:
        started = time.perf_counter()
        client = self._connect(node)
        full_command = _env_prefix(env) + command
        stdout_buf: list[str] = []
        stderr_buf: list[str] = []
        cancelled = False
        try:
            channel = client.get_transport().open_session()
            channel.settimeout(1.0)
            channel.exec_command(full_command)
            partial = {"stdout": "", "stderr": ""}

            def feed(stream: str, data: str) -> None:
                partial[stream] += data
                while "\n" in partial[stream]:
                    line, partial[stream] = partial[stream].split("\n", 1)
                    (stdout_buf if stream == "stdout" else stderr_buf).append(line)
                    if on_line:
                        on_line(stream, line)

            while True:
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    channel.close()
                    break
                if timeout is not None and time.perf_counter() - started > timeout:
                    cancelled = True
                    channel.close()
                    feed("stderr", f"command timed out after {timeout} s\n")
                    break
                readable, _, _ = select.select([channel], [], [], 0.2)
                if readable:
                    while channel.recv_ready():
                        feed("stdout", channel.recv(65536).decode(errors="replace"))
                    while channel.recv_stderr_ready():
                        feed("stderr", channel.recv_stderr(65536).decode(errors="replace"))
                if (
                    channel.exit_status_ready()
                    and not channel.recv_ready()
                    and not channel.recv_stderr_ready()
                ):
                    break
            for stream in ("stdout", "stderr"):
                if partial[stream]:
                    feed(stream, "\n")
            exit_status = 255 if cancelled else channel.recv_exit_status()
        finally:
            client.close()
        return CommandOutcome(
            node.id,
            command,
            exit_status,
            "\n".join(stdout_buf),
            "\n".join(stderr_buf),
            time.perf_counter() - started,
            cancelled,
        )

    def fetch(self, node: Node, remote_path: str, local_path: Path) -> Path:
        client = self._connect(node)
        try:
            sftp = client.open_sftp()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_path, str(local_path))
            sftp.close()
        finally:
            client.close()
        return local_path


# --- Local mode --------------------------------------------------------------------------------


class LocalTransport:
    """Runs commands as local subprocesses, one working directory per node."""

    def __init__(
        self, workdirs: Mapping[str, Path], base_env: Mapping[str, str] | None = None
    ) -> None:
        self.workdirs = {k: Path(v) for k, v in workdirs.items()}
        self.base_env = dict(base_env or {})

    def workdir(self, node: Node) -> Path:
        if node.id not in self.workdirs:
            if node.workdir:
                self.workdirs[node.id] = Path(node.workdir)
            else:
                raise UnreachableHostError(node.id, "no local working directory configured")
        path = self.workdirs[node.id]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run(
        self,
        node: Node,
        command: str,
        *,
        on_line: LineSink | None = None,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandOutcome:
        import os

        started = time.perf_counter()
        workdir = self.workdir(node)
        full_env = dict(os.environ)
        full_env.update(self.base_env)
        full_env.update(env or {})
        full_env.update(
            {
                "TESTBED_NODE_ID": node.id,
                "TESTBED_LOCAL_MODE": "1",
                "TESTBED_NODE_DIR": str(workdir),
            }
        )
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=workdir,
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        buffers: dict[str, list[str]] = {"stdout": [], "stderr": []}
        lock = threading.Lock()

        def pump(stream_name: str, pipe: Any) -> None:
            for raw in iter(pipe.readline, ""):
                line = raw.rstrip("\n")
                with lock:
                    buffers[stream_name].append(line)
                    if on_line:
                        on_line(stream_name, line)
            pipe.close()

        threads = [
            threading.Thread(target=pump, args=("stdout", process.stdout), daemon=True),
            threading.Thread(target=pump, args=("stderr", process.stderr), daemon=True),
        ]
        for t in threads:
            t.start()
        cancelled = False
        while True:
            try:
                process.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    process.terminate()
                    process.wait(timeout=5)
                    break
                if timeout is not None and time.perf_counter() - started > timeout:
                    cancelled = True
                    process.terminate()
                    process.wait(timeout=5)
                    with lock:
                        buffers["stderr"].append(f"command timed out after {timeout} s")
                    break
        for t in threads:
            t.join(timeout=2)
        return CommandOutcome(
            node.id,
            command,
            255 if cancelled else process.returncode,
            "\n".join(buffers["stdout"]),
            "\n".join(buffers["stderr"]),
            time.perf_counter() - started,
            cancelled,
        )

    def fetch(self, node: Node, remote_path: str, local_path: Path) -> Path:
        source = Path(remote_path)
        if not source.is_absolute():
            source = self.workdir(node) / source
        if not source.exists():
            raise FileNotFoundError(f"{node.id}: {remote_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local_path)
        return local_path


# --- Fake -----------------------------------------------------------------------------------------


@dataclass
class ScriptedOutcome:
    exit_status: int = 0
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)
    delay_s: float = 0.0
    raise_error: type[TransportError] | None = None


@dataclass
class FakeTransport:
    """Scripted outcomes keyed by (node id, command predicate); records every call."""

    scripts: list[tuple[Callable[[str, str], bool], ScriptedOutcome]] = field(default_factory=list)
    default: ScriptedOutcome = field(default_factory=ScriptedOutcome)
    calls: list[tuple[str, str]] = field(default_factory=list)
    files: dict[tuple[str, str], str] = field(default_factory=dict)

    def on(self, node_id: str | None, substring: str, outcome: ScriptedOutcome) -> FakeTransport:
        def match(nid: str, command: str) -> bool:
            return (node_id is None or nid == node_id) and substring in command

        self.scripts.append((match, outcome))
        return self

    def run(
        self,
        node: Node,
        command: str,
        *,
        on_line: LineSink | None = None,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandOutcome:
        self.calls.append((node.id, command))
        outcome = next((o for m, o in self.scripts if m(node.id, command)), self.default)
        if outcome.raise_error is not None:
            raise outcome.raise_error(node.id, "scripted failure")
        if outcome.delay_s:
            deadline = time.perf_counter() + outcome.delay_s
            while time.perf_counter() < deadline:
                if cancel is not None and cancel.is_set():
                    return CommandOutcome(node.id, command, 255, cancelled=True)
                time.sleep(0.01)
        for line in outcome.stdout_lines:
            if on_line:
                on_line("stdout", line)
        for line in outcome.stderr_lines:
            if on_line:
                on_line("stderr", line)
        return CommandOutcome(
            node.id,
            command,
            outcome.exit_status,
            "\n".join(outcome.stdout_lines),
            "\n".join(outcome.stderr_lines),
        )

    def fetch(self, node: Node, remote_path: str, local_path: Path) -> Path:
        key = (node.id, remote_path)
        if key not in self.files:
            raise FileNotFoundError(f"{node.id}: {remote_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(self.files[key], encoding="utf-8")
        return local_path

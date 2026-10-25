"""Tagged, detached device-side processes.

The testbed is daemonless: everything that runs on a device (monitoring agent,
load generator, latency injector, inference worker, power monitor) is started
over SSH for the duration of an experiment and stopped at its end. Each such
process is started in its own session and process group with ``setsid``, so
that stopping it also stops its children; the group id is written to
``<run_dir>/<component>.pid``. The process also carries a tag
``testbed-<run_id>-<component>`` in the environment variable ``TESTBED_TAG``;
the testbed's own tools repeat it as a ``--tag`` argument, so that
``pkill -f`` can find survivors whose PID file was lost.

This module only builds shell command strings and parses their output; the
SSH executor runs them.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

TAG_PREFIX = "testbed"
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _self_safe(pattern: str) -> str:
    """``pkill -f``/``pgrep -f`` pattern that does not match the shell running it.

    The first character goes into a bracket expression: ``[t]estbed-...`` matches
    ``testbed-...`` in other processes' command lines, but the pattern itself,
    which appears in the stopping shell's command line, does not match.
    """
    return f"[{pattern[0]}]{pattern[1:]}"


def make_tag(run_id: str, component: str) -> str:
    for part in (run_id, component):
        if not _TAG_RE.match(part):
            raise ValueError(f"invalid tag component: {part!r}")
    return f"{TAG_PREFIX}-{run_id}-{component}"


@dataclass(frozen=True)
class TaggedProcess:
    run_id: str
    component: str
    run_dir: str  # directory on the device, for the PID file and the log

    @property
    def tag(self) -> str:
        return make_tag(self.run_id, self.component)

    @property
    def pid_file(self) -> str:
        return f"{self.run_dir}/{self.component}.pid"

    @property
    def log_file(self) -> str:
        return f"{self.run_dir}/{self.component}.log"

    def start_command(self, command: str, env: dict[str, str] | None = None) -> str:
        """Shell command that starts ``command`` detached in its own process group.

        The background job of a non-interactive shell is not a group leader, so
        ``setsid`` does not fork: the PID the shell reports is the new group id.
        It is written to the PID file and printed.
        """
        assignments = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted((env or {}).items()))
        # The tag and the environment are exported, then the command runs in the wrapper
        # shell (it may start with "cd ... &&"); the process group covers both.
        inner = f"export TESTBED_TAG={self.tag} {assignments}; {command}"
        # The braces keep the directory creation in the foreground and make $! the PID of
        # the setsid command alone; a background job is not a group leader, so setsid does
        # not fork and that PID is the new process group id.
        return (
            f"mkdir -p {shlex.quote(self.run_dir)} && {{ "
            f"setsid nohup sh -c {shlex.quote(inner)} "
            f"> {shlex.quote(self.log_file)} 2>&1 < /dev/null & "
            f"echo $! > {shlex.quote(self.pid_file)}; }} && cat {shlex.quote(self.pid_file)}"
        )

    def stop_command(self, grace_s: float = 2.0) -> str:
        """SIGTERM the process group (and anything carrying the tag), wait, then SIGKILL."""
        pid_file = shlex.quote(self.pid_file)
        safe = _self_safe(self.tag)
        return (
            f"pgid=$(cat {pid_file} 2>/dev/null); "
            f'if [ -n "$pgid" ]; then kill -s TERM -- "-$pgid" 2>/dev/null; fi; '
            f"pkill -TERM -f {safe} 2>/dev/null; "
            f"sleep {grace_s:g}; "
            f'if [ -n "$pgid" ]; then kill -s KILL -- "-$pgid" 2>/dev/null; fi; '
            f"pkill -KILL -f {safe} 2>/dev/null; "
            f"rm -f {pid_file}; "
            f"true"
        )

    def check_command(self) -> str:
        """Prints ``running`` if the group or a tagged process exists, else ``stopped``."""
        pid_file = shlex.quote(self.pid_file)
        safe = _self_safe(self.tag)
        return (
            f"pgid=$(cat {pid_file} 2>/dev/null); "
            f'if {{ [ -n "$pgid" ] && kill -s 0 -- "-$pgid" 2>/dev/null; }} '
            f"|| pgrep -f {safe} > /dev/null 2>&1; "
            f"then echo running; else echo stopped; fi"
        )


def is_running(check_output: str) -> bool:
    return check_output.strip().splitlines()[-1:] == ["running"]


def stop_all_command(run_id: str, grace_s: float = 2.0) -> str:
    """Stop every tagged process of a run that still carries the tag (clean-up by tag)."""
    safe = _self_safe(f"{TAG_PREFIX}-{run_id}-")
    return (
        f"pkill -TERM -f {safe} 2>/dev/null; sleep {grace_s:g}; "
        f"pkill -KILL -f {safe} 2>/dev/null; true"
    )


def parse_pids(output: str) -> list[int]:
    """PIDs from ``pgrep`` output."""
    pids = []
    for line in output.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids

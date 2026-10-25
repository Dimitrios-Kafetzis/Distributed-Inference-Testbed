"""Command runners: run external commands, or pretend to.

Modules that call ``tc``, ``ping``, ``iperf3``, ``chronyc``, ``vcgencmd``,
``scp``, ``pkill`` and the like take a :class:`CommandRunner`. Production code
uses :class:`SubprocessRunner`; tests use :class:`FakeRunner`, which replays
scripted results and records every call; ``--dry-run`` modes use
:class:`DryRunRunner`, which prints the commands and reports success.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TextIO


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def command(self) -> str:
        return shlex.join(self.argv)


class CommandError(RuntimeError):
    """A command returned a non-zero exit status and ``check`` was requested."""

    def __init__(self, result: CommandResult) -> None:
        self.result = result
        detail = result.stderr.strip() or result.stdout.strip()
        message = f"command failed with status {result.returncode}: {result.command}"
        super().__init__(f"{message}\n{detail}" if detail else message)


class CommandRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
        check: bool = False,
        input: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult: ...


class SubprocessRunner:
    """Runs commands with :mod:`subprocess`, capturing text output."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
        check: bool = False,
        input: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout,
                input=input,
                env=dict(env) if env is not None else None,
                check=False,
            )
            result = CommandResult(
                tuple(argv),
                completed.returncode,
                completed.stdout,
                completed.stderr,
                time.perf_counter() - started,
            )
        except FileNotFoundError as exc:
            result = CommandResult(
                tuple(argv), 127, "", f"{exc.filename}: not found", time.perf_counter() - started
            )
        except subprocess.TimeoutExpired as exc:
            result = CommandResult(
                tuple(argv),
                124,
                _text(exc.stdout),
                _text(exc.stderr) + f"\ntimed out after {timeout} s",
                time.perf_counter() - started,
            )
        if check and not result.ok:
            raise CommandError(result)
        return result


def _text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    return data.decode(errors="replace") if isinstance(data, bytes) else data


Matcher = Callable[[Sequence[str]], bool]
Reply = CommandResult | Callable[[Sequence[str]], CommandResult]


def prefix(*words: str) -> Matcher:
    """Matcher: the command starts with ``words``."""

    def match(argv: Sequence[str]) -> bool:
        return tuple(argv[: len(words)]) == words

    return match


def contains(*words: str) -> Matcher:
    """Matcher: every word appears somewhere in the command."""

    def match(argv: Sequence[str]) -> bool:
        return all(w in argv for w in words)

    return match


def ok(stdout: str = "") -> Callable[[Sequence[str]], CommandResult]:
    return lambda argv: CommandResult(tuple(argv), 0, stdout)


def fail(returncode: int = 1, stderr: str = "") -> Callable[[Sequence[str]], CommandResult]:
    return lambda argv: CommandResult(tuple(argv), returncode, "", stderr)


@dataclass
class FakeRunner:
    """Replays scripted replies and records every call.

    ``replies`` is an ordered list of (matcher, reply); the first matching entry
    answers a call. With ``strict`` (default) an unmatched command raises, so a
    test fails on any command it did not expect; otherwise it succeeds silently.
    """

    replies: list[tuple[Matcher, Reply]] = field(default_factory=list)
    strict: bool = True
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def on(self, matcher: Matcher, reply: Reply) -> FakeRunner:
        self.replies.append((matcher, reply))
        return self

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
        check: bool = False,
        input: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        argv_t = tuple(argv)
        self.calls.append(argv_t)
        for matcher, reply in self.replies:
            if matcher(argv_t):
                result = reply(argv_t) if callable(reply) else reply
                break
        else:
            if self.strict:
                raise AssertionError(f"unexpected command: {shlex.join(argv_t)}")
            result = CommandResult(argv_t, 0)
        if check and not result.ok:
            raise CommandError(result)
        return result

    def commands(self) -> list[str]:
        return [shlex.join(c) for c in self.calls]


@dataclass
class DryRunRunner:
    """Prints each command instead of running it and reports success."""

    stream: TextIO = field(default_factory=lambda: sys.stdout)
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float | None = None,
        check: bool = False,
        input: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        argv_t = tuple(argv)
        self.calls.append(argv_t)
        print(f"[dry-run] {shlex.join(argv_t)}", file=self.stream)
        return CommandResult(argv_t, 0)

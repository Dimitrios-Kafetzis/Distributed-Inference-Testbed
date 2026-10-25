"""Injectable clocks.

Interval timing uses ``perf_counter_ns`` (monotonic, nanosecond resolution,
Appendix B); the NTP-synchronised wall clock ``time.time()`` is used only for
the start-signal origin of every experiment.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def perf_counter_ns(self) -> int: ...

    def time(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def perf_counter_ns(self) -> int:
        return time.perf_counter_ns()

    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class FakeClock:
    """A clock that advances only when told to, or when something sleeps."""

    def __init__(self, start_ns: int = 1_000_000_000, start_time: float = 1_700_000_000.0) -> None:
        self._ns = start_ns
        self._time = start_time
        self.sleeps: list[float] = []

    def perf_counter_ns(self) -> int:
        return self._ns

    def time(self) -> float:
        return self._time

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a clock cannot go backwards")
        self._ns += round(seconds * 1e9)
        self._time += seconds

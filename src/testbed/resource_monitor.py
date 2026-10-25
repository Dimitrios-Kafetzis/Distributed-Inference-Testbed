"""Resource-monitoring agent: CPU, memory, temperature and clock frequency at 1 Hz.

A lightweight agent runs on each device throughout every experiment (started
over SSH, stopped at the end) and samples CPU utilisation, memory
consumption, CPU temperature and the clock frequency of every core at
one-second intervals, plus the firmware throttling flags where available.
The trace confirms that the CPU stayed at its nominal frequency (no thermal
throttling) and quantifies memory pressure. It is written as a
``ResourceTrace`` record, the same serialisation family as the inference
timings, and merged with them during aggregation.

Usage::

    python -m src.testbed.resource_monitor --output trace.json --run-id R --node-id device1
        [--interval 1.0] [--duration S] [--origin-wall T --origin-perf NS]
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import psutil

from src.benchmarking.benchmark_results import Origin, ResourceSample, ResourceTrace, new_record
from src.common.clock import Clock, SystemClock
from src.common.runner import CommandRunner, SubprocessRunner
from src.common.sysfs import SysFs

THERMAL_GLOB = "/sys/class/thermal/thermal_zone*/temp"
CUR_FREQ_GLOB = "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"
MAX_FREQ = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"


class SystemStats(Protocol):
    def cpu_percent(self) -> float: ...

    def memory(self) -> tuple[float, float]:
        """(used MB, available MB)"""
        ...


class PsutilStats:
    def __init__(self) -> None:
        psutil.cpu_percent(interval=None)  # prime the counter

    def cpu_percent(self) -> float:
        return float(psutil.cpu_percent(interval=None))

    def memory(self) -> tuple[float, float]:
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / 2**20, vm.available / 2**20


class ResourceMonitor:
    def __init__(
        self,
        *,
        run_id: str,
        node_id: str,
        output: Path,
        interval_s: float = 1.0,
        origin: Origin | None = None,
        hostname: str | None = None,
        stats: SystemStats | None = None,
        sysfs: SysFs | None = None,
        runner: CommandRunner | None = None,
        clock: Clock | None = None,
        flush_every: int = 10,
        nominal_khz: int | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.clock = clock or SystemClock()
        self.stats = stats or PsutilStats()
        self.sysfs = sysfs or SysFs()
        self.runner = runner or SubprocessRunner()
        self.output = output
        self.interval_s = interval_s
        self.flush_every = flush_every
        origin = origin or Origin(self.clock.time(), self.clock.perf_counter_ns())
        if nominal_khz is None and self.sysfs.exists(MAX_FREQ):
            nominal_khz = self.sysfs.read_int(MAX_FREQ)
        record = new_record(
            "resource_trace",
            run_id=run_id,
            hostname=hostname or socket.gethostname(),
            node_id=node_id,
            origin=origin,
            interval_s=interval_s,
            nominal_frequency_khz=nominal_khz,
            metadata={"pid": os.getpid()},
        )
        assert isinstance(record, ResourceTrace)
        self.trace = record
        self._stop = threading.Event()
        self._vcgencmd_available: bool | None = None

    # --- readings -------------------------------------------------------------------------

    def temperature_c(self) -> float | None:
        for path in self.sysfs.glob(THERMAL_GLOB):
            try:
                return self.sysfs.read_int(path) / 1000.0
            except ValueError:
                continue
        result = self._vcgencmd("measure_temp")
        if result and result.startswith("temp="):
            try:
                return float(result[5:].split("'")[0])
            except ValueError:
                return None
        return None

    def frequencies_khz(self) -> list[int]:
        values = []
        for path in self.sysfs.glob(CUR_FREQ_GLOB):
            try:
                values.append(self.sysfs.read_int(path))
            except ValueError:
                continue
        return values

    def throttled(self) -> str | None:
        result = self._vcgencmd("get_throttled")
        if result and result.startswith("throttled="):
            return result.split("=", 1)[1].strip()
        return None

    def _vcgencmd(self, argument: str) -> str | None:
        if self._vcgencmd_available is False:
            return None
        result = self.runner.run(["vcgencmd", argument], timeout=2)
        if result.returncode == 127 or not result.ok:
            self._vcgencmd_available = False
            return None
        self._vcgencmd_available = True
        return result.stdout.strip()

    def sample(self) -> ResourceSample:
        used, available = self.stats.memory()
        return ResourceSample(
            perf_counter_ns=self.clock.perf_counter_ns(),
            cpu_percent=self.stats.cpu_percent(),
            memory_used_mb=round(used, 2),
            memory_available_mb=round(available, 2),
            temperature_c=self.temperature_c(),
            frequencies_khz=self.frequencies_khz(),
            throttled=self.throttled(),
        )

    # --- loop ------------------------------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    def flush(self) -> None:
        self.trace.save(self.output)

    def run(self, duration_s: float | None = None, max_samples: int | None = None) -> ResourceTrace:
        started = self.clock.perf_counter_ns()
        next_tick = started
        interval_ns = int(self.interval_s * 1e9)
        try:
            while not self._stop.is_set():
                if (
                    duration_s is not None
                    and (self.clock.perf_counter_ns() - started) >= duration_s * 1e9
                ):
                    break
                if max_samples is not None and len(self.trace.samples) >= max_samples:
                    break
                self.trace.samples.append(self.sample())
                if len(self.trace.samples) % self.flush_every == 0:
                    self.flush()
                next_tick += interval_ns
                wait = (next_tick - self.clock.perf_counter_ns()) / 1e9
                if wait > 0:
                    self._wait(wait)
        finally:
            self.flush()
        return self.trace

    def _wait(self, seconds: float) -> None:
        if isinstance(self.clock, SystemClock):
            self._stop.wait(seconds)
        else:
            self.clock.sleep(seconds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample CPU, memory, temperature and frequency.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--origin-wall", type=float, help="wall-clock origin from the start signal")
    parser.add_argument(
        "--origin-perf", type=int, help="perf_counter_ns origin from the start signal"
    )
    parser.add_argument("--nominal-khz", type=int, default=None)
    parser.add_argument("--sysfs-root", default="/")
    parser.add_argument("--tag", help="process tag (ignored; present for pkill -f)")
    args = parser.parse_args(argv)

    origin = None
    if args.origin_wall is not None and args.origin_perf is not None:
        origin = Origin(args.origin_wall, args.origin_perf)
    monitor = ResourceMonitor(
        run_id=args.run_id,
        node_id=args.node_id,
        output=args.output,
        interval_s=args.interval,
        origin=origin,
        sysfs=SysFs(args.sysfs_root),
        nominal_khz=args.nominal_khz,
    )

    def handle_signal(signum: int, _frame: Any) -> None:
        monitor.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    trace = monitor.run(duration_s=args.duration)
    print(f"{len(trace.samples)} samples written to {args.output}", file=sys.stderr)


_: Callable[..., Any] | None = None  # keeps Callable imported for type checkers in stubs

if __name__ == "__main__":
    main()

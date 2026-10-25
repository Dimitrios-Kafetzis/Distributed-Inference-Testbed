"""INA260 current-and-power monitor, and energy metrics.

Each Raspberry Pi 4 can carry an INA260 current/power sensor connected over
I2C. This module, built on the Adafruit CircuitPython INA260 driver, reads
current (mA), voltage (V) and power (mW) at a user-configurable interval and
records the timestamped measurements to a JSON file (a ``PowerTrace``
record), together with per-metric averages. The sensor reports current in
steps of 1.25 mA, bus voltage in steps of 1.25 mV and power in steps of
10 mW.

The ``energy`` functions integrate a trace into total energy and derive
energy per inference, energy per token and the energy-delay product (EDP).

Usage::

    python -m src.network.power_measurement --interval 0.5 --duration 60 --output power.json
        --run-id R --node-id device1 [--fake]
"""

from __future__ import annotations

import argparse
import itertools
import signal
import socket
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from src.benchmarking.benchmark_results import Origin, PowerSample, PowerTrace, new_record
from src.common.clock import Clock, SystemClock


class Sensor(Protocol):
    def read(self) -> tuple[float, float, float]:
        """(current mA, voltage V, power mW)"""
        ...


class INA260Sensor:
    """The real sensor; imports the hardware libraries only when instantiated."""

    def __init__(self, address: int = 0x40) -> None:
        try:
            import adafruit_ina260
            import board
        except ImportError as exc:  # pragma: no cover - hardware only
            raise RuntimeError(
                "INA260 libraries not installed: pip install -r requirements/power.in"
            ) from exc
        self._device = adafruit_ina260.INA260(board.I2C(), address=address)

    def read(self) -> tuple[float, float, float]:  # pragma: no cover - hardware only
        return float(self._device.current), float(self._device.voltage), float(self._device.power)


class FakeSensor:
    """Returns scripted or constant readings (tests, local mode)."""

    def __init__(self, readings: Sequence[tuple[float, float, float]] | None = None) -> None:
        self.readings = list(readings or [(800.0, 5.0, 4000.0)])
        self.calls = 0

    def read(self) -> tuple[float, float, float]:
        value = self.readings[min(self.calls, len(self.readings) - 1)]
        self.calls += 1
        return value


class PowerMonitor:
    def __init__(
        self,
        sensor: Sensor,
        *,
        run_id: str,
        node_id: str,
        output: Path,
        interval_s: float = 1.0,
        origin: Origin | None = None,
        hostname: str | None = None,
        clock: Clock | None = None,
        flush_every: int = 20,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.sensor = sensor
        self.clock = clock or SystemClock()
        self.output = output
        self.interval_s = interval_s
        self.flush_every = flush_every
        record = new_record(
            "power_trace",
            run_id=run_id,
            hostname=hostname or socket.gethostname(),
            node_id=node_id,
            origin=origin or Origin(self.clock.time(), self.clock.perf_counter_ns()),
            interval_s=interval_s,
        )
        assert isinstance(record, PowerTrace)
        self.trace = record
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def flush(self) -> None:
        self.trace.metadata["averages"] = self.trace.averages()
        self.trace.metadata["energy_j"] = integrate_energy_j(self.trace.samples)
        self.trace.save(self.output)

    def run(self, duration_s: float | None = None, max_samples: int | None = None) -> PowerTrace:
        started = self.clock.perf_counter_ns()
        next_tick = started
        interval_ns = int(self.interval_s * 1e9)
        try:
            while not self._stop.is_set():
                if (
                    duration_s is not None
                    and self.clock.perf_counter_ns() - started >= duration_s * 1e9
                ):
                    break
                if max_samples is not None and len(self.trace.samples) >= max_samples:
                    break
                current, voltage, power = self.sensor.read()
                self.trace.samples.append(
                    PowerSample(self.clock.perf_counter_ns(), current, voltage, power)
                )
                if len(self.trace.samples) % self.flush_every == 0:
                    self.flush()
                next_tick += interval_ns
                wait = (next_tick - self.clock.perf_counter_ns()) / 1e9
                if wait > 0:
                    if isinstance(self.clock, SystemClock):
                        self._stop.wait(wait)
                    else:
                        self.clock.sleep(wait)
        finally:
            self.flush()
        return self.trace


# --- energy ------------------------------------------------------------------------------------


def integrate_energy_j(
    samples: Sequence[PowerSample], start_ns: int | None = None, end_ns: int | None = None
) -> float:
    """Trapezoidal integral of power (mW) over time (ns) -> joules, between two timestamps."""
    points = [
        s
        for s in samples
        if (start_ns is None or s.perf_counter_ns >= start_ns)
        and (end_ns is None or s.perf_counter_ns <= end_ns)
    ]
    if len(points) < 2:
        return 0.0
    energy_mj = 0.0
    for a, b in itertools.pairwise(points):
        dt_s = (b.perf_counter_ns - a.perf_counter_ns) / 1e9
        energy_mj += (a.power_mw + b.power_mw) / 2.0 * dt_s
    return energy_mj / 1000.0


def energy_per_unit_j(energy_j: float, units: int) -> float:
    """Energy per inference or per generated token."""
    if units <= 0:
        raise ValueError("units must be positive")
    return energy_j / units


def energy_delay_product(energy_j: float, latency_s: float) -> float:
    """EDP = total energy x end-to-end latency (J s)."""
    return energy_j * latency_s


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record INA260 current, voltage and power.")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds (may be fractional)")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--address", type=lambda v: int(v, 0), default=0x40, help="I2C address")
    parser.add_argument("--origin-wall", type=float)
    parser.add_argument("--origin-perf", type=int)
    parser.add_argument("--fake", action="store_true", help="use a fake sensor (no hardware)")
    parser.add_argument("--tag", help="process tag (ignored; present for pkill -f)")
    args = parser.parse_args(argv)

    sensor: Sensor = FakeSensor() if args.fake else INA260Sensor(args.address)
    origin = None
    if args.origin_wall is not None and args.origin_perf is not None:
        origin = Origin(args.origin_wall, args.origin_perf)
    monitor = PowerMonitor(
        sensor,
        run_id=args.run_id,
        node_id=args.node_id,
        output=args.output,
        interval_s=args.interval,
        origin=origin,
    )

    def handle_signal(signum: int, _frame: Any) -> None:
        monitor.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    trace = monitor.run(duration_s=args.duration)
    averages = trace.averages()
    print(
        f"{len(trace.samples)} samples; average {averages['current_ma']:.1f} mA, "
        f"{averages['voltage_v']:.3f} V, {averages['power_mw']:.1f} mW; "
        f"energy {trace.metadata['energy_j']:.3f} J -> {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

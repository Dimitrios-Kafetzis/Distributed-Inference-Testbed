"""Network shaping with tc-netem: delay, jitter, loss and a bandwidth cap.

Network conditions are emulated on a device's wireless interface with the
Linux traffic-control subsystem. Four parameters are configured
independently:

* **added latency**: a fixed delay appended to every outgoing packet
  (``tc qdisc add dev <iface> root netem delay <latency>ms``), with three
  time-variation modes: ``constant``, ``uniform`` (a new delay drawn from
  [min, max] every period) and ``square`` (periodic alternation between the
  delay and no delay);
* **jitter**: a uniformly distributed random component added to the delay;
* **packet loss**: a drop probability in percent;
* **bandwidth cap**: a token-bucket filter (``tbf``) that limits egress
  throughput to a rate in Mbit/s.

Shaping can apply to all traffic of the interface, or per destination
address (one netem leaf per destination under an HTB root), which gives a
different synthetic delay to every link of the map.

The old mode names ``once``, ``continuous`` and ``periodic`` are accepted as
aliases of ``constant``, ``uniform`` and ``square``.

Usage::

    python -m src.testbed.latency_injector wlan0 --delay 10
    python -m src.testbed.latency_injector wlan0 --delay 5 20 --mode uniform --period 30 \
        --duration 300
    python -m src.testbed.latency_injector wlan0 --delay 10 --jitter 2 --loss 0.5 --rate 25
    python -m src.testbed.latency_injector wlan0 --dst 192.0.2.11 --dst 192.0.2.12 --delay 15
    python -m src.testbed.latency_injector wlan0 --clear
"""

from __future__ import annotations

import argparse
import random
import signal
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from src.common.clock import Clock, SystemClock
from src.common.runner import CommandRunner, DryRunRunner, SubprocessRunner

MODE_ALIASES = {"once": "constant", "continuous": "uniform", "periodic": "square"}
MODES = ("constant", "uniform", "square")
UNSHAPED_RATE_MBPS = 10_000.0  # HTB class rate for destinations without a cap


@dataclass(frozen=True)
class NetemParams:
    delay_ms: float | None = None
    jitter_ms: float | None = None
    loss_percent: float | None = None
    rate_mbps: float | None = None

    def validate(self) -> None:
        for name, value in (
            ("delay_ms", self.delay_ms),
            ("jitter_ms", self.jitter_ms),
            ("loss_percent", self.loss_percent),
            ("rate_mbps", self.rate_mbps),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.loss_percent is not None and self.loss_percent > 100:
            raise ValueError("loss_percent must not exceed 100")
        if self.jitter_ms is not None and self.delay_ms is None:
            raise ValueError("jitter needs a delay")

    @property
    def is_empty(self) -> bool:
        return all(
            v is None for v in (self.delay_ms, self.jitter_ms, self.loss_percent, self.rate_mbps)
        )

    def netem_args(self) -> list[str]:
        args: list[str] = []
        if self.delay_ms is not None:
            args += ["delay", _ms(self.delay_ms)]
            if self.jitter_ms is not None:
                args += [_ms(self.jitter_ms), "distribution", "uniform"]
        if self.loss_percent is not None:
            args += ["loss", f"{self.loss_percent:g}%"]
        return args

    def tbf_args(self) -> list[str]:
        assert self.rate_mbps is not None
        # Burst of 32 kbit and a 400 ms queue are the usual tbf defaults for these rates.
        return ["rate", f"{self.rate_mbps:g}mbit", "burst", "32kbit", "latency", "400ms"]


def _ms(value: float) -> str:
    return f"{value:g}ms"


class TcPlan:
    """Builds and runs the ``tc`` commands for one interface."""

    def __init__(self, interface: str, runner: CommandRunner, sudo: bool = True) -> None:
        self.interface = interface
        self.runner = runner
        self.sudo = sudo

    def _tc(self, *args: str) -> list[str]:
        prefix = ["sudo", "tc"] if self.sudo else ["tc"]
        return [*prefix, *args]

    # --- all traffic of the interface ---------------------------------------------

    def commands_all(self, params: NetemParams, change: bool = False) -> list[list[str]]:
        params.validate()
        verb = "change" if change else "add"
        if params.rate_mbps is None:
            # The form Appendix B documents: tc qdisc add dev <iface> root netem delay <latency>ms
            return [
                self._tc(
                    "qdisc", verb, "dev", self.interface, "root", "netem", *params.netem_args()
                )
            ]
        commands = [
            self._tc(
                "qdisc",
                verb,
                "dev",
                self.interface,
                "root",
                "handle",
                "1:",
                "netem",
                *params.netem_args(),
            ),
            self._tc(
                "qdisc",
                verb,
                "dev",
                self.interface,
                "parent",
                "1:",
                "handle",
                "10:",
                "tbf",
                *params.tbf_args(),
            ),
        ]
        return commands

    # --- per destination ---------------------------------------------------------------

    def commands_per_destination(self, shaping: dict[str, NetemParams]) -> list[list[str]]:
        """HTB root with a default unshaped class; per destination one HTB class (rate
        cap) with a netem leaf (delay, jitter, loss) selected by a u32 filter on the
        destination address."""
        commands = [
            self._tc(
                "qdisc",
                "add",
                "dev",
                self.interface,
                "root",
                "handle",
                "1:",
                "htb",
                "default",
                "999",
            ),
            self._tc(
                "class",
                "add",
                "dev",
                self.interface,
                "parent",
                "1:",
                "classid",
                "1:999",
                "htb",
                "rate",
                f"{UNSHAPED_RATE_MBPS:g}mbit",
            ),
        ]
        for index, (destination, params) in enumerate(sorted(shaping.items()), start=1):
            params.validate()
            rate = params.rate_mbps if params.rate_mbps is not None else UNSHAPED_RATE_MBPS
            classid = f"1:{index}"
            handle = f"{index + 1}:"
            commands.append(
                self._tc(
                    "class",
                    "add",
                    "dev",
                    self.interface,
                    "parent",
                    "1:",
                    "classid",
                    classid,
                    "htb",
                    "rate",
                    f"{rate:g}mbit",
                    "ceil",
                    f"{rate:g}mbit",
                )
            )
            netem = params.netem_args()
            if netem:
                commands.append(
                    self._tc(
                        "qdisc",
                        "add",
                        "dev",
                        self.interface,
                        "parent",
                        classid,
                        "handle",
                        handle,
                        "netem",
                        *netem,
                    )
                )
            commands.append(
                self._tc(
                    "filter",
                    "add",
                    "dev",
                    self.interface,
                    "protocol",
                    "ip",
                    "parent",
                    "1:",
                    "prio",
                    "1",
                    "u32",
                    "match",
                    "ip",
                    "dst",
                    f"{destination}/32",
                    "flowid",
                    classid,
                )
            )
        return commands

    def commands_clear(self) -> list[list[str]]:
        return [self._tc("qdisc", "del", "dev", self.interface, "root")]

    # --- execution ----------------------------------------------------------------------

    def has_root_qdisc(self) -> bool:
        result = self.runner.run(self._tc("qdisc", "show", "dev", self.interface))
        return result.ok and any(
            line.split()[:3] in (["qdisc", "netem"], ["qdisc", "htb"], ["qdisc", "tbf"])
            or (
                line.startswith("qdisc")
                and " root " in f" {line} "
                and "noqueue" not in line
                and "mq " not in line
                and "fq_codel" not in line
                and "pfifo_fast" not in line
            )
            for line in result.stdout.splitlines()
        )

    def clear(self) -> None:
        """Remove the root qdisc; succeeds also when nothing is installed."""
        for command in self.commands_clear():
            self.runner.run(command)

    def apply_all(self, params: NetemParams) -> list[list[str]]:
        """Idempotent: an existing testbed qdisc is replaced."""
        if self.has_root_qdisc():
            self.clear()
        commands = self.commands_all(params)
        for command in commands:
            self.runner.run(command, check=True)
        return commands

    def apply_per_destination(self, shaping: dict[str, NetemParams]) -> list[list[str]]:
        if self.has_root_qdisc():
            self.clear()
        commands = self.commands_per_destination(shaping)
        for command in commands:
            self.runner.run(command, check=True)
        return commands


# --- time variation ---------------------------------------------------------------------------


def normalise_mode(mode: str) -> str:
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES} or {tuple(MODE_ALIASES)}")
    return mode


def draw_delay(delay_range: tuple[float, float], rng: random.Random) -> float:
    low, high = delay_range
    if low > high:
        raise ValueError("delay range must be min <= max")
    return round(rng.uniform(low, high), 3)


class VariationRunner:
    """Applies a delay profile over time: constant, uniform redraw or square wave."""

    def __init__(
        self,
        plan: TcPlan,
        base: NetemParams,
        delay_range: tuple[float, float],
        mode: str,
        period_s: float,
        duration_s: float | None,
        *,
        destinations: Sequence[str] = (),
        clock: Clock | None = None,
        seed: int = 0,
    ) -> None:
        self.plan = plan
        self.base = base
        self.delay_range = delay_range
        self.mode = normalise_mode(mode)
        self.period_s = period_s
        self.duration_s = duration_s
        self.destinations = list(destinations)
        self.clock = clock or SystemClock()
        self.rng = random.Random(seed)
        self.applied: list[float | None] = []
        self._stop = threading.Event()
        if self.mode != "constant" and period_s <= 0:
            raise ValueError("uniform and square modes need a positive period")

    def stop(self) -> None:
        self._stop.set()

    def _apply(self, delay_ms: float | None) -> None:
        params = replace(
            self.base,
            delay_ms=delay_ms,
            jitter_ms=self.base.jitter_ms if delay_ms is not None else None,
        )
        if params.is_empty:
            self.plan.clear()
        elif self.destinations:
            self.plan.apply_per_destination(dict.fromkeys(self.destinations, params))
        else:
            self.plan.apply_all(params)
        self.applied.append(delay_ms)

    def _wait(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if isinstance(self.clock, SystemClock):
            self._stop.wait(seconds)
        else:
            self.clock.sleep(seconds)

    def run(self) -> list[float | None]:
        """Returns the sequence of applied delays (None = cleared). With a duration,
        shaping is removed at the end; without one, a constant profile is left in
        place (the orchestrator removes it at teardown) and the other modes run until
        stopped."""
        started = self.clock.time()

        def elapsed() -> float:
            return self.clock.time() - started

        def time_left() -> bool:
            return self.duration_s is None or elapsed() < self.duration_s

        if self.mode == "constant":
            self._apply(draw_delay(self.delay_range, self.rng))
            if self.duration_s is not None:
                while time_left() and not self._stop.is_set():
                    self._wait(min(self.period_s or 1.0, self.duration_s - elapsed()))
                self.plan.clear()
                self.applied.append(None)
            return self.applied

        on = True
        while time_left() and not self._stop.is_set():
            if self.mode == "uniform":
                self._apply(draw_delay(self.delay_range, self.rng))
            else:
                self._apply(draw_delay(self.delay_range, self.rng) if on else None)
                on = not on
            wait = self.period_s
            if self.duration_s is not None:
                wait = min(wait, self.duration_s - elapsed())
            self._wait(wait)
        if self.applied and self.applied[-1] is not None:
            self.plan.clear()
            self.applied.append(None)
        return self.applied


# --- command line ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply or clear network shaping with tc-netem.")
    parser.add_argument("interface")
    parser.add_argument("--delay", type=float, nargs="+", metavar="MS", help="delay, or MIN MAX")
    parser.add_argument("--jitter", type=float, metavar="MS")
    parser.add_argument("--loss", type=float, metavar="PERCENT")
    parser.add_argument("--rate", type=float, metavar="MBPS", help="bandwidth cap (tbf)")
    parser.add_argument("--dst", action="append", default=[], help="destination IP (repeatable)")
    parser.add_argument(
        "--mode", default="constant", help="constant|uniform|square (or once|continuous|periodic)"
    )
    parser.add_argument("--period", type=float, default=0.0, help="seconds between redraws")
    parser.add_argument(
        "--duration", type=float, default=None, help="seconds; remove shaping at the end"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clear", action="store_true", help="remove all shaping and exit")
    parser.add_argument("--no-sudo", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the tc commands only")
    parser.add_argument("--tag", help="process tag (ignored; present for pkill -f)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    runner: CommandRunner = DryRunRunner() if args.dry_run else SubprocessRunner()
    plan = TcPlan(args.interface, runner, sudo=not args.no_sudo)
    if args.clear:
        plan.clear()
        print("shaping cleared", file=sys.stderr)
        return
    if not args.delay and args.loss is None and args.rate is None:
        raise SystemExit("nothing to apply: give --delay, --loss or --rate (or --clear)")
    if args.delay and len(args.delay) > 2:
        raise SystemExit("--delay takes one value or MIN MAX")
    base = NetemParams(jitter_ms=args.jitter, loss_percent=args.loss, rate_mbps=args.rate)
    if not args.delay:
        params = base
        if args.dst:
            plan.apply_per_destination(dict.fromkeys(args.dst, params))
        else:
            plan.apply_all(params)
        return
    delay_range = (args.delay[0], args.delay[-1])
    runner_v = VariationRunner(
        plan,
        base,
        delay_range,
        args.mode,
        args.period,
        args.duration,
        destinations=args.dst,
        seed=args.seed,
    )

    def handle_signal(signum: int, _frame: Any) -> None:
        runner_v.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    applied = runner_v.run()
    print(f"applied delays: {applied}", file=sys.stderr)


if __name__ == "__main__":
    main()

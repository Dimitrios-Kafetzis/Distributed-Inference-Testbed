"""Coordinated start: every device waits for the same wall-clock instant T0.

NNPM chooses T0 a few seconds in the future and passes it to every device-side
command. Each device sleeps until its NTP-synchronised clock reaches T0 and
then captures its time origin (the wall clock and the monotonic counter
together), which every trace carries so that the per-device traces can be
re-aligned during aggregation.
"""

from __future__ import annotations

import argparse
import json

from src.benchmarking.benchmark_results import Origin
from src.common.clock import Clock, SystemClock


def wait_until(
    t0_wall_s: float | None, clock: Clock | None = None, max_wait_s: float = 600.0
) -> Origin:
    """Sleep until ``t0_wall_s`` (if given and in the future) and return the origin."""
    clock = clock or SystemClock()
    if t0_wall_s is not None:
        remaining = t0_wall_s - clock.time()
        if remaining > max_wait_s:
            raise ValueError(
                f"start signal is {remaining:.0f} s away; "
                f"refusing to wait longer than {max_wait_s:.0f} s"
            )
        # Coarse sleep, then a short busy wait for the last milliseconds.
        if remaining > 0.01:
            clock.sleep(remaining - 0.01)
        while clock.time() < t0_wall_s:
            clock.sleep(0.0005)
    return Origin(clock.time(), clock.perf_counter_ns())


def add_start_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--start-at", type=float, default=None, help="wall-clock T0 (seconds since the epoch)"
    )
    parser.add_argument("--origin-wall", type=float, default=None, help="origin captured elsewhere")
    parser.add_argument("--origin-perf", type=int, default=None)


def origin_from_arguments(args: argparse.Namespace, clock: Clock | None = None) -> Origin:
    if args.origin_wall is not None and args.origin_perf is not None:
        return Origin(args.origin_wall, args.origin_perf)
    return wait_until(args.start_at, clock)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Wait for the start signal and print the origin.")
    parser.add_argument("--start-at", type=float, default=None)
    args = parser.parse_args(argv)
    origin = wait_until(args.start_at)
    print(
        json.dumps({"wall_time_s": origin.wall_time_s, "perf_counter_ns": origin.perf_counter_ns})
    )


if __name__ == "__main__":
    main()

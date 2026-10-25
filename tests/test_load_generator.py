from __future__ import annotations

import json
import random
import time
from pathlib import Path

import psutil
import pytest

from src.common.clock import FakeClock
from src.testbed import load_generator as lg


def test_duty_cycle_schedule_with_fake_clock() -> None:
    clock = FakeClock(start_ns=0)
    periods = 0

    def should_stop() -> bool:
        return clock.perf_counter_ns() >= 1_000_000_000  # one second

    done = lg.run_duty_cycle(30.0, 0.1, should_stop, clock, spin=lambda: clock.advance(0.001))
    periods = done
    assert periods == 10
    # Each period: 30 ms busy (spun in 1 ms steps), then a 70 ms sleep.
    assert clock.sleeps == pytest.approx([0.07] * 10, abs=1e-3)


def test_duty_cycle_extremes_and_validation() -> None:
    clock = FakeClock(start_ns=0)
    stop_at = [200_000_000]
    lg.run_duty_cycle(0.0, 0.1, lambda: clock.perf_counter_ns() >= stop_at[0], clock)
    assert clock.sleeps == pytest.approx([0.1, 0.1])
    clock2 = FakeClock(start_ns=0)
    lg.run_duty_cycle(
        100.0,
        0.1,
        lambda: clock2.perf_counter_ns() >= 200_000_000,
        clock2,
        spin=lambda: clock2.advance(0.05),
    )
    assert clock2.sleeps == []
    with pytest.raises(ValueError):
        lg.run_duty_cycle(150.0, 0.1, lambda: True)
    with pytest.raises(ValueError):
        lg.run_duty_cycle(10.0, 0, lambda: True)


def test_memory_holder_fills_and_releases_against_system_usage() -> None:
    total = 1000 * 1024  # 1000 blocks of 1 KB
    base_used = 300 * 1024
    holder: lg.MemoryHolder
    holder = lg.MemoryHolder(total, lambda: base_used + holder.allocated_bytes)
    assert holder.fill_to(50.0) == 200  # 500 blocks target - 300 already used
    assert holder.fill_to(40.0) == 100  # releases down to the lower target
    assert holder.fill_to(10.0) == 0  # cannot go below what others use
    assert holder.fill_to(99.0) == 650  # capped at 95 %
    holder.release()
    assert holder.blocks == 0


def test_spec_targets_are_seeded_and_regimes_match_chapter_3() -> None:
    light = lg.LoadSpec.regime("light")
    assert (
        light.mode == "continuous"
        and light.cpu_range == (10.0, 20.0)
        and light.memory_range == (10.0, 20.0)
    )
    heavy = lg.LoadSpec.regime("heavy")
    assert heavy.cpu_range == (60.0, 80.0) and heavy.memory_range == (50.0, 60.0)
    a = [heavy.targets(random.Random(7)) for _ in range(3)]
    b = [heavy.targets(random.Random(7)) for _ in range(3)]
    assert a == b
    assert all(60 <= t.cpu_percent <= 80 and 50 <= t.memory_percent <= 60 for t in a)
    fixed = lg.LoadSpec(cpu_percent=35.0, memory_percent=20.0)
    assert fixed.targets(random.Random(1)) == lg.LoadTarget(35.0, 20.0)
    processes = lg.LoadSpec(processes=3)
    assert processes.targets(random.Random(1)).cpu_percent == 100.0 and processes.worker_count == 3


class RecordingCpuLoad:
    def __init__(self) -> None:
        self.started: list[float] = []
        self.stopped = 0

    def start(self, target_percent: float) -> None:
        self.started.append(target_percent)

    def stop(self) -> None:
        self.stopped += 1


def test_generator_continuous_mode_resamples_each_cycle(tmp_path: Path) -> None:
    clock = FakeClock(start_time=0.0)
    cpu = RecordingCpuLoad()
    memory = lg.MemoryHolder(1000 * 1024, lambda: memory.allocated_bytes)
    spec = lg.LoadSpec.regime("moderate", cycle_s=10.0, duration_s=35.0, seed=3)
    generator = lg.LoadGenerator(
        spec, clock=clock, cpu_load=cpu, memory=memory, log_path=tmp_path / "load.json"
    )  # type: ignore[arg-type]
    cycles = generator.run()
    assert [c.cycle for c in cycles] == [0, 1, 2, 3]
    assert [c.wall_time_s for c in cycles] == [0.0, 10.0, 20.0, 30.0]
    assert len(cpu.started) == 4 and cpu.stopped >= 1
    assert memory.blocks == 0  # released at the end
    log = json.loads((tmp_path / "load.json").read_text())
    assert log["finished"] is True and len(log["cycles"]) == 4
    assert all(30 <= c["target"]["cpu_percent"] <= 50 for c in log["cycles"])


def test_generator_fixed_mode_applies_once(tmp_path: Path) -> None:
    clock = FakeClock(start_time=0.0)
    cpu = RecordingCpuLoad()
    memory = lg.MemoryHolder(1000 * 1024, lambda: memory.allocated_bytes)
    spec = lg.LoadSpec(cpu_percent=25.0, memory_percent=10.0, cycle_s=5.0, duration_s=12.0)
    cycles = lg.LoadGenerator(spec, clock=clock, cpu_load=cpu, memory=memory).run()  # type: ignore[arg-type]
    assert cpu.started == [25.0]
    assert [c.memory_blocks for c in cycles] == [100, 100, 100]


def test_cli_spec_from_args() -> None:
    args = lg.build_parser().parse_args(["--regime", "light", "--duration", "5", "--seed", "9"])
    spec = lg.spec_from_args(args)
    assert spec.cpu_range == (10.0, 20.0) and spec.duration_s == 5 and spec.seed == 9
    args = lg.build_parser().parse_args(["--cpu", "40", "--memory", "30", "--cores", "2"])
    spec = lg.spec_from_args(args)
    assert spec == lg.LoadSpec(cpu_percent=40.0, memory_percent=30.0, cores=2)


@pytest.mark.slow
def test_real_worker_reaches_the_target_utilisation() -> None:
    cpu = lg.CpuLoad(cores=1, period_s=0.05)
    cpu.start(50.0)
    try:
        time.sleep(0.3)
        proc = psutil.Process(cpu.pids[0])
        proc.cpu_percent(interval=None)
        time.sleep(1.5)
        usage = proc.cpu_percent(interval=None)
    finally:
        cpu.stop()
    assert 40.0 <= usage <= 62.0, usage
    assert cpu.pids == []

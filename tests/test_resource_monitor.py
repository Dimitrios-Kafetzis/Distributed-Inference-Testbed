from __future__ import annotations

from pathlib import Path

from src.benchmarking import benchmark_results as br
from src.common.clock import FakeClock
from src.common.runner import CommandResult, FakeRunner, ok, prefix
from src.common.sysfs import SysFs
from src.testbed.resource_monitor import ResourceMonitor


class FakeStats:
    def __init__(self) -> None:
        self.calls = 0

    def cpu_percent(self) -> float:
        self.calls += 1
        return 10.0 * self.calls

    def memory(self) -> tuple[float, float]:
        return 700.0, 3100.0


def make_tree(sysfs: SysFs) -> None:
    sysfs.write("/sys/class/thermal/thermal_zone0/temp", 52345)
    for cpu in range(4):
        sysfs.write(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq", 1_800_000)
    sysfs.write("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", 1_800_000)


def test_samples_at_the_interval_and_writes_a_trace(sysfs: SysFs, tmp_path: Path) -> None:
    make_tree(sysfs)
    runner = FakeRunner().on(prefix("vcgencmd", "get_throttled"), ok("throttled=0x50000\n"))
    clock = FakeClock(start_ns=0, start_time=100.0)
    monitor = ResourceMonitor(
        run_id="r1",
        node_id="device1",
        output=tmp_path / "m.json",
        interval_s=1.0,
        stats=FakeStats(),
        sysfs=sysfs,
        runner=runner,
        clock=clock,
        hostname="d1",
        flush_every=2,
    )
    trace = monitor.run(max_samples=3)
    assert [s.perf_counter_ns for s in trace.samples] == [0, 10**9, 2 * 10**9]
    assert [s.cpu_percent for s in trace.samples] == [10.0, 20.0, 30.0]
    assert trace.samples[0].temperature_c == 52.345
    assert trace.samples[0].frequencies_khz == [1_800_000] * 4
    assert trace.samples[0].throttled == "0x50000"
    assert trace.nominal_frequency_khz == 1_800_000
    assert trace.origin.wall_time_s == 100.0
    loaded = br.load(tmp_path / "m.json")
    assert isinstance(loaded, br.ResourceTrace) and len(loaded.samples) == 3


def test_without_vcgencmd_or_thermal_zone(sysfs: SysFs, tmp_path: Path) -> None:
    sysfs.write("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1_500_000)
    runner = FakeRunner().on(prefix("vcgencmd"), CommandResult(("vcgencmd",), 127, "", "not found"))
    monitor = ResourceMonitor(
        run_id="r1",
        node_id="d",
        output=tmp_path / "m.json",
        stats=FakeStats(),
        sysfs=sysfs,
        runner=runner,
        clock=FakeClock(),
        hostname="h",
        nominal_khz=1_800_000,
    )
    trace = monitor.run(max_samples=2)
    assert trace.samples[0].temperature_c is None and trace.samples[0].throttled is None
    assert len(runner.calls) == 1  # vcgencmd is not retried once it is known to be absent
    assert len(trace.below_nominal()) == 2


def test_vcgencmd_temperature_fallback(sysfs: SysFs, tmp_path: Path) -> None:
    runner = FakeRunner().on(prefix("vcgencmd", "measure_temp"), ok("temp=48.3'C\n"))
    runner.on(prefix("vcgencmd", "get_throttled"), ok("throttled=0x0\n"))
    monitor = ResourceMonitor(
        run_id="r",
        node_id="d",
        output=tmp_path / "m.json",
        stats=FakeStats(),
        sysfs=sysfs,
        runner=runner,
        clock=FakeClock(),
        hostname="h",
    )
    assert monitor.run(max_samples=1).samples[0].temperature_c == 48.3


def test_duration_and_stop(sysfs: SysFs, tmp_path: Path) -> None:
    clock = FakeClock(start_ns=0)
    monitor = ResourceMonitor(
        run_id="r",
        node_id="d",
        output=tmp_path / "m.json",
        stats=FakeStats(),
        sysfs=sysfs,
        runner=FakeRunner(strict=False),
        clock=clock,
        hostname="h",
        interval_s=0.5,
    )
    trace = monitor.run(duration_s=2.0)
    assert len(trace.samples) == 4

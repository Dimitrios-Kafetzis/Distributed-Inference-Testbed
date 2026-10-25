"""Tests for sysfs, clock and seeds."""

from __future__ import annotations

import random

import pytest

from src.common.clock import FakeClock, SystemClock
from src.common.seeds import seed_everything
from src.common.sysfs import SysFs


def test_sysfs_maps_absolute_paths_under_root(sysfs: SysFs) -> None:
    sysfs.write("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1800000)
    assert sysfs.exists("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    assert sysfs.read_int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") == 1800000
    assert sysfs.path("/sys/x").is_relative_to(sysfs.root)


def test_sysfs_glob_returns_device_paths(sysfs: SysFs) -> None:
    for cpu in range(4):
        sysfs.write(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq", 1800000)
    found = list(sysfs.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq"))
    assert found == [
        f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_max_freq" for cpu in range(4)
    ]


def test_sysfs_rejects_relative_paths(sysfs: SysFs) -> None:
    with pytest.raises(ValueError):
        sysfs.read("sys/relative")


def test_sysfs_mkdir_rmdir(sysfs: SysFs) -> None:
    sysfs.mkdir("/sys/fs/cgroup/testbed-run")
    assert sysfs.exists("/sys/fs/cgroup/testbed-run")
    sysfs.rmdir("/sys/fs/cgroup/testbed-run")
    assert not sysfs.exists("/sys/fs/cgroup/testbed-run")


def test_fake_clock_advances_on_sleep() -> None:
    clock = FakeClock(start_ns=0, start_time=100.0)
    clock.sleep(1.5)
    assert clock.perf_counter_ns() == 1_500_000_000
    assert clock.time() == 101.5
    assert clock.sleeps == [1.5]
    with pytest.raises(ValueError):
        clock.advance(-1)


def test_system_clock_is_monotonic() -> None:
    clock = SystemClock()
    a = clock.perf_counter_ns()
    clock.sleep(0)
    assert clock.perf_counter_ns() >= a
    assert clock.time() > 1_600_000_000


def test_seed_everything_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    seeded = seed_everything(1234)
    first = [random.random() for _ in range(3)]
    seed_everything(1234)
    assert [random.random() for _ in range(3)] == first
    assert seeded["random"] is True
    import os

    assert os.environ["PYTHONHASHSEED"] == "1234"
    with pytest.raises(ValueError):
        seed_everything(-1)

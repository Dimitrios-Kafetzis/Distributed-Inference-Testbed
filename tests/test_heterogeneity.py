from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.sysfs import SysFs
from src.testbed import heterogeneity as het


def make_cpus(sysfs: SysFs, count: int = 4, max_khz: int = 1_800_000) -> None:
    for cpu in range(count):
        base = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq"
        sysfs.write(f"{base}/cpuinfo_max_freq", max_khz)
        sysfs.write(f"{base}/scaling_max_freq", max_khz)
        sysfs.write(f"{base}/scaling_cur_freq", max_khz)
        sysfs.write(f"{base}/scaling_governor", "ondemand")


def test_frequency_cap_apply_verify_restore(sysfs: SysFs) -> None:
    make_cpus(sysfs)
    cap = het.FrequencyCap(sysfs)
    previous = cap.apply(1_200_000)
    assert set(previous.values()) == {1_800_000} and len(previous) == 4
    assert cap.verify(1_200_000)
    assert sysfs.read_int("/sys/devices/system/cpu/cpu3/cpufreq/scaling_max_freq") == 1_200_000
    cap.restore(previous)
    assert cap.verify(1_800_000)


def test_frequency_cap_rejects_above_hardware_max_and_missing_cpufreq(sysfs: SysFs) -> None:
    with pytest.raises(RuntimeError, match="cpufreq"):
        het.FrequencyCap(sysfs).apply(1_000_000)
    make_cpus(sysfs)
    with pytest.raises(ValueError, match="exceeds"):
        het.FrequencyCap(sysfs).apply(2_000_000)


def test_memory_limit_cgroup(sysfs: SysFs) -> None:
    limit = het.MemoryLimit(sysfs)
    assert limit.limit("testbed-r1") is None
    path = limit.create("testbed-r1", 2 * 1024**3)
    assert path == "/sys/fs/cgroup/testbed-r1"
    assert limit.limit("testbed-r1") == 2 * 1024**3
    assert limit.attach_command("testbed-r1", "python3 -m x") == (
        "echo $$ > /sys/fs/cgroup/testbed-r1/cgroup.procs && exec python3 -m x"
    )
    # Removing a cgroup directory requires it to be empty of files on a real
    # system; the fake tree holds memory.max, so remove that first.
    sysfs.path("/sys/fs/cgroup/testbed-r1/memory.max").unlink()
    limit.remove("testbed-r1")
    assert not sysfs.exists("/sys/fs/cgroup/testbed-r1")
    with pytest.raises(ValueError):
        limit.create("a/b", 1)
    with pytest.raises(ValueError):
        limit.create("x", 0)


def test_thread_env() -> None:
    env = het.thread_env(2)
    assert env["OMP_NUM_THREADS"] == "2" and env["TESTBED_THREADS"] == "2"
    with pytest.raises(ValueError):
        het.thread_env(0)


def test_cli_freq_and_cgroup(
    sysfs: SysFs, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_cpus(sysfs)
    state = tmp_path / "state.json"
    het.main(["--sysfs-root", str(sysfs.root), "freq", "--max-mhz", "600", "--state", str(state)])
    assert json.loads(state.read_text())["/sys/devices/system/cpu/cpu0/cpufreq"] == 1_800_000
    assert sysfs.read_int("/sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq") == 600_000
    het.main(["--sysfs-root", str(sysfs.root), "freq", "--restore", str(state)])
    assert sysfs.read_int("/sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq") == 1_800_000
    het.main(
        ["--sysfs-root", str(sysfs.root), "cgroup", "create", "--name", "g", "--memory-mb", "2048"]
    )
    assert sysfs.read("/sys/fs/cgroup/g/memory.max") == str(2048 * 1024 * 1024)
    het.main(["--sysfs-root", str(sysfs.root), "threads", "--count", "3"])
    assert "OMP_NUM_THREADS=3" in capsys.readouterr().out

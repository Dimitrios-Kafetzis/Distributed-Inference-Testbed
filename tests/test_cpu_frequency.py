from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.sysfs import SysFs
from src.testbed import cpu_frequency as cf


def make_intel(sysfs: SysFs, cores: int = 8) -> None:
    for cpu in range(cores):
        base = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq"
        sysfs.write(f"{base}/scaling_governor", "powersave")
        sysfs.write(f"{base}/scaling_max_freq", 4_900_000)
        sysfs.write(f"{base}/scaling_cur_freq", 800_000)
        sysfs.write(f"{base}/base_frequency", 3_600_000)
    sysfs.write(cf.NO_TURBO, 0)


def test_apply_pins_nominal_frequency_and_restore(sysfs: SysFs) -> None:
    make_intel(sysfs)
    policy = cf.NominalFrequencyPolicy(sysfs)
    previous = policy.apply()
    for cpu in range(8):
        base = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq"
        assert sysfs.read(f"{base}/scaling_governor") == "performance"
        assert sysfs.read_int(f"{base}/scaling_max_freq") == 3_600_000
    assert sysfs.read(cf.NO_TURBO) == "1"
    assert previous["turbo"] == {"path": cf.NO_TURBO, "value": "0"}
    policy.restore(previous)
    assert sysfs.read(cf.NO_TURBO) == "0"
    assert sysfs.read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") == "powersave"
    assert sysfs.read_int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq") == 4_900_000


def test_report_detects_nominal(sysfs: SysFs) -> None:
    make_intel(sysfs, cores=2)
    policy = cf.NominalFrequencyPolicy(sysfs)
    report = policy.report()
    assert report.turbo_disabled is False and not report.all_at_nominal()
    policy.apply()
    for cpu in range(2):
        sysfs.write(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq", 3_590_000)
    report = policy.report()
    assert report.turbo_disabled is True and report.all_at_nominal()


def test_generic_boost_switch(sysfs: SysFs) -> None:
    base = "/sys/devices/system/cpu/cpu0/cpufreq"
    sysfs.write(f"{base}/scaling_governor", "schedutil")
    sysfs.write(f"{base}/scaling_max_freq", 1_800_000)
    sysfs.write(f"{base}/scaling_cur_freq", 1_800_000)
    sysfs.write(cf.BOOST, 1)
    policy = cf.NominalFrequencyPolicy(sysfs)
    previous = policy.apply()
    assert sysfs.read(cf.BOOST) == "0" and previous["turbo"]["path"] == cf.BOOST
    assert policy.report().base_khz is None and policy.report().all_at_nominal()


def test_no_cpufreq(sysfs: SysFs) -> None:
    with pytest.raises(RuntimeError):
        cf.NominalFrequencyPolicy(sysfs).apply()


def test_cli(sysfs: SysFs, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_intel(sysfs, cores=1)
    state = tmp_path / "s.json"
    cf.main(["apply", "--state", str(state), "--sysfs-root", str(sysfs.root)])
    assert json.loads(state.read_text())["turbo"]["value"] == "0"
    with pytest.raises(SystemExit) as exc:
        cf.main(["verify", "--sysfs-root", str(sysfs.root)])  # cur freq still 800 MHz
    assert exc.value.code == 2
    sysfs.write("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 3_600_000)
    cf.main(["verify", "--sysfs-root", str(sysfs.root)])
    cf.main(["restore", "--state", str(state), "--sysfs-root", str(sysfs.root)])
    assert sysfs.read(cf.NO_TURBO) == "0"
    capsys.readouterr()

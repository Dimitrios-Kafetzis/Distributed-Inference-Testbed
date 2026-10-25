"""Device probe: prints the software and hardware facts recorded in run metadata.

Run on each node at the start of a run (``python -m src.common.probe``). Prints
one JSON object: hostname, kernel version, OS release, Python version, the
TensorFlow and PyTorch version strings (or null), CPU model, core count,
maximum frequency, total memory, NTP synchronisation status.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
from typing import Any

from src.common.runner import CommandRunner, SubprocessRunner
from src.common.sysfs import SysFs


def os_release(sysfs: SysFs) -> str | None:
    for path in ("/etc/os-release", "/usr/lib/os-release"):
        if sysfs.exists(path):
            for line in sysfs.read(path).splitlines():
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    return None


def cpu_model(sysfs: SysFs) -> str | None:
    if not sysfs.exists("/proc/cpuinfo"):
        return None
    for line in sysfs.read("/proc/cpuinfo").splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in ("model name", "model", "hardware"):
            return value.strip()
    return None


def max_frequency_khz(sysfs: SysFs) -> int | None:
    path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
    return sysfs.read_int(path) if sysfs.exists(path) else None


def total_memory_mb(sysfs: SysFs) -> float | None:
    if not sysfs.exists("/proc/meminfo"):
        return None
    for line in sysfs.read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 1024, 1)
    return None


def framework_version(name: str, runner: CommandRunner) -> str | None:
    """Import the framework in a child interpreter (TensorFlow import is slow and noisy)."""
    code = f"import {name}; print({name}.__version__)"
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    result = runner.run([sys.executable, "-c", code], timeout=120, env=env)
    return result.stdout.strip().splitlines()[-1] if result.ok and result.stdout.strip() else None


def ntp_status(runner: CommandRunner) -> dict[str, Any] | None:
    """chrony tracking: offset and whether the clock is synchronised."""
    result = runner.run(["chronyc", "tracking"], timeout=5)
    if not result.ok:
        return None
    info: dict[str, Any] = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "reference id":
            info["reference"] = value
        elif key == "leap status":
            info["synchronised"] = value.lower() == "normal"
        elif key in ("system time", "last offset"):
            try:
                number = float(value.split()[0])
                sign = -1.0 if "slow" in value else 1.0
                info[key.replace(" ", "_") + "_ms"] = sign * number * 1000.0
            except ValueError:
                pass
    return info or None


def probe(
    sysfs: SysFs | None = None, runner: CommandRunner | None = None, frameworks: bool = True
) -> dict[str, Any]:
    sysfs = sysfs or SysFs()
    runner = runner or SubprocessRunner()
    return {
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "os_release": os_release(sysfs),
        "python": platform.python_version(),
        "tensorflow": framework_version("tensorflow", runner) if frameworks else None,
        "torch": framework_version("torch", runner) if frameworks else None,
        "cpu_model": cpu_model(sysfs),
        "cores": os.cpu_count(),
        "max_frequency_khz": max_frequency_khz(sysfs),
        "total_memory_mb": total_memory_mb(sysfs),
        "ntp": ntp_status(runner),
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Print this node's software and hardware facts as JSON."
    )
    parser.add_argument(
        "--no-frameworks", action="store_true", help="skip the TensorFlow and PyTorch probes"
    )
    parser.add_argument("--sysfs-root", default="/")
    args = parser.parse_args(argv)
    print(json.dumps(probe(SysFs(args.sysfs_root), frameworks=not args.no_frameworks)))


if __name__ == "__main__":
    main()

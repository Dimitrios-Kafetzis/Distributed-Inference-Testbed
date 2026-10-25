"""Emulated device heterogeneity: CPU frequency caps, memory limits, thread counts.

Heterogeneity is emulated on identical Raspberry Pi 4 devices with Linux
``cpufreq`` (a lower maximum frequency, for example 1.8 GHz -> 1.2 GHz or
0.6 GHz), cgroups v2 (a memory limit, for example 4 GB -> 2 GB, applied to the
experiment processes) and the number of CPU threads the inference runtime may
use. Every change records the previous state so that it can be reverted at
the end of the experiment.

Usage (on a device, with sudo for the sysfs writes)::

    python -m src.testbed.heterogeneity freq --max-mhz 1200
    python -m src.testbed.heterogeneity freq --restore STATE.json
    python -m src.testbed.heterogeneity cgroup create --name testbed-run1 --memory-mb 2048
    python -m src.testbed.heterogeneity cgroup remove --name testbed-run1
    python -m src.testbed.heterogeneity threads --count 2
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from src.common.sysfs import SysFs

CPU_GLOB = "/sys/devices/system/cpu/cpu[0-9]*/cpufreq"
CGROUP_ROOT = "/sys/fs/cgroup"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "TORCH_NUM_THREADS",
    "TESTBED_THREADS",
)


class FrequencyCap:
    """Caps ``scaling_max_freq`` of every CPU; remembers the previous values."""

    def __init__(self, sysfs: SysFs | None = None) -> None:
        self.sysfs = sysfs or SysFs()

    def cpu_dirs(self) -> list[str]:
        return list(self.sysfs.glob(CPU_GLOB))

    def current(self) -> dict[str, int]:
        return {d: self.sysfs.read_int(f"{d}/scaling_max_freq") for d in self.cpu_dirs()}

    def hardware_max_khz(self) -> int | None:
        dirs = self.cpu_dirs()
        if not dirs:
            return None
        return self.sysfs.read_int(f"{dirs[0]}/cpuinfo_max_freq")

    def apply(self, max_khz: int) -> dict[str, int]:
        """Set the cap on every CPU; return the previous values for ``restore``."""
        dirs = self.cpu_dirs()
        if not dirs:
            raise RuntimeError("no cpufreq directories found; is cpufreq available?")
        hardware_max = self.hardware_max_khz()
        if hardware_max is not None and max_khz > hardware_max:
            raise ValueError(f"{max_khz} kHz exceeds the hardware maximum {hardware_max} kHz")
        previous = self.current()
        for d in dirs:
            self.sysfs.write(f"{d}/scaling_max_freq", max_khz)
        return previous

    def restore(self, previous: dict[str, int]) -> None:
        for d, khz in previous.items():
            self.sysfs.write(f"{d}/scaling_max_freq", khz)

    def verify(self, max_khz: int) -> bool:
        return all(v == max_khz for v in self.current().values())


class MemoryLimit:
    """A cgroup v2 group with ``memory.max``; experiment processes join it."""

    def __init__(self, sysfs: SysFs | None = None, root: str = CGROUP_ROOT) -> None:
        self.sysfs = sysfs or SysFs()
        self.root = root

    def path(self, name: str) -> str:
        if "/" in name or not name:
            raise ValueError(f"invalid cgroup name {name!r}")
        return f"{self.root}/{name}"

    def create(self, name: str, limit_bytes: int) -> str:
        if limit_bytes <= 0:
            raise ValueError("limit_bytes must be positive")
        path = self.path(name)
        self.sysfs.mkdir(path)
        self.sysfs.write(f"{path}/memory.max", limit_bytes)
        return path

    def limit(self, name: str) -> int | None:
        path = self.path(name)
        if not self.sysfs.exists(f"{path}/memory.max"):
            return None
        value = self.sysfs.read(f"{path}/memory.max")
        return None if value == "max" else int(value)

    def remove(self, name: str) -> None:
        path = self.path(name)
        if self.sysfs.exists(path):
            self.sysfs.rmdir(path)

    def attach_command(self, name: str, command: str) -> str:
        """Shell command that moves itself into the group and then runs ``command``."""
        procs = shlex.quote(f"{self.path(name)}/cgroup.procs")
        return f"echo $$ > {procs} && exec {command}"


def thread_env(count: int) -> dict[str, str]:
    if count < 1:
        raise ValueError("thread count must be at least 1")
    return dict.fromkeys(THREAD_ENV_VARS, str(count))


def apply_thread_count(count: int) -> dict[str, bool]:
    """Set the thread count of the runtimes present in this interpreter (device side)."""
    applied = {}
    try:
        import torch

        torch.set_num_threads(count)
        applied["torch"] = True
    except ImportError:
        applied["torch"] = False
    try:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(count)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        applied["tensorflow"] = True
    except (ImportError, RuntimeError):
        applied["tensorflow"] = False
    return applied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emulate device heterogeneity.")
    parser.add_argument("--sysfs-root", default="/", help="root of /sys (tests use a fake tree)")
    sub = parser.add_subparsers(dest="command", required=True)

    freq = sub.add_parser("freq", help="cap the CPU frequency")
    group = freq.add_mutually_exclusive_group(required=True)
    group.add_argument("--max-mhz", type=float)
    group.add_argument("--restore", type=Path, metavar="STATE_JSON")
    group.add_argument("--show", action="store_true")
    freq.add_argument("--state", type=Path, help="where to save the previous values")

    cg = sub.add_parser("cgroup", help="memory limit through cgroups v2")
    cg.add_argument("action", choices=["create", "remove", "show"])
    cg.add_argument("--name", required=True)
    cg.add_argument("--memory-mb", type=float)
    cg.add_argument("--cgroup-root", default=CGROUP_ROOT)

    threads = sub.add_parser("threads", help="print the environment for a thread count")
    threads.add_argument("--count", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sysfs = SysFs(args.sysfs_root)
    if args.command == "freq":
        cap = FrequencyCap(sysfs)
        if args.show:
            print(json.dumps(cap.current(), indent=2))
        elif args.restore is not None:
            cap.restore(json.loads(args.restore.read_text(encoding="utf-8")))
            print("restored", file=sys.stderr)
        else:
            previous = cap.apply(int(args.max_mhz * 1000))
            if args.state:
                args.state.write_text(json.dumps(previous, indent=2), encoding="utf-8")
            print(json.dumps(previous))
    elif args.command == "cgroup":
        limit = MemoryLimit(sysfs, args.cgroup_root)
        if args.action == "create":
            if args.memory_mb is None:
                raise SystemExit("--memory-mb is required for create")
            print(limit.create(args.name, int(args.memory_mb * 1024 * 1024)))
        elif args.action == "remove":
            limit.remove(args.name)
        else:
            print(json.dumps({"memory.max": limit.limit(args.name)}))
    else:
        for key, value in thread_env(args.count).items():
            print(f"{key}={value}")


if __name__ == "__main__":
    main()

"""Edge-server CPU frequency policy: nominal clock throughout each experiment.

To make timing measurements reproducible, the Linux kernel is configured to
keep each CPU core of the edge server at its nominal clock frequency,
disabling the dynamic frequency scaling with which the kernel lowers the
frequency when the CPU is idle. "Nominal" is the base frequency: the
``performance`` governor is selected, turbo is disabled (``intel_pstate``
``no_turbo``, or the generic ``cpufreq/boost`` switch) and the maximum
scaling frequency is pinned to the base frequency when the driver exposes it.
The previous settings are saved so that they can be restored at the end.

Usage (edge server, with sudo)::

    python -m src.testbed.cpu_frequency apply --state /tmp/freq_state.json
    python -m src.testbed.cpu_frequency verify
    python -m src.testbed.cpu_frequency restore --state /tmp/freq_state.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.common.sysfs import SysFs

CPU_GLOB = "/sys/devices/system/cpu/cpu[0-9]*/cpufreq"
NO_TURBO = "/sys/devices/system/cpu/intel_pstate/no_turbo"
BOOST = "/sys/devices/system/cpu/cpufreq/boost"


@dataclass
class FrequencyReport:
    governors: dict[str, str]
    current_khz: dict[str, int]
    max_khz: dict[str, int]
    base_khz: int | None
    turbo_disabled: bool | None  # None when the platform has no turbo switch

    def all_at_nominal(self, tolerance: float = 0.02) -> bool:
        """Every core within ``tolerance`` of the nominal (base, else max) frequency."""
        target = self.base_khz or (max(self.max_khz.values()) if self.max_khz else None)
        if target is None:
            return False
        return all(abs(v - target) <= tolerance * target for v in self.current_khz.values())


class NominalFrequencyPolicy:
    def __init__(self, sysfs: SysFs | None = None) -> None:
        self.sysfs = sysfs or SysFs()

    def cpu_dirs(self) -> list[str]:
        return list(self.sysfs.glob(CPU_GLOB))

    def base_khz(self) -> int | None:
        dirs = self.cpu_dirs()
        if dirs and self.sysfs.exists(f"{dirs[0]}/base_frequency"):
            return self.sysfs.read_int(f"{dirs[0]}/base_frequency")
        return None

    def apply(self) -> dict[str, Any]:
        """Governor ``performance``, turbo off, max = base; return the previous state."""
        dirs = self.cpu_dirs()
        if not dirs:
            raise RuntimeError("no cpufreq directories found")
        previous: dict[str, Any] = {"governors": {}, "max_khz": {}, "turbo": None}
        for d in dirs:
            previous["governors"][d] = self.sysfs.read(f"{d}/scaling_governor")
            previous["max_khz"][d] = self.sysfs.read_int(f"{d}/scaling_max_freq")
            self.sysfs.write(f"{d}/scaling_governor", "performance")
        if self.sysfs.exists(NO_TURBO):
            previous["turbo"] = {"path": NO_TURBO, "value": self.sysfs.read(NO_TURBO)}
            self.sysfs.write(NO_TURBO, 1)
        elif self.sysfs.exists(BOOST):
            previous["turbo"] = {"path": BOOST, "value": self.sysfs.read(BOOST)}
            self.sysfs.write(BOOST, 0)
        base = self.base_khz()
        if base is not None:
            for d in dirs:
                self.sysfs.write(f"{d}/scaling_max_freq", base)
        return previous

    def restore(self, previous: dict[str, Any]) -> None:
        for d, governor in previous.get("governors", {}).items():
            self.sysfs.write(f"{d}/scaling_governor", governor)
        for d, khz in previous.get("max_khz", {}).items():
            self.sysfs.write(f"{d}/scaling_max_freq", khz)
        turbo = previous.get("turbo")
        if turbo:
            self.sysfs.write(turbo["path"], turbo["value"])

    def report(self) -> FrequencyReport:
        dirs = self.cpu_dirs()
        turbo: bool | None = None
        if self.sysfs.exists(NO_TURBO):
            turbo = self.sysfs.read(NO_TURBO) == "1"
        elif self.sysfs.exists(BOOST):
            turbo = self.sysfs.read(BOOST) == "0"
        return FrequencyReport(
            governors={d: self.sysfs.read(f"{d}/scaling_governor") for d in dirs},
            current_khz={d: self.sysfs.read_int(f"{d}/scaling_cur_freq") for d in dirs},
            max_khz={d: self.sysfs.read_int(f"{d}/scaling_max_freq") for d in dirs},
            base_khz=self.base_khz(),
            turbo_disabled=turbo,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pin the edge-server CPUs to their nominal frequency."
    )
    parser.add_argument("action", choices=["apply", "restore", "verify"])
    parser.add_argument("--state", type=Path, help="JSON file with the previous settings")
    parser.add_argument("--sysfs-root", default="/")
    args = parser.parse_args(argv)
    policy = NominalFrequencyPolicy(SysFs(args.sysfs_root))
    if args.action == "apply":
        previous = policy.apply()
        if args.state:
            args.state.write_text(json.dumps(previous, indent=2), encoding="utf-8")
        print(json.dumps(asdict(policy.report()), indent=2))
    elif args.action == "restore":
        if not args.state:
            raise SystemExit("--state is required to restore")
        policy.restore(json.loads(args.state.read_text(encoding="utf-8")))
        print("restored", file=sys.stderr)
    else:
        report = policy.report()
        print(json.dumps(asdict(report), indent=2))
        if not report.all_at_nominal():
            raise SystemExit(2)


if __name__ == "__main__":
    main()

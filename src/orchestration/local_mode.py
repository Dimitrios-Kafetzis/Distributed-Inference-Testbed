"""Local mode: the whole testbed simulated on one machine.

Every node becomes a working directory under a root folder; device-side
commands run as local subprocesses in that directory (``LocalTransport``),
privileged tools run in dry-run mode or against a fake sysfs tree, and the
inference service listens on loopback. This is how the pipeline runs in
continuous integration and how NNPM can be demonstrated without hardware.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.common.nodes import EDGE_SERVER_ROLE, END_DEVICE_ROLE, Node, Topology, save_topology
from src.common.sysfs import SysFs
from src.config import Config, load_config

LOOPBACK = "127.0.0.1"


def fake_sysfs(root: Path, cores: int = 4, max_khz: int = 1_800_000) -> SysFs:
    sysfs = SysFs(root)
    for cpu in range(cores):
        base = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq"
        sysfs.write(f"{base}/cpuinfo_max_freq", max_khz)
        sysfs.write(f"{base}/scaling_max_freq", max_khz)
        sysfs.write(f"{base}/scaling_cur_freq", max_khz)
        sysfs.write(f"{base}/scaling_governor", "ondemand")
    sysfs.write("/sys/class/thermal/thermal_zone0/temp", 48500)
    sysfs.write("/etc/os-release", 'PRETTY_NAME="local mode"\n')
    sysfs.mkdir("/sys/fs/cgroup")
    return sysfs


def create_local_testbed(
    root: Path, devices: int = 2, write_files: bool = True
) -> tuple[Config, Topology]:
    """A configuration and topology for ``devices`` local nodes plus the edge server."""
    root = Path(root)
    nodes = [
        Node(
            id="edge-server",
            role=EDGE_SERVER_ROLE,
            ip=LOOPBACK,
            lat=37.9838,
            lon=23.7275,
            transport="local",
            workdir=str(root / "nodes" / "edge-server"),
        )
    ]
    for i in range(1, devices + 1):
        workdir = root / "nodes" / f"device{i}"
        workdir.mkdir(parents=True, exist_ok=True)
        fake_sysfs(workdir / "sysfs")
        nodes.append(
            Node(
                id=f"device{i}",
                role=END_DEVICE_ROLE,
                ip=LOOPBACK,
                lat=37.98 + 0.002 * i,
                lon=23.73 + 0.003 * i,
                transport="local",
                workdir=str(workdir),
            )
        )
    (root / "nodes" / "edge-server").mkdir(parents=True, exist_ok=True)
    topology = Topology(nodes, comment="local mode (synthetic)")
    # TESTBED_* variables of the process still apply; the local-mode paths override them.
    environ = {
        **os.environ,
        "TESTBED_BS_ROOT": str(root),
        "TESTBED_BS_RESULTS_DIR": str(root / "results"),
        "TESTBED_BS_MODELS_DIR": str(root / "models"),
        "TESTBED_ED_ROOT": ".",
        "TESTBED_NNPM_HOST": LOOPBACK,
    }
    config = load_config(
        root / "config.yaml" if (root / "config.yaml").exists() else None, environ=environ
    )
    if write_files:
        save_topology(topology, root / "nodes.json")
    return config, topology


def is_local(node: Node) -> bool:
    return node.transport == "local"


def python_for(node: Node, config: Config) -> str:
    return sys.executable if is_local(node) else f"{config.ed_venv_dir}/bin/python"


def code_dir_for(node: Node, config: Config, repo_root: Path) -> str:
    return str(repo_root) if is_local(node) else config.ed_code_dir


def sysfs_root_for(node: Node) -> str:
    return f"{node.workdir}/sysfs" if is_local(node) and node.workdir else "/"

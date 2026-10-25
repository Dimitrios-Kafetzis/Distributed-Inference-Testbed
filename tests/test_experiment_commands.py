"""Shape of the privileged device-side commands built by the orchestrator."""

from __future__ import annotations

from pathlib import Path

from src.common.nodes import Node
from src.orchestration.experiment import DevicePaths, ExperimentContext


def make_paths() -> DevicePaths:
    return DevicePaths(
        code_dir="/opt/testbed/code",
        python="/opt/testbed/venv/bin/python",
        models_dir="/opt/testbed/models",
        data_dir="/opt/testbed/data",
        results_dir="/opt/testbed/results/r1",
        run_dir="/opt/testbed/run/r1",
        sysfs_root="/",
    )


def test_sudo_goes_in_front_of_the_interpreter_not_cd(tmp_path: Path) -> None:
    """``sudo -n cd ...`` cannot work (cd is a shell builtin); the sudoers rule installed by
    provisioning allows exactly ``<venv>/bin/python -m src.testbed.heterogeneity``."""
    node = Node("device1", "End Device", "192.0.2.11", 0.0, 0.0)
    command = make_paths().module("src.testbed.heterogeneity", "freq", "--max-mhz", "1200")
    privileged = ExperimentContext.sudo(None, node, command)  # type: ignore[arg-type]
    assert privileged == (
        "cd /opt/testbed/code && sudo -n /opt/testbed/venv/bin/python -m "
        "src.testbed.heterogeneity freq --max-mhz 1200"
    )
    assert ExperimentContext.sudo(None, node, "tc qdisc show") == "sudo -n tc qdisc show"  # type: ignore[arg-type]
    local = Node(
        "device1", "End Device", "127.0.0.1", 0.0, 0.0, transport="local", workdir=str(tmp_path)
    )
    assert ExperimentContext.sudo(None, local, command) == command  # type: ignore[arg-type]

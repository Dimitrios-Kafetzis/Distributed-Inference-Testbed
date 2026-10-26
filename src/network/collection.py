"""Result collection: device result files into the run's results directory.

Two modes:

* ``push``: ``transfer_files.sh`` runs on each device and copies its result
  JSON files to the edge server with scp, prepending the device hostname to
  each filename;
* ``pull`` (the mode of the experiment pipeline, as Appendix B describes): the
  edge server fetches the files over SSH through the transport (SFTP, or a
  file copy in local mode) and prefixes them with the hostname itself.

Both leave ``<hostname>_<file>`` in the target directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.common.nodes import Node
from src.config import Config
from src.network.ssh_executor import Executor
from src.network.transport import LocalTransport


@dataclass(frozen=True)
class CollectedFile:
    node_id: str
    remote_path: str
    local_path: Path


def push_command(
    config: Config, edge_server_ip: str, target_dir: Path, remote_files: Sequence[str]
) -> str:
    import shlex

    script = f"{config.ed_scripts_path}/transfer_files.sh"
    return shlex.join([script, config.ssh_username, edge_server_ip, str(target_dir), *remote_files])


def collect(
    executor: Executor,
    nodes: Sequence[Node],
    remote_files: Sequence[str],
    target_dir: Path,
    *,
    config: Config | None = None,
    edge_server_ip: str | None = None,
    mode: str = "pull",
    hostnames: dict[str, str] | None = None,
) -> list[CollectedFile]:
    """Collect ``remote_files`` (paths on the devices) from every node into ``target_dir``."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(executor.transport, LocalTransport):
        mode = "pull"
    collected: list[CollectedFile] = []
    if mode == "push":
        if config is None or edge_server_ip is None:
            raise ValueError("push mode needs the configuration and the edge-server address")
        command = push_command(config, edge_server_ip, target_dir, remote_files)
        executor.run_on_nodes(nodes, command)
        for node in nodes:
            prefix = (hostnames or {}).get(node.id, node.id)
            for remote in remote_files:
                collected.append(
                    CollectedFile(node.id, remote, target_dir / f"{prefix}_{Path(remote).name}")
                )
        return collected
    for node in nodes:
        prefix = (hostnames or {}).get(node.id, node.id)
        for remote in remote_files:
            local = target_dir / f"{prefix}_{Path(remote).name}"
            executor.fetch(node, remote, local)
            collected.append(CollectedFile(node.id, remote, local))
    return collected

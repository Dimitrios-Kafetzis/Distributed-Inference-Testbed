"""Node topology: nodes.json at the repository root.

Each entry records a node identifier, its addresses, its role (``Base Station``,
the name of the edge server in the software, or ``End Device``), the map
coordinates and the statistics collected during the last benchmarking run.
The NNPM web interface rewrites the file whenever nodes are added, removed or
reconfigured; every other component reads it, so it is the single source of
truth for the topology.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from src.common.runs import write_json_atomic
from src.config import REPO_ROOT

SCHEMA_PATH = REPO_ROOT / "docs" / "schemas" / "nodes.schema.json"
DEFAULT_NODES_PATH = REPO_ROOT / "nodes.json"
# Like TESTBED_CONFIG for config.yaml: another topology file for tools and scripts.
NODES_PATH_ENV = "TESTBED_NODES"


def default_nodes_path() -> Path:
    override = os.environ.get(NODES_PATH_ENV)
    return Path(override) if override else DEFAULT_NODES_PATH


EDGE_SERVER_ROLE = "Base Station"
END_DEVICE_ROLE = "End Device"


class NodesError(ValueError):
    """Raised when nodes.json is invalid."""


@dataclass
class NodeStatistics:
    average_latency_ms: float | None
    available_memory_mb: float
    cpu_usage_percent: float
    measured_at: str
    run_id: str


@dataclass
class Node:
    id: str
    role: str
    ip: str
    lat: float
    lon: float
    management_ip: str | None = None
    transport: str = "ssh"
    workdir: str | None = None
    statistics: NodeStatistics | None = None

    @property
    def is_edge_server(self) -> bool:
        return self.role == EDGE_SERVER_ROLE

    @property
    def data_address(self) -> str:
        """Address on the wireless data plane (ping, shaping, inference traffic)."""
        return self.ip

    @property
    def management_address(self) -> str:
        """Address used for SSH and result collection."""
        return self.management_ip or self.ip

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class Topology:
    nodes: list[Node] = field(default_factory=list)
    comment: str | None = None

    def __post_init__(self) -> None:
        ids = [n.id for n in self.nodes]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise NodesError(f"duplicate node ids: {sorted(duplicates)}")
        servers = [n for n in self.nodes if n.is_edge_server]
        if len(servers) > 1:
            raise NodesError("nodes.json may hold at most one 'Base Station' entry")

    def get(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def edge_server(self) -> Node | None:
        return next((n for n in self.nodes if n.is_edge_server), None)

    def end_devices(self) -> list[Node]:
        return [n for n in self.nodes if not n.is_edge_server]

    def select(self, node_ids: Iterable[str] | None) -> list[Node]:
        """End devices by id; all of them when ``node_ids`` is None."""
        if node_ids is None:
            return self.end_devices()
        return [self.get(i) for i in node_ids]

    def upsert(self, node: Node) -> None:
        for index, existing in enumerate(self.nodes):
            if existing.id == node.id:
                self.nodes[index] = node
                break
        else:
            self.nodes.append(node)
        Topology(self.nodes)  # re-run the invariants

    def remove(self, node_id: str) -> None:
        self.nodes = [n for n in self.nodes if n.id != node_id]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"schema_version": 1}
        if self.comment:
            data["_comment"] = self.comment
        data["nodes"] = [n.to_dict() for n in self.nodes]
        return data


def load_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        schema: dict[str, Any] = json.load(fh)
    return schema


def _from_legacy(data: dict[str, Any]) -> dict[str, Any]:
    """Convert the old repository's format (a mapping keyed by id with a ``type`` field)."""
    nodes = []
    for node_id, info in data.items():
        stats = info.get("additional_info")
        nodes.append(
            {
                "id": re.sub(r"[^A-Za-z0-9_.-]+", "-", str(info.get("id", node_id))),
                "role": info.get("role", info.get("type")),
                "ip": info["ip"],
                "lat": info["lat"],
                "lon": info["lon"],
                "management_ip": None,
                "statistics": None
                if not stats
                else {
                    "average_latency_ms": stats.get("average_latency"),
                    "available_memory_mb": stats.get("average_available_memory_MB", 0.0),
                    "cpu_usage_percent": stats.get("average_cpu_usage_percent", 0.0),
                    "measured_at": "1970-01-01T00:00:00+00:00",
                    "run_id": "legacy",
                },
            }
        )
    return {"schema_version": 1, "nodes": nodes}


def _validate(data: dict[str, Any]) -> None:
    try:
        jsonschema.validate(data, load_schema())
    except jsonschema.ValidationError as exc:
        location = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise NodesError(f"invalid nodes.json at {location}: {exc.message}") from exc


def parse_topology(data: dict[str, Any]) -> Topology:
    if "nodes" not in data and data and all(isinstance(v, dict) for v in data.values()):
        data = _from_legacy(data)
    _validate(data)
    nodes = []
    for raw in data["nodes"]:
        stats = raw.get("statistics")
        nodes.append(
            Node(
                id=raw["id"],
                role=raw["role"],
                ip=raw["ip"],
                lat=raw["lat"],
                lon=raw["lon"],
                management_ip=raw.get("management_ip"),
                transport=raw.get("transport", "ssh"),
                workdir=raw.get("workdir"),
                statistics=NodeStatistics(**stats) if stats else None,
            )
        )
    return Topology(nodes=nodes, comment=data.get("_comment"))


def load_topology(path: str | Path | None = None) -> Topology:
    nodes_path = Path(path) if path is not None else default_nodes_path()
    if not nodes_path.exists():
        raise NodesError(f"topology file not found: {nodes_path} (copy nodes.example.json)")
    with nodes_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise NodesError(f"{nodes_path}: expected a JSON object")
    return parse_topology(data)


def save_topology(topology: Topology, path: str | Path | None = None) -> None:
    """Validate and write atomically."""
    nodes_path = Path(path) if path is not None else default_nodes_path()
    data = topology.to_dict()
    _validate(data)
    write_json_atomic(nodes_path, data)

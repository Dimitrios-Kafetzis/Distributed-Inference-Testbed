from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.nodes import (
    EDGE_SERVER_ROLE,
    END_DEVICE_ROLE,
    Node,
    NodesError,
    Topology,
    load_topology,
    parse_topology,
    save_topology,
)


def test_example_topology_loads(repo_root: Path) -> None:
    topology = load_topology(repo_root / "nodes.example.json")
    server = topology.edge_server()
    assert server is not None and server.role == EDGE_SERVER_ROLE
    assert [n.id for n in topology.end_devices()] == [f"device{i}" for i in range(1, 6)]
    device1 = topology.get("device1")
    assert device1.management_address == "10.0.0.11"
    assert device1.data_address == "192.168.0.11"
    assert device1.statistics is not None and device1.statistics.average_latency_ms == 5.1


def test_management_address_falls_back_to_data_address() -> None:
    node = Node(id="d", role=END_DEVICE_ROLE, ip="192.168.0.5", lat=0, lon=0)
    assert node.management_address == "192.168.0.5"


def test_duplicate_ids_and_two_servers_rejected() -> None:
    a = Node(id="a", role=END_DEVICE_ROLE, ip="1.1.1.1", lat=0, lon=0)
    with pytest.raises(NodesError, match="duplicate"):
        Topology([a, a])
    s1 = Node(id="s1", role=EDGE_SERVER_ROLE, ip="1.1.1.2", lat=0, lon=0)
    s2 = Node(id="s2", role=EDGE_SERVER_ROLE, ip="1.1.1.3", lat=0, lon=0)
    with pytest.raises(NodesError, match="at most one"):
        Topology([s1, s2])


def test_schema_rejects_bad_role_and_ip() -> None:
    bad = {
        "schema_version": 1,
        "nodes": [{"id": "x", "role": "Controller", "ip": "1.2.3.4", "lat": 0, "lon": 0}],
    }
    with pytest.raises(NodesError, match="role"):
        parse_topology(bad)
    bad_ip = {
        "schema_version": 1,
        "nodes": [{"id": "x", "role": END_DEVICE_ROLE, "ip": "999.1.1.1", "lat": 0, "lon": 0}],
    }
    with pytest.raises(NodesError, match="ip"):
        parse_topology(bad_ip)


def test_save_is_atomic_and_round_trips(tmp_path: Path, repo_root: Path) -> None:
    topology = load_topology(repo_root / "nodes.example.json")
    target = tmp_path / "nodes.json"
    save_topology(topology, target)
    assert [p.name for p in tmp_path.iterdir()] == ["nodes.json"]
    again = load_topology(target)
    assert again.to_dict() == topology.to_dict()


def test_upsert_remove_select() -> None:
    topology = Topology([Node(id="a", role=END_DEVICE_ROLE, ip="1.1.1.1", lat=0, lon=0)])
    topology.upsert(Node(id="b", role=END_DEVICE_ROLE, ip="1.1.1.2", lat=0, lon=0))
    topology.upsert(Node(id="a", role=END_DEVICE_ROLE, ip="1.1.1.9", lat=0, lon=0))
    assert topology.get("a").ip == "1.1.1.9"
    assert [n.id for n in topology.select(["b"])] == ["b"]
    assert [n.id for n in topology.select(None)] == ["a", "b"]
    topology.remove("a")
    assert [n.id for n in topology.nodes] == ["b"]
    with pytest.raises(KeyError):
        topology.get("a")


def test_legacy_format_is_converted(tmp_path: Path) -> None:
    legacy = {
        "Base Station": {
            "id": "Base Station",
            "ip": "192.168.0.153",
            "lat": 37.9,
            "lon": 23.7,
            "type": "Base Station",
            "additional_info": None,
        },
        "node1": {
            "id": "node1",
            "ip": "192.168.0.191",
            "lat": 37.9,
            "lon": 23.7,
            "type": "End Device",
            "additional_info": {
                "average_latency": 24.26,
                "average_total_memory_MB": 3744.4,
                "average_available_memory_MB": 3455.3,
                "average_available_memory_percent": 92.3,
                "average_cpu_usage_percent": 0.0,
            },
        },
    }
    path = tmp_path / "nodes.json"
    path.write_text(json.dumps(legacy))
    topology = load_topology(path)
    assert topology.edge_server() is not None
    node1 = topology.get("node1")
    assert node1.role == END_DEVICE_ROLE
    assert node1.statistics is not None and node1.statistics.average_latency_ms == 24.26


def test_missing_file_message(tmp_path: Path) -> None:
    with pytest.raises(NodesError, match=r"nodes\.example\.json"):
        load_topology(tmp_path / "nodes.json")

"""Topology: nodes, positions, links (nodes.json is the single source of truth)."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, request

from gui.blueprints.common import error, state
from src.common.nodes import EDGE_SERVER_ROLE, END_DEVICE_ROLE, Node, NodesError, NodeStatistics
from src.orchestration.geo import calculate_distance, calculate_latency

bp = Blueprint("topology", __name__, url_prefix="/api")


@bp.get("/nodes")
def list_nodes() -> dict[str, Any]:
    return {"nodes": [n.to_dict() for n in state().topology.nodes]}


@bp.get("/nodes/<node_id>")
def node_info(node_id: str) -> Any:
    try:
        node = state().topology.get(node_id)
    except KeyError:
        return error(f"unknown node {node_id}", 404)
    info = node.to_dict()
    info["status"] = (
        state().prober.statuses.get(node_id).__dict__
        if node_id in state().prober.statuses
        else None
    )
    return {"node": info}


def _node_from_payload(payload: dict[str, Any], existing: Node | None = None) -> Node:
    role = payload.get("role", existing.role if existing else END_DEVICE_ROLE)
    if role not in (EDGE_SERVER_ROLE, END_DEVICE_ROLE):
        raise NodesError(f"role must be '{EDGE_SERVER_ROLE}' or '{END_DEVICE_ROLE}'")
    stats = existing.statistics if existing else None
    if payload.get("statistics"):
        stats = NodeStatistics(**payload["statistics"])
    return Node(
        id=str(payload.get("id", existing.id if existing else "")).strip(),
        role=role,
        ip=str(payload.get("ip", existing.ip if existing else "")).strip(),
        lat=float(payload.get("lat", existing.lat if existing else 0.0)),
        lon=float(payload.get("lon", existing.lon if existing else 0.0)),
        management_ip=(payload.get("management_ip") or None)
        if "management_ip" in payload
        else (existing.management_ip if existing else None),
        transport=payload.get("transport", existing.transport if existing else "ssh"),
        workdir=payload.get("workdir", existing.workdir if existing else None),
        statistics=stats,
    )


@bp.post("/nodes")
def add_node() -> Any:
    payload = request.get_json(silent=True) or {}
    try:
        node = _node_from_payload(payload)
        if not node.id or not node.ip:
            return error("id and ip are required")
        topology = state().topology
        if any(n.id == node.id for n in topology.nodes):
            return error(f"node {node.id} exists", 409)
        topology.upsert(node)
        state().save_topology()
    except (NodesError, ValueError) as exc:
        state().reload_topology()
        return error(str(exc))
    return {"node": node.to_dict()}, 201


@bp.put("/nodes/<node_id>")
def update_node(node_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    topology = state().topology
    try:
        existing = topology.get(node_id)
    except KeyError:
        return error(f"unknown node {node_id}", 404)
    try:
        node = _node_from_payload(payload, existing)
        if node.id != node_id:
            topology.remove(node_id)
        topology.upsert(node)
        state().save_topology()
    except (NodesError, ValueError) as exc:
        state().reload_topology()
        return error(str(exc))
    return {"node": node.to_dict()}


@bp.delete("/nodes/<node_id>")
def delete_node(node_id: str) -> Any:
    topology = state().topology
    try:
        topology.get(node_id)
    except KeyError:
        return error(f"unknown node {node_id}", 404)
    topology.remove(node_id)
    state().save_topology()
    return {"deleted": node_id}


@bp.get("/links")
def links() -> dict[str, Any]:
    """Every pair of markers: great-circle distance and the synthetic latency."""
    nodes = state().topology.nodes
    result = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            distance = calculate_distance(a.lat, a.lon, b.lat, b.lon)
            result.append(
                {
                    "from": a.id,
                    "to": b.id,
                    "distance_m": round(distance, 1),
                    "latency_ms": round(calculate_latency(distance), 3),
                    "from_coords": [a.lat, a.lon],
                    "to_coords": [b.lat, b.lon],
                }
            )
    return {"links": result}


@bp.post("/latency-from-map")
def latency_from_map() -> dict[str, Any]:
    """Per-device synthetic latency to the edge server (and to every other device)."""
    topology = state().topology
    edge = topology.edge_server()
    out: dict[str, Any] = {}
    for device in topology.end_devices():
        entry: dict[str, Any] = {}
        if edge is not None:
            entry["edge_server"] = round(
                calculate_latency(calculate_distance(edge.lat, edge.lon, device.lat, device.lon)), 3
            )
        for other in topology.end_devices():
            if other.id != device.id:
                entry[other.id] = round(
                    calculate_latency(
                        calculate_distance(other.lat, other.lon, device.lat, device.lon)
                    ),
                    3,
                )
        out[device.id] = entry
    return {"latency_ms": out}

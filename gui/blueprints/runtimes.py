"""Runtimes: readiness checks of the devices and maintenance actions (tasks) per runtime."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from flask import Blueprint, Response, request

from gui.blueprints.common import error, sse, state
from src.orchestration import runtimes

bp = Blueprint("runtimes", __name__)


@bp.get("/api/runtimes")
def list_runtimes() -> dict[str, Any]:
    return {"runtimes": runtimes.describe(), "default": runtimes.DEFAULT_RUNTIME}


@bp.get("/api/runtimes/<name>/status")
def runtime_status(name: str) -> Any:
    """Every readiness check of every node, run concurrently through the live transport.

    Query parameters are passed to the checks (for example ``model`` and ``profile``)."""
    try:
        runtime = runtimes.get_runtime(name)
    except KeyError:
        return error("unknown runtime", 404)
    st = state()
    if st.runs.busy:
        return error("an experiment is running; checks would disturb it", 409)
    params = {k: v for k, v in request.args.items()}
    nodes = st.topology.nodes
    session = st.tasks.session(log=lambda line: None)

    def work(node: Any) -> tuple[str, list[dict[str, str]]]:
        try:
            return node.id, [c.to_dict() for c in runtime.check_node(session, node, params)]
        except Exception as exc:  # unreachable node, authentication failure
            return node.id, [{"name": "reachability", "status": "fail", "detail": str(exc)}]

    with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
        results = dict(pool.map(work, nodes))
    return {"runtime": name, "nodes": results}


@bp.post("/api/runtimes/<name>/actions/<action>")
def start_action(name: str, action: str) -> Any:
    try:
        runtime = runtimes.get_runtime(name)
    except KeyError:
        return error("unknown runtime", 404)
    if action not in {a.name for a in runtime.actions()}:
        return error(f"runtime {name} has no action {action}", 404)
    params = request.get_json(silent=True) or {}
    try:
        record = state().tasks.start(runtime, action, params)
    except RuntimeError as exc:
        return error(str(exc), 409)
    return {"task_id": record.task_id, "status": record.status}, 202


@bp.get("/api/tasks")
def list_tasks() -> dict[str, Any]:
    tasks = state().tasks
    return {
        "tasks": [t.to_dict() for t in tasks.history.values()],
        "current": tasks.current.task_id if tasks.current else None,
        "busy": tasks.busy,
    }


@bp.get("/api/tasks/<task_id>")
def task_detail(task_id: str) -> Any:
    record = state().tasks.history.get(task_id)
    if record is None:
        return error("unknown task", 404)
    return {**record.to_dict(), "log": list(record.lines)}


@bp.post("/api/tasks/<task_id>/stop")
def stop_task(task_id: str) -> Any:
    if not state().tasks.stop(task_id):
        return error("no such running task", 404)
    return {"task_id": task_id, "stopping": True}


@bp.get("/events/tasks/<task_id>")
def task_events(task_id: str) -> Response:
    tasks = state().tasks
    listener = tasks.subscribe(task_id)
    record = tasks.history.get(task_id)
    if record is not None:
        for line in list(record.lines):
            listener.put(line)
        if record.status != "running":
            listener.put(f"[task][done] {record.status}")
    limit = request.args.get("limit", type=int)
    return sse(listener, event="log", limit=limit)

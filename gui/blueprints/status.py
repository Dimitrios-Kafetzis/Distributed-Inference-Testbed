"""Reachability status of the nodes: latest snapshot and a server-sent event stream."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, Response, request

from gui.blueprints.common import sse, state

bp = Blueprint("status", __name__)


@bp.get("/api/status")
def status() -> dict[str, Any]:
    return state().prober.snapshot()


@bp.post("/api/status/probe")
def probe_now() -> dict[str, Any]:
    state().prober.probe_round()
    return state().prober.snapshot()


@bp.get("/events/status")
def status_events() -> Response:
    prober = state().prober
    listener = prober.subscribe()
    listener.put(prober.snapshot())
    limit = request.args.get("limit", type=int)
    return sse(listener, event="status", limit=limit)

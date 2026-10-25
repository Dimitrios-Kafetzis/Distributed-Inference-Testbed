"""Helpers shared by the blueprints."""

from __future__ import annotations

import json
import queue
from collections.abc import Iterator
from typing import Any

from flask import Response, current_app

from gui.state import AppState


def state() -> AppState:
    app_state: AppState = current_app.extensions["nnpm"]
    return app_state


def error(message: str, code: int = 400) -> tuple[dict[str, str], int]:
    return {"error": message}, code


def sse(
    source: queue.Queue[Any],
    *,
    event: str = "message",
    keepalive_s: float = 15.0,
    limit: int | None = None,
) -> Response:
    """Server-sent events from a queue; a comment line keeps the connection alive."""

    def stream() -> Iterator[str]:
        sent = 0
        while limit is None or sent < limit:
            try:
                item = source.get(timeout=keepalive_s)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            payload = item if isinstance(item, str) else json.dumps(item)
            yield f"event: {event}\ndata: {payload}\n\n"
            sent += 1

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

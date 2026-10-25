"""NNPM web application (Flask).

The main view renders the topology on an interactive map (Leaflet.js), with one
marker per registered node showing its address and a colour-coded
reachability status; clicking a marker opens an information panel with the
last-measured latency, available memory and CPU usage; new nodes are added by
clicking on the map. The distance between two markers is converted to a
synthetic per-link latency by :func:`calculate_distance` (Haversine formula)
and :func:`calculate_latency` (5 ms + 1 ms/km, clipped to [5, 50] ms), which
the scenario dialog can inject with tc-netem.

Usage::

    python -m gui.app [--host H] [--port P] [--local [ROOT]] [--config FILE] [--nodes-file FILE]
"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from typing import Any

from flask import Flask, Response, render_template, request

from gui import blueprints
from gui.state import AppState
from src.config import REPO_ROOT, load_config
from src.orchestration import local_mode
from src.orchestration.geo import calculate_distance, calculate_latency

APP_TITLE = "Neural Network Architecture Partitioning (NNPM) manager"
APP_SHORT_TITLE = "NNPM"

__all__ = ["APP_TITLE", "calculate_distance", "calculate_latency", "create_app", "main"]


def _basic_auth_guard(app: Flask, credentials: str) -> None:
    expected = credentials.encode("utf-8")

    @app.before_request
    def check() -> Response | None:
        header = request.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                given = base64.b64decode(header[6:])
            except ValueError:
                given = b""
            if given == expected:
                return None
        return Response("authentication required", 401, {"WWW-Authenticate": 'Basic realm="NNPM"'})


def create_app(state: AppState, *, start_prober: bool = True) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["JSON_SORT_KEYS"] = False
    app.extensions["nnpm"] = state
    auth_env = state.config.nnpm_basic_auth_env
    if auth_env and os.environ.get(auth_env):
        _basic_auth_guard(app, os.environ[auth_env])
    for blueprint in blueprints.all_blueprints():
        app.register_blueprint(blueprint)

    @app.route("/")
    def index() -> str:
        from src.models import registry
        from src.partitioning import algorithms

        return render_template(
            "index.html",
            title=APP_TITLE,
            short_title=APP_SHORT_TITLE,
            nodes=[n.to_dict() for n in state.topology.nodes],
            map_tiles_url=state.config.map_tiles_url,
            map_offline=state.config.map_offline,
            local_mode=state.local,
            models=registry.names(),
            algorithms=algorithms.names(),
            policies=["SJF", "RR"],
            interface=state.config.network_interface,
        )

    @app.route("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "nodes": len(state.topology.nodes), "busy": state.runs.busy}

    @app.errorhandler(404)
    def not_found(_error: Any) -> tuple[dict[str, str], int]:
        return {"error": "not found"}, 404

    if start_prober:
        state.prober.start()
    return app


def build_state(args: argparse.Namespace) -> AppState:
    if args.local is not None:
        root = Path(args.local).resolve()
        return AppState.local_testbed(root, devices=args.devices)
    config = load_config(args.config)
    nodes_path = Path(args.nodes_file) if args.nodes_file else REPO_ROOT / "nodes.json"
    return AppState(config, nodes_path, allow_dirty=args.allow_dirty)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the NNPM web application.")
    parser.add_argument("--host", default=None, help="default: nnpm_host from config.yaml")
    parser.add_argument(
        "--port", type=int, default=None, help="default: nnpm_port from config.yaml"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--nodes-file", default=None)
    parser.add_argument("--local", nargs="?", const=".local_testbed", default=None, metavar="ROOT")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--debug", action="store_true", help="Flask debug mode (never on the lab network)"
    )
    args = parser.parse_args(argv)
    state = build_state(args)
    app = create_app(state)
    host = args.host or (local_mode.LOOPBACK if args.local is not None else state.config.nnpm_host)
    port = args.port or state.config.nnpm_port
    print(
        f"{APP_TITLE} on http://{host}:{port}/ ({'local mode' if state.local else 'testbed'})",
        flush=True,
    )
    try:
        app.run(host=host, port=port, debug=args.debug, threaded=True, use_reloader=False)
    finally:
        state.prober.stop()


if __name__ == "__main__":
    main()

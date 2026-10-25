"""Runs: start an experiment or the benchmarking pipeline, stop, stream the log, browse results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, abort, request, send_file

from gui.blueprints.common import error, sse, state
from src.orchestration import scenario as scenario_module

bp = Blueprint("runs", __name__)


def _summary_from_disk(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {"run_id": run_dir.name, "status": "unknown"}
    )
    scenario_path = run_dir / "scenario.json"
    if scenario_path.exists():
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        summary["scenario_name"] = scenario.get("name")
        summary["pipeline"] = scenario.get("pipeline")
    summary["files"] = sorted(
        str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()
    )[:500]
    return summary


@bp.get("/api/runs")
def list_runs() -> dict[str, Any]:
    results_dir = state().results_dir
    runs = []
    if results_dir.exists():
        # `results/latest` is a symbolic link to the newest run: skip it, or that run appears twice.
        run_dirs = (p for p in results_dir.iterdir() if p.is_dir() and not p.is_symlink())
        for run_dir in sorted(run_dirs, reverse=True):
            summary = _summary_from_disk(run_dir)
            record = state().runs.history.get(run_dir.name)
            if record is not None:
                summary["status"] = record.status
            runs.append({k: v for k, v in summary.items() if k != "files"})
    current = state().runs.current
    return {
        "runs": runs,
        "current": current.run_id if current is not None else None,
        "busy": state().runs.busy,
    }


@bp.post("/api/runs")
def start_run() -> Any:
    payload = request.get_json(silent=True) or {}
    st = state()
    if st.runs.busy:
        return error("an experiment is already running", 409)
    try:
        if "scenario" in payload and isinstance(payload["scenario"], dict):
            saved = st.scenarios.save(payload["scenario"])
            scenario, meta = st.scenarios.load(saved.name, saved.version)
        else:
            scenario, meta = st.scenarios.load(payload["name"], payload.get("version"))
    except (KeyError, scenario_module.ScenarioError) as exc:
        return error(f"scenario: {exc}")
    version = {"name": meta.name, "version": meta.version, "sha256": meta.sha256}
    try:
        record = st.runs.start(
            scenario, version, pipeline=payload.get("pipeline"), run_id=payload.get("run_id")
        )
    except RuntimeError as exc:
        return error(str(exc), 409)
    return {"run_id": record.run_id, "scenario": version, "status": record.status}, 202


@bp.post("/api/runs/<run_id>/stop")
def stop_run(run_id: str) -> Any:
    current = state().runs.current
    if current is None or current.run_id != run_id or not state().runs.busy:
        return error("no such running experiment", 404)
    state().runs.stop()
    return {"run_id": run_id, "stopping": True}


@bp.get("/api/runs/<run_id>")
def run_detail(run_id: str) -> Any:
    run_dir = state().results_dir / run_id
    record = state().runs.history.get(run_id)
    if not run_dir.exists() and record is None:
        return error("unknown run", 404)
    summary: dict[str, Any] = (
        _summary_from_disk(run_dir) if run_dir.exists() else {"run_id": run_id}
    )
    if record is not None:
        summary.update(
            {
                "status": record.status,
                "error": record.error,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
            }
        )
        summary["log_tail"] = list(record.lines)[-200:]
    return summary


@bp.get("/api/runs/<run_id>/files/<path:relative>")
def run_file(run_id: str, relative: str) -> Response:
    run_dir = (state().results_dir / run_id).resolve()
    target = (run_dir / relative).resolve()
    if not target.is_relative_to(run_dir) or not target.is_file():
        abort(404)
    return send_file(target, as_attachment=request.args.get("download") == "1")


@bp.get("/events/runs/<run_id>")
def run_events(run_id: str) -> Response:
    manager = state().runs
    listener = manager.subscribe(run_id)
    record = manager.history.get(run_id)
    if record is not None:
        for line in list(record.lines)[-500:]:
            listener.put(line)
        if record.status != "running":
            listener.put(f"[nnpm][done] {record.status}")
    limit = request.args.get("limit", type=int)
    return sse(listener, event="log", limit=limit)

"""Scenarios: saved versions, validation, the Create Scenario dialog's output."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, request

from gui.blueprints.common import error, state
from src.orchestration import scenario as scenario_module

bp = Blueprint("scenarios", __name__, url_prefix="/api/scenarios")


@bp.get("")
def list_scenarios() -> dict[str, Any]:
    store = state().scenarios
    return {
        "scenarios": [
            {
                "name": name,
                "versions": [
                    {"version": v.version, "sha256": v.sha256} for v in store.versions(name)
                ],
            }
            for name in store.names()
        ]
    }


@bp.get("/<name>")
def latest(name: str) -> Any:
    return get_version(name, None)


@bp.get("/<name>/<int:version>")
def get_version(name: str, version: int | None) -> Any:
    try:
        data, meta = state().scenarios.load(name, version)
    except scenario_module.ScenarioError as exc:
        return error(str(exc), 404)
    return {"scenario": data, "version": meta.version, "sha256": meta.sha256}


@bp.post("")
def save() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return error("a scenario object is required")
    try:
        saved = state().scenarios.save(payload)
    except scenario_module.ScenarioError as exc:
        return error(str(exc))
    return {"name": saved.name, "version": saved.version, "sha256": saved.sha256}, 201


@bp.post("/validate")
def validate() -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return error("a scenario object is required")
    try:
        filled = scenario_module.validate(payload)
    except scenario_module.ScenarioError as exc:
        return {"valid": False, "error": str(exc)}
    return {"valid": True, "scenario": filled, "sha256": scenario_module.content_hash(filled)}

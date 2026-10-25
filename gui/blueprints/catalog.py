"""Catalogue: registered models, algorithms, policies, regimes and example scenarios."""

from __future__ import annotations

from typing import Any

from flask import Blueprint

from gui.blueprints.common import state
from src.config import REPO_ROOT
from src.orchestration import scenario as scenario_module
from src.testbed.load_generator import REGIMES

bp = Blueprint("catalog", __name__, url_prefix="/api/catalog")


@bp.get("")
def catalog() -> dict[str, Any]:
    from src.models import registry
    from src.partitioning import algorithms

    examples = []
    for path in sorted((REPO_ROOT / "examples" / "scenarios").glob("*.yaml")):
        try:
            scenario = scenario_module.load_scenario_file(path)
            examples.append(
                {
                    "file": str(path.relative_to(REPO_ROOT)),
                    "name": scenario["name"],
                    "description": scenario.get("description", ""),
                    "scenario": scenario,
                }
            )
        except scenario_module.ScenarioError:
            continue
    return {
        "models": registry.names(),
        "algorithms": algorithms.names(),
        "policies": ["SJF", "RR"],
        "regimes": {
            name: {"cpu": list(cpu), "memory": list(mem)} for name, (cpu, mem) in REGIMES.items()
        },
        "objectives": ["total_delay", "variance"],
        "interface": state().config.network_interface,
        "local_mode": state().local,
        "examples": examples,
        "schema": scenario_module.load_schema(),
    }

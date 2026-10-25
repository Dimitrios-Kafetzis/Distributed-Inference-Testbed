"""Flask blueprints of NNPM: topology, status, scenarios, runs, optimisation, catalogue."""

from __future__ import annotations

from flask import Blueprint

from gui.blueprints import catalog, optimise, runs, runtimes, scenarios, status, topology


def all_blueprints() -> list[Blueprint]:
    return [
        topology.bp,
        status.bp,
        scenarios.bp,
        runs.bp,
        optimise.bp,
        catalog.bp,
        runtimes.bp,
    ]

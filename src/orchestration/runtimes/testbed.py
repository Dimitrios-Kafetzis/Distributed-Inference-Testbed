"""The built-in runtime: per-layer profiling and partitioned inference (Chapter 3).

The profile stage proper lives in ``src/orchestration/experiment.py``
(``profile_with_testbed``); this module gives it the runtime interface so that
NNPM lists it next to the external runtimes. It needs no scenario block, no
readiness checks beyond provisioning and offers no maintenance action.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.common.nodes import Node
from src.orchestration.runtimes import ActionSession, ActionSpec, Check

if TYPE_CHECKING:
    from src.orchestration.experiment import ExperimentContext


class TestbedRuntime:
    name = "testbed"
    label = "Testbed (per-layer profiling and partitioned inference)"
    external = False
    description = (
        "Profiles the model layer by layer on every device, measures the edge server's "
        "capacity, optionally runs partitioned inference against the edge-server service, "
        "then aggregates and optimises."
    )

    @property
    def schema(self) -> dict[str, Any]:
        return {}

    def run(self, ctx: ExperimentContext) -> None:
        from src.orchestration.experiment import profile_with_testbed

        profile_with_testbed(ctx)

    def check_node(self, session: ActionSession, node: Node, params: dict[str, Any]) -> list[Check]:
        return []

    def actions(self) -> list[ActionSpec]:
        return []

    def perform(self, action: str, params: dict[str, Any], session: ActionSession) -> None:
        raise KeyError(f"the testbed runtime has no action {action!r}")

    def results(self, run_dir: Path) -> dict[str, Any] | None:
        return None


RUNTIME = TestbedRuntime()

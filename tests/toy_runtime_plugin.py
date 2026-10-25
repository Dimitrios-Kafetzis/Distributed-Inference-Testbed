"""A minimal runtime plug-in, loaded through TESTBED_RUNTIME_MODULES in the tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.nodes import Node
from src.orchestration.runtimes import ActionSession, ActionSpec, Check


class ToyRuntime:
    name = "toy"
    label = "Toy runtime"
    external = True
    description = "records that it ran"
    ran: list[str] = []

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {"knob": {"type": "integer", "default": 1}},
        }

    def run(self, ctx: Any) -> None:
        self.ran.append(ctx.run_id)
        ctx.emit("toy runtime ran")

    def check_node(self, session: ActionSession, node: Node, params: dict[str, Any]) -> list[Check]:
        return [Check("toy", "ok", node.id)]

    def actions(self) -> list[ActionSpec]:
        return [ActionSpec("wave", "Wave", "logs a greeting", {"type": "object"})]

    def perform(self, action: str, params: dict[str, Any], session: ActionSession) -> None:
        session.log(f"[task][info] {action}: hello {params.get('who', 'world')}")

    def results(self, run_dir: Path) -> dict[str, Any] | None:
        return {"kind": "toy"}


RUNTIME = ToyRuntime()

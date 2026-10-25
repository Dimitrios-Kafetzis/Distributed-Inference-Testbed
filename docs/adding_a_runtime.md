# Adding a runtime

A **runtime** is what the profile stage of an experiment runs. The built-in `testbed` runtime profiles the model layer by layer on every device and runs partitioned inference against the edge-server service. An **external runtime** runs its own distributed inference on the devices (HeadSplit is the first), under the testbed's scenario: shaping, load, heterogeneity, monitors, power traces, metadata, journal and teardown. Aggregation and optimisation are skipped for external runtimes, since they produce no per-layer profile.

The interface (`src/orchestration/runtimes/__init__.py`, `Runtime`) is what NNPM needs to configure and manage a runtime without knowing it:

| Member | Purpose |
|---|---|
| `name`, `label`, `description`, `external` | the catalogue entry and the runtime selector of the Create scenario dialog |
| `schema` | JSON Schema of the scenario block `scenario[name]`; the dialog renders it (`x-nnpm: {widget: node}` gives a device drop-down, `nodes` device check boxes, `hidden` skips a field); the block joins the scenario schema, so `validate()` accepts it |
| `run(ctx)` | the profile stage: start device processes with `ctx.start_device_process` (tagged, journaled, stopped at teardown), run commands with `ctx.executor`, write results under `ctx.paths.root`, put the main figures in `ctx.outputs` |
| `check_node(session, node, params)` | readiness checks of one device, as `Check(name, status, detail)` rows (`ok`, `warn`, `fail`, `skip`); the Runtimes dialog shows them as a table |
| `actions()` and `perform(action, params, session)` | maintenance actions (install, build a profile, distribute a checkpoint) with their parameter schemas; they run as tasks, one at a time, never during an experiment, logging to the terminal |
| `results(run_dir)` | the parsed results of a finished run for the run summary page |

`ActionSession` gives checks and actions the live transport (`session.executor`, SSH or local processes), a command runner for the edge server (`session.run_local`, injectable in tests), `session.copy_to_node` (rsync, or a plain copy in local mode), the configuration, the topology, a log sink and a cancel flag.

## A minimal runtime

```python
"""Echo runtime: runs one command per device and records its output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.nodes import Node
from src.common.runs import write_json_atomic
from src.orchestration.runtimes import ActionSession, ActionSpec, Check


class EchoRuntime:
    name = "echo"
    label = "Echo (example)"
    description = "Runs `hostname` on every device and stores the answers."
    external = True

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"greeting": {"type": "string", "default": "hi"}}}

    def run(self, ctx: Any) -> None:
        settings = ctx.scenario.get("echo", {})
        outcomes = ctx.executor.run_on_nodes(ctx.devices, f"echo {settings['greeting']}; hostname")
        answers = {node_id: o.stdout.strip() for node_id, o in outcomes.items()}
        write_json_atomic(ctx.paths.root / "echo" / "results.json", answers)
        ctx.outputs["echo"] = answers

    def check_node(self, session: ActionSession, node: Node, params: dict[str, Any]) -> list[Check]:
        outcome = session.executor.run_on_node(node, "command -v hostname", check=False)
        return [Check("hostname", "ok" if outcome.ok else "fail", outcome.stdout.strip())]

    def actions(self) -> list[ActionSpec]:
        return []

    def perform(self, action: str, params: dict[str, Any], session: ActionSession) -> None:
        raise KeyError(action)

    def results(self, run_dir: Path) -> dict[str, Any] | None:
        path = run_dir / "echo" / "results.json"
        return {"kind": "echo", "answers": path.read_text()} if path.exists() else None


RUNTIME = EchoRuntime()
```

Register it: a module in `TESTBED_RUNTIME_MODULES` (comma-separated), a `testbed.runtimes` entry point, or, inside the repository, a module under `src/orchestration/runtimes/` listed in `BUILTIN_MODULES`. A scenario then says `runtime: echo` and carries an `echo` block; NNPM lists the runtime in the Create scenario dialog and in the Runtimes dialog.

## Rules every runtime follows

- **Daemonless.** Device-side processes are started through `ctx.start_device_process` (a tagged detached process, recorded in the journal) or run to completion through the executor. Teardown stops what the journal lists and anything with the run's tag.
- **Undo at teardown.** A runtime applies no device setting of its own; the scenario's shaping, load and heterogeneity are applied and undone by the pipeline. If a runtime must change something, it records an undo entry with `ctx.record` and handles it in `undo_entry` of `src/orchestration/experiment.py`.
- **Testable without hardware.** Checks and actions are exercised against `FakeTransport` and `FakeRunner` (every command string asserted) and in local mode against stub scripts (`tests/test_runtimes.py`, `tests/integration/test_headsplit_runtime.py`).
- **Results in the run directory.** Under `results/<run id>/<runtime name>/`, with the main figures in `ctx.outputs` so that the Runs panel, the summary page and the sweep table can show them.

HeadSplit (`src/orchestration/runtimes/headsplit.py`) is the reference implementation; [headsplit_integration.md](headsplit_integration.md) describes what it does.

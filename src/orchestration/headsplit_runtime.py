"""HeadSplit as an optional external LLM runtime (filled in by the LLM package).

The HeadSplit controller and workers are started on the devices as tagged
processes for the duration of the run, under the same monitors, shaping and
metadata as any other experiment, and stopped at teardown. This module is
completed in the LLM work package; until then a scenario with
``runtime: headsplit`` is refused with a clear message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.orchestration.experiment import ExperimentContext


def run_headsplit(ctx: ExperimentContext) -> None:
    raise NotImplementedError("the HeadSplit runtime integration is not available yet")

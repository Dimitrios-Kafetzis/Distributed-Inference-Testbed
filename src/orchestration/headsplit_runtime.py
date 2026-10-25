"""HeadSplit runtime: compatibility module.

The implementation moved to :mod:`src.orchestration.runtimes.headsplit` when the
runtime plug-in interface was introduced; this module keeps the earlier import
path for scripts and documentation.
"""

from __future__ import annotations

from src.orchestration.runtimes.headsplit import (
    RUNTIME,
    HeadSplitError,
    HeadSplitRuntime,
    checkpoint_dirname,
    controller_command,
    headsplit_config,
    headsplit_dir,
    headsplit_python,
    load_results,
    parse_controller_json,
    parse_controller_output,
    run_headsplit,
    summarise_results,
    worker_command,
)

__all__ = [
    "RUNTIME",
    "HeadSplitError",
    "HeadSplitRuntime",
    "checkpoint_dirname",
    "controller_command",
    "headsplit_config",
    "headsplit_dir",
    "headsplit_python",
    "load_results",
    "parse_controller_json",
    "parse_controller_output",
    "run_headsplit",
    "summarise_results",
    "worker_command",
]

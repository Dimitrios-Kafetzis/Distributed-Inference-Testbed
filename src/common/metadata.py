"""Run metadata: the scenario, the software and the hardware behind every result.

At the start of each run, before the first device command, NNPM records the
scenario JSON, the Git commit hash of the testbed software (and whether the
tree was clean), the TensorFlow and PyTorch version strings, and the kernel
version of each device, together with the other facts the device probe
reports. The metadata is stored next to the results
(``results/<run_id>/metadata.json``), so any measurement can be traced back
to the exact software and configuration that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.nodes import Node
from src.common.probe import probe
from src.common.runner import CommandRunner, SubprocessRunner
from src.common.runs import write_json_atomic
from src.config import REPO_ROOT, Config
from src.network.ssh_executor import Executor

METADATA_SCHEMA_VERSION = 1


class DirtyTreeError(RuntimeError):
    """The testbed source tree has uncommitted changes."""


@dataclass
class GitState:
    commit: str | None
    dirty: bool
    describe: str | None = None


@dataclass
class RunMetadata:
    run_id: str
    created_at: str
    scenario: dict[str, Any]
    testbed: dict[str, Any]
    edge_server: dict[str, Any]
    nodes: dict[str, dict[str, Any]]
    seeds: dict[str, int] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    schema_version: int = METADATA_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "scenario": self.scenario,
            "testbed": self.testbed,
            "edge_server": self.edge_server,
            "nodes": self.nodes,
            "seeds": self.seeds,
            "notes": self.notes,
        }

    def save(self, path: Path) -> Path:
        write_json_atomic(path, self.to_dict())
        return path


def git_state(runner: CommandRunner | None = None, repo: Path = REPO_ROOT) -> GitState:
    runner = runner or SubprocessRunner()
    head = runner.run(["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=10)
    if not head.ok:
        return GitState(None, False)
    status = runner.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], timeout=10
    )
    describe = runner.run(["git", "-C", str(repo), "describe", "--tags", "--always"], timeout=10)
    return GitState(
        head.stdout.strip(),
        bool(status.stdout.strip()) if status.ok else False,
        describe.stdout.strip() if describe.ok else None,
    )


def probe_command(config: Config, frameworks: bool = True) -> str:
    """The probe as run on a device: from the code directory with its virtual environment."""
    flag = "" if frameworks else " --no-frameworks"
    return f"cd {config.ed_code_dir} && {config.ed_venv_dir}/bin/python -m src.common.probe{flag}"


def probe_nodes(
    executor: Executor,
    nodes: list[Node],
    config: Config,
    *,
    frameworks: bool = True,
    local: bool = False,
) -> dict[str, dict[str, Any]]:
    import sys

    if local:
        flag = "" if frameworks else " --no-frameworks"
        command = f"{sys.executable} -m src.common.probe{flag}"
    else:
        command = probe_command(config, frameworks)
    outcomes = executor.run_on_nodes(nodes, command, timeout=180)
    facts: dict[str, dict[str, Any]] = {}
    for node_id, outcome in outcomes.items():
        try:
            facts[node_id] = json.loads(outcome.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(
                f"{node_id}: probe output is not JSON: {outcome.stdout[-200:]}"
            ) from exc
    return facts


def collect_run_metadata(
    *,
    run_id: str,
    scenario: dict[str, Any],
    config: Config,
    executor: Executor,
    nodes: list[Node],
    seeds: dict[str, int] | None = None,
    allow_dirty: bool = False,
    local_runner: CommandRunner | None = None,
    frameworks: bool = True,
    local: bool = False,
    notes: dict[str, Any] | None = None,
) -> RunMetadata:
    """Gather everything; refuse a dirty source tree unless ``allow_dirty`` (recorded)."""
    state = git_state(local_runner)
    if state.dirty and not allow_dirty:
        raise DirtyTreeError(
            "the testbed source tree has uncommitted changes; commit them or pass allow_dirty "
            "(the override is recorded in the metadata)"
        )
    edge = probe(runner=local_runner, frameworks=frameworks)
    node_facts = probe_nodes(executor, nodes, config, frameworks=frameworks, local=local)
    return RunMetadata(
        run_id=run_id,
        created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        scenario=scenario,
        testbed={
            "git_commit": state.commit,
            "git_dirty": state.dirty,
            "git_describe": state.describe,
            "dirty_override": state.dirty and allow_dirty,
            "config": {k: v for k, v in config.__dict__.items() if k != "config_dir"},
        },
        edge_server=edge,
        nodes=node_facts,
        seeds=dict(seeds or {}),
        notes=dict(notes or {}),
    )

"""Run identifiers, the results directory layout and the run journal.

Results layout::

    results/<run_id>/
      metadata.json      software and hardware facts, written before any device command
      scenario.json      the scenario version that was run
      journal.json       applied changes and their undo actions (RunJournal)
      logs/merged.log    host- and stream-tagged output of every remote command
      raw/               one BenchmarkResults file per device and repetition
      monitor/  power/  latency/  optimisation/
      total_averages_of_benchmark_results.json
      calculated_max_partition_points.json
      final_latencies_averages.json
    results/latest -> <run_id>

The journal makes every scenario-level change on a device reversible: each
applied change is recorded with an undo specification before the next step
runs, and teardown undoes the pending entries in reverse order, also after a
failure or a restart of NNPM.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.clock import Clock, SystemClock

OUTPUT_FILES = (
    "total_averages_of_benchmark_results.json",
    "calculated_max_partition_points.json",
    "final_latencies_averages.json",
)


def new_run_id(clock: Clock | None = None) -> str:
    """``YYYYMMDD-HHMMSS-<6 hex>``, in UTC."""
    clock = clock or SystemClock()
    stamp = datetime.fromtimestamp(clock.time(), tz=UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON through a temporary file and rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class RunPaths:
    results_dir: Path
    run_id: str

    @property
    def root(self) -> Path:
        return self.results_dir / self.run_id

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def scenario(self) -> Path:
        return self.root / "scenario.json"

    @property
    def journal(self) -> Path:
        return self.root / "journal.json"

    @property
    def merged_log(self) -> Path:
        return self.root / "logs" / "merged.log"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def monitor(self) -> Path:
        return self.root / "monitor"

    @property
    def power(self) -> Path:
        return self.root / "power"

    @property
    def latency(self) -> Path:
        return self.root / "latency"

    @property
    def optimisation(self) -> Path:
        return self.root / "optimisation"

    @property
    def total_averages(self) -> Path:
        return self.root / OUTPUT_FILES[0]

    @property
    def max_partition_points(self) -> Path:
        return self.root / OUTPUT_FILES[1]

    @property
    def final_latencies_averages(self) -> Path:
        return self.root / OUTPUT_FILES[2]

    def create(self) -> RunPaths:
        for directory in (
            self.root,
            self.merged_log.parent,
            self.raw,
            self.monitor,
            self.power,
            self.latency,
            self.optimisation,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def link_latest(self) -> None:
        link = self.results_dir / "latest"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(self.run_id, target_is_directory=True)


@dataclass
class JournalEntry:
    """One applied change and how to undo it.

    ``undo`` is a serialisable specification (``{"kind": ..., ...}``) that the
    orchestration layer maps to a function; the journal itself does not know
    how to undo anything, which keeps it free of device dependencies.
    """

    sequence: int
    node_id: str
    kind: str
    description: str
    undo: dict[str, Any]
    applied_at: str
    undone: bool = False
    undo_error: str | None = None


@dataclass
class RunJournal:
    path: Path
    entries: list[JournalEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> RunJournal:
        journal = cls(path)
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            journal.entries = [JournalEntry(**e) for e in data.get("entries", [])]
        return journal

    def save(self) -> None:
        write_json_atomic(self.path, {"entries": [asdict(e) for e in self.entries]})

    def record(
        self,
        node_id: str,
        kind: str,
        description: str,
        undo: dict[str, Any],
        clock: Clock | None = None,
    ) -> JournalEntry:
        clock = clock or SystemClock()
        entry = JournalEntry(
            sequence=len(self.entries) + 1,
            node_id=node_id,
            kind=kind,
            description=description,
            undo=dict(undo),
            applied_at=datetime.fromtimestamp(clock.time(), tz=UTC).isoformat(),
        )
        self.entries.append(entry)
        self.save()
        return entry

    def pending(self) -> list[JournalEntry]:
        return [e for e in self.entries if not e.undone]

    def undo_all(self, perform: Callable[[JournalEntry], None]) -> list[JournalEntry]:
        """Undo pending entries in reverse order; continue after errors; return the failures."""
        failures: list[JournalEntry] = []
        for entry in reversed(self.pending()):
            try:
                perform(entry)
                entry.undone = True
                entry.undo_error = None
            except Exception as exc:
                entry.undo_error = f"{type(exc).__name__}: {exc}"
                failures.append(entry)
            self.save()
        return failures


def open_journals(results_dir: Path) -> Iterator[RunJournal]:
    """Journals of runs that still have pending entries (for teardown after a restart)."""
    if not results_dir.exists():
        return
    for journal_path in sorted(results_dir.glob("*/journal.json")):
        journal = RunJournal.load(journal_path)
        if journal.pending():
            yield journal

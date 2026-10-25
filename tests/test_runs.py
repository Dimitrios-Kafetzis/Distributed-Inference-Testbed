from __future__ import annotations

import json
import re
from pathlib import Path

from src.common.clock import FakeClock
from src.common.runs import (
    OUTPUT_FILES,
    JournalEntry,
    RunJournal,
    RunPaths,
    new_run_id,
    open_journals,
    write_json_atomic,
)


def test_run_id_format() -> None:
    run_id = new_run_id(FakeClock(start_time=1_700_000_000.0))
    assert re.fullmatch(r"20231114-\d{6}-[0-9a-f]{6}", run_id)


def test_run_paths_create_and_latest(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path / "results", "r1").create()
    for sub in (paths.raw, paths.monitor, paths.power, paths.latency, paths.optimisation):
        assert sub.is_dir()
    assert paths.merged_log.parent.is_dir()
    assert paths.total_averages.name == OUTPUT_FILES[0]
    paths.link_latest()
    assert (tmp_path / "results" / "latest").resolve() == paths.root.resolve()
    RunPaths(tmp_path / "results", "r2").create().link_latest()
    assert (tmp_path / "results" / "latest").resolve().name == "r2"


def test_write_json_atomic_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b.json"
    write_json_atomic(target, {"x": 1})
    assert json.loads(target.read_text()) == {"x": 1}
    assert [p.name for p in target.parent.iterdir()] == ["b.json"]


def test_journal_records_and_undoes_in_reverse(tmp_path: Path) -> None:
    journal = RunJournal(tmp_path / "journal.json")
    clock = FakeClock()
    journal.record("device1", "netem", "delay 10 ms on wlan0", {"kind": "tc_del"}, clock)
    journal.record("device1", "load", "load generator", {"kind": "stop", "tag": "t"}, clock)
    journal.record("device2", "netem", "delay 20 ms", {"kind": "tc_del"}, clock)
    assert [e.sequence for e in journal.pending()] == [1, 2, 3]

    order: list[int] = []

    def perform(entry: JournalEntry) -> None:
        order.append(entry.sequence)
        if entry.sequence == 2:
            raise RuntimeError("device offline")

    failures = journal.undo_all(perform)
    assert order == [3, 2, 1]
    assert [f.sequence for f in failures] == [2]
    assert failures[0].undo_error == "RuntimeError: device offline"
    assert [e.sequence for e in journal.pending()] == [2]

    reloaded = RunJournal.load(tmp_path / "journal.json")
    assert [e.sequence for e in reloaded.pending()] == [2]
    assert reloaded.entries[0].undone is True


def test_open_journals_finds_runs_with_pending_entries(tmp_path: Path) -> None:
    results = tmp_path / "results"
    done = RunPaths(results, "done").create()
    RunJournal(done.journal).record("d", "x", "", {}).undone = True
    j = RunJournal.load(done.journal)
    j.undo_all(lambda e: None)
    pending = RunPaths(results, "pending").create()
    RunJournal(pending.journal).record("d", "x", "", {})
    assert [j.path.parent.name for j in open_journals(results)] == ["pending"]
    assert list(open_journals(tmp_path / "nowhere")) == []

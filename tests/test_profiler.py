from __future__ import annotations

import os
import re
from pathlib import Path

from src.benchmarking.benchmark_results import Origin, StageRecord
from src.models import profiler
from src.models.fake import FakeStagesAdapter

REPO = Path(__file__).resolve().parents[1]


def test_profile_once_times_stages_in_isolation_and_records_peaks() -> None:
    specs = [
        ("slow", 40.0, 3.0, 64, 1_000_000),
        ("fast", 5.0, 0.0, 16, 10_000),
        ("alloc", 30.0, 6.0, 8, 5_000),
    ]
    adapter = FakeStagesAdapter(specs)
    adapter.build(0)
    x = adapter.make_input(0)
    record = profiler.profile_once(
        adapter,
        x,
        run_id="r",
        node_id="d",
        repetition=0,
        warmup_runs=0,
        origin=Origin(1.0, 0),
        hostname="h",
        sampler_interval_s=0.002,
    )
    assert [s.name for s in record.stages] == ["slow", "fast", "alloc"]
    assert record.stages[0].latency_ns >= 40_000_000
    assert record.stages[1].latency_ns < record.stages[0].latency_ns
    assert record.stages[2].memory_peak_mb >= 5.0  # the 6 MB allocation is visible as RSS
    assert record.stages[0].flops == 1_000_000 and record.stages[0].output_bytes == 256
    points = record.partition_points
    assert points[0].s == 0 and points[0].transmitted_bytes == adapter.input_bytes
    assert points[3].cumulative_latency_ns == sum(s.latency_ns for s in record.stages)
    assert points[3].cumulative_memory_mb == max(s.memory_peak_mb for s in record.stages)
    assert record.metadata["framework"] == "fake"
    # CPU peaks are utilisations, never the artefacts of a too-short sampling window.
    assert all(0.0 <= s.cpu_peak_percent <= 100.0 * (os.cpu_count() or 1) for s in record.stages)
    assert profiler.capacity_flops_per_second(record) > 0


def test_cumulative_points_rules() -> None:
    stages = [
        StageRecord(1, "a", 10, 50.0, 4.0, 1, 1, 100),
        StageRecord(2, "b", 20, 30.0, 2.0, 1, 1, 50),
    ]
    points = profiler.cumulative_points(stages, 400)
    assert [
        (
            p.s,
            p.cumulative_latency_ns,
            p.cumulative_cpu_percent,
            p.cumulative_memory_mb,
            p.transmitted_bytes,
        )
        for p in points
    ] == [
        (0, 0, 0.0, 0.0, 400),
        (1, 10, 50.0, 4.0, 100),
        (2, 30, 80.0, 4.0, 50),
    ]


def test_timing_code_uses_perf_counter_ns_only() -> None:
    """Interval timing must use perf_counter_ns; time.time() only for wall-clock stamps."""
    offences = []
    for path in list((REPO / "src" / "models").rglob("*.py")) + list(
        (REPO / "src" / "inference").rglob("*.py")
    ):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if (
                re.search(r"\btime\.time\(\)", line)
                and "Origin(" not in line
                and "wall" not in line
            ):
                offences.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
            if re.search(r"\btime\.perf_counter\(\)", line) or re.search(
                r"\btime\.monotonic\(\)", line
            ):
                offences.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert not offences, "\n".join(offences)

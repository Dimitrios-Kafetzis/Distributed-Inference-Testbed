"""Per-layer profiler: execution time, peak CPU and peak memory of every stage.

The model is executed stage by stage. Each stage is timed with
``time.perf_counter_ns()``; while it runs, a sampler thread records the
process's CPU utilisation and resident memory at a short interval, and the
peaks are kept. Warm-up passes precede the measured repetitions. The result
of one repetition is a :class:`BenchmarkResults` record; the cumulative
values per partition point are the sums of the per-stage latencies and CPU
peaks and, for memory, the largest peak among the stages up to s (the
memory needed to run stages 1..s).
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import psutil

from src.benchmarking.benchmark_results import (
    BenchmarkResults,
    Origin,
    PartitionPointRecord,
    StageRecord,
    new_record,
)
from src.models.registry import Array, ModelAdapter


@dataclass
class PeakSample:
    cpu_percent: float
    rss_mb: float


class PeakSampler:
    """Samples the current process at ``interval_s`` and keeps the peaks."""

    def __init__(self, interval_s: float = 0.005, process: psutil.Process | None = None) -> None:
        self.interval_s = interval_s
        self.process = process or psutil.Process()
        self.peak = PeakSample(0.0, 0.0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> PeakSample:
        rss = self.process.memory_info().rss / 2**20
        cpu = self.process.cpu_percent(interval=None)
        return PeakSample(float(cpu), float(rss))

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._sample_once()
            self.peak = PeakSample(
                max(self.peak.cpu_percent, sample.cpu_percent), max(self.peak.rss_mb, sample.rss_mb)
            )
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PeakSampler:
        self.process.cpu_percent(interval=None)  # prime the counter
        self.peak = PeakSample(0.0, self.process.memory_info().rss / 2**20)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        final = self._sample_once()
        self.peak = PeakSample(
            max(self.peak.cpu_percent, final.cpu_percent), max(self.peak.rss_mb, final.rss_mb)
        )


def cumulative_points(stages: list[StageRecord], input_bytes: int) -> list[PartitionPointRecord]:
    points = [PartitionPointRecord(0, 0, 0.0, 0.0, input_bytes)]
    latency = 0
    cpu = 0.0
    memory = 0.0
    for stage in sorted(stages, key=lambda st: st.index):
        latency += stage.latency_ns
        cpu += stage.cpu_peak_percent
        memory = max(memory, stage.memory_peak_mb)
        points.append(PartitionPointRecord(stage.index, latency, cpu, memory, stage.output_bytes))
    return points


def profile_once(
    adapter: ModelAdapter,
    x: Array,
    *,
    run_id: str,
    node_id: str,
    repetition: int,
    warmup_runs: int,
    origin: Origin,
    hostname: str | None = None,
    sampler_interval_s: float = 0.005,
    baseline_rss_mb: float | None = None,
    perf_counter_ns: Callable[[], int] = time.perf_counter_ns,
) -> BenchmarkResults:
    """One measured pass over all stages, each stage in isolation, peaks per stage."""
    process = psutil.Process()
    if baseline_rss_mb is None:
        baseline_rss_mb = process.memory_info().rss / 2**20
    stage_records: list[StageRecord] = []
    current = x
    for info in adapter.stages:
        with PeakSampler(sampler_interval_s, process) as sampler:
            started = perf_counter_ns()
            current = adapter.run_stages(current, info.index, info.index)
            elapsed = perf_counter_ns() - started
        stage_records.append(
            StageRecord(
                index=info.index,
                name=info.name,
                latency_ns=int(elapsed),
                cpu_peak_percent=round(sampler.peak.cpu_percent, 2),
                memory_peak_mb=round(max(0.0, sampler.peak.rss_mb - baseline_rss_mb), 3),
                flops=info.flops,
                macs=info.macs,
                output_bytes=info.output_bytes,
            )
        )
    record = new_record(
        "benchmark",
        run_id=run_id,
        hostname=hostname or socket.gethostname(),
        node_id=node_id,
        origin=origin,
        model=adapter.name,
        repetition=repetition,
        warmup_runs=warmup_runs,
        stages=stage_records,
        partition_points=cumulative_points(stage_records, adapter.input_bytes),
        metadata={
            "framework": adapter.framework,
            "input_shape": list(adapter.input_shape),
            "baseline_rss_mb": round(baseline_rss_mb, 3),
        },
    )
    assert isinstance(record, BenchmarkResults)
    return record


def warm_up(adapter: ModelAdapter, x: Array, runs: int) -> None:
    for _ in range(runs):
        adapter.run_stages(x, 1, len(adapter.stages))


def capacity_flops_per_second(record: BenchmarkResults) -> float:
    """Average FLOP/s over the measured stages (the device's processing rate)."""
    total_flops = sum(s.flops for s in record.stages)
    total_ns = sum(s.latency_ns for s in record.stages)
    return total_flops / (total_ns / 1e9) if total_ns else 0.0

"""Synthetic measurement records for tests and sample data. Every value is invented."""

from __future__ import annotations

import random
from datetime import UTC, datetime

from src.benchmarking.benchmark_results import (
    BenchmarkResults,
    Origin,
    PhaseRecord,
    PowerSample,
    PowerTrace,
    ResourceSample,
    ResourceTrace,
    StageRecord,
    new_record,
)
from src.network.records import LatencyRecord, PingResult, PingSample, SystemSnapshot

STAGE_NAMES = [
    "conv1",
    "pool1",
    "conv2",
    "pool2",
    "conv3",
    "conv4",
    "conv5",
    "pool3",
    "fc1",
    "fc2",
    "fc3",
]
OUTPUT_BYTES = [1161600, 279936, 746496, 173056, 259584, 259584, 173056, 36864, 16384, 16384, 4000]
INPUT_BYTES = 618348


def origin(device_index: int = 0, run_offset_s: float = 0.0) -> Origin:
    # Each device has its own monotonic counter; the wall clock is shared (NTP).
    return Origin(
        wall_time_s=1_760_000_000.0 + run_offset_s, perf_counter_ns=10**12 * (device_index + 1)
    )


def benchmark(
    node_id: str,
    repetition: int,
    *,
    run_id: str = "20261001-120000-synth1",
    seed: int = 0,
    device_index: int = 0,
    stage_scale: float = 1.0,
    with_phases: bool = False,
) -> BenchmarkResults:
    rng = random.Random(seed * 1000 + repetition)
    stages = []
    for i, name in enumerate(STAGE_NAMES, start=1):
        base_ms = 40.0 if name.startswith("conv") else 3.0 if name.startswith("pool") else 12.0
        latency_ms = base_ms * stage_scale * rng.uniform(0.9, 1.1)
        stages.append(
            StageRecord(
                index=i,
                name=name,
                latency_ns=int(latency_ms * 1e6),
                cpu_peak_percent=round(rng.uniform(20, 95), 1),
                memory_peak_mb=round(rng.uniform(10, 120), 1),
                flops=int(2e8 / i),
                macs=int(1e8 / i),
                output_bytes=OUTPUT_BYTES[i - 1],
            )
        )
    record = new_record(
        "benchmark",
        run_id=run_id,
        hostname=f"{node_id}-host",
        node_id=node_id,
        origin=origin(device_index),
        model="alexnet",
        repetition=repetition,
        warmup_runs=2,
        stages=stages,
        partition_points=BenchmarkResults.cumulative_from_stages(stages, INPUT_BYTES),
        metadata={"synthetic": True},
    )
    assert isinstance(record, BenchmarkResults)
    if with_phases:
        for s in (0, 4, 8, 11):
            local = sum(st.latency_ns for st in stages[:s])
            record.phases.append(
                PhaseRecord(
                    s=s,
                    local_ns=local,
                    transmission_ns=int(rng.uniform(5, 40) * 1e6),
                    remote_ns=int(rng.uniform(10, 80) * 1e6),
                    queue_wait_ns=int(rng.uniform(0, 30) * 1e6),
                )
            )
    return record


def resource_trace(
    node_id: str,
    *,
    run_id: str = "20261001-120000-synth1",
    seconds: int = 10,
    device_index: int = 0,
    throttle_at: int | None = None,
) -> ResourceTrace:
    rng = random.Random(device_index)
    base = origin(device_index)
    samples = []
    for t in range(seconds):
        freq = 1_800_000 if throttle_at is None or t < throttle_at else 1_500_000
        samples.append(
            ResourceSample(
                perf_counter_ns=base.perf_counter_ns + t * 10**9,
                cpu_percent=round(rng.uniform(5, 60), 1),
                memory_used_mb=round(rng.uniform(600, 900), 1),
                memory_available_mb=round(rng.uniform(2800, 3200), 1),
                temperature_c=round(rng.uniform(45, 70), 1),
                frequencies_khz=[freq] * 4,
                throttled="0x0" if freq == 1_800_000 else "0x20000",
            )
        )
    record = new_record(
        "resource_trace",
        run_id=run_id,
        hostname=f"{node_id}-host",
        node_id=node_id,
        origin=base,
        interval_s=1.0,
        nominal_frequency_khz=1_800_000,
        samples=samples,
        metadata={"synthetic": True},
    )
    assert isinstance(record, ResourceTrace)
    return record


def power_trace(
    node_id: str,
    *,
    run_id: str = "20261001-120000-synth1",
    seconds: int = 10,
    device_index: int = 0,
    power_mw: float = 4000.0,
) -> PowerTrace:
    base = origin(device_index)
    samples = [
        PowerSample(
            perf_counter_ns=base.perf_counter_ns + t * 10**9,
            current_ma=power_mw / 5.0,
            voltage_v=5.0,
            power_mw=power_mw,
        )
        for t in range(seconds)
    ]
    record = new_record(
        "power_trace",
        run_id=run_id,
        hostname=f"{node_id}-host",
        node_id=node_id,
        origin=base,
        interval_s=1.0,
        samples=samples,
        metadata={"synthetic": True},
    )
    assert isinstance(record, PowerTrace)
    return record


def latency_record(
    node_id: str,
    *,
    run_id: str = "20261001-120000-synth1",
    rtts: list[float | None] | None = None,
    available_mb: float = 3000.0,
    cpu: float = 5.0,
    extra_targets: dict[str, list[float | None]] | None = None,
) -> LatencyRecord:
    rtts = rtts if rtts is not None else [5.1, 5.3, None, 4.9, 5.0]
    pings = [
        PingResult(
            target_id="edge_server",
            target_ip="192.0.2.1",
            sent=len(rtts),
            received=sum(1 for r in rtts if r is not None),
            samples=[PingSample(i + 1, r) for i, r in enumerate(rtts)],
        )
    ]
    for target, values in (extra_targets or {}).items():
        pings.append(
            PingResult(
                target_id=target,
                target_ip="192.0.2.2",
                sent=len(values),
                received=sum(1 for r in values if r is not None),
                samples=[PingSample(i + 1, r) for i, r in enumerate(values)],
            )
        )
    return LatencyRecord(
        run_id=run_id,
        node_id=node_id,
        hostname=f"{node_id}-host",
        measured_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC).isoformat(),
        pings=pings,
        system=SystemSnapshot(
            total_memory_mb=3800.0,
            available_memory_mb=available_mb,
            available_memory_percent=100.0 * available_mb / 3800.0,
            cpu_usage_percent=cpu,
        ),
        metadata={"synthetic": True},
    )

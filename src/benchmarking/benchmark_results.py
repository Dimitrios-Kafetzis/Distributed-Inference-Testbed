"""Typed measurement records and their JSON serialisation.

Every benchmarking measurement is written to a structured JSON file, one per
experiment run and repetition, through the typed :class:`BenchmarkResults`
wrapper, which maps each layer (stage) index to its measured latency, CPU
utilisation and memory requirement, and each partition point to the
cumulative values up to it. The resource-monitoring agent and the power
monitor write their traces in the same serialisation family
(:class:`ResourceTrace`, :class:`PowerTrace`), so the aggregation step reads
all of them back with one loader.

Conventions:
  * intervals are integers in nanoseconds from ``time.perf_counter_ns()``;
  * every record carries an origin: the NTP-synchronised wall clock
    (``time.time()``) and the monotonic counter captured together at the
    start signal of the experiment, so traces from different devices can be
    re-aligned during aggregation;
  * memory is in megabytes (MiB), CPU in percent, sizes in bytes.

The JSON Schema is ``docs/schemas/benchmark_results.schema.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import jsonschema

from src.common.runs import write_json_atomic
from src.config import REPO_ROOT

SCHEMA_VERSION = 1
SCHEMA_PATH = REPO_ROOT / "docs" / "schemas" / "benchmark_results.schema.json"

RecordKind = Literal["benchmark", "resource_trace", "power_trace"]


class RecordError(ValueError):
    """Raised when a record file does not match the schema."""


@dataclass(frozen=True)
class Origin:
    """Common time origin of a device's traces, captured at the start signal."""

    wall_time_s: float
    perf_counter_ns: int

    def to_wall_time(self, perf_ns: int) -> float:
        """Wall-clock time of a monotonic timestamp taken on the same device."""
        return self.wall_time_s + (perf_ns - self.perf_counter_ns) / 1e9


@dataclass(frozen=True)
class StageRecord:
    """One model stage (layer), executed once, in isolation from the other stages."""

    index: int
    name: str
    latency_ns: int
    cpu_peak_percent: float
    memory_peak_mb: float
    flops: int
    macs: int
    output_bytes: int


@dataclass(frozen=True)
class PartitionPointRecord:
    """Cumulative values for executing stages 1..s locally; s = 0 means nothing local."""

    s: int
    cumulative_latency_ns: int
    cumulative_cpu_percent: float
    cumulative_memory_mb: float
    transmitted_bytes: int


@dataclass(frozen=True)
class PhaseRecord:
    """Per-phase timing of one end-to-end partitioned inference at partition point s."""

    s: int
    local_ns: int
    transmission_ns: int
    remote_ns: int
    queue_wait_ns: int
    server_arrival_wall_s: float | None = None
    server_service_start_wall_s: float | None = None
    server_service_end_wall_s: float | None = None


@dataclass
class RecordBase:
    schema_version: int
    kind: str
    run_id: str
    hostname: str
    node_id: str
    origin: Origin
    metadata: dict[str, Any] = field(default_factory=dict)

    KIND: ClassVar[RecordKind]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        payload = self.to_dict()
        validate(payload)
        write_json_atomic(target, payload)
        return target


@dataclass
class BenchmarkResults(RecordBase):
    """Per-layer and per-partition-point measurements of one run and repetition."""

    KIND: ClassVar[RecordKind] = "benchmark"

    model: str = ""
    repetition: int = 0
    warmup_runs: int = 0
    stages: list[StageRecord] = field(default_factory=list)
    partition_points: list[PartitionPointRecord] = field(default_factory=list)
    phases: list[PhaseRecord] = field(default_factory=list)

    # --- the mapping App. B describes: layer index -> latency, CPU, memory ---------

    def stage(self, index: int) -> StageRecord:
        for stage in self.stages:
            if stage.index == index:
                return stage
        raise KeyError(index)

    def latency_ns(self, index: int) -> int:
        return self.stage(index).latency_ns

    def cpu_percent(self, index: int) -> float:
        return self.stage(index).cpu_peak_percent

    def memory_mb(self, index: int) -> float:
        return self.stage(index).memory_peak_mb

    def partition_point(self, s: int) -> PartitionPointRecord:
        for point in self.partition_points:
            if point.s == s:
                return point
        raise KeyError(s)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @staticmethod
    def cumulative_from_stages(
        stages: list[StageRecord], input_bytes: int
    ) -> list[PartitionPointRecord]:
        """Partition points 0..S from per-stage values: cumulative sums, and the bytes
        transmitted when cutting after stage s (the input itself for s = 0)."""
        points = [PartitionPointRecord(0, 0, 0.0, 0.0, input_bytes)]
        latency = 0
        cpu = 0.0
        memory = 0.0
        for stage in sorted(stages, key=lambda st: st.index):
            latency += stage.latency_ns
            cpu += stage.cpu_peak_percent
            memory += stage.memory_peak_mb
            points.append(
                PartitionPointRecord(stage.index, latency, cpu, memory, stage.output_bytes)
            )
        return points


@dataclass(frozen=True)
class ResourceSample:
    perf_counter_ns: int
    cpu_percent: float
    memory_used_mb: float
    memory_available_mb: float
    temperature_c: float | None
    frequencies_khz: list[int]
    throttled: str | None  # raw flags, e.g. "0x50000" from vcgencmd, when available


@dataclass
class ResourceTrace(RecordBase):
    """1 Hz samples of CPU, memory, temperature and clock frequency during a run."""

    KIND: ClassVar[RecordKind] = "resource_trace"

    interval_s: float = 1.0
    nominal_frequency_khz: int | None = None
    samples: list[ResourceSample] = field(default_factory=list)

    def below_nominal(self) -> list[ResourceSample]:
        """Samples in which any core ran below the nominal frequency (throttling)."""
        if self.nominal_frequency_khz is None:
            return []
        return [
            s
            for s in self.samples
            if any(f < self.nominal_frequency_khz for f in s.frequencies_khz)
        ]


@dataclass(frozen=True)
class PowerSample:
    perf_counter_ns: int
    current_ma: float
    voltage_v: float
    power_mw: float


@dataclass
class PowerTrace(RecordBase):
    """INA260 readings at a configurable interval, with per-metric averages."""

    KIND: ClassVar[RecordKind] = "power_trace"

    interval_s: float = 1.0
    samples: list[PowerSample] = field(default_factory=list)

    def averages(self) -> dict[str, float]:
        if not self.samples:
            return {"current_ma": 0.0, "voltage_v": 0.0, "power_mw": 0.0}
        n = len(self.samples)
        return {
            "current_ma": sum(s.current_ma for s in self.samples) / n,
            "voltage_v": sum(s.voltage_v for s in self.samples) / n,
            "power_mw": sum(s.power_mw for s in self.samples) / n,
        }


Record = BenchmarkResults | ResourceTrace | PowerTrace

_KINDS: dict[str, type[Record]] = {
    BenchmarkResults.KIND: BenchmarkResults,
    ResourceTrace.KIND: ResourceTrace,
    PowerTrace.KIND: PowerTrace,
}


def load_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        schema: dict[str, Any] = json.load(fh)
    return schema


def validate(payload: dict[str, Any]) -> None:
    try:
        jsonschema.validate(payload, load_schema())
    except jsonschema.ValidationError as exc:
        location = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise RecordError(f"invalid record at {location}: {exc.message}") from exc


def from_dict(payload: dict[str, Any]) -> Record:
    validate(payload)
    kind = payload["kind"]
    data = dict(payload)
    data["origin"] = Origin(**data["origin"])
    if kind == BenchmarkResults.KIND:
        data["stages"] = [StageRecord(**s) for s in data.get("stages", [])]
        data["partition_points"] = [
            PartitionPointRecord(**p) for p in data.get("partition_points", [])
        ]
        data["phases"] = [PhaseRecord(**p) for p in data.get("phases", [])]
        return BenchmarkResults(**data)
    if kind == ResourceTrace.KIND:
        data["samples"] = [ResourceSample(**s) for s in data.get("samples", [])]
        return ResourceTrace(**data)
    data["samples"] = [PowerSample(**s) for s in data.get("samples", [])]
    return PowerTrace(**data)


def from_json(text: str) -> Record:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RecordError(f"not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RecordError("a record must be a JSON object")
    return from_dict(payload)


def load(path: str | Path) -> Record:
    return from_json(Path(path).read_text(encoding="utf-8"))


def load_benchmark(path: str | Path) -> BenchmarkResults:
    record = load(path)
    if not isinstance(record, BenchmarkResults):
        raise RecordError(f"{path}: expected a benchmark record, found {record.kind}")
    return record


def new_record(kind: RecordKind, **kwargs: Any) -> Record:
    cls = _KINDS[kind]
    return cls(schema_version=SCHEMA_VERSION, kind=kind, **kwargs)


def main() -> None:
    """Validate and summarise a record file (command-line check)."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate a measurement record file.")
    parser.add_argument("path")
    args = parser.parse_args()
    record = load(args.path)
    print(f"{record.kind}: run {record.run_id}, node {record.node_id} ({record.hostname})")
    if isinstance(record, BenchmarkResults):
        print(f"model {record.model}, repetition {record.repetition}, {record.num_stages} stages")
        for stage in record.stages:
            print(
                f"  stage {stage.index:2d} {stage.name:12s} {stage.latency_ns / 1e6:9.3f} ms"
                f"  cpu {stage.cpu_peak_percent:6.1f} %  mem {stage.memory_peak_mb:8.1f} MB"
            )
    else:
        print(f"{len(record.samples)} samples at {record.interval_s} s")


if __name__ == "__main__":
    main()

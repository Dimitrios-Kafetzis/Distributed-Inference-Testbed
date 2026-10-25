"""The latency-and-system-metric record written by each end device.

``src/network/latency_measurement.py`` runs on a device, pings the edge server
(and any other registered node), takes a psutil snapshot of the device's
memory and CPU, and writes one of these records to a JSON file on the device.
``src/network/node_statistics.py`` on the edge server aggregates the collected
records into ``final_latencies_averages.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.common.runs import write_json_atomic

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PingSample:
    sequence: int
    rtt_ms: float | None  # None = no reply


@dataclass(frozen=True)
class PingResult:
    target_id: str
    target_ip: str
    sent: int
    received: int
    samples: list[PingSample]

    @property
    def rtts(self) -> list[float]:
        return [s.rtt_ms for s in self.samples if s.rtt_ms is not None]

    @property
    def loss_percent(self) -> float:
        return 100.0 * (self.sent - self.received) / self.sent if self.sent else 0.0


@dataclass(frozen=True)
class SystemSnapshot:
    total_memory_mb: float
    available_memory_mb: float
    available_memory_percent: float
    cpu_usage_percent: float  # averaged over a one-second window


@dataclass
class LatencyRecord:
    run_id: str
    node_id: str
    hostname: str
    measured_at: str  # ISO 8601, wall clock
    pings: list[PingResult]
    system: SystemSnapshot
    schema_version: int = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def ping_to(self, target_id: str) -> PingResult | None:
        return next((p for p in self.pings if p.target_id == target_id), None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        write_json_atomic(Path(path), self.to_dict())
        return Path(path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatencyRecord:
        pings = [
            PingResult(
                target_id=p["target_id"],
                target_ip=p["target_ip"],
                sent=p["sent"],
                received=p["received"],
                samples=[PingSample(**s) for s in p["samples"]],
            )
            for p in data["pings"]
        ]
        return cls(
            run_id=data["run_id"],
            node_id=data["node_id"],
            hostname=data["hostname"],
            measured_at=data["measured_at"],
            pings=pings,
            system=SystemSnapshot(**data["system"]),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> LatencyRecord:
        with Path(path).open(encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

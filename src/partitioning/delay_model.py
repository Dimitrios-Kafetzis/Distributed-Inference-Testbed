"""The per-device delay model: local, transmission and remote computation.

For device i with partition point s_i in {0, ..., S}:

* local delay: the profiled cumulative execution time of stages 1..s_i
  (``measured`` mode), or F_i^L(s_i) / C_i under n_i backlogged background
  processes with quantum q (``analytical`` mode, Chapter 4 eq. local);
* transmission delay: A_i(s_i) / W_i plus the one-way latency (data size over
  the measured rate, ``size_over_rate``), or the ping round-trip time alone
  (``rtt_only``, the behaviour of the earlier release); zero when s_i = S;
* remote delay: the time from when the edge server begins processing the
  offloaded parts until device i's task is complete, under Round-Robin
  (eq. dnn-rr-delay: F_i^R / C^R x N_off - (N_off - 1) q^R) or
  Shortest-Job-First (eq. dnn-sjf-delay: (F_i^R + sum of the workloads
  ranked before i) / C^R, ties by device index).

The total delay of a vector is the sum over devices; the variance is the
population variance of the per-device delays (Chapter 4, problem P2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

POLICIES = ("RR", "SJF")
TRANSMISSION_MODES = ("size_over_rate", "rtt_only")
LOCAL_MODES = ("measured", "analytical")


@dataclass
class DeviceProfile:
    node_id: str
    index: int
    num_stages: int
    stage_flops: dict[int, int]  # stage index -> FLOPs
    cumulative_latency_s: dict[int, float]  # s -> seconds for stages 1..s (s = 0 -> 0)
    transmitted_bytes: dict[int, int]  # s -> bytes of the cut (s = 0 -> the input)
    max_partition_point: int
    rtt_ms: float | None = None
    throughput_mbps: float | None = None
    capacity_flops: float | None = None  # C_i (analytical local mode)
    background_processes: int = 0  # n_i (analytical local mode)

    def offloaded_flops(self, s: int) -> int:
        return sum(f for i, f in self.stage_flops.items() if i > s)

    def local_flops(self, s: int) -> int:
        return sum(f for i, f in self.stage_flops.items() if i <= s)

    @property
    def feasible_points(self) -> list[int]:
        return list(range(0, self.max_partition_point + 1))


@dataclass
class DelayModelConfig:
    policy: str = "SJF"
    edge_capacity_flops: float = 2.0e10  # C^R
    rr_quantum_s: float = 0.0  # q^R
    transmission_mode: str = "size_over_rate"
    local_mode: str = "measured"
    device_quantum_s: float = 0.0  # q (analytical local mode)

    def __post_init__(self) -> None:
        self.policy = self.policy.upper()
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if self.transmission_mode not in TRANSMISSION_MODES:
            raise ValueError(f"transmission_mode must be one of {TRANSMISSION_MODES}")
        if self.local_mode not in LOCAL_MODES:
            raise ValueError(f"local_mode must be one of {LOCAL_MODES}")
        if self.edge_capacity_flops <= 0:
            raise ValueError("edge_capacity_flops must be positive")


@dataclass(frozen=True)
class DeviceDelays:
    node_id: str
    s: int
    local_s: float
    transmission_s: float
    remote_s: float

    @property
    def total_s(self) -> float:
        return self.local_s + self.transmission_s + self.remote_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "s": self.s,
            "local_ms": self.local_s * 1e3,
            "transmission_ms": self.transmission_s * 1e3,
            "remote_ms": self.remote_s * 1e3,
            "total_ms": self.total_s * 1e3,
        }


@dataclass
class DelayModel:
    devices: list[DeviceProfile]
    config: DelayModelConfig = field(default_factory=DelayModelConfig)

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("the delay model needs at least one device")
        stages = {d.num_stages for d in self.devices}
        if len(stages) != 1:
            raise ValueError(f"devices disagree on the number of stages: {sorted(stages)}")
        self.num_stages = stages.pop()

    # --- components -----------------------------------------------------------------------------

    def local(self, device: DeviceProfile, s: int) -> float:
        if self.config.local_mode == "analytical":
            if not device.capacity_flops:
                raise ValueError(f"{device.node_id}: analytical mode needs capacity_flops")
            raw = device.local_flops(s) / device.capacity_flops
            n = device.background_processes
            return max(0.0, raw * (n + 1) - n * self.config.device_quantum_s) if s > 0 else 0.0
        return device.cumulative_latency_s.get(s, 0.0)

    def transmission(self, device: DeviceProfile, s: int) -> float:
        if s >= self.num_stages:
            return 0.0
        rtt_s = (device.rtt_ms or 0.0) / 1e3
        if self.config.transmission_mode == "rtt_only":
            return rtt_s
        if not device.throughput_mbps:
            raise ValueError(
                f"{device.node_id}: size_over_rate needs throughput_mbps (or use rtt_only)"
            )
        size_bits = device.transmitted_bytes.get(s, 0) * 8
        return size_bits / (device.throughput_mbps * 1e6) + rtt_s / 2.0

    def remote(self, vector: list[int]) -> list[float]:
        """Remote delay of every device for the full partition vector."""
        capacity = self.config.edge_capacity_flops
        offloaded = [d.offloaded_flops(s) for d, s in zip(self.devices, vector, strict=True)]
        active = [i for i, s in enumerate(vector) if s < self.num_stages]
        n_off = len(active)
        result = [0.0] * len(vector)
        if self.config.policy == "RR":
            q = self.config.rr_quantum_s
            for i in active:
                service = offloaded[i] / capacity
                result[i] = max(service, service * n_off - (n_off - 1) * q)
            return result
        order = sorted(active, key=lambda i: (offloaded[i], self.devices[i].index))
        waited = 0.0
        for i in order:
            result[i] = (offloaded[i] + waited) / capacity
            waited += offloaded[i]
        return result

    # --- vectors ----------------------------------------------------------------------------

    def check_vector(self, vector: list[int]) -> None:
        if len(vector) != len(self.devices):
            raise ValueError(f"vector has {len(vector)} entries for {len(self.devices)} devices")
        for d, s in zip(self.devices, vector, strict=True):
            if not 0 <= s <= self.num_stages:
                raise ValueError(f"{d.node_id}: s={s} outside 0..{self.num_stages}")

    def is_feasible(self, vector: list[int]) -> bool:
        return all(s <= d.max_partition_point for d, s in zip(self.devices, vector, strict=True))

    def per_device(self, vector: list[int]) -> list[DeviceDelays]:
        self.check_vector(vector)
        remote = self.remote(vector)
        return [
            DeviceDelays(d.node_id, s, self.local(d, s), self.transmission(d, s), remote[i])
            for i, (d, s) in enumerate(zip(self.devices, vector, strict=True))
        ]

    def total(self, vector: list[int]) -> float:
        return sum(d.total_s for d in self.per_device(vector))

    def variance(self, vector: list[int]) -> float:
        totals = [d.total_s for d in self.per_device(vector)]
        mean = sum(totals) / len(totals)
        return sum((t - mean) ** 2 for t in totals) / len(totals)

    def objective(self, vector: list[int], name: str) -> float:
        if name == "total_delay":
            return self.total(vector)
        if name == "variance":
            return self.variance(vector)
        raise ValueError("objective must be 'total_delay' or 'variance'")

    def all_local(self) -> list[int]:
        return [self.num_stages] * len(self.devices)


# --- loading from the pipeline's output files ----------------------------------------------------


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def profiles_from_files(
    total_averages: dict[str, Any],
    max_points: dict[str, int],
    latencies: dict[str, Any] | None = None,
    throughput: dict[str, float] | None = None,
    nodes: list[str] | None = None,
) -> list[DeviceProfile]:
    """Device profiles from total_averages_of_benchmark_results.json,
    calculated_max_partition_points.json, final_latencies_averages.json and
    optionally per-node throughput (Mbit/s)."""
    node_ids = nodes or sorted(max_points)
    profiles = []
    for index, node_id in enumerate(node_ids):
        if node_id not in total_averages:
            raise KeyError(f"no profile for node {node_id}")
        profile = total_averages[node_id]
        stage_flops = {int(i): int(v["flops"]) for i, v in profile["stages"].items()}
        cumulative = {
            int(s): float(v["cumulative_latency_ns"]["mean"]) / 1e9
            for s, v in profile["partition_points"].items()
        }
        transmitted = {
            int(s): int(v["transmitted_bytes"]) for s, v in profile["partition_points"].items()
        }
        latency = (latencies or {}).get(node_id, {})
        profiles.append(
            DeviceProfile(
                node_id=node_id,
                index=index,
                num_stages=max(stage_flops) if stage_flops else 0,
                stage_flops=stage_flops,
                cumulative_latency_s=cumulative,
                transmitted_bytes=transmitted,
                max_partition_point=int(
                    max_points.get(node_id, max(stage_flops) if stage_flops else 0)
                ),
                rtt_ms=latency.get("average_latency_ms"),
                throughput_mbps=(throughput or {}).get(node_id),
            )
        )
    return profiles

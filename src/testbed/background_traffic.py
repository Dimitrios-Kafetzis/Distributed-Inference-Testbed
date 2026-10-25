"""Background traffic on the wireless link ("additional traffic", network load).

Alongside tc-netem shaping, a scenario can load the wireless link with real
traffic: each selected device sends a UDP (or TCP) stream at a target rate to
an iperf3 server on the edge server for the duration of the experiment. The
stream runs as a tagged device-side process and is stopped at teardown.

This module builds the commands; the orchestrator starts them over SSH.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.network.throughput_measurement import DEFAULT_PORT, client_command, server_command


@dataclass(frozen=True)
class BackgroundTraffic:
    server: str
    rate_mbps: float
    duration_s: float
    udp: bool = True
    port: int = DEFAULT_PORT + 1  # separate from the throughput-measurement port
    bind: str | None = None

    def __post_init__(self) -> None:
        if self.rate_mbps <= 0 or self.duration_s <= 0:
            raise ValueError("rate_mbps and duration_s must be positive")

    def client_argv(self) -> list[str]:
        return client_command(
            self.server,
            self.duration_s,
            port=self.port,
            udp=self.udp,
            bitrate=f"{self.rate_mbps:g}M",
            bind=self.bind,
        )

    def server_argv(self) -> list[str]:
        """A persistent server (several devices may stream at once)."""
        return server_command(self.port, one_off=False)

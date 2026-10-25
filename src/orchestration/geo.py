"""The map's distance-to-latency rule.

The great-circle distance between two markers (Haversine formula) is mapped
to a synthetic per-link latency by the affine rule latency_ms = 5 + d_km,
clipped to [5, 50] ms: a 5 ms baseline plus 1 ms per kilometre of separation.
``gui/app.py`` exposes these two functions under the names the thesis cites.
"""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_000.0
MIN_LATENCY_MS = 5.0
MAX_LATENCY_MS = 50.0
MS_PER_KM = 1.0


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two coordinates (Haversine formula)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_latency(distance_m: float) -> float:
    """Synthetic link latency in milliseconds: 5 ms + 1 ms/km, clipped to [5, 50] ms."""
    latency = MIN_LATENCY_MS + MS_PER_KM * distance_m / 1000.0
    return max(MIN_LATENCY_MS, min(MAX_LATENCY_MS, latency))

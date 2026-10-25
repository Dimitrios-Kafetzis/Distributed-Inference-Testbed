"""Summary statistics used by every aggregation step."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Summary:
    mean: float
    std: float  # sample standard deviation (n - 1); 0 for a single value
    min: float
    max: float
    p95: float
    count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, q in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarise(values: Sequence[float]) -> Summary:
    if not values:
        raise ValueError("summary of an empty sequence")
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return Summary(mean, std, min(values), max(values), percentile(values, 95), n)

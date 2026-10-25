"""Every file, module, function and output name the thesis cites must exist.

Entries not yet implemented are marked xfail(strict=True) with the work package
that delivers them; when a package lands, its marks are removed. A strict
xfail that starts passing fails the suite, which is the reminder to do so.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def wp(n: int) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=f"delivered by WP{n}")


FILES = [
    pytest.param("config.example.yaml", id="config.yaml (example)"),
    pytest.param("nodes.example.json", id="nodes.json (example)"),
    pytest.param("src/config.py", id="src/config.py"),
    pytest.param("src/network/ssh_executor.py", marks=wp(3)),
    pytest.param("src/network/latency_measurement.py"),
    pytest.param("src/network/node_statistics.py"),
    pytest.param("src/network/power_measurement.py"),
    pytest.param("src/benchmarking/benchmark_results.py"),
    pytest.param("src/benchmarking/result_aggregation.py"),
    pytest.param("src/benchmarking/partition_calculator.py"),
    pytest.param("src/partitioning/delay_optimizer.py", marks=wp(6)),
    pytest.param("src/testbed/load_generator.py"),
    pytest.param("src/testbed/latency_injector.py"),
    pytest.param("src/models/inference.py", marks=wp(4)),
    pytest.param("gui/app.py", marks=wp(9)),
    pytest.param("scripts/end_device/apply_latency.sh"),
    pytest.param("scripts/end_device/train_and_run_benchmark.sh", marks=wp(4)),
    pytest.param("scripts/end_device/transfer_files.sh", marks=wp(3)),
    pytest.param("scripts/base_station/run_full_benchmarking.sh", marks=wp(7)),
]


@pytest.mark.parametrize("relative", FILES)
def test_thesis_file_exists(relative: str) -> None:
    assert (REPO / relative).exists(), relative


SYMBOLS = [
    pytest.param("src.config", "load_config", id="src.config.load_config"),
    pytest.param("src.benchmarking.benchmark_results", "BenchmarkResults"),
    pytest.param("gui.app", "calculate_distance", marks=wp(9)),
    pytest.param("gui.app", "calculate_latency", marks=wp(9)),
]


@pytest.mark.parametrize(("module", "name"), SYMBOLS)
def test_thesis_symbol_exists(module: str, name: str) -> None:
    mod = importlib.import_module(module)
    assert hasattr(mod, name), f"{module}.{name}"


OUTPUT_NAMES = [
    "total_averages_of_benchmark_results.json",
    "calculated_max_partition_points.json",
    "final_latencies_averages.json",
]


def test_output_file_names_are_the_thesis_names() -> None:
    from src.common.runs import OUTPUT_FILES

    assert list(OUTPUT_FILES) == OUTPUT_NAMES

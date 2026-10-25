from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import pytest

from src.partitioning import algorithms, delay_optimizer
from src.partitioning.algorithms import SearchLog, evaluate
from src.partitioning.algorithms.exhaustive import optimality_gap_percent
from src.partitioning.delay_model import DelayModel, DelayModelConfig, DeviceProfile

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "data" / "sample"

# Chapter 4 AlexNet FLOPs (millions) per stage, as tabulated.
CH4_FLOPS = {1: 211, 2: 0.3, 3: 448, 4: 0.2, 5: 150, 6: 112, 7: 74.9, 8: 0.1, 9: 38, 10: 17, 11: 4}


def ch4_device(node_id: str, index: int) -> DeviceProfile:
    flops = {i: int(v * 1e6) for i, v in CH4_FLOPS.items()}
    return DeviceProfile(
        node_id=node_id,
        index=index,
        num_stages=11,
        stage_flops=flops,
        cumulative_latency_s={s: 0.0 for s in range(12)},
        transmitted_bytes={s: 1000 for s in range(12)},
        max_partition_point=11,
        rtt_ms=10.0,
        capacity_flops=2e9,
    )


def test_chapter4_toy_example_sjf_remote_delays() -> None:
    """F_1^R(2) = 844, F_2^R(4) = 396, F_3^R(2) = 844 MFLOPs at C^R = 20 GFLOP/s."""
    model = DelayModel(
        [ch4_device("d1", 0), ch4_device("d2", 1), ch4_device("d3", 2)],
        DelayModelConfig("SJF", 20e9, transmission_mode="rtt_only"),
    )
    remote = [r * 1e3 for r in model.remote([2, 4, 2])]
    assert remote == pytest.approx([62.0, 19.8, 104.2], abs=0.05)
    remote_b = [r * 1e3 for r in model.remote([2, 4, 4])]
    assert remote_b == pytest.approx([81.8, 19.8, 39.6], abs=0.05)


def test_round_robin_remote_delay_formula() -> None:
    model = DelayModel(
        [ch4_device("d1", 0), ch4_device("d2", 1), ch4_device("d3", 2)],
        DelayModelConfig("RR", 20e9, rr_quantum_s=0.001, transmission_mode="rtt_only"),
    )
    remote = model.remote([2, 4, 11])  # d3 all local -> N_off = 2
    f1 = model.devices[0].offloaded_flops(2) / 20e9
    f2 = model.devices[1].offloaded_flops(4) / 20e9
    assert remote[0] == pytest.approx(f1 * 2 - 0.001)
    assert remote[1] == pytest.approx(f2 * 2 - 0.001)
    assert remote[2] == 0.0
    assert model.remote([11, 11, 11]) == [0.0, 0.0, 0.0]


def test_transmission_modes_and_analytical_local() -> None:
    device = ch4_device("d1", 0)
    device.throughput_mbps = 20.0
    device.transmitted_bytes[1] = 1_161_600
    rtt_only = DelayModel([device], DelayModelConfig("SJF", 20e9, transmission_mode="rtt_only"))
    assert rtt_only.transmission(device, 1) == pytest.approx(0.010)
    assert rtt_only.transmission(device, 11) == 0.0
    rate = DelayModel([device], DelayModelConfig("SJF", 20e9, transmission_mode="size_over_rate"))
    assert rate.transmission(device, 1) == pytest.approx(1_161_600 * 8 / 20e6 + 0.005)
    device.throughput_mbps = None
    with pytest.raises(ValueError, match="throughput"):
        rate.transmission(device, 1)
    analytical = DelayModel(
        [device],
        DelayModelConfig(
            "SJF",
            20e9,
            transmission_mode="rtt_only",
            local_mode="analytical",
            device_quantum_s=0.01,
        ),
    )
    device.background_processes = 2
    expected = device.local_flops(2) / 2e9 * 3 - 2 * 0.01
    assert analytical.local(device, 2) == pytest.approx(expected)
    assert analytical.local(device, 0) == 0.0
    with pytest.raises(ValueError):
        DelayModelConfig("FCFS")


def random_model(seed: int, n: int = 3, stages: int = 5, policy: str = "SJF") -> DelayModel:
    rng = random.Random(seed)
    devices = []
    for i in range(n):
        flops = {k: rng.randint(10, 100) * 10**6 for k in range(1, stages + 1)}
        cumulative = {0: 0.0}
        for k in range(1, stages + 1):
            cumulative[k] = cumulative[k - 1] + flops[k] / rng.uniform(1e9, 3e9)
        devices.append(
            DeviceProfile(
                node_id=f"d{i}",
                index=i,
                num_stages=stages,
                stage_flops=flops,
                cumulative_latency_s=cumulative,
                transmitted_bytes={k: rng.randint(1_000, 500_000) for k in range(stages + 1)},
                max_partition_point=rng.randint(2, stages),
                rtt_ms=rng.uniform(2, 30),
            )
        )
    return DelayModel(
        devices, DelayModelConfig(policy, rng.uniform(5e9, 4e10), transmission_mode="rtt_only")
    )


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("objective", ["total_delay", "variance"])
def test_greedy_is_monotone_and_ends_in_a_local_optimum(seed: int, objective: str) -> None:
    model = random_model(seed, policy="RR" if seed % 2 else "SJF")
    initial = [d.max_partition_point for d in model.devices]
    result = algorithms.get_algorithm("greedy")(model, objective, initial)
    assert result.value <= model.objective(initial, objective) + 1e-12
    assert model.is_feasible(result.vector)
    for i, device in enumerate(model.devices):
        for direction in (-1, 1):
            s = result.vector[i] + direction
            if 0 <= s <= device.max_partition_point:
                neighbour = list(result.vector)
                neighbour[i] = s
                assert model.objective(neighbour, objective) >= result.value - 1e-12


@pytest.mark.parametrize("seed", range(4))
def test_exhaustive_equals_brute_force_and_reports_gap(seed: int) -> None:
    model = random_model(seed, n=3, stages=4)
    result = algorithms.get_algorithm("exhaustive")(model, "total_delay", initial=model.all_local())
    ranges = [d.feasible_points for d in model.devices]
    brute = min(model.total(list(v)) for v in itertools.product(*ranges))
    assert result.value == pytest.approx(brute)
    assert result.extra["vectors"] == len(list(itertools.product(*ranges)))
    greedy = algorithms.get_algorithm("greedy")(model, "total_delay", model.all_local())
    assert greedy.value >= result.value - 1e-12
    assert optimality_gap_percent(greedy.value, result.value) >= 0.0
    assert result.extra["optimality_gap_percent"] >= 0.0


def test_constraints_make_vectors_infeasible() -> None:
    model = random_model(1)
    vector = [d.max_partition_point for d in model.devices]  # the most local feasible vector
    assert evaluate(model, vector, "total_delay", None) < float("inf")
    tight = [1e-9] * len(model.devices)
    assert evaluate(model, vector, "total_delay", tight) == float("inf")
    limited = min(model.devices, key=lambda d: d.max_partition_point)
    if limited.max_partition_point < model.num_stages:
        beyond = list(vector)
        beyond[limited.index] = limited.max_partition_point + 1
        assert evaluate(model, beyond, "total_delay", None) == float("inf")
    greedy = algorithms.get_algorithm("greedy")(
        model, "total_delay", [0] * len(model.devices), max_total_delays_s=tight
    )
    assert greedy.value == float("inf")
    with pytest.raises(ValueError):
        evaluate(model, vector, "speed", None)


def test_trend_based_improves_and_respects_time_budget() -> None:
    model = random_model(3, n=4, stages=8)
    initial = [0] * 4
    result = algorithms.get_algorithm("trend_based")(
        model, "total_delay", initial, num_solutions=3, max_time_s=5.0
    )
    assert result.value <= model.total(initial) + 1e-12
    assert model.is_feasible(result.vector)
    quick = algorithms.get_algorithm("trend_based")(model, "total_delay", initial, max_time_s=0.0)
    assert quick.vector == initial


def test_search_log_records_every_evaluation(tmp_path: Path) -> None:
    model = random_model(2, n=2, stages=3)
    log = SearchLog(tmp_path / "search_log.txt")
    log.start(model, "greedy", "total_delay")
    result = algorithms.get_algorithm("greedy")(model, "total_delay", [1, 1], log=log)
    log.close()
    text = (tmp_path / "search_log.txt").read_text()
    assert text.startswith("Partition search log: algorithm=greedy")
    assert log.evaluations == result.evaluations
    assert text.count("\n[") == result.evaluations


def test_initial_vector_rules() -> None:
    devices = [ch4_device("a", 0), ch4_device("b", 1)]
    devices[1].max_partition_point = 4
    assert delay_optimizer.initial_vector(devices, None, None, 11) == [1, 1]
    assert delay_optimizer.initial_vector(devices, "50%", "percentage", 11) == [6, 4]
    assert delay_optimizer.initial_vector(devices, "[3, 2]", "explicit", 11) == [3, 2]
    assert delay_optimizer.initial_vector(devices, '{"b": 2, "a": 9}', "explicit", 11) == [9, 2]
    with pytest.raises(ValueError):
        delay_optimizer.initial_vector(devices, "[3]", "explicit", 11)
    assert delay_optimizer.max_delays_seconds("[100, 200]", devices) == [0.1, 0.2]
    assert delay_optimizer.max_delays_seconds('{"a": 50, "b": 75}', devices) == [0.05, 0.075]


def test_algorithm_plugin_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plugin = tmp_path / "toy_algorithm.py"
    plugin.write_text(
        "from src.partitioning.algorithms import SearchResult\nNAME = 'all_local'\n"
        "def run(model, objective, initial, max_total_delays_s=None, log=None, **_):\n"
        "    v = model.all_local()\n"
        "    return SearchResult(v, model.objective(v, objective), objective, NAME, 1, 0)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(algorithms.MODULES_ENV, "toy_algorithm")
    algorithms.reset_for_tests()
    try:
        assert "all_local" in algorithms.names()
        model = random_model(0)
        assert (
            algorithms.get_algorithm("all_local")(model, "total_delay", [0, 0, 0]).vector
            == model.all_local()
        )
    finally:
        monkeypatch.delenv(algorithms.MODULES_ENV)
        algorithms.reset_for_tests()
    with pytest.raises(KeyError, match="unknown algorithm"):
        algorithms.get_algorithm("magic")


def sample_args(tmp_path: Path) -> list[str]:
    return [
        str(SAMPLE / "calculated_max_partition_points.json"),
        str(SAMPLE / "total_averages_of_benchmark_results.json"),
        str(SAMPLE / "final_latencies_averages.json"),
        "--output-dir",
        str(tmp_path / "opt"),
        "--edge-capacity",
        "2e10",
    ]


def test_cli_optimise_on_sample_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    delay_optimizer.main(
        [
            *sample_args(tmp_path),
            "--scheduling_algorithm",
            "SJF",
            "--mode",
            "optimize",
            "--partition_scheme",
            "50%",
            "--partition_scheme_mode",
            "percentage",
        ]
    )
    out = capsys.readouterr().out
    assert "Optimized Partition Scheme:" in out
    result = json.loads((tmp_path / "opt" / "optimized_partition_scheme.json").read_text())
    assert set(result["by_node"]) == {"device1", "device2"}
    assert result["algorithm"] == "greedy" and result["policy"] == "SJF"
    assert result["inputs"]["total_averages"]["sha256"]
    assert (tmp_path / "opt" / "search_log.txt").exists() and (
        tmp_path / "opt" / "inputs.json"
    ).exists()
    assert result["by_node"]["device2"] == 0  # the sample device2 has max partition point 0


def test_cli_modes_default_exhaustive_trend_and_infeasible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    delay_optimizer.main([*sample_args(tmp_path), "--mode", "default"])
    assert (tmp_path / "opt" / "delays.json").exists() and "s= 0" in capsys.readouterr().out
    delay_optimizer.main(
        [*sample_args(tmp_path), "--mode", "exhaustive_search", "--objective", "variance"]
    )
    result = json.loads((tmp_path / "opt" / "optimized_partition_scheme.json").read_text())
    assert result["algorithm"] == "exhaustive" and result["objective"] == "variance"
    delay_optimizer.main(
        [
            *sample_args(tmp_path),
            "--mode",
            "trend_based_optimize",
            "--num_solutions",
            "2",
            "--max_time",
            "2",
        ]
    )
    assert (
        json.loads((tmp_path / "opt" / "optimized_partition_scheme.json").read_text())["algorithm"]
        == "trend_based"
    )
    with pytest.raises(SystemExit) as exc:
        delay_optimizer.main([*sample_args(tmp_path), "--max_total_delays", "[0.001, 0.001]"])
    assert exc.value.code == 2

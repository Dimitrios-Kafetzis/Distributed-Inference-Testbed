from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.benchmarking import benchmark_results as br
from src.models import inference
from src.models.fake import FakeStagesAdapter
from src.models.registry import deserialize_array

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "end_device" / "train_and_run_benchmark.sh"


def test_prepare_writes_weights_and_image(tmp_path: Path) -> None:
    weights = tmp_path / "models" / "fake.npz"
    image = tmp_path / "data" / "test_image.png"
    inference.main(
        [
            "prepare",
            "--model",
            "fake",
            "--weights",
            str(weights),
            "--image",
            str(image),
            "--seed",
            "4",
        ]
    )
    assert weights.exists() and image.exists()
    mtime = weights.stat().st_mtime
    inference.main(
        [
            "prepare",
            "--model",
            "fake",
            "--weights",
            str(weights),
            "--image",
            str(image),
            "--seed",
            "4",
        ]
    )
    assert weights.stat().st_mtime == mtime  # not rewritten without --force


def test_benchmark_writes_records(tmp_path: Path) -> None:
    out = tmp_path / "results"
    inference.main(
        [
            "benchmark",
            "--model",
            "fake",
            "--output-dir",
            str(out),
            "--run-id",
            "r1",
            "--node-id",
            "d1",
            "--repetitions",
            "2",
            "--warmup",
            "1",
            "--sampler-interval",
            "0.002",
        ]
    )
    files = sorted(out.glob("*.json"))
    assert [f.name for f in files] == ["benchmark_r1_rep0.json", "benchmark_r1_rep1.json"]
    record = br.load_benchmark(files[1])
    assert record.repetition == 1 and record.warmup_runs == 1 and record.model == "fake"
    assert record.metadata["flops_per_second"] > 0
    assert len(record.partition_points) == len(record.stages) + 1


def test_partial_and_complete_match_the_full_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    weights = tmp_path / "fake.npz"
    inference.main(["prepare", "--model", "fake", "--weights", str(weights), "--seed", "2"])
    cut = tmp_path / "cut.npy"
    inference.main(
        [
            "partial",
            "--model",
            "fake",
            "--weights",
            str(weights),
            "--s",
            "2",
            "--output",
            str(cut),
            "--seed",
            "2",
        ]
    )
    partial_info = json.loads(capsys.readouterr().out)
    assert partial_info["s"] == 2 and partial_info["shape"] == [1, 256]
    final = tmp_path / "out.npy"
    inference.main(
        [
            "complete",
            "--model",
            "fake",
            "--weights",
            str(weights),
            "--s",
            "2",
            "--input",
            str(cut),
            "--output",
            str(final),
            "--seed",
            "2",
        ]
    )
    complete_info = json.loads(capsys.readouterr().out)
    adapter = FakeStagesAdapter()
    adapter.load_weights(weights)
    expected = adapter.run_stages(adapter.make_input(2), 1, 5)
    np.testing.assert_allclose(deserialize_array(final.read_bytes()), expected)
    assert complete_info["argmax"] == [int(expected.argmax())]


def test_capacity_prints_flops_per_second(capsys: pytest.CaptureFixture[str]) -> None:
    inference.main(["capacity", "--model", "fake", "--repetitions", "2", "--warmup", "0"])
    data = json.loads(capsys.readouterr().out)
    assert data["model"] == "fake" and data["flops_per_second"] > 0 and data["repetitions"] == 2


def test_thread_count_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TESTBED_THREADS", "2")
    assert inference.apply_thread_count() == 2
    monkeypatch.delenv("TESTBED_THREADS")
    assert inference.apply_thread_count() is None


def test_train_and_run_benchmark_script_both_mode(tmp_path: Path) -> None:
    env = dict(
        os.environ,
        TESTBED_PYTHON=sys.executable,
        TESTBED_ED_DATA_DIR=str(tmp_path / "data"),
        TESTBED_NODE_ID="dev9",
        PYTHONPATH=str(REPO),
    )
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "both",
            str(tmp_path / "models"),
            "fake",
            str(tmp_path / "results"),
            "--run-id",
            "run5",
            "--repetitions",
            "1",
            "--warmup",
            "0",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "models" / "fake.npz").exists()
    assert (tmp_path / "data" / "test_image_fake.png").exists()
    record = br.load_benchmark(tmp_path / "results" / "benchmark_run5_rep0.json")
    assert record.node_id == "dev9"
    missing = subprocess.run(
        ["bash", str(SCRIPT), "benchmark", str(tmp_path / "nowhere"), "fake", str(tmp_path / "r")],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
    )
    assert missing.returncode == 1 and "weights missing" in missing.stderr
    bad_mode = subprocess.run(
        ["bash", str(SCRIPT), "evaluate", "a", "b", "c"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
    )
    assert bad_mode.returncode == 1 and "invalid mode" in bad_mode.stderr

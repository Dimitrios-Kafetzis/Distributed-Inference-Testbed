"""Input sources, the model catalogue, and the partition-equivalence check (fake model)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.models import inference, inputs, weights
from src.models.fake import FakeStagesAdapter
from src.models.registry import deserialize_array

REPO = Path(__file__).resolve().parents[1]


def fake() -> FakeStagesAdapter:
    adapter = FakeStagesAdapter(input_elements=16, time_scale=0.0)
    adapter.build(1)
    return adapter


# --- input sources -------------------------------------------------------------------------------


def test_synthetic_and_numpy_sources(tmp_path: Path) -> None:
    adapter = fake()
    sample = inputs.load_sample(None, adapter, seed=3)
    assert sample.array.shape == (1, 16) and sample.description == {
        "source": "synthetic",
        "seed": 3,
    }
    array = np.arange(16, dtype=np.float32)
    np.save(tmp_path / "x.npy", array)
    sample = inputs.load_sample({"source": "numpy_file", "path": "x.npy"}, adapter, 0, tmp_path)
    assert sample.array.shape == (1, 16) and sample.label is None
    assert sample.description["path"] == str(tmp_path / "x.npy")
    np.save(tmp_path / "bad.npy", np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(inputs.InputError, match="does not match"):
        inputs.load_sample({"source": "numpy_file", "path": "bad.npy"}, adapter, 0, tmp_path)
    with pytest.raises(inputs.InputError, match="unknown input source"):
        inputs.load_sample({"source": "camera"}, adapter, 0)


def test_image_folder_with_class_directories_gives_labels(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    adapter = FakeStagesAdapter(time_scale=0.0)
    adapter.build(0)
    adapter.input_shape = (8, 8, 3)  # an image-shaped input for the loader
    for label, name in enumerate(["cat", "dog"]):
        (tmp_path / "set" / name).mkdir(parents=True)
        for k in range(2):
            Image.new("RGB", (12, 10), (label * 100, k * 50, 0)).save(
                tmp_path / "set" / name / f"img{k}.png"
            )
    listing = inputs.image_folder_listing(tmp_path / "set")
    assert [(p.name, label) for p, label in listing] == [
        ("img0.png", 0),
        ("img1.png", 0),
        ("img0.png", 1),
        ("img1.png", 1),
    ]
    sample = inputs.load_sample(
        {"source": "image_folder", "path": "set", "index": 2}, adapter, 0, tmp_path
    )
    assert sample.array.shape == (1, 8, 8, 3) and sample.label == 1
    assert sample.description["images"] == 4 and sample.description["index"] == 2
    # The index wraps, so a sweep over indices never runs out of images.
    assert (
        inputs.load_sample(
            {"source": "image_folder", "path": "set", "index": 5}, adapter, 0, tmp_path
        ).label
        == 0
    )
    (tmp_path / "flat").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "flat" / "a.jpg")
    assert (
        inputs.load_sample({"source": "image_folder", "path": "flat"}, adapter, 0, tmp_path).label
        is None
    )
    with pytest.raises(inputs.InputError, match="no image"):
        (tmp_path / "empty").mkdir()
        inputs.image_folder_listing(tmp_path / "empty")


def test_text_prompt_needs_a_tokenizer() -> None:
    with pytest.raises(inputs.InputError, match="tokenizer"):
        inputs.load_sample({"source": "text_prompt", "text": "hello"}, fake(), 0)
    with pytest.raises(inputs.InputError, match="text"):
        inputs.load_sample({"source": "text_prompt"}, fake(), 0)


# --- verify command ------------------------------------------------------------------------------


def test_verify_reports_every_partition_point(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "verification.json"
    inference.main(
        [
            "verify",
            "--model",
            "fake",
            "--model-args",
            '{"input_elements": 16, "time_scale": 0}',
            "--output",
            str(output),
            "--node-id",
            "device1",
        ]
    )
    result = json.loads(output.read_text())
    assert result["ok"] is True and result["stages"] == 5 and result["node_id"] == "device1"
    assert [p["s"] for p in result["partition_points"]] == [0, 1, 2, 3, 4, 5]
    assert all(p["argmax_equal"] for p in result["partition_points"])
    assert result["max_rel_deviation"] == 0.0 and result["input"]["source"] == "synthetic"
    assert result["top1"]["correct"] is None
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True


def test_verify_fails_on_a_deviation(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = fake()
    original = adapter.run_stages

    def drifting(x: np.ndarray, start: int, stop: int) -> np.ndarray:  # type: ignore[type-arg]
        out = original(x, start, stop)
        return out * 1.01 if start > 1 else out  # the remote half is slightly off

    adapter.run_stages = drifting  # type: ignore[method-assign]
    result = inference.verify_partition(adapter, adapter.make_input(0), None, 1e-4)
    assert result["ok"] is False and result["max_rel_deviation"] > 1e-4
    # s = 0 runs stages 1..S in one go, like the reference; from s = 1 on the remote half
    # starts at a later stage and drifts; s = S runs everything locally, like the reference.
    assert result["partition_points"][0]["max_rel_deviation"] == 0
    assert result["partition_points"][1]["max_rel_deviation"] > 0
    assert result["partition_points"][-1]["max_rel_deviation"] == 0


def test_input_source_reaches_the_benchmark_record(tmp_path: Path) -> None:
    np.save(tmp_path / "x.npy", np.ones(16, dtype=np.float32))
    inference.main(
        [
            "benchmark",
            "--model",
            "fake",
            "--model-args",
            '{"input_elements": 16, "time_scale": 0}',
            "--input-source",
            json.dumps({"source": "numpy_file", "path": str(tmp_path / "x.npy")}),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--node-id",
            "d1",
            "--repetitions",
            "1",
            "--warmup",
            "0",
        ]
    )
    record = json.loads((tmp_path / "out" / "benchmark_r1_rep0.json").read_text())
    assert record["metadata"]["input"] == {
        "source": "numpy_file",
        "path": str(tmp_path / "x.npy"),
    }
    out = tmp_path / "cut.npy"
    inference.main(
        [
            "partial",
            "--model",
            "fake",
            "--model-args",
            '{"input_elements": 16, "time_scale": 0}',
            "--input-source",
            json.dumps({"source": "numpy_file", "path": str(tmp_path / "x.npy")}),
            "--s",
            "0",
            "--output",
            str(out),
        ]
    )
    np.testing.assert_array_equal(deserialize_array(out.read_bytes()), np.ones((1, 16), np.float32))


# --- model catalogue -----------------------------------------------------------------------------


def write_catalogue(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / weights.CATALOGUE_NAME).write_text(
        "models:\n"
        "  - name: fake\n    kind: npz\n    path: fake.npz\n    seed: 7\n"
        "    model_args: {input_elements: 16}\n    source: seeded\n    licence: none\n"
        "  - name: tinyllama\n    kind: hf_checkpoint\n    path: TinyLlama__x\n"
        "    repo_id: TinyLlama/x\n    licence: Apache-2.0\n"
    )


def test_catalogue_fetch_npz_checksums_and_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_catalogue(tmp_path)
    entries = weights.load_catalogue(tmp_path)
    assert [e.name for e in entries] == ["fake", "tinyllama"]
    assert weights.main(["list", "--models-dir", str(tmp_path)]) == 0
    listed = capsys.readouterr().out
    assert "fake             npz             absent" in listed
    assert weights.main(["fetch", "fake", "--models-dir", str(tmp_path)]) == 0
    assert (tmp_path / "fake.npz").exists()
    sums = weights.read_checksums(tmp_path / weights.CHECKSUMS_NAME)
    assert set(sums) == {"fake.npz"}
    assert weights.verify(entries[0], tmp_path) == []
    assert weights.status(entries[0], tmp_path)["verified"] is True
    (tmp_path / "fake.npz").write_bytes(b"corrupt")
    assert weights.verify(entries[0], tmp_path) == ["fake: fake.npz changed"]
    assert weights.main(["verify", "fake", "--models-dir", str(tmp_path)]) == 1
    # Checkpoint directories carry their own SHA256SUMS.
    checkpoint = tmp_path / "TinyLlama__x"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    assert weights.verify(entries[1], tmp_path) == [
        f"tinyllama: {checkpoint / 'SHA256SUMS'} is missing"
    ]
    weights.checksums(entries[1], tmp_path)
    assert weights.verify(entries[1], tmp_path) == []
    (checkpoint / "config.json").write_text("{ }")
    assert weights.verify(entries[1], tmp_path) == ["tinyllama: changed config.json"]
    # Downloads need an explicit permission, as in the provisioning scripts.
    with pytest.raises(weights.WeightsError, match="allow-upstream"):
        weights.fetch(entries[1], tmp_path, allow_upstream=False)
    assert weights.main(["fetch", "--models-dir", str(tmp_path)]) == 1  # no names, no --all
    assert weights.main(["verify", "nope", "--models-dir", str(tmp_path)]) == 1


def test_example_catalogue_loads_and_names_registered_models(tmp_path: Path) -> None:
    from src.models import registry

    (tmp_path / weights.CATALOGUE_NAME).write_text((REPO / "examples" / "models.yaml").read_text())
    entries = weights.load_catalogue(tmp_path)
    names = registry.names()
    assert all(e.name in names for e in entries), [e.name for e in entries if e.name not in names]
    assert all(e.licence for e in entries)
    assert all(e.repo_id for e in entries if e.kind == "hf_checkpoint")

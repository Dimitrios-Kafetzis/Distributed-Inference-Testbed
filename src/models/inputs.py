"""Input sources: what the profiler and the inference client feed to the model.

A scenario's ``input`` block names a source; the device resolves it against its
data directory. Every source is deterministic (a seed, a file, or the k-th
file of a sorted listing), and describes itself so that the measurement
record says what was measured.

Sources: ``synthetic`` (the adapter's seeded input, the default),
``numpy_file`` (a ``.npy`` array), ``image_file`` (a PNG or JPEG resized to the
model's input), ``image_folder`` (the ``index``-th image of a directory, or of
class sub-directories, which also gives a label), ``text_prompt`` (a prompt
tokenised with a checkpoint's tokeniser to the decoder's sequence length).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.models.registry import Array, ModelAdapter

SOURCES = ("synthetic", "numpy_file", "image_file", "image_folder", "text_prompt")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


class InputError(ValueError):
    pass


@dataclass
class InputSample:
    array: Array
    label: int | None = None
    description: dict[str, Any] = field(default_factory=dict)


def _resolve(path: str, data_dir: Path | None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and data_dir is not None:
        candidate = data_dir / candidate
    return candidate


def _batched(array: Array, adapter: ModelAdapter) -> Array:
    if array.ndim == len(adapter.input_shape) + 1:
        return array
    if array.ndim == len(adapter.input_shape):
        return np.asarray(array[np.newaxis, ...])
    raise InputError(
        f"input of shape {array.shape} does not match the model input {adapter.input_shape}"
    )


def image_folder_listing(folder: Path) -> list[tuple[Path, int | None]]:
    """Images in a directory in sorted order. Sub-directories are classes (sorted), and
    each image carries its class index; a flat directory carries no label."""
    if not folder.is_dir():
        raise InputError(f"{folder} is not a directory")
    classes = sorted(p for p in folder.iterdir() if p.is_dir())
    listing: list[tuple[Path, int | None]] = []
    if classes:
        for label, class_dir in enumerate(classes):
            for image in sorted(class_dir.iterdir()):
                if image.suffix.lower() in IMAGE_SUFFIXES:
                    listing.append((image, label))
    else:
        for image in sorted(folder.iterdir()):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                listing.append((image, None))
    if not listing:
        raise InputError(f"{folder} holds no image ({', '.join(IMAGE_SUFFIXES)})")
    return listing


def load_sample(
    spec: dict[str, Any] | None,
    adapter: ModelAdapter,
    seed: int,
    data_dir: Path | None = None,
) -> InputSample:
    spec = dict(spec or {})
    source = spec.get("source", "synthetic")
    if source not in SOURCES:
        raise InputError(f"unknown input source {source!r}; one of {', '.join(SOURCES)}")
    if source == "synthetic":
        return InputSample(adapter.make_input(seed), None, {"source": "synthetic", "seed": seed})
    if source == "numpy_file":
        path = _resolve(spec["path"], data_dir)
        array = np.load(path, allow_pickle=False)
        return InputSample(
            _batched(np.asarray(array), adapter), None, {"source": source, "path": str(path)}
        )
    if source == "image_file":
        from src.models.images import load_image

        path = _resolve(spec["path"], data_dir)
        return InputSample(
            load_image(path, adapter.input_shape), None, {"source": source, "path": str(path)}
        )
    if source == "image_folder":
        from src.models.images import load_image

        folder = _resolve(spec["path"], data_dir)
        listing = image_folder_listing(folder)
        index = int(spec.get("index", 0)) % len(listing)
        path, label = listing[index]
        return InputSample(
            load_image(path, adapter.input_shape),
            label,
            {
                "source": source,
                "path": str(path),
                "index": index,
                "label": label,
                "images": len(listing),
            },
        )
    # text_prompt
    text = spec.get("text")
    if not text:
        raise InputError("text_prompt needs `text`")
    seq_len = int(adapter.input_shape[0])
    tokenizer_ref = spec.get("tokenizer") or getattr(adapter, "checkpoint", None)
    if not tokenizer_ref:
        raise InputError("text_prompt needs `tokenizer` (a checkpoint directory or name)")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(_resolve(tokenizer_ref, data_dir)))
    ids = list(tokenizer(text)["input_ids"])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    truncated = len(ids) > seq_len
    ids = (ids + [pad] * seq_len)[:seq_len]
    return InputSample(
        np.asarray([ids], dtype=np.int64),
        None,
        {
            "source": source,
            "tokenizer": str(tokenizer_ref),
            "tokens": seq_len,
            "truncated": truncated,
            "text": text[:200],
        },
    )

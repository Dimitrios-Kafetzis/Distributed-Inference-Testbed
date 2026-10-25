"""Test input images: a deterministic synthetic image, and image loading."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.models.registry import Array


def synthetic_pixels(shape: Sequence[int], seed: int) -> Array:
    """Smooth, seeded colour gradients plus noise, as uint8 H x W x C."""
    height, width = int(shape[0]), int(shape[1])
    channels = int(shape[2]) if len(shape) > 2 else 1
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    planes = []
    for c in range(channels):
        phase = rng.uniform(0, 2 * np.pi)
        base = 127.5 + 100.0 * np.sin(2 * np.pi * (xx / width + c / channels) + phase) * np.cos(
            2 * np.pi * yy / height
        )
        noise = rng.normal(0, 8.0, size=(height, width))
        planes.append(np.clip(base + noise, 0, 255))
    stacked: Array = np.stack(planes, axis=-1).astype(np.uint8)
    return stacked


NUMPY_MAGIC = b"\x93NUMPY"


def is_image_shape(shape: Sequence[int]) -> bool:
    """H x W or H x W x C with 1 or 3 channels."""
    return len(shape) == 2 or (len(shape) == 3 and int(shape[2]) in (1, 3))


def write_synthetic_image(path: Path, input_shape: Sequence[int], seed: int) -> Path:
    """Write a PNG for image-shaped inputs; other inputs (vectors, sequences) are
    written as a NumPy array at the same path, whatever its extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not is_image_shape(input_shape):
        rng = np.random.default_rng(seed)
        array = rng.standard_normal((1, *input_shape)).astype(np.float32)
        with path.open("wb") as fh:
            np.save(fh, array, allow_pickle=False)
        return path
    from PIL import Image

    pixels = synthetic_pixels(input_shape, seed)
    image = Image.fromarray(pixels.squeeze(-1) if pixels.shape[-1] == 1 else pixels)
    image.save(path)
    return path


def is_numpy_file(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(len(NUMPY_MAGIC)) == NUMPY_MAGIC


def load_image(path: Path, input_shape: Sequence[int]) -> Array:
    """Load, resize to the model's spatial size, scale to [0, 1], add the batch axis."""
    from PIL import Image

    height, width = int(input_shape[0]), int(input_shape[1])
    channels = int(input_shape[2]) if len(input_shape) > 2 else 1
    image = Image.open(path).convert("RGB" if channels == 3 else "L").resize((width, height))
    array: Array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    if channels == 1:
        array = array[..., np.newaxis]
    batched: Array = array[np.newaxis, ...]
    return batched

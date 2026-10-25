"""Random-seed pinning.

The seeds of every stochastic library used by the inference pipeline are set
at the start of every experiment to the value recorded in the scenario
configuration (Chapter 3, Appendix B). Libraries that are not installed are
skipped and reported.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import random


def seed_everything(seed: int) -> dict[str, bool]:
    """Seed ``random``, NumPy, PyTorch and TensorFlow; return what was seeded.

    ``PYTHONHASHSEED`` is exported for child processes; the current interpreter
    keeps the hash seed it started with, so the orchestrator also passes the
    variable when it starts device-side processes.
    """
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    seeded: dict[str, bool] = {"random": True, "hashseed": True}
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    numpy = _optional("numpy")
    if numpy is not None:
        numpy.random.seed(seed)
    seeded["numpy"] = numpy is not None

    torch = _optional("torch")
    if torch is not None:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    seeded["torch"] = torch is not None

    # This may be the first TensorFlow import of the process: same defaults as src/models/tf.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    tensorflow = _optional("tensorflow")
    if tensorflow is not None:
        tensorflow.random.set_seed(seed)
        with contextlib.suppress(AttributeError):
            tensorflow.config.experimental.enable_op_determinism()
    seeded["tensorflow"] = tensorflow is not None
    return seeded


def _optional(name: str):  # type: ignore[no-untyped-def]
    try:
        return importlib.import_module(name)
    except ImportError:
        return None

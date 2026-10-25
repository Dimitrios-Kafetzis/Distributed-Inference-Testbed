"""A third-party model adapter, loaded through TESTBED_MODEL_MODULES in tests."""

from __future__ import annotations

from src.models.fake import FakeStagesAdapter
from src.models.registry import register


class ToyAdapter(FakeStagesAdapter):
    name = "toy"

    def __init__(self) -> None:
        super().__init__([("a", 0.0, 0.0, 8, 100), ("b", 0.0, 0.0, 4, 50)], input_elements=16)


register("toy", ToyAdapter)

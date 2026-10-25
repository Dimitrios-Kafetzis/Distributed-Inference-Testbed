"""Shared fixtures."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.common.clock import FakeClock
from src.common.runner import FakeRunner
from src.common.sysfs import SysFs

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every TESTBED_* variable so tests start from the defaults."""
    for name in list(os.environ):
        if name.startswith("TESTBED_"):
            monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake_runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def sysfs(tmp_path: Path) -> SysFs:
    return SysFs(tmp_path / "root")


@pytest.fixture
def example_nodes() -> dict[str, object]:
    with (REPO_ROOT / "nodes.example.json").open(encoding="utf-8") as fh:
        data: dict[str, object] = json.load(fh)
    return data

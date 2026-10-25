"""Scenarios: one configuration object per experiment, versioned and replayable.

All parameters of an experiment (network conditions, compute load, model,
partition strategy, heterogeneity, monitoring, seeds, repetitions) are held in
one JSON object validated against ``docs/schemas/scenario.schema.json``.
NNPM stores every saved scenario as a new version under
``scenarios/<name>/v<N>.json`` with its content hash; identical content does
not create a new version. A run records the version it used, so the same
scenario can be replayed later.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from src.common.runs import write_json_atomic
from src.config import REPO_ROOT

SCHEMA_PATH = REPO_ROOT / "docs" / "schemas" / "scenario.schema.json"
DEFAULT_STORE = REPO_ROOT / "scenarios"


class ScenarioError(ValueError):
    pass


def load_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        schema: dict[str, Any] = json.load(fh)
    return schema


def _apply_defaults(instance: Any, schema: dict[str, Any], root: dict[str, Any]) -> Any:
    """Fill in the schema's ``default`` values (one level of $ref resolution)."""
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        schema = root["$defs"][ref]
    if schema.get("type") == "object" and isinstance(instance, dict):
        for key, prop in schema.get("properties", {}).items():
            if key not in instance and "default" in prop:
                instance[key] = copy.deepcopy(prop["default"])
            if key in instance:
                instance[key] = _apply_defaults(instance[key], prop, root)
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            for key, value in instance.items():
                if key not in schema.get("properties", {}):
                    instance[key] = _apply_defaults(value, extra, root)
        # Absent sub-objects whose properties all have defaults are created from them.
        for key, prop in schema.get("properties", {}).items():
            props = prop.get("properties", {})
            if (
                key not in instance
                and prop.get("type") == "object"
                and props
                and not prop.get("required")
                and all("default" in p for p in props.values())
            ):
                instance[key] = _apply_defaults({}, prop, root)
    return instance


def validate(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and return a copy with the defaults filled in."""
    schema = load_schema()
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as exc:
        location = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise ScenarioError(f"invalid scenario at {location}: {exc.message}") from exc
    filled = copy.deepcopy(data)
    filled.setdefault("schema_version", 1)
    result: dict[str, Any] = _apply_defaults(filled, schema, schema)
    return result


def load_scenario_file(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) if Path(path).suffix in (".yaml", ".yml") else json.loads(text)
    if not isinstance(data, dict):
        raise ScenarioError(f"{path}: a scenario must be a mapping")
    return validate(data)


def content_hash(data: dict[str, Any]) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def get_path(data: dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        node = node[part]
    return node


def set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: Any = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def expand_sweep(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """One scenario per sweep value (the sweep field itself is removed), or the scenario."""
    sweep = scenario.get("sweep")
    if not sweep:
        return [scenario]
    variants = []
    for value in sweep["values"]:
        variant = copy.deepcopy(scenario)
        del variant["sweep"]
        set_path(variant, sweep["path"], value)
        variant["name"] = f"{scenario['name']}__{sweep['path'].rsplit('.', 1)[-1]}-{value}"
        variants.append(validate(variant))
    return variants


def for_node(section: dict[str, Any] | None, node_id: str) -> dict[str, Any]:
    """Per-node settings: the ``default`` entry overlaid with the node's own entry."""
    if not section:
        return {}
    merged: dict[str, Any] = copy.deepcopy(section.get("default", {}))
    merged.update(copy.deepcopy(section.get(node_id, {})))
    return merged


@dataclass(frozen=True)
class ScenarioVersion:
    name: str
    version: int
    sha256: str
    path: Path

    def load(self) -> dict[str, Any]:
        with self.path.open(encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        return dict(data)


class ScenarioStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_STORE

    def versions(self, name: str) -> list[ScenarioVersion]:
        folder = self.root / name
        if not folder.exists():
            return []
        found = []
        for path in sorted(folder.glob("v*.json"), key=lambda p: int(p.stem[1:])):
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            found.append(ScenarioVersion(name, int(path.stem[1:]), data.get("_sha256", ""), path))
        return found

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and any(p.glob("v*.json")))

    def save(self, data: dict[str, Any]) -> ScenarioVersion:
        """Validate and store as the next version; an unchanged scenario returns its version."""
        scenario = validate(data)
        scenario.pop("_sha256", None)
        scenario.pop("_version", None)
        digest = content_hash(scenario)
        existing = self.versions(scenario["name"])
        for version in existing:
            if version.sha256 == digest:
                return version
        number = (existing[-1].version + 1) if existing else 1
        stored = dict(scenario, _version=number, _sha256=digest)
        path = self.root / scenario["name"] / f"v{number}.json"
        write_json_atomic(path, stored)
        return ScenarioVersion(scenario["name"], number, digest, path)

    def load(self, name: str, version: int | None = None) -> tuple[dict[str, Any], ScenarioVersion]:
        versions = self.versions(name)
        if not versions:
            raise ScenarioError(f"no saved scenario named {name!r}")
        chosen = (
            versions[-1]
            if version is None
            else next((v for v in versions if v.version == version), None)
        )
        if chosen is None:
            raise ScenarioError(f"scenario {name!r} has no version {version}")
        data = chosen.load()
        data.pop("_version", None)
        data.pop("_sha256", None)
        return validate(data), chosen

    def iter_all(self) -> Iterator[ScenarioVersion]:
        for name in self.names():
            yield from self.versions(name)

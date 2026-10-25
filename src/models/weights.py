"""The model catalogue: which weights live in ``bs_models_dir``, where they came from,
and whether they are intact.

``<bs_models_dir>/models.yaml`` lists one entry per model (see
``examples/models.yaml``): the registered adapter name, the kind of weights
(``npz`` prepared on the testbed with a seed, ``keras_imagenet`` from
``tf.keras.applications``, ``hf_checkpoint`` from the Hugging Face hub), the
file or directory under ``bs_models_dir``, the source and the licence. The
devices receive ``bs_models_dir`` at provisioning and NNPM distributes
checkpoints from it; ``SHA256SUMS`` files make every copy verifiable.

Usage::

    python -m src.models.weights list      [--models-dir DIR]
    python -m src.models.weights verify    [--models-dir DIR] [NAME ...]
    python -m src.models.weights checksums [--models-dir DIR] [NAME ...]
    python -m src.models.weights fetch     [--models-dir DIR] --allow-upstream (NAME ... | --all)

``fetch`` is the only command that reaches the internet, and only with
``--allow-upstream`` (the provisioning scripts' rule); ``build_mirror.sh models``
calls it once and copies the directory into the mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.config import load_config

CATALOGUE_NAME = "models.yaml"
CHECKSUMS_NAME = "SHA256SUMS"
KINDS = ("npz", "keras_imagenet", "hf_checkpoint")
HF_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "*.tiktoken"]
KERAS_CLASS_BY_NAME = {"resnet50": "ResNet50", "vgg16": "VGG16"}


class WeightsError(RuntimeError):
    pass


@dataclass
class Entry:
    name: str
    kind: str
    path: str
    source: str = ""
    licence: str = ""
    repo_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def location(self, models_dir: Path) -> Path:
        return models_dir / self.path

    @property
    def is_directory(self) -> bool:
        return self.kind == "hf_checkpoint"


def load_catalogue(models_dir: Path) -> list[Entry]:
    path = models_dir / CATALOGUE_NAME
    if not path.exists():
        raise WeightsError(f"{path} is missing (see examples/{CATALOGUE_NAME})")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = []
    for raw in data.get("models", []):
        raw = dict(raw)
        kind = raw.pop("kind", "npz")
        if kind not in KINDS:
            raise WeightsError(f"{raw.get('name')}: unknown kind {kind!r}; one of {KINDS}")
        entries.append(
            Entry(
                raw.pop("name"),
                kind,
                raw.pop("path"),
                raw.pop("source", ""),
                raw.pop("licence", ""),
                raw.pop("repo_id", None),
                raw,
            )
        )
    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    """``SHA256SUMS`` lines (``<hash>  <name>``) as ``{name: hash}``."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            out[parts[1].lstrip("*")] = parts[0]
    return out


def write_checksums(path: Path, files: dict[str, str]) -> None:
    lines = [f"{digest}  {name}" for name, digest in sorted(files.items())]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def directory_checksums(directory: Path) -> dict[str, str]:
    return {
        p.relative_to(directory).as_posix(): sha256_file(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != CHECKSUMS_NAME
    }


def checksums(entry: Entry, models_dir: Path) -> Path:
    """Write the checksums of an entry: into its directory for a checkpoint, into the
    models directory's ``SHA256SUMS`` for a single file."""
    location = entry.location(models_dir)
    if not location.exists():
        raise WeightsError(f"{entry.name}: {location} is missing")
    if entry.is_directory:
        target = location / CHECKSUMS_NAME
        write_checksums(target, directory_checksums(location))
        return target
    target = models_dir / CHECKSUMS_NAME
    recorded = read_checksums(target)
    recorded[entry.path] = sha256_file(location)
    write_checksums(target, recorded)
    return target


def verify(entry: Entry, models_dir: Path) -> list[str]:
    """Problems with an entry's files (empty when present and matching its checksums)."""
    location = entry.location(models_dir)
    if not location.exists():
        return [f"{entry.name}: {location} is missing"]
    problems = []
    if entry.is_directory:
        recorded = read_checksums(location / CHECKSUMS_NAME)
        if not recorded:
            return [f"{entry.name}: {location / CHECKSUMS_NAME} is missing"]
        current = directory_checksums(location)
        problems += [f"{entry.name}: missing {n}" for n in sorted(set(recorded) - set(current))]
        problems += [
            f"{entry.name}: changed {n}"
            for n in sorted(set(recorded) & set(current))
            if recorded[n] != current[n]
        ]
        return problems
    recorded = read_checksums(models_dir / CHECKSUMS_NAME)
    if entry.path not in recorded:
        return [f"{entry.name}: no checksum for {entry.path} in {models_dir / CHECKSUMS_NAME}"]
    if recorded[entry.path] != sha256_file(location):
        problems.append(f"{entry.name}: {entry.path} changed")
    return problems


def status(entry: Entry, models_dir: Path) -> dict[str, Any]:
    location = entry.location(models_dir)
    problems = verify(entry, models_dir) if location.exists() else []
    return {
        "name": entry.name,
        "kind": entry.kind,
        "path": str(location),
        "present": location.exists(),
        "verified": location.exists() and not problems,
        "problems": problems,
        "source": entry.source,
        "licence": entry.licence,
    }


def fetch(entry: Entry, models_dir: Path, *, allow_upstream: bool) -> Path:
    """Obtain the entry's weights: a seeded build for ``npz``, the ImageNet weights of a
    Keras application, or a Hugging Face snapshot. Then write its checksums."""
    location = entry.location(models_dir)
    if entry.kind == "npz":
        from src.models import registry

        seeded = registry.get(entry.name, **entry.extra.get("model_args", {}))
        seeded.build(int(entry.extra.get("seed", 0)))
        seeded.save_weights(location)
    elif entry.kind == "keras_imagenet":
        if not allow_upstream:
            raise WeightsError(
                f"{entry.name}: ImageNet weights are downloaded; give --allow-upstream"
            )
        import tensorflow as tf

        from src.models import registry

        keras_adapter: Any = registry.get(entry.name, **entry.extra.get("model_args", {}))
        keras_adapter.build(0)
        class_name = getattr(keras_adapter, "application", None) or KERAS_CLASS_BY_NAME.get(
            entry.name
        )
        if class_name is None:
            raise WeightsError(f"{entry.name}: not a Keras application")
        pretrained = getattr(tf.keras.applications, class_name)(
            weights="imagenet", input_shape=tuple(keras_adapter.input_shape)
        )
        keras_adapter._model.set_weights(pretrained.get_weights())
        keras_adapter.save_weights(location)
    else:
        if not allow_upstream:
            raise WeightsError(f"{entry.name}: a checkpoint is downloaded; give --allow-upstream")
        if not entry.repo_id:
            raise WeightsError(f"{entry.name}: hf_checkpoint needs repo_id")
        from huggingface_hub import snapshot_download

        location.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            entry.repo_id,
            local_dir=str(location),
            allow_patterns=entry.extra.get("allow_patterns", HF_PATTERNS),
            revision=entry.extra.get("revision"),
        )
    checksums(entry, models_dir)
    return location


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The model catalogue of bs_models_dir.")
    parser.add_argument("command", choices=["list", "verify", "checksums", "fetch"])
    parser.add_argument("names", nargs="*", help="catalogue entries (default: all)")
    parser.add_argument("--models-dir", default=None, help="default: bs_models_dir of config.yaml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--all", action="store_true", help="fetch: every entry")
    parser.add_argument("--allow-upstream", action="store_true")
    args = parser.parse_args(argv)
    models_dir = (
        Path(args.models_dir) if args.models_dir else load_config(args.config).bs_models_path
    )
    try:
        entries = load_catalogue(models_dir)
        chosen = [e for e in entries if not args.names or e.name in args.names]
        unknown = set(args.names) - {e.name for e in entries}
        if unknown:
            raise WeightsError(f"not in the catalogue: {', '.join(sorted(unknown))}")
        if args.command == "list":
            for entry in chosen:
                s = status(entry, models_dir)
                state = "verified" if s["verified"] else ("present" if s["present"] else "absent")
                print(f"{entry.name:16} {entry.kind:15} {state:9} {entry.path}  [{entry.licence}]")
            return 0
        if args.command == "verify":
            problems = [p for entry in chosen for p in verify(entry, models_dir)]
            for line in problems:
                print(line)
            print("catalogue verified" if not problems else f"{len(problems)} problem(s)")
            return 0 if not problems else 1
        if args.command == "checksums":
            for entry in chosen:
                print(f"{entry.name}: {checksums(entry, models_dir)}")
            return 0
        if not args.names and not args.all:
            raise WeightsError("fetch: give entry names or --all")
        for entry in chosen:
            print(f"{entry.name}: {fetch(entry, models_dir, allow_upstream=args.allow_upstream)}")
        return 0
    except WeightsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Manifest of the local package mirror.

The mirror is a directory on the edge server (``mirror_dir``) served over HTTP:

* ``wheels/<arch>/``: Python wheels downloaded for each lock file
  (``x86_64`` for the edge server, ``aarch64`` for the devices,
  ``aarch64-headsplit`` for the HeadSplit runtime);
* ``apt/``: Debian packages for the devices with ``Packages.gz``;
* ``src/headsplit.git``: a bare clone of HeadSplit at the pinned commit;
* ``MANIFEST.json``: the SHA-256 of every file, the lock files' hashes and the
  build time. The manifest is what makes the mirror a versioned artefact:
  ``verify`` detects any file that changed, appeared or disappeared since the
  build, so a later deployment installs the same stack or fails loudly.

Usage::

    python -m src.provisioning.mirror manifest [--mirror DIR]
    python -m src.provisioning.mirror verify   [--mirror DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT, load_config

MANIFEST_NAME = "MANIFEST.json"
LOCK_FILES = (
    "requirements/edge_server.lock",
    "requirements/edge_device.lock",
    "requirements/edge_device_llm.lock",
    "requirements/dev.lock",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mirror_files(mirror: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(mirror.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        out[path.relative_to(mirror).as_posix()] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    return out


def build_manifest(mirror: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    locks = {rel: sha256_file(repo_root / rel) for rel in LOCK_FILES if (repo_root / rel).exists()}
    files = mirror_files(mirror)
    return {
        "schema_version": 1,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "lock_files": locks,
        "file_count": len(files),
        "total_bytes": sum(f["size"] for f in files.values()),
        "files": files,
    }


def write_manifest(mirror: Path, repo_root: Path = REPO_ROOT) -> Path:
    mirror.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(mirror, repo_root)
    target = mirror / MANIFEST_NAME
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def verify(mirror: Path) -> list[str]:
    """Differences between the manifest and the directory (empty when identical)."""
    target = mirror / MANIFEST_NAME
    if not target.exists():
        return [f"{target} is missing; run `manifest` first"]
    recorded = json.loads(target.read_text(encoding="utf-8"))["files"]
    current = mirror_files(mirror)
    problems = [f"missing: {rel}" for rel in sorted(set(recorded) - set(current))]
    problems += [f"unexpected: {rel}" for rel in sorted(set(current) - set(recorded))]
    problems += [
        f"changed: {rel}"
        for rel in sorted(set(recorded) & set(current))
        if recorded[rel]["sha256"] != current[rel]["sha256"]
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify the mirror manifest.")
    parser.add_argument("command", choices=["manifest", "verify"])
    parser.add_argument("--mirror", default=None, help="default: mirror_dir from config.yaml")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    mirror = Path(args.mirror) if args.mirror else load_config(args.config).mirror_path
    if args.command == "manifest":
        path = write_manifest(mirror)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        print(f"{path}: {manifest['file_count']} files, {manifest['total_bytes']} bytes")
        return 0
    problems = verify(mirror)
    for line in problems:
        print(line)
    print("mirror matches its manifest" if not problems else f"{len(problems)} difference(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

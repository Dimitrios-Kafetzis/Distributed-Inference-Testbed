"""No IP addresses, user names or home directories hard-coded in code.

All paths and per-node metadata are externalised in configuration files
(Chapter 3). The only places that may hold such literals are the example
configuration and topology files, the configuration defaults, documentation,
schemas and tests.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCANNED = ("src", "gui", "scripts")
EXTENSIONS = {".py", ".sh", ".html", ".js", ".yaml", ".yml", ".json", ".toml", ".cfg"}
ALLOWED = {
    "src/config.py",  # the single place for configuration defaults
}
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
HOME_DIR = re.compile(r"/home/[A-Za-z0-9_-]+")
ALLOWED_IPS = {"0.0.0.0", "127.0.0.1"}


def scanned_files() -> list[Path]:
    files: list[Path] = []
    for top in SCANNED:
        root = REPO / top
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in EXTENSIONS and "vendor" not in path.parts:
                files.append(path)
    return files


def test_no_ip_or_home_literals_in_code() -> None:
    offences = []
    for path in scanned_files():
        rel = path.relative_to(REPO).as_posix()
        if rel in ALLOWED:
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            ips = [ip for ip in IPV4.findall(line) if ip not in ALLOWED_IPS]
            if ips or HOME_DIR.search(line):
                offences.append(f"{rel}:{number}: {line.strip()}")
    assert not offences, "hard-coded addresses or home directories:\n" + "\n".join(offences)

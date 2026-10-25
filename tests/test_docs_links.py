"""Every relative link in the Markdown documentation resolves to a file or directory.

External links are not fetched (the suite runs offline); anchors are ignored.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SKIP_DIRS = {".venv", "node_modules", ".git", "mirror", "results"}


def markdown_files() -> list[Path]:
    return [
        p
        for p in REPO.rglob("*.md")
        if not any(part in SKIP_DIRS for part in p.relative_to(REPO).parts)
    ]


def test_relative_links_resolve() -> None:
    broken = []
    for path in markdown_files():
        for target in LINK.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            plain = target.split("#", 1)[0]
            if not plain:
                continue
            if not (path.parent / plain).exists():
                broken.append(f"{path.relative_to(REPO)} -> {target}")
    assert broken == [], "broken links:\n" + "\n".join(broken)


def test_documentation_set_is_complete() -> None:
    expected = [
        "README.md",
        "CITATION.cff",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "docs/architecture.md",
        "docs/installation.md",
        "docs/setup_guide.md",
        "docs/nnpm_user_guide.md",
        "docs/module_reference.md",
        "docs/scenario_reference.md",
        "docs/results_format.md",
        "docs/headsplit_integration.md",
        "docs/hardware_validation_checklist.md",
        "scripts/provision/README.md",
        "requirements/README.md",
        "data/sample/README.md",
    ]
    missing = [rel for rel in expected if not (REPO / rel).exists()]
    assert missing == []

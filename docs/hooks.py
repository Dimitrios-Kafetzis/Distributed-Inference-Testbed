"""MkDocs hooks: include repository files that live outside ``docs/`` and rewrite links.

A page may consist of one line, ``{{ include "README.md" }}``: the hook replaces it
with the file's content. Links are then rewritten for the site, whatever page they
are on: a link into ``docs/`` becomes a link between pages, a link to a repository
file that has its own page (the README, the changelog, the provisioning README)
points at that page, and any other repository path points at GitHub. The source
documents stay valid on GitHub and in an editor; only the site is rewritten.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/Dimitrios-Kafetzis/Distributed-Inference-Testbed"
INCLUDE = re.compile(r'^\{\{\s*include\s+"([^"]+)"\s*\}\}\s*$', re.M)
# Every link or image target, nested badges included: the text before "](" is left alone.
LINK = re.compile(r"(\]\()([^)\s]+)(\))")

# Repository files that have a page of their own on the site.
PAGES = {
    "README.md": "index.md",
    "CHANGELOG.md": "changelog.md",
    "CONTRIBUTING.md": "contributing.md",
    "SECURITY.md": "security.md",
    "THIRD_PARTY_NOTICES.md": "third_party_notices.md",
    "scripts/provision/README.md": "provisioning.md",
    "requirements/README.md": "requirements.md",
    "data/sample/README.md": "sample_data.md",
}


def _source_dir(page_src: str, included: str | None) -> str:
    """The repository directory the page's links are relative to."""
    if included is not None:
        return posixpath.dirname(included)
    return posixpath.dirname(posixpath.join("docs", page_src))


def rewrite_links(markdown: str, page_src: str, included: str | None) -> str:
    base = _source_dir(page_src, included)
    page_dir = posixpath.dirname(page_src)

    def to_page(repo_rel: str, anchor: str) -> str:
        target = PAGES.get(repo_rel)
        if target is None and repo_rel.startswith("docs/"):
            target = repo_rel[len("docs/") :]
        if target is None:
            kind = "tree" if repo_rel.endswith("/") or (REPO / repo_rel).is_dir() else "blob"
            return f"{REPO_URL}/{kind}/main/{repo_rel}{anchor}"
        relative = posixpath.relpath(target, page_dir) if page_dir else target
        return relative + anchor

    def replace(match: re.Match[str]) -> str:
        prefix, target, suffix = match.groups()
        if target.startswith(("http://", "https://", "mailto:", "#", "/")):
            return match.group(0)
        plain, _, fragment = target.partition("#")
        anchor = f"#{fragment}" if fragment else ""
        repo_rel = posixpath.normpath(posixpath.join(base, plain))
        if plain.endswith("/"):
            repo_rel += "/"
        if repo_rel.startswith(".."):
            return match.group(0)
        return f"{prefix}{to_page(repo_rel, anchor)}{suffix}"

    return LINK.sub(replace, markdown)


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    included: str | None = None
    match = INCLUDE.search(markdown)
    if match:
        included = match.group(1)
        content = (REPO / included).read_text(encoding="utf-8")
        markdown = markdown[: match.start()] + content + markdown[match.end() :]
    return rewrite_links(markdown, page.file.src_uri, included)

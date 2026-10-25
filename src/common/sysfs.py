"""Rooted access to /sys, /proc and cgroupfs.

Modules that read CPU frequencies, temperatures or throttling flags, or that
write frequency limits and cgroup settings, do so through a :class:`SysFs`
rooted at ``/`` in production and at a temporary directory in tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path, PurePosixPath


class SysFs:
    def __init__(self, root: str | Path = "/") -> None:
        self.root = Path(root)

    def path(self, posix_path: str) -> Path:
        """Map an absolute path such as ``/sys/class/thermal`` under the root."""
        pure = PurePosixPath(posix_path)
        if not pure.is_absolute():
            raise ValueError(f"sysfs paths must be absolute: {posix_path}")
        return self.root.joinpath(*pure.parts[1:])

    def exists(self, posix_path: str) -> bool:
        return self.path(posix_path).exists()

    def read(self, posix_path: str) -> str:
        return self.path(posix_path).read_text(encoding="utf-8").strip()

    def read_int(self, posix_path: str) -> int:
        return int(self.read(posix_path))

    def write(self, posix_path: str, value: str | int) -> None:
        target = self.path(posix_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{value}\n", encoding="utf-8")

    def mkdir(self, posix_path: str) -> None:
        self.path(posix_path).mkdir(parents=True, exist_ok=True)

    def rmdir(self, posix_path: str) -> None:
        target = self.path(posix_path)
        if target.exists():
            target.rmdir()

    def glob(self, posix_pattern: str) -> Iterator[str]:
        """Glob under the root; yields absolute POSIX paths as the device would see them."""
        pure = PurePosixPath(posix_pattern)
        if not pure.is_absolute():
            raise ValueError(f"sysfs patterns must be absolute: {posix_pattern}")
        for match in sorted(self.root.glob(str(PurePosixPath(*pure.parts[1:])))):
            yield "/" + match.relative_to(self.root).as_posix()

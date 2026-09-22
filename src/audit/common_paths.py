"""Focused path-normalization helpers for deterministic audit artifacts."""

from __future__ import annotations

from pathlib import Path


def resolved_relative_to_root(path: Path, root: Path) -> str:
    """Return ``path.resolve().relative_to(root)`` serialized with ``/`` separators.

    The helper preserves the existing wrapper contract: the input is expected to
    be a ``Path``, missing paths are accepted by ``Path.resolve()``, and paths
    outside ``root`` raise ``ValueError``.
    """

    return str(path.resolve().relative_to(root)).replace("\\", "/")


def relative_to_root(path: Path, root: Path) -> str:
    """Return ``path.relative_to(root)`` serialized with ``/`` separators."""

    return str(path.relative_to(root)).replace("\\", "/")


def resolved_relative_to_root_or_absolute(path: str | Path, root: Path) -> str:
    """Return a resolved repository-relative path, or the resolved absolute path.

    The repository-relative branch uses POSIX separators. The absolute fallback
    preserves the platform string representation produced by ``Path.resolve()``.
    """

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def path_as_posix(path: Path) -> str:
    """Return ``path.as_posix()`` without resolving or requiring existence."""

    return path.as_posix()

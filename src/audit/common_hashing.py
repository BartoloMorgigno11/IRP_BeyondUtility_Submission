"""Generic deterministic hashing helpers.

This module intentionally contains only hashing primitives for inputs that
callers have already selected. It does not decide which repository paths,
datasets, model outputs, or scientific artifacts should be hashed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 hex digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the lowercase SHA-256 hex digest for UTF-8 encoded text."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 hex digest for a file read as bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(DEFAULT_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_or_empty(path: str | Path) -> str:
    """Return a file SHA-256 digest, or an empty string when file I/O fails."""
    try:
        return sha256_file(path)
    except OSError:
        return ""


def aggregate_named_hashes(named_hashes: Mapping[str, str] | Iterable[tuple[str, str]]) -> str:
    """Hash sorted name/digest pairs using explicit NUL and newline separators."""
    items = named_hashes.items() if isinstance(named_hashes, Mapping) else named_hashes
    digest = hashlib.sha256()
    for name, value in sorted((str(name), str(value)) for name, value in items):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_directory_tree(root: str | Path) -> str:
    """Hash file-relative paths, sizes, and file digests for a directory tree."""
    root_path = Path(root)
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root_path.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root_path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()

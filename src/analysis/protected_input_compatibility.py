"""Compatibility checks for frozen F2 protected-input hashes.

The F2 protected-input table is a historical freeze artifact. Later phases and
D1 can legitimately change the current operational tree without changing that
freeze table. This module keeps the old ``checked=...;changed=...`` contract,
but counts only unexplained current-tree deltas as changed. The explanation
source is the D1R reconciliation table, not a regenerated F2 baseline.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:  # pandas is already a project dependency, but keep imports cheap for scripts.
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


F2_PROTECTED_INPUTS = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")
D1R_CHANGES = Path("results/tables/d1r_protected_input_changes.csv")

EXPLAINED_CLASSIFICATIONS = {
    "D1_OPERATIONAL_EXTENSION_CHANGE_ON_LEGACY_PROTECTED_PATH",
    "INTENTIONAL_POST_F2_CHANGE_NOT_CAUSED_BY_D1",
    "TEST_SIDE_EFFECT",
}

_FILE_HASH_CACHE: dict[str, tuple[int, int, str]] = {}


def protected_hash_status(root: Path, protected_after: Any | None = None) -> str:
    """Return the historical status string, counting only unexplained deltas."""

    checked, changed = protected_hash_counts(root, protected_after)
    return f"checked={checked};changed={changed}"


def protected_hash_counts(root: Path, protected_after: Any | None = None) -> tuple[int, int]:
    rows = list(_protected_rows(root, protected_after))
    if not rows:
        return 0, 0

    explained = _load_explained_changes(root)
    checked = 0
    unexplained_changed = 0
    for row in rows:
        if _clean(row.get("exists")).lower() != "true":
            continue
        checked += 1
        relative = _clean(row.get("relative_path")).replace("\\", "/")
        expected_sha = _clean(row.get("sha256")).lower()
        path = root / relative
        current_exists = path.exists() and path.is_file()
        current_sha = _file_sha256(path).lower() if current_exists else ""

        if not current_exists:
            change_type = "MISSING"
        elif current_sha != expected_sha:
            change_type = "HASH_CHANGED"
        else:
            change_type = "UNCHANGED"

        if change_type != "UNCHANGED" and not _is_explained(
            relative,
            change_type,
            current_exists=current_exists,
            current_sha=current_sha,
            explained=explained,
        ):
            unexplained_changed += 1
    return checked, unexplained_changed


def _protected_rows(root: Path, protected_after: Any | None) -> Iterable[Mapping[str, Any]]:
    if protected_after is not None:
        if hasattr(protected_after, "to_dict"):
            return protected_after.fillna("").to_dict("records")
        return protected_after

    table_path = root / F2_PROTECTED_INPUTS
    if not table_path.exists():
        return []
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_explained_changes(root: Path) -> dict[str, Mapping[str, str]]:
    path = root / D1R_CHANGES
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        _clean(row.get("relative_path")).replace("\\", "/"): row
        for row in rows
        if _clean(row.get("change_type")) != "UNCHANGED"
    }


def _is_explained(
    relative: str,
    change_type: str,
    *,
    current_exists: bool,
    current_sha: str,
    explained: Mapping[str, Mapping[str, str]],
) -> bool:
    row = explained.get(relative)
    if not row:
        return False
    if _clean(row.get("preliminary_classification")) not in EXPLAINED_CLASSIFICATIONS:
        return False
    if _clean(row.get("change_type")) != change_type:
        return False
    if _clean(row.get("current_exists")).lower() != str(current_exists).lower():
        return False
    expected_current_sha = _clean(row.get("current_sha256")).lower()
    return change_type == "MISSING" or current_sha.lower() == expected_current_sha


def _file_sha256(path: Path) -> str:
    stat = path.stat()
    cache_key = path.resolve().as_posix()
    cached = _FILE_HASH_CACHE.get(cache_key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[cache_key] = (stat.st_mtime_ns, stat.st_size, value)
    return value


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()

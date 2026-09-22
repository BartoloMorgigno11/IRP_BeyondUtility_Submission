"""Shared manifest serialization helpers with narrow manifest scope."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.audit import common_io


PathLike = str | Path


def canonicalize_manifest_payload(
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    separators: tuple[str, str] | None = None,
    newline: bool = True,
    default: Any | None = None,
) -> str:
    """Serialize a JSON manifest payload with explicit deterministic options."""

    kwargs: dict[str, Any] = {
        "ensure_ascii": ensure_ascii,
        "sort_keys": sort_keys,
    }
    if indent is not None:
        kwargs["indent"] = indent
    if separators is not None:
        kwargs["separators"] = separators
    if default is not None:
        kwargs["default"] = default
    text = json.dumps(payload, **kwargs)
    return text + "\n" if newline and not text.endswith("\n") else text


def write_json_manifest(
    path: PathLike,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    separators: tuple[str, str] | None = None,
    newline: bool = True,
    create_parent: bool = True,
    overwrite: bool = True,
    atomic: bool = False,
    default: Any | None = None,
    encoding: str = "utf-8",
) -> Path:
    """Write a JSON manifest while preserving explicit overwrite and newline policy."""

    target = Path(path)
    if not overwrite and target.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {target}")
    if create_parent:
        target.parent.mkdir(parents=True, exist_ok=True)

    text = canonicalize_manifest_payload(
        payload,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        separators=separators,
        newline=newline,
        default=default,
    )
    write_target = target.with_name(target.name + ".tmp") if atomic else target
    try:
        write_target.write_text(text, encoding=encoding)
        if atomic:
            os.replace(write_target, target)
    finally:
        if atomic and write_target.exists():
            write_target.unlink()
    return target


def read_json_manifest(path: PathLike, *, encoding: str = "utf-8") -> Any:
    """Read a JSON manifest payload from disk."""

    return json.loads(Path(path).read_text(encoding=encoding))


def read_csv_manifest_rows(
    path: PathLike,
    *,
    required_columns: Sequence[str] | None = None,
    missing: str = "raise",
    strip: bool = True,
    project_columns: Sequence[str] | None = None,
    root: PathLike | None = None,
    encoding: str = "utf-8",
    newline: str = "",
) -> list[dict[str, str]]:
    """Read a CSV manifest and optionally enforce manifest columns."""

    target = Path(root) / Path(path) if root is not None else Path(path)
    if not target.exists():
        if missing == "empty":
            return []
        raise FileNotFoundError(f"Manifest not found: {target}")
    if missing not in {"raise", "empty"}:
        raise ValueError(f"unsupported missing policy: {missing}")

    rows = common_io.read_csv_rows(target, encoding=encoding, newline=newline)
    if required_columns is not None:
        fieldnames = _csv_fieldnames(target, encoding=encoding, newline=newline)
        missing_columns = [column for column in required_columns if column not in fieldnames]
        if missing_columns:
            raise ValueError("Manifest is missing required column(s): " + ", ".join(missing_columns))

    columns = list(project_columns) if project_columns is not None else None
    prepared: list[dict[str, str]] = []
    for row in rows:
        source = {column: row.get(column, "") for column in columns} if columns is not None else dict(row)
        if strip:
            prepared.append({key: (value or "").strip() for key, value in source.items()})
        else:
            prepared.append({key: value or "" for key, value in source.items()})
    return prepared


def write_csv_manifest_rows(
    path: PathLike,
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    create_parent: bool = True,
    overwrite: bool = True,
    fill_missing: bool = True,
    extrasaction: str = "raise",
    lineterminator: str | None = None,
    encoding: str = "utf-8",
    newline: str = "",
    atomic: bool = False,
) -> None:
    """Write CSV manifest rows through the canonical CSV helper."""

    prepared = []
    for row in rows:
        complete: dict[str, Any] = {column: "" for column in columns} if fill_missing else {}
        complete.update(row)
        prepared.append(complete)
    common_io.write_csv_rows(
        path,
        columns,
        prepared,
        create_parent=create_parent,
        overwrite=overwrite,
        extrasaction=extrasaction,
        lineterminator=lineterminator,
        encoding=encoding,
        newline=newline,
        project_columns=False,
        atomic=atomic,
    )


def compare_csv_manifest_rows(left_path: PathLike, right_path: PathLike) -> bool:
    """Return whether two CSV manifests have identical row mappings and order."""

    return read_csv_manifest_rows(left_path, strip=False) == read_csv_manifest_rows(right_path, strip=False)


def manifest_shape_summary(frame: Any) -> dict[str, str]:
    """Return a deterministic row/column summary for a manifest-like table."""

    columns = [str(column) for column in getattr(frame, "columns", [])]
    return {
        "row_count": str(len(frame)),
        "column_count": str(len(columns)),
        "columns": ";".join(columns),
    }


def manifest_inventory_has_value(rows_or_frame: Any, column: str, expected: str) -> bool:
    """Return whether a manifest inventory contains ``expected`` in ``column``."""

    if hasattr(rows_or_frame, "__getitem__") and hasattr(rows_or_frame, "astype"):
        return bool(rows_or_frame[column].astype(str).eq(str(expected)).any())
    return any(str(row.get(column, "")) == str(expected) for row in rows_or_frame)


def all_manifest_rows_unique(rows: Iterable[Mapping[str, Any]], key_columns: Sequence[str]) -> bool:
    """Return whether manifest rows are unique for the selected key columns."""

    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in key_columns)
        if key in seen:
            return False
        seen.add(key)
    return True


def _csv_fieldnames(path: Path, *, encoding: str, newline: str) -> list[str]:
    import csv

    with path.open("r", encoding=encoding, newline=newline) as handle:
        return list(csv.DictReader(handle).fieldnames or [])

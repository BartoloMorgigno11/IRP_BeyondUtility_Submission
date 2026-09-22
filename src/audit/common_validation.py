"""Generic validation table helpers for audit and experiment wrappers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.audit import common_io


PathLike = str | Path
Row = Mapping[str, Any]

KNOWN_STATUSES = (
    "pass",
    "fail",
    "passed",
    "failed",
    "warning",
    "blocked",
    "skipped",
    "unresolved",
    "not_applicable",
    "informational",
    "expected_failure",
)


def status_from_outcome(
    passed: bool | None,
    *,
    pass_status: str = "pass",
    fail_status: str = "fail",
    none_status: str | None = None,
) -> str:
    """Return the existing status spelling for a computed boolean outcome."""

    if passed is None:
        if none_status is None:
            raise ValueError("none_status is required when passed is None")
        return none_status
    return pass_status if passed else fail_status


def make_validation_row(
    *,
    columns: Sequence[str],
    values: Mapping[str, Any],
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one validation row while preserving caller supplied column order."""

    default_values = defaults or {}
    return {column: values.get(column, default_values.get(column, "")) for column in columns}


def normalize_validation_row(
    row: Mapping[str, Any],
    *,
    columns: Sequence[str],
    required_columns: Sequence[str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project a row to a validation schema and require selected source fields."""

    missing = [column for column in required_columns or () if column not in row]
    if missing:
        raise ValueError("validation row missing required column(s): " + ", ".join(missing))
    return make_validation_row(columns=columns, values=row, defaults=defaults)


def build_validation_table(
    rows: Iterable[Mapping[str, Any]],
    *,
    columns: Sequence[str],
    required_columns: Sequence[str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return validation rows projected to the requested schema in input order."""

    return [
        normalize_validation_row(
            row,
            columns=columns,
            required_columns=required_columns,
            defaults=defaults,
        )
        for row in rows
    ]


def require_validation_columns(columns: Sequence[str], required_columns: Sequence[str]) -> None:
    """Raise when a validation schema does not contain every required column."""

    available = set(columns)
    missing = [column for column in required_columns if column not in available]
    if missing:
        raise ValueError("validation table missing required column(s): " + ", ".join(missing))


def summarize_validation_statuses(
    rows: Iterable[Mapping[str, Any]],
    *,
    status_column: str = "status",
    known_statuses: Sequence[str] = KNOWN_STATUSES,
    include_unknown: bool = True,
) -> dict[str, int]:
    """Count validation statuses without merging distinct existing spellings."""

    counts: Counter[str] = Counter(str(row.get(status_column, "")) for row in rows)
    summary = {status: counts.get(status, 0) for status in known_statuses}
    if include_unknown:
        for status, count in sorted(counts.items()):
            if status not in summary:
                summary[status] = count
    return summary


def validation_passed(
    rows: Iterable[Mapping[str, Any]],
    *,
    status_column: str = "status",
    passing_statuses: Sequence[str] = ("pass", "passed"),
    allow_empty: bool = True,
    unknown_status: str = "raise",
) -> bool:
    """Return whether all rows have a caller-approved passing status."""

    row_list = list(rows)
    if not row_list:
        return allow_empty
    passing = set(passing_statuses)
    for row in row_list:
        status = str(row.get(status_column, ""))
        if status in passing:
            continue
        if unknown_status == "retain":
            return False
        raise ValueError(f"validation row has non-passing status: {status}")
    return True


def write_validation_table(
    path: PathLike,
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
    **kwargs: Any,
) -> None:
    """Write validation rows through the existing canonical CSV helper."""

    common_io.write_csv_rows(path, columns, rows, **kwargs)

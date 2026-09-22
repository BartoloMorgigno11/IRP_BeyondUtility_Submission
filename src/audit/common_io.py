"""Deterministic CSV I/O helpers shared by audit and experiment wrappers."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    import pandas as pd


PathLike = str | Path
Row = Mapping[str, Any]
ValueTransform = Callable[[Any], Any]
ExceptionFactory = Callable[[str], Exception]


def read_csv_dataframe(
    path: PathLike,
    *,
    root: PathLike | None = None,
    dtype: Any = str,
    keep_default_na: bool = False,
    require_file: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV into a DataFrame while preserving existing wrapper defaults."""

    import pandas as pd

    target = Path(root) / Path(path) if root is not None else Path(path)
    if require_file and not target.is_file():
        raise FileNotFoundError(target)
    return pd.read_csv(target, dtype=dtype, keep_default_na=keep_default_na, **kwargs)


def write_dataframe_csv(
    path: PathLike,
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    create_parent: bool = True,
    overwrite: bool = True,
    existing_error_factory: ExceptionFactory | None = None,
    existing_error_message: str | None = None,
    lineterminator: str | None = None,
) -> None:
    """Write a DataFrame to CSV with explicit column, overwrite, and newline policy."""

    target = Path(path)
    if not overwrite and target.exists():
        message = existing_error_message or f"refusing to overwrite existing output: {target}"
        raise (existing_error_factory or FileExistsError)(message)
    if create_parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    output = frame.loc[:, list(columns)] if columns is not None else frame
    kwargs: dict[str, Any] = {"index": False}
    if lineterminator is not None:
        kwargs["lineterminator"] = lineterminator
    output.to_csv(target, **kwargs)


def read_csv_rows(
    path: PathLike,
    *,
    root: PathLike | None = None,
    encoding: str = "utf-8",
    newline: str = "",
) -> list[dict[str, str]]:
    """Read CSV rows with csv.DictReader and deterministic path handling."""

    target = Path(root) / Path(path) if root is not None else Path(path)
    with target.open("r", encoding=encoding, newline=newline) as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: PathLike,
    columns: Sequence[str],
    rows: Iterable[Row],
    *,
    create_parent: bool = True,
    overwrite: bool = True,
    existing_error_factory: ExceptionFactory | None = None,
    existing_error_message: str | None = None,
    required_name_prefix: str | None = None,
    prefix_error_factory: ExceptionFactory | None = None,
    prefix_error_message: str | None = None,
    extrasaction: str = "raise",
    lineterminator: str | None = None,
    encoding: str = "utf-8",
    newline: str = "",
    value_transform: ValueTransform | None = None,
    project_columns: bool = True,
    atomic: bool = False,
) -> None:
    """Write row mappings to CSV while preserving wrapper-specific policies."""

    target = Path(path)
    if required_name_prefix is not None and not target.name.startswith(required_name_prefix):
        message = prefix_error_message or f"refusing to write output with unexpected prefix: {target}"
        raise (prefix_error_factory or RuntimeError)(message)
    if not overwrite and target.exists():
        message = existing_error_message or f"refusing to overwrite existing output: {target}"
        raise (existing_error_factory or FileExistsError)(message)
    if create_parent:
        target.parent.mkdir(parents=True, exist_ok=True)

    write_target = target.with_name(target.name + ".tmp") if atomic else target
    writer_kwargs: dict[str, Any] = {"fieldnames": list(columns), "extrasaction": extrasaction}
    if lineterminator is not None:
        writer_kwargs["lineterminator"] = lineterminator
    try:
        with write_target.open("w", encoding=encoding, newline=newline) as handle:
            writer = csv.DictWriter(handle, **writer_kwargs)
            writer.writeheader()
            for row in rows:
                writer.writerow(_prepare_row(row, columns, project_columns, value_transform))
        if atomic:
            os.replace(write_target, target)
    finally:
        if atomic and write_target.exists():
            write_target.unlink()


def _prepare_row(
    row: Row,
    columns: Sequence[str],
    project_columns: bool,
    value_transform: ValueTransform | None,
) -> dict[str, Any]:
    if project_columns:
        return {
            column: value_transform(row.get(column, "")) if value_transform else row.get(column, "")
            for column in columns
        }
    if value_transform is None:
        return dict(row)
    return {key: value_transform(value) for key, value in row.items()}

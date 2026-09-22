"""History-length prompt input utilities.

These helpers only transform already-provided user histories. They do not call
APIs, generate recommendations, or infer missing interactions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import pandas as pd


HistoryLength = int | Literal["full"]

DEFAULT_HISTORY_LENGTHS: tuple[HistoryLength, ...] = (0, 1, 2, 5, 10, "full")
DEFAULT_USER_COLUMN = "user_id"
DEFAULT_ITEM_COLUMN = "item_id"
DEFAULT_TIMESTAMP_COLUMN = "timestamp"
DEFAULT_RANK_COLUMN = "rank"
HISTORY_POSITION_COLUMN = "history_position"


def normalize_history_length(length: HistoryLength | str) -> HistoryLength:
    """Validate and normalize one requested history length."""

    if isinstance(length, str):
        value = length.strip().lower()
        if value == "full":
            return "full"
        if value.isdigit():
            return int(value)
        raise ValueError("history length must be a non-negative integer or 'full'")

    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("history length must be a non-negative integer or 'full'")
    return length


def truncate_user_histories(
    histories: pd.DataFrame,
    length: HistoryLength | str,
    *,
    user_col: str = DEFAULT_USER_COLUMN,
    item_col: str = DEFAULT_ITEM_COLUMN,
    timestamp_col: str = DEFAULT_TIMESTAMP_COLUMN,
    rank_col: str = DEFAULT_RANK_COLUMN,
) -> pd.DataFrame:
    """Return the latest requested interactions for each user.

    Interactions are ordered chronologically within each user. If timestamps are
    available they drive ordering; otherwise rank values are used; otherwise the
    input order is preserved. The returned rows remain in chronological order
    after truncation, so ``length=1`` returns the latest item and ``length=full``
    returns the full ordered history.
    """

    requested_length = normalize_history_length(length)
    user_col = _normalize_column_name(user_col)
    item_col = _normalize_column_name(item_col)
    timestamp_col = _normalize_column_name(timestamp_col)
    rank_col = _normalize_column_name(rank_col)

    frame = _normalize_dataframe(histories, frame_name="histories")
    if frame.empty:
        return _empty_history_frame(frame, requested_length, user_col=user_col)

    _require_columns(frame, frame_name="histories", columns=(user_col, item_col))
    _require_non_empty(frame, frame_name="histories", columns=(user_col, item_col))

    ordered = _ordered_histories(
        frame,
        user_col=user_col,
        timestamp_col=timestamp_col,
        rank_col=rank_col,
    )

    if requested_length == 0:
        return _empty_history_frame(ordered, requested_length, user_col=user_col)

    pieces: list[pd.DataFrame] = []
    for _, group in ordered.groupby(user_col, sort=False, dropna=False):
        selected = group if requested_length == "full" else group.tail(requested_length)
        pieces.append(selected)

    result = pd.concat(pieces, axis=0).reset_index(drop=True) if pieces else ordered.iloc[0:0].copy()
    result[HISTORY_POSITION_COLUMN] = result.groupby(user_col, sort=False).cumcount() + 1
    result["requested_history_length"] = _history_length_label(requested_length)
    result["history_length"] = _history_length_label(requested_length)
    result["actual_history_length"] = result.groupby(user_col, sort=False)[item_col].transform("count")
    return result


def create_history_length_variants(
    histories: pd.DataFrame,
    *,
    lengths: Iterable[HistoryLength | str] = DEFAULT_HISTORY_LENGTHS,
    user_col: str = DEFAULT_USER_COLUMN,
    item_col: str = DEFAULT_ITEM_COLUMN,
    timestamp_col: str = DEFAULT_TIMESTAMP_COLUMN,
    rank_col: str = DEFAULT_RANK_COLUMN,
) -> pd.DataFrame:
    """Return one prompt-input row per user and requested history length.

    ``history_item_ids`` is a list in chronological order after truncation. The
    function is deterministic for a fixed input dataframe and requested length
    sequence.
    """

    user_col = _normalize_column_name(user_col)
    item_col = _normalize_column_name(item_col)
    timestamp_col = _normalize_column_name(timestamp_col)
    rank_col = _normalize_column_name(rank_col)

    frame = _normalize_dataframe(histories, frame_name="histories")
    columns = [
        user_col,
        "history_length",
        "requested_history_length",
        "actual_history_length",
        "history_item_count",
        "history_item_ids",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    _require_columns(frame, frame_name="histories", columns=(user_col, item_col))
    _require_non_empty(frame, frame_name="histories", columns=(user_col, item_col))

    ordered = _ordered_histories(
        frame,
        user_col=user_col,
        timestamp_col=timestamp_col,
        rank_col=rank_col,
    )
    users = ordered[user_col].drop_duplicates().tolist()

    rows: list[dict[str, Any]] = []
    for requested_length in [normalize_history_length(length) for length in lengths]:
        label = _history_length_label(requested_length)
        truncated = truncate_user_histories(
            ordered,
            requested_length,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            rank_col=rank_col,
        )
        items_by_user = {
            user_id: group[item_col].tolist()
            for user_id, group in truncated.groupby(user_col, sort=False, dropna=False)
        }
        for user_id in users:
            item_ids = items_by_user.get(user_id, [])
            rows.append(
                {
                    user_col: user_id,
                    "history_length": label,
                    "requested_history_length": label,
                    "actual_history_length": len(item_ids),
                    "history_item_count": len(item_ids),
                    "history_item_ids": item_ids,
                }
            )

    return pd.DataFrame(rows, columns=columns)


def _ordered_histories(
    frame: pd.DataFrame,
    *,
    user_col: str,
    timestamp_col: str,
    rank_col: str,
) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["__input_order"] = range(len(ordered))

    sort_columns = [user_col]
    if timestamp_col in ordered.columns:
        ordered["__timestamp_value"] = _sortable_values(ordered[timestamp_col])
        ordered["__timestamp_missing"] = ordered["__timestamp_value"].isna()
        sort_columns.extend(["__timestamp_missing", "__timestamp_value"])
    elif rank_col in ordered.columns:
        ordered["__rank_value"] = pd.to_numeric(ordered[rank_col], errors="coerce")
        ordered["__rank_missing"] = ordered["__rank_value"].isna()
        sort_columns.extend(["__rank_missing", "__rank_value"])

    sort_columns.append("__input_order")
    ordered = ordered.sort_values(sort_columns, kind="mergesort")
    helper_columns = [column for column in ordered.columns if column.startswith("__")]
    return ordered.drop(columns=helper_columns)


def _sortable_values(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric

    datetime_values = pd.to_datetime(series, errors="coerce", utc=True)
    return pd.Series(datetime_values.view("int64"), index=series.index).where(datetime_values.notna())


def _empty_history_frame(
    frame: pd.DataFrame,
    requested_length: HistoryLength,
    *,
    user_col: str,
) -> pd.DataFrame:
    result = frame.iloc[0:0].copy()
    for column in (
        HISTORY_POSITION_COLUMN,
        "requested_history_length",
        "history_length",
        "actual_history_length",
    ):
        if column not in result.columns:
            result[column] = pd.Series(dtype="object")
    result["requested_history_length"] = result["requested_history_length"].astype("object")
    result["history_length"] = result["history_length"].astype("object")
    if user_col not in result.columns:
        result[user_col] = pd.Series(dtype="object")
    return result.assign(
        requested_history_length=_history_length_label(requested_length),
        history_length=_history_length_label(requested_length),
    )


def _history_length_label(length: HistoryLength) -> int | str:
    return "full" if length == "full" else int(length)


def _normalize_dataframe(df: pd.DataFrame, *, frame_name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{frame_name} must be a pandas DataFrame")

    normalized = df.copy()
    normalized.columns = [_normalize_column_name(column) for column in normalized.columns]
    duplicated = normalized.columns[normalized.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            f"{frame_name} dataframe has duplicate column name(s) after "
            f"normalization: {', '.join(duplicated)}"
        )
    return normalized


def _normalize_column_name(column: object) -> str:
    return str(column).strip().lower()


def _require_columns(
    df: pd.DataFrame,
    *,
    frame_name: str,
    columns: tuple[str, ...],
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{frame_name} dataframe is missing required column(s): "
            f"{', '.join(missing)}"
        )


def _require_non_empty(
    df: pd.DataFrame,
    *,
    frame_name: str,
    columns: tuple[str, ...],
) -> None:
    empty_columns = [column for column in columns if _is_empty_series(df[column]).any()]
    if empty_columns:
        raise ValueError(
            f"{frame_name} dataframe has missing value(s) in required "
            f"column(s): {', '.join(empty_columns)}"
        )


def _is_empty_series(series: pd.Series) -> pd.Series:
    as_text = series.astype("string")
    return series.isna() | as_text.str.strip().eq("")

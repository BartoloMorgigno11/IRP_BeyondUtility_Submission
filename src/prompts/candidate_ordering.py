"""Deterministic candidate ordering utilities for prompt construction.

These helpers only transform already-provided candidate sets. They do not call
APIs, generate recommendations, or infer missing candidate data.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from typing import Any

import pandas as pd


ORDER_ORIGINAL = "original"
ORDER_RANDOM = "random"
ORDER_REVERSED = "reversed"
ORDER_POSITIVE_AT_START = "positive_at_start"
ORDER_POSITIVE_AT_END = "positive_at_end"
ORDERING_VARIANTS = (
    ORDER_ORIGINAL,
    ORDER_RANDOM,
    ORDER_REVERSED,
    ORDER_POSITIVE_AT_START,
    ORDER_POSITIVE_AT_END,
)

DEFAULT_USER_COLUMN = "user_id"
DEFAULT_ITEM_COLUMN = "candidate_item_id"
DEFAULT_POSITION_COLUMN = "candidate_position"
DEFAULT_POSITIVE_COLUMN = "is_ground_truth"
ORIGINAL_POSITION_COLUMN = "original_candidate_position"
ORDERING_COLUMN = "candidate_ordering"


def apply_candidate_ordering(
    candidates: pd.DataFrame,
    variant: str,
    *,
    seed: int = 42,
    user_col: str = DEFAULT_USER_COLUMN,
    item_col: str = DEFAULT_ITEM_COLUMN,
    position_col: str = DEFAULT_POSITION_COLUMN,
    positive_col: str = DEFAULT_POSITIVE_COLUMN,
) -> pd.DataFrame:
    """Return candidates reordered according to one deterministic variant.

    The returned dataframe preserves candidate IDs, text/title columns and
    other extra metadata. ``candidate_position`` is renumbered from 1 within
    each user after the transformation, and ``original_candidate_position``
    records the incoming position.
    """

    if variant not in ORDERING_VARIANTS:
        allowed = ", ".join(ORDERING_VARIANTS)
        raise ValueError(f"Unknown candidate ordering variant {variant!r}; expected one of: {allowed}")

    user_col = _normalize_column_name(user_col)
    item_col = _normalize_column_name(item_col)
    position_col = _normalize_column_name(position_col)
    positive_col = _normalize_column_name(positive_col)

    frame = _normalize_dataframe(candidates, frame_name="candidates")
    _require_columns(
        frame,
        frame_name="candidates",
        columns=(user_col, item_col, position_col),
    )
    _require_non_empty(frame, frame_name="candidates", columns=(user_col, item_col, position_col))

    ordered_input = _sort_by_original_position(frame, user_col=user_col, position_col=position_col)
    if ORIGINAL_POSITION_COLUMN not in ordered_input.columns:
        ordered_input[ORIGINAL_POSITION_COLUMN] = ordered_input[position_col]

    ordered_groups: list[pd.DataFrame] = []
    for user_id, group in ordered_input.groupby(user_col, sort=False, dropna=False):
        ordered_groups.append(
            _order_one_user(
                group,
                variant=variant,
                seed=seed,
                user_id=user_id,
                positive_col=positive_col,
            )
        )

    if ordered_groups:
        result = pd.concat(ordered_groups, axis=0)
    else:
        result = ordered_input.copy()

    result = result.reset_index(drop=True)
    result[position_col] = result.groupby(user_col, sort=False).cumcount() + 1
    result[ORDERING_COLUMN] = variant
    return result


def candidate_ordering_variants(
    candidates: pd.DataFrame,
    *,
    variants: Iterable[str] = ORDERING_VARIANTS,
    seed: int = 42,
    user_col: str = DEFAULT_USER_COLUMN,
    item_col: str = DEFAULT_ITEM_COLUMN,
    position_col: str = DEFAULT_POSITION_COLUMN,
    positive_col: str = DEFAULT_POSITIVE_COLUMN,
) -> dict[str, pd.DataFrame]:
    """Return a mapping of ordering variant name to reordered candidates."""

    return {
        variant: apply_candidate_ordering(
            candidates,
            variant,
            seed=seed,
            user_col=user_col,
            item_col=item_col,
            position_col=position_col,
            positive_col=positive_col,
        )
        for variant in variants
    }


def _order_one_user(
    group: pd.DataFrame,
    *,
    variant: str,
    seed: int,
    user_id: Any,
    positive_col: str,
) -> pd.DataFrame:
    if variant == ORDER_ORIGINAL:
        return group.copy()

    if variant == ORDER_REVERSED:
        return group.iloc[::-1].copy()

    if variant == ORDER_RANDOM:
        ordered_indices = list(group.index)
        rng = random.Random(_stable_seed(seed, user_id))
        rng.shuffle(ordered_indices)
        return group.loc[ordered_indices].copy()

    if variant in {ORDER_POSITIVE_AT_START, ORDER_POSITIVE_AT_END}:
        if positive_col not in group.columns:
            return group.copy()
        positive_mask = _truthy_mask(group[positive_col])
        positives = group.loc[positive_mask]
        negatives = group.loc[~positive_mask]
        pieces = (
            [positives, negatives]
            if variant == ORDER_POSITIVE_AT_START
            else [negatives, positives]
        )
        return pd.concat(pieces, axis=0).copy()

    raise ValueError(f"Unhandled candidate ordering variant: {variant}")


def _sort_by_original_position(
    frame: pd.DataFrame,
    *,
    user_col: str,
    position_col: str,
) -> pd.DataFrame:
    sortable = frame.copy()
    sortable["__input_order"] = range(len(sortable))
    sortable["__position_value"] = pd.to_numeric(sortable[position_col], errors="coerce")
    sortable["__position_missing"] = sortable["__position_value"].isna()
    sortable = sortable.sort_values(
        [user_col, "__position_missing", "__position_value", "__input_order"],
        kind="mergesort",
    )
    return sortable.drop(columns=["__input_order", "__position_value", "__position_missing"])


def _stable_seed(seed: int, user_id: Any) -> int:
    digest = hashlib.sha256(f"{seed}|{user_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _truthy_mask(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.isin({"1", "true", "t", "yes", "y"})


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

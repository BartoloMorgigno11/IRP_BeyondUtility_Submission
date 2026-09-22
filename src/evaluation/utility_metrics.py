"""Utility metrics for ranked recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Any

import pandas as pd


DEFAULT_USER_COLUMN = "user_id"
DEFAULT_RECOMMENDATION_COLUMNS = ("recommended_item_id", "recommended_item_text")
DEFAULT_GROUND_TRUTH_COLUMN = "ground_truth_item_id"
DEFAULT_RANK_COLUMN = "rank"


@dataclass(frozen=True)
class _PreparedUtilityData:
    relevant_by_user: dict[Any, set[Any]]
    recommendations_by_user: dict[Any, list[Any]]

    @property
    def user_count(self) -> int:
        return len(self.relevant_by_user)


def hit_rate_at_k(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int,
    *,
    user_col: str = DEFAULT_USER_COLUMN,
    recommendation_col: str | None = None,
    ground_truth_col: str = DEFAULT_GROUND_TRUTH_COLUMN,
    rank_col: str = DEFAULT_RANK_COLUMN,
) -> float:
    """Return mean binary Hit Rate@K over users with ground truth.

    A user contributes 1.0 if any relevant item appears in the top-k ranked
    recommendations, otherwise 0.0. Users without ground truth are excluded from
    the denominator. Duplicate recommendation items are not counted as repeated
    hits, and empty recommendation lists score 0.0 for users with ground truth.
    """

    prepared = _prepare_utility_data(
        recommendations,
        ground_truth,
        k,
        user_col=user_col,
        recommendation_col=recommendation_col,
        ground_truth_col=ground_truth_col,
        rank_col=rank_col,
    )
    if prepared.user_count == 0:
        return 0.0

    hits = 0
    for user_id, relevant_items in prepared.relevant_by_user.items():
        ranked_items = prepared.recommendations_by_user.get(user_id, [])[:k]
        seen_items: set[Any] = set()

        for item_id in ranked_items:
            if item_id in seen_items:
                continue
            seen_items.add(item_id)

            if item_id in relevant_items:
                hits += 1
                break

    return hits / prepared.user_count


def ndcg_at_k(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int,
    *,
    user_col: str = DEFAULT_USER_COLUMN,
    recommendation_col: str | None = None,
    ground_truth_col: str = DEFAULT_GROUND_TRUTH_COLUMN,
    rank_col: str = DEFAULT_RANK_COLUMN,
) -> float:
    """Return mean nDCG@K over users with ground truth.

    Relevance is binary. Users may have one or many relevant items. Duplicate
    recommendations do not add repeated gain, but they keep their ranked
    positions, so duplicates can still displace later unique recommendations.
    Users without ground truth are excluded from the denominator.
    """

    prepared = _prepare_utility_data(
        recommendations,
        ground_truth,
        k,
        user_col=user_col,
        recommendation_col=recommendation_col,
        ground_truth_col=ground_truth_col,
        rank_col=rank_col,
    )
    if prepared.user_count == 0:
        return 0.0

    total = 0.0
    for user_id, relevant_items in prepared.relevant_by_user.items():
        ranked_items = prepared.recommendations_by_user.get(user_id, [])[:k]
        dcg = _discounted_gain(ranked_items, relevant_items)
        idcg = _ideal_discounted_gain(len(relevant_items), k)
        total += dcg / idcg if idcg > 0 else 0.0

    return total / prepared.user_count


def evaluate_utility_at_k(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int,
    *,
    user_col: str = DEFAULT_USER_COLUMN,
    recommendation_col: str | None = None,
    ground_truth_col: str = DEFAULT_GROUND_TRUTH_COLUMN,
    rank_col: str = DEFAULT_RANK_COLUMN,
) -> dict[str, float | int]:
    """Return a compact utility metric summary for one cutoff.

    The returned dictionary contains ``k``, ``user_count``, ``hit_rate`` and
    ``ndcg``. It is intentionally simple so callers can convert it directly to a
    row in a CSV or a pandas dataframe.
    """

    prepared = _prepare_utility_data(
        recommendations,
        ground_truth,
        k,
        user_col=user_col,
        recommendation_col=recommendation_col,
        ground_truth_col=ground_truth_col,
        rank_col=rank_col,
    )

    if prepared.user_count == 0:
        return {"k": k, "user_count": 0, "hit_rate": 0.0, "ndcg": 0.0}

    hits = 0
    ndcg_total = 0.0
    for user_id, relevant_items in prepared.relevant_by_user.items():
        ranked_items = prepared.recommendations_by_user.get(user_id, [])[:k]
        seen_items: set[Any] = set()
        hit = False

        for item_id in ranked_items:
            if item_id in seen_items:
                continue
            seen_items.add(item_id)
            if item_id in relevant_items:
                hit = True
                break

        if hit:
            hits += 1

        dcg = _discounted_gain(ranked_items, relevant_items)
        idcg = _ideal_discounted_gain(len(relevant_items), k)
        ndcg_total += dcg / idcg if idcg > 0 else 0.0

    return {
        "k": k,
        "user_count": prepared.user_count,
        "hit_rate": hits / prepared.user_count,
        "ndcg": ndcg_total / prepared.user_count,
    }


def _prepare_utility_data(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    k: int,
    *,
    user_col: str,
    recommendation_col: str | None,
    ground_truth_col: str,
    rank_col: str,
) -> _PreparedUtilityData:
    _validate_k(k)
    normalized_user_col = _normalize_column_name(user_col)
    normalized_ground_truth_col = _normalize_column_name(ground_truth_col)
    normalized_rank_col = _normalize_column_name(rank_col)
    normalized_recommendation_col = (
        _normalize_column_name(recommendation_col)
        if recommendation_col is not None
        else None
    )

    relevant_by_user = _relevant_items_by_user(
        ground_truth,
        user_col=normalized_user_col,
        ground_truth_col=normalized_ground_truth_col,
    )
    recommendations_by_user = _recommendations_by_user(
        recommendations,
        user_col=normalized_user_col,
        recommendation_col=normalized_recommendation_col,
        rank_col=normalized_rank_col,
    )
    return _PreparedUtilityData(
        relevant_by_user=relevant_by_user,
        recommendations_by_user=recommendations_by_user,
    )


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def _relevant_items_by_user(
    ground_truth: pd.DataFrame,
    *,
    user_col: str,
    ground_truth_col: str,
) -> dict[Any, set[Any]]:
    ground_truth = _normalize_dataframe(ground_truth, frame_name="ground_truth")
    if ground_truth.empty:
        return {}

    _require_columns(
        ground_truth,
        frame_name="ground_truth",
        columns=(user_col, ground_truth_col),
    )

    usable = ground_truth.loc[
        ~_is_missing(ground_truth[user_col])
        & ~_is_missing(ground_truth[ground_truth_col]),
        [user_col, ground_truth_col],
    ]
    if usable.empty:
        return {}

    return {
        user_id: set(group[ground_truth_col].tolist())
        for user_id, group in usable.groupby(user_col, sort=False)
    }


def _recommendations_by_user(
    recommendations: pd.DataFrame,
    *,
    user_col: str,
    recommendation_col: str | None,
    rank_col: str,
) -> dict[Any, list[Any]]:
    recommendations = _normalize_dataframe(
        recommendations,
        frame_name="recommendations",
    )
    if recommendations.empty:
        return {}

    _require_columns(
        recommendations,
        frame_name="recommendations",
        columns=(user_col,),
    )
    item_values = _recommendation_values(
        recommendations,
        recommendation_col=recommendation_col,
    )

    usable = recommendations.copy()
    usable["__item_value"] = item_values
    usable["__input_order"] = range(len(usable))
    usable = usable.loc[
        ~_is_missing(usable[user_col]) & ~_is_missing(usable["__item_value"])
    ]
    if usable.empty:
        return {}

    if rank_col in usable.columns:
        usable["__rank_value"] = pd.to_numeric(usable[rank_col], errors="coerce")
        usable["__rank_missing"] = usable["__rank_value"].isna()
        sort_columns = [
            user_col,
            "__rank_missing",
            "__rank_value",
            "__input_order",
        ]
    else:
        sort_columns = [user_col, "__input_order"]

    usable = usable.sort_values(sort_columns, kind="mergesort")

    return {
        user_id: group["__item_value"].tolist()
        for user_id, group in usable.groupby(user_col, sort=False)
    }


def _recommendation_values(
    recommendations: pd.DataFrame,
    *,
    recommendation_col: str | None,
) -> pd.Series:
    if recommendation_col is not None:
        _require_columns(
            recommendations,
            frame_name="recommendations",
            columns=(recommendation_col,),
        )
        return recommendations[recommendation_col]

    available_columns = [
        column
        for column in DEFAULT_RECOMMENDATION_COLUMNS
        if column in recommendations.columns
    ]
    if not available_columns:
        names = ", ".join(DEFAULT_RECOMMENDATION_COLUMNS)
        raise ValueError(
            "recommendations dataframe must contain a recommendation column: "
            f"{names}"
        )

    item_values = pd.Series(pd.NA, index=recommendations.index, dtype="object")
    for column in available_columns:
        fill_mask = _is_missing(item_values) & ~_is_missing(recommendations[column])
        item_values.loc[fill_mask] = recommendations.loc[fill_mask, column]

    return item_values


def _discounted_gain(ranked_items: list[Any], relevant_items: set[Any]) -> float:
    seen_items: set[Any] = set()
    gain = 0.0

    for position, item_id in enumerate(ranked_items, start=1):
        if item_id in seen_items:
            continue
        seen_items.add(item_id)

        if item_id in relevant_items:
            gain += 1.0 / log2(position + 1)

    return gain


def _ideal_discounted_gain(relevant_count: int, k: int) -> float:
    ideal_hits = min(relevant_count, k)
    return sum(1.0 / log2(position + 1) for position in range(1, ideal_hits + 1))


def _normalize_dataframe(df: pd.DataFrame, *, frame_name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{frame_name} must be a pandas DataFrame")

    normalized = df.copy()
    normalized.columns = [
        _normalize_column_name(column) for column in normalized.columns
    ]

    duplicated = normalized.columns[normalized.columns.duplicated()].unique().tolist()
    if duplicated:
        names = ", ".join(duplicated)
        raise ValueError(
            f"{frame_name} dataframe has duplicate column name(s) after "
            f"normalization: {names}"
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
        names = ", ".join(missing)
        raise ValueError(
            f"{frame_name} dataframe is missing required column(s): {names}"
        )


def _is_missing(series: pd.Series) -> pd.Series:
    as_text = series.astype("string")
    return series.isna() | as_text.str.strip().eq("")

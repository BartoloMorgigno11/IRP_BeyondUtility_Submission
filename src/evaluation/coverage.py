"""Catalog coverage metrics for ranked recommendation outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_USER_COLUMN = "user_id"
DEFAULT_MODEL_COLUMN = "model_name"
DEFAULT_RUN_COLUMN = "run_id"
DEFAULT_RANK_COLUMN = "rank"
DEFAULT_RECOMMENDATION_COLUMNS = ("recommended_item_id", "recommended_item_text")
DEFAULT_CATALOG_COLUMNS = ("item_id", "item_title", "title", "name")

COVERAGE_BY_MODEL_COLUMNS = [
    "model_name",
    "run_id",
    "k",
    "metric",
    "value",
    "catalog_size",
    "unique_recommended_items",
]


def catalog_coverage_at_k(
    recommendations: pd.DataFrame,
    catalog: pd.DataFrame,
    k: int,
    *,
    user_col: str = DEFAULT_USER_COLUMN,
    recommendation_col: str | None = None,
    catalog_col: str | None = None,
    rank_col: str = DEFAULT_RANK_COLUMN,
) -> float:
    """Return Catalog Coverage@K.

    The metric is the fraction of unique catalog items that appear at least
    once in users' top-k recommendation lists. Duplicate recommendations,
    whether repeated for one user or across users, contribute only once.

    Empty recommendation inputs return ``0.0``. Empty catalogs also return
    ``0.0`` because the denominator is unavailable; callers that need stricter
    behavior should validate catalog availability before calling this function.
    """

    summary = _coverage_counts(
        recommendations,
        catalog,
        k,
        user_col=user_col,
        recommendation_col=recommendation_col,
        catalog_col=catalog_col,
        rank_col=rank_col,
    )
    catalog_size = summary["catalog_size"]
    if catalog_size == 0:
        return 0.0
    return summary["unique_recommended_items"] / catalog_size


def coverage_by_model(
    recommendations: pd.DataFrame,
    catalog: pd.DataFrame,
    k: int,
    *,
    model_col: str = DEFAULT_MODEL_COLUMN,
    run_col: str = DEFAULT_RUN_COLUMN,
    user_col: str = DEFAULT_USER_COLUMN,
    recommendation_col: str | None = None,
    catalog_col: str | None = None,
    rank_col: str = DEFAULT_RANK_COLUMN,
) -> pd.DataFrame:
    """Return Catalog Coverage@K rows grouped by model and run.

    The returned dataframe is intentionally CSV-ready and deterministic. If the
    recommendations dataframe is empty, an empty dataframe with stable columns
    is returned.
    """

    _validate_k(k)
    frame = _normalize_dataframe(recommendations, frame_name="recommendations")
    if frame.empty:
        return pd.DataFrame(columns=COVERAGE_BY_MODEL_COLUMNS)

    model_col = _normalize_column_name(model_col)
    run_col = _normalize_column_name(run_col)
    _require_columns(frame, frame_name="recommendations", columns=(model_col,))

    group_columns = [model_col]
    has_run_col = run_col in frame.columns
    if has_run_col:
        group_columns.append(run_col)

    rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        model_name = group_key[0]
        run_id = group_key[1] if has_run_col else pd.NA
        counts = _coverage_counts(
            group,
            catalog,
            k,
            user_col=user_col,
            recommendation_col=recommendation_col,
            catalog_col=catalog_col,
            rank_col=rank_col,
        )
        catalog_size = counts["catalog_size"]
        value = (
            counts["unique_recommended_items"] / catalog_size
            if catalog_size
            else 0.0
        )
        rows.append(
            {
                "model_name": model_name,
                "run_id": run_id,
                "k": k,
                "metric": "catalog_coverage",
                "value": value,
                "catalog_size": catalog_size,
                "unique_recommended_items": counts["unique_recommended_items"],
            }
        )

    return pd.DataFrame(rows, columns=COVERAGE_BY_MODEL_COLUMNS)


def _coverage_counts(
    recommendations: pd.DataFrame,
    catalog: pd.DataFrame,
    k: int,
    *,
    user_col: str,
    recommendation_col: str | None,
    catalog_col: str | None,
    rank_col: str,
) -> dict[str, int]:
    _validate_k(k)
    catalog_items = _catalog_items(catalog, catalog_col=catalog_col)
    if not catalog_items:
        return {"catalog_size": 0, "unique_recommended_items": 0}

    recommended_items = set(
        _top_k_recommendation_values(
            recommendations,
            k=k,
            user_col=user_col,
            recommendation_col=recommendation_col,
            rank_col=rank_col,
        )
    )
    covered_items = recommended_items.intersection(catalog_items)
    return {
        "catalog_size": len(catalog_items),
        "unique_recommended_items": len(covered_items),
    }


def _catalog_items(catalog: pd.DataFrame, *, catalog_col: str | None) -> set[str]:
    frame = _normalize_dataframe(catalog, frame_name="catalog")
    if frame.empty:
        return set()

    column = _select_catalog_column(frame, catalog_col)
    values = _canonical_values(frame[column])
    return set(values.tolist())


def _top_k_recommendation_values(
    recommendations: pd.DataFrame,
    *,
    k: int,
    user_col: str,
    recommendation_col: str | None,
    rank_col: str,
) -> list[str]:
    frame = _normalize_dataframe(recommendations, frame_name="recommendations")
    if frame.empty:
        return []

    user_col = _normalize_column_name(user_col)
    rank_col = _normalize_column_name(rank_col)
    item_values = _recommendation_values(frame, recommendation_col=recommendation_col)

    usable = frame.copy()
    usable["__item_value"] = item_values
    usable["__input_order"] = range(len(usable))
    usable = usable.loc[~_is_empty_series(usable["__item_value"])].copy()
    if usable.empty:
        return []

    sort_columns = ["__input_order"]
    if rank_col in usable.columns:
        usable["__rank_value"] = pd.to_numeric(usable[rank_col], errors="coerce")
        usable["__rank_missing"] = usable["__rank_value"].isna()
        sort_columns = ["__rank_missing", "__rank_value", "__input_order"]

    if user_col in usable.columns:
        usable = usable.sort_values([user_col, *sort_columns], kind="mergesort")
        usable = usable.groupby(user_col, sort=False).head(k)
    else:
        usable = usable.sort_values(sort_columns, kind="mergesort").head(k)

    return _canonical_values(usable["__item_value"]).tolist()


def _recommendation_values(
    recommendations: pd.DataFrame,
    *,
    recommendation_col: str | None,
) -> pd.Series:
    if recommendation_col is not None:
        column = _normalize_column_name(recommendation_col)
        _require_columns(
            recommendations,
            frame_name="recommendations",
            columns=(column,),
        )
        return recommendations[column]

    available = [
        column
        for column in DEFAULT_RECOMMENDATION_COLUMNS
        if column in recommendations.columns
    ]
    if not available:
        raise ValueError(
            "recommendations dataframe must contain a recommendation column: "
            f"{', '.join(DEFAULT_RECOMMENDATION_COLUMNS)}"
        )

    values = pd.Series(pd.NA, index=recommendations.index, dtype="object")
    for column in available:
        fill_mask = _is_empty_series(values) & ~_is_empty_series(recommendations[column])
        values.loc[fill_mask] = recommendations.loc[fill_mask, column]
    return values


def _select_catalog_column(frame: pd.DataFrame, catalog_col: str | None) -> str:
    if catalog_col is not None:
        column = _normalize_column_name(catalog_col)
        _require_columns(frame, frame_name="catalog", columns=(column,))
        return column

    for column in DEFAULT_CATALOG_COLUMNS:
        if column in frame.columns:
            return column

    raise ValueError(
        "catalog dataframe must contain a catalog item column: "
        f"{', '.join(DEFAULT_CATALOG_COLUMNS)}"
    )


def _canonical_values(series: pd.Series) -> pd.Series:
    as_text = series.astype("string").str.strip()
    return as_text.loc[~series.isna() & as_text.ne("")]


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


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


def _is_empty_series(series: pd.Series) -> pd.Series:
    as_text = series.astype("string")
    return series.isna() | as_text.str.strip().eq("")

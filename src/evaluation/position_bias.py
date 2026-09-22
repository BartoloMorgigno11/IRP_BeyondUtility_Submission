"""Candidate position bias utilities for normalized recommendation outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_USER_COLUMN = "user_id"
DEFAULT_MODEL_COLUMN = "model_name"
DEFAULT_RUN_COLUMN = "run_id"
DEFAULT_RANK_COLUMN = "rank"
DEFAULT_CANDIDATE_ITEM_COLUMN = "candidate_item_id"
DEFAULT_CANDIDATE_TEXT_COLUMN = "candidate_item_title"
DEFAULT_CANDIDATE_POSITION_COLUMN = "candidate_position"
DEFAULT_RECOMMENDATION_ID_COLUMN = "recommended_item_id"
DEFAULT_RECOMMENDATION_TEXT_COLUMN = "recommended_item_text"

POSITION_BIAS_SUMMARY_COLUMNS = [
    "model_name",
    "run_id",
    "k",
    "metric",
    "candidate_position",
    "count",
    "fraction",
    "matched_recommendations",
    "missing_recommendations",
    "duplicate_recommendations",
    "empty_recommendation_rows",
    "total_recommendation_rows",
]


def selected_candidate_position_distribution(
    recommendations: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    k: int | None = None,
    user_col: str = DEFAULT_USER_COLUMN,
    rank_col: str = DEFAULT_RANK_COLUMN,
    recommendation_id_col: str = DEFAULT_RECOMMENDATION_ID_COLUMN,
    recommendation_text_col: str = DEFAULT_RECOMMENDATION_TEXT_COLUMN,
    candidate_item_col: str = DEFAULT_CANDIDATE_ITEM_COLUMN,
    candidate_text_col: str = DEFAULT_CANDIDATE_TEXT_COLUMN,
    candidate_position_col: str = DEFAULT_CANDIDATE_POSITION_COLUMN,
) -> dict[str, Any]:
    """Return the selected candidate-position distribution.

    Recommendations are matched to candidates first by candidate item ID and
    then, when available, by exact candidate text/title. Duplicate selections of
    the same candidate for the same user are counted once. Empty recommendation
    rows and recommendations with no matching candidate are reported separately
    instead of being folded into the distribution.
    """

    if k is not None:
        _validate_k(k)

    user_col = _normalize_column_name(user_col)
    rank_col = _normalize_column_name(rank_col)
    recommendation_id_col = _normalize_column_name(recommendation_id_col)
    recommendation_text_col = _normalize_column_name(recommendation_text_col)
    candidate_item_col = _normalize_column_name(candidate_item_col)
    candidate_text_col = _normalize_column_name(candidate_text_col)
    candidate_position_col = _normalize_column_name(candidate_position_col)

    candidate_frame = _prepare_candidates(
        candidates,
        user_col=user_col,
        candidate_item_col=candidate_item_col,
        candidate_text_col=candidate_text_col,
        candidate_position_col=candidate_position_col,
    )
    recommendation_frame = _prepare_recommendations(
        recommendations,
        user_col=user_col,
        rank_col=rank_col,
        recommendation_id_col=recommendation_id_col,
        recommendation_text_col=recommendation_text_col,
    )

    id_lookup, text_lookup = _candidate_lookups(
        candidate_frame,
        user_col=user_col,
        candidate_item_col=candidate_item_col,
        candidate_text_col=candidate_text_col,
        candidate_position_col=candidate_position_col,
    )

    top_recommendations = _top_k_recommendations(
        recommendation_frame,
        k=k,
        user_col=user_col,
        rank_col=rank_col,
    )

    position_counts: dict[int, int] = {}
    seen_recommendations: set[tuple[str, str, str]] = set()
    seen_candidates: set[tuple[str, str]] = set()
    total_rows = 0
    empty_rows = 0
    duplicate_rows = 0
    missing_rows = 0
    matched_rows = 0

    for _, row in top_recommendations.iterrows():
        total_rows += 1
        user_key = _canonical_value(row[user_col])
        rec_id = (
            _canonical_value(row[recommendation_id_col])
            if recommendation_id_col in top_recommendations.columns
            else ""
        )
        rec_text = (
            _canonical_value(row[recommendation_text_col])
            if recommendation_text_col in top_recommendations.columns
            else ""
        )

        if not rec_id and not rec_text:
            empty_rows += 1
            continue

        recommendation_key = (user_key, rec_id, rec_text)
        if recommendation_key in seen_recommendations:
            duplicate_rows += 1
            continue
        seen_recommendations.add(recommendation_key)

        match = _match_candidate(
            user_key=user_key,
            rec_id=rec_id,
            rec_text=rec_text,
            id_lookup=id_lookup,
            text_lookup=text_lookup,
        )
        if match is None:
            missing_rows += 1
            continue

        position, candidate_key = match
        matched_candidate_key = (user_key, candidate_key)
        if matched_candidate_key in seen_candidates:
            duplicate_rows += 1
            continue
        seen_candidates.add(matched_candidate_key)

        position_counts[position] = position_counts.get(position, 0) + 1
        matched_rows += 1

    distribution = {
        position: count / matched_rows
        for position, count in sorted(position_counts.items())
    } if matched_rows else {}

    return {
        "k": k,
        "position_counts": dict(sorted(position_counts.items())),
        "position_distribution": distribution,
        "matched_recommendations": matched_rows,
        "missing_recommendations": missing_rows,
        "duplicate_recommendations": duplicate_rows,
        "empty_recommendation_rows": empty_rows,
        "total_recommendation_rows": total_rows,
    }


def position_bias_summary(
    recommendations: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    k: int | None = None,
    model_col: str = DEFAULT_MODEL_COLUMN,
    run_col: str = DEFAULT_RUN_COLUMN,
    user_col: str = DEFAULT_USER_COLUMN,
    rank_col: str = DEFAULT_RANK_COLUMN,
    recommendation_id_col: str = DEFAULT_RECOMMENDATION_ID_COLUMN,
    recommendation_text_col: str = DEFAULT_RECOMMENDATION_TEXT_COLUMN,
    candidate_item_col: str = DEFAULT_CANDIDATE_ITEM_COLUMN,
    candidate_text_col: str = DEFAULT_CANDIDATE_TEXT_COLUMN,
    candidate_position_col: str = DEFAULT_CANDIDATE_POSITION_COLUMN,
) -> pd.DataFrame:
    """Return CSV-ready selected-position fractions grouped by model and run."""

    if k is not None:
        _validate_k(k)

    recommendation_frame = _normalize_dataframe(recommendations, frame_name="recommendations")
    if recommendation_frame.empty:
        return pd.DataFrame(columns=POSITION_BIAS_SUMMARY_COLUMNS)

    model_col = _normalize_column_name(model_col)
    run_col = _normalize_column_name(run_col)

    group_columns: list[str] = []
    if model_col in recommendation_frame.columns:
        group_columns.append(model_col)
    if run_col in recommendation_frame.columns:
        group_columns.append(run_col)

    if not group_columns:
        recommendation_frame = recommendation_frame.copy()
        recommendation_frame["__model_name"] = "unknown"
        group_columns = ["__model_name"]

    rows: list[dict[str, Any]] = []
    grouped = recommendation_frame.groupby(group_columns, sort=False, dropna=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key_map = dict(zip(group_columns, group_key, strict=False))
        model_name = key_map.get(model_col, key_map.get("__model_name", "unknown"))
        run_id = key_map.get(run_col, pd.NA)

        distribution = selected_candidate_position_distribution(
            group,
            candidates,
            k=k,
            user_col=user_col,
            rank_col=rank_col,
            recommendation_id_col=recommendation_id_col,
            recommendation_text_col=recommendation_text_col,
            candidate_item_col=candidate_item_col,
            candidate_text_col=candidate_text_col,
            candidate_position_col=candidate_position_col,
        )
        counts = distribution["position_counts"]

        if not counts:
            rows.append(
                _summary_row(
                    model_name=model_name,
                    run_id=run_id,
                    k=k,
                    candidate_position=pd.NA,
                    count=0,
                    fraction=0.0,
                    distribution=distribution,
                )
            )
            continue

        for position, count in counts.items():
            rows.append(
                _summary_row(
                    model_name=model_name,
                    run_id=run_id,
                    k=k,
                    candidate_position=position,
                    count=count,
                    fraction=distribution["position_distribution"][position],
                    distribution=distribution,
                )
            )

    return pd.DataFrame(rows, columns=POSITION_BIAS_SUMMARY_COLUMNS)


def _prepare_candidates(
    candidates: pd.DataFrame,
    *,
    user_col: str,
    candidate_item_col: str,
    candidate_text_col: str,
    candidate_position_col: str,
) -> pd.DataFrame:
    frame = _normalize_dataframe(candidates, frame_name="candidates")
    _require_columns(
        frame,
        frame_name="candidates",
        columns=(user_col, candidate_position_col),
    )
    if candidate_item_col not in frame.columns and candidate_text_col not in frame.columns:
        raise ValueError(
            "candidates dataframe must contain candidate_item_id or candidate_item_title"
        )
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["__candidate_position_value"] = pd.to_numeric(
        frame[candidate_position_col],
        errors="coerce",
    )
    if frame["__candidate_position_value"].isna().any():
        raise ValueError("candidates dataframe has non-numeric candidate_position values")
    return frame


def _prepare_recommendations(
    recommendations: pd.DataFrame,
    *,
    user_col: str,
    rank_col: str,
    recommendation_id_col: str,
    recommendation_text_col: str,
) -> pd.DataFrame:
    frame = _normalize_dataframe(recommendations, frame_name="recommendations")
    if frame.empty:
        return frame
    _require_columns(frame, frame_name="recommendations", columns=(user_col,))
    if (
        recommendation_id_col not in frame.columns
        and recommendation_text_col not in frame.columns
    ):
        raise ValueError(
            "recommendations dataframe must contain recommended_item_id or recommended_item_text"
        )
    return frame


def _candidate_lookups(
    candidates: pd.DataFrame,
    *,
    user_col: str,
    candidate_item_col: str,
    candidate_text_col: str,
    candidate_position_col: str,
) -> tuple[dict[tuple[str, str], tuple[int, str]], dict[tuple[str, str], tuple[int, str]]]:
    id_lookup: dict[tuple[str, str], tuple[int, str]] = {}
    text_lookup: dict[tuple[str, str], tuple[int, str]] = {}

    for _, row in candidates.iterrows():
        user_key = _canonical_value(row[user_col])
        position = _position_value(row[candidate_position_col])

        item_key = (
            _canonical_value(row[candidate_item_col])
            if candidate_item_col in candidates.columns
            else ""
        )
        text_key = (
            _canonical_value(row[candidate_text_col])
            if candidate_text_col in candidates.columns
            else ""
        )
        identity = item_key or f"text:{text_key}|position:{position}"

        if item_key:
            id_lookup.setdefault((user_key, item_key), (position, identity))
        if text_key:
            text_lookup.setdefault((user_key, text_key), (position, identity))

    return id_lookup, text_lookup


def _top_k_recommendations(
    recommendations: pd.DataFrame,
    *,
    k: int | None,
    user_col: str,
    rank_col: str,
) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations.copy()

    sortable = recommendations.copy()
    sortable["__input_order"] = range(len(sortable))
    sort_columns = ["__input_order"]
    if rank_col in sortable.columns:
        sortable["__rank_value"] = pd.to_numeric(sortable[rank_col], errors="coerce")
        sortable["__rank_missing"] = sortable["__rank_value"].isna()
        sort_columns = ["__rank_missing", "__rank_value", "__input_order"]

    sortable = sortable.sort_values([user_col, *sort_columns], kind="mergesort")
    if k is not None:
        sortable = sortable.groupby(user_col, sort=False).head(k)

    drop_columns = [column for column in ("__input_order", "__rank_value", "__rank_missing") if column in sortable.columns]
    return sortable.drop(columns=drop_columns)


def _match_candidate(
    *,
    user_key: str,
    rec_id: str,
    rec_text: str,
    id_lookup: dict[tuple[str, str], tuple[int, str]],
    text_lookup: dict[tuple[str, str], tuple[int, str]],
) -> tuple[int, str] | None:
    if rec_id and (user_key, rec_id) in id_lookup:
        return id_lookup[(user_key, rec_id)]
    if rec_text and (user_key, rec_text) in text_lookup:
        return text_lookup[(user_key, rec_text)]
    return None


def _summary_row(
    *,
    model_name: Any,
    run_id: Any,
    k: int | None,
    candidate_position: Any,
    count: int,
    fraction: float,
    distribution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "run_id": run_id,
        "k": k,
        "metric": "selected_candidate_position_fraction",
        "candidate_position": candidate_position,
        "count": count,
        "fraction": fraction,
        "matched_recommendations": distribution["matched_recommendations"],
        "missing_recommendations": distribution["missing_recommendations"],
        "duplicate_recommendations": distribution["duplicate_recommendations"],
        "empty_recommendation_rows": distribution["empty_recommendation_rows"],
        "total_recommendation_rows": distribution["total_recommendation_rows"],
    }


def _position_value(value: Any) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"candidate_position must be numeric, got {value!r}")
    return int(numeric)


def _canonical_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


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

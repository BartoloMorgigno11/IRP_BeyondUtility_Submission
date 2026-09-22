"""Hallucination metrics for recommendation outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.output_parser import (
    DEFAULT_ITEM_ID_COLUMN,
    DEFAULT_ITEM_TITLE_COLUMN,
    MATCH_EXACT,
    MATCH_FUZZY,
    MATCH_ITEM_ID,
    MATCH_NORMALIZED,
    MATCH_UNMATCHED,
    MatchResult,
    match_catalog_item,
)


DEFAULT_USER_COLUMN = "user_id"
DEFAULT_RECOMMENDED_ITEM_ID_COLUMN = "recommended_item_id"
DEFAULT_RECOMMENDED_ITEM_TEXT_COLUMN = "recommended_item_text"
DEFAULT_CANDIDATE_ITEM_ID_COLUMN = "candidate_item_id"
DEFAULT_CANDIDATE_ITEM_TITLE_COLUMN = "candidate_item_title"
DEFAULT_MATCH_TYPE_COLUMN = "match_type"
DEFAULT_RANK_COLUMN = "rank"

KNOWN_MATCH_TYPES = (
    MATCH_ITEM_ID,
    MATCH_EXACT,
    MATCH_NORMALIZED,
    MATCH_FUZZY,
    MATCH_UNMATCHED,
)


def hallucination_rate(
    recommendations: pd.DataFrame,
    *,
    match_type_col: str = DEFAULT_MATCH_TYPE_COLUMN,
    rank_col: str = DEFAULT_RANK_COLUMN,
    user_col: str = DEFAULT_USER_COLUMN,
    k: int | None = None,
) -> dict[str, float | int]:
    """Summarize hallucination rate from explicit match-type labels.

    ``hallucination_rate`` is the share of parsed recommendations labeled
    ``unmatched``. Fuzzy matches are counted separately and are never counted as
    exact matches. ``strict_hallucination_rate`` treats both fuzzy and unmatched
    outputs as unsafe for exact utility evaluation.
    """

    frame = _normalize_dataframe(recommendations, frame_name="recommendations")
    if frame.empty:
        return _summary_from_counts({})

    match_type_col = _normalize_column_name(match_type_col)
    rank_col = _normalize_column_name(rank_col)
    user_col = _normalize_column_name(user_col)
    _require_columns(
        frame,
        frame_name="recommendations",
        columns=(match_type_col,),
    )
    frame = _limit_to_k(frame, k=k, user_col=user_col, rank_col=rank_col)

    match_types = frame[match_type_col].map(_normalize_match_type)
    unknown = sorted(
        {
            match_type
            for match_type in match_types.dropna().unique().tolist()
            if match_type not in KNOWN_MATCH_TYPES
        }
    )
    if unknown:
        raise ValueError(f"unknown match_type value(s): {', '.join(unknown)}")

    counts = match_types.value_counts(dropna=False).to_dict()
    return _summary_from_counts(counts)


def catalog_hallucination_rate(
    recommendations: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    recommended_item_id_col: str = DEFAULT_RECOMMENDED_ITEM_ID_COLUMN,
    recommended_item_text_col: str = DEFAULT_RECOMMENDED_ITEM_TEXT_COLUMN,
    catalog_item_id_col: str = DEFAULT_ITEM_ID_COLUMN,
    catalog_item_title_col: str = DEFAULT_ITEM_TITLE_COLUMN,
    rank_col: str = DEFAULT_RANK_COLUMN,
    user_col: str = DEFAULT_USER_COLUMN,
    k: int | None = None,
    enable_fuzzy: bool = False,
    fuzzy_threshold: float = 90.0,
) -> dict[str, float | int]:
    """Return hallucination summary against a global item catalog.

    Matching is attempted by item id first, then exact title, normalized title,
    and optionally fuzzy title. Match types remain separate in the returned
    summary.
    """

    frame = _normalize_dataframe(recommendations, frame_name="recommendations")
    if frame.empty:
        return _summary_from_counts({})

    frame = _limit_to_k(
        frame,
        k=k,
        user_col=_normalize_column_name(user_col),
        rank_col=_normalize_column_name(rank_col),
    )
    catalog = _normalize_dataframe(catalog, frame_name="catalog")

    match_types = [
        _match_recommendation_row(
            row,
            catalog=catalog,
            recommended_item_id_col=_normalize_column_name(recommended_item_id_col),
            recommended_item_text_col=_normalize_column_name(recommended_item_text_col),
            catalog_item_id_col=_normalize_column_name(catalog_item_id_col),
            catalog_item_title_col=_normalize_column_name(catalog_item_title_col),
            enable_fuzzy=enable_fuzzy,
            fuzzy_threshold=fuzzy_threshold,
        ).match_type
        for _, row in frame.iterrows()
    ]
    return _summary_from_counts(pd.Series(match_types).value_counts().to_dict())


def candidate_hallucination_rate(
    recommendations: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    user_col: str = DEFAULT_USER_COLUMN,
    recommended_item_id_col: str = DEFAULT_RECOMMENDED_ITEM_ID_COLUMN,
    recommended_item_text_col: str = DEFAULT_RECOMMENDED_ITEM_TEXT_COLUMN,
    candidate_item_id_col: str = DEFAULT_CANDIDATE_ITEM_ID_COLUMN,
    candidate_item_title_col: str = DEFAULT_CANDIDATE_ITEM_TITLE_COLUMN,
    rank_col: str = DEFAULT_RANK_COLUMN,
    k: int | None = None,
    enable_fuzzy: bool = False,
    fuzzy_threshold: float = 90.0,
) -> dict[str, float | int]:
    """Return hallucination summary against configured candidate items.

    If the candidates dataframe contains ``user_id``, each recommendation is
    matched only against that user's candidates. Otherwise the candidate set is
    treated as global.
    """

    frame = _normalize_dataframe(recommendations, frame_name="recommendations")
    if frame.empty:
        return _summary_from_counts({})

    user_col = _normalize_column_name(user_col)
    rank_col = _normalize_column_name(rank_col)
    frame = _limit_to_k(frame, k=k, user_col=user_col, rank_col=rank_col)
    candidates = _normalize_dataframe(candidates, frame_name="candidates")

    has_user_specific_candidates = user_col in candidates.columns
    if has_user_specific_candidates and user_col not in frame.columns:
        raise ValueError(
            "recommendations dataframe is missing required column(s): "
            f"{user_col}"
        )

    match_types: list[str] = []
    for _, row in frame.iterrows():
        candidate_subset = candidates
        if has_user_specific_candidates:
            candidate_subset = candidates.loc[candidates[user_col] == row[user_col]]

        if candidate_subset.empty:
            match_types.append(MATCH_UNMATCHED)
            continue

        match = _match_recommendation_row(
            row,
            catalog=candidate_subset,
            recommended_item_id_col=_normalize_column_name(recommended_item_id_col),
            recommended_item_text_col=_normalize_column_name(recommended_item_text_col),
            catalog_item_id_col=_normalize_column_name(candidate_item_id_col),
            catalog_item_title_col=_normalize_column_name(candidate_item_title_col),
            enable_fuzzy=enable_fuzzy,
            fuzzy_threshold=fuzzy_threshold,
        )
        match_types.append(match.match_type)

    return _summary_from_counts(pd.Series(match_types).value_counts().to_dict())


def _match_recommendation_row(
    row: pd.Series,
    *,
    catalog: pd.DataFrame,
    recommended_item_id_col: str,
    recommended_item_text_col: str,
    catalog_item_id_col: str,
    catalog_item_title_col: str,
    enable_fuzzy: bool,
    fuzzy_threshold: float,
) -> MatchResult:
    if recommended_item_id_col in row.index and not _is_missing_scalar(
        row[recommended_item_id_col]
    ):
        id_match = match_catalog_item(
            row[recommended_item_id_col],
            catalog=catalog,
            item_id_col=catalog_item_id_col,
            item_title_col=catalog_item_title_col,
            enable_fuzzy=False,
        )
        if id_match.match_type != MATCH_UNMATCHED:
            return id_match

    if recommended_item_text_col in row.index and not _is_missing_scalar(
        row[recommended_item_text_col]
    ):
        return match_catalog_item(
            row[recommended_item_text_col],
            catalog=catalog,
            item_id_col=catalog_item_id_col,
            item_title_col=catalog_item_title_col,
            enable_fuzzy=enable_fuzzy,
            fuzzy_threshold=fuzzy_threshold,
        )

    return MatchResult(pd.NA, pd.NA, MATCH_UNMATCHED, None)


def _summary_from_counts(counts: dict[Any, Any]) -> dict[str, float | int]:
    normalized_counts = {
        match_type: int(counts.get(match_type, 0))
        for match_type in KNOWN_MATCH_TYPES
    }
    total = sum(normalized_counts.values())
    hallucinated = normalized_counts[MATCH_UNMATCHED]
    fuzzy = normalized_counts[MATCH_FUZZY]
    exact_like = normalized_counts[MATCH_ITEM_ID] + normalized_counts[MATCH_EXACT]
    strict_hallucinated = hallucinated + fuzzy

    return {
        "total": total,
        "hallucinated": hallucinated,
        "hallucination_rate": _safe_rate(hallucinated, total),
        "strict_hallucinated": strict_hallucinated,
        "strict_hallucination_rate": _safe_rate(strict_hallucinated, total),
        "item_id_matches": normalized_counts[MATCH_ITEM_ID],
        "exact_matches": normalized_counts[MATCH_EXACT],
        "exact_or_item_id_matches": exact_like,
        "normalized_matches": normalized_counts[MATCH_NORMALIZED],
        "fuzzy_matches": fuzzy,
        "unmatched": hallucinated,
    }


def _limit_to_k(
    frame: pd.DataFrame,
    *,
    k: int | None,
    user_col: str,
    rank_col: str,
) -> pd.DataFrame:
    if k is None:
        return frame
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    limited = frame.copy()
    limited["__input_order"] = range(len(limited))
    if rank_col in limited.columns:
        limited["__rank_value"] = pd.to_numeric(limited[rank_col], errors="coerce")
        limited["__rank_missing"] = limited["__rank_value"].isna()
        sort_columns = ["__rank_missing", "__rank_value", "__input_order"]
    else:
        sort_columns = ["__input_order"]

    if user_col in limited.columns:
        limited = limited.sort_values(
            [user_col, *sort_columns],
            kind="mergesort",
        )
        return limited.groupby(user_col, sort=False).head(k)

    return limited.sort_values(sort_columns, kind="mergesort").head(k)


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


def _normalize_match_type(value: object) -> str:
    if _is_missing_scalar(value):
        return MATCH_UNMATCHED
    return str(value).strip().lower()


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


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except ValueError:
        return False
    return str(value).strip() == ""

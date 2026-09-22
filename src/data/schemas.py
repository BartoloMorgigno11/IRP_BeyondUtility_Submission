"""Canonical dataframe schema validation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd


INTERACTIONS_REQUIRED_COLUMNS = ("user_id", "item_id", "split")
INTERACTIONS_OPTIONAL_COLUMNS = ("timestamp", "rating")

CANDIDATES_REQUIRED_COLUMNS = (
    "user_id",
    "candidate_item_id",
    "candidate_position",
)
CANDIDATES_OPTIONAL_COLUMNS = ("is_ground_truth", "candidate_item_title")

RECOMMENDATIONS_REQUIRED_COLUMNS = (
    "user_id",
    "model_name",
    "dataset",
    "rank",
    "run_id",
)
RECOMMENDATION_ITEM_COLUMNS = ("recommended_item_id", "recommended_item_text")
RECOMMENDATIONS_OPTIONAL_COLUMNS = (
    "raw_output",
    "match_type",
    "source_file",
    "prompt_variant",
    "candidate_ordering",
    "history_length",
)

GROUND_TRUTH_REQUIRED_COLUMNS = ("user_id", "ground_truth_item_id")
GROUND_TRUTH_OPTIONAL_COLUMNS = ("dataset", "split")


def validate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an interactions dataframe.

    Required columns are ``user_id``, ``item_id`` and ``split``. Optional
    canonical columns are kept when present, and extra columns are preserved
    after the canonical columns.
    """

    return _validate_frame(
        df,
        frame_name="interactions",
        required_columns=INTERACTIONS_REQUIRED_COLUMNS,
        optional_columns=INTERACTIONS_OPTIONAL_COLUMNS,
    )


def validate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a candidate-set dataframe.

    Required columns are ``user_id``, ``candidate_item_id`` and
    ``candidate_position``. Optional canonical columns are kept when present,
    and extra columns are preserved after the canonical columns.
    """

    return _validate_frame(
        df,
        frame_name="candidates",
        required_columns=CANDIDATES_REQUIRED_COLUMNS,
        optional_columns=CANDIDATES_OPTIONAL_COLUMNS,
    )


def validate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a recommendations dataframe.

    Required columns are ``user_id``, ``model_name``, ``dataset``, ``rank`` and
    ``run_id``. Each row must also contain at least one non-empty value in
    either ``recommended_item_id`` or ``recommended_item_text``.
    """

    normalized = _validate_frame(
        df,
        frame_name="recommendations",
        required_columns=RECOMMENDATIONS_REQUIRED_COLUMNS,
        optional_columns=(
            *RECOMMENDATION_ITEM_COLUMNS,
            *RECOMMENDATIONS_OPTIONAL_COLUMNS,
        ),
    )
    _require_any_column(
        normalized,
        frame_name="recommendations",
        columns=RECOMMENDATION_ITEM_COLUMNS,
    )
    _require_row_values_in_any_column(
        normalized,
        frame_name="recommendations",
        columns=RECOMMENDATION_ITEM_COLUMNS,
    )
    return _order_columns(
        normalized,
        required_columns=RECOMMENDATIONS_REQUIRED_COLUMNS,
        optional_columns=(
            *RECOMMENDATION_ITEM_COLUMNS,
            *RECOMMENDATIONS_OPTIONAL_COLUMNS,
        ),
    )


def validate_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a ground-truth dataframe.

    Required columns are ``user_id`` and ``ground_truth_item_id``. Optional
    canonical columns are kept when present, and extra columns are preserved
    after the canonical columns.
    """

    return _validate_frame(
        df,
        frame_name="ground_truth",
        required_columns=GROUND_TRUTH_REQUIRED_COLUMNS,
        optional_columns=GROUND_TRUTH_OPTIONAL_COLUMNS,
    )


def _validate_frame(
    df: pd.DataFrame,
    *,
    frame_name: str,
    required_columns: Sequence[str],
    optional_columns: Sequence[str],
) -> pd.DataFrame:
    normalized = _normalize_columns(df, frame_name=frame_name)
    _require_columns(
        normalized,
        frame_name=frame_name,
        columns=required_columns,
    )
    _require_non_empty_values(
        normalized,
        frame_name=frame_name,
        columns=required_columns,
    )
    return _order_columns(
        normalized,
        required_columns=required_columns,
        optional_columns=optional_columns,
    )


def _normalize_columns(df: pd.DataFrame, *, frame_name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{frame_name} must be a pandas DataFrame")

    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    duplicated = normalized.columns[normalized.columns.duplicated()].unique().tolist()
    if duplicated:
        names = ", ".join(duplicated)
        raise ValueError(
            f"{frame_name} dataframe has duplicate column name(s) after "
            f"normalization: {names}"
        )

    return normalized


def _require_columns(
    df: pd.DataFrame,
    *,
    frame_name: str,
    columns: Iterable[str],
) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            f"{frame_name} dataframe is missing required column(s): {names}"
        )


def _require_any_column(
    df: pd.DataFrame,
    *,
    frame_name: str,
    columns: Iterable[str],
) -> None:
    available = [column for column in columns if column in df.columns]
    if not available:
        names = ", ".join(columns)
        raise ValueError(
            f"{frame_name} dataframe must contain at least one of: {names}"
        )


def _require_non_empty_values(
    df: pd.DataFrame,
    *,
    frame_name: str,
    columns: Iterable[str],
) -> None:
    empty_columns = [
        column for column in columns if _is_empty_value(df[column]).any()
    ]
    if empty_columns:
        names = ", ".join(empty_columns)
        raise ValueError(
            f"{frame_name} dataframe has missing value(s) in required "
            f"column(s): {names}"
        )


def _require_row_values_in_any_column(
    df: pd.DataFrame,
    *,
    frame_name: str,
    columns: Iterable[str],
) -> None:
    available = [column for column in columns if column in df.columns]
    has_value = pd.Series(False, index=df.index)

    for column in available:
        has_value = has_value | ~_is_empty_value(df[column])

    if not has_value.all():
        raise ValueError(
            f"{frame_name} dataframe has row(s) without any recommendation "
            f"value in: {', '.join(columns)}"
        )


def _is_empty_value(series: pd.Series) -> pd.Series:
    as_text = series.astype("string")
    return series.isna() | as_text.str.strip().eq("")


def _order_columns(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    optional_columns: Sequence[str],
) -> pd.DataFrame:
    canonical_columns = [
        column
        for column in (*required_columns, *optional_columns)
        if column in df.columns
    ]
    extra_columns = [column for column in df.columns if column not in canonical_columns]
    return df.loc[:, [*canonical_columns, *extra_columns]]

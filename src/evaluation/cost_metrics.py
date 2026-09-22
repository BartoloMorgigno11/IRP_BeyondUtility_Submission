"""Cost, token, and latency utilities for LLM recommender outputs.

The functions in this module only consume recorded token/latency fields and
explicit pricing assumptions supplied by the caller. They never call APIs,
download prices, or estimate token counts from text.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import pandas as pd


DEFAULT_MODEL_COLUMN = "model_name"
PROMPT_TOKENS_COLUMN = "prompt_tokens"
COMPLETION_TOKENS_COLUMN = "completion_tokens"
TOTAL_TOKENS_COLUMN = "total_tokens"
LATENCY_SECONDS_COLUMN = "latency_seconds"

TOKEN_SUMMARY_COLUMNS = [
    "model_name",
    "query_count",
    "token_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "average_total_tokens_per_query",
    "token_status",
]

LATENCY_SUMMARY_COLUMNS = [
    "model_name",
    "query_count",
    "latency_count",
    "total_latency_seconds",
    "mean_latency_seconds",
    "min_latency_seconds",
    "max_latency_seconds",
    "latency_status",
]

COST_SUMMARY_COLUMNS = [
    "model_name",
    "query_count",
    "priced_query_count",
    "estimated_cost_total",
    "estimated_cost_per_query",
    "estimated_cost_per_1000_users",
    "cost_status",
]

PROMPT_PRICE_KEYS = ("prompt_per_1k_tokens", "prompt_price_per_1k_tokens")
COMPLETION_PRICE_KEYS = (
    "completion_per_1k_tokens",
    "completion_price_per_1k_tokens",
)
TOTAL_PRICE_KEYS = ("total_per_1k_tokens", "total_price_per_1k_tokens")


def compute_total_tokens(
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    total_tokens: Any = None,
) -> int | None:
    """Return total tokens from explicit counts, or ``None`` if unavailable.

    A provided ``total_tokens`` value is accepted directly. Otherwise both
    ``prompt_tokens`` and ``completion_tokens`` must be present. Missing values
    are not inferred from text or any external tokenizer.
    """

    direct_total = _token_value(total_tokens)
    if direct_total is not None:
        return direct_total

    prompt = _token_value(prompt_tokens)
    completion = _token_value(completion_tokens)
    if prompt is None or completion is None:
        return None
    return prompt + completion


def estimate_query_cost(
    *,
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    total_tokens: Any = None,
    price_record: Mapping[str, Any] | None = None,
) -> float | None:
    """Return estimated cost for one query when tokens and prices are present.

    Separate prompt/completion prices are used when both corresponding token
    counts are available. Otherwise a total-token price is used when available.
    Missing prices or token counts return ``None``.
    """

    if not price_record:
        return None

    prompt_price = _first_price(price_record, PROMPT_PRICE_KEYS)
    completion_price = _first_price(price_record, COMPLETION_PRICE_KEYS)
    prompt = _token_value(prompt_tokens)
    completion = _token_value(completion_tokens)
    if (
        prompt_price is not None
        and completion_price is not None
        and prompt is not None
        and completion is not None
    ):
        return (prompt / 1000.0 * prompt_price) + (completion / 1000.0 * completion_price)

    total_price = _first_price(price_record, TOTAL_PRICE_KEYS)
    total = compute_total_tokens(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    if total_price is None or total is None:
        return None
    return total / 1000.0 * total_price


def estimate_cost_per_1000_users(
    estimated_cost_per_query: Any = None,
    *,
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    total_tokens: Any = None,
    price_record: Mapping[str, Any] | None = None,
) -> float | None:
    """Return estimated cost per 1,000 users, assuming one query per user."""

    query_cost = _nonnegative_number(estimated_cost_per_query)
    if query_cost is None:
        query_cost = estimate_query_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            price_record=price_record,
        )
    if query_cost is None:
        return None
    return query_cost * 1000.0


def summarize_tokens_by_model(
    records: pd.DataFrame,
    *,
    model_col: str = DEFAULT_MODEL_COLUMN,
) -> pd.DataFrame:
    """Summarize recorded token counts by model.

    Missing token counts are reported with ``token_status`` values of
    ``unavailable`` or ``partial`` rather than filled or inferred.
    """

    frame = _normalize_dataframe(records, frame_name="records")
    if frame.empty:
        return pd.DataFrame(columns=TOKEN_SUMMARY_COLUMNS)

    model_col = _normalize_column_name(model_col)
    _require_columns(frame, frame_name="records", columns=(model_col,))

    rows: list[dict[str, Any]] = []
    for model_name, group in frame.groupby(model_col, sort=False, dropna=False):
        total_values = [
            _row_total_tokens(row)
            for _, row in group.iterrows()
        ]
        available_totals = [value for value in total_values if value is not None]
        prompt_values = _available_tokens(group, PROMPT_TOKENS_COLUMN)
        completion_values = _available_tokens(group, COMPLETION_TOKENS_COLUMN)
        token_count = len(available_totals)
        status = _availability_status(token_count, len(group))

        rows.append(
            {
                "model_name": model_name,
                "query_count": len(group),
                "token_count": token_count,
                "prompt_tokens": sum(prompt_values) if prompt_values else pd.NA,
                "completion_tokens": sum(completion_values) if completion_values else pd.NA,
                "total_tokens": sum(available_totals) if available_totals else pd.NA,
                "average_total_tokens_per_query": (
                    sum(available_totals) / token_count if token_count else pd.NA
                ),
                "token_status": status,
            }
        )

    return pd.DataFrame(rows, columns=TOKEN_SUMMARY_COLUMNS)


def summarize_latency_by_model(
    records: pd.DataFrame,
    *,
    model_col: str = DEFAULT_MODEL_COLUMN,
) -> pd.DataFrame:
    """Summarize recorded latency values by model."""

    frame = _normalize_dataframe(records, frame_name="records")
    if frame.empty:
        return pd.DataFrame(columns=LATENCY_SUMMARY_COLUMNS)

    model_col = _normalize_column_name(model_col)
    _require_columns(frame, frame_name="records", columns=(model_col,))

    rows: list[dict[str, Any]] = []
    for model_name, group in frame.groupby(model_col, sort=False, dropna=False):
        latency_values = _available_numbers(group, LATENCY_SECONDS_COLUMN)
        latency_count = len(latency_values)
        status = _availability_status(latency_count, len(group))
        rows.append(
            {
                "model_name": model_name,
                "query_count": len(group),
                "latency_count": latency_count,
                "total_latency_seconds": sum(latency_values) if latency_values else pd.NA,
                "mean_latency_seconds": (
                    sum(latency_values) / latency_count if latency_count else pd.NA
                ),
                "min_latency_seconds": min(latency_values) if latency_values else pd.NA,
                "max_latency_seconds": max(latency_values) if latency_values else pd.NA,
                "latency_status": status,
            }
        )

    return pd.DataFrame(rows, columns=LATENCY_SUMMARY_COLUMNS)


def aggregate_cost_metrics_by_model(
    records: pd.DataFrame,
    pricing: Mapping[str, Any] | None,
    *,
    model_col: str = DEFAULT_MODEL_COLUMN,
) -> pd.DataFrame:
    """Aggregate estimated cost metrics by model using explicit prices."""

    frame = _normalize_dataframe(records, frame_name="records")
    if frame.empty:
        return pd.DataFrame(columns=COST_SUMMARY_COLUMNS)

    model_col = _normalize_column_name(model_col)
    _require_columns(frame, frame_name="records", columns=(model_col,))

    rows: list[dict[str, Any]] = []
    for model_name, group in frame.groupby(model_col, sort=False, dropna=False):
        price_record = _price_for_model(pricing, str(model_name))
        query_costs = [
            estimate_query_cost(
                prompt_tokens=row.get(PROMPT_TOKENS_COLUMN),
                completion_tokens=row.get(COMPLETION_TOKENS_COLUMN),
                total_tokens=row.get(TOTAL_TOKENS_COLUMN),
                price_record=price_record,
            )
            for _, row in group.iterrows()
        ]
        available_costs = [cost for cost in query_costs if cost is not None]
        priced_query_count = len(available_costs)
        status = _availability_status(priced_query_count, len(group))
        total_cost = sum(available_costs) if available_costs else pd.NA
        per_query = (
            total_cost / priced_query_count if priced_query_count else pd.NA
        )
        rows.append(
            {
                "model_name": model_name,
                "query_count": len(group),
                "priced_query_count": priced_query_count,
                "estimated_cost_total": total_cost,
                "estimated_cost_per_query": per_query,
                "estimated_cost_per_1000_users": (
                    estimate_cost_per_1000_users(per_query)
                    if priced_query_count
                    else pd.NA
                ),
                "cost_status": status,
            }
        )

    return pd.DataFrame(rows, columns=COST_SUMMARY_COLUMNS)


def _row_total_tokens(row: pd.Series) -> int | None:
    return compute_total_tokens(
        prompt_tokens=row.get(PROMPT_TOKENS_COLUMN),
        completion_tokens=row.get(COMPLETION_TOKENS_COLUMN),
        total_tokens=row.get(TOTAL_TOKENS_COLUMN),
    )


def _available_tokens(frame: pd.DataFrame, column: str) -> list[int]:
    if column not in frame.columns:
        return []
    values: list[int] = []
    for value in frame[column]:
        token_value = _token_value(value)
        if token_value is not None:
            values.append(token_value)
    return values


def _available_numbers(frame: pd.DataFrame, column: str) -> list[float]:
    if column not in frame.columns:
        return []
    values: list[float] = []
    for value in frame[column]:
        number = _nonnegative_number(value)
        if number is not None:
            values.append(number)
    return values


def _token_value(value: Any) -> int | None:
    number = _nonnegative_number(value)
    if number is None:
        return None
    if not float(number).is_integer():
        raise ValueError(f"token counts must be whole numbers, got {value!r}")
    return int(number)


def _nonnegative_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("numeric values must not be booleans")
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, str) and not value.strip():
        return None

    number = float(value)
    if not isfinite(number):
        return None
    if number < 0:
        raise ValueError(f"numeric values must be non-negative, got {value!r}")
    return number


def _first_price(price_record: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    normalized = {
        _normalize_column_name(key): value
        for key, value in price_record.items()
    }
    for key in keys:
        if key in normalized:
            return _nonnegative_number(normalized[key])
    return None


def _price_for_model(
    pricing: Mapping[str, Any] | None,
    model_name: str,
) -> Mapping[str, Any] | None:
    if not pricing:
        return None

    if "pricing" in pricing and isinstance(pricing["pricing"], Mapping):
        return _price_for_model(pricing["pricing"], model_name)
    if "models" in pricing and isinstance(pricing["models"], Mapping):
        models = pricing["models"]
        price_record = models.get(model_name)
        if isinstance(price_record, Mapping):
            return price_record
        return None

    price_record = pricing.get(model_name)
    if isinstance(price_record, Mapping):
        return price_record
    return None


def _availability_status(available_count: int, total_count: int) -> str:
    if available_count == 0:
        return "unavailable"
    if available_count == total_count:
        return "available"
    return "partial"


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

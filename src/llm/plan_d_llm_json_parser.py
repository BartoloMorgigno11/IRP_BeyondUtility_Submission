from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SmokeJsonValidation:
    """Parsed and validated JSON result for the Plan D Qwen smoke test."""

    parse_success: bool
    schema_valid: bool
    selected_ids: tuple[str, ...]
    selected_count: int
    candidate_validity_rate: float
    duplicate_count: int
    hallucinated_count: int
    parse_error: str | None
    schema_error: str | None


def parse_and_validate_smoke_json(
    response_text: str,
    *,
    candidate_ids: set[str],
    expected_count: int = 3,
) -> SmokeJsonValidation:
    """Strictly parse the toy smoke-test JSON and compute candidate validity."""

    try:
        parsed = json.loads(response_text.strip())
    except json.JSONDecodeError as exc:
        return SmokeJsonValidation(
            parse_success=False,
            schema_valid=False,
            selected_ids=(),
            selected_count=0,
            candidate_validity_rate=0.0,
            duplicate_count=0,
            hallucinated_count=0,
            parse_error=str(exc),
            schema_error=None,
        )

    if not isinstance(parsed, dict):
        return _schema_failure("top-level JSON value is not an object", ())

    ranked_item_ids = parsed.get("ranked_item_ids")
    status_value = parsed.get("status")
    selected_ids = tuple(item for item in ranked_item_ids if isinstance(item, str)) if isinstance(ranked_item_ids, list) else ()
    candidate_validity_rate, duplicate_count, hallucinated_count = _candidate_counts(
        selected_ids,
        candidate_ids,
    )

    schema_errors: list[str] = []
    if not isinstance(ranked_item_ids, list):
        schema_errors.append("ranked_item_ids missing or not a list")
    elif len(ranked_item_ids) != expected_count:
        schema_errors.append(f"ranked_item_ids length is not {expected_count}")
    elif not all(isinstance(item, str) for item in ranked_item_ids):
        schema_errors.append("ranked_item_ids must contain only strings")

    if status_value != "ok":
        schema_errors.append("status must equal ok")

    return SmokeJsonValidation(
        parse_success=True,
        schema_valid=not schema_errors,
        selected_ids=selected_ids,
        selected_count=len(selected_ids),
        candidate_validity_rate=candidate_validity_rate,
        duplicate_count=duplicate_count,
        hallucinated_count=hallucinated_count,
        parse_error=None,
        schema_error="; ".join(schema_errors) if schema_errors else None,
    )


def _schema_failure(message: str, selected_ids: tuple[str, ...]) -> SmokeJsonValidation:
    return SmokeJsonValidation(
        parse_success=True,
        schema_valid=False,
        selected_ids=selected_ids,
        selected_count=len(selected_ids),
        candidate_validity_rate=0.0,
        duplicate_count=0,
        hallucinated_count=0,
        parse_error=None,
        schema_error=message,
    )


def _candidate_counts(selected_ids: tuple[str, ...], candidate_ids: set[str]) -> tuple[float, int, int]:
    selected_count = len(selected_ids)
    if selected_count == 0:
        return 0.0, 0, 0

    valid_count = sum(1 for item_id in selected_ids if item_id in candidate_ids)
    duplicate_count = selected_count - len(set(selected_ids))
    hallucinated_count = selected_count - valid_count
    return valid_count / selected_count, duplicate_count, hallucinated_count

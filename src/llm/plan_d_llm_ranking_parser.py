from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class RankingParseResult:
    """Strict parser result for future Plan D LLM ranking responses."""

    parse_success: bool
    schema_valid: bool
    ranked_item_ids: list[str]
    selected_count: int
    valid_item_ids: list[str]
    invalid_item_ids: list[str]
    duplicate_item_ids: list[str]
    duplicate_count: int
    hallucinated_count: int
    candidate_validity_rate: float
    status: str
    error_message: str


def parse_llm_ranking_response(
    raw_response_text: str,
    *,
    candidate_ids: set[str],
    expected_top_k: int,
) -> RankingParseResult:
    """Parse and validate a strict-JSON ranking response without repairing it."""

    if expected_top_k < 1:
        raise ValueError("expected_top_k must be positive")
    text = str(raw_response_text or "").strip()
    if not text:
        return _empty_result("empty_response", "empty response")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return _empty_result("parse_failed", str(exc))

    if not isinstance(parsed, dict):
        return _empty_result("parsed_invalid_schema", "top-level JSON value must be an object", parse_success=True)

    ranked = parsed.get("ranked_item_ids")
    if not isinstance(ranked, list):
        return _empty_result(
            "parsed_invalid_schema",
            "ranked_item_ids is missing or not a list",
            parse_success=True,
        )

    ranked_item_ids = [str(item) for item in ranked if isinstance(item, str)]
    selected_count = len(ranked_item_ids)
    valid_item_ids = [item_id for item_id in ranked_item_ids if item_id in candidate_ids]
    invalid_item_ids = [item_id for item_id in ranked_item_ids if item_id not in candidate_ids]
    duplicate_item_ids = _duplicate_items(ranked_item_ids)
    duplicate_count = selected_count - len(set(ranked_item_ids))
    hallucinated_count = len(invalid_item_ids)
    candidate_validity_rate = len(valid_item_ids) / selected_count if selected_count else 0.0

    schema_errors: list[str] = []
    if len(ranked) != expected_top_k:
        schema_errors.append(f"ranked_item_ids length is {len(ranked)}, expected {expected_top_k}")
    if not all(isinstance(item, str) for item in ranked):
        schema_errors.append("ranked_item_ids must contain only strings")
    if "status" in parsed and not isinstance(parsed["status"], str):
        schema_errors.append("status field must be a string when present")

    if schema_errors:
        status = "parsed_invalid_schema"
        schema_valid = False
        error_message = "; ".join(schema_errors)
    elif hallucinated_count > 0:
        status = "parsed_with_hallucinations"
        schema_valid = True
        error_message = ""
    elif duplicate_count > 0:
        status = "parsed_with_duplicates"
        schema_valid = True
        error_message = ""
    else:
        status = "parsed_valid"
        schema_valid = True
        error_message = ""

    return RankingParseResult(
        parse_success=True,
        schema_valid=schema_valid,
        ranked_item_ids=ranked_item_ids,
        selected_count=selected_count,
        valid_item_ids=valid_item_ids,
        invalid_item_ids=invalid_item_ids,
        duplicate_item_ids=duplicate_item_ids,
        duplicate_count=duplicate_count,
        hallucinated_count=hallucinated_count,
        candidate_validity_rate=candidate_validity_rate,
        status=status,
        error_message=error_message,
    )


def _empty_result(status: str, error_message: str, *, parse_success: bool = False) -> RankingParseResult:
    return RankingParseResult(
        parse_success=parse_success,
        schema_valid=False,
        ranked_item_ids=[],
        selected_count=0,
        valid_item_ids=[],
        invalid_item_ids=[],
        duplicate_item_ids=[],
        duplicate_count=0,
        hallucinated_count=0,
        candidate_validity_rate=0.0,
        status=status,
        error_message=error_message,
    )


def _duplicate_items(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    duplicate_seen: set[str] = set()
    for value in values:
        if value in seen and value not in duplicate_seen:
            duplicates.append(value)
            duplicate_seen.add(value)
        seen.add(value)
    return duplicates

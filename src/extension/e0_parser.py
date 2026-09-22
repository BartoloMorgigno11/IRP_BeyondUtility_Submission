"""Strict top-10 parser and fail-strict utility helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from src.evaluation.utility_metrics import evaluate_utility_at_k
from src.extension.e0_protocol import METRIC_K, PARSER_CONTRACT_ID, REQUESTED_OUTPUT_LENGTH, stable_json
from src.llm.canonical_llm_contract import StrictParseResult, parse_strict_index_response


@dataclass(frozen=True)
class ParsedUserOutput:
    user_id: str
    dataset: str
    model_name: str
    prompt_strategy: str
    raw_output: str
    candidate_items: tuple[str, ...]
    parse_result: StrictParseResult


def parse_extension_top10_response(
    response_text: str,
    *,
    candidate_items: Sequence[str],
) -> StrictParseResult:
    """Parse the primary extension JSON/index output without repair."""

    text = str(response_text or "").strip()
    if not text:
        return StrictParseResult(False, tuple(), tuple(), 0, 0, 0, "empty_output", tuple(), "empty output")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return StrictParseResult(False, tuple(), tuple(), 0, 0, 0, "parse_failed", tuple(), str(exc))
    if not isinstance(parsed, dict):
        return StrictParseResult(False, tuple(), tuple(), 0, 0, 0, "invalid_schema", tuple(), "root JSON must be an object")
    if set(parsed) != {"ranked_candidate_indices"}:
        return StrictParseResult(
            False,
            tuple(),
            tuple(),
            0,
            0,
            0,
            "invalid_schema",
            tuple(),
            "expected only ranked_candidate_indices",
        )
    return parse_strict_index_response(
        stable_json(parsed),
        candidate_items=[str(item) for item in candidate_items],
        expected_count=REQUESTED_OUTPUT_LENGTH,
        require_full_permutation=False,
    )


def parsed_outputs_to_recommendations(parsed_outputs: Sequence[ParsedUserOutput], *, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for parsed in parsed_outputs:
        if not parsed.parse_result.parse_valid:
            continue
        for rank, item_id in enumerate(parsed.parse_result.output_items, start=1):
            rows.append(
                {
                    "user_id": parsed.user_id,
                    "model_name": parsed.model_name,
                    "dataset": parsed.dataset,
                    "rank": rank,
                    "recommended_item_id": item_id,
                    "run_id": run_id,
                    "prompt_variant": parsed.prompt_strategy,
                    "parser_contract": PARSER_CONTRACT_ID,
                    "parse_status": parsed.parse_result.status,
                    "raw_output": parsed.raw_output,
                }
            )
    return pd.DataFrame(rows)


def evaluate_fail_strict_outputs(
    parsed_outputs: Sequence[ParsedUserOutput],
    ground_truth: pd.DataFrame,
    *,
    run_id: str,
    k: int = METRIC_K,
) -> dict[str, float | int]:
    """Evaluate valid parses while invalid users remain in the denominator."""

    recommendations = parsed_outputs_to_recommendations(parsed_outputs, run_id=run_id)
    metrics = evaluate_utility_at_k(
        recommendations,
        ground_truth,
        k,
        recommendation_col="recommended_item_id",
    )
    metrics["strict_valid_count"] = sum(1 for output in parsed_outputs if output.parse_result.parse_valid)
    metrics["strict_invalid_count"] = sum(1 for output in parsed_outputs if not output.parse_result.parse_valid)
    return metrics

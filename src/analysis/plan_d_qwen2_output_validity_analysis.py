from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PLAN = "Plan D"
PHASE = "D5E_qwen2_ml1m_ranking_diagnostic"
ANALYSIS_PHASE = "D5E.1_qwen2_output_validity_analysis"
DATASET = "ml1m"
SCOPE = "diagnostic_subset"
BACKEND = "ollama"
MODEL = "qwen2:7b-instruct"
PROMPT_STRATEGY = "openp5_base"
TOP_K = 5

BREAKDOWN_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "backend",
    "model",
    "prompt_strategy",
    "num_instances",
    "num_responses_received",
    "num_parse_success",
    "num_schema_valid",
    "num_exact_top_k",
    "num_empty_outputs",
    "total_selected_items",
    "total_valid_selected_items",
    "total_hallucinated_items",
    "total_duplicate_items",
    "response_received_rate",
    "parse_success_rate",
    "schema_valid_rate",
    "exact_top_k_rate",
    "empty_output_rate",
    "mean_candidate_validity_rate",
    "micro_candidate_validity_rate",
    "hallucination_rate",
    "duplicate_rate",
    "status",
    "notes",
]

VALID_ONLY_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "backend",
    "model",
    "prompt_strategy",
    "utility_variant",
    "num_instances",
    "hit_rate_at_5",
    "ndcg_at_5",
    "mean_positive_rank",
    "num_hits",
    "notes",
]

PROFILE_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "instance_id",
    "user_id",
    "selected_count",
    "candidate_validity_rate",
    "hallucinated_count",
    "duplicate_count",
    "parse_success",
    "schema_valid",
    "positive_item_id",
    "positive_rank",
    "hit_at_5",
    "ndcg_at_5",
    "has_hallucination",
    "hallucination_severity",
    "latency_seconds",
    "status",
    "notes",
]

INVALID_PATTERN_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "model",
    "invalid_item_id",
    "frequency",
    "pattern_type",
    "example_instance_ids",
    "notes",
]


@dataclass(frozen=True)
class OutputValidityAnalysis:
    """Container for D5E.1 analysis tables."""

    breakdown: pd.DataFrame
    valid_only_utility: pd.DataFrame
    hallucination_profile: pd.DataFrame
    invalid_item_patterns: pd.DataFrame
    raw_available: bool


def build_output_validity_analysis(
    *,
    instance_summary_path: Path,
    aggregate_summary_path: Path,
    raw_output_path: Path,
) -> OutputValidityAnalysis:
    """Build D5E.1 analysis tables without calling any LLM backend."""

    instances = _read_csv(instance_summary_path)
    aggregate = _read_csv(aggregate_summary_path)
    _validate_d5e_inputs(instances, aggregate)
    raw_rows = _read_raw_trace_rows(raw_output_path)
    raw_available = bool(raw_rows)
    return OutputValidityAnalysis(
        breakdown=build_validity_breakdown(instances),
        valid_only_utility=build_valid_only_utility(instances, raw_rows),
        hallucination_profile=build_hallucination_profile(instances),
        invalid_item_patterns=build_invalid_item_patterns(raw_rows),
        raw_available=raw_available,
    )


def build_validity_breakdown(instances: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate output-validity counts and rates from D5E instance rows."""

    num_instances = len(instances)
    selected = _numeric(instances["selected_count"])
    hallucinated = _numeric(instances["hallucinated_count"])
    duplicates = _numeric(instances["duplicate_count"])
    total_selected = int(selected.sum())
    total_hallucinated = int(hallucinated.sum())
    total_duplicates = int(duplicates.sum())
    total_valid = int(total_selected - total_hallucinated)
    row = {
        "plan": PLAN,
        "phase": ANALYSIS_PHASE,
        "dataset": DATASET,
        "scope": SCOPE,
        "backend": BACKEND,
        "model": MODEL,
        "prompt_strategy": PROMPT_STRATEGY,
        "num_instances": str(num_instances),
        "num_responses_received": str(_bool_count(instances["response_received"])),
        "num_parse_success": str(_bool_count(instances["parse_success"])),
        "num_schema_valid": str(_bool_count(instances["schema_valid"])),
        "num_exact_top_k": str(int((selected == TOP_K).sum())),
        "num_empty_outputs": str(int((selected == 0).sum())),
        "total_selected_items": str(total_selected),
        "total_valid_selected_items": str(total_valid),
        "total_hallucinated_items": str(total_hallucinated),
        "total_duplicate_items": str(total_duplicates),
        "response_received_rate": _format_rate(_bool_count(instances["response_received"]), num_instances),
        "parse_success_rate": _format_rate(_bool_count(instances["parse_success"]), num_instances),
        "schema_valid_rate": _format_rate(_bool_count(instances["schema_valid"]), num_instances),
        "exact_top_k_rate": _format_rate(int((selected == TOP_K).sum()), num_instances),
        "empty_output_rate": _format_rate(int((selected == 0).sum()), num_instances),
        "mean_candidate_validity_rate": _format_float(_numeric(instances["candidate_validity_rate"]).mean()),
        "micro_candidate_validity_rate": _format_rate(total_valid, total_selected),
        "hallucination_rate": _format_rate(total_hallucinated, total_selected),
        "duplicate_rate": _format_rate(total_duplicates, total_selected),
        "status": "completed_no_inference",
        "notes": "Computed from D5E committed instance summary; no new LLM inference was run.",
    }
    return pd.DataFrame([row], columns=BREAKDOWN_COLUMNS)


def build_hallucination_profile(instances: pd.DataFrame) -> pd.DataFrame:
    """Create one compact hallucination profile row per D5E instance."""

    rows: list[dict[str, str]] = []
    for _, row in instances.iterrows():
        hallucinated_count = int(float(row["hallucinated_count"]))
        rows.append(
            {
                "plan": PLAN,
                "phase": ANALYSIS_PHASE,
                "dataset": DATASET,
                "instance_id": str(row["instance_id"]),
                "user_id": str(row["user_id"]),
                "selected_count": str(int(float(row["selected_count"]))),
                "candidate_validity_rate": _format_float(float(row["candidate_validity_rate"])),
                "hallucinated_count": str(hallucinated_count),
                "duplicate_count": str(int(float(row["duplicate_count"]))),
                "parse_success": _bool_text(_as_bool(row["parse_success"])),
                "schema_valid": _bool_text(_as_bool(row["schema_valid"])),
                "positive_item_id": str(row["positive_item_id"]),
                "positive_rank": str(row.get("positive_rank", "")),
                "hit_at_5": _format_float(float(row["hit_at_5"])),
                "ndcg_at_5": _format_float(float(row["ndcg_at_5"])),
                "has_hallucination": _bool_text(hallucinated_count > 0),
                "hallucination_severity": _hallucination_severity(hallucinated_count),
                "latency_seconds": _format_float(float(row["latency_seconds"])),
                "status": "completed_no_inference",
                "notes": _profile_notes(hallucinated_count),
            }
        )
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def build_valid_only_utility(instances: pd.DataFrame, raw_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build original, valid-only, and invalid-as-failure utility summaries."""

    rows = [
        _utility_row(
            utility_variant="original_parsed_output",
            hit_values=_numeric(instances["hit_at_5"]).tolist(),
            ndcg_values=_numeric(instances["ndcg_at_5"]).tolist(),
            positive_ranks=_positive_ranks_from_series(instances["positive_rank"]),
            notes="Original D5E utility from parsed output, including invalid IDs in the returned rank positions.",
        )
    ]

    if raw_rows:
        by_instance = {str(row["instance_id"]): row for row in raw_rows}
        valid_hits: list[float] = []
        valid_ndcgs: list[float] = []
        valid_ranks: list[int] = []
        for _, row in instances.iterrows():
            trace = by_instance.get(str(row["instance_id"]), {})
            parsed = trace.get("parsed_result", {}) if isinstance(trace, dict) else {}
            valid_ids = [str(item_id) for item_id in parsed.get("valid_item_ids", [])]
            positive_item_id = str(row["positive_item_id"])
            rank = _rank_in_top_k(valid_ids, positive_item_id)
            if rank is None:
                valid_hits.append(0.0)
                valid_ndcgs.append(0.0)
            else:
                valid_hits.append(1.0)
                valid_ndcgs.append(1.0 / math.log2(rank + 1))
                valid_ranks.append(rank)
        rows.append(
            _utility_row(
                utility_variant="valid_only_output",
                hit_values=valid_hits,
                ndcg_values=valid_ndcgs,
                positive_ranks=valid_ranks,
                notes=(
                    "Recomputed from ignored raw JSONL parsed valid_item_ids only, preserving order after "
                    "removing hallucinated IDs; no raw response text is committed."
                ),
            )
        )
    else:
        rows.append(
            {
                "plan": PLAN,
                "phase": ANALYSIS_PHASE,
                "dataset": DATASET,
                "scope": SCOPE,
                "backend": BACKEND,
                "model": MODEL,
                "prompt_strategy": PROMPT_STRATEGY,
                "utility_variant": "valid_only_output",
                "num_instances": "0",
                "hit_rate_at_5": "",
                "ndcg_at_5": "",
                "mean_positive_rank": "",
                "num_hits": "",
                "notes": "Unavailable because ignored raw JSONL was not present; values were not fabricated.",
            }
        )

    strict_hits: list[float] = []
    strict_ndcgs: list[float] = []
    strict_ranks: list[int] = []
    for _, row in instances.iterrows():
        if int(float(row["hallucinated_count"])) > 0:
            strict_hits.append(0.0)
            strict_ndcgs.append(0.0)
            continue
        hit = float(row["hit_at_5"])
        ndcg = float(row["ndcg_at_5"])
        strict_hits.append(hit)
        strict_ndcgs.append(ndcg)
        if hit > 0 and str(row.get("positive_rank", "")).strip():
            strict_ranks.append(int(float(row["positive_rank"])))
    rows.append(
        _utility_row(
            utility_variant="invalid_as_failure",
            hit_values=strict_hits,
            ndcg_values=strict_ndcgs,
            positive_ranks=strict_ranks,
            notes="Strict operational variant: any instance with hallucinated IDs is counted as utility failure.",
        )
    )
    return pd.DataFrame(rows, columns=VALID_ONLY_COLUMNS)


def build_invalid_item_patterns(raw_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Count compact invalid item patterns from ignored raw JSONL parsed results."""

    if not raw_rows:
        return pd.DataFrame(
            [
                {
                    "plan": PLAN,
                    "phase": ANALYSIS_PHASE,
                    "dataset": DATASET,
                    "model": MODEL,
                    "invalid_item_id": "raw_output_unavailable",
                    "frequency": "0",
                    "pattern_type": "unavailable",
                    "example_instance_ids": "",
                    "notes": "Detailed invalid item pattern analysis requires the ignored D5E raw JSONL file.",
                }
            ],
            columns=INVALID_PATTERN_COLUMNS,
        )

    counter: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for row in raw_rows:
        parsed = row.get("parsed_result", {}) if isinstance(row, dict) else {}
        instance_id = str(row.get("instance_id", ""))
        for invalid_item_id in parsed.get("invalid_item_ids", []):
            item_id = "" if invalid_item_id is None else str(invalid_item_id)
            counter[item_id] += 1
            if len(examples[item_id]) < 5:
                examples[item_id].append(instance_id)

    if not counter:
        rows = [
            {
                "plan": PLAN,
                "phase": ANALYSIS_PHASE,
                "dataset": DATASET,
                "model": MODEL,
                "invalid_item_id": "none_observed",
                "frequency": "0",
                "pattern_type": "none",
                "example_instance_ids": "",
                "notes": "No invalid item IDs were observed in the ignored D5E raw JSONL parsed results.",
            }
        ]
    else:
        rows = []
        for invalid_item_id, frequency in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "plan": PLAN,
                    "phase": ANALYSIS_PHASE,
                    "dataset": DATASET,
                    "model": MODEL,
                    "invalid_item_id": invalid_item_id,
                    "frequency": str(frequency),
                    "pattern_type": _invalid_item_pattern(invalid_item_id),
                    "example_instance_ids": "|".join(examples[invalid_item_id]),
                    "notes": "Extracted from parsed invalid_item_ids only; raw response text is not committed.",
                }
            )
    return pd.DataFrame(rows, columns=INVALID_PATTERN_COLUMNS)


def _utility_row(
    *,
    utility_variant: str,
    hit_values: list[float],
    ndcg_values: list[float],
    positive_ranks: list[int],
    notes: str,
) -> dict[str, str]:
    num_instances = len(hit_values)
    num_hits = int(sum(1 for value in hit_values if value > 0))
    return {
        "plan": PLAN,
        "phase": ANALYSIS_PHASE,
        "dataset": DATASET,
        "scope": SCOPE,
        "backend": BACKEND,
        "model": MODEL,
        "prompt_strategy": PROMPT_STRATEGY,
        "utility_variant": utility_variant,
        "num_instances": str(num_instances),
        "hit_rate_at_5": _format_float(sum(hit_values) / num_instances if num_instances else 0.0),
        "ndcg_at_5": _format_float(sum(ndcg_values) / num_instances if num_instances else 0.0),
        "mean_positive_rank": _format_float(sum(positive_ranks) / len(positive_ranks)) if positive_ranks else "",
        "num_hits": str(num_hits),
        "notes": notes,
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required D5E analysis input: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def _read_raw_trace_rows(raw_output_path: Path) -> list[dict[str, Any]]:
    if not raw_output_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with raw_output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _validate_d5e_inputs(instances: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    errors: list[str] = []
    if len(instances) != 20:
        errors.append(f"expected 20 instance rows, found {len(instances)}")
    if len(aggregate) != 1:
        errors.append(f"expected one aggregate row, found {len(aggregate)}")
    for column, expected in (
        ("dataset", DATASET),
        ("scope", SCOPE),
        ("backend", BACKEND),
        ("model", MODEL),
        ("prompt_strategy", PROMPT_STRATEGY),
    ):
        observed = set(instances[column].astype(str))
        if observed != {expected}:
            errors.append(f"instance {column} expected {expected}, observed {sorted(observed)}")
    if errors:
        raise ValueError("Invalid D5E analysis inputs: " + "; ".join(errors))


def _bool_count(series: pd.Series) -> int:
    return sum(1 for value in series if _as_bool(value))


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _positive_ranks_from_series(series: pd.Series) -> list[int]:
    ranks: list[int] = []
    for value in series:
        text = str(value).strip()
        if text:
            ranks.append(int(float(text)))
    return ranks


def _rank_in_top_k(item_ids: list[str], positive_item_id: str) -> int | None:
    for index, item_id in enumerate(item_ids[:TOP_K], start=1):
        if item_id == positive_item_id:
            return index
    return None


def _hallucination_severity(hallucinated_count: int) -> str:
    if hallucinated_count == 0:
        return "none"
    if hallucinated_count == 1:
        return "low"
    if hallucinated_count == 2:
        return "medium"
    return "high"


def _profile_notes(hallucinated_count: int) -> str:
    if hallucinated_count == 0:
        return "No hallucinated candidate IDs in this parsed output."
    return "Candidate adherence failure: parsed output includes item IDs outside the D5E candidate set."


def _invalid_item_pattern(invalid_item_id: str) -> str:
    text = str(invalid_item_id).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return "empty_or_null"
    if re.fullmatch(r"\d+", text):
        return "non_candidate_existing_format"
    if re.search(r"\s", text) or not re.fullmatch(r"[A-Za-z0-9_:\-]+", text):
        return "malformed_id"
    return "unknown"


def _format_rate(numerator: int, denominator: int) -> str:
    return _format_float((numerator / denominator) if denominator else 0.0)


def _format_float(value: float) -> str:
    return f"{float(value):.6f}"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"

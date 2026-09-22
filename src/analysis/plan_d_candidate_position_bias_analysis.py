from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Any

import pandas as pd


PLAN = "Plan D"
PHASE = "D5F_qwen2_ml1m_candidate_position_bias_diagnostic"
DATASET = "ml1m"
SCOPE = "diagnostic_subset"
BACKEND = "ollama"
MODEL = "qwen2:7b-instruct"
PROMPT_STRATEGY = "openp5_base"
METHOD = "qwen2_ollama_openp5_position_bias"
TEMPERATURE = 0.0
TOP_K = 5
CANDIDATE_POOL_SIZE = 20
VARIANTS = ("reverse_order", "positive_first", "positive_last")

MANIFEST_COLUMNS = [
    "diagnostic_id",
    "original_instance_id",
    "user_id",
    "dataset",
    "variant",
    "original_positive_position",
    "variant_positive_position",
    "candidate_pool_size",
    "positive_item_id",
    "candidate_order_signature",
    "candidate_ids",
    "status",
]

INSTANCE_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "diagnostic_id",
    "original_instance_id",
    "user_id",
    "variant",
    "backend",
    "model",
    "prompt_strategy",
    "temperature",
    "top_k",
    "candidate_pool_size",
    "positive_item_id",
    "original_positive_position",
    "variant_positive_position",
    "response_received",
    "parse_success",
    "schema_valid",
    "selected_count",
    "candidate_validity_rate",
    "duplicate_count",
    "hallucinated_count",
    "positive_rank",
    "hit_at_5",
    "ndcg_at_5",
    "selected_input_positions",
    "valid_selected_input_positions",
    "valid_selected_item_ids",
    "mean_selected_input_position",
    "front_half_selection_rate",
    "back_half_selection_rate",
    "position_1_selected",
    "position_20_selected",
    "latency_seconds",
    "parse_status",
    "status",
    "notes",
]

VARIANT_SUMMARY_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "backend",
    "model",
    "prompt_strategy",
    "variant",
    "total_instances",
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
    "hr_at_5",
    "ndcg_at_5",
    "mean_selected_input_position",
    "median_selected_input_position",
    "front_half_selection_rate",
    "back_half_selection_rate",
    "position_1_selection_rate",
    "position_20_selection_rate",
    "positive_input_position",
    "positive_selected_rate",
    "positive_rank_when_selected",
    "valid_recommendation_jaccard_vs_original",
    "selected_valid_items_overlap_vs_original",
    "hr_at_5_delta_vs_original",
    "ndcg_at_5_delta_vs_original",
    "hallucination_rate_delta_vs_original",
    "candidate_validity_delta_vs_original",
    "status",
    "notes",
]

PAIRWISE_COLUMNS = [
    "variant",
    "metric_name",
    "original_value",
    "variant_value",
    "delta",
    "interpretation",
]

DISTRIBUTION_COLUMNS = [
    "variant",
    "input_position",
    "selected_count",
    "valid_selected_count",
    "selected_rate",
    "valid_selected_rate",
]

METRIC_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "method",
    "backend",
    "model",
    "prompt_strategy",
    "variant",
    "metric",
    "k",
    "value",
    "num_instances",
    "notes",
]


@dataclass(frozen=True)
class ParsedInstance:
    diagnostic_id: str
    original_instance_id: str
    user_id: str
    variant: str
    candidate_ids: list[str]
    positive_item_id: str
    original_positive_position: int
    variant_positive_position: int
    response_received: bool
    parse_success: bool
    schema_valid: bool
    ranked_item_ids: list[str]
    valid_item_ids: list[str]
    invalid_item_ids: list[str]
    selected_count: int
    candidate_validity_rate: float
    duplicate_count: int
    hallucinated_count: int
    latency_seconds: float
    parse_status: str
    error_message: str


def candidate_order_for_variant(candidate_ids: list[str], positive_item_id: str, variant: str) -> list[str]:
    """Return a deterministic D5F candidate ordering variant."""

    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ids must be unique")
    if positive_item_id not in set(candidate_ids):
        raise ValueError("positive_item_id must be in candidate_ids")
    if variant == "reverse_order":
        return list(reversed(candidate_ids))
    if variant == "positive_first":
        return [positive_item_id] + [item_id for item_id in candidate_ids if item_id != positive_item_id]
    if variant == "positive_last":
        return [item_id for item_id in candidate_ids if item_id != positive_item_id] + [positive_item_id]
    raise ValueError(f"Unsupported position-bias variant: {variant}")


def candidate_order_signature(candidate_ids: list[str]) -> str:
    """Return a short deterministic signature for an ordered candidate list."""

    joined = "|".join(candidate_ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def build_position_bias_manifest(d5e_manifest: pd.DataFrame) -> pd.DataFrame:
    """Build the 60-row D5F variant manifest from the 20-row D5D/D5E manifest."""

    rows: list[dict[str, str]] = []
    for _, row in d5e_manifest.iterrows():
        candidate_ids = json.loads(str(row["candidate_ids"]))
        positive_item_id = str(row["positive_item_id"])
        original_position = candidate_ids.index(positive_item_id) + 1
        for variant in VARIANTS:
            variant_ids = candidate_order_for_variant(candidate_ids, positive_item_id, variant)
            variant_position = variant_ids.index(positive_item_id) + 1
            diagnostic_id = f"{PHASE}_{variant}_{str(row['user_id'])}"
            rows.append(
                {
                    "diagnostic_id": diagnostic_id,
                    "original_instance_id": str(row["instance_id"]),
                    "user_id": str(row["user_id"]),
                    "dataset": DATASET,
                    "variant": variant,
                    "original_positive_position": str(original_position),
                    "variant_positive_position": str(variant_position),
                    "candidate_pool_size": str(len(variant_ids)),
                    "positive_item_id": positive_item_id,
                    "candidate_order_signature": candidate_order_signature(variant_ids),
                    "candidate_ids": json.dumps(variant_ids, separators=(",", ":")),
                    "status": "ready_for_local_qwen_inference",
                }
            )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def evaluate_parsed_instance(instance: ParsedInstance) -> dict[str, str]:
    """Convert one parsed D5F response to a compact per-instance summary row."""

    position_map = {item_id: index for index, item_id in enumerate(instance.candidate_ids, start=1)}
    valid_positions = [position_map[item_id] for item_id in instance.ranked_item_ids if item_id in position_map]
    positive_rank = _positive_rank(instance.ranked_item_ids, instance.positive_item_id)
    hit_at_5 = 1.0 if positive_rank is not None else 0.0
    ndcg_at_5 = 1.0 / math.log2(positive_rank + 1) if positive_rank is not None else 0.0
    position_metrics = selected_position_metrics(valid_positions)
    status = _instance_status(instance)
    notes = instance.error_message or _instance_notes(instance)
    return {
        "plan": PLAN,
        "phase": PHASE,
        "dataset": DATASET,
        "scope": SCOPE,
        "diagnostic_id": instance.diagnostic_id,
        "original_instance_id": instance.original_instance_id,
        "user_id": instance.user_id,
        "variant": instance.variant,
        "backend": BACKEND,
        "model": MODEL,
        "prompt_strategy": PROMPT_STRATEGY,
        "temperature": _format_float(TEMPERATURE),
        "top_k": str(TOP_K),
        "candidate_pool_size": str(CANDIDATE_POOL_SIZE),
        "positive_item_id": instance.positive_item_id,
        "original_positive_position": str(instance.original_positive_position),
        "variant_positive_position": str(instance.variant_positive_position),
        "response_received": _bool_text(instance.response_received),
        "parse_success": _bool_text(instance.parse_success),
        "schema_valid": _bool_text(instance.schema_valid),
        "selected_count": str(instance.selected_count),
        "candidate_validity_rate": _format_float(instance.candidate_validity_rate),
        "duplicate_count": str(instance.duplicate_count),
        "hallucinated_count": str(instance.hallucinated_count),
        "positive_rank": "" if positive_rank is None else str(positive_rank),
        "hit_at_5": _format_float(hit_at_5),
        "ndcg_at_5": _format_float(ndcg_at_5),
        "selected_input_positions": json.dumps(valid_positions, separators=(",", ":")),
        "valid_selected_input_positions": json.dumps(valid_positions, separators=(",", ":")),
        "valid_selected_item_ids": json.dumps(instance.valid_item_ids, separators=(",", ":")),
        "mean_selected_input_position": _format_optional(position_metrics["mean_selected_input_position"]),
        "front_half_selection_rate": _format_float(position_metrics["front_half_selection_rate"]),
        "back_half_selection_rate": _format_float(position_metrics["back_half_selection_rate"]),
        "position_1_selected": _bool_text(1 in valid_positions),
        "position_20_selected": _bool_text(20 in valid_positions),
        "latency_seconds": _format_float(instance.latency_seconds),
        "parse_status": instance.parse_status,
        "status": status,
        "notes": notes,
    }


def selected_position_metrics(valid_positions: list[int]) -> dict[str, float | None]:
    """Compute simple position-bias metrics from valid selected input positions."""

    if not valid_positions:
        return {
            "mean_selected_input_position": None,
            "median_selected_input_position": None,
            "front_half_selection_rate": 0.0,
            "back_half_selection_rate": 0.0,
            "position_1_selection_rate": 0.0,
            "position_20_selection_rate": 0.0,
        }
    total = len(valid_positions)
    return {
        "mean_selected_input_position": sum(valid_positions) / total,
        "median_selected_input_position": float(median(valid_positions)),
        "front_half_selection_rate": sum(1 for position in valid_positions if position <= 10) / total,
        "back_half_selection_rate": sum(1 for position in valid_positions if position >= 11) / total,
        "position_1_selection_rate": sum(1 for position in valid_positions if position == 1) / total,
        "position_20_selection_rate": sum(1 for position in valid_positions if position == 20) / total,
    }


def build_selected_position_distribution(instance_summary: pd.DataFrame) -> pd.DataFrame:
    """Build selected-position counts for each variant and input position 1..20."""

    rows: list[dict[str, str]] = []
    for variant in VARIANTS:
        subset = instance_summary.loc[instance_summary["variant"] == variant].copy()
        total_slots = max(1, len(subset) * TOP_K)
        position_counts: Counter[int] = Counter()
        for encoded in subset["valid_selected_input_positions"]:
            for position in json.loads(str(encoded)):
                position_counts[int(position)] += 1
        for position in range(1, CANDIDATE_POOL_SIZE + 1):
            count = position_counts[position]
            rows.append(
                {
                    "variant": variant,
                    "input_position": str(position),
                    "selected_count": str(count),
                    "valid_selected_count": str(count),
                    "selected_rate": _format_float(count / total_slots),
                    "valid_selected_rate": _format_float(count / total_slots),
                }
            )
    return pd.DataFrame(rows, columns=DISTRIBUTION_COLUMNS)


def build_variant_summary(
    instance_summary: pd.DataFrame,
    *,
    original_reference: dict[str, Any],
    original_valid_items_by_instance: dict[str, set[str]],
) -> pd.DataFrame:
    """Aggregate D5F rows by variant and compare each variant against D5E original order."""

    rows: list[dict[str, str]] = []
    for variant in VARIANTS:
        subset = instance_summary.loc[instance_summary["variant"] == variant].copy()
        selected = _numeric(subset["selected_count"])
        hallucinated = _numeric(subset["hallucinated_count"])
        duplicates = _numeric(subset["duplicate_count"])
        total_selected = int(selected.sum())
        total_hallucinated = int(hallucinated.sum())
        total_duplicate = int(duplicates.sum())
        total_valid = total_selected - total_hallucinated
        all_positions = _all_positions(subset)
        position_metrics = selected_position_metrics(all_positions)
        hr_at_5 = float(_numeric(subset["hit_at_5"]).mean())
        ndcg_at_5 = float(_numeric(subset["ndcg_at_5"]).mean())
        hallucination_rate = total_hallucinated / total_selected if total_selected else 0.0
        mean_candidate_validity = float(_numeric(subset["candidate_validity_rate"]).mean())
        positive_ranks = [int(float(value)) for value in subset["positive_rank"] if str(value).strip()]
        jaccards, overlaps = _overlap_vs_original(subset, original_valid_items_by_instance)
        row = {
            "plan": PLAN,
            "phase": PHASE,
            "dataset": DATASET,
            "scope": SCOPE,
            "backend": BACKEND,
            "model": MODEL,
            "prompt_strategy": PROMPT_STRATEGY,
            "variant": variant,
            "total_instances": str(len(subset)),
            "num_responses_received": str(_bool_count(subset["response_received"])),
            "num_parse_success": str(_bool_count(subset["parse_success"])),
            "num_schema_valid": str(_bool_count(subset["schema_valid"])),
            "num_exact_top_k": str(int((selected == TOP_K).sum())),
            "num_empty_outputs": str(int((selected == 0).sum())),
            "total_selected_items": str(total_selected),
            "total_valid_selected_items": str(total_valid),
            "total_hallucinated_items": str(total_hallucinated),
            "total_duplicate_items": str(total_duplicate),
            "response_received_rate": _format_rate(_bool_count(subset["response_received"]), len(subset)),
            "parse_success_rate": _format_rate(_bool_count(subset["parse_success"]), len(subset)),
            "schema_valid_rate": _format_rate(_bool_count(subset["schema_valid"]), len(subset)),
            "exact_top_k_rate": _format_rate(int((selected == TOP_K).sum()), len(subset)),
            "empty_output_rate": _format_rate(int((selected == 0).sum()), len(subset)),
            "mean_candidate_validity_rate": _format_float(mean_candidate_validity),
            "micro_candidate_validity_rate": _format_rate(total_valid, total_selected),
            "hallucination_rate": _format_float(hallucination_rate),
            "duplicate_rate": _format_rate(total_duplicate, total_selected),
            "hr_at_5": _format_float(hr_at_5),
            "ndcg_at_5": _format_float(ndcg_at_5),
            "mean_selected_input_position": _format_optional(position_metrics["mean_selected_input_position"]),
            "median_selected_input_position": _format_optional(position_metrics["median_selected_input_position"]),
            "front_half_selection_rate": _format_float(position_metrics["front_half_selection_rate"]),
            "back_half_selection_rate": _format_float(position_metrics["back_half_selection_rate"]),
            "position_1_selection_rate": _format_float(position_metrics["position_1_selection_rate"]),
            "position_20_selection_rate": _format_float(position_metrics["position_20_selection_rate"]),
            "positive_input_position": _positive_input_position_summary(subset),
            "positive_selected_rate": _format_rate(sum(1 for rank in positive_ranks if rank <= TOP_K), len(subset)),
            "positive_rank_when_selected": _format_float(sum(positive_ranks) / len(positive_ranks)) if positive_ranks else "",
            "valid_recommendation_jaccard_vs_original": _format_float(sum(jaccards) / len(jaccards)) if jaccards else "",
            "selected_valid_items_overlap_vs_original": _format_float(sum(overlaps) / len(overlaps)) if overlaps else "",
            "hr_at_5_delta_vs_original": _format_float(hr_at_5 - float(original_reference["hit_rate_at_5"])),
            "ndcg_at_5_delta_vs_original": _format_float(ndcg_at_5 - float(original_reference["ndcg_at_5"])),
            "hallucination_rate_delta_vs_original": _format_float(hallucination_rate - float(original_reference["hallucination_rate"])),
            "candidate_validity_delta_vs_original": _format_float(mean_candidate_validity - float(original_reference["mean_candidate_validity_rate"])),
            "status": "completed_diagnostic_subset",
            "notes": "Compared with existing D5E original-order reference; original order was not rerun.",
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=VARIANT_SUMMARY_COLUMNS)


def build_pairwise_delta(variant_summary: pd.DataFrame, original_reference: dict[str, Any]) -> pd.DataFrame:
    """Build a long pairwise delta table comparing each D5F variant with D5E original order."""

    metric_map = {
        "HR@5": ("hit_rate_at_5", "hr_at_5"),
        "NDCG@5": ("ndcg_at_5", "ndcg_at_5"),
        "hallucination_rate": ("hallucination_rate", "hallucination_rate"),
        "mean_candidate_validity_rate": ("mean_candidate_validity_rate", "mean_candidate_validity_rate"),
        "micro_candidate_validity_rate": ("micro_candidate_validity_rate", "micro_candidate_validity_rate"),
        "exact_top_k_rate": ("exact_top_k_rate", "exact_top_k_rate"),
    }
    rows: list[dict[str, str]] = []
    for _, row in variant_summary.iterrows():
        for metric_name, (original_key, variant_key) in metric_map.items():
            original_value = float(original_reference[original_key])
            variant_value = float(row[variant_key])
            delta = variant_value - original_value
            rows.append(
                {
                    "variant": str(row["variant"]),
                    "metric_name": metric_name,
                    "original_value": _format_float(original_value),
                    "variant_value": _format_float(variant_value),
                    "delta": _format_float(delta),
                    "interpretation": _delta_interpretation(metric_name, delta),
                }
            )
    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)


def build_metric_rows(variant_summary: pd.DataFrame) -> pd.DataFrame:
    """Build metric-style rows for D5F variant summaries."""

    metric_columns = [
        ("response_received_rate", ""),
        ("parse_success_rate", ""),
        ("schema_valid_rate", ""),
        ("exact_top_k_rate", ""),
        ("empty_output_rate", ""),
        ("mean_candidate_validity_rate", ""),
        ("micro_candidate_validity_rate", ""),
        ("hallucination_rate", ""),
        ("duplicate_rate", ""),
        ("hr_at_5", "5"),
        ("ndcg_at_5", "5"),
        ("mean_selected_input_position", ""),
        ("median_selected_input_position", ""),
        ("front_half_selection_rate", ""),
        ("back_half_selection_rate", ""),
        ("position_1_selection_rate", ""),
        ("position_20_selection_rate", ""),
        ("positive_selected_rate", ""),
        ("valid_recommendation_jaccard_vs_original", ""),
        ("selected_valid_items_overlap_vs_original", ""),
    ]
    rows: list[dict[str, str]] = []
    for _, row in variant_summary.iterrows():
        for metric, k in metric_columns:
            value = str(row[metric])
            if value == "":
                continue
            rows.append(
                {
                    "plan": PLAN,
                    "phase": PHASE,
                    "dataset": DATASET,
                    "scope": SCOPE,
                    "method": METHOD,
                    "backend": BACKEND,
                    "model": MODEL,
                    "prompt_strategy": PROMPT_STRATEGY,
                    "variant": str(row["variant"]),
                    "metric": metric,
                    "k": k,
                    "value": value,
                    "num_instances": str(row["total_instances"]),
                    "notes": "D5F candidate position-bias diagnostic; original order was not rerun.",
                }
            )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def original_reference_from_d5e_summary(d5e_summary: pd.DataFrame) -> dict[str, str]:
    row = d5e_summary.iloc[0]
    return {
        "hit_rate_at_5": str(row["hit_rate_at_5"]),
        "ndcg_at_5": str(row["ndcg_at_5"]),
        "hallucination_rate": str(row["hallucination_rate"]),
        "mean_candidate_validity_rate": str(row["mean_candidate_validity_rate"]),
        "micro_candidate_validity_rate": str(row.get("mean_candidate_validity_rate", "")),
        "exact_top_k_rate": str(row["exact_top_k_rate"]),
    }


def original_valid_items_from_raw(raw_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for row in raw_rows:
        parsed = row.get("parsed_result", {})
        result[str(row.get("instance_id", ""))] = {str(item_id) for item_id in parsed.get("valid_item_ids", [])}
    return result


def raw_position_bias_trace_row(
    *,
    parsed_instance: ParsedInstance,
    raw_response_text: str,
) -> dict[str, Any]:
    return {
        "diagnostic_id": parsed_instance.diagnostic_id,
        "original_instance_id": parsed_instance.original_instance_id,
        "user_id": parsed_instance.user_id,
        "variant": parsed_instance.variant,
        "model": MODEL,
        "backend": BACKEND,
        "prompt_strategy": PROMPT_STRATEGY,
        "raw_response_text": raw_response_text,
        "parsed_result": {
            "parse_success": parsed_instance.parse_success,
            "schema_valid": parsed_instance.schema_valid,
            "ranked_item_ids": parsed_instance.ranked_item_ids,
            "valid_item_ids": parsed_instance.valid_item_ids,
            "invalid_item_ids": parsed_instance.invalid_item_ids,
            "selected_count": parsed_instance.selected_count,
            "candidate_validity_rate": parsed_instance.candidate_validity_rate,
            "duplicate_count": parsed_instance.duplicate_count,
            "hallucinated_count": parsed_instance.hallucinated_count,
            "status": parsed_instance.parse_status,
            "error_message": parsed_instance.error_message,
        },
        "latency_seconds": parsed_instance.latency_seconds,
        "error_message": parsed_instance.error_message,
    }


def _positive_rank(ranked_item_ids: list[str], positive_item_id: str) -> int | None:
    for index, item_id in enumerate(ranked_item_ids[:TOP_K], start=1):
        if item_id == positive_item_id:
            return index
    return None


def _instance_status(instance: ParsedInstance) -> str:
    if not instance.response_received:
        return "failed_backend"
    if instance.parse_status == "parsed_valid":
        return "completed"
    if instance.parse_status == "parsed_with_hallucinations":
        return "completed_with_hallucinations"
    if instance.parse_status == "parsed_with_duplicates":
        return "completed_with_duplicates"
    if instance.parse_status == "parsed_invalid_schema":
        return "completed_invalid_schema"
    return "failed_parse"


def _instance_notes(instance: ParsedInstance) -> str:
    if instance.hallucinated_count > 0:
        return "Parsed response contains IDs outside the reordered candidate set."
    if instance.duplicate_count > 0:
        return "Parsed response contains duplicate IDs."
    return "Parsed response evaluated without manual correction."


def _all_positions(subset: pd.DataFrame) -> list[int]:
    positions: list[int] = []
    for encoded in subset["valid_selected_input_positions"]:
        positions.extend(int(position) for position in json.loads(str(encoded)))
    return positions


def _overlap_vs_original(
    subset: pd.DataFrame,
    original_valid_items_by_instance: dict[str, set[str]],
) -> tuple[list[float], list[float]]:
    jaccards: list[float] = []
    overlaps: list[float] = []
    for _, row in subset.iterrows():
        original_set = original_valid_items_by_instance.get(str(row["original_instance_id"]))
        if original_set is None:
            continue
        variant_valid = {str(item_id) for item_id in json.loads(str(row["valid_selected_item_ids"]))}
        union = original_set | variant_valid
        intersection = original_set & variant_valid
        jaccards.append(len(intersection) / len(union) if union else 0.0)
        overlaps.append(float(len(intersection)))
    return jaccards, overlaps


def _positive_input_position_summary(subset: pd.DataFrame) -> str:
    values = sorted(set(str(value) for value in subset["variant_positive_position"]))
    return values[0] if len(values) == 1 else "|".join(values)


def _delta_interpretation(metric_name: str, delta: float) -> str:
    if abs(delta) < 1e-12:
        return "no_change_vs_original"
    direction = "increased" if delta > 0 else "decreased"
    return f"{metric_name}_{direction}_vs_original"


def _bool_count(series: pd.Series) -> int:
    return sum(1 for value in series if str(value).strip().lower() == "true")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _format_rate(numerator: int, denominator: int) -> str:
    return _format_float((numerator / denominator) if denominator else 0.0)


def _format_optional(value: float | None) -> str:
    return "" if value is None else _format_float(value)


def _format_float(value: float) -> str:
    return f"{float(value):.6f}"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"

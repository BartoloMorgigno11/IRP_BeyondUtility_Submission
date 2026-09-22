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
PHASE = "D5J_llm_ml1m_reranking_diagnostic"
DATASET = "ml1m"
SCOPE = "diagnostic_subset"
BACKEND = "ollama"
QWEN_MODEL = "qwen2:7b-instruct"
MISTRAL_MODEL = "mistral:7b-instruct"
REFERENCE_MODEL = "candidate_order_reference"
PROMPT_STRATEGY = "strict_candidate_id_only_rerank"
METHOD = "local_llm_reranking_strict_candidate_id"
TEMPERATURE = 0.0
TOP_K = 5
CANDIDATE_POOL_SIZE = 20

REFERENCE_CONDITION = "candidate_order_reference"
QWEN_CONDITION = "qwen2_rerank_prompt"
MISTRAL_CONDITION = "mistral_rerank_prompt"
CONDITIONS = (REFERENCE_CONDITION, QWEN_CONDITION, MISTRAL_CONDITION)

MANIFEST_COLUMNS = [
    "diagnostic_id",
    "user_id",
    "dataset",
    "condition",
    "candidate_pool_size",
    "positive_item_id",
    "candidate_order_signature",
    "prompt_strategy",
    "model",
    "backend",
    "status",
]

INSTANCE_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "condition",
    "user_id",
    "source_instance_id",
    "diagnostic_id",
    "backend",
    "model",
    "prompt_strategy",
    "temperature",
    "top_k",
    "candidate_pool_size",
    "candidate_ids",
    "positive_item_id",
    "response_received",
    "parse_success",
    "schema_valid",
    "selected_count",
    "valid_selected_count",
    "candidate_validity_rate",
    "duplicate_count",
    "hallucinated_count",
    "positive_rank",
    "hit_at_5",
    "ndcg_at_5",
    "selected_item_ids",
    "valid_selected_item_ids",
    "selected_input_positions",
    "latency_seconds",
    "parse_status",
    "status",
    "notes",
]

CONDITION_SUMMARY_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "backend",
    "model",
    "condition",
    "prompt_strategy",
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
    "exact_5_valid_ids_rate",
    "hr_at_5",
    "ndcg_at_5",
    "mean_latency_seconds",
    "total_runtime_seconds",
    "mean_selected_input_position",
    "median_selected_input_position",
    "front_half_selection_rate",
    "back_half_selection_rate",
    "position_1_selection_rate",
    "position_20_selection_rate",
    "mean_absolute_position_shift_vs_reference",
    "mean_signed_position_shift_vs_reference",
    "selected_item_jaccard_vs_reference",
    "valid_item_jaccard_vs_reference",
    "positive_selected_rate",
    "positive_rank_when_selected",
    "positive_input_position",
    "positive_position_shift_when_selected",
    "status",
    "notes",
]

PAIRWISE_COLUMNS = [
    "comparison",
    "metric_name",
    "reference_value",
    "qwen2_value",
    "mistral_value",
    "condition_value",
    "delta",
    "pairwise_value",
    "interpretation",
]

AGREEMENT_COLUMNS = [
    "diagnostic_id",
    "user_id",
    "positive_item_id",
    "reference_top5_ids",
    "qwen_selected_ids",
    "mistral_selected_ids",
    "qwen_valid_selected_ids",
    "mistral_valid_selected_ids",
    "qwen_hit",
    "mistral_hit",
    "reference_hit",
    "agreement_category",
    "selected_item_jaccard_qwen_vs_mistral",
    "valid_item_jaccard_qwen_vs_mistral",
]

POSITION_SHIFT_COLUMNS = [
    "diagnostic_id",
    "user_id",
    "model",
    "selected_item_id",
    "output_rank",
    "original_input_position",
    "position_shift",
    "is_positive",
    "is_valid_candidate",
]

METRIC_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "method",
    "backend",
    "model",
    "condition",
    "prompt_strategy",
    "metric",
    "k",
    "value",
    "num_instances",
    "notes",
]


@dataclass(frozen=True)
class RerankInstance:
    condition: str
    user_id: str
    source_instance_id: str
    diagnostic_id: str
    candidate_ids: list[str]
    positive_item_id: str
    response_received: bool
    parse_success: bool
    schema_valid: bool
    ranked_item_ids: list[str]
    valid_item_ids: list[str]
    invalid_item_ids: list[str]
    selected_count: int
    valid_selected_count: int
    candidate_validity_rate: float
    duplicate_count: int
    hallucinated_count: int
    positive_rank: int | None
    hit_at_5: float
    ndcg_at_5: float
    selected_input_positions: list[int]
    latency_seconds: float
    parse_status: str
    status: str
    notes: str


def build_reranking_prompt(
    *,
    user_id: str,
    history_preview: str,
    candidate_ids: list[str],
    top_k: int = TOP_K,
) -> str:
    """Build a strict candidate-ID-only prompt for the D5J re-ranking task."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ids must be unique")
    if top_k > len(candidate_ids):
        raise ValueError("top_k cannot exceed candidate count")
    candidate_lines = "\n".join(
        f"- input_rank {index}: candidate_id {item_id}" for index, item_id in enumerate(candidate_ids, start=1)
    )
    return f"""You are re-ranking a pre-ranked baseline candidate list for user `{user_id}`.

User preference context, not valid output IDs:
{_history_titles_only(history_preview)}

The candidate list below is already ordered by a baseline retriever. Treat `input_rank 1` as the original first-ranked candidate.
Candidate IDs with original input ranks:
{candidate_lines}

Rules:
- This is a re-ranking task.
- Return only valid JSON.
- Return exactly {top_k} item IDs.
- Every returned ID must be copied exactly from the candidate list.
- Do not invent IDs.
- Do not output titles.
- Do not output explanations.
- If uncertain, choose only from the provided candidate IDs.

Required JSON object:
ranked_item_ids must be an array of exactly {top_k} strings copied from the candidate list; status must be "ok".
"""


def reference_top_k(candidate_ids: list[str], top_k: int = TOP_K) -> list[str]:
    """Return the baseline/retriever top-k from the original candidate order."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    return list(candidate_ids[:top_k])


def candidate_order_signature(candidate_ids: list[str]) -> str:
    return hashlib.sha256("|".join(candidate_ids).encode("utf-8")).hexdigest()[:16]


def build_reranking_manifest(d5e_manifest: pd.DataFrame) -> pd.DataFrame:
    """Build a 60-row D5J manifest for reference, Qwen2, and Mistral conditions."""

    rows: list[dict[str, str]] = []
    for _, row in d5e_manifest.iterrows():
        candidate_ids = parse_candidate_ids(str(row["candidate_ids"]))
        for condition in CONDITIONS:
            rows.append(
                {
                    "diagnostic_id": f"{PHASE}_{condition}_{str(row['user_id'])}",
                    "user_id": str(row["user_id"]),
                    "dataset": DATASET,
                    "condition": condition,
                    "candidate_pool_size": str(len(candidate_ids)),
                    "positive_item_id": str(row["positive_item_id"]),
                    "candidate_order_signature": candidate_order_signature(candidate_ids),
                    "prompt_strategy": PROMPT_STRATEGY if condition != REFERENCE_CONDITION else "candidate_order_reference",
                    "model": model_for_condition(condition),
                    "backend": BACKEND if condition != REFERENCE_CONDITION else "none",
                    "status": manifest_status(condition),
                }
            )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def parse_candidate_ids(encoded: str) -> list[str]:
    value = json.loads(encoded)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("candidate_ids must be a JSON list of strings")
    return value


def evaluate_reranked_ids(
    ranked_item_ids: list[str],
    *,
    condition: str,
    user_id: str,
    source_instance_id: str,
    diagnostic_id: str,
    candidate_ids: list[str],
    positive_item_id: str,
    response_received: bool,
    parse_success: bool,
    schema_valid: bool,
    latency_seconds: float,
    parse_status: str,
    notes: str,
) -> RerankInstance:
    selected = [str(item_id) for item_id in ranked_item_ids]
    candidate_set = set(candidate_ids)
    valid_item_ids = [item_id for item_id in selected if item_id in candidate_set]
    invalid_item_ids = [item_id for item_id in selected if item_id not in candidate_set]
    position_map = {item_id: index for index, item_id in enumerate(candidate_ids, start=1)}
    selected_positions = [position_map[item_id] for item_id in selected if item_id in position_map]
    selected_count = len(selected)
    duplicate_count = selected_count - len(set(selected))
    hallucinated_count = len(invalid_item_ids)
    candidate_validity_rate = len(valid_item_ids) / selected_count if selected_count else 0.0
    positive_rank = positive_rank_at_k(selected, positive_item_id, TOP_K)
    hit_at_5 = 1.0 if positive_rank is not None else 0.0
    ndcg_at_5 = 1.0 / math.log2(positive_rank + 1) if positive_rank is not None else 0.0
    return RerankInstance(
        condition=condition,
        user_id=user_id,
        source_instance_id=source_instance_id,
        diagnostic_id=diagnostic_id,
        candidate_ids=candidate_ids,
        positive_item_id=positive_item_id,
        response_received=response_received,
        parse_success=parse_success,
        schema_valid=schema_valid,
        ranked_item_ids=selected,
        valid_item_ids=valid_item_ids,
        invalid_item_ids=invalid_item_ids,
        selected_count=selected_count,
        valid_selected_count=len(valid_item_ids),
        candidate_validity_rate=candidate_validity_rate,
        duplicate_count=duplicate_count,
        hallucinated_count=hallucinated_count,
        positive_rank=positive_rank,
        hit_at_5=hit_at_5,
        ndcg_at_5=ndcg_at_5,
        selected_input_positions=selected_positions,
        latency_seconds=latency_seconds,
        parse_status=parse_status,
        status=instance_status(response_received=response_received, parse_status=parse_status),
        notes=notes,
    )


def instance_row(instance: RerankInstance) -> dict[str, str]:
    return {
        "plan": PLAN,
        "phase": PHASE,
        "dataset": DATASET,
        "scope": SCOPE,
        "condition": instance.condition,
        "user_id": instance.user_id,
        "source_instance_id": instance.source_instance_id,
        "diagnostic_id": instance.diagnostic_id,
        "backend": BACKEND if instance.condition != REFERENCE_CONDITION else "none",
        "model": model_for_condition(instance.condition),
        "prompt_strategy": PROMPT_STRATEGY if instance.condition != REFERENCE_CONDITION else "candidate_order_reference",
        "temperature": format_float(TEMPERATURE) if instance.condition != REFERENCE_CONDITION else "",
        "top_k": str(TOP_K),
        "candidate_pool_size": str(CANDIDATE_POOL_SIZE),
        "candidate_ids": json.dumps(instance.candidate_ids, separators=(",", ":")),
        "positive_item_id": instance.positive_item_id,
        "response_received": bool_text(instance.response_received),
        "parse_success": bool_text(instance.parse_success),
        "schema_valid": bool_text(instance.schema_valid),
        "selected_count": str(instance.selected_count),
        "valid_selected_count": str(instance.valid_selected_count),
        "candidate_validity_rate": format_float(instance.candidate_validity_rate),
        "duplicate_count": str(instance.duplicate_count),
        "hallucinated_count": str(instance.hallucinated_count),
        "positive_rank": "" if instance.positive_rank is None else str(instance.positive_rank),
        "hit_at_5": format_float(instance.hit_at_5),
        "ndcg_at_5": format_float(instance.ndcg_at_5),
        "selected_item_ids": json.dumps(instance.ranked_item_ids, separators=(",", ":")),
        "valid_selected_item_ids": json.dumps(instance.valid_item_ids, separators=(",", ":")),
        "selected_input_positions": json.dumps(instance.selected_input_positions, separators=(",", ":")),
        "latency_seconds": format_float(instance.latency_seconds),
        "parse_status": instance.parse_status,
        "status": instance.status,
        "notes": instance.notes,
    }


def build_condition_summary(instance_summary: pd.DataFrame, *, runtimes: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    reference_by_instance = _reference_selected_by_instance(instance_summary)
    for condition in CONDITIONS:
        subset = instance_summary.loc[instance_summary["condition"] == condition].copy()
        if subset.empty:
            continue
        selected = numeric(subset["selected_count"])
        valid = numeric(subset["valid_selected_count"])
        hallucinated = numeric(subset["hallucinated_count"])
        duplicates = numeric(subset["duplicate_count"])
        total_selected = int(selected.sum())
        total_valid = int(valid.sum())
        total_hallucinated = int(hallucinated.sum())
        total_duplicates = int(duplicates.sum())
        selected_positions = all_selected_positions(subset)
        position_metrics = selected_position_metrics(selected_positions)
        shift_metrics = position_shift_metrics(subset, reference_by_instance)
        positive_metrics = positive_selection_metrics(subset)
        num_instances = len(subset)
        rows.append(
            {
                "plan": PLAN,
                "phase": PHASE,
                "dataset": DATASET,
                "scope": SCOPE,
                "backend": BACKEND if condition != REFERENCE_CONDITION else "none",
                "model": model_for_condition(condition),
                "condition": condition,
                "prompt_strategy": PROMPT_STRATEGY if condition != REFERENCE_CONDITION else "candidate_order_reference",
                "total_instances": str(num_instances),
                "num_responses_received": str(bool_count(subset["response_received"])),
                "num_parse_success": str(bool_count(subset["parse_success"])),
                "num_schema_valid": str(bool_count(subset["schema_valid"])),
                "num_exact_top_k": str(int((selected == TOP_K).sum())),
                "num_empty_outputs": str(int((selected == 0).sum())),
                "total_selected_items": str(total_selected),
                "total_valid_selected_items": str(total_valid),
                "total_hallucinated_items": str(total_hallucinated),
                "total_duplicate_items": str(total_duplicates),
                "response_received_rate": format_rate(bool_count(subset["response_received"]), num_instances),
                "parse_success_rate": format_rate(bool_count(subset["parse_success"]), num_instances),
                "schema_valid_rate": format_rate(bool_count(subset["schema_valid"]), num_instances),
                "exact_top_k_rate": format_rate(int((selected == TOP_K).sum()), num_instances),
                "empty_output_rate": format_rate(int((selected == 0).sum()), num_instances),
                "mean_candidate_validity_rate": format_float(float(numeric(subset["candidate_validity_rate"]).mean())),
                "micro_candidate_validity_rate": format_rate(total_valid, total_selected),
                "hallucination_rate": format_rate(total_hallucinated, total_selected),
                "duplicate_rate": format_rate(total_duplicates, total_selected),
                "exact_5_valid_ids_rate": format_rate(int((valid == TOP_K).sum()), num_instances),
                "hr_at_5": format_float(float(numeric(subset["hit_at_5"]).mean())),
                "ndcg_at_5": format_float(float(numeric(subset["ndcg_at_5"]).mean())),
                "mean_latency_seconds": "" if condition == REFERENCE_CONDITION else format_float(float(numeric(subset["latency_seconds"]).mean())),
                "total_runtime_seconds": "" if condition == REFERENCE_CONDITION else format_float(runtimes.get(condition, 0.0)),
                "mean_selected_input_position": format_optional(position_metrics["mean"]),
                "median_selected_input_position": format_optional(position_metrics["median"]),
                "front_half_selection_rate": format_float(position_metrics["front_half"]),
                "back_half_selection_rate": format_float(position_metrics["back_half"]),
                "position_1_selection_rate": format_float(position_metrics["position_1"]),
                "position_20_selection_rate": format_float(position_metrics["position_20"]),
                "mean_absolute_position_shift_vs_reference": format_float(shift_metrics["mean_abs"]),
                "mean_signed_position_shift_vs_reference": format_float(shift_metrics["mean_signed"]),
                "selected_item_jaccard_vs_reference": format_float(shift_metrics["selected_jaccard"]),
                "valid_item_jaccard_vs_reference": format_float(shift_metrics["valid_jaccard"]),
                "positive_selected_rate": format_float(positive_metrics["selected_rate"]),
                "positive_rank_when_selected": format_optional(positive_metrics["rank_when_selected"]),
                "positive_input_position": positive_metrics["input_position"],
                "positive_position_shift_when_selected": format_optional(positive_metrics["position_shift_when_selected"]),
                "status": "completed_diagnostic_subset",
                "notes": condition_notes(condition),
            }
        )
    return pd.DataFrame(rows, columns=CONDITION_SUMMARY_COLUMNS)


def build_pairwise_delta(condition_summary: pd.DataFrame, agreement_profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    reference = summary_row(condition_summary, REFERENCE_CONDITION)
    for condition in (QWEN_CONDITION, MISTRAL_CONDITION):
        current = summary_row(condition_summary, condition)
        comparison = f"{condition}_vs_reference"
        for metric_name, column in {
            "HR@5_delta": "hr_at_5",
            "NDCG@5_delta": "ndcg_at_5",
            "candidate_validity_delta": "mean_candidate_validity_rate",
            "hallucination_rate_delta": "hallucination_rate",
            "duplicate_rate_delta": "duplicate_rate",
            "mean_selected_input_position_delta": "mean_selected_input_position",
            "mean_absolute_position_shift": "mean_absolute_position_shift_vs_reference",
        }.items():
            reference_value = float(reference[column] or 0.0)
            condition_value = float(current[column] or 0.0)
            delta = condition_value - reference_value
            rows.append(
                _pairwise_row(
                    comparison=comparison,
                    metric_name=metric_name,
                    reference_value=reference_value,
                    condition_value=condition_value,
                    delta=delta,
                    interpretation=delta_interpretation(metric_name, delta),
                )
            )
        for metric_name, column in {
            "selected_item_jaccard": "selected_item_jaccard_vs_reference",
            "valid_item_jaccard": "valid_item_jaccard_vs_reference",
        }.items():
            rows.append(
                _pairwise_row(
                    comparison=comparison,
                    metric_name=metric_name,
                    reference_value=1.0,
                    condition_value=float(current[column] or 0.0),
                    delta=float(current[column] or 0.0) - 1.0,
                    interpretation="overlap_with_candidate_order_reference",
                )
            )

    qwen = summary_row(condition_summary, QWEN_CONDITION)
    mistral = summary_row(condition_summary, MISTRAL_CONDITION)
    for metric_name, column in {
        "HR@5_delta": "hr_at_5",
        "NDCG@5_delta": "ndcg_at_5",
        "candidate_validity_delta": "mean_candidate_validity_rate",
        "hallucination_rate_delta": "hallucination_rate",
    }.items():
        qwen_value = float(qwen[column] or 0.0)
        mistral_value = float(mistral[column] or 0.0)
        rows.append(
            {
                "comparison": "qwen2_rerank_vs_mistral_rerank",
                "metric_name": metric_name,
                "reference_value": "",
                "qwen2_value": format_float(qwen_value),
                "mistral_value": format_float(mistral_value),
                "condition_value": "",
                "delta": format_float(mistral_value - qwen_value),
                "pairwise_value": "",
                "interpretation": delta_interpretation(metric_name, mistral_value - qwen_value),
            }
        )
    for metric_name, value in qwen_mistral_agreement_rates(agreement_profile).items():
        rows.append(
            {
                "comparison": "qwen2_rerank_vs_mistral_rerank",
                "metric_name": metric_name,
                "reference_value": "",
                "qwen2_value": "",
                "mistral_value": "",
                "condition_value": "",
                "delta": "",
                "pairwise_value": format_float(value),
                "interpretation": f"{metric_name}_computed_over_20_instances",
            }
        )
    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)


def build_agreement_profile(instance_summary: pd.DataFrame) -> pd.DataFrame:
    by_condition = {
        condition: {
            str(row["source_instance_id"]): row
            for _, row in instance_summary.loc[instance_summary["condition"] == condition].iterrows()
        }
        for condition in CONDITIONS
    }
    rows: list[dict[str, str]] = []
    for source_id, reference in by_condition[REFERENCE_CONDITION].items():
        qwen = by_condition[QWEN_CONDITION][source_id]
        mistral = by_condition[MISTRAL_CONDITION][source_id]
        reference_ids = json.loads(str(reference["selected_item_ids"]))
        qwen_ids = json.loads(str(qwen["selected_item_ids"]))
        mistral_ids = json.loads(str(mistral["selected_item_ids"]))
        qwen_valid = json.loads(str(qwen["valid_selected_item_ids"]))
        mistral_valid = json.loads(str(mistral["valid_selected_item_ids"]))
        reference_hit = float(reference["hit_at_5"]) > 0.0
        qwen_hit = float(qwen["hit_at_5"]) > 0.0
        mistral_hit = float(mistral["hit_at_5"]) > 0.0
        rows.append(
            {
                "diagnostic_id": str(qwen["diagnostic_id"]),
                "user_id": str(qwen["user_id"]),
                "positive_item_id": str(qwen["positive_item_id"]),
                "reference_top5_ids": json.dumps(reference_ids, separators=(",", ":")),
                "qwen_selected_ids": json.dumps(qwen_ids, separators=(",", ":")),
                "mistral_selected_ids": json.dumps(mistral_ids, separators=(",", ":")),
                "qwen_valid_selected_ids": json.dumps(qwen_valid, separators=(",", ":")),
                "mistral_valid_selected_ids": json.dumps(mistral_valid, separators=(",", ":")),
                "qwen_hit": bool_text(qwen_hit),
                "mistral_hit": bool_text(mistral_hit),
                "reference_hit": bool_text(reference_hit),
                "agreement_category": rerank_agreement_category(reference_hit, qwen_hit, mistral_hit),
                "selected_item_jaccard_qwen_vs_mistral": format_float(jaccard(qwen_ids, mistral_ids)),
                "valid_item_jaccard_qwen_vs_mistral": format_float(jaccard(qwen_valid, mistral_valid)),
            }
        )
    return pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)


def build_position_shift_profile(instance_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in instance_summary.loc[instance_summary["condition"].isin([QWEN_CONDITION, MISTRAL_CONDITION])].iterrows():
        selected = json.loads(str(row["selected_item_ids"]))
        valid_items = json.loads(str(row["valid_selected_item_ids"]))
        positions = json.loads(str(row["selected_input_positions"]))
        valid_index = 0
        for output_rank, item_id in enumerate(selected, start=1):
            is_valid = valid_index < len(valid_items) and item_id == valid_items[valid_index]
            original_position = int(positions[valid_index]) if is_valid else None
            if is_valid:
                valid_index += 1
            position_shift = "" if original_position is None else str(output_rank - int(original_position))
            rows.append(
                {
                    "diagnostic_id": str(row["diagnostic_id"]),
                    "user_id": str(row["user_id"]),
                    "model": str(row["model"]),
                    "selected_item_id": str(item_id),
                    "output_rank": str(output_rank),
                    "original_input_position": "" if original_position is None else str(original_position),
                    "position_shift": position_shift,
                    "is_positive": bool_text(str(item_id) == str(row["positive_item_id"])),
                    "is_valid_candidate": bool_text(is_valid),
                }
            )
    return pd.DataFrame(rows, columns=POSITION_SHIFT_COLUMNS)


def build_metric_rows(condition_summary: pd.DataFrame) -> pd.DataFrame:
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
        ("exact_5_valid_ids_rate", ""),
        ("hr_at_5", "5", "hit_rate"),
        ("ndcg_at_5", "5", "ndcg"),
        ("mean_selected_input_position", ""),
        ("median_selected_input_position", ""),
        ("front_half_selection_rate", ""),
        ("back_half_selection_rate", ""),
        ("position_1_selection_rate", ""),
        ("position_20_selection_rate", ""),
        ("mean_absolute_position_shift_vs_reference", ""),
        ("mean_signed_position_shift_vs_reference", ""),
        ("selected_item_jaccard_vs_reference", ""),
        ("valid_item_jaccard_vs_reference", ""),
        ("positive_selected_rate", ""),
        ("total_runtime_seconds", ""),
    ]
    rows: list[dict[str, str]] = []
    for _, row in condition_summary.iterrows():
        for spec in metric_columns:
            if len(spec) == 3:
                column, k, metric = spec
            else:
                column, k = spec
                metric = column
            value = str(row[column])
            if value == "":
                continue
            rows.append(
                {
                    "plan": PLAN,
                    "phase": PHASE,
                    "dataset": DATASET,
                    "scope": SCOPE,
                    "method": METHOD,
                    "backend": str(row["backend"]),
                    "model": str(row["model"]),
                    "condition": str(row["condition"]),
                    "prompt_strategy": str(row["prompt_strategy"]),
                    "metric": metric,
                    "k": k,
                    "value": value,
                    "num_instances": str(row["total_instances"]),
                    "notes": "D5J local LLM re-ranking diagnostic; not exact paper reproduction.",
                }
            )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def rerank_agreement_category(reference_hit: bool, qwen_hit: bool, mistral_hit: bool) -> str:
    if reference_hit and qwen_hit and mistral_hit:
        return "all_hit"
    if qwen_hit and not reference_hit and not mistral_hit:
        return "qwen_only_hit"
    if mistral_hit and not reference_hit and not qwen_hit:
        return "mistral_only_hit"
    if reference_hit and not qwen_hit and not mistral_hit:
        return "reference_only_hit"
    if qwen_hit and mistral_hit and not reference_hit:
        return "qwen_mistral_hit_only"
    if reference_hit and qwen_hit and not mistral_hit:
        return "reference_qwen_hit_only"
    if reference_hit and mistral_hit and not qwen_hit:
        return "reference_mistral_hit_only"
    return "none_hit"


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def positive_rank_at_k(ranked_item_ids: list[str], positive_item_id: str, top_k: int) -> int | None:
    for index, item_id in enumerate(ranked_item_ids[:top_k], start=1):
        if item_id == positive_item_id:
            return index
    return None


def selected_position_metrics(positions: list[int]) -> dict[str, float | None]:
    if not positions:
        return {"mean": None, "median": None, "front_half": 0.0, "back_half": 0.0, "position_1": 0.0, "position_20": 0.0}
    total = len(positions)
    return {
        "mean": sum(positions) / total,
        "median": float(median(positions)),
        "front_half": sum(1 for position in positions if position <= 10) / total,
        "back_half": sum(1 for position in positions if position >= 11) / total,
        "position_1": sum(1 for position in positions if position == 1) / total,
        "position_20": sum(1 for position in positions if position == 20) / total,
    }


def position_shift_metrics(subset: pd.DataFrame, reference_by_instance: dict[str, list[str]]) -> dict[str, float]:
    shifts: list[int] = []
    abs_shifts: list[int] = []
    selected_jaccards: list[float] = []
    valid_jaccards: list[float] = []
    for _, row in subset.iterrows():
        selected = json.loads(str(row["selected_item_ids"]))
        valid = json.loads(str(row["valid_selected_item_ids"]))
        positions = json.loads(str(row["selected_input_positions"]))
        reference = reference_by_instance[str(row["source_instance_id"])]
        selected_jaccards.append(jaccard(selected, reference))
        valid_jaccards.append(jaccard(valid, reference))
        valid_index = 0
        for output_rank, item_id in enumerate(selected, start=1):
            if valid_index < len(valid) and item_id == valid[valid_index]:
                shift = output_rank - int(positions[valid_index])
                shifts.append(shift)
                abs_shifts.append(abs(shift))
                valid_index += 1
    return {
        "mean_abs": sum(abs_shifts) / len(abs_shifts) if abs_shifts else 0.0,
        "mean_signed": sum(shifts) / len(shifts) if shifts else 0.0,
        "selected_jaccard": sum(selected_jaccards) / len(selected_jaccards) if selected_jaccards else 0.0,
        "valid_jaccard": sum(valid_jaccards) / len(valid_jaccards) if valid_jaccards else 0.0,
    }


def positive_selection_metrics(subset: pd.DataFrame) -> dict[str, float | str | None]:
    ranks: list[int] = []
    shifts: list[int] = []
    input_positions: list[str] = []
    selected_count = 0
    for _, row in subset.iterrows():
        positive_item_id = str(row["positive_item_id"])
        selected = json.loads(str(row["selected_item_ids"]))
        candidate_ids = json.loads(str(row.get("candidate_ids", "[]"))) if "candidate_ids" in row else []
        positive_input_position = ""
        if candidate_ids and positive_item_id in candidate_ids:
            positive_input_position = str(candidate_ids.index(positive_item_id) + 1)
            input_positions.append(positive_input_position)
        if positive_item_id in selected[:TOP_K]:
            rank = selected.index(positive_item_id) + 1
            ranks.append(rank)
            selected_count += 1
            if positive_input_position:
                shifts.append(rank - int(positive_input_position))
    total = len(subset)
    input_position = sorted(set(input_positions))
    return {
        "selected_rate": selected_count / total if total else 0.0,
        "rank_when_selected": sum(ranks) / len(ranks) if ranks else None,
        "input_position": "|".join(input_position),
        "position_shift_when_selected": sum(shifts) / len(shifts) if shifts else None,
    }


def qwen_mistral_agreement_rates(agreement_profile: pd.DataFrame) -> dict[str, float]:
    categories = Counter(agreement_profile["agreement_category"].astype(str))
    total = len(agreement_profile)
    agreement = sum(
        1
        for _, row in agreement_profile.iterrows()
        if str(row["qwen_hit"]).lower() == str(row["mistral_hit"]).lower()
    )
    return {
        "selected_item_jaccard": float(numeric(agreement_profile["selected_item_jaccard_qwen_vs_mistral"]).mean()) if total else 0.0,
        "valid_item_jaccard": float(numeric(agreement_profile["valid_item_jaccard_qwen_vs_mistral"]).mean()) if total else 0.0,
        "positive_selection_agreement_rate": agreement / total if total else 0.0,
        "both_hit_rate": (categories["all_hit"] + categories["qwen_mistral_hit_only"]) / total if total else 0.0,
        "qwen_only_hit_rate": categories["qwen_only_hit"] / total if total else 0.0,
        "mistral_only_hit_rate": categories["mistral_only_hit"] / total if total else 0.0,
        "neither_hit_rate": (categories["none_hit"] + categories["reference_only_hit"]) / total if total else 0.0,
    }


def raw_trace_row(*, instance: RerankInstance, raw_response_text: str, parsed_result: Any) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "diagnostic_id": instance.diagnostic_id,
        "source_instance_id": instance.source_instance_id,
        "user_id": instance.user_id,
        "condition": instance.condition,
        "backend": BACKEND,
        "model": model_for_condition(instance.condition),
        "prompt_strategy": PROMPT_STRATEGY,
        "raw_response_text": raw_response_text,
        "parsed_result": parsed_result,
        "latency_seconds": instance.latency_seconds,
        "status": instance.status,
    }


def model_for_condition(condition: str) -> str:
    if condition == QWEN_CONDITION:
        return QWEN_MODEL
    if condition == MISTRAL_CONDITION:
        return MISTRAL_MODEL
    return REFERENCE_MODEL


def manifest_status(condition: str) -> str:
    if condition == REFERENCE_CONDITION:
        return "reference_ready_no_inference"
    return "ready_for_local_llm_reranking"


def instance_status(*, response_received: bool, parse_status: str) -> str:
    if parse_status == "reference_from_candidate_order":
        return "completed_reference"
    if not response_received:
        return "failed_backend"
    if parse_status == "parsed_valid":
        return "completed"
    if parse_status == "parsed_with_hallucinations":
        return "completed_with_hallucinations"
    if parse_status == "parsed_with_duplicates":
        return "completed_with_duplicates"
    if parse_status == "parsed_invalid_schema":
        return "completed_invalid_schema"
    return "failed_parse"


def condition_notes(condition: str) -> str:
    if condition == REFERENCE_CONDITION:
        return "Reference top-5 from original candidate order; no LLM inference."
    return "Local LLM re-ranking prompt over the original candidate order."


def all_selected_positions(subset: pd.DataFrame) -> list[int]:
    positions: list[int] = []
    for encoded in subset["selected_input_positions"]:
        positions.extend(int(position) for position in json.loads(str(encoded)))
    return positions


def _reference_selected_by_instance(instance_summary: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for _, row in instance_summary.loc[instance_summary["condition"] == REFERENCE_CONDITION].iterrows():
        result[str(row["source_instance_id"])] = json.loads(str(row["selected_item_ids"]))
    return result


def summary_row(condition_summary: pd.DataFrame, condition: str) -> pd.Series:
    rows = condition_summary.loc[condition_summary["condition"] == condition]
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row for {condition}, found {len(rows)}")
    return rows.iloc[0]


def _pairwise_row(
    *,
    comparison: str,
    metric_name: str,
    reference_value: float,
    condition_value: float,
    delta: float,
    interpretation: str,
) -> dict[str, str]:
    return {
        "comparison": comparison,
        "metric_name": metric_name,
        "reference_value": format_float(reference_value),
        "qwen2_value": "",
        "mistral_value": "",
        "condition_value": format_float(condition_value),
        "delta": format_float(delta),
        "pairwise_value": "",
        "interpretation": interpretation,
    }


def delta_interpretation(metric_name: str, delta: float) -> str:
    if abs(delta) < 1e-12:
        return "no_change"
    return f"{metric_name}_{'increased' if delta > 0 else 'decreased'}"


def _history_titles_only(history_preview: str) -> str:
    parts = [part.strip() for part in str(history_preview or "").split("|") if part.strip()]
    titles: list[str] = []
    for part in parts:
        titles.append(part.split(":", 1)[1].strip() if ":" in part else part)
    return "; ".join(titles[:5]) if titles else "No title-level history preview is available."


def bool_count(series: pd.Series) -> int:
    return sum(1 for value in series if str(value).strip().lower() == "true")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def format_rate(numerator: int, denominator: int) -> str:
    return format_float((numerator / denominator) if denominator else 0.0)


def format_optional(value: float | None) -> str:
    return "" if value is None else format_float(value)


def format_float(value: float) -> str:
    return f"{float(value):.6f}"


def bool_text(value: bool) -> str:
    return "true" if value else "false"

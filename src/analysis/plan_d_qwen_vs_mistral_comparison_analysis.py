from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd


PLAN = "Plan D"
PHASE = "D5I_qwen_vs_mistral_ml1m_strict_prompt_diagnostic"
DATASET = "ml1m"
SCOPE = "diagnostic_subset"
BACKEND = "ollama"
QWEN_MODEL = "qwen2:7b-instruct"
MISTRAL_MODEL = "mistral:7b-instruct"
PROMPT_STRATEGY = "strict_candidate_id_only"
METHOD = "qwen_vs_mistral_strict_candidate_id"
TEMPERATURE = 0.0
TOP_K = 5
CANDIDATE_POOL_SIZE = 20

QWEN_CONDITION = "qwen2_strict_reference"
MISTRAL_CONDITION = "mistral_strict_candidate_id_prompt"

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
    "latency_seconds",
    "parse_status",
    "status",
    "notes",
]

MODEL_SUMMARY_COLUMNS = [
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
    "status",
    "notes",
]

PAIRWISE_COLUMNS = [
    "comparison",
    "metric_name",
    "qwen2_value",
    "mistral_value",
    "delta",
    "pairwise_value",
    "interpretation",
]

AGREEMENT_COLUMNS = [
    "diagnostic_id",
    "user_id",
    "positive_item_id",
    "qwen_selected_ids",
    "mistral_selected_ids",
    "qwen_valid_selected_ids",
    "mistral_valid_selected_ids",
    "selected_item_overlap_count",
    "valid_item_overlap_count",
    "selected_item_jaccard",
    "valid_item_jaccard",
    "qwen_hit",
    "mistral_hit",
    "agreement_category",
    "qwen_positive_rank",
    "mistral_positive_rank",
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
class ModelInstance:
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
    latency_seconds: float
    parse_status: str
    status: str
    notes: str


def candidate_order_signature(candidate_ids: list[str]) -> str:
    """Return a stable short signature for an ordered candidate list."""

    return hashlib.sha256("|".join(candidate_ids).encode("utf-8")).hexdigest()[:16]


def build_mistral_manifest(d5e_manifest: pd.DataFrame) -> pd.DataFrame:
    """Build the D5I 20-row Mistral manifest from the D5E/D5G instances."""

    rows: list[dict[str, str]] = []
    for _, row in d5e_manifest.iterrows():
        candidate_ids = parse_candidate_ids(str(row["candidate_ids"]))
        rows.append(
            {
                "diagnostic_id": f"{PHASE}_{MISTRAL_CONDITION}_{str(row['user_id'])}",
                "user_id": str(row["user_id"]),
                "dataset": DATASET,
                "condition": MISTRAL_CONDITION,
                "candidate_pool_size": str(len(candidate_ids)),
                "positive_item_id": str(row["positive_item_id"]),
                "candidate_order_signature": candidate_order_signature(candidate_ids),
                "prompt_strategy": PROMPT_STRATEGY,
                "model": MISTRAL_MODEL,
                "backend": BACKEND,
                "status": "ready_for_local_mistral_inference",
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def parse_candidate_ids(encoded: str) -> list[str]:
    value = json.loads(encoded)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("candidate_ids must be a JSON list of strings")
    return value


def evaluate_ranked_ids(
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
) -> ModelInstance:
    """Evaluate one strict-prompt selected list for D5I."""

    selected = [str(item_id) for item_id in ranked_item_ids]
    candidates = set(candidate_ids)
    valid_item_ids = [item_id for item_id in selected if item_id in candidates]
    invalid_item_ids = [item_id for item_id in selected if item_id not in candidates]
    selected_count = len(selected)
    duplicate_count = selected_count - len(set(selected))
    hallucinated_count = len(invalid_item_ids)
    candidate_validity_rate = len(valid_item_ids) / selected_count if selected_count else 0.0
    positive_rank = positive_rank_at_k(selected, positive_item_id, TOP_K)
    hit_at_5 = 1.0 if positive_rank is not None else 0.0
    ndcg_at_5 = 1.0 / math.log2(positive_rank + 1) if positive_rank is not None else 0.0
    return ModelInstance(
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
        latency_seconds=latency_seconds,
        parse_status=parse_status,
        status=instance_status(response_received=response_received, parse_status=parse_status),
        notes=notes,
    )


def instance_row(instance: ModelInstance) -> dict[str, str]:
    return {
        "plan": PLAN,
        "phase": PHASE,
        "dataset": DATASET,
        "scope": SCOPE,
        "condition": instance.condition,
        "user_id": instance.user_id,
        "source_instance_id": instance.source_instance_id,
        "diagnostic_id": instance.diagnostic_id,
        "backend": BACKEND,
        "model": model_for_condition(instance.condition),
        "prompt_strategy": PROMPT_STRATEGY,
        "temperature": format_float(TEMPERATURE),
        "top_k": str(TOP_K),
        "candidate_pool_size": str(CANDIDATE_POOL_SIZE),
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
        "latency_seconds": format_float(instance.latency_seconds),
        "parse_status": instance.parse_status,
        "status": instance.status,
        "notes": instance.notes,
    }


def build_model_summary(instance_summary: pd.DataFrame, *, mistral_runtime_seconds: float) -> pd.DataFrame:
    """Aggregate D5I rows by local model condition."""

    rows: list[dict[str, str]] = []
    for condition in (QWEN_CONDITION, MISTRAL_CONDITION):
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
        num_instances = len(subset)
        rows.append(
            {
                "plan": PLAN,
                "phase": PHASE,
                "dataset": DATASET,
                "scope": SCOPE,
                "backend": BACKEND,
                "model": model_for_condition(condition),
                "condition": condition,
                "prompt_strategy": PROMPT_STRATEGY,
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
                "mean_latency_seconds": format_float(float(numeric(subset["latency_seconds"]).mean())),
                "total_runtime_seconds": "" if condition == QWEN_CONDITION else format_float(mistral_runtime_seconds),
                "status": "completed_diagnostic_subset",
                "notes": condition_notes(condition),
            }
        )
    return pd.DataFrame(rows, columns=MODEL_SUMMARY_COLUMNS)


def build_agreement_profile(instance_summary: pd.DataFrame) -> pd.DataFrame:
    """Build per-instance Qwen-vs-Mistral selected-item agreement rows."""

    qwen = {
        str(row["source_instance_id"]): row
        for _, row in instance_summary.loc[instance_summary["condition"] == QWEN_CONDITION].iterrows()
    }
    rows: list[dict[str, str]] = []
    for _, mistral in instance_summary.loc[instance_summary["condition"] == MISTRAL_CONDITION].iterrows():
        key = str(mistral["source_instance_id"])
        qwen_row = qwen[key]
        qwen_selected = json.loads(str(qwen_row["selected_item_ids"]))
        mistral_selected = json.loads(str(mistral["selected_item_ids"]))
        qwen_valid = json.loads(str(qwen_row["valid_selected_item_ids"]))
        mistral_valid = json.loads(str(mistral["valid_selected_item_ids"]))
        qwen_hit = float(qwen_row["hit_at_5"]) > 0.0
        mistral_hit = float(mistral["hit_at_5"]) > 0.0
        rows.append(
            {
                "diagnostic_id": str(mistral["diagnostic_id"]),
                "user_id": str(mistral["user_id"]),
                "positive_item_id": str(mistral["positive_item_id"]),
                "qwen_selected_ids": json.dumps(qwen_selected, separators=(",", ":")),
                "mistral_selected_ids": json.dumps(mistral_selected, separators=(",", ":")),
                "qwen_valid_selected_ids": json.dumps(qwen_valid, separators=(",", ":")),
                "mistral_valid_selected_ids": json.dumps(mistral_valid, separators=(",", ":")),
                "selected_item_overlap_count": str(len(set(qwen_selected) & set(mistral_selected))),
                "valid_item_overlap_count": str(len(set(qwen_valid) & set(mistral_valid))),
                "selected_item_jaccard": format_float(jaccard(qwen_selected, mistral_selected)),
                "valid_item_jaccard": format_float(jaccard(qwen_valid, mistral_valid)),
                "qwen_hit": bool_text(qwen_hit),
                "mistral_hit": bool_text(mistral_hit),
                "agreement_category": agreement_category(qwen_hit, mistral_hit),
                "qwen_positive_rank": str(qwen_row["positive_rank"]),
                "mistral_positive_rank": str(mistral["positive_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)


def build_pairwise_delta(model_summary: pd.DataFrame, agreement_profile: pd.DataFrame) -> pd.DataFrame:
    """Build long-form D5I pairwise deltas and agreement rates."""

    qwen = summary_row(model_summary, QWEN_CONDITION)
    mistral = summary_row(model_summary, MISTRAL_CONDITION)
    metric_map = {
        "HR@5_delta": "hr_at_5",
        "NDCG@5_delta": "ndcg_at_5",
        "candidate_validity_delta": "mean_candidate_validity_rate",
        "hallucination_rate_delta": "hallucination_rate",
        "parse_success_delta": "parse_success_rate",
        "schema_valid_delta": "schema_valid_rate",
        "exact_top_k_delta": "exact_top_k_rate",
        "duplicate_rate_delta": "duplicate_rate",
        "mean_latency_delta": "mean_latency_seconds",
    }
    rows: list[dict[str, str]] = []
    for metric_name, column in metric_map.items():
        qwen_value = float(qwen[column])
        mistral_value = float(mistral[column])
        rows.append(
            {
                "comparison": "mistral_vs_qwen2_strict",
                "metric_name": metric_name,
                "qwen2_value": format_float(qwen_value),
                "mistral_value": format_float(mistral_value),
                "delta": format_float(mistral_value - qwen_value),
                "pairwise_value": "",
                "interpretation": delta_interpretation(metric_name, mistral_value - qwen_value),
            }
        )

    pairwise_values = agreement_rates(agreement_profile)
    for metric_name, value in pairwise_values.items():
        rows.append(
            {
                "comparison": "mistral_vs_qwen2_strict",
                "metric_name": metric_name,
                "qwen2_value": "",
                "mistral_value": "",
                "delta": "",
                "pairwise_value": format_float(value),
                "interpretation": agreement_interpretation(metric_name, value),
            }
        )
    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)


def build_metric_rows(model_summary: pd.DataFrame) -> pd.DataFrame:
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
        ("mean_latency_seconds", ""),
        ("total_runtime_seconds", ""),
        ("total_selected_items", ""),
        ("total_valid_selected_items", ""),
        ("total_hallucinated_items", ""),
    ]
    rows: list[dict[str, str]] = []
    for _, row in model_summary.iterrows():
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
                    "backend": BACKEND,
                    "model": str(row["model"]),
                    "condition": str(row["condition"]),
                    "prompt_strategy": PROMPT_STRATEGY,
                    "metric": metric,
                    "k": k,
                    "value": value,
                    "num_instances": str(row["total_instances"]),
                    "notes": "D5I local Qwen2-vs-Mistral strict-prompt comparison; not exact paper reproduction.",
                }
            )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def add_selected_id_columns(instance_summary: pd.DataFrame, instances: list[ModelInstance]) -> pd.DataFrame:
    """Attach compact selected-ID JSON columns needed for agreement analysis."""

    selected = {
        instance.diagnostic_id: (
            json.dumps(instance.ranked_item_ids, separators=(",", ":")),
            json.dumps(instance.valid_item_ids, separators=(",", ":")),
        )
        for instance in instances
    }
    frame = instance_summary.copy()
    frame["selected_item_ids"] = [selected[str(row["diagnostic_id"])][0] for _, row in frame.iterrows()]
    frame["valid_selected_item_ids"] = [selected[str(row["diagnostic_id"])][1] for _, row in frame.iterrows()]
    return frame


def raw_mistral_trace_row(*, instance: ModelInstance, raw_response_text: str, parsed_result: Any) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "diagnostic_id": instance.diagnostic_id,
        "source_instance_id": instance.source_instance_id,
        "user_id": instance.user_id,
        "condition": MISTRAL_CONDITION,
        "backend": BACKEND,
        "model": MISTRAL_MODEL,
        "prompt_strategy": PROMPT_STRATEGY,
        "raw_response_text": raw_response_text,
        "parsed_result": parsed_result,
        "latency_seconds": instance.latency_seconds,
        "status": instance.status,
    }


def positive_rank_at_k(ranked_item_ids: list[str], positive_item_id: str, top_k: int) -> int | None:
    for index, item_id in enumerate(ranked_item_ids[:top_k], start=1):
        if item_id == positive_item_id:
            return index
    return None


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def agreement_category(qwen_hit: bool, mistral_hit: bool) -> str:
    if qwen_hit and mistral_hit:
        return "both_hit"
    if qwen_hit:
        return "qwen_only_hit"
    if mistral_hit:
        return "mistral_only_hit"
    return "neither_hit"


def agreement_rates(agreement_profile: pd.DataFrame) -> dict[str, float]:
    if agreement_profile.empty:
        return {
            "valid_recommendation_jaccard": 0.0,
            "selected_item_jaccard": 0.0,
            "positive_selection_agreement_rate": 0.0,
            "both_hit_rate": 0.0,
            "qwen_only_hit_rate": 0.0,
            "mistral_only_hit_rate": 0.0,
            "neither_hit_rate": 0.0,
        }
    categories = Counter(agreement_profile["agreement_category"].astype(str))
    total = len(agreement_profile)
    agreement = sum(
        1
        for _, row in agreement_profile.iterrows()
        if str(row["qwen_hit"]).lower() == str(row["mistral_hit"]).lower()
    )
    return {
        "valid_recommendation_jaccard": float(numeric(agreement_profile["valid_item_jaccard"]).mean()),
        "selected_item_jaccard": float(numeric(agreement_profile["selected_item_jaccard"]).mean()),
        "positive_selection_agreement_rate": agreement / total,
        "both_hit_rate": categories["both_hit"] / total,
        "qwen_only_hit_rate": categories["qwen_only_hit"] / total,
        "mistral_only_hit_rate": categories["mistral_only_hit"] / total,
        "neither_hit_rate": categories["neither_hit"] / total,
    }


def model_for_condition(condition: str) -> str:
    return QWEN_MODEL if condition == QWEN_CONDITION else MISTRAL_MODEL


def instance_status(*, response_received: bool, parse_status: str) -> str:
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
    if condition == QWEN_CONDITION:
        return "Reused D5G Qwen2 strict-prompt output; Qwen2 was not rerun."
    return "New D5I Mistral strict-prompt local Ollama output over the same 20 D5G instances."


def summary_row(model_summary: pd.DataFrame, condition: str) -> pd.Series:
    rows = model_summary.loc[model_summary["condition"] == condition]
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row for {condition}, found {len(rows)}")
    return rows.iloc[0]


def delta_interpretation(metric_name: str, delta: float) -> str:
    if abs(delta) < 1e-12:
        return "no_change_vs_qwen2"
    return f"{metric_name}_{'increased' if delta > 0 else 'decreased'}_for_mistral"


def agreement_interpretation(metric_name: str, value: float) -> str:
    return f"{metric_name}_computed_over_20_instances"


def bool_count(series: pd.Series) -> int:
    return sum(1 for value in series if str(value).strip().lower() == "true")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def format_rate(numerator: int, denominator: int) -> str:
    return format_float((numerator / denominator) if denominator else 0.0)


def format_float(value: float) -> str:
    return f"{float(value):.6f}"


def bool_text(value: bool) -> str:
    return "true" if value else "false"

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


PLAN = "Plan D"
PHASE = "D5G_qwen2_ml1m_strict_prompt_diagnostic"
DATASET = "ml1m"
SCOPE = "diagnostic_subset"
BACKEND = "ollama"
MODEL = "qwen2:7b-instruct"
PROMPT_STRATEGY = "strict_candidate_id_only"
ORIGINAL_PROMPT_STRATEGY = "openp5_base"
METHOD = "qwen2_ollama_strict_candidate_id"
TEMPERATURE = 0.0
TOP_K = 5
CANDIDATE_POOL_SIZE = 20

ORIGINAL_CONDITION = "original_parsed_output"
ORIGINAL_POST_FILTERED_CONDITION = "original_post_filtered"
STRICT_CONDITION = "strict_candidate_id_prompt"
STRICT_POST_FILTERED_CONDITION = "strict_post_filtered"
CONDITIONS = (
    ORIGINAL_CONDITION,
    ORIGINAL_POST_FILTERED_CONDITION,
    STRICT_CONDITION,
    STRICT_POST_FILTERED_CONDITION,
)
MANIFEST_CONDITIONS = (
    ORIGINAL_POST_FILTERED_CONDITION,
    STRICT_CONDITION,
    STRICT_POST_FILTERED_CONDITION,
)

MANIFEST_COLUMNS = [
    "diagnostic_id",
    "user_id",
    "dataset",
    "condition",
    "candidate_pool_size",
    "positive_item_id",
    "candidate_order_signature",
    "prompt_strategy",
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
    "items_removed_by_filter",
    "filtered_from_condition",
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
    "hr_at_5",
    "ndcg_at_5",
    "mean_output_length",
    "exact_5_valid_ids_rate",
    "status",
    "notes",
]

PAIRWISE_COLUMNS = [
    "condition",
    "metric_name",
    "original_value",
    "condition_value",
    "delta",
    "interpretation",
]

POST_FILTER_PROFILE_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "condition",
    "source_condition",
    "num_instances",
    "mean_items_removed_by_filter",
    "total_items_removed_by_filter",
    "filtered_list_mean_length",
    "filtered_list_exact_top_k_rate",
    "filtered_HR@5",
    "filtered_NDCG@5",
    "filtered_candidate_validity_rate",
    "filtered_hallucination_rate",
    "utility_delta_after_filter",
    "ndcg_delta_after_filter",
    "hallucination_delta_after_filter",
    "validity_delta_after_filter",
    "notes",
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
class SelectionEvaluation:
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


@dataclass(frozen=True)
class ConditionInstance:
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
    items_removed_by_filter: int
    filtered_from_condition: str
    status: str
    notes: str


def build_strict_candidate_id_prompt(
    *,
    user_id: str,
    history_preview: str,
    candidate_ids: list[str],
    top_k: int = TOP_K,
) -> str:
    """Build the D5G candidate-ID-only mitigation prompt."""

    _validate_candidate_ids(candidate_ids, top_k)
    allowed_ids = json.dumps(candidate_ids, separators=(",", ":"))
    history_context = _history_titles_only(history_preview)
    return f"""You are running a local recommender diagnostic for user `{user_id}`.

Allowed candidate IDs, copy from this list only:
{allowed_ids}

User preference context, not valid output IDs:
{history_context}

Rules:
- Return only valid JSON.
- Select exactly {top_k} item IDs.
- Every selected ID must be copied exactly from the allowed candidate ID list.
- Do not invent IDs.
- Do not output movie titles.
- Do not output explanations.
- If uncertain, still choose from the provided candidate IDs only.
- Do not use any ID not shown in the allowed candidate ID list.

Required JSON object:
ranked_item_ids must be an array of exactly {top_k} strings copied from the allowed list; status must be "ok".
"""


def post_filter_ranked_ids(ranked_item_ids: list[str], candidate_ids: set[str]) -> list[str]:
    """Remove non-candidate IDs while preserving the original valid order; do not backfill."""

    return [item_id for item_id in ranked_item_ids if item_id in candidate_ids]


def evaluate_selected_ids(
    ranked_item_ids: list[str],
    *,
    candidate_ids: set[str],
    positive_item_id: str,
    top_k: int = TOP_K,
) -> SelectionEvaluation:
    """Compute D5G utility and validity metrics for one selected ID list."""

    selected = [str(item_id) for item_id in ranked_item_ids]
    valid_item_ids = [item_id for item_id in selected if item_id in candidate_ids]
    invalid_item_ids = [item_id for item_id in selected if item_id not in candidate_ids]
    selected_count = len(selected)
    duplicate_count = selected_count - len(set(selected))
    hallucinated_count = len(invalid_item_ids)
    candidate_validity_rate = len(valid_item_ids) / selected_count if selected_count else 0.0
    positive_rank = _positive_rank(selected, positive_item_id, top_k)
    hit_at_5 = 1.0 if positive_rank is not None else 0.0
    ndcg_at_5 = 1.0 / math.log2(positive_rank + 1) if positive_rank is not None else 0.0
    return SelectionEvaluation(
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
    )


def candidate_order_signature(candidate_ids: list[str]) -> str:
    """Return a stable short signature for an ordered candidate list."""

    return hashlib.sha256("|".join(candidate_ids).encode("utf-8")).hexdigest()[:16]


def build_strict_prompt_manifest(d5e_manifest: pd.DataFrame) -> pd.DataFrame:
    """Build the D5G condition manifest from the D5E 20-user manifest."""

    rows: list[dict[str, str]] = []
    for _, row in d5e_manifest.iterrows():
        candidate_ids = _parse_candidate_ids(str(row["candidate_ids"]))
        positive_item_id = str(row["positive_item_id"])
        signature = candidate_order_signature(candidate_ids)
        for condition in MANIFEST_CONDITIONS:
            prompt_strategy = PROMPT_STRATEGY if "strict" in condition else f"{ORIGINAL_PROMPT_STRATEGY}_post_filter"
            rows.append(
                {
                    "diagnostic_id": f"{PHASE}_{condition}_{str(row['user_id'])}",
                    "user_id": str(row["user_id"]),
                    "dataset": DATASET,
                    "condition": condition,
                    "candidate_pool_size": str(len(candidate_ids)),
                    "positive_item_id": positive_item_id,
                    "candidate_order_signature": signature,
                    "prompt_strategy": prompt_strategy,
                    "status": _manifest_status(condition),
                }
            )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def condition_instance_from_selection(
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
    ranked_item_ids: list[str],
    latency_seconds: float,
    parse_status: str,
    items_removed_by_filter: int,
    filtered_from_condition: str,
    notes: str,
) -> ConditionInstance:
    """Create a D5G condition instance from selected IDs and parser flags."""

    evaluation = evaluate_selected_ids(
        ranked_item_ids,
        candidate_ids=set(candidate_ids),
        positive_item_id=positive_item_id,
        top_k=TOP_K,
    )
    return ConditionInstance(
        condition=condition,
        user_id=user_id,
        source_instance_id=source_instance_id,
        diagnostic_id=diagnostic_id,
        candidate_ids=candidate_ids,
        positive_item_id=positive_item_id,
        response_received=response_received,
        parse_success=parse_success,
        schema_valid=schema_valid,
        ranked_item_ids=evaluation.ranked_item_ids,
        valid_item_ids=evaluation.valid_item_ids,
        invalid_item_ids=evaluation.invalid_item_ids,
        selected_count=evaluation.selected_count,
        valid_selected_count=evaluation.valid_selected_count,
        candidate_validity_rate=evaluation.candidate_validity_rate,
        duplicate_count=evaluation.duplicate_count,
        hallucinated_count=evaluation.hallucinated_count,
        positive_rank=evaluation.positive_rank,
        hit_at_5=evaluation.hit_at_5,
        ndcg_at_5=evaluation.ndcg_at_5,
        latency_seconds=latency_seconds,
        parse_status=parse_status,
        items_removed_by_filter=items_removed_by_filter,
        filtered_from_condition=filtered_from_condition,
        status=_instance_status(response_received=response_received, parse_status=parse_status),
        notes=notes,
    )


def post_filtered_instance(source: ConditionInstance, *, condition: str) -> ConditionInstance:
    """Build a deterministic post-filtered instance from one parsed LLM output."""

    filtered_ids = post_filter_ranked_ids(source.ranked_item_ids, set(source.candidate_ids))
    removed = source.selected_count - len(filtered_ids)
    return condition_instance_from_selection(
        condition=condition,
        user_id=source.user_id,
        source_instance_id=source.source_instance_id,
        diagnostic_id=f"{PHASE}_{condition}_{source.user_id}",
        candidate_ids=source.candidate_ids,
        positive_item_id=source.positive_item_id,
        response_received=source.response_received,
        parse_success=source.parse_success,
        schema_valid=source.schema_valid,
        ranked_item_ids=filtered_ids,
        latency_seconds=0.0,
        parse_status="post_filtered",
        items_removed_by_filter=removed,
        filtered_from_condition=source.condition,
        notes="Deterministic post-filter removed non-candidate IDs and did not backfill.",
    )


def instance_row(instance: ConditionInstance) -> dict[str, str]:
    """Return one compact committed per-instance summary row."""

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
        "model": MODEL,
        "prompt_strategy": _prompt_strategy_for_condition(instance.condition),
        "temperature": _format_float(TEMPERATURE),
        "top_k": str(TOP_K),
        "candidate_pool_size": str(CANDIDATE_POOL_SIZE),
        "positive_item_id": instance.positive_item_id,
        "response_received": _bool_text(instance.response_received),
        "parse_success": _bool_text(instance.parse_success),
        "schema_valid": _bool_text(instance.schema_valid),
        "selected_count": str(instance.selected_count),
        "valid_selected_count": str(instance.valid_selected_count),
        "candidate_validity_rate": _format_float(instance.candidate_validity_rate),
        "duplicate_count": str(instance.duplicate_count),
        "hallucinated_count": str(instance.hallucinated_count),
        "positive_rank": "" if instance.positive_rank is None else str(instance.positive_rank),
        "hit_at_5": _format_float(instance.hit_at_5),
        "ndcg_at_5": _format_float(instance.ndcg_at_5),
        "latency_seconds": _format_float(instance.latency_seconds),
        "parse_status": instance.parse_status,
        "items_removed_by_filter": str(instance.items_removed_by_filter),
        "filtered_from_condition": instance.filtered_from_condition,
        "status": instance.status,
        "notes": instance.notes,
    }


def build_condition_summary(instance_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate D5G metrics by condition."""

    rows: list[dict[str, str]] = []
    for condition in CONDITIONS:
        subset = instance_summary.loc[instance_summary["condition"] == condition].copy()
        if subset.empty:
            continue
        selected = _numeric(subset["selected_count"])
        valid = _numeric(subset["valid_selected_count"])
        hallucinated = _numeric(subset["hallucinated_count"])
        duplicates = _numeric(subset["duplicate_count"])
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
                "model": MODEL,
                "condition": condition,
                "prompt_strategy": _prompt_strategy_for_condition(condition),
                "total_instances": str(num_instances),
                "num_responses_received": str(_bool_count(subset["response_received"])),
                "num_parse_success": str(_bool_count(subset["parse_success"])),
                "num_schema_valid": str(_bool_count(subset["schema_valid"])),
                "num_exact_top_k": str(int((selected == TOP_K).sum())),
                "num_empty_outputs": str(int((selected == 0).sum())),
                "total_selected_items": str(total_selected),
                "total_valid_selected_items": str(total_valid),
                "total_hallucinated_items": str(total_hallucinated),
                "total_duplicate_items": str(total_duplicates),
                "response_received_rate": _format_rate(_bool_count(subset["response_received"]), num_instances),
                "parse_success_rate": _format_rate(_bool_count(subset["parse_success"]), num_instances),
                "schema_valid_rate": _format_rate(_bool_count(subset["schema_valid"]), num_instances),
                "exact_top_k_rate": _format_rate(int((selected == TOP_K).sum()), num_instances),
                "empty_output_rate": _format_rate(int((selected == 0).sum()), num_instances),
                "mean_candidate_validity_rate": _format_float(float(_numeric(subset["candidate_validity_rate"]).mean())),
                "micro_candidate_validity_rate": _format_rate(total_valid, total_selected),
                "hallucination_rate": _format_rate(total_hallucinated, total_selected),
                "duplicate_rate": _format_rate(total_duplicates, total_selected),
                "hr_at_5": _format_float(float(_numeric(subset["hit_at_5"]).mean())),
                "ndcg_at_5": _format_float(float(_numeric(subset["ndcg_at_5"]).mean())),
                "mean_output_length": _format_float(float(selected.mean())),
                "exact_5_valid_ids_rate": _format_rate(int((valid == TOP_K).sum()), num_instances),
                "status": "completed_diagnostic_subset",
                "notes": _condition_notes(condition),
            }
        )
    return pd.DataFrame(rows, columns=CONDITION_SUMMARY_COLUMNS)


def build_pairwise_delta(condition_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare D5G conditions against the original D5E parsed-output condition."""

    original = _summary_row(condition_summary, ORIGINAL_CONDITION)
    metric_map = {
        "HR@5": "hr_at_5",
        "NDCG@5": "ndcg_at_5",
        "hallucination_rate": "hallucination_rate",
        "mean_candidate_validity_rate": "mean_candidate_validity_rate",
        "micro_candidate_validity_rate": "micro_candidate_validity_rate",
        "exact_top_k_rate": "exact_top_k_rate",
        "exact_5_valid_ids_rate": "exact_5_valid_ids_rate",
    }
    rows: list[dict[str, str]] = []
    for condition in CONDITIONS:
        condition_row = _summary_row(condition_summary, condition)
        for metric_name, column in metric_map.items():
            original_value = float(original[column])
            condition_value = float(condition_row[column])
            delta = condition_value - original_value
            rows.append(
                {
                    "condition": condition,
                    "metric_name": metric_name,
                    "original_value": _format_float(original_value),
                    "condition_value": _format_float(condition_value),
                    "delta": _format_float(delta),
                    "interpretation": _delta_interpretation(metric_name, delta),
                }
            )
    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)


def build_post_filter_profile(instance_summary: pd.DataFrame, condition_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize deterministic post-filter effects for original and strict outputs."""

    rows: list[dict[str, str]] = []
    for filtered_condition, source_condition in (
        (ORIGINAL_POST_FILTERED_CONDITION, ORIGINAL_CONDITION),
        (STRICT_POST_FILTERED_CONDITION, STRICT_CONDITION),
    ):
        filtered_instances = instance_summary.loc[instance_summary["condition"] == filtered_condition].copy()
        filtered_summary = _summary_row(condition_summary, filtered_condition)
        source_summary = _summary_row(condition_summary, source_condition)
        removed = _numeric(filtered_instances["items_removed_by_filter"])
        rows.append(
            {
                "plan": PLAN,
                "phase": PHASE,
                "dataset": DATASET,
                "scope": SCOPE,
                "condition": filtered_condition,
                "source_condition": source_condition,
                "num_instances": str(len(filtered_instances)),
                "mean_items_removed_by_filter": _format_float(float(removed.mean()) if len(removed) else 0.0),
                "total_items_removed_by_filter": str(int(removed.sum())),
                "filtered_list_mean_length": str(filtered_summary["mean_output_length"]),
                "filtered_list_exact_top_k_rate": str(filtered_summary["exact_top_k_rate"]),
                "filtered_HR@5": str(filtered_summary["hr_at_5"]),
                "filtered_NDCG@5": str(filtered_summary["ndcg_at_5"]),
                "filtered_candidate_validity_rate": str(filtered_summary["mean_candidate_validity_rate"]),
                "filtered_hallucination_rate": str(filtered_summary["hallucination_rate"]),
                "utility_delta_after_filter": _format_float(
                    float(filtered_summary["hr_at_5"]) - float(source_summary["hr_at_5"])
                ),
                "ndcg_delta_after_filter": _format_float(
                    float(filtered_summary["ndcg_at_5"]) - float(source_summary["ndcg_at_5"])
                ),
                "hallucination_delta_after_filter": _format_float(
                    float(filtered_summary["hallucination_rate"]) - float(source_summary["hallucination_rate"])
                ),
                "validity_delta_after_filter": _format_float(
                    float(filtered_summary["mean_candidate_validity_rate"])
                    - float(source_summary["mean_candidate_validity_rate"])
                ),
                "notes": "Post-filter removes invalid IDs, preserves valid order, and does not backfill.",
            }
        )
    return pd.DataFrame(rows, columns=POST_FILTER_PROFILE_COLUMNS)


def build_metric_rows(condition_summary: pd.DataFrame) -> pd.DataFrame:
    """Build metric-style rows for D5G condition summaries."""

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
        ("hit_rate", "5", "hr_at_5"),
        ("ndcg", "5", "ndcg_at_5"),
        ("mean_output_length", ""),
        ("exact_5_valid_ids_rate", ""),
        ("total_selected_items", ""),
        ("total_valid_selected_items", ""),
        ("total_hallucinated_items", ""),
    ]
    rows: list[dict[str, str]] = []
    for _, row in condition_summary.iterrows():
        for metric_spec in metric_columns:
            if len(metric_spec) == 3:
                metric, k, column = metric_spec
            else:
                metric, k = metric_spec
                column = metric
            rows.append(
                {
                    "plan": PLAN,
                    "phase": PHASE,
                    "dataset": DATASET,
                    "scope": SCOPE,
                    "method": METHOD,
                    "backend": BACKEND,
                    "model": MODEL,
                    "condition": str(row["condition"]),
                    "prompt_strategy": str(row["prompt_strategy"]),
                    "metric": metric,
                    "k": k,
                    "value": str(row[column]),
                    "num_instances": str(row["total_instances"]),
                    "notes": "D5G strict prompt and post-filter diagnostic; not exact paper reproduction.",
                }
            )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def raw_strict_trace_row(
    *,
    instance: ConditionInstance,
    raw_response_text: str,
    parsed_result: Any,
) -> dict[str, Any]:
    """Return one JSONL-safe raw D5G strict prompt trace row without saving prompts."""

    parsed_payload = parsed_result if isinstance(parsed_result, dict) else getattr(parsed_result, "__dict__", {})
    return {
        "diagnostic_id": instance.diagnostic_id,
        "source_instance_id": instance.source_instance_id,
        "user_id": instance.user_id,
        "condition": STRICT_CONDITION,
        "model": MODEL,
        "backend": BACKEND,
        "prompt_strategy": PROMPT_STRATEGY,
        "raw_response_text": raw_response_text,
        "parsed_result": parsed_payload,
        "latency_seconds": instance.latency_seconds,
        "error_message": instance.notes if instance.status.startswith("failed") else "",
    }


def _validate_candidate_ids(candidate_ids: list[str], top_k: int) -> None:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ids must be unique")
    if top_k > len(candidate_ids):
        raise ValueError("top_k cannot exceed candidate count")


def _history_titles_only(history_preview: str) -> str:
    parts = [part.strip() for part in str(history_preview or "").split("|") if part.strip()]
    titles: list[str] = []
    for part in parts:
        if ":" in part:
            titles.append(part.split(":", 1)[1].strip())
        else:
            titles.append(part)
    return "; ".join(titles[:5]) if titles else "No title-level history preview is available."


def _parse_candidate_ids(encoded: str) -> list[str]:
    value = json.loads(encoded)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("candidate_ids must be a JSON list of strings")
    return value


def _positive_rank(ranked_item_ids: list[str], positive_item_id: str, top_k: int) -> int | None:
    for index, item_id in enumerate(ranked_item_ids[:top_k], start=1):
        if item_id == positive_item_id:
            return index
    return None


def _manifest_status(condition: str) -> str:
    if condition == STRICT_CONDITION:
        return "ready_for_local_qwen_inference"
    return "ready_for_deterministic_post_filter"


def _prompt_strategy_for_condition(condition: str) -> str:
    if condition in {STRICT_CONDITION, STRICT_POST_FILTERED_CONDITION}:
        return PROMPT_STRATEGY
    return ORIGINAL_PROMPT_STRATEGY


def _instance_status(*, response_received: bool, parse_status: str) -> str:
    if not response_received:
        return "failed_backend"
    if parse_status == "post_filtered":
        return "completed_post_filter"
    if parse_status == "parsed_valid":
        return "completed"
    if parse_status == "parsed_with_hallucinations":
        return "completed_with_hallucinations"
    if parse_status == "parsed_with_duplicates":
        return "completed_with_duplicates"
    if parse_status == "parsed_invalid_schema":
        return "completed_invalid_schema"
    return "failed_parse"


def _condition_notes(condition: str) -> str:
    if condition == ORIGINAL_CONDITION:
        return "Original D5E parsed output reference; no new inference."
    if condition == ORIGINAL_POST_FILTERED_CONDITION:
        return "D5E output after deterministic invalid-ID removal; no backfill and no new inference."
    if condition == STRICT_CONDITION:
        return "Strict candidate-ID-only prompt with 20 local Qwen2/Ollama requests."
    return "Strict prompt output after deterministic invalid-ID removal; no extra inference."


def _summary_row(condition_summary: pd.DataFrame, condition: str) -> pd.Series:
    rows = condition_summary.loc[condition_summary["condition"] == condition]
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row for {condition}, found {len(rows)}")
    return rows.iloc[0]


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


def _format_float(value: float) -> str:
    return f"{float(value):.6f}"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"

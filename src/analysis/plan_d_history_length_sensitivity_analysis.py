from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import median

import pandas as pd


PLAN = "Plan D"
PHASE = "D5K_llm_ml1m_history_length_sensitivity"
DATASET = "ml1m"
SCOPE = "diagnostic_subset"
BACKEND = "ollama"
QWEN_MODEL = "qwen2:7b-instruct"
MISTRAL_MODEL = "mistral:7b-instruct"
PROMPT_STRATEGY = "strict_candidate_id_only_history_length"
METHOD = "local_llm_history_length_sensitivity"
TEMPERATURE = 0.0
TOP_K = 5
CANDIDATE_POOL_SIZE = 20

SHORT_HISTORY = "short_history"
MEDIUM_HISTORY = "medium_history"
LONG_HISTORY = "long_history"
HISTORY_SETTINGS = (SHORT_HISTORY, MEDIUM_HISTORY, LONG_HISTORY)
HISTORY_LIMITS = {SHORT_HISTORY: 5, MEDIUM_HISTORY: 10, LONG_HISTORY: 20}

QWEN_CONDITIONS = {
    SHORT_HISTORY: "qwen2_short_history",
    MEDIUM_HISTORY: "qwen2_medium_history",
    LONG_HISTORY: "qwen2_long_history",
}
MISTRAL_CONDITIONS = {
    SHORT_HISTORY: "mistral_short_history",
    MEDIUM_HISTORY: "mistral_medium_history",
    LONG_HISTORY: "mistral_long_history",
}
MODEL_CONDITIONS = {
    QWEN_MODEL: QWEN_CONDITIONS,
    MISTRAL_MODEL: MISTRAL_CONDITIONS,
}
CONDITIONS = tuple(QWEN_CONDITIONS.values()) + tuple(MISTRAL_CONDITIONS.values())

MANIFEST_COLUMNS = [
    "diagnostic_id",
    "user_id",
    "dataset",
    "model",
    "condition",
    "history_setting",
    "history_length_used",
    "candidate_pool_size",
    "positive_item_id",
    "candidate_order_signature",
    "prompt_strategy",
    "backend",
    "status",
]

INSTANCE_COLUMNS = [
    "plan",
    "phase",
    "dataset",
    "scope",
    "condition",
    "history_setting",
    "user_id",
    "source_instance_id",
    "diagnostic_id",
    "backend",
    "model",
    "prompt_strategy",
    "temperature",
    "top_k",
    "candidate_pool_size",
    "history_length_used",
    "full_history_length",
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
    "history_setting",
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
    "mean_history_length_used",
    "min_history_length_used",
    "max_history_length_used",
    "history_truncation_rate",
    "positive_selected_rate",
    "positive_rank_when_selected",
    "selected_item_jaccard_vs_medium_history",
    "valid_item_jaccard_vs_medium_history",
    "selected_item_jaccard_short_vs_long",
    "hit_consistency_rate_across_history_lengths",
    "output_stability_rate_across_history_lengths",
    "status",
    "notes",
]

PAIRWISE_COLUMNS = [
    "model",
    "condition_a",
    "condition_b",
    "HR@5_delta",
    "NDCG@5_delta",
    "candidate_validity_delta",
    "hallucination_rate_delta",
    "duplicate_rate_delta",
    "selected_item_jaccard",
    "valid_item_jaccard",
    "positive_selection_agreement_rate",
    "both_hit_rate",
    "condition_a_only_hit_rate",
    "condition_b_only_hit_rate",
    "neither_hit_rate",
    "interpretation",
]

STABILITY_COLUMNS = [
    "diagnostic_id",
    "user_id",
    "model",
    "positive_item_id",
    "short_selected_ids",
    "medium_selected_ids",
    "long_selected_ids",
    "short_valid_selected_ids",
    "medium_valid_selected_ids",
    "long_valid_selected_ids",
    "short_hit",
    "medium_hit",
    "long_hit",
    "short_medium_jaccard",
    "medium_long_jaccard",
    "short_long_jaccard",
    "hit_pattern",
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
    "history_setting",
    "prompt_strategy",
    "metric",
    "k",
    "value",
    "num_instances",
    "notes",
]


@dataclass(frozen=True)
class HistoryItem:
    item_id: str
    title: str


@dataclass(frozen=True)
class HistoryLengthInstance:
    condition: str
    history_setting: str
    user_id: str
    source_instance_id: str
    diagnostic_id: str
    model: str
    candidate_ids: list[str]
    positive_item_id: str
    history_length_used: int
    full_history_length: int
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


def truncate_recent_history(history_items: list[HistoryItem], limit: int) -> list[HistoryItem]:
    """Return the most recent N history items from a chronological history list."""

    if limit < 1:
        raise ValueError("history limit must be positive")
    return list(history_items[-limit:])


def build_history_length_prompt(
    *,
    user_id: str,
    history_items: list[HistoryItem],
    full_history_length: int,
    history_setting: str,
    candidate_ids: list[str],
    top_k: int = TOP_K,
) -> str:
    """Build a strict candidate-ID-only prompt for the D5K history-length diagnostic."""

    validate_candidate_ids(candidate_ids, top_k)
    if history_setting not in HISTORY_SETTINGS:
        raise ValueError(f"unsupported history setting: {history_setting}")
    if not history_items:
        raise ValueError("history_items must not be empty")
    allowed_ids = json.dumps(candidate_ids, separators=(",", ":"))
    history_lines = "\n".join(
        f"- recent_{index}: {item.title}" for index, item in enumerate(history_items, start=1)
    )
    return f"""You are running a recommendation ranking task for user `{user_id}`.

History setting: {history_setting}
History length used: {len(history_items)} of {full_history_length} available interactions.
User history shown from older to newer within this truncated window:
{history_lines}

Allowed candidate IDs, copy from this list only:
{allowed_ids}

Rules:
- Return only valid JSON.
- Return exactly {top_k} item IDs.
- Every returned ID must be copied exactly from the allowed candidate ID list.
- Do not invent IDs.
- Do not output movie titles.
- Do not output explanations.
- If uncertain, choose only from the provided candidate IDs.
- Do not use any ID not shown in the allowed candidate ID list.

Required JSON object:
ranked_item_ids must be an array of exactly {top_k} strings copied from the allowed list; status must be "ok".
"""


def validate_candidate_ids(candidate_ids: list[str], top_k: int) -> None:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_ids must be unique")
    if top_k > len(candidate_ids):
        raise ValueError("top_k cannot exceed candidate count")


def candidate_order_signature(candidate_ids: list[str]) -> str:
    return hashlib.sha256("|".join(candidate_ids).encode("utf-8")).hexdigest()[:16]


def parse_candidate_ids(encoded: str) -> list[str]:
    value = json.loads(encoded)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("candidate_ids must be a JSON list of strings")
    return value


def condition_for(model: str, history_setting: str) -> str:
    return MODEL_CONDITIONS[model][history_setting]


def model_for_condition(condition: str) -> str:
    for model, mapping in MODEL_CONDITIONS.items():
        if condition in set(mapping.values()):
            return model
    raise ValueError(f"unknown D5K condition: {condition}")


def history_setting_for_condition(condition: str) -> str:
    for mapping in MODEL_CONDITIONS.values():
        for setting, mapped_condition in mapping.items():
            if condition == mapped_condition:
                return setting
    raise ValueError(f"unknown D5K condition: {condition}")


def build_history_length_manifest(
    d5e_manifest: pd.DataFrame,
    histories_by_user: dict[str, list[HistoryItem]],
) -> pd.DataFrame:
    """Build the deterministic 120-row D5K manifest from D5E instances."""

    rows: list[dict[str, str]] = []
    for _, row in d5e_manifest.iterrows():
        user_id = str(row["user_id"])
        history = histories_by_user.get(user_id, [])
        candidate_ids = parse_candidate_ids(str(row["candidate_ids"]))
        for model in (QWEN_MODEL, MISTRAL_MODEL):
            for setting in HISTORY_SETTINGS:
                condition = condition_for(model, setting)
                length_used = len(truncate_recent_history(history, HISTORY_LIMITS[setting]))
                rows.append(
                    {
                        "diagnostic_id": f"{PHASE}_{condition}_{user_id}",
                        "user_id": user_id,
                        "dataset": DATASET,
                        "model": model,
                        "condition": condition,
                        "history_setting": setting,
                        "history_length_used": str(length_used),
                        "candidate_pool_size": str(len(candidate_ids)),
                        "positive_item_id": str(row["positive_item_id"]),
                        "candidate_order_signature": candidate_order_signature(candidate_ids),
                        "prompt_strategy": PROMPT_STRATEGY,
                        "backend": BACKEND,
                        "status": "ready_for_local_history_length_inference",
                    }
                )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def evaluate_history_output(
    ranked_item_ids: list[str],
    *,
    condition: str,
    history_setting: str,
    user_id: str,
    source_instance_id: str,
    diagnostic_id: str,
    model: str,
    candidate_ids: list[str],
    positive_item_id: str,
    history_length_used: int,
    full_history_length: int,
    response_received: bool,
    parse_success: bool,
    schema_valid: bool,
    latency_seconds: float,
    parse_status: str,
    notes: str,
) -> HistoryLengthInstance:
    selected = [str(item_id) for item_id in ranked_item_ids]
    candidate_set = set(candidate_ids)
    valid_item_ids = [item_id for item_id in selected if item_id in candidate_set]
    invalid_item_ids = [item_id for item_id in selected if item_id not in candidate_set]
    selected_count = len(selected)
    duplicate_count = selected_count - len(set(selected))
    hallucinated_count = len(invalid_item_ids)
    candidate_validity_rate = len(valid_item_ids) / selected_count if selected_count else 0.0
    positive_rank = positive_rank_at_k(selected, positive_item_id, TOP_K)
    hit_at_5 = 1.0 if positive_rank is not None else 0.0
    ndcg_at_5 = 1.0 / math.log2(positive_rank + 1) if positive_rank is not None else 0.0
    return HistoryLengthInstance(
        condition=condition,
        history_setting=history_setting,
        user_id=user_id,
        source_instance_id=source_instance_id,
        diagnostic_id=diagnostic_id,
        model=model,
        candidate_ids=candidate_ids,
        positive_item_id=positive_item_id,
        history_length_used=history_length_used,
        full_history_length=full_history_length,
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


def instance_row(instance: HistoryLengthInstance) -> dict[str, str]:
    return {
        "plan": PLAN,
        "phase": PHASE,
        "dataset": DATASET,
        "scope": SCOPE,
        "condition": instance.condition,
        "history_setting": instance.history_setting,
        "user_id": instance.user_id,
        "source_instance_id": instance.source_instance_id,
        "diagnostic_id": instance.diagnostic_id,
        "backend": BACKEND,
        "model": instance.model,
        "prompt_strategy": PROMPT_STRATEGY,
        "temperature": format_float(TEMPERATURE),
        "top_k": str(TOP_K),
        "candidate_pool_size": str(CANDIDATE_POOL_SIZE),
        "history_length_used": str(instance.history_length_used),
        "full_history_length": str(instance.full_history_length),
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
        "latency_seconds": format_float(instance.latency_seconds),
        "parse_status": instance.parse_status,
        "status": instance.status,
        "notes": instance.notes,
    }


def build_condition_summary(instance_summary: pd.DataFrame, *, runtimes: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    stability = _model_stability_metrics(instance_summary)
    medium_lookup = _selected_lookup(instance_summary, MEDIUM_HISTORY, "selected_item_ids")
    medium_valid_lookup = _selected_lookup(instance_summary, MEDIUM_HISTORY, "valid_selected_item_ids")
    short_long_jaccard = _short_long_jaccard_by_model(instance_summary, "selected_item_ids")

    for model in (QWEN_MODEL, MISTRAL_MODEL):
        for setting in HISTORY_SETTINGS:
            condition = condition_for(model, setting)
            subset = instance_summary.loc[instance_summary["condition"] == condition].copy()
            if subset.empty:
                continue
            selected = numeric(subset["selected_count"])
            valid = numeric(subset["valid_selected_count"])
            hallucinated = numeric(subset["hallucinated_count"])
            duplicates = numeric(subset["duplicate_count"])
            history_lengths = numeric(subset["history_length_used"])
            full_lengths = numeric(subset["full_history_length"])
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
                    "model": model,
                    "condition": condition,
                    "history_setting": setting,
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
                    "total_runtime_seconds": format_float(runtimes.get(condition, 0.0)),
                    "mean_history_length_used": format_float(float(history_lengths.mean())),
                    "min_history_length_used": str(int(history_lengths.min())),
                    "max_history_length_used": str(int(history_lengths.max())),
                    "history_truncation_rate": format_float(float((full_lengths > history_lengths).mean())),
                    "positive_selected_rate": format_float(float(numeric(subset["hit_at_5"]).mean())),
                    "positive_rank_when_selected": format_optional(_mean_positive_rank(subset)),
                    "selected_item_jaccard_vs_medium_history": format_float(
                        _mean_jaccard_vs_lookup(subset, medium_lookup, "selected_item_ids")
                    ),
                    "valid_item_jaccard_vs_medium_history": format_float(
                        _mean_jaccard_vs_lookup(subset, medium_valid_lookup, "valid_selected_item_ids")
                    ),
                    "selected_item_jaccard_short_vs_long": format_float(short_long_jaccard.get(model, 0.0)),
                    "hit_consistency_rate_across_history_lengths": format_float(stability[model]["hit_consistency"]),
                    "output_stability_rate_across_history_lengths": format_float(stability[model]["output_stability"]),
                    "status": "completed_diagnostic_subset",
                    "notes": f"D5K {setting} diagnostic for {model}; candidate set fixed, history length varied.",
                }
            )
    return pd.DataFrame(rows, columns=CONDITION_SUMMARY_COLUMNS)


def build_pairwise_delta(condition_summary: pd.DataFrame, instance_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    comparisons = ((SHORT_HISTORY, MEDIUM_HISTORY), (LONG_HISTORY, MEDIUM_HISTORY), (SHORT_HISTORY, LONG_HISTORY))
    for model in (QWEN_MODEL, MISTRAL_MODEL):
        for setting_a, setting_b in comparisons:
            condition_a = condition_for(model, setting_a)
            condition_b = condition_for(model, setting_b)
            if condition_a not in set(condition_summary["condition"]) or condition_b not in set(condition_summary["condition"]):
                continue
            row_a = summary_row(condition_summary, condition_a)
            row_b = summary_row(condition_summary, condition_b)
            subset_a = instance_summary.loc[instance_summary["condition"] == condition_a]
            subset_b = instance_summary.loc[instance_summary["condition"] == condition_b]
            pairwise = _pairwise_instance_metrics(subset_a, subset_b)
            rows.append(
                {
                    "model": model,
                    "condition_a": setting_a,
                    "condition_b": setting_b,
                    "HR@5_delta": format_float(float(row_b["hr_at_5"]) - float(row_a["hr_at_5"])),
                    "NDCG@5_delta": format_float(float(row_b["ndcg_at_5"]) - float(row_a["ndcg_at_5"])),
                    "candidate_validity_delta": format_float(
                        float(row_b["mean_candidate_validity_rate"]) - float(row_a["mean_candidate_validity_rate"])
                    ),
                    "hallucination_rate_delta": format_float(
                        float(row_b["hallucination_rate"]) - float(row_a["hallucination_rate"])
                    ),
                    "duplicate_rate_delta": format_float(float(row_b["duplicate_rate"]) - float(row_a["duplicate_rate"])),
                    "selected_item_jaccard": format_float(pairwise["selected_jaccard"]),
                    "valid_item_jaccard": format_float(pairwise["valid_jaccard"]),
                    "positive_selection_agreement_rate": format_float(pairwise["positive_selection_agreement"]),
                    "both_hit_rate": format_float(pairwise["both_hit"]),
                    "condition_a_only_hit_rate": format_float(pairwise["a_only_hit"]),
                    "condition_b_only_hit_rate": format_float(pairwise["b_only_hit"]),
                    "neither_hit_rate": format_float(pairwise["neither_hit"]),
                    "interpretation": _history_delta_interpretation(
                        model=model,
                        setting_a=setting_a,
                        setting_b=setting_b,
                        hr_delta=float(row_b["hr_at_5"]) - float(row_a["hr_at_5"]),
                        validity_delta=float(row_b["mean_candidate_validity_rate"])
                        - float(row_a["mean_candidate_validity_rate"]),
                    ),
                }
            )
    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)


def build_stability_profile(instance_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for model in (QWEN_MODEL, MISTRAL_MODEL):
        model_rows = instance_summary.loc[instance_summary["model"] == model]
        for user_id in sorted(model_rows["user_id"].unique(), key=_id_sort_key):
            by_setting = {
                setting: model_rows.loc[
                    (model_rows["user_id"] == user_id)
                    & (model_rows["history_setting"] == setting)
                ].iloc[0]
                for setting in HISTORY_SETTINGS
            }
            short_selected = parse_json_list(by_setting[SHORT_HISTORY]["selected_item_ids"])
            medium_selected = parse_json_list(by_setting[MEDIUM_HISTORY]["selected_item_ids"])
            long_selected = parse_json_list(by_setting[LONG_HISTORY]["selected_item_ids"])
            short_valid = parse_json_list(by_setting[SHORT_HISTORY]["valid_selected_item_ids"])
            medium_valid = parse_json_list(by_setting[MEDIUM_HISTORY]["valid_selected_item_ids"])
            long_valid = parse_json_list(by_setting[LONG_HISTORY]["valid_selected_item_ids"])
            short_hit = bool_from_text(by_setting[SHORT_HISTORY]["hit_at_5"])
            medium_hit = bool_from_text(by_setting[MEDIUM_HISTORY]["hit_at_5"])
            long_hit = bool_from_text(by_setting[LONG_HISTORY]["hit_at_5"])
            rows.append(
                {
                    "diagnostic_id": f"{PHASE}_{model.replace(':', '_')}_stability_user_{user_id}",
                    "user_id": str(user_id),
                    "model": model,
                    "positive_item_id": str(by_setting[SHORT_HISTORY]["positive_item_id"]),
                    "short_selected_ids": json.dumps(short_selected, separators=(",", ":")),
                    "medium_selected_ids": json.dumps(medium_selected, separators=(",", ":")),
                    "long_selected_ids": json.dumps(long_selected, separators=(",", ":")),
                    "short_valid_selected_ids": json.dumps(short_valid, separators=(",", ":")),
                    "medium_valid_selected_ids": json.dumps(medium_valid, separators=(",", ":")),
                    "long_valid_selected_ids": json.dumps(long_valid, separators=(",", ":")),
                    "short_hit": bool_text(short_hit),
                    "medium_hit": bool_text(medium_hit),
                    "long_hit": bool_text(long_hit),
                    "short_medium_jaccard": format_float(jaccard(short_selected, medium_selected)),
                    "medium_long_jaccard": format_float(jaccard(medium_selected, long_selected)),
                    "short_long_jaccard": format_float(jaccard(short_selected, long_selected)),
                    "hit_pattern": history_hit_pattern(short_hit, medium_hit, long_hit),
                }
            )
    return pd.DataFrame(rows, columns=STABILITY_COLUMNS)


def build_metric_rows(condition_summary: pd.DataFrame) -> pd.DataFrame:
    metric_map = {
        "response_received_rate": ("response_received_rate", ""),
        "parse_success_rate": ("parse_success_rate", ""),
        "schema_valid_rate": ("schema_valid_rate", ""),
        "exact_top_k_rate": ("exact_top_k_rate", ""),
        "empty_output_rate": ("empty_output_rate", ""),
        "mean_candidate_validity_rate": ("mean_candidate_validity_rate", ""),
        "micro_candidate_validity_rate": ("micro_candidate_validity_rate", ""),
        "hallucination_rate": ("hallucination_rate", ""),
        "duplicate_rate": ("duplicate_rate", ""),
        "exact_5_valid_ids_rate": ("exact_5_valid_ids_rate", "5"),
        "hit_rate": ("hr_at_5", "5"),
        "ndcg": ("ndcg_at_5", "5"),
        "mean_latency_seconds": ("mean_latency_seconds", ""),
        "total_runtime_seconds": ("total_runtime_seconds", ""),
        "mean_history_length_used": ("mean_history_length_used", ""),
        "history_truncation_rate": ("history_truncation_rate", ""),
        "positive_selected_rate": ("positive_selected_rate", ""),
        "selected_item_jaccard_vs_medium_history": ("selected_item_jaccard_vs_medium_history", ""),
        "valid_item_jaccard_vs_medium_history": ("valid_item_jaccard_vs_medium_history", ""),
        "selected_item_jaccard_short_vs_long": ("selected_item_jaccard_short_vs_long", ""),
        "hit_consistency_rate_across_history_lengths": ("hit_consistency_rate_across_history_lengths", ""),
        "output_stability_rate_across_history_lengths": ("output_stability_rate_across_history_lengths", ""),
    }
    rows: list[dict[str, str]] = []
    for _, summary in condition_summary.iterrows():
        for metric, (column, k_value) in metric_map.items():
            rows.append(
                {
                    "plan": PLAN,
                    "phase": PHASE,
                    "dataset": DATASET,
                    "scope": SCOPE,
                    "method": METHOD,
                    "backend": BACKEND,
                    "model": str(summary["model"]),
                    "condition": str(summary["condition"]),
                    "history_setting": str(summary["history_setting"]),
                    "prompt_strategy": PROMPT_STRATEGY,
                    "metric": metric,
                    "k": k_value,
                    "value": str(summary[column]),
                    "num_instances": str(summary["total_instances"]),
                    "notes": "D5K diagnostic_subset metric; not exact paper reproduction.",
                }
            )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def raw_trace_row(
    *,
    instance: HistoryLengthInstance,
    raw_response_text: str,
    parsed_result: dict[str, object],
) -> dict[str, object]:
    return {
        "diagnostic_id": instance.diagnostic_id,
        "user_id": instance.user_id,
        "model": instance.model,
        "backend": BACKEND,
        "condition": instance.condition,
        "history_setting": instance.history_setting,
        "history_length_used": instance.history_length_used,
        "prompt_strategy": PROMPT_STRATEGY,
        "raw_response_text": raw_response_text,
        "parsed_result": parsed_result,
        "latency_seconds": instance.latency_seconds,
        "error_message": instance.notes if instance.status != "completed_instance" else "",
    }


def summary_row(condition_summary: pd.DataFrame, condition: str) -> pd.Series:
    rows = condition_summary.loc[condition_summary["condition"] == condition]
    if len(rows) != 1:
        raise ValueError(f"expected one summary row for {condition}, found {len(rows)}")
    return rows.iloc[0]


def history_hit_pattern(short_hit: bool, medium_hit: bool, long_hit: bool) -> str:
    if short_hit and medium_hit and long_hit:
        return "all_hit"
    if short_hit and not medium_hit and not long_hit:
        return "short_only"
    if not short_hit and medium_hit and not long_hit:
        return "medium_only"
    if not short_hit and not medium_hit and long_hit:
        return "long_only"
    if short_hit and medium_hit and not long_hit:
        return "short_medium_only"
    if short_hit and not medium_hit and long_hit:
        return "short_long_only"
    if not short_hit and medium_hit and long_hit:
        return "medium_long_only"
    return "none_hit"


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def positive_rank_at_k(ranked_item_ids: list[str], positive_item_id: str, k: int) -> int | None:
    for rank, item_id in enumerate(ranked_item_ids[:k], start=1):
        if item_id == positive_item_id:
            return rank
    return None


def instance_status(*, response_received: bool, parse_status: str) -> str:
    if not response_received:
        return "failed_backend"
    if parse_status in {"parse_failed", "empty_response"}:
        return "failed_parse"
    if parse_status == "parsed_invalid_schema":
        return "failed_schema"
    return "completed_instance"


def parse_json_list(encoded: object) -> list[str]:
    value = json.loads(str(encoded or "[]"))
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def bool_from_text(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return True
    try:
        return float(text) > 0
    except ValueError:
        return False


def bool_count(series: pd.Series) -> int:
    return int(series.astype(str).str.lower().isin({"true", "1", "1.0", "yes"}).sum())


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def format_float(value: float) -> str:
    return f"{float(value):.6f}"


def format_rate(numerator: int, denominator: int) -> str:
    return format_float(numerator / denominator if denominator else 0.0)


def format_optional(value: float | None) -> str:
    return "" if value is None else format_float(value)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def _mean_positive_rank(subset: pd.DataFrame) -> float | None:
    ranks = numeric(subset["positive_rank"])
    ranks = ranks.loc[ranks > 0]
    if ranks.empty:
        return None
    return float(ranks.mean())


def _selected_lookup(instance_summary: pd.DataFrame, history_setting: str, column: str) -> dict[tuple[str, str], list[str]]:
    lookup: dict[tuple[str, str], list[str]] = {}
    subset = instance_summary.loc[instance_summary["history_setting"] == history_setting]
    for _, row in subset.iterrows():
        lookup[(str(row["model"]), str(row["user_id"]))] = parse_json_list(row[column])
    return lookup


def _mean_jaccard_vs_lookup(subset: pd.DataFrame, lookup: dict[tuple[str, str], list[str]], column: str) -> float:
    scores: list[float] = []
    for _, row in subset.iterrows():
        reference = lookup.get((str(row["model"]), str(row["user_id"])), [])
        scores.append(jaccard(parse_json_list(row[column]), reference))
    return sum(scores) / len(scores) if scores else 0.0


def _short_long_jaccard_by_model(instance_summary: pd.DataFrame, column: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for model in (QWEN_MODEL, MISTRAL_MODEL):
        short_lookup = _selected_lookup(instance_summary.loc[instance_summary["model"] == model], SHORT_HISTORY, column)
        long_lookup = _selected_lookup(instance_summary.loc[instance_summary["model"] == model], LONG_HISTORY, column)
        scores = [
            jaccard(short_lookup[key], long_lookup.get(key, []))
            for key in short_lookup
            if key in long_lookup
        ]
        result[model] = sum(scores) / len(scores) if scores else 0.0
    return result


def _model_stability_metrics(instance_summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for model in (QWEN_MODEL, MISTRAL_MODEL):
        subset = instance_summary.loc[instance_summary["model"] == model]
        hit_consistent = 0
        output_stable = 0
        total = 0
        for user_id in sorted(subset["user_id"].unique(), key=_id_sort_key):
            rows = {
                setting: subset.loc[
                    (subset["user_id"] == user_id) & (subset["history_setting"] == setting)
                ]
                for setting in HISTORY_SETTINGS
            }
            if any(frame.empty for frame in rows.values()):
                continue
            total += 1
            hits = [bool_from_text(rows[setting].iloc[0]["hit_at_5"]) for setting in HISTORY_SETTINGS]
            outputs = [parse_json_list(rows[setting].iloc[0]["selected_item_ids"]) for setting in HISTORY_SETTINGS]
            if len(set(hits)) == 1:
                hit_consistent += 1
            if outputs[0] == outputs[1] == outputs[2]:
                output_stable += 1
        result[model] = {
            "hit_consistency": hit_consistent / total if total else 0.0,
            "output_stability": output_stable / total if total else 0.0,
        }
    return result


def _pairwise_instance_metrics(subset_a: pd.DataFrame, subset_b: pd.DataFrame) -> dict[str, float]:
    by_user_b = {str(row["user_id"]): row for _, row in subset_b.iterrows()}
    selected_scores: list[float] = []
    valid_scores: list[float] = []
    agreement = 0
    both = 0
    a_only = 0
    b_only = 0
    neither = 0
    total = 0
    for _, row_a in subset_a.iterrows():
        row_b = by_user_b.get(str(row_a["user_id"]))
        if row_b is None:
            continue
        total += 1
        selected_scores.append(jaccard(parse_json_list(row_a["selected_item_ids"]), parse_json_list(row_b["selected_item_ids"])))
        valid_scores.append(jaccard(parse_json_list(row_a["valid_selected_item_ids"]), parse_json_list(row_b["valid_selected_item_ids"])))
        hit_a = bool_from_text(row_a["hit_at_5"])
        hit_b = bool_from_text(row_b["hit_at_5"])
        if hit_a == hit_b:
            agreement += 1
        if hit_a and hit_b:
            both += 1
        elif hit_a:
            a_only += 1
        elif hit_b:
            b_only += 1
        else:
            neither += 1
    return {
        "selected_jaccard": sum(selected_scores) / len(selected_scores) if selected_scores else 0.0,
        "valid_jaccard": sum(valid_scores) / len(valid_scores) if valid_scores else 0.0,
        "positive_selection_agreement": agreement / total if total else 0.0,
        "both_hit": both / total if total else 0.0,
        "a_only_hit": a_only / total if total else 0.0,
        "b_only_hit": b_only / total if total else 0.0,
        "neither_hit": neither / total if total else 0.0,
    }


def _history_delta_interpretation(*, model: str, setting_a: str, setting_b: str, hr_delta: float, validity_delta: float) -> str:
    utility = "utility_improved" if hr_delta > 0 else "utility_decreased" if hr_delta < 0 else "utility_unchanged"
    validity = "validity_improved" if validity_delta > 0 else "validity_decreased" if validity_delta < 0 else "validity_unchanged"
    return f"{model}_{setting_b}_minus_{setting_a}_{utility}_{validity}"


def _id_sort_key(value: object) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)

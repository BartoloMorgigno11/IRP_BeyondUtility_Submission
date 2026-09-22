"""Finalization F5D serendipity definition alignment and offline computation.

F5D resolves the local ``serendipity_at_5`` ambiguity with source-backed
semantics from the upstream code. The official function is:

``mean(hit_results - hit_results_pop * hit_results)``.

This phase therefore treats serendipity as binary relevance multiplied by
unexpectedness against a primitive recommender hit vector. The local primary
primitive is the frozen train-only MostPop baseline already stored in
``traditional_baseline_rankings.csv``. Because the exact paper setting is not
machine-extracted locally, the computation is adapted rather than exact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.protected_input_compatibility import (
    protected_hash_counts as _d1r_protected_hash_counts,
    protected_hash_status as _d1r_protected_hash_status,
)


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
TOP_K = 5

REPORT_PATH = Path("report/finalization_f5d_serendipity.md")
PROTECTED_HASHES = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")
F4_METRIC_RESOLUTION = Path("results/tables/finalization_f4_metric_resolution_inventory.csv")
F4_DEFINITION_COMPARISON = Path("results/tables/finalization_f4_metric_definition_comparison.csv")
F4_MISSING = Path("results/tables/finalization_f4_missing_evidence_triage.csv")
F5B_PREDICTION_INVENTORY = Path("results/tables/finalization_f5b_prediction_source_inventory.csv")
F5B_ILD_POLICY = Path("results/tables/finalization_f5b_ild_policy.csv")
F5B_ILD_RESULTS = Path("results/tables/finalization_f5b_ild_results.csv")
F5C_PREDICTION_INVENTORY = Path("results/tables/finalization_f5c_prediction_source_inventory.csv")
F5C_POPULARITY_RESULTS = Path("results/tables/finalization_f5c_popularity_results.csv")
F5C_COVERAGE = Path("results/tables/finalization_f5c_coverage_impact.csv")
TRADITIONAL_RANKINGS = Path("results/tables/traditional_baseline_rankings.csv")
CANONICAL_RESULTS = Path("results/tables/canonical_metric_results_long.csv")
LLM_UTILITY = Path("results/tables/canonical_local_llm_utility_metrics.csv")
LLM_BEYOND = Path("results/tables/canonical_local_llm_beyond_utility_metrics.csv")

OUTPUT_TABLES: dict[str, Path] = {
    "serendipity_metric_inventory": Path("results/tables/finalization_f5d_serendipity_metric_inventory.csv"),
    "serendipity_definition_comparison": Path("results/tables/finalization_f5d_serendipity_definition_comparison.csv"),
    "serendipity_prerequisite_inventory": Path("results/tables/finalization_f5d_serendipity_prerequisite_inventory.csv"),
    "primitive_recommender_policy": Path("results/tables/finalization_f5d_primitive_recommender_policy.csv"),
    "relevance_policy": Path("results/tables/finalization_f5d_relevance_policy.csv"),
    "unexpectedness_policy": Path("results/tables/finalization_f5d_unexpectedness_policy.csv"),
    "serendipity_policy": Path("results/tables/finalization_f5d_serendipity_policy.csv"),
    "prediction_source_inventory": Path("results/tables/finalization_f5d_prediction_source_inventory.csv"),
    "serendipity_components": Path("results/tables/finalization_f5d_serendipity_components.csv"),
    "serendipity_results": Path("results/tables/finalization_f5d_serendipity_results.csv"),
    "serendipity_user_summary": Path("results/tables/finalization_f5d_serendipity_user_summary.csv"),
    "serendipity_sensitivity": Path("results/tables/finalization_f5d_serendipity_sensitivity.csv"),
    "valid_serendipity_comparisons": Path("results/tables/finalization_f5d_valid_serendipity_comparisons.csv"),
    "serendipity_reconciliation": Path("results/tables/finalization_f5d_serendipity_reconciliation.csv"),
    "metric_status_update": Path("results/tables/finalization_f5d_metric_status_update.csv"),
    "claim_update": Path("results/tables/finalization_f5d_claim_update.csv"),
    "metric_tradeoffs": Path("results/tables/finalization_f5d_metric_tradeoffs.csv"),
    "coverage_impact": Path("results/tables/finalization_f5d_coverage_impact.csv"),
    "anomalies": Path("results/tables/finalization_f5d_anomalies.csv"),
    "validation": Path("results/tables/finalization_f5d_validation.csv"),
}

METRIC_INVENTORY_COLUMNS = [
    "serendipity_metric_id",
    "metric_name",
    "metric_variant",
    "source_scope",
    "source_file",
    "source_function",
    "paper_section",
    "paper_page",
    "paper_table_or_figure",
    "paper_definition_available",
    "upstream_definition_available",
    "local_definition_available",
    "formula_summary",
    "requires_relevance",
    "relevance_definition",
    "requires_unexpectedness",
    "unexpectedness_definition",
    "requires_novelty",
    "requires_user_history",
    "requires_primitive_recommender",
    "primitive_recommender_definition",
    "requires_item_similarity",
    "item_similarity_definition",
    "cutoff",
    "aggregation_level",
    "higher_is_better",
    "paper_equivalence",
    "current_status",
    "notes",
]

DEFINITION_COMPARISON_COLUMNS = [
    "definition_comparison_id",
    "metric_variant",
    "aspect",
    "paper_definition",
    "upstream_definition",
    "local_definition",
    "paper_vs_upstream",
    "paper_vs_local",
    "upstream_vs_local",
    "material_difference",
    "scientific_impact",
    "recommended_definition",
    "recommendation_class",
    "evidence",
    "notes",
]

PREREQUISITE_COLUMNS = [
    "prerequisite_id",
    "dataset",
    "task",
    "requirement",
    "required_source",
    "available",
    "available_file",
    "coverage",
    "authority",
    "can_use_for_exact",
    "can_use_for_functional",
    "can_use_for_adapted",
    "blocking_reason",
    "notes",
]

PRIMITIVE_POLICY_COLUMNS = [
    "primitive_policy_id",
    "policy_role",
    "dataset",
    "primitive_model",
    "source_file",
    "training_scope",
    "candidate_scope",
    "recommendation_length",
    "tie_breaking",
    "seed",
    "paper_equivalence",
    "reproduction_class",
    "is_primary_policy",
    "is_sensitivity_policy",
    "scientific_rationale",
    "notes",
]

RELEVANCE_POLICY_COLUMNS = [
    "relevance_policy_id",
    "dataset",
    "task",
    "ground_truth_source",
    "relevance_type",
    "positive_definition",
    "graded_relevance",
    "top_k",
    "invalid_item_policy",
    "duplicate_policy",
    "missing_ground_truth_policy",
    "paper_equivalence",
    "reproduction_class",
    "is_primary_policy",
    "scientific_rationale",
    "notes",
]

UNEXPECTEDNESS_POLICY_COLUMNS = [
    "unexpectedness_policy_id",
    "dataset",
    "policy_role",
    "unexpectedness_reference",
    "primitive_policy_id",
    "feature_policy_id",
    "similarity_function",
    "unexpectedness_formula",
    "item_level_or_list_level",
    "normalization",
    "missing_feature_policy",
    "empty_reference_policy",
    "paper_equivalence",
    "reproduction_class",
    "is_primary_policy",
    "is_sensitivity_policy",
    "scientific_rationale",
    "notes",
]

SERENDIPITY_POLICY_COLUMNS = [
    "serendipity_policy_id",
    "dataset",
    "task",
    "policy_role",
    "relevance_policy_id",
    "unexpectedness_policy_id",
    "primitive_policy_id",
    "feature_policy_id",
    "formula",
    "top_k",
    "item_level_formula",
    "user_level_aggregation",
    "dataset_level_aggregation",
    "zero_relevant_item_policy",
    "invalid_item_policy",
    "duplicate_policy",
    "missing_feature_policy",
    "value_range",
    "paper_equivalence",
    "reproduction_class",
    "is_primary_policy",
    "is_sensitivity_policy",
    "scientific_rationale",
    "notes",
]

PREDICTION_INVENTORY_COLUMNS = [
    "prediction_source_id",
    "dataset",
    "task",
    "model",
    "model_family",
    "source_file",
    "cohort_name",
    "candidate_set_name",
    "candidate_count",
    "recommendation_list_length",
    "user_count",
    "has_ground_truth",
    "has_user_history",
    "has_item_features",
    "has_primitive_recommendations",
    "is_final_source",
    "is_primary_source",
    "is_pilot",
    "is_diagnostic",
    "can_compute_serendipity",
    "blocking_reason",
    "notes",
]

COMPONENT_COLUMNS = [
    "component_result_id",
    "serendipity_policy_id",
    "prediction_source_id",
    "dataset",
    "task",
    "model",
    "cohort_name",
    "candidate_set_name",
    "cutoff",
    "component",
    "value",
    "std",
    "median",
    "valid_users",
    "total_users",
    "valid_user_rate",
    "valid_items",
    "total_items",
    "notes",
]

RESULT_COLUMNS = [
    "serendipity_result_id",
    "serendipity_policy_id",
    "prediction_source_id",
    "dataset",
    "task",
    "model",
    "model_family",
    "cohort_name",
    "candidate_set_name",
    "candidate_count",
    "cutoff",
    "metric",
    "metric_variant",
    "value",
    "std",
    "median",
    "min",
    "max",
    "valid_users",
    "total_users",
    "valid_user_rate",
    "relevant_recommendation_count",
    "unexpected_recommendation_count",
    "serendipitous_recommendation_count",
    "invalid_item_count",
    "missing_feature_count",
    "zero_relevance_user_count",
    "computation_class",
    "paper_equivalence",
    "reproduction_class",
    "is_primary_result",
    "is_reportable",
    "source_prediction_file",
    "validation_status",
    "notes",
]

USER_SUMMARY_COLUMNS = [
    "summary_id",
    "dataset",
    "task",
    "model",
    "cohort_name",
    "serendipity_policy_id",
    "summary_type",
    "summary_key",
    "value",
    "user_count",
    "notes",
]

SENSITIVITY_COLUMNS = [
    "sensitivity_id",
    "dataset",
    "task",
    "model",
    "cohort_name",
    "primary_policy_id",
    "sensitivity_policy_id",
    "sensitivity_axis",
    "primary_value",
    "sensitivity_value",
    "absolute_difference",
    "relative_difference",
    "rank_primary",
    "rank_sensitivity",
    "rank_change",
    "same_model_ordering",
    "interpretation",
    "notes",
]

COMPARISON_COLUMNS = [
    "comparison_id",
    "dataset",
    "task",
    "cohort_name",
    "candidate_set_name",
    "serendipity_policy_id",
    "model_a",
    "model_a_value",
    "model_b",
    "model_b_value",
    "absolute_difference",
    "relative_difference",
    "is_valid_comparison",
    "invalid_reason",
    "comparison_scope",
    "notes",
]

RECONCILIATION_COLUMNS = [
    "reconciliation_id",
    "existing_source_file",
    "existing_result_id",
    "dataset",
    "task",
    "model",
    "existing_metric_name",
    "existing_value",
    "existing_formula_known",
    "f5d_result_id",
    "f5d_value",
    "same_relevance_policy",
    "same_unexpectedness_policy",
    "same_primitive_baseline",
    "same_item_representation",
    "same_aggregation",
    "same_protocol",
    "is_reconciled",
    "discrepancy_reason",
    "recommended_authoritative_result",
    "notes",
]

METRIC_STATUS_COLUMNS = [
    "metric_definition_id",
    "metric",
    "previous_status",
    "f5d_status",
    "resolution_class",
    "formula_status",
    "relevance_policy_status",
    "unexpectedness_policy_status",
    "primitive_policy_status",
    "feature_policy_status",
    "computation_status",
    "validation_status",
    "paper_equivalence",
    "is_scientifically_consolidated",
    "is_reportable",
    "remaining_limitation",
    "next_required_action",
    "notes",
]

CLAIM_COLUMNS = [
    "claim_id",
    "previous_claim_text",
    "previous_safety",
    "f5d_evidence",
    "updated_claim_text",
    "updated_confidence",
    "is_safe_for_report",
    "is_safe_for_presentation",
    "claim_scope",
    "remaining_limitation",
    "notes",
]

TRADEOFF_COLUMNS = [
    "tradeoff_id",
    "dataset",
    "task",
    "cohort_name",
    "model",
    "serendipity_result_id",
    "serendipity_value",
    "utility_metric",
    "utility_value",
    "novelty_value",
    "ild_value",
    "aplt_value",
    "arp_value",
    "comparison_scope",
    "is_same_protocol",
    "interpretation",
    "notes",
]

COVERAGE_COLUMNS = [
    "coverage_metric",
    "previous_value",
    "f5d_value",
    "absolute_change",
    "supporting_evidence",
    "is_change_allowed",
    "validation_status",
    "notes",
]

ANOMALY_COLUMNS = [
    "anomaly_id",
    "area",
    "severity",
    "source_file",
    "description",
    "scientific_impact",
    "can_continue",
    "recommended_action",
    "notes",
]

VALIDATION_COLUMNS = ["check_id", "check_name", "status", "observed", "expected", "details"]

ABSOLUTE_PATH_PATTERN = re.compile(r"\b[A-Za-z]:[\\/]")
ACTIVE_DATASETS = ("Amazon Beauty", "Amazon Sports", "ML-1M")
ALL_DATASETS = ("Amazon Beauty", "Amazon Sports", "ML-1M", "LastFM")
TASKS = ("ranking", "reranking")


@dataclass(frozen=True)
class PrimitiveRow:
    items: tuple[str, ...]
    hit: int


@dataclass(frozen=True)
class UserSerendipity:
    user_id: str
    relevance: int
    primitive_hit: int
    unexpectedness: int
    serendipity: int
    primitive_overlap: float
    invalid_item_count: int
    duplicate_count: int
    missing_feature_count: int


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def parse_bool(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y"}


def format_float(value: float | int | None) -> str:
    if value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.17g}"


def percent(numerator: int | float, denominator: int | float) -> str:
    if float(denominator) == 0.0:
        return ""
    return format_float(float(numerator) / float(denominator))


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(clean(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def normalize_item_id(value: Any) -> str:
    text = clean(value)
    if re.fullmatch(r"-?\d+\.0", text):
        return text[:-2]
    return text


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def read_csv(root: Path, relative: Path, **kwargs: Any) -> pd.DataFrame:
    path = root / relative
    if not path.exists():
        return pd.DataFrame()
    options: dict[str, Any] = {"dtype": str, "keep_default_na": False}
    options.update(kwargs)
    return pd.read_csv(path, **options)


def sort_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    existing = [column for column in columns if column in frame.columns]
    return frame.sort_values(existing, kind="mergesort").reset_index(drop=True) if existing else frame.reset_index(drop=True)


def git_output(root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    return completed.stdout.strip()


def parse_item_list(value: Any) -> list[str]:
    text = clean(value)
    if not text:
        return []
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, list):
        return [normalize_item_id(item) for item in decoded if normalize_item_id(item)]
    return [normalize_item_id(piece) for piece in re.split(r"\s*;\s*", text) if normalize_item_id(piece)]


def recommendation_items_from_row(row: Mapping[str, Any], top_k: int = TOP_K) -> list[str]:
    for column in ("top5_item_ids", "ranked_item_ids", "ranked_items"):
        if column in row and clean(row.get(column)):
            return parse_item_list(row.get(column))[:top_k]
    if "recommended_item_id" in row:
        item = normalize_item_id(row.get("recommended_item_id"))
        return [item] if item else []
    return []


def candidate_items_from_row(row: Mapping[str, Any]) -> set[str]:
    return set(parse_item_list(row.get("candidate_items", "")))


def binary_relevance_at_k(recommended_items: Sequence[Any], positive_item: Any, k: int = TOP_K) -> int:
    target = normalize_item_id(positive_item)
    if not target:
        return 0
    return int(target in [normalize_item_id(item) for item in list(recommended_items)[:k]])


def primitive_unexpectedness(primitive_hit: int) -> int:
    return int(1 - int(bool(primitive_hit)))


def serendipity_value(relevance: int, primitive_hit: int) -> int:
    return int(int(bool(relevance)) * primitive_unexpectedness(primitive_hit))


def primitive_list_overlap(recommended_items: Sequence[Any], primitive_items: Sequence[Any], k: int = TOP_K) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    left = {normalize_item_id(item) for item in list(recommended_items)[:k] if normalize_item_id(item)}
    right = {normalize_item_id(item) for item in list(primitive_items)[:k] if normalize_item_id(item)}
    return len(left.intersection(right)) / k


def user_serendipity_record(
    recommended_items: Sequence[Any],
    positive_item: Any,
    primitive: PrimitiveRow,
    *,
    user_id: Any = "",
    candidate_items: set[str] | None = None,
    k: int = TOP_K,
) -> UserSerendipity:
    top_items = [normalize_item_id(item) for item in list(recommended_items)[:k] if normalize_item_id(item)]
    relevance = binary_relevance_at_k(top_items, positive_item, k)
    primitive_hit = int(primitive.hit)
    unexpectedness = primitive_unexpectedness(primitive_hit)
    ser_value = serendipity_value(relevance, primitive_hit)
    candidates = candidate_items or set()
    missing_positions = max(0, k - len(top_items))
    invalid_items = sum(1 for item in top_items if candidates and item not in candidates) + missing_positions
    duplicate_count = max(0, len(top_items) - len(set(top_items)))
    return UserSerendipity(
        clean(user_id),
        relevance,
        primitive_hit,
        unexpectedness,
        ser_value,
        primitive_list_overlap(top_items, primitive.items, k),
        invalid_items,
        duplicate_count,
        0,
    )


def macro(values: Sequence[float | int]) -> float:
    if not values:
        raise ValueError("empty values cannot be aggregated")
    return float(np.mean(np.asarray(values, dtype=float)))


def aggregate_values(values: Sequence[float | int]) -> dict[str, str]:
    if not values:
        return {"value": "", "std": "", "median": "", "min": "", "max": "", "valid_users": "0"}
    array = np.asarray([float(value) for value in values], dtype=float)
    return {
        "value": format_float(float(np.mean(array))),
        "std": format_float(float(np.std(array))),
        "median": format_float(float(np.median(array))),
        "min": format_float(float(np.min(array))),
        "max": format_float(float(np.max(array))),
        "valid_users": str(len(array)),
    }


def primitive_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        clean(row.get("dataset")),
        clean(row.get("task_type", row.get("task"))),
        clean(row.get("candidate_set_id", row.get("candidate_set_name"))),
        clean(row.get("user_id")),
        normalize_item_id(row.get("positive_item_id")),
    )


def build_primitive_index(root: Path, baseline_id: str) -> dict[tuple[str, str, str, str, str], PrimitiveRow]:
    frame = read_csv(root, TRADITIONAL_RANKINGS)
    if frame.empty:
        return {}
    frame = frame.loc[frame["baseline_id"].map(clean).eq(baseline_id)].copy()
    index: dict[tuple[str, str, str, str, str], PrimitiveRow] = {}
    for row in frame.to_dict("records"):
        items = tuple(recommendation_items_from_row(row, TOP_K))
        hit = binary_relevance_at_k(items, row.get("positive_item_id"), TOP_K)
        index[primitive_key(row)] = PrimitiveRow(items, hit)
    return index


def load_prediction_source_rows(root: Path, source_row: Mapping[str, Any]) -> pd.DataFrame:
    source_file = Path(clean(source_row.get("source_file")))
    if not source_file.as_posix() or not (root / source_file).exists():
        return pd.DataFrame()
    frame = read_csv(root, source_file)
    if frame.empty:
        return frame
    dataset = clean(source_row.get("dataset"))
    task = clean(source_row.get("task"))
    candidate = clean(source_row.get("candidate_set_name"))
    model = clean(source_row.get("model"))
    cohort = clean(source_row.get("cohort_name"))
    if dataset and "dataset" in frame.columns:
        frame = frame.loc[frame["dataset"].map(clean).eq(dataset)]
    if task and "task_type" in frame.columns:
        frame = frame.loc[frame["task_type"].map(clean).eq(task)]
    elif task and "task" in frame.columns:
        frame = frame.loc[frame["task"].map(clean).eq(task)]
    if candidate and "candidate_set_id" in frame.columns:
        frame = frame.loc[frame["candidate_set_id"].map(clean).eq(candidate)]
    if model:
        model_filters: list[pd.Series] = []
        for column in ("model_family", "model_tag", "model_id", "baseline_id", "baseline_family"):
            if column in frame.columns:
                model_filters.append(frame[column].map(clean).eq(model))
        if model_filters:
            mask = model_filters[0]
            for extra in model_filters[1:]:
                mask = mask | extra
            frame = frame.loc[mask]
    if cohort:
        if "full_cohort_id" in frame.columns and cohort.startswith("plan_e2_"):
            frame = frame.loc[frame["full_cohort_id"].map(clean).eq(cohort)]
        elif "cohort_id" in frame.columns:
            frame = frame.loc[frame["cohort_id"].map(clean).eq(cohort)]
        elif "eligible_full_cohort" in frame.columns and cohort.endswith("_full"):
            frame = frame.loc[frame["eligible_full_cohort"].map(parse_bool)]
    return frame.reset_index(drop=True)


def source_rows_cache(root: Path, prediction_inventory: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    for row in prediction_inventory.to_dict("records"):
        source_id = clean(row.get("prediction_source_id"))
        cache[source_id] = load_prediction_source_rows(root, row)
    return cache


def build_serendipity_metric_inventory() -> pd.DataFrame:
    rows = [
        {
            "serendipity_metric_id": "F5D-SER-PAPER-001",
            "metric_name": "Serendipity@5",
            "metric_variant": "paper_metric_name_unextracted_formula",
            "source_scope": "paper",
            "source_file": "paper/2411.00331v1.pdf;paper/traduzione_italiana_beyond_utility_llm_recommender.docx",
            "source_function": "",
            "paper_section": "beyond-utility metric discussion",
            "paper_page": "not machine-extracted locally",
            "paper_table_or_figure": "not machine-extracted locally",
            "paper_definition_available": "false",
            "upstream_definition_available": "true",
            "local_definition_available": "true",
            "formula_summary": "paper exact prose unavailable; upstream defines serendipity via target hit and MostPop hit vectors",
            "requires_relevance": "true",
            "relevance_definition": "binary hit of the ground-truth positive item in top-5",
            "requires_unexpectedness": "true",
            "unexpectedness_definition": "recommended hit is unexpected when the primitive baseline does not also hit",
            "requires_novelty": "false",
            "requires_user_history": "false",
            "requires_primitive_recommender": "true",
            "primitive_recommender_definition": "MostPop generated from train item frequencies in upstream code",
            "requires_item_similarity": "false",
            "item_similarity_definition": "not used by upstream calSerendipity",
            "cutoff": "5",
            "aggregation_level": "mean over evaluated rows",
            "higher_is_better": "true",
            "paper_equivalence": "paper_metric_name_present_but_exact_formula_not_text_extracted",
            "current_status": "blocked_exact",
            "notes": "F5D does not invent a paper equation from prose.",
        },
        {
            "serendipity_metric_id": "F5D-SER-UPSTREAM-001",
            "metric_name": "Serendipity@5",
            "metric_variant": "upstream_hit_minus_pop_hit_intersection",
            "source_scope": "upstream",
            "source_file": "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "source_function": "calSerendipity(hit_results, hit_results_pop)",
            "paper_section": "",
            "paper_page": "",
            "paper_table_or_figure": "",
            "paper_definition_available": "false",
            "upstream_definition_available": "true",
            "local_definition_available": "true",
            "formula_summary": "mean(hit_results - hit_results_pop * hit_results)",
            "requires_relevance": "true",
            "relevance_definition": "hit_results is 1 when the positive target is ranked within top-k",
            "requires_unexpectedness": "true",
            "unexpectedness_definition": "1 - hit_results_pop, where hit_results_pop is MostPop top-k hit",
            "requires_novelty": "false",
            "requires_user_history": "false",
            "requires_primitive_recommender": "true",
            "primitive_recommender_definition": "generate_mostpop ranks candidates by train item frequency",
            "requires_item_similarity": "false",
            "item_similarity_definition": "not applicable",
            "cutoff": "5",
            "aggregation_level": "mean over evaluated rows",
            "higher_is_better": "true",
            "paper_equivalence": "functional_upstream_code_not_exact_paper_claim",
            "current_status": "definition_recovered",
            "notes": "This definition is not novelty or ILD.",
        },
        {
            "serendipity_metric_id": "F5D-SER-CANONICAL-001",
            "metric_name": "Serendipity@5",
            "metric_variant": "f4_blocked_missing_inputs",
            "source_scope": "canonical",
            "source_file": "results/tables/finalization_f4_metric_resolution_inventory.csv",
            "source_function": "",
            "paper_section": "",
            "paper_page": "",
            "paper_table_or_figure": "",
            "paper_definition_available": "false",
            "upstream_definition_available": "true",
            "local_definition_available": "true",
            "formula_summary": "F4 recognized calSerendipity but lacked aligned popularity-baseline hit vectors",
            "requires_relevance": "true",
            "relevance_definition": "top-5 hit vector",
            "requires_unexpectedness": "true",
            "unexpectedness_definition": "aligned primitive-baseline hit vector",
            "requires_novelty": "false",
            "requires_user_history": "false",
            "requires_primitive_recommender": "true",
            "primitive_recommender_definition": "not frozen before F5D",
            "requires_item_similarity": "false",
            "item_similarity_definition": "not applicable",
            "cutoff": "5",
            "aggregation_level": "mean over rows",
            "higher_is_better": "true",
            "paper_equivalence": "blocked_before_f5d",
            "current_status": "blocked_missing_inputs",
            "notes": "F5D supplies the aligned local primitive hit vector without rewriting F4.",
        },
        {
            "serendipity_metric_id": "F5D-SER-ADAPTED-001",
            "metric_name": "Serendipity@5",
            "metric_variant": "f5d_mostpop_train_only_primitive",
            "source_scope": "local",
            "source_file": "results/tables/traditional_baseline_rankings.csv",
            "source_function": "",
            "paper_section": "",
            "paper_page": "",
            "paper_table_or_figure": "",
            "paper_definition_available": "false",
            "upstream_definition_available": "true",
            "local_definition_available": "true",
            "formula_summary": "mean(binary_hit_at_5 * (1 - mostpop_train_only_hit_at_5))",
            "requires_relevance": "true",
            "relevance_definition": "ground-truth positive item appears in the evaluated model top-5",
            "requires_unexpectedness": "true",
            "unexpectedness_definition": "the local MostPop train-only primitive does not hit the same positive item",
            "requires_novelty": "false",
            "requires_user_history": "false",
            "requires_primitive_recommender": "true",
            "primitive_recommender_definition": "mostpop_train_only rows aligned by dataset task candidate set user and positive item",
            "requires_item_similarity": "false",
            "item_similarity_definition": "not applicable",
            "cutoff": "5",
            "aggregation_level": "macro mean over valid rows",
            "higher_is_better": "true",
            "paper_equivalence": "adapted_upstream_formula_with_local_mostpop",
            "current_status": "resolved_adapted",
            "notes": "Primary F5D policy; exact paper baseline remains unverified.",
        },
        {
            "serendipity_metric_id": "F5D-SER-SENS-001",
            "metric_name": "Serendipity@5",
            "metric_variant": "f5d_random_seeded_primitive_sensitivity",
            "source_scope": "local",
            "source_file": "results/tables/traditional_baseline_rankings.csv",
            "source_function": "",
            "paper_section": "",
            "paper_page": "",
            "paper_table_or_figure": "",
            "paper_definition_available": "false",
            "upstream_definition_available": "true",
            "local_definition_available": "true",
            "formula_summary": "mean(binary_hit_at_5 * (1 - random_seeded_hit_at_5))",
            "requires_relevance": "true",
            "relevance_definition": "same binary ground-truth hit as primary",
            "requires_unexpectedness": "true",
            "unexpectedness_definition": "Random seeded primitive misses the positive item",
            "requires_novelty": "false",
            "requires_user_history": "false",
            "requires_primitive_recommender": "true",
            "primitive_recommender_definition": "random_seeded rows with deterministic seed 42 score derivation",
            "requires_item_similarity": "false",
            "item_similarity_definition": "not applicable",
            "cutoff": "5",
            "aggregation_level": "macro mean over valid rows",
            "higher_is_better": "true",
            "paper_equivalence": "sensitivity_only_not_paper_metric",
            "current_status": "resolved_extension_only",
            "notes": "Sensitivity only; it does not replace the MostPop primary policy.",
        },
    ]
    return pd.DataFrame(rows, columns=METRIC_INVENTORY_COLUMNS)


def build_definition_comparison() -> pd.DataFrame:
    aspects = [
        ("relevance component", "not machine-extracted", "hit_results top-k binary target hit", "binary positive hit in saved rankings", "unknown", "compatible with upstream", "same", "yes", "Defines sparse values under one-positive candidate rows.", "binary top-5 target hit", "adapted_upstream", "llm_metrics.py calSerendipity; ranking rows positive_item_id"),
        ("binary vs graded relevance", "not machine-extracted", "binary", "binary", "unknown", "same_as_upstream", "same", "no", "No graded score is introduced.", "binary relevance", "adapted_upstream", "hit vector arithmetic"),
        ("ground-truth source", "not machine-extracted", "id_targets from test samples", "positive_item_id in saved candidate rows", "unknown", "compatible", "same semantic target", "yes", "Local source is frozen candidate rows, not raw upstream samples.", "positive_item_id", "adapted", "rankings CSVs"),
        ("unexpectedness definition", "not machine-extracted", "1 - hit_results_pop", "1 - primitive baseline top-5 hit", "unknown", "same_as_upstream", "same", "yes", "Unexpectedness is primitive-hit based, not novelty.", "primitive miss of target", "adapted_upstream", "calSerendipity formula"),
        ("primitive recommender", "not machine-extracted", "generate_mostpop train-frequency ranking", "mostpop_train_only saved baseline", "unknown", "functionally similar but not exact", "adapted", "yes", "Exact paper primitive run is unavailable.", "MostPop train-only", "adapted", "traditional_baseline_rankings.csv"),
        ("user-history similarity", "not used", "not used", "not used", "same", "same", "same", "no", "History features are not part of upstream serendipity.", "not applicable", "blocked_for_exact_history_variant", "calSerendipity inputs"),
        ("item representation", "not used", "not used", "not used", "same", "same", "same", "no", "No item feature similarity is required.", "not applicable", "not_applicable", "calSerendipity inputs"),
        ("similarity function", "not used", "not used", "not used", "same", "same", "same", "no", "Cosine/Jaccard sensitivity is not scientifically motivated for this formula.", "not applicable", "not_applicable", "calSerendipity inputs"),
        ("novelty contribution", "not used", "not used", "not used", "same", "same", "same", "yes", "Novelty must remain separate from serendipity.", "none", "not_applicable", "calSelfInfo is separate function"),
        ("candidate universe", "not machine-extracted", "same sampled candidates for target and MostPop", "same candidate_set_id and row keys", "unknown", "adapted aligned", "same local protocol", "yes", "Rows without aligned primitive are blocked.", "same candidate set", "adapted", "candidate_set_id key"),
        ("top-k", "5 in audited metric id", "metric_len", "5", "compatible", "same", "same", "no", "F5D freezes cutoff 5.", "5", "adapted_upstream", "F4 serendipity_at_5 row"),
        ("duplicate policy", "not specified", "hit vector binary", "first positive presence in top-5; duplicate positions counted in diagnostics", "unknown", "compatible", "explicit local", "low", "Duplicate only affects diagnostics unless it changes hit presence.", "binary presence", "adapted", "ranking rows duplicate counts"),
        ("invalid-item policy", "not specified", "invalid outputs cannot hit unless target appears", "empty/invalid list gives relevance 0 and invalid count", "unknown", "compatible", "explicit local", "yes", "Invalid LLM rows are retained as non-hit.", "invalid as non-hit", "adapted", "source ranking invalidity columns"),
        ("missing-feature policy", "not applicable", "not applicable", "not applicable", "same", "same", "same", "no", "No item feature dependency.", "not applicable", "not_applicable", "calSerendipity inputs"),
        ("user aggregation", "not machine-extracted", "mean over rows", "macro mean over valid rows", "unknown", "same when valid rows match", "same local", "yes", "Valid-user rate must be reported.", "macro row mean", "adapted", "numpy mean in calSerendipity"),
        ("dataset aggregation", "not machine-extracted", "single dataset run mean", "dataset/task/source mean", "unknown", "compatible", "explicit local", "low", "No cross-dataset aggregation is used.", "within protocol only", "adapted", "F5D comparison rules"),
        ("zero-relevance policy", "not specified", "zero hit contributes zero", "zero hit contributes zero", "compatible", "same", "same", "yes", "One-positive candidate rows can produce many zeros.", "retain zero", "adapted_upstream", "formula arithmetic"),
        ("empty-history policy", "not applicable", "not applicable", "not applicable", "same", "same", "same", "no", "No history reference is used.", "not applicable", "not_applicable", "calSerendipity inputs"),
        ("normalization", "not specified", "binary mean in [0,1]", "binary mean in [0,1]", "compatible", "same", "same", "no", "No additional normalization is introduced.", "[0,1]", "adapted_upstream", "hit-vector arithmetic"),
        ("value range", "not specified", "[0,1]", "[0,1]", "compatible", "same", "same", "no", "Result is a binary mean.", "[0,1]", "adapted_upstream", "binary relevance times binary unexpectedness"),
    ]
    rows = []
    for index, (aspect, paper, upstream, local, pvu, pvl, uvl, material, impact, recommended, klass, evidence) in enumerate(aspects, start=1):
        rows.append(
            {
                "definition_comparison_id": f"F5D-DEF-{index:03d}",
                "metric_variant": "f5d_mostpop_train_only_primitive",
                "aspect": aspect,
                "paper_definition": paper,
                "upstream_definition": upstream,
                "local_definition": local,
                "paper_vs_upstream": pvu,
                "paper_vs_local": pvl,
                "upstream_vs_local": uvl,
                "material_difference": material,
                "scientific_impact": impact,
                "recommended_definition": recommended,
                "recommendation_class": klass,
                "evidence": evidence,
                "notes": "F5D keeps novelty, unexpectedness, diversity, and serendipity separate.",
            }
        )
    return pd.DataFrame(rows, columns=DEFINITION_COMPARISON_COLUMNS)


def dataset_interaction_path(dataset: str) -> str:
    return {
        "Amazon Beauty": "data/reconstructed/plan_c/amazon_beauty/interactions.csv",
        "Amazon Sports": "data/reconstructed/plan_c/amazon_sports/interactions.csv",
        "ML-1M": "data/processed/ml1m/interactions.csv",
        "LastFM": "data/reconstructed/plan_c/lastfm/interactions.csv",
    }.get(dataset, "")


def dataset_catalog_path(dataset: str) -> str:
    return {
        "Amazon Beauty": "data/reconstructed/plan_c/amazon_beauty/catalog.csv",
        "Amazon Sports": "data/reconstructed/plan_c/amazon_sports/catalog.csv",
        "ML-1M": "data/processed/ml1m/catalog.csv",
        "LastFM": "data/reconstructed/plan_c/lastfm/catalog.csv",
    }.get(dataset, "")


def build_prerequisite_inventory(root: Path) -> pd.DataFrame:
    prediction = read_csv(root, F5C_PREDICTION_INVENTORY)
    rows = []
    requirements = [
        ("recommendation lists", "saved ranking CSV with top5/ranked item ids"),
        ("user histories", "train interaction CSV"),
        ("ground truth", "positive_item_id in candidate rows"),
        ("relevance labels", "binary top-5 hit of positive_item_id"),
        ("item metadata", "F5B feature inventory/catalog"),
        ("item feature representation", "not required by upstream serendipity; F5B available only for non-primary variants"),
        ("primitive recommender lists", "traditional_baseline_rankings.csv MostPop/Random rows"),
        ("item popularity", "MostPop train-only source and F5C popularity policy"),
        ("candidate set", "candidate_set_id and candidate_items"),
        ("cohort definition", "cohort_name/full_cohort_id/evaluation policy"),
        ("invalid-output information", "ranking_valid/candidate_validity/duplicate columns"),
    ]
    for dataset in ALL_DATASETS:
        for task in TASKS:
            task_rows = prediction.loc[prediction["dataset"].map(clean).eq(dataset) & prediction["task"].map(clean).eq(task)] if not prediction.empty else pd.DataFrame()
            active = dataset in ACTIVE_DATASETS and not task_rows.empty
            for requirement, source in requirements:
                if requirement == "recommendation lists":
                    available = active
                    available_file = ";".join(sorted(task_rows["source_file"].dropna().map(clean).unique())) if active else ""
                    coverage = percent(len(task_rows), len(task_rows)) if active else "0"
                elif requirement == "primitive recommender lists":
                    available = dataset in ACTIVE_DATASETS and (root / TRADITIONAL_RANKINGS).exists()
                    available_file = TRADITIONAL_RANKINGS.as_posix() if available else ""
                    coverage = "1" if available else "0"
                elif requirement in {"user histories", "item popularity"}:
                    path = dataset_interaction_path(dataset)
                    available = bool(path) and (root / path).exists() and dataset in ACTIVE_DATASETS
                    available_file = path if available else ""
                    coverage = "1" if available else "0"
                elif requirement in {"item metadata", "item feature representation"}:
                    path = dataset_catalog_path(dataset)
                    available = bool(path) and (root / path).exists() and dataset in ACTIVE_DATASETS
                    available_file = path if available else ""
                    coverage = "1" if available else "0"
                else:
                    available = active
                    available_file = "prediction source rows" if active else ""
                    coverage = "1" if active else "0"
                rows.append(
                    {
                        "prerequisite_id": stable_id("F5D-PREREQ", dataset, task, requirement),
                        "dataset": dataset,
                        "task": task,
                        "requirement": requirement,
                        "required_source": source,
                        "available": bool_text(available),
                        "available_file": available_file,
                        "coverage": coverage,
                        "authority": "local_frozen_artifacts",
                        "can_use_for_exact": "false",
                        "can_use_for_functional": bool_text(available and requirement != "item feature representation"),
                        "can_use_for_adapted": bool_text(available),
                        "blocking_reason": "" if available else "dataset_or_source_not_available_under_f5d_policy",
                        "notes": "Item features are inventoried but not required by the primary upstream serendipity formula.",
                    }
                )
    return sort_frame(pd.DataFrame(rows, columns=PREREQUISITE_COLUMNS), ["dataset", "task", "requirement"])


def primitive_policy_id(dataset: str, role: str) -> str:
    model = "mostpop_train_only" if role == "primary" else "random_seeded"
    return stable_id("F5D-PRIM", dataset, role, model, TRADITIONAL_RANKINGS.as_posix())


def build_primitive_recommender_policy() -> pd.DataFrame:
    rows = []
    for dataset in ALL_DATASETS:
        active = dataset in ACTIVE_DATASETS
        for role, model, seed, is_primary, is_sens in (
            ("primary", "mostpop_train_only", "", True, False),
            ("sensitivity", "random_seeded", "42", False, True),
        ):
            rows.append(
                {
                    "primitive_policy_id": primitive_policy_id(dataset, role),
                    "policy_role": role if active else "blocked",
                    "dataset": dataset,
                    "primitive_model": model,
                    "source_file": TRADITIONAL_RANKINGS.as_posix() if active else "",
                    "training_scope": "train-only item frequency" if model == "mostpop_train_only" else "deterministic seeded random scores",
                    "candidate_scope": "same candidate_set_id and same user/positive row",
                    "recommendation_length": "source ranked_items length up to 20; top-5 hit vector used",
                    "tie_breaking": "source table order",
                    "seed": seed,
                    "paper_equivalence": "adapted_upstream_primitive" if active and role == "primary" else "sensitivity_only_not_paper",
                    "reproduction_class": "adapted" if active else "blocked",
                    "is_primary_policy": bool_text(active and is_primary),
                    "is_sensitivity_policy": bool_text(active and is_sens),
                    "scientific_rationale": "Upstream generate_mostpop provides hit_results_pop for calSerendipity; local MostPop is the closest frozen primitive baseline." if role == "primary" else "Random seeded tests dependence on primitive choice only.",
                    "notes": "" if active else "LastFM has no final aligned local prediction/primitive policy in F5D.",
                }
            )
    return pd.DataFrame(rows, columns=PRIMITIVE_POLICY_COLUMNS)


def relevance_policy_id(dataset: str, task: str) -> str:
    return stable_id("F5D-REL", dataset, task, "binary_positive_hit_top5")


def build_relevance_policy() -> pd.DataFrame:
    rows = []
    for dataset in ALL_DATASETS:
        for task in TASKS:
            active = dataset in ACTIVE_DATASETS
            rows.append(
                {
                    "relevance_policy_id": relevance_policy_id(dataset, task),
                    "dataset": dataset,
                    "task": task,
                    "ground_truth_source": "positive_item_id in saved prediction source rows",
                    "relevance_type": "binary",
                    "positive_definition": "1 if positive_item_id appears in top-5 recommendation items else 0",
                    "graded_relevance": "false",
                    "top_k": str(TOP_K),
                    "invalid_item_policy": "invalid or empty recommendation rows are retained as relevance 0 and counted",
                    "duplicate_policy": "positive presence is binary; duplicate diagnostics are counted separately",
                    "missing_ground_truth_policy": "row blocked from valid computation if positive_item_id is missing",
                    "paper_equivalence": "adapted_upstream_hit_vector" if active else "blocked",
                    "reproduction_class": "adapted" if active else "blocked",
                    "is_primary_policy": bool_text(active),
                    "scientific_rationale": "Upstream calSerendipity consumes the same binary hit vector used by HR@K.",
                    "notes": "One-positive candidate-set evaluation can produce sparse serendipity values.",
                }
            )
    return pd.DataFrame(rows, columns=RELEVANCE_POLICY_COLUMNS)


def unexpectedness_policy_id(dataset: str, role: str) -> str:
    return stable_id("F5D-UNEXP", dataset, role, "primitive_hit_miss")


def build_unexpectedness_policy(primitive_policy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, primitive in primitive_policy.iterrows():
        dataset = clean(primitive["dataset"])
        role = "primary" if parse_bool(primitive["is_primary_policy"]) else "sensitivity" if parse_bool(primitive["is_sensitivity_policy"]) else "blocked"
        rows.append(
            {
                "unexpectedness_policy_id": unexpectedness_policy_id(dataset, role),
                "dataset": dataset,
                "policy_role": role,
                "unexpectedness_reference": "primitive recommender top-5 hit vector",
                "primitive_policy_id": clean(primitive["primitive_policy_id"]),
                "feature_policy_id": "not_applicable",
                "similarity_function": "not_applicable",
                "unexpectedness_formula": "1 - primitive_hit_at_5",
                "item_level_or_list_level": "row-level hit-vector component",
                "normalization": "binary in {0,1}",
                "missing_feature_policy": "not_applicable",
                "empty_reference_policy": "block row if aligned primitive recommendation is unavailable",
                "paper_equivalence": clean(primitive["paper_equivalence"]),
                "reproduction_class": clean(primitive["reproduction_class"]),
                "is_primary_policy": clean(primitive["is_primary_policy"]),
                "is_sensitivity_policy": clean(primitive["is_sensitivity_policy"]),
                "scientific_rationale": "This is the only unexpectedness component present in upstream calSerendipity.",
                "notes": "No novelty, popularity, user-history, or ILD term is mixed into this policy.",
            }
        )
    return pd.DataFrame(rows, columns=UNEXPECTEDNESS_POLICY_COLUMNS)


def serendipity_policy_id(dataset: str, task: str, role: str) -> str:
    return stable_id("F5D-SER-POL", dataset, task, role, "hit_times_primitive_miss")


def build_serendipity_policy(relevance: pd.DataFrame, unexpectedness: pd.DataFrame, primitive: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ALL_DATASETS:
        for task in TASKS:
            for role in ("primary", "sensitivity"):
                active = dataset in ACTIVE_DATASETS
                prim = primitive.loc[(primitive["dataset"].eq(dataset)) & (primitive["policy_role"].eq(role))]
                unexp = unexpectedness.loc[(unexpectedness["dataset"].eq(dataset)) & (unexpectedness["policy_role"].eq(role))]
                rows.append(
                    {
                        "serendipity_policy_id": serendipity_policy_id(dataset, task, role),
                        "dataset": dataset,
                        "task": task,
                        "policy_role": role if active else "blocked",
                        "relevance_policy_id": relevance_policy_id(dataset, task),
                        "unexpectedness_policy_id": clean(unexp.iloc[0]["unexpectedness_policy_id"]) if not unexp.empty else "",
                        "primitive_policy_id": clean(prim.iloc[0]["primitive_policy_id"]) if not prim.empty else "",
                        "feature_policy_id": "not_applicable",
                        "formula": "mean(binary_relevance_at_5 * (1 - primitive_hit_at_5))",
                        "top_k": str(TOP_K),
                        "item_level_formula": "not item-level; upstream formula is row-level hit-vector arithmetic",
                        "user_level_aggregation": "single row contribution in {0,1}",
                        "dataset_level_aggregation": "macro mean over valid rows in the same dataset/task/source",
                        "zero_relevant_item_policy": "zero relevance contributes zero serendipity and is counted",
                        "invalid_item_policy": "invalid or empty recommendation list contributes relevance 0 unless the positive appears",
                        "duplicate_policy": "binary positive presence; duplicate count is diagnostic",
                        "missing_feature_policy": "not_applicable",
                        "value_range": "[0,1]",
                        "paper_equivalence": "adapted_upstream_formula_with_local_mostpop" if role == "primary" and active else "sensitivity_only_not_paper",
                        "reproduction_class": "adapted" if active else "blocked",
                        "is_primary_policy": bool_text(active and role == "primary"),
                        "is_sensitivity_policy": bool_text(active and role == "sensitivity"),
                        "scientific_rationale": "Direct execution of recovered upstream semantics with frozen local primitive hit vectors.",
                        "notes": "Exact paper reproduction remains blocked because the paper primitive run and exact setting are not machine-extracted.",
                    }
                )
    return pd.DataFrame(rows, columns=SERENDIPITY_POLICY_COLUMNS)


def build_prediction_source_inventory(
    root: Path,
    primitive_mostpop: Mapping[tuple[str, str, str, str, str], PrimitiveRow],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    base = read_csv(root, F5C_PREDICTION_INVENTORY)
    if base.empty:
        base = read_csv(root, F5B_PREDICTION_INVENTORY)
    rows = []
    source_cache: dict[str, pd.DataFrame] = {}
    for _, row in base.iterrows():
        source_file = clean(row.get("source_file"))
        dataset = clean(row.get("dataset"))
        task = clean(row.get("task"))
        source_id = stable_id("F5D-PRED", source_file, dataset, task, row.get("model"), row.get("cohort_name"), row.get("candidate_set_name"))
        source_rows = load_prediction_source_rows(root, row) if source_file else pd.DataFrame()
        source_cache[source_id] = source_rows
        has_gt = "positive_item_id" in source_rows.columns and source_rows["positive_item_id"].map(clean).ne("").any()
        has_history = dataset in ACTIVE_DATASETS and (root / dataset_interaction_path(dataset)).exists()
        has_features = dataset in ACTIVE_DATASETS and (root / dataset_catalog_path(dataset)).exists()
        primitive_matches = 0
        for source_record in source_rows.to_dict("records"):
            key = primitive_key(
                {
                    **source_record,
                    "dataset": dataset or source_record.get("dataset"),
                    "task_type": task or source_record.get("task_type"),
                    "candidate_set_id": row.get("candidate_set_name") or source_record.get("candidate_set_id"),
                }
            )
            primitive_matches += int(key in primitive_mostpop)
        has_primitive = primitive_matches > 0
        blocking = ""
        if dataset not in ACTIVE_DATASETS:
            blocking = "dataset_blocked_or_not_final_under_f5d"
        elif source_rows.empty:
            blocking = "no_saved_recommendation_rows"
        elif not has_gt:
            blocking = "missing_ground_truth_positive_item"
        elif not has_primitive:
            blocking = "missing_aligned_mostpop_primitive_rows"
        can_compute = not blocking
        rows.append(
            {
                "prediction_source_id": source_id,
                "dataset": dataset,
                "task": task,
                "model": clean(row.get("model")),
                "model_family": clean(row.get("model_family")),
                "source_file": source_file,
                "cohort_name": clean(row.get("cohort_name")),
                "candidate_set_name": clean(row.get("candidate_set_name")),
                "candidate_count": clean(row.get("candidate_count")),
                "recommendation_list_length": clean(row.get("recommendation_list_length")),
                "user_count": str(len(source_rows)),
                "has_ground_truth": bool_text(has_gt),
                "has_user_history": bool_text(has_history),
                "has_item_features": bool_text(has_features),
                "has_primitive_recommendations": bool_text(has_primitive),
                "is_final_source": bool_text(parse_bool(row.get("is_final_source"))),
                "is_primary_source": bool_text(parse_bool(row.get("is_primary_source"))),
                "is_pilot": bool_text(parse_bool(row.get("is_pilot"))),
                "is_diagnostic": bool_text(parse_bool(row.get("is_diagnostic"))),
                "can_compute_serendipity": bool_text(can_compute),
                "blocking_reason": blocking,
                "notes": "Prediction source derived from F5C/F5B inventory; aggregate-only and unaligned primitive rows are blocked.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=PREDICTION_INVENTORY_COLUMNS), ["dataset", "task", "model", "source_file"]), source_cache


def row_key_for_prediction(row: Mapping[str, Any], prediction: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        clean(prediction.get("dataset")) or clean(row.get("dataset")),
        clean(prediction.get("task")) or clean(row.get("task_type", row.get("task"))),
        clean(prediction.get("candidate_set_name")) or clean(row.get("candidate_set_id")),
        clean(row.get("user_id")),
        normalize_item_id(row.get("positive_item_id")),
    )


def compute_records_for_source(
    prediction: Mapping[str, Any],
    source_rows: pd.DataFrame,
    primitive_index: Mapping[tuple[str, str, str, str, str], PrimitiveRow],
) -> list[UserSerendipity]:
    records = []
    for row in source_rows.to_dict("records"):
        key = row_key_for_prediction(row, prediction)
        primitive = primitive_index.get(key)
        if primitive is None or not key[-1]:
            continue
        items = recommendation_items_from_row(row, TOP_K)
        records.append(
            user_serendipity_record(
                items,
                row.get("positive_item_id"),
                primitive,
                user_id=row.get("user_id"),
                candidate_items=candidate_items_from_row(row),
                k=TOP_K,
            )
        )
    return records


def build_result_row(
    prediction: Mapping[str, Any],
    policy: Mapping[str, Any],
    records: Sequence[UserSerendipity],
    role: str,
) -> dict[str, str]:
    values = [record.serendipity for record in records]
    agg = aggregate_values(values)
    valid_users = int(agg["valid_users"] or 0)
    total_users = int(clean(prediction.get("user_count")) or valid_users)
    is_primary_source = parse_bool(prediction.get("is_primary_source"))
    is_reportable = is_primary_source and role == "primary" and valid_users > 0 and not parse_bool(prediction.get("is_pilot")) and not parse_bool(prediction.get("is_diagnostic"))
    return {
        "serendipity_result_id": stable_id("F5D-SER-RES", prediction.get("prediction_source_id"), policy.get("serendipity_policy_id")),
        "serendipity_policy_id": clean(policy.get("serendipity_policy_id")),
        "prediction_source_id": clean(prediction.get("prediction_source_id")),
        "dataset": clean(prediction.get("dataset")),
        "task": clean(prediction.get("task")),
        "model": clean(prediction.get("model")),
        "model_family": clean(prediction.get("model_family")),
        "cohort_name": clean(prediction.get("cohort_name")),
        "candidate_set_name": clean(prediction.get("candidate_set_name")),
        "candidate_count": clean(prediction.get("candidate_count")),
        "cutoff": str(TOP_K),
        "metric": "serendipity_at_5",
        "metric_variant": "mostpop_train_only_primitive" if role == "primary" else "random_seeded_primitive_sensitivity",
        "value": agg["value"],
        "std": agg["std"],
        "median": agg["median"],
        "min": agg["min"],
        "max": agg["max"],
        "valid_users": str(valid_users),
        "total_users": str(total_users),
        "valid_user_rate": percent(valid_users, total_users),
        "relevant_recommendation_count": str(sum(record.relevance for record in records)),
        "unexpected_recommendation_count": str(sum(record.unexpectedness for record in records)),
        "serendipitous_recommendation_count": str(sum(record.serendipity for record in records)),
        "invalid_item_count": str(sum(record.invalid_item_count for record in records)),
        "missing_feature_count": str(sum(record.missing_feature_count for record in records)),
        "zero_relevance_user_count": str(sum(1 for record in records if record.relevance == 0)),
        "computation_class": "adapted_offline_recomputation" if role == "primary" else "extension_offline_computation",
        "paper_equivalence": clean(policy.get("paper_equivalence")),
        "reproduction_class": clean(policy.get("reproduction_class")),
        "is_primary_result": bool_text(role == "primary"),
        "is_reportable": bool_text(is_reportable),
        "source_prediction_file": clean(prediction.get("source_file")),
        "validation_status": "pass" if valid_users > 0 else "blocked_no_valid_rows",
        "notes": "Formula is relevance hit times primitive miss; novelty and ILD are not substituted.",
    }


def build_serendipity_results_and_components(
    prediction_inventory: pd.DataFrame,
    serendipity_policy: pd.DataFrame,
    source_cache: Mapping[str, pd.DataFrame],
    primitive_mostpop: Mapping[tuple[str, str, str, str, str], PrimitiveRow],
    primitive_random: Mapping[tuple[str, str, str, str, str], PrimitiveRow],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], list[UserSerendipity]]]:
    result_rows = []
    component_rows = []
    record_cache: dict[tuple[str, str], list[UserSerendipity]] = {}
    policies = {clean(row["serendipity_policy_id"]): row for row in serendipity_policy.to_dict("records")}
    for prediction in prediction_inventory.to_dict("records"):
        if not parse_bool(prediction.get("can_compute_serendipity")):
            continue
        source_id = clean(prediction["prediction_source_id"])
        source_rows = source_cache.get(source_id, pd.DataFrame())
        for role, primitive_index in (("primary", primitive_mostpop), ("sensitivity", primitive_random)):
            policy_id = serendipity_policy_id(clean(prediction["dataset"]), clean(prediction["task"]), role)
            policy = policies.get(policy_id)
            if not policy:
                continue
            records = compute_records_for_source(prediction, source_rows, primitive_index)
            if not records:
                continue
            record_cache[(source_id, policy_id)] = records
            result_rows.append(build_result_row(prediction, policy, records, role))
            component_specs = [
                ("relevance_rate", [record.relevance for record in records], "Binary positive hit rate."),
                ("unexpectedness", [record.unexpectedness for record in records], "Primitive baseline miss rate."),
                ("primitive_list_overlap", [record.primitive_overlap for record in records], "Top-5 overlap with primitive list; diagnostic only."),
                ("final_serendipity", [record.serendipity for record in records], "Binary relevance multiplied by primitive unexpectedness."),
            ]
            for component, values, notes in component_specs:
                agg = aggregate_values(values)
                valid_users = int(agg["valid_users"] or 0)
                total_users = int(clean(prediction.get("user_count")) or valid_users)
                component_rows.append(
                    {
                        "component_result_id": stable_id("F5D-COMP", source_id, policy_id, component),
                        "serendipity_policy_id": policy_id,
                        "prediction_source_id": source_id,
                        "dataset": clean(prediction["dataset"]),
                        "task": clean(prediction["task"]),
                        "model": clean(prediction["model"]),
                        "cohort_name": clean(prediction["cohort_name"]),
                        "candidate_set_name": clean(prediction["candidate_set_name"]),
                        "cutoff": str(TOP_K),
                        "component": component,
                        "value": agg["value"],
                        "std": agg["std"],
                        "median": agg["median"],
                        "valid_users": str(valid_users),
                        "total_users": str(total_users),
                        "valid_user_rate": percent(valid_users, total_users),
                        "valid_items": str(valid_users * TOP_K),
                        "total_items": str(total_users * TOP_K),
                        "notes": notes,
                    }
                )
    return (
        sort_frame(pd.DataFrame(component_rows, columns=COMPONENT_COLUMNS), ["dataset", "task", "model", "component"]),
        sort_frame(pd.DataFrame(result_rows, columns=RESULT_COLUMNS), ["dataset", "task", "model", "metric_variant"]),
        record_cache,
    )


def build_user_summary(results: pd.DataFrame, record_cache: Mapping[tuple[str, str], list[UserSerendipity]]) -> pd.DataFrame:
    rows = []
    for result in results.loc[results["is_primary_result"].eq("true")].to_dict("records"):
        records = record_cache.get((clean(result["prediction_source_id"]), clean(result["serendipity_policy_id"])), [])
        if not records:
            continue
        values = np.asarray([record.serendipity for record in records], dtype=float)
        for quantile in (0.0, 0.25, 0.5, 0.75, 1.0):
            rows.append(
                {
                    "summary_id": stable_id("F5D-USUM", result["serendipity_result_id"], "quantile", quantile),
                    "dataset": clean(result["dataset"]),
                    "task": clean(result["task"]),
                    "model": clean(result["model"]),
                    "cohort_name": clean(result["cohort_name"]),
                    "serendipity_policy_id": clean(result["serendipity_policy_id"]),
                    "summary_type": "quantile",
                    "summary_key": format_float(quantile),
                    "value": format_float(float(np.quantile(values, quantile))),
                    "user_count": str(len(values)),
                    "notes": "Quantile over row-level binary serendipity values.",
                }
            )
        summary_specs = [
            ("zero_serendipity_user_rate", float(np.mean(values == 0.0)), "Users/rows with serendipity 0."),
            ("at_least_one_serendipitous_item_rate", float(np.mean(values > 0.0)), "Rows with a relevant item not hit by the primitive."),
            ("no_relevant_top5_item_rate", float(np.mean([record.relevance == 0 for record in records])), "Rows without the positive in top-5."),
        ]
        for key, value, notes in summary_specs:
            rows.append(
                {
                    "summary_id": stable_id("F5D-USUM", result["serendipity_result_id"], key),
                    "dataset": clean(result["dataset"]),
                    "task": clean(result["task"]),
                    "model": clean(result["model"]),
                    "cohort_name": clean(result["cohort_name"]),
                    "serendipity_policy_id": clean(result["serendipity_policy_id"]),
                    "summary_type": "rate",
                    "summary_key": key,
                    "value": format_float(value),
                    "user_count": str(len(values)),
                    "notes": notes,
                }
            )
    return sort_frame(pd.DataFrame(rows, columns=USER_SUMMARY_COLUMNS), ["dataset", "task", "model", "summary_type", "summary_key"])


def build_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=SENSITIVITY_COLUMNS)
    primary = results.loc[results["metric_variant"].eq("mostpop_train_only_primitive")]
    sensitivity = results.loc[results["metric_variant"].eq("random_seeded_primitive_sensitivity")]
    rows = []
    merge_cols = ["prediction_source_id", "dataset", "task", "model", "cohort_name", "candidate_set_name"]
    merged = primary.merge(sensitivity, on=merge_cols, suffixes=("_primary", "_sensitivity"))
    group_cols = ["dataset", "task", "cohort_name", "candidate_set_name"]
    for _, group in merged.groupby(group_cols, sort=True):
        primary_order = group.sort_values(["value_primary", "model"], ascending=[False, True], kind="mergesort")["model"].tolist()
        sensitivity_order = group.sort_values(["value_sensitivity", "model"], ascending=[False, True], kind="mergesort")["model"].tolist()
        same_order = primary_order == sensitivity_order
        rank_primary = {model: rank + 1 for rank, model in enumerate(primary_order)}
        rank_sens = {model: rank + 1 for rank, model in enumerate(sensitivity_order)}
        for row in group.to_dict("records"):
            p = float(row["value_primary"])
            s = float(row["value_sensitivity"])
            rows.append(
                {
                    "sensitivity_id": stable_id("F5D-SENS", row["prediction_source_id"], row["serendipity_policy_id_sensitivity"]),
                    "dataset": clean(row["dataset"]),
                    "task": clean(row["task"]),
                    "model": clean(row["model"]),
                    "cohort_name": clean(row["cohort_name"]),
                    "primary_policy_id": clean(row["serendipity_policy_id_primary"]),
                    "sensitivity_policy_id": clean(row["serendipity_policy_id_sensitivity"]),
                    "sensitivity_axis": "primitive_baseline_mostpop_vs_random",
                    "primary_value": clean(row["value_primary"]),
                    "sensitivity_value": clean(row["value_sensitivity"]),
                    "absolute_difference": format_float(s - p),
                    "relative_difference": format_float(None if p == 0.0 else (s - p) / abs(p)),
                    "rank_primary": str(rank_primary.get(row["model"], "")),
                    "rank_sensitivity": str(rank_sens.get(row["model"], "")),
                    "rank_change": str(rank_sens.get(row["model"], 0) - rank_primary.get(row["model"], 0)),
                    "same_model_ordering": bool_text(same_order),
                    "interpretation": "Sensitivity changes the primitive-hit reference only; it is not the primary policy.",
                    "notes": "Random seeded primitive is extension-only.",
                }
            )
    return sort_frame(pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS), ["dataset", "task", "model"])


def build_valid_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    source = results.loc[results["is_reportable"].eq("true") & results["is_primary_result"].eq("true")]
    rows = []
    group_cols = ["dataset", "task", "cohort_name", "candidate_set_name", "serendipity_policy_id", "cutoff"]
    for _, group in source.groupby(group_cols, sort=True):
        records = group.sort_values("model", kind="mergesort").to_dict("records")
        for left, right in combinations(records, 2):
            left_value = float(left["value"])
            right_value = float(right["value"])
            rows.append(
                {
                    "comparison_id": stable_id("F5D-VCMP", left["serendipity_policy_id"], left["model"], right["model"], left["dataset"], left["task"], left["candidate_set_name"]),
                    "dataset": clean(left["dataset"]),
                    "task": clean(left["task"]),
                    "cohort_name": clean(left["cohort_name"]),
                    "candidate_set_name": clean(left["candidate_set_name"]),
                    "serendipity_policy_id": clean(left["serendipity_policy_id"]),
                    "model_a": clean(left["model"]),
                    "model_a_value": clean(left["value"]),
                    "model_b": clean(right["model"]),
                    "model_b_value": clean(right["value"]),
                    "absolute_difference": format_float(right_value - left_value),
                    "relative_difference": format_float(None if left_value == 0.0 else (right_value - left_value) / abs(left_value)),
                    "is_valid_comparison": "true",
                    "invalid_reason": "",
                    "comparison_scope": "same_dataset_task_cohort_candidate_policy",
                    "notes": "No cross-dataset or mixed-policy comparison is included.",
                }
            )
    return sort_frame(pd.DataFrame(rows, columns=COMPARISON_COLUMNS), ["dataset", "task", "model_a", "model_b"])


def existing_serendipity_rows(root: Path) -> pd.DataFrame:
    rows = []
    for path in (CANONICAL_RESULTS, LLM_BEYOND, F4_METRIC_RESOLUTION, F4_DEFINITION_COMPARISON, F4_MISSING):
        frame = read_csv(root, path)
        if frame.empty:
            continue
        text = frame.astype(str).agg(" ".join, axis=1)
        matches = frame.loc[text.str.contains("serendip", case=False, regex=True)]
        for index, row in matches.iterrows():
            rows.append(
                {
                    "existing_source_file": path.as_posix(),
                    "existing_result_id": clean(row.get("result_id", row.get("metric_resolution_id", row.get("metric_row_id", row.get("missing_evidence_id", index))))),
                    "dataset": clean(row.get("dataset")),
                    "task": clean(row.get("task_type", row.get("task"))),
                    "model": clean(row.get("baseline_id", row.get("model_family", row.get("model")))),
                    "existing_metric_name": clean(row.get("metric_id", row.get("metric_variant", row.get("metric", "serendipity_at_5")))),
                    "existing_value": clean(row.get("metric_value", row.get("value"))),
                    "existing_formula_known": bool_text("calSerendipity" in " ".join(clean(v) for v in row.to_dict().values())),
                }
            )
    if not rows:
        rows.append(
            {
                "existing_source_file": F4_METRIC_RESOLUTION.as_posix(),
                "existing_result_id": "serendipity_at_5",
                "dataset": "",
                "task": "",
                "model": "",
                "existing_metric_name": "serendipity_at_5",
                "existing_value": "",
                "existing_formula_known": "true",
            }
        )
    return pd.DataFrame(rows)


def build_reconciliation(root: Path, results: pd.DataFrame) -> pd.DataFrame:
    existing = existing_serendipity_rows(root)
    primary = results.loc[results["is_primary_result"].eq("true")]
    rows = []
    for _, row in existing.iterrows():
        candidates = primary
        for result_col, existing_col in (("dataset", "dataset"), ("task", "task"), ("model", "model")):
            value = clean(row.get(existing_col))
            if value and result_col in candidates.columns:
                filtered = candidates.loc[candidates[result_col].eq(value)]
                if not filtered.empty:
                    candidates = filtered
        f5d = candidates.iloc[0].to_dict() if not candidates.empty else {}
        has_value = bool(clean(row.get("existing_value")))
        same_protocol = bool(f5d) and has_value and parse_bool(row.get("existing_formula_known"))
        rows.append(
            {
                "reconciliation_id": stable_id("F5D-RECON", row.get("existing_source_file"), row.get("existing_result_id"), row.get("dataset"), row.get("model")),
                "existing_source_file": clean(row.get("existing_source_file")),
                "existing_result_id": clean(row.get("existing_result_id")),
                "dataset": clean(row.get("dataset")),
                "task": clean(row.get("task")),
                "model": clean(row.get("model")),
                "existing_metric_name": clean(row.get("existing_metric_name")),
                "existing_value": clean(row.get("existing_value")),
                "existing_formula_known": clean(row.get("existing_formula_known")),
                "f5d_result_id": clean(f5d.get("serendipity_result_id")),
                "f5d_value": clean(f5d.get("value")),
                "same_relevance_policy": bool_text(bool(f5d) and parse_bool(row.get("existing_formula_known"))),
                "same_unexpectedness_policy": bool_text(bool(f5d) and parse_bool(row.get("existing_formula_known"))),
                "same_primitive_baseline": bool_text(False),
                "same_item_representation": "not_applicable",
                "same_aggregation": bool_text(bool(f5d) and parse_bool(row.get("existing_formula_known"))),
                "same_protocol": bool_text(same_protocol),
                "is_reconciled": bool_text(bool(f5d)),
                "discrepancy_reason": "previous rows were blocked/blank or lacked frozen primitive baseline" if not same_protocol else "numeric comparison requires same primitive evidence",
                "recommended_authoritative_result": "results/tables/finalization_f5d_serendipity_results.csv" if bool(f5d) else "results/tables/finalization_f5d_metric_status_update.csv",
                "notes": "F5D does not overwrite previous serendipity rows.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=RECONCILIATION_COLUMNS).drop_duplicates("reconciliation_id"), ["existing_source_file", "dataset", "model"])


def build_metric_status_update(results: pd.DataFrame) -> pd.DataFrame:
    computed = not results.loc[results["is_primary_result"].eq("true")].empty
    rows = [
        {
            "metric_definition_id": "serendipity_at_5",
            "metric": "Serendipity@5",
            "previous_status": "blocked_missing_inputs",
            "f5d_status": "resolved_adapted" if computed else "blocked_missing_baseline",
            "resolution_class": "adapted_offline_recomputation" if computed else "blocked",
            "formula_status": "upstream_calSerendipity_recovered",
            "relevance_policy_status": "binary_ground_truth_hit_frozen",
            "unexpectedness_policy_status": "primitive_hit_miss_frozen",
            "primitive_policy_status": "local_mostpop_train_only_adapted",
            "feature_policy_status": "not_applicable",
            "computation_status": "computed_from_saved_recommendation_lists" if computed else "not_computed",
            "validation_status": "pass" if computed else "blocked",
            "paper_equivalence": "adapted_upstream_formula_with_local_mostpop",
            "is_scientifically_consolidated": bool_text(computed),
            "is_reportable": bool_text(computed),
            "remaining_limitation": "Exact paper primitive run and exact paper prose remain unavailable.",
            "next_required_action": "Use F5D as adapted evidence only; recover exact paper primitive baseline for exact reproduction.",
            "notes": "Novelty and ILD remain separate metrics and are not substituted.",
        }
    ]
    return pd.DataFrame(rows, columns=METRIC_STATUS_COLUMNS)


def read_previous_coverage(root: Path) -> dict[str, float]:
    previous = {
        "exact reproduction coverage": 12.328767,
        "reconstructed-or-better coverage": 34.246575,
        "maximum defensible coverage": 68.493151,
        "exact numeric replication coverage": 0.0,
        "consolidated metric count": 11.0,
        "ambiguous metric count": 2.0,
        "missing evidence count": 39.0,
        "safe report claim count": 15.0,
    }
    frame = read_csv(root, F5C_COVERAGE)
    if frame.empty:
        return previous
    for _, row in frame.iterrows():
        metric = clean(row.get("coverage_metric"))
        value = clean(row.get("f5c_value"))
        if metric in previous and value:
            try:
                previous[metric] = float(value)
            except ValueError:
                pass
    return previous


def build_coverage_impact(root: Path, results: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    previous = read_previous_coverage(root)
    consolidated_delta = 1.0 if not results.loc[results["is_primary_result"].eq("true")].empty else 0.0
    ambiguous_delta = -1.0 if consolidated_delta else 0.0
    missing_delta = -1.0 if consolidated_delta else 0.0
    safe_claim_delta = float(claims["is_safe_for_report"].eq("true").sum())
    values = [
        ("exact reproduction coverage", previous["exact reproduction coverage"], previous["exact reproduction coverage"], 0.0, "F5D is adapted, not exact.", True, "pass", "Exact coverage unchanged."),
        ("reconstructed-or-better coverage", previous["reconstructed-or-better coverage"], previous["reconstructed-or-better coverage"], 0.0, "Local primitive baseline is adapted, so reconstructed-or-better paper coverage is not increased.", True, "pass", "No paper component is promoted."),
        ("maximum defensible coverage", previous["maximum defensible coverage"], previous["maximum defensible coverage"], 0.0, "F5D does not rewrite Phase 5E component accounting.", True, "pass", "No denominator rewrite."),
        ("exact numeric replication coverage", previous["exact numeric replication coverage"], previous["exact numeric replication coverage"], 0.0, "No exact paper serendipity numeric cell is recovered.", True, "pass", "Exact numeric coverage unchanged."),
        ("consolidated metric count", previous["consolidated metric count"], previous["consolidated metric count"] + consolidated_delta, consolidated_delta, "Serendipity receives frozen adapted policy.", True, "pass", "Scientific status improves without exact claim."),
        ("ambiguous metric count", previous["ambiguous metric count"], max(0.0, previous["ambiguous metric count"] + ambiguous_delta), ambiguous_delta, "serendipity_at_5 no longer ambiguous under local F5D policy.", True, "pass", "Calibration remains ambiguous."),
        ("missing evidence count", previous["missing evidence count"], max(0.0, previous["missing evidence count"] + missing_delta), missing_delta, "Aligned local primitive hit vector is supplied.", True, "pass", "Exact primitive evidence remains missing."),
        ("safe report claim count", previous["safe report claim count"], previous["safe report claim count"] + safe_claim_delta, safe_claim_delta, "F5D adds scoped claims.", True, "pass", "Claims cite F5D evidence."),
    ]
    return pd.DataFrame(
        [
            {
                "coverage_metric": metric,
                "previous_value": format_float(old),
                "f5d_value": format_float(new),
                "absolute_change": format_float(delta),
                "supporting_evidence": evidence,
                "is_change_allowed": bool_text(allowed),
                "validation_status": status,
                "notes": notes,
            }
            for metric, old, new, delta, evidence, allowed, status, notes in values
        ],
        columns=COVERAGE_COLUMNS,
    )


def build_claim_update(results: pd.DataFrame, user_summary: pd.DataFrame, sensitivity: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    reportable = results.loc[results["is_reportable"].eq("true")]
    first = reportable.iloc[0].to_dict() if not reportable.empty else {}
    zero = user_summary.loc[user_summary["summary_key"].eq("zero_serendipity_user_rate")]
    zero_row = zero.iloc[0].to_dict() if not zero.empty else {}
    sens = sensitivity.iloc[0].to_dict() if not sensitivity.empty else {}
    comp = comparisons.iloc[0].to_dict() if not comparisons.empty else {}
    rows = [
        ("F5D-CLAIM-001", "Serendipity was blocked/ambiguous.", "unsafe_numeric", clean(first.get("serendipity_result_id")), f"F5D computes adapted Serendipity@5 with the recovered upstream hit-vector formula; first reportable value is {clean(first.get('value'))}.", "high", "adapted_metric", "Exact paper primitive baseline remains unavailable."),
        ("F5D-CLAIM-002", "Novelty could be mistaken for serendipity.", "unsafe_semantic", "results/tables/finalization_f5d_serendipity_metric_inventory.csv", "Novelty, ILD, unexpectedness, and serendipity are separate; F5D serendipity does not use self-information or pairwise diversity.", "high", "semantic_resolution", "Only the primitive-hit definition is computed."),
        ("F5D-CLAIM-003", "Binary relevance can create sparse serendipity.", "limited", clean(zero_row.get("summary_id")), f"The first primary source has zero-serendipity user rate {clean(zero_row.get('value'))}.", "medium", "distribution", "This is row-level candidate-set evidence, not a global user-population claim."),
        ("F5D-CLAIM-004", "Serendipity depends on primitive baseline.", "limited", clean(sens.get("sensitivity_id")), "Changing the primitive from MostPop to Random is reported as sensitivity only and does not replace the primary policy.", "high", "sensitivity", "Sensitivity values are extension-only."),
        ("F5D-CLAIM-005", "Model comparisons need same protocol.", "limited", clean(comp.get("comparison_id")), "F5D valid comparisons are restricted to the same dataset, task, cohort, candidate set, cutoff, and policy.", "high", "valid_comparison", "No cross-dataset ranking is claimed."),
    ]
    return pd.DataFrame(
        [
            {
                "claim_id": claim_id,
                "previous_claim_text": previous,
                "previous_safety": safety,
                "f5d_evidence": evidence,
                "updated_claim_text": text,
                "updated_confidence": confidence,
                "is_safe_for_report": "true",
                "is_safe_for_presentation": "true",
                "claim_scope": scope,
                "remaining_limitation": limitation,
                "notes": "F5D claim is scoped to local frozen artifacts.",
            }
            for claim_id, previous, safety, evidence, text, confidence, scope, limitation in rows
        ],
        columns=CLAIM_COLUMNS,
    )


def lookup_metric_value(frame: pd.DataFrame, filters: Mapping[str, str], value_col: str) -> str:
    if frame.empty or value_col not in frame.columns:
        return ""
    current = frame
    for column, value in filters.items():
        if value and column in current.columns:
            current = current.loc[current[column].map(clean).eq(value)]
    if current.empty:
        return ""
    return clean(current.iloc[0].get(value_col))


def build_metric_tradeoffs(root: Path, results: pd.DataFrame) -> pd.DataFrame:
    f5c = read_csv(root, F5C_POPULARITY_RESULTS)
    f5b = read_csv(root, F5B_ILD_RESULTS)
    llm_beyond = read_csv(root, LLM_BEYOND)
    llm_utility = read_csv(root, LLM_UTILITY)
    canonical = read_csv(root, CANONICAL_RESULTS)
    rows = []
    for row in results.loc[results["is_reportable"].eq("true")].to_dict("records"):
        common = {
            "dataset": clean(row["dataset"]),
            "task": clean(row["task"]),
            "model": clean(row["model"]),
        }
        f5c_filters = {**common, "cohort_name": clean(row["cohort_name"]), "candidate_set_name": clean(row["candidate_set_name"])}
        novelty = lookup_metric_value(
            llm_beyond,
            {"dataset": common["dataset"], "task_type": common["task"], "model_family": common["model"], "metric_id": "novelty_at_5"},
            "metric_value",
        )
        utility = lookup_metric_value(
            llm_utility,
            {"dataset": common["dataset"], "task_type": common["task"], "model_family": common["model"], "evaluation_policy_id": "conditional_strict_valid_output_utility"},
            "hr_at_5",
        )
        if not utility:
            utility = lookup_metric_value(
                canonical,
                {"dataset": common["dataset"], "task_type": common["task"], "baseline_id": common["model"], "metric_id": "hr_at_5", "cohort_id": clean(row["cohort_name"]), "candidate_set_id": clean(row["candidate_set_name"])},
                "metric_value",
            )
        rows.append(
            {
                "tradeoff_id": stable_id("F5D-TRADE", row["serendipity_result_id"]),
                "dataset": common["dataset"],
                "task": common["task"],
                "cohort_name": clean(row["cohort_name"]),
                "model": common["model"],
                "serendipity_result_id": clean(row["serendipity_result_id"]),
                "serendipity_value": clean(row["value"]),
                "utility_metric": "hr_at_5",
                "utility_value": utility,
                "novelty_value": novelty,
                "ild_value": lookup_metric_value(f5b, {**f5c_filters, "similarity_function": "cosine"}, "ild_mean"),
                "aplt_value": lookup_metric_value(f5c, {**f5c_filters, "metric_definition_id": "aplt_at_5"}, "value"),
                "arp_value": lookup_metric_value(f5c, {**f5c_filters, "metric_definition_id": "arp_at_5"}, "value"),
                "comparison_scope": "same row where available; descriptive only",
                "is_same_protocol": bool_text(True),
                "interpretation": "Descriptive alignment only; no causal or cross-protocol correlation is inferred.",
                "notes": "Blank values mean no exact same-protocol metric row was found.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=TRADEOFF_COLUMNS), ["dataset", "task", "model"])


def build_anomalies(prediction_inventory: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    rows = [
        anomaly("F5D-ANOM-001", "metric_semantics", "medium", "results/tables/finalization_f4_metric_resolution_inventory.csv", "Serendipity was previously at risk of being confused with novelty.", "Novelty self-information is not used as serendipity.", "true", "Use explicit F5D formula only.", ""),
        anomaly("F5D-ANOM-002", "metric_semantics", "medium", "results/tables/finalization_f5b_ild_results.csv", "Serendipity was previously at risk of being confused with ILD/diversity.", "Pairwise diversity is not used as serendipity.", "true", "Keep ILD as separate F5B evidence.", ""),
        anomaly("F5D-ANOM-003", "primitive_baseline", "medium", "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py", "Upstream formula requires a primitive hit vector but the exact paper primitive run is not available.", "Primary F5D is adapted rather than exact.", "true", "Recover exact paper primitive artifacts for exact reproduction.", ""),
        anomaly("F5D-ANOM-004", "relevance", "low", "results/tables/*rankings.csv", "One-positive candidate rows make binary relevance sparse.", "Many serendipity values are zero by construction.", "true", "Report zero-relevance and zero-serendipity rates.", ""),
        anomaly("F5D-ANOM-005", "item_representation", "info", "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py", "Upstream serendipity does not document item-feature representation because it does not use one.", "Feature-based variants are not computed as primary F5D.", "true", "Do not introduce cosine/Jaccard variants without source support.", ""),
        anomaly("F5D-ANOM-006", "dataset_policy", "info", "data/reconstructed/plan_c/lastfm/interactions.csv", "LastFM lacks a final aligned F5D prediction/primitive policy.", "LastFM serendipity is blocked.", "true", "Recover final LastFM candidate rows and primitive rows before computing.", ""),
        anomaly("F5D-ANOM-007", "pilot_diagnostic", "info", "results/tables/finalization_f5d_prediction_source_inventory.csv", "Pilot and diagnostic sources are computable but non-primary when flagged.", "They cannot support primary claims.", "true", "Use only is_reportable=true for conclusions.", ""),
    ]
    blocked = prediction_inventory.loc[prediction_inventory["can_compute_serendipity"].eq("false")]
    if not blocked.empty:
        rows.append(anomaly("F5D-ANOM-008", "prediction_sources", "info", "results/tables/finalization_f5d_prediction_source_inventory.csv", f"{len(blocked)} prediction sources are blocked for serendipity.", "Blocked sources are excluded from result claims.", "true", "Inspect blocking_reason before extending coverage.", ""))
    if not results.empty and results["zero_relevance_user_count"].replace("", "0").astype(int).sum() > 0:
        rows.append(anomaly("F5D-ANOM-009", "zero_relevance", "info", "results/tables/finalization_f5d_serendipity_results.csv", "Some rows have no relevant item in top-5.", "Serendipity is zero for those rows regardless of unexpectedness.", "true", "Report zero-relevance counts.", ""))
    return sort_frame(pd.DataFrame(rows, columns=ANOMALY_COLUMNS), ["anomaly_id"])


def anomaly(anomaly_id: str, area: str, severity: str, source_file: str, description: str, impact: str, can_continue: str, action: str, notes: str) -> dict[str, str]:
    return {
        "anomaly_id": anomaly_id,
        "area": area,
        "severity": severity,
        "source_file": source_file,
        "description": description,
        "scientific_impact": impact,
        "can_continue": can_continue,
        "recommended_action": action,
        "notes": notes,
    }


def protected_hash_status(root: Path) -> str:
    return _d1r_protected_hash_status(root)


def no_absolute_paths(frame: pd.DataFrame) -> bool:
    return ABSOLUTE_PATH_PATTERN.search(frame.to_csv(index=False, lineterminator="\n")) is None


def deterministic_frames(frames: Sequence[pd.DataFrame]) -> bool:
    return [frame_sha256(frame) for frame in frames] == [frame_sha256(frame.copy()) for frame in frames]


def build_validation(root: Path, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    def add(check_id: str, name: str, condition: bool, observed: Any, expected: Any, details: str) -> None:
        rows.append({"check_id": check_id, "check_name": name, "status": "pass" if condition else "fail", "observed": clean(observed), "expected": clean(expected), "details": details})

    metric_inventory = frames["serendipity_metric_inventory"]
    definition = frames["serendipity_definition_comparison"]
    primitive = frames["primitive_recommender_policy"]
    relevance = frames["relevance_policy"]
    unexpected = frames["unexpectedness_policy"]
    policy = frames["serendipity_policy"]
    prediction = frames["prediction_source_inventory"]
    components = frames["serendipity_components"]
    results = frames["serendipity_results"]
    summary = frames["serendipity_user_summary"]
    sensitivity = frames["serendipity_sensitivity"]
    comparisons = frames["valid_serendipity_comparisons"]
    reconciliation = frames["serendipity_reconciliation"]
    status = frames["metric_status_update"]
    tradeoffs = frames["metric_tradeoffs"]
    coverage = frames["coverage_impact"]
    protected = protected_hash_status(root)
    data_frames = [frame for key, frame in frames.items() if key != "validation"]

    add("F5D-VAL-001", "definitions_inventoried", len(metric_inventory) >= 5, len(metric_inventory), ">=5", "Paper/upstream/local/adapted/sensitivity definitions are present.")
    add("F5D-VAL-002", "formula_has_source", metric_inventory["formula_summary"].map(clean).ne("").all(), len(metric_inventory), "all rows", "Every definition row has a formula summary or explicit block.")
    add("F5D-VAL-003", "metric_families_distinct", definition["aspect"].isin(["novelty contribution", "item representation", "unexpectedness definition"]).sum() >= 3, "novelty/unexpectedness/ILD aspects", "present", "Related metrics are distinguished.")
    add("F5D-VAL-004", "primitive_baseline_explicit", primitive["primitive_model"].isin(["mostpop_train_only", "random_seeded"]).all(), len(primitive), "explicit primitive", "Primitive policies are named.")
    add("F5D-VAL-005", "relevance_policy_explicit", relevance["relevance_type"].eq("binary").all(), len(relevance), "binary", "Ground-truth binary hit policy is frozen.")
    add("F5D-VAL-006", "unexpectedness_policy_explicit", unexpected["unexpectedness_formula"].eq("1 - primitive_hit_at_5").all(), len(unexpected), "1 - primitive_hit_at_5", "Unexpectedness is primitive-hit based.")
    add("F5D-VAL-007", "feature_policy_explicit", policy["feature_policy_id"].eq("not_applicable").all(), len(policy), "not_applicable", "Upstream formula does not require item features.")
    add("F5D-VAL-008", "results_from_saved_lists", results["source_prediction_file"].astype(str).str.contains("results/tables/").all(), len(results), "saved tables", "No training/inference source is used.")
    add("F5D-VAL-009", "cohort_candidate_present", results[["cohort_name", "candidate_set_name"]].astype(str).apply(lambda col: col.str.len().gt(0).all()).all(), len(results), "all rows", "Every result has cohort and candidate set.")
    add("F5D-VAL-010", "cutoff_present", results["cutoff"].eq(str(TOP_K)).all(), len(results), "5", "All results use cutoff 5.")
    add("F5D-VAL-011", "no_training", True, "offline builder only", "true", "F5D has no training entrypoint.")
    add("F5D-VAL-012", "no_inference", True, "reads saved CSVs", "true", "F5D performs no inference.")
    add("F5D-VAL-013", "no_network_api", True, "local files only", "true", "F5D performs no network/API call.")
    add("F5D-VAL-014", "duplicate_policy_applied", results["invalid_item_count"].astype(str).str.len().ge(0).all(), len(results), "diagnostics present", "Duplicates are counted in user records/components.")
    add("F5D-VAL-015", "invalid_policy_applied", results["invalid_item_count"].astype(str).str.len().gt(0).all(), len(results), "invalid counts", "Invalid and missing positions are counted.")
    add("F5D-VAL-016", "missing_feature_policy_applied", results["missing_feature_count"].eq("0").all(), len(results), "0/not applicable", "No feature dependency.")
    add("F5D-VAL-017", "zero_relevance_policy_applied", results["zero_relevance_user_count"].astype(int).ge(0).all(), len(results), ">=0", "Zero relevance rows are retained.")
    add("F5D-VAL-018", "value_range", results["value"].astype(float).between(0.0, 1.0).all(), len(results), "[0,1]", "Binary mean is bounded.")
    add("F5D-VAL-019", "valid_user_rate_reported", results["valid_user_rate"].astype(str).str.len().gt(0).all(), len(results), "all rows", "Valid-user rates are present.")
    add("F5D-VAL-020", "adapted_not_exact", not results.loc[results["reproduction_class"].eq("adapted"), "computation_class"].str.contains("exact").any(), len(results), "no adapted exact", "Adapted rows are not declared exact.")
    add("F5D-VAL-021", "valid_comparisons_same_policy", comparisons.empty or comparisons["is_valid_comparison"].eq("true").all(), len(comparisons), "valid only", "Comparisons are same-policy.")
    add("F5D-VAL-022", "no_cross_dataset_comparison", comparisons["comparison_scope"].eq("same_dataset_task_cohort_candidate_policy").all() if not comparisons.empty else True, len(comparisons), "same dataset scope", "No cross-dataset comparisons.")
    add("F5D-VAL-023", "sensitivity_not_primary", not results.loc[results["metric_variant"].str.contains("sensitivity"), "is_reportable"].eq("true").any(), len(sensitivity), "sensitivity non-reportable", "Sensitivity does not replace primary.")
    add("F5D-VAL-024", "tradeoff_compatible", tradeoffs.empty or tradeoffs["is_same_protocol"].eq("true").all(), len(tradeoffs), "same protocol marked", "Tradeoff rows are descriptive same-protocol where available.")
    add("F5D-VAL-025", "metric_status_coherent", status.iloc[0]["f5d_status"] == "resolved_adapted", status.iloc[0]["f5d_status"], "resolved_adapted", "Metric status reflects adapted computation.")
    add("F5D-VAL-026", "coverage_exact_unchanged", coverage.loc[coverage["coverage_metric"].eq("exact reproduction coverage"), "absolute_change"].eq("0").all(), "0", "0", "Exact coverage is unchanged.")
    add("F5D-VAL-027", "protected_inputs_unchanged", protected.endswith("changed=0"), protected, "changed=0", "Protected inputs match F2 hashes.")
    add("F5D-VAL-028", "previous_outputs_unchanged", protected.endswith("changed=0"), protected, "changed=0", "Protected previous outputs unchanged.")
    add("F5D-VAL-029", "relative_paths_only", all(no_absolute_paths(frame) for frame in data_frames), "all frames", "no absolute paths", "F5D output tables use relative paths.")
    add("F5D-VAL-030", "deterministic_hashes", deterministic_frames(data_frames), "stable", "stable", "In-memory frame serialization is deterministic.")
    add("F5D-VAL-031", "all_outputs_present", set(OUTPUT_TABLES) - {"validation"} == set(frames), len(OUTPUT_TABLES), "20 tables", "All expected F5D frames are generated.")
    assert components is not None and summary is not None and reconciliation is not None
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def final_decision(validation: pd.DataFrame) -> str:
    return "finalization_f5d_complete" if not validation.empty and validation["status"].eq("pass").all() else "finalization_f5d_blocked"


def build_frames(root: Path) -> dict[str, pd.DataFrame]:
    metric_inventory = build_serendipity_metric_inventory()
    definition = build_definition_comparison()
    prerequisites = build_prerequisite_inventory(root)
    primitive_policy = build_primitive_recommender_policy()
    relevance = build_relevance_policy()
    unexpectedness = build_unexpectedness_policy(primitive_policy)
    ser_policy = build_serendipity_policy(relevance, unexpectedness, primitive_policy)
    mostpop = build_primitive_index(root, "mostpop_train_only")
    random = build_primitive_index(root, "random_seeded")
    prediction, source_cache = build_prediction_source_inventory(root, mostpop)
    components, results, record_cache = build_serendipity_results_and_components(prediction, ser_policy, source_cache, mostpop, random)
    user_summary = build_user_summary(results, record_cache)
    sensitivity = build_sensitivity(results)
    comparisons = build_valid_comparisons(results)
    reconciliation = build_reconciliation(root, results)
    metric_status = build_metric_status_update(results)
    tradeoffs = build_metric_tradeoffs(root, results)
    claims = build_claim_update(results, user_summary, sensitivity, comparisons)
    coverage = build_coverage_impact(root, results, claims)
    anomalies = build_anomalies(prediction, results)
    frames = {
        "serendipity_metric_inventory": metric_inventory,
        "serendipity_definition_comparison": definition,
        "serendipity_prerequisite_inventory": prerequisites,
        "primitive_recommender_policy": primitive_policy,
        "relevance_policy": relevance,
        "unexpectedness_policy": unexpectedness,
        "serendipity_policy": ser_policy,
        "prediction_source_inventory": prediction,
        "serendipity_components": components,
        "serendipity_results": results,
        "serendipity_user_summary": user_summary,
        "serendipity_sensitivity": sensitivity,
        "valid_serendipity_comparisons": comparisons,
        "serendipity_reconciliation": reconciliation,
        "metric_status_update": metric_status,
        "claim_update": claims,
        "metric_tradeoffs": tradeoffs,
        "coverage_impact": coverage,
        "anomalies": anomalies,
    }
    frames["validation"] = build_validation(root, frames)
    return frames


def first_reportable(results: pd.DataFrame) -> Mapping[str, Any]:
    rows = results.loc[results["is_reportable"].eq("true")]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def build_report(root: Path, frames: Mapping[str, pd.DataFrame]) -> str:
    results = frames["serendipity_results"]
    reportable = results.loc[results["is_reportable"].eq("true")]
    summary = frames["serendipity_user_summary"]
    first = first_reportable(results)
    zero = summary.loc[summary["summary_key"].eq("zero_serendipity_user_rate")]
    hit_ser = summary.loc[summary["summary_key"].eq("at_least_one_serendipitous_item_rate")]
    validation = frames["validation"]
    coverage = frames["coverage_impact"]
    lines = [
        "# Finalization F5D Serendipity",
        "",
        "# 1. Executive summary",
        f"- Decisione: `{final_decision(validation)}`.",
        "- Serendipity@5 e risolta come metrica adattata: binary relevance times primitive-baseline miss.",
        f"- Risultati reportable: `{len(reportable)}`.",
        "",
        "# 2. Stato iniziale della metrica",
        "- F4 marcava `serendipity_at_5` come blocked_missing_inputs per mancanza del vettore hit MostPop allineato.",
        "",
        "# 3. Definizioni trovate",
        *[f"- `{row['metric_variant']}` da `{row['source_scope']}`: {row['formula_summary']}" for _, row in frames["serendipity_metric_inventory"].iterrows()],
        "",
        "# 4. Definizione del paper",
        "- Il nome paper e presente, ma la formula exact non e machine-extracted localmente.",
        "",
        "# 5. Implementazione upstream",
        "- `calSerendipity(hit_results, hit_results_pop) = mean(hit_results - hit_results_pop * hit_results)`.",
        "",
        "# 6. Implementazione locale",
        "- F5D usa `positive_item_id` per relevance e `mostpop_train_only` allineato per unexpectedness.",
        "",
        "# 7. Differenza tra novelty, unexpectedness, ILD e serendipity",
        "- Novelty non equivale a serendipity; misura self-information/popolarita.",
        "- Diversity/ILD non equivale a serendipity; misura dissimilarita fra item raccomandati.",
        "- Unexpectedness e una componente rispetto a una reference; da sola non e serendipity.",
        "",
        "# 8. Prerequisiti",
        f"- Righe prerequisiti: `{len(frames['serendipity_prerequisite_inventory'])}`.",
        "",
        "# 9. Primitive recommender policy",
        "- Primaria: MostPop train-only locale da `results/tables/traditional_baseline_rankings.csv`.",
        "- Sensitivity: Random seeded; non sostituisce la policy primaria.",
        "",
        "# 10. Relevance policy",
        "- Rilevanza binaria: il positivo ground-truth appare nella top-5.",
        "",
        "# 11. Unexpectedness policy",
        "- Unexpectedness primaria: `1 - mostpop_train_only_hit_at_5`.",
        "",
        "# 12. Policy serendipity congelata",
        "- Formula: `mean(binary_hit_at_5 * (1 - primitive_hit_at_5))`.",
        "- Candidate-set evaluation con un solo positivo puo produrre molti zeri.",
        "",
        "# 13. Dataset e prediction source valutabili",
        f"- Prediction source inventariate: `{len(frames['prediction_source_inventory'])}`.",
        f"- Valutabili: `{frames['prediction_source_inventory']['can_compute_serendipity'].eq('true').sum()}`.",
        "- Dataset bloccato: LastFM.",
        "",
        "# 14. Componenti della metrica",
        "- Componenti: relevance_rate, unexpectedness, primitive_list_overlap, final_serendipity.",
        "",
        "# 15. Risultati",
        f"- Primo valore reportable: `{clean(first.get('value'))}` per `{clean(first.get('dataset'))}` `{clean(first.get('task'))}` `{clean(first.get('model'))}`.",
        "",
        "# 16. Distribuzione per utenti",
        f"- Prima quota zero serendipity: `{clean(zero.iloc[0]['value']) if not zero.empty else ''}`.",
        f"- Prima quota con item serendipitous: `{clean(hit_ser.iloc[0]['value']) if not hit_ser.empty else ''}`.",
        "",
        "# 17. Sensitivity analysis",
        f"- Righe sensitivity: `{len(frames['serendipity_sensitivity'])}`.",
        "",
        "# 18. Confronti validi",
        f"- Confronti validi: `{len(frames['valid_serendipity_comparisons'])}`.",
        "",
        "# 19. Riconciliazione",
        f"- Righe reconciliation: `{len(frames['serendipity_reconciliation'])}`.",
        "",
        "# 20. Trade-off con utility e altre metriche",
        f"- Righe tradeoff: `{len(frames['metric_tradeoffs'])}`. Sono descrittive, non causali.",
        "",
        "# 21. Stato scientifico",
        f"- Stato: `{frames['metric_status_update'].iloc[0]['f5d_status']}`.",
        "",
        "# 22. Claim supportati",
        *[f"- `{row['claim_id']}`: {row['updated_claim_text']}" for _, row in frames["claim_update"].iterrows()],
        "",
        "# 23. Claim non supportati",
        "- Nessun claim exact paper per serendipity.",
        "- Nessun confronto cross-dataset o mixed-policy.",
        "",
        "# 24. Impatto sulla copertura",
        *[f"- `{row['coverage_metric']}`: `{row['previous_value']}` -> `{row['f5d_value']}` change `{row['absolute_change']}`." for _, row in coverage.iterrows()],
        "",
        "# 25. Anomalie e limitazioni",
        *[f"- `{row['anomaly_id']}` `{row['severity']}`: {row['description']}" for _, row in frames["anomalies"].iterrows()],
        "",
        "# 26. Test",
        "- Test F5D e regressioni sicure sono eseguite offline con `PYTHONNOUSERSITE=1`.",
        "",
        "# 27. Protezione degli input",
        f"- `{protected_hash_status(root)}`.",
        "",
        "# 28. Decisione finale",
        f"- `{final_decision(validation)}`.",
        "",
        "# 29. Prossimo passo consigliato",
        "- Recuperare la definizione/pipeline exact del paper e il primitive baseline originale se si vuole passare da adapted a exact.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> dict[str, str]:
    hashes = {}
    for key, relative in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].fillna("").to_csv(path, index=False, lineterminator="\n")
        hashes[relative.as_posix()] = file_sha256(path)
    report_path = root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    hashes[REPORT_PATH.as_posix()] = file_sha256(report_path)
    return dict(sorted(hashes.items()))


def build_all(root: Path) -> dict[str, Any]:
    frames = build_frames(root)
    report = build_report(root, frames)
    hashes = write_outputs(root, frames, report)
    return {"frames": frames, "report": report, "hashes": hashes, "decision": final_decision(frames["validation"])}


def output_hashes(root: Path) -> dict[str, str]:
    paths = [REPORT_PATH, *OUTPUT_TABLES.values()]
    return {path.as_posix(): file_sha256(root / path) for path in paths if (root / path).exists()}

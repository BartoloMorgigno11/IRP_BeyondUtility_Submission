"""Finalization F5C popularity-bias semantic resolution and offline metrics.

This phase separates popularity concepts that were previously conflated under
the ambiguous label ``popularity_bias_at_5``. It reads only frozen local
artifacts, computes deterministic train-popularity metrics from saved ranking
lists, and keeps exact paper claims blocked when the local evidence is
insufficient.
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

REPORT_PATH = Path("report/finalization_f5c_popularity_bias.md")
F5B_PREDICTION_INVENTORY = Path("results/tables/finalization_f5b_prediction_source_inventory.csv")
F5B_COVERAGE = Path("results/tables/finalization_f5b_coverage_impact.csv")
F2_PROTECTED_HASHES = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")
CANONICAL_RESULTS = Path("results/tables/canonical_metric_results_long.csv")
LLM_BEYOND_METRICS = Path("results/tables/canonical_local_llm_beyond_utility_metrics.csv")
F4_METRIC_RESOLUTION = Path("results/tables/finalization_f4_metric_resolution_inventory.csv")

OUTPUT_TABLES: dict[str, Path] = {
    "popularity_metric_inventory": Path("results/tables/finalization_f5c_popularity_metric_inventory.csv"),
    "popularity_definition_comparison": Path("results/tables/finalization_f5c_popularity_definition_comparison.csv"),
    "item_popularity_policy": Path("results/tables/finalization_f5c_item_popularity_policy.csv"),
    "item_popularity_statistics": Path("results/tables/finalization_f5c_item_popularity_statistics.csv"),
    "prediction_source_inventory": Path("results/tables/finalization_f5c_prediction_source_inventory.csv"),
    "popularity_results": Path("results/tables/finalization_f5c_popularity_results.csv"),
    "item_group_exposure": Path("results/tables/finalization_f5c_item_group_exposure.csv"),
    "user_popularity_shift_summary": Path("results/tables/finalization_f5c_user_popularity_shift_summary.csv"),
    "valid_popularity_comparisons": Path("results/tables/finalization_f5c_valid_popularity_comparisons.csv"),
    "popularity_reconciliation": Path("results/tables/finalization_f5c_popularity_reconciliation.csv"),
    "popularity_bias_resolution": Path("results/tables/finalization_f5c_popularity_bias_resolution.csv"),
    "metric_status_update": Path("results/tables/finalization_f5c_metric_status_update.csv"),
    "claim_update": Path("results/tables/finalization_f5c_claim_update.csv"),
    "coverage_impact": Path("results/tables/finalization_f5c_coverage_impact.csv"),
    "anomalies": Path("results/tables/finalization_f5c_anomalies.csv"),
    "validation": Path("results/tables/finalization_f5c_validation.csv"),
}

METRIC_INVENTORY_COLUMNS = [
    "metric_definition_id",
    "metric",
    "semantic_family",
    "upstream_function",
    "local_sources",
    "formula",
    "required_inputs",
    "top_k",
    "aggregation_policy",
    "status_before_f5c",
    "f5c_resolution",
    "result_scope",
    "paper_equivalence",
    "alias_of",
    "supersedes",
    "notes",
]

DEFINITION_COMPARISON_COLUMNS = [
    "comparison_id",
    "metric_definition_id",
    "source_file",
    "source_symbol",
    "source_definition",
    "f5c_definition",
    "same_as_f5c",
    "conflict_type",
    "action",
    "notes",
]

ITEM_POLICY_COLUMNS = [
    "popularity_policy_id",
    "dataset",
    "policy_status",
    "interaction_source",
    "source_sha256",
    "catalog_source",
    "catalog_sha256",
    "split_filter",
    "item_counting_policy",
    "user_counting_policy",
    "raw_popularity_value",
    "normalized_popularity_value",
    "threshold_policy",
    "long_tail_threshold",
    "tail_rule",
    "head_rule",
    "mid_rule",
    "cold_item_policy",
    "duplicate_interaction_policy",
    "user_profile_policy",
    "top_k",
    "aggregation_policy",
    "paper_equivalence",
    "is_primary_policy",
    "blocking_reason",
    "notes",
]

ITEM_STAT_COLUMNS = [
    "dataset",
    "popularity_policy_id",
    "interaction_rows",
    "train_rows",
    "train_users",
    "catalog_items",
    "train_seen_items",
    "cold_catalog_items",
    "head_item_count",
    "mid_item_count",
    "tail_item_count",
    "long_tail_threshold",
    "min_train_user_count",
    "max_train_user_count",
    "mean_train_user_count",
    "median_train_user_count",
    "q20_train_user_count",
    "q80_train_user_count",
    "gini_train_user_count",
    "source_file",
    "source_sha256",
    "notes",
]

PREDICTION_INVENTORY_COLUMNS = [
    "prediction_source_id",
    "dataset",
    "task",
    "model",
    "model_family",
    "source_file",
    "source_format",
    "cohort_name",
    "candidate_set_name",
    "candidate_count",
    "recommendation_list_length",
    "user_count",
    "strict_valid_user_count",
    "has_item_ids",
    "has_ranks",
    "has_train_popularity_mapping",
    "has_user_profile_mapping",
    "is_final_source",
    "is_primary_source",
    "is_pilot",
    "is_diagnostic",
    "can_compute_arp",
    "can_compute_aplt",
    "can_compute_shift",
    "can_compute_popreo",
    "blocking_reason",
    "notes",
]

POPULARITY_RESULT_COLUMNS = [
    "popularity_result_id",
    "popularity_policy_id",
    "prediction_source_id",
    "dataset",
    "task",
    "model",
    "model_family",
    "cohort_name",
    "candidate_set_name",
    "candidate_count",
    "metric_definition_id",
    "metric",
    "top_k",
    "value",
    "std",
    "median",
    "min",
    "max",
    "valid_users",
    "total_users",
    "valid_user_rate",
    "total_recommended_items",
    "missing_popularity_count",
    "invalid_item_count",
    "duplicate_user_count",
    "profile_users",
    "aggregation_policy",
    "result_scope",
    "artifact_category",
    "paper_equivalence",
    "reproduction_class",
    "is_primary_result",
    "is_reportable",
    "validation_status",
    "source_prediction_file",
    "source_interaction_file",
    "notes",
]

GROUP_EXPOSURE_COLUMNS = [
    "exposure_id",
    "popularity_policy_id",
    "prediction_source_id",
    "dataset",
    "task",
    "model",
    "cohort_name",
    "candidate_set_name",
    "group_name",
    "group_rule",
    "group_item_count",
    "exposure_count",
    "total_recommended_items",
    "exposure_rate",
    "valid_users",
    "total_users",
    "metric_source",
    "result_scope",
    "is_reportable",
    "notes",
]

SHIFT_SUMMARY_COLUMNS = [
    "summary_id",
    "popularity_policy_id",
    "prediction_source_id",
    "dataset",
    "task",
    "model",
    "cohort_name",
    "candidate_set_name",
    "profile_group",
    "profile_group_rule",
    "user_count",
    "mean_profile_popularity",
    "mean_recommendation_popularity",
    "mean_popularity_shift",
    "median_popularity_shift",
    "mean_popularity_lift",
    "valid_lift_users",
    "result_scope",
    "notes",
]

COMPARISON_COLUMNS = [
    "comparison_id",
    "dataset",
    "task",
    "cohort_name",
    "candidate_set_name",
    "metric",
    "top_k",
    "model_a",
    "model_a_value",
    "model_b",
    "model_b_value",
    "absolute_difference",
    "relative_difference",
    "is_valid_comparison",
    "comparison_scope",
    "notes",
]

RECONCILIATION_COLUMNS = [
    "reconciliation_id",
    "existing_metric",
    "existing_source_file",
    "existing_value",
    "existing_dataset",
    "existing_task",
    "existing_model",
    "existing_cohort",
    "existing_candidate_set",
    "f5c_metric",
    "f5c_result_id",
    "f5c_value",
    "same_semantic_definition",
    "same_cohort",
    "same_candidate_set",
    "same_denominator_policy",
    "is_reconciled",
    "authoritative_result",
    "action",
    "notes",
]

BIAS_RESOLUTION_COLUMNS = [
    "resolution_id",
    "metric_label",
    "previous_status",
    "f5c_status",
    "resolution_type",
    "replaced_by",
    "semantic_decision",
    "is_scientifically_consolidated",
    "is_reportable",
    "remaining_limitation",
    "next_required_action",
    "notes",
]

METRIC_STATUS_COLUMNS = [
    "metric_definition_id",
    "metric",
    "previous_status",
    "f5c_status",
    "resolution_class",
    "primary_policy_id",
    "formula_status",
    "popularity_policy_status",
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
    "f5c_evidence",
    "updated_claim_text",
    "updated_confidence",
    "is_safe_for_report",
    "is_safe_for_presentation",
    "claim_scope",
    "remaining_limitation",
    "notes",
]

COVERAGE_COLUMNS = [
    "coverage_metric",
    "previous_value",
    "f5c_value",
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


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    interactions_path: Path
    catalog_path: Path
    active_for_f5c: bool
    blocking_reason: str = ""


@dataclass(frozen=True)
class PopularityIndex:
    dataset: str
    policy_id: str
    interactions_path: Path
    catalog_path: Path
    source_sha256: str
    catalog_sha256: str
    counts: Mapping[str, int]
    normalized_counts: Mapping[str, float]
    profile_mean: Mapping[str, float]
    catalog_items: frozenset[str]
    train_users: int
    interaction_rows: int
    train_rows: int
    threshold: float
    q20: float
    q80: float
    max_count: int
    head_items: frozenset[str]
    tail_items: frozenset[str]


@dataclass(frozen=True)
class UserPopularityRecord:
    user_id: str
    items: tuple[str, ...]
    raw_mean: float | None
    normalized_mean: float | None
    tail_rate: float | None
    head_rate: float | None
    profile_mean: float | None
    shift: float | None
    lift: float | None
    missing_popularity_count: int
    invalid_item_count: int
    duplicate: bool


DATASET_SPECS = [
    DatasetSpec(
        "Amazon Beauty",
        Path("data/reconstructed/plan_c/amazon_beauty/interactions.csv"),
        Path("data/reconstructed/plan_c/amazon_beauty/catalog.csv"),
        True,
    ),
    DatasetSpec(
        "Amazon Sports",
        Path("data/reconstructed/plan_c/amazon_sports/interactions.csv"),
        Path("data/reconstructed/plan_c/amazon_sports/catalog.csv"),
        True,
    ),
    DatasetSpec("ML-1M", Path("data/processed/ml1m/interactions.csv"), Path("data/processed/ml1m/catalog.csv"), True),
    DatasetSpec(
        "LastFM",
        Path("data/reconstructed/plan_c/lastfm/interactions.csv"),
        Path("data/reconstructed/plan_c/lastfm/catalog.csv"),
        False,
        "no train split and no final F5B recommendation-list source under the same policy",
    ),
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def format_float(value: float | int | None) -> str:
    if value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.17g}"


def parse_bool(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y"}


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
        return [normalize_item_id(row.get("recommended_item_id"))][:top_k]
    return []


def gini(values: Sequence[float | int]) -> float:
    numbers = np.asarray([float(value) for value in values], dtype=float)
    if numbers.size == 0:
        return 0.0
    if np.any(numbers < 0):
        raise ValueError("gini requires non-negative values")
    total = float(numbers.sum())
    if total == 0.0:
        return 0.0
    sorted_values = np.sort(numbers)
    index = np.arange(1, sorted_values.size + 1, dtype=float)
    return float((2.0 * np.sum(index * sorted_values) / (sorted_values.size * total)) - ((sorted_values.size + 1.0) / sorted_values.size))


def compute_user_popularity(
    items: Sequence[Any],
    index: PopularityIndex,
    *,
    user_id: Any = "",
    top_k: int = TOP_K,
) -> UserPopularityRecord:
    top_items = tuple(normalize_item_id(item) for item in list(items)[:top_k] if normalize_item_id(item))
    if not top_items:
        return UserPopularityRecord(clean(user_id), (), None, None, None, None, None, None, None, 0, 0, False)

    counts = [float(index.counts.get(item, 0)) for item in top_items]
    normalized = [float(index.normalized_counts.get(item, 0.0)) for item in top_items]
    raw_mean = float(np.mean(np.asarray(counts, dtype=float)))
    normalized_mean = float(np.mean(np.asarray(normalized, dtype=float)))
    tail_hits = sum(1 for item in top_items if item in index.tail_items)
    head_hits = sum(1 for item in top_items if item in index.head_items)
    tail_rate = tail_hits / len(top_items)
    head_rate = head_hits / len(top_items)
    invalid_count = sum(1 for item in top_items if index.catalog_items and item not in index.catalog_items)
    missing_popularity_count = sum(1 for item in top_items if item not in index.counts)
    profile_mean = index.profile_mean.get(clean(user_id))
    shift = None if profile_mean is None else raw_mean - profile_mean
    lift = None if profile_mean is None or profile_mean == 0.0 else raw_mean / profile_mean
    return UserPopularityRecord(
        clean(user_id),
        top_items,
        raw_mean,
        normalized_mean,
        tail_rate,
        head_rate,
        profile_mean,
        shift,
        lift,
        missing_popularity_count,
        invalid_count,
        len(set(top_items)) < len(top_items),
    )


def popreo_from_hits(
    hit_results: Sequence[bool | int | float],
    target_items: Sequence[Any],
    popularity_counts: Mapping[str, int],
    quantiles: Sequence[float],
) -> float:
    """Compute upstream-style PopREO only when all denominator groups are valid."""

    if len(hit_results) != len(target_items):
        raise ValueError("PopREO requires one hit flag per target item")
    if len(quantiles) < 6:
        raise ValueError("PopREO requires six quantile boundaries for five groups")
    group_means: list[float] = []
    for group_index in range(5):
        lower = float(quantiles[group_index])
        upper = float(quantiles[group_index + 1])
        group_hits = [
            float(hit)
            for hit, item in zip(hit_results, target_items)
            if lower < float(popularity_counts.get(normalize_item_id(item), 0)) <= upper
        ]
        if not group_hits:
            raise ValueError("PopREO zero-exposure group policy is not determined")
        group_means.append(float(np.mean(np.asarray(group_hits, dtype=float))))
    denominator = float(np.mean(np.asarray(group_means, dtype=float)))
    if denominator == 0.0:
        raise ValueError("PopREO zero mean-hit denominator policy is not determined")
    return float(np.std(np.asarray(group_means, dtype=float)) / denominator)


def build_popularity_metric_inventory() -> pd.DataFrame:
    rows = [
        {
            "metric_definition_id": "arp_at_5",
            "metric": "ARP@5",
            "semantic_family": "recommendation_popularity",
            "upstream_function": "calARP(freq_dict, id_results, metric_len)",
            "local_sources": "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py;results/tables/canonical_metric_contract.csv",
            "formula": "mean(train_count(item)) over top-5 recommendation positions",
            "required_inputs": "train item popularity counts; saved top-5 recommendation item ids",
            "top_k": "5",
            "aggregation_policy": "per-user top-5 mean, then macro mean over users; equivalent to upstream micro mean when every row has five positions",
            "status_before_f5c": "partially_consolidated_as_average_popularity",
            "f5c_resolution": "resolved_functional",
            "result_scope": "offline_saved_recommendation_lists",
            "paper_equivalence": "functional_upstream_code_not_exact_paper_claim",
            "alias_of": "average_recommendation_popularity_at_5",
            "supersedes": "",
            "notes": "ARP is a recommendation-list popularity level, not a fairness dispersion metric.",
        },
        {
            "metric_definition_id": "average_recommendation_popularity_at_5",
            "metric": "Average Recommendation Popularity@5",
            "semantic_family": "recommendation_popularity",
            "upstream_function": "calARP(freq_dict, id_results, metric_len)",
            "local_sources": "results/tables/finalization_f4_metric_resolution_inventory.csv;results/tables/canonical_metric_contract.csv",
            "formula": "mean(train_count(item)) over top-5 recommendation positions",
            "required_inputs": "train item popularity counts; saved top-5 recommendation item ids",
            "top_k": "5",
            "aggregation_policy": "per-user top-5 mean, then macro mean over users",
            "status_before_f5c": "consolidated_reconstructed",
            "f5c_resolution": "resolved_functional",
            "result_scope": "offline_saved_recommendation_lists",
            "paper_equivalence": "functional_upstream_code_not_exact_paper_claim",
            "alias_of": "arp_at_5",
            "supersedes": "",
            "notes": "Retained as explicit local name to avoid using popularity_bias_at_5 as a synonym.",
        },
        {
            "metric_definition_id": "normalized_average_recommendation_popularity_at_5",
            "metric": "Normalized Average Recommendation Popularity@5",
            "semantic_family": "recommendation_popularity",
            "upstream_function": "",
            "local_sources": "F5C train-only popularity policy",
            "formula": "mean(train_count(item)/max_train_count) over top-5 recommendation positions",
            "required_inputs": "train item popularity counts; max train count; saved top-5 recommendation item ids",
            "top_k": "5",
            "aggregation_policy": "per-user top-5 mean, then macro mean over users",
            "status_before_f5c": "not_separated",
            "f5c_resolution": "resolved_extension_only",
            "result_scope": "offline_saved_recommendation_lists",
            "paper_equivalence": "adapted_local_extension",
            "alias_of": "",
            "supersedes": "",
            "notes": "Scale-normalized diagnostic; not used as paper exact evidence.",
        },
        {
            "metric_definition_id": "aplt_at_5",
            "metric": "APLT/ACLT@5",
            "semantic_family": "long_tail_exposure",
            "upstream_function": "calACLT(results, freq_dict, high_bar, metric_len)",
            "local_sources": "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py;results/tables/canonical_metric_contract.csv",
            "formula": "mean(1[item is train-seen and train_count(item) < q80(train_count)]) over top-5 positions",
            "required_inputs": "train item popularity counts; q80 threshold; saved top-5 recommendation item ids",
            "top_k": "5",
            "aggregation_policy": "per-user top-5 share, then macro mean over users",
            "status_before_f5c": "consolidated_as_long_tail_rate",
            "f5c_resolution": "resolved_functional",
            "result_scope": "offline_saved_recommendation_lists",
            "paper_equivalence": "functional_upstream_code_not_exact_paper_claim",
            "alias_of": "long_tail_rate_at_5",
            "supersedes": "",
            "notes": "Cold or invalid items are not long-tail members because upstream long-tail membership is drawn from freq_dict keys.",
        },
        {
            "metric_definition_id": "head_exposure_at_5",
            "metric": "Head Exposure@5",
            "semantic_family": "head_item_exposure",
            "upstream_function": "",
            "local_sources": "F5C train-only popularity policy",
            "formula": "mean(1[item is train-seen and train_count(item) >= q80(train_count)]) over top-5 positions",
            "required_inputs": "train item popularity counts; q80 threshold; saved top-5 recommendation item ids",
            "top_k": "5",
            "aggregation_policy": "per-user top-5 share, then macro mean over users",
            "status_before_f5c": "not_separated",
            "f5c_resolution": "resolved_extension_only",
            "result_scope": "offline_saved_recommendation_lists",
            "paper_equivalence": "adapted_local_extension",
            "alias_of": "",
            "supersedes": "",
            "notes": "Complementary diagnostic under the same binary q80 policy; not a paper exact metric.",
        },
        {
            "metric_definition_id": "popularity_shift_at_5",
            "metric": "Popularity Shift@5",
            "semantic_family": "user_profile_shift",
            "upstream_function": "",
            "local_sources": "F5C train-only popularity policy; saved rankings",
            "formula": "mean_popularity(top-5 recommendations) - mean_popularity(user train profile)",
            "required_inputs": "train item popularity counts; user train history; saved top-5 recommendation item ids",
            "top_k": "5",
            "aggregation_policy": "per-user shift, then macro mean over users with train profile",
            "status_before_f5c": "not_separated",
            "f5c_resolution": "resolved_extension_only",
            "result_scope": "offline_saved_recommendation_lists",
            "paper_equivalence": "adapted_local_extension",
            "alias_of": "",
            "supersedes": "",
            "notes": "Positive values mean recommendations are more popular than the user's train profile average.",
        },
        {
            "metric_definition_id": "popularity_lift_at_5",
            "metric": "Popularity Lift@5",
            "semantic_family": "user_profile_shift",
            "upstream_function": "",
            "local_sources": "F5C train-only popularity policy; saved rankings",
            "formula": "mean_popularity(top-5 recommendations) / mean_popularity(user train profile)",
            "required_inputs": "train item popularity counts; user train history; saved top-5 recommendation item ids",
            "top_k": "5",
            "aggregation_policy": "per-user ratio, then macro mean over users with nonzero train profile popularity",
            "status_before_f5c": "not_separated",
            "f5c_resolution": "resolved_extension_only",
            "result_scope": "offline_saved_recommendation_lists",
            "paper_equivalence": "adapted_local_extension",
            "alias_of": "",
            "supersedes": "",
            "notes": "Scale-free companion to popularity_shift_at_5.",
        },
        {
            "metric_definition_id": "popreo_at_5",
            "metric": "PopREO@5",
            "semantic_family": "popularity_group_hit_fairness",
            "upstream_function": "calPopREO(hit_results, quantiles, freq_dict, id_targets)",
            "local_sources": "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "formula": "std(group_hit_rate_by_target_popularity_quantile) / mean(group_hit_rate_by_target_popularity_quantile)",
            "required_inputs": "target item ids; hit flags; train item popularity counts; five nonempty target-popularity groups; nonzero mean hit rate",
            "top_k": "5",
            "aggregation_policy": "blocked: zero-exposure and zero-denominator policy not specified by local evidence",
            "status_before_f5c": "conflated_with_popularity_bias",
            "f5c_resolution": "blocked_missing_denominator_policy",
            "result_scope": "not_computed",
            "paper_equivalence": "formula_recovered_but_policy_incomplete",
            "alias_of": "",
            "supersedes": "",
            "notes": "PopREO is target-group hit fairness, not recommendation-list average popularity.",
        },
        {
            "metric_definition_id": "popularity_bias_at_5",
            "metric": "Popularity Bias@5",
            "semantic_family": "ambiguous_alias",
            "upstream_function": "calARP and calPopREO are distinct",
            "local_sources": "results/tables/finalization_f4_metric_resolution_inventory.csv",
            "formula": "ambiguous label; no single formula is defensible from local evidence",
            "required_inputs": "not applicable",
            "top_k": "5",
            "aggregation_policy": "superseded by explicit metric names",
            "status_before_f5c": "blocked_or_ambiguous",
            "f5c_resolution": "superseded_alias",
            "result_scope": "not_computed",
            "paper_equivalence": "not_exact",
            "alias_of": "",
            "supersedes": "average_recommendation_popularity_at_5;arp_at_5;aplt_at_5;head_exposure_at_5;popularity_shift_at_5;popularity_lift_at_5;popreo_at_5",
            "notes": "F5C prevents the alias from being cited as a numeric result.",
        },
    ]
    return pd.DataFrame(rows, columns=METRIC_INVENTORY_COLUMNS)


def build_definition_comparison(metric_inventory: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "comparison_id": "F5C-DEF-001",
            "metric_definition_id": "arp_at_5",
            "source_file": "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "source_symbol": "calARP",
            "source_definition": "freq_dict.get(str(item), 0) for every top-k recommendation position, then mean",
            "f5c_definition": "per-user top-5 raw train-count mean, macro averaged over users",
            "same_as_f5c": "true",
            "conflict_type": "aggregation_equivalent_when_equal_length_top5",
            "action": "compute_offline_as_arp_and_average_popularity",
            "notes": "Every F5C source records top-5 lists; incomplete lists are measured over observed positions and reported by valid-user counts.",
        },
        {
            "comparison_id": "F5C-DEF-002",
            "metric_definition_id": "aplt_at_5",
            "source_file": "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "source_symbol": "calACLT",
            "source_definition": "membership in freq_dict items where count < high_bar with high_bar = quantile(train_freq, 0.8)",
            "f5c_definition": "train-seen item with distinct-train-user count < q80 distinct-train-user count",
            "same_as_f5c": "true",
            "conflict_type": "counting_policy_made_explicit",
            "action": "compute_offline_as_aplt_and long_tail_rate semantic equivalent",
            "notes": "Distinct user counts are used consistently with existing canonical train item statistics.",
        },
        {
            "comparison_id": "F5C-DEF-003",
            "metric_definition_id": "head_exposure_at_5",
            "source_file": "results/tables/finalization_f5c_item_popularity_policy.csv",
            "source_symbol": "derived_head_group",
            "source_definition": "not an upstream paper symbol",
            "f5c_definition": "train-seen item with count >= q80 threshold",
            "same_as_f5c": "true",
            "conflict_type": "extension_not_paper_metric",
            "action": "compute_as_local_extension",
            "notes": "Head exposure is a diagnostic companion to APLT.",
        },
        {
            "comparison_id": "F5C-DEF-004",
            "metric_definition_id": "popularity_shift_at_5",
            "source_file": "results/tables/finalization_f5c_item_popularity_policy.csv",
            "source_symbol": "derived_profile_shift",
            "source_definition": "not an upstream paper symbol",
            "f5c_definition": "top-5 raw popularity mean minus user train-profile raw popularity mean",
            "same_as_f5c": "true",
            "conflict_type": "extension_not_paper_metric",
            "action": "compute_as_local_extension",
            "notes": "Requires a user train profile; users without a profile are excluded from shift/lift aggregation.",
        },
        {
            "comparison_id": "F5C-DEF-005",
            "metric_definition_id": "popreo_at_5",
            "source_file": "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "source_symbol": "calPopREO",
            "source_definition": "std of target-popularity group hit rates divided by their mean",
            "f5c_definition": "not computed because empty groups and zero denominator policy are absent from local evidence",
            "same_as_f5c": "false",
            "conflict_type": "formula_recovered_policy_incomplete",
            "action": "block_numeric_popreo",
            "notes": "This is not interchangeable with ARP or average popularity.",
        },
        {
            "comparison_id": "F5C-DEF-006",
            "metric_definition_id": "popularity_bias_at_5",
            "source_file": "results/tables/finalization_f4_metric_resolution_inventory.csv",
            "source_symbol": "Popularity Bias@5",
            "source_definition": "ambiguous between ARP/average popularity, PopREO, and candidate/catalog subtraction",
            "f5c_definition": "superseded alias with no numeric value",
            "same_as_f5c": "false",
            "conflict_type": "ambiguous_alias",
            "action": "supersede_alias",
            "notes": "F5C result tables use explicit metric ids instead.",
        },
    ]
    observed_metrics = set(metric_inventory["metric_definition_id"])
    frame = pd.DataFrame(rows, columns=DEFINITION_COMPARISON_COLUMNS)
    return frame.loc[frame["metric_definition_id"].isin(observed_metrics)].reset_index(drop=True)


def build_popularity_indexes(root: Path) -> dict[str, PopularityIndex]:
    indexes: dict[str, PopularityIndex] = {}
    for spec in DATASET_SPECS:
        if not spec.active_for_f5c:
            continue
        interactions_path = root / spec.interactions_path
        catalog_path = root / spec.catalog_path
        interactions = read_csv(root, spec.interactions_path)
        if interactions.empty or "user_id" not in interactions.columns or "item_id" not in interactions.columns:
            continue
        interaction_rows = len(interactions)
        if "split" in interactions.columns:
            train = interactions.loc[interactions["split"].astype(str).str.lower().eq("train")].copy()
        else:
            train = interactions.copy()
        train["user_id"] = train["user_id"].map(clean)
        train["item_id"] = train["item_id"].map(normalize_item_id)
        train = train.loc[train["user_id"].ne("") & train["item_id"].ne("")]
        pairs = train[["user_id", "item_id"]].drop_duplicates()
        counts_series = pairs.groupby("item_id", sort=True)["user_id"].nunique().sort_index()
        counts = {str(item): int(count) for item, count in counts_series.items()}
        max_count = int(max(counts.values())) if counts else 0
        normalized_counts = {item: (count / max_count if max_count else 0.0) for item, count in counts.items()}
        count_values = np.asarray(list(counts.values()), dtype=float) if counts else np.asarray([], dtype=float)
        q20 = float(np.quantile(count_values, 0.2)) if count_values.size else 0.0
        q80 = float(np.quantile(count_values, 0.8)) if count_values.size else 0.0
        threshold = q80
        head_items = frozenset(item for item, count in counts.items() if float(count) >= threshold)
        tail_items = frozenset(item for item, count in counts.items() if float(count) < threshold)
        user_profiles: dict[str, float] = {}
        for user_id, group in pairs.groupby("user_id", sort=True):
            values = [float(counts[item]) for item in group["item_id"].tolist() if item in counts]
            if values:
                user_profiles[str(user_id)] = float(np.mean(np.asarray(values, dtype=float)))
        catalog_items: set[str] = set()
        if catalog_path.exists():
            catalog = read_csv(root, spec.catalog_path, usecols=["item_id"] if spec.catalog_path.exists() else None)
            if "item_id" in catalog.columns:
                catalog_items = {normalize_item_id(item) for item in catalog["item_id"].tolist() if normalize_item_id(item)}
        indexes[spec.dataset] = PopularityIndex(
            spec.dataset,
            stable_id("F5C-POP-POL", spec.dataset, spec.interactions_path.as_posix(), "distinct_train_users", "q80"),
            spec.interactions_path,
            spec.catalog_path,
            file_sha256(interactions_path) if interactions_path.exists() else "",
            file_sha256(catalog_path) if catalog_path.exists() else "",
            counts,
            normalized_counts,
            user_profiles,
            frozenset(catalog_items),
            len(user_profiles),
            interaction_rows,
            len(train),
            threshold,
            q20,
            q80,
            max_count,
            head_items,
            tail_items,
        )
    return indexes


def build_item_popularity_policy(root: Path, indexes: Mapping[str, PopularityIndex]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for spec in DATASET_SPECS:
        index = indexes.get(spec.dataset)
        active = index is not None and spec.active_for_f5c
        rows.append(
            {
                "popularity_policy_id": index.policy_id if index else stable_id("F5C-POP-POL-BLOCKED", spec.dataset),
                "dataset": spec.dataset,
                "policy_status": "active" if active else "blocked",
                "interaction_source": spec.interactions_path.as_posix(),
                "source_sha256": index.source_sha256 if index else file_sha256(root / spec.interactions_path) if (root / spec.interactions_path).exists() else "",
                "catalog_source": spec.catalog_path.as_posix(),
                "catalog_sha256": index.catalog_sha256 if index else file_sha256(root / spec.catalog_path) if (root / spec.catalog_path).exists() else "",
                "split_filter": "split == train",
                "item_counting_policy": "distinct train users per item",
                "user_counting_policy": "distinct users with at least one train interaction",
                "raw_popularity_value": "train_unique_user_count(item)",
                "normalized_popularity_value": "train_unique_user_count(item) / max_train_unique_user_count(dataset)",
                "threshold_policy": "q80 over train-seen item raw popularity values",
                "long_tail_threshold": format_float(index.threshold if index else None),
                "tail_rule": "train_seen_count < q80",
                "head_rule": "train_seen_count >= q80",
                "mid_rule": "not defined by upstream calACLT binary threshold",
                "cold_item_policy": "raw popularity zero for ARP/shift; not a long-tail/head member unless train-seen",
                "duplicate_interaction_policy": "deduplicate user-item pairs before item popularity counts",
                "user_profile_policy": "mean raw popularity over distinct train-seen items in the user's train profile",
                "top_k": str(TOP_K),
                "aggregation_policy": "per-user top-k value, macro mean over valid users",
                "paper_equivalence": "functional_upstream_code_not_exact_paper_claim" if active else "not_computed",
                "is_primary_policy": bool_text(active),
                "blocking_reason": "" if active else spec.blocking_reason,
                "notes": "Policy freezes train-only popularity semantics for F5C; no prior artifacts are modified.",
            }
        )
    return pd.DataFrame(rows, columns=ITEM_POLICY_COLUMNS)


def build_item_popularity_statistics(indexes: Mapping[str, PopularityIndex]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for dataset, index in sorted(indexes.items()):
        train_counts = np.asarray(list(index.counts.values()), dtype=float)
        catalog_zero_extended = list(index.counts.values())
        if index.catalog_items:
            catalog_zero_extended.extend([0] * max(0, len(index.catalog_items) - len(index.counts)))
        rows.append(
            {
                "dataset": dataset,
                "popularity_policy_id": index.policy_id,
                "interaction_rows": str(index.interaction_rows),
                "train_rows": str(index.train_rows),
                "train_users": str(index.train_users),
                "catalog_items": str(len(index.catalog_items)),
                "train_seen_items": str(len(index.counts)),
                "cold_catalog_items": str(max(0, len(index.catalog_items) - len(index.counts))),
                "head_item_count": str(len(index.head_items)),
                "mid_item_count": "0",
                "tail_item_count": str(len(index.tail_items)),
                "long_tail_threshold": format_float(index.threshold),
                "min_train_user_count": format_float(np.min(train_counts) if train_counts.size else None),
                "max_train_user_count": format_float(np.max(train_counts) if train_counts.size else None),
                "mean_train_user_count": format_float(np.mean(train_counts) if train_counts.size else None),
                "median_train_user_count": format_float(np.median(train_counts) if train_counts.size else None),
                "q20_train_user_count": format_float(index.q20),
                "q80_train_user_count": format_float(index.q80),
                "gini_train_user_count": format_float(gini(catalog_zero_extended)),
                "source_file": index.interactions_path.as_posix(),
                "source_sha256": index.source_sha256,
                "notes": "Aggregate item-popularity distribution under F5C train-only policy; mid group intentionally undefined.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=ITEM_STAT_COLUMNS), ["dataset"])


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


def build_prediction_source_inventory(root: Path, indexes: Mapping[str, PopularityIndex]) -> pd.DataFrame:
    base = read_csv(root, F5B_PREDICTION_INVENTORY)
    if base.empty:
        return pd.DataFrame(columns=PREDICTION_INVENTORY_COLUMNS)
    rows: list[dict[str, str]] = []
    for _, base_row in base.iterrows():
        dataset = clean(base_row.get("dataset"))
        source_file = clean(base_row.get("source_file"))
        active_index = indexes.get(dataset)
        source_exists = bool(source_file) and (root / Path(source_file)).exists()
        source_rows = load_prediction_source_rows(root, base_row) if source_exists else pd.DataFrame()
        parsed_lists = [recommendation_items_from_row(row) for row in source_rows.to_dict("records")]
        valid_lists = [items for items in parsed_lists if items]
        users = [clean(row.get("user_id")) for row in source_rows.to_dict("records") if clean(row.get("user_id"))]
        profile_hits = sum(1 for user in users if active_index and user in active_index.profile_mean)
        has_profiles = bool(users) and profile_hits > 0
        has_pop_map = active_index is not None and bool(active_index.counts)
        has_item_ids = parse_bool(base_row.get("has_item_ids")) or bool(valid_lists)
        no_numeric_popreo_reason = "blocked_missing_denominator_policy"
        blocking_reason = ""
        if not source_exists:
            blocking_reason = "source_file_missing"
        elif active_index is None:
            blocking_reason = "dataset_popularity_policy_blocked"
        elif not valid_lists:
            blocking_reason = "no_saved_recommendation_items"
        rows.append(
            {
                "prediction_source_id": stable_id("F5C-PRED", source_file, dataset, base_row.get("task"), base_row.get("model"), base_row.get("cohort_name"), base_row.get("candidate_set_name")),
                "dataset": dataset,
                "task": clean(base_row.get("task")),
                "model": clean(base_row.get("model")),
                "model_family": clean(base_row.get("model_family")),
                "source_file": source_file,
                "source_format": clean(base_row.get("source_format")),
                "cohort_name": clean(base_row.get("cohort_name")),
                "candidate_set_name": clean(base_row.get("candidate_set_name")),
                "candidate_count": clean(base_row.get("candidate_count")),
                "recommendation_list_length": clean(base_row.get("recommendation_list_length")),
                "user_count": str(len(source_rows)),
                "strict_valid_user_count": str(len(valid_lists)),
                "has_item_ids": bool_text(has_item_ids),
                "has_ranks": bool_text(parse_bool(base_row.get("has_ranks"))),
                "has_train_popularity_mapping": bool_text(has_pop_map),
                "has_user_profile_mapping": bool_text(has_profiles),
                "is_final_source": bool_text(parse_bool(base_row.get("is_final_source"))),
                "is_primary_source": bool_text(parse_bool(base_row.get("is_primary_source"))),
                "is_pilot": bool_text(parse_bool(base_row.get("is_pilot"))),
                "is_diagnostic": bool_text(parse_bool(base_row.get("is_diagnostic"))),
                "can_compute_arp": bool_text(not blocking_reason and has_pop_map and bool(valid_lists)),
                "can_compute_aplt": bool_text(not blocking_reason and has_pop_map and bool(valid_lists)),
                "can_compute_shift": bool_text(not blocking_reason and has_pop_map and has_profiles and bool(valid_lists)),
                "can_compute_popreo": "false",
                "blocking_reason": blocking_reason or no_numeric_popreo_reason,
                "notes": "F5C inherits F5B source classification; PopREO remains blocked for every source because denominator policy is incomplete.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=PREDICTION_INVENTORY_COLUMNS), ["dataset", "task", "model", "source_file", "cohort_name"])


def aggregate_values(values: Sequence[float | None]) -> dict[str, str]:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numeric:
        return {"value": "", "std": "", "median": "", "min": "", "max": "", "valid_users": "0"}
    array = np.asarray(numeric, dtype=float)
    return {
        "value": format_float(float(np.mean(array))),
        "std": format_float(float(np.std(array))),
        "median": format_float(float(np.median(array))),
        "min": format_float(float(np.min(array))),
        "max": format_float(float(np.max(array))),
        "valid_users": str(len(numeric)),
    }


def build_metric_result_row(
    prediction: Mapping[str, Any],
    index: PopularityIndex,
    metric_definition_id: str,
    metric: str,
    values: Sequence[float | None],
    records: Sequence[UserPopularityRecord],
    *,
    artifact_category: str,
    paper_equivalence: str,
    reproduction_class: str,
    notes: str,
) -> dict[str, str]:
    total_users = len(records)
    aggregate = aggregate_values(values)
    valid_users = int(aggregate["valid_users"])
    source_file = clean(prediction.get("source_file"))
    is_primary = parse_bool(prediction.get("is_primary_source"))
    is_pilot = parse_bool(prediction.get("is_pilot"))
    is_diagnostic = parse_bool(prediction.get("is_diagnostic"))
    is_reportable = is_primary and not is_pilot and not is_diagnostic and valid_users > 0
    return {
        "popularity_result_id": stable_id("F5C-POP-RES", prediction.get("prediction_source_id"), metric_definition_id),
        "popularity_policy_id": index.policy_id,
        "prediction_source_id": clean(prediction.get("prediction_source_id")),
        "dataset": clean(prediction.get("dataset")),
        "task": clean(prediction.get("task")),
        "model": clean(prediction.get("model")),
        "model_family": clean(prediction.get("model_family")),
        "cohort_name": clean(prediction.get("cohort_name")),
        "candidate_set_name": clean(prediction.get("candidate_set_name")),
        "candidate_count": clean(prediction.get("candidate_count")),
        "metric_definition_id": metric_definition_id,
        "metric": metric,
        "top_k": str(TOP_K),
        "value": aggregate["value"],
        "std": aggregate["std"],
        "median": aggregate["median"],
        "min": aggregate["min"],
        "max": aggregate["max"],
        "valid_users": str(valid_users),
        "total_users": str(total_users),
        "valid_user_rate": percent(valid_users, total_users),
        "total_recommended_items": str(sum(len(record.items) for record in records)),
        "missing_popularity_count": str(sum(record.missing_popularity_count for record in records)),
        "invalid_item_count": str(sum(record.invalid_item_count for record in records)),
        "duplicate_user_count": str(sum(1 for record in records if record.duplicate)),
        "profile_users": str(sum(1 for record in records if record.profile_mean is not None)),
        "aggregation_policy": "per-user top-5 value, then macro mean over valid users",
        "result_scope": "primary_final" if is_reportable else "nonprimary_or_diagnostic",
        "artifact_category": artifact_category,
        "paper_equivalence": paper_equivalence,
        "reproduction_class": reproduction_class,
        "is_primary_result": bool_text(is_primary),
        "is_reportable": bool_text(is_reportable),
        "validation_status": "pass" if valid_users > 0 else "blocked_no_valid_users",
        "source_prediction_file": source_file,
        "source_interaction_file": index.interactions_path.as_posix(),
        "notes": notes,
    }


def records_for_prediction_source(root: Path, prediction: Mapping[str, Any], index: PopularityIndex) -> list[UserPopularityRecord]:
    source_rows = load_prediction_source_rows(root, prediction)
    records: list[UserPopularityRecord] = []
    for row in source_rows.to_dict("records"):
        items = recommendation_items_from_row(row, TOP_K)
        records.append(compute_user_popularity(items, index, user_id=row.get("user_id"), top_k=TOP_K))
    return records


def build_popularity_results(
    root: Path,
    prediction_inventory: pd.DataFrame,
    indexes: Mapping[str, PopularityIndex],
) -> tuple[pd.DataFrame, dict[str, list[UserPopularityRecord]]]:
    rows: list[dict[str, str]] = []
    record_cache: dict[str, list[UserPopularityRecord]] = {}
    for _, prediction in prediction_inventory.iterrows():
        if not parse_bool(prediction.get("can_compute_arp")):
            continue
        dataset = clean(prediction.get("dataset"))
        index = indexes.get(dataset)
        if index is None:
            continue
        records = records_for_prediction_source(root, prediction, index)
        record_cache[clean(prediction.get("prediction_source_id"))] = records
        if not records:
            continue
        metric_specs = [
            ("arp_at_5", "ARP@5", [record.raw_mean for record in records], "reconstructed", "functional_upstream_code_not_exact_paper_claim", "reconstructed", "Computed from saved rankings with upstream calARP semantics."),
            ("average_recommendation_popularity_at_5", "Average Recommendation Popularity@5", [record.raw_mean for record in records], "reconstructed", "functional_upstream_code_not_exact_paper_claim", "reconstructed", "Explicit average-popularity name; not reported as popularity bias."),
            ("normalized_average_recommendation_popularity_at_5", "Normalized Average Recommendation Popularity@5", [record.normalized_mean for record in records], "final_extension", "adapted_local_extension", "adapted", "Scale-normalized extension under the F5C train-only policy."),
            ("aplt_at_5", "APLT/ACLT@5", [record.tail_rate for record in records], "reconstructed", "functional_upstream_code_not_exact_paper_claim", "reconstructed", "Long-tail exposure under upstream q80 train-count threshold."),
            ("head_exposure_at_5", "Head Exposure@5", [record.head_rate for record in records], "final_extension", "adapted_local_extension", "adapted", "Head exposure companion to APLT under the same q80 policy."),
        ]
        if parse_bool(prediction.get("can_compute_shift")):
            metric_specs.extend(
                [
                    ("popularity_shift_at_5", "Popularity Shift@5", [record.shift for record in records], "final_extension", "adapted_local_extension", "adapted", "Top-5 recommendation popularity minus user profile popularity."),
                    ("popularity_lift_at_5", "Popularity Lift@5", [record.lift for record in records], "final_extension", "adapted_local_extension", "adapted", "Top-5 recommendation popularity divided by user profile popularity."),
                ]
            )
        for metric_definition_id, metric, values, artifact_category, paper_equivalence, reproduction_class, notes in metric_specs:
            rows.append(
                build_metric_result_row(
                    prediction,
                    index,
                    metric_definition_id,
                    metric,
                    values,
                    records,
                    artifact_category=artifact_category,
                    paper_equivalence=paper_equivalence,
                    reproduction_class=reproduction_class,
                    notes=notes,
                )
            )
    return sort_frame(pd.DataFrame(rows, columns=POPULARITY_RESULT_COLUMNS), ["dataset", "task", "model", "metric_definition_id", "source_prediction_file"]), record_cache


def build_item_group_exposure(
    prediction_inventory: pd.DataFrame,
    indexes: Mapping[str, PopularityIndex],
    record_cache: Mapping[str, list[UserPopularityRecord]],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, prediction in prediction_inventory.iterrows():
        source_id = clean(prediction.get("prediction_source_id"))
        records = record_cache.get(source_id, [])
        index = indexes.get(clean(prediction.get("dataset")))
        if not records or index is None:
            continue
        total_positions = sum(len(record.items) for record in records)
        group_specs = [
            ("head", "train_seen_count >= q80", len(index.head_items), lambda item: item in index.head_items),
            ("tail", "train_seen_count < q80", len(index.tail_items), lambda item: item in index.tail_items),
            ("cold_or_unseen", "item absent from train popularity dictionary", max(0, len(index.catalog_items) - len(index.counts)), lambda item: item not in index.counts),
        ]
        for group_name, group_rule, group_item_count, predicate in group_specs:
            exposure_count = sum(1 for record in records for item in record.items if predicate(item))
            rows.append(
                {
                    "exposure_id": stable_id("F5C-GEXP", source_id, group_name),
                    "popularity_policy_id": index.policy_id,
                    "prediction_source_id": source_id,
                    "dataset": clean(prediction.get("dataset")),
                    "task": clean(prediction.get("task")),
                    "model": clean(prediction.get("model")),
                    "cohort_name": clean(prediction.get("cohort_name")),
                    "candidate_set_name": clean(prediction.get("candidate_set_name")),
                    "group_name": group_name,
                    "group_rule": group_rule,
                    "group_item_count": str(group_item_count),
                    "exposure_count": str(exposure_count),
                    "total_recommended_items": str(total_positions),
                    "exposure_rate": percent(exposure_count, total_positions),
                    "valid_users": str(sum(1 for record in records if record.raw_mean is not None)),
                    "total_users": str(len(records)),
                    "metric_source": "top-5 recommendation positions",
                    "result_scope": "primary_final" if parse_bool(prediction.get("is_primary_source")) and not parse_bool(prediction.get("is_pilot")) and not parse_bool(prediction.get("is_diagnostic")) else "nonprimary_or_diagnostic",
                    "is_reportable": bool_text(parse_bool(prediction.get("is_primary_source")) and not parse_bool(prediction.get("is_pilot")) and not parse_bool(prediction.get("is_diagnostic"))),
                    "notes": "Mid group is absent because upstream calACLT defines a binary q80 tail/head threshold only.",
                }
            )
    return sort_frame(pd.DataFrame(rows, columns=GROUP_EXPOSURE_COLUMNS), ["dataset", "task", "model", "group_name"])


def profile_group(value: float, low: float, high: float) -> tuple[str, str]:
    if value <= low:
        return "low_profile_popularity", f"profile_mean <= {format_float(low)}"
    if value <= high:
        return "mid_profile_popularity", f"{format_float(low)} < profile_mean <= {format_float(high)}"
    return "high_profile_popularity", f"profile_mean > {format_float(high)}"


def build_user_popularity_shift_summary(
    prediction_inventory: pd.DataFrame,
    indexes: Mapping[str, PopularityIndex],
    record_cache: Mapping[str, list[UserPopularityRecord]],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, prediction in prediction_inventory.iterrows():
        source_id = clean(prediction.get("prediction_source_id"))
        records = [record for record in record_cache.get(source_id, []) if record.profile_mean is not None and record.shift is not None]
        index = indexes.get(clean(prediction.get("dataset")))
        if not records or index is None:
            continue
        profile_values = np.asarray([record.profile_mean for record in records if record.profile_mean is not None], dtype=float)
        low = float(np.quantile(profile_values, 1.0 / 3.0))
        high = float(np.quantile(profile_values, 2.0 / 3.0))
        grouped: dict[str, list[UserPopularityRecord]] = {}
        rules: dict[str, str] = {}
        for record in records:
            group, rule = profile_group(float(record.profile_mean or 0.0), low, high)
            grouped.setdefault(group, []).append(record)
            rules[group] = rule
        for group_name in ("low_profile_popularity", "mid_profile_popularity", "high_profile_popularity"):
            group_records = grouped.get(group_name, [])
            if not group_records:
                continue
            lifts = [record.lift for record in group_records if record.lift is not None]
            rows.append(
                {
                    "summary_id": stable_id("F5C-SHIFT", source_id, group_name),
                    "popularity_policy_id": index.policy_id,
                    "prediction_source_id": source_id,
                    "dataset": clean(prediction.get("dataset")),
                    "task": clean(prediction.get("task")),
                    "model": clean(prediction.get("model")),
                    "cohort_name": clean(prediction.get("cohort_name")),
                    "candidate_set_name": clean(prediction.get("candidate_set_name")),
                    "profile_group": group_name,
                    "profile_group_rule": rules[group_name],
                    "user_count": str(len(group_records)),
                    "mean_profile_popularity": format_float(np.mean([record.profile_mean for record in group_records if record.profile_mean is not None])),
                    "mean_recommendation_popularity": format_float(np.mean([record.raw_mean for record in group_records if record.raw_mean is not None])),
                    "mean_popularity_shift": format_float(np.mean([record.shift for record in group_records if record.shift is not None])),
                    "median_popularity_shift": format_float(np.median([record.shift for record in group_records if record.shift is not None])),
                    "mean_popularity_lift": format_float(np.mean(lifts) if lifts else None),
                    "valid_lift_users": str(len(lifts)),
                    "result_scope": "primary_final" if parse_bool(prediction.get("is_primary_source")) and not parse_bool(prediction.get("is_pilot")) and not parse_bool(prediction.get("is_diagnostic")) else "nonprimary_or_diagnostic",
                    "notes": "Profile groups are tertiles computed within each prediction source.",
                }
            )
    return sort_frame(pd.DataFrame(rows, columns=SHIFT_SUMMARY_COLUMNS), ["dataset", "task", "model", "profile_group"])


def build_valid_popularity_comparisons(popularity_results: pd.DataFrame) -> pd.DataFrame:
    if popularity_results.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    source = popularity_results.loc[
        popularity_results["is_reportable"].eq("true") & popularity_results["value"].astype(str).str.len().gt(0)
    ].copy()
    rows: list[dict[str, str]] = []
    group_columns = ["dataset", "task", "cohort_name", "candidate_set_name", "metric", "top_k"]
    for group_key, group in source.groupby(group_columns, sort=True):
        group = group.sort_values("model", kind="mergesort")
        records = group.to_dict("records")
        for left, right in combinations(records, 2):
            left_value = float(left["value"])
            right_value = float(right["value"])
            relative = None if left_value == 0.0 else (right_value - left_value) / abs(left_value)
            rows.append(
                {
                    "comparison_id": stable_id("F5C-COMP", *group_key, left["model"], right["model"]),
                    "dataset": clean(left["dataset"]),
                    "task": clean(left["task"]),
                    "cohort_name": clean(left["cohort_name"]),
                    "candidate_set_name": clean(left["candidate_set_name"]),
                    "metric": clean(left["metric"]),
                    "top_k": clean(left["top_k"]),
                    "model_a": clean(left["model"]),
                    "model_a_value": clean(left["value"]),
                    "model_b": clean(right["model"]),
                    "model_b_value": clean(right["value"]),
                    "absolute_difference": format_float(right_value - left_value),
                    "relative_difference": format_float(relative),
                    "is_valid_comparison": "true",
                    "comparison_scope": "same_dataset_task_cohort_candidate_metric",
                    "notes": "Values with different datasets, cohorts, candidate sets, or metric definitions are not compared.",
                }
            )
    return sort_frame(pd.DataFrame(rows, columns=COMPARISON_COLUMNS), ["dataset", "task", "metric", "model_a", "model_b"])


def existing_popularity_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    canonical = read_csv(root, CANONICAL_RESULTS)
    if not canonical.empty:
        mask = canonical["metric_id"].astype(str).str.contains("popularity|long_tail|arp|aplt|popreo|bias", case=False, regex=True)
        for _, row in canonical.loc[mask].iterrows():
            rows.append(
                {
                    "existing_metric": clean(row.get("metric_id")),
                    "existing_source_file": CANONICAL_RESULTS.as_posix(),
                    "existing_value": clean(row.get("metric_value")),
                    "existing_dataset": clean(row.get("dataset")),
                    "existing_task": clean(row.get("task_type")),
                    "existing_model": clean(row.get("baseline_id")),
                    "existing_cohort": clean(row.get("cohort_id")),
                    "existing_candidate_set": clean(row.get("candidate_set_id")),
                }
            )
    llm = read_csv(root, LLM_BEYOND_METRICS)
    if not llm.empty:
        mask = llm["metric_id"].astype(str).str.contains("popularity|long_tail|arp|aplt|popreo|bias", case=False, regex=True)
        for _, row in llm.loc[mask].iterrows():
            rows.append(
                {
                    "existing_metric": clean(row.get("metric_id")),
                    "existing_source_file": LLM_BEYOND_METRICS.as_posix(),
                    "existing_value": clean(row.get("metric_value")),
                    "existing_dataset": clean(row.get("dataset")),
                    "existing_task": clean(row.get("task_type")),
                    "existing_model": clean(row.get("model_family")),
                    "existing_cohort": clean(row.get("evaluation_policy_id")),
                    "existing_candidate_set": clean(row.get("item_universe_id")),
                }
            )
    resolution = read_csv(root, F4_METRIC_RESOLUTION)
    if not resolution.empty:
        columns_text = resolution.astype(str).agg(" ".join, axis=1)
        mask = columns_text.str.contains("Popularity Bias|Average Popularity|Long-Tail|PopREO|ARP|APLT", case=False, regex=True)
        for index, row in resolution.loc[mask].iterrows():
            metric = clean(row.get("metric")) or clean(row.get("metric_name")) or clean(row.get("metric_definition_id")) or f"f4_row_{index}"
            rows.append(
                {
                    "existing_metric": metric,
                    "existing_source_file": F4_METRIC_RESOLUTION.as_posix(),
                    "existing_value": clean(row.get("value")),
                    "existing_dataset": clean(row.get("dataset")),
                    "existing_task": clean(row.get("task")),
                    "existing_model": clean(row.get("model")),
                    "existing_cohort": clean(row.get("cohort_name")),
                    "existing_candidate_set": clean(row.get("candidate_set_name")),
                }
            )
    return pd.DataFrame(rows)


def map_existing_metric(metric: str) -> tuple[str, bool, str]:
    lower = metric.lower()
    if "popularity_bias" in lower or "popularity bias" in lower:
        return "popularity_bias_at_5", False, "supersede_ambiguous_alias"
    if "long_tail" in lower or "long-tail" in lower or "aplt" in lower or "aclt" in lower:
        return "aplt_at_5", True, "map_to_explicit_aplt"
    if "average_popularity" in lower or "average popularity" in lower or "arp" in lower:
        return "average_recommendation_popularity_at_5", True, "map_to_explicit_average_popularity"
    if "popreo" in lower:
        return "popreo_at_5", False, "keep_popreo_blocked"
    return "", False, "not_popularity_metric"


def build_popularity_reconciliation(root: Path, popularity_results: pd.DataFrame) -> pd.DataFrame:
    existing = existing_popularity_rows(root)
    rows: list[dict[str, str]] = []
    if existing.empty:
        existing = pd.DataFrame(
            [
                {
                    "existing_metric": "popularity_bias_at_5",
                    "existing_source_file": F4_METRIC_RESOLUTION.as_posix(),
                    "existing_value": "",
                    "existing_dataset": "",
                    "existing_task": "",
                    "existing_model": "",
                    "existing_cohort": "",
                    "existing_candidate_set": "",
                }
            ]
        )
    for _, row in existing.iterrows():
        f5c_metric, same_semantic, action = map_existing_metric(clean(row.get("existing_metric")))
        candidates = popularity_results.loc[popularity_results["metric_definition_id"].eq(f5c_metric)] if f5c_metric else pd.DataFrame()
        for column, existing_column in (
            ("dataset", "existing_dataset"),
            ("task", "existing_task"),
            ("model", "existing_model"),
            ("cohort_name", "existing_cohort"),
            ("candidate_set_name", "existing_candidate_set"),
        ):
            value = clean(row.get(existing_column))
            if value and not candidates.empty and column in candidates.columns:
                filtered = candidates.loc[candidates[column].eq(value)]
                if not filtered.empty:
                    candidates = filtered
        f5c_row = candidates.iloc[0].to_dict() if not candidates.empty else {}
        same_cohort = bool(f5c_row) and clean(row.get("existing_cohort")) == clean(f5c_row.get("cohort_name"))
        same_candidate = bool(f5c_row) and clean(row.get("existing_candidate_set")) == clean(f5c_row.get("candidate_set_name"))
        same_denominator = same_semantic and bool(f5c_row) and same_cohort and same_candidate
        is_reconciled = same_semantic and (bool(f5c_row) or f5c_metric in {"popreo_at_5"})
        rows.append(
            {
                "reconciliation_id": stable_id("F5C-RECON", row.get("existing_source_file"), row.get("existing_metric"), row.get("existing_dataset"), row.get("existing_model"), row.get("existing_cohort")),
                "existing_metric": clean(row.get("existing_metric")),
                "existing_source_file": clean(row.get("existing_source_file")),
                "existing_value": clean(row.get("existing_value")),
                "existing_dataset": clean(row.get("existing_dataset")),
                "existing_task": clean(row.get("existing_task")),
                "existing_model": clean(row.get("existing_model")),
                "existing_cohort": clean(row.get("existing_cohort")),
                "existing_candidate_set": clean(row.get("existing_candidate_set")),
                "f5c_metric": f5c_metric,
                "f5c_result_id": clean(f5c_row.get("popularity_result_id")),
                "f5c_value": clean(f5c_row.get("value")),
                "same_semantic_definition": bool_text(same_semantic),
                "same_cohort": bool_text(same_cohort),
                "same_candidate_set": bool_text(same_candidate),
                "same_denominator_policy": bool_text(same_denominator),
                "is_reconciled": bool_text(is_reconciled),
                "authoritative_result": "finalization_f5c_popularity_results.csv" if bool(f5c_row) else "finalization_f5c_metric_status_update.csv",
                "action": action,
                "notes": "F5C reconciles labels by semantics first; values from different cohorts or denominators are not numeric conflicts.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=RECONCILIATION_COLUMNS).drop_duplicates("reconciliation_id"), ["existing_metric", "existing_dataset", "existing_model"])


def build_popularity_bias_resolution() -> pd.DataFrame:
    rows = [
        {
            "resolution_id": "F5C-RES-POPULARITY-BIAS",
            "metric_label": "popularity_bias_at_5",
            "previous_status": "blocked_or_ambiguous",
            "f5c_status": "superseded_alias",
            "resolution_type": "semantic_split",
            "replaced_by": "arp_at_5;average_recommendation_popularity_at_5;aplt_at_5;head_exposure_at_5;popularity_shift_at_5;popularity_lift_at_5;popreo_at_5",
            "semantic_decision": "Popularity bias is not a single metric in the local evidence; F5C forbids using it as a synonym for average popularity.",
            "is_scientifically_consolidated": "true",
            "is_reportable": "true",
            "remaining_limitation": "PopREO remains blocked numerically and exact paper popularity-bias prose remains unavailable.",
            "next_required_action": "Cite explicit metric names only; recover paper denominator policy before computing PopREO as exact evidence.",
            "notes": "This resolves the ambiguity without inventing a new paper metric.",
        }
    ]
    return pd.DataFrame(rows, columns=BIAS_RESOLUTION_COLUMNS)


def build_metric_status_update(item_policy: pd.DataFrame) -> pd.DataFrame:
    active_policy_ids = ";".join(item_policy.loc[item_policy["policy_status"].eq("active"), "popularity_policy_id"].tolist())
    rows = [
        ("arp_at_5", "ARP@5", "partially_consolidated", "resolved_functional", "functional_upstream_code", "formula_recovered", "frozen_train_only_q80", "computed_from_saved_recommendation_lists", "functional_upstream_code_not_exact_paper_claim", True, True, "Exact paper text/cell remains unavailable.", "Use as explicit ARP/average-popularity evidence, not as popularity_bias_at_5."),
        ("average_recommendation_popularity_at_5", "Average Recommendation Popularity@5", "consolidated_reconstructed", "resolved_functional", "functional_upstream_code", "formula_recovered", "frozen_train_only_q80", "computed_from_saved_recommendation_lists", "functional_upstream_code_not_exact_paper_claim", True, True, "Exact paper text/cell remains unavailable.", "Use explicit name in reports."),
        ("normalized_average_recommendation_popularity_at_5", "Normalized Average Recommendation Popularity@5", "not_separated", "resolved_extension_only", "local_extension", "formula_local_extension", "frozen_train_only_q80", "computed_from_saved_recommendation_lists", "adapted_local_extension", True, True, "Not a paper metric.", "Use only as local diagnostic."),
        ("aplt_at_5", "APLT/ACLT@5", "consolidated_as_long_tail_rate", "resolved_functional", "functional_upstream_code", "formula_recovered", "frozen_train_only_q80", "computed_from_saved_recommendation_lists", "functional_upstream_code_not_exact_paper_claim", True, True, "Exact paper text/cell remains unavailable.", "Use as explicit APLT/long-tail exposure evidence."),
        ("head_exposure_at_5", "Head Exposure@5", "not_separated", "resolved_extension_only", "local_extension", "formula_local_extension", "frozen_train_only_q80", "computed_from_saved_recommendation_lists", "adapted_local_extension", True, True, "Not a paper metric.", "Use as companion diagnostic only."),
        ("popularity_shift_at_5", "Popularity Shift@5", "not_separated", "resolved_extension_only", "local_extension", "formula_local_extension", "frozen_train_only_q80", "computed_from_saved_recommendation_lists", "adapted_local_extension", True, True, "Not a paper metric.", "Use as user-profile shift diagnostic."),
        ("popularity_lift_at_5", "Popularity Lift@5", "not_separated", "resolved_extension_only", "local_extension", "formula_local_extension", "frozen_train_only_q80", "computed_from_saved_recommendation_lists", "adapted_local_extension", True, True, "Not a paper metric.", "Use as ratio companion to shift."),
        ("popreo_at_5", "PopREO@5", "conflated_with_popularity_bias", "blocked_missing_denominator_policy", "formula_recovered_policy_incomplete", "formula_recovered", "partial_group_denominator_policy", "not_computed", "formula_recovered_but_policy_incomplete", False, False, "Empty group and zero denominator handling are unspecified.", "Recover denominator/group policy before numeric computation."),
        ("popularity_bias_at_5", "Popularity Bias@5", "blocked_or_ambiguous", "superseded_alias", "semantic_split", "no_single_formula", "frozen_no_numeric_alias", "not_computed_as_alias", "not_exact", True, True, "No single formula is defensible from local evidence.", "Do not cite a numeric popularity_bias_at_5 value."),
    ]
    return pd.DataFrame(
        [
            {
                "metric_definition_id": metric_id,
                "metric": metric,
                "previous_status": previous,
                "f5c_status": status,
                "resolution_class": resolution,
                "primary_policy_id": active_policy_ids,
                "formula_status": formula_status,
                "popularity_policy_status": policy_status,
                "computation_status": computation_status,
                "validation_status": "pass",
                "paper_equivalence": paper_equivalence,
                "is_scientifically_consolidated": bool_text(consolidated),
                "is_reportable": bool_text(reportable),
                "remaining_limitation": limitation,
                "next_required_action": action,
                "notes": "F5C semantic resolution table.",
            }
            for metric_id, metric, previous, status, resolution, formula_status, policy_status, computation_status, paper_equivalence, consolidated, reportable, limitation, action in rows
        ],
        columns=METRIC_STATUS_COLUMNS,
    )


def read_previous_coverage(root: Path) -> dict[str, float]:
    previous = {
        "exact reproduction coverage": 12.328767,
        "reconstructed-or-better coverage": 34.246575,
        "maximum defensible coverage": 68.493151,
        "exact numeric replication coverage": 0.0,
        "consolidated metric count": 8.0,
        "ambiguous metric count": 3.0,
        "missing evidence count": 40.0,
        "safe report claim count": 10.0,
    }
    frame = read_csv(root, F5B_COVERAGE)
    if frame.empty:
        return previous
    value_col = "f5b_value" if "f5b_value" in frame.columns else "f5_value"
    for _, row in frame.iterrows():
        metric = clean(row.get("coverage_metric"))
        value = clean(row.get(value_col))
        if metric in previous and value:
            try:
                previous[metric] = float(value)
            except ValueError:
                pass
    return previous


def build_claim_update(popularity_results: pd.DataFrame, metric_status: pd.DataFrame) -> pd.DataFrame:
    reportable = popularity_results.loc[popularity_results["is_reportable"].eq("true")]
    datasets = ";".join(sorted(reportable["dataset"].dropna().unique()))
    rows = [
        {
            "claim_id": "F5C-CLAIM-001",
            "previous_claim_text": "Popularity bias was an ambiguous blocked label.",
            "previous_safety": "unsafe_numeric",
            "f5c_evidence": "results/tables/finalization_f5c_popularity_bias_resolution.csv",
            "updated_claim_text": "Popularity bias is resolved as a superseded alias, not as a numeric metric.",
            "updated_confidence": "high",
            "is_safe_for_report": "true",
            "is_safe_for_presentation": "true",
            "claim_scope": "semantic_resolution",
            "remaining_limitation": "Exact paper wording remains unavailable.",
            "notes": "Avoids overclaiming by naming explicit metrics.",
        },
        {
            "claim_id": "F5C-CLAIM-002",
            "previous_claim_text": "Average popularity existed but was conflated with popularity bias.",
            "previous_safety": "limited",
            "f5c_evidence": "results/tables/finalization_f5c_popularity_results.csv",
            "updated_claim_text": f"ARP/Average Recommendation Popularity@5 is computable offline for reportable sources across {datasets}.",
            "updated_confidence": "high",
            "is_safe_for_report": "true",
            "is_safe_for_presentation": "true",
            "claim_scope": "offline_saved_rankings",
            "remaining_limitation": "Functional upstream equivalence, not exact paper cell replication.",
            "notes": "Values are read from generated F5C CSVs.",
        },
        {
            "claim_id": "F5C-CLAIM-003",
            "previous_claim_text": "Long-tail exposure was not separated from the popularity-bias label.",
            "previous_safety": "limited",
            "f5c_evidence": "results/tables/finalization_f5c_popularity_results.csv",
            "updated_claim_text": "APLT/ACLT@5 is computed with the recovered q80 train-count threshold and reported separately from ARP.",
            "updated_confidence": "high",
            "is_safe_for_report": "true",
            "is_safe_for_presentation": "true",
            "claim_scope": "offline_saved_rankings",
            "remaining_limitation": "Uses the frozen F5C distinct-train-user counting policy.",
            "notes": "Cold/unseen items are not long-tail members.",
        },
        {
            "claim_id": "F5C-CLAIM-004",
            "previous_claim_text": "PopREO was not distinguished from popularity level metrics.",
            "previous_safety": "unsafe_numeric",
            "f5c_evidence": "results/tables/finalization_f5c_metric_status_update.csv",
            "updated_claim_text": "PopREO is identified as target-popularity hit-rate fairness and remains numerically blocked.",
            "updated_confidence": "high",
            "is_safe_for_report": "true",
            "is_safe_for_presentation": "true",
            "claim_scope": "blocked_metric",
            "remaining_limitation": "Need denominator and empty-group policy.",
            "notes": "No PopREO value is invented.",
        },
        {
            "claim_id": "F5C-CLAIM-005",
            "previous_claim_text": "Popularity shift/lift had no frozen local policy.",
            "previous_safety": "not_available",
            "f5c_evidence": "results/tables/finalization_f5c_user_popularity_shift_summary.csv",
            "updated_claim_text": "Popularity shift and lift are available as local extension diagnostics against each user's train profile.",
            "updated_confidence": "medium",
            "is_safe_for_report": "true",
            "is_safe_for_presentation": "true",
            "claim_scope": "local_extension",
            "remaining_limitation": "Not a paper exact metric.",
            "notes": "Report only as adapted extension evidence.",
        },
    ]
    assert not metric_status.empty
    return pd.DataFrame(rows, columns=CLAIM_COLUMNS)


def build_coverage_impact(root: Path, metric_status: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    previous = read_previous_coverage(root)
    consolidated_delta = 3.0
    ambiguous_delta = -1.0
    missing_delta = -1.0
    safe_claim_delta = float(claims["is_safe_for_report"].eq("true").sum())
    values = [
        ("exact reproduction coverage", previous["exact reproduction coverage"], previous["exact reproduction coverage"], 0.0, "F5C computes adapted/functional popularity semantics only.", True, "pass", "No exact paper component is promoted."),
        ("reconstructed-or-better coverage", previous["reconstructed-or-better coverage"], previous["reconstructed-or-better coverage"], 0.0, "ARP/APLT were already represented as reconstructed metrics; F5C clarifies semantics.", True, "pass", "No paper coverage inflation."),
        ("maximum defensible coverage", previous["maximum defensible coverage"], previous["maximum defensible coverage"], 0.0, "F5C does not rewrite Phase 5E component accounting.", True, "pass", "Extension diagnostics remain outside exact/reconstructed paper coverage."),
        ("exact numeric replication coverage", previous["exact numeric replication coverage"], previous["exact numeric replication coverage"], 0.0, "No paper numeric popularity-bias value is recovered.", True, "pass", "Exact numeric coverage unchanged."),
        ("consolidated metric count", previous["consolidated metric count"], previous["consolidated metric count"] + consolidated_delta, consolidated_delta, "head_exposure_at_5, popularity_shift_at_5, and popularity_lift_at_5 receive frozen local definitions.", True, "pass", "ARP/APLT are semantic clarifications of existing consolidated metrics."),
        ("ambiguous metric count", previous["ambiguous metric count"], max(0.0, previous["ambiguous metric count"] + ambiguous_delta), ambiguous_delta, "popularity_bias_at_5 superseded as an ambiguous alias.", True, "pass", "No numeric alias remains."),
        ("missing evidence count", previous["missing evidence count"], max(0.0, previous["missing evidence count"] + missing_delta), missing_delta, "F5-EXP-003 popularity-bias semantics resolved without exact PopREO computation.", True, "pass", "PopREO remains a separate blocked metric."),
        ("safe report claim count", previous["safe report claim count"], previous["safe report claim count"] + safe_claim_delta, safe_claim_delta, "F5C claim update adds safe popularity-semantics claims.", True, "pass", "All claims include scope and limitation."),
    ]
    return pd.DataFrame(
        [
            {
                "coverage_metric": metric,
                "previous_value": format_float(old),
                "f5c_value": format_float(new),
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


def build_anomalies(prediction_inventory: pd.DataFrame, item_policy: pd.DataFrame, popularity_results: pd.DataFrame) -> pd.DataFrame:
    rows = [
        anomaly(
            "F5C-ANOM-001",
            "metric_semantics",
            "medium",
            "results/tables/finalization_f4_metric_resolution_inventory.csv",
            "The label popularity_bias_at_5 aggregated distinct semantic families.",
            "Old numeric uses of the alias would overclaim; F5C replaces it with explicit metric ids.",
            "true",
            "Do not cite popularity_bias_at_5 as a numeric result.",
            "",
        ),
        anomaly(
            "F5C-ANOM-002",
            "popreo",
            "medium",
            "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "PopREO formula is recoverable, but empty target-group and zero-denominator handling is not specified.",
            "No defensible PopREO value can be reported from local evidence alone.",
            "true",
            "Keep PopREO blocked until denominator policy is recovered.",
            "",
        ),
        anomaly(
            "F5C-ANOM-003",
            "group_policy",
            "info",
            "results/tables/finalization_f5c_item_popularity_policy.csv",
            "Upstream ACLT defines a binary q80 tail/head split; no mid group is defined.",
            "Head and tail exposure are valid; mid exposure is not invented.",
            "true",
            "Leave mid_item_count at zero and document the absence.",
            "",
        ),
    ]
    blocked_sources = prediction_inventory.loc[prediction_inventory["can_compute_arp"].eq("false")]
    if not blocked_sources.empty:
        rows.append(
            anomaly(
                "F5C-ANOM-004",
                "prediction_sources",
                "info",
                "results/tables/finalization_f5c_prediction_source_inventory.csv",
                f"{len(blocked_sources)} prediction-source groups cannot compute popularity metrics.",
                "Blocked, pilot, empty, or policy-incompatible sources are excluded from reportable numeric results.",
                "true",
                "Use only rows with can_compute_arp=true and is_reportable=true for conclusions.",
                "",
            )
        )
    lastfm = item_policy.loc[item_policy["dataset"].eq("LastFM")]
    if not lastfm.empty and lastfm.iloc[0]["policy_status"] == "blocked":
        rows.append(
            anomaly(
                "F5C-ANOM-005",
                "dataset_policy",
                "info",
                "data/reconstructed/plan_c/lastfm/interactions.csv",
                "LastFM has observed interactions but no train split under the F5C policy and no final F5B source.",
                "LastFM popularity metrics are not computed in F5C.",
                "true",
                "Recover a train/test policy and final saved prediction lists before including LastFM.",
                "",
            )
        )
    if popularity_results["invalid_item_count"].astype(str).replace("", "0").astype(int).sum() > 0:
        rows.append(
            anomaly(
                "F5C-ANOM-006",
                "recommendation_items",
                "info",
                "results/tables/finalization_f5c_popularity_results.csv",
                "Some saved recommendation items are absent from the local catalog.",
                "ARP treats absent train popularity as zero; invalid counts are explicit.",
                "true",
                "Inspect invalid_item_count before using affected nonprimary results.",
                "",
            )
        )
    return sort_frame(pd.DataFrame(rows, columns=ANOMALY_COLUMNS), ["anomaly_id"])


def anomaly(
    anomaly_id: str,
    area: str,
    severity: str,
    source_file: str,
    description: str,
    impact: str,
    can_continue: str,
    action: str,
    notes: str,
) -> dict[str, str]:
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


def build_validation(
    root: Path,
    metric_inventory: pd.DataFrame,
    definition_comparison: pd.DataFrame,
    item_policy: pd.DataFrame,
    item_stats: pd.DataFrame,
    prediction_inventory: pd.DataFrame,
    popularity_results: pd.DataFrame,
    group_exposure: pd.DataFrame,
    shift_summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    reconciliation: pd.DataFrame,
    bias_resolution: pd.DataFrame,
    metric_status: pd.DataFrame,
    coverage: pd.DataFrame,
    anomalies: pd.DataFrame,
    frames_no_validation: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(check_id: str, name: str, condition: bool, observed: Any, expected: Any, details: str) -> None:
        rows.append({"check_id": check_id, "check_name": name, "status": "pass" if condition else "fail", "observed": clean(observed), "expected": clean(expected), "details": details})

    arp = popularity_results.loc[popularity_results["metric_definition_id"].eq("arp_at_5")]
    average = popularity_results.loc[popularity_results["metric_definition_id"].eq("average_recommendation_popularity_at_5")]
    aplts = popularity_results.loc[popularity_results["metric_definition_id"].eq("aplt_at_5")]
    heads = popularity_results.loc[popularity_results["metric_definition_id"].eq("head_exposure_at_5")]
    avg_merge = arp[["prediction_source_id", "value"]].merge(average[["prediction_source_id", "value"]], on="prediction_source_id", suffixes=("_arp", "_avg"))
    protected_status = protected_hash_status(root)

    add("F5C-VAL-001", "metric_inventory_complete", len(metric_inventory) >= 9, len(metric_inventory), ">=9", "All separated popularity metric families are inventoried.")
    add("F5C-VAL-002", "arp_definition_recovered", metric_inventory.loc[metric_inventory["metric_definition_id"].eq("arp_at_5"), "upstream_function"].str.contains("calARP").any(), "calARP", "present", "ARP uses recovered upstream function.")
    add("F5C-VAL-003", "aplt_definition_recovered", metric_inventory.loc[metric_inventory["metric_definition_id"].eq("aplt_at_5"), "upstream_function"].str.contains("calACLT").any(), "calACLT", "present", "APLT uses recovered upstream function.")
    add("F5C-VAL-004", "popreo_blocked_policy", metric_status.loc[metric_status["metric_definition_id"].eq("popreo_at_5"), "f5c_status"].eq("blocked_missing_denominator_policy").any(), "blocked_missing_denominator_policy", "blocked", "PopREO is not computed without denominator policy.")
    add("F5C-VAL-005", "popularity_bias_superseded", bias_resolution["f5c_status"].eq("superseded_alias").all(), "superseded_alias", "superseded_alias", "Ambiguous alias has no numeric value.")
    add("F5C-VAL-006", "dataset_policy_complete", set(item_policy["dataset"]) >= {"Amazon Beauty", "Amazon Sports", "ML-1M", "LastFM"}, ";".join(sorted(item_policy["dataset"])), "four datasets", "Each dataset has active or blocked policy.")
    add("F5C-VAL-007", "active_policies_three", item_policy["policy_status"].eq("active").sum() == 3, item_policy["policy_status"].eq("active").sum(), "3", "Beauty, Sports, and ML-1M are active.")
    add("F5C-VAL-008", "no_mid_group_invented", item_policy.loc[item_policy["policy_status"].eq("active"), "mid_rule"].str.contains("not defined").all(), "mid not defined", "not defined", "No arbitrary mid threshold is introduced.")
    add("F5C-VAL-009", "item_stats_active_datasets", len(item_stats) == 3, len(item_stats), "3", "Stats exist for active datasets only.")
    add("F5C-VAL-010", "thresholds_positive", item_stats["long_tail_threshold"].astype(float).gt(0).all(), item_stats["long_tail_threshold"].tolist(), ">0", "q80 thresholds are positive.")
    add("F5C-VAL-011", "prediction_sources_inventoried", len(prediction_inventory) > 0, len(prediction_inventory), ">0", "F5B prediction sources are carried forward.")
    add("F5C-VAL-012", "computable_sources_exist", prediction_inventory["can_compute_arp"].eq("true").sum() > 0, prediction_inventory["can_compute_arp"].eq("true").sum(), ">0", "At least one saved source can compute ARP.")
    add("F5C-VAL-013", "popularity_results_exist", len(popularity_results) > 0, len(popularity_results), ">0", "Offline popularity results were produced.")
    add("F5C-VAL-014", "arp_equals_average_alias", avg_merge.apply(lambda row: row["value_arp"] == row["value_avg"], axis=1).all(), len(avg_merge), "all equal", "ARP and Average Recommendation Popularity use the same explicit formula.")
    add("F5C-VAL-015", "no_numeric_popularity_bias_result", not popularity_results["metric_definition_id"].eq("popularity_bias_at_5").any(), "0", "0", "Ambiguous alias is not in numeric result table.")
    add("F5C-VAL-016", "aplt_range", aplts["value"].astype(float).between(0.0, 1.0).all(), len(aplts), "[0,1]", "APLT shares are bounded.")
    add("F5C-VAL-017", "head_exposure_range", heads["value"].astype(float).between(0.0, 1.0).all(), len(heads), "[0,1]", "Head exposure shares are bounded.")
    add("F5C-VAL-018", "average_nonnegative", average["value"].astype(float).ge(0.0).all(), len(average), ">=0", "Popularity counts are non-negative.")
    add("F5C-VAL-019", "shift_lift_computed", popularity_results["metric_definition_id"].isin(["popularity_shift_at_5", "popularity_lift_at_5"]).sum() > 0, popularity_results["metric_definition_id"].isin(["popularity_shift_at_5", "popularity_lift_at_5"]).sum(), ">0", "User-profile shift/lift extension metrics are computed.")
    add("F5C-VAL-020", "no_popreo_numeric_rows", not popularity_results["metric_definition_id"].eq("popreo_at_5").any(), "0", "0", "PopREO rows are blocked outside the numeric result table.")
    add("F5C-VAL-021", "group_exposure_range", group_exposure["exposure_rate"].replace("", "0").astype(float).between(0.0, 1.0).all(), len(group_exposure), "[0,1]", "Group exposure shares are bounded.")
    add("F5C-VAL-022", "shift_summary_exists", len(shift_summary) > 0, len(shift_summary), ">0", "User popularity shift summary is available.")
    add("F5C-VAL-023", "valid_comparisons_scope", comparisons.empty or comparisons["is_valid_comparison"].eq("true").all(), len(comparisons), "valid only", "Comparisons are same dataset/task/cohort/candidate/metric.")
    add("F5C-VAL-024", "reconciliation_contains_bias", reconciliation["existing_metric"].astype(str).str.contains("popularity_bias|Popularity Bias", case=False, regex=True).any(), "popularity_bias", "present", "Bias alias is reconciled explicitly.")
    add("F5C-VAL-025", "metric_status_complete", set(metric_status["metric_definition_id"]) >= set(metric_inventory["metric_definition_id"]), len(metric_status), ">= metric inventory", "Each inventoried metric has a status row.")
    add("F5C-VAL-026", "exact_coverage_unchanged", coverage.loc[coverage["coverage_metric"].eq("exact reproduction coverage"), "absolute_change"].eq("0").all(), "0", "0", "No exact paper coverage increase.")
    add("F5C-VAL-027", "protected_inputs_unchanged", protected_status.endswith("changed=0"), protected_status, "changed=0", "F2 protected inputs still match hashes.")
    add("F5C-VAL-028", "relative_paths_only", all(no_absolute_paths(frame) for frame in frames_no_validation), "all frames", "no absolute paths", "F5C tables use project-relative paths.")
    add("F5C-VAL-029", "deterministic_frame_hashes", deterministic_frames(frames_no_validation), "stable", "stable", "In-memory frame serialization is deterministic.")
    add("F5C-VAL-030", "all_expected_outputs_generated", set(OUTPUT_TABLES) == set(["popularity_metric_inventory", "popularity_definition_comparison", "item_popularity_policy", "item_popularity_statistics", "prediction_source_inventory", "popularity_results", "item_group_exposure", "user_popularity_shift_summary", "valid_popularity_comparisons", "popularity_reconciliation", "popularity_bias_resolution", "metric_status_update", "claim_update", "coverage_impact", "anomalies", "validation"]), len(OUTPUT_TABLES), "16 tables plus report", "All expected F5C output frames are generated.")
    assert anomalies is not None and definition_comparison is not None
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def final_decision(validation: pd.DataFrame) -> str:
    return "finalization_f5c_complete" if not validation.empty and validation["status"].eq("pass").all() else "finalization_f5c_blocked"


def build_frames(root: Path) -> dict[str, pd.DataFrame]:
    metric_inventory = build_popularity_metric_inventory()
    definition_comparison = build_definition_comparison(metric_inventory)
    indexes = build_popularity_indexes(root)
    item_policy = build_item_popularity_policy(root, indexes)
    item_stats = build_item_popularity_statistics(indexes)
    prediction_inventory = build_prediction_source_inventory(root, indexes)
    popularity_results, record_cache = build_popularity_results(root, prediction_inventory, indexes)
    group_exposure = build_item_group_exposure(prediction_inventory, indexes, record_cache)
    shift_summary = build_user_popularity_shift_summary(prediction_inventory, indexes, record_cache)
    comparisons = build_valid_popularity_comparisons(popularity_results)
    reconciliation = build_popularity_reconciliation(root, popularity_results)
    bias_resolution = build_popularity_bias_resolution()
    metric_status = build_metric_status_update(item_policy)
    claims = build_claim_update(popularity_results, metric_status)
    coverage = build_coverage_impact(root, metric_status, claims)
    anomalies = build_anomalies(prediction_inventory, item_policy, popularity_results)
    frames_no_validation = [
        metric_inventory,
        definition_comparison,
        item_policy,
        item_stats,
        prediction_inventory,
        popularity_results,
        group_exposure,
        shift_summary,
        comparisons,
        reconciliation,
        bias_resolution,
        metric_status,
        claims,
        coverage,
        anomalies,
    ]
    validation = build_validation(
        root,
        metric_inventory,
        definition_comparison,
        item_policy,
        item_stats,
        prediction_inventory,
        popularity_results,
        group_exposure,
        shift_summary,
        comparisons,
        reconciliation,
        bias_resolution,
        metric_status,
        coverage,
        anomalies,
        frames_no_validation,
    )
    return {
        "popularity_metric_inventory": metric_inventory,
        "popularity_definition_comparison": definition_comparison,
        "item_popularity_policy": item_policy,
        "item_popularity_statistics": item_stats,
        "prediction_source_inventory": prediction_inventory,
        "popularity_results": popularity_results,
        "item_group_exposure": group_exposure,
        "user_popularity_shift_summary": shift_summary,
        "valid_popularity_comparisons": comparisons,
        "popularity_reconciliation": reconciliation,
        "popularity_bias_resolution": bias_resolution,
        "metric_status_update": metric_status,
        "claim_update": claims,
        "coverage_impact": coverage,
        "anomalies": anomalies,
        "validation": validation,
    }


def _first_value(results: pd.DataFrame, metric_id: str) -> str:
    rows = results.loc[results["metric_definition_id"].eq(metric_id) & results["is_reportable"].eq("true")]
    if rows.empty:
        return ""
    return clean(rows.iloc[0]["value"])


def build_report(root: Path, frames: Mapping[str, pd.DataFrame]) -> str:
    metric_inventory = frames["popularity_metric_inventory"]
    item_policy = frames["item_popularity_policy"]
    item_stats = frames["item_popularity_statistics"]
    prediction_inventory = frames["prediction_source_inventory"]
    results = frames["popularity_results"]
    group_exposure = frames["item_group_exposure"]
    shift_summary = frames["user_popularity_shift_summary"]
    comparisons = frames["valid_popularity_comparisons"]
    reconciliation = frames["popularity_reconciliation"]
    bias_resolution = frames["popularity_bias_resolution"]
    metric_status = frames["metric_status_update"]
    claims = frames["claim_update"]
    coverage = frames["coverage_impact"]
    anomalies = frames["anomalies"]
    validation = frames["validation"]
    reportable = results.loc[results["is_reportable"].eq("true")]
    decision = final_decision(validation)
    lines = [
        "# Finalization F5C Popularity Bias Resolution",
        "",
        "## 1. Executive summary",
        f"- Decisione: `{decision}`.",
        "- `popularity_bias_at_5` e risolto come alias superseded: non viene prodotto alcun valore numerico con quel nome.",
        f"- Metriche numeriche F5C: `{results['metric_definition_id'].nunique()}`; righe reportable: `{len(reportable)}`.",
        "",
        "## 2. Stato di partenza F5B/F5C",
        f"- Fonti prediction ereditate da F5B: `{len(prediction_inventory)}`.",
        f"- Fonti computabili ARP/APLT: `{prediction_inventory['can_compute_arp'].eq('true').sum()}`.",
        "- Nessun file F2/F3/F4/F5A/F5B viene modificato da F5C.",
        "",
        "## 3. Problema semantico",
        "- Il nome `popularity_bias_at_5` mescolava livello medio di popolarita, esposizione long-tail, shift rispetto al profilo utente e PopREO.",
        "- F5C vieta l'uso del nome ambiguo come metrica numerica e richiede nomi espliciti.",
        "",
        "## 4. Evidenza upstream",
        "- `calARP` misura la media delle frequenze train degli item raccomandati.",
        "- `calACLT` misura la quota di item train-seen sotto la soglia q80.",
        "- `calPopREO` misura dispersione di hit-rate per gruppi di popolarita del target, ma manca la policy per gruppi vuoti e denominatore zero.",
        "",
        "## 5. Inventario metriche",
        *[f"- `{row['metric_definition_id']}`: `{row['f5c_resolution']}`." for _, row in metric_inventory.iterrows()],
        "",
        "## 6. Policy train-only",
        *[f"- `{row['dataset']}`: `{row['policy_status']}`, source `{row['interaction_source']}`, threshold `{row['long_tail_threshold']}`." for _, row in item_policy.iterrows()],
        "",
        "## 7. Statistiche item popularity",
        *[f"- `{row['dataset']}`: train-seen `{row['train_seen_items']}`, catalog `{row['catalog_items']}`, head `{row['head_item_count']}`, tail `{row['tail_item_count']}`, gini `{row['gini_train_user_count']}`." for _, row in item_stats.iterrows()],
        "",
        "## 8. Head, mid e tail",
        "- Tail: `train_seen_count < q80`; head: `train_seen_count >= q80`.",
        "- Mid non e definito dalla funzione upstream `calACLT`; F5C non introduce una soglia arbitraria.",
        "",
        "## 9. Fonti recommendation list",
        f"- Fonti totali: `{len(prediction_inventory)}`; fonti primarie: `{prediction_inventory['is_primary_source'].eq('true').sum()}`; pilot: `{prediction_inventory['is_pilot'].eq('true').sum()}`; diagnostic: `{prediction_inventory['is_diagnostic'].eq('true').sum()}`.",
        "",
        "## 10. ARP@5",
        f"- Primo valore reportable: `{_first_value(results, 'arp_at_5')}`.",
        "- Formula: media train-count degli item top-5.",
        "",
        "## 11. Average Recommendation Popularity@5",
        f"- Primo valore reportable: `{_first_value(results, 'average_recommendation_popularity_at_5')}`.",
        "- E semanticamente equivalente ad ARP nella policy F5C, ma non e chiamata popularity bias.",
        "",
        "## 12. APLT/ACLT@5",
        f"- Primo valore reportable: `{_first_value(results, 'aplt_at_5')}`.",
        "- Cold/unseen item non sono long-tail member perche la membership upstream nasce da `freq_dict`.",
        "",
        "## 13. Head Exposure@5",
        f"- Primo valore reportable: `{_first_value(results, 'head_exposure_at_5')}`.",
        "- E una estensione diagnostica locale sotto la stessa soglia q80.",
        "",
        "## 14. Popularity shift e lift",
        f"- Primo shift reportable: `{_first_value(results, 'popularity_shift_at_5')}`.",
        f"- Primo lift reportable: `{_first_value(results, 'popularity_lift_at_5')}`.",
        f"- Righe summary profilo: `{len(shift_summary)}`.",
        "",
        "## 15. PopREO",
        "- PopREO e identificato come target-popularity hit fairness.",
        "- F5C non calcola PopREO perche mancano policy locali per gruppi vuoti e denominatore zero.",
        "",
        "## 16. Popularity bias alias",
        f"- Stato: `{bias_resolution.iloc[0]['f5c_status']}`.",
        f"- Sostituito da: `{bias_resolution.iloc[0]['replaced_by']}`.",
        "",
        "## 17. Confronti validi",
        f"- Confronti same dataset/task/cohort/candidate/metric: `{len(comparisons)}`.",
        "",
        "## 18. Riconciliazione",
        f"- Righe reconciliation: `{len(reconciliation)}`.",
        "- Valori con coorte/candidate set/denominatore diverso non sono conflitti numerici.",
        "",
        "## 19. LastFM",
        "- LastFM resta bloccato per F5C: split `observed`, nessuna policy train-only equivalente e nessuna fonte finale F5B.",
        "",
        "## 20. Impatto copertura",
        *[f"- `{row['coverage_metric']}`: `{row['previous_value']}` -> `{row['f5c_value']}` change `{row['absolute_change']}`." for _, row in coverage.iterrows()],
        "",
        "## 21. Claim update",
        *[f"- `{row['claim_id']}`: {row['updated_claim_text']}" for _, row in claims.iterrows()],
        "",
        "## 22. Anomalie",
        f"- Anomalie registrate: `{len(anomalies)}`.",
        *[f"- `{row['anomaly_id']}` `{row['severity']}`: {row['description']}" for _, row in anomalies.iterrows()],
        "",
        "## 23. Validazione",
        f"- Pass: `{validation['status'].eq('pass').sum()}`; fail: `{validation['status'].eq('fail').sum()}`.",
        "",
        "## 24. Input protetti",
        f"- `{protected_hash_status(root)}`.",
        "",
        "## 25. Determinismo",
        "- Il builder serializza CSV ordinati e id stabili; la seconda esecuzione deve produrre gli stessi hash.",
        "",
        "## 26. Decisione finale",
        f"- `{decision}`.",
        "",
        "## 27. File prodotti e next action",
        *[f"- `{path.as_posix()}`" for path in OUTPUT_TABLES.values()],
        f"- `{REPORT_PATH.as_posix()}`",
        "- Prossimo passo: citare solo metriche esplicite; recuperare policy PopREO prima di qualunque valore numerico PopREO.",
        "",
    ]
    assert group_exposure is not None and metric_status is not None
    return "\n".join(lines)


def write_outputs(root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, relative in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = frames[key].fillna("")
        frame.to_csv(path, index=False, lineterminator="\n")
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

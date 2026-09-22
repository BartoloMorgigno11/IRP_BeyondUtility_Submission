"""Validation helpers and schemas for Phase 4B canonical metric results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.audit import common_validation


MANIFEST_COLUMNS = [
    "computation_id",
    "baseline_id",
    "baseline_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "ranking_source_path",
    "ranking_source_sha256",
    "contract_path",
    "contract_registry_sha256",
    "same_cohort_preflight_path",
    "expected_rows",
    "observed_rows",
    "candidate_count",
    "top_k",
    "popularity_source_id",
    "popularity_source_sha256",
    "catalog_universe_id",
    "candidate_union_sha256",
    "long_tail_policy_id",
    "long_tail_threshold",
    "execution_environment",
    "computation_status",
    "blocking_reason",
    "notes",
]

TRAIN_ITEM_COLUMNS = [
    "dataset",
    "item_id",
    "train_interaction_count",
    "train_unique_user_count",
    "total_train_interactions",
    "popularity_probability",
    "self_information_bits",
    "popularity_rank_descending",
    "popularity_quantile",
    "long_tail_threshold",
    "long_tail_member",
    "train_seen_catalog_member",
    "strict_full_candidate_union_member",
    "strict_matched_candidate_union_member",
    "full_plan_e2_candidate_union_member",
    "source_path",
    "source_sha256",
    "popularity_policy_id",
    "long_tail_policy_id",
    "notes",
]

RANKING_ROW_COLUMNS = [
    "baseline_id",
    "baseline_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "positive_item_id",
    "candidate_items",
    "candidate_scores",
    "ranked_items",
    "ranked_scores",
    "positive_rank_zero_based",
    "positive_rank_one_based",
    "eligible_strict_full",
    "eligible_strict_matched",
    "full_cohort_id",
    "matched_cohort_id",
    "ranking_source_path",
    "ranking_source_row_key",
    "ranking_row_valid",
    "notes",
]

TOP5_COLUMNS = [
    "baseline_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "cohort_full_id",
    "eligible_strict_matched",
    "rank_position_one_based",
    "item_id",
    "item_score",
    "is_positive_item",
    "train_interaction_count",
    "train_unique_user_count",
    "popularity_probability",
    "self_information_bits",
    "long_tail_member",
    "item_in_candidate_list",
    "item_in_train_seen_catalog",
    "valid_item",
    "duplicate_within_top_5",
    "source_ranking_path",
    "notes",
]

ROW_CONTRIBUTION_COLUMNS = [
    "baseline_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "positive_item_id",
    "positive_rank_zero_based",
    "eligible_strict_matched",
    "hr_at_5",
    "ndcg_at_5",
    "novelty_at_5",
    "long_tail_rate_at_5",
    "average_popularity_at_5",
    "duplicate_rate_at_5",
    "candidate_validity_rate",
    "top5_unique_item_count",
    "top5_valid_item_count",
    "top5_items",
    "top5_train_popularity_counts",
    "top5_self_information_bits",
    "top5_long_tail_membership",
    "row_valid",
    "notes",
]

COVERAGE_COLUMNS = [
    "baseline_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "evaluated_rows",
    "top5_position_count",
    "unique_top5_item_count",
    "train_seen_catalog_item_count",
    "strict_candidate_union_item_count",
    "catalog_coverage_at_5",
    "item_coverage_at_5",
    "top5_item_union_sha256",
    "train_seen_catalog_sha256",
    "strict_candidate_union_sha256",
    "candidate_union_identical_across_baselines",
    "coverage_validation_status",
    "notes",
]

LONG_COLUMNS = [
    "result_id",
    "baseline_id",
    "baseline_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "metric_id",
    "metric_family",
    "cutoff",
    "metric_value",
    "numerator",
    "denominator",
    "evaluated_rows",
    "aggregation_level",
    "higher_is_better",
    "contract_status",
    "computation_status",
    "paper_alignment",
    "contract_sha256",
    "ranking_source_path",
    "supporting_output_path",
    "blocking_reason",
    "notes",
]

WIDE_COLUMNS = [
    "baseline_id",
    "baseline_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "evaluated_rows",
    "hr_at_5",
    "ndcg_at_5",
    "mean_positive_rank_zero_based",
    "median_positive_rank_zero_based",
    "novelty_at_5",
    "catalog_coverage_at_5",
    "item_coverage_at_5",
    "long_tail_rate_at_5",
    "average_popularity_at_5",
    "popularity_bias_at_5",
    "intra_list_diversity_at_5",
    "duplicate_rate_at_5",
    "candidate_validity_rate",
    "popularity_bias_at_5_status",
    "intra_list_diversity_at_5_status",
    "result_validation_status",
    "notes",
]

BLOCKED_COLUMNS = [
    "result_id",
    "metric_id",
    "baseline_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "contract_status",
    "computation_status",
    "blocking_reason",
    "phase4a_issue_id",
    "missing_or_ambiguous_evidence",
    "adapted_alternative_available",
    "adapted_alternative_not_computed",
    "notes",
]

VALIDATION_COLUMNS = [
    "validation_id",
    "validation_scope",
    "baseline_id",
    "dataset",
    "task_type",
    "cohort_id",
    "metric_id",
    "check_name",
    "expected_value",
    "observed_value",
    "status",
    "severity",
    "evidence",
    "notes",
]

OUTPUT_PATHS = {
    "manifest": "results/tables/canonical_metric_computation_manifest.csv",
    "train_items": "results/tables/canonical_metric_train_item_statistics.csv",
    "ranking_rows": "results/tables/canonical_metric_ranking_rows.csv",
    "top5": "results/tables/canonical_metric_top5_exposures.csv",
    "row_contributions": "results/tables/canonical_metric_row_contributions.csv",
    "coverage": "results/tables/canonical_metric_coverage_components.csv",
    "long": "results/tables/canonical_metric_results_long.csv",
    "wide": "results/tables/canonical_metric_results_wide.csv",
    "blocked": "results/tables/canonical_metric_blocked_results.csv",
    "validation": "results/tables/canonical_metric_results_validation.csv",
    "report": "report/canonical_metric_results_seven_baselines.md",
}

OUTPUT_COLUMNS = {
    "manifest": MANIFEST_COLUMNS,
    "train_items": TRAIN_ITEM_COLUMNS,
    "ranking_rows": RANKING_ROW_COLUMNS,
    "top5": TOP5_COLUMNS,
    "row_contributions": ROW_CONTRIBUTION_COLUMNS,
    "coverage": COVERAGE_COLUMNS,
    "long": LONG_COLUMNS,
    "wide": WIDE_COLUMNS,
    "blocked": BLOCKED_COLUMNS,
    "validation": VALIDATION_COLUMNS,
}

EXPECTED_ROWS = {
    "manifest": 84,
    "train_items": 34_142,
    "ranking_rows": 41_412,
    "top5": 207_060,
    "row_contributions": 41_412,
    "coverage": 84,
    "long": 1_092,
    "wide": 84,
    "blocked": 168,
}

ALLOWED_COMPUTATION_STATUS = {
    "computed_exact",
    "computed_reconstructed",
    "computed_adapted",
    "recomputed_reconciled",
    "diagnostic_computed",
    "structural_invariant_verified",
    "blocked_ambiguous_definition",
    "blocked_missing_input",
    "not_applicable",
    "validation_failed",
}

ALLOWED_VALIDATION_STATUS = {"passed", "failed", "blocked", "not_applicable"}
ALLOWED_VALIDATION_SEVERITY = {"info", "warning", "error"}


def validation_row(
    validation_id: str,
    validation_scope: str,
    check_name: str,
    *,
    expected_value: object,
    observed_value: object,
    passed: bool | None,
    severity: str = "info",
    baseline_id: str = "",
    dataset: str = "",
    task_type: str = "",
    cohort_id: str = "",
    metric_id: str = "",
    evidence: str = "",
    notes: str = "",
) -> dict[str, str]:
    status = common_validation.status_from_outcome(
        passed,
        pass_status="passed",
        fail_status="failed",
        none_status="not_applicable",
    )
    return common_validation.make_validation_row(
        columns=VALIDATION_COLUMNS,
        values={
            "validation_id": validation_id,
            "validation_scope": validation_scope,
            "baseline_id": baseline_id,
            "dataset": dataset,
            "task_type": task_type,
            "cohort_id": cohort_id,
            "metric_id": metric_id,
            "check_name": check_name,
            "expected_value": str(expected_value),
            "observed_value": str(observed_value),
            "status": status,
            "severity": severity,
            "evidence": evidence,
            "notes": notes,
        },
    )


def blocked_validation_row(
    validation_id: str,
    validation_scope: str,
    check_name: str,
    *,
    expected_value: object,
    observed_value: object,
    severity: str = "warning",
    metric_id: str = "",
    evidence: str = "",
    notes: str = "",
) -> dict[str, str]:
    return common_validation.make_validation_row(
        columns=VALIDATION_COLUMNS,
        values={
            "validation_id": validation_id,
            "validation_scope": validation_scope,
            "baseline_id": "",
            "dataset": "",
            "task_type": "",
            "cohort_id": "",
            "metric_id": metric_id,
            "check_name": check_name,
            "expected_value": str(expected_value),
            "observed_value": str(observed_value),
            "status": "blocked",
            "severity": severity,
            "evidence": evidence,
            "notes": notes,
        },
    )


def validate_output_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    for key, columns in OUTPUT_COLUMNS.items():
        if key not in frames:
            raise ValueError(f"missing output frame: {key}")
        if list(frames[key].columns) != columns:
            raise ValueError(f"{key} schema mismatch")
    for key, expected in EXPECTED_ROWS.items():
        observed = len(frames[key])
        if observed != expected:
            raise ValueError(f"{key} row count mismatch: expected {expected}, observed {observed}")

    long = frames["long"]
    if not set(long["computation_status"].astype(str)).issubset(ALLOWED_COMPUTATION_STATUS):
        raise ValueError("long results contain invalid computation_status")
    if long.loc[long["computation_status"].astype(str).str.startswith("blocked"), "metric_value"].astype(str).ne("").any():
        raise ValueError("blocked long-result rows must have empty metric_value")
    blocked_long = long.loc[long["computation_status"].astype(str).str.startswith("blocked")]
    if len(blocked_long) != 168:
        raise ValueError("expected 168 blocked long rows")
    computed_long = long.loc[~long["computation_status"].astype(str).str.startswith("blocked")]
    if len(computed_long) != 924:
        raise ValueError("expected 924 computed/verified long rows")

    wide = frames["wide"]
    if wide["popularity_bias_at_5"].astype(str).ne("").any():
        raise ValueError("wide popularity_bias_at_5 must be blank")
    if wide["intra_list_diversity_at_5"].astype(str).ne("").any():
        raise ValueError("wide intra_list_diversity_at_5 must be blank")

    validation = frames["validation"]
    if not set(validation["status"].astype(str)).issubset(ALLOWED_VALIDATION_STATUS):
        raise ValueError("validation contains invalid status")
    if not set(validation["severity"].astype(str)).issubset(ALLOWED_VALIDATION_SEVERITY):
        raise ValueError("validation contains invalid severity")
    failed_errors = validation.loc[validation["status"].eq("failed") & validation["severity"].eq("error")]
    if not failed_errors.empty:
        raise ValueError("validation contains failed error-level checks")

    _validate_long_wide_equivalence(long, wide)
    _validate_structural_invariants(frames["row_contributions"], long, wide)
    _validate_top5_counts(frames["ranking_rows"], frames["top5"])
    _validate_train_probabilities(frames["train_items"])


def _validate_long_wide_equivalence(long: pd.DataFrame, wide: pd.DataFrame) -> None:
    key_cols = ["baseline_id", "dataset", "task_type", "candidate_set_id", "cohort_id"]
    metric_cols = [
        "hr_at_5",
        "ndcg_at_5",
        "mean_positive_rank_zero_based",
        "median_positive_rank_zero_based",
        "novelty_at_5",
        "catalog_coverage_at_5",
        "item_coverage_at_5",
        "long_tail_rate_at_5",
        "average_popularity_at_5",
        "duplicate_rate_at_5",
        "candidate_validity_rate",
    ]
    wide_lookup = {
        tuple(str(row[col]) for col in key_cols): row
        for row in wide.to_dict("records")
    }
    for row in long.loc[long["metric_id"].isin(metric_cols)].to_dict("records"):
        key = tuple(str(row[col]) for col in key_cols)
        if key not in wide_lookup:
            raise ValueError(f"missing wide row for key {key}")
        left = str(row["metric_value"])
        right = str(wide_lookup[key][row["metric_id"]])
        if left != right:
            raise ValueError(f"long/wide mismatch for {key} {row['metric_id']}: {left} != {right}")


def _validate_structural_invariants(row_contrib: pd.DataFrame, long: pd.DataFrame, wide: pd.DataFrame) -> None:
    if row_contrib["duplicate_rate_at_5"].astype(float).ne(0.0).any():
        raise ValueError("row duplicate structural invariant failed")
    if row_contrib["candidate_validity_rate"].astype(float).ne(1.0).any():
        raise ValueError("row candidate-validity structural invariant failed")
    for metric_id, expected in {"duplicate_rate_at_5": "0", "candidate_validity_rate": "1"}.items():
        values = set(long.loc[long["metric_id"].eq(metric_id), "metric_value"].astype(str))
        if values != {expected}:
            raise ValueError(f"long structural invariant failed for {metric_id}: {values}")
        wide_values = set(wide[metric_id].astype(str))
        if wide_values != {expected}:
            raise ValueError(f"wide structural invariant failed for {metric_id}: {wide_values}")


def _validate_top5_counts(ranking_rows: pd.DataFrame, top5: pd.DataFrame) -> None:
    key_cols = ["baseline_id", "dataset", "task_type", "candidate_set_id", "user_id"]
    expected = ranking_rows.groupby(key_cols, sort=False).size()
    if not expected.eq(1).all():
        raise ValueError("ranking row keys are not unique")
    observed = top5.groupby(key_cols, sort=False).size()
    if not observed.eq(5).all() or len(observed) != len(ranking_rows):
        raise ValueError("top5 exposure counts are not exactly five per ranking row")
    if top5["valid_item"].astype(str).ne("true").any():
        raise ValueError("top5 contains invalid item rows")


def _validate_train_probabilities(train_items: pd.DataFrame) -> None:
    if train_items["train_interaction_count"].astype(int).le(0).any():
        raise ValueError("train statistics contain zero-count item")
    grouped = train_items.groupby("dataset", sort=False)["popularity_probability"].apply(lambda s: s.astype(float).sum())
    for dataset, total in grouped.items():
        if abs(float(total) - 1.0) > 1e-12:
            raise ValueError(f"probabilities do not sum to one for {dataset}: {total}")


def validate_report(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("# Canonical Utility and Beyond-Utility Results for Seven Baselines"):
        raise ValueError("report title mismatch")
    sections = [line for line in text.splitlines() if line.startswith("## ")]
    if len(sections) != 61:
        raise ValueError(f"expected 61 report sections, found {len(sections)}")


def json_array_cell(value: str) -> list[object]:
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        raise ValueError("expected JSON array")
    return decoded

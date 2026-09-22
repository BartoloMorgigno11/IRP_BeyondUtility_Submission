"""Same-cohort consolidation for Plan E BPRMF and non-neural baselines."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.audit.common_io import (
    read_csv_dataframe as _read_csv_dataframe,
    write_dataframe_csv as _write_dataframe_csv,
)


ROW_KEY_COLUMNS = ["dataset", "task_type", "candidate_set_id", "user_id"]
FULL_COHORT_ID = "plan_e2_bprmf_strict_train_known_full"
MATCHED_COHORT_ID = "plan_e2_bprmf_strict_train_known_matched"
BPRMF_BASELINE_ID = "bprmf_reconstructed"
NON_NEURAL_BASELINES = ("random_seeded", "mostpop_train_only", "bm25_like_e3b")

SAME_COHORT_METRIC_COLUMNS = [
    "baseline_id",
    "baseline_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "evaluated_rows",
    "candidate_count",
    "hr_at_5",
    "ndcg_at_5",
    "mean_positive_rank_zero_based",
    "median_positive_rank_zero_based",
    "valid_ranking_rate",
    "duplicate_ranked_item_rate",
    "missing_score_rate",
    "ranking_source_path",
    "cohort_membership_source_path",
    "paper_alignment",
    "notes",
]

COMPARISON_COLUMNS = [
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "evaluated_rows",
    "comparator_baseline_id",
    "bprmf_hr_at_5",
    "comparator_hr_at_5",
    "hr_at_5_difference_bprmf_minus_comparator",
    "bprmf_ndcg_at_5",
    "comparator_ndcg_at_5",
    "ndcg_at_5_difference_bprmf_minus_comparator",
    "bprmf_mean_positive_rank",
    "comparator_mean_positive_rank",
    "mean_rank_difference_bprmf_minus_comparator",
    "bprmf_better_positive_rank_rows",
    "equal_positive_rank_rows",
    "comparator_better_positive_rank_rows",
    "bprmf_positive_rank_win_rate",
    "comparator_positive_rank_win_rate",
    "exact_row_key_match",
    "exact_candidate_list_match",
    "notes",
]


def parse_json_array(value: Any) -> list[str]:
    """Decode a JSON array cell and preserve values as strings."""

    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError("cell is not valid JSON") from exc
    if not isinstance(decoded, list):
        raise ValueError("cell does not decode to a JSON array")
    return [str(item) for item in decoded]


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row[column]) for column in ROW_KEY_COLUMNS)  # type: ignore[return-value]


def filter_rankings_to_cohort(
    rankings: pd.DataFrame,
    cohort_keys: Iterable[tuple[str, str, str, str]],
    *,
    require_no_extras: bool = False,
) -> pd.DataFrame:
    """Return ranking rows whose keys exactly cover the requested cohort."""

    expected = set(cohort_keys)
    if not expected:
        raise ValueError("empty cohort cannot be evaluated")
    frame = rankings.copy(deep=True)
    present = {_frame_key(row) for row in frame.to_dict("records")}
    missing = expected.difference(present)
    if missing:
        raise ValueError(f"missing ranking rows for cohort keys: {sorted(missing)[:3]}")
    extras = present.difference(expected)
    if require_no_extras and extras:
        raise ValueError(f"extra ranking rows outside cohort keys: {sorted(extras)[:3]}")
    mask = frame.apply(lambda row: _frame_key(row) in expected, axis=1)
    filtered = frame.loc[mask].copy()
    if len(filtered) != len(expected):
        duplicate_count = len(filtered) - len({_frame_key(row) for row in filtered.to_dict("records")})
        raise ValueError(f"filtered cohort has duplicate row keys: {duplicate_count}")
    return filtered.sort_values(ROW_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def verify_candidate_list_equality(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Require exact candidate-list equality for matching row keys."""

    left_map = {_frame_key(row): parse_json_array(row["candidate_items"]) for row in left.to_dict("records")}
    right_map = {_frame_key(row): parse_json_array(row["candidate_items"]) for row in right.to_dict("records")}
    if set(left_map) != set(right_map):
        raise ValueError("row-key sets differ before candidate-list comparison")
    mismatches = [key for key in left_map if left_map[key] != right_map[key]]
    if mismatches:
        raise ValueError(f"candidate-list mismatch for row key: {mismatches[0]}")


def metrics_from_rankings(
    rows: pd.DataFrame,
    *,
    baseline_id: str,
    baseline_family: str,
    cohort_id: str,
    ranking_source_path: str,
    cohort_membership_source_path: str,
    paper_alignment: str,
    notes: str,
) -> dict[str, Any]:
    """Recompute utility and structural metrics from filtered ranking rows."""

    if rows.empty:
        raise ValueError("empty cohort cannot be evaluated")
    task_values = sorted(rows["task_type"].astype(str).unique())
    candidate_set_values = sorted(rows["candidate_set_id"].astype(str).unique())
    dataset_values = sorted(rows["dataset"].astype(str).unique())
    if len(task_values) != 1 or len(candidate_set_values) != 1 or len(dataset_values) != 1:
        raise ValueError("metric input must contain one dataset/task/candidate-set unit")
    ranks = rows["positive_rank_zero_based"].astype(int).to_numpy()
    candidate_counts = {len(parse_json_array(value)) for value in rows["candidate_items"].tolist()}
    if len(candidate_counts) != 1:
        raise ValueError("mixed candidate counts in one metric unit")
    duplicate_rate = np.mean(rows["duplicate_ranked_item_count"].astype(int).to_numpy() > 0)
    missing_rate = np.mean(rows["missing_score_count"].astype(int).to_numpy() > 0)
    valid_rate = np.mean([bool_value(value) for value in rows["ranking_valid"].tolist()])
    return {
        "baseline_id": baseline_id,
        "baseline_family": baseline_family,
        "dataset": dataset_values[0],
        "task_type": task_values[0],
        "candidate_set_id": candidate_set_values[0],
        "cohort_id": cohort_id,
        "evaluated_rows": int(len(rows)),
        "candidate_count": int(next(iter(candidate_counts))),
        "hr_at_5": float(np.mean([bool_value(value) for value in rows["positive_in_top_5"].tolist()])),
        "ndcg_at_5": float(rows["ndcg_at_5"].astype(float).mean()),
        "mean_positive_rank_zero_based": float(np.mean(ranks)),
        "median_positive_rank_zero_based": float(np.median(ranks)),
        "valid_ranking_rate": float(valid_rate),
        "duplicate_ranked_item_rate": float(duplicate_rate),
        "missing_score_rate": float(missing_rate),
        "ranking_source_path": ranking_source_path,
        "cohort_membership_source_path": cohort_membership_source_path,
        "paper_alignment": paper_alignment,
        "notes": notes,
    }


def paired_positive_rank_summary(bprmf_rows: pd.DataFrame, comparator_rows: pd.DataFrame) -> dict[str, Any]:
    """Count paired positive-rank wins/losses after exact row-key alignment."""

    verify_candidate_list_equality(bprmf_rows, comparator_rows)
    bprmf = bprmf_rows.sort_values(ROW_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    comp = comparator_rows.sort_values(ROW_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    bprmf_ranks = bprmf["positive_rank_zero_based"].astype(int).to_numpy()
    comp_ranks = comp["positive_rank_zero_based"].astype(int).to_numpy()
    bprmf_better = int(np.sum(bprmf_ranks < comp_ranks))
    equal = int(np.sum(bprmf_ranks == comp_ranks))
    comparator_better = int(np.sum(comp_ranks < bprmf_ranks))
    total = int(len(bprmf))
    return {
        "bprmf_better_positive_rank_rows": bprmf_better,
        "equal_positive_rank_rows": equal,
        "comparator_better_positive_rank_rows": comparator_better,
        "bprmf_positive_rank_win_rate": safe_divide(bprmf_better, total),
        "comparator_positive_rank_win_rate": safe_divide(comparator_better, total),
    }


def normalize_bprmf_rankings(frame: pd.DataFrame, *, source_path: str) -> pd.DataFrame:
    """Normalize Phase 3D and Phase 3E BPRMF ranking schemas."""

    output = frame.copy()
    if "full_cohort_id" not in output.columns and "pilot_full_cohort_id" in output.columns:
        output["full_cohort_id"] = output["pilot_full_cohort_id"]
    if "matched_cohort_id" not in output.columns and "pilot_matched_cohort_id" in output.columns:
        output["matched_cohort_id"] = output["pilot_matched_cohort_id"]
    if "eligible_bprmf_strict_full" not in output.columns and "eligible_pilot_full" in output.columns:
        output["eligible_bprmf_strict_full"] = output["eligible_pilot_full"]
    if "eligible_bprmf_strict_matched" not in output.columns and "eligible_pilot_matched" in output.columns:
        output["eligible_bprmf_strict_matched"] = output["eligible_pilot_matched"]
    output["baseline_id"] = BPRMF_BASELINE_ID
    output["ranking_source_path"] = source_path
    return output


def cohort_keys_from_bprmf_rankings(frame: pd.DataFrame, cohort_id: str) -> list[tuple[str, str, str, str]]:
    flag = "eligible_bprmf_strict_full" if cohort_id == FULL_COHORT_ID else "eligible_bprmf_strict_matched"
    selected = frame.loc[frame[flag].map(bool_value)].copy()
    return sorted({_frame_key(row) for row in selected.to_dict("records")})


def build_same_cohort_metrics(
    *,
    traditional_rankings: pd.DataFrame,
    bprmf_rankings: pd.DataFrame,
    cohort_source_path_by_dataset: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str, str], pd.DataFrame]]:
    """Create the 48-row same-cohort metric table and row frames for comparisons."""

    metric_rows: list[dict[str, Any]] = []
    filtered_frames: dict[tuple[str, str, str, str, str], pd.DataFrame] = {}
    units = sorted(
        {
            (str(row["dataset"]), str(row["task_type"]), str(row["candidate_set_id"]))
            for row in bprmf_rankings.to_dict("records")
        }
    )
    for dataset, task_type, candidate_set_id in units:
        unit_bprmf = _select_unit(bprmf_rankings, dataset, task_type, candidate_set_id)
        for cohort_id in (FULL_COHORT_ID, MATCHED_COHORT_ID):
            keys = cohort_keys_from_bprmf_rankings(unit_bprmf, cohort_id)
            bprmf_filtered = filter_rankings_to_cohort(unit_bprmf, keys)
            bprmf_key = (BPRMF_BASELINE_ID, dataset, task_type, candidate_set_id, cohort_id)
            filtered_frames[bprmf_key] = bprmf_filtered
            metric_rows.append(
                metrics_from_rankings(
                    bprmf_filtered,
                    baseline_id=BPRMF_BASELINE_ID,
                    baseline_family="neural_matrix_factorization",
                    cohort_id=cohort_id,
                    ranking_source_path=_single_value(bprmf_filtered, "ranking_source_path"),
                    cohort_membership_source_path=cohort_source_path_by_dataset[dataset],
                    paper_alignment="adapted_paper_like_reconstructed",
                    notes="BPRMF primary-run rankings filtered to strict train-known cohort",
                )
            )
            for baseline_id in NON_NEURAL_BASELINES:
                source_unit = _select_unit(
                    traditional_rankings.loc[traditional_rankings["baseline_id"].astype(str) == baseline_id],
                    dataset,
                    task_type,
                    candidate_set_id,
                )
                filtered = filter_rankings_to_cohort(source_unit, keys)
                verify_candidate_list_equality(bprmf_filtered, filtered)
                key = (baseline_id, dataset, task_type, candidate_set_id, cohort_id)
                filtered_frames[key] = filtered
                metric_rows.append(
                    metrics_from_rankings(
                        filtered,
                        baseline_id=baseline_id,
                        baseline_family=baseline_family(baseline_id),
                        cohort_id=cohort_id,
                        ranking_source_path="results/tables/traditional_baseline_rankings.csv",
                        cohort_membership_source_path=cohort_source_path_by_dataset[dataset],
                        paper_alignment=paper_alignment(baseline_id),
                        notes="Existing Phase 3A rankings read once and filtered to exact BPRMF cohort",
                    )
                )
    frame = pd.DataFrame(metric_rows, columns=SAME_COHORT_METRIC_COLUMNS)
    frame = frame.sort_values(
        ["baseline_id", "dataset", "task_type", "cohort_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return frame, filtered_frames


def build_paired_comparisons(
    metrics: pd.DataFrame,
    filtered_frames: Mapping[tuple[str, str, str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    units = sorted(
        {
            (dataset, task_type, candidate_set_id, cohort_id)
            for baseline_id, dataset, task_type, candidate_set_id, cohort_id in filtered_frames
            if baseline_id == BPRMF_BASELINE_ID
        }
    )
    metric_map = {
        (
            str(row["baseline_id"]),
            str(row["dataset"]),
            str(row["task_type"]),
            str(row["candidate_set_id"]),
            str(row["cohort_id"]),
        ): row
        for row in metrics.to_dict("records")
    }
    for dataset, task_type, candidate_set_id, cohort_id in units:
        bprmf_key = (BPRMF_BASELINE_ID, dataset, task_type, candidate_set_id, cohort_id)
        bprmf_rows = filtered_frames[bprmf_key]
        bprmf_metric = metric_map[bprmf_key]
        for comparator in NON_NEURAL_BASELINES:
            comp_key = (comparator, dataset, task_type, candidate_set_id, cohort_id)
            comp_rows = filtered_frames[comp_key]
            if {_frame_key(row) for row in bprmf_rows.to_dict("records")} != {
                _frame_key(row) for row in comp_rows.to_dict("records")
            }:
                raise ValueError("paired comparison row-key mismatch")
            verify_candidate_list_equality(bprmf_rows, comp_rows)
            paired = paired_positive_rank_summary(bprmf_rows, comp_rows)
            comp_metric = metric_map[comp_key]
            rows.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": candidate_set_id,
                    "cohort_id": cohort_id,
                    "evaluated_rows": int(bprmf_metric["evaluated_rows"]),
                    "comparator_baseline_id": comparator,
                    "bprmf_hr_at_5": float(bprmf_metric["hr_at_5"]),
                    "comparator_hr_at_5": float(comp_metric["hr_at_5"]),
                    "hr_at_5_difference_bprmf_minus_comparator": float(bprmf_metric["hr_at_5"])
                    - float(comp_metric["hr_at_5"]),
                    "bprmf_ndcg_at_5": float(bprmf_metric["ndcg_at_5"]),
                    "comparator_ndcg_at_5": float(comp_metric["ndcg_at_5"]),
                    "ndcg_at_5_difference_bprmf_minus_comparator": float(bprmf_metric["ndcg_at_5"])
                    - float(comp_metric["ndcg_at_5"]),
                    "bprmf_mean_positive_rank": float(bprmf_metric["mean_positive_rank_zero_based"]),
                    "comparator_mean_positive_rank": float(comp_metric["mean_positive_rank_zero_based"]),
                    "mean_rank_difference_bprmf_minus_comparator": float(
                        bprmf_metric["mean_positive_rank_zero_based"]
                    )
                    - float(comp_metric["mean_positive_rank_zero_based"]),
                    **paired,
                    "exact_row_key_match": "true",
                    "exact_candidate_list_match": "true",
                    "notes": "paired on dataset, task, candidate_set_id and user_id",
                }
            )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS).sort_values(
        ["dataset", "task_type", "cohort_id", "comparator_baseline_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def baseline_family(baseline_id: str) -> str:
    return {
        "random_seeded": "non_neural_random",
        "mostpop_train_only": "non_neural_popularity",
        "bm25_like_e3b": "non_neural_bm25_like",
        BPRMF_BASELINE_ID: "neural_matrix_factorization",
    }[baseline_id]


def paper_alignment(baseline_id: str) -> str:
    if baseline_id == "bm25_like_e3b":
        return "adapted_bm25_like_reconstructed"
    if baseline_id == BPRMF_BASELINE_ID:
        return "adapted_paper_like_reconstructed"
    return "paper_like_reconstructed"


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def validate_metric_ranges(frame: pd.DataFrame) -> None:
    for column in ("hr_at_5", "ndcg_at_5", "valid_ranking_rate", "duplicate_ranked_item_rate", "missing_score_rate"):
        bad = frame.loc[(frame[column].astype(float) < 0) | (frame[column].astype(float) > 1)]
        if not bad.empty:
            raise ValueError(f"metric out of range: {column}")
    if not np.isfinite(frame["mean_positive_rank_zero_based"].astype(float).to_numpy()).all():
        raise ValueError("non-finite mean rank")


def validate_comparison_frame(frame: pd.DataFrame) -> None:
    for column in ("bprmf_positive_rank_win_rate", "comparator_positive_rank_win_rate"):
        bad = frame.loc[(frame[column].astype(float) < 0) | (frame[column].astype(float) > 1)]
        if not bad.empty:
            raise ValueError(f"comparison rate out of range: {column}")
    for row in frame.to_dict("records"):
        total = (
            int(row["bprmf_better_positive_rank_rows"])
            + int(row["equal_positive_rank_rows"])
            + int(row["comparator_better_positive_rank_rows"])
        )
        if total != int(row["evaluated_rows"]):
            raise ValueError("paired counts do not sum to evaluated rows")


def read_csv(path: str | Path) -> pd.DataFrame:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_dataframe(path, dtype="string", keep_default_na=False)


def write_csv(path: str | Path, frame: pd.DataFrame, columns: Sequence[str]) -> None:
    target = Path(path)
    _write_dataframe_csv(
        target,
        frame,
        columns=columns,
        overwrite=False,
        existing_error_message=f"refusing to overwrite existing output: {target}",
    )


def _frame_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row[column]) for column in ROW_KEY_COLUMNS)  # type: ignore[return-value]


def _select_unit(frame: pd.DataFrame, dataset: str, task_type: str, candidate_set_id: str) -> pd.DataFrame:
    selected = frame.loc[
        (frame["dataset"].astype(str) == dataset)
        & (frame["task_type"].astype(str) == task_type)
        & (frame["candidate_set_id"].astype(str) == candidate_set_id)
    ].copy()
    if selected.empty:
        raise ValueError(f"missing unit rows: {dataset} {task_type} {candidate_set_id}")
    return selected.reset_index(drop=True)


def _single_value(frame: pd.DataFrame, column: str) -> str:
    values = sorted(set(frame[column].astype(str).tolist()))
    if len(values) != 1:
        raise ValueError(f"expected one value in {column}, found {values}")
    return values[0]

"""Same-cohort consolidation for Plan E SASRec and prior baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.audit.common_io import (
    read_csv_dataframe as _read_csv_dataframe,
    write_dataframe_csv as _write_dataframe_csv,
)


ROW_KEY_COLUMNS = ["dataset", "task_type", "candidate_set_id", "user_id"]
FULL_COHORT_ID = "plan_e2_sasrec_strict_causal_full"
MATCHED_COHORT_ID = "plan_e2_sasrec_strict_causal_matched"
SASREC_BASELINE_ID = "sasrec_reconstructed"
BPRMF_BASELINE_ID = "bprmf_reconstructed"
GRU4REC_BASELINE_ID = "gru4rec_reconstructed"
NON_NEURAL_BASELINES = ("random_seeded", "mostpop_train_only", "bm25_like_e3b")
ALL_BASELINES = (*NON_NEURAL_BASELINES, BPRMF_BASELINE_ID, GRU4REC_BASELINE_ID, SASREC_BASELINE_ID)
COMPARATORS = (*NON_NEURAL_BASELINES, BPRMF_BASELINE_ID, GRU4REC_BASELINE_ID)

BASELINE_FAMILIES = {
    "random_seeded": "non_neural_random",
    "mostpop_train_only": "non_neural_popularity",
    "bm25_like_e3b": "non_neural_text_similarity",
    "bprmf_reconstructed": "neural_matrix_factorization",
    "gru4rec_reconstructed": "sequential_neural",
    "sasrec_reconstructed": "sequential_transformer",
}

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
    "sasrec_hr_at_5",
    "comparator_hr_at_5",
    "hr_at_5_difference_sasrec_minus_comparator",
    "sasrec_ndcg_at_5",
    "comparator_ndcg_at_5",
    "ndcg_at_5_difference_sasrec_minus_comparator",
    "sasrec_mean_positive_rank",
    "comparator_mean_positive_rank",
    "mean_rank_difference_sasrec_minus_comparator",
    "sasrec_better_positive_rank_rows",
    "equal_positive_rank_rows",
    "comparator_better_positive_rank_rows",
    "sasrec_positive_rank_win_rate",
    "comparator_positive_rank_win_rate",
    "exact_row_key_match",
    "exact_candidate_list_match",
    "notes",
]

WARNING_COLUMNS = [
    "warning_id",
    "dataset",
    "execution_role",
    "stage",
    "warning_category",
    "warning_class",
    "warning_message",
    "warning_source",
    "occurrence_count",
    "scientific_correctness_risk",
    "requires_stop",
    "resolution_status",
    "evidence",
    "notes",
]
ALLOWED_WARNING_CATEGORIES = {"deprecation", "performance", "tensor_layout", "mask_semantics", "numerical", "environment", "other"}
ALLOWED_WARNING_RISKS = {"none", "low", "medium", "high", "unknown"}
ALLOWED_WARNING_RESOLUTIONS = {"documented_no_action", "reviewed_safe", "blocking", "not_reproduced", "unknown"}


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


def normalize_sasrec_rankings(frame: pd.DataFrame, *, source_path: str) -> pd.DataFrame:
    """Normalize Phase 3J/3K SASRec ranking schemas to SASRec cohort IDs."""

    output = frame.copy(deep=True)
    if "full_cohort_id" not in output.columns and "pilot_full_cohort_id" in output.columns:
        output["full_cohort_id"] = output["pilot_full_cohort_id"]
    if "matched_cohort_id" not in output.columns and "pilot_matched_cohort_id" in output.columns:
        output["matched_cohort_id"] = output["pilot_matched_cohort_id"]
    if "eligible_sasrec_strict_full" not in output.columns and "eligible_pilot_full" in output.columns:
        output["eligible_sasrec_strict_full"] = output["eligible_pilot_full"]
    if "eligible_sasrec_strict_matched" not in output.columns and "eligible_pilot_matched" in output.columns:
        output["eligible_sasrec_strict_matched"] = output["eligible_pilot_matched"]
    output["baseline_id"] = SASREC_BASELINE_ID
    output["ranking_source_path"] = source_path
    return output


def normalize_bprmf_rankings(frame: pd.DataFrame, *, source_path: str) -> pd.DataFrame:
    output = frame.copy(deep=True)
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


def normalize_gru4rec_rankings(frame: pd.DataFrame, *, source_path: str) -> pd.DataFrame:
    output = frame.copy(deep=True)
    if "full_cohort_id" not in output.columns and "pilot_full_cohort_id" in output.columns:
        output["full_cohort_id"] = output["pilot_full_cohort_id"]
    if "matched_cohort_id" not in output.columns and "pilot_matched_cohort_id" in output.columns:
        output["matched_cohort_id"] = output["pilot_matched_cohort_id"]
    if "eligible_gru4rec_strict_full" not in output.columns and "eligible_pilot_full" in output.columns:
        output["eligible_gru4rec_strict_full"] = output["eligible_pilot_full"]
    if "eligible_gru4rec_strict_matched" not in output.columns and "eligible_pilot_matched" in output.columns:
        output["eligible_gru4rec_strict_matched"] = output["eligible_pilot_matched"]
    output["baseline_id"] = GRU4REC_BASELINE_ID
    output["ranking_source_path"] = source_path
    return output


def cohort_keys_from_sasrec_rankings(frame: pd.DataFrame, cohort_id: str) -> list[tuple[str, str, str, str]]:
    flag = "eligible_sasrec_strict_full" if cohort_id == FULL_COHORT_ID else "eligible_sasrec_strict_matched"
    selected = frame.loc[frame[flag].map(bool_value)].copy()
    return sorted({row_key(row) for row in selected.to_dict("records")})


def filter_rankings_to_cohort(
    rankings: pd.DataFrame,
    cohort_keys: Iterable[tuple[str, str, str, str]],
    *,
    require_no_extras: bool = False,
) -> pd.DataFrame:
    """Return ranking rows whose row keys exactly cover the requested cohort."""

    expected = set(cohort_keys)
    if not expected:
        raise ValueError("empty cohort cannot be evaluated")
    frame = rankings.copy(deep=True)
    present = {row_key(row) for row in frame.to_dict("records")}
    missing = expected.difference(present)
    if missing:
        raise ValueError(f"missing ranking rows for cohort keys: {sorted(missing)[:3]}")
    extras = present.difference(expected)
    if require_no_extras and extras:
        raise ValueError(f"extra ranking rows outside cohort keys: {sorted(extras)[:3]}")
    mask = frame.apply(lambda row: row_key(row) in expected, axis=1)
    filtered = frame.loc[mask].copy()
    if len(filtered) != len(expected):
        duplicate_count = len(filtered) - len({row_key(row) for row in filtered.to_dict("records")})
        raise ValueError(f"filtered cohort has duplicate row keys: {duplicate_count}")
    return filtered.sort_values(ROW_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def require_exact_row_key_equality(left: pd.DataFrame, right: pd.DataFrame) -> None:
    left_keys = {row_key(row) for row in left.to_dict("records")}
    right_keys = {row_key(row) for row in right.to_dict("records")}
    if left_keys != right_keys:
        missing = sorted(left_keys.difference(right_keys))[:3]
        extra = sorted(right_keys.difference(left_keys))[:3]
        raise ValueError(f"row-key sets differ; missing={missing}; extra={extra}")


def verify_candidate_list_equality(left: pd.DataFrame, right: pd.DataFrame) -> None:
    left_map = {row_key(row): parse_json_array(row["candidate_items"]) for row in left.to_dict("records")}
    right_map = {row_key(row): parse_json_array(row["candidate_items"]) for row in right.to_dict("records")}
    if set(left_map) != set(right_map):
        raise ValueError("row-key sets differ before candidate-list comparison")
    mismatches = [key for key in left_map if left_map[key] != right_map[key]]
    if mismatches:
        raise ValueError(f"candidate-list mismatch for row key: {mismatches[0]}")


def metrics_from_rankings(
    rows: pd.DataFrame,
    *,
    baseline_id: str,
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
    return {
        "baseline_id": baseline_id,
        "baseline_family": BASELINE_FAMILIES[baseline_id],
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
        "valid_ranking_rate": float(np.mean([bool_value(value) for value in rows["ranking_valid"].tolist()])),
        "duplicate_ranked_item_rate": float(np.mean(rows["duplicate_ranked_item_count"].astype(int).to_numpy() > 0)),
        "missing_score_rate": float(np.mean(rows["missing_score_count"].astype(int).to_numpy() > 0)),
        "ranking_source_path": ranking_source_path,
        "cohort_membership_source_path": cohort_membership_source_path,
        "paper_alignment": paper_alignment,
        "notes": notes,
    }


def paired_positive_rank_summary(sasrec_rows: pd.DataFrame, comparator_rows: pd.DataFrame) -> dict[str, Any]:
    """Count paired positive-rank wins/losses after exact row-key alignment."""

    require_exact_row_key_equality(sasrec_rows, comparator_rows)
    verify_candidate_list_equality(sasrec_rows, comparator_rows)
    sas = sasrec_rows.sort_values(ROW_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    comp = comparator_rows.sort_values(ROW_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    sas_ranks = sas["positive_rank_zero_based"].astype(int).to_numpy()
    comp_ranks = comp["positive_rank_zero_based"].astype(int).to_numpy()
    sas_better = int(np.sum(sas_ranks < comp_ranks))
    equal = int(np.sum(sas_ranks == comp_ranks))
    comp_better = int(np.sum(comp_ranks < sas_ranks))
    total = int(len(sas))
    return {
        "sasrec_better_positive_rank_rows": sas_better,
        "equal_positive_rank_rows": equal,
        "comparator_better_positive_rank_rows": comp_better,
        "sasrec_positive_rank_win_rate": safe_divide(sas_better, total),
        "comparator_positive_rank_win_rate": safe_divide(comp_better, total),
    }


def build_same_cohort_metrics(
    *,
    traditional_rankings: pd.DataFrame,
    bprmf_rankings: pd.DataFrame,
    gru4rec_rankings: pd.DataFrame,
    sasrec_rankings: pd.DataFrame,
    cohort_source_path_by_dataset: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str, str], pd.DataFrame]]:
    """Create the 72-row same-cohort metric table and aligned row frames."""

    metric_rows: list[dict[str, Any]] = []
    filtered_frames: dict[tuple[str, str, str, str, str], pd.DataFrame] = {}
    units = sorted(
        {
            (str(row["dataset"]), str(row["task_type"]), str(row["candidate_set_id"]))
            for row in sasrec_rankings.to_dict("records")
        }
    )
    for dataset, task_type, candidate_set_id in units:
        unit_sas = select_unit(sasrec_rankings, dataset, task_type, candidate_set_id)
        unit_bprmf = select_unit(bprmf_rankings, dataset, task_type, candidate_set_id)
        unit_gru = select_unit(gru4rec_rankings, dataset, task_type, candidate_set_id)
        require_exact_row_key_equality(unit_sas, unit_bprmf)
        require_exact_row_key_equality(unit_sas, unit_gru)
        verify_candidate_list_equality(unit_sas, unit_bprmf)
        verify_candidate_list_equality(unit_sas, unit_gru)
        for cohort_id in (FULL_COHORT_ID, MATCHED_COHORT_ID):
            keys = cohort_keys_from_sasrec_rankings(unit_sas, cohort_id)
            sas_cohort = filter_rankings_to_cohort(unit_sas, keys)
            for baseline_id in ALL_BASELINES:
                source_frame = source_frame_for_baseline(
                    baseline_id,
                    traditional_rankings=traditional_rankings,
                    bprmf_rankings=bprmf_rankings,
                    gru4rec_rankings=gru4rec_rankings,
                    sasrec_rankings=sasrec_rankings,
                    dataset=dataset,
                    task_type=task_type,
                    candidate_set_id=candidate_set_id,
                )
                filtered = filter_rankings_to_cohort(source_frame, keys)
                verify_candidate_list_equality(sas_cohort, filtered)
                frame_key = (baseline_id, dataset, task_type, candidate_set_id, cohort_id)
                filtered_frames[frame_key] = filtered
                metric_rows.append(
                    metrics_from_rankings(
                        filtered,
                        baseline_id=baseline_id,
                        cohort_id=cohort_id,
                        ranking_source_path=single_value(filtered, "ranking_source_path"),
                        cohort_membership_source_path=cohort_source_path_by_dataset[dataset],
                        paper_alignment="adapted_paper_like_reconstructed",
                        notes=f"{baseline_id} rankings filtered to SASRec strict causal cohort",
                    )
                )
    frame = pd.DataFrame(metric_rows, columns=SAME_COHORT_METRIC_COLUMNS).sort_values(
        ["baseline_id", "dataset", "task_type", "cohort_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return frame, filtered_frames


def build_paired_comparisons(
    same_cohort_metrics: pd.DataFrame,
    filtered_frames: Mapping[tuple[str, str, str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    """Build SASRec-vs-comparator paired comparison rows."""

    rows: list[dict[str, Any]] = []
    metric_lookup = {
        (
            str(row["baseline_id"]),
            str(row["dataset"]),
            str(row["task_type"]),
            str(row["candidate_set_id"]),
            str(row["cohort_id"]),
        ): row
        for row in same_cohort_metrics.to_dict("records")
    }
    units = sorted({key[1:] for key in filtered_frames if key[0] == SASREC_BASELINE_ID})
    for dataset, task_type, candidate_set_id, cohort_id in units:
        sas_key = (SASREC_BASELINE_ID, dataset, task_type, candidate_set_id, cohort_id)
        sas_rows = filtered_frames[sas_key]
        sas_metrics = metric_lookup[sas_key]
        for comparator in COMPARATORS:
            comp_key = (comparator, dataset, task_type, candidate_set_id, cohort_id)
            comp_rows = filtered_frames[comp_key]
            comp_metrics = metric_lookup[comp_key]
            paired = paired_positive_rank_summary(sas_rows, comp_rows)
            rows.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": candidate_set_id,
                    "cohort_id": cohort_id,
                    "evaluated_rows": int(sas_metrics["evaluated_rows"]),
                    "comparator_baseline_id": comparator,
                    "sasrec_hr_at_5": float(sas_metrics["hr_at_5"]),
                    "comparator_hr_at_5": float(comp_metrics["hr_at_5"]),
                    "hr_at_5_difference_sasrec_minus_comparator": float(sas_metrics["hr_at_5"]) - float(comp_metrics["hr_at_5"]),
                    "sasrec_ndcg_at_5": float(sas_metrics["ndcg_at_5"]),
                    "comparator_ndcg_at_5": float(comp_metrics["ndcg_at_5"]),
                    "ndcg_at_5_difference_sasrec_minus_comparator": float(sas_metrics["ndcg_at_5"]) - float(comp_metrics["ndcg_at_5"]),
                    "sasrec_mean_positive_rank": float(sas_metrics["mean_positive_rank_zero_based"]),
                    "comparator_mean_positive_rank": float(comp_metrics["mean_positive_rank_zero_based"]),
                    "mean_rank_difference_sasrec_minus_comparator": float(sas_metrics["mean_positive_rank_zero_based"]) - float(comp_metrics["mean_positive_rank_zero_based"]),
                    **paired,
                    "exact_row_key_match": True,
                    "exact_candidate_list_match": True,
                    "notes": "Lower positive rank is better; no significance test performed.",
                }
            )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS).sort_values(
        ["dataset", "task_type", "cohort_id", "comparator_baseline_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def validate_metric_ranges(frame: pd.DataFrame) -> None:
    for field in ("hr_at_5", "ndcg_at_5", "valid_ranking_rate", "duplicate_ranked_item_rate", "missing_score_rate"):
        values = frame[field].astype(float)
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{field} outside [0,1]")
    if (frame["evaluated_rows"].astype(int) <= 0).any():
        raise ValueError("evaluated_rows must be positive")


def validate_comparison_frame(frame: pd.DataFrame) -> None:
    if set(frame["comparator_baseline_id"]) != set(COMPARATORS):
        raise ValueError("comparison frame has unexpected comparators")
    for field in ("sasrec_positive_rank_win_rate", "comparator_positive_rank_win_rate"):
        values = frame[field].astype(float)
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{field} outside [0,1]")
    for row in frame.to_dict("records"):
        total = int(row["sasrec_better_positive_rank_rows"]) + int(row["equal_positive_rank_rows"]) + int(row["comparator_better_positive_rank_rows"])
        if total != int(row["evaluated_rows"]):
            raise ValueError("paired positive-rank counts do not sum to evaluated_rows")
        if not bool_value(row["exact_row_key_match"]) or not bool_value(row["exact_candidate_list_match"]):
            raise ValueError("comparison frame contains a false identity flag")


def aggregate_warning_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Aggregate warning records without fabricating missing messages."""

    if not records:
        return pd.DataFrame(columns=WARNING_COLUMNS)
    grouping = [
        "dataset",
        "execution_role",
        "stage",
        "warning_category",
        "warning_class",
        "warning_message",
        "warning_source",
        "scientific_correctness_risk",
        "requires_stop",
        "resolution_status",
        "evidence",
        "notes",
    ]
    frame = pd.DataFrame([dict(record) for record in records])
    for column in grouping:
        if column not in frame.columns:
            frame[column] = ""
    if "occurrence_count" not in frame.columns:
        frame["occurrence_count"] = 1
    grouped = (
        frame.groupby(grouping, dropna=False, sort=True)["occurrence_count"]
        .apply(lambda values: int(pd.to_numeric(values, errors="raise").sum()))
        .reset_index()
    )
    grouped.insert(0, "warning_id", [f"warn_{idx:03d}" for idx in range(1, len(grouped) + 1)])
    result = grouped.loc[:, WARNING_COLUMNS].copy()
    validate_warning_inventory(result)
    return result


def validate_warning_inventory(frame: pd.DataFrame) -> None:
    if list(frame.columns) != WARNING_COLUMNS:
        raise ValueError("warning inventory schema mismatch")
    if frame.empty:
        return
    if frame["warning_id"].duplicated().any():
        raise ValueError("warning_id values must be unique")
    if not set(frame["warning_category"].astype(str)).issubset(ALLOWED_WARNING_CATEGORIES):
        raise ValueError("warning inventory has invalid category")
    if not set(frame["scientific_correctness_risk"].astype(str)).issubset(ALLOWED_WARNING_RISKS):
        raise ValueError("warning inventory has invalid risk")
    if not set(frame["resolution_status"].astype(str)).issubset(ALLOWED_WARNING_RESOLUTIONS):
        raise ValueError("warning inventory has invalid resolution status")
    if (pd.to_numeric(frame["occurrence_count"], errors="raise") < 0).any():
        raise ValueError("warning occurrence_count must be non-negative")


def source_frame_for_baseline(
    baseline_id: str,
    *,
    traditional_rankings: pd.DataFrame,
    bprmf_rankings: pd.DataFrame,
    gru4rec_rankings: pd.DataFrame,
    sasrec_rankings: pd.DataFrame,
    dataset: str,
    task_type: str,
    candidate_set_id: str,
) -> pd.DataFrame:
    if baseline_id == SASREC_BASELINE_ID:
        return select_unit(sasrec_rankings, dataset, task_type, candidate_set_id)
    if baseline_id == GRU4REC_BASELINE_ID:
        return select_unit(gru4rec_rankings, dataset, task_type, candidate_set_id)
    if baseline_id == BPRMF_BASELINE_ID:
        return select_unit(bprmf_rankings, dataset, task_type, candidate_set_id)
    frame = traditional_rankings.loc[traditional_rankings["baseline_id"].astype(str).eq(baseline_id)].copy()
    frame["ranking_source_path"] = "results/tables/traditional_baseline_rankings.csv"
    return select_unit(frame, dataset, task_type, candidate_set_id)


def select_unit(frame: pd.DataFrame, dataset: str, task_type: str, candidate_set_id: str) -> pd.DataFrame:
    selected = frame.loc[
        frame["dataset"].astype(str).eq(dataset)
        & frame["task_type"].astype(str).eq(task_type)
        & frame["candidate_set_id"].astype(str).eq(candidate_set_id)
    ].copy()
    if selected.empty:
        raise ValueError(f"missing unit {dataset}/{task_type}/{candidate_set_id}")
    return selected.reset_index(drop=True)


def single_value(frame: pd.DataFrame, column: str) -> str:
    values = sorted(set(frame[column].astype(str).tolist()))
    if len(values) != 1:
        raise ValueError(f"expected one value in {column}, found {values}")
    return values[0]


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0

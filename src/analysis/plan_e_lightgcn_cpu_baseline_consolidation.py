"""Same-cohort consolidation for CPU LightGCN and prior Plan E baselines."""

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
FULL_COHORT_ID = "plan_e2_lightgcn_strict_connected_full"
MATCHED_COHORT_ID = "plan_e2_lightgcn_strict_connected_matched"

LIGHTGCN_BASELINE_ID = "lightgcn_reconstructed"
BPRMF_BASELINE_ID = "bprmf_reconstructed"
GRU4REC_BASELINE_ID = "gru4rec_reconstructed"
SASREC_BASELINE_ID = "sasrec_reconstructed"
NON_NEURAL_BASELINES = ("random_seeded", "mostpop_train_only", "bm25_like_e3b")
NEURAL_BASELINES = (BPRMF_BASELINE_ID, GRU4REC_BASELINE_ID, SASREC_BASELINE_ID, LIGHTGCN_BASELINE_ID)
ALL_BASELINES = (*NON_NEURAL_BASELINES, *NEURAL_BASELINES)
COMPARATORS = (*NON_NEURAL_BASELINES, BPRMF_BASELINE_ID, GRU4REC_BASELINE_ID, SASREC_BASELINE_ID)

BASELINE_FAMILIES = {
    "random_seeded": "non_neural_random",
    "mostpop_train_only": "non_neural_popularity",
    "bm25_like_e3b": "non_neural_bm25_like",
    "bprmf_reconstructed": "neural_matrix_factorization",
    "gru4rec_reconstructed": "sequential_neural",
    "sasrec_reconstructed": "sequential_transformer",
    "lightgcn_reconstructed": "graph_neural_collaborative_filtering",
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
    "source_execution_device",
    "source_determinism_status",
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
    "lightgcn_execution_device",
    "lightgcn_determinism_status",
    "lightgcn_hr_at_5",
    "comparator_hr_at_5",
    "hr_at_5_difference_lightgcn_minus_comparator",
    "lightgcn_ndcg_at_5",
    "comparator_ndcg_at_5",
    "ndcg_at_5_difference_lightgcn_minus_comparator",
    "lightgcn_mean_positive_rank",
    "comparator_mean_positive_rank",
    "mean_rank_difference_lightgcn_minus_comparator",
    "lightgcn_better_positive_rank_rows",
    "equal_positive_rank_rows",
    "comparator_better_positive_rank_rows",
    "lightgcn_positive_rank_win_rate",
    "comparator_positive_rank_win_rate",
    "exact_row_key_match",
    "exact_candidate_list_match",
    "notes",
]

DEVICE_PROVENANCE_COLUMNS = [
    "dataset",
    "execution_device",
    "source_phase",
    "source_config_path",
    "source_rankings_path",
    "source_metrics_path",
    "primary_execution_status",
    "repeat_execution_status",
    "determinism_classification",
    "exact_final_model_state",
    "exact_propagated_embeddings",
    "exact_candidate_scores",
    "exact_rankings",
    "loss_telemetry_exact",
    "cuda_sparse_forward_limitation",
    "cpu_scale_up_decision_source",
    "paper_alignment",
    "notes",
]

ALL_LIGHTGCN_METRIC_COLUMNS = [
    "model_id",
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
    "mean_user_train_degree",
    "median_user_train_degree",
    "mean_positive_item_train_degree",
    "median_positive_item_train_degree",
    "mean_candidate_item_train_degree",
    "median_candidate_item_train_degree",
    "execution_device",
    "execution_status",
    "determinism_classification",
    "source_metrics_path",
    "paper_alignment",
    "notes",
]


def parse_json_array(value: Any) -> list[str]:
    """Decode a JSON array cell while preserving values as strings."""

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


def require_cpu_tensor(value: Any, *, name: str = "tensor") -> None:
    """Reject CUDA or non-CPU tensors in project-data CPU LightGCN paths."""

    device = getattr(value, "device", None)
    if device is None:
        raise ValueError(f"{name} does not expose a torch-like device")
    if getattr(device, "type", str(device)) != "cpu":
        raise ValueError(f"{name} must be on CPU, got {device}")


def require_cpu_module(module: Any) -> None:
    """Reject a model whose parameters or buffers are not all on CPU."""

    for name, parameter in module.named_parameters():
        require_cpu_tensor(parameter, name=f"parameter {name}")
    for name, buffer in module.named_buffers():
        require_cpu_tensor(buffer, name=f"buffer {name}")


def exact_determinism_row(
    *,
    check_id: str,
    experiment_id: str,
    dataset: str,
    primary_run_id: str,
    repeat_run_id: str,
    check_name: str,
    primary_value: Any,
    repeat_value: Any,
    evidence: str,
    notes: str = "",
) -> dict[str, Any]:
    """Create one exact-equality determinism row with zero tolerance."""

    primary_text = str(primary_value)
    repeat_text = str(repeat_value)
    status = "passed" if primary_text == repeat_text else "failed"
    return {
        "determinism_check_id": check_id,
        "experiment_id": experiment_id,
        "dataset": dataset,
        "primary_run_id": primary_run_id,
        "repeat_run_id": repeat_run_id,
        "check_name": check_name,
        "primary_value": primary_text,
        "repeat_value": repeat_text,
        "absolute_difference": "0" if status == "passed" else "",
        "tolerance": "0",
        "bitwise_equality_required": "true",
        "status": status,
        "severity": "info" if status == "passed" else "error",
        "evidence": evidence,
        "notes": notes,
    }


def validate_exact_determinism(frame: pd.DataFrame) -> None:
    if frame["tolerance"].astype(str).ne("0").any():
        raise ValueError("CPU determinism table must use zero tolerance")
    if frame["bitwise_equality_required"].astype(str).str.lower().ne("true").any():
        raise ValueError("CPU determinism rows must require bitwise equality")
    if frame["status"].astype(str).str.lower().ne("passed").any():
        raise ValueError("CPU determinism contains a failed check")


def normalize_lightgcn_rankings(
    frame: pd.DataFrame,
    *,
    source_path: str,
    source_execution_device: str,
    source_determinism_status: str,
) -> pd.DataFrame:
    output = frame.copy(deep=True)
    if "full_cohort_id" not in output.columns and "pilot_full_cohort_id" in output.columns:
        output["full_cohort_id"] = output["pilot_full_cohort_id"]
    if "matched_cohort_id" not in output.columns and "pilot_matched_cohort_id" in output.columns:
        output["matched_cohort_id"] = output["pilot_matched_cohort_id"]
    if "eligible_lightgcn_strict_full" not in output.columns and "eligible_pilot_full" in output.columns:
        output["eligible_lightgcn_strict_full"] = output["eligible_pilot_full"]
    if "eligible_lightgcn_strict_matched" not in output.columns and "eligible_pilot_matched" in output.columns:
        output["eligible_lightgcn_strict_matched"] = output["eligible_pilot_matched"]
    output["baseline_id"] = LIGHTGCN_BASELINE_ID
    output["ranking_source_path"] = source_path
    output["source_execution_device"] = source_execution_device
    output["source_determinism_status"] = source_determinism_status
    return output


def normalize_bprmf_rankings(
    frame: pd.DataFrame,
    *,
    source_path: str,
    source_execution_device: str = "cuda",
    source_determinism_status: str = "completed",
) -> pd.DataFrame:
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
    output["source_execution_device"] = source_execution_device
    output["source_determinism_status"] = source_determinism_status
    return output


def normalize_gru4rec_rankings(
    frame: pd.DataFrame,
    *,
    source_path: str,
    source_execution_device: str = "cuda",
    source_determinism_status: str = "completed",
) -> pd.DataFrame:
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
    output["source_execution_device"] = source_execution_device
    output["source_determinism_status"] = source_determinism_status
    return output


def normalize_sasrec_rankings(
    frame: pd.DataFrame,
    *,
    source_path: str,
    source_execution_device: str = "cuda",
    source_determinism_status: str = "completed",
) -> pd.DataFrame:
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
    output["source_execution_device"] = source_execution_device
    output["source_determinism_status"] = source_determinism_status
    return output


def normalize_traditional_rankings(frame: pd.DataFrame, *, source_path: str) -> pd.DataFrame:
    output = frame.copy(deep=True)
    output["ranking_source_path"] = source_path
    output["source_execution_device"] = "not_applicable"
    output["source_determinism_status"] = "deterministic_saved_rankings"
    return output


def cohort_keys_from_lightgcn_rankings(frame: pd.DataFrame, cohort_id: str) -> list[tuple[str, str, str, str]]:
    flag = "eligible_lightgcn_strict_full" if cohort_id == FULL_COHORT_ID else "eligible_lightgcn_strict_matched"
    if flag not in frame.columns:
        raise ValueError(f"missing LightGCN cohort flag: {flag}")
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
    source_execution_device: str,
    source_determinism_status: str,
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
        "source_execution_device": source_execution_device,
        "source_determinism_status": source_determinism_status,
        "paper_alignment": paper_alignment,
        "notes": notes,
    }


def paired_positive_rank_summary(lightgcn_rows: pd.DataFrame, comparator_rows: pd.DataFrame) -> dict[str, Any]:
    """Count paired positive-rank wins/losses after exact row-key alignment."""

    require_exact_row_key_equality(lightgcn_rows, comparator_rows)
    verify_candidate_list_equality(lightgcn_rows, comparator_rows)
    lg = lightgcn_rows.sort_values(ROW_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    comp = comparator_rows.sort_values(ROW_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    lg_ranks = lg["positive_rank_zero_based"].astype(int).to_numpy()
    comp_ranks = comp["positive_rank_zero_based"].astype(int).to_numpy()
    lightgcn_better = int(np.sum(lg_ranks < comp_ranks))
    equal = int(np.sum(lg_ranks == comp_ranks))
    comparator_better = int(np.sum(comp_ranks < lg_ranks))
    total = int(len(lg))
    return {
        "lightgcn_better_positive_rank_rows": lightgcn_better,
        "equal_positive_rank_rows": equal,
        "comparator_better_positive_rank_rows": comparator_better,
        "lightgcn_positive_rank_win_rate": safe_divide(lightgcn_better, total),
        "comparator_positive_rank_win_rate": safe_divide(comparator_better, total),
    }


def build_same_cohort_metrics(
    *,
    traditional_rankings: pd.DataFrame,
    bprmf_rankings: pd.DataFrame,
    gru4rec_rankings: pd.DataFrame,
    sasrec_rankings: pd.DataFrame,
    lightgcn_rankings: pd.DataFrame,
    cohort_source_path_by_dataset: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str, str], pd.DataFrame]]:
    """Create the 84-row same-cohort metric table and aligned row frames."""

    metric_rows: list[dict[str, Any]] = []
    filtered_frames: dict[tuple[str, str, str, str, str], pd.DataFrame] = {}
    units = sorted(
        {
            (str(row["dataset"]), str(row["task_type"]), str(row["candidate_set_id"]))
            for row in lightgcn_rankings.to_dict("records")
        }
    )
    for dataset, task_type, candidate_set_id in units:
        unit_lightgcn = select_unit(lightgcn_rankings, dataset, task_type, candidate_set_id)
        unit_bprmf = select_unit(bprmf_rankings, dataset, task_type, candidate_set_id)
        unit_gru = select_unit(gru4rec_rankings, dataset, task_type, candidate_set_id)
        unit_sas = select_unit(sasrec_rankings, dataset, task_type, candidate_set_id)
        for neural in (unit_bprmf, unit_gru, unit_sas):
            require_exact_row_key_equality(unit_lightgcn, neural)
            verify_candidate_list_equality(unit_lightgcn, neural)
        for cohort_id in (FULL_COHORT_ID, MATCHED_COHORT_ID):
            keys = cohort_keys_from_lightgcn_rankings(unit_lightgcn, cohort_id)
            lightgcn_filtered = filter_rankings_to_cohort(unit_lightgcn, keys)
            for baseline_id in ALL_BASELINES:
                source_frame = source_frame_for_baseline(
                    baseline_id,
                    traditional_rankings=traditional_rankings,
                    bprmf_rankings=bprmf_rankings,
                    gru4rec_rankings=gru4rec_rankings,
                    sasrec_rankings=sasrec_rankings,
                    lightgcn_rankings=lightgcn_rankings,
                    dataset=dataset,
                    task_type=task_type,
                    candidate_set_id=candidate_set_id,
                )
                filtered = filter_rankings_to_cohort(source_frame, keys)
                verify_candidate_list_equality(lightgcn_filtered, filtered)
                frame_key = (baseline_id, dataset, task_type, candidate_set_id, cohort_id)
                filtered_frames[frame_key] = filtered
                metric_rows.append(
                    metrics_from_rankings(
                        filtered,
                        baseline_id=baseline_id,
                        cohort_id=cohort_id,
                        ranking_source_path=single_value(filtered, "ranking_source_path"),
                        cohort_membership_source_path=cohort_source_path_by_dataset[dataset],
                        source_execution_device=single_value(filtered, "source_execution_device"),
                        source_determinism_status=single_value(filtered, "source_determinism_status"),
                        paper_alignment=paper_alignment(baseline_id),
                        notes=f"{baseline_id} rankings filtered to LightGCN strict connected cohort",
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
    """Build LightGCN-vs-comparator paired comparison rows."""

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
    units = sorted({key[1:] for key in filtered_frames if key[0] == LIGHTGCN_BASELINE_ID})
    for dataset, task_type, candidate_set_id, cohort_id in units:
        lightgcn_key = (LIGHTGCN_BASELINE_ID, dataset, task_type, candidate_set_id, cohort_id)
        lightgcn_rows = filtered_frames[lightgcn_key]
        lightgcn_metrics = metric_lookup[lightgcn_key]
        for comparator in COMPARATORS:
            comp_key = (comparator, dataset, task_type, candidate_set_id, cohort_id)
            comp_rows = filtered_frames[comp_key]
            comp_metrics = metric_lookup[comp_key]
            paired = paired_positive_rank_summary(lightgcn_rows, comp_rows)
            rows.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": candidate_set_id,
                    "cohort_id": cohort_id,
                    "evaluated_rows": int(lightgcn_metrics["evaluated_rows"]),
                    "comparator_baseline_id": comparator,
                    "lightgcn_execution_device": str(lightgcn_metrics["source_execution_device"]),
                    "lightgcn_determinism_status": str(lightgcn_metrics["source_determinism_status"]),
                    "lightgcn_hr_at_5": float(lightgcn_metrics["hr_at_5"]),
                    "comparator_hr_at_5": float(comp_metrics["hr_at_5"]),
                    "hr_at_5_difference_lightgcn_minus_comparator": float(lightgcn_metrics["hr_at_5"])
                    - float(comp_metrics["hr_at_5"]),
                    "lightgcn_ndcg_at_5": float(lightgcn_metrics["ndcg_at_5"]),
                    "comparator_ndcg_at_5": float(comp_metrics["ndcg_at_5"]),
                    "ndcg_at_5_difference_lightgcn_minus_comparator": float(lightgcn_metrics["ndcg_at_5"])
                    - float(comp_metrics["ndcg_at_5"]),
                    "lightgcn_mean_positive_rank": float(lightgcn_metrics["mean_positive_rank_zero_based"]),
                    "comparator_mean_positive_rank": float(comp_metrics["mean_positive_rank_zero_based"]),
                    "mean_rank_difference_lightgcn_minus_comparator": float(
                        lightgcn_metrics["mean_positive_rank_zero_based"]
                    )
                    - float(comp_metrics["mean_positive_rank_zero_based"]),
                    **paired,
                    "exact_row_key_match": "true",
                    "exact_candidate_list_match": "true",
                    "notes": "Lower positive rank is better; no significance test performed.",
                }
            )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS).sort_values(
        ["dataset", "task_type", "cohort_id", "comparator_baseline_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def source_frame_for_baseline(
    baseline_id: str,
    *,
    traditional_rankings: pd.DataFrame,
    bprmf_rankings: pd.DataFrame,
    gru4rec_rankings: pd.DataFrame,
    sasrec_rankings: pd.DataFrame,
    lightgcn_rankings: pd.DataFrame,
    dataset: str,
    task_type: str,
    candidate_set_id: str,
) -> pd.DataFrame:
    if baseline_id == LIGHTGCN_BASELINE_ID:
        return select_unit(lightgcn_rankings, dataset, task_type, candidate_set_id)
    if baseline_id == SASREC_BASELINE_ID:
        return select_unit(sasrec_rankings, dataset, task_type, candidate_set_id)
    if baseline_id == GRU4REC_BASELINE_ID:
        return select_unit(gru4rec_rankings, dataset, task_type, candidate_set_id)
    if baseline_id == BPRMF_BASELINE_ID:
        return select_unit(bprmf_rankings, dataset, task_type, candidate_set_id)
    frame = traditional_rankings.loc[traditional_rankings["baseline_id"].astype(str).eq(baseline_id)].copy()
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
    for field in ("lightgcn_positive_rank_win_rate", "comparator_positive_rank_win_rate"):
        values = frame[field].astype(float)
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{field} outside [0,1]")
    for row in frame.to_dict("records"):
        total = (
            int(row["lightgcn_better_positive_rank_rows"])
            + int(row["equal_positive_rank_rows"])
            + int(row["comparator_better_positive_rank_rows"])
        )
        if total != int(row["evaluated_rows"]):
            raise ValueError("paired positive-rank counts do not sum to evaluated_rows")
        if not bool_value(row["exact_row_key_match"]) or not bool_value(row["exact_candidate_list_match"]):
            raise ValueError("comparison frame contains a false identity flag")


def validate_all_lightgcn_metrics(frame: pd.DataFrame) -> None:
    if list(frame.columns) != ALL_LIGHTGCN_METRIC_COLUMNS:
        raise ValueError("all-dataset LightGCN metrics schema mismatch")
    validate_metric_ranges(frame)
    if len(frame) != 12:
        raise ValueError(f"expected 12 LightGCN metric rows, found {len(frame)}")


def validate_device_provenance(frame: pd.DataFrame) -> None:
    if list(frame.columns) != DEVICE_PROVENANCE_COLUMNS:
        raise ValueError("device provenance schema mismatch")
    if len(frame) != 3:
        raise ValueError(f"expected 3 device provenance rows, found {len(frame)}")
    if set(frame["dataset"].astype(str)) != {"Amazon Beauty", "Amazon Sports", "ML-1M"}:
        raise ValueError("device provenance datasets mismatch")


def single_value(frame: pd.DataFrame, column: str) -> str:
    values = sorted(set(frame[column].astype(str).tolist()))
    if len(values) != 1:
        raise ValueError(f"expected one value in {column}, found {values}")
    return values[0]


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def paper_alignment(baseline_id: str) -> str:
    if baseline_id == "bm25_like_e3b":
        return "adapted_bm25_like_reconstructed"
    if baseline_id in {BPRMF_BASELINE_ID, GRU4REC_BASELINE_ID, SASREC_BASELINE_ID, LIGHTGCN_BASELINE_ID}:
        return "adapted_paper_like_reconstructed"
    return "paper_like_reconstructed"

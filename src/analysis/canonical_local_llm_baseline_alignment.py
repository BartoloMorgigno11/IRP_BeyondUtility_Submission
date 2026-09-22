"""Baseline alignment and paired comparison helpers for Phase 5C."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.canonical_local_llm_full_evaluation import (
    DATASET_ORDER,
    MODEL_ORDER,
    TASK_ORDER,
    bool_text,
    clean_cell,
    format_float,
    frame,
    parse_bool,
    parse_json_array,
    row_key,
)
from src.metrics.canonical_local_llm_beyond_utility import (
    SUPPORTED_METRICS,
    TrainContext,
    aggregate_rows,
    build_train_context,
    prompt_candidate_union,
    row_metric_values,
)


BASELINES = [
    "random_seeded",
    "mostpop_train_only",
    "bm25_like_e3b",
    "bprmf_reconstructed",
    "gru4rec_reconstructed",
    "sasrec_reconstructed",
    "lightgcn_reconstructed",
]

BASELINE_COHORT_ALIGNMENT_COLUMNS = [
    "alignment_id",
    "baseline_id",
    "dataset",
    "task_type",
    "llm_prompt_compatible_rows",
    "baseline_available_rows",
    "exact_overlap_rows",
    "missing_baseline_rows",
    "extra_baseline_rows",
    "candidate_identity_match_rows",
    "candidate_order_match_rows",
    "positive_identity_match_rows",
    "exact_same_cohort",
    "alignment_status",
    "blocking_reason",
    "notes",
]

BASELINE_RECOMPUTED_COLUMNS = [
    "metric_row_id",
    "baseline_id",
    "comparison_model_family",
    "dataset",
    "task_type",
    "comparison_scope",
    "metric_id",
    "cohort_rows",
    "evaluated_rows",
    "metric_value",
    "metric_numerator",
    "metric_denominator",
    "alignment_status",
    "metric_status",
    "notes",
]

BASELINE_COMPARISON_COLUMNS = [
    "comparison_id",
    "model_family",
    "baseline_id",
    "dataset",
    "task_type",
    "comparison_scope",
    "metric_id",
    "cohort_rows",
    "llm_evaluated_rows",
    "baseline_evaluated_rows",
    "llm_value",
    "baseline_value",
    "absolute_delta",
    "relative_delta",
    "bootstrap_seed",
    "bootstrap_resamples",
    "bootstrap_confidence_level",
    "paired_delta_ci_low",
    "paired_delta_ci_high",
    "denominator_policy",
    "alignment_status",
    "interpretation_status",
    "notes",
]

COMPARISON_METRICS = [
    "hr_at_5",
    "ndcg_at_5",
    "novelty_at_5",
    "catalog_coverage_at_5",
    "item_coverage_at_5",
    "long_tail_rate_at_5",
    "average_popularity_at_5",
    "duplicate_rate_at_5",
    "candidate_validity_rate",
]


def _key_set(frame_: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    return {row_key(row) for row in frame_.to_dict("records")}


def _key_to_row(frame_: pd.DataFrame) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    return {row_key(row): row for row in frame_.to_dict("records")}


def _candidate_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return parse_json_array(left["candidate_items"]) == parse_json_array(right["candidate_items"])


def _positive_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return clean_cell(left.get("positive_item_id", "")) == clean_cell(right.get("positive_item_id", ""))


def _status(exact_same: bool, overlap: int, llm_count: int, baseline_count: int, candidate_match: int, positive_match: int) -> tuple[str, str]:
    if exact_same and overlap == llm_count == baseline_count and candidate_match == overlap and positive_match == overlap:
        return "exact_same_prompt_compatible_cohort", ""
    if overlap > 0 and candidate_match == overlap and positive_match == overlap:
        return "partial_baseline_coverage", "baseline cohort does not exactly equal the LLM prompt-compatible cohort"
    if overlap > 0 and candidate_match < overlap:
        return "candidate_mismatch", "candidate list/order differs for at least one overlapping row"
    if overlap > 0:
        return "exact_intersection_only", "only an exact row-key intersection is usable"
    return "comparison_blocked", "no overlapping row identities"


def build_baseline_cohort_alignment(llm_rankings: pd.DataFrame, baseline_rows: pd.DataFrame) -> pd.DataFrame:
    unique_llm = llm_rankings.drop_duplicates(["dataset", "task_type", "candidate_set_id", "user_id"])
    llm_lookup = _key_to_row(unique_llm)
    rows: list[dict[str, Any]] = []
    counter = 1
    for baseline_id in BASELINES:
        baseline = baseline_rows[baseline_rows["baseline_id"].eq(baseline_id)].copy()
        for dataset in DATASET_ORDER:
            for task in TASK_ORDER:
                llm_unit = unique_llm[unique_llm["dataset"].eq(dataset) & unique_llm["task_type"].eq(task)]
                base_unit = baseline[baseline["dataset"].eq(dataset) & baseline["task_type"].eq(task)]
                llm_keys = _key_set(llm_unit)
                base_keys = _key_set(base_unit)
                overlap = sorted(llm_keys.intersection(base_keys))
                base_lookup = _key_to_row(base_unit)
                candidate_match = sum(1 for key in overlap if _candidate_equal(llm_lookup[key], base_lookup[key]))
                positive_match = sum(1 for key in overlap if _positive_equal(llm_lookup[key], base_lookup[key]))
                exact_same = llm_keys == base_keys and candidate_match == len(overlap) and positive_match == len(overlap)
                status, reason = _status(exact_same, len(overlap), len(llm_keys), len(base_keys), candidate_match, positive_match)
                rows.append(
                    {
                        "alignment_id": f"BASEALIGN-{counter:03d}",
                        "baseline_id": baseline_id,
                        "dataset": dataset,
                        "task_type": task,
                        "llm_prompt_compatible_rows": str(len(llm_keys)),
                        "baseline_available_rows": str(len(base_keys)),
                        "exact_overlap_rows": str(len(overlap)),
                        "missing_baseline_rows": str(len(llm_keys.difference(base_keys))),
                        "extra_baseline_rows": str(len(base_keys.difference(llm_keys))),
                        "candidate_identity_match_rows": str(candidate_match),
                        "candidate_order_match_rows": str(candidate_match),
                        "positive_identity_match_rows": str(positive_match),
                        "exact_same_cohort": bool_text(exact_same),
                        "alignment_status": status,
                        "blocking_reason": "" if exact_same else reason,
                        "notes": "row identity is dataset/task/candidate_set_id/user_id; candidate order checked by JSON array equality",
                    }
                )
                counter += 1
    return frame(rows, BASELINE_COHORT_ALIGNMENT_COLUMNS)


def _filter_by_keys(frame_: pd.DataFrame, keys: set[tuple[str, str, str, str]]) -> pd.DataFrame:
    if not keys:
        return frame_.iloc[0:0].copy()
    mask = frame_.apply(lambda row: row_key(row) in keys, axis=1)
    return frame_.loc[mask].sort_values(["dataset", "task_type", "candidate_set_id", "user_id"], kind="mergesort").reset_index(drop=True)


def _utility_values(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if rows.empty:
        return {
            "hr_at_5": {"metric_value": "", "metric_numerator": "", "metric_denominator": "", "evaluated_rows": "0", "metric_status": "blocked_empty_cohort"},
            "ndcg_at_5": {"metric_value": "", "metric_numerator": "", "metric_denominator": "", "evaluated_rows": "0", "metric_status": "blocked_empty_cohort"},
        }
    if "positive_in_top_5" in rows.columns:
        hits = [1.0 if parse_bool(value) else 0.0 for value in rows["positive_in_top_5"].tolist()]
    else:
        ranks = pd.to_numeric(rows["positive_rank_zero_based"], errors="coerce").fillna(999999).astype(int).tolist()
        hits = [1.0 if rank < 5 else 0.0 for rank in ranks]
    if "ndcg_at_5" in rows.columns:
        ndcg = pd.to_numeric(rows["ndcg_at_5"], errors="coerce").fillna(0.0).astype(float).tolist()
    else:
        ranks = pd.to_numeric(rows["positive_rank_zero_based"], errors="coerce").fillna(999999).astype(int).tolist()
        ndcg = [0.0 if rank >= 5 else 1.0 / math.log2(rank + 2.0) for rank in ranks]
    return {
        "hr_at_5": {
            "metric_value": format_float(float(np.mean(np.asarray(hits, dtype=float)))),
            "metric_numerator": format_float(float(np.sum(np.asarray(hits, dtype=float)))),
            "metric_denominator": str(len(rows)),
            "evaluated_rows": str(len(rows)),
            "metric_status": "complete",
        },
        "ndcg_at_5": {
            "metric_value": format_float(float(np.mean(np.asarray(ndcg, dtype=float)))),
            "metric_numerator": format_float(float(np.sum(np.asarray(ndcg, dtype=float)))),
            "metric_denominator": str(len(rows)),
            "evaluated_rows": str(len(rows)),
            "metric_status": "complete",
        },
    }


def aggregate_comparison_metrics(
    rows: pd.DataFrame,
    *,
    dataset: str,
    task_type: str,
    planned_rows: int,
    train: TrainContext,
    candidate_unions: Mapping[tuple[str, str], set[str]],
) -> dict[str, dict[str, Any]]:
    utility = _utility_values(rows)
    beyond = aggregate_rows(
        rows,
        dataset=dataset,
        task_type=task_type,
        planned_rows=planned_rows,
        strict_valid_rows=len(rows),
        train=train,
        candidate_unions=candidate_unions,
    )
    result: dict[str, dict[str, Any]] = {**utility}
    for metric_id in COMPARISON_METRICS:
        if metric_id in utility:
            continue
        metric = beyond[metric_id]
        result[metric_id] = {
            "metric_value": metric["metric_value"],
            "metric_numerator": metric["metric_numerator"],
            "metric_denominator": metric["metric_denominator"],
            "evaluated_rows": metric["evaluated_rows"],
            "metric_status": metric["metric_status"],
        }
    return result


def build_baseline_recomputed_metrics(
    llm_rankings: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    alignment: pd.DataFrame,
    train_items: pd.DataFrame,
) -> pd.DataFrame:
    train = build_train_context(train_items)
    candidate_unions = prompt_candidate_union(llm_rankings)
    rows: list[dict[str, Any]] = []
    counter = 1
    unique_llm = llm_rankings.drop_duplicates(["dataset", "task_type", "candidate_set_id", "user_id"])
    for model in MODEL_ORDER:
        model_rows = llm_rankings[llm_rankings["model_family"].eq(model)]
        for baseline_id in BASELINES:
            baseline = baseline_rows[baseline_rows["baseline_id"].eq(baseline_id)]
            for dataset in DATASET_ORDER:
                for task in TASK_ORDER:
                    llm_unit = unique_llm[unique_llm["dataset"].eq(dataset) & unique_llm["task_type"].eq(task)]
                    model_unit = model_rows[model_rows["dataset"].eq(dataset) & model_rows["task_type"].eq(task)]
                    base_unit = baseline[baseline["dataset"].eq(dataset) & baseline["task_type"].eq(task)]
                    llm_keys = _key_set(llm_unit)
                    base_keys = _key_set(base_unit)
                    align_row = alignment[
                        alignment["baseline_id"].eq(baseline_id)
                        & alignment["dataset"].eq(dataset)
                        & alignment["task_type"].eq(task)
                    ].iloc[0]
                    for metric_id in COMPARISON_METRICS:
                        rows.append(
                            {
                                "metric_row_id": f"BASEMET-{counter:05d}",
                                "baseline_id": baseline_id,
                                "comparison_model_family": model,
                                "dataset": dataset,
                                "task_type": task,
                                "comparison_scope": "full_prompt_compatible_cohort",
                                "metric_id": metric_id,
                                "cohort_rows": str(len(llm_keys)),
                                "evaluated_rows": "0",
                                "metric_value": "",
                                "metric_numerator": "",
                                "metric_denominator": "",
                                "alignment_status": clean_cell(align_row["alignment_status"]),
                                "metric_status": "blocked_cohort_mismatch" if clean_cell(align_row["exact_same_cohort"]) != "true" else "not_computed",
                                "notes": "full prompt-compatible baseline comparison blocked because baseline rows do not exactly match the LLM cohort",
                            }
                        )
                        counter += 1
                    valid_llm_keys = _key_set(model_unit[model_unit["strict_parse_valid"].eq("true")])
                    paired_keys = valid_llm_keys.intersection(base_keys)
                    paired_base = _filter_by_keys(base_unit, paired_keys)
                    metrics = aggregate_comparison_metrics(
                        paired_base,
                        dataset=dataset,
                        task_type=task,
                        planned_rows=len(paired_keys),
                        train=train,
                        candidate_unions=candidate_unions,
                    )
                    for metric_id in COMPARISON_METRICS:
                        metric = metrics[metric_id]
                        rows.append(
                            {
                                "metric_row_id": f"BASEMET-{counter:05d}",
                                "baseline_id": baseline_id,
                                "comparison_model_family": model,
                                "dataset": dataset,
                                "task_type": task,
                                "comparison_scope": "paired_llm_strict_valid_subset",
                                "metric_id": metric_id,
                                "cohort_rows": str(len(paired_keys)),
                                "evaluated_rows": metric["evaluated_rows"],
                                "metric_value": metric["metric_value"],
                                "metric_numerator": metric["metric_numerator"],
                                "metric_denominator": metric["metric_denominator"],
                                "alignment_status": "exact_intersection_only",
                                "metric_status": metric["metric_status"],
                                "notes": "baseline recomputed on exact row intersection with this LLM strict-valid subset",
                            }
                        )
                        counter += 1
    return frame(rows, BASELINE_RECOMPUTED_COLUMNS)


def paired_bootstrap_delta_ci(
    llm_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    seed: int = 42,
    resamples: int = 10000,
    confidence: float = 0.95,
    chunk_size: int = 1000,
) -> tuple[float, float]:
    if len(llm_values) != len(baseline_values):
        raise ValueError("paired bootstrap requires equal-length arrays")
    if not llm_values:
        raise ValueError("paired bootstrap requires at least one row")
    delta = np.asarray(llm_values, dtype=float) - np.asarray(baseline_values, dtype=float)
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    n = len(delta)
    remaining = int(resamples)
    while remaining > 0:
        take = min(chunk_size, remaining)
        indices = rng.integers(0, n, size=(take, n))
        means.append(delta[indices].mean(axis=1))
        remaining -= take
    samples = np.concatenate(means)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def _row_series_for_metric(rows: pd.DataFrame, metric_id: str, train: TrainContext) -> list[float]:
    if metric_id == "hr_at_5":
        if "positive_in_top_5" in rows.columns:
            return [1.0 if parse_bool(value) else 0.0 for value in rows["positive_in_top_5"].tolist()]
        ranks = pd.to_numeric(rows["positive_rank_zero_based"], errors="coerce").fillna(999999).astype(int).tolist()
        return [1.0 if rank < 5 else 0.0 for rank in ranks]
    if metric_id == "ndcg_at_5":
        if "ndcg_at_5" in rows.columns:
            return pd.to_numeric(rows["ndcg_at_5"], errors="coerce").fillna(0.0).astype(float).tolist()
        ranks = pd.to_numeric(rows["positive_rank_zero_based"], errors="coerce").fillna(999999).astype(int).tolist()
        return [0.0 if rank >= 5 else 1.0 / math.log2(rank + 2.0) for rank in ranks]
    values: list[float] = []
    for row in rows.to_dict("records"):
        row_values = row_metric_values(row, train)
        if metric_id in row_values:
            values.append(float(row_values[metric_id]))
    return values


def _metric_lookup(metrics: pd.DataFrame) -> dict[tuple[str, str, str, str, str, str], Mapping[str, Any]]:
    return {
        (
            row["baseline_id"],
            row["comparison_model_family"],
            row["dataset"],
            row["task_type"],
            row["comparison_scope"],
            row["metric_id"],
        ): row
        for row in metrics.to_dict("records")
    }


def build_baseline_comparison(
    llm_rankings: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    train_items: pd.DataFrame,
    *,
    bootstrap_resamples: int = 10000,
) -> pd.DataFrame:
    train = build_train_context(train_items)
    candidate_unions = prompt_candidate_union(llm_rankings)
    metric_lookup = _metric_lookup(baseline_metrics)
    rows: list[dict[str, Any]] = []
    counter = 1
    for model in MODEL_ORDER:
        model_rows = llm_rankings[llm_rankings["model_family"].eq(model)]
        for baseline_id in BASELINES:
            baseline = baseline_rows[baseline_rows["baseline_id"].eq(baseline_id)]
            for dataset in DATASET_ORDER:
                for task in TASK_ORDER:
                    llm_unit_all = model_rows[model_rows["dataset"].eq(dataset) & model_rows["task_type"].eq(task)]
                    base_unit = baseline[baseline["dataset"].eq(dataset) & baseline["task_type"].eq(task)]
                    full_keys = _key_set(llm_unit_all)
                    for scope in ("full_prompt_compatible_cohort", "paired_llm_strict_valid_subset"):
                        if scope == "paired_llm_strict_valid_subset":
                            llm_keys = _key_set(llm_unit_all[llm_unit_all["strict_parse_valid"].eq("true")]).intersection(_key_set(base_unit))
                            llm_subset = _filter_by_keys(llm_unit_all, llm_keys)
                            base_subset = _filter_by_keys(base_unit, llm_keys)
                            llm_metrics = aggregate_comparison_metrics(
                                llm_subset,
                                dataset=dataset,
                                task_type=task,
                                planned_rows=len(llm_keys),
                                train=train,
                                candidate_unions=candidate_unions,
                            )
                            denominator_policy = "exact_row_intersection_of_llm_strict_valid_subset"
                            interpretation = "conditional_diagnostic_exact_intersection"
                        else:
                            llm_keys = full_keys
                            llm_subset = llm_unit_all
                            base_subset = _filter_by_keys(base_unit, set())
                            llm_metrics = aggregate_comparison_metrics(
                                llm_unit_all[llm_unit_all["strict_parse_valid"].eq("true")],
                                dataset=dataset,
                                task_type=task,
                                planned_rows=len(llm_unit_all),
                                train=train,
                                candidate_unions=candidate_unions,
                            )
                            llm_metrics.update(_operational_utility_metrics(llm_unit_all))
                            denominator_policy = "comparison_blocked_baseline_not_exact_prompt_compatible"
                            interpretation = "blocked_cohort_mismatch"
                        for metric_id in COMPARISON_METRICS:
                            base_metric = metric_lookup[(baseline_id, model, dataset, task, scope, metric_id)]
                            llm_metric = llm_metrics.get(metric_id, {})
                            base_value = clean_cell(base_metric.get("metric_value", ""))
                            llm_value = clean_cell(llm_metric.get("metric_value", ""))
                            if scope == "full_prompt_compatible_cohort":
                                base_value = ""
                                delta = ""
                                relative = ""
                                ci_low = ""
                                ci_high = ""
                                seed = ""
                                resamples = ""
                                confidence = ""
                                alignment_status = clean_cell(base_metric.get("alignment_status", "partial_baseline_coverage"))
                            else:
                                delta_value = _safe_delta(llm_value, base_value)
                                delta = format_float(delta_value) if delta_value is not None else ""
                                relative_value = _safe_relative(delta_value, base_value)
                                relative = format_float(relative_value) if relative_value is not None else ""
                                alignment_status = "exact_intersection_only"
                                if metric_id in {"hr_at_5", "ndcg_at_5"} and len(llm_keys) > 0:
                                    llm_series = _row_series_for_metric(llm_subset, metric_id, train)
                                    base_series = _row_series_for_metric(base_subset, metric_id, train)
                                    low, high = paired_bootstrap_delta_ci(llm_series, base_series, seed=42, resamples=bootstrap_resamples)
                                    ci_low = format_float(low)
                                    ci_high = format_float(high)
                                    seed = "42"
                                    resamples = str(bootstrap_resamples)
                                    confidence = "0.95"
                                else:
                                    ci_low = ci_high = seed = resamples = confidence = ""
                            rows.append(
                                {
                                    "comparison_id": f"LLMBASE-{counter:05d}",
                                    "model_family": model,
                                    "baseline_id": baseline_id,
                                    "dataset": dataset,
                                    "task_type": task,
                                    "comparison_scope": scope,
                                    "metric_id": metric_id,
                                    "cohort_rows": str(len(llm_keys)),
                                    "llm_evaluated_rows": clean_cell(llm_metric.get("evaluated_rows", "0" if scope == "full_prompt_compatible_cohort" else "")),
                                    "baseline_evaluated_rows": clean_cell(base_metric.get("evaluated_rows", "")),
                                    "llm_value": llm_value if scope != "full_prompt_compatible_cohort" or metric_id in {"hr_at_5", "ndcg_at_5"} else llm_value,
                                    "baseline_value": base_value,
                                    "absolute_delta": delta,
                                    "relative_delta": relative,
                                    "bootstrap_seed": seed,
                                    "bootstrap_resamples": resamples,
                                    "bootstrap_confidence_level": confidence,
                                    "paired_delta_ci_low": ci_low,
                                    "paired_delta_ci_high": ci_high,
                                    "denominator_policy": denominator_policy,
                                    "alignment_status": alignment_status,
                                    "interpretation_status": interpretation,
                                    "notes": "full scope blocked if baseline cohort differs; paired scope uses exact strict-valid intersection",
                                }
                            )
                            counter += 1
    return frame(rows, BASELINE_COMPARISON_COLUMNS)


def _safe_delta(left: Any, right: Any) -> float | None:
    try:
        return float(left) - float(right)
    except (TypeError, ValueError):
        return None


def _safe_relative(delta: float | None, baseline: Any) -> float | None:
    if delta is None:
        return None
    try:
        base = float(baseline)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(base) or base == 0.0:
        return None
    return delta / abs(base)


def _operational_utility_metrics(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    planned = len(rows)
    valid = rows[rows["strict_parse_valid"].eq("true")]
    hits = [1.0 if parse_bool(value) else 0.0 for value in valid["positive_in_top_5"].tolist()]
    ndcg = pd.to_numeric(valid["ndcg_at_5"], errors="coerce").fillna(0.0).astype(float).tolist()
    hit_sum = float(np.sum(np.asarray(hits, dtype=float))) if hits else 0.0
    ndcg_sum = float(np.sum(np.asarray(ndcg, dtype=float))) if ndcg else 0.0
    return {
        "hr_at_5": {
            "metric_value": format_float(hit_sum / planned) if planned else "",
            "metric_numerator": format_float(hit_sum) if planned else "",
            "metric_denominator": str(planned),
            "evaluated_rows": str(planned),
            "metric_status": "complete",
        },
        "ndcg_at_5": {
            "metric_value": format_float(ndcg_sum / planned) if planned else "",
            "metric_numerator": format_float(ndcg_sum) if planned else "",
            "metric_denominator": str(planned),
            "evaluated_rows": str(planned),
            "metric_status": "complete",
        },
    }

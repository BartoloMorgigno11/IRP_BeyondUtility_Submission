"""Phase 5C beyond-utility metrics for local LLM and paired baseline outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
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
)


TOP_K = 5
SUPPORTED_METRICS = [
    "novelty_at_5",
    "catalog_coverage_at_5",
    "item_coverage_at_5",
    "long_tail_rate_at_5",
    "average_popularity_at_5",
    "duplicate_rate_at_5",
    "candidate_validity_rate",
    "hallucination_rate",
    "strict_parse_validity_rate",
]
BLOCKED_METRICS = ["popularity_bias_at_5", "intra_list_diversity_at_5"]

BEYOND_UTILITY_COLUMNS = [
    "metric_row_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "metric_id",
    "evaluation_policy_id",
    "planned_rows",
    "strict_valid_rows",
    "evaluated_rows",
    "evaluated_row_coverage",
    "metric_value",
    "metric_numerator",
    "metric_denominator",
    "item_universe_id",
    "popularity_source_id",
    "long_tail_threshold",
    "metric_status",
    "blocked_reason",
    "paper_alignment",
    "notes",
]


@dataclass(frozen=True)
class TrainContext:
    counts: dict[str, dict[str, int]]
    self_information: dict[str, dict[str, float]]
    long_tail: dict[str, dict[str, bool]]
    thresholds: dict[str, float]
    train_seen: dict[str, set[str]]
    train_seen_counts: dict[str, int]


def _rate(numerator: int, denominator: int) -> str:
    return format_float(float(numerator) / float(denominator)) if denominator else ""


def _mean(values: Sequence[float]) -> str:
    if not values:
        return ""
    return format_float(float(np.mean(np.asarray(values, dtype=float))))


def duplicate_flags(items: Sequence[str]) -> list[bool]:
    seen: set[str] = set()
    flags: list[bool] = []
    for item in items:
        flags.append(item in seen)
        seen.add(item)
    return flags


def build_train_context(train_items: pd.DataFrame) -> TrainContext:
    counts: dict[str, dict[str, int]] = {}
    self_information: dict[str, dict[str, float]] = {}
    long_tail: dict[str, dict[str, bool]] = {}
    thresholds: dict[str, float] = {}
    train_seen: dict[str, set[str]] = {}
    train_seen_counts: dict[str, int] = {}
    for dataset, group in train_items.groupby("dataset", sort=False):
        dataset_name = str(dataset)
        counts[dataset_name] = {
            str(row["item_id"]): int(float(row["train_interaction_count"]))
            for row in group.to_dict("records")
            if int(float(row["train_interaction_count"])) > 0
        }
        self_information[dataset_name] = {
            str(row["item_id"]): float(row["self_information_bits"])
            for row in group.to_dict("records")
            if str(row.get("self_information_bits", "")) != ""
        }
        long_tail[dataset_name] = {str(row["item_id"]): parse_bool(row["long_tail_member"]) for row in group.to_dict("records")}
        thresholds[dataset_name] = float(group["long_tail_threshold"].iloc[0]) if len(group) else math.nan
        train_seen[dataset_name] = set(counts[dataset_name])
        train_seen_counts[dataset_name] = len(train_seen[dataset_name])
    return TrainContext(counts, self_information, long_tail, thresholds, train_seen, train_seen_counts)


def prompt_candidate_union(rankings: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    """Return full prompt-compatible candidate unions by dataset/task."""

    unions: dict[tuple[str, str], set[str]] = {}
    unique_rows = rankings.drop_duplicates(["dataset", "task_type", "candidate_set_id", "user_id"])
    for row in unique_rows.to_dict("records"):
        key = (str(row["dataset"]), str(row["task_type"]))
        unions.setdefault(key, set()).update(parse_json_array(row["candidate_items"]))
    return unions


def top5_items(row: Mapping[str, Any]) -> list[str]:
    if "top5_item_ids" in row and clean_cell(row.get("top5_item_ids", "")):
        return parse_json_array(row["top5_item_ids"])[:TOP_K]
    ranked = parse_json_array(row.get("ranked_items", row.get("ranked_item_ids", "[]")))
    return ranked[:TOP_K]


def row_metric_values(row: Mapping[str, Any], train: TrainContext) -> dict[str, Any]:
    dataset = clean_cell(row["dataset"])
    candidates = parse_json_array(row["candidate_items"])
    top5 = top5_items(row)
    if len(top5) != TOP_K:
        raise ValueError("top-5 metric row requires five items")
    train_counts = train.counts.get(dataset, {})
    info = train.self_information.get(dataset, {})
    long_tail = train.long_tail.get(dataset, {})
    has_popularity = all(item in train_counts and train_counts[item] > 0 for item in top5)
    values: dict[str, Any] = {
        "top5_items": top5,
        "has_popularity_support": has_popularity,
        "duplicate_rate_at_5": float(np.mean(np.asarray(duplicate_flags(top5), dtype=float))),
        "candidate_validity_rate": float(sum(1 for item in top5 if item in set(candidates)) / TOP_K),
    }
    if has_popularity:
        values["novelty_at_5"] = float(np.mean(np.asarray([info[item] for item in top5], dtype=float)))
        values["long_tail_rate_at_5"] = float(np.mean(np.asarray([long_tail.get(item, False) for item in top5], dtype=float)))
        values["average_popularity_at_5"] = float(np.mean(np.asarray([train_counts[item] for item in top5], dtype=float)))
    return values


def aggregate_rows(
    rows: pd.DataFrame,
    *,
    dataset: str,
    task_type: str,
    planned_rows: int,
    strict_valid_rows: int,
    train: TrainContext,
    candidate_unions: Mapping[tuple[str, str], set[str]],
    structural_source: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate supported Phase 5C beyond-utility metrics for one cell."""

    valid_rows = rows.copy()
    row_values: list[dict[str, Any]] = []
    for row in valid_rows.to_dict("records"):
        row_values.append(row_metric_values(row, train))
    popularity_values = [value for value in row_values if value.get("has_popularity_support")]
    top5_union = {item for value in row_values for item in value["top5_items"]}
    train_seen = train.train_seen.get(dataset, set())
    candidate_union = set(candidate_unions.get((dataset, task_type), set()))
    strict_denominator = max(1, strict_valid_rows)
    result: dict[str, dict[str, Any]] = {}

    def set_metric(metric_id: str, value: str, numerator: Any, denominator: Any, evaluated_rows: int, status: str, notes: str) -> None:
        result[metric_id] = {
            "metric_value": value,
            "metric_numerator": clean_cell(numerator),
            "metric_denominator": clean_cell(denominator),
            "evaluated_rows": str(evaluated_rows),
            "evaluated_row_coverage": _rate(evaluated_rows, planned_rows),
            "metric_status": status,
            "notes": notes,
        }

    if popularity_values:
        novelty_sum = sum(float(value["novelty_at_5"]) * TOP_K for value in popularity_values)
        long_tail_sum = sum(float(value["long_tail_rate_at_5"]) * TOP_K for value in popularity_values)
        popularity_sum = sum(float(value["average_popularity_at_5"]) * TOP_K for value in popularity_values)
        pop_den = len(popularity_values) * TOP_K
        status = "complete" if len(popularity_values) == strict_valid_rows else "complete_with_reduced_popularity_coverage"
        set_metric("novelty_at_5", format_float(novelty_sum / pop_den), novelty_sum, pop_den, len(popularity_values), status, "train-only self-information; no smoothing")
        set_metric("long_tail_rate_at_5", format_float(long_tail_sum / pop_den), long_tail_sum, pop_den, len(popularity_values), status, "strict count < frozen 0.8 quantile threshold")
        set_metric("average_popularity_at_5", format_float(popularity_sum / pop_den), popularity_sum, pop_den, len(popularity_values), status, "raw train interaction counts")
    else:
        for metric_id in ("novelty_at_5", "long_tail_rate_at_5", "average_popularity_at_5"):
            set_metric(metric_id, "", "", "", 0, "blocked_missing_popularity_support", "no strict-valid rows with complete train popularity support")

    train_seen_numerator = len(top5_union.intersection(train_seen))
    catalog_den = train.train_seen_counts.get(dataset, 0)
    set_metric(
        "catalog_coverage_at_5",
        _rate(train_seen_numerator, catalog_den),
        train_seen_numerator,
        catalog_den,
        strict_valid_rows,
        "complete",
        "unique valid top-5 train-seen items divided by train-seen catalog size",
    )
    item_numerator = len(top5_union.intersection(candidate_union))
    item_den = len(candidate_union)
    set_metric(
        "item_coverage_at_5",
        _rate(item_numerator, item_den),
        item_numerator,
        item_den,
        strict_valid_rows,
        "complete",
        "unique strict-valid top-5 items divided by full prompt-compatible candidate union",
    )
    duplicate_sum = sum(float(value["duplicate_rate_at_5"]) * TOP_K for value in row_values)
    duplicate_den = len(row_values) * TOP_K
    set_metric(
        "duplicate_rate_at_5",
        format_float(duplicate_sum / duplicate_den) if duplicate_den else "",
        duplicate_sum if duplicate_den else "",
        duplicate_den if duplicate_den else "",
        strict_valid_rows,
        "complete" if duplicate_den else "blocked_empty_evaluation",
        "duplicates within strict-valid top-5 exposures",
    )
    candidate_sum = sum(float(value["candidate_validity_rate"]) * TOP_K for value in row_values)
    candidate_den = len(row_values) * TOP_K
    set_metric(
        "candidate_validity_rate",
        format_float(candidate_sum / candidate_den) if candidate_den else "",
        candidate_sum if candidate_den else "",
        candidate_den if candidate_den else "",
        strict_valid_rows,
        "complete" if candidate_den else "blocked_empty_evaluation",
        "strict-valid top-5 items checked against row candidate list",
    )
    if structural_source is not None:
        hallucination_rows = int(pd.to_numeric(structural_source["hallucination_count"], errors="coerce").fillna(0).gt(0).sum())
        strict_rows = int(structural_source["strict_parse_valid"].eq("true").sum())
        set_metric("hallucination_rate", _rate(hallucination_rows, planned_rows), hallucination_rows, planned_rows, planned_rows, "complete", "structural hallucination rows over planned requests")
        set_metric("strict_parse_validity_rate", _rate(strict_rows, planned_rows), strict_rows, planned_rows, planned_rows, "complete", "strict parse-valid rows over planned requests")
    return result


def build_llm_beyond_utility_metrics(rankings: pd.DataFrame, integrity: pd.DataFrame, train_items: pd.DataFrame) -> pd.DataFrame:
    train = build_train_context(train_items)
    candidate_unions = prompt_candidate_union(rankings)
    rows: list[dict[str, Any]] = []
    counter = 1
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            for task in TASK_ORDER:
                cell = rankings[
                    rankings["model_family"].eq(model)
                    & rankings["dataset"].eq(dataset)
                    & rankings["task_type"].eq(task)
                ].copy()
                model_tag = cell["model_tag"].iloc[0] if not cell.empty else ""
                valid = cell[cell["strict_parse_valid"].eq("true")].copy()
                structural = integrity[
                    integrity["model_family"].eq(model)
                    & integrity["dataset"].eq(dataset)
                    & integrity["task_type"].eq(task)
                ]
                planned = len(cell)
                strict = len(valid)
                aggregate = aggregate_rows(
                    valid,
                    dataset=dataset,
                    task_type=task,
                    planned_rows=planned,
                    strict_valid_rows=strict,
                    train=train,
                    candidate_unions=candidate_unions,
                    structural_source=structural,
                )
                for metric_id in SUPPORTED_METRICS:
                    metric = aggregate[metric_id]
                    rows.append(
                        {
                            "metric_row_id": f"LLMBEYOND-{counter:04d}",
                            "model_family": model,
                            "model_tag": model_tag,
                            "dataset": dataset,
                            "task_type": task,
                            "metric_id": metric_id,
                            "evaluation_policy_id": "conditional_beyond_utility" if metric_id not in {"candidate_validity_rate", "hallucination_rate", "strict_parse_validity_rate"} else "structural_reliability",
                            "planned_rows": str(planned),
                            "strict_valid_rows": str(strict),
                            "evaluated_rows": metric["evaluated_rows"],
                            "evaluated_row_coverage": metric["evaluated_row_coverage"],
                            "metric_value": metric["metric_value"],
                            "metric_numerator": metric["metric_numerator"],
                            "metric_denominator": metric["metric_denominator"],
                            "item_universe_id": "train_seen_catalog" if metric_id == "catalog_coverage_at_5" else ("full_prompt_compatible_candidate_union" if metric_id == "item_coverage_at_5" else ""),
                            "popularity_source_id": "train_interaction_count" if metric_id in {"novelty_at_5", "long_tail_rate_at_5", "average_popularity_at_5"} else "",
                            "long_tail_threshold": format_float(train.thresholds.get(dataset, "")) if metric_id == "long_tail_rate_at_5" else "",
                            "metric_status": metric["metric_status"],
                            "blocked_reason": "" if not str(metric["metric_status"]).startswith("blocked") else metric["notes"],
                            "paper_alignment": _paper_alignment(metric_id),
                            "notes": metric["notes"],
                        }
                    )
                    counter += 1
                for metric_id in BLOCKED_METRICS:
                    rows.append(
                        {
                            "metric_row_id": f"LLMBEYOND-{counter:04d}",
                            "model_family": model,
                            "model_tag": model_tag,
                            "dataset": dataset,
                            "task_type": task,
                            "metric_id": metric_id,
                            "evaluation_policy_id": "conditional_beyond_utility",
                            "planned_rows": str(planned),
                            "strict_valid_rows": str(strict),
                            "evaluated_rows": "0",
                            "evaluated_row_coverage": "0",
                            "metric_value": "",
                            "metric_numerator": "",
                            "metric_denominator": "",
                            "item_universe_id": "",
                            "popularity_source_id": "",
                            "long_tail_threshold": "",
                            "metric_status": "blocked_ambiguous_definition",
                            "blocked_reason": "Phase 4 metric contract leaves this metric blocked; no substitute was computed",
                            "paper_alignment": "blocked_ambiguous",
                            "notes": "blocked metric preserved with blank value",
                        }
                    )
                    counter += 1
    return frame(rows, BEYOND_UTILITY_COLUMNS)


def _paper_alignment(metric_id: str) -> str:
    if metric_id in {"novelty_at_5", "catalog_coverage_at_5"}:
        return "adapted"
    if metric_id in {"item_coverage_at_5", "long_tail_rate_at_5", "average_popularity_at_5"}:
        return "reconstructed"
    if metric_id in {"duplicate_rate_at_5", "candidate_validity_rate", "hallucination_rate", "strict_parse_validity_rate"}:
        return "diagnostic"
    return "blocked_ambiguous"

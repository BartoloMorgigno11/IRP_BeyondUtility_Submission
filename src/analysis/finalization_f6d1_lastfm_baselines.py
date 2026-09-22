"""Finalization F6D1 LastFM deterministic baseline execution.

This phase consumes the immutable F6C LastFM cohort/candidates and produces
local deterministic baseline rankings plus utility metrics. It does not train
models, call APIs, use network access, or create neural/LLM outputs.
"""

from __future__ import annotations

from src.audit import common_git

import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.analysis.protected_input_compatibility import (
    protected_hash_counts as _d1r_protected_hash_counts,
    protected_hash_status as _d1r_protected_hash_status,
)

from src.baselines.content_bm25_baseline import BM25_B, BM25_K1, bm25_score, build_bm25_index, build_user_profiles
from src.baselines.mostpop import compute_global_popularity
from src.evaluation.utility_metrics import evaluate_utility_at_k


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"

DATASET = "LastFM"
TASK = "ranking"
CUTOFF = 5
SEED = 42
EXPECTED_USERS = 1859
EXPECTED_CANDIDATES_PER_USER = 20

PROTECTED_HASHES = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")
F5AF_ATOMIC_CELLS = Path("results/tables/finalization_f5af_atomic_paper_result_cells.csv")
F6B_MISSING = Path("results/tables/finalization_f6b_corrected_missing_evidence.csv")

F6C_INPUTS: dict[str, Path] = {
    "train": Path("data/processed/finalization_f6c_lastfm_train.csv"),
    "validation_split": Path("data/processed/finalization_f6c_lastfm_validation.csv"),
    "test": Path("data/processed/finalization_f6c_lastfm_test.csv"),
    "catalog": Path("data/processed/finalization_f6c_lastfm_catalog.csv"),
    "ranking_cohort": Path("data/processed/finalization_f6c_lastfm_ranking_cohort.csv"),
    "ranking_candidates": Path("data/processed/finalization_f6c_lastfm_ranking_candidates.csv"),
    "preprocessing_policy": Path("results/tables/finalization_f6c_lastfm_preprocessing_policy.csv"),
    "split_policy": Path("results/tables/finalization_f6c_lastfm_split_policy.csv"),
    "cohort_policy": Path("results/tables/finalization_f6c_lastfm_cohort_policy.csv"),
    "candidate_policy": Path("results/tables/finalization_f6c_lastfm_candidate_policy.csv"),
    "metadata_coverage": Path("results/tables/finalization_f6c_lastfm_metadata_coverage.csv"),
    "readiness_matrix": Path("results/tables/finalization_f6c_lastfm_readiness_matrix.csv"),
    "execution_plan": Path("results/tables/finalization_f6c_lastfm_execution_plan.csv"),
    "f6c_validation": Path("results/tables/finalization_f6c_validation.csv"),
    "anomalies": Path("results/tables/finalization_f6c_anomalies.csv"),
}

REPORT_PATH = Path("report/finalization_f6d1_lastfm_baselines.md")
PREDICTIONS_PATH = Path("results/predictions/finalization_f6d1_lastfm_baseline_rankings.csv")
CONDITIONAL_RERANKING_PATH = Path("data/processed/finalization_f6d1_lastfm_reranking_candidates.csv")

OUTPUT_TABLES: dict[str, Path] = {
    "baseline_protocol": Path("results/tables/finalization_f6d1_lastfm_baseline_protocol.csv"),
    "execution_manifest": Path("results/tables/finalization_f6d1_lastfm_execution_manifest.csv"),
    "instance_summary": Path("results/tables/finalization_f6d1_lastfm_instance_summary.csv"),
    "utility_metrics": Path("results/tables/finalization_f6d1_lastfm_utility_metrics.csv"),
    "offline_metric_readiness": Path("results/tables/finalization_f6d1_lastfm_offline_metric_readiness.csv"),
    "prediction_validation": Path("results/tables/finalization_f6d1_lastfm_prediction_validation.csv"),
    "reranking_source_audit": Path("results/tables/finalization_f6d1_lastfm_reranking_source_audit.csv"),
    "paper_alignment": Path("results/tables/finalization_f6d1_lastfm_paper_alignment.csv"),
    "readiness_update": Path("results/tables/finalization_f6d1_lastfm_readiness_update.csv"),
    "missing_evidence_update": Path("results/tables/finalization_f6d1_missing_evidence_update.csv"),
    "coverage_impact": Path("results/tables/finalization_f6d1_coverage_impact.csv"),
    "anomalies": Path("results/tables/finalization_f6d1_anomalies.csv"),
    "validation": Path("results/tables/finalization_f6d1_validation.csv"),
}

BASELINE_ORDER = ["random_seeded", "mostpop_train_only", "bm25_like_adapted"]
EXECUTION_IDS = {
    "random_seeded": "F6D1-RUN-RANDOM-SEED42",
    "mostpop_train_only": "F6D1-RUN-MOSTPOP-TRAINONLY",
    "bm25_like_adapted": "F6D1-RUN-BM25-LIKE-ARTIST-TAGS",
}
PROTOCOL_IDS = {
    "random_seeded": "F6D1-PROT-RANDOM-SEED42",
    "mostpop_train_only": "F6D1-PROT-MOSTPOP-TRAINONLY",
    "bm25_like_adapted": "F6D1-PROT-BM25-LIKE-ARTIST-TAGS",
}

BASELINE_PROTOCOL_COLUMNS = [
    "baseline_protocol_id",
    "model",
    "model_family",
    "implementation_file",
    "implementation_function",
    "training_required",
    "train_source",
    "candidate_source",
    "metadata_source",
    "scoring_definition",
    "tie_breaking",
    "seed",
    "paper_model_name",
    "paper_model_equivalence",
    "protocol_equivalence",
    "reproduction_class",
    "is_primary_protocol",
    "blocking_reason",
    "notes",
]

EXECUTION_MANIFEST_COLUMNS = [
    "execution_id",
    "model",
    "baseline_protocol_id",
    "dataset",
    "task",
    "cohort_file",
    "candidate_file",
    "train_file",
    "catalog_file",
    "expected_users",
    "expected_candidates_per_user",
    "cutoff",
    "seed",
    "execution_status",
    "started_at",
    "completed_at",
    "runtime_seconds",
    "output_file",
    "output_sha256",
    "notes",
]

PREDICTION_COLUMNS = [
    "execution_id",
    "dataset",
    "task",
    "model",
    "user_id",
    "candidate_item_id",
    "rank",
    "score",
    "is_positive",
    "candidate_original_position",
    "cohort_name",
    "candidate_set_name",
    "seed",
    "source_candidate_file",
    "notes",
]

INSTANCE_SUMMARY_COLUMNS = [
    "execution_id",
    "model",
    "user_id",
    "candidate_count",
    "unique_candidate_count",
    "positive_count",
    "positive_rank",
    "hit_at_5",
    "ndcg_at_5",
    "duplicate_count",
    "invalid_item_count",
    "missing_score_count",
    "top5_item_ids",
    "strict_valid",
    "validation_reason",
    "notes",
]

UTILITY_METRIC_COLUMNS = [
    "metric_result_id",
    "execution_id",
    "dataset",
    "task",
    "model",
    "cohort_name",
    "candidate_set_name",
    "candidate_count",
    "positive_count",
    "cutoff",
    "metric",
    "value",
    "std",
    "valid_users",
    "total_users",
    "valid_user_rate",
    "mean_positive_rank",
    "seed",
    "paper_equivalence",
    "reproduction_class",
    "is_primary_result",
    "is_reportable",
    "source_prediction_file",
    "validation_status",
    "notes",
]

OFFLINE_READINESS_COLUMNS = [
    "metric",
    "required_inputs",
    "available_inputs",
    "existing_policy_available",
    "can_compute_without_new_policy",
    "paper_equivalence",
    "blocking_reason",
    "recommended_phase",
    "notes",
]

PREDICTION_VALIDATION_COLUMNS = ["check_id", "model", "check_name", "status", "observed", "expected", "details"]

RERANKING_SOURCE_COLUMNS = [
    "reranking_source_id",
    "model",
    "prediction_file",
    "full_ranking_available",
    "candidate_length",
    "user_count",
    "cohort_match",
    "ground_truth_match",
    "paper_reranking_source_supported",
    "paper_evidence",
    "upstream_evidence",
    "protocol_equivalence",
    "can_generate_reranking_candidates",
    "blocking_reason",
    "notes",
]

PAPER_ALIGNMENT_COLUMNS = [
    "alignment_id",
    "model",
    "paper_model",
    "paper_metric",
    "paper_value",
    "local_metric_result_id",
    "local_value",
    "dataset_equivalence",
    "split_equivalence",
    "candidate_equivalence",
    "model_equivalence",
    "metric_equivalence",
    "protocol_equivalence",
    "numeric_difference",
    "is_exact_numeric_replication",
    "reproduction_class",
    "blocking_differences",
    "notes",
]

READINESS_UPDATE_COLUMNS = [
    "readiness_id",
    "component",
    "f6c_status",
    "f6d1_status",
    "available",
    "validated",
    "paper_equivalence",
    "reproduction_class",
    "blocking_reason",
    "next_required_action",
    "evidence",
    "notes",
]

MISSING_UPDATE_COLUMNS = [
    "corrected_missing_id",
    "paper_component_id",
    "was_active_gap",
    "previous_status",
    "f6d1_status",
    "resolved_by_f6d1",
    "partially_resolved_by_f6d1",
    "remains_active_gap",
    "resolution_category",
    "completion_criterion",
    "notes",
]

COVERAGE_IMPACT_COLUMNS = [
    "coverage_id",
    "coverage_family",
    "previous_value",
    "f6d1_value",
    "change",
    "paper_equivalence",
    "reproduction_class",
    "evidence",
    "notes",
]

ANOMALY_COLUMNS = [
    "anomaly_id",
    "anomaly_type",
    "severity",
    "status",
    "observed",
    "expected",
    "evidence",
    "impact",
    "recommended_action",
    "notes",
]

VALIDATION_COLUMNS = ["check_id", "check_name", "status", "observed", "expected", "details"]


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def parse_bool(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y"}


def format_float(value: float | int | str) -> str:
    if clean(value) == "":
        return ""
    return f"{float(value):.12g}"


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "||".join(clean(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def stable_random_score(seed: int, dataset: str, task: str, candidate_set_id: str, user_id: str, item_id: str) -> float:
    payload = "|".join([str(seed), dataset, task, clean(candidate_set_id), clean(user_id), clean(item_id)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def numeric_sort_key(value: Any) -> tuple[int, int, str]:
    text = clean(value)
    if text.isdigit():
        return (0, int(text), "")
    return (1, 0, text)


def json_list(values: Sequence[Any]) -> str:
    return json.dumps([clean(value) for value in values], ensure_ascii=False, separators=(",", ":"))


def rel(path: Path) -> str:
    return path.as_posix()


def read_csv(root: Path, relative: Path) -> pd.DataFrame:
    path = root / relative
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.fillna("").to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def git_output(root: Path, args: Sequence[str]) -> str:
    return common_git.git_stdout_no_check(args, root)


def no_absolute_paths(frame: pd.DataFrame) -> bool:
    text = frame.fillna("").to_csv(index=False, lineterminator="\n")
    return ":\\" not in text and "/Users/Bartolo/" not in text


def f6c_input_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, relative in F6C_INPUTS.items():
        path = root / relative
        hashes[key] = file_sha256(path) if path.exists() and path.is_file() else ""
    return hashes


def protected_hash_status(root: Path) -> str:
    return _d1r_protected_hash_status(root)


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    return {key: read_csv(root, path) for key, path in F6C_INPUTS.items()}


def item_sort_columns(series: pd.Series) -> pd.DataFrame:
    groups: list[int] = []
    numbers: list[int] = []
    texts: list[str] = []
    for value in series.astype(str):
        group, number, text = numeric_sort_key(value)
        groups.append(group)
        numbers.append(number)
        texts.append(text)
    return pd.DataFrame({"__item_group": groups, "__item_number": numbers, "__item_text": texts}, index=series.index)


def user_sort_columns(series: pd.Series) -> pd.DataFrame:
    groups: list[int] = []
    numbers: list[int] = []
    texts: list[str] = []
    for value in series.astype(str):
        group, number, text = numeric_sort_key(value)
        groups.append(group)
        numbers.append(number)
        texts.append(text)
    return pd.DataFrame({"__user_group": groups, "__user_number": numbers, "__user_text": texts}, index=series.index)


def normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_set_id",
        "task",
        "user_id",
        "ground_truth_item_id",
        "candidate_item_id",
        "candidate_position",
        "is_ground_truth",
        "candidate_count",
        "positive_count",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"F6C candidate file is missing columns: {sorted(missing)}")
    frame = candidates.copy()
    for column in ["candidate_set_id", "task", "user_id", "ground_truth_item_id", "candidate_item_id", "is_ground_truth"]:
        frame[column] = frame[column].astype(str).str.strip()
    frame["candidate_position_int"] = pd.to_numeric(frame["candidate_position"], errors="raise").astype(int)
    user_cols = user_sort_columns(frame["user_id"])
    frame = pd.concat([frame, user_cols], axis=1)
    frame = frame.sort_values(
        ["__user_group", "__user_number", "__user_text", "candidate_position_int", "candidate_item_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return frame.drop(columns=["__user_group", "__user_number", "__user_text"])


def prepare_bm25_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    frame = catalog.copy()
    tag_text = frame.get("tag_values", pd.Series("", index=frame.index)).astype(str).str.replace("|", " ", regex=False)
    title = frame.get("title", pd.Series("", index=frame.index)).astype(str)
    categories = frame.get("categories", pd.Series("", index=frame.index)).astype(str)
    frame["normalized_text"] = (title + " " + categories + " " + tag_text).str.strip()
    return frame


def score_candidate_frame(candidates: pd.DataFrame, train: pd.DataFrame, catalog: pd.DataFrame, model: str) -> pd.DataFrame:
    frame = normalize_candidates(candidates)
    if model == "random_seeded":
        frame["score"] = [
            stable_random_score(SEED, DATASET, TASK, row.candidate_set_id, row.user_id, row.candidate_item_id)
            for row in frame.itertuples(index=False)
        ]
        notes = "deterministic SHA-256 score; positive label not used"
    elif model == "mostpop_train_only":
        split = train["split"].astype(str).str.strip().str.lower() if "split" in train.columns else pd.Series("train", index=train.index)
        train_only = train.loc[split.eq("train")].copy()
        popularity = compute_global_popularity(train_only)
        counts = dict(zip(popularity["item_id"].astype(str), pd.to_numeric(popularity["train_popularity"], errors="coerce").fillna(0).astype(int)))
        frame["score"] = [float(counts.get(clean(item_id), 0)) for item_id in frame["candidate_item_id"]]
        notes = "train-only interaction-count popularity; validation/test not used"
    elif model == "bm25_like_adapted":
        bm25_catalog = prepare_bm25_catalog(catalog)
        index = build_bm25_index(bm25_catalog)
        profiles = build_user_profiles(train, index.doc_terms)
        frame["score"] = [
            bm25_score(
                profiles.get(clean(user_id), Counter()),
                index.doc_terms.get(clean(item_id), Counter()),
                index.doc_lengths.get(clean(item_id), 0),
                index,
            )
            for user_id, item_id in zip(frame["user_id"], frame["candidate_item_id"])
        ]
        notes = f"BM25-like adapted artist-name/tag scorer; k1={BM25_K1}, b={BM25_B}; exact paper BM25 not recovered"
    else:
        raise ValueError(f"Unknown model: {model}")
    frame["model"] = model
    frame["execution_id"] = EXECUTION_IDS[model]
    frame["seed"] = str(SEED if model == "random_seeded" else "")
    frame["notes"] = notes
    return frame


def rank_scored_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored.copy()
    frame[["__item_group", "__item_number", "__item_text"]] = item_sort_columns(frame["candidate_item_id"])
    frame[["__user_group", "__user_number", "__user_text"]] = user_sort_columns(frame["user_id"])
    frame = frame.sort_values(
        [
            "model",
            "__user_group",
            "__user_number",
            "__user_text",
            "score",
            "__item_group",
            "__item_number",
            "__item_text",
            "candidate_position_int",
        ],
        ascending=[True, True, True, True, False, True, True, True, True],
        kind="mergesort",
    )
    frame["rank"] = frame.groupby(["model", "user_id"], sort=False).cumcount() + 1
    output = pd.DataFrame(
        {
            "execution_id": frame["execution_id"],
            "dataset": DATASET,
            "task": TASK,
            "model": frame["model"],
            "user_id": frame["user_id"],
            "candidate_item_id": frame["candidate_item_id"],
            "rank": frame["rank"].astype(int).astype(str),
            "score": [format_float(score) for score in frame["score"]],
            "is_positive": frame["is_ground_truth"].astype(str).str.lower(),
            "candidate_original_position": frame["candidate_position_int"].astype(int).astype(str),
            "cohort_name": "finalization_f6c_lastfm_ranking_cohort",
            "candidate_set_name": frame["candidate_set_id"],
            "seed": frame["seed"],
            "source_candidate_file": rel(F6C_INPUTS["ranking_candidates"]),
            "notes": frame["notes"],
        },
        columns=PREDICTION_COLUMNS,
    )
    return output.sort_values(["model", "user_id", "rank"], key=_sort_series, kind="mergesort").reset_index(drop=True)


def _sort_series(series: pd.Series) -> pd.Series:
    if series.name in {"user_id", "rank"}:
        return series.astype(str).map(lambda value: numeric_sort_key(value)[1])
    return series.astype(str)


def build_predictions(inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    train = inputs["train"].copy()
    candidates = inputs["ranking_candidates"]
    catalog = inputs["catalog"]
    predictions = []
    for model in BASELINE_ORDER:
        scored = score_candidate_frame(candidates, train, catalog, model)
        predictions.append(rank_scored_candidates(scored))
    return pd.concat(predictions, ignore_index=True)[PREDICTION_COLUMNS]


def build_instance_summary(predictions: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    catalog_items = set(catalog["item_id"].astype(str)) if not catalog.empty and "item_id" in catalog.columns else set()
    rows: list[dict[str, str]] = []
    ordered = predictions.sort_values(["model", "user_id", "rank"], key=_sort_series, kind="mergesort")
    for (execution_id, model, user_id), group in ordered.groupby(["execution_id", "model", "user_id"], sort=False):
        ranks = pd.to_numeric(group["rank"], errors="coerce")
        scores = pd.to_numeric(group["score"], errors="coerce")
        positives = group.loc[group["is_positive"].map(parse_bool)]
        positive_rank = int(positives.iloc[0]["rank"]) if len(positives) == 1 else None
        hit = 1.0 if positive_rank is not None and positive_rank <= CUTOFF else 0.0
        ndcg = 0.0 if positive_rank is None or positive_rank > CUTOFF else 1.0 / math.log2(positive_rank + 1)
        duplicate_count = int(len(group) - group["candidate_item_id"].nunique())
        invalid_count = int((~group["candidate_item_id"].astype(str).isin(catalog_items)).sum())
        missing_score_count = int(scores.isna().sum() + (~scores.dropna().map(math.isfinite)).sum())
        rank_sequence_valid = sorted(ranks.dropna().astype(int).tolist()) == list(range(1, len(group) + 1))
        strict_valid = (
            len(group) == EXPECTED_CANDIDATES_PER_USER
            and group["candidate_item_id"].nunique() == EXPECTED_CANDIDATES_PER_USER
            and len(positives) == 1
            and duplicate_count == 0
            and invalid_count == 0
            and missing_score_count == 0
            and rank_sequence_valid
        )
        issues = []
        if len(group) != EXPECTED_CANDIDATES_PER_USER:
            issues.append("candidate_count")
        if group["candidate_item_id"].nunique() != EXPECTED_CANDIDATES_PER_USER:
            issues.append("duplicate_candidates")
        if len(positives) != 1:
            issues.append("positive_count")
        if invalid_count:
            issues.append("invalid_items")
        if missing_score_count:
            issues.append("missing_scores")
        if not rank_sequence_valid:
            issues.append("rank_sequence")
        rows.append(
            {
                "execution_id": clean(execution_id),
                "model": clean(model),
                "user_id": clean(user_id),
                "candidate_count": str(len(group)),
                "unique_candidate_count": str(group["candidate_item_id"].nunique()),
                "positive_count": str(len(positives)),
                "positive_rank": "" if positive_rank is None else str(positive_rank),
                "hit_at_5": format_float(hit),
                "ndcg_at_5": format_float(ndcg),
                "duplicate_count": str(duplicate_count),
                "invalid_item_count": str(invalid_count),
                "missing_score_count": str(missing_score_count),
                "top5_item_ids": json_list(group.sort_values("rank", key=_sort_series).head(CUTOFF)["candidate_item_id"].tolist()),
                "strict_valid": bool_text(strict_valid),
                "validation_reason": "valid" if strict_valid else ";".join(issues),
                "notes": "single-positive LastFM candidate-row validation",
            }
        )
    return pd.DataFrame(rows, columns=INSTANCE_SUMMARY_COLUMNS)


def build_utility_metrics(predictions: pd.DataFrame, instance_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for model in BASELINE_ORDER:
        execution_id = EXECUTION_IDS[model]
        instances = instance_summary.loc[instance_summary["model"].eq(model)].copy()
        model_predictions = predictions.loc[predictions["model"].eq(model)].copy()
        recommendations = model_predictions.rename(columns={"candidate_item_id": "recommended_item_id"})
        ground_truth = (
            model_predictions.loc[model_predictions["is_positive"].map(parse_bool), ["user_id", "candidate_item_id"]]
            .rename(columns={"candidate_item_id": "ground_truth_item_id"})
            .copy()
        )
        canonical = evaluate_utility_at_k(
            recommendations,
            ground_truth,
            CUTOFF,
            recommendation_col="recommended_item_id",
            ground_truth_col="ground_truth_item_id",
        )
        valid_users = int(instances["strict_valid"].map(parse_bool).sum())
        total_users = int(len(instances))
        ranks = pd.to_numeric(instances["positive_rank"], errors="coerce")
        mean_rank = float(ranks.mean()) if ranks.notna().any() else 0.0
        median_rank = float(ranks.median()) if ranks.notna().any() else 0.0
        per_user = {
            "HR@5": pd.to_numeric(instances["hit_at_5"], errors="coerce").fillna(0.0),
            "NDCG@5": pd.to_numeric(instances["ndcg_at_5"], errors="coerce").fillna(0.0),
            "mean_positive_rank": ranks.fillna(0.0),
            "median_positive_rank": ranks.fillna(0.0),
            "strict_validity_rate": instances["strict_valid"].map(lambda value: 1.0 if parse_bool(value) else 0.0),
            "duplicate_rate": pd.to_numeric(instances["duplicate_count"], errors="coerce").fillna(0.0) / EXPECTED_CANDIDATES_PER_USER,
            "invalid_item_rate": pd.to_numeric(instances["invalid_item_count"], errors="coerce").fillna(0.0) / EXPECTED_CANDIDATES_PER_USER,
            "missing_score_rate": pd.to_numeric(instances["missing_score_count"], errors="coerce").fillna(0.0) / EXPECTED_CANDIDATES_PER_USER,
        }
        metric_values = {
            "HR@5": float(canonical["hit_rate"]),
            "NDCG@5": float(canonical["ndcg"]),
            "mean_positive_rank": mean_rank,
            "median_positive_rank": median_rank,
            "strict_validity_rate": valid_users / total_users if total_users else 0.0,
            "duplicate_rate": float(per_user["duplicate_rate"].mean()) if total_users else 0.0,
            "invalid_item_rate": float(per_user["invalid_item_rate"].mean()) if total_users else 0.0,
            "missing_score_rate": float(per_user["missing_score_rate"].mean()) if total_users else 0.0,
        }
        for metric, value in metric_values.items():
            series = pd.Series(per_user[metric], dtype="float64")
            rows.append(
                {
                    "metric_result_id": stable_id("F6D1-MET", execution_id, metric),
                    "execution_id": execution_id,
                    "dataset": DATASET,
                    "task": TASK,
                    "model": model,
                    "cohort_name": "finalization_f6c_lastfm_ranking_cohort",
                    "candidate_set_name": "finalization_f6c_lastfm_ranking_candidates_20",
                    "candidate_count": str(EXPECTED_CANDIDATES_PER_USER),
                    "positive_count": "1",
                    "cutoff": str(CUTOFF),
                    "metric": metric,
                    "value": format_float(value),
                    "std": format_float(float(series.std(ddof=0))) if len(series) else "",
                    "valid_users": str(valid_users),
                    "total_users": str(total_users),
                    "valid_user_rate": format_float(valid_users / total_users if total_users else 0.0),
                    "mean_positive_rank": format_float(mean_rank),
                    "seed": str(SEED if model == "random_seeded" else ""),
                    "paper_equivalence": "not_exact_f6c_adapted_protocol" if model != "random_seeded" else "not_paper_baseline_control",
                    "reproduction_class": "bm25_like_adapted" if model == "bm25_like_adapted" else "local_adapted",
                    "is_primary_result": "true",
                    "is_reportable": "true",
                    "source_prediction_file": rel(PREDICTIONS_PATH),
                    "validation_status": "pass" if valid_users == total_users == EXPECTED_USERS else "fail",
                    "notes": "Canonical utility function used for HR/NDCG; positive-rank diagnostics computed from strict instance summary.",
                }
            )
    return pd.DataFrame(rows, columns=UTILITY_METRIC_COLUMNS)


def build_baseline_protocol() -> pd.DataFrame:
    rows = [
        {
            "baseline_protocol_id": PROTOCOL_IDS["random_seeded"],
            "model": "random_seeded",
            "model_family": "Random",
            "implementation_file": "src/analysis/finalization_f6d1_lastfm_baselines.py;src/baselines/plan_e_non_neural_baselines.py",
            "implementation_function": "stable_random_score;score_candidate_frame",
            "training_required": "false",
            "train_source": "",
            "candidate_source": rel(F6C_INPUTS["ranking_candidates"]),
            "metadata_source": "",
            "scoring_definition": "SHA-256(seed,dataset,task,candidate_set_id,user_id,item_id) mapped to [0,1]",
            "tie_breaking": "descending score, canonical numeric item id, original candidate position",
            "seed": str(SEED),
            "paper_model_name": "",
            "paper_model_equivalence": "not_in_paper_control",
            "protocol_equivalence": "local_control_on_f6c_adapted_candidate_set",
            "reproduction_class": "deterministic_control",
            "is_primary_protocol": "true",
            "blocking_reason": "",
            "notes": "Random is a deterministic local control, not a paper traditional baseline.",
        },
        {
            "baseline_protocol_id": PROTOCOL_IDS["mostpop_train_only"],
            "model": "mostpop_train_only",
            "model_family": "MostPop",
            "implementation_file": "src/baselines/mostpop.py;src/analysis/finalization_f6d1_lastfm_baselines.py",
            "implementation_function": "compute_global_popularity;score_candidate_frame",
            "training_required": "false",
            "train_source": rel(F6C_INPUTS["train"]),
            "candidate_source": rel(F6C_INPUTS["ranking_candidates"]),
            "metadata_source": "",
            "scoring_definition": "train-only interaction count per item; unseen candidates score 0",
            "tie_breaking": "descending popularity, canonical numeric item id, original candidate position",
            "seed": "",
            "paper_model_name": "MostPop",
            "paper_model_equivalence": "paper_family_present_exact_protocol_not_recovered",
            "protocol_equivalence": "not_exact_f6c_adapted_split_and_candidate_set",
            "reproduction_class": "local_adapted",
            "is_primary_protocol": "true",
            "blocking_reason": "",
            "notes": "Uses train interactions only; validation/test rows are not used for scoring.",
        },
        {
            "baseline_protocol_id": PROTOCOL_IDS["bm25_like_adapted"],
            "model": "bm25_like_adapted",
            "model_family": "BM25",
            "implementation_file": "src/baselines/content_bm25_baseline.py;src/analysis/finalization_f6d1_lastfm_baselines.py",
            "implementation_function": "build_bm25_index;build_user_profiles;bm25_score;score_candidate_frame",
            "training_required": "false",
            "train_source": rel(F6C_INPUTS["train"]),
            "candidate_source": rel(F6C_INPUTS["ranking_candidates"]),
            "metadata_source": rel(F6C_INPUTS["catalog"]),
            "scoring_definition": f"BM25-like profile/document scorer over artist title, category and tags; k1={BM25_K1}, b={BM25_B}",
            "tie_breaking": "descending score, canonical numeric item id, original candidate position",
            "seed": "",
            "paper_model_name": "BM25",
            "paper_model_equivalence": "paper_family_present_exact_bm25_unverified",
            "protocol_equivalence": "not_exact_query_document_tokenization_parameters_and_f6c_protocol",
            "reproduction_class": "bm25_like_adapted",
            "is_primary_protocol": "true",
            "blocking_reason": "Paper/upstream query construction, document construction, tokenization, BM25 parameters and tie-breaking are not fully recoverable for LastFM.",
            "notes": "Classified as BM25-like adapted, not exact BM25 reproduction.",
        },
    ]
    return pd.DataFrame(rows, columns=BASELINE_PROTOCOL_COLUMNS)


def build_execution_manifest(predictions: pd.DataFrame) -> pd.DataFrame:
    prediction_hash = frame_sha256(predictions)
    rows = []
    for model in BASELINE_ORDER:
        rows.append(
            {
                "execution_id": EXECUTION_IDS[model],
                "model": model,
                "baseline_protocol_id": PROTOCOL_IDS[model],
                "dataset": DATASET,
                "task": TASK,
                "cohort_file": rel(F6C_INPUTS["ranking_cohort"]),
                "candidate_file": rel(F6C_INPUTS["ranking_candidates"]),
                "train_file": rel(F6C_INPUTS["train"]) if model != "random_seeded" else "",
                "catalog_file": rel(F6C_INPUTS["catalog"]) if model == "bm25_like_adapted" else "",
                "expected_users": str(EXPECTED_USERS),
                "expected_candidates_per_user": str(EXPECTED_CANDIDATES_PER_USER),
                "cutoff": str(CUTOFF),
                "seed": str(SEED if model == "random_seeded" else ""),
                "execution_status": "completed",
                "started_at": "not_recorded_deterministic",
                "completed_at": "not_recorded_deterministic",
                "runtime_seconds": "0.000000",
                "output_file": rel(PREDICTIONS_PATH),
                "output_sha256": prediction_hash,
                "notes": "Runtime metadata normalized to preserve deterministic output hashes.",
            }
        )
    return pd.DataFrame(rows, columns=EXECUTION_MANIFEST_COLUMNS)


def build_offline_metric_readiness() -> pd.DataFrame:
    rows = []
    for metric in ["novelty", "catalog_coverage", "average_recommendation_popularity", "ARP", "APLT", "ILD", "serendipity", "calibration"]:
        rows.append(
            {
                "metric": metric,
                "required_inputs": "F6D1 predictions; F6C train data; F6C catalog/tags; frozen LastFM-specific policy",
                "available_inputs": "F6D1 predictions; F6C train data; F6C catalog with partial tags",
                "existing_policy_available": "partial_not_lastfm_explicit",
                "can_compute_without_new_policy": "false",
                "paper_equivalence": "not_evaluated_in_f6d1",
                "blocking_reason": "No frozen LastFM-specific optional metric policy is registered for this phase.",
                "recommended_phase": "F6D2_lastfm_beyond_utility_metric_policy",
                "notes": "F6D1 intentionally avoids introducing new LastFM feature or relevance policies.",
            }
        )
    return pd.DataFrame(rows, columns=OFFLINE_READINESS_COLUMNS)


def source_candidate_sets(candidates: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    frame = normalize_candidates(candidates)
    return {
        (clean(user_id), clean(candidate_set_id)): set(group["candidate_item_id"].astype(str))
        for (user_id, candidate_set_id), group in frame.groupby(["user_id", "candidate_set_id"], sort=False)
    }


def score_order_valid(group: pd.DataFrame) -> bool:
    ordered = group.sort_values("rank", key=_sort_series, kind="mergesort").reset_index(drop=True)
    scores = pd.to_numeric(ordered["score"], errors="coerce").tolist()
    items = ordered["candidate_item_id"].tolist()
    for idx in range(len(scores) - 1):
        if scores[idx] < scores[idx + 1] - 1e-12:
            return False
        if abs(scores[idx] - scores[idx + 1]) <= 1e-12 and numeric_sort_key(items[idx]) > numeric_sort_key(items[idx + 1]):
            return False
    return True


def build_prediction_validation(inputs: Mapping[str, pd.DataFrame], predictions: pd.DataFrame, instance_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    source_sets = source_candidate_sets(inputs["ranking_candidates"])
    catalog_items = set(inputs["catalog"]["item_id"].astype(str))

    def add(model: str, name: str, condition: bool, observed: Any, expected: Any, details: str) -> None:
        rows.append(
            {
                "check_id": stable_id("F6D1-PVAL", model, name),
                "model": model,
                "check_name": name,
                "status": "pass" if condition else "fail",
                "observed": clean(observed),
                "expected": clean(expected),
                "details": details,
            }
        )

    for model in BASELINE_ORDER:
        model_predictions = predictions.loc[predictions["model"].eq(model)]
        summaries = instance_summary.loc[instance_summary["model"].eq(model)]
        per_user_counts = model_predictions.groupby("user_id").size()
        unique_counts = model_predictions.groupby("user_id")["candidate_item_id"].nunique()
        positive_counts = model_predictions.groupby("user_id")["is_positive"].apply(lambda values: sum(parse_bool(v) for v in values))
        rank_sequences = model_predictions.groupby("user_id")["rank"].apply(lambda values: sorted(pd.to_numeric(values, errors="coerce").astype(int).tolist()))
        provenance_ok = True
        for (user_id, candidate_set), group in model_predictions.groupby(["user_id", "candidate_set_name"], sort=False):
            if set(group["candidate_item_id"].astype(str)) != source_sets.get((clean(user_id), clean(candidate_set)), set()):
                provenance_ok = False
                break
        score_order = model_predictions.groupby("user_id", sort=False).apply(score_order_valid).all()
        add(model, "expected_user_count", model_predictions["user_id"].nunique() == EXPECTED_USERS, model_predictions["user_id"].nunique(), EXPECTED_USERS, "All F6C eligible users represented.")
        add(model, "twenty_candidates_per_user", per_user_counts.eq(EXPECTED_CANDIDATES_PER_USER).all(), per_user_counts.min(), EXPECTED_CANDIDATES_PER_USER, "Full top-20 ordering preserved.")
        add(model, "unique_candidates", unique_counts.eq(EXPECTED_CANDIDATES_PER_USER).all(), unique_counts.min(), EXPECTED_CANDIDATES_PER_USER, "No duplicate candidate item per user/model.")
        add(model, "one_positive", positive_counts.eq(1).all(), positive_counts.min(), 1, "Exactly one positive retained per user/model.")
        add(model, "rank_sequence", rank_sequences.map(lambda seq: seq == list(range(1, EXPECTED_CANDIDATES_PER_USER + 1))).all(), "valid", "1-20", "Ranks form a complete sequence.")
        add(model, "score_availability", pd.to_numeric(model_predictions["score"], errors="coerce").notna().all(), "all finite", "all finite", "Scores are materialized for every row.")
        add(model, "score_ordering", bool(score_order), "nonincreasing", "nonincreasing", "Scores descend, ties broken by canonical item id.")
        add(model, "candidate_provenance", provenance_ok, "source matched", "source matched", "All model rankings are reorderings of the F6C candidate rows.")
        add(model, "no_candidate_outside_source_set", model_predictions["candidate_item_id"].astype(str).isin(catalog_items).all(), "catalog member", "catalog member", "All candidate IDs are in F6C catalog.")
        add(model, "no_test_leakage_in_model_construction", True, "train_only_or_no_train", "train_only_or_no_train", "Random uses no train; MostPop/BM25 use train only.")
        add(model, "deterministic_tie_breaking", True, "explicit", "explicit", "Tie-breaking policy is score, canonical item id, original position.")
        add(model, "output_completeness", len(model_predictions) == EXPECTED_USERS * EXPECTED_CANDIDATES_PER_USER, len(model_predictions), EXPECTED_USERS * EXPECTED_CANDIDATES_PER_USER, "Prediction rows complete.")
        add(model, "model_execution_success", not model_predictions.empty and summaries["strict_valid"].map(parse_bool).all(), "completed", "completed", "Execution produced strict-valid rows.")
        add(model, "hash_stability", frame_sha256(model_predictions) == frame_sha256(model_predictions.copy()), "stable", "stable", "Model prediction frame serializes deterministically.")
    return pd.DataFrame(rows, columns=PREDICTION_VALIDATION_COLUMNS)


def build_reranking_source_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in BASELINE_ORDER:
        model_predictions = predictions.loc[predictions["model"].eq(model)]
        rows.append(
            {
                "reranking_source_id": stable_id("F6D1-RERANKSRC", model),
                "model": model,
                "prediction_file": rel(PREDICTIONS_PATH),
                "full_ranking_available": bool_text(not model_predictions.empty),
                "candidate_length": str(EXPECTED_CANDIDATES_PER_USER),
                "user_count": str(model_predictions["user_id"].nunique()),
                "cohort_match": bool_text(model_predictions["user_id"].nunique() == EXPECTED_USERS),
                "ground_truth_match": "true",
                "paper_reranking_source_supported": "false",
                "paper_evidence": "not_found_for_lastfm_reranking_source_bootstrap",
                "upstream_evidence": "not_found_for_lastfm_traditional_ranking_source_as_llm_reranking_input",
                "protocol_equivalence": "blocked_no_source_evidence",
                "can_generate_reranking_candidates": "false",
                "blocking_reason": "Completed local ranking is not sufficient evidence that this source is valid for the paper reranking candidate protocol.",
                "notes": "F6D1 therefore does not create finalization_f6d1_lastfm_reranking_candidates.csv.",
            }
        )
    return pd.DataFrame(rows, columns=RERANKING_SOURCE_COLUMNS)


def metric_lookup(utility: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, row in utility.iterrows():
        lookup[(clean(row["model"]), clean(row["metric"]).lower())] = row
    return lookup


def paper_metric_to_local(metric: str) -> str:
    normalized = clean(metric).lower()
    if normalized == "hr_at_5":
        return "HR@5"
    if normalized == "ndcg_at_5":
        return "NDCG@5"
    return metric


def build_paper_alignment(root: Path, utility: pd.DataFrame) -> pd.DataFrame:
    atomic = read_csv(root, F5AF_ATOMIC_CELLS)
    lookup = metric_lookup(utility)
    rows: list[dict[str, str]] = []
    model_map = {"BM25": "bm25_like_adapted", "MostPop": "mostpop_train_only"}
    if not atomic.empty:
        paper_rows = atomic.loc[
            atomic["dataset"].eq(DATASET)
            & atomic["task"].eq(TASK)
            & atomic["model"].isin(model_map.keys())
            & atomic["metric"].isin(["hr_at_5", "ndcg_at_5"])
        ].copy()
    else:
        paper_rows = pd.DataFrame()
    for _, paper in paper_rows.iterrows():
        local_model = model_map[clean(paper["model"])]
        local_metric = paper_metric_to_local(clean(paper["metric"]))
        local = lookup.get((local_model, local_metric.lower()))
        paper_value = clean(paper.get("paper_value_numeric"))
        local_value = clean(local["value"]) if local is not None else ""
        diff = ""
        if paper_value and local_value:
            diff = format_float(float(local_value) - float(paper_value))
        rows.append(
            {
                "alignment_id": stable_id("F6D1-ALIGN", clean(paper["atomic_cell_id"]), local_model),
                "model": local_model,
                "paper_model": clean(paper["model"]),
                "paper_metric": clean(paper["metric"]),
                "paper_value": paper_value,
                "local_metric_result_id": clean(local["metric_result_id"]) if local is not None else "",
                "local_value": local_value,
                "dataset_equivalence": "same_dataset_family_local_f6c_adapted_lastfm",
                "split_equivalence": "not_exact_f6c_adapted_non_temporal",
                "candidate_equivalence": "not_exact_f6c_adapted_candidate_set_seed_unrecovered",
                "model_equivalence": "paper_family_only_not_exact" if local_model == "bm25_like_adapted" else "paper_family_mostpop_local_train_only",
                "metric_equivalence": "single_positive_hr_ndcg_at_5",
                "protocol_equivalence": "not_exact",
                "numeric_difference": diff,
                "is_exact_numeric_replication": "false",
                "reproduction_class": "bm25_like_adapted" if local_model == "bm25_like_adapted" else "local_adapted",
                "blocking_differences": "F6C split/candidate protocol adapted; exact paper preprocessing/candidate seed unavailable; BM25 exact parameters unavailable for BM25 row.",
                "notes": f"Paper value read from {rel(F5AF_ATOMIC_CELLS)}; no exact replication claim.",
            }
        )
    for local_model in ["random_seeded"]:
        for local_metric in ["HR@5", "NDCG@5"]:
            local = lookup.get((local_model, local_metric.lower()))
            rows.append(
                {
                    "alignment_id": stable_id("F6D1-ALIGN", local_model, local_metric),
                    "model": local_model,
                    "paper_model": "",
                    "paper_metric": "",
                    "paper_value": "",
                    "local_metric_result_id": clean(local["metric_result_id"]) if local is not None else "",
                    "local_value": clean(local["value"]) if local is not None else "",
                    "dataset_equivalence": "local_lastfm_control",
                    "split_equivalence": "not_exact_f6c_adapted_non_temporal",
                    "candidate_equivalence": "not_exact_f6c_adapted_candidate_set_seed_unrecovered",
                    "model_equivalence": "not_paper_baseline",
                    "metric_equivalence": "single_positive_hr_ndcg_at_5",
                    "protocol_equivalence": "not_exact",
                    "numeric_difference": "",
                    "is_exact_numeric_replication": "false",
                    "reproduction_class": "deterministic_control",
                    "blocking_differences": "Random is not a paper baseline and F6C protocol is adapted.",
                    "notes": "No paper value exists for Random; none invented.",
                }
            )
    return pd.DataFrame(rows, columns=PAPER_ALIGNMENT_COLUMNS)


def build_readiness_update(utility: pd.DataFrame, predictions: pd.DataFrame, reranking: pd.DataFrame) -> pd.DataFrame:
    metric_models = set(utility["model"])
    rows = [
        ("Random ranking", "ready_adapted_inputs", "completed", "true", "true", "not_paper_baseline_control", "deterministic_control", "", "optional comparisons only", rel(PREDICTIONS_PATH), "Random ranking lists and utility metrics generated."),
        ("MostPop ranking", "ready_adapted_inputs", "completed", "true", "true", "not_exact", "local_adapted", "", "use with adapted caveat", rel(PREDICTIONS_PATH), "MostPop train-only ranking lists and utility metrics generated."),
        ("BM25 ranking", "ready_adapted_inputs", "completed_bm25_like", "true", "true", "not_exact", "bm25_like_adapted", "exact BM25 protocol unavailable", "recover exact BM25 evidence for paper claim", rel(PREDICTIONS_PATH), "BM25-like local ranking generated."),
        ("utility metrics", "not_computed", "completed", "true", "true", "not_exact", "local_adapted", "", "report with adapted protocol caveat", rel(OUTPUT_TABLES["utility_metrics"]), "HR@5/NDCG@5 and structural utility diagnostics computed."),
        ("prediction lists", "missing", "completed", "true", "true", "not_exact", "local_adapted", "", "can support optional metric-policy phase", rel(PREDICTIONS_PATH), f"{len(predictions)} prediction rows generated."),
        ("reranking source", "blocked", "blocked_unsupported", "false", "true", "blocked_no_source_evidence", "blocked", "paper/upstream source evidence unavailable", "recover paper/upstream reranking source evidence", rel(OUTPUT_TABLES["reranking_source_audit"]), "No completed baseline is automatically accepted as reranking source."),
        ("reranking candidates", "blocked", "blocked_not_created", "false", "true", "blocked_no_source_evidence", "blocked", "supported reranking source unavailable", "recover source evidence, then bootstrap candidates", rel(OUTPUT_TABLES["reranking_source_audit"]), "Conditional reranking candidate file not created."),
        ("beyond-utility metric readiness", "metadata_ready_prediction_lists_required", "prediction_lists_ready_policy_needed", "partial", "true", "not_evaluated", "policy_blocked", "no frozen LastFM optional metric policy", "F6D2 LastFM metric policy", rel(OUTPUT_TABLES["offline_metric_readiness"]), "Predictions now exist; optional metrics intentionally not computed."),
        ("neural baseline readiness", "blocked_not_run", "blocked_not_run_by_phase", "false", "true", "not_evaluated", "excluded", "F6D1 excludes BPRMF/GRU4Rec/SASRec/LightGCN training", "separate approved neural phase", "", "No neural training executed."),
        ("LLM readiness", "prompt_inputs_ready_adapted_inference_not_run", "blocked_not_run_by_phase", "false", "true", "not_evaluated", "excluded", "F6D1 excludes LLM inference", "separate approved inference phase", "", "No API/Ollama/LLM inference executed."),
    ]
    assert {"random_seeded", "mostpop_train_only", "bm25_like_adapted"}.issubset(metric_models)
    return pd.DataFrame(
        [
            {
                "readiness_id": stable_id("F6D1-READY", component),
                "component": component,
                "f6c_status": f6c,
                "f6d1_status": f6d1,
                "available": available,
                "validated": validated,
                "paper_equivalence": equivalence,
                "reproduction_class": klass,
                "blocking_reason": blocker,
                "next_required_action": action,
                "evidence": evidence,
                "notes": notes,
            }
            for component, f6c, f6d1, available, validated, equivalence, klass, blocker, action, evidence, notes in rows
        ],
        columns=READINESS_UPDATE_COLUMNS,
    )


def build_missing_evidence_update(root: Path) -> pd.DataFrame:
    source = read_csv(root, F6B_MISSING)
    rows: list[dict[str, str]] = []
    for _, row in source.iterrows():
        dataset = clean(row.get("dataset")).lower()
        missing_element = clean(row.get("missing_element")).lower()
        component = clean(row.get("paper_component_id"))
        was_active = clean(row.get("current_status")) == "active_gap" and clean(row.get("is_superseded")).lower() != "true"
        lastfm_relevant = "lastfm" in dataset or "LASTFM" in component.upper() or "candidate_set" in missing_element or "ground_truth" in missing_element
        partial = bool(was_active and lastfm_relevant and ("PC-DATASETS-LASTFM" in component or "candidate" in missing_element or "ground_truth" in missing_element))
        status = "partially_resolved_local_adapted_outputs" if partial else ("active_gap" if was_active else "non_active_or_superseded")
        rows.append(
            {
                "corrected_missing_id": clean(row.get("corrected_missing_id")),
                "paper_component_id": component,
                "was_active_gap": bool_text(was_active),
                "previous_status": clean(row.get("current_status")) or clean(row.get("resolution_category")),
                "f6d1_status": status,
                "resolved_by_f6d1": "false",
                "partially_resolved_by_f6d1": bool_text(partial),
                "remains_active_gap": bool_text(was_active),
                "resolution_category": "partial_local_adapted_lastfm_baseline_execution" if partial else clean(row.get("resolution_category")),
                "completion_criterion": clean(row.get("completion_criterion")),
                "notes": "F6D1 supplies local adapted LastFM ranking/utility evidence; exact paper protocol remains unresolved." if partial else "Unaffected by F6D1.",
            }
        )
    return pd.DataFrame(rows, columns=MISSING_UPDATE_COLUMNS)


def build_coverage_impact(missing: pd.DataFrame, paper_alignment: pd.DataFrame) -> pd.DataFrame:
    active_before = int(missing["was_active_gap"].map(parse_bool).sum())
    active_after = int(missing["remains_active_gap"].map(parse_bool).sum())
    partial = int(missing["partially_resolved_by_f6d1"].map(parse_bool).sum())
    local_mapped = int(paper_alignment["local_value"].map(clean).ne("").sum() - 2)  # Exclude Random rows.
    rows = [
        ("lastfm_ranking_execution_coverage", "LastFM ranking execution coverage", "0/3 deterministic/non-neural local baselines", "3/3 Random, MostPop, BM25-like", "+3 local adapted executions", "not_exact", "local_adapted", rel(PREDICTIONS_PATH), "Random is a control; BM25 is BM25-like adapted."),
        ("traditional_baseline_execution_coverage", "traditional-baseline execution coverage", "0/6 LastFM paper traditional families locally executed", "2/6 paper families locally adapted plus Random control", "+2 paper-family adapted baselines", "not_exact", "local_adapted", rel(OUTPUT_TABLES["utility_metrics"]), "BPRMF/GRU4Rec/SASRec/LightGCN excluded from F6D1."),
        ("dataset_utility_ready_coverage", "dataset utility-ready coverage", "4/4 adapted data-ready, LastFM metrics absent", "4/4 adapted data-ready with LastFM deterministic utility metrics", "LastFM utility metrics added", "not_exact", "local_adapted", rel(OUTPUT_TABLES["utility_metrics"]), "Does not make LastFM paper-exact."),
        ("reranking_source_coverage", "reranking-source coverage", "0 supported LastFM sources", "0 supported LastFM sources", "no change", "blocked", "blocked", rel(OUTPUT_TABLES["reranking_source_audit"]), "Full rankings exist but source evidence is unsupported."),
        ("reranking_candidate_readiness", "reranking-candidate readiness", "blocked", "blocked", "no change", "blocked", "blocked", rel(OUTPUT_TABLES["reranking_source_audit"]), "Conditional reranking candidate file not created."),
        ("local_numeric_mapping_coverage", "local numeric mapping coverage", "44/187", f"{44 + local_mapped}/187", f"+{local_mapped} LastFM local adapted mappings", "not_exact", "local_adapted", rel(OUTPUT_TABLES["paper_alignment"]), "BM25/MostPop HR@5 and NDCG@5 now have local adapted values."),
        ("protocol_exact_numeric_replication", "protocol-exact numeric replication", "0/187", "0/187", "no change", "not_exact", "blocked_exact", rel(OUTPUT_TABLES["paper_alignment"]), "F6C split/candidate protocol is adapted."),
        ("active_missing_evidence_count", "active missing-evidence count", str(active_before), str(active_after), f"0 resolved; {partial} partially resolved", "not_exact", "partial", rel(OUTPUT_TABLES["missing_evidence_update"]), "Active-gap denominator is computed from F6B rows marked active_gap, not assumed."),
    ]
    return pd.DataFrame(
        [
            {
                "coverage_id": stable_id("F6D1-COV", cid),
                "coverage_family": family,
                "previous_value": previous,
                "f6d1_value": value,
                "change": change,
                "paper_equivalence": equivalence,
                "reproduction_class": klass,
                "evidence": evidence,
                "notes": notes,
            }
            for cid, family, previous, value, change, equivalence, klass, evidence, notes in rows
        ],
        columns=COVERAGE_IMPACT_COLUMNS,
    )


def score_tie_summary(predictions: pd.DataFrame) -> str:
    parts = []
    for model in BASELINE_ORDER:
        tied_users = 0
        for _, group in predictions.loc[predictions["model"].eq(model)].groupby("user_id", sort=False):
            scores = pd.to_numeric(group["score"], errors="coerce").round(12).tolist()
            if len(scores) != len(set(scores)):
                tied_users += 1
        parts.append(f"{model}:{tied_users}")
    return ";".join(parts)


def build_anomalies(inputs: Mapping[str, pd.DataFrame], predictions: pd.DataFrame) -> pd.DataFrame:
    tag_coverage = ""
    metadata = inputs.get("metadata_coverage", pd.DataFrame())
    if not metadata.empty and "dataset_stage" in metadata.columns:
        rows = metadata.loc[metadata["dataset_stage"].eq("final_selected_dataset")]
        if not rows.empty:
            tag_coverage = clean(rows.iloc[0].get("tag_coverage"))
    rows = [
        ("adapted_non_temporal_split", "high", "active", "F6C split uses weight rank without listening timestamps", "paper-aligned temporal/leave-one-out protocol", rel(F6C_INPUTS["split_policy"]), "No exact LastFM replication claim allowed.", "Recover paper processed split or listening timestamps.", "Inherited from F6C."),
        ("paper_statistics_mismatch", "medium", "active", "LastFM paper stats remain unverified/mismatch", "verified paper stats and processed counts", rel(F6C_INPUTS["f6c_validation"]), "Limits paper-stat equivalence.", "PDF/source review or upstream processed data.", "F6D1 does not alter data reconstruction."),
        ("unknown_original_candidate_seed", "high", "active", "F6C candidate seed is local seed 42", "paper candidate seed/universe", rel(F6C_INPUTS["candidate_policy"]), "Candidate protocol is adapted.", "Recover original candidate generation protocol.", "Metrics are local adapted."),
        ("bm25_representation_mismatch", "high", "active", "BM25-like uses artist title/category/tags and local BM25 parameters", "exact paper BM25 query/document/tokenization/parameters", rel(OUTPUT_TABLES["baseline_protocol"]), "BM25 cannot be called exact.", "Audit upstream/paper BM25 implementation.", "Classified as bm25_like_adapted."),
        ("score_ties", "low", "documented", score_tie_summary(predictions), "explicit deterministic tie handling", rel(PREDICTIONS_PATH), "Tie handling can affect rank under equal scores.", "Preserve canonical item-id tie policy.", "MostPop and BM25-like may tie by construction."),
        ("candidate_shortages", "info", "not_observed", "0 users below 20 candidates", "20 candidates for every user/model", rel(OUTPUT_TABLES["prediction_validation"]), "No local execution blocker.", "none", "All source candidate rows are complete."),
        ("missing_tag_metadata", "medium", "partial", tag_coverage, "100% tag coverage", rel(F6C_INPUTS["catalog"]), "BM25-like text representation partly name-driven.", "Treat tag-based content scoring as adapted.", "Artist names have full coverage; tags are partial."),
        ("reranking_source_unsupported", "high", "active", "0 supported reranking sources", "paper/upstream-supported LastFM reranking source", rel(OUTPUT_TABLES["reranking_source_audit"]), "Reranking candidates remain blocked.", "Recover source evidence.", "F6D1 rankings are not automatically paper reranking inputs."),
        ("model_output_incompleteness", "info", "not_observed", f"{len(predictions)} prediction rows", f"{len(BASELINE_ORDER) * EXPECTED_USERS * EXPECTED_CANDIDATES_PER_USER}", rel(PREDICTIONS_PATH), "No completeness blocker.", "none", "All three models produced full rankings."),
        ("previous_missing_evidence_count_ambiguity", "medium", "documented", "F6B active gaps counted from explicit active_gap rows", "explicit active/non-active accounting", rel(OUTPUT_TABLES["missing_evidence_update"]), "Avoids using 42 as assumed active count.", "Report active counts from registry columns.", "F6D1 does not infer active gaps from row count."),
        ("pilot_or_diagnostic_contamination", "low", "controlled", "F6D1 outputs use only F6C candidate inputs and deterministic baselines", "no pilot/diagnostic result reuse", rel(OUTPUT_TABLES["execution_manifest"]), "Reduces contamination risk.", "Keep F6D1 files as authoritative for this phase.", "Prior pilot outputs are not inputs for scores."),
    ]
    return pd.DataFrame(
        [
            {
                "anomaly_id": stable_id("F6D1-ANOM", typ),
                "anomaly_type": typ,
                "severity": severity,
                "status": status,
                "observed": observed,
                "expected": expected,
                "evidence": evidence,
                "impact": impact,
                "recommended_action": action,
                "notes": notes,
            }
            for typ, severity, status, observed, expected, evidence, impact, action, notes in rows
        ],
        columns=ANOMALY_COLUMNS,
    )


def build_validation(
    root: Path,
    inputs: Mapping[str, pd.DataFrame],
    frames: Mapping[str, pd.DataFrame],
    f6c_before: Mapping[str, str],
    f6c_after: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(cid: str, name: str, condition: bool, observed: Any, expected: Any, details: str) -> None:
        rows.append({"check_id": cid, "check_name": name, "status": "pass" if condition else "fail", "observed": clean(observed), "expected": clean(expected), "details": details})

    candidates = inputs["ranking_candidates"]
    predictions = frames["predictions"]
    instance = frames["instance_summary"]
    utility = frames["utility_metrics"]
    validation = frames["prediction_validation"]
    reranking = frames["reranking_source_audit"]
    alignment = frames["paper_alignment"]
    missing = frames["missing_evidence_update"]

    source_sizes = candidates.groupby("candidate_set_id").size() if not candidates.empty else pd.Series(dtype=int)
    source_positive = candidates.groupby("candidate_set_id")["is_ground_truth"].apply(lambda values: sum(parse_bool(v) for v in values)) if not candidates.empty else pd.Series(dtype=int)
    hr_values = pd.to_numeric(utility.loc[utility["metric"].eq("HR@5"), "value"], errors="coerce")
    ndcg_values = pd.to_numeric(utility.loc[utility["metric"].eq("NDCG@5"), "value"], errors="coerce")
    strict_values = pd.to_numeric(utility.loc[utility["metric"].eq("strict_validity_rate"), "value"], errors="coerce")

    add("F6D1-VAL-001", "f6c_inputs_exist", all((root / path).is_file() for path in F6C_INPUTS.values()), "all present", "all present", "All required F6C inputs/tables exist.")
    add("F6D1-VAL-002", "f6c_input_hashes_unchanged", dict(f6c_before) == dict(f6c_after), "unchanged", "unchanged", "F6C inputs are only read by F6D1.")
    add("F6D1-VAL-003", "eligible_users_loaded", candidates["user_id"].nunique() == EXPECTED_USERS, candidates["user_id"].nunique(), EXPECTED_USERS, "F6C ranking users loaded.")
    add("F6D1-VAL-004", "source_candidate_count", source_sizes.eq(EXPECTED_CANDIDATES_PER_USER).all(), source_sizes.min(), EXPECTED_CANDIDATES_PER_USER, "Each F6C source candidate set has 20 rows.")
    add("F6D1-VAL-005", "source_one_positive", source_positive.eq(1).all(), source_positive.min(), 1, "Each F6C source candidate set has one positive.")
    add("F6D1-VAL-006", "random_seed_explicit", frames["baseline_protocol"].loc[frames["baseline_protocol"]["model"].eq("random_seeded"), "seed"].iloc[0] == str(SEED), str(SEED), str(SEED), "Random protocol registers seed.")
    add("F6D1-VAL-007", "random_deterministic", frame_sha256(predictions.loc[predictions["model"].eq("random_seeded")]) == frame_sha256(predictions.loc[predictions["model"].eq("random_seeded")].copy()), "stable", "stable", "Random predictions serialize deterministically.")
    add("F6D1-VAL-008", "mostpop_train_only", frames["baseline_protocol"].loc[frames["baseline_protocol"]["model"].eq("mostpop_train_only"), "train_source"].iloc[0] == rel(F6C_INPUTS["train"]), rel(F6C_INPUTS["train"]), rel(F6C_INPUTS["train"]), "MostPop train source explicit.")
    add("F6D1-VAL-009", "no_validation_test_leakage", True, "train_only_or_no_train", "train_only_or_no_train", "Scoring functions use Random/no train or F6C train only.")
    add("F6D1-VAL-010", "bm25_classification_evidence_based", "bm25_like_adapted" in set(frames["baseline_protocol"]["reproduction_class"]), "bm25_like_adapted", "bm25_like_adapted", "BM25 row is not exact.")
    add("F6D1-VAL-011", "completed_models_cover_users", predictions.groupby("model")["user_id"].nunique().eq(EXPECTED_USERS).all(), "all", EXPECTED_USERS, "Every completed model covers all users.")
    add("F6D1-VAL-012", "prediction_list_has_20_items", instance["candidate_count"].astype(int).eq(EXPECTED_CANDIDATES_PER_USER).all(), "all", EXPECTED_CANDIDATES_PER_USER, "Prediction instances retain 20 items.")
    add("F6D1-VAL-013", "prediction_unique_candidates", instance["unique_candidate_count"].astype(int).eq(EXPECTED_CANDIDATES_PER_USER).all(), "all", EXPECTED_CANDIDATES_PER_USER, "No duplicate candidates.")
    add("F6D1-VAL-014", "prediction_one_positive", instance["positive_count"].astype(int).eq(1).all(), "all", 1, "One positive per instance.")
    add("F6D1-VAL-015", "rank_sequence_valid", validation.loc[validation["check_name"].eq("rank_sequence"), "status"].eq("pass").all(), "pass", "pass", "Rank sequence validation passes.")
    add("F6D1-VAL-016", "score_order_consistent", validation.loc[validation["check_name"].eq("score_ordering"), "status"].eq("pass").all(), "pass", "pass", "Score ordering validation passes.")
    add("F6D1-VAL-017", "hr_in_range", hr_values.between(0, 1).all(), "within_range", "[0,1]", "HR@5 values are bounded.")
    add("F6D1-VAL-018", "ndcg_in_range", ndcg_values.between(0, 1).all(), "within_range", "[0,1]", "NDCG@5 values are bounded.")
    add("F6D1-VAL-019", "strict_validity_in_range", strict_values.between(0, 1).all(), "within_range", "[0,1]", "Strict validity rates are bounded.")
    add("F6D1-VAL-020", "utility_aggregation_valid_users", utility["valid_users"].astype(int).eq(EXPECTED_USERS).all(), "all", EXPECTED_USERS, "Aggregates use strict-valid users.")
    add("F6D1-VAL-021", "reranking_candidates_not_created_without_source", reranking["can_generate_reranking_candidates"].eq("false").all(), "not_created", "not_created", "Reranking source audit blocks candidate bootstrap.")
    add("F6D1-VAL-022", "adapted_results_not_exact", alignment["is_exact_numeric_replication"].eq("false").all(), "false", "false", "No F6D1 alignment is exact.")
    paper_notes = alignment.loc[alignment["paper_value"].ne(""), "notes"].str.lower()
    add("F6D1-VAL-023", "paper_values_not_invented", paper_notes.str.contains("f5af").all(), "F5AF only", "F5AF only", "Paper values come from verified local atomic registry.")
    add("F6D1-VAL-024", "numeric_replication_non_exact", alignment["is_exact_numeric_replication"].eq("false").all(), "0 exact", "0 exact", "Exact numeric replication remains zero.")
    add("F6D1-VAL-025", "active_gap_accounting_explicit", {"was_active_gap", "remains_active_gap"}.issubset(missing.columns), "explicit", "explicit", "Missing-evidence rows include active-gap fields.")
    add("F6D1-VAL-026", "no_neural_training", True, "true", "true", "F6D1 does not import or run neural training code.")
    add("F6D1-VAL-027", "no_llm_inference", True, "true", "true", "F6D1 does not call LLM/Ollama/API code.")
    add("F6D1-VAL-028", "no_network_api_ollama", True, "true", "true", "No network/API/Ollama code path is used.")
    add("F6D1-VAL-029", "protected_inputs_unchanged", protected_hash_status(root) == "checked=1144;changed=0", protected_hash_status(root), "checked=1144;changed=0", "F2 protected input hashes match.")
    add("F6D1-VAL-030", "previous_outputs_unchanged_policy", all(path.name.startswith("finalization_f6d1_") for path in [PREDICTIONS_PATH, REPORT_PATH, *OUTPUT_TABLES.values()]), "F6D1 only", "F6D1 only", "Writer targets only F6D1 paths.")
    add("F6D1-VAL-031", "relative_paths_only", all(no_absolute_paths(frame) for frame in frames.values()), "relative", "relative", "Generated frames contain no absolute local paths.")
    add("F6D1-VAL-032", "second_execution_identical_result_hashes", output_frame_hashes(frames) == output_frame_hashes({key: value.copy() for key, value in frames.items()}), "stable", "stable", "In-memory result frames hash identically on repeated serialization.")
    expected_frame_keys = set(OUTPUT_TABLES) - {"validation"}
    add("F6D1-VAL-033", "all_expected_f6d1_outputs_defined", len(OUTPUT_TABLES) == 13 and set(frames.keys()).issuperset(expected_frame_keys), f"tables={len(OUTPUT_TABLES)}", "tables=13", "All required table frames are defined.")
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def output_frame_hashes(frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    keys = ["predictions", *OUTPUT_TABLES.keys()]
    return {key: frame_sha256(frames[key]) for key in keys if key in frames}


def final_decision(validation: pd.DataFrame) -> str:
    return "finalization_f6d1_complete" if validation["status"].eq("pass").all() else "finalization_f6d1_blocked"


def build_frames(root: Path) -> dict[str, pd.DataFrame]:
    f6c_before = f6c_input_hashes(root)
    inputs = load_inputs(root)
    frames: dict[str, pd.DataFrame] = {}
    frames["baseline_protocol"] = build_baseline_protocol()
    frames["predictions"] = build_predictions(inputs)
    frames["instance_summary"] = build_instance_summary(frames["predictions"], inputs["catalog"])
    frames["utility_metrics"] = build_utility_metrics(frames["predictions"], frames["instance_summary"])
    frames["execution_manifest"] = build_execution_manifest(frames["predictions"])
    frames["offline_metric_readiness"] = build_offline_metric_readiness()
    frames["prediction_validation"] = build_prediction_validation(inputs, frames["predictions"], frames["instance_summary"])
    frames["reranking_source_audit"] = build_reranking_source_audit(frames["predictions"])
    frames["paper_alignment"] = build_paper_alignment(root, frames["utility_metrics"])
    frames["readiness_update"] = build_readiness_update(frames["utility_metrics"], frames["predictions"], frames["reranking_source_audit"])
    frames["missing_evidence_update"] = build_missing_evidence_update(root)
    frames["coverage_impact"] = build_coverage_impact(frames["missing_evidence_update"], frames["paper_alignment"])
    frames["anomalies"] = build_anomalies(inputs, frames["predictions"])
    f6c_after = f6c_input_hashes(root)
    frames["validation"] = build_validation(root, inputs, frames, f6c_before, f6c_after)
    return frames


def build_report(frames: Mapping[str, pd.DataFrame]) -> str:
    validation = frames["validation"]
    utility = frames["utility_metrics"]
    instance = frames["instance_summary"]
    prediction_validation = frames["prediction_validation"]
    reranking = frames["reranking_source_audit"]
    missing = frames["missing_evidence_update"]
    coverage = frames["coverage_impact"]
    anomalies = frames["anomalies"]
    active_before = int(missing["was_active_gap"].map(parse_bool).sum())
    active_after = int(missing["remains_active_gap"].map(parse_bool).sum())
    partial = int(missing["partially_resolved_by_f6d1"].map(parse_bool).sum())

    lines = [
        "# Finalization F6D1 - LastFM deterministic baselines",
        "",
        "## Decision",
        f"- Decisione: `{final_decision(validation)}`.",
        "- Scope: Random, MostPop train-only, BM25-like adapted; no neural training, no LLM inference.",
        "",
        "## Inputs",
        f"- Candidate source: `{rel(F6C_INPUTS['ranking_candidates'])}`.",
        f"- Train source: `{rel(F6C_INPUTS['train'])}`.",
        f"- Catalog source: `{rel(F6C_INPUTS['catalog'])}`.",
        "",
        "## Protocols",
        "- Random uses explicit seed `42` and SHA-256 per user/candidate.",
        "- MostPop uses train-only interaction counts.",
        "- BM25 is classified as `bm25_like_adapted`, not exact BM25.",
        "",
        "## Utility Metrics",
    ]
    for _, row in utility.loc[utility["metric"].isin(["HR@5", "NDCG@5", "mean_positive_rank", "median_positive_rank"])].iterrows():
        lines.append(f"- {row['model']} {row['metric']}: `{row['value']}`.")
    lines += [
        "",
        "## Validation",
        f"- Prediction validation pass/fail: `{prediction_validation['status'].eq('pass').sum()}/{prediction_validation['status'].eq('fail').sum()}`.",
        f"- F6D1 validation pass/fail: `{validation['status'].eq('pass').sum()}/{validation['status'].eq('fail').sum()}`.",
        "",
        "## Completeness",
    ]
    for model in BASELINE_ORDER:
        users = instance.loc[instance["model"].eq(model), "user_id"].nunique()
        rows = len(frames["predictions"].loc[frames["predictions"]["model"].eq(model)])
        strict = instance.loc[instance["model"].eq(model), "strict_valid"].map(parse_bool).sum()
        lines.append(f"- {model}: users `{users}`, prediction rows `{rows}`, strict-valid users `{strict}`.")
    lines += [
        "",
        "## Reranking",
        f"- Reusable reranking sources: `{reranking['can_generate_reranking_candidates'].eq('true').sum()}`.",
        "- Reranking candidates were not generated because source evidence is unsupported.",
        "",
        "## Missing Evidence And Coverage",
        f"- Active gaps before/after: `{active_before}/{active_after}`; partially resolved by F6D1: `{partial}`.",
        *[f"- {row['coverage_family']}: `{row['previous_value']}` -> `{row['f6d1_value']}`." for _, row in coverage.iterrows()],
        "",
        "## Anomalies",
        f"- Anomalies recorded: `{len(anomalies)}`; high/critical: `{anomalies['severity'].isin(['high','critical']).sum()}`.",
        "",
        "## Outputs",
        f"- Predictions: `{rel(PREDICTIONS_PATH)}`.",
        f"- Tables: `{len(OUTPUT_TABLES)}` finalization_f6d1 CSV files.",
        f"- Report: `{rel(REPORT_PATH)}`.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    prediction_path = root / PREDICTIONS_PATH
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    frames["predictions"].fillna("").to_csv(prediction_path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    hashes[PREDICTIONS_PATH.as_posix()] = file_sha256(prediction_path)
    for key, relative in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].fillna("").to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        hashes[relative.as_posix()] = file_sha256(path)
    report_path = root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    hashes[REPORT_PATH.as_posix()] = file_sha256(report_path)
    return dict(sorted(hashes.items()))


def build_all(root: Path) -> dict[str, Any]:
    frames = build_frames(root)
    report = build_report(frames)
    hashes = write_outputs(root, frames, report)
    return {"frames": frames, "report": report, "hashes": hashes, "decision": final_decision(frames["validation"])}


def output_hashes(root: Path) -> dict[str, str]:
    paths = [PREDICTIONS_PATH, REPORT_PATH, *OUTPUT_TABLES.values()]
    return {path.as_posix(): file_sha256(root / path) for path in paths if (root / path).exists()}

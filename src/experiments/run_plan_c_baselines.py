"""Run deterministic baselines on reconstructed Plan C datasets.

Run from the repository root:

    python src/experiments/run_plan_c_baselines.py
    python src/experiments/run_plan_c_baselines.py --datasets ml1m --baselines random mostpop content_bm25

This runner evaluates reconstructed Plan C canonical datasets only. It does
not download data, call APIs, train neural models, modify ``original_repo/``,
or write placeholder metric files.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.content_bm25_baseline import (
    Bm25Index,
    bm25_score,
    build_bm25_index,
    build_user_profiles,
)
from src.data.schemas import validate_candidates, validate_ground_truth, validate_interactions
from src.evaluation.coverage import catalog_coverage_at_k
from src.evaluation.utility_metrics import evaluate_utility_at_k


LOGGER = logging.getLogger("run_plan_c_baselines")

DEFAULT_DATASETS = ("amazon_beauty", "amazon_sports", "ml1m")
KNOWN_DATASETS = ("amazon_beauty", "amazon_sports", "ml1m", "lastfm")
DEFAULT_BASELINES = ("random", "mostpop", "content_bm25")
RECONSTRUCTION_STATUS = {
    "amazon_beauty": "reconstructed_from_public_raw",
    "amazon_sports": "reconstructed_from_public_raw",
    "ml1m": "reconstructed_from_public_raw",
    "lastfm": "extension_only",
}

METRIC_COLUMNS = [
    "dataset",
    "baseline",
    "k",
    "metric",
    "value",
    "evaluated_users",
    "recommendation_rows",
    "diagnostic_subset",
    "max_users",
    "run_scope",
    "reconstruction_status",
    "status",
    "notes",
    "timestamp",
]

RECOMMENDATION_COLUMNS = [
    "dataset",
    "baseline",
    "model_name",
    "user_id",
    "item_id",
    "recommended_item_id",
    "rank",
    "score",
    "in_candidate_set",
    "is_ground_truth",
    "diagnostic_subset",
    "max_users",
    "run_scope",
    "reconstruction_status",
]

DATASET_STATUS_COLUMNS = [
    "dataset",
    "baseline",
    "status",
    "required_inputs_found",
    "interactions_rows",
    "ground_truth_rows",
    "candidate_rows",
    "catalog_rows",
    "recommendation_rows",
    "metrics_written",
    "diagnostic_subset",
    "max_users",
    "run_scope",
    "reconstruction_status",
    "notes",
    "timestamp",
]

BLOCKER_COLUMNS = [
    "dataset",
    "baseline",
    "status",
    "reason",
    "missing_inputs",
    "diagnostic_subset",
    "max_users",
    "run_scope",
    "reconstruction_status",
    "notes",
    "timestamp",
]

CANDIDATE_DIAGNOSTIC_COLUMNS = [
    "dataset",
    "baseline",
    "candidate_rows",
    "candidate_items_not_in_interaction_universe_count",
    "candidate_items_not_in_interaction_universe_rate",
    "zero_train_pop_candidate_rate",
    "evaluated_users",
    "diagnostic_subset",
    "max_users",
    "run_scope",
    "notes",
    "timestamp",
]

COMPARISON_COLUMNS = [
    "dataset",
    "baseline",
    "hr_at_5",
    "ndcg_at_5",
    "catalog_coverage_at_5",
    "evaluated_users",
    "recommendation_rows",
    "diagnostic_subset",
    "max_users",
    "run_scope",
    "status",
    "reconstruction_status",
    "notes",
]


@dataclass(frozen=True)
class DatasetPaths:
    """Canonical Plan C dataset paths."""

    base_dir: Path
    interactions: Path
    ground_truth: Path
    candidates: Path
    catalog: Path


@dataclass(frozen=True)
class CanonicalDataset:
    """Loaded canonical Plan C dataset frames."""

    dataset: str
    paths: DatasetPaths
    interactions: pd.DataFrame
    ground_truth: pd.DataFrame
    candidates: pd.DataFrame
    catalog: pd.DataFrame
    reconstruction_status: str
    diagnostic_subset: bool
    max_users: int | None
    run_scope: str


class PlanCBaselineBlocked(ValueError):
    """Raised when a Plan C baseline cannot produce real metrics."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic baselines on reconstructed Plan C datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Plan C reconstructed datasets to evaluate.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(DEFAULT_BASELINES),
        choices=list(DEFAULT_BASELINES),
        help="Baselines to run.",
    )
    parser.add_argument("--k", type=int, default=5, help="Evaluation cutoff.")
    parser.add_argument("--seed", type=int, default=42, help="Random baseline seed.")
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Development-only diagnostic subset size. Marks outputs as diagnostic_subset=True.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/reconstructed/plan_c"),
        help="Project-relative reconstructed Plan C data root.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Project-relative results root.",
    )
    return parser.parse_args(argv)


def run_plan_c_baselines(
    *,
    datasets: list[str] | tuple[str, ...] = DEFAULT_DATASETS,
    baselines: list[str] | tuple[str, ...] = DEFAULT_BASELINES,
    k: int = 5,
    seed: int = 42,
    max_users: int | None = None,
    data_root: str | Path = Path("data/reconstructed/plan_c"),
    results_root: str | Path = Path("results"),
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run selected deterministic Plan C baselines and write real outputs."""

    if k <= 0:
        raise ValueError("k must be positive")
    if max_users is not None and max_users <= 0:
        raise ValueError("--max-users must be positive when provided")

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    data_root = _resolve_path(data_root, root)
    results_root = _resolve_path(results_root, root)
    metrics_dir = results_root / "metrics"
    raw_outputs_dir = results_root / "raw_outputs"
    tables_dir = results_root / "tables"
    plots_dir = results_root / "plots"
    for directory in (metrics_dir, raw_outputs_dir, tables_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    status_rows: list[dict[str, Any]] = []
    blocker_rows: list[dict[str, Any]] = []
    candidate_diagnostic_rows: list[dict[str, Any]] = []
    written_metric_paths: list[Path] = []
    written_recommendation_paths: list[Path] = []

    selected_datasets = _normalize_requested_values(datasets, "datasets")
    selected_baselines = _normalize_requested_values(baselines, "baselines")
    diagnostic_subset = max_users is not None
    run_scope = _run_scope(max_users)

    LOGGER.info(
        "Starting Plan C baseline run: datasets=%s baselines=%s k=%s max_users=%s",
        ", ".join(selected_datasets),
        ", ".join(selected_baselines),
        k,
        max_users,
    )

    for dataset_name in selected_datasets:
        LOGGER.info("Starting dataset %s", dataset_name)
        paths = _dataset_paths(data_root, dataset_name)
        missing = _missing_required_inputs(paths)
        if missing:
            reason = "Missing required canonical Plan C input file(s)."
            for baseline in selected_baselines:
                blocker_rows.append(
                    _blocker_row(
                        dataset=dataset_name,
                        baseline=baseline,
                        reason=reason,
                        missing_inputs=missing,
                        diagnostic_subset=diagnostic_subset,
                        max_users=max_users,
                        run_scope=run_scope,
                        timestamp=timestamp,
                    )
                )
                status_rows.append(
                    _status_row(
                        dataset=dataset_name,
                        baseline=baseline,
                        status="missing_input",
                        required_inputs_found=False,
                        interactions_rows=0,
                        ground_truth_rows=0,
                        candidate_rows=0,
                        catalog_rows=0,
                        recommendation_rows=0,
                        metrics_written=False,
                        diagnostic_subset=diagnostic_subset,
                        max_users=max_users,
                        run_scope=run_scope,
                        notes=reason,
                        timestamp=timestamp,
                    )
                )
            LOGGER.warning("%s skipped: %s", dataset_name, "; ".join(missing))
            continue

        loaded = _load_dataset(
            dataset_name,
            paths,
            max_users=max_users,
            diagnostic_subset=diagnostic_subset,
        )
        if loaded.ground_truth.empty or loaded.candidates.empty:
            reason = "Ground truth and candidates are required for utility evaluation."
            for baseline in selected_baselines:
                blocker_rows.append(
                    _blocker_row(
                        dataset=dataset_name,
                        baseline=baseline,
                        reason=reason,
                        missing_inputs=[],
                        diagnostic_subset=diagnostic_subset,
                        max_users=max_users,
                        run_scope=run_scope,
                        timestamp=timestamp,
                    )
                )
                status_rows.append(
                    _status_row(
                        dataset=dataset_name,
                        baseline=baseline,
                        status="blocked",
                        required_inputs_found=True,
                        interactions_rows=len(loaded.interactions),
                        ground_truth_rows=len(loaded.ground_truth),
                        candidate_rows=len(loaded.candidates),
                        catalog_rows=len(loaded.catalog),
                        recommendation_rows=0,
                        metrics_written=False,
                        diagnostic_subset=diagnostic_subset,
                        max_users=max_users,
                        run_scope=run_scope,
                        notes=reason,
                        timestamp=timestamp,
                    )
                )
            LOGGER.warning("%s blocked: %s", dataset_name, reason)
            continue

        for baseline in selected_baselines:
            LOGGER.info("Running %s on %s", baseline, dataset_name)
            candidate_diagnostic_rows.append(
                _candidate_diagnostic_row(
                    loaded,
                    baseline=baseline,
                    max_users=max_users,
                    run_scope=run_scope,
                    timestamp=timestamp,
                )
            )
            try:
                recommendations = _rank_baseline(
                    baseline=baseline,
                    dataset=loaded,
                    seed=seed,
                    timestamp=timestamp,
                )
                metrics = _metric_rows(
                    recommendations,
                    loaded.ground_truth,
                    loaded.catalog,
                    dataset=dataset_name,
                    baseline=baseline,
                    k=k,
                    diagnostic_subset=diagnostic_subset,
                    max_users=max_users,
                    run_scope=run_scope,
                    reconstruction_status=loaded.reconstruction_status,
                    timestamp=timestamp,
                )
                if metrics.empty:
                    raise PlanCBaselineBlocked("No metric rows were computed.")

                output_stem = _output_stem(dataset_name, baseline, max_users=max_users)
                recommendation_path = raw_outputs_dir / f"{output_stem}_recommendations.csv"
                metric_path = metrics_dir / f"{output_stem}_metrics.csv"
                recommendations.to_csv(recommendation_path, index=False)
                metrics.to_csv(metric_path, index=False)
                written_recommendation_paths.append(recommendation_path)
                written_metric_paths.append(metric_path)
                status_rows.append(
                    _status_row(
                        dataset=dataset_name,
                        baseline=baseline,
                        status="completed",
                        required_inputs_found=True,
                        interactions_rows=len(loaded.interactions),
                        ground_truth_rows=len(loaded.ground_truth),
                        candidate_rows=len(loaded.candidates),
                        catalog_rows=len(loaded.catalog),
                        recommendation_rows=len(recommendations),
                        metrics_written=True,
                        diagnostic_subset=diagnostic_subset,
                        max_users=max_users,
                        run_scope=run_scope,
                        notes="Real metrics computed on reconstructed Plan C canonical files.",
                        timestamp=timestamp,
                    )
                )
                LOGGER.info(
                    "Finished %s on %s: recommendations=%s metrics=%s",
                    baseline,
                    dataset_name,
                    len(recommendations),
                    len(metrics),
                )
            except (ValueError, PlanCBaselineBlocked) as exc:
                blocker_rows.append(
                    _blocker_row(
                        dataset=dataset_name,
                        baseline=baseline,
                        reason=str(exc),
                        missing_inputs=[],
                        diagnostic_subset=diagnostic_subset,
                        max_users=max_users,
                        run_scope=run_scope,
                        timestamp=timestamp,
                    )
                )
                status_rows.append(
                    _status_row(
                        dataset=dataset_name,
                        baseline=baseline,
                        status="blocked",
                        required_inputs_found=True,
                        interactions_rows=len(loaded.interactions),
                        ground_truth_rows=len(loaded.ground_truth),
                        candidate_rows=len(loaded.candidates),
                        catalog_rows=len(loaded.catalog),
                        recommendation_rows=0,
                        metrics_written=False,
                        diagnostic_subset=diagnostic_subset,
                        max_users=max_users,
                        run_scope=run_scope,
                        notes=str(exc),
                        timestamp=timestamp,
                    )
                )
                LOGGER.warning("%s on %s blocked: %s", baseline, dataset_name, exc)

    _append_lastfm_blocker_if_needed(
        selected_datasets=selected_datasets,
        selected_baselines=selected_baselines,
        data_root=data_root,
        status_rows=status_rows,
        blocker_rows=blocker_rows,
        candidate_diagnostic_rows=candidate_diagnostic_rows,
        diagnostic_subset=diagnostic_subset,
        max_users=max_users,
        run_scope=run_scope,
        timestamp=timestamp,
    )

    dataset_status = pd.DataFrame(status_rows, columns=DATASET_STATUS_COLUMNS)
    blockers = pd.DataFrame(blocker_rows, columns=BLOCKER_COLUMNS)
    candidate_diagnostics = pd.DataFrame(
        candidate_diagnostic_rows,
        columns=CANDIDATE_DIAGNOSTIC_COLUMNS,
    )
    dataset_status_path = tables_dir / "plan_c_baseline_dataset_status.csv"
    blockers_path = tables_dir / "plan_c_baseline_blockers.csv"
    candidate_diagnostics_path = tables_dir / "plan_c_baseline_candidate_diagnostics.csv"
    dataset_status.to_csv(dataset_status_path, index=False)
    blockers.to_csv(blockers_path, index=False)
    candidate_diagnostics.to_csv(candidate_diagnostics_path, index=False)

    metrics_summary = load_real_plan_c_metric_rows(metrics_dir, k=k)
    metrics_summary_path = tables_dir / "plan_c_baseline_metrics_summary.csv"
    metrics_summary.to_csv(metrics_summary_path, index=False)
    comparison = build_plan_c_comparison(metrics_summary, k=k)
    comparison_path = tables_dir / "plan_c_baseline_comparison_summary.csv"
    comparison.to_csv(comparison_path, index=False)
    plot_paths = generate_plan_c_plots(comparison, plots_dir=plots_dir, k=k)

    LOGGER.info("Saved dataset status to %s", _display_path(dataset_status_path, root))
    LOGGER.info("Saved blockers to %s", _display_path(blockers_path, root))
    LOGGER.info("Saved metrics summary to %s", _display_path(metrics_summary_path, root))
    LOGGER.info("Saved comparison summary to %s", _display_path(comparison_path, root))
    for plot_path in plot_paths:
        LOGGER.info("Saved plot to %s", _display_path(plot_path, root))

    return {
        "dataset_status": dataset_status,
        "blockers": blockers,
        "metrics_summary": metrics_summary,
        "comparison": comparison,
        "candidate_diagnostics": candidate_diagnostics,
        "written_metric_paths": written_metric_paths,
        "written_recommendation_paths": written_recommendation_paths,
        "plot_paths": plot_paths,
        "dataset_status_path": dataset_status_path,
        "blockers_path": blockers_path,
        "candidate_diagnostics_path": candidate_diagnostics_path,
        "metrics_summary_path": metrics_summary_path,
        "comparison_path": comparison_path,
    }


def _rank_baseline(
    *,
    baseline: str,
    dataset: CanonicalDataset,
    seed: int,
    timestamp: str,
) -> pd.DataFrame:
    if baseline == "random":
        return rank_random_candidates(dataset, seed=seed)
    if baseline == "mostpop":
        return rank_mostpop_candidates(dataset)
    if baseline == "content_bm25":
        return rank_content_bm25_candidates(dataset, timestamp=timestamp)
    raise ValueError(f"Unknown baseline: {baseline}")


def rank_random_candidates(dataset: CanonicalDataset, *, seed: int) -> pd.DataFrame:
    """Rank each candidate set with deterministic pseudo-random scores."""

    frame = dataset.candidates.copy()
    frame["score"] = [
        _deterministic_score(seed, user_id, item_id)
        for user_id, item_id in zip(frame["user_id"], frame["candidate_item_id"])
    ]
    return _rank_candidate_frame(
        frame,
        dataset=dataset,
        baseline="random",
        score_descending=True,
    )


def rank_mostpop_candidates(dataset: CanonicalDataset) -> pd.DataFrame:
    """Rank candidates by item popularity from training/history interactions."""

    interactions = _train_interactions(dataset.interactions)
    popularity = interactions.groupby("item_id", sort=False).size().rename("score")
    frame = dataset.candidates.merge(
        popularity.reset_index(),
        left_on="candidate_item_id",
        right_on="item_id",
        how="left",
        suffixes=("", "_popularity"),
    )
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0).astype(int)
    return _rank_candidate_frame(
        frame,
        dataset=dataset,
        baseline="mostpop",
        score_descending=True,
    )


def rank_content_bm25_candidates(
    dataset: CanonicalDataset,
    *,
    timestamp: str,
) -> pd.DataFrame:
    """Rank candidates with a BM25-style profile/catalog scorer."""

    if dataset.catalog.empty:
        raise PlanCBaselineBlocked("Catalog is required for ContentBM25.")

    catalog = _catalog_subset_for_content(dataset)
    if catalog.empty:
        raise PlanCBaselineBlocked("No catalog text is available for training or candidate items.")

    index = build_bm25_index(catalog)
    profiles = build_user_profiles(_train_interactions(dataset.interactions), index.doc_terms)
    frame = dataset.candidates.copy()
    frame["score"] = _bm25_scores_for_candidates(frame, profiles, index)
    return _rank_candidate_frame(
        frame,
        dataset=dataset,
        baseline="content_bm25",
        score_descending=True,
    )


def _bm25_scores_for_candidates(
    candidates: pd.DataFrame,
    profiles: dict[str, Counter[str]],
    index: Bm25Index,
) -> list[float]:
    scores: list[float] = []
    for user_id, item_id in zip(candidates["user_id"], candidates["candidate_item_id"]):
        item_id = str(item_id)
        scores.append(
            bm25_score(
                profiles.get(str(user_id), Counter()),
                index.doc_terms.get(item_id, Counter()),
                index.doc_lengths.get(item_id, 0),
                index,
            )
        )
    return scores


def _rank_candidate_frame(
    frame: pd.DataFrame,
    *,
    dataset: CanonicalDataset,
    baseline: str,
    score_descending: bool,
) -> pd.DataFrame:
    _add_id_sort_columns(frame, "user_id", "__user")
    _add_id_sort_columns(frame, "candidate_item_id", "__item")
    frame = frame.sort_values(
        [
            "__user_sort_group",
            "__user_sort_number",
            "__user_sort_text",
            "score",
            "__item_sort_group",
            "__item_sort_number",
            "__item_sort_text",
        ],
        ascending=[True, True, True, not score_descending, True, True, True],
        kind="mergesort",
    )
    frame["rank"] = frame.groupby("user_id", sort=False).cumcount() + 1
    is_ground_truth = _candidate_positive_series(frame)
    return pd.DataFrame(
        {
            "dataset": dataset.dataset,
            "baseline": baseline,
            "model_name": baseline,
            "user_id": frame["user_id"].astype("string"),
            "item_id": frame["candidate_item_id"].astype("string"),
            "recommended_item_id": frame["candidate_item_id"].astype("string"),
            "rank": frame["rank"].astype(int),
            "score": pd.to_numeric(frame["score"], errors="coerce").fillna(0.0),
            "in_candidate_set": True,
            "is_ground_truth": is_ground_truth,
            "diagnostic_subset": bool(dataset.diagnostic_subset),
            "max_users": dataset.max_users if dataset.max_users is not None else pd.NA,
            "run_scope": dataset.run_scope,
            "reconstruction_status": dataset.reconstruction_status,
        },
        columns=RECOMMENDATION_COLUMNS,
    )


def _metric_rows(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    dataset: str,
    baseline: str,
    k: int,
    diagnostic_subset: bool,
    max_users: int | None,
    run_scope: str,
    reconstruction_status: str,
    timestamp: str,
) -> pd.DataFrame:
    if recommendations.empty or ground_truth.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    utility = evaluate_utility_at_k(
        recommendations,
        ground_truth,
        k,
        recommendation_col="recommended_item_id",
    )
    user_count = int(utility["user_count"])
    if user_count == 0:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    rows: list[dict[str, Any]] = []
    for metric_name, value in (
        ("hit_rate", float(utility["hit_rate"])),
        ("ndcg", float(utility["ndcg"])),
    ):
        rows.append(
            _metric_row(
                dataset=dataset,
                baseline=baseline,
                k=k,
                metric=metric_name,
                value=value,
                evaluated_users=user_count,
                recommendation_rows=len(recommendations),
                diagnostic_subset=diagnostic_subset,
                max_users=max_users,
                run_scope=run_scope,
                reconstruction_status=reconstruction_status,
                notes="Computed against reconstructed Plan C ground truth and candidates.",
                timestamp=timestamp,
            )
        )

    if not catalog.empty:
        coverage_value = catalog_coverage_at_k(
            recommendations,
            catalog,
            k,
            recommendation_col="recommended_item_id",
            catalog_col="item_id",
        )
        rows.append(
            _metric_row(
                dataset=dataset,
                baseline=baseline,
                k=k,
                metric="catalog_coverage",
                value=float(coverage_value),
                evaluated_users=user_count,
                recommendation_rows=len(recommendations),
                diagnostic_subset=diagnostic_subset,
                max_users=max_users,
                run_scope=run_scope,
                reconstruction_status=reconstruction_status,
                notes="Catalog Coverage@K over reconstructed Plan C catalog.",
                timestamp=timestamp,
            )
        )

    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def _candidate_diagnostic_row(
    dataset: CanonicalDataset,
    *,
    baseline: str,
    max_users: int | None,
    run_scope: str,
    timestamp: str,
) -> dict[str, Any]:
    train_items = set(_train_interactions(dataset.interactions)["item_id"].astype("string"))
    full_interaction_items = set(dataset.interactions["item_id"].astype("string"))
    full_interaction_items.update(dataset.ground_truth["ground_truth_item_id"].astype("string"))
    candidate_items = dataset.candidates["candidate_item_id"].astype("string")
    candidate_rows = len(candidate_items)
    outside_count = int((~candidate_items.isin(full_interaction_items)).sum()) if candidate_rows else 0
    zero_pop_count = int((~candidate_items.isin(train_items)).sum()) if candidate_rows else 0
    return {
        "dataset": dataset.dataset,
        "baseline": baseline,
        "candidate_rows": int(candidate_rows),
        "candidate_items_not_in_interaction_universe_count": outside_count,
        "candidate_items_not_in_interaction_universe_rate": (
            outside_count / candidate_rows if candidate_rows else 0.0
        ),
        "zero_train_pop_candidate_rate": zero_pop_count / candidate_rows if candidate_rows else 0.0,
        "evaluated_users": int(dataset.ground_truth["user_id"].nunique()) if not dataset.ground_truth.empty else 0,
        "diagnostic_subset": bool(dataset.diagnostic_subset),
        "max_users": int(max_users) if max_users is not None else pd.NA,
        "run_scope": run_scope,
        "notes": (
            "Candidate diagnostics compare candidates against reconstructed train+ground-truth item universe; "
            "zero train popularity is expected for some held-out positives and cold candidate items."
        ),
        "timestamp": timestamp,
    }


def _metric_row(
    *,
    dataset: str,
    baseline: str,
    k: int,
    metric: str,
    value: float,
    evaluated_users: int,
    recommendation_rows: int,
    diagnostic_subset: bool,
    max_users: int | None,
    run_scope: str,
    reconstruction_status: str,
    notes: str,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "baseline": baseline,
        "k": int(k),
        "metric": metric,
        "value": float(value),
        "evaluated_users": int(evaluated_users),
        "recommendation_rows": int(recommendation_rows),
        "diagnostic_subset": bool(diagnostic_subset),
        "max_users": int(max_users) if max_users is not None else pd.NA,
        "run_scope": run_scope,
        "reconstruction_status": reconstruction_status,
        "status": "computed",
        "notes": notes,
        "timestamp": timestamp,
    }


def load_real_plan_c_metric_rows(metrics_dir: str | Path, *, k: int = 5) -> pd.DataFrame:
    """Load real Plan C metric rows from non-empty plan_c metric files only."""

    metrics_dir = Path(metrics_dir)
    rows: list[pd.DataFrame] = []
    for path in sorted(metrics_dir.glob("plan_c_*_metrics.csv")):
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if frame.empty:
            continue
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        required = {"dataset", "baseline", "k", "metric", "value", "status"}
        if not required.issubset(frame.columns):
            continue
        filtered = frame.loc[
            (pd.to_numeric(frame["k"], errors="coerce") == int(k))
            & (frame["status"].astype("string").str.lower() == "computed")
            & (pd.to_numeric(frame["value"], errors="coerce").notna())
        ].copy()
        if filtered.empty:
            continue
        filtered["source_file"] = str(path)
        rows.append(filtered)

    if not rows:
        return pd.DataFrame(columns=[*METRIC_COLUMNS, "source_file"])

    summary = pd.concat(rows, ignore_index=True, sort=False)
    summary = _normalize_metric_metadata(summary)
    return summary.loc[:, [column for column in [*METRIC_COLUMNS, "source_file"] if column in summary.columns]]


def _normalize_metric_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "source_file" not in normalized.columns:
        normalized["source_file"] = ""
    if "diagnostic_subset" not in normalized.columns:
        normalized["diagnostic_subset"] = False
    if "max_users" not in normalized.columns:
        normalized["max_users"] = pd.NA
    if "run_scope" not in normalized.columns:
        normalized["run_scope"] = pd.NA
    if "evaluated_users" not in normalized.columns and "num_users" in normalized.columns:
        normalized["evaluated_users"] = normalized["num_users"]
    if "recommendation_rows" not in normalized.columns and "num_recommendation_rows" in normalized.columns:
        normalized["recommendation_rows"] = normalized["num_recommendation_rows"]

    diagnostic_values: list[bool] = []
    max_user_values: list[Any] = []
    run_scope_values: list[str] = []
    for _, row in normalized.iterrows():
        filename_scope = _scope_from_metric_filename(str(row.get("source_file", "")))
        row_max_users = _safe_optional_int(row.get("max_users", pd.NA))
        filename_max_users = filename_scope["max_users"]
        max_users = row_max_users if row_max_users is not None else filename_max_users

        diagnostic = _parse_bool(row.get("diagnostic_subset", False)) or filename_scope["diagnostic_subset"]
        if max_users is not None:
            diagnostic = True

        run_scope = str(row.get("run_scope", "")).strip()
        if not run_scope or run_scope.lower() in {"<na>", "nan", "none"}:
            run_scope = _run_scope(max_users)

        diagnostic_values.append(bool(diagnostic))
        max_user_values.append(int(max_users) if max_users is not None else pd.NA)
        run_scope_values.append(run_scope)

    normalized["diagnostic_subset"] = diagnostic_values
    normalized["max_users"] = max_user_values
    normalized["run_scope"] = run_scope_values
    if "evaluated_users" not in normalized.columns:
        normalized["evaluated_users"] = 0
    if "recommendation_rows" not in normalized.columns:
        normalized["recommendation_rows"] = 0
    return normalized


def build_plan_c_comparison(metrics_summary: pd.DataFrame, *, k: int = 5) -> pd.DataFrame:
    """Build a wide comparison table from real Plan C metric rows."""

    if metrics_summary.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    frame = metrics_summary.copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])
    rows: list[dict[str, Any]] = []
    for (dataset, baseline, run_scope, max_users_value), group in frame.groupby(
        ["dataset", "baseline", "run_scope", "max_users"],
        sort=True,
        dropna=False,
    ):
        metrics = {
            str(row["metric"]): float(row["value"])
            for _, row in group.iterrows()
            if pd.notna(row["value"])
        }
        first = group.iloc[0]
        diagnostic_subset = bool(_parse_bool(first.get("diagnostic_subset", False)))
        rows.append(
            {
                "dataset": dataset,
                "baseline": baseline,
                "hr_at_5": metrics.get("hit_rate", pd.NA) if k == 5 else pd.NA,
                "ndcg_at_5": metrics.get("ndcg", pd.NA) if k == 5 else pd.NA,
                "catalog_coverage_at_5": metrics.get("catalog_coverage", pd.NA) if k == 5 else pd.NA,
                "evaluated_users": _safe_int(first.get("evaluated_users", 0)),
                "recommendation_rows": _safe_int(first.get("recommendation_rows", 0)),
                "diagnostic_subset": diagnostic_subset,
                "max_users": int(max_users_value) if pd.notna(max_users_value) else pd.NA,
                "run_scope": run_scope,
                "status": "available",
                "reconstruction_status": first.get("reconstruction_status", ""),
                "notes": "Real Plan C reconstructed-dataset baseline metrics.",
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def generate_plan_c_plots(
    comparison: pd.DataFrame,
    *,
    plots_dir: str | Path,
    k: int = 5,
) -> list[Path]:
    """Generate simple comparison plots from real comparison rows."""

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if comparison.empty:
        return []

    plot_paths: list[Path] = []
    metric_specs = [
        ("hr_at_5", f"Plan C ML-style reconstructed HR@{k}", f"HR@{k}", "plan_c_baseline_hr_at_5.png"),
        ("ndcg_at_5", f"Plan C ML-style reconstructed NDCG@{k}", f"NDCG@{k}", "plan_c_baseline_ndcg_at_5.png"),
    ]
    for metric_col, title, ylabel, filename in metric_specs:
        path = plots_dir / filename
        if _plot_metric(comparison, metric_col=metric_col, title=title, ylabel=ylabel, output_path=path):
            plot_paths.append(path)

    combined_path = plots_dir / "plan_c_baseline_comparison.png"
    if _plot_combined(comparison, output_path=combined_path, k=k):
        plot_paths.append(combined_path)
    return plot_paths


def _plot_metric(
    comparison: pd.DataFrame,
    *,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> bool:
    frame = comparison.dropna(subset=[metric_col]).copy()
    if frame.empty:
        return False

    datasets = _ordered_present_values(frame["dataset"], DEFAULT_DATASETS)
    baselines = _ordered_present_values(frame["baseline"], DEFAULT_BASELINES)
    if not datasets or not baselines:
        return False

    x_positions = list(range(len(datasets)))
    width = min(0.8 / max(len(baselines), 1), 0.25)
    fig, ax = plt.subplots(figsize=(max(7, len(datasets) * 1.6), 4.5))
    for offset, baseline in enumerate(baselines):
        values = []
        for dataset in datasets:
            matched = frame.loc[
                (frame["dataset"] == dataset) & (frame["baseline"] == baseline),
                metric_col,
            ]
            values.append(float(matched.iloc[0]) if not matched.empty else 0.0)
        positions = [x + (offset - (len(baselines) - 1) / 2) * width for x in x_positions]
        ax.bar(positions, values, width=width, label=baseline)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Reconstructed dataset")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _plot_combined(comparison: pd.DataFrame, *, output_path: Path, k: int) -> bool:
    frame = comparison.dropna(subset=["hr_at_5", "ndcg_at_5"], how="all").copy()
    if frame.empty:
        return False
    labels = [
        f"{row.dataset}\n{row.baseline}"
        for row in frame.itertuples(index=False)
    ]
    x_positions = list(range(len(labels)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4.8))
    ax.bar([x - width / 2 for x in x_positions], frame["hr_at_5"].fillna(0.0), width=width, label=f"HR@{k}")
    ax.bar([x + width / 2 for x in x_positions], frame["ndcg_at_5"].fillna(0.0), width=width, label=f"NDCG@{k}")
    ax.set_title(f"Plan C reconstructed baseline comparison at {k}")
    ax.set_ylabel("Metric value")
    ax.set_xlabel("Dataset and baseline")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _load_dataset(
    dataset_name: str,
    paths: DatasetPaths,
    *,
    max_users: int | None,
    diagnostic_subset: bool,
) -> CanonicalDataset:
    LOGGER.info("Loading canonical files for %s", dataset_name)
    interactions = validate_interactions(_read_csv(paths.interactions))
    ground_truth = validate_ground_truth(_read_csv(paths.ground_truth))
    candidates = validate_candidates(_read_csv(paths.candidates))
    catalog = _read_catalog(paths.catalog)
    _coerce_columns_to_string(interactions, ("user_id", "item_id", "split"))
    _coerce_columns_to_string(ground_truth, ("user_id", "ground_truth_item_id"))
    _coerce_columns_to_string(candidates, ("user_id", "candidate_item_id"))
    _coerce_columns_to_string(catalog, ("item_id",))

    if max_users is not None:
        LOGGER.info("Applying diagnostic --max-users=%s to %s", max_users, dataset_name)
        selected_users = _first_n_users(ground_truth["user_id"], max_users)
        user_set = set(selected_users)
        interactions = interactions.loc[interactions["user_id"].astype("string").isin(user_set)].copy()
        ground_truth = ground_truth.loc[ground_truth["user_id"].astype("string").isin(user_set)].copy()
        candidates = candidates.loc[candidates["user_id"].astype("string").isin(user_set)].copy()

    return CanonicalDataset(
        dataset=dataset_name,
        paths=paths,
        interactions=interactions.reset_index(drop=True),
        ground_truth=ground_truth.reset_index(drop=True),
        candidates=candidates.reset_index(drop=True),
        catalog=catalog.reset_index(drop=True),
        reconstruction_status=RECONSTRUCTION_STATUS.get(dataset_name, "reconstructed_from_public_raw"),
        diagnostic_subset=diagnostic_subset,
        max_users=max_users,
        run_scope=_run_scope(max_users),
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def _read_catalog(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "item_id" not in frame.columns:
        raise ValueError("catalog.csv must contain item_id")
    return frame


def _train_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    if interactions.empty:
        return interactions.copy()
    split = interactions["split"].astype("string").str.strip().str.lower()
    train = interactions.loc[split == "train"].copy()
    return train if not train.empty else interactions.copy()


def _catalog_subset_for_content(dataset: CanonicalDataset) -> pd.DataFrame:
    needed_items = set(dataset.candidates["candidate_item_id"].astype("string"))
    needed_items.update(dataset.interactions["item_id"].astype("string"))
    catalog = dataset.catalog.copy()
    catalog["item_id"] = catalog["item_id"].astype("string").str.strip()
    subset = catalog.loc[catalog["item_id"].isin(needed_items)].copy()
    return subset if not subset.empty else catalog


def _candidate_positive_series(frame: pd.DataFrame) -> pd.Series:
    if "is_ground_truth" in frame.columns:
        return _boolean_series(frame["is_ground_truth"])
    if "is_positive" in frame.columns:
        return _boolean_series(frame["is_positive"])
    return pd.Series(False, index=frame.index)


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _coerce_columns_to_string(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()


def _deterministic_score(seed: int, user_id: Any, item_id: Any) -> float:
    payload = f"{seed}|{user_id}|{item_id}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def _output_stem(dataset: str, baseline: str, *, max_users: int | None) -> str:
    base = f"plan_c_{dataset}_{baseline}"
    if max_users is None:
        return base
    return f"{base}_diagnostic_max_users_{int(max_users)}"


def _run_scope(max_users: int | None) -> str:
    if max_users is None:
        return "full"
    return f"diagnostic_max_users_{int(max_users)}"


def _scope_from_metric_filename(path_text: str) -> dict[str, Any]:
    name = Path(path_text).name
    marker = "_diagnostic_max_users_"
    if marker not in name:
        return {"diagnostic_subset": False, "max_users": None}
    after = name.split(marker, 1)[1]
    number_text = after.split("_metrics.csv", 1)[0]
    try:
        max_users = int(number_text)
    except ValueError:
        max_users = None
    return {"diagnostic_subset": True, "max_users": max_users}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_optional_int(value: Any) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def _safe_int(value: Any) -> int:
    parsed = _safe_optional_int(value)
    return parsed if parsed is not None else 0


def _first_n_users(users: pd.Series, max_users: int) -> list[str]:
    unique_users = sorted({str(user) for user in users.astype("string")}, key=_id_sort_key)
    return unique_users[:max_users]


def _add_id_sort_columns(frame: pd.DataFrame, column: str, prefix: str) -> None:
    groups: list[int] = []
    numbers: list[int] = []
    texts: list[str] = []
    for value in frame[column].astype("string"):
        group, number, text = _id_sort_key(value)
        groups.append(group)
        numbers.append(number)
        texts.append(text)
    frame[f"{prefix}_sort_group"] = groups
    frame[f"{prefix}_sort_number"] = numbers
    frame[f"{prefix}_sort_text"] = texts


def _id_sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value).strip()
    try:
        number = int(text)
    except ValueError:
        return (1, 0, text)
    return (0, number, "")


def _dataset_paths(data_root: Path, dataset: str) -> DatasetPaths:
    base_dir = data_root / dataset
    return DatasetPaths(
        base_dir=base_dir,
        interactions=base_dir / "interactions.csv",
        ground_truth=base_dir / "ground_truth.csv",
        candidates=base_dir / "candidates.csv",
        catalog=base_dir / "catalog.csv",
    )


def _missing_required_inputs(paths: DatasetPaths) -> list[str]:
    required = {
        "interactions": paths.interactions,
        "ground_truth": paths.ground_truth,
        "candidates": paths.candidates,
        "catalog": paths.catalog,
    }
    return [f"{name}={path}" for name, path in required.items() if not path.is_file()]


def _append_lastfm_blocker_if_needed(
    *,
    selected_datasets: list[str],
    selected_baselines: list[str],
    data_root: Path,
    status_rows: list[dict[str, Any]],
    blocker_rows: list[dict[str, Any]],
    candidate_diagnostic_rows: list[dict[str, Any]],
    diagnostic_subset: bool,
    max_users: int | None,
    run_scope: str,
    timestamp: str,
) -> None:
    if any(row["dataset"] == "lastfm" for row in blocker_rows):
        return
    paths = _dataset_paths(data_root, "lastfm")
    if not paths.base_dir.exists():
        return
    lastfm_selected = "lastfm" in selected_datasets
    baselines = selected_baselines if lastfm_selected else ["all"]
    lastfm_diagnostic_subset = diagnostic_subset if lastfm_selected else False
    reason = (
        "LastFM is extension_only in C5: no ground_truth.csv/candidates.csv protocol "
        "is available for utility metrics."
    )
    for baseline in baselines:
        blocker_rows.append(
            _blocker_row(
                dataset="lastfm",
                baseline=baseline,
                reason=reason,
                missing_inputs=[],
                diagnostic_subset=lastfm_diagnostic_subset,
                max_users=max_users if lastfm_selected else None,
                run_scope=run_scope if lastfm_selected else "full",
                timestamp=timestamp,
            )
        )
        status_rows.append(
            _status_row(
                dataset="lastfm",
                baseline=baseline,
                status="blocked_extension_only",
                required_inputs_found=False,
                interactions_rows=_safe_row_count(paths.interactions),
                ground_truth_rows=_safe_row_count(paths.ground_truth),
                candidate_rows=_safe_row_count(paths.candidates),
                catalog_rows=_safe_row_count(paths.catalog),
                recommendation_rows=0,
                metrics_written=False,
                diagnostic_subset=lastfm_diagnostic_subset,
                max_users=max_users if lastfm_selected else None,
                run_scope=run_scope if lastfm_selected else "full",
                notes=reason,
                timestamp=timestamp,
            )
        )


def _safe_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8")) - 1
    except OSError:
        return 0


def _blocker_row(
    *,
    dataset: str,
    baseline: str,
    reason: str,
    missing_inputs: list[str],
    diagnostic_subset: bool,
    max_users: int | None,
    run_scope: str,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "baseline": baseline,
        "status": "blocked",
        "reason": reason,
        "missing_inputs": "; ".join(missing_inputs),
        "diagnostic_subset": bool(diagnostic_subset),
        "max_users": int(max_users) if max_users is not None else pd.NA,
        "run_scope": run_scope,
        "reconstruction_status": RECONSTRUCTION_STATUS.get(dataset, "unknown"),
        "notes": "No fake metric file was written.",
        "timestamp": timestamp,
    }


def _status_row(
    *,
    dataset: str,
    baseline: str,
    status: str,
    required_inputs_found: bool,
    interactions_rows: int,
    ground_truth_rows: int,
    candidate_rows: int,
    catalog_rows: int,
    recommendation_rows: int,
    metrics_written: bool,
    diagnostic_subset: bool,
    max_users: int | None,
    run_scope: str,
    notes: str,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "baseline": baseline,
        "status": status,
        "required_inputs_found": bool(required_inputs_found),
        "interactions_rows": int(interactions_rows),
        "ground_truth_rows": int(ground_truth_rows),
        "candidate_rows": int(candidate_rows),
        "catalog_rows": int(catalog_rows),
        "recommendation_rows": int(recommendation_rows),
        "metrics_written": bool(metrics_written),
        "diagnostic_subset": bool(diagnostic_subset),
        "max_users": int(max_users) if max_users is not None else pd.NA,
        "run_scope": run_scope,
        "reconstruction_status": RECONSTRUCTION_STATUS.get(dataset, "unknown"),
        "notes": notes,
        "timestamp": timestamp,
    }


def _normalize_requested_values(values: list[str] | tuple[str, ...], label: str) -> list[str]:
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if not normalized:
        raise ValueError(f"At least one {label} value is required")
    if label == "baselines":
        invalid = sorted(set(normalized).difference(DEFAULT_BASELINES))
        if invalid:
            raise ValueError(f"Unknown baseline(s): {', '.join(invalid)}")
    return normalized


def _ordered_present_values(values: pd.Series, preferred_order: tuple[str, ...]) -> list[str]:
    present = [str(value) for value in values.dropna().unique().tolist()]
    ordered = [value for value in preferred_order if value in present]
    ordered.extend(sorted(value for value in present if value not in ordered))
    return ordered


def _resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    run_plan_c_baselines(
        datasets=args.datasets,
        baselines=args.baselines,
        k=args.k,
        seed=args.seed,
        max_users=args.max_users,
        data_root=args.data_root,
        results_root=args.results_root,
        project_root=PROJECT_ROOT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

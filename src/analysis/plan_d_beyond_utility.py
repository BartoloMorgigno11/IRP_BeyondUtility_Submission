"""Plan D beyond-utility metrics over existing Plan C recommendations."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("plan_d_beyond_utility")

PLAN_NAME = "Plan D"
DEFAULT_CUTOFF = 5
CHUNKSIZE = 500_000

METRIC_COLUMNS = [
    "plan",
    "dataset",
    "method",
    "scope",
    "cutoff",
    "metric",
    "value",
    "num_users_evaluated",
    "num_recommendations_evaluated",
    "catalog_size",
    "has_candidates",
    "has_item_features",
    "input_recommendation_file",
    "notes",
    "timestamp",
]

STATUS_COLUMNS = [
    "plan",
    "dataset",
    "method",
    "scope",
    "cutoff",
    "status",
    "required_inputs_found",
    "missing_inputs",
    "metrics_written",
    "num_users_evaluated",
    "num_recommendations_evaluated",
    "catalog_size",
    "interactions_rows",
    "candidate_rows",
    "has_candidates",
    "has_item_features",
    "input_recommendation_file",
    "notes",
    "timestamp",
]

COMPARISON_METADATA_COLUMNS = [
    "plan",
    "dataset",
    "method",
    "scope",
    "cutoff",
    "num_users_evaluated",
    "num_recommendations_evaluated",
    "catalog_size",
    "has_candidates",
    "has_item_features",
    "input_recommendation_file",
]

COMPARISON_METRIC_ORDER = [
    "catalog_coverage_at_5",
    "item_coverage_at_5",
    "average_popularity_at_5",
    "novelty_at_5",
    "long_tail_rate_at_5",
    "popularity_bias_at_5",
    "popularity_bias_vs_catalog_at_5",
    "intra_list_diversity_at_5",
    "candidate_validity_rate",
    "catalog_validity_rate",
    "duplicate_rate_at_5",
]


@dataclass(frozen=True)
class PlanDRunSpec:
    """A Plan D evaluation target backed by one Plan C recommendation file."""

    dataset: str
    method: str
    scope: str
    recommendation_filename: str


@dataclass(frozen=True)
class DatasetPaths:
    """Canonical reconstructed Plan C file paths used by Plan D."""

    base_dir: Path
    interactions: Path
    candidates: Path
    catalog: Path


@dataclass(frozen=True)
class DatasetResources:
    """Loaded dataset resources shared across Plan D methods."""

    dataset: str
    interactions: pd.DataFrame
    candidates: pd.DataFrame | None
    catalog: pd.DataFrame
    item_features: dict[str, set[str]]


DEFAULT_RUN_SPECS: tuple[PlanDRunSpec, ...] = (
    PlanDRunSpec(
        "ml1m",
        "random",
        "full",
        "plan_c_ml1m_random_recommendations.csv",
    ),
    PlanDRunSpec(
        "ml1m",
        "mostpop",
        "full",
        "plan_c_ml1m_mostpop_recommendations.csv",
    ),
    PlanDRunSpec(
        "ml1m",
        "content_bm25",
        "full",
        "plan_c_ml1m_content_bm25_recommendations.csv",
    ),
    PlanDRunSpec(
        "amazon_beauty",
        "random",
        "full",
        "plan_c_amazon_beauty_random_recommendations.csv",
    ),
    PlanDRunSpec(
        "amazon_beauty",
        "mostpop",
        "full",
        "plan_c_amazon_beauty_mostpop_recommendations.csv",
    ),
    PlanDRunSpec(
        "amazon_sports",
        "random",
        "full",
        "plan_c_amazon_sports_random_recommendations.csv",
    ),
    PlanDRunSpec(
        "amazon_sports",
        "mostpop",
        "full",
        "plan_c_amazon_sports_mostpop_recommendations.csv",
    ),
    PlanDRunSpec(
        "amazon_beauty",
        "content_bm25",
        "diagnostic_subset",
        "plan_c_amazon_beauty_content_bm25_diagnostic_max_users_1000_recommendations.csv",
    ),
    PlanDRunSpec(
        "amazon_sports",
        "content_bm25",
        "diagnostic_subset",
        "plan_c_amazon_sports_content_bm25_diagnostic_max_users_1000_recommendations.csv",
    ),
)


def run_plan_d_beyond_utility(
    *,
    cutoff: int = DEFAULT_CUTOFF,
    data_root: str | Path = Path("data/reconstructed/plan_c"),
    results_root: str | Path = Path("results"),
    project_root: str | Path | None = None,
    run_specs: tuple[PlanDRunSpec, ...] | list[PlanDRunSpec] = DEFAULT_RUN_SPECS,
) -> dict[str, Any]:
    """Compute Plan D beyond-utility metrics from existing Plan C outputs.

    The runner never writes Plan C files. It writes ``plan_d_`` CSVs only when
    real metric rows exist, and always writes a dataset status table so missing
    inputs are visible.
    """

    if cutoff != 5:
        raise ValueError("Plan D D1 is defined for cutoff=5")

    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    data_root = _resolve_path(data_root, root)
    results_root = _resolve_path(results_root, root)
    metrics_dir = results_root / "metrics"
    tables_dir = results_root / "tables"
    plots_dir = results_root / "plots"
    raw_outputs_dir = results_root / "raw_outputs"
    for directory in (metrics_dir, tables_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    metrics_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    resource_cache: dict[str, DatasetResources] = {}

    for spec in run_specs:
        recommendation_path = raw_outputs_dir / spec.recommendation_filename
        paths = _dataset_paths(data_root, spec.dataset)
        missing_required = _missing_required_inputs(paths, recommendation_path)
        if missing_required:
            status_rows.append(
                _status_row(
                    spec=spec,
                    cutoff=cutoff,
                    status="MISSING",
                    required_inputs_found=False,
                    missing_inputs=missing_required,
                    metrics_written=False,
                    num_users_evaluated=0,
                    num_recommendations_evaluated=0,
                    catalog_size=_safe_unique_item_count(paths.catalog, "item_id"),
                    interactions_rows=_safe_row_count(paths.interactions),
                    candidate_rows=_safe_row_count(paths.candidates),
                    has_candidates=paths.candidates.is_file(),
                    has_item_features=False,
                    recommendation_path=recommendation_path,
                    notes="Required Plan C input file(s) are missing; no Plan D metric rows were written.",
                    timestamp=timestamp,
                    root=root,
                )
            )
            continue

        try:
            resources = resource_cache.get(spec.dataset)
            if resources is None:
                resources = load_dataset_resources(spec.dataset, paths)
                resource_cache[spec.dataset] = resources

            recommendations = load_top_k_recommendations(recommendation_path, cutoff)
            if recommendations.empty:
                status_rows.append(
                    _status_row(
                        spec=spec,
                        cutoff=cutoff,
                        status="MISSING",
                        required_inputs_found=True,
                        missing_inputs=[],
                        metrics_written=False,
                        num_users_evaluated=0,
                        num_recommendations_evaluated=0,
                        catalog_size=catalog_size(resources.catalog),
                        interactions_rows=len(resources.interactions),
                        candidate_rows=len(resources.candidates) if resources.candidates is not None else 0,
                        has_candidates=resources.candidates is not None,
                        has_item_features=has_usable_item_features(resources.item_features),
                        recommendation_path=recommendation_path,
                        notes="Recommendation file exists but contains no evaluable top-5 rows.",
                        timestamp=timestamp,
                        root=root,
                    )
                )
                continue

            metrics = compute_beyond_utility_metrics(
                recommendations,
                catalog=resources.catalog,
                interactions=resources.interactions,
                candidates=resources.candidates,
                item_features=resources.item_features,
                dataset=spec.dataset,
                method=spec.method,
                scope=spec.scope,
                cutoff=cutoff,
                input_recommendation_file=_display_path(recommendation_path, root),
                timestamp=timestamp,
            )
            metrics_rows.extend(metrics.to_dict("records"))
            status, notes = _availability_status(metrics, resources)
            status_rows.append(
                _status_row(
                    spec=spec,
                    cutoff=cutoff,
                    status=status,
                    required_inputs_found=True,
                    missing_inputs=[],
                    metrics_written=not metrics.empty,
                    num_users_evaluated=int(metrics["num_users_evaluated"].iloc[0]) if not metrics.empty else 0,
                    num_recommendations_evaluated=(
                        int(metrics["num_recommendations_evaluated"].iloc[0]) if not metrics.empty else 0
                    ),
                    catalog_size=catalog_size(resources.catalog),
                    interactions_rows=len(resources.interactions),
                    candidate_rows=len(resources.candidates) if resources.candidates is not None else 0,
                    has_candidates=resources.candidates is not None,
                    has_item_features=has_usable_item_features(resources.item_features),
                    recommendation_path=recommendation_path,
                    notes=notes,
                    timestamp=timestamp,
                    root=root,
                )
            )
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            status_rows.append(
                _status_row(
                    spec=spec,
                    cutoff=cutoff,
                    status="MISSING",
                    required_inputs_found=False,
                    missing_inputs=[],
                    metrics_written=False,
                    num_users_evaluated=0,
                    num_recommendations_evaluated=0,
                    catalog_size=_safe_unique_item_count(paths.catalog, "item_id"),
                    interactions_rows=_safe_row_count(paths.interactions),
                    candidate_rows=_safe_row_count(paths.candidates),
                    has_candidates=paths.candidates.is_file(),
                    has_item_features=False,
                    recommendation_path=recommendation_path,
                    notes=f"Failed to load or evaluate inputs: {exc}",
                    timestamp=timestamp,
                    root=root,
                )
            )

    status = pd.DataFrame(status_rows, columns=STATUS_COLUMNS)
    status_path = tables_dir / "plan_d_beyond_utility_dataset_status.csv"
    status.to_csv(status_path, index=False)

    metrics = pd.DataFrame(metrics_rows, columns=METRIC_COLUMNS)
    metrics_path = metrics_dir / "plan_d_beyond_utility_metrics.csv"
    comparison_path = tables_dir / "plan_d_beyond_utility_comparison.csv"
    plot_paths: list[Path] = []
    comparison = pd.DataFrame()
    if not metrics.empty:
        metrics.to_csv(metrics_path, index=False)
        comparison = build_beyond_utility_comparison(metrics)
        comparison.to_csv(comparison_path, index=False)
        plot_paths = generate_beyond_utility_plots(comparison, plots_dir=plots_dir)
    else:
        LOGGER.warning("No Plan D metric rows computed; metrics and comparison CSVs were not written.")

    return {
        "metrics": metrics,
        "status": status,
        "comparison": comparison,
        "plot_paths": plot_paths,
        "metrics_path": metrics_path if not metrics.empty else None,
        "comparison_path": comparison_path if not metrics.empty else None,
        "status_path": status_path,
    }


def compute_beyond_utility_metrics(
    recommendations: pd.DataFrame,
    *,
    catalog: pd.DataFrame,
    interactions: pd.DataFrame,
    candidates: pd.DataFrame | None,
    item_features: dict[str, set[str]] | None = None,
    dataset: str = "",
    method: str = "",
    scope: str = "full",
    cutoff: int = DEFAULT_CUTOFF,
    input_recommendation_file: str = "",
    timestamp: str = "",
) -> pd.DataFrame:
    """Return CSV-ready Plan D beyond-utility metric rows at ``cutoff``."""

    if cutoff != 5:
        raise ValueError("Plan D D1 metrics are defined for cutoff=5")

    top_k = top_k_recommendations(recommendations, cutoff)
    catalog_items = catalog_item_set(catalog)
    catalog_count = len(catalog_items)
    popularity = training_popularity(interactions)
    train_total = int(popularity.sum())
    features = item_features if item_features is not None else build_item_feature_sets(catalog, dataset=dataset)
    has_features = has_usable_item_features(features)
    has_candidates = candidates is not None and not candidates.empty
    num_users = int(top_k["user_id"].nunique()) if "user_id" in top_k.columns else 0
    num_recommendations = int(len(top_k))

    values = {
        "catalog_coverage_at_5": catalog_coverage_at_k(top_k, catalog, cutoff),
        "item_coverage_at_5": item_coverage_at_k(top_k, cutoff),
        "average_popularity_at_5": average_popularity_at_k(top_k, popularity, cutoff),
        "novelty_at_5": novelty_at_k(
            top_k,
            popularity,
            train_interaction_count=train_total,
            catalog_size=catalog_count,
            k=cutoff,
        ),
        "long_tail_rate_at_5": long_tail_rate_at_k(top_k, popularity, catalog_items, cutoff),
        "intra_list_diversity_at_5": intra_list_diversity_at_k(top_k, features, cutoff),
        "duplicate_rate_at_5": duplicate_rate_at_k(top_k, cutoff),
    }

    if has_candidates:
        values["popularity_bias_at_5"] = popularity_bias_at_k(
            top_k,
            popularity,
            candidates=candidates,
            catalog=catalog,
            k=cutoff,
        )
        values["candidate_validity_rate"] = candidate_validity_rate_at_k(
            top_k,
            candidates=candidates,
            catalog=catalog,
            k=cutoff,
        )
    else:
        values["popularity_bias_vs_catalog_at_5"] = popularity_bias_at_k(
            top_k,
            popularity,
            candidates=None,
            catalog=catalog,
            k=cutoff,
        )
        values["catalog_validity_rate"] = candidate_validity_rate_at_k(
            top_k,
            candidates=None,
            catalog=catalog,
            k=cutoff,
        )

    metric_notes = {
        "catalog_coverage_at_5": "Unique top-5 catalog items divided by catalog size.",
        "item_coverage_at_5": "Unique non-empty item ids recommended in top-5.",
        "average_popularity_at_5": "Mean training interaction count of top-5 recommendations.",
        "novelty_at_5": (
            "Mean -log2((training_popularity_count + 1) / "
            "(total_train_interactions + catalog_size)) over top-5 recommendations."
        ),
        "long_tail_rate_at_5": (
            "Fraction of top-5 recommendations with training popularity below the "
            "80th percentile among interacted catalog items."
        ),
        "popularity_bias_at_5": "Top-5 mean training popularity minus candidate-list mean training popularity.",
        "popularity_bias_vs_catalog_at_5": "Top-5 mean training popularity minus catalog mean training popularity.",
        "intra_list_diversity_at_5": (
            "Mean 1 - pairwise Jaccard similarity over item feature sets in each top-5 list."
        ),
        "candidate_validity_rate": "Fraction of top-5 recommended items present in each user's candidate set.",
        "catalog_validity_rate": "Fraction of top-5 recommended items present in the evaluation catalog.",
        "duplicate_rate_at_5": "Mean within-user duplicate fraction in top-5 lists.",
    }

    rows = [
        {
            "plan": PLAN_NAME,
            "dataset": dataset,
            "method": method,
            "scope": scope,
            "cutoff": cutoff,
            "metric": metric,
            "value": value,
            "num_users_evaluated": num_users,
            "num_recommendations_evaluated": num_recommendations,
            "catalog_size": catalog_count,
            "has_candidates": bool(has_candidates),
            "has_item_features": bool(has_features),
            "input_recommendation_file": input_recommendation_file,
            "notes": metric_notes[metric],
            "timestamp": timestamp,
        }
        for metric, value in values.items()
    ]
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def top_k_recommendations(recommendations: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return deterministic per-user top-k recommendation rows."""

    _validate_k(k)
    frame = _normalize_dataframe(recommendations, frame_name="recommendations")
    if frame.empty:
        return pd.DataFrame(columns=["user_id", "recommended_item_id", "rank"])
    _require_columns(frame, frame_name="recommendations", columns=("user_id",))
    item_col = _select_recommendation_column(frame)

    usable = frame.copy()
    usable["recommended_item_id"] = _clean_string_series(usable[item_col])
    usable["user_id"] = _clean_string_series(usable["user_id"])
    usable["__input_order"] = range(len(usable))
    usable = usable.loc[usable["user_id"].notna()].copy()
    if usable.empty:
        return usable[["user_id", "recommended_item_id"]]

    if "rank" in usable.columns:
        usable["__rank_value"] = pd.to_numeric(usable["rank"], errors="coerce")
        usable["__rank_missing"] = usable["__rank_value"].isna()
        sort_columns = ["__rank_missing", "__rank_value", "__input_order"]
    else:
        sort_columns = ["__input_order"]

    usable = usable.sort_values(["user_id", *sort_columns], kind="mergesort")
    limited = usable.groupby("user_id", sort=False).head(k).copy()
    output_columns = [
        column
        for column in ("user_id", "recommended_item_id", "rank", "in_candidate_set")
        if column in limited.columns
    ]
    return limited[output_columns].reset_index(drop=True)


def catalog_coverage_at_k(recommendations: pd.DataFrame, catalog: pd.DataFrame, k: int) -> float:
    """Return unique valid top-k catalog recommendations divided by catalog size."""

    catalog_items = catalog_item_set(catalog)
    if not catalog_items:
        return 0.0
    recommended_items = set(_recommended_items(top_k_recommendations(recommendations, k)))
    return len(recommended_items.intersection(catalog_items)) / len(catalog_items)


def item_coverage_at_k(recommendations: pd.DataFrame, k: int) -> int:
    """Return the number of unique non-empty item ids recommended in top-k."""

    top_k = top_k_recommendations(recommendations, k)
    return len(set(_recommended_items(top_k)))


def average_popularity_at_k(
    recommendations: pd.DataFrame,
    popularity: pd.Series,
    k: int,
) -> float:
    """Return mean training popularity count over top-k recommendation rows."""

    counts = _top_k_popularity_counts(recommendations, popularity, k)
    return float(counts.mean()) if len(counts) else 0.0


def novelty_at_k(
    recommendations: pd.DataFrame,
    popularity: pd.Series,
    *,
    train_interaction_count: int,
    catalog_size: int,
    k: int,
) -> float:
    """Return mean smoothed self-information of top-k recommendations.

    The smoothing formula is
    ``-log2((training_popularity_count + 1) / (total_train_interactions + catalog_size))``.
    This avoids infinite values for zero-popularity recommendations while using
    only training interactions in the numerator and denominator.
    """

    counts = _top_k_popularity_counts(recommendations, popularity, k).astype(float)
    denominator = int(train_interaction_count) + int(catalog_size)
    if denominator <= 0 or len(counts) == 0:
        return 0.0
    probabilities = (counts + 1.0) / float(denominator)
    return float((-np.log2(probabilities)).mean())


def long_tail_rate_at_k(
    recommendations: pd.DataFrame,
    popularity: pd.Series,
    catalog_items: set[str],
    k: int,
) -> float:
    """Return fraction of top-k recommendations below the 80th popularity percentile.

    Long-tail membership is defined over interacted catalog items only: an item
    is long-tail when its training popularity is strictly below the 80th
    percentile of positive training popularity counts among catalog items.
    """

    if popularity.empty or not catalog_items:
        return float("nan")
    interacted_catalog_counts = popularity.loc[
        popularity.index.astype("string").isin(catalog_items)
    ]
    interacted_catalog_counts = interacted_catalog_counts.loc[interacted_catalog_counts > 0]
    if interacted_catalog_counts.empty:
        return float("nan")

    threshold = float(np.percentile(interacted_catalog_counts.astype(float).to_numpy(), 80))
    long_tail_items = set(interacted_catalog_counts.loc[interacted_catalog_counts < threshold].index.astype(str))
    items = _recommended_items(top_k_recommendations(recommendations, k))
    if not items:
        return 0.0
    return sum(1 for item in items if item in long_tail_items) / len(items)


def popularity_bias_at_k(
    recommendations: pd.DataFrame,
    popularity: pd.Series,
    *,
    candidates: pd.DataFrame | None,
    catalog: pd.DataFrame,
    k: int,
) -> float:
    """Return recommendation popularity bias against candidates or catalog."""

    rec_counts = _top_k_popularity_counts(recommendations, popularity, k)
    if len(rec_counts) == 0:
        return 0.0
    recommendation_mean = float(rec_counts.mean())
    if candidates is not None and not candidates.empty:
        basis = _candidate_rows_for_evaluated_users(recommendations, candidates)
        if basis.empty:
            return float("nan")
        basis_counts = _popularity_for_items(basis["candidate_item_id"], popularity)
        basis_mean = float(basis_counts.mean()) if len(basis_counts) else 0.0
    else:
        catalog_items = _clean_string_series(_normalize_dataframe(catalog, frame_name="catalog")["item_id"])
        basis_counts = _popularity_for_items(catalog_items.dropna(), popularity)
        basis_mean = float(basis_counts.mean()) if len(basis_counts) else 0.0
    return recommendation_mean - basis_mean


def candidate_validity_rate_at_k(
    recommendations: pd.DataFrame,
    *,
    candidates: pd.DataFrame | None,
    catalog: pd.DataFrame,
    k: int,
) -> float:
    """Return top-k validity against user candidates or, as fallback, catalog."""

    top_k = top_k_recommendations(recommendations, k)
    if top_k.empty:
        return 0.0
    if candidates is not None and not candidates.empty:
        candidate_frame = _normalize_dataframe(candidates, frame_name="candidates")
        _require_columns(candidate_frame, frame_name="candidates", columns=("user_id", "candidate_item_id"))
        pairs = candidate_frame[["user_id", "candidate_item_id"]].copy()
        pairs["user_id"] = _clean_string_series(pairs["user_id"])
        pairs["candidate_item_id"] = _clean_string_series(pairs["candidate_item_id"])
        pairs = pairs.dropna().drop_duplicates()
        probe = top_k[["user_id", "recommended_item_id"]].copy()
        probe["__row_id"] = range(len(probe))
        merged = probe.merge(
            pairs,
            left_on=["user_id", "recommended_item_id"],
            right_on=["user_id", "candidate_item_id"],
            how="left",
            indicator=True,
        )
        valid = merged.drop_duplicates("__row_id")["_merge"].eq("both")
        return float(valid.mean()) if len(valid) else 0.0

    valid_catalog_items = catalog_item_set(catalog)
    if not valid_catalog_items:
        return 0.0
    items = _recommended_items(top_k)
    return sum(1 for item in items if item in valid_catalog_items) / len(items) if items else 0.0


def duplicate_rate_at_k(recommendations: pd.DataFrame, k: int) -> float:
    """Return mean within-user duplicate rate for top-k recommendation lists."""

    top_k = top_k_recommendations(recommendations, k)
    if top_k.empty:
        return 0.0
    rates: list[float] = []
    for _, group in top_k.groupby("user_id", sort=False):
        items = _recommended_items(group)
        if not items:
            rates.append(0.0)
            continue
        rates.append(1.0 - (len(set(items)) / len(items)))
    return float(np.mean(rates)) if rates else 0.0


def intra_list_diversity_at_k(
    recommendations: pd.DataFrame,
    item_features: dict[str, set[str]],
    k: int,
) -> float:
    """Return mean intra-list diversity using pairwise Jaccard dissimilarity."""

    if not has_usable_item_features(item_features):
        return float("nan")
    top_k = top_k_recommendations(recommendations, k)
    if top_k.empty:
        return 0.0

    user_scores: list[float] = []
    for _, group in top_k.groupby("user_id", sort=False):
        items = _recommended_items(group)
        pair_scores: list[float] = []
        for left_idx in range(len(items)):
            for right_idx in range(left_idx + 1, len(items)):
                left_features = item_features.get(items[left_idx], set())
                right_features = item_features.get(items[right_idx], set())
                similarity = _jaccard_similarity(left_features, right_features)
                if similarity is not None:
                    pair_scores.append(1.0 - similarity)
        if pair_scores:
            user_scores.append(float(np.mean(pair_scores)))

    return float(np.mean(user_scores)) if user_scores else float("nan")


def training_popularity(interactions: pd.DataFrame) -> pd.Series:
    """Return item popularity counts from training interactions only when split exists."""

    frame = _normalize_dataframe(interactions, frame_name="interactions")
    _require_columns(frame, frame_name="interactions", columns=("item_id",))
    if "split" in frame.columns:
        split = frame["split"].astype("string").str.strip().str.lower()
        frame = frame.loc[split == "train"].copy()
    frame["item_id"] = _clean_string_series(frame["item_id"])
    frame = frame.loc[frame["item_id"].notna()]
    if frame.empty:
        return pd.Series(dtype="int64")
    return frame.groupby("item_id", sort=False).size().astype("int64")


def build_item_feature_sets(catalog: pd.DataFrame, *, dataset: str = "") -> dict[str, set[str]]:
    """Build deterministic item feature sets from catalog metadata."""

    frame = _normalize_dataframe(catalog, frame_name="catalog")
    _require_columns(frame, frame_name="catalog", columns=("item_id",))
    frame["item_id"] = _clean_string_series(frame["item_id"])
    column = _select_feature_column(frame, dataset=dataset)
    if column is None:
        return {}

    features: dict[str, set[str]] = {}
    for item_id, raw_value in zip(frame["item_id"], frame[column]):
        if pd.isna(item_id):
            continue
        parsed = _parse_feature_value(raw_value, textual=column in {"normalized_text", "description", "title"})
        if parsed:
            features[str(item_id)] = parsed
    return features


def has_usable_item_features(item_features: dict[str, set[str]]) -> bool:
    """Return whether at least one catalog item has a non-empty feature set."""

    return any(bool(features) for features in item_features.values())


def catalog_item_set(catalog: pd.DataFrame) -> set[str]:
    """Return normalized catalog item ids."""

    frame = _normalize_dataframe(catalog, frame_name="catalog")
    _require_columns(frame, frame_name="catalog", columns=("item_id",))
    return set(_clean_string_series(frame["item_id"]).dropna().astype(str).tolist())


def catalog_size(catalog: pd.DataFrame) -> int:
    """Return unique catalog item count."""

    return len(catalog_item_set(catalog))


def load_dataset_resources(dataset: str, paths: DatasetPaths) -> DatasetResources:
    """Load reconstructed Plan C resources needed by Plan D."""

    interactions = _read_csv_selected(paths.interactions, ["user_id", "item_id", "split"])
    candidates = (
        _read_csv_selected(paths.candidates, ["user_id", "candidate_item_id"])
        if paths.candidates.is_file()
        else None
    )
    catalog = load_catalog_for_features(paths.catalog, dataset=dataset)
    return DatasetResources(
        dataset=dataset,
        interactions=interactions,
        candidates=candidates,
        catalog=catalog,
        item_features=build_item_feature_sets(catalog, dataset=dataset),
    )


def load_catalog_for_features(path: Path, *, dataset: str) -> pd.DataFrame:
    """Load catalog item ids and the smallest useful feature columns."""

    columns = [str(column).strip().lower() for column in pd.read_csv(path, nrows=0).columns]
    feature_priority = _feature_priority(dataset)
    feature_columns = [column for column in feature_priority if column in columns]
    selected = ["item_id", *feature_columns]
    frame = _read_csv_selected(path, selected)
    feature_sets = build_item_feature_sets(frame, dataset=dataset)
    if has_usable_item_features(feature_sets) or "normalized_text" not in columns:
        return frame

    fallback_selected = ["item_id", "normalized_text"]
    return _read_csv_selected(path, fallback_selected)


def load_top_k_recommendations(path: Path, k: int) -> pd.DataFrame:
    """Load only columns needed for Plan D and retain top-k rows."""

    _validate_k(k)
    header = [str(column).strip().lower() for column in pd.read_csv(path, nrows=0).columns]
    selected = [
        column
        for column in ("user_id", "recommended_item_id", "item_id", "rank", "in_candidate_set")
        if column in header
    ]
    if "user_id" not in selected:
        raise ValueError(f"{path} must contain user_id")
    if "recommended_item_id" not in selected and "item_id" not in selected:
        raise ValueError(f"{path} must contain recommended_item_id or item_id")

    chunks: list[pd.DataFrame] = []
    has_rank = "rank" in selected
    for chunk in pd.read_csv(path, dtype="string", usecols=selected, chunksize=CHUNKSIZE):
        chunk = _normalize_dataframe(chunk, frame_name="recommendations")
        if has_rank:
            rank = pd.to_numeric(chunk["rank"], errors="coerce")
            chunk = chunk.loc[rank.le(k)].copy()
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=selected)
    return top_k_recommendations(pd.concat(chunks, ignore_index=True), k)


def build_beyond_utility_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a wide Plan D comparison table from long metric rows."""

    if metrics.empty:
        return pd.DataFrame(columns=COMPARISON_METADATA_COLUMNS + COMPARISON_METRIC_ORDER)
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "method", "scope"]
    for _, group in metrics.groupby(group_columns, sort=True, dropna=False):
        first = group.iloc[0]
        row = {column: first[column] for column in COMPARISON_METADATA_COLUMNS}
        for metric in COMPARISON_METRIC_ORDER:
            matched = group.loc[group["metric"] == metric, "value"]
            row[metric] = matched.iloc[0] if not matched.empty else pd.NA
        rows.append(row)
    columns = COMPARISON_METADATA_COLUMNS + COMPARISON_METRIC_ORDER
    return pd.DataFrame(rows, columns=columns)


def generate_beyond_utility_plots(
    comparison: pd.DataFrame,
    *,
    plots_dir: str | Path,
) -> list[Path]:
    """Generate simple Plan D bar plots from computed comparison rows."""

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if comparison.empty:
        return []

    plot_paths: list[Path] = []
    for metric in COMPARISON_METRIC_ORDER:
        if metric not in comparison.columns:
            continue
        frame = comparison.dropna(subset=[metric]).copy()
        if frame.empty:
            continue
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frame = frame.dropna(subset=[metric])
        if frame.empty:
            continue
        output_path = plots_dir / f"plan_d_beyond_utility_{metric}.png"
        if _plot_metric(frame, metric=metric, output_path=output_path):
            plot_paths.append(output_path)
    return plot_paths


def _plot_metric(frame: pd.DataFrame, *, metric: str, output_path: Path) -> bool:
    labels = [
        f"{row.dataset}\n{row.method}\n{row.scope}"
        for row in frame.itertuples(index=False)
    ]
    if not labels:
        return False
    x_positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.0), 4.8))
    ax.bar(x_positions, frame[metric])
    ax.set_title(f"Plan D {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Dataset, method, scope")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _top_k_popularity_counts(
    recommendations: pd.DataFrame,
    popularity: pd.Series,
    k: int,
) -> pd.Series:
    top_k = top_k_recommendations(recommendations, k)
    return _popularity_for_items(top_k["recommended_item_id"], popularity)


def _popularity_for_items(items: pd.Series, popularity: pd.Series) -> pd.Series:
    normalized_popularity = popularity.copy()
    normalized_popularity.index = normalized_popularity.index.astype("string").str.strip()
    item_ids = _clean_string_series(items)
    return item_ids.map(normalized_popularity).fillna(0).astype(float)


def _candidate_rows_for_evaluated_users(
    recommendations: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    top_k = top_k_recommendations(recommendations, DEFAULT_CUTOFF)
    candidate_frame = _normalize_dataframe(candidates, frame_name="candidates")
    _require_columns(candidate_frame, frame_name="candidates", columns=("user_id", "candidate_item_id"))
    user_ids = set(top_k["user_id"].dropna().astype(str).tolist())
    candidate_frame["user_id"] = _clean_string_series(candidate_frame["user_id"])
    candidate_frame["candidate_item_id"] = _clean_string_series(candidate_frame["candidate_item_id"])
    return candidate_frame.loc[candidate_frame["user_id"].isin(user_ids)].dropna(subset=["candidate_item_id"])


def _recommended_items(recommendations: pd.DataFrame) -> list[str]:
    if recommendations.empty or "recommended_item_id" not in recommendations.columns:
        return []
    return _clean_string_series(recommendations["recommended_item_id"]).dropna().astype(str).tolist()


def _jaccard_similarity(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    union = left.union(right)
    if not union:
        return None
    return len(left.intersection(right)) / len(union)


def _select_feature_column(frame: pd.DataFrame, *, dataset: str) -> str | None:
    for column in _feature_priority(dataset):
        if column not in frame.columns:
            continue
        parsed_any = frame[column].fillna("").astype("string").str.strip().ne("").any()
        if bool(parsed_any):
            return column
    return None


def _feature_priority(dataset: str) -> list[str]:
    if dataset == "ml1m":
        return ["genres", "categories", "normalized_text", "description", "title"]
    return ["categories", "genres", "normalized_text", "description", "title"]


def _parse_feature_value(value: Any, *, textual: bool) -> set[str]:
    if pd.isna(value):
        return set()
    text = str(value).strip().lower()
    if not text:
        return set()
    if textual:
        return set(re.findall(r"[a-z0-9]+", text))
    return {
        part.strip()
        for part in re.split(r"\s*\|\s*|,|>", text)
        if part.strip()
    }


def _availability_status(metrics: pd.DataFrame, resources: DatasetResources) -> tuple[str, str]:
    if metrics.empty:
        return "MISSING", "No Plan D metric rows were computed."
    partial_reasons: list[str] = []
    if resources.candidates is None or resources.candidates.empty:
        partial_reasons.append("candidate lists unavailable; catalog fallback metrics were used")
    if not has_usable_item_features(resources.item_features):
        partial_reasons.append("item features unavailable; diversity is NaN")
    if pd.to_numeric(metrics["value"], errors="coerce").isna().any():
        partial_reasons.append("one or more metrics are NaN because required optional inputs are unavailable")
    if partial_reasons:
        return "PARTIAL", "; ".join(dict.fromkeys(partial_reasons))
    return "AVAILABLE", "All requested Plan D metrics with available inputs were computed."


def _status_row(
    *,
    spec: PlanDRunSpec,
    cutoff: int,
    status: str,
    required_inputs_found: bool,
    missing_inputs: list[str],
    metrics_written: bool,
    num_users_evaluated: int,
    num_recommendations_evaluated: int,
    catalog_size: int,
    interactions_rows: int,
    candidate_rows: int,
    has_candidates: bool,
    has_item_features: bool,
    recommendation_path: Path,
    notes: str,
    timestamp: str,
    root: Path,
) -> dict[str, Any]:
    return {
        "plan": PLAN_NAME,
        "dataset": spec.dataset,
        "method": spec.method,
        "scope": spec.scope,
        "cutoff": cutoff,
        "status": status,
        "required_inputs_found": bool(required_inputs_found),
        "missing_inputs": "; ".join(missing_inputs),
        "metrics_written": bool(metrics_written),
        "num_users_evaluated": int(num_users_evaluated),
        "num_recommendations_evaluated": int(num_recommendations_evaluated),
        "catalog_size": int(catalog_size),
        "interactions_rows": int(interactions_rows),
        "candidate_rows": int(candidate_rows),
        "has_candidates": bool(has_candidates),
        "has_item_features": bool(has_item_features),
        "input_recommendation_file": _display_path(recommendation_path, root),
        "notes": notes,
        "timestamp": timestamp,
    }


def _dataset_paths(data_root: Path, dataset: str) -> DatasetPaths:
    base_dir = data_root / dataset
    return DatasetPaths(
        base_dir=base_dir,
        interactions=base_dir / "interactions.csv",
        candidates=base_dir / "candidates.csv",
        catalog=base_dir / "catalog.csv",
    )


def _missing_required_inputs(paths: DatasetPaths, recommendation_path: Path) -> list[str]:
    required = {
        "recommendations": recommendation_path,
        "interactions": paths.interactions,
        "catalog": paths.catalog,
    }
    return [f"{name}={path}" for name, path in required.items() if not path.is_file()]


def _read_csv_selected(path: Path, selected_columns: list[str]) -> pd.DataFrame:
    header = [str(column).strip().lower() for column in pd.read_csv(path, nrows=0).columns]
    usecols = [column for column in selected_columns if column in header]
    if not usecols:
        raise ValueError(f"{path} does not contain any requested columns: {selected_columns}")
    return _normalize_dataframe(
        pd.read_csv(path, dtype="string", keep_default_na=False, usecols=usecols),
        frame_name=path.name,
    )


def _safe_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return 0


def _safe_unique_item_count(path: Path, column: str) -> int:
    if not path.is_file():
        return 0
    try:
        frame = _read_csv_selected(path, [column])
    except (OSError, ValueError, pd.errors.ParserError):
        return 0
    if column not in frame.columns:
        return 0
    return int(_clean_string_series(frame[column]).dropna().nunique())


def _select_recommendation_column(frame: pd.DataFrame) -> str:
    if "recommended_item_id" in frame.columns:
        return "recommended_item_id"
    if "item_id" in frame.columns:
        return "item_id"
    raise ValueError("recommendations dataframe must contain recommended_item_id or item_id")


def _clean_string_series(series: pd.Series) -> pd.Series:
    as_text = series.astype("string").str.strip()
    return as_text.mask(as_text.eq(""))


def _normalize_dataframe(df: pd.DataFrame, *, frame_name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{frame_name} must be a pandas DataFrame")
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    duplicated = normalized.columns[normalized.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            f"{frame_name} dataframe has duplicate column name(s) after normalization: "
            f"{', '.join(duplicated)}"
        )
    return normalized


def _require_columns(df: pd.DataFrame, *, frame_name: str, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} dataframe is missing required column(s): {', '.join(missing)}")


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def _resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


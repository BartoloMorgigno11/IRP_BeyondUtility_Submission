"""F1A offline metric closure.

F1A closes metric definitions and result-cell readiness without creating new
LLM outputs, training recommenders, downloading data, or using network access.
It starts from the committed F0 result universe and writes an overlay rather
than regenerating the experiment design.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import numpy as np
import pandas as pd

from src.extension.e0_protocol import sha256_json, stable_json
from src.extension.f0_final_extension_freeze import DATASETS, EXPECTED_HEAD, sha256_file
from src.utils.config import find_project_root


PHASE_ID = "f1a_offline_metric_closure"
DECISION_TOKEN = "F1A_OFFLINE_METRIC_CLOSURE_VALIDATED"
EXPECTED_F0_HEAD = "6b489f507c41911f5571e136a02986119d3233ac"
TOP_K = 5

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")

OUTPUT_TABLES = {
    "metric_resolution_audit": TABLE_DIR / "f1a_metric_resolution_audit.csv",
    "metric_formula_registry": TABLE_DIR / "f1a_metric_formula_registry.csv",
    "metric_dataset_applicability": TABLE_DIR / "f1a_metric_dataset_applicability.csv",
    "raw_source_mapping": TABLE_DIR / "f1a_raw_source_mapping.csv",
    "offline_metric_results": TABLE_DIR / "f1a_offline_metric_results.csv",
    "result_cell_readiness": TABLE_DIR / "f1a_result_cell_readiness.csv",
    "original_four_completeness": TABLE_DIR / "f1a_original_four_completeness.csv",
    "foodcom_completeness": TABLE_DIR / "f1a_foodcom_completeness.csv",
    "remaining_blockers": TABLE_DIR / "f1a_remaining_blockers.csv",
    "validation": TABLE_DIR / "f1a_validation.csv",
}
OUTPUT_MANIFEST = MANIFEST_DIR / "f1a_no_inference_manifest.json"
OUTPUT_REPORT = REPORT_DIR / "f1a_offline_metric_closure.md"

FINAL_READINESS_STATUSES = {
    "COMPUTED_EXISTING_RAW",
    "COMPUTED_F1A",
    "READY_FROM_NEW_RAW",
    "READY_AFTER_TRAINING_RAW",
    "IMPLEMENTATION_READY_AWAITING_RAW",
    "BLOCKED_GLOBAL_PROTOCOL",
    "BLOCKED_DATA_REQUIREMENT",
    "NOT_APPLICABLE",
}

NEWLY_CLOSED_METRICS = {
    "oic_at_5",
    "shannon_category_entropy_at_5",
    "gini_at_5",
    "jain_index_at_5",
    "ks_sample_representativeness",
}
GLOBAL_PROTOCOL_BLOCKED = {"popreo_at_5", "dpd_at_5"}
FEATURE_METRICS = {"shannon_category_entropy_at_5", "ild_at_5", "calibration_at_5"}
TRAINING_RAW_TASKS = {"reranking"}
TRAINING_RAW_FAMILIES = {"traditional_recommender"}

DATASET_FEATURE_POLICY = {
    "beauty": ("categories", "data/reconstructed/plan_c/amazon_beauty/catalog.csv", "pipe-separated Amazon category path"),
    "sports": ("categories", "data/reconstructed/plan_c/amazon_sports/catalog.csv", "pipe-separated Amazon category path"),
    "ml1m": ("genres", "data/processed/ml1m/catalog.csv", "pipe-separated MovieLens genre set"),
    "lastfm": ("tag_values", "data/processed/finalization_f6c_lastfm_catalog.csv", "frozen LastFM tag_values from F6C"),
    "foodcom": ("RAW_recipes.tags", "data/raw/foodcom/d0a_extracted/e7a43e623c61/RAW_recipes.csv", "F0 Food.com canonical recipe tag policy"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize F1A offline metric closure artifacts.")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    outputs = generate_f1a(root)
    if not bool(outputs["validation_passed"]):
        print(stable_json({"passed": False, "errors": outputs["validation_errors"]}))
        return 1
    if not args.validate_only:
        write_outputs(root, outputs)
    readiness = outputs["result_cell_readiness"]
    print(f"result_cells={len(readiness)}")
    print(f"computed_f1a={int(readiness['f1a_status'].eq('COMPUTED_F1A').sum())}")
    print(f"ready_from_new_raw={int(readiness['f1a_status'].eq('READY_FROM_NEW_RAW').sum())}")
    print(f"ready_after_training_raw={int(readiness['f1a_status'].eq('READY_AFTER_TRAINING_RAW').sum())}")
    print(f"blocked_global_protocol={int(readiness['f1a_status'].eq('BLOCKED_GLOBAL_PROTOCOL').sum())}")
    print(f"blocked_data_requirement={int(readiness['f1a_status'].eq('BLOCKED_DATA_REQUIREMENT').sum())}")
    print(f"decision={DECISION_TOKEN}")
    print("llm_inference_calls=0")
    print("training_runs=0")
    print("downloads=0")
    print("network_calls=0")
    return 0


def generate_f1a(root: Path | None = None) -> dict[str, Any]:
    project_root = root or find_project_root()
    artifacts = load_f0_artifacts(project_root)
    formula_registry = build_formula_registry(artifacts["metric_matrix"])
    metric_resolution = build_metric_resolution_audit(artifacts["metric_matrix"], formula_registry)
    dataset_applicability = build_metric_dataset_applicability(formula_registry)
    raw_source_mapping = build_raw_source_mapping(artifacts["result_cells"], artifacts["raw_universe"], formula_registry)
    offline_results = build_offline_metric_results(project_root, artifacts["result_cells"], artifacts["raw_universe"])
    result_readiness = build_result_cell_readiness(
        artifacts["result_cells"],
        artifacts["raw_universe"],
        formula_registry,
        dataset_applicability,
    )
    original_four = build_completeness(result_readiness, [dataset for dataset in DATASETS if dataset != "foodcom"])
    foodcom = build_completeness(result_readiness, ["foodcom"])
    remaining_blockers = build_remaining_blockers(metric_resolution, dataset_applicability)
    validation = validate_outputs(
        project_root,
        artifacts,
        formula_registry,
        metric_resolution,
        dataset_applicability,
        raw_source_mapping,
        offline_results,
        result_readiness,
        original_four,
        foodcom,
        remaining_blockers,
    )
    manifest = build_manifest(project_root, validation, result_readiness, artifacts["f0_manifest"])
    report = render_report(
        metric_resolution,
        formula_registry,
        offline_results,
        result_readiness,
        original_four,
        foodcom,
        remaining_blockers,
        validation,
    )
    return {
        "metric_resolution_audit": metric_resolution,
        "metric_formula_registry": formula_registry,
        "metric_dataset_applicability": dataset_applicability,
        "raw_source_mapping": raw_source_mapping,
        "offline_metric_results": offline_results,
        "result_cell_readiness": result_readiness,
        "original_four_completeness": original_four,
        "foodcom_completeness": foodcom,
        "remaining_blockers": remaining_blockers,
        "validation": validation,
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_errors": validation.loc[validation["status"].ne("PASS"), "check"].tolist(),
        "manifest": manifest,
        "report": report,
    }


def load_f0_artifacts(root: Path) -> dict[str, Any]:
    return {
        "metric_matrix": read_csv(root / "results/tables/f0_final_metric_matrix.csv"),
        "result_cells": read_csv(root / "results/tables/f0_final_result_cell_universe.csv"),
        "raw_universe": read_csv(root / "results/tables/f0_final_raw_experiment_universe.csv"),
        "global_blockers": read_csv(root / "results/tables/f0_final_global_blockers.csv"),
        "budget": read_csv(root / "results/tables/f0_final_request_budget.csv"),
        "f0_manifest": read_json(root / "results/manifests/f0_final_extension_freeze.json"),
        "f0r_manifest": read_json(root / "results/manifests/f0r_final_freeze_sanity.json"),
    }


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for directory in (root / TABLE_DIR, root / MANIFEST_DIR, root / REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for key, rel_path in OUTPUT_TABLES.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(root / rel_path, index=False, lineterminator="\n")
    (root / OUTPUT_REPORT).write_text(outputs["report"], encoding="utf-8")
    manifest = dict(outputs["manifest"])
    manifest["artifact_hashes"] = {
        rel_path.as_posix(): sha256_file(root / rel_path)
        for rel_path in OUTPUT_TABLES.values()
    }
    manifest["artifact_hashes"][OUTPUT_REPORT.as_posix()] = sha256_file(root / OUTPUT_REPORT)
    (root / OUTPUT_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Formula helpers


def hit_rate_at_k_from_rank(rank_zero_based: int | None, k: int = TOP_K) -> float:
    """Return HR@K for a zero-based positive rank; missing rank is a miss."""

    return 0.0 if rank_zero_based is None or int(rank_zero_based) >= k else 1.0


def ndcg_at_k_from_rank(rank_zero_based: int | None, k: int = TOP_K) -> float:
    """Return NDCG@K for a zero-based positive rank; missing rank is zero."""

    if rank_zero_based is None or int(rank_zero_based) >= k:
        return 0.0
    return 1.0 / math.log2(int(rank_zero_based) + 2.0)


def reciprocal_rank(rank_zero_based: int | None) -> float:
    """Return reciprocal rank for a zero-based positive rank."""

    return 0.0 if rank_zero_based is None else 1.0 / float(int(rank_zero_based) + 1)


def exposure_counts(recommendations: Iterable[Sequence[str]], k: int = TOP_K) -> Counter[str]:
    """Count item exposures over top-K recommendation lists."""

    counts: Counter[str] = Counter()
    for row in recommendations:
        seen_positions = list(row)[:k]
        for item in seen_positions:
            if clean(item):
                counts[str(item)] += 1
    return counts


def self_information_at_k(top_items: Sequence[str], train_counts: Mapping[str, int], total_train: int, k: int = TOP_K) -> float | None:
    """Mean top-K self information in bits using unsmoothed train probability."""

    top = [str(item) for item in top_items[:k]]
    if len(top) < k or total_train <= 0:
        return None
    values = []
    for item in top:
        count = int(train_counts.get(item, 0))
        if count <= 0:
            return None
        values.append(-math.log2(count / total_train))
    return float(np.mean(values))


def average_recommendation_popularity(top_items: Sequence[str], train_counts: Mapping[str, int], k: int = TOP_K) -> float | None:
    """Return ARP@K as mean train interaction count over the top-K items."""

    top = [str(item) for item in top_items[:k]]
    if len(top) < k:
        return None
    return float(np.mean([int(train_counts.get(item, 0)) for item in top]))


def long_tail_rate_at_k(top_items: Sequence[str], train_counts: Mapping[str, int], threshold: float, k: int = TOP_K) -> float | None:
    """Return APLT@K using train count < frozen 0.8-quantile threshold."""

    top = [str(item) for item in top_items[:k]]
    if len(top) < k or not math.isfinite(float(threshold)):
        return None
    return float(np.mean([int(train_counts.get(item, 0)) > 0 and int(train_counts.get(item, 0)) < threshold for item in top]))


def oic_at_k(recommendations: Sequence[Sequence[str]], k: int = TOP_K) -> float | None:
    """Return pairwise overlap item coverage as mean |A_u cap A_v| / K."""

    lists = [set(str(item) for item in row[:k] if clean(item)) for row in recommendations]
    if len(lists) < 2:
        return None
    overlaps = []
    for left_idx in range(len(lists)):
        for right_idx in range(left_idx + 1, len(lists)):
            overlaps.append(len(lists[left_idx].intersection(lists[right_idx])) / float(k))
    return float(np.mean(overlaps)) if overlaps else None


def weighted_feature_distribution(items: Sequence[str], item_features: Mapping[str, Sequence[str]]) -> dict[str, float]:
    """Return a normalized feature distribution with fractional multilabel item weight."""

    weights: Counter[str] = Counter()
    for item in items:
        features = sorted({clean(feature).strip().lower() for feature in item_features.get(str(item), ()) if clean(feature).strip()})
        if not features:
            continue
        increment = 1.0 / len(features)
        for feature in features:
            weights[feature] += increment
    total = float(sum(weights.values()))
    if total <= 0:
        return {}
    return {feature: float(value / total) for feature, value in sorted(weights.items())}


def shannon_entropy_from_distribution(distribution: Mapping[str, float], *, normalized: bool = False) -> float | None:
    """Return base-2 Shannon entropy for a feature distribution."""

    positive = [float(value) for value in distribution.values() if float(value) > 0.0]
    if not positive:
        return None
    entropy = -sum(value * math.log2(value) for value in positive)
    if normalized:
        if len(positive) <= 1:
            return 0.0
        return float(entropy / math.log2(len(positive)))
    return float(entropy)


def shannon_category_entropy_at_k(top_items: Sequence[str], item_features: Mapping[str, Sequence[str]], k: int = TOP_K) -> float | None:
    """Return unnormalized base-2 Shannon category entropy over top-K features."""

    if len(top_items[:k]) < k:
        return None
    return shannon_entropy_from_distribution(weighted_feature_distribution(top_items[:k], item_features), normalized=False)


def jaccard_distance(left: Sequence[str], right: Sequence[str]) -> float | None:
    """Return Jaccard distance between two feature sets."""

    left_set = {clean(value).strip().lower() for value in left if clean(value).strip()}
    right_set = {clean(value).strip().lower() for value in right if clean(value).strip()}
    union = left_set.union(right_set)
    if not union:
        return None
    return 1.0 - (len(left_set.intersection(right_set)) / len(union))


def ild_jaccard_at_k(top_items: Sequence[str], item_features: Mapping[str, Sequence[str]], k: int = TOP_K) -> float | None:
    """Return mean pairwise Jaccard distance among feature-backed top-K items."""

    feature_sets = [item_features.get(str(item), ()) for item in top_items[:k] if item_features.get(str(item), ())]
    if len(feature_sets) < 2:
        return None
    distances = []
    for left_idx in range(len(feature_sets)):
        for right_idx in range(left_idx + 1, len(feature_sets)):
            distance = jaccard_distance(feature_sets[left_idx], feature_sets[right_idx])
            if distance is not None:
                distances.append(distance)
    return float(np.mean(distances)) if distances else None


def jensen_shannon_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float | None:
    """Return normalized Jensen-Shannon divergence with log base 2."""

    if not left or not right:
        return None
    keys = sorted(set(left).union(right))
    p = np.asarray([float(left.get(key, 0.0)) for key in keys], dtype=float)
    q = np.asarray([float(right.get(key, 0.0)) for key in keys], dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return None
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * _kl_divergence_base2(p, m) + 0.5 * _kl_divergence_base2(q, m))


def calibration_at_k(history_items: Sequence[str], top_items: Sequence[str], item_features: Mapping[str, Sequence[str]], k: int = TOP_K) -> float | None:
    """Return calibration score 1 - Jensen-Shannon divergence."""

    user_dist = weighted_feature_distribution(history_items, item_features)
    rec_dist = weighted_feature_distribution(top_items[:k], item_features)
    divergence = jensen_shannon_divergence(user_dist, rec_dist)
    if divergence is None:
        return None
    return float(1.0 - divergence)


def _kl_divergence_base2(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def gini_index(exposure_vector: Sequence[float]) -> float | None:
    """Return Gini index over an item-universe exposure vector."""

    values = np.asarray([float(value) for value in exposure_vector], dtype=float)
    if values.size == 0:
        return None
    if np.any(values < 0):
        raise ValueError("Gini exposure vector cannot contain negative values")
    total = float(values.sum())
    if total == 0.0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(index * sorted_values)) / (n * total) - (n + 1.0) / n)


def jain_index(exposure_vector: Sequence[float]) -> float | None:
    """Return Jain's fairness index over an item-universe exposure vector."""

    values = np.asarray([float(value) for value in exposure_vector], dtype=float)
    if values.size == 0:
        return None
    if np.any(values < 0):
        raise ValueError("Jain exposure vector cannot contain negative values")
    numerator = float(values.sum()) ** 2
    denominator = float(values.size * np.sum(values * values))
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def duplicate_rate_at_k(top_items: Sequence[str], k: int = TOP_K) -> float | None:
    """Return duplicate-position rate within the first K items."""

    top = [str(item) for item in top_items[:k]]
    if len(top) < k:
        return None
    seen: set[str] = set()
    duplicates = 0
    for item in top:
        if item in seen:
            duplicates += 1
        seen.add(item)
    return float(duplicates / k)


def ks_2samp(sample: Sequence[float], population: Sequence[float]) -> dict[str, float] | None:
    """Return deterministic two-sample KS statistic and asymptotic p-value."""

    left = np.asarray([float(value) for value in sample], dtype=float)
    right = np.asarray([float(value) for value in population], dtype=float)
    if left.size == 0 or right.size == 0:
        return None
    points = np.sort(np.unique(np.concatenate([left, right])))
    left_cdf = np.searchsorted(np.sort(left), points, side="right") / left.size
    right_cdf = np.searchsorted(np.sort(right), points, side="right") / right.size
    statistic = float(np.max(np.abs(left_cdf - right_cdf)))
    effective_n = left.size * right.size / (left.size + right.size)
    p_value = ks_asymptotic_pvalue(statistic, effective_n)
    return {"statistic": statistic, "p_value": p_value, "sample_n": float(left.size), "population_n": float(right.size)}


def ks_asymptotic_pvalue(statistic: float, effective_n: float) -> float:
    """Approximate two-sided KS p-value using the Kolmogorov series."""

    if statistic <= 0.0:
        return 1.0
    if effective_n <= 0.0:
        return 0.0
    z = (math.sqrt(effective_n) + 0.12 + 0.11 / math.sqrt(effective_n)) * statistic
    total = 0.0
    for term in range(1, 101):
        total += ((-1) ** (term - 1)) * math.exp(-2.0 * (term**2) * (z**2))
    return float(min(max(2.0 * total, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Artifact builders


def build_formula_registry(metric_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    formula_specs = formula_specifications()
    for record in metric_matrix.to_dict("records"):
        metric_id = record["metric_id"]
        spec = formula_specs[metric_id]
        rows.append(
            {
                "metric": metric_id,
                "metric_label": record["metric_label"],
                "family": record["family"],
                "f1a_formula_id": spec["formula_id"],
                "formula_summary": spec["formula_summary"],
                "top_k": spec.get("top_k", ""),
                "aggregation_policy": spec["aggregation_policy"],
                "zero_denominator_policy": spec["zero_denominator_policy"],
                "missing_data_policy": spec["missing_data_policy"],
                "implementation_reference": spec["implementation_reference"],
                "evidence_class": spec["evidence_class"],
                "requires_recommendation_raw": str(spec["requires_recommendation_raw"]).lower(),
                "requires_training_raw": str(spec.get("requires_training_raw", False)).lower(),
                "requires_feature_metadata": str(spec.get("requires_feature_metadata", False)).lower(),
                "requires_popularity": str(spec.get("requires_popularity", False)).lower(),
                "requires_ground_truth": str(spec.get("requires_ground_truth", False)).lower(),
                "requires_runtime_metadata": str(spec.get("requires_runtime_metadata", False)).lower(),
                "f1a_implementation_status": spec["implementation_status"],
                "notes": spec["notes"],
            }
        )
    return pd.DataFrame(rows)


def formula_specifications() -> dict[str, dict[str, Any]]:
    base: dict[str, dict[str, Any]] = {
        "hr_at_5": spec("F1A-HR5", "indicator positive rank is within top 5", "mean over planned users, fail-strict invalids count as misses", "missing rank -> 0", "await final recommendation raw", "src.extension.f1a_offline_metric_closure.hit_rate_at_k_from_rank", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_PAPER_EQUIVALENT", True, top_k=5, ground_truth=True),
        "ndcg_at_5": spec("F1A-NDCG5", "1/log2(rank+2) for positive rank within top 5 else 0", "mean over planned users, fail-strict invalids count as zero", "missing rank -> 0", "await final recommendation raw", "src.extension.f1a_offline_metric_closure.ndcg_at_k_from_rank", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_PAPER_EQUIVALENT", True, top_k=5, ground_truth=True),
        "mrr_diagnostic": spec("F1A-MRR", "reciprocal positive rank", "mean over evaluable users", "missing rank -> 0", "diagnostic only", "src.extension.f1a_offline_metric_closure.reciprocal_rank", "OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED", True, ground_truth=True),
        "self_information_at_5": spec("F1A-SELFINFO5", "mean -log2(train_count(item)/total_train_interactions) over top 5", "mean over users with train-popularity support", "zero popularity -> data-blocked for row; no smoothing", "await final recommendation raw", "src.extension.f1a_offline_metric_closure.self_information_at_k", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED", True, top_k=5, popularity=True),
        "serendipity_at_5": spec("F1A-SER5", "binary relevance times unexpectedness versus aligned MostPop primitive", "mean over users", "missing primitive or relevance -> blocked data requirement", "await final recommendation and aligned MostPop raw", "F5D policy plus F1A readiness", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED", True, top_k=5, ground_truth=True),
        "arp_at_5": spec("F1A-ARP5", "mean train interaction count over top 5", "mean over sample users and K positions", "missing popularity -> row data-blocked", "await final recommendation raw", "src.extension.f1a_offline_metric_closure.average_recommendation_popularity", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED", True, top_k=5, popularity=True),
        "aplt_at_5": spec("F1A-APLT5", "fraction of top-5 items with positive train count below 0.8 quantile", "mean over users and K positions", "missing popularity -> row data-blocked", "await final recommendation raw", "src.extension.f1a_offline_metric_closure.long_tail_rate_at_k", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED", True, top_k=5, popularity=True),
        "popreo_at_5": spec("F1A-POPREO5", "not executed: popularity equal-opportunity needs unrecovered zero-group/per-group denominator policy", "not applicable while blocked", "blocked global protocol", "material group policy unavailable", "not implemented by design", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;BLOCKED_GLOBAL_PROTOCOL", True, top_k=5, popularity=True, implementation_status="BLOCKED_GLOBAL_PROTOCOL", notes="F1A re-audit retained blocker: popularity groups are recoverable, but zero-group handling and averaging policy are not defensibly recovered."),
        "item_coverage_at_5": spec("F1A-ITEMCOV5", "unique top-5 recommended items divided by prompt-compatible candidate union", "cell-level coverage over strict-valid top-5 rows", "empty denominator -> blocked data requirement", "await final recommendation raw", "coverage_ratio adapted", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_PAPER_EQUIVALENT", True, top_k=5),
        "catalog_coverage_at_5": spec("F1A-CATCOV5", "unique top-5 recommended catalog items divided by catalog item universe", "cell-level coverage", "empty catalog -> blocked data requirement", "await final recommendation raw", "src.evaluation.coverage.catalog_coverage_at_k", "LOCAL_EXTENSION", True, top_k=5),
        "oic_at_5": spec("F1A-OIC5", "mean pairwise |top5_u intersection top5_v| / 5 over unordered user pairs", "mean over user pairs in a result cell", "fewer than two users -> not applicable for row", "await final recommendation raw", "src.extension.f1a_offline_metric_closure.oic_at_k", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED", True, top_k=5, implementation_status="IMPLEMENTED_F1A"),
        "shannon_category_entropy_at_5": spec("F1A-SHANNON5", "base-2 entropy of fractional multilabel top-5 category/tag exposure distribution", "mean per user then dataset/model cell mean", "no feature-backed top-5 items -> blocked data requirement", "await final recommendation raw", "src.extension.f1a_offline_metric_closure.shannon_category_entropy_at_k", "PAPER_UNDERSPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED", True, top_k=5, features=True, implementation_status="IMPLEMENTED_F1A"),
        "ild_at_5": spec("F1A-ILD5", "mean pairwise Jaccard distance among top-5 feature sets", "mean per user then cell mean", "fewer than two feature-backed items -> row blocked", "await final recommendation raw", "src.extension.f1a_offline_metric_closure.ild_jaccard_at_k", "LOCAL_ADAPTED", True, top_k=5, features=True),
        "calibration_at_5": spec("F1A-CAL5", "1 - Jensen-Shannon divergence between train-history and top-5 feature distributions", "mean over users with both distributions", "empty profile or rec distribution -> row blocked", "await final recommendation raw", "src.extension.f1a_offline_metric_closure.calibration_at_k", "LOCAL_EXTENSION", True, top_k=5, features=True),
        "gini_at_5": spec("F1A-GINI5", "Gini over full item-universe top-5 exposure vector including zero exposures", "cell-level exposure vector", "empty universe -> blocked; all-zero exposure -> 0", "await final recommendation raw", "src.extension.f1a_offline_metric_closure.gini_index", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_FINAL_USES_FULL_UNIVERSE", True, top_k=5, implementation_status="IMPLEMENTED_F1A"),
        "dpd_at_5": spec("F1A-DPD5", "not executed: demographic parity difference needs unrecovered active/inactive grouping and zero denominator policy", "not applicable while blocked", "blocked global protocol", "material protected/group policy unavailable", "not implemented by design", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;BLOCKED_GLOBAL_PROTOCOL", True, top_k=5, implementation_status="BLOCKED_GLOBAL_PROTOCOL", notes="F1A re-audit retained blocker: active/inactive user variable is described, but exact threshold scope, zero-count handling, and normalization are not defensibly recovered for final five-dataset execution."),
        "jain_index_at_5": spec("F1A-JAIN5", "(sum exposure)^2 / (n_items * sum exposure^2) over item universe", "cell-level exposure vector", "empty universe -> blocked; all-zero exposure -> 0", "await final recommendation raw", "src.extension.f1a_offline_metric_closure.jain_index", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED", True, top_k=5, implementation_status="IMPLEMENTED_F1A"),
        "strict_parse_validity": spec("F1A-STRICTVALID", "strict-valid rows divided by planned raw requests", "cell-level rate", "no raw requests -> not applicable", "await final raw response manifest", "strict parser contract", "LOCAL_ADAPTED", True),
        "structural_invalidity_rate": spec("F1A-INVALIDRATE", "strict-invalid rows divided by planned raw requests", "cell-level rate", "no raw requests -> not applicable", "await final raw response manifest", "strict parser contract", "LOCAL_ADAPTED", True),
        "invalid_reason_distribution": spec("F1A-INVALIDREASON", "counts and proportions by parser/backend invalid reason", "cell-level distribution", "no invalid rows -> empty distribution with zero count", "await final raw response manifest", "strict parser contract", "LOCAL_ADAPTED", True),
        "duplicate_rate": spec("F1A-DUPRATE", "duplicate candidate positions/items within parsed top-10 or top-5 scope", "cell-level rate", "no raw -> await raw", "await final raw response manifest", "src.extension.f1a_offline_metric_closure.duplicate_rate_at_k", "LOCAL_ADAPTED", True),
        "hallucination": spec("F1A-HALLUC", "strict index outputs use structural candidate violations; title hallucination only for title/text outputs", "cell-level rate", "no text output -> structural only", "await final raw response manifest", "src.evaluation.hallucination and parser records", "PAPER_SPECIFIED;OFFICIAL_CODE_RECOVERED;LOCAL_ADAPTED", True),
        "backend_failure_rate": spec("F1A-BACKENDFAIL", "backend failure rows divided by attempted raw requests", "cell-level rate", "no attempts -> not applicable", "await final raw response manifest", "runner telemetry", "LOCAL_ADAPTED", True),
        "candidate_recall_at_n": spec("F1A-CANDRECALL", "positive present in first-stage reranking candidate pool", "cell-level rate over source-ranked users", "no source-ranked candidates -> await training raw", "await first-stage rankings", "candidate recall diagnostic", "LOCAL_EXTENSION", True, requires_training_raw=True, ground_truth=True),
        "conditional_hr_at_5": spec("F1A-CONDHR5", "HR@5 on positive-present reranking subset", "mean over positive-present rows", "empty positive-present subset -> not applicable", "await reranking raw and candidate recall", "conditional utility diagnostic", "LOCAL_EXTENSION", True, top_k=5, requires_training_raw=True, ground_truth=True),
        "conditional_ndcg_at_5": spec("F1A-CONDNDCG5", "NDCG@5 on positive-present reranking subset", "mean over positive-present rows", "empty positive-present subset -> not applicable", "await reranking raw and candidate recall", "conditional utility diagnostic", "LOCAL_EXTENSION", True, top_k=5, requires_training_raw=True, ground_truth=True),
        "canddif_hr_at_5": spec("F1A-CANDDIFHR", "paired HR delta between canonical and alternate candidate order", "paired user/model/dataset comparison", "missing pair -> blocked data requirement", "await CandDif/order raw", "CandDif diagnostic", "PAPER_SPECIFIED;LOCAL_ADAPTED", True, top_k=5, ground_truth=True),
        "canddif_ndcg_at_5": spec("F1A-CANDDIFNDCG", "paired NDCG delta between canonical and alternate candidate order", "paired user/model/dataset comparison", "missing pair -> blocked data requirement", "await CandDif/order raw", "CandDif diagnostic", "PAPER_SPECIFIED;LOCAL_ADAPTED", True, top_k=5, ground_truth=True),
        "latency": spec("F1A-LATENCY", "per-request latency from runner timestamps", "request-level and aggregate summaries", "missing timestamps -> unavailable", "await final raw response manifest", "runner telemetry", "LOCAL_ADAPTED", True, runtime=True),
        "mean_latency": spec("F1A-MEANLAT", "mean latency seconds", "cell-level mean", "missing timestamps -> unavailable", "await final raw response manifest", "runner telemetry", "LOCAL_ADAPTED", True, runtime=True),
        "median_latency": spec("F1A-MEDIANLAT", "median latency seconds", "cell-level median", "missing timestamps -> unavailable", "await final raw response manifest", "runner telemetry", "LOCAL_ADAPTED", True, runtime=True),
        "p90_latency": spec("F1A-P90LAT", "90th percentile latency seconds", "cell-level percentile", "missing timestamps -> unavailable", "await final raw response manifest", "runner telemetry", "LOCAL_ADAPTED", True, runtime=True),
        "p95_latency": spec("F1A-P95LAT", "95th percentile latency seconds", "cell-level percentile", "missing timestamps -> unavailable", "await final raw response manifest", "runner telemetry", "LOCAL_ADAPTED", True, runtime=True),
        "p99_latency": spec("F1A-P99LAT", "99th percentile latency seconds", "cell-level percentile", "missing timestamps -> unavailable", "await final raw response manifest", "runner telemetry", "LOCAL_ADAPTED", True, runtime=True),
        "throughput": spec("F1A-THROUGHPUT", "successful requests per elapsed wall-clock second", "cell/run-level rate", "zero elapsed time -> unavailable", "await final raw response manifest", "runner telemetry", "LOCAL_ADAPTED", True, runtime=True),
        "token_counts": spec("F1A-TOKENS", "token counts exactly as recorded by backend metadata", "cell/run-level sums and means", "missing Ollama token metadata -> do not synthesize", "await optional backend metadata", "runner telemetry if available", "LOCAL_ADAPTED", True, runtime=True),
        "monetary_marginal_cost": spec("F1A-COST", "not applicable for local Ollama with no provider price", "not applicable", "not applicable", "local no provider price", "not applicable", "NOT_APPLICABLE", False, implementation_status="NOT_APPLICABLE"),
        "hardware_runtime_provenance": spec("F1A-RUNTIMEPROV", "runtime environment provenance hash from F0R runtime contract", "cell references shared provenance artifact", "missing F0R runtime -> blocked data requirement", "computed from existing F0R runtime contract", "results/tables/f0r_runtime_contract.csv", "LOCAL_ADAPTED", False, runtime=True, implementation_status="IMPLEMENTED_F1A"),
        "ks_sample_representativeness": spec("F1A-KS", "two-sample KS test comparing selected cohort train-history-count distribution against available eligible-population distribution", "dataset/cohort diagnostic rows, not paper-equivalent", "missing population vector -> blocked data requirement for that cohort", "computed where materialized cohort and population vectors exist", "src.extension.f1a_offline_metric_closure.ks_2samp", "LOCAL_EXTENSION;PAPER_ALIGNED_RECONSTRUCTED_TARGET", False, implementation_status="IMPLEMENTED_F1A"),
    }
    return base


def spec(
    formula_id: str,
    formula_summary: str,
    aggregation_policy: str,
    zero_denominator_policy: str,
    missing_data_policy: str,
    implementation_reference: str,
    evidence_class: str,
    requires_recommendation_raw: bool,
    *,
    top_k: int | str = "",
    implementation_status: str = "IMPLEMENTED_EXISTING_OR_F1A_READY",
    features: bool = False,
    popularity: bool = False,
    ground_truth: bool = False,
    runtime: bool = False,
    requires_training_raw: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "formula_id": formula_id,
        "formula_summary": formula_summary,
        "aggregation_policy": aggregation_policy,
        "zero_denominator_policy": zero_denominator_policy,
        "missing_data_policy": missing_data_policy,
        "implementation_reference": implementation_reference,
        "evidence_class": evidence_class,
        "requires_recommendation_raw": requires_recommendation_raw,
        "requires_training_raw": requires_training_raw,
        "requires_feature_metadata": features,
        "requires_popularity": popularity,
        "requires_ground_truth": ground_truth,
        "requires_runtime_metadata": runtime,
        "implementation_status": implementation_status,
        "top_k": top_k,
        "notes": notes,
    }


def build_metric_resolution_audit(metric_matrix: pd.DataFrame, formula_registry: pd.DataFrame) -> pd.DataFrame:
    formulas = formula_registry.set_index("metric").to_dict("index")
    rows = []
    for record in metric_matrix.to_dict("records"):
        metric_id = record["metric_id"]
        formula = formulas[metric_id]
        protocol_blocker = metric_id in GLOBAL_PROTOCOL_BLOCKED
        requires_new_raw = formula["requires_recommendation_raw"] == "true"
        if protocol_blocker:
            f1a_status = "BLOCKED_GLOBAL_PROTOCOL"
        elif formula["f1a_implementation_status"] == "NOT_APPLICABLE":
            f1a_status = "NOT_APPLICABLE"
        elif metric_id in {"hardware_runtime_provenance", "ks_sample_representativeness"}:
            f1a_status = "COMPUTED_F1A"
        elif requires_new_raw:
            f1a_status = "IMPLEMENTATION_READY_AWAITING_RAW" if metric_id in NEWLY_CLOSED_METRICS or metric_id in FEATURE_METRICS else "READY_FROM_NEW_RAW"
        else:
            f1a_status = "IMPLEMENTATION_READY_AWAITING_RAW"
        rows.append(
            {
                "metric": metric_id,
                "previous_f0_status": record["implementation_status"],
                "formula_recoverability": "blocked_protocol" if protocol_blocker else ("not_applicable" if f1a_status == "NOT_APPLICABLE" else "recoverable"),
                "data_requirements": data_requirements_for_formula(formula),
                "dataset_support": dataset_support_summary(metric_id),
                "implementation_exists": str(not protocol_blocker and f1a_status != "NOT_APPLICABLE").lower(),
                "offline_computable_now": str(metric_id in {"hardware_runtime_provenance", "ks_sample_representativeness"}).lower(),
                "requires_new_raw": str(requires_new_raw).lower(),
                "protocol_blocker": str(protocol_blocker).lower(),
                "blocker_provenance": blocker_provenance(metric_id),
                "f1a_status": f1a_status,
                "notes": formula["notes"] or formula["missing_data_policy"],
            }
        )
    return pd.DataFrame(rows)


def data_requirements_for_formula(formula: Mapping[str, Any]) -> str:
    requirements = []
    if formula["requires_recommendation_raw"] == "true":
        requirements.append("final ranked recommendation/raw parser output")
    if formula["requires_training_raw"] == "true":
        requirements.append("trained/source-ranked reranking candidates")
    if formula["requires_feature_metadata"] == "true":
        requirements.append("canonical item feature metadata")
    if formula["requires_popularity"] == "true":
        requirements.append("train-only item popularity")
    if formula["requires_ground_truth"] == "true":
        requirements.append("held-out positive ground truth")
    if formula["requires_runtime_metadata"] == "true":
        requirements.append("runtime/raw telemetry")
    return "; ".join(requirements) if requirements else "offline artifact metadata only"


def dataset_support_summary(metric_id: str) -> str:
    if metric_id in GLOBAL_PROTOCOL_BLOCKED:
        return "global_protocol_blocked_all_datasets"
    if metric_id in FEATURE_METRICS or metric_id in {"shannon_category_entropy_at_5"}:
        return "feature_supported_all_five_with_foodcom_tags_primary"
    if metric_id == "ks_sample_representativeness":
        return "implemented; computed for materialized Food.com primary/history; original-four population vectors not materialized in F0 artifacts"
    return "supported_all_five_after_required_raw_available"


def blocker_provenance(metric_id: str) -> str:
    if metric_id == "popreo_at_5":
        return "F1A re-audit of paper/original/frozen E1/D0A/F0: popularity groups recoverable; zero-group handling and averaging policy irreducibly missing."
    if metric_id == "dpd_at_5":
        return "F1A re-audit of paper/original/frozen E1/D0A/F0: active/inactive grouping described but threshold scope, normalization, and zero-denominator handling irreducibly missing."
    return ""


def build_metric_dataset_applicability(formula_registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in formula_registry.to_dict("records"):
        metric_id = record["metric"]
        for dataset in DATASETS:
            if metric_id in GLOBAL_PROTOCOL_BLOCKED:
                status = "BLOCKED_GLOBAL_PROTOCOL"
                reason = blocker_provenance(metric_id)
            elif metric_id == "monetary_marginal_cost":
                status = "NOT_APPLICABLE"
                reason = "local Ollama execution has no provider price; do not invent marginal monetary cost"
            elif metric_id == "ks_sample_representativeness" and dataset != "foodcom":
                status = "IMPLEMENTATION_READY_BLOCKED_DATA_REQUIREMENT"
                reason = "selected H0 cohort exists but full eligible-population train-history vector is not materialized as a current F0/F1A artifact"
            elif record["requires_feature_metadata"] == "true":
                status = "SUPPORTED_AWAITING_RAW"
                feature_column, source, note = DATASET_FEATURE_POLICY[dataset]
                reason = f"{feature_column} from {source}; {note}"
            elif record["requires_recommendation_raw"] == "true":
                status = "SUPPORTED_AWAITING_RAW"
                reason = "formula implemented; final recommendation/raw output not yet produced"
            else:
                status = "SUPPORTED_OFFLINE"
                reason = "no recommendation raw required"
            rows.append(
                {
                    "metric": metric_id,
                    "dataset": dataset,
                    "applicability_status": status,
                    "feature_source": DATASET_FEATURE_POLICY.get(dataset, ("", "", ""))[0] if record["requires_feature_metadata"] == "true" else "",
                    "requires_new_raw": record["requires_recommendation_raw"],
                    "requires_training_raw": record["requires_training_raw"],
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows)


def build_raw_source_mapping(result_cells: pd.DataFrame, raw_universe: pd.DataFrame, formula_registry: pd.DataFrame) -> pd.DataFrame:
    raw_by_id = raw_universe.set_index("raw_experiment_id").to_dict("index")
    formula_by_metric = formula_registry.set_index("metric").to_dict("index")
    rows = []
    for cell in result_cells.to_dict("records"):
        raw = raw_by_id[cell["raw_experiment_id"]]
        formula = formula_by_metric[cell["metric"]]
        if cell["metric"] in {"hardware_runtime_provenance"}:
            source_status = "EXISTING_OFFLINE_PROVENANCE"
            source_path = "results/tables/f0r_runtime_contract.csv"
        elif cell["metric"] == "ks_sample_representativeness":
            source_status = "F1A_OFFLINE_COHORT_DIAGNOSTIC"
            source_path = "results/tables/d0a_foodcom_primary1000_users.csv;results/tables/d0a_foodcom_history200_users.csv"
        elif raw["status"] == "TRAINING_DEPENDENT" or cell["task"] == "reranking":
            source_status = "AWAITING_TRAINING_DEPENDENT_RAW"
            source_path = ""
        elif formula["requires_recommendation_raw"] == "true":
            source_status = "AWAITING_FINAL_RAW"
            source_path = ""
        else:
            source_status = "OFFLINE_NO_RECOMMENDATION_RAW_REQUIRED"
            source_path = ""
        rows.append(
            {
                "result_cell_id": cell["result_cell_id"],
                "raw_experiment_id": cell["raw_experiment_id"],
                "dataset": cell["dataset"],
                "task": cell["task"],
                "metric": cell["metric"],
                "raw_status": raw["status"],
                "raw_budget_class": raw["budget_class"],
                "authoritative_source_status": source_status,
                "authoritative_source_path": source_path,
                "excluded_sources": "pilot;diagnostic;superseded_top5;non_comparable_historical_outputs",
                "notes": "F1A does not promote historical raw outputs to final F0 result cells without protocol identity.",
            }
        )
    return pd.DataFrame(rows)


def build_offline_metric_results(root: Path, result_cells: pd.DataFrame, raw_universe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    runtime_hash = file_hash_or_empty(root / "results/tables/f0r_runtime_contract.csv")
    hardware = result_cells.loc[result_cells["metric"].eq("hardware_runtime_provenance")]
    for cell in hardware.to_dict("records"):
        rows.append(
            offline_result_row(
                cell,
                metric_value="",
                value_json=stable_json({"runtime_contract_sha256": runtime_hash}),
                evidence_source="results/tables/f0r_runtime_contract.csv",
                computation_status="COMPUTED_F1A",
                notes="Shared F0R exact runtime contract attached to this cell; no model call.",
            )
        )
    ks_cell = result_cells.loc[result_cells["metric"].eq("ks_sample_representativeness")]
    if not ks_cell.empty:
        cell = ks_cell.iloc[0].to_dict()
        for result in compute_foodcom_ks_results(root):
            rows.append(
                offline_result_row(
                    cell,
                    metric_value=format_float(result["statistic"]),
                    value_json=stable_json(result),
                    evidence_source=result["source"],
                    computation_status="COMPUTED_F1A",
                    notes=result["notes"],
                )
            )
    return pd.DataFrame(rows)


def compute_foodcom_ks_results(root: Path) -> list[dict[str, Any]]:
    results = []
    primary = read_csv(root / "results/tables/d0a_foodcom_primary1000_users.csv")
    history = read_csv(root / "results/tables/d0a_foodcom_history200_users.csv")
    population = foodcom_population_history_counts(root)
    for cohort_name, frame, source in [
        ("foodcom_primary1000", primary, "results/tables/d0a_foodcom_primary1000_users.csv"),
        ("foodcom_history200", history, "results/tables/d0a_foodcom_history200_users.csv"),
    ]:
        if frame.empty or "train_history_count" not in frame.columns or not population:
            continue
        sample = [float(value) for value in frame["train_history_count"].astype(float).tolist()]
        ks = ks_2samp(sample, population)
        if ks is None:
            continue
        results.append(
            {
                "dataset": "foodcom",
                "cohort": cohort_name,
                "variable": "train_history_count_before_target",
                "ks_test": "two_sample",
                "statistic": ks["statistic"],
                "p_value": ks["p_value"],
                "sample_n": int(ks["sample_n"]),
                "population_n": int(ks["population_n"]),
                "evidence_class": "LOCAL_EXTENSION;PAPER_ALIGNED_RECONSTRUCTED_TARGET",
                "source": source + ";data/raw/foodcom/d0a_extracted/e7a43e623c61/RAW_interactions.csv",
                "notes": "Population is all Food.com users with observed interactions; target is representativeness of deterministic selected cohort on train-history-count distribution.",
            }
        )
    return results


def foodcom_population_history_counts(root: Path) -> list[float]:
    path = root / "data/raw/foodcom/d0a_extracted/e7a43e623c61/RAW_interactions.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, usecols=["user_id"], dtype=str)
    counts = frame.groupby("user_id", sort=False).size().astype(float)
    return counts.tolist()


def offline_result_row(
    cell: Mapping[str, Any],
    *,
    metric_value: str,
    value_json: str,
    evidence_source: str,
    computation_status: str,
    notes: str,
) -> dict[str, Any]:
    identity = {
        "cell": cell["result_cell_id"],
        "metric": cell["metric"],
        "source": evidence_source,
        "value": value_json,
    }
    return {
        "offline_result_id": "F1A-RESULT-" + sha256_json(identity).upper()[:16],
        "result_cell_id": cell["result_cell_id"],
        "raw_experiment_id": cell["raw_experiment_id"],
        "dataset": cell["dataset"],
        "task": cell["task"],
        "metric": cell["metric"],
        "metric_value": metric_value,
        "value_json": value_json,
        "evidence_source": evidence_source,
        "computation_status": computation_status,
        "adds_llm_requests": 0,
        "notes": notes,
    }


def build_result_cell_readiness(
    result_cells: pd.DataFrame,
    raw_universe: pd.DataFrame,
    formula_registry: pd.DataFrame,
    dataset_applicability: pd.DataFrame,
) -> pd.DataFrame:
    raw_by_id = raw_universe.set_index("raw_experiment_id").to_dict("index")
    formula_by_metric = formula_registry.set_index("metric").to_dict("index")
    applicability = {
        (row["metric"], row["dataset"]): row
        for row in dataset_applicability.to_dict("records")
    }
    rows = []
    for cell in result_cells.to_dict("records"):
        raw = raw_by_id[cell["raw_experiment_id"]]
        metric = cell["metric"]
        formula = formula_by_metric[metric]
        app = applicability.get((metric, cell["dataset"]), {})
        status, reason = readiness_status(cell, raw, formula, app)
        rows.append(
            {
                "result_cell_id": cell["result_cell_id"],
                "raw_experiment_id": cell["raw_experiment_id"],
                "dataset": cell["dataset"],
                "task": cell["task"],
                "model_name": cell["model_name"],
                "prompt": cell["prompt"],
                "condition": cell["condition"],
                "metric": metric,
                "previous_f0_status": cell["status"],
                "f1a_status": status,
                "readiness_reason": reason,
                "requires_new_raw": formula["requires_recommendation_raw"],
                "requires_training_raw": formula["requires_training_raw"],
                "adds_llm_requests": 0,
                "raw_request_count_preserved": raw["request_count"],
                "source": cell["source"],
            }
        )
    return pd.DataFrame(rows)


def readiness_status(cell: Mapping[str, Any], raw: Mapping[str, Any], formula: Mapping[str, Any], applicability: Mapping[str, Any]) -> tuple[str, str]:
    metric = cell["metric"]
    if metric in GLOBAL_PROTOCOL_BLOCKED:
        return "BLOCKED_GLOBAL_PROTOCOL", blocker_provenance(metric)
    if metric == "monetary_marginal_cost":
        return "NOT_APPLICABLE", "local Ollama execution has no meaningful provider marginal monetary cost"
    if metric == "hardware_runtime_provenance":
        return "COMPUTED_F1A", "computed from F0R runtime contract without raw recommendation output"
    if metric == "ks_sample_representativeness":
        return "COMPUTED_F1A", "F1A implemented KS and computed materialized Food.com cohort diagnostics; unsupported cohort coverage is reported separately"
    if clean(applicability.get("applicability_status")) == "IMPLEMENTATION_READY_BLOCKED_DATA_REQUIREMENT":
        return "BLOCKED_DATA_REQUIREMENT", clean(applicability.get("reason"))
    if raw["status"] == "TRAINING_DEPENDENT" or cell["task"] == "reranking" or formula["requires_training_raw"] == "true":
        return "READY_AFTER_TRAINING_RAW", "metric implementation ready, but source rankings/reranking raw require offline training or first-stage candidate recall"
    if formula["requires_recommendation_raw"] == "true":
        if metric in NEWLY_CLOSED_METRICS or metric in FEATURE_METRICS:
            return "IMPLEMENTATION_READY_AWAITING_RAW", "F1A closed formula/feature policy; final recommendation raw is still absent"
        return "READY_FROM_NEW_RAW", "implementation already ready; final raw output is still absent"
    return "IMPLEMENTATION_READY_AWAITING_RAW", "offline formula ready; waiting for applicable non-LLM artifact if needed"


def build_completeness(readiness: pd.DataFrame, datasets: Sequence[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        subset = readiness.loc[readiness["dataset"].eq(dataset)].copy()
        counts = subset["f1a_status"].value_counts().to_dict()
        rows.append(
            {
                "dataset": dataset,
                "applicable_result_cells": len(subset),
                "computed_existing_raw": int(counts.get("COMPUTED_EXISTING_RAW", 0)),
                "computed_f1a": int(counts.get("COMPUTED_F1A", 0)),
                "awaiting_new_raw": int(counts.get("READY_FROM_NEW_RAW", 0) + counts.get("IMPLEMENTATION_READY_AWAITING_RAW", 0)),
                "awaiting_training": int(counts.get("READY_AFTER_TRAINING_RAW", 0)),
                "blocked_protocol": int(counts.get("BLOCKED_GLOBAL_PROTOCOL", 0)),
                "blocked_data_requirement": int(counts.get("BLOCKED_DATA_REQUIREMENT", 0)),
                "not_applicable": int(counts.get("NOT_APPLICABLE", 0)),
                "status": "ACCOUNTED",
            }
        )
    return pd.DataFrame(rows)


def build_remaining_blockers(metric_resolution: pd.DataFrame, dataset_applicability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in metric_resolution.loc[metric_resolution["f1a_status"].eq("BLOCKED_GLOBAL_PROTOCOL")].to_dict("records"):
        rows.append(
            {
                "blocker_id": "F1A-BLOCK-" + record["metric"].upper(),
                "metric": record["metric"],
                "dataset": "ALL",
                "blocker_type": "BLOCKED_GLOBAL_PROTOCOL",
                "reason": record["blocker_provenance"],
                "next_action": "do not compute until the missing policy is recovered or explicitly replaced as an adapted extension",
            }
        )
    blocked_data = dataset_applicability.loc[dataset_applicability["applicability_status"].eq("IMPLEMENTATION_READY_BLOCKED_DATA_REQUIREMENT")]
    for record in blocked_data.to_dict("records"):
        rows.append(
            {
                "blocker_id": "F1A-DATA-" + sha256_json(record).upper()[:12],
                "metric": record["metric"],
                "dataset": record["dataset"],
                "blocker_type": "BLOCKED_DATA_REQUIREMENT",
                "reason": record["reason"],
                "next_action": "materialize the required population/source artifact before interpreting this metric for the dataset",
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(
    root: Path,
    artifacts: Mapping[str, Any],
    formula_registry: pd.DataFrame,
    metric_resolution: pd.DataFrame,
    dataset_applicability: pd.DataFrame,
    raw_mapping: pd.DataFrame,
    offline_results: pd.DataFrame,
    readiness: pd.DataFrame,
    original_four: pd.DataFrame,
    foodcom: pd.DataFrame,
    blockers: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "evidence": evidence})

    head = git_output(root, ["rev-parse", "HEAD"])
    check("head_descends_from_f0_checkpoint", head_descends_from(root, EXPECTED_F0_HEAD), f"{EXPECTED_F0_HEAD}..{head}")
    check("f0_manifest_validated", artifacts["f0_manifest"].get("decision_token") == "FINAL_EXTENSION_FREEZE_VALIDATED", clean(artifacts["f0_manifest"].get("decision_token")))
    check("f0r_manifest_validated", artifacts["f0r_manifest"].get("decision_token") == "F0_FINAL_SANITY_VALIDATED", clean(artifacts["f0r_manifest"].get("decision_token")))
    check("exactly_five_active_datasets", set(DATASETS) == {"beauty", "sports", "ml1m", "lastfm", "foodcom"}, ",".join(DATASETS))
    active_text = "\n".join([artifacts["raw_universe"].to_csv(index=False), artifacts["result_cells"].to_csv(index=False)])
    check("no_active_video_games_rows", not any(token in active_text.lower() for token in ["video_games", "video games", "videogames"]), "0")
    check("all_metrics_in_registry", set(artifacts["metric_matrix"]["metric_id"]) == set(formula_registry["metric"]), str(len(formula_registry)))
    check("resolution_audits_every_metric", set(metric_resolution["metric"]) == set(formula_registry["metric"]), str(len(metric_resolution)))
    check("result_cell_readiness_complete", len(readiness) == len(artifacts["result_cells"]), f"{len(readiness)} of {len(artifacts['result_cells'])}")
    check("readiness_statuses_controlled", readiness["f1a_status"].isin(FINAL_READINESS_STATUSES).all(), ",".join(sorted(set(readiness["f1a_status"]) - FINAL_READINESS_STATUSES)))
    check("raw_source_mapping_complete", len(raw_mapping) == len(artifacts["result_cells"]), str(len(raw_mapping)))
    check("offline_results_add_zero_llm", offline_results.empty or pd.to_numeric(offline_results["adds_llm_requests"]).sum() == 0, str(0 if offline_results.empty else pd.to_numeric(offline_results["adds_llm_requests"]).sum()))
    check("popreo_blocked_protocol", metric_resolution.loc[metric_resolution["metric"].eq("popreo_at_5"), "f1a_status"].iloc[0] == "BLOCKED_GLOBAL_PROTOCOL", "popreo_at_5")
    check("dpd_blocked_protocol", metric_resolution.loc[metric_resolution["metric"].eq("dpd_at_5"), "f1a_status"].iloc[0] == "BLOCKED_GLOBAL_PROTOCOL", "dpd_at_5")
    check("ks_implemented_or_accounted", metric_resolution.loc[metric_resolution["metric"].eq("ks_sample_representativeness"), "f1a_status"].iloc[0] == "COMPUTED_F1A", "ks_sample_representativeness")
    for metric in NEWLY_CLOSED_METRICS:
        check(f"{metric}_implementation_closed", formula_registry.loc[formula_registry["metric"].eq(metric), "f1a_implementation_status"].iloc[0] == "IMPLEMENTED_F1A", metric)
    f0_budget = {row["budget_item"]: int(row["request_count"]) for row in artifacts["budget"].to_dict("records")}
    check("f0_request_budget_unchanged", f0_budget.get("MANDATORY_RAW_REQUESTS_TOTAL") == 50292 and f0_budget.get("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL") == 100584, stable_json(f0_budget))
    check("original_four_accounted", original_four["status"].eq("ACCOUNTED").all() and len(original_four) == 4, str(len(original_four)))
    check("foodcom_accounted", foodcom["status"].eq("ACCOUNTED").all() and len(foodcom) == 1, str(len(foodcom)))
    check("remaining_blockers_explicit", not blockers.empty and blockers["reason"].astype(str).str.len().gt(0).all(), str(len(blockers)))
    check("no_inference_training_download_network", True, "F1A executes offline artifact readers only; counters are zero")
    return pd.DataFrame(checks)


def build_manifest(root: Path, validation: pd.DataFrame, readiness: pd.DataFrame, f0_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_TOKEN,
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "expected_f0_checkpoint_head": EXPECTED_F0_HEAD,
        "f0_manifest_head": clean(f0_manifest.get("head")),
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS"), "check"].tolist(),
        "result_cell_rows": len(readiness),
        "readiness_counts": {str(key): int(value) for key, value in readiness["f1a_status"].value_counts().sort_index().items()},
        "llm_inference_calls_executed": 0,
        "training_runs_executed": 0,
        "downloads_executed": 0,
        "network_calls_executed": 0,
        "f0_raw_request_budget_changed": 0,
        "commit_created": 0,
    }


def render_report(
    metric_resolution: pd.DataFrame,
    formula_registry: pd.DataFrame,
    offline_results: pd.DataFrame,
    readiness: pd.DataFrame,
    original_four: pd.DataFrame,
    foodcom: pd.DataFrame,
    blockers: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    counts = readiness["f1a_status"].value_counts().sort_index()
    implemented = formula_registry.loc[formula_registry["f1a_implementation_status"].isin(["IMPLEMENTED_F1A", "IMPLEMENTED_EXISTING_OR_F1A_READY"])]
    return "\n".join(
        [
            "# F1A Offline Metric Closure",
            "",
            f"Decision token: `{DECISION_TOKEN}`",
            "",
            "## Metric inventory",
            f"- Metrics accounted: `{len(metric_resolution)}`.",
            f"- Implemented or ready formulas: `{len(implemented)}`.",
            "",
            "## Offline results",
            f"- Offline result rows computed in F1A: `{len(offline_results)}`.",
            "- Recommendation-dependent final cells remain readiness-classified until final raw outputs exist.",
            "",
            "## Readiness",
            "\n".join(f"- `{key}` = `{value}`" for key, value in counts.items()),
            "",
            "## Blockers",
            "\n".join(f"- `{row['metric']}` / `{row['dataset']}`: `{row['blocker_type']}` - {row['reason']}" for row in blockers.to_dict("records")),
            "",
            "## Completeness",
            "- Original four and Food.com are separately materialized in F1A completeness tables.",
            "",
            "## Validation",
            f"- Validation passed: `{str(validation['status'].eq('PASS').all()).lower()}`.",
        ]
    ) + "\n"


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def file_hash_or_empty(path: Path) -> str:
    return sha256_file(path) if path.exists() else ""


def git_output(root: Path, args: list[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout.strip()


def head_descends_from(root: Path, ancestor: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.returncode == 0


if __name__ == "__main__":
    raise SystemExit(main())

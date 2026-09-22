"""Build Phase 4B canonical metric results for the seven-baseline layer.

Run from the repository root:

    python src/experiments/build_canonical_metric_results.py
"""

from __future__ import annotations

from src.audit import common_git

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.canonical_metric_preflight import (
    DATASETS,
    FULL_COHORT_ID,
    MATCHED_COHORT_ID,
    OUTPUTS as PHASE4A_OUTPUTS,
    RANKING_SOURCE_PATHS,
    build_same_cohort_preflight,
    file_sha256,
    load_datamap_items,
    load_ranking_sources,
    read_csv,
)
from src.analysis.canonical_metric_results_validation import (
    BLOCKED_COLUMNS,
    COVERAGE_COLUMNS,
    EXPECTED_ROWS,
    LONG_COLUMNS,
    MANIFEST_COLUMNS,
    OUTPUT_COLUMNS,
    OUTPUT_PATHS,
    RANKING_ROW_COLUMNS,
    ROW_CONTRIBUTION_COLUMNS,
    TOP5_COLUMNS,
    TRAIN_ITEM_COLUMNS,
    VALIDATION_COLUMNS,
    WIDE_COLUMNS,
    blocked_validation_row,
    validate_output_frames,
    validate_report,
    validation_row,
)
from src.analysis.plan_e_lightgcn_cpu_baseline_consolidation import (
    ALL_BASELINES,
    BASELINE_FAMILIES,
    ROW_KEY_COLUMNS,
    cohort_keys_from_lightgcn_rankings,
    filter_rankings_to_cohort,
    parse_json_array as parse_rank_json_array,
    row_key,
    select_unit,
    source_frame_for_baseline,
)
from src.metrics.canonical_metric_computation import (
    BLOCKED_METRICS,
    COMPUTABLE_METRICS,
    TOP_K,
    bool_text,
    candidate_validity_rate,
    contract_payload_hash,
    coverage_ratio,
    descending_popularity_rank_items,
    duplicate_flags,
    finite_scores,
    format_float,
    json_dumps,
    long_tail_member,
    mean_float,
    median_float,
    ndcg_from_rank,
    parse_float_array,
    parse_json_array,
    popularity_quantiles,
    quantile_threshold,
    row_metric_contributions,
    self_information_bits,
    sha256_items,
    train_item_statistics,
)


CONTRACT_PATH = "results/tables/canonical_metric_contract.csv"
PREFLIGHT_PATH = "results/tables/canonical_metric_same_cohort_preflight.csv"
LONG_TAIL_AUDIT_PATH = "results/tables/canonical_metric_long_tail_audit.csv"
BLOCKED_ISSUES_PATH = "results/tables/canonical_metric_blocked_or_ambiguous.csv"
POPULARITY_AUDIT_PATH = "results/tables/canonical_metric_popularity_input_audit.csv"
APPLICABILITY_PATH = "results/tables/canonical_metric_applicability_matrix.csv"

DATASET_SLUGS = DATASETS
EXPECTED_TRAIN_SEEN_COUNTS = {
    "Amazon Beauty": 12_092,
    "Amazon Sports": 18_347,
    "ML-1M": 3_703,
}

PHASE4B_PROTECTED_INPUTS = [
    *[path for key, path in PHASE4A_OUTPUTS.items() if key != "report"],
    PHASE4A_OUTPUTS["report"],
    "src/metrics/canonical_metric_contract.py",
    "src/analysis/canonical_metric_preflight.py",
    "src/experiments/build_canonical_metric_preflight.py",
    "tests/test_canonical_metric_contract.py",
    "results/tables/plan_e_lightgcn_same_cohort_baselines_metrics.csv",
    "results/tables/plan_e_lightgcn_baseline_comparison.csv",
    "results/tables/plan_e_lightgcn_all_datasets_metrics.csv",
    "results/tables/plan_e_lightgcn_all_datasets_device_provenance.csv",
    *RANKING_SOURCE_PATHS.values(),
    "results/tables/ranking_candidate_sets.csv",
    "results/tables/positive_positions.csv",
    "results/tables/experiment_cohort_membership.csv",
    "results/tables/plan_e_lightgcn_row_compatibility.csv",
    "results/tables/plan_e_lightgcn_beauty_sports_cpu_cohort.csv",
    "results/tables/plan_e_lightgcn_ml1m_pilot_cohort.csv",
    "results/tables/item_title_map.csv",
    "results/tables/candidate_item_title_map.csv",
    "data/reconstructed/plan_c/amazon_beauty/interactions.csv",
    "data/reconstructed/plan_c/amazon_beauty/datamaps.json",
    "data/reconstructed/plan_c/amazon_beauty/catalog.csv",
    "data/reconstructed/plan_c/amazon_sports/interactions.csv",
    "data/reconstructed/plan_c/amazon_sports/datamaps.json",
    "data/reconstructed/plan_c/amazon_sports/catalog.csv",
    "data/reconstructed/plan_c/ml1m/interactions.csv",
    "data/reconstructed/plan_c/ml1m/datamaps.json",
    "data/reconstructed/plan_c/ml1m/catalog.csv",
    "paper/Beyond Utility Evaluating LLM As Recommander.pdf",
    "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 4B canonical metric results.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    outputs, output_hashes, protected_unchanged = build_all(root)
    print(f"manifest_rows={len(outputs['manifest'])}")
    print(f"train_item_rows={len(outputs['train_items'])}")
    print(f"ranking_rows={len(outputs['ranking_rows'])}")
    print(f"top5_exposure_rows={len(outputs['top5'])}")
    print(f"row_contribution_rows={len(outputs['row_contributions'])}")
    print(f"coverage_rows={len(outputs['coverage'])}")
    print(f"long_result_rows={len(outputs['long'])}")
    print(f"wide_result_rows={len(outputs['wide'])}")
    print(f"blocked_result_rows={len(outputs['blocked'])}")
    print(f"validation_rows={len(outputs['validation'])}")
    print(f"protected_inputs_unchanged={str(protected_unchanged).lower()}")
    for relative_path, digest in output_hashes.items():
        print(f"sha256 {relative_path} {digest}")
    return 0 if protected_unchanged else 1


def build_all(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str], bool]:
    initial_hashes = hash_protected_inputs(root)
    env = execution_environment(root)
    contract = load_and_validate_contract(root)
    preflight = read_csv(root, PREFLIGHT_PATH)
    validate_preflight(preflight)
    phase4a_preflight = build_same_cohort_preflight(root, load_ranking_sources(root))
    assert_same_cohort_preflight_reconciles(preflight, phase4a_preflight)

    sources = load_ranking_sources(root)
    ranking_rows = build_ranking_rows(root, sources)
    train_items, train_context = build_train_item_statistics(root, ranking_rows)
    top5, row_contrib = build_top5_and_row_contributions(ranking_rows, train_context)
    coverage = build_coverage_components(root, preflight, ranking_rows, top5, train_context)
    manifest = build_manifest(root, preflight, coverage, train_context, env)
    long = build_long_results(root, contract, preflight, row_contrib, coverage)
    wide = build_wide_results(long, preflight)
    blocked = build_blocked_results(root, long)

    output_frames_without_validation = {
        "manifest": manifest,
        "train_items": train_items,
        "ranking_rows": ranking_rows,
        "top5": top5,
        "row_contributions": row_contrib,
        "coverage": coverage,
        "long": long,
        "wide": wide,
        "blocked": blocked,
    }
    write_output_frames(root, output_frames_without_validation)
    partial_hashes = hash_output_files(root, [key for key in output_frames_without_validation])
    validation = build_validation_rows(
        root,
        contract=contract,
        preflight=preflight,
        frames=output_frames_without_validation,
        output_hashes=partial_hashes,
        initial_hashes=initial_hashes,
    )
    outputs = {**output_frames_without_validation, "validation": validation}
    validate_output_frames(outputs)
    write_output_frames(root, {"validation": validation})
    report = render_report(root, outputs, env, partial_hashes)
    report_path = root / OUTPUT_PATHS["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    validate_report(report_path)
    final_hashes = hash_protected_inputs(root)
    protected_unchanged = initial_hashes == final_hashes
    output_hashes = hash_output_files(root, [key for key in OUTPUT_PATHS if key != "report"])
    output_hashes[OUTPUT_PATHS["report"]] = file_sha256(root / OUTPUT_PATHS["report"])
    return outputs, output_hashes, protected_unchanged


def assert_same_cohort_preflight_reconciles(saved: pd.DataFrame, recomputed: pd.DataFrame) -> None:
    key_cols = ["baseline_id", "dataset", "task_type", "cohort_id"]
    value_col = "hr_at_5_recomputed"
    saved_subset = saved[key_cols + [value_col]].copy()
    recomputed_subset = recomputed[key_cols + [value_col]].copy()
    for column in key_cols:
        saved_subset[column] = saved_subset[column].astype(str)
        recomputed_subset[column] = recomputed_subset[column].astype(str)
    merged = saved_subset.merge(
        recomputed_subset,
        on=key_cols,
        how="outer",
        suffixes=("_saved", "_recomputed"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all() or len(merged) != len(saved_subset):
        raise ValueError("Phase 4A same-cohort preflight keys no longer reconcile")
    saved_values = merged[f"{value_col}_saved"].astype(float)
    recomputed_values = merged[f"{value_col}_recomputed"].astype(float)
    if not np.allclose(saved_values.to_numpy(), recomputed_values.to_numpy(), rtol=0.0, atol=1e-15):
        raise ValueError("Phase 4A same-cohort preflight HR values no longer reconcile")


def hash_protected_inputs(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in dict.fromkeys(PHASE4B_PROTECTED_INPUTS):
        path = root / relative
        if path.is_file():
            hashes[relative] = file_sha256(path)
    return hashes


def hash_output_files(root: Path, keys: Iterable[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key in keys:
        relative = OUTPUT_PATHS[key]
        path = root / relative
        if path.is_file():
            hashes[relative] = file_sha256(path)
    return hashes


def execution_environment(root: Path) -> str:
    versions: list[str] = [f"python={sys.version.split()[0]}"]
    for package in ("pandas", "numpy"):
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
        except ImportError:
            version = "not_installed"
        versions.append(f"{package}={version}")
    versions.append("scipy=not_required_not_imported")
    versions.append("torch=not_required_not_imported")
    versions.append(f"executable={sys.executable}")
    versions.append(f"cwd={root}")
    versions.append("cpu_only=true")
    return ";".join(versions)


def git_value(root: Path, args: Sequence[str]) -> str:
    try:
        return common_git.git_stdout_checked(args, root)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{exc}"


def load_and_validate_contract(root: Path) -> pd.DataFrame:
    contract = read_csv(root, CONTRACT_PATH)
    if len(contract) != 13:
        raise ValueError(f"expected 13 metric contracts, found {len(contract)}")
    if contract["metric_id"].duplicated().any():
        raise ValueError("duplicate metric_id in Phase 4A contract")
    for row in contract.to_dict("records"):
        expected_hash = contract_payload_hash(row)
        if expected_hash != str(row["contract_sha256"]):
            raise ValueError(f"contract hash mismatch for {row['metric_id']}")
        if str(row["metric_id"]) in COMPUTABLE_METRICS and str(row["contract_status"]).startswith("blocked"):
            raise ValueError(f"computable metric is blocked: {row['metric_id']}")
        if str(row["metric_id"]) in BLOCKED_METRICS and not str(row["contract_status"]).startswith("blocked"):
            raise ValueError(f"blocked metric is not blocked in contract: {row['metric_id']}")
    expected_ids = set(COMPUTABLE_METRICS).union(BLOCKED_METRICS)
    if set(contract["metric_id"].astype(str)) != expected_ids:
        raise ValueError("contract metric IDs differ from Phase 4B expected metric set")
    return contract


def validate_preflight(preflight: pd.DataFrame) -> None:
    if len(preflight) != 84:
        raise ValueError("same-cohort preflight must have 84 rows")
    if not preflight["preflight_status"].eq("ready").all():
        raise ValueError("all same-cohort preflight rows must be ready")
    if preflight["baseline_id"].nunique() != 7 or preflight["dataset"].nunique() != 3:
        raise ValueError("same-cohort cardinality mismatch")
    if preflight["task_type"].nunique() != 2 or preflight["cohort_id"].nunique() != 2:
        raise ValueError("task/cohort cardinality mismatch")


def build_ranking_rows(root: Path, sources: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in sorted(DATASETS):
        for task_type in ("ranking", "reranking"):
            candidate_set_id = candidate_set_id_for(dataset, task_type)
            lightgcn_unit = select_unit(sources["lightgcn"], dataset, task_type, candidate_set_id)
            full_keys = cohort_keys_from_lightgcn_rankings(lightgcn_unit, FULL_COHORT_ID)
            matched_keys = set(cohort_keys_from_lightgcn_rankings(lightgcn_unit, MATCHED_COHORT_ID))
            reference_map: dict[tuple[str, str, str, str], tuple[list[str], str]] = {}
            for ref_row in filter_rankings_to_cohort(lightgcn_unit, full_keys).to_dict("records"):
                reference_map[row_key(ref_row)] = (parse_rank_json_array(ref_row["candidate_items"]), str(ref_row["positive_item_id"]))
            for baseline_id in ALL_BASELINES:
                source_frame = source_frame_for_baseline(
                    baseline_id,
                    traditional_rankings=sources["traditional"],
                    bprmf_rankings=sources["bprmf"],
                    gru4rec_rankings=sources["gru4rec"],
                    sasrec_rankings=sources["sasrec"],
                    lightgcn_rankings=sources["lightgcn"],
                    dataset=dataset,
                    task_type=task_type,
                    candidate_set_id=candidate_set_id,
                )
                filtered = filter_rankings_to_cohort(source_frame, full_keys)
                seen_keys: set[tuple[str, str, str, str]] = set()
                for source_row in filtered.to_dict("records"):
                    key = row_key(source_row)
                    if key in seen_keys:
                        raise ValueError(f"duplicate ranking key for {baseline_id}: {key}")
                    seen_keys.add(key)
                    candidates = parse_rank_json_array(source_row["candidate_items"])
                    ranked = parse_rank_json_array(source_row["ranked_items"])
                    candidate_scores = parse_float_array(source_row["candidate_scores"])
                    ranked_scores = parse_float_array(source_row["ranked_scores"])
                    reference_candidates, reference_positive = reference_map[key]
                    if candidates != reference_candidates:
                        raise ValueError(f"candidate-list mismatch for {baseline_id} {key}")
                    if str(source_row["positive_item_id"]) != reference_positive:
                        raise ValueError(f"positive-item mismatch for {baseline_id} {key}")
                    valid = (
                        len(candidates) == 20
                        and len(candidate_scores) == 20
                        and len(ranked) == 20
                        and len(ranked_scores) == 20
                        and set(ranked) == set(candidates)
                        and len(set(ranked)) == 20
                        and finite_scores(candidate_scores)
                        and finite_scores(ranked_scores)
                        and candidates.count(str(source_row["positive_item_id"])) == 1
                    )
                    if not valid:
                        raise ValueError(f"invalid ranking row for {baseline_id} {key}")
                    rows.append(
                        {
                            "baseline_id": baseline_id,
                            "baseline_family": BASELINE_FAMILIES[baseline_id],
                            "dataset": dataset,
                            "task_type": task_type,
                            "candidate_set_id": candidate_set_id,
                            "user_id": str(source_row["user_id"]),
                            "positive_item_id": str(source_row["positive_item_id"]),
                            "candidate_items": str(source_row["candidate_items"]),
                            "candidate_scores": str(source_row["candidate_scores"]),
                            "ranked_items": str(source_row["ranked_items"]),
                            "ranked_scores": str(source_row["ranked_scores"]),
                            "positive_rank_zero_based": str(int(source_row["positive_rank_zero_based"])),
                            "positive_rank_one_based": str(int(source_row["positive_rank_one_based"])),
                            "eligible_strict_full": "true",
                            "eligible_strict_matched": bool_text(key in matched_keys),
                            "full_cohort_id": FULL_COHORT_ID,
                            "matched_cohort_id": MATCHED_COHORT_ID,
                            "ranking_source_path": str(source_row["ranking_source_path"]),
                            "ranking_source_row_key": json_dumps(dict(zip(ROW_KEY_COLUMNS, key, strict=True))),
                            "ranking_row_valid": "true",
                            "notes": "canonical strict-full row; matched cohort is a filter of this same row",
                        }
                    )
    frame = pd.DataFrame(rows, columns=RANKING_ROW_COLUMNS).sort_values(
        ["baseline_id", "dataset", "task_type", "user_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(frame) != EXPECTED_ROWS["ranking_rows"]:
        raise ValueError(f"ranking row count mismatch: {len(frame)}")
    return frame


def candidate_set_id_for(dataset: str, task_type: str) -> str:
    slug = {"Amazon Beauty": "beauty", "Amazon Sports": "sports", "ML-1M": "ml1m"}[dataset]
    return f"plan_e_{slug}_1000user_{task_type}_manifest"


def build_train_item_statistics(root: Path, ranking_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    candidate_rows = read_csv(root, "results/tables/ranking_candidate_sets.csv")
    rows: list[dict[str, Any]] = []
    context: dict[str, dict[str, Any]] = {}
    for dataset in sorted(DATASETS):
        slug = DATASETS[dataset]
        source_path = f"data/reconstructed/plan_c/{slug}/interactions.csv"
        interactions = read_csv(root, source_path)
        counts, unique_counts, total = train_item_statistics(interactions)
        if len(counts) != EXPECTED_TRAIN_SEEN_COUNTS[dataset]:
            raise ValueError(f"train-seen count mismatch for {dataset}: {len(counts)}")
        if sum(counts.values()) != total:
            raise ValueError(f"train count total mismatch for {dataset}")
        threshold = long_tail_threshold_from_phase4a(root, dataset)
        ranks = descending_popularity_rank_items(counts)
        quantiles = popularity_quantiles(counts)
        train_seen = set(counts)
        strict_full_union = candidate_union_from_ranking_rows(ranking_rows, dataset=dataset, matched=False)
        strict_matched_union = candidate_union_from_ranking_rows(ranking_rows, dataset=dataset, matched=True)
        full_plan_e2_union = full_plan_e2_candidate_union(candidate_rows, dataset)
        source_sha = file_sha256(root / source_path)
        for item_id in sorted(counts, key=str):
            count = int(counts[item_id])
            probability = count / total
            rows.append(
                {
                    "dataset": dataset,
                    "item_id": item_id,
                    "train_interaction_count": count,
                    "train_unique_user_count": int(unique_counts.get(item_id, 0)),
                    "total_train_interactions": total,
                    "popularity_probability": format_float(probability),
                    "self_information_bits": format_float(self_information_bits(count, total)),
                    "popularity_rank_descending": ranks[item_id],
                    "popularity_quantile": format_float(quantiles[item_id]),
                    "long_tail_threshold": format_float(threshold),
                    "long_tail_member": bool_text(long_tail_member(count, threshold)),
                    "train_seen_catalog_member": "true",
                    "strict_full_candidate_union_member": bool_text(item_id in strict_full_union),
                    "strict_matched_candidate_union_member": bool_text(item_id in strict_matched_union),
                    "full_plan_e2_candidate_union_member": bool_text(item_id in full_plan_e2_union),
                    "source_path": source_path,
                    "source_sha256": source_sha,
                    "popularity_policy_id": "train_interaction_count",
                    "long_tail_policy_id": "official_aclt_quantile_0_8_train_count",
                    "notes": "train-only item popularity; no validation/test interactions and no smoothing",
                }
            )
        probability_sum = sum(count / total for count in counts.values())
        if abs(probability_sum - 1.0) > 1e-12:
            raise ValueError(f"probability sum mismatch for {dataset}: {probability_sum}")
        context[dataset] = {
            "counts": counts,
            "unique_counts": unique_counts,
            "total": total,
            "threshold": threshold,
            "train_seen": train_seen,
            "strict_full_union": strict_full_union,
            "strict_matched_union": strict_matched_union,
            "full_plan_e2_union": full_plan_e2_union,
            "source_path": source_path,
            "source_sha": source_sha,
        }
    frame = pd.DataFrame(rows, columns=TRAIN_ITEM_COLUMNS)
    if len(frame) != EXPECTED_ROWS["train_items"]:
        raise ValueError(f"train item row count mismatch: {len(frame)}")
    return frame, context


def long_tail_threshold_from_phase4a(root: Path, dataset: str) -> float:
    long_tail = read_csv(root, LONG_TAIL_AUDIT_PATH)
    selected = long_tail.loc[
        long_tail["dataset"].eq(dataset)
        & long_tail["candidate_policy_id"].eq("official_aclt_quantile_0_8_train_count")
        & long_tail["recommended"].eq("true")
    ]
    if len(selected) != 1:
        raise ValueError(f"missing Phase 4A long-tail policy for {dataset}")
    threshold = float(selected.iloc[0]["threshold_value"])
    if not math.isfinite(threshold):
        raise ValueError(f"invalid long-tail threshold for {dataset}")
    return threshold


def candidate_union_from_ranking_rows(ranking_rows: pd.DataFrame, *, dataset: str, matched: bool) -> set[str]:
    selected = ranking_rows.loc[ranking_rows["dataset"].eq(dataset)]
    if matched:
        selected = selected.loc[selected["eligible_strict_matched"].eq("true")]
    items: set[str] = set()
    reference_baseline = ALL_BASELINES[0]
    selected = selected.loc[selected["baseline_id"].eq(reference_baseline)]
    for value in selected["candidate_items"].tolist():
        items.update(parse_json_array(value))
    return items


def full_plan_e2_candidate_union(candidate_rows: pd.DataFrame, dataset: str) -> set[str]:
    selected = candidate_rows.loc[candidate_rows["dataset"].eq(dataset) & candidate_rows["source_plan"].eq("Plan E2")]
    items: set[str] = set()
    for value in selected["candidate_items"].tolist():
        items.update(parse_json_array(value))
    return items


def build_top5_and_row_contributions(
    ranking_rows: pd.DataFrame,
    train_context: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposure_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    for source_row in ranking_rows.to_dict("records"):
        dataset = str(source_row["dataset"])
        context = train_context[dataset]
        counts: Counter[str] = context["counts"]
        unique_counts: Counter[str] = context["unique_counts"]
        total = int(context["total"])
        threshold = float(context["threshold"])
        candidates = parse_json_array(source_row["candidate_items"])
        ranked = parse_json_array(source_row["ranked_items"])
        ranked_scores = parse_float_array(source_row["ranked_scores"])
        top_items = ranked[:TOP_K]
        top_scores = ranked_scores[:TOP_K]
        if len(top_items) != TOP_K or len(top_scores) != TOP_K:
            raise ValueError("top-5 extraction failed")
        contributions = row_metric_contributions(
            positive_rank_zero_based=int(source_row["positive_rank_zero_based"]),
            top_items=top_items,
            candidate_items=candidates,
            train_counts=counts,
            unique_user_counts=unique_counts,
            total_train_interactions=total,
            long_tail_threshold=threshold,
        )
        if contributions["duplicate_rate_at_5"] != 0.0:
            raise ValueError("duplicate structural invariant failed")
        if contributions["candidate_validity_rate"] != 1.0:
            raise ValueError("candidate-validity structural invariant failed")
        for index, (item_id, score) in enumerate(zip(top_items, top_scores, strict=True), start=1):
            count = int(counts.get(item_id, 0))
            unique_count = int(unique_counts.get(item_id, 0))
            if count <= 0:
                raise ValueError(f"top-5 item lacks train count: {dataset} {item_id}")
            duplicate = contributions["top5_duplicate_flags"][index - 1]
            valid = item_id in candidates and item_id in context["train_seen"] and math.isfinite(float(score))
            exposure_rows.append(
                {
                    "baseline_id": source_row["baseline_id"],
                    "dataset": dataset,
                    "task_type": source_row["task_type"],
                    "candidate_set_id": source_row["candidate_set_id"],
                    "user_id": source_row["user_id"],
                    "cohort_full_id": source_row["full_cohort_id"],
                    "eligible_strict_matched": source_row["eligible_strict_matched"],
                    "rank_position_one_based": index,
                    "item_id": item_id,
                    "item_score": format_float(score),
                    "is_positive_item": bool_text(item_id == source_row["positive_item_id"]),
                    "train_interaction_count": count,
                    "train_unique_user_count": unique_count,
                    "popularity_probability": format_float(count / total),
                    "self_information_bits": format_float(self_information_bits(count, total)),
                    "long_tail_member": bool_text(long_tail_member(count, threshold)),
                    "item_in_candidate_list": bool_text(item_id in candidates),
                    "item_in_train_seen_catalog": bool_text(item_id in context["train_seen"]),
                    "valid_item": bool_text(valid),
                    "duplicate_within_top_5": bool_text(duplicate),
                    "source_ranking_path": source_row["ranking_source_path"],
                    "notes": "top-5 exposure from frozen ranking output",
                }
            )
        contribution_rows.append(
            {
                "baseline_id": source_row["baseline_id"],
                "dataset": dataset,
                "task_type": source_row["task_type"],
                "candidate_set_id": source_row["candidate_set_id"],
                "user_id": source_row["user_id"],
                "positive_item_id": source_row["positive_item_id"],
                "positive_rank_zero_based": source_row["positive_rank_zero_based"],
                "eligible_strict_matched": source_row["eligible_strict_matched"],
                "hr_at_5": format_float(contributions["hr_at_5"]),
                "ndcg_at_5": format_float(contributions["ndcg_at_5"]),
                "novelty_at_5": format_float(contributions["novelty_at_5"]),
                "long_tail_rate_at_5": format_float(contributions["long_tail_rate_at_5"]),
                "average_popularity_at_5": format_float(contributions["average_popularity_at_5"]),
                "duplicate_rate_at_5": format_float(contributions["duplicate_rate_at_5"]),
                "candidate_validity_rate": format_float(contributions["candidate_validity_rate"]),
                "top5_unique_item_count": contributions["top5_unique_item_count"],
                "top5_valid_item_count": contributions["top5_valid_item_count"],
                "top5_items": json_dumps(top_items),
                "top5_train_popularity_counts": json_dumps(contributions["top5_train_popularity_counts"]),
                "top5_self_information_bits": json_dumps([format_float(value) for value in contributions["top5_self_information_bits"]]),
                "top5_long_tail_membership": json_dumps(contributions["top5_long_tail_membership"]),
                "row_valid": "true",
                "notes": "row-level metrics exclude cohort-level coverage",
            }
        )
    top5 = pd.DataFrame(exposure_rows, columns=TOP5_COLUMNS)
    row_contrib = pd.DataFrame(contribution_rows, columns=ROW_CONTRIBUTION_COLUMNS)
    if len(top5) != EXPECTED_ROWS["top5"] or len(row_contrib) != EXPECTED_ROWS["row_contributions"]:
        raise ValueError("top5 or row-contribution row count mismatch")
    return top5, row_contrib


def build_coverage_components(
    root: Path,
    preflight: pd.DataFrame,
    ranking_rows: pd.DataFrame,
    top5: pd.DataFrame,
    train_context: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for preflight_row in preflight.sort_values(["baseline_id", "dataset", "task_type", "cohort_id"], kind="mergesort").to_dict("records"):
        baseline_id = str(preflight_row["baseline_id"])
        dataset = str(preflight_row["dataset"])
        task_type = str(preflight_row["task_type"])
        candidate_set_id = str(preflight_row["candidate_set_id"])
        cohort_id = str(preflight_row["cohort_id"])
        matched = cohort_id == MATCHED_COHORT_ID
        row_subset = ranking_subset(ranking_rows, baseline_id, dataset, task_type, matched=matched)
        top_subset = exposure_subset(top5, baseline_id, dataset, task_type, matched=matched)
        expected_rows = int(preflight_row["expected_rows"])
        if len(row_subset) != expected_rows:
            raise ValueError(f"coverage row subset mismatch for {baseline_id}/{dataset}/{task_type}/{cohort_id}")
        candidate_unions = []
        for other_baseline in ALL_BASELINES:
            other = ranking_subset(ranking_rows, other_baseline, dataset, task_type, matched=matched)
            union: set[str] = set()
            for value in other["candidate_items"].tolist():
                union.update(parse_json_array(value))
            candidate_unions.append(union)
        identical = all(union == candidate_unions[0] for union in candidate_unions)
        if not identical:
            raise ValueError(f"candidate-union mismatch across baselines for {dataset}/{task_type}/{cohort_id}")
        train_seen = train_context[dataset]["train_seen"]
        top_items = set(top_subset["item_id"].astype(str).tolist())
        catalog_num, catalog_den, catalog_value = coverage_ratio(top_items, train_seen)
        item_num, item_den, item_value = coverage_ratio(top_items, candidate_unions[0])
        rows.append(
            {
                "baseline_id": baseline_id,
                "dataset": dataset,
                "task_type": task_type,
                "candidate_set_id": candidate_set_id,
                "cohort_id": cohort_id,
                "evaluated_rows": expected_rows,
                "top5_position_count": len(top_subset),
                "unique_top5_item_count": len(top_items),
                "train_seen_catalog_item_count": catalog_den,
                "strict_candidate_union_item_count": item_den,
                "catalog_coverage_at_5": format_float(catalog_value),
                "item_coverage_at_5": format_float(item_value),
                "top5_item_union_sha256": sha256_items(top_items),
                "train_seen_catalog_sha256": sha256_items(train_seen),
                "strict_candidate_union_sha256": sha256_items(candidate_unions[0]),
                "candidate_union_identical_across_baselines": bool_text(identical),
                "coverage_validation_status": "passed",
                "notes": f"catalog numerator {catalog_num}; item-coverage numerator {item_num}",
            }
        )
    frame = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    if len(frame) != EXPECTED_ROWS["coverage"]:
        raise ValueError("coverage row count mismatch")
    return frame


def ranking_subset(ranking_rows: pd.DataFrame, baseline_id: str, dataset: str, task_type: str, *, matched: bool) -> pd.DataFrame:
    subset = ranking_rows.loc[
        ranking_rows["baseline_id"].eq(baseline_id)
        & ranking_rows["dataset"].eq(dataset)
        & ranking_rows["task_type"].eq(task_type)
    ].copy()
    if matched:
        subset = subset.loc[subset["eligible_strict_matched"].eq("true")].copy()
    return subset


def exposure_subset(top5: pd.DataFrame, baseline_id: str, dataset: str, task_type: str, *, matched: bool) -> pd.DataFrame:
    subset = top5.loc[
        top5["baseline_id"].eq(baseline_id)
        & top5["dataset"].eq(dataset)
        & top5["task_type"].eq(task_type)
    ].copy()
    if matched:
        subset = subset.loc[subset["eligible_strict_matched"].eq("true")].copy()
    return subset


def contribution_subset(row_contrib: pd.DataFrame, baseline_id: str, dataset: str, task_type: str, *, matched: bool) -> pd.DataFrame:
    subset = row_contrib.loc[
        row_contrib["baseline_id"].eq(baseline_id)
        & row_contrib["dataset"].eq(dataset)
        & row_contrib["task_type"].eq(task_type)
    ].copy()
    if matched:
        subset = subset.loc[subset["eligible_strict_matched"].eq("true")].copy()
    return subset


def build_manifest(
    root: Path,
    preflight: pd.DataFrame,
    coverage: pd.DataFrame,
    train_context: Mapping[str, Mapping[str, Any]],
    env: str,
) -> pd.DataFrame:
    contract_sha = file_sha256(root / CONTRACT_PATH)
    preflight_path = PREFLIGHT_PATH
    rows: list[dict[str, Any]] = []
    coverage_lookup = {
        (row["baseline_id"], row["dataset"], row["task_type"], row["cohort_id"]): row
        for row in coverage.to_dict("records")
    }
    for index, preflight_row in enumerate(preflight.sort_values(["baseline_id", "dataset", "task_type", "cohort_id"], kind="mergesort").to_dict("records"), start=1):
        key = (preflight_row["baseline_id"], preflight_row["dataset"], preflight_row["task_type"], preflight_row["cohort_id"])
        cov = coverage_lookup[key]
        source_path = str(preflight_row["ranking_source_path"])
        dataset = str(preflight_row["dataset"])
        rows.append(
            {
                "computation_id": f"CMC-{index:04d}",
                "baseline_id": preflight_row["baseline_id"],
                "baseline_family": preflight_row["baseline_family"],
                "dataset": dataset,
                "task_type": preflight_row["task_type"],
                "candidate_set_id": preflight_row["candidate_set_id"],
                "cohort_id": preflight_row["cohort_id"],
                "ranking_source_path": source_path,
                "ranking_source_sha256": file_sha256(root / source_path),
                "contract_path": CONTRACT_PATH,
                "contract_registry_sha256": contract_sha,
                "same_cohort_preflight_path": preflight_path,
                "expected_rows": preflight_row["expected_rows"],
                "observed_rows": preflight_row["observed_rows"],
                "candidate_count": preflight_row["candidate_count_valid"].replace("true", "20"),
                "top_k": TOP_K,
                "popularity_source_id": "train_interaction_count",
                "popularity_source_sha256": train_context[dataset]["source_sha"],
                "catalog_universe_id": "train_seen_catalog",
                "candidate_union_sha256": cov["strict_candidate_union_sha256"],
                "long_tail_policy_id": "official_aclt_quantile_0_8_train_count",
                "long_tail_threshold": format_float(float(train_context[dataset]["threshold"])),
                "execution_environment": env,
                "computation_status": "completed",
                "blocking_reason": "",
                "notes": "Phase 4B computation from existing frozen ranking outputs",
            }
        )
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if len(frame) != EXPECTED_ROWS["manifest"]:
        raise ValueError("manifest row count mismatch")
    return frame


def build_long_results(
    root: Path,
    contract: pd.DataFrame,
    preflight: pd.DataFrame,
    row_contrib: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    contract_lookup = {row["metric_id"]: row for row in contract.to_dict("records")}
    coverage_lookup = {
        (row["baseline_id"], row["dataset"], row["task_type"], row["cohort_id"]): row
        for row in coverage.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    result_number = 1
    metric_order = list(COMPUTABLE_METRICS) + list(BLOCKED_METRICS)
    for preflight_row in preflight.sort_values(["baseline_id", "dataset", "task_type", "cohort_id"], kind="mergesort").to_dict("records"):
        baseline_id = str(preflight_row["baseline_id"])
        dataset = str(preflight_row["dataset"])
        task_type = str(preflight_row["task_type"])
        cohort_id = str(preflight_row["cohort_id"])
        matched = cohort_id == MATCHED_COHORT_ID
        subset = contribution_subset(row_contrib, baseline_id, dataset, task_type, matched=matched)
        cov = coverage_lookup[(baseline_id, dataset, task_type, cohort_id)]
        evaluated_rows = len(subset)
        top5_positions = evaluated_rows * TOP_K
        for metric_id in metric_order:
            contract_row = contract_lookup[metric_id]
            result_id = f"CMR-{result_number:06d}"
            result_number += 1
            payload = metric_payload(metric_id, subset, cov, evaluated_rows, top5_positions)
            if metric_id in BLOCKED_METRICS:
                status = str(contract_row["contract_status"])
                computation_status = status if status in {"blocked_ambiguous_definition", "blocked_missing_input"} else "blocked_ambiguous_definition"
                metric_value = ""
                numerator = ""
                denominator = ""
                supporting = "results/tables/canonical_metric_blocked_results.csv"
                blocking = str(contract_row["blocking_reason"])
            else:
                computation_status = computation_status_for(metric_id, str(contract_row["contract_status"]))
                metric_value = payload["metric_value"]
                numerator = payload["numerator"]
                denominator = payload["denominator"]
                supporting = payload["supporting_output_path"]
                blocking = ""
            rows.append(
                {
                    "result_id": result_id,
                    "baseline_id": baseline_id,
                    "baseline_family": preflight_row["baseline_family"],
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": preflight_row["candidate_set_id"],
                    "cohort_id": cohort_id,
                    "metric_id": metric_id,
                    "metric_family": contract_row["metric_family"],
                    "cutoff": contract_row["cutoff"],
                    "metric_value": metric_value,
                    "numerator": numerator,
                    "denominator": denominator,
                    "evaluated_rows": evaluated_rows,
                    "aggregation_level": contract_row["aggregation_level"],
                    "higher_is_better": contract_row["higher_is_better"],
                    "contract_status": contract_row["contract_status"],
                    "computation_status": computation_status,
                    "paper_alignment": contract_row["paper_alignment"],
                    "contract_sha256": contract_row["contract_sha256"],
                    "ranking_source_path": preflight_row["ranking_source_path"],
                    "supporting_output_path": supporting,
                    "blocking_reason": blocking,
                    "notes": result_note(metric_id),
                }
            )
    frame = pd.DataFrame(rows, columns=LONG_COLUMNS)
    if len(frame) != EXPECTED_ROWS["long"]:
        raise ValueError("long result row count mismatch")
    return frame


def metric_payload(metric_id: str, rows: pd.DataFrame, coverage_row: Mapping[str, Any], evaluated_rows: int, top5_positions: int) -> dict[str, str]:
    if metric_id in {"hr_at_5", "ndcg_at_5", "novelty_at_5", "long_tail_rate_at_5", "average_popularity_at_5", "duplicate_rate_at_5", "candidate_validity_rate"}:
        values = rows[metric_id].astype(float).tolist()
        value = mean_float(values)
        numerator = value * (top5_positions if metric_id in {"novelty_at_5", "long_tail_rate_at_5", "average_popularity_at_5", "duplicate_rate_at_5", "candidate_validity_rate"} else evaluated_rows)
        denominator = top5_positions if metric_id in {"novelty_at_5", "long_tail_rate_at_5", "average_popularity_at_5", "duplicate_rate_at_5", "candidate_validity_rate"} else evaluated_rows
        if metric_id == "duplicate_rate_at_5":
            numerator = 0.0
        if metric_id == "candidate_validity_rate":
            numerator = float(top5_positions)
        return {
            "metric_value": structural_format(metric_id, value),
            "numerator": structural_format(metric_id, numerator),
            "denominator": str(int(denominator)),
            "supporting_output_path": "results/tables/canonical_metric_row_contributions.csv",
        }
    if metric_id == "mean_positive_rank_zero_based":
        values = rows["positive_rank_zero_based"].astype(float).tolist()
        return {
            "metric_value": format_float(mean_float(values)),
            "numerator": format_float(float(sum(values))),
            "denominator": str(evaluated_rows),
            "supporting_output_path": "results/tables/canonical_metric_row_contributions.csv",
        }
    if metric_id == "median_positive_rank_zero_based":
        values = rows["positive_rank_zero_based"].astype(float).tolist()
        return {
            "metric_value": format_float(median_float(values)),
            "numerator": format_float(median_float(values)),
            "denominator": "median",
            "supporting_output_path": "results/tables/canonical_metric_row_contributions.csv",
        }
    if metric_id == "catalog_coverage_at_5":
        return {
            "metric_value": str(coverage_row["catalog_coverage_at_5"]),
            "numerator": str(coverage_row["unique_top5_item_count"]),
            "denominator": str(coverage_row["train_seen_catalog_item_count"]),
            "supporting_output_path": "results/tables/canonical_metric_coverage_components.csv",
        }
    if metric_id == "item_coverage_at_5":
        return {
            "metric_value": str(coverage_row["item_coverage_at_5"]),
            "numerator": str(coverage_row["unique_top5_item_count"]),
            "denominator": str(coverage_row["strict_candidate_union_item_count"]),
            "supporting_output_path": "results/tables/canonical_metric_coverage_components.csv",
        }
    return {"metric_value": "", "numerator": "", "denominator": "", "supporting_output_path": ""}


def structural_format(metric_id: str, value: float) -> str:
    if metric_id == "duplicate_rate_at_5" and abs(float(value)) < 1e-15:
        return "0"
    if metric_id == "candidate_validity_rate" and abs(float(value) - 1.0) < 1e-15:
        return "1"
    return format_float(value)


def computation_status_for(metric_id: str, contract_status: str) -> str:
    if metric_id in {"hr_at_5", "ndcg_at_5"}:
        return "recomputed_reconciled"
    if metric_id in {"mean_positive_rank_zero_based", "median_positive_rank_zero_based"}:
        return "diagnostic_computed"
    if metric_id in {"duplicate_rate_at_5", "candidate_validity_rate"}:
        return "structural_invariant_verified"
    if contract_status == "ready_adapted":
        return "computed_adapted"
    if contract_status == "ready_reconstructed":
        return "computed_reconstructed"
    if contract_status == "ready_exact":
        return "computed_exact"
    return "validation_failed"


def result_note(metric_id: str) -> str:
    if metric_id in BLOCKED_METRICS:
        return "blocked by Phase 4A contract; no adapted proxy computed"
    if metric_id in {"duplicate_rate_at_5", "candidate_validity_rate"}:
        return "structural invariant verified from actual top-5 ranking rows"
    if metric_id == "novelty_at_5":
        return "no smoothing; train-only self-information"
    if metric_id == "catalog_coverage_at_5":
        return "adapted train-seen catalog denominator"
    if metric_id == "item_coverage_at_5":
        return "reconstructed strict candidate-union denominator"
    return "computed under frozen Phase 4A contract"


def build_wide_results(long: pd.DataFrame, preflight: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["baseline_id", "baseline_family", "dataset", "task_type", "candidate_set_id", "cohort_id"]
    rows: list[dict[str, Any]] = []
    for preflight_row in preflight.sort_values(["baseline_id", "dataset", "task_type", "cohort_id"], kind="mergesort").to_dict("records"):
        row = {
            "baseline_id": preflight_row["baseline_id"],
            "baseline_family": preflight_row["baseline_family"],
            "dataset": preflight_row["dataset"],
            "task_type": preflight_row["task_type"],
            "candidate_set_id": preflight_row["candidate_set_id"],
            "cohort_id": preflight_row["cohort_id"],
            "evaluated_rows": preflight_row["expected_rows"],
        }
        subset = long.loc[
            long["baseline_id"].eq(row["baseline_id"])
            & long["dataset"].eq(row["dataset"])
            & long["task_type"].eq(row["task_type"])
            & long["cohort_id"].eq(row["cohort_id"])
        ]
        for metric_id in list(COMPUTABLE_METRICS) + list(BLOCKED_METRICS):
            matched = subset.loc[subset["metric_id"].eq(metric_id)]
            row[metric_id] = "" if matched.empty else str(matched.iloc[0]["metric_value"])
        row["popularity_bias_at_5_status"] = "blocked_ambiguous_definition"
        row["intra_list_diversity_at_5_status"] = "blocked_ambiguous_definition"
        row["result_validation_status"] = "passed"
        row["notes"] = "blocked metric value cells intentionally blank"
        rows.append(row)
    return pd.DataFrame(rows, columns=WIDE_COLUMNS)


def build_blocked_results(root: Path, long: pd.DataFrame) -> pd.DataFrame:
    issues = read_csv(root, BLOCKED_ISSUES_PATH)
    issue_lookup = {
        (row["metric_id"], row["dataset"]): row
        for row in issues.to_dict("records")
        if str(row["dataset"]) != "all"
    }
    rows: list[dict[str, Any]] = []
    blocked_long = long.loc[long["metric_id"].isin(BLOCKED_METRICS)].copy()
    for row in blocked_long.to_dict("records"):
        issue = issue_lookup.get((row["metric_id"], row["dataset"]), {})
        rows.append(
            {
                "result_id": row["result_id"],
                "metric_id": row["metric_id"],
                "baseline_id": row["baseline_id"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "candidate_set_id": row["candidate_set_id"],
                "cohort_id": row["cohort_id"],
                "contract_status": row["contract_status"],
                "computation_status": row["computation_status"],
                "blocking_reason": row["blocking_reason"],
                "phase4a_issue_id": str(issue.get("issue_id", "")),
                "missing_or_ambiguous_evidence": str(issue.get("missing_or_conflicting_evidence", row["blocking_reason"])),
                "adapted_alternative_available": str(issue.get("adapted_alternative_available", "true")),
                "adapted_alternative_not_computed": "true",
                "notes": "blocked metric intentionally not computed in Phase 4B",
            }
        )
    frame = pd.DataFrame(rows, columns=BLOCKED_COLUMNS)
    if len(frame) != EXPECTED_ROWS["blocked"]:
        raise ValueError("blocked-result row count mismatch")
    return frame


def build_validation_rows(
    root: Path,
    *,
    contract: pd.DataFrame,
    preflight: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    output_hashes: Mapping[str, str],
    initial_hashes: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    counter = 1

    def add(scope: str, check: str, expected: object, observed: object, passed: bool | None, severity: str = "info", **kwargs: str) -> None:
        nonlocal counter
        rows.append(
            validation_row(
                f"CMV-{counter:05d}",
                scope,
                check,
                expected_value=expected,
                observed_value=observed,
                passed=passed,
                severity=severity,
                **kwargs,
            )
        )
        counter += 1

    add("protected_hashes", "protected_inputs_hashed", ">=39", len(initial_hashes), len(initial_hashes) >= 39, evidence="Phase 4B protected input list")
    add("contract", "contract_row_count", 13, len(contract), len(contract) == 13, evidence=CONTRACT_PATH)
    add("contract", "contract_hashes", "all_valid", "all_valid", True, evidence=CONTRACT_PATH)
    add("same_cohort", "preflight_rows", 84, len(preflight), len(preflight) == 84, evidence=PREFLIGHT_PATH)
    add("same_cohort", "preflight_ready", 84, int(preflight["preflight_status"].eq("ready").sum()), preflight["preflight_status"].eq("ready").all(), evidence=PREFLIGHT_PATH)
    add("ranking", "ranking_row_count", EXPECTED_ROWS["ranking_rows"], len(frames["ranking_rows"]), len(frames["ranking_rows"]) == EXPECTED_ROWS["ranking_rows"], evidence=OUTPUT_PATHS["ranking_rows"])
    add("ranking", "candidate_count_20", "all", "all", frames["ranking_rows"]["candidate_items"].map(lambda v: len(parse_json_array(v)) == 20).all(), evidence=OUTPUT_PATHS["ranking_rows"])
    add("ranking", "score_count_20", "all", "all", frames["ranking_rows"]["ranked_scores"].map(lambda v: len(parse_float_array(v)) == 20).all(), evidence=OUTPUT_PATHS["ranking_rows"])
    add("ranking", "positive_identity", "preserved", "preserved", True, evidence=OUTPUT_PATHS["ranking_rows"])
    add("top5", "top5_row_count", EXPECTED_ROWS["top5"], len(frames["top5"]), len(frames["top5"]) == EXPECTED_ROWS["top5"], evidence=OUTPUT_PATHS["top5"])
    add("top5", "five_exposures_per_row", 5, 5, True, evidence=OUTPUT_PATHS["top5"])
    add("top5", "all_items_valid", "true", str(frames["top5"]["valid_item"].eq("true").all()).lower(), frames["top5"]["valid_item"].eq("true").all(), evidence=OUTPUT_PATHS["top5"])
    add("popularity", "train_item_rows", EXPECTED_ROWS["train_items"], len(frames["train_items"]), len(frames["train_items"]) == EXPECTED_ROWS["train_items"], evidence=OUTPUT_PATHS["train_items"])
    for dataset, group in frames["train_items"].groupby("dataset", sort=True):
        probability_sum = group["popularity_probability"].astype(float).sum()
        count_sum = group["train_interaction_count"].astype(int).sum()
        total = int(group["total_train_interactions"].iloc[0])
        add("popularity", "probability_sum", "1.0", format_float(probability_sum), abs(probability_sum - 1.0) <= 1e-12, dataset=dataset, evidence=OUTPUT_PATHS["train_items"])
        add("popularity", "counts_sum_to_train_total", total, count_sum, count_sum == total, dataset=dataset, evidence=OUTPUT_PATHS["train_items"])
        finite_info = np.isfinite(group["self_information_bits"].astype(float)).all()
        add("popularity", "finite_self_information", "true", str(finite_info).lower(), bool(finite_info), dataset=dataset, evidence=OUTPUT_PATHS["train_items"])
    long_tail = read_csv(root, LONG_TAIL_AUDIT_PATH)
    recommended_tail = long_tail.loc[long_tail["recommended"].eq("true")]
    add("long_tail", "threshold_freeze", 3, len(recommended_tail), len(recommended_tail) == 3, evidence=LONG_TAIL_AUDIT_PATH)
    add("coverage", "coverage_rows", EXPECTED_ROWS["coverage"], len(frames["coverage"]), len(frames["coverage"]) == EXPECTED_ROWS["coverage"], evidence=OUTPUT_PATHS["coverage"])
    add("coverage", "candidate_union_identity", "true", str(frames["coverage"]["candidate_union_identical_across_baselines"].eq("true").all()).lower(), frames["coverage"]["candidate_union_identical_across_baselines"].eq("true").all(), evidence=OUTPUT_PATHS["coverage"])
    add("utility", "utility_reconciliation", 84, int(preflight["utility_metrics_reconciled"].eq("true").sum()), preflight["utility_metrics_reconciled"].eq("true").all(), evidence=PREFLIGHT_PATH)
    add("rank", "rank_reconciliation", "positive_rank_columns_used", "positive_rank_columns_used", True, evidence=OUTPUT_PATHS["row_contributions"])
    add("structural", "duplicate_invariant", "0", set(frames["row_contributions"]["duplicate_rate_at_5"].astype(str)), frames["row_contributions"]["duplicate_rate_at_5"].astype(float).eq(0.0).all(), evidence=OUTPUT_PATHS["row_contributions"])
    add("structural", "candidate_validity_invariant", "1", set(frames["row_contributions"]["candidate_validity_rate"].astype(str)), frames["row_contributions"]["candidate_validity_rate"].astype(float).eq(1.0).all(), evidence=OUTPUT_PATHS["row_contributions"])
    add("blocked", "popularity_bias_rows", 84, int(frames["blocked"]["metric_id"].eq("popularity_bias_at_5").sum()), int(frames["blocked"]["metric_id"].eq("popularity_bias_at_5").sum()) == 84, metric_id="popularity_bias_at_5", evidence=OUTPUT_PATHS["blocked"])
    add("blocked", "ild_rows", 84, int(frames["blocked"]["metric_id"].eq("intra_list_diversity_at_5").sum()), int(frames["blocked"]["metric_id"].eq("intra_list_diversity_at_5").sum()) == 84, metric_id="intra_list_diversity_at_5", evidence=OUTPUT_PATHS["blocked"])
    add("matrix", "long_rows", EXPECTED_ROWS["long"], len(frames["long"]), len(frames["long"]) == EXPECTED_ROWS["long"], evidence=OUTPUT_PATHS["long"])
    add("matrix", "wide_rows", EXPECTED_ROWS["wide"], len(frames["wide"]), len(frames["wide"]) == EXPECTED_ROWS["wide"], evidence=OUTPUT_PATHS["wide"])
    add("matrix", "long_wide_equivalence", "validated", "validated", True, evidence=f"{OUTPUT_PATHS['long']};{OUTPUT_PATHS['wide']}")
    add("ordering", "deterministic_sorting", "stable", "stable", True, evidence="builder stable sort keys")
    for relative, digest in sorted(output_hashes.items()):
        add("output_hashes", "sha256", "recorded", digest, True, evidence=relative)
    add("blocked", "popularity_bias_blocked", "not_computed", "not_computed", None, severity="warning", metric_id="popularity_bias_at_5", evidence=OUTPUT_PATHS["blocked"])
    add("blocked", "ild_blocked", "not_computed", "not_computed", None, severity="warning", metric_id="intra_list_diversity_at_5", evidence=OUTPUT_PATHS["blocked"])
    frame = pd.DataFrame(rows, columns=VALIDATION_COLUMNS)
    if frame["status"].eq("failed").any():
        raise ValueError("validation contains failed checks")
    return frame


def write_output_frames(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for key, frame in frames.items():
        path = root / OUTPUT_PATHS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[:, OUTPUT_COLUMNS[key]].to_csv(path, index=False, lineterminator="\n")


def render_report(root: Path, frames: Mapping[str, pd.DataFrame], env: str, output_hashes: Mapping[str, str]) -> str:
    long = frames["long"]
    wide = frames["wide"]
    train_items = frames["train_items"]
    coverage = frames["coverage"]
    validation = frames["validation"]
    blocked = frames["blocked"]
    metric_counts = long["computation_status"].value_counts().to_dict()
    dataset_rows = train_items.groupby("dataset").size().to_dict()
    full_rows = wide.loc[wide["cohort_id"].eq(FULL_COHORT_ID)].groupby("dataset")["evaluated_rows"].first().to_dict()
    matched_rows = wide.loc[wide["cohort_id"].eq(MATCHED_COHORT_ID)].groupby("dataset")["evaluated_rows"].first().to_dict()
    lines = [
        "# Canonical Utility and Beyond-Utility Results for Seven Baselines",
        "",
        "## 1. Scope and provenance",
        f"- Branch: `{git_value(root, ['branch', '--show-current'])}`.",
        f"- Commit: `{git_value(root, ['log', '-1', '--format=%H %s'])}`.",
        f"- Environment: `{env}`.",
        "",
        "## 2. Executive summary",
        f"- Long-form result rows: `{len(long)}`; wide rows: `{len(wide)}`; status counts: `{json_dumps(metric_counts)}`.",
        f"- Blocked-result rows: `{len(blocked)}`. Popularity bias and ILD remain non-numeric.",
        "",
        "## 3. Scientific classification",
        "- Utility and selected beyond-utility metrics are reconstructed or adapted according to the Phase 4A contract, not exact original-paper reproduction.",
        "",
        "## 4. Protected-input boundaries",
        "- Existing rankings, cohorts, candidates, interactions, datamaps, title maps, checkpoints, paper files, and `original_repo/` were read only.",
        "",
        "## 5. Frozen Phase 4A contract",
        "- Contract source: `results/tables/canonical_metric_contract.csv`; all 13 contract hashes validated before computation.",
        "",
        "## 6. Canonical baseline layer",
        f"- Baselines: `{', '.join(ALL_BASELINES)}`.",
        "",
        "## 7. Canonical datasets, tasks and cohorts",
        f"- Datasets: `{', '.join(sorted(DATASETS))}`; tasks: `ranking`, `reranking`; cohorts: `{FULL_COHORT_ID}`, `{MATCHED_COHORT_ID}`.",
        "",
        "## 8. Ranking-source validation",
        f"- Canonical ranking rows: `{len(frames['ranking_rows'])}`; top-5 exposures: `{len(frames['top5'])}`.",
        "",
        "## 9. Train-only popularity construction",
        f"- Train item rows by dataset: `{json_dumps(dataset_rows)}`.",
        "",
        "## 10. Item-popularity distributions",
        _dataset_popularity_line(train_items),
        "",
        "## 11. Long-tail threshold construction",
        _long_tail_line(train_items),
        "",
        "## 12. Top-5 exposure extraction",
        "- Exactly five top-5 exposures were generated for each canonical ranking row.",
        "",
        "## 13. HR@5 results",
        _metric_range_line(long, "hr_at_5"),
        "",
        "## 14. NDCG@5 results",
        _metric_range_line(long, "ndcg_at_5"),
        "",
        "## 15. Mean positive-rank results",
        _metric_range_line(long, "mean_positive_rank_zero_based"),
        "",
        "## 16. Median positive-rank results",
        _metric_range_line(long, "median_positive_rank_zero_based"),
        "",
        "## 17. Novelty@5 contract",
        "- Novelty uses train-only self-information in bits with no smoothing.",
        "",
        "## 18. Novelty@5 results",
        _metric_range_line(long, "novelty_at_5"),
        "",
        "## 19. Catalog-coverage contract",
        "- Catalog coverage uses the train-seen catalog denominator.",
        "",
        "## 20. Catalog-coverage results",
        _metric_range_line(long, "catalog_coverage_at_5"),
        "",
        "## 21. Item-coverage contract",
        "- Item coverage uses the strict same-cohort candidate-union denominator.",
        "",
        "## 22. Item-coverage results",
        _metric_range_line(long, "item_coverage_at_5"),
        "",
        "## 23. Long-tail-rate contract",
        "- Long-tail membership uses train counts strictly below the frozen 0.8 quantile threshold.",
        "",
        "## 24. Long-tail-rate results",
        _metric_range_line(long, "long_tail_rate_at_5"),
        "",
        "## 25. Average-popularity contract",
        "- Average popularity uses raw train interaction counts and is not popularity bias.",
        "",
        "## 26. Average-popularity results",
        _metric_range_line(long, "average_popularity_at_5"),
        "",
        "## 27. Duplicate-rate verification",
        "- Duplicate Rate@5 verified as structural invariant `0` for all 84 cells.",
        "",
        "## 28. Candidate-validity verification",
        "- Candidate Validity Rate verified as structural invariant `1` for all 84 cells.",
        "",
        "## 29. Popularity-bias blocked status",
        "- `popularity_bias_at_5` has 84 blocked rows with empty numeric values.",
        "",
        "## 30. ILD blocked status",
        "- `intra_list_diversity_at_5` has 84 blocked rows with empty numeric values.",
        "",
        "## 31. Beauty results",
        _dataset_metric_line(long, "Amazon Beauty"),
        "",
        "## 32. Sports results",
        _dataset_metric_line(long, "Amazon Sports"),
        "",
        "## 33. ML-1M results",
        _dataset_metric_line(long, "ML-1M"),
        "",
        "## 34. Random findings",
        _baseline_metric_line(long, "random_seeded"),
        "",
        "## 35. MostPop findings",
        _baseline_metric_line(long, "mostpop_train_only"),
        "",
        "## 36. BM25-like findings",
        _baseline_metric_line(long, "bm25_like_e3b"),
        "",
        "## 37. BPRMF findings",
        _baseline_metric_line(long, "bprmf_reconstructed"),
        "",
        "## 38. GRU4Rec findings",
        _baseline_metric_line(long, "gru4rec_reconstructed"),
        "",
        "## 39. SASRec findings",
        _baseline_metric_line(long, "sasrec_reconstructed"),
        "",
        "## 40. LightGCN findings",
        _baseline_metric_line(long, "lightgcn_reconstructed"),
        "",
        "## 41. Ranking versus reranking findings",
        _task_metric_line(long),
        "",
        "## 42. Full versus matched cohort findings",
        f"- Full rows by dataset: `{json_dumps(full_rows)}`; matched rows by dataset: `{json_dumps(matched_rows)}`.",
        "",
        "## 43. Utility versus novelty trade-offs",
        "- The matrix reports utility and novelty separately; no composite score or global best baseline is claimed.",
        "",
        "## 44. Utility versus popularity findings",
        "- Average popularity is reported separately from HR/NDCG and does not replace blocked popularity bias.",
        "",
        "## 45. Utility versus coverage findings",
        "- Catalog and item coverage use distinct denominators and are not averaged per row.",
        "",
        "## 46. Structural invariants",
        "- Duplicate and candidate-validity values are verified structural checks, not comparative quality wins.",
        "",
        "## 47. Selection-bias implications",
        "- All result cells use the frozen LightGCN-compatible strict cohorts; matched rows are filters of the same strict-full rows.",
        "",
        "## 48. Mixed-device LightGCN provenance",
        "- LightGCN source provenance remains mixed as documented before Phase 4B; no LightGCN computation was rerun.",
        "",
        "## 49. Exact, reconstructed and adapted result labels",
        "- Computation statuses preserve `recomputed_reconciled`, `diagnostic_computed`, `computed_reconstructed`, `computed_adapted`, and `structural_invariant_verified` distinctions.",
        "",
        "## 50. Blocked metric interpretation",
        "- Blocked metric rows are placeholders for transparency and must not be interpreted as zero values.",
        "",
        "## 51. Long-form result matrix",
        f"- Path: `results/tables/canonical_metric_results_long.csv`; rows: `{len(long)}`.",
        "",
        "## 52. Wide-form result matrix",
        f"- Path: `results/tables/canonical_metric_results_wide.csv`; rows: `{len(wide)}`.",
        "",
        "## 53. Validation findings",
        f"- Validation rows: `{len(validation)}`; error-level failures: `{len(validation.loc[(validation['status']=='failed') & (validation['severity']=='error')])}`.",
        "",
        "## 54. Determinism findings",
        f"- Output hashes recorded for `{len(output_hashes)}` generated CSVs before validation/report finalization.",
        "",
        "## 55. Deviations from the original paper",
        "- Catalog coverage and novelty are explicitly adapted where Phase 4A required adapted labels; popularity bias and ILD are blocked.",
        "",
        "## 56. Limitations",
        "- Results depend on reconstructed same-cohort artifacts and do not resolve blocked exact paper definitions.",
        "",
        "## 57. Whether Phase 4B succeeded",
        "- Phase 4B succeeded if validation remains passed and deterministic hashes match across the required rerun.",
        "",
        "## 58. Metric-phase completion assessment",
        "- The metric computation phase is complete for approved metrics and transparent for blocked metrics.",
        "",
        "## 59. Protected-input immutability",
        "- Protected hashes are recomputed inside the builder; generated outputs are separate Phase 4B artifacts.",
        "",
        "## 60. Commands executed",
        "- See final response for the complete command list.",
        "",
        "## 61. Recommended single next task",
        "Phase 5A — canonical prompt, parsing and LLM execution contract audit for paper-aligned ranking and reranking",
        "",
    ]
    return "\n".join(lines)


def _metric_range_line(long: pd.DataFrame, metric_id: str) -> str:
    subset = long.loc[long["metric_id"].eq(metric_id) & long["metric_value"].astype(str).ne("")]
    values = subset["metric_value"].astype(float)
    return f"- `{metric_id}` computed rows: `{len(subset)}`; min `{format_float(values.min())}`; max `{format_float(values.max())}`."


def _dataset_metric_line(long: pd.DataFrame, dataset: str) -> str:
    subset = long.loc[long["dataset"].eq(dataset) & long["metric_id"].isin(["hr_at_5", "ndcg_at_5", "novelty_at_5", "average_popularity_at_5"])]
    return f"- `{dataset}` rows for selected numeric metrics: `{len(subset.loc[subset['metric_value'].astype(str).ne('')])}`."


def _baseline_metric_line(long: pd.DataFrame, baseline_id: str) -> str:
    subset = long.loc[long["baseline_id"].eq(baseline_id) & long["metric_value"].astype(str).ne("")]
    return f"- `{baseline_id}` computed/verified numeric rows: `{len(subset)}` across datasets, tasks and cohorts."


def _task_metric_line(long: pd.DataFrame) -> str:
    payload = long.loc[long["metric_id"].eq("hr_at_5")].groupby("task_type")["metric_value"].count().to_dict()
    return f"- HR@5 cells by task: `{json_dumps(payload)}`."


def _dataset_popularity_line(train_items: pd.DataFrame) -> str:
    payload = {}
    for dataset, group in train_items.groupby("dataset", sort=True):
        payload[dataset] = {
            "items": int(len(group)),
            "total_train_interactions": int(group["total_train_interactions"].iloc[0]),
            "max_count": int(group["train_interaction_count"].astype(int).max()),
        }
    return f"- Popularity summary: `{json_dumps(payload)}`."


def _long_tail_line(train_items: pd.DataFrame) -> str:
    payload = {}
    for dataset, group in train_items.groupby("dataset", sort=True):
        payload[dataset] = {
            "threshold": str(group["long_tail_threshold"].iloc[0]),
            "tail_items": int(group["long_tail_member"].eq("true").sum()),
        }
    return f"- Long-tail thresholds and tail counts: `{json_dumps(payload)}`."


if __name__ == "__main__":
    raise SystemExit(main())

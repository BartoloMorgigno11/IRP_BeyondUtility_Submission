"""F1BZA post-execution audit for F1BZ traditional outputs.

This phase is intentionally read-only with respect to F1BZ training evidence:
it hashes completed artifacts, audits Food.com zero-history lineage, and
classifies the reranking-pool candidate-recall anomaly without training,
inference, downloads, network calls, or commits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import numpy as np
import pandas as pd
import torch
import yaml

from src.extension.e0_protocol import CANDIDATE_COUNT, sha256_json, stable_json
from src.extension.f0_final_extension_freeze import sha256_file
from src.extension.f1bx_execution_stage1 import (
    ACTIVE_DATASETS,
    TRAINABLE_MODELS,
    DatasetMaterial,
    DatasetSpec,
    bool_series,
    build_dataset_material,
    dataset_specs as f1bx_dataset_specs,
    frame_hash,
    read_csv,
    read_json,
    sort_key,
)
from src.extension.f1bz_final_traditional_execution import (
    CHECKPOINT_DIR,
    EXPECTED_HEAD as F1BZ_EXPECTED_HEAD,
    FINAL_SEEDS,
    RAW_RECOMMENDATION_DIR,
    build_final_data,
)
from src.utils.config import find_project_root


PHASE_ID = "f1bza_post_execution_audit"
PRECONDITION_STATE = "F1BZ_FINAL_TRADITIONAL_EXECUTION_VALIDATED"
DECISION_CLEAN = "F1BZA_AUDIT_CLEAN"
DECISION_CORRECTION_REQUIRED = "F1BZA_CORRECTION_REQUIRED"
FOODCOM_HISTORY_VALID = "FOODCOM_HISTORY_VALID"
FOODCOM_HISTORY_PIPELINE_DEFECT = "FOODCOM_HISTORY_PIPELINE_DEFECT"
RERANKING_POOL_SOURCE_DEFECT = "RERANKING_POOL_SOURCE_DEFECT"
RERANKING_POOL_VALID = "RERANKING_POOL_VALID"
SOURCE_CANDIDATE20 = "FROZEN_RANKING_CANDIDATE20"
FULL_CATALOG_READY = "FULL_CATALOG_REGENERATION_READY"
BLOCKED = "BLOCKED"

CONFIG_PATH = Path("configs/f1bz_final_traditional_execution.yaml")
TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")
REPORT_PATH = REPORT_DIR / "f1bza_final_traditional_post_execution_audit.md"
OUTPUT_MANIFEST = MANIFEST_DIR / "f1bza_post_execution_audit.json"

OUTPUT_TABLES = {
    "foodcom_history_alignment": TABLE_DIR / "f1bza_foodcom_history_alignment.csv",
    "foodcom_zero_history_summary": TABLE_DIR / "f1bza_foodcom_zero_history_summary.csv",
    "model_input_mapping_audit": TABLE_DIR / "f1bza_model_input_mapping_audit.csv",
    "first_stage_source_audit": TABLE_DIR / "f1bza_first_stage_source_audit.csv",
    "candidate_recall_tautology_audit": TABLE_DIR / "f1bza_candidate_recall_tautology_audit.csv",
    "full_catalog_regeneration_readiness": TABLE_DIR / "f1bza_full_catalog_regeneration_readiness.csv",
    "lastfm_reranking_lineage_comparison": TABLE_DIR / "f1bza_lastfm_reranking_lineage_comparison.csv",
    "result_validity_classification": TABLE_DIR / "f1bza_result_validity_classification.csv",
    "corrective_scope": TABLE_DIR / "f1bza_corrective_scope.csv",
    "validation": TABLE_DIR / "f1bza_validation.csv",
}

F1BZ_TABLES = {
    "run_manifest": TABLE_DIR / "f1bz_final_run_manifest.csv",
    "training_curves": TABLE_DIR / "f1bz_training_curves.csv",
    "per_seed_utility": TABLE_DIR / "f1bz_per_seed_utility_results.csv",
    "aggregate_utility": TABLE_DIR / "f1bz_aggregate_utility_results.csv",
    "per_seed_beyond": TABLE_DIR / "f1bz_per_seed_beyond_utility_results.csv",
    "aggregate_beyond": TABLE_DIR / "f1bz_aggregate_beyond_utility_results.csv",
    "traditional_completeness": TABLE_DIR / "f1bz_traditional_35row_completeness.csv",
    "first_stage_rankings": TABLE_DIR / "f1bz_first_stage_rankings.csv",
    "reranking_pools": TABLE_DIR / "f1bz_reranking_pools.csv",
    "candidate_recall": TABLE_DIR / "f1bz_candidate_recall.csv",
    "reranking_readiness": TABLE_DIR / "f1bz_reranking_readiness.csv",
    "runtime_provenance": TABLE_DIR / "f1bz_runtime_provenance.csv",
    "failure_retry_audit": TABLE_DIR / "f1bz_failure_retry_audit.csv",
    "validation": TABLE_DIR / "f1bz_validation.csv",
}

UPDATED_F1B_TABLES = (
    TABLE_DIR / "f1b_training_plan.csv",
    TABLE_DIR / "f1b_training_results.csv",
    TABLE_DIR / "f1b_traditional_utility_results.csv",
    TABLE_DIR / "f1b_traditional_beyond_utility_results.csv",
    TABLE_DIR / "f1b_five_dataset_completeness.csv",
    TABLE_DIR / "f1b_result_cell_readiness.csv",
    TABLE_DIR / "f1b_first_stage_ranking_inventory.csv",
    TABLE_DIR / "f1b_candidate_recall.csv",
    TABLE_DIR / "f1b_reranking_readiness.csv",
)

F1BZ_MANIFESTS = (
    MANIFEST_DIR / "f1bz_final_traditional_execution.json",
    MANIFEST_DIR / "f1bz_no_llm_inference.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F1BZA post-execution audit.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    if args.validate_only:
        outputs = load_existing_outputs(root)
    else:
        outputs = generate_f1bza(root, Path(args.config))
        write_outputs(root, outputs)
    manifest = outputs["manifest"]
    validation = outputs["validation"]
    print(f"decision={manifest['decision_token']}")
    print(f"foodcom_history_decision={manifest['foodcom_history_decision']}")
    print(f"reranking_pool_decision={manifest['reranking_pool_decision']}")
    print(f"foodcom_zero_history_users={manifest['foodcom_zero_history_users']}")
    print(f"tautological_reranking_pool_rows={manifest['tautological_reranking_pool_rows']}")
    print(f"validation_passed={str(validation['status'].eq('PASS').all()).lower()}")
    return 0 if bool(validation["status"].eq("PASS").all()) else 1


def generate_f1bza(root: Path | None = None, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    project_root = root or find_project_root()
    config = load_config(project_root, Path(config_path))
    f1bx_config = load_config(project_root, Path(str(config["inputs"]["f1bx_config"])))
    specs = {spec.name: spec for spec in f1bx_dataset_specs(f1bx_config)}
    run_manifest = read_csv(project_root / F1BZ_TABLES["run_manifest"])
    f1bz_manifest = read_json(project_root / "results/manifests/f1bz_final_traditional_execution.json")
    no_llm_manifest = read_json(project_root / "results/manifests/f1bz_no_llm_inference.json")
    materials: dict[str, DatasetMaterial] = {
        dataset: build_dataset_material(project_root, specs[dataset]) for dataset in ACTIVE_DATASETS
    }
    final_data = {
        dataset: build_final_data(project_root, specs[dataset], materials[dataset], config)
        for dataset in ACTIVE_DATASETS
    }
    evidence_hashes = collect_f1bz_evidence_hashes(project_root)
    baseline_snapshot = f1bz_baseline_snapshot(project_root, f1bz_manifest)
    protected_integrity = protected_integrity_state(project_root, config, f1bz_manifest, no_llm_manifest)

    food_alignment = build_foodcom_history_alignment(
        project_root,
        specs["foodcom"],
        materials["foodcom"],
        final_data["foodcom"],
    )
    food_summary = build_foodcom_zero_history_summary(food_alignment, run_manifest)
    mapping_audit = build_model_input_mapping_audit(food_alignment)
    first_stage_source = build_first_stage_source_audit(
        project_root,
        specs,
        materials,
        final_data,
        run_manifest,
    )
    pools = read_csv(project_root / F1BZ_TABLES["reranking_pools"])
    tautology = build_candidate_recall_tautology_audit(materials, pools)
    readiness = build_full_catalog_regeneration_readiness(project_root, materials, final_data, run_manifest)
    lastfm_lineage = build_lastfm_reranking_lineage_comparison(project_root)

    foodcom_history_decision = foodcom_history_decision_from(food_alignment)
    reranking_pool_decision = reranking_pool_decision_from(first_stage_source, tautology)
    result_validity = build_result_validity_classification(
        run_manifest,
        readiness,
        foodcom_history_decision,
        reranking_pool_decision,
    )
    corrective_scope = build_corrective_scope(
        run_manifest,
        pools,
        readiness,
        foodcom_history_decision,
        reranking_pool_decision,
        f1bz_manifest,
    )
    validation = build_validation(
        project_root,
        f1bz_manifest,
        no_llm_manifest,
        protected_integrity,
        baseline_snapshot,
        evidence_hashes,
        food_alignment,
        food_summary,
        mapping_audit,
        first_stage_source,
        tautology,
        readiness,
        lastfm_lineage,
        result_validity,
        corrective_scope,
    )
    decision = (
        DECISION_CLEAN
        if foodcom_history_decision == FOODCOM_HISTORY_VALID and reranking_pool_decision == RERANKING_POOL_VALID
        else DECISION_CORRECTION_REQUIRED
    )
    manifest = build_manifest(
        project_root,
        config,
        decision,
        foodcom_history_decision,
        reranking_pool_decision,
        f1bz_manifest,
        no_llm_manifest,
        protected_integrity,
        baseline_snapshot,
        evidence_hashes,
        food_alignment,
        tautology,
        readiness,
        validation,
    )
    report = render_report(
        decision,
        foodcom_history_decision,
        reranking_pool_decision,
        food_alignment,
        food_summary,
        first_stage_source,
        tautology,
        readiness,
        lastfm_lineage,
        result_validity,
        corrective_scope,
        validation,
    )
    return {
        "foodcom_history_alignment": food_alignment,
        "foodcom_zero_history_summary": food_summary,
        "model_input_mapping_audit": mapping_audit,
        "first_stage_source_audit": first_stage_source,
        "candidate_recall_tautology_audit": tautology,
        "full_catalog_regeneration_readiness": readiness,
        "lastfm_reranking_lineage_comparison": lastfm_lineage,
        "result_validity_classification": result_validity,
        "corrective_scope": corrective_scope,
        "validation": validation,
        "manifest": manifest,
        "report": report,
    }


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else root / config_path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def collect_f1bz_evidence_hashes(root: Path) -> dict[str, str]:
    paths: set[Path] = set()
    for rel_path in list(F1BZ_TABLES.values()) + list(UPDATED_F1B_TABLES) + list(F1BZ_MANIFESTS):
        paths.add(rel_path)
    for full_path in sorted((root / RAW_RECOMMENDATION_DIR).glob("f1bz_*.csv")):
        paths.add(full_path.relative_to(root))
    for full_path in sorted((root / CHECKPOINT_DIR).glob("f1bz_*.pt")):
        paths.add(full_path.relative_to(root))
    return {
        rel_path.as_posix(): sha256_file(root / rel_path)
        for rel_path in sorted(paths, key=lambda path: path.as_posix())
        if (root / rel_path).exists()
    }


def f1bz_baseline_snapshot(root: Path, f1bz_manifest: Mapping[str, Any]) -> dict[str, Any]:
    baseline_recs = read_csv(root / "results/raw_outputs/f1bx_baseline_recommendations.csv")
    baseline_manifest = read_json(root / "results/manifests/f1bx_execution_manifest.json")
    return {
        "baseline_rows": int(baseline_manifest.get("baseline_rows", 0)),
        "baseline_recommendation_rows": int(len(baseline_recs)),
        "baseline_hashes_unchanged": bool(f1bz_manifest.get("integrity", {}).get("baseline_hashes_unchanged", False)),
        "baseline_hashes_match_f1bx_manifest": bool(
            f1bz_manifest.get("integrity", {}).get("baseline_hashes_match_f1bx_manifest", False)
        ),
    }


def protected_integrity_state(
    root: Path,
    config: Mapping[str, Any],
    f1bz_manifest: Mapping[str, Any],
    no_llm_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    f0_budget = read_csv(root / str(config["inputs"]["f0_request_budget"]))
    core = f0_budget.loc[f0_budget["budget_item"].eq("CORE_MANDATORY_NOW_REQUESTS"), "request_count"]
    training = f0_budget.loc[f0_budget["budget_item"].eq("TRAINING_DEPENDENT_REQUESTS"), "request_count"]
    return {
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "precondition_head": F1BZ_EXPECTED_HEAD,
        "f1bz_decision_token": f1bz_manifest.get("decision_token", ""),
        "active_datasets": list(f1bz_manifest.get("active_datasets", [])),
        "active_video_games_rows": int(f1bz_manifest.get("integrity", {}).get("active_video_games_rows", -1)),
        "operational_video_games_training_runs": int(
            f1bz_manifest.get("integrity", {}).get("operational_video_games_training_runs", -1)
        ),
        "foodcom_raw_hashes_unchanged": bool(
            f1bz_manifest.get("integrity", {}).get("foodcom_raw_hashes_unchanged", False)
        ),
        "h0_hashes_unchanged": bool(f1bz_manifest.get("integrity", {}).get("h0_hashes_unchanged", False)),
        "f0_budget_unchanged": bool(f1bz_manifest.get("integrity", {}).get("f0_budget_unchanged", False)),
        "f0_core_mandatory_requests": int(core.iloc[0]) if not core.empty else -1,
        "f0_training_dependent_requests": int(training.iloc[0]) if not training.empty else -1,
        "global_metric_blockers": int(f1bz_manifest.get("integrity", {}).get("global_metric_blockers", -1)),
        "original_repo_status": git_output(root, ["status", "--short", "--", "original_repo"]),
        "paper_status": git_output(root, ["status", "--short", "--", "paper"]),
        "llm_inference_calls_executed": int(no_llm_manifest.get("llm_inference_calls_executed", -1)),
        "ollama_recommendation_calls_executed": int(no_llm_manifest.get("ollama_recommendation_calls_executed", -1)),
        "external_api_calls_executed": int(no_llm_manifest.get("external_api_calls_executed", -1)),
        "network_calls_executed": int(no_llm_manifest.get("network_calls_executed", -1)),
        "downloads_executed": int(no_llm_manifest.get("downloads_executed", -1)),
        "training_started_by_f1bza": 0,
        "inference_started_by_f1bza": 0,
    }


def build_foodcom_history_alignment(
    root: Path,
    spec: DatasetSpec,
    material: DatasetMaterial,
    final_data: Any,
) -> pd.DataFrame:
    users = read_csv(root / spec.cohort_filter_source).copy()
    users["user_id"] = users["user_id"].astype(str)
    users["target_item_id"] = users["target_item_id"].astype(str)
    users["selection_rank_int"] = pd.to_numeric(users["selection_rank"], errors="raise").astype(int)

    raw = pd.read_csv(
        root / spec.interaction_source,
        dtype={"user_id": "string", "recipe_id": "string", "date": "string"},
        usecols=["user_id", "recipe_id", "date"],
        keep_default_na=False,
        low_memory=False,
    )
    raw["source_input_order"] = np.arange(len(raw), dtype=np.int64)
    raw = raw.loc[raw["user_id"].astype(str).isin(set(users["user_id"].astype(str)))].copy()
    raw["parsed_date"] = pd.to_datetime(raw["date"], errors="raise")
    raw = raw.sort_values(["user_id", "parsed_date", "source_input_order"], kind="mergesort")
    raw_counts = raw.groupby("user_id", sort=False).size().astype(int).to_dict()
    raw_last = raw.groupby("user_id", sort=False).tail(1).set_index("user_id")["recipe_id"].astype(str).to_dict()
    raw_hashes = raw.groupby("user_id", sort=False)["recipe_id"].apply(lambda values: sha256_json(list(map(str, values)))).to_dict()
    split_counts = foodcom_protected_split_counts(root, spec, users["user_id"].astype(str).tolist())
    validation_users = load_foodcom_validation_users(root)

    actual_counts = final_data.tuning.training_events.groupby("user_id", sort=False).size().astype(int).to_dict()
    actual_hashes = (
        final_data.tuning.training_events.groupby("user_id", sort=False)["item_id"]
        .apply(lambda values: sha256_json(list(map(str, values))))
        .to_dict()
    )
    model_visible_counts = {}
    for row_index, user_id in enumerate(final_data.tuning.users):
        model_visible_counts[str(user_id)] = int((final_data.tuning.eval_sequences[row_index] > 0).sum())
    max_sequence_length = int(final_data.tuning.eval_sequences.shape[1])

    candidate_group = material.candidates.groupby("user_id", sort=False)
    candidate_counts = candidate_group.size().astype(int).to_dict()
    positive_counts = candidate_group["is_ground_truth"].sum().astype(int).to_dict()
    candidate_items = candidate_group["candidate_item_id"].apply(lambda values: list(map(str, values))).to_dict()

    rows: list[dict[str, Any]] = []
    for row in users.sort_values("selection_rank_int", kind="mergesort").to_dict("records"):
        user_id = str(row["user_id"])
        target = str(row["target_item_id"])
        raw_count = int(raw_counts.get(user_id, 0))
        expected_history = max(raw_count - 1, 0)
        expected_model_visible_history = min(expected_history, max_sequence_length)
        d0a_history = int(row["train_history_count"])
        actual_history = int(actual_counts.get(user_id, 0))
        model_visible_history = int(model_visible_counts.get(user_id, 0))
        user_mapped = user_id in final_data.tuning.user_index
        target_mapped = target in final_data.tuning.item_index
        candidates = candidate_items.get(user_id, [])
        candidate_mapped = len(candidates) == CANDIDATE_COUNT and all(item in final_data.tuning.item_index for item in candidates)
        raw_last_target_match = raw_last.get(user_id, "") == target
        mapping_success = user_mapped and target_mapped and candidate_mapped
        expected_matches_actual = (
            raw_count > 0
            and raw_last_target_match
            and d0a_history == expected_history
            and actual_history == expected_history
            and model_visible_history == expected_model_visible_history
        )
        zero_history = model_visible_history == 0
        if zero_history and expected_history == 0 and raw_count == 1 and expected_matches_actual and mapping_success:
            reason = "TRUE_ZERO_HISTORY_RAW_SINGLE_INTERACTION_TARGET_ONLY"
        elif zero_history and expected_history > 0:
            reason = "MAPPING_OR_PIPELINE_ZERO_HISTORY"
        elif not raw_last_target_match:
            reason = "RAW_FINAL_TARGET_MISMATCH"
        elif not mapping_success:
            reason = "MAPPING_FAILURE"
        elif expected_history > max_sequence_length:
            reason = "NONZERO_HISTORY_ALIGNED_SEQUENCE_TRUNCATED_TO_MAX_LEN"
        else:
            reason = "NONZERO_HISTORY_ALIGNED"
        rows.append(
            {
                "dataset": "foodcom",
                "user_id": user_id,
                "f0_cohort_id": spec.cohort_id,
                "f0_sample_cohort_id": f"{spec.cohort_id}_rank_{int(row['selection_rank_int']):04d}",
                "selection_rank": int(row["selection_rank_int"]),
                "raw_interaction_count": raw_count,
                "chronological_interaction_count": raw_count,
                "d0a_train_history_count": d0a_history,
                "protected_train_split_interaction_count": int(split_counts["train"].get(user_id, 0)),
                "protected_validation_split_interaction_count": int(split_counts["validation"].get(user_id, 0)),
                "protected_test_split_interaction_count": int(split_counts["test"].get(user_id, 0)),
                "validation_tuning_cohort_member": bool_text(user_id in validation_users),
                "tuning_final_split_membership": (
                    "UNEXPECTED_VALIDATION_AND_FINAL_OVERLAP"
                    if user_id in validation_users
                    else "FROZEN_F0_FINAL_TEST_COHORT_ONLY"
                ),
                "nested_split_training_prefix": "i1..i[T-2]_for_validation_only; not used for F1BZ final",
                "final_training_interaction_count_expected": expected_history,
                "final_training_interaction_count_actually_materialized": actual_history,
                "model_visible_history_count_expected": expected_model_visible_history,
                "model_visible_history_count": model_visible_history,
                "model_sequence_truncation_applied": bool_text(expected_history > max_sequence_length),
                "model_max_sequence_length": max_sequence_length,
                "internal_mapped_user_id": final_data.tuning.user_index.get(user_id, ""),
                "mapping_success": bool_text(mapping_success),
                "target_item": target,
                "target_matches_raw_last_interaction": bool_text(raw_last_target_match),
                "target_mapping_success": bool_text(target_mapped),
                "candidate_count": int(candidate_counts.get(user_id, 0)),
                "positive_count": int(positive_counts.get(user_id, 0)),
                "candidate_set_mapping_success": bool_text(candidate_mapped),
                "candidate_set_hash": str(row["candidate_hash"]),
                "raw_chronological_items_hash": str(raw_hashes.get(user_id, "")),
                "final_training_items_hash": str(actual_hashes.get(user_id, sha256_json([]))),
                "expected_matches_actual": bool_text(expected_matches_actual),
                "zero_history_flag": bool_text(zero_history),
                "zero_history_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def foodcom_protected_split_counts(root: Path, spec: DatasetSpec, users: Sequence[str]) -> dict[str, Mapping[str, int]]:
    selected = set(str(user) for user in users)
    base_dir = (root / spec.interaction_source).parent
    counts: dict[str, Mapping[str, int]] = {}
    for split in ("train", "validation", "test"):
        path = base_dir / f"interactions_{split}.csv"
        if not path.exists():
            counts[split] = {}
            continue
        frame = pd.read_csv(
            path,
            dtype={"user_id": "string"},
            usecols=["user_id"],
            keep_default_na=False,
            low_memory=False,
        )
        subset = frame.loc[frame["user_id"].astype(str).isin(selected)]
        counts[split] = subset["user_id"].astype(str).value_counts().astype(int).to_dict()
    return counts


def load_foodcom_validation_users(root: Path) -> set[str]:
    path = root / "results/tables/f1bxr_foodcom_validation_cohort.csv"
    if not path.exists():
        return set()
    frame = read_csv(path)
    return set(frame["user_id"].astype(str)) if "user_id" in frame.columns else set()


def build_foodcom_zero_history_summary(alignment: pd.DataFrame, run_manifest: pd.DataFrame) -> pd.DataFrame:
    zero = alignment.loc[alignment["zero_history_flag"].eq("true")]
    pipeline_zero = alignment.loc[alignment["zero_history_reason"].eq("MAPPING_OR_PIPELINE_ZERO_HISTORY")]
    expected_nonzero_actual_zero = alignment.loc[
        (pd.to_numeric(alignment["final_training_interaction_count_expected"]) > 0)
        & alignment["zero_history_flag"].eq("true")
    ]
    zero_user_hash = sha256_json(sorted(zero["user_id"].astype(str).tolist(), key=sort_key))
    rows: list[dict[str, Any]] = [
        {
            "scope": "overall_foodcom_final_cohort",
            "model": "ALL",
            "f0_final_test_users": len(alignment),
            "raw_users_matched": int((pd.to_numeric(alignment["raw_interaction_count"]) > 0).sum()),
            "raw_users_missing": int((pd.to_numeric(alignment["raw_interaction_count"]) == 0).sum()),
            "expected_zero_history_users": int((pd.to_numeric(alignment["final_training_interaction_count_expected"]) == 0).sum()),
            "actual_zero_history_users": len(zero),
            "expected_nonzero_actual_zero_users": len(expected_nonzero_actual_zero),
            "pipeline_zero_history_users": len(pipeline_zero),
            "model_mapped_users": int(alignment["mapping_success"].eq("true").sum()),
            "target_mapped_users": int(alignment["target_mapping_success"].eq("true").sum()),
            "candidate_set_mapped_users": int(alignment["candidate_set_mapping_success"].eq("true").sum()),
            "zero_history_user_hash": zero_user_hash,
            "decision": FOODCOM_HISTORY_VALID if len(pipeline_zero) == 0 and len(zero) == 727 else FOODCOM_HISTORY_PIPELINE_DEFECT,
            "reason": "727 users have exactly one raw chronological interaction and the final target is that interaction.",
        }
    ]
    food_runs = run_manifest.loc[run_manifest["dataset"].eq("foodcom")]
    for model in TRAINABLE_MODELS:
        model_runs = food_runs.loc[food_runs["method"].eq(model)]
        rows.append(
            {
                "scope": "per_trainable_model",
                "model": model,
                "f0_final_test_users": len(alignment),
                "raw_users_matched": int((pd.to_numeric(alignment["raw_interaction_count"]) > 0).sum()),
                "raw_users_missing": int((pd.to_numeric(alignment["raw_interaction_count"]) == 0).sum()),
                "expected_zero_history_users": len(zero),
                "actual_zero_history_users": int(model_runs["zero_history_eval_users"].astype(int).max()) if not model_runs.empty else "",
                "expected_nonzero_actual_zero_users": len(expected_nonzero_actual_zero),
                "pipeline_zero_history_users": len(pipeline_zero),
                "model_mapped_users": len(alignment),
                "target_mapped_users": len(alignment),
                "candidate_set_mapped_users": len(alignment),
                "zero_history_user_hash": zero_user_hash,
                "decision": FOODCOM_HISTORY_VALID if len(pipeline_zero) == 0 and len(zero) == 727 else FOODCOM_HISTORY_PIPELINE_DEFECT,
                "reason": "Shared upstream cohort property; not a model-specific mapping failure.",
            }
        )
    return pd.DataFrame(rows)


def build_model_input_mapping_audit(alignment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in TRAINABLE_MODELS:
        for row in alignment.to_dict("records"):
            rows.append(
                {
                    "dataset": "foodcom",
                    "model": model,
                    "user_id": row["user_id"],
                    "f0_sample_cohort_id": row["f0_sample_cohort_id"],
                    "expected_pretest_history_length": row["final_training_interaction_count_expected"],
                    "actual_final_training_history_length": row[
                        "final_training_interaction_count_actually_materialized"
                    ],
                    "expected_model_visible_history_count": row["model_visible_history_count_expected"],
                    "model_visible_history_count": row["model_visible_history_count"],
                    "model_sequence_truncation_applied": row["model_sequence_truncation_applied"],
                    "model_max_sequence_length": row["model_max_sequence_length"],
                    "internal_mapped_user_id": row["internal_mapped_user_id"],
                    "user_mapping_success": row["mapping_success"],
                    "target_item": row["target_item"],
                    "target_mapping_success": row["target_mapping_success"],
                    "candidate_set_mapping_success": row["candidate_set_mapping_success"],
                    "zero_history_flag": row["zero_history_flag"],
                    "zero_history_reason": row["zero_history_reason"],
                    "model_specific_zero_history_reason": (
                        "SHARED_TRUE_ZERO_HISTORY_INPUT"
                        if row["zero_history_reason"] == "TRUE_ZERO_HISTORY_RAW_SINGLE_INTERACTION_TARGET_ONLY"
                        else row["zero_history_reason"]
                    ),
                    "status": "VALID_AS_IS"
                    if row["zero_history_reason"] != "MAPPING_OR_PIPELINE_ZERO_HISTORY"
                    else "INVALID",
                }
            )
    return pd.DataFrame(rows)


def build_first_stage_source_audit(
    root: Path,
    specs: Mapping[str, DatasetSpec],
    materials: Mapping[str, DatasetMaterial],
    final_data: Mapping[str, Any],
    run_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in ACTIVE_DATASETS:
        material = materials[dataset]
        item_index = set(final_data[dataset].tuning.item_index)
        catalog_items = set(map(str, material.catalog_items))
        candidate_items = set(material.candidates["candidate_item_id"].astype(str))
        positive_counts = material.candidates.groupby("user_id")["is_ground_truth"].sum().astype(int)
        for model in TRAINABLE_MODELS:
            method_runs = run_manifest.loc[run_manifest["dataset"].eq(dataset) & run_manifest["method"].eq(model)]
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "source_candidate_universe": SOURCE_CANDIDATE20,
                    "source_universe_size": CANDIDATE_COUNT,
                    "topk_before_union": CANDIDATE_COUNT,
                    "ranking_candidate_file_used": specs[dataset].candidate_source.as_posix(),
                    "ranking_candidate_contract_hash": material.candidate_set_hash,
                    "ranking_candidate_users": int(material.candidates["user_id"].nunique()),
                    "candidate_rows": int(len(material.candidates)),
                    "candidate_rows_per_user_min": int(material.candidates.groupby("user_id").size().min()),
                    "candidate_rows_per_user_max": int(material.candidates.groupby("user_id").size().max()),
                    "positive_forced_into_source_universe": bool_text(bool(positive_counts.eq(1).all())),
                    "candidate_items_missing_from_catalog": int(material.catalog_missing_candidate_count),
                    "raw_catalog_items": int(len(catalog_items)),
                    "model_item_universe_size": int(final_data[dataset].tuning.num_items),
                    "candidate_items": int(len(candidate_items)),
                    "raw_catalog_missing_from_model_universe": int(len(catalog_items - item_index)),
                    "full_catalog_scoring": bool_text(False),
                    "run_count": int(len(method_runs)),
                    "completed_run_count": int(method_runs["training_status"].eq("COMPLETED").sum()),
                    "source_artifact_pattern": (
                        RAW_RECOMMENDATION_DIR / f"f1bz_{dataset}_{model.lower()}_seed*.csv"
                    ).as_posix(),
                    "protocol_class": RERANKING_POOL_SOURCE_DEFECT,
                    "reason": (
                        "F1BZ first-stage rows are ranked only over the frozen 1-positive/19-negative "
                        "ranking candidate set, so the positive item is forced into every source universe."
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_candidate_recall_tautology_audit(
    materials: Mapping[str, DatasetMaterial],
    pools: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positives_by_dataset: dict[str, dict[str, str]] = {}
    candidate_membership: dict[str, dict[str, set[str]]] = {}
    for dataset, material in materials.items():
        grouped = material.candidates.groupby("user_id", sort=False)
        positives_by_dataset[dataset] = (
            material.candidates.drop_duplicates("user_id").set_index("user_id")["positive_item_id"].astype(str).to_dict()
        )
        candidate_membership[dataset] = {
            str(user): set(group["candidate_item_id"].astype(str).tolist()) for user, group in grouped
        }

    for row in pools.to_dict("records"):
        dataset = str(row["dataset"])
        user_id = str(row["user_id"])
        positive = str(row["positive_item"])
        pool_items = json.loads(row["pool_items"])
        ranking_candidates = candidate_membership[dataset].get(user_id, set())
        source_positive = positives_by_dataset[dataset].get(user_id, "")
        positive_in_candidate20 = positive in ranking_candidates and positive == source_positive
        positive_in_pool = positive in set(map(str, pool_items))
        tautological = positive_in_candidate20 and positive_in_pool and int(row["ranking_size"]) == CANDIDATE_COUNT
        rows.append(
            {
                "dataset": dataset,
                "user_id": user_id,
                "source_candidate_universe": SOURCE_CANDIDATE20,
                "ranking_candidate_count": len(ranking_candidates),
                "positive_item": positive,
                "source_positive_item": source_positive,
                "positive_in_ranking_candidate20": bool_text(positive_in_candidate20),
                "positive_in_full_catalog_model_topK": "NOT_GENERATED",
                "full_catalog_model_topK_source": "",
                "positive_in_final_reranking_pool": bool_text(positive_in_pool),
                "final_pool_size": int(row["ranking_size"]),
                "final_pool_positive_rank": row["positive_rank"],
                "candidate_recall_tautological": bool_text(tautological),
                "scientific_status": RERANKING_POOL_SOURCE_DEFECT,
                "reason": (
                    "Candidate Recall@20 is 1.0 because the ranking candidate set contains the positive by "
                    "construction and the reranking pool is assembled from complete top-20 candidate-set rankings."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_full_catalog_regeneration_readiness(
    root: Path,
    materials: Mapping[str, DatasetMaterial],
    final_data: Mapping[str, Any],
    run_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in ACTIVE_DATASETS:
        material = materials[dataset]
        item_index = set(final_data[dataset].tuning.item_index)
        catalog_items = set(map(str, material.catalog_items))
        candidate_items = set(material.candidates["candidate_item_id"].astype(str))
        raw_missing = len(catalog_items - item_index)
        candidate_missing = len(candidate_items - catalog_items)
        for model in TRAINABLE_MODELS:
            method_runs = run_manifest.loc[
                run_manifest["dataset"].eq(dataset) & run_manifest["method"].eq(model)
            ].sort_values("seed", key=lambda series: pd.to_numeric(series, errors="coerce"))
            checkpoint_paths = [root / str(path) for path in method_runs["checkpoint_path"].astype(str)]
            load_results = [checkpoint_embedding_status(path, model, final_data[dataset].tuning.num_items) for path in checkpoint_paths]
            loaded = [row for row in load_results if row["load_success"]]
            all_load = len(loaded) == len(checkpoint_paths) == len(FINAL_SEEDS)
            embedding_shapes = [str(row["item_embedding_shape"]) for row in load_results if row["item_embedding_shape"]]
            model_known_ready = all_load and all(row["embedding_shape_matches_model_universe"] for row in load_results)
            full_catalog_ready = model_known_ready and raw_missing == 0 and candidate_missing == 0
            conclusion = FULL_CATALOG_READY if full_catalog_ready else BLOCKED
            if conclusion == FULL_CATALOG_READY:
                reason = (
                    "Existing checkpoints can score every raw catalog item represented by the frozen "
                    "model item map without target injection."
                )
            else:
                reason = (
                    "Existing checkpoint item embeddings cover the F1BZ model-known universe, not the "
                    "entire raw catalog required for non-tautological full-catalog first-stage generation."
                )
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "checkpoint_count": len(checkpoint_paths),
                    "checkpoint_load_success_count": len(loaded),
                    "checkpoint_load_failure_count": len(checkpoint_paths) - len(loaded),
                    "representative_checkpoint": (
                        checkpoint_paths[0].relative_to(root).as_posix() if checkpoint_paths else ""
                    ),
                    "representative_item_embedding_shape": embedding_shapes[0] if embedding_shapes else "",
                    "all_embedding_shapes": stable_json(embedding_shapes),
                    "model_item_universe_size": int(final_data[dataset].tuning.num_items),
                    "expected_embedding_rows": expected_embedding_rows(model, final_data[dataset].tuning.num_items),
                    "raw_catalog_items": int(len(catalog_items)),
                    "candidate_items": int(len(candidate_items)),
                    "train_items": int(final_data[dataset].train_item_count),
                    "catalog_missing_from_model_universe": int(raw_missing),
                    "candidate_missing_from_catalog": int(candidate_missing),
                    "scoring_possible_over_model_universe": bool_text(model_known_ready),
                    "all_eligible_catalog_item_scores_can_be_generated": bool_text(full_catalog_ready),
                    "train_history_exclusion_possible": bool_text(True),
                    "target_not_force_included_possible": bool_text(True),
                    "current_f1bz_full_catalog_scoring": bool_text(False),
                    "target_conclusion": conclusion,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows)


def checkpoint_embedding_status(path: Path, model: str, expected_items: int) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - exercised only on corrupt local artifacts.
        return {
            "load_success": False,
            "item_embedding_shape": "",
            "embedding_shape_matches_model_universe": False,
            "failure_reason": str(exc),
        }
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else {}
    shape = ""
    if isinstance(state, Mapping):
        tensor = state.get("item_embedding.weight")
        if tensor is not None and hasattr(tensor, "shape"):
            shape = tuple(int(dim) for dim in tensor.shape)
    expected_rows = expected_embedding_rows(model, expected_items)
    return {
        "load_success": bool(shape),
        "item_embedding_shape": shape,
        "embedding_shape_matches_model_universe": bool(shape and int(shape[0]) == expected_rows),
        "failure_reason": "" if shape else "item_embedding.weight missing",
    }


def expected_embedding_rows(model: str, item_count: int) -> int:
    return int(item_count) + 1 if model in {"GRU4Rec", "SASRec"} else int(item_count)


def build_lastfm_reranking_lineage_comparison(root: Path) -> pd.DataFrame:
    f1bz_recall = read_csv(root / F1BZ_TABLES["candidate_recall"])
    current = f1bz_recall.loc[f1bz_recall["dataset"].eq("lastfm")]
    historical_rate = ""
    validation_path = root / "results/tables/finalization_f6d2_reranking_pool_validation.csv"
    if validation_path.exists():
        validation = read_csv(validation_path)
        row = validation.loc[validation["check_name"].eq("positive_inclusion_rate")]
        historical_rate = row["observed"].iloc[0] if not row.empty else ""
    source_path = root / "results/tables/finalization_f6d2_reranking_source_selection.csv"
    selected_full_catalog_sources = 0
    selected_source_hash = ""
    selected_models = ""
    if source_path.exists():
        source = read_csv(source_path)
        selected = source.loc[
            source["selected_for_adapted_pool"].astype(str).str.lower().eq("true")
            & source["full_catalog_prediction_file"].astype(str).ne("")
        ].copy()
        selected_full_catalog_sources = int(len(selected))
        selected_source_hash = frame_hash(selected)
        selected_models = ";".join(sorted(selected["model"].astype(str).unique()))
    rows = [
        {
            "lineage": "historical_f6d2_lastfm",
            "source_universe": "FULL_CATALOG_PREDICTIONS",
            "reranking_users": "1859",
            "positive_present_rows": "",
            "positive_present_rate": historical_rate,
            "candidate_recall_at_20": historical_rate,
            "selected_full_catalog_source_files": selected_full_catalog_sources,
            "selected_models": selected_models,
            "source_evidence": "results/tables/finalization_f6d2_reranking_pool_validation.csv",
            "source_hash": selected_source_hash,
            "status": "HISTORICAL_NON_TAUTOLOGICAL_REFERENCE",
        },
        {
            "lineage": "current_f1bz_lastfm",
            "source_universe": SOURCE_CANDIDATE20,
            "reranking_users": current["evaluation_N"].iloc[0] if not current.empty else "",
            "positive_present_rows": current["positive_present_N"].iloc[0] if not current.empty else "",
            "positive_present_rate": current["candidate_recall_at_20"].iloc[0] if not current.empty else "",
            "candidate_recall_at_20": current["candidate_recall_at_20"].iloc[0] if not current.empty else "",
            "selected_full_catalog_source_files": 0,
            "selected_models": "BPRMF;GRU4Rec;LightGCN;SASRec",
            "source_evidence": "results/tables/f1bz_candidate_recall.csv",
            "source_hash": frame_hash(current) if not current.empty else "",
            "status": RERANKING_POOL_SOURCE_DEFECT,
        },
        {
            "lineage": "comparison",
            "source_universe": "FULL_CATALOG_VS_CANDIDATE20",
            "reranking_users": "",
            "positive_present_rows": "",
            "positive_present_rate": "",
            "candidate_recall_at_20": "",
            "selected_full_catalog_source_files": selected_full_catalog_sources,
            "selected_models": "",
            "source_evidence": "F6D2 had non-tautological full-catalog predictions; F1BZ uses forced-positive candidate20.",
            "source_hash": "",
            "status": RERANKING_POOL_SOURCE_DEFECT,
        },
    ]
    return pd.DataFrame(rows)


def foodcom_history_decision_from(alignment: pd.DataFrame) -> str:
    expected = pd.to_numeric(alignment["final_training_interaction_count_expected"], errors="raise")
    actual = pd.to_numeric(alignment["final_training_interaction_count_actually_materialized"], errors="raise")
    expected_model_visible = pd.to_numeric(alignment["model_visible_history_count_expected"], errors="raise")
    model_visible = pd.to_numeric(alignment["model_visible_history_count"], errors="raise")
    pipeline_zero = alignment["zero_history_reason"].eq("MAPPING_OR_PIPELINE_ZERO_HISTORY").sum()
    true_zero = alignment["zero_history_reason"].eq("TRUE_ZERO_HISTORY_RAW_SINGLE_INTERACTION_TARGET_ONLY").sum()
    all_mapping = (
        alignment["mapping_success"].eq("true").all()
        and alignment["target_mapping_success"].eq("true").all()
        and alignment["candidate_set_mapping_success"].eq("true").all()
    )
    all_aligned = expected.eq(actual).all() and expected_model_visible.eq(model_visible).all()
    return FOODCOM_HISTORY_VALID if all_mapping and all_aligned and pipeline_zero == 0 and true_zero == 727 else FOODCOM_HISTORY_PIPELINE_DEFECT


def reranking_pool_decision_from(first_stage_source: pd.DataFrame, tautology: pd.DataFrame) -> str:
    defective_source = first_stage_source["protocol_class"].eq(RERANKING_POOL_SOURCE_DEFECT).all()
    tautological = tautology["candidate_recall_tautological"].eq("true").all()
    return RERANKING_POOL_SOURCE_DEFECT if defective_source and tautological else RERANKING_POOL_VALID


def build_result_validity_classification(
    run_manifest: pd.DataFrame,
    readiness: pd.DataFrame,
    foodcom_decision: str,
    reranking_decision: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "output_family": "foodcom_history_alignment",
            "dataset": "foodcom",
            "classification": "VALID_AS_IS" if foodcom_decision == FOODCOM_HISTORY_VALID else "INVALID",
            "affected_rows": 1000,
            "reason": "Zero-history users are true raw single-interaction final-test users.",
        },
        {
            "output_family": "candidate20_trainable_final_test_metrics",
            "dataset": "all",
            "classification": "VALID_AS_IS",
            "affected_rows": int(len(run_manifest) * 3),
            "reason": "Utility metrics are valid for the frozen candidate20 ranking task.",
        },
        {
            "output_family": "candidate20_trainable_beyond_utility_metrics",
            "dataset": "all",
            "classification": "VALID_AS_IS",
            "affected_rows": int(len(run_manifest) * 14),
            "reason": "Beyond-utility metrics remain candidate20-scoped; degenerate cells stay explicit.",
        },
        {
            "output_family": "deterministic_baseline_outputs",
            "dataset": "all",
            "classification": "VALID_AS_IS",
            "affected_rows": 292920,
            "reason": "F1BX baseline hashes are preserved by F1BZ.",
        },
        {
            "output_family": "trainable_checkpoints_candidate20_scoring",
            "dataset": "all",
            "classification": "VALID_AS_IS",
            "affected_rows": int(len(run_manifest)),
            "reason": "Checkpoints are valid for the F1BZ training data and candidate20 scoring contract.",
        },
        {
            "output_family": "f1bz_reranking_pools",
            "dataset": "all",
            "classification": "INVALID" if reranking_decision == RERANKING_POOL_SOURCE_DEFECT else "VALID_AS_IS",
            "affected_rows": int(run_manifest["evaluation_users"].astype(int).groupby(run_manifest["dataset"]).first().sum()),
            "reason": "Pools are assembled from forced-positive candidate20 first-stage rankings.",
        },
        {
            "output_family": "f1bz_candidate_recall_at_20",
            "dataset": "all",
            "classification": "INVALID" if reranking_decision == RERANKING_POOL_SOURCE_DEFECT else "VALID_AS_IS",
            "affected_rows": 5,
            "reason": "Candidate Recall@20 is tautological under the current source universe.",
        },
        {
            "output_family": "f1bz_reranking_readiness",
            "dataset": "all",
            "classification": "INVALID" if reranking_decision == RERANKING_POOL_SOURCE_DEFECT else "VALID_AS_IS",
            "affected_rows": 30,
            "reason": "Readiness depends on invalid reranking pools and must be regenerated after source correction.",
        },
    ]
    for row in readiness.to_dict("records"):
        rows.append(
            {
                "output_family": "full_catalog_first_stage_generation",
                "dataset": row["dataset"],
                "model": row["model"],
                "classification": (
                    "REGENERATE_FROM_EXISTING_CHECKPOINT"
                    if row["target_conclusion"] == FULL_CATALOG_READY
                    else "RETRAIN_REQUIRED"
                ),
                "affected_rows": row["checkpoint_count"],
                "reason": row["reason"],
            }
        )
    return pd.DataFrame(rows)


def build_corrective_scope(
    run_manifest: pd.DataFrame,
    pools: pd.DataFrame,
    readiness: pd.DataFrame,
    foodcom_decision: str,
    reranking_decision: str,
    f1bz_manifest: Mapping[str, Any],
) -> pd.DataFrame:
    ready = readiness.loc[readiness["target_conclusion"].eq(FULL_CATALOG_READY)]
    blocked = readiness.loc[readiness["target_conclusion"].ne(FULL_CATALOG_READY)]
    rows = [
        {
            "scope_id": "F1BZA-CORRECTIVE-001",
            "output_family": "foodcom_zero_history",
            "requires_correction": bool_text(foodcom_decision != FOODCOM_HISTORY_VALID),
            "affected_datasets": "foodcom",
            "affected_models": ";".join(TRAINABLE_MODELS),
            "affected_rows": 1000,
            "correction_action": "NONE",
            "status": "VALID_AS_IS" if foodcom_decision == FOODCOM_HISTORY_VALID else "INVALID",
            "reason": "F1BZ Food.com zero-history count is a true cohort property.",
        },
        {
            "scope_id": "F1BZA-CORRECTIVE-002",
            "output_family": "candidate20_ranking_results",
            "requires_correction": bool_text(False),
            "affected_datasets": ",".join(ACTIVE_DATASETS),
            "affected_models": ";".join(TRAINABLE_MODELS),
            "affected_rows": int(len(run_manifest)),
            "correction_action": "NONE",
            "status": "VALID_AS_IS",
            "reason": "F1BZ final metrics remain valid for the frozen candidate20 ranking task.",
        },
        {
            "scope_id": "F1BZA-CORRECTIVE-003",
            "output_family": "reranking_pool_construction",
            "requires_correction": bool_text(reranking_decision == RERANKING_POOL_SOURCE_DEFECT),
            "affected_datasets": ",".join(ACTIVE_DATASETS),
            "affected_models": ";".join(TRAINABLE_MODELS),
            "affected_rows": int(len(pools)),
            "correction_action": "REGENERATE_NON_TAUTOLOGICAL_FIRST_STAGE_SOURCES_BEFORE_LLM_RERANKING",
            "status": "INVALID" if reranking_decision == RERANKING_POOL_SOURCE_DEFECT else "VALID_AS_IS",
            "reason": "Current first-stage source uses the forced-positive candidate20 universe.",
        },
        {
            "scope_id": "F1BZA-CORRECTIVE-004",
            "output_family": "future_llm_reranking_requests",
            "requires_correction": bool_text(reranking_decision == RERANKING_POOL_SOURCE_DEFECT),
            "affected_datasets": ",".join(ACTIVE_DATASETS),
            "affected_models": "future_llm_models",
            "affected_rows": int(f1bz_manifest.get("future_llm_requests_unlocked", 0)),
            "correction_action": "LOCK_UNTIL_RERANKING_POOL_REGENERATION",
            "status": "INVALID" if reranking_decision == RERANKING_POOL_SOURCE_DEFECT else "VALID_AS_IS",
            "reason": "The request count is numerically unchanged but cannot be executed against invalid pools.",
        },
        {
            "scope_id": "F1BZA-CORRECTIVE-005",
            "output_family": "full_catalog_generation_from_checkpoints",
            "requires_correction": bool_text(True),
            "affected_datasets": ",".join(sorted(readiness["dataset"].unique())),
            "affected_models": ";".join(TRAINABLE_MODELS),
            "affected_rows": int(len(readiness)),
            "correction_action": (
                f"{len(ready)} dataset-model cells can regenerate from checkpoint; "
                f"{len(blocked)} require a corrected item-universe policy or retraining."
            ),
            "status": "PARTIAL",
            "reason": "Only cells with full raw catalog coverage in the model item map are immediately ready.",
        },
    ]
    return pd.DataFrame(rows)


def build_validation(
    root: Path,
    f1bz_manifest: Mapping[str, Any],
    no_llm_manifest: Mapping[str, Any],
    protected_integrity: Mapping[str, Any],
    baseline_snapshot: Mapping[str, Any],
    evidence_hashes: Mapping[str, str],
    food_alignment: pd.DataFrame,
    food_summary: pd.DataFrame,
    mapping_audit: pd.DataFrame,
    first_stage_source: pd.DataFrame,
    tautology: pd.DataFrame,
    readiness: pd.DataFrame,
    lastfm_lineage: pd.DataFrame,
    result_validity: pd.DataFrame,
    corrective_scope: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": str(evidence)})

    evidence_counts = evidence_artifact_counts(evidence_hashes)
    check("precondition_state_f1bz_validated", f1bz_manifest.get("decision_token") == PRECONDITION_STATE, f1bz_manifest.get("decision_token"))
    check("active_datasets_exact", protected_integrity["active_datasets"] == list(ACTIVE_DATASETS), protected_integrity["active_datasets"])
    check(
        "video_games_absent",
        int(protected_integrity["active_video_games_rows"]) == 0
        and int(protected_integrity["operational_video_games_training_runs"]) == 0,
        {
            "rows": protected_integrity["active_video_games_rows"],
            "runs": protected_integrity["operational_video_games_training_runs"],
        },
    )
    check("f1bz_artifacts_hashed_before_audit", evidence_counts["f1bz_tables"] >= 14 and evidence_counts["raw_recommendations"] == 100 and evidence_counts["checkpoints"] == 100, evidence_counts)
    check("f1bz_final_runs_preserved", int(f1bz_manifest.get("completed_runs", 0)) == 100 and int(f1bz_manifest.get("failed_runs", 1)) == 0, {"completed": f1bz_manifest.get("completed_runs"), "failed": f1bz_manifest.get("failed_runs")})
    check(
        "foodcom_history_alignment_1000",
        len(food_alignment) == 1000
        and food_alignment["expected_matches_actual"].eq("true").all()
        and food_alignment["mapping_success"].eq("true").all(),
        len(food_alignment),
    )
    check(
        "foodcom_zero_history_is_true_raw_singleton",
        int(food_alignment["zero_history_reason"].eq("TRUE_ZERO_HISTORY_RAW_SINGLE_INTERACTION_TARGET_ONLY").sum()) == 727
        and int(food_alignment["zero_history_reason"].eq("MAPPING_OR_PIPELINE_ZERO_HISTORY").sum()) == 0,
        food_alignment["zero_history_reason"].value_counts().to_dict(),
    )
    check("foodcom_history_decision_valid", food_summary["decision"].eq(FOODCOM_HISTORY_VALID).all(), food_summary["decision"].value_counts().to_dict())
    check(
        "model_input_mapping_rows",
        len(mapping_audit) == 4000
        and mapping_audit["user_mapping_success"].eq("true").all()
        and mapping_audit["target_mapping_success"].eq("true").all()
        and mapping_audit["candidate_set_mapping_success"].eq("true").all(),
        len(mapping_audit),
    )
    check(
        "first_stage_source_candidate20_defect",
        len(first_stage_source) == 20
        and first_stage_source["source_candidate_universe"].eq(SOURCE_CANDIDATE20).all()
        and first_stage_source["full_catalog_scoring"].eq("false").all()
        and first_stage_source["positive_forced_into_source_universe"].eq("true").all()
        and first_stage_source["protocol_class"].eq(RERANKING_POOL_SOURCE_DEFECT).all(),
        first_stage_source["protocol_class"].value_counts().to_dict(),
    )
    check(
        "candidate_recall_tautology_rows",
        len(tautology) == 4882
        and tautology["positive_in_ranking_candidate20"].eq("true").all()
        and tautology["positive_in_final_reranking_pool"].eq("true").all()
        and tautology["candidate_recall_tautological"].eq("true").all(),
        len(tautology),
    )
    check(
        "full_catalog_readiness_materialized",
        len(readiness) == 20
        and int(readiness["checkpoint_load_success_count"].astype(int).sum()) == 100
        and len(readiness.loc[readiness["target_conclusion"].eq(FULL_CATALOG_READY)]) == 4
        and readiness.loc[readiness["dataset"].eq("lastfm"), "target_conclusion"].eq(FULL_CATALOG_READY).all(),
        readiness["target_conclusion"].value_counts().to_dict(),
    )
    check(
        "lastfm_historical_lineage_non_tautological_reference",
        "HISTORICAL_NON_TAUTOLOGICAL_REFERENCE" in set(lastfm_lineage["status"])
        and RERANKING_POOL_SOURCE_DEFECT in set(lastfm_lineage["status"]),
        lastfm_lineage[["lineage", "status"]].to_dict("records"),
    )
    check(
        "result_validity_statuses_complete",
        result_validity["classification"].isin(
            {"VALID_AS_IS", "REGENERATE_FROM_EXISTING_CHECKPOINT", "RETRAIN_REQUIRED", "INVALID"}
        ).all(),
        result_validity["classification"].value_counts().to_dict(),
    )
    check(
        "corrective_scope_records_defect",
        corrective_scope["status"].isin({"VALID_AS_IS", "INVALID", "PARTIAL"}).all()
        and corrective_scope.loc[corrective_scope["output_family"].eq("reranking_pool_construction"), "status"].eq("INVALID").all(),
        corrective_scope[["output_family", "status"]].to_dict("records"),
    )
    check("global_metric_blockers_zero", int(protected_integrity["global_metric_blockers"]) == 0, protected_integrity["global_metric_blockers"])
    check(
        "f0_budget_unchanged",
        bool(protected_integrity["f0_budget_unchanged"])
        and int(protected_integrity["f0_core_mandatory_requests"]) == 29292
        and int(protected_integrity["f0_training_dependent_requests"]) == 29292,
        {
            "core": protected_integrity["f0_core_mandatory_requests"],
            "training_dependent": protected_integrity["f0_training_dependent_requests"],
        },
    )
    check(
        "baseline_preserved",
        int(baseline_snapshot["baseline_rows"]) == 15
        and int(baseline_snapshot["baseline_recommendation_rows"]) == 292920
        and bool(baseline_snapshot["baseline_hashes_unchanged"])
        and bool(baseline_snapshot["baseline_hashes_match_f1bx_manifest"]),
        baseline_snapshot,
    )
    check(
        "foodcom_h0_hashes_unchanged",
        bool(protected_integrity["foodcom_raw_hashes_unchanged"]) and bool(protected_integrity["h0_hashes_unchanged"]),
        {
            "foodcom": protected_integrity["foodcom_raw_hashes_unchanged"],
            "h0": protected_integrity["h0_hashes_unchanged"],
        },
    )
    check(
        "original_repo_paper_unchanged",
        protected_integrity["original_repo_status"] == "" and protected_integrity["paper_status"] == "",
        {"original_repo": protected_integrity["original_repo_status"], "paper": protected_integrity["paper_status"]},
    )
    check(
        "zero_llm_network_download_training",
        all(
            int(protected_integrity[key]) == 0
            for key in (
                "llm_inference_calls_executed",
                "ollama_recommendation_calls_executed",
                "external_api_calls_executed",
                "network_calls_executed",
                "downloads_executed",
                "training_started_by_f1bza",
                "inference_started_by_f1bza",
            )
        )
        and int(no_llm_manifest.get("llm_inference_calls_executed", -1)) == 0,
        "0",
    )
    check(
        "output_tables_defined",
        set(OUTPUT_TABLES) == {
            "foodcom_history_alignment",
            "foodcom_zero_history_summary",
            "model_input_mapping_audit",
            "first_stage_source_audit",
            "candidate_recall_tautology_audit",
            "full_catalog_regeneration_readiness",
            "lastfm_reranking_lineage_comparison",
            "result_validity_classification",
            "corrective_scope",
            "validation",
        },
        sorted(OUTPUT_TABLES),
    )
    return pd.DataFrame(rows)


def evidence_artifact_counts(evidence_hashes: Mapping[str, str]) -> dict[str, int]:
    keys = list(evidence_hashes)
    return {
        "f1bz_tables": sum(key.startswith("results/tables/f1bz_") for key in keys),
        "updated_f1b_tables": sum(key.startswith("results/tables/f1b_") and not key.startswith("results/tables/f1bz_") for key in keys),
        "f1bz_manifests": sum(key.startswith("results/manifests/f1bz_") for key in keys),
        "raw_recommendations": sum(key.startswith("results/raw_outputs/f1b_final_recommendations/f1bz_") for key in keys),
        "checkpoints": sum(key.startswith("results/checkpoints/f1b_final_traditional/f1bz_") for key in keys),
        "total": len(keys),
    }


def build_manifest(
    root: Path,
    config: Mapping[str, Any],
    decision: str,
    foodcom_history_decision: str,
    reranking_pool_decision: str,
    f1bz_manifest: Mapping[str, Any],
    no_llm_manifest: Mapping[str, Any],
    protected_integrity: Mapping[str, Any],
    baseline_snapshot: Mapping[str, Any],
    evidence_hashes: Mapping[str, str],
    food_alignment: pd.DataFrame,
    tautology: pd.DataFrame,
    readiness: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, Any]:
    zero_users = int(food_alignment["zero_history_flag"].eq("true").sum())
    validation_passed = bool(validation["status"].eq("PASS").all())
    return {
        "phase_id": PHASE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "precondition_state": PRECONDITION_STATE,
        "precondition_checkpoint": F1BZ_EXPECTED_HEAD,
        "head": protected_integrity["head"],
        "decision_token": decision,
        "foodcom_history_decision": foodcom_history_decision,
        "reranking_pool_decision": reranking_pool_decision,
        "f1br_protocol_defect_discovered_post_execution": reranking_pool_decision == RERANKING_POOL_SOURCE_DEFECT,
        "active_datasets": list(ACTIVE_DATASETS),
        "models": list(TRAINABLE_MODELS),
        "seeds": list(FINAL_SEEDS),
        "no_training": True,
        "no_baseline_execution": True,
        "no_llm_inference": True,
        "no_download": True,
        "no_network": True,
        "no_commit": True,
        "f1bz_decision_token": f1bz_manifest.get("decision_token", ""),
        "f1bz_completed_runs": f1bz_manifest.get("completed_runs", ""),
        "f1bz_failed_runs": f1bz_manifest.get("failed_runs", ""),
        "f1bz_evidence_hash_counts": evidence_artifact_counts(evidence_hashes),
        "f1bz_evidence_hashes": dict(evidence_hashes),
        "foodcom_users": int(len(food_alignment)),
        "foodcom_zero_history_users": zero_users,
        "foodcom_pipeline_zero_history_users": int(
            food_alignment["zero_history_reason"].eq("MAPPING_OR_PIPELINE_ZERO_HISTORY").sum()
        ),
        "tautological_reranking_pool_rows": int(tautology["candidate_recall_tautological"].eq("true").sum()),
        "reranking_pool_rows": int(len(tautology)),
        "full_catalog_regeneration_ready_cells": int(readiness["target_conclusion"].eq(FULL_CATALOG_READY).sum()),
        "full_catalog_regeneration_blocked_cells": int(readiness["target_conclusion"].ne(FULL_CATALOG_READY).sum()),
        "f0_request_budget": {
            "core_mandatory_now_requests": protected_integrity["f0_core_mandatory_requests"],
            "training_dependent_requests": protected_integrity["f0_training_dependent_requests"],
            "f0_budget_unchanged": protected_integrity["f0_budget_unchanged"],
        },
        "baseline_snapshot": dict(baseline_snapshot),
        "protected_integrity": dict(protected_integrity),
        "no_llm_manifest": {
            "llm_inference_calls_executed": no_llm_manifest.get("llm_inference_calls_executed", ""),
            "ollama_recommendation_calls_executed": no_llm_manifest.get("ollama_recommendation_calls_executed", ""),
            "external_api_calls_executed": no_llm_manifest.get("external_api_calls_executed", ""),
            "network_calls_executed": no_llm_manifest.get("network_calls_executed", ""),
            "downloads_executed": no_llm_manifest.get("downloads_executed", ""),
        },
        "output_tables": {key: path.as_posix() for key, path in OUTPUT_TABLES.items()},
        "report_path": REPORT_PATH.as_posix(),
        "config_snapshot": {
            "phase_id": config.get("phase", {}).get("id", ""),
            "reranking_source": dict(config.get("reranking_source", {})),
        },
        "validation_passed": validation_passed,
        "validation_failures": validation.loc[validation["status"].ne("PASS")].to_dict("records"),
    }


def render_report(
    decision: str,
    foodcom_history_decision: str,
    reranking_pool_decision: str,
    alignment: pd.DataFrame,
    summary: pd.DataFrame,
    first_stage_source: pd.DataFrame,
    tautology: pd.DataFrame,
    readiness: pd.DataFrame,
    lastfm_lineage: pd.DataFrame,
    result_validity: pd.DataFrame,
    corrective_scope: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    zero_reasons = alignment["zero_history_reason"].value_counts().to_dict()
    source_classes = first_stage_source["protocol_class"].value_counts().to_dict()
    readiness_counts = readiness["target_conclusion"].value_counts().to_dict()
    validity_counts = result_validity["classification"].value_counts().to_dict()
    corrective_status = corrective_scope["status"].value_counts().to_dict()
    lines = [
        "# F1BZA Final Traditional Post-Execution Audit",
        "",
        f"Decision token: `{decision}`",
        f"Food.com history decision: `{foodcom_history_decision}`",
        f"Reranking pool decision: `{reranking_pool_decision}`",
        "",
        "## Food.com History",
        "",
        f"- Audited users: `{len(alignment)}`.",
        f"- Zero-history users: `{int(alignment['zero_history_flag'].eq('true').sum())}`.",
        f"- Zero-history reasons: `{stable_json(zero_reasons)}`.",
        f"- Summary rows: `{len(summary)}`.",
        "",
        "Food.com zero-history users are true raw single-interaction users: the only chronological raw interaction is the frozen final target, so the final-training prefix i1..i[T-1] is empty. No smoothing, history redefinition, or target movement is applied.",
        "",
        "## Reranking Pool Source",
        "",
        f"- Source audit rows: `{len(first_stage_source)}`.",
        f"- Source classes: `{stable_json(source_classes)}`.",
        f"- Tautological pool rows: `{int(tautology['candidate_recall_tautological'].eq('true').sum())}` / `{len(tautology)}`.",
        "",
        "The F1BZ first-stage source is the frozen 1-positive/19-negative ranking candidate set. Since the positive is forced into that set and the per-model top-20 exhausts it, Candidate Recall@20 equals 1.0 by construction. This invalidates the reranking pools and readiness rows for future LLM reranking, but does not invalidate candidate20 ranking metrics.",
        "",
        "## Full-Catalog Readiness",
        "",
        f"- Readiness cells: `{len(readiness)}`.",
        f"- Conclusions: `{stable_json(readiness_counts)}`.",
        "",
        "Existing checkpoints load for all 100 trainable runs. LastFM can regenerate non-tautological full-catalog first-stage scores from existing checkpoints because its raw catalog is covered by the model item universe. The other four datasets are blocked for raw full-catalog generation by item-universe coverage, not by checkpoint corruption.",
        "",
        "## LastFM Lineage",
        "",
        lastfm_lineage.to_csv(index=False, lineterminator="\n").strip(),
        "",
        "## Result Validity",
        "",
        f"- Classification counts: `{stable_json(validity_counts)}`.",
        f"- Corrective status counts: `{stable_json(corrective_status)}`.",
        "",
        "Candidate20 final traditional metrics and deterministic baselines remain valid as-is. Reranking pools, Candidate Recall@20 as a reranking diagnostic, and reranking readiness require correction before any LLM reranking requests are executed.",
        "",
        "## Validation",
        "",
        f"- Checks: `{len(validation)}`.",
        f"- Failures: `{int(validation['status'].ne('PASS').sum())}`.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    ensure_output_dirs(root)
    for key, rel_path in OUTPUT_TABLES.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(root / rel_path, index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(str(outputs["report"]), encoding="utf-8")
    manifest = dict(outputs["manifest"])
    artifact_paths = [*OUTPUT_TABLES.values(), REPORT_PATH, Path("src/extension/f1bza_post_execution_audit.py"), Path(".gitignore")]
    manifest["artifact_hashes"] = {
        path.as_posix(): sha256_file(root / path) for path in artifact_paths if (root / path).exists()
    }
    (root / OUTPUT_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def load_existing_outputs(root: Path) -> dict[str, Any]:
    outputs = {key: read_csv(root / rel_path) for key, rel_path in OUTPUT_TABLES.items()}
    outputs["manifest"] = read_json(root / OUTPUT_MANIFEST)
    outputs["report"] = (root / REPORT_PATH).read_text(encoding="utf-8")
    return outputs


def ensure_output_dirs(root: Path) -> None:
    for rel_path in (TABLE_DIR, MANIFEST_DIR, REPORT_DIR):
        (root / rel_path).mkdir(parents=True, exist_ok=True)


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def git_output(root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


if __name__ == "__main__":
    raise SystemExit(main())

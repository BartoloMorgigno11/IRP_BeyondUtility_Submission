"""Materialize the current operational extension cell universe.

This D1-compatible generator keeps the original four datasets plus Food.com as
the only active dataset set. It is offline by construction: no model inference,
training, downloads, or raw-data reconstruction are performed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.audit.common_hashing import sha256_file_or_empty
from src.extension.e0_protocol import (
    CANDIDATE_COUNT,
    HISTORY_LENGTHS,
    METRIC_K,
    PARSER_CONTRACT_ID,
    PROMPT_BASE,
    PROMPT_CONTRACT_BASE_ID,
    PROMPT_CONTRACT_LLMRANK_ID,
    PROMPT_LLMRANK,
    sha256_json,
    stable_json,
)
from src.utils.config import find_project_root


SCHEMA_VERSION = "extension_cell_universe_v2_foodcom_active"
GENERATOR_ID = "src.extension.e2a_cell_universe"
DECISION_TOKEN = "EXTENSION_DATASET_FOODCOM_SELECTED_FOR_EXECUTION"

STATUS_COMPUTED = "COMPUTED"
STATUS_EXISTING_REUSABLE = "EXISTING_REUSABLE"
STATUS_TO_INFER = "TO_INFER"
STATUS_TO_TRAIN = "TO_TRAIN"
STATUS_TO_COMPUTE_OFFLINE = "TO_COMPUTE_OFFLINE"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUS_BLOCKED = "BLOCKED"

ALLOWED_STATUSES = {
    STATUS_COMPUTED,
    STATUS_EXISTING_REUSABLE,
    STATUS_TO_INFER,
    STATUS_TO_TRAIN,
    STATUS_TO_COMPUTE_OFFLINE,
    STATUS_NOT_APPLICABLE,
    STATUS_BLOCKED,
}

LOCAL_LLMS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
GEMMA_MODEL = "gemma2:9b"
ORIGINAL_DATASETS = ("beauty", "sports", "ml1m", "lastfm")
FIFTH_DATASET = "foodcom"
ACTIVE_DATASETS = (*ORIGINAL_DATASETS, FIFTH_DATASET)
GEMMA_CANDDIF_DATASETS = ("beauty", "sports", "lastfm")
PROMPT_STRATEGIES = (PROMPT_BASE, PROMPT_LLMRANK)
TRAINABLE_MODELS = ("BPRMF", "GRU4Rec", "SASRec", "LightGCN")
OFFLINE_BASELINES = ("Random", "MostPop", "BM25-like")
TRADITIONAL_MODELS = (*OFFLINE_BASELINES, *TRAINABLE_MODELS)
GLOBAL_POLICY_METRICS = ("popreo_at_5", "dpd_at_5", "ks_sample_representativeness")

ORIGINAL_COHORT_SIZES = {
    "beauty": 990,
    "sports": 892,
    "ml1m": 1000,
    "lastfm": 1000,
}

SOURCE_ARTIFACTS = (
    "src/extension/e2a_cell_universe.py",
    "tests/test_extension_e2a_cell_universe.py",
    "src/extension/d0a_foodcom_material_feasibility.py",
    "src/extension/d0b_final_fifth_dataset_selection.py",
    "results/tables/d0a_foodcom_primary1000_feasibility.csv",
    "results/tables/d0a_foodcom_history200_feasibility.csv",
    "results/tables/d0b_final_selection_decision.csv",
    "results/manifests/d0b_no_inference_manifest.json",
    "results/manifests/extension_h0_no_inference_manifest.json",
    "results/tables/extension_h0_72_cell_history_matrix.csv",
)

RAW_COLUMNS = (
    "raw_experiment_id",
    "required_for_claim",
    "claim_family",
    "experiment_family",
    "dataset",
    "task",
    "model_name",
    "model_family",
    "prompt_strategy",
    "prompt_contract_id",
    "candidate_set_id",
    "candidate_pool_size",
    "cohort_id",
    "cohort_size",
    "history_length",
    "candidate_order_condition",
    "profile_condition",
    "source_model_or_ranker",
    "parser_contract",
    "requires_llm_inference",
    "requires_training",
    "request_count",
    "dependency_ids",
    "existing_source",
    "status",
    "blocking_reason",
    "evidence_class",
    "planned_phase",
    "scenario",
    "execution_budget_class",
)

RESULT_COLUMNS = (
    "cell_id",
    "raw_experiment_id",
    "required_for_claim",
    "claim_family",
    "experiment_family",
    "dataset",
    "task",
    "model_name",
    "model_family",
    "prompt_strategy",
    "prompt_contract_id",
    "candidate_set_id",
    "candidate_pool_size",
    "history_length",
    "candidate_order_condition",
    "profile_condition",
    "seed_or_run",
    "parser_contract",
    "k",
    "metric_scope",
    "metric_id",
    "denominator_policy",
    "validity_policy",
    "requires_llm_inference",
    "requires_training",
    "can_compute_offline",
    "existing_source",
    "status",
    "blocking_reason",
    "evidence_class",
    "planned_phase",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the active extension cell universe.")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    outputs = generate_universe(root)
    validation = validate_universe(outputs["raw"], outputs["cells"])
    if args.validate_only:
        print(stable_json(validation))
        return 0 if validation["passed"] else 1
    write_outputs(root, outputs, validation)
    print(f"raw_experiments={len(outputs['raw'])}")
    print(f"result_cells={len(outputs['cells'])}")
    print(f"validation_passed={validation['passed']}")
    return 0 if validation["passed"] else 1


def generate_universe(root: Path | None = None) -> dict[str, pd.DataFrame]:
    raw = pd.DataFrame(_build_raw_rows())
    raw = _finalize_raw(raw)
    cells = pd.DataFrame(_build_result_rows(raw))
    cells = _finalize_cells(cells)
    metric_applicability = _build_metric_applicability()
    parity = _build_parity_summary(raw, cells)
    completeness = _build_completeness_summary(raw, cells, parity)
    return {
        "raw": raw,
        "cells": cells,
        "metric_applicability": metric_applicability,
        "parity": parity,
        "completeness": completeness,
    }


def write_outputs(root: Path, outputs: Mapping[str, pd.DataFrame], validation: Mapping[str, Any]) -> None:
    tables_dir = root / "results" / "tables"
    manifests_dir = root / "results" / "manifests"
    report_dir = root / "report"
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "cells": tables_dir / "extension_cell_universe.csv",
        "raw": tables_dir / "extension_raw_experiment_universe.csv",
        "completeness": tables_dir / "extension_cell_completeness_summary.csv",
        "metric_applicability": tables_dir / "extension_metric_applicability_matrix.csv",
        "parity": tables_dir / "extension_parity_summary.csv",
    }
    outputs["cells"].to_csv(paths["cells"], index=False)
    outputs["raw"].to_csv(paths["raw"], index=False)
    outputs["completeness"].to_csv(paths["completeness"], index=False)
    outputs["metric_applicability"].to_csv(paths["metric_applicability"], index=False)
    outputs["parity"].to_csv(paths["parity"], index=False)

    manifest = _build_freeze_manifest(root, outputs, paths, validation)
    (manifests_dir / "extension_cell_universe_freeze.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_dir / "extension_cell_universe.md").write_text(
        _render_cell_universe_report(outputs, validation, manifest),
        encoding="utf-8",
    )


def validate_universe(raw: pd.DataFrame, cells: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    _require_columns(raw, RAW_COLUMNS, "raw", errors)
    _require_columns(cells, RESULT_COLUMNS, "cells", errors)
    if errors:
        return {"passed": False, "errors": errors}

    active = set(ACTIVE_DATASETS)
    raw_datasets = set(raw["dataset"].astype(str))
    cell_datasets = set(cells["dataset"].astype(str))
    inactive = sorted((raw_datasets | cell_datasets) - active)
    if inactive:
        errors.append(f"inactive datasets present: {inactive}")
    missing_active = sorted(active - raw_datasets)
    if missing_active:
        errors.append(f"active datasets missing raw rows: {missing_active}")

    if raw["raw_experiment_id"].duplicated().any():
        errors.append("duplicate raw_experiment_id")
    if cells["cell_id"].duplicated().any():
        errors.append("duplicate cell_id")

    raw_key_fields = [
        "experiment_family",
        "dataset",
        "task",
        "model_name",
        "prompt_strategy",
        "candidate_set_id",
        "cohort_id",
        "history_length",
        "candidate_order_condition",
        "source_model_or_ranker",
        "profile_condition",
    ]
    if raw.duplicated(raw_key_fields).any():
        errors.append("duplicate raw experiment semantic configuration")

    missing_raw_refs = sorted(set(cells["raw_experiment_id"]) - set(raw["raw_experiment_id"]))
    if missing_raw_refs:
        errors.append(f"result cells reference missing raw experiments: {missing_raw_refs[:5]}")

    for frame_name, frame in (("raw", raw), ("cells", cells)):
        invalid = sorted(set(frame["status"]) - ALLOWED_STATUSES)
        if invalid:
            errors.append(f"{frame_name} invalid statuses: {invalid}")
        blocked_or_na = frame["status"].isin([STATUS_BLOCKED, STATUS_NOT_APPLICABLE])
        if frame.loc[blocked_or_na, "blocking_reason"].fillna("").eq("").any():
            errors.append(f"{frame_name} blocked/not-applicable rows missing reasons")

    computed_without_source = cells.loc[
        cells["status"].isin([STATUS_COMPUTED, STATUS_EXISTING_REUSABLE])
        & cells["existing_source"].fillna("").eq("")
    ]
    if not computed_without_source.empty:
        errors.append("computed/reusable result cells missing source artifact")

    required = cells["required_for_claim"].astype(str).str.startswith("REQUIRED")
    if cells.loc[required, "status"].fillna("").eq("").any():
        errors.append("required result cell has blank status")

    return {"passed": not errors, "errors": errors}


def _build_raw_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for task in ("ranking", "reranking"):
        for dataset in ORIGINAL_DATASETS:
            for prompt in PROMPT_STRATEGIES:
                rows.append(
                    _raw_row(
                        required_for_claim="REQUIRED",
                        claim_family="gemma_original_four_parity",
                        experiment_family=f"gemma_primary_{task}",
                        dataset=dataset,
                        task=task,
                        model_name=GEMMA_MODEL,
                        model_family="local_llm",
                        prompt_strategy=prompt,
                        prompt_contract_id=_prompt_contract(prompt),
                        candidate_set_id=_candidate_set_id(dataset, task),
                        cohort_id=f"{dataset}_canonical_{task}_{ORIGINAL_COHORT_SIZES[dataset]}",
                        cohort_size=ORIGINAL_COHORT_SIZES[dataset],
                        request_count=ORIGINAL_COHORT_SIZES[dataset],
                        status=STATUS_TO_INFER,
                        evidence_class="missing_gemma_counterpart_to_qwen_mistral",
                        planned_phase="FUTURE_GEMMA_ORIGINAL_FOUR_INFERENCE",
                        scenario="GEMMA_ORIGINAL_FOUR",
                        execution_budget_class="mandatory_currently_executable",
                        dependency_ids=_candidate_set_id(dataset, task),
                    )
                )

    for dataset in ORIGINAL_DATASETS:
        for length in HISTORY_LENGTHS:
            rows.append(
                _raw_row(
                    required_for_claim="REQUIRED",
                    claim_family="gemma_history_parity",
                    experiment_family="gemma_history_sensitivity",
                    dataset=dataset,
                    task="ranking",
                    model_name=GEMMA_MODEL,
                    model_family="local_llm",
                    prompt_strategy=PROMPT_BASE,
                    prompt_contract_id=PROMPT_CONTRACT_BASE_ID,
                    candidate_set_id=_candidate_set_id(dataset, "ranking"),
                    cohort_id=f"{dataset}_history_paired_200",
                    cohort_size=200,
                    history_length=length,
                    request_count=200,
                    status=STATUS_TO_INFER,
                    evidence_class="missing_gemma_counterpart_to_final_history",
                    planned_phase="FUTURE_GEMMA_HISTORY_INFERENCE",
                    scenario="GEMMA_ORIGINAL_FOUR",
                    execution_budget_class="mandatory_currently_executable",
                    dependency_ids=f"{dataset}_history_eligible_200;{_candidate_set_id(dataset, 'ranking')}",
                )
            )

    for dataset in GEMMA_CANDDIF_DATASETS:
        for order in ("FIRST_ORDER", "RANDOM_ORDER"):
            rows.append(
                _raw_row(
                    required_for_claim="REQUIRED",
                    claim_family="gemma_canddif_parity",
                    experiment_family="gemma_canddif_candidate_order",
                    dataset=dataset,
                    task="ranking",
                    model_name=GEMMA_MODEL,
                    model_family="local_llm",
                    prompt_strategy=PROMPT_BASE,
                    prompt_contract_id="r5b_corrected_canddif_strict_index_top5_v1",
                    candidate_set_id=f"r5b_corrected_{dataset}_canddif_100user_pairs",
                    cohort_id=f"{dataset}_canddif_100",
                    cohort_size=100,
                    candidate_order_condition=order,
                    request_count=100,
                    status=STATUS_TO_INFER,
                    evidence_class="missing_gemma_counterpart_to_r5b_canddif",
                    planned_phase="FUTURE_GEMMA_CANDDIF_INFERENCE",
                    scenario="GEMMA_ORIGINAL_FOUR",
                    execution_budget_class="mandatory_currently_executable",
                    dependency_ids=f"r5b_corrected_{dataset}_canddif_pairs",
                )
            )

    for dataset in ORIGINAL_DATASETS:
        rows.append(_profile_not_applicable_row(dataset, GEMMA_MODEL, "gemma_profile_generation_core_parity"))

    rows.extend(_foodcom_primary_rows())
    rows.extend(_foodcom_history_rows())
    rows.extend(_foodcom_canddif_rows())
    rows.extend(_foodcom_traditional_rows())
    rows.extend(_foodcom_reranking_rows())
    rows.extend(_foodcom_profile_rows())

    for dataset in ACTIVE_DATASETS:
        for metric in GLOBAL_POLICY_METRICS:
            rows.append(
                _raw_row(
                    required_for_claim="BLOCKED_POLICY_CELL",
                    claim_family="unresolved_original_metric_policy",
                    experiment_family="metric_policy_blocker",
                    dataset=dataset,
                    task="metric_policy",
                    model_name="ALL",
                    model_family="metric_policy",
                    prompt_strategy="NOT_APPLICABLE",
                    prompt_contract_id="NOT_APPLICABLE",
                    candidate_set_id="NOT_APPLICABLE",
                    cohort_id="NOT_APPLICABLE",
                    cohort_size=0,
                    source_model_or_ranker=metric,
                    parser_contract="NOT_APPLICABLE",
                    requires_llm_inference=False,
                    requires_training=False,
                    request_count=0,
                    status=STATUS_BLOCKED,
                    blocking_reason="POLICY_MISSING_OR_INPUT_UNAVAILABLE_IN_FROZEN_REPRODUCTION",
                    evidence_class="unresolved_original_metric_not_manufactured",
                    planned_phase="NOT_EXECUTABLE_UNTIL_POLICY_DEFINED",
                    scenario="GLOBAL_POLICY",
                    execution_budget_class="blocked_policy",
                )
            )
    return rows


def _foodcom_primary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in LOCAL_LLMS:
        for prompt in PROMPT_STRATEGIES:
            rows.append(
                _raw_row(
                    required_for_claim="REQUIRED",
                    claim_family="foodcom_primary_parity",
                    experiment_family="foodcom_primary_ranking",
                    dataset=FIFTH_DATASET,
                    task="ranking",
                    model_name=model,
                    model_family="local_llm",
                    prompt_strategy=prompt,
                    prompt_contract_id=_prompt_contract(prompt),
                    candidate_set_id="d0a_foodcom_primary1000_candidates",
                    cohort_id="foodcom_primary_1000",
                    cohort_size=1000,
                    request_count=1000,
                    status=STATUS_TO_INFER,
                    evidence_class="foodcom_d0a_primary_ready",
                    planned_phase="FUTURE_FOODCOM_PRIMARY_INFERENCE",
                    scenario="FOODCOM_PRIMARY_1000",
                    execution_budget_class="mandatory_currently_executable",
                    dependency_ids="results/tables/d0a_foodcom_primary1000_candidates.csv",
                )
            )
    return rows


def _foodcom_history_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in LOCAL_LLMS:
        for length in HISTORY_LENGTHS:
            rows.append(
                _raw_row(
                    required_for_claim="REQUIRED",
                    claim_family="foodcom_history_parity",
                    experiment_family="foodcom_history_sensitivity",
                    dataset=FIFTH_DATASET,
                    task="ranking",
                    model_name=model,
                    model_family="local_llm",
                    prompt_strategy=PROMPT_BASE,
                    prompt_contract_id=PROMPT_CONTRACT_BASE_ID,
                    candidate_set_id="d0a_foodcom_history200_candidates",
                    cohort_id="foodcom_history_paired_200",
                    cohort_size=200,
                    history_length=length,
                    request_count=200,
                    status=STATUS_TO_INFER,
                    evidence_class="foodcom_d0a_history_ready",
                    planned_phase="FUTURE_FOODCOM_HISTORY_INFERENCE",
                    scenario="FOODCOM_HISTORY_200",
                    execution_budget_class="mandatory_currently_executable",
                    dependency_ids="results/tables/d0a_foodcom_history200_users.csv",
                )
            )
    return rows


def _foodcom_canddif_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in LOCAL_LLMS:
        rows.append(
            _raw_row(
                required_for_claim="REQUIRED",
                claim_family="foodcom_canddif_primary",
                experiment_family="foodcom_candidate_order_sensitivity",
                dataset=FIFTH_DATASET,
                task="ranking",
                model_name=model,
                model_family="local_llm",
                prompt_strategy=PROMPT_BASE,
                prompt_contract_id=PROMPT_CONTRACT_BASE_ID,
                candidate_set_id="d0a_foodcom_primary1000_candidates",
                cohort_id="foodcom_primary_1000",
                cohort_size=1000,
                candidate_order_condition="ALTERNATE_SEEDED_ORDER",
                request_count=1000,
                status=STATUS_TO_INFER,
                evidence_class="foodcom_order_variant_required",
                planned_phase="FUTURE_FOODCOM_CANDDIF_INFERENCE",
                scenario="FOODCOM_CANDDIF_1000",
                execution_budget_class="mandatory_currently_executable",
                dependency_ids="results/tables/d0a_foodcom_primary1000_candidates.csv",
            )
        )
    return rows


def _foodcom_traditional_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in TRADITIONAL_MODELS:
        trainable = model in TRAINABLE_MODELS
        rows.append(
            _raw_row(
                required_for_claim="REQUIRED",
                claim_family="foodcom_traditional_parity",
                experiment_family="foodcom_traditional_baseline",
                dataset=FIFTH_DATASET,
                task="ranking",
                model_name=model,
                model_family="trainable_recommender_baseline" if trainable else "nontrainable_baseline",
                prompt_strategy="NOT_APPLICABLE",
                prompt_contract_id="NOT_APPLICABLE",
                candidate_set_id="d0a_foodcom_primary1000_candidates",
                cohort_id="foodcom_primary_1000",
                cohort_size=1000,
                source_model_or_ranker=model,
                requires_llm_inference=False,
                requires_training=trainable,
                request_count=0,
                status=STATUS_TO_TRAIN if trainable else STATUS_TO_COMPUTE_OFFLINE,
                evidence_class="candidate_constrained_baseline_adaptation",
                planned_phase="FUTURE_FOODCOM_TRADITIONAL_BASELINES",
                scenario="FOODCOM_PRIMARY_1000",
                execution_budget_class="mandatory_currently_executable",
                dependency_ids="results/tables/d0a_foodcom_primary1000_candidates.csv",
            )
        )
    return rows


def _foodcom_reranking_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in LOCAL_LLMS:
        for prompt in PROMPT_STRATEGIES:
            rows.append(
                _raw_row(
                    required_for_claim="REQUIRED_CONDITIONAL",
                    claim_family="foodcom_reranking_parity",
                    experiment_family="foodcom_reranking",
                    dataset=FIFTH_DATASET,
                    task="reranking",
                    model_name=model,
                    model_family="local_llm",
                    prompt_strategy=prompt,
                    prompt_contract_id=_prompt_contract(prompt),
                    candidate_set_id="foodcom_reranking_source_ranks_pending",
                    cohort_id="foodcom_primary_1000",
                    cohort_size=1000,
                    source_model_or_ranker="CANONICAL_FOODCOM_SOURCE_RANKS",
                    request_count=1000,
                    status=STATUS_BLOCKED,
                    blocking_reason="BLOCKED_PENDING_SOURCE_RANKS",
                    evidence_class="ready_after_training_source_ranks_required",
                    planned_phase="FUTURE_FOODCOM_RERANKING_AFTER_SOURCE_RANKS",
                    scenario="FOODCOM_RERANKING_AFTER_TRAINING",
                    execution_budget_class="required_conditional_after_training",
                    dependency_ids="foodcom_source_ranked_lists;positive_present_audit;denominator_policy",
                )
            )
    return rows


def _foodcom_profile_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in LOCAL_LLMS:
        for condition in ("PROFILE_GENERATION", "PROFILE_ONLY", "PROFILE_PLUS_HISTORY"):
            rows.append(
                _raw_row(
                    required_for_claim="REQUIRED_CONDITIONAL",
                    claim_family="foodcom_profile_generation",
                    experiment_family="foodcom_profile_dimension",
                    dataset=FIFTH_DATASET,
                    task="profile_generation",
                    model_name=model,
                    model_family="local_llm",
                    prompt_strategy=PROMPT_BASE,
                    prompt_contract_id=PROMPT_CONTRACT_BASE_ID,
                    candidate_set_id="d0a_foodcom_history200_candidates",
                    cohort_id="foodcom_history_paired_200",
                    cohort_size=200,
                    profile_condition=condition,
                    request_count=200,
                    status=STATUS_TO_INFER,
                    evidence_class="documented_domain_adaptation",
                    planned_phase="FUTURE_FOODCOM_PROFILE_ANALYSIS",
                    scenario="FOODCOM_PROFILE_CONDITIONAL",
                    execution_budget_class="required_conditional",
                    dependency_ids="results/tables/d0a_foodcom_history200_users.csv",
                )
            )
    return rows


def _build_result_rows(raw: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in raw.to_dict("records"):
        for metric in _metrics_for_raw(record):
            rows.append(_cell_row(record, metric))
    return rows


def _metrics_for_raw(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    family = str(raw["experiment_family"])
    task = str(raw["task"])
    if family == "metric_policy_blocker":
        return [_metric("blocked_policy_metric", str(raw["source_model_or_ranker"]), "", "not_computable")]
    if family.endswith("profile_generation_core_parity"):
        return [_metric("profile_generation", "profile_generation_performance", "", "not_core_parity")]
    if family == "foodcom_profile_dimension":
        return [
            _metric("profile_generation", "profile_generation_performance", "paired_users", "domain_adapted"),
            _metric("reliability", "strict_valid_rate", "all_planned_requests", "strict_index_parser"),
        ]
    if "traditional_baseline" in family:
        return [
            _metric("utility", "hr_at_5", "all_planned_users_fail_strict", "candidate_constrained"),
            _metric("utility", "ndcg_at_5", "all_planned_users_fail_strict", "candidate_constrained"),
            _metric("beyond_utility", "novelty_at_5", "valid_recommendations_top5", "adapted_offline"),
            _metric("beyond_utility", "arp_at_5", "valid_recommendations_top5", "adapted_offline"),
            _metric("beyond_utility", "aplt_at_5", "valid_recommendations_top5", "adapted_offline"),
            _metric("beyond_utility", "catalog_coverage_at_5", "valid_recommendations_top5", "adapted_offline"),
            _metric("beyond_utility", "item_coverage_at_5", "valid_recommendations_top5", "adapted_offline"),
            _metric("beyond_utility", "ild_at_5", "metadata_available_rows", "metadata_dependent"),
            _metric("beyond_utility", "serendipity_at_5", "metadata_available_rows", "metadata_dependent"),
            _metric("beyond_utility", "calibration_score_at_5", "history_available_rows", "metadata_dependent"),
        ]
    if "canddif" in family or "candidate_order_sensitivity" in family:
        return [
            _metric("utility", "hr_at_5", "all_planned_users_fail_strict", "strict_index_parser"),
            _metric("utility", "ndcg_at_5", "all_planned_users_fail_strict", "strict_index_parser"),
            _metric("diagnostic", "canddif_hr_at_5", "paired_candidate_order_rows", "strict_index_parser"),
            _metric("diagnostic", "canddif_ndcg_at_5", "paired_candidate_order_rows", "strict_index_parser"),
            _metric("reliability", "strict_valid_rate", "all_planned_requests", "strict_index_parser"),
            _metric("reliability", "structural_invalid_rate", "all_planned_requests", "strict_index_parser"),
            _metric("runtime", "latency_seconds_mean", "all_planned_requests", "runtime_observed"),
        ]
    if task in {"ranking", "reranking"}:
        metrics = [
            _metric("utility", "hr_at_5", "all_planned_users_fail_strict", "strict_index_parser"),
            _metric("utility", "ndcg_at_5", "all_planned_users_fail_strict", "strict_index_parser"),
            _metric("utility_diagnostic", "hr_at_5_complete_case", "strict_valid_outputs_only", "strict_index_parser"),
            _metric("utility_diagnostic", "ndcg_at_5_complete_case", "strict_valid_outputs_only", "strict_index_parser"),
            _metric("reliability", "strict_valid_rate", "all_planned_requests", "strict_index_parser"),
            _metric("reliability", "structural_invalid_rate", "all_planned_requests", "strict_index_parser"),
            _metric("reliability", "duplicate_rate", "all_planned_requests", "strict_index_parser"),
            _metric("reliability", "malformed_rate", "all_planned_requests", "strict_index_parser"),
            _metric("reliability", "candidate_invalid_rate", "all_planned_requests", "strict_index_parser"),
            _metric("beyond_utility", "novelty_at_5", "valid_recommendations_top5", "adapted_offline"),
            _metric("beyond_utility", "arp_at_5", "valid_recommendations_top5", "adapted_offline"),
            _metric("beyond_utility", "aplt_at_5", "valid_recommendations_top5", "adapted_offline"),
            _metric("beyond_utility", "catalog_coverage_at_5", "valid_recommendations_top5", "adapted_offline"),
            _metric("beyond_utility", "item_coverage_at_5", "valid_recommendations_top5", "adapted_offline"),
            _metric("beyond_utility", "ild_at_5", "metadata_available_rows", "metadata_dependent"),
            _metric("beyond_utility", "serendipity_at_5", "metadata_available_rows", "metadata_dependent"),
            _metric("beyond_utility", "calibration_score_at_5", "history_available_rows", "metadata_dependent"),
            _metric("runtime", "latency_seconds_mean", "all_planned_requests", "runtime_observed"),
            _metric("runtime", "throughput_requests_per_hour", "all_planned_requests", "runtime_observed"),
        ]
        if task == "reranking":
            metrics.extend(
                [
                    _metric("reranking_diagnostic", "hr_at_5_positive_present_diagnostic", "positive_present_rows_only", "strict_index_parser"),
                    _metric("reranking_diagnostic", "ndcg_at_5_positive_present_diagnostic", "positive_present_rows_only", "strict_index_parser"),
                    _metric("reranking_diagnostic", "candidate_pool_positive_present_rate", "all_planned_users", "candidate_pool_audit"),
                ]
            )
        return metrics
    return [_metric("accounting", "cell_classified", "", "not_metric_applicable")]


def _cell_row(raw: Mapping[str, Any], metric: Mapping[str, str]) -> dict[str, Any]:
    status = str(raw["status"])
    reason = str(raw["blocking_reason"])
    evidence_class = str(raw["evidence_class"])
    metric_id = str(metric["metric_id"])
    if metric_id in GLOBAL_POLICY_METRICS:
        status = STATUS_BLOCKED
        reason = "POLICY_MISSING_OR_INPUT_UNAVAILABLE_IN_FROZEN_REPRODUCTION"
        evidence_class = "unresolved_original_metric_not_manufactured"
    if metric_id == "calibration_score_at_5" and str(raw.get("history_length", "")) == "0":
        status = STATUS_NOT_APPLICABLE
        reason = "L0_CALIBRATION_NOT_APPLICABLE_NO_HISTORY_DISTRIBUTION"
        evidence_class = f"{evidence_class};calibration_l0_not_applicable"
    elif metric_id == "calibration_score_at_5" and str(raw.get("history_length", "")) == "1":
        evidence_class = f"{evidence_class};calibration_l1_fragile"

    can_compute_offline = "false" if metric["metric_scope"] == "profile_generation" else "true"
    row = {
        "raw_experiment_id": raw["raw_experiment_id"],
        "required_for_claim": raw["required_for_claim"],
        "claim_family": raw["claim_family"],
        "experiment_family": raw["experiment_family"],
        "dataset": raw["dataset"],
        "task": raw["task"],
        "model_name": raw["model_name"],
        "model_family": raw["model_family"],
        "prompt_strategy": raw["prompt_strategy"],
        "prompt_contract_id": raw["prompt_contract_id"],
        "candidate_set_id": raw["candidate_set_id"],
        "candidate_pool_size": raw["candidate_pool_size"],
        "history_length": raw["history_length"],
        "candidate_order_condition": raw["candidate_order_condition"],
        "profile_condition": raw["profile_condition"],
        "seed_or_run": "E2A_ACTIVE_PLANNED_OR_EXISTING",
        "parser_contract": raw["parser_contract"],
        "k": str(METRIC_K if metric_id.endswith("_at_5") or metric_id in {"hr_at_5", "ndcg_at_5"} else ""),
        "metric_scope": metric["metric_scope"],
        "metric_id": metric_id,
        "denominator_policy": metric["denominator_policy"],
        "validity_policy": metric["validity_policy"],
        "requires_llm_inference": str(raw["requires_llm_inference"]),
        "requires_training": str(raw["requires_training"]),
        "can_compute_offline": can_compute_offline,
        "existing_source": raw["existing_source"],
        "status": status,
        "blocking_reason": reason,
        "evidence_class": evidence_class,
        "planned_phase": raw["planned_phase"],
    }
    row["cell_id"] = _id("EXT-CELL", row, RESULT_COLUMNS[1:])
    return row


def _build_metric_applicability() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        ("hr_at_5", "utility", "APPLICABLE", ""),
        ("ndcg_at_5", "utility", "APPLICABLE", ""),
        ("strict_valid_rate", "reliability", "APPLICABLE_LLM_ONLY", ""),
        ("structural_invalid_rate", "reliability", "APPLICABLE_LLM_ONLY", ""),
        ("duplicate_rate", "reliability", "APPLICABLE_LLM_ONLY", ""),
        ("malformed_rate", "reliability", "APPLICABLE_LLM_ONLY", ""),
        ("candidate_invalid_rate", "reliability", "APPLICABLE_LLM_ONLY", ""),
        ("novelty_at_5", "beyond_utility", "APPLICABLE_ADAPTED", ""),
        ("arp_at_5", "beyond_utility", "APPLICABLE_ADAPTED", ""),
        ("aplt_at_5", "beyond_utility", "APPLICABLE_ADAPTED", ""),
        ("catalog_coverage_at_5", "beyond_utility", "APPLICABLE_ADAPTED", ""),
        ("item_coverage_at_5", "beyond_utility", "APPLICABLE_ADAPTED", ""),
        ("ild_at_5", "beyond_utility", "APPLICABLE_WITH_METADATA_ADAPTATION", ""),
        ("serendipity_at_5", "beyond_utility", "APPLICABLE_WITH_METADATA_ADAPTATION", ""),
        ("calibration_score_at_5", "beyond_utility", "APPLICABLE_WITH_HISTORY_LIMITATION", "L=0 not applicable; L=1 fragile"),
        ("latency_seconds_mean", "runtime", "APPLICABLE_LLM_ONLY", ""),
        ("throughput_requests_per_hour", "runtime", "APPLICABLE_LLM_ONLY", ""),
    ]
    for metric in GLOBAL_POLICY_METRICS:
        metrics.append((metric, "blocked_policy_metric", "BLOCKED", "POLICY_MISSING_OR_INPUT_UNAVAILABLE_IN_FROZEN_REPRODUCTION"))

    for dataset in ACTIVE_DATASETS:
        for metric_id, scope, applicability, reason in metrics:
            rows.append(
                {
                    "metric_id": metric_id,
                    "metric_scope": scope,
                    "dataset": dataset,
                    "applicability_status": applicability,
                    "status": STATUS_BLOCKED if applicability == "BLOCKED" else STATUS_TO_COMPUTE_OFFLINE,
                    "blocking_reason": reason if applicability == "BLOCKED" else "",
                    "evidence_class": "adapted_local_metric_contract" if applicability != "BLOCKED" else "unresolved_original_metric_not_manufactured",
                    "required_for_claim": "BLOCKED_POLICY_CELL" if applicability == "BLOCKED" else "REQUIRED_WHERE_MODEL_OUTPUT_EXISTS",
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "metric_scope", "metric_id"]).reset_index(drop=True)


def _build_parity_summary(raw: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _parity_row("Gemma parity", "original_four", raw, cells, raw["claim_family"].str.startswith("gemma_")),
        _parity_row("Food.com primary", "FOODCOM_PRIMARY_1000", raw, cells, raw["claim_family"].eq("foodcom_primary_parity"), cohort_size=1000),
        _parity_row("Food.com history", "FOODCOM_HISTORY_200", raw, cells, raw["claim_family"].eq("foodcom_history_parity"), cohort_size=200),
        _parity_row("Food.com CandDif", "FOODCOM_CANDDIF_1000", raw, cells, raw["claim_family"].eq("foodcom_canddif_primary"), cohort_size=1000),
        _parity_row("Food.com profiles", "FOODCOM_PROFILE_CONDITIONAL", raw, cells, raw["claim_family"].eq("foodcom_profile_generation"), cohort_size=200),
    ]
    return pd.DataFrame(rows)


def _parity_row(
    parity_family: str,
    scenario: str,
    raw: pd.DataFrame,
    cells: pd.DataFrame,
    mask: pd.Series,
    *,
    cohort_size: int | None = None,
) -> dict[str, Any]:
    raw_subset = raw.loc[mask].copy()
    cell_subset = cells.loc[cells["raw_experiment_id"].isin(raw_subset["raw_experiment_id"])].copy()
    llm = raw_subset.loc[raw_subset["requires_llm_inference"].eq("true")]
    executable = llm.loc[llm["status"].eq(STATUS_TO_INFER) & ~llm["required_for_claim"].eq("OPTIONAL_SENSITIVITY")]
    blocked = llm.loc[llm["status"].eq(STATUS_BLOCKED)]
    optional = llm.loc[llm["required_for_claim"].eq("OPTIONAL_SENSITIVITY")]
    return {
        "parity_family": parity_family,
        "scenario": scenario,
        "cohort_size": "" if cohort_size is None else cohort_size,
        "raw_experiment_count": len(raw_subset),
        "result_cell_count": len(cell_subset),
        "qwen_requests": _model_request_sum(llm, "qwen2:7b-instruct"),
        "mistral_requests": _model_request_sum(llm, "mistral:7b-instruct"),
        "gemma_requests": _model_request_sum(llm, GEMMA_MODEL),
        "total_llm_requests": int(pd.to_numeric(llm["request_count"], errors="coerce").fillna(0).sum()),
        "executable_llm_requests": int(pd.to_numeric(executable["request_count"], errors="coerce").fillna(0).sum()),
        "blocked_llm_requests": int(pd.to_numeric(blocked["request_count"], errors="coerce").fillna(0).sum()),
        "optional_sensitivity_requests": int(pd.to_numeric(optional["request_count"], errors="coerce").fillna(0).sum()),
        "training_runs": int(raw_subset.loc[raw_subset["requires_training"].eq("true")].shape[0]),
        "training_runs_currently_executable": int(raw_subset.loc[raw_subset["requires_training"].eq("true") & raw_subset["status"].eq(STATUS_TO_TRAIN)].shape[0]),
        "offline_only_cells": int(cell_subset.loc[cell_subset["requires_llm_inference"].eq("false") & cell_subset["requires_training"].eq("false") & cell_subset["status"].eq(STATUS_TO_COMPUTE_OFFLINE)].shape[0]),
        "blocked_cells": int(cell_subset.loc[cell_subset["status"].eq(STATUS_BLOCKED)].shape[0]),
        "not_applicable_cells": int(cell_subset.loc[cell_subset["status"].eq(STATUS_NOT_APPLICABLE)].shape[0]),
        "completion_status": "classified_incomplete_pre_execution",
    }


def _build_completeness_summary(raw: pd.DataFrame, cells: pd.DataFrame, parity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for universe, frame in (("raw_experiment", raw), ("result_cell", cells)):
        for (claim_family, status), group in frame.groupby(["claim_family", "status"], dropna=False):
            rows.append(
                {
                    "summary_scope": universe,
                    "claim_family": claim_family,
                    "scenario": "ALL",
                    "status": status,
                    "row_count": len(group),
                    "llm_request_count": int(pd.to_numeric(group.get("request_count", pd.Series(dtype="int64")), errors="coerce").fillna(0).sum()) if universe == "raw_experiment" else 0,
                    "training_run_count": int(group.loc[group.get("requires_training", pd.Series(dtype="str")).eq("true")].shape[0]) if universe == "raw_experiment" else 0,
                }
            )
    for row in parity.to_dict("records"):
        rows.append(
            {
                "summary_scope": "budget",
                "claim_family": row["parity_family"],
                "scenario": row["scenario"],
                "status": "BUDGET_FROM_RAW_EXPERIMENTS_ONLY",
                "row_count": row["raw_experiment_count"],
                "llm_request_count": row["total_llm_requests"],
                "training_run_count": row["training_runs"],
            }
        )
    return pd.DataFrame(rows).sort_values(["summary_scope", "claim_family", "scenario", "status"]).reset_index(drop=True)


def _build_freeze_manifest(
    root: Path,
    outputs: Mapping[str, pd.DataFrame],
    paths: Mapping[str, Path],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    output_hashes = {name: sha256_file_or_empty(path) for name, path in paths.items()}
    source_hashes = {
        rel: {
            "exists": (root / rel).exists(),
            "sha256": sha256_file_or_empty(root / rel),
        }
        for rel in SOURCE_ARTIFACTS
    }
    semantic_payload = {
        "schema_version": SCHEMA_VERSION,
        "raw": outputs["raw"].sort_values("raw_experiment_id").to_dict("records"),
        "cells": outputs["cells"].sort_values("cell_id").to_dict("records"),
        "metric_applicability": outputs["metric_applicability"].sort_values(["dataset", "metric_id"]).to_dict("records"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_identity": GENERATOR_ID,
        "source_commit": _git(["rev-parse", "HEAD"], root),
        "source_branch": _git(["branch", "--show-current"], root),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "active_datasets": list(ACTIVE_DATASETS),
        "source_artifacts": source_hashes,
        "output_artifacts": output_hashes,
        "semantic_universe_hash": sha256_json(semantic_payload),
        "validation": validation,
        "decision_token": DECISION_TOKEN,
        "no_inference": True,
        "no_training": True,
        "no_downloads": True,
        "no_raw_reconstruction": True,
    }


def _render_cell_universe_report(
    outputs: Mapping[str, pd.DataFrame],
    validation: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    raw = outputs["raw"]
    cells = outputs["cells"]
    parity = outputs["parity"]
    gemma = parity.loc[parity["parity_family"].eq("Gemma parity")].iloc[0]
    primary = parity.loc[parity["parity_family"].eq("Food.com primary")].iloc[0]
    history = parity.loc[parity["parity_family"].eq("Food.com history")].iloc[0]
    blocked = int(cells["status"].eq(STATUS_BLOCKED).sum())
    offline = int(cells["status"].eq(STATUS_TO_COMPUTE_OFFLINE).sum())
    training = int(raw["requires_training"].eq("true").sum())
    return f"""# Extension Cell Universe

Decision token: `{DECISION_TOKEN}`

This report materializes the current operational raw-experiment and result-cell
universes over Beauty, Sports, ML-1M, LastFM, and Food.com. No LLM inference,
recommender training, downloads, or raw-data reconstruction were performed.

## Complete Gemma parity

Gemma raw experiments classified: {gemma['raw_experiment_count']}
Gemma result cells classified: {gemma['result_cell_count']}
Gemma planned LLM requests from raw table: {gemma['total_llm_requests']}

## Food.com final fifth dataset

Food.com primary raw experiments: {primary['raw_experiment_count']}
Food.com primary planned LLM requests: {primary['total_llm_requests']}
Food.com history raw experiments: {history['raw_experiment_count']}
Food.com history planned LLM requests: {history['total_llm_requests']}

## Offline and training cells

Offline-only result cells currently classified: {offline}
Training-required raw rows: {training}

## Blocked cells

Blocked rows are limited to source-rank prerequisites and global policy/input
definitions that are not resolved in the frozen protocol. Blocked result cells:
{blocked}

## Active dataset set

{', '.join(manifest['active_datasets'])}

## Validation

Passed: `{validation['passed']}`
"""


def _raw_row(
    *,
    required_for_claim: str,
    claim_family: str,
    experiment_family: str,
    dataset: str,
    task: str,
    model_name: str,
    model_family: str,
    prompt_strategy: str,
    prompt_contract_id: str,
    candidate_set_id: str,
    cohort_id: str,
    cohort_size: int | str,
    request_count: int,
    status: str,
    evidence_class: str,
    planned_phase: str,
    scenario: str,
    execution_budget_class: str,
    candidate_pool_size: int | str = CANDIDATE_COUNT,
    history_length: int | str = "",
    candidate_order_condition: str = "CANONICAL_ORDER",
    profile_condition: str = "NOT_APPLICABLE",
    source_model_or_ranker: str = "",
    parser_contract: str = PARSER_CONTRACT_ID,
    requires_llm_inference: bool = True,
    requires_training: bool = False,
    dependency_ids: str = "",
    existing_source: str = "",
    blocking_reason: str = "",
) -> dict[str, Any]:
    row = {
        "required_for_claim": required_for_claim,
        "claim_family": claim_family,
        "experiment_family": experiment_family,
        "dataset": dataset,
        "task": task,
        "model_name": model_name,
        "model_family": model_family,
        "prompt_strategy": prompt_strategy,
        "prompt_contract_id": prompt_contract_id,
        "candidate_set_id": candidate_set_id,
        "candidate_pool_size": candidate_pool_size,
        "cohort_id": cohort_id,
        "cohort_size": cohort_size,
        "history_length": history_length,
        "candidate_order_condition": candidate_order_condition,
        "profile_condition": profile_condition,
        "source_model_or_ranker": source_model_or_ranker,
        "parser_contract": parser_contract,
        "requires_llm_inference": _bool(requires_llm_inference),
        "requires_training": _bool(requires_training),
        "request_count": request_count,
        "dependency_ids": dependency_ids,
        "existing_source": existing_source,
        "status": status,
        "blocking_reason": blocking_reason,
        "evidence_class": evidence_class,
        "planned_phase": planned_phase,
        "scenario": scenario,
        "execution_budget_class": execution_budget_class,
    }
    row["raw_experiment_id"] = _id("EXT-RAW", row, RAW_COLUMNS[1:])
    return row


def _profile_not_applicable_row(dataset: str, model: str, family: str) -> dict[str, Any]:
    return _raw_row(
        required_for_claim="NOT_APPLICABLE_CORE_PARITY",
        claim_family="profile_generation",
        experiment_family=family,
        dataset=dataset,
        task="profile_generation",
        model_name=model,
        model_family="local_llm",
        prompt_strategy="NOT_APPLICABLE",
        prompt_contract_id="NOT_APPLICABLE",
        candidate_set_id="NOT_APPLICABLE",
        cohort_id="NOT_APPLICABLE",
        cohort_size=0,
        candidate_pool_size="",
        profile_condition="PROFILE_GENERATION_NOT_REOPENED",
        parser_contract="NOT_APPLICABLE",
        requires_llm_inference=False,
        request_count=0,
        status=STATUS_NOT_APPLICABLE,
        blocking_reason="NOT_APPLICABLE_CORE_PARITY_FROZEN_PROFILE_RQ_UNRESOLVED",
        evidence_class="profile_generation_not_core_extension",
        planned_phase="NOT_IN_CORE_EXTENSION",
        scenario="CORE_PARITY",
        execution_budget_class="not_applicable",
    )


def _metric(scope: str, metric_id: str, denominator: str, validity: str) -> dict[str, str]:
    return {
        "metric_scope": scope,
        "metric_id": metric_id,
        "denominator_policy": denominator,
        "validity_policy": validity,
    }


def _finalize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.loc[:, RAW_COLUMNS].sort_values("raw_experiment_id").reset_index(drop=True)


def _finalize_cells(cells: pd.DataFrame) -> pd.DataFrame:
    return cells.loc[:, RESULT_COLUMNS].sort_values("cell_id").reset_index(drop=True)


def _candidate_set_id(dataset: str, task: str) -> str:
    if dataset == "lastfm":
        return f"phase_f6m_lastfm_{task}_candidates_20"
    return f"canonical_local_llm_{dataset}_{task}_candidates_20"


def _prompt_contract(prompt: str) -> str:
    if prompt == PROMPT_BASE:
        return PROMPT_CONTRACT_BASE_ID
    if prompt == PROMPT_LLMRANK:
        return PROMPT_CONTRACT_LLMRANK_ID
    raise ValueError(f"unknown prompt: {prompt}")


def _model_request_sum(frame: pd.DataFrame, model: str) -> int:
    return int(pd.to_numeric(frame.loc[frame["model_name"].eq(model), "request_count"], errors="coerce").fillna(0).sum())


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str, errors: list[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        errors.append(f"{name} missing columns: {missing}")


def _id(prefix: str, row: Mapping[str, Any], fields: Iterable[str]) -> str:
    payload = {field: str(row.get(field, "")) for field in fields}
    return f"{prefix}-{sha256_json(payload)[:16].upper()}"


def _bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def _git(args: list[str], root: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


if __name__ == "__main__":
    raise SystemExit(main())

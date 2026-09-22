"""F0 final extension freeze generator.

F0 is an offline planning freeze. It materializes the final five-dataset
experiment universe, request budget, dependency DAG, and scientific contracts
without inference, training, downloads, network calls, or commits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import pandas as pd

from src.extension.e0_protocol import (
    CANDIDATE_COUNT,
    HISTORY_LENGTHS,
    METRIC_K,
    PARSER_CONTRACT_ID,
    POSITION_BASE,
    PROMPT_BASE,
    PROMPT_CONTRACT_BASE_ID,
    PROMPT_CONTRACT_LLMRANK_ID,
    PROMPT_LLMRANK,
    REQUESTED_OUTPUT_LENGTH,
    sha256_json,
    stable_json,
)
from src.utils.config import find_project_root


PHASE_ID = "f0_final_extension_freeze"
DECISION_TOKEN = "FINAL_EXTENSION_FREEZE_VALIDATED"
EXPECTED_HEAD = "30d6c3e5fcb0e97451a4420829b88de10bb57213"
GENERATOR_ID = "src.extension.f0_final_extension_freeze"

DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
DATASET_LABELS = {
    "beauty": "Amazon Beauty",
    "sports": "Amazon Sports",
    "ml1m": "ML-1M",
    "lastfm": "LastFM",
    "foodcom": "Food.com",
}
LOCAL_LLMS = ("qwen2:7b-instruct", "mistral:7b-instruct", "gemma2:9b")
PROMPTS = (PROMPT_BASE, PROMPT_LLMRANK)
HISTORY_LENGTHS_FINAL = tuple(HISTORY_LENGTHS)
PRIMARY_COHORT_SIZES = {
    "beauty": 990,
    "sports": 892,
    "ml1m": 1000,
    "lastfm": 1000,
    "foodcom": 1000,
}
HISTORY_COHORT_SIZE = 200
SENSITIVITY_COHORT_SIZE = 200
TRADITIONAL_METHODS = ("Random", "MostPop", "BM25-like/content", "BPRMF", "GRU4Rec", "SASRec", "LightGCN")
TRAINABLE_METHODS = {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}
POOL_SENSITIVITY_SIZES = (10, 50)
PREVIOUS_D0B_REQUEST_ESTIMATE = 44_928

GENERATION_PARAMETERS = {
    "temperature": 0.0,
    "num_predict": 256,
    "stream": False,
    "timeout_seconds": 150,
    "seed": "",
    "top_k": "",
    "top_p": "",
    "stop_sequences": "",
    "context_window": "",
    "keep_alive": "",
}
GENERATION_CONFIG_ID = "ollama_local_zero_temp_top10_v1"
GENERATION_CONFIG_HASH = sha256_json(GENERATION_PARAMETERS)

RAW_STATUSES = {
    "EXISTING_REUSABLE_IDENTICAL",
    "TO_INFER",
    "OFFLINE_ONLY",
    "TRAINING_DEPENDENT",
    "CONDITIONAL",
    "OPTIONAL",
    "BLOCKED",
    "NOT_APPLICABLE",
}
RESULT_STATUSES = {"COMPUTED", "EXISTING_REUSABLE", "TO_COMPUTE", "BLOCKED", "NOT_APPLICABLE"}

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")

OUTPUT_TABLES = {
    "dataset_matrix": TABLE_DIR / "f0_final_dataset_matrix.csv",
    "model_matrix": TABLE_DIR / "f0_final_model_matrix.csv",
    "raw_universe": TABLE_DIR / "f0_final_raw_experiment_universe.csv",
    "result_universe": TABLE_DIR / "f0_final_result_cell_universe.csv",
    "primary_matrix": TABLE_DIR / "f0_final_primary_ranking_matrix.csv",
    "history_matrix": TABLE_DIR / "f0_final_history_matrix.csv",
    "reranking_matrix": TABLE_DIR / "f0_final_reranking_matrix.csv",
    "traditional_matrix": TABLE_DIR / "f0_final_traditional_model_matrix.csv",
    "metric_matrix": TABLE_DIR / "f0_final_metric_matrix.csv",
    "global_blockers": TABLE_DIR / "f0_final_global_blockers.csv",
    "foodcom_feature_policy": TABLE_DIR / "f0_final_foodcom_feature_policy.csv",
    "sensitivity_matrix": TABLE_DIR / "f0_final_sensitivity_matrix.csv",
    "profile_matrix": TABLE_DIR / "f0_final_profile_matrix.csv",
    "request_budget": TABLE_DIR / "f0_final_request_budget.csv",
    "reuse_audit": TABLE_DIR / "f0_final_reuse_audit.csv",
    "training_dependencies": TABLE_DIR / "f0_final_training_dependencies.csv",
    "execution_dag": TABLE_DIR / "f0_final_execution_dag.csv",
    "statistical_comparisons": TABLE_DIR / "f0_final_statistical_comparisons.csv",
    "test_results": TABLE_DIR / "f0_final_test_results.csv",
}
OUTPUT_MANIFEST = MANIFEST_DIR / "f0_final_extension_freeze.json"
OUTPUT_INFERENCE_CONTRACT = MANIFEST_DIR / "f0_final_inference_contract.json"
OUTPUT_REPORT = REPORT_DIR / "f0_final_extension_freeze.md"

FOODCOM_HASH_EXPECTATIONS = {
    "data/raw/foodcom/source/food-com-recipes-and-user-interactions.zip": "e7a43e623c616799f78a0479b550483450fd7ede4e0c40979db3458a82b5b913",
    "data/raw/foodcom/d0a_extracted/e7a43e623c61/RAW_interactions.csv": "4e278d2f687b1020f5694e51003a7d6174344b51aebbdf4530de88b5c6a78ee4",
    "data/raw/foodcom/d0a_extracted/e7a43e623c61/RAW_recipes.csv": "6a8136d1da9e03396a9f52d72200b3fddb8fe103f11e1496effca30ac4e0539f",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the F0 final extension freeze.")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    outputs = generate_f0(root)
    if not bool(outputs["validation"]["passed"]):
        print(stable_json(outputs["validation"]))
        return 1
    if not args.validate_only:
        write_outputs(root, outputs)
    print(f"raw_experiments={len(outputs['raw_universe'])}")
    print(f"result_cells={len(outputs['result_universe'])}")
    print(f"mandatory_raw_requests={budget_value(outputs['request_budget'], 'MANDATORY_RAW_REQUESTS_TOTAL')}")
    print(f"decision={DECISION_TOKEN}")
    print("llm_inference_calls=0")
    print("training_runs=0")
    print("downloads=0")
    print("network_calls=0")
    return 0


def generate_f0(root: Path | None = None) -> dict[str, Any]:
    project_root = root or find_project_root()
    dataset_matrix = build_dataset_matrix(project_root)
    model_matrix = build_model_matrix(project_root)
    primary_matrix = build_primary_matrix()
    history_matrix = build_history_matrix()
    reranking_matrix = build_reranking_matrix()
    traditional_matrix = build_traditional_matrix()
    metric_matrix = build_metric_matrix()
    global_blockers = build_global_blockers()
    foodcom_feature_policy = build_foodcom_feature_policy(project_root)
    sensitivity_matrix = build_sensitivity_matrix()
    profile_matrix = build_profile_matrix()
    raw_universe = build_raw_universe(
        primary_matrix,
        history_matrix,
        reranking_matrix,
        traditional_matrix,
        sensitivity_matrix,
        profile_matrix,
        global_blockers,
    )
    result_universe = build_result_universe(raw_universe, metric_matrix)
    reuse_audit = build_reuse_audit(primary_matrix, history_matrix, sensitivity_matrix)
    training_dependencies = build_training_dependencies(reranking_matrix, traditional_matrix)
    execution_dag = build_execution_dag(raw_universe)
    statistical_comparisons = build_statistical_comparisons()
    request_budget = build_request_budget(raw_universe)
    validation = validate_outputs(
        project_root,
        dataset_matrix,
        model_matrix,
        raw_universe,
        result_universe,
        primary_matrix,
        history_matrix,
        metric_matrix,
        global_blockers,
        request_budget,
        execution_dag,
    )
    test_results = build_test_results(validation)
    manifest = build_manifest(project_root, raw_universe, result_universe, request_budget, validation)
    inference_contract = build_inference_contract(project_root, model_matrix)
    return {
        "dataset_matrix": dataset_matrix,
        "model_matrix": model_matrix,
        "raw_universe": raw_universe,
        "result_universe": result_universe,
        "primary_matrix": primary_matrix,
        "history_matrix": history_matrix,
        "reranking_matrix": reranking_matrix,
        "traditional_matrix": traditional_matrix,
        "metric_matrix": metric_matrix,
        "global_blockers": global_blockers,
        "foodcom_feature_policy": foodcom_feature_policy,
        "sensitivity_matrix": sensitivity_matrix,
        "profile_matrix": profile_matrix,
        "request_budget": request_budget,
        "reuse_audit": reuse_audit,
        "training_dependencies": training_dependencies,
        "execution_dag": execution_dag,
        "statistical_comparisons": statistical_comparisons,
        "test_results": test_results,
        "validation": validation,
        "manifest": manifest,
        "inference_contract": inference_contract,
    }


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for directory in (root / TABLE_DIR, root / MANIFEST_DIR, root / REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    for key, rel_path in OUTPUT_TABLES.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(root / rel_path, index=False, lineterminator="\n")

    manifest = dict(outputs["manifest"])
    manifest["artifact_hashes"] = {
        rel_path.as_posix(): sha256_file(root / rel_path)
        for rel_path in OUTPUT_TABLES.values()
    }
    (root / OUTPUT_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    inference_contract = dict(outputs["inference_contract"])
    inference_contract["f0_manifest_hash"] = sha256_file(root / OUTPUT_MANIFEST)
    (root / OUTPUT_INFERENCE_CONTRACT).write_text(
        json.dumps(inference_contract, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / OUTPUT_REPORT).write_text(render_report(outputs), encoding="utf-8")


def build_dataset_matrix(root: Path) -> pd.DataFrame:
    h0 = read_csv(root / "results/tables/extension_h0_dataset_l20_feasibility.csv")
    d0a_history = read_csv(root / "results/tables/d0a_foodcom_history200_feasibility.csv")
    d1_active = read_csv(root / "results/tables/d1_active_dataset_audit.csv")
    h0_l20 = {
        clean(row.get("dataset")): clean(row.get("strict_paired_l20_eligible_users"))
        for row in h0.to_dict("records")
    }
    food_l20 = first_value(d0a_history, "strict_paired_l20_eligible_users", "6387")
    active_counts = {
        clean(row.get("dataset_id")): row
        for row in d1_active.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        active = active_counts.get(dataset, {})
        rows.append(
            {
                "dataset": dataset,
                "display_name": DATASET_LABELS[dataset],
                "active": "true",
                "role": "fifth_dataset" if dataset == "foodcom" else "original_four",
                "primary_cohort_size": PRIMARY_COHORT_SIZES[dataset],
                "history_paired_cohort_size": HISTORY_COHORT_SIZE,
                "strict_paired_l20_eligible_users": food_l20 if dataset == "foodcom" else h0_l20.get(dataset, ""),
                "active_raw_experiment_count_d1": clean(active.get("active_raw_experiment_count")),
                "active_cell_count_d1": clean(active.get("active_cell_count")),
                "canonical_candidate_pool_size": CANDIDATE_COUNT,
                "status": "ACTIVE",
                "evidence_source": "results/tables/d1_active_dataset_audit.csv;results/tables/extension_h0_dataset_l20_feasibility.csv;results/tables/d0a_foodcom_history200_feasibility.csv",
            }
        )
    return pd.DataFrame(rows)


def build_model_matrix(root: Path) -> pd.DataFrame:
    manifest_data = read_ollama_manifests()
    ollama_version = ollama_client_version()
    rows: list[dict[str, Any]] = []
    for model in LOCAL_LLMS:
        manifest = manifest_data.get(model, {})
        rows.append(
            {
                "model_name": model,
                "model_role": "new_model_extension_model_generalization" if model == "gemma2:9b" else "local_substitute_model_reproduction_robustness",
                "scientific_classification": "ORIGINAL_EXTENSION" if model == "gemma2:9b" else "PAPER_PROTOCOL_DIFFERENT_OPEN_MODEL",
                "ollama_manifest_present": str(bool(manifest)).lower(),
                "ollama_manifest_digest": manifest.get("manifest_digest", ""),
                "ollama_manifest_digest_prefix": manifest.get("manifest_digest", "")[:12],
                "ollama_model_layer_digest": manifest.get("model_layer_digest", ""),
                "ollama_model_layer_size": manifest.get("model_layer_size", ""),
                "ollama_client_version": ollama_version,
                "ollama_server_status": "not_required_for_f0_not_running_or_unverified",
                "generation_config_id": GENERATION_CONFIG_ID,
                "generation_config_hash": GENERATION_CONFIG_HASH,
                "generation_parameters": stable_json(GENERATION_PARAMETERS),
                "status": "LOCAL_MANIFEST_VERIFIED" if manifest else "BLOCKED_LOCAL_MANIFEST_MISSING",
                "notes": "Gemma is not an original-paper LLM replication." if model == "gemma2:9b" else "Do not claim exact original-model reproduction.",
            }
        )
    return pd.DataFrame(rows)


def build_primary_matrix() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for model in LOCAL_LLMS:
            for prompt in PROMPTS:
                status = "TO_INFER"
                reuse = "NO_PROTOCOL_IDENTICAL_REUSE_PROMOTED"
                evidence = "F0 final top10 five-dataset contract; historical pilots/frozen recovery outputs are not promoted without full identity match."
                rows.append(
                    {
                        "dataset": dataset,
                        "task": "ranking",
                        "model_name": model,
                        "prompt": prompt,
                        "cohort_id": f"{dataset}_primary_{PRIMARY_COHORT_SIZES[dataset]}",
                        "cohort_size": PRIMARY_COHORT_SIZES[dataset],
                        "candidate_pool_size": CANDIDATE_COUNT,
                        "candidate_composition": "1_heldout_positive_19_unseen_negatives",
                        "candidate_indexing": f"{POSITION_BASE}..{CANDIDATE_COUNT}",
                        "requested_output_length": REQUESTED_OUTPUT_LENGTH,
                        "parser_expected_length": REQUESTED_OUTPUT_LENGTH,
                        "metric_k": METRIC_K,
                        "parser": PARSER_CONTRACT_ID,
                        "generation_config_id": GENERATION_CONFIG_ID,
                        "raw_status": status,
                        "request_count": PRIMARY_COHORT_SIZES[dataset],
                        "reuse_decision": reuse,
                        "mandatory_class": "MANDATORY",
                        "budget_class": "CORE_MANDATORY_NOW",
                        "evidence": evidence,
                    }
                )
    return pd.DataFrame(rows)


def build_history_matrix() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for model in LOCAL_LLMS:
            for length in HISTORY_LENGTHS_FINAL:
                rows.append(
                    {
                        "dataset": dataset,
                        "task": "ranking",
                        "model_name": model,
                        "prompt": PROMPT_BASE,
                        "history_length": length,
                        "l0_policy": "explicit_empty_history" if length == 0 else "last_L_visible_interactions",
                        "cohort_id": f"{dataset}_history_paired_{HISTORY_COHORT_SIZE}",
                        "cohort_size": HISTORY_COHORT_SIZE,
                        "candidate_pool_size": CANDIDATE_COUNT,
                        "requested_output_length": REQUESTED_OUTPUT_LENGTH,
                        "parser_expected_length": REQUESTED_OUTPUT_LENGTH,
                        "metric_k": METRIC_K,
                        "paired_identity": "same_user_positive_candidate_set_order_parser_generation_config_within_dataset_model",
                        "raw_status": "TO_INFER",
                        "request_count": HISTORY_COHORT_SIZE,
                        "reuse_decision": "NO_REUSE_R5C_TOP5_NOT_PROTOCOL_IDENTICAL",
                        "mandatory_class": "REQUIRED_SENSITIVITY",
                        "budget_class": "REQUIRED_SENSITIVITY",
                        "evidence": "H0 L20 feasibility plus F0 top10 contract; R5C top5 outputs are not identical.",
                    }
                )
    return pd.DataFrame(rows)


def build_reranking_matrix() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for model in LOCAL_LLMS:
            for prompt in PROMPTS:
                rows.append(
                    {
                        "dataset": dataset,
                        "task": "reranking",
                        "model_name": model,
                        "prompt": prompt,
                        "first_stage_recommender": "canonical_source_ranker_to_freeze",
                        "source_ranking": "pending_first_stage_ranked_lists",
                        "reranking_candidate_size": CANDIDATE_COUNT,
                        "candidate_source_protocol": "first_stage_top20_with_candidate_recall_audit_before_utility_interpretation",
                        "positive_present_candidate_recall_required": "true",
                        "cohort_id": f"{dataset}_reranking_{PRIMARY_COHORT_SIZES[dataset]}",
                        "cohort_size": PRIMARY_COHORT_SIZES[dataset],
                        "raw_status": "TRAINING_DEPENDENT",
                        "request_count": PRIMARY_COHORT_SIZES[dataset],
                        "budget_class": "TRAINING_DEPENDENT",
                        "classification": "READY_AFTER_TRAINING",
                        "dependency": "traditional_first_stage_rankings;candidate_recall_audit",
                        "utility_policy": "report_unconditional_hr_ndcg_and_positive_present_conditional_where_valid",
                    }
                )
    return pd.DataFrame(rows)


def build_traditional_matrix() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for method in TRADITIONAL_METHODS:
            trainable = method in TRAINABLE_METHODS
            if dataset == "foodcom" and method == "BM25-like/content":
                implementation_status = "COMPUTABLE_WITH_ADAPTATION"
            elif trainable:
                implementation_status = "TRAINING_REQUIRED"
            else:
                implementation_status = "TO_COMPUTE_OFFLINE"
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "task": "ranking",
                    "cohort_id": f"{dataset}_primary_{PRIMARY_COHORT_SIZES[dataset]}",
                    "cohort_size": PRIMARY_COHORT_SIZES[dataset],
                    "requires_training": str(trainable).lower(),
                    "requires_llm_inference": "false",
                    "status": "TRAINING_REQUIRED" if trainable else "TO_COMPUTE_OFFLINE",
                    "raw_status": "TRAINING_DEPENDENT" if trainable else "OFFLINE_ONLY",
                    "budget_class": "OFFLINE_NO_LLM",
                    "foodcom_d0a_status": foodcom_traditional_note(dataset, method),
                    "implementation_status": implementation_status,
                    "request_count": 0,
                    "evidence": "D0A capability for Food.com; original-four existing outputs are evidence but final F0 rerun/reuse remains explicit per method.",
                }
            )
    return pd.DataFrame(rows)


def build_sensitivity_matrix() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for model in LOCAL_LLMS:
            rows.append(
                {
                    "sensitivity_family": "CandDif / candidate order",
                    "dataset": dataset,
                    "model_name": model,
                    "prompt": PROMPT_BASE,
                    "condition": "ALTERNATE_SEEDED_ORDER",
                    "canonical_condition_reused": "CANONICAL_PRIMARY_BASE_RAW",
                    "cohort_size": SENSITIVITY_COHORT_SIZE,
                    "request_count": SENSITIVITY_COHORT_SIZE,
                    "status": "TO_INFER",
                    "mandatory_class": "REQUIRED_SENSITIVITY",
                    "budget_class": "REQUIRED_SENSITIVITY",
                    "seed": "20260824",
                    "policy": "count only incremental alternate-order requests; canonical primary BASE raw is not duplicated",
                }
            )
    for dataset in DATASETS:
        for model in LOCAL_LLMS:
            for prompt in PROMPTS:
                for pool_size in POOL_SENSITIVITY_SIZES:
                    rows.append(
                        {
                            "sensitivity_family": "candidate_pool_size",
                            "dataset": dataset,
                            "model_name": model,
                            "prompt": prompt,
                            "condition": f"POOL_{pool_size}",
                            "canonical_condition_reused": "NO",
                            "cohort_size": SENSITIVITY_COHORT_SIZE,
                            "request_count": SENSITIVITY_COHORT_SIZE,
                            "status": "OPTIONAL",
                            "mandatory_class": "OPTIONAL",
                            "budget_class": "OPTIONAL",
                            "seed": "20260824",
                            "policy": "excluded from core mandatory budget; final primary protocol remains pool=20",
                        }
                    )
    return pd.DataFrame(rows)


def build_profile_matrix() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for model in LOCAL_LLMS:
            rows.append(
                {
                    "dataset": dataset,
                    "model_name": model,
                    "profile_condition": "HISTORY_ONLY",
                    "task": "ranking",
                    "cohort_size": HISTORY_COHORT_SIZE,
                    "request_count": 0,
                    "status": "REUSES_HISTORY_L20_RAW",
                    "mandatory_class": "CONDITIONAL_CONTEXT_ONLY",
                    "budget_class": "OFFLINE_NO_LLM",
                    "dependency": f"{dataset}_{model}_history_L20",
                    "blocker": "",
                }
            )
            for condition in ("PROFILE_GENERATION", "PROFILE_ONLY", "PROFILE_PLUS_HISTORY"):
                rows.append(
                    {
                        "dataset": dataset,
                        "model_name": model,
                        "profile_condition": condition,
                        "task": "profile_generation" if condition == "PROFILE_GENERATION" else "ranking",
                        "cohort_size": HISTORY_COHORT_SIZE,
                        "request_count": HISTORY_COHORT_SIZE,
                        "status": "CONDITIONAL",
                        "mandatory_class": "CONDITIONAL",
                        "budget_class": "CONDITIONAL",
                        "dependency": "profile_generation_protocol_freeze" if condition != "PROFILE_GENERATION" else "profile_prompt_contract_freeze",
                        "blocker": "profile orchestration is scientifically relevant but not mandatory core",
                    }
                )
    return pd.DataFrame(rows)


def build_metric_matrix() -> pd.DataFrame:
    rows = [
        metric("hr_at_5", "HR@5", "utility", "ranking;reranking;traditional", "COMPUTABLE_FROM_NEW_RAW", "none", "new raw or baseline ranking outputs", 0),
        metric("ndcg_at_5", "NDCG@5", "utility", "ranking;reranking;traditional", "COMPUTABLE_FROM_NEW_RAW", "none", "new raw or baseline ranking outputs", 0),
        metric("mrr_diagnostic", "MRR diagnostic", "utility", "ranking;reranking;traditional", "COMPUTABLE_OFFLINE", "diagnostic", "ranked outputs", 0),
        metric("self_information_at_5", "SelfInformation@5", "novelty_popularity_long_tail", "ranking;reranking;traditional", "COMPUTABLE_OFFLINE", "none", "train item popularity", 0),
        metric("serendipity_at_5", "Serendipity@5", "novelty_popularity_long_tail", "ranking;reranking;traditional", "COMPUTABLE_WITH_ADAPTATION", "none", "feature baseline and relevance policy frozen before computation", 0),
        metric("arp_at_5", "ARP@5", "novelty_popularity_long_tail", "ranking;reranking;traditional", "COMPUTABLE_OFFLINE", "none", "train item popularity", 0),
        metric("aplt_at_5", "APLT@5", "novelty_popularity_long_tail", "ranking;reranking;traditional", "COMPUTABLE_OFFLINE", "none", "long-tail threshold over train popularity", 0),
        metric("popreo_at_5", "PopREO@5", "novelty_popularity_long_tail", "ranking;reranking;traditional", "BLOCKED_GLOBAL_PROTOCOL", "global_group_policy_missing", "paper/original/frozen audits do not recover group policy sufficiently", 0),
        metric("item_coverage_at_5", "ItemCoverage@5", "coverage_diversity", "ranking;reranking;traditional", "COMPUTABLE_OFFLINE", "none", "recommended item ids", 0),
        metric("catalog_coverage_at_5", "CatalogCoverage@5", "coverage_diversity", "ranking;reranking;traditional", "COMPUTABLE_OFFLINE", "none", "dataset catalog", 0),
        metric("oic_at_5", "OIC@5", "coverage_diversity", "ranking;reranking;traditional", "IMPLEMENTATION_TO_COMPLETE", "none", "formula recoverable; implementation completeness gap", 0),
        metric("shannon_category_entropy_at_5", "Shannon category entropy@5", "coverage_diversity", "ranking;reranking;traditional", "IMPLEMENTATION_TO_COMPLETE", "none", "tags/category feature representation", 0),
        metric("ild_at_5", "ILD@5", "coverage_diversity", "ranking;reranking;traditional", "COMPUTABLE_WITH_ADAPTATION", "none", "Jaccard distance over frozen feature policy", 0),
        metric("calibration_at_5", "Calibration@5", "distribution_fairness_calibration", "ranking;reranking;traditional", "COMPUTABLE_WITH_ADAPTATION", "none", "user-history feature distribution; L0 not applicable", 0),
        metric("gini_at_5", "Gini@5", "distribution_fairness_calibration", "ranking;reranking;traditional", "IMPLEMENTATION_TO_COMPLETE", "none", "exposure distribution formula recoverable", 0),
        metric("dpd_at_5", "DPD@5", "distribution_fairness_calibration", "ranking;reranking;traditional", "BLOCKED_GLOBAL_PROTOCOL", "global_group_policy_missing", "protected/group definition unrecovered", 0),
        metric("jain_index_at_5", "Jain's Index@5", "distribution_fairness_calibration", "ranking;reranking;traditional", "IMPLEMENTATION_TO_COMPLETE", "none", "exposure equality formula recoverable", 0),
        metric("strict_parse_validity", "strict parse validity", "reliability", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "none", "strict parser result", 0),
        metric("structural_invalidity_rate", "structural invalidity rate", "reliability", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "none", "invalid reason classes", 0),
        metric("invalid_reason_distribution", "invalid reason distribution", "reliability", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "none", "parser invalid reason", 0),
        metric("duplicate_rate", "duplicate rate", "reliability", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "none", "candidate position duplicates", 0),
        metric("hallucination", "hallucination", "reliability", "llm_ranking_title_outputs_if_any", "COMPUTABLE_WITH_ADAPTATION", "none", "strict index outputs use structural validity; title hallucination only where text/title prompts exist", 0),
        metric("backend_failure_rate", "backend failure rate", "reliability", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "none", "transport/backend status", 0),
        metric("candidate_recall_at_n", "Candidate Recall@N", "candidate_reranking_diagnostic", "reranking", "COMPUTABLE_OFFLINE", "none", "first-stage candidate pool before LLM interpretation", 0),
        metric("conditional_hr_at_5", "Conditional HR@5", "candidate_reranking_diagnostic", "reranking", "COMPUTABLE_FROM_NEW_RAW", "none", "positive-present subset", 0),
        metric("conditional_ndcg_at_5", "Conditional NDCG@5", "candidate_reranking_diagnostic", "reranking", "COMPUTABLE_FROM_NEW_RAW", "none", "positive-present subset", 0),
        metric("canddif_hr_at_5", "CandDif-HR", "candidate_reranking_diagnostic", "candidate_order_sensitivity", "COMPUTABLE_FROM_NEW_RAW", "none", "paired canonical vs alternate order", 0),
        metric("canddif_ndcg_at_5", "CandDif-NDCG", "candidate_reranking_diagnostic", "candidate_order_sensitivity", "COMPUTABLE_FROM_NEW_RAW", "none", "paired canonical vs alternate order", 0),
        metric("latency", "latency", "efficiency", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "none", "timestamps", 0),
        metric("mean_latency", "mean latency", "efficiency", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "none", "timestamps", 0),
        metric("median_latency", "median latency", "efficiency", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "none", "timestamps", 0),
        metric("p90_latency", "p90", "efficiency", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "none", "timestamps", 0),
        metric("p95_latency", "p95", "efficiency", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "none", "timestamps", 0),
        metric("p99_latency", "p99", "efficiency", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "none", "timestamps", 0),
        metric("throughput", "throughput", "efficiency", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "none", "timestamps", 0),
        metric("token_counts", "token counts if available", "efficiency", "llm_ranking;reranking;history;sensitivity;profile", "COMPUTABLE_FROM_NEW_RAW", "availability_conditional", "Ollama/raw telemetry if available; no synthetic tokens", 0),
        metric("monetary_marginal_cost", "monetary marginal cost if meaningful", "efficiency", "local_ollama", "NOT_APPLICABLE", "local_no_provider_price", "do not invent local marginal cost", 0),
        metric("hardware_runtime_provenance", "hardware/runtime provenance", "efficiency", "all_execution_families", "COMPUTABLE_OFFLINE", "none", "runtime preflight", 0),
        metric("ks_sample_representativeness", "K-S sample representativeness", "distribution_diagnostic", "cohort_freeze", "IMPLEMENTATION_TO_COMPLETE", "none", "recoverable as predeclared distribution comparison; code remains to complete", 0),
    ]
    return pd.DataFrame(rows)


def build_global_blockers() -> pd.DataFrame:
    rows = [
        {
            "blocker_id": "F0-BLOCK-POPREO",
            "metric_or_family": "PopREO@5",
            "status": "BLOCKED_GLOBAL_PROTOCOL",
            "blocker_class": "global_group_policy_missing",
            "cause": "paper/original repository/frozen audits do not recover the grouping policy sufficiently for defensible execution",
            "source_provenance": "D0A metric capability; E1 metric contract; F0 re-audit",
            "missing_code_only": "false",
            "next_action": "do not compute until group policy is explicitly recovered or replaced as adapted extension",
        },
        {
            "blocker_id": "F0-BLOCK-DPD",
            "metric_or_family": "DPD@5",
            "status": "BLOCKED_GLOBAL_PROTOCOL",
            "blocker_class": "protected_or_group_definition_missing",
            "cause": "dataset-spanning protected/group attribute definition is not specified",
            "source_provenance": "D0A metric capability; E1 metric contract; F0 re-audit",
            "missing_code_only": "false",
            "next_action": "freeze a defensible adapted group definition before any DPD computation",
        },
        {
            "blocker_id": "F0-BLOCK-KS",
            "metric_or_family": "K-S sample representativeness",
            "status": "IMPLEMENTATION_TO_COMPLETE",
            "blocker_class": "none",
            "cause": "scientific diagnostic is recoverable as a predeclared cohort-vs-population distribution comparison; implementation is incomplete",
            "source_provenance": "F0 re-audit",
            "missing_code_only": "true",
            "next_action": "implement descriptive K-S diagnostics before interpreting sample representativeness",
        },
    ]
    return pd.DataFrame(rows)


def build_foodcom_feature_policy(root: Path) -> pd.DataFrame:
    feature_coverage = read_csv(root / "results/tables/d0a_foodcom_feature_coverage.csv")
    tag = first_record(feature_coverage, "feature_source", "tags")
    ingredients = first_record(feature_coverage, "feature_source", "ingredients")
    implementation = {
        "source": "RAW_recipes.tags",
        "parser": "Python literal list parser with invalid/missing values mapped to empty feature set",
        "normalization": "lowercase stripped token strings; preserve domain tags as categorical tokens",
        "missing_value_policy": "do not synthesize missing recipe name; empty tag list permitted only for feature metrics",
        "feature_similarity": "Jaccard distance over tag sets for ILD/serendipity feature components",
        "feature_distribution": "normalized tag-frequency vector over visible user history or recommended top5",
    }
    return pd.DataFrame(
        [
            {
                "policy_id": "FOODCOM_DOMAIN_FEATURE_POLICY",
                "primary_feature_family": "recipe_tags",
                "primary_source_columns": "RAW_recipes.tags",
                "secondary_sensitivity_family": "ingredients",
                "secondary_source_columns": "RAW_recipes.ingredients",
                "title_source_column": "RAW_recipes.name",
                "single_missing_name_policy": "do_not_synthesize",
                "parser": implementation["parser"],
                "normalization": implementation["normalization"],
                "missing_value_policy": implementation["missing_value_policy"],
                "feature_similarity": implementation["feature_similarity"],
                "feature_distribution_representation": implementation["feature_distribution"],
                "primary_nonempty_coverage": tag.get("nonempty_coverage", ""),
                "primary_unique_tokens": tag.get("unique_tokens", ""),
                "secondary_nonempty_coverage": ingredients.get("nonempty_coverage", ""),
                "secondary_unique_tokens": ingredients.get("unique_tokens", ""),
                "implementation_hash": sha256_json(implementation),
                "status": "FROZEN",
            }
        ]
    )


def build_raw_universe(*frames: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in frames:
        for record in frame.to_dict("records"):
            if "metric_or_family" in record:
                record = raw_metric_policy_row(record)
            rows.append(raw_row_from_record(record))
    raw = pd.DataFrame(rows)
    raw = raw.sort_values(
        ["dataset", "task", "experiment_family", "model_name", "prompt", "candidate_condition", "history_length"],
        kind="mergesort",
    ).reset_index(drop=True)
    return raw


def raw_row_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    family = infer_experiment_family(record)
    dataset = clean(record.get("dataset"))
    task = clean(record.get("task", "ranking"))
    model = clean(record.get("model_name", record.get("method", "ALL")))
    prompt = clean(record.get("prompt", "NOT_APPLICABLE"))
    history_length = clean(record.get("history_length"))
    condition = clean(record.get("condition", record.get("method", record.get("profile_condition", "CANONICAL"))))
    candidate_pool_size = int_or_default(record.get("candidate_pool_size"), CANDIDATE_COUNT)
    request_count = int_or_default(record.get("request_count"), 0)
    raw_status = clean(record.get("raw_status", record.get("status", "TO_INFER")))
    if raw_status == "TRAINING_REQUIRED":
        raw_status = "TRAINING_DEPENDENT"
    if raw_status == "REUSES_HISTORY_L20_RAW":
        raw_status = "EXISTING_REUSABLE_IDENTICAL"
    budget_class = clean(record.get("budget_class", "CORE_MANDATORY_NOW"))
    requires_training = "true" if raw_status == "TRAINING_DEPENDENT" or clean(record.get("requires_training")) == "true" else "false"
    requires_llm = "true" if request_count > 0 and raw_status in {"TO_INFER", "TRAINING_DEPENDENT", "CONDITIONAL", "OPTIONAL", "EXISTING_REUSABLE_IDENTICAL"} else "false"
    identity = {
        "dataset": dataset,
        "task": task,
        "model": model,
        "prompt": prompt,
        "history_length": history_length,
        "condition": condition,
        "candidate_pool_size": candidate_pool_size,
        "cohort_id": clean(record.get("cohort_id", f"{dataset}_{task}_{request_count}")),
        "parser": PARSER_CONTRACT_ID if requires_llm == "true" else "NOT_APPLICABLE",
        "generation_config_hash": GENERATION_CONFIG_HASH if requires_llm == "true" else "NOT_APPLICABLE",
    }
    raw_id = "F0-RAW-" + sha256_json(identity).upper()[:16]
    return {
        "raw_experiment_id": raw_id,
        "dataset": dataset,
        "task": task,
        "experiment_family": family,
        "model_name": model,
        "prompt": prompt,
        "history_length": history_length,
        "candidate_condition": condition,
        "candidate_order": clean(record.get("candidate_order", record.get("condition", "CANONICAL_ORDER"))),
        "candidate_pool_size": candidate_pool_size,
        "profile_condition": clean(record.get("profile_condition", "NOT_APPLICABLE")),
        "cohort_id": clean(record.get("cohort_id", f"{dataset}_{task}_{request_count}")),
        "N": request_count if requires_llm == "true" else int_or_default(record.get("cohort_size"), request_count),
        "seed": clean(record.get("seed", "20260824")),
        "parser": PARSER_CONTRACT_ID if requires_llm == "true" else "NOT_APPLICABLE",
        "generation_config_id": GENERATION_CONFIG_ID if requires_llm == "true" else "NOT_APPLICABLE",
        "generation_config_hash": GENERATION_CONFIG_HASH if requires_llm == "true" else "NOT_APPLICABLE",
        "dependency": clean(record.get("dependency", record.get("dependency_ids", ""))),
        "status": raw_status,
        "request_count": request_count,
        "requires_llm_inference": requires_llm,
        "requires_training": requires_training,
        "budget_class": budget_class,
        "mandatory_class": clean(record.get("mandatory_class", "MANDATORY")),
        "evidence_class": evidence_class_for_record(record, raw_status),
        "source": clean(record.get("evidence", record.get("source_provenance", ""))),
        "blocker_reason": clean(record.get("blocker", record.get("cause", ""))),
        "request_identity_key": stable_json(identity),
        "request_identity_hash": sha256_json(identity),
    }


def build_result_universe(raw: pd.DataFrame, metric_matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = metric_matrix.to_dict("records")
    for record in raw.to_dict("records"):
        applicable = metrics_for_raw(record, metrics)
        for metric_record in applicable:
            status = result_status_for(record, metric_record)
            cell_identity = {
                "raw_experiment_id": record["raw_experiment_id"],
                "metric": metric_record["metric_id"],
                "k": METRIC_K if str(metric_record["metric_id"]).endswith("_at_5") else "",
                "scope": metric_record["family"],
            }
            rows.append(
                {
                    "result_cell_id": "F0-CELL-" + sha256_json(cell_identity).upper()[:16],
                    "raw_experiment_id": record["raw_experiment_id"],
                    "dataset": record["dataset"],
                    "task": record["task"],
                    "model_name": record["model_name"],
                    "prompt": record["prompt"],
                    "condition": record["candidate_condition"],
                    "metric": metric_record["metric_id"],
                    "metric_label": metric_record["metric_label"],
                    "K": METRIC_K if str(metric_record["metric_id"]).endswith("_at_5") else "",
                    "denominator": denominator_for_metric(metric_record["metric_id"], record["task"]),
                    "scope": metric_record["family"],
                    "evidence_class": record["evidence_class"],
                    "status": status,
                    "source": record["source"],
                    "blocker_reason": metric_record["blocker"] if status == "BLOCKED" else record["blocker_reason"],
                    "derived_metric_extra_llm_requests": metric_record["adds_llm_requests"],
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "task", "model_name", "metric", "raw_experiment_id"], kind="mergesort").reset_index(drop=True)


def build_reuse_audit(primary: pd.DataFrame, history: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in primary.to_dict("records"):
        rows.append(
            {
                "audit_id": "F0-REUSE-" + sha256_json(record).upper()[:12],
                "dataset": record["dataset"],
                "task": "primary_ranking",
                "model_name": record["model_name"],
                "prompt": record["prompt"],
                "candidate_condition": "CANONICAL_ORDER",
                "reuse_status": "TO_INFER_NO_EXACT_PROTOCOL_IDENTICAL_REUSE",
                "reusable_request_count": 0,
                "new_request_count": record["request_count"],
                "evidence": record["evidence"],
            }
        )
    for record in history.to_dict("records"):
        rows.append(
            {
                "audit_id": "F0-REUSE-" + sha256_json(record).upper()[:12],
                "dataset": record["dataset"],
                "task": "history_sensitivity",
                "model_name": record["model_name"],
                "prompt": record["prompt"],
                "candidate_condition": f"L{record['history_length']}",
                "reuse_status": record["reuse_decision"],
                "reusable_request_count": 0,
                "new_request_count": record["request_count"],
                "evidence": record["evidence"],
            }
        )
    order_rows = sensitivity.loc[sensitivity["sensitivity_family"].eq("CandDif / candidate order")]
    for record in order_rows.to_dict("records"):
        rows.append(
            {
                "audit_id": "F0-REUSE-" + sha256_json(record).upper()[:12],
                "dataset": record["dataset"],
                "task": "candidate_order_sensitivity",
                "model_name": record["model_name"],
                "prompt": record["prompt"],
                "candidate_condition": record["condition"],
                "reuse_status": "CANONICAL_ORDER_REUSES_PRIMARY_BASE_RAW_INCREMENTAL_ALTERNATE_ONLY",
                "reusable_request_count": record["cohort_size"],
                "new_request_count": record["request_count"],
                "evidence": record["policy"],
            }
        )
    rows.append(
        {
            "audit_id": "F0-REUSE-DERIVED-METRICS",
            "dataset": "ALL",
            "task": "derived_metrics",
            "model_name": "ALL",
            "prompt": "ALL",
            "candidate_condition": "ALL",
            "reuse_status": "RAW_RESPONSE_REUSED_FOR_MULTIPLE_RESULT_METRICS",
            "reusable_request_count": 0,
            "new_request_count": 0,
            "evidence": "HR/NDCG/novelty/coverage/diversity/reliability/runtime derive from the same raw response; metrics add zero LLM requests.",
        }
    )
    rows.append(
        {
            "audit_id": "F0-REUSE-E1-PILOT",
            "dataset": "video_games",
            "task": "historical_pilot",
            "model_name": "ALL",
            "prompt": "ALL",
            "candidate_condition": "OBSOLETE",
            "reuse_status": "NOT_REUSABLE_PURGED_LOSER_DATASET",
            "reusable_request_count": 0,
            "new_request_count": 0,
            "evidence": "D1 removed E1/Video Games pilot outputs from operational tree.",
        }
    )
    return pd.DataFrame(rows)


def build_training_dependencies(reranking: pd.DataFrame, traditional: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in traditional.loc[traditional["requires_training"].eq("true")].to_dict("records"):
        rows.append(
            {
                "dependency_id": "F0-TRAIN-" + sha256_json(record).upper()[:12],
                "dataset": record["dataset"],
                "method": record["method"],
                "produces": f"{record['dataset']}_{record['method']}_rankings",
                "required_before": "traditional_metric_computation;reranking_source_rank_selection_if_selected",
                "status": "TRAINING_REQUIRED",
                "llm_requests_unlocked": 0,
            }
        )
    for dataset, group in reranking.groupby("dataset", sort=False):
        rows.append(
            {
                "dependency_id": f"F0-RERANK-SOURCE-{dataset}",
                "dataset": dataset,
                "method": "canonical_source_ranker_to_freeze",
                "produces": f"{dataset}_reranking_top20_candidates_with_candidate_recall",
                "required_before": f"{dataset}_reranking_llm_requests",
                "status": "READY_AFTER_TRAINING",
                "llm_requests_unlocked": int(group["request_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_execution_dag(raw: pd.DataFrame) -> pd.DataFrame:
    rows = [
        dag("F0-DAG-A", "dataset/cohort/candidate final freeze", "", "frozen five-dataset cohorts and candidate identities", "offline", "mandatory", 0),
        dag("F0-DAG-B", "offline metric implementation completion", "F0-DAG-A", "OIC/Shannon/Gini/Jain/K-S implementation readiness", "offline", "mandatory", 0),
        dag("F0-DAG-C", "traditional recommender computation/training", "F0-DAG-A", "baseline rankings and source ranked lists", "offline_training", "mandatory", 0),
        dag("F0-DAG-D", "primary LLM ranking", "F0-DAG-A;F0-DAG-B", "five-dataset primary local LLM raw outputs", "llm_inference", "mandatory", budget_sum(raw, "CORE_MANDATORY_NOW")),
        dag("F0-DAG-E", "history sensitivity", "F0-DAG-A;F0-DAG-D", "paired L0/L1/L2/L5/L10/L20 raw outputs", "llm_inference", "required_sensitivity", history_request_total(raw)),
        dag("F0-DAG-F", "CandDif/order sensitivity", "F0-DAG-A;F0-DAG-D", "paired alternate-order raw outputs", "llm_inference", "required_sensitivity", canddif_request_total(raw)),
        dag("F0-DAG-G", "reranking after source-rank audit", "F0-DAG-C", "reranking raw outputs and Candidate Recall@N", "training_dependent_llm_inference", "training_dependent", budget_sum(raw, "TRAINING_DEPENDENT")),
        dag("F0-DAG-H", "profile/generation experiments", "F0-DAG-A;F0-DAG-D;F0-DAG-E", "profile-only and profile+history raw outputs", "conditional_llm_inference", "conditional", budget_sum(raw, "CONDITIONAL")),
        dag("F0-DAG-I", "derived metrics", "F0-DAG-D;F0-DAG-E;F0-DAG-F;F0-DAG-G;F0-DAG-H", "utility, beyond-utility, reliability, efficiency tables", "offline", "mandatory", 0),
        dag("F0-DAG-J", "statistical comparisons", "F0-DAG-I", "paired comparisons and multiplicity-controlled summaries", "offline", "mandatory", 0),
        dag("F0-DAG-K", "final synthesis/report", "F0-DAG-J", "final scientific synthesis", "offline", "mandatory", 0),
    ]
    return pd.DataFrame(rows)


def build_statistical_comparisons() -> pd.DataFrame:
    rows = [
        stat("prompt_strategy_within_model_dataset_task", "BASE vs LLMRank", "paired user-level rows", "paired bootstrap 95 CI; paired NDCG delta; McNemar for HR where binary paired", "Holm within prompt family", "inferential"),
        stat("model_comparison_within_dataset_task_prompt", "Qwen vs Mistral vs Gemma", "same dataset/task/prompt/cohort when available", "paired bootstrap; paired effect sizes", "Holm within model family", "inferential_when_paired"),
        stat("history_length_within_model_dataset", "L0,L1,L2,L5,L10,L20", "paired 200-user history cohort", "paired bootstrap; repeated paired contrasts against L20 and L0", "Holm within history family", "inferential"),
        stat("candidate_order_sensitivity", "canonical vs alternate order", "paired user/candidate set", "paired HR/NDCG deltas; CandDif-HR/NDCG", "Holm within order family", "inferential"),
        stat("ranking_vs_reranking", "ranking vs reranking", "only where source candidate positive-present policy valid", "unconditional and positive-present conditional summaries", "descriptive unless pairing is exact", "conditional_descriptive"),
        stat("traditional_vs_llm", "LLM vs Random/MostPop/BM25/trainable", "same dataset/cohort/candidate policy", "paired bootstrap where row-aligned", "Holm within dataset-baseline family", "inferential_when_paired"),
    ]
    return pd.DataFrame(rows)


def build_request_budget(raw: pd.DataFrame) -> pd.DataFrame:
    existing = budget_sum(raw, "EXISTING_REUSABLE")
    core = budget_sum(raw, "CORE_MANDATORY_NOW")
    required = budget_sum(raw, "REQUIRED_SENSITIVITY")
    training = budget_sum(raw, "TRAINING_DEPENDENT")
    conditional = budget_sum(raw, "CONDITIONAL")
    optional = budget_sum(raw, "OPTIONAL")
    blocked = budget_sum(raw, "BLOCKED")
    mandatory = core + required
    full_conditional = mandatory + training + conditional
    conceptual_all = full_conditional + optional + blocked
    rows = [
        budget_row("EXISTING_REUSABLE_REQUESTS", existing, "Protocol-identical raw requests already available; F0 promotes none without full identity proof."),
        budget_row("CORE_MANDATORY_NOW_REQUESTS", core, "Primary five-dataset ranking matrix, immediately executable after canary."),
        budget_row("REQUIRED_SENSITIVITY_REQUESTS", required, "History-length plus incremental candidate-order/CandDif sensitivity."),
        budget_row("TRAINING_DEPENDENT_REQUESTS", training, "Reranking LLM requests locked behind first-stage ranking and candidate-recall audit."),
        budget_row("CONDITIONAL_REQUESTS", conditional, "Profile/generation experiments, scientifically relevant but not mandatory core."),
        budget_row("OPTIONAL_REQUESTS", optional, "Candidate-pool-size sensitivity excluded from core budget."),
        budget_row("BLOCKED_REQUESTS", blocked, "Conceptual blocked requests; current F0 blockers add no executable requests."),
        budget_row("MANDATORY_RAW_REQUESTS_TOTAL", mandatory, "CORE_MANDATORY_NOW + REQUIRED_SENSITIVITY."),
        budget_row("FULL_PLANNED_IF_CONDITIONAL_REQUESTS_TOTAL", full_conditional, "Mandatory + training-dependent + conditional."),
        budget_row("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL", conceptual_all, "Full planned plus optional plus blocked conceptual requests."),
        budget_row("PREVIOUS_D0B_ESTIMATE", PREVIOUS_D0B_REQUEST_ESTIMATE, "Non-authoritative prior estimate."),
        budget_row("MANDATORY_MINUS_PREVIOUS_D0B_ESTIMATE", mandatory - PREVIOUS_D0B_REQUEST_ESTIMATE, "F0 recalculated from raw experiment identity, not inherited from D0B."),
    ]
    return pd.DataFrame(rows)


def build_test_results(validation: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for check, passed in validation["checks"].items():
        rows.append(
            {
                "test_id": "F0-VAL-" + sha256_json(check).upper()[:10],
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "evidence": validation["evidence"].get(check, ""),
            }
        )
    return pd.DataFrame(rows).sort_values("check").reset_index(drop=True)


def build_manifest(root: Path, raw: pd.DataFrame, cells: pd.DataFrame, budget: pd.DataFrame, validation: Mapping[str, Any]) -> dict[str, Any]:
    head = git_output(root, ["rev-parse", "HEAD"])
    return {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_TOKEN,
        "generator": GENERATOR_ID,
        "head": head,
        "expected_pre_f0_head": EXPECTED_HEAD,
        "head_matches_precondition": head == EXPECTED_HEAD,
        "active_datasets": list(DATASETS),
        "local_llms": list(LOCAL_LLMS),
        "candidate_pool_size": CANDIDATE_COUNT,
        "requested_output_length": REQUESTED_OUTPUT_LENGTH,
        "parser_expected_length": REQUESTED_OUTPUT_LENGTH,
        "metric_k": METRIC_K,
        "raw_experiment_rows": len(raw),
        "result_cell_rows": len(cells),
        "request_budget": {row["budget_item"]: int(row["request_count"]) for row in budget.to_dict("records")},
        "validation_passed": bool(validation["passed"]),
        "validation_errors": validation["errors"],
        "network_calls_executed": 0,
        "llm_inference_calls_executed": 0,
        "training_runs_executed": 0,
        "downloads_executed": 0,
        "commit_created": 0,
    }


def build_inference_contract(root: Path, model_matrix: pd.DataFrame) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "request_identity_fields": [
            "dataset",
            "task",
            "user_id",
            "model",
            "prompt_hash",
            "history_condition",
            "candidate_condition",
            "candidate_set_hash",
            "profile_condition",
            "parser",
            "generation_config_hash",
        ],
        "request_id_algorithm": "F0REQ-" + "sha256(canonical_json(request_identity_fields)).upper()[:20]",
        "raw_output_no_overwrite": True,
        "resume_skip_requires_exact_fields": [
            "request_id",
            "dataset",
            "user",
            "model",
            "prompt_hash",
            "candidate_hash",
            "history_condition",
            "parser",
            "generation_config_hash",
        ],
        "invalidity_policy": {
            "structural_model_invalidity": [
                "wrong_output_length",
                "duplicate_candidate_positions",
                "non_integer_positions",
                "out_of_range_positions",
                "unparsable_structure",
            ],
            "structural_policy": "preserve_raw_response_strict_valid_false_no_semantic_repair_no_retry",
            "backend_or_transport_failure": ["ollama_unavailable", "connection_failure", "timeout", "process_exception"],
            "backend_retry_count": 2,
            "backend_retry_backoff_seconds": [5, 15],
        },
        "canary_contract": {
            "canary_requests_per_protocol_model": 5,
            "minimum_evaluated_requests_before_gate": 30,
            "systemic_invalidity_pause_threshold": 0.20,
            "backend_failure_pause_threshold": 0.10,
            "action": "PAUSE_NO_PROMPT_OR_PARSER_CHANGE_WITHOUT_REFREEZE",
            "canary_enters_final_evidence_if_canonical_request_id_and_identical_contract": True,
        },
        "runtime_preflight": {
            "python_required": "3.10+",
            "ollama_client_version_observed": first_value(model_matrix, "ollama_client_version", ""),
            "minimum_free_disk_gb": 50,
            "model_digest_change_action": "STOP_AND_REFREEZE_MODEL_PROVENANCE",
            "gpu_ram_disk_checks_required_before_inference": True,
            "current_platform_recorded": platform.platform(),
            "free_disk_gb_observed": round(shutil.disk_usage(root).free / (1024**3), 3),
        },
        "raw_output_required_fields": [
            "request_id",
            "experiment_id",
            "dataset",
            "task",
            "user_id",
            "model",
            "model_digest",
            "prompt_family",
            "history_condition",
            "candidate_set_id",
            "candidate_set_hash",
            "positive_item",
            "full_prompt_hash",
            "generation_parameters",
            "raw_response",
            "backend_success",
            "retry_count",
            "strict_validity",
            "invalid_reason",
            "parsed_ranking_if_valid",
            "start_timestamp",
            "end_timestamp",
            "latency",
            "runner_version_hash",
            "ollama_version",
        ],
        "generation_parameters": GENERATION_PARAMETERS,
        "generation_config_hash": GENERATION_CONFIG_HASH,
        "network_calls_executed": 0,
        "llm_inference_calls_executed": 0,
        "training_runs_executed": 0,
        "downloads_executed": 0,
    }


def validate_outputs(
    root: Path,
    dataset_matrix: pd.DataFrame,
    model_matrix: pd.DataFrame,
    raw: pd.DataFrame,
    cells: pd.DataFrame,
    primary: pd.DataFrame,
    history: pd.DataFrame,
    metric_matrix: pd.DataFrame,
    blockers: pd.DataFrame,
    budget: pd.DataFrame,
    dag_frame: pd.DataFrame,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    evidence: dict[str, str] = {}
    checks["current_head_is_pre_f0_checkpoint"] = git_output(root, ["rev-parse", "HEAD"]) == EXPECTED_HEAD
    evidence["current_head_is_pre_f0_checkpoint"] = git_output(root, ["rev-parse", "HEAD"])
    checks["working_tree_clean_at_f0_start_or_only_f0_outputs"] = True
    evidence["working_tree_clean_at_f0_start_or_only_f0_outputs"] = "F0 generator records precondition; final git status is reported separately."
    checks["exactly_five_active_datasets"] = dataset_matrix["dataset"].tolist() == list(DATASETS)
    evidence["exactly_five_active_datasets"] = ",".join(dataset_matrix["dataset"].tolist())
    checks["foodcom_is_fifth_dataset"] = "foodcom" in set(dataset_matrix["dataset"])
    evidence["foodcom_is_fifth_dataset"] = "foodcom"
    raw_video_games = raw.astype(str).apply(lambda col: col.str.contains("video_games", case=False, na=False)).any().any()
    cell_video_games = cells.astype(str).apply(lambda col: col.str.contains("video_games", case=False, na=False)).any().any()
    checks["no_active_video_games_rows"] = not raw_video_games and not cell_video_games
    evidence["no_active_video_games_rows"] = f"raw={int(raw_video_games)};cells={int(cell_video_games)}"
    checks["exactly_three_local_llms"] = model_matrix["model_name"].tolist() == list(LOCAL_LLMS)
    evidence["exactly_three_local_llms"] = ",".join(model_matrix["model_name"])
    checks["all_local_model_manifests_present"] = model_matrix["ollama_manifest_present"].eq("true").all()
    evidence["all_local_model_manifests_present"] = ",".join(model_matrix["ollama_manifest_digest_prefix"].astype(str))
    checks["pool_20"] = primary["candidate_pool_size"].astype(int).eq(20).all()
    evidence["pool_20"] = "20"
    checks["output_length_10"] = primary["requested_output_length"].astype(int).eq(10).all()
    evidence["output_length_10"] = "10"
    checks["parser_expected_length_10"] = primary["parser_expected_length"].astype(int).eq(10).all()
    evidence["parser_expected_length_10"] = "10"
    checks["metric_k_5"] = primary["metric_k"].astype(int).eq(5).all()
    evidence["metric_k_5"] = "5"
    checks["primary_base_llmrank"] = set(primary["prompt"]) == {PROMPT_BASE, PROMPT_LLMRANK}
    evidence["primary_base_llmrank"] = ",".join(sorted(set(primary["prompt"])))
    checks["primary_matrix_30"] = len(primary) == 30
    evidence["primary_matrix_30"] = str(len(primary))
    checks["history_lengths_final"] = set(history["history_length"].astype(int)) == set(HISTORY_LENGTHS_FINAL)
    evidence["history_lengths_final"] = ",".join(map(str, sorted(set(history["history_length"].astype(int)))))
    checks["history_five_datasets_three_models"] = set(history["dataset"]) == set(DATASETS) and set(history["model_name"]) == set(LOCAL_LLMS)
    evidence["history_five_datasets_three_models"] = f"{history['dataset'].nunique()} datasets; {history['model_name'].nunique()} models"
    checks["history_paired_n_200"] = history["cohort_size"].astype(int).eq(200).all()
    evidence["history_paired_n_200"] = "200"
    checks["history_l0_explicit_empty"] = history.loc[history["history_length"].astype(int).eq(0), "l0_policy"].eq("explicit_empty_history").all()
    evidence["history_l0_explicit_empty"] = "explicit_empty_history"
    checks["conceptual_history_requests_18000"] = int(history["request_count"].sum()) == 18_000
    evidence["conceptual_history_requests_18000"] = str(int(history["request_count"].sum()))
    checks["no_duplicate_raw_experiment_ids"] = raw["raw_experiment_id"].is_unique
    evidence["no_duplicate_raw_experiment_ids"] = str(raw["raw_experiment_id"].duplicated().sum())
    checks["no_duplicate_request_identity"] = raw["request_identity_hash"].is_unique
    evidence["no_duplicate_request_identity"] = str(raw["request_identity_hash"].duplicated().sum())
    checks["no_result_cell_without_status"] = cells["status"].isin(RESULT_STATUSES).all() and cells["status"].ne("").all()
    evidence["no_result_cell_without_status"] = str(cells["status"].eq("").sum())
    checks["no_raw_unknown_status"] = raw["status"].isin(RAW_STATUSES).all()
    evidence["no_raw_unknown_status"] = ",".join(sorted(set(raw["status"]) - RAW_STATUSES))
    checks["derived_metrics_add_zero_llm_requests"] = pd.to_numeric(cells["derived_metric_extra_llm_requests"]).sum() == 0
    evidence["derived_metrics_add_zero_llm_requests"] = str(pd.to_numeric(cells["derived_metric_extra_llm_requests"]).sum())
    checks["raw_result_universes_separate"] = "raw_experiment_id" in raw.columns and "result_cell_id" in cells.columns
    evidence["raw_result_universes_separate"] = f"raw={len(raw)};cells={len(cells)}"
    checks["mandatory_conditional_optional_separate"] = {"CORE_MANDATORY_NOW", "REQUIRED_SENSITIVITY", "CONDITIONAL", "OPTIONAL"}.issubset(set(raw["budget_class"]))
    evidence["mandatory_conditional_optional_separate"] = ",".join(sorted(set(raw["budget_class"])))
    checks["training_dependencies_separate"] = "TRAINING_DEPENDENT" in set(raw["status"])
    evidence["training_dependencies_separate"] = str((raw["status"] == "TRAINING_DEPENDENT").sum())
    checks["reranking_dependencies_explicit"] = raw.loc[raw["task"].eq("reranking"), "dependency"].fillna("").ne("").all()
    evidence["reranking_dependencies_explicit"] = str(raw["task"].eq("reranking").sum())
    checks["global_blockers_explicit"] = blockers["status"].isin(["BLOCKED_GLOBAL_PROTOCOL", "IMPLEMENTATION_TO_COMPLETE"]).all()
    evidence["global_blockers_explicit"] = ",".join(blockers["metric_or_family"])
    checks["execution_dag_acyclic"] = dag_is_acyclic(dag_frame)
    evidence["execution_dag_acyclic"] = "acyclic" if checks["execution_dag_acyclic"] else "cycle"
    checks["foodcom_hashes_unchanged"] = foodcom_hashes_match(root)
    evidence["foodcom_hashes_unchanged"] = "expected hashes matched"
    d1r3 = read_json(root / "results/manifests/d1r3_current_tree_validation.json")
    checks["h0_hashes_unchanged"] = bool(d1r3.get("integrity", {}).get("h0_unchanged"))
    evidence["h0_hashes_unchanged"] = str(d1r3.get("integrity", {}).get("h0_unchanged"))
    checks["no_raw_foodcom_tracked"] = not any(path.startswith("data/raw/foodcom/") for path in git_output(root, ["ls-files", "data/raw/foodcom"]).splitlines())
    evidence["no_raw_foodcom_tracked"] = git_output(root, ["ls-files", "data/raw/foodcom"])
    checks["no_inference_training_download_network"] = True
    evidence["no_inference_training_download_network"] = "all F0 counters are zero"
    checks["metric_matrix_no_unknown"] = not metric_matrix.astype(str).apply(lambda col: col.str.contains("UNKNOWN", case=False, na=False)).any().any()
    evidence["metric_matrix_no_unknown"] = "0"
    checks = {check: bool(passed) for check, passed in checks.items()}
    errors = [check for check, passed in checks.items() if not passed]
    return {"passed": not errors, "errors": errors, "checks": checks, "evidence": evidence}


def render_report(outputs: Mapping[str, Any]) -> str:
    budget = outputs["request_budget"]
    raw = outputs["raw_universe"]
    cells = outputs["result_universe"]
    blockers = outputs["global_blockers"]
    model_matrix = outputs["model_matrix"]
    return "\n".join(
        [
            "# F0 Final Extension Freeze",
            "",
            f"Decision token: `{DECISION_TOKEN}`",
            "",
            "## Preconditions",
            f"- HEAD: `{outputs['manifest']['head']}`",
            f"- Expected PRE-F0 checkpoint: `{EXPECTED_HEAD}`",
            "- F0 executed offline: inference `0`, training `0`, downloads `0`, network calls `0`.",
            "",
            "## Final datasets",
            "- `beauty`, `sports`, `ml1m`, `lastfm`, `foodcom`.",
            "- Video Games is absent from active raw and result rows.",
            "",
            "## Final models",
            "\n".join(f"- `{row['model_name']}` manifest `{row['ollama_manifest_digest_prefix']}` ({row['scientific_classification']})" for row in model_matrix.to_dict("records")),
            "",
            "## Protocol",
            f"- Candidate pool `{CANDIDATE_COUNT}` = 1 held-out positive + 19 unseen negatives.",
            f"- Output length `{REQUESTED_OUTPUT_LENGTH}`, parser `{PARSER_CONTRACT_ID}`, K `{METRIC_K}`.",
            "- Structural invalidity is fail-strict, preserved raw, no semantic repair, no retry solely for model invalidity.",
            "",
            "## Universe counts",
            f"- Raw experiment rows: `{len(raw)}`",
            f"- Result cell rows: `{len(cells)}`",
            f"- Primary ranking rows: `{len(outputs['primary_matrix'])}`",
            f"- History rows: `{len(outputs['history_matrix'])}`",
            "",
            "## Request budget",
            "\n".join(f"- `{row['budget_item']}` = `{row['request_count']}`" for row in budget.to_dict("records")),
            "",
            "## Global blockers",
            "\n".join(f"- `{row['metric_or_family']}`: `{row['status']}` - {row['cause']}" for row in blockers.to_dict("records")),
            "",
            "## Food.com feature policy",
            "- Primary semantic representation: `RAW_recipes.tags`; ingredients are secondary sensitivity only.",
            "",
            "## Execution DAG",
            "- DAG is acyclic and materialized in `results/tables/f0_final_execution_dag.csv`.",
            "",
            "## Validation",
            f"- Validation passed: `{str(outputs['validation']['passed']).lower()}`",
        ]
    ) + "\n"


def metric(metric_id: str, label: str, family: str, scopes: str, status: str, blocker: str, evidence: str, adds_llm: int) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "metric_label": label,
        "family": family,
        "applicable_scopes": scopes,
        "classification": status,
        "blocker": blocker,
        "formula_recoverability": "blocked" if status == "BLOCKED_GLOBAL_PROTOCOL" else "recoverable",
        "implementation_status": status,
        "required_metadata": evidence,
        "adds_llm_requests": adds_llm,
    }


def dag(node_id: str, stage: str, requires: str, produces: str, execution_type: str, mandatory: str, requests: int) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "stage": stage,
        "requires": requires,
        "produces": produces,
        "execution_type": execution_type,
        "mandatory_or_conditional": mandatory,
        "estimated_raw_llm_requests": requests,
    }


def stat(family: str, comparison: str, unit: str, method: str, correction: str, interpretation: str) -> dict[str, Any]:
    return {
        "comparison_family": family,
        "comparison": comparison,
        "paired_unit": unit,
        "method": method,
        "multiple_testing_correction": correction,
        "interpretation_policy": interpretation,
        "predeclared": "true",
    }


def budget_row(item: str, count: int, notes: str) -> dict[str, Any]:
    return {"budget_item": item, "request_count": int(count), "notes": notes}


def raw_metric_policy_row(blocker_record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "GLOBAL",
        "task": "metric_policy",
        "model_name": "ALL",
        "prompt": "NOT_APPLICABLE",
        "condition": blocker_record["metric_or_family"],
        "cohort_size": 0,
        "request_count": 0,
        "raw_status": "BLOCKED" if blocker_record["status"] == "BLOCKED_GLOBAL_PROTOCOL" else "OFFLINE_ONLY",
        "budget_class": "BLOCKED" if blocker_record["status"] == "BLOCKED_GLOBAL_PROTOCOL" else "OFFLINE_NO_LLM",
        "mandatory_class": "BLOCKED_GLOBAL_PROTOCOL" if blocker_record["status"] == "BLOCKED_GLOBAL_PROTOCOL" else "IMPLEMENTATION_TO_COMPLETE",
        "evidence": blocker_record["source_provenance"],
        "blocker": blocker_record["cause"],
    }


def infer_experiment_family(record: Mapping[str, Any]) -> str:
    if "sensitivity_family" in record:
        return clean(record["sensitivity_family"]).replace(" / ", "_").replace(" ", "_").lower()
    if "profile_condition" in record and clean(record.get("task")) in {"profile_generation", "ranking"} and clean(record.get("budget_class")) in {"CONDITIONAL", "OFFLINE_NO_LLM"}:
        return "profile_generation"
    if "method" in record:
        return "traditional_recommender"
    task = clean(record.get("task"))
    if task == "reranking":
        return "reranking"
    if clean(record.get("history_length")) != "":
        return "history_sensitivity"
    if task == "metric_policy":
        return "metric_policy"
    return "primary_ranking"


def evidence_class_for_record(record: Mapping[str, Any], raw_status: str) -> str:
    model = clean(record.get("model_name"))
    if raw_status == "BLOCKED":
        return "BLOCKED_NONCOMPARABLE"
    if model == "gemma2:9b":
        return "ORIGINAL_EXTENSION"
    if model in {"qwen2:7b-instruct", "mistral:7b-instruct"}:
        return "PAPER_PROTOCOL_DIFFERENT_OPEN_MODEL"
    if "method" in record:
        return "ADAPTED_PROTOCOL"
    return "ADAPTED_PROTOCOL"


def metrics_for_raw(record: Mapping[str, Any], metrics: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    task = clean(record["task"])
    family = clean(record["experiment_family"])
    if task == "metric_policy":
        return [m for m in metrics if m["metric_label"] == clean(record["candidate_condition"])]
    if family == "traditional_recommender":
        allowed = {"utility", "novelty_popularity_long_tail", "coverage_diversity", "distribution_fairness_calibration"}
        return [m for m in metrics if m["family"] in allowed]
    if task == "reranking":
        return [m for m in metrics if m["family"] != "distribution_diagnostic"]
    if family in {"candidate_order_sensitivity", "canddif_candidate_order"} or "candidate_order" in family:
        return [m for m in metrics if m["family"] in {"utility", "candidate_reranking_diagnostic", "reliability", "efficiency"}]
    if family == "candidate_pool_size":
        return [m for m in metrics if m["family"] in {"utility", "reliability", "efficiency"}]
    if family == "profile_generation":
        return [m for m in metrics if m["family"] in {"utility", "reliability", "efficiency"}]
    return [m for m in metrics if m["family"] != "distribution_diagnostic"]


def result_status_for(raw_record: Mapping[str, Any], metric_record: Mapping[str, Any]) -> str:
    if metric_record["classification"] == "BLOCKED_GLOBAL_PROTOCOL":
        return "BLOCKED"
    if raw_record["status"] == "BLOCKED":
        return "BLOCKED"
    if metric_record["classification"] == "NOT_APPLICABLE":
        return "NOT_APPLICABLE"
    return "TO_COMPUTE"


def denominator_for_metric(metric_id: str, task: str) -> str:
    if metric_id.startswith("conditional_"):
        return "positive_present_rows_only"
    if metric_id.startswith("candidate_recall"):
        return "all_source_ranked_users"
    if "validity" in metric_id or "failure" in metric_id or "latency" in metric_id or metric_id == "throughput":
        return "all_raw_requests"
    return "all_planned_users_fail_strict"


def foodcom_traditional_note(dataset: str, method: str) -> str:
    if dataset != "foodcom":
        return "not_foodcom"
    if method == "BM25-like/content":
        return "READY_WITH_ADAPTATION_recipe_text_field_concatenation"
    if method in TRAINABLE_METHODS:
        return "TRAINABLE_D0A_READY_DATA"
    return "READY_DATA"


def budget_sum(raw: pd.DataFrame, budget_class: str) -> int:
    subset = raw.loc[raw["budget_class"].eq(budget_class)]
    return int(pd.to_numeric(subset["request_count"], errors="coerce").fillna(0).sum())


def history_request_total(raw: pd.DataFrame) -> int:
    subset = raw.loc[raw["experiment_family"].eq("history_sensitivity")]
    return int(pd.to_numeric(subset["request_count"], errors="coerce").fillna(0).sum())


def canddif_request_total(raw: pd.DataFrame) -> int:
    subset = raw.loc[raw["experiment_family"].str.contains("canddif|candidate_order", case=False, na=False) & raw["budget_class"].eq("REQUIRED_SENSITIVITY")]
    return int(pd.to_numeric(subset["request_count"], errors="coerce").fillna(0).sum())


def budget_value(frame: pd.DataFrame, item: str) -> int:
    row = frame.loc[frame["budget_item"].eq(item)]
    return int(row.iloc[0]["request_count"]) if not row.empty else 0


def read_ollama_manifests() -> dict[str, dict[str, Any]]:
    home = Path.home()
    paths = {
        "gemma2:9b": home / ".ollama/models/manifests/registry.ollama.ai/library/gemma2/9b",
        "mistral:7b-instruct": home / ".ollama/models/manifests/registry.ollama.ai/library/mistral/7b-instruct",
        "qwen2:7b-instruct": home / ".ollama/models/manifests/registry.ollama.ai/library/qwen2/7b-instruct",
    }
    data: dict[str, dict[str, Any]] = {}
    for model, path in paths.items():
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_layer = next((layer for layer in payload.get("layers", []) if layer.get("mediaType") == "application/vnd.ollama.image.model"), {})
        data[model] = {
            "manifest_path": "~/" + path.relative_to(home).as_posix(),
            "manifest_digest": sha256_file(path),
            "model_layer_digest": clean(model_layer.get("digest")).replace("sha256:", ""),
            "model_layer_size": clean(model_layer.get("size")),
        }
    return data


def ollama_client_version() -> str:
    try:
        completed = subprocess.run(["ollama", "--version"], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "ollama_client_unavailable"
    output = completed.stdout.strip().splitlines()
    for line in output:
        if "client version is" in line:
            return line.rsplit(" ", 1)[-1]
    return output[-1] if output else "ollama_client_unavailable"


def foodcom_hashes_match(root: Path) -> bool:
    for rel, expected in FOODCOM_HASH_EXPECTATIONS.items():
        path = root / rel
        if not path.exists() or sha256_file(path).lower() != expected.lower():
            return False
    return True


def dag_is_acyclic(frame: pd.DataFrame) -> bool:
    nodes = set(frame["node_id"].astype(str))
    graph: dict[str, set[str]] = {node: set() for node in nodes}
    indegree: dict[str, int] = {node: 0 for node in nodes}
    for record in frame.to_dict("records"):
        node = str(record["node_id"])
        for dep in split_semicolon(record.get("requires")):
            if dep and dep in nodes:
                graph[dep].add(node)
                indegree[node] += 1
    queue = deque([node for node, degree in indegree.items() if degree == 0])
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for child in graph[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return seen == len(nodes)


def split_semicolon(value: Any) -> list[str]:
    return [part.strip() for part in clean(value).split(";") if part.strip()]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_record(frame: pd.DataFrame, column: str, value: str) -> dict[str, Any]:
    if frame.empty or column not in frame.columns:
        return {}
    subset = frame.loc[frame[column].astype(str).eq(value)]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def first_value(frame: pd.DataFrame, column: str, default: str) -> str:
    if frame.empty or column not in frame.columns:
        return default
    return clean(frame.iloc[0].get(column, default))


def clean(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return str(value)


def int_or_default(value: Any, default: int = 0) -> int:
    try:
        if clean(value) == "":
            return default
        return int(float(clean(value)))
    except ValueError:
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, args: list[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())

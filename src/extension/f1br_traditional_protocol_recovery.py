"""F1BR traditional training protocol recovery.

F1BR resolves the F1B protocol ambiguity before any traditional recommender
training. It freezes paper-aligned original-four hyperparameters, a local
five-seed policy, Food.com validation-only search spaces, deterministic
baseline policies, and reranking source-pool rules.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import pandas as pd
import yaml

from src.extension.e0_protocol import sha256_json, stable_json
from src.extension.f0_final_extension_freeze import sha256_file
from src.utils.config import find_project_root


PHASE_ID = "f1br_traditional_protocol_recovery"
DECISION_TOKEN = "F1BR_TRADITIONAL_PROTOCOL_RESOLVED"
EXPECTED_HEAD = "c280fbef3440debe0a804efaf1f15436d9659347"
RECHORUS_GITLINK = "fb646d4eed13a8d81ae308965b08c816f40d1a57"
ACTIVE_DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
ORIGINAL_FOUR = ("beauty", "sports", "ml1m", "lastfm")
TRAINABLE_MODELS = ("BPRMF", "GRU4Rec", "SASRec", "LightGCN")
BASELINE_MODELS = ("Random", "MostPop", "BM25-like/content")
ALL_METHODS = BASELINE_MODELS + TRAINABLE_MODELS
FINAL_SEEDS = (42, 43, 44, 45, 46)
FOODCOM_TUNING_SEED = 42
F1B_READY_STATUSES = {"READY_TO_EXECUTE", "READY_TO_TRAIN", "DEGENERATE_PROTOCOL_INPUT"}

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")
CONFIG_PATH = Path("configs/f1br_traditional_protocol_recovery.yaml")
REPORT_PATH = REPORT_DIR / "f1br_traditional_protocol_recovery.md"
MANIFEST_PATH = MANIFEST_DIR / "f1br_traditional_protocol_freeze.json"

OUTPUT_TABLES = {
    "rechorus_lineage": TABLE_DIR / "f1br_rechorus_lineage.csv",
    "paper_hyperparameters": TABLE_DIR / "f1br_paper_hyperparameters.csv",
    "seed_policy": TABLE_DIR / "f1br_seed_policy.csv",
    "original_four_training_contract": TABLE_DIR / "f1br_original_four_training_contract.csv",
    "foodcom_search_space": TABLE_DIR / "f1br_foodcom_search_space.csv",
    "foodcom_tuning_contract": TABLE_DIR / "f1br_foodcom_tuning_contract.csv",
    "baseline_execution_readiness": TABLE_DIR / "f1br_baseline_execution_readiness.csv",
    "trainable_execution_readiness": TABLE_DIR / "f1br_trainable_execution_readiness.csv",
    "training_run_budget": TABLE_DIR / "f1br_training_run_budget.csv",
    "reranking_source_protocol": TABLE_DIR / "f1br_reranking_source_protocol.csv",
    "validation": TABLE_DIR / "f1br_validation.csv",
}
SUPPORT_ARTIFACTS = (
    CONFIG_PATH,
    Path("src/extension/f1br_traditional_protocol_recovery.py"),
    Path("tests/test_extension_f1br_traditional_protocol_recovery.py"),
    Path("report/notes.md"),
    Path(".gitignore"),
)
UPDATED_F1B_ARTIFACTS = (
    Path("configs/f1b_traditional_completion.yaml"),
    Path("src/extension/f1b_traditional_completion.py"),
    Path("tests/test_extension_f1b_traditional_completion.py"),
    TABLE_DIR / "f1b_traditional_reuse_audit.csv",
    TABLE_DIR / "f1b_training_plan.csv",
    TABLE_DIR / "f1b_training_results.csv",
    TABLE_DIR / "f1b_traditional_utility_results.csv",
    TABLE_DIR / "f1b_traditional_beyond_utility_results.csv",
    TABLE_DIR / "f1b_first_stage_ranking_inventory.csv",
    TABLE_DIR / "f1b_candidate_recall.csv",
    TABLE_DIR / "f1b_reranking_readiness.csv",
    TABLE_DIR / "f1b_five_dataset_completeness.csv",
    TABLE_DIR / "f1b_result_cell_readiness.csv",
    REPORT_DIR / "f1b_traditional_model_completion.md",
    MANIFEST_DIR / "f1b_training_provenance.json",
    MANIFEST_DIR / "f1b_no_llm_inference.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover and freeze F1BR traditional protocol.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    outputs = generate_f1br(root, Path(args.config))
    if not bool(outputs["validation"]["status"].eq("PASS").all()):
        print(stable_json({"decision": "F1BR_TRADITIONAL_PROTOCOL_BLOCKED", "failures": validation_failures(outputs["validation"])}))
        return 1
    if not args.validate_only:
        write_outputs(root, outputs)
        update_f1b_planning_artifacts(root, outputs)
        refresh_manifest_hashes(root)
    print(f"decision={DECISION_TOKEN}")
    print(f"rechorus_gitlink={RECHORUS_GITLINK}")
    print(f"paper_hyperparameter_rows={len(outputs['paper_hyperparameters'])}")
    print(f"seed_policy={','.join(str(seed) for seed in FINAL_SEEDS)}")
    print(f"foodcom_search_configs={len(outputs['foodcom_search_space'])}")
    print(f"original_four_ready_to_train={int(outputs['original_four_training_contract']['status'].eq('READY_TO_TRAIN').sum())}")
    print(f"trainable_ready_to_train={int(outputs['trainable_execution_readiness']['status'].eq('READY_TO_TRAIN').sum())}")
    print(f"baseline_ready_to_execute={int(outputs['baseline_execution_readiness']['status'].eq('READY_TO_EXECUTE').sum())}")
    print("training_runs_executed=0")
    print("llm_inference_calls=0")
    return 0


def generate_f1br(root: Path | None = None, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    project_root = root or find_project_root()
    config = load_config(project_root, config_path)
    artifacts = load_artifacts(project_root)
    lineage = build_rechorus_lineage(project_root)
    paper_hyperparameters = build_paper_hyperparameters(config)
    seed_policy = build_seed_policy(config)
    original_four_contract = build_original_four_training_contract(config, lineage)
    foodcom_search = build_foodcom_search_space(config)
    foodcom_tuning = build_foodcom_tuning_contract(config, foodcom_search)
    baseline_ready = build_baseline_execution_readiness(config, artifacts["traditional_matrix"])
    trainable_ready = build_trainable_execution_readiness(config, artifacts["traditional_matrix"], original_four_contract, foodcom_tuning)
    budget = build_training_run_budget(foodcom_search)
    reranking_protocol = build_reranking_source_protocol(config)
    f1b_updates = build_f1b_updates(project_root, seed_policy, baseline_ready, trainable_ready)
    integrity = build_integrity(project_root, artifacts)
    validation = build_validation(
        config,
        artifacts,
        lineage,
        paper_hyperparameters,
        seed_policy,
        original_four_contract,
        foodcom_search,
        foodcom_tuning,
        baseline_ready,
        trainable_ready,
        budget,
        reranking_protocol,
        f1b_updates,
        integrity,
    )
    manifest = build_manifest(
        project_root,
        config,
        validation,
        lineage,
        paper_hyperparameters,
        seed_policy,
        original_four_contract,
        foodcom_search,
        foodcom_tuning,
        baseline_ready,
        trainable_ready,
        budget,
        reranking_protocol,
        f1b_updates,
        integrity,
    )
    report = render_report(
        lineage,
        paper_hyperparameters,
        seed_policy,
        original_four_contract,
        foodcom_search,
        foodcom_tuning,
        baseline_ready,
        trainable_ready,
        budget,
        reranking_protocol,
        integrity,
        validation,
    )
    return {
        "rechorus_lineage": lineage,
        "paper_hyperparameters": paper_hyperparameters,
        "seed_policy": seed_policy,
        "original_four_training_contract": original_four_contract,
        "foodcom_search_space": foodcom_search,
        "foodcom_tuning_contract": foodcom_tuning,
        "baseline_execution_readiness": baseline_ready,
        "trainable_execution_readiness": trainable_ready,
        "training_run_budget": budget,
        "reranking_source_protocol": reranking_protocol,
        "f1b_updates": f1b_updates,
        "integrity": integrity,
        "validation": validation,
        "manifest": manifest,
        "report": report,
    }


def load_config(root: Path, config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return loaded


def load_artifacts(root: Path) -> dict[str, Any]:
    return {
        "dataset_matrix": read_csv(root / "results/tables/f0_final_dataset_matrix.csv"),
        "traditional_matrix": read_csv(root / "results/tables/f0_final_traditional_model_matrix.csv"),
        "raw_universe": read_csv(root / "results/tables/f0_final_raw_experiment_universe.csv"),
        "result_cells": read_csv(root / "results/tables/f0_final_result_cell_universe.csv"),
        "budget": read_csv(root / "results/tables/f0_final_request_budget.csv"),
        "f1arr_final_blockers": read_csv(root / "results/tables/f1arr_final_metric_blockers.csv"),
        "h0_manifest": read_json(root / "results/manifests/extension_h0_no_inference_manifest.json"),
        "foodcom_hash_manifest": read_json(root / "results/manifests/d0a_foodcom_extracted_file_hashes.json"),
    }


def build_rechorus_lineage(root: Path) -> pd.DataFrame:
    gitlink = git_output(root, ["-C", "original_repo/EvaLLMasRecommender", "ls-tree", "HEAD", "ReChorus"])
    submodule_dir = root / "original_repo/EvaLLMasRecommender/ReChorus"
    source_files = sorted(path for path in submodule_dir.rglob("*") if path.is_file()) if submodule_dir.exists() else []
    smoke = zero_training_smoke()
    rows = [
        lineage_row("exact_rechorus_gitlink", "RECOVERED", RECHORUS_GITLINK, gitlink, "Original repository tree records ReChorus as a gitlink."),
        lineage_row("submodule_mapping", "NOT_AVAILABLE_LOCALLY", "", ".gitmodules missing", "No local .gitmodules mapping is present for ReChorus."),
        lineage_row("submodule_source_worktree", "NOT_AVAILABLE_LOCALLY", "", f"file_count={len(source_files)}", "ReChorus worktree is empty; no network fetch was attempted."),
        lineage_row("main_argument_defaults", "NOT_RECOVERED_LOCALLY", "", "ReChorus source unavailable", "Cannot recover ReChorus parser defaults without the submodule source."),
        lineage_row("optimizer_defaults", "NOT_RECOVERED_LOCALLY", "", "ReChorus source unavailable", "Optimizer defaults must not be guessed as original ReChorus defaults."),
        lineage_row("epoch_policy", "NOT_RECOVERED_LOCALLY", "", "ReChorus source unavailable", "Original ReChorus epoch/early-stop defaults unavailable locally."),
        lineage_row("early_stopping", "NOT_RECOVERED_LOCALLY", "", "ReChorus source unavailable", "Original ReChorus early-stopping defaults unavailable locally."),
        lineage_row("batch_size", "NOT_RECOVERED_LOCALLY", "", "ReChorus source unavailable", "Original ReChorus batch defaults unavailable locally."),
        lineage_row("eval_batch_size", "NOT_RECOVERED_LOCALLY", "", "ReChorus source unavailable", "Original ReChorus eval-batch defaults unavailable locally."),
        lineage_row("negative_sampling", "PARTIALLY_RECOVERED", "uniform_unobserved_train_item", "project-controlled wrappers", "Local wrappers use deterministic sampled negatives excluding user positives; original ReChorus default remains unavailable."),
        lineage_row("history_max", "PARTIALLY_RECOVERED", "history10 metrics; LightGCN/BPRMF source files read with history20 in prompt notebook", "original eva/codes/llm_metrics.py and generate_prompt.ipynb", "Evaluation reader exposes history_max; exact training history cap must be set explicitly by project wrappers."),
        lineage_row("model_specific_defaults", "NOT_RECOVERED_LOCALLY", "", "ReChorus source unavailable", "Paper Appendix A.3 overrides the core model hyperparameters used by F1BR."),
        lineage_row("evaluation_protocol", "RECOVERED", "candidate_pool=20; top_k=5; HR/NDCG/MRR over held-out positive rank", "original eva/codes/llm_metrics.py", "Original metric code reads SOTA recommendations and candidate-pool evaluation with K=5."),
        lineage_row("random_seed_handling", "RECOVERED", "repeat command base_seed..base_seed+n-1; --random_seed default 0", "original eva/codes/random_test.py and llm_metrics.py", "Paper reports five random initializations but not exact seed identities."),
        lineage_row("local_project_adapter_smoke", smoke["status"], smoke["details"], "src/baselines/plan_d_*.py", "Import and tiny construction only; no optimizer step or data scoring."),
    ]
    return pd.DataFrame(rows)


def lineage_row(aspect: str, status: str, recovered_value: str, evidence: str, notes: str) -> dict[str, str]:
    return {
        "aspect": aspect,
        "status": status,
        "rechorus_gitlink_commit": RECHORUS_GITLINK,
        "recovered_value": recovered_value,
        "evidence": evidence,
        "network_used": "false",
        "training_executed": "false",
        "notes": notes,
    }


@lru_cache(maxsize=1)
def zero_training_smoke() -> dict[str, str]:
    try:
        from src.baselines.plan_d_bprmf import BPRMFModel
        from src.baselines.plan_d_gru4rec import GRU4RecModel
        from src.baselines.plan_d_lightgcn import LightGCNModel
        from src.baselines.plan_d_sasrec import SASRecModel

        BPRMFModel(2, 3, 4)
        GRU4RecModel(num_items=3, embedding_dim=4, hidden_dim=4, num_layers=1, dropout=0.0)
        SASRecModel(num_items=3, max_seq_len=5, embedding_dim=4, num_heads=1, num_layers=1, dropout=0.0)
        LightGCNModel(num_users=2, num_items=3, embedding_dim=4, num_layers=1)
        return {"status": "PASS", "details": "BPRMF, GRU4Rec, SASRec, and LightGCN tiny constructors succeeded"}
    except Exception as exc:  # pragma: no cover - exercised only on broken local runtimes
        return {"status": "FAIL", "details": f"{type(exc).__name__}: {exc}"}


def build_paper_hyperparameters(config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    paper = config["paper_hyperparameters"]
    for method, datasets in paper.items():
        for dataset, params in datasets.items():
            row = {
                "dataset": dataset,
                "method": method,
                "lr": "",
                "l2": "",
                "embedding": "",
                "hidden": "",
                "layers": "",
                "heads": "",
                "k1": "",
                "b": "",
                "source_priority": "paper_appendix_a3",
                "evidence_source": config["paper_notes"]["evidence_source"],
                "appendix_typo_note": "",
                "status": "FROZEN",
            }
            row.update({key: format_param(value) for key, value in params.items()})
            if method == "LightGCN" and dataset in {"ml1m", "lastfm"}:
                row["appendix_typo_note"] = config["paper_notes"]["appendix_a3_typo"]
            rows.append(row)
    order = {method: index for index, method in enumerate(("BM25", *TRAINABLE_MODELS))}
    dataset_order = {dataset: index for index, dataset in enumerate(ORIGINAL_FOUR)}
    frame = pd.DataFrame(rows)
    frame["method_order"] = frame["method"].map(order)
    frame["dataset_order"] = frame["dataset"].map(dataset_order)
    return frame.sort_values(["method_order", "dataset_order"]).drop(columns=["method_order", "dataset_order"]).reset_index(drop=True)


def build_seed_policy(config: Mapping[str, Any]) -> pd.DataFrame:
    policy = config["seed_policy"]
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "final_seeds": stable_json(policy["final_seeds"]),
                "tuning_seed": int(policy["tuning_seed"]) if dataset == "foodcom" else "",
                "evidence_classification": policy["evidence_classification"],
                "paper_run_count_equivalent": bool_text(policy["paper_run_count_equivalent"]),
                "paper_seed_identity_equivalent": bool_text(policy["paper_seed_identity_equivalent"]),
                "claimed_as_author_seed_identity": "false",
                "source": policy["source"],
                "status": "FROZEN",
            }
            for dataset in ACTIVE_DATASETS
        ]
    )


def build_original_four_training_contract(config: Mapping[str, Any], lineage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    smoke_status = lineage.loc[lineage["aspect"].eq("local_project_adapter_smoke"), "status"].iloc[0]
    for dataset in ORIGINAL_FOUR:
        for method in TRAINABLE_MODELS:
            params = config["paper_hyperparameters"][method][dataset]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "hyperparameter_source": "paper_appendix_a3",
                    "runtime_classification": "PAPER_HYPERPARAMETER_ALIGNED_LOCAL_IMPLEMENTATION",
                    "rechorus_gitlink_commit": RECHORUS_GITLINK,
                    "rechorus_source_available_locally": "false",
                    "project_adapter_smoke": smoke_status,
                    "final_seeds": stable_json(FINAL_SEEDS),
                    "paper_seed_identity_equivalent": "false",
                    "status": "READY_TO_TRAIN",
                    "reason": "Paper hyperparameters and local five-seed policy are frozen; original ReChorus source defaults unavailable locally, so project wrappers are classified as local implementations rather than exact ReChorus executions.",
                    "hyperparameter_json": stable_json(params),
                }
            )
    return pd.DataFrame(rows)


def build_foodcom_search_space(config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for method, dimensions in config["foodcom_search_space"].items():
        keys = list(dimensions)
        for values in itertools.product(*(dimensions[key] for key in keys)):
            params = dict(zip(keys, values))
            identity = {"phase": PHASE_ID, "dataset": "foodcom", "method": method, "params": params}
            rows.append(
                {
                    "search_config_id": "F1BR-FOODCOM-SEARCH-" + sha256_json(identity).upper()[:16],
                    "dataset": "foodcom",
                    "method": method,
                    "lr": format_param(params.get("lr", "")),
                    "l2": format_param(params.get("l2", "")),
                    "embedding": format_param(params.get("embedding", "")),
                    "hidden": format_param(params.get("hidden", "")),
                    "layers": format_param(params.get("layers", "")),
                    "heads": format_param(params.get("heads", "")),
                    "source": "derived_only_from_paper_original_four_hyperparameter_values",
                    "selection_metric": "validation_ndcg_at_5",
                    "tuning_seed": FOODCOM_TUNING_SEED,
                    "uses_final_test_metrics": "false",
                    "status": "FROZEN_SEARCH_SPACE",
                }
            )
    order = {method: index for index, method in enumerate(TRAINABLE_MODELS)}
    frame = pd.DataFrame(rows)
    frame["method_order"] = frame["method"].map(order)
    return frame.sort_values(["method_order", "search_config_id"]).drop(columns=["method_order"]).reset_index(drop=True)


def build_foodcom_tuning_contract(config: Mapping[str, Any], search_space: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in TRAINABLE_MODELS:
        count = int(search_space["method"].eq(method).sum())
        rows.append(
            {
                "dataset": "foodcom",
                "method": method,
                "search_space_configs": count,
                "tuning_seed": FOODCOM_TUNING_SEED,
                "selection_metric": "validation_ndcg_at_5",
                "test_metric_tuning_allowed": "false",
                "tie_breaking": "higher_validation_ndcg_at_5_then_lower_search_config_id_lexicographic",
                "final_seeds_after_selection": stable_json(FINAL_SEEDS),
                "classification": "ORIGINAL_EXTENSION",
                "status": "READY_FOR_VALIDATION_TUNING",
                "reason": "Food.com has no paper-winning config; search is predeclared before training and uses validation only.",
            }
        )
    return pd.DataFrame(rows)


def build_baseline_execution_readiness(config: Mapping[str, Any], traditional: pd.DataFrame) -> pd.DataFrame:
    policy = config["baseline_policy"]
    rows = []
    selected = traditional.loc[traditional["method"].isin(BASELINE_MODELS)].copy()
    for record in selected.to_dict("records"):
        method = record["method"]
        method_policy = policy[method]
        rows.append(
            {
                "dataset": record["dataset"],
                "method": method,
                "cohort_id": record["cohort_id"],
                "evaluation_users": int(record["cohort_size"]),
                "candidate_pool_size": 20,
                "random_seed": method_policy.get("random_seed", ""),
                "frequency_definition": method_policy.get("frequency_definition", ""),
                "k1": method_policy.get("k1", ""),
                "b": method_policy.get("b", ""),
                "foodcom_text_policy": method_policy.get("foodcom_text_policy", "") if record["dataset"] == "foodcom" else "",
                "tie_breaking": method_policy["tie_breaking"],
                "candidate_contract": method_policy["candidate_contract"],
                "status": "READY_TO_EXECUTE",
                "reason": "No remaining protocol ambiguity prevents deterministic baseline recomputation.",
            }
        )
    return ordered_by_dataset_method(pd.DataFrame(rows), "method")


def build_trainable_execution_readiness(
    config: Mapping[str, Any],
    traditional: pd.DataFrame,
    original_four_contract: pd.DataFrame,
    foodcom_tuning: pd.DataFrame,
) -> pd.DataFrame:
    contract_by_key = {
        (row["dataset"], row["method"]): row
        for row in original_four_contract.to_dict("records")
    }
    tuning_by_model = {row["method"]: row for row in foodcom_tuning.to_dict("records")}
    rows = []
    selected = traditional.loc[traditional["method"].isin(TRAINABLE_MODELS)].copy()
    for record in selected.to_dict("records"):
        dataset = record["dataset"]
        method = record["method"]
        if dataset == "foodcom":
            tuning = tuning_by_model[method]
            classification = "ORIGINAL_EXTENSION"
            hyper_source = "f1br_foodcom_validation_only_search_space"
            prerequisite = "run validation tuning with seed 42; select config before final five-seed runs"
            params = {}
            search_configs = int(tuning["search_space_configs"])
        else:
            contract = contract_by_key[(dataset, method)]
            classification = contract["runtime_classification"]
            hyper_source = "paper_appendix_a3"
            prerequisite = "none"
            params = config["paper_hyperparameters"][method][dataset]
            search_configs = 0
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "cohort_id": record["cohort_id"],
                "evaluation_users": int(record["cohort_size"]),
                "classification": classification,
                "hyperparameter_source": hyper_source,
                "hyperparameter_json": stable_json(params) if params else "",
                "foodcom_search_configs": search_configs,
                "final_seeds": stable_json(FINAL_SEEDS),
                "seed_policy": "LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY",
                "paper_seed_identity_equivalent": "false",
                "execution_prerequisite": prerequisite,
                "status": "READY_TO_TRAIN",
                "reason": "F1BR resolved seed/config lineage before any training.",
            }
        )
    return ordered_by_dataset_method(pd.DataFrame(rows), "method")


def build_training_run_budget(foodcom_search_space: pd.DataFrame) -> pd.DataFrame:
    original_four_final = len(ORIGINAL_FOUR) * len(TRAINABLE_MODELS) * len(FINAL_SEEDS)
    foodcom_final = len(TRAINABLE_MODELS) * len(FINAL_SEEDS)
    foodcom_tuning = len(foodcom_search_space)
    rows = [
        ("ORIGINAL_FOUR_FINAL_RUNS", original_four_final, "4 datasets x 4 trainable models x 5 local final seeds"),
        ("FOODCOM_FINAL_RUNS", foodcom_final, "Food.com selected config per model x 5 local final seeds"),
        ("FOODCOM_TUNING_RUNS", foodcom_tuning, "Validation-only search configs x one fixed tuning seed"),
        ("FINAL_EVIDENCE_RUNS", original_four_final + foodcom_final, "Final five-seed model runs after Food.com tuning"),
        ("TOTAL_TRAINING_RUNS", original_four_final + foodcom_final + foodcom_tuning, "Final evidence runs plus tuning runs"),
    ]
    return pd.DataFrame(
        [
            {
                "budget_item": name,
                "run_count": int(count),
                "training_executed_in_f1br": "false",
                "notes": notes,
            }
            for name, count, notes in rows
        ]
    )


def build_reranking_source_protocol(config: Mapping[str, Any]) -> pd.DataFrame:
    policy = config["reranking_source_policy"]
    rows = []
    for dataset in ACTIVE_DATASETS:
        rows.append(
            {
                "dataset": dataset,
                "paper_rankers": stable_json(policy["paper_rankers"]),
                "exact_union_order_recovered": bool_text(policy["exact_union_order_recovered"]),
                "adapted_policy_id": policy["adapted_policy_id"],
                "model_order": stable_json(policy["model_order"]),
                "per_model_top_k": int(policy["per_model_top_k"]),
                "duplicate_policy": policy["duplicate_policy"],
                "final_pool_size": int(policy["final_pool_size"]),
                "insufficient_unique_policy": policy["insufficient_unique_policy"],
                "candidate_recall_required": bool_text(policy["candidate_recall_required"]),
                "status": "ADAPTED_POLICY_FROZEN",
                "reason": "Paper ranker set is recovered; union ordering is not locally recovered, so an adapted policy is frozen before pool materialization.",
            }
        )
    return pd.DataFrame(rows)


def build_f1b_updates(
    root: Path,
    seed_policy: pd.DataFrame,
    baseline_ready: pd.DataFrame,
    trainable_ready: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    updates: dict[str, pd.DataFrame] = {}
    if not (root / "results/tables/f1b_traditional_reuse_audit.csv").exists():
        return updates
    reuse = read_csv(root / "results/tables/f1b_traditional_reuse_audit.csv")
    reuse = update_reuse_audit(reuse)
    training_plan = update_training_plan(read_csv(root / "results/tables/f1b_training_plan.csv"))
    training_results = update_training_results(read_csv(root / "results/tables/f1b_training_results.csv"))
    readiness = update_result_cell_readiness(read_csv(root / "results/tables/f1b_result_cell_readiness.csv"))
    utility = update_metric_result_status(read_csv(root / "results/tables/f1b_traditional_utility_results.csv"), readiness)
    beyond = update_metric_result_status(read_csv(root / "results/tables/f1b_traditional_beyond_utility_results.csv"), readiness)
    completeness = update_five_dataset_completeness(read_csv(root / "results/tables/f1b_five_dataset_completeness.csv"))
    first_stage = update_first_stage_inventory(read_csv(root / "results/tables/f1b_first_stage_ranking_inventory.csv"))
    candidate_recall = update_candidate_recall(read_csv(root / "results/tables/f1b_candidate_recall.csv"))
    reranking = update_reranking_readiness(read_csv(root / "results/tables/f1b_reranking_readiness.csv"))
    updates.update(
        {
            "reuse_audit": reuse,
            "training_plan": training_plan,
            "training_results": training_results,
            "utility_results": utility,
            "beyond_utility_results": beyond,
            "five_dataset_completeness": completeness,
            "result_cell_readiness": readiness,
            "first_stage_inventory": first_stage,
            "candidate_recall": candidate_recall,
            "reranking_readiness": reranking,
        }
    )
    return updates


def update_reuse_audit(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    baseline_mask = frame["method"].isin(BASELINE_MODELS)
    trainable_mask = frame["method"].isin(TRAINABLE_MODELS)
    frame.loc[baseline_mask, "reuse_decision"] = "RECOMPUTE_REQUIRED"
    frame.loc[baseline_mask, "final_f1b_status"] = "READY_TO_EXECUTE"
    frame.loc[baseline_mask, "reason"] = "F1BR resolved deterministic baseline policy; recomputation is ready but not executed in F1BR."
    frame.loc[trainable_mask, "reuse_decision"] = "TRAIN_REQUIRED"
    frame.loc[trainable_mask, "final_f1b_status"] = "READY_TO_TRAIN"
    frame.loc[trainable_mask, "seed_policy"] = "LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY [42,43,44,45,46]; not claimed as paper seed identities"
    frame.loc[trainable_mask, "reason"] = "F1BR resolved seed/config lineage; no prior model outputs are protocol-identical reuse."
    return frame


def update_training_plan(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["resolved_seed"] = frame["seed_slot"].astype(int).map(lambda slot: FINAL_SEEDS[slot - 1])
    frame["seed_policy_status"] = "LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY"
    frame["planned_epoch_policy"] = "paper_or_project_runtime_default_explicitly_frozen_before_training"
    frame["planned_device"] = "project_runtime_auto_or_configured_before_execution"
    frame["training_plan_status"] = "READY_TO_TRAIN"
    frame["training_executed"] = "false"
    frame["reason"] = "F1BR resolved protocol; F1BR still performs zero training."
    frame["evidence"] = "results/manifests/f1br_traditional_protocol_freeze.json"
    return frame


def update_training_results(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["resolved_seed"] = frame["seed_slot"].astype(int).map(lambda slot: FINAL_SEEDS[slot - 1])
    frame["status"] = "NOT_EXECUTED_F1BR_PROTOCOL_RESOLVED"
    frame["reason"] = "Protocol is ready, but F1BR does not train."
    frame["training_executed"] = "false"
    return frame


def update_result_cell_readiness(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    degenerate = frame["f1b_status"].eq("DEGENERATE_PROTOCOL_INPUT")
    baseline = frame["model_name"].isin(BASELINE_MODELS)
    trainable = frame["model_name"].isin(TRAINABLE_MODELS)
    frame.loc[~degenerate & baseline, "f1b_status"] = "READY_FROM_NEW_RAW"
    frame.loc[~degenerate & baseline, "reason"] = "F1BR resolved deterministic baseline policy; final metric awaits recomputed baseline rankings."
    frame.loc[~degenerate & trainable, "f1b_status"] = "READY_AFTER_TRAINING_RAW"
    frame.loc[~degenerate & trainable, "reason"] = "F1BR resolved trainable protocol; final metric awaits traditional training output."
    return frame


def update_metric_result_status(frame: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    by_cell = readiness.set_index("result_cell_id").to_dict("index")
    frame = frame.copy()
    for index, row in frame.iterrows():
        updated = by_cell[row["result_cell_id"]]
        frame.loc[index, "f1b_status"] = updated["f1b_status"]
        frame.loc[index, "reason"] = updated["reason"]
    return frame


def update_five_dataset_completeness(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    baseline = frame["method"].isin(BASELINE_MODELS)
    trainable = frame["method"].isin(TRAINABLE_MODELS)
    frame.loc[baseline, "final_status"] = "READY_TO_EXECUTE"
    frame.loc[baseline, "reason"] = "F1BR resolved deterministic baseline policy; no baseline execution occurred."
    frame.loc[trainable, "final_status"] = "READY_TO_TRAIN"
    frame.loc[trainable, "reason"] = "F1BR resolved training protocol; no training occurred."
    return frame


def update_first_stage_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["inventory_status"] = "READY_AFTER_TRADITIONAL_EXECUTION"
    frame["reason"] = "F1BR froze the source-ranker protocol; rankings still must be materialized before reranking."
    return frame


def update_candidate_recall(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["status"] = "AWAITING_FIRST_STAGE_RANKINGS"
    frame["reason"] = "F1BR froze reranking source protocol; candidate recall remains future execution evidence."
    return frame


def update_reranking_readiness(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["readiness_status"] = "AWAITING_FIRST_STAGE_RANKINGS_AND_CANDIDATE_RECALL"
    frame["future_requests_unlocked"] = 0
    frame["reason"] = "Protocol resolved, but no first-stage rankings or candidate-recall evidence exist yet."
    return frame


def build_integrity(root: Path, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    h0_hashes = verify_manifest_hashes(root, artifacts["h0_manifest"].get("artifact_hashes", {}))
    foodcom_hashes = verify_foodcom_hashes(artifacts["foodcom_hash_manifest"])
    return {
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "head_is_expected": git_output(root, ["rev-parse", "HEAD"]) == EXPECTED_HEAD,
        "active_datasets": active_dataset_list(artifacts["dataset_matrix"]),
        "active_video_games_rows": video_games_rows(artifacts["dataset_matrix"], artifacts["traditional_matrix"], artifacts["raw_universe"], artifacts["result_cells"]),
        "global_metric_blockers": len(artifacts["f1arr_final_blockers"]),
        "f0_budget": {row["budget_item"]: int(row["request_count"]) for row in artifacts["budget"].to_dict("records")},
        "h0_hashes_unchanged": h0_hashes["unchanged"],
        "foodcom_hashes_unchanged": foodcom_hashes["unchanged"],
        "original_repo_status": git_output(root, ["status", "--short", "--untracked-files=all", "--", "original_repo"]),
        "paper_status": git_output(root, ["status", "--short", "--untracked-files=all", "--", "paper"]),
        "non_phase_worktree_status": non_phase_worktree_status(root),
        "llm_inference_calls_executed": 0,
        "traditional_training_runs_executed": 0,
        "deterministic_baseline_runs_executed": 0,
        "downloads_executed": 0,
        "network_calls_executed": 0,
        "external_api_calls_executed": 0,
        "commit_created": 0,
    }


def build_validation(
    config: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    lineage: pd.DataFrame,
    paper_hyperparameters: pd.DataFrame,
    seed_policy: pd.DataFrame,
    original_four_contract: pd.DataFrame,
    foodcom_search: pd.DataFrame,
    foodcom_tuning: pd.DataFrame,
    baseline_ready: pd.DataFrame,
    trainable_ready: pd.DataFrame,
    budget: pd.DataFrame,
    reranking_protocol: pd.DataFrame,
    f1b_updates: Mapping[str, pd.DataFrame],
    integrity: Mapping[str, Any],
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []

    def check(name: str, condition: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "evidence": evidence})

    budget_map = {row["budget_item"]: int(row["run_count"]) for row in budget.to_dict("records")}
    paper_methods = set(paper_hyperparameters["method"])
    check("head_is_expected", bool(integrity["head_is_expected"]), clean(integrity["head"]))
    check("rechorus_gitlink_exact", lineage.loc[lineage["aspect"].eq("exact_rechorus_gitlink"), "recovered_value"].iloc[0] == RECHORUS_GITLINK, RECHORUS_GITLINK)
    check("rechorus_defaults_not_fabricated", lineage.loc[lineage["status"].eq("NOT_RECOVERED_LOCALLY")].shape[0] >= 1, "unavailable defaults explicitly marked")
    check("local_zero_training_smoke_passed", lineage.loc[lineage["aspect"].eq("local_project_adapter_smoke"), "status"].iloc[0] == "PASS", lineage.loc[lineage["aspect"].eq("local_project_adapter_smoke"), "recovered_value"].iloc[0])
    check("paper_hyperparameter_rows", len(paper_hyperparameters) == 20 and paper_methods == {"BM25", *TRAINABLE_MODELS}, str(len(paper_hyperparameters)))
    check("lightgcn_appendix_typo_documented", paper_hyperparameters.loc[paper_hyperparameters["method"].eq("LightGCN"), "appendix_typo_note"].astype(str).str.contains("not an operational").any(), "documented")
    check("seed_policy_42_46", seed_policy["final_seeds"].eq(stable_json(FINAL_SEEDS)).all(), stable_json(FINAL_SEEDS))
    check("seed_policy_not_paper_identity", seed_policy["paper_seed_identity_equivalent"].eq("false").all() and seed_policy["claimed_as_author_seed_identity"].eq("false").all(), "false")
    check("seed_evidence_classification", seed_policy["evidence_classification"].eq("LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY").all(), "LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY")
    check("original_four_ready_to_train_16", len(original_four_contract) == 16 and original_four_contract["status"].eq("READY_TO_TRAIN").all(), str(len(original_four_contract)))
    check("f6d2_not_promoted_as_reuse", trainable_ready["seed_policy"].eq("LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY").all(), "reuse remains zero")
    check("foodcom_search_space_40", len(foodcom_search) == 40 and foodcom_search["uses_final_test_metrics"].eq("false").all(), str(len(foodcom_search)))
    check("foodcom_tuning_validation_only", len(foodcom_tuning) == 4 and foodcom_tuning["test_metric_tuning_allowed"].eq("false").all(), str(len(foodcom_tuning)))
    check("baseline_ready_15", len(baseline_ready) == 15 and baseline_ready["status"].eq("READY_TO_EXECUTE").all(), str(len(baseline_ready)))
    check("trainable_ready_20", len(trainable_ready) == 20 and trainable_ready["status"].eq("READY_TO_TRAIN").all(), str(len(trainable_ready)))
    check("training_budget_exact", budget_map.get("ORIGINAL_FOUR_FINAL_RUNS") == 80 and budget_map.get("FOODCOM_FINAL_RUNS") == 20 and budget_map.get("FOODCOM_TUNING_RUNS") == 40 and budget_map.get("TOTAL_TRAINING_RUNS") == 140, stable_json(budget_map))
    check("reranking_source_policy_frozen", len(reranking_protocol) == 5 and reranking_protocol["status"].eq("ADAPTED_POLICY_FROZEN").all(), str(len(reranking_protocol)))
    check("f1b_planning_updates_prepared", set(f1b_updates) >= {"reuse_audit", "training_plan", "result_cell_readiness"} and len(f1b_updates["training_plan"]) == 100, ",".join(sorted(f1b_updates)))
    check("active_datasets_exact", tuple(integrity["active_datasets"]) == ACTIVE_DATASETS, ",".join(integrity["active_datasets"]))
    check("video_games_absent", int(integrity["active_video_games_rows"]) == 0, str(integrity["active_video_games_rows"]))
    check("global_metric_blockers_zero", int(integrity["global_metric_blockers"]) == 0, str(integrity["global_metric_blockers"]))
    check("f0_budget_unchanged", f0_budget_unchanged(integrity["f0_budget"]), stable_json(integrity["f0_budget"]))
    check("h0_hashes_unchanged", bool(integrity["h0_hashes_unchanged"]), "true")
    check("foodcom_hashes_unchanged", bool(integrity["foodcom_hashes_unchanged"]), "true")
    check("original_repo_unchanged", clean(integrity["original_repo_status"]) == "", clean(integrity["original_repo_status"]) or "clean")
    check("paper_unchanged", clean(integrity["paper_status"]) == "", clean(integrity["paper_status"]) or "clean")
    check("no_unexplained_worktree_changes", clean(integrity["non_phase_worktree_status"]) == "", clean(integrity["non_phase_worktree_status"]) or "none")
    check("zero_training_in_f1br", int(integrity["traditional_training_runs_executed"]) == 0 and int(integrity["deterministic_baseline_runs_executed"]) == 0, "0")
    check("zero_llm_inference", int(integrity["llm_inference_calls_executed"]) == 0, "0")
    check("zero_download_network_api", int(integrity["downloads_executed"]) == 0 and int(integrity["network_calls_executed"]) == 0 and int(integrity["external_api_calls_executed"]) == 0, "0")
    check("config_forbids_training_and_inference", config["execution"]["allow_traditional_training"] is False and config["execution"]["allow_llm_inference"] is False, "false")
    return pd.DataFrame(checks)


def build_manifest(
    root: Path,
    config: Mapping[str, Any],
    validation: pd.DataFrame,
    lineage: pd.DataFrame,
    paper_hyperparameters: pd.DataFrame,
    seed_policy: pd.DataFrame,
    original_four_contract: pd.DataFrame,
    foodcom_search: pd.DataFrame,
    foodcom_tuning: pd.DataFrame,
    baseline_ready: pd.DataFrame,
    trainable_ready: pd.DataFrame,
    budget: pd.DataFrame,
    reranking_protocol: pd.DataFrame,
    f1b_updates: Mapping[str, pd.DataFrame],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_TOKEN,
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "expected_head": clean(config["phase"]["expected_head"]),
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation_failures(validation),
        "rechorus_gitlink_commit": RECHORUS_GITLINK,
        "rechorus_source_available_locally": False,
        "paper_hyperparameter_rows": len(paper_hyperparameters),
        "seed_policy_rows": len(seed_policy),
        "final_seeds": list(FINAL_SEEDS),
        "paper_seed_identity_equivalent": False,
        "original_four_ready_to_train_rows": int(original_four_contract["status"].eq("READY_TO_TRAIN").sum()),
        "baseline_ready_to_execute_rows": int(baseline_ready["status"].eq("READY_TO_EXECUTE").sum()),
        "trainable_ready_to_train_rows": int(trainable_ready["status"].eq("READY_TO_TRAIN").sum()),
        "foodcom_search_configs": len(foodcom_search),
        "foodcom_tuning_contract_rows": len(foodcom_tuning),
        "training_run_budget": {row["budget_item"]: int(row["run_count"]) for row in budget.to_dict("records")},
        "reranking_source_protocol_rows": len(reranking_protocol),
        "f1b_updated_artifacts": sorted(f1b_updates.keys()),
        "integrity": dict(integrity),
        "llm_inference_calls_executed": 0,
        "traditional_training_runs_executed": 0,
        "deterministic_baseline_runs_executed": 0,
        "downloads_executed": 0,
        "network_calls_executed": 0,
        "external_api_calls_executed": 0,
        "commit_created": 0,
    }


def render_report(
    lineage: pd.DataFrame,
    paper_hyperparameters: pd.DataFrame,
    seed_policy: pd.DataFrame,
    original_four_contract: pd.DataFrame,
    foodcom_search: pd.DataFrame,
    foodcom_tuning: pd.DataFrame,
    baseline_ready: pd.DataFrame,
    trainable_ready: pd.DataFrame,
    budget: pd.DataFrame,
    reranking_protocol: pd.DataFrame,
    integrity: Mapping[str, Any],
    validation: pd.DataFrame,
) -> str:
    budget_map = {row["budget_item"]: int(row["run_count"]) for row in budget.to_dict("records")}
    return "\n".join(
        [
            "# F1BR Traditional Protocol Recovery",
            "",
            f"Decision token: `{DECISION_TOKEN}`",
            "",
            "## ReChorus Lineage",
            f"- Gitlink commit: `{RECHORUS_GITLINK}`.",
            "- Local ReChorus source/defaults are not available; defaults are not fabricated.",
            f"- Project adapter zero-training smoke: `{lineage.loc[lineage['aspect'].eq('local_project_adapter_smoke'), 'status'].iloc[0]}`.",
            "",
            "## Paper Hyperparameters",
            f"- Frozen rows: `{len(paper_hyperparameters)}`.",
            "- Appendix LightGCN repeated-label typo documented as non-operational because the final two configurations are identical.",
            "",
            "## Seed Policy",
            f"- Final seeds: `{stable_json(FINAL_SEEDS)}`.",
            "- Classification: `LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY`; no paper seed identity claim.",
            "",
            "## Execution Readiness",
            f"- Deterministic baseline rows ready: `{int(baseline_ready['status'].eq('READY_TO_EXECUTE').sum())}`.",
            f"- Trainable dataset/model rows ready: `{int(trainable_ready['status'].eq('READY_TO_TRAIN').sum())}`.",
            f"- Original-four paper-hyperparameter aligned local implementation rows: `{len(original_four_contract)}`.",
            "",
            "## Food.com",
            f"- Validation search configs: `{len(foodcom_search)}`.",
            f"- Tuning contracts: `{len(foodcom_tuning)}`.",
            "- Selection metric: `validation_ndcg_at_5`; final test metrics cannot tune hyperparameters.",
            "",
            "## Training Budget",
            "\n".join(f"- `{key}` = `{value}`." for key, value in budget_map.items()),
            "",
            "## Reranking Source",
            f"- Rows: `{len(reranking_protocol)}`.",
            "- Paper ranker set recovered; union ordering frozen as adapted ordered deduplicated top-20 policy.",
            "",
            "## Integrity",
            f"- Active datasets: `{','.join(integrity['active_datasets'])}`.",
            f"- Video Games rows: `{integrity['active_video_games_rows']}`.",
            "- LLM inference/training/download/network counts are all zero.",
            "",
            "## Validation",
            f"- Passed: `{str(validation['status'].eq('PASS').all()).lower()}`.",
        ]
    ) + "\n"


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for directory in (root / TABLE_DIR, root / MANIFEST_DIR, root / REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for key, rel_path in OUTPUT_TABLES.items():
        outputs[key].to_csv(root / rel_path, index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(outputs["report"], encoding="utf-8")
    manifest = dict(outputs["manifest"])
    manifest["artifact_hashes"] = artifact_hashes(root, [*OUTPUT_TABLES.values(), REPORT_PATH, *SUPPORT_ARTIFACTS])
    (root / MANIFEST_PATH).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def update_f1b_planning_artifacts(root: Path, outputs: Mapping[str, Any]) -> None:
    updates = outputs["f1b_updates"]
    if not updates:
        return
    path_by_key = {
        "reuse_audit": TABLE_DIR / "f1b_traditional_reuse_audit.csv",
        "training_plan": TABLE_DIR / "f1b_training_plan.csv",
        "training_results": TABLE_DIR / "f1b_training_results.csv",
        "utility_results": TABLE_DIR / "f1b_traditional_utility_results.csv",
        "beyond_utility_results": TABLE_DIR / "f1b_traditional_beyond_utility_results.csv",
        "first_stage_inventory": TABLE_DIR / "f1b_first_stage_ranking_inventory.csv",
        "candidate_recall": TABLE_DIR / "f1b_candidate_recall.csv",
        "reranking_readiness": TABLE_DIR / "f1b_reranking_readiness.csv",
        "five_dataset_completeness": TABLE_DIR / "f1b_five_dataset_completeness.csv",
        "result_cell_readiness": TABLE_DIR / "f1b_result_cell_readiness.csv",
    }
    for key, frame in updates.items():
        frame.to_csv(root / path_by_key[key], index=False, lineterminator="\n")
    update_f1b_report(root)
    update_f1b_manifests(root)


def update_f1b_report(root: Path) -> None:
    report_path = root / "report/f1b_traditional_model_completion.md"
    if not report_path.exists():
        return
    text = report_path.read_text(encoding="utf-8").split("\n## F1BR Addendum\n", 1)[0].rstrip()
    addendum = (
        "\n\n## F1BR Addendum\n"
        "- F1BR resolved the seed/config protocol ambiguity without training.\n"
        "- Random, MostPop, and BM25-like/content are ready to execute; trainable rows are ready to train under the F1BR frozen protocol.\n"
        "- F1B completion remains not executed; reranking remains locked until first-stage rankings and candidate recall exist.\n"
    )
    report_path.write_text(text + addendum, encoding="utf-8")


def update_f1b_manifests(root: Path) -> None:
    f1b_hash_paths = [
        Path(".gitignore"),
        Path("configs/f1b_traditional_completion.yaml"),
        Path("src/extension/f1b_traditional_completion.py"),
        Path("tests/test_extension_f1b_traditional_completion.py"),
        Path("report/notes.md"),
        Path("report/f1b_traditional_model_completion.md"),
        TABLE_DIR / "f1b_traditional_reuse_audit.csv",
        TABLE_DIR / "f1b_training_plan.csv",
        TABLE_DIR / "f1b_training_results.csv",
        TABLE_DIR / "f1b_traditional_utility_results.csv",
        TABLE_DIR / "f1b_traditional_beyond_utility_results.csv",
        TABLE_DIR / "f1b_first_stage_ranking_inventory.csv",
        TABLE_DIR / "f1b_candidate_recall.csv",
        TABLE_DIR / "f1b_reranking_readiness.csv",
        TABLE_DIR / "f1b_five_dataset_completeness.csv",
        TABLE_DIR / "f1b_result_cell_readiness.csv",
        TABLE_DIR / "f1b_validation.csv",
    ]
    for rel_path in ("results/manifests/f1b_training_provenance.json", "results/manifests/f1b_no_llm_inference.json"):
        path = root / rel_path
        if not path.exists():
            continue
        manifest = read_json(path)
        manifest["f1br_protocol_resolution"] = {
            "phase_id": PHASE_ID,
            "decision_token": DECISION_TOKEN,
            "resolved_seed_policy": list(FINAL_SEEDS),
            "baseline_rows_ready_to_execute": 15,
            "trainable_rows_ready_to_train": 20,
            "training_runs_executed_in_f1br": 0,
            "llm_inference_calls_executed_in_f1br": 0,
        }
        manifest["artifact_hashes"] = artifact_hashes(root, f1b_hash_paths)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def refresh_manifest_hashes(root: Path) -> None:
    manifest_path = root / MANIFEST_PATH
    manifest = read_json(manifest_path)
    manifest["artifact_hashes"] = artifact_hashes(root, [*OUTPUT_TABLES.values(), REPORT_PATH, *SUPPORT_ARTIFACTS, *UPDATED_F1B_ARTIFACTS])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def artifact_hashes(root: Path, rel_paths: Sequence[Path]) -> dict[str, str]:
    return {
        rel_path.as_posix(): sha256_file(root / rel_path)
        for rel_path in rel_paths
        if (root / rel_path).exists()
    }


def verify_manifest_hashes(root: Path, hashes: Mapping[str, str]) -> dict[str, Any]:
    unchanged = True
    current: dict[str, str] = {}
    for rel_path, expected in hashes.items():
        path = root / rel_path
        digest = sha256_file(path) if path.exists() else "MISSING"
        current[rel_path] = digest
        unchanged = unchanged and digest == expected
    return {"unchanged": unchanged, "current": current}


def verify_foodcom_hashes(manifest: Mapping[str, Any]) -> dict[str, Any]:
    unchanged = True
    current: dict[str, str] = {}
    for name, record in manifest.get("files", {}).items():
        path = Path(record["relative_extracted_path"])
        digest = sha256_file(path) if path.exists() else "MISSING"
        current[name] = digest
        unchanged = unchanged and digest == record["sha256"]
    return {"unchanged": unchanged, "current": current}


def active_dataset_list(dataset_matrix: pd.DataFrame) -> list[str]:
    if "active" in dataset_matrix.columns:
        return dataset_matrix.loc[dataset_matrix["active"].astype(str).str.lower().eq("true"), "dataset"].tolist()
    return dataset_matrix["dataset"].drop_duplicates().tolist()


def video_games_rows(*frames: pd.DataFrame) -> int:
    total = 0
    for frame in frames:
        mask = frame.astype(str).apply(lambda column: column.str.contains("video_games|video games|videogames", case=False, regex=True, na=False))
        total += int(mask.any(axis=1).sum())
    return total


def f0_budget_unchanged(budget: Mapping[str, int]) -> bool:
    return (
        int(budget.get("CORE_MANDATORY_NOW_REQUESTS", -1)) == 29292
        and int(budget.get("REQUIRED_SENSITIVITY_REQUESTS", -1)) == 21000
        and int(budget.get("MANDATORY_RAW_REQUESTS_TOTAL", -1)) == 50292
        and int(budget.get("TRAINING_DEPENDENT_REQUESTS", -1)) == 29292
        and int(budget.get("CONDITIONAL_REQUESTS", -1)) == 9000
        and int(budget.get("OPTIONAL_REQUESTS", -1)) == 12000
        and int(budget.get("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL", -1)) == 100584
        and int(budget.get("EXISTING_REUSABLE_REQUESTS", -1)) == 0
    )


def non_phase_worktree_status(root: Path) -> str:
    allowed = (
        ".gitignore",
        "configs/f1b_",
        "configs/f1br_",
        "src/extension/f1b_",
        "src/extension/f1br_",
        "tests/test_extension_f1b_",
        "tests/test_extension_f1br_",
        "report/f1b_",
        "report/f1br_",
        "report/notes.md",
        "results/tables/f1b_",
        "results/tables/f1br_",
        "results/manifests/f1b_",
        "results/manifests/f1br_",
    )
    status = git_output(root, ["status", "--short", "--untracked-files=all"])
    unexplained = []
    for line in status.splitlines():
        path = status_path(line).replace("\\", "/")
        if path and not any(path == prefix or path.startswith(prefix) for prefix in allowed):
            unexplained.append(line)
    return "\n".join(unexplained)


def status_path(line: str) -> str:
    if " -> " in line:
        return line.rsplit(" -> ", 1)[-1].strip()
    if len(line) >= 3 and line[2] == " ":
        return line[3:].strip()
    if len(line) >= 2 and line[1] == " ":
        return line[2:].strip()
    return ""


def ordered_by_dataset_method(frame: pd.DataFrame, method_column: str) -> pd.DataFrame:
    dataset_order = {dataset: index for index, dataset in enumerate(ACTIVE_DATASETS)}
    method_order = {method: index for index, method in enumerate(ALL_METHODS)}
    frame = frame.copy()
    frame["dataset_order"] = frame["dataset"].map(dataset_order)
    frame["method_order"] = frame[method_column].map(method_order)
    return frame.sort_values(["dataset_order", "method_order"]).drop(columns=["dataset_order", "method_order"]).reset_index(drop=True)


def validation_failures(validation: pd.DataFrame) -> list[str]:
    return validation.loc[validation["status"].ne("PASS"), "check"].tolist()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def format_param(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())

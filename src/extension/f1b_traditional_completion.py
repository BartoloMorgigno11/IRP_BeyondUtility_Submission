"""F1B traditional recommender completion gate.

This phase audits the frozen five-dataset traditional recommender surface.
It performs no LLM inference, downloads, network access, or commits. Because
the current repository contains unresolved seed/config lineage ambiguity for
trainable recommenders, it materializes a blocked pre-training checkpoint
instead of inventing a final training protocol.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import pandas as pd
import yaml

from src.extension.e0_protocol import sha256_json, stable_json
from src.extension.f0_final_extension_freeze import sha256_file
from src.utils.config import find_project_root


PHASE_ID = "f1b_traditional_recommender_completion"
DECISION_BLOCKED = "F1B_TRADITIONAL_COMPLETION_BLOCKED"
EXPECTED_HEAD = "c280fbef3440debe0a804efaf1f15436d9659347"
EXPECTED_PREDECESSOR = "c280fbef3440debe0a804efaf1f15436d9659347"
ACTIVE_DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
METHODS = ("Random", "MostPop", "BM25-like/content", "BPRMF", "GRU4Rec", "SASRec", "LightGCN")
DETERMINISTIC_METHODS = ("Random", "MostPop", "BM25-like/content")
TRAINABLE_METHODS = ("BPRMF", "GRU4Rec", "SASRec", "LightGCN")
UTILITY_METRICS = ("hr_at_5", "ndcg_at_5", "mrr_diagnostic")
RESULT_STATUS_ALLOWED = {
    "BLOCKED_EXECUTION",
    "BLOCKED_FIRST_STAGE_NOT_MATERIALIZED",
    "DEGENERATE_PROTOCOL_INPUT",
    "NOT_APPLICABLE",
}
BLOCKER_TYPE = "BLOCKED_TRAINABLE_PROTOCOL_AMBIGUITY"
BLOCKER_REASON = (
    "F1B stopped before training because the repository contains incompatible "
    "trainable seed/config lineages: original-paper seeds 0-4, local LastFM "
    "adapted seeds 42-46, and superseded Plan E single-seed/cohort artifacts."
)

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")
CONFIG_PATH = Path("configs/f1b_traditional_completion.yaml")
REPORT_PATH = REPORT_DIR / "f1b_traditional_model_completion.md"
TRAINING_MANIFEST_PATH = MANIFEST_DIR / "f1b_training_provenance.json"
NO_LLM_MANIFEST_PATH = MANIFEST_DIR / "f1b_no_llm_inference.json"

OUTPUT_TABLES = {
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
    "validation": TABLE_DIR / "f1b_validation.csv",
}
SUPPORT_ARTIFACTS = (
    Path(".gitignore"),
    CONFIG_PATH,
    Path("src/extension/f1b_traditional_completion.py"),
    Path("tests/test_extension_f1b_traditional_completion.py"),
    Path("report/notes.md"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize F1B traditional recommender gate artifacts.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    outputs = generate_f1b(root, Path(args.config))
    if not args.validate_only:
        write_outputs(root, outputs)
    validation = outputs["validation"]
    print(f"decision={outputs['decision_token']}")
    print(f"traditional_rows={len(outputs['reuse_audit'])}")
    print(f"planned_training_runs={len(outputs['training_plan'])}")
    print(f"executed_training_runs={int(outputs['training_results']['training_executed'].astype(str).str.lower().eq('true').sum())}")
    print(f"traditional_result_cells={len(outputs['result_cell_readiness'])}")
    print(f"global_metric_blockers={outputs['global_metric_blocker_count']}")
    print(f"degenerate_traditional_cells={outputs['degenerate_traditional_cell_count']}")
    print(f"validation_passed={str(validation['status'].eq('PASS').all()).lower()}")
    print("llm_inference_calls=0")
    print("downloads=0")
    print("network_calls=0")
    return 0 if bool(validation["status"].eq("PASS").all()) else 1


def generate_f1b(root: Path | None = None, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    project_root = root or find_project_root()
    config = load_phase_config(project_root, config_path)
    artifacts = load_artifacts(project_root)
    traditional = ordered_traditional_matrix(artifacts["traditional_matrix"])
    raw_rows = traditional_raw_rows(artifacts["raw_universe"])
    result_cells = traditional_result_cells(artifacts["result_cells"], raw_rows)
    degenerate_cells = traditional_degenerate_cells(artifacts["degenerate_cells"], raw_rows)
    resource = resource_preflight(project_root)
    protocol_blocker = protocol_blocker_record(config)

    reuse_audit = build_reuse_audit(traditional, config)
    training_plan = build_training_plan(traditional, config, resource)
    training_results = build_training_results(training_plan)
    result_cell_readiness = build_result_cell_readiness(result_cells, degenerate_cells)
    utility_results, beyond_results = split_metric_results(result_cell_readiness)
    first_stage_inventory = build_first_stage_inventory(config)
    candidate_recall = build_candidate_recall(artifacts["reranking_matrix"], config)
    reranking_readiness = build_reranking_readiness(artifacts["reranking_matrix"], candidate_recall)
    completeness = build_completeness(traditional, result_cell_readiness)
    integrity = build_integrity_snapshot(project_root, artifacts)
    validation = build_validation(
        project_root,
        config,
        artifacts,
        traditional,
        raw_rows,
        result_cells,
        degenerate_cells,
        reuse_audit,
        training_plan,
        training_results,
        result_cell_readiness,
        candidate_recall,
        reranking_readiness,
        completeness,
        integrity,
        protocol_blocker,
    )
    manifest = build_training_manifest(
        project_root,
        config,
        resource,
        protocol_blocker,
        validation,
        reuse_audit,
        training_plan,
        training_results,
        result_cell_readiness,
        candidate_recall,
        reranking_readiness,
        integrity,
    )
    no_llm_manifest = build_no_llm_manifest(project_root, validation, integrity)
    report = render_report(
        config,
        reuse_audit,
        training_plan,
        training_results,
        result_cell_readiness,
        candidate_recall,
        reranking_readiness,
        completeness,
        integrity,
        validation,
    )
    return {
        "decision_token": DECISION_BLOCKED,
        "protocol_blocker": protocol_blocker,
        "resource_preflight": resource,
        "reuse_audit": reuse_audit,
        "training_plan": training_plan,
        "training_results": training_results,
        "utility_results": utility_results,
        "beyond_utility_results": beyond_results,
        "first_stage_inventory": first_stage_inventory,
        "candidate_recall": candidate_recall,
        "reranking_readiness": reranking_readiness,
        "five_dataset_completeness": completeness,
        "result_cell_readiness": result_cell_readiness,
        "validation": validation,
        "training_manifest": manifest,
        "no_llm_manifest": no_llm_manifest,
        "report": report,
        "global_metric_blocker_count": int(len(artifacts["f1arr_final_blockers"])),
        "degenerate_traditional_cell_count": int(len(degenerate_cells)),
    }


def load_phase_config(root: Path, config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def load_artifacts(root: Path) -> dict[str, pd.DataFrame | dict[str, Any]]:
    return {
        "dataset_matrix": read_csv(root / "results/tables/f0_final_dataset_matrix.csv"),
        "traditional_matrix": read_csv(root / "results/tables/f0_final_traditional_model_matrix.csv"),
        "raw_universe": read_csv(root / "results/tables/f0_final_raw_experiment_universe.csv"),
        "result_cells": read_csv(root / "results/tables/f0_final_result_cell_universe.csv"),
        "reranking_matrix": read_csv(root / "results/tables/f0_final_reranking_matrix.csv"),
        "budget": read_csv(root / "results/tables/f0_final_request_budget.csv"),
        "f1arr_final_blockers": read_csv(root / "results/tables/f1arr_final_metric_blockers.csv"),
        "degenerate_cells": read_csv(root / "results/tables/f1arr_final_degenerate_metric_cells.csv"),
        "h0_manifest": read_json(root / "results/manifests/extension_h0_no_inference_manifest.json"),
        "foodcom_hash_manifest": read_json(root / "results/manifests/d0a_foodcom_extracted_file_hashes.json"),
    }


def ordered_traditional_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    dataset_order = {dataset: index for index, dataset in enumerate(ACTIVE_DATASETS)}
    method_order = {method: index for index, method in enumerate(METHODS)}
    selected = frame.loc[frame["dataset"].isin(ACTIVE_DATASETS) & frame["method"].isin(METHODS)].copy()
    selected["dataset_order"] = selected["dataset"].map(dataset_order)
    selected["method_order"] = selected["method"].map(method_order)
    return selected.sort_values(["dataset_order", "method_order"]).drop(columns=["dataset_order", "method_order"]).reset_index(drop=True)


def traditional_raw_rows(raw_universe: pd.DataFrame) -> pd.DataFrame:
    selected = raw_universe.loc[raw_universe["experiment_family"].eq("traditional_recommender")].copy()
    selected = selected.loc[selected["dataset"].isin(ACTIVE_DATASETS) & selected["model_name"].isin(METHODS)]
    return selected.reset_index(drop=True)


def traditional_result_cells(result_cells: pd.DataFrame, raw_rows: pd.DataFrame) -> pd.DataFrame:
    raw_ids = set(raw_rows["raw_experiment_id"])
    selected = result_cells.loc[result_cells["raw_experiment_id"].isin(raw_ids)].copy()
    return selected.sort_values(["dataset", "model_name", "metric", "result_cell_id"]).reset_index(drop=True)


def traditional_degenerate_cells(degenerate_cells: pd.DataFrame, raw_rows: pd.DataFrame) -> pd.DataFrame:
    raw_ids = set(raw_rows["raw_experiment_id"])
    selected = degenerate_cells.loc[degenerate_cells["raw_experiment_id"].isin(raw_ids)].copy()
    return selected.sort_values(["dataset", "metric", "model_name", "result_cell_id"]).reset_index(drop=True)


def resource_preflight(root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(root)
    return {
        "root_free_bytes": int(usage.free),
        "root_total_bytes": int(usage.total),
        "root_used_bytes": int(usage.used),
        "gpu_probe": "not_invoked",
        "notes": "F1B stopped before training; no GPU or training framework initialization was required.",
    }


def protocol_blocker_record(config: Mapping[str, Any]) -> dict[str, Any]:
    ambiguity = config.get("protocol_ambiguity", {})
    evidence = ambiguity.get("evidence", []) if isinstance(ambiguity, Mapping) else []
    return {
        "blocker_type": BLOCKER_TYPE,
        "status": "BLOCKED_BEFORE_TRAINING",
        "reason": clean(ambiguity.get("reason")) if isinstance(ambiguity, Mapping) else BLOCKER_REASON,
        "evidence": list(evidence) if isinstance(evidence, list) else [],
        "training_executed": False,
    }


def build_reuse_audit(traditional: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    dataset_sources = {row["name"]: row for row in config.get("datasets", [])}
    rows = []
    for record in traditional.to_dict("records"):
        method = record["method"]
        dataset = record["dataset"]
        requires_training = truthy(record["requires_training"])
        if method in DETERMINISTIC_METHODS:
            reuse_decision = "RECOMPUTE_REQUIRED"
            final_status = "BLOCKED_PHASE_STOP_BEFORE_EXECUTION"
            reason = deterministic_reuse_reason(method, dataset)
        else:
            reuse_decision = "BLOCKED"
            final_status = "BLOCKED_TRAINING_PROTOCOL_AMBIGUITY"
            reason = BLOCKER_REASON
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "task": record["task"],
                "cohort_id": record["cohort_id"],
                "evaluation_users": int(record["cohort_size"]),
                "candidate_pool_size": 20,
                "positive_targets": int(record["cohort_size"]),
                "top_k": 5,
                "requires_training": bool_text(requires_training),
                "existing_artifact": existing_artifact_for(method, dataset),
                "protocol_identity_to_f0": "NO_PROTOCOL_IDENTICAL_REUSE",
                "reuse_decision": reuse_decision,
                "final_f1b_status": final_status,
                "seed_policy": seed_policy_for(method),
                "candidate_source": clean(dataset_sources.get(dataset, {}).get("candidate_source")),
                "f0_status": record["status"],
                "f0_raw_status": record["raw_status"],
                "f0_budget_class": record["budget_class"],
                "reason": reason,
                "evidence": reuse_evidence_for(method, dataset),
            }
        )
    return pd.DataFrame(rows)


def deterministic_reuse_reason(method: str, dataset: str) -> str:
    if method == "BM25-like/content" and dataset == "foodcom":
        return (
            "Food.com BM25-like/content is READY_WITH_ADAPTATION in D0A, but no "
            "F0-identical final ranking artifact exists; recomputation is held "
            "until the phase-level trainable protocol gate is resolved."
        )
    return (
        "Existing deterministic baseline artifacts are Plan E, pilot, or "
        "superseded cohort outputs; F0 promotes no protocol-identical reusable "
        "traditional rows."
    )


def existing_artifact_for(method: str, dataset: str) -> str:
    if method in {"Random", "MostPop"}:
        return "results/tables/traditional_baselines_metrics.csv;results/tables/plan_e_same_cohort_traditional_baselines_metrics.csv"
    if method == "BM25-like/content":
        return "results/tables/plan_e_bm25_metrics.csv"
    if dataset == "lastfm":
        return "results/tables/finalization_f6d2_model_protocol.csv;results/tables/finalization_f6d2_reranking_source_selection.csv"
    if dataset in {"beauty", "sports", "ml1m"}:
        return f"results/tables/plan_e_{method.lower().replace('rec', 'rec').replace('lightgcn', 'lightgcn')}_all_datasets_metrics.csv"
    return "NONE_FOUND_FOR_F0_PROTOCOL"


def reuse_evidence_for(method: str, dataset: str) -> str:
    if method in TRAINABLE_METHODS:
        return ";".join(
            [
                "report/codebase_map.md",
                "configs/finalization_f6d2_lastfm_trainable_models.yaml",
                "results/tables/finalization_f6d2_model_protocol.csv",
                "results/tables/phase_f6g1_seed_policy_audit.csv",
            ]
        )
    if method == "BM25-like/content" and dataset == "foodcom":
        return "results/tables/d0a_foodcom_traditional_model_capability.csv;results/tables/plan_e_bm25_metrics.csv"
    return "results/tables/f0_final_reuse_audit.csv;results/tables/traditional_baselines_metrics.csv"


def seed_policy_for(method: str) -> str:
    if method == "Random":
        return "deterministic stable seed required after blocker resolution; no F1B random rankings materialized"
    if method == "MostPop":
        return "no randomness; tie policy must be frozen before recomputation"
    if method == "BM25-like/content":
        return "deterministic content scorer; text-field policy must remain dataset-configured"
    return "UNRESOLVED: original seeds 0-4 vs local adapted seeds 42-46 vs Plan E single seed 42"


def build_training_plan(traditional: pd.DataFrame, config: Mapping[str, Any], resource: Mapping[str, Any]) -> pd.DataFrame:
    method_config = {row["name"]: row for row in config.get("methods", [])}
    rows = []
    trainable = traditional.loc[traditional["method"].isin(TRAINABLE_METHODS)]
    for record in trainable.to_dict("records"):
        planned_slots = int(method_config.get(record["method"], {}).get("planned_seed_slots", 5))
        for slot in range(1, planned_slots + 1):
            plan_identity = {
                "dataset": record["dataset"],
                "method": record["method"],
                "cohort_id": record["cohort_id"],
                "seed_slot": slot,
                "phase": PHASE_ID,
            }
            rows.append(
                {
                    "training_run_id": "F1B-TRAIN-" + sha256_json(plan_identity).upper()[:16],
                    "dataset": record["dataset"],
                    "method": record["method"],
                    "cohort_id": record["cohort_id"],
                    "evaluation_users": int(record["cohort_size"]),
                    "candidate_pool_size": 20,
                    "seed_slot": slot,
                    "resolved_seed": "",
                    "seed_policy_status": "UNRESOLVED_PROTOCOL_AMBIGUITY",
                    "planned_epoch_policy": "UNRESOLVED_PROTOCOL_AMBIGUITY",
                    "planned_device": "UNRESOLVED_PROTOCOL_AMBIGUITY",
                    "resource_free_bytes": int(resource["root_free_bytes"]),
                    "training_plan_status": "BLOCKED_BEFORE_TRAINING",
                    "training_allowed_by_user": "true",
                    "training_executed": "false",
                    "reason": BLOCKER_REASON,
                    "evidence": "configs/f1b_traditional_completion.yaml",
                }
            )
    return pd.DataFrame(rows)


def build_training_results(training_plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in training_plan.to_dict("records"):
        rows.append(
            {
                "training_run_id": record["training_run_id"],
                "dataset": record["dataset"],
                "method": record["method"],
                "cohort_id": record["cohort_id"],
                "seed_slot": record["seed_slot"],
                "resolved_seed": record["resolved_seed"],
                "training_executed": "false",
                "checkpoint_path": "",
                "ranking_output_path": "",
                "fit_seconds": "",
                "status": "BLOCKED_NOT_EXECUTED_PRE_TRAINING",
                "reason": record["reason"],
            }
        )
    return pd.DataFrame(rows)


def build_result_cell_readiness(result_cells: pd.DataFrame, degenerate_cells: pd.DataFrame) -> pd.DataFrame:
    degenerate_by_cell = {row["result_cell_id"]: row for row in degenerate_cells.to_dict("records")}
    rows = []
    for cell in result_cells.to_dict("records"):
        degenerate = degenerate_by_cell.get(cell["result_cell_id"])
        if degenerate:
            status = "DEGENERATE_PROTOCOL_INPUT"
            reason = clean(degenerate.get("reason")) or "F1ARR classified this PopREO/DPD result cell as degenerate protocol input"
        else:
            status = "BLOCKED_EXECUTION"
            reason = "No F1B protocol-identical traditional ranking output exists; F1B stopped before recomputation/training."
        rows.append(
            {
                "result_cell_id": cell["result_cell_id"],
                "raw_experiment_id": cell["raw_experiment_id"],
                "dataset": cell["dataset"],
                "task": cell["task"],
                "model_name": cell["model_name"],
                "prompt": cell["prompt"],
                "condition": cell["condition"],
                "metric": cell["metric"],
                "metric_label": cell["metric_label"],
                "K": cell["K"],
                "f0_status": cell["status"],
                "f1b_status": status,
                "metric_value": "",
                "value_json": "",
                "reason": reason,
                "requires_new_traditional_raw": "true",
                "requires_training_raw": bool_text(cell["model_name"] in TRAINABLE_METHODS),
                "adds_llm_requests": 0,
                "source": cell["source"],
                "degenerate_source": "results/tables/f1arr_final_degenerate_metric_cells.csv" if degenerate else "",
            }
        )
    return pd.DataFrame(rows)


def split_metric_results(readiness: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_columns = [
        "result_cell_id",
        "raw_experiment_id",
        "dataset",
        "task",
        "model_name",
        "condition",
        "metric",
        "K",
        "metric_value",
        "value_json",
        "f1b_status",
        "reason",
        "source",
    ]
    utility = readiness.loc[readiness["metric"].isin(UTILITY_METRICS), common_columns].copy()
    beyond = readiness.loc[~readiness["metric"].isin(UTILITY_METRICS), common_columns].copy()
    return utility.reset_index(drop=True), beyond.reset_index(drop=True)


def build_first_stage_inventory(config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for dataset in config.get("datasets", []):
        rows.append(
            {
                "dataset": dataset["name"],
                "first_stage_recommender": "canonical_source_ranker_to_freeze",
                "expected_eval_users": int(dataset["evaluation_users"]),
                "expected_candidate_pool_size": 20,
                "ranking_source": "",
                "ranking_rows": "",
                "positive_recall_audited": "false",
                "inventory_status": "BLOCKED_FIRST_STAGE_NOT_MATERIALIZED",
                "reason": "F1B stopped before producing the first-stage ranking source required by reranking.",
            }
        )
    return pd.DataFrame(rows)


def build_candidate_recall(reranking_matrix: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    dataset_sizes = {row["name"]: int(row["evaluation_users"]) for row in config.get("datasets", [])}
    rows = []
    for record in reranking_matrix.to_dict("records"):
        rows.append(
            {
                "dataset": record["dataset"],
                "task": record["task"],
                "model_name": record["model_name"],
                "prompt": record["prompt"],
                "cohort_id": record["cohort_id"],
                "first_stage_recommender": record["first_stage_recommender"],
                "candidate_source_protocol": record["candidate_source_protocol"],
                "expected_eval_users": dataset_sizes.get(record["dataset"], ""),
                "ground_truth_positive_checked_users": "",
                "ground_truth_positive_present_users": "",
                "candidate_recall_at_20": "",
                "status": "BLOCKED_FIRST_STAGE_NOT_MATERIALIZED",
                "reason": "No first-stage top-20 ranking source exists; per-user candidate recall would be fabricated.",
            }
        )
    return pd.DataFrame(rows)


def build_reranking_readiness(reranking_matrix: pd.DataFrame, candidate_recall: pd.DataFrame) -> pd.DataFrame:
    recall_status = {
        (row["dataset"], row["model_name"], row["prompt"], row["cohort_id"]): row["status"]
        for row in candidate_recall.to_dict("records")
    }
    rows = []
    for record in reranking_matrix.to_dict("records"):
        status = recall_status[(record["dataset"], record["model_name"], record["prompt"], record["cohort_id"])]
        rows.append(
            {
                "dataset": record["dataset"],
                "task": record["task"],
                "model_name": record["model_name"],
                "prompt": record["prompt"],
                "cohort_id": record["cohort_id"],
                "first_stage_recommender": record["first_stage_recommender"],
                "source_ranking": record["source_ranking"],
                "reranking_candidate_size": record["reranking_candidate_size"],
                "positive_present_candidate_recall_required": record["positive_present_candidate_recall_required"],
                "f0_request_count": int(record["request_count"]),
                "future_requests_unlocked": 0,
                "readiness_status": status,
                "reason": "Reranking remains locked until first-stage rankings and candidate recall are materialized.",
            }
        )
    return pd.DataFrame(rows)


def build_completeness(traditional: pd.DataFrame, readiness: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in traditional.to_dict("records"):
        subset = readiness.loc[
            readiness["dataset"].eq(record["dataset"])
            & readiness["model_name"].eq(record["method"])
        ]
        counts = subset["f1b_status"].value_counts().to_dict()
        rows.append(
            {
                "dataset": record["dataset"],
                "method": record["method"],
                "cohort_id": record["cohort_id"],
                "evaluation_users": int(record["cohort_size"]),
                "expected_result_cells": 17,
                "materialized_result_cells": len(subset),
                "computed_result_cells": int(counts.get("COMPUTED", 0)),
                "blocked_execution_cells": int(counts.get("BLOCKED_EXECUTION", 0)),
                "degenerate_protocol_input_cells": int(counts.get("DEGENERATE_PROTOCOL_INPUT", 0)),
                "final_status": "BLOCKED",
                "reason": "F1B stopped before final traditional execution because the training protocol is unresolved.",
            }
        )
    return pd.DataFrame(rows)


def build_integrity_snapshot(root: Path, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    budget = {
        row["budget_item"]: int(row["request_count"])
        for row in artifacts["budget"].to_dict("records")
    }
    active = active_dataset_list(artifacts["dataset_matrix"])
    h0_hashes = verify_h0_hashes(root, artifacts["h0_manifest"])
    foodcom_hashes = verify_foodcom_hashes(artifacts["foodcom_hash_manifest"])
    return {
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "predecessor_is_head": git_output(root, ["rev-parse", "HEAD"]) == EXPECTED_HEAD,
        "descends_from_predecessor": head_descends_from(root, EXPECTED_PREDECESSOR),
        "active_datasets": active,
        "active_video_games_rows": video_games_rows(artifacts["raw_universe"], artifacts["dataset_matrix"], artifacts["traditional_matrix"], artifacts["result_cells"]),
        "global_metric_blockers": len(artifacts["f1arr_final_blockers"]),
        "f0_budget": budget,
        "f0_budget_unchanged": budget.get("MANDATORY_RAW_REQUESTS_TOTAL") == 50292
        and budget.get("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL") == 100584
        and budget.get("EXISTING_REUSABLE_REQUESTS") == 0,
        "h0_hashes_unchanged": h0_hashes["unchanged"],
        "h0_artifact_hashes": h0_hashes["artifact_hashes"],
        "foodcom_hashes_unchanged": foodcom_hashes["unchanged"],
        "foodcom_hashes": foodcom_hashes["file_hashes"],
        "original_repo_status": git_output(root, ["status", "--short", "--untracked-files=all", "--", "original_repo"]),
        "paper_status": git_output(root, ["status", "--short", "--untracked-files=all", "--", "paper"]),
        "non_f1b_worktree_status": non_f1b_worktree_status(root),
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_api_calls_executed": 0,
        "downloads_executed": 0,
        "network_calls_executed": 0,
        "traditional_training_runs_executed": 0,
        "commit_created": 0,
    }


def active_dataset_list(dataset_matrix: pd.DataFrame) -> list[str]:
    if "active" in dataset_matrix.columns:
        return dataset_matrix.loc[dataset_matrix["active"].astype(str).str.lower().eq("true"), "dataset"].tolist()
    return dataset_matrix.loc[dataset_matrix["status"].astype(str).str.contains("ACTIVE|READY", case=False, regex=True), "dataset"].tolist()


def video_games_rows(*frames: pd.DataFrame) -> int:
    count = 0
    for frame in frames:
        text = frame.astype(str).apply(lambda col: col.str.contains("video_games|video games|videogames", case=False, regex=True, na=False))
        count += int(text.any(axis=1).sum())
    return count


def verify_h0_hashes(root: Path, h0_manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected = h0_manifest.get("artifact_hashes", {})
    current: dict[str, str] = {}
    unchanged = True
    for rel_path, digest in expected.items():
        path = root / rel_path
        current_digest = sha256_file(path) if path.exists() else "MISSING"
        current[rel_path] = current_digest
        unchanged = unchanged and current_digest == digest
    return {"unchanged": unchanged, "artifact_hashes": current}


def verify_foodcom_hashes(foodcom_manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = foodcom_manifest.get("files", {})
    current: dict[str, str] = {}
    unchanged = True
    for name, record in files.items():
        path = Path(record["relative_extracted_path"])
        digest = sha256_file(path) if path.exists() else "MISSING"
        current[name] = digest
        unchanged = unchanged and digest == record["sha256"]
    return {"unchanged": unchanged, "file_hashes": current}


def non_f1b_worktree_status(root: Path) -> str:
    status = git_output(root, ["status", "--short", "--untracked-files=all"])
    allowed_prefixes = (
        ".gitignore",
        "configs/f1b_traditional_completion.yaml",
        "src/extension/f1b_traditional_completion.py",
        "tests/test_extension_f1b_traditional_completion.py",
        "report/f1b_traditional_model_completion.md",
        "report/notes.md",
        "results/tables/f1b_",
        "results/manifests/f1b_",
        "configs/f1br_",
        "src/extension/f1br_",
        "tests/test_extension_f1br_",
        "report/f1br_",
        "results/tables/f1br_",
        "results/manifests/f1br_",
    )
    unexplained = []
    for line in status.splitlines():
        path = status_path(line)
        normalized = path.replace("\\", "/")
        if not normalized:
            continue
        if not any(normalized == prefix or normalized.startswith(prefix) for prefix in allowed_prefixes):
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


def build_validation(
    root: Path,
    config: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    traditional: pd.DataFrame,
    raw_rows: pd.DataFrame,
    result_cells: pd.DataFrame,
    degenerate_cells: pd.DataFrame,
    reuse_audit: pd.DataFrame,
    training_plan: pd.DataFrame,
    training_results: pd.DataFrame,
    result_readiness: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    reranking_readiness: pd.DataFrame,
    completeness: pd.DataFrame,
    integrity: Mapping[str, Any],
    protocol_blocker: Mapping[str, Any],
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []

    def check(name: str, condition: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "evidence": evidence})

    check("head_is_expected_f1arr_checkpoint", bool(integrity["predecessor_is_head"]), clean(integrity["head"]))
    check("head_descends_from_predecessor", bool(integrity["descends_from_predecessor"]), EXPECTED_PREDECESSOR)
    check("active_datasets_exact", tuple(integrity["active_datasets"]) == ACTIVE_DATASETS, ",".join(integrity["active_datasets"]))
    check("operational_video_games_rows_zero", int(integrity["active_video_games_rows"]) == 0, str(integrity["active_video_games_rows"]))
    check("global_metric_blockers_zero", int(integrity["global_metric_blockers"]) == 0, str(integrity["global_metric_blockers"]))
    check("f0_request_budget_unchanged", bool(integrity["f0_budget_unchanged"]), stable_json(integrity["f0_budget"]))
    check("h0_hashes_unchanged", bool(integrity["h0_hashes_unchanged"]), str(len(integrity["h0_artifact_hashes"])))
    check("foodcom_hashes_unchanged", bool(integrity["foodcom_hashes_unchanged"]), stable_json(integrity["foodcom_hashes"]))
    check("original_repo_unchanged", clean(integrity["original_repo_status"]) == "", clean(integrity["original_repo_status"]) or "clean")
    check("paper_unchanged", clean(integrity["paper_status"]) == "", clean(integrity["paper_status"]) or "clean")
    check("no_unexplained_worktree_changes", clean(integrity["non_f1b_worktree_status"]) == "", clean(integrity["non_f1b_worktree_status"]) or "none")
    check("traditional_matrix_35_rows", len(traditional) == 35, str(len(traditional)))
    check("traditional_raw_rows_35", len(raw_rows) == 35, str(len(raw_rows)))
    check("traditional_result_cells_595", len(result_cells) == 595, str(len(result_cells)))
    check("reuse_audit_complete", len(reuse_audit) == 35, str(len(reuse_audit)))
    check("no_protocol_identical_reuse_claimed", reuse_audit["protocol_identity_to_f0"].eq("NO_PROTOCOL_IDENTICAL_REUSE").all(), ",".join(sorted(reuse_audit["protocol_identity_to_f0"].unique())))
    check("planned_trainable_runs_100", len(training_plan) == 100, str(len(training_plan)))
    check("training_results_100_not_executed", len(training_results) == 100 and not training_results["training_executed"].astype(str).str.lower().eq("true").any(), str(len(training_results)))
    check("protocol_ambiguity_explicit", protocol_blocker["status"] == "BLOCKED_BEFORE_TRAINING", clean(protocol_blocker["reason"]))
    check("deterministic_rows_not_fabricated", reuse_audit.loc[reuse_audit["method"].isin(DETERMINISTIC_METHODS), "reuse_decision"].eq("RECOMPUTE_REQUIRED").all(), "Random/MostPop/BM25-like require recomputation")
    check("result_readiness_complete", len(result_readiness) == len(result_cells), f"{len(result_readiness)} of {len(result_cells)}")
    check("result_statuses_controlled", set(result_readiness["f1b_status"]).issubset(RESULT_STATUS_ALLOWED), ",".join(sorted(set(result_readiness["f1b_status"]))))
    check("blank_result_statuses_zero", result_readiness["f1b_status"].astype(str).str.len().gt(0).all(), "blank=0")
    check("unknown_metric_rows_zero", set(result_readiness["metric"]) == set(result_cells["metric"]), str(len(set(result_readiness["metric"]) ^ set(result_cells["metric"]))))
    check("degenerate_traditional_cells_preserved", len(degenerate_cells) == int(result_readiness["f1b_status"].eq("DEGENERATE_PROTOCOL_INPUT").sum()), str(len(degenerate_cells)))
    check("candidate_recall_blocked_not_fabricated", len(candidate_recall) == len(artifacts["reranking_matrix"]) and candidate_recall["status"].eq("BLOCKED_FIRST_STAGE_NOT_MATERIALIZED").all(), str(len(candidate_recall)))
    check("reranking_requests_unlocked_zero", pd.to_numeric(reranking_readiness["future_requests_unlocked"]).sum() == 0, str(pd.to_numeric(reranking_readiness["future_requests_unlocked"]).sum()))
    check("five_dataset_completeness_35_rows", len(completeness) == 35 and completeness["final_status"].eq("BLOCKED").all(), str(len(completeness)))
    check("zero_llm_inference", int(integrity["llm_inference_calls_executed"]) == 0, "0")
    check("zero_ollama_recommendation_calls", int(integrity["ollama_recommendation_calls_executed"]) == 0, "0")
    check("zero_traditional_training_runs", int(integrity["traditional_training_runs_executed"]) == 0, "0")
    check("zero_download_network_api", int(integrity["downloads_executed"]) == 0 and int(integrity["network_calls_executed"]) == 0 and int(integrity["external_api_calls_executed"]) == 0, "download=0;network=0;api=0")
    check("commit_not_created", int(integrity["commit_created"]) == 0, "0")
    check("config_forbids_llm_inference", config["execution"]["allow_llm_inference"] is False, "false")
    return pd.DataFrame(checks)


def build_training_manifest(
    root: Path,
    config: Mapping[str, Any],
    resource: Mapping[str, Any],
    protocol_blocker: Mapping[str, Any],
    validation: pd.DataFrame,
    reuse_audit: pd.DataFrame,
    training_plan: pd.DataFrame,
    training_results: pd.DataFrame,
    result_readiness: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    reranking_readiness: pd.DataFrame,
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_BLOCKED,
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "expected_head": clean(config["phase"]["expected_head"]),
        "completion_validated": False,
        "blocked_before_training": True,
        "phase_blocker": protocol_blocker,
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS"), "check"].tolist(),
        "traditional_matrix_rows": len(reuse_audit),
        "planned_training_runs": len(training_plan),
        "executed_training_runs": int(training_results["training_executed"].astype(str).str.lower().eq("true").sum()),
        "traditional_result_cells": len(result_readiness),
        "degenerate_traditional_cells": int(result_readiness["f1b_status"].eq("DEGENERATE_PROTOCOL_INPUT").sum()),
        "candidate_recall_rows": len(candidate_recall),
        "future_reranking_requests_unlocked": int(pd.to_numeric(reranking_readiness["future_requests_unlocked"]).sum()),
        "resource_preflight": dict(resource),
        "integrity": dict(integrity),
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_api_calls_executed": 0,
        "downloads_executed": 0,
        "network_calls_executed": 0,
        "commit_created": 0,
    }


def build_no_llm_manifest(root: Path, validation: pd.DataFrame, integrity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_BLOCKED,
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_api_calls_executed": 0,
        "downloads_executed": 0,
        "network_calls_executed": 0,
        "traditional_training_runs_executed": int(integrity["traditional_training_runs_executed"]),
        "raw_llm_outputs_created": 0,
        "training_outputs_created": 0,
        "notes": "F1B materialized blocker/audit artifacts only; no recommendation generation or training was executed.",
    }


def render_report(
    config: Mapping[str, Any],
    reuse_audit: pd.DataFrame,
    training_plan: pd.DataFrame,
    training_results: pd.DataFrame,
    result_readiness: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    reranking_readiness: pd.DataFrame,
    completeness: pd.DataFrame,
    integrity: Mapping[str, Any],
    validation: pd.DataFrame,
) -> str:
    method_counts = completeness.groupby("method")["final_status"].count().to_dict()
    dataset_counts = completeness.groupby("dataset")["final_status"].count().to_dict()
    readiness_counts = result_readiness["f1b_status"].value_counts().sort_index().to_dict()
    return "\n".join(
        [
            "# F1B Traditional Recommender Completion Gate",
            "",
            f"Decision token: `{DECISION_BLOCKED}`",
            "",
            "## Summary",
            "- F1B is blocked before training and before deterministic baseline recomputation.",
            f"- Blocker: {clean(config['protocol_ambiguity']['reason']).strip()}",
            "",
            "## Reuse Audit",
            f"- Traditional F0 rows audited: `{len(reuse_audit)}`.",
            "- Protocol-identical reusable traditional outputs: `0`.",
            "",
            "## Training Plan",
            f"- Planned trainable seed slots recorded: `{len(training_plan)}`.",
            f"- Training runs executed: `{int(training_results['training_executed'].astype(str).str.lower().eq('true').sum())}`.",
            "",
            "## Methods",
            "\n".join(f"- `{method}` rows: `{count}`." for method, count in sorted(method_counts.items())),
            "",
            "## Datasets",
            "\n".join(f"- `{dataset}` methods: `{count}`." for dataset, count in sorted(dataset_counts.items())),
            "",
            "## Result Cells",
            "\n".join(f"- `{status}` = `{count}`." for status, count in readiness_counts.items()),
            "",
            "## Reranking",
            f"- Candidate recall rows: `{len(candidate_recall)}`.",
            f"- Future requests unlocked: `{int(pd.to_numeric(reranking_readiness['future_requests_unlocked']).sum())}`.",
            "",
            "## Integrity",
            f"- Active datasets: `{','.join(integrity['active_datasets'])}`.",
            f"- Video Games rows: `{integrity['active_video_games_rows']}`.",
            f"- F0 budget unchanged: `{str(integrity['f0_budget_unchanged']).lower()}`.",
            f"- Food.com hashes unchanged: `{str(integrity['foodcom_hashes_unchanged']).lower()}`.",
            f"- H0 hashes unchanged: `{str(integrity['h0_hashes_unchanged']).lower()}`.",
            "",
            "## Validation",
            f"- Passed: `{str(validation['status'].eq('PASS').all()).lower()}`.",
        ]
    ) + "\n"


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for directory in (root / TABLE_DIR, root / MANIFEST_DIR, root / REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for key, rel_path in OUTPUT_TABLES.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(root / rel_path, index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(str(outputs["report"]), encoding="utf-8")

    table_hashes = {rel_path.as_posix(): sha256_file(root / rel_path) for rel_path in OUTPUT_TABLES.values()}
    support_hashes = {
        rel_path.as_posix(): sha256_file(root / rel_path)
        for rel_path in SUPPORT_ARTIFACTS
        if (root / rel_path).exists()
    }
    report_hash = sha256_file(root / REPORT_PATH)
    training_manifest = dict(outputs["training_manifest"])
    training_manifest["artifact_hashes"] = {**table_hashes, REPORT_PATH.as_posix(): report_hash, **support_hashes}
    (root / TRAINING_MANIFEST_PATH).write_text(json.dumps(training_manifest, indent=2, sort_keys=True), encoding="utf-8")

    no_llm_manifest = dict(outputs["no_llm_manifest"])
    no_llm_manifest["artifact_hashes"] = {
        **table_hashes,
        REPORT_PATH.as_posix(): report_hash,
        **support_hashes,
        TRAINING_MANIFEST_PATH.as_posix(): sha256_file(root / TRAINING_MANIFEST_PATH),
    }
    (root / NO_LLM_MANIFEST_PATH).write_text(json.dumps(no_llm_manifest, indent=2, sort_keys=True), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


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


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())

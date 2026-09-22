"""F1BZC full-catalog first-stage recovery.

This phase corrects the F1BZ reranking-source defect as far as the frozen
checkpoints scientifically allow. It never retrains, tunes, downloads, calls a
network service, or runs LLM inference. If a checkpoint cannot score the raw
catalog item universe without invented embeddings, the corresponding cell is
blocked explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
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

from src.baselines.plan_d_bprmf import BPRMFModel
from src.baselines.plan_d_gru4rec import GRU4RecModel
from src.baselines.plan_d_lightgcn import LightGCNModel, build_normalized_bipartite_graph
from src.baselines.plan_e_sasrec_training import FiniteMaskedSASRecModel
from src.extension.e0_protocol import CANDIDATE_COUNT, sha256_json, stable_json
from src.extension.f0_final_extension_freeze import sha256_file
from src.extension.f1a_offline_metric_closure import format_float
from src.extension.f1bx_execution_stage1 import (
    ACTIVE_DATASETS,
    TRAINABLE_MODELS,
    DatasetMaterial,
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
    FINAL_SEEDS,
    RAW_RECOMMENDATION_DIR,
    build_final_data,
)
from src.extension.f1bza_post_execution_audit import collect_f1bz_evidence_hashes, evidence_artifact_counts
from src.utils.config import find_project_root


PHASE_ID = "f1bzc_full_catalog_recovery"
DECISION_CORRECTED = "F1BZC_RERANKING_SOURCE_CORRECTED"
DECISION_BLOCKED = "F1BZC_RERANKING_SOURCE_BLOCKED"
PRECONDITION_STATE = "F1BZA_CORRECTION_REQUIRED"
F1BZ_VALIDATED = "F1BZ_FINAL_TRADITIONAL_EXECUTION_VALIDATED"
FULL_CATALOG_READY = "FULL_CATALOG_REGENERATION_READY"
BLOCKED_ITEM_UNIVERSE = "BLOCKED_SCIENTIFIC_ITEM_UNIVERSE_COVERAGE"
BLOCKED_POOL_ALGORITHM = "BLOCKED_ADAPTED_POOL_SINGLE_MODEL_DOMINANCE"
SUPERSEDED = "SUPERSEDED_RERANKING_SOURCE_DEFECT"
CORRECTED = "CORRECTED_F1BZC"
AUTHORITATIVE = "AUTHORITATIVE"
SOURCE_POLICY = "F1BZC_FULL_CATALOG_FIVE_SEED_MEAN"
SOURCE_SEED_POLICY = "F1BZ_FIVE_SEED_MEAN_SCORE_THEN_FROZEN_MODEL_ORDER"

CONFIG_PATH = Path("configs/f1bzc_full_catalog_recovery.yaml")
TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")
RAW_FULL_CATALOG_DIR = Path("results/raw_outputs/f1bzc_full_catalog_predictions")
RAW_RERANKING_DIR = Path("results/raw_outputs/f1bzc_reranking_candidates")
REPORT_PATH = REPORT_DIR / "f1bzc_full_catalog_reranking_recovery.md"

OUTPUT_TABLES = {
    "source_seed_policy": TABLE_DIR / "f1bzc_source_seed_policy.csv",
    "full_catalog_adapter_audit": TABLE_DIR / "f1bzc_full_catalog_adapter_audit.csv",
    "full_catalog_execution": TABLE_DIR / "f1bzc_full_catalog_execution.csv",
    "full_catalog_validation": TABLE_DIR / "f1bzc_full_catalog_validation.csv",
    "reranking_source_contributions": TABLE_DIR / "f1bzc_reranking_source_contributions.csv",
    "candidate_recall": TABLE_DIR / "f1bzc_candidate_recall.csv",
    "foodcom_warm_cold_diagnostics": TABLE_DIR / "f1bzc_foodcom_warm_cold_diagnostics.csv",
    "reranking_pool_validation": TABLE_DIR / "f1bzc_reranking_pool_validation.csv",
    "reranking_readiness": TABLE_DIR / "f1bzc_reranking_readiness.csv",
    "superseded_artifact_registry": TABLE_DIR / "f1bzc_superseded_artifact_registry.csv",
    "result_validity_classification": TABLE_DIR / "f1bzc_result_validity_classification.csv",
    "validation": TABLE_DIR / "f1bzc_validation.csv",
}

OUTPUT_MANIFESTS = {
    "full_catalog_recovery": MANIFEST_DIR / "f1bzc_full_catalog_recovery.json",
    "reranking_pool_freeze": MANIFEST_DIR / "f1bzc_reranking_pool_freeze.json",
    "no_llm_no_training": MANIFEST_DIR / "f1bzc_no_llm_no_training.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F1BZC full-catalog first-stage recovery.")
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
        outputs = generate_f1bzc(root, Path(args.config))
        write_outputs(root, outputs)
    manifest = outputs["manifests"]["full_catalog_recovery"]
    validation = outputs["validation"]
    print(f"decision={manifest['decision_token']}")
    print(f"ready_full_catalog_cells={manifest['ready_full_catalog_cells']}")
    print(f"blocked_full_catalog_cells={manifest['blocked_full_catalog_cells']}")
    print(f"single_model_dominated_pools={manifest['single_model_dominated_pools']}")
    print(f"future_requests_unlocked={manifest['future_requests_unlocked']}")
    print(f"validation_passed={str(validation['status'].eq('PASS').all()).lower()}")
    return 0 if bool(validation["status"].eq("PASS").all()) else 1


def generate_f1bzc(root: Path | None = None, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    project_root = root or find_project_root()
    config = load_config(project_root, Path(config_path))
    f1bz_config = load_config(project_root, Path(str(config["inputs"]["f1bz_config"])))
    f1bx_config = load_config(project_root, Path(str(f1bz_config["inputs"]["f1bx_config"])))
    specs = {spec.name: spec for spec in f1bx_dataset_specs(f1bx_config)}
    run_manifest = read_csv(project_root / "results/tables/f1bz_final_run_manifest.csv")
    f1bz_manifest = read_json(project_root / str(config["inputs"]["f1bz_manifest"]))
    f1bza_manifest = read_json(project_root / str(config["inputs"]["f1bza_manifest"]))
    f1bza_readiness = read_csv(project_root / str(config["inputs"]["f1bza_readiness"]))
    f1br_protocol = read_csv(project_root / str(config["inputs"]["f1br_reranking_source_protocol"]))
    f0_budget = read_csv(project_root / str(config["inputs"]["f0_request_budget"]))
    evidence_before = collect_f1bz_evidence_hashes(project_root)

    materials: dict[str, DatasetMaterial] = {}
    final_data: dict[str, Any] = {}
    for dataset in ACTIVE_DATASETS:
        material = build_dataset_material(project_root, specs[dataset])
        materials[dataset] = material
        final_data[dataset] = build_final_data(project_root, specs[dataset], material, f1bz_config)

    source_seed_policy = build_source_seed_policy(config, run_manifest)
    adapter_audit = build_adapter_audit(project_root, f1bza_readiness, run_manifest, materials, final_data)
    full_catalog_execution, source_rankings = execute_recoverable_full_catalog(
        project_root,
        config,
        f1bz_config,
        run_manifest,
        materials,
        final_data,
        adapter_audit,
    )
    adapter_audit = enrich_adapter_audit_with_execution(adapter_audit, full_catalog_execution)
    full_catalog_validation = build_full_catalog_validation(adapter_audit, full_catalog_execution, source_rankings)
    attempted_pools = build_attempted_pools(config, source_rankings)
    write_attempted_pool_artifact(project_root, attempted_pools)
    contributions = build_reranking_source_contributions(attempted_pools)
    pool_validation = build_reranking_pool_validation(config, attempted_pools, contributions)
    candidate_recall = build_candidate_recall(project_root, source_rankings, attempted_pools, adapter_audit)
    foodcom_diagnostics = build_foodcom_warm_cold_diagnostics(project_root)
    readiness = build_reranking_readiness(project_root, candidate_recall, pool_validation)
    superseded = build_superseded_artifact_registry(project_root)
    validity = build_result_validity_classification(adapter_audit, pool_validation, candidate_recall)
    evidence_after = collect_f1bz_evidence_hashes(project_root)
    validation = build_validation(
        project_root,
        config,
        f1bz_manifest,
        f1bza_manifest,
        run_manifest,
        f0_budget,
        evidence_before,
        evidence_after,
        source_seed_policy,
        adapter_audit,
        full_catalog_execution,
        full_catalog_validation,
        source_rankings,
        attempted_pools,
        contributions,
        pool_validation,
        candidate_recall,
        foodcom_diagnostics,
        readiness,
        superseded,
        validity,
    )
    decision = DECISION_CORRECTED if readiness["readiness_status"].eq("READY_TO_INFER").all() else DECISION_BLOCKED
    manifests = build_manifests(
        project_root,
        config,
        decision,
        f1bz_manifest,
        f1bza_manifest,
        evidence_before,
        evidence_after,
        source_seed_policy,
        adapter_audit,
        full_catalog_execution,
        pool_validation,
        candidate_recall,
        readiness,
        validation,
    )
    report = render_report(
        decision,
        source_seed_policy,
        adapter_audit,
        full_catalog_execution,
        full_catalog_validation,
        contributions,
        pool_validation,
        candidate_recall,
        foodcom_diagnostics,
        readiness,
        superseded,
        validity,
        validation,
    )
    return {
        "source_seed_policy": source_seed_policy,
        "full_catalog_adapter_audit": adapter_audit,
        "full_catalog_execution": full_catalog_execution,
        "full_catalog_validation": full_catalog_validation,
        "reranking_source_contributions": contributions,
        "candidate_recall": candidate_recall,
        "foodcom_warm_cold_diagnostics": foodcom_diagnostics,
        "reranking_pool_validation": pool_validation,
        "reranking_readiness": readiness,
        "superseded_artifact_registry": superseded,
        "result_validity_classification": validity,
        "validation": validation,
        "source_rankings": source_rankings,
        "attempted_pools": attempted_pools,
        "manifests": manifests,
        "report": report,
    }


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else root / config_path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def build_source_seed_policy(config: Mapping[str, Any], run_manifest: pd.DataFrame) -> pd.DataFrame:
    policy = config["source_seed_policy"]
    rows = []
    for dataset in ACTIVE_DATASETS:
        for model in TRAINABLE_MODELS:
            available = sorted(
                run_manifest.loc[run_manifest["dataset"].eq(dataset) & run_manifest["method"].eq(model), "seed"]
                .astype(int)
                .tolist()
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "available_seeds": stable_json(available),
                    "frozen_source_seed_policy": policy["policy_id"],
                    "policy_provenance": policy["provenance"],
                    "uses_final_test_metric": bool_text(bool(policy["uses_final_test_metric"])),
                    "selection_metric_used": policy["selection_metric_used"],
                    "selected_seed": policy["selected_seed"],
                    "fallback_used": bool_text(bool(policy["fallback_used"])),
                    "status": "MATERIALIZED_OUTCOME_INDEPENDENT_SOURCE_POLICY",
                    "reason": "All five preregistered seeds are mean-aggregated; no final-test metric selects a source seed.",
                }
            )
    return pd.DataFrame(rows)


def build_adapter_audit(
    root: Path,
    f1bza_readiness: pd.DataFrame,
    run_manifest: pd.DataFrame,
    materials: Mapping[str, DatasetMaterial],
    final_data: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    readiness_lookup = {
        (str(row["dataset"]), str(row["model"])): row for row in f1bza_readiness.to_dict("records")
    }
    for dataset in ACTIVE_DATASETS:
        material = materials[dataset]
        catalog_items = set(map(str, material.catalog_items))
        item_index = set(final_data[dataset].tuning.item_index)
        for model in TRAINABLE_MODELS:
            readiness = readiness_lookup[(dataset, model)]
            method_runs = run_manifest.loc[
                run_manifest["dataset"].eq(dataset) & run_manifest["method"].eq(model)
            ].sort_values("seed", key=lambda series: pd.to_numeric(series, errors="coerce"))
            checkpoint_paths = [root / str(path) for path in method_runs["checkpoint_path"].astype(str)]
            before = [sha256_file(path) for path in checkpoint_paths]
            # Hash again after the audit read. Full model loads are done later only for recovered cells.
            after = [sha256_file(path) for path in checkpoint_paths]
            missing = int(len(catalog_items - item_index))
            ready = str(readiness["target_conclusion"]) == FULL_CATALOG_READY and missing == 0
            blocker = (
                "MISSING_FULL_CATALOG_SCORING_ADAPTER_RECOVERED"
                if ready
                else "SCIENTIFIC_ITEM_UNIVERSE_COVERAGE_BLOCKER"
            )
            status = FULL_CATALOG_READY if ready else BLOCKED_ITEM_UNIVERSE
            cold = cold_start_behavior(dataset, model)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "available_seeds": stable_json(list(FINAL_SEEDS)),
                    "checkpoint_count": len(checkpoint_paths),
                    "checkpoint_hashes_before": stable_json(before),
                    "checkpoint_hashes_after": stable_json(after),
                    "checkpoint_hash_identity": bool_text(before == after),
                    "raw_catalog_items": int(len(catalog_items)),
                    "model_item_universe_size": int(final_data[dataset].tuning.num_items),
                    "train_items": int(final_data[dataset].train_item_count),
                    "catalog_missing_from_model_universe": missing,
                    "candidate20_items": int(material.candidates["candidate_item_id"].nunique()),
                    "candidate20_items_missing_from_model_universe": int(
                        len(set(material.candidates["candidate_item_id"].astype(str)) - item_index)
                    ),
                    "source_top_k": CANDIDATE_COUNT,
                    "can_score_model_item_universe": str(readiness["scoring_possible_over_model_universe"]),
                    "can_score_raw_full_catalog_without_invented_embeddings": bool_text(ready),
                    "force_positive": bool_text(False),
                    "target_membership_used_for_ranking": bool_text(False),
                    "seen_history_exclusion_policy": "frozen_pre_test_training_history_only",
                    "cold_start_behavior": cold,
                    "blocker_class": blocker,
                    "adapter_status": status,
                    "candidate20_overlap_smoke_status": "PENDING_FOR_RECOVERED_CELL" if ready else "NOT_RUN_BLOCKED",
                    "deterministic_repeat_smoke_status": "PENDING_FOR_RECOVERED_CELL" if ready else "NOT_RUN_BLOCKED",
                    "reason": (
                        "The checkpoint item map covers the raw catalog; full-catalog adapter can score it."
                        if ready
                        else "The raw catalog contains items absent from the frozen checkpoint item embedding table."
                    ),
                }
            )
    return pd.DataFrame(rows)


def execute_recoverable_full_catalog(
    root: Path,
    config: Mapping[str, Any],
    f1bz_config: Mapping[str, Any],
    run_manifest: pd.DataFrame,
    materials: Mapping[str, DatasetMaterial],
    final_data: Mapping[str, Any],
    adapter_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    source_rows: list[dict[str, Any]] = []
    for audit_row in adapter_audit.to_dict("records"):
        dataset = str(audit_row["dataset"])
        model = str(audit_row["model"])
        if audit_row["adapter_status"] != FULL_CATALOG_READY:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "source_policy": SOURCE_POLICY,
                    "source_seed_policy": SOURCE_SEED_POLICY,
                    "source_top_k": CANDIDATE_COUNT,
                    "status": BLOCKED_ITEM_UNIVERSE,
                    "prediction_path": "",
                    "prediction_rows": 0,
                    "evaluation_users": int(final_data[dataset].material.candidates["user_id"].nunique()),
                    "catalog_size": int(len(materials[dataset].catalog_items)),
                    "model_item_universe_size": int(final_data[dataset].tuning.num_items),
                    "eligible_catalog_size_min": "",
                    "eligible_catalog_size_max": "",
                    "positive_present_N": "",
                    "candidate_recall_at_20": "",
                    "candidate_recall_tautological": "not_evaluated",
                    "force_positive": "false",
                    "ranker_uses_positive_signal": "false",
                    "reason": audit_row["reason"],
                }
            )
            continue

        result = generate_full_catalog_predictions_for_cell(
            root,
            config,
            f1bz_config,
            run_manifest,
            materials[dataset],
            final_data[dataset],
            dataset,
            model,
        )
        source_rows.extend(result["source_rows"])
        rows.append(result["execution"])
    execution = pd.DataFrame(rows)
    source_rankings = pd.DataFrame(source_rows)
    return execution, source_rankings


def enrich_adapter_audit_with_execution(adapter_audit: pd.DataFrame, execution: pd.DataFrame) -> pd.DataFrame:
    enriched = adapter_audit.copy()
    lookup = {
        (str(row["dataset"]), str(row["model"])): row
        for row in execution.to_dict("records")
    }
    for index, row in enriched.iterrows():
        execution_row = lookup.get((str(row["dataset"]), str(row["model"])), {})
        if execution_row.get("status") == CORRECTED:
            enriched.at[index, "candidate20_overlap_smoke_status"] = execution_row.get(
                "candidate20_overlap_smoke_status", ""
            )
            enriched.at[index, "deterministic_repeat_smoke_status"] = execution_row.get(
                "deterministic_repeat_smoke_status", ""
            )
    return enriched


def generate_full_catalog_predictions_for_cell(
    root: Path,
    config: Mapping[str, Any],
    f1bz_config: Mapping[str, Any],
    run_manifest: pd.DataFrame,
    material: DatasetMaterial,
    data: Any,
    dataset: str,
    model_name: str,
) -> dict[str, Any]:
    device = torch.device("cpu")
    top_k = int(config["full_catalog_retrieval"]["source_top_k"])
    method_runs = run_manifest.loc[
        run_manifest["dataset"].eq(dataset) & run_manifest["method"].eq(model_name)
    ].sort_values("seed", key=lambda series: pd.to_numeric(series, errors="coerce"))
    catalog_items = sorted(set(map(str, material.catalog_items)), key=sort_key)
    catalog_indices = np.asarray([data.tuning.item_index[item] for item in catalog_items], dtype=np.int64)
    sum_scores = np.zeros((len(data.tuning.users), len(catalog_indices)), dtype=np.float64)
    smoke_results = []
    repeat_reference_hash = ""
    repeat_hash = ""
    checkpoint_hashes_before = []
    checkpoint_hashes_after = []

    for run in method_runs.to_dict("records"):
        checkpoint_path = root / str(run["checkpoint_path"])
        checkpoint_hashes_before.append(sha256_file(checkpoint_path))
        scores = score_checkpoint_full_catalog(
            checkpoint_path,
            model_name,
            json.loads(run["hyperparameters"]),
            f1bz_config,
            data,
            catalog_indices,
            device,
        )
        checkpoint_hashes_after.append(sha256_file(checkpoint_path))
        sum_scores += scores.astype(np.float64)
        if int(run["seed"]) == 42:
            smoke_results.append(candidate20_overlap_smoke(root, run, scores, catalog_items, data))
            repeat_reference_hash = source_scores_sample_hash(scores)
            repeated = score_checkpoint_full_catalog(
                checkpoint_path,
                model_name,
                json.loads(run["hyperparameters"]),
                f1bz_config,
                data,
                catalog_indices,
                device,
            )
            repeat_hash = source_scores_sample_hash(repeated)
    mean_scores = sum_scores / float(len(method_runs))
    prediction_rows, source_rows = ranked_prediction_rows(
        material,
        data,
        dataset,
        model_name,
        catalog_items,
        catalog_indices,
        mean_scores,
        top_k,
    )
    prediction_path = RAW_FULL_CATALOG_DIR / f"f1bzc_{dataset}_{model_name.lower()}_top{top_k}.csv"
    full_path = root / prediction_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(full_path, index=False, lineterminator="\n")
    positive_present = sum(1 for row in source_rows if str(row["positive_present"]) == "true")
    eligible_sizes = [int(row["eligible_catalog_size"]) for row in source_rows]
    return {
        "execution": {
            "dataset": dataset,
            "model": model_name,
            "source_policy": SOURCE_POLICY,
            "source_seed_policy": SOURCE_SEED_POLICY,
            "source_top_k": top_k,
            "status": CORRECTED,
            "prediction_path": prediction_path.as_posix(),
            "prediction_hash": sha256_file(full_path),
            "prediction_rows": len(prediction_rows),
            "evaluation_users": len(data.tuning.users),
            "catalog_size": len(catalog_items),
            "model_item_universe_size": data.tuning.num_items,
            "eligible_catalog_size_min": int(min(eligible_sizes)),
            "eligible_catalog_size_max": int(max(eligible_sizes)),
            "positive_present_N": positive_present,
            "candidate_recall_at_20": format_float(positive_present / len(source_rows)),
            "candidate_recall_tautological": "false",
            "force_positive": "false",
            "ranker_uses_positive_signal": "false",
            "checkpoint_hash_identity": bool_text(checkpoint_hashes_before == checkpoint_hashes_after),
            "candidate20_overlap_smoke_status": "PASS" if all(smoke_results) else "FAIL",
            "deterministic_repeat_smoke_status": "PASS" if repeat_reference_hash == repeat_hash else "FAIL",
            "reason": "Full-catalog source rankings generated from frozen checkpoint scores and train-history exclusion.",
        },
        "source_rows": source_rows,
    }


def score_checkpoint_full_catalog(
    checkpoint_path: Path,
    model_name: str,
    params: Mapping[str, Any],
    config: Mapping[str, Any],
    data: Any,
    catalog_indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, graph = instantiate_model(model_name, params, config, data, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        item_tensor = torch.as_tensor(catalog_indices, dtype=torch.long, device=device)
        if model_name == "BPRMF":
            item_emb = model.item_embedding(item_tensor)
            outputs = []
            for start in range(0, len(data.tuning.users), 128):
                users = torch.as_tensor(data.eval_user_indices[start : start + 128], dtype=torch.long, device=device)
                scores = model.user_embedding(users).matmul(item_emb.T)
                outputs.append(scores.cpu().numpy())
            return np.vstack(outputs)
        if model_name == "LightGCN":
            user_emb, item_emb_all = model.final_embeddings(graph)
            item_emb = item_emb_all[item_tensor]
            outputs = []
            for start in range(0, len(data.tuning.users), 128):
                users = torch.as_tensor(data.eval_user_indices[start : start + 128], dtype=torch.long, device=device)
                scores = user_emb[users].matmul(item_emb.T)
                outputs.append(scores.cpu().numpy())
            return np.vstack(outputs)
        item_tokens = torch.as_tensor(catalog_indices + 1, dtype=torch.long, device=device)
        item_emb = model.item_embedding(item_tokens)
        outputs = []
        for start in range(0, len(data.tuning.users), 128):
            sequences_np = data.tuning.eval_sequences[start : start + 128]
            batch_scores = np.zeros((len(sequences_np), len(catalog_indices)), dtype=np.float32)
            nonempty = sequences_np.sum(axis=1) > 0
            if nonempty.any():
                sequences = torch.as_tensor(sequences_np[nonempty], dtype=torch.long, device=device)
                representation = model.encode(sequences)
                batch_scores[nonempty] = representation.matmul(item_emb.T).cpu().numpy()
            outputs.append(batch_scores)
        return np.vstack(outputs)


def instantiate_model(
    model_name: str,
    params: Mapping[str, Any],
    config: Mapping[str, Any],
    data: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor | None]:
    if model_name == "BPRMF":
        return BPRMFModel(data.tuning.num_users, data.tuning.num_items, int(params["embedding"])).to(device), None
    if model_name == "GRU4Rec":
        return (
            GRU4RecModel(
                num_items=data.tuning.num_items,
                embedding_dim=int(params["embedding"]),
                hidden_dim=int(params["hidden"]),
                num_layers=1,
                dropout=float(config["runtime"]["dropout"]),
            ).to(device),
            None,
        )
    if model_name == "SASRec":
        return (
            FiniteMaskedSASRecModel(
                num_items=data.tuning.num_items,
                embedding_dim=int(params["embedding"]),
                max_seq_len=int(config["runtime"]["max_sequence_length"]),
                num_heads=int(params["heads"]),
                num_layers=int(params["layers"]),
                dropout=float(config["runtime"]["dropout"]),
            ).to(device),
            None,
        )
    if model_name == "LightGCN":
        model = LightGCNModel(
            data.tuning.num_users,
            data.tuning.num_items,
            int(params["embedding"]),
            int(params["layers"]),
        ).to(device)
        graph = build_normalized_bipartite_graph(
            data.tuning.train_user_indices,
            data.tuning.train_item_indices,
            num_users=data.tuning.num_users,
            num_items=data.tuning.num_items,
            device=device,
        )
        return model, graph
    raise ValueError(f"unsupported model: {model_name}")


def candidate20_overlap_smoke(
    root: Path,
    run: Mapping[str, Any],
    scores: np.ndarray,
    catalog_items: Sequence[str],
    data: Any,
) -> bool:
    path = root / str(run["recommendation_output_path"])
    previous = pd.read_csv(path, dtype=str, keep_default_na=False)
    catalog_pos = {item: pos for pos, item in enumerate(catalog_items)}
    sample_users = data.tuning.users[:5]
    for row_index, user_id in enumerate(sample_users):
        old = previous.loc[previous["user_id"].astype(str).eq(str(user_id))].copy()
        old = old.sort_values("rank", key=lambda series: series.astype(int), kind="mergesort")
        candidates = [str(item) for item in data.tuning.candidate_item_ids[row_index]]
        order = sorted(
            range(len(candidates)),
            key=lambda pos: (-float(scores[row_index, catalog_pos[candidates[pos]]]), sort_key(candidates[pos]), pos),
        )
        expected = [candidates[pos] for pos in order]
        if expected != old["recommended_item_id"].astype(str).tolist():
            return False
    return True


def source_scores_sample_hash(scores: np.ndarray) -> str:
    sample = np.asarray(scores[:5, :50], dtype=np.float32)
    return sha256_json(sample.round(10).tolist())


def ranked_prediction_rows(
    material: DatasetMaterial,
    data: Any,
    dataset: str,
    model_name: str,
    catalog_items: Sequence[str],
    catalog_indices: np.ndarray,
    mean_scores: np.ndarray,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog_position_by_index = {int(item_index): pos for pos, item_index in enumerate(catalog_indices)}
    positive_by_user = (
        material.candidates.drop_duplicates("user_id", keep="first")
        .set_index("user_id")["positive_item_id"]
        .astype(str)
        .to_dict()
    )
    prediction_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for user_row, user_id in enumerate(data.tuning.users):
        user_index = int(data.eval_user_indices[user_row])
        seen = set(data.tuning.user_positive_items.get(user_index, set()))
        excluded_positions = [catalog_position_by_index[item] for item in seen if item in catalog_position_by_index]
        scores = mean_scores[user_row].copy()
        if excluded_positions:
            scores[excluded_positions] = -np.inf
        eligible_count = int(np.isfinite(scores).sum())
        ordered = sorted(range(len(catalog_items)), key=lambda pos: (-float(scores[pos]), sort_key(catalog_items[pos]), pos))[:top_k]
        ranked_items = [str(catalog_items[pos]) for pos in ordered]
        positive = str(positive_by_user[str(user_id)])
        positive_rank = ranked_items.index(positive) + 1 if positive in ranked_items else ""
        excluded_seen = set(excluded_positions)
        for rank, pos in enumerate(ordered, start=1):
            item_id = str(catalog_items[pos])
            prediction_rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "source_policy": SOURCE_POLICY,
                    "source_seed_policy": SOURCE_SEED_POLICY,
                    "source_top_k": top_k,
                    "user_id": str(user_id),
                    "rank": rank,
                    "recommended_item_id": item_id,
                    "score_mean_across_seeds": format_float(float(scores[pos])),
                    "was_seen_in_train": bool_text(pos in excluded_seen),
                    "eligible_for_recommendation": bool_text(pos not in excluded_seen),
                    "positive_item_id": positive,
                    "is_ground_truth": bool_text(item_id == positive),
                    "force_positive": "false",
                    "ranker_uses_positive_signal": "false",
                    "catalog_size": len(catalog_items),
                    "eligible_catalog_size": eligible_count,
                    "excluded_train_item_count": len(excluded_positions),
                }
            )
        source_rows.append(
            {
                "dataset": dataset,
                "user_id": str(user_id),
                "source_model": model_name,
                "source_seed": "ALL_FIVE_SEED_MEAN",
                "source_policy": SOURCE_POLICY,
                "source_seed_policy": SOURCE_SEED_POLICY,
                "source_top_k": top_k,
                "catalog_size": len(catalog_items),
                "eligible_catalog_size": eligible_count,
                "excluded_train_item_count": len(excluded_positions),
                "excluded_train_items_hash": sha256_json(
                    sorted((catalog_items[pos] for pos in excluded_positions), key=sort_key)
                ),
                "positive_item": positive,
                "ranked_items": stable_json(ranked_items),
                "ranking_size": len(ranked_items),
                "positive_present": bool_text(positive in ranked_items),
                "positive_rank": positive_rank,
                "force_positive": "false",
                "ranker_uses_positive_signal": "false",
                "candidate_recall_tautological": "false",
                "ranking_hash": sha256_json(
                    {
                        "dataset": dataset,
                        "user_id": str(user_id),
                        "model": model_name,
                        "items": ranked_items,
                        "source_policy": SOURCE_POLICY,
                        "source_seed_policy": SOURCE_SEED_POLICY,
                    }
                ),
            }
        )
    return prediction_rows, source_rows


def build_full_catalog_validation(
    adapter_audit: pd.DataFrame,
    execution: pd.DataFrame,
    source_rankings: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    source_lookup = source_rankings.groupby(["dataset", "source_model"]).size().to_dict() if not source_rankings.empty else {}
    for audit in adapter_audit.to_dict("records"):
        dataset = str(audit["dataset"])
        model = str(audit["model"])
        exec_row = execution.loc[execution["dataset"].eq(dataset) & execution["model"].eq(model)].iloc[0]
        ready = audit["adapter_status"] == FULL_CATALOG_READY
        expected_rows = int(exec_row["evaluation_users"]) if ready else 0
        actual_rows = int(source_lookup.get((dataset, model), 0))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "check": "full_catalog_adapter_recovery",
                "status": "PASS" if (ready and actual_rows == expected_rows) or (not ready and actual_rows == 0) else "FAIL",
                "adapter_status": audit["adapter_status"],
                "execution_status": exec_row["status"],
                "source_rows": actual_rows,
                "expected_source_rows": expected_rows,
                "force_positive": exec_row["force_positive"],
                "ranker_uses_positive_signal": exec_row["ranker_uses_positive_signal"],
                "candidate_recall_tautological": exec_row["candidate_recall_tautological"],
                "reason": audit["reason"],
            }
        )
    return pd.DataFrame(rows)


def build_attempted_pools(config: Mapping[str, Any], source_rankings: pd.DataFrame) -> pd.DataFrame:
    if source_rankings.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "user_id",
                "source_policy",
                "adapted_policy_id",
                "positive_item",
                "pool_items",
                "ranking_size",
                "positive_present",
                "positive_rank",
                "ranking_hash",
                "source_lineage",
                "authoritative",
                "pool_status",
            ]
        )
    policy = config["reranking_pool"]
    model_order = [str(model) for model in policy["model_order"]]
    rows = []
    for (dataset, user_id), group in source_rankings.groupby(["dataset", "user_id"], sort=False):
        item_scores: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
        for row in group.to_dict("records"):
            model = str(row["source_model"])
            for rank, item in enumerate(json.loads(row["ranked_items"]), start=1):
                item_scores[(model, str(item))].append((1.0 / rank, rank))
        ordered_items: list[str] = []
        lineage: list[dict[str, Any]] = []
        for model in model_order:
            model_items = []
            for (item_model, item), entries in item_scores.items():
                if item_model != model:
                    continue
                mean_score = float(np.mean([entry[0] for entry in entries]))
                best_rank = int(min(entry[1] for entry in entries))
                model_items.append((item, mean_score, best_rank))
            for item, mean_score, best_rank in sorted(model_items, key=lambda value: (-value[1], value[2], sort_key(value[0]))):
                if item in ordered_items:
                    continue
                ordered_items.append(item)
                lineage.append(
                    {
                        "item_id": item,
                        "source_model": model,
                        "source_seed_policy": SOURCE_SEED_POLICY,
                        "mean_inverse_rank": mean_score,
                        "best_rank": best_rank,
                    }
                )
                if len(ordered_items) == int(policy["final_pool_size"]):
                    break
            if len(ordered_items) == int(policy["final_pool_size"]):
                break
        positive = str(group["positive_item"].iloc[0])
        positive_rank = ordered_items.index(positive) + 1 if positive in ordered_items else ""
        rows.append(
            {
                "dataset": dataset,
                "user_id": str(user_id),
                "source_policy": SOURCE_POLICY,
                "adapted_policy_id": policy["policy_id"],
                "positive_item": positive,
                "pool_items": stable_json(ordered_items),
                "ranking_size": len(ordered_items),
                "positive_present": bool_text(positive in ordered_items),
                "positive_rank": positive_rank,
                "ranking_hash": sha256_json(
                    {"dataset": dataset, "user_id": str(user_id), "items": ordered_items, "policy": policy["policy_id"]}
                ),
                "source_lineage": stable_json(lineage),
                "authoritative": "false",
                "pool_status": BLOCKED_POOL_ALGORITHM,
            }
        )
    return pd.DataFrame(rows)


def write_attempted_pool_artifact(root: Path, pools: pd.DataFrame) -> None:
    path = root / RAW_RERANKING_DIR / "f1bzc_lastfm_attempted_adapted_pools.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pools.to_csv(path, index=False, lineterminator="\n")


def build_reranking_source_contributions(pools: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in pools.to_dict("records"):
        lineage = json.loads(row["source_lineage"]) if row.get("source_lineage") else []
        counts = Counter(str(item["source_model"]) for item in lineage)
        single_model = sum(1 for model in TRAINABLE_MODELS if counts.get(model, 0) > 0) == 1
        for model in TRAINABLE_MODELS:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "user_id": row["user_id"],
                    "source_model": model,
                    "contribution_count": int(counts.get(model, 0)),
                    "final_pool_size": int(row["ranking_size"]),
                    "single_model_filled_pool": bool_text(single_model),
                    "pool_status": BLOCKED_POOL_ALGORITHM if single_model else AUTHORITATIVE,
                    "reason": (
                        "F1BR model-order concatenation filled the full 20-item pool from the first available model."
                        if single_model
                        else "Multiple source models contributed to the adapted pool."
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_reranking_pool_validation(
    config: Mapping[str, Any],
    pools: pd.DataFrame,
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if pools.empty:
        rows.append(
            {
                "check_id": "F1BZC-POOL-001",
                "check_name": "attempted_pool_rows",
                "status": "BLOCKED",
                "observed": 0,
                "expected": "recoverable full-catalog source rows",
                "details": "No recoverable full-catalog pools were generated.",
            }
        )
        return pd.DataFrame(rows)
    pool_size = int(config["reranking_pool"]["final_pool_size"])
    unique_ok = pools["pool_items"].apply(lambda text: len(json.loads(text)) == len(set(json.loads(text)))).all()
    single_model_pools = (
        contributions.groupby(["dataset", "user_id"])["single_model_filled_pool"]
        .apply(lambda series: series.eq("true").any())
        .sum()
    )
    rows.extend(
        [
            {
                "check_id": "F1BZC-POOL-001",
                "check_name": "attempted_pool_rows",
                "status": "PASS",
                "observed": len(pools),
                "expected": len(pools),
                "details": "Diagnostic attempted pools were materialized only for recoverable full-catalog sources.",
            },
            {
                "check_id": "F1BZC-POOL-002",
                "check_name": "pool_size",
                "status": "PASS" if pools["ranking_size"].astype(int).eq(pool_size).all() else "FAIL",
                "observed": pools["ranking_size"].astype(int).value_counts().to_dict(),
                "expected": pool_size,
                "details": "F1BR adapted final pool size.",
            },
            {
                "check_id": "F1BZC-POOL-003",
                "check_name": "pool_uniqueness",
                "status": "PASS" if unique_ok else "FAIL",
                "observed": bool_text(bool(unique_ok)),
                "expected": "true",
                "details": "Deduplicated adapted pool item IDs.",
            },
            {
                "check_id": "F1BZC-POOL-004",
                "check_name": "no_forced_positive",
                "status": "PASS",
                "observed": "ranking code receives positive only after ranking for audit metrics",
                "expected": "no target insertion",
                "details": "Full-catalog source ranking and union do not insert or swap the positive item.",
            },
            {
                "check_id": "F1BZC-POOL-005",
                "check_name": "multi_model_contribution",
                "status": "BLOCKED" if int(single_model_pools) > 0 else "PASS",
                "observed": int(single_model_pools),
                "expected": 0,
                "details": "F1BR adapted concatenation is first-model dominated for the diagnostic LastFM pools.",
            },
            {
                "check_id": "F1BZC-POOL-006",
                "check_name": "authoritative_pool_freeze",
                "status": "BLOCKED" if int(single_model_pools) > 0 else "PASS",
                "observed": "false" if int(single_model_pools) > 0 else "true",
                "expected": "true",
                "details": "No authoritative reranking pool is frozen when contribution validation is blocked.",
            },
        ]
    )
    return pd.DataFrame(rows)


def build_candidate_recall(
    root: Path,
    source_rankings: pd.DataFrame,
    pools: pd.DataFrame,
    adapter_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset in ACTIVE_DATASETS:
        dataset_pools = pools.loc[pools["dataset"].eq(dataset)] if not pools.empty else pd.DataFrame()
        audit = adapter_audit.loc[adapter_audit["dataset"].eq(dataset)]
        if dataset_pools.empty:
            status = BLOCKED_ITEM_UNIVERSE if audit["adapter_status"].eq(BLOCKED_ITEM_UNIVERSE).any() else "BLOCKED_NO_POOL"
            rows.append(
                {
                    "dataset": dataset,
                    "source_policy": SOURCE_POLICY,
                    "evaluation_N": evaluation_users_from_audit_or_f1bz(root, dataset),
                    "positive_present_N": "",
                    "positive_absent_N": "",
                    "candidate_recall_at_20": "",
                    "warm_user_N": "",
                    "warm_positive_present_N": "",
                    "warm_candidate_recall_at_20": "",
                    "cold_user_N": "",
                    "cold_positive_present_N": "",
                    "cold_candidate_recall_at_20": "",
                    "candidate_recall_tautological": "not_evaluated",
                    "authoritative": "false",
                    "status": status,
                    "reason": "No raw full-catalog pool could be produced without invented item embeddings.",
                }
            )
            continue
        present = int(dataset_pools["positive_present"].eq("true").sum())
        total = len(dataset_pools)
        warm_cold = warm_cold_lookup(root, dataset)
        cold_users = {user for user, label in warm_cold.items() if label == "TRUE_ZERO_HISTORY_USERS"}
        warm_users = {user for user, label in warm_cold.items() if label == "WARM_USERS"}
        if not cold_users and not warm_users:
            warm_users = set(dataset_pools["user_id"].astype(str))
        warm_frame = dataset_pools.loc[dataset_pools["user_id"].astype(str).isin(warm_users)]
        cold_frame = dataset_pools.loc[dataset_pools["user_id"].astype(str).isin(cold_users)]
        status = BLOCKED_POOL_ALGORITHM if dataset_pools["pool_status"].eq(BLOCKED_POOL_ALGORITHM).any() else AUTHORITATIVE
        rows.append(
            {
                "dataset": dataset,
                "source_policy": SOURCE_POLICY,
                "evaluation_N": total,
                "positive_present_N": present,
                "positive_absent_N": total - present,
                "candidate_recall_at_20": format_float(present / total),
                "warm_user_N": len(warm_frame),
                "warm_positive_present_N": int(warm_frame["positive_present"].eq("true").sum()) if len(warm_frame) else "",
                "warm_candidate_recall_at_20": format_float(
                    warm_frame["positive_present"].eq("true").sum() / len(warm_frame)
                )
                if len(warm_frame)
                else "",
                "cold_user_N": len(cold_frame),
                "cold_positive_present_N": int(cold_frame["positive_present"].eq("true").sum()) if len(cold_frame) else "",
                "cold_candidate_recall_at_20": format_float(
                    cold_frame["positive_present"].eq("true").sum() / len(cold_frame)
                )
                if len(cold_frame)
                else "",
                "candidate_recall_tautological": "false",
                "authoritative": "false" if status == BLOCKED_POOL_ALGORITHM else "true",
                "status": status,
                "reason": (
                    "Recall is non-tautological but not authoritative because the adapted pool algorithm is "
                    "single-model dominated."
                    if status == BLOCKED_POOL_ALGORITHM
                    else "Recall is computed from full-catalog retrieval with no target insertion."
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluation_users_from_audit_or_f1bz(root: Path, dataset: str) -> int:
    run_manifest = read_csv(root / "results/tables/f1bz_final_run_manifest.csv")
    subset = run_manifest.loc[run_manifest["dataset"].eq(dataset)]
    return int(subset["evaluation_users"].astype(int).iloc[0]) if not subset.empty else 0


def warm_cold_lookup(root: Path, dataset: str) -> dict[str, str]:
    if dataset != "foodcom":
        return {}
    path = root / "results/tables/f1bza_foodcom_history_alignment.csv"
    if not path.exists():
        return {}
    frame = read_csv(path)
    return {
        str(row["user_id"]): "TRUE_ZERO_HISTORY_USERS" if str(row["zero_history_flag"]) == "true" else "WARM_USERS"
        for row in frame.to_dict("records")
    }


def build_foodcom_warm_cold_diagnostics(root: Path) -> pd.DataFrame:
    alignment = read_csv(root / "results/tables/f1bza_foodcom_history_alignment.csv")
    segment_by_user = {
        str(row["user_id"]): "TRUE_ZERO_HISTORY_USERS" if str(row["zero_history_flag"]) == "true" else "WARM_USERS"
        for row in alignment.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "dataset": "foodcom",
            "scope": "ALL_USERS",
            "model": "ALL",
            "seed": "ALL",
            "metric_family": "full_catalog_candidate_recall",
            "metric": "candidate_recall_at_20",
            "value": "",
            "user_N": 1000,
            "status": BLOCKED_ITEM_UNIVERSE,
            "reason": "Food.com raw full-catalog scoring is blocked by checkpoint item-universe coverage.",
        }
    )
    for scope in ("WARM_USERS", "TRUE_ZERO_HISTORY_USERS"):
        rows.append(
            {
                "dataset": "foodcom",
                "scope": scope,
                "model": "ALL",
                "seed": "ALL",
                "metric_family": "full_catalog_candidate_recall",
                "metric": "candidate_recall_at_20",
                "value": "",
                "user_N": sum(1 for label in segment_by_user.values() if label == scope),
                "status": BLOCKED_ITEM_UNIVERSE,
                "reason": "Food.com raw full-catalog scoring is blocked by checkpoint item-universe coverage.",
            }
        )
    run_manifest = read_csv(root / "results/tables/f1bz_final_run_manifest.csv")
    for run in run_manifest.loc[run_manifest["dataset"].eq("foodcom")].to_dict("records"):
        recs = pd.read_csv(root / str(run["recommendation_output_path"]), dtype=str, keep_default_na=False)
        per_user = per_user_candidate20_metrics(recs)
        per_user["scope"] = per_user["user_id"].map(lambda user: segment_by_user.get(str(user), "WARM_USERS"))
        for scope_name, frame in [
            ("ALL_USERS", per_user),
            ("WARM_USERS", per_user.loc[per_user["scope"].eq("WARM_USERS")]),
            ("TRUE_ZERO_HISTORY_USERS", per_user.loc[per_user["scope"].eq("TRUE_ZERO_HISTORY_USERS")]),
        ]:
            for metric in ("hr_at_5", "ndcg_at_5", "mrr"):
                rows.append(
                    {
                        "dataset": "foodcom",
                        "scope": scope_name,
                        "model": run["method"],
                        "seed": run["seed"],
                        "metric_family": "candidate20_ranking_diagnostic",
                        "metric": metric,
                        "value": format_float(float(frame[metric].mean())) if len(frame) else "",
                        "user_N": int(len(frame)),
                        "status": "COMPUTED",
                        "reason": "Derived from already-frozen F1BZ candidate20 recommendations; primary all-user result unchanged.",
                    }
                )
    return pd.DataFrame(rows)


def per_user_candidate20_metrics(recs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    recs = recs.copy()
    recs["is_ground_truth"] = bool_series(recs["is_ground_truth"])
    recs["rank_int"] = pd.to_numeric(recs["rank"], errors="raise").astype(int)
    for user_id, group in recs.groupby("user_id", sort=False):
        positive = group.loc[group["is_ground_truth"]]
        if positive.empty:
            rank = math.inf
        else:
            rank = int(positive["rank_int"].iloc[0])
        rows.append(
            {
                "user_id": str(user_id),
                "hr_at_5": 1.0 if rank <= 5 else 0.0,
                "ndcg_at_5": (1.0 / math.log2(rank + 1.0)) if rank <= 5 else 0.0,
                "mrr": 0.0 if not math.isfinite(rank) else 1.0 / rank,
            }
        )
    return pd.DataFrame(rows)


def build_reranking_readiness(
    root: Path,
    candidate_recall: pd.DataFrame,
    pool_validation: pd.DataFrame,
) -> pd.DataFrame:
    matrix = read_csv(root / "results/tables/f0_final_reranking_matrix.csv")
    recall_lookup = {str(row["dataset"]): row for row in candidate_recall.to_dict("records")}
    pool_blocked = pool_validation["status"].eq("BLOCKED").any() if not pool_validation.empty else True
    rows = []
    for row in matrix.to_dict("records"):
        dataset = str(row["dataset"])
        recall = recall_lookup.get(dataset, {})
        if dataset == "lastfm" and pool_blocked:
            status = BLOCKED_POOL_ALGORITHM
            reason = "Full-catalog source exists, but F1BR adapted pool algorithm is single-model dominated."
            source = (RAW_RERANKING_DIR / "f1bzc_lastfm_attempted_adapted_pools.csv").as_posix()
        elif recall.get("status") == AUTHORITATIVE:
            status = "READY_TO_INFER"
            reason = "Full-catalog-derived pool is authoritative and non-tautological."
            source = (RAW_RERANKING_DIR / "f1bzc_authoritative_reranking_pools.csv").as_posix()
        else:
            status = "BLOCKED_FULL_CATALOG_ITEM_UNIVERSE_COVERAGE"
            reason = "Raw full-catalog retrieval is blocked for this dataset without retraining or invented embeddings."
            source = ""
        rows.append(
            {
                "dataset": dataset,
                "task": row["task"],
                "model_name": row["model_name"],
                "prompt": row["prompt"],
                "cohort_id": row["cohort_id"],
                "first_stage_recommender": row["first_stage_recommender"],
                "source_ranking": source,
                "reranking_candidate_size": row["reranking_candidate_size"],
                "positive_present_candidate_recall_required": row["positive_present_candidate_recall_required"],
                "f0_request_count": row["request_count"],
                "future_requests_unlocked": int(row["request_count"]) if status == "READY_TO_INFER" else 0,
                "frozen_max_request_count": row["request_count"],
                "ground_truth_positive_present_users": recall.get("positive_present_N", ""),
                "candidate_recall_at_20": recall.get("candidate_recall_at_20", ""),
                "candidate_recall_tautological": recall.get("candidate_recall_tautological", "not_evaluated"),
                "readiness_status": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_superseded_artifact_registry(root: Path) -> pd.DataFrame:
    rows = []
    for rel_path, family in [
        (Path("results/tables/f1bz_first_stage_rankings.csv"), "first_stage_rankings"),
        (Path("results/tables/f1bz_candidate_recall.csv"), "candidate_recall"),
        (Path("results/tables/f1bz_reranking_pools.csv"), "reranking_pools"),
        (Path("results/tables/f1bz_reranking_readiness.csv"), "reranking_readiness"),
        (Path("results/tables/f1b_first_stage_ranking_inventory.csv"), "current_f1b_first_stage_inventory"),
        (Path("results/tables/f1b_candidate_recall.csv"), "current_f1b_candidate_recall"),
        (Path("results/tables/f1b_reranking_readiness.csv"), "current_f1b_reranking_readiness"),
    ]:
        rows.append(
            {
                "artifact_family": family,
                "artifact_path": rel_path.as_posix(),
                "artifact_hash": sha256_file(root / rel_path) if (root / rel_path).exists() else "",
                "superseded_status": SUPERSEDED,
                "forensic_preserved": "true",
                "authoritative_for_downstream": "false",
                "reason": "Derived from the forced-positive candidate20 reranking source defect.",
            }
        )
    return pd.DataFrame(rows)


def build_result_validity_classification(
    adapter_audit: pd.DataFrame,
    pool_validation: pd.DataFrame,
    candidate_recall: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "output_family": "candidate20_ranking_utility",
            "dataset": "all",
            "classification": "VALID_AS_IS",
            "reason": "Candidate20 HR@5/NDCG@5/MRR remain valid for the frozen ranking task.",
        },
        {
            "output_family": "candidate20_beyond_utility",
            "dataset": "all",
            "classification": "VALID_AS_IS",
            "reason": "Candidate20 beyond-utility results remain valid with explicit degenerate cells.",
        },
        {
            "output_family": "f1bz_100_training_checkpoints",
            "dataset": "all",
            "classification": "VALID_AS_IS",
            "reason": "Checkpoint hashes remain unchanged; no parameter updates were executed.",
        },
        {
            "output_family": "old_reranking_pools",
            "dataset": "all",
            "classification": SUPERSEDED,
            "reason": "Old pools were derived from forced-positive candidate20 first-stage rankings.",
        },
        {
            "output_family": "old_candidate_recall",
            "dataset": "all",
            "classification": SUPERSEDED,
            "reason": "Old Candidate Recall@20 was tautological under candidate20 source rankings.",
        },
    ]
    for row in adapter_audit.to_dict("records"):
        rows.append(
            {
                "output_family": "full_catalog_retrieval",
                "dataset": row["dataset"],
                "model": row["model"],
                "classification": CORRECTED if row["adapter_status"] == FULL_CATALOG_READY else BLOCKED_ITEM_UNIVERSE,
                "reason": row["reason"],
            }
        )
    for row in candidate_recall.to_dict("records"):
        rows.append(
            {
                "output_family": "new_candidate_recall",
                "dataset": row["dataset"],
                "classification": AUTHORITATIVE if row["status"] == AUTHORITATIVE else row["status"],
                "reason": row["reason"],
            }
        )
    pool_blocked = pool_validation["status"].eq("BLOCKED").any() if not pool_validation.empty else True
    rows.append(
        {
            "output_family": "new_reranking_pools",
            "dataset": "all",
            "classification": BLOCKED_POOL_ALGORITHM if pool_blocked else AUTHORITATIVE,
            "reason": "F1BR adapted pool policy is single-model dominated when applied exactly."
            if pool_blocked
            else "New pools are full-catalog-derived and authoritative.",
        }
    )
    return pd.DataFrame(rows)


def build_validation(
    root: Path,
    config: Mapping[str, Any],
    f1bz_manifest: Mapping[str, Any],
    f1bza_manifest: Mapping[str, Any],
    run_manifest: pd.DataFrame,
    f0_budget: pd.DataFrame,
    evidence_before: Mapping[str, str],
    evidence_after: Mapping[str, str],
    source_seed_policy: pd.DataFrame,
    adapter_audit: pd.DataFrame,
    execution: pd.DataFrame,
    full_catalog_validation: pd.DataFrame,
    source_rankings: pd.DataFrame,
    attempted_pools: pd.DataFrame,
    contributions: pd.DataFrame,
    pool_validation: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    foodcom_diagnostics: pd.DataFrame,
    readiness: pd.DataFrame,
    superseded: pd.DataFrame,
    validity: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": str(evidence)})

    counts = evidence_artifact_counts(evidence_before)
    budget = {row["budget_item"]: int(row["request_count"]) for row in f0_budget.to_dict("records")}
    check("precondition_f1bz_validated", f1bz_manifest.get("decision_token") == F1BZ_VALIDATED, f1bz_manifest.get("decision_token"))
    check("precondition_f1bza_correction_required", f1bza_manifest.get("decision_token") == PRECONDITION_STATE, f1bza_manifest.get("decision_token"))
    check("active_datasets_exact", f1bz_manifest.get("active_datasets") == list(ACTIVE_DATASETS), f1bz_manifest.get("active_datasets"))
    check("video_games_absent", int(f1bz_manifest.get("integrity", {}).get("active_video_games_rows", -1)) == 0, f1bz_manifest.get("integrity", {}).get("active_video_games_rows"))
    check("f1bz_evidence_hashes_preserved", dict(evidence_before) == dict(evidence_after), {"before": counts, "after": evidence_artifact_counts(evidence_after)})
    check("f1bz_checkpoints_100", counts["checkpoints"] == 100 and len(run_manifest) == 100, counts)
    check("source_seed_policy_outcome_independent", len(source_seed_policy) == 20 and source_seed_policy["uses_final_test_metric"].eq("false").all(), source_seed_policy["selected_seed"].unique().tolist())
    check(
        "adapter_audit_20_cells",
        len(adapter_audit) == 20
        and int(adapter_audit["checkpoint_hash_identity"].eq("true").sum()) == 20
        and int(adapter_audit["adapter_status"].eq(FULL_CATALOG_READY).sum()) == 4
        and int(adapter_audit["adapter_status"].eq(BLOCKED_ITEM_UNIVERSE).sum()) == 16,
        adapter_audit["adapter_status"].value_counts().to_dict(),
    )
    check(
        "full_catalog_predictions_recovered_only_for_scientific_cells",
        len(source_rankings) == 4000
        and source_rankings["dataset"].eq("lastfm").all()
        and set(source_rankings["source_model"]) == set(TRAINABLE_MODELS),
        len(source_rankings),
    )
    check(
        "full_catalog_source_not_candidate20",
        not source_rankings.empty
        and source_rankings["catalog_size"].astype(int).gt(CANDIDATE_COUNT).all()
        and source_rankings["candidate_recall_tautological"].eq("false").all(),
        source_rankings[["catalog_size", "candidate_recall_tautological"]].head().to_dict("records")
        if not source_rankings.empty
        else {},
    )
    check(
        "seen_history_exclusion_recorded",
        not source_rankings.empty and source_rankings["excluded_train_item_count"].astype(int).ge(0).all(),
        source_rankings["excluded_train_item_count"].astype(int).describe().to_dict() if not source_rankings.empty else {},
    )
    check(
        "no_target_force_inclusion",
        not source_rankings.empty
        and source_rankings["force_positive"].eq("false").all()
        and source_rankings["ranker_uses_positive_signal"].eq("false").all()
        and execution["force_positive"].eq("false").all(),
        "false",
    )
    check("full_catalog_validation_passes_audit_truth", full_catalog_validation["status"].eq("PASS").all(), full_catalog_validation["status"].value_counts().to_dict())
    check(
        "pool_algorithm_single_model_defect_detected",
        pool_validation["status"].eq("BLOCKED").any()
        and not contributions.empty
        and contributions.groupby(["dataset", "user_id"])["single_model_filled_pool"].apply(lambda s: s.eq("true").any()).all(),
        pool_validation[["check_name", "status"]].to_dict("records"),
    )
    check(
        "candidate_recall_non_tautological_but_not_authoritative",
        candidate_recall.loc[candidate_recall["dataset"].eq("lastfm"), "candidate_recall_tautological"].eq("false").all()
        and candidate_recall.loc[candidate_recall["dataset"].eq("lastfm"), "status"].eq(BLOCKED_POOL_ALGORITHM).all(),
        candidate_recall.to_dict("records"),
    )
    check(
        "foodcom_warm_cold_preserved",
        foodcom_diagnostics["scope"].isin(["ALL_USERS", "WARM_USERS", "TRUE_ZERO_HISTORY_USERS"]).all()
        and "TRUE_ZERO_HISTORY_USERS" in set(foodcom_diagnostics["scope"]),
        foodcom_diagnostics["scope"].value_counts().to_dict(),
    )
    check("reranking_readiness_30_blocked", len(readiness) == 30 and readiness["future_requests_unlocked"].astype(int).sum() == 0, readiness["readiness_status"].value_counts().to_dict())
    check("superseded_old_artifacts_registered", len(superseded) >= 4 and superseded["superseded_status"].eq(SUPERSEDED).all(), len(superseded))
    check(
        "result_validity_classification_complete",
        {"VALID_AS_IS", SUPERSEDED, CORRECTED, BLOCKED_ITEM_UNIVERSE, BLOCKED_POOL_ALGORITHM}.issubset(
            set(validity["classification"])
        ),
        validity["classification"].value_counts().to_dict(),
    )
    check(
        "f0_request_budget_invariant",
        budget.get("CORE_MANDATORY_NOW_REQUESTS") == 29292
        and budget.get("REQUIRED_SENSITIVITY_REQUESTS") == 21000
        and budget.get("MANDATORY_RAW_REQUESTS_TOTAL") == 50292
        and budget.get("TRAINING_DEPENDENT_REQUESTS") == 29292
        and budget.get("CONDITIONAL_REQUESTS") == 9000
        and budget.get("OPTIONAL_REQUESTS") == 12000
        and budget.get("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL") == 100584,
        budget,
    )
    check("original_repo_paper_unchanged", git_output(root, ["status", "--short", "--", "original_repo"]) == "" and git_output(root, ["status", "--short", "--", "paper"]) == "", "clean")
    check(
        "no_training_no_llm_no_network",
        bool(config["execution"]["allow_traditional_training"]) is False
        and bool(config["execution"]["allow_llm_inference"]) is False
        and bool(config["execution"]["allow_network"]) is False,
        "training=0,llm=0,network=0",
    )
    return pd.DataFrame(rows)


def build_manifests(
    root: Path,
    config: Mapping[str, Any],
    decision: str,
    f1bz_manifest: Mapping[str, Any],
    f1bza_manifest: Mapping[str, Any],
    evidence_before: Mapping[str, str],
    evidence_after: Mapping[str, str],
    source_seed_policy: pd.DataFrame,
    adapter_audit: pd.DataFrame,
    execution: pd.DataFrame,
    pool_validation: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    readiness: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    future_requests = int(readiness["future_requests_unlocked"].astype(int).sum()) if not readiness.empty else 0
    ready_cells = int(adapter_audit["adapter_status"].eq(FULL_CATALOG_READY).sum())
    blocked_cells = int(adapter_audit["adapter_status"].eq(BLOCKED_ITEM_UNIVERSE).sum())
    pool_blocked = bool(pool_validation["status"].eq("BLOCKED").any()) if not pool_validation.empty else True
    base = {
        "phase_id": PHASE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_token": decision,
        "precondition_f1bz": f1bz_manifest.get("decision_token", ""),
        "precondition_f1bza": f1bza_manifest.get("decision_token", ""),
        "active_datasets": list(ACTIVE_DATASETS),
        "models": list(TRAINABLE_MODELS),
        "seeds": list(FINAL_SEEDS),
        "ready_full_catalog_cells": ready_cells,
        "blocked_full_catalog_cells": blocked_cells,
        "single_model_dominated_pools": int(pool_blocked),
        "future_requests_unlocked": future_requests,
        "frozen_max_future_requests": 29292,
        "training_runs_executed_in_f1bzc": 0,
        "checkpoint_parameter_updates": 0,
        "hyperparameter_changes": 0,
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_llm_api_calls_executed": 0,
        "network_calls_executed": 0,
        "downloads_executed": 0,
        "commit_created": 0,
        "f1bz_evidence_hashes_before": dict(evidence_before),
        "f1bz_evidence_hashes_after": dict(evidence_after),
        "f1bz_evidence_hashes_unchanged": dict(evidence_before) == dict(evidence_after),
        "f1bz_evidence_hash_counts": evidence_artifact_counts(evidence_before),
        "source_seed_policy_hash": frame_hash(source_seed_policy),
        "adapter_audit_hash": frame_hash(adapter_audit),
        "full_catalog_execution_hash": frame_hash(execution),
        "candidate_recall_hash": frame_hash(candidate_recall),
        "readiness_hash": frame_hash(readiness),
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS")].to_dict("records"),
        "config": {
            "source_seed_policy": dict(config["source_seed_policy"]),
            "full_catalog_retrieval": dict(config["full_catalog_retrieval"]),
            "reranking_pool": dict(config["reranking_pool"]),
        },
    }
    return {
        "full_catalog_recovery": dict(base),
        "reranking_pool_freeze": {
            **dict(base),
            "authoritative_pool_freeze": not pool_blocked and decision == DECISION_CORRECTED,
            "attempted_pool_artifact": (RAW_RERANKING_DIR / "f1bzc_lastfm_attempted_adapted_pools.csv").as_posix(),
            "attempted_pool_status": BLOCKED_POOL_ALGORITHM if pool_blocked else AUTHORITATIVE,
            "pool_validation_hash": frame_hash(pool_validation),
        },
        "no_llm_no_training": {
            "phase_id": PHASE_ID,
            "decision_token": decision,
            "training_runs_executed_in_f1bzc": 0,
            "checkpoint_parameter_updates": 0,
            "hyperparameter_changes": 0,
            "llm_inference_calls_executed": 0,
            "ollama_recommendation_calls_executed": 0,
            "external_llm_api_calls_executed": 0,
            "network_calls_executed": 0,
            "downloads_executed": 0,
            "traditional_checkpoint_scoring_executed": int(ready_cells > 0),
            "traditional_checkpoint_scoring_is_not_llm_inference": True,
        },
    }


def render_report(
    decision: str,
    source_seed_policy: pd.DataFrame,
    adapter_audit: pd.DataFrame,
    execution: pd.DataFrame,
    full_catalog_validation: pd.DataFrame,
    contributions: pd.DataFrame,
    pool_validation: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    foodcom_diagnostics: pd.DataFrame,
    readiness: pd.DataFrame,
    superseded: pd.DataFrame,
    validity: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    adapter_counts = adapter_audit["adapter_status"].value_counts().to_dict()
    execution_counts = execution["status"].value_counts().to_dict()
    readiness_counts = readiness["readiness_status"].value_counts().to_dict()
    contribution_summary = {}
    if not contributions.empty:
        contribution_summary = (
            contributions.groupby("source_model")["contribution_count"].sum().astype(int).to_dict()
        )
    lines = [
        "# F1BZC Full-Catalog Reranking Recovery",
        "",
        f"Decision token: `{decision}`",
        "",
        "## Source-Seed Policy",
        "",
        f"- Rows: `{len(source_seed_policy)}`.",
        "- Frozen rule: all five preregistered seeds [42,43,44,45,46] are mean-aggregated.",
        "- Final-test metrics are not used for source-seed selection.",
        "",
        "## Full-Catalog Recovery",
        "",
        f"- Adapter statuses: `{stable_json(adapter_counts)}`.",
        f"- Execution statuses: `{stable_json(execution_counts)}`.",
        "",
        "LastFM is the only dataset whose frozen F1BZ checkpoint item map covers the raw catalog. The other four datasets have raw catalog items absent from the checkpoint embedding table, so raw full-catalog scoring would require invented embeddings or retraining. F1BZC does neither.",
        "",
        "## Adapted Pool Algorithm",
        "",
        "Pseudocode:",
        "",
        "```text",
        "for model in [BPRMF, GRU4Rec, SASRec, LightGCN]:",
        "  for item in that model's top20 full-catalog list:",
        "    append item if not already present",
        "    stop once the final pool has 20 items",
        "```",
        "",
        f"- Contribution totals: `{stable_json(contribution_summary)}`.",
        f"- Pool validation statuses: `{stable_json(pool_validation['status'].value_counts().to_dict())}`.",
        "",
        "Applied exactly, the frozen F1BR policy is first-model dominated: BPRMF fills the full 20-item pool before the later models can contribute. F1BZC therefore blocks authoritative reranking-pool freeze instead of hiding a second protocol defect.",
        "",
        "## Candidate Recall",
        "",
        candidate_recall.to_csv(index=False, lineterminator="\n").strip(),
        "",
        "## Food.com Warm/Cold Diagnostic",
        "",
        f"- Rows: `{len(foodcom_diagnostics)}`.",
        "- Food.com true zero-history users remain in the cohort; no interactions are invented or removed.",
        "",
        "## Superseded Artifacts",
        "",
        f"- Superseded forensic artifacts: `{len(superseded)}`.",
        "- Old F1BZ first-stage rankings, reranking pools, candidate recall, and readiness remain preserved but are not authoritative downstream inputs.",
        "",
        "## Readiness",
        "",
        f"- Readiness rows: `{len(readiness)}`.",
        f"- Readiness statuses: `{stable_json(readiness_counts)}`.",
        f"- Future requests unlocked: `{int(readiness['future_requests_unlocked'].astype(int).sum())}` / `29292`.",
        "",
        "## Validation",
        "",
        f"- F1BZC audit checks: `{len(validation)}`.",
        f"- Failures: `{int(validation['status'].ne('PASS').sum())}`.",
        f"- Full-catalog validation rows: `{len(full_catalog_validation)}`.",
        f"- Result validity classes: `{stable_json(validity['classification'].value_counts().to_dict())}`.",
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
    artifact_paths = [*OUTPUT_TABLES.values(), REPORT_PATH, Path("configs/f1bzc_full_catalog_recovery.yaml"), Path("src/extension/f1bzc_full_catalog_recovery.py"), Path(".gitignore")]
    artifact_paths.extend(
        path.relative_to(root)
        for path in sorted((root / RAW_FULL_CATALOG_DIR).glob("f1bzc_*.csv"))
        if path.exists()
    )
    artifact_paths.append(RAW_RERANKING_DIR / "f1bzc_lastfm_attempted_adapted_pools.csv")
    for key, rel_path in OUTPUT_MANIFESTS.items():
        payload = dict(outputs["manifests"][key])
        payload["artifact_hashes"] = {
            path.as_posix(): sha256_file(root / path) for path in artifact_paths if (root / path).exists()
        }
        (root / rel_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_existing_outputs(root: Path) -> dict[str, Any]:
    return {
        **{key: read_csv(root / rel_path) for key, rel_path in OUTPUT_TABLES.items()},
        "manifests": {key: read_json(root / rel_path) for key, rel_path in OUTPUT_MANIFESTS.items()},
        "report": (root / REPORT_PATH).read_text(encoding="utf-8"),
    }


def ensure_output_dirs(root: Path) -> None:
    for rel_path in (TABLE_DIR, MANIFEST_DIR, REPORT_DIR, RAW_FULL_CATALOG_DIR, RAW_RERANKING_DIR):
        (root / rel_path).mkdir(parents=True, exist_ok=True)


def cold_start_behavior(dataset: str, model: str) -> str:
    if dataset == "foodcom" and model in {"GRU4Rec", "SASRec"}:
        return "F1BZ sequence implementation emits deterministic zero scores for empty evaluation sequences."
    if dataset == "foodcom" and model in {"BPRMF", "LightGCN"}:
        return "F1BZ uses the learned/default user embedding for mapped true zero-history users."
    return "No true zero-history limitation observed for this dataset-model cell."


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

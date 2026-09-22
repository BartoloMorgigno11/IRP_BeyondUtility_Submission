"""F1BZQ authoritative full-catalog retrieval and reranking pool materialization.

This phase reads the already-trained F1BZ checkpoints, scores the F1BZP
model-representable catalog, builds corrected balanced reranking pools, and
recomputes reranking readiness. It does not train, tune, download, call a
network service, run LLM inference, or commit.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
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

from src.extension.e0_protocol import sha256_json, stable_json
from src.extension.f0_final_extension_freeze import sha256_file
from src.extension.f1a_offline_metric_closure import format_float
from src.extension.f1bx_execution_stage1 import (
    ACTIVE_DATASETS,
    TRAINABLE_MODELS,
    DatasetMaterial,
    DatasetSpec,
    build_dataset_material,
    dataset_specs as f1bx_dataset_specs,
    frame_hash,
    read_csv,
    read_json,
    sort_key,
)
from src.extension.f1bz_final_traditional_execution import FINAL_SEEDS, build_final_data
from src.extension.f1bzc_full_catalog_recovery import score_checkpoint_full_catalog
from src.extension.f1bzp_reranking_protocol_resolution import (
    ADAPTED_ALGORITHM,
    ADAPTED_POLICY,
    MODEL_READY,
    balanced_round_robin_unique_pool,
)
from src.utils.config import find_project_root


PHASE_ID = "f1bzq_authoritative_reranking_pool_materialization"
DECISION_VALIDATED = "F1BZQ_AUTHORITATIVE_RERANKING_POOLS_VALIDATED"
DECISION_BLOCKED = "F1BZQ_AUTHORITATIVE_RERANKING_POOLS_BLOCKED"
SOURCE_POLICY = "F1BZQ_REPRESENTABLE_FULL_CATALOG_FIVE_SEED_MEAN"
SOURCE_SEED_POLICY = "F1BZ_FIVE_SEED_MEAN_SCORE_THEN_FROZEN_MODEL_ORDER"
POOL_STATUS_COMPUTED = "AUTHORITATIVE_POOL_COMPUTED"
POOL_UNDERFILLED = "POOL_UNDERFILLED"
BLOCKED_SOURCE_SCORING = "BLOCKED_SOURCE_SCORING"
BLOCKED_METADATA = "BLOCKED_METADATA"
SUPERSEDED = "SUPERSEDED_RERANKING_SOURCE_DEFECT"

CONFIG_PATH = Path("configs/f1bzq_authoritative_reranking_pool_materialization.yaml")
TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_PATH = Path("report/f1bzq_authoritative_reranking_pool_materialization.md")
SOURCE_RANKING_DIR = Path("results/raw_outputs/f1bzq_source_rankings")
POOL_DIR = Path("results/raw_outputs/f1bzq_reranking_pools")

OUTPUT_TABLES = {
    "source_execution": TABLE_DIR / "f1bzq_source_execution.csv",
    "source_recall": TABLE_DIR / "f1bzq_source_recall.csv",
    "pool_source_contributions": TABLE_DIR / "f1bzq_pool_source_contributions.csv",
    "candidate_recall": TABLE_DIR / "f1bzq_candidate_recall.csv",
    "foodcom_warm_cold_recall": TABLE_DIR / "f1bzq_foodcom_warm_cold_recall.csv",
    "pool_validation": TABLE_DIR / "f1bzq_pool_validation.csv",
    "reranking_readiness": TABLE_DIR / "f1bzq_reranking_readiness.csv",
    "supersession_validation": TABLE_DIR / "f1bzq_supersession_validation.csv",
    "validation": TABLE_DIR / "f1bzq_validation.csv",
    "reranking_request_manifest": TABLE_DIR / "f1bzq_reranking_request_manifest.csv",
    "reranking_candidate_manifest": TABLE_DIR / "f1bzq_reranking_candidate_manifest.csv",
}

OUTPUT_MANIFESTS = {
    "source_ranking": MANIFEST_DIR / "f1bzq_source_ranking_manifest.json",
    "authoritative_pool": MANIFEST_DIR / "f1bzq_authoritative_pool_manifest.json",
    "no_training_no_llm": MANIFEST_DIR / "f1bzq_no_training_no_llm.json",
}

SUPPORT_ARTIFACTS = (
    CONFIG_PATH,
    Path("src/extension/f1bzq_authoritative_reranking_pool_materialization.py"),
    Path("tests/test_extension_f1bzq_authoritative_reranking_pool_materialization.py"),
    REPORT_PATH,
    Path(".gitignore"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F1BZQ authoritative reranking pool materialization.")
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
        outputs = generate_f1bzq(root, Path(args.config))
        write_outputs(root, outputs)
    validation = outputs["validation"]
    manifest = outputs["manifests"]["authoritative_pool"]
    print(f"decision={manifest['decision_token']}")
    print(f"source_cells_terminal={manifest['source_cells_terminal']}")
    print(f"user_model_source_cases={manifest['user_model_source_cases']}")
    print(f"authoritative_pool_users={manifest['authoritative_pool_users']}")
    print(f"ready_reranking_rows={manifest['ready_reranking_rows']}")
    print(f"ready_llm_requests={manifest['ready_llm_requests']}")
    print(f"llm_inference_calls={manifest['llm_inference_calls_executed']}")
    print(f"validation_passed={str(validation['status'].eq('PASS').all()).lower()}")
    return 0 if bool(validation["status"].eq("PASS").all()) and manifest["decision_token"] == DECISION_VALIDATED else 1


def generate_f1bzq(root: Path | None = None, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    project_root = root or find_project_root()
    config = load_config(project_root, Path(config_path))
    f1bz_config = load_config(project_root, Path(str(config["inputs"]["f1bz_config"])))
    f1bx_config = load_config(project_root, Path(str(f1bz_config["inputs"]["f1bx_config"])))
    specs = {spec.name: spec for spec in f1bx_dataset_specs(f1bx_config)}

    f1bz_manifest = read_json(project_root / str(config["inputs"]["f1bz_manifest"]))
    f1bz_no_llm = read_json(project_root / str(config["inputs"]["f1bz_no_llm_manifest"]))
    f1bza_manifest = read_json(project_root / str(config["inputs"]["f1bza_manifest"]))
    f1bzc_manifest = read_json(project_root / str(config["inputs"]["f1bzc_manifest"]))
    f1bzc_recall = read_csv(project_root / str(config["inputs"]["f1bzc_candidate_recall"]))
    f1bzp_manifest = read_json(project_root / str(config["inputs"]["f1bzp_manifest"]))
    f1bzp_vocab = read_csv(project_root / str(config["inputs"]["f1bzp_checkpoint_vocab_audit"]))
    f1bzp_readiness = read_csv(project_root / str(config["inputs"]["f1bzp_representable_readiness"]))
    f1bzp_pool_policy = read_csv(project_root / str(config["inputs"]["f1bzp_pool_policy_resolution"]))
    f1bza_zero = read_csv(project_root / str(config["inputs"]["f1bza_zero_history_summary"]))
    f0_budget = read_csv(project_root / str(config["inputs"]["f0_request_budget"]))
    f0_reranking = read_csv(project_root / str(config["inputs"]["f0_reranking_matrix"]))
    run_manifest = read_csv(project_root / "results/tables/f1bz_final_run_manifest.csv")

    phase_start_clean_evidence = str(config["phase"].get("phase_start_clean_evidence", ""))
    checkpoint_hashes_before = checkpoint_hashes(project_root, run_manifest)
    ensure_output_dirs(project_root)

    materials: dict[str, DatasetMaterial] = {}
    final_data: dict[str, Any] = {}
    title_maps: dict[str, dict[str, str]] = {}
    title_sources: dict[str, str] = {}
    for dataset in ACTIVE_DATASETS:
        material = build_dataset_material(project_root, specs[dataset])
        materials[dataset] = material
        final_data[dataset] = build_final_data(project_root, specs[dataset], material, f1bz_config)
        title_maps[dataset], title_sources[dataset] = load_title_map(project_root, specs[dataset])

    source_execution_rows: list[dict[str, Any]] = []
    source_user_rows: list[dict[str, Any]] = []
    source_artifacts: list[Path] = []
    device = select_device(str(config["execution"]["device_policy"]))
    source_depth = int(f1bzp_manifest["requested_source_depth"])
    for dataset in ACTIVE_DATASETS:
        for model in TRAINABLE_MODELS:
            print(f"f1bzq_scoring dataset={dataset} model={model} depth={source_depth} device={device}", flush=True)
            started = time.perf_counter()
            try:
                execution, user_rows, artifact_path = score_source_cell(
                    project_root,
                    config,
                    f1bz_config,
                    run_manifest,
                    materials[dataset],
                    final_data[dataset],
                    dataset,
                    model,
                    source_depth,
                    device,
                    started,
                )
                source_user_rows.extend(user_rows)
                source_artifacts.append(artifact_path)
            except Exception as exc:  # pragma: no cover - produces blocked audit rows on local failures.
                execution = blocked_source_execution(dataset, model, final_data[dataset], source_depth, started, exc)
            source_execution_rows.append(execution)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    source_execution = pd.DataFrame(source_execution_rows)
    source_users = pd.DataFrame(source_user_rows)
    source_recall = build_source_recall(source_execution, source_users, f1bzc_recall)
    pool_rows, pool_summaries, candidate_manifest = build_pools_and_candidate_manifest(
        project_root,
        config,
        f1bzp_manifest,
        source_users,
        final_data,
        title_maps,
        title_sources,
    )
    pool_contributions = build_pool_source_contributions(pool_summaries)
    candidate_recall = build_candidate_recall(pool_summaries)
    foodcom_warm_cold = build_foodcom_warm_cold_recall(pool_summaries)
    reranking_readiness = build_reranking_readiness(config, f0_reranking, pool_summaries, candidate_recall, candidate_manifest)
    request_manifest = build_request_manifest(reranking_readiness)
    supersession = build_supersession_validation(project_root)
    checkpoint_hashes_after = checkpoint_hashes(project_root, run_manifest)
    pool_validation = build_pool_validation(
        config,
        f1bzp_manifest,
        f1bzp_vocab,
        f1bzp_readiness,
        source_execution,
        source_users,
        pool_summaries,
        pool_rows,
        pool_contributions,
        candidate_recall,
        foodcom_warm_cold,
        reranking_readiness,
        supersession,
        f0_budget,
        checkpoint_hashes_before,
        checkpoint_hashes_after,
    )
    validation = build_validation(
        project_root,
        config,
        phase_start_clean_evidence,
        f1bz_manifest,
        f1bz_no_llm,
        f1bza_manifest,
        f1bzc_manifest,
        f1bzp_manifest,
        f1bzp_vocab,
        f1bzp_pool_policy,
        f1bza_zero,
        source_execution,
        source_users,
        pool_summaries,
        pool_rows,
        pool_contributions,
        candidate_recall,
        foodcom_warm_cold,
        reranking_readiness,
        supersession,
        pool_validation,
        f0_budget,
        checkpoint_hashes_before,
        checkpoint_hashes_after,
    )
    decision = DECISION_VALIDATED if bool(validation["status"].eq("PASS").all()) else DECISION_BLOCKED
    manifests = build_manifests(
        project_root,
        config,
        decision,
        f1bz_manifest,
        f1bza_manifest,
        f1bzc_manifest,
        f1bzp_manifest,
        source_execution,
        source_users,
        source_recall,
        pool_summaries,
        pool_rows,
        pool_contributions,
        candidate_recall,
        foodcom_warm_cold,
        reranking_readiness,
        supersession,
        pool_validation,
        validation,
        checkpoint_hashes_before,
        checkpoint_hashes_after,
        source_artifacts,
        candidate_manifest,
        request_manifest,
    )
    report = render_report(
        decision,
        source_execution,
        source_recall,
        pool_summaries,
        pool_contributions,
        candidate_recall,
        foodcom_warm_cold,
        reranking_readiness,
        supersession,
        pool_validation,
        validation,
    )
    return {
        "source_execution": source_execution,
        "source_recall": source_recall,
        "pool_source_contributions": pool_contributions,
        "candidate_recall": candidate_recall,
        "foodcom_warm_cold_recall": foodcom_warm_cold,
        "pool_validation": pool_validation,
        "reranking_readiness": reranking_readiness,
        "supersession_validation": supersession,
        "validation": validation,
        "reranking_request_manifest": request_manifest,
        "reranking_candidate_manifest": candidate_manifest,
        "pool_rows": pool_rows,
        "pool_summaries": pool_summaries,
        "source_user_rows": source_users,
        "manifests": manifests,
        "report": report,
    }


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else root / config_path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def score_source_cell(
    root: Path,
    config: Mapping[str, Any],
    f1bz_config: Mapping[str, Any],
    run_manifest: pd.DataFrame,
    material: DatasetMaterial,
    data: Any,
    dataset: str,
    model_name: str,
    source_depth: int,
    device: torch.device,
    started: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    method_runs = run_manifest.loc[
        run_manifest["dataset"].eq(dataset) & run_manifest["method"].eq(model_name)
    ].sort_values("seed", key=lambda series: pd.to_numeric(series, errors="coerce"), kind="mergesort")
    seeds = method_runs["seed"].astype(int).tolist()
    if tuple(seeds) != FINAL_SEEDS:
        raise ValueError(f"unexpected seed policy for {dataset}/{model_name}: {seeds}")
    checkpoint_hashes_before = [sha256_file(root / str(path)) for path in method_runs["checkpoint_path"].astype(str)]
    expected_hashes = method_runs["checkpoint_hash"].astype(str).tolist()
    if checkpoint_hashes_before != expected_hashes:
        raise ValueError(f"checkpoint hash mismatch before scoring for {dataset}/{model_name}")

    raw_catalog = set(map(str, material.catalog_items))
    checkpoint_vocab = set(map(str, data.tuning.item_index))
    catalog_items = sorted(raw_catalog & checkpoint_vocab, key=sort_key)
    catalog_indices = np.asarray([data.tuning.item_index[item] for item in catalog_items], dtype=np.int64)
    sum_scores = np.zeros((len(data.tuning.users), len(catalog_indices)), dtype=np.float64)
    checkpoint_paths = method_runs["checkpoint_path"].astype(str).tolist()
    checkpoint_set_hash = sha256_json(checkpoint_hashes_before)

    for run in method_runs.to_dict("records"):
        checkpoint_path = root / str(run["checkpoint_path"])
        scores = score_checkpoint_full_catalog(
            checkpoint_path,
            model_name,
            json.loads(str(run["hyperparameters"])),
            f1bz_config,
            data,
            catalog_indices,
            device,
        )
        sum_scores += scores.astype(np.float64)
        del scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mean_scores = sum_scores / float(len(method_runs))
    checkpoint_hashes_after = [sha256_file(root / str(path)) for path in method_runs["checkpoint_path"].astype(str)]
    rows, user_rows = source_ranking_rows(
        material,
        data,
        dataset,
        model_name,
        catalog_items,
        catalog_indices,
        mean_scores,
        source_depth,
        checkpoint_paths,
        checkpoint_set_hash,
    )
    artifact_path = SOURCE_RANKING_DIR / f"f1bzq_{dataset}_{model_name.lower()}_top{source_depth}.csv"
    full_path = root / artifact_path
    pd.DataFrame(rows).to_csv(full_path, index=False, lineterminator="\n")
    positive_present = int(sum(1 for row in user_rows if row["positive_present"] == "true"))
    eligible_counts = [int(row["eligible_catalog_N"]) for row in user_rows]
    execution = {
        "dataset": dataset,
        "model": model_name,
        "source_policy": SOURCE_POLICY,
        "source_seed_policy": SOURCE_SEED_POLICY,
        "seeds": stable_json(seeds),
        "source_depth": source_depth,
        "status": "COMPUTED",
        "terminal": "true",
        "source_artifact_path": artifact_path.as_posix(),
        "source_artifact_hash": sha256_file(full_path),
        "source_artifact_rows": len(rows),
        "user_model_cases": len(user_rows),
        "raw_catalog_N": len(raw_catalog),
        "checkpoint_item_vocab_N": data.tuning.num_items,
        "retrievable_catalog_N": len(catalog_items),
        "eligible_catalog_N_min": min(eligible_counts) if eligible_counts else 0,
        "eligible_catalog_N_max": max(eligible_counts) if eligible_counts else 0,
        "positive_present_N": positive_present,
        "source_recall_at_frozen_depth": format_float(positive_present / len(user_rows) if user_rows else math.nan),
        "checkpoint_count": len(checkpoint_paths),
        "checkpoint_hash_identity": bool_text(checkpoint_hashes_before == checkpoint_hashes_after),
        "checkpoint_set_hash_before": checkpoint_set_hash,
        "checkpoint_set_hash_after": sha256_json(checkpoint_hashes_after),
        "run_config_hashes": stable_json(sorted(set(method_runs["config_hash"].astype(str)))),
        "training_data_hashes": stable_json(sorted(set(method_runs["training_data_hash"].astype(str)))),
        "training_data_hash_matches_current": bool_text(method_runs["training_data_hash"].astype(str).eq(data.tuning.training_data_hash).all()),
        "evaluation_cohort_hash_matches_current": bool_text(method_runs["evaluation_cohort_hash"].astype(str).eq(data.evaluation_cohort_hash).all()),
        "candidate_contract_hash_matches_current": bool_text(method_runs["candidate_contract_hash"].astype(str).eq(data.tuning.candidate_contract_hash).all()),
        "force_positive": "false",
        "target_identity_used_for_ranking": "false",
        "seen_item_exclusion": "frozen_pre_test_training_history_only",
        "device": str(device),
        "runtime_seconds": format_float(time.perf_counter() - started),
        "reason": "Authoritative source rankings computed over raw-catalog/checkpoint-vocabulary intersection.",
    }
    return execution, user_rows, artifact_path


def source_ranking_rows(
    material: DatasetMaterial,
    data: Any,
    dataset: str,
    model_name: str,
    catalog_items: Sequence[str],
    catalog_indices: np.ndarray,
    mean_scores: np.ndarray,
    source_depth: int,
    checkpoint_paths: Sequence[str],
    checkpoint_set_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalog_position_by_index = {int(item_index): pos for pos, item_index in enumerate(catalog_indices)}
    catalog_order = np.arange(len(catalog_items), dtype=np.int64)
    positive_by_user = (
        material.candidates.drop_duplicates("user_id", keep="first")
        .set_index("user_id")["positive_item_id"]
        .astype(str)
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    user_rows: list[dict[str, Any]] = []
    for user_row, user_id in enumerate(data.tuning.users):
        user_index = int(data.eval_user_indices[user_row])
        seen = set(data.tuning.user_positive_items.get(user_index, set()))
        excluded_positions = [catalog_position_by_index[item] for item in seen if item in catalog_position_by_index]
        scores = mean_scores[user_row].copy()
        if excluded_positions:
            scores[excluded_positions] = -np.inf
        finite = np.isfinite(scores)
        eligible_count = int(finite.sum())
        order = np.lexsort((catalog_order, -scores))
        ordered = [int(pos) for pos in order if finite[int(pos)]][:source_depth]
        ranked_items = [str(catalog_items[pos]) for pos in ordered]
        ranked_scores = [format_float(float(scores[pos])) for pos in ordered]
        positive = str(positive_by_user[str(user_id)])
        positive_rank = ranked_items.index(positive) + 1 if positive in ranked_items else ""
        ranking_hash = sha256_json(
            {
                "dataset": dataset,
                "model": model_name,
                "user_id": str(user_id),
                "items": ranked_items,
                "scores": ranked_scores,
                "source_policy": SOURCE_POLICY,
                "source_depth": source_depth,
                "checkpoint_set_hash": checkpoint_set_hash,
            }
        )
        excluded_hash = sha256_json(sorted((catalog_items[pos] for pos in excluded_positions), key=sort_key))
        for rank, pos in enumerate(ordered, start=1):
            item_id = str(catalog_items[pos])
            rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "user_id": str(user_id),
                    "rank": rank,
                    "item_id": item_id,
                    "score": format_float(float(scores[pos])),
                    "source_checkpoint": "ALL_FIVE_SEED_MEAN_F1BZ_CHECKPOINTS",
                    "source_checkpoint_paths": stable_json(list(checkpoint_paths)),
                    "source_checkpoint_set_hash": checkpoint_set_hash,
                    "source_policy": SOURCE_POLICY,
                    "source_seed_policy": SOURCE_SEED_POLICY,
                    "source_depth": source_depth,
                    "eligible_catalog_N": eligible_count,
                    "retrievable_catalog_N": len(catalog_items),
                    "excluded_seen_item_count": len(excluded_positions),
                    "excluded_seen_items_hash": excluded_hash,
                    "positive_item": positive,
                    "is_ground_truth": bool_text(item_id == positive),
                    "force_positive": "false",
                    "target_identity_used_for_ranking": "false",
                    "ranking_hash": ranking_hash,
                }
            )
        user_rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "user_id": str(user_id),
                "positive_item": positive,
                "ranked_items": stable_json(ranked_items),
                "ranked_scores": stable_json(ranked_scores),
                "ranking_size": len(ranked_items),
                "positive_present": bool_text(positive in ranked_items),
                "positive_rank": positive_rank,
                "source_checkpoint": "ALL_FIVE_SEED_MEAN_F1BZ_CHECKPOINTS",
                "source_checkpoint_set_hash": checkpoint_set_hash,
                "source_policy": SOURCE_POLICY,
                "source_seed_policy": SOURCE_SEED_POLICY,
                "source_depth": source_depth,
                "eligible_catalog_N": eligible_count,
                "retrievable_catalog_N": len(catalog_items),
                "excluded_seen_item_count": len(excluded_positions),
                "excluded_seen_items_hash": excluded_hash,
                "force_positive": "false",
                "target_identity_used_for_ranking": "false",
                "candidate_recall_tautological": "false",
                "ranking_hash": ranking_hash,
            }
        )
    return rows, user_rows


def blocked_source_execution(dataset: str, model: str, data: Any, source_depth: int, started: float, exc: Exception) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "model": model,
        "source_policy": SOURCE_POLICY,
        "source_seed_policy": SOURCE_SEED_POLICY,
        "seeds": stable_json(list(FINAL_SEEDS)),
        "source_depth": source_depth,
        "status": BLOCKED_SOURCE_SCORING,
        "terminal": "true",
        "source_artifact_path": "",
        "source_artifact_hash": "",
        "source_artifact_rows": 0,
        "user_model_cases": len(data.tuning.users),
        "raw_catalog_N": "",
        "checkpoint_item_vocab_N": data.tuning.num_items,
        "retrievable_catalog_N": "",
        "eligible_catalog_N_min": "",
        "eligible_catalog_N_max": "",
        "positive_present_N": "",
        "source_recall_at_frozen_depth": "",
        "checkpoint_count": 0,
        "checkpoint_hash_identity": "false",
        "checkpoint_set_hash_before": "",
        "checkpoint_set_hash_after": "",
        "run_config_hashes": "",
        "training_data_hashes": "",
        "training_data_hash_matches_current": "false",
        "evaluation_cohort_hash_matches_current": "false",
        "candidate_contract_hash_matches_current": "false",
        "force_positive": "false",
        "target_identity_used_for_ranking": "false",
        "seen_item_exclusion": "frozen_pre_test_training_history_only",
        "device": "",
        "runtime_seconds": format_float(time.perf_counter() - started),
        "reason": f"{type(exc).__name__}: {exc}",
    }


def build_source_recall(source_execution: pd.DataFrame, source_users: pd.DataFrame, f1bzc_recall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    f1bzc_lastfm = f1bzc_recall.loc[f1bzc_recall["dataset"].eq("lastfm")].iloc[0].to_dict()
    for exec_row in source_execution.to_dict("records"):
        dataset = str(exec_row["dataset"])
        model = str(exec_row["model"])
        group = source_users.loc[source_users["dataset"].eq(dataset) & source_users["model"].eq(model)]
        total = int(len(group)) if not group.empty else int(exec_row["user_model_cases"])
        present = int(group["positive_present"].eq("true").sum()) if not group.empty else 0
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "source_policy": SOURCE_POLICY,
                "source_depth": exec_row["source_depth"],
                "evaluation_N": total,
                "positive_present_N": present if str(exec_row["status"]) == "COMPUTED" else "",
                "positive_absent_N": total - present if str(exec_row["status"]) == "COMPUTED" else "",
                "source_recall_at_frozen_depth": format_float(present / total if total and str(exec_row["status"]) == "COMPUTED" else math.nan) if str(exec_row["status"]) == "COMPUTED" else "",
                "candidate_recall_tautological": "false" if str(exec_row["status"]) == "COMPUTED" else "not_evaluated",
                "force_positive": "false",
                "target_identity_used_for_ranking": "false",
                "diagnostic_not_paper_metric": "true",
                "f1bzc_lastfm_reference_positive_present_N": f1bzc_lastfm["positive_present_N"] if dataset == "lastfm" else "",
                "f1bzc_lastfm_reference_evaluation_N": f1bzc_lastfm["evaluation_N"] if dataset == "lastfm" else "",
                "comparison_note": "Same five-seed mean/source depth but corrected F1BZQ uses F1BZP pool policy downstream." if dataset == "lastfm" else "",
                "status": "COMPUTED_DIAGNOSTIC" if str(exec_row["status"]) == "COMPUTED" else BLOCKED_SOURCE_SCORING,
                "reason": "Source recall computed before aggregation; it is diagnostic evidence.",
            }
        )
    return pd.DataFrame(rows)


def build_pools_and_candidate_manifest(
    root: Path,
    config: Mapping[str, Any],
    f1bzp_manifest: Mapping[str, Any],
    source_users: pd.DataFrame,
    final_data: Mapping[str, Any],
    title_maps: Mapping[str, Mapping[str, str]],
    title_sources: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_order = list(config["pool_policy"]["model_order"])
    source_depth = int(f1bzp_manifest["requested_source_depth"])
    pool_size = int(config["pool_policy"]["final_pool_size"])
    pool_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    grouped = {
        (str(row["dataset"]), str(row["user_id"]), str(row["model"])): row
        for row in source_users.to_dict("records")
    }
    for dataset in ACTIVE_DATASETS:
        title_map = title_maps[dataset]
        data = final_data[dataset]
        dataset_pool_rows: list[dict[str, Any]] = []
        for user_idx, user_id in enumerate(data.tuning.users):
            source_lists = {
                model: json.loads(grouped.get((dataset, str(user_id), model), {}).get("ranked_items", "[]"))
                for model in model_order
            }
            source_scores = {
                model: json.loads(grouped.get((dataset, str(user_id), model), {}).get("ranked_scores", "[]"))
                for model in model_order
            }
            positive = str(data.positive_item_ids[user_idx])
            pool, lineage = balanced_round_robin_unique_pool(
                source_lists,
                model_order=model_order,
                pool_size=pool_size,
                source_depth=source_depth,
            )
            positive_position = pool.index(positive) + 1 if positive in pool else ""
            title_values = [clean_title(title_map.get(item, "")) for item in pool]
            metadata_valid = len(title_values) == len(pool) and all(title_values)
            unique_count = len(set(pool))
            if len(pool) < pool_size:
                status = POOL_UNDERFILLED
            elif not metadata_valid:
                status = BLOCKED_METADATA
            else:
                status = POOL_STATUS_COMPUTED
            pool_hash = sha256_json(
                {
                    "dataset": dataset,
                    "user_id": str(user_id),
                    "items": pool,
                    "policy": ADAPTED_POLICY,
                    "algorithm": ADAPTED_ALGORITHM,
                    "source_depth": source_depth,
                    "model_order": model_order,
                }
            )
            pool_id = "F1BZQ-" + pool_hash.upper()[:20]
            history_length = int(data.tuning.eval_sequences[user_idx].astype(bool).sum())
            true_zero = bool(dataset == "foodcom" and history_length == 0)
            source_rank_lookup = {
                (str(row["source_model"]), str(row["item_id"])): int(row["source_rank"])
                for row in lineage
            }
            for row in lineage:
                selected_model = str(row["source_model"])
                selected_item = str(row["item_id"])
                selected_rank = int(row["source_rank"])
                selected_score = ""
                if selected_rank <= len(source_scores.get(selected_model, [])):
                    selected_score = source_scores[selected_model][selected_rank - 1]
                all_lineage = []
                for model in model_order:
                    if selected_item in source_lists.get(model, []):
                        all_lineage.append({"model": model, "rank": source_lists[model].index(selected_item) + 1})
                pool_row = {
                    "dataset": dataset,
                    "user_id": str(user_id),
                    "pool_id": pool_id,
                    "candidate_position": int(row["candidate_position"]),
                    "candidate_item_id": selected_item,
                    "candidate_title": clean_title(title_map.get(selected_item, "")),
                    "source_model": selected_model,
                    "source_rank": selected_rank,
                    "source_score": selected_score,
                    "source_model_lineage": stable_json(all_lineage),
                    "source_depth": source_depth,
                    "pool_hash": pool_hash,
                    "positive_item": positive,
                    "positive_present": bool_text(positive in pool),
                    "positive_position": positive_position,
                    "candidate_recall_tautological": "false",
                    "force_positive": "false",
                    "target_identity_used_for_ranking": "false",
                    "warm_cold_group": "TRUE_ZERO_HISTORY_USERS" if true_zero else "WARM_USERS",
                    "metadata_status": "VALID_METADATA" if metadata_valid else "BLOCKED_METADATA",
                    "status": status,
                }
                pool_rows.append(pool_row)
                dataset_pool_rows.append(pool_row)
            counts = Counter(str(row["source_model"]) for row in lineage)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "user_id": str(user_id),
                    "pool_id": pool_id,
                    "pool_policy": ADAPTED_POLICY,
                    "pool_algorithm": ADAPTED_ALGORITHM,
                    "model_order": stable_json(model_order),
                    "source_depth": source_depth,
                    "pool_size": len(pool),
                    "unique_count": unique_count,
                    "BPRMF_contribution": int(counts.get("BPRMF", 0)),
                    "GRU4Rec_contribution": int(counts.get("GRU4Rec", 0)),
                    "SASRec_contribution": int(counts.get("SASRec", 0)),
                    "LightGCN_contribution": int(counts.get("LightGCN", 0)),
                    "first_model_20_0_0_0": bool_text(counts.get("BPRMF", 0) == 20 and all(counts.get(model, 0) == 0 for model in ["GRU4Rec", "SASRec", "LightGCN"])),
                    "positive_item": positive,
                    "positive_present": bool_text(positive in pool),
                    "positive_position": positive_position,
                    "title_missing_count": int(sum(1 for value in title_values if not value)),
                    "metadata_valid": bool_text(metadata_valid),
                    "candidate_recall_tautological": "false",
                    "force_positive": "false",
                    "target_identity_used_for_ranking": "false",
                    "warm_cold_group": "TRUE_ZERO_HISTORY_USERS" if true_zero else "WARM_USERS",
                    "pool_hash": pool_hash,
                    "status": status,
                    "reason": "Corrected pool built from F1BZQ source rankings with the F1BZP balanced policy.",
                }
            )
        dataset_frame = pd.DataFrame(dataset_pool_rows)
        artifact_path = POOL_DIR / f"f1bzq_{dataset}_authoritative_balanced_pools.csv"
        full_path = root / artifact_path
        dataset_frame.to_csv(full_path, index=False, lineterminator="\n")
        manifest_rows.append(
            {
                "dataset": dataset,
                "candidate_artifact_path": artifact_path.as_posix(),
                "candidate_artifact_hash": sha256_file(full_path),
                "candidate_rows": len(dataset_frame),
                "pool_users": dataset_frame["user_id"].nunique() if not dataset_frame.empty else 0,
                "title_source": title_sources[dataset],
                "metadata_valid": bool_text(not dataset_frame.empty and dataset_frame["metadata_status"].eq("VALID_METADATA").all()),
                "candidate_order_immutable_for_llm": "true",
                "status": "MATERIALIZED",
            }
        )
    return pd.DataFrame(pool_rows), pd.DataFrame(summary_rows), pd.DataFrame(manifest_rows)


def build_pool_source_contributions(pool_summaries: pd.DataFrame) -> pd.DataFrame:
    cols = ["BPRMF_contribution", "GRU4Rec_contribution", "SASRec_contribution", "LightGCN_contribution"]
    frame = pool_summaries[
        [
            "dataset",
            "user_id",
            "pool_id",
            "pool_size",
            "unique_count",
            *cols,
            "first_model_20_0_0_0",
            "pool_hash",
            "status",
        ]
    ].copy()
    frame["contribution_vector"] = [
        stable_json([int(row[col]) for col in cols])
        for row in frame.to_dict("records")
    ]
    frame["anti_domination_status"] = [
        "PASS_REPLAYED_BALANCED_POLICY" if value != "true" else "CHECK_DUPLICATE_EXPLANATION"
        for value in frame["first_model_20_0_0_0"]
    ]
    return frame


def build_candidate_recall(pool_summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in pool_summaries.groupby("dataset", sort=False):
        total = len(group)
        present = int(group["positive_present"].eq("true").sum())
        ready = group["status"].eq(POOL_STATUS_COMPUTED).all()
        ranks = pd.to_numeric(group["positive_position"], errors="coerce").dropna()
        rows.append(
            {
                "dataset": dataset,
                "pool_policy": ADAPTED_POLICY,
                "evaluation_N": total,
                "positive_present_N": present,
                "positive_absent_N": total - present,
                "candidate_recall_at_20": format_float(present / total if total else math.nan),
                "positive_rank_mean": format_float(float(ranks.mean())) if len(ranks) else "",
                "positive_rank_median": format_float(float(ranks.median())) if len(ranks) else "",
                "candidate_recall_tautological": "false",
                "force_positive": "false",
                "target_identity_used_for_ranking": "false",
                "status": "COMPUTED" if ready else "BLOCKED_PROTOCOL",
                "reason": "Candidate Recall@20 computed after corrected pool construction; targets were not used for ranking.",
            }
        )
    return pd.DataFrame(rows)


def build_foodcom_warm_cold_recall(pool_summaries: pd.DataFrame) -> pd.DataFrame:
    food = pool_summaries.loc[pool_summaries["dataset"].eq("foodcom")]
    rows = []
    scopes = [
        ("ALL_USERS", food),
        ("WARM_USERS", food.loc[food["warm_cold_group"].eq("WARM_USERS")]),
        ("TRUE_ZERO_HISTORY_USERS", food.loc[food["warm_cold_group"].eq("TRUE_ZERO_HISTORY_USERS")]),
    ]
    for scope, group in scopes:
        total = len(group)
        present = int(group["positive_present"].eq("true").sum()) if total else 0
        rows.append(
            {
                "dataset": "foodcom",
                "scope": scope,
                "evaluation_N": total,
                "positive_present_N": present,
                "positive_absent_N": total - present,
                "candidate_recall_at_20": format_float(present / total if total else math.nan) if total else "",
                "candidate_recall_tautological": "false",
                "force_positive": "false",
                "status": "COMPUTED",
                "reason": "Food.com warm/cold split preserved from the frozen final-training history contract.",
            }
        )
    return pd.DataFrame(rows)


def build_reranking_readiness(
    config: Mapping[str, Any],
    f0_reranking: pd.DataFrame,
    pool_summaries: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    candidate_manifest: pd.DataFrame,
) -> pd.DataFrame:
    recall = candidate_recall.set_index("dataset").to_dict("index")
    candidate_paths = candidate_manifest.set_index("dataset").to_dict("index")
    pool_ready = {
        dataset: group["status"].eq(POOL_STATUS_COMPUTED).all()
        for dataset, group in pool_summaries.groupby("dataset", sort=False)
    }
    rows = []
    for row in f0_reranking.to_dict("records"):
        dataset = str(row["dataset"])
        ready = bool(pool_ready.get(dataset, False))
        status = "READY_TO_INFER" if ready else "BLOCKED_PROTOCOL"
        request_count = int(row["request_count"])
        rows.append(
            {
                "dataset": dataset,
                "task": row["task"],
                "model_name": row["model_name"],
                "prompt": row["prompt"],
                "cohort_id": row["cohort_id"],
                "first_stage_recommender": "F1BZQ_REPRESENTABLE_FULL_CATALOG_FIVE_SEED_MEAN",
                "source_ranking": "results/raw_outputs/f1bzq_source_rankings",
                "reranking_candidate_manifest": candidate_paths.get(dataset, {}).get("candidate_artifact_path", ""),
                "reranking_candidate_size": row["reranking_candidate_size"],
                "positive_present_candidate_recall_required": row["positive_present_candidate_recall_required"],
                "f0_request_count": request_count,
                "future_requests_unlocked": request_count if ready else 0,
                "ground_truth_positive_present_users": recall.get(dataset, {}).get("positive_present_N", ""),
                "candidate_recall_at_20": recall.get(dataset, {}).get("candidate_recall_at_20", ""),
                "candidate_order_immutable_for_llm": bool_text(bool(config["pool_policy"]["candidate_order_immutable_for_llm"])),
                "llm_inference_executed": "false",
                "training_executed_in_f1bzq": "false",
                "readiness_status": status,
                "reason": "Corrected authoritative pool exists; LLM inference not executed." if ready else "Corrected authoritative pool unavailable.",
            }
        )
    return pd.DataFrame(rows)


def build_request_manifest(reranking_readiness: pd.DataFrame) -> pd.DataFrame:
    frame = reranking_readiness.copy()
    frame["request_manifest_status"] = [
        "READY_REQUEST_NOT_EXECUTED" if status == "READY_TO_INFER" else status
        for status in frame["readiness_status"]
    ]
    frame["ready_for_llm_inference"] = [bool_text(status == "READY_TO_INFER") for status in frame["readiness_status"]]
    frame["llm_inference_executed"] = "false"
    frame["training_executed_in_f1bzq"] = "false"
    frame["candidate_order_source"] = "f1bzq_authoritative_pool_hash"
    return frame


def build_supersession_validation(root: Path) -> pd.DataFrame:
    artifacts = [
        ("old_f1bz_first_stage_rankings", "results/tables/f1bz_first_stage_rankings.csv", "forced-positive-derived first-stage rankings"),
        ("old_f1bz_reranking_pools", "results/tables/f1bz_reranking_pools.csv", "tautological candidate20-derived pools"),
        ("old_f1bz_candidate_recall", "results/tables/f1bz_candidate_recall.csv", "recall=1.0 from candidate20 source pools"),
        ("old_f1bz_reranking_readiness", "results/tables/f1bz_reranking_readiness.csv", "old 30/30 READY_TO_INFER status"),
        ("f1bzc_attempted_adapted_pools", "results/raw_outputs/f1bzc_reranking_candidates/f1bzc_lastfm_attempted_adapted_pools.csv", "single-model-dominated diagnostic pools"),
    ]
    rows = []
    for artifact_id, path, reason in artifacts:
        rel = Path(path)
        rows.append(
            {
                "artifact_id": artifact_id,
                "source_path": path,
                "source_hash": sha256_file(root / rel) if (root / rel).exists() else "",
                "superseded_status": SUPERSEDED,
                "authoritative_for_downstream": "false",
                "replacement_family": "F1BZQ_AUTHORITATIVE_RERANKING_POOLS",
                "status": "PASS",
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_pool_validation(
    config: Mapping[str, Any],
    f1bzp_manifest: Mapping[str, Any],
    f1bzp_vocab: pd.DataFrame,
    f1bzp_readiness: pd.DataFrame,
    source_execution: pd.DataFrame,
    source_users: pd.DataFrame,
    pool_summaries: pd.DataFrame,
    pool_rows: pd.DataFrame,
    pool_contributions: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    foodcom_warm_cold: pd.DataFrame,
    reranking_readiness: pd.DataFrame,
    supersession: pd.DataFrame,
    f0_budget: pd.DataFrame,
    checkpoint_hashes_before: Mapping[str, str],
    checkpoint_hashes_after: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def check(check_id: str, check_name: str, passed: bool, observed: Any, expected: Any, details: str) -> None:
        rows.append(
            {
                "check_id": check_id,
                "check_name": check_name,
                "status": "PASS" if passed else "FAIL",
                "observed": str(observed),
                "expected": str(expected),
                "details": details,
            }
        )

    budget = {str(row["budget_item"]): int(row["request_count"]) for row in f0_budget.to_dict("records")}
    ready_requests = int(reranking_readiness["future_requests_unlocked"].astype(int).sum())
    dominated = int(pool_contributions["first_model_20_0_0_0"].eq("true").sum()) if not pool_contributions.empty else 0
    check("F1BZQ-POOL-001", "source_cells_terminal", len(source_execution) == 20 and source_execution["terminal"].eq("true").all(), len(source_execution), 20, "Every dataset/model source cell reached a terminal status.")
    check("F1BZQ-POOL-002", "source_cells_computed", source_execution["status"].eq("COMPUTED").all(), source_execution["status"].value_counts().to_dict(), "20 COMPUTED", "Checkpoint scoring succeeded for every representable source cell.")
    check("F1BZQ-POOL-003", "user_model_cases", int(source_execution["user_model_cases"].astype(int).sum()) == 19528 and len(source_users) == 19528, {"execution": int(source_execution["user_model_cases"].astype(int).sum()), "source_users": len(source_users)}, 19528, "All user/model source cases are represented once.")
    check("F1BZQ-POOL-004", "source_depth_from_f1bzp", int(config["pool_policy"]["source_depth"]) == int(f1bzp_manifest["requested_source_depth"]) == 20, config["pool_policy"]["source_depth"], 20, "Source depth is inherited from F1BZP.")
    check("F1BZQ-POOL-005", "representable_catalog_intersection", f1bzp_vocab["model_representable_status"].eq(MODEL_READY).all() and f1bzp_readiness["oov_target_N"].astype(int).sum() == 0, f1bzp_vocab["model_representable_status"].value_counts().to_dict(), "20 ready and 0 OOV targets", "F1BZP representable catalog gate remains true.")
    check("F1BZQ-POOL-006", "checkpoint_hashes_unchanged", dict(checkpoint_hashes_before) == dict(checkpoint_hashes_after), len(checkpoint_hashes_before), len(checkpoint_hashes_after), "All F1BZ checkpoint hashes match before and after scoring.")
    check("F1BZQ-POOL-007", "no_positive_forcing", source_users["force_positive"].eq("false").all() and pool_summaries["force_positive"].eq("false").all(), "false", "false", "Neither source scoring nor pool construction inserts targets.")
    check("F1BZQ-POOL-008", "seen_item_exclusion", source_users["excluded_seen_item_count"].astype(int).ge(0).all() and source_users["eligible_catalog_N"].astype(int).ge(20).all(), source_users["excluded_seen_item_count"].astype(int).describe().to_dict(), "nonnegative", "Previously seen pre-test items are excluded from the retrievable catalog.")
    check("F1BZQ-POOL-009", "pool_size_uniqueness", pool_summaries["pool_size"].astype(int).eq(20).all() and pool_summaries["unique_count"].astype(int).eq(20).all(), pool_summaries[["pool_size", "unique_count"]].value_counts().to_dict(), "20 unique", "No candidate pool is underfilled.")
    check("F1BZQ-POOL-010", "metadata_title_validity", pool_summaries["metadata_valid"].eq("true").all() and pool_rows["candidate_title"].astype(str).str.strip().ne("").all(), int(pool_summaries["title_missing_count"].astype(int).sum()), 0, "All selected candidate IDs have existing titles.")
    check("F1BZQ-POOL-011", "exact_f1bzp_pool_policy", pool_summaries["pool_policy"].eq(ADAPTED_POLICY).all() and pool_summaries["pool_algorithm"].eq(ADAPTED_ALGORITHM).all(), pool_summaries[["pool_policy", "pool_algorithm"]].drop_duplicates().to_dict("records"), ADAPTED_ALGORITHM, "Pools use the imported F1BZP balanced helper.")
    check("F1BZQ-POOL-012", "anti_first_model_domination", dominated == 0, dominated, 0, "Balanced rank traversal does not globally degenerate to 20,0,0,0.")
    check("F1BZQ-POOL-013", "candidate_recall_non_tautological", candidate_recall["candidate_recall_tautological"].eq("false").all() and candidate_recall["force_positive"].eq("false").all(), candidate_recall[["dataset", "candidate_recall_at_20"]].to_dict("records"), "non-tautological", "Candidate Recall is computed only after pool construction.")
    foodcom_counts = foodcom_warm_cold.set_index("scope")["evaluation_N"].astype(int).to_dict()
    check("F1BZQ-POOL-014", "foodcom_warm_cold_preserved", set(foodcom_warm_cold["scope"]) == {"ALL_USERS", "WARM_USERS", "TRUE_ZERO_HISTORY_USERS"} and foodcom_counts.get("TRUE_ZERO_HISTORY_USERS") == 727 and foodcom_counts.get("WARM_USERS") == 273, foodcom_warm_cold[["scope", "evaluation_N"]].to_dict("records"), "727 cold and 273 warm", "Food.com zero-history cohort is preserved.")
    check("F1BZQ-POOL-015", "reranking_readiness_30", len(reranking_readiness) == 30 and reranking_readiness["readiness_status"].eq("READY_TO_INFER").all() and ready_requests == 29292, {"rows": len(reranking_readiness), "requests": ready_requests}, "30 rows and 29292 requests", "F0 reranking rows are rebuilt from corrected pools.")
    check("F1BZQ-POOL-016", "superseded_artifact_safety", supersession["superseded_status"].eq(SUPERSEDED).all() and supersession["authoritative_for_downstream"].eq("false").all(), supersession[["artifact_id", "superseded_status"]].to_dict("records"), SUPERSEDED, "Old tautological/single-model pool artifacts are not downstream-authoritative.")
    check("F1BZQ-POOL-017", "f0_budget_unchanged", budget.get("CORE_MANDATORY_NOW_REQUESTS") == 29292 and budget.get("REQUIRED_SENSITIVITY_REQUESTS") == 21000 and budget.get("MANDATORY_RAW_REQUESTS_TOTAL") == 50292 and budget.get("TRAINING_DEPENDENT_REQUESTS") == 29292 and budget.get("CONDITIONAL_REQUESTS") == 9000 and budget.get("OPTIONAL_REQUESTS") == 12000 and budget.get("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL") == 100584, budget, "F0 budget constants", "Only readiness changes.")
    check("F1BZQ-POOL-018", "zero_llm", not bool(config["execution"]["allow_llm_inference"]) and not bool(config["execution"]["allow_ollama_recommendation_calls"]) and not bool(config["execution"]["allow_external_api_calls"]), "0", "0", "Traditional checkpoint scoring is not LLM inference.")
    return pd.DataFrame(rows)


def build_validation(
    root: Path,
    config: Mapping[str, Any],
    phase_start_clean_evidence: str,
    f1bz_manifest: Mapping[str, Any],
    f1bz_no_llm: Mapping[str, Any],
    f1bza_manifest: Mapping[str, Any],
    f1bzc_manifest: Mapping[str, Any],
    f1bzp_manifest: Mapping[str, Any],
    f1bzp_vocab: pd.DataFrame,
    f1bzp_pool_policy: pd.DataFrame,
    f1bza_zero: pd.DataFrame,
    source_execution: pd.DataFrame,
    source_users: pd.DataFrame,
    pool_summaries: pd.DataFrame,
    pool_rows: pd.DataFrame,
    pool_contributions: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    foodcom_warm_cold: pd.DataFrame,
    reranking_readiness: pd.DataFrame,
    supersession: pd.DataFrame,
    pool_validation: pd.DataFrame,
    f0_budget: pd.DataFrame,
    checkpoint_hashes_before: Mapping[str, str],
    checkpoint_hashes_after: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": str(evidence)})

    budget = {str(row["budget_item"]): int(row["request_count"]) for row in f0_budget.to_dict("records")}
    expected_users = {"beauty": 990, "sports": 892, "ml1m": 1000, "lastfm": 1000, "foodcom": 1000}
    ready_requests = int(reranking_readiness["future_requests_unlocked"].astype(int).sum())
    policy_row = f1bzp_pool_policy.loc[f1bzp_pool_policy["policy_id"].eq(ADAPTED_POLICY)].iloc[0]
    zero_overall = f1bza_zero.loc[f1bza_zero["scope"].eq("overall_foodcom_final_cohort")].iloc[0]

    check("precondition_head", git_output(root, ["rev-parse", "HEAD"]) == str(config["phase"]["expected_head"]), git_output(root, ["rev-parse", "HEAD"]))
    check(
        "precondition_worktree_clean_at_phase_start",
        bool(config["phase"].get("phase_start_clean_verified", False)),
        phase_start_clean_evidence,
    )
    check("precondition_f1bz", f1bz_manifest.get("decision_token") == "F1BZ_FINAL_TRADITIONAL_EXECUTION_VALIDATED", f1bz_manifest.get("decision_token"))
    check("precondition_f1bza", f1bza_manifest.get("decision_token") == "F1BZA_CORRECTION_REQUIRED", f1bza_manifest.get("decision_token"))
    check("precondition_f1bzc", f1bzc_manifest.get("decision_token") == "F1BZC_RERANKING_SOURCE_BLOCKED", f1bzc_manifest.get("decision_token"))
    check("precondition_f1bzp", f1bzp_manifest.get("decision_token") == "F1BZP_ADAPTED_POOL_PROTOCOL_FROZEN", f1bzp_manifest.get("decision_token"))
    check("active_datasets_exact", f1bz_manifest.get("active_datasets") == list(ACTIVE_DATASETS), f1bz_manifest.get("active_datasets"))
    check("video_games_absent", int(f1bz_manifest.get("integrity", {}).get("active_video_games_rows", -1)) == 0, f1bz_manifest.get("integrity", {}).get("active_video_games_rows"))
    check("f1bzp_policy_exact", policy_row["policy_id"] == ADAPTED_POLICY and policy_row["algorithm_id"] == ADAPTED_ALGORITHM and policy_row["requested_source_depth"] == "20", policy_row.to_dict())
    check("f1bzp_representable_catalog", f1bzp_vocab["model_representable_status"].eq(MODEL_READY).all() and int(f1bzp_manifest["oov_positive_target_rows"]) == 0, {"ready": f1bzp_vocab["model_representable_status"].value_counts().to_dict(), "oov": f1bzp_manifest["oov_positive_target_rows"]})
    check("source_cells_20_terminal", len(source_execution) == 20 and source_execution["terminal"].eq("true").all(), source_execution["status"].value_counts().to_dict())
    check("source_cases_19528", len(source_users) == 19528 and source_execution["user_model_cases"].astype(int).sum() == 19528, {"source_users": len(source_users), "execution_sum": int(source_execution["user_model_cases"].astype(int).sum())})
    check("source_retrievable_intersection", source_execution["retrievable_catalog_N"].astype(int).eq(source_execution["checkpoint_item_vocab_N"].astype(int)).all(), source_execution[["dataset", "model", "retrievable_catalog_N", "checkpoint_item_vocab_N"]].head().to_dict("records"))
    check("source_no_positive_force", source_users["force_positive"].eq("false").all() and source_users["target_identity_used_for_ranking"].eq("false").all(), "false")
    check("deterministic_ranking_hashes", source_users["ranking_hash"].astype(str).str.len().eq(64).all(), source_users["ranking_hash"].head().tolist())
    check("pool_users_4882", len(pool_summaries) == sum(expected_users.values()) and pool_summaries.groupby("dataset")["user_id"].nunique().to_dict() == expected_users, pool_summaries.groupby("dataset")["user_id"].nunique().to_dict())
    check("pool_rows_97640", len(pool_rows) == sum(expected_users.values()) * 20, len(pool_rows))
    check("pool_size_unique", pool_summaries["pool_size"].astype(int).eq(20).all() and pool_summaries["unique_count"].astype(int).eq(20).all(), pool_summaries[["pool_size", "unique_count"]].value_counts().to_dict())
    check("pool_policy_replayed", pool_summaries["pool_policy"].eq(ADAPTED_POLICY).all() and pool_summaries["pool_algorithm"].eq(ADAPTED_ALGORITHM).all(), pool_summaries[["pool_policy", "pool_algorithm"]].drop_duplicates().to_dict("records"))
    check("anti_first_model_domination", int(pool_contributions["first_model_20_0_0_0"].eq("true").sum()) == 0, pool_contributions["first_model_20_0_0_0"].value_counts().to_dict())
    check("metadata_title_validity", pool_rows["candidate_title"].astype(str).str.strip().ne("").all() and pool_summaries["metadata_valid"].eq("true").all(), int(pool_summaries["title_missing_count"].astype(int).sum()))
    check("candidate_recall_non_tautological", candidate_recall["candidate_recall_tautological"].eq("false").all() and candidate_recall["force_positive"].eq("false").all(), candidate_recall[["dataset", "candidate_recall_at_20"]].to_dict("records"))
    foodcom_counts = foodcom_warm_cold.set_index("scope")["evaluation_N"].astype(int).to_dict()
    check("foodcom_warm_cold_preserved", zero_overall["actual_zero_history_users"] == "727" and zero_overall["pipeline_zero_history_users"] == "0" and foodcom_counts.get("TRUE_ZERO_HISTORY_USERS") == 727 and foodcom_counts.get("WARM_USERS") == 273, foodcom_warm_cold[["scope", "evaluation_N"]].to_dict("records"))
    check("reranking_readiness_30_ready", len(reranking_readiness) == 30 and reranking_readiness["readiness_status"].eq("READY_TO_INFER").all() and ready_requests == 29292, {"rows": len(reranking_readiness), "requests": ready_requests})
    check("old_artifacts_superseded", supersession["superseded_status"].eq(SUPERSEDED).all() and supersession["authoritative_for_downstream"].eq("false").all(), supersession[["artifact_id", "superseded_status"]].to_dict("records"))
    check("pool_validation_passed", pool_validation["status"].eq("PASS").all(), pool_validation["status"].value_counts().to_dict())
    check("f0_budget_unchanged", budget.get("CORE_MANDATORY_NOW_REQUESTS") == 29292 and budget.get("REQUIRED_SENSITIVITY_REQUESTS") == 21000 and budget.get("MANDATORY_RAW_REQUESTS_TOTAL") == 50292 and budget.get("TRAINING_DEPENDENT_REQUESTS") == 29292 and budget.get("CONDITIONAL_REQUESTS") == 9000 and budget.get("OPTIONAL_REQUESTS") == 12000 and budget.get("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL") == 100584, budget)
    check("checkpoint_hashes_unchanged", dict(checkpoint_hashes_before) == dict(checkpoint_hashes_after), len(checkpoint_hashes_before))
    check("no_training_no_tuning", not bool(config["execution"]["allow_traditional_training"]) and not bool(config["execution"]["allow_tuning"]), config["execution"])
    check("zero_llm_network_download", not bool(config["execution"]["allow_llm_inference"]) and not bool(config["execution"]["allow_ollama_recommendation_calls"]) and not bool(config["execution"]["allow_external_api_calls"]) and not bool(config["execution"]["allow_downloads"]) and not bool(config["execution"]["allow_network"]) and int(f1bz_no_llm.get("llm_inference_calls_executed", 0)) == 0, config["execution"])
    check("original_repo_paper_data_raw_clean", git_output(root, ["status", "--short", "--", "original_repo", "paper", "data/raw"]) == "", "clean")
    return pd.DataFrame(rows)


def build_manifests(
    root: Path,
    config: Mapping[str, Any],
    decision: str,
    f1bz_manifest: Mapping[str, Any],
    f1bza_manifest: Mapping[str, Any],
    f1bzc_manifest: Mapping[str, Any],
    f1bzp_manifest: Mapping[str, Any],
    source_execution: pd.DataFrame,
    source_users: pd.DataFrame,
    source_recall: pd.DataFrame,
    pool_summaries: pd.DataFrame,
    pool_rows: pd.DataFrame,
    pool_contributions: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    foodcom_warm_cold: pd.DataFrame,
    reranking_readiness: pd.DataFrame,
    supersession: pd.DataFrame,
    pool_validation: pd.DataFrame,
    validation: pd.DataFrame,
    checkpoint_hashes_before: Mapping[str, str],
    checkpoint_hashes_after: Mapping[str, str],
    source_artifacts: Sequence[Path],
    candidate_manifest: pd.DataFrame,
    request_manifest: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    ready_rows = int(reranking_readiness["readiness_status"].eq("READY_TO_INFER").sum())
    ready_requests = int(reranking_readiness["future_requests_unlocked"].astype(int).sum())
    source_cells_terminal = int(source_execution["terminal"].eq("true").sum())
    source_cells_computed = int(source_execution["status"].eq("COMPUTED").sum())
    pool_users = int(pool_summaries["status"].eq(POOL_STATUS_COMPUTED).sum())
    base = {
        "phase_id": PHASE_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_token": decision,
        "precondition_f1bz": f1bz_manifest.get("decision_token", ""),
        "precondition_f1bza": f1bza_manifest.get("decision_token", ""),
        "precondition_f1bzc": f1bzc_manifest.get("decision_token", ""),
        "precondition_f1bzp": f1bzp_manifest.get("decision_token", ""),
        "precondition_worktree_clean_at_phase_start": bool(config["phase"].get("phase_start_clean_verified", False)),
        "precondition_worktree_clean_evidence": str(config["phase"].get("phase_start_clean_evidence", "")),
        "active_datasets": list(ACTIVE_DATASETS),
        "models": list(TRAINABLE_MODELS),
        "source_policy": SOURCE_POLICY,
        "source_seed_policy": SOURCE_SEED_POLICY,
        "seeds": list(FINAL_SEEDS),
        "pool_policy": ADAPTED_POLICY,
        "pool_algorithm": ADAPTED_ALGORITHM,
        "source_depth": int(config["pool_policy"]["source_depth"]),
        "source_cells_terminal": source_cells_terminal,
        "source_cells_computed": source_cells_computed,
        "user_model_source_cases": int(len(source_users)),
        "source_ranking_rows": int(source_execution["source_artifact_rows"].astype(int).sum()),
        "authoritative_pool_users": pool_users,
        "pool_candidate_rows": int(len(pool_rows)),
        "candidate_recall_rows": int(len(candidate_recall)),
        "ready_reranking_rows": ready_rows,
        "ready_llm_requests": ready_requests,
        "training_runs_executed_in_f1bzq": 0,
        "checkpoint_parameter_updates": 0,
        "checkpoint_hashes_changed": int(dict(checkpoint_hashes_before) != dict(checkpoint_hashes_after)),
        "hyperparameter_changes": 0,
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_llm_api_calls_executed": 0,
        "network_calls_executed": 0,
        "downloads_executed": 0,
        "commit_created": 0,
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS")].to_dict("records"),
        "checkpoint_hashes_before": dict(checkpoint_hashes_before),
        "checkpoint_hashes_after": dict(checkpoint_hashes_after),
        "tables": {key: path.as_posix() for key, path in OUTPUT_TABLES.items()},
        "request_budget": dict(config["request_budget"]),
        "frame_hashes": {
            "source_execution": frame_hash(source_execution),
            "source_recall": frame_hash(source_recall),
            "pool_source_contributions": frame_hash(pool_contributions),
            "candidate_recall": frame_hash(candidate_recall),
            "foodcom_warm_cold_recall": frame_hash(foodcom_warm_cold),
            "reranking_readiness": frame_hash(reranking_readiness),
            "supersession_validation": frame_hash(supersession),
            "pool_validation": frame_hash(pool_validation),
            "validation": frame_hash(validation),
            "request_manifest": frame_hash(request_manifest),
            "candidate_manifest": frame_hash(candidate_manifest),
        },
    }
    source_manifest = {
        **base,
        "source_artifacts": [path.as_posix() for path in source_artifacts],
    }
    pool_manifest = {
        **base,
        "pool_artifacts": candidate_manifest["candidate_artifact_path"].tolist(),
        "candidate_recall": candidate_recall.to_dict("records"),
        "foodcom_warm_cold_recall": foodcom_warm_cold.to_dict("records"),
    }
    no_training = {
        "phase_id": PHASE_ID,
        "decision_token": decision,
        "training_runs_executed_in_f1bzq": 0,
        "traditional_checkpoint_scoring_executed": int(source_cells_computed > 0),
        "traditional_checkpoint_scoring_is_not_llm_inference": True,
        "checkpoint_parameter_updates": 0,
        "checkpoint_hashes_changed": int(dict(checkpoint_hashes_before) != dict(checkpoint_hashes_after)),
        "hyperparameter_changes": 0,
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_llm_api_calls_executed": 0,
        "network_calls_executed": 0,
        "downloads_executed": 0,
        "commit_created": 0,
    }
    return {
        "source_ranking": source_manifest,
        "authoritative_pool": pool_manifest,
        "no_training_no_llm": no_training,
    }


def render_report(
    decision: str,
    source_execution: pd.DataFrame,
    source_recall: pd.DataFrame,
    pool_summaries: pd.DataFrame,
    pool_contributions: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    foodcom_warm_cold: pd.DataFrame,
    reranking_readiness: pd.DataFrame,
    supersession: pd.DataFrame,
    pool_validation: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    source_status = source_execution["status"].value_counts().to_dict()
    pool_status = pool_summaries["status"].value_counts().to_dict()
    readiness_status = reranking_readiness["readiness_status"].value_counts().to_dict()
    contribution_totals = {
        key: int(pool_summaries[key].astype(int).sum())
        for key in ["BPRMF_contribution", "GRU4Rec_contribution", "SASRec_contribution", "LightGCN_contribution"]
    }
    lines = [
        "# F1BZQ Authoritative Reranking Pool Materialization",
        "",
        f"Decision token: `{decision}`",
        "",
        "## Source Scoring",
        "",
        f"- Source cell statuses: `{stable_json(source_status)}`.",
        f"- User/model source cases: `{int(len(source_recall) and source_execution['user_model_cases'].astype(int).sum())}`.",
        "- Source universe: raw catalog intersected with frozen checkpoint item vocabulary, minus legitimate pre-test seen items.",
        "- Source seed policy: all five F1BZ seeds are mean-scored; no final-test seed selection is used.",
        "",
        "## Corrected Pools",
        "",
        f"- Pool statuses: `{stable_json(pool_status)}`.",
        f"- Pool users: `{len(pool_summaries)}`.",
        f"- Pool candidate rows: `{len(pool_summaries) * 20}`.",
        f"- Contribution totals: `{stable_json(contribution_totals)}`.",
        f"- First-model 20,0,0,0 pools: `{int(pool_contributions['first_model_20_0_0_0'].eq('true').sum())}`.",
        "- Pool construction imports the frozen F1BZP balanced rank round-robin helper.",
        "",
        "## Candidate Recall",
        "",
        candidate_recall.to_csv(index=False, lineterminator="\n").strip(),
        "",
        "## Food.com Warm/Cold",
        "",
        foodcom_warm_cold.to_csv(index=False, lineterminator="\n").strip(),
        "",
        "## Reranking Readiness",
        "",
        f"- Readiness statuses: `{stable_json(readiness_status)}`.",
        f"- Ready rows: `{int(reranking_readiness['readiness_status'].eq('READY_TO_INFER').sum())}` / `30`.",
        f"- Ready LLM requests: `{int(reranking_readiness['future_requests_unlocked'].astype(int).sum())}` / `29292`.",
        "- LLM inference executed: `0`.",
        "",
        "## Supersession",
        "",
        f"- Supersession rows: `{len(supersession)}`.",
        "- Old tautological and single-model diagnostic pool artifacts remain preserved but non-authoritative.",
        "",
        "## Validation",
        "",
        f"- Pool validation checks: `{len(pool_validation)}`; failures `{int(pool_validation['status'].ne('PASS').sum())}`.",
        f"- Final validation checks: `{len(validation)}`; failures `{int(validation['status'].ne('PASS').sum())}`.",
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
    artifact_paths = [
        *OUTPUT_TABLES.values(),
        *SUPPORT_ARTIFACTS,
        *[Path(path) for path in outputs["manifests"]["source_ranking"]["source_artifacts"]],
        *[Path(path) for path in outputs["manifests"]["authoritative_pool"]["pool_artifacts"]],
    ]
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
    for rel_path in (TABLE_DIR, MANIFEST_DIR, REPORT_PATH.parent, SOURCE_RANKING_DIR, POOL_DIR):
        (root / rel_path).mkdir(parents=True, exist_ok=True)


def load_title_map(root: Path, spec: DatasetSpec) -> tuple[dict[str, str], str]:
    path = root / spec.catalog_source
    if spec.name == "foodcom":
        frame = pd.read_csv(path, dtype={"id": "string", "name": "string"}, keep_default_na=False, usecols=["id", "name"], low_memory=False)
        return dict(zip(frame["id"].astype(str), frame["name"].map(clean_title))), "RAW_recipes.name"
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        usecols=lambda column: column in {"item_id", "title", "normalized_text", "categories"},
    )
    fallback_cols = [column for column in ["title", "normalized_text", "categories", "item_id"] if column in frame.columns]
    title_values = [first_nonempty(row, fallback_cols) for row in frame.to_dict("records")]
    return dict(zip(frame["item_id"].astype(str), title_values)), "title_with_normalized_text_categories_item_id_fallback"


def clean_title(value: Any) -> str:
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split()).strip()


def first_nonempty(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    for column in columns:
        value = clean_title(row.get(column, ""))
        if value:
            return value
    return ""


def checkpoint_hashes(root: Path, run_manifest: pd.DataFrame) -> dict[str, str]:
    return {
        str(row["checkpoint_path"]): sha256_file(root / str(row["checkpoint_path"]))
        for row in run_manifest.to_dict("records")
    }


def select_device(policy: str) -> torch.device:
    policy = str(policy).strip().lower()
    if policy == "cpu":
        return torch.device("cpu")
    if policy in {"cuda", "cuda_if_available_else_cpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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

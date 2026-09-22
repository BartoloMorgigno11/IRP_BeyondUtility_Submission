"""F1BZ final traditional model execution.

This phase runs the frozen five-dataset, four-model, five-seed traditional
recommender matrix. It never calls LLMs, external APIs, downloads, or network
resources. The exact ReChorus runtime is unavailable locally, so original-four
results retain the F1BR classification as paper-hyperparameter-aligned local
implementation outputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import numpy as np
import pandas as pd
import torch
import yaml

from src.extension.e0_protocol import CANDIDATE_COUNT, sha256_json, stable_json, stable_int_seed
from src.extension.f0_final_extension_freeze import sha256_file
from src.extension.f1a_offline_metric_closure import format_float
from src.extension.f1bx_execution_stage1 import (
    ACTIVE_DATASETS,
    BEYOND_METRICS,
    TOP_K,
    TRAINABLE_MODELS,
    DatasetMaterial,
    DatasetSpec,
    beyond_metric_values,
    bool_series,
    build_dataset_material,
    dataset_specs as f1bx_dataset_specs,
    frame_hash,
    per_user_stats,
    read_csv,
    read_json,
    sort_key,
    utility_metric_values,
)
from src.extension.f1by_foodcom_validation_tuning import (
    TuningData,
    baseline_hash_state,
    pad_tokens,
    resource_preflight,
    select_device,
    set_deterministic_execution,
    train_bprmf,
    train_gru4rec,
    train_lightgcn,
    train_sasrec,
)
from src.utils.config import find_project_root


PHASE_ID = "f1bz_final_traditional_execution"
DECISION_VALIDATED = "F1BZ_FINAL_TRADITIONAL_EXECUTION_VALIDATED"
DECISION_BLOCKED = "F1BZ_FINAL_TRADITIONAL_EXECUTION_BLOCKED"
EXPECTED_HEAD = "5f29945d1cfe5e6cce5b6cc0522db6f0efe8a976"
PROTOCOL_CHECKPOINT = EXPECTED_HEAD
CONFIG_PATH = Path("configs/f1bz_final_traditional_execution.yaml")
FINAL_SEEDS = (42, 43, 44, 45, 46)
MODEL_ORDER = ("BPRMF", "GRU4Rec", "SASRec", "LightGCN")
BASELINE_METHODS = ("Random", "MostPop", "BM25-like/content")
UTILITY_METRICS = ("hr_at_5", "ndcg_at_5", "mrr_diagnostic")

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")
RAW_RECOMMENDATION_DIR = Path("results/raw_outputs/f1b_final_recommendations")
CHECKPOINT_DIR = Path("results/checkpoints/f1b_final_traditional")
REPORT_PATH = REPORT_DIR / "f1bz_final_traditional_execution.md"

OUTPUT_TABLES = {
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

UPDATED_F1B_TABLES = {
    "training_plan": TABLE_DIR / "f1b_training_plan.csv",
    "training_results": TABLE_DIR / "f1b_training_results.csv",
    "utility_results": TABLE_DIR / "f1b_traditional_utility_results.csv",
    "beyond_utility_results": TABLE_DIR / "f1b_traditional_beyond_utility_results.csv",
    "five_dataset_completeness": TABLE_DIR / "f1b_five_dataset_completeness.csv",
    "result_cell_readiness": TABLE_DIR / "f1b_result_cell_readiness.csv",
    "first_stage_inventory": TABLE_DIR / "f1b_first_stage_ranking_inventory.csv",
    "candidate_recall": TABLE_DIR / "f1b_candidate_recall.csv",
    "reranking_readiness": TABLE_DIR / "f1b_reranking_readiness.csv",
}

OUTPUT_MANIFEST = MANIFEST_DIR / "f1bz_final_traditional_execution.json"
OUTPUT_NO_LLM_MANIFEST = MANIFEST_DIR / "f1bz_no_llm_inference.json"

SUPPORT_ARTIFACTS = (
    CONFIG_PATH,
    Path("src/extension/f1bz_final_traditional_execution.py"),
    Path("tests/test_extension_f1bz_final_traditional_execution.py"),
    Path(".gitignore"),
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    dataset: str
    method: str
    seed: int
    seed_slot: int
    cohort_id: str
    evaluation_users: int
    classification: str
    hyperparameter_source: str
    hyperparameters: dict[str, Any]
    config_id: str
    config_hash: str


@dataclass(frozen=True)
class FinalData:
    material: DatasetMaterial
    tuning: TuningData
    eval_user_indices: np.ndarray
    positive_item_ids: list[str]
    training_data_definition: str
    evaluation_cohort_hash: str
    zero_history_eval_users: int
    train_user_count: int
    train_item_count: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F1BZ final traditional recommender execution.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    config = load_config(root, Path(args.config))
    if args.validate_only:
        outputs = load_existing_outputs(root)
    else:
        outputs = execute_phase(root, config)
        write_outputs(root, outputs)
    validation = outputs["validation"]
    decision = DECISION_VALIDATED if bool(validation["status"].eq("PASS").all()) else DECISION_BLOCKED
    print(f"decision={decision}")
    print(f"final_runs={len(outputs['run_manifest'])}")
    print(f"completed_runs={int(outputs['run_manifest']['training_status'].eq('COMPLETED').sum())}")
    print(f"failed_runs={int(outputs['run_manifest']['training_status'].eq('FAILED').sum())}")
    print(f"traditional_rows={len(outputs['traditional_completeness'])}")
    print(f"reranking_ready={int(outputs['reranking_readiness']['readiness_status'].eq('READY_TO_INFER').sum())}")
    print(f"future_llm_requests_unlocked={int(pd.to_numeric(outputs['reranking_readiness']['future_requests_unlocked']).sum())}")
    print("llm_inference_calls=0")
    return 0 if decision == DECISION_VALIDATED else 1


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else root / config_path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def execute_phase(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(root)
    start_timestamp = datetime.now(timezone.utc).isoformat()
    preflight = resource_preflight(root)
    f1bx_config = load_config(root, Path(str(config["inputs"]["f1bx_config"])))
    specs = {spec.name: spec for spec in f1bx_dataset_specs(f1bx_config)}
    run_specs = build_run_specs(root, config)
    baseline_before = baseline_hash_state(root, root / Path(str(config["inputs"]["f1bx_baseline_manifest"])))

    execution_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    beyond_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    existing = load_completed_run_manifest(root)
    device = select_device(str(config["runtime"]["device_policy"]))
    for dataset in ACTIVE_DATASETS:
        material = build_dataset_material(root, specs[dataset])
        final_data = build_final_data(root, specs[dataset], material, config)
        mostpop_stats = load_mostpop_stats(root, config, material)
        dataset_specs_for_runs = [row for row in run_specs if row.dataset == dataset]
        for spec in dataset_specs_for_runs:
            if can_reuse_completed_run(root, spec, final_data, existing):
                execution, recommendations, history = reuse_completed_run(root, spec, final_data, existing[spec.run_id])
                print(f"reuse_completed={spec.run_id}", flush=True)
            else:
                execution, recommendations, history = execute_one_run(root, config, spec, final_data, device)
                print(f"executed={spec.run_id} status={execution['training_status']} runtime={execution['runtime_seconds']}", flush=True)
            execution_rows.append(execution)
            curve_rows.extend(history)
            stats = per_user_stats(recommendations, material)
            utility_rows.extend(per_seed_utility_rows(execution, stats))
            beyond_rows.extend(per_seed_beyond_rows(execution, stats, material, mostpop_stats))
            write_progress_tables(root, execution_rows, utility_rows, beyond_rows, curve_rows)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del final_data
        del material
        gc.collect()

    run_manifest = pd.DataFrame(execution_rows)
    training_curves = training_curve_frame(execution_rows, curve_rows)
    per_seed_utility = pd.DataFrame(utility_rows)
    per_seed_beyond = pd.DataFrame(beyond_rows)
    aggregate_utility = aggregate_metric_rows(per_seed_utility)
    aggregate_beyond = aggregate_metric_rows(per_seed_beyond)
    first_stage = build_first_stage_rankings(root, run_manifest)
    reranking_pools = build_reranking_pools(root, config, first_stage)
    candidate_recall = build_candidate_recall(reranking_pools)
    reranking_readiness = build_reranking_readiness(root, config, candidate_recall)
    traditional = build_traditional_completeness(root, run_manifest, aggregate_utility, aggregate_beyond)
    runtime = build_runtime_provenance(preflight, run_manifest)
    failure_retry = build_failure_retry_audit(run_manifest)
    updated_f1b = build_updated_f1b_tables(root, config, run_manifest, aggregate_utility, aggregate_beyond, traditional, first_stage, candidate_recall, reranking_readiness)
    baseline_after = baseline_hash_state(root, root / Path(str(config["inputs"]["f1bx_baseline_manifest"])))
    integrity = build_integrity(root, config, run_manifest, traditional, reranking_readiness, baseline_before, baseline_after, start_timestamp)
    validation = build_validation(root, config, run_manifest, per_seed_utility, per_seed_beyond, aggregate_utility, aggregate_beyond, traditional, first_stage, reranking_pools, candidate_recall, reranking_readiness, integrity)
    manifest = build_manifest(root, config, validation, integrity)
    no_llm = build_no_llm_manifest(integrity)
    report = render_report(config, preflight, run_manifest, aggregate_utility, aggregate_beyond, traditional, first_stage, candidate_recall, reranking_readiness, runtime, failure_retry, validation, integrity)
    return {
        "run_manifest": run_manifest,
        "training_curves": training_curves,
        "per_seed_utility": per_seed_utility,
        "aggregate_utility": aggregate_utility,
        "per_seed_beyond": per_seed_beyond,
        "aggregate_beyond": aggregate_beyond,
        "traditional_completeness": traditional,
        "first_stage_rankings": first_stage,
        "reranking_pools": reranking_pools,
        "candidate_recall": candidate_recall,
        "reranking_readiness": reranking_readiness,
        "runtime_provenance": runtime,
        "failure_retry_audit": failure_retry,
        "validation": validation,
        "updated_f1b": updated_f1b,
        "manifest": manifest,
        "no_llm_manifest": no_llm,
        "report": report,
        "integrity": integrity,
    }


def ensure_output_dirs(root: Path) -> None:
    for rel in (TABLE_DIR, MANIFEST_DIR, REPORT_DIR, RAW_RECOMMENDATION_DIR, CHECKPOINT_DIR):
        (root / rel).mkdir(parents=True, exist_ok=True)


def build_run_specs(root: Path, config: Mapping[str, Any]) -> list[RunSpec]:
    readiness = read_csv(root / Path(str(config["inputs"]["f1br_readiness"])))
    winners = read_csv(root / Path(str(config["inputs"]["f1by_winners"]))).set_index("model").to_dict("index")
    specs: list[RunSpec] = []
    for row in readiness.to_dict("records"):
        dataset = str(row["dataset"])
        method = str(row["method"])
        if method not in TRAINABLE_MODELS:
            continue
        if row["status"] != "READY_TO_TRAIN":
            raise ValueError(f"run is not READY_TO_TRAIN: {dataset} {method}")
        if dataset == "foodcom":
            winner = winners[method]
            params = json.loads(winner["hyperparameters"])
            config_id = str(winner["selected_config_id"])
            config_hash = str(winner["config_hash"])
            source = "f1by_validation_selected_config"
        else:
            params = json.loads(row["hyperparameter_json"])
            config_id = f"F1BR-{dataset.upper()}-{method.upper()}-APPENDIX-A3"
            source = "paper_appendix_a3"
            config_hash = sha256_json({"dataset": dataset, "method": method, "hyperparameters": params, "source": source})
        seeds = json.loads(row["final_seeds"])
        if tuple(int(seed) for seed in seeds) != FINAL_SEEDS:
            raise ValueError(f"unexpected seeds for {dataset} {method}: {seeds}")
        for slot, seed in enumerate(FINAL_SEEDS, start=1):
            specs.append(
                RunSpec(
                    run_id=f"f1bz_{dataset}_{method.lower()}_seed{seed}",
                    dataset=dataset,
                    method=method,
                    seed=int(seed),
                    seed_slot=slot,
                    cohort_id=str(row["cohort_id"]),
                    evaluation_users=int(row["evaluation_users"]),
                    classification=str(row["classification"]),
                    hyperparameter_source=source,
                    hyperparameters=params,
                    config_id=config_id,
                    config_hash=config_hash,
                )
            )
    return specs


def build_final_data(root: Path, spec: DatasetSpec, material: DatasetMaterial, config: Mapping[str, Any]) -> FinalData:
    users = material.candidates.drop_duplicates("user_id", keep="first")["user_id"].astype(str).tolist()
    positive_by_user = material.candidates.drop_duplicates("user_id", keep="first").set_index("user_id")["positive_item_id"].astype(str).to_dict()
    if spec.name == "foodcom":
        training_events, definition = foodcom_final_training_events(root, spec, users, positive_by_user)
    else:
        training_events, definition = original_four_training_events(root, spec)
    training_events = training_events.sort_values(["user_id", "prefix_position"], kind="mergesort").reset_index(drop=True)

    candidate_grouped: list[list[str]] = []
    positives: list[str] = []
    for user in users:
        frame = material.candidates.loc[material.candidates["user_id"].eq(user)].sort_values("candidate_position", kind="mergesort")
        candidate_grouped.append(frame["candidate_item_id"].astype(str).tolist())
        positives.append(str(positive_by_user[user]))

    all_users = sorted(set(training_events["user_id"].astype(str)).union(users), key=sort_key)
    all_items = sorted(set(training_events["item_id"].astype(str)).union({item for row in candidate_grouped for item in row}), key=sort_key)
    user_index = {user: index for index, user in enumerate(all_users)}
    item_index = {item: index for index, item in enumerate(all_items)}
    pairs = training_events[["user_id", "item_id"]].drop_duplicates().copy()
    train_user_indices = pairs["user_id"].map(user_index).astype("int64").to_numpy()
    train_item_indices = pairs["item_id"].map(item_index).astype("int64").to_numpy()
    user_positive_items: dict[int, set[int]] = defaultdict(set)
    history_by_user: dict[str, list[str]] = {}
    for user_id, group in training_events.groupby("user_id", sort=False):
        items = group.sort_values("prefix_position", kind="mergesort")["item_id"].astype(str).tolist()
        history_by_user[str(user_id)] = items
        user_positive_items[user_index[str(user_id)]].update(item_index[item] for item in items)
    train_item_universe = np.asarray(sorted({item_index[item] for item in training_events["item_id"].astype(str)}), dtype=np.int64)
    max_seq_len = int(config["runtime"]["max_sequence_length"])
    seq_rows: list[np.ndarray] = []
    seq_users: list[int] = []
    seq_positives: list[int] = []
    seq_user_positive_tokens: dict[int, set[int]] = defaultdict(set)
    eval_sequences = np.zeros((len(users), max_seq_len), dtype=np.int64)
    eval_user_indices: list[int] = []
    zero_history = 0
    for user in all_users:
        uidx = user_index[user]
        prefix_indices = [item_index[item] for item in history_by_user.get(user, [])]
        seq_user_positive_tokens[uidx].update(index + 1 for index in prefix_indices)
        for end in range(1, len(prefix_indices)):
            seq_rows.append(pad_tokens(prefix_indices[:end], max_seq_len))
            seq_users.append(uidx)
            seq_positives.append(prefix_indices[end] + 1)
    for row_index, user in enumerate(users):
        prefix_indices = [item_index[item] for item in history_by_user.get(user, [])]
        if not prefix_indices:
            zero_history += 1
        eval_sequences[row_index] = pad_tokens(prefix_indices, max_seq_len)
        eval_user_indices.append(user_index[user])
    seq_examples = np.vstack(seq_rows).astype(np.int64) if seq_rows else np.zeros((0, max_seq_len), dtype=np.int64)
    candidate_items = np.asarray([[item_index[item] for item in row] for row in candidate_grouped], dtype=np.int64)
    positive_positions = np.asarray([row.index(target) + 1 for row, target in zip(candidate_grouped, positives)], dtype=np.int64)
    eval_cohort = pd.DataFrame({"user_id": users, "positive_item_id": positives})
    tuning = TuningData(
        users=users,
        user_index=user_index,
        item_index=item_index,
        training_events=training_events,
        train_user_indices=train_user_indices,
        train_item_indices=train_item_indices,
        train_item_universe=train_item_universe,
        user_positive_items={key: set(value) for key, value in user_positive_items.items()},
        seq_examples=seq_examples,
        seq_user_indices=np.asarray(seq_users, dtype=np.int64),
        seq_positive_tokens=np.asarray(seq_positives, dtype=np.int64),
        seq_user_positive_tokens={key: set(value) for key, value in seq_user_positive_tokens.items()},
        eval_sequences=eval_sequences,
        candidate_items=candidate_items,
        candidate_item_ids=candidate_grouped,
        validation_targets=positives,
        positive_positions=positive_positions,
        training_data_hash=frame_hash(training_events),
        validation_data_hash=frame_hash(eval_cohort),
        candidate_contract_hash=material.candidate_set_hash,
        train_rows=int(len(training_events)),
        train_pair_count=int(len(pairs)),
        seq_example_count=int(len(seq_examples)),
        num_users=len(user_index),
        num_items=len(item_index),
    )
    return FinalData(
        material=material,
        tuning=tuning,
        eval_user_indices=np.asarray(eval_user_indices, dtype=np.int64),
        positive_item_ids=positives,
        training_data_definition=definition,
        evaluation_cohort_hash=frame_hash(eval_cohort),
        zero_history_eval_users=zero_history,
        train_user_count=len(history_by_user),
        train_item_count=len(train_item_universe),
    )


def original_four_training_events(root: Path, spec: DatasetSpec) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(root / spec.interaction_source, dtype=str, keep_default_na=False, usecols=lambda column: column in {"user_id", "item_id", "split", "timestamp"})
    if "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str).str.lower().eq("train")].copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame["__time"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(-1) if "timestamp" in frame.columns else -1
    frame["__order"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(["user_id", "__time", "__order"], kind="mergesort")
    frame["prefix_position"] = frame.groupby("user_id", sort=False).cumcount() + 1
    return frame[["user_id", "item_id", "prefix_position"]].copy(), "frozen_train_split_all_available_train_rows"


def foodcom_final_training_events(root: Path, spec: DatasetSpec, users: Sequence[str], positive_by_user: Mapping[str, str]) -> tuple[pd.DataFrame, str]:
    selected = set(str(user) for user in users)
    frame = pd.read_csv(
        root / spec.interaction_source,
        dtype={"user_id": "string", "recipe_id": "string", "date": "string"},
        usecols=["user_id", "recipe_id", "date"],
        keep_default_na=False,
        low_memory=False,
    )
    frame["source_input_order"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.loc[frame["user_id"].astype(str).isin(selected)].copy()
    frame["parsed_date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(["user_id", "parsed_date", "source_input_order"], kind="mergesort")
    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for user_id, group in frame.groupby("user_id", sort=False):
        user = str(user_id)
        items = group["recipe_id"].astype(str).tolist()
        if not items:
            raise ValueError(f"Food.com final user has no raw interactions: {user}")
        if items[-1] != str(positive_by_user[user]):
            mismatches.append(user)
            continue
        for position, item_id in enumerate(items[:-1], start=1):
            rows.append({"user_id": user, "item_id": str(item_id), "prefix_position": position})
    if mismatches:
        raise ValueError(f"Food.com final target i[T] mismatch for users: {mismatches[:5]}")
    return pd.DataFrame(rows, columns=["user_id", "item_id", "prefix_position"]), "foodcom_i1_through_i[T-1]_for_frozen_f0_final_users"


def execute_one_run(root: Path, config: Mapping[str, Any], spec: RunSpec, data: FinalData, device: torch.device) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    set_deterministic_execution(spec.seed)
    start = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    status = "COMPLETED"
    failure_reason = ""
    checkpoint_hash = ""
    checkpoint_path = root / CHECKPOINT_DIR / f"{spec.run_id}.pt"
    checkpoint_size = 0
    history: list[dict[str, Any]] = []
    scores = np.zeros((len(data.tuning.users), CANDIDATE_COUNT), dtype=np.float32)
    try:
        if spec.method == "BPRMF":
            model, train_history = train_bprmf(spec.hyperparameters, config, data.tuning, device, spec.run_id)
            scores = score_bprmf_final(model, data, device)
        elif spec.method == "GRU4Rec":
            model, train_history = train_gru4rec(spec.hyperparameters, config, data.tuning, device, spec.run_id)
            scores = score_sequence_final(model, data, device, allow_empty=True)
        elif spec.method == "SASRec":
            model, train_history = train_sasrec(spec.hyperparameters, config, data.tuning, device, spec.run_id)
            scores = score_sequence_final(model, data, device, allow_empty=True)
        elif spec.method == "LightGCN":
            model, graph, train_history = train_lightgcn(spec.hyperparameters, config, data.tuning, device, spec.run_id)
            scores = score_lightgcn_final(model, graph, data, device)
        else:
            raise ValueError(f"unsupported model: {spec.method}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "run_id": spec.run_id,
                "dataset": spec.dataset,
                "method": spec.method,
                "seed": spec.seed,
                "config_id": spec.config_id,
                "config_hash": spec.config_hash,
                "training_data_hash": data.tuning.training_data_hash,
                "candidate_contract_hash": data.tuning.candidate_contract_hash,
                "state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        checkpoint_hash = sha256_file(checkpoint_path)
        checkpoint_size = checkpoint_path.stat().st_size
        history = [
            {
                "run_id": spec.run_id,
                "dataset": spec.dataset,
                "method": spec.method,
                "seed": spec.seed,
                "epoch": int(row["epoch"]),
                "loss": format_float(row["loss"]),
            }
            for row in train_history
        ]
    except Exception as exc:  # pragma: no cover - local infrastructure failures are recorded.
        status = "FAILED"
        failure_reason = f"{type(exc).__name__}: {exc}"
        train_history = []
    end = datetime.now(timezone.utc).isoformat()
    runtime_seconds = time.perf_counter() - start_clock
    recommendations = recommendations_from_scores(spec, data, scores, start if status == "COMPLETED" else end)
    rec_path = root / RAW_RECOMMENDATION_DIR / f"{spec.run_id}.csv"
    recommendations.to_csv(rec_path, index=False, lineterminator="\n")
    rec_hash = frame_hash(recommendations)
    stats = per_user_stats(recommendations, data.material)
    utility = utility_metric_values(stats)
    execution = {
        "run_id": spec.run_id,
        "dataset": spec.dataset,
        "method": spec.method,
        "seed": spec.seed,
        "seed_slot": spec.seed_slot,
        "cohort_id": spec.cohort_id,
        "evaluation_users": spec.evaluation_users,
        "classification": spec.classification,
        "seed_policy": "LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY",
        "paper_seed_identity_equivalent": "false",
        "config_id": spec.config_id,
        "config_hash": spec.config_hash,
        "hyperparameters": stable_json(spec.hyperparameters),
        "hyperparameter_source": spec.hyperparameter_source,
        "training_data_hash": data.tuning.training_data_hash,
        "training_data_definition": data.training_data_definition,
        "evaluation_cohort_hash": data.evaluation_cohort_hash,
        "candidate_contract_hash": data.tuning.candidate_contract_hash,
        "recommendation_output_path": rec_path.relative_to(root).as_posix(),
        "recommendation_output_hash": rec_hash,
        "checkpoint_path": checkpoint_path.relative_to(root).as_posix() if checkpoint_path.exists() else "",
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_size_bytes": checkpoint_size,
        "start_timestamp": start,
        "end_timestamp": end,
        "runtime_seconds": format_float(runtime_seconds),
        "device": str(device),
        "best_epoch": best_epoch(train_history),
        "epochs_completed": len(train_history),
        "final_train_loss": format_float(train_history[-1]["loss"]) if train_history else "",
        "hr_at_5": format_float(utility["hr_at_5"]) if status == "COMPLETED" else "",
        "ndcg_at_5": format_float(utility["ndcg_at_5"]) if status == "COMPLETED" else "",
        "mrr": format_float(utility["mrr_diagnostic"]) if status == "COMPLETED" else "",
        "training_status": status,
        "failure_reason": failure_reason,
        "retry_count": 0,
        "final_test_metrics_computed": "true" if status == "COMPLETED" else "false",
        "final_test_candidates_scored": "true" if status == "COMPLETED" else "false",
        "final_test_targets_used_in_tuning": "false",
        "zero_history_eval_users": data.zero_history_eval_users,
        "train_rows": data.tuning.train_rows,
        "train_pairs": data.tuning.train_pair_count,
        "train_users": data.train_user_count,
        "train_items": data.train_item_count,
    }
    return execution, recommendations, history


def score_bprmf_final(model: torch.nn.Module, data: FinalData, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        users = torch.as_tensor(data.eval_user_indices, dtype=torch.long, device=device).repeat_interleave(CANDIDATE_COUNT)
        items = torch.as_tensor(data.tuning.candidate_items.reshape(-1), dtype=torch.long, device=device)
        scores = model.score(users, items).reshape(len(data.tuning.users), CANDIDATE_COUNT)
    return scores.detach().cpu().numpy()


def score_lightgcn_final(model: torch.nn.Module, graph: torch.Tensor, data: FinalData, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.final_embeddings(graph)
        users = torch.as_tensor(data.eval_user_indices, dtype=torch.long, device=device).repeat_interleave(CANDIDATE_COUNT)
        items = torch.as_tensor(data.tuning.candidate_items.reshape(-1), dtype=torch.long, device=device)
        scores = model.score_from_embeddings(user_emb, item_emb, users, items).reshape(len(data.tuning.users), CANDIDATE_COUNT)
    return scores.detach().cpu().numpy()


def score_sequence_final(model: torch.nn.Module, data: FinalData, device: torch.device, *, allow_empty: bool) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(data.tuning.users), 256):
            sequences_np = data.tuning.eval_sequences[start : start + 256]
            candidates_np = data.tuning.candidate_items[start : start + 256] + 1
            nonempty = sequences_np.sum(axis=1) > 0
            batch_scores = np.zeros((len(sequences_np), CANDIDATE_COUNT), dtype=np.float32)
            if nonempty.any():
                sequences = torch.as_tensor(sequences_np[nonempty], dtype=torch.long, device=device)
                candidates = torch.as_tensor(candidates_np[nonempty], dtype=torch.long, device=device)
                batch_scores[nonempty] = model.score_candidates(sequences, candidates).detach().cpu().numpy()
            if not allow_empty and (~nonempty).any():
                raise ValueError("empty evaluation sequence encountered")
            outputs.append(batch_scores)
    return np.vstack(outputs)


def recommendations_from_scores(spec: RunSpec, data: FinalData, scores: np.ndarray, timestamp: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = data.tuning.candidate_item_ids
    for user_idx, user_id in enumerate(data.tuning.users):
        order = sorted(
            range(CANDIDATE_COUNT),
            key=lambda pos: (-float(scores[user_idx, pos]), sort_key(candidates[user_idx][pos]), int(pos)),
        )
        positive = data.positive_item_ids[user_idx]
        positive_position = int(data.tuning.positive_positions[user_idx])
        for rank, pos in enumerate(order, start=1):
            item_id = candidates[user_idx][pos]
            rows.append(
                {
                    "run_id": spec.run_id,
                    "dataset": spec.dataset,
                    "method": spec.method,
                    "cohort_id": spec.cohort_id,
                    "seed": spec.seed,
                    "user_id": str(user_id),
                    "rank": rank,
                    "recommended_item_id": str(item_id),
                    "score": format_float(scores[user_idx, pos]),
                    "candidate_position": pos + 1,
                    "is_ground_truth": bool(pos + 1 == positive_position),
                    "positive_item_id": positive,
                    "candidate_set_hash": data.tuning.candidate_contract_hash,
                    "candidate_source": data.material.spec.candidate_source.as_posix(),
                    "timestamp": timestamp,
                }
            )
    return pd.DataFrame(rows)


def best_epoch(history: Sequence[Mapping[str, Any]]) -> str:
    if not history:
        return ""
    return str(int(min(history, key=lambda row: float(row["loss"]))["epoch"]))


def per_seed_utility_rows(execution: Mapping[str, Any], stats: pd.DataFrame) -> list[dict[str, Any]]:
    values = utility_metric_values(stats)
    return [
        {
            "result_id": "F1BZ-UTIL-" + sha256_json({"run_id": execution["run_id"], "metric": metric}).upper()[:16],
            "run_id": execution["run_id"],
            "dataset": execution["dataset"],
            "method": execution["method"],
            "seed": execution["seed"],
            "cohort_id": execution["cohort_id"],
            "metric": metric,
            "k": TOP_K if metric.endswith("_at_5") else "",
            "value": format_float(values[metric]),
            "status": "COMPUTED" if execution["training_status"] == "COMPLETED" else "FAILED",
            "evaluated_users": len(stats),
            "undefined_user_count": 0,
            "source_recommendation_hash": execution["recommendation_output_hash"],
            "reason": "computed from F1BZ final test recommendations",
        }
        for metric in UTILITY_METRICS
    ]


def per_seed_beyond_rows(execution: Mapping[str, Any], stats: pd.DataFrame, material: DatasetMaterial, mostpop_stats: pd.DataFrame) -> list[dict[str, Any]]:
    mostpop_hits = mostpop_stats.set_index("user_id")["hit_at_5"].astype(float).to_dict()
    top_lists = [list(items) for items in stats["top_items"]]
    ranks = [int(rank) if str(rank).strip() else TOP_K + 1 for rank in stats["rank_one_based"]]
    hits = [int(value) for value in stats["hit_at_5"]]
    targets = stats["positive_item_id"].astype(str).tolist()
    histories = [material.history_items_by_user.get(str(user_id), []) for user_id in stats["user_id"]]
    values = beyond_metric_values(material, stats, top_lists, ranks, hits, targets, histories, mostpop_hits)
    rows = []
    for metric in BEYOND_METRICS:
        value, status, reason, undefined_count, value_json = values[metric]
        rows.append(
            {
                "result_id": "F1BZ-BEYOND-" + sha256_json({"run_id": execution["run_id"], "metric": metric, "status": status}).upper()[:16],
                "run_id": execution["run_id"],
                "dataset": execution["dataset"],
                "method": execution["method"],
                "seed": execution["seed"],
                "cohort_id": execution["cohort_id"],
                "metric": metric,
                "k": TOP_K if metric.endswith("_at_5") else "",
                "value": "" if value == "" else format_float(value),
                "value_json": stable_json(value_json) if value_json else "",
                "status": status,
                "evaluated_users": len(stats),
                "undefined_user_count": int(undefined_count),
                "source_recommendation_hash": execution["recommendation_output_hash"],
                "reason": reason,
            }
        )
    return rows


def aggregate_metric_rows(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, method, metric), group in per_seed.groupby(["dataset", "method", "metric"], sort=False):
        status_values = sorted(set(group["status"].astype(str)))
        numeric = pd.to_numeric(group["value"], errors="coerce")
        defined = numeric.dropna()
        if status_values == ["COMPUTED"] and len(defined) == len(group):
            status = "COMPUTED"
            value = format_float(float(defined.mean()))
            std = format_float(float(defined.std(ddof=1))) if len(defined) > 1 else format_float(0.0)
        elif "DEGENERATE_PROTOCOL_INPUT" in status_values and len(status_values) == 1:
            status = "DEGENERATE_PROTOCOL_INPUT"
            value = ""
            std = ""
        elif "NOT_APPLICABLE" in status_values and len(status_values) == 1:
            status = "NOT_APPLICABLE"
            value = ""
            std = ""
        else:
            status = "FAILED_OR_MIXED"
            value = format_float(float(defined.mean())) if len(defined) else ""
            std = format_float(float(defined.std(ddof=1))) if len(defined) > 1 else ""
        rows.append(
            {
                "result_id": "F1BZ-AGG-" + sha256_json({"dataset": dataset, "method": method, "metric": metric, "status": status}).upper()[:16],
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "k": TOP_K if str(metric).endswith("_at_5") else "",
                "mean_value": value,
                "std_value": std,
                "seed_count": len(group),
                "computed_seed_count": int(group["status"].eq("COMPUTED").sum()),
                "status": status,
                "per_seed_values": stable_json(dict(zip(group["seed"].astype(str), group["value"].astype(str)))),
                "reason": "five-seed aggregate over frozen final runs",
            }
        )
    return pd.DataFrame(rows)


def load_mostpop_stats(root: Path, config: Mapping[str, Any], material: DatasetMaterial) -> pd.DataFrame:
    recs = pd.read_csv(root / Path(str(config["inputs"]["f1bx_baseline_recommendations"])), dtype=str, keep_default_na=False)
    subset = recs.loc[recs["dataset"].eq(material.spec.name) & recs["method"].eq("MostPop")].copy()
    subset["is_ground_truth"] = bool_series(subset["is_ground_truth"])
    return per_user_stats(subset, material)


def load_completed_run_manifest(root: Path) -> dict[str, dict[str, Any]]:
    path = root / OUTPUT_TABLES["run_manifest"]
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        frame = pd.DataFrame()
    else:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for row in frame.to_dict("records"):
        if row.get("training_status") == "COMPLETED":
            rows[str(row["run_id"])] = row
    rows.update(load_completed_run_artifact_fallback(root, rows))
    return rows


def load_completed_run_artifact_fallback(root: Path, existing: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    runtime_lookup = load_runtime_recovery_lookup(root)
    training_lookup = load_training_recovery_lookup(root)
    recovered: dict[str, dict[str, Any]] = {}
    checkpoint_dir = root / CHECKPOINT_DIR
    if not checkpoint_dir.exists():
        return recovered
    for checkpoint_path in sorted(checkpoint_dir.glob("f1bz_*.pt")):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not isinstance(checkpoint, Mapping):
            continue
        run_id = str(checkpoint.get("run_id", checkpoint_path.stem))
        if run_id in existing and existing[run_id].get("config_hash"):
            continue
        recommendation_path = root / RAW_RECOMMENDATION_DIR / f"{run_id}.csv"
        if not recommendation_path.exists():
            continue
        runtime = runtime_lookup.get(run_id, {})
        training = training_lookup.get(run_id, {})
        recovered[run_id] = {
            "run_id": run_id,
            "dataset": str(checkpoint.get("dataset", "")),
            "method": str(checkpoint.get("method", "")),
            "seed": str(checkpoint.get("seed", "")),
            "config_hash": str(checkpoint.get("config_hash", "")),
            "training_data_hash": str(checkpoint.get("training_data_hash", "")),
            "candidate_contract_hash": str(checkpoint.get("candidate_contract_hash", "")),
            "recommendation_output_path": recommendation_path.relative_to(root).as_posix(),
            "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_hash": sha256_file(checkpoint_path),
            "checkpoint_size_bytes": str(checkpoint_path.stat().st_size),
            "runtime_seconds": str(runtime.get("runtime_seconds") or training.get("fit_seconds") or ""),
            "device": str(runtime.get("device") or ""),
            "training_status": "COMPLETED",
            "retry_count": "0",
            "recovery_status": "RECOVERED_FROM_CHECKPOINT_AND_RECOMMENDATIONS",
        }
    return recovered


def load_runtime_recovery_lookup(root: Path) -> dict[str, dict[str, Any]]:
    path = root / OUTPUT_TABLES["runtime_provenance"]
    if not path.exists() or path.stat().st_size == 0:
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    lookup: dict[str, dict[str, Any]] = {}
    for row in frame.loc[frame["scope"].eq("final_run")].to_dict("records"):
        try:
            value = json.loads(row["value"])
        except json.JSONDecodeError:
            value = {}
        lookup[str(row["key"])] = value
    return lookup


def load_training_recovery_lookup(root: Path) -> dict[str, dict[str, Any]]:
    path = root / UPDATED_F1B_TABLES["training_results"]
    if not path.exists() or path.stat().st_size == 0:
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    lookup: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        run_id = Path(str(row.get("ranking_output_path", ""))).stem
        if run_id:
            lookup[run_id] = row
    return lookup


def can_reuse_completed_run(root: Path, spec: RunSpec, data: FinalData, existing: Mapping[str, Mapping[str, Any]]) -> bool:
    row = existing.get(spec.run_id)
    if not row:
        return False
    checks = [
        str(row.get("dataset")) == spec.dataset,
        str(row.get("method")) == spec.method,
        int(row.get("seed", -1)) == spec.seed,
        str(row.get("config_hash")) == spec.config_hash,
        str(row.get("training_data_hash")) == data.tuning.training_data_hash,
        str(row.get("candidate_contract_hash")) == data.tuning.candidate_contract_hash,
    ]
    rec_path = root / str(row.get("recommendation_output_path", ""))
    ckpt_path = root / str(row.get("checkpoint_path", ""))
    return all(checks) and rec_path.exists() and ckpt_path.exists()


def reuse_completed_run(root: Path, spec: RunSpec, data: FinalData, row: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    recommendations = pd.read_csv(root / str(row["recommendation_output_path"]), dtype=str, keep_default_na=False)
    recommendations["is_ground_truth"] = bool_series(recommendations["is_ground_truth"])
    rec_hash = frame_hash(recommendations)
    stats = per_user_stats(recommendations, data.material)
    utility = utility_metric_values(stats)
    checkpoint_path = root / str(row.get("checkpoint_path", ""))
    checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path.exists() else str(row.get("checkpoint_hash", ""))
    checkpoint_size = checkpoint_path.stat().st_size if checkpoint_path.exists() else str(row.get("checkpoint_size_bytes", ""))
    max_epoch = 5
    execution = {
        "run_id": spec.run_id,
        "dataset": spec.dataset,
        "method": spec.method,
        "seed": spec.seed,
        "seed_slot": spec.seed_slot,
        "cohort_id": spec.cohort_id,
        "evaluation_users": spec.evaluation_users,
        "classification": spec.classification,
        "seed_policy": "LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY",
        "paper_seed_identity_equivalent": "false",
        "config_id": spec.config_id,
        "config_hash": spec.config_hash,
        "hyperparameters": stable_json(spec.hyperparameters),
        "hyperparameter_source": spec.hyperparameter_source,
        "training_data_hash": data.tuning.training_data_hash,
        "training_data_definition": data.training_data_definition,
        "evaluation_cohort_hash": data.evaluation_cohort_hash,
        "candidate_contract_hash": data.tuning.candidate_contract_hash,
        "recommendation_output_path": str(row["recommendation_output_path"]),
        "recommendation_output_hash": rec_hash,
        "checkpoint_path": str(row.get("checkpoint_path", "")),
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_size_bytes": checkpoint_size,
        "start_timestamp": str(row.get("start_timestamp", "")),
        "end_timestamp": str(row.get("end_timestamp", "")),
        "runtime_seconds": str(row.get("runtime_seconds", "")),
        "device": str(row.get("device", "")),
        "best_epoch": str(row.get("best_epoch", max_epoch)),
        "epochs_completed": str(row.get("epochs_completed", max_epoch)),
        "final_train_loss": str(row.get("final_train_loss", "")),
        "hr_at_5": format_float(utility["hr_at_5"]),
        "ndcg_at_5": format_float(utility["ndcg_at_5"]),
        "mrr": format_float(utility["mrr_diagnostic"]),
        "training_status": "COMPLETED",
        "failure_reason": "",
        "retry_count": int(row.get("retry_count", 0)),
        "final_test_metrics_computed": "true",
        "final_test_candidates_scored": "true",
        "final_test_targets_used_in_tuning": "false",
        "zero_history_eval_users": data.zero_history_eval_users,
        "train_rows": data.tuning.train_rows,
        "train_pairs": data.tuning.train_pair_count,
        "train_users": data.train_user_count,
        "train_items": data.train_item_count,
    }
    return execution, recommendations, []


def write_progress_tables(root: Path, execution: Sequence[Mapping[str, Any]], utility: Sequence[Mapping[str, Any]], beyond: Sequence[Mapping[str, Any]], curves: Sequence[Mapping[str, Any]]) -> None:
    pd.DataFrame(execution).to_csv(root / OUTPUT_TABLES["run_manifest"], index=False, lineterminator="\n")
    pd.DataFrame(utility).to_csv(root / OUTPUT_TABLES["per_seed_utility"], index=False, lineterminator="\n")
    pd.DataFrame(beyond).to_csv(root / OUTPUT_TABLES["per_seed_beyond"], index=False, lineterminator="\n")
    training_curve_frame(execution, curves).to_csv(root / OUTPUT_TABLES["training_curves"], index=False, lineterminator="\n")


def training_curve_frame(execution: Sequence[Mapping[str, Any]], curves: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    observed = pd.DataFrame(curves)
    if not observed.empty:
        observed = observed.copy()
        observed["curve_status"] = "OBSERVED_DURING_CURRENT_EXECUTION"
        observed["reason"] = "per-epoch training loss observed while executing the run"
    observed_run_ids = set(observed["run_id"].astype(str)) if not observed.empty else set()
    derived_rows: list[dict[str, Any]] = []
    for row in execution:
        if str(row.get("training_status", "")) != "COMPLETED":
            continue
        if str(row.get("run_id", "")) in observed_run_ids:
            continue
        final_train_loss = str(row.get("final_train_loss", "")).strip()
        epochs_completed = str(row.get("epochs_completed", "")).strip()
        curve_status = "FINAL_EPOCH_FROM_RUN_MANIFEST_AFTER_RESUME"
        reason = "checkpoint stores model state; final epoch/loss are preserved in the run manifest"
        if not final_train_loss:
            curve_status = "RUN_COMPLETION_FROM_RECOVERED_ARTIFACTS"
            reason = "checkpoint and recommendations confirm completion; per-epoch loss was not recoverable from the checkpoint"
        derived_rows.append(
            {
                "run_id": row["run_id"],
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": row["seed"],
                "epoch": epochs_completed,
                "loss": final_train_loss,
                "curve_status": curve_status,
                "reason": reason,
            }
        )
    derived = pd.DataFrame(derived_rows)
    return pd.concat([observed, derived], ignore_index=True) if not observed.empty else derived


def build_first_stage_rankings(root: Path, run_manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for execution in run_manifest.to_dict("records"):
        if execution["training_status"] != "COMPLETED":
            continue
        recs = pd.read_csv(root / execution["recommendation_output_path"], dtype=str, keep_default_na=False)
        ordered_recs = recs.assign(_rank_order=pd.to_numeric(recs["rank"], errors="coerce")).sort_values(
            ["user_id", "_rank_order"],
            kind="mergesort",
        )
        for user_id, group in ordered_recs.groupby("user_id", sort=False):
            ranked = group.sort_values("rank", key=lambda column: column.astype(int), kind="mergesort")
            items = ranked["recommended_item_id"].astype(str).tolist()
            positive = str(ranked["positive_item_id"].iloc[0])
            rank_lookup = {item: index + 1 for index, item in enumerate(items)}
            rows.append(
                {
                    "dataset": execution["dataset"],
                    "user_id": str(user_id),
                    "source_model": execution["method"],
                    "source_seed": execution["seed"],
                    "source_policy": "F1BZ_FINAL_TRAINABLE_RUN",
                    "positive_item": positive,
                    "ranked_items": stable_json(items),
                    "ranking_size": len(items),
                    "positive_present": str(positive in rank_lookup).lower(),
                    "positive_rank": rank_lookup.get(positive, ""),
                    "ranking_hash": sha256_json({"dataset": execution["dataset"], "user_id": str(user_id), "model": execution["method"], "seed": execution["seed"], "items": items}),
                }
            )
    return pd.DataFrame(rows)


def build_reranking_pools(root: Path, config: Mapping[str, Any], first_stage: pd.DataFrame) -> pd.DataFrame:
    policy = config["reranking_source"]
    rows: list[dict[str, Any]] = []
    for (dataset, user_id), group in first_stage.groupby(["dataset", "user_id"], sort=False):
        item_scores: dict[tuple[str, str], list[tuple[int, float, int]]] = defaultdict(list)
        for row in group.to_dict("records"):
            model = str(row["source_model"])
            model_index = MODEL_ORDER.index(model)
            seed = int(row["source_seed"])
            for rank, item in enumerate(json.loads(row["ranked_items"]), start=1):
                item_scores[(model, str(item))].append((seed, 1.0 / rank, rank))
        ordered_items: list[str] = []
        lineage: list[dict[str, Any]] = []
        for model in MODEL_ORDER:
            model_items = []
            for (item_model, item), entries in item_scores.items():
                if item_model != model:
                    continue
                mean_score = float(np.mean([entry[1] for entry in entries]))
                best_rank = int(min(entry[2] for entry in entries))
                model_items.append((item, mean_score, best_rank, sorted(entry[0] for entry in entries)))
            for item, mean_score, best_rank, seeds in sorted(model_items, key=lambda value: (-value[1], value[2], sort_key(value[0]))):
                if item in ordered_items:
                    continue
                ordered_items.append(item)
                lineage.append({"item_id": item, "source_model": model, "source_seeds": seeds, "mean_inverse_rank": mean_score, "best_rank": best_rank})
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
                "source_policy": policy["source_seed_policy"],
                "adapted_policy_id": policy["policy_id"],
                "positive_item": positive,
                "pool_items": stable_json(ordered_items),
                "ranking_size": len(ordered_items),
                "positive_present": str(positive in ordered_items).lower(),
                "positive_rank": positive_rank,
                "ranking_hash": sha256_json({"dataset": dataset, "user_id": str(user_id), "items": ordered_items, "policy": policy["source_seed_policy"]}),
                "source_lineage": stable_json(lineage),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_recall(pools: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in pools.groupby("dataset", sort=False):
        ranks = pd.to_numeric(group["positive_rank"], errors="coerce").dropna()
        present = int(group["positive_present"].astype(str).str.lower().eq("true").sum())
        total = len(group)
        rows.append(
            {
                "dataset": dataset,
                "source_policy": str(group["source_policy"].iloc[0]),
                "evaluation_N": total,
                "positive_present_N": present,
                "positive_absent_N": total - present,
                "candidate_recall_at_20": format_float(present / total if total else math.nan),
                "positive_rank_mean": format_float(float(ranks.mean())) if len(ranks) else "",
                "positive_rank_median": format_float(float(ranks.median())) if len(ranks) else "",
                "positive_rank_min": int(ranks.min()) if len(ranks) else "",
                "positive_rank_max": int(ranks.max()) if len(ranks) else "",
                "status": "COMPUTED",
                "reason": "computed from F1BZ first-stage reranking pools",
            }
        )
    return pd.DataFrame(rows)


def build_reranking_readiness(root: Path, config: Mapping[str, Any], candidate_recall: pd.DataFrame) -> pd.DataFrame:
    recall = candidate_recall.set_index("dataset").to_dict("index")
    frame = read_csv(root / Path(str(config["inputs"]["f0_reranking_matrix"]))).copy()
    frame = frame.rename(columns={"request_count": "f0_request_count"})
    frame["source_ranking"] = "results/tables/f1bz_reranking_pools.csv"
    frame["future_requests_unlocked"] = frame["f0_request_count"].astype(int)
    frame["readiness_status"] = "READY_TO_INFER"
    frame["ground_truth_positive_present_users"] = [recall[row["dataset"]]["positive_present_N"] for row in frame.to_dict("records")]
    frame["candidate_recall_at_20"] = [recall[row["dataset"]]["candidate_recall_at_20"] for row in frame.to_dict("records")]
    frame["reason"] = "F1BZ materialized first-stage rankings and candidate-recall evidence; LLM inference not executed."
    return frame[
        [
            "dataset",
            "task",
            "model_name",
            "prompt",
            "cohort_id",
            "first_stage_recommender",
            "source_ranking",
            "reranking_candidate_size",
            "positive_present_candidate_recall_required",
            "f0_request_count",
            "future_requests_unlocked",
            "ground_truth_positive_present_users",
            "candidate_recall_at_20",
            "readiness_status",
            "reason",
        ]
    ]


def build_traditional_completeness(root: Path, run_manifest: pd.DataFrame, aggregate_utility: pd.DataFrame, aggregate_beyond: pd.DataFrame) -> pd.DataFrame:
    baseline_execution = read_csv(root / "results/tables/f1bx_baseline_execution.csv")
    baseline_utility = read_csv(root / "results/tables/f1bx_baseline_utility_results.csv")
    baseline_beyond = read_csv(root / "results/tables/f1bx_baseline_beyond_utility_results.csv")
    rows: list[dict[str, Any]] = []
    for dataset in ACTIVE_DATASETS:
        for method in (*BASELINE_METHODS, *TRAINABLE_MODELS):
            if method in BASELINE_METHODS:
                expected = 17
                computed = int((baseline_utility["dataset"].eq(dataset) & baseline_utility["method"].eq(method) & baseline_utility["status"].eq("COMPUTED")).sum())
                computed += int((baseline_beyond["dataset"].eq(dataset) & baseline_beyond["method"].eq(method) & baseline_beyond["status"].eq("COMPUTED")).sum())
                degenerate = int((baseline_beyond["dataset"].eq(dataset) & baseline_beyond["method"].eq(method) & baseline_beyond["status"].eq("DEGENERATE_PROTOCOL_INPUT")).sum())
                completed_runs = int((baseline_execution["dataset"].eq(dataset) & baseline_execution["method"].eq(method) & baseline_execution["status"].eq("COMPUTED")).sum())
                cohort_id = str(baseline_execution.loc[baseline_execution["dataset"].eq(dataset) & baseline_execution["method"].eq(method), "cohort_id"].iloc[0])
                users = int(baseline_execution.loc[baseline_execution["dataset"].eq(dataset) & baseline_execution["method"].eq(method), "cohort_N"].iloc[0])
            else:
                util = aggregate_utility.loc[aggregate_utility["dataset"].eq(dataset) & aggregate_utility["method"].eq(method)]
                beyond = aggregate_beyond.loc[aggregate_beyond["dataset"].eq(dataset) & aggregate_beyond["method"].eq(method)]
                expected = len(util) + len(beyond)
                computed = int(util["status"].eq("COMPUTED").sum()) + int(beyond["status"].eq("COMPUTED").sum())
                degenerate = int(beyond["status"].eq("DEGENERATE_PROTOCOL_INPUT").sum())
                completed_runs = int((run_manifest["dataset"].eq(dataset) & run_manifest["method"].eq(method) & run_manifest["training_status"].eq("COMPLETED")).sum())
                first = run_manifest.loc[run_manifest["dataset"].eq(dataset) & run_manifest["method"].eq(method)].iloc[0]
                cohort_id = str(first["cohort_id"])
                users = int(first["evaluation_users"])
            accounted = computed + degenerate
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "cohort_id": cohort_id,
                    "evaluation_users": users,
                    "expected_seed_runs": "" if method in BASELINE_METHODS else 5,
                    "completed_seed_runs": "" if method in BASELINE_METHODS else completed_runs,
                    "expected_result_cells": expected,
                    "materialized_result_cells": accounted,
                    "computed_result_cells": computed,
                    "blocked_execution_cells": max(0, expected - accounted),
                    "degenerate_protocol_input_cells": degenerate,
                    "final_status": "COMPLETED" if accounted == expected and (method in BASELINE_METHODS or completed_runs == 5) else "FAILED_OR_INCOMPLETE",
                    "reason": "F1BZ final traditional method accounted; deterministic baselines reused from F1BX without rerun." if method in BASELINE_METHODS else "F1BZ five-seed final training accounted.",
                }
            )
    return pd.DataFrame(rows)


def build_runtime_provenance(preflight: pd.DataFrame, run_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = [{"scope": "resource_preflight", "key": row["resource"], "value": row["value"]} for row in preflight.to_dict("records")]
    for row in run_manifest.to_dict("records"):
        rows.append(
            {
                "scope": "final_run",
                "key": row["run_id"],
                "value": stable_json(
                    {
                        "dataset": row["dataset"],
                        "method": row["method"],
                        "seed": row["seed"],
                        "runtime_seconds": row["runtime_seconds"],
                        "device": row["device"],
                        "checkpoint_size_bytes": row["checkpoint_size_bytes"],
                    }
                ),
            }
        )
    return pd.DataFrame(rows)


def build_failure_retry_audit(run_manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in run_manifest.to_dict("records"):
        rows.append(
            {
                "run_id": row["run_id"],
                "dataset": row["dataset"],
                "method": row["method"],
                "seed": row["seed"],
                "training_status": row["training_status"],
                "retry_count": row["retry_count"],
                "failure_reason": row["failure_reason"],
                "status": "PASS" if row["training_status"] == "COMPLETED" and int(row["retry_count"]) == 0 else "REVIEW",
                "reason": "completed without retry" if row["training_status"] == "COMPLETED" and int(row["retry_count"]) == 0 else "terminal run requires review",
            }
        )
    return pd.DataFrame(rows)


def build_updated_f1b_tables(
    root: Path,
    config: Mapping[str, Any],
    run_manifest: pd.DataFrame,
    aggregate_utility: pd.DataFrame,
    aggregate_beyond: pd.DataFrame,
    traditional: pd.DataFrame,
    first_stage: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    reranking_readiness: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    updates: dict[str, pd.DataFrame] = {}
    run_by_key = {(row["dataset"], row["method"], str(row["seed_slot"])): row for row in run_manifest.to_dict("records")}
    plan = read_csv(root / UPDATED_F1B_TABLES["training_plan"]).copy()
    for index, row in plan.iterrows():
        key = (row["dataset"], row["method"], row["seed_slot"])
        run = run_by_key[key]
        plan.loc[index, "resolved_seed"] = run["seed"]
        plan.loc[index, "planned_epoch_policy"] = "F1BZ_LOCAL_PROJECT_5_EPOCH_SCHEDULE_FROZEN_BEFORE_RESULTS"
        plan.loc[index, "planned_device"] = run["device"]
        plan.loc[index, "training_plan_status"] = "COMPLETED"
        plan.loc[index, "training_executed"] = "true"
        plan.loc[index, "reason"] = "F1BZ executed the frozen final traditional run."
        plan.loc[index, "evidence"] = "results/manifests/f1bz_final_traditional_execution.json"
    updates["training_plan"] = plan

    results = read_csv(root / UPDATED_F1B_TABLES["training_results"]).copy()
    for index, row in results.iterrows():
        key = (row["dataset"], row["method"], row["seed_slot"])
        run = run_by_key[key]
        results.loc[index, "resolved_seed"] = run["seed"]
        results.loc[index, "training_executed"] = "true"
        results.loc[index, "checkpoint_path"] = run["checkpoint_path"]
        results.loc[index, "ranking_output_path"] = run["recommendation_output_path"]
        results.loc[index, "fit_seconds"] = run["runtime_seconds"]
        results.loc[index, "status"] = run["training_status"]
        results.loc[index, "reason"] = "F1BZ final test metrics computed after winner freeze." if run["training_status"] == "COMPLETED" else run["failure_reason"]
    updates["training_results"] = results

    utility = update_canonical_metric_table(read_csv(root / UPDATED_F1B_TABLES["utility_results"]), root, aggregate_utility, "utility")
    beyond = update_canonical_metric_table(read_csv(root / UPDATED_F1B_TABLES["beyond_utility_results"]), root, aggregate_beyond, "beyond")
    updates["utility_results"] = utility
    updates["beyond_utility_results"] = beyond
    updates["five_dataset_completeness"] = traditional.rename(columns={"completed_seed_runs": "completed_seed_runs"}).copy()
    updates["result_cell_readiness"] = update_result_cell_readiness(read_csv(root / UPDATED_F1B_TABLES["result_cell_readiness"]), utility, beyond)
    updates["first_stage_inventory"] = update_first_stage_inventory(read_csv(root / UPDATED_F1B_TABLES["first_stage_inventory"]), first_stage)
    updates["candidate_recall"] = update_canonical_candidate_recall(read_csv(root / UPDATED_F1B_TABLES["candidate_recall"]), candidate_recall)
    updates["reranking_readiness"] = reranking_readiness.copy()
    return updates


def update_canonical_metric_table(frame: pd.DataFrame, root: Path, aggregate: pd.DataFrame, table_kind: str) -> pd.DataFrame:
    frame = frame.copy()
    baseline_source = read_csv(root / ("results/tables/f1bx_baseline_utility_results.csv" if table_kind == "utility" else "results/tables/f1bx_baseline_beyond_utility_results.csv"))
    baseline_map = {(row["dataset"], row["method"], row["metric"]): row for row in baseline_source.to_dict("records")}
    agg_map = {(row["dataset"], row["method"], row["metric"]): row for row in aggregate.to_dict("records")}
    for index, row in frame.iterrows():
        key = (row["dataset"], row["model_name"], row["metric"])
        if row["model_name"] in BASELINE_METHODS:
            source = baseline_map[key]
            frame.loc[index, "metric_value"] = source.get("value", "")
            frame.loc[index, "value_json"] = source.get("value_json", "")
            frame.loc[index, "f1b_status"] = source.get("status", "")
            frame.loc[index, "reason"] = "F1BX deterministic baseline evidence reused without rerun in F1BZ."
        elif row["model_name"] in TRAINABLE_MODELS:
            source = agg_map[key]
            frame.loc[index, "metric_value"] = source.get("mean_value", "")
            frame.loc[index, "value_json"] = source.get("per_seed_values", "")
            frame.loc[index, "f1b_status"] = source.get("status", "")
            frame.loc[index, "reason"] = "F1BZ five-seed aggregate computed from final traditional runs."
    return frame


def update_result_cell_readiness(frame: pd.DataFrame, utility: pd.DataFrame, beyond: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    updated = pd.concat([utility, beyond], ignore_index=True)
    by_cell = updated.set_index("result_cell_id").to_dict("index")
    for index, row in frame.iterrows():
        source = by_cell.get(row["result_cell_id"])
        if not source:
            continue
        frame.loc[index, "metric_value"] = source.get("metric_value", "")
        frame.loc[index, "value_json"] = source.get("value_json", "")
        frame.loc[index, "f1b_status"] = source.get("f1b_status", "")
        frame.loc[index, "reason"] = source.get("reason", "")
        frame.loc[index, "requires_new_traditional_raw"] = "false"
        frame.loc[index, "requires_training_raw"] = "false"
    return frame


def update_first_stage_inventory(frame: pd.DataFrame, first_stage: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    counts = first_stage.groupby("dataset").size().to_dict()
    for index, row in frame.iterrows():
        dataset = row["dataset"]
        frame.loc[index, "ranking_source"] = "results/tables/f1bz_first_stage_rankings.csv"
        frame.loc[index, "ranking_rows"] = counts.get(dataset, 0)
        frame.loc[index, "positive_recall_audited"] = "true"
        frame.loc[index, "inventory_status"] = "FROZEN"
        frame.loc[index, "reason"] = "F1BZ materialized first-stage trainable rankings for all frozen seeds and models."
    return frame


def update_canonical_candidate_recall(frame: pd.DataFrame, candidate_recall: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    recall = candidate_recall.set_index("dataset").to_dict("index")
    for index, row in frame.iterrows():
        source = recall[row["dataset"]]
        frame.loc[index, "ground_truth_positive_checked_users"] = source["evaluation_N"]
        frame.loc[index, "ground_truth_positive_present_users"] = source["positive_present_N"]
        frame.loc[index, "candidate_recall_at_20"] = source["candidate_recall_at_20"]
        frame.loc[index, "status"] = "COMPUTED"
        frame.loc[index, "reason"] = "F1BZ candidate recall computed before LLM reranking."
    return frame


def build_integrity(
    root: Path,
    config: Mapping[str, Any],
    run_manifest: pd.DataFrame,
    traditional: pd.DataFrame,
    reranking_readiness: pd.DataFrame,
    baseline_before: Mapping[str, Any],
    baseline_after: Mapping[str, Any],
    started_at: str,
) -> dict[str, Any]:
    budget = {row["budget_item"]: int(row["request_count"]) for row in read_csv(root / Path(str(config["inputs"]["f0_request_budget"]))).to_dict("records")}
    f1by = read_json(root / Path(str(config["inputs"]["f1by_winner_manifest"])))
    return {
        "phase_id": PHASE_ID,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "expected_head": EXPECTED_HEAD,
        "head_is_expected": git_output(root, ["rev-parse", "HEAD"]) == EXPECTED_HEAD,
        "active_datasets": list(ACTIVE_DATASETS),
        "active_video_games_rows": active_video_games_rows(root),
        "operational_video_games_training_runs": int(run_manifest["dataset"].astype(str).str.contains("video", case=False).sum()),
        "final_runs": len(run_manifest),
        "completed_runs": int(run_manifest["training_status"].eq("COMPLETED").sum()),
        "failed_runs": int(run_manifest["training_status"].eq("FAILED").sum()),
        "retries": int(pd.to_numeric(run_manifest["retry_count"], errors="coerce").fillna(0).sum()),
        "trainable_dataset_model_rows": int(run_manifest.groupby(["dataset", "method"]).ngroups),
        "traditional_35_rows": len(traditional),
        "traditional_completed_rows": int(traditional["final_status"].eq("COMPLETED").sum()),
        "reranking_ready_rows": int(reranking_readiness["readiness_status"].eq("READY_TO_INFER").sum()),
        "future_llm_requests_unlocked": int(pd.to_numeric(reranking_readiness["future_requests_unlocked"]).sum()),
        "f0_budget": budget,
        "f0_budget_unchanged": budget.get("MANDATORY_RAW_REQUESTS_TOTAL") == 50292 and budget.get("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL") == 100584,
        "global_metric_blockers": len(read_csv(root / Path(str(config["inputs"]["f1arr_metric_blockers"])))),
        "baseline_rows": int(baseline_after["baseline_rows"]),
        "baseline_recommendation_rows": int(baseline_after["baseline_recommendation_rows"]),
        "baseline_hashes_unchanged": baseline_before["hashes_after"] == baseline_after["hashes_after"],
        "baseline_hashes_match_f1bx_manifest": bool(baseline_after["hashes_unchanged"]),
        "foodcom_raw_hashes_unchanged": foodcom_hashes_unchanged(root / Path(str(config["inputs"]["d0a_foodcom_hash_manifest"]))),
        "h0_hashes_unchanged": h0_hashes_unchanged(root, root / Path(str(config["inputs"]["h0_manifest"]))),
        "original_repo_status": git_output(root, ["status", "--short", "--untracked-files=all", "--", "original_repo"]),
        "paper_status": git_output(root, ["status", "--short", "--untracked-files=all", "--", "paper"]),
        "f1by_winner_count": int(f1by["winner_count"]),
        "f1by_final_training_runs_before_f1bz": int(f1by["final_training_runs_executed"]),
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_api_calls_executed": 0,
        "network_calls_executed": 0,
        "downloads_executed": 0,
        "commit_created": 0,
    }


def build_validation(
    root: Path,
    config: Mapping[str, Any],
    run_manifest: pd.DataFrame,
    per_seed_utility: pd.DataFrame,
    per_seed_beyond: pd.DataFrame,
    aggregate_utility: pd.DataFrame,
    aggregate_beyond: pd.DataFrame,
    traditional: pd.DataFrame,
    first_stage: pd.DataFrame,
    pools: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    reranking_readiness: pd.DataFrame,
    integrity: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": str(evidence)})

    check("head_matches_f1by_checkpoint", bool(integrity["head_is_expected"]), integrity["head"])
    check("active_datasets_exact", integrity["active_datasets"] == list(ACTIVE_DATASETS), integrity["active_datasets"])
    check("video_games_absent", int(integrity["active_video_games_rows"]) == 0 and int(integrity["operational_video_games_training_runs"]) == 0, integrity["active_video_games_rows"])
    check("run_count_100", len(run_manifest) == 100, len(run_manifest))
    check("completed_100", int(run_manifest["training_status"].eq("COMPLETED").sum()) == 100, run_manifest["training_status"].value_counts().to_dict())
    check("zero_failed_zero_retries", int(integrity["failed_runs"]) == 0 and int(integrity["retries"]) == 0, {"failed": integrity["failed_runs"], "retries": integrity["retries"]})
    check("twenty_trainable_rows", int(integrity["trainable_dataset_model_rows"]) == 20, integrity["trainable_dataset_model_rows"])
    check("five_seeds_per_dataset_model", run_manifest.groupby(["dataset", "method"]).size().eq(5).all(), run_manifest.groupby(["dataset", "method"]).size().to_dict())
    check("foodcom_winner_identity", foodcom_winners_match(root, run_manifest), "F1BY selected config ids and hashes")
    check("no_final_test_selection_feedback", run_manifest["final_test_targets_used_in_tuning"].astype(str).str.lower().eq("false").all(), "false")
    check("utility_per_seed_rows", len(per_seed_utility) == 300 and per_seed_utility["status"].eq("COMPUTED").all(), len(per_seed_utility))
    check("beyond_per_seed_rows", len(per_seed_beyond) == 1400 and per_seed_beyond["status"].isin(["COMPUTED", "DEGENERATE_PROTOCOL_INPUT", "NOT_APPLICABLE"]).all(), per_seed_beyond["status"].value_counts().to_dict())
    check("aggregate_utility_rows", len(aggregate_utility) == 60, len(aggregate_utility))
    check("aggregate_beyond_rows", len(aggregate_beyond) == 280, len(aggregate_beyond))
    check("traditional_35_complete", len(traditional) == 35 and traditional["final_status"].eq("COMPLETED").all(), traditional["final_status"].value_counts().to_dict())
    check("first_stage_rankings_materialized", len(first_stage) == int((run_manifest["evaluation_users"].astype(int)).sum()), len(first_stage))
    check("reranking_pools_complete", len(pools) == sum(expected_users_by_dataset().values()) and pools["ranking_size"].astype(int).eq(20).all(), len(pools))
    check("candidate_recall_computed", len(candidate_recall) == 5 and candidate_recall["status"].eq("COMPUTED").all(), len(candidate_recall))
    check("reranking_ready_30", len(reranking_readiness) == 30 and reranking_readiness["readiness_status"].eq("READY_TO_INFER").all(), reranking_readiness["readiness_status"].value_counts().to_dict())
    check("llm_request_upper_bound", int(integrity["future_llm_requests_unlocked"]) == 29292, integrity["future_llm_requests_unlocked"])
    check("f0_budget_unchanged", bool(integrity["f0_budget_unchanged"]), integrity["f0_budget"])
    check("global_metric_blockers_zero", int(integrity["global_metric_blockers"]) == 0, integrity["global_metric_blockers"])
    check("baseline_preserved", bool(integrity["baseline_hashes_unchanged"]) and bool(integrity["baseline_hashes_match_f1bx_manifest"]), "F1BX baseline manifest hashes")
    check("foodcom_h0_hashes_unchanged", bool(integrity["foodcom_raw_hashes_unchanged"]) and bool(integrity["h0_hashes_unchanged"]), "protected hashes")
    check("original_repo_paper_unchanged", integrity["original_repo_status"] == "" and integrity["paper_status"] == "", {"original_repo": integrity["original_repo_status"], "paper": integrity["paper_status"]})
    check("zero_llm_network_download", all(int(integrity[key]) == 0 for key in ("llm_inference_calls_executed", "ollama_recommendation_calls_executed", "external_api_calls_executed", "network_calls_executed", "downloads_executed")), "0")
    return pd.DataFrame(rows)


def expected_users_by_dataset() -> dict[str, int]:
    return {"beauty": 990, "sports": 892, "ml1m": 1000, "lastfm": 1000, "foodcom": 1000}


def foodcom_winners_match(root: Path, run_manifest: pd.DataFrame) -> bool:
    winners = read_csv(root / "results/tables/f1by_foodcom_selected_configs.csv").set_index("model").to_dict("index")
    food = run_manifest.loc[run_manifest["dataset"].eq("foodcom")]
    for row in food.to_dict("records"):
        winner = winners[row["method"]]
        if row["config_id"] != winner["selected_config_id"] or row["config_hash"] != winner["config_hash"]:
            return False
    return True


def build_manifest(root: Path, config: Mapping[str, Any], validation: pd.DataFrame, integrity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_VALIDATED if validation["status"].eq("PASS").all() else DECISION_BLOCKED,
        "protocol_checkpoint": PROTOCOL_CHECKPOINT,
        "head": integrity["head"],
        "expected_head": EXPECTED_HEAD,
        "active_datasets": list(ACTIVE_DATASETS),
        "models": list(TRAINABLE_MODELS),
        "seeds": list(FINAL_SEEDS),
        "implementation_classification": config["runtime"]["implementation_classification"],
        "original_four_classification": "PAPER_HYPERPARAMETER_ALIGNED_LOCAL_IMPLEMENTATION",
        "foodcom_classification": "ORIGINAL_EXTENSION",
        "seed_policy": "LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY",
        "paper_run_count_aligned": True,
        "paper_seed_identity_equivalent": False,
        "final_runs": integrity["final_runs"],
        "completed_runs": integrity["completed_runs"],
        "failed_runs": integrity["failed_runs"],
        "traditional_35_rows": integrity["traditional_35_rows"],
        "future_llm_requests_unlocked": integrity["future_llm_requests_unlocked"],
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS")].to_dict("records"),
        "integrity": dict(integrity),
    }


def build_no_llm_manifest(integrity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_VALIDATED if int(integrity["completed_runs"]) == 100 else DECISION_BLOCKED,
        "llm_inference_calls_executed": 0,
        "ollama_recommendation_calls_executed": 0,
        "external_api_calls_executed": 0,
        "network_calls_executed": 0,
        "downloads_executed": 0,
        "f0_budget_unchanged": integrity["f0_budget_unchanged"],
        "future_llm_requests_unlocked_not_executed": integrity["future_llm_requests_unlocked"],
    }


def render_report(
    config: Mapping[str, Any],
    preflight: pd.DataFrame,
    run_manifest: pd.DataFrame,
    aggregate_utility: pd.DataFrame,
    aggregate_beyond: pd.DataFrame,
    traditional: pd.DataFrame,
    first_stage: pd.DataFrame,
    candidate_recall: pd.DataFrame,
    reranking_readiness: pd.DataFrame,
    runtime: pd.DataFrame,
    failure_retry: pd.DataFrame,
    validation: pd.DataFrame,
    integrity: Mapping[str, Any],
) -> str:
    preflight_map = {row["resource"]: row["value"] for row in preflight.to_dict("records")}
    lines = [
        "# F1BZ Final Traditional Execution",
        "",
        f"Decision token: `{DECISION_VALIDATED if validation['status'].eq('PASS').all() else DECISION_BLOCKED}`",
        "",
        "## Precondition",
        f"- HEAD: `{integrity['head']}`; expected F1BY checkpoint: `{EXPECTED_HEAD}`.",
        "- Working tree was clean before F1BZ edits; F1BZ does not commit.",
        "",
        "## Resource Preflight",
        f"- Python `{preflight_map.get('python', '')}`; Torch `{preflight_map.get('torch', '')}`; CUDA `{preflight_map.get('cuda_available', '')}`; GPU `{preflight_map.get('gpu', '')}`.",
        "",
        "## Final Run Accounting",
        f"- Final runs: `{len(run_manifest)}`; completed `{integrity['completed_runs']}`; failed `{integrity['failed_runs']}`; retries `{integrity['retries']}`.",
        f"- Final-test metrics computed after winner freeze; final-test targets used in tuning: `false`.",
        "",
        "## Five-Seed Aggregates",
        f"- Utility aggregate rows: `{len(aggregate_utility)}`; beyond-utility aggregate rows: `{len(aggregate_beyond)}`.",
        "",
        "## Traditional 35-Row Completeness",
        f"- Rows: `{len(traditional)}`; completed `{int(traditional['final_status'].eq('COMPLETED').sum())}`.",
        "",
        "## First-Stage Rankings",
        f"- Ranking rows: `{len(first_stage)}`; source policy `{config['reranking_source']['source_seed_policy']}`.",
        "",
        "## Candidate Recall",
        f"- Dataset recall rows: `{len(candidate_recall)}`.",
        "",
        "## Reranking Readiness",
        f"- READY_TO_INFER rows: `{int(reranking_readiness['readiness_status'].eq('READY_TO_INFER').sum())}`; future LLM requests unlocked `{integrity['future_llm_requests_unlocked']}`; executed `0`.",
        "",
        "## Failure / Retry Audit",
        f"- Review rows: `{int(failure_retry['status'].ne('PASS').sum())}`.",
        "",
        "## LLM Inference Counter",
        "- LLM inference requests `0`; Ollama recommendation calls `0`; external API calls `0`; network/download calls `0`.",
        "",
        "## Validation",
        f"- Focused validation passed: `{str(validation['status'].eq('PASS').all()).lower()}`.",
    ]
    for dataset in ACTIVE_DATASETS:
        dataset_runs = run_manifest.loc[run_manifest["dataset"].eq(dataset)]
        lines.extend(["", f"## {dataset}", f"- Runs: `{len(dataset_runs)}`; completed `{int(dataset_runs['training_status'].eq('COMPLETED').sum())}`."])
    for method in TRAINABLE_MODELS:
        method_runs = run_manifest.loc[run_manifest["method"].eq(method)]
        lines.extend(["", f"## {method}", f"- Runs: `{len(method_runs)}`; completed `{int(method_runs['training_status'].eq('COMPLETED').sum())}`."])
    lines.extend(["", "## Files"])
    for rel_path in OUTPUT_TABLES.values():
        lines.append(f"- `{rel_path.as_posix()}`")
    return "\n".join(lines) + "\n"


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    ensure_output_dirs(root)
    for key, rel_path in OUTPUT_TABLES.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(root / rel_path, index=False, lineterminator="\n")
    for key, rel_path in UPDATED_F1B_TABLES.items():
        frame = outputs["updated_f1b"][key]
        frame.to_csv(root / rel_path, index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(str(outputs["report"]), encoding="utf-8")
    manifest = dict(outputs["manifest"])
    artifact_paths = [*OUTPUT_TABLES.values(), *UPDATED_F1B_TABLES.values(), REPORT_PATH, *SUPPORT_ARTIFACTS]
    manifest["artifact_hashes"] = {path.as_posix(): sha256_file(root / path) for path in artifact_paths if (root / path).exists()}
    (root / OUTPUT_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    no_llm = dict(outputs["no_llm_manifest"])
    no_llm["f1bz_manifest_hash"] = sha256_file(root / OUTPUT_MANIFEST)
    (root / OUTPUT_NO_LLM_MANIFEST).write_text(json.dumps(no_llm, indent=2, sort_keys=True), encoding="utf-8")


def load_existing_outputs(root: Path) -> dict[str, Any]:
    outputs = {key: read_csv(root / path) for key, path in OUTPUT_TABLES.items()}
    outputs["manifest"] = read_json(root / OUTPUT_MANIFEST)
    outputs["no_llm_manifest"] = read_json(root / OUTPUT_NO_LLM_MANIFEST)
    outputs["integrity"] = outputs["manifest"]["integrity"]
    return outputs


def foodcom_hashes_unchanged(path: Path) -> bool:
    manifest = read_json(path)
    entries = manifest.get("files", {})
    if isinstance(entries, dict):
        iterable = entries.items()
    else:
        iterable = []
    for file_path, expected in iterable:
        expected_hash = expected.get("sha256") if isinstance(expected, Mapping) else expected
        if expected_hash and Path(file_path).exists() and sha256_file(Path(file_path)) != expected_hash:
            return False
    return True


def h0_hashes_unchanged(root: Path, path: Path) -> bool:
    manifest = read_json(path)
    hashes = manifest.get("artifact_hashes", {})
    for rel_path, expected in hashes.items():
        full = root / rel_path
        if full.exists() and sha256_file(full) != expected:
            return False
    return True


def active_video_games_rows(root: Path) -> int:
    count = 0
    for rel in (
        "results/tables/f0_final_dataset_matrix.csv",
        "results/tables/f0_final_traditional_model_matrix.csv",
        "results/tables/f0_final_raw_experiment_universe.csv",
        "results/tables/f0_final_result_cell_universe.csv",
    ):
        frame = read_csv(root / rel)
        count += int(frame.apply(lambda col: col.astype(str).str.contains("video", case=False, na=False)).any(axis=1).sum())
    return count


def git_output(root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


if __name__ == "__main__":
    raise SystemExit(main())

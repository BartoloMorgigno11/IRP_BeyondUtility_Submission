"""F1BY Food.com validation-only hyperparameter tuning.

The runner executes the 40 F1BR/F1BXR Food.com tuning configurations with
local project PyTorch wrappers. It trains only on the nested tuning prefix and
selects winners only from validation NDCG@5.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
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

from src.baselines.plan_d_bprmf import BPRMFModel, bpr_loss
from src.baselines.plan_d_gru4rec import GRU4RecModel
from src.baselines.plan_d_lightgcn import LightGCNModel, build_normalized_bipartite_graph
from src.baselines.plan_e_sasrec_training import FiniteMaskedSASRecModel
from src.extension.e0_protocol import CANDIDATE_COUNT, bool_text, sha256_json, stable_int_seed, stable_json
from src.extension.f0_final_extension_freeze import sha256_file
from src.utils.config import find_project_root


PHASE_ID = "f1by_foodcom_validation_only_tuning"
DECISION_SELECTED = "F1BY_FOODCOM_WINNERS_SELECTED"
DECISION_BLOCKED = "F1BY_FOODCOM_TUNING_BLOCKED"
EXPECTED_HEAD = "3842beb94d2303a3987795a3a22854a691e526df"
PROTOCOL_CHECKPOINT = "61fbde6d048faaed9e20508484efc90fa95fa5bf"
DATASET = "foodcom"
ACTIVE_DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")
TRAINABLE_MODELS = ("BPRMF", "GRU4Rec", "SASRec", "LightGCN")
TOP_K = 5
TUNING_SEED = 42
CONFIG_PATH = Path("configs/f1by_foodcom_validation_tuning.yaml")

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")
REPORT_PATH = REPORT_DIR / "f1by_foodcom_validation_tuning.md"

OUTPUT_TABLES = {
    "tuning_execution": TABLE_DIR / "f1by_foodcom_tuning_execution.csv",
    "tuning_metrics": TABLE_DIR / "f1by_foodcom_tuning_metrics.csv",
    "config_ranking": TABLE_DIR / "f1by_foodcom_config_ranking.csv",
    "selected_configs": TABLE_DIR / "f1by_foodcom_selected_configs.csv",
    "training_runtime": TABLE_DIR / "f1by_training_runtime.csv",
    "test_leakage_audit": TABLE_DIR / "f1by_test_leakage_audit.csv",
    "validation": TABLE_DIR / "f1by_validation.csv",
}

OUTPUT_MANIFESTS = {
    "tuning_execution_manifest": MANIFEST_DIR / "f1by_foodcom_tuning_execution.json",
    "winner_selection_manifest": MANIFEST_DIR / "f1by_foodcom_winner_selection.json",
    "no_llm_inference_manifest": MANIFEST_DIR / "f1by_no_llm_inference.json",
}

SUPPORT_ARTIFACTS = (
    CONFIG_PATH,
    Path("src/extension/f1by_foodcom_validation_tuning.py"),
    Path("tests/test_extension_f1by_foodcom_validation_tuning.py"),
    Path(".gitignore"),
)

F1BX_BASELINE_ARTIFACTS = (
    Path("results/tables/f1bx_baseline_execution.csv"),
    Path("results/tables/f1bx_baseline_utility_results.csv"),
    Path("results/tables/f1bx_baseline_beyond_utility_results.csv"),
    Path("results/raw_outputs/f1bx_baseline_recommendations.csv"),
)

FUTURE_FOODCOM_FINAL_CONTRACT = {
    "selected_config_source": "results/tables/f1by_foodcom_selected_configs.csv",
    "selected_config_policy": "use exactly one frozen F1BY validation-selected configuration per trainable model",
    "train_history": "i1..i[T-1]",
    "final_seeds": [42, 43, 44, 45, 46],
    "final_target": "i[T]",
    "final_candidate_contract": "pre_existing_frozen_F0_foodcom_final_test_candidate_contract",
    "evaluation_policy": "evaluate once on the frozen final target after winner freeze",
    "retune_on_final_test_performance": False,
    "status": "FROZEN_NOT_EXECUTED_F1BY",
}


@dataclass(frozen=True)
class TuningData:
    users: list[str]
    user_index: dict[str, int]
    item_index: dict[str, int]
    training_events: pd.DataFrame
    train_user_indices: np.ndarray
    train_item_indices: np.ndarray
    train_item_universe: np.ndarray
    user_positive_items: dict[int, set[int]]
    seq_examples: np.ndarray
    seq_user_indices: np.ndarray
    seq_positive_tokens: np.ndarray
    seq_user_positive_tokens: dict[int, set[int]]
    eval_sequences: np.ndarray
    candidate_items: np.ndarray
    candidate_item_ids: list[list[str]]
    validation_targets: list[str]
    positive_positions: np.ndarray
    training_data_hash: str
    validation_data_hash: str
    candidate_contract_hash: str
    train_rows: int
    train_pair_count: int
    seq_example_count: int
    num_users: int
    num_items: int


@dataclass(frozen=True)
class RunOutcome:
    execution: dict[str, Any]
    metrics: list[dict[str, Any]]
    ranking: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F1BY Food.com validation-only tuning.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_project_root()
    outputs = generate_f1by(root, args.config, execute=True)
    write_outputs(root, outputs)
    manifest = outputs["tuning_execution_manifest"]
    print(f"decision={manifest['decision_token']}")
    print(f"planned={manifest['foodcom_tuning_runs_planned']}")
    print(f"terminal={manifest['foodcom_tuning_runs_terminal']}")
    print(f"completed={manifest['foodcom_tuning_runs_completed']}")
    print(f"winner_count={manifest['winner_count']}")
    print(f"final_training_runs_executed={manifest['final_training_runs_executed']}")
    print(f"llm_inference_calls={manifest['llm_inference_calls_executed']}")
    print(f"validation_passed={bool_text(manifest['validation_passed'])}")
    return 0 if manifest["decision_token"] == DECISION_SELECTED else 1


def generate_f1by(root: Path | None = None, config_path: Path | str = CONFIG_PATH, *, execute: bool = True) -> dict[str, Any]:
    project_root = root or find_project_root()
    config = load_config(project_root, config_path)
    paths = resolved_paths(project_root, config)
    preflight = resource_preflight(project_root)
    baseline_before = baseline_hash_state(project_root, paths["f1bx_execution_manifest"])
    readiness = pd.read_csv(paths["f1bxr_tuning_readiness"] if "f1bxr_tuning_readiness" in paths else project_root / "results/tables/f1bxr_tuning_readiness.csv", dtype=str, keep_default_na=False)
    search = pd.read_csv(paths["f1br_search_space"], dtype=str, keep_default_na=False)
    tuning_data = build_tuning_data(paths, config)
    if execute:
        execution, metrics, ranking, selected, runtime = execute_tuning(project_root, config, search, readiness, tuning_data, preflight)
    else:
        execution, metrics, ranking, selected, runtime = planned_outputs(search, readiness, tuning_data, preflight)
    baseline_after = baseline_hash_state(project_root, paths["f1bx_execution_manifest"])
    leakage = build_test_leakage_audit(project_root, paths, execution, metrics, selected)
    integrity = build_integrity(project_root, paths, baseline_before, baseline_after, execution, selected)
    validation = build_validation(search, readiness, tuning_data, execution, metrics, ranking, selected, leakage, integrity)
    decision = DECISION_SELECTED if validation["status"].eq("PASS").all() and int(selected["selected"].astype(str).str.lower().eq("true").sum()) == 4 else DECISION_BLOCKED
    manifests = build_manifests(project_root, config, decision, preflight, tuning_data, execution, metrics, ranking, selected, runtime, leakage, integrity, validation)
    report = render_report(decision, preflight, execution, ranking, selected, runtime, leakage, integrity, validation)
    return {
        "decision_token": decision,
        "tuning_execution": execution,
        "tuning_metrics": metrics,
        "config_ranking": ranking,
        "selected_configs": selected,
        "training_runtime": runtime,
        "test_leakage_audit": leakage,
        "validation": validation,
        "baseline_before": baseline_before,
        "baseline_after": baseline_after,
        "integrity": integrity,
        "tuning_data": tuning_data,
        "report": report,
        **manifests,
    }


def load_config(root: Path, config_path: Path | str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    loaded = __import__("yaml").safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def resolved_paths(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    raw = config.get("paths", {})
    if not isinstance(raw, Mapping):
        raise ValueError("config.paths must be a mapping")
    return {key: root / Path(value) for key, value in raw.items()}


def build_tuning_data(paths: Mapping[str, Path], config: Mapping[str, Any]) -> TuningData:
    cohort = pd.read_csv(paths["f1bxr_validation_cohort"], dtype=str, keep_default_na=False)
    candidates = pd.read_csv(paths["f1bxr_validation_candidates"], dtype=str, keep_default_na=False)
    nested = pd.read_csv(paths["f1bxr_nested_split"], dtype=str, keep_default_na=False)
    if cohort["user_id"].nunique() != 1000:
        raise ValueError("F1BY requires exactly 1000 validation users")
    selected_users = set(cohort["user_id"].astype(str))
    interactions = pd.read_csv(
        paths["raw_interactions"],
        dtype={"user_id": "string", "recipe_id": "string", "date": "string"},
        usecols=["user_id", "recipe_id", "date"],
        keep_default_na=False,
        low_memory=False,
    )
    interactions["source_input_order"] = np.arange(1, len(interactions) + 1, dtype=np.int64)
    interactions = interactions.loc[interactions["user_id"].astype(str).isin(selected_users)].copy()
    interactions["parsed_date"] = pd.to_datetime(interactions["date"], errors="coerce")
    if interactions["parsed_date"].isna().any():
        raise ValueError("unparseable Food.com date in selected validation users")
    interactions = interactions.sort_values(["user_id", "parsed_date", "source_input_order"], kind="mergesort")
    split_by_user = nested.set_index("user_id").to_dict("index")

    event_rows: list[dict[str, Any]] = []
    prefix_by_user: dict[str, list[str]] = {}
    for user_id, group in interactions.groupby("user_id", sort=False):
        user = str(user_id)
        split = split_by_user[user]
        prefix_length = int(split["training_history_length"])
        items = group["recipe_id"].astype(str).tolist()
        if len(items) < prefix_length + 1:
            raise ValueError(f"user {user} has fewer rows than its frozen split")
        if str(items[prefix_length]) != str(split["validation_target"]):
            raise ValueError(f"user {user} validation target mismatch")
        prefix = items[:prefix_length]
        prefix_by_user[user] = prefix
        for position, item_id in enumerate(prefix, start=1):
            event_rows.append(
                {
                    "user_id": user,
                    "item_id": str(item_id),
                    "prefix_position": position,
                    "validation_sample_id": split["validation_sample_id"],
                }
            )

    training_events = pd.DataFrame(event_rows)
    users = cohort.sort_values("selection_rank", key=lambda column: column.astype(int))["user_id"].astype(str).tolist()
    candidate_grouped = []
    validation_targets = []
    for user in users:
        user_candidates = candidates.loc[candidates["user_id"].eq(user)].sort_values("candidate_position", key=lambda column: column.astype(int))
        if len(user_candidates) != CANDIDATE_COUNT:
            raise ValueError(f"user {user} does not have 20 validation candidates")
        candidate_grouped.append(user_candidates["candidate_item_id"].astype(str).tolist())
        positives = user_candidates.loc[user_candidates["is_validation_positive"].astype(str).str.lower().eq("true"), "candidate_item_id"].astype(str).tolist()
        if len(positives) != 1:
            raise ValueError(f"user {user} does not have one validation positive")
        validation_targets.append(positives[0])

    item_ids = sorted(
        set(training_events["item_id"].astype(str)).union({item for row in candidate_grouped for item in row}),
        key=sort_key,
    )
    user_index = {user: idx for idx, user in enumerate(users)}
    item_index = {item: idx for idx, item in enumerate(item_ids)}

    pairs = training_events[["user_id", "item_id"]].drop_duplicates().copy()
    train_user_indices = pairs["user_id"].map(user_index).astype("int64").to_numpy()
    train_item_indices = pairs["item_id"].map(item_index).astype("int64").to_numpy()
    user_positive_items: dict[int, set[int]] = defaultdict(set)
    for user_id, item_id in zip(training_events["user_id"], training_events["item_id"]):
        user_positive_items[int(user_index[str(user_id)])].add(int(item_index[str(item_id)]))
    train_item_universe = np.asarray(sorted({int(item_index[item]) for item in training_events["item_id"].astype(str)}), dtype=np.int64)

    max_seq_len = int(config["runtime"]["max_sequence_length"])
    seq_rows: list[np.ndarray] = []
    seq_users: list[int] = []
    seq_positives: list[int] = []
    seq_user_positive_tokens: dict[int, set[int]] = defaultdict(set)
    eval_sequences = np.zeros((len(users), max_seq_len), dtype=np.int64)
    for user in users:
        uidx = user_index[user]
        prefix_indices = [item_index[item] for item in prefix_by_user[user]]
        seq_user_positive_tokens[uidx].update(index + 1 for index in prefix_indices)
        eval_sequences[uidx] = pad_tokens(prefix_indices, max_seq_len)
        for end in range(1, len(prefix_indices)):
            seq_rows.append(pad_tokens(prefix_indices[:end], max_seq_len))
            seq_users.append(uidx)
            seq_positives.append(prefix_indices[end] + 1)
    seq_examples = np.vstack(seq_rows).astype(np.int64) if seq_rows else np.zeros((0, max_seq_len), dtype=np.int64)
    seq_user_indices = np.asarray(seq_users, dtype=np.int64)
    seq_positive_tokens = np.asarray(seq_positives, dtype=np.int64)

    candidate_items = np.asarray([[item_index[item] for item in row] for row in candidate_grouped], dtype=np.int64)
    positive_positions = np.asarray([row.index(target) + 1 for row, target in zip(candidate_grouped, validation_targets)], dtype=np.int64)
    manifest = read_json(paths["f1bxr_validation_contract_manifest"])
    return TuningData(
        users=users,
        user_index=user_index,
        item_index=item_index,
        training_events=training_events,
        train_user_indices=train_user_indices,
        train_item_indices=train_item_indices,
        train_item_universe=train_item_universe,
        user_positive_items={key: set(value) for key, value in user_positive_items.items()},
        seq_examples=seq_examples,
        seq_user_indices=seq_user_indices,
        seq_positive_tokens=seq_positive_tokens,
        seq_user_positive_tokens={key: set(value) for key, value in seq_user_positive_tokens.items()},
        eval_sequences=eval_sequences,
        candidate_items=candidate_items,
        candidate_item_ids=candidate_grouped,
        validation_targets=validation_targets,
        positive_positions=positive_positions,
        training_data_hash=frame_hash(training_events),
        validation_data_hash=manifest["validation_cohort_hash"],
        candidate_contract_hash=manifest["validation_candidate_contract_hash"],
        train_rows=int(len(training_events)),
        train_pair_count=int(len(pairs)),
        seq_example_count=int(len(seq_examples)),
        num_users=len(users),
        num_items=len(item_ids),
    )


def execute_tuning(
    root: Path,
    config: Mapping[str, Any],
    search: pd.DataFrame,
    readiness: pd.DataFrame,
    data: TuningData,
    preflight: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(search) != 40 or not readiness["status"].eq("READY_TO_TUNE").all():
        raise ValueError("F1BY requires 40 READY_TO_TUNE configs before execution")
    device = select_device(str(config["runtime"]["device_policy"]))
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for row in search.to_dict("records"):
        outcome = execute_one_config(row, config, data, device)
        rows.append(outcome.execution)
        metric_rows.extend(outcome.metrics)
        runtime_rows.append(runtime_row(outcome.execution, config, preflight))
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    execution = pd.DataFrame(rows)
    metrics = pd.DataFrame(metric_rows)
    ranking, selected = select_winners(execution)
    runtime = pd.DataFrame(runtime_rows)
    return execution, metrics, ranking, selected, runtime


def execute_one_config(row: Mapping[str, Any], config: Mapping[str, Any], data: TuningData, device: torch.device) -> RunOutcome:
    run_id = "f1by_foodcom_tuning_" + str(row["search_config_id"]).lower()
    hyperparameters = hyperparameters_from_row(row)
    config_hash = sha256_json(
        {
            "search_config_id": row["search_config_id"],
            "hyperparameters": hyperparameters,
            "runtime": config["runtime"],
            "validation_contract": "F1BXR_FOODCOM_VALIDATION_CONTRACT_V1",
        }
    )
    set_deterministic_execution(TUNING_SEED)
    start = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    status = "COMPLETED"
    failure_reason = ""
    best_epoch: int | str = ""
    final_loss = math.nan
    epochs_completed = 0
    hr = math.nan
    ndcg = math.nan
    try:
        if str(row["method"]) == "BPRMF":
            model, history = train_bprmf(hyperparameters, config, data, device, run_id)
            scores = score_bprmf(model, data, device)
        elif str(row["method"]) == "GRU4Rec":
            model, history = train_gru4rec(hyperparameters, config, data, device, run_id)
            scores = score_sequence_model(model, data, device)
        elif str(row["method"]) == "SASRec":
            model, history = train_sasrec(hyperparameters, config, data, device, run_id)
            scores = score_sequence_model(model, data, device)
        elif str(row["method"]) == "LightGCN":
            model, graph, history = train_lightgcn(hyperparameters, config, data, device, run_id)
            scores = score_lightgcn(model, graph, data, device)
        else:
            raise ValueError(f"unknown model: {row['method']}")
        epochs_completed = len(history)
        final_loss = float(history[-1]["loss"]) if history else math.nan
        best_epoch = int(min(history, key=lambda item: float(item["loss"]))["epoch"]) if history else ""
        hr, ndcg = validation_metrics(scores, data)
    except Exception as exc:  # pragma: no cover - exercised on local infrastructure/model failures
        status = "FAILED"
        failure_reason = f"{type(exc).__name__}: {exc}"
    end = datetime.now(timezone.utc).isoformat()
    runtime_seconds = time.perf_counter() - start_clock
    execution = {
        "run_id": run_id,
        "model": row["method"],
        "config_id": row["search_config_id"],
        "dataset": DATASET,
        "hyperparameters": stable_json(hyperparameters),
        "config_hash": config_hash,
        "seed": TUNING_SEED,
        "training_data_hash": data.training_data_hash,
        "validation_data_hash": data.validation_data_hash,
        "candidate_hash": data.candidate_contract_hash,
        "start_timestamp": start,
        "end_timestamp": end,
        "runtime_seconds": format_float(runtime_seconds),
        "device": str(device),
        "training_status": status,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "final_train_loss": format_float(final_loss),
        "validation_hr_at_5": format_float(hr),
        "validation_ndcg_at_5": format_float(ndcg),
        "failure_reason": failure_reason,
        "retry_count": 0,
        "final_test_metrics_computed": "false",
        "final_test_candidates_scored": "false",
        "final_test_targets_used": "false",
    }
    metrics = [
        metric_row(execution, "validation_hr_at_5", hr),
        metric_row(execution, "validation_ndcg_at_5", ndcg),
    ]
    ranking = {"model": row["method"], "config_id": row["search_config_id"], "validation_ndcg_at_5": ndcg}
    return RunOutcome(execution=execution, metrics=metrics, ranking=ranking)


def train_bprmf(
    params: Mapping[str, Any],
    config: Mapping[str, Any],
    data: TuningData,
    device: torch.device,
    run_id: str,
) -> tuple[BPRMFModel, list[dict[str, Any]]]:
    model = BPRMFModel(data.num_users, data.num_items, int(params["embedding"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["l2"]))
    return model, train_pairwise_embeddings(
        model,
        optimizer,
        data.train_user_indices,
        data.train_item_indices,
        data.user_positive_items,
        data.train_item_universe,
        int(config["runtime"]["bprmf_batch_size"]),
        int(config["runtime"]["epochs"]),
        device,
        run_id,
        mode="bprmf",
    )


def train_gru4rec(
    params: Mapping[str, Any],
    config: Mapping[str, Any],
    data: TuningData,
    device: torch.device,
    run_id: str,
) -> tuple[GRU4RecModel, list[dict[str, Any]]]:
    if data.seq_example_count <= 0:
        raise ValueError("GRU4Rec training examples are empty")
    model = GRU4RecModel(
        num_items=data.num_items,
        embedding_dim=int(params["embedding"]),
        hidden_dim=int(params["hidden"]),
        num_layers=1,
        dropout=float(config["runtime"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["l2"]))
    return model, train_sequence_model(
        model,
        optimizer,
        data,
        int(config["runtime"]["gru4rec_batch_size"]),
        int(config["runtime"]["epochs"]),
        device,
        run_id,
    )


def train_sasrec(
    params: Mapping[str, Any],
    config: Mapping[str, Any],
    data: TuningData,
    device: torch.device,
    run_id: str,
) -> tuple[FiniteMaskedSASRecModel, list[dict[str, Any]]]:
    if data.seq_example_count <= 0:
        raise ValueError("SASRec training examples are empty")
    model = FiniteMaskedSASRecModel(
        num_items=data.num_items,
        embedding_dim=int(params["embedding"]),
        max_seq_len=int(config["runtime"]["max_sequence_length"]),
        num_heads=int(params["heads"]),
        num_layers=int(params["layers"]),
        dropout=float(config["runtime"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["l2"]))
    return model, train_sequence_model(
        model,
        optimizer,
        data,
        int(config["runtime"]["sasrec_batch_size"]),
        int(config["runtime"]["epochs"]),
        device,
        run_id,
    )


def train_lightgcn(
    params: Mapping[str, Any],
    config: Mapping[str, Any],
    data: TuningData,
    device: torch.device,
    run_id: str,
) -> tuple[LightGCNModel, torch.Tensor, list[dict[str, Any]]]:
    model = LightGCNModel(data.num_users, data.num_items, int(params["embedding"]), int(params["layers"])).to(device)
    graph = build_normalized_bipartite_graph(
        data.train_user_indices,
        data.train_item_indices,
        num_users=data.num_users,
        num_items=data.num_items,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["l2"]))
    history: list[dict[str, Any]] = []
    users = data.train_user_indices
    positives = data.train_item_indices
    for epoch in range(1, int(config["runtime"]["epochs"]) + 1):
        rng = np.random.default_rng(stable_int_seed(TUNING_SEED, run_id, "lightgcn", epoch))
        order = rng.permutation(len(users))
        negatives = sample_negatives(users, data.train_item_universe, data.user_positive_items, rng)
        losses = []
        for start in range(0, len(order), int(config["runtime"]["lightgcn_batch_size"])):
            batch = order[start : start + int(config["runtime"]["lightgcn_batch_size"])]
            batch_users = torch.as_tensor(users[batch], dtype=torch.long, device=device)
            batch_pos = torch.as_tensor(positives[batch], dtype=torch.long, device=device)
            batch_neg = torch.as_tensor(negatives[batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            user_emb, item_emb = model.final_embeddings(graph)
            pos_scores = model.score_from_embeddings(user_emb, item_emb, batch_users, batch_pos)
            neg_scores = model.score_from_embeddings(user_emb, item_emb, batch_users, batch_neg)
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
    return model, graph, history


def train_pairwise_embeddings(
    model: BPRMFModel,
    optimizer: torch.optim.Optimizer,
    users: np.ndarray,
    positives: np.ndarray,
    user_positive_items: Mapping[int, set[int]],
    item_universe: np.ndarray,
    batch_size: int,
    epochs: int,
    device: torch.device,
    run_id: str,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(stable_int_seed(TUNING_SEED, run_id, mode, epoch))
        order = rng.permutation(len(users))
        negatives = sample_negatives(users, item_universe, user_positive_items, rng)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            batch_users = torch.as_tensor(users[batch], dtype=torch.long, device=device)
            batch_pos = torch.as_tensor(positives[batch], dtype=torch.long, device=device)
            batch_neg = torch.as_tensor(negatives[batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            pos_scores, neg_scores = model(batch_users, batch_pos, batch_neg)
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
    return history


def train_sequence_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: TuningData,
    batch_size: int,
    epochs: int,
    device: torch.device,
    run_id: str,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(stable_int_seed(TUNING_SEED, run_id, "sequence", epoch))
        order = rng.permutation(data.seq_example_count)
        negatives = sample_negative_tokens(data.seq_user_indices, data.train_item_universe + 1, data.seq_user_positive_tokens, rng)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            sequences = torch.as_tensor(data.seq_examples[batch], dtype=torch.long, device=device)
            positives = torch.as_tensor(data.seq_positive_tokens[batch], dtype=torch.long, device=device)
            neg = torch.as_tensor(negatives[batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            pos_scores, neg_scores = model(sequences, positives, neg)
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
    return history


def score_bprmf(model: BPRMFModel, data: TuningData, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        users = torch.arange(data.num_users, dtype=torch.long, device=device).repeat_interleave(CANDIDATE_COUNT)
        items = torch.as_tensor(data.candidate_items.reshape(-1), dtype=torch.long, device=device)
        scores = model.score(users, items).reshape(data.num_users, CANDIDATE_COUNT)
    return scores.detach().cpu().numpy()


def score_sequence_model(model: torch.nn.Module, data: TuningData, device: torch.device) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, data.num_users, 256):
            sequences = torch.as_tensor(data.eval_sequences[start : start + 256], dtype=torch.long, device=device)
            candidates = torch.as_tensor(data.candidate_items[start : start + 256] + 1, dtype=torch.long, device=device)
            scores = model.score_candidates(sequences, candidates)
            outputs.append(scores.detach().cpu().numpy())
    return np.vstack(outputs)


def score_lightgcn(model: LightGCNModel, graph: torch.Tensor, data: TuningData, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.final_embeddings(graph)
        users = torch.arange(data.num_users, dtype=torch.long, device=device).repeat_interleave(CANDIDATE_COUNT)
        items = torch.as_tensor(data.candidate_items.reshape(-1), dtype=torch.long, device=device)
        scores = model.score_from_embeddings(user_emb, item_emb, users, items).reshape(data.num_users, CANDIDATE_COUNT)
    return scores.detach().cpu().numpy()


def validation_metrics(scores: np.ndarray, data: TuningData) -> tuple[float, float]:
    ranks = []
    for row_index, row_scores in enumerate(scores):
        candidate_ids = data.candidate_item_ids[row_index]
        order = sorted(
            range(CANDIDATE_COUNT),
            key=lambda pos: (-float(row_scores[pos]), sort_key(candidate_ids[pos]), int(pos)),
        )
        positive_position_zero = int(data.positive_positions[row_index]) - 1
        rank = order.index(positive_position_zero) + 1
        ranks.append(rank)
    hits = [1.0 if rank <= TOP_K else 0.0 for rank in ranks]
    ndcgs = [(1.0 / math.log2(rank + 1.0)) if rank <= TOP_K else 0.0 for rank in ranks]
    return float(np.mean(hits)), float(np.mean(ndcgs))


def select_winners(execution: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for model in TRAINABLE_MODELS:
        model_rows = execution.loc[execution["model"].eq(model)].copy()
        valid = model_rows.loc[model_rows["training_status"].eq("COMPLETED")].copy()
        valid["validation_ndcg_numeric"] = pd.to_numeric(valid["validation_ndcg_at_5"], errors="coerce")
        valid = valid.sort_values(["validation_ndcg_numeric", "config_id"], ascending=[False, True], kind="mergesort")
        for rank, row in enumerate(valid.to_dict("records"), start=1):
            ranking_rows.append(
                {
                    "model": model,
                    "selection_rank": rank,
                    "config_id": row["config_id"],
                    "config_hash": row["config_hash"],
                    "hyperparameters": row["hyperparameters"],
                    "validation_hr_at_5": row["validation_hr_at_5"],
                    "validation_ndcg_at_5": row["validation_ndcg_at_5"],
                    "training_status": row["training_status"],
                    "tie_break_fields": "validation_ndcg_at_5_desc;config_id_asc",
                    "selected": "true" if rank == 1 else "false",
                }
            )
        if valid.empty:
            selected_rows.append(
                {
                    "model": model,
                    "selected_config_id": "",
                    "config_hash": "",
                    "hyperparameters": "",
                    "validation_hr_at_5": "",
                    "validation_ndcg_at_5": "",
                    "selection_rank": "",
                    "tie_break_fields": "validation_ndcg_at_5_desc;config_id_asc",
                    "selected": "false",
                    "final_test_metrics_available_at_selection": "false",
                    "status": "BLOCKED_NO_VALID_WINNER",
                    "reason": "all configurations failed",
                }
            )
        else:
            winner = valid.iloc[0]
            selected_rows.append(
                {
                    "model": model,
                    "selected_config_id": winner["config_id"],
                    "config_hash": winner["config_hash"],
                    "hyperparameters": winner["hyperparameters"],
                    "validation_hr_at_5": winner["validation_hr_at_5"],
                    "validation_ndcg_at_5": winner["validation_ndcg_at_5"],
                    "selection_rank": 1,
                    "tie_break_fields": "validation_ndcg_at_5_desc;config_id_asc",
                    "selected": "true",
                    "final_test_metrics_available_at_selection": "false",
                    "status": "SELECTED",
                    "reason": "highest validation_ndcg_at_5 with deterministic config_id tie-break",
                }
            )
    return pd.DataFrame(ranking_rows), pd.DataFrame(selected_rows)


def planned_outputs(
    search: pd.DataFrame,
    readiness: pd.DataFrame,
    data: TuningData,
    preflight: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    execution = pd.DataFrame(
        [
            {
                "run_id": "f1by_foodcom_tuning_" + str(row["search_config_id"]).lower(),
                "model": row["method"],
                "config_id": row["search_config_id"],
                "dataset": DATASET,
                "hyperparameters": stable_json(hyperparameters_from_row(row)),
                "config_hash": sha256_json(hyperparameters_from_row(row)),
                "seed": TUNING_SEED,
                "training_data_hash": data.training_data_hash,
                "validation_data_hash": data.validation_data_hash,
                "candidate_hash": data.candidate_contract_hash,
                "start_timestamp": "",
                "end_timestamp": "",
                "runtime_seconds": "",
                "device": "",
                "training_status": "PLANNED_ONLY",
                "best_epoch": "",
                "epochs_completed": 0,
                "final_train_loss": "",
                "validation_hr_at_5": "",
                "validation_ndcg_at_5": "",
                "failure_reason": "",
                "retry_count": 0,
                "final_test_metrics_computed": "false",
                "final_test_candidates_scored": "false",
                "final_test_targets_used": "false",
            }
            for row in search.to_dict("records")
        ]
    )
    return execution, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def build_test_leakage_audit(root: Path, paths: Mapping[str, Path], execution: pd.DataFrame, metrics: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    final_artifacts = find_final_training_result_artifacts(root)
    rows = [
        audit_row("f1bxr_contract_inputs_only", True, "runner reads f1bxr validation cohort/candidates and raw interactions for prefix reconstruction"),
        audit_row("f0_final_test_candidates_not_scored", execution["final_test_candidates_scored"].astype(str).str.lower().eq("false").all(), "execution table"),
        audit_row("f0_final_test_targets_not_used", execution["final_test_targets_used"].astype(str).str.lower().eq("false").all(), "execution table"),
        audit_row("final_test_hr_not_computed", not metrics["metric"].astype(str).str.contains("final_test_hr", case=False, na=False).any(), "metrics table"),
        audit_row("final_test_ndcg_not_computed", not metrics["metric"].astype(str).str.contains("final_test_ndcg", case=False, na=False).any(), "metrics table"),
        audit_row("winner_selection_before_final_test_results", selected["final_test_metrics_available_at_selection"].astype(str).str.lower().eq("false").all(), "selected configs"),
        audit_row("final_result_artifacts_absent_before_winner_freeze", len(final_artifacts) == 0, ";".join(final_artifacts)),
        audit_row("protected_final_files_hashed_only_in_prior_contract", paths["f1bxr_validation_contract_manifest"].exists(), paths["f1bxr_validation_contract_manifest"].name),
    ]
    return pd.DataFrame(rows)


def build_integrity(
    root: Path,
    paths: Mapping[str, Path],
    baseline_before: Mapping[str, Any],
    baseline_after: Mapping[str, Any],
    execution: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, Any]:
    budget = budget_values(paths["f0_request_budget"])
    return {
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "expected_head": EXPECTED_HEAD,
        "head_is_expected": git_output(root, ["rev-parse", "HEAD"]) == EXPECTED_HEAD,
        "active_datasets": list(ACTIVE_DATASETS),
        "active_video_games_rows": active_video_games_rows(root, execution),
        "foodcom_raw_hashes_unchanged": foodcom_hashes_unchanged(paths["d0a_extracted_hashes"]),
        "h0_hashes_unchanged": h0_hashes_unchanged(root, paths["h0_manifest"]),
        "f0_budget": budget,
        "f0_budget_unchanged": budget.get("MANDATORY_RAW_REQUESTS_TOTAL") == 50292 and budget.get("CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL") == 100584,
        "global_metric_blockers": global_metric_blocker_count(paths["f1arr_final_metric_blockers"]),
        "baseline_rows": int(baseline_after["baseline_rows"]),
        "baseline_recommendation_rows": int(baseline_after["baseline_recommendation_rows"]),
        "baseline_hashes_before_after_unchanged": baseline_before["hashes_after"] == baseline_after["hashes_after"],
        "baseline_hashes_match_f1bx_manifest": bool(baseline_after["hashes_unchanged"]),
        "foodcom_tuning_runs_planned": 40,
        "foodcom_tuning_runs_terminal": int(execution["training_status"].isin(["COMPLETED", "FAILED"]).sum()),
        "foodcom_tuning_runs_completed": int(execution["training_status"].eq("COMPLETED").sum()),
        "winner_count": int(selected["selected"].astype(str).str.lower().eq("true").sum()) if not selected.empty else 0,
        "original_four_final_runs_executed": 0,
        "foodcom_final_five_seed_runs_executed": 0,
        "final_training_runs_executed": 0,
        "llm_inference_calls_executed": 0,
        "external_api_calls_executed": 0,
        "network_calls_executed": 0,
        "downloads_executed": 0,
        "commit_created": 0,
    }


def build_validation(
    search: pd.DataFrame,
    readiness: pd.DataFrame,
    data: TuningData,
    execution: pd.DataFrame,
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    selected: pd.DataFrame,
    leakage: pd.DataFrame,
    integrity: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": str(evidence)})

    check("head_matches_f1bxrf_checkpoint", bool(integrity["head_is_expected"]), integrity["head"])
    check("active_datasets_exact", integrity["active_datasets"] == list(ACTIVE_DATASETS), ",".join(integrity["active_datasets"]))
    check("video_games_absent", int(integrity["active_video_games_rows"]) == 0, integrity["active_video_games_rows"])
    check("f0_budget_unchanged", bool(integrity["f0_budget_unchanged"]), integrity["f0_budget"])
    check("foodcom_raw_hashes_unchanged", bool(integrity["foodcom_raw_hashes_unchanged"]), integrity["foodcom_raw_hashes_unchanged"])
    check("h0_hashes_unchanged", bool(integrity["h0_hashes_unchanged"]), integrity["h0_hashes_unchanged"])
    check("global_metric_blockers_zero", int(integrity["global_metric_blockers"]) == 0, integrity["global_metric_blockers"])
    check("search_space_40", len(search) == 40, len(search))
    check("f1bxr_readiness_40_ready", len(readiness) == 40 and readiness["status"].eq("READY_TO_TUNE").all(), readiness["status"].value_counts().to_dict())
    check("seed_42", execution["seed"].astype(int).eq(42).all(), execution["seed"].unique().tolist())
    check("execution_40_terminal", len(execution) == 40 and execution["training_status"].isin(["COMPLETED", "FAILED"]).all(), execution["training_status"].value_counts().to_dict())
    check("execution_40_completed", int(execution["training_status"].eq("COMPLETED").sum()) == 40, execution["training_status"].value_counts().to_dict())
    check("metrics_validation_only", set(metrics["metric"]) == {"validation_hr_at_5", "validation_ndcg_at_5"}, sorted(metrics["metric"].unique()))
    check("no_final_test_metrics", execution["final_test_metrics_computed"].astype(str).str.lower().eq("false").all(), "execution table")
    check("ranking_has_40_rows", len(ranking) == 40, len(ranking))
    check("selected_four_winners", len(selected) == 4 and selected["selected"].astype(str).str.lower().eq("true").all(), selected.to_dict("records"))
    check("winner_models_exact", set(selected["model"]) == set(TRAINABLE_MODELS), sorted(selected["model"].unique()))
    check("final_test_metrics_unavailable_at_selection", selected["final_test_metrics_available_at_selection"].astype(str).str.lower().eq("false").all(), "selected configs")
    check("baseline_rows_preserved", int(integrity["baseline_rows"]) == 15, integrity["baseline_rows"])
    check("baseline_recommendation_rows_preserved", int(integrity["baseline_recommendation_rows"]) == 292920, integrity["baseline_recommendation_rows"])
    check("baseline_hashes_unchanged", bool(integrity["baseline_hashes_before_after_unchanged"]) and bool(integrity["baseline_hashes_match_f1bx_manifest"]), "F1BX manifest hashes")
    check("no_test_leakage", leakage["status"].eq("PASS").all(), leakage.loc[leakage["status"].ne("PASS"), "check"].tolist())
    check("zero_final_training", int(integrity["original_four_final_runs_executed"]) == 0 and int(integrity["foodcom_final_five_seed_runs_executed"]) == 0, "0")
    check("zero_llm_network_download", int(integrity["llm_inference_calls_executed"]) == 0 and int(integrity["network_calls_executed"]) == 0 and int(integrity["downloads_executed"]) == 0, "0")
    check("training_contract_scale", data.train_rows == 24708 and data.num_users == 1000, {"train_rows": data.train_rows, "users": data.num_users})
    return pd.DataFrame(rows)


def build_manifests(
    root: Path,
    config: Mapping[str, Any],
    decision: str,
    preflight: pd.DataFrame,
    data: TuningData,
    execution: pd.DataFrame,
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    selected: pd.DataFrame,
    runtime: pd.DataFrame,
    leakage: pd.DataFrame,
    integrity: Mapping[str, Any],
    validation: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    common = {
        "phase_id": PHASE_ID,
        "decision_token": decision,
        "head": integrity["head"],
        "expected_head": EXPECTED_HEAD,
        "authoritative_protocol_checkpoint": PROTOCOL_CHECKPOINT,
        "dataset": DATASET,
        "implementation_classification": config["runtime"]["implementation_classification"],
        "classification": config["contract"]["classification"],
        "selection_metric": "validation_ndcg_at_5",
        "tuning_seed": TUNING_SEED,
        "candidate_scope": "candidate_set_20",
        "training_data_hash": data.training_data_hash,
        "validation_data_hash": data.validation_data_hash,
        "candidate_hash": data.candidate_contract_hash,
        "foodcom_tuning_runs_planned": 40,
        "foodcom_tuning_runs_terminal": int(execution["training_status"].isin(["COMPLETED", "FAILED"]).sum()),
        "foodcom_tuning_runs_completed": int(execution["training_status"].eq("COMPLETED").sum()),
        "future_foodcom_final_contract": FUTURE_FOODCOM_FINAL_CONTRACT,
        "winner_count": int(selected["selected"].astype(str).str.lower().eq("true").sum()) if not selected.empty else 0,
        "original_four_final_runs_executed": 0,
        "foodcom_final_five_seed_runs_executed": 0,
        "final_training_runs_executed": 0,
        "llm_inference_calls_executed": 0,
        "external_api_calls_executed": 0,
        "network_calls_executed": 0,
        "downloads_executed": 0,
        "validation_passed": bool(validation["status"].eq("PASS").all()),
        "validation_failures": validation.loc[validation["status"].ne("PASS")].to_dict("records"),
        "resource_preflight": {row["resource"]: row["value"] for row in preflight.to_dict("records")},
        "integrity": dict(integrity),
    }
    return {
        "tuning_execution_manifest": {
            **common,
            "run_status_counts": execution["training_status"].value_counts().to_dict(),
            "metric_rows": int(len(metrics)),
            "runtime_rows": int(len(runtime)),
        },
        "winner_selection_manifest": {
            **common,
            "final_test_metrics_available_at_selection": False,
            "selection_rule": "validation_ndcg_at_5_desc;config_id_asc",
            "selected_rows": selected.to_dict("records"),
            "ranking_hash": frame_hash(ranking),
        },
        "no_llm_inference_manifest": {
            **common,
            "ollama_recommendation_calls_executed": 0,
            "raw_llm_outputs_created": 0,
            "final_test_metrics_computed": False,
        },
    }


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for directory in (root / TABLE_DIR, root / MANIFEST_DIR, root / REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    for key, rel_path in OUTPUT_TABLES.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(root / rel_path, index=False, lineterminator="\n")
    (root / REPORT_PATH).write_text(str(outputs["report"]), encoding="utf-8")
    artifact_paths = [*OUTPUT_TABLES.values(), REPORT_PATH, *SUPPORT_ARTIFACTS]
    hashes = artifact_hashes(root, artifact_paths)
    for key, rel_path in OUTPUT_MANIFESTS.items():
        payload = dict(outputs[key])
        payload["artifact_hashes"] = hashes
        (root / rel_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def render_report(
    decision: str,
    preflight: pd.DataFrame,
    execution: pd.DataFrame,
    ranking: pd.DataFrame,
    selected: pd.DataFrame,
    runtime: pd.DataFrame,
    leakage: pd.DataFrame,
    integrity: Mapping[str, Any],
    validation: pd.DataFrame,
) -> str:
    preflight_map = {row["resource"]: row["value"] for row in preflight.to_dict("records")}
    lines = [
        "# F1BY Food.com Validation Tuning",
        "",
        f"Decision token: `{decision}`",
        "",
        "## Execution",
        f"- Planned tuning configs: `{len(execution)}`.",
        f"- Completed configs: `{int(execution['training_status'].eq('COMPLETED').sum())}`.",
        f"- Failed configs: `{int(execution['training_status'].eq('FAILED').sum())}`.",
        f"- Winners selected: `{int(selected['selected'].astype(str).str.lower().eq('true').sum())}`.",
        "- Classification: `ORIGINAL_EXTENSION_VALIDATION_PROTOCOL`.",
        "",
        "## Winners",
    ]
    for row in selected.to_dict("records"):
        lines.append(f"- `{row['model']}`: `{row['selected_config_id']}` NDCG@5 `{row['validation_ndcg_at_5']}` HR@5 `{row['validation_hr_at_5']}`.")
    lines.extend(
        [
            "",
            "## Runtime",
            f"- Python: `{preflight_map.get('python', '')}`.",
            f"- Torch: `{preflight_map.get('torch', '')}`.",
            f"- CUDA: `{preflight_map.get('cuda_available', '')}`.",
            f"- Device rows: `{','.join(sorted(runtime['device'].astype(str).unique())) if not runtime.empty else ''}`.",
            "",
            "## Leakage",
            f"- Leakage audit failures: `{int(leakage['status'].ne('PASS').sum())}`.",
            "- Final-test metrics available at selection: `false`.",
            "",
            "## Preservation",
            f"- Baseline rows: `{integrity['baseline_rows']}`.",
            f"- Baseline recommendation rows: `{integrity['baseline_recommendation_rows']}`.",
            f"- Baseline hashes unchanged: `{bool_text(integrity['baseline_hashes_before_after_unchanged'])}`.",
            "- Original-four final runs: `0`; Food.com final five-seed runs: `0`; LLM inference: `0`.",
            "",
            "## Future Final Contract",
            "- Food.com final runs will use each selected config, train on `i1..i[T-1]`, use seeds `[42,43,44,45,46]`, evaluate once on frozen final target `i[T]` with the pre-existing F0 final-test candidate contract, and never retune on final-test performance.",
            "",
            "## Validation",
            f"- Focused validation passed: `{bool_text(validation['status'].eq('PASS').all())}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def hyperparameters_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {"lr": float(row["lr"]), "l2": float(row["l2"]), "embedding": int(row["embedding"])}
    for key in ("hidden", "layers", "heads"):
        value = str(row.get(key, "")).strip()
        if value:
            params[key] = int(value)
    return params


def sample_negatives(users: np.ndarray, item_universe: np.ndarray, user_positive_items: Mapping[int, set[int]], rng: np.random.Generator) -> np.ndarray:
    negatives = np.empty(len(users), dtype=np.int64)
    for row_index, user_index in enumerate(users):
        positives = user_positive_items.get(int(user_index), set())
        while True:
            candidate = int(item_universe[rng.integers(0, len(item_universe))])
            if candidate not in positives:
                negatives[row_index] = candidate
                break
    return negatives


def sample_negative_tokens(users: np.ndarray, item_universe_tokens: np.ndarray, user_positive_tokens: Mapping[int, set[int]], rng: np.random.Generator) -> np.ndarray:
    negatives = np.empty(len(users), dtype=np.int64)
    for row_index, user_index in enumerate(users):
        positives = user_positive_tokens.get(int(user_index), set())
        while True:
            candidate = int(item_universe_tokens[rng.integers(0, len(item_universe_tokens))])
            if candidate not in positives:
                negatives[row_index] = candidate
                break
    return negatives


def pad_tokens(item_indices: Sequence[int], max_seq_len: int) -> np.ndarray:
    output = np.zeros(max_seq_len, dtype=np.int64)
    clipped = list(item_indices)[-max_seq_len:]
    if clipped:
        output[-len(clipped) :] = np.asarray([int(item) + 1 for item in clipped], dtype=np.int64)
    return output


def set_deterministic_execution(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def select_device(policy: str) -> torch.device:
    normalized = policy.strip().lower()
    if normalized in {"cuda_if_available_else_cpu", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized in {"cuda", "cpu"}:
        return torch.device(normalized)
    raise ValueError(f"unknown device policy: {policy}")


def resource_preflight(root: Path) -> pd.DataFrame:
    values = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool_text(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda or ""),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "vram_bytes": str(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else "0",
        "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
        "free_disk_bytes": str(__import__("shutil").disk_usage(root).free),
        "total_ram_bytes": str(memory_status().get("total_ram_bytes", "")),
        "available_ram_bytes": str(memory_status().get("available_ram_bytes", "")),
    }
    return pd.DataFrame([{"resource": key, "value": value, "status": "RECORDED", "notes": ""} for key, value in values.items()])


def memory_status() -> dict[str, int]:
    if platform.system().lower() != "windows":
        return {}
    class MEMORYSTATUSEX(__import__("ctypes").Structure):
        _fields_ = [
            ("dwLength", __import__("ctypes").c_ulong),
            ("dwMemoryLoad", __import__("ctypes").c_ulong),
            ("ullTotalPhys", __import__("ctypes").c_ulonglong),
            ("ullAvailPhys", __import__("ctypes").c_ulonglong),
            ("ullTotalPageFile", __import__("ctypes").c_ulonglong),
            ("ullAvailPageFile", __import__("ctypes").c_ulonglong),
            ("ullTotalVirtual", __import__("ctypes").c_ulonglong),
            ("ullAvailVirtual", __import__("ctypes").c_ulonglong),
            ("ullAvailExtendedVirtual", __import__("ctypes").c_ulonglong),
        ]
    status = MEMORYSTATUSEX()
    status.dwLength = __import__("ctypes").sizeof(MEMORYSTATUSEX)
    __import__("ctypes").windll.kernel32.GlobalMemoryStatusEx(__import__("ctypes").byref(status))
    return {"total_ram_bytes": int(status.ullTotalPhys), "available_ram_bytes": int(status.ullAvailPhys)}


def runtime_row(execution: Mapping[str, Any], config: Mapping[str, Any], preflight: pd.DataFrame) -> dict[str, Any]:
    return {
        "run_id": execution["run_id"],
        "model": execution["model"],
        "config_id": execution["config_id"],
        "device": execution["device"],
        "runtime_seconds": execution["runtime_seconds"],
        "epochs_completed": execution["epochs_completed"],
        "best_epoch": execution["best_epoch"],
        "status": execution["training_status"],
        "retry_count": execution["retry_count"],
        "optimizer": config["runtime"]["optimizer"],
        "batch_size": batch_size_for(str(execution["model"]), config),
        "runtime_policy_hash": sha256_json(config["runtime"]),
        "resource_preflight_hash": frame_hash(preflight),
    }


def batch_size_for(model: str, config: Mapping[str, Any]) -> int:
    key = {
        "BPRMF": "bprmf_batch_size",
        "GRU4Rec": "gru4rec_batch_size",
        "SASRec": "sasrec_batch_size",
        "LightGCN": "lightgcn_batch_size",
    }[model]
    return int(config["runtime"][key])


def metric_row(execution: Mapping[str, Any], metric: str, value: float) -> dict[str, Any]:
    return {
        "run_id": execution["run_id"],
        "config_id": execution["config_id"],
        "model": execution["model"],
        "dataset": DATASET,
        "metric": metric,
        "k": TOP_K,
        "value": format_float(value),
        "selection_metric": bool_text(metric == "validation_ndcg_at_5"),
        "source": "F1BXR nested validation candidate_set_20",
        "final_test_metric": "false",
    }


def baseline_hash_state(root: Path, f1bx_manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(f1bx_manifest_path)
    expected = manifest.get("artifact_hashes", {})
    actual = {rel.as_posix(): sha256_file(root / rel) for rel in F1BX_BASELINE_ARTIFACTS}
    baseline_execution = pd.read_csv(root / F1BX_BASELINE_ARTIFACTS[0], dtype=str, keep_default_na=False)
    recommendation_rows = line_count(root / F1BX_BASELINE_ARTIFACTS[3]) - 1
    return {
        "baseline_rows": int(len(baseline_execution)),
        "baseline_computed_rows": int(baseline_execution["status"].eq("COMPUTED").sum()),
        "baseline_recommendation_rows": int(recommendation_rows),
        "hashes_after": actual,
        "hashes_unchanged": all(expected.get(path, "") == value for path, value in actual.items()),
    }


def budget_values(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {str(row.budget_item): int(row.request_count) for row in frame.itertuples(index=False)}


def global_metric_blocker_count(path: Path) -> int:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return int(len(frame))


def foodcom_hashes_unchanged(extracted_hashes_path: Path) -> bool:
    manifest = read_json(extracted_hashes_path)
    for info in manifest.get("files", {}).values():
        path = Path(info["relative_extracted_path"])
        if not path.exists() or sha256_file(path) != info["sha256"]:
            return False
    return True


def h0_hashes_unchanged(root: Path, h0_manifest_path: Path) -> bool:
    manifest = read_json(h0_manifest_path)
    for rel_path, expected_hash in manifest.get("artifact_hashes", {}).items():
        path = root / Path(rel_path)
        if not path.exists() or sha256_file(path) != expected_hash:
            return False
    return True


def active_video_games_rows(root: Path, execution: pd.DataFrame) -> int:
    total = int(execution.astype(str).apply(lambda column: column.str.contains("video_games|video games|videogames", case=False, regex=True, na=False)).any(axis=1).sum())
    for rel in ("results/tables/f0_final_dataset_matrix.csv", "results/tables/f0_final_raw_experiment_universe.csv"):
        frame = pd.read_csv(root / rel, dtype=str, keep_default_na=False)
        mask = frame.astype(str).apply(lambda column: column.str.contains("video_games|video games|videogames", case=False, regex=True, na=False))
        total += int(mask.any(axis=1).sum())
    return total


def find_final_training_result_artifacts(root: Path) -> list[str]:
    matches: list[str] = []
    for directory in (root / "results/tables", root / "results/raw_outputs", root / "results/metrics"):
        if not directory.exists():
            continue
        for path in directory.glob("f1b*"):
            name = path.name.lower()
            if "final" in name and ("result" in name or "training" in name or "recommendation" in name):
                matches.append(path.relative_to(root).as_posix())
    return sorted(matches)


def artifact_hashes(root: Path, rel_paths: Sequence[Path]) -> dict[str, str]:
    return {rel.as_posix(): sha256_file(root / rel) for rel in rel_paths if (root / rel).exists()}


def frame_hash(frame: pd.DataFrame) -> str:
    return sha256_json(frame.to_dict("records"))


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return format(number, ".17g")


def sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value)
    try:
        return (0, int(text), "")
    except ValueError:
        return (1, 0, text)


def audit_row(check: str, passed: bool, evidence: str) -> dict[str, str]:
    return {"check": check, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_output(root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


if __name__ == "__main__":
    raise SystemExit(main())

"""Controlled Plan E LightGCN training utilities for the ML-1M pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from src.baselines.plan_d_bprmf import bpr_loss
from src.baselines.plan_d_lightgcn import LightGCNModel
from src.baselines.plan_e_lightgcn_adapter import (
    GraphContract,
    NormalizedSparseGraph,
    bool_value,
    build_graph_contract,
    filter_train_interactions,
    hash_float_array,
    hash_int_arrays,
    hash_normalized_graph,
    hash_raw_pairs,
    json_array,
    natural_id_sort_key,
    parse_candidate_items,
    read_csv_strings,
    safe_raw_id,
)


class LightGCNPilotSafetyError(RuntimeError):
    """Raised when a Phase 3M safety condition fails."""


@dataclass(frozen=True)
class LightGCNPilotConfig:
    experiment_id: str
    model_id: str
    dataset: str
    seed: int
    embedding_dim: int
    propagation_layers: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    device: str
    evaluation_k: int
    max_run_minutes: float


@dataclass(frozen=True)
class TrainingArrays:
    user_indices: np.ndarray
    item_indices: np.ndarray
    item_universe: np.ndarray
    user_positive_items: dict[int, set[int]]
    train_rows: int
    train_pair_count: int
    duplicate_pair_count: int
    unique_train_users: int
    unique_train_items: int


@dataclass(frozen=True)
class NegativeSchedule:
    epoch_orders: tuple[np.ndarray, ...]
    epoch_negatives: tuple[np.ndarray, ...]
    evidence_sha256: str
    checked_triple_count: int


@dataclass(frozen=True)
class TrainingRunResult:
    run_id: str
    execution_role: str
    model: LightGCNModel
    history: list[dict[str, Any]]
    initial_state_hash: str
    final_state_hash: str
    final_embedding_hash: str
    started_at_utc: str
    completed_at_utc: str
    duration_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def set_deterministic_execution(seed: int) -> dict[str, Any]:
    """Apply deterministic random-state settings before model initialization."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return {
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "torch_cuda_seed_all": bool(torch.cuda.is_available()),
        "torch_backends_cudnn_deterministic": True,
        "torch_backends_cudnn_benchmark": False,
        "torch_use_deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "num_workers": 0,
    }


def load_train_interactions(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load ML-1M reconstructed interactions and return verified train rows."""

    frame = read_csv_strings(path)
    train = filter_train_interactions(frame)
    if not train["split"].astype(str).str.lower().eq("train").all():
        raise LightGCNPilotSafetyError("non-train row entered training interactions")
    pair_count = len(train[["user_id", "item_id"]].drop_duplicates())
    per_user = train.groupby("user_id", sort=False).size()
    per_item = train.groupby("item_id", sort=False).size()
    summary = {
        "training_interaction_count": int(len(train)),
        "unique_train_edges": int(pair_count),
        "duplicate_train_edge_count": int(len(train) - pair_count),
        "unique_training_users": int(train["user_id"].nunique()),
        "unique_training_items": int(train["item_id"].nunique()),
        "minimum_user_degree": int(per_user.min()),
        "maximum_user_degree": int(per_user.max()),
        "mean_user_degree": float(per_user.mean()),
        "median_user_degree": float(per_user.median()),
        "minimum_item_degree": int(per_item.min()),
        "maximum_item_degree": int(per_item.max()),
        "mean_item_degree": float(per_item.mean()),
        "median_item_degree": float(per_item.median()),
    }
    return train.reset_index(drop=True), summary


def build_training_arrays(contract: GraphContract) -> TrainingArrays:
    """Create deterministic integer-indexed BPR training arrays from the frozen graph."""

    user_indices = np.asarray(
        [contract.user_mapping.raw_to_local[user_id] for user_id, _ in contract.train_pairs],
        dtype=np.int64,
    )
    item_indices = np.asarray(
        [contract.item_mapping.raw_to_local[item_id] for _, item_id in contract.train_pairs],
        dtype=np.int64,
    )
    positives: dict[int, set[int]] = {}
    for user, item in zip(user_indices, item_indices):
        positives.setdefault(int(user), set()).add(int(item))
    return TrainingArrays(
        user_indices=user_indices,
        item_indices=item_indices,
        item_universe=np.arange(contract.item_mapping.count, dtype=np.int64),
        user_positive_items=positives,
        train_rows=len(contract.train_pairs),
        train_pair_count=len(contract.train_pairs),
        duplicate_pair_count=contract.duplicate_user_item_pair_count,
        unique_train_users=contract.user_mapping.count,
        unique_train_items=contract.item_mapping.count,
    )


def sparse_tensor_from_graph(graph: NormalizedSparseGraph, *, device: torch.device) -> torch.Tensor:
    """Move a coalesced normalized sparse graph to a torch sparse COO tensor."""

    indices = torch.as_tensor(
        np.vstack([graph.rows, graph.cols]).astype(np.int64, copy=False),
        dtype=torch.long,
        device=device,
    )
    values = torch.as_tensor(graph.values.astype(np.float32, copy=False), dtype=torch.float32, device=device)
    tensor = torch.sparse_coo_tensor(indices, values, (graph.total_nodes, graph.total_nodes), device=device)
    tensor = tensor.coalesce()
    if not tensor.is_coalesced():
        raise LightGCNPilotSafetyError("normalized adjacency is not coalesced")
    if not torch.isfinite(tensor.values()).all():
        raise LightGCNPilotSafetyError("normalized adjacency contains non-finite weights")
    return tensor


def validate_project_sparse_backend(normalized_adj: torch.Tensor) -> dict[str, Any]:
    """Run a bounded project-graph sparse multiplication before optimizer creation."""

    if not normalized_adj.is_sparse:
        raise LightGCNPilotSafetyError("normalized adjacency is not sparse")
    device = normalized_adj.device
    probe = torch.ones((normalized_adj.shape[0], 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        result = torch.sparse.mm(normalized_adj, probe)
    if not torch.isfinite(result).all():
        raise LightGCNPilotSafetyError("project sparse backend probe returned non-finite values")
    return {
        "device": str(device),
        "shape": tuple(result.shape),
        "finite": True,
        "sum": float(result.sum().detach().cpu().item()),
    }


def sample_negatives(
    user_indices: np.ndarray,
    *,
    item_universe: np.ndarray,
    user_positive_items: Mapping[int, set[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one train-universe negative that is not a user's train positive."""

    negatives = np.empty(len(user_indices), dtype=np.int64)
    for row_index, user_index in enumerate(user_indices):
        positives = user_positive_items.get(int(user_index), set())
        if len(positives) >= len(item_universe):
            raise LightGCNPilotSafetyError(f"user index {user_index} has no valid negative items")
        while True:
            candidate = int(item_universe[rng.integers(0, len(item_universe))])
            if candidate not in positives:
                negatives[row_index] = candidate
                break
    return negatives


def validate_negative_samples(
    users: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    *,
    item_universe: np.ndarray,
    user_positive_items: Mapping[int, set[int]],
) -> None:
    """Validate sampled triples before they are used for training."""

    universe = set(int(item) for item in item_universe.tolist())
    for user, positive, negative in zip(users, positives, negatives):
        user_id = int(user)
        positive_id = int(positive)
        negative_id = int(negative)
        if positive_id not in user_positive_items.get(user_id, set()):
            raise LightGCNPilotSafetyError("positive item is not a training positive for user")
        if negative_id in user_positive_items.get(user_id, set()):
            raise LightGCNPilotSafetyError("negative item was sampled from user positives")
        if negative_id not in universe:
            raise LightGCNPilotSafetyError("negative item is outside training item universe")


def build_negative_schedule(*, arrays: TrainingArrays, seed: int, epochs: int) -> NegativeSchedule:
    """Create deterministic epoch orderings and negative samples for both runs."""

    order_rng = np.random.default_rng(seed)
    negative_rng = np.random.default_rng(seed + 10_000)
    hasher = hashlib.sha256()
    orders: list[np.ndarray] = []
    negatives_by_epoch: list[np.ndarray] = []
    checked = 0
    for epoch in range(1, epochs + 1):
        order = np.arange(len(arrays.user_indices), dtype=np.int64)
        order_rng.shuffle(order)
        users = arrays.user_indices[order]
        positives = arrays.item_indices[order]
        negatives = sample_negatives(
            users,
            item_universe=arrays.item_universe,
            user_positive_items=arrays.user_positive_items,
            rng=negative_rng,
        )
        validate_negative_samples(
            users,
            positives,
            negatives,
            item_universe=arrays.item_universe,
            user_positive_items=arrays.user_positive_items,
        )
        hasher.update(np.asarray([epoch], dtype="<i8").tobytes())
        hasher.update(order.astype("<i8", copy=False).tobytes())
        hasher.update(negatives.astype("<i8", copy=False).tobytes())
        orders.append(order)
        negatives_by_epoch.append(negatives)
        checked += len(order)
    return NegativeSchedule(
        epoch_orders=tuple(orders),
        epoch_negatives=tuple(negatives_by_epoch),
        evidence_sha256=hasher.hexdigest().upper(),
        checked_triple_count=checked,
    )


def train_one_run(
    *,
    config: LightGCNPilotConfig,
    arrays: TrainingArrays,
    normalized_adj: torch.Tensor,
    schedule: NegativeSchedule,
    execution_role: str,
) -> TrainingRunResult:
    """Train one deterministic LightGCN run."""

    started = utc_now()
    start = time.perf_counter()
    set_deterministic_execution(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = LightGCNModel(
        num_users=arrays.unique_train_users,
        num_items=arrays.unique_train_items,
        embedding_dim=config.embedding_dim,
        num_layers=config.propagation_layers,
    ).to(device)
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, Any]] = []
    max_seconds = config.max_run_minutes * 60.0
    for epoch, (order, negatives_for_epoch) in enumerate(zip(schedule.epoch_orders, schedule.epoch_negatives), start=1):
        epoch_start = time.perf_counter()
        model.train()
        losses: list[float] = []
        batch_count = 0
        for start_index in range(0, len(order), config.batch_size):
            batch_slice = slice(start_index, start_index + config.batch_size)
            batch_order = order[batch_slice]
            users_np = arrays.user_indices[batch_order]
            positives_np = arrays.item_indices[batch_order]
            negatives_np = negatives_for_epoch[batch_slice]
            validate_negative_samples(
                users_np,
                positives_np,
                negatives_np,
                item_universe=arrays.item_universe,
                user_positive_items=arrays.user_positive_items,
            )
            users = torch.as_tensor(users_np, dtype=torch.long, device=device)
            positives = torch.as_tensor(positives_np, dtype=torch.long, device=device)
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            user_embeddings, item_embeddings = model.final_embeddings(normalized_adj)
            positive_scores = model.score_from_embeddings(user_embeddings, item_embeddings, users, positives)
            negative_scores = model.score_from_embeddings(user_embeddings, item_embeddings, users, negatives)
            loss = bpr_loss(positive_scores, negative_scores)
            if not torch.isfinite(loss):
                raise LightGCNPilotSafetyError("non-finite BPR loss encountered")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            batch_count += 1
        if time.perf_counter() - start > max_seconds:
            raise LightGCNPilotSafetyError("training run exceeded maximum runtime")
        mean_loss = float(np.mean(losses))
        if not math.isfinite(mean_loss):
            raise LightGCNPilotSafetyError("non-finite epoch loss encountered")
        device_obj = normalized_adj.device
        history.append(
            {
                "epoch": epoch,
                "batch_count": batch_count,
                "training_example_count": int(len(arrays.user_indices)),
                "mean_bpr_loss": mean_loss,
                "mean_regularization_loss": "",
                "mean_total_loss": mean_loss,
                "learning_rate": config.learning_rate,
                "epoch_duration_seconds": time.perf_counter() - epoch_start,
                "cuda_peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device_obj) if device_obj.type == "cuda" else 0
                ),
                "cuda_peak_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device_obj) if device_obj.type == "cuda" else 0
                ),
                "finite_loss": True,
                "sparse_warning_count": 0,
                "notes": "regularization loss is not separated by the reused Plan D bpr_loss",
            }
        )
    user_final, item_final = propagate_final_embeddings(model, normalized_adj)
    completed = utc_now()
    return TrainingRunResult(
        run_id=f"{config.experiment_id}_{execution_role}",
        execution_role=execution_role,
        model=model,
        history=history,
        initial_state_hash=initial_hash,
        final_state_hash=state_dict_sha256(model.state_dict()),
        final_embedding_hash=embedding_sha256(user_final, item_final),
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=time.perf_counter() - start,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0),
    )


def propagate_final_embeddings(model: LightGCNModel, normalized_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate final LightGCN embeddings and validate finiteness."""

    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model.final_embeddings(normalized_adj)
    if not torch.isfinite(user_embeddings).all() or not torch.isfinite(item_embeddings).all():
        raise LightGCNPilotSafetyError("non-finite propagated embedding encountered")
    return user_embeddings, item_embeddings


def score_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    user_index: Mapping[str, int],
    item_index: Mapping[str, int],
    user_degrees: Mapping[str, float],
    item_degrees: Mapping[str, float],
    config: LightGCNPilotConfig,
    training_run_id: str,
) -> list[dict[str, Any]]:
    """Score strict pilot candidates with propagated LightGCN embeddings."""

    device = user_embeddings.device
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        user_id = safe_raw_id(row["user_id"])
        positive_id = safe_raw_id(row["positive_item_id"])
        candidates = parse_candidate_items(row["candidate_items"])
        if user_id not in user_index:
            raise LightGCNPilotSafetyError(f"pilot user missing from LightGCN index: {user_id}")
        if any(item not in item_index for item in candidates):
            raise LightGCNPilotSafetyError("pilot candidate missing from LightGCN item index")
        if positive_id not in candidates:
            raise LightGCNPilotSafetyError("positive item not present in candidate list")
        user_tensor = torch.full((len(candidates),), int(user_index[user_id]), dtype=torch.long, device=device)
        item_indices = [int(item_index[item]) for item in candidates]
        item_tensor = torch.as_tensor(item_indices, dtype=torch.long, device=device)
        with torch.no_grad():
            scores_tensor = LightGCNModel.score_from_embeddings(user_embeddings, item_embeddings, user_tensor, item_tensor)
        scores = scores_tensor.detach().cpu().numpy().astype(float).tolist()
        if any(not math.isfinite(float(score)) for score in scores):
            raise LightGCNPilotSafetyError("non-finite candidate score encountered")
        ranked_positions = sorted(range(len(candidates)), key=lambda pos: (-float(scores[pos]), str(candidates[pos]), pos))
        ranked_items = [candidates[pos] for pos in ranked_positions]
        ranked_scores = [float(scores[pos]) for pos in ranked_positions]
        if set(ranked_items) != set(candidates):
            raise LightGCNPilotSafetyError("ranked membership differs from candidate membership")
        duplicate_ranked = len(ranked_items) - len(set(ranked_items))
        positive_rank_zero = ranked_items.index(positive_id)
        ndcg = 1.0 / math.log2(positive_rank_zero + 2) if positive_rank_zero < config.evaluation_k else 0.0
        candidate_degrees = [float(item_degrees[item]) for item in candidates]
        if any(degree <= 0 for degree in candidate_degrees):
            raise LightGCNPilotSafetyError("strict pilot candidate has non-positive train degree")
        output_rows.append(
            {
                "experiment_id": config.experiment_id,
                "training_run_id": training_run_id,
                "model_id": config.model_id,
                "dataset": "ML-1M",
                "task_type": str(row["task_type"]),
                "candidate_set_id": str(row["candidate_set_id"]),
                "source_path": str(row["source_path"]),
                "user_id": user_id,
                "positive_item_id": positive_id,
                "user_train_degree": float(user_degrees[user_id]),
                "positive_item_train_degree": float(item_degrees[positive_id]),
                "candidate_items": json_array(candidates),
                "candidate_item_train_degrees": json.dumps(candidate_degrees, ensure_ascii=True, separators=(",", ":")),
                "candidate_scores": json.dumps([float(score) for score in scores], ensure_ascii=True, separators=(",", ":")),
                "ranked_items": json_array(ranked_items),
                "ranked_scores": json.dumps(ranked_scores, ensure_ascii=True, separators=(",", ":")),
                "positive_rank_zero_based": positive_rank_zero,
                "positive_rank_one_based": positive_rank_zero + 1,
                "positive_in_top_5": positive_rank_zero < config.evaluation_k,
                "ndcg_at_5": ndcg,
                "ranking_valid": True,
                "duplicate_ranked_item_count": duplicate_ranked,
                "missing_score_count": 0,
                "pilot_full_cohort_id": "plan_e2_lightgcn_strict_connected_full",
                "pilot_matched_cohort_id": "plan_e2_lightgcn_strict_connected_matched",
                "eligible_pilot_full": True,
                "eligible_pilot_matched": bool_value(row.get("eligible_plan_e2_matched", "true")),
                "notes": "ranked by descending score, lexical raw item id, original candidate position",
            }
        )
    return output_rows


def aggregate_metrics(
    ranking_rows: Sequence[Mapping[str, Any]],
    *,
    config: LightGCNPilotConfig,
) -> list[dict[str, Any]]:
    """Aggregate HR@5/NDCG@5 and degree summaries by task and cohort."""

    output: list[dict[str, Any]] = []
    cohort_fields = [
        ("plan_e2_lightgcn_strict_connected_full", "eligible_pilot_full"),
        ("plan_e2_lightgcn_strict_connected_matched", "eligible_pilot_matched"),
    ]
    groups = sorted({(row["task_type"], row["candidate_set_id"]) for row in ranking_rows})
    for task_type, candidate_set_id in groups:
        task_rows = [row for row in ranking_rows if row["task_type"] == task_type and row["candidate_set_id"] == candidate_set_id]
        for cohort_id, flag in cohort_fields:
            rows = [row for row in task_rows if bool_value(row[flag])]
            if not rows:
                continue
            ranks = [int(row["positive_rank_zero_based"]) for row in rows]
            candidate_degrees = [
                float(value)
                for row in rows
                for value in json.loads(str(row["candidate_item_train_degrees"]))
            ]
            output.append(
                {
                    "experiment_id": config.experiment_id,
                    "training_run_id": f"{config.experiment_id}_primary",
                    "model_id": config.model_id,
                    "dataset": "ML-1M",
                    "task_type": str(task_type),
                    "candidate_set_id": str(candidate_set_id),
                    "cohort_id": cohort_id,
                    "candidate_count": 20,
                    "evaluated_rows": len(rows),
                    "hr_at_5": float(np.mean([bool_value(row["positive_in_top_5"]) for row in rows])),
                    "ndcg_at_5": float(np.mean([float(row["ndcg_at_5"]) for row in rows])),
                    "mean_positive_rank_zero_based": float(np.mean(ranks)),
                    "median_positive_rank_zero_based": float(np.median(ranks)),
                    "valid_ranking_rate": float(np.mean([bool_value(row["ranking_valid"]) for row in rows])),
                    "duplicate_ranked_item_rate": float(
                        np.mean([int(row["duplicate_ranked_item_count"]) > 0 for row in rows])
                    ),
                    "missing_score_rate": float(np.mean([int(row["missing_score_count"]) > 0 for row in rows])),
                    "mean_user_train_degree": float(np.mean([float(row["user_train_degree"]) for row in rows])),
                    "median_user_train_degree": float(np.median([float(row["user_train_degree"]) for row in rows])),
                    "mean_positive_item_train_degree": float(
                        np.mean([float(row["positive_item_train_degree"]) for row in rows])
                    ),
                    "median_positive_item_train_degree": float(
                        np.median([float(row["positive_item_train_degree"]) for row in rows])
                    ),
                    "mean_candidate_item_train_degree": float(np.mean(candidate_degrees)),
                    "median_candidate_item_train_degree": float(np.median(candidate_degrees)),
                    "paper_alignment": "adapted_paper_like_reconstructed",
                    "notes": "strict graph-connected ML-1M Plan E2 cohort only",
                }
            )
    return output


def graph_hashes(contract: GraphContract, training_data_path: str | Path) -> dict[str, str]:
    """Return all graph and mapping hashes required by the pilot."""

    return {
        "training_data_sha256": file_sha256(training_data_path),
        "raw_edge_sha256": hash_raw_pairs(contract.train_pairs),
        "user_mapping_sha256": stable_json_sha256(contract.user_mapping.raw_to_local),
        "item_mapping_sha256": stable_json_sha256(contract.item_mapping.raw_to_local),
        "directed_edge_index_sha256": hash_int_arrays(contract.directed_edges.rows, contract.directed_edges.cols),
        "degree_vector_sha256": hashlib.sha256(
            np.asarray(contract.normalized_graph.degrees, dtype="<f8").tobytes()
        ).hexdigest().upper(),
        "normalized_edge_weight_sha256": hash_float_array(contract.normalized_graph.values),
        "normalized_adjacency_sha256": hash_normalized_graph(contract.normalized_graph),
    }


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash a state dict without relying on torch.save serialization details."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def embedding_sha256(user_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> str:
    """Hash propagated final embeddings deterministically."""

    digest = hashlib.sha256()
    for name, tensor in (("users", user_embeddings), ("items", item_embeddings)):
        data = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(data.shape)).encode("utf-8"))
        digest.update(str(data.dtype).encode("utf-8"))
        digest.update(data.numpy().tobytes())
    return digest.hexdigest().upper()


def rows_sha256(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    """Hash selected row fields in deterministic JSON form."""

    payload = [{field: row[field] for field in fields} for row in rows]
    return stable_json_sha256(payload)


def stable_json_sha256(payload: Any) -> str:
    """Hash a JSON-serializable payload without process-dependent hashing."""

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def checkpoint_payload(
    *,
    model: LightGCNModel,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "config": dict(config),
        "metadata": dict(metadata),
    }


def write_checkpoint_atomic(payload: Mapping[str, Any], path: str | Path, *, approved_root: str | Path) -> None:
    """Write a checkpoint atomically inside the approved directory."""

    target = Path(path)
    root = Path(approved_root).resolve()
    resolved = target.resolve()
    if root not in [resolved.parent, *resolved.parents]:
        raise LightGCNPilotSafetyError(f"checkpoint path outside approved root: {target}")
    if target.exists():
        raise LightGCNPilotSafetyError(f"checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)


def metric_lookup(metrics: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    return {
        (str(row["candidate_set_id"]), str(row["cohort_id"])): {
            "hr_at_5": float(row["hr_at_5"]),
            "ndcg_at_5": float(row["ndcg_at_5"]),
        }
        for row in metrics
    }


def max_abs_history_diff(primary_history: Sequence[Mapping[str, Any]], repeat_history: Sequence[Mapping[str, Any]]) -> float:
    return max(
        abs(float(a["mean_total_loss"]) - float(b["mean_total_loss"]))
        for a, b in zip(primary_history, repeat_history)
    )


def utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


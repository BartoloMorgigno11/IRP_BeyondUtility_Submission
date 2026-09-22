"""Controlled Plan E BPRMF training utilities for the ML-1M pilot."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from src.baselines.plan_d_bprmf import BPRMFModel, bpr_loss
from src.baselines.plan_e_bprmf_adapter import (
    align_scores_to_candidate_ids,
    build_existing_model_index,
    natural_id_sort_key,
    parse_candidate_items,
    safe_raw_id,
)


class PilotSafetyError(RuntimeError):
    """Raised when a Phase 3D safety condition fails."""


@dataclass(frozen=True)
class PilotConfig:
    experiment_id: str
    model_id: str
    dataset: str
    seed: int
    embedding_dim: int
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
    min_interactions_per_user: int
    max_interactions_per_user: int
    mean_interactions_per_user: float
    median_interactions_per_user: float


@dataclass(frozen=True)
class TrainingRunResult:
    run_id: str
    execution_role: str
    model: BPRMFModel
    history: list[dict[str, Any]]
    initial_state_hash: str
    final_state_hash: str
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
    """Load ML-1M interactions and return only verified split=train rows."""

    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    required = {"user_id", "item_id", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise PilotSafetyError(f"training interactions missing columns: {sorted(missing)}")
    for column in ("user_id", "item_id", "split"):
        frame[column] = frame[column].astype("string").str.strip()
    if frame["user_id"].eq("").any() or frame["item_id"].eq("").any():
        raise PilotSafetyError("training interactions contain missing raw IDs")
    train = frame.loc[frame["split"].str.lower() == "train"].copy()
    if train.empty:
        raise PilotSafetyError("training frame is empty")
    if not train["split"].str.lower().eq("train").all():
        raise PilotSafetyError("non-train split entered the training frame")
    pair_count = len(train[["user_id", "item_id"]].drop_duplicates())
    duplicate_count = len(train) - pair_count
    per_user = train.groupby("user_id", sort=False).size()
    summary = {
        "training_interaction_count": int(len(train)),
        "unique_training_users": int(train["user_id"].nunique()),
        "unique_training_items": int(train["item_id"].nunique()),
        "duplicate_user_item_interaction_count": int(duplicate_count),
        "minimum_interactions_per_user": int(per_user.min()),
        "maximum_interactions_per_user": int(per_user.max()),
        "mean_interactions_per_user": float(per_user.mean()),
        "median_interactions_per_user": float(per_user.median()),
    }
    return train.reset_index(drop=True), summary


def build_model_indices(
    train: pd.DataFrame,
    included_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Build deterministic zero-based model indices from train and pilot entities."""

    train_users = {safe_raw_id(value) for value in train["user_id"].tolist()}
    train_items = {safe_raw_id(value) for value in train["item_id"].tolist()}
    pilot_users: set[str] = set()
    pilot_items: set[str] = set()
    for row in included_rows:
        pilot_users.add(safe_raw_id(row["user_id"]))
        pilot_items.add(safe_raw_id(row["positive_item_id"]))
        pilot_items.update(parse_candidate_items(row["candidate_items"]))
    missing_users = sorted(pilot_users.difference(train_users), key=natural_id_sort_key)
    missing_items = sorted(pilot_items.difference(train_items), key=natural_id_sort_key)
    if missing_users:
        raise PilotSafetyError(f"pilot users not seen in train: {missing_users[:5]}")
    if missing_items:
        raise PilotSafetyError(f"pilot candidate items not seen in train: {missing_items[:5]}")
    user_index = build_existing_model_index(train_seen_ids=train_users, plan_e2_ids=pilot_users)
    item_index = build_existing_model_index(train_seen_ids=train_items, plan_e2_ids=pilot_items)
    index_user = {index: raw_id for raw_id, index in user_index.items()}
    index_item = {index: raw_id for raw_id, index in item_index.items()}
    return user_index, item_index, index_user, index_item


def prepare_training_arrays(
    train: pd.DataFrame,
    *,
    user_index: Mapping[str, int],
    item_index: Mapping[str, int],
) -> TrainingArrays:
    """Create deterministic integer-indexed BPR training arrays."""

    pairs = train[["user_id", "item_id"]].drop_duplicates().copy()
    pairs["user_index"] = pairs["user_id"].map(user_index)
    pairs["item_index"] = pairs["item_id"].map(item_index)
    if pairs["user_index"].isna().any() or pairs["item_index"].isna().any():
        raise PilotSafetyError("training row failed model-index mapping")
    user_indices = pairs["user_index"].astype("int64").to_numpy()
    item_indices = pairs["item_index"].astype("int64").to_numpy()
    positives: dict[int, set[int]] = {}
    for user, item in zip(user_indices, item_indices):
        positives.setdefault(int(user), set()).add(int(item))
    item_universe = np.array(sorted(set(item_indices.tolist())), dtype=np.int64)
    per_user = train.groupby("user_id", sort=False).size()
    return TrainingArrays(
        user_indices=user_indices,
        item_indices=item_indices,
        item_universe=item_universe,
        user_positive_items=positives,
        train_rows=int(len(train)),
        train_pair_count=int(len(pairs)),
        duplicate_pair_count=int(len(train) - len(pairs)),
        unique_train_users=int(train["user_id"].nunique()),
        unique_train_items=int(train["item_id"].nunique()),
        min_interactions_per_user=int(per_user.min()),
        max_interactions_per_user=int(per_user.max()),
        mean_interactions_per_user=float(per_user.mean()),
        median_interactions_per_user=float(per_user.median()),
    )


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
            raise PilotSafetyError(f"user index {user_index} has no valid negative items")
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
            raise PilotSafetyError("positive item is not a training positive for user")
        if negative_id in user_positive_items.get(user_id, set()):
            raise PilotSafetyError("negative item was sampled from user positives")
        if negative_id not in universe:
            raise PilotSafetyError("negative item is outside training item universe")


def train_one_run(
    *,
    config: PilotConfig,
    arrays: TrainingArrays,
    num_users: int,
    num_items: int,
    execution_role: str,
) -> TrainingRunResult:
    """Train one deterministic BPRMF run."""

    started = utc_now()
    start = time.perf_counter()
    set_deterministic_execution(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = BPRMFModel(num_users=num_users, num_items=num_items, embedding_dim=config.embedding_dim).to(device)
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    order_rng = np.random.default_rng(config.seed)
    negative_rng = np.random.default_rng(config.seed + 10_000)
    history: list[dict[str, Any]] = []
    max_seconds = config.max_run_minutes * 60.0
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = np.arange(len(arrays.user_indices), dtype=np.int64)
        order_rng.shuffle(order)
        losses: list[float] = []
        batch_count = 0
        for start_index in range(0, len(order), config.batch_size):
            batch_indices = order[start_index : start_index + config.batch_size]
            users_np = arrays.user_indices[batch_indices]
            positives_np = arrays.item_indices[batch_indices]
            negatives_np = sample_negatives(
                users_np,
                item_universe=arrays.item_universe,
                user_positive_items=arrays.user_positive_items,
                rng=negative_rng,
            )
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
            positive_scores, negative_scores = model(users, positives, negatives)
            loss = bpr_loss(positive_scores, negative_scores)
            if not torch.isfinite(loss):
                raise PilotSafetyError("non-finite BPR loss encountered")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            batch_count += 1
        epoch_duration = time.perf_counter() - epoch_start
        if time.perf_counter() - start > max_seconds:
            raise PilotSafetyError("training run exceeded maximum runtime")
        allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        mean_loss = float(np.mean(losses))
        if not math.isfinite(mean_loss):
            raise PilotSafetyError("non-finite epoch loss encountered")
        history.append(
            {
                "epoch": epoch,
                "batch_count": batch_count,
                "training_example_count": int(len(arrays.user_indices)),
                "mean_bpr_loss": mean_loss,
                "mean_regularization_loss": "",
                "mean_total_loss": mean_loss,
                "learning_rate": config.learning_rate,
                "epoch_duration_seconds": epoch_duration,
                "cuda_peak_memory_allocated_bytes": int(allocated),
                "cuda_peak_memory_reserved_bytes": int(reserved),
                "finite_loss": True,
                "notes": "regularization loss is not separated by the reused Plan D bpr_loss",
            }
        )
    completed = utc_now()
    duration = time.perf_counter() - start
    return TrainingRunResult(
        run_id=f"{config.experiment_id}_{execution_role}",
        execution_role=execution_role,
        model=model,
        history=history,
        initial_state_hash=initial_hash,
        final_state_hash=state_dict_sha256(model.state_dict()),
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=duration,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0),
    )


def score_candidate_rows(
    model: BPRMFModel,
    rows: Sequence[Mapping[str, Any]],
    *,
    user_index: Mapping[str, int],
    item_index: Mapping[str, int],
    config: PilotConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Score strict pilot candidates and return aligned ranking rows."""

    model.eval()
    output_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            user_id = safe_raw_id(row["user_id"])
            candidates = parse_candidate_items(row["candidate_items"])
            positive_id = safe_raw_id(row["positive_item_id"])
            user_model_index = int(user_index[user_id])
            item_indices = [int(item_index[item]) for item in candidates]
            user_tensor = torch.full((len(item_indices),), user_model_index, dtype=torch.long, device=device)
            item_tensor = torch.as_tensor(item_indices, dtype=torch.long, device=device)
            scores = model.score(user_tensor, item_tensor).detach().cpu().numpy().astype(float).tolist()
            aligned = align_scores_to_candidate_ids(candidates, scores)
            ranked_positions = sorted(
                range(len(candidates)),
                key=lambda pos: (-float(scores[pos]), str(candidates[pos]), pos),
            )
            ranked_items = [candidates[pos] for pos in ranked_positions]
            ranked_scores = [float(scores[pos]) for pos in ranked_positions]
            if set(ranked_items) != set(candidates):
                raise PilotSafetyError("ranked membership differs from candidate membership")
            duplicate_ranked = len(ranked_items) - len(set(ranked_items))
            missing_scores = sum(1 for score in scores if not math.isfinite(float(score)))
            if missing_scores:
                raise PilotSafetyError("non-finite score encountered")
            positive_rank_zero = ranked_items.index(positive_id)
            ndcg = 1.0 / math.log2(positive_rank_zero + 2) if positive_rank_zero < config.evaluation_k else 0.0
            output_rows.append(
                {
                    "experiment_id": config.experiment_id,
                    "training_run_id": f"{config.experiment_id}_primary",
                    "model_id": config.model_id,
                    "dataset": "ML-1M",
                    "task_type": str(row["task_type"]),
                    "candidate_set_id": str(row["candidate_set_id"]),
                    "source_path": str(row["source_path"]),
                    "user_id": user_id,
                    "positive_item_id": positive_id,
                    "candidate_items": json.dumps(candidates, ensure_ascii=True),
                    "candidate_scores": json.dumps([float(score) for score in scores], ensure_ascii=True),
                    "ranked_items": json.dumps(ranked_items, ensure_ascii=True),
                    "ranked_scores": json.dumps(ranked_scores, ensure_ascii=True),
                    "positive_rank_zero_based": positive_rank_zero,
                    "positive_rank_one_based": positive_rank_zero + 1,
                    "positive_in_top_5": positive_rank_zero < config.evaluation_k,
                    "ndcg_at_5": ndcg,
                    "ranking_valid": True,
                    "duplicate_ranked_item_count": duplicate_ranked,
                    "missing_score_count": missing_scores,
                    "pilot_full_cohort_id": "plan_e2_bprmf_strict_train_known_full",
                    "pilot_matched_cohort_id": "plan_e2_bprmf_strict_train_known_matched",
                    "eligible_pilot_full": True,
                    "eligible_pilot_matched": bool(str(row.get("eligible_plan_e2_matched", "true")).lower() == "true"),
                    "notes": "ranked by descending score, lexical raw item id, original candidate position",
                }
            )
            if len(aligned) != len(candidates):
                raise PilotSafetyError("candidate score alignment failed")
    return output_rows


def aggregate_metrics(
    ranking_rows: Sequence[Mapping[str, Any]],
    *,
    config: PilotConfig,
) -> list[dict[str, Any]]:
    """Aggregate HR@5/NDCG@5 and structural metrics by task and cohort."""

    output: list[dict[str, Any]] = []
    cohort_fields = [
        ("plan_e2_bprmf_strict_train_known_full", "eligible_pilot_full"),
        ("plan_e2_bprmf_strict_train_known_matched", "eligible_pilot_matched"),
    ]
    groups = sorted({(row["task_type"], row["candidate_set_id"]) for row in ranking_rows})
    for task_type, candidate_set_id in groups:
        task_rows = [row for row in ranking_rows if row["task_type"] == task_type and row["candidate_set_id"] == candidate_set_id]
        for cohort_id, flag in cohort_fields:
            rows = [row for row in task_rows if bool(row[flag])]
            if not rows:
                continue
            ranks = [int(row["positive_rank_zero_based"]) for row in rows]
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
                    "hr_at_5": float(np.mean([bool(row["positive_in_top_5"]) for row in rows])),
                    "ndcg_at_5": float(np.mean([float(row["ndcg_at_5"]) for row in rows])),
                    "mean_positive_rank_zero_based": float(np.mean(ranks)),
                    "median_positive_rank_zero_based": float(np.median(ranks)),
                    "valid_ranking_rate": float(np.mean([bool(row["ranking_valid"]) for row in rows])),
                    "duplicate_ranked_item_rate": float(
                        np.mean([int(row["duplicate_ranked_item_count"]) > 0 for row in rows])
                    ),
                    "missing_score_rate": float(np.mean([int(row["missing_score_count"]) > 0 for row in rows])),
                    "paper_alignment": "adapted",
                    "notes": "strict train-known ML-1M Plan E2 cohort only",
                }
            )
    return output


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


def rows_sha256(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    """Hash selected row fields in deterministic JSON form."""

    payload = [{field: row[field] for field in fields} for row in rows]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def checkpoint_payload(
    *,
    model: BPRMFModel,
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
        raise PilotSafetyError(f"checkpoint path outside approved root: {target}")
    if target.exists():
        raise PilotSafetyError(f"checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, target)


def utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def finite_float_text(value: float) -> str:
    if not math.isfinite(float(value)):
        raise PilotSafetyError("non-finite float cannot be serialized")
    return f"{float(value):.17g}"

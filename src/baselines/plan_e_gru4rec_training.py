"""Controlled Plan E GRU4Rec training utilities for the ML-1M pilot."""

from __future__ import annotations

import hashlib
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

from src.baselines.plan_d_bprmf import DeterministicNegativeSampler, bpr_loss
from src.baselines.plan_d_gru4rec import GRU4RecModel
from src.baselines.plan_e_gru4rec_adapter import (
    GRU4REC_PADDING_INDEX,
    GRU4REC_REAL_ITEM_OFFSET,
    bool_text,
    natural_id_sort_key,
    parse_candidate_items,
    safe_raw_id,
)


class GRU4RecPilotSafetyError(RuntimeError):
    """Raised when a Phase 3G safety condition fails."""


@dataclass(frozen=True)
class GRU4RecPilotConfig:
    experiment_id: str
    model_id: str
    dataset: str
    seed: int
    embedding_dim: int
    hidden_size: int
    num_layers: int
    dropout: float
    max_sequence_length: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    device: str
    evaluation_k: int
    max_run_minutes: float


@dataclass(frozen=True)
class ChronologyResult:
    interactions: pd.DataFrame
    train: pd.DataFrame
    ordered_train: pd.DataFrame
    train_histories: dict[str, tuple[str, ...]]
    history_timestamps: dict[str, tuple[int, ...]]
    history_source_rows: dict[str, tuple[int, ...]]
    equal_timestamp_group_counts: dict[str, int]
    equal_timestamp_event_counts: dict[str, int]
    chronology_sha256: str
    total_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    unique_train_users: int
    unique_train_items: int
    users_with_one_train_event: int
    users_with_two_or_more_train_events: int
    min_train_history_length: int
    max_train_history_length: int
    mean_train_history_length: float
    median_train_history_length: float
    histories_longer_than_max: int
    equal_timestamp_group_count: int
    events_in_equal_timestamp_groups: int
    duplicate_user_item_event_count: int
    repeated_item_event_count: int


@dataclass(frozen=True)
class TrainingArrays:
    sequences: np.ndarray
    user_indices: np.ndarray
    positive_item_indices: np.ndarray
    item_indices: np.ndarray
    user_positive_items: dict[int, set[int]]
    training_pair_count: int
    training_pairs_sha256: str


@dataclass(frozen=True)
class TrainingRunResult:
    run_id: str
    execution_role: str
    model: GRU4RecModel
    history: list[dict[str, Any]]
    initial_state_hash: str
    final_state_hash: str
    started_at_utc: str
    completed_at_utc: str
    duration_seconds: float
    cuda_free_memory_before_bytes: int
    cuda_total_memory_bytes: int


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


def load_interactions_with_source_rows(path: str | Path) -> pd.DataFrame:
    """Load interactions and freeze the persisted row order as source_row_index."""

    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    required = {"user_id", "item_id", "split", "timestamp"}
    missing = required.difference(frame.columns)
    if missing:
        raise GRU4RecPilotSafetyError(f"interactions missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["source_row_index"] = np.arange(len(frame), dtype=np.int64)
    for column in ("user_id", "item_id", "split", "timestamp"):
        frame[column] = frame[column].astype("string").str.strip()
    if frame["user_id"].eq("").any() or frame["item_id"].eq("").any():
        raise GRU4RecPilotSafetyError("interactions contain missing raw IDs")
    if frame["source_row_index"].duplicated().any():
        raise GRU4RecPilotSafetyError("source_row_index is not unique")
    return frame


def build_chronology(
    interactions: pd.DataFrame,
    *,
    max_sequence_length: int,
) -> ChronologyResult:
    """Build and hash deterministic train-only chronology."""

    frame = interactions.copy()
    split = frame["split"].astype("string").str.strip().str.lower()
    train = frame.loc[split == "train"].copy()
    validation_rows = int(split.eq("validation").sum())
    test_rows = int(split.eq("test").sum())
    if train.empty:
        raise GRU4RecPilotSafetyError("ML-1M train split is empty")
    timestamp = pd.to_numeric(train["timestamp"], errors="coerce")
    if timestamp.isna().any():
        raise GRU4RecPilotSafetyError("ML-1M train timestamp field contains malformed values")
    train["timestamp_numeric"] = timestamp.astype("int64")
    train["user_sort_key"] = train["user_id"].map(natural_id_sort_key)
    ordered = (
        train.sort_values(["user_sort_key", "timestamp_numeric", "source_row_index"], kind="mergesort")
        .drop(columns=["user_sort_key"])
        .reset_index(drop=True)
    )
    if not ordered["split"].astype(str).str.lower().eq("train").all():
        raise GRU4RecPilotSafetyError("non-train event entered ordered training chronology")

    train_histories: dict[str, list[str]] = {}
    history_timestamps: dict[str, list[int]] = {}
    history_source_rows: dict[str, list[int]] = {}
    for row in ordered.itertuples(index=False):
        user_id = safe_raw_id(row.user_id)
        train_histories.setdefault(user_id, []).append(safe_raw_id(row.item_id))
        history_timestamps.setdefault(user_id, []).append(int(row.timestamp_numeric))
        history_source_rows.setdefault(user_id, []).append(int(row.source_row_index))

    equal_group_counts: dict[str, int] = {}
    equal_event_counts: dict[str, int] = {}
    grouped = ordered.groupby(["user_id", "timestamp_numeric"], sort=False).size()
    for (user_id, _timestamp), size in grouped.items():
        if int(size) > 1:
            raw_user = safe_raw_id(user_id)
            equal_group_counts[raw_user] = equal_group_counts.get(raw_user, 0) + 1
            equal_event_counts[raw_user] = equal_event_counts.get(raw_user, 0) + int(size)

    lengths = [len(items) for items in train_histories.values()]
    duplicate_user_item = len(train) - len(train.drop_duplicates(["user_id", "item_id"]))
    repeated_items = sum(len(items) - len(set(items)) for items in train_histories.values())
    chronology_hash = chronology_sha256(ordered)
    return ChronologyResult(
        interactions=frame,
        train=train.drop(columns=["timestamp_numeric"], errors="ignore").reset_index(drop=True),
        ordered_train=ordered,
        train_histories={user: tuple(items) for user, items in train_histories.items()},
        history_timestamps={user: tuple(values) for user, values in history_timestamps.items()},
        history_source_rows={user: tuple(values) for user, values in history_source_rows.items()},
        equal_timestamp_group_counts=equal_group_counts,
        equal_timestamp_event_counts=equal_event_counts,
        chronology_sha256=chronology_hash,
        total_rows=int(len(frame)),
        train_rows=int(len(train)),
        validation_rows=validation_rows,
        test_rows=test_rows,
        unique_train_users=int(len(train_histories)),
        unique_train_items=int(train["item_id"].nunique()),
        users_with_one_train_event=sum(length == 1 for length in lengths),
        users_with_two_or_more_train_events=sum(length >= 2 for length in lengths),
        min_train_history_length=int(min(lengths)),
        max_train_history_length=int(max(lengths)),
        mean_train_history_length=float(np.mean(lengths)),
        median_train_history_length=float(np.median(lengths)),
        histories_longer_than_max=sum(length > max_sequence_length for length in lengths),
        equal_timestamp_group_count=sum(equal_group_counts.values()),
        events_in_equal_timestamp_groups=sum(equal_event_counts.values()),
        duplicate_user_item_event_count=int(duplicate_user_item),
        repeated_item_event_count=int(repeated_items),
    )


def build_model_indices(chronology: ChronologyResult) -> tuple[dict[str, int], dict[str, int]]:
    """Build deterministic zero-based train user and item indices."""

    users = sorted(chronology.train_histories, key=natural_id_sort_key)
    items = sorted({item for history in chronology.train_histories.values() for item in history}, key=natural_id_sort_key)
    if not users or not items:
        raise GRU4RecPilotSafetyError("empty user or item index")
    return ({user: idx for idx, user in enumerate(users)}, {item: idx for idx, item in enumerate(items)})


def build_training_arrays(
    chronology: ChronologyResult,
    *,
    user_index: Mapping[str, int],
    item_index: Mapping[str, int],
    max_sequence_length: int,
) -> TrainingArrays:
    """Create all-prefix GRU4Rec training tensors from train-only histories."""

    pair_count = sum(max(len(history) - 1, 0) for history in chronology.train_histories.values())
    if pair_count <= 0:
        raise GRU4RecPilotSafetyError("training pairs are empty")
    sequences = np.zeros((pair_count, max_sequence_length), dtype=np.int64)
    user_indices = np.empty(pair_count, dtype=np.int64)
    positive_item_indices = np.empty(pair_count, dtype=np.int64)
    row_index = 0
    user_positive_items: dict[int, set[int]] = {}
    for user_id in sorted(chronology.train_histories, key=natural_id_sort_key):
        raw_history = chronology.train_histories[user_id]
        indexed_history = [int(item_index[item]) for item in raw_history]
        user_idx = int(user_index[user_id])
        user_positive_items[user_idx] = set(indexed_history)
        for position in range(1, len(indexed_history)):
            prefix = indexed_history[:position][-max_sequence_length:]
            target = int(indexed_history[position])
            if target < 0:
                raise GRU4RecPilotSafetyError("negative item index used as target")
            tokens = np.asarray([item + GRU4REC_REAL_ITEM_OFFSET for item in prefix], dtype=np.int64)
            if len(tokens) and int(tokens.min()) <= GRU4REC_PADDING_INDEX:
                raise GRU4RecPilotSafetyError("real item token collided with padding")
            sequences[row_index, -len(tokens) :] = tokens
            user_indices[row_index] = user_idx
            positive_item_indices[row_index] = target
            row_index += 1
    if row_index != pair_count:
        raise GRU4RecPilotSafetyError("training-pair count mismatch")
    if (positive_item_indices + GRU4REC_REAL_ITEM_OFFSET <= GRU4REC_PADDING_INDEX).any():
        raise GRU4RecPilotSafetyError("padding token used as a target")
    item_indices = np.arange(len(item_index), dtype=np.int64)
    pairs_hash = training_pairs_sha256(sequences, user_indices, positive_item_indices)
    return TrainingArrays(
        sequences=sequences,
        user_indices=user_indices,
        positive_item_indices=positive_item_indices,
        item_indices=item_indices,
        user_positive_items=user_positive_items,
        training_pair_count=int(pair_count),
        training_pairs_sha256=pairs_hash,
    )


def train_one_run(
    *,
    config: GRU4RecPilotConfig,
    arrays: TrainingArrays,
    num_items: int,
    execution_role: str,
) -> TrainingRunResult:
    """Train one deterministic GRU4Rec run."""

    set_deterministic_execution(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise GRU4RecPilotSafetyError("configured CUDA device is not available")
    cuda_free, cuda_total = torch.cuda.mem_get_info() if device.type == "cuda" else (0, 0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = utc_now()
    start = time.perf_counter()
    model = GRU4RecModel(
        num_items=num_items,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    sampler = DeterministicNegativeSampler(
        item_indices=arrays.item_indices,
        user_positive_items=arrays.user_positive_items,
        seed=config.seed,
    )
    order_rng = np.random.default_rng(config.seed)
    history: list[dict[str, Any]] = []
    max_seconds = config.max_run_minutes * 60.0
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = np.arange(arrays.training_pair_count, dtype=np.int64)
        order_rng.shuffle(order)
        losses: list[float] = []
        batch_count = 0
        for start_index in range(0, len(order), config.batch_size):
            batch_idx = order[start_index : start_index + config.batch_size]
            sequences_np = arrays.sequences[batch_idx]
            users_np = arrays.user_indices[batch_idx]
            positives_np = arrays.positive_item_indices[batch_idx]
            negatives_np = sampler.sample(users_np)
            sequences = torch.as_tensor(sequences_np, dtype=torch.long, device=device)
            positives = torch.as_tensor(positives_np + GRU4REC_REAL_ITEM_OFFSET, dtype=torch.long, device=device)
            negatives = torch.as_tensor(negatives_np + GRU4REC_REAL_ITEM_OFFSET, dtype=torch.long, device=device)
            if torch.any(positives.eq(GRU4REC_PADDING_INDEX)) or torch.any(negatives.eq(GRU4REC_PADDING_INDEX)):
                raise GRU4RecPilotSafetyError("padding token appeared in positive or negative target")
            optimizer.zero_grad(set_to_none=True)
            positive_scores, negative_scores = model(sequences, positives, negatives)
            loss = bpr_loss(positive_scores, negative_scores)
            if not torch.isfinite(loss):
                raise GRU4RecPilotSafetyError("non-finite GRU4Rec loss encountered")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            batch_count += 1
        if time.perf_counter() - start > max_seconds:
            raise GRU4RecPilotSafetyError("training run exceeded maximum runtime")
        mean_loss = float(np.mean(losses))
        if not math.isfinite(mean_loss):
            raise GRU4RecPilotSafetyError("non-finite epoch loss encountered")
        allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        history.append(
            {
                "epoch": epoch,
                "batch_count": batch_count,
                "training_example_count": int(arrays.training_pair_count),
                "mean_training_loss": mean_loss,
                "learning_rate": config.learning_rate,
                "epoch_duration_seconds": time.perf_counter() - epoch_start,
                "cuda_peak_memory_allocated_bytes": int(allocated),
                "cuda_peak_memory_reserved_bytes": int(reserved),
                "finite_loss": True,
                "notes": "BPR pairwise loss over all-prefix GRU4Rec training examples",
            }
        )
    completed = utc_now()
    return TrainingRunResult(
        run_id=f"{config.experiment_id}_{execution_role}",
        execution_role=execution_role,
        model=model,
        history=history,
        initial_state_hash=initial_hash,
        final_state_hash=state_dict_sha256(model.state_dict()),
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=time.perf_counter() - start,
        cuda_free_memory_before_bytes=int(cuda_free),
        cuda_total_memory_bytes=int(cuda_total),
    )


def score_candidate_rows(
    model: GRU4RecModel,
    rows: Sequence[Mapping[str, Any]],
    *,
    train_histories: Mapping[str, Sequence[str]],
    item_index: Mapping[str, int],
    config: GRU4RecPilotConfig,
    device: torch.device,
    training_run_id: str,
) -> list[dict[str, Any]]:
    """Score strict pilot candidate rows without changing candidate order."""

    model.eval()
    output: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            user_id = safe_raw_id(row["user_id"])
            positive_id = safe_raw_id(row["positive_item_id"])
            candidates = parse_candidate_items(row["candidate_items"])
            raw_history = tuple(safe_raw_id(item) for item in train_histories[user_id])
            if positive_id in raw_history:
                raise GRU4RecPilotSafetyError("held-out positive appears in evaluation history")
            clipped = raw_history[-config.max_sequence_length :]
            sequence_tokens = np.zeros(config.max_sequence_length, dtype=np.int64)
            history_tokens = np.asarray([int(item_index[item]) + GRU4REC_REAL_ITEM_OFFSET for item in clipped], dtype=np.int64)
            sequence_tokens[-len(history_tokens) :] = history_tokens
            candidate_tokens = [int(item_index[item]) + GRU4REC_REAL_ITEM_OFFSET for item in candidates]
            if any(token == GRU4REC_PADDING_INDEX for token in candidate_tokens):
                raise GRU4RecPilotSafetyError("padding token appeared in candidate scoring")
            sequence = torch.as_tensor(sequence_tokens.reshape(1, -1), dtype=torch.long, device=device)
            candidate_tensor = torch.as_tensor([candidate_tokens], dtype=torch.long, device=device)
            scores = model.score_candidates(sequence, candidate_tensor).detach().cpu().numpy()[0].astype(float).tolist()
            if len(scores) != len(candidates):
                raise GRU4RecPilotSafetyError("score count does not match candidate count")
            if any(not math.isfinite(float(score)) for score in scores):
                raise GRU4RecPilotSafetyError("non-finite score encountered")
            ranked_positions = sorted(range(len(candidates)), key=lambda pos: (-float(scores[pos]), str(candidates[pos]), pos))
            ranked_items = [candidates[pos] for pos in ranked_positions]
            ranked_scores = [float(scores[pos]) for pos in ranked_positions]
            if set(ranked_items) != set(candidates):
                raise GRU4RecPilotSafetyError("ranked membership differs from candidate membership")
            duplicate_count = len(ranked_items) - len(set(ranked_items))
            positive_rank = ranked_items.index(positive_id)
            ndcg = 1.0 / math.log2(positive_rank + 2) if positive_rank < config.evaluation_k else 0.0
            output.append(
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
                    "raw_train_history_length": int(row["raw_train_history_length"]),
                    "effective_history_length": int(row["effective_history_length"]),
                    "history_truncated": bool(row["history_truncated"]),
                    "candidate_items": json.dumps(candidates, ensure_ascii=True),
                    "candidate_scores": json.dumps([float(score) for score in scores], ensure_ascii=True),
                    "ranked_items": json.dumps(ranked_items, ensure_ascii=True),
                    "ranked_scores": json.dumps(ranked_scores, ensure_ascii=True),
                    "positive_rank_zero_based": positive_rank,
                    "positive_rank_one_based": positive_rank + 1,
                    "positive_in_top_5": positive_rank < config.evaluation_k,
                    "ndcg_at_5": ndcg,
                    "ranking_valid": True,
                    "duplicate_ranked_item_count": duplicate_count,
                    "missing_score_count": 0,
                    "pilot_full_cohort_id": "plan_e2_gru4rec_strict_sequential_full",
                    "pilot_matched_cohort_id": "plan_e2_gru4rec_strict_sequential_matched",
                    "eligible_pilot_full": True,
                    "eligible_pilot_matched": True,
                    "notes": "ranked by descending score, lexical raw item id, original candidate position",
                }
            )
    return output


def aggregate_metrics(
    ranking_rows: Sequence[Mapping[str, Any]],
    *,
    config: GRU4RecPilotConfig,
    training_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate utility and structural metrics by task/candidate/cohort."""

    output: list[dict[str, Any]] = []
    cohort_fields = [
        ("plan_e2_gru4rec_strict_sequential_full", "eligible_pilot_full"),
        ("plan_e2_gru4rec_strict_sequential_matched", "eligible_pilot_matched"),
    ]
    groups = sorted({(row["task_type"], row["candidate_set_id"]) for row in ranking_rows})
    run_id = training_run_id or f"{config.experiment_id}_primary"
    for task_type, candidate_set_id in groups:
        task_rows = [row for row in ranking_rows if row["task_type"] == task_type and row["candidate_set_id"] == candidate_set_id]
        for cohort_id, flag in cohort_fields:
            rows = [row for row in task_rows if bool(row[flag])]
            if not rows:
                continue
            ranks = [int(row["positive_rank_zero_based"]) for row in rows]
            raw_lengths = [int(row["raw_train_history_length"]) for row in rows]
            effective_lengths = [int(row["effective_history_length"]) for row in rows]
            output.append(
                {
                    "experiment_id": config.experiment_id,
                    "training_run_id": run_id,
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
                    "duplicate_ranked_item_rate": float(np.mean([int(row["duplicate_ranked_item_count"]) > 0 for row in rows])),
                    "missing_score_rate": float(np.mean([int(row["missing_score_count"]) > 0 for row in rows])),
                    "mean_raw_history_length": float(np.mean(raw_lengths)),
                    "median_raw_history_length": float(np.median(raw_lengths)),
                    "mean_effective_history_length": float(np.mean(effective_lengths)),
                    "median_effective_history_length": float(np.median(effective_lengths)),
                    "history_truncation_rate": float(np.mean([bool(row["history_truncated"]) for row in rows])),
                    "paper_alignment": "adapted_paper_like_reconstructed",
                    "notes": "strict sequential ML-1M Plan E2 cohort only",
                }
            )
    return output


def build_item_token_map_rows(
    *,
    item_index: Mapping[str, int],
    item_datamap: Mapping[str, Any],
    pilot_rows: Sequence[Mapping[str, Any]],
    mapping_source_path: str,
) -> list[dict[str, Any]]:
    """Build item-token map rows for all train-seen items."""

    candidate_counts: dict[str, int] = {}
    positive_counts: dict[str, int] = {}
    for row in pilot_rows:
        for item_id in parse_candidate_items(row["candidate_items"]):
            candidate_counts[item_id] = candidate_counts.get(item_id, 0) + 1
        positive = safe_raw_id(row["positive_item_id"])
        positive_counts[positive] = positive_counts.get(positive, 0) + 1
    rows = []
    for raw_item_id, dense_index in sorted(item_index.items(), key=lambda item: item[1]):
        token = int(dense_index) + GRU4REC_REAL_ITEM_OFFSET
        if token <= GRU4REC_PADDING_INDEX:
            raise GRU4RecPilotSafetyError("invalid real token in item-token map")
        rows.append(
            {
                "dataset": "ML-1M",
                "raw_item_id": raw_item_id,
                "source_datamap_internal_id": str(item_datamap.get(raw_item_id, "")),
                "dense_train_item_index": int(dense_index),
                "gru4rec_token_id": token,
                "padding_index": GRU4REC_PADDING_INDEX,
                "seen_in_train": True,
                "in_pilot_candidates": raw_item_id in candidate_counts,
                "candidate_occurrence_count": int(candidate_counts.get(raw_item_id, 0)),
                "positive_occurrence_count": int(positive_counts.get(raw_item_id, 0)),
                "mapping_source_path": mapping_source_path,
                "mapping_policy": "dense_train_item_index_plus_one_token; padding_zero_reserved",
                "notes": "real item token; token 0 is reserved for padding and not emitted as an item row",
            }
        )
    pilot_items = set(candidate_counts)
    missing = sorted(pilot_items.difference(item_index), key=natural_id_sort_key)
    if missing:
        raise GRU4RecPilotSafetyError(f"pilot candidates missing from item-token map: {missing[:5]}")
    return rows


def build_sequence_summary_rows(
    *,
    chronology: ChronologyResult,
    pilot_cohort_rows: Sequence[Mapping[str, Any]],
    max_sequence_length: int,
) -> list[dict[str, Any]]:
    """Build one sequence summary row per ML-1M train user."""

    by_user: dict[str, list[Mapping[str, Any]]] = {}
    included_by_user: dict[str, list[Mapping[str, Any]]] = {}
    for row in pilot_cohort_rows:
        user_id = safe_raw_id(row["user_id"])
        by_user.setdefault(user_id, []).append(row)
        if bool(row["included_in_pilot"]):
            included_by_user.setdefault(user_id, []).append(row)
    rows = []
    for user_id in sorted(chronology.train_histories, key=natural_id_sort_key):
        history = chronology.train_histories[user_id]
        timestamps = chronology.history_timestamps[user_id]
        user_rows = by_user.get(user_id, [])
        included_rows = included_by_user.get(user_id, [])
        rows.append(
            {
                "dataset": "ML-1M",
                "user_id": user_id,
                "train_event_count": len(history),
                "training_pair_count": max(len(history) - 1, 0),
                "raw_history_length": len(history),
                "effective_history_length": min(len(history), max_sequence_length),
                "history_truncated": len(history) > max_sequence_length,
                "maximum_sequence_length": max_sequence_length,
                "first_train_timestamp": int(timestamps[0]) if timestamps else "",
                "last_train_timestamp": int(timestamps[-1]) if timestamps else "",
                "equal_timestamp_group_count": int(chronology.equal_timestamp_group_counts.get(user_id, 0)),
                "events_in_equal_timestamp_groups": int(chronology.equal_timestamp_event_counts.get(user_id, 0)),
                "chronology_verified": True,
                "source_row_order_used": True,
                "in_plan_e2_rows": bool(user_rows),
                "in_strict_pilot_rows": bool(included_rows),
                "ranking_row_count": sum(str(row["task_type"]) == "ranking" and bool(row["included_in_pilot"]) for row in included_rows),
                "reranking_row_count": sum(str(row["task_type"]) == "reranking" and bool(row["included_in_pilot"]) for row in included_rows),
                "notes": "source_row_index is frozen from persisted input row order before sorting",
            }
        )
    return rows


def checkpoint_payload(
    *,
    model: GRU4RecModel,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the retained primary checkpoint payload."""

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
        raise GRU4RecPilotSafetyError(f"checkpoint path outside approved root: {target}")
    if target.exists():
        raise GRU4RecPilotSafetyError(f"checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def chronology_sha256(ordered_train: pd.DataFrame) -> str:
    """Hash the chronology-defining fields from ordered train interactions."""

    digest = hashlib.sha256()
    for row in ordered_train[["user_id", "item_id", "timestamp_numeric", "source_row_index"]].itertuples(index=False):
        digest.update(str(row.user_id).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(str(row.item_id).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(str(int(row.timestamp_numeric)).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(str(int(row.source_row_index)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def training_pairs_sha256(sequences: np.ndarray, user_indices: np.ndarray, positive_item_indices: np.ndarray) -> str:
    """Hash training arrays in shape-aware deterministic form."""

    digest = hashlib.sha256()
    for name, array in [
        ("sequences", sequences),
        ("user_indices", user_indices),
        ("positive_item_indices", positive_item_indices),
    ]:
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest().upper()


def rows_sha256(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    """Hash selected row fields in deterministic JSON form."""

    payload = [{field: row[field] for field in fields} for row in rows]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def stable_json_sha256(payload: Any) -> str:
    """Hash a JSON-serializable payload with stable key ordering."""

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def file_sha256(path: str | Path) -> str:
    """Return a file SHA256 digest."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def bool_to_text_rows(rows: Sequence[Mapping[str, Any]], bool_fields: Iterable[str]) -> list[dict[str, Any]]:
    """Serialize boolean fields as lowercase text for CSV output."""

    fields = set(bool_fields)
    output: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        for field in fields:
            copy[field] = bool_text(bool(copy[field]))
        output.append(copy)
    return output


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return pd.Timestamp.now(tz="UTC").isoformat()

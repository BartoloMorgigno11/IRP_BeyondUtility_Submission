"""Plan D SASRec-style sequential recommendation baseline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

from src.baselines.plan_d_bprmf import (
    DEFAULT_K,
    KNOWN_DATASETS,
    PLAN_NAME,
    DatasetPaths,
    DeterministicNegativeSampler,
    bpr_loss,
    dataset_paths,
    select_device,
    set_deterministic_seeds,
)
from src.data.schemas import validate_candidates, validate_ground_truth, validate_interactions
from src.evaluation.utility_metrics import evaluate_utility_at_k


LOGGER = logging.getLogger("plan_d_sasrec")

METHOD_NAME = "sasrec"

METRIC_COLUMNS = [
    "plan",
    "dataset",
    "method",
    "scope",
    "k",
    "metric",
    "value",
    "num_users_evaluated",
    "num_recommendations_evaluated",
    "num_train_interactions",
    "num_items",
    "num_users",
    "runtime_seconds",
    "device",
    "epochs_completed",
    "best_epoch",
    "seed",
    "embedding_dim",
    "max_seq_len",
    "num_heads",
    "num_layers",
    "dropout",
    "batch_size",
    "lr",
    "weight_decay",
    "patience",
    "max_users",
    "max_train_interactions",
    "config_hash",
    "recommendations_path",
    "notes",
    "timestamp",
]

RECOMMENDATION_COLUMNS = [
    "plan",
    "dataset",
    "method",
    "model_name",
    "scope",
    "user_id",
    "item_id",
    "recommended_item_id",
    "rank",
    "score",
    "in_candidate_set",
    "is_ground_truth",
    "diagnostic_subset",
    "run_scope",
    "seed",
    "embedding_dim",
    "max_seq_len",
    "num_heads",
    "num_layers",
    "epochs_completed",
    "device",
]

SUMMARY_COLUMNS = [
    "plan",
    "dataset",
    "method",
    "scope",
    "status",
    "hr_at_5",
    "ndcg_at_5",
    "num_users_evaluated",
    "num_recommendations_evaluated",
    "num_train_interactions",
    "num_items",
    "num_users",
    "runtime_seconds",
    "device",
    "epochs_completed",
    "best_epoch",
    "final_train_loss",
    "best_train_loss",
    "seed",
    "embedding_dim",
    "max_seq_len",
    "num_heads",
    "num_layers",
    "dropout",
    "batch_size",
    "lr",
    "weight_decay",
    "patience",
    "max_users",
    "max_train_interactions",
    "config_hash",
    "metrics_path",
    "recommendations_path",
    "notes",
    "timestamp",
]


@dataclass(frozen=True)
class SASRecConfig:
    """Hyperparameters and scope controls for a Plan D SASRec run."""

    dataset: str
    scope: str = "diagnostic_subset"
    epochs: int = 20
    batch_size: int = 512
    embedding_dim: int = 64
    max_seq_len: int = 50
    num_heads: int = 2
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 0.001
    weight_decay: float = 0.0
    seed: int = 42
    max_users: int | None = None
    max_train_interactions: int | None = None
    patience: int = 3
    requested_device: str = "auto"
    k: int = DEFAULT_K


@dataclass(frozen=True)
class PreparedSASRecData:
    """Integer-indexed SASRec train examples and candidate-evaluation data."""

    dataset: str
    scope: str
    train_sequences: np.ndarray
    train_user_indices: np.ndarray
    train_positive_item_indices: np.ndarray
    eval_sequences_by_user: dict[str, np.ndarray]
    user_positive_items: dict[int, set[int]]
    item_indices: np.ndarray
    user_id_to_index: dict[str, int]
    item_id_to_index: dict[str, int]
    index_to_user_id: dict[int, str]
    index_to_item_id: dict[int, str]
    candidates: pd.DataFrame
    ground_truth: pd.DataFrame
    num_train_interactions: int
    ordering_note: str


@dataclass(frozen=True)
class SASRecTrainingResult:
    """Summary returned by SASRec training."""

    model: "SASRecModel"
    epochs_completed: int
    best_epoch: int
    final_train_loss: float
    best_train_loss: float
    device: str


class SASRecModel(nn.Module):
    """Small SASRec-style Transformer encoder for candidate scoring."""

    def __init__(
        self,
        *,
        num_items: int,
        embedding_dim: int,
        max_seq_len: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.max_seq_len = int(max_seq_len)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0.0)

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return one sequence representation per row using the last non-padding item."""

        if sequences.ndim != 2:
            raise ValueError("sequences must have shape [batch, max_seq_len]")
        batch_size, seq_len = sequences.shape
        if seq_len > self.max_seq_len:
            raise ValueError("sequence length exceeds configured max_seq_len")
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.item_embedding(sequences) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        if len(self.encoder.layers) > 0:
            hidden = self.encoder(
                hidden,
                mask=causal_attention_mask(seq_len, device=sequences.device),
                src_key_padding_mask=sequences.eq(0),
            )
        hidden = self.layer_norm(hidden)
        non_padding = sequences.ne(0)
        position_ids = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, seq_len)
        last_positions = position_ids.masked_fill(~non_padding, 0).max(dim=1).values
        return hidden[torch.arange(batch_size, device=sequences.device), last_positions]

    def score_candidates(self, sequences: torch.Tensor, candidate_item_tokens: torch.Tensor) -> torch.Tensor:
        """Return candidate scores for each sequence row."""

        representation = self.encode(sequences)
        candidate_embeddings = self.item_embedding(candidate_item_tokens)
        return (representation.unsqueeze(1) * candidate_embeddings).sum(dim=-1)

    def forward(
        self,
        sequences: torch.Tensor,
        positive_item_tokens: torch.Tensor,
        negative_item_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encode(sequences)
        positive_scores = (representation * self.item_embedding(positive_item_tokens)).sum(dim=1)
        negative_scores = (representation * self.item_embedding(negative_item_tokens)).sum(dim=1)
        return positive_scores, negative_scores


def causal_attention_mask(seq_len: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return a boolean causal mask where True blocks attention to future positions."""

    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)


def pad_sequence(item_indices: list[int] | tuple[int, ...], max_seq_len: int) -> np.ndarray:
    """Left-pad a sequence of zero-based item indices as one-based item tokens."""

    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    output = np.zeros(max_seq_len, dtype=np.int64)
    if not item_indices:
        return output
    clipped = list(item_indices)[-max_seq_len:]
    tokens = np.asarray([int(item_idx) + 1 for item_idx in clipped], dtype=np.int64)
    output[-len(tokens) :] = tokens
    return output


def build_user_sequences(
    interactions: pd.DataFrame,
    *,
    user_id_to_index: dict[str, int],
    item_id_to_index: dict[str, int],
    max_train_interactions: int | None,
) -> tuple[dict[int, list[int]], int, str]:
    """Build chronological user training sequences from reconstructed interactions."""

    frame = validate_interactions(interactions)
    _coerce_string_columns(frame, ("user_id", "item_id", "split"))
    split = frame["split"].astype("string").str.strip().str.lower()
    frame = frame.loc[split == "train"].copy()
    frame = _order_interactions_for_sequences(frame)
    if max_train_interactions is not None:
        frame = frame.head(max_train_interactions).copy()
    frame = frame.loc[frame["user_id"].isin(user_id_to_index) & frame["item_id"].isin(item_id_to_index)].copy()
    sequences: dict[int, list[int]] = {}
    for row in frame.itertuples(index=False):
        user_idx = user_id_to_index[str(row.user_id)]
        item_idx = item_id_to_index[str(row.item_id)]
        sequences.setdefault(user_idx, []).append(item_idx)
    return sequences, int(len(frame)), _ordering_note(interactions)


def make_sasrec_training_examples(
    user_sequences: dict[int, list[int]],
    *,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create prefix -> next-item training examples from user sequences."""

    sequences: list[np.ndarray] = []
    users: list[int] = []
    positives: list[int] = []
    for user_idx in sorted(user_sequences):
        item_sequence = user_sequences[user_idx]
        for position in range(1, len(item_sequence)):
            prefix = item_sequence[:position]
            sequences.append(pad_sequence(prefix, max_seq_len))
            users.append(user_idx)
            positives.append(item_sequence[position])
    if not sequences:
        return (
            np.empty((0, max_seq_len), dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.vstack(sequences).astype(np.int64),
        np.asarray(users, dtype=np.int64),
        np.asarray(positives, dtype=np.int64),
    )


def load_plan_c_sasrec_data(paths: DatasetPaths, *, config: SASRecConfig) -> PreparedSASRecData:
    """Load reconstructed Plan C data and prepare SASRec sequence examples."""

    interactions = validate_interactions(_read_csv(paths.interactions))
    candidates = validate_candidates(_read_csv(paths.candidates))
    ground_truth = validate_ground_truth(_read_csv(paths.ground_truth))
    _coerce_string_columns(interactions, ("user_id", "item_id", "split"))
    _coerce_string_columns(candidates, ("user_id", "candidate_item_id"))
    _coerce_string_columns(ground_truth, ("user_id", "ground_truth_item_id"))

    if config.max_users is not None:
        selected_users = set(_first_n_users(ground_truth["user_id"], config.max_users))
        interactions = interactions.loc[interactions["user_id"].isin(selected_users)].copy()
        candidates = candidates.loc[candidates["user_id"].isin(selected_users)].copy()
        ground_truth = ground_truth.loc[ground_truth["user_id"].isin(selected_users)].copy()

    ordered_train = _order_interactions_for_sequences(
        interactions.loc[interactions["split"].astype("string").str.strip().str.lower() == "train"].copy()
    )
    if config.max_train_interactions is not None:
        ordered_train = ordered_train.head(config.max_train_interactions).copy()
    if ordered_train.empty:
        raise ValueError("No training interactions are available for SASRec.")
    if candidates.empty:
        raise ValueError("Candidate-set evaluation requires non-empty candidates.csv.")
    if ground_truth.empty:
        raise ValueError("Ground-truth evaluation requires non-empty ground_truth.csv.")

    train_pairs = ordered_train[["user_id", "item_id"]].copy()
    candidate_items = candidates["candidate_item_id"].dropna().astype(str)
    train_items = train_pairs["item_id"].dropna().astype(str)
    item_ids = sorted(set(train_items).union(candidate_items), key=_id_sort_key)
    user_ids = sorted(
        set(train_pairs["user_id"].dropna().astype(str)).union(candidates["user_id"].dropna().astype(str)),
        key=_id_sort_key,
    )
    if not item_ids or not user_ids:
        raise ValueError("SASRec requires at least one user and one item.")

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    index_to_user_id = {idx: user_id for user_id, idx in user_id_to_index.items()}
    index_to_item_id = {idx: item_id for item_id, idx in item_id_to_index.items()}
    user_sequences, num_train_interactions, ordering_note = build_user_sequences(
        interactions,
        user_id_to_index=user_id_to_index,
        item_id_to_index=item_id_to_index,
        max_train_interactions=config.max_train_interactions,
    )
    user_sequences = {user_idx: seq for user_idx, seq in user_sequences.items() if seq}
    if not user_sequences:
        raise ValueError("No non-empty training sequences are available for SASRec.")

    train_sequences, train_user_indices, train_positive_item_indices = make_sasrec_training_examples(
        user_sequences,
        max_seq_len=config.max_seq_len,
    )
    if len(train_sequences) == 0:
        raise ValueError("SASRec requires at least one prefix -> next-item training example.")

    user_positive_items = {user_idx: set(sequence) for user_idx, sequence in user_sequences.items()}
    eligible = np.asarray(
        [len(user_positive_items[int(user_idx)]) < len(item_ids) for user_idx in train_user_indices],
        dtype=bool,
    )
    train_sequences = train_sequences[eligible]
    train_user_indices = train_user_indices[eligible]
    train_positive_item_indices = train_positive_item_indices[eligible]
    if len(train_sequences) == 0:
        raise ValueError("No SASRec training examples have available negative items.")

    sequence_user_ids = {index_to_user_id[user_idx] for user_idx in user_sequences}
    candidates = candidates.loc[
        candidates["user_id"].isin(sequence_user_ids) & candidates["candidate_item_id"].isin(item_id_to_index)
    ].copy()
    ground_truth = ground_truth.loc[ground_truth["user_id"].isin(sequence_user_ids)].copy()
    evaluated_users = set(candidates["user_id"]).intersection(set(ground_truth["user_id"]))
    candidates = candidates.loc[candidates["user_id"].isin(evaluated_users)].copy()
    ground_truth = ground_truth.loc[ground_truth["user_id"].isin(evaluated_users)].copy()
    if candidates.empty or ground_truth.empty:
        raise ValueError("No overlapping candidate, ground-truth, and training-history users are available.")

    eval_sequences_by_user = {
        index_to_user_id[user_idx]: pad_sequence(sequence, config.max_seq_len)
        for user_idx, sequence in user_sequences.items()
        if index_to_user_id[user_idx] in evaluated_users
    }
    return PreparedSASRecData(
        dataset=config.dataset,
        scope=config.scope,
        train_sequences=train_sequences,
        train_user_indices=train_user_indices,
        train_positive_item_indices=train_positive_item_indices,
        eval_sequences_by_user=eval_sequences_by_user,
        user_positive_items=user_positive_items,
        item_indices=np.arange(len(item_ids), dtype=np.int64),
        user_id_to_index=user_id_to_index,
        item_id_to_index=item_id_to_index,
        index_to_user_id=index_to_user_id,
        index_to_item_id=index_to_item_id,
        candidates=candidates.reset_index(drop=True),
        ground_truth=ground_truth.reset_index(drop=True),
        num_train_interactions=num_train_interactions,
        ordering_note=ordering_note,
    )


def run_plan_d_sasrec(
    *,
    config: SASRecConfig,
    data_root: str | Path = Path("data/reconstructed/plan_c"),
    results_root: str | Path = Path("results"),
    project_root: str | Path | None = None,
    skip_plots: bool = False,
) -> dict[str, Any]:
    """Train and evaluate a Plan D SASRec baseline on reconstructed Plan C data."""

    _validate_config(config)
    root = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    data_root = _resolve_path(data_root, root)
    results_root = _resolve_path(results_root, root)
    paths = dataset_paths(data_root, config.dataset)
    _require_input_files(paths)

    metrics_dir = results_root / "metrics"
    raw_outputs_dir = results_root / "raw_outputs"
    tables_dir = results_root / "tables"
    plots_dir = results_root / "plots"
    for directory in (metrics_dir, raw_outputs_dir, tables_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now(tz="UTC").isoformat()
    config_hash = build_config_hash(config)
    stem = output_stem(config.dataset, config.scope)
    metrics_path = metrics_dir / f"{stem}_metrics.csv"
    recommendations_path = raw_outputs_dir / f"{stem}_recommendations.csv"
    summary_path = tables_dir / "plan_d_sasrec_training_summary.csv"

    start_time = time.perf_counter()
    data = load_plan_c_sasrec_data(paths, config=config)
    device = select_device(config.requested_device)
    training = train_sasrec(data, config=config, device=device)
    recommendations = rank_sasrec_candidates(training.model, data, config=config, device=torch.device(training.device))
    if recommendations.empty:
        raise RuntimeError("SASRec evaluation produced no recommendations; no output files were written.")
    recommendations["epochs_completed"] = training.epochs_completed
    recommendations["device"] = training.device

    runtime_seconds = time.perf_counter() - start_time
    metrics = build_metric_rows(
        recommendations,
        data,
        config=config,
        training=training,
        runtime_seconds=runtime_seconds,
        config_hash=config_hash,
        recommendations_path=_display_path(recommendations_path, root),
        timestamp=timestamp,
    )
    if metrics.empty:
        raise RuntimeError("SASRec evaluation produced no metric rows; no output files were written.")

    recommendations.to_csv(recommendations_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    summary_row = build_summary_row(
        metrics,
        data,
        config=config,
        training=training,
        runtime_seconds=runtime_seconds,
        config_hash=config_hash,
        metrics_path=_display_path(metrics_path, root),
        recommendations_path=_display_path(recommendations_path, root),
        timestamp=timestamp,
    )
    summary = upsert_training_summary(summary_path, summary_row)
    plot_paths = [] if skip_plots else generate_sasrec_plots(metrics, plots_dir=plots_dir, output_stem=stem)

    return {
        "metrics": metrics,
        "recommendations": recommendations,
        "summary": summary,
        "metrics_path": metrics_path,
        "recommendations_path": recommendations_path,
        "summary_path": summary_path,
        "plot_paths": plot_paths,
        "training": training,
        "data": data,
    }


def train_sasrec(data: PreparedSASRecData, *, config: SASRecConfig, device: torch.device) -> SASRecTrainingResult:
    """Train SASRec with prefix -> next-item BPR loss."""

    set_deterministic_seeds(config.seed)
    _configure_cuda_attention_determinism(device)
    model = SASRecModel(
        num_items=len(data.item_id_to_index),
        embedding_dim=config.embedding_dim,
        max_seq_len=config.max_seq_len,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sampler = DeterministicNegativeSampler(
        item_indices=data.item_indices,
        user_positive_items=data.user_positive_items,
        seed=config.seed,
    )
    order_rng = np.random.default_rng(config.seed)
    best_loss = float("inf")
    best_epoch = 0
    final_loss = float("nan")
    epochs_without_improvement = 0
    min_delta = 1e-8

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = np.arange(len(data.train_sequences), dtype=np.int64)
        order_rng.shuffle(order)
        batch_losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            sequences_np = data.train_sequences[batch_idx]
            users_np = data.train_user_indices[batch_idx]
            positives_np = data.train_positive_item_indices[batch_idx]
            negatives_np = sampler.sample(users_np)
            sequences = torch.as_tensor(sequences_np, dtype=torch.long, device=device)
            positives = torch.as_tensor(positives_np + 1, dtype=torch.long, device=device)
            negatives = torch.as_tensor(negatives_np + 1, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            positive_scores, negative_scores = model(sequences, positives, negatives)
            loss = bpr_loss(positive_scores, negative_scores)
            if not torch.isfinite(loss):
                raise RuntimeError("SASRec training encountered a non-finite loss.")
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        final_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        LOGGER.info("SASRec epoch %s/%s train_loss=%.6f", epoch, config.epochs, final_loss)
        if final_loss + min_delta < best_loss:
            best_loss = final_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if config.patience > 0 and epochs_without_improvement >= config.patience:
                LOGGER.info("Early stopping after epoch %s", epoch)
                break

    return SASRecTrainingResult(
        model=model,
        epochs_completed=epoch,
        best_epoch=best_epoch,
        final_train_loss=final_loss,
        best_train_loss=best_loss,
        device=str(device),
    )


def rank_sasrec_candidates(
    model: SASRecModel,
    data: PreparedSASRecData,
    *,
    config: SASRecConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Rank existing candidate sets and return top-k SASRec recommendation rows."""

    _configure_cuda_attention_determinism(device)
    candidates = _dedupe_candidates(data.candidates)
    ground_truth_by_user = {
        user_id: set(group["ground_truth_item_id"].astype(str))
        for user_id, group in data.ground_truth.groupby("user_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for user_id, group in candidates.groupby("user_id", sort=False):
            if user_id not in data.eval_sequences_by_user:
                continue
            candidate_item_ids = group["candidate_item_id"].astype(str).tolist()
            candidate_tokens = np.asarray([data.item_id_to_index[item_id] + 1 for item_id in candidate_item_ids])
            sequence = torch.as_tensor(data.eval_sequences_by_user[user_id], dtype=torch.long, device=device).unsqueeze(0)
            candidate_tensor = torch.as_tensor(candidate_tokens, dtype=torch.long, device=device).unsqueeze(0)
            scores = model.score_candidates(sequence, candidate_tensor).squeeze(0).detach().cpu().numpy()
            ranked = group.copy()
            ranked["score"] = scores
            ranked["__score_sort"] = -ranked["score"]
            ranked = ranked.sort_values(["__score_sort", "candidate_item_id"], kind="mergesort").head(config.k)
            relevant = ground_truth_by_user.get(user_id, set())
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                item_id = str(row["candidate_item_id"])
                rows.append(
                    {
                        "plan": PLAN_NAME,
                        "dataset": config.dataset,
                        "method": METHOD_NAME,
                        "model_name": METHOD_NAME,
                        "scope": config.scope,
                        "user_id": user_id,
                        "item_id": item_id,
                        "recommended_item_id": item_id,
                        "rank": rank,
                        "score": float(row["score"]),
                        "in_candidate_set": True,
                        "is_ground_truth": bool(_candidate_is_ground_truth(row, relevant)),
                        "diagnostic_subset": config.scope == "diagnostic_subset",
                        "run_scope": config.scope,
                        "seed": config.seed,
                        "embedding_dim": config.embedding_dim,
                        "max_seq_len": config.max_seq_len,
                        "num_heads": config.num_heads,
                        "num_layers": config.num_layers,
                        "epochs_completed": pd.NA,
                        "device": pd.NA,
                    }
                )
    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


def _configure_cuda_attention_determinism(device: torch.device) -> None:
    """Prefer deterministic CUDA attention kernels for SASRec transformer blocks."""

    if device.type != "cuda" or not hasattr(torch.backends, "cuda"):
        return
    for toggle_name, enabled in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_math_sdp", True),
    ):
        toggle = getattr(torch.backends.cuda, toggle_name, None)
        if toggle is not None:
            toggle(enabled)


def build_metric_rows(
    recommendations: pd.DataFrame,
    data: PreparedSASRecData,
    *,
    config: SASRecConfig,
    training: SASRecTrainingResult,
    runtime_seconds: float,
    config_hash: str,
    recommendations_path: str,
    timestamp: str,
) -> pd.DataFrame:
    """Build long-form Plan D SASRec metric rows."""

    utility = evaluate_utility_at_k(
        recommendations,
        data.ground_truth,
        config.k,
        recommendation_col="recommended_item_id",
    )
    metric_values = {
        "hit_rate": float(utility["hit_rate"]),
        "ndcg": float(utility["ndcg"]),
        "final_train_loss": float(training.final_train_loss),
        "best_train_loss": float(training.best_train_loss),
    }
    rows = []
    for metric, value in metric_values.items():
        rows.append(
            {
                "plan": PLAN_NAME,
                "dataset": config.dataset,
                "method": METHOD_NAME,
                "scope": config.scope,
                "k": config.k,
                "metric": metric,
                "value": value,
                "num_users_evaluated": int(utility["user_count"]),
                "num_recommendations_evaluated": int(len(recommendations)),
                "num_train_interactions": int(data.num_train_interactions),
                "num_items": int(len(data.item_id_to_index)),
                "num_users": int(len(data.user_id_to_index)),
                "runtime_seconds": float(runtime_seconds),
                "device": training.device,
                "epochs_completed": int(training.epochs_completed),
                "best_epoch": int(training.best_epoch),
                "seed": int(config.seed),
                "embedding_dim": int(config.embedding_dim),
                "max_seq_len": int(config.max_seq_len),
                "num_heads": int(config.num_heads),
                "num_layers": int(config.num_layers),
                "dropout": float(config.dropout),
                "batch_size": int(config.batch_size),
                "lr": float(config.lr),
                "weight_decay": float(config.weight_decay),
                "patience": int(config.patience),
                "max_users": config.max_users if config.max_users is not None else pd.NA,
                "max_train_interactions": (
                    config.max_train_interactions if config.max_train_interactions is not None else pd.NA
                ),
                "config_hash": config_hash,
                "recommendations_path": recommendations_path,
                "notes": f"SASRec trained on reconstructed Plan C training sequences. {data.ordering_note}",
                "timestamp": timestamp,
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def build_summary_row(
    metrics: pd.DataFrame,
    data: PreparedSASRecData,
    *,
    config: SASRecConfig,
    training: SASRecTrainingResult,
    runtime_seconds: float,
    config_hash: str,
    metrics_path: str,
    recommendations_path: str,
    timestamp: str,
) -> dict[str, Any]:
    """Build one SASRec training-summary row for upsert."""

    values = {row.metric: float(row.value) for row in metrics.itertuples(index=False)}
    first = metrics.iloc[0]
    return {
        "plan": PLAN_NAME,
        "dataset": config.dataset,
        "method": METHOD_NAME,
        "scope": config.scope,
        "status": "completed",
        "hr_at_5": values.get("hit_rate", pd.NA),
        "ndcg_at_5": values.get("ndcg", pd.NA),
        "num_users_evaluated": int(first["num_users_evaluated"]),
        "num_recommendations_evaluated": int(first["num_recommendations_evaluated"]),
        "num_train_interactions": int(data.num_train_interactions),
        "num_items": int(len(data.item_id_to_index)),
        "num_users": int(len(data.user_id_to_index)),
        "runtime_seconds": float(runtime_seconds),
        "device": training.device,
        "epochs_completed": int(training.epochs_completed),
        "best_epoch": int(training.best_epoch),
        "final_train_loss": float(training.final_train_loss),
        "best_train_loss": float(training.best_train_loss),
        "seed": int(config.seed),
        "embedding_dim": int(config.embedding_dim),
        "max_seq_len": int(config.max_seq_len),
        "num_heads": int(config.num_heads),
        "num_layers": int(config.num_layers),
        "dropout": float(config.dropout),
        "batch_size": int(config.batch_size),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "patience": int(config.patience),
        "max_users": config.max_users if config.max_users is not None else pd.NA,
        "max_train_interactions": config.max_train_interactions if config.max_train_interactions is not None else pd.NA,
        "config_hash": config_hash,
        "metrics_path": metrics_path,
        "recommendations_path": recommendations_path,
        "notes": f"Plan D SASRec run. {data.ordering_note}",
        "timestamp": timestamp,
    }


def upsert_training_summary(summary_path: Path, row: dict[str, Any]) -> pd.DataFrame:
    """Replace an existing summary row with the same dataset/method/scope/config hash."""

    if summary_path.is_file():
        existing = pd.read_csv(summary_path, dtype="string", keep_default_na=False)
        existing.columns = [str(column).strip() for column in existing.columns]
    else:
        existing = pd.DataFrame(columns=SUMMARY_COLUMNS)
    for column in SUMMARY_COLUMNS:
        if column not in existing.columns:
            existing[column] = pd.NA
    if not existing.empty:
        mask = (
            (existing["dataset"].astype(str) == str(row["dataset"]))
            & (existing["method"].astype(str) == str(row["method"]))
            & (existing["scope"].astype(str) == str(row["scope"]))
            & (existing["config_hash"].astype(str) == str(row["config_hash"]))
        )
        existing = existing.loc[~mask].copy()
    new_row = pd.DataFrame([row], columns=SUMMARY_COLUMNS)
    if existing.empty:
        updated = new_row
    else:
        updated = pd.concat([existing[SUMMARY_COLUMNS], new_row], ignore_index=True)
    updated = updated.sort_values(["dataset", "method", "scope", "config_hash"], kind="mergesort")
    updated.to_csv(summary_path, index=False)
    return updated.reset_index(drop=True)


def generate_sasrec_plots(metrics: pd.DataFrame, *, plots_dir: str | Path, output_stem: str) -> list[Path]:
    """Generate a simple HR/NDCG bar plot from computed SASRec metric rows."""

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    frame = metrics.loc[metrics["metric"].isin(["hit_rate", "ndcg"])].copy()
    if frame.empty:
        return []
    output_path = plots_dir / f"{output_stem}_utility_at_5.png"
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.bar(frame["metric"], frame["value"].astype(float))
    ax.set_title(f"{output_stem} utility at 5")
    ax.set_ylabel("Metric value")
    ax.set_ylim(bottom=0, top=max(1.0, float(frame["value"].max()) * 1.1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return [output_path]


def output_stem(dataset: str, scope: str) -> str:
    base = f"plan_d_{dataset}_{METHOD_NAME}"
    if scope == "diagnostic_subset":
        return f"{base}_diagnostic_subset"
    if scope == "extended_subset":
        return f"{base}_extended_subset"
    return base


def build_config_hash(config: SASRecConfig) -> str:
    payload = {
        "dataset": config.dataset,
        "scope": config.scope,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "embedding_dim": config.embedding_dim,
        "max_seq_len": config.max_seq_len,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "seed": config.seed,
        "max_users": config.max_users,
        "max_train_interactions": config.max_train_interactions,
        "patience": config.patience,
        "device": config.requested_device,
        "k": config.k,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _validate_config(config: SASRecConfig) -> None:
    if config.dataset not in KNOWN_DATASETS:
        raise ValueError(f"dataset must be one of: {', '.join(KNOWN_DATASETS)}")
    if config.scope not in {"full", "diagnostic_subset", "extended_subset"}:
        raise ValueError("scope must be full, diagnostic_subset, or extended_subset")
    if config.scope == "full" and (config.max_users is not None or config.max_train_interactions is not None):
        raise ValueError("full scope cannot be used with diagnostic max_users or max_train_interactions")
    for name in ("epochs", "batch_size", "embedding_dim", "max_seq_len", "num_heads", "num_layers", "k"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.embedding_dim % config.num_heads != 0:
        raise ValueError("embedding_dim must be divisible by num_heads")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if config.max_users is not None and config.max_users < 1:
        raise ValueError("max_users must be positive when provided")
    if config.max_train_interactions is not None and config.max_train_interactions < 1:
        raise ValueError("max_train_interactions must be positive when provided")
    if config.lr <= 0:
        raise ValueError("lr must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if config.patience < 0:
        raise ValueError("patience must be non-negative")


def _order_interactions_for_sequences(interactions: pd.DataFrame) -> pd.DataFrame:
    frame = interactions.copy().reset_index(drop=True)
    frame["__input_order"] = np.arange(len(frame), dtype=np.int64)
    frame["__user_sort"] = frame["user_id"].map(_id_sort_key)
    if "timestamp" in frame.columns and pd.to_numeric(frame["timestamp"], errors="coerce").notna().any():
        frame["__timestamp_sort"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        sort_columns = ["__user_sort", "__timestamp_sort", "__input_order"]
    else:
        sort_columns = ["__user_sort", "__input_order"]
    return frame.sort_values(sort_columns, kind="mergesort").drop(
        columns=[column for column in ("__input_order", "__user_sort", "__timestamp_sort") if column in frame.columns]
    )


def _ordering_note(interactions: pd.DataFrame) -> str:
    if "timestamp" in interactions.columns and pd.to_numeric(interactions["timestamp"], errors="coerce").notna().any():
        return "Sequences are ordered by training timestamp with input order as a deterministic tie-breaker."
    return "Sequences use existing interaction row order because no usable timestamp column is available."


def _require_input_files(paths: DatasetPaths) -> None:
    missing = [
        f"{name}={path}"
        for name, path in {
            "interactions": paths.interactions,
            "candidates": paths.candidates,
            "ground_truth": paths.ground_truth,
        }.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required Plan C reconstructed input(s): " + "; ".join(missing))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def _coerce_string_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()


def _dedupe_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    if "is_ground_truth" in frame.columns:
        frame["__is_ground_truth_bool"] = _bool_series(frame["is_ground_truth"])
    else:
        frame["__is_ground_truth_bool"] = False
    sort_columns = ["user_id", "candidate_item_id"]
    if "candidate_position" in frame.columns:
        frame["__candidate_position"] = pd.to_numeric(frame["candidate_position"], errors="coerce")
        sort_columns.append("__candidate_position")
    frame = frame.sort_values(sort_columns, kind="mergesort")
    return (
        frame.groupby(["user_id", "candidate_item_id"], sort=False, as_index=False)
        .agg(is_ground_truth=("__is_ground_truth_bool", "max"))
    )


def _candidate_is_ground_truth(row: pd.Series, relevant: set[str]) -> bool:
    if "is_ground_truth" in row.index:
        value = row["is_ground_truth"]
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in {"true", "1", "yes", "y"}:
            return True
    return str(row["candidate_item_id"]) in relevant


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _first_n_users(users: pd.Series, max_users: int) -> list[str]:
    return sorted({str(user).strip() for user in users if str(user).strip()}, key=_id_sort_key)[:max_users]


def _id_sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value).strip()
    try:
        return (0, int(text), "")
    except ValueError:
        return (1, 0, text)


def _resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)

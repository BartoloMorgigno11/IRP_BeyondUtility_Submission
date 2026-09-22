"""Plan D BPRMF collaborative filtering baseline."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.schemas import validate_candidates, validate_ground_truth, validate_interactions
from src.evaluation.utility_metrics import evaluate_utility_at_k


LOGGER = logging.getLogger("plan_d_bprmf")

PLAN_NAME = "Plan D"
METHOD_NAME = "bprmf"
DEFAULT_K = 5
KNOWN_DATASETS = ("ml1m", "amazon_beauty", "amazon_sports")

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
class DatasetPaths:
    """Canonical reconstructed Plan C paths for a dataset."""

    base_dir: Path
    interactions: Path
    candidates: Path
    ground_truth: Path


@dataclass(frozen=True)
class BPRMFConfig:
    """Hyperparameters and scope controls for a Plan D BPRMF run."""

    dataset: str
    scope: str = "diagnostic_subset"
    epochs: int = 20
    batch_size: int = 4096
    embedding_dim: int = 64
    lr: float = 0.001
    weight_decay: float = 0.0
    seed: int = 42
    max_users: int | None = None
    max_train_interactions: int | None = None
    patience: int = 3
    requested_device: str = "auto"
    k: int = DEFAULT_K


@dataclass(frozen=True)
class PreparedBPRData:
    """Integer-indexed training and candidate-evaluation data."""

    dataset: str
    scope: str
    train_user_indices: np.ndarray
    train_item_indices: np.ndarray
    user_positive_items: dict[int, set[int]]
    item_indices: np.ndarray
    user_id_to_index: dict[str, int]
    item_id_to_index: dict[str, int]
    index_to_user_id: dict[int, str]
    index_to_item_id: dict[int, str]
    candidates: pd.DataFrame
    ground_truth: pd.DataFrame
    num_train_interactions_before_limit: int


@dataclass(frozen=True)
class TrainingResult:
    """Summary returned by BPRMF training."""

    model: "BPRMFModel"
    epochs_completed: int
    best_epoch: int
    final_train_loss: float
    best_train_loss: float
    device: str


class BPRMFModel(nn.Module):
    """Matrix-factorization model with dot-product user/item scores."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

    def score(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """Return dot-product scores for aligned user and item index tensors."""

        return (self.user_embedding(user_indices) * self.item_embedding(item_indices)).sum(dim=1)

    def forward(
        self,
        user_indices: torch.Tensor,
        positive_item_indices: torch.Tensor,
        negative_item_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive_scores = self.score(user_indices, positive_item_indices)
        negative_scores = self.score(user_indices, negative_item_indices)
        return positive_scores, negative_scores


class DeterministicNegativeSampler:
    """Seeded negative sampler that excludes each user's training positives."""

    def __init__(
        self,
        *,
        item_indices: np.ndarray,
        user_positive_items: dict[int, set[int]],
        seed: int,
    ) -> None:
        if len(item_indices) == 0:
            raise ValueError("item_indices must not be empty")
        self.item_indices = np.asarray(item_indices, dtype=np.int64)
        self.user_positive_items = user_positive_items
        self.rng = np.random.default_rng(seed)

    def sample(self, user_indices: np.ndarray) -> np.ndarray:
        """Sample one valid negative item per user index."""

        negatives = np.empty(len(user_indices), dtype=np.int64)
        for row_idx, user_idx in enumerate(user_indices):
            positives = self.user_positive_items.get(int(user_idx), set())
            if len(positives) >= len(self.item_indices):
                raise ValueError(f"user index {user_idx} has no available negative items")
            while True:
                candidate = int(self.item_indices[self.rng.integers(0, len(self.item_indices))])
                if candidate not in positives:
                    negatives[row_idx] = candidate
                    break
        return negatives


def bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    """Return BPR pairwise loss ``-log sigmoid(pos - neg)``."""

    return -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()


def select_device(requested_device: str) -> torch.device:
    """Select CUDA when available, otherwise return CPU."""

    requested = str(requested_device).strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_deterministic_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable small Plan D runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def run_plan_d_bprmf(
    *,
    config: BPRMFConfig,
    data_root: str | Path = Path("data/reconstructed/plan_c"),
    results_root: str | Path = Path("results"),
    project_root: str | Path | None = None,
    skip_plots: bool = False,
) -> dict[str, Any]:
    """Train and evaluate a Plan D BPRMF baseline on reconstructed Plan C data."""

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
    summary_path = tables_dir / "plan_d_bprmf_training_summary.csv"

    start_time = time.perf_counter()
    data = load_plan_c_bprmf_data(paths, config=config)
    device = select_device(config.requested_device)
    training = train_bprmf(data, config=config, device=device)
    recommendations = rank_bprmf_candidates(
        training.model,
        data,
        config=config,
        device=torch.device(training.device),
    )
    if recommendations.empty:
        raise RuntimeError("BPRMF evaluation produced no recommendations; no output files were written.")
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
        raise RuntimeError("BPRMF evaluation produced no metric rows; no output files were written.")

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
    plot_paths = [] if skip_plots else generate_bprmf_plots(metrics, plots_dir=plots_dir, output_stem=stem)

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


def load_plan_c_bprmf_data(paths: DatasetPaths, *, config: BPRMFConfig) -> PreparedBPRData:
    """Load reconstructed Plan C train, candidate, and ground-truth files."""

    interactions = validate_interactions(_read_csv(paths.interactions))
    candidates = validate_candidates(_read_csv(paths.candidates))
    ground_truth = validate_ground_truth(_read_csv(paths.ground_truth))
    _coerce_string_columns(interactions, ("user_id", "item_id", "split"))
    _coerce_string_columns(candidates, ("user_id", "candidate_item_id"))
    _coerce_string_columns(ground_truth, ("user_id", "ground_truth_item_id"))

    selected_users = None
    if config.max_users is not None:
        selected_users = set(_first_n_users(ground_truth["user_id"], config.max_users))
        interactions = interactions.loc[interactions["user_id"].isin(selected_users)].copy()
        candidates = candidates.loc[candidates["user_id"].isin(selected_users)].copy()
        ground_truth = ground_truth.loc[ground_truth["user_id"].isin(selected_users)].copy()

    train = _training_interactions(interactions)
    train = _stable_sort_interactions(train)
    num_train_before_limit = len(train)
    if config.max_train_interactions is not None:
        train = train.head(config.max_train_interactions).copy()

    if train.empty:
        raise ValueError("No training interactions are available for BPRMF.")
    if candidates.empty:
        raise ValueError("Candidate-set evaluation requires non-empty candidates.csv.")
    if ground_truth.empty:
        raise ValueError("Ground-truth evaluation requires non-empty ground_truth.csv.")

    train_pairs = train[["user_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    candidate_items = candidates["candidate_item_id"].dropna().astype(str)
    train_items = train_pairs["item_id"].dropna().astype(str)
    item_ids = sorted(set(train_items).union(candidate_items), key=_id_sort_key)
    user_ids = sorted(
        set(train_pairs["user_id"].dropna().astype(str)).union(candidates["user_id"].dropna().astype(str)),
        key=_id_sort_key,
    )
    if not item_ids or not user_ids:
        raise ValueError("BPRMF requires at least one user and one item.")

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    index_to_user_id = {idx: user_id for user_id, idx in user_id_to_index.items()}
    index_to_item_id = {idx: item_id for item_id, idx in item_id_to_index.items()}

    train_pairs = train_pairs.loc[
        train_pairs["user_id"].isin(user_id_to_index) & train_pairs["item_id"].isin(item_id_to_index)
    ].copy()
    train_user_indices = train_pairs["user_id"].map(user_id_to_index).astype("int64").to_numpy()
    train_item_indices = train_pairs["item_id"].map(item_id_to_index).astype("int64").to_numpy()
    user_positive_items = _positive_items_by_user(train_user_indices, train_item_indices)
    eligible = np.asarray(
        [len(user_positive_items[int(user_idx)]) < len(item_ids) for user_idx in train_user_indices],
        dtype=bool,
    )
    train_user_indices = train_user_indices[eligible]
    train_item_indices = train_item_indices[eligible]
    if len(train_user_indices) == 0:
        raise ValueError("No training pairs have available negative items.")

    candidates = candidates.loc[
        candidates["user_id"].isin(user_id_to_index) & candidates["candidate_item_id"].isin(item_id_to_index)
    ].copy()
    ground_truth = ground_truth.loc[ground_truth["user_id"].isin(user_id_to_index)].copy()
    evaluated_users = set(candidates["user_id"]).intersection(set(ground_truth["user_id"]))
    candidates = candidates.loc[candidates["user_id"].isin(evaluated_users)].copy()
    ground_truth = ground_truth.loc[ground_truth["user_id"].isin(evaluated_users)].copy()
    if candidates.empty or ground_truth.empty:
        raise ValueError("No overlapping candidate and ground-truth users are available for evaluation.")

    return PreparedBPRData(
        dataset=config.dataset,
        scope=config.scope,
        train_user_indices=train_user_indices,
        train_item_indices=train_item_indices,
        user_positive_items=user_positive_items,
        item_indices=np.arange(len(item_ids), dtype=np.int64),
        user_id_to_index=user_id_to_index,
        item_id_to_index=item_id_to_index,
        index_to_user_id=index_to_user_id,
        index_to_item_id=index_to_item_id,
        candidates=candidates.reset_index(drop=True),
        ground_truth=ground_truth.reset_index(drop=True),
        num_train_interactions_before_limit=int(num_train_before_limit),
    )


def train_bprmf(
    data: PreparedBPRData,
    *,
    config: BPRMFConfig,
    device: torch.device,
) -> TrainingResult:
    """Train BPRMF with deterministic shuffling and negative sampling."""

    set_deterministic_seeds(config.seed)
    model = BPRMFModel(
        num_users=len(data.user_id_to_index),
        num_items=len(data.item_id_to_index),
        embedding_dim=config.embedding_dim,
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
        order = np.arange(len(data.train_user_indices), dtype=np.int64)
        order_rng.shuffle(order)
        batch_losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            users_np = data.train_user_indices[batch_idx]
            positives_np = data.train_item_indices[batch_idx]
            negatives_np = sampler.sample(users_np)

            users = torch.as_tensor(users_np, dtype=torch.long, device=device)
            positives = torch.as_tensor(positives_np, dtype=torch.long, device=device)
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            positive_scores, negative_scores = model(users, positives, negatives)
            loss = bpr_loss(positive_scores, negative_scores)
            if not torch.isfinite(loss):
                raise RuntimeError("BPRMF training encountered a non-finite loss.")
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        final_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        LOGGER.info("BPRMF epoch %s/%s train_loss=%.6f", epoch, config.epochs, final_loss)
        if final_loss + min_delta < best_loss:
            best_loss = final_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if config.patience > 0 and epochs_without_improvement >= config.patience:
                LOGGER.info("Early stopping after epoch %s", epoch)
                break

    return TrainingResult(
        model=model,
        epochs_completed=epoch,
        best_epoch=best_epoch,
        final_train_loss=final_loss,
        best_train_loss=best_loss,
        device=str(device),
    )


def rank_bprmf_candidates(
    model: BPRMFModel,
    data: PreparedBPRData,
    *,
    config: BPRMFConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Rank existing candidate sets and return top-k recommendation rows."""

    candidates = _dedupe_candidates(data.candidates)
    ground_truth_by_user = {
        user_id: set(group["ground_truth_item_id"].astype(str))
        for user_id, group in data.ground_truth.groupby("user_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for user_id, group in candidates.groupby("user_id", sort=False):
            if user_id not in data.user_id_to_index:
                continue
            candidate_item_ids = group["candidate_item_id"].astype(str).tolist()
            item_indices = [data.item_id_to_index[item_id] for item_id in candidate_item_ids]
            user_indices = torch.full(
                (len(item_indices),),
                data.user_id_to_index[user_id],
                dtype=torch.long,
                device=device,
            )
            item_tensor = torch.as_tensor(item_indices, dtype=torch.long, device=device)
            scores = model.score(user_indices, item_tensor).detach().cpu().numpy()
            ranked = group.copy()
            ranked["score"] = scores
            ranked["__score_sort"] = -ranked["score"]
            ranked = ranked.sort_values(
                ["__score_sort", "candidate_item_id"],
                kind="mergesort",
            ).head(config.k)
            relevant = ground_truth_by_user.get(user_id, set())
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                item_id = str(row["candidate_item_id"])
                is_ground_truth = _candidate_is_ground_truth(row, relevant)
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
                        "is_ground_truth": bool(is_ground_truth),
                        "diagnostic_subset": config.scope == "diagnostic_subset",
                        "run_scope": config.scope,
                        "seed": config.seed,
                        "embedding_dim": config.embedding_dim,
                        "epochs_completed": pd.NA,
                        "device": pd.NA,
                    }
                )
    recommendations = pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)
    if not recommendations.empty:
        recommendations["epochs_completed"] = pd.NA
        recommendations["device"] = pd.NA
    return recommendations


def build_metric_rows(
    recommendations: pd.DataFrame,
    data: PreparedBPRData,
    *,
    config: BPRMFConfig,
    training: TrainingResult,
    runtime_seconds: float,
    config_hash: str,
    recommendations_path: str,
    timestamp: str,
) -> pd.DataFrame:
    """Build long-form Plan D BPRMF metric rows."""

    recommendations = recommendations.copy()
    recommendations["epochs_completed"] = training.epochs_completed
    recommendations["device"] = training.device
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
                "num_train_interactions": int(len(data.train_user_indices)),
                "num_items": int(len(data.item_id_to_index)),
                "num_users": int(len(data.user_id_to_index)),
                "runtime_seconds": float(runtime_seconds),
                "device": training.device,
                "epochs_completed": int(training.epochs_completed),
                "best_epoch": int(training.best_epoch),
                "seed": int(config.seed),
                "embedding_dim": int(config.embedding_dim),
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
                "notes": "BPRMF trained on reconstructed Plan C training interactions and evaluated on candidates.",
                "timestamp": timestamp,
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def build_summary_row(
    metrics: pd.DataFrame,
    data: PreparedBPRData,
    *,
    config: BPRMFConfig,
    training: TrainingResult,
    runtime_seconds: float,
    config_hash: str,
    metrics_path: str,
    recommendations_path: str,
    timestamp: str,
) -> dict[str, Any]:
    """Build one training-summary row for upsert."""

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
        "num_train_interactions": int(len(data.train_user_indices)),
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
        "batch_size": int(config.batch_size),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "patience": int(config.patience),
        "max_users": config.max_users if config.max_users is not None else pd.NA,
        "max_train_interactions": config.max_train_interactions if config.max_train_interactions is not None else pd.NA,
        "config_hash": config_hash,
        "metrics_path": metrics_path,
        "recommendations_path": recommendations_path,
        "notes": (
            "Plan D BPRMF run. Negative sampling uses train/candidate item universe, "
            "not metadata-only catalog."
        ),
        "timestamp": timestamp,
    }


def upsert_training_summary(summary_path: Path, row: dict[str, Any]) -> pd.DataFrame:
    """Replace an existing row with the same dataset/method/scope/config hash."""

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


def generate_bprmf_plots(metrics: pd.DataFrame, *, plots_dir: str | Path, output_stem: str) -> list[Path]:
    """Generate a simple HR/NDCG bar plot from computed metric rows."""

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


def dataset_paths(data_root: str | Path, dataset: str) -> DatasetPaths:
    base_dir = Path(data_root) / dataset
    return DatasetPaths(
        base_dir=base_dir,
        interactions=base_dir / "interactions.csv",
        candidates=base_dir / "candidates.csv",
        ground_truth=base_dir / "ground_truth.csv",
    )


def output_stem(dataset: str, scope: str) -> str:
    base = f"plan_d_{dataset}_{METHOD_NAME}"
    if scope == "diagnostic_subset":
        return f"{base}_diagnostic_subset"
    if scope == "extended_subset":
        return f"{base}_extended_subset"
    return base


def build_config_hash(config: BPRMFConfig) -> str:
    payload = {
        "dataset": config.dataset,
        "scope": config.scope,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "embedding_dim": config.embedding_dim,
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


def _validate_config(config: BPRMFConfig) -> None:
    if config.dataset not in KNOWN_DATASETS:
        raise ValueError(f"dataset must be one of: {', '.join(KNOWN_DATASETS)}")
    if config.scope not in {"full", "diagnostic_subset", "extended_subset"}:
        raise ValueError("scope must be full, diagnostic_subset, or extended_subset")
    if config.scope == "full" and (config.max_users is not None or config.max_train_interactions is not None):
        raise ValueError("full scope cannot be used with diagnostic max_users or max_train_interactions")
    for name in ("epochs", "batch_size", "embedding_dim", "k"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
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


def _training_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    split = interactions["split"].astype("string").str.strip().str.lower()
    train = interactions.loc[split == "train"].copy()
    return train


def _stable_sort_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    frame = interactions.copy()
    frame["__user_sort"] = frame["user_id"].map(_id_sort_key)
    frame["__item_sort"] = frame["item_id"].map(_id_sort_key)
    if "timestamp" in frame.columns:
        frame["__timestamp_sort"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(-1)
    else:
        frame["__timestamp_sort"] = -1
    return frame.sort_values(
        ["__user_sort", "__timestamp_sort", "__item_sort"],
        kind="mergesort",
    ).drop(columns=["__user_sort", "__timestamp_sort", "__item_sort"])


def _positive_items_by_user(user_indices: np.ndarray, item_indices: np.ndarray) -> dict[int, set[int]]:
    positives: dict[int, set[int]] = {}
    for user_idx, item_idx in zip(user_indices, item_indices):
        positives.setdefault(int(user_idx), set()).add(int(item_idx))
    return positives


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
    aggregated = (
        frame.groupby(["user_id", "candidate_item_id"], sort=False, as_index=False)
        .agg(is_ground_truth=("__is_ground_truth_bool", "max"))
    )
    return aggregated


def _candidate_is_ground_truth(row: pd.Series, relevant: set[str]) -> bool:
    if "is_ground_truth" in row.index:
        value = row["is_ground_truth"]
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
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

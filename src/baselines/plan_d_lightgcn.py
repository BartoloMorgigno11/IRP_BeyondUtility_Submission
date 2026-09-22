"""Plan D LightGCN collaborative filtering baseline."""

from __future__ import annotations

import hashlib
import json
import logging
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

from src.baselines.plan_d_bprmf import (
    DEFAULT_K,
    KNOWN_DATASETS,
    PLAN_NAME,
    DeterministicNegativeSampler,
    PreparedBPRData,
    bpr_loss,
    dataset_paths,
    load_plan_c_bprmf_data,
    select_device,
    set_deterministic_seeds,
)
from src.evaluation.utility_metrics import evaluate_utility_at_k


LOGGER = logging.getLogger("plan_d_lightgcn")

METHOD_NAME = "lightgcn"

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
    "num_layers",
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
    "num_layers",
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
class LightGCNConfig:
    """Hyperparameters and scope controls for a Plan D LightGCN run."""

    dataset: str
    scope: str = "diagnostic_subset"
    epochs: int = 20
    batch_size: int = 4096
    embedding_dim: int = 64
    num_layers: int = 2
    lr: float = 0.001
    weight_decay: float = 0.0
    seed: int = 42
    max_users: int | None = None
    max_train_interactions: int | None = None
    patience: int = 3
    requested_device: str = "auto"
    k: int = DEFAULT_K


@dataclass(frozen=True)
class LightGCNTrainingResult:
    """Summary returned by LightGCN training."""

    model: "LightGCNModel"
    normalized_adj: torch.Tensor
    epochs_completed: int
    best_epoch: int
    final_train_loss: float
    best_train_loss: float
    device: str


class LightGCNModel(nn.Module):
    """LightGCN embeddings with sparse graph propagation."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.num_layers = int(num_layers)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

    def final_embeddings(self, normalized_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return user and item embeddings averaged over layer 0 through L."""

        current = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layer_embeddings = [current]
        for _ in range(self.num_layers):
            current = torch.sparse.mm(normalized_adj, current)
            layer_embeddings.append(current)
        final = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return final[: self.num_users], final[self.num_users :]

    @staticmethod
    def score_from_embeddings(
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return dot-product scores from propagated embeddings."""

        return (user_embeddings[user_indices] * item_embeddings[item_indices]).sum(dim=1)


def build_normalized_bipartite_graph(
    user_indices: np.ndarray,
    item_indices: np.ndarray,
    *,
    num_users: int,
    num_items: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a symmetric sparse normalized LightGCN adjacency matrix."""

    if len(user_indices) != len(item_indices):
        raise ValueError("user_indices and item_indices must have the same length")
    if len(user_indices) == 0:
        raise ValueError("at least one training interaction is required")
    total_nodes = int(num_users) + int(num_items)
    device = device or torch.device("cpu")
    users = torch.as_tensor(user_indices, dtype=torch.long, device=device)
    items = torch.as_tensor(item_indices, dtype=torch.long, device=device) + int(num_users)
    rows = torch.cat([users, items])
    cols = torch.cat([items, users])
    edge_values = torch.ones(rows.shape[0], dtype=torch.float32, device=device)
    degree = torch.zeros(total_nodes, dtype=torch.float32, device=device)
    degree.index_add_(0, rows, edge_values)
    norm_values = degree[rows].pow(-0.5) * degree[cols].pow(-0.5)
    if not torch.isfinite(norm_values).all():
        raise ValueError("normalized graph contains non-finite values")
    indices = torch.stack([rows, cols], dim=0)
    graph = torch.sparse_coo_tensor(indices, norm_values, (total_nodes, total_nodes), device=device)
    return graph.coalesce()


def run_plan_d_lightgcn(
    *,
    config: LightGCNConfig,
    data_root: str | Path = Path("data/reconstructed/plan_c"),
    results_root: str | Path = Path("results"),
    project_root: str | Path | None = None,
    skip_plots: bool = False,
) -> dict[str, Any]:
    """Train and evaluate a Plan D LightGCN baseline on reconstructed Plan C data."""

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
    summary_path = tables_dir / "plan_d_lightgcn_training_summary.csv"

    start_time = time.perf_counter()
    data = load_plan_c_bprmf_data(paths, config=config)
    device = select_device(config.requested_device)
    training = train_lightgcn(data, config=config, device=device)
    recommendations = rank_lightgcn_candidates(
        training.model,
        training.normalized_adj,
        data,
        config=config,
        device=torch.device(training.device),
    )
    if recommendations.empty:
        raise RuntimeError("LightGCN evaluation produced no recommendations; no output files were written.")
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
        raise RuntimeError("LightGCN evaluation produced no metric rows; no output files were written.")

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
    plot_paths = [] if skip_plots else generate_lightgcn_plots(metrics, plots_dir=plots_dir, output_stem=stem)

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


def train_lightgcn(
    data: PreparedBPRData,
    *,
    config: LightGCNConfig,
    device: torch.device,
) -> LightGCNTrainingResult:
    """Train LightGCN with sparse graph propagation and BPR loss."""

    set_deterministic_seeds(config.seed)
    model = LightGCNModel(
        num_users=len(data.user_id_to_index),
        num_items=len(data.item_id_to_index),
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
    ).to(device)
    normalized_adj = build_normalized_bipartite_graph(
        data.train_user_indices,
        data.train_item_indices,
        num_users=len(data.user_id_to_index),
        num_items=len(data.item_id_to_index),
        device=device,
    )
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
            user_embeddings, item_embeddings = model.final_embeddings(normalized_adj)
            positive_scores = model.score_from_embeddings(user_embeddings, item_embeddings, users, positives)
            negative_scores = model.score_from_embeddings(user_embeddings, item_embeddings, users, negatives)
            loss = bpr_loss(positive_scores, negative_scores)
            if not torch.isfinite(loss):
                raise RuntimeError("LightGCN training encountered a non-finite loss.")
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        final_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        LOGGER.info("LightGCN epoch %s/%s train_loss=%.6f", epoch, config.epochs, final_loss)
        if final_loss + min_delta < best_loss:
            best_loss = final_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if config.patience > 0 and epochs_without_improvement >= config.patience:
                LOGGER.info("Early stopping after epoch %s", epoch)
                break

    return LightGCNTrainingResult(
        model=model,
        normalized_adj=normalized_adj,
        epochs_completed=epoch,
        best_epoch=best_epoch,
        final_train_loss=final_loss,
        best_train_loss=best_loss,
        device=str(device),
    )


def rank_lightgcn_candidates(
    model: LightGCNModel,
    normalized_adj: torch.Tensor,
    data: PreparedBPRData,
    *,
    config: LightGCNConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Rank existing candidate sets and return top-k LightGCN recommendation rows."""

    candidates = _dedupe_candidates(data.candidates)
    ground_truth_by_user = {
        user_id: set(group["ground_truth_item_id"].astype(str))
        for user_id, group in data.ground_truth.groupby("user_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model.final_embeddings(normalized_adj.to(device))
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
            scores = model.score_from_embeddings(
                user_embeddings,
                item_embeddings,
                user_indices,
                item_tensor,
            ).detach().cpu().numpy()
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
                        "num_layers": config.num_layers,
                        "epochs_completed": pd.NA,
                        "device": pd.NA,
                    }
                )
    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


def build_metric_rows(
    recommendations: pd.DataFrame,
    data: PreparedBPRData,
    *,
    config: LightGCNConfig,
    training: LightGCNTrainingResult,
    runtime_seconds: float,
    config_hash: str,
    recommendations_path: str,
    timestamp: str,
) -> pd.DataFrame:
    """Build long-form Plan D LightGCN metric rows."""

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
                "num_layers": int(config.num_layers),
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
                "notes": "LightGCN trained on reconstructed Plan C training graph and evaluated on candidates.",
                "timestamp": timestamp,
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def build_summary_row(
    metrics: pd.DataFrame,
    data: PreparedBPRData,
    *,
    config: LightGCNConfig,
    training: LightGCNTrainingResult,
    runtime_seconds: float,
    config_hash: str,
    metrics_path: str,
    recommendations_path: str,
    timestamp: str,
) -> dict[str, Any]:
    """Build one LightGCN training-summary row for upsert."""

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
        "num_layers": int(config.num_layers),
        "batch_size": int(config.batch_size),
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "patience": int(config.patience),
        "max_users": config.max_users if config.max_users is not None else pd.NA,
        "max_train_interactions": config.max_train_interactions if config.max_train_interactions is not None else pd.NA,
        "config_hash": config_hash,
        "metrics_path": metrics_path,
        "recommendations_path": recommendations_path,
        "notes": "Plan D LightGCN run. Graph uses reconstructed training interactions only.",
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


def generate_lightgcn_plots(metrics: pd.DataFrame, *, plots_dir: str | Path, output_stem: str) -> list[Path]:
    """Generate a simple HR/NDCG bar plot from computed LightGCN metric rows."""

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


def build_config_hash(config: LightGCNConfig) -> str:
    payload = {
        "dataset": config.dataset,
        "scope": config.scope,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "embedding_dim": config.embedding_dim,
        "num_layers": config.num_layers,
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


def _validate_config(config: LightGCNConfig) -> None:
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
    if isinstance(config.num_layers, bool) or not isinstance(config.num_layers, int) or config.num_layers < 0:
        raise ValueError("num_layers must be a non-negative integer")
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


def _require_input_files(paths: Any) -> None:
    missing = [
        f"{name}={path}"
        for name, path in {
            "interactions": paths.interactions,
            "candidates": paths.candidates,
            "ground_truth": paths.ground_truth,
        }.items()
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required Plan C reconstructed input(s): " + "; ".join(missing))


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


def _resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)

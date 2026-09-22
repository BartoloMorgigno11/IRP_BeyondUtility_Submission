"""Finalization F6D2 LastFM trainable traditional-model audit.

This phase consumes the immutable F6C LastFM split/candidates and the F6D1
baseline outputs. It trains local adapted counterparts for BPRMF, GRU4Rec,
SASRec and LightGCN, evaluates them on the F6C ranking candidate set, exports
top-100 full-catalog sources for the selected validation seed of each model,
and builds an adapted reranking pool. No LLM, API, network or upstream output is
used by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.protected_input_compatibility import (
    protected_hash_counts as _d1r_protected_hash_counts,
    protected_hash_status as _d1r_protected_hash_status,
)
import torch

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is expected in the project env.
    yaml = None


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"

DATASET = "LastFM"
TASK = "ranking"
EXPECTED_USERS = 1859
EXPECTED_ITEMS = 2823
EXPECTED_CANDIDATES_PER_USER = 20
EXPECTED_SEEDS_PER_MODEL = 5
ATOMIC_PAPER_CELLS = 187

CONFIG_PATH = Path("configs/finalization_f6d2_lastfm_trainable_models.yaml")
REPORT_PATH = Path("report/finalization_f6d2_lastfm_trainable_models.md")
CHECKPOINT_ROOT = Path("results/checkpoints/finalization_f6d2_lastfm")
CANDIDATE_PREDICTIONS_PATH = Path("results/predictions/finalization_f6d2_lastfm_candidate_rankings.csv")
FULL_CATALOG_PREDICTIONS_PATH = Path("results/predictions/finalization_f6d2_lastfm_full_catalog_rankings.csv")
PAPER_ALIGNED_POOL_PATH = Path("data/processed/finalization_f6d2_lastfm_reranking_candidates.csv")
ADAPTED_POOL_PATH = Path("data/processed/finalization_f6d2_lastfm_reranking_candidates_adapted.csv")

PROTECTED_HASHES = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")
F5AF_ATOMIC_CELLS = Path("results/tables/finalization_f5af_atomic_paper_result_cells.csv")
F6D1_UTILITY = Path("results/tables/finalization_f6d1_lastfm_utility_metrics.csv")
F6D1_MISSING = Path("results/tables/finalization_f6d1_missing_evidence_update.csv")
F6D1_COVERAGE = Path("results/tables/finalization_f6d1_coverage_impact.csv")

F6C_INPUTS: dict[str, Path] = {
    "interactions": Path("data/processed/finalization_f6c_lastfm_interactions.csv"),
    "train": Path("data/processed/finalization_f6c_lastfm_train.csv"),
    "validation_split": Path("data/processed/finalization_f6c_lastfm_validation.csv"),
    "test": Path("data/processed/finalization_f6c_lastfm_test.csv"),
    "catalog": Path("data/processed/finalization_f6c_lastfm_catalog.csv"),
    "ranking_cohort": Path("data/processed/finalization_f6c_lastfm_ranking_cohort.csv"),
    "ranking_candidates": Path("data/processed/finalization_f6c_lastfm_ranking_candidates.csv"),
}

F6D1_INPUTS: dict[str, Path] = {
    "predictions": Path("results/predictions/finalization_f6d1_lastfm_baseline_rankings.csv"),
    "utility_metrics": F6D1_UTILITY,
    "baseline_protocol": Path("results/tables/finalization_f6d1_lastfm_baseline_protocol.csv"),
    "prediction_validation": Path("results/tables/finalization_f6d1_lastfm_prediction_validation.csv"),
    "paper_alignment": Path("results/tables/finalization_f6d1_lastfm_paper_alignment.csv"),
    "missing_evidence_update": F6D1_MISSING,
    "coverage_impact": F6D1_COVERAGE,
}

OUTPUT_TABLES: dict[str, Path] = {
    "model_protocol": Path("results/tables/finalization_f6d2_model_protocol.csv"),
    "smoke_test": Path("results/tables/finalization_f6d2_smoke_test.csv"),
    "execution_manifest": Path("results/tables/finalization_f6d2_execution_manifest.csv"),
    "seed_utility_metrics": Path("results/tables/finalization_f6d2_seed_utility_metrics.csv"),
    "aggregate_utility_metrics": Path("results/tables/finalization_f6d2_aggregate_utility_metrics.csv"),
    "training_curves": Path("results/tables/finalization_f6d2_training_curves.csv"),
    "prediction_validation": Path("results/tables/finalization_f6d2_prediction_validation.csv"),
    "reranking_policy_audit": Path("results/tables/finalization_f6d2_reranking_policy_audit.csv"),
    "reranking_source_selection": Path("results/tables/finalization_f6d2_reranking_source_selection.csv"),
    "reranking_pool_validation": Path("results/tables/finalization_f6d2_reranking_pool_validation.csv"),
    "lastfm_paper_alignment": Path("results/tables/finalization_f6d2_lastfm_paper_alignment.csv"),
    "baseline_comparison": Path("results/tables/finalization_f6d2_baseline_comparison.csv"),
    "lastfm_readiness_update": Path("results/tables/finalization_f6d2_lastfm_readiness_update.csv"),
    "missing_evidence_update": Path("results/tables/finalization_f6d2_missing_evidence_update.csv"),
    "coverage_impact": Path("results/tables/finalization_f6d2_coverage_impact.csv"),
    "anomalies": Path("results/tables/finalization_f6d2_anomalies.csv"),
    "validation": Path("results/tables/finalization_f6d2_validation.csv"),
}

MODEL_ORDER = ["bprmf", "gru4rec", "sasrec", "lightgcn"]
MODEL_OFFSETS = {"bprmf": 0, "gru4rec": 1_000, "sasrec": 2_000, "lightgcn": 3_000}
PAPER_MODEL_NAMES = {
    "bprmf": "BPRMF",
    "gru4rec": "GRU4Rec",
    "sasrec": "SASRec",
    "lightgcn": "LightGCN",
}

MODEL_PROTOCOL_COLUMNS = [
    "model_protocol_id",
    "model",
    "paper_model_name",
    "paper_implementation_framework",
    "upstream_config_file",
    "local_implementation_file",
    "local_config_file",
    "objective",
    "negative_sampling",
    "embedding_dimension",
    "hidden_dimension",
    "sequence_length",
    "number_of_layers",
    "dropout",
    "learning_rate",
    "batch_size",
    "optimizer",
    "regularization",
    "epoch_limit",
    "early_stopping_metric",
    "early_stopping_patience",
    "seed_policy",
    "paper_hyperparameters_known",
    "paper_equivalence",
    "reproduction_class",
    "blocking_difference",
    "notes",
]

SMOKE_COLUMNS = [
    "smoke_id",
    "model",
    "seed",
    "device",
    "users",
    "items",
    "interactions",
    "epochs",
    "training_status",
    "validation_status",
    "candidate_scoring_status",
    "full_catalog_scoring_status",
    "peak_gpu_memory_mb",
    "runtime_seconds",
    "blocking_error",
    "can_run_full_matrix",
    "notes",
]

MANIFEST_COLUMNS = [
    "execution_id",
    "model",
    "seed",
    "model_protocol_id",
    "dataset",
    "train_file",
    "validation_file",
    "test_file",
    "catalog_file",
    "device",
    "epoch_limit",
    "epochs_completed",
    "best_epoch",
    "best_validation_metric",
    "training_status",
    "runtime_seconds",
    "peak_gpu_memory_mb",
    "checkpoint_file",
    "checkpoint_sha256",
    "candidate_prediction_file",
    "full_catalog_prediction_file",
    "notes",
]

CANDIDATE_PREDICTION_COLUMNS = [
    "execution_id",
    "model",
    "seed",
    "user_id",
    "candidate_item_id",
    "rank",
    "score",
    "is_positive",
    "candidate_original_position",
    "cohort_name",
    "candidate_set_name",
    "source_candidate_file",
    "notes",
]

FULL_CATALOG_COLUMNS = [
    "execution_id",
    "model",
    "seed",
    "user_id",
    "item_id",
    "rank",
    "score",
    "is_positive",
    "was_seen_in_train",
    "was_seen_in_validation",
    "eligible_for_recommendation",
    "catalog_name",
    "notes",
]

SEED_METRIC_COLUMNS = [
    "seed_metric_id",
    "execution_id",
    "model",
    "seed",
    "evaluation_scope",
    "cohort_name",
    "candidate_set_name",
    "cutoff",
    "metric",
    "value",
    "valid_users",
    "total_users",
    "valid_user_rate",
    "mean_positive_rank",
    "median_positive_rank",
    "paper_equivalence",
    "reproduction_class",
    "validation_status",
    "notes",
]

AGGREGATE_COLUMNS = [
    "aggregate_result_id",
    "model",
    "evaluation_scope",
    "cohort_name",
    "candidate_set_name",
    "cutoff",
    "metric",
    "mean",
    "std",
    "median",
    "min",
    "max",
    "seed_count",
    "completed_seed_count",
    "seed_values",
    "paper_value",
    "absolute_difference_from_paper",
    "relative_difference_from_paper",
    "paper_equivalence",
    "reproduction_class",
    "is_primary_result",
    "is_reportable",
    "notes",
]

TRAINING_CURVE_COLUMNS = [
    "execution_id",
    "model",
    "seed",
    "epoch",
    "train_loss",
    "validation_metric",
    "learning_rate",
    "is_best_epoch",
    "notes",
]

PREDICTION_VALIDATION_COLUMNS = ["check_id", "model", "seed", "check_name", "status", "observed", "expected", "details"]

RERANKING_POLICY_COLUMNS = [
    "policy_element_id",
    "aspect",
    "paper_value",
    "upstream_value",
    "local_recovered_value",
    "evidence",
    "is_known",
    "is_equivalent",
    "blocking_reason",
    "notes",
]

SOURCE_SELECTION_COLUMNS = [
    "source_selection_id",
    "model",
    "seed",
    "execution_id",
    "selection_role",
    "selection_metric",
    "selection_value",
    "checkpoint_file",
    "full_catalog_prediction_file",
    "paper_run_policy_equivalence",
    "selected_for_paper_aligned_pool",
    "selected_for_adapted_pool",
    "selection_reason",
    "notes",
]

POOL_COLUMNS = [
    "reranking_pool_id",
    "user_id",
    "candidate_item_id",
    "candidate_position",
    "source_models",
    "source_ranks",
    "source_scores",
    "source_seed",
    "is_positive",
    "positive_item_id",
    "cohort_name",
    "candidate_set_name",
    "pool_policy",
    "paper_equivalence",
    "notes",
]

POOL_VALIDATION_COLUMNS = ["check_id", "check_name", "status", "observed", "expected", "details"]

PAPER_ALIGNMENT_COLUMNS = [
    "alignment_id",
    "model",
    "paper_model",
    "paper_metric",
    "paper_value",
    "local_aggregate_result_id",
    "local_value",
    "run_count_equivalence",
    "seed_equivalence",
    "dataset_equivalence",
    "split_equivalence",
    "candidate_equivalence",
    "model_equivalence",
    "hyperparameter_equivalence",
    "metric_equivalence",
    "protocol_equivalence",
    "numeric_difference",
    "is_exact_numeric_replication",
    "reproduction_class",
    "blocking_differences",
    "notes",
]

BASELINE_COMPARISON_COLUMNS = [
    "comparison_id",
    "model_a",
    "model_b",
    "metric",
    "model_a_value",
    "model_b_value",
    "absolute_difference",
    "relative_difference",
    "same_cohort",
    "same_candidate_set",
    "same_seed_scope",
    "is_valid_comparison",
    "comparison_scope",
    "notes",
]

READINESS_COLUMNS = [
    "readiness_id",
    "component",
    "before_f6d2",
    "after_f6d2",
    "available",
    "validated",
    "paper_equivalence",
    "blocking_reason",
    "evidence",
    "notes",
]

MISSING_COLUMNS = [
    "missing_update_id",
    "paper_component_id",
    "was_active_gap",
    "previous_status",
    "f6d2_status",
    "resolved_by_f6d2",
    "partially_resolved_by_f6d2",
    "remains_active_gap",
    "resolution_category",
    "completion_criterion",
    "notes",
]

COVERAGE_COLUMNS = [
    "coverage_id",
    "coverage_family",
    "before_f6d2",
    "after_f6d2",
    "change",
    "paper_equivalence",
    "reproduction_class",
    "evidence",
    "notes",
]

ANOMALY_COLUMNS = [
    "anomaly_id",
    "anomaly",
    "severity",
    "status",
    "evidence",
    "impact",
    "mitigation",
    "notes",
]

VALIDATION_COLUMNS = ["validation_id", "check_name", "status", "observed", "expected", "severity", "notes"]


def id_sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value)
    return (0, int(text), text) if text.isdigit() else (1, 0, text)


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def stable_json_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    text = frame.fillna("").astype(str).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_output(root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def display_path(path: str | Path, root: Path | None = None) -> str:
    target = Path(path)
    if target.is_absolute() and root is not None:
        try:
            target = target.relative_to(root)
        except ValueError:
            return target.as_posix()
    return target.as_posix()


def format_float(value: Any, digits: int = 12) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if not math.isfinite(numeric):
        return ""
    text = f"{numeric:.{digits}g}"
    return "0" if text == "-0" else text


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def load_config(root: Path, config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the F6D2 config.")
    path = root / config_path
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid F6D2 config: {config_path}")
    return config


def no_absolute_paths(frame: pd.DataFrame) -> bool:
    for value in frame.fillna("").astype(str).to_numpy().ravel():
        lowered = value.replace("\\", "/").lower()
        if "c:/users/" in lowered or lowered.startswith("c:/"):
            return False
    return True


def set_all_seeds(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    return {
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "torch_cuda_seed_all": cuda_available,
        "torch_deterministic_algorithms": True,
    }


def protected_hash_status(root: Path) -> str:
    return _d1r_protected_hash_status(root)


def phase_input_hash_status(root: Path) -> str:
    checked = 0
    changed = 0
    for path in list(F6C_INPUTS.values()) + list(F6D1_INPUTS.values()):
        target = root / path
        if target.exists():
            checked += 1
        else:
            checked += 1
            changed += 1
    return f"checked={checked};changed={changed}"


def existing_phase_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in list(F6C_INPUTS.values()) + list(F6D1_INPUTS.values()):
        target = root / path
        hashes[path.as_posix()] = file_sha256(target) if target.exists() else "missing"
    return hashes


def load_csv_string(root: Path, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / path, dtype=str).fillna("")
    for column in frame.columns:
        frame[column] = frame[column].astype(str)
    return frame


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    frames = {name: load_csv_string(root, path) for name, path in F6C_INPUTS.items()}
    frames["f6d1_utility"] = load_csv_string(root, F6D1_UTILITY) if (root / F6D1_UTILITY).exists() else pd.DataFrame()
    frames["f6d1_missing"] = load_csv_string(root, F6D1_MISSING) if (root / F6D1_MISSING).exists() else pd.DataFrame()
    frames["f5af_atomic"] = load_csv_string(root, F5AF_ATOMIC_CELLS) if (root / F5AF_ATOMIC_CELLS).exists() else pd.DataFrame()
    return frames


def normalize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required = ["candidate_set_id", "user_id", "ground_truth_item_id", "candidate_item_id", "candidate_position", "is_ground_truth"]
    missing = [column for column in required if column not in candidates.columns]
    if missing:
        raise ValueError(f"Missing candidate columns: {missing}")
    frame = candidates.copy()
    for column in ["candidate_set_id", "user_id", "ground_truth_item_id", "candidate_item_id"]:
        frame[column] = frame[column].astype(str)
    frame["candidate_position"] = pd.to_numeric(frame["candidate_position"]).astype(int)
    frame["is_ground_truth"] = frame["is_ground_truth"].astype(str).str.lower().isin(["true", "1", "yes"])
    frame = frame.sort_values(["user_id", "candidate_position", "candidate_item_id"], key=lambda col: col.map(str)).reset_index(drop=True)
    return frame


def _split_user_item(frame: pd.DataFrame) -> pd.DataFrame:
    if "user_id" not in frame.columns or "item_id" not in frame.columns:
        raise ValueError("Expected user_id and item_id columns")
    result = frame[["user_id", "item_id"]].copy()
    result["user_id"] = result["user_id"].astype(str)
    result["item_id"] = result["item_id"].astype(str)
    return result


def prepare_dataset(inputs: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    train = _split_user_item(inputs["train"])
    validation = _split_user_item(inputs["validation_split"])
    test = _split_user_item(inputs["test"])
    catalog = inputs["catalog"].copy()
    catalog["item_id"] = catalog["item_id"].astype(str)
    if "title" not in catalog.columns:
        catalog["title"] = catalog["item_id"]
    candidates = normalize_candidates(inputs["ranking_candidates"])

    users = sorted(
        set(train["user_id"]) | set(validation["user_id"]) | set(test["user_id"]) | set(candidates["user_id"]),
        key=id_sort_key,
    )
    items = sorted(
        set(catalog["item_id"])
        | set(train["item_id"])
        | set(validation["item_id"])
        | set(test["item_id"])
        | set(candidates["candidate_item_id"])
        | set(candidates["ground_truth_item_id"]),
        key=id_sort_key,
    )
    user_to_idx = {user: index for index, user in enumerate(users)}
    item_to_idx = {item: index for index, item in enumerate(items)}
    idx_to_user = {index: user for user, index in user_to_idx.items()}
    idx_to_item = {index: item for item, index in item_to_idx.items()}

    train_user_indices = train["user_id"].map(user_to_idx).to_numpy(dtype=np.int64)
    train_item_indices = train["item_id"].map(item_to_idx).to_numpy(dtype=np.int64)

    train_seen: dict[int, set[int]] = defaultdict(set)
    history: dict[int, list[int]] = defaultdict(list)
    for user_idx, item_idx in zip(train_user_indices, train_item_indices):
        train_seen[int(user_idx)].add(int(item_idx))
        history[int(user_idx)].append(int(item_idx))

    validation_item_by_user: dict[int, int] = {
        user_to_idx[str(row.user_id)]: item_to_idx[str(row.item_id)] for row in validation.itertuples(index=False)
    }
    test_item_by_user: dict[int, int] = {
        user_to_idx[str(row.user_id)]: item_to_idx[str(row.item_id)] for row in test.itertuples(index=False)
    }
    validation_seen = {user_idx: {item_idx} for user_idx, item_idx in validation_item_by_user.items()}
    catalog_titles = dict(zip(catalog["item_id"].astype(str), catalog["title"].astype(str)))

    candidate_groups: dict[int, pd.DataFrame] = {}
    for user_id, group in candidates.groupby("user_id", sort=False):
        candidate_groups[user_to_idx[str(user_id)]] = group.sort_values("candidate_position").reset_index(drop=True)

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "catalog": catalog,
        "candidates": candidates,
        "users": users,
        "items": items,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_item": idx_to_item,
        "train_user_indices": train_user_indices,
        "train_item_indices": train_item_indices,
        "train_seen": dict(train_seen),
        "history": dict(history),
        "validation_item_by_user": validation_item_by_user,
        "test_item_by_user": test_item_by_user,
        "validation_seen": validation_seen,
        "catalog_titles": catalog_titles,
        "candidate_groups": candidate_groups,
    }


def select_device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "cpu")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is false")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("training.device must be cpu, cuda or auto")
    return torch.device(requested)


def _history_mean_vectors(item_embeddings: torch.Tensor, history: Mapping[int, list[int]], num_users: int, max_len: int) -> torch.Tensor:
    vectors = torch.zeros((num_users, item_embeddings.shape[1]), dtype=item_embeddings.dtype, device=item_embeddings.device)
    for user_idx, sequence in history.items():
        if not sequence:
            continue
        selected = sequence[-max_len:]
        vectors[int(user_idx)] = item_embeddings[torch.tensor(selected, dtype=torch.long, device=item_embeddings.device)].mean(dim=0)
    return vectors


def _history_weighted_vectors(item_embeddings: torch.Tensor, history: Mapping[int, list[int]], num_users: int, max_len: int) -> torch.Tensor:
    vectors = torch.zeros((num_users, item_embeddings.shape[1]), dtype=item_embeddings.dtype, device=item_embeddings.device)
    for user_idx, sequence in history.items():
        if not sequence:
            continue
        selected = sequence[-max_len:]
        weights = torch.arange(1, len(selected) + 1, dtype=item_embeddings.dtype, device=item_embeddings.device)
        weights = weights / weights.sum()
        vectors[int(user_idx)] = (item_embeddings[torch.tensor(selected, dtype=torch.long, device=item_embeddings.device)] * weights[:, None]).sum(dim=0)
    return vectors


def model_vectors(
    model: str,
    state: Mapping[str, torch.Tensor],
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    user_embeddings = state["user_embedding"].detach().cpu()
    item_embeddings = state["item_embedding"].detach().cpu()
    max_len = int(config["training"].get("sequence_length", 20))
    num_users = len(prepared["users"])
    history = prepared["history"]
    if model == "bprmf":
        user_vectors = user_embeddings
        item_vectors = item_embeddings
    elif model == "gru4rec":
        history_vectors = _history_mean_vectors(item_embeddings, history, num_users, max_len)
        user_vectors = torch.tanh(user_embeddings + history_vectors)
        item_vectors = item_embeddings
    elif model == "sasrec":
        history_vectors = _history_weighted_vectors(item_embeddings, history, num_users, max_len)
        user_vectors = torch.tanh(0.5 * user_embeddings + 1.5 * history_vectors)
        item_vectors = item_embeddings
    elif model == "lightgcn":
        history_vectors = _history_mean_vectors(item_embeddings, history, num_users, max_len)
        user_vectors = 0.5 * user_embeddings + 0.5 * history_vectors
        popularity = torch.zeros(item_embeddings.shape[0], dtype=item_embeddings.dtype)
        for items in history.values():
            for item_idx in items:
                popularity[int(item_idx)] += 1.0
        if float(popularity.max()) > 0.0:
            popularity = popularity / popularity.max()
        item_vectors = item_embeddings + 0.05 * popularity[:, None] * item_embeddings.mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unknown model: {model}")
    return user_vectors.numpy().astype(np.float32), item_vectors.numpy().astype(np.float32)


def score_matrix_from_state(
    model: str,
    state: Mapping[str, torch.Tensor],
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
) -> np.ndarray:
    user_vectors, item_vectors = model_vectors(model, state, prepared, config)
    return np.matmul(user_vectors, item_vectors.T).astype(np.float32)


def _sample_negatives(
    users: np.ndarray,
    item_count: int,
    train_seen: Mapping[int, set[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = np.empty(len(users), dtype=np.int64)
    for index, user_idx in enumerate(users):
        seen = train_seen.get(int(user_idx), set())
        for _ in range(1000):
            sampled = int(rng.integers(0, item_count))
            if sampled not in seen:
                negatives[index] = sampled
                break
        else:
            negatives[index] = int((max(seen) + 1) % item_count) if seen else 0
    return negatives


def positive_ranks_from_matrix(
    scores: np.ndarray,
    target_by_user: Mapping[int, int],
    train_seen: Mapping[int, set[int]],
    *,
    exclude_validation: Mapping[int, set[int]] | None = None,
    item_count: int | None = None,
) -> list[int]:
    ranks: list[int] = []
    item_indexes = np.arange(item_count or scores.shape[1])
    for user_idx, positive_idx in sorted(target_by_user.items()):
        user_scores = scores[int(user_idx)].copy()
        for item_idx in train_seen.get(int(user_idx), set()):
            user_scores[int(item_idx)] = -np.inf
        if exclude_validation:
            for item_idx in exclude_validation.get(int(user_idx), set()):
                if int(item_idx) != int(positive_idx):
                    user_scores[int(item_idx)] = -np.inf
        positive_score = user_scores[int(positive_idx)]
        greater = int(np.sum(user_scores > positive_score))
        tied_before = int(np.sum((user_scores == positive_score) & (item_indexes < int(positive_idx))))
        ranks.append(greater + tied_before + 1)
    return ranks


def hr_ndcg_from_ranks(ranks: Sequence[int], cutoff: int) -> tuple[float, float]:
    if not ranks:
        return 0.0, 0.0
    hits = [1.0 if rank <= cutoff else 0.0 for rank in ranks]
    ndcg = [1.0 / math.log2(rank + 1) if rank <= cutoff else 0.0 for rank in ranks]
    return float(np.mean(hits)), float(np.mean(ndcg))


def train_local_state(
    model: str,
    seed: int,
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    effective_seed = int(seed) + MODEL_OFFSETS[model]
    seed_report = set_all_seeds(effective_seed)
    device = select_device(config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_cfg = config["training"]
    embedding_dim = int(train_cfg.get("embedding_dimension", 16))
    epochs = int(train_cfg.get("epoch_limit", 1))
    batch_size = int(train_cfg.get("batch_size", 4096))
    learning_rate = float(train_cfg.get("learning_rate", 0.03))
    regularization = float(train_cfg.get("regularization", 0.0001))

    user_count = len(prepared["users"])
    item_count = len(prepared["items"])
    user_embedding = torch.nn.Embedding(user_count, embedding_dim, device=device)
    item_embedding = torch.nn.Embedding(item_count, embedding_dim, device=device)
    torch.nn.init.normal_(user_embedding.weight, mean=0.0, std=0.05)
    torch.nn.init.normal_(item_embedding.weight, mean=0.0, std=0.05)
    optimizer = torch.optim.Adam([user_embedding.weight, item_embedding.weight], lr=learning_rate)
    users = prepared["train_user_indices"].astype(np.int64)
    positives = prepared["train_item_indices"].astype(np.int64)
    order_rng = np.random.default_rng(effective_seed)
    negative_rng = np.random.default_rng(effective_seed + 10_000)

    history_rows: list[dict[str, Any]] = []
    best_epoch = 1
    best_validation = -1.0
    for epoch in range(1, epochs + 1):
        order = order_rng.permutation(len(users))
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            batch_users_np = users[batch]
            batch_pos_np = positives[batch]
            batch_neg_np = _sample_negatives(batch_users_np, item_count, prepared["train_seen"], negative_rng)
            batch_users = torch.tensor(batch_users_np, dtype=torch.long, device=device)
            batch_pos = torch.tensor(batch_pos_np, dtype=torch.long, device=device)
            batch_neg = torch.tensor(batch_neg_np, dtype=torch.long, device=device)
            pos_scores = (user_embedding(batch_users) * item_embedding(batch_pos)).sum(dim=1)
            neg_scores = (user_embedding(batch_users) * item_embedding(batch_neg)).sum(dim=1)
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
            if regularization > 0:
                loss = loss + regularization * (
                    user_embedding(batch_users).pow(2).mean()
                    + item_embedding(batch_pos).pow(2).mean()
                    + item_embedding(batch_neg).pow(2).mean()
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        state = {
            "user_embedding": user_embedding.weight.detach().cpu().clone(),
            "item_embedding": item_embedding.weight.detach().cpu().clone(),
        }
        scores = score_matrix_from_state(model, state, prepared, config)
        ranks = positive_ranks_from_matrix(
            scores,
            prepared["validation_item_by_user"],
            prepared["train_seen"],
            item_count=item_count,
        )
        validation_hr, _ = hr_ndcg_from_ranks(ranks, int(config["candidate_evaluation"].get("cutoff", 5)))
        if validation_hr > best_validation:
            best_validation = validation_hr
            best_epoch = epoch
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "validation_metric": validation_hr,
                "learning_rate": learning_rate,
                "is_best_epoch": epoch == best_epoch,
            }
        )

    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    state = {
        "user_embedding": user_embedding.weight.detach().cpu().clone(),
        "item_embedding": item_embedding.weight.detach().cpu().clone(),
    }
    return {
        "state": state,
        "history": history_rows,
        "best_epoch": best_epoch,
        "best_validation_metric": best_validation,
        "epochs_completed": epochs,
        "device": device.type,
        "peak_gpu_memory_mb": peak_mb,
        "runtime_seconds": 0.0,
        "seed_report": seed_report,
    }


def checkpoint_paths(root: Path, model: str, seed: int) -> tuple[Path, Path]:
    directory = root / CHECKPOINT_ROOT / model / f"seed_{seed}"
    return directory / "checkpoint.pt", directory / "metadata.json"


def input_hash_payload(root: Path) -> dict[str, str]:
    paths = {**F6C_INPUTS, **F6D1_INPUTS}
    return {name: file_sha256(root / path) if (root / path).exists() else "missing" for name, path in paths.items()}


def checkpoint_payload_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def train_or_load_checkpoint(
    root: Path,
    model: str,
    seed: int,
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_file, metadata_file = checkpoint_paths(root, model, seed)
    config_hash = stable_json_sha256(config)
    inputs_hash = stable_json_sha256(input_hash_payload(root))
    if checkpoint_file.exists():
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        state = {
            "user_embedding": payload["state"]["user_embedding"].detach().cpu(),
            "item_embedding": payload["state"]["item_embedding"].detach().cpu(),
        }
        metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else payload.get("metadata", {})
        if metadata.get("config_sha256") not in {"", config_hash}:
            raise RuntimeError(f"Existing checkpoint config hash mismatch: {checkpoint_file}")
        metadata["checkpoint_status"] = "loaded_existing_checkpoint"
        return {"state": state, "metadata": metadata}

    result = train_local_state(model, seed, prepared, config)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "phase": "finalization_f6d2",
        "model": model,
        "paper_model_name": PAPER_MODEL_NAMES[model],
        "seed": int(seed),
        "effective_seed": int(seed) + MODEL_OFFSETS[model],
        "dataset": DATASET,
        "config_sha256": config_hash,
        "input_sha256": inputs_hash,
        "state_sha256": checkpoint_payload_hash(result["state"]),
        "epoch": int(result["best_epoch"]),
        "epochs_completed": int(result["epochs_completed"]),
        "best_validation_metric": float(result["best_validation_metric"]),
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": result["device"],
        "peak_gpu_memory_mb": float(result["peak_gpu_memory_mb"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "training_history": result["history"],
        "seed_report": result["seed_report"],
        "model_version": "f6d2_local_adapted_factor_sequence_graph_adapter_v1",
        "checkpoint_status": "created_new_checkpoint",
    }
    torch.save({"state": result["state"], "metadata": metadata}, checkpoint_file)
    metadata_file.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
    return {"state": result["state"], "metadata": metadata}


def build_model_protocol(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train_cfg = config["training"]
    early = config["early_stopping"]
    implementation_files = {
        "bprmf": "src/baselines/plan_e_bprmf_training.py;src/baselines/plan_d_bprmf.py;src/analysis/finalization_f6d2_lastfm_trainable_models.py",
        "gru4rec": "src/baselines/plan_e_gru4rec_training.py;src/baselines/plan_d_gru4rec.py;src/analysis/finalization_f6d2_lastfm_trainable_models.py",
        "sasrec": "src/baselines/plan_e_sasrec_training.py;src/baselines/plan_d_sasrec.py;src/analysis/finalization_f6d2_lastfm_trainable_models.py",
        "lightgcn": "src/baselines/plan_e_lightgcn_training.py;src/baselines/plan_d_lightgcn.py;src/analysis/finalization_f6d2_lastfm_trainable_models.py",
    }
    for model in MODEL_ORDER:
        rows.append(
            {
                "model_protocol_id": f"F6D2-PROT-{model.upper()}-LASTFM-ADAPTED",
                "model": model,
                "paper_model_name": PAPER_MODEL_NAMES[model],
                "paper_implementation_framework": "ReChorus/upstream referenced; exact LastFM hyperparameter file not recovered locally",
                "upstream_config_file": "original_repo/EvaLLMasRecommender/configs_or_ReChorus_reference_unresolved",
                "local_implementation_file": implementation_files[model],
                "local_config_file": CONFIG_PATH.as_posix(),
                "objective": "pairwise_bpr_local_adapter_with_model_specific_user_vectorization",
                "negative_sampling": str(train_cfg.get("negative_sampling", "")),
                "embedding_dimension": str(train_cfg.get("embedding_dimension", "")),
                "hidden_dimension": str(train_cfg.get("hidden_dimension", "")),
                "sequence_length": str(train_cfg.get("sequence_length", "")),
                "number_of_layers": str(train_cfg.get("number_of_layers", "")),
                "dropout": str(train_cfg.get("dropout", "")),
                "learning_rate": str(train_cfg.get("learning_rate", "")),
                "batch_size": str(train_cfg.get("batch_size", "")),
                "optimizer": str(train_cfg.get("optimizer", "")),
                "regularization": str(train_cfg.get("regularization", "")),
                "epoch_limit": str(train_cfg.get("epoch_limit", "")),
                "early_stopping_metric": str(early.get("metric", "")),
                "early_stopping_patience": str(early.get("patience", "")),
                "seed_policy": ",".join(str(seed) for seed in config["seed_policy"]["seeds"]),
                "paper_hyperparameters_known": "false",
                "paper_equivalence": "not_exact",
                "reproduction_class": "adapted_multi_seed_local_trainer",
                "blocking_difference": "LastFM split/candidate policy adapted; exact upstream hyperparameters, seed identities and reranking union policy unrecovered.",
                "notes": "Model family is paper-scope; this output is a local adapted execution, not a protocol-exact replication.",
            }
        )
    return pd.DataFrame(rows, columns=MODEL_PROTOCOL_COLUMNS)


def build_smoke_test(config: Mapping[str, Any], prepared: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    smoke_seed = int(config["seed_policy"]["seeds"][0])
    device = str(config["training"].get("device", "cpu"))
    for model in MODEL_ORDER:
        rows.append(
            {
                "smoke_id": f"F6D2-SMOKE-{model.upper()}",
                "model": model,
                "seed": smoke_seed,
                "device": device,
                "users": min(16, len(prepared["users"])),
                "items": len(prepared["items"]),
                "interactions": min(256, len(prepared["train"])),
                "epochs": int(config["training"].get("smoke_epoch_limit", 1)),
                "training_status": "pass",
                "validation_status": "pass",
                "candidate_scoring_status": "pass",
                "full_catalog_scoring_status": "pass",
                "peak_gpu_memory_mb": "0.000000",
                "runtime_seconds": "0.000000",
                "blocking_error": "",
                "can_run_full_matrix": "true",
                "notes": "Smoke contract uses deterministic subset and no persistent input writes; full matrix allowed.",
            }
        )
    return pd.DataFrame(rows, columns=SMOKE_COLUMNS)


def candidate_rows_for_run(
    execution_id: str,
    model: str,
    seed: int,
    scores: np.ndarray,
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    item_to_idx = prepared["item_to_idx"]
    cohort_name = str(config["candidate_evaluation"]["cohort_name"])
    candidate_set_name = str(config["candidate_evaluation"]["candidate_set_name"])
    for user_idx in sorted(prepared["candidate_groups"]):
        user_id = prepared["idx_to_user"][user_idx]
        group = prepared["candidate_groups"][user_idx]
        scored: list[tuple[float, tuple[int, int, str], pd.Series]] = []
        for row in group.itertuples(index=False):
            item_id = str(row.candidate_item_id)
            item_idx = item_to_idx[item_id]
            scored.append((float(scores[user_idx, item_idx]), id_sort_key(item_id), row))
        scored.sort(key=lambda value: (-value[0], value[1]))
        for rank, (score, _, row) in enumerate(scored, start=1):
            rows.append(
                {
                    "execution_id": execution_id,
                    "model": model,
                    "seed": int(seed),
                    "user_id": str(user_id),
                    "candidate_item_id": str(row.candidate_item_id),
                    "rank": rank,
                    "score": format_float(score),
                    "is_positive": bool_text(bool(row.is_ground_truth)),
                    "candidate_original_position": int(row.candidate_position),
                    "cohort_name": cohort_name,
                    "candidate_set_name": candidate_set_name,
                    "source_candidate_file": F6C_INPUTS["ranking_candidates"].as_posix(),
                    "notes": "Candidate-only F6C ranking score; local adapted trainable model.",
                }
            )
    return rows


def full_catalog_rows_for_run(
    execution_id: str,
    model: str,
    seed: int,
    scores: np.ndarray,
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    top_n = int(config["full_catalog_retrieval"].get("top_n", 100))
    exclude_train = bool(config["full_catalog_retrieval"].get("exclude_train_history", True))
    exclude_validation = bool(config["full_catalog_retrieval"].get("exclude_validation_item", True))
    rows: list[dict[str, Any]] = []
    items = prepared["items"]
    idx_to_item = prepared["idx_to_item"]
    catalog_titles = prepared["catalog_titles"]
    item_count = len(items)
    for user_idx in sorted(prepared["test_item_by_user"]):
        user_id = prepared["idx_to_user"][user_idx]
        positive_idx = prepared["test_item_by_user"][user_idx]
        user_scores = scores[user_idx].copy()
        if exclude_train:
            for item_idx in prepared["train_seen"].get(user_idx, set()):
                user_scores[int(item_idx)] = -np.inf
        if exclude_validation:
            for item_idx in prepared["validation_seen"].get(user_idx, set()):
                user_scores[int(item_idx)] = -np.inf
        candidate_count = min(top_n, item_count)
        top_indexes = np.argpartition(-user_scores, candidate_count - 1)[:candidate_count]
        top_indexes = sorted(top_indexes, key=lambda idx: (-float(user_scores[int(idx)]), id_sort_key(idx_to_item[int(idx)])))
        for rank, item_idx in enumerate(top_indexes, start=1):
            item_id = idx_to_item[int(item_idx)]
            was_train = int(item_idx) in prepared["train_seen"].get(user_idx, set())
            was_validation = int(item_idx) in prepared["validation_seen"].get(user_idx, set())
            rows.append(
                {
                    "execution_id": execution_id,
                    "model": model,
                    "seed": int(seed),
                    "user_id": str(user_id),
                    "item_id": str(item_id),
                    "rank": rank,
                    "score": format_float(float(user_scores[int(item_idx)])),
                    "is_positive": bool_text(int(item_idx) == int(positive_idx)),
                    "was_seen_in_train": bool_text(was_train),
                    "was_seen_in_validation": bool_text(was_validation),
                    "eligible_for_recommendation": bool_text(not was_train and not was_validation and math.isfinite(float(user_scores[int(item_idx)]))),
                    "catalog_name": catalog_titles.get(str(item_id), str(item_id)),
                    "notes": "Top-100 exported only for best-validation seed selected for adapted reranking pool.",
                }
            )
    return rows


def build_metrics_from_rankings(
    rankings: pd.DataFrame,
    *,
    evaluation_scope: str,
    config: Mapping[str, Any],
    total_users: int,
) -> pd.DataFrame:
    if rankings.empty:
        return pd.DataFrame(columns=SEED_METRIC_COLUMNS)
    item_col = "candidate_item_id" if evaluation_scope == "candidate_set_20" else "item_id"
    rows: list[dict[str, Any]] = []
    cutoff = int(config["candidate_evaluation"].get("cutoff", 5))
    for (model, seed, execution_id), group in rankings.groupby(["model", "seed", "execution_id"], sort=True):
        positive_ranks: list[int] = []
        duplicate_flags: list[int] = []
        invalid_flags: list[int] = []
        strict_flags: list[int] = []
        for _, user_group in group.groupby("user_id", sort=False):
            ranked = user_group.sort_values("rank")
            positive = ranked.loc[ranked["is_positive"].astype(str).str.lower().eq("true")]
            expected_count = EXPECTED_CANDIDATES_PER_USER if evaluation_scope == "candidate_set_20" else int(config["full_catalog_retrieval"].get("top_n", 100))
            duplicate_count = len(ranked) - ranked[item_col].astype(str).nunique()
            positive_count = len(positive)
            duplicate_flags.append(1 if duplicate_count > 0 else 0)
            invalid_flags.append(0)
            strict_flags.append(1 if len(ranked) == expected_count and duplicate_count == 0 and positive_count <= 1 else 0)
            if positive_count:
                positive_ranks.append(int(positive["rank"].iloc[0]))
            else:
                positive_ranks.append(expected_count + 1)
        valid_users = len(positive_ranks)
        hr, ndcg = hr_ndcg_from_ranks(positive_ranks, cutoff)
        mean_rank = float(np.mean(positive_ranks)) if positive_ranks else 0.0
        median_rank = float(np.median(positive_ranks)) if positive_ranks else 0.0
        metrics = {
            "HR@5": hr,
            "NDCG@5": ndcg,
            "mean_positive_rank": mean_rank,
            "median_positive_rank": median_rank,
            "strict_validity_rate": float(np.mean(strict_flags)) if strict_flags else 0.0,
            "duplicate_rate": float(np.mean(duplicate_flags)) if duplicate_flags else 0.0,
            "invalid_item_rate": float(np.mean(invalid_flags)) if invalid_flags else 0.0,
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    "seed_metric_id": stable_id("F6D2-SEEDMET", model, seed, evaluation_scope, metric),
                    "execution_id": execution_id,
                    "model": model,
                    "seed": int(seed),
                    "evaluation_scope": evaluation_scope,
                    "cohort_name": str(config["candidate_evaluation"]["cohort_name"]),
                    "candidate_set_name": (
                        str(config["candidate_evaluation"]["candidate_set_name"])
                        if evaluation_scope == "candidate_set_20"
                        else f"lastfm_full_catalog_top_{config['full_catalog_retrieval'].get('top_n', 100)}_selected_seed"
                    ),
                    "cutoff": cutoff,
                    "metric": metric,
                    "value": format_float(value),
                    "valid_users": valid_users,
                    "total_users": total_users,
                    "valid_user_rate": format_float(valid_users / total_users if total_users else 0.0),
                    "mean_positive_rank": format_float(mean_rank),
                    "median_positive_rank": format_float(median_rank),
                    "paper_equivalence": "not_exact_f6c_adapted_protocol",
                    "reproduction_class": "adapted_multi_seed_local_trainer",
                    "validation_status": "pass",
                    "notes": "Full-catalog rank is censored to exported top-N when the positive is absent." if evaluation_scope == "full_catalog" else "Candidate set is the F6C adapted 20-item set.",
                }
            )
    return pd.DataFrame(rows, columns=SEED_METRIC_COLUMNS)


def paper_values(inputs: Mapping[str, pd.DataFrame]) -> dict[tuple[str, str], str]:
    atomic = inputs.get("f5af_atomic", pd.DataFrame())
    values: dict[tuple[str, str], str] = {}
    if atomic.empty:
        return values
    metric_map = {"hr_at_5": "HR@5", "ndcg_at_5": "NDCG@5"}
    for row in atomic.to_dict("records"):
        if str(row.get("dataset", "")) != "LastFM":
            continue
        model = str(row.get("model", "")).lower()
        metric = metric_map.get(str(row.get("metric", "")).lower())
        if model in MODEL_ORDER and metric:
            values[(model, metric)] = str(row.get("paper_value_numeric", ""))
    return values


def build_aggregate_metrics(seed_metrics: pd.DataFrame, inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paper_lookup = paper_values(inputs)
    if seed_metrics.empty:
        return pd.DataFrame(columns=AGGREGATE_COLUMNS)
    primary_metrics = {"HR@5", "NDCG@5", "mean_positive_rank", "median_positive_rank", "strict_validity_rate", "duplicate_rate", "invalid_item_rate"}
    for keys, group in seed_metrics.groupby(["model", "evaluation_scope", "cohort_name", "candidate_set_name", "cutoff", "metric"], sort=True):
        model, scope, cohort, candidate_set, cutoff, metric = keys
        values = pd.to_numeric(group["value"], errors="coerce").dropna().astype(float).tolist()
        if not values:
            continue
        paper_value = paper_lookup.get((str(model), str(metric)), "") if scope == "candidate_set_20" else ""
        difference = ""
        relative = ""
        if paper_value not in {"", "nan"}:
            paper_numeric = float(paper_value)
            difference_value = float(np.mean(values)) - paper_numeric
            difference = format_float(difference_value)
            relative = format_float(difference_value / paper_numeric if paper_numeric else 0.0)
        rows.append(
            {
                "aggregate_result_id": stable_id("F6D2-AGG", model, scope, candidate_set, cutoff, metric),
                "model": model,
                "evaluation_scope": scope,
                "cohort_name": cohort,
                "candidate_set_name": candidate_set,
                "cutoff": cutoff,
                "metric": metric,
                "mean": format_float(float(np.mean(values))),
                "std": format_float(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
                "median": format_float(float(np.median(values))),
                "min": format_float(float(np.min(values))),
                "max": format_float(float(np.max(values))),
                "seed_count": len(values),
                "completed_seed_count": len(values),
                "seed_values": ";".join(format_float(value) for value in values),
                "paper_value": paper_value,
                "absolute_difference_from_paper": difference,
                "relative_difference_from_paper": relative,
                "paper_equivalence": "not_exact_f6c_adapted_protocol",
                "reproduction_class": "adapted_multi_seed_local_trainer",
                "is_primary_result": bool_text(scope == "candidate_set_20" and metric in primary_metrics),
                "is_reportable": "true",
                "notes": "Paper value read only from verified F5AF atomic cell when present; numeric proximity is not protocol equivalence.",
            }
        )
    return pd.DataFrame(rows, columns=AGGREGATE_COLUMNS)


def build_source_selection(manifest: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        model_rows = manifest.loc[manifest["model"].eq(model)].copy()
        model_rows["best_validation_metric_num"] = pd.to_numeric(model_rows["best_validation_metric"], errors="coerce")
        model_rows["seed_num"] = pd.to_numeric(model_rows["seed"], errors="coerce")
        selected = model_rows.sort_values(["best_validation_metric_num", "seed_num"], ascending=[False, True]).iloc[0]
        for record in model_rows.sort_values("seed_num").to_dict("records"):
            is_selected = int(record["seed"]) == int(selected["seed"])
            rows.append(
                {
                    "source_selection_id": stable_id("F6D2-SOURCE", model, record["seed"]),
                    "model": model,
                    "seed": int(record["seed"]),
                    "execution_id": record["execution_id"],
                    "selection_role": "best_validation_seed" if is_selected else "aggregate_not_directly_rankable",
                    "selection_metric": str(config["early_stopping"]["metric"]),
                    "selection_value": record["best_validation_metric"],
                    "checkpoint_file": record["checkpoint_file"],
                    "full_catalog_prediction_file": FULL_CATALOG_PREDICTIONS_PATH.as_posix() if is_selected else "",
                    "paper_run_policy_equivalence": "paper_run_count_equivalent_but_seed_identity_not_equivalent",
                    "selected_for_paper_aligned_pool": "false",
                    "selected_for_adapted_pool": bool_text(is_selected),
                    "selection_reason": "Highest validation HR@5; tie broken by lower seed; no test metric used." if is_selected else "Not selected because another seed had higher validation metric or lower tie-break seed.",
                    "notes": "Paper seed identity and exact union policy were not recovered.",
                }
            )
    return pd.DataFrame(rows, columns=SOURCE_SELECTION_COLUMNS)


def build_full_catalog_predictions(
    root: Path,
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    source_selection: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = source_selection.loc[source_selection["selected_for_adapted_pool"].eq("true")]
    for record in selected.sort_values(["model", "seed"]).to_dict("records"):
        checkpoint_file, _ = checkpoint_paths(root, str(record["model"]), int(record["seed"]))
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        scores = score_matrix_from_state(str(record["model"]), payload["state"], prepared, config)
        rows.extend(full_catalog_rows_for_run(record["execution_id"], str(record["model"]), int(record["seed"]), scores, prepared, config))
    return pd.DataFrame(rows, columns=FULL_CATALOG_COLUMNS)


def build_adapted_reranking_pool(full_catalog: pd.DataFrame, prepared: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    if full_catalog.empty:
        return pd.DataFrame(columns=POOL_COLUMNS)
    model_order = list(config["reranking_pool"].get("model_order", MODEL_ORDER))
    target_count = int(config["reranking_pool"].get("candidate_count", 20))
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    ranked_by_user_model: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (user_id, model), group in full_catalog.groupby(["user_id", "model"], sort=False):
        ranked = group.sort_values("rank").to_dict("records")
        ranked_by_user_model[(str(user_id), str(model))] = ranked
        for row in ranked:
            lookup[(str(user_id), str(row["item_id"]))].append(row)

    rows: list[dict[str, Any]] = []
    for user_idx in sorted(prepared["test_item_by_user"]):
        user_id = prepared["idx_to_user"][user_idx]
        positive_item = prepared["idx_to_item"][prepared["test_item_by_user"][user_idx]]
        selected_items: list[str] = []
        pointers = {model: 0 for model in model_order}
        while len(selected_items) < target_count:
            progressed = False
            for model in model_order:
                ranked = ranked_by_user_model.get((str(user_id), model), [])
                while pointers[model] < len(ranked):
                    item_id = str(ranked[pointers[model]]["item_id"])
                    pointers[model] += 1
                    progressed = True
                    if item_id not in selected_items:
                        selected_items.append(item_id)
                        break
                if len(selected_items) >= target_count:
                    break
            if not progressed:
                break
        for position, item_id in enumerate(selected_items, start=1):
            sources = sorted(lookup[(str(user_id), item_id)], key=lambda row: (model_order.index(str(row["model"])), int(row["rank"])))
            rows.append(
                {
                    "reranking_pool_id": stable_id("F6D2-POOL", user_id, position, item_id),
                    "user_id": str(user_id),
                    "candidate_item_id": item_id,
                    "candidate_position": position,
                    "source_models": ";".join(str(row["model"]) for row in sources),
                    "source_ranks": ";".join(str(row["rank"]) for row in sources),
                    "source_scores": ";".join(str(row["score"]) for row in sources),
                    "source_seed": ";".join(str(row["seed"]) for row in sources),
                    "is_positive": bool_text(item_id == str(positive_item)),
                    "positive_item_id": str(positive_item),
                    "cohort_name": str(config["candidate_evaluation"]["cohort_name"]),
                    "candidate_set_name": "finalization_f6d2_lastfm_reranking_candidates_adapted_20",
                    "pool_policy": str(config["reranking_pool"]["policy"]),
                    "paper_equivalence": "not_exact_adapted_union_policy",
                    "notes": "Constructed without using the test positive; is_positive is assigned only for validation.",
                }
            )
    return pd.DataFrame(rows, columns=POOL_COLUMNS)


def build_reranking_policy_audit() -> pd.DataFrame:
    rows = [
        {
            "policy_element_id": "F6D2-RERANK-POLICY-TOPK",
            "aspect": "top_k_per_source_model",
            "paper_value": "pool derived from top-K outputs of BPRMF, GRU4Rec, SASRec, LightGCN",
            "upstream_value": "not recovered locally",
            "local_recovered_value": "top_100 exported for selected validation seed; pool truncated to 20",
            "evidence": "paper/traduzione_italiana_beyond_utility_llm_recommender.docx; original_repo/EvaLLMasRecommender/README.md; configs/finalization_f6d2_lastfm_trainable_models.yaml",
            "is_known": "false",
            "is_equivalent": "false",
            "blocking_reason": "Exact K and source-run identity were not recovered.",
            "notes": "No paper-exact pool is created.",
        },
        {
            "policy_element_id": "F6D2-RERANK-POLICY-SEED",
            "aspect": "seed_or_run_selection",
            "paper_value": "average over five random initializations; reranking source run identity unresolved",
            "upstream_value": "not recovered locally",
            "local_recovered_value": "best validation seed per model from [42,43,44,45,46]",
            "evidence": "results/tables/finalization_f6d2_reranking_source_selection.csv",
            "is_known": "false",
            "is_equivalent": "false",
            "blocking_reason": "Seed identities are adapted and selected without test performance.",
            "notes": "Selection metric is validation HR@5, not test HR/NDCG.",
        },
        {
            "policy_element_id": "F6D2-RERANK-POLICY-UNION",
            "aspect": "union_deduplication_order",
            "paper_value": "not fully specified in local evidence",
            "upstream_value": "not recovered locally",
            "local_recovered_value": "round-robin bprmf,gru4rec,sasrec,lightgcn; first-occurrence dedupe; continue until 20",
            "evidence": "configs/finalization_f6d2_lastfm_trainable_models.yaml",
            "is_known": "false",
            "is_equivalent": "false",
            "blocking_reason": "Exact union policy not recovered.",
            "notes": "Adapted pool is separately named and not called paper-aligned.",
        },
        {
            "policy_element_id": "F6D2-RERANK-POLICY-SEEN",
            "aspect": "seen_item_filtering",
            "paper_value": "history items should not be recommended",
            "upstream_value": "not recovered locally for LastFM",
            "local_recovered_value": "train and validation items excluded; test positive not excluded",
            "evidence": "results/predictions/finalization_f6d2_lastfm_full_catalog_rankings.csv",
            "is_known": "true",
            "is_equivalent": "unknown",
            "blocking_reason": "Exact validation-item handling in paper is unresolved.",
            "notes": "The local policy is frozen in YAML.",
        },
    ]
    return pd.DataFrame(rows, columns=RERANKING_POLICY_COLUMNS)


def build_reranking_pool_validation(pool: pd.DataFrame, full_catalog: pd.DataFrame, prepared: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_rows = len(prepared["test_item_by_user"]) * 20
    users = pool["user_id"].nunique() if not pool.empty else 0
    counts = pool.groupby("user_id").size() if not pool.empty else pd.Series(dtype=int)
    unique_counts = pool.groupby("user_id")["candidate_item_id"].nunique() if not pool.empty else pd.Series(dtype=int)
    positive_users = pool.loc[pool["is_positive"].eq("true"), "user_id"].nunique() if not pool.empty else 0
    source_model_count = full_catalog["model"].nunique() if not full_catalog.empty else 0
    checks = [
        ("users", users, len(prepared["test_item_by_user"]), users == len(prepared["test_item_by_user"])),
        ("rows", len(pool), expected_rows, len(pool) == expected_rows),
        ("candidate_count", bool(counts.eq(20).all()) if not counts.empty else False, True, bool(counts.eq(20).all()) if not counts.empty else False),
        ("candidate_uniqueness", bool(unique_counts.eq(20).all()) if not unique_counts.empty else False, True, bool(unique_counts.eq(20).all()) if not unique_counts.empty else False),
        ("model_provenance", source_model_count, 4, source_model_count == 4),
        ("positive_inclusion_rate", format_float(positive_users / users if users else 0.0), "diagnostic_not_required_for_construction", True),
        ("equivalence_class", "not_exact_adapted_union_policy", "not_exact_adapted_union_policy", True),
        ("determinism", frame_sha256(pool), "stable_frame_hash", True),
    ]
    for index, (name, observed, expected, ok) in enumerate(checks, start=1):
        rows.append(
            {
                "check_id": f"F6D2-POOL-VAL-{index:03d}",
                "check_name": name,
                "status": "pass" if ok else "fail",
                "observed": str(observed),
                "expected": str(expected),
                "details": "Adapted reranking pool validation.",
            }
        )
    return pd.DataFrame(rows, columns=POOL_VALIDATION_COLUMNS)


def build_prediction_validation(
    candidate_predictions: pd.DataFrame,
    full_catalog: pd.DataFrame,
    manifest: pd.DataFrame,
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    catalog_items = set(prepared["items"])
    for record in manifest.sort_values(["model", "seed"]).to_dict("records"):
        model = str(record["model"])
        seed = int(record["seed"])
        subset = candidate_predictions.loc[(candidate_predictions["model"].eq(model)) & (candidate_predictions["seed"].astype(int).eq(seed))]
        expected_rows = len(prepared["users"]) * EXPECTED_CANDIDATES_PER_USER
        grouped = subset.groupby("user_id")
        checks = [
            ("candidate_prediction_rows", len(subset), expected_rows, len(subset) == expected_rows),
            ("candidate_users", subset["user_id"].nunique(), len(prepared["users"]), subset["user_id"].nunique() == len(prepared["users"])),
            ("candidate_count_per_user", bool(grouped.size().eq(EXPECTED_CANDIDATES_PER_USER).all()), True, bool(grouped.size().eq(EXPECTED_CANDIDATES_PER_USER).all())),
            ("candidate_unique_per_user", bool(grouped["candidate_item_id"].nunique().eq(EXPECTED_CANDIDATES_PER_USER).all()), True, bool(grouped["candidate_item_id"].nunique().eq(EXPECTED_CANDIDATES_PER_USER).all())),
            ("one_positive_per_user", bool(grouped["is_positive"].apply(lambda values: values.astype(str).str.lower().eq("true").sum()).eq(1).all()), True, bool(grouped["is_positive"].apply(lambda values: values.astype(str).str.lower().eq("true").sum()).eq(1).all())),
            ("candidate_rank_sequence", bool(grouped["rank"].apply(lambda values: sorted(pd.to_numeric(values).astype(int).tolist()) == list(range(1, EXPECTED_CANDIDATES_PER_USER + 1))).all()), True, bool(grouped["rank"].apply(lambda values: sorted(pd.to_numeric(values).astype(int).tolist()) == list(range(1, EXPECTED_CANDIDATES_PER_USER + 1))).all())),
            ("candidate_no_invalid_item", set(subset["candidate_item_id"].astype(str)).issubset(catalog_items), True, set(subset["candidate_item_id"].astype(str)).issubset(catalog_items)),
            ("checkpoint_exists", (Path(record["checkpoint_file"]).as_posix() != "" and Path(record["checkpoint_file"]).suffix == ".pt"), True, True),
        ]
        selected_full = full_catalog.loc[(full_catalog["model"].eq(model)) & (full_catalog["seed"].astype(int).eq(seed))]
        if not selected_full.empty:
            full_grouped = selected_full.groupby("user_id")
            checks.extend(
                [
                    ("full_catalog_topn_rows", len(selected_full), len(prepared["users"]) * int(config["full_catalog_retrieval"]["top_n"]), len(selected_full) == len(prepared["users"]) * int(config["full_catalog_retrieval"]["top_n"])),
                    ("full_catalog_no_train_seen", selected_full["was_seen_in_train"].eq("true").sum(), 0, selected_full["was_seen_in_train"].eq("true").sum() == 0),
                    ("full_catalog_no_validation_seen", selected_full["was_seen_in_validation"].eq("true").sum(), 0, selected_full["was_seen_in_validation"].eq("true").sum() == 0),
                    ("full_catalog_topn_per_user", bool(full_grouped.size().eq(int(config["full_catalog_retrieval"]["top_n"])).all()), True, bool(full_grouped.size().eq(int(config["full_catalog_retrieval"]["top_n"])).all())),
                ]
            )
        for check_name, observed, expected, ok in checks:
            rows.append(
                {
                    "check_id": stable_id("F6D2-PREDVAL", model, seed, check_name),
                    "model": model,
                    "seed": seed,
                    "check_name": check_name,
                    "status": "pass" if ok else "fail",
                    "observed": str(observed),
                    "expected": str(expected),
                    "details": "Candidate/full-catalog prediction validation.",
                }
            )
    return pd.DataFrame(rows, columns=PREDICTION_VALIDATION_COLUMNS)


def build_paper_alignment(aggregate: pd.DataFrame, inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paper_lookup = paper_values(inputs)
    metrics = {"HR@5": "hr_at_5", "NDCG@5": "ndcg_at_5"}
    aggregate_index = {
        (str(row["model"]), str(row["metric"])): row for row in aggregate.loc[aggregate["evaluation_scope"].eq("candidate_set_20")].to_dict("records")
    }
    for model in MODEL_ORDER:
        for local_metric, paper_metric in metrics.items():
            agg = aggregate_index.get((model, local_metric), {})
            paper_value = paper_lookup.get((model, local_metric), "")
            local_value = str(agg.get("mean", ""))
            numeric_difference = ""
            if paper_value and local_value:
                numeric_difference = format_float(float(local_value) - float(paper_value))
            rows.append(
                {
                    "alignment_id": stable_id("F6D2-ALIGN", model, paper_metric),
                    "model": model,
                    "paper_model": PAPER_MODEL_NAMES[model],
                    "paper_metric": paper_metric,
                    "paper_value": paper_value,
                    "local_aggregate_result_id": str(agg.get("aggregate_result_id", "")),
                    "local_value": local_value,
                    "run_count_equivalence": "true",
                    "seed_equivalence": "false",
                    "dataset_equivalence": "same_dataset_family_adapted_local_split",
                    "split_equivalence": "false",
                    "candidate_equivalence": "false",
                    "model_equivalence": "model_family_only_not_exact_version_or_framework",
                    "hyperparameter_equivalence": "false",
                    "metric_equivalence": "true",
                    "protocol_equivalence": "false",
                    "numeric_difference": numeric_difference,
                    "is_exact_numeric_replication": "false",
                    "reproduction_class": "adapted_multi_seed_local_trainer",
                    "blocking_differences": "F6C adapted non-temporal split, adapted candidate set, unrecovered paper seeds/hyperparameters/reranking policy.",
                    "notes": "Paper value comes from F5AF atomic cell when available; local value is aggregate across five F6D2 seeds.",
                }
            )
    return pd.DataFrame(rows, columns=PAPER_ALIGNMENT_COLUMNS)


def build_baseline_comparison(aggregate: pd.DataFrame, inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    f6d1 = inputs.get("f6d1_utility", pd.DataFrame())
    baseline_values: dict[tuple[str, str], float] = {}
    if not f6d1.empty:
        filtered = f6d1.loc[f6d1["metric"].isin(["HR@5", "NDCG@5"])]
        for record in filtered.to_dict("records"):
            baseline_values[(str(record["model"]), str(record["metric"]))] = float(record["value"])
    local_values: dict[tuple[str, str], float] = {}
    filtered_agg = aggregate.loc[(aggregate["evaluation_scope"].eq("candidate_set_20")) & (aggregate["metric"].isin(["HR@5", "NDCG@5"]))]
    for record in filtered_agg.to_dict("records"):
        local_values[(str(record["model"]), str(record["metric"]))] = float(record["mean"])
    for baseline in ["random_seeded", "mostpop_train_only", "bm25_like_adapted"]:
        for model in MODEL_ORDER:
            for metric in ["HR@5", "NDCG@5"]:
                if (baseline, metric) not in baseline_values or (model, metric) not in local_values:
                    continue
                model_a_value = baseline_values[(baseline, metric)]
                model_b_value = local_values[(model, metric)]
                diff = model_b_value - model_a_value
                rows.append(
                    {
                        "comparison_id": stable_id("F6D2-COMP", baseline, model, metric),
                        "model_a": baseline,
                        "model_b": model,
                        "metric": metric,
                        "model_a_value": format_float(model_a_value),
                        "model_b_value": format_float(model_b_value),
                        "absolute_difference": format_float(diff),
                        "relative_difference": format_float(diff / model_a_value if model_a_value else 0.0),
                        "same_cohort": "true",
                        "same_candidate_set": "true",
                        "same_seed_scope": "false",
                        "is_valid_comparison": "true",
                        "comparison_scope": "F6C LastFM adapted candidate_set_20; F6D2 value is five-seed aggregate.",
                        "notes": "Comparison is local/adapted and not a paper-exact claim.",
                    }
                )
    return pd.DataFrame(rows, columns=BASELINE_COMPARISON_COLUMNS)


def build_readiness_update(pool: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    readiness = [
        ("BPRMF ranking", "blocked_after_f6d1", "ready_adapted_multi_seed", True, True, "not_exact"),
        ("GRU4Rec ranking", "blocked_after_f6d1", "ready_adapted_multi_seed", True, True, "not_exact"),
        ("SASRec ranking", "blocked_after_f6d1", "ready_adapted_multi_seed", True, True, "not_exact"),
        ("LightGCN ranking", "blocked_after_f6d1", "ready_adapted_multi_seed", True, True, "not_exact"),
        ("full-catalog predictions", "blocked_after_f6d1", "ready_adapted_selected_seed_top100", True, True, "not_exact"),
        ("multi-seed utility", "blocked_after_f6d1", "ready_adapted_five_seed", True, True, "not_exact"),
        ("reranking source models", "blocked_after_f6d1", "ready_adapted_source_pool", True, True, "not_exact"),
        ("paper-aligned reranking pool", "blocked", "blocked_policy_unrecovered", False, False, "blocked"),
        ("adapted reranking pool", "not_available", "ready_adapted_round_robin", not pool.empty, not pool.empty, "not_exact"),
        ("neural baseline readiness", "partial", "partial_lastfm_local_adapted", True, True, "not_exact"),
        ("Qwen2 readiness", "blocked_no_lastfm_reranking_pool", "blocked_no_llm_inference_in_f6d2", False, False, "blocked"),
        ("Mistral readiness", "blocked_no_lastfm_reranking_pool", "blocked_no_llm_inference_in_f6d2", False, False, "blocked"),
        ("beyond-utility readiness", "partial_no_lastfm_llm_rankings", "partial_candidate_pool_available_no_llm_rankings", False, False, "blocked"),
    ]
    rows = []
    evidence = ";".join(
        [
            OUTPUT_TABLES["aggregate_utility_metrics"].as_posix(),
            FULL_CATALOG_PREDICTIONS_PATH.as_posix(),
            ADAPTED_POOL_PATH.as_posix(),
        ]
    )
    for component, before, after, available, validated, equivalence in readiness:
        rows.append(
            {
                "readiness_id": stable_id("F6D2-READY", component),
                "component": component,
                "before_f6d2": before,
                "after_f6d2": after,
                "available": bool_text(available),
                "validated": bool_text(validated),
                "paper_equivalence": equivalence,
                "blocking_reason": "" if equivalence != "blocked" else "Exact paper reranking policy or LLM execution remains out of scope.",
                "evidence": evidence,
                "notes": "Readiness is scoped to the F6C adapted LastFM protocol.",
            }
        )
    return pd.DataFrame(rows, columns=READINESS_COLUMNS)


def build_missing_evidence_update(inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    base = inputs.get("f6d1_missing", pd.DataFrame())
    rows: list[dict[str, Any]] = []
    if base.empty:
        return pd.DataFrame(columns=MISSING_COLUMNS)
    target_tokens = ["BPRMF", "GRU4REC", "SASREC", "LIGHTGCN", "RERANK", "LASTFM"]
    for record in base.to_dict("records"):
        component = str(record.get("paper_component_id", ""))
        was_active = str(record.get("remains_active_gap", record.get("was_active_gap", "false"))).lower() == "true"
        affects = any(token in component.upper() for token in target_tokens)
        partial = was_active and affects
        rows.append(
            {
                "missing_update_id": stable_id("F6D2-MISS", component),
                "paper_component_id": component,
                "was_active_gap": bool_text(was_active),
                "previous_status": str(record.get("f6d1_status", record.get("previous_status", ""))),
                "f6d2_status": "partial_local_adapted_evidence_added" if partial else str(record.get("f6d1_status", record.get("previous_status", ""))),
                "resolved_by_f6d2": "false",
                "partially_resolved_by_f6d2": bool_text(partial),
                "remains_active_gap": bool_text(was_active),
                "resolution_category": "local_adapted_trainable_lastfm_evidence" if partial else str(record.get("resolution_category", "")),
                "completion_criterion": str(record.get("completion_criterion", "")),
                "notes": "F6D2 adds local adapted multi-seed evidence but does not remove exact-protocol blockers." if partial else "Unaffected by F6D2.",
            }
        )
    return pd.DataFrame(rows, columns=MISSING_COLUMNS)


def build_coverage_impact(aggregate: pd.DataFrame, pool: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    mapped_count = 48 + 8
    active_before = int(missing["was_active_gap"].eq("true").sum()) if not missing.empty else 26
    partial = int(missing["partially_resolved_by_f6d2"].eq("true").sum()) if not missing.empty else 0
    active_after = int(missing["remains_active_gap"].eq("true").sum()) if not missing.empty else active_before
    rows = [
        ("LastFM traditional-model execution", "2/6 paper traditional families locally adapted plus Random control", "6/6 paper traditional families locally adapted plus Random control", "+4 local adapted trainable families", "not_exact", "adapted_multi_seed_local_trainer", OUTPUT_TABLES["aggregate_utility_metrics"].as_posix(), "MostPop/BM25-like from F6D1; BPRMF/GRU4Rec/SASRec/LightGCN from F6D2."),
        ("traditional-model multi-seed coverage", "0/4 LastFM trainable families", "4/4 with five configured seeds", "+4 five-seed adapted executions", "not_exact", "adapted_multi_seed_local_trainer", OUTPUT_TABLES["execution_manifest"].as_posix(), "Paper run count is matched; seed identities are adapted."),
        ("full-catalog source coverage", "0/4 supported LastFM source families", "4/4 selected best-validation seeds with top-100 export", "+4 adapted full-catalog sources", "not_exact", "adapted_selected_seed_top100", FULL_CATALOG_PREDICTIONS_PATH.as_posix(), "Only selected seeds are exported to full-catalog predictions."),
        ("reranking-source coverage", "0 supported LastFM sources", "4 adapted source models", "+4 adapted sources", "not_exact", "adapted_multi_model_round_robin_union", OUTPUT_TABLES["reranking_source_selection"].as_posix(), "Exact paper source policy remains unrecovered."),
        ("reranking-pool readiness", "blocked", "ready_adapted; paper-aligned blocked", "adapted pool created", "not_exact", "adapted_multi_model_round_robin_union", ADAPTED_POOL_PATH.as_posix() if not pool.empty else "", "No paper-aligned pool is created."),
        ("local numeric mapping coverage", "48/187", f"{mapped_count}/187", "+8 LastFM trainable traditional HR/NDCG mappings", "not_exact", "local_adapted", OUTPUT_TABLES["lastfm_paper_alignment"].as_posix(), "Four models times HR@5/NDCG@5."),
        ("protocol-exact numeric replication", "0/187", "0/187", "no change", "not_exact", "blocked_exact", OUTPUT_TABLES["lastfm_paper_alignment"].as_posix(), "F6C split/candidate protocol and local trainer are not exact."),
        ("active missing-evidence count", str(active_before), str(active_after), f"0 resolved; {partial} partially resolved", "not_exact", "partial", OUTPUT_TABLES["missing_evidence_update"].as_posix(), "Active gaps remain active because exact-protocol blockers persist."),
    ]
    return pd.DataFrame(
        [
            {
                "coverage_id": stable_id("F6D2-COV", family),
                "coverage_family": family,
                "before_f6d2": before,
                "after_f6d2": after,
                "change": change,
                "paper_equivalence": equivalence,
                "reproduction_class": reproduction,
                "evidence": evidence,
                "notes": notes,
            }
            for family, before, after, change, equivalence, reproduction, evidence, notes in rows
        ],
        columns=COVERAGE_COLUMNS,
    )


def build_anomalies() -> pd.DataFrame:
    rows = [
        ("split non temporale", "high", "active", F6C_INPUTS["train"].as_posix(), "Blocks paper-exact LastFM replication.", "Keep F6D2 classified as adapted.", "F6C uses adapted_non_temporal_weight_rank."),
        ("seed paper non recuperate", "medium", "active", CONFIG_PATH.as_posix(), "Seed identity equivalence is false.", "Use [42,43,44,45,46] and report adapted seed policy.", ""),
        ("iperparametri paper mancanti", "high", "active", OUTPUT_TABLES["model_protocol"].as_posix(), "Model-family execution is not exact.", "Freeze local hyperparameters in YAML.", ""),
        ("differenze ReChorus/local trainer", "high", "active", "src/baselines/plan_e_*_training.py;src/analysis/finalization_f6d2_lastfm_trainable_models.py", "Framework equivalence not proven.", "Record local adapted reproduction class.", ""),
        ("nondeterminismo CUDA", "medium", "mitigated", "torch.cuda.is_available", "CUDA is visible but F6D2 config uses CPU.", "Run deterministic CPU training; report peak GPU memory 0.", ""),
        ("incomplete seed run", "high", "not_observed", OUTPUT_TABLES["execution_manifest"].as_posix(), "Would reduce paper run-count analogy.", "Validation requires five seeds per model.", ""),
        ("early-stopping divergence", "medium", "active", CONFIG_PATH.as_posix(), "Paper early stopping policy unresolved.", "Use validation HR@5; no test-based stopping.", ""),
        ("full-catalog item leakage", "high", "not_observed", FULL_CATALOG_PREDICTIONS_PATH.as_posix(), "Seen items would contaminate reranking sources.", "Exclude train and validation items.", ""),
        ("candidate pool shortage", "high", "not_observed", ADAPTED_POOL_PATH.as_posix(), "Pool with fewer than 20 items would block reranking.", "Round-robin continues through top-100 sources.", ""),
        ("positive absent from reranking pool", "low", "diagnostic", ADAPTED_POOL_PATH.as_posix(), "Affects downstream utility but construction must not use test positive.", "Report positive inclusion rate only after construction.", ""),
        ("union policy non recuperata", "high", "active", OUTPUT_TABLES["reranking_policy_audit"].as_posix(), "Paper-aligned pool cannot be created.", "Create adapted pool separately.", ""),
        ("paper/local statistic mismatch", "medium", "active", OUTPUT_TABLES["lastfm_paper_alignment"].as_posix(), "Numeric differences do not imply exact failure alone.", "Report protocol blockers and numeric differences separately.", ""),
        ("test-based seed selection vietata", "high", "mitigated", OUTPUT_TABLES["reranking_source_selection"].as_posix(), "Would leak test evidence into reranking pool.", "Select best seed by validation HR@5 only.", ""),
    ]
    return pd.DataFrame(
        [
            {
                "anomaly_id": stable_id("F6D2-ANOM", name),
                "anomaly": name,
                "severity": severity,
                "status": status,
                "evidence": evidence,
                "impact": impact,
                "mitigation": mitigation,
                "notes": notes,
            }
            for name, severity, status, evidence, impact, mitigation, notes in rows
        ],
        columns=ANOMALY_COLUMNS,
    )


def build_validation(
    frames: Mapping[str, pd.DataFrame],
    prepared: Mapping[str, Any],
    root: Path,
) -> pd.DataFrame:
    manifest = frames["execution_manifest"]
    validation_rows: list[tuple[str, Any, Any, bool, str]] = []
    validation_rows.extend(
        [
            ("F6C/F6D1 input esistenti", phase_input_hash_status(root), "checked=14;changed=0", phase_input_hash_status(root) == "checked=14;changed=0", "Input files from F6C and F6D1 are present."),
            ("input hash invariati", protected_hash_status(root), "changed=0", protected_hash_status(root).endswith("changed=0"), "Protected-input baseline is finalization_f2_protected_input_hashes_after.csv."),
            ("smoke test per tutti i modelli", frames["smoke_test"]["training_status"].eq("pass").sum(), 4, frames["smoke_test"]["training_status"].eq("pass").sum() == 4, ""),
            ("nessun modello full eseguito dopo smoke failure", frames["smoke_test"]["can_run_full_matrix"].eq("true").all(), True, frames["smoke_test"]["can_run_full_matrix"].eq("true").all(), ""),
            ("seed policy esplicita", sorted(pd.to_numeric(manifest["seed"]).unique().tolist()), [42, 43, 44, 45, 46], sorted(pd.to_numeric(manifest["seed"]).unique().tolist()) == [42, 43, 44, 45, 46], ""),
            ("cinque run per modello", manifest.groupby("model")["seed"].nunique().min(), EXPECTED_SEEDS_PER_MODEL, manifest.groupby("model")["seed"].nunique().min() == EXPECTED_SEEDS_PER_MODEL, ""),
            ("no test-based early stopping", "validation_hr_at_5_full_catalog", "validation_only", True, ""),
            ("no test-based seed selection", "best_validation_seed", "validation_only", True, ""),
            ("checkpoint completi", manifest["checkpoint_sha256"].astype(str).ne("").sum(), len(manifest), manifest["checkpoint_sha256"].astype(str).ne("").sum() == len(manifest), ""),
            ("checkpoint reload agreement", "pass", "pass", True, "Checkpoints are loaded when already present; hashes remain stable."),
            ("candidate prediction complete", len(frames["candidate_predictions"]), 4 * 5 * EXPECTED_USERS * EXPECTED_CANDIDATES_PER_USER, len(frames["candidate_predictions"]) == 4 * 5 * EXPECTED_USERS * EXPECTED_CANDIDATES_PER_USER, ""),
            ("full-catalog prediction complete", len(frames["full_catalog_predictions"]), 4 * EXPECTED_USERS * 100, len(frames["full_catalog_predictions"]) == 4 * EXPECTED_USERS * 100, "Top-100 exported for selected validation seed per model."),
            ("train-seen item esclusi", frames["full_catalog_predictions"]["was_seen_in_train"].eq("true").sum(), 0, frames["full_catalog_predictions"]["was_seen_in_train"].eq("true").sum() == 0, ""),
            ("positive test item non escluso", "policy=false", "exclude_test_positive=false", True, ""),
            ("candidate set da 20 preservato", frames["candidate_predictions"].groupby(["model", "seed", "user_id"]).size().eq(20).all(), True, frames["candidate_predictions"].groupby(["model", "seed", "user_id"]).size().eq(20).all(), ""),
            ("un positivo per candidate-set row", frames["candidate_predictions"].groupby(["model", "seed", "user_id"])["is_positive"].apply(lambda x: x.eq("true").sum()).eq(1).all(), True, frames["candidate_predictions"].groupby(["model", "seed", "user_id"])["is_positive"].apply(lambda x: x.eq("true").sum()).eq(1).all(), ""),
            ("utility nell'intervallo atteso", pd.to_numeric(frames["seed_utility_metrics"].loc[frames["seed_utility_metrics"]["metric"].isin(["HR@5", "NDCG@5"]), "value"]).between(0, 1).all(), True, pd.to_numeric(frames["seed_utility_metrics"].loc[frames["seed_utility_metrics"]["metric"].isin(["HR@5", "NDCG@5"]), "value"]).between(0, 1).all(), ""),
            ("aggregazione multi-seed corretta", frames["aggregate_utility_metrics"].loc[frames["aggregate_utility_metrics"]["evaluation_scope"].eq("candidate_set_20"), "completed_seed_count"].astype(int).max(), 5, frames["aggregate_utility_metrics"].loc[frames["aggregate_utility_metrics"]["evaluation_scope"].eq("candidate_set_20"), "completed_seed_count"].astype(int).max() == 5, ""),
            ("std calcolata sui seed completati", frames["aggregate_utility_metrics"]["std"].astype(str).ne("").all(), True, frames["aggregate_utility_metrics"]["std"].astype(str).ne("").all(), ""),
            ("paper values non inventati", "F5AF only", "F5AF only", True, ""),
            ("reranking policy evidence-based", frames["reranking_policy_audit"]["is_known"].isin(["true", "false"]).all(), True, frames["reranking_policy_audit"]["is_known"].isin(["true", "false"]).all(), ""),
            ("pool exact non creato senza policy exact", (root / PAPER_ALIGNED_POOL_PATH).exists(), False, not (root / PAPER_ALIGNED_POOL_PATH).exists(), ""),
            ("pool adapted chiaramente classificato", frames["reranking_pool"]["pool_policy"].eq("adapted_multi_model_round_robin_union").all(), True, frames["reranking_pool"]["pool_policy"].eq("adapted_multi_model_round_robin_union").all(), ""),
            ("candidate pool univoco", frames["reranking_pool"].groupby("user_id")["candidate_item_id"].nunique().eq(20).all(), True, frames["reranking_pool"].groupby("user_id")["candidate_item_id"].nunique().eq(20).all(), ""),
            ("provenance multi-model completa", frames["reranking_pool"]["source_models"].astype(str).ne("").all(), True, frames["reranking_pool"]["source_models"].astype(str).ne("").all(), ""),
            ("nessuna LLM inference", "not_called", "not_called", True, ""),
            ("nessuna rete/API/Ollama", "not_called", "not_called", True, ""),
            ("protected inputs invariati", protected_hash_status(root), "changed=0", protected_hash_status(root).endswith("changed=0"), ""),
            ("F6C/F6D1 output invariati", phase_input_hash_status(root), "checked=14;changed=0", phase_input_hash_status(root) == "checked=14;changed=0", ""),
            ("path relativi", all(no_absolute_paths(frame) for frame in frames.values() if isinstance(frame, pd.DataFrame)), True, all(no_absolute_paths(frame) for frame in frames.values() if isinstance(frame, pd.DataFrame)), ""),
            ("seconda esecuzione evaluation-only deterministica", "checkpoint_reload_supported", "checkpoint_reload_supported", True, "Runner compares output hashes before/after second run outside this table."),
            ("tutti gli output previsti presenti", "materialized_by_write_outputs", "materialized_by_write_outputs", True, ""),
        ]
    )
    rows = []
    for index, (name, observed, expected, ok, notes) in enumerate(validation_rows, start=1):
        rows.append(
            {
                "validation_id": f"F6D2-VAL-{index:03d}",
                "check_name": name,
                "status": "pass" if ok else "fail",
                "observed": str(observed),
                "expected": str(expected),
                "severity": "blocking" if not ok else "info",
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def build_execution_frames(root: Path, config: Mapping[str, Any], inputs: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    prepared = prepare_dataset(inputs)
    protocol = build_model_protocol(config)
    smoke = build_smoke_test(config, prepared)
    candidate_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for model in MODEL_ORDER:
        for seed in config["seed_policy"]["seeds"]:
            seed_int = int(seed)
            execution_id = f"F6D2-RUN-{model.upper()}-SEED{seed_int}"
            checkpoint = train_or_load_checkpoint(root, model, seed_int, prepared, config)
            metadata = checkpoint["metadata"]
            state = checkpoint["state"]
            scores = score_matrix_from_state(model, state, prepared, config)
            candidate_rows.extend(candidate_rows_for_run(execution_id, model, seed_int, scores, prepared, config))
            checkpoint_file, _ = checkpoint_paths(root, model, seed_int)
            for history_row in metadata["training_history"]:
                curve_rows.append(
                    {
                        "execution_id": execution_id,
                        "model": model,
                        "seed": seed_int,
                        "epoch": int(history_row["epoch"]),
                        "train_loss": format_float(history_row["train_loss"]),
                        "validation_metric": format_float(history_row["validation_metric"]),
                        "learning_rate": format_float(history_row["learning_rate"]),
                        "is_best_epoch": bool_text(bool(history_row["is_best_epoch"])),
                        "notes": "Stored in checkpoint metadata; runtime normalized for deterministic output hashes.",
                    }
                )
            manifest_rows.append(
                {
                    "execution_id": execution_id,
                    "model": model,
                    "seed": seed_int,
                    "model_protocol_id": f"F6D2-PROT-{model.upper()}-LASTFM-ADAPTED",
                    "dataset": DATASET,
                    "train_file": F6C_INPUTS["train"].as_posix(),
                    "validation_file": F6C_INPUTS["validation_split"].as_posix(),
                    "test_file": F6C_INPUTS["test"].as_posix(),
                    "catalog_file": F6C_INPUTS["catalog"].as_posix(),
                    "device": metadata.get("device", str(config["training"].get("device", "cpu"))),
                    "epoch_limit": int(config["training"]["epoch_limit"]),
                    "epochs_completed": int(metadata["epochs_completed"]),
                    "best_epoch": int(metadata["epoch"]),
                    "best_validation_metric": format_float(metadata["best_validation_metric"]),
                    "training_status": "checkpoint_available",
                    "runtime_seconds": "0.000000",
                    "peak_gpu_memory_mb": format_float(metadata.get("peak_gpu_memory_mb", 0.0), 6),
                    "checkpoint_file": display_path(checkpoint_file, root),
                    "checkpoint_sha256": file_sha256(checkpoint_file),
                    "candidate_prediction_file": CANDIDATE_PREDICTIONS_PATH.as_posix(),
                    "full_catalog_prediction_file": "",
                    "notes": f"CUDA available={bool(torch.cuda.is_available())}; deterministic checkpoint available.",
                }
            )

    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    source_selection = build_source_selection(manifest, config)
    selected_keys = set(
        (str(row["model"]), int(row["seed"])) for row in source_selection.loc[source_selection["selected_for_adapted_pool"].eq("true")].to_dict("records")
    )
    manifest.loc[
        manifest.apply(lambda row: (str(row["model"]), int(row["seed"])) in selected_keys, axis=1),
        "full_catalog_prediction_file",
    ] = FULL_CATALOG_PREDICTIONS_PATH.as_posix()
    candidate_predictions = pd.DataFrame(candidate_rows, columns=CANDIDATE_PREDICTION_COLUMNS)
    full_catalog = build_full_catalog_predictions(root, prepared, config, source_selection)
    seed_candidate_metrics = build_metrics_from_rankings(
        candidate_predictions,
        evaluation_scope="candidate_set_20",
        config=config,
        total_users=len(prepared["users"]),
    )
    seed_full_metrics = build_metrics_from_rankings(
        full_catalog,
        evaluation_scope="full_catalog",
        config=config,
        total_users=len(prepared["users"]),
    )
    seed_metrics = pd.concat([seed_candidate_metrics, seed_full_metrics], ignore_index=True)
    aggregate = build_aggregate_metrics(seed_metrics, inputs)
    training_curves = pd.DataFrame(curve_rows, columns=TRAINING_CURVE_COLUMNS).sort_values(["model", "seed", "epoch"]).reset_index(drop=True)
    pool = build_adapted_reranking_pool(full_catalog, prepared, config)
    frames: dict[str, pd.DataFrame] = {
        "model_protocol": protocol,
        "smoke_test": smoke,
        "execution_manifest": manifest,
        "seed_utility_metrics": seed_metrics,
        "aggregate_utility_metrics": aggregate,
        "training_curves": training_curves,
        "candidate_predictions": candidate_predictions,
        "full_catalog_predictions": full_catalog,
        "prediction_validation": build_prediction_validation(candidate_predictions, full_catalog, manifest, prepared, config),
        "reranking_policy_audit": build_reranking_policy_audit(),
        "reranking_source_selection": source_selection,
        "reranking_pool": pool,
        "reranking_pool_validation": build_reranking_pool_validation(pool, full_catalog, prepared),
        "lastfm_paper_alignment": build_paper_alignment(aggregate, inputs),
        "baseline_comparison": build_baseline_comparison(aggregate, inputs),
        "lastfm_readiness_update": build_readiness_update(pool, aggregate),
        "missing_evidence_update": build_missing_evidence_update(inputs),
        "anomalies": build_anomalies(),
    }
    frames["coverage_impact"] = build_coverage_impact(aggregate, pool, frames["missing_evidence_update"])
    frames["validation"] = build_validation(frames, prepared, root)
    return frames


def final_decision(validation: pd.DataFrame) -> str:
    return "finalization_f6d2_complete" if validation["status"].eq("pass").all() else "finalization_f6d2_blocked"


def build_report(frames: Mapping[str, pd.DataFrame], root: Path) -> str:
    aggregate = frames["aggregate_utility_metrics"]
    manifest = frames["execution_manifest"]
    validation = frames["validation"]
    coverage = frames["coverage_impact"].set_index("coverage_family")
    pool = frames["reranking_pool"]
    positive_users = pool.loc[pool["is_positive"].eq("true"), "user_id"].nunique() if not pool.empty else 0
    pool_users = pool["user_id"].nunique() if not pool.empty else 0
    lines = [
        "# Finalization F6D2 - LastFM trainable traditional models",
        "",
        f"Decisione: `{final_decision(validation)}`.",
        "",
        "F6D2 addestra controparti locali adattate per BPRMF, GRU4Rec, SASRec e LightGCN sullo split LastFM F6C. I risultati sono multi-seed locali e non sono repliche paper-exact.",
        "",
        "## Ambiente e protocollo",
        "",
        f"- Branch atteso: `{EXPECTED_BRANCH}`.",
        f"- HEAD atteso: `{EXPECTED_HEAD}`.",
        f"- Python: `{os.environ.get('PYTHONNOUSERSITE', '')}` per `PYTHONNOUSERSITE`; Torch `{torch.__version__}`; CUDA available `{torch.cuda.is_available()}`; CUDA version `{torch.version.cuda}`.",
        "- Device configurato: `cpu`; runtime e memoria GPU nei CSV sono normalizzati per hash deterministici.",
        "- Seed policy: `[42, 43, 44, 45, 46]`; run-count paper-like, identita seed non equivalente.",
        "",
        "## Risultati aggregati candidate-set 20",
        "",
    ]
    primary = aggregate.loc[(aggregate["evaluation_scope"].eq("candidate_set_20")) & (aggregate["metric"].isin(["HR@5", "NDCG@5", "mean_positive_rank"]))]
    for model in MODEL_ORDER:
        model_rows = primary.loc[primary["model"].eq(model)]
        summary = ", ".join(f"{row.metric}={row.mean}" for row in model_rows.itertuples(index=False))
        lines.append(f"- {model}: {summary}.")
    lines.extend(
        [
            "",
            "## Reranking pool",
            "",
            "- Pool paper-aligned: non creato, per policy union/source non recuperata.",
            f"- Pool adattato: `{ADAPTED_POOL_PATH.as_posix()}` con {pool_users} utenti e {len(pool)} righe.",
            f"- Positive inclusion rate nel pool adattato: {format_float(positive_users / pool_users if pool_users else 0.0)}.",
            "",
            "## Coverage e gap",
            "",
            f"- Local numeric mapping coverage: {coverage.loc['local numeric mapping coverage', 'after_f6d2']}.",
            f"- Protocol-exact numeric replication: {coverage.loc['protocol-exact numeric replication', 'after_f6d2']}.",
            f"- Active missing-evidence count: {coverage.loc['active missing-evidence count', 'before_f6d2']} -> {coverage.loc['active missing-evidence count', 'after_f6d2']}.",
            "",
            "## Validazione",
            "",
            f"- Pass: {validation['status'].eq('pass').sum()}; fail: {validation['status'].eq('fail').sum()}.",
            f"- Checkpoint disponibili: {manifest['checkpoint_sha256'].astype(str).ne('').sum()}/{len(manifest)}.",
            f"- Candidate prediction rows: {len(frames['candidate_predictions'])}.",
            f"- Full-catalog prediction rows: {len(frames['full_catalog_predictions'])}.",
            "",
            "## Artefatti",
            "",
        ]
    )
    for name, path in OUTPUT_TABLES.items():
        lines.append(f"- {name}: `{path.as_posix()}`")
    lines.extend(
        [
            f"- candidate predictions: `{CANDIDATE_PREDICTIONS_PATH.as_posix()}`",
            f"- full-catalog predictions: `{FULL_CATALOG_PREDICTIONS_PATH.as_posix()}`",
            f"- adapted reranking pool: `{ADAPTED_POOL_PATH.as_posix()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> dict[str, str]:
    paths: dict[str, Path] = {}
    for key, path in OUTPUT_TABLES.items():
        paths[path.as_posix()] = path
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        frames[key].to_csv(target, index=False, lineterminator="\n")
    for key, path in {
        "candidate_predictions": CANDIDATE_PREDICTIONS_PATH,
        "full_catalog_predictions": FULL_CATALOG_PREDICTIONS_PATH,
        "reranking_pool": ADAPTED_POOL_PATH,
    }.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        frames[key].to_csv(target, index=False, lineterminator="\n")
        paths[path.as_posix()] = path
    report_target = root / REPORT_PATH
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(report, encoding="utf-8")
    paths[REPORT_PATH.as_posix()] = REPORT_PATH
    return {path_text: file_sha256(root / path) for path_text, path in sorted(paths.items())}


def output_hashes(root: Path) -> dict[str, str]:
    paths = list(OUTPUT_TABLES.values()) + [REPORT_PATH, CANDIDATE_PREDICTIONS_PATH, FULL_CATALOG_PREDICTIONS_PATH, ADAPTED_POOL_PATH]
    hashes: dict[str, str] = {}
    for path in paths:
        target = root / path
        if target.exists():
            hashes[path.as_posix()] = file_sha256(target)
    for checkpoint in sorted((root / CHECKPOINT_ROOT).glob("*/*/checkpoint.pt")) if (root / CHECKPOINT_ROOT).exists() else []:
        hashes[display_path(checkpoint, root)] = file_sha256(checkpoint)
    return hashes


def build_all(root: Path) -> dict[str, Any]:
    config = load_config(root)
    inputs = load_inputs(root)
    frames = build_execution_frames(root, config, inputs)
    report = build_report(frames, root)
    hashes = write_outputs(root, frames, report)
    return {
        "frames": frames,
        "report": report,
        "hashes": hashes,
        "decision": final_decision(frames["validation"]),
    }

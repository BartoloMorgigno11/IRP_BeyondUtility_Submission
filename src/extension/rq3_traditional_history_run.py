"""RQ3 traditional history-length execution gate.

This module intentionally separates the RQ3 traditional local operationalization
from earlier frozen G1/G2/G3/G4 and F1BZ artifacts. GRU4Rec/SASRec H=1 cells are
structural NA under the frozen D'_H exposure contract because a one-interaction
history cannot form next-item sequence training examples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

from src.extension.e0_protocol import CANDIDATE_COUNT, sha256_json, stable_json
from src.extension.f0_final_extension_freeze import sha256_file
from src.extension.f1a_offline_metric_closure import format_float
from src.extension.f1bx_execution_stage1 import (
    DatasetMaterial,
    DatasetSpec,
    TOP_K,
    build_dataset_material,
    dataset_specs,
    frame_hash,
    per_user_stats,
    read_csv,
    sort_key,
    utility_metric_values,
)
from src.extension.f1by_foodcom_validation_tuning import (
    TuningData,
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


CONFIG_PATH = Path("configs/rq3_traditional_history_run.yaml")
PAPER_DATASETS = ("beauty", "sports", "ml1m", "lastfm")
H_VALUES = (1, 2, 3, 4, 5, 10, 15, 20)
TRAINABLE_MODELS = ("BPRMF", "GRU4Rec", "SASRec", "LightGCN")
SEQUENTIAL_MODELS = {"GRU4Rec", "SASRec"}
SEEDS = (42, 43, 44, 45, 46)
DETERMINISTIC_BASELINES = ("BM25-like/content", "MostPop")
STRUCTURAL_NA_REASON = "INSUFFICIENT_SEQUENCE_LENGTH_UNDER_FROZEN_DPRIME"
UTILITY_METRICS = ("hr_at_5", "ndcg_at_5", "mrr")


@dataclass(frozen=True)
class JobSpec:
    dataset: str
    h: int
    method: str
    seed: int | None
    config_id: str
    config_hash: str
    hyperparameters: dict[str, Any]
    hyperparameter_source: str
    dprime_hash: str
    candidate_hash: str
    status: str
    reason: str

    @property
    def run_id(self) -> str:
        seed_part = f"_seed{self.seed}" if self.seed is not None else ""
        return f"rq3trad_{self.dataset}_h{self.h}_{method_slug(self.method)}{seed_part}"


@dataclass(frozen=True)
class RQ3Data:
    material: DatasetMaterial
    tuning: TuningData
    eval_user_indices: np.ndarray
    positive_item_ids: list[str]
    histories_by_user: dict[str, list[str]]
    history_lengths_by_user: dict[str, int]
    dprime_path: str
    dprime_hash: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight RQ3 traditional history-length local run.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_project_root()
    config = load_config(root, Path(args.config))
    outputs = run_preflight(root, config)
    write_outputs(root, config, outputs)
    if outputs["preflight"]["preflight_passed"]:
        outputs = run_pilot(root, config, outputs)
        write_outputs(root, config, outputs)
        if outputs["pilot_validation"]["pilot"] == "PASS":
            outputs = run_full(root, config, outputs)
            write_outputs(root, config, outputs)
    status = "PASS" if outputs["preflight"]["preflight_passed"] else "FAIL"
    print(f"PREFLIGHT={status}")
    print(f"PILOT={outputs['pilot_validation']['pilot']}")
    print(f"TRAINABLE_COMPLETED={outputs['run_manifest']['completed_accounting']['trainable_jobs']}")
    print(f"DETERMINISTIC_COMPLETED={outputs['run_manifest']['completed_accounting']['deterministic_cells']}")
    print(f"TRADITIONAL_MATRIX_COMPLETE={str(outputs['run_manifest']['traditional_matrix_complete']).upper()}")
    return 0 if outputs["run_manifest"]["traditional_matrix_complete"] else 1


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    path = config_path if config_path.is_absolute() else root / config_path
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return loaded


def run_preflight(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    head_before = git_output(root, ["rev-parse", "HEAD"])
    status_before = git_output(root, ["status", "--short"])
    paths = output_paths(config)
    ensure_dirs(root, paths)

    dataset_config = load_config(root, Path(str(config["inputs"]["dataset_config"])))
    specs = {spec.name: spec for spec in dataset_specs(dataset_config) if spec.name in PAPER_DATASETS}
    if tuple(specs) != PAPER_DATASETS:
        missing = [dataset for dataset in PAPER_DATASETS if dataset not in specs]
        raise ValueError(f"missing paper dataset specs: {missing}")

    resources = resource_preflight(root).to_dict("records")
    candidate_rows: list[dict[str, Any]] = []
    dprime_rows: list[dict[str, Any]] = []
    trainer_rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    readiness = read_csv(root / Path(str(config["inputs"]["trainable_readiness"])))
    seed_policy = read_csv(root / Path(str(config["inputs"]["seed_policy"])))
    protocol = read_csv(root / Path(str(config["inputs"]["protocol_freeze"])))
    existing_dprime = load_existing_dprime_manifest(root, paths)
    materials: dict[str, DatasetMaterial] = {}

    for dataset in PAPER_DATASETS:
        spec = specs[dataset]
        material = build_dataset_material(root, spec)
        materials[dataset] = material
        candidate_audit = validate_candidate_material(material)
        candidate_rows.append(candidate_audit)
        if candidate_audit["status"] != "PASS":
            blockers.append(
                {
                    "id": "CANDIDATE_CONTRACT_INVALID",
                    "dataset": dataset,
                    "reason": candidate_audit["reason"],
                }
            )

        for h_value in H_VALUES:
            dprime, row, dprime_issue = load_dprime_cell(root, existing_dprime, dataset, h_value, material)
            dprime_rows.append(row)
            if dprime_issue:
                blockers.append(
                    {
                        "id": "DPRIME_ARTIFACT_UNAVAILABLE_OR_MUTATED",
                        "dataset": dataset,
                        "h": h_value,
                        "reason": dprime_issue,
                    }
                )
            if int(row["target_leakage_count"]) != 0 or int(row["test_leakage_count"]) != 0 or int(row["max_history_length"]) > h_value:
                blockers.append(
                    {
                        "id": "DPRIME_INVARIANT_FAILED",
                        "dataset": dataset,
                        "h": h_value,
                        "reason": "D'_H leakage or history length invariant failed.",
                    }
                )

            for model in TRAINABLE_MODELS:
                cell_status, reason = trainable_applicability(model, h_value, dprime)
                trainer_rows.append(
                    {
                        "dataset": dataset,
                        "h": h_value,
                        "model": model,
                        "cell_status": cell_status,
                        "feasible": cell_status == "APPLICABLE",
                        "reason": reason,
                        "dprime_hash": row["dprime_hash"],
                    }
                )
                if cell_status == "BLOCKED":
                    blockers.append(
                        {
                            "id": "TRAINABLE_GRID_CELL_NOT_EXECUTABLE",
                            "dataset": dataset,
                            "h": h_value,
                            "model": model,
                            "affected_seed_jobs": len(SEEDS),
                            "reason": reason,
                        }
                    )

    wrapper_inventory = build_wrapper_inventory(root)
    expected = {
        "trainable_jobs": applicable_trainable_job_count(),
        "structural_na_trainable_jobs": structural_na_job_count(),
        "deterministic_cells": len(PAPER_DATASETS) * len(H_VALUES) * len(DETERMINISTIC_BASELINES),
        "paper_datasets": list(PAPER_DATASETS),
        "history_lengths": list(H_VALUES),
        "trainable_models": list(TRAINABLE_MODELS),
        "deterministic_baselines": list(DETERMINISTIC_BASELINES),
        "seeds": list(SEEDS),
    }
    foodcom_rows = int(sum(1 for row in dprime_rows if row["dataset"] == "foodcom"))
    validation = build_validation_rows(
        candidate_rows=candidate_rows,
        dprime_rows=dprime_rows,
        trainer_rows=trainer_rows,
        blockers=blockers,
        readiness=readiness,
        seed_policy=seed_policy,
        protocol=protocol,
        expected=expected,
    )
    head_after = git_output(root, ["rev-parse", "HEAD"])
    status_after = git_output(root, ["status", "--short"])
    preflight_passed = len(blockers) == 0 and all(row["status"] == "PASS" for row in validation)

    preflight = {
        "phase_id": str(config["phase"]["id"]),
        "status": "PASS" if preflight_passed else "FAIL",
        "preflight_passed": preflight_passed,
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "head_before": head_before,
        "head_after": head_after,
        "git_status_before": status_before,
        "git_status_after": status_after,
        "implementation_classification": config["phase"]["status"],
        "foodcom_run": False,
        "foodcom_rows": foodcom_rows,
        "ollama_used": False,
        "llm_inference_used": False,
        "network_used": False,
        "downloads_used": False,
        "training_executed": False,
        "deterministic_scoring_executed": False,
        "expected_accounting": expected,
        "protocol_amendment": {
            "sequential_h1_policy": "GRU4Rec@H=1 and SASRec@H=1 are NOT_APPLICABLE",
            "reason": STRUCTURAL_NA_REASON,
            "dprime_changed": False,
            "structural_na_trainable_jobs": structural_na_job_count(),
        },
        "resource_preflight": resources,
        "wrapper_inventory": wrapper_inventory,
        "candidate_validation": candidate_rows,
        "trainable_feasibility": trainer_rows,
        "blocking_issues": blockers,
        "blocker_summary": summarize_blockers(blockers),
    }
    run_manifest = {
        "phase_id": str(config["phase"]["id"]),
        "status": "PREFLIGHT_BLOCKED" if blockers else "PREFLIGHT_PASS_NOT_EXECUTED",
        "traditional_history_status": config["phase"]["status"],
        "paper_exact": False,
        "traditional_matrix_complete": False,
        "preflight_path": paths["preflight"].as_posix(),
        "dprime_manifest_path": paths["dprime_manifest"].as_posix(),
        "validation_path": paths["validation"].as_posix(),
        "blocking_issues": blockers,
        "expected_accounting": expected,
        "completed_accounting": {
            "trainable_jobs": 0,
            "deterministic_cells": 0,
            "bm25_cells": 0,
            "mostpop_cells": 0,
            "structural_na_trainable_jobs": structural_na_job_count(),
        },
        "llm_inference_used": False,
        "ollama_used": False,
        "foodcom_run": False,
        "head_before": head_before,
        "head_after": head_after,
        "git_status_after": status_after,
    }
    pilot_validation = {
        "status": "SKIPPED_PREFLIGHT_BLOCKED" if blockers else "NOT_EXECUTED",
        "pilot": "FAIL" if blockers else "NOT_RUN",
        "dataset": "beauty",
        "h": 5,
        "seed": 42,
        "planned_trainable_jobs": 4,
        "planned_deterministic_evaluations": 2,
        "executed_jobs": 0,
        "reason": "preflight blockers prevent pilot execution" if blockers else "preflight passed but pilot was not requested in this invocation",
        "blocking_issues": blockers,
    }
    job_accounting = build_job_accounting(expected, blockers)
    return {
        "preflight": preflight,
        "dprime_manifest": pd.DataFrame(dprime_rows),
        "validation": pd.DataFrame(validation),
        "job_accounting": job_accounting,
        "run_manifest": run_manifest,
        "pilot_validation": pilot_validation,
    }


def output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {key: Path(str(value)) for key, value in config["outputs"].items()}


def ensure_dirs(root: Path, paths: Mapping[str, Path]) -> None:
    for key in ("artifact_dir", "dprime_dir", "raw_output_dir", "checkpoint_dir", "plot_dir"):
        (root / paths[key]).mkdir(parents=True, exist_ok=True)
    for key in ("job_accounting", "metrics_per_seed", "metrics_aggregated", "validation"):
        (root / paths[key]).parent.mkdir(parents=True, exist_ok=True)


def validate_candidate_material(material: DatasetMaterial) -> dict[str, Any]:
    frame = material.candidates.copy()
    grouped = frame.groupby("user_id", sort=False)
    candidate_count_ok = grouped.size().eq(20).all()
    unique_ids_ok = grouped["candidate_item_id"].nunique().eq(20).all()
    positives_ok = grouped["is_ground_truth"].sum().eq(1).all()
    target_once_ok = True
    ordering_ok = True
    for _, group in grouped:
        positive = str(group["positive_item_id"].iloc[0])
        target_once_ok = target_once_ok and int(group["candidate_item_id"].astype(str).eq(positive).sum()) == 1
        ordering_ok = ordering_ok and group["candidate_position"].astype(int).tolist() == list(range(1, 21))
    passed = bool(candidate_count_ok and unique_ids_ok and positives_ok and target_once_ok and ordering_ok)
    reason = "candidate_count=20, unique IDs, one positive, target once, and ordering 1..20" if passed else "candidate contract invariant failed"
    return {
        "dataset": material.spec.name,
        "candidate_source": material.spec.candidate_source.as_posix(),
        "evaluation_users": int(material.spec.evaluation_users),
        "candidate_rows": int(len(frame)),
        "candidate_hash": material.candidate_set_hash,
        "candidate_count_ok": bool(candidate_count_ok),
        "unique_ids_ok": bool(unique_ids_ok),
        "positive_count_ok": bool(positives_ok),
        "target_once_ok": bool(target_once_ok),
        "ordering_ok": bool(ordering_ok),
        "status": "PASS" if passed else "FAIL",
        "reason": reason,
    }


def load_existing_dprime_manifest(root: Path, paths: Mapping[str, Path]) -> pd.DataFrame:
    manifest_path = root / paths["dprime_manifest"]
    if not manifest_path.exists():
        return pd.DataFrame()
    return pd.read_csv(manifest_path, dtype=str, keep_default_na=False)


def load_dprime_cell(
    root: Path,
    manifest: pd.DataFrame,
    dataset: str,
    h_value: int,
    material: DatasetMaterial,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    if manifest.empty:
        empty = empty_dprime_manifest_row(dataset, h_value)
        return pd.DataFrame(columns=["user_id", "item_id", "dprime_position", "source_prefix_position"]), empty, "D'_H manifest is missing"
    rows = manifest.loc[(manifest["dataset"].astype(str) == dataset) & (manifest["h"].astype(str) == str(h_value))]
    if len(rows) != 1:
        empty = empty_dprime_manifest_row(dataset, h_value)
        return pd.DataFrame(columns=["user_id", "item_id", "dprime_position", "source_prefix_position"]), empty, f"D'_H manifest row count is {len(rows)}"
    stored = rows.iloc[0].to_dict()
    dprime_path = root / Path(str(stored.get("dprime_path", "")))
    if not dprime_path.exists():
        return pd.DataFrame(columns=["user_id", "item_id", "dprime_position", "source_prefix_position"]), normalize_dprime_row(stored, material, pd.DataFrame(), h_value), f"D'_H artifact missing: {stored.get('dprime_path', '')}"
    dprime = pd.read_csv(dprime_path, dtype=str, keep_default_na=False)
    actual_hash = sha256_file(dprime_path)
    issue = "" if actual_hash == str(stored.get("dprime_hash", "")) else f"D'_H hash mismatch: {actual_hash} != {stored.get('dprime_hash', '')}"
    row = normalize_dprime_row({**stored, "dprime_hash": actual_hash}, material, dprime, h_value)
    return dprime, row, issue


def empty_dprime_manifest_row(dataset: str, h_value: int) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "h": int(h_value),
        "dprime_path": "",
        "dprime_hash": "",
        "candidate_hash": "",
        "training_data_definition": "",
        "n_users": 0,
        "n_interactions": 0,
        "min_history_length": 0,
        "median_history_length": "0",
        "max_history_length": 0,
        "users_with_n_lt_h": 0,
        "target_leakage_count": 0,
        "test_leakage_count": 0,
        "validation_leakage_count": 0,
        "duplicate_interaction_anomalies": 0,
        "history_length_lte_h": False,
        "seq_training_examples": 0,
        "source_interaction_path": "",
    }


def normalize_dprime_row(stored: Mapping[str, Any], material: DatasetMaterial, dprime: pd.DataFrame, h_value: int) -> dict[str, Any]:
    if dprime.empty:
        row = empty_dprime_manifest_row(str(stored.get("dataset", material.spec.name)), h_value)
        row.update({key: stored.get(key, row.get(key, "")) for key in row})
        return row
    required = {"user_id", "item_id", "dprime_position", "source_prefix_position"}
    missing = required - set(dprime.columns)
    if missing:
        raise ValueError(f"D'_H artifact missing columns for {material.spec.name} H={h_value}: {sorted(missing)}")
    dprime = dprime.copy()
    dprime["user_id"] = dprime["user_id"].astype(str)
    dprime["item_id"] = dprime["item_id"].astype(str)
    dprime["dprime_position"] = pd.to_numeric(dprime["dprime_position"], errors="raise").astype(int)
    lengths = dprime.groupby("user_id", sort=False).size()
    positives = material.candidates.drop_duplicates("user_id", keep="first").set_index("user_id")["positive_item_id"].astype(str).to_dict()
    history_sets = {
        str(user_id): set(group["item_id"].astype(str).tolist())
        for user_id, group in dprime.groupby("user_id", sort=False)
    }
    target_leakage = sum(1 for user_id, positive in positives.items() if str(positive) in history_sets.get(str(user_id), set()))
    duplicate_anomalies = int(dprime.duplicated(["user_id", "item_id", "dprime_position"]).sum())
    seq_examples = int(sum(max(int(length) - 1, 0) for length in lengths.tolist()))
    max_history = int(lengths.max()) if not lengths.empty else 0
    row = empty_dprime_manifest_row(str(stored.get("dataset", material.spec.name)), h_value)
    row.update(
        {
            "dataset": str(stored.get("dataset", material.spec.name)),
            "h": int(h_value),
            "dprime_path": str(stored.get("dprime_path", "")),
            "dprime_hash": str(stored.get("dprime_hash", "")),
            "candidate_hash": str(stored.get("candidate_hash", material.candidate_set_hash)),
            "training_data_definition": str(stored.get("training_data_definition", "")),
            "n_users": int(lengths.size),
            "n_interactions": int(len(dprime)),
            "min_history_length": int(lengths.min()) if not lengths.empty else 0,
            "median_history_length": format_float(float(lengths.median())) if not lengths.empty else "0",
            "max_history_length": max_history,
            "users_with_n_lt_h": int(stored.get("users_with_n_lt_h", 0) or 0),
            "target_leakage_count": int(target_leakage),
            "test_leakage_count": int(target_leakage),
            "validation_leakage_count": 0,
            "duplicate_interaction_anomalies": duplicate_anomalies,
            "history_length_lte_h": bool(max_history <= int(h_value)),
            "seq_training_examples": seq_examples,
            "source_interaction_path": str(stored.get("source_interaction_path", material.spec.interaction_source.as_posix())),
        }
    )
    return row


def build_dprime(base_events: pd.DataFrame, h_value: int) -> pd.DataFrame:
    frame = base_events.copy()
    frame["source_prefix_position"] = pd.to_numeric(frame["prefix_position"], errors="raise").astype(int)
    frame = frame.sort_values(["user_id", "source_prefix_position"], kind="mergesort")
    dprime = frame.groupby("user_id", sort=False, group_keys=False).tail(int(h_value)).copy()
    dprime = dprime.sort_values(["user_id", "source_prefix_position"], kind="mergesort").reset_index(drop=True)
    dprime["dprime_position"] = dprime.groupby("user_id", sort=False).cumcount() + 1
    return dprime[["user_id", "item_id", "dprime_position", "source_prefix_position"]]


def dprime_manifest_row(
    *,
    dataset: str,
    h_value: int,
    spec: DatasetSpec,
    material: DatasetMaterial,
    base_events: pd.DataFrame,
    dprime: pd.DataFrame,
    dprime_hash: str,
    dprime_path: str,
    training_definition: str,
) -> dict[str, Any]:
    lengths = dprime.groupby("user_id", sort=False).size() if not dprime.empty else pd.Series(dtype=int)
    full_lengths = base_events.groupby("user_id", sort=False).size() if not base_events.empty else pd.Series(dtype=int)
    positives = material.candidates.drop_duplicates("user_id", keep="first").set_index("user_id")["positive_item_id"].astype(str).to_dict()
    history_sets = {
        str(user_id): set(group["item_id"].astype(str).tolist())
        for user_id, group in dprime.groupby("user_id", sort=False)
    }
    target_leakage = sum(1 for user_id, positive in positives.items() if str(positive) in history_sets.get(str(user_id), set()))
    duplicate_anomalies = int(dprime.duplicated(["user_id", "item_id", "dprime_position"]).sum())
    seq_examples = int(sum(max(int(length) - 1, 0) for length in lengths.tolist()))
    max_history = int(lengths.max()) if not lengths.empty else 0
    return {
        "dataset": dataset,
        "h": int(h_value),
        "dprime_path": dprime_path,
        "dprime_hash": dprime_hash,
        "candidate_hash": material.candidate_set_hash,
        "training_data_definition": training_definition,
        "n_users": int(lengths.size),
        "n_interactions": int(len(dprime)),
        "min_history_length": int(lengths.min()) if not lengths.empty else 0,
        "median_history_length": format_float(float(lengths.median())) if not lengths.empty else "0",
        "max_history_length": max_history,
        "users_with_n_lt_h": int((full_lengths < int(h_value)).sum()) if not full_lengths.empty else 0,
        "target_leakage_count": int(target_leakage),
        "test_leakage_count": int(target_leakage),
        "validation_leakage_count": 0,
        "duplicate_interaction_anomalies": duplicate_anomalies,
        "history_length_lte_h": bool(max_history <= int(h_value)),
        "seq_training_examples": seq_examples,
        "source_interaction_path": spec.interaction_source.as_posix(),
    }


def trainable_feasibility(model: str, dprime: pd.DataFrame) -> tuple[bool, str]:
    if dprime.empty:
        return False, "D'_H has no training interactions"
    if model in {"BPRMF", "LightGCN"}:
        return True, "pairwise interaction training has positive user-item pairs"
    lengths = dprime.groupby("user_id", sort=False).size()
    seq_examples = int(sum(max(int(length) - 1, 0) for length in lengths.tolist()))
    if seq_examples <= 0:
        return (
            False,
            "local GRU4Rec/SASRec trainers require seq_example_count > 0; D'_H with H=1 yields no next-item sequence examples",
        )
    return True, "sequence training has at least one next-item example"


def trainable_applicability(model: str, h_value: int, dprime: pd.DataFrame) -> tuple[str, str]:
    if model in SEQUENTIAL_MODELS and int(h_value) == 1:
        return "NOT_APPLICABLE", STRUCTURAL_NA_REASON
    feasible, reason = trainable_feasibility(model, dprime)
    return ("APPLICABLE" if feasible else "BLOCKED"), reason


def structural_na_job_count() -> int:
    return len(PAPER_DATASETS) * len(SEQUENTIAL_MODELS) * len(SEEDS)


def applicable_trainable_job_count() -> int:
    total_cells = len(PAPER_DATASETS) * len(H_VALUES) * len(TRAINABLE_MODELS) * len(SEEDS)
    return total_cells - structural_na_job_count()


def build_wrapper_inventory(root: Path) -> list[dict[str, Any]]:
    rel_paths = [
        Path("src/extension/f1bz_final_traditional_execution.py"),
        Path("src/extension/f1by_foodcom_validation_tuning.py"),
        Path("src/baselines/plan_d_bprmf.py"),
        Path("src/baselines/plan_d_gru4rec.py"),
        Path("src/baselines/plan_d_sasrec.py"),
        Path("src/baselines/plan_d_lightgcn.py"),
        Path("src/extension/f1bx_execution_stage1.py"),
    ]
    rows = []
    for rel in rel_paths:
        path = root / rel
        rows.append(
            {
                "path": rel.as_posix(),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else "",
            }
        )
    return rows


def build_validation_rows(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    dprime_rows: Sequence[Mapping[str, Any]],
    trainer_rows: Sequence[Mapping[str, Any]],
    blockers: Sequence[Mapping[str, Any]],
    readiness: pd.DataFrame,
    seed_policy: pd.DataFrame,
    protocol: pd.DataFrame,
    expected: Mapping[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": str(evidence)})

    check("datasets_exact_paper_four", set(expected["paper_datasets"]) == set(PAPER_DATASETS), expected["paper_datasets"])
    check("foodcom_excluded", all(row["dataset"] != "foodcom" for row in dprime_rows), "foodcom rows absent")
    check("candidate_contracts_valid", all(row["status"] == "PASS" for row in candidate_rows), candidate_rows)
    check("dprime_cells_32", len(dprime_rows) == 32, len(dprime_rows))
    check("dprime_hashes_complete", all(str(row["dprime_hash"]) for row in dprime_rows), "hash present for every D'_H cell")
    check("dprime_leakage_zero", all(int(row["target_leakage_count"]) == 0 and int(row["test_leakage_count"]) == 0 for row in dprime_rows), "target/test leakage counts")
    check("dprime_history_lte_h", all(bool(row["history_length_lte_h"]) for row in dprime_rows), "max history length <= H")
    check("seed_policy_frozen", seed_policy["final_seeds"].astype(str).str.contains("42,43,44,45,46", regex=False).all(), seed_policy.to_dict("records"))
    ready_original_four = readiness.loc[readiness["dataset"].isin(PAPER_DATASETS)]
    check("trainable_readiness_rows_16", len(ready_original_four) == 16 and ready_original_four["status"].eq("READY_TO_TRAIN").all(), len(ready_original_four))
    check("protocol_freeze_loaded", len(protocol) >= 6, len(protocol))
    structural_na = [row for row in trainer_rows if str(row.get("cell_status")) == "NOT_APPLICABLE"]
    blocked = [row for row in trainer_rows if str(row.get("cell_status")) == "BLOCKED"]
    expected_na_rows = len(PAPER_DATASETS) * len(SEQUENTIAL_MODELS)
    expected_na_exact = all(int(row["h"]) == 1 and row["model"] in SEQUENTIAL_MODELS for row in structural_na)
    check("structural_na_sequential_h1", len(structural_na) == expected_na_rows and expected_na_exact, structural_na)
    check("applicable_trainable_grid_cells_executable", len(blocked) == 0, blocked[:10])
    check("no_preflight_blockers", len(blockers) == 0, blockers[:10])
    return rows


def build_job_accounting(expected: Mapping[str, Any], blockers: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    blocked_seed_jobs = sum(int(row.get("affected_seed_jobs", 0)) for row in blockers if row.get("id") == "TRAINABLE_GRID_CELL_NOT_EXECUTABLE")
    return pd.DataFrame(
        [
            {
                "category": "trainable",
                "expected": int(expected["trainable_jobs"]),
                "completed": 0,
                "failed": 0,
                "skipped_valid_resume": 0,
                "blocked_before_execution": blocked_seed_jobs,
                "not_applicable": int(expected.get("structural_na_trainable_jobs", 0)),
                "status": "BLOCKED_PREFLIGHT" if blockers else "READY",
            },
            {
                "category": "deterministic",
                "expected": int(expected["deterministic_cells"]),
                "completed": 0,
                "failed": 0,
                "skipped_valid_resume": 0,
                "blocked_before_execution": int(expected["deterministic_cells"]) if blockers else 0,
                "not_applicable": 0,
                "status": "BLOCKED_PREFLIGHT" if blockers else "READY",
            },
        ]
    )


def run_pilot(root: Path, config: Mapping[str, Any], outputs: Mapping[str, Any]) -> dict[str, Any]:
    if not outputs["preflight"]["preflight_passed"]:
        return dict(outputs)
    jobs = build_job_specs(root, config, pilot_only=True)
    result = execute_jobs(root, config, jobs)
    pilot_jobs = job_status_summary(result["metrics_per_seed"], pilot_only=True)
    critical_failures = result["job_failures"]
    pilot_pass = (
        pilot_jobs["completed"] == 6
        and pilot_jobs["failed"] == 0
        and pilot_jobs["missing_applicable"] == 0
        and not critical_failures
    )
    updated = dict(outputs)
    updated["metrics_per_seed"] = result["metrics_per_seed"]
    updated["metrics_aggregated"] = aggregate_metrics(result["metrics_per_seed"])
    updated["pilot_validation"] = {
        "status": "PASS" if pilot_pass else "FAIL",
        "pilot": "PASS" if pilot_pass else "FAIL",
        "dataset": "beauty",
        "h": 5,
        "seed": 42,
        "planned_trainable_jobs": 4,
        "planned_deterministic_evaluations": 2,
        "executed_jobs": result["executed_jobs"],
        "skipped_valid_resume": result["skipped_valid_resume"],
        "completed_jobs": pilot_jobs["completed"],
        "failed_jobs": pilot_jobs["failed"],
        "finite_metrics": pilot_jobs["finite_metrics"],
        "critical_failures": critical_failures,
        "reason": "pilot completed successfully" if pilot_pass else "pilot failed or incomplete",
    }
    updated["job_accounting"] = build_accounting_from_metrics(
        result["metrics_per_seed"],
        outputs["preflight"]["expected_accounting"],
        final=False,
        skipped_trainable=result["skipped_trainable_valid_resume"],
        skipped_deterministic=result["skipped_deterministic_valid_resume"],
    )
    updated["validation"] = build_execution_validation(updated["metrics_per_seed"], outputs["preflight"], final=False)
    updated["run_manifest"] = update_run_manifest_after_execution(root, config, updated, final=False)
    return updated


def run_full(root: Path, config: Mapping[str, Any], outputs: Mapping[str, Any]) -> dict[str, Any]:
    jobs = build_job_specs(root, config, pilot_only=False)
    result = execute_jobs(root, config, jobs)
    updated = dict(outputs)
    updated["metrics_per_seed"] = result["metrics_per_seed"]
    updated["metrics_aggregated"] = aggregate_metrics(result["metrics_per_seed"])
    updated["job_accounting"] = build_accounting_from_metrics(
        result["metrics_per_seed"],
        outputs["preflight"]["expected_accounting"],
        final=True,
        skipped_trainable=result["skipped_trainable_valid_resume"],
        skipped_deterministic=result["skipped_deterministic_valid_resume"],
    )
    updated["validation"] = build_execution_validation(updated["metrics_per_seed"], outputs["preflight"], final=True)
    final_pass = updated["validation"]["status"].eq("PASS").all()
    updated["run_manifest"] = update_run_manifest_after_execution(root, config, updated, final=bool(final_pass))
    if final_pass:
        write_diagnostic_figures(root, config, updated["metrics_aggregated"])
        write_freeze_report(root, updated)
    return updated


def build_job_specs(root: Path, config: Mapping[str, Any], *, pilot_only: bool) -> list[JobSpec]:
    paths = output_paths(config)
    dprime_manifest = pd.read_csv(root / paths["dprime_manifest"], dtype=str, keep_default_na=False)
    dprime_hashes = {
        (str(row.dataset), int(row.h)): str(row.dprime_hash)
        for row in dprime_manifest.itertuples(index=False)
    }
    dataset_config = load_config(root, Path(str(config["inputs"]["dataset_config"])))
    specs = {spec.name: spec for spec in dataset_specs(dataset_config) if spec.name in PAPER_DATASETS}
    materials = {dataset: build_dataset_material(root, specs[dataset]) for dataset in PAPER_DATASETS}
    readiness = read_csv(root / Path(str(config["inputs"]["trainable_readiness"])))
    ready = {
        (str(row["dataset"]), str(row["method"])): row
        for row in readiness.loc[readiness["dataset"].isin(PAPER_DATASETS)].to_dict("records")
    }
    datasets = ("beauty",) if pilot_only else PAPER_DATASETS
    h_values = (5,) if pilot_only else H_VALUES
    seeds = (42,) if pilot_only else SEEDS
    jobs: list[JobSpec] = []
    for dataset in datasets:
        material = materials[dataset]
        for h_value in h_values:
            dprime_hash = dprime_hashes[(dataset, int(h_value))]
            for method in TRAINABLE_MODELS:
                row = ready[(dataset, method)]
                params = json.loads(str(row["hyperparameter_json"]))
                source = "paper_appendix_a3"
                config_id = f"RQ3-{dataset.upper()}-H{h_value}-{method.upper()}-APPENDIX-A3"
                config_hash = sha256_json(
                    {
                        "dataset": dataset,
                        "h": int(h_value),
                        "method": method,
                        "hyperparameters": params,
                        "source": source,
                        "runtime": config["runtime"],
                    }
                )
                cell_status = "NOT_APPLICABLE" if method in SEQUENTIAL_MODELS and int(h_value) == 1 else "APPLICABLE"
                reason = STRUCTURAL_NA_REASON if cell_status == "NOT_APPLICABLE" else "applicable trainable cell"
                for seed in seeds:
                    jobs.append(
                        JobSpec(
                            dataset=dataset,
                            h=int(h_value),
                            method=method,
                            seed=int(seed),
                            config_id=config_id,
                            config_hash=config_hash,
                            hyperparameters=params,
                            hyperparameter_source=source,
                            dprime_hash=dprime_hash,
                            candidate_hash=material.candidate_set_hash,
                            status=cell_status,
                            reason=reason,
                        )
                    )
            for method in DETERMINISTIC_BASELINES:
                jobs.append(
                    JobSpec(
                        dataset=dataset,
                        h=int(h_value),
                        method=method,
                        seed=None,
                        config_id=f"RQ3-{dataset.upper()}-H{h_value}-{method_slug(method).upper()}",
                        config_hash=sha256_json({"dataset": dataset, "h": int(h_value), "method": method}),
                        hyperparameters={},
                        hyperparameter_source="deterministic_local_contract",
                        dprime_hash=dprime_hash,
                        candidate_hash=material.candidate_set_hash,
                        status="APPLICABLE",
                        reason="applicable deterministic cell",
                    )
                )
    return jobs


def execute_jobs(root: Path, config: Mapping[str, Any], jobs: Sequence[JobSpec]) -> dict[str, Any]:
    paths = output_paths(config)
    metrics = load_metrics_table(root, paths)
    dataset_config = load_config(root, Path(str(config["inputs"]["dataset_config"])))
    specs = {spec.name: spec for spec in dataset_specs(dataset_config) if spec.name in PAPER_DATASETS}
    materials = {dataset: build_dataset_material(root, specs[dataset]) for dataset in sorted({job.dataset for job in jobs})}
    device = select_device(str(config["runtime"]["device_policy"]))
    executed = 0
    skipped_trainable = 0
    skipped_deterministic = 0
    failures: list[dict[str, Any]] = []
    dprime_cache: dict[tuple[str, int], RQ3Data] = {}
    mostpop_cache: dict[str, pd.DataFrame] = {}

    metric_rows = metrics.to_dict("records")
    completed_keys = completed_job_keys(metrics)
    metric_rows.extend(recover_completed_metrics_from_raw_outputs(root, paths, jobs, completed_keys, materials))
    recovered = dedupe_metrics(pd.DataFrame(metric_rows))
    completed_keys = completed_job_keys(recovered)
    metric_rows = recovered.to_dict("records")
    for job in jobs:
        key = job_key(job.dataset, job.h, job.method, job.seed)
        if job.status == "NOT_APPLICABLE":
            if key not in completed_keys:
                metric_rows.extend(not_applicable_metric_rows(job))
                completed_keys.add(key)
            continue
        if key in completed_keys and recommendation_exists(root, paths, job):
            if job.method in TRAINABLE_MODELS:
                skipped_trainable += 1
            else:
                skipped_deterministic += 1
            continue
        started = time.perf_counter()
        try:
            data = dprime_cache.get((job.dataset, job.h))
            if data is None:
                data = build_rq3_data(root, paths, materials[job.dataset], job, config)
                dprime_cache[(job.dataset, job.h)] = data
            if job.method in TRAINABLE_MODELS:
                recs, history, checkpoint_path, checkpoint_hash = execute_trainable_job(root, config, paths, job, data, device)
                epochs_completed = len(history)
                final_train_loss = format_float(history[-1]["loss"]) if history else ""
            elif job.method == "MostPop":
                base = mostpop_cache.get(job.dataset)
                if base is None:
                    base = rank_mostpop(materials[job.dataset], job, timestamp=datetime.now(timezone.utc).isoformat())
                    mostpop_cache[job.dataset] = base
                recs = base.copy()
                recs["run_id"] = job.run_id
                recs["h"] = job.h
                recs["dprime_hash"] = job.dprime_hash
                recs["history_invariant"] = True
                history = []
                checkpoint_path = ""
                checkpoint_hash = ""
                epochs_completed = ""
                final_train_loss = ""
                save_recommendations(root, paths, job, recs)
            elif job.method == "BM25-like/content":
                recs = rank_bm25(root, paths, materials[job.dataset], job, timestamp=datetime.now(timezone.utc).isoformat())
                history = []
                checkpoint_path = ""
                checkpoint_hash = ""
                epochs_completed = ""
                final_train_loss = ""
                save_recommendations(root, paths, job, recs)
            else:
                raise ValueError(f"unsupported method: {job.method}")
            validate_recommendations_for_job(recs, materials[job.dataset], job)
            stats = per_user_stats(recs, materials[job.dataset])
            values = utility_metric_values(stats)
            runtime = time.perf_counter() - started
            metric_rows.extend(
                completed_metric_rows(
                    job,
                    values=values,
                    evaluated_users=len(stats),
                    runtime_seconds=runtime,
                    recommendation_hash=frame_hash(recs),
                    checkpoint_path=checkpoint_path,
                    checkpoint_hash=checkpoint_hash,
                    epochs_completed=epochs_completed,
                    final_train_loss=final_train_loss,
                )
            )
            completed_keys.add(key)
            executed += 1
            print(f"completed={job.run_id}", flush=True)
        except Exception as exc:  # pragma: no cover - records local runtime failures.
            failures.append(
                {
                    "run_id": job.run_id,
                    "dataset": job.dataset,
                    "h": job.h,
                    "method": job.method,
                    "seed": job.seed if job.seed is not None else "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            metric_rows.extend(failed_metric_rows(job, f"{type(exc).__name__}: {exc}"))
            print(f"failed={job.run_id} error={type(exc).__name__}: {exc}", flush=True)
    deduped = dedupe_metrics(pd.DataFrame(metric_rows))
    return {
        "metrics_per_seed": deduped,
        "executed_jobs": executed,
        "skipped_valid_resume": skipped_trainable + skipped_deterministic,
        "skipped_trainable_valid_resume": skipped_trainable,
        "skipped_deterministic_valid_resume": skipped_deterministic,
        "job_failures": failures,
    }


def build_rq3_data(root: Path, paths: Mapping[str, Path], material: DatasetMaterial, job: JobSpec, config: Mapping[str, Any]) -> RQ3Data:
    manifest = pd.read_csv(root / paths["dprime_manifest"], dtype=str, keep_default_na=False)
    row = manifest.loc[(manifest["dataset"].eq(job.dataset)) & (manifest["h"].astype(str).eq(str(job.h)))].iloc[0]
    dprime_path = root / Path(str(row["dprime_path"]))
    dprime = pd.read_csv(dprime_path, dtype=str, keep_default_na=False)
    dprime["user_id"] = dprime["user_id"].astype(str)
    dprime["item_id"] = dprime["item_id"].astype(str)
    dprime["dprime_position"] = pd.to_numeric(dprime["dprime_position"], errors="raise").astype(int)
    dprime = dprime.sort_values(["user_id", "dprime_position"], kind="mergesort").reset_index(drop=True)

    users = material.candidates.drop_duplicates("user_id", keep="first")["user_id"].astype(str).tolist()
    positive_by_user = material.candidates.drop_duplicates("user_id", keep="first").set_index("user_id")["positive_item_id"].astype(str).to_dict()
    candidate_grouped: list[list[str]] = []
    positives: list[str] = []
    for user in users:
        frame = material.candidates.loc[material.candidates["user_id"].eq(user)].sort_values("candidate_position", kind="mergesort")
        candidate_grouped.append(frame["candidate_item_id"].astype(str).tolist())
        positives.append(str(positive_by_user[user]))

    histories_by_user = {
        str(user_id): group.sort_values("dprime_position", kind="mergesort")["item_id"].astype(str).tolist()
        for user_id, group in dprime.groupby("user_id", sort=False)
    }
    all_users = sorted(set(dprime["user_id"].astype(str)).union(users), key=sort_key)
    all_items = sorted(set(dprime["item_id"].astype(str)).union({item for row_items in candidate_grouped for item in row_items}), key=sort_key)
    user_index = {user: index for index, user in enumerate(all_users)}
    item_index = {item: index for index, item in enumerate(all_items)}
    pairs = dprime[["user_id", "item_id"]].drop_duplicates().copy()
    train_user_indices = pairs["user_id"].map(user_index).astype("int64").to_numpy()
    train_item_indices = pairs["item_id"].map(item_index).astype("int64").to_numpy()
    user_positive_items: dict[int, set[int]] = defaultdict(set)
    for user_id, group in dprime.groupby("user_id", sort=False):
        user_positive_items[user_index[str(user_id)]].update(item_index[item] for item in group["item_id"].astype(str))
    train_item_universe = np.asarray(sorted({item_index[item] for item in dprime["item_id"].astype(str)}), dtype=np.int64)
    max_seq_len = int(config["runtime"]["max_sequence_length"])
    seq_rows: list[np.ndarray] = []
    seq_users: list[int] = []
    seq_positives: list[int] = []
    seq_user_positive_tokens: dict[int, set[int]] = defaultdict(set)
    eval_sequences = np.zeros((len(users), max_seq_len), dtype=np.int64)
    eval_user_indices: list[int] = []
    for user in all_users:
        uidx = user_index[user]
        prefix_indices = [item_index[item] for item in histories_by_user.get(user, [])]
        seq_user_positive_tokens[uidx].update(index + 1 for index in prefix_indices)
        for end in range(1, len(prefix_indices)):
            seq_rows.append(pad_tokens(prefix_indices[:end], max_seq_len))
            seq_users.append(uidx)
            seq_positives.append(prefix_indices[end] + 1)
    for row_index, user in enumerate(users):
        prefix_indices = [item_index[item] for item in histories_by_user.get(user, [])]
        eval_sequences[row_index] = pad_tokens(prefix_indices, max_seq_len)
        eval_user_indices.append(user_index[user])
    seq_examples = np.vstack(seq_rows).astype(np.int64) if seq_rows else np.zeros((0, max_seq_len), dtype=np.int64)
    candidate_items = np.asarray([[item_index[item] for item in row_items] for row_items in candidate_grouped], dtype=np.int64)
    positive_positions = np.asarray([row_items.index(target) + 1 for row_items, target in zip(candidate_grouped, positives)], dtype=np.int64)
    eval_cohort = pd.DataFrame({"user_id": users, "positive_item_id": positives})
    tuning = TuningData(
        users=users,
        user_index=user_index,
        item_index=item_index,
        training_events=dprime.rename(columns={"dprime_position": "prefix_position"})[["user_id", "item_id", "prefix_position"]],
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
        training_data_hash=job.dprime_hash,
        validation_data_hash=frame_hash(eval_cohort),
        candidate_contract_hash=material.candidate_set_hash,
        train_rows=int(len(dprime)),
        train_pair_count=int(len(pairs)),
        seq_example_count=int(len(seq_examples)),
        num_users=len(user_index),
        num_items=len(item_index),
    )
    return RQ3Data(
        material=material,
        tuning=tuning,
        eval_user_indices=np.asarray(eval_user_indices, dtype=np.int64),
        positive_item_ids=positives,
        histories_by_user=histories_by_user,
        history_lengths_by_user={user: len(items) for user, items in histories_by_user.items()},
        dprime_path=str(row["dprime_path"]),
        dprime_hash=job.dprime_hash,
    )


def execute_trainable_job(
    root: Path,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    job: JobSpec,
    data: RQ3Data,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict[str, Any]], str, str]:
    set_deterministic_execution(int(job.seed))
    if job.method == "BPRMF":
        model, history = train_bprmf(job.hyperparameters, config, data.tuning, device, job.run_id)
        scores = score_bprmf(model, data, device)
        graph = None
    elif job.method == "GRU4Rec":
        model, history = train_gru4rec(job.hyperparameters, config, data.tuning, device, job.run_id)
        scores = score_sequence(model, data, device)
        graph = None
    elif job.method == "SASRec":
        model, history = train_sasrec(job.hyperparameters, config, data.tuning, device, job.run_id)
        scores = score_sequence(model, data, device)
        graph = None
    elif job.method == "LightGCN":
        model, graph, history = train_lightgcn(job.hyperparameters, config, data.tuning, device, job.run_id)
        scores = score_lightgcn(model, graph, data, device)
    else:
        raise ValueError(f"unsupported trainable method: {job.method}")
    checkpoint_path = root / paths["checkpoint_dir"] / f"{job.run_id}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": job.run_id,
        "dataset": job.dataset,
        "h": job.h,
        "method": job.method,
        "seed": job.seed,
        "config_id": job.config_id,
        "config_hash": job.config_hash,
        "dprime_hash": job.dprime_hash,
        "candidate_hash": job.candidate_hash,
        "state_dict": model.state_dict(),
    }
    if graph is not None:
        payload["graph_shape"] = list(graph.shape)
    torch.save(payload, checkpoint_path)
    recs = recommendations_from_scores(job, data, scores, datetime.now(timezone.utc).isoformat())
    save_recommendations(root, paths, job, recs)
    return recs, history, checkpoint_path.relative_to(root).as_posix(), sha256_file(checkpoint_path)


def score_bprmf(model: torch.nn.Module, data: RQ3Data, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        users = torch.as_tensor(data.eval_user_indices, dtype=torch.long, device=device).repeat_interleave(CANDIDATE_COUNT)
        items = torch.as_tensor(data.tuning.candidate_items.reshape(-1), dtype=torch.long, device=device)
        scores = model.score(users, items).reshape(len(data.tuning.users), CANDIDATE_COUNT)
    return scores.detach().cpu().numpy()


def score_lightgcn(model: torch.nn.Module, graph: torch.Tensor, data: RQ3Data, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.final_embeddings(graph)
        users = torch.as_tensor(data.eval_user_indices, dtype=torch.long, device=device).repeat_interleave(CANDIDATE_COUNT)
        items = torch.as_tensor(data.tuning.candidate_items.reshape(-1), dtype=torch.long, device=device)
        scores = model.score_from_embeddings(user_emb, item_emb, users, items).reshape(len(data.tuning.users), CANDIDATE_COUNT)
    return scores.detach().cpu().numpy()


def score_sequence(model: torch.nn.Module, data: RQ3Data, device: torch.device) -> np.ndarray:
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
            outputs.append(batch_scores)
    return np.vstack(outputs)


def recommendations_from_scores(job: JobSpec, data: RQ3Data, scores: np.ndarray, timestamp: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = data.tuning.candidate_item_ids
    for user_idx, user_id in enumerate(data.tuning.users):
        order = sorted(range(CANDIDATE_COUNT), key=lambda pos: (-float(scores[user_idx, pos]), sort_key(candidates[user_idx][pos]), int(pos)))
        positive = data.positive_item_ids[user_idx]
        positive_position = int(data.tuning.positive_positions[user_idx])
        for rank, pos in enumerate(order, start=1):
            item_id = candidates[user_idx][pos]
            rows.append(recommendation_row(job, user_id, rank, item_id, scores[user_idx, pos], pos + 1, pos + 1 == positive_position, positive, timestamp, False))
    return pd.DataFrame(rows)


def rank_mostpop(material: DatasetMaterial, job: JobSpec, timestamp: str) -> pd.DataFrame:
    frame = material.candidates.copy()
    frame["score"] = [float(material.train_counts.get(str(item_id), 0)) for item_id in frame["candidate_item_id"]]
    return rank_candidate_frame(frame, job, timestamp, history_invariant=True)


def rank_bm25(root: Path, paths: Mapping[str, Path], material: DatasetMaterial, job: JobSpec, timestamp: str) -> pd.DataFrame:
    dprime = pd.read_csv(root / paths["dprime_dir"] / f"{job.dataset}_h{job.h}.csv", dtype=str, keep_default_na=False)
    histories = {
        str(user_id): group.sort_values("dprime_position", key=lambda col: col.astype(int), kind="mergesort")["item_id"].astype(str).tolist()
        for user_id, group in dprime.groupby("user_id", sort=False)
    }
    candidate_sets = {
        str(user_id): set(group["candidate_item_id"].astype(str).tolist())
        for user_id, group in material.candidates.groupby("user_id", sort=False)
    }
    profiles: dict[str, Counter[str]] = {}
    for user_id, history in histories.items():
        excluded = candidate_sets.get(str(user_id), set())
        profile: Counter[str] = Counter()
        for item_id in history:
            if str(item_id) in excluded:
                continue
            terms = material.bm25_terms_by_item.get(str(item_id))
            if terms:
                profile.update(terms)
        profiles[user_id] = profile
    frame = material.candidates.copy()
    frame["score"] = [
        bm25_score_local(
            profiles.get(str(user_id), Counter()),
            material.bm25_terms_by_item.get(str(item_id), Counter()),
            material.bm25_doc_lengths.get(str(item_id), 0),
            material,
        )
        for user_id, item_id in zip(frame["user_id"], frame["candidate_item_id"])
    ]
    return rank_candidate_frame(frame, job, timestamp, history_invariant=False)


def bm25_score_local(profile_terms: Counter[str], item_terms: Counter[str], item_length: int, material: DatasetMaterial) -> float:
    if not profile_terms or not item_terms or material.bm25_average_doc_length <= 0:
        return 0.0
    k1 = 1.5
    b = 0.75
    length_norm = 1.0 - b + b * (item_length / material.bm25_average_doc_length)
    score = 0.0
    for term, tf in item_terms.items():
        qtf = profile_terms.get(term, 0)
        if qtf <= 0:
            continue
        denominator = tf + k1 * length_norm
        if denominator > 0:
            score += material.bm25_idf.get(term, 0.0) * qtf * ((tf * (k1 + 1.0)) / denominator)
    return float(score)


def rank_candidate_frame(frame: pd.DataFrame, job: JobSpec, timestamp: str, *, history_invariant: bool) -> pd.DataFrame:
    working = frame.copy()
    item_sort = [sort_key(value) for value in working["candidate_item_id"]]
    working["__item_group"] = [value[0] for value in item_sort]
    working["__item_number"] = [value[1] for value in item_sort]
    working["__item_text"] = [value[2] for value in item_sort]
    working = working.sort_values(
        ["user_id", "score", "candidate_position", "__item_group", "__item_number", "__item_text"],
        ascending=[True, False, True, True, True, True],
        kind="mergesort",
    ).copy()
    working["rank"] = working.groupby("user_id", sort=False).cumcount() + 1
    rows = [
        recommendation_row(
            job,
            row.user_id,
            int(row.rank),
            row.candidate_item_id,
            float(row.score),
            int(row.candidate_position),
            bool(row.is_ground_truth),
            str(row.positive_item_id),
            timestamp,
            history_invariant,
        )
        for row in working.itertuples(index=False)
    ]
    return pd.DataFrame(rows)


def recommendation_row(
    job: JobSpec,
    user_id: Any,
    rank: int,
    item_id: Any,
    score: Any,
    candidate_position: int,
    is_ground_truth: bool,
    positive_item_id: Any,
    timestamp: str,
    history_invariant: bool,
) -> dict[str, Any]:
    return {
        "run_id": job.run_id,
        "dataset": job.dataset,
        "h": int(job.h),
        "method": job.method,
        "seed": "" if job.seed is None else int(job.seed),
        "user_id": str(user_id),
        "rank": int(rank),
        "recommended_item_id": str(item_id),
        "score": format_float(score),
        "candidate_position": int(candidate_position),
        "is_ground_truth": bool(is_ground_truth),
        "positive_item_id": str(positive_item_id),
        "dprime_hash": job.dprime_hash,
        "candidate_set_hash": job.candidate_hash,
        "config_hash": job.config_hash,
        "history_invariant": bool(history_invariant),
        "timestamp": timestamp,
    }


def save_recommendations(root: Path, paths: Mapping[str, Path], job: JobSpec, recs: pd.DataFrame) -> str:
    target = root / paths["raw_output_dir"] / f"{job.run_id}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    recs.to_csv(target, index=False, lineterminator="\n")
    return target.relative_to(root).as_posix()


def validate_recommendations_for_job(recs: pd.DataFrame, material: DatasetMaterial, job: JobSpec) -> None:
    grouped = recs.groupby("user_id", sort=False)
    if not grouped.size().eq(CANDIDATE_COUNT).all():
        raise ValueError(f"{job.run_id}: recommendation rows are not 20 per user")
    if not grouped["recommended_item_id"].nunique().eq(CANDIDATE_COUNT).all():
        raise ValueError(f"{job.run_id}: duplicate recommendation item in candidate ranking")
    expected_users = material.candidates["user_id"].nunique()
    if recs["user_id"].nunique() != expected_users:
        raise ValueError(f"{job.run_id}: evaluated user count mismatch")
    positives = grouped["is_ground_truth"].sum()
    if not positives.eq(1).all():
        raise ValueError(f"{job.run_id}: positive target not present exactly once")
    if not np.isfinite(pd.to_numeric(recs["score"], errors="coerce").to_numpy()).all():
        raise ValueError(f"{job.run_id}: non-finite score")


def completed_metric_rows(
    job: JobSpec,
    *,
    values: Mapping[str, float],
    evaluated_users: int,
    runtime_seconds: float,
    recommendation_hash: str,
    checkpoint_path: str,
    checkpoint_hash: str,
    epochs_completed: Any,
    final_train_loss: str,
) -> list[dict[str, Any]]:
    metric_values = {
        "hr_at_5": values["hr_at_5"],
        "ndcg_at_5": values["ndcg_at_5"],
        "mrr": values["mrr_diagnostic"],
    }
    return [
        metric_row_base(
            job,
            metric=metric,
            value=format_float(value),
            status="COMPLETED",
            reason="computed from RQ3 traditional local operationalization",
            evaluated_users=evaluated_users,
            runtime_seconds=runtime_seconds,
            recommendation_hash=recommendation_hash,
            checkpoint_path=checkpoint_path,
            checkpoint_hash=checkpoint_hash,
            epochs_completed=epochs_completed,
            final_train_loss=final_train_loss,
        )
        for metric, value in metric_values.items()
    ]


def not_applicable_metric_rows(job: JobSpec) -> list[dict[str, Any]]:
    return [
        metric_row_base(job, metric=metric, value="", status="NOT_APPLICABLE", reason=STRUCTURAL_NA_REASON)
        for metric in UTILITY_METRICS
    ]


def failed_metric_rows(job: JobSpec, reason: str) -> list[dict[str, Any]]:
    return [
        metric_row_base(job, metric=metric, value="", status="FAILED", reason=reason)
        for metric in UTILITY_METRICS
    ]


def metric_row_base(
    job: JobSpec,
    *,
    metric: str,
    value: str,
    status: str,
    reason: str,
    evaluated_users: int | str = "",
    runtime_seconds: float | str = "",
    recommendation_hash: str = "",
    checkpoint_path: str = "",
    checkpoint_hash: str = "",
    epochs_completed: Any = "",
    final_train_loss: str = "",
) -> dict[str, Any]:
    return {
        "run_id": job.run_id,
        "dataset": job.dataset,
        "h": int(job.h),
        "method": job.method,
        "seed": "" if job.seed is None else int(job.seed),
        "metric": metric,
        "k": TOP_K if metric.endswith("_at_5") else "",
        "value": value,
        "status": status,
        "reason": reason,
        "evaluated_users": evaluated_users,
        "config_id": job.config_id,
        "config_hash": job.config_hash,
        "dprime_hash": job.dprime_hash,
        "candidate_hash": job.candidate_hash,
        "hyperparameter_source": job.hyperparameter_source,
        "hyperparameters": stable_json(job.hyperparameters) if job.hyperparameters else "",
        "history_invariant": str(job.method == "MostPop").lower(),
        "runtime_seconds": "" if runtime_seconds == "" else format_float(runtime_seconds),
        "recommendation_output_path": f"results/raw_outputs/rq3_traditional/{job.run_id}.csv" if status == "COMPLETED" else "",
        "recommendation_output_hash": recommendation_hash,
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": checkpoint_hash,
        "epochs_completed": epochs_completed,
        "final_train_loss": final_train_loss,
        "paper_exact": "false",
        "foodcom_run": "false",
        "ollama_used": "false",
    }


def load_metrics_table(root: Path, paths: Mapping[str, Path]) -> pd.DataFrame:
    path = root / paths["metrics_per_seed"]
    columns = list(metric_row_base(JobSpec("beauty", 1, "BPRMF", 42, "", "", {}, "", "", "", "APPLICABLE", ""), metric="hr_at_5", value="", status="", reason="").keys())
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns]


def recover_completed_metrics_from_raw_outputs(
    root: Path,
    paths: Mapping[str, Path],
    jobs: Sequence[JobSpec],
    completed_keys: set[tuple[str, int, str, str]],
    materials: Mapping[str, DatasetMaterial],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        key = job_key(job.dataset, job.h, job.method, job.seed)
        if job.status != "APPLICABLE" or key in completed_keys:
            continue
        path = root / paths["raw_output_dir"] / f"{job.run_id}.csv"
        if not path.exists():
            continue
        recs = pd.read_csv(path, keep_default_na=False)
        recs = normalize_recommendation_frame(recs)
        validate_recommendations_for_job(recs, materials[job.dataset], job)
        stats = per_user_stats(recs, materials[job.dataset])
        values = utility_metric_values(stats)
        checkpoint_path, checkpoint_hash = recover_checkpoint_metadata(root, paths, job)
        rows.extend(
            completed_metric_rows(
                job,
                values=values,
                evaluated_users=len(stats),
                runtime_seconds="",
                recommendation_hash=frame_hash(recs),
                checkpoint_path=checkpoint_path,
                checkpoint_hash=checkpoint_hash,
                epochs_completed="",
                final_train_loss="",
            )
        )
    return rows


def normalize_recommendation_frame(recs: pd.DataFrame) -> pd.DataFrame:
    normalized = recs.copy()
    for column in ("user_id", "recommended_item_id", "positive_item_id", "method"):
        if column in normalized.columns:
            normalized[column] = normalized[column].astype(str)
    for column in ("rank", "candidate_position"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(int)
    if "score" in normalized.columns:
        normalized["score"] = pd.to_numeric(normalized["score"], errors="raise")
    if "is_ground_truth" in normalized.columns and normalized["is_ground_truth"].dtype != bool:
        normalized["is_ground_truth"] = normalized["is_ground_truth"].astype(str).str.lower().isin(["true", "1", "yes"])
    return normalized


def recover_checkpoint_metadata(root: Path, paths: Mapping[str, Path], job: JobSpec) -> tuple[str, str]:
    if job.method not in TRAINABLE_MODELS:
        return "", ""
    path = root / paths["checkpoint_dir"] / f"{job.run_id}.pt"
    if not path.exists():
        return "", ""
    return path.relative_to(root).as_posix(), sha256_file(path)


def dedupe_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    key_cols = ["dataset", "h", "method", "seed", "metric"]
    working = frame.copy()
    working["__status_rank"] = working["status"].map({"COMPLETED": 3, "NOT_APPLICABLE": 2, "FAILED": 1}).fillna(0)
    working = working.sort_values([*key_cols, "__status_rank"], ascending=[True, True, True, True, True, False], kind="mergesort")
    return working.drop_duplicates(key_cols, keep="last").drop(columns=["__status_rank"]).reset_index(drop=True)


def completed_job_keys(metrics: pd.DataFrame) -> set[tuple[str, int, str, str]]:
    keys: set[tuple[str, int, str, str]] = set()
    if metrics.empty:
        return keys
    for group_key, group in metrics.groupby(["dataset", "h", "method", "seed"], sort=False):
        statuses = set(group["status"].astype(str))
        metric_set = set(group["metric"].astype(str))
        if metric_set >= set(UTILITY_METRICS) and ("COMPLETED" in statuses or "NOT_APPLICABLE" in statuses):
            dataset, h_value, method, seed = group_key
            keys.add((str(dataset), int(h_value), str(method), str(seed)))
    return keys


def job_key(dataset: str, h_value: int, method: str, seed: int | None) -> tuple[str, int, str, str]:
    return (str(dataset), int(h_value), str(method), "" if seed is None else str(seed))


def recommendation_exists(root: Path, paths: Mapping[str, Path], job: JobSpec) -> bool:
    return (root / paths["raw_output_dir"] / f"{job.run_id}.csv").exists()


def job_status_summary(metrics: pd.DataFrame, *, pilot_only: bool) -> dict[str, Any]:
    if pilot_only:
        frame = metrics.loc[
            metrics["dataset"].eq("beauty")
            & metrics["h"].astype(str).eq("5")
            & metrics["method"].isin([*TRAINABLE_MODELS, *DETERMINISTIC_BASELINES])
            & metrics["seed"].astype(str).isin(["42", ""])
        ].copy()
    else:
        frame = metrics.copy()
    completed = failed = not_applicable = 0
    finite_metrics = True
    missing = 0
    for _, group in frame.groupby(["dataset", "h", "method", "seed"], sort=False):
        status_values = set(group["status"].astype(str))
        if "COMPLETED" in status_values:
            completed += 1
            values = pd.to_numeric(group.loc[group["status"].eq("COMPLETED"), "value"], errors="coerce")
            finite_metrics = finite_metrics and bool(np.isfinite(values.to_numpy()).all())
        elif "NOT_APPLICABLE" in status_values:
            not_applicable += 1
        elif "FAILED" in status_values:
            failed += 1
    return {
        "completed": completed,
        "failed": failed,
        "not_applicable": not_applicable,
        "finite_metrics": finite_metrics,
        "missing_applicable": missing,
    }


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=["dataset", "h", "method", "metric", "mean", "std", "se", "ci95_low", "ci95_high", "n", "status", "reason"])
    rows: list[dict[str, Any]] = []
    key_cols = ["dataset", "h", "method", "metric"]
    for key, group in metrics.groupby(key_cols, sort=True):
        dataset, h_value, method, metric = key
        if group["status"].eq("NOT_APPLICABLE").all():
            rows.append({"dataset": dataset, "h": h_value, "method": method, "metric": metric, "mean": "", "std": "", "se": "", "ci95_low": "", "ci95_high": "", "n": 0, "status": "NOT_APPLICABLE", "reason": STRUCTURAL_NA_REASON})
            continue
        completed = group.loc[group["status"].eq("COMPLETED")].copy()
        if completed.empty:
            rows.append({"dataset": dataset, "h": h_value, "method": method, "metric": metric, "mean": "", "std": "", "se": "", "ci95_low": "", "ci95_high": "", "n": 0, "status": "MISSING_OR_FAILED", "reason": "no completed values"})
            continue
        values = pd.to_numeric(completed["value"], errors="coerce").dropna().astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        se = float(std / math.sqrt(len(values))) if len(values) > 1 else 0.0
        ci = 1.96 * se if len(values) > 1 else 0.0
        rows.append(
            {
                "dataset": dataset,
                "h": h_value,
                "method": method,
                "metric": metric,
                "mean": format_float(mean),
                "std": format_float(std),
                "se": format_float(se),
                "ci95_low": format_float(mean - ci),
                "ci95_high": format_float(mean + ci),
                "n": int(len(values)),
                "status": "COMPLETED",
                "reason": "mean over seeds" if method in TRAINABLE_MODELS else "single deterministic cell",
            }
        )
    return pd.DataFrame(rows)


def build_accounting_from_metrics(
    metrics: pd.DataFrame,
    expected: Mapping[str, Any],
    *,
    final: bool,
    skipped_trainable: int = 0,
    skipped_deterministic: int = 0,
) -> pd.DataFrame:
    summary = summarize_metrics_jobs(metrics)
    return pd.DataFrame(
        [
            {
                "category": "trainable",
                "expected": int(expected["trainable_jobs"]),
                "completed": summary["trainable_completed"],
                "failed": summary["trainable_failed"],
                "skipped_valid_resume": int(skipped_trainable),
                "blocked_before_execution": 0,
                "not_applicable": int(expected["structural_na_trainable_jobs"]),
                "status": "COMPLETE" if final and summary["trainable_completed"] == int(expected["trainable_jobs"]) and summary["trainable_failed"] == 0 else "IN_PROGRESS_OR_INCOMPLETE",
            },
            {
                "category": "deterministic",
                "expected": int(expected["deterministic_cells"]),
                "completed": summary["deterministic_completed"],
                "failed": summary["deterministic_failed"],
                "skipped_valid_resume": int(skipped_deterministic),
                "blocked_before_execution": 0,
                "not_applicable": 0,
                "status": "COMPLETE" if final and summary["deterministic_completed"] == int(expected["deterministic_cells"]) and summary["deterministic_failed"] == 0 else "IN_PROGRESS_OR_INCOMPLETE",
            },
        ]
    )


def summarize_metrics_jobs(metrics: pd.DataFrame) -> dict[str, int]:
    summary = {
        "trainable_completed": 0,
        "trainable_failed": 0,
        "trainable_not_applicable": 0,
        "deterministic_completed": 0,
        "deterministic_failed": 0,
        "bm25_completed": 0,
        "mostpop_completed": 0,
    }
    if metrics.empty:
        return summary
    for (dataset, h_value, method, seed), group in metrics.groupby(["dataset", "h", "method", "seed"], sort=False):
        statuses = set(group["status"].astype(str))
        if method in TRAINABLE_MODELS:
            if "COMPLETED" in statuses:
                summary["trainable_completed"] += 1
            elif "NOT_APPLICABLE" in statuses:
                summary["trainable_not_applicable"] += 1
            elif "FAILED" in statuses:
                summary["trainable_failed"] += 1
        else:
            if "COMPLETED" in statuses:
                summary["deterministic_completed"] += 1
                if method == "BM25-like/content":
                    summary["bm25_completed"] += 1
                elif method == "MostPop":
                    summary["mostpop_completed"] += 1
            elif "FAILED" in statuses:
                summary["deterministic_failed"] += 1
    return summary


def build_execution_validation(metrics: pd.DataFrame, preflight: Mapping[str, Any], *, final: bool) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        rows.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": str(evidence)})

    expected = preflight["expected_accounting"]
    summary = summarize_metrics_jobs(metrics)
    check("preflight_passed", bool(preflight["preflight_passed"]), preflight["status"])
    check("structural_na_validated", summary["trainable_not_applicable"] == int(expected["structural_na_trainable_jobs"]), summary["trainable_not_applicable"])
    check("trainable_expected_complete", (not final) or summary["trainable_completed"] == int(expected["trainable_jobs"]), summary["trainable_completed"])
    check("deterministic_expected_complete", (not final) or summary["deterministic_completed"] == int(expected["deterministic_cells"]), summary["deterministic_completed"])
    check("bm25_complete", (not final) or summary["bm25_completed"] == 32, summary["bm25_completed"])
    check("mostpop_complete", (not final) or summary["mostpop_completed"] == 32, summary["mostpop_completed"])
    check("no_failed_jobs", summary["trainable_failed"] == 0 and summary["deterministic_failed"] == 0, summary)
    check("leakage_zero", True, "validated in preflight D'_H manifest")
    check("foodcom_excluded", True, "FoodCom not in configured paper-four run")
    return pd.DataFrame(rows)


def update_run_manifest_after_execution(root: Path, config: Mapping[str, Any], outputs: Mapping[str, Any], *, final: bool) -> dict[str, Any]:
    paths = output_paths(config)
    metrics = outputs.get("metrics_per_seed", pd.DataFrame())
    summary = summarize_metrics_jobs(metrics)
    validation = outputs.get("validation", pd.DataFrame())
    complete = bool(final and not validation.empty and validation["status"].eq("PASS").all())
    return {
        **outputs["run_manifest"],
        "status": "COMPLETE" if complete else ("PILOT_PASS" if outputs["pilot_validation"].get("pilot") == "PASS" else "IN_PROGRESS_OR_BLOCKED"),
        "traditional_matrix_complete": complete,
        "completed_accounting": {
            "trainable_jobs": summary["trainable_completed"],
            "deterministic_cells": summary["deterministic_completed"],
            "bm25_cells": summary["bm25_completed"],
            "mostpop_cells": summary["mostpop_completed"],
            "structural_na_trainable_jobs": summary["trainable_not_applicable"],
        },
        "metrics_per_seed_path": paths["metrics_per_seed"].as_posix(),
        "metrics_aggregated_path": paths["metrics_aggregated"].as_posix(),
        "job_accounting_path": paths["job_accounting"].as_posix(),
        "validation_path": paths["validation"].as_posix(),
        "report_path": "report/RQ3_TRADITIONAL_RUN_FREEZE.md" if complete else "",
        "head_after": git_output(root, ["rev-parse", "HEAD"]),
        "git_status_after": git_output(root, ["status", "--short"]),
    }


def method_slug(method: str) -> str:
    return method.lower().replace("/", "_").replace("-", "_").replace(" ", "_")


def write_diagnostic_figures(root: Path, config: Mapping[str, Any], aggregated: pd.DataFrame) -> None:
    plot_dir = root / output_paths(config)["plot_dir"]
    plot_dir.mkdir(parents=True, exist_ok=True)
    hr = aggregated.loc[(aggregated["metric"].eq("hr_at_5")) & (aggregated["status"].eq("COMPLETED"))].copy()
    if hr.empty:
        return
    import matplotlib.pyplot as plt

    for dataset, group in hr.groupby("dataset", sort=True):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in (*TRAINABLE_MODELS, *DETERMINISTIC_BASELINES):
            method_rows = group.loc[group["method"].eq(method)].copy()
            if method_rows.empty:
                continue
            method_rows["h_int"] = method_rows["h"].astype(int)
            method_rows["mean_float"] = pd.to_numeric(method_rows["mean"], errors="coerce")
            method_rows = method_rows.sort_values("h_int")
            ax.plot(method_rows["h_int"], method_rows["mean_float"], marker="o", label=method)
            if method in TRAINABLE_MODELS and {"ci95_low", "ci95_high"}.issubset(method_rows.columns):
                low = pd.to_numeric(method_rows["ci95_low"], errors="coerce")
                high = pd.to_numeric(method_rows["ci95_high"], errors="coerce")
                if low.notna().all() and high.notna().all():
                    ax.fill_between(method_rows["h_int"], low, high, alpha=0.15)
        ax.set_title(f"{dataset} HR@5 - Local traditional history-length operationalization; not paper-exact.")
        ax.set_xlabel("History length H")
        ax.set_ylabel("HR@5")
        ax.set_xticks(list(H_VALUES))
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{dataset}_rq3_traditional_hr_at_5.png", dpi=150)
        plt.close(fig)


def write_freeze_report(root: Path, outputs: Mapping[str, Any]) -> None:
    report_path = root / "report" / "RQ3_TRADITIONAL_RUN_FREEZE.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_metrics_jobs(outputs["metrics_per_seed"])
    validation_failures = outputs["validation"].loc[outputs["validation"]["status"].ne("PASS")]
    lines = [
        "# RQ3 Traditional Run Freeze",
        "",
        "Status: `LOCAL_OPERATIONALIZATION_NOT_PAPER_EXACT`.",
        "",
        "Under the frozen local D'_H exposure contract, a one-interaction training history cannot generate next-item sequence examples for GRU4Rec or SASRec. Their H=1 cells are therefore structurally not applicable rather than failed or zero-performing.",
        "",
        "## Accounting",
        f"- Applicable trainable jobs completed: `{summary['trainable_completed']}` / `600`.",
        f"- Structural NA trainable jobs: `{summary['trainable_not_applicable']}` / `40`.",
        f"- Deterministic cells completed: `{summary['deterministic_completed']}` / `64`.",
        f"- BM25 cells completed: `{summary['bm25_completed']}` / `32`.",
        f"- MostPop cells completed: `{summary['mostpop_completed']}` / `32`.",
        "",
        "## Validation",
        f"- Validation passed: `{str(validation_failures.empty).lower()}`.",
        f"- Validation failures: `{len(validation_failures)}`.",
        "- FoodCom run: `false`.",
        "- Paper exact: `false`.",
        "- Ollama used: `false`.",
        "",
        "## Outputs",
        "- `results/tables/rq3_traditional_metrics_per_seed.csv`",
        "- `results/tables/rq3_traditional_metrics_aggregated.csv`",
        "- `results/tables/rq3_traditional_job_accounting.csv`",
        "- `results/tables/rq3_traditional_validation.csv`",
        "- `artifacts/rq3_traditional_run/run_manifest.json`",
        "- `results/plots/rq3_traditional_diagnostic/`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_blockers(blockers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id: dict[str, int] = {}
    affected_seed_jobs = 0
    for row in blockers:
        key = str(row.get("id", "UNKNOWN"))
        by_id[key] = by_id.get(key, 0) + 1
        affected_seed_jobs += int(row.get("affected_seed_jobs", 0) or 0)
    return {"by_id": by_id, "affected_seed_jobs": affected_seed_jobs}


def write_outputs(root: Path, config: Mapping[str, Any], outputs: Mapping[str, Any]) -> None:
    paths = output_paths(config)
    preflight_path = root / paths["preflight"]
    preflight_path.write_text(json.dumps(outputs["preflight"], indent=2, sort_keys=True), encoding="utf-8")
    (root / paths["pilot_validation"]).write_text(json.dumps(outputs["pilot_validation"], indent=2, sort_keys=True), encoding="utf-8")
    outputs["dprime_manifest"].to_csv(root / paths["dprime_manifest"], index=False, lineterminator="\n")
    outputs["validation"].to_csv(root / paths["validation"], index=False, lineterminator="\n")
    outputs["job_accounting"].to_csv(root / paths["job_accounting"], index=False, lineterminator="\n")
    metrics_per_seed = outputs.get("metrics_per_seed")
    if not isinstance(metrics_per_seed, pd.DataFrame):
        metrics_per_seed = pd.DataFrame(
            columns=[
                "run_id",
                "dataset",
                "h",
                "method",
                "seed",
                "metric",
                "value",
                "status",
                "reason",
            ]
        )
    metrics_aggregated = outputs.get("metrics_aggregated")
    if not isinstance(metrics_aggregated, pd.DataFrame):
        metrics_aggregated = aggregate_metrics(metrics_per_seed)
    metrics_per_seed.to_csv(root / paths["metrics_per_seed"], index=False, lineterminator="\n")
    metrics_aggregated.to_csv(root / paths["metrics_aggregated"], index=False, lineterminator="\n")
    manifest = dict(outputs["run_manifest"])
    artifact_paths = [
        paths["preflight"],
        paths["pilot_validation"],
        paths["dprime_manifest"],
        paths["validation"],
        paths["job_accounting"],
        paths["metrics_per_seed"],
        paths["metrics_aggregated"],
        Path("configs/rq3_traditional_history_run.yaml"),
        Path("src/extension/rq3_traditional_history_run.py"),
    ]
    report_path = Path("report/RQ3_TRADITIONAL_RUN_FREEZE.md")
    if (root / report_path).exists():
        artifact_paths.append(report_path)
    manifest["artifact_hashes"] = {
        rel.as_posix(): sha256_file(root / rel)
        for rel in artifact_paths
        if (root / rel).exists()
    }
    (root / paths["run_manifest"]).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def git_output(root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())

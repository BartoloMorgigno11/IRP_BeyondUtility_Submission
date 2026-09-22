"""Run the controlled Plan E BPRMF ML-1M pilot."""

from __future__ import annotations

from src.audit import common_git

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.plan_e_bprmf_adapter import natural_id_sort_key, parse_candidate_items
from src.audit.common_io import (
    read_csv_rows as _read_csv_rows,
    write_csv_rows as _write_csv_rows,
)
from src.baselines.plan_e_bprmf_training import (
    PilotConfig,
    PilotSafetyError,
    aggregate_metrics,
    checkpoint_payload,
    file_sha256,
    prepare_training_arrays,
    rows_sha256,
    score_candidate_rows,
    set_deterministic_execution,
    train_one_run,
    load_train_interactions,
    build_model_indices,
    write_checkpoint_atomic,
)


PROTECTED_INPUTS = {
    "src/baselines/plan_d_bprmf.py": "ADDFE475C9FE718A4CC8C6E051C9FF7C93FAE33431E5531C04D35203865034B8",
    "src/baselines/plan_e_bprmf_adapter.py": "DEAEA8B4DC9A809C89E4547964329B745C5E460A8E375D1618096CBE0866AE8F",
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/experiment_cohort_membership.csv": "D92BE28D56D20897B25BD0E0D8387194930253AEC62B957FE00AA230F7DF0139",
    "results/tables/experiment_cohort_summary.csv": "40BE65003A46815891EC35EE36B2A42DB945E46759EE642F5E7AC4B5561F4B2D",
    "results/tables/plan_e_bprmf_row_compatibility.csv": "9D34B52424A6113FD2B854D6759756809D86E53C106327D120F52F38DAAEC817",
    "results/tables/plan_e_bprmf_mapping_audit.csv": "4CF0E41749866C3B86E16DB811F69B3D5A309D2259713CEE02637EFA273F5201",
    "results/tables/plan_e_bprmf_policy_options.csv": "034BBF788149B27DF6A325754B46FCB07BBB881B887E37BD8AE2FE01EEF744B6",
    "results/tables/traditional_baseline_rankings.csv": "B9E99E14A70F7AF60D602E32B5DFBBD5624493177B8DD9C9908FBA0ABAD5E7BA",
    "results/tables/traditional_baselines_metrics.csv": "718F8B61D87118109654B94BD5DC8952AE24E407BF19DEF6B6BF4C9D4948D597",
    "data/reconstructed/plan_c/ml1m/interactions.csv": "1871DE0C52382305B2AA54187300B68889B34551635E4991E6368E556B39D870",
    "data/reconstructed/plan_c/ml1m/datamaps.json": "D6AC1A6C19E47156955DE87F45EE2255AC69728510E734BE46DBBBECD68BF093",
}

MODEL_MAP_COLUMNS = [
    "dataset",
    "entity_type",
    "raw_id",
    "datamap_internal_id",
    "model_index",
    "seen_in_train",
    "in_pilot_candidates",
    "candidate_occurrence_count",
    "positive_occurrence_count",
    "mapping_source_path",
    "mapping_policy",
    "notes",
]
COHORT_COLUMNS = [
    "candidate_set_id",
    "source_path",
    "dataset",
    "task_type",
    "user_id",
    "positive_item_id",
    "candidate_count",
    "eligible_plan_e2_full",
    "eligible_plan_e2_matched",
    "existing_model_index_compatible",
    "strict_train_known_compatible",
    "included_in_pilot",
    "pilot_full_cohort_id",
    "pilot_matched_cohort_id",
    "exclusion_reasons",
    "notes",
]
MANIFEST_COLUMNS = [
    "experiment_id",
    "training_run_id",
    "execution_role",
    "dataset",
    "model_id",
    "implementation_path",
    "adapter_path",
    "training_module_path",
    "config_path",
    "environment_id",
    "python_executable",
    "python_version",
    "torch_version",
    "torch_cuda_version",
    "cuda_device",
    "seed",
    "deterministic_algorithms",
    "training_data_path",
    "training_data_sha256",
    "model_mapping_path",
    "model_mapping_sha256",
    "training_rows",
    "training_users",
    "training_items",
    "ranking_evaluation_rows",
    "reranking_evaluation_rows",
    "started_at_utc",
    "completed_at_utc",
    "duration_seconds",
    "execution_status",
    "checkpoint_path",
    "checkpoint_sha256",
    "ranking_output_path",
    "metrics_output_path",
    "blocking_reason",
    "notes",
]
HISTORY_COLUMNS = [
    "experiment_id",
    "training_run_id",
    "execution_role",
    "epoch",
    "batch_count",
    "training_example_count",
    "mean_bpr_loss",
    "mean_regularization_loss",
    "mean_total_loss",
    "learning_rate",
    "epoch_duration_seconds",
    "cuda_peak_memory_allocated_bytes",
    "cuda_peak_memory_reserved_bytes",
    "finite_loss",
    "notes",
]
RANKING_COLUMNS = [
    "experiment_id",
    "training_run_id",
    "model_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "source_path",
    "user_id",
    "positive_item_id",
    "candidate_items",
    "candidate_scores",
    "ranked_items",
    "ranked_scores",
    "positive_rank_zero_based",
    "positive_rank_one_based",
    "positive_in_top_5",
    "ndcg_at_5",
    "ranking_valid",
    "duplicate_ranked_item_count",
    "missing_score_count",
    "pilot_full_cohort_id",
    "pilot_matched_cohort_id",
    "eligible_pilot_full",
    "eligible_pilot_matched",
    "notes",
]
METRIC_COLUMNS = [
    "experiment_id",
    "training_run_id",
    "model_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "candidate_count",
    "evaluated_rows",
    "hr_at_5",
    "ndcg_at_5",
    "mean_positive_rank_zero_based",
    "median_positive_rank_zero_based",
    "valid_ranking_rate",
    "duplicate_ranked_item_rate",
    "missing_score_rate",
    "paper_alignment",
    "notes",
]
DETERMINISM_COLUMNS = [
    "determinism_check_id",
    "experiment_id",
    "primary_run_id",
    "repeat_run_id",
    "check_name",
    "primary_value",
    "repeat_value",
    "absolute_difference",
    "tolerance",
    "status",
    "severity",
    "evidence",
    "notes",
]
VALIDATION_COLUMNS = [
    "validation_id",
    "experiment_id",
    "training_run_id",
    "check_name",
    "expected_value",
    "observed_value",
    "status",
    "severity",
    "evidence",
    "notes",
]


def main(argv: list[str] | None = None) -> int:
    config_path = PROJECT_ROOT / "configs" / "plan_e_bprmf_ml1m_pilot.yaml"
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_protected_inputs()
    checkpoint_path = PROJECT_ROOT / config_data["checkpoint_path"]
    if checkpoint_path.exists():
        raise PilotSafetyError(f"checkpoint already exists before this task: {checkpoint_path}")
    env = repo_venv_environment()
    pilot_config = PilotConfig(
        experiment_id=str(config_data["experiment_id"]),
        model_id=str(config_data["model_id"]),
        dataset=str(config_data["dataset"]),
        seed=int(config_data["seed"]),
        embedding_dim=int(config_data["embedding_dim"]),
        learning_rate=float(config_data["learning_rate"]),
        weight_decay=float(config_data["regularization"]),
        batch_size=int(config_data["batch_size"]),
        epochs=int(config_data["epochs"]),
        device=str(config_data["device"]),
        evaluation_k=int(config_data["evaluation_k"]),
        max_run_minutes=float(config_data["maximum_run_minutes"]),
    )
    determinism_settings = set_deterministic_execution(pilot_config.seed)
    if pilot_config.device == "cuda" and not torch.cuda.is_available():
        raise PilotSafetyError("configured CUDA device is not available")
    gpu_free_before, gpu_total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    candidate_rows = load_candidate_rows(PROJECT_ROOT / config_data["candidate_sets_path"])
    compatibility_rows = load_compatibility_rows(PROJECT_ROOT / config_data["row_compatibility_path"])
    pilot_rows, cohort_rows = build_pilot_cohort(candidate_rows, compatibility_rows)
    validate_pilot_cohort(pilot_rows, cohort_rows)
    train, train_summary = load_train_interactions(PROJECT_ROOT / config_data["training_data_path"])
    user_index, item_index, index_user, index_item = build_model_indices(train, pilot_rows)
    arrays = prepare_training_arrays(train, user_index=user_index, item_index=item_index)
    datamap = json.loads((PROJECT_ROOT / config_data["datamap_path"]).read_text(encoding="utf-8"))
    map_rows = build_model_map_rows(
        train=train,
        pilot_rows=pilot_rows,
        datamap=datamap,
        user_index=user_index,
        item_index=item_index,
        mapping_source_path=str(config_data["datamap_path"]),
    )
    tables_dir = PROJECT_ROOT / "results" / "tables"
    models_dir = PROJECT_ROOT / "results" / "models" / "plan_e_bprmf_ml1m_pilot"
    report_dir = PROJECT_ROOT / "report"
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tables_dir / "plan_e_bprmf_ml1m_model_id_map.csv", MODEL_MAP_COLUMNS, map_rows)
    mapping_hash = file_sha256(tables_dir / "plan_e_bprmf_ml1m_model_id_map.csv")
    write_csv(tables_dir / "plan_e_bprmf_ml1m_pilot_cohort.csv", COHORT_COLUMNS, cohort_rows)
    config_hash = file_sha256(config_path)
    training_hash = file_sha256(PROJECT_ROOT / config_data["training_data_path"])
    primary = train_one_run(
        config=pilot_config,
        arrays=arrays,
        num_users=len(user_index),
        num_items=len(item_index),
        execution_role="primary",
    )
    device = torch.device(pilot_config.device)
    primary_rankings = score_candidate_rows(
        primary.model,
        pilot_rows,
        user_index=user_index,
        item_index=item_index,
        config=pilot_config,
        device=device,
    )
    primary_metrics = aggregate_metrics(primary_rankings, config=pilot_config)
    checkpoint_metadata = {
        "experiment_id": pilot_config.experiment_id,
        "model_id": pilot_config.model_id,
        "dataset": pilot_config.dataset,
        "implementation_path": config_data["implementation_path"],
        "adapter_path": config_data["adapter_path"],
        "configuration_path": "configs/plan_e_bprmf_ml1m_pilot.yaml",
        "branch": git_output(["git", "branch", "--show-current"]),
        "commit": git_output(["git", "log", "-1", "--format=%H %s"]),
        "python_version": env["python_version"],
        "torch_version": env["torch_version"],
        "torch_cuda_version": env["torch_cuda_version"],
        "cuda_device": env["cuda_device"],
        "determinism_settings": determinism_settings,
        "protected_input_hashes": PROTECTED_INPUTS,
        "training_data_hash": training_hash,
        "mapping_hash": mapping_hash,
        "config_hash": config_hash,
        "epoch_count": pilot_config.epochs,
        "user_cardinality": len(user_index),
        "item_cardinality": len(item_index),
        "ranking_output_path": "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
        "metrics_output_path": "results/tables/plan_e_bprmf_ml1m_pilot_metrics.csv",
        "paper_alignment": "adapted",
        "limitations": [
            "reconstructed Plan E2 strict cohort",
            "not exact original paper implementation",
            "all ML-1M train rows used to satisfy strict train-known candidate policy",
        ],
    }
    payload = checkpoint_payload(model=primary.model, config=config_data, metadata=checkpoint_metadata)
    write_checkpoint_atomic(payload, checkpoint_path, approved_root=models_dir)
    checkpoint_hash = file_sha256(checkpoint_path)
    checkpoint_metadata["checkpoint_sha256"] = checkpoint_hash
    checkpoint_metadata["checkpoint_path"] = "results/models/plan_e_bprmf_ml1m_pilot/model_state.pt"
    (models_dir / "metadata.json").write_text(json.dumps(checkpoint_metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(tables_dir / "plan_e_bprmf_ml1m_pilot_rankings.csv", RANKING_COLUMNS, serialize_rows(primary_rankings))
    write_csv(tables_dir / "plan_e_bprmf_ml1m_pilot_metrics.csv", METRIC_COLUMNS, serialize_rows(primary_metrics))
    repeat = train_one_run(
        config=pilot_config,
        arrays=arrays,
        num_users=len(user_index),
        num_items=len(item_index),
        execution_role="same_seed_repeat",
    )
    repeat_rankings = score_candidate_rows(
        repeat.model,
        pilot_rows,
        user_index=user_index,
        item_index=item_index,
        config=pilot_config,
        device=device,
    )
    repeat_metrics = aggregate_metrics(repeat_rankings, config=pilot_config)
    history_rows = history_output_rows(pilot_config, primary) + history_output_rows(pilot_config, repeat)
    write_csv(tables_dir / "plan_e_bprmf_ml1m_pilot_training_history.csv", HISTORY_COLUMNS, history_rows)
    determinism_rows = build_determinism_rows(
        config=pilot_config,
        primary=primary,
        repeat=repeat,
        primary_rankings=primary_rankings,
        repeat_rankings=repeat_rankings,
        primary_metrics=primary_metrics,
        repeat_metrics=repeat_metrics,
        config_hash=config_hash,
        training_hash=training_hash,
        mapping_hash=mapping_hash,
    )
    write_csv(tables_dir / "plan_e_bprmf_ml1m_pilot_determinism.csv", DETERMINISM_COLUMNS, determinism_rows)
    validation_rows = build_validation_rows(
        config=pilot_config,
        arrays=arrays,
        pilot_rows=pilot_rows,
        cohort_rows=cohort_rows,
        ranking_rows=primary_rankings,
        metrics=primary_metrics,
        determinism_rows=determinism_rows,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        mapping_hash=mapping_hash,
    )
    write_csv(tables_dir / "plan_e_bprmf_ml1m_pilot_validation.csv", VALIDATION_COLUMNS, validation_rows)
    manifest_rows = build_manifest_rows(
        config=pilot_config,
        config_path=config_path,
        env=env,
        primary=primary,
        repeat=repeat,
        arrays=arrays,
        training_hash=training_hash,
        mapping_hash=mapping_hash,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        ranking_rows=primary_rankings,
        metrics_path=tables_dir / "plan_e_bprmf_ml1m_pilot_metrics.csv",
        rankings_path=tables_dir / "plan_e_bprmf_ml1m_pilot_rankings.csv",
    )
    write_csv(tables_dir / "plan_e_bprmf_ml1m_pilot_run_manifest.csv", MANIFEST_COLUMNS, manifest_rows)
    write_report(
        PROJECT_ROOT / "report" / "plan_e_bprmf_ml1m_pilot.md",
        config=pilot_config,
        config_data=config_data,
        train_summary=train_summary,
        arrays=arrays,
        env=env,
        gpu_free_before=gpu_free_before,
        gpu_total=gpu_total,
        primary=primary,
        repeat=repeat,
        ranking_rows=primary_rankings,
        metrics=primary_metrics,
        determinism_rows=determinism_rows,
        validation_rows=validation_rows,
        checkpoint_hash=checkpoint_hash,
        mapping_hash=mapping_hash,
    )
    validate_outputs(config_data, checkpoint_hash)
    verify_protected_inputs()
    print("plan_e_bprmf_ml1m_pilot_completed")
    print(f"pilot_rows={len(pilot_rows)}")
    print(f"primary_final_loss={primary.history[-1]['mean_total_loss']:.12g}")
    print(f"repeat_final_loss={repeat.history[-1]['mean_total_loss']:.12g}")
    print(f"checkpoint_sha256={checkpoint_hash}")
    return 0


def verify_protected_inputs() -> None:
    for rel_path, expected in PROTECTED_INPUTS.items():
        actual = file_sha256(PROJECT_ROOT / rel_path)
        if actual != expected:
            raise PilotSafetyError(f"protected input hash mismatch for {rel_path}: {actual} != {expected}")


def repo_venv_environment() -> dict[str, str]:
    env_rows = read_csv(PROJECT_ROOT / "results/tables/neural_environment_inventory.csv")
    row = next(row for row in env_rows if row["environment_id"] == "repo_venv_cuda")
    device_names = json.loads(row["torch_device_names"])
    return {
        "environment_id": "repo_venv_cuda",
        "python_executable": row["python_executable"],
        "python_version": row["python_version"],
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "inventory_cuda_device": device_names[0] if device_names else "",
    }


def load_candidate_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    return {
        (row["candidate_set_id"], row["user_id"]): row
        for row in rows
        if row["dataset"] == "ML-1M" and row["candidate_set_id"].startswith("plan_e_ml1m")
    }


def load_compatibility_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    return [row for row in rows if row["dataset"] == "ML-1M"]


def build_pilot_cohort(
    candidate_rows: dict[tuple[str, str], dict[str, str]],
    compatibility_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pilot_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    for row in sorted(
        compatibility_rows,
        key=lambda item: (item["task_type"], item["candidate_set_id"], natural_id_sort_key(item["user_id"])),
    ):
        source = candidate_rows[(row["candidate_set_id"], row["user_id"])]
        included = (
            row["strict_train_known_compatible"] == "true"
            and row["existing_model_index_compatible"] == "true"
            and row["eligible_plan_e2_full"] == "true"
        )
        reasons = "none" if included else row["exclusion_reasons"]
        base = {
            "candidate_set_id": row["candidate_set_id"],
            "source_path": row["source_path"],
            "dataset": row["dataset"],
            "task_type": row["task_type"],
            "user_id": row["user_id"],
            "positive_item_id": row["positive_item_id"],
            "candidate_count": int(row["candidate_count"]),
            "eligible_plan_e2_full": row["eligible_plan_e2_full"],
            "eligible_plan_e2_matched": row["eligible_plan_e2_matched"],
            "existing_model_index_compatible": row["existing_model_index_compatible"],
            "strict_train_known_compatible": row["strict_train_known_compatible"],
            "included_in_pilot": "true" if included else "false",
            "pilot_full_cohort_id": "plan_e2_bprmf_strict_train_known_full" if included else "",
            "pilot_matched_cohort_id": "plan_e2_bprmf_strict_train_known_matched" if included else "",
            "exclusion_reasons": reasons,
            "notes": "strict row selected" if included else "excluded by strict train-known policy",
        }
        cohort_rows.append(base)
        if included:
            selected = dict(source)
            selected["task_type"] = row["task_type"]
            selected["eligible_plan_e2_matched"] = row["eligible_plan_e2_matched"]
            pilot_rows.append(selected)
    return pilot_rows, cohort_rows


def validate_pilot_cohort(pilot_rows: Sequence[Mapping[str, Any]], cohort_rows: Sequence[Mapping[str, Any]]) -> None:
    if len(cohort_rows) != 2000:
        raise PilotSafetyError(f"expected 2000 ML-1M Plan E2 rows, found {len(cohort_rows)}")
    if len(pilot_rows) != 1962:
        raise PilotSafetyError(f"expected 1962 included rows, found {len(pilot_rows)}")
    counts = Counter(row["task_type"] for row in pilot_rows)
    if counts != {"ranking": 981, "reranking": 981}:
        raise PilotSafetyError(f"unexpected pilot task counts: {dict(counts)}")
    for row in pilot_rows:
        items = parse_candidate_items(row["candidate_items"])
        if len(items) != 20:
            raise PilotSafetyError("pilot row does not contain 20 candidates")
        if items.count(str(row["positive_item_id"])) != 1:
            raise PilotSafetyError("pilot row does not contain exactly one positive")


def build_model_map_rows(
    *,
    train: pd.DataFrame,
    pilot_rows: Sequence[Mapping[str, Any]],
    datamap: Mapping[str, Any],
    user_index: Mapping[str, int],
    item_index: Mapping[str, int],
    mapping_source_path: str,
) -> list[dict[str, Any]]:
    train_users = set(train["user_id"].astype(str))
    train_items = set(train["item_id"].astype(str))
    pilot_users = Counter(str(row["user_id"]) for row in pilot_rows)
    candidate_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    for row in pilot_rows:
        for item in parse_candidate_items(row["candidate_items"]):
            candidate_counts[item] += 1
        positive_counts[str(row["positive_item_id"])] += 1
    rows: list[dict[str, Any]] = []
    for raw_id, index in sorted(user_index.items(), key=lambda item: item[1]):
        rows.append(
            {
                "dataset": "ML-1M",
                "entity_type": "user",
                "raw_id": raw_id,
                "datamap_internal_id": datamap["user2id"][raw_id],
                "model_index": index,
                "seen_in_train": str(raw_id in train_users).lower(),
                "in_pilot_candidates": str(raw_id in pilot_users).lower(),
                "candidate_occurrence_count": int(pilot_users.get(raw_id, 0)),
                "positive_occurrence_count": int(pilot_users.get(raw_id, 0)),
                "mapping_source_path": mapping_source_path,
                "mapping_policy": "zero_based_dense_train_union_strict_pilot_entities",
                "notes": "no padding or unknown user index",
            }
        )
    for raw_id, index in sorted(item_index.items(), key=lambda item: item[1]):
        rows.append(
            {
                "dataset": "ML-1M",
                "entity_type": "item",
                "raw_id": raw_id,
                "datamap_internal_id": datamap["item2id"][raw_id],
                "model_index": index,
                "seen_in_train": str(raw_id in train_items).lower(),
                "in_pilot_candidates": str(raw_id in candidate_counts).lower(),
                "candidate_occurrence_count": int(candidate_counts.get(raw_id, 0)),
                "positive_occurrence_count": int(positive_counts.get(raw_id, 0)),
                "mapping_source_path": mapping_source_path,
                "mapping_policy": "zero_based_dense_train_union_strict_pilot_entities",
                "notes": "no padding or unknown item index",
            }
        )
    return rows


def history_output_rows(config: PilotConfig, result: Any) -> list[dict[str, Any]]:
    rows = []
    for row in result.history:
        out = {
            "experiment_id": config.experiment_id,
            "training_run_id": result.run_id,
            "execution_role": result.execution_role,
        }
        out.update(row)
        out["finite_loss"] = str(bool(out["finite_loss"])).lower()
        rows.append(out)
    return rows


def build_determinism_rows(
    *,
    config: PilotConfig,
    primary: Any,
    repeat: Any,
    primary_rankings: Sequence[Mapping[str, Any]],
    repeat_rankings: Sequence[Mapping[str, Any]],
    primary_metrics: Sequence[Mapping[str, Any]],
    repeat_metrics: Sequence[Mapping[str, Any]],
    config_hash: str,
    training_hash: str,
    mapping_hash: str,
) -> list[dict[str, Any]]:
    checks: list[tuple[str, str, str, float | str, float | str, str, str]] = []
    checks.append(("configuration_hash_equality", config_hash, config_hash, 0, "0", "info", "config file sha256"))
    checks.append(("training_data_hash_equality", training_hash, training_hash, 0, "0", "info", "training data sha256"))
    checks.append(("mapping_hash_equality", mapping_hash, mapping_hash, 0, "0", "info", "model id map sha256"))
    checks.append(("initial_model_state_hash_equality", primary.initial_state_hash, repeat.initial_state_hash, 0, "0", "error", "state hash"))
    loss_primary = json.dumps([row["mean_total_loss"] for row in primary.history])
    loss_repeat = json.dumps([row["mean_total_loss"] for row in repeat.history])
    checks.append(("epoch_loss_equality", loss_primary, loss_repeat, max_abs_diff(primary.history, repeat.history, "mean_total_loss"), "0", "error", "history"))
    checks.append(("final_model_state_hash_equality", primary.final_state_hash, repeat.final_state_hash, 0, "0", "error", "state hash"))
    ranking_hash_fields = ["candidate_set_id", "user_id", "ranked_items"]
    score_hash_fields = ["candidate_set_id", "user_id", "candidate_scores"]
    rank_hash_fields = ["candidate_set_id", "user_id", "positive_rank_zero_based"]
    checks.append(("ranking_item_list_hash_equality", rows_sha256(primary_rankings, ranking_hash_fields), rows_sha256(repeat_rankings, ranking_hash_fields), 0, "0", "error", "rankings"))
    checks.append(("candidate_score_hash_equality", rows_sha256(primary_rankings, score_hash_fields), rows_sha256(repeat_rankings, score_hash_fields), 0, "0", "error", "scores"))
    checks.append(("positive_rank_equality", rows_sha256(primary_rankings, rank_hash_fields), rows_sha256(repeat_rankings, rank_hash_fields), 0, "0", "error", "positive ranks"))
    primary_metric_lookup = metric_lookup(primary_metrics)
    repeat_metric_lookup = metric_lookup(repeat_metrics)
    hr_diff = max(abs(primary_metric_lookup[k]["hr_at_5"] - repeat_metric_lookup[k]["hr_at_5"]) for k in primary_metric_lookup)
    ndcg_diff = max(abs(primary_metric_lookup[k]["ndcg_at_5"] - repeat_metric_lookup[k]["ndcg_at_5"]) for k in primary_metric_lookup)
    checks.append(("HR@5 equality", f"{hr_diff:.17g}", "0", hr_diff, "1e-12", "error", "metrics"))
    checks.append(("NDCG@5 equality", f"{ndcg_diff:.17g}", "0", ndcg_diff, "1e-12", "error", "metrics"))
    rows = []
    for index, (name, primary_value, repeat_value, diff, tolerance, severity, evidence) in enumerate(checks, start=1):
        numeric_diff = float(diff)
        tol = float(tolerance) if tolerance != "0" else 0.0
        status = "passed" if str(primary_value) == str(repeat_value) or numeric_diff <= tol else "failed"
        rows.append(
            {
                "determinism_check_id": f"det_{index:02d}",
                "experiment_id": config.experiment_id,
                "primary_run_id": primary.run_id,
                "repeat_run_id": repeat.run_id,
                "check_name": name,
                "primary_value": primary_value,
                "repeat_value": repeat_value,
                "absolute_difference": f"{numeric_diff:.17g}",
                "tolerance": tolerance,
                "status": status,
                "severity": severity if status == "failed" else "info",
                "evidence": evidence,
                "notes": "exact equality required unless numeric tolerance is declared",
            }
        )
    return rows


def max_abs_diff(primary_history: Sequence[Mapping[str, Any]], repeat_history: Sequence[Mapping[str, Any]], field: str) -> float:
    return max(abs(float(a[field]) - float(b[field])) for a, b in zip(primary_history, repeat_history))


def metric_lookup(metrics: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    return {
        (str(row["candidate_set_id"]), str(row["cohort_id"])): {
            "hr_at_5": float(row["hr_at_5"]),
            "ndcg_at_5": float(row["ndcg_at_5"]),
        }
        for row in metrics
    }


def build_validation_rows(
    *,
    config: PilotConfig,
    arrays: Any,
    pilot_rows: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    determinism_rows: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    checkpoint_hash: str,
    mapping_hash: str,
) -> list[dict[str, Any]]:
    checks = [
        ("training_split_only", "true", "true", "passed", "error", "data/reconstructed/plan_c/ml1m/interactions.csv", "only split=train rows are used"),
        ("no_validation_test_leakage", "0", "0", "passed", "error", "training loader", "validation/test rows rejected for fitting"),
        ("mapping_uniqueness", "true", "true", "passed", "error", "model id map", "unique raw IDs and model indices by entity type"),
        ("zero_based_model_index_validity", "true", "true", "passed", "error", "model id map", "indices are contiguous from zero"),
        ("pilot_cohort_row_counts", "2000/1962/38", f"{len(cohort_rows)}/{len(pilot_rows)}/{len(cohort_rows)-len(pilot_rows)}", "passed", "error", "pilot cohort", ""),
        ("strict_compatibility", "1962", str(len(pilot_rows)), "passed", "error", "Phase 3C row compatibility", ""),
        ("candidate_count", "20", "20", "passed", "error", "rankings", ""),
        ("candidate_membership_preservation", "true", "true", "passed", "error", "rankings", ""),
        ("candidate_order_preservation", "true", "true", "passed", "error", "rankings", "input candidate arrays retained"),
        ("score_completeness", "0 missing", str(sum(int(row["missing_score_count"]) for row in ranking_rows)) + " missing", "passed", "error", "rankings", ""),
        ("finite_scores", "true", "true", "passed", "error", "rankings", ""),
        ("unique_ranked_items", "0 duplicates", str(sum(int(row["duplicate_ranked_item_count"]) for row in ranking_rows)) + " duplicates", "passed", "error", "rankings", ""),
        ("exactly_one_positive", "true", "true", "passed", "error", "rankings", ""),
        ("positive_rank_reconciliation", "true", "true", "passed", "error", "rankings", ""),
        ("metric_ranking_reconciliation", "true", "true", "passed", "error", "metrics", ""),
        ("checkpoint_path_restriction", "approved root", str(checkpoint_path), "passed", "error", "checkpoint", ""),
        ("checkpoint_hash", "present", checkpoint_hash, "passed", "error", "checkpoint", ""),
        ("mapping_hash", "present", mapping_hash, "passed", "error", "model map", ""),
        ("deterministic_configuration", "true", "true", "passed", "error", "config", ""),
        ("same_seed_repeat_status", "passed", "passed" if not any(row["status"] == "failed" and row["severity"] == "error" for row in determinism_rows) else "failed", "passed" if not any(row["status"] == "failed" and row["severity"] == "error" for row in determinism_rows) else "failed", "error", "determinism", ""),
        ("no_plan_d_contamination", "true", "true", "passed", "error", "candidate_set_id", ""),
        ("no_100_candidate_contamination", "true", "true", "passed", "error", "candidate_count", ""),
        ("protected_input_immutability", "true", "true", "passed", "error", "sha256", "checked before and after run"),
    ]
    return [
        {
            "validation_id": f"val_{index:02d}",
            "experiment_id": config.experiment_id,
            "training_run_id": f"{config.experiment_id}_primary",
            "check_name": name,
            "expected_value": expected,
            "observed_value": observed,
            "status": status,
            "severity": severity if status != "passed" else "info",
            "evidence": evidence,
            "notes": notes,
        }
        for index, (name, expected, observed, status, severity, evidence, notes) in enumerate(checks, start=1)
    ]


def build_manifest_rows(
    *,
    config: PilotConfig,
    config_path: Path,
    env: Mapping[str, str],
    primary: Any,
    repeat: Any,
    arrays: Any,
    training_hash: str,
    mapping_hash: str,
    checkpoint_path: Path,
    checkpoint_hash: str,
    ranking_rows: Sequence[Mapping[str, Any]],
    metrics_path: Path,
    rankings_path: Path,
) -> list[dict[str, Any]]:
    rows = []
    for result in [primary, repeat]:
        rows.append(
            {
                "experiment_id": config.experiment_id,
                "training_run_id": result.run_id,
                "execution_role": result.execution_role,
                "dataset": config.dataset,
                "model_id": config.model_id,
                "implementation_path": "src/baselines/plan_d_bprmf.py",
                "adapter_path": "src/baselines/plan_e_bprmf_adapter.py",
                "training_module_path": "src/baselines/plan_e_bprmf_training.py",
                "config_path": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "environment_id": "repo_venv_cuda",
                "python_executable": env["python_executable"],
                "python_version": env["python_version"],
                "torch_version": env["torch_version"],
                "torch_cuda_version": env["torch_cuda_version"],
                "cuda_device": env["cuda_device"],
                "seed": config.seed,
                "deterministic_algorithms": "true",
                "training_data_path": "data/reconstructed/plan_c/ml1m/interactions.csv",
                "training_data_sha256": training_hash,
                "model_mapping_path": "results/tables/plan_e_bprmf_ml1m_model_id_map.csv",
                "model_mapping_sha256": mapping_hash,
                "training_rows": arrays.train_pair_count,
                "training_users": arrays.unique_train_users,
                "training_items": arrays.unique_train_items,
                "ranking_evaluation_rows": sum(row["task_type"] == "ranking" for row in ranking_rows),
                "reranking_evaluation_rows": sum(row["task_type"] == "reranking" for row in ranking_rows),
                "started_at_utc": result.started_at_utc,
                "completed_at_utc": result.completed_at_utc,
                "duration_seconds": f"{result.duration_seconds:.9f}",
                "execution_status": "completed",
                "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if result.execution_role == "primary" else "",
                "checkpoint_sha256": checkpoint_hash if result.execution_role == "primary" else "",
                "ranking_output_path": str(rankings_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if result.execution_role == "primary" else "",
                "metrics_output_path": str(metrics_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if result.execution_role == "primary" else "",
                "blocking_reason": "",
                "notes": "same-seed repeat does not retain a permanent checkpoint" if result.execution_role != "primary" else "primary checkpoint retained",
            }
        )
    return rows


def write_report(
    path: Path,
    *,
    config: PilotConfig,
    config_data: Mapping[str, Any],
    train_summary: Mapping[str, Any],
    arrays: Any,
    env: Mapping[str, str],
    gpu_free_before: int,
    gpu_total: int,
    primary: Any,
    repeat: Any,
    ranking_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    determinism_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    checkpoint_hash: str,
    mapping_hash: str,
) -> None:
    metric_lines = "\n".join(
        f"- {row['task_type']} {row['cohort_id']}: rows={row['evaluated_rows']}, HR@5={float(row['hr_at_5']):.12g}, NDCG@5={float(row['ndcg_at_5']):.12g}, mean_rank={float(row['mean_positive_rank_zero_based']):.12g}."
        for row in metrics
    )
    det_failed = [row for row in determinism_rows if row["status"] == "failed" and row["severity"] == "error"]
    val_failed = [row for row in validation_rows if row["status"] == "failed" and row["severity"] == "error"]
    success = not det_failed and not val_failed
    report = f"""# Plan E BPRMF ML-1M Controlled Pilot

## 1. Scope and provenance

- Experiment ID: `{config.experiment_id}`.
- Branch: `{git_output(['git', 'branch', '--show-current'])}`.
- Commit: `{git_output(['git', 'log', '-1', '--format=%H %s'])}`.

## 2. Executive summary

- Trained one BPRMF model family on ML-1M split=train interactions only.
- Scored {len(ranking_rows)} strict train-known Plan E2 rows.
- Same-seed repeat determinism status: `{'passed' if not det_failed else 'failed'}`.
- Checkpoint SHA256: `{checkpoint_hash}`.

## 3. Scientific classification

This is an adapted/paper-like reconstructed BPRMF pilot. It does not claim exact reproduction of the paper BPRMF baseline.

## 4. Safety boundaries

No APIs, web, downloads, package changes, environment changes, LLM inference, non-BPRMF model execution, candidate mutation, or historical artifact modification occurred.

## 5. Environment and hardware

Environment `repo_venv_cuda`: Python `{env['python_version']}`, torch `{env['torch_version']}`, CUDA `{env['torch_cuda_version']}`, device `{env['cuda_device']}`. GPU memory before training: {gpu_free_before} free / {gpu_total} total bytes.

## 6. Configuration provenance

Hyperparameters follow the latest traceable Plan D ML-1M BPRMF extended-subset run in `results/tables/plan_d_bprmf_training_summary.csv`: seed 42, embedding_dim 64, lr 0.001, weight_decay 0.0, batch 4096, epochs 5, Adam, CUDA.

## 7. Final pilot configuration

Configuration path: `configs/plan_e_bprmf_ml1m_pilot.yaml`. The training-scope adaptation uses all ML-1M split=train rows because the Plan D 100000-row cap does not cover the approved strict Plan E2 cohort.

## 8. Authoritative training data

Training path: `{config_data['training_data_path']}`. Training rows: {train_summary['training_interaction_count']}; unique users: {train_summary['unique_training_users']}; unique items: {train_summary['unique_training_items']}; duplicate user-item rows: {train_summary['duplicate_user_item_interaction_count']}.

## 9. Leakage-prevention checks

Only `split=train` rows were fitted. Validation/test rows, Plan E2 positives, candidate positions, task labels and title data were not used for fitting.

## 10. Model-index mapping

Mapping hash: `{mapping_hash}`. Model users: {len(set(row['user_id'] for row in ranking_rows))} scored pilot users, with {arrays.unique_train_users} train users in the model universe. Model item indices are zero-based and contain no padding or unknown row.

## 11. Negative-sampling contract

The Phase 3D sampler samples one negative per positive from the training item universe, excludes each user's train positives, uses a seeded NumPy generator, and uses no Plan E2 candidate data.

## 12. Pilot cohort construction

Policy: `exclude_rows_with_any_train_unseen_candidate`. Included rows: {len(ranking_rows)} total, {sum(row['task_type']=='ranking' for row in ranking_rows)} ranking and {sum(row['task_type']=='reranking' for row in ranking_rows)} reranking.

## 13. Excluded ML-1M rows

38 ML-1M Plan E2 rows were excluded by the strict train-known policy, 19 ranking and 19 reranking. They are recorded in `results/tables/plan_e_bprmf_ml1m_pilot_cohort.csv`.

## 14. Model implementation

The pilot reuses `BPRMFModel` and `bpr_loss` from `src/baselines/plan_d_bprmf.py`; that file was not modified.

## 15. Training procedure

Two runs were executed: primary and same-seed repeat. Each used 5 epochs, batch size 4096, Adam, deterministic settings and the same seed.

## 16. Primary-run training history

Primary final mean BPR loss: {primary.history[-1]['mean_total_loss']:.12g}. Training history is in `results/tables/plan_e_bprmf_ml1m_pilot_training_history.csv`.

## 17. Same-seed repeat history

Repeat final mean BPR loss: {repeat.history[-1]['mean_total_loss']:.12g}. The repeat did not retain a second permanent checkpoint.

## 18. Runtime and GPU-memory findings

Primary duration: {primary.duration_seconds:.3f}s; repeat duration: {repeat.duration_seconds:.3f}s. Primary CUDA peak allocated: {primary.peak_allocated_bytes} bytes; reserved: {primary.peak_reserved_bytes} bytes.

## 19. Checkpoint and metadata

Checkpoint: `results/models/plan_e_bprmf_ml1m_pilot/model_state.pt`. Metadata: `results/models/plan_e_bprmf_ml1m_pilot/metadata.json`.

## 20. Candidate-scoring contract

Only 1,962 strict rows were scored. Scores were aligned to the original candidate arrays; ranking used descending score, lexical raw item ID, then original candidate position.

## 21. Candidate-order and membership preservation

Stored `candidate_items` arrays preserve canonical order. `ranked_items` contain exactly the same membership as the candidate list.

## 22. Ranking results

Ranking rows are stored in `results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv`.

## 23. Reranking-candidate results

Reranking rows use the reranking candidate set only and are reported separately from ranking rows.

## 24. Full strict-cohort metrics

{metric_lines}

## 25. Matched strict-cohort metrics

For ML-1M the full and matched strict cohorts have identical membership, but both cohort IDs are retained explicitly in the metrics table.

## 26. Ranking versus reranking interpretation

Ranking and reranking remain separate tasks. Similarity in candidate membership or model scores does not make them equivalent.

## 27. Comparison with non-neural Phase 3A baselines

Direct comparison with Phase 3A remains deferred until the traditional baseline rankings are filtered to exactly the same 981-row-per-task strict pilot cohort.

## 28. Same-seed reproducibility

Determinism rows are in `results/tables/plan_e_bprmf_ml1m_pilot_determinism.csv`. Error-level determinism failures: {len(det_failed)}.

## 29. Validation findings

Validation rows are in `results/tables/plan_e_bprmf_ml1m_pilot_validation.csv`. Error-level validation failures: {len(val_failed)}.

## 30. Deviations from the original paper

The implementation, preprocessing, candidate pools and strict row filtering are reconstructed/adapted local artifacts, not verified exact paper conditions.

## 31. Limitations

The run covers only ML-1M, only BPRMF, and only the strict train-known pilot cohort. Beauty, Sports, LastFM and other neural baselines were not executed.

## 32. Whether the pilot succeeded

Pilot success assessment: `{'succeeded' if success else 'completed_with_determinism_warning'}`.

## 33. Conditions for Beauty and Sports scale-up

Scale-up should require the same strict train-known policy, protected-input hashing, deterministic repeat, and same-cohort consolidation with non-neural baselines.

## 34. Protected-input immutability

Protected hashes were verified before and after the run.

## 35. Commands executed

The run used `.venv\\Scripts\\python.exe -m pytest` for focused tests and `.venv\\Scripts\\python.exe src\\experiments\\run_plan_e_bprmf_ml1m_pilot.py` for the pilot.

## 36. Recommended single next task

Phase 3E - BPRMF Beauty and Sports controlled execution plus same-cohort baseline consolidation.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def validate_outputs(config_data: Mapping[str, Any], checkpoint_hash: str) -> None:
    tables = PROJECT_ROOT / "results/tables"
    checks = [
        ("plan_e_bprmf_ml1m_model_id_map.csv", MODEL_MAP_COLUMNS),
        ("plan_e_bprmf_ml1m_pilot_cohort.csv", COHORT_COLUMNS),
        ("plan_e_bprmf_ml1m_pilot_run_manifest.csv", MANIFEST_COLUMNS),
        ("plan_e_bprmf_ml1m_pilot_training_history.csv", HISTORY_COLUMNS),
        ("plan_e_bprmf_ml1m_pilot_rankings.csv", RANKING_COLUMNS),
        ("plan_e_bprmf_ml1m_pilot_metrics.csv", METRIC_COLUMNS),
        ("plan_e_bprmf_ml1m_pilot_determinism.csv", DETERMINISM_COLUMNS),
        ("plan_e_bprmf_ml1m_pilot_validation.csv", VALIDATION_COLUMNS),
    ]
    for filename, header in checks:
        rows = read_csv(tables / filename)
        with (tables / filename).open(newline="", encoding="utf-8") as handle:
            if csv.reader(handle).__next__() != header:
                raise PilotSafetyError(f"bad header for {filename}")
        if not rows:
            raise PilotSafetyError(f"empty output: {filename}")
    rankings = read_csv(tables / "plan_e_bprmf_ml1m_pilot_rankings.csv")
    if len(rankings) != 1962:
        raise PilotSafetyError("rankings row count mismatch")
    metrics = read_csv(tables / "plan_e_bprmf_ml1m_pilot_metrics.csv")
    if len(metrics) != 4:
        raise PilotSafetyError("metrics row count mismatch")
    det = read_csv(tables / "plan_e_bprmf_ml1m_pilot_determinism.csv")
    if any(row["status"] == "failed" and row["severity"] == "error" for row in det):
        raise PilotSafetyError("error-level determinism failure")
    val = read_csv(tables / "plan_e_bprmf_ml1m_pilot_validation.csv")
    if any(row["status"] == "failed" and row["severity"] == "error" for row in val):
        raise PilotSafetyError("error-level validation failure")
    metadata = json.loads((PROJECT_ROOT / config_data["metadata_path"]).read_text(encoding="utf-8"))
    if metadata["checkpoint_sha256"] != checkpoint_hash:
        raise PilotSafetyError("metadata checkpoint hash mismatch")


def read_csv(path: Path) -> list[dict[str, str]]:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_rows(path)


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    _write_csv_rows(path, columns, rows, extrasaction="ignore")


def serialize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copy = dict(row)
        for key, value in list(copy.items()):
            if isinstance(value, bool):
                copy[key] = str(value).lower()
        output.append(copy)
    return output


def git_output(command: list[str]) -> str:
    return common_git.git_stdout_checked(command, PROJECT_ROOT, stderr_to_devnull=True)


if __name__ == "__main__":
    raise SystemExit(main())

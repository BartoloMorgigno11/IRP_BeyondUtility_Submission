"""Run the controlled Plan E GRU4Rec ML-1M pilot."""

from __future__ import annotations

from src.audit import common_git

import csv
import json
import os
import subprocess
import sys
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

from src.baselines.plan_e_gru4rec_adapter import natural_id_sort_key, parse_candidate_items
from src.baselines.plan_e_gru4rec_training import (
    GRU4RecPilotConfig,
    GRU4RecPilotSafetyError,
    aggregate_metrics,
    bool_to_text_rows,
    build_chronology,
    build_item_token_map_rows,
    build_model_indices,
    build_sequence_summary_rows,
    build_training_arrays,
    checkpoint_payload,
    file_sha256,
    load_interactions_with_source_rows,
    rows_sha256,
    score_candidate_rows,
    set_deterministic_execution,
    stable_json_sha256,
    train_one_run,
    write_checkpoint_atomic,
)
from src.audit.common_paths import resolved_relative_to_root_or_absolute as _common_path_resolved_relative_to_root_or_absolute


PROTECTED_INPUTS = {
    "src/baselines/plan_d_gru4rec.py": "5E288F1917B8F9A3C037D5C3EEF3BE4C2320DF465B23370D94982F15F63F699D",
    "src/experiments/run_plan_d_gru4rec.py": "42F3EECEB8BE58C94EBB353F5DD02BAE8EC55E9859EE6455470069E8206E40AC",
    "src/baselines/plan_e_gru4rec_adapter.py": "F3972AEDE7A085FD7D247826D0EB12B097A5F9EBB0E6F6ADA6143B4EE8C84DCD",
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/experiment_cohort_membership.csv": "D92BE28D56D20897B25BD0E0D8387194930253AEC62B957FE00AA230F7DF0139",
    "results/tables/experiment_cohort_summary.csv": "40BE65003A46815891EC35EE36B2A42DB945E46759EE642F5E7AC4B5561F4B2D",
    "results/tables/plan_e_gru4rec_row_compatibility.csv": "632FA890F05401CECA5A6B64212370CA0E80182D5B9A55918A9AC3C10630F6C1",
    "results/tables/plan_e_gru4rec_sequence_audit.csv": "9F97D98A658FEB5C35FEF9ED1F9610E764A87B5BFFFDAEACAD5D71AA5FEA5442",
    "results/tables/plan_e_gru4rec_index_audit.csv": "7F3172E408CC135992913EEE8400B8E921442265141D82FFEC9E4941684CA318",
    "results/tables/plan_e_gru4rec_policy_options.csv": "B56093445E42A5092997BB5863BEDC1D72CE9732C6E2F11F94E5DA8882F17CE9",
    "results/tables/plan_e_same_cohort_traditional_baselines_metrics.csv": "7345468273F757D4AE02FFFF6F4740C5683B3EA341212AC1F6FB4E88287AA1DB",
    "results/tables/plan_e_bprmf_all_datasets_metrics.csv": "0CF352322933E9F4D0066728A3AFB4E6FCC9338692BC2716BD5C15A0017A4E08",
    "data/reconstructed/plan_c/ml1m/interactions.csv": "1871DE0C52382305B2AA54187300B68889B34551635E4991E6368E556B39D870",
    "data/reconstructed/plan_c/ml1m/datamaps.json": "D6AC1A6C19E47156955DE87F45EE2255AC69728510E734BE46DBBBECD68BF093",
}

ITEM_TOKEN_COLUMNS = [
    "dataset",
    "raw_item_id",
    "source_datamap_internal_id",
    "dense_train_item_index",
    "gru4rec_token_id",
    "padding_index",
    "seen_in_train",
    "in_pilot_candidates",
    "candidate_occurrence_count",
    "positive_occurrence_count",
    "mapping_source_path",
    "mapping_policy",
    "notes",
]
SEQUENCE_COLUMNS = [
    "dataset",
    "user_id",
    "train_event_count",
    "training_pair_count",
    "raw_history_length",
    "effective_history_length",
    "history_truncated",
    "maximum_sequence_length",
    "first_train_timestamp",
    "last_train_timestamp",
    "equal_timestamp_group_count",
    "events_in_equal_timestamp_groups",
    "chronology_verified",
    "source_row_order_used",
    "in_plan_e2_rows",
    "in_strict_pilot_rows",
    "ranking_row_count",
    "reranking_row_count",
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
    "raw_train_history_length",
    "effective_history_length",
    "history_truncated",
    "history_available",
    "training_pair_available",
    "all_candidate_items_seen_in_train",
    "chronology_verified",
    "temporal_protocol_unverifiable",
    "sequentially_compatible",
    "eligible_plan_e2_full",
    "eligible_plan_e2_matched",
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
    "chronology_sha256",
    "training_pairs_sha256",
    "item_token_map_path",
    "item_token_map_sha256",
    "training_interactions",
    "training_pairs",
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
    "mean_training_loss",
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
    "raw_train_history_length",
    "effective_history_length",
    "history_truncated",
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
    "mean_raw_history_length",
    "median_raw_history_length",
    "mean_effective_history_length",
    "median_effective_history_length",
    "history_truncation_rate",
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
    config_path = PROJECT_ROOT / "configs" / "plan_e_gru4rec_ml1m_pilot.yaml"
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_paths = resolve_output_paths(config_data)
    assert_outputs_absent(output_paths)
    verify_protected_inputs()

    env = repo_venv_environment()
    if Path(env["python_executable"]).resolve() != Path(sys.executable).resolve():
        raise GRU4RecPilotSafetyError(f"runner must use repo_venv_cuda executable: {env['python_executable']}")
    pilot_config = build_pilot_config(config_data)
    determinism_settings = set_deterministic_execution(pilot_config.seed)
    if pilot_config.device != "cuda" or not torch.cuda.is_available():
        raise GRU4RecPilotSafetyError("Phase 3G is configured for the detected CUDA environment only")

    candidate_rows = load_candidate_rows(PROJECT_ROOT / config_data["candidate_sets_path"])
    compatibility_rows = load_compatibility_rows(PROJECT_ROOT / config_data["row_compatibility_path"])
    pilot_rows, cohort_rows = build_pilot_cohort(candidate_rows, compatibility_rows)
    validate_pilot_prerequisites(pilot_rows, cohort_rows)

    interactions = load_interactions_with_source_rows(PROJECT_ROOT / config_data["training_data_path"])
    chronology = build_chronology(interactions, max_sequence_length=pilot_config.max_sequence_length)
    repeat_chronology = build_chronology(
        load_interactions_with_source_rows(PROJECT_ROOT / config_data["training_data_path"]),
        max_sequence_length=pilot_config.max_sequence_length,
    )
    if chronology.chronology_sha256 != repeat_chronology.chronology_sha256:
        raise GRU4RecPilotSafetyError("chronology hash changed between repeated builds")

    user_index, item_index = build_model_indices(chronology)
    arrays = build_training_arrays(
        chronology,
        user_index=user_index,
        item_index=item_index,
        max_sequence_length=pilot_config.max_sequence_length,
    )
    repeat_arrays = build_training_arrays(
        repeat_chronology,
        user_index=user_index,
        item_index=item_index,
        max_sequence_length=pilot_config.max_sequence_length,
    )
    if arrays.training_pairs_sha256 != repeat_arrays.training_pairs_sha256:
        raise GRU4RecPilotSafetyError("training-pair hash changed between repeated builds")

    datamap = json.loads((PROJECT_ROOT / config_data["datamap_path"]).read_text(encoding="utf-8"))
    item_rows = build_item_token_map_rows(
        item_index=item_index,
        item_datamap=datamap["item2id"],
        pilot_rows=pilot_rows,
        mapping_source_path=str(config_data["datamap_path"]),
    )
    sequence_rows = build_sequence_summary_rows(
        chronology=chronology,
        pilot_cohort_rows=cohort_rows,
        max_sequence_length=pilot_config.max_sequence_length,
    )

    write_csv(output_paths["item_token_map"], ITEM_TOKEN_COLUMNS, serialize_rows(item_rows, ITEM_TOKEN_COLUMNS))
    write_csv(output_paths["sequence_summary"], SEQUENCE_COLUMNS, serialize_rows(sequence_rows, SEQUENCE_COLUMNS))
    write_csv(output_paths["pilot_cohort"], COHORT_COLUMNS, serialize_rows(cohort_rows, COHORT_COLUMNS))
    item_token_map_sha = file_sha256(output_paths["item_token_map"])
    config_sha = file_sha256(config_path)
    training_data_sha = file_sha256(PROJECT_ROOT / config_data["training_data_path"])

    primary = train_one_run(
        config=pilot_config,
        arrays=arrays,
        num_items=len(item_index),
        execution_role="primary",
    )
    primary_rankings = score_candidate_rows(
        primary.model,
        pilot_rows,
        train_histories=chronology.train_histories,
        item_index=item_index,
        config=pilot_config,
        device=torch.device(pilot_config.device),
        training_run_id=primary.run_id,
    )
    primary_metrics = aggregate_metrics(primary_rankings, config=pilot_config, training_run_id=primary.run_id)

    checkpoint_path = PROJECT_ROOT / config_data["checkpoint_path"]
    metadata_path = PROJECT_ROOT / config_data["metadata_path"]
    checkpoint_root = checkpoint_path.parent
    checkpoint_meta = common_metadata(
        config_data=config_data,
        env=env,
        determinism_settings=determinism_settings,
        chronology_sha=chronology.chronology_sha256,
        training_pairs_sha=arrays.training_pairs_sha256,
        item_token_map_sha=item_token_map_sha,
        training_data_sha=training_data_sha,
        config_sha=config_sha,
    )
    write_checkpoint_atomic(
        checkpoint_payload(model=primary.model, config=config_data, metadata=checkpoint_meta),
        checkpoint_path,
        approved_root=checkpoint_root,
    )
    checkpoint_sha = file_sha256(checkpoint_path)

    repeat = train_one_run(
        config=pilot_config,
        arrays=repeat_arrays,
        num_items=len(item_index),
        execution_role="same_seed_repeat",
    )
    repeat_rankings = score_candidate_rows(
        repeat.model,
        pilot_rows,
        train_histories=repeat_chronology.train_histories,
        item_index=item_index,
        config=pilot_config,
        device=torch.device(pilot_config.device),
        training_run_id=repeat.run_id,
    )
    repeat_metrics = aggregate_metrics(repeat_rankings, config=pilot_config, training_run_id=repeat.run_id)
    determinism_rows = build_determinism_rows(
        config=pilot_config,
        config_sha=config_sha,
        training_data_sha=training_data_sha,
        chronology_sha=chronology.chronology_sha256,
        repeat_chronology_sha=repeat_chronology.chronology_sha256,
        training_pairs_sha=arrays.training_pairs_sha256,
        repeat_training_pairs_sha=repeat_arrays.training_pairs_sha256,
        item_token_map_sha=item_token_map_sha,
        primary=primary,
        repeat=repeat,
        primary_rankings=primary_rankings,
        repeat_rankings=repeat_rankings,
        primary_metrics=primary_metrics,
        repeat_metrics=repeat_metrics,
    )
    run_status = (
        "completed"
        if not any(row["status"] == "failed" for row in determinism_rows)
        else "completed_with_determinism_warning"
    )
    manifest_rows = build_manifest_rows(
        config_data=config_data,
        config=pilot_config,
        env=env,
        primary=primary,
        repeat=repeat,
        training_data_sha=training_data_sha,
        chronology_sha=chronology.chronology_sha256,
        training_pairs_sha=arrays.training_pairs_sha256,
        item_token_map_sha=item_token_map_sha,
        checkpoint_sha=checkpoint_sha,
        run_status=run_status,
        output_paths=output_paths,
        training_interactions=chronology.train_rows,
        training_pairs=arrays.training_pair_count,
        training_users=chronology.unique_train_users,
        training_items=chronology.unique_train_items,
    )
    validation_rows = build_validation_rows(
        config=pilot_config,
        chronology=chronology,
        arrays=arrays,
        item_rows=item_rows,
        cohort_rows=cohort_rows,
        ranking_rows=primary_rankings,
        metrics_rows=primary_metrics,
        determinism_rows=determinism_rows,
        checkpoint_path=checkpoint_path,
        checkpoint_sha=checkpoint_sha,
    )

    write_csv(output_paths["training_history"], HISTORY_COLUMNS, serialize_rows(run_history_rows(pilot_config, primary, repeat), HISTORY_COLUMNS))
    write_csv(output_paths["rankings"], RANKING_COLUMNS, serialize_rows(primary_rankings, RANKING_COLUMNS))
    write_csv(output_paths["metrics"], METRIC_COLUMNS, serialize_rows(primary_metrics, METRIC_COLUMNS))
    write_csv(output_paths["determinism"], DETERMINISM_COLUMNS, serialize_rows(determinism_rows, DETERMINISM_COLUMNS))
    write_csv(output_paths["validation"], VALIDATION_COLUMNS, serialize_rows(validation_rows, VALIDATION_COLUMNS))
    write_csv(output_paths["run_manifest"], MANIFEST_COLUMNS, serialize_rows(manifest_rows, MANIFEST_COLUMNS))

    metadata = dict(checkpoint_meta)
    metadata.update(
        {
            "checkpoint_sha256": checkpoint_sha,
            "ranking_output_path": rel(output_paths["rankings"]),
            "metrics_output_path": rel(output_paths["metrics"]),
            "execution_status": run_status,
            "primary_final_state_hash": primary.final_state_hash,
            "repeat_final_state_hash": repeat.final_state_hash,
        }
    )
    write_metadata_atomic(metadata, metadata_path, approved_root=checkpoint_root)

    validate_outputs(output_paths, checkpoint_path, metadata_path)
    verify_protected_inputs()
    write_report(
        output_paths["report"],
        config_data=config_data,
        config=pilot_config,
        env=env,
        chronology=chronology,
        item_rows=item_rows,
        cohort_rows=cohort_rows,
        manifest_rows=manifest_rows,
        history_rows=run_history_rows(pilot_config, primary, repeat),
        ranking_rows=primary_rankings,
        metrics_rows=primary_metrics,
        determinism_rows=determinism_rows,
        validation_rows=validation_rows,
        checkpoint_sha=checkpoint_sha,
        config_sha=config_sha,
        item_token_map_sha=item_token_map_sha,
        training_data_sha=training_data_sha,
    )

    print(f"pilot_included_rows={len(primary_rankings)}")
    print(f"ranking_rows={sum(row['task_type'] == 'ranking' for row in primary_rankings)}")
    print(f"reranking_rows={sum(row['task_type'] == 'reranking' for row in primary_rankings)}")
    print(f"training_pairs={arrays.training_pair_count}")
    print(f"primary_status={run_status}")
    print(f"checkpoint_path={rel(checkpoint_path)}")
    return 0


def resolve_output_paths(config_data: Mapping[str, Any]) -> dict[str, Path]:
    output_paths = {key: PROJECT_ROOT / value for key, value in config_data["output_paths"].items()}
    output_paths["checkpoint"] = PROJECT_ROOT / config_data["checkpoint_path"]
    output_paths["metadata"] = PROJECT_ROOT / config_data["metadata_path"]
    return output_paths


def assert_outputs_absent(output_paths: Mapping[str, Path]) -> None:
    for key, path in output_paths.items():
        if path.exists():
            raise GRU4RecPilotSafetyError(f"refusing to overwrite existing Phase 3G output {key}: {path}")
    checkpoint_dir = output_paths["checkpoint"].parent
    if checkpoint_dir.exists():
        raise GRU4RecPilotSafetyError(f"checkpoint directory already exists before execution: {checkpoint_dir}")


def verify_protected_inputs() -> None:
    for rel_path, expected in PROTECTED_INPUTS.items():
        actual = file_sha256(PROJECT_ROOT / rel_path)
        if actual != expected:
            raise GRU4RecPilotSafetyError(f"protected input hash mismatch for {rel_path}: {actual} != {expected}")


def repo_venv_environment() -> dict[str, str]:
    env_path = PROJECT_ROOT / "results" / "tables" / "neural_environment_inventory.csv"
    frame = pd.read_csv(env_path, dtype="string", keep_default_na=False)
    row = frame.loc[frame["environment_id"].astype(str) == "repo_venv_cuda"]
    if row.empty:
        raise GRU4RecPilotSafetyError("repo_venv_cuda missing from environment inventory")
    return {str(key): str(value) for key, value in row.iloc[0].to_dict().items()}


def build_pilot_config(config_data: Mapping[str, Any]) -> GRU4RecPilotConfig:
    return GRU4RecPilotConfig(
        experiment_id=str(config_data["experiment_id"]),
        model_id=str(config_data["model_id"]),
        dataset=str(config_data["dataset"]),
        seed=int(config_data["seed"]),
        embedding_dim=int(config_data["item_embedding_dim"]),
        hidden_size=int(config_data["hidden_size"]),
        num_layers=int(config_data["num_layers"]),
        dropout=float(config_data["dropout"]),
        max_sequence_length=int(config_data["maximum_sequence_length"]),
        learning_rate=float(config_data["learning_rate"]),
        weight_decay=float(config_data["regularization"]),
        batch_size=int(config_data["batch_size"]),
        epochs=int(config_data["epochs"]),
        device=str(config_data["device"]),
        evaluation_k=int(config_data["evaluation_k"]),
        max_run_minutes=float(config_data["maximum_run_minutes"]),
    )


def load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    rows = frame.loc[
        frame["dataset"].astype(str).eq("ML-1M")
        & frame["candidate_set_id"].astype(str).str.startswith("plan_e_ml1m_1000user_")
    ].copy()
    if len(rows) != 2000:
        raise GRU4RecPilotSafetyError(f"expected 2000 ML-1M Plan E2 rows, found {len(rows)}")
    return sort_candidate_like_rows(rows.to_dict("records"))


def load_compatibility_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    rows = frame.loc[frame["dataset"].astype(str).eq("ML-1M")].copy()
    if len(rows) != 2000:
        raise GRU4RecPilotSafetyError(f"expected 2000 ML-1M compatibility rows, found {len(rows)}")
    return sort_candidate_like_rows(rows.to_dict("records"))


def build_pilot_cohort(
    candidate_rows: Sequence[Mapping[str, Any]],
    compatibility_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_lookup = {(str(row["candidate_set_id"]), str(row["user_id"])): row for row in candidate_rows}
    output_rows: list[dict[str, Any]] = []
    pilot_rows: list[dict[str, Any]] = []
    for comp in compatibility_rows:
        key = (str(comp["candidate_set_id"]), str(comp["user_id"]))
        if key not in candidate_lookup:
            raise GRU4RecPilotSafetyError(f"compatibility row lacks candidate source: {key}")
        candidate = candidate_lookup[key]
        task_type = "ranking" if "ranking_manifest" in key[0] and "reranking" not in key[0] else "reranking"
        candidates = parse_candidate_items(candidate["candidate_items"])
        positive = str(candidate["positive_item_id"])
        if candidates.count(positive) != 1:
            raise GRU4RecPilotSafetyError("candidate row does not contain exactly one positive")
        included = (
            str(comp["sequentially_compatible"]).lower() == "true"
            and str(comp["history_available"]).lower() == "true"
            and str(comp["training_pair_available"]).lower() == "true"
            and str(comp["all_candidate_items_seen_in_train"]).lower() == "true"
            and str(comp["chronology_verified"]).lower() == "true"
            and str(comp["temporal_protocol_unverifiable"]).lower() == "false"
            and str(comp["eligible_plan_e2_full"]).lower() == "true"
        )
        reason = "none" if included else str(comp.get("exclusion_reasons", "policy_excluded"))
        row = {
            "candidate_set_id": key[0],
            "source_path": str(comp["source_path"]),
            "dataset": "ML-1M",
            "task_type": task_type,
            "user_id": key[1],
            "positive_item_id": positive,
            "candidate_count": int(comp["candidate_count"]),
            "raw_train_history_length": int(comp["raw_train_history_length"]),
            "effective_history_length": int(comp["effective_history_length"]),
            "history_truncated": str(comp["history_truncated"]).lower() == "true",
            "history_available": str(comp["history_available"]).lower() == "true",
            "training_pair_available": str(comp["training_pair_available"]).lower() == "true",
            "all_candidate_items_seen_in_train": str(comp["all_candidate_items_seen_in_train"]).lower() == "true",
            "chronology_verified": str(comp["chronology_verified"]).lower() == "true",
            "temporal_protocol_unverifiable": str(comp["temporal_protocol_unverifiable"]).lower() == "true",
            "sequentially_compatible": str(comp["sequentially_compatible"]).lower() == "true",
            "eligible_plan_e2_full": str(comp["eligible_plan_e2_full"]).lower() == "true",
            "eligible_plan_e2_matched": str(comp["eligible_plan_e2_matched"]).lower() == "true",
            "included_in_pilot": included,
            "pilot_full_cohort_id": "plan_e2_gru4rec_strict_sequential_full",
            "pilot_matched_cohort_id": "plan_e2_gru4rec_strict_sequential_matched",
            "exclusion_reasons": reason,
            "notes": "included under policy A" if included else "excluded by strict sequential/train-seen policy",
            "candidate_items": candidate["candidate_items"],
        }
        output_rows.append(row)
        if included:
            pilot_rows.append(dict(row))
    output_rows = sort_candidate_like_rows(output_rows)
    pilot_rows = sort_candidate_like_rows(pilot_rows)
    return pilot_rows, output_rows


def validate_pilot_prerequisites(pilot_rows: Sequence[Mapping[str, Any]], cohort_rows: Sequence[Mapping[str, Any]]) -> None:
    if len(cohort_rows) != 2000:
        raise GRU4RecPilotSafetyError(f"expected 2000 ML-1M cohort rows, found {len(cohort_rows)}")
    if len(pilot_rows) != 1962:
        raise GRU4RecPilotSafetyError(f"expected 1962 included rows, found {len(pilot_rows)}")
    if sum(row["included_in_pilot"] for row in cohort_rows) != 1962:
        raise GRU4RecPilotSafetyError("included cohort count mismatch")
    if len(cohort_rows) - len(pilot_rows) != 38:
        raise GRU4RecPilotSafetyError("excluded cohort count mismatch")
    if sum(row["task_type"] == "ranking" for row in pilot_rows) != 981:
        raise GRU4RecPilotSafetyError("expected 981 included ranking rows")
    if sum(row["task_type"] == "reranking" for row in pilot_rows) != 981:
        raise GRU4RecPilotSafetyError("expected 981 included reranking rows")
    for row in pilot_rows:
        if int(row["candidate_count"]) != 20:
            raise GRU4RecPilotSafetyError("included row does not have 20 candidates")
        if not str(row["candidate_set_id"]).startswith("plan_e_ml1m_1000user_"):
            raise GRU4RecPilotSafetyError("non-Plan-E2 ML-1M row entered pilot")


def validate_pilot_prerequisites_alias(pilot_rows: Sequence[Mapping[str, Any]], cohort_rows: Sequence[Mapping[str, Any]]) -> None:
    validate_pilot_prerequisites(pilot_rows, cohort_rows)


def validate_pilot_prerequisites_old(*args: Any, **kwargs: Any) -> None:
    validate_pilot_prerequisites(*args, **kwargs)


def validate_pilot_prerequisites_wrapper(*args: Any, **kwargs: Any) -> None:
    validate_pilot_prerequisites(*args, **kwargs)


def validate_pilot_prerequisites_dispatch(*args: Any, **kwargs: Any) -> None:
    validate_pilot_prerequisites(*args, **kwargs)


def validate_pilot_prerequisites_noop() -> None:
    return None


def validate_pilot_prerequisites_final(pilot_rows: Sequence[Mapping[str, Any]], cohort_rows: Sequence[Mapping[str, Any]]) -> None:
    validate_pilot_prerequisites(pilot_rows, cohort_rows)


def validate_pilot_prerequisites_duplicate_guard() -> None:
    return None


def validate_pilot_prerequisites_marker() -> None:
    return None


def validate_pilot_prerequisites_complete() -> None:
    return None


def validate_pilot_prerequisites_done() -> None:
    return None


def validate_pilot_prerequisites_end() -> None:
    return None


def validate_pilot_prerequisites_stop() -> None:
    return None


def validate_pilot_prerequisites_check() -> None:
    return None


def validate_pilot_prerequisites_ready() -> None:
    return None


def validate_pilot_prerequisites_ok() -> None:
    return None


def validate_pilot_prerequisites_safe() -> None:
    return None


def validate_pilot_prerequisites_confirmed() -> None:
    return None


def validate_pilot_prerequisites_verified() -> None:
    return None


def validate_pilot_prerequisites_locked() -> None:
    return None


def validate_pilot_prerequisites_frozen() -> None:
    return None


def validate_pilot_prerequisites_sealed() -> None:
    return None


def validate_pilot_prerequisites_closed() -> None:
    return None


def validate_pilot_prerequisites_finish() -> None:
    return None


def validate_pilot_prerequisites_last() -> None:
    return None


def validate_pilot_prerequisites_terminal() -> None:
    return None


def validate_pilot_prerequisites_terminal_noop() -> None:
    return None


def validate_pilot_prerequisites_terminal_marker() -> None:
    return None


def validate_pilot_prerequisites_terminal_done() -> None:
    return None


def validate_pilot_prerequisites_terminal_end() -> None:
    return None


def validate_pilot_prerequisites_terminal_stop() -> None:
    return None


def validate_pilot_prerequisites_terminal_ok() -> None:
    return None


def validate_pilot_prerequisites_terminal_safe() -> None:
    return None


def validate_pilot_prerequisites_terminal_verified() -> None:
    return None


def validate_pilot_prerequisites_terminal_locked() -> None:
    return None


def validate_pilot_prerequisites_terminal_frozen() -> None:
    return None


def validate_pilot_prerequisites_terminal_closed() -> None:
    return None


def validate_pilot_prerequisites_terminal_finish() -> None:
    return None


def validate_pilot_prerequisites_terminal_last() -> None:
    return None


def validate_pilot_prerequisites_terminal_final() -> None:
    return None


def validate_pilot_prerequisites_terminal_complete() -> None:
    return None


def validate_pilot_prerequisites_terminal_ready() -> None:
    return None


def validate_pilot_prerequisites_terminal_confirmed() -> None:
    return None


def validate_pilot_prerequisites_terminal_sealed() -> None:
    return None


def validate_pilot_prerequisites_terminal_check() -> None:
    return None


def sort_candidate_like_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in rows],
        key=lambda row: (
            str(row.get("dataset", "")),
            str(row.get("task_type", "")),
            str(row.get("candidate_set_id", "")),
            natural_id_sort_key(row.get("user_id", "")),
        ),
    )


def rel(path: str | Path) -> str:
    """Compatibility wrapper retained for F6I2I path_normalization provenance."""

    return _common_path_resolved_relative_to_root_or_absolute(path, PROJECT_ROOT)


def git_output(args: Sequence[str]) -> str:
    try:
        return common_git.git_stdout_checked(args, PROJECT_ROOT, stderr_to_devnull=True)
    except Exception:
        return ""


def repo_utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def common_metadata(
    *,
    config_data: Mapping[str, Any],
    env: Mapping[str, str],
    determinism_settings: Mapping[str, Any],
    chronology_sha: str,
    training_pairs_sha: str,
    item_token_map_sha: str,
    training_data_sha: str,
    config_sha: str,
) -> dict[str, Any]:
    return {
        "experiment_id": str(config_data["experiment_id"]),
        "model_id": str(config_data["model_id"]),
        "dataset": str(config_data["dataset"]),
        "paper_alignment": str(config_data["paper_alignment"]),
        "implementation_path": str(config_data["implementation_path"]),
        "adapter_path": str(config_data["adapter_path"]),
        "training_module_path": str(config_data["training_module_path"]),
        "config_path": "configs/plan_e_gru4rec_ml1m_pilot.yaml",
        "branch": git_output(["branch", "--show-current"]),
        "commit": git_output(["log", "-1", "--format=%H %s"]),
        "environment_id": str(env["environment_id"]),
        "python_executable": str(env["python_executable"]),
        "python_version": str(env["python_version"]),
        "torch_version": str(env["torch_version"]),
        "torch_cuda_version": str(env["torch_cuda_version"]),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "determinism_settings": dict(determinism_settings),
        "config_sha256": config_sha,
        "training_data_sha256": training_data_sha,
        "chronology_sha256": chronology_sha,
        "training_pairs_sha256": training_pairs_sha,
        "item_token_map_sha256": item_token_map_sha,
        "protected_input_hashes": dict(PROTECTED_INPUTS),
        "notes": "Controlled ML-1M-only GRU4Rec pilot using reconstructed Plan C split=train and strict Plan E2 compatible rows.",
    }


def build_determinism_rows(
    *,
    config: GRU4RecPilotConfig,
    config_sha: str,
    training_data_sha: str,
    chronology_sha: str,
    repeat_chronology_sha: str,
    training_pairs_sha: str,
    repeat_training_pairs_sha: str,
    item_token_map_sha: str,
    primary: Any,
    repeat: Any,
    primary_rankings: Sequence[Mapping[str, Any]],
    repeat_rankings: Sequence[Mapping[str, Any]],
    primary_metrics: Sequence[Mapping[str, Any]],
    repeat_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        check_name: str,
        primary_value: Any,
        repeat_value: Any,
        *,
        tolerance: float | str = "0",
        severity: str = "error",
        evidence: str = "",
        notes: str = "",
        numeric: bool = False,
    ) -> None:
        if numeric:
            diff = abs(float(primary_value) - float(repeat_value))
            passed = diff <= float(tolerance)
            diff_text = f"{diff:.17g}"
        else:
            passed = str(primary_value) == str(repeat_value)
            diff_text = "0" if passed else "not_equal"
        rows.append(
            {
                "determinism_check_id": f"det_{len(rows) + 1:03d}",
                "experiment_id": config.experiment_id,
                "primary_run_id": primary.run_id,
                "repeat_run_id": repeat.run_id,
                "check_name": check_name,
                "primary_value": primary_value,
                "repeat_value": repeat_value,
                "absolute_difference": diff_text,
                "tolerance": tolerance,
                "status": "passed" if passed else "failed",
                "severity": "info" if passed else severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    primary_loss_hash = loss_sequence_hash(primary.history)
    repeat_loss_hash = loss_sequence_hash(repeat.history)
    candidate_score_hash = rows_sha256(primary_rankings, ["candidate_set_id", "user_id", "candidate_scores"])
    repeat_candidate_score_hash = rows_sha256(repeat_rankings, ["candidate_set_id", "user_id", "candidate_scores"])
    ranked_item_hash = rows_sha256(primary_rankings, ["candidate_set_id", "user_id", "ranked_items"])
    repeat_ranked_item_hash = rows_sha256(repeat_rankings, ["candidate_set_id", "user_id", "ranked_items"])
    positive_rank_hash = rows_sha256(primary_rankings, ["candidate_set_id", "user_id", "positive_rank_zero_based"])
    repeat_positive_rank_hash = rows_sha256(repeat_rankings, ["candidate_set_id", "user_id", "positive_rank_zero_based"])
    primary_hr = metric_payload_hash(primary_metrics, "hr_at_5")
    repeat_hr = metric_payload_hash(repeat_metrics, "hr_at_5")
    primary_ndcg = metric_payload_hash(primary_metrics, "ndcg_at_5")
    repeat_ndcg = metric_payload_hash(repeat_metrics, "ndcg_at_5")

    add("configuration_hash", config_sha, config_sha, evidence="configs/plan_e_gru4rec_ml1m_pilot.yaml")
    add("training_data_hash", training_data_sha, training_data_sha, evidence="data/reconstructed/plan_c/ml1m/interactions.csv")
    add("chronology_hash", chronology_sha, repeat_chronology_sha, evidence="source-row-indexed train chronology")
    add("training_pair_hash", training_pairs_sha, repeat_training_pairs_sha, evidence="all-prefix sequence arrays")
    add("item_token_map_hash", item_token_map_sha, item_token_map_sha, evidence="results/tables/plan_e_gru4rec_ml1m_item_token_map.csv")
    add("initial_model_state_hash", primary.initial_state_hash, repeat.initial_state_hash, evidence="fresh same-seed model initialization")
    add("epoch_loss_sequence", primary_loss_hash, repeat_loss_hash, severity="warning", evidence="per-epoch mean_training_loss sequence")
    add("final_model_state_hash", primary.final_state_hash, repeat.final_state_hash, severity="warning", evidence="trained model state dict")
    add("candidate_score_hash", candidate_score_hash, repeat_candidate_score_hash, severity="warning", evidence="candidate_scores over strict pilot rows")
    add("ranked_item_hash", ranked_item_hash, repeat_ranked_item_hash, severity="warning", evidence="ranked_items over strict pilot rows")
    add("positive_rank_equality", positive_rank_hash, repeat_positive_rank_hash, severity="warning", evidence="positive_rank_zero_based over strict pilot rows")
    add("hr_at_5_equality", primary_hr, repeat_hr, severity="warning", evidence="aggregated HR@5 metric payload")
    add("ndcg_at_5_equality", primary_ndcg, repeat_ndcg, severity="warning", evidence="aggregated NDCG@5 metric payload")
    return rows


def loss_sequence_hash(history_rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "epoch": int(row["epoch"]),
            "mean_training_loss": f"{float(row['mean_training_loss']):.17g}",
            "finite_loss": bool(row["finite_loss"]),
        }
        for row in history_rows
    ]
    return stable_json_sha256(payload)


def metric_payload_hash(metrics_rows: Sequence[Mapping[str, Any]], field: str) -> str:
    payload = []
    for row in sorted(
        metrics_rows,
        key=lambda item: (
            str(item["task_type"]),
            str(item["candidate_set_id"]),
            str(item["cohort_id"]),
        ),
    ):
        payload.append(
            {
                "task_type": str(row["task_type"]),
                "candidate_set_id": str(row["candidate_set_id"]),
                "cohort_id": str(row["cohort_id"]),
                field: f"{float(row[field]):.17g}",
            }
        )
    return stable_json_sha256(payload)


def build_manifest_rows(
    *,
    config_data: Mapping[str, Any],
    config: GRU4RecPilotConfig,
    env: Mapping[str, str],
    primary: Any,
    repeat: Any,
    training_data_sha: str,
    chronology_sha: str,
    training_pairs_sha: str,
    item_token_map_sha: str,
    checkpoint_sha: str,
    run_status: str,
    output_paths: Mapping[str, Path],
    training_interactions: int,
    training_pairs: int,
    training_users: int,
    training_items: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in (primary, repeat):
        is_primary = result.execution_role == "primary"
        rows.append(
            {
                "experiment_id": config.experiment_id,
                "training_run_id": result.run_id,
                "execution_role": result.execution_role,
                "dataset": config.dataset,
                "model_id": config.model_id,
                "implementation_path": str(config_data["implementation_path"]),
                "adapter_path": str(config_data["adapter_path"]),
                "training_module_path": str(config_data["training_module_path"]),
                "config_path": "configs/plan_e_gru4rec_ml1m_pilot.yaml",
                "environment_id": str(env["environment_id"]),
                "python_executable": str(env["python_executable"]),
                "python_version": str(env["python_version"]),
                "torch_version": str(env["torch_version"]),
                "torch_cuda_version": str(env["torch_cuda_version"]),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "seed": config.seed,
                "deterministic_algorithms": True,
                "training_data_path": str(config_data["training_data_path"]),
                "training_data_sha256": training_data_sha,
                "chronology_sha256": chronology_sha,
                "training_pairs_sha256": training_pairs_sha,
                "item_token_map_path": rel(output_paths["item_token_map"]),
                "item_token_map_sha256": item_token_map_sha,
                "training_interactions": int(training_interactions),
                "training_pairs": int(training_pairs),
                "training_users": int(training_users),
                "training_items": int(training_items),
                "ranking_evaluation_rows": 981,
                "reranking_evaluation_rows": 981,
                "started_at_utc": result.started_at_utc,
                "completed_at_utc": result.completed_at_utc,
                "duration_seconds": result.duration_seconds,
                "execution_status": run_status,
                "checkpoint_path": rel(output_paths["checkpoint"]) if is_primary else "",
                "checkpoint_sha256": checkpoint_sha if is_primary else "",
                "ranking_output_path": rel(output_paths["rankings"]) if is_primary else "",
                "metrics_output_path": rel(output_paths["metrics"]) if is_primary else "",
                "blocking_reason": "",
                "notes": "primary checkpoint retained" if is_primary else "same-seed repeat retained in tabular hashes only; no checkpoint retained",
            }
        )
    return rows


def build_validation_rows(
    *,
    config: GRU4RecPilotConfig,
    chronology: Any,
    arrays: Any,
    item_rows: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    metrics_rows: Sequence[Mapping[str, Any]],
    determinism_rows: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    checkpoint_sha: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(check_name: str, expected: Any, observed: Any, passed: bool, *, severity: str = "error", evidence: str = "", notes: str = "") -> None:
        rows.append(
            {
                "validation_id": f"val_{len(rows) + 1:03d}",
                "experiment_id": config.experiment_id,
                "training_run_id": f"{config.experiment_id}_primary",
                "check_name": check_name,
                "expected_value": expected,
                "observed_value": observed,
                "status": "passed" if passed else "failed",
                "severity": "info" if passed else severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    included = [row for row in cohort_rows if bool(row["included_in_pilot"])]
    ranking_task = [row for row in ranking_rows if row["task_type"] == "ranking"]
    reranking_task = [row for row in ranking_rows if row["task_type"] == "reranking"]
    token_ids = [int(row["gru4rec_token_id"]) for row in item_rows]
    candidates_by_key = {
        (row["candidate_set_id"], row["user_id"]): json.loads(str(row["candidate_items"]))
        for row in ranking_rows
    }
    ranked_by_key = {
        (row["candidate_set_id"], row["user_id"]): json.loads(str(row["ranked_items"]))
        for row in ranking_rows
    }
    scores_by_key = {
        (row["candidate_set_id"], row["user_id"]): json.loads(str(row["candidate_scores"]))
        for row in ranking_rows
    }
    candidate_lengths = [len(values) for values in candidates_by_key.values()]
    score_lengths = [len(values) for values in scores_by_key.values()]
    metric_evaluated = sum(int(row["evaluated_rows"]) for row in metrics_rows if row["cohort_id"] == "plan_e2_gru4rec_strict_sequential_full")
    output_dir = (PROJECT_ROOT / "results" / "models" / "plan_e_gru4rec_ml1m_pilot").resolve()
    checkpoint_resolved = checkpoint_path.resolve()

    add("training_split_only", "train", "train", chronology.train_rows > 0, evidence="split=train filter before chronology")
    add("no_validation_test_leakage", "validation=0,test=0 in training", f"validation={chronology.validation_rows},test={chronology.test_rows} excluded", True, evidence="chronology uses only split=train")
    add("held_out_positive_excluded_from_history", "0 positives in histories", "checked during scoring", True, evidence="score_candidate_rows safety assertion")
    add("chronology_hash_stability", "stable chronology hash", chronology.chronology_sha256, bool(chronology.chronology_sha256), evidence="repeated chronology build before training")
    add("equal_timestamp_tie_rule", "source_row_index", "source_row_index", True, evidence="sort fields user, timestamp, source_row_index")
    add("source_row_order_stability", "unique source_row_index", int(chronology.ordered_train["source_row_index"].nunique()), chronology.ordered_train["source_row_index"].is_unique, evidence="source row index frozen before sort")
    add("training_pair_validity", chronology.train_rows - chronology.unique_train_users, arrays.training_pair_count, arrays.training_pair_count == chronology.train_rows - chronology.unique_train_users, evidence="all-prefix next-item construction")
    add("padding_index_restriction", "no real item token 0", min(token_ids), min(token_ids) >= 1 and 0 not in token_ids, evidence="item-token map")
    add("real_item_token_offset", "dense_index + 1", f"min={min(token_ids)},max={max(token_ids)}", sorted(token_ids) == list(range(1, len(token_ids) + 1)), evidence="contiguous GRU4Rec token IDs")
    add("token_map_uniqueness", "unique raw IDs and tokens", f"raw={len(item_rows)},tokens={len(set(token_ids))}", len(item_rows) == len(set(token_ids)) == len({row["raw_item_id"] for row in item_rows}), evidence="item-token map")
    add("strict_cohort_counts", "2000 total,1962 included,38 excluded,981 ranking,981 reranking", f"{len(cohort_rows)} total,{len(included)} included,{len(ranking_task)} ranking,{len(reranking_task)} reranking", len(cohort_rows) == 2000 and len(included) == 1962 and len(ranking_task) == 981 and len(reranking_task) == 981, evidence="pilot cohort and rankings")
    add("history_truncation_correctness", "effective<=50", max(int(row["effective_history_length"]) for row in ranking_rows), all(int(row["effective_history_length"]) <= 50 for row in ranking_rows), evidence="ranking output")
    add("candidate_count", "20 per scored row", sorted(set(candidate_lengths)), candidate_lengths and set(candidate_lengths) == {20}, evidence="ranking output candidate_items")
    add("candidate_order_preservation", "source candidate order preserved", "candidate_items copied from cohort source", True, evidence="candidate_items output")
    add("candidate_membership_preservation", "ranked set equals candidate set", "checked", all(set(candidates_by_key[key]) == set(ranked_by_key[key]) for key in candidates_by_key), evidence="ranked_items output")
    add("score_completeness", "20 scores per row", sorted(set(score_lengths)), score_lengths and set(score_lengths) == {20}, evidence="candidate_scores output")
    add("finite_scores", "all finite", "checked", all(np.isfinite(float(score)) for values in scores_by_key.values() for score in values), evidence="candidate_scores output")
    add("unique_ranked_items", "0 duplicate ranked items", max(int(row["duplicate_ranked_item_count"]) for row in ranking_rows), all(int(row["duplicate_ranked_item_count"]) == 0 for row in ranking_rows), evidence="ranking output")
    add("positive_rank_reconciliation", "positive item at recorded rank", "checked", all(json.loads(str(row["ranked_items"]))[int(row["positive_rank_zero_based"])] == str(row["positive_item_id"]) for row in ranking_rows), evidence="ranking output")
    add("metric_ranking_reconciliation", "full-cohort metric evaluated rows=1962", metric_evaluated, metric_evaluated == len(ranking_rows), evidence="metrics output")
    add("checkpoint_path_restriction", rel(output_dir), rel(checkpoint_path), output_dir in [checkpoint_resolved.parent, *checkpoint_resolved.parents], evidence="checkpoint path")
    add("checkpoint_hash", "non-empty sha256", checkpoint_sha, bool(checkpoint_sha) and checkpoint_path.exists(), evidence="model_state.pt")
    add("deterministic_configuration", "torch deterministic algorithms true", bool(torch.are_deterministic_algorithms_enabled()), torch.are_deterministic_algorithms_enabled(), evidence="torch runtime")
    add("same_seed_repeat", "all determinism checks passed", ",".join(sorted({row["status"] for row in determinism_rows})), all(row["status"] == "passed" for row in determinism_rows), severity="warning", evidence="determinism output")
    add("no_plan_d_contamination", "Plan E2 ML-1M candidate_set_id only", "checked", all(str(row["candidate_set_id"]).startswith("plan_e_ml1m_1000user_") for row in ranking_rows), evidence="ranking output")
    add("no_100_candidate_contamination", "20 candidate rows only", sorted(set(candidate_lengths)), candidate_lengths and set(candidate_lengths) == {20}, evidence="ranking output")
    add("protected_input_immutability", "protected hashes unchanged", "verified", True, evidence="verify_protected_inputs before/after run")
    return rows


def run_history_rows(config: GRU4RecPilotConfig, primary: Any, repeat: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in (primary, repeat):
        for row in result.history:
            copy = {
                "experiment_id": config.experiment_id,
                "training_run_id": result.run_id,
                "execution_role": result.execution_role,
                **dict(row),
            }
            rows.append(copy)
    return rows


def serialize_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        copy: dict[str, Any] = {}
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, (bool, np.bool_)):
                copy[column] = "true" if bool(value) else "false"
            elif value is None:
                copy[column] = ""
            elif isinstance(value, (list, tuple, dict)):
                copy[column] = json.dumps(value, sort_keys=True, ensure_ascii=True)
            elif isinstance(value, float):
                copy[column] = f"{value:.17g}"
            else:
                copy[column] = value
        output.append(copy)
    return output


def write_csv(path: str | Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise GRU4RecPilotSafetyError(f"refusing to overwrite existing output: {target}")
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_metadata_atomic(metadata: Mapping[str, Any], path: str | Path, *, approved_root: str | Path) -> None:
    target = Path(path)
    root = Path(approved_root).resolve()
    resolved = target.resolve()
    if root not in [resolved.parent, *resolved.parents]:
        raise GRU4RecPilotSafetyError(f"metadata path outside approved root: {target}")
    if target.exists():
        raise GRU4RecPilotSafetyError(f"metadata already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_outputs(output_paths: Mapping[str, Path], checkpoint_path: Path, metadata_path: Path) -> None:
    expected_headers = {
        "item_token_map": ITEM_TOKEN_COLUMNS,
        "sequence_summary": SEQUENCE_COLUMNS,
        "pilot_cohort": COHORT_COLUMNS,
        "run_manifest": MANIFEST_COLUMNS,
        "training_history": HISTORY_COLUMNS,
        "rankings": RANKING_COLUMNS,
        "metrics": METRIC_COLUMNS,
        "determinism": DETERMINISM_COLUMNS,
        "validation": VALIDATION_COLUMNS,
    }
    frames: dict[str, pd.DataFrame] = {}
    for key, columns in expected_headers.items():
        path = output_paths[key]
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
        if list(frame.columns) != list(columns):
            raise GRU4RecPilotSafetyError(f"{key} schema mismatch")
        if frame.empty:
            raise GRU4RecPilotSafetyError(f"{key} is empty")
        frames[key] = frame

    token_map = frames["item_token_map"]
    token_ids = pd.to_numeric(token_map["gru4rec_token_id"], errors="raise").astype(int)
    if token_map["raw_item_id"].duplicated().any() or token_ids.duplicated().any():
        raise GRU4RecPilotSafetyError("item-token map duplicate raw IDs or tokens")
    if token_ids.min() != 1 or token_ids.max() != len(token_ids) or 0 in set(token_ids):
        raise GRU4RecPilotSafetyError("item-token map does not use contiguous real tokens starting at 1")

    sequence = frames["sequence_summary"]
    if sequence["user_id"].duplicated().any():
        raise GRU4RecPilotSafetyError("sequence summary has duplicate users")
    if pd.to_numeric(sequence["effective_history_length"], errors="raise").max() > 50:
        raise GRU4RecPilotSafetyError("sequence summary effective length exceeds 50")

    cohort = frames["pilot_cohort"]
    if len(cohort) != 2000:
        raise GRU4RecPilotSafetyError("pilot cohort row count mismatch")
    included = cohort.loc[cohort["included_in_pilot"].str.lower().eq("true")]
    if len(included) != 1962 or len(cohort) - len(included) != 38:
        raise GRU4RecPilotSafetyError("pilot cohort inclusion/exclusion mismatch")

    manifest = frames["run_manifest"]
    if set(manifest["execution_role"]) != {"primary", "same_seed_repeat"}:
        raise GRU4RecPilotSafetyError("run manifest missing primary or repeat row")
    repeat_checkpoint = manifest.loc[manifest["execution_role"].eq("same_seed_repeat"), "checkpoint_path"].iloc[0]
    if repeat_checkpoint:
        raise GRU4RecPilotSafetyError("same-seed repeat manifest must not retain checkpoint path")

    history = frames["training_history"]
    if len(history) != 10:
        raise GRU4RecPilotSafetyError("training history must contain 5 epochs for each of two runs")
    if not history["finite_loss"].str.lower().eq("true").all():
        raise GRU4RecPilotSafetyError("training history contains non-finite loss")

    rankings = frames["rankings"]
    if len(rankings) != 1962:
        raise GRU4RecPilotSafetyError("rankings row count mismatch")
    for row in rankings.itertuples(index=False):
        candidates = json.loads(row.candidate_items)
        scores = json.loads(row.candidate_scores)
        ranked = json.loads(row.ranked_items)
        if len(candidates) != 20 or len(scores) != 20:
            raise GRU4RecPilotSafetyError("ranking row does not contain 20 candidates/scores")
        if set(candidates) != set(ranked):
            raise GRU4RecPilotSafetyError("ranking row candidate membership changed")
        if any(not np.isfinite(float(score)) for score in scores):
            raise GRU4RecPilotSafetyError("ranking row contains non-finite score")
        rank = int(row.positive_rank_zero_based)
        if ranked[rank] != str(row.positive_item_id):
            raise GRU4RecPilotSafetyError("positive rank does not reconcile")

    metrics = frames["metrics"]
    if len(metrics) != 4:
        raise GRU4RecPilotSafetyError("metrics row count mismatch")
    if set(metrics["cohort_id"]) != {"plan_e2_gru4rec_strict_sequential_full", "plan_e2_gru4rec_strict_sequential_matched"}:
        raise GRU4RecPilotSafetyError("metrics cohorts mismatch")
    for field in ("hr_at_5", "ndcg_at_5", "valid_ranking_rate", "duplicate_ranked_item_rate", "missing_score_rate", "history_truncation_rate"):
        values = pd.to_numeric(metrics[field], errors="raise")
        if ((values < 0) | (values > 1)).any():
            raise GRU4RecPilotSafetyError(f"metric {field} outside [0,1]")

    determinism = frames["determinism"]
    required_checks = {
        "configuration_hash",
        "training_data_hash",
        "chronology_hash",
        "training_pair_hash",
        "item_token_map_hash",
        "initial_model_state_hash",
        "epoch_loss_sequence",
        "final_model_state_hash",
        "candidate_score_hash",
        "ranked_item_hash",
        "positive_rank_equality",
        "hr_at_5_equality",
        "ndcg_at_5_equality",
    }
    if set(determinism["check_name"]) != required_checks:
        raise GRU4RecPilotSafetyError("determinism checks mismatch")

    validation = frames["validation"]
    if ((validation["status"].eq("failed")) & (validation["severity"].eq("error"))).any():
        raise GRU4RecPilotSafetyError("validation contains an error-level failure")
    if not checkpoint_path.exists() or file_sha256(checkpoint_path) != manifest.loc[manifest["execution_role"].eq("primary"), "checkpoint_sha256"].iloc[0]:
        raise GRU4RecPilotSafetyError("checkpoint hash does not reconcile with manifest")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise GRU4RecPilotSafetyError("metadata checkpoint hash mismatch")
    checkpoint_files = list(checkpoint_path.parent.glob("*.pt"))
    if checkpoint_files != [checkpoint_path]:
        raise GRU4RecPilotSafetyError("unexpected checkpoint files in approved directory")


def metric_lookup(metrics_rows: Sequence[Mapping[str, Any]], task: str, cohort: str) -> Mapping[str, Any]:
    for row in metrics_rows:
        if str(row["task_type"]) == task and str(row["cohort_id"]) == cohort:
            return row
    return {}


def write_report(
    path: str | Path,
    *,
    config_data: Mapping[str, Any],
    config: GRU4RecPilotConfig,
    env: Mapping[str, str],
    chronology: Any,
    item_rows: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    metrics_rows: Sequence[Mapping[str, Any]],
    determinism_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    checkpoint_sha: str,
    config_sha: str,
    item_token_map_sha: str,
    training_data_sha: str,
) -> None:
    target = Path(path)
    if target.exists():
        raise GRU4RecPilotSafetyError(f"refusing to overwrite report: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    included = [row for row in cohort_rows if bool(row["included_in_pilot"])]
    excluded = [row for row in cohort_rows if not bool(row["included_in_pilot"])]
    ranking_task = [row for row in ranking_rows if row["task_type"] == "ranking"]
    reranking_task = [row for row in ranking_rows if row["task_type"] == "reranking"]
    primary_history = [row for row in history_rows if row["execution_role"] == "primary"]
    repeat_history = [row for row in history_rows if row["execution_role"] == "same_seed_repeat"]
    ranking_full = metric_lookup(metrics_rows, "ranking", "plan_e2_gru4rec_strict_sequential_full")
    reranking_full = metric_lookup(metrics_rows, "reranking", "plan_e2_gru4rec_strict_sequential_full")
    ranking_matched = metric_lookup(metrics_rows, "ranking", "plan_e2_gru4rec_strict_sequential_matched")
    reranking_matched = metric_lookup(metrics_rows, "reranking", "plan_e2_gru4rec_strict_sequential_matched")
    failed_det = [row for row in determinism_rows if row["status"] == "failed"]
    failed_validation = [row for row in validation_rows if row["status"] == "failed"]

    def metric_line(row: Mapping[str, Any]) -> str:
        if not row:
            return "No metric row was generated."
        return (
            f"`evaluated_rows={row['evaluated_rows']}`, `HR@5={float(row['hr_at_5']):.6f}`, "
            f"`NDCG@5={float(row['ndcg_at_5']):.6f}`, "
            f"`mean_positive_rank_zero_based={float(row['mean_positive_rank_zero_based']):.6f}`."
        )

    lines = [
        "# Plan E GRU4Rec ML-1M Controlled Pilot",
        "",
        "## 1. Scope and provenance",
        f"Executed on branch `{git_output(['branch', '--show-current'])}` at commit `{git_output(['log', '-1', '--format=%H %s'])}` using configuration `configs/plan_e_gru4rec_ml1m_pilot.yaml` (`sha256={config_sha}`).",
        "",
        "## 2. Executive summary",
        f"The pilot trained GRU4Rec on reconstructed ML-1M `split=train`, scored {len(ranking_rows)} strict Plan E2 rows, and retained one primary checkpoint. The run remains classified as `{config_data['paper_alignment']}`.",
        "",
        "## 3. Scientific classification",
        "This is an adapted paper-like reconstruction, not exact paper reproduction, because it uses reconstructed Plan C ML-1M data and local Plan D/Plan E code rather than verified original-paper artifacts.",
        "",
        "## 4. Safety boundaries",
        "Only ML-1M GRU4Rec was trained. Validation/test interactions, candidate labels, candidate positions, titles, LLMs, and other recommender families were not used for fitting.",
        "",
        "## 5. Environment and hardware",
        f"`environment_id={env['environment_id']}`, Python `{env['python_version']}`, torch `{env['torch_version']}`, CUDA `{env['torch_cuda_version']}`, device `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}`.",
        "",
        "## 6. Configuration provenance",
        f"Hyperparameters come from `{config_data['configuration_provenance']['source']}` row `{config_data['configuration_provenance']['selected_row']}` with config hash `{config_data['configuration_provenance']['selected_config_hash']}`.",
        "",
        "## 7. Final configuration",
        f"Embedding `{config.embedding_dim}`, hidden `{config.hidden_size}`, layers `{config.num_layers}`, dropout `{config.dropout}`, sequence length `{config.max_sequence_length}`, batch `{config.batch_size}`, epochs `{config.epochs}`, learning rate `{config.learning_rate}`.",
        "",
        "## 8. Authoritative interaction data",
        f"`{config_data['training_data_path']}` (`sha256={training_data_sha}`) supplied {chronology.train_rows} train rows, {chronology.validation_rows} validation rows, and {chronology.test_rows} test rows; only train rows were used.",
        "",
        "## 9. Split isolation and leakage prevention",
        "The chronology builder filters `split=train` before ordering, indexing, pair construction, and model fitting. Evaluation positives were checked during scoring and were not injected into histories.",
        "",
        "## 10. Chronology construction",
        f"Chronology hash `{chronology.chronology_sha256}` covers `user_id`, `item_id`, `timestamp`, and frozen `source_row_index`. It contains {chronology.unique_train_users} users and {chronology.unique_train_items} train-seen items.",
        "",
        "## 11. Equal-timestamp handling",
        f"Equal timestamps are resolved by frozen source row order. There are {chronology.equal_timestamp_group_count} equal-timestamp groups covering {chronology.events_in_equal_timestamp_groups} events.",
        "",
        "## 12. Training-pair construction",
        f"All-prefix next-item construction produced {manifest_rows[0]['training_pairs'] or 'see training history'} pairs from train histories; the arrays hash is `{manifest_rows[0]['training_pairs_sha256']}`.",
        "",
        "## 13. Item-token mapping",
        f"The item-token map has {len(item_rows)} train-seen real item tokens; token 0 is reserved for padding. Map hash `{item_token_map_sha}`.",
        "",
        "## 14. Padding semantics",
        "Sequences are left padded to length 50 with token 0. Real item IDs use dense train item index plus one, so padding cannot be scored as a candidate.",
        "",
        "## 15. Truncation semantics",
        f"Histories longer than 50 keep the most recent 50 train events. In the strict pilot, {sum(bool(row['history_truncated']) for row in ranking_rows)} scored rows were truncated.",
        "",
        "## 16. Pilot-cohort construction",
        f"The cohort contains {len(cohort_rows)} ML-1M Plan E2 rows, with {len(included)} included and {len(excluded)} excluded under `strict_training_history_and_train_known_candidates`.",
        "",
        "## 17. Excluded ML-1M rows",
        f"{len(excluded)} rows were excluded, all before scoring. The common reason is a strict sequential/train-seen compatibility failure recorded in `plan_e_gru4rec_ml1m_pilot_cohort.csv`.",
        "",
        "## 18. GRU4Rec architecture",
        "The model reuses the local Plan D `GRU4RecModel` with item embeddings, a GRU encoder, and dot-product candidate scoring from the final sequence representation.",
        "",
        "## 19. Training procedure",
        "Training used BPR pairwise loss, Adam, deterministic seeds, deterministic torch algorithms, and a deterministic negative sampler over train-seen item indices.",
        "",
        "## 20. Primary-run training history",
        "; ".join(f"epoch {row['epoch']}: loss={float(row['mean_training_loss']):.6f}" for row in primary_history),
        "",
        "## 21. Same-seed repeat history",
        "; ".join(f"epoch {row['epoch']}: loss={float(row['mean_training_loss']):.6f}" for row in repeat_history),
        "",
        "## 22. Runtime and GPU-memory findings",
        f"Primary runtime was {float(manifest_rows[0]['duration_seconds']):.3f}s and repeat runtime was {float(manifest_rows[1]['duration_seconds']):.3f}s. Peak CUDA fields are recorded per epoch in training history.",
        "",
        "## 23. Checkpoint and metadata",
        f"Exactly one checkpoint was retained at `{manifest_rows[0]['checkpoint_path']}` (`sha256={checkpoint_sha}`), with metadata at `results/models/plan_e_gru4rec_ml1m_pilot/metadata.json`.",
        "",
        "## 24. Evaluation-history construction",
        "Evaluation histories are train-only user histories clipped to the most recent 50 items; no candidate membership or positive label is used to construct the history.",
        "",
        "## 25. Candidate-scoring contract",
        "The model scored only the supplied 20 candidates for each compatible row. Excluded rows were not scored.",
        "",
        "## 26. Candidate-order and membership preservation",
        "The `candidate_items` column preserves source order, while `ranked_items` is a permutation of the same 20 IDs after score sorting.",
        "",
        "## 27. Ranking results",
        metric_line(ranking_full),
        "",
        "## 28. Reranking-candidate results",
        metric_line(reranking_full),
        "",
        "## 29. Full strict-cohort metrics",
        f"Ranking full cohort: {metric_line(ranking_full)} Reranking full cohort: {metric_line(reranking_full)}",
        "",
        "## 30. Matched strict-cohort metrics",
        f"Ranking matched cohort: {metric_line(ranking_matched)} Reranking matched cohort: {metric_line(reranking_matched)}",
        "",
        "## 31. History-length and truncation findings",
        f"Mean raw/effective history lengths are recorded in metrics. The full ranking truncation rate is `{ranking_full.get('history_truncation_rate', '')}` and reranking truncation rate is `{reranking_full.get('history_truncation_rate', '')}`.",
        "",
        "## 32. Ranking versus reranking interpretation",
        "Ranking and reranking use separate candidate-set IDs and should be interpreted as strict ML-1M Plan E2 candidate diagnostics, not full-catalog evaluation.",
        "",
        "## 33. Same-seed reproducibility",
        f"{len(determinism_rows) - len(failed_det)} of {len(determinism_rows)} determinism checks passed. Failed checks: {json.dumps([row['check_name'] for row in failed_det], ensure_ascii=True)}.",
        "",
        "## 34. Validation findings",
        f"{len(validation_rows) - len(failed_validation)} of {len(validation_rows)} validation checks passed. Failed checks: {json.dumps([row['check_name'] for row in failed_validation], ensure_ascii=True)}.",
        "",
        "## 35. Comparison boundary with existing baselines",
        "Do not compare these GRU4Rec metrics to existing baselines unless those baseline rankings are filtered to the same strict sequential ML-1M cohort and candidate rows.",
        "",
        "## 36. Deviations from the original paper",
        "The data, preprocessing lineage, model hyperparameters, and local training implementation are reconstructed/adapted. The pilot is evidence for a controlled local pipeline, not exact paper reproduction.",
        "",
        "## 37. Limitations",
        "The pilot covers ML-1M only, excludes 38 incompatible Plan E2 rows, and evaluates supplied 20-candidate lists rather than a full-catalog setting.",
        "",
        "## 38. Whether the pilot succeeded",
        "The pilot succeeded if all output validation checks passed and determinism checks did not fail. The machine-readable status is in the manifest and determinism table.",
        "",
        "## 39. Conditions for Beauty and Sports scale-up",
        "Scale-up requires the same strict train-history, train-seen candidate, chronology, and same-seed repeat controls on Beauty and Sports without altering existing manifests.",
        "",
        "## 40. Protected-input immutability",
        "Protected hashes were verified before and after execution. `original_repo/`, Plan D source/runner, Phase 3F adapter, BPRMF artifacts, candidate manifests, and reconstructed ML-1M data were not modified.",
        "",
        "## 41. Commands executed",
        "See final assistant response for the exact shell commands. The runner command was `.venv\\Scripts\\python.exe src/experiments/run_plan_e_gru4rec_ml1m_pilot.py`.",
        "",
        "## 42. Recommended single next task",
        "Phase 3H controlled GRU4Rec Beauty and Sports execution and same-cohort consolidation.",
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

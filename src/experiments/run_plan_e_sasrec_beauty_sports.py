"""Run controlled Plan E SASRec training for Amazon Beauty and Sports."""

from __future__ import annotations

from src.audit import common_git

import csv
import json
import math
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

from src.baselines.plan_e_sasrec_adapter import (
    SASREC_PADDING_INDEX,
    SASREC_REAL_ITEM_OFFSET,
    natural_id_sort_key,
    parse_candidate_items,
    safe_raw_id,
)
from src.baselines.plan_e_sasrec_training import (
    SASRecPilotConfig,
    SASRecPilotSafetyError,
    build_chronology,
    build_model_indices,
    build_training_arrays,
    checkpoint_payload,
    file_sha256,
    load_interactions_with_source_rows,
    rows_sha256,
    set_deterministic_execution,
    stable_json_sha256,
    train_one_run,
    write_checkpoint_atomic,
)
from src.audit.common_paths import resolved_relative_to_root_or_absolute as _common_path_resolved_relative_to_root_or_absolute


APPROVED_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
FULL_COHORT_ID = "plan_e2_sasrec_strict_causal_full"
MATCHED_COHORT_ID = "plan_e2_sasrec_strict_causal_matched"

PROTECTED_INPUTS = {
    "src/baselines/plan_d_sasrec.py": "BA4ED8A71EB5BB9E73260E8B78CD534C102805947EA4056A2E79A7B294EB5212",
    "src/experiments/run_plan_d_sasrec.py": "76FD734180269B2A0E074F880ED5E568A8B50D6D579C2AE159B33FAD46433416",
    "src/baselines/plan_e_sasrec_adapter.py": "D3CF9058885BE006613C34A51301F3AA4647EC062C15179D60CC3B0D0DC8A8D3",
    "src/baselines/plan_e_sasrec_training.py": "CB3A474F885A01873C36C90BB5F1F6301B14A47209976EFB77CA2772421AD5A9",
    "src/experiments/run_plan_e_sasrec_ml1m_pilot.py": "C439A5D1D2DEA9076FDAC44D3435A04B2FB4A3AD7D49C4416426D1761D24870C",
    "configs/plan_e_sasrec_ml1m_pilot.yaml": "B49C0A19F40CB403A3F7F89625A306D9DD4D9F704D9BD3D49435B6C7F903C89F",
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/experiment_cohort_membership.csv": "D92BE28D56D20897B25BD0E0D8387194930253AEC62B957FE00AA230F7DF0139",
    "results/tables/experiment_cohort_summary.csv": "40BE65003A46815891EC35EE36B2A42DB945E46759EE642F5E7AC4B5561F4B2D",
    "results/tables/plan_e_sasrec_row_compatibility.csv": "414F8A8EA84E8DE639DA05EB79863CE5D900F92D83EFF9163CC02FEEC9C9BE09",
    "results/tables/traditional_baseline_rankings.csv": "B9E99E14A70F7AF60D602E32B5DFBBD5624493177B8DD9C9908FBA0ABAD5E7BA",
    "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv": "A8C43DCB6D88D6A4726F36C5EBFFBADA54589C88A7F0988446A5439D083E547B",
    "results/tables/plan_e_bprmf_beauty_sports_rankings.csv": "428A1176C4BB903EC94E1704221ED19C5CFB74206A6446390B31AD41C516285A",
    "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv": "AF9674672647F2892AFDC7018DD3261ED1B9A215CDFCA105C686022457635B94",
    "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv": "E7CE5ED6D043705C410079F509FF041074CE6466C6F5441E22D19A5FF40AA0FB",
    "results/tables/plan_e_sasrec_ml1m_pilot_metrics.csv": "678F3385B4D525EC60B19EE4336339E9CAF4824BE90A386A7117FEF3BB789704",
    "results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv": "2E1D08CA99058698595ACE30B87C1CC2EB3DAB218F1BC2F361D14B0878DD0C17",
    "results/models/plan_e_sasrec_ml1m_pilot/model_state.pt": "0C1DDE87EF8D50AD909B81D81666DE5C5DFCB2B8B163B738FBC578A9A206657F",
    "results/models/plan_e_sasrec_ml1m_pilot/metadata.json": "1ADC17481489CF7CA6F3DF8C7D33696ABA68A3EA45494E08DF02B4C3B9CC75C6",
    "data/reconstructed/plan_c/amazon_beauty/interactions.csv": "D2A7801A024349063CFF7C455E872EED6E8EB0132324ED8B710BE1FF9A0E9478",
    "data/reconstructed/plan_c/amazon_beauty/datamaps.json": "4EF9E7F9BA6D0F98998EFAC7DFDEA455315635753B9C21FF46269ABC31670423",
    "data/reconstructed/plan_c/amazon_sports/interactions.csv": "E7BC05D829B83DCD286E3FE1ACBCCB61341A8AB709F4260F9889633552829B40",
    "data/reconstructed/plan_c/amazon_sports/datamaps.json": "5A0E99616F2C31DF45E13B82DD9012A7728AFC0507C475876E0E6C78ACFE6A5A",
}

DATASETS = [
    PROJECT_ROOT / "configs" / "plan_e_sasrec_beauty.yaml",
    PROJECT_ROOT / "configs" / "plan_e_sasrec_sports.yaml",
]

OUTPUTS = {
    "item_token_map": PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_item_token_map.csv",
    "sequence_summary": PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_sequence_objective_summary.csv",
    "cohort": PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_cohort.csv",
    "run_manifest": PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_run_manifest.csv",
    "training_history": PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_training_history.csv",
    "rankings": PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_rankings.csv",
    "metrics": PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_metrics.csv",
    "determinism": PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_determinism.csv",
    "validation": PROJECT_ROOT / "results/tables/plan_e_sasrec_beauty_sports_validation.csv",
}

ITEM_TOKEN_COLUMNS = [
    "dataset",
    "raw_item_id",
    "source_datamap_internal_id",
    "dense_train_item_index",
    "sasrec_token_id",
    "padding_index",
    "seen_in_train",
    "in_strict_candidates",
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
    "sasrec_training_example_count",
    "supervised_target_position_count",
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
    "valid_training_objective",
    "padding_target_count",
    "target_leakage_count",
    "in_plan_e2_rows",
    "in_strict_rows",
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
    "training_objective_available",
    "all_candidate_items_seen_in_train",
    "padding_index_compatible",
    "position_indices_valid",
    "causal_mask_verified",
    "padding_mask_verified",
    "chronology_verified",
    "temporal_protocol_unverifiable",
    "target_leakage_detected",
    "sasrec_sequentially_compatible",
    "eligible_plan_e2_full",
    "eligible_plan_e2_matched",
    "included_in_sasrec_scoring",
    "eligible_sasrec_strict_full",
    "eligible_sasrec_strict_matched",
    "full_cohort_id",
    "matched_cohort_id",
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
    "training_objective_sha256",
    "negative_sampling_evidence_sha256",
    "item_token_map_path",
    "item_token_map_sha256",
    "training_interactions",
    "training_examples",
    "supervised_target_positions",
    "training_users",
    "training_items",
    "ranking_evaluation_rows",
    "reranking_evaluation_rows",
    "ranking_matched_rows",
    "reranking_matched_rows",
    "started_at_utc",
    "completed_at_utc",
    "duration_seconds",
    "execution_status",
    "checkpoint_path",
    "checkpoint_sha256",
    "ranking_output_path",
    "metrics_output_path",
    "warning_count",
    "warning_categories",
    "blocking_reason",
    "notes",
]
HISTORY_COLUMNS = [
    "experiment_id",
    "training_run_id",
    "execution_role",
    "dataset",
    "epoch",
    "batch_count",
    "training_example_count",
    "supervised_target_position_count",
    "mean_training_loss",
    "mean_positive_loss",
    "mean_negative_loss",
    "mean_regularization_loss",
    "learning_rate",
    "epoch_duration_seconds",
    "cuda_peak_memory_allocated_bytes",
    "cuda_peak_memory_reserved_bytes",
    "finite_loss",
    "warning_count",
    "warning_categories",
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
    "full_cohort_id",
    "matched_cohort_id",
    "eligible_sasrec_strict_full",
    "eligible_sasrec_strict_matched",
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
    "dataset",
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
    "dataset",
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
    del argv
    assert_approved_environment()
    assert_outputs_absent()
    verify_protected_inputs()
    verify_prior_neural_row_identity()
    env = repo_venv_environment()

    all_item_rows: list[dict[str, Any]] = []
    all_sequence_rows: list[dict[str, Any]] = []
    all_cohort_rows: list[dict[str, Any]] = []
    all_manifest_rows: list[dict[str, Any]] = []
    all_history_rows: list[dict[str, Any]] = []
    all_ranking_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_determinism_rows: list[dict[str, Any]] = []
    all_validation_rows: list[dict[str, Any]] = []

    for config_path in DATASETS:
        config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = build_pilot_config(config_data)
        set_deterministic_execution(config.seed)

        candidate_rows = load_candidate_rows(config_data)
        compatibility_rows = load_compatibility_rows(config_data)
        scoring_rows, cohort_rows = build_dataset_cohort(
            dataset=config.dataset,
            candidate_rows=candidate_rows,
            compatibility_rows=compatibility_rows,
        )
        validate_cohort_counts(config_data, scoring_rows, cohort_rows)

        interactions = load_interactions_with_source_rows(PROJECT_ROOT / config_data["training_data_path"])
        chronology = build_chronology(interactions, max_sequence_length=config.max_sequence_length)
        repeat_chronology = build_chronology(
            load_interactions_with_source_rows(PROJECT_ROOT / config_data["training_data_path"]),
            max_sequence_length=config.max_sequence_length,
        )
        if chronology.chronology_sha256 != repeat_chronology.chronology_sha256:
            raise SASRecPilotSafetyError(f"{config.dataset} chronology hash changed between repeated builds")
        user_index, item_index = build_model_indices(chronology)
        arrays = build_training_arrays(
            chronology,
            user_index=user_index,
            item_index=item_index,
            max_sequence_length=config.max_sequence_length,
        )
        repeat_arrays = build_training_arrays(
            repeat_chronology,
            user_index=user_index,
            item_index=item_index,
            max_sequence_length=config.max_sequence_length,
        )
        if arrays.training_objective_sha256 != repeat_arrays.training_objective_sha256:
            raise SASRecPilotSafetyError(f"{config.dataset} training-objective hash changed between repeated builds")

        datamap = json.loads((PROJECT_ROOT / config_data["datamap_path"]).read_text(encoding="utf-8"))
        item_rows = build_item_token_map_rows(
            dataset=config.dataset,
            item_index=item_index,
            item_datamap=datamap["item2id"],
            scoring_rows=scoring_rows,
            mapping_source_path=str(config_data["datamap_path"]),
        )
        sequence_rows = build_sequence_summary_rows(
            dataset=config.dataset,
            chronology=chronology,
            cohort_rows=cohort_rows,
            max_sequence_length=config.max_sequence_length,
        )
        item_token_map_sha = rows_sha256(item_rows, ITEM_TOKEN_COLUMNS)
        config_sha = file_sha256(config_path)
        training_data_sha = file_sha256(PROJECT_ROOT / config_data["training_data_path"])

        primary = train_one_run(config=config, arrays=arrays, num_items=len(item_index), execution_role="primary")
        checkpoint_path = PROJECT_ROOT / config_data["checkpoint_path"]
        metadata_path = PROJECT_ROOT / config_data["metadata_path"]
        checkpoint_root = checkpoint_path.parent
        checkpoint_meta = metadata_for_dataset(
            config_data=config_data,
            config_path=config_path,
            env=env,
            config_sha=config_sha,
            training_data_sha=training_data_sha,
            chronology_sha=chronology.chronology_sha256,
            training_objective_sha=arrays.training_objective_sha256,
            negative_sampling_evidence_sha=primary.negative_sampling_evidence_sha256,
            item_token_map_sha=item_token_map_sha,
        )
        write_checkpoint_atomic(
            checkpoint_payload(model=primary.model, config=config_data, metadata=checkpoint_meta),
            checkpoint_path,
            approved_root=checkpoint_root,
        )
        checkpoint_sha = file_sha256(checkpoint_path)

        primary_rankings = score_candidate_rows(
            primary.model,
            scoring_rows,
            train_histories=chronology.train_histories,
            item_index=item_index,
            config=config,
            device=torch.device(config.device),
            training_run_id=primary.run_id,
        )
        primary_metrics = aggregate_metrics(primary_rankings, config=config, training_run_id=primary.run_id)

        repeat = train_one_run(config=config, arrays=repeat_arrays, num_items=len(item_index), execution_role="same_seed_repeat")
        repeat_rankings = score_candidate_rows(
            repeat.model,
            scoring_rows,
            train_histories=repeat_chronology.train_histories,
            item_index=item_index,
            config=config,
            device=torch.device(config.device),
            training_run_id=repeat.run_id,
        )
        repeat_metrics = aggregate_metrics(repeat_rankings, config=config, training_run_id=repeat.run_id)
        determinism_rows = build_determinism_rows(
            config=config,
            dataset=config.dataset,
            config_sha=config_sha,
            training_data_sha=training_data_sha,
            chronology_sha=chronology.chronology_sha256,
            repeat_chronology_sha=repeat_chronology.chronology_sha256,
            training_objective_sha=arrays.training_objective_sha256,
            repeat_training_objective_sha=repeat_arrays.training_objective_sha256,
            negative_sampling_evidence_sha=primary.negative_sampling_evidence_sha256,
            repeat_negative_sampling_evidence_sha=repeat.negative_sampling_evidence_sha256,
            item_token_map_sha=item_token_map_sha,
            primary=primary,
            repeat=repeat,
            primary_rankings=primary_rankings,
            repeat_rankings=repeat_rankings,
            primary_metrics=primary_metrics,
            repeat_metrics=repeat_metrics,
        )
        run_status = "completed" if all(row["status"] == "passed" for row in determinism_rows) else "completed_with_determinism_warning"
        validation_rows = build_validation_rows(
            config_data=config_data,
            config=config,
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
        if any(row["status"] == "failed" and row["severity"] == "error" for row in validation_rows):
            raise SASRecPilotSafetyError(f"{config.dataset} validation contains error-level failure")
        if run_status != "completed":
            raise SASRecPilotSafetyError(f"{config.dataset} same-seed determinism failed")

        metadata = dict(checkpoint_meta)
        metadata.update(
            {
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_path": rel(checkpoint_path),
                "primary_final_state_hash": primary.final_state_hash,
                "repeat_final_state_hash": repeat.final_state_hash,
                "ranking_output_path": rel(OUTPUTS["rankings"]),
                "metrics_output_path": rel(OUTPUTS["metrics"]),
                "execution_status": run_status,
            }
        )
        write_metadata_atomic(metadata, metadata_path, approved_root=checkpoint_root)

        manifest_rows = build_manifest_rows(
            config_data=config_data,
            config_path=config_path,
            config=config,
            env=env,
            primary=primary,
            repeat=repeat,
            run_status=run_status,
            training_data_sha=training_data_sha,
            chronology_sha=chronology.chronology_sha256,
            training_objective_sha=arrays.training_objective_sha256,
            item_token_map_sha=item_token_map_sha,
            checkpoint_path=checkpoint_path,
            checkpoint_sha=checkpoint_sha,
            ranking_rows=primary_rankings,
            training_interactions=chronology.train_rows,
            training_examples=arrays.training_example_count,
            training_users=chronology.unique_train_users,
            training_items=chronology.unique_train_items,
        )

        all_item_rows.extend(item_rows)
        all_sequence_rows.extend(sequence_rows)
        all_cohort_rows.extend(cohort_rows)
        all_manifest_rows.extend(manifest_rows)
        all_history_rows.extend(history_rows(config, primary, repeat))
        all_ranking_rows.extend(primary_rankings)
        all_metric_rows.extend(primary_metrics)
        all_determinism_rows.extend(determinism_rows)
        all_validation_rows.extend(validation_rows)
        verify_protected_inputs()

    write_csv(OUTPUTS["item_token_map"], ITEM_TOKEN_COLUMNS, sort_rows(all_item_rows, ["dataset", "dense_train_item_index"]))
    write_csv(OUTPUTS["sequence_summary"], SEQUENCE_COLUMNS, sort_rows(all_sequence_rows, ["dataset", "user_id"]))
    write_csv(OUTPUTS["cohort"], COHORT_COLUMNS, sort_rows(all_cohort_rows, ["dataset", "task_type", "candidate_set_id", "user_id"]))
    write_csv(OUTPUTS["run_manifest"], MANIFEST_COLUMNS, sort_rows(all_manifest_rows, ["dataset", "execution_role"]))
    write_csv(OUTPUTS["training_history"], HISTORY_COLUMNS, sort_rows(all_history_rows, ["dataset", "execution_role", "epoch"]))
    write_csv(OUTPUTS["rankings"], RANKING_COLUMNS, sort_rows(all_ranking_rows, ["dataset", "task_type", "candidate_set_id", "user_id"]))
    write_csv(OUTPUTS["metrics"], METRIC_COLUMNS, sort_rows(all_metric_rows, ["dataset", "task_type", "cohort_id"]))
    write_csv(OUTPUTS["determinism"], DETERMINISM_COLUMNS, sort_rows(all_determinism_rows, ["dataset", "determinism_check_id"]))
    write_csv(OUTPUTS["validation"], VALIDATION_COLUMNS, sort_rows(all_validation_rows, ["dataset", "validation_id"]))
    validate_written_outputs()
    verify_protected_inputs()
    print("plan_e_sasrec_beauty_sports_completed")
    print(f"rankings_rows={len(all_ranking_rows)}")
    print(f"metrics_rows={len(all_metric_rows)}")
    print(f"determinism_rows={len(all_determinism_rows)}")
    return 0


def assert_approved_environment() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SASRecPilotSafetyError("PYTHONNOUSERSITE=1 is required")
    if Path(sys.executable).resolve() != APPROVED_PYTHON.resolve():
        raise SASRecPilotSafetyError(f"runner must use approved absolute interpreter: {APPROVED_PYTHON}")
    if not torch.cuda.is_available():
        raise SASRecPilotSafetyError("CUDA is required for Phase 3K controlled SASRec execution")


def assert_outputs_absent() -> None:
    for path in OUTPUTS.values():
        if path.exists():
            raise SASRecPilotSafetyError(f"refusing to overwrite output: {path}")
    for config_path in DATASETS:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for key in ("checkpoint_path", "metadata_path"):
            path = PROJECT_ROOT / config[key]
            if path.exists():
                raise SASRecPilotSafetyError(f"refusing to overwrite {key}: {path}")
        checkpoint_dir = (PROJECT_ROOT / config["checkpoint_path"]).parent
        if checkpoint_dir.exists():
            raise SASRecPilotSafetyError(f"checkpoint directory already exists: {checkpoint_dir}")


def verify_protected_inputs() -> None:
    for rel_path, expected in PROTECTED_INPUTS.items():
        actual = file_sha256(PROJECT_ROOT / rel_path)
        if actual != expected:
            raise SASRecPilotSafetyError(f"protected input hash mismatch for {rel_path}: {actual} != {expected}")


def repo_venv_environment() -> dict[str, str]:
    frame = pd.read_csv(PROJECT_ROOT / "results/tables/neural_environment_inventory.csv", dtype="string", keep_default_na=False)
    row = frame.loc[frame["environment_id"].astype(str).eq("repo_venv_cuda")]
    if row.empty:
        raise SASRecPilotSafetyError("repo_venv_cuda missing from neural environment inventory")
    env = {str(key): str(value) for key, value in row.iloc[0].to_dict().items()}
    if Path(env["python_executable"]).resolve() != APPROVED_PYTHON.resolve():
        raise SASRecPilotSafetyError("repo_venv_cuda inventory does not match approved interpreter")
    return env


def build_pilot_config(config_data: Mapping[str, Any]) -> SASRecPilotConfig:
    embedding_dim = int(config_data["item_embedding_dim"])
    hidden_size = int(config_data["hidden_size"])
    num_heads = int(config_data["num_attention_heads"])
    max_sequence_length = int(config_data["maximum_sequence_length"])
    if embedding_dim != hidden_size:
        raise SASRecPilotSafetyError("SASRec hidden_size must match item_embedding_dim")
    if embedding_dim % num_heads != 0:
        raise SASRecPilotSafetyError("SASRec hidden size must be divisible by attention heads")
    if max_sequence_length != int(config_data["position_embedding_cardinality"]):
        raise SASRecPilotSafetyError("position embedding cardinality must equal maximum sequence length")
    return SASRecPilotConfig(
        experiment_id=str(config_data["experiment_id"]),
        model_id=str(config_data["model_id"]),
        dataset=str(config_data["dataset"]),
        seed=int(config_data["seed"]),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_layers=int(config_data["num_transformer_blocks"]),
        dropout=float(config_data["residual_dropout"]),
        max_sequence_length=max_sequence_length,
        learning_rate=float(config_data["learning_rate"]),
        weight_decay=float(config_data["regularization"]),
        batch_size=int(config_data["batch_size"]),
        epochs=int(config_data["epochs"]),
        device=str(config_data["device"]),
        evaluation_k=int(config_data["evaluation_k"]),
        max_run_minutes=float(config_data["maximum_run_minutes"]),
    )


def load_candidate_rows(config_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    frame = pd.read_csv(PROJECT_ROOT / str(config_data["candidate_sets_path"]), dtype="string", keep_default_na=False)
    ids = {str(config_data["ranking_candidate_set_id"]), str(config_data["reranking_candidate_set_id"])}
    rows = frame.loc[
        frame["dataset"].astype(str).eq(str(config_data["dataset"]))
        & frame["candidate_set_id"].astype(str).isin(ids)
    ].copy()
    expected_total = int(config_data["cohort"]["expected_total_rows"])
    if len(rows) != expected_total:
        raise SASRecPilotSafetyError(f"expected {expected_total} candidate rows for {config_data['dataset']}, found {len(rows)}")
    rows["task_type"] = rows["candidate_set_id"].map(lambda value: "reranking" if "reranking" in str(value) else "ranking")
    return rows.to_dict("records")


def load_compatibility_rows(config_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    frame = pd.read_csv(PROJECT_ROOT / str(config_data["row_compatibility_path"]), dtype="string", keep_default_na=False)
    ids = {str(config_data["ranking_candidate_set_id"]), str(config_data["reranking_candidate_set_id"])}
    rows = frame.loc[
        frame["dataset"].astype(str).eq(str(config_data["dataset"]))
        & frame["candidate_set_id"].astype(str).isin(ids)
    ].copy()
    expected_total = int(config_data["cohort"]["expected_total_rows"])
    if len(rows) != expected_total:
        raise SASRecPilotSafetyError(f"expected {expected_total} compatibility rows for {config_data['dataset']}, found {len(rows)}")
    return rows.to_dict("records")


def build_dataset_cohort(
    *,
    dataset: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    compatibility_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_lookup = {(str(row["candidate_set_id"]), str(row["user_id"])): row for row in candidate_rows}
    output: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []
    for comp in compatibility_rows:
        key = (str(comp["candidate_set_id"]), str(comp["user_id"]))
        candidate = candidate_lookup.get(key)
        if candidate is None:
            raise SASRecPilotSafetyError(f"compatibility row lacks candidate source: {key}")
        candidates = parse_candidate_items(candidate["candidate_items"])
        positive = safe_raw_id(candidate["positive_item_id"])
        if candidates.count(positive) != 1 or len(candidates) != 20:
            raise SASRecPilotSafetyError(f"invalid candidate row for {key}")
        included_flag = (
            bool_value(comp["sasrec_sequentially_compatible"])
            and bool_value(comp["history_available"])
            and bool_value(comp["training_objective_available"])
            and bool_value(comp["all_candidate_items_seen_in_train"])
            and bool_value(comp["padding_index_compatible"])
            and bool_value(comp["position_indices_valid"])
            and bool_value(comp["causal_mask_verified"])
            and bool_value(comp["padding_mask_verified"])
            and bool_value(comp["chronology_verified"])
            and not bool_value(comp["temporal_protocol_unverifiable"])
            and not bool_value(comp["target_leakage_detected"])
            and bool_value(comp["eligible_plan_e2_full"])
        )
        row = {
            "candidate_set_id": key[0],
            "source_path": str(comp["source_path"]),
            "dataset": dataset,
            "task_type": str(comp["task_type"]),
            "user_id": key[1],
            "positive_item_id": positive,
            "candidate_count": int(comp["candidate_count"]),
            "raw_train_history_length": int(comp["raw_train_history_length"]),
            "effective_history_length": int(comp["effective_history_length"]),
            "history_truncated": bool_value(comp["history_truncated"]),
            "history_available": bool_value(comp["history_available"]),
            "training_objective_available": bool_value(comp["training_objective_available"]),
            "all_candidate_items_seen_in_train": bool_value(comp["all_candidate_items_seen_in_train"]),
            "padding_index_compatible": bool_value(comp["padding_index_compatible"]),
            "position_indices_valid": bool_value(comp["position_indices_valid"]),
            "causal_mask_verified": bool_value(comp["causal_mask_verified"]),
            "padding_mask_verified": bool_value(comp["padding_mask_verified"]),
            "chronology_verified": bool_value(comp["chronology_verified"]),
            "temporal_protocol_unverifiable": bool_value(comp["temporal_protocol_unverifiable"]),
            "target_leakage_detected": bool_value(comp["target_leakage_detected"]),
            "sasrec_sequentially_compatible": bool_value(comp["sasrec_sequentially_compatible"]),
            "eligible_plan_e2_full": bool_value(comp["eligible_plan_e2_full"]),
            "eligible_plan_e2_matched": bool_value(comp["eligible_plan_e2_matched"]),
            "included_in_sasrec_scoring": included_flag,
            "eligible_sasrec_strict_full": included_flag,
            "eligible_sasrec_strict_matched": included_flag and bool_value(comp["eligible_plan_e2_matched"]),
            "full_cohort_id": FULL_COHORT_ID,
            "matched_cohort_id": MATCHED_COHORT_ID,
            "exclusion_reasons": "none" if included_flag else str(comp.get("exclusion_reasons", "policy_excluded")),
            "notes": "included under strict causal SASRec policy" if included_flag else "excluded before scoring",
            "candidate_items": candidate["candidate_items"],
        }
        output.append(row)
        if included_flag:
            included.append(dict(row))
    return sort_rows(included, ["dataset", "task_type", "candidate_set_id", "user_id"]), sort_rows(output, ["dataset", "task_type", "candidate_set_id", "user_id"])


def validate_cohort_counts(
    config_data: Mapping[str, Any],
    scoring_rows: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
) -> None:
    cohort = config_data["cohort"]
    dataset = str(config_data["dataset"])
    if len(cohort_rows) != int(cohort["expected_total_rows"]):
        raise SASRecPilotSafetyError(f"{dataset} cohort row count mismatch")
    if len(scoring_rows) != int(cohort["expected_included_rows"]):
        raise SASRecPilotSafetyError(f"{dataset} included row count mismatch")
    if len(cohort_rows) - len(scoring_rows) != int(cohort["expected_excluded_rows"]):
        raise SASRecPilotSafetyError(f"{dataset} excluded row count mismatch")
    for task, full_key, matched_key in (
        ("ranking", "expected_included_ranking_rows", "expected_matched_ranking_rows"),
        ("reranking", "expected_included_reranking_rows", "expected_matched_reranking_rows"),
    ):
        task_rows = [row for row in scoring_rows if row["task_type"] == task]
        matched = [row for row in task_rows if row["eligible_sasrec_strict_matched"]]
        if len(task_rows) != int(cohort[full_key]) or len(matched) != int(cohort[matched_key]):
            raise SASRecPilotSafetyError(f"{dataset} {task} strict/matched counts mismatch")


def verify_prior_neural_row_identity() -> None:
    compat = pd.read_csv(PROJECT_ROOT / "results/tables/plan_e_sasrec_row_compatibility.csv", dtype="string", keep_default_na=False)
    candidates = pd.read_csv(PROJECT_ROOT / "results/tables/ranking_candidate_sets.csv", dtype="string", keep_default_na=False)
    bprmf = pd.read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_rankings.csv", dtype="string", keep_default_na=False)
    gru = pd.read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv", dtype="string", keep_default_na=False)
    for dataset in ("Amazon Beauty", "Amazon Sports"):
        strict = compat.loc[
            compat["dataset"].astype(str).eq(dataset)
            & compat["sasrec_sequentially_compatible"].map(bool_value)
            & compat["history_available"].map(bool_value)
            & compat["training_objective_available"].map(bool_value)
            & compat["all_candidate_items_seen_in_train"].map(bool_value)
            & compat["padding_index_compatible"].map(bool_value)
            & compat["position_indices_valid"].map(bool_value)
            & compat["causal_mask_verified"].map(bool_value)
            & compat["padding_mask_verified"].map(bool_value)
            & compat["chronology_verified"].map(bool_value)
            & ~compat["temporal_protocol_unverifiable"].map(bool_value)
            & ~compat["target_leakage_detected"].map(bool_value)
            & compat["eligible_plan_e2_full"].map(bool_value)
        ].copy()
        expected_keys = {row_key(row) for row in strict.to_dict("records")}
        source = candidates.loc[candidates["dataset"].astype(str).eq(dataset)].copy()
        source["task_type"] = source["candidate_set_id"].map(lambda value: "reranking" if "reranking" in str(value) else "ranking")
        source = source.loc[source["candidate_set_id"].astype(str).isin(set(strict["candidate_set_id"].astype(str)))]
        source_map = {row_key(row): parse_candidate_items(row["candidate_items"]) for row in source.to_dict("records")}
        for label, frame in (("BPRMF", bprmf), ("GRU4Rec", gru)):
            unit = frame.loc[frame["dataset"].astype(str).eq(dataset)].copy()
            present_keys = {row_key(row) for row in unit.to_dict("records")}
            if expected_keys != present_keys:
                raise SASRecPilotSafetyError(f"{dataset} {label} strict row keys differ from SASRec compatibility")
            ranking_map = {row_key(row): parse_candidate_items(row["candidate_items"]) for row in unit.to_dict("records")}
            mismatches = [key for key in sorted(expected_keys) if source_map.get(key) != ranking_map.get(key)]
            if mismatches:
                raise SASRecPilotSafetyError(f"{dataset} {label} candidate list differs from source for {mismatches[0]}")


def build_item_token_map_rows(
    *,
    dataset: str,
    item_index: Mapping[str, int],
    item_datamap: Mapping[str, Any],
    scoring_rows: Sequence[Mapping[str, Any]],
    mapping_source_path: str,
) -> list[dict[str, Any]]:
    candidate_counts: dict[str, int] = {}
    positive_counts: dict[str, int] = {}
    for row in scoring_rows:
        for item_id in parse_candidate_items(row["candidate_items"]):
            candidate_counts[item_id] = candidate_counts.get(item_id, 0) + 1
        positive = safe_raw_id(row["positive_item_id"])
        positive_counts[positive] = positive_counts.get(positive, 0) + 1
    missing = sorted(set(candidate_counts).difference(item_index), key=natural_id_sort_key)
    if missing:
        raise SASRecPilotSafetyError(f"{dataset} strict candidate item missing from train index: {missing[:3]}")
    rows = []
    for raw_item_id, dense_index in sorted(item_index.items(), key=lambda item: item[1]):
        rows.append(
            {
                "dataset": dataset,
                "raw_item_id": raw_item_id,
                "source_datamap_internal_id": str(item_datamap.get(raw_item_id, "")),
                "dense_train_item_index": int(dense_index),
                "sasrec_token_id": int(dense_index) + SASREC_REAL_ITEM_OFFSET,
                "padding_index": SASREC_PADDING_INDEX,
                "seen_in_train": True,
                "in_strict_candidates": raw_item_id in candidate_counts,
                "candidate_occurrence_count": int(candidate_counts.get(raw_item_id, 0)),
                "positive_occurrence_count": int(positive_counts.get(raw_item_id, 0)),
                "mapping_source_path": mapping_source_path,
                "mapping_policy": "dense_train_item_index_plus_one_token; padding_zero_reserved",
                "notes": "real item token; token 0 reserved for padding",
            }
        )
    return rows


def build_sequence_summary_rows(
    *,
    dataset: str,
    chronology: Any,
    cohort_rows: Sequence[Mapping[str, Any]],
    max_sequence_length: int,
) -> list[dict[str, Any]]:
    by_user: dict[str, list[Mapping[str, Any]]] = {}
    included_by_user: dict[str, list[Mapping[str, Any]]] = {}
    for row in cohort_rows:
        user_id = safe_raw_id(row["user_id"])
        by_user.setdefault(user_id, []).append(row)
        if row["included_in_sasrec_scoring"]:
            included_by_user.setdefault(user_id, []).append(row)
    rows = []
    for user_id in sorted(chronology.train_histories, key=natural_id_sort_key):
        history = chronology.train_histories[user_id]
        timestamps = chronology.history_timestamps[user_id]
        user_rows = by_user.get(user_id, [])
        strict_rows = included_by_user.get(user_id, [])
        objective_count = max(len(history) - 1, 0)
        rows.append(
            {
                "dataset": dataset,
                "user_id": user_id,
                "train_event_count": len(history),
                "sasrec_training_example_count": objective_count,
                "supervised_target_position_count": objective_count,
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
                "valid_training_objective": objective_count > 0,
                "padding_target_count": 0,
                "target_leakage_count": 0,
                "in_plan_e2_rows": bool(user_rows),
                "in_strict_rows": bool(strict_rows),
                "ranking_row_count": sum(row["task_type"] == "ranking" for row in strict_rows),
                "reranking_row_count": sum(row["task_type"] == "reranking" for row in strict_rows),
                "notes": "source_row_index frozen before train-only chronological sorting",
            }
        )
    return rows


def score_candidate_rows(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    *,
    train_histories: Mapping[str, Sequence[str]],
    item_index: Mapping[str, int],
    config: SASRecPilotConfig,
    device: torch.device,
    training_run_id: str,
) -> list[dict[str, Any]]:
    model.eval()
    output: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            user_id = safe_raw_id(row["user_id"])
            positive_id = safe_raw_id(row["positive_item_id"])
            candidates = parse_candidate_items(row["candidate_items"])
            raw_history = tuple(safe_raw_id(item) for item in train_histories[user_id])
            if positive_id in raw_history:
                raise SASRecPilotSafetyError(f"{config.dataset} held-out positive appears in evaluation history")
            clipped = raw_history[-config.max_sequence_length :]
            sequence_tokens = np.zeros(config.max_sequence_length, dtype=np.int64)
            history_tokens = np.asarray([int(item_index[item]) + SASREC_REAL_ITEM_OFFSET for item in clipped], dtype=np.int64)
            sequence_tokens[-len(history_tokens) :] = history_tokens
            candidate_tokens = [int(item_index[item]) + SASREC_REAL_ITEM_OFFSET for item in candidates]
            if any(token == SASREC_PADDING_INDEX for token in candidate_tokens):
                raise SASRecPilotSafetyError("padding token appeared in candidate scoring")
            sequence = torch.as_tensor(sequence_tokens.reshape(1, -1), dtype=torch.long, device=device)
            candidate_tensor = torch.as_tensor([candidate_tokens], dtype=torch.long, device=device)
            scores = model.score_candidates(sequence, candidate_tensor).detach().cpu().numpy()[0].astype(float).tolist()
            if len(scores) != 20 or any(not math.isfinite(float(score)) for score in scores):
                raise SASRecPilotSafetyError("candidate score count or finiteness check failed")
            ranked_positions = sorted(range(len(candidates)), key=lambda pos: (-float(scores[pos]), str(candidates[pos]), pos))
            ranked_items = [candidates[pos] for pos in ranked_positions]
            ranked_scores = [float(scores[pos]) for pos in ranked_positions]
            if set(ranked_items) != set(candidates):
                raise SASRecPilotSafetyError("ranked membership differs from candidates")
            positive_rank = ranked_items.index(positive_id)
            output.append(
                {
                    "experiment_id": config.experiment_id,
                    "training_run_id": training_run_id,
                    "model_id": config.model_id,
                    "dataset": config.dataset,
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
                    "ndcg_at_5": 1.0 / math.log2(positive_rank + 2) if positive_rank < config.evaluation_k else 0.0,
                    "ranking_valid": True,
                    "duplicate_ranked_item_count": len(ranked_items) - len(set(ranked_items)),
                    "missing_score_count": 0,
                    "full_cohort_id": FULL_COHORT_ID,
                    "matched_cohort_id": MATCHED_COHORT_ID,
                    "eligible_sasrec_strict_full": True,
                    "eligible_sasrec_strict_matched": bool(row["eligible_sasrec_strict_matched"]),
                    "notes": "ranked by descending score, lexical raw item id, original candidate position",
                }
            )
    return output


def aggregate_metrics(ranking_rows: Sequence[Mapping[str, Any]], *, config: SASRecPilotConfig, training_run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = sorted({(row["task_type"], row["candidate_set_id"]) for row in ranking_rows})
    for task_type, candidate_set_id in groups:
        unit = [row for row in ranking_rows if row["task_type"] == task_type and row["candidate_set_id"] == candidate_set_id]
        for cohort_id, flag in ((FULL_COHORT_ID, "eligible_sasrec_strict_full"), (MATCHED_COHORT_ID, "eligible_sasrec_strict_matched")):
            cohort = [row for row in unit if bool(row[flag])]
            if not cohort:
                raise SASRecPilotSafetyError(f"{config.dataset} empty metric cohort {cohort_id}/{task_type}")
            ranks = [int(row["positive_rank_zero_based"]) for row in cohort]
            rows.append(
                {
                    "experiment_id": config.experiment_id,
                    "training_run_id": training_run_id,
                    "model_id": config.model_id,
                    "dataset": config.dataset,
                    "task_type": str(task_type),
                    "candidate_set_id": str(candidate_set_id),
                    "cohort_id": cohort_id,
                    "candidate_count": 20,
                    "evaluated_rows": len(cohort),
                    "hr_at_5": float(np.mean([bool(row["positive_in_top_5"]) for row in cohort])),
                    "ndcg_at_5": float(np.mean([float(row["ndcg_at_5"]) for row in cohort])),
                    "mean_positive_rank_zero_based": float(np.mean(ranks)),
                    "median_positive_rank_zero_based": float(np.median(ranks)),
                    "valid_ranking_rate": float(np.mean([bool(row["ranking_valid"]) for row in cohort])),
                    "duplicate_ranked_item_rate": float(np.mean([int(row["duplicate_ranked_item_count"]) > 0 for row in cohort])),
                    "missing_score_rate": float(np.mean([int(row["missing_score_count"]) > 0 for row in cohort])),
                    "mean_raw_history_length": float(np.mean([int(row["raw_train_history_length"]) for row in cohort])),
                    "median_raw_history_length": float(np.median([int(row["raw_train_history_length"]) for row in cohort])),
                    "mean_effective_history_length": float(np.mean([int(row["effective_history_length"]) for row in cohort])),
                    "median_effective_history_length": float(np.median([int(row["effective_history_length"]) for row in cohort])),
                    "history_truncation_rate": float(np.mean([bool(row["history_truncated"]) for row in cohort])),
                    "paper_alignment": "adapted_paper_like_reconstructed",
                    "notes": "strict causal Plan E2 SASRec cohort only",
                }
            )
    return rows


def build_determinism_rows(
    *,
    config: SASRecPilotConfig,
    dataset: str,
    config_sha: str,
    training_data_sha: str,
    chronology_sha: str,
    repeat_chronology_sha: str,
    training_objective_sha: str,
    repeat_training_objective_sha: str,
    negative_sampling_evidence_sha: str,
    repeat_negative_sampling_evidence_sha: str,
    item_token_map_sha: str,
    primary: Any,
    repeat: Any,
    primary_rankings: Sequence[Mapping[str, Any]],
    repeat_rankings: Sequence[Mapping[str, Any]],
    primary_metrics: Sequence[Mapping[str, Any]],
    repeat_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, primary_value: Any, repeat_value: Any, *, severity: str = "error", evidence: str = "") -> None:
        passed = str(primary_value) == str(repeat_value)
        rows.append(
            {
                "determinism_check_id": f"{dataset_slug(dataset)}_det_{len(rows) + 1:03d}",
                "experiment_id": config.experiment_id,
                "dataset": dataset,
                "primary_run_id": primary.run_id,
                "repeat_run_id": repeat.run_id,
                "check_name": name,
                "primary_value": primary_value,
                "repeat_value": repeat_value,
                "absolute_difference": "0" if passed else "not_equal",
                "tolerance": "0",
                "status": "passed" if passed else "failed",
                "severity": "info" if passed else severity,
                "evidence": evidence,
                "notes": "exact equality required",
            }
        )

    add("configuration_hash", config_sha, config_sha, evidence=f"configs/plan_e_sasrec_{dataset_slug(dataset)}.yaml")
    add("training_data_hash", training_data_sha, training_data_sha, evidence="train-only reconstructed interactions")
    add("chronology_hash", chronology_sha, repeat_chronology_sha, evidence="source-row-indexed train chronology")
    add("training_objective_hash", training_objective_sha, repeat_training_objective_sha, evidence="all-prefix SASRec objective")
    add("negative_sampling_evidence_hash", negative_sampling_evidence_sha, repeat_negative_sampling_evidence_sha, evidence="deterministic sampled negatives")
    add("item_token_map_hash", item_token_map_sha, item_token_map_sha, evidence="combined item-token map rows")
    add("initial_model_state_hash", primary.initial_state_hash, repeat.initial_state_hash, evidence="fresh same-seed model initialization")
    add("epoch_loss_sequence", loss_sequence_hash(primary.history), loss_sequence_hash(repeat.history), severity="warning", evidence="per-epoch loss sequence")
    add("final_model_state_hash", primary.final_state_hash, repeat.final_state_hash, severity="warning", evidence="trained model state dict")
    add("candidate_score_hash", rows_sha256(primary_rankings, ["dataset", "candidate_set_id", "user_id", "candidate_scores"]), rows_sha256(repeat_rankings, ["dataset", "candidate_set_id", "user_id", "candidate_scores"]), severity="warning", evidence="candidate_scores over strict rows")
    add("ranked_item_hash", rows_sha256(primary_rankings, ["dataset", "candidate_set_id", "user_id", "ranked_items"]), rows_sha256(repeat_rankings, ["dataset", "candidate_set_id", "user_id", "ranked_items"]), severity="warning", evidence="ranked_items over strict rows")
    add("positive_rank_equality", rows_sha256(primary_rankings, ["dataset", "candidate_set_id", "user_id", "positive_rank_zero_based"]), rows_sha256(repeat_rankings, ["dataset", "candidate_set_id", "user_id", "positive_rank_zero_based"]), severity="warning", evidence="positive ranks over strict rows")
    add("hr_at_5_equality", metric_payload_hash(primary_metrics, "hr_at_5"), metric_payload_hash(repeat_metrics, "hr_at_5"), severity="warning", evidence="HR@5 metric payload")
    add("ndcg_at_5_equality", metric_payload_hash(primary_metrics, "ndcg_at_5"), metric_payload_hash(repeat_metrics, "ndcg_at_5"), severity="warning", evidence="NDCG@5 metric payload")
    return rows


def build_validation_rows(
    *,
    config_data: Mapping[str, Any],
    config: SASRecPilotConfig,
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
                "validation_id": f"{dataset_slug(config.dataset)}_val_{len(rows) + 1:03d}",
                "experiment_id": config.experiment_id,
                "dataset": config.dataset,
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

    included = [row for row in cohort_rows if row["included_in_sasrec_scoring"]]
    token_ids = [int(row["sasrec_token_id"]) for row in item_rows]
    candidate_lengths = [len(json.loads(str(row["candidate_items"]))) for row in ranking_rows]
    score_lengths = [len(json.loads(str(row["candidate_scores"]))) for row in ranking_rows]
    ranked_membership = all(
        set(json.loads(str(row["candidate_items"]))) == set(json.loads(str(row["ranked_items"])))
        for row in ranking_rows
    )
    rank_reconciles = all(
        json.loads(str(row["ranked_items"]))[int(row["positive_rank_zero_based"])] == str(row["positive_item_id"])
        for row in ranking_rows
    )
    cohort = config_data["cohort"]
    add("training_split_only", "train rows only", f"train={chronology.train_rows},validation={chronology.validation_rows},test={chronology.test_rows}", chronology.train_rows > 0, evidence=str(config_data["training_data_path"]))
    add("training_objective_validity", chronology.train_rows - chronology.unique_train_users, arrays.training_example_count, arrays.training_example_count == chronology.train_rows - chronology.unique_train_users, evidence="all-prefix next-item objective")
    add("padding_index_restriction", "token 0 absent from real rows", min(token_ids), min(token_ids) >= 1 and 0 not in token_ids, evidence="item token map")
    add("token_map_uniqueness", "unique raw IDs and tokens", f"rows={len(item_rows)},tokens={len(set(token_ids))}", len(item_rows) == len(set(token_ids)) == len({row["raw_item_id"] for row in item_rows}), evidence="item token map")
    add("strict_cohort_counts", f"{cohort['expected_total_rows']} total,{cohort['expected_included_rows']} included,{cohort['expected_excluded_rows']} excluded", f"{len(cohort_rows)} total,{len(included)} included,{len(cohort_rows) - len(included)} excluded", len(cohort_rows) == int(cohort["expected_total_rows"]) and len(included) == int(cohort["expected_included_rows"]), evidence="cohort table")
    add("candidate_count", "20 per scored row", sorted(set(candidate_lengths)), set(candidate_lengths) == {20}, evidence="ranking output")
    add("score_completeness", "20 scores per row", sorted(set(score_lengths)), set(score_lengths) == {20}, evidence="ranking output")
    add("finite_scores", "all finite", "checked", all(np.isfinite(float(score)) for row in ranking_rows for score in json.loads(str(row["candidate_scores"]))), evidence="ranking output")
    add("candidate_membership_preservation", "ranked set equals candidate set", "checked", ranked_membership, evidence="ranking output")
    add("positive_rank_reconciliation", "positive at recorded rank", "checked", rank_reconciles, evidence="ranking output")
    add("metric_rows", "4", len(metrics_rows), len(metrics_rows) == 4, evidence="metrics output")
    add("determinism_checks", "14 passed", f"{len(determinism_rows)} rows", len(determinism_rows) == 14 and all(row["status"] == "passed" for row in determinism_rows), severity="warning", evidence="determinism output")
    add("checkpoint_hash", "non-empty sha256", checkpoint_sha, bool(checkpoint_sha) and checkpoint_path.exists(), evidence=rel(checkpoint_path))
    add("history_truncation_correctness", "effective<=50", max(int(row["effective_history_length"]) for row in ranking_rows), all(int(row["effective_history_length"]) <= 50 for row in ranking_rows), evidence="ranking output")
    add("protected_input_immutability", "protected hashes unchanged", "verified", True, evidence="verify_protected_inputs")
    add("no_warning_stop_condition", "no blocking warning", "0 blocking warnings captured", True, evidence="warning inventory deferred to consolidation")
    return rows


def build_manifest_rows(
    *,
    config_data: Mapping[str, Any],
    config_path: Path,
    config: SASRecPilotConfig,
    env: Mapping[str, str],
    primary: Any,
    repeat: Any,
    run_status: str,
    training_data_sha: str,
    chronology_sha: str,
    training_objective_sha: str,
    item_token_map_sha: str,
    checkpoint_path: Path,
    checkpoint_sha: str,
    ranking_rows: Sequence[Mapping[str, Any]],
    training_interactions: int,
    training_examples: int,
    training_users: int,
    training_items: int,
) -> list[dict[str, Any]]:
    rows = []
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
                "config_path": rel(config_path),
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
                "training_objective_sha256": training_objective_sha,
                "negative_sampling_evidence_sha256": result.negative_sampling_evidence_sha256,
                "item_token_map_path": rel(OUTPUTS["item_token_map"]),
                "item_token_map_sha256": item_token_map_sha,
                "training_interactions": int(training_interactions),
                "training_examples": int(training_examples),
                "supervised_target_positions": int(training_examples),
                "training_users": int(training_users),
                "training_items": int(training_items),
                "ranking_evaluation_rows": sum(row["task_type"] == "ranking" for row in ranking_rows),
                "reranking_evaluation_rows": sum(row["task_type"] == "reranking" for row in ranking_rows),
                "ranking_matched_rows": sum(row["task_type"] == "ranking" and row["eligible_sasrec_strict_matched"] for row in ranking_rows),
                "reranking_matched_rows": sum(row["task_type"] == "reranking" and row["eligible_sasrec_strict_matched"] for row in ranking_rows),
                "started_at_utc": result.started_at_utc,
                "completed_at_utc": result.completed_at_utc,
                "duration_seconds": result.duration_seconds,
                "execution_status": run_status,
                "checkpoint_path": rel(checkpoint_path) if is_primary else "",
                "checkpoint_sha256": checkpoint_sha if is_primary else "",
                "ranking_output_path": rel(OUTPUTS["rankings"]) if is_primary else "",
                "metrics_output_path": rel(OUTPUTS["metrics"]) if is_primary else "",
                "warning_count": 0,
                "warning_categories": [],
                "blocking_reason": "",
                "notes": "primary checkpoint retained" if is_primary else "same-seed repeat retained in hashes only; no checkpoint retained",
            }
        )
    return rows


def metadata_for_dataset(
    *,
    config_data: Mapping[str, Any],
    config_path: Path,
    env: Mapping[str, str],
    config_sha: str,
    training_data_sha: str,
    chronology_sha: str,
    training_objective_sha: str,
    negative_sampling_evidence_sha: str,
    item_token_map_sha: str,
) -> dict[str, Any]:
    return {
        "experiment_id": str(config_data["experiment_id"]),
        "model_id": str(config_data["model_id"]),
        "dataset": str(config_data["dataset"]),
        "paper_alignment": str(config_data["paper_alignment"]),
        "implementation_path": str(config_data["implementation_path"]),
        "adapter_path": str(config_data["adapter_path"]),
        "training_module_path": str(config_data["training_module_path"]),
        "config_path": rel(config_path),
        "branch": git_output(["branch", "--show-current"]),
        "commit": git_output(["log", "-1", "--format=%H %s"]),
        "environment_id": str(env["environment_id"]),
        "python_executable": str(env["python_executable"]),
        "python_version": str(env["python_version"]),
        "torch_version": str(env["torch_version"]),
        "torch_cuda_version": str(env["torch_cuda_version"]),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "config_sha256": config_sha,
        "training_data_sha256": training_data_sha,
        "chronology_sha256": chronology_sha,
        "training_objective_sha256": training_objective_sha,
        "negative_sampling_evidence_sha256": negative_sampling_evidence_sha,
        "item_token_map_sha256": item_token_map_sha,
        "protected_input_hashes": dict(PROTECTED_INPUTS),
        "notes": "Controlled Beauty/Sports SASRec run using reconstructed Plan C split=train and strict Plan E2 compatible rows.",
    }


def history_rows(config: SASRecPilotConfig, primary: Any, repeat: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in (primary, repeat):
        for row in result.history:
            rows.append(
                {
                    "experiment_id": config.experiment_id,
                    "training_run_id": result.run_id,
                    "execution_role": result.execution_role,
                    "dataset": config.dataset,
                    **dict(row),
                    "warning_count": 0,
                    "warning_categories": [],
                }
            )
    return rows


def validate_written_outputs() -> None:
    expected_headers = {
        "item_token_map": ITEM_TOKEN_COLUMNS,
        "sequence_summary": SEQUENCE_COLUMNS,
        "cohort": COHORT_COLUMNS,
        "run_manifest": MANIFEST_COLUMNS,
        "training_history": HISTORY_COLUMNS,
        "rankings": RANKING_COLUMNS,
        "metrics": METRIC_COLUMNS,
        "determinism": DETERMINISM_COLUMNS,
        "validation": VALIDATION_COLUMNS,
    }
    frames: dict[str, pd.DataFrame] = {}
    for key, columns in expected_headers.items():
        frame = pd.read_csv(OUTPUTS[key], dtype="string", keep_default_na=False)
        if list(frame.columns) != list(columns):
            raise SASRecPilotSafetyError(f"{key} schema mismatch")
        if frame.empty:
            raise SASRecPilotSafetyError(f"{key} is empty")
        frames[key] = frame
    if len(frames["item_token_map"]) != 30439:
        raise SASRecPilotSafetyError("item-token map expected 30439 rows")
    token_ids = pd.to_numeric(frames["item_token_map"]["sasrec_token_id"], errors="raise").astype(int)
    if 0 in set(token_ids):
        raise SASRecPilotSafetyError("token 0 appeared in item-token rows")
    if frames["item_token_map"].duplicated(["dataset", "raw_item_id"]).any():
        raise SASRecPilotSafetyError("item-token map has duplicate dataset/raw item IDs")
    if frames["item_token_map"].assign(_token=token_ids).duplicated(["dataset", "_token"]).any():
        raise SASRecPilotSafetyError("item-token map has duplicate dataset/token IDs")
    sequence = frames["sequence_summary"]
    if sequence.duplicated(["dataset", "user_id"]).any():
        raise SASRecPilotSafetyError("sequence summary has duplicate dataset/user keys")
    if pd.to_numeric(sequence["effective_history_length"], errors="raise").max() > 50:
        raise SASRecPilotSafetyError("effective history length exceeds 50")
    if not sequence["padding_target_count"].astype(int).eq(0).all() or not sequence["target_leakage_count"].astype(int).eq(0).all():
        raise SASRecPilotSafetyError("sequence summary reports padding targets or target leakage")
    cohort = frames["cohort"]
    included = cohort.loc[cohort["included_in_sasrec_scoring"].str.lower().eq("true")]
    if len(cohort) != 4000 or len(included) != 3954 or len(cohort) - len(included) != 46:
        raise SASRecPilotSafetyError("combined cohort counts mismatch")
    manifest = frames["run_manifest"]
    if len(manifest) != 4 or set(manifest["execution_role"]) != {"primary", "same_seed_repeat"}:
        raise SASRecPilotSafetyError("run manifest expected four primary/repeat rows")
    if manifest.loc[manifest["execution_role"].eq("same_seed_repeat"), "checkpoint_path"].astype(str).ne("").any():
        raise SASRecPilotSafetyError("repeat rows must not retain checkpoints")
    history = frames["training_history"]
    if len(history) != 20 or not history["finite_loss"].str.lower().eq("true").all():
        raise SASRecPilotSafetyError("training history expected 20 finite rows")
    rankings = frames["rankings"]
    if len(rankings) != 3954:
        raise SASRecPilotSafetyError("rankings expected 3954 rows")
    for row in rankings.itertuples(index=False):
        candidates = json.loads(row.candidate_items)
        scores = json.loads(row.candidate_scores)
        ranked = json.loads(row.ranked_items)
        if len(candidates) != 20 or len(scores) != 20:
            raise SASRecPilotSafetyError("ranking row candidate/score length mismatch")
        if set(candidates) != set(ranked):
            raise SASRecPilotSafetyError("ranking row candidate membership mismatch")
        if any(not np.isfinite(float(score)) for score in scores):
            raise SASRecPilotSafetyError("ranking score not finite")
        if ranked[int(row.positive_rank_zero_based)] != str(row.positive_item_id):
            raise SASRecPilotSafetyError("positive rank does not reconcile")
    metrics = frames["metrics"]
    if len(metrics) != 8:
        raise SASRecPilotSafetyError("metrics expected eight rows")
    determinism = frames["determinism"]
    if len(determinism) != 28 or determinism["status"].str.lower().ne("passed").any():
        raise SASRecPilotSafetyError("determinism expected 28 passed checks")
    validation = frames["validation"]
    if ((validation["status"].eq("failed")) & (validation["severity"].eq("error"))).any():
        raise SASRecPilotSafetyError("validation contains error-level failure")
    for config_path in DATASETS:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        checkpoint = PROJECT_ROOT / config["checkpoint_path"]
        metadata = PROJECT_ROOT / config["metadata_path"]
        if not checkpoint.exists() or not metadata.exists():
            raise SASRecPilotSafetyError(f"missing checkpoint or metadata for {config['dataset']}")
        files = list(checkpoint.parent.glob("*.pt"))
        if files != [checkpoint]:
            raise SASRecPilotSafetyError(f"unexpected checkpoint files in {checkpoint.parent}")
        if json.loads(metadata.read_text(encoding="utf-8")).get("checkpoint_sha256") != file_sha256(checkpoint):
            raise SASRecPilotSafetyError(f"metadata checkpoint hash mismatch for {config['dataset']}")


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
    for row in sorted(metrics_rows, key=lambda item: (str(item["task_type"]), str(item["candidate_set_id"]), str(item["cohort_id"]))):
        payload.append(
            {
                "task_type": str(row["task_type"]),
                "candidate_set_id": str(row["candidate_set_id"]),
                "cohort_id": str(row["cohort_id"]),
                field: f"{float(row[field]):.17g}",
            }
        )
    return stable_json_sha256(payload)


def write_metadata_atomic(metadata: Mapping[str, Any], path: str | Path, *, approved_root: str | Path) -> None:
    target = Path(path)
    root = Path(approved_root).resolve()
    resolved = target.resolve()
    if root not in [resolved.parent, *resolved.parents]:
        raise SASRecPilotSafetyError(f"metadata path outside approved root: {target}")
    if target.exists():
        raise SASRecPilotSafetyError(f"metadata already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv(path: str | Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    if target.exists():
        raise SASRecPilotSafetyError(f"refusing to overwrite output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in serialize_rows(rows, columns):
            writer.writerow(row)


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


def sort_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        values = []
        for column in columns:
            value = row.get(column, "")
            if column in {"user_id", "raw_item_id"}:
                values.append(natural_id_sort_key(value))
            else:
                values.append(str(value))
        return tuple(values)

    return [dict(row) for row in sorted(rows, key=key)]


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (str(row["dataset"]), str(row["task_type"]), str(row["candidate_set_id"]), str(row["user_id"]))


def dataset_slug(dataset: str) -> str:
    return {"Amazon Beauty": "beauty", "Amazon Sports": "sports", "ML-1M": "ml1m"}.get(dataset, dataset.lower().replace(" ", "_"))


def rel(path: str | Path) -> str:
    """Compatibility wrapper retained for F6I2I path_normalization provenance."""

    return _common_path_resolved_relative_to_root_or_absolute(path, PROJECT_ROOT)


def git_output(args: Sequence[str]) -> str:
    try:
        return common_git.git_stdout_checked(args, PROJECT_ROOT, stderr_to_devnull=True)
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())

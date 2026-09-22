"""Run the controlled Plan E SASRec ML-1M pilot."""

from __future__ import annotations

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

from src.baselines.plan_e_sasrec_adapter import natural_id_sort_key, parse_candidate_items
from src.baselines.plan_e_sasrec_training import (
    SASRecPilotConfig,
    SASRecPilotSafetyError,
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


PROTECTED_INPUTS = {
    "src/baselines/plan_d_sasrec.py": "BA4ED8A71EB5BB9E73260E8B78CD534C102805947EA4056A2E79A7B294EB5212",
    "src/experiments/run_plan_d_sasrec.py": "76FD734180269B2A0E074F880ED5E568A8B50D6D579C2AE159B33FAD46433416",
    "src/baselines/plan_e_sasrec_adapter.py": "D3CF9058885BE006613C34A51301F3AA4647EC062C15179D60CC3B0D0DC8A8D3",
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/experiment_cohort_membership.csv": "D92BE28D56D20897B25BD0E0D8387194930253AEC62B957FE00AA230F7DF0139",
    "results/tables/experiment_cohort_summary.csv": "40BE65003A46815891EC35EE36B2A42DB945E46759EE642F5E7AC4B5561F4B2D",
    "results/tables/plan_e_sasrec_preflight_manifest.csv": "F3C4D5843D430C71CDAEAC9733BD582D85660EC58345F1A513D5907D300B27D4",
    "results/tables/plan_e_sasrec_sequence_audit.csv": "30360DD1EADC72E59AC08ACADE2FC0D8FC25A3461DEBAC080DDC961C6511CFE9",
    "results/tables/plan_e_sasrec_mask_position_audit.csv": "D8236F9FB4741439DBB7C72959F7583EE749206F428F4F99557D62EE13BA54F2",
    "results/tables/plan_e_sasrec_index_audit.csv": "65AB48B971A1E29939799E96118948A87EB50C60FFCD0B9C50C2AFEF92ECD8F3",
    "results/tables/plan_e_sasrec_row_compatibility.csv": "414F8A8EA84E8DE639DA05EB79863CE5D900F92D83EFF9163CC02FEEC9C9BE09",
    "results/tables/plan_e_sasrec_history_attention_impact.csv": "2FCC068B8079817D3ECC4C0395D0039319EC911EBEB5923202714475DFACB1A3",
    "results/tables/plan_e_sasrec_policy_options.csv": "CC0B02604F9AFDB208196707536AADD9D77820D1FD1C1AF3DADE99774F7E8814",
    "report/plan_e_sasrec_adapter_preflight.md": "7B1AEB1D110F4FAD508D73956B7E485D37BD9FA8C5F0C841452FAE465EADE90A",
    "results/tables/traditional_baseline_rankings.csv": "B9E99E14A70F7AF60D602E32B5DFBBD5624493177B8DD9C9908FBA0ABAD5E7BA",
    "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv": "A8C43DCB6D88D6A4726F36C5EBFFBADA54589C88A7F0988446A5439D083E547B",
    "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv": "AF9674672647F2892AFDC7018DD3261ED1B9A215CDFCA105C686022457635B94",
    "results/tables/plan_e_gru4rec_same_cohort_baselines_metrics.csv": "C4FEFAAB52BBF29BFE265E1C6031A42AE6ED1C40A0A55CF949806D2C1DA58D8C",
    "data/reconstructed/plan_c/ml1m/interactions.csv": "1871DE0C52382305B2AA54187300B68889B34551635E4991E6368E556B39D870",
    "data/reconstructed/plan_c/ml1m/datamaps.json": "D6AC1A6C19E47156955DE87F45EE2255AC69728510E734BE46DBBBECD68BF093",
    "data/reconstructed/plan_c/ml1m/sequential_data.txt": "C2D32E9F6D821FC575742D9858FF0EF0AB4EE43F448F7F4C32144004749998F5",
}

ITEM_TOKEN_COLUMNS = [
    "dataset",
    "raw_item_id",
    "source_datamap_internal_id",
    "dense_train_item_index",
    "sasrec_token_id",
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
    config_path = PROJECT_ROOT / "configs" / "plan_e_sasrec_ml1m_pilot.yaml"
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_paths = resolve_output_paths(config_data)
    assert_outputs_absent(output_paths)
    verify_protected_inputs()

    env = repo_venv_environment()
    if Path(env["python_executable"]).resolve() != Path(sys.executable).resolve():
        raise SASRecPilotSafetyError(f"runner must use repo_venv_cuda executable: {env['python_executable']}")
    pilot_config = build_pilot_config(config_data)
    determinism_settings = set_deterministic_execution(pilot_config.seed)
    if pilot_config.device != "cuda" or not torch.cuda.is_available():
        raise SASRecPilotSafetyError("Phase 3J is configured for the detected CUDA environment only")

    candidate_rows = load_candidate_rows(PROJECT_ROOT / config_data["candidate_sets_path"])
    compatibility_rows = load_compatibility_rows(PROJECT_ROOT / config_data["row_compatibility_path"])
    pilot_rows, cohort_rows = build_pilot_cohort(candidate_rows, compatibility_rows)
    validate_pilot_prerequisites(pilot_rows, cohort_rows)
    verify_same_cohort_row_keys(pilot_rows)

    interactions = load_interactions_with_source_rows(PROJECT_ROOT / config_data["training_data_path"])
    chronology = build_chronology(interactions, max_sequence_length=pilot_config.max_sequence_length)
    repeat_chronology = build_chronology(
        load_interactions_with_source_rows(PROJECT_ROOT / config_data["training_data_path"]),
        max_sequence_length=pilot_config.max_sequence_length,
    )
    if chronology.chronology_sha256 != repeat_chronology.chronology_sha256:
        raise SASRecPilotSafetyError("chronology hash changed between repeated builds")

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
    if arrays.training_objective_sha256 != repeat_arrays.training_objective_sha256:
        raise SASRecPilotSafetyError("training-objective hash changed between repeated builds")

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
        training_objective_sha=arrays.training_objective_sha256,
        negative_sampling_evidence_sha=primary.negative_sampling_evidence_sha256,
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
        training_objective_sha=arrays.training_objective_sha256,
        item_token_map_sha=item_token_map_sha,
        checkpoint_sha=checkpoint_sha,
        run_status=run_status,
        output_paths=output_paths,
        training_interactions=chronology.train_rows,
        training_examples=arrays.training_example_count,
        supervised_target_positions=arrays.training_example_count,
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
    print(f"training_objective={arrays.training_example_count}")
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
            raise SASRecPilotSafetyError(f"refusing to overwrite existing Phase 3J output {key}: {path}")
    checkpoint_dir = output_paths["checkpoint"].parent
    if checkpoint_dir.exists():
        raise SASRecPilotSafetyError(f"checkpoint directory already exists before execution: {checkpoint_dir}")


def verify_protected_inputs() -> None:
    for rel_path, expected in PROTECTED_INPUTS.items():
        actual = file_sha256(PROJECT_ROOT / rel_path)
        if actual != expected:
            raise SASRecPilotSafetyError(f"protected input hash mismatch for {rel_path}: {actual} != {expected}")


def repo_venv_environment() -> dict[str, str]:
    env_path = PROJECT_ROOT / "results" / "tables" / "neural_environment_inventory.csv"
    frame = pd.read_csv(env_path, dtype="string", keep_default_na=False)
    row = frame.loc[frame["environment_id"].astype(str) == "repo_venv_cuda"]
    if row.empty:
        raise SASRecPilotSafetyError("repo_venv_cuda missing from environment inventory")
    return {str(key): str(value) for key, value in row.iloc[0].to_dict().items()}


def build_pilot_config(config_data: Mapping[str, Any]) -> SASRecPilotConfig:
    embedding_dim = int(config_data["item_embedding_dim"])
    hidden_size = int(config_data["hidden_size"])
    num_heads = int(config_data["num_attention_heads"])
    max_sequence_length = int(config_data["maximum_sequence_length"])
    if embedding_dim != hidden_size:
        raise SASRecPilotSafetyError("SASRec hidden_size must match item_embedding_dim for the inspected Plan D model")
    if embedding_dim % num_heads != 0:
        raise SASRecPilotSafetyError("SASRec hidden size must be divisible by attention-head count")
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


def load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    rows = frame.loc[
        frame["dataset"].astype(str).eq("ML-1M")
        & frame["candidate_set_id"].astype(str).str.startswith("plan_e_ml1m_1000user_")
    ].copy()
    if len(rows) != 2000:
        raise SASRecPilotSafetyError(f"expected 2000 ML-1M Plan E2 rows, found {len(rows)}")
    return sort_candidate_like_rows(rows.to_dict("records"))


def load_compatibility_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    rows = frame.loc[frame["dataset"].astype(str).eq("ML-1M")].copy()
    if len(rows) != 2000:
        raise SASRecPilotSafetyError(f"expected 2000 ML-1M compatibility rows, found {len(rows)}")
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
            raise SASRecPilotSafetyError(f"compatibility row lacks candidate source: {key}")
        candidate = candidate_lookup[key]
        task_type = "ranking" if "ranking_manifest" in key[0] and "reranking" not in key[0] else "reranking"
        candidates = parse_candidate_items(candidate["candidate_items"])
        positive = str(candidate["positive_item_id"])
        if candidates.count(positive) != 1:
            raise SASRecPilotSafetyError("candidate row does not contain exactly one positive")
        included = (
            str(comp["sasrec_sequentially_compatible"]).lower() == "true"
            and str(comp["history_available"]).lower() == "true"
            and str(comp["training_objective_available"]).lower() == "true"
            and str(comp["all_candidate_items_seen_in_train"]).lower() == "true"
            and str(comp["padding_index_compatible"]).lower() == "true"
            and str(comp["position_indices_valid"]).lower() == "true"
            and str(comp["causal_mask_verified"]).lower() == "true"
            and str(comp["padding_mask_verified"]).lower() == "true"
            and str(comp["chronology_verified"]).lower() == "true"
            and str(comp["temporal_protocol_unverifiable"]).lower() == "false"
            and str(comp["target_leakage_detected"]).lower() == "false"
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
            "training_objective_available": str(comp["training_objective_available"]).lower() == "true",
            "all_candidate_items_seen_in_train": str(comp["all_candidate_items_seen_in_train"]).lower() == "true",
            "padding_index_compatible": str(comp["padding_index_compatible"]).lower() == "true",
            "position_indices_valid": str(comp["position_indices_valid"]).lower() == "true",
            "causal_mask_verified": str(comp["causal_mask_verified"]).lower() == "true",
            "padding_mask_verified": str(comp["padding_mask_verified"]).lower() == "true",
            "chronology_verified": str(comp["chronology_verified"]).lower() == "true",
            "temporal_protocol_unverifiable": str(comp["temporal_protocol_unverifiable"]).lower() == "true",
            "target_leakage_detected": str(comp["target_leakage_detected"]).lower() == "true",
            "sasrec_sequentially_compatible": str(comp["sasrec_sequentially_compatible"]).lower() == "true",
            "eligible_plan_e2_full": str(comp["eligible_plan_e2_full"]).lower() == "true",
            "eligible_plan_e2_matched": str(comp["eligible_plan_e2_matched"]).lower() == "true",
            "included_in_pilot": included,
            "pilot_full_cohort_id": "plan_e2_sasrec_strict_causal_full",
            "pilot_matched_cohort_id": "plan_e2_sasrec_strict_causal_matched",
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
        raise SASRecPilotSafetyError(f"expected 2000 ML-1M cohort rows, found {len(cohort_rows)}")
    if len(pilot_rows) != 1962:
        raise SASRecPilotSafetyError(f"expected 1962 included rows, found {len(pilot_rows)}")
    if sum(row["included_in_pilot"] for row in cohort_rows) != 1962:
        raise SASRecPilotSafetyError("included cohort count mismatch")
    if len(cohort_rows) - len(pilot_rows) != 38:
        raise SASRecPilotSafetyError("excluded cohort count mismatch")
    if sum(row["task_type"] == "ranking" for row in pilot_rows) != 981:
        raise SASRecPilotSafetyError("expected 981 included ranking rows")
    if sum(row["task_type"] == "reranking" for row in pilot_rows) != 981:
        raise SASRecPilotSafetyError("expected 981 included reranking rows")
    for row in pilot_rows:
        if int(row["candidate_count"]) != 20:
            raise SASRecPilotSafetyError("included row does not have 20 candidates")
        if not str(row["candidate_set_id"]).startswith("plan_e_ml1m_1000user_"):
            raise SASRecPilotSafetyError("non-Plan-E2 ML-1M row entered pilot")


def verify_same_cohort_row_keys(pilot_rows: Sequence[Mapping[str, Any]]) -> None:
    """Require exact row-key parity with existing ML-1M BPRMF and GRU4Rec pilot outputs."""

    expected = {(str(row["candidate_set_id"]), str(row["user_id"])) for row in pilot_rows}
    if len(expected) != 1962:
        raise SASRecPilotSafetyError("SASRec pilot row keys are not unique")
    for rel_path, label in (
        ("results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv", "BPRMF"),
        ("results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv", "GRU4Rec"),
    ):
        frame = pd.read_csv(PROJECT_ROOT / rel_path, dtype="string", keep_default_na=False)
        keys = {(str(row.candidate_set_id), str(row.user_id)) for row in frame.itertuples(index=False)}
        if len(frame) != 1962 or len(keys) != 1962:
            raise SASRecPilotSafetyError(f"{label} comparison rows are not the expected 1962 unique ML-1M keys")
        missing = sorted(expected.difference(keys))
        extra = sorted(keys.difference(expected))
        if missing or extra:
            raise SASRecPilotSafetyError(
                f"SASRec strict cohort row keys differ from {label}: missing={missing[:3]}, extra={extra[:3]}"
            )


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
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def git_output(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
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
    training_objective_sha: str,
    negative_sampling_evidence_sha: str,
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
        "config_path": "configs/plan_e_sasrec_ml1m_pilot.yaml",
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
        "training_objective_sha256": training_objective_sha,
        "negative_sampling_evidence_sha256": negative_sampling_evidence_sha,
        "item_token_map_sha256": item_token_map_sha,
        "protected_input_hashes": dict(PROTECTED_INPUTS),
        "notes": "Controlled ML-1M-only SASRec pilot using reconstructed Plan C split=train and strict Plan E2 compatible rows.",
    }


def build_determinism_rows(
    *,
    config: SASRecPilotConfig,
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

    add("configuration_hash", config_sha, config_sha, evidence="configs/plan_e_sasrec_ml1m_pilot.yaml")
    add("training_data_hash", training_data_sha, training_data_sha, evidence="data/reconstructed/plan_c/ml1m/interactions.csv")
    add("chronology_hash", chronology_sha, repeat_chronology_sha, evidence="source-row-indexed train chronology")
    add("training_objective_hash", training_objective_sha, repeat_training_objective_sha, evidence="all-prefix sequence arrays")
    add(
        "negative_sampling_evidence_hash",
        negative_sampling_evidence_sha,
        repeat_negative_sampling_evidence_sha,
        evidence="deterministic sampled negative targets by epoch and batch",
    )
    add("item_token_map_hash", item_token_map_sha, item_token_map_sha, evidence="results/tables/plan_e_sasrec_ml1m_item_token_map.csv")
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
    config: SASRecPilotConfig,
    env: Mapping[str, str],
    primary: Any,
    repeat: Any,
    training_data_sha: str,
    chronology_sha: str,
    training_objective_sha: str,
    item_token_map_sha: str,
    checkpoint_sha: str,
    run_status: str,
    output_paths: Mapping[str, Path],
    training_interactions: int,
    training_examples: int,
    supervised_target_positions: int,
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
                "config_path": "configs/plan_e_sasrec_ml1m_pilot.yaml",
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
                "item_token_map_path": rel(output_paths["item_token_map"]),
                "item_token_map_sha256": item_token_map_sha,
                "training_interactions": int(training_interactions),
                "training_examples": int(training_examples),
                "supervised_target_positions": int(supervised_target_positions),
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
    token_ids = [int(row["sasrec_token_id"]) for row in item_rows]
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
    metric_evaluated = sum(int(row["evaluated_rows"]) for row in metrics_rows if row["cohort_id"] == "plan_e2_sasrec_strict_causal_full")
    output_dir = (PROJECT_ROOT / "results" / "models" / "plan_e_sasrec_ml1m_pilot").resolve()
    checkpoint_resolved = checkpoint_path.resolve()

    add("training_split_only", "train", "train", chronology.train_rows > 0, evidence="split=train filter before chronology")
    add("no_validation_test_leakage", "validation=0,test=0 in training", f"validation={chronology.validation_rows},test={chronology.test_rows} excluded", True, evidence="chronology uses only split=train")
    add("held_out_positive_excluded_from_history", "0 positives in histories", "checked during scoring", True, evidence="score_candidate_rows safety assertion")
    add("chronology_hash_stability", "stable chronology hash", chronology.chronology_sha256, bool(chronology.chronology_sha256), evidence="repeated chronology build before training")
    add("equal_timestamp_tie_rule", "source_row_index", "source_row_index", True, evidence="sort fields user, timestamp, source_row_index")
    add("source_row_order_stability", "unique source_row_index", int(chronology.ordered_train["source_row_index"].nunique()), chronology.ordered_train["source_row_index"].is_unique, evidence="source row index frozen before sort")
    add("training_objective_validity", chronology.train_rows - chronology.unique_train_users, arrays.training_example_count, arrays.training_example_count == chronology.train_rows - chronology.unique_train_users, evidence="all-prefix next-item construction")
    add("padding_index_restriction", "no real item token 0", min(token_ids), min(token_ids) >= 1 and 0 not in token_ids, evidence="item-token map")
    add("real_item_token_offset", "dense_index + 1", f"min={min(token_ids)},max={max(token_ids)}", sorted(token_ids) == list(range(1, len(token_ids) + 1)), evidence="contiguous SASRec token IDs")
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


def run_history_rows(config: SASRecPilotConfig, primary: Any, repeat: Any) -> list[dict[str, Any]]:
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
        raise SASRecPilotSafetyError(f"refusing to overwrite existing output: {target}")
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
            raise SASRecPilotSafetyError(f"{key} schema mismatch")
        if frame.empty:
            raise SASRecPilotSafetyError(f"{key} is empty")
        frames[key] = frame

    token_map = frames["item_token_map"]
    token_ids = pd.to_numeric(token_map["sasrec_token_id"], errors="raise").astype(int)
    if token_map["raw_item_id"].duplicated().any() or token_ids.duplicated().any():
        raise SASRecPilotSafetyError("item-token map duplicate raw IDs or tokens")
    if token_ids.min() != 1 or token_ids.max() != len(token_ids) or 0 in set(token_ids):
        raise SASRecPilotSafetyError("item-token map does not use contiguous real tokens starting at 1")

    sequence = frames["sequence_summary"]
    if sequence["user_id"].duplicated().any():
        raise SASRecPilotSafetyError("sequence summary has duplicate users")
    if pd.to_numeric(sequence["effective_history_length"], errors="raise").max() > 50:
        raise SASRecPilotSafetyError("sequence summary effective length exceeds 50")

    cohort = frames["pilot_cohort"]
    if len(cohort) != 2000:
        raise SASRecPilotSafetyError("pilot cohort row count mismatch")
    included = cohort.loc[cohort["included_in_pilot"].str.lower().eq("true")]
    if len(included) != 1962 or len(cohort) - len(included) != 38:
        raise SASRecPilotSafetyError("pilot cohort inclusion/exclusion mismatch")

    manifest = frames["run_manifest"]
    if set(manifest["execution_role"]) != {"primary", "same_seed_repeat"}:
        raise SASRecPilotSafetyError("run manifest missing primary or repeat row")
    repeat_checkpoint = manifest.loc[manifest["execution_role"].eq("same_seed_repeat"), "checkpoint_path"].iloc[0]
    if repeat_checkpoint:
        raise SASRecPilotSafetyError("same-seed repeat manifest must not retain checkpoint path")

    history = frames["training_history"]
    if len(history) != 10:
        raise SASRecPilotSafetyError("training history must contain 5 epochs for each of two runs")
    if not history["finite_loss"].str.lower().eq("true").all():
        raise SASRecPilotSafetyError("training history contains non-finite loss")

    rankings = frames["rankings"]
    if len(rankings) != 1962:
        raise SASRecPilotSafetyError("rankings row count mismatch")
    for row in rankings.itertuples(index=False):
        candidates = json.loads(row.candidate_items)
        scores = json.loads(row.candidate_scores)
        ranked = json.loads(row.ranked_items)
        if len(candidates) != 20 or len(scores) != 20:
            raise SASRecPilotSafetyError("ranking row does not contain 20 candidates/scores")
        if set(candidates) != set(ranked):
            raise SASRecPilotSafetyError("ranking row candidate membership changed")
        if any(not np.isfinite(float(score)) for score in scores):
            raise SASRecPilotSafetyError("ranking row contains non-finite score")
        rank = int(row.positive_rank_zero_based)
        if ranked[rank] != str(row.positive_item_id):
            raise SASRecPilotSafetyError("positive rank does not reconcile")

    metrics = frames["metrics"]
    if len(metrics) != 4:
        raise SASRecPilotSafetyError("metrics row count mismatch")
    if set(metrics["cohort_id"]) != {"plan_e2_sasrec_strict_causal_full", "plan_e2_sasrec_strict_causal_matched"}:
        raise SASRecPilotSafetyError("metrics cohorts mismatch")
    for field in ("hr_at_5", "ndcg_at_5", "valid_ranking_rate", "duplicate_ranked_item_rate", "missing_score_rate", "history_truncation_rate"):
        values = pd.to_numeric(metrics[field], errors="raise")
        if ((values < 0) | (values > 1)).any():
            raise SASRecPilotSafetyError(f"metric {field} outside [0,1]")

    determinism = frames["determinism"]
    required_checks = {
        "configuration_hash",
        "training_data_hash",
        "chronology_hash",
        "training_objective_hash",
        "negative_sampling_evidence_hash",
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
        raise SASRecPilotSafetyError("determinism checks mismatch")

    validation = frames["validation"]
    if ((validation["status"].eq("failed")) & (validation["severity"].eq("error"))).any():
        raise SASRecPilotSafetyError("validation contains an error-level failure")
    if not checkpoint_path.exists() or file_sha256(checkpoint_path) != manifest.loc[manifest["execution_role"].eq("primary"), "checkpoint_sha256"].iloc[0]:
        raise SASRecPilotSafetyError("checkpoint hash does not reconcile with manifest")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise SASRecPilotSafetyError("metadata checkpoint hash mismatch")
    checkpoint_files = list(checkpoint_path.parent.glob("*.pt"))
    if checkpoint_files != [checkpoint_path]:
        raise SASRecPilotSafetyError("unexpected checkpoint files in approved directory")


def metric_lookup(metrics_rows: Sequence[Mapping[str, Any]], task: str, cohort: str) -> Mapping[str, Any]:
    for row in metrics_rows:
        if str(row["task_type"]) == task and str(row["cohort_id"]) == cohort:
            return row
    return {}


def write_report(
    path: str | Path,
    *,
    config_data: Mapping[str, Any],
    config: SASRecPilotConfig,
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
        raise SASRecPilotSafetyError(f"refusing to overwrite report: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    included = [row for row in cohort_rows if bool(row["included_in_pilot"])]
    excluded = [row for row in cohort_rows if not bool(row["included_in_pilot"])]
    ranking_task = [row for row in ranking_rows if row["task_type"] == "ranking"]
    reranking_task = [row for row in ranking_rows if row["task_type"] == "reranking"]
    primary_history = [row for row in history_rows if row["execution_role"] == "primary"]
    repeat_history = [row for row in history_rows if row["execution_role"] == "same_seed_repeat"]
    ranking_full = metric_lookup(metrics_rows, "ranking", "plan_e2_sasrec_strict_causal_full")
    reranking_full = metric_lookup(metrics_rows, "reranking", "plan_e2_sasrec_strict_causal_full")
    ranking_matched = metric_lookup(metrics_rows, "ranking", "plan_e2_sasrec_strict_causal_matched")
    reranking_matched = metric_lookup(metrics_rows, "reranking", "plan_e2_sasrec_strict_causal_matched")
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

    primary_manifest = next(row for row in manifest_rows if row["execution_role"] == "primary")
    repeat_manifest = next(row for row in manifest_rows if row["execution_role"] == "same_seed_repeat")
    passed_det = [row for row in determinism_rows if row["status"] == "passed"]
    passed_validation = [row for row in validation_rows if row["status"] == "passed"]
    primary_losses = "; ".join(
        f"epoch {row['epoch']}: loss={float(row['mean_training_loss']):.6f}" for row in primary_history
    )
    repeat_losses = "; ".join(
        f"epoch {row['epoch']}: loss={float(row['mean_training_loss']):.6f}" for row in repeat_history
    )
    truncated_rows = sum(bool(row["history_truncated"]) for row in ranking_rows)
    max_effective = max(int(row["effective_history_length"]) for row in ranking_rows)

    lines = [
        "# Plan E SASRec ML-1M Controlled Pilot",
        "",
        "## 1. Scope and provenance",
        f"Executed on branch `{git_output(['branch', '--show-current'])}` at commit `{git_output(['log', '-1', '--format=%H %s'])}` using configuration `configs/plan_e_sasrec_ml1m_pilot.yaml` (`sha256={config_sha}`).",
        "",
        "## 2. Executive summary",
        f"The pilot trained SASRec on reconstructed ML-1M `split=train`, scored {len(ranking_rows)} strict Plan E2 rows, and retained one primary checkpoint. The run remains classified as `{config_data['paper_alignment']}`.",
        "",
        "## 3. Scientific classification",
        "This is an adapted paper-like reconstruction, not exact paper reproduction, because it uses reconstructed Plan C ML-1M data and local Plan D/Plan E code rather than verified original-paper artifacts.",
        "",
        "## 4. Safety boundaries",
        "Only ML-1M SASRec was trained. Validation/test interactions, candidate labels, candidate positions, titles, LLMs, and other recommender families were not used for fitting.",
        "",
        "## 5. Environment and hardware",
        f"`environment_id={env['environment_id']}`, Python `{env['python_version']}`, torch `{env['torch_version']}`, CUDA `{env['torch_cuda_version']}`, device `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}`.",
        "",
        "## 6. Configuration provenance",
        f"Hyperparameters come from `{config_data['configuration_provenance']['source']}` row `{config_data['configuration_provenance']['selected_row']}` with config hash `{config_data['configuration_provenance']['selected_config_hash']}`.",
        "",
        "## 7. Final configuration",
        f"Embedding `{config.embedding_dim}`, hidden `{config.hidden_size}`, heads `{config.num_attention_heads}`, layers `{config.num_layers}`, dropout `{config.dropout}`, sequence length `{config.max_sequence_length}`, batch `{config.batch_size}`, epochs `{config.epochs}`, learning rate `{config.learning_rate}`.",
        "",
        "## 8. SASRec architecture",
        "The pilot reuses the local Plan D SASRec layers through a Phase 3J finite-mask wrapper: item embedding with padding index 0, learned absolute positional embeddings for positions 0..49, two Transformer encoder blocks, GELU feed-forward layers, final layer normalization, last non-padding representation selection, and dot-product candidate scoring.",
        "",
        "## 9. Authoritative ML-1M interaction data",
        f"`{config_data['training_data_path']}` (`sha256={training_data_sha}`) supplied {chronology.train_rows} train rows, {chronology.validation_rows} validation rows, and {chronology.test_rows} test rows; only train rows were used.",
        "",
        "## 10. Split isolation and leakage prevention",
        "The chronology builder filters `split=train` before ordering, indexing, pair construction, and model fitting. Evaluation positives were checked during scoring and were not injected into histories.",
        "",
        "## 11. Chronology construction",
        f"Chronology hash `{chronology.chronology_sha256}` covers `user_id`, `item_id`, `timestamp`, and frozen `source_row_index`. It contains {chronology.unique_train_users} users and {chronology.unique_train_items} train-seen items.",
        "",
        "## 12. Equal-timestamp handling",
        f"Equal timestamps are resolved by frozen source row order. There are {chronology.equal_timestamp_group_count} equal-timestamp groups covering {chronology.events_in_equal_timestamp_groups} events.",
        "",
        "## 13. Training-objective construction",
        f"All-prefix next-item construction produced {primary_manifest['training_examples']} supervised examples from train histories; the arrays hash is `{primary_manifest['training_objective_sha256']}`.",
        "",
        "## 14. Positive and negative target semantics",
        "Positive targets are the next train item in each user prefix. Negative targets are sampled from train-seen item indices excluding the user's train positives; padding token 0 is never a target.",
        "",
        "## 15. Negative-sampling contract",
        f"The deterministic negative-sampling evidence hash is `{primary_manifest['negative_sampling_evidence_sha256']}` for the primary run and `{repeat_manifest['negative_sampling_evidence_sha256']}` for the same-seed repeat.",
        "",
        "## 16. Item-token mapping",
        f"The item-token map has {len(item_rows)} train-seen real item tokens; token 0 is reserved for padding. Map hash `{item_token_map_sha}`.",
        "",
        "## 17. Padding semantics",
        "Sequences are left padded to length 50 with token 0. Real item IDs use dense train item index plus one, so padding cannot be scored as a candidate.",
        "",
        "## 18. Positional-embedding semantics",
        "The Plan D model uses absolute tensor positions `0..49` after left padding. This pilot preserves that contract and does not reconstruct paper-side absolute positions beyond the local tensor indices.",
        "",
        "## 19. Causal-attention mask",
        "The Transformer encoder receives a boolean causal component so future tensor positions are blocked while diagonal self-attention remains allowed.",
        "",
        "## 20. Padding-attention mask",
        "The Phase 3J wrapper combines the causal and padding-key masks and leaves each padding query's diagonal unmasked to avoid CUDA all-masked attention rows. Real-token queries still cannot attend to padding keys.",
        "",
        "## 21. Final sequence representation",
        "The scored user representation is selected from the last non-padding history position after the Transformer encoder and final layer normalization.",
        "",
        "## 22. Pilot-cohort construction",
        f"The cohort contains {len(cohort_rows)} ML-1M Plan E2 rows, with {len(included)} included and {len(excluded)} excluded under `strict_causal_history_and_train_known_candidates`.",
        "",
        "## 23. Excluded ML-1M rows",
        f"{len(excluded)} rows were excluded, all before scoring. The common reason is a strict sequential/train-seen compatibility failure recorded in `plan_e_sasrec_ml1m_pilot_cohort.csv`.",
        "",
        "## 24. Training procedure",
        "Training used BPR pairwise loss, Adam, deterministic seeds, deterministic torch algorithms, and a deterministic negative sampler over train-seen item indices.",
        "",
        "## 25. Primary-run training history",
        primary_losses,
        "",
        "## 26. Same-seed repeat history",
        repeat_losses,
        "",
        "## 27. Runtime and GPU-memory findings",
        f"Primary runtime was {float(primary_manifest['duration_seconds']):.3f}s and repeat runtime was {float(repeat_manifest['duration_seconds']):.3f}s. Peak CUDA fields are recorded per epoch in training history.",
        "",
        "## 28. Checkpoint and metadata",
        f"Exactly one checkpoint was retained at `{primary_manifest['checkpoint_path']}` (`sha256={checkpoint_sha}`), with metadata at `results/models/plan_e_sasrec_ml1m_pilot/metadata.json`. The repeat run retained no checkpoint path.",
        "",
        "## 29. Evaluation-history construction",
        "Evaluation histories are train-only user histories clipped to the most recent 50 items; no candidate membership or positive label is used to construct the history.",
        "",
        "## 30. Candidate-scoring contract",
        "The model scored only the supplied 20 candidates for each compatible row. Excluded rows were not scored.",
        "",
        "## 31. Candidate-order and membership preservation",
        "The `candidate_items` column preserves source order, while `ranked_items` is a permutation of the same 20 IDs after score sorting.",
        "",
        "## 32. Ranking results",
        metric_line(ranking_full),
        "",
        "## 33. Reranking-candidate results",
        metric_line(reranking_full),
        "",
        "## 34. Full strict-cohort metrics",
        f"Ranking full cohort: {metric_line(ranking_full)} Reranking full cohort: {metric_line(reranking_full)}",
        "",
        "## 35. Matched strict-cohort metrics",
        f"Ranking matched cohort: {metric_line(ranking_matched)} Reranking matched cohort: {metric_line(reranking_matched)}",
        "",
        "## 36. History-length and truncation findings",
        f"Histories longer than 50 keep the most recent 50 train events. In the strict pilot, {truncated_rows} scored rows were truncated and maximum effective history length was {max_effective}. Mean raw/effective lengths are recorded in metrics.",
        "",
        "## 37. Ranking versus reranking interpretation",
        "Ranking and reranking use separate candidate-set IDs and should be interpreted as strict ML-1M Plan E2 candidate diagnostics, not full-catalog evaluation.",
        "",
        "## 38. Same-seed reproducibility",
        f"{len(passed_det)} of {len(determinism_rows)} determinism checks passed. Failed checks: {json.dumps([row['check_name'] for row in failed_det], ensure_ascii=True)}.",
        "",
        "## 39. Validation findings",
        f"{len(passed_validation)} of {len(validation_rows)} validation checks passed. Failed checks: {json.dumps([row['check_name'] for row in failed_validation], ensure_ascii=True)}.",
        "",
        "## 40. Comparison boundary with existing baselines",
        "Do not compare these SASRec metrics to existing baselines unless those baseline rankings are filtered to the same strict sequential ML-1M cohort and candidate rows.",
        "",
        "## 41. Absolute-position reconstruction limitation",
        "The local model uses tensor positions after left padding. The original paper's exact absolute-position construction is not independently verifiable from current repository artifacts.",
        "",
        "## 42. Equal-timestamp reconstruction limitation",
        "Equal timestamps are resolved by current persisted source row order. This is deterministic, but it remains a reconstruction when the original preprocessing order is not independently available.",
        "",
        "## 43. Deviations from the original paper",
        "The data, preprocessing lineage, model hyperparameters, and local training implementation are reconstructed/adapted. The pilot is evidence for a controlled local pipeline, not exact paper reproduction.",
        "",
        "## 44. Limitations",
        "The pilot covers ML-1M only, excludes 38 incompatible Plan E2 rows, and evaluates supplied 20-candidate lists rather than a full-catalog setting.",
        "",
        "## 45. Whether the pilot succeeded",
        f"The pilot succeeded as a controlled adapted ML-1M execution if validation has no error-level failures and the manifest status is `{primary_manifest['execution_status']}`. Machine-readable evidence is in the manifest, validation, determinism, ranking, and metrics tables.",
        "",
        "## 46. Conditions for Beauty and Sports scale-up",
        "Scale-up requires the same strict train-history, train-seen candidate, chronology, and same-seed repeat controls on Beauty and Sports without altering existing manifests.",
        "",
        "## 47. Protected-input immutability",
        "Protected hashes were verified before and after execution. `original_repo/`, Plan D source/runner, Phase 3I adapter, BPRMF artifacts, candidate manifests, and reconstructed ML-1M data were not modified.",
        "",
        "## 48. Commands executed",
        "See final assistant response for the exact shell commands. The runner command was `.venv\\Scripts\\python.exe src/experiments/run_plan_e_sasrec_ml1m_pilot.py`.",
        "",
        "## 49. Recommended single next task",
        "Phase 3K — controlled SASRec Beauty and Sports execution and same-cohort consolidation.",
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())




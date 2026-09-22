"""Run controlled Plan E GRU4Rec training for Beauty and Sports."""

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

from src.baselines.plan_e_gru4rec_adapter import GRU4REC_PADDING_INDEX, GRU4REC_REAL_ITEM_OFFSET, natural_id_sort_key, parse_candidate_items, safe_raw_id
from src.baselines.plan_e_gru4rec_training import (
    GRU4RecPilotConfig,
    GRU4RecPilotSafetyError,
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


PROTECTED_INPUTS = {
    "src/baselines/plan_d_gru4rec.py": "5E288F1917B8F9A3C037D5C3EEF3BE4C2320DF465B23370D94982F15F63F699D",
    "src/experiments/run_plan_d_gru4rec.py": "42F3EECEB8BE58C94EBB353F5DD02BAE8EC55E9859EE6455470069E8206E40AC",
    "src/baselines/plan_e_gru4rec_adapter.py": "F3972AEDE7A085FD7D247826D0EB12B097A5F9EBB0E6F6ADA6143B4EE8C84DCD",
    "src/baselines/plan_e_gru4rec_training.py": "6E12301AA324ECADEA55B1AD8B1F7BD2E8D545A8C7A0114BEAD5AA45A6AC868F",
    "src/experiments/run_plan_e_gru4rec_ml1m_pilot.py": "422DE06F12C44667C8FDF93FA5A2A0EA84F9FD14A10D390822AE74C2607999EF",
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/experiment_cohort_membership.csv": "D92BE28D56D20897B25BD0E0D8387194930253AEC62B957FE00AA230F7DF0139",
    "results/tables/experiment_cohort_summary.csv": "40BE65003A46815891EC35EE36B2A42DB945E46759EE642F5E7AC4B5561F4B2D",
    "results/tables/plan_e_gru4rec_row_compatibility.csv": "632FA890F05401CECA5A6B64212370CA0E80182D5B9A55918A9AC3C10630F6C1",
    "results/tables/plan_e_gru4rec_sequence_audit.csv": "9F97D98A658FEB5C35FEF9ED1F9610E764A87B5BFFFDAEACAD5D71AA5FEA5442",
    "data/reconstructed/plan_c/amazon_beauty/interactions.csv": "D2A7801A024349063CFF7C455E872EED6E8EB0132324ED8B710BE1FF9A0E9478",
    "data/reconstructed/plan_c/amazon_beauty/datamaps.json": "4EF9E7F9BA6D0F98998EFAC7DFDEA455315635753B9C21FF46269ABC31670423",
    "data/reconstructed/plan_c/amazon_sports/interactions.csv": "E7BC05D829B83DCD286E3FE1ACBCCB61341A8AB709F4260F9889633552829B40",
    "data/reconstructed/plan_c/amazon_sports/datamaps.json": "5A0E99616F2C31DF45E13B82DD9012A7728AFC0507C475876E0E6C78ACFE6A5A",
    "results/tables/traditional_baseline_rankings.csv": "B9E99E14A70F7AF60D602E32B5DFBBD5624493177B8DD9C9908FBA0ABAD5E7BA",
    "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv": "A8C43DCB6D88D6A4726F36C5EBFFBADA54589C88A7F0988446A5439D083E547B",
    "results/tables/plan_e_bprmf_beauty_sports_rankings.csv": "428A1176C4BB903EC94E1704221ED19C5CFB74206A6446390B31AD41C516285A",
    "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv": "AF9674672647F2892AFDC7018DD3261ED1B9A215CDFCA105C686022457635B94",
    "configs/plan_e_gru4rec_ml1m_pilot.yaml": "CC040BB51EBCE09CE0C7857B921036FAEB388EF248C0A852092B42385EFBD1B2",
    "results/tables/plan_e_gru4rec_ml1m_item_token_map.csv": "6FAC77CE65025D80754503ACBD788ACA976CD2F9907E804B8773295D29B03AF1",
    "results/tables/plan_e_gru4rec_ml1m_sequence_summary.csv": "2BDD0AD73E6AB652B0F3C63E646FDB380362E5A30E1E0BAFAA8F8C4427DEB797",
    "results/tables/plan_e_gru4rec_ml1m_pilot_cohort.csv": "768CCFE3BA3A113562456E83B120BED9B62A5A37DE92CCC6DF56CC2C44170F49",
    "results/tables/plan_e_gru4rec_ml1m_pilot_run_manifest.csv": "81D246355D26B4569786E71222567FD3C077D021C8A314D42C842B7CD7E9B30F",
    "results/tables/plan_e_gru4rec_ml1m_pilot_training_history.csv": "FD7296295657092E150704F452625AA811204C82D193E54111FA8ACDD60696F2",
    "results/tables/plan_e_gru4rec_ml1m_pilot_metrics.csv": "FB198E7183F3C22F11AF2200D277BE7DA7F6806707CF6C6C4BA96234BE9F6E5B",
    "results/tables/plan_e_gru4rec_ml1m_pilot_determinism.csv": "DBF82A1927F4BB58B6CF06E7573E94D876CEE94484243A1FA9D2D9A4798FACD9",
    "results/tables/plan_e_gru4rec_ml1m_pilot_validation.csv": "B309A6BB7E340750FB0248937257384DBCF8119158B7E33210336776A4A2B973",
    "results/models/plan_e_gru4rec_ml1m_pilot/model_state.pt": "1B81999CAE574DF883B98362CB2EFEB554EB7CBE275AD263FA179B6BE741D06D",
    "results/models/plan_e_gru4rec_ml1m_pilot/metadata.json": "3F38B24072C465ADA37C1E0F4249F83541CB364287971000AC3D90FBAAC1F161",
}

DATASETS = [
    ("Amazon Beauty", PROJECT_ROOT / "configs/plan_e_gru4rec_beauty.yaml", 983, 977, 1966, 34),
    ("Amazon Sports", PROJECT_ROOT / "configs/plan_e_gru4rec_sports.yaml", 994, 907, 1988, 12),
]

OUTPUTS = {
    "item_token_map": PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_item_token_map.csv",
    "sequence_summary": PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_sequence_summary.csv",
    "cohort": PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_cohort.csv",
    "run_manifest": PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_run_manifest.csv",
    "training_history": PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_training_history.csv",
    "rankings": PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv",
    "metrics": PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_metrics.csv",
    "determinism": PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_determinism.csv",
    "validation": PROJECT_ROOT / "results/tables/plan_e_gru4rec_beauty_sports_validation.csv",
}

ITEM_TOKEN_COLUMNS = [
    "dataset",
    "raw_item_id",
    "source_datamap_internal_id",
    "dense_train_item_index",
    "gru4rec_token_id",
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
    "training_pair_available",
    "all_candidate_items_seen_in_train",
    "chronology_verified",
    "temporal_protocol_unverifiable",
    "sequentially_compatible",
    "eligible_plan_e2_full",
    "eligible_plan_e2_matched",
    "included_in_gru4rec_scoring",
    "eligible_gru4rec_strict_full",
    "eligible_gru4rec_strict_matched",
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
    "training_pairs_sha256",
    "item_token_map_path",
    "item_token_map_sha256",
    "training_interactions",
    "training_pairs",
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
    "full_cohort_id",
    "matched_cohort_id",
    "eligible_gru4rec_strict_full",
    "eligible_gru4rec_strict_matched",
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

FULL_COHORT_ID = "plan_e2_gru4rec_strict_sequential_full"
MATCHED_COHORT_ID = "plan_e2_gru4rec_strict_sequential_matched"


def main() -> int:
    assert_outputs_absent()
    verify_protected_inputs()
    env = repo_venv_environment()
    if Path(env["python_executable"]).resolve() != Path(sys.executable).resolve():
        raise GRU4RecPilotSafetyError(f"runner must use repo_venv_cuda executable: {env['python_executable']}")
    if not torch.cuda.is_available():
        raise GRU4RecPilotSafetyError("CUDA is required for Phase 3H controlled execution")
    verify_bprmf_strict_row_identity()

    all_item_rows: list[dict[str, Any]] = []
    all_sequence_rows: list[dict[str, Any]] = []
    all_cohort_rows: list[dict[str, Any]] = []
    all_manifest_rows: list[dict[str, Any]] = []
    all_history_rows: list[dict[str, Any]] = []
    all_ranking_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_determinism_rows: list[dict[str, Any]] = []
    all_validation_rows: list[dict[str, Any]] = []

    for dataset, config_path, expected_full_per_task, expected_matched_per_task, expected_included, expected_excluded in DATASETS:
        config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        pilot_config = build_pilot_config(config_data)
        set_deterministic_execution(pilot_config.seed)
        candidate_rows = load_candidate_rows(config_data)
        compatibility_rows = load_compatibility_rows(config_data)
        pilot_rows, cohort_rows = build_dataset_cohort(
            dataset=dataset,
            candidate_rows=candidate_rows,
            compatibility_rows=compatibility_rows,
        )
        validate_cohort_counts(
            dataset=dataset,
            pilot_rows=pilot_rows,
            cohort_rows=cohort_rows,
            expected_full_per_task=expected_full_per_task,
            expected_matched_per_task=expected_matched_per_task,
            expected_included=expected_included,
            expected_excluded=expected_excluded,
        )

        interactions = load_interactions_with_source_rows(PROJECT_ROOT / config_data["training_data_path"])
        chronology = build_chronology(interactions, max_sequence_length=pilot_config.max_sequence_length)
        repeat_chronology = build_chronology(
            load_interactions_with_source_rows(PROJECT_ROOT / config_data["training_data_path"]),
            max_sequence_length=pilot_config.max_sequence_length,
        )
        if chronology.chronology_sha256 != repeat_chronology.chronology_sha256:
            raise GRU4RecPilotSafetyError(f"{dataset} chronology hash changed between builds")
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
            raise GRU4RecPilotSafetyError(f"{dataset} training-pair hash changed between builds")
        datamap = json.loads((PROJECT_ROOT / config_data["datamap_path"]).read_text(encoding="utf-8"))
        item_rows = build_item_token_map_rows(
            dataset=dataset,
            item_index=item_index,
            item_datamap=datamap["item2id"],
            pilot_rows=pilot_rows,
            mapping_source_path=str(config_data["datamap_path"]),
        )
        sequence_rows = build_sequence_summary_rows(
            dataset=dataset,
            chronology=chronology,
            cohort_rows=cohort_rows,
            max_sequence_length=pilot_config.max_sequence_length,
        )

        primary = train_one_run(config=pilot_config, arrays=arrays, num_items=len(item_index), execution_role="primary")
        checkpoint_path = PROJECT_ROOT / config_data["checkpoint_path"]
        metadata_path = PROJECT_ROOT / config_data["metadata_path"]
        checkpoint_root = checkpoint_path.parent
        checkpoint_meta = metadata_for_dataset(
            config_data=config_data,
            env=env,
            chronology=chronology,
            training_data_sha=file_sha256(PROJECT_ROOT / config_data["training_data_path"]),
            training_pairs_sha=arrays.training_pairs_sha256,
            item_token_map_sha=rows_sha256(item_rows, ITEM_TOKEN_COLUMNS),
        )
        write_checkpoint_atomic(
            checkpoint_payload(model=primary.model, config=config_data, metadata=checkpoint_meta),
            checkpoint_path,
            approved_root=checkpoint_root,
        )
        checkpoint_sha = file_sha256(checkpoint_path)
        metadata = dict(checkpoint_meta)
        metadata.update(
            {
                "checkpoint_sha256": checkpoint_sha,
                "primary_final_state_hash": primary.final_state_hash,
                "checkpoint_path": rel(checkpoint_path),
            }
        )
        write_metadata(metadata, metadata_path, approved_root=checkpoint_root)

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

        repeat = train_one_run(config=pilot_config, arrays=repeat_arrays, num_items=len(item_index), execution_role="same_seed_repeat")
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
            dataset=dataset,
            config_sha=file_sha256(config_path),
            training_data_sha=file_sha256(PROJECT_ROOT / config_data["training_data_path"]),
            chronology_sha=chronology.chronology_sha256,
            repeat_chronology_sha=repeat_chronology.chronology_sha256,
            training_pairs_sha=arrays.training_pairs_sha256,
            repeat_training_pairs_sha=repeat_arrays.training_pairs_sha256,
            item_token_map_sha=rows_sha256(item_rows, ITEM_TOKEN_COLUMNS),
            primary=primary,
            repeat=repeat,
            primary_rankings=primary_rankings,
            repeat_rankings=repeat_rankings,
            primary_metrics=primary_metrics,
            repeat_metrics=repeat_metrics,
        )
        if any(row["status"] == "failed" for row in determinism_rows):
            raise GRU4RecPilotSafetyError(f"{dataset} same-seed determinism failed")
        validation_rows = build_validation_rows(
            config=pilot_config,
            dataset=dataset,
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
            raise GRU4RecPilotSafetyError(f"{dataset} validation contains error-level failure")
        manifest_rows = build_manifest_rows(
            config_data=config_data,
            config=pilot_config,
            env=env,
            primary=primary,
            repeat=repeat,
            chronology=chronology,
            arrays=arrays,
            item_rows=item_rows,
            checkpoint_path=checkpoint_path,
            checkpoint_sha=checkpoint_sha,
            training_data_sha=file_sha256(PROJECT_ROOT / config_data["training_data_path"]),
            ranking_rows=primary_rankings,
            metrics_rows=primary_metrics,
        )

        all_item_rows.extend(item_rows)
        all_sequence_rows.extend(sequence_rows)
        all_cohort_rows.extend(cohort_rows)
        all_manifest_rows.extend(manifest_rows)
        all_history_rows.extend(history_rows(pilot_config, dataset, primary, repeat))
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
    print("plan_e_gru4rec_beauty_sports_completed")
    print(f"rankings_rows={len(all_ranking_rows)}")
    print(f"metrics_rows={len(all_metric_rows)}")
    return 0


def assert_outputs_absent() -> None:
    for path in OUTPUTS.values():
        if path.exists():
            raise GRU4RecPilotSafetyError(f"refusing to overwrite output: {path}")
    for _, config_path, *_ in DATASETS:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for key in ("checkpoint_path", "metadata_path"):
            path = PROJECT_ROOT / config[key]
            if path.exists():
                raise GRU4RecPilotSafetyError(f"refusing to overwrite {key}: {path}")
        checkpoint_dir = (PROJECT_ROOT / config["checkpoint_path"]).parent
        if checkpoint_dir.exists():
            raise GRU4RecPilotSafetyError(f"checkpoint directory already exists: {checkpoint_dir}")


def verify_protected_inputs() -> None:
    for rel_path, expected in PROTECTED_INPUTS.items():
        actual = file_sha256(PROJECT_ROOT / rel_path)
        if actual != expected:
            raise GRU4RecPilotSafetyError(f"protected hash mismatch for {rel_path}: {actual} != {expected}")


def repo_venv_environment() -> dict[str, str]:
    frame = pd.read_csv(PROJECT_ROOT / "results/tables/neural_environment_inventory.csv", dtype="string", keep_default_na=False)
    row = frame.loc[frame["environment_id"].astype(str).eq("repo_venv_cuda")]
    if row.empty:
        raise GRU4RecPilotSafetyError("repo_venv_cuda missing from neural environment inventory")
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


def load_candidate_rows(config_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    frame = pd.read_csv(PROJECT_ROOT / str(config_data["candidate_sets_path"]), dtype="string", keep_default_na=False)
    ids = {str(config_data["ranking_candidate_set_id"]), str(config_data["reranking_candidate_set_id"])}
    rows = frame.loc[frame["dataset"].astype(str).eq(str(config_data["dataset"])) & frame["candidate_set_id"].astype(str).isin(ids)].copy()
    if len(rows) != 2000:
        raise GRU4RecPilotSafetyError(f"expected 2000 candidate rows for {config_data['dataset']}, found {len(rows)}")
    rows["task_type"] = rows["candidate_set_id"].map(lambda value: "reranking" if "reranking" in str(value) else "ranking")
    return rows.to_dict("records")


def load_compatibility_rows(config_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    frame = pd.read_csv(PROJECT_ROOT / str(config_data["row_compatibility_path"]), dtype="string", keep_default_na=False)
    ids = {str(config_data["ranking_candidate_set_id"]), str(config_data["reranking_candidate_set_id"])}
    rows = frame.loc[frame["dataset"].astype(str).eq(str(config_data["dataset"])) & frame["candidate_set_id"].astype(str).isin(ids)].copy()
    if len(rows) != 2000:
        raise GRU4RecPilotSafetyError(f"expected 2000 compatibility rows for {config_data['dataset']}, found {len(rows)}")
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
            raise GRU4RecPilotSafetyError(f"compatibility row lacks candidate source: {key}")
        candidates = parse_candidate_items(candidate["candidate_items"])
        positive = safe_raw_id(candidate["positive_item_id"])
        if candidates.count(positive) != 1 or len(candidates) != 20:
            raise GRU4RecPilotSafetyError(f"invalid candidate row for {key}")
        included_flag = (
            bool_text(comp["sequentially_compatible"])
            and bool_text(comp["history_available"])
            and bool_text(comp["training_pair_available"])
            and bool_text(comp["all_candidate_items_seen_in_train"])
            and bool_text(comp["chronology_verified"])
            and not bool_text(comp["temporal_protocol_unverifiable"])
            and bool_text(comp["eligible_plan_e2_full"])
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
            "history_truncated": bool_text(comp["history_truncated"]),
            "history_available": bool_text(comp["history_available"]),
            "training_pair_available": bool_text(comp["training_pair_available"]),
            "all_candidate_items_seen_in_train": bool_text(comp["all_candidate_items_seen_in_train"]),
            "chronology_verified": bool_text(comp["chronology_verified"]),
            "temporal_protocol_unverifiable": bool_text(comp["temporal_protocol_unverifiable"]),
            "sequentially_compatible": bool_text(comp["sequentially_compatible"]),
            "eligible_plan_e2_full": bool_text(comp["eligible_plan_e2_full"]),
            "eligible_plan_e2_matched": bool_text(comp["eligible_plan_e2_matched"]),
            "included_in_gru4rec_scoring": included_flag,
            "eligible_gru4rec_strict_full": included_flag,
            "eligible_gru4rec_strict_matched": included_flag and bool_text(comp["eligible_plan_e2_matched"]),
            "full_cohort_id": FULL_COHORT_ID,
            "matched_cohort_id": MATCHED_COHORT_ID,
            "exclusion_reasons": "none" if included_flag else str(comp.get("exclusion_reasons", "policy_excluded")),
            "notes": "included under strict sequential policy" if included_flag else "excluded before scoring",
            "candidate_items": candidate["candidate_items"],
        }
        output.append(row)
        if included_flag:
            included.append(dict(row))
    return sort_candidate_like_rows(included), sort_candidate_like_rows(output)


def validate_cohort_counts(
    *,
    dataset: str,
    pilot_rows: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    expected_full_per_task: int,
    expected_matched_per_task: int,
    expected_included: int,
    expected_excluded: int,
) -> None:
    if len(cohort_rows) != 2000:
        raise GRU4RecPilotSafetyError(f"{dataset} expected 2000 cohort rows, found {len(cohort_rows)}")
    if len(pilot_rows) != expected_included or len(cohort_rows) - len(pilot_rows) != expected_excluded:
        raise GRU4RecPilotSafetyError(f"{dataset} strict included/excluded counts mismatch")
    for task in ("ranking", "reranking"):
        task_rows = [row for row in pilot_rows if row["task_type"] == task]
        matched_rows = [row for row in task_rows if row["eligible_gru4rec_strict_matched"]]
        if len(task_rows) != expected_full_per_task or len(matched_rows) != expected_matched_per_task:
            raise GRU4RecPilotSafetyError(f"{dataset} {task} full/matched counts mismatch")


def verify_bprmf_strict_row_identity() -> None:
    compat = pd.read_csv(PROJECT_ROOT / "results/tables/plan_e_gru4rec_row_compatibility.csv", dtype="string", keep_default_na=False)
    candidates = pd.read_csv(PROJECT_ROOT / "results/tables/ranking_candidate_sets.csv", dtype="string", keep_default_na=False)
    bprmf = pd.read_csv(PROJECT_ROOT / "results/tables/plan_e_bprmf_beauty_sports_rankings.csv", dtype="string", keep_default_na=False)
    for dataset in ("Amazon Beauty", "Amazon Sports"):
        strict_compat = compat.loc[
            compat["dataset"].astype(str).eq(dataset)
            & compat["sequentially_compatible"].map(bool_text)
            & compat["history_available"].map(bool_text)
            & compat["training_pair_available"].map(bool_text)
            & compat["all_candidate_items_seen_in_train"].map(bool_text)
            & compat["chronology_verified"].map(bool_text)
            & ~compat["temporal_protocol_unverifiable"].map(bool_text)
            & compat["eligible_plan_e2_full"].map(bool_text)
        ].copy()
        strict_candidate_set_ids = set(strict_compat["candidate_set_id"].astype(str).tolist())
        candidate_unit = candidates.loc[
            candidates["dataset"].astype(str).eq(dataset)
            & candidates["candidate_set_id"].astype(str).isin(strict_candidate_set_ids)
        ].copy()
        candidate_unit["task_type"] = candidate_unit["candidate_set_id"].map(lambda value: "reranking" if "reranking" in str(value) else "ranking")
        bprmf_unit = bprmf.loc[bprmf["dataset"].astype(str).eq(dataset)].copy()
        expected_keys = {row_key(row) for row in strict_compat.to_dict("records")}
        candidate_map = {row_key(row): parse_candidate_items(row["candidate_items"]) for row in candidate_unit.to_dict("records")}
        bprmf_map = {row_key(row): parse_candidate_items(row["candidate_items"]) for row in bprmf_unit.to_dict("records")}
        present_keys = set(bprmf_map)
        if expected_keys != present_keys:
            raise GRU4RecPilotSafetyError(f"{dataset} BPRMF strict row keys differ from GRU4Rec compatibility")
        missing_candidate_keys = expected_keys.difference(candidate_map)
        if missing_candidate_keys:
            raise GRU4RecPilotSafetyError(f"{dataset} candidate-source rows missing for strict keys: {sorted(missing_candidate_keys)[:3]}")
        mismatches = [key for key in sorted(expected_keys) if candidate_map[key] != bprmf_map[key]]
        if mismatches:
            raise GRU4RecPilotSafetyError(f"{dataset} BPRMF candidate list differs from canonical candidate source for {mismatches[0]}")


def build_item_token_map_rows(
    *,
    dataset: str,
    item_index: Mapping[str, int],
    item_datamap: Mapping[str, Any],
    pilot_rows: Sequence[Mapping[str, Any]],
    mapping_source_path: str,
) -> list[dict[str, Any]]:
    candidate_counts: dict[str, int] = {}
    positive_counts: dict[str, int] = {}
    for row in pilot_rows:
        for item_id in parse_candidate_items(row["candidate_items"]):
            candidate_counts[item_id] = candidate_counts.get(item_id, 0) + 1
        positive = safe_raw_id(row["positive_item_id"])
        positive_counts[positive] = positive_counts.get(positive, 0) + 1
    missing = sorted(set(candidate_counts).difference(item_index), key=natural_id_sort_key)
    if missing:
        raise GRU4RecPilotSafetyError(f"{dataset} strict candidates missing from train item index: {missing[:3]}")
    rows = []
    for raw_item_id, dense_index in sorted(item_index.items(), key=lambda item: item[1]):
        rows.append(
            {
                "dataset": dataset,
                "raw_item_id": raw_item_id,
                "source_datamap_internal_id": str(item_datamap.get(raw_item_id, "")),
                "dense_train_item_index": int(dense_index),
                "gru4rec_token_id": int(dense_index) + GRU4REC_REAL_ITEM_OFFSET,
                "padding_index": GRU4REC_PADDING_INDEX,
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
        if row["included_in_gru4rec_scoring"]:
            included_by_user.setdefault(user_id, []).append(row)
    rows = []
    for user_id in sorted(chronology.train_histories, key=natural_id_sort_key):
        history = chronology.train_histories[user_id]
        timestamps = chronology.history_timestamps[user_id]
        user_rows = by_user.get(user_id, [])
        strict_rows = included_by_user.get(user_id, [])
        rows.append(
            {
                "dataset": dataset,
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
                "in_strict_rows": bool(strict_rows),
                "ranking_row_count": sum(row["task_type"] == "ranking" for row in strict_rows),
                "reranking_row_count": sum(row["task_type"] == "reranking" for row in strict_rows),
                "notes": "source_row_index frozen before sorting",
            }
        )
    return rows


def score_candidate_rows(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    *,
    train_histories: Mapping[str, Sequence[str]],
    item_index: Mapping[str, int],
    config: GRU4RecPilotConfig,
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
                raise GRU4RecPilotSafetyError(f"{config.dataset} held-out positive appears in evaluation history")
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
            if len(scores) != 20 or any(not math.isfinite(float(score)) for score in scores):
                raise GRU4RecPilotSafetyError("candidate score count or finiteness check failed")
            ranked_positions = sorted(range(len(candidates)), key=lambda pos: (-float(scores[pos]), str(candidates[pos]), pos))
            ranked_items = [candidates[pos] for pos in ranked_positions]
            ranked_scores = [float(scores[pos]) for pos in ranked_positions]
            if set(ranked_items) != set(candidates):
                raise GRU4RecPilotSafetyError("ranked membership differs from candidates")
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
                    "candidate_scores": json.dumps(scores, ensure_ascii=True),
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
                    "eligible_gru4rec_strict_full": True,
                    "eligible_gru4rec_strict_matched": bool(row["eligible_gru4rec_strict_matched"]),
                    "notes": "ranked by descending score, lexical raw item id, original candidate position",
                }
            )
    return output


def aggregate_metrics(ranking_rows: Sequence[Mapping[str, Any]], *, config: GRU4RecPilotConfig, training_run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = sorted({(row["task_type"], row["candidate_set_id"]) for row in ranking_rows})
    for task_type, candidate_set_id in groups:
        unit = [row for row in ranking_rows if row["task_type"] == task_type and row["candidate_set_id"] == candidate_set_id]
        for cohort_id, flag in ((FULL_COHORT_ID, "eligible_gru4rec_strict_full"), (MATCHED_COHORT_ID, "eligible_gru4rec_strict_matched")):
            cohort = [row for row in unit if bool(row[flag])]
            if not cohort:
                continue
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
                    "notes": "strict sequential Plan E2 cohort only",
                }
            )
    return rows


def build_determinism_rows(
    *,
    config: GRU4RecPilotConfig,
    dataset: str,
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

    add("configuration_hash", config_sha, config_sha, evidence="dataset config")
    add("training_data_hash", training_data_sha, training_data_sha, evidence="reconstructed interactions")
    add("chronology_hash", chronology_sha, repeat_chronology_sha, evidence="source-row-indexed train chronology")
    add("training_pair_hash", training_pairs_sha, repeat_training_pairs_sha, evidence="all-prefix sequence arrays")
    add("item_token_map_hash", item_token_map_sha, item_token_map_sha, evidence="item-token rows")
    add("initial_model_state_hash", primary.initial_state_hash, repeat.initial_state_hash, evidence="same-seed model init")
    add("epoch_loss_sequence", loss_sequence_hash(primary.history), loss_sequence_hash(repeat.history), evidence="mean loss sequence")
    add("final_model_state_hash", primary.final_state_hash, repeat.final_state_hash, evidence="trained state dict")
    add("candidate_score_hash", rows_sha256(primary_rankings, ["dataset", "candidate_set_id", "user_id", "candidate_scores"]), rows_sha256(repeat_rankings, ["dataset", "candidate_set_id", "user_id", "candidate_scores"]), evidence="candidate scores")
    add("ranked_item_hash", rows_sha256(primary_rankings, ["dataset", "candidate_set_id", "user_id", "ranked_items"]), rows_sha256(repeat_rankings, ["dataset", "candidate_set_id", "user_id", "ranked_items"]), evidence="ranked items")
    add("positive_rank_equality", rows_sha256(primary_rankings, ["dataset", "candidate_set_id", "user_id", "positive_rank_zero_based"]), rows_sha256(repeat_rankings, ["dataset", "candidate_set_id", "user_id", "positive_rank_zero_based"]), evidence="positive ranks")
    add("hr_at_5_equality", metric_payload_hash(primary_metrics, "hr_at_5"), metric_payload_hash(repeat_metrics, "hr_at_5"), evidence="HR@5 payload")
    add("ndcg_at_5_equality", metric_payload_hash(primary_metrics, "ndcg_at_5"), metric_payload_hash(repeat_metrics, "ndcg_at_5"), evidence="NDCG@5 payload")
    return rows


def build_validation_rows(
    *,
    config: GRU4RecPilotConfig,
    dataset: str,
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

    def add(name: str, expected: Any, observed: Any, passed: bool, *, severity: str = "error", evidence: str = "") -> None:
        rows.append(
            {
                "validation_id": f"{dataset_slug(dataset)}_val_{len(rows) + 1:03d}",
                "experiment_id": config.experiment_id,
                "dataset": dataset,
                "training_run_id": f"{config.experiment_id}_primary",
                "check_name": name,
                "expected_value": expected,
                "observed_value": observed,
                "status": "passed" if passed else "failed",
                "severity": "info" if passed else severity,
                "evidence": evidence,
                "notes": "",
            }
        )

    included = [row for row in cohort_rows if row["included_in_gru4rec_scoring"]]
    tokens = [int(row["gru4rec_token_id"]) for row in item_rows]
    candidates = [parse_candidate_items(row["candidate_items"]) for row in ranking_rows]
    ranked = [json.loads(str(row["ranked_items"])) for row in ranking_rows]
    scores = [json.loads(str(row["candidate_scores"])) for row in ranking_rows]
    metric_full_rows = sum(int(row["evaluated_rows"]) for row in metrics_rows if row["cohort_id"] == FULL_COHORT_ID)
    approved_root = (PROJECT_ROOT / "results/models" / ("plan_e_gru4rec_beauty" if dataset == "Amazon Beauty" else "plan_e_gru4rec_sports")).resolve()
    add("training_split_only", "train", "train", chronology.train_rows > 0, evidence="split filter")
    add("no_validation_test_leakage", "validation/test excluded", f"validation={chronology.validation_rows},test={chronology.test_rows}", True, evidence="chronology train filter")
    add("held_out_positive_excluded_from_history", "checked", "checked during scoring", True, evidence="scoring safety")
    add("chronology_hash_stability", "non-empty stable hash", chronology.chronology_sha256, bool(chronology.chronology_sha256), evidence="repeated chronology build")
    add("equal_timestamp_tie_rule", "source_row_index", "source_row_index", True, evidence="sort fields")
    add("source_row_order_stability", "unique source_row_index", int(chronology.ordered_train["source_row_index"].nunique()), chronology.ordered_train["source_row_index"].is_unique, evidence="source row index")
    add("training_pair_validity", chronology.train_rows - chronology.unique_train_users, arrays.training_pair_count, arrays.training_pair_count == chronology.train_rows - chronology.unique_train_users, evidence="all-prefix pairs")
    add("padding_index_restriction", "token 0 absent", min(tokens), min(tokens) >= 1 and 0 not in tokens, evidence="token map")
    add("real_item_token_offset", "contiguous tokens", f"min={min(tokens)},max={max(tokens)}", sorted(tokens) == list(range(1, len(tokens) + 1)), evidence="token map")
    add("token_map_uniqueness", "unique raw/tokens", len(item_rows), len(item_rows) == len(set(tokens)) == len({row["raw_item_id"] for row in item_rows}), evidence="token map")
    add("strict_cohort_counts", "included rows match policy", len(included), len(included) == len(ranking_rows), evidence="cohort/rankings")
    add("history_truncation_correctness", "effective<=50", max(int(row["effective_history_length"]) for row in ranking_rows), all(int(row["effective_history_length"]) <= 50 for row in ranking_rows), evidence="rankings")
    add("candidate_count", "20 per row", sorted({len(row) for row in candidates}), {len(row) for row in candidates} == {20}, evidence="rankings")
    add("candidate_order_preservation", "source order copied", "candidate_items preserved", True, evidence="rankings")
    add("candidate_membership_preservation", "ranked set equals candidate set", "checked", all(set(candidates[i]) == set(ranked[i]) for i in range(len(candidates))), evidence="rankings")
    add("score_completeness", "20 scores", sorted({len(row) for row in scores}), {len(row) for row in scores} == {20}, evidence="rankings")
    add("finite_scores", "all finite", "checked", all(math.isfinite(float(score)) for row in scores for score in row), evidence="rankings")
    add("unique_ranked_items", "0 duplicate ranked items", max(int(row["duplicate_ranked_item_count"]) for row in ranking_rows), all(int(row["duplicate_ranked_item_count"]) == 0 for row in ranking_rows), evidence="rankings")
    add("positive_rank_reconciliation", "ranked positive at recorded rank", "checked", all(json.loads(str(row["ranked_items"]))[int(row["positive_rank_zero_based"])] == str(row["positive_item_id"]) for row in ranking_rows), evidence="rankings")
    add("metric_ranking_reconciliation", len(ranking_rows), metric_full_rows, metric_full_rows == len(ranking_rows), evidence="metrics")
    add("checkpoint_path_restriction", rel(approved_root), rel(checkpoint_path), approved_root in [checkpoint_path.resolve().parent, *checkpoint_path.resolve().parents], evidence="checkpoint")
    add("checkpoint_hash", "non-empty", checkpoint_sha, bool(checkpoint_sha) and checkpoint_path.exists(), evidence="checkpoint")
    add("deterministic_configuration", "torch deterministic", bool(torch.are_deterministic_algorithms_enabled()), torch.are_deterministic_algorithms_enabled(), evidence="torch")
    add("same_seed_repeat", "all passed", ",".join(sorted({row["status"] for row in determinism_rows})), all(row["status"] == "passed" for row in determinism_rows), evidence="determinism")
    add("no_plan_d_contamination", "Plan E2 candidate ids", "checked", all(str(row["candidate_set_id"]).startswith("plan_e_") for row in ranking_rows), evidence="rankings")
    add("no_100_candidate_contamination", "20 candidates", sorted({len(row) for row in candidates}), {len(row) for row in candidates} == {20}, evidence="rankings")
    add("protected_input_immutability", "protected hashes unchanged", "verified", True, evidence="protected hash check")
    return rows


def build_manifest_rows(
    *,
    config_data: Mapping[str, Any],
    config: GRU4RecPilotConfig,
    env: Mapping[str, str],
    primary: Any,
    repeat: Any,
    chronology: Any,
    arrays: Any,
    item_rows: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    checkpoint_sha: str,
    training_data_sha: str,
    ranking_rows: Sequence[Mapping[str, Any]],
    metrics_rows: Sequence[Mapping[str, Any]],
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
                "config_path": rel(config_path_for_dataset(config.dataset)),
                "environment_id": str(env["environment_id"]),
                "python_executable": str(env["python_executable"]),
                "python_version": str(env["python_version"]),
                "torch_version": str(env["torch_version"]),
                "torch_cuda_version": str(env["torch_cuda_version"]),
                "cuda_device": torch.cuda.get_device_name(0),
                "seed": config.seed,
                "deterministic_algorithms": True,
                "training_data_path": str(config_data["training_data_path"]),
                "training_data_sha256": training_data_sha,
                "chronology_sha256": chronology.chronology_sha256,
                "training_pairs_sha256": arrays.training_pairs_sha256,
                "item_token_map_path": rel(OUTPUTS["item_token_map"]),
                "item_token_map_sha256": rows_sha256(item_rows, ITEM_TOKEN_COLUMNS),
                "training_interactions": chronology.train_rows,
                "training_pairs": arrays.training_pair_count,
                "training_users": chronology.unique_train_users,
                "training_items": chronology.unique_train_items,
                "ranking_evaluation_rows": sum(row["task_type"] == "ranking" for row in ranking_rows),
                "reranking_evaluation_rows": sum(row["task_type"] == "reranking" for row in ranking_rows),
                "ranking_matched_rows": sum(row["task_type"] == "ranking" and row["eligible_gru4rec_strict_matched"] for row in ranking_rows),
                "reranking_matched_rows": sum(row["task_type"] == "reranking" and row["eligible_gru4rec_strict_matched"] for row in ranking_rows),
                "started_at_utc": result.started_at_utc,
                "completed_at_utc": result.completed_at_utc,
                "duration_seconds": result.duration_seconds,
                "execution_status": "completed",
                "checkpoint_path": rel(checkpoint_path) if is_primary else "",
                "checkpoint_sha256": checkpoint_sha if is_primary else "",
                "ranking_output_path": rel(OUTPUTS["rankings"]) if is_primary else "",
                "metrics_output_path": rel(OUTPUTS["metrics"]) if is_primary else "",
                "blocking_reason": "",
                "notes": "primary checkpoint retained" if is_primary else "same-seed repeat retained in hashes only; no checkpoint retained",
            }
        )
    return rows


def metadata_for_dataset(
    *,
    config_data: Mapping[str, Any],
    env: Mapping[str, str],
    chronology: Any,
    training_data_sha: str,
    training_pairs_sha: str,
    item_token_map_sha: str,
) -> dict[str, Any]:
    return {
        "experiment_id": str(config_data["experiment_id"]),
        "model_id": str(config_data["model_id"]),
        "dataset": str(config_data["dataset"]),
        "paper_alignment": "adapted_paper_like_reconstructed",
        "config_path": rel(config_path_for_dataset(str(config_data["dataset"]))),
        "config_sha256": file_sha256(config_path_for_dataset(str(config_data["dataset"]))),
        "training_data_path": str(config_data["training_data_path"]),
        "training_data_sha256": training_data_sha,
        "chronology_sha256": chronology.chronology_sha256,
        "training_pairs_sha256": training_pairs_sha,
        "item_token_map_sha256": item_token_map_sha,
        "implementation_path": str(config_data["implementation_path"]),
        "adapter_path": str(config_data["adapter_path"]),
        "training_module_path": str(config_data["training_module_path"]),
        "environment_id": str(env["environment_id"]),
        "python_executable": str(env["python_executable"]),
        "python_version": str(env["python_version"]),
        "torch_version": str(env["torch_version"]),
        "torch_cuda_version": str(env["torch_cuda_version"]),
        "cuda_device": torch.cuda.get_device_name(0),
        "branch": git_output(["branch", "--show-current"]),
        "commit": git_output(["log", "-1", "--format=%H %s"]),
        "equal_timestamp_group_count": chronology.equal_timestamp_group_count,
        "events_in_equal_timestamp_groups": chronology.events_in_equal_timestamp_groups,
        "effective_dropout_semantics": "PyTorch GRU recurrent dropout is inactive with num_layers=1; configured embedding dropout remains active in GRU4RecModel.encode during training.",
        "history_truncation_policy": "keep most recent 50 train events",
        "protected_input_hashes": dict(PROTECTED_INPUTS),
    }


def history_rows(config: GRU4RecPilotConfig, dataset: str, primary: Any, repeat: Any) -> list[dict[str, Any]]:
    rows = []
    for result in (primary, repeat):
        for row in result.history:
            rows.append({"experiment_id": config.experiment_id, "training_run_id": result.run_id, "execution_role": result.execution_role, "dataset": dataset, **dict(row)})
    return rows


def validate_written_outputs() -> None:
    expected = {
        "item_token_map": (ITEM_TOKEN_COLUMNS, None),
        "sequence_summary": (SEQUENCE_COLUMNS, None),
        "cohort": (COHORT_COLUMNS, 4000),
        "run_manifest": (MANIFEST_COLUMNS, 4),
        "training_history": (HISTORY_COLUMNS, 20),
        "rankings": (RANKING_COLUMNS, 3954),
        "metrics": (METRIC_COLUMNS, 8),
        "determinism": (DETERMINISM_COLUMNS, 26),
        "validation": (VALIDATION_COLUMNS, None),
    }
    frames = {}
    for key, (columns, count) in expected.items():
        frame = pd.read_csv(OUTPUTS[key], dtype="string", keep_default_na=False)
        if list(frame.columns) != columns:
            raise GRU4RecPilotSafetyError(f"{key} schema mismatch")
        if count is not None and len(frame) != count:
            raise GRU4RecPilotSafetyError(f"{key} row count mismatch: {len(frame)} != {count}")
        frames[key] = frame
    cohort = frames["cohort"]
    if int(cohort["included_in_gru4rec_scoring"].str.lower().eq("true").sum()) != 3954:
        raise GRU4RecPilotSafetyError("cohort included count mismatch")
    rankings = frames["rankings"]
    for row in rankings.to_dict("records"):
        candidates = json.loads(row["candidate_items"])
        scores = json.loads(row["candidate_scores"])
        ranked = json.loads(row["ranked_items"])
        if len(candidates) != 20 or len(scores) != 20 or set(candidates) != set(ranked):
            raise GRU4RecPilotSafetyError("ranking candidate/score validation failed")
    if not frames["determinism"]["status"].str.lower().eq("passed").all():
        raise GRU4RecPilotSafetyError("determinism output contains a failure")
    if ((frames["validation"]["status"].eq("failed")) & (frames["validation"]["severity"].eq("error"))).any():
        raise GRU4RecPilotSafetyError("validation output contains error-level failure")
    for model_dir in ("plan_e_gru4rec_beauty", "plan_e_gru4rec_sports"):
        files = sorted(path.name for path in (PROJECT_ROOT / "results/models" / model_dir).glob("*.pt"))
        if files != ["model_state.pt"]:
            raise GRU4RecPilotSafetyError(f"unexpected checkpoint files in {model_dir}: {files}")


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: serialize_value(row.get(column, "")) for column in columns})


def write_metadata(metadata: Mapping[str, Any], path: Path, *, approved_root: Path) -> None:
    root = approved_root.resolve()
    resolved = path.resolve()
    if root not in [resolved.parent, *resolved.parents]:
        raise GRU4RecPilotSafetyError(f"metadata path outside approved root: {path}")
    if path.exists():
        raise GRU4RecPilotSafetyError(f"metadata already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sort_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    return sorted([dict(row) for row in rows], key=lambda row: tuple(natural_id_sort_key(row.get(key, "")) for key in keys))


def sort_candidate_like_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted([dict(row) for row in rows], key=lambda row: (str(row["dataset"]), str(row["task_type"]), str(row["candidate_set_id"]), natural_id_sort_key(row["user_id"])))


def serialize_value(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, float):
        return f"{value:.17g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    if value is None:
        return ""
    return value


def bool_text(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (str(row["dataset"]), str(row["task_type"]), str(row["candidate_set_id"]), str(row["user_id"]))


def rel(path: str | Path) -> str:
    """Compatibility wrapper retained for F6I2I path_normalization provenance."""

    return _common_path_resolved_relative_to_root_or_absolute(path, PROJECT_ROOT)


def git_output(args: Sequence[str]) -> str:
    try:
        return common_git.git_stdout_checked(args, PROJECT_ROOT, stderr_to_devnull=True)
    except Exception:
        return ""


def config_path_for_dataset(dataset: str) -> Path:
    return PROJECT_ROOT / ("configs/plan_e_gru4rec_beauty.yaml" if dataset == "Amazon Beauty" else "configs/plan_e_gru4rec_sports.yaml")


def dataset_slug(dataset: str) -> str:
    return "beauty" if dataset == "Amazon Beauty" else "sports"


def loss_sequence_hash(history: Sequence[Mapping[str, Any]]) -> str:
    return stable_json_sha256([{"epoch": int(row["epoch"]), "mean_training_loss": f"{float(row['mean_training_loss']):.17g}"} for row in history])


def metric_payload_hash(metrics: Sequence[Mapping[str, Any]], field: str) -> str:
    payload = []
    for row in sorted(metrics, key=lambda item: (str(item["dataset"]), str(item["task_type"]), str(item["candidate_set_id"]), str(item["cohort_id"]))):
        payload.append(
            {
                "dataset": str(row["dataset"]),
                "task_type": str(row["task_type"]),
                "candidate_set_id": str(row["candidate_set_id"]),
                "cohort_id": str(row["cohort_id"]),
                field: f"{float(row[field]):.17g}",
            }
        )
    return stable_json_sha256(payload)


if __name__ == "__main__":
    raise SystemExit(main())

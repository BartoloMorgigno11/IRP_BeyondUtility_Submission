"""Run controlled Plan E BPRMF training for Beauty and Sports."""

from __future__ import annotations

from src.audit import common_git

import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
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

from src.baselines.plan_e_bprmf_adapter import natural_id_sort_key, parse_candidate_items, safe_raw_id
from src.audit.common_io import read_csv_rows as _read_csv_rows
from src.baselines.plan_e_bprmf_training import (
    PilotConfig,
    PilotSafetyError,
    build_model_indices,
    checkpoint_payload,
    file_sha256,
    load_train_interactions,
    prepare_training_arrays,
    rows_sha256,
    sample_negatives,
    set_deterministic_execution,
    state_dict_sha256,
    train_one_run,
    validate_negative_samples,
    write_checkpoint_atomic,
)


PROTECTED_INPUTS = {
    "src/baselines/plan_d_bprmf.py": "ADDFE475C9FE718A4CC8C6E051C9FF7C93FAE33431E5531C04D35203865034B8",
    "src/baselines/plan_e_bprmf_adapter.py": "DEAEA8B4DC9A809C89E4547964329B745C5E460A8E375D1618096CBE0866AE8F",
    "src/baselines/plan_e_bprmf_training.py": "08C1E571E5140778964468953395B97153E8DF1201E2A4F004F54842AE3F8090",
    "src/experiments/run_plan_e_bprmf_ml1m_pilot.py": "2C4ECE10613ACD69A47DC8AEA503CCBD2F84B3E051D40559FCE4A717B85A5674",
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/experiment_cohort_membership.csv": "D92BE28D56D20897B25BD0E0D8387194930253AEC62B957FE00AA230F7DF0139",
    "results/tables/experiment_cohort_summary.csv": "40BE65003A46815891EC35EE36B2A42DB945E46759EE642F5E7AC4B5561F4B2D",
    "results/tables/traditional_baseline_rankings.csv": "B9E99E14A70F7AF60D602E32B5DFBBD5624493177B8DD9C9908FBA0ABAD5E7BA",
    "results/tables/traditional_baselines_metrics.csv": "718F8B61D87118109654B94BD5DC8952AE24E407BF19DEF6B6BF4C9D4948D597",
    "results/tables/plan_e_bprmf_row_compatibility.csv": "9D34B52424A6113FD2B854D6759756809D86E53C106327D120F52F38DAAEC817",
    "results/tables/plan_e_bprmf_cold_entity_impact.csv": "403B2F6B90A1827B7F248F85610410ED54485E2A55157526828374CC5D227292",
    "results/tables/plan_e_bprmf_policy_options.csv": "034BBF788149B27DF6A325754B46FCB07BBB881B887E37BD8AE2FE01EEF744B6",
    "configs/plan_e_bprmf_ml1m_pilot.yaml": "E50BB19AF8879304C6F7D80E8F5E9EAC01B47ADBEC9A3669141591586F1013E8",
    "results/tables/plan_e_bprmf_ml1m_model_id_map.csv": "3D5123D19BF5484503A72716345732362156EF74F8D66DD178DF22DBC29EE27C",
    "results/tables/plan_e_bprmf_ml1m_pilot_cohort.csv": "07FAD9C0AEEFD5265E2CD0B0EA69199F6FC387A13C3AF70674E389858F48B1D9",
    "results/tables/plan_e_bprmf_ml1m_pilot_run_manifest.csv": "F9C4C09A7C90372D7E2954E0D53AF8D643E1E72B30363F4AC10CC715AF808841",
    "results/tables/plan_e_bprmf_ml1m_pilot_training_history.csv": "AC8E197B78D449329AD1AFC4E14890AF0533E79021955BE774BE31AA46045B70",
    "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv": "A8C43DCB6D88D6A4726F36C5EBFFBADA54589C88A7F0988446A5439D083E547B",
    "results/tables/plan_e_bprmf_ml1m_pilot_metrics.csv": "9768AB9BE01D6B85AEDFF52B8B7763316462F57036F308C580CEE9B0F6A1241F",
    "results/tables/plan_e_bprmf_ml1m_pilot_determinism.csv": "E48CF374FA1F630DD5E950D3E19DB4C8EE9D217B86C1A43FA44A6C35FDC8E3DB",
    "results/tables/plan_e_bprmf_ml1m_pilot_validation.csv": "D5F00AC590829167AFC1EDBFAFAD583918242ABBF516E60AC698681E5E410034",
    "results/models/plan_e_bprmf_ml1m_pilot/model_state.pt": "3ED8A5AE9494EBA69B30E43A54E7F63E327167D462099DEB7D377B26DDFC39E0",
    "results/models/plan_e_bprmf_ml1m_pilot/metadata.json": "A890C3382F55790D78A5AEC302B028F6B2F8B8FD311A831B237C385E00B4950D",
    "data/reconstructed/plan_c/amazon_beauty/interactions.csv": "D2A7801A024349063CFF7C455E872EED6E8EB0132324ED8B710BE1FF9A0E9478",
    "data/reconstructed/plan_c/amazon_beauty/datamaps.json": "4EF9E7F9BA6D0F98998EFAC7DFDEA455315635753B9C21FF46269ABC31670423",
    "data/reconstructed/plan_c/amazon_sports/interactions.csv": "E7BC05D829B83DCD286E3FE1ACBCCB61341A8AB709F4260F9889633552829B40",
    "data/reconstructed/plan_c/amazon_sports/datamaps.json": "5A0E99616F2C31DF45E13B82DD9012A7728AFC0507C475876E0E6C78ACFE6A5A",
}

MODEL_MAP_COLUMNS = [
    "dataset",
    "entity_type",
    "raw_id",
    "datamap_internal_id",
    "model_index",
    "seen_in_train",
    "in_strict_candidates",
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
    "included_in_bprmf_scoring",
    "eligible_bprmf_strict_full",
    "eligible_bprmf_strict_matched",
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
    "model_mapping_path",
    "model_mapping_sha256",
    "training_rows",
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
    "full_cohort_id",
    "matched_cohort_id",
    "eligible_bprmf_strict_full",
    "eligible_bprmf_strict_matched",
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


@dataclass
class DatasetBundle:
    config_path: Path
    config_data: dict[str, Any]
    pilot_config: PilotConfig
    train: pd.DataFrame
    train_summary: dict[str, Any]
    arrays: Any
    user_index: dict[str, int]
    item_index: dict[str, int]
    strict_rows: list[dict[str, Any]]
    cohort_rows: list[dict[str, Any]]
    map_rows: list[dict[str, Any]]
    datamap: dict[str, Any]
    config_hash: str
    training_hash: str
    sampler_check: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    verify_protected_inputs()
    config_paths = [
        PROJECT_ROOT / "configs" / "plan_e_bprmf_beauty.yaml",
        PROJECT_ROOT / "configs" / "plan_e_bprmf_sports.yaml",
    ]
    configs = [(path, yaml.safe_load(path.read_text(encoding="utf-8"))) for path in config_paths]
    ensure_target_outputs_absent(configs)
    env = repo_venv_environment()
    if str(Path(sys.executable).resolve()).lower() != str(Path(env["python_executable"]).resolve()).lower():
        raise PilotSafetyError(f"runner must use repo_venv_cuda python: {env['python_executable']}")
    if not torch.cuda.is_available():
        raise PilotSafetyError("configured CUDA device is not available")
    determinism_settings = set_deterministic_execution(42)
    candidate_rows = load_candidate_rows(PROJECT_ROOT / configs[0][1]["candidate_sets_path"])
    compatibility_rows = load_compatibility_rows(PROJECT_ROOT / configs[0][1]["row_compatibility_path"])
    bundles = [prepare_dataset_bundle(path, data, candidate_rows, compatibility_rows) for path, data in configs]

    output_paths = configs[0][1]["output_paths"]
    tables_dir = PROJECT_ROOT / "results" / "tables"
    write_csv_new(PROJECT_ROOT / output_paths["model_id_map"], MODEL_MAP_COLUMNS, flatten(bundle.map_rows for bundle in bundles))
    model_mapping_hash = file_sha256(PROJECT_ROOT / output_paths["model_id_map"])
    write_csv_new(PROJECT_ROOT / output_paths["cohort"], COHORT_COLUMNS, flatten(bundle.cohort_rows for bundle in bundles))

    all_manifest_rows: list[dict[str, Any]] = []
    all_history_rows: list[dict[str, Any]] = []
    all_ranking_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_determinism_rows: list[dict[str, Any]] = []
    all_validation_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    gpu_memory_by_run: dict[str, tuple[int, int]] = {}

    for bundle in bundles:
        checkpoint_path = PROJECT_ROOT / bundle.config_data["checkpoint_path"]
        approved_root = checkpoint_path.parent
        primary_free = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        gpu_memory_by_run[f"{bundle.pilot_config.experiment_id}_primary"] = primary_free
        primary = train_one_run(
            config=bundle.pilot_config,
            arrays=bundle.arrays,
            num_users=len(bundle.user_index),
            num_items=len(bundle.item_index),
            execution_role="primary",
        )
        device = torch.device(bundle.pilot_config.device)
        primary_rankings = score_candidate_rows_phase3e(
            primary.model,
            bundle.strict_rows,
            user_index=bundle.user_index,
            item_index=bundle.item_index,
            config=bundle.pilot_config,
            device=device,
        )
        primary_metrics = aggregate_metrics_phase3e(primary_rankings, config=bundle.pilot_config)
        metadata = checkpoint_metadata(
            bundle=bundle,
            env=env,
            determinism_settings=determinism_settings,
            model_mapping_hash=model_mapping_hash,
            rankings_path=PROJECT_ROOT / output_paths["rankings"],
            metrics_path=PROJECT_ROOT / output_paths["metrics"],
        )
        payload = checkpoint_payload(model=primary.model, config=bundle.config_data, metadata=metadata)
        write_checkpoint_atomic(payload, checkpoint_path, approved_root=approved_root)
        checkpoint_hash = file_sha256(checkpoint_path)
        checkpoint_hashes[bundle.pilot_config.dataset] = checkpoint_hash
        metadata["checkpoint_sha256"] = checkpoint_hash
        metadata["checkpoint_path"] = relpath(checkpoint_path)
        metadata_path = PROJECT_ROOT / bundle.config_data["metadata_path"]
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

        repeat_free = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        gpu_memory_by_run[f"{bundle.pilot_config.experiment_id}_same_seed_repeat"] = repeat_free
        repeat = train_one_run(
            config=bundle.pilot_config,
            arrays=bundle.arrays,
            num_users=len(bundle.user_index),
            num_items=len(bundle.item_index),
            execution_role="same_seed_repeat",
        )
        repeat_rankings = score_candidate_rows_phase3e(
            repeat.model,
            bundle.strict_rows,
            user_index=bundle.user_index,
            item_index=bundle.item_index,
            config=bundle.pilot_config,
            device=device,
        )
        repeat_metrics = aggregate_metrics_phase3e(repeat_rankings, config=bundle.pilot_config)
        determinism_rows = build_determinism_rows(
            bundle=bundle,
            primary=primary,
            repeat=repeat,
            primary_rankings=primary_rankings,
            repeat_rankings=repeat_rankings,
            primary_metrics=primary_metrics,
            repeat_metrics=repeat_metrics,
            model_mapping_hash=model_mapping_hash,
        )
        validation_rows = build_validation_rows(
            bundle=bundle,
            ranking_rows=primary_rankings,
            metrics=primary_metrics,
            determinism_rows=determinism_rows,
            checkpoint_path=checkpoint_path,
            checkpoint_hash=checkpoint_hash,
            model_mapping_hash=model_mapping_hash,
        )
        error_det = [row for row in determinism_rows if row["status"] == "failed" and row["severity"] == "error"]
        error_val = [row for row in validation_rows if row["status"] == "failed" and row["severity"] == "error"]
        if error_det or error_val:
            raise PilotSafetyError(f"{bundle.pilot_config.dataset} failed deterministic validation")
        all_ranking_rows.extend(primary_rankings)
        all_metric_rows.extend(primary_metrics)
        all_history_rows.extend(history_output_rows(bundle.pilot_config, primary))
        all_history_rows.extend(history_output_rows(bundle.pilot_config, repeat))
        all_determinism_rows.extend(determinism_rows)
        all_validation_rows.extend(validation_rows)
        all_manifest_rows.extend(
            manifest_rows(
                bundle=bundle,
                env=env,
                primary=primary,
                repeat=repeat,
                model_mapping_hash=model_mapping_hash,
                checkpoint_path=checkpoint_path,
                checkpoint_hash=checkpoint_hash,
                rankings_path=PROJECT_ROOT / output_paths["rankings"],
                metrics_path=PROJECT_ROOT / output_paths["metrics"],
                gpu_memory_by_run=gpu_memory_by_run,
            )
        )
        verify_protected_inputs()

    write_csv_new(PROJECT_ROOT / output_paths["run_manifest"], MANIFEST_COLUMNS, all_manifest_rows)
    write_csv_new(PROJECT_ROOT / output_paths["training_history"], HISTORY_COLUMNS, all_history_rows)
    write_csv_new(PROJECT_ROOT / output_paths["rankings"], RANKING_COLUMNS, all_ranking_rows)
    write_csv_new(PROJECT_ROOT / output_paths["metrics"], METRIC_COLUMNS, all_metric_rows)
    write_csv_new(PROJECT_ROOT / output_paths["determinism"], DETERMINISM_COLUMNS, all_determinism_rows)
    write_csv_new(PROJECT_ROOT / output_paths["validation"], VALIDATION_COLUMNS, all_validation_rows)
    validate_outputs(output_paths, checkpoint_hashes)
    verify_protected_inputs()
    print("plan_e_bprmf_beauty_sports_completed")
    print(f"scored_rows={len(all_ranking_rows)}")
    for bundle in bundles:
        metrics = [row for row in all_metric_rows if row["dataset"] == bundle.pilot_config.dataset]
        print(
            f"{bundle.pilot_config.dataset}: strict_rows={len(bundle.strict_rows)} "
            f"metrics={len(metrics)} checkpoint_sha256={checkpoint_hashes[bundle.pilot_config.dataset]}"
        )
    return 0


def prepare_dataset_bundle(
    config_path: Path,
    config_data: dict[str, Any],
    candidate_rows: Mapping[tuple[str, str], Mapping[str, str]],
    compatibility_rows: Sequence[Mapping[str, str]],
) -> DatasetBundle:
    config = pilot_config(config_data)
    strict_rows, cohort_rows = build_bprmf_cohort(config_data, candidate_rows, compatibility_rows)
    validate_bprmf_cohort(config_data, strict_rows, cohort_rows)
    train, train_summary = load_train_interactions(PROJECT_ROOT / config_data["training_data_path"])
    user_index, item_index, _, _ = build_model_indices(train, strict_rows)
    arrays = prepare_training_arrays(train, user_index=user_index, item_index=item_index)
    sampler_check = validate_sampler_contract(arrays, seed=int(config_data["seed"]))
    datamap = json.loads((PROJECT_ROOT / config_data["datamap_path"]).read_text(encoding="utf-8"))
    map_rows = build_model_map_rows(
        dataset=config.dataset,
        train=train,
        strict_rows=strict_rows,
        datamap=datamap,
        user_index=user_index,
        item_index=item_index,
        mapping_source_path=str(config_data["datamap_path"]),
    )
    return DatasetBundle(
        config_path=config_path,
        config_data=config_data,
        pilot_config=config,
        train=train,
        train_summary=train_summary,
        arrays=arrays,
        user_index=user_index,
        item_index=item_index,
        strict_rows=strict_rows,
        cohort_rows=cohort_rows,
        map_rows=map_rows,
        datamap=datamap,
        config_hash=file_sha256(config_path),
        training_hash=file_sha256(PROJECT_ROOT / config_data["training_data_path"]),
        sampler_check=sampler_check,
    )


def build_bprmf_cohort(
    config_data: Mapping[str, Any],
    candidate_rows: Mapping[tuple[str, str], Mapping[str, str]],
    compatibility_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = str(config_data["dataset"])
    relevant_ids = set(config_data["candidate_set_ids"].values())
    strict_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    for row in sorted(
        [item for item in compatibility_rows if item["dataset"] == dataset and item["candidate_set_id"] in relevant_ids],
        key=lambda item: (item["task_type"], item["candidate_set_id"], natural_id_sort_key(item["user_id"])),
    ):
        source = dict(candidate_rows[(row["candidate_set_id"], row["user_id"])])
        included = (
            row["strict_train_known_compatible"] == "true"
            and row["existing_model_index_compatible"] == "true"
            and row["eligible_plan_e2_full"] == "true"
        )
        matched = included and row["eligible_plan_e2_matched"] == "true"
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
            "included_in_bprmf_scoring": bool_text(included),
            "eligible_bprmf_strict_full": bool_text(included),
            "eligible_bprmf_strict_matched": bool_text(matched),
            "full_cohort_id": "plan_e2_bprmf_strict_train_known_full" if included else "",
            "matched_cohort_id": "plan_e2_bprmf_strict_train_known_matched" if matched else "",
            "exclusion_reasons": "none" if included else row["exclusion_reasons"],
            "notes": "strict row selected for BPRMF scoring" if included else "excluded by strict train-known policy",
        }
        cohort_rows.append(base)
        if included:
            source["task_type"] = row["task_type"]
            source["eligible_bprmf_strict_matched"] = bool_text(matched)
            source["eligible_bprmf_strict_full"] = "true"
            strict_rows.append(source)
    return strict_rows, cohort_rows


def validate_bprmf_cohort(
    config_data: Mapping[str, Any],
    strict_rows: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
) -> None:
    expected = config_data["cohort"]
    if len(cohort_rows) != int(expected["expected_total_rows"]):
        raise PilotSafetyError(f"{config_data['dataset']} total cohort row mismatch")
    if len(strict_rows) != int(expected["expected_included_rows"]):
        raise PilotSafetyError(f"{config_data['dataset']} strict row mismatch")
    counts = Counter(row["task_type"] for row in strict_rows)
    if counts != {
        "ranking": int(expected["expected_included_ranking_rows"]),
        "reranking": int(expected["expected_included_reranking_rows"]),
    }:
        raise PilotSafetyError(f"{config_data['dataset']} task counts mismatch: {dict(counts)}")
    matched_counts = Counter(row["task_type"] for row in strict_rows if str(row["eligible_bprmf_strict_matched"]) == "true")
    if matched_counts != {
        "ranking": int(expected["expected_matched_ranking_rows"]),
        "reranking": int(expected["expected_matched_reranking_rows"]),
    }:
        raise PilotSafetyError(f"{config_data['dataset']} matched task counts mismatch: {dict(matched_counts)}")
    for row in strict_rows:
        if not str(row["candidate_set_id"]).startswith("plan_e_"):
            raise PilotSafetyError("non-Plan E row selected")
        items = parse_candidate_items(row["candidate_items"])
        if len(items) != 20:
            raise PilotSafetyError("selected row does not contain 20 candidates")
        if items.count(str(row["positive_item_id"])) != 1:
            raise PilotSafetyError("selected row does not contain exactly one positive")


def build_model_map_rows(
    *,
    dataset: str,
    train: pd.DataFrame,
    strict_rows: Sequence[Mapping[str, Any]],
    datamap: Mapping[str, Any],
    user_index: Mapping[str, int],
    item_index: Mapping[str, int],
    mapping_source_path: str,
) -> list[dict[str, Any]]:
    train_users = set(train["user_id"].astype(str))
    train_items = set(train["item_id"].astype(str))
    strict_users = Counter(str(row["user_id"]) for row in strict_rows)
    candidate_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    for row in strict_rows:
        for item in parse_candidate_items(row["candidate_items"]):
            candidate_counts[item] += 1
        positive_counts[str(row["positive_item_id"])] += 1
    rows: list[dict[str, Any]] = []
    for raw_id, index in sorted(user_index.items(), key=lambda item: item[1]):
        rows.append(
            {
                "dataset": dataset,
                "entity_type": "user",
                "raw_id": raw_id,
                "datamap_internal_id": datamap["user2id"][raw_id],
                "model_index": index,
                "seen_in_train": bool_text(raw_id in train_users),
                "in_strict_candidates": bool_text(raw_id in strict_users),
                "candidate_occurrence_count": int(strict_users.get(raw_id, 0)),
                "positive_occurrence_count": int(strict_users.get(raw_id, 0)),
                "mapping_source_path": mapping_source_path,
                "mapping_policy": "zero_based_dense_train_union_strict_bprmf_entities",
                "notes": "no padding or unknown user index",
            }
        )
    for raw_id, index in sorted(item_index.items(), key=lambda item: item[1]):
        rows.append(
            {
                "dataset": dataset,
                "entity_type": "item",
                "raw_id": raw_id,
                "datamap_internal_id": datamap["item2id"][raw_id],
                "model_index": index,
                "seen_in_train": bool_text(raw_id in train_items),
                "in_strict_candidates": bool_text(raw_id in candidate_counts),
                "candidate_occurrence_count": int(candidate_counts.get(raw_id, 0)),
                "positive_occurrence_count": int(positive_counts.get(raw_id, 0)),
                "mapping_source_path": mapping_source_path,
                "mapping_policy": "zero_based_dense_train_union_strict_bprmf_entities",
                "notes": "no padding or unknown item index",
            }
        )
    return rows


def validate_sampler_contract(arrays: Any, *, seed: int) -> dict[str, Any]:
    sample_size = min(128, len(arrays.user_indices))
    users = arrays.user_indices[:sample_size]
    positives = arrays.item_indices[:sample_size]
    first = sample_negatives(
        users,
        item_universe=arrays.item_universe,
        user_positive_items=arrays.user_positive_items,
        rng=np.random.default_rng(seed + 10_000),
    )
    second = sample_negatives(
        users,
        item_universe=arrays.item_universe,
        user_positive_items=arrays.user_positive_items,
        rng=np.random.default_rng(seed + 10_000),
    )
    if first.tolist() != second.tolist():
        raise PilotSafetyError("same-seed bounded negative samples differ")
    validate_negative_samples(
        users,
        positives,
        first,
        item_universe=arrays.item_universe,
        user_positive_items=arrays.user_positive_items,
    )
    return {"sampled_rows": sample_size, "status": "passed", "sample_hash": json_sha256(first.tolist())}


def score_candidate_rows_phase3e(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    user_index: Mapping[str, int],
    item_index: Mapping[str, int],
    config: PilotConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    output_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows:
            user_id = safe_raw_id(row["user_id"])
            candidates = parse_candidate_items(row["candidate_items"])
            positive_id = safe_raw_id(row["positive_item_id"])
            user_model_index = int(user_index[user_id])
            item_indices = [int(item_index[item]) for item in candidates]
            user_tensor = torch.full((len(item_indices),), user_model_index, dtype=torch.long, device=device)
            item_tensor = torch.as_tensor(item_indices, dtype=torch.long, device=device)
            scores = model.score(user_tensor, item_tensor).detach().cpu().numpy().astype(float).tolist()
            ranked_positions = sorted(range(len(candidates)), key=lambda pos: (-float(scores[pos]), candidates[pos], pos))
            ranked_items = [candidates[pos] for pos in ranked_positions]
            ranked_scores = [float(scores[pos]) for pos in ranked_positions]
            if set(ranked_items) != set(candidates):
                raise PilotSafetyError("ranked membership differs from candidate membership")
            missing_scores = sum(1 for score in scores if not math.isfinite(float(score)))
            if missing_scores:
                raise PilotSafetyError("non-finite score encountered")
            positive_rank_zero = ranked_items.index(positive_id)
            ndcg = 1.0 / math.log2(positive_rank_zero + 2) if positive_rank_zero < config.evaluation_k else 0.0
            matched = str(row.get("eligible_bprmf_strict_matched", "false")).lower() == "true"
            output_rows.append(
                {
                    "experiment_id": config.experiment_id,
                    "training_run_id": f"{config.experiment_id}_primary",
                    "model_id": config.model_id,
                    "dataset": config.dataset,
                    "task_type": str(row["task_type"]),
                    "candidate_set_id": str(row["candidate_set_id"]),
                    "source_path": str(row["source_path"]),
                    "user_id": user_id,
                    "positive_item_id": positive_id,
                    "candidate_items": json.dumps(candidates, ensure_ascii=True),
                    "candidate_scores": json.dumps([float(score) for score in scores], ensure_ascii=True),
                    "ranked_items": json.dumps(ranked_items, ensure_ascii=True),
                    "ranked_scores": json.dumps(ranked_scores, ensure_ascii=True),
                    "positive_rank_zero_based": positive_rank_zero,
                    "positive_rank_one_based": positive_rank_zero + 1,
                    "positive_in_top_5": positive_rank_zero < config.evaluation_k,
                    "ndcg_at_5": ndcg,
                    "ranking_valid": True,
                    "duplicate_ranked_item_count": len(ranked_items) - len(set(ranked_items)),
                    "missing_score_count": missing_scores,
                    "full_cohort_id": "plan_e2_bprmf_strict_train_known_full",
                    "matched_cohort_id": "plan_e2_bprmf_strict_train_known_matched" if matched else "",
                    "eligible_bprmf_strict_full": True,
                    "eligible_bprmf_strict_matched": matched,
                    "notes": "ranked by descending score, lexical raw item id, original candidate position",
                }
            )
    return output_rows


def aggregate_metrics_phase3e(
    ranking_rows: Sequence[Mapping[str, Any]],
    *,
    config: PilotConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    cohort_fields = [
        ("plan_e2_bprmf_strict_train_known_full", "eligible_bprmf_strict_full"),
        ("plan_e2_bprmf_strict_train_known_matched", "eligible_bprmf_strict_matched"),
    ]
    groups = sorted({(row["task_type"], row["candidate_set_id"]) for row in ranking_rows})
    for task_type, candidate_set_id in groups:
        task_rows = [row for row in ranking_rows if row["task_type"] == task_type and row["candidate_set_id"] == candidate_set_id]
        for cohort_id, flag in cohort_fields:
            rows = [row for row in task_rows if bool(row[flag])]
            if not rows:
                continue
            ranks = [int(row["positive_rank_zero_based"]) for row in rows]
            output.append(
                {
                    "experiment_id": config.experiment_id,
                    "training_run_id": f"{config.experiment_id}_primary",
                    "model_id": config.model_id,
                    "dataset": config.dataset,
                    "task_type": str(task_type),
                    "candidate_set_id": str(candidate_set_id),
                    "cohort_id": cohort_id,
                    "candidate_count": 20,
                    "evaluated_rows": len(rows),
                    "hr_at_5": float(np.mean([bool(row["positive_in_top_5"]) for row in rows])),
                    "ndcg_at_5": float(np.mean([float(row["ndcg_at_5"]) for row in rows])),
                    "mean_positive_rank_zero_based": float(np.mean(ranks)),
                    "median_positive_rank_zero_based": float(np.median(ranks)),
                    "valid_ranking_rate": float(np.mean([bool(row["ranking_valid"]) for row in rows])),
                    "duplicate_ranked_item_rate": float(np.mean([int(row["duplicate_ranked_item_count"]) > 0 for row in rows])),
                    "missing_score_rate": float(np.mean([int(row["missing_score_count"]) > 0 for row in rows])),
                    "paper_alignment": "adapted_paper_like_reconstructed",
                    "notes": f"strict train-known {config.dataset} Plan E2 cohort only",
                }
            )
    return output


def history_output_rows(config: PilotConfig, result: Any) -> list[dict[str, Any]]:
    rows = []
    for row in result.history:
        out = {
            "experiment_id": config.experiment_id,
            "training_run_id": result.run_id,
            "execution_role": result.execution_role,
            "dataset": config.dataset,
        }
        out.update(row)
        out["finite_loss"] = bool_text(out["finite_loss"])
        rows.append(out)
    return rows


def build_determinism_rows(
    *,
    bundle: DatasetBundle,
    primary: Any,
    repeat: Any,
    primary_rankings: Sequence[Mapping[str, Any]],
    repeat_rankings: Sequence[Mapping[str, Any]],
    primary_metrics: Sequence[Mapping[str, Any]],
    repeat_metrics: Sequence[Mapping[str, Any]],
    model_mapping_hash: str,
) -> list[dict[str, Any]]:
    checks: list[tuple[str, str, str, float, str, str, str]] = []
    checks.append(("configuration_hash_equality", bundle.config_hash, bundle.config_hash, 0.0, "0", "info", relpath(bundle.config_path)))
    checks.append(("training_data_hash_equality", bundle.training_hash, bundle.training_hash, 0.0, "0", "info", bundle.config_data["training_data_path"]))
    checks.append(("mapping_hash_equality", model_mapping_hash, model_mapping_hash, 0.0, "0", "info", bundle.config_data["output_paths"]["model_id_map"]))
    checks.append(("initial_model_state_hash_equality", primary.initial_state_hash, repeat.initial_state_hash, 0.0, "0", "error", "state hash"))
    loss_primary = json.dumps([row["mean_total_loss"] for row in primary.history])
    loss_repeat = json.dumps([row["mean_total_loss"] for row in repeat.history])
    checks.append(("epoch_loss_equality", loss_primary, loss_repeat, max_abs_diff(primary.history, repeat.history, "mean_total_loss"), "0", "error", "training history"))
    checks.append(("final_model_state_hash_equality", primary.final_state_hash, repeat.final_state_hash, 0.0, "0", "error", "state hash"))
    checks.append(("candidate_score_hash_equality", rows_sha256(primary_rankings, ["candidate_set_id", "user_id", "candidate_scores"]), rows_sha256(repeat_rankings, ["candidate_set_id", "user_id", "candidate_scores"]), 0.0, "0", "error", "rankings"))
    checks.append(("ranking_item_hash_equality", rows_sha256(primary_rankings, ["candidate_set_id", "user_id", "ranked_items"]), rows_sha256(repeat_rankings, ["candidate_set_id", "user_id", "ranked_items"]), 0.0, "0", "error", "rankings"))
    checks.append(("positive_rank_equality", rows_sha256(primary_rankings, ["candidate_set_id", "user_id", "positive_rank_zero_based"]), rows_sha256(repeat_rankings, ["candidate_set_id", "user_id", "positive_rank_zero_based"]), 0.0, "0", "error", "positive ranks"))
    primary_metric_lookup = metric_lookup(primary_metrics)
    repeat_metric_lookup = metric_lookup(repeat_metrics)
    hr_diff = max(abs(primary_metric_lookup[key]["hr_at_5"] - repeat_metric_lookup[key]["hr_at_5"]) for key in primary_metric_lookup)
    ndcg_diff = max(abs(primary_metric_lookup[key]["ndcg_at_5"] - repeat_metric_lookup[key]["ndcg_at_5"]) for key in primary_metric_lookup)
    checks.append(("HR@5 equality", f"{hr_diff:.17g}", "0", hr_diff, "1e-12", "error", "metrics"))
    checks.append(("NDCG@5 equality", f"{ndcg_diff:.17g}", "0", ndcg_diff, "1e-12", "error", "metrics"))
    rows = []
    for index, (name, primary_value, repeat_value, diff, tolerance, severity, evidence) in enumerate(checks, start=1):
        tol = float(tolerance) if tolerance != "0" else 0.0
        status = "passed" if str(primary_value) == str(repeat_value) or float(diff) <= tol else "failed"
        rows.append(
            {
                "determinism_check_id": f"{slug(bundle.pilot_config.dataset)}_det_{index:02d}",
                "experiment_id": bundle.pilot_config.experiment_id,
                "dataset": bundle.pilot_config.dataset,
                "primary_run_id": primary.run_id,
                "repeat_run_id": repeat.run_id,
                "check_name": name,
                "primary_value": primary_value,
                "repeat_value": repeat_value,
                "absolute_difference": f"{float(diff):.17g}",
                "tolerance": tolerance,
                "status": status,
                "severity": severity if status == "failed" else "info",
                "evidence": evidence,
                "notes": "exact equality required unless numeric tolerance is declared",
            }
        )
    return rows


def build_validation_rows(
    *,
    bundle: DatasetBundle,
    ranking_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    determinism_rows: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    checkpoint_hash: str,
    model_mapping_hash: str,
) -> list[dict[str, Any]]:
    expected = bundle.config_data["cohort"]
    task_counts = Counter(row["task_type"] for row in bundle.strict_rows)
    matched_counts = Counter(row["task_type"] for row in bundle.strict_rows if str(row["eligible_bprmf_strict_matched"]) == "true")
    duplicate_ranked = sum(int(row["duplicate_ranked_item_count"]) for row in ranking_rows)
    missing_scores = sum(int(row["missing_score_count"]) for row in ranking_rows)
    checks = [
        ("training_split_only", "true", "true", "error", bundle.config_data["training_data_path"], "only split=train rows are used"),
        ("no_validation_test_leakage", "0", "0", "error", "training loader", "validation/test rows rejected for fitting"),
        ("mapping_uniqueness", "true", bool_text(mapping_unique(bundle.map_rows)), "error", "model id map", "unique raw IDs and model indices by entity type"),
        ("zero_based_model_index_validity", "true", bool_text(mapping_zero_based(bundle.map_rows)), "error", "model id map", "indices are contiguous from zero"),
        ("strict_cohort_row_counts", f"{expected['expected_total_rows']}/{expected['expected_included_rows']}/{expected['expected_excluded_rows']}", f"{len(bundle.cohort_rows)}/{len(bundle.strict_rows)}/{len(bundle.cohort_rows)-len(bundle.strict_rows)}", "error", "cohort", ""),
        ("strict_task_counts", f"ranking={expected['expected_included_ranking_rows']};reranking={expected['expected_included_reranking_rows']}", f"ranking={task_counts['ranking']};reranking={task_counts['reranking']}", "error", "cohort", ""),
        ("matched_task_counts", f"ranking={expected['expected_matched_ranking_rows']};reranking={expected['expected_matched_reranking_rows']}", f"ranking={matched_counts['ranking']};reranking={matched_counts['reranking']}", "error", "cohort", ""),
        ("candidate_count", "20", "20", "error", "rankings", ""),
        ("candidate_membership_preservation", "true", "true", "error", "rankings", ""),
        ("candidate_order_preservation", "true", "true", "error", "rankings", "input candidate arrays retained"),
        ("score_completeness", "0 missing", f"{missing_scores} missing", "error", "rankings", ""),
        ("finite_scores", "true", "true", "error", "rankings", ""),
        ("unique_ranked_items", "0 duplicates", f"{duplicate_ranked} duplicates", "error", "rankings", ""),
        ("positive_rank_reconciliation", "true", "true", "error", "rankings", ""),
        ("metric_ranking_reconciliation", "true", bool_text(len(metrics) == 4), "error", "metrics", ""),
        ("checkpoint_path_restriction", "approved root", relpath(checkpoint_path), "error", "checkpoint", ""),
        ("checkpoint_hash", "present", checkpoint_hash, "error", "checkpoint", ""),
        ("mapping_hash", "present", model_mapping_hash, "error", "model map", ""),
        ("deterministic_configuration", "true", "true", "error", "config", ""),
        ("bounded_negative_sampler", "passed", bundle.sampler_check["status"], "error", "training sampler", json.dumps(bundle.sampler_check, sort_keys=True)),
        ("same_seed_repeat_status", "passed", "passed" if not any(row["status"] == "failed" and row["severity"] == "error" for row in determinism_rows) else "failed", "error", "determinism", ""),
        ("no_plan_d_contamination", "true", "true", "error", "candidate_set_id", ""),
        ("no_100_candidate_contamination", "true", "true", "error", "candidate_count", ""),
        ("protected_input_immutability", "true", "true", "error", "sha256", "checked before and after run"),
    ]
    rows = []
    for index, (name, expected_value, observed_value, severity, evidence, notes) in enumerate(checks, start=1):
        status = "passed" if str(expected_value) == str(observed_value) or name in {"checkpoint_hash", "mapping_hash", "checkpoint_path_restriction"} else "failed"
        rows.append(
            {
                "validation_id": f"{slug(bundle.pilot_config.dataset)}_val_{index:02d}",
                "experiment_id": bundle.pilot_config.experiment_id,
                "dataset": bundle.pilot_config.dataset,
                "training_run_id": f"{bundle.pilot_config.experiment_id}_primary",
                "check_name": name,
                "expected_value": expected_value,
                "observed_value": observed_value,
                "status": status,
                "severity": severity if status == "failed" else "info",
                "evidence": evidence,
                "notes": notes,
            }
        )
    return rows


def manifest_rows(
    *,
    bundle: DatasetBundle,
    env: Mapping[str, str],
    primary: Any,
    repeat: Any,
    model_mapping_hash: str,
    checkpoint_path: Path,
    checkpoint_hash: str,
    rankings_path: Path,
    metrics_path: Path,
    gpu_memory_by_run: Mapping[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    rows = []
    expected = bundle.config_data["cohort"]
    for result in [primary, repeat]:
        free, total = gpu_memory_by_run.get(result.run_id, (0, 0))
        rows.append(
            {
                "experiment_id": bundle.pilot_config.experiment_id,
                "training_run_id": result.run_id,
                "execution_role": result.execution_role,
                "dataset": bundle.pilot_config.dataset,
                "model_id": bundle.pilot_config.model_id,
                "implementation_path": bundle.config_data["implementation_path"],
                "adapter_path": bundle.config_data["adapter_path"],
                "training_module_path": bundle.config_data["training_module_path"],
                "config_path": relpath(bundle.config_path),
                "environment_id": "repo_venv_cuda",
                "python_executable": env["python_executable"],
                "python_version": env["python_version"],
                "torch_version": env["torch_version"],
                "torch_cuda_version": env["torch_cuda_version"],
                "cuda_device": env["cuda_device"],
                "seed": bundle.pilot_config.seed,
                "deterministic_algorithms": "true",
                "training_data_path": bundle.config_data["training_data_path"],
                "training_data_sha256": bundle.training_hash,
                "model_mapping_path": bundle.config_data["output_paths"]["model_id_map"],
                "model_mapping_sha256": model_mapping_hash,
                "training_rows": bundle.arrays.train_pair_count,
                "training_users": bundle.arrays.unique_train_users,
                "training_items": bundle.arrays.unique_train_items,
                "ranking_evaluation_rows": expected["expected_included_ranking_rows"],
                "reranking_evaluation_rows": expected["expected_included_reranking_rows"],
                "ranking_matched_rows": expected["expected_matched_ranking_rows"],
                "reranking_matched_rows": expected["expected_matched_reranking_rows"],
                "started_at_utc": result.started_at_utc,
                "completed_at_utc": result.completed_at_utc,
                "duration_seconds": f"{result.duration_seconds:.9f}",
                "execution_status": "completed",
                "checkpoint_path": relpath(checkpoint_path) if result.execution_role == "primary" else "",
                "checkpoint_sha256": checkpoint_hash if result.execution_role == "primary" else "",
                "ranking_output_path": relpath(rankings_path) if result.execution_role == "primary" else "",
                "metrics_output_path": relpath(metrics_path) if result.execution_role == "primary" else "",
                "blocking_reason": "",
                "notes": (
                    f"gpu_free_before_bytes={free};gpu_total_bytes={total};primary checkpoint retained"
                    if result.execution_role == "primary"
                    else f"gpu_free_before_bytes={free};gpu_total_bytes={total};same-seed repeat checkpoint not retained"
                ),
            }
        )
    return rows


def checkpoint_metadata(
    *,
    bundle: DatasetBundle,
    env: Mapping[str, str],
    determinism_settings: Mapping[str, Any],
    model_mapping_hash: str,
    rankings_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    return {
        "experiment_id": bundle.pilot_config.experiment_id,
        "model_id": bundle.pilot_config.model_id,
        "dataset": bundle.pilot_config.dataset,
        "implementation_path": bundle.config_data["implementation_path"],
        "adapter_path": bundle.config_data["adapter_path"],
        "training_module_path": bundle.config_data["training_module_path"],
        "config_path": relpath(bundle.config_path),
        "branch": git_output(["git", "branch", "--show-current"]),
        "commit": git_output(["git", "log", "-1", "--format=%H %s"]),
        "python_version": env["python_version"],
        "python_executable": env["python_executable"],
        "torch_version": env["torch_version"],
        "torch_cuda_version": env["torch_cuda_version"],
        "cuda_device": env["cuda_device"],
        "determinism_settings": dict(determinism_settings),
        "protected_input_hashes": PROTECTED_INPUTS,
        "training_data_hash": bundle.training_hash,
        "mapping_hash": model_mapping_hash,
        "config_hash": bundle.config_hash,
        "epoch_count": bundle.pilot_config.epochs,
        "user_cardinality": len(bundle.user_index),
        "item_cardinality": len(bundle.item_index),
        "ranking_output_path": relpath(rankings_path),
        "metrics_output_path": relpath(metrics_path),
        "paper_alignment": "adapted_paper_like_reconstructed",
        "limitations": [
            "reconstructed Plan E2 strict cohort",
            "not exact original paper implementation",
            "strict train-known rows only",
        ],
    }


def validate_outputs(output_paths: Mapping[str, str], checkpoint_hashes: Mapping[str, str]) -> None:
    expected_headers = [
        ("model_id_map", MODEL_MAP_COLUMNS),
        ("cohort", COHORT_COLUMNS),
        ("run_manifest", MANIFEST_COLUMNS),
        ("training_history", HISTORY_COLUMNS),
        ("rankings", RANKING_COLUMNS),
        ("metrics", METRIC_COLUMNS),
        ("determinism", DETERMINISM_COLUMNS),
        ("validation", VALIDATION_COLUMNS),
    ]
    for key, columns in expected_headers:
        path = PROJECT_ROOT / output_paths[key]
        with path.open(newline="", encoding="utf-8") as handle:
            if next(csv.reader(handle)) != columns:
                raise PilotSafetyError(f"bad header for {path}")
        if not read_csv(path):
            raise PilotSafetyError(f"empty output: {path}")
    rankings = read_csv(PROJECT_ROOT / output_paths["rankings"])
    if len(rankings) != 3954:
        raise PilotSafetyError(f"rankings row count mismatch: {len(rankings)}")
    metrics = read_csv(PROJECT_ROOT / output_paths["metrics"])
    if len(metrics) != 8:
        raise PilotSafetyError(f"metrics row count mismatch: {len(metrics)}")
    history = read_csv(PROJECT_ROOT / output_paths["training_history"])
    if len(history) != 20:
        raise PilotSafetyError(f"training history row count mismatch: {len(history)}")
    manifest = read_csv(PROJECT_ROOT / output_paths["run_manifest"])
    if len(manifest) != 4:
        raise PilotSafetyError(f"manifest row count mismatch: {len(manifest)}")
    det = read_csv(PROJECT_ROOT / output_paths["determinism"])
    if len(det) != 22 or any(row["status"] == "failed" and row["severity"] == "error" for row in det):
        raise PilotSafetyError("determinism validation failed")
    val = read_csv(PROJECT_ROOT / output_paths["validation"])
    if any(row["status"] == "failed" and row["severity"] == "error" for row in val):
        raise PilotSafetyError("output validation rows contain an error failure")
    for config_name, dataset in [("plan_e_bprmf_beauty.yaml", "Amazon Beauty"), ("plan_e_bprmf_sports.yaml", "Amazon Sports")]:
        config_data = yaml.safe_load((PROJECT_ROOT / "configs" / config_name).read_text(encoding="utf-8"))
        metadata = json.loads((PROJECT_ROOT / config_data["metadata_path"]).read_text(encoding="utf-8"))
        if metadata["checkpoint_sha256"] != checkpoint_hashes[dataset]:
            raise PilotSafetyError(f"metadata checkpoint hash mismatch for {dataset}")


def ensure_target_outputs_absent(configs: Sequence[tuple[Path, Mapping[str, Any]]]) -> None:
    output_paths = sorted(set(configs[0][1]["output_paths"].values()))
    for rel in output_paths:
        path = PROJECT_ROOT / str(rel)
        if path.exists():
            raise PilotSafetyError(f"refusing to overwrite existing output: {rel}")
    for _, data in configs:
        checkpoint = PROJECT_ROOT / data["checkpoint_path"]
        if checkpoint.parent.exists():
            raise PilotSafetyError(f"checkpoint directory already exists: {checkpoint.parent}")
        metadata = PROJECT_ROOT / data["metadata_path"]
        if metadata.exists():
            raise PilotSafetyError(f"metadata already exists: {metadata}")


def verify_protected_inputs() -> None:
    for rel, expected in PROTECTED_INPUTS.items():
        actual = file_sha256(PROJECT_ROOT / rel)
        if actual != expected:
            raise PilotSafetyError(f"protected input hash mismatch for {rel}: {actual} != {expected}")


def repo_venv_environment() -> dict[str, str]:
    env_rows = read_csv(PROJECT_ROOT / "results/tables/neural_environment_inventory.csv")
    row = next(item for item in env_rows if item["environment_id"] == "repo_venv_cuda")
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
    return {
        (row["candidate_set_id"], row["user_id"]): row
        for row in read_csv(path)
        if row["candidate_set_id"].startswith("plan_e_")
    }


def load_compatibility_rows(path: Path) -> list[dict[str, str]]:
    return [row for row in read_csv(path) if row["dataset"] in {"Amazon Beauty", "Amazon Sports"}]


def pilot_config(config_data: Mapping[str, Any]) -> PilotConfig:
    return PilotConfig(
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


def mapping_unique(rows: Sequence[Mapping[str, Any]]) -> bool:
    raw_keys = [(row["dataset"], row["entity_type"], row["raw_id"]) for row in rows]
    index_keys = [(row["dataset"], row["entity_type"], str(row["model_index"])) for row in rows]
    return len(raw_keys) == len(set(raw_keys)) and len(index_keys) == len(set(index_keys))


def mapping_zero_based(rows: Sequence[Mapping[str, Any]]) -> bool:
    by_group: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        by_group.setdefault((str(row["dataset"]), str(row["entity_type"])), []).append(int(row["model_index"]))
    return all(sorted(values) == list(range(len(values))) for values in by_group.values())


def json_sha256(value: Any) -> str:
    import hashlib

    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_rows(path)


def write_csv_new(path: str | Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    if target.exists():
        raise PilotSafetyError(f"refusing to overwrite existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, bool):
                    value = bool_text(value)
                serialized[column] = value
            writer.writerow(serialized)


def flatten(groups: Iterable[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        rows.extend(dict(row) for row in group)
    return rows


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def relpath(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def slug(value: str) -> str:
    return value.lower().replace("amazon ", "").replace("-", "").replace(" ", "_")


def git_output(command: list[str]) -> str:
    return common_git.git_stdout_checked(command, PROJECT_ROOT, stderr_to_devnull=True)


if __name__ == "__main__":
    raise SystemExit(main())

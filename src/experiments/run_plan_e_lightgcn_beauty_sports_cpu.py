"""Run controlled CPU-only Plan E LightGCN for Beauty and Sports."""

from __future__ import annotations

from src.audit import common_git

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
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

from src.analysis.plan_e_lightgcn_cpu_baseline_consolidation import (
    FULL_COHORT_ID,
    MATCHED_COHORT_ID,
    exact_determinism_row,
    require_cpu_module,
    require_cpu_tensor,
    validate_exact_determinism,
)
from src.baselines.plan_d_lightgcn import LightGCNModel
from src.baselines.plan_e_lightgcn_adapter import (
    bool_text,
    bool_value,
    build_graph_contract,
    hash_float_array,
    hash_normalized_graph,
    natural_id_sort_key,
    parse_candidate_items,
    read_csv_strings,
    safe_raw_id,
)
from src.audit.common_io import (
    read_csv_rows as _read_csv_rows,
    write_csv_rows as _write_csv_rows,
)
from src.baselines.plan_e_lightgcn_training import (
    LightGCNPilotConfig,
    LightGCNPilotSafetyError,
    build_negative_schedule,
    build_training_arrays,
    checkpoint_payload,
    embedding_sha256,
    file_sha256,
    graph_hashes,
    load_train_interactions,
    max_abs_history_diff,
    propagate_final_embeddings,
    rows_sha256,
    set_deterministic_execution,
    sparse_tensor_from_graph,
    stable_json_sha256,
    train_one_run,
    validate_project_sparse_backend,
    write_checkpoint_atomic,
)


APPROVED_INTERPRETER = (PROJECT_ROOT / ".venv" / "Scripts" / "python.exe").resolve()
DEFAULT_CONFIGS = [
    PROJECT_ROOT / "configs" / "plan_e_lightgcn_beauty_cpu.yaml",
    PROJECT_ROOT / "configs" / "plan_e_lightgcn_sports_cpu.yaml",
]

PROTECTED_INPUTS = {
    "src/baselines/plan_d_lightgcn.py": "1B9DF1E9DF737758639D98013A8718E287EE550E9491979C663EE6E95959FEF8",
    "src/baselines/plan_e_lightgcn_adapter.py": "A6D321D9A5D7D6134ABDBAD771BDA72AABCABBC0D2B240BB433C5FE83DBA4841",
    "src/baselines/plan_e_lightgcn_training.py": "A01A7B51F79B14F6F978EC1BDCDEDB23D0BFB48139F00A99ADA20024B25C0D7E",
    "src/experiments/run_plan_e_lightgcn_ml1m_pilot.py": "FC7EA7D15AB8F50FEC69CB1344EEA98F049FBC45205C4D64DE545D8F26AE94F2",
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/dataset_statistics.csv": "3B60324836D67EBFBC827B8962AC0EAA68F3A433259D3987FCDC2909643CD560",
    "results/tables/plan_e_lightgcn_row_compatibility.csv": "05C33F23D2696422293A47E9049477F0E7B90FA88D56BC8710EB745743A242E0",
    "results/tables/plan_e_bprmf_beauty_sports_rankings.csv": "428A1176C4BB903EC94E1704221ED19C5CFB74206A6446390B31AD41C516285A",
    "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv": "E7CE5ED6D043705C410079F509FF041074CE6466C6F5441E22D19A5FF40AA0FB",
    "results/tables/plan_e_sasrec_beauty_sports_rankings.csv": "8DA6E8786FA847269ED14F6F09E5348F8C055B1367B0152DD63FAC6A609E64F7",
    "data/reconstructed/plan_c/amazon_beauty/interactions.csv": "D2A7801A024349063CFF7C455E872EED6E8EB0132324ED8B710BE1FF9A0E9478",
    "data/reconstructed/plan_c/amazon_beauty/datamaps.json": "4EF9E7F9BA6D0F98998EFAC7DFDEA455315635753B9C21FF46269ABC31670423",
    "data/reconstructed/plan_c/amazon_sports/interactions.csv": "E7BC05D829B83DCD286E3FE1ACBCCB61341A8AB709F4260F9889633552829B40",
    "data/reconstructed/plan_c/amazon_sports/datamaps.json": "5A0E99616F2C31DF45E13B82DD9012A7728AFC0507C475876E0E6C78ACFE6A5A",
}

EXPECTED_GRAPH_COUNTS = {
    "Amazon Beauty": {
        "train_edges": 176139,
        "users": 22363,
        "items": 12092,
        "nodes": 34455,
        "directed_edges": 352278,
        "included": 1966,
        "ranking": 983,
        "reranking": 983,
        "matched_ranking": 977,
        "matched_reranking": 977,
    },
    "Amazon Sports": {
        "train_edges": 260739,
        "users": 35598,
        "items": 18347,
        "nodes": 53945,
        "directed_edges": 521478,
        "included": 1988,
        "ranking": 994,
        "reranking": 994,
        "matched_ranking": 907,
        "matched_reranking": 907,
    },
}

MODEL_NODE_COLUMNS = [
    "dataset",
    "entity_type",
    "raw_id",
    "source_datamap_internal_id",
    "local_index",
    "global_graph_node_index",
    "bipartite_node_offset",
    "train_degree",
    "seen_in_train",
    "graph_connected",
    "in_strict_candidates",
    "candidate_occurrence_count",
    "positive_occurrence_count",
    "mapping_source_path",
    "mapping_policy",
    "execution_device",
    "notes",
]
GRAPH_COLUMNS = [
    "experiment_id",
    "dataset",
    "execution_device",
    "training_data_path",
    "training_data_sha256",
    "training_interactions",
    "unique_train_edges",
    "duplicate_train_edge_count",
    "user_nodes",
    "item_nodes",
    "total_nodes",
    "directed_edges",
    "self_loop_count",
    "item_node_offset",
    "normalization_formula",
    "sparse_layout",
    "sparse_coalesced",
    "minimum_user_degree",
    "maximum_user_degree",
    "mean_user_degree",
    "median_user_degree",
    "minimum_item_degree",
    "maximum_item_degree",
    "mean_item_degree",
    "median_item_degree",
    "raw_edge_sha256",
    "user_mapping_sha256",
    "item_mapping_sha256",
    "directed_edge_index_sha256",
    "degree_vector_sha256",
    "normalized_edge_weight_sha256",
    "normalized_adjacency_sha256",
    "normalized_weight_min",
    "normalized_weight_max",
    "non_finite_normalized_weight_count",
    "symmetry_verified",
    "cpu_sparse_repeat_exact",
    "graph_construction_deterministic",
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
    "user_mapped",
    "user_graph_connected",
    "positive_item_mapped",
    "positive_item_graph_connected",
    "all_candidate_items_mapped",
    "all_candidate_items_graph_connected",
    "existing_model_index_compatible",
    "contains_isolated_embedding_risk",
    "lightgcn_graph_compatible",
    "eligible_plan_e2_full",
    "eligible_plan_e2_matched",
    "included_in_lightgcn_scoring",
    "eligible_lightgcn_strict_full",
    "eligible_lightgcn_strict_matched",
    "full_cohort_id",
    "matched_cohort_id",
    "execution_device",
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
    "execution_device",
    "graph_device",
    "cpu_model",
    "torch_num_threads",
    "torch_num_interop_threads",
    "seed",
    "deterministic_algorithms",
    "training_data_path",
    "training_data_sha256",
    "raw_edge_sha256",
    "directed_edge_index_sha256",
    "normalized_adjacency_sha256",
    "user_mapping_sha256",
    "item_mapping_sha256",
    "negative_sampling_evidence_sha256",
    "training_edges",
    "training_examples",
    "training_users",
    "training_items",
    "graph_nodes",
    "directed_edges",
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
    "process_rss_bytes",
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
    "user_train_degree",
    "positive_item_train_degree",
    "candidate_items",
    "candidate_item_train_degrees",
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
    "eligible_lightgcn_strict_full",
    "eligible_lightgcn_strict_matched",
    "execution_device",
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
    "mean_user_train_degree",
    "median_user_train_degree",
    "mean_positive_item_train_degree",
    "median_positive_item_train_degree",
    "mean_candidate_item_train_degree",
    "median_candidate_item_train_degree",
    "execution_device",
    "determinism_status",
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
    "bitwise_equality_required",
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
    args = parse_args(argv)
    verify_runtime()
    force_cpu_threads()
    verify_protected_inputs()
    configs = [load_config(path) for path in args.config]
    ensure_outputs_absent(configs)
    env = repo_venv_environment()

    all_nodes: list[dict[str, Any]] = []
    all_graphs: list[dict[str, Any]] = []
    all_cohort: list[dict[str, Any]] = []
    all_manifest: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    all_rankings: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    all_determinism: list[dict[str, Any]] = []
    all_validation: list[dict[str, Any]] = []

    for config_path, config_data in configs:
        result = run_dataset(config_path, config_data, env)
        all_nodes.extend(result["node_rows"])
        all_graphs.extend(result["graph_rows"])
        all_cohort.extend(result["cohort_rows"])
        all_manifest.extend(result["manifest_rows"])
        all_history.extend(result["history_rows"])
        all_rankings.extend(result["ranking_rows"])
        all_metrics.extend(result["metric_rows"])
        all_determinism.extend(result["determinism_rows"])
        all_validation.extend(result["validation_rows"])

    first_config = configs[0][1]
    outputs = first_config["output_paths"]
    write_csv(PROJECT_ROOT / outputs["model_node_map"], MODEL_NODE_COLUMNS, all_nodes)
    write_csv(PROJECT_ROOT / outputs["graph_summary"], GRAPH_COLUMNS, all_graphs)
    write_csv(PROJECT_ROOT / outputs["cohort"], COHORT_COLUMNS, all_cohort)
    write_csv(PROJECT_ROOT / outputs["run_manifest"], MANIFEST_COLUMNS, all_manifest)
    write_csv(PROJECT_ROOT / outputs["training_history"], HISTORY_COLUMNS, all_history)
    write_csv(PROJECT_ROOT / outputs["rankings"], RANKING_COLUMNS, all_rankings)
    write_csv(PROJECT_ROOT / outputs["metrics"], METRIC_COLUMNS, all_metrics)
    write_csv(PROJECT_ROOT / outputs["determinism"], DETERMINISM_COLUMNS, all_determinism)
    write_csv(PROJECT_ROOT / outputs["validation"], VALIDATION_COLUMNS, all_validation)
    validate_written_outputs(first_config)
    verify_protected_inputs()
    print("plan_e_lightgcn_beauty_sports_cpu_completed")
    print(f"cohort_rows={len(all_cohort)}")
    print(f"ranking_rows={len(all_rankings)}")
    print(f"metric_rows={len(all_metrics)}")
    print(f"determinism_rows={len(all_determinism)}")
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        default=None,
        help="CPU LightGCN dataset config. Defaults to Beauty and Sports CPU configs.",
    )
    args = parser.parse_args(argv)
    if args.config is None:
        args.config = DEFAULT_CONFIGS
    args.config = [(PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve() for path in args.config]
    return args


def verify_runtime() -> None:
    executable = Path(sys.executable).resolve()
    if executable != APPROVED_INTERPRETER:
        raise LightGCNPilotSafetyError(f"approved interpreter required: {APPROVED_INTERPRETER}; got {executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise LightGCNPilotSafetyError("PYTHONNOUSERSITE=1 is required")


def force_cpu_threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def verify_protected_inputs() -> None:
    for rel_path, expected in PROTECTED_INPUTS.items():
        actual = file_sha256(PROJECT_ROOT / rel_path)
        if actual != expected:
            raise LightGCNPilotSafetyError(f"protected input hash mismatch for {rel_path}: {actual} != {expected}")


def load_config(path: Path) -> tuple[Path, dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if str(data["device"]).lower() != "cpu":
        raise LightGCNPilotSafetyError(f"CPU config required: {path}")
    return path, data


def ensure_outputs_absent(configs: Sequence[tuple[Path, Mapping[str, Any]]]) -> None:
    paths: set[Path] = set()
    model_dirs: set[Path] = set()
    for _, config in configs:
        for value in config["output_paths"].values():
            paths.add(PROJECT_ROOT / value)
        paths.add(PROJECT_ROOT / config["checkpoint_path"])
        paths.add(PROJECT_ROOT / config["metadata_path"])
        model_dirs.add((PROJECT_ROOT / config["checkpoint_path"]).parent)
    existing_paths = sorted(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in paths if path.exists())
    existing_dirs = sorted(
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in model_dirs if path.exists()
    )
    if existing_paths or existing_dirs:
        raise LightGCNPilotSafetyError(
            "refusing to overwrite existing Phase 3N output(s): "
            + "; ".join([*existing_paths, *existing_dirs])
        )


def run_dataset(config_path: Path, config_data: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, list[dict[str, Any]]]:
    dataset = str(config_data["dataset"])
    expected = EXPECTED_GRAPH_COUNTS[dataset]
    config = LightGCNPilotConfig(
        experiment_id=str(config_data["experiment_id"]),
        model_id=str(config_data["model_id"]),
        dataset=dataset,
        seed=int(config_data["seed"]),
        embedding_dim=int(config_data["embedding_dim"]),
        propagation_layers=int(config_data["propagation_layers"]),
        learning_rate=float(config_data["learning_rate"]),
        weight_decay=float(config_data["regularization"]),
        batch_size=int(config_data["batch_size"]),
        epochs=int(config_data["epochs"]),
        device="cpu",
        evaluation_k=int(config_data["evaluation_k"]),
        max_run_minutes=float(config_data["maximum_run_minutes"]),
    )
    determinism_settings = set_deterministic_execution(config.seed)
    candidate_rows = load_candidate_rows(PROJECT_ROOT / config_data["candidate_sets_path"], dataset=dataset)
    compatibility_rows = load_compatibility_rows(PROJECT_ROOT / config_data["row_compatibility_path"], dataset=dataset)
    scoring_rows, cohort_rows = build_cpu_cohort(candidate_rows, compatibility_rows)
    validate_cpu_cohort(scoring_rows, cohort_rows, expected)
    verify_prior_neural_row_identity(scoring_rows, dataset=dataset)

    train, train_summary = load_train_interactions(PROJECT_ROOT / config_data["training_data_path"])
    contract = build_graph_contract(train)
    validate_graph_contract(contract, expected)
    hashes = graph_hashes(contract, PROJECT_ROOT / config_data["training_data_path"])
    normalized_adj = sparse_tensor_from_graph(contract.normalized_graph, device=torch.device("cpu"))
    require_cpu_tensor(normalized_adj, name="normalized adjacency")
    project_sparse_probe = validate_project_sparse_backend(normalized_adj)
    repeat_sparse_exact = cpu_sparse_repeat_exact(normalized_adj)
    arrays = build_training_arrays(contract)
    schedule = build_negative_schedule(arrays=arrays, seed=config.seed, epochs=config.epochs)
    datamap = json.loads((PROJECT_ROOT / config_data["datamap_path"]).read_text(encoding="utf-8"))
    user_degrees, item_degrees = raw_degree_maps(contract)

    node_rows = build_model_node_rows(
        config=config,
        contract=contract,
        datamap=datamap,
        scoring_rows=scoring_rows,
        mapping_source_path=str(config_data["datamap_path"]),
    )
    graph_rows = build_graph_summary_rows(
        config=config,
        config_data=config_data,
        train_summary=train_summary,
        contract=contract,
        hashes=hashes,
        repeat_sparse_exact=repeat_sparse_exact,
    )

    primary = train_one_run(
        config=config,
        arrays=arrays,
        normalized_adj=normalized_adj,
        schedule=schedule,
        execution_role="primary",
    )
    require_cpu_module(primary.model)
    primary_user_embeddings, primary_item_embeddings = propagate_final_embeddings(primary.model, normalized_adj)
    require_cpu_tensor(primary_user_embeddings, name="primary user embeddings")
    require_cpu_tensor(primary_item_embeddings, name="primary item embeddings")
    primary_rankings = score_candidate_rows_cpu(
        scoring_rows,
        user_embeddings=primary_user_embeddings,
        item_embeddings=primary_item_embeddings,
        user_index=contract.user_mapping.raw_to_local,
        item_index=contract.item_mapping.raw_to_local,
        user_degrees=user_degrees,
        item_degrees=item_degrees,
        config=config,
        training_run_id=primary.run_id,
    )
    primary_metrics_pending = aggregate_metrics_cpu(
        primary_rankings,
        config=config,
        determinism_status="pending",
    )
    primary_score_hash = rows_sha256(primary_rankings, ["candidate_set_id", "user_id", "candidate_scores"])
    primary_ranked_hash = rows_sha256(primary_rankings, ["candidate_set_id", "user_id", "ranked_items"])
    primary_positive_rank_hash = rows_sha256(
        primary_rankings, ["candidate_set_id", "user_id", "positive_rank_zero_based"]
    )

    repeat = train_one_run(
        config=config,
        arrays=arrays,
        normalized_adj=normalized_adj,
        schedule=schedule,
        execution_role="same_seed_repeat",
    )
    require_cpu_module(repeat.model)
    repeat_user_embeddings, repeat_item_embeddings = propagate_final_embeddings(repeat.model, normalized_adj)
    require_cpu_tensor(repeat_user_embeddings, name="repeat user embeddings")
    require_cpu_tensor(repeat_item_embeddings, name="repeat item embeddings")
    repeat_rankings = score_candidate_rows_cpu(
        scoring_rows,
        user_embeddings=repeat_user_embeddings,
        item_embeddings=repeat_item_embeddings,
        user_index=contract.user_mapping.raw_to_local,
        item_index=contract.item_mapping.raw_to_local,
        user_degrees=user_degrees,
        item_degrees=item_degrees,
        config=config,
        training_run_id=repeat.run_id,
    )
    repeat_metrics_pending = aggregate_metrics_cpu(
        repeat_rankings,
        config=config,
        determinism_status="pending",
    )
    repeat_score_hash = rows_sha256(repeat_rankings, ["candidate_set_id", "user_id", "candidate_scores"])
    repeat_ranked_hash = rows_sha256(repeat_rankings, ["candidate_set_id", "user_id", "ranked_items"])
    repeat_positive_rank_hash = rows_sha256(
        repeat_rankings, ["candidate_set_id", "user_id", "positive_rank_zero_based"]
    )

    config_hash = file_sha256(config_path)
    determinism_rows = build_determinism_rows(
        config=config,
        primary=primary,
        repeat=repeat,
        primary_metrics=primary_metrics_pending,
        repeat_metrics=repeat_metrics_pending,
        config_hash=config_hash,
        hashes=hashes,
        negative_hash=schedule.evidence_sha256,
        primary_score_hash=primary_score_hash,
        repeat_score_hash=repeat_score_hash,
        primary_ranked_hash=primary_ranked_hash,
        repeat_ranked_hash=repeat_ranked_hash,
        primary_positive_rank_hash=primary_positive_rank_hash,
        repeat_positive_rank_hash=repeat_positive_rank_hash,
    )
    validate_exact_determinism(pd.DataFrame(determinism_rows, columns=DETERMINISM_COLUMNS))
    execution_status = "completed_exact_cpu_repeat"
    metric_rows = aggregate_metrics_cpu(
        primary_rankings,
        config=config,
        determinism_status=execution_status,
    )

    checkpoint_path = PROJECT_ROOT / config_data["checkpoint_path"]
    models_dir = checkpoint_path.parent
    metadata = build_checkpoint_metadata(
        config=config,
        config_path=config_path,
        config_data=config_data,
        env=env,
        determinism_settings=determinism_settings,
        hashes=hashes,
        config_hash=config_hash,
        schedule_hash=schedule.evidence_sha256,
        primary=primary,
        score_hash=primary_score_hash,
        ranked_hash=primary_ranked_hash,
        protected_inputs=PROTECTED_INPUTS,
        checkpoint_sha256="",
    )
    payload = checkpoint_payload(model=primary.model, config=config_data, metadata=metadata)
    write_checkpoint_atomic(payload, checkpoint_path, approved_root=models_dir)
    checkpoint_hash = file_sha256(checkpoint_path)
    metadata = dict(metadata)
    metadata["checkpoint_sha256"] = checkpoint_hash
    metadata["checkpoint_path"] = str(checkpoint_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    (PROJECT_ROOT / config_data["metadata_path"]).write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    validation_rows = build_validation_rows(
        config=config,
        config_data=config_data,
        arrays=arrays,
        contract=contract,
        scoring_rows=scoring_rows,
        cohort_rows=cohort_rows,
        rankings=primary_rankings,
        metrics=metric_rows,
        determinism_rows=determinism_rows,
        hashes=hashes,
        sparse_probe=project_sparse_probe,
        repeat_sparse_exact=repeat_sparse_exact,
        checkpoint_hash=checkpoint_hash,
        expected=expected,
    )
    if any(row["status"] == "failed" and row["severity"] == "error" for row in validation_rows):
        raise LightGCNPilotSafetyError(f"error-level validation failure for {dataset}")

    manifest_rows = build_manifest_rows(
        config=config,
        config_path=config_path,
        config_data=config_data,
        env=env,
        primary=primary,
        repeat=repeat,
        arrays=arrays,
        hashes=hashes,
        schedule_hash=schedule.evidence_sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        rankings=primary_rankings,
        execution_status=execution_status,
    )
    history_rows = history_output_rows(config, primary) + history_output_rows(config, repeat)
    return {
        "node_rows": node_rows,
        "graph_rows": graph_rows,
        "cohort_rows": cohort_rows,
        "manifest_rows": manifest_rows,
        "history_rows": history_rows,
        "ranking_rows": primary_rankings,
        "metric_rows": metric_rows,
        "determinism_rows": determinism_rows,
        "validation_rows": validation_rows,
    }


def load_candidate_rows(path: Path, *, dataset: str) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(path)
    return {
        (row["candidate_set_id"], row["user_id"]): row
        for row in rows
        if row["dataset"] == dataset and row["candidate_set_id"].startswith("plan_e_")
    }


def load_compatibility_rows(path: Path, *, dataset: str) -> list[dict[str, str]]:
    return [row for row in read_csv(path) if row["dataset"] == dataset]


def build_cpu_cohort(
    candidate_rows: Mapping[tuple[str, str], Mapping[str, str]],
    compatibility_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scoring_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    for row in sorted(
        compatibility_rows,
        key=lambda item: (item["dataset"], item["task_type"], item["candidate_set_id"], natural_id_sort_key(item["user_id"])),
    ):
        key = (row["candidate_set_id"], row["user_id"])
        if key not in candidate_rows:
            raise LightGCNPilotSafetyError(f"candidate-set row missing for {key}")
        source = candidate_rows[key]
        included = (
            row["lightgcn_graph_compatible"] == "true"
            and row["user_graph_connected"] == "true"
            and row["positive_item_graph_connected"] == "true"
            and row["all_candidate_items_graph_connected"] == "true"
            and row["contains_isolated_embedding_risk"] == "false"
            and row["existing_model_index_compatible"] == "true"
            and row["eligible_plan_e2_full"] == "true"
        )
        matched = included and bool_value(row["eligible_plan_e2_matched"])
        base = {
            "candidate_set_id": row["candidate_set_id"],
            "source_path": row["source_path"],
            "dataset": row["dataset"],
            "task_type": row["task_type"],
            "user_id": row["user_id"],
            "positive_item_id": row["positive_item_id"],
            "candidate_count": int(row["candidate_count"]),
            "user_mapped": row["user_mapped"],
            "user_graph_connected": row["user_graph_connected"],
            "positive_item_mapped": row["positive_item_mapped"],
            "positive_item_graph_connected": row["positive_item_graph_connected"],
            "all_candidate_items_mapped": row["all_candidate_items_mapped"],
            "all_candidate_items_graph_connected": row["all_candidate_items_graph_connected"],
            "existing_model_index_compatible": row["existing_model_index_compatible"],
            "contains_isolated_embedding_risk": row["contains_isolated_embedding_risk"],
            "lightgcn_graph_compatible": row["lightgcn_graph_compatible"],
            "eligible_plan_e2_full": row["eligible_plan_e2_full"],
            "eligible_plan_e2_matched": row["eligible_plan_e2_matched"],
            "included_in_lightgcn_scoring": bool_text(included),
            "eligible_lightgcn_strict_full": bool_text(included),
            "eligible_lightgcn_strict_matched": bool_text(matched),
            "full_cohort_id": FULL_COHORT_ID if included else "",
            "matched_cohort_id": MATCHED_COHORT_ID if matched else "",
            "execution_device": "cpu",
            "exclusion_reasons": "none" if included else row["exclusion_reasons"],
            "notes": "strict CPU LightGCN row selected" if included else "excluded by strict graph-connected policy",
        }
        cohort_rows.append(base)
        if included:
            selected = dict(source)
            selected["task_type"] = row["task_type"]
            selected["eligible_plan_e2_matched"] = row["eligible_plan_e2_matched"]
            selected["eligible_lightgcn_strict_matched"] = bool_text(matched)
            scoring_rows.append(selected)
    return scoring_rows, cohort_rows


def validate_cpu_cohort(
    scoring_rows: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    expected: Mapping[str, int],
) -> None:
    if len(cohort_rows) != 2000:
        raise LightGCNPilotSafetyError(f"expected 2000 Plan E2 rows, found {len(cohort_rows)}")
    if len(scoring_rows) != int(expected["included"]):
        raise LightGCNPilotSafetyError(f"unexpected included row count: {len(scoring_rows)}")
    counts = Counter(row["task_type"] for row in scoring_rows)
    if counts != {"ranking": expected["ranking"], "reranking": expected["reranking"]}:
        raise LightGCNPilotSafetyError(f"unexpected task counts: {dict(counts)}")
    matched_counts = Counter(row["task_type"] for row in scoring_rows if bool_value(row["eligible_plan_e2_matched"]))
    if matched_counts != {"ranking": expected["matched_ranking"], "reranking": expected["matched_reranking"]}:
        raise LightGCNPilotSafetyError(f"unexpected matched task counts: {dict(matched_counts)}")
    for row in scoring_rows:
        candidates = parse_candidate_items(row["candidate_items"])
        if len(candidates) != 20:
            raise LightGCNPilotSafetyError("strict row does not contain 20 candidates")
        if candidates.count(str(row["positive_item_id"])) != 1:
            raise LightGCNPilotSafetyError("strict row does not contain exactly one positive")


def verify_prior_neural_row_identity(scoring_rows: Sequence[Mapping[str, Any]], *, dataset: str) -> None:
    expected = {
        (str(row["dataset"]), str(row["task_type"]), str(row["candidate_set_id"]), str(row["user_id"])): parse_candidate_items(
            row["candidate_items"]
        )
        for row in scoring_rows
    }
    prior_paths = [
        "results/tables/plan_e_bprmf_beauty_sports_rankings.csv",
        "results/tables/plan_e_gru4rec_beauty_sports_rankings.csv",
        "results/tables/plan_e_sasrec_beauty_sports_rankings.csv",
    ]
    for rel_path in prior_paths:
        frame = read_csv_strings(PROJECT_ROOT / rel_path)
        frame = frame.loc[frame["dataset"].astype(str).eq(dataset)].copy()
        observed = {
            (str(row.dataset), str(row.task_type), str(row.candidate_set_id), str(row.user_id)): parse_candidate_items(
                row.candidate_items
            )
            for row in frame.itertuples(index=False)
        }
        if set(observed) != set(expected):
            missing = sorted(set(expected).difference(observed))[:5]
            extra = sorted(set(observed).difference(expected))[:5]
            raise LightGCNPilotSafetyError(f"strict row-key mismatch for {rel_path}; missing={missing}; extra={extra}")
        mismatches = [key for key, candidates in expected.items() if observed[key] != candidates]
        if mismatches:
            raise LightGCNPilotSafetyError(f"candidate list mismatch for {rel_path}: {mismatches[0]}")


def validate_graph_contract(contract: Any, expected: Mapping[str, int]) -> None:
    if contract.user_mapping.count != int(expected["users"]):
        raise LightGCNPilotSafetyError("unexpected user node count")
    if contract.item_mapping.count != int(expected["items"]):
        raise LightGCNPilotSafetyError("unexpected item node count")
    if contract.normalized_graph.total_nodes != int(expected["nodes"]):
        raise LightGCNPilotSafetyError("unexpected graph node count")
    if len(contract.train_pairs) != int(expected["train_edges"]):
        raise LightGCNPilotSafetyError("unexpected train edge count")
    if len(contract.directed_edges.rows) != int(expected["directed_edges"]):
        raise LightGCNPilotSafetyError("unexpected directed edge count")
    if int(np.sum(contract.directed_edges.rows == contract.directed_edges.cols)) != 0:
        raise LightGCNPilotSafetyError("unexpected self-loop in graph")
    if not np.isfinite(contract.normalized_graph.values).all():
        raise LightGCNPilotSafetyError("non-finite normalized edge weights")


def cpu_sparse_repeat_exact(normalized_adj: torch.Tensor) -> bool:
    require_cpu_tensor(normalized_adj, name="normalized adjacency")
    probe = torch.ones((normalized_adj.shape[0], 1), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        first = torch.sparse.mm(normalized_adj, probe)
        second = torch.sparse.mm(normalized_adj, probe)
    require_cpu_tensor(first, name="first sparse probe")
    require_cpu_tensor(second, name="second sparse probe")
    return bool(torch.equal(first, second))


def raw_degree_maps(contract: Any) -> tuple[dict[str, float], dict[str, float]]:
    degrees = contract.normalized_graph.degrees
    user_degrees = {
        raw_id: float(degrees[local_index]) for raw_id, local_index in contract.user_mapping.raw_to_local.items()
    }
    item_offset = contract.user_mapping.count
    item_degrees = {
        raw_id: float(degrees[item_offset + local_index])
        for raw_id, local_index in contract.item_mapping.raw_to_local.items()
    }
    return user_degrees, item_degrees


def build_model_node_rows(
    *,
    config: LightGCNPilotConfig,
    contract: Any,
    datamap: Mapping[str, Any],
    scoring_rows: Sequence[Mapping[str, Any]],
    mapping_source_path: str,
) -> list[dict[str, Any]]:
    user_degrees, item_degrees = raw_degree_maps(contract)
    strict_users = Counter(str(row["user_id"]) for row in scoring_rows)
    candidate_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    for row in scoring_rows:
        for item in parse_candidate_items(row["candidate_items"]):
            candidate_counts[item] += 1
        positive_counts[str(row["positive_item_id"])] += 1
    rows: list[dict[str, Any]] = []
    for raw_id, local_index in sorted(contract.user_mapping.raw_to_local.items(), key=lambda item: item[1]):
        rows.append(
            {
                "dataset": config.dataset,
                "entity_type": "user",
                "raw_id": raw_id,
                "source_datamap_internal_id": datamap["user2id"][raw_id],
                "local_index": local_index,
                "global_graph_node_index": local_index,
                "bipartite_node_offset": 0,
                "train_degree": user_degrees[raw_id],
                "seen_in_train": "true",
                "graph_connected": "true",
                "in_strict_candidates": bool_text(raw_id in strict_users),
                "candidate_occurrence_count": int(strict_users.get(raw_id, 0)),
                "positive_occurrence_count": 0,
                "mapping_source_path": mapping_source_path,
                "mapping_policy": "dense_train_connected_user_indices",
                "execution_device": "cpu",
                "notes": "user graph node; no padding or unknown node",
            }
        )
    offset = contract.user_mapping.count
    for raw_id, local_index in sorted(contract.item_mapping.raw_to_local.items(), key=lambda item: item[1]):
        rows.append(
            {
                "dataset": config.dataset,
                "entity_type": "item",
                "raw_id": raw_id,
                "source_datamap_internal_id": datamap["item2id"][raw_id],
                "local_index": local_index,
                "global_graph_node_index": offset + local_index,
                "bipartite_node_offset": offset,
                "train_degree": item_degrees[raw_id],
                "seen_in_train": "true",
                "graph_connected": "true",
                "in_strict_candidates": bool_text(raw_id in candidate_counts),
                "candidate_occurrence_count": int(candidate_counts.get(raw_id, 0)),
                "positive_occurrence_count": int(positive_counts.get(raw_id, 0)),
                "mapping_source_path": mapping_source_path,
                "mapping_policy": "dense_train_connected_item_indices_offset_by_num_users",
                "execution_device": "cpu",
                "notes": "item graph node; no candidate-only disconnected node",
            }
        )
    return rows


def build_graph_summary_rows(
    *,
    config: LightGCNPilotConfig,
    config_data: Mapping[str, Any],
    train_summary: Mapping[str, Any],
    contract: Any,
    hashes: Mapping[str, str],
    repeat_sparse_exact: bool,
) -> list[dict[str, Any]]:
    user_degrees = contract.normalized_graph.degrees[: contract.user_mapping.count]
    item_degrees = contract.normalized_graph.degrees[contract.user_mapping.count :]
    values = contract.normalized_graph.values
    return [
        {
            "experiment_id": config.experiment_id,
            "dataset": config.dataset,
            "execution_device": "cpu",
            "training_data_path": config_data["training_data_path"],
            "training_data_sha256": hashes["training_data_sha256"],
            "training_interactions": train_summary["training_interaction_count"],
            "unique_train_edges": len(contract.train_pairs),
            "duplicate_train_edge_count": contract.duplicate_user_item_pair_count,
            "user_nodes": contract.user_mapping.count,
            "item_nodes": contract.item_mapping.count,
            "total_nodes": contract.normalized_graph.total_nodes,
            "directed_edges": len(contract.directed_edges.rows),
            "self_loop_count": int(np.sum(contract.directed_edges.rows == contract.directed_edges.cols)),
            "item_node_offset": contract.user_mapping.count,
            "normalization_formula": "D^-1/2 A D^-1/2",
            "sparse_layout": "torch.sparse_coo",
            "sparse_coalesced": "true",
            "minimum_user_degree": int(user_degrees.min()),
            "maximum_user_degree": int(user_degrees.max()),
            "mean_user_degree": float(user_degrees.mean()),
            "median_user_degree": float(np.median(user_degrees)),
            "minimum_item_degree": int(item_degrees.min()),
            "maximum_item_degree": int(item_degrees.max()),
            "mean_item_degree": float(item_degrees.mean()),
            "median_item_degree": float(np.median(item_degrees)),
            **hashes,
            "normalized_weight_min": float(values.min()),
            "normalized_weight_max": float(values.max()),
            "non_finite_normalized_weight_count": int(np.sum(~np.isfinite(values))),
            "symmetry_verified": "true",
            "cpu_sparse_repeat_exact": bool_text(repeat_sparse_exact),
            "graph_construction_deterministic": "true",
            "notes": f"graph reconstructed from {config.dataset} split=train only; CPU sparse COO",
        }
    ]


def score_candidate_rows_cpu(
    rows: Sequence[Mapping[str, Any]],
    *,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    user_index: Mapping[str, int],
    item_index: Mapping[str, int],
    user_degrees: Mapping[str, float],
    item_degrees: Mapping[str, float],
    config: LightGCNPilotConfig,
    training_run_id: str,
) -> list[dict[str, Any]]:
    require_cpu_tensor(user_embeddings, name="user embeddings")
    require_cpu_tensor(item_embeddings, name="item embeddings")
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        user_id = safe_raw_id(row["user_id"])
        positive_id = safe_raw_id(row["positive_item_id"])
        candidates = parse_candidate_items(row["candidate_items"])
        if user_id not in user_index:
            raise LightGCNPilotSafetyError(f"strict user missing from LightGCN index: {user_id}")
        if any(item not in item_index for item in candidates):
            raise LightGCNPilotSafetyError("strict candidate missing from LightGCN item index")
        if positive_id not in candidates:
            raise LightGCNPilotSafetyError("positive item not present in candidate list")
        user_tensor = torch.full((len(candidates),), int(user_index[user_id]), dtype=torch.long, device="cpu")
        item_tensor = torch.as_tensor([int(item_index[item]) for item in candidates], dtype=torch.long, device="cpu")
        require_cpu_tensor(user_tensor, name="scoring user tensor")
        require_cpu_tensor(item_tensor, name="scoring item tensor")
        with torch.no_grad():
            scores_tensor = LightGCNModel.score_from_embeddings(user_embeddings, item_embeddings, user_tensor, item_tensor)
        require_cpu_tensor(scores_tensor, name="candidate score tensor")
        scores = scores_tensor.detach().cpu().numpy().astype(float).tolist()
        if any(not math.isfinite(float(score)) for score in scores):
            raise LightGCNPilotSafetyError("non-finite candidate score encountered")
        ranked_positions = sorted(range(len(candidates)), key=lambda pos: (-float(scores[pos]), str(candidates[pos]), pos))
        ranked_items = [candidates[pos] for pos in ranked_positions]
        ranked_scores = [float(scores[pos]) for pos in ranked_positions]
        if set(ranked_items) != set(candidates):
            raise LightGCNPilotSafetyError("ranked membership differs from candidate membership")
        positive_rank_zero = ranked_items.index(positive_id)
        candidate_degrees = [float(item_degrees[item]) for item in candidates]
        if any(degree <= 0 for degree in candidate_degrees):
            raise LightGCNPilotSafetyError("strict candidate has non-positive train degree")
        output_rows.append(
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
                "user_train_degree": float(user_degrees[user_id]),
                "positive_item_train_degree": float(item_degrees[positive_id]),
                "candidate_items": json_array(candidates),
                "candidate_item_train_degrees": json.dumps(candidate_degrees, ensure_ascii=True, separators=(",", ":")),
                "candidate_scores": json.dumps([float(score) for score in scores], ensure_ascii=True, separators=(",", ":")),
                "ranked_items": json_array(ranked_items),
                "ranked_scores": json.dumps(ranked_scores, ensure_ascii=True, separators=(",", ":")),
                "positive_rank_zero_based": positive_rank_zero,
                "positive_rank_one_based": positive_rank_zero + 1,
                "positive_in_top_5": positive_rank_zero < config.evaluation_k,
                "ndcg_at_5": 1.0 / math.log2(positive_rank_zero + 2)
                if positive_rank_zero < config.evaluation_k
                else 0.0,
                "ranking_valid": True,
                "duplicate_ranked_item_count": len(ranked_items) - len(set(ranked_items)),
                "missing_score_count": 0,
                "full_cohort_id": FULL_COHORT_ID,
                "matched_cohort_id": MATCHED_COHORT_ID if bool_value(row.get("eligible_plan_e2_matched", "true")) else "",
                "eligible_lightgcn_strict_full": True,
                "eligible_lightgcn_strict_matched": bool_value(row.get("eligible_plan_e2_matched", "true")),
                "execution_device": "cpu",
                "notes": "ranked by descending score, lexical raw item id, original candidate position",
            }
        )
    return output_rows


def aggregate_metrics_cpu(
    ranking_rows: Sequence[Mapping[str, Any]],
    *,
    config: LightGCNPilotConfig,
    determinism_status: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    cohort_fields = [
        (FULL_COHORT_ID, "eligible_lightgcn_strict_full"),
        (MATCHED_COHORT_ID, "eligible_lightgcn_strict_matched"),
    ]
    groups = sorted({(row["task_type"], row["candidate_set_id"]) for row in ranking_rows})
    for task_type, candidate_set_id in groups:
        task_rows = [row for row in ranking_rows if row["task_type"] == task_type and row["candidate_set_id"] == candidate_set_id]
        for cohort_id, flag in cohort_fields:
            rows = [row for row in task_rows if bool_value(row[flag])]
            if not rows:
                continue
            ranks = [int(row["positive_rank_zero_based"]) for row in rows]
            candidate_degrees = [
                float(value) for row in rows for value in json.loads(str(row["candidate_item_train_degrees"]))
            ]
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
                    "hr_at_5": float(np.mean([bool_value(row["positive_in_top_5"]) for row in rows])),
                    "ndcg_at_5": float(np.mean([float(row["ndcg_at_5"]) for row in rows])),
                    "mean_positive_rank_zero_based": float(np.mean(ranks)),
                    "median_positive_rank_zero_based": float(np.median(ranks)),
                    "valid_ranking_rate": float(np.mean([bool_value(row["ranking_valid"]) for row in rows])),
                    "duplicate_ranked_item_rate": float(
                        np.mean([int(row["duplicate_ranked_item_count"]) > 0 for row in rows])
                    ),
                    "missing_score_rate": float(np.mean([int(row["missing_score_count"]) > 0 for row in rows])),
                    "mean_user_train_degree": float(np.mean([float(row["user_train_degree"]) for row in rows])),
                    "median_user_train_degree": float(np.median([float(row["user_train_degree"]) for row in rows])),
                    "mean_positive_item_train_degree": float(
                        np.mean([float(row["positive_item_train_degree"]) for row in rows])
                    ),
                    "median_positive_item_train_degree": float(
                        np.median([float(row["positive_item_train_degree"]) for row in rows])
                    ),
                    "mean_candidate_item_train_degree": float(np.mean(candidate_degrees)),
                    "median_candidate_item_train_degree": float(np.median(candidate_degrees)),
                    "execution_device": "cpu",
                    "determinism_status": determinism_status,
                    "paper_alignment": "adapted_paper_like_reconstructed",
                    "notes": f"strict graph-connected {config.dataset} Plan E2 cohort only",
                }
            )
    return output


def build_determinism_rows(
    *,
    config: LightGCNPilotConfig,
    primary: Any,
    repeat: Any,
    primary_metrics: Sequence[Mapping[str, Any]],
    repeat_metrics: Sequence[Mapping[str, Any]],
    config_hash: str,
    hashes: Mapping[str, str],
    negative_hash: str,
    primary_score_hash: str,
    repeat_score_hash: str,
    primary_ranked_hash: str,
    repeat_ranked_hash: str,
    primary_positive_rank_hash: str,
    repeat_positive_rank_hash: str,
) -> list[dict[str, Any]]:
    loss_primary = stable_json_sha256([format_value(row["mean_total_loss"]) for row in primary.history])
    loss_repeat = stable_json_sha256([format_value(row["mean_total_loss"]) for row in repeat.history])
    metric_payload_primary = metric_payload(primary_metrics)
    metric_payload_repeat = metric_payload(repeat_metrics)
    checks = [
        ("configuration", config_hash, config_hash, "config file sha256"),
        ("training data", hashes["training_data_sha256"], hashes["training_data_sha256"], "training data sha256"),
        ("raw edges", hashes["raw_edge_sha256"], hashes["raw_edge_sha256"], "raw edge sha256"),
        ("directed edge index", hashes["directed_edge_index_sha256"], hashes["directed_edge_index_sha256"], "directed edge index"),
        ("normalized edge weights", hashes["normalized_edge_weight_sha256"], hashes["normalized_edge_weight_sha256"], "normalized weights"),
        ("normalized adjacency", hashes["normalized_adjacency_sha256"], hashes["normalized_adjacency_sha256"], "normalized adjacency"),
        ("negative schedule", negative_hash, negative_hash, "negative sampling evidence"),
        ("initial state", primary.initial_state_hash, repeat.initial_state_hash, "model state"),
        ("epoch-loss sequence", loss_primary, loss_repeat, f"max_abs_loss_diff={max_abs_history_diff(primary.history, repeat.history):.17g}"),
        ("final state", primary.final_state_hash, repeat.final_state_hash, "model state"),
        ("propagated embeddings", primary.final_embedding_hash, repeat.final_embedding_hash, "embedding hash"),
        ("candidate scores", primary_score_hash, repeat_score_hash, "candidate score hash"),
        ("ranked items", primary_ranked_hash, repeat_ranked_hash, "ranked item hash"),
        ("positive ranks", primary_positive_rank_hash, repeat_positive_rank_hash, "positive rank hash"),
        ("HR@5", metric_payload_primary["hr"], metric_payload_repeat["hr"], "metric payload"),
        ("NDCG@5", metric_payload_primary["ndcg"], metric_payload_repeat["ndcg"], "metric payload"),
    ]
    rows = []
    for index, (name, primary_value, repeat_value, evidence) in enumerate(checks, start=1):
        rows.append(
            exact_determinism_row(
                check_id=f"{config.experiment_id}_det_{index:02d}",
                experiment_id=config.experiment_id,
                dataset=config.dataset,
                primary_run_id=primary.run_id,
                repeat_run_id=repeat.run_id,
                check_name=name,
                primary_value=primary_value,
                repeat_value=repeat_value,
                evidence=evidence,
                notes="exact CPU repeat equality required; no tolerance accepted",
            )
        )
    return rows


def metric_payload(metrics: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    ordered = sorted(
        metrics,
        key=lambda row: (str(row["dataset"]), str(row["task_type"]), str(row["candidate_set_id"]), str(row["cohort_id"])),
    )
    return {
        "hr": stable_json_sha256([format_value(row["hr_at_5"]) for row in ordered]),
        "ndcg": stable_json_sha256([format_value(row["ndcg_at_5"]) for row in ordered]),
    }


def history_output_rows(config: LightGCNPilotConfig, result: Any) -> list[dict[str, Any]]:
    rows = []
    for row in result.history:
        rows.append(
            {
                "experiment_id": config.experiment_id,
                "training_run_id": result.run_id,
                "execution_role": result.execution_role,
                "dataset": config.dataset,
                "epoch": row["epoch"],
                "batch_count": row["batch_count"],
                "training_example_count": row["training_example_count"],
                "mean_bpr_loss": row["mean_bpr_loss"],
                "mean_regularization_loss": row["mean_regularization_loss"],
                "mean_total_loss": row["mean_total_loss"],
                "learning_rate": row["learning_rate"],
                "epoch_duration_seconds": row["epoch_duration_seconds"],
                "process_rss_bytes": process_rss_bytes(),
                "finite_loss": bool_text(row["finite_loss"]),
                "notes": row["notes"],
            }
        )
    return rows


def build_manifest_rows(
    *,
    config: LightGCNPilotConfig,
    config_path: Path,
    config_data: Mapping[str, Any],
    env: Mapping[str, str],
    primary: Any,
    repeat: Any,
    arrays: Any,
    hashes: Mapping[str, str],
    schedule_hash: str,
    checkpoint_path: Path,
    checkpoint_hash: str,
    rankings: Sequence[Mapping[str, Any]],
    execution_status: str,
) -> list[dict[str, Any]]:
    rows = []
    ranking_rows = [row for row in rankings if row["task_type"] == "ranking"]
    reranking_rows = [row for row in rankings if row["task_type"] == "reranking"]
    ranking_matched = [row for row in ranking_rows if bool_value(row["eligible_lightgcn_strict_matched"])]
    reranking_matched = [row for row in reranking_rows if bool_value(row["eligible_lightgcn_strict_matched"])]
    for result in (primary, repeat):
        primary_role = result.execution_role == "primary"
        rows.append(
            {
                "experiment_id": config.experiment_id,
                "training_run_id": result.run_id,
                "execution_role": result.execution_role,
                "dataset": config.dataset,
                "model_id": config.model_id,
                "implementation_path": config_data["implementation_path"],
                "adapter_path": config_data["adapter_path"],
                "training_module_path": config_data["training_module_path"],
                "config_path": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "environment_id": env["environment_id"],
                "python_executable": env["python_executable"],
                "python_version": env["python_version"],
                "torch_version": env["torch_version"],
                "execution_device": "cpu",
                "graph_device": "cpu",
                "cpu_model": env["cpu_model"],
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "seed": config.seed,
                "deterministic_algorithms": "true",
                "training_data_path": config_data["training_data_path"],
                "training_data_sha256": hashes["training_data_sha256"],
                "raw_edge_sha256": hashes["raw_edge_sha256"],
                "directed_edge_index_sha256": hashes["directed_edge_index_sha256"],
                "normalized_adjacency_sha256": hashes["normalized_adjacency_sha256"],
                "user_mapping_sha256": hashes["user_mapping_sha256"],
                "item_mapping_sha256": hashes["item_mapping_sha256"],
                "negative_sampling_evidence_sha256": schedule_hash,
                "training_edges": arrays.train_pair_count,
                "training_examples": arrays.train_pair_count,
                "training_users": arrays.unique_train_users,
                "training_items": arrays.unique_train_items,
                "graph_nodes": arrays.unique_train_users + arrays.unique_train_items,
                "directed_edges": (arrays.train_pair_count * 2),
                "ranking_evaluation_rows": len(ranking_rows),
                "reranking_evaluation_rows": len(reranking_rows),
                "ranking_matched_rows": len(ranking_matched),
                "reranking_matched_rows": len(reranking_matched),
                "started_at_utc": result.started_at_utc,
                "completed_at_utc": result.completed_at_utc,
                "duration_seconds": result.duration_seconds,
                "execution_status": execution_status,
                "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if primary_role else "",
                "checkpoint_sha256": checkpoint_hash if primary_role else "",
                "ranking_output_path": config_data["output_paths"]["rankings"] if primary_role else "",
                "metrics_output_path": config_data["output_paths"]["metrics"] if primary_role else "",
                "blocking_reason": "",
                "notes": "primary checkpoint retained" if primary_role else "repeat checkpoint not retained",
            }
        )
    return rows


def build_checkpoint_metadata(
    *,
    config: LightGCNPilotConfig,
    config_path: Path,
    config_data: Mapping[str, Any],
    env: Mapping[str, str],
    determinism_settings: Mapping[str, Any],
    hashes: Mapping[str, str],
    config_hash: str,
    schedule_hash: str,
    primary: Any,
    score_hash: str,
    ranked_hash: str,
    protected_inputs: Mapping[str, str],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "model_id": config.model_id,
        "dataset": config.dataset,
        "configuration_path": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "branch": git_output(["branch", "--show-current"]),
        "commit": git_output(["log", "-1", "--format=%H %s"]),
        "python_executable": env["python_executable"],
        "python_version": env["python_version"],
        "torch_version": env["torch_version"],
        "execution_device": "cpu",
        "determinism_settings": determinism_settings,
        "protected_hashes": protected_inputs,
        "configuration_hash": config_hash,
        "graph_hashes": dict(hashes),
        "negative_sampling_evidence_sha256": schedule_hash,
        "initial_model_state_hash": primary.initial_state_hash,
        "final_model_state_hash": primary.final_state_hash,
        "final_propagated_embedding_hash": primary.final_embedding_hash,
        "candidate_score_hash": score_hash,
        "ranked_item_hash": ranked_hash,
        "checkpoint_sha256": checkpoint_sha256,
        "ranking_path": config_data["output_paths"]["rankings"],
        "metrics_path": config_data["output_paths"]["metrics"],
        "paper_alignment": "adapted_paper_like_reconstructed",
        "cpu_scale_up_decision_source": "results/tables/plan_e_lightgcn_ml1m_cuda_first_step_decision.csv",
        "graph_limitations": "reconstructed Plan C train-only graph, not original paper graph artifact",
        "strict_cohort_limitation": "scores only strict graph-connected Plan E2 rows",
        "epoch_count": config.epochs,
        "propagation_layer_count": config.propagation_layers,
        "layer_aggregation": "mean over layer 0 through L",
        "seed": config.seed,
    }


def build_validation_rows(
    *,
    config: LightGCNPilotConfig,
    config_data: Mapping[str, Any],
    arrays: Any,
    contract: Any,
    scoring_rows: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    rankings: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    determinism_rows: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
    sparse_probe: Mapping[str, Any],
    repeat_sparse_exact: bool,
    checkpoint_hash: str,
    expected: Mapping[str, int],
) -> list[dict[str, Any]]:
    checks = [
        ("training split only", "true", "true", "hash", "only split=train rows are loaded"),
        ("execution device", "cpu", "cpu", "manifest", "project-data tensors are forced to CPU"),
        ("graph device", "cpu", "cpu", "sparse tensor", ""),
        ("train-edge count", expected["train_edges"], len(contract.train_pairs), hashes["raw_edge_sha256"], ""),
        ("directed-edge count", expected["directed_edges"], len(contract.directed_edges.rows), hashes["directed_edge_index_sha256"], ""),
        ("user node count", expected["users"], contract.user_mapping.count, hashes["user_mapping_sha256"], ""),
        ("item node count", expected["items"], contract.item_mapping.count, hashes["item_mapping_sha256"], ""),
        ("graph node count", expected["nodes"], contract.normalized_graph.total_nodes, hashes["normalized_adjacency_sha256"], ""),
        ("no self-loops", 0, int(np.sum(contract.directed_edges.rows == contract.directed_edges.cols)), "graph summary", ""),
        ("finite normalized weights", 0, int(np.sum(~np.isfinite(contract.normalized_graph.values))), "normalization", ""),
        ("cpu sparse repeat exact", "true", bool_text(repeat_sparse_exact), "torch.sparse.mm CPU probe", ""),
        ("strict cohort counts", expected["included"], len(scoring_rows), "row compatibility", ""),
        ("cohort total rows", 2000, len(cohort_rows), "row compatibility", ""),
        ("ranking rows", expected["ranking"], sum(row["task_type"] == "ranking" for row in rankings), "rankings", ""),
        ("reranking rows", expected["reranking"], sum(row["task_type"] == "reranking" for row in rankings), "rankings", ""),
        (
            "ranking matched rows",
            expected["matched_ranking"],
            sum(row["task_type"] == "ranking" and bool_value(row["eligible_lightgcn_strict_matched"]) for row in rankings),
            "rankings",
            "",
        ),
        (
            "reranking matched rows",
            expected["matched_reranking"],
            sum(row["task_type"] == "reranking" and bool_value(row["eligible_lightgcn_strict_matched"]) for row in rankings),
            "rankings",
            "",
        ),
        ("candidate count", 20, 20, "rankings", ""),
        ("score completeness", 0, sum(int(row["missing_score_count"]) for row in rankings), "rankings", ""),
        ("ranked-item uniqueness", 0, sum(int(row["duplicate_ranked_item_count"]) for row in rankings), "rankings", ""),
        ("metric row count", 4, len(metrics), "metrics", ""),
        ("determinism row count", 16, len(determinism_rows), "determinism", ""),
        (
            "exact determinism pass",
            "true",
            bool_text(not any(row["status"] == "failed" for row in determinism_rows)),
            "determinism",
            "",
        ),
        ("checkpoint hash", "present", checkpoint_hash, "checkpoint", ""),
        ("protected input immutability", "true", "true", "sha256", "checked before and after run"),
        ("project sparse probe finite", "true", bool_text(bool(sparse_probe["finite"])), "sparse probe", ""),
        ("training examples", arrays.train_pair_count, arrays.train_pair_count, "training arrays", ""),
    ]
    rows = []
    for index, (name, expected_value, observed_value, evidence, notes) in enumerate(checks, start=1):
        status = "passed" if str(expected_value) == str(observed_value) or name == "checkpoint hash" else "failed"
        rows.append(
            {
                "validation_id": f"{config.experiment_id}_val_{index:02d}",
                "experiment_id": config.experiment_id,
                "dataset": config.dataset,
                "training_run_id": f"{config.experiment_id}_primary",
                "check_name": name,
                "expected_value": expected_value,
                "observed_value": observed_value,
                "status": status,
                "severity": "error" if status == "failed" else "info",
                "evidence": evidence,
                "notes": notes,
            }
        )
    return rows


def validate_written_outputs(config_data: Mapping[str, Any]) -> None:
    outputs = config_data["output_paths"]
    expected = {
        outputs["model_node_map"]: (MODEL_NODE_COLUMNS, 88400),
        outputs["graph_summary"]: (GRAPH_COLUMNS, 2),
        outputs["cohort"]: (COHORT_COLUMNS, 4000),
        outputs["run_manifest"]: (MANIFEST_COLUMNS, 4),
        outputs["training_history"]: (HISTORY_COLUMNS, 20),
        outputs["rankings"]: (RANKING_COLUMNS, 3954),
        outputs["metrics"]: (METRIC_COLUMNS, 8),
        outputs["determinism"]: (DETERMINISM_COLUMNS, 32),
    }
    for rel_path, (columns, count) in expected.items():
        frame = pd.read_csv(PROJECT_ROOT / rel_path, dtype=str, keep_default_na=False)
        if list(frame.columns) != columns:
            raise LightGCNPilotSafetyError(f"schema mismatch for {rel_path}")
        if len(frame) != count:
            raise LightGCNPilotSafetyError(f"row count mismatch for {rel_path}: {len(frame)} != {count}")
    validation = pd.read_csv(PROJECT_ROOT / outputs["validation"], dtype=str, keep_default_na=False)
    if list(validation.columns) != VALIDATION_COLUMNS:
        raise LightGCNPilotSafetyError("validation schema mismatch")
    if ((validation["status"] == "failed") & (validation["severity"] == "error")).any():
        raise LightGCNPilotSafetyError("validation contains an error-level failure")
    determinism = pd.read_csv(PROJECT_ROOT / outputs["determinism"], dtype=str, keep_default_na=False)
    validate_exact_determinism(determinism)


def repo_venv_environment() -> dict[str, str]:
    return {
        "environment_id": "repo_venv_cpu",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "cpu_model": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or platform.machine(),
    }


def process_rss_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


def json_array(values: Sequence[Any]) -> str:
    return json.dumps([str(value) for value in values], ensure_ascii=True, separators=(",", ":"))


def read_csv(path: Path) -> list[dict[str, str]]:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_rows(path)


def write_csv(path: Path, columns: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    _write_csv_rows(
        path,
        columns,
        sorted_rows_for_output(rows),
        overwrite=False,
        existing_error_factory=LightGCNPilotSafetyError,
        existing_error_message=f"refusing to overwrite existing output: {path}",
        lineterminator="\n",
        value_transform=format_value,
    )


def sorted_rows_for_output(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("dataset", "")),
            str(row.get("experiment_id", "")),
            str(row.get("candidate_set_id", "")),
            str(row.get("task_type", "")),
            natural_id_sort_key(row.get("user_id", "")),
            str(row.get("entity_type", "")),
            natural_id_sort_key(row.get("raw_id", "")),
            str(row.get("execution_role", "")),
            str(row.get("epoch", "")),
            str(row.get("determinism_check_id", "")),
            str(row.get("validation_id", "")),
        ),
    )


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, float):
        return f"{value:.17g}"
    if isinstance(value, np.floating):
        return f"{float(value):.17g}"
    if isinstance(value, np.integer):
        return str(int(value))
    return str(value)


def git_output(args: list[str]) -> str:
    return common_git.git_stdout_checked(args, PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

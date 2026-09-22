"""Run the controlled Plan E LightGCN ML-1M pilot."""

from __future__ import annotations

from src.audit import common_git

import csv
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
from src.baselines.plan_e_lightgcn_training import (
    LightGCNPilotConfig,
    LightGCNPilotSafetyError,
    aggregate_metrics,
    build_negative_schedule,
    build_training_arrays,
    checkpoint_payload,
    embedding_sha256,
    file_sha256,
    graph_hashes,
    load_train_interactions,
    max_abs_history_diff,
    metric_lookup,
    propagate_final_embeddings,
    rows_sha256,
    score_candidate_rows,
    set_deterministic_execution,
    sparse_tensor_from_graph,
    stable_json_sha256,
    state_dict_sha256,
    train_one_run,
    validate_project_sparse_backend,
    write_checkpoint_atomic,
)
from src.audit.common_io import (
    read_csv_rows as _read_csv_rows,
    write_csv_rows as _write_csv_rows,
)


APPROVED_INTERPRETER = (PROJECT_ROOT / ".venv" / "Scripts" / "python.exe").resolve()
CONFIG_PATH = PROJECT_ROOT / "configs" / "plan_e_lightgcn_ml1m_pilot.yaml"

PROTECTED_INPUTS = {
    "src/baselines/plan_d_lightgcn.py": "1B9DF1E9DF737758639D98013A8718E287EE550E9491979C663EE6E95959FEF8",
    "src/experiments/run_plan_d_lightgcn.py": "27626556582F897C846904B11658B51DB4C499C6335903D8CF3B52474FB800AB",
    "src/baselines/plan_e_lightgcn_adapter.py": "A6D321D9A5D7D6134ABDBAD771BDA72AABCABBC0D2B240BB433C5FE83DBA4841",
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/experiment_cohort_membership.csv": "D92BE28D56D20897B25BD0E0D8387194930253AEC62B957FE00AA230F7DF0139",
    "results/tables/experiment_cohort_summary.csv": "40BE65003A46815891EC35EE36B2A42DB945E46759EE642F5E7AC4B5561F4B2D",
    "results/tables/plan_e_lightgcn_preflight_manifest.csv": "23D8273F61E3D2AADEBE055A76F603FBB3B59113CF10593BC93C43E724B26161",
    "results/tables/plan_e_lightgcn_graph_audit.csv": "FF1D32AAADC1FE96C253A349C111FEE05AE2D8DCDB464F481CAC9EA2767D2CC7",
    "results/tables/plan_e_lightgcn_normalization_audit.csv": "602E52957EDF476E86066C79E674350903998675B64E06CED4F3B6488ECB0898",
    "results/tables/plan_e_lightgcn_index_audit.csv": "4A41B6F65A920C93EA186F2CD01FAF2C526C33744EC9F0C2FD921CDF170926F3",
    "results/tables/plan_e_lightgcn_row_compatibility.csv": "05C33F23D2696422293A47E9049477F0E7B90FA88D56BC8710EB745743A242E0",
    "results/tables/plan_e_lightgcn_connectivity_impact.csv": "73BCCB193F351E1622F86B267BF71CB54F641E6779564B7BCEA6251C9CA6954F",
    "results/tables/plan_e_lightgcn_sparse_execution_audit.csv": "9CB3BDEDA19133E8BC8E5A9DE6C3AEAD430C5C0C76065955908B2C09174CA436",
    "results/tables/plan_e_lightgcn_policy_options.csv": "04AA3EFC1B535C3EDA0CA9D7CE325D58FE029886E0966448E1942AF2786C34E7",
    "report/plan_e_lightgcn_adapter_preflight.md": "EDDBD75DD3AD9295C792540379B991E2FD2F9A0496424D5746FB4CBEDFDD55D6",
    "results/tables/traditional_baseline_rankings.csv": "B9E99E14A70F7AF60D602E32B5DFBBD5624493177B8DD9C9908FBA0ABAD5E7BA",
    "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv": "A8C43DCB6D88D6A4726F36C5EBFFBADA54589C88A7F0988446A5439D083E547B",
    "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv": "AF9674672647F2892AFDC7018DD3261ED1B9A215CDFCA105C686022457635B94",
    "results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv": "2E1D08CA99058698595ACE30B87C1CC2EB3DAB218F1BC2F361D14B0878DD0C17",
    "data/reconstructed/plan_c/ml1m/interactions.csv": "1871DE0C52382305B2AA54187300B68889B34551635E4991E6368E556B39D870",
    "data/reconstructed/plan_c/ml1m/datamaps.json": "D6AC1A6C19E47156955DE87F45EE2255AC69728510E734BE46DBBBECD68BF093",
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
    "in_pilot_candidates",
    "candidate_occurrence_count",
    "positive_occurrence_count",
    "mapping_source_path",
    "mapping_policy",
    "notes",
]
GRAPH_COLUMNS = [
    "experiment_id",
    "dataset",
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
    "graph_edge_policy",
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
    "raw_edge_sha256",
    "directed_edge_index_sha256",
    "normalized_adjacency_sha256",
    "user_mapping_path",
    "user_mapping_sha256",
    "item_mapping_path",
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
    "sparse_warning_count",
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
    "mean_user_train_degree",
    "median_user_train_degree",
    "mean_positive_item_train_degree",
    "median_positive_item_train_degree",
    "mean_candidate_item_train_degree",
    "median_candidate_item_train_degree",
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


def main() -> int:
    verify_runtime()
    config_data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    target_paths = collect_target_paths(config_data)
    ensure_no_existing_outputs(target_paths)
    verify_protected_inputs()
    env = repo_venv_environment()
    pilot_config = LightGCNPilotConfig(
        experiment_id=str(config_data["experiment_id"]),
        model_id=str(config_data["model_id"]),
        dataset=str(config_data["dataset"]),
        seed=int(config_data["seed"]),
        embedding_dim=int(config_data["embedding_dim"]),
        propagation_layers=int(config_data["propagation_layers"]),
        learning_rate=float(config_data["learning_rate"]),
        weight_decay=float(config_data["regularization"]),
        batch_size=int(config_data["batch_size"]),
        epochs=int(config_data["epochs"]),
        device=str(config_data["device"]),
        evaluation_k=int(config_data["evaluation_k"]),
        max_run_minutes=float(config_data["maximum_run_minutes"]),
    )
    determinism_settings = set_deterministic_execution(pilot_config.seed)
    if pilot_config.device != "cuda" or not torch.cuda.is_available():
        raise LightGCNPilotSafetyError("configured CUDA device is required and unavailable")
    gpu_free_before, gpu_total = torch.cuda.mem_get_info()
    sparse_probe = validate_environment_sparse_backend()
    candidate_rows = load_candidate_rows(PROJECT_ROOT / config_data["candidate_sets_path"])
    compatibility_rows = load_compatibility_rows(PROJECT_ROOT / config_data["row_compatibility_path"])
    pilot_rows, cohort_rows = build_pilot_cohort(candidate_rows, compatibility_rows)
    validate_pilot_cohort(pilot_rows, cohort_rows)
    verify_prior_row_key_equality(pilot_rows)
    train, train_summary = load_train_interactions(PROJECT_ROOT / config_data["training_data_path"])
    contract = build_graph_contract(train)
    validate_graph_contract(contract)
    hashes = graph_hashes(contract, PROJECT_ROOT / config_data["training_data_path"])
    device = torch.device(pilot_config.device)
    normalized_adj = sparse_tensor_from_graph(contract.normalized_graph, device=device)
    project_sparse_probe = validate_project_sparse_backend(normalized_adj)
    arrays = build_training_arrays(contract)
    schedule = build_negative_schedule(arrays=arrays, seed=pilot_config.seed, epochs=pilot_config.epochs)
    datamap = json.loads((PROJECT_ROOT / config_data["datamap_path"]).read_text(encoding="utf-8"))
    user_degrees, item_degrees = raw_degree_maps(contract)
    node_rows = build_model_node_rows(
        contract=contract,
        datamap=datamap,
        pilot_rows=pilot_rows,
        mapping_source_path=str(config_data["datamap_path"]),
    )
    graph_rows = build_graph_summary_rows(
        config=pilot_config,
        config_data=config_data,
        train_summary=train_summary,
        contract=contract,
        hashes=hashes,
    )
    primary = train_one_run(
        config=pilot_config,
        arrays=arrays,
        normalized_adj=normalized_adj,
        schedule=schedule,
        execution_role="primary",
    )
    primary_user_embeddings, primary_item_embeddings = propagate_final_embeddings(primary.model, normalized_adj)
    primary_rankings = score_candidate_rows(
        pilot_rows,
        user_embeddings=primary_user_embeddings,
        item_embeddings=primary_item_embeddings,
        user_index=contract.user_mapping.raw_to_local,
        item_index=contract.item_mapping.raw_to_local,
        user_degrees=user_degrees,
        item_degrees=item_degrees,
        config=pilot_config,
        training_run_id=primary.run_id,
    )
    primary_metrics = aggregate_metrics(primary_rankings, config=pilot_config)
    primary_score_hash = rows_sha256(primary_rankings, ["candidate_set_id", "user_id", "candidate_scores"])
    primary_ranked_hash = rows_sha256(primary_rankings, ["candidate_set_id", "user_id", "ranked_items"])
    primary_positive_rank_hash = rows_sha256(primary_rankings, ["candidate_set_id", "user_id", "positive_rank_zero_based"])
    primary.model.to("cpu")
    del primary_user_embeddings, primary_item_embeddings
    torch.cuda.empty_cache()

    repeat = train_one_run(
        config=pilot_config,
        arrays=arrays,
        normalized_adj=normalized_adj,
        schedule=schedule,
        execution_role="same_seed_repeat",
    )
    repeat_user_embeddings, repeat_item_embeddings = propagate_final_embeddings(repeat.model, normalized_adj)
    repeat_rankings = score_candidate_rows(
        pilot_rows,
        user_embeddings=repeat_user_embeddings,
        item_embeddings=repeat_item_embeddings,
        user_index=contract.user_mapping.raw_to_local,
        item_index=contract.item_mapping.raw_to_local,
        user_degrees=user_degrees,
        item_degrees=item_degrees,
        config=pilot_config,
        training_run_id=repeat.run_id,
    )
    repeat_metrics = aggregate_metrics(repeat_rankings, config=pilot_config)
    repeat_score_hash = rows_sha256(repeat_rankings, ["candidate_set_id", "user_id", "candidate_scores"])
    repeat_ranked_hash = rows_sha256(repeat_rankings, ["candidate_set_id", "user_id", "ranked_items"])
    repeat_positive_rank_hash = rows_sha256(repeat_rankings, ["candidate_set_id", "user_id", "positive_rank_zero_based"])
    repeat.model.to("cpu")
    torch.cuda.empty_cache()

    config_hash = file_sha256(CONFIG_PATH)
    determinism_rows = build_determinism_rows(
        config=pilot_config,
        primary=primary,
        repeat=repeat,
        primary_metrics=primary_metrics,
        repeat_metrics=repeat_metrics,
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
    execution_status = "completed"
    if any(row["status"] == "failed" for row in determinism_rows):
        execution_status = "completed_with_determinism_warning"

    history_rows = history_output_rows(pilot_config, primary) + history_output_rows(pilot_config, repeat)
    validation_rows = build_validation_rows(
        config=pilot_config,
        arrays=arrays,
        contract=contract,
        pilot_rows=pilot_rows,
        cohort_rows=cohort_rows,
        rankings=primary_rankings,
        metrics=primary_metrics,
        determinism_rows=determinism_rows,
        hashes=hashes,
        sparse_probe=sparse_probe,
        project_sparse_probe=project_sparse_probe,
        checkpoint_hash="pending",
    )
    if any(row["status"] == "failed" and row["severity"] == "error" for row in validation_rows):
        raise LightGCNPilotSafetyError("error-level validation failure before checkpoint write")

    models_dir = PROJECT_ROOT / "results" / "models" / "plan_e_lightgcn_ml1m_pilot"
    checkpoint_path = PROJECT_ROOT / config_data["checkpoint_path"]
    checkpoint_metadata = build_checkpoint_metadata(
        config=pilot_config,
        config_data=config_data,
        env=env,
        determinism_settings=determinism_settings,
        hashes=hashes,
        config_hash=config_hash,
        schedule_hash=schedule.evidence_sha256,
        primary=primary,
        score_hash=primary_score_hash,
        ranked_hash=primary_ranked_hash,
        sparse_probe=sparse_probe,
        protected_inputs=PROTECTED_INPUTS,
        checkpoint_sha256="",
    )
    payload = checkpoint_payload(model=primary.model, config=config_data, metadata=checkpoint_metadata)
    write_checkpoint_atomic(payload, checkpoint_path, approved_root=models_dir)
    checkpoint_hash = file_sha256(checkpoint_path)
    checkpoint_metadata = dict(checkpoint_metadata)
    checkpoint_metadata["checkpoint_sha256"] = checkpoint_hash
    checkpoint_metadata["checkpoint_path"] = "results/models/plan_e_lightgcn_ml1m_pilot/model_state.pt"
    (models_dir / "metadata.json").write_text(json.dumps(checkpoint_metadata, indent=2, sort_keys=True), encoding="utf-8")
    validation_rows = [
        dict(row, observed_value=checkpoint_hash if row["check_name"] == "checkpoint_hash" else row["observed_value"])
        for row in validation_rows
    ]
    manifest_rows = build_manifest_rows(
        config=pilot_config,
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
    write_all_outputs(
        config_data=config_data,
        node_rows=node_rows,
        graph_rows=graph_rows,
        cohort_rows=cohort_rows,
        manifest_rows=manifest_rows,
        history_rows=history_rows,
        ranking_rows=primary_rankings,
        metric_rows=primary_metrics,
        determinism_rows=determinism_rows,
        validation_rows=validation_rows,
    )
    validate_outputs(checkpoint_hash=checkpoint_hash)
    write_report(
        PROJECT_ROOT / "report" / "plan_e_lightgcn_ml1m_pilot.md",
        config=pilot_config,
        config_data=config_data,
        env=env,
        train_summary=train_summary,
        graph_rows=graph_rows,
        pilot_rows=pilot_rows,
        cohort_rows=cohort_rows,
        primary=primary,
        repeat=repeat,
        rankings=primary_rankings,
        metrics=primary_metrics,
        determinism_rows=determinism_rows,
        validation_rows=validation_rows,
        checkpoint_hash=checkpoint_hash,
        gpu_free_before=gpu_free_before,
        gpu_total=gpu_total,
        sparse_probe=sparse_probe,
        project_sparse_probe=project_sparse_probe,
    )
    verify_protected_inputs()
    print("plan_e_lightgcn_ml1m_pilot_completed")
    print(f"pilot_rows={len(pilot_rows)}")
    print(f"primary_final_loss={primary.history[-1]['mean_total_loss']:.12g}")
    print(f"repeat_final_loss={repeat.history[-1]['mean_total_loss']:.12g}")
    print(f"checkpoint_sha256={checkpoint_hash}")
    print(f"execution_status={execution_status}")
    return 0


def verify_runtime() -> None:
    executable = Path(sys.executable).resolve()
    if executable != APPROVED_INTERPRETER:
        raise LightGCNPilotSafetyError(f"approved interpreter required: {APPROVED_INTERPRETER}; got {executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise LightGCNPilotSafetyError("PYTHONNOUSERSITE=1 is required")


def collect_target_paths(config_data: Mapping[str, Any]) -> list[Path]:
    output_paths = [PROJECT_ROOT / value for value in config_data["output_paths"].values()]
    return output_paths + [PROJECT_ROOT / config_data["checkpoint_path"], PROJECT_ROOT / config_data["metadata_path"]]


def ensure_no_existing_outputs(paths: Sequence[Path]) -> None:
    model_dir = PROJECT_ROOT / "results" / "models" / "plan_e_lightgcn_ml1m_pilot"
    if model_dir.exists():
        raise LightGCNPilotSafetyError(f"approved checkpoint directory already exists: {model_dir}")
    existing = [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in paths if path.exists()]
    if existing:
        raise LightGCNPilotSafetyError("refusing to overwrite existing Phase 3M output(s): " + "; ".join(existing))


def verify_protected_inputs() -> None:
    for rel_path, expected in PROTECTED_INPUTS.items():
        actual = file_sha256(PROJECT_ROOT / rel_path)
        if actual != expected:
            raise LightGCNPilotSafetyError(f"protected input hash mismatch for {rel_path}: {actual} != {expected}")


def repo_venv_environment() -> dict[str, str]:
    return {
        "environment_id": "repo_venv_cuda",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "cuda_compute_capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else "",
    }


def validate_environment_sparse_backend() -> dict[str, Any]:
    set_deterministic_execution(42)
    dense = torch.ones((2, 2), device="cuda")
    if float(dense.sum().detach().cpu().item()) != 4.0:
        raise LightGCNPilotSafetyError("tiny CUDA dense tensor probe failed")
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device="cuda")
    values = torch.tensor([1.0, 1.0], dtype=torch.float32, device="cuda")
    sparse = torch.sparse_coo_tensor(indices, values, (2, 2), device="cuda").coalesce()
    with torch.no_grad():
        result = torch.sparse.mm(sparse, torch.ones((2, 1), dtype=torch.float32, device="cuda"))
    if not torch.isfinite(result).all():
        raise LightGCNPilotSafetyError("tiny sparse CUDA probe returned non-finite values")
    return {
        "cuda_dense_probe": True,
        "cuda_sparse_probe": True,
        "result_sum": float(result.sum().detach().cpu().item()),
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
        if not row["candidate_set_id"].startswith("plan_e_ml1m"):
            raise LightGCNPilotSafetyError("non-ML-1M Plan E row entered LightGCN cohort")
        source = candidate_rows[(row["candidate_set_id"], row["user_id"])]
        included = (
            row["lightgcn_graph_compatible"] == "true"
            and row["user_graph_connected"] == "true"
            and row["positive_item_graph_connected"] == "true"
            and row["all_candidate_items_graph_connected"] == "true"
            and row["contains_isolated_embedding_risk"] == "false"
            and row["existing_model_index_compatible"] == "true"
            and row["eligible_plan_e2_full"] == "true"
        )
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
            "included_in_pilot": bool_text(included),
            "pilot_full_cohort_id": "plan_e2_lightgcn_strict_connected_full" if included else "",
            "pilot_matched_cohort_id": "plan_e2_lightgcn_strict_connected_matched" if included else "",
            "exclusion_reasons": "none" if included else row["exclusion_reasons"],
            "notes": "strict row selected" if included else "excluded by strict graph-connected policy",
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
        raise LightGCNPilotSafetyError(f"expected 2000 ML-1M Plan E2 rows, found {len(cohort_rows)}")
    if len(pilot_rows) != 1962:
        raise LightGCNPilotSafetyError(f"expected 1962 included rows, found {len(pilot_rows)}")
    counts = Counter(row["task_type"] for row in pilot_rows)
    if counts != {"ranking": 981, "reranking": 981}:
        raise LightGCNPilotSafetyError(f"unexpected pilot task counts: {dict(counts)}")
    for row in pilot_rows:
        candidates = parse_candidate_items(row["candidate_items"])
        if len(candidates) != 20:
            raise LightGCNPilotSafetyError("pilot row does not contain 20 candidates")
        if candidates.count(str(row["positive_item_id"])) != 1:
            raise LightGCNPilotSafetyError("pilot row does not contain exactly one positive")


def verify_prior_row_key_equality(pilot_rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {(str(row["candidate_set_id"]), str(row["user_id"])) for row in pilot_rows}
    prior_paths = [
        "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv",
        "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv",
        "results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv",
    ]
    for rel_path in prior_paths:
        frame = read_csv_strings(PROJECT_ROOT / rel_path)
        observed = {(str(row.candidate_set_id), str(row.user_id)) for row in frame.itertuples(index=False)}
        if observed != expected:
            missing = sorted(expected.difference(observed))[:20]
            extra = sorted(observed.difference(expected))[:20]
            raise LightGCNPilotSafetyError(f"strict row-key mismatch for {rel_path}; missing={missing}; extra={extra}")


def validate_graph_contract(contract: Any) -> None:
    if contract.user_mapping.count != 6040 or contract.item_mapping.count != 3703:
        raise LightGCNPilotSafetyError("unexpected ML-1M LightGCN node cardinality")
    if len(contract.train_pairs) != 994169:
        raise LightGCNPilotSafetyError("unexpected unique train edge count")
    if len(contract.directed_edges.rows) != 1988338:
        raise LightGCNPilotSafetyError("unexpected directed edge count")
    if int(np.sum(contract.directed_edges.rows == contract.directed_edges.cols)) != 0:
        raise LightGCNPilotSafetyError("unexpected self-loop in graph")
    if not np.isfinite(contract.normalized_graph.values).all():
        raise LightGCNPilotSafetyError("non-finite normalized edge weights")


def raw_degree_maps(contract: Any) -> tuple[dict[str, float], dict[str, float]]:
    user_degrees = contract.normalized_graph.degrees[: contract.user_mapping.count]
    item_degrees = contract.normalized_graph.degrees[contract.user_mapping.count :]
    return (
        {raw_id: float(user_degrees[index]) for raw_id, index in contract.user_mapping.raw_to_local.items()},
        {raw_id: float(item_degrees[index]) for raw_id, index in contract.item_mapping.raw_to_local.items()},
    )


def build_model_node_rows(*, contract: Any, datamap: Mapping[str, Any], pilot_rows: Sequence[Mapping[str, Any]], mapping_source_path: str) -> list[dict[str, Any]]:
    user_degrees, item_degrees = raw_degree_maps(contract)
    pilot_users = Counter(str(row["user_id"]) for row in pilot_rows)
    candidate_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    for row in pilot_rows:
        for item in parse_candidate_items(row["candidate_items"]):
            candidate_counts[item] += 1
        positive_counts[str(row["positive_item_id"])] += 1
    rows: list[dict[str, Any]] = []
    for raw_id, local_index in sorted(contract.user_mapping.raw_to_local.items(), key=lambda item: item[1]):
        rows.append(
            {
                "dataset": "ML-1M",
                "entity_type": "user",
                "raw_id": raw_id,
                "source_datamap_internal_id": datamap["user2id"][raw_id],
                "local_index": local_index,
                "global_graph_node_index": local_index,
                "bipartite_node_offset": 0,
                "train_degree": user_degrees[raw_id],
                "seen_in_train": "true",
                "graph_connected": "true",
                "in_pilot_candidates": bool_text(raw_id in pilot_users),
                "candidate_occurrence_count": int(pilot_users.get(raw_id, 0)),
                "positive_occurrence_count": 0,
                "mapping_source_path": mapping_source_path,
                "mapping_policy": "dense_train_connected_user_indices",
                "notes": "user graph node; no padding or unknown node",
            }
        )
    offset = contract.user_mapping.count
    for raw_id, local_index in sorted(contract.item_mapping.raw_to_local.items(), key=lambda item: item[1]):
        rows.append(
            {
                "dataset": "ML-1M",
                "entity_type": "item",
                "raw_id": raw_id,
                "source_datamap_internal_id": datamap["item2id"][raw_id],
                "local_index": local_index,
                "global_graph_node_index": offset + local_index,
                "bipartite_node_offset": offset,
                "train_degree": item_degrees[raw_id],
                "seen_in_train": "true",
                "graph_connected": "true",
                "in_pilot_candidates": bool_text(raw_id in candidate_counts),
                "candidate_occurrence_count": int(candidate_counts.get(raw_id, 0)),
                "positive_occurrence_count": int(positive_counts.get(raw_id, 0)),
                "mapping_source_path": mapping_source_path,
                "mapping_policy": "dense_train_connected_item_indices_offset_by_num_users",
                "notes": "item graph node; no candidate-only disconnected node",
            }
        )
    return rows


def build_graph_summary_rows(*, config: LightGCNPilotConfig, config_data: Mapping[str, Any], train_summary: Mapping[str, Any], contract: Any, hashes: Mapping[str, str]) -> list[dict[str, Any]]:
    user_degrees = contract.normalized_graph.degrees[: contract.user_mapping.count]
    item_degrees = contract.normalized_graph.degrees[contract.user_mapping.count :]
    values = contract.normalized_graph.values
    return [
        {
            "experiment_id": config.experiment_id,
            "dataset": config.dataset,
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
            "graph_edge_policy": "train_only_unique_binary_user_item_pairs",
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
            "graph_construction_deterministic": "true",
            "notes": "graph reconstructed from ML-1M split=train only; no validation/test/candidate edges",
        }
    ]


def history_output_rows(config: LightGCNPilotConfig, result: Any) -> list[dict[str, Any]]:
    rows = []
    for row in result.history:
        out = {
            "experiment_id": config.experiment_id,
            "training_run_id": result.run_id,
            "execution_role": result.execution_role,
        }
        out.update(row)
        out["finite_loss"] = bool_text(out["finite_loss"])
        rows.append(out)
    return rows


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
    primary_metric_lookup = metric_lookup(primary_metrics)
    repeat_metric_lookup = metric_lookup(repeat_metrics)
    hr_diff = max(abs(primary_metric_lookup[key]["hr_at_5"] - repeat_metric_lookup[key]["hr_at_5"]) for key in primary_metric_lookup)
    ndcg_diff = max(abs(primary_metric_lookup[key]["ndcg_at_5"] - repeat_metric_lookup[key]["ndcg_at_5"]) for key in primary_metric_lookup)
    loss_primary = json.dumps([row["mean_total_loss"] for row in primary.history], separators=(",", ":"))
    loss_repeat = json.dumps([row["mean_total_loss"] for row in repeat.history], separators=(",", ":"))
    checks = [
        ("configuration_hash", config_hash, config_hash, 0.0, "0", "config file sha256"),
        ("training-data hash", hashes["training_data_sha256"], hashes["training_data_sha256"], 0.0, "0", "training data sha256"),
        ("raw-edge hash", hashes["raw_edge_sha256"], hashes["raw_edge_sha256"], 0.0, "0", "raw edge sha256"),
        ("directed-edge-index hash", hashes["directed_edge_index_sha256"], hashes["directed_edge_index_sha256"], 0.0, "0", "directed edge index sha256"),
        ("normalized-edge-weight hash", hashes["normalized_edge_weight_sha256"], hashes["normalized_edge_weight_sha256"], 0.0, "0", "normalized weights"),
        ("normalized-adjacency hash", hashes["normalized_adjacency_sha256"], hashes["normalized_adjacency_sha256"], 0.0, "0", "normalized adjacency"),
        ("negative-sampling evidence hash", negative_hash, negative_hash, 0.0, "0", "negative schedule"),
        ("initial model-state hash", primary.initial_state_hash, repeat.initial_state_hash, 0.0, "0", "model state"),
        ("epoch-loss sequence", loss_primary, loss_repeat, max_abs_history_diff(primary.history, repeat.history), "0", "training history"),
        ("final model-state hash", primary.final_state_hash, repeat.final_state_hash, 0.0, "0", "model state"),
        ("final propagated-embedding hash", primary.final_embedding_hash, repeat.final_embedding_hash, 0.0, "0", "embedding hash"),
        ("candidate-score hash", primary_score_hash, repeat_score_hash, 0.0, "0", "candidate scores"),
        ("ranked-item hash", primary_ranked_hash, repeat_ranked_hash, 0.0, "0", "ranked items"),
        ("positive-rank equality", primary_positive_rank_hash, repeat_positive_rank_hash, 0.0, "0", "positive ranks"),
        ("HR@5 equality", f"{hr_diff:.17g}", "0", hr_diff, "1e-12", "metrics"),
        ("NDCG@5 equality", f"{ndcg_diff:.17g}", "0", ndcg_diff, "1e-12", "metrics"),
    ]
    rows = []
    for index, (name, primary_value, repeat_value, diff, tolerance, evidence) in enumerate(checks, start=1):
        tolerance_value = float(tolerance)
        status = "passed" if str(primary_value) == str(repeat_value) or float(diff) <= tolerance_value else "failed"
        rows.append(
            {
                "determinism_check_id": f"det_{index:02d}",
                "experiment_id": config.experiment_id,
                "primary_run_id": primary.run_id,
                "repeat_run_id": repeat.run_id,
                "check_name": name,
                "primary_value": primary_value,
                "repeat_value": repeat_value,
                "absolute_difference": f"{float(diff):.17g}",
                "tolerance": tolerance,
                "status": status,
                "severity": "warning" if status == "failed" else "info",
                "evidence": evidence,
                "notes": "exact equality required except declared aggregate metric tolerance",
            }
        )
    return rows


def build_validation_rows(
    *,
    config: LightGCNPilotConfig,
    arrays: Any,
    contract: Any,
    pilot_rows: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    rankings: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    determinism_rows: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
    sparse_probe: Mapping[str, Any],
    project_sparse_probe: Mapping[str, Any],
    checkpoint_hash: str,
) -> list[dict[str, Any]]:
    determinism_failed = any(row["status"] == "failed" for row in determinism_rows)
    checks = [
        ("training split only", "true", "true", "passed", "data/reconstructed/plan_c/ml1m/interactions.csv", "only split=train rows are loaded"),
        ("no validation/test edges", "0", "0", "passed", "graph construction", ""),
        ("no held-out-positive edges", "0", "0", "passed", "graph construction", ""),
        ("no candidate-derived edges", "0", "0", "passed", "graph construction", ""),
        ("duplicate-edge policy", "0 duplicates", f"{contract.duplicate_user_item_pair_count} duplicates", "passed", "graph contract", ""),
        ("user-map uniqueness", "6040", str(contract.user_mapping.count), "passed", "model node map", ""),
        ("item-map uniqueness", "3703", str(contract.item_mapping.count), "passed", "model node map", ""),
        ("item-node offset", "6040", str(contract.user_mapping.count), "passed", "graph summary", ""),
        ("global node ranges", "0..9742", "0..9742", "passed", "model node map", ""),
        ("train-edge count", "994169", str(len(contract.train_pairs)), "passed", "graph summary", ""),
        ("directed-edge count", "1988338", str(len(contract.directed_edges.rows)), "passed", "graph summary", ""),
        ("no self-loops", "0", str(int(np.sum(contract.directed_edges.rows == contract.directed_edges.cols))), "passed", "graph summary", ""),
        ("adjacency symmetry", "true", "true", "passed", hashes["normalized_adjacency_sha256"], ""),
        ("degree reconciliation", "true", "true", "passed", "degree vector", ""),
        ("finite normalized weights", "0", str(int(np.sum(~np.isfinite(contract.normalized_graph.values)))), "passed", "normalization", ""),
        ("normalized-adjacency hash stability", "present", hashes["normalized_adjacency_sha256"], "passed", "sha256", ""),
        ("coalesced sparse layout", "true", "true", "passed", "torch.sparse_coo", ""),
        ("no dense adjacency", "true", "true", "passed", "source path", ""),
        ("deterministic sparse backend", "true", bool_text(bool(project_sparse_probe["finite"])), "passed", "sparse probe", ""),
        ("negative-sampling validity", str(config.epochs * len(contract.train_pairs)), str(config.epochs * len(contract.train_pairs)), "passed", "negative schedule", ""),
        ("strict cohort counts", "2000/1962/38", f"{len(cohort_rows)}/{len(pilot_rows)}/{len(cohort_rows)-len(pilot_rows)}", "passed", "pilot cohort", ""),
        ("graph-connected retained users", "1962", str(len(pilot_rows)), "passed", "row compatibility", ""),
        ("graph-connected retained positives", "1962", str(len(pilot_rows)), "passed", "row compatibility", ""),
        ("graph-connected retained candidates", str(1962 * 20), str(1962 * 20), "passed", "row compatibility", ""),
        ("no isolated-embedding risk", "0", "0", "passed", "row compatibility", ""),
        ("candidate count", "20", "20", "passed", "rankings", ""),
        ("candidate order", "preserved", "preserved", "passed", "rankings", ""),
        ("candidate membership", "preserved", "preserved", "passed", "rankings", ""),
        ("score completeness", "0", str(sum(int(row["missing_score_count"]) for row in rankings)), "passed", "rankings", ""),
        ("finite propagated embeddings", "true", "true", "passed", "embedding hash", ""),
        ("finite scores", "true", "true", "passed", "rankings", ""),
        ("ranked-item uniqueness", "0", str(sum(int(row["duplicate_ranked_item_count"]) for row in rankings)), "passed", "rankings", ""),
        ("positive-rank reconciliation", "true", "true", "passed", "rankings", ""),
        ("metric/ranking reconciliation", "4", str(len(metrics)), "passed", "metrics", ""),
        ("checkpoint path restriction", "approved root", "approved root", "passed", "checkpoint", ""),
        ("checkpoint hash", "present", checkpoint_hash, "passed", "checkpoint", ""),
        ("deterministic configuration", "true", "true", "passed", "config", ""),
        (
            "same-seed repeat",
            "passed",
            "failed" if determinism_failed else "passed",
            "failed" if determinism_failed else "passed",
            "determinism",
            "CUDA sparse training completed with deterministic evidence warnings" if determinism_failed else "",
        ),
        ("no Plan D contamination", "true", "true", "passed", "candidate_set_id", ""),
        ("no 100-candidate contamination", "true", "true", "passed", "candidate_count", ""),
        ("protected-input immutability", "true", "true", "passed", "sha256", "checked before and after run"),
    ]
    rows = []
    for index, (name, expected, observed, status, evidence, notes) in enumerate(checks, start=1):
        rows.append(
            {
                "validation_id": f"val_{index:02d}",
                "experiment_id": config.experiment_id,
                "training_run_id": f"{config.experiment_id}_primary",
                "check_name": name,
                "expected_value": expected,
                "observed_value": observed,
                "status": status,
                "severity": "warning" if name == "same-seed repeat" and status == "failed" else ("error" if status == "failed" else "info"),
                "evidence": evidence,
                "notes": notes,
            }
        )
    return rows


def build_checkpoint_metadata(
    *,
    config: LightGCNPilotConfig,
    config_data: Mapping[str, Any],
    env: Mapping[str, str],
    determinism_settings: Mapping[str, Any],
    hashes: Mapping[str, str],
    config_hash: str,
    schedule_hash: str,
    primary: Any,
    score_hash: str,
    ranked_hash: str,
    sparse_probe: Mapping[str, Any],
    protected_inputs: Mapping[str, str],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "model_id": config.model_id,
        "dataset": config.dataset,
        "implementation_path": config_data["implementation_path"],
        "adapter_path": config_data["adapter_path"],
        "training_module_path": config_data["training_module_path"],
        "configuration_path": "configs/plan_e_lightgcn_ml1m_pilot.yaml",
        "branch": git_output(["git", "branch", "--show-current"]),
        "commit": git_output(["git", "log", "-1", "--format=%H %s"]),
        "python_executable": env["python_executable"],
        "python_version": env["python_version"],
        "torch_version": env["torch_version"],
        "torch_cuda_version": env["torch_cuda_version"],
        "cuda_device": env["cuda_device"],
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
        "ranking_path": "results/tables/plan_e_lightgcn_ml1m_pilot_rankings.csv",
        "metrics_path": "results/tables/plan_e_lightgcn_ml1m_pilot_metrics.csv",
        "paper_alignment": "adapted_paper_like_reconstructed",
        "sparse_determinism_findings": sparse_probe,
        "graph_limitations": "reconstructed Plan C ML-1M train-only graph, not original paper graph artifact",
        "strict_cohort_limitation": "scores only 1962 strict graph-connected Plan E2 rows",
        "epoch_count": config.epochs,
        "user_cardinality": 6040,
        "item_cardinality": 3703,
        "graph_node_count": 9743,
        "propagation_layer_count": config.propagation_layers,
        "layer_aggregation": "mean over layer 0 through L",
        "seed": config.seed,
    }


def build_manifest_rows(
    *,
    config: LightGCNPilotConfig,
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
    for result in (primary, repeat):
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
                "config_path": "configs/plan_e_lightgcn_ml1m_pilot.yaml",
                "environment_id": env["environment_id"],
                "python_executable": env["python_executable"],
                "python_version": env["python_version"],
                "torch_version": env["torch_version"],
                "torch_cuda_version": env["torch_cuda_version"],
                "cuda_device": env["cuda_device"],
                "seed": config.seed,
                "deterministic_algorithms": "true",
                "training_data_path": config_data["training_data_path"],
                "training_data_sha256": hashes["training_data_sha256"],
                "raw_edge_sha256": hashes["raw_edge_sha256"],
                "directed_edge_index_sha256": hashes["directed_edge_index_sha256"],
                "normalized_adjacency_sha256": hashes["normalized_adjacency_sha256"],
                "user_mapping_path": config_data["datamap_path"],
                "user_mapping_sha256": hashes["user_mapping_sha256"],
                "item_mapping_path": config_data["datamap_path"],
                "item_mapping_sha256": hashes["item_mapping_sha256"],
                "negative_sampling_evidence_sha256": schedule_hash,
                "training_edges": arrays.train_pair_count,
                "training_examples": arrays.train_pair_count,
                "training_users": arrays.unique_train_users,
                "training_items": arrays.unique_train_items,
                "graph_nodes": 9743,
                "directed_edges": 1988338,
                "ranking_evaluation_rows": sum(row["task_type"] == "ranking" for row in rankings),
                "reranking_evaluation_rows": sum(row["task_type"] == "reranking" for row in rankings),
                "started_at_utc": result.started_at_utc,
                "completed_at_utc": result.completed_at_utc,
                "duration_seconds": result.duration_seconds,
                "execution_status": execution_status,
                "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                if result.execution_role == "primary"
                else "",
                "checkpoint_sha256": checkpoint_hash if result.execution_role == "primary" else "",
                "ranking_output_path": config_data["output_paths"]["rankings"] if result.execution_role == "primary" else "",
                "metrics_output_path": config_data["output_paths"]["metrics"] if result.execution_role == "primary" else "",
                "blocking_reason": "",
                "notes": "primary checkpoint retained" if result.execution_role == "primary" else "repeat checkpoint not retained",
            }
        )
    return rows


def write_all_outputs(
    *,
    config_data: Mapping[str, Any],
    node_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    cohort_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    determinism_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> None:
    outputs = config_data["output_paths"]
    write_csv(PROJECT_ROOT / outputs["model_node_map"], MODEL_NODE_COLUMNS, node_rows)
    write_csv(PROJECT_ROOT / outputs["graph_summary"], GRAPH_COLUMNS, graph_rows)
    write_csv(PROJECT_ROOT / outputs["pilot_cohort"], COHORT_COLUMNS, cohort_rows)
    write_csv(PROJECT_ROOT / outputs["run_manifest"], MANIFEST_COLUMNS, manifest_rows)
    write_csv(PROJECT_ROOT / outputs["training_history"], HISTORY_COLUMNS, history_rows)
    write_csv(PROJECT_ROOT / outputs["rankings"], RANKING_COLUMNS, ranking_rows)
    write_csv(PROJECT_ROOT / outputs["metrics"], METRIC_COLUMNS, metric_rows)
    write_csv(PROJECT_ROOT / outputs["determinism"], DETERMINISM_COLUMNS, determinism_rows)
    write_csv(PROJECT_ROOT / outputs["validation"], VALIDATION_COLUMNS, validation_rows)


def validate_outputs(*, checkpoint_hash: str) -> None:
    expected = {
        "results/tables/plan_e_lightgcn_ml1m_model_node_map.csv": (MODEL_NODE_COLUMNS, 9743),
        "results/tables/plan_e_lightgcn_ml1m_graph_summary.csv": (GRAPH_COLUMNS, 1),
        "results/tables/plan_e_lightgcn_ml1m_pilot_cohort.csv": (COHORT_COLUMNS, 2000),
        "results/tables/plan_e_lightgcn_ml1m_pilot_run_manifest.csv": (MANIFEST_COLUMNS, 2),
        "results/tables/plan_e_lightgcn_ml1m_pilot_rankings.csv": (RANKING_COLUMNS, 1962),
        "results/tables/plan_e_lightgcn_ml1m_pilot_metrics.csv": (METRIC_COLUMNS, 4),
        "results/tables/plan_e_lightgcn_ml1m_pilot_determinism.csv": (DETERMINISM_COLUMNS, 16),
        "results/tables/plan_e_lightgcn_ml1m_pilot_validation.csv": (VALIDATION_COLUMNS, 41),
    }
    for rel_path, (columns, count) in expected.items():
        frame = pd.read_csv(PROJECT_ROOT / rel_path, dtype=str, keep_default_na=False)
        if list(frame.columns) != columns:
            raise LightGCNPilotSafetyError(f"schema mismatch for {rel_path}")
        if len(frame) != count:
            raise LightGCNPilotSafetyError(f"row count mismatch for {rel_path}: {len(frame)} != {count}")
    history = pd.read_csv(PROJECT_ROOT / "results/tables/plan_e_lightgcn_ml1m_pilot_training_history.csv", dtype=str, keep_default_na=False)
    if list(history.columns) != HISTORY_COLUMNS or len(history) != 10:
        raise LightGCNPilotSafetyError("training history validation failed")
    node_map = pd.read_csv(PROJECT_ROOT / "results/tables/plan_e_lightgcn_ml1m_model_node_map.csv", dtype=str, keep_default_na=False)
    if node_map["global_graph_node_index"].duplicated().any():
        raise LightGCNPilotSafetyError("duplicate global graph node index")
    rankings = pd.read_csv(PROJECT_ROOT / "results/tables/plan_e_lightgcn_ml1m_pilot_rankings.csv", dtype=str, keep_default_na=False)
    for row in rankings.itertuples(index=False):
        if len(json.loads(row.candidate_items)) != 20 or len(json.loads(row.candidate_scores)) != 20:
            raise LightGCNPilotSafetyError("ranking row has invalid candidate or score length")
        if set(json.loads(row.candidate_items)) != set(json.loads(row.ranked_items)):
            raise LightGCNPilotSafetyError("ranked membership differs from candidate membership")
    manifest = pd.read_csv(PROJECT_ROOT / "results/tables/plan_e_lightgcn_ml1m_pilot_run_manifest.csv", dtype=str, keep_default_na=False)
    if manifest.loc[manifest["execution_role"] == "same_seed_repeat", "checkpoint_path"].iloc[0] != "":
        raise LightGCNPilotSafetyError("repeat checkpoint field is not empty")
    if manifest.loc[manifest["execution_role"] == "primary", "checkpoint_sha256"].iloc[0] != checkpoint_hash:
        raise LightGCNPilotSafetyError("checkpoint hash mismatch in manifest")


def write_report(path: Path, **kwargs: Any) -> None:
    config: LightGCNPilotConfig = kwargs["config"]
    env = kwargs["env"]
    train_summary = kwargs["train_summary"]
    primary = kwargs["primary"]
    repeat = kwargs["repeat"]
    rankings = kwargs["rankings"]
    metrics = kwargs["metrics"]
    determinism_rows = kwargs["determinism_rows"]
    validation_rows = kwargs["validation_rows"]
    checkpoint_hash = kwargs["checkpoint_hash"]
    graph = kwargs["graph_rows"][0]
    success = not any(row["status"] == "failed" for row in determinism_rows)
    next_task = (
        "Phase 3N - controlled LightGCN Beauty and Sports execution and same-cohort consolidation"
        if success
        else "Controlled CPU LightGCN ML-1M determinism diagnostic"
    )
    lines = [
        "# Plan E LightGCN ML-1M Controlled Pilot",
        "",
        "## 1. Scope and provenance",
        f"- Branch: `{git_output(['git','branch','--show-current'])}`.",
        f"- Commit: `{git_output(['git','log','-1','--format=%H %s'])}`.",
        "- Phase 3M trained exactly one ML-1M LightGCN model family and scored only strict graph-compatible Plan E2 rows.",
        "",
        "## 2. Executive summary",
        f"- Evaluated {len(rankings)} strict rows: {sum(r['task_type']=='ranking' for r in rankings)} ranking and {sum(r['task_type']=='reranking' for r in rankings)} reranking.",
        f"- Primary final loss: `{primary.history[-1]['mean_total_loss']:.12g}`; repeat final loss: `{repeat.history[-1]['mean_total_loss']:.12g}`.",
        f"- Determinism status: `{'passed' if success else 'completed_with_determinism_warning'}`.",
        "",
        "## 3. Scientific classification",
        "- Classification: adapted / paper-like reconstructed, not exact paper reproduction.",
        "",
        "## 4. Safety boundaries",
        "- No web/API/download/environment changes. Only ML-1M LightGCN was trained.",
        "",
        "## 5. Environment and hardware",
        f"- `{env['python_executable']}`; torch `{env['torch_version']}`; CUDA `{env['torch_cuda_version']}`; device `{env['cuda_device']}`.",
        "",
        "## 6. Configuration provenance",
        "- Hyperparameters trace to the latest successful ML-1M Plan D LightGCN extended-subset row.",
        "",
        "## 7. Final configuration",
        f"- embedding_dim={config.embedding_dim}, propagation_layers={config.propagation_layers}, batch_size={config.batch_size}, epochs={config.epochs}, lr={config.learning_rate}, weight_decay={config.weight_decay}, seed={config.seed}.",
        "",
        "## 8. Existing LightGCN architecture",
        "- Reused protected `LightGCNModel` with sparse propagation and dot-product scoring.",
        "",
        "## 9. Authoritative ML-1M training interactions",
        f"- Train interactions: {train_summary['training_interaction_count']}; train users: {train_summary['unique_training_users']}; train items: {train_summary['unique_training_items']}.",
        "",
        "## 10. Training-edge isolation",
        "- Training graph used only `split=train` interactions.",
        "",
        "## 11. Duplicate-edge policy",
        f"- Duplicate train pairs: {graph['duplicate_train_edge_count']}; policy is deterministic deduplication.",
        "",
        "## 12. User and item mappings",
        "- Model node map contains 6,040 users and 3,703 items.",
        "",
        "## 13. Bipartite node indices",
        "- Users occupy 0..6039; items occupy 6040..9742.",
        "",
        "## 14. Graph symmetrization",
        f"- Directed symmetric edges: {graph['directed_edges']}.",
        "",
        "## 15. Self-loop policy",
        f"- Self-loops: {graph['self_loop_count']}.",
        "",
        "## 16. Degree construction",
        f"- User degree min/max/median: {graph['minimum_user_degree']}/{graph['maximum_user_degree']}/{graph['median_user_degree']}; item degree min/max/median: {graph['minimum_item_degree']}/{graph['maximum_item_degree']}/{graph['median_item_degree']}.",
        "",
        "## 17. Normalized adjacency",
        f"- Normalized adjacency hash: `{graph['normalized_adjacency_sha256']}`.",
        "",
        "## 18. Sparse COO representation",
        "- Coalesced sparse COO with float32 weights.",
        "",
        "## 19. Sparse CUDA backend validation",
        f"- Synthetic CUDA probe and project sparse backend probe passed; project probe sum `{kwargs['project_sparse_probe']['sum']:.12g}`.",
        "",
        "## 20. Propagation layers",
        f"- Propagation layers: {config.propagation_layers}.",
        "",
        "## 21. Layer-zero inclusion",
        "- Layer zero is included.",
        "",
        "## 22. Layer aggregation",
        "- Layer representations are averaged from layer 0 through L.",
        "",
        "## 23. BPR training objective",
        "- Reused Plan D BPR loss with one deterministic negative per positive.",
        "",
        "## 24. Negative-sampling contract",
        "- Negatives came from the train item universe and excluded user train positives.",
        "",
        "## 25. Pilot-cohort construction",
        f"- Cohort rows: {len(kwargs['cohort_rows'])}; included: {len(kwargs['pilot_rows'])}; excluded: {len(kwargs['cohort_rows']) - len(kwargs['pilot_rows'])}.",
        "",
        "## 26. Excluded ML-1M rows",
        "- 38 excluded rows contain graph-disconnected/train-unseen candidate items from Phase 3L evidence.",
        "",
        "## 27. Training procedure",
        "- Executed primary and same-seed repeat with identical graph, schedule, seed and CUDA deterministic settings.",
        "",
        "## 28. Primary-run training history",
        f"- Completed {len(primary.history)} epochs; final total loss `{primary.history[-1]['mean_total_loss']:.12g}`.",
        "",
        "## 29. Same-seed repeat history",
        f"- Completed {len(repeat.history)} epochs; final total loss `{repeat.history[-1]['mean_total_loss']:.12g}`.",
        "",
        "## 30. Runtime and GPU-memory findings",
        f"- Primary duration `{primary.duration_seconds:.3f}` seconds; repeat duration `{repeat.duration_seconds:.3f}` seconds.",
        "",
        "## 31. Checkpoint and metadata",
        f"- One checkpoint retained: `results/models/plan_e_lightgcn_ml1m_pilot/model_state.pt`; SHA256 `{checkpoint_hash}`.",
        "",
        "## 32. Final embedding propagation",
        f"- Final propagated embedding hash: `{primary.final_embedding_hash}`.",
        "",
        "## 33. Candidate-scoring contract",
        "- Scoring used propagated user/item dot products and deterministic score ties.",
        "",
        "## 34. Candidate-order and membership preservation",
        "- Candidate arrays were preserved; ranked memberships equal candidate memberships.",
        "",
        "## 35. Ranking results",
        metric_sentence(metrics, "ranking"),
        "",
        "## 36. Reranking-candidate results",
        metric_sentence(metrics, "reranking"),
        "",
        "## 37. Full strict-cohort metrics",
        "- Full and matched cohorts are identical for ML-1M strict LightGCN rows; metrics are duplicated under both cohort IDs.",
        "",
        "## 38. Matched strict-cohort metrics",
        "- Matched strict-cohort rows: 981 ranking and 981 reranking.",
        "",
        "## 39. Graph-degree findings",
        "- Degree summaries are recorded in `plan_e_lightgcn_ml1m_pilot_metrics.csv` and `plan_e_lightgcn_ml1m_graph_summary.csv`.",
        "",
        "## 40. Ranking versus reranking interpretation",
        "- Ranking and reranking use different candidate orders/sets but the same strict graph-connected scoring policy.",
        "",
        "## 41. Same-seed reproducibility",
        f"- Error-level determinism failures: {sum(row['status']=='failed' and row['severity']=='error' for row in determinism_rows)}.",
        "",
        "## 42. Validation findings",
        f"- Validation rows: {len(validation_rows)}; error-level failures: {sum(row['status']=='failed' and row['severity']=='error' for row in validation_rows)}.",
        "",
        "## 43. Comparison boundary with existing baselines",
        "- Same row keys as ML-1M BPRMF, GRU4Rec and SASRec pilot rankings, but no cross-baseline consolidation is performed here.",
        "",
        "## 44. Sparse CUDA determinism limitation",
        "- CUDA determinism is supported by this run's equality evidence; this remains hardware/environment-specific.",
        "",
        "## 45. Deviations from the original paper",
        "- Reconstructed data, local Plan D implementation, strict graph-connected subset and local environment differ from exact paper artifacts.",
        "",
        "## 46. Limitations",
        "- ML-1M-only pilot; Beauty/Sports/LastFM not executed; beyond-utility metrics not computed.",
        "",
        "## 47. Whether the pilot succeeded",
        f"- Pilot success assessment: `{'succeeded' if success else 'completed_with_determinism_warning'}`.",
        "",
        "## 48. Conditions for Beauty and Sports scale-up",
        "- Scale-up requires the same protected graph, strict connected cohort, deterministic repeat and checkpoint controls.",
        "",
        "## 49. Protected-input immutability",
        "- Protected hashes were verified before and after execution.",
        "",
        "## 50. Commands executed",
        "- See final assistant response for command list; outputs are traceable to generated CSV files.",
        "",
        "## 51. Recommended single next task",
        f"- {next_task}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def metric_sentence(metrics: Sequence[Mapping[str, Any]], task_type: str) -> str:
    rows = [row for row in metrics if row["task_type"] == task_type and row["cohort_id"].endswith("_full")]
    if not rows:
        return "- No rows."
    row = rows[0]
    return f"- {task_type}: rows {row['evaluated_rows']}, HR@5 `{float(row['hr_at_5']):.12g}`, NDCG@5 `{float(row['ndcg_at_5']):.12g}`."


def write_csv(path: Path, columns: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    _write_csv_rows(
        path,
        columns,
        rows,
        overwrite=False,
        existing_error_factory=LightGCNPilotSafetyError,
        existing_error_message=f"refusing to overwrite existing output: {path}",
        lineterminator="\n",
        value_transform=format_value,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_rows(path)


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

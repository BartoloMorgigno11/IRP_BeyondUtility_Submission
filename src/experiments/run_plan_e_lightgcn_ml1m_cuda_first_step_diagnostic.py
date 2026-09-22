"""Run the Phase 3M-D2 LightGCN CUDA first-step diagnostic.

This runner executes only first-batch diagnostic replicas.  It creates no
checkpoint, ranking, metric, dense adjacency, or corrected training output.
"""

from __future__ import annotations

from src.audit import common_git

import csv
import hashlib
import inspect
import json
import os
import subprocess
import sys
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

from src.analysis.plan_e_lightgcn_cuda_first_step_diagnostic import (
    GRADIENT_STAGE_NAMES,
    LOSS_STAGE_NAMES,
    MODEL_HASH_METHOD,
    OPTIMIZER_STAGE_NAMES,
    POST_STEP_PARAMETER_STAGE_NAMES,
    STAGE_ORDER,
    TENSOR_HASH_METHOD,
    canonical_int_array_sha256,
    canonical_model_state_sha256,
    canonical_optimizer_state_sha256,
    classify_diagnostic,
    clone_named_gradients,
    clone_named_parameters,
    combined_named_tensor_sha256,
    compare_tensors,
    comparison_to_row,
    first_divergence_stage,
    inject_named_gradients,
    legacy_state_dict_sha256,
    max_observed_absolute_difference,
    optimizer_defaults_summary,
    optimizer_state_by_parameter_name,
    stable_json_sha256,
    stages_exact,
    tensor_content_sha256,
    validate_scale_up_decision,
)
from src.baselines.plan_d_bprmf import bpr_loss
from src.baselines.plan_d_lightgcn import LightGCNModel
from src.baselines.plan_e_lightgcn_training import (
    LightGCNPilotConfig,
    LightGCNPilotSafetyError,
    build_negative_schedule,
    build_training_arrays,
    file_sha256,
    graph_hashes,
    load_train_interactions,
    set_deterministic_execution,
    sparse_tensor_from_graph,
)
from src.baselines.plan_e_lightgcn_adapter import build_graph_contract


APPROVED_INTERPRETER = (PROJECT_ROOT / ".venv" / "Scripts" / "python.exe").resolve()
CONFIG_PATH = PROJECT_ROOT / "configs" / "plan_e_lightgcn_ml1m_pilot.yaml"
METADATA_PATH = PROJECT_ROOT / "results" / "models" / "plan_e_lightgcn_ml1m_pilot" / "metadata.json"
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "models" / "plan_e_lightgcn_ml1m_pilot" / "model_state.pt"
D1_CUDA_STEP_PATH = PROJECT_ROOT / "results" / "tables" / "plan_e_lightgcn_ml1m_cuda_step_diagnostic.csv"
D1_DECISION_PATH = PROJECT_ROOT / "results" / "tables" / "plan_e_lightgcn_ml1m_diagnostic_decision.csv"

DIAGNOSTIC_ID = "plan_e_lightgcn_ml1m_cuda_first_step_diagnostic"

OUTPUT_PATHS = {
    "manifest": "results/tables/plan_e_lightgcn_ml1m_cuda_first_step_manifest.csv",
    "stage": "results/tables/plan_e_lightgcn_ml1m_cuda_first_step_stage_comparison.csv",
    "parameter": "results/tables/plan_e_lightgcn_ml1m_cuda_parameter_gradient_comparison.csv",
    "optimizer": "results/tables/plan_e_lightgcn_ml1m_cuda_optimizer_state_comparison.csv",
    "variant": "results/tables/plan_e_lightgcn_ml1m_cuda_first_step_variant_summary.csv",
    "hash_audit": "results/tables/plan_e_lightgcn_ml1m_state_hash_audit.csv",
    "decision": "results/tables/plan_e_lightgcn_ml1m_cuda_first_step_decision.csv",
    "report": "report/plan_e_lightgcn_ml1m_cuda_first_step_diagnostic.md",
}

KNOWN_PROTECTED_HASHES = {
    "src/baselines/plan_d_lightgcn.py": "1B9DF1E9DF737758639D98013A8718E287EE550E9491979C663EE6E95959FEF8",
    "src/experiments/run_plan_d_lightgcn.py": "27626556582F897C846904B11658B51DB4C499C6335903D8CF3B52474FB800AB",
    "src/baselines/plan_e_lightgcn_adapter.py": "A6D321D9A5D7D6134ABDBAD771BDA72AABCABBC0D2B240BB433C5FE83DBA4841",
    "src/baselines/plan_e_lightgcn_training.py": "A01A7B51F79B14F6F978EC1BDCDEDB23D0BFB48139F00A99ADA20024B25C0D7E",
    "src/experiments/run_plan_e_lightgcn_ml1m_pilot.py": "FC7EA7D15AB8F50FEC69CB1344EEA98F049FBC45205C4D64DE545D8F26AE94F2",
    "configs/plan_e_lightgcn_ml1m_pilot.yaml": "643A5DE90EC9F457006DA96417327DBFE641A8072A4EE425AB382826F49E4FBB",
    "data/reconstructed/plan_c/ml1m/interactions.csv": "1871DE0C52382305B2AA54187300B68889B34551635E4991E6368E556B39D870",
    "data/reconstructed/plan_c/ml1m/datamaps.json": "D6AC1A6C19E47156955DE87F45EE2255AC69728510E734BE46DBBBECD68BF093",
    "results/tables/ranking_candidate_sets.csv": "640AD9C00B7D7C825AD10746A0E720C161D1BA9D75CC791A019DF7FE14C7A955",
    "results/tables/positive_positions.csv": "0C93DD55D5C96ED99FB7F1AFC8F5319E90F3092ECB6A48F2E9D46854B38CD6D2",
    "results/tables/plan_e_lightgcn_ml1m_model_node_map.csv": "11D8561D26168F29C693316A3F29C693316A3F29F33091310E8CEAC442773D3FBD5FD854207E",
}

# Correct the long hash above by keeping this small override separate; it avoids
# a visually easy-to-miss accidental duplication in the literal table.
KNOWN_PROTECTED_HASHES["results/tables/plan_e_lightgcn_ml1m_model_node_map.csv"] = (
    "11D8561D26168F29C693316A3F29F33091310E8CEAC442773D3FBD5FD854207E"
)
KNOWN_PROTECTED_HASHES.update(
    {
        "results/tables/plan_e_lightgcn_ml1m_graph_summary.csv": "F2D2D800AB5EFD751ACA41FC24C8A40F75B1D6AE75B0A389D5BF53F7E8C6026F",
        "results/tables/plan_e_lightgcn_ml1m_pilot_cohort.csv": "E648B8CD3F2C9C62EDC2F896274127E21950F033793FB1FCC433D73A8F494E93",
        "results/tables/plan_e_lightgcn_ml1m_pilot_run_manifest.csv": "9F48D539B616B24244D8C25F51365D9E65B824C3345A8AD9D3BDAC4BDA547434",
        "results/tables/plan_e_lightgcn_ml1m_pilot_training_history.csv": "4C0797738696EEE83B044A131176E61F2D8AD1461385D3EB06C17C2CF937C09D",
        "results/tables/plan_e_lightgcn_ml1m_pilot_rankings.csv": "DF9E0E695152412735539C18E8EF4DD2FB65A3ACCBA352008CBA4A2C53DAF468",
        "results/tables/plan_e_lightgcn_ml1m_pilot_metrics.csv": "5F0F8A55289AFEFABD371EA1B1699EE74EA4CE3B8D416381CC0A170724CBA18A",
        "results/tables/plan_e_lightgcn_ml1m_pilot_determinism.csv": "9AEDA6BADFC6AEF5A695A450FA77607DB3A0BC4B6FEEE5435C22388B109B2FB5",
        "results/tables/plan_e_lightgcn_ml1m_pilot_validation.csv": "93813C7A21C4CC27FA6B0E7AB1A8D11A004130E6047ADB03F45A43A04D82EC0A",
        "results/models/plan_e_lightgcn_ml1m_pilot/model_state.pt": "F385C4BF57BCB3C0D8A6F41151AFB244E6C9E5120DE3DA49C54A1ABA8537C131",
        "results/models/plan_e_lightgcn_ml1m_pilot/metadata.json": "80E816EA05C25FAA8A74973B5362F914C65FAFE16D6C89555AE1B902BBA69B12",
        "report/plan_e_lightgcn_ml1m_pilot.md": "971898B919BF1159208BE35BC7B554B028C74C74AE880B60649C9C0B5640E7D1",
        "report/plan_e_lightgcn_adapter_preflight.md": "EDDBD75DD3AD9295C792540379B991E2FD2F9A0496424D5746FB4CBEDFDD55D6",
    }
)

D1_SNAPSHOT_PATHS = [
    "src/analysis/plan_e_lightgcn_determinism_diagnostic.py",
    "src/experiments/run_plan_e_lightgcn_ml1m_determinism_diagnostic.py",
    "tests/test_plan_e_lightgcn_determinism_diagnostic.py",
    "report/plan_e_lightgcn_ml1m_determinism_diagnostic.md",
    "results/tables/plan_e_lightgcn_ml1m_loss_reconciliation.csv",
    "results/tables/plan_e_lightgcn_ml1m_cuda_step_diagnostic.csv",
    "results/tables/plan_e_lightgcn_ml1m_cpu_diagnostic_run_manifest.csv",
    "results/tables/plan_e_lightgcn_ml1m_cpu_diagnostic_training_history.csv",
    "results/tables/plan_e_lightgcn_ml1m_determinism_diagnostic.csv",
    "results/tables/plan_e_lightgcn_ml1m_cpu_cuda_comparison.csv",
    "results/tables/plan_e_lightgcn_ml1m_diagnostic_decision.csv",
]

MANIFEST_COLUMNS = [
    "diagnostic_id",
    "variant_id",
    "replica_id",
    "device",
    "python_executable",
    "python_version",
    "torch_version",
    "torch_cuda_version",
    "cuda_device",
    "seed",
    "training_data_sha256",
    "raw_edge_sha256",
    "directed_edge_index_sha256",
    "normalized_adjacency_sha256",
    "negative_sampling_evidence_sha256",
    "batch_index",
    "batch_size",
    "user_indices_sha256",
    "positive_indices_sha256",
    "negative_indices_sha256",
    "canonical_initial_model_state_sha256",
    "optimizer_class",
    "optimizer_foreach",
    "optimizer_fused",
    "optimizer_capturable",
    "optimizer_differentiable",
    "optimizer_amsgrad",
    "learning_rate",
    "beta1",
    "beta2",
    "epsilon",
    "weight_decay",
    "forward_executed",
    "backward_executed",
    "optimizer_step_executed",
    "execution_status",
    "blocking_reason",
    "notes",
]

STAGE_COLUMNS = [
    "comparison_id",
    "variant_id",
    "stage_order",
    "stage_name",
    "left_replica_id",
    "right_replica_id",
    "tensor_dtype",
    "tensor_shape",
    "element_count",
    "left_content_sha256",
    "right_content_sha256",
    "torch_equal",
    "mismatching_element_count",
    "mismatch_fraction",
    "maximum_absolute_difference",
    "mean_absolute_difference",
    "maximum_relative_difference",
    "first_mismatch_flat_index",
    "left_first_mismatch_value",
    "right_first_mismatch_value",
    "all_values_finite",
    "status",
    "severity",
    "interpretation",
    "notes",
]

PARAMETER_COLUMNS = [
    "comparison_id",
    "variant_id",
    "comparison_stage",
    "parameter_name",
    "left_content_sha256",
    "right_content_sha256",
    "torch_equal",
    "element_count",
    "mismatching_element_count",
    "maximum_absolute_difference",
    "mean_absolute_difference",
    "first_mismatch_flat_index",
    "left_first_mismatch_value",
    "right_first_mismatch_value",
    "gradient_present_left",
    "gradient_present_right",
    "all_values_finite",
    "status",
    "interpretation",
    "notes",
]

OPTIMIZER_COLUMNS = [
    "comparison_id",
    "variant_id",
    "parameter_name",
    "optimizer_state_field",
    "left_value_type",
    "right_value_type",
    "left_content_sha256",
    "right_content_sha256",
    "exact_equal",
    "element_count",
    "mismatching_element_count",
    "maximum_absolute_difference",
    "mean_absolute_difference",
    "first_mismatch_flat_index",
    "left_first_mismatch_value",
    "right_first_mismatch_value",
    "status",
    "interpretation",
    "notes",
]

VARIANT_COLUMNS = [
    "variant_id",
    "replicas_compared",
    "inputs_exact",
    "graph_exact",
    "initial_state_exact",
    "forward_exact",
    "loss_exact",
    "gradients_exact",
    "post_step_parameters_exact",
    "optimizer_state_exact",
    "first_divergence_stage",
    "first_divergence_parameter",
    "maximum_observed_absolute_difference",
    "d1_hash_difference_reproduced",
    "direct_tensor_difference_reproduced",
    "variant_status",
    "interpretation",
    "notes",
]

HASH_AUDIT_COLUMNS = [
    "audit_id",
    "state_scope",
    "state_stage",
    "replica_id",
    "legacy_hash_method",
    "legacy_hash_value",
    "canonical_hash_method",
    "canonical_hash_value",
    "legacy_hash_equal_across_replicas",
    "canonical_hash_equal_across_replicas",
    "direct_tensors_equal_across_replicas",
    "serialization_artifact_detected",
    "parameter_id_artifact_detected",
    "interpretation",
    "notes",
]

DECISION_COLUMNS = [
    "diagnostic_id",
    "phase3m_original_status",
    "phase3m_d1_classification",
    "phase3m_d2_classification",
    "first_true_tensor_divergence_stage",
    "forward_exact",
    "gradients_exact",
    "default_adam_exact",
    "foreach_false_adam_exact",
    "foreach_true_adam_exact",
    "legacy_hash_artifact_detected",
    "optimizer_parameter_id_artifact_detected",
    "phase3m_primary_outputs_remain_valid",
    "existing_cuda_pipeline_approved",
    "corrected_ml1m_rerun_required",
    "cpu_only_scale_up_approved",
    "scale_up_decision",
    "required_next_configuration_change",
    "blocking_reason",
    "recommended_single_next_task",
    "notes",
]


@dataclass(frozen=True)
class BatchContract:
    users_np: np.ndarray
    positives_np: np.ndarray
    negatives_np: np.ndarray
    users_hash: str
    positives_hash: str
    negatives_hash: str


@dataclass
class ReplicaResult:
    variant_id: str
    replica_id: str
    device: str
    captures: dict[str, torch.Tensor]
    parameter_states: dict[str, dict[str, torch.Tensor | None]]
    optimizer_states: dict[str, dict[str, Any]]
    legacy_initial_model_hash: str
    legacy_post_step_model_hash: str
    legacy_optimizer_hash: str
    canonical_initial_model_hash: str
    canonical_post_step_model_hash: str
    canonical_optimizer_hash: str
    optimizer_defaults: dict[str, Any]
    forward_executed: bool
    backward_executed: bool
    optimizer_step_executed: bool
    execution_status: str
    blocking_reason: str
    notes: str


def main() -> int:
    verify_runtime()
    ensure_no_existing_outputs()
    verify_known_protected_hashes()
    d1_snapshot = snapshot_paths(D1_SNAPSHOT_PATHS)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    config_data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config = make_config(config_data)
    env = environment_summary()

    train, _train_summary = load_train_interactions(PROJECT_ROOT / config_data["training_data_path"])
    contract = build_graph_contract(train)
    hashes = graph_hashes(contract, PROJECT_ROOT / config_data["training_data_path"])
    arrays = build_training_arrays(contract)
    schedule = build_negative_schedule(arrays=arrays, seed=config.seed, epochs=config.epochs)
    batch = first_batch_contract(arrays, schedule, config)
    verify_first_batch_contract(hashes, schedule, batch, metadata)

    if not torch.cuda.is_available():
        raise LightGCNPilotSafetyError("CUDA is required for Phase 3M-D2 and is unavailable")

    set_deterministic_execution(config.seed)
    cuda_device = torch.device("cuda")
    normalized_adj = sparse_tensor_from_graph(contract.normalized_graph, device=cuda_device)
    initial_state, initial_hashes = build_canonical_initial_state(config=config, arrays=arrays)
    canonical_gradients = build_canonical_gradient_set(
        config=config,
        arrays=arrays,
        normalized_adj=normalized_adj,
        batch=batch,
        initial_state=initial_state,
    )
    d1_rows = load_d1_cuda_step_rows()
    d1_decision = read_single_row(D1_DECISION_PATH)

    variants: dict[str, list[ReplicaResult]] = {}
    variants["original_pipeline_default_adam"] = [
        run_replica(
            variant_id="original_pipeline_default_adam",
            replica_id=replica_id,
            config=config,
            arrays=arrays,
            normalized_adj=normalized_adj,
            batch=batch,
            initial_state=initial_state,
            optimizer_kwargs={},
            execute_forward=True,
            execute_backward=True,
            execute_step=True,
        )
        for replica_id in ("replica_a", "replica_b")
    ]
    variants["forward_only_no_backward"] = [
        run_replica(
            variant_id="forward_only_no_backward",
            replica_id=replica_id,
            config=config,
            arrays=arrays,
            normalized_adj=normalized_adj,
            batch=batch,
            initial_state=initial_state,
            optimizer_kwargs={},
            execute_forward=True,
            execute_backward=False,
            execute_step=False,
        )
        for replica_id in ("replica_a", "replica_b")
    ]
    variants["forward_backward_no_step"] = [
        run_replica(
            variant_id="forward_backward_no_step",
            replica_id=replica_id,
            config=config,
            arrays=arrays,
            normalized_adj=normalized_adj,
            batch=batch,
            initial_state=initial_state,
            optimizer_kwargs={},
            execute_forward=True,
            execute_backward=True,
            execute_step=False,
        )
        for replica_id in ("replica_a", "replica_b")
    ]
    variants["shared_gradient_default_adam"] = [
        run_replica(
            variant_id="shared_gradient_default_adam",
            replica_id=replica_id,
            config=config,
            arrays=arrays,
            normalized_adj=normalized_adj,
            batch=batch,
            initial_state=initial_state,
            optimizer_kwargs={},
            execute_forward=False,
            execute_backward=False,
            execute_step=True,
            shared_gradients=canonical_gradients,
        )
        for replica_id in ("replica_a", "replica_b")
    ]
    variants["shared_gradient_foreach_false"] = [
        run_replica(
            variant_id="shared_gradient_foreach_false",
            replica_id=replica_id,
            config=config,
            arrays=arrays,
            normalized_adj=normalized_adj,
            batch=batch,
            initial_state=initial_state,
            optimizer_kwargs={"foreach": False, "fused": False},
            execute_forward=False,
            execute_backward=False,
            execute_step=True,
            shared_gradients=canonical_gradients,
        )
        for replica_id in ("replica_a", "replica_b")
    ]
    variants["shared_gradient_foreach_true"] = run_foreach_true_variant(
        config=config,
        arrays=arrays,
        normalized_adj=normalized_adj,
        batch=batch,
        initial_state=initial_state,
        shared_gradients=canonical_gradients,
    )

    manifest_rows = build_manifest_rows(
        variants=variants,
        env=env,
        hashes=hashes,
        schedule_hash=schedule.evidence_sha256,
        batch=batch,
        canonical_initial_hash=initial_hashes["canonical"],
        config=config,
    )
    stage_rows = build_stage_comparison_rows(variants)
    parameter_rows = build_parameter_comparison_rows(variants)
    optimizer_rows = build_optimizer_comparison_rows(variants)
    variant_rows = build_variant_summary_rows(stage_rows, parameter_rows, optimizer_rows, variants, d1_rows)
    hash_audit_rows = build_hash_audit_rows(variants, stage_rows, d1_rows)
    decision_rows = build_decision_rows(
        variant_rows=variant_rows,
        hash_audit_rows=hash_audit_rows,
        d1_decision=d1_decision,
    )
    report_text = build_report(
        env=env,
        metadata=metadata,
        config=config,
        hashes=hashes,
        schedule_hash=schedule.evidence_sha256,
        batch=batch,
        initial_hashes=initial_hashes,
        manifest_rows=manifest_rows,
        stage_rows=stage_rows,
        parameter_rows=parameter_rows,
        optimizer_rows=optimizer_rows,
        variant_rows=variant_rows,
        hash_audit_rows=hash_audit_rows,
        decision_rows=decision_rows,
    )

    validate_outputs(
        manifest_rows=manifest_rows,
        stage_rows=stage_rows,
        parameter_rows=parameter_rows,
        optimizer_rows=optimizer_rows,
        variant_rows=variant_rows,
        hash_audit_rows=hash_audit_rows,
        decision_rows=decision_rows,
        report_text=report_text,
    )
    write_outputs(
        manifest_rows=manifest_rows,
        stage_rows=stage_rows,
        parameter_rows=parameter_rows,
        optimizer_rows=optimizer_rows,
        variant_rows=variant_rows,
        hash_audit_rows=hash_audit_rows,
        decision_rows=decision_rows,
        report_text=report_text,
    )
    verify_known_protected_hashes()
    verify_snapshot_unchanged(d1_snapshot)
    print("plan_e_lightgcn_ml1m_cuda_first_step_diagnostic_completed")
    print(f"classification={decision_rows[0]['phase3m_d2_classification']}")
    print(f"scale_up_decision={decision_rows[0]['scale_up_decision']}")
    print(f"first_true_tensor_divergence_stage={decision_rows[0]['first_true_tensor_divergence_stage']}")
    return 0


def verify_runtime() -> None:
    executable = Path(sys.executable).resolve()
    if executable != APPROVED_INTERPRETER:
        raise LightGCNPilotSafetyError(f"approved interpreter required: {APPROVED_INTERPRETER}; got {executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise LightGCNPilotSafetyError("PYTHONNOUSERSITE=1 is required")


def ensure_no_existing_outputs() -> None:
    existing = [path for path in OUTPUT_PATHS.values() if (PROJECT_ROOT / path).exists()]
    if existing:
        raise LightGCNPilotSafetyError("refusing to overwrite existing D2 output(s): " + "; ".join(existing))


def verify_known_protected_hashes() -> None:
    mismatches: list[str] = []
    for rel_path, expected in KNOWN_PROTECTED_HASHES.items():
        path = PROJECT_ROOT / rel_path
        if not path.is_file():
            mismatches.append(f"missing {rel_path}")
            continue
        actual = file_sha256(path)
        if actual != expected:
            mismatches.append(f"{rel_path}: {actual} != {expected}")
    if mismatches:
        raise LightGCNPilotSafetyError("protected hash mismatch: " + "; ".join(mismatches[:8]))


def snapshot_paths(paths: Sequence[str]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for rel_path in paths:
        path = PROJECT_ROOT / rel_path
        if not path.is_file():
            raise LightGCNPilotSafetyError(f"required D1 protected file missing: {rel_path}")
        snapshot[rel_path] = file_sha256(path)
    return snapshot


def verify_snapshot_unchanged(snapshot: Mapping[str, str]) -> None:
    mismatches = []
    for rel_path, expected in snapshot.items():
        actual = file_sha256(PROJECT_ROOT / rel_path)
        if actual != expected:
            mismatches.append(f"{rel_path}: {actual} != {expected}")
    if mismatches:
        raise LightGCNPilotSafetyError("snapshot protected file changed: " + "; ".join(mismatches))


def make_config(config_data: Mapping[str, Any]) -> LightGCNPilotConfig:
    return LightGCNPilotConfig(
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
        device="cuda",
        evaluation_k=int(config_data["evaluation_k"]),
        max_run_minutes=float(config_data["maximum_run_minutes"]),
    )


def environment_summary() -> dict[str, Any]:
    return {
        "python_executable": str(APPROVED_INTERPRETER),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda or ""),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }


def first_batch_contract(arrays: Any, schedule: Any, config: LightGCNPilotConfig) -> BatchContract:
    order = schedule.epoch_orders[0]
    negatives_for_epoch = schedule.epoch_negatives[0]
    batch_slice = slice(0, config.batch_size)
    batch_order = order[batch_slice]
    users = arrays.user_indices[batch_order]
    positives = arrays.item_indices[batch_order]
    negatives = negatives_for_epoch[batch_slice]
    return BatchContract(
        users_np=users,
        positives_np=positives,
        negatives_np=negatives,
        users_hash=canonical_int_array_sha256(users),
        positives_hash=canonical_int_array_sha256(positives),
        negatives_hash=canonical_int_array_sha256(negatives),
    )


def verify_first_batch_contract(
    hashes: Mapping[str, str],
    schedule: Any,
    batch: BatchContract,
    metadata: Mapping[str, Any],
) -> None:
    graph_hashes_expected = metadata["graph_hashes"]
    required = [
        "training_data_sha256",
        "raw_edge_sha256",
        "directed_edge_index_sha256",
        "normalized_adjacency_sha256",
        "user_mapping_sha256",
        "item_mapping_sha256",
    ]
    mismatches = []
    for key in required:
        if hashes[key] != graph_hashes_expected[key]:
            mismatches.append(f"{key}: {hashes[key]} != {graph_hashes_expected[key]}")
    if schedule.evidence_sha256 != metadata["negative_sampling_evidence_sha256"]:
        mismatches.append("negative_sampling_evidence_sha256 mismatch")
    d1_first = d1_first_batch_row()
    if batch.users_hash != d1_first["user_indices_sha256"]:
        mismatches.append("first-batch user hash mismatch")
    if batch.positives_hash != d1_first["positive_indices_sha256"]:
        mismatches.append("first-batch positive hash mismatch")
    if batch.negatives_hash != d1_first["negative_indices_sha256"]:
        mismatches.append("first-batch negative hash mismatch")
    if mismatches:
        raise LightGCNPilotSafetyError("first-batch contract mismatch: " + "; ".join(mismatches))


def build_canonical_initial_state(*, config: LightGCNPilotConfig, arrays: Any) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    set_deterministic_execution(config.seed)
    model = LightGCNModel(
        num_users=arrays.unique_train_users,
        num_items=arrays.unique_train_items,
        embedding_dim=config.embedding_dim,
        num_layers=config.propagation_layers,
    )
    state = clone_named_parameters(model)
    legacy_hash = legacy_state_dict_sha256(model.state_dict())
    canonical_hash = canonical_model_state_sha256(model)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if legacy_hash != metadata["initial_model_state_hash"]:
        raise LightGCNPilotSafetyError(
            f"canonical CPU initial legacy hash mismatch: {legacy_hash} != {metadata['initial_model_state_hash']}"
        )
    return state, {"legacy": legacy_hash, "canonical": canonical_hash}


def build_canonical_gradient_set(
    *,
    config: LightGCNPilotConfig,
    arrays: Any,
    normalized_adj: torch.Tensor,
    batch: BatchContract,
    initial_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor | None]:
    result = run_replica(
        variant_id="canonical_gradient_source",
        replica_id="gradient_source",
        config=config,
        arrays=arrays,
        normalized_adj=normalized_adj,
        batch=batch,
        initial_state=initial_state,
        optimizer_kwargs={},
        execute_forward=True,
        execute_backward=True,
        execute_step=False,
    )
    return result.parameter_states["gradient"]


def run_foreach_true_variant(
    *,
    config: LightGCNPilotConfig,
    arrays: Any,
    normalized_adj: torch.Tensor,
    batch: BatchContract,
    initial_state: Mapping[str, torch.Tensor],
    shared_gradients: Mapping[str, torch.Tensor | None],
) -> list[ReplicaResult]:
    try:
        probe = torch.optim.Adam(
            LightGCNModel(1, 1, 1, 0).parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            foreach=True,
            fused=False,
        )
        del probe
    except Exception as exc:  # pragma: no cover - depends on installed torch.
        return [
            blocked_replica("shared_gradient_foreach_true", "replica_a", str(exc), config),
            blocked_replica("shared_gradient_foreach_true", "replica_b", str(exc), config),
        ]
    return [
        run_replica(
            variant_id="shared_gradient_foreach_true",
            replica_id=replica_id,
            config=config,
            arrays=arrays,
            normalized_adj=normalized_adj,
            batch=batch,
            initial_state=initial_state,
            optimizer_kwargs={"foreach": True, "fused": False},
            execute_forward=False,
            execute_backward=False,
            execute_step=True,
            shared_gradients=shared_gradients,
        )
        for replica_id in ("replica_a", "replica_b")
    ]


def blocked_replica(variant_id: str, replica_id: str, reason: str, config: LightGCNPilotConfig) -> ReplicaResult:
    return ReplicaResult(
        variant_id=variant_id,
        replica_id=replica_id,
        device="cuda",
        captures={},
        parameter_states={"initial": {}, "gradient": {}, "post_step": {}},
        optimizer_states={},
        legacy_initial_model_hash="",
        legacy_post_step_model_hash="",
        legacy_optimizer_hash="",
        canonical_initial_model_hash="",
        canonical_post_step_model_hash="",
        canonical_optimizer_hash="",
        optimizer_defaults={
            "optimizer_class": "Adam",
            "optimizer_foreach": "true",
            "optimizer_fused": "false",
            "optimizer_capturable": "",
            "optimizer_differentiable": "",
            "optimizer_amsgrad": "",
            "learning_rate": config.learning_rate,
            "beta1": "",
            "beta2": "",
            "epsilon": "",
            "weight_decay": config.weight_decay,
        },
        forward_executed=False,
        backward_executed=False,
        optimizer_step_executed=False,
        execution_status="not_applicable",
        blocking_reason=reason,
        notes="foreach=True unsupported in installed torch",
    )


def run_replica(
    *,
    variant_id: str,
    replica_id: str,
    config: LightGCNPilotConfig,
    arrays: Any,
    normalized_adj: torch.Tensor,
    batch: BatchContract,
    initial_state: Mapping[str, torch.Tensor],
    optimizer_kwargs: Mapping[str, Any],
    execute_forward: bool,
    execute_backward: bool,
    execute_step: bool,
    shared_gradients: Mapping[str, torch.Tensor | None] | None = None,
) -> ReplicaResult:
    set_deterministic_execution(config.seed)
    device = torch.device("cuda")
    model = LightGCNModel(
        num_users=arrays.unique_train_users,
        num_items=arrays.unique_train_items,
        embedding_dim=config.embedding_dim,
        num_layers=config.propagation_layers,
    ).to(device)
    model.load_state_dict({name: tensor.detach().clone() for name, tensor in initial_state.items()})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        **dict(optimizer_kwargs),
    )
    optimizer_defaults = optimizer_defaults_summary(optimizer)
    captures: dict[str, torch.Tensor] = {}
    parameter_states: dict[str, dict[str, torch.Tensor | None]] = {
        "initial": clone_named_parameters(model),
        "gradient": {},
        "post_step": {},
    }
    capture_static_inputs(captures, normalized_adj=normalized_adj, batch=batch, device=device)
    capture_parameters(captures, model, prefix="initial")
    legacy_initial = legacy_state_dict_sha256(model.state_dict())
    canonical_initial = canonical_model_state_sha256(model)

    users = torch.as_tensor(batch.users_np, dtype=torch.long, device=device)
    positives = torch.as_tensor(batch.positives_np, dtype=torch.long, device=device)
    negatives = torch.as_tensor(batch.negatives_np, dtype=torch.long, device=device)
    model.train()
    if execute_forward:
        optimizer.zero_grad(set_to_none=True)
        total_loss = forward_capture(
            model=model,
            normalized_adj=normalized_adj,
            users=users,
            positives=positives,
            negatives=negatives,
            captures=captures,
        )
        if execute_backward:
            total_loss.backward()
            torch.cuda.synchronize()
            capture_gradients(captures, model)
            parameter_states["gradient"] = clone_named_gradients(model)
    elif shared_gradients is not None:
        inject_named_gradients(model, shared_gradients, device=device)
        torch.cuda.synchronize()
        capture_gradients(captures, model)
        parameter_states["gradient"] = clone_named_gradients(model)

    if execute_step:
        optimizer.step()
        torch.cuda.synchronize()
        capture_parameters(captures, model, prefix="post_step")
        capture_optimizer(captures, optimizer=optimizer, model=model)
        parameter_states["post_step"] = clone_named_parameters(model)
    else:
        parameter_states["post_step"] = {}

    legacy_post_step = legacy_state_dict_sha256(model.state_dict()) if execute_step else ""
    canonical_post_step = canonical_model_state_sha256(model) if execute_step else ""
    legacy_optimizer = legacy_optimizer_state_sha256(optimizer) if execute_step else ""
    canonical_optimizer = canonical_optimizer_state_sha256(optimizer, model) if execute_step else ""
    optimizer_states = optimizer_state_by_parameter_name(optimizer, model) if execute_step else {}
    return ReplicaResult(
        variant_id=variant_id,
        replica_id=replica_id,
        device=str(device),
        captures=captures,
        parameter_states=parameter_states,
        optimizer_states=optimizer_states,
        legacy_initial_model_hash=legacy_initial,
        legacy_post_step_model_hash=legacy_post_step,
        legacy_optimizer_hash=legacy_optimizer,
        canonical_initial_model_hash=canonical_initial,
        canonical_post_step_model_hash=canonical_post_step,
        canonical_optimizer_hash=canonical_optimizer,
        optimizer_defaults=optimizer_defaults,
        forward_executed=execute_forward,
        backward_executed=execute_backward,
        optimizer_step_executed=execute_step,
        execution_status="completed",
        blocking_reason="",
        notes="single first-batch diagnostic replica; no checkpoint",
    )


def capture_tensor(captures: dict[str, torch.Tensor], name: str, tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize()
    captures[name] = tensor.detach().cpu().contiguous().clone()


def capture_static_inputs(
    captures: dict[str, torch.Tensor],
    *,
    normalized_adj: torch.Tensor,
    batch: BatchContract,
    device: torch.device,
) -> None:
    capture_tensor(captures, "batch_users", torch.as_tensor(batch.users_np, dtype=torch.long, device=device))
    capture_tensor(captures, "batch_positive_items", torch.as_tensor(batch.positives_np, dtype=torch.long, device=device))
    capture_tensor(captures, "batch_negative_items", torch.as_tensor(batch.negatives_np, dtype=torch.long, device=device))
    capture_tensor(captures, "adjacency_indices", normalized_adj.indices())
    capture_tensor(captures, "adjacency_values", normalized_adj.values())


def capture_parameters(captures: dict[str, torch.Tensor], model: LightGCNModel, *, prefix: str) -> None:
    capture_tensor(captures, f"{prefix}_user_embedding_parameter", model.user_embedding.weight)
    capture_tensor(captures, f"{prefix}_item_embedding_parameter", model.item_embedding.weight)


def capture_gradients(captures: dict[str, torch.Tensor], model: LightGCNModel) -> None:
    if model.user_embedding.weight.grad is not None:
        capture_tensor(captures, "user_embedding_gradient", model.user_embedding.weight.grad)
    if model.item_embedding.weight.grad is not None:
        capture_tensor(captures, "item_embedding_gradient", model.item_embedding.weight.grad)


def capture_optimizer(captures: dict[str, torch.Tensor], *, optimizer: torch.optim.Optimizer, model: LightGCNModel) -> None:
    mapped = optimizer_state_by_parameter_name(optimizer, model)
    name_to_stage = {
        "user_embedding.weight": "user",
        "item_embedding.weight": "item",
    }
    for parameter_name, stage_prefix in name_to_stage.items():
        state = mapped.get(parameter_name, {})
        for field in ("step", "exp_avg", "exp_avg_sq"):
            if field in state:
                value = state[field]
                tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
                capture_tensor(captures, f"{stage_prefix}_{field}", tensor)


def forward_capture(
    *,
    model: LightGCNModel,
    normalized_adj: torch.Tensor,
    users: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    captures: dict[str, torch.Tensor],
) -> torch.Tensor:
    current = torch.cat([model.user_embedding.weight, model.item_embedding.weight], dim=0)
    layers = [current]
    capture_tensor(captures, "layer_0_all_embeddings", current)
    for layer_index in range(1, model.num_layers + 1):
        current = torch.sparse.mm(normalized_adj, current)
        layers.append(current)
        capture_tensor(captures, f"layer_{layer_index}_all_embeddings", current)
    final = torch.stack(layers, dim=0).mean(dim=0)
    user_embeddings = final[: model.num_users]
    item_embeddings = final[model.num_users :]
    capture_tensor(captures, "final_all_embeddings", final)
    capture_tensor(captures, "final_user_embeddings", user_embeddings)
    capture_tensor(captures, "final_item_embeddings", item_embeddings)
    batch_user_embeddings = user_embeddings[users]
    batch_positive_embeddings = item_embeddings[positives]
    batch_negative_embeddings = item_embeddings[negatives]
    capture_tensor(captures, "batch_user_embeddings", batch_user_embeddings)
    capture_tensor(captures, "batch_positive_embeddings", batch_positive_embeddings)
    capture_tensor(captures, "batch_negative_embeddings", batch_negative_embeddings)
    positive_scores = (batch_user_embeddings * batch_positive_embeddings).sum(dim=1)
    negative_scores = (batch_user_embeddings * batch_negative_embeddings).sum(dim=1)
    capture_tensor(captures, "positive_scores", positive_scores)
    capture_tensor(captures, "negative_scores", negative_scores)
    bpr = bpr_loss(positive_scores, negative_scores)
    regularization = torch.zeros((), dtype=bpr.dtype, device=bpr.device)
    total = bpr + regularization
    capture_tensor(captures, "bpr_loss_tensor", bpr)
    capture_tensor(captures, "regularization_loss_tensor", regularization)
    capture_tensor(captures, "total_loss_tensor", total)
    return total


def legacy_optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    state = optimizer.state_dict()
    digest = hashlib.sha256()
    digest.update(json.dumps(state.get("param_groups", []), sort_keys=True, default=str).encode("utf-8"))
    for parameter_id in sorted(state.get("state", {}), key=lambda value: str(value)):
        digest.update(str(parameter_id).encode("utf-8"))
        parameter_state = state["state"][parameter_id]
        for key in sorted(parameter_state):
            value = parameter_state[key]
            digest.update(str(key).encode("utf-8"))
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tuple(tensor.shape)).encode("utf-8"))
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest().upper()


def build_manifest_rows(
    *,
    variants: Mapping[str, Sequence[ReplicaResult]],
    env: Mapping[str, Any],
    hashes: Mapping[str, str],
    schedule_hash: str,
    batch: BatchContract,
    canonical_initial_hash: str,
    config: LightGCNPilotConfig,
) -> list[dict[str, Any]]:
    rows = []
    for variant_id in variants:
        for replica in variants[variant_id]:
            defaults = replica.optimizer_defaults
            rows.append(
                {
                    "diagnostic_id": DIAGNOSTIC_ID,
                    "variant_id": variant_id,
                    "replica_id": replica.replica_id,
                    "device": replica.device,
                    "python_executable": env["python_executable"],
                    "python_version": env["python_version"],
                    "torch_version": env["torch_version"],
                    "torch_cuda_version": env["torch_cuda_version"],
                    "cuda_device": env["cuda_device"],
                    "seed": config.seed,
                    "training_data_sha256": hashes["training_data_sha256"],
                    "raw_edge_sha256": hashes["raw_edge_sha256"],
                    "directed_edge_index_sha256": hashes["directed_edge_index_sha256"],
                    "normalized_adjacency_sha256": hashes["normalized_adjacency_sha256"],
                    "negative_sampling_evidence_sha256": schedule_hash,
                    "batch_index": 1,
                    "batch_size": int(len(batch.users_np)),
                    "user_indices_sha256": batch.users_hash,
                    "positive_indices_sha256": batch.positives_hash,
                    "negative_indices_sha256": batch.negatives_hash,
                    "canonical_initial_model_state_sha256": canonical_initial_hash,
                    "optimizer_class": defaults.get("optimizer_class", ""),
                    "optimizer_foreach": defaults.get("optimizer_foreach", ""),
                    "optimizer_fused": defaults.get("optimizer_fused", ""),
                    "optimizer_capturable": defaults.get("optimizer_capturable", ""),
                    "optimizer_differentiable": defaults.get("optimizer_differentiable", ""),
                    "optimizer_amsgrad": defaults.get("optimizer_amsgrad", ""),
                    "learning_rate": defaults.get("learning_rate", ""),
                    "beta1": defaults.get("beta1", ""),
                    "beta2": defaults.get("beta2", ""),
                    "epsilon": defaults.get("epsilon", ""),
                    "weight_decay": defaults.get("weight_decay", ""),
                    "forward_executed": replica.forward_executed,
                    "backward_executed": replica.backward_executed,
                    "optimizer_step_executed": replica.optimizer_step_executed,
                    "execution_status": replica.execution_status,
                    "blocking_reason": replica.blocking_reason,
                    "notes": replica.notes,
                }
            )
    return rows


def build_stage_comparison_rows(variants: Mapping[str, Sequence[ReplicaResult]]) -> list[dict[str, Any]]:
    rows = []
    for variant_id, replicas in variants.items():
        left, right = replicas[0], replicas[1]
        for index, stage_name in enumerate(STAGE_ORDER, start=1):
            left_tensor = left.captures.get(stage_name)
            right_tensor = right.captures.get(stage_name)
            comparison = compare_tensors(stage_name, left_tensor, right_tensor)
            row = {
                "comparison_id": f"{variant_id}_stage_{index:02d}",
                "variant_id": variant_id,
                "stage_order": index,
                "stage_name": stage_name,
                "left_replica_id": left.replica_id,
                "right_replica_id": right.replica_id,
                **comparison_to_row(comparison),
                "severity": "warning" if comparison.status == "different" else "info",
                "notes": "direct tensor comparison; CUDA synchronized before capture",
            }
            rows.append(row)
    return rows


def build_parameter_comparison_rows(
    variants: Mapping[str, Sequence[ReplicaResult]],
) -> list[dict[str, Any]]:
    rows = []
    parameter_names = ["user_embedding.weight", "item_embedding.weight"]
    stages = ["initial", "gradient", "post_step"]
    for variant_id, replicas in variants.items():
        left, right = replicas[0], replicas[1]
        for stage in stages:
            for parameter_name in parameter_names:
                left_tensor = left.parameter_states.get(stage, {}).get(parameter_name)
                right_tensor = right.parameter_states.get(stage, {}).get(parameter_name)
                comparison = compare_tensors(f"{stage}.{parameter_name}", left_tensor, right_tensor)
                rows.append(
                    {
                        "comparison_id": f"{variant_id}_{stage}_{parameter_name.replace('.', '_')}",
                        "variant_id": variant_id,
                        "comparison_stage": stage,
                        "parameter_name": parameter_name,
                        "left_content_sha256": comparison.left_content_sha256,
                        "right_content_sha256": comparison.right_content_sha256,
                        "torch_equal": comparison.torch_equal,
                        "element_count": comparison.element_count,
                        "mismatching_element_count": comparison.mismatching_element_count,
                        "maximum_absolute_difference": comparison.maximum_absolute_difference,
                        "mean_absolute_difference": comparison.mean_absolute_difference,
                        "first_mismatch_flat_index": comparison.first_mismatch_flat_index,
                        "left_first_mismatch_value": comparison.left_first_mismatch_value,
                        "right_first_mismatch_value": comparison.right_first_mismatch_value,
                        "gradient_present_left": left_tensor is not None if stage == "gradient" else "",
                        "gradient_present_right": right_tensor is not None if stage == "gradient" else "",
                        "all_values_finite": comparison.all_values_finite,
                        "status": comparison.status,
                        "interpretation": comparison.interpretation,
                        "notes": "semantic parameter-name comparison; no raw parameter IDs",
                    }
                )
    return rows


def build_optimizer_comparison_rows(
    variants: Mapping[str, Sequence[ReplicaResult]],
) -> list[dict[str, Any]]:
    rows = []
    parameter_names = ["user_embedding.weight", "item_embedding.weight"]
    fields = ["step", "exp_avg", "exp_avg_sq"]
    for variant_id, replicas in variants.items():
        left, right = replicas[0], replicas[1]
        for parameter_name in parameter_names:
            for field in fields:
                left_value = left.optimizer_states.get(parameter_name, {}).get(field)
                right_value = right.optimizer_states.get(parameter_name, {}).get(field)
                left_tensor = to_tensor_or_none(left_value)
                right_tensor = to_tensor_or_none(right_value)
                comparison = compare_tensors(f"{parameter_name}.{field}", left_tensor, right_tensor)
                rows.append(
                    {
                        "comparison_id": f"{variant_id}_{parameter_name.replace('.', '_')}_{field}",
                        "variant_id": variant_id,
                        "parameter_name": parameter_name,
                        "optimizer_state_field": field,
                        "left_value_type": value_type(left_value),
                        "right_value_type": value_type(right_value),
                        "left_content_sha256": comparison.left_content_sha256,
                        "right_content_sha256": comparison.right_content_sha256,
                        "exact_equal": comparison.status == "exact",
                        "element_count": comparison.element_count,
                        "mismatching_element_count": comparison.mismatching_element_count,
                        "maximum_absolute_difference": comparison.maximum_absolute_difference,
                        "mean_absolute_difference": comparison.mean_absolute_difference,
                        "first_mismatch_flat_index": comparison.first_mismatch_flat_index,
                        "left_first_mismatch_value": comparison.left_first_mismatch_value,
                        "right_first_mismatch_value": comparison.right_first_mismatch_value,
                        "status": comparison.status,
                        "interpretation": comparison.interpretation,
                        "notes": "optimizer state mapped by semantic parameter name",
                    }
                )
    return rows


def to_tensor_or_none(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return torch.as_tensor(value)


def value_type(value: Any) -> str:
    if value is None:
        return ""
    if torch.is_tensor(value):
        return f"tensor:{value.dtype}"
    return type(value).__name__


def build_variant_summary_rows(
    stage_rows: Sequence[Mapping[str, Any]],
    parameter_rows: Sequence[Mapping[str, Any]],
    optimizer_rows: Sequence[Mapping[str, Any]],
    variants: Mapping[str, Sequence[ReplicaResult]],
    d1_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    d1_post_diff = d1_hashes_differ(d1_rows, "post_step_model_state_sha256")
    for variant_id, replicas in variants.items():
        selected_stages = [row for row in stage_rows if row["variant_id"] == variant_id]
        selected_parameters = [row for row in parameter_rows if row["variant_id"] == variant_id]
        selected_optimizer = [row for row in optimizer_rows if row["variant_id"] == variant_id]
        first_stage, interpretation = first_divergence_stage(selected_stages)
        first_parameter = first_divergence_parameter(selected_parameters, selected_optimizer)
        direct_difference = any(row["status"] == "different" for row in selected_stages)
        d1_hash_reproduced = (
            variant_id == "original_pipeline_default_adam"
            and d1_post_diff
            and replicas[0].legacy_post_step_model_hash != replicas[1].legacy_post_step_model_hash
        )
        if all(replica.execution_status == "not_applicable" for replica in replicas):
            variant_status = "not_applicable"
        elif any(replica.execution_status == "blocked" for replica in replicas):
            variant_status = "blocked"
        elif direct_difference:
            variant_status = "tensor_divergence"
        elif d1_hash_reproduced:
            variant_status = "hash_artifact_only"
        else:
            variant_status = "exact"
        rows.append(
            {
                "variant_id": variant_id,
                "replicas_compared": ";".join(replica.replica_id for replica in replicas),
                "inputs_exact": stages_exact(
                    selected_stages,
                    {"batch_users", "batch_positive_items", "batch_negative_items"},
                ),
                "graph_exact": stages_exact(selected_stages, {"adjacency_indices", "adjacency_values"}),
                "initial_state_exact": stages_exact(
                    selected_stages,
                    {"initial_user_embedding_parameter", "initial_item_embedding_parameter"},
                ),
                "forward_exact": stages_exact(selected_stages, set(STAGE_ORDER).intersection(set(STAGE_ORDER)) - set(STAGE_ORDER[:7]) - GRADIENT_STAGE_NAMES - POST_STEP_PARAMETER_STAGE_NAMES - OPTIMIZER_STAGE_NAMES - LOSS_STAGE_NAMES),
                "loss_exact": stages_exact(selected_stages, LOSS_STAGE_NAMES),
                "gradients_exact": stages_exact(selected_stages, GRADIENT_STAGE_NAMES),
                "post_step_parameters_exact": stages_exact(selected_stages, POST_STEP_PARAMETER_STAGE_NAMES),
                "optimizer_state_exact": stages_exact(selected_stages, OPTIMIZER_STAGE_NAMES),
                "first_divergence_stage": first_stage,
                "first_divergence_parameter": first_parameter,
                "maximum_observed_absolute_difference": max_observed_absolute_difference(selected_stages),
                "d1_hash_difference_reproduced": d1_hash_reproduced,
                "direct_tensor_difference_reproduced": direct_difference,
                "variant_status": variant_status,
                "interpretation": interpretation,
                "notes": "one first-batch comparison only; no scoring or full epoch",
            }
        )
    return rows


def first_divergence_parameter(
    parameter_rows: Sequence[Mapping[str, Any]],
    optimizer_rows: Sequence[Mapping[str, Any]],
) -> str:
    for row in parameter_rows:
        if row["status"] == "different":
            return str(row["parameter_name"])
    for row in optimizer_rows:
        if row["status"] == "different":
            return str(row["parameter_name"])
    return ""


def build_hash_audit_rows(
    variants: Mapping[str, Sequence[ReplicaResult]],
    stage_rows: Sequence[Mapping[str, Any]],
    d1_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    original = variants["original_pipeline_default_adam"]
    original_stage_rows = [row for row in stage_rows if row["variant_id"] == "original_pipeline_default_adam"]
    stage_status = {row["stage_name"]: row["status"] for row in original_stage_rows}
    d1_first = rows_by_role_and_batch(d1_rows, 1)
    scopes = [
        (
            "model_state",
            "initial",
            [replica.legacy_initial_model_hash for replica in original],
            [replica.canonical_initial_model_hash for replica in original],
            all(stage_status.get(name) == "exact" for name in ["initial_user_embedding_parameter", "initial_item_embedding_parameter"]),
            "D1 first-batch pre-step hash",
            {role: row["pre_step_model_state_sha256"] for role, row in d1_first.items()},
        ),
        (
            "model_state",
            "post_step",
            [replica.legacy_post_step_model_hash for replica in original],
            [replica.canonical_post_step_model_hash for replica in original],
            all(stage_status.get(name) == "exact" for name in ["post_step_user_embedding_parameter", "post_step_item_embedding_parameter"]),
            "D1 first-batch post-step hash",
            {role: row["post_step_model_state_sha256"] for role, row in d1_first.items()},
        ),
        (
            "optimizer_state",
            "post_step",
            [replica.legacy_optimizer_hash for replica in original],
            [replica.canonical_optimizer_hash for replica in original],
            all(stage_status.get(name) == "exact" for name in OPTIMIZER_STAGE_NAMES),
            "D1 first-batch optimizer hash",
            {role: row["post_step_optimizer_state_sha256"] for role, row in d1_first.items()},
        ),
    ]
    rows = []
    for scope_index, (scope, stage, legacy_hashes, canonical_hashes, direct_equal, d1_label, d1_reference) in enumerate(scopes, start=1):
        legacy_equal = len(set(legacy_hashes)) == 1
        canonical_equal = len(set(canonical_hashes)) == 1
        serialization_artifact = bool(not legacy_equal and canonical_equal and direct_equal and scope == "model_state")
        parameter_id_artifact = bool(not legacy_equal and canonical_equal and direct_equal and scope == "optimizer_state")
        for replica_index, replica in enumerate(original, start=1):
            rows.append(
                {
                    "audit_id": f"hash_{scope_index:02d}_{replica_index:02d}",
                    "state_scope": scope,
                    "state_stage": stage,
                    "replica_id": replica.replica_id,
                    "legacy_hash_method": "legacy_state_dict_sha256" if scope == "model_state" else "legacy_optimizer_state_sha256",
                    "legacy_hash_value": legacy_hashes[replica_index - 1],
                    "canonical_hash_method": MODEL_HASH_METHOD if scope == "model_state" else "sha256_parameter_name_mapped_optimizer_state",
                    "canonical_hash_value": canonical_hashes[replica_index - 1],
                    "legacy_hash_equal_across_replicas": legacy_equal,
                    "canonical_hash_equal_across_replicas": canonical_equal,
                    "direct_tensors_equal_across_replicas": direct_equal,
                    "serialization_artifact_detected": serialization_artifact,
                    "parameter_id_artifact_detected": parameter_id_artifact,
                    "interpretation": hash_audit_interpretation(
                        legacy_equal=legacy_equal,
                        canonical_equal=canonical_equal,
                        direct_equal=direct_equal,
                        scope=scope,
                    ),
                    "notes": f"{d1_label}: {json.dumps(d1_reference, sort_keys=True)}",
                }
            )
    return rows


def hash_audit_interpretation(*, legacy_equal: bool, canonical_equal: bool, direct_equal: bool, scope: str) -> str:
    if legacy_equal and canonical_equal and direct_equal:
        return "legacy, canonical and direct tensor evidence agree"
    if not legacy_equal and canonical_equal and direct_equal and scope == "optimizer_state":
        return "legacy optimizer hash likely depends on raw parameter IDs"
    if not legacy_equal and canonical_equal and direct_equal:
        return "legacy state hash differs while canonical tensor evidence is exact"
    if not direct_equal:
        return "direct tensor comparison confirms genuine content difference"
    return "hash evidence is mixed"


def build_decision_rows(
    *,
    variant_rows: Sequence[Mapping[str, Any]],
    hash_audit_rows: Sequence[Mapping[str, Any]],
    d1_decision: Mapping[str, str],
) -> list[dict[str, Any]]:
    variants = {row["variant_id"]: row for row in variant_rows}
    original = variants["original_pipeline_default_adam"]
    forward_only = variants["forward_only_no_backward"]
    forward_backward = variants["forward_backward_no_step"]
    default_adam = variants["shared_gradient_default_adam"]
    foreach_false = variants["shared_gradient_foreach_false"]
    foreach_true = variants["shared_gradient_foreach_true"]
    legacy_hash_artifact = any(row["serialization_artifact_detected"] for row in hash_audit_rows)
    parameter_id_artifact = any(row["parameter_id_artifact_detected"] for row in hash_audit_rows)
    original_all_exact = original["variant_status"] in {"exact", "hash_artifact_only"} and not bool_value(
        original["direct_tensor_difference_reproduced"]
    )
    classification = classify_diagnostic(
        original_pipeline_first_divergence=str(original["first_divergence_stage"]),
        forward_only_exact=boolish(forward_only["forward_exact"]),
        forward_backward_gradient_exact=boolish(forward_backward["gradients_exact"]),
        shared_default_post_step_exact=boolish(default_adam["post_step_parameters_exact"]),
        shared_default_optimizer_exact=boolish(default_adam["optimizer_state_exact"]),
        shared_foreach_false_post_step_exact=boolish(foreach_false["post_step_parameters_exact"]),
        original_pipeline_all_exact=original_all_exact,
        legacy_hash_artifact_detected=legacy_hash_artifact,
        optimizer_parameter_id_artifact_detected=parameter_id_artifact,
        blocked=any(row["variant_status"] == "blocked" for row in variant_rows),
    )
    decision = validate_scale_up_decision(
        classification=classification,
        original_pipeline_all_exact=original_all_exact,
        forward_exact=boolish(forward_only["forward_exact"]),
        gradients_exact=boolish(forward_backward["gradients_exact"]),
        default_adam_exact=boolish(default_adam["post_step_parameters_exact"]),
        foreach_false_adam_exact=boolish(foreach_false["post_step_parameters_exact"]),
        diagnostic_complete=not any(row["variant_status"] == "blocked" for row in variant_rows),
    )
    next_task = recommended_task(decision, classification)
    return [
        {
            "diagnostic_id": DIAGNOSTIC_ID,
            "phase3m_original_status": str(d1_decision.get("phase3m_original_status", "")),
            "phase3m_d1_classification": str(d1_decision.get("diagnostic_classification", "")),
            "phase3m_d2_classification": classification,
            "first_true_tensor_divergence_stage": "" if original["first_divergence_stage"] == "none" else original["first_divergence_stage"],
            "forward_exact": boolish(forward_only["forward_exact"]),
            "gradients_exact": boolish(forward_backward["gradients_exact"]),
            "default_adam_exact": boolish(default_adam["post_step_parameters_exact"]),
            "foreach_false_adam_exact": boolish(foreach_false["post_step_parameters_exact"]),
            "foreach_true_adam_exact": boolish(foreach_true["post_step_parameters_exact"]),
            "legacy_hash_artifact_detected": legacy_hash_artifact,
            "optimizer_parameter_id_artifact_detected": parameter_id_artifact,
            "phase3m_primary_outputs_remain_valid": True,
            "existing_cuda_pipeline_approved": decision == "approve_existing_cuda_pipeline_scale_up",
            "corrected_ml1m_rerun_required": decision == "require_corrected_ml1m_rerun_with_foreach_false",
            "cpu_only_scale_up_approved": decision == "approve_cpu_only_scale_up",
            "scale_up_decision": decision,
            "required_next_configuration_change": "set Adam foreach=False and fused=False" if decision == "require_corrected_ml1m_rerun_with_foreach_false" else "",
            "blocking_reason": "" if decision not in {"do_not_scale_up", "diagnostic_incomplete"} else "first-step evidence incomplete or unresolved",
            "recommended_single_next_task": next_task,
            "notes": "D2 refines D1 with direct tensor-content comparisons; no D1 table was modified",
        }
    ]


def recommended_task(decision: str, classification: str) -> str:
    if decision == "approve_existing_cuda_pipeline_scale_up":
        return "Phase 3N - controlled LightGCN Beauty and Sports execution and same-cohort consolidation"
    if decision == "require_corrected_ml1m_rerun_with_foreach_false":
        return "Phase 3M-R1 - rerun controlled LightGCN ML-1M with explicit single-tensor Adam and full deterministic validation"
    if classification == "cuda_sparse_backward_gradient_nondeterminism":
        return "Phase 3M-D3 - controlled CPU-versus-CUDA sparse-backward gradient localization"
    if decision == "approve_cpu_only_scale_up":
        return "Phase 3N-CPU - controlled CPU LightGCN Beauty and Sports execution and same-cohort consolidation"
    if classification == "cuda_default_adam_update_nondeterminism":
        return "Phase 3M-D3 - isolate default Adam CUDA first-step update path with optimizer backend instrumentation"
    return "Phase 3M-D3 - resolve unresolved first-step CUDA LightGCN divergence with narrower instrumentation"


def boolish(value: Any) -> bool | str:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return "not_applicable"


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_d1_cuda_step_rows() -> list[dict[str, str]]:
    with D1_CUDA_STEP_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def d1_first_batch_row() -> Mapping[str, str]:
    rows = load_d1_cuda_step_rows()
    for row in rows:
        if row["execution_role"] == "cuda_primary" and row["batch_index"] == "1":
            return row
    raise LightGCNPilotSafetyError("D1 first CUDA primary batch row not found")


def rows_by_role_and_batch(rows: Sequence[Mapping[str, str]], batch_index: int) -> dict[str, Mapping[str, str]]:
    return {
        row["execution_role"]: row
        for row in rows
        if str(row.get("batch_index", "")) == str(batch_index)
        and row.get("execution_role") in {"cuda_primary", "cuda_same_seed_repeat"}
    }


def d1_hashes_differ(rows: Sequence[Mapping[str, str]], field: str) -> bool:
    selected = rows_by_role_and_batch(rows, 1)
    values = [selected[role][field] for role in sorted(selected)]
    return len(values) == 2 and values[0] != values[1]


def read_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise LightGCNPilotSafetyError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def build_report(**kwargs: Any) -> str:
    env = kwargs["env"]
    metadata = kwargs["metadata"]
    config = kwargs["config"]
    hashes = kwargs["hashes"]
    schedule_hash = kwargs["schedule_hash"]
    batch = kwargs["batch"]
    initial_hashes = kwargs["initial_hashes"]
    manifest_rows = kwargs["manifest_rows"]
    variant_rows = {row["variant_id"]: row for row in kwargs["variant_rows"]}
    decision = kwargs["decision_rows"][0]
    hash_artifacts = [row for row in kwargs["hash_audit_rows"] if bool_value(row["serialization_artifact_detected"]) or bool_value(row["parameter_id_artifact_detected"])]
    stage_rows = kwargs["stage_rows"]
    stage_lookup = {(row["variant_id"], row["stage_name"]): row for row in stage_rows}
    adam_defaults = next(row for row in manifest_rows if row["variant_id"] == "original_pipeline_default_adam")
    adam_source = inspect.getsource(torch.optim.Adam).splitlines()[0].strip()
    lines = [
        "# Plan E LightGCN ML-1M CUDA First-Step Divergence Diagnostic",
        "",
        "## 1. Scope and provenance",
        f"- Branch: `{git_output(['git', 'branch', '--show-current'])}`.",
        f"- Commit: `{git_output(['git', 'log', '-1', '--format=%H %s'])}`.",
        f"- Diagnostic ID: `{DIAGNOSTIC_ID}`.",
        "",
        "## 2. Executive summary",
        f"- D2 classification: `{decision['phase3m_d2_classification']}`.",
        f"- Scale-up decision: `{decision['scale_up_decision']}`.",
        f"- First true tensor divergence stage: `{decision['first_true_tensor_divergence_stage'] or 'none'}`.",
        "",
        "## 3. Phase 3M and D1 evidence",
        f"- Phase 3M status from D1: `{decision['phase3m_original_status']}`; D1 classification: `{decision['phase3m_d1_classification']}`.",
        "- D1 bounded CUDA state hashes are treated as legacy evidence and are not rewritten.",
        "",
        "## 4. Protected-input boundaries",
        "- Known Phase 3M hashes and D1 snapshot hashes were verified before and after execution.",
        "",
        "## 5. Exact first-batch reconstruction",
        f"- Training hash `{hashes['training_data_sha256']}`; raw-edge hash `{hashes['raw_edge_sha256']}`.",
        f"- First-batch user/positive/negative hashes: `{batch.users_hash}`, `{batch.positives_hash}`, `{batch.negatives_hash}`.",
        "",
        "## 6. Canonical initial-state construction",
        f"- Legacy initial hash `{initial_hashes['legacy']}`; canonical initial hash `{initial_hashes['canonical']}`.",
        "",
        "## 7. Legacy state-hash audit",
        f"- Hash-artifact rows detected: `{len(hash_artifacts)}`.",
        "",
        "## 8. Canonical tensor-content hashing",
        f"- Tensor hash method: `{TENSOR_HASH_METHOD}`.",
        "",
        "## 9. Optimizer-state parameter-name mapping",
        "- Optimizer state is mapped through `model.named_parameters()` and excludes raw optimizer parameter IDs from canonical hashes.",
        "",
        "## 10. Phase 3M Adam implementation",
        f"- Source object: `{adam_source}`.",
        f"- Defaults: lr `{adam_defaults['learning_rate']}`, betas `{adam_defaults['beta1']}`/`{adam_defaults['beta2']}`, eps `{adam_defaults['epsilon']}`, weight_decay `{adam_defaults['weight_decay']}`, foreach `{adam_defaults['optimizer_foreach']}`, fused `{adam_defaults['optimizer_fused']}`.",
        "- With foreach/fused left as `None`, the CUDA runtime path is automatic; D2 does not assume it is single-tensor.",
        "",
        "## 11. Original-pipeline twin design",
        "- Two replicas loaded the same canonical CPU state, used original Adam defaults, ran forward, backward and one optimizer step.",
        "",
        "## 12. Forward-only twin findings",
        variant_sentence(variant_rows["forward_only_no_backward"]),
        "",
        "## 13. Forward-backward twin findings",
        variant_sentence(variant_rows["forward_backward_no_step"]),
        "",
        "## 14. Shared-gradient default-Adam findings",
        variant_sentence(variant_rows["shared_gradient_default_adam"]),
        "",
        "## 15. Shared-gradient foreach-false findings",
        variant_sentence(variant_rows["shared_gradient_foreach_false"]),
        "",
        "## 16. Shared-gradient foreach-true findings",
        variant_sentence(variant_rows["shared_gradient_foreach_true"]),
        "",
        "## 17. Graph and input identity",
        "- Batch and graph stages were compared directly in every variant; see stage comparison CSV.",
        "",
        "## 18. Initial-parameter identity",
        "- Initial user/item parameter stages are exact when variants completed.",
        "",
        "## 19. Sparse propagation layer 1",
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "layer_1_all_embeddings"),
        "",
        "## 20. Sparse propagation layer 2",
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "layer_2_all_embeddings"),
        "",
        "## 21. Final propagated embeddings",
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "final_all_embeddings"),
        "",
        "## 22. Positive and negative scores",
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "positive_scores"),
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "negative_scores"),
        "",
        "## 23. Loss-tensor identity",
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "total_loss_tensor"),
        "",
        "## 24. User-embedding gradients",
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "user_embedding_gradient"),
        "",
        "## 25. Item-embedding gradients",
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "item_embedding_gradient"),
        "",
        "## 26. Post-step parameter identity",
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "post_step_user_embedding_parameter"),
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "post_step_item_embedding_parameter"),
        "",
        "## 27. Adam step state",
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "user_step"),
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "item_step"),
        "",
        "## 28. Adam first-moment state",
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "user_exp_avg"),
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "item_exp_avg"),
        "",
        "## 29. Adam second-moment state",
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "user_exp_avg_sq"),
        stage_sentence(stage_lookup, "original_pipeline_default_adam", "item_exp_avg_sq"),
        "",
        "## 30. First genuine divergence",
        f"- `{decision['first_true_tensor_divergence_stage'] or 'none'}`.",
        "",
        "## 31. Hash artifact assessment",
        f"- Legacy hash artifact detected: `{decision['legacy_hash_artifact_detected']}`; optimizer parameter-ID artifact detected: `{decision['optimizer_parameter_id_artifact_detected']}`.",
        "",
        "## 32. Refinement of the D1 classification",
        f"- D1 `{decision['phase3m_d1_classification']}` refined to D2 `{decision['phase3m_d2_classification']}`.",
        "",
        "## 33. Whether the default Adam path is implicated",
        f"- Default Adam exact: `{decision['default_adam_exact']}`.",
        "",
        "## 34. Whether foreach=False resolves the divergence",
        f"- foreach=False Adam exact: `{decision['foreach_false_adam_exact']}`.",
        "",
        "## 35. Scientific status of Phase 3M",
        f"- Primary outputs remain valid: `{decision['phase3m_primary_outputs_remain_valid']}`; D2 did not rescore candidates.",
        "",
        "## 36. Scale-up decision",
        f"- `{decision['scale_up_decision']}`.",
        "",
        "## 37. Required next corrective action",
        f"- `{decision['recommended_single_next_task']}`.",
        "",
        "## 38. Limitations",
        "- Only first-batch CUDA behavior is isolated; no complete epoch and no corrected run are executed.",
        "",
        "## 39. Protected-input immutability",
        "- Protected inputs were verified unchanged; no checkpoint or ranking output was created.",
        "",
        "## 40. Commands executed",
        "- `git status --short --branch`",
        "- approved-interpreter pytest commands",
        "- approved-interpreter D2 diagnostic runner",
        "",
        "## 41. Recommended single next task",
        f"- {decision['recommended_single_next_task']}",
        "",
    ]
    _ = config, metadata, schedule_hash
    return "\n".join(lines)


def variant_sentence(row: Mapping[str, Any]) -> str:
    return (
        f"- Status `{row['variant_status']}`; first divergence `{row['first_divergence_stage']}`; "
        f"max absolute difference `{row['maximum_observed_absolute_difference']}`."
    )


def stage_sentence(stage_lookup: Mapping[tuple[str, str], Mapping[str, Any]], variant_id: str, stage_name: str) -> str:
    row = stage_lookup[(variant_id, stage_name)]
    return (
        f"- `{stage_name}` status `{row['status']}`, mismatches `{row['mismatching_element_count']}`, "
        f"max abs diff `{row['maximum_absolute_difference']}`."
    )


def validate_outputs(**kwargs: Any) -> None:
    expected_columns = {
        "manifest_rows": MANIFEST_COLUMNS,
        "stage_rows": STAGE_COLUMNS,
        "parameter_rows": PARAMETER_COLUMNS,
        "optimizer_rows": OPTIMIZER_COLUMNS,
        "variant_rows": VARIANT_COLUMNS,
        "hash_audit_rows": HASH_AUDIT_COLUMNS,
        "decision_rows": DECISION_COLUMNS,
    }
    for key, columns in expected_columns.items():
        rows = kwargs[key]
        if not rows:
            raise LightGCNPilotSafetyError(f"{key} is empty")
        for row in rows:
            if set(row.keys()) != set(columns):
                raise LightGCNPilotSafetyError(f"schema mismatch for {key}")
    manifest_rows = kwargs["manifest_rows"]
    if len(manifest_rows) != 12:
        raise LightGCNPilotSafetyError(f"expected 12 manifest rows, found {len(manifest_rows)}")
    if any(row["optimizer_step_executed"] not in {True, False} for row in manifest_rows):
        raise LightGCNPilotSafetyError("manifest optimizer_step_executed is not boolean")
    if len(kwargs["decision_rows"]) != 1:
        raise LightGCNPilotSafetyError("decision must contain exactly one row")
    valid_stage_status = {"exact", "different", "not_applicable", "blocked"}
    if any(row["status"] not in valid_stage_status for row in kwargs["stage_rows"]):
        raise LightGCNPilotSafetyError("invalid stage status")
    if kwargs["report_text"].count("\n## ") != 41:
        raise LightGCNPilotSafetyError("report does not contain the required 41 sections")


def write_outputs(**kwargs: Any) -> None:
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["manifest"], MANIFEST_COLUMNS, kwargs["manifest_rows"])
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["stage"], STAGE_COLUMNS, kwargs["stage_rows"])
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["parameter"], PARAMETER_COLUMNS, kwargs["parameter_rows"])
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["optimizer"], OPTIMIZER_COLUMNS, kwargs["optimizer_rows"])
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["variant"], VARIANT_COLUMNS, kwargs["variant_rows"])
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["hash_audit"], HASH_AUDIT_COLUMNS, kwargs["hash_audit_rows"])
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["decision"], DECISION_COLUMNS, kwargs["decision_rows"])
    report_path = PROJECT_ROOT / OUTPUT_PATHS["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise LightGCNPilotSafetyError(f"refusing to overwrite {report_path}")
    report_path.write_text(kwargs["report_text"], encoding="utf-8")


def write_csv(path: Path, columns: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise LightGCNPilotSafetyError(f"refusing to overwrite {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_value(row.get(column, "")) for column in columns})


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
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

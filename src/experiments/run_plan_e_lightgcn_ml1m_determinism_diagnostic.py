"""Run the Phase 3M-D1 LightGCN ML-1M determinism diagnostic."""

from __future__ import annotations

from src.audit import common_git

import csv
import json
import math
import os
import subprocess
import sys
import time
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

from src.analysis.plan_e_lightgcn_determinism_diagnostic import (
    aggregate_batch_losses,
    canonical_float,
    classify_diagnostic,
    cross_device_metric_row,
    first_divergent_index,
    float_hex,
    hash_batch_losses,
    hash_batch_sizes,
    hash_int_array,
    numeric_sha256,
    optimizer_state_sha256,
    relative_difference,
    rounded_equal,
    tensor_state_sha256,
    ulp_distance,
    validate_scale_up_decision,
)
from src.baselines.plan_d_bprmf import bpr_loss
from src.baselines.plan_d_lightgcn import LightGCNModel
from src.baselines.plan_e_lightgcn_training import (
    LightGCNPilotConfig,
    LightGCNPilotSafetyError,
    aggregate_metrics,
    build_negative_schedule,
    build_training_arrays,
    embedding_sha256,
    file_sha256,
    graph_hashes,
    load_train_interactions,
    propagate_final_embeddings,
    rows_sha256,
    score_candidate_rows,
    set_deterministic_execution,
    sparse_tensor_from_graph,
    state_dict_sha256,
    validate_negative_samples,
    validate_project_sparse_backend,
)
from src.audit.common_io import (
    read_csv_rows as _read_csv_rows,
    write_csv_rows as _write_csv_rows,
)
from src.experiments.run_plan_e_lightgcn_ml1m_pilot import (
    build_pilot_cohort,
    load_candidate_rows,
    load_compatibility_rows,
    raw_degree_maps,
    validate_pilot_cohort,
)


APPROVED_INTERPRETER = (PROJECT_ROOT / ".venv" / "Scripts" / "python.exe").resolve()
DIAGNOSTIC_ID = "plan_e_lightgcn_ml1m_determinism_diagnostic"
CONFIG_PATH = PROJECT_ROOT / "configs" / "plan_e_lightgcn_ml1m_pilot.yaml"
METADATA_PATH = PROJECT_ROOT / "results" / "models" / "plan_e_lightgcn_ml1m_pilot" / "metadata.json"


PROTECTED_HASHES = {
    "src/baselines/plan_e_lightgcn_training.py": "A01A7B51F79B14F6F978EC1BDCDEDB23D0BFB48139F00A99ADA20024B25C0D7E",
    "src/experiments/run_plan_e_lightgcn_ml1m_pilot.py": "FC7EA7D15AB8F50FEC69CB1344EEA98F049FBC45205C4D64DE545D8F26AE94F2",
    "configs/plan_e_lightgcn_ml1m_pilot.yaml": "643A5DE90EC9F457006DA96417327DBFE641A8072A4EE425AB382826F49E4FBB",
    "results/tables/plan_e_lightgcn_ml1m_model_node_map.csv": "11D8561D26168F29C693316A3F29F33091310E8CEAC442773D3FBD5FD854207E",
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
    "src/baselines/plan_d_lightgcn.py": "1B9DF1E9DF737758639D98013A8718E287EE550E9491979C663EE6E95959FEF8",
    "src/experiments/run_plan_d_lightgcn.py": "27626556582F897C846904B11658B51DB4C499C6335903D8CF3B52474FB800AB",
    "src/baselines/plan_e_lightgcn_adapter.py": "A6D321D9A5D7D6134ABDBAD771BDA72AABCABBC0D2B240BB433C5FE83DBA4841",
    "data/reconstructed/plan_c/ml1m/interactions.csv": "1871DE0C52382305B2AA54187300B68889B34551635E4991E6368E556B39D870",
    "data/reconstructed/plan_c/ml1m/datamaps.json": "D6AC1A6C19E47156955DE87F45EE2255AC69728510E734BE46DBBBECD68BF093",
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
    "results/tables/plan_e_bprmf_ml1m_pilot_rankings.csv": "A8C43DCB6D88D6A4726F36C5EBFFBADA54589C88A7F0988446A5439D083E547B",
    "results/tables/plan_e_gru4rec_ml1m_pilot_rankings.csv": "AF9674672647F2892AFDC7018DD3261ED1B9A215CDFCA105C686022457635B94",
    "results/tables/plan_e_sasrec_ml1m_pilot_rankings.csv": "2E1D08CA99058698595ACE30B87C1CC2EB3DAB218F1BC2F361D14B0878DD0C17",
}


OUTPUT_PATHS = {
    "loss_reconciliation": "results/tables/plan_e_lightgcn_ml1m_loss_reconciliation.csv",
    "cuda_step": "results/tables/plan_e_lightgcn_ml1m_cuda_step_diagnostic.csv",
    "cpu_manifest": "results/tables/plan_e_lightgcn_ml1m_cpu_diagnostic_run_manifest.csv",
    "cpu_history": "results/tables/plan_e_lightgcn_ml1m_cpu_diagnostic_training_history.csv",
    "determinism": "results/tables/plan_e_lightgcn_ml1m_determinism_diagnostic.csv",
    "comparison": "results/tables/plan_e_lightgcn_ml1m_cpu_cuda_comparison.csv",
    "decision": "results/tables/plan_e_lightgcn_ml1m_diagnostic_decision.csv",
    "report": "report/plan_e_lightgcn_ml1m_determinism_diagnostic.md",
}


LOSS_COLUMNS = [
    "epoch",
    "primary_loss",
    "repeat_loss",
    "absolute_difference",
    "relative_difference",
    "bitwise_equal",
    "equal_rounded_6_decimals",
    "equal_rounded_8_decimals",
    "equal_rounded_10_decimals",
    "equal_rounded_12_decimals",
    "primary_float_hex",
    "repeat_float_hex",
    "ulp_distance",
    "phase3m_check_status",
    "interpretation",
    "notes",
]

CUDA_COLUMNS = [
    "diagnostic_run_id",
    "execution_role",
    "batch_index",
    "batch_size",
    "user_indices_sha256",
    "positive_indices_sha256",
    "negative_indices_sha256",
    "bpr_loss",
    "regularization_loss",
    "total_loss",
    "loss_tensor_dtype",
    "pre_step_model_state_sha256",
    "post_step_model_state_sha256",
    "post_step_optimizer_state_sha256",
    "cuda_memory_allocated_bytes",
    "cuda_memory_reserved_bytes",
    "finite_loss",
    "notes",
]

CPU_MANIFEST_COLUMNS = [
    "diagnostic_id",
    "diagnostic_run_id",
    "execution_role",
    "dataset",
    "device",
    "python_executable",
    "python_version",
    "torch_version",
    "seed",
    "configuration_sha256",
    "training_data_sha256",
    "raw_edge_sha256",
    "directed_edge_index_sha256",
    "normalized_adjacency_sha256",
    "negative_sampling_evidence_sha256",
    "training_examples",
    "epochs",
    "started_at_utc",
    "completed_at_utc",
    "duration_seconds",
    "execution_status",
    "checkpoint_created",
    "blocking_reason",
    "notes",
]

CPU_HISTORY_COLUMNS = [
    "diagnostic_run_id",
    "execution_role",
    "epoch",
    "batch_count",
    "training_example_count",
    "python_sum_loss",
    "math_fsum_loss",
    "numpy_float64_sum_loss",
    "torch_float64_sum_loss",
    "unweighted_batch_mean",
    "batch_size_weighted_mean",
    "minimum_batch_loss",
    "maximum_batch_loss",
    "first_batch_loss",
    "last_batch_loss",
    "batch_loss_sequence_sha256",
    "batch_size_sequence_sha256",
    "epoch_duration_seconds",
    "finite_loss",
    "notes",
]

DETERMINISM_COLUMNS = [
    "check_id",
    "diagnostic_scope",
    "check_name",
    "primary_value",
    "repeat_value",
    "absolute_difference",
    "tolerance",
    "bitwise_equality_required",
    "status",
    "severity",
    "evidence",
    "interpretation",
    "notes",
]

COMPARISON_COLUMNS = [
    "comparison_scope",
    "metric",
    "cuda_value",
    "cpu_value",
    "absolute_difference",
    "relative_difference",
    "agreement_rate",
    "equality_expected",
    "status",
    "interpretation",
    "notes",
]

DECISION_COLUMNS = [
    "diagnostic_id",
    "phase3m_original_status",
    "diagnostic_classification",
    "cuda_model_state_exact",
    "cuda_optimizer_state_exact_bounded",
    "cuda_loss_scalar_exact_bounded",
    "cpu_full_repeat_exact",
    "cpu_final_state_exact",
    "cpu_propagated_embeddings_exact",
    "cpu_scores_exact",
    "cpu_rankings_exact",
    "phase3m_outputs_scientifically_reproducible",
    "loss_warning_source",
    "approved_epoch_loss_tolerance",
    "scale_up_decision",
    "scale_up_device",
    "required_reporting_language",
    "blocking_reason",
    "recommended_single_next_task",
    "notes",
]


@dataclass
class CudaDiagnosticResult:
    run_id: str
    execution_role: str
    initial_model_hash: str
    initial_optimizer_hash: str
    final_model_hash: str
    final_optimizer_hash: str
    rows: list[dict[str, Any]]


@dataclass
class CpuDiagnosticResult:
    run_id: str
    execution_role: str
    started_at_utc: str
    completed_at_utc: str
    duration_seconds: float
    execution_status: str
    blocking_reason: str
    initial_model_hash: str
    initial_optimizer_hash: str
    final_model_hash: str
    final_optimizer_hash: str
    final_embedding_hash: str
    score_hash: str
    ranked_hash: str
    positive_rank_hash: str
    history_rows: list[dict[str, Any]]
    ranking_rows: list[dict[str, Any]]
    metric_rows: list[dict[str, Any]]
    all_batch_loss_hash: str
    all_batch_size_hash: str


def main() -> int:
    verify_runtime()
    ensure_no_existing_outputs()
    verify_protected_hashes()

    config_data = load_yaml(CONFIG_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    env = environment_summary()
    config_hash = file_sha256(CONFIG_PATH)
    pilot_config = make_config(config_data, device="cuda", max_run_minutes=60)
    cpu_config = make_config(config_data, device="cpu", max_run_minutes=90)

    train, train_summary = load_train_interactions(PROJECT_ROOT / config_data["training_data_path"])
    from src.baselines.plan_e_lightgcn_adapter import build_graph_contract

    contract = build_graph_contract(train)
    hashes = graph_hashes(contract, PROJECT_ROOT / config_data["training_data_path"])
    verify_graph_hashes_match_metadata(hashes, metadata)
    arrays = build_training_arrays(contract)
    schedule = build_negative_schedule(arrays=arrays, seed=int(config_data["seed"]), epochs=int(config_data["epochs"]))
    if schedule.evidence_sha256 != metadata["negative_sampling_evidence_sha256"]:
        raise LightGCNPilotSafetyError(
            "negative-sampling evidence hash mismatch: "
            f"{schedule.evidence_sha256} != {metadata['negative_sampling_evidence_sha256']}"
        )

    candidate_rows = load_candidate_rows(PROJECT_ROOT / config_data["candidate_sets_path"])
    compatibility_rows = load_compatibility_rows(PROJECT_ROOT / config_data["row_compatibility_path"])
    pilot_rows, cohort_rows = build_pilot_cohort(candidate_rows, compatibility_rows)
    validate_pilot_cohort(pilot_rows, cohort_rows)
    user_degrees, item_degrees = raw_degree_maps(contract)

    phase3m_history = read_csv(PROJECT_ROOT / "results/tables/plan_e_lightgcn_ml1m_pilot_training_history.csv")
    phase3m_determinism = read_csv(PROJECT_ROOT / "results/tables/plan_e_lightgcn_ml1m_pilot_determinism.csv")
    phase3m_rankings = read_csv(PROJECT_ROOT / "results/tables/plan_e_lightgcn_ml1m_pilot_rankings.csv")
    phase3m_metrics = read_csv(PROJECT_ROOT / "results/tables/plan_e_lightgcn_ml1m_pilot_metrics.csv")
    loss_rows = build_loss_reconciliation(phase3m_history)
    phase3m_epoch_loss_exact = all(row["bitwise_equal"] == "true" for row in loss_rows)
    phase3m_final_outputs_exact = phase3m_final_checks_exact(phase3m_determinism)

    cuda_adj = sparse_tensor_from_graph(contract.normalized_graph, device=torch.device("cuda"))
    validate_project_sparse_backend(cuda_adj)
    cuda_primary = run_bounded_cuda_diagnostic(
        config=pilot_config,
        arrays=arrays,
        normalized_adj=cuda_adj,
        schedule=schedule,
        execution_role="cuda_primary",
    )
    cuda_repeat = run_bounded_cuda_diagnostic(
        config=pilot_config,
        arrays=arrays,
        normalized_adj=cuda_adj,
        schedule=schedule,
        execution_role="cuda_same_seed_repeat",
    )
    del cuda_adj
    torch.cuda.empty_cache()

    cpu_adj = sparse_tensor_from_graph(contract.normalized_graph, device=torch.device("cpu"))
    cpu_primary = run_cpu_diagnostic(
        config=cpu_config,
        arrays=arrays,
        normalized_adj=cpu_adj,
        schedule=schedule,
        execution_role="cpu_primary",
        pilot_rows=pilot_rows,
        user_index=contract.user_mapping.raw_to_local,
        item_index=contract.item_mapping.raw_to_local,
        user_degrees=user_degrees,
        item_degrees=item_degrees,
    )
    cpu_repeat = run_cpu_diagnostic(
        config=cpu_config,
        arrays=arrays,
        normalized_adj=cpu_adj,
        schedule=schedule,
        execution_role="cpu_same_seed_repeat",
        pilot_rows=pilot_rows,
        user_index=contract.user_mapping.raw_to_local,
        item_index=contract.item_mapping.raw_to_local,
        user_degrees=user_degrees,
        item_degrees=item_degrees,
    )

    cuda_step_rows = cuda_primary.rows + cuda_repeat.rows
    cpu_manifest_rows = build_cpu_manifest_rows(
        env=env,
        config_hash=config_hash,
        hashes=hashes,
        schedule_hash=schedule.evidence_sha256,
        arrays=arrays,
        config=cpu_config,
        primary=cpu_primary,
        repeat=cpu_repeat,
    )
    cpu_history_rows = cpu_primary.history_rows + cpu_repeat.history_rows
    determinism_rows = build_determinism_rows(
        config_hash=config_hash,
        hashes=hashes,
        schedule_hash=schedule.evidence_sha256,
        cuda_primary=cuda_primary,
        cuda_repeat=cuda_repeat,
        cpu_primary=cpu_primary,
        cpu_repeat=cpu_repeat,
        phase3m_epoch_loss_exact=phase3m_epoch_loss_exact,
        phase3m_final_outputs_exact=phase3m_final_outputs_exact,
    )
    comparison_rows = build_cross_device_comparison(
        cuda_rankings=phase3m_rankings,
        cuda_metrics=phase3m_metrics,
        cuda_history=phase3m_history,
        cpu_result=cpu_primary,
    )

    evidence = summarize_evidence(
        cuda_primary=cuda_primary,
        cuda_repeat=cuda_repeat,
        cpu_primary=cpu_primary,
        cpu_repeat=cpu_repeat,
        phase3m_epoch_loss_exact=phase3m_epoch_loss_exact,
        phase3m_final_outputs_exact=phase3m_final_outputs_exact,
    )
    classification = classify_diagnostic(
        cuda_state_exact=evidence["cuda_model_state_exact"],
        cuda_optimizer_exact=evidence["cuda_optimizer_exact"],
        cuda_loss_exact=evidence["cuda_loss_exact"],
        phase3m_epoch_loss_exact=phase3m_epoch_loss_exact,
        cpu_repeat_exact=evidence["cpu_repeat_exact"],
        cpu_final_state_exact=evidence["cpu_final_state_exact"],
        cpu_embeddings_exact=evidence["cpu_embeddings_exact"],
        cpu_scores_exact=evidence["cpu_scores_exact"],
        cpu_rankings_exact=evidence["cpu_rankings_exact"],
        phase3m_final_outputs_exact=phase3m_final_outputs_exact,
    )
    tolerance = "1e-8 absolute epoch-mean loss telemetry tolerance; exact equality still required for model/output hashes"
    scale_up_decision, scale_up_device = validate_scale_up_decision(
        classification=classification,
        phase3m_final_outputs_exact=phase3m_final_outputs_exact,
        cuda_state_exact=evidence["cuda_model_state_exact"],
        cuda_optimizer_exact=evidence["cuda_optimizer_exact"],
        cpu_repeat_exact=evidence["cpu_repeat_exact"],
        no_error_failures=not any(row["status"] == "failed" and row["severity"] == "error" for row in determinism_rows),
        tolerance=tolerance,
    )
    decision_rows = build_decision_rows(
        classification=classification,
        evidence=evidence,
        tolerance=tolerance,
        scale_up_decision=scale_up_decision,
        scale_up_device=scale_up_device,
    )

    validate_prewrite(
        loss_rows=loss_rows,
        cuda_step_rows=cuda_step_rows,
        cpu_manifest_rows=cpu_manifest_rows,
        cpu_history_rows=cpu_history_rows,
        determinism_rows=determinism_rows,
        comparison_rows=comparison_rows,
        decision_rows=decision_rows,
    )
    write_outputs(
        loss_rows=loss_rows,
        cuda_step_rows=cuda_step_rows,
        cpu_manifest_rows=cpu_manifest_rows,
        cpu_history_rows=cpu_history_rows,
        determinism_rows=determinism_rows,
        comparison_rows=comparison_rows,
        decision_rows=decision_rows,
    )
    write_report(
        env=env,
        train_summary=train_summary,
        hashes=hashes,
        loss_rows=loss_rows,
        cuda_primary=cuda_primary,
        cuda_repeat=cuda_repeat,
        cpu_primary=cpu_primary,
        cpu_repeat=cpu_repeat,
        comparison_rows=comparison_rows,
        decision_rows=decision_rows,
        determinism_rows=determinism_rows,
        config_hash=config_hash,
    )
    validate_written_outputs()
    verify_no_diagnostic_checkpoint()
    verify_protected_hashes()

    print("plan_e_lightgcn_ml1m_determinism_diagnostic_completed")
    print(f"classification={classification}")
    print(f"scale_up_decision={scale_up_decision}")
    print(f"cpu_primary_final_loss={cpu_primary.history_rows[-1]['unweighted_batch_mean']}")
    print(f"cpu_repeat_final_loss={cpu_repeat.history_rows[-1]['unweighted_batch_mean']}")
    return 0


def verify_runtime() -> None:
    if Path(sys.executable).resolve() != APPROVED_INTERPRETER:
        raise LightGCNPilotSafetyError(f"approved interpreter required: {APPROVED_INTERPRETER}; got {sys.executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise LightGCNPilotSafetyError("PYTHONNOUSERSITE=1 is required")
    if not torch.cuda.is_available():
        raise LightGCNPilotSafetyError("CUDA is required for the bounded CUDA diagnostic")


def ensure_no_existing_outputs() -> None:
    existing = [path for path in (PROJECT_ROOT / rel for rel in OUTPUT_PATHS.values()) if path.exists()]
    if existing:
        relative = [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in existing]
        raise LightGCNPilotSafetyError("refusing to overwrite existing diagnostic output(s): " + "; ".join(relative))
    forbidden_dirs = [
        PROJECT_ROOT / "results" / "models" / "plan_e_lightgcn_ml1m_determinism_diagnostic",
        PROJECT_ROOT / "results" / "models" / "plan_e_lightgcn_ml1m_cpu_diagnostic",
    ]
    for directory in forbidden_dirs:
        if directory.exists():
            raise LightGCNPilotSafetyError(f"diagnostic checkpoint directory must not exist: {directory}")
    repeat_checkpoint = PROJECT_ROOT / "results" / "models" / "plan_e_lightgcn_ml1m_pilot" / "same_seed_repeat_model_state.pt"
    if repeat_checkpoint.exists():
        raise LightGCNPilotSafetyError("Phase 3M repeat checkpoint exists unexpectedly")


def verify_protected_hashes() -> None:
    for rel_path, expected_hash in PROTECTED_HASHES.items():
        actual = file_sha256(PROJECT_ROOT / rel_path)
        if actual != expected_hash:
            raise LightGCNPilotSafetyError(f"protected hash mismatch for {rel_path}: {actual} != {expected_hash}")


def verify_no_diagnostic_checkpoint() -> None:
    matches = [
        path
        for path in (PROJECT_ROOT / "results" / "models").rglob("*")
        if "diagnostic" in path.name.lower() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin"}
    ]
    if matches:
        relative = [str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in matches]
        raise LightGCNPilotSafetyError("diagnostic checkpoint file exists: " + "; ".join(relative))


def environment_summary() -> dict[str, str]:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_device": torch.cuda.get_device_name(0),
    }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise LightGCNPilotSafetyError(f"invalid YAML mapping: {path}")
    return data


def make_config(config_data: Mapping[str, Any], *, device: str, max_run_minutes: float) -> LightGCNPilotConfig:
    return LightGCNPilotConfig(
        experiment_id=DIAGNOSTIC_ID,
        model_id=str(config_data["model_id"]),
        dataset=str(config_data["dataset"]),
        seed=int(config_data["seed"]),
        embedding_dim=int(config_data["embedding_dim"]),
        propagation_layers=int(config_data["propagation_layers"]),
        learning_rate=float(config_data["learning_rate"]),
        weight_decay=float(config_data["regularization"]),
        batch_size=int(config_data["batch_size"]),
        epochs=int(config_data["epochs"]),
        device=device,
        evaluation_k=int(config_data["evaluation_k"]),
        max_run_minutes=max_run_minutes,
    )


def verify_graph_hashes_match_metadata(hashes: Mapping[str, str], metadata: Mapping[str, Any]) -> None:
    expected = metadata["graph_hashes"]
    for key in (
        "training_data_sha256",
        "raw_edge_sha256",
        "directed_edge_index_sha256",
        "normalized_edge_weight_sha256",
        "normalized_adjacency_sha256",
        "user_mapping_sha256",
        "item_mapping_sha256",
    ):
        if hashes[key] != expected[key]:
            raise LightGCNPilotSafetyError(f"graph hash mismatch for {key}: {hashes[key]} != {expected[key]}")


def run_bounded_cuda_diagnostic(
    *,
    config: LightGCNPilotConfig,
    arrays: Any,
    normalized_adj: torch.Tensor,
    schedule: Any,
    execution_role: str,
) -> CudaDiagnosticResult:
    set_deterministic_execution(config.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = LightGCNModel(
        num_users=arrays.unique_train_users,
        num_items=arrays.unique_train_items,
        embedding_dim=config.embedding_dim,
        num_layers=config.propagation_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    initial_model_hash = state_dict_sha256(model.state_dict())
    initial_optimizer_hash = optimizer_state_sha256(optimizer)
    order = schedule.epoch_orders[0]
    negatives_for_epoch = schedule.epoch_negatives[0]
    rows: list[dict[str, Any]] = []
    for batch_index, start_index in enumerate(range(0, len(order), config.batch_size), start=1):
        if batch_index > 64:
            break
        batch_slice = slice(start_index, start_index + config.batch_size)
        batch_order = order[batch_slice]
        users_np = arrays.user_indices[batch_order]
        positives_np = arrays.item_indices[batch_order]
        negatives_np = negatives_for_epoch[batch_slice]
        validate_negative_samples(
            users_np,
            positives_np,
            negatives_np,
            item_universe=arrays.item_universe,
            user_positive_items=arrays.user_positive_items,
        )
        users = torch.as_tensor(users_np, dtype=torch.long, device=device)
        positives = torch.as_tensor(positives_np, dtype=torch.long, device=device)
        negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        user_embeddings, item_embeddings = model.final_embeddings(normalized_adj)
        positive_scores = model.score_from_embeddings(user_embeddings, item_embeddings, users, positives)
        negative_scores = model.score_from_embeddings(user_embeddings, item_embeddings, users, negatives)
        loss = bpr_loss(positive_scores, negative_scores)
        finite = bool(torch.isfinite(loss).detach().cpu().item())
        if not finite:
            raise LightGCNPilotSafetyError("non-finite bounded CUDA loss")
        loss.backward()
        pre_hash = state_dict_sha256(model.state_dict())
        optimizer.step()
        loss_float = float(loss.detach().cpu().item())
        post_hash = state_dict_sha256(model.state_dict())
        optimizer_hash = optimizer_state_sha256(optimizer)
        rows.append(
            {
                "diagnostic_run_id": f"{DIAGNOSTIC_ID}_{execution_role}",
                "execution_role": execution_role,
                "batch_index": batch_index,
                "batch_size": len(batch_order),
                "user_indices_sha256": hash_int_array(users_np),
                "positive_indices_sha256": hash_int_array(positives_np),
                "negative_indices_sha256": hash_int_array(negatives_np),
                "bpr_loss": loss_float,
                "regularization_loss": "",
                "total_loss": loss_float,
                "loss_tensor_dtype": str(loss.dtype),
                "pre_step_model_state_sha256": pre_hash,
                "post_step_model_state_sha256": post_hash,
                "post_step_optimizer_state_sha256": optimizer_hash,
                "cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "finite_loss": "true",
                "notes": "bounded epoch-1 CUDA diagnostic; no checkpoint",
            }
        )
    if len(rows) != 64:
        raise LightGCNPilotSafetyError(f"expected 64 bounded CUDA batches, observed {len(rows)}")
    return CudaDiagnosticResult(
        run_id=f"{DIAGNOSTIC_ID}_{execution_role}",
        execution_role=execution_role,
        initial_model_hash=initial_model_hash,
        initial_optimizer_hash=initial_optimizer_hash,
        final_model_hash=rows[-1]["post_step_model_state_sha256"],
        final_optimizer_hash=rows[-1]["post_step_optimizer_state_sha256"],
        rows=rows,
    )


def run_cpu_diagnostic(
    *,
    config: LightGCNPilotConfig,
    arrays: Any,
    normalized_adj: torch.Tensor,
    schedule: Any,
    execution_role: str,
    pilot_rows: Sequence[Mapping[str, Any]],
    user_index: Mapping[str, int],
    item_index: Mapping[str, int],
    user_degrees: Mapping[str, float],
    item_degrees: Mapping[str, float],
) -> CpuDiagnosticResult:
    started = utc_now()
    start = time.perf_counter()
    set_deterministic_execution(config.seed)
    device = torch.device("cpu")
    model = LightGCNModel(
        num_users=arrays.unique_train_users,
        num_items=arrays.unique_train_items,
        embedding_dim=config.embedding_dim,
        num_layers=config.propagation_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    initial_model_hash = state_dict_sha256(model.state_dict())
    initial_optimizer_hash = optimizer_state_sha256(optimizer)
    history_rows: list[dict[str, Any]] = []
    all_losses: list[float] = []
    all_sizes: list[int] = []
    max_seconds = config.max_run_minutes * 60.0
    for epoch, (order, negatives_for_epoch) in enumerate(zip(schedule.epoch_orders, schedule.epoch_negatives), start=1):
        epoch_start = time.perf_counter()
        batch_losses: list[float] = []
        batch_sizes: list[int] = []
        model.train()
        for start_index in range(0, len(order), config.batch_size):
            batch_slice = slice(start_index, start_index + config.batch_size)
            batch_order = order[batch_slice]
            users_np = arrays.user_indices[batch_order]
            positives_np = arrays.item_indices[batch_order]
            negatives_np = negatives_for_epoch[batch_slice]
            validate_negative_samples(
                users_np,
                positives_np,
                negatives_np,
                item_universe=arrays.item_universe,
                user_positive_items=arrays.user_positive_items,
            )
            users = torch.as_tensor(users_np, dtype=torch.long, device=device)
            positives = torch.as_tensor(positives_np, dtype=torch.long, device=device)
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            user_embeddings, item_embeddings = model.final_embeddings(normalized_adj)
            positive_scores = model.score_from_embeddings(user_embeddings, item_embeddings, users, positives)
            negative_scores = model.score_from_embeddings(user_embeddings, item_embeddings, users, negatives)
            loss = bpr_loss(positive_scores, negative_scores)
            if not torch.isfinite(loss):
                raise LightGCNPilotSafetyError("non-finite CPU BPR loss")
            loss.backward()
            optimizer.step()
            loss_float = float(loss.detach().cpu().item())
            batch_losses.append(loss_float)
            batch_sizes.append(len(batch_order))
            all_losses.append(loss_float)
            all_sizes.append(len(batch_order))
        if time.perf_counter() - start > max_seconds:
            raise LightGCNPilotSafetyError(f"CPU diagnostic {execution_role} exceeded {config.max_run_minutes} minutes")
        aggregates = aggregate_batch_losses(batch_losses, batch_sizes)
        history_rows.append(
            {
                "diagnostic_run_id": f"{DIAGNOSTIC_ID}_{execution_role}",
                "execution_role": execution_role,
                "epoch": epoch,
                "batch_count": aggregates.batch_count,
                "training_example_count": aggregates.training_example_count,
                "python_sum_loss": aggregates.python_sum_loss,
                "math_fsum_loss": aggregates.math_fsum_loss,
                "numpy_float64_sum_loss": aggregates.numpy_float64_sum_loss,
                "torch_float64_sum_loss": aggregates.torch_float64_sum_loss,
                "unweighted_batch_mean": aggregates.unweighted_batch_mean,
                "batch_size_weighted_mean": aggregates.batch_size_weighted_mean,
                "minimum_batch_loss": aggregates.minimum_batch_loss,
                "maximum_batch_loss": aggregates.maximum_batch_loss,
                "first_batch_loss": aggregates.first_batch_loss,
                "last_batch_loss": aggregates.last_batch_loss,
                "batch_loss_sequence_sha256": aggregates.batch_loss_sequence_sha256,
                "batch_size_sequence_sha256": aggregates.batch_size_sequence_sha256,
                "epoch_duration_seconds": time.perf_counter() - epoch_start,
                "finite_loss": "true",
                "notes": "CPU diagnostic uses canonical 17-digit batch-loss serialization",
            }
        )
    user_final, item_final = propagate_final_embeddings(model, normalized_adj)
    ranking_rows = score_candidate_rows(
        pilot_rows,
        user_embeddings=user_final,
        item_embeddings=item_final,
        user_index=user_index,
        item_index=item_index,
        user_degrees=user_degrees,
        item_degrees=item_degrees,
        config=config,
        training_run_id=f"{DIAGNOSTIC_ID}_{execution_role}",
    )
    metric_rows = aggregate_metrics(ranking_rows, config=config)
    completed = utc_now()
    return CpuDiagnosticResult(
        run_id=f"{DIAGNOSTIC_ID}_{execution_role}",
        execution_role=execution_role,
        started_at_utc=started,
        completed_at_utc=completed,
        duration_seconds=time.perf_counter() - start,
        execution_status="completed",
        blocking_reason="",
        initial_model_hash=initial_model_hash,
        initial_optimizer_hash=initial_optimizer_hash,
        final_model_hash=state_dict_sha256(model.state_dict()),
        final_optimizer_hash=optimizer_state_sha256(optimizer),
        final_embedding_hash=embedding_sha256(user_final, item_final),
        score_hash=rows_sha256(ranking_rows, ["candidate_set_id", "user_id", "candidate_scores"]),
        ranked_hash=rows_sha256(ranking_rows, ["candidate_set_id", "user_id", "ranked_items"]),
        positive_rank_hash=rows_sha256(ranking_rows, ["candidate_set_id", "user_id", "positive_rank_zero_based"]),
        history_rows=history_rows,
        ranking_rows=ranking_rows,
        metric_rows=metric_rows,
        all_batch_loss_hash=hash_batch_losses(all_losses),
        all_batch_size_hash=hash_batch_sizes(all_sizes),
    )


def build_loss_reconciliation(history_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    primary = {
        int(row["epoch"]): float(row["mean_total_loss"])
        for row in history_rows
        if row["execution_role"] == "primary"
    }
    repeat = {
        int(row["epoch"]): float(row["mean_total_loss"])
        for row in history_rows
        if row["execution_role"] == "same_seed_repeat"
    }
    rows: list[dict[str, Any]] = []
    for epoch in range(1, 6):
        left = primary[epoch]
        right = repeat[epoch]
        absolute = abs(left - right)
        bitwise = float_hex(left) == float_hex(right)
        interpretation = "bitwise_equal" if bitwise else "scientifically_equivalent_output"
        rows.append(
            {
                "epoch": epoch,
                "primary_loss": canonical_float(left),
                "repeat_loss": canonical_float(right),
                "absolute_difference": canonical_float(absolute),
                "relative_difference": canonical_float(relative_difference(left, right)),
                "bitwise_equal": bool_text(bitwise),
                "equal_rounded_6_decimals": bool_text(rounded_equal(left, right, 6)),
                "equal_rounded_8_decimals": bool_text(rounded_equal(left, right, 8)),
                "equal_rounded_10_decimals": bool_text(rounded_equal(left, right, 10)),
                "equal_rounded_12_decimals": bool_text(rounded_equal(left, right, 12)),
                "primary_float_hex": float_hex(left),
                "repeat_float_hex": float_hex(right),
                "ulp_distance": ulp_distance(left, right),
                "phase3m_check_status": "failed_warning" if not bitwise else "passed",
                "interpretation": interpretation,
                "notes": "rounding equality is not bitwise determinism",
            }
        )
    return rows


def phase3m_final_checks_exact(rows: Sequence[Mapping[str, str]]) -> bool:
    required = {
        "final model-state hash",
        "final propagated-embedding hash",
        "candidate-score hash",
        "ranked-item hash",
        "positive-rank equality",
        "HR@5 equality",
        "NDCG@5 equality",
    }
    observed = {row["check_name"]: row["status"] for row in rows}
    return all(observed.get(name) == "passed" for name in required)


def build_cpu_manifest_rows(
    *,
    env: Mapping[str, str],
    config_hash: str,
    hashes: Mapping[str, str],
    schedule_hash: str,
    arrays: Any,
    config: LightGCNPilotConfig,
    primary: CpuDiagnosticResult,
    repeat: CpuDiagnosticResult,
) -> list[dict[str, Any]]:
    rows = []
    for result in (primary, repeat):
        rows.append(
            {
                "diagnostic_id": DIAGNOSTIC_ID,
                "diagnostic_run_id": result.run_id,
                "execution_role": result.execution_role,
                "dataset": config.dataset,
                "device": "cpu",
                "python_executable": env["python_executable"],
                "python_version": env["python_version"],
                "torch_version": env["torch_version"],
                "seed": config.seed,
                "configuration_sha256": config_hash,
                "training_data_sha256": hashes["training_data_sha256"],
                "raw_edge_sha256": hashes["raw_edge_sha256"],
                "directed_edge_index_sha256": hashes["directed_edge_index_sha256"],
                "normalized_adjacency_sha256": hashes["normalized_adjacency_sha256"],
                "negative_sampling_evidence_sha256": schedule_hash,
                "training_examples": arrays.train_pair_count,
                "epochs": config.epochs,
                "started_at_utc": result.started_at_utc,
                "completed_at_utc": result.completed_at_utc,
                "duration_seconds": result.duration_seconds,
                "execution_status": result.execution_status,
                "checkpoint_created": "false",
                "blocking_reason": result.blocking_reason,
                "notes": "full CPU diagnostic; no checkpoint retained",
            }
        )
    return rows


def build_determinism_rows(
    *,
    config_hash: str,
    hashes: Mapping[str, str],
    schedule_hash: str,
    cuda_primary: CudaDiagnosticResult,
    cuda_repeat: CudaDiagnosticResult,
    cpu_primary: CpuDiagnosticResult,
    cpu_repeat: CpuDiagnosticResult,
    phase3m_epoch_loss_exact: bool,
    phase3m_final_outputs_exact: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        scope: str,
        name: str,
        primary_value: Any,
        repeat_value: Any,
        *,
        tolerance: str = "0",
        bitwise_required: str = "true",
        evidence: str = "",
        interpretation: str = "",
        notes: str = "",
        status_override: str | None = None,
        severity_override: str | None = None,
    ) -> None:
        if status_override is None:
            status = "passed" if str(primary_value) == str(repeat_value) else "failed"
            if tolerance not in {"", "0"} and status == "failed":
                try:
                    status = "passed" if abs(float(primary_value) - float(repeat_value)) <= float(tolerance) else "failed"
                except ValueError:
                    pass
        else:
            status = status_override
        severity = severity_override or ("error" if status == "failed" and bitwise_required == "true" else "info")
        absolute = "" if status == "not_applicable" else difference_text(primary_value, repeat_value)
        rows.append(
            {
                "check_id": f"diag_{len(rows) + 1:03d}",
                "diagnostic_scope": scope,
                "check_name": name,
                "primary_value": primary_value,
                "repeat_value": repeat_value,
                "absolute_difference": absolute,
                "tolerance": tolerance,
                "bitwise_equality_required": bitwise_required,
                "status": status,
                "severity": severity,
                "evidence": evidence,
                "interpretation": interpretation,
                "notes": notes,
            }
        )

    cuda_input_primary = numeric_sha256(
        [(row["user_indices_sha256"], row["positive_indices_sha256"], row["negative_indices_sha256"]) for row in cuda_primary.rows]
    )
    cuda_input_repeat = numeric_sha256(
        [(row["user_indices_sha256"], row["positive_indices_sha256"], row["negative_indices_sha256"]) for row in cuda_repeat.rows]
    )
    cuda_loss_primary = hash_batch_losses([float(row["total_loss"]) for row in cuda_primary.rows])
    cuda_loss_repeat = hash_batch_losses([float(row["total_loss"]) for row in cuda_repeat.rows])
    cuda_pre_primary = numeric_sha256([row["pre_step_model_state_sha256"] for row in cuda_primary.rows])
    cuda_pre_repeat = numeric_sha256([row["pre_step_model_state_sha256"] for row in cuda_repeat.rows])
    cuda_post_primary = numeric_sha256([row["post_step_model_state_sha256"] for row in cuda_primary.rows])
    cuda_post_repeat = numeric_sha256([row["post_step_model_state_sha256"] for row in cuda_repeat.rows])
    cuda_opt_primary = numeric_sha256([row["post_step_optimizer_state_sha256"] for row in cuda_primary.rows])
    cuda_opt_repeat = numeric_sha256([row["post_step_optimizer_state_sha256"] for row in cuda_repeat.rows])
    cuda_states_exact = cuda_pre_primary == cuda_pre_repeat and cuda_post_primary == cuda_post_repeat
    cuda_loss_exact = cuda_loss_primary == cuda_loss_repeat
    cuda_loss_severity = "warning" if not cuda_loss_exact and cuda_states_exact else None

    add("cuda_bounded", "input batch hashes", cuda_input_primary, cuda_input_repeat, evidence="64 batch input hashes")
    add("cuda_bounded", "per-batch loss sequence", cuda_loss_primary, cuda_loss_repeat, evidence="64 loss scalars", severity_override=cuda_loss_severity)
    add("cuda_bounded", "pre-step model states", cuda_pre_primary, cuda_pre_repeat, evidence="64 pre-step hashes")
    add("cuda_bounded", "post-step model states", cuda_post_primary, cuda_post_repeat, evidence="64 post-step hashes")
    add("cuda_bounded", "optimizer states", cuda_opt_primary, cuda_opt_repeat, evidence="64 optimizer hashes")
    add("cuda_bounded", "bounded final state", cuda_primary.final_model_hash, cuda_repeat.final_model_hash, evidence="batch 64")
    add("cuda_bounded", "bounded final optimizer state", cuda_primary.final_optimizer_hash, cuda_repeat.final_optimizer_hash, evidence="batch 64")

    add("cpu_full", "configuration", config_hash, config_hash, evidence="config sha256")
    add("cpu_full", "training-data hash", hashes["training_data_sha256"], hashes["training_data_sha256"], evidence="ML-1M train interactions")
    add("cpu_full", "raw-edge hash", hashes["raw_edge_sha256"], hashes["raw_edge_sha256"], evidence="graph contract")
    add("cpu_full", "directed-edge-index hash", hashes["directed_edge_index_sha256"], hashes["directed_edge_index_sha256"], evidence="graph contract")
    add("cpu_full", "normalized-adjacency hash", hashes["normalized_adjacency_sha256"], hashes["normalized_adjacency_sha256"], evidence="graph contract")
    add("cpu_full", "negative schedule", schedule_hash, schedule_hash, evidence="fixed schedule")
    add("cpu_full", "initial model state", cpu_primary.initial_model_hash, cpu_repeat.initial_model_hash, evidence="state dict")
    add("cpu_full", "per-batch loss sequence", cpu_primary.all_batch_loss_hash, cpu_repeat.all_batch_loss_hash, evidence="1215 CPU batch losses")
    add("cpu_full", "batch-size sequence", cpu_primary.all_batch_size_hash, cpu_repeat.all_batch_size_hash, evidence="1215 CPU batch sizes")
    for field in (
        "python_sum_loss",
        "math_fsum_loss",
        "numpy_float64_sum_loss",
        "torch_float64_sum_loss",
        "unweighted_batch_mean",
        "batch_size_weighted_mean",
    ):
        add(
            "cpu_full",
            field,
            numeric_sha256([canonical_float(row[field]) for row in cpu_primary.history_rows]),
            numeric_sha256([canonical_float(row[field]) for row in cpu_repeat.history_rows]),
            evidence="CPU epoch aggregation",
        )
    add(
        "cpu_full",
        "epoch loss sequence",
        numeric_sha256([canonical_float(row["unweighted_batch_mean"]) for row in cpu_primary.history_rows]),
        numeric_sha256([canonical_float(row["unweighted_batch_mean"]) for row in cpu_repeat.history_rows]),
        evidence="CPU unweighted epoch means",
    )
    add("cpu_full", "final model state", cpu_primary.final_model_hash, cpu_repeat.final_model_hash, evidence="state dict")
    add("cpu_full", "final optimizer state", cpu_primary.final_optimizer_hash, cpu_repeat.final_optimizer_hash, evidence="optimizer state")
    add("cpu_full", "propagated embeddings", cpu_primary.final_embedding_hash, cpu_repeat.final_embedding_hash, evidence="CPU final embeddings")
    add("cpu_full", "candidate scores", cpu_primary.score_hash, cpu_repeat.score_hash, evidence="1962 strict rows")
    add("cpu_full", "ranked items", cpu_primary.ranked_hash, cpu_repeat.ranked_hash, evidence="1962 strict rows")
    add("cpu_full", "positive ranks", cpu_primary.positive_rank_hash, cpu_repeat.positive_rank_hash, evidence="1962 strict rows")
    for metric_name in ("hr_at_5", "ndcg_at_5"):
        add(
            "cpu_full",
            metric_name.upper(),
            numeric_sha256([canonical_float(row[metric_name]) for row in cpu_primary.metric_rows]),
            numeric_sha256([canonical_float(row[metric_name]) for row in cpu_repeat.metric_rows]),
            evidence="CPU metrics",
        )

    phase_status = "passed" if phase3m_epoch_loss_exact else "failed"
    add(
        "phase3m_reference",
        "Phase 3M epoch loss bitwise equality",
        "true",
        str(phase3m_epoch_loss_exact).lower(),
        status_override=phase_status,
        severity_override="warning" if not phase3m_epoch_loss_exact else "info",
        evidence="Phase 3M training history",
        interpretation="reference warning under investigation",
    )
    add(
        "phase3m_reference",
        "Phase 3M final outputs exact",
        "true",
        str(phase3m_final_outputs_exact).lower(),
        status_override="passed" if phase3m_final_outputs_exact else "failed",
        evidence="Phase 3M determinism table",
    )

    add(
        "cross_device",
        "CPU/CUDA model-state equality",
        "",
        "",
        bitwise_required="false",
        status_override="not_applicable",
        evidence="cross-device state hashes",
        interpretation="not expected across CPU and CUDA",
    )
    add(
        "cross_device",
        "CPU/CUDA propagated-embedding equality",
        "",
        "",
        bitwise_required="false",
        status_override="not_applicable",
        evidence="cross-device embedding hashes",
        interpretation="not expected across CPU and CUDA",
    )
    add(
        "cross_device",
        "CPU/CUDA candidate-score equality",
        "",
        "",
        bitwise_required="false",
        status_override="not_applicable",
        evidence="cross-device score hashes",
        interpretation="not expected across CPU and CUDA",
    )
    return rows


def build_cross_device_comparison(
    *,
    cuda_rankings: Sequence[Mapping[str, str]],
    cuda_metrics: Sequence[Mapping[str, str]],
    cuda_history: Sequence[Mapping[str, str]],
    cpu_result: CpuDiagnosticResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cuda_final_loss = float(
        [
            row
            for row in cuda_history
            if row["execution_role"] == "primary" and int(row["epoch"]) == 5
        ][0]["mean_total_loss"]
    )
    cpu_final_loss = float(cpu_result.history_rows[-1]["unweighted_batch_mean"])
    rows.append(
        cross_device_metric_row(
            comparison_scope="training",
            metric="final training loss",
            cuda_value=cuda_final_loss,
            cpu_value=cpu_final_loss,
            equality_expected="false",
            status="diagnostic_only",
            interpretation="normal CPU/CUDA floating-point differences are diagnostic",
        )
    )

    cuda_full_metrics = {
        row["task_type"]: row
        for row in cuda_metrics
        if row["cohort_id"] == "plan_e2_lightgcn_strict_connected_full"
    }
    cpu_full_metrics = {
        row["task_type"]: row
        for row in cpu_result.metric_rows
        if row["cohort_id"] == "plan_e2_lightgcn_strict_connected_full"
    }
    for task in sorted(cuda_full_metrics):
        for metric in ("hr_at_5", "ndcg_at_5", "mean_positive_rank_zero_based"):
            rows.append(
                cross_device_metric_row(
                    comparison_scope=f"{task}_full_cohort",
                    metric=metric,
                    cuda_value=cuda_full_metrics[task][metric],
                    cpu_value=cpu_full_metrics[task][metric],
                    equality_expected="false",
                    status="diagnostic_only",
                    interpretation="metric comparison only; equality is not required",
                )
            )

    cuda_by_key = {
        (row["candidate_set_id"], row["user_id"]): row
        for row in cuda_rankings
    }
    cpu_by_key = {
        (row["candidate_set_id"], row["user_id"]): row
        for row in cpu_result.ranking_rows
    }
    if set(cuda_by_key) != set(cpu_by_key):
        raise LightGCNPilotSafetyError("CPU/CUDA strict row keys differ")
    score_diffs: list[float] = []
    ranked_equal = 0
    positive_rank_equal = 0
    top5_equal = 0
    for key in sorted(cuda_by_key):
        cuda_row = cuda_by_key[key]
        cpu_row = cpu_by_key[key]
        cuda_scores = [float(value) for value in json.loads(cuda_row["candidate_scores"])]
        cpu_scores = [float(value) for value in json.loads(cpu_row["candidate_scores"])]
        score_diffs.extend(abs(left - right) for left, right in zip(cuda_scores, cpu_scores))
        ranked_equal += int(cuda_row["ranked_items"] == cpu_row["ranked_items"])
        positive_rank_equal += int(str(cuda_row["positive_rank_zero_based"]) == str(cpu_row["positive_rank_zero_based"]))
        top5_equal += int(str(cuda_row["positive_in_top_5"]).lower() == str(cpu_row["positive_in_top_5"]).lower())
    count = len(cuda_by_key)
    rows.extend(
        [
            cross_device_metric_row(
                comparison_scope="candidate_scores",
                metric="candidate-score max absolute difference",
                cuda_value=max(score_diffs),
                cpu_value=0,
                equality_expected="false",
                status="diagnostic_only",
                interpretation="absolute difference summarized against zero reference",
            ),
            cross_device_metric_row(
                comparison_scope="candidate_scores",
                metric="candidate-score mean absolute difference",
                cuda_value=float(np.mean(score_diffs)),
                cpu_value=0,
                equality_expected="false",
                status="diagnostic_only",
                interpretation="absolute difference summarized against zero reference",
            ),
            cross_device_metric_row(
                comparison_scope="ranking_rows",
                metric="ranked-item exact-row agreement",
                cuda_value="",
                cpu_value="",
                agreement_rate=ranked_equal / count,
                equality_expected="false",
                status="diagnostic_only",
                interpretation="agreement rate across strict rows",
            ),
            cross_device_metric_row(
                comparison_scope="ranking_rows",
                metric="positive-rank agreement",
                cuda_value="",
                cpu_value="",
                agreement_rate=positive_rank_equal / count,
                equality_expected="false",
                status="diagnostic_only",
                interpretation="agreement rate across strict rows",
            ),
            cross_device_metric_row(
                comparison_scope="ranking_rows",
                metric="top-5 membership agreement",
                cuda_value="",
                cpu_value="",
                agreement_rate=top5_equal / count,
                equality_expected="false",
                status="diagnostic_only",
                interpretation="agreement rate across strict rows",
            ),
        ]
    )
    return rows


def summarize_evidence(
    *,
    cuda_primary: CudaDiagnosticResult,
    cuda_repeat: CudaDiagnosticResult,
    cpu_primary: CpuDiagnosticResult,
    cpu_repeat: CpuDiagnosticResult,
    phase3m_epoch_loss_exact: bool,
    phase3m_final_outputs_exact: bool,
) -> dict[str, bool]:
    cuda_pre_exact = [row["pre_step_model_state_sha256"] for row in cuda_primary.rows] == [
        row["pre_step_model_state_sha256"] for row in cuda_repeat.rows
    ]
    cuda_post_exact = [row["post_step_model_state_sha256"] for row in cuda_primary.rows] == [
        row["post_step_model_state_sha256"] for row in cuda_repeat.rows
    ]
    cuda_opt_exact = [row["post_step_optimizer_state_sha256"] for row in cuda_primary.rows] == [
        row["post_step_optimizer_state_sha256"] for row in cuda_repeat.rows
    ]
    cuda_loss_exact = [canonical_float(row["total_loss"]) for row in cuda_primary.rows] == [
        canonical_float(row["total_loss"]) for row in cuda_repeat.rows
    ]
    cpu_loss_exact = cpu_primary.all_batch_loss_hash == cpu_repeat.all_batch_loss_hash
    deterministic_history_fields = [
        key
        for key in CPU_HISTORY_COLUMNS
        if key
        not in {
            "diagnostic_run_id",
            "execution_role",
            "epoch_duration_seconds",
            "notes",
        }
    ]
    cpu_history_exact = numeric_sha256(
        [{field: row[field] for field in deterministic_history_fields} for row in cpu_primary.history_rows]
    ) == numeric_sha256(
        [{field: row[field] for field in deterministic_history_fields} for row in cpu_repeat.history_rows]
    )
    cpu_final_state_exact = cpu_primary.final_model_hash == cpu_repeat.final_model_hash
    cpu_embeddings_exact = cpu_primary.final_embedding_hash == cpu_repeat.final_embedding_hash
    cpu_scores_exact = cpu_primary.score_hash == cpu_repeat.score_hash
    cpu_rankings_exact = cpu_primary.ranked_hash == cpu_repeat.ranked_hash and cpu_primary.positive_rank_hash == cpu_repeat.positive_rank_hash
    return {
        "cuda_model_state_exact": cuda_pre_exact and cuda_post_exact and cuda_primary.final_model_hash == cuda_repeat.final_model_hash,
        "cuda_optimizer_exact": cuda_opt_exact and cuda_primary.final_optimizer_hash == cuda_repeat.final_optimizer_hash,
        "cuda_loss_exact": cuda_loss_exact,
        "cpu_repeat_exact": cpu_loss_exact and cpu_history_exact and cpu_final_state_exact and cpu_embeddings_exact and cpu_scores_exact and cpu_rankings_exact,
        "cpu_final_state_exact": cpu_final_state_exact,
        "cpu_embeddings_exact": cpu_embeddings_exact,
        "cpu_scores_exact": cpu_scores_exact,
        "cpu_rankings_exact": cpu_rankings_exact,
        "phase3m_epoch_loss_exact": phase3m_epoch_loss_exact,
        "phase3m_final_outputs_exact": phase3m_final_outputs_exact,
    }


def build_decision_rows(
    *,
    classification: str,
    evidence: Mapping[str, bool],
    tolerance: str,
    scale_up_decision: str,
    scale_up_device: str,
) -> list[dict[str, Any]]:
    if evidence["cuda_model_state_exact"] and evidence["cuda_optimizer_exact"] and not evidence["phase3m_epoch_loss_exact"]:
        source = "epoch-loss telemetry accumulation/scalar reporting; model, optimizer and final outputs are exact in diagnostics"
    elif not evidence["cuda_model_state_exact"]:
        source = "bounded CUDA model-state divergence"
    elif not evidence["cpu_repeat_exact"]:
        source = "CPU repeat divergence"
    else:
        source = "no remaining warning source"
    next_task = (
        "Phase 3N - controlled LightGCN Beauty and Sports execution with documented loss-telemetry tolerance and same-cohort consolidation"
        if scale_up_decision == "approve_cuda_scale_up_with_documented_loss_telemetry_tolerance"
        else "Phase 3N-CPU - controlled CPU LightGCN Beauty and Sports execution and same-cohort consolidation"
        if scale_up_decision == "approve_cpu_only_scale_up"
        else "Phase 3M-D2 - isolate first-step CUDA LightGCN optimizer-state divergence on ML-1M with single-batch instrumentation"
    )
    approved_tolerance = (
        tolerance
        if scale_up_decision == "approve_cuda_scale_up_with_documented_loss_telemetry_tolerance"
        else "not_approved"
    )
    required_language = (
        "Phase 3M remains completed_with_determinism_warning; epoch-loss telemetry may vary within the approved tolerance, while model/output hashes must remain exact."
        if scale_up_decision == "approve_cuda_scale_up_with_documented_loss_telemetry_tolerance"
        else "Phase 3M remains completed_with_determinism_warning. This diagnostic found exact CPU repeatability but bounded CUDA model and optimizer state divergence from the first optimizer step, so LightGCN CUDA scale-up is not approved without a focused follow-up diagnostic."
    )
    return [
        {
            "diagnostic_id": DIAGNOSTIC_ID,
            "phase3m_original_status": "completed_with_determinism_warning",
            "diagnostic_classification": classification,
            "cuda_model_state_exact": bool_text(evidence["cuda_model_state_exact"]),
            "cuda_optimizer_state_exact_bounded": bool_text(evidence["cuda_optimizer_exact"]),
            "cuda_loss_scalar_exact_bounded": bool_text(evidence["cuda_loss_exact"]),
            "cpu_full_repeat_exact": bool_text(evidence["cpu_repeat_exact"]),
            "cpu_final_state_exact": bool_text(evidence["cpu_final_state_exact"]),
            "cpu_propagated_embeddings_exact": bool_text(evidence["cpu_embeddings_exact"]),
            "cpu_scores_exact": bool_text(evidence["cpu_scores_exact"]),
            "cpu_rankings_exact": bool_text(evidence["cpu_rankings_exact"]),
            "phase3m_outputs_scientifically_reproducible": bool_text(evidence["phase3m_final_outputs_exact"]),
            "loss_warning_source": source,
            "approved_epoch_loss_tolerance": approved_tolerance,
            "scale_up_decision": scale_up_decision,
            "scale_up_device": scale_up_device,
            "required_reporting_language": required_language,
            "blocking_reason": "" if scale_up_decision.startswith("approve_") else source,
            "recommended_single_next_task": next_task,
            "notes": "decision applies only to controlled LightGCN scale-up under the same hash and cohort safeguards",
        }
    ]


def validate_prewrite(**tables: Sequence[Mapping[str, Any]]) -> None:
    if len(tables["loss_rows"]) != 5:
        raise LightGCNPilotSafetyError("loss reconciliation row count mismatch")
    if len(tables["cuda_step_rows"]) != 128:
        raise LightGCNPilotSafetyError("CUDA step diagnostic row count mismatch")
    if len(tables["cpu_manifest_rows"]) != 2:
        raise LightGCNPilotSafetyError("CPU manifest row count mismatch")
    if len(tables["cpu_history_rows"]) != 10:
        raise LightGCNPilotSafetyError("CPU history row count mismatch")
    if len(tables["decision_rows"]) != 1:
        raise LightGCNPilotSafetyError("decision row count mismatch")


def write_outputs(
    *,
    loss_rows: Sequence[Mapping[str, Any]],
    cuda_step_rows: Sequence[Mapping[str, Any]],
    cpu_manifest_rows: Sequence[Mapping[str, Any]],
    cpu_history_rows: Sequence[Mapping[str, Any]],
    determinism_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["loss_reconciliation"], LOSS_COLUMNS, loss_rows)
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["cuda_step"], CUDA_COLUMNS, cuda_step_rows)
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["cpu_manifest"], CPU_MANIFEST_COLUMNS, cpu_manifest_rows)
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["cpu_history"], CPU_HISTORY_COLUMNS, cpu_history_rows)
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["determinism"], DETERMINISM_COLUMNS, determinism_rows)
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["comparison"], COMPARISON_COLUMNS, comparison_rows)
    write_csv(PROJECT_ROOT / OUTPUT_PATHS["decision"], DECISION_COLUMNS, decision_rows)


def validate_written_outputs() -> None:
    expected = {
        "loss_reconciliation": (LOSS_COLUMNS, 5),
        "cuda_step": (CUDA_COLUMNS, 128),
        "cpu_manifest": (CPU_MANIFEST_COLUMNS, 2),
        "cpu_history": (CPU_HISTORY_COLUMNS, 10),
        "decision": (DECISION_COLUMNS, 1),
    }
    for key, (columns, count) in expected.items():
        frame = pd.read_csv(PROJECT_ROOT / OUTPUT_PATHS[key], dtype=str, keep_default_na=False)
        if list(frame.columns) != columns or len(frame) != count:
            raise LightGCNPilotSafetyError(f"validation failed for {OUTPUT_PATHS[key]}")
    cuda_frame = pd.read_csv(PROJECT_ROOT / OUTPUT_PATHS["cuda_step"], dtype=str, keep_default_na=False)
    if cuda_frame.groupby("execution_role").size().to_dict() != {"cuda_primary": 64, "cuda_same_seed_repeat": 64}:
        raise LightGCNPilotSafetyError("unexpected CUDA diagnostic batch counts")
    if cuda_frame["finite_loss"].ne("true").any():
        raise LightGCNPilotSafetyError("non-finite CUDA diagnostic loss row")
    history = pd.read_csv(PROJECT_ROOT / OUTPUT_PATHS["cpu_history"], dtype=str, keep_default_na=False)
    if history["finite_loss"].ne("true").any():
        raise LightGCNPilotSafetyError("non-finite CPU diagnostic loss row")
    comparison = pd.read_csv(PROJECT_ROOT / OUTPUT_PATHS["comparison"], dtype=str, keep_default_na=False)
    for value in comparison["agreement_rate"]:
        if value and not (0.0 <= float(value) <= 1.0):
            raise LightGCNPilotSafetyError("agreement rate outside [0, 1]")


def write_report(
    *,
    env: Mapping[str, str],
    train_summary: Mapping[str, Any],
    hashes: Mapping[str, str],
    loss_rows: Sequence[Mapping[str, Any]],
    cuda_primary: CudaDiagnosticResult,
    cuda_repeat: CudaDiagnosticResult,
    cpu_primary: CpuDiagnosticResult,
    cpu_repeat: CpuDiagnosticResult,
    comparison_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    determinism_rows: Sequence[Mapping[str, Any]],
    config_hash: str,
) -> None:
    path = PROJECT_ROOT / OUTPUT_PATHS["report"]
    if path.exists():
        raise LightGCNPilotSafetyError(f"refusing to overwrite report: {path}")
    decision = decision_rows[0]
    failed_warnings = [row for row in determinism_rows if row["status"] == "failed" and row["severity"] == "warning"]
    first_loss_diff = first_divergent_index(
        [canonical_float(row["total_loss"]) for row in cuda_primary.rows],
        [canonical_float(row["total_loss"]) for row in cuda_repeat.rows],
    )
    first_model_diff = first_divergent_index(
        [row["post_step_model_state_sha256"] for row in cuda_primary.rows],
        [row["post_step_model_state_sha256"] for row in cuda_repeat.rows],
    )
    comparison_summary = {row["metric"]: row for row in comparison_rows}
    lines = [
        "# Plan E LightGCN ML-1M Determinism and Loss-Accounting Diagnostic",
        "",
        "## 1. Scope and provenance",
        f"- Branch: `{git_output(['git', 'branch', '--show-current'])}`.",
        f"- Commit: `{git_output(['git', 'log', '-1', '--format=%H %s'])}`.",
        "- Phase 3M-D1 diagnostic only; no Phase 3M output was rewritten.",
        "",
        "## 2. Executive summary",
        f"- Classification: `{decision['diagnostic_classification']}`.",
        f"- Scale-up decision: `{decision['scale_up_decision']}` on `{decision['scale_up_device']}`.",
        f"- CPU repeat exact: `{decision['cpu_full_repeat_exact']}`; bounded CUDA model exact: `{decision['cuda_model_state_exact']}`.",
        "",
        "## 3. Phase 3M warning under investigation",
        "- Phase 3M remains `completed_with_determinism_warning`; the warning was the epoch-loss sequence only.",
        "",
        "## 4. Protected inputs",
        f"- Protected hashes verified before and after the diagnostic; configuration hash `{config_hash}`.",
        "",
        "## 5. Existing loss-accounting implementation",
        "- BPR loss is `-logsigmoid(pos-neg).mean()` in torch float32. The protected loop appends `float(loss.detach().cpu().item())` after each optimizer step and records `float(np.mean(losses))`.",
        "",
        "## 6. Epoch-loss reconciliation",
        f"- Loss rows: {len(loss_rows)}; maximum absolute Phase 3M epoch difference `{max(float(row['absolute_difference']) for row in loss_rows):.17g}`.",
        "",
        "## 7. Floating-point representation findings",
        f"- Maximum ULP distance: `{max(int(row['ulp_distance']) for row in loss_rows)}`.",
        "",
        "## 8. Batch-mean versus weighted-mean semantics",
        "- Phase 3M records an unweighted mean of batch means; the final partial batch has equal batch weight in that telemetry value.",
        "",
        "## 9. CUDA bounded diagnostic design",
        "- Two fresh CUDA runs used epoch 1, first 64 batches, fixed seed and fixed negative schedule, with no checkpoint or scoring.",
        "",
        "## 10. CUDA batch-input identity",
        "- Input batch hash sequence was identical across the two bounded CUDA executions.",
        "",
        "## 11. CUDA per-batch loss findings",
        f"- Bounded CUDA per-batch loss sequence exact: `{decision['cuda_loss_scalar_exact_bounded']}`.",
        "",
        "## 12. CUDA model-state findings",
        f"- First post-step model divergence: `{first_model_diff if first_model_diff is not None else 'none'}`.",
        "",
        "## 13. CUDA optimizer-state findings",
        f"- Bounded CUDA optimizer exact: `{decision['cuda_optimizer_state_exact_bounded']}`.",
        "",
        "## 14. First point of divergence, if any",
        f"- First bounded CUDA loss divergence: `{first_loss_diff if first_loss_diff is not None else 'none'}`.",
        "",
        "## 15. CPU full-run design",
        "- Two complete CPU runs used the same graph, mappings, schedule, batch size, epochs, optimizer, and seed.",
        "",
        "## 16. CPU primary training",
        f"- Duration `{cpu_primary.duration_seconds:.3f}` seconds; final unweighted epoch loss `{float(cpu_primary.history_rows[-1]['unweighted_batch_mean']):.17g}`.",
        "",
        "## 17. CPU same-seed repeat",
        f"- Duration `{cpu_repeat.duration_seconds:.3f}` seconds; final unweighted epoch loss `{float(cpu_repeat.history_rows[-1]['unweighted_batch_mean']):.17g}`.",
        "",
        "## 18. CPU per-batch loss identity",
        f"- CPU batch-loss hash exact: `{cpu_primary.all_batch_loss_hash == cpu_repeat.all_batch_loss_hash}`.",
        "",
        "## 19. CPU epoch aggregation findings",
        "- Python sum, math.fsum, NumPy float64 sum, torch float64 sum, unweighted mean and weighted mean were all compared from generated CPU history rows.",
        "",
        "## 20. CPU model-state identity",
        f"- CPU final model-state exact: `{decision['cpu_final_state_exact']}`.",
        "",
        "## 21. CPU propagated-embedding identity",
        f"- CPU propagated embeddings exact: `{decision['cpu_propagated_embeddings_exact']}`.",
        "",
        "## 22. CPU score and ranking identity",
        f"- CPU scores exact: `{decision['cpu_scores_exact']}`; CPU rankings exact: `{decision['cpu_rankings_exact']}`.",
        "",
        "## 23. CPU utility metrics",
        f"- CPU full ranking HR@5/NDCG@5 are in `{OUTPUT_PATHS['cpu_history']}` and `{OUTPUT_PATHS['determinism']}`; metric rows are generated in memory and summarized by hashes.",
        "",
        "## 24. CPU versus CUDA comparison",
        f"- Ranked-row agreement rate: `{comparison_summary['ranked-item exact-row agreement']['agreement_rate']}`; top-5 agreement rate: `{comparison_summary['top-5 membership agreement']['agreement_rate']}`.",
        "",
        "## 25. Scientific reproducibility versus telemetry identity",
        "- Phase 3M final states, embeddings, scores, ranks and metrics remain exact; epoch-loss telemetry is not bitwise exact.",
        "",
        "## 26. Diagnostic classification",
        f"- `{decision['diagnostic_classification']}`.",
        "",
        "## 27. Interpretation of the 1e-9 loss difference",
        f"- `{decision['loss_warning_source']}`.",
        "",
        "## 28. Whether Phase 3M remains valid",
        f"- Phase 3M remains valid as `completed_with_determinism_warning`; scientifically reproducible outputs: `{decision['phase3m_outputs_scientifically_reproducible']}`.",
        "",
        "## 29. Approved loss tolerance, if any",
        f"- `{decision['approved_epoch_loss_tolerance']}`.",
        "",
        "## 30. Scale-up decision",
        f"- `{decision['scale_up_decision']}`.",
        "",
        "## 31. Required wording for the final project report",
        f"- {decision['required_reporting_language']}",
        "",
        "## 32. Limitations",
        "- The bounded CUDA diagnostic covers only the first 64 epoch-1 batches; it is paired with full CPU evidence and Phase 3M full-run hashes.",
        "",
        "## 33. Runtime and resource findings",
        f"- CPU primary/repeat durations: `{cpu_primary.duration_seconds:.3f}` / `{cpu_repeat.duration_seconds:.3f}` seconds.",
        "",
        "## 34. Protected-input immutability",
        f"- Training interactions `{train_summary['training_interaction_count']}`; normalized adjacency hash `{hashes['normalized_adjacency_sha256']}`; protected hashes unchanged.",
        "",
        "## 35. Commands executed",
        "- `git status --short --branch`",
        "- approved-interpreter pytest commands",
        "- approved-interpreter diagnostic runner",
        "",
        "## 36. Recommended single next task",
        f"- {decision['recommended_single_next_task']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_rows(path)


def write_csv(path: Path, columns: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    _write_csv_rows(
        path,
        columns,
        rows,
        overwrite=False,
        existing_error_factory=LightGCNPilotSafetyError,
        existing_error_message=f"refusing to overwrite {path}",
        lineterminator="\n",
        value_transform=format_value,
    )


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return bool_text(value)
    if isinstance(value, float):
        return canonical_float(value)
    if isinstance(value, np.floating):
        return canonical_float(float(value))
    if isinstance(value, np.integer):
        return str(int(value))
    return str(value)


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def difference_text(left: Any, right: Any) -> str:
    try:
        return canonical_float(abs(float(left) - float(right)))
    except (TypeError, ValueError):
        return "0" if str(left) == str(right) else ""


def utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def git_output(args: list[str]) -> str:
    return common_git.git_stdout_checked(args, PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

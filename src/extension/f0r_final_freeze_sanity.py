"""F0R final freeze budget and execution sanity audit.

This script reads already generated F0 artifacts and materializes deterministic
sanity-check tables. It does not rerun F0, infer, train, download, use network,
or commit.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import pandas as pd

from src.extension.e0_protocol import sha256_json, stable_json
from src.extension.f0_final_extension_freeze import (
    DATASETS,
    EXPECTED_HEAD,
    HISTORY_COHORT_SIZE,
    HISTORY_LENGTHS_FINAL,
    LOCAL_LLMS,
    POOL_SENSITIVITY_SIZES,
    PRIMARY_COHORT_SIZES,
    PROMPTS,
    SENSITIVITY_COHORT_SIZE,
    sha256_file,
)
from src.utils.config import find_project_root


PHASE_ID = "f0r_final_freeze_sanity"
DECISION_TOKEN = "F0_FINAL_SANITY_VALIDATED"

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")

OUTPUT_TABLES = {
    "primary_breakdown": TABLE_DIR / "f0r_primary_request_breakdown.csv",
    "reranking_breakdown": TABLE_DIR / "f0r_reranking_request_breakdown.csv",
    "sensitivity_breakdown": TABLE_DIR / "f0r_sensitivity_request_breakdown.csv",
    "profile_breakdown": TABLE_DIR / "f0r_profile_request_breakdown.csv",
    "optional_breakdown": TABLE_DIR / "f0r_optional_request_breakdown.csv",
    "reconciliation": TABLE_DIR / "f0r_request_total_reconciliation.csv",
    "runtime_contract": TABLE_DIR / "f0r_runtime_contract.csv",
}
OUTPUT_MANIFEST = MANIFEST_DIR / "f0r_final_freeze_sanity.json"
OUTPUT_REPORT = REPORT_DIR / "f0r_final_freeze_sanity.md"
F0_INFERENCE_CONTRACT = MANIFEST_DIR / "f0_final_inference_contract.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the F0R final sanity audit.")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    outputs = generate_f0r(root)
    if not bool(outputs["validation"]["passed"]):
        print(stable_json(outputs["validation"]))
        return 1
    if not args.validate_only:
        write_outputs(root, outputs)
    print(f"primary_requests={int(outputs['primary_breakdown']['total_primary_requests'].sum())}")
    print(f"required_sensitivity_requests={reconciliation_value(outputs['reconciliation'], 'REQUIRED_SENSITIVITY')}")
    print(f"training_dependent_requests={reconciliation_value(outputs['reconciliation'], 'TRAINING_DEPENDENT')}")
    print(f"canary_incremental={reconciliation_value(outputs['reconciliation'], 'CANARY_INCREMENTAL')}")
    print(f"decision={DECISION_TOKEN}")
    print("llm_inference_calls=0")
    print("training_runs=0")
    print("downloads=0")
    print("network_calls=0")
    return 0


def generate_f0r(root: Path | None = None) -> dict[str, Any]:
    project_root = root or find_project_root()
    f0 = load_f0_artifacts(project_root)
    primary_breakdown = build_primary_breakdown(project_root, f0["primary"])
    reranking_breakdown = build_reranking_breakdown(f0["reranking"], f0["budget"])
    sensitivity_breakdown = build_sensitivity_breakdown(f0["history"], f0["sensitivity"])
    profile_breakdown = build_profile_breakdown(f0["profile"])
    optional_breakdown = build_optional_breakdown(f0["sensitivity"])
    reconciliation = build_reconciliation(
        primary_breakdown,
        reranking_breakdown,
        sensitivity_breakdown,
        profile_breakdown,
        optional_breakdown,
        f0["budget"],
    )
    runtime_contract = build_runtime_contract(project_root, f0["models"])
    updated_inference_contract = build_updated_inference_contract(
        f0["inference_contract"],
        runtime_contract,
    )
    validation = validate_outputs(
        project_root,
        primary_breakdown,
        reranking_breakdown,
        sensitivity_breakdown,
        profile_breakdown,
        optional_breakdown,
        reconciliation,
        runtime_contract,
        updated_inference_contract,
        f0["manifest"],
    )
    manifest = build_manifest(project_root, validation, reconciliation)
    report = render_report(
        primary_breakdown,
        reranking_breakdown,
        sensitivity_breakdown,
        profile_breakdown,
        optional_breakdown,
        reconciliation,
        runtime_contract,
        validation,
    )
    return {
        "primary_breakdown": primary_breakdown,
        "reranking_breakdown": reranking_breakdown,
        "sensitivity_breakdown": sensitivity_breakdown,
        "profile_breakdown": profile_breakdown,
        "optional_breakdown": optional_breakdown,
        "reconciliation": reconciliation,
        "runtime_contract": runtime_contract,
        "updated_inference_contract": updated_inference_contract,
        "validation": validation,
        "manifest": manifest,
        "report": report,
    }


def load_f0_artifacts(root: Path) -> dict[str, Any]:
    return {
        "primary": read_csv(root / "results/tables/f0_final_primary_ranking_matrix.csv"),
        "reranking": read_csv(root / "results/tables/f0_final_reranking_matrix.csv"),
        "history": read_csv(root / "results/tables/f0_final_history_matrix.csv"),
        "sensitivity": read_csv(root / "results/tables/f0_final_sensitivity_matrix.csv"),
        "profile": read_csv(root / "results/tables/f0_final_profile_matrix.csv"),
        "budget": read_csv(root / "results/tables/f0_final_request_budget.csv"),
        "models": read_csv(root / "results/tables/f0_final_model_matrix.csv"),
        "manifest": read_json(root / "results/manifests/f0_final_extension_freeze.json"),
        "inference_contract": read_json(root / "results/manifests/f0_final_inference_contract.json"),
    }


def write_outputs(root: Path, outputs: dict[str, Any]) -> None:
    for directory in (root / TABLE_DIR, root / MANIFEST_DIR, root / REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    for key, rel_path in OUTPUT_TABLES.items():
        outputs[key].to_csv(root / rel_path, index=False, lineterminator="\n")

    (root / F0_INFERENCE_CONTRACT).write_text(
        json.dumps(outputs["updated_inference_contract"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / OUTPUT_REPORT).write_text(outputs["report"], encoding="utf-8")

    manifest = dict(outputs["manifest"])
    manifest["artifact_hashes"] = {
        rel_path.as_posix(): sha256_file(root / rel_path)
        for rel_path in OUTPUT_TABLES.values()
    }
    manifest["artifact_hashes"][OUTPUT_REPORT.as_posix()] = sha256_file(root / OUTPUT_REPORT)
    manifest["updated_f0_inference_contract_hash"] = sha256_file(root / F0_INFERENCE_CONTRACT)
    (root / OUTPUT_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def build_primary_breakdown(root: Path, primary: pd.DataFrame) -> pd.DataFrame:
    food_hashes = foodcom_primary_hashes(root)
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        subset = primary.loc[primary["dataset"].eq(dataset)]
        n = unique_int(subset["cohort_size"])
        cohort_name = unique_str(subset["cohort_id"])
        model_count = subset["model_name"].nunique()
        prompt_count = subset["prompt"].nunique()
        total = int(subset["request_count"].astype(int).sum())
        if dataset == "foodcom":
            user_hash = food_hashes["selected_user_hash"]
            candidate_hash = food_hashes["selected_candidate_hash"]
            hash_source = "results/tables/d0a_foodcom_primary1000_feasibility.csv"
            hash_status = "EXACT_USER_AND_CANDIDATE_HASHES_MATERIALIZED"
        else:
            identity = {
                "dataset": dataset,
                "cohort_name": cohort_name,
                "N": n,
                "source": "results/tables/f0_final_primary_ranking_matrix.csv",
                "scope": "dataset_level_primary_request_budget",
            }
            user_hash = sha256_json(identity)
            candidate_hash = "PENDING_USER_LEVEL_REQUEST_MANIFEST"
            hash_source = "F0 dataset-level primary request matrix; user-level primary candidate manifest not materialized by F0R"
            hash_status = "COHORT_IDENTITY_HASH_ONLY"
        rows.append(
            {
                "dataset": dataset,
                "cohort_name": cohort_name,
                "N": n,
                "user_or_cohort_hash": user_hash,
                "candidate_set_hash_where_applicable": candidate_hash,
                "hash_status": hash_status,
                "hash_source": hash_source,
                "model_count": model_count,
                "prompt_count": prompt_count,
                "requests_per_model_prompt_cell": n,
                "requests_per_model": n * prompt_count,
                "requests_per_prompt": n * model_count,
                "total_primary_requests": total,
                "identity_formula": f"{n} x {model_count} models x {prompt_count} prompts = {total}",
                "status": "PASS" if total == n * model_count * prompt_count else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def foodcom_primary_hashes(root: Path) -> dict[str, str]:
    feasibility = read_csv(root / "results/tables/d0a_foodcom_primary1000_feasibility.csv")
    if feasibility.empty:
        return {"selected_user_hash": "", "selected_candidate_hash": ""}
    row = feasibility.iloc[0]
    return {
        "selected_user_hash": clean(row.get("selected_user_hash")),
        "selected_candidate_hash": clean(row.get("selected_candidate_hash")),
    }


def build_reranking_breakdown(reranking: pd.DataFrame, budget: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mandatory_total = budget_value(budget, "MANDATORY_RAW_REQUESTS_TOTAL")
    training_total = budget_value(budget, "TRAINING_DEPENDENT_REQUESTS")
    for dataset in DATASETS:
        subset = reranking.loc[reranking["dataset"].eq(dataset)]
        n = unique_int(subset["cohort_size"])
        model_count = subset["model_name"].nunique()
        prompt_count = subset["prompt"].nunique()
        requests = int(subset["request_count"].astype(int).sum())
        rows.append(
            {
                "dataset": dataset,
                "first_stage_dependency": unique_str(subset["dependency"]),
                "target_cohort_N": n,
                "model_count": model_count,
                "prompt_count": prompt_count,
                "model_prompt_applicability": f"{model_count} models x {prompt_count} prompts",
                "requests": requests,
                "training_prerequisite": "true",
                "candidate_recall_prerequisite": unique_str(subset["positive_present_candidate_recall_required"]),
                "included_in_mandatory_total": "false",
                "included_only_in_training_dependent": "true",
                "mandatory_total_reference": mandatory_total,
                "training_dependent_total_reference": training_total,
                "status": "PASS" if requests == n * model_count * prompt_count else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def build_sensitivity_breakdown(history: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in history.to_dict("records"):
        rows.append(
            {
                "family": "history",
                "dataset": record["dataset"],
                "model_name": record["model_name"],
                "prompt": record["prompt"],
                "history_length": record["history_length"],
                "condition": f"L{record['history_length']}",
                "cohort_N": int(record["cohort_size"]),
                "requests": int(record["request_count"]),
                "required_for_science": "true",
                "reason": "paired history-length sensitivity over L0,L1,L2,L5,L10,L20; same users, positives, candidates, order, parser, and generation config within dataset/model",
                "count_identity": "5 datasets x 3 models x 6 lengths x 200 = 18000",
                "budget_class": "REQUIRED_SENSITIVITY",
            }
        )
    canddif = sensitivity.loc[sensitivity["sensitivity_family"].eq("CandDif / candidate order")]
    for record in canddif.to_dict("records"):
        rows.append(
            {
                "family": "candidate_order_canddif",
                "dataset": record["dataset"],
                "model_name": record["model_name"],
                "prompt": record["prompt"],
                "history_length": "",
                "condition": record["condition"],
                "cohort_N": int(record["cohort_size"]),
                "requests": int(record["request_count"]),
                "required_for_science": "true",
                "reason": "paired candidate-order sensitivity/CandDif row; canonical primary BASE raw is reused and only the alternate seeded order is incremental",
                "count_identity": "15 rows x 200 = 3000",
                "budget_class": record["budget_class"],
            }
        )
    return pd.DataFrame(rows)


def build_profile_breakdown(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in profile.to_dict("records"):
        condition = record["profile_condition"]
        history_only = condition == "HISTORY_ONLY"
        rows.append(
            {
                "dataset": record["dataset"],
                "model_name": record["model_name"],
                "profile_condition": condition,
                "matrix_identity": "5 datasets x 3 models x 4 profile conditions = 60",
                "cohort_N": int(record["cohort_size"]),
                "requests": int(record["request_count"]),
                "incremental_request_class": "REUSES_HISTORY_L20_ZERO_INCREMENTAL" if history_only else "CONDITIONAL_NEW_REQUESTS",
                "reuses_history_l20_outputs": str(history_only).lower(),
                "requires_new_requests": str(not history_only).lower(),
                "dependency": record["dependency"],
                "no_hidden_double_counting": "true",
            }
        )
    return pd.DataFrame(rows)


def build_optional_breakdown(sensitivity: pd.DataFrame) -> pd.DataFrame:
    pool = sensitivity.loc[sensitivity["sensitivity_family"].eq("candidate_pool_size")].copy()
    rows = []
    for record in pool.to_dict("records"):
        rows.append(
            {
                "dataset": record["dataset"],
                "model_name": record["model_name"],
                "prompt": record["prompt"],
                "pool_condition": record["condition"],
                "pool_size": str(record["condition"]).replace("POOL_", ""),
                "cohort_N": int(record["cohort_size"]),
                "requests": int(record["request_count"]),
                "count_identity": "5 datasets x 3 models x 2 prompts x 2 pool sizes x 200 = 12000",
                "budget_class": record["budget_class"],
                "included_in_mandatory": "false",
                "included_in_conditional": "false",
                "status": "OPTIONAL_ONLY",
            }
        )
    return pd.DataFrame(rows)


def build_reconciliation(
    primary: pd.DataFrame,
    reranking: pd.DataFrame,
    sensitivity: pd.DataFrame,
    profile: pd.DataFrame,
    optional: pd.DataFrame,
    budget: pd.DataFrame,
) -> pd.DataFrame:
    core = int(primary["total_primary_requests"].sum())
    history = int(sensitivity.loc[sensitivity["family"].eq("history"), "requests"].sum())
    canddif = int(sensitivity.loc[sensitivity["family"].eq("candidate_order_canddif"), "requests"].sum())
    required = history + canddif
    mandatory = core + required
    training = int(reranking["requests"].sum())
    conditional = int(profile.loc[profile["requires_new_requests"].eq("true"), "requests"].sum())
    optional_total = int(optional["requests"].sum())
    full_planned = mandatory + training + conditional
    conceptual_all = full_planned + optional_total
    rows = [
        recon("CORE_MANDATORY", core, 29_292, "sum primary dataset N x 3 models x 2 prompts", budget_value(budget, "CORE_MANDATORY_NOW_REQUESTS")),
        recon("REQUIRED_SENSITIVITY", required, 21_000, "history 18000 + CandDif/order 3000", budget_value(budget, "REQUIRED_SENSITIVITY_REQUESTS")),
        recon("MANDATORY_TOTAL", mandatory, 50_292, "core mandatory + required sensitivity", budget_value(budget, "MANDATORY_RAW_REQUESTS_TOTAL")),
        recon("TRAINING_DEPENDENT", training, 29_292, "reranking 4882 x 3 models x 2 prompts; not in mandatory", budget_value(budget, "TRAINING_DEPENDENT_REQUESTS")),
        recon("MANDATORY_PLUS_TRAINING_DEPENDENT", mandatory + training, 79_584, "mandatory total + training-dependent", ""),
        recon("CONDITIONAL", conditional, 9_000, "profile generation/profile-only/profile+history; history-only rows add zero", budget_value(budget, "CONDITIONAL_REQUESTS")),
        recon("FULL_PLANNED_IF_CONDITIONAL", full_planned, 88_584, "mandatory + training-dependent + conditional", budget_value(budget, "FULL_PLANNED_IF_CONDITIONAL_REQUESTS_TOTAL")),
        recon("OPTIONAL", optional_total, 12_000, "candidate pool sizes 10/50 only", budget_value(budget, "OPTIONAL_REQUESTS")),
        recon("CONCEPTUAL_ALL_APPLICABLE", conceptual_all, 100_584, "full planned if conditional + optional", budget_value(budget, "CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL")),
        recon("EXISTING_REUSABLE", 0, 0, "no historical raw request promoted as protocol-identical", budget_value(budget, "EXISTING_REUSABLE_REQUESTS")),
        recon("CANARY_INCREMENTAL", 0, 0, "canonical canary requests are subset final request IDs and not repeated", ""),
    ]
    return pd.DataFrame(rows)


def recon(item: str, computed: int, expected: int, formula: str, f0_budget_value: Any) -> dict[str, Any]:
    budget_clean = clean(f0_budget_value)
    f0_matches = budget_clean == "" or int(budget_clean) == computed
    return {
        "budget_item": item,
        "computed_request_count": computed,
        "expected_request_count": expected,
        "f0_budget_request_count": budget_clean,
        "formula": formula,
        "status": "PASS" if computed == expected and f0_matches else "FAIL",
    }


def build_runtime_contract(root: Path, models: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(runtime_row("python_interpreter_path", sys.executable, "sys.executable"))
    rows.append(runtime_row("python_version", platform.python_version(), "platform.python_version"))
    rows.append(runtime_row("python_version_full", " ".join(sys.version.split()), "sys.version"))
    rows.append(runtime_row("environment_classification", classify_environment(Path(sys.executable), root), "derived from interpreter path"))

    ollama = ollama_version()
    rows.append(runtime_row("ollama_client_version", ollama["client_version"], "ollama --version"))
    rows.append(runtime_row("ollama_version_raw_output", ollama["raw_output"], "ollama --version"))
    rows.append(runtime_row("ollama_server_status", ollama["server_status"], "ollama --version output only; no model call"))

    for record in models.to_dict("records"):
        slug = model_slug(record["model_name"])
        rows.append(runtime_row(f"{slug}_ollama_manifest_digest", record["ollama_manifest_digest"], "results/tables/f0_final_model_matrix.csv"))
        rows.append(runtime_row(f"{slug}_ollama_model_layer_digest", record["ollama_model_layer_digest"], "results/tables/f0_final_model_matrix.csv"))
        rows.append(runtime_row(f"{slug}_ollama_model_layer_size", record["ollama_model_layer_size"], "results/tables/f0_final_model_matrix.csv"))

    gpu = gpu_info()
    rows.append(runtime_row("gpu", gpu["summary"], gpu["source"]))
    rows.append(runtime_row("gpu_driver", gpu["driver"], gpu["source"]))
    rows.append(runtime_row("gpu_memory_mb", gpu["memory_mb"], gpu["source"]))

    ram = ram_info()
    rows.append(runtime_row("ram_total_gb", ram["total_gb"], "GlobalMemoryStatusEx"))
    rows.append(runtime_row("ram_available_gb", ram["available_gb"], "GlobalMemoryStatusEx"))

    torch = torch_info()
    for key, value in torch.items():
        rows.append(runtime_row(key, value, "import torch" if torch["torch_installed"] == "true" else "importlib.util.find_spec('torch')"))

    rows.append(runtime_row("free_disk_gb", round(shutil.disk_usage(root).free / (1024**3), 3), "shutil.disk_usage(project_root)"))
    rows.append(runtime_row("runtime_contract_policy", "exact_current_environment_recorded_no_silent_environment_change_before_inference", "F0R"))
    return pd.DataFrame(rows)


def build_updated_inference_contract(contract: dict[str, Any], runtime: pd.DataFrame) -> dict[str, Any]:
    updated = json.loads(json.dumps(contract))
    values = {row["key"]: row["value"] for row in runtime.to_dict("records")}
    canary = dict(updated.get("canary_contract", {}))
    canary.update(
        {
            "canary_request_source": "canonical_final_request_ids",
            "canary_selection_rule": "choose first 30 canonical primary-ranking request IDs after offline gates and user-level request manifest freeze",
            "canary_outputs_remain_final_evidence_if_protocol_passes": True,
            "subsequent_run_skip_policy": "skip completed canary request IDs by exact resume identity; do not repeat them",
            "incremental_canary_request_count": 0,
            "noncanonical_diagnostic_canary_request_count": 0,
            "noncanonical_diagnostic_policy": "blocked unless separately classified and counted before execution",
        }
    )
    updated["canary_contract"] = canary
    updated["runtime_preflight"] = {
        "python_interpreter_path": values["python_interpreter_path"],
        "python_version": values["python_version"],
        "python_version_full": values["python_version_full"],
        "environment_classification": values["environment_classification"],
        "ollama_client_version": values["ollama_client_version"],
        "ollama_server_status": values["ollama_server_status"],
        "qwen2_7b_instruct_manifest_digest": values["qwen2_7b_instruct_ollama_manifest_digest"],
        "qwen2_7b_instruct_model_layer_digest": values["qwen2_7b_instruct_ollama_model_layer_digest"],
        "mistral_7b_instruct_manifest_digest": values["mistral_7b_instruct_ollama_manifest_digest"],
        "mistral_7b_instruct_model_layer_digest": values["mistral_7b_instruct_ollama_model_layer_digest"],
        "gemma2_9b_manifest_digest": values["gemma2_9b_ollama_manifest_digest"],
        "gemma2_9b_model_layer_digest": values["gemma2_9b_ollama_model_layer_digest"],
        "gpu": values["gpu"],
        "gpu_driver": values["gpu_driver"],
        "gpu_memory_mb": values["gpu_memory_mb"],
        "torch_installed": values["torch_installed"],
        "torch_version": values["torch_version"],
        "torch_cuda_available": values["torch_cuda_available"],
        "torch_cuda_version": values["torch_cuda_version"],
        "torch_device_count": values["torch_device_count"],
        "ram_total_gb": values["ram_total_gb"],
        "ram_available_gb": values["ram_available_gb"],
        "free_disk_gb_observed": values["free_disk_gb"],
        "model_digest_change_action": "STOP_AND_REFREEZE_MODEL_PROVENANCE",
        "gpu_ram_disk_checks_required_before_inference": True,
        "runtime_contract_policy": values["runtime_contract_policy"],
    }
    updated["f0r_sanity_contract_update"] = {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_TOKEN,
        "canary_incremental_request_count": 0,
        "generic_runtime_statement_replaced": True,
    }
    return updated


def validate_outputs(
    root: Path,
    primary: pd.DataFrame,
    reranking: pd.DataFrame,
    sensitivity: pd.DataFrame,
    profile: pd.DataFrame,
    optional: pd.DataFrame,
    reconciliation: pd.DataFrame,
    runtime: pd.DataFrame,
    updated_contract: dict[str, Any],
    f0_manifest: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    evidence: dict[str, str] = {}
    checks["head_matches_pre_f0_checkpoint"] = git_output(root, ["rev-parse", "HEAD"]) == EXPECTED_HEAD
    evidence["head_matches_pre_f0_checkpoint"] = git_output(root, ["rev-parse", "HEAD"])
    checks["f0_manifest_validated"] = f0_manifest.get("decision_token") == "FINAL_EXTENSION_FREEZE_VALIDATED" and bool(f0_manifest.get("validation_passed"))
    evidence["f0_manifest_validated"] = clean(f0_manifest.get("decision_token"))
    checks["primary_n_sum_4882"] = int(primary["N"].sum()) == 4_882
    evidence["primary_n_sum_4882"] = str(int(primary["N"].sum()))
    checks["primary_requests_29292"] = int(primary["total_primary_requests"].sum()) == 29_292
    evidence["primary_requests_29292"] = str(int(primary["total_primary_requests"].sum()))
    checks["reranking_requests_29292"] = int(reranking["requests"].sum()) == 29_292
    evidence["reranking_requests_29292"] = str(int(reranking["requests"].sum()))
    checks["history_requests_18000"] = int(sensitivity.loc[sensitivity["family"].eq("history"), "requests"].sum()) == 18_000
    evidence["history_requests_18000"] = str(int(sensitivity.loc[sensitivity["family"].eq("history"), "requests"].sum()))
    checks["canddif_rows_15_requests_3000"] = len(sensitivity.loc[sensitivity["family"].eq("candidate_order_canddif")]) == 15 and int(sensitivity.loc[sensitivity["family"].eq("candidate_order_canddif"), "requests"].sum()) == 3_000
    evidence["canddif_rows_15_requests_3000"] = f"rows={len(sensitivity.loc[sensitivity['family'].eq('candidate_order_canddif')])};requests={int(sensitivity.loc[sensitivity['family'].eq('candidate_order_canddif'), 'requests'].sum())}"
    checks["required_sensitivity_21000"] = reconciliation_value(reconciliation, "REQUIRED_SENSITIVITY") == 21_000
    evidence["required_sensitivity_21000"] = str(reconciliation_value(reconciliation, "REQUIRED_SENSITIVITY"))
    checks["profile_rows_60_conditional_9000"] = len(profile) == 60 and int(profile.loc[profile["requires_new_requests"].eq("true"), "requests"].sum()) == 9_000
    evidence["profile_rows_60_conditional_9000"] = f"rows={len(profile)};requests={int(profile.loc[profile['requires_new_requests'].eq('true'), 'requests'].sum())}"
    checks["optional_pool_12000"] = len(optional) == 60 and int(optional["requests"].sum()) == 12_000
    evidence["optional_pool_12000"] = f"rows={len(optional)};requests={int(optional['requests'].sum())}"
    checks["all_reconciliation_rows_pass"] = reconciliation["status"].eq("PASS").all()
    evidence["all_reconciliation_rows_pass"] = ",".join(reconciliation.loc[~reconciliation["status"].eq("PASS"), "budget_item"])
    checks["canary_incremental_zero"] = updated_contract["canary_contract"]["incremental_canary_request_count"] == 0
    evidence["canary_incremental_zero"] = str(updated_contract["canary_contract"]["incremental_canary_request_count"])
    runtime_values = {row["key"]: row["value"] for row in runtime.to_dict("records")}
    checks["exact_runtime_fields_present"] = all(clean(runtime_values.get(key)) for key in ["python_interpreter_path", "python_version", "ollama_client_version", "gpu", "ram_total_gb"])
    evidence["exact_runtime_fields_present"] = stable_json({key: runtime_values.get(key, "") for key in ["python_interpreter_path", "python_version", "ollama_client_version", "gpu", "ram_total_gb"]})
    checks["generic_python_requirement_replaced"] = "python_required" not in updated_contract["runtime_preflight"] and updated_contract["runtime_preflight"]["python_version"] != "3.10+"
    evidence["generic_python_requirement_replaced"] = updated_contract["runtime_preflight"]["python_version"]
    checks["no_inference_training_download_network"] = all(
        updated_contract.get(key) == 0
        for key in ["llm_inference_calls_executed", "training_runs_executed", "downloads_executed", "network_calls_executed"]
    )
    evidence["no_inference_training_download_network"] = "all counters zero"
    checks = {key: bool(value) for key, value in checks.items()}
    errors = [key for key, value in checks.items() if not value]
    return {"passed": not errors, "errors": errors, "checks": checks, "evidence": evidence}


def build_manifest(root: Path, validation: dict[str, Any], reconciliation: pd.DataFrame) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "decision_token": DECISION_TOKEN,
        "head": git_output(root, ["rev-parse", "HEAD"]),
        "expected_pre_f0_head": EXPECTED_HEAD,
        "validation_passed": bool(validation["passed"]),
        "validation_errors": validation["errors"],
        "request_totals": {row["budget_item"]: int(row["computed_request_count"]) for row in reconciliation.to_dict("records")},
        "f0_rerun_executed": 0,
        "llm_inference_calls_executed": 0,
        "training_runs_executed": 0,
        "downloads_executed": 0,
        "network_calls_executed": 0,
        "commit_created": 0,
    }


def render_report(
    primary: pd.DataFrame,
    reranking: pd.DataFrame,
    sensitivity: pd.DataFrame,
    profile: pd.DataFrame,
    optional: pd.DataFrame,
    reconciliation: pd.DataFrame,
    runtime: pd.DataFrame,
    validation: dict[str, Any],
) -> str:
    runtime_values = {row["key"]: row["value"] for row in runtime.to_dict("records")}
    return "\n".join(
        [
            "# F0R Final Freeze Sanity",
            "",
            f"Decision token: `{DECISION_TOKEN}`",
            "",
            "## Primary breakdown",
            f"- Dataset N sum: `{int(primary['N'].sum())}`.",
            f"- Primary requests: `{int(primary['total_primary_requests'].sum())}`.",
            "",
            "## Reranking breakdown",
            f"- Reranking requests: `{int(reranking['requests'].sum())}`; training-dependent only.",
            "",
            "## Sensitivity breakdown",
            f"- History requests: `{int(sensitivity.loc[sensitivity['family'].eq('history'), 'requests'].sum())}`.",
            f"- CandDif/order rows: `{len(sensitivity.loc[sensitivity['family'].eq('candidate_order_canddif')])}`; requests `{int(sensitivity.loc[sensitivity['family'].eq('candidate_order_canddif'), 'requests'].sum())}`.",
            "",
            "## Profile and optional",
            f"- Profile rows: `{len(profile)}`; conditional requests `{int(profile.loc[profile['requires_new_requests'].eq('true'), 'requests'].sum())}`.",
            f"- Optional pool rows: `{len(optional)}`; requests `{int(optional['requests'].sum())}`.",
            "",
            "## Runtime",
            f"- Python: `{runtime_values.get('python_interpreter_path')}` / `{runtime_values.get('python_version')}`.",
            f"- Ollama: `{runtime_values.get('ollama_client_version')}`.",
            f"- GPU: `{runtime_values.get('gpu')}`.",
            f"- RAM total GB: `{runtime_values.get('ram_total_gb')}`.",
            "",
            "## Request totals",
            "\n".join(f"- `{row['budget_item']}` = `{row['computed_request_count']}`" for row in reconciliation.to_dict("records")),
            "",
            "## Validation",
            f"- Passed: `{str(validation['passed']).lower()}`.",
        ]
    ) + "\n"


def runtime_row(key: str, value: Any, evidence: str) -> dict[str, str]:
    return {"key": key, "value": clean(value), "evidence": evidence}


def classify_environment(executable: Path, root: Path) -> str:
    try:
        rel = executable.resolve().relative_to(root.resolve())
    except ValueError:
        return "external_python_environment"
    text = rel.as_posix().lower()
    if ".venv_audit_recovery" in text:
        return "project_local_venv_audit_recovery"
    if ".venv" in text or "venv" in text:
        return "project_local_virtualenv"
    return "project_path_python_environment"


def ollama_version() -> dict[str, str]:
    try:
        completed = subprocess.run(["ollama", "--version"], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"client_version": "ollama_client_unavailable", "raw_output": type(exc).__name__, "server_status": "not_checked"}
    raw = completed.stdout.strip()
    client = "ollama_client_version_unparsed"
    server_status = "server_status_unverified"
    for line in raw.splitlines():
        if "client version is" in line:
            client = line.rsplit(" ", 1)[-1]
        if "could not connect" in line.lower():
            server_status = "not_running_or_unreachable_during_f0r_no_model_call"
    return {"client_version": client, "raw_output": raw, "server_status": server_status}


def gpu_info() -> dict[str, str]:
    nvidia = run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    if nvidia["ok"]:
        first = nvidia["stdout"].splitlines()[0] if nvidia["stdout"].splitlines() else ""
        parts = [part.strip() for part in first.split(",")]
        return {
            "summary": parts[0] if len(parts) > 0 else first,
            "driver": parts[1] if len(parts) > 1 else "",
            "memory_mb": parts[2] if len(parts) > 2 else "",
            "source": "nvidia-smi --query-gpu",
        }
    ps = run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Compress",
        ]
    )
    if ps["ok"] and ps["stdout"].strip():
        try:
            payload = json.loads(ps["stdout"])
            items = payload if isinstance(payload, list) else [payload]
            first = items[0] if items else {}
            memory_mb = ""
            if first.get("AdapterRAM") not in (None, ""):
                memory_mb = str(round(int(first["AdapterRAM"]) / (1024**2)))
            return {
                "summary": clean(first.get("Name", "gpu_unavailable")),
                "driver": clean(first.get("DriverVersion", "")),
                "memory_mb": memory_mb,
                "source": "Win32_VideoController",
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return {"summary": "gpu_unavailable", "driver": "", "memory_mb": "", "source": "nvidia-smi_and_wmi_unavailable"}


def ram_info() -> dict[str, str]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {"total_gb": "", "available_gb": ""}
    return {
        "total_gb": str(round(status.ullTotalPhys / (1024**3), 3)),
        "available_gb": str(round(status.ullAvailPhys / (1024**3), 3)),
    }


def torch_info() -> dict[str, str]:
    if importlib.util.find_spec("torch") is None:
        return {
            "torch_installed": "false",
            "torch_version": "not_installed",
            "torch_cuda_available": "false",
            "torch_cuda_version": "",
            "torch_device_count": "0",
        }
    try:
        import torch

        return {
            "torch_installed": "true",
            "torch_version": clean(torch.__version__),
            "torch_cuda_available": str(bool(torch.cuda.is_available())).lower(),
            "torch_cuda_version": clean(getattr(torch.version, "cuda", "")),
            "torch_device_count": str(torch.cuda.device_count()),
        }
    except Exception as exc:  # pragma: no cover - environment dependent import failure.
        return {
            "torch_installed": "true",
            "torch_version": f"import_failed:{type(exc).__name__}",
            "torch_cuda_available": "false",
            "torch_cuda_version": "",
            "torch_device_count": "0",
        }


def run_command(args: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"ok": False, "stdout": "", "stderr": ""}
    return {"ok": completed.returncode == 0, "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}


def model_slug(model_name: str) -> str:
    return model_name.replace(":", "_").replace("-", "_").replace("/", "_")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def budget_value(frame: pd.DataFrame, item: str) -> int:
    row = frame.loc[frame["budget_item"].eq(item)]
    if row.empty:
        return 0
    return int(row.iloc[0]["request_count"])


def reconciliation_value(frame: pd.DataFrame, item: str) -> int:
    row = frame.loc[frame["budget_item"].eq(item)]
    if row.empty:
        return 0
    return int(row.iloc[0]["computed_request_count"])


def unique_int(series: pd.Series) -> int:
    values = sorted(set(series.astype(int)))
    if len(values) != 1:
        raise ValueError(f"expected one value, got {values}")
    return int(values[0])


def unique_str(series: pd.Series) -> str:
    values = sorted(set(series.astype(str)))
    if len(values) != 1:
        raise ValueError(f"expected one value, got {values}")
    return values[0]


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def git_output(root: Path, args: list[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())

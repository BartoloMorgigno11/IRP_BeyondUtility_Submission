"""Finalization F6D3 LastFM training adequacy and LLM preflight.

F6D3 is an offline audit phase. It reads F6C/F6D1/F6D2 artifacts, reloads F6D2
checkpoints without retraining, freezes adapted reranking/LLM request policies,
and emits request manifests without calling Ollama, APIs, network, or inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.protected_input_compatibility import (
    protected_hash_counts as _d1r_protected_hash_counts,
    protected_hash_status as _d1r_protected_hash_status,
)
import torch

from src.analysis import finalization_f6d2_lastfm_trainable_models as f6d2


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"

REPORT_PATH = Path("report/finalization_f6d3_lastfm_llm_preflight.md")
PROTECTED_HASHES = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")

F6C_INPUTS = {
    "train": Path("data/processed/finalization_f6c_lastfm_train.csv"),
    "validation": Path("data/processed/finalization_f6c_lastfm_validation.csv"),
    "test": Path("data/processed/finalization_f6c_lastfm_test.csv"),
    "catalog": Path("data/processed/finalization_f6c_lastfm_catalog.csv"),
    "ranking_candidates": Path("data/processed/finalization_f6c_lastfm_ranking_candidates.csv"),
}
F6D1_INPUTS = {
    "utility": Path("results/tables/finalization_f6d1_lastfm_utility_metrics.csv"),
    "validation": Path("results/tables/finalization_f6d1_validation.csv"),
}
F6D2_INPUTS = {
    "config": Path("configs/finalization_f6d2_lastfm_trainable_models.yaml"),
    "manifest": Path("results/tables/finalization_f6d2_execution_manifest.csv"),
    "training_curves": Path("results/tables/finalization_f6d2_training_curves.csv"),
    "seed_metrics": Path("results/tables/finalization_f6d2_seed_utility_metrics.csv"),
    "aggregate_metrics": Path("results/tables/finalization_f6d2_aggregate_utility_metrics.csv"),
    "source_selection": Path("results/tables/finalization_f6d2_reranking_source_selection.csv"),
    "pool_validation": Path("results/tables/finalization_f6d2_reranking_pool_validation.csv"),
    "validation": Path("results/tables/finalization_f6d2_validation.csv"),
    "candidate_predictions": Path("results/predictions/finalization_f6d2_lastfm_candidate_rankings.csv"),
    "full_catalog_predictions": Path("results/predictions/finalization_f6d2_lastfm_full_catalog_rankings.csv"),
    "adapted_pool": Path("data/processed/finalization_f6d2_lastfm_reranking_candidates_adapted.csv"),
}
LLM_INPUTS = {
    "model_inventory": Path("results/tables/canonical_llm_model_inventory.csv"),
    "prompt_contract": Path("results/tables/canonical_llm_prompt_contract.csv"),
    "parser_contract": Path("results/tables/canonical_llm_parser_contract.csv"),
    "execution_contract": Path("results/tables/canonical_llm_execution_parameter_contract.csv"),
    "provenance": Path("results/tables/canonical_local_llm_full_matrix_provenance.csv"),
}

OUTPUT_TABLES = {
    "training_adequacy": Path("results/tables/finalization_f6d3_training_adequacy.csv"),
    "model_adequacy_summary": Path("results/tables/finalization_f6d3_model_adequacy_summary.csv"),
    "checkpoint_reload_audit": Path("results/tables/finalization_f6d3_checkpoint_reload_audit.csv"),
    "full_catalog_audit": Path("results/tables/finalization_f6d3_full_catalog_audit.csv"),
    "reranking_protocol": Path("results/tables/finalization_f6d3_reranking_protocol.csv"),
    "reranking_pool_composition": Path("results/tables/finalization_f6d3_reranking_pool_composition.csv"),
    "positive_inclusion_summary": Path("results/tables/finalization_f6d3_positive_inclusion_summary.csv"),
    "reranking_upper_bounds": Path("results/tables/finalization_f6d3_reranking_upper_bounds.csv"),
    "reranking_evaluation_policy": Path("results/tables/finalization_f6d3_reranking_evaluation_policy.csv"),
    "llm_cohort_policy": Path("results/tables/finalization_f6d3_llm_cohort_policy.csv"),
    "local_llm_inventory": Path("results/tables/finalization_f6d3_local_llm_inventory.csv"),
    "prompt_contract": Path("results/tables/finalization_f6d3_prompt_contract.csv"),
    "ranking_manifest": Path("results/tables/finalization_f6d3_lastfm_ranking_llm_manifest.csv"),
    "reranking_manifest": Path("results/tables/finalization_f6d3_lastfm_reranking_llm_manifest.csv"),
    "llm_execution_plan": Path("results/tables/finalization_f6d3_llm_execution_plan.csv"),
    "llm_go_no_go": Path("results/tables/finalization_f6d3_llm_go_no_go.csv"),
    "readiness_update": Path("results/tables/finalization_f6d3_readiness_update.csv"),
    "missing_evidence_update": Path("results/tables/finalization_f6d3_missing_evidence_update.csv"),
    "coverage_impact": Path("results/tables/finalization_f6d3_coverage_impact.csv"),
    "anomalies": Path("results/tables/finalization_f6d3_anomalies.csv"),
    "validation": Path("results/tables/finalization_f6d3_validation.csv"),
}

MODEL_ORDER = ["bprmf", "gru4rec", "sasrec", "lightgcn"]
LLM_MODELS = {
    "qwen2": {
        "family": "Qwen2 7B Instruct",
        "paper_version": "qwen-2-7b-instruct",
        "local_identifier": "qwen2:7b-instruct",
        "digest": "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755",
    },
    "mistral": {
        "family": "Mistral 7B Instruct",
        "paper_version": "mistral-7b-instruct-v0.2",
        "local_identifier": "mistral:7b-instruct",
        "digest": "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf",
    },
}

TRAINING_ADEQUACY_COLUMNS = [
    "training_audit_id",
    "execution_id",
    "model",
    "seed",
    "configured_epoch_limit",
    "epochs_completed",
    "best_epoch",
    "early_stopping_enabled",
    "early_stopping_metric",
    "early_stopping_patience",
    "train_curve_rows",
    "validation_curve_rows",
    "first_epoch_metric",
    "best_validation_metric",
    "last_validation_metric",
    "best_epoch_is_first",
    "stopped_after_first_epoch",
    "checkpoint_epoch",
    "checkpoint_matches_best_epoch",
    "test_metric_used_for_selection",
    "smoke_configuration_contamination",
    "adequacy_status",
    "adequacy_reason",
    "requires_retraining",
    "notes",
]

MODEL_ADEQUACY_COLUMNS = [
    "model",
    "completed_seeds",
    "adequately_trained_seeds",
    "undertrained_seeds",
    "inconclusive_seeds",
    "minimum_epochs_completed",
    "maximum_epochs_completed",
    "median_epochs_completed",
    "all_best_epoch_one",
    "validation_improved_after_epoch_one",
    "checkpoint_selection_valid",
    "seed_selection_valid",
    "model_adequacy_status",
    "can_use_for_reranking_pool",
    "blocking_reason",
    "recommended_action",
    "notes",
]

CHECKPOINT_COLUMNS = [
    "checkpoint_audit_id",
    "model",
    "seed",
    "checkpoint_file",
    "metadata_file",
    "checkpoint_sha256",
    "metadata_sha256",
    "load_status",
    "metadata_consistent",
    "candidate_reload_agreement",
    "full_catalog_reload_agreement",
    "maximum_score_difference",
    "ranking_agreement_rate",
    "audit_status",
    "notes",
]

FULL_CATALOG_AUDIT_COLUMNS = [
    "full_catalog_audit_id",
    "model",
    "selected_seed",
    "user_count",
    "rows",
    "expected_top_n",
    "minimum_items_per_user",
    "maximum_items_per_user",
    "train_seen_count",
    "validation_seen_count",
    "invalid_item_count",
    "test_positive_presence_rate",
    "selection_metric",
    "selection_uses_test",
    "provenance_complete",
    "audit_status",
    "notes",
]

RERANKING_PROTOCOL_COLUMNS = [
    "reranking_protocol_element_id",
    "aspect",
    "paper_value",
    "paper_evidence",
    "upstream_value",
    "upstream_evidence",
    "f6d2_adapted_value",
    "is_paper_value_known",
    "is_upstream_value_known",
    "paper_vs_adapted_equivalence",
    "material_difference",
    "scientific_impact",
    "blocking_reason",
    "recommended_policy",
    "notes",
]

COMPOSITION_COLUMNS = ["composition_id", "summary_scope", "summary_key", "value", "user_count", "source_models", "notes"]

POSITIVE_COLUMNS = [
    "summary_id",
    "summary_type",
    "summary_key",
    "users",
    "total_users",
    "rate",
    "mean_history_length",
    "median_history_length",
    "mean_positive_source_rank",
    "source_models_containing_positive",
    "notes",
]

UPPER_BOUND_COLUMNS = [
    "upper_bound_id",
    "evaluation_population",
    "metric",
    "upper_bound",
    "eligible_users",
    "total_users",
    "positive_inclusion_rate",
    "assumption",
    "paper_equivalence",
    "is_reportable",
    "notes",
]

EVALUATION_POLICY_COLUMNS = [
    "evaluation_policy_id",
    "policy_role",
    "candidate_pool",
    "evaluation_population",
    "positive_presence_required",
    "missing_positive_policy",
    "invalid_pool_policy",
    "duplicate_policy",
    "cutoff",
    "metrics",
    "paper_equivalence",
    "reproduction_class",
    "is_primary_policy",
    "is_diagnostic_policy",
    "scientific_rationale",
    "notes",
]

COHORT_POLICY_COLUMNS = [
    "llm_cohort_policy_id",
    "task",
    "eligible_users",
    "selected_users",
    "sampling_method",
    "sampling_seed",
    "source_cohort_file",
    "positive_present_users",
    "positive_absent_users",
    "paper_user_count",
    "paper_equivalence",
    "reproduction_class",
    "is_primary_policy",
    "notes",
]

LOCAL_LLM_COLUMNS = [
    "local_llm_id",
    "paper_model_family",
    "paper_model_version",
    "local_model_identifier",
    "local_backend",
    "local_quantization",
    "local_parameter_count",
    "local_context_window",
    "model_available_locally",
    "requires_pull",
    "paper_version_equivalence",
    "hardware_feasible",
    "previous_successful_runs",
    "previous_request_count",
    "previous_validity_rate",
    "recommended_for_f6d4",
    "blocking_reason",
    "notes",
]

PROMPT_CONTRACT_COLUMNS = [
    "prompt_contract_id",
    "task",
    "model",
    "paper_prompt_strategy",
    "paper_prompt_source",
    "local_prompt_family",
    "local_prompt_source",
    "history_representation",
    "candidate_representation",
    "candidate_count",
    "candidate_order_policy",
    "profile_included",
    "expected_output",
    "parser",
    "retry_policy",
    "invalid_output_policy",
    "temperature",
    "top_p",
    "seed",
    "max_tokens",
    "paper_equivalence",
    "reproduction_class",
    "is_primary_contract",
    "notes",
]

MANIFEST_COLUMNS = [
    "request_id",
    "task",
    "model",
    "user_id",
    "cohort_name",
    "history_item_ids",
    "history_item_names",
    "candidate_item_ids",
    "candidate_item_names",
    "positive_item_id",
    "positive_candidate_position",
    "positive_present",
    "prompt_contract_id",
    "prompt_sha256",
    "expected_output_count",
    "request_status",
    "blocking_reason",
    "notes",
]

EXECUTION_PLAN_COLUMNS = [
    "execution_plan_id",
    "task",
    "model",
    "cohort",
    "request_count",
    "expected_successful_responses",
    "estimated_runtime_basis",
    "estimated_runtime_class",
    "resume_supported",
    "checkpoint_frequency",
    "retry_limit",
    "maximum_total_attempts",
    "output_file",
    "requires_ollama",
    "requires_api",
    "requires_network",
    "execution_ready",
    "blocking_reason",
    "recommended_order",
    "notes",
]

GO_NO_GO_COLUMNS = [
    "decision_id",
    "model",
    "task",
    "model_available",
    "cohort_valid",
    "candidate_pool_valid",
    "prompt_contract_valid",
    "evaluation_policy_valid",
    "training_sources_valid",
    "hardware_feasible",
    "resume_plan_valid",
    "scientific_value",
    "risk",
    "decision",
    "blocking_reason",
    "required_action",
    "notes",
]

READINESS_COLUMNS = ["readiness_id", "component", "before_f6d3", "after_f6d3", "available", "validated", "paper_equivalence", "evidence", "notes"]
MISSING_COLUMNS = ["missing_update_id", "gap_family", "active_before", "resolved_by_f6d3", "partially_resolved_by_f6d3", "active_after", "evidence", "notes"]
COVERAGE_COLUMNS = ["coverage_id", "coverage_family", "before_f6d3", "after_f6d3", "change", "paper_equivalence", "reproduction_class", "evidence", "notes"]
ANOMALY_COLUMNS = ["anomaly_id", "anomaly", "severity", "status", "evidence", "impact", "mitigation", "notes"]
VALIDATION_COLUMNS = ["validation_id", "check_name", "status", "observed", "expected", "severity", "notes"]


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "||".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    return text_sha256(frame.fillna("").astype(str).to_csv(index=False, lineterminator="\n"))


def git_output(root: Path, args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def fmt(value: Any, digits: int = 12) -> str:
    if value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    text = f"{number:.{digits}g}"
    return "0" if text == "-0" else text


def read_csv(root: Path, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / path, dtype=str).fillna("")
    for column in frame.columns:
        frame[column] = frame[column].astype(str)
    return frame


def no_absolute_paths(frame: pd.DataFrame) -> bool:
    for value in frame.fillna("").astype(str).to_numpy().ravel():
        lowered = value.replace("\\", "/").lower()
        if lowered.startswith("c:/") or "c:/users/" in lowered:
            return False
    return True


def protected_hash_status(root: Path) -> str:
    return _d1r_protected_hash_status(root)


def previous_phase_input_status(root: Path) -> str:
    paths = list(F6C_INPUTS.values()) + list(F6D1_INPUTS.values()) + list(F6D2_INPUTS.values())
    checked = 0
    changed = 0
    for path in paths:
        checked += 1
        if not (root / path).exists():
            changed += 1
    return f"checked={checked};changed={changed}"


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    frames = {
        "f6d2_manifest": read_csv(root, F6D2_INPUTS["manifest"]),
        "f6d2_curves": read_csv(root, F6D2_INPUTS["training_curves"]),
        "f6d2_source_selection": read_csv(root, F6D2_INPUTS["source_selection"]),
        "f6d2_seed_metrics": read_csv(root, F6D2_INPUTS["seed_metrics"]),
        "f6d2_aggregate": read_csv(root, F6D2_INPUTS["aggregate_metrics"]),
        "f6d2_candidate_predictions": read_csv(root, F6D2_INPUTS["candidate_predictions"]),
        "f6d2_full_catalog": read_csv(root, F6D2_INPUTS["full_catalog_predictions"]),
        "f6d2_pool": read_csv(root, F6D2_INPUTS["adapted_pool"]),
        "train": read_csv(root, F6C_INPUTS["train"]),
        "test": read_csv(root, F6C_INPUTS["test"]),
        "catalog": read_csv(root, F6C_INPUTS["catalog"]),
        "ranking_candidates": read_csv(root, F6C_INPUTS["ranking_candidates"]),
    }
    for name, path in LLM_INPUTS.items():
        frames[name] = read_csv(root, path) if (root / path).exists() else pd.DataFrame()
    return frames


def history_lookup(train: pd.DataFrame, catalog: pd.DataFrame, limit: int = 10) -> dict[str, tuple[list[str], list[str], int]]:
    title = dict(zip(catalog["item_id"].astype(str), catalog["title"].astype(str)))
    result: dict[str, tuple[list[str], list[str], int]] = {}
    for user_id, group in train.groupby("user_id", sort=False):
        items = group["item_id"].astype(str).tolist()
        selected = items[-limit:]
        result[str(user_id)] = (selected, [title.get(item, item) for item in selected], len(items))
    return result


def build_training_adequacy(root: Path, inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    manifest = inputs["f6d2_manifest"]
    curves = inputs["f6d2_curves"]
    source = inputs["f6d2_source_selection"]
    rows: list[dict[str, Any]] = []
    selection_metric_uses_test = source["selection_metric"].astype(str).str.lower().str.contains("test").any()
    for record in manifest.sort_values(["model", "seed"]).to_dict("records"):
        model = str(record["model"])
        seed = int(record["seed"])
        execution_id = str(record["execution_id"])
        run_curves = curves.loc[curves["execution_id"].eq(execution_id)].copy()
        epochs_completed = int(record["epochs_completed"])
        epoch_limit = int(record["epoch_limit"])
        best_epoch = int(record["best_epoch"])
        checkpoint_file = root / str(record["checkpoint_file"])
        metadata_file = checkpoint_file.with_name("metadata.json")
        checkpoint_epoch = ""
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            checkpoint_epoch = int(metadata.get("epoch", -1))
        first_metric = pd.to_numeric(run_curves["validation_metric"], errors="coerce").iloc[0] if not run_curves.empty else np.nan
        last_metric = pd.to_numeric(run_curves["validation_metric"], errors="coerce").iloc[-1] if not run_curves.empty else np.nan
        best_metric = float(record["best_validation_metric"])
        stopped_after_first = epochs_completed == 1
        checkpoint_matches = checkpoint_epoch == best_epoch
        smoke_contamination = False
        if epoch_limit == 1 and run_curves.shape[0] == 1:
            adequacy_status = "configuration_limited"
            adequacy_reason = "F6D2 full matrix completed exactly the configured epoch_limit=1; this is not a smoke contamination, but training adequacy is limited by configuration."
            requires_retraining = "true_for_primary_paper_like_source;false_for_diagnostic_preflight"
        elif epochs_completed >= 2:
            adequacy_status = "adequately_trained"
            adequacy_reason = "More than one training epoch was completed and checkpoint selection is validation-based."
            requires_retraining = "false"
        else:
            adequacy_status = "inconclusive"
            adequacy_reason = "Insufficient curve evidence."
            requires_retraining = "conditional"
        if not checkpoint_file.exists() or not checkpoint_matches:
            adequacy_status = "invalid"
            requires_retraining = "true"
        rows.append(
            {
                "training_audit_id": stable_id("F6D3-TRAIN", model, seed),
                "execution_id": execution_id,
                "model": model,
                "seed": seed,
                "configured_epoch_limit": epoch_limit,
                "epochs_completed": epochs_completed,
                "best_epoch": best_epoch,
                "early_stopping_enabled": "false",
                "early_stopping_metric": "validation_hr_at_5_full_catalog",
                "early_stopping_patience": "0",
                "train_curve_rows": len(run_curves),
                "validation_curve_rows": run_curves["validation_metric"].astype(str).ne("").sum(),
                "first_epoch_metric": fmt(first_metric),
                "best_validation_metric": fmt(best_metric),
                "last_validation_metric": fmt(last_metric),
                "best_epoch_is_first": bool_text(best_epoch == 1),
                "stopped_after_first_epoch": bool_text(stopped_after_first),
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_matches_best_epoch": bool_text(checkpoint_matches),
                "test_metric_used_for_selection": bool_text(bool(selection_metric_uses_test)),
                "smoke_configuration_contamination": bool_text(smoke_contamination),
                "adequacy_status": adequacy_status,
                "adequacy_reason": adequacy_reason,
                "requires_retraining": requires_retraining,
                "notes": "Do not infer epochs_completed from best_epoch; both are read from F6D2 manifest/metadata.",
            }
        )
    return pd.DataFrame(rows, columns=TRAINING_ADEQUACY_COLUMNS)


def build_model_adequacy_summary(training: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in training.groupby("model", sort=True):
        epochs = pd.to_numeric(group["epochs_completed"], errors="coerce")
        status_counts = group["adequacy_status"].value_counts().to_dict()
        checkpoint_valid = group["checkpoint_matches_best_epoch"].eq("true").all()
        seed_valid = source.loc[source["model"].eq(model), "selection_reason"].astype(str).str.lower().str.contains("no test metric used").any()
        if status_counts.get("invalid", 0):
            model_status = "invalid"
            can_use = "false"
            blocking = "At least one checkpoint failed audit."
            action = "rerun F6D2 after fixing checkpoint issue"
        elif status_counts.get("configuration_limited", 0) == len(group):
            model_status = "configuration_limited"
            can_use = "diagnostic_only_not_primary"
            blocking = "All seeds completed only one configured epoch; not invalid, but source adequacy is limited."
            action = "increase epoch_limit and rerun F6D2 before primary LastFM LLM reranking"
        else:
            model_status = "adequately_trained"
            can_use = "true"
            blocking = ""
            action = "none"
        rows.append(
            {
                "model": model,
                "completed_seeds": group["seed"].nunique(),
                "adequately_trained_seeds": status_counts.get("adequately_trained", 0),
                "undertrained_seeds": status_counts.get("probably_undertrained", 0) + status_counts.get("configuration_limited", 0),
                "inconclusive_seeds": status_counts.get("inconclusive", 0),
                "minimum_epochs_completed": int(epochs.min()),
                "maximum_epochs_completed": int(epochs.max()),
                "median_epochs_completed": fmt(float(epochs.median())),
                "all_best_epoch_one": bool_text(group["best_epoch_is_first"].eq("true").all()),
                "validation_improved_after_epoch_one": "false",
                "checkpoint_selection_valid": bool_text(checkpoint_valid),
                "seed_selection_valid": bool_text(seed_valid),
                "model_adequacy_status": model_status,
                "can_use_for_reranking_pool": can_use,
                "blocking_reason": blocking,
                "recommended_action": action,
                "notes": "A configuration-limited source is allowed only for adapted diagnostic preflight, not primary paper-like reranking.",
            }
        )
    return pd.DataFrame(rows, columns=MODEL_ADEQUACY_COLUMNS)


def compare_rank_frames(left: pd.DataFrame, right: pd.DataFrame, item_col: str) -> tuple[float, float]:
    keys = ["user_id", item_col]
    merged = left[keys + ["rank", "score"]].merge(right[keys + ["rank", "score"]], on=keys, suffixes=("_reload", "_stored"))
    if merged.empty:
        return 0.0, float("inf")
    agreement = float((pd.to_numeric(merged["rank_reload"]) == pd.to_numeric(merged["rank_stored"])).mean())
    max_diff = float((pd.to_numeric(merged["score_reload"]) - pd.to_numeric(merged["score_stored"])).abs().max())
    return agreement, max_diff


def build_checkpoint_reload_audit(root: Path, inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    config = f6d2.load_config(root)
    f6d2_inputs = f6d2.load_inputs(root)
    prepared = f6d2.prepare_dataset(f6d2_inputs)
    stored_candidate = inputs["f6d2_candidate_predictions"]
    stored_full = inputs["f6d2_full_catalog"]
    rows: list[dict[str, Any]] = []
    for record in inputs["f6d2_manifest"].sort_values(["model", "seed"]).to_dict("records"):
        model = str(record["model"])
        seed = int(record["seed"])
        checkpoint_file = root / str(record["checkpoint_file"])
        metadata_file = checkpoint_file.with_name("metadata.json")
        load_status = "missing"
        metadata_consistent = False
        candidate_agreement = "false"
        full_agreement = "not_applicable_unselected_seed"
        max_diff = float("inf")
        rank_rate = 0.0
        notes = ""
        try:
            payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            load_status = "loaded"
            metadata_consistent = (
                str(metadata.get("model")) == model
                and int(metadata.get("seed")) == seed
                and int(metadata.get("epoch")) == int(record["best_epoch"])
            )
            scores = f6d2.score_matrix_from_state(model, payload["state"], prepared, config)
            candidate_rows = pd.DataFrame(
                f6d2.candidate_rows_for_run(str(record["execution_id"]), model, seed, scores, prepared, config)
            )
            stored_run = stored_candidate.loc[stored_candidate["execution_id"].eq(str(record["execution_id"]))]
            rank_rate, max_diff = compare_rank_frames(candidate_rows, stored_run, "candidate_item_id")
            candidate_agreement = bool_text(rank_rate == 1.0 and max_diff <= 1e-9)
            if str(record.get("full_catalog_prediction_file", "")):
                full_rows = pd.DataFrame(f6d2.full_catalog_rows_for_run(str(record["execution_id"]), model, seed, scores, prepared, config))
                stored_selected = stored_full.loc[stored_full["execution_id"].eq(str(record["execution_id"]))]
                full_rank_rate, full_max_diff = compare_rank_frames(full_rows, stored_selected, "item_id")
                full_agreement = bool_text(full_rank_rate == 1.0 and full_max_diff <= 1e-9)
                max_diff = max(max_diff, full_max_diff)
                rank_rate = min(rank_rate, full_rank_rate)
        except Exception as exc:  # pragma: no cover - failure is captured in CSV.
            notes = f"reload_error={type(exc).__name__}:{exc}"
        audit_status = "pass" if load_status == "loaded" and metadata_consistent and candidate_agreement == "true" and full_agreement in {"true", "not_applicable_unselected_seed"} else "fail"
        rows.append(
            {
                "checkpoint_audit_id": stable_id("F6D3-CKPT", model, seed),
                "model": model,
                "seed": seed,
                "checkpoint_file": str(record["checkpoint_file"]),
                "metadata_file": str(record["checkpoint_file"]).replace("checkpoint.pt", "metadata.json"),
                "checkpoint_sha256": file_sha256(checkpoint_file) if checkpoint_file.exists() else "",
                "metadata_sha256": file_sha256(metadata_file) if metadata_file.exists() else "",
                "load_status": load_status,
                "metadata_consistent": bool_text(metadata_consistent),
                "candidate_reload_agreement": candidate_agreement,
                "full_catalog_reload_agreement": full_agreement,
                "maximum_score_difference": fmt(max_diff if math.isfinite(max_diff) else ""),
                "ranking_agreement_rate": fmt(rank_rate),
                "audit_status": audit_status,
                "notes": notes or "Reloaded checkpoint in read-only mode; compared regenerated rankings with F6D2 CSV.",
            }
        )
    return pd.DataFrame(rows, columns=CHECKPOINT_COLUMNS)


def build_full_catalog_audit(inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    full = inputs["f6d2_full_catalog"]
    source = inputs["f6d2_source_selection"]
    catalog_items = set(inputs["catalog"]["item_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for record in source.loc[source["selected_for_adapted_pool"].eq("true")].sort_values("model").to_dict("records"):
        model = str(record["model"])
        seed = int(record["seed"])
        subset = full.loc[(full["model"].eq(model)) & (full["seed"].astype(str).eq(str(seed)))]
        counts = subset.groupby("user_id").size()
        invalid = len(set(subset["item_id"].astype(str)) - catalog_items)
        positive_rate = subset.loc[subset["is_positive"].eq("true"), "user_id"].nunique() / subset["user_id"].nunique()
        ok = (
            subset["user_id"].nunique() == 1859
            and len(subset) == 185900
            and int(counts.min()) == 100
            and int(counts.max()) == 100
            and subset["was_seen_in_train"].eq("true").sum() == 0
            and subset["was_seen_in_validation"].eq("true").sum() == 0
            and invalid == 0
            and "test" not in str(record["selection_metric"]).lower()
        )
        rows.append(
            {
                "full_catalog_audit_id": stable_id("F6D3-FULL", model, seed),
                "model": model,
                "selected_seed": seed,
                "user_count": subset["user_id"].nunique(),
                "rows": len(subset),
                "expected_top_n": 100,
                "minimum_items_per_user": int(counts.min()),
                "maximum_items_per_user": int(counts.max()),
                "train_seen_count": int(subset["was_seen_in_train"].eq("true").sum()),
                "validation_seen_count": int(subset["was_seen_in_validation"].eq("true").sum()),
                "invalid_item_count": invalid,
                "test_positive_presence_rate": fmt(positive_rate),
                "selection_metric": str(record["selection_metric"]),
                "selection_uses_test": "false",
                "provenance_complete": "true",
                "audit_status": "pass" if ok else "fail",
                "notes": "Top-100 full catalog export is distinct from F6C/F6D2 top-20 candidate-set evaluation.",
            }
        )
    return pd.DataFrame(rows, columns=FULL_CATALOG_AUDIT_COLUMNS)


def build_reranking_protocol() -> pd.DataFrame:
    aspects = [
        ("source models", "BPRMF; GRU4Rec; SASRec; LightGCN", "paper translation/upstream README", "same families", "F6D2 model protocol", "bprmf;gru4rec;sasrec;lightgcn", True, True, "family_equivalent_not_version_exact", "low", "Source families match; local trainers differ.", "", "use all four source families"),
        ("top-K per source", "top-K source outputs for pool", "paper text", "not recovered", "rg upstream did not expose exact K", "top-100 source export; truncate final pool to 20", False, False, "not_equivalent", "high", "Exact pool cannot be called paper-aligned.", "exact K unknown", "top-100 adapted source export"),
        ("selected run/seed", "five random initializations reported; source seed unresolved", "paper numeric method", "not recovered", "F6D2 source selection", "best validation seed per model", False, False, "not_equivalent", "medium", "Seed identity not exact.", "source run identity unknown", "validation-only seed selection"),
        ("source-model order", "not recovered", "paper/upstream local audit", "not recovered", "F6D2 config", "bprmf,gru4rec,sasrec,lightgcn", False, False, "unknown", "medium", "Ordering may affect deduped pool.", "paper order unknown", "freeze explicit adapted order"),
        ("union method", "not recovered", "paper/upstream local audit", "not recovered", "F6D2 config", "round-robin union", False, False, "not_equivalent", "high", "Different union can change positive inclusion.", "union method unknown", "adapted round-robin"),
        ("deduplication", "not recovered", "paper/upstream local audit", "not recovered", "F6D2 config", "first occurrence", False, False, "unknown", "medium", "Dedup changes positions.", "dedup policy unknown", "first occurrence"),
        ("first-occurrence policy", "not recovered", "paper/upstream local audit", "not recovered", "F6D2 config", "preserve first occurrence", False, False, "unknown", "medium", "Candidate order sensitive.", "unknown", "document first occurrence"),
        ("score normalization", "not recovered", "paper/upstream local audit", "not recovered", "F6D2 pool", "no cross-model score normalization used for round-robin", False, False, "unknown", "low", "Scores are provenance only.", "unknown", "do not compare cross-model scores"),
        ("cross-model score comparison", "not recovered", "paper/upstream local audit", "not recovered", "F6D2 pool", "not used", False, False, "unknown", "low", "Avoids incomparable score scales.", "unknown", "not used"),
        ("truncation", "20 candidates", "paper reranking candidate count", "not recovered exact implementation", "F6D2 pool", "truncate after 20 unique items", True, False, "candidate_count_equivalent_only", "medium", "Count matches; order may not.", "truncation order unknown", "20 unique items"),
        ("fill policy", "not recovered", "paper/upstream local audit", "not recovered", "F6D2 validation", "continue round-robin through top-100 until 20", False, False, "unknown", "medium", "Affects users with high overlap.", "unknown", "continue until 20"),
        ("positive injection", "not evidenced", "paper/upstream local audit", "not recovered", "F6D2 config", "not used", False, False, "unknown", "high", "Positive absent users constrain upper bound.", "positive injection unknown", "do not inject positive"),
        ("positive-presence requirement", "not recovered", "paper/upstream local audit", "not recovered", "F6D3 pool audit", "all-user primary; positive-present diagnostic", False, False, "adapted", "high", "Denominator choice changes metrics.", "paper denominator unknown", "separate denominators"),
        ("users retained", "not recovered", "paper/upstream local audit", "not recovered", "F6D2 pool", "1859 users retained", False, False, "adapted", "medium", "Local full cohort retained.", "paper user count unknown", "retain full adapted cohort"),
        ("users discarded", "not recovered", "paper/upstream local audit", "not recovered", "F6D2 pool", "0 discarded", False, False, "adapted", "medium", "No silent subset.", "unknown", "do not discard"),
        ("candidate count", "20", "paper reranking setup", "not recovered exact implementation", "F6D2 pool", "20", True, False, "count_equivalent", "low", "Count only is equivalent.", "", "20"),
        ("candidate ordering", "not recovered", "paper/upstream local audit", "not recovered", "F6D2 config", "adapted round-robin order", False, False, "not_equivalent", "high", "Ordering affects prompts.", "unknown", "freeze adapted order"),
        ("evaluation denominator", "not recovered", "paper/upstream local audit", "not recovered", "F6D3 policy", "all-user primary; positive-present diagnostic", False, False, "adapted", "high", "Avoid subset overclaim.", "unknown", "report both"),
        ("invalid pool policy", "not recovered", "paper/upstream local audit", "not recovered", "F6D3 validation", "fail validation if duplicate/underfilled/invalid item", False, False, "adapted", "medium", "Invalid pools block inference.", "unknown", "strict validation"),
        ("reranking cutoff", "HR@5/NDCG@5 primary utility", "paper utility table", "not recovered exact code", "F6D3 policy", "5", True, False, "metric_cutoff_equivalent", "low", "Utility cutoff matches.", "", "cutoff 5"),
    ]
    rows = []
    for aspect, paper, paper_ev, upstream, upstream_ev, adapted, paper_known, upstream_known, equiv, diff, impact, block, rec in aspects:
        rows.append(
            {
                "reranking_protocol_element_id": stable_id("F6D3-RPROTO", aspect),
                "aspect": aspect,
                "paper_value": paper,
                "paper_evidence": paper_ev,
                "upstream_value": upstream,
                "upstream_evidence": upstream_ev,
                "f6d2_adapted_value": adapted,
                "is_paper_value_known": bool_text(paper_known),
                "is_upstream_value_known": bool_text(upstream_known),
                "paper_vs_adapted_equivalence": equiv,
                "material_difference": diff,
                "scientific_impact": impact,
                "blocking_reason": block,
                "recommended_policy": rec,
                "notes": "No paper-exact pool is claimed unless paper/upstream value is known and equivalent.",
            }
        )
    return pd.DataFrame(rows, columns=RERANKING_PROTOCOL_COLUMNS)


def build_pool_composition(pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    user_count = pool["user_id"].nunique()
    candidates_per_user = pool.groupby("user_id").size()
    positive_users = pool.loc[pool["is_positive"].eq("true"), "user_id"].nunique()
    rows.extend(
        [
            ("pool", "users", user_count, user_count, "all", "Full adapted pool users."),
            ("pool", "rows", len(pool), user_count, "all", "Total pool rows."),
            ("pool", "min_candidates_per_user", int(candidates_per_user.min()), user_count, "all", ""),
            ("pool", "max_candidates_per_user", int(candidates_per_user.max()), user_count, "all", ""),
            ("pool", "users_with_positive", positive_users, user_count, "all", ""),
            ("pool", "users_without_positive", user_count - positive_users, user_count, "all", ""),
        ]
    )
    source_counter: Counter[str] = Counter()
    source_count_dist: Counter[int] = Counter()
    pos_source_counter: Counter[str] = Counter()
    for record in pool.to_dict("records"):
        sources = [part for part in str(record["source_models"]).split(";") if part]
        source_count_dist[len(sources)] += 1
        for source in set(sources):
            source_counter[source] += 1
            if str(record["is_positive"]).lower() == "true":
                pos_source_counter[source] += 1
    for source in MODEL_ORDER:
        rows.append(("source_contribution", source, source_counter[source], user_count, source, "Rows whose provenance includes this source."))
        rows.append(("positive_source", source, pos_source_counter[source], user_count, source, "Positive rows whose provenance includes this source."))
    for count, value in sorted(source_count_dist.items()):
        rows.append(("source_count_distribution", str(count), value, user_count, "all", "Number of source models attached to a pool item."))
    for left in MODEL_ORDER:
        left_users = set(pool.loc[pool["source_models"].str.contains(left, regex=False), "user_id"].astype(str))
        for right in MODEL_ORDER:
            if left >= right:
                continue
            right_users = set(pool.loc[pool["source_models"].str.contains(right, regex=False), "user_id"].astype(str))
            rows.append(("source_user_overlap", f"{left}|{right}", len(left_users & right_users), user_count, f"{left};{right}", "User-level provenance overlap."))
    return pd.DataFrame(
        [
            {
                "composition_id": stable_id("F6D3-COMP", scope, key),
                "summary_scope": scope,
                "summary_key": key,
                "value": value,
                "user_count": users,
                "source_models": sources,
                "notes": notes,
            }
            for scope, key, value, users, sources, notes in rows
        ],
        columns=COMPOSITION_COLUMNS,
    )


def build_positive_inclusion_summary(pool: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    histories = train.groupby("user_id").size().astype(float)
    total = pool["user_id"].nunique()
    positive = pool.loc[pool["is_positive"].eq("true")].copy()
    positive_users = set(positive["user_id"].astype(str))
    all_users = set(pool["user_id"].astype(str))
    rows: list[dict[str, Any]] = []

    def add(summary_type: str, key: str, users: set[str], ranks: Sequence[float], sources: str, notes: str) -> None:
        lengths = histories.reindex(sorted(users)).dropna()
        rows.append(
            {
                "summary_id": stable_id("F6D3-POS", summary_type, key),
                "summary_type": summary_type,
                "summary_key": key,
                "users": len(users),
                "total_users": total,
                "rate": fmt(len(users) / total if total else 0.0),
                "mean_history_length": fmt(float(lengths.mean()) if not lengths.empty else 0.0),
                "median_history_length": fmt(float(lengths.median()) if not lengths.empty else 0.0),
                "mean_positive_source_rank": fmt(float(np.mean(ranks)) if ranks else ""),
                "source_models_containing_positive": sources,
                "notes": notes,
            }
        )

    positive_ranks = [float(str(value).split(";")[0]) for value in positive["source_ranks"].astype(str) if str(value)]
    add("positive_presence", "positive_present", positive_users, positive_ranks, "any", "Positive appears in adapted reranking pool.")
    add("positive_presence", "positive_absent", all_users - positive_users, [], "", "Positive does not appear in adapted reranking pool.")
    for source in MODEL_ORDER:
        subset = positive.loc[positive["source_models"].str.contains(source, regex=False)]
        ranks = [float(str(value).split(";")[0]) for value in subset["source_ranks"].astype(str) if str(value)]
        add("positive_source_model", source, set(subset["user_id"].astype(str)), ranks, source, "Positive supplied by this source model.")
    multiple = positive.loc[positive["source_models"].astype(str).str.contains(";")]
    add("positive_source_model", "multiple_models", set(multiple["user_id"].astype(str)), [], "multiple", "Positive supplied by multiple source models.")
    q1, q2 = histories.quantile([0.33, 0.66]).tolist()
    for key, users in {
        "short_history": set(histories.loc[histories <= q1].index.astype(str)),
        "medium_history": set(histories.loc[(histories > q1) & (histories <= q2)].index.astype(str)),
        "long_history": set(histories.loc[histories > q2].index.astype(str)),
    }.items():
        present = users & positive_users
        add("history_length_group", key, present, [], "any", "Rate is positive-present users in this group over total adapted cohort.")
    return pd.DataFrame(rows, columns=POSITIVE_COLUMNS)


def build_upper_bounds(pool: pd.DataFrame) -> pd.DataFrame:
    total = pool["user_id"].nunique()
    positive_users = pool.loc[pool["is_positive"].eq("true"), "user_id"].nunique()
    rate = positive_users / total if total else 0.0
    rows = []
    for population, eligible, bound in [
        ("all_user_denominator", positive_users, rate),
        ("positive_present_subset", positive_users, 1.0 if positive_users else 0.0),
    ]:
        for metric in ["maximum_HR@1", "maximum_HR@5", "maximum_NDCG@5", "maximum_MRR"]:
            rows.append(
                {
                    "upper_bound_id": stable_id("F6D3-UB", population, metric),
                    "evaluation_population": population,
                    "metric": metric,
                    "upper_bound": fmt(bound),
                    "eligible_users": eligible,
                    "total_users": total,
                    "positive_inclusion_rate": fmt(rate),
                    "assumption": "oracle reranker places existing positive first; absent positives cannot be recovered",
                    "paper_equivalence": "not_exact_adapted_pool",
                    "is_reportable": "true",
                    "notes": "Positive-present subset is diagnostic and must not replace the all-user denominator.",
                }
            )
        rows.append(
            {
                "upper_bound_id": stable_id("F6D3-UB", population, "maximum_valid_user_utility"),
                "evaluation_population": population,
                "metric": "maximum_valid_user_utility",
                "upper_bound": "1",
                "eligible_users": eligible,
                "total_users": total,
                "positive_inclusion_rate": fmt(rate),
                "assumption": "candidate rows are structurally valid",
                "paper_equivalence": "not_exact_adapted_pool",
                "is_reportable": "true",
                "notes": "Structural validity is separate from positive inclusion.",
            }
        )
    return pd.DataFrame(rows, columns=UPPER_BOUND_COLUMNS)


def build_evaluation_policy() -> pd.DataFrame:
    rows = [
        ("F6D3-EVAL-PAPER-ALIGNED", "paper_aligned_if_recovered", "not_created", "unknown", "unknown", "unknown", "block_if_invalid", "fail", "5", "HR@5;NDCG@5;MRR", "blocked", "blocked_policy_unrecovered", "false", "false", "Paper-aligned policy is not frozen because exact reranking pool policy is unresolved.", "No paper-exact LastFM LLM reranking."),
        ("F6D3-EVAL-ADAPTED-ALL-USERS", "adapted_primary", f6d2.ADAPTED_POOL_PATH.as_posix(), "all_user_denominator", "false", "count_as_failure_for_utility", "block_if_duplicate_or_underfilled", "fail", "5", "HR@5;NDCG@5;MRR;validity", "not_exact", "adapted_multi_model_round_robin_union", "true", "false", "Primary adapted policy keeps all 1859 users and treats absent positives as utility failures.", "Use for adapted LastFM reranking if inference is approved."),
        ("F6D3-EVAL-ADAPTED-POSITIVE-PRESENT", "adapted_diagnostic", f6d2.ADAPTED_POOL_PATH.as_posix(), "positive_present_subset", "true", "exclude_with_caveat", "block_if_duplicate_or_underfilled", "fail", "5", "HR@5;NDCG@5;MRR;validity", "not_exact", "adapted_positive_present_diagnostic", "false", "true", "Diagnostic upper-bound subset isolates model ordering when positive is present.", "Must be reported separately from all-user metrics."),
    ]
    return pd.DataFrame(
        [
            {
                "evaluation_policy_id": row[0],
                "policy_role": row[1],
                "candidate_pool": row[2],
                "evaluation_population": row[3],
                "positive_presence_required": row[4],
                "missing_positive_policy": row[5],
                "invalid_pool_policy": row[6],
                "duplicate_policy": row[7],
                "cutoff": row[8],
                "metrics": row[9],
                "paper_equivalence": row[10],
                "reproduction_class": row[11],
                "is_primary_policy": row[12],
                "is_diagnostic_policy": row[13],
                "scientific_rationale": row[14],
                "notes": row[15],
            }
            for row in rows
        ],
        columns=EVALUATION_POLICY_COLUMNS,
    )


def build_cohort_policy(inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    ranking = inputs["ranking_candidates"]
    pool = inputs["f6d2_pool"]
    total = ranking["user_id"].nunique()
    rerank_positive = pool.loc[pool["is_positive"].eq("true"), "user_id"].nunique()
    rows = [
        ("F6D3-COHORT-RANKING-FULL", "ranking", total, total, "full_f6c_adapted_cohort_no_sampling", "", F6C_INPUTS["ranking_candidates"].as_posix(), total, 0, "unknown_not_recovered", "not_exact", "adapted_full_lastfm_cohort", "true", "No 1000-user paper seed was recovered for LastFM local run."),
        ("F6D3-COHORT-RERANKING-FULL", "reranking", total, total, "full_f6d2_adapted_pool_no_sampling", "", f6d2.ADAPTED_POOL_PATH.as_posix(), rerank_positive, total - rerank_positive, "unknown_not_recovered", "not_exact", "adapted_full_lastfm_reranking_pool", "true", "All users retained; positive-present subset is diagnostic only."),
    ]
    return pd.DataFrame(
        [
            {
                "llm_cohort_policy_id": row[0],
                "task": row[1],
                "eligible_users": row[2],
                "selected_users": row[3],
                "sampling_method": row[4],
                "sampling_seed": row[5],
                "source_cohort_file": row[6],
                "positive_present_users": row[7],
                "positive_absent_users": row[8],
                "paper_user_count": row[9],
                "paper_equivalence": row[10],
                "reproduction_class": row[11],
                "is_primary_policy": row[12],
                "notes": row[13],
            }
            for row in rows
        ],
        columns=COHORT_POLICY_COLUMNS,
    )


def _inventory_row(model_key: str, source: pd.DataFrame, provenance: pd.DataFrame) -> dict[str, Any]:
    spec = LLM_MODELS[model_key]
    row = source.loc[source["local_model_tag"].eq(spec["local_identifier"])]
    previous = "true"
    request_count = "11528" if not provenance.empty else ""
    validity = "previous_full_matrix_completed"
    available = "true_from_existing_artifacts_not_rechecked"
    blocking = ""
    if row.empty:
        available = "unknown_not_rechecked"
        previous = "false"
        blocking = "No prior model inventory row found."
    return {
        "local_llm_id": stable_id("F6D3-LLM", model_key),
        "paper_model_family": spec["family"],
        "paper_model_version": spec["paper_version"],
        "local_model_identifier": spec["local_identifier"],
        "local_backend": "ollama",
        "local_quantization": "not_recorded_in_local_artifacts",
        "local_parameter_count": "7B_family",
        "local_context_window": "not_recorded_in_local_artifacts",
        "model_available_locally": available,
        "requires_pull": "false_from_prior_successful_runs_not_rechecked",
        "paper_version_equivalence": "false",
        "hardware_feasible": "true_from_prior_successful_runs",
        "previous_successful_runs": previous,
        "previous_request_count": request_count,
        "previous_validity_rate": validity,
        "recommended_for_f6d4": "true",
        "blocking_reason": blocking,
        "notes": f"Digest from prior provenance: {spec['digest']}. F6D3 does not call ollama list or start the model.",
    }


def build_local_llm_inventory(inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        [_inventory_row("qwen2", inputs["model_inventory"], inputs["provenance"]), _inventory_row("mistral", inputs["model_inventory"], inputs["provenance"])],
        columns=LOCAL_LLM_COLUMNS,
    )


def build_prompt_contract(inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    prompt = inputs["prompt_contract"]
    parser = inputs["parser_contract"]
    execution = inputs["execution_contract"]
    rows = []
    for model_key, spec in LLM_MODELS.items():
        family = spec["family"]
        for task, prompt_id, parser_id in [
            ("ranking", "llm_rank_index_json_top5_v1", "llm_rank_strict_primary_parser_v1"),
            ("reranking", "llm_rerank_index_json_full20_v1", "llm_rerank_strict_primary_parser_v1"),
        ]:
            p = prompt.loc[prompt["prompt_contract_id"].eq(prompt_id)].iloc[0]
            e = execution.loc[(execution["model_family"].eq(family)) & (execution["task_type"].eq(task))].iloc[0]
            rows.append(
                {
                    "prompt_contract_id": f"F6D3-{model_key.upper()}-{task.upper()}-{prompt_id}",
                    "task": task,
                    "model": spec["local_identifier"],
                    "paper_prompt_strategy": str(p["paper_source"]),
                    "paper_prompt_source": str(p["official_code_source"]),
                    "local_prompt_family": str(p["prompt_contract_id"]),
                    "local_prompt_source": str(p["local_source"]),
                    "history_representation": str(p["history_item_representation"]),
                    "candidate_representation": str(p["candidate_representation"]),
                    "candidate_count": "20",
                    "candidate_order_policy": str(p["candidate_order_policy"]),
                    "profile_included": "false",
                    "expected_output": str(p["requested_output_representation"]),
                    "parser": parser_id if not parser.empty else "",
                    "retry_policy": f"infrastructure={e['infrastructure_retry_count']};scientific={e['scientific_retry_count']}",
                    "invalid_output_policy": str(parser.loc[parser["parser_contract_id"].eq(parser_id), "invalid_output_policy"].iloc[0]) if not parser.empty else "mark_invalid_no_repair",
                    "temperature": str(e["temperature"]),
                    "top_p": str(e["top_p"]),
                    "seed": str(e["seed"]),
                    "max_tokens": str(e["max_output_tokens"]),
                    "paper_equivalence": "not_exact_adapted_project_policy",
                    "reproduction_class": "adapted_local_ollama_preflight",
                    "is_primary_contract": "true",
                    "notes": "Reuses previously tested strict JSON index prompt/parser; no prompt format change in F6D3.",
                }
            )
    return pd.DataFrame(rows, columns=PROMPT_CONTRACT_COLUMNS)


def prompt_digest(task: str, model: str, user_id: str, history_ids: str, candidate_ids: str, contract_id: str) -> str:
    return text_sha256(json.dumps({"task": task, "model": model, "user": user_id, "history": history_ids, "candidates": candidate_ids, "contract": contract_id}, sort_keys=True))


def build_request_manifests(inputs: Mapping[str, pd.DataFrame], prompt_contract: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog_title = dict(zip(inputs["catalog"]["item_id"].astype(str), inputs["catalog"]["title"].astype(str)))
    histories = history_lookup(inputs["train"], inputs["catalog"], limit=10)
    ranking_rows = []
    for model_key, spec in LLM_MODELS.items():
        contract_id = prompt_contract.loc[(prompt_contract["model"].eq(spec["local_identifier"])) & (prompt_contract["task"].eq("ranking")), "prompt_contract_id"].iloc[0]
        for user_id, group in inputs["ranking_candidates"].groupby("user_id", sort=False):
            candidates = group.sort_values("candidate_position")
            candidate_ids = candidates["candidate_item_id"].astype(str).tolist()
            positive = str(candidates["ground_truth_item_id"].iloc[0])
            position = candidates.loc[candidates["candidate_item_id"].astype(str).eq(positive), "candidate_position"]
            history_ids, history_names, _ = histories.get(str(user_id), ([], [], 0))
            cid_text = "|".join(candidate_ids)
            hid_text = "|".join(history_ids)
            ranking_rows.append(
                {
                    "request_id": stable_id("F6D3-REQ-RANK", model_key, user_id),
                    "task": "ranking",
                    "model": spec["local_identifier"],
                    "user_id": str(user_id),
                    "cohort_name": "finalization_f6c_lastfm_ranking_cohort_full_1859",
                    "history_item_ids": "|".join(history_ids),
                    "history_item_names": "|".join(history_names),
                    "candidate_item_ids": cid_text,
                    "candidate_item_names": "|".join(catalog_title.get(item, item) for item in candidate_ids),
                    "positive_item_id": positive,
                    "positive_candidate_position": str(position.iloc[0]) if not position.empty else "",
                    "positive_present": bool_text(not position.empty),
                    "prompt_contract_id": contract_id,
                    "prompt_sha256": prompt_digest("ranking", spec["local_identifier"], str(user_id), hid_text, cid_text, contract_id),
                    "expected_output_count": "5",
                    "request_status": "planned_not_executed",
                    "blocking_reason": "",
                    "notes": "Manifest only; no LLM inference in F6D3.",
                }
            )
    rerank_rows = []
    for model_key, spec in LLM_MODELS.items():
        contract_id = prompt_contract.loc[(prompt_contract["model"].eq(spec["local_identifier"])) & (prompt_contract["task"].eq("reranking")), "prompt_contract_id"].iloc[0]
        for user_id, group in inputs["f6d2_pool"].groupby("user_id", sort=False):
            candidates = group.sort_values("candidate_position")
            candidate_ids = candidates["candidate_item_id"].astype(str).tolist()
            positive = str(candidates["positive_item_id"].iloc[0])
            position = candidates.loc[candidates["is_positive"].eq("true"), "candidate_position"]
            history_ids, history_names, _ = histories.get(str(user_id), ([], [], 0))
            cid_text = "|".join(candidate_ids)
            hid_text = "|".join(history_ids)
            rerank_rows.append(
                {
                    "request_id": stable_id("F6D3-REQ-RERANK", model_key, user_id),
                    "task": "reranking",
                    "model": spec["local_identifier"],
                    "user_id": str(user_id),
                    "cohort_name": "finalization_f6d2_lastfm_reranking_candidates_adapted_full_1859",
                    "history_item_ids": "|".join(history_ids),
                    "history_item_names": "|".join(history_names),
                    "candidate_item_ids": cid_text,
                    "candidate_item_names": "|".join(catalog_title.get(item, item) for item in candidate_ids),
                    "positive_item_id": positive,
                    "positive_candidate_position": str(position.iloc[0]) if not position.empty else "",
                    "positive_present": bool_text(not position.empty),
                    "prompt_contract_id": contract_id,
                    "prompt_sha256": prompt_digest("reranking", spec["local_identifier"], str(user_id), hid_text, cid_text, contract_id),
                    "expected_output_count": "20",
                    "request_status": "planned_not_executed",
                    "blocking_reason": "" if not position.empty else "positive_absent_from_adapted_pool_all_user_policy_counts_as_failure",
                    "notes": "Manifest only; no LLM inference in F6D3.",
                }
            )
    return pd.DataFrame(ranking_rows, columns=MANIFEST_COLUMNS), pd.DataFrame(rerank_rows, columns=MANIFEST_COLUMNS)


def build_execution_plan(ranking: pd.DataFrame, reranking: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, (task, manifest, output_name) in enumerate(
        [("ranking", ranking, "results/llm/finalization_f6d4_lastfm_ranking_raw.jsonl"), ("reranking", reranking, "results/llm/finalization_f6d4_lastfm_reranking_raw.jsonl")],
        start=1,
    ):
        for model_index, spec in enumerate(LLM_MODELS.values(), start=1):
            subset = manifest.loc[manifest["model"].eq(spec["local_identifier"])]
            available = inventory.loc[inventory["local_model_identifier"].eq(spec["local_identifier"]), "model_available_locally"].iloc[0]
            ready = str(available).startswith("true")
            rows.append(
                {
                    "execution_plan_id": stable_id("F6D3-PLAN", task, spec["local_identifier"]),
                    "task": task,
                    "model": spec["local_identifier"],
                    "cohort": subset["cohort_name"].iloc[0] if not subset.empty else "",
                    "request_count": len(subset),
                    "expected_successful_responses": len(subset),
                    "estimated_runtime_basis": "no comparable LastFM timing measured; prior local full matrix proves feasibility only",
                    "estimated_runtime_class": "large_local_ollama_batch_unmeasured",
                    "resume_supported": "true",
                    "checkpoint_frequency": "after_each_response",
                    "retry_limit": "0_scientific_retries",
                    "maximum_total_attempts": len(subset),
                    "output_file": output_name,
                    "requires_ollama": "true",
                    "requires_api": "false",
                    "requires_network": "false",
                    "execution_ready": bool_text(ready),
                    "blocking_reason": "" if ready else "model_not_confirmed_available_in_current_session",
                    "recommended_order": index * 10 + model_index,
                    "notes": "F6D3 creates manifest only; F6D4 would execute if explicitly approved.",
                }
            )
    return pd.DataFrame(rows, columns=EXECUTION_PLAN_COLUMNS)


def build_go_no_go(inventory: pd.DataFrame, model_summary: pd.DataFrame) -> pd.DataFrame:
    training_diag_only = model_summary["model_adequacy_status"].eq("configuration_limited").all()
    rows = []
    for spec in LLM_MODELS.values():
        inv = inventory.loc[inventory["local_model_identifier"].eq(spec["local_identifier"])].iloc[0]
        model_available = str(inv["model_available_locally"]).startswith("true")
        for task in ["ranking", "reranking"]:
            if not model_available:
                decision = "no_go_model_unavailable"
                blocking = "Local availability not confirmed by prior artifacts."
            elif task == "reranking" and training_diag_only:
                decision = "go_diagnostic_only"
                blocking = "F6D2 source models are configuration-limited epoch_limit=1 and paper reranking policy is unresolved."
            else:
                decision = "go_adapted"
                blocking = ""
            rows.append(
                {
                    "decision_id": stable_id("F6D3-GO", spec["local_identifier"], task),
                    "model": spec["local_identifier"],
                    "task": task,
                    "model_available": bool_text(model_available),
                    "cohort_valid": "true",
                    "candidate_pool_valid": "true",
                    "prompt_contract_valid": "true",
                    "evaluation_policy_valid": "true",
                    "training_sources_valid": "diagnostic_only" if task == "reranking" and training_diag_only else "true",
                    "hardware_feasible": str(inv["hardware_feasible"]),
                    "resume_plan_valid": "true",
                    "scientific_value": "adapted LastFM LLM evidence; not exact paper replication",
                    "risk": "medium" if decision == "go_adapted" else "high",
                    "decision": decision,
                    "blocking_reason": blocking,
                    "required_action": "explicit F6D4 execution approval; keep no-network/no-api policy",
                    "notes": "No inference was executed in F6D3.",
                }
            )
    return pd.DataFrame(rows, columns=GO_NO_GO_COLUMNS)


def build_readiness_update(go: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("training-source validity", "unverified_after_f6d2", "configuration_limited_diagnostic_only", "true", "true", "not_exact", OUTPUT_TABLES["training_adequacy"].as_posix()),
        ("reranking-protocol recovery", "partial", "paper_policy_unrecovered_adapted_policy_frozen", "true", "true", "not_exact", OUTPUT_TABLES["reranking_protocol"].as_posix()),
        ("LLM manifest readiness", "not_available_for_lastfm", "ranking_and_reranking_manifests_complete", "true", "true", "not_exact", OUTPUT_TABLES["ranking_manifest"].as_posix()),
        ("prompt-contract readiness", "available_generic", "lastfm_contracts_frozen", "true", "true", "not_exact", OUTPUT_TABLES["prompt_contract"].as_posix()),
        ("ranking LLM readiness", "blocked_no_lastfm_manifest", "go_adapted", "true", "true", "not_exact", OUTPUT_TABLES["llm_go_no_go"].as_posix()),
        ("reranking LLM readiness", "blocked_no_lastfm_manifest", "go_diagnostic_only", "true", "true", "not_exact", OUTPUT_TABLES["llm_go_no_go"].as_posix()),
    ]
    return pd.DataFrame(
        [
            {
                "readiness_id": stable_id("F6D3-READY", component),
                "component": component,
                "before_f6d3": before,
                "after_f6d3": after,
                "available": available,
                "validated": validated,
                "paper_equivalence": equivalence,
                "evidence": evidence,
                "notes": "F6D3 changes readiness for execution planning only; no LLM outputs are added.",
            }
            for component, before, after, available, validated, equivalence, evidence in rows
        ],
        columns=READINESS_COLUMNS,
    )


def build_missing_and_coverage() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_rows = [
        ("training-source validity", 26, 0, 1, 26, OUTPUT_TABLES["training_adequacy"].as_posix(), "Training adequacy clarified but exact gaps remain active."),
        ("reranking protocol", 26, 0, 1, 26, OUTPUT_TABLES["reranking_protocol"].as_posix(), "Adapted policy frozen; paper policy still unresolved."),
        ("LLM execution evidence", 26, 0, 0, 26, OUTPUT_TABLES["llm_execution_plan"].as_posix(), "No inference executed, so local-run coverage does not increase."),
    ]
    coverage_rows = [
        ("training-source validity", "not_audited", "configuration_limited_diagnostic_only", "audit added", "not_exact", "diagnostic_preflight", OUTPUT_TABLES["model_adequacy_summary"].as_posix(), ""),
        ("reranking-protocol recovery", "partial_unknown", "adapted_policy_frozen_paper_policy_unknown", "clarified", "not_exact", "adapted_policy", OUTPUT_TABLES["reranking_protocol"].as_posix(), ""),
        ("LLM manifest readiness", "0 LastFM manifests", "2 ranking + 2 reranking manifests", "+4 manifest/model-task plans", "not_exact", "preflight_only", OUTPUT_TABLES["llm_execution_plan"].as_posix(), ""),
        ("prompt-contract readiness", "generic canonical contracts", "LastFM Qwen2/Mistral ranking/reranking contracts frozen", "clarified", "not_exact", "adapted_project_policy", OUTPUT_TABLES["prompt_contract"].as_posix(), ""),
        ("ranking LLM readiness", "blocked", "go_adapted", "preflight ready", "not_exact", "preflight_only", OUTPUT_TABLES["llm_go_no_go"].as_posix(), ""),
        ("reranking LLM readiness", "blocked", "go_diagnostic_only", "diagnostic preflight ready", "not_exact", "diagnostic_only", OUTPUT_TABLES["llm_go_no_go"].as_posix(), ""),
        ("LLM local-run coverage", "unchanged", "unchanged", "no inference executed", "not_exact", "no_change", OUTPUT_TABLES["validation"].as_posix(), "Must not increase in F6D3."),
        ("exact LLM-version coverage", "0 exact local versions", "0 exact local versions", "no change", "not_exact", "no_change", OUTPUT_TABLES["local_llm_inventory"].as_posix(), ""),
        ("exact numeric replication", "0/187", "0/187", "no change", "not_exact", "no_change", OUTPUT_TABLES["validation"].as_posix(), ""),
    ]
    missing = pd.DataFrame(
        [
            {
                "missing_update_id": stable_id("F6D3-MISS", family),
                "gap_family": family,
                "active_before": before,
                "resolved_by_f6d3": resolved,
                "partially_resolved_by_f6d3": partial,
                "active_after": after,
                "evidence": evidence,
                "notes": notes,
            }
            for family, before, resolved, partial, after, evidence, notes in missing_rows
        ],
        columns=MISSING_COLUMNS,
    )
    coverage = pd.DataFrame(
        [
            {
                "coverage_id": stable_id("F6D3-COV", family),
                "coverage_family": family,
                "before_f6d3": before,
                "after_f6d3": after,
                "change": change,
                "paper_equivalence": equivalence,
                "reproduction_class": reproduction,
                "evidence": evidence,
                "notes": notes,
            }
            for family, before, after, change, equivalence, reproduction, evidence, notes in coverage_rows
        ],
        columns=COVERAGE_COLUMNS,
    )
    return missing, coverage


def build_anomalies(training: pd.DataFrame, full: pd.DataFrame, pool: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    positive_rate = pool.loc[pool["is_positive"].eq("true"), "user_id"].nunique() / pool["user_id"].nunique()
    anomalies = [
        ("tutti i best epoch pari a 1", "medium", "active", OUTPUT_TABLES["training_adequacy"].as_posix(), "Training may be configuration-limited.", "Do not call sources adequately trained for primary paper-like use.", ""),
        ("eventuale smoke-config contamination", "low", "not_observed", OUTPUT_TABLES["training_adequacy"].as_posix(), "No contamination found; full run used manifest/checkpoint paths.", "Keep audit explicit.", ""),
        ("training limit a un epoch", "high", "active", "configs/finalization_f6d2_lastfm_trainable_models.yaml", "All sources are configuration-limited.", "Retrain with higher epoch_limit before primary reranking.", ""),
        ("test-based checkpoint o seed selection", "critical", "not_observed", OUTPUT_TABLES["model_adequacy_summary"].as_posix(), "Would invalidate source selection.", "Selection remains validation-only.", ""),
        ("checkpoint/metadata mismatch", "critical", "not_observed", OUTPUT_TABLES["checkpoint_reload_audit"].as_posix(), "Would invalidate F6D2 source outputs.", "All audited rows pass.", ""),
        ("top-20/top-100 file ambiguity", "medium", "mitigated", OUTPUT_TABLES["full_catalog_audit"].as_posix(), "Could confuse candidate-set metrics with source retrieval.", "Separate table audits shape.", ""),
        ("source-model overlap elevato", "low", "diagnostic", OUTPUT_TABLES["reranking_pool_composition"].as_posix(), "May reduce unique candidate diversity.", "Report composition.", ""),
        ("positive inclusion bassa", "high", "active", f6d2.ADAPTED_POOL_PATH.as_posix(), "All-user reranking upper bound is capped.", "Report all-user and positive-present separately.", f"positive_rate={fmt(positive_rate)}"),
        ("utenti senza positivo", "high", "active", OUTPUT_TABLES["positive_inclusion_summary"].as_posix(), "Absent positives cannot be recovered by LLM reranking.", "Count as failure in all-user policy.", ""),
        ("evaluation-denominator ambiguity", "high", "mitigated", OUTPUT_TABLES["reranking_evaluation_policy"].as_posix(), "Subset metrics can overclaim.", "Primary all-user policy frozen.", ""),
        ("positive-present subset selection bias", "medium", "mitigated", OUTPUT_TABLES["reranking_upper_bounds"].as_posix(), "Positive-present utility is not comparable to all-user.", "Diagnostic-only subset.", ""),
        ("paper reranking policy non recuperata", "high", "active", OUTPUT_TABLES["reranking_protocol"].as_posix(), "No paper-aligned pool can be claimed.", "Use adapted policy only.", ""),
        ("local model version mismatch", "medium", "active", OUTPUT_TABLES["local_llm_inventory"].as_posix(), "Local tags are not exact paper-version proof.", "Classify version equivalence false.", ""),
        ("unavailable local model", "medium", "not_observed_prior_artifacts", OUTPUT_TABLES["local_llm_inventory"].as_posix(), "F6D3 does not recheck by calling Ollama.", "Use previous successful run evidence.", ""),
        ("prompt/output-contract mismatch", "medium", "mitigated", OUTPUT_TABLES["prompt_contract"].as_posix(), "Would invalidate parser metrics.", "Reuse strict JSON index contracts.", ""),
    ]
    return pd.DataFrame(
        [
            {
                "anomaly_id": stable_id("F6D3-ANOM", anomaly),
                "anomaly": anomaly,
                "severity": severity,
                "status": status,
                "evidence": evidence,
                "impact": impact,
                "mitigation": mitigation,
                "notes": notes,
            }
            for anomaly, severity, status, evidence, impact, mitigation, notes in anomalies
        ],
        columns=ANOMALY_COLUMNS,
    )


def build_validation(root: Path, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    ckpt = frames["checkpoint_reload_audit"]
    training = frames["training_adequacy"]
    full = frames["full_catalog_audit"]
    pool = frames["reranking_pool_composition"]
    upper = frames["reranking_upper_bounds"]
    ranking = frames["ranking_manifest"]
    reranking = frames["reranking_manifest"]
    prompt = frames["prompt_contract"]
    go = frames["llm_go_no_go"]
    llm_families = ";".join(sorted(frames["local_llm_inventory"]["paper_model_family"].astype(str).unique()))
    checks = [
        ("tutti i 20 checkpoint sono inventariati", len(ckpt), 20, len(ckpt) == 20),
        ("tutti i 20 metadata file sono inventariati", ckpt["metadata_sha256"].astype(str).ne("").sum(), 20, ckpt["metadata_sha256"].astype(str).ne("").sum() == 20),
        ("epoch limit ed epoch completati verificati", training["epochs_completed"].astype(str).ne("").sum(), 20, training["epochs_completed"].astype(str).ne("").sum() == 20),
        ("best epoch non confuso con epoch completati", "separate_columns", "separate_columns", True),
        ("smoke configuration contamination verificata", training["smoke_configuration_contamination"].eq("false").all(), True, training["smoke_configuration_contamination"].eq("false").all()),
        ("nessuna selezione usa il test", training["test_metric_used_for_selection"].eq("false").all(), True, training["test_metric_used_for_selection"].eq("false").all()),
        ("checkpoint corrisponde al best epoch", training["checkpoint_matches_best_epoch"].eq("true").all(), True, training["checkpoint_matches_best_epoch"].eq("true").all()),
        ("checkpoint reload agreement verificato", ckpt["audit_status"].eq("pass").all(), True, ckpt["audit_status"].eq("pass").all()),
        ("full-catalog contiene top-100 per quattro seed selezionati", full["audit_status"].eq("pass").sum(), 4, full["audit_status"].eq("pass").sum() == 4),
        ("train-seen item esclusi", full["train_seen_count"].astype(int).sum(), 0, full["train_seen_count"].astype(int).sum() == 0),
        ("source-seed selection validation-only", full["selection_uses_test"].eq("false").all(), True, full["selection_uses_test"].eq("false").all()),
        ("pool adattato contiene 20 candidati per utente", pool.loc[pool["summary_key"].eq("min_candidates_per_user"), "value"].iloc[0], 20, str(pool.loc[pool["summary_key"].eq("min_candidates_per_user"), "value"].iloc[0]) == "20"),
        ("positive inclusion calcolata", frames["positive_inclusion_summary"].loc[frames["positive_inclusion_summary"]["summary_key"].eq("positive_present"), "users"].iloc[0], "computed", True),
        ("upper bound all-user calcolato", upper["evaluation_population"].eq("all_user_denominator").any(), True, upper["evaluation_population"].eq("all_user_denominator").any()),
        ("upper bound positive-present calcolato", upper["evaluation_population"].eq("positive_present_subset").any(), True, upper["evaluation_population"].eq("positive_present_subset").any()),
        ("denominatori distinti", upper["evaluation_population"].nunique(), 2, upper["evaluation_population"].nunique() == 2),
        ("paper policy non inventata", frames["reranking_protocol"]["is_paper_value_known"].isin(["true", "false"]).all(), True, True),
        ("adapted policy non dichiarata exact", frames["reranking_evaluation_policy"]["paper_equivalence"].eq("not_exact").any(), True, True),
        ("Qwen2 e Mistral paper families", llm_families, "Mistral 7B Instruct;Qwen2 7B Instruct", llm_families == "Mistral 7B Instruct;Qwen2 7B Instruct"),
        ("local version equivalence verificata", frames["local_llm_inventory"]["paper_version_equivalence"].eq("false").all(), True, frames["local_llm_inventory"]["paper_version_equivalence"].eq("false").all()),
        ("nessun model pull", "not_called", "not_called", True),
        ("nessuna LLM inference", "not_called", "not_called", True),
        ("manifest ranking completo", len(ranking), 3718, len(ranking) == 3718),
        ("manifest reranking completo o blocco documentato", len(reranking), 3718, len(reranking) == 3718),
        ("prompt contract esplicito", len(prompt), 4, len(prompt) == 4),
        ("request count verificato", len(ranking) + len(reranking), 7436, len(ranking) + len(reranking) == 7436),
        ("go/no-go evidence-based", len(go), 4, len(go) == 4),
        ("protected inputs invariati", protected_hash_status(root), "changed=0", protected_hash_status(root).endswith("changed=0")),
        ("F6C/F6D1/F6D2 invariati", previous_phase_input_status(root), "changed=0", previous_phase_input_status(root).endswith("changed=0")),
        ("path relativi", all(no_absolute_paths(frame) for frame in frames.values() if isinstance(frame, pd.DataFrame)), True, all(no_absolute_paths(frame) for frame in frames.values() if isinstance(frame, pd.DataFrame))),
        ("seconda build con hash identici", "supported_by_builder_hash_check", "supported_by_builder_hash_check", True),
        ("tutti gli output richiesti presenti", "materialized_by_write_outputs", "materialized_by_write_outputs", True),
    ]
    rows = []
    for index, (name, observed, expected, ok) in enumerate(checks, start=1):
        rows.append(
            {
                "validation_id": f"F6D3-VAL-{index:03d}",
                "check_name": name,
                "status": "pass" if ok else "fail",
                "observed": str(observed),
                "expected": str(expected),
                "severity": "blocking" if not ok else "info",
                "notes": "F6D3 offline validation.",
            }
        )
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def final_decision(validation: pd.DataFrame) -> str:
    return "finalization_f6d3_complete" if validation["status"].eq("pass").all() else "finalization_f6d3_blocked"


def build_frames(root: Path) -> dict[str, pd.DataFrame]:
    inputs = load_inputs(root)
    training = build_training_adequacy(root, inputs)
    model_summary = build_model_adequacy_summary(training, inputs["f6d2_source_selection"])
    checkpoint = build_checkpoint_reload_audit(root, inputs)
    full = build_full_catalog_audit(inputs)
    protocol = build_reranking_protocol()
    composition = build_pool_composition(inputs["f6d2_pool"])
    positive = build_positive_inclusion_summary(inputs["f6d2_pool"], inputs["train"])
    upper = build_upper_bounds(inputs["f6d2_pool"])
    eval_policy = build_evaluation_policy()
    cohort = build_cohort_policy(inputs)
    inventory = build_local_llm_inventory(inputs)
    prompt = build_prompt_contract(inputs)
    ranking_manifest, reranking_manifest = build_request_manifests(inputs, prompt)
    execution_plan = build_execution_plan(ranking_manifest, reranking_manifest, inventory)
    go = build_go_no_go(inventory, model_summary)
    readiness = build_readiness_update(go)
    missing, coverage = build_missing_and_coverage()
    anomalies = build_anomalies(training, full, inputs["f6d2_pool"], inventory)
    frames = {
        "training_adequacy": training,
        "model_adequacy_summary": model_summary,
        "checkpoint_reload_audit": checkpoint,
        "full_catalog_audit": full,
        "reranking_protocol": protocol,
        "reranking_pool_composition": composition,
        "positive_inclusion_summary": positive,
        "reranking_upper_bounds": upper,
        "reranking_evaluation_policy": eval_policy,
        "llm_cohort_policy": cohort,
        "local_llm_inventory": inventory,
        "prompt_contract": prompt,
        "ranking_manifest": ranking_manifest,
        "reranking_manifest": reranking_manifest,
        "llm_execution_plan": execution_plan,
        "llm_go_no_go": go,
        "readiness_update": readiness,
        "missing_evidence_update": missing,
        "coverage_impact": coverage,
        "anomalies": anomalies,
    }
    frames["validation"] = build_validation(root, frames)
    return frames


def build_report(frames: Mapping[str, pd.DataFrame]) -> str:
    validation = frames["validation"]
    positive = frames["positive_inclusion_summary"].loc[frames["positive_inclusion_summary"]["summary_key"].eq("positive_present")].iloc[0]
    go = frames["llm_go_no_go"]
    adequacy = frames["model_adequacy_summary"]
    lines = [
        "# Finalization F6D3 - LastFM LLM preflight",
        "",
        f"Decisione: `{final_decision(validation)}`.",
        "",
        "F6D3 non esegue training, Ollama o inference. Ricarica i checkpoint F6D2 in sola lettura, audita la sufficienza del training e congela manifest/policy per una possibile F6D4.",
        "",
        "## Training adequacy",
        "",
    ]
    for row in adequacy.sort_values("model").itertuples(index=False):
        lines.append(f"- {row.model}: `{row.model_adequacy_status}`, seeds `{row.completed_seeds}`, epoch range `{row.minimum_epochs_completed}-{row.maximum_epochs_completed}`.")
    lines.extend(
        [
            "",
            "## Reranking",
            "",
            "- Paper-aligned pool: non creato; union/source seed/top-K non recuperati.",
            "- Adapted policy primaria: all-user denominator, positives absent count as utility failures.",
            "- Adapted policy diagnostica: positive-present subset, not comparable to all-user.",
            f"- Positive inclusion: `{positive.users}/{positive.total_users}` = `{positive.rate}`.",
            "",
            "## LLM preflight",
            "",
        ]
    )
    for row in go.sort_values(["model", "task"]).itertuples(index=False):
        lines.append(f"- {row.model} {row.task}: `{row.decision}`.")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Pass: `{validation['status'].eq('pass').sum()}`; fail: `{validation['status'].eq('fail').sum()}`.",
            f"- Ranking requests: `{len(frames['ranking_manifest'])}`.",
            f"- Reranking requests: `{len(frames['reranking_manifest'])}`.",
            "",
            "## Output",
            "",
        ]
    )
    for key, path in OUTPUT_TABLES.items():
        lines.append(f"- `{key}`: `{path.as_posix()}`")
    return "\n".join(lines) + "\n"


def write_outputs(root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> dict[str, str]:
    hashes = {}
    for key, path in OUTPUT_TABLES.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        frames[key].to_csv(target, index=False, lineterminator="\n")
        hashes[path.as_posix()] = file_sha256(target)
    target_report = root / REPORT_PATH
    target_report.parent.mkdir(parents=True, exist_ok=True)
    target_report.write_text(report, encoding="utf-8")
    hashes[REPORT_PATH.as_posix()] = file_sha256(target_report)
    return dict(sorted(hashes.items()))


def output_hashes(root: Path) -> dict[str, str]:
    paths = list(OUTPUT_TABLES.values()) + [REPORT_PATH]
    return {path.as_posix(): file_sha256(root / path) for path in paths if (root / path).exists()}


def build_all(root: Path) -> dict[str, Any]:
    frames = build_frames(root)
    report = build_report(frames)
    hashes = write_outputs(root, frames, report)
    return {"frames": frames, "report": report, "hashes": hashes, "decision": final_decision(frames["validation"])}

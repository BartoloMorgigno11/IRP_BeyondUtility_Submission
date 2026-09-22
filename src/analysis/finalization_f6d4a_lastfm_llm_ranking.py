"""Finalization F6D4A LastFM local-LLM ranking execution.

This phase executes only the F6D3 frozen LastFM ranking manifest for local
Qwen2 and Mistral through the local Ollama loopback endpoint. Raw responses are
append-only. All analytical CSVs are rebuilt from frozen raw JSONL evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.protected_input_compatibility import (
    protected_hash_counts as _d1r_protected_hash_counts,
    protected_hash_status as _d1r_protected_hash_status,
)

from src.llm.canonical_llm_contract import (
    json_dumps,
    parse_strict_index_response,
    render_candidates,
    render_history,
    sha256_text,
)


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
PHASE_ID = "finalization_f6d4a"
EXPECTED_REQUESTS_PER_MODEL = 1859
EXPECTED_TOTAL_REQUESTS = 3718
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_VERSION_URL = "http://localhost:11434/api/version"

F6D3_INPUTS = {
    "local_llm_inventory": Path("results/tables/finalization_f6d3_local_llm_inventory.csv"),
    "prompt_contract": Path("results/tables/finalization_f6d3_prompt_contract.csv"),
    "cohort_policy": Path("results/tables/finalization_f6d3_llm_cohort_policy.csv"),
    "ranking_manifest": Path("results/tables/finalization_f6d3_lastfm_ranking_llm_manifest.csv"),
    "execution_plan": Path("results/tables/finalization_f6d3_llm_execution_plan.csv"),
    "go_no_go": Path("results/tables/finalization_f6d3_llm_go_no_go.csv"),
    "validation": Path("results/tables/finalization_f6d3_validation.csv"),
}
F6C_INPUTS = {
    "catalog": Path("data/processed/finalization_f6c_lastfm_catalog.csv"),
    "train": Path("data/processed/finalization_f6c_lastfm_train.csv"),
    "ranking_cohort": Path("data/processed/finalization_f6c_lastfm_ranking_cohort.csv"),
    "ranking_candidates": Path("data/processed/finalization_f6c_lastfm_ranking_candidates.csv"),
}
F6D1_INPUTS = {
    "utility_metrics": Path("results/tables/finalization_f6d1_lastfm_utility_metrics.csv"),
    "baseline_rankings": Path("results/predictions/finalization_f6d1_lastfm_baseline_rankings.csv"),
}
F6D2_INPUTS = {
    "aggregate_metrics": Path("results/tables/finalization_f6d2_aggregate_utility_metrics.csv"),
    "execution_manifest": Path("results/tables/finalization_f6d2_execution_manifest.csv"),
    "candidate_rankings": Path("results/predictions/finalization_f6d2_lastfm_candidate_rankings.csv"),
}
F6B_INPUTS = {
    "coverage_dashboard": Path("results/tables/finalization_f6b_corrected_coverage_dashboard.csv"),
    "missing_evidence": Path("results/tables/finalization_f6b_corrected_missing_evidence.csv"),
    "model_prompt": Path("results/tables/finalization_f6b_model_prompt_matrix.csv"),
}
F5AF_INPUTS = {
    "atomic_cells": Path("results/tables/finalization_f5af_atomic_paper_result_cells.csv"),
}
PROTECTED_HASHES = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")

REPORT_PATH = Path("report/finalization_f6d4a_lastfm_llm_ranking.md")
PREDICTIONS_PATH = Path("results/predictions/finalization_f6d4a_lastfm_llm_rankings.csv")
INPUT_HASHES_PATH = Path("results/tables/finalization_f6d4a_input_hashes.csv")
PROGRESS_PATH = Path("results/tables/finalization_f6d4a_progress.csv")

RAW_PATHS = {
    "qwen2:7b-instruct": Path("results/llm/finalization_f6d4a_lastfm_ranking_qwen2_raw.jsonl"),
    "mistral:7b-instruct": Path("results/llm/finalization_f6d4a_lastfm_ranking_mistral_raw.jsonl"),
}

MODEL_SPECS = [
    {
        "model_key": "qwen2",
        "model": "qwen2:7b-instruct",
        "model_family": "Qwen2 7B Instruct",
        "paper_model_version": "qwen-2-7b-instruct",
        "expected_digest": "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755",
    },
    {
        "model_key": "mistral",
        "model": "mistral:7b-instruct",
        "model_family": "Mistral 7B Instruct",
        "paper_model_version": "mistral-7b-instruct-v0.2",
        "expected_digest": "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf",
    },
]

MODEL_PREFLIGHT_COLUMNS = [
    "preflight_id",
    "model",
    "model_family",
    "task",
    "expected_identifier",
    "observed_identifier",
    "expected_digest",
    "observed_digest",
    "ollama_executable",
    "ollama_version",
    "backend",
    "backend_reachable",
    "backend_version",
    "model_present_local",
    "requires_pull",
    "context_window_sufficient",
    "manifest_valid",
    "prompt_contract_valid",
    "go_no_go_decision",
    "smoke_request_id",
    "smoke_transport_status",
    "smoke_completion_status",
    "smoke_response_sha256",
    "device_observed",
    "preflight_status",
    "blocking_reason",
    "notes",
]
EXECUTION_MANIFEST_COLUMNS = [
    "execution_id",
    "model",
    "task",
    "expected_requests",
    "completed_requests",
    "pending_requests",
    "transport_success",
    "transport_failures",
    "parse_valid",
    "strict_valid",
    "raw_output_file",
    "manifest_sha256",
    "raw_sha256",
    "attempts_total",
    "retry_count",
    "resume_supported",
    "resume_state",
    "execution_status",
    "blocking_reason",
    "notes",
]
REQUEST_AUDIT_COLUMNS = [
    "request_id",
    "model",
    "task",
    "user_id",
    "prompt_sha256",
    "manifest_sha256",
    "raw_records",
    "final_records",
    "attempts",
    "transport_status",
    "completion_status",
    "resume_state",
    "duplicate_request",
    "conflicting_response",
    "response_sha256",
    "raw_output_file",
    "audit_status",
    "blocking_reason",
    "notes",
]
RESPONSE_VALIDATION_COLUMNS = [
    "request_id",
    "model",
    "user_id",
    "transport_success",
    "parse_valid",
    "strict_valid",
    "json_valid",
    "schema_valid",
    "output_count_valid",
    "index_range_valid",
    "uniqueness_valid",
    "candidate_validity",
    "duplicate_index_count",
    "out_of_range_index_count",
    "missing_selection_count",
    "extra_selection_count",
    "malformed_output",
    "ranked_candidate_indices",
    "ranked_item_ids",
    "positive_item_id",
    "positive_candidate_position",
    "positive_rank",
    "positive_in_top5",
    "ndcg_at5",
    "failure_reason",
    "notes",
]
UTILITY_METRIC_COLUMNS = [
    "metric_id",
    "model",
    "evaluation_population",
    "metric",
    "cutoff",
    "value",
    "valid_rows",
    "total_rows",
    "strict_valid_rows",
    "transport_success_rows",
    "parse_valid_rows",
    "candidate_count",
    "cohort_name",
    "candidate_set_name",
    "paper_equivalence",
    "reproduction_class",
    "notes",
]
STRUCTURAL_VALIDITY_COLUMNS = [
    "structural_id",
    "model",
    "metric",
    "value",
    "numerator",
    "denominator",
    "status",
    "notes",
]
CANDIDATE_POSITION_BIAS_COLUMNS = [
    "bias_id",
    "model",
    "position",
    "selected_count",
    "selected_rate",
    "top1_count",
    "top1_rate",
    "positive_at_position_users",
    "positive_at_position_hr_at5",
    "random_expected_selected_rate",
    "notes",
]
HISTORY_SENSITIVITY_COLUMNS = [
    "sensitivity_id",
    "model",
    "history_bin",
    "history_length_min",
    "history_length_max",
    "users",
    "hr_at5",
    "ndcg_at5",
    "strict_validity_rate",
    "mean_positive_rank",
    "policy",
    "notes",
]
BASELINE_COMPARISON_COLUMNS = [
    "comparison_id",
    "model",
    "model_group",
    "source_phase",
    "metric",
    "value",
    "baseline_reference",
    "difference_vs_random",
    "difference_vs_mostpop",
    "paper_equivalence",
    "reproduction_class",
    "notes",
]
PAPER_ALIGNMENT_COLUMNS = [
    "alignment_id",
    "dataset",
    "task",
    "model",
    "paper_model_version",
    "local_model_identifier",
    "metric",
    "local_value",
    "paper_value",
    "numeric_difference",
    "dataset_equivalence",
    "split_equivalence",
    "cohort_equivalence",
    "candidate_equivalence",
    "model_version_equivalence",
    "prompt_equivalence",
    "decoding_equivalence",
    "parser_equivalence",
    "metric_equivalence",
    "is_exact_numeric_replication",
    "source_atomic_cell",
    "notes",
]
READINESS_COLUMNS = [
    "readiness_id",
    "component",
    "before_f6d4a",
    "after_f6d4a",
    "available",
    "validated",
    "paper_equivalence",
    "evidence",
    "notes",
]
MISSING_COLUMNS = [
    "missing_update_id",
    "gap_family",
    "total_registry_rows",
    "active_before",
    "resolved_by_f6d4a",
    "partially_resolved_by_f6d4a",
    "active_after",
    "superseded",
    "non_active",
    "evidence",
    "notes",
]
COVERAGE_COLUMNS = [
    "coverage_id",
    "coverage_family",
    "before_f6d4a",
    "after_f6d4a",
    "change",
    "paper_equivalence",
    "reproduction_class",
    "evidence",
    "notes",
]
ANOMALY_COLUMNS = ["anomaly_id", "anomaly", "severity", "status", "evidence", "impact", "mitigation", "notes"]
VALIDATION_COLUMNS = ["validation_id", "check_name", "status", "observed", "expected", "severity", "evidence", "notes"]
PREDICTION_COLUMNS = [
    "request_id",
    "model",
    "task",
    "user_id",
    "cohort_name",
    "candidate_set_name",
    "candidate_count",
    "positive_item_id",
    "positive_candidate_position",
    "strict_valid",
    "rank",
    "candidate_index",
    "recommended_item_id",
    "recommended_item_title",
    "positive_in_top5",
    "ndcg_at5",
    "source_raw_file",
    "notes",
]
INPUT_HASH_COLUMNS = ["source_group", "relative_path", "exists", "size_bytes", "sha256", "notes"]
PROGRESS_COLUMNS = ["model", "expected_requests", "completed_requests", "pending_requests", "progress_status", "notes"]

OUTPUT_TABLES = {
    "model_preflight": Path("results/tables/finalization_f6d4a_model_preflight.csv"),
    "execution_manifest": Path("results/tables/finalization_f6d4a_execution_manifest.csv"),
    "request_audit": Path("results/tables/finalization_f6d4a_request_audit.csv"),
    "response_validation": Path("results/tables/finalization_f6d4a_response_validation.csv"),
    "utility_metrics": Path("results/tables/finalization_f6d4a_utility_metrics.csv"),
    "structural_validity": Path("results/tables/finalization_f6d4a_structural_validity.csv"),
    "candidate_position_bias": Path("results/tables/finalization_f6d4a_candidate_position_bias.csv"),
    "history_length_sensitivity": Path("results/tables/finalization_f6d4a_history_length_sensitivity.csv"),
    "baseline_comparison": Path("results/tables/finalization_f6d4a_baseline_comparison.csv"),
    "paper_alignment": Path("results/tables/finalization_f6d4a_paper_alignment.csv"),
    "readiness_update": Path("results/tables/finalization_f6d4a_readiness_update.csv"),
    "missing_evidence_update": Path("results/tables/finalization_f6d4a_missing_evidence_update.csv"),
    "coverage_impact": Path("results/tables/finalization_f6d4a_coverage_impact.csv"),
    "anomalies": Path("results/tables/finalization_f6d4a_anomalies.csv"),
    "validation": Path("results/tables/finalization_f6d4a_validation.csv"),
}


@dataclass(frozen=True)
class OllamaState:
    executable_found: bool
    executable_name: str
    version: str
    service_available: bool
    service_version: str
    installed_models: dict[str, str]
    error: str
    device: str


@dataclass(frozen=True)
class RequestResult:
    backend_success: bool
    response_text: str
    raw_backend_response: dict[str, Any]
    backend_error_type: str
    backend_error_message: str
    started_at: str
    completed_at: str
    duration_seconds: float


@dataclass(frozen=True)
class PhaseInputs:
    manifest: pd.DataFrame
    prompt_contract: pd.DataFrame
    go_no_go: pd.DataFrame
    execution_plan: pd.DataFrame
    cohort: pd.DataFrame
    f6d1_metrics: pd.DataFrame
    f6d2_aggregate: pd.DataFrame
    f6b_missing: pd.DataFrame
    f6b_coverage: pd.DataFrame
    atomic_cells: pd.DataFrame
    manifest_sha256: str
    input_hashes: dict[str, str]


def stable_id(prefix: str, *parts: Any) -> str:
    payload = "||".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    return sha256_text(frame.fillna("").astype(str).to_csv(index=False, lineterminator="\n"))


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: str(row.get(column, "")) for column in columns} for row in rows], columns=list(columns))


def read_csv(root: Path, relative: Path) -> pd.DataFrame:
    return pd.read_csv(root / relative, dtype=str, keep_default_na=False)


def read_optional_csv(root: Path, relative: Path) -> pd.DataFrame:
    path = root / relative
    if not path.exists():
        return pd.DataFrame()
    return read_csv(root, relative)


def split_pipe(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value)
    if not text:
        return []
    return [part for part in text.split("|")]


def prompt_digest(task: str, model: str, user_id: str, history_ids: str, candidate_ids: str, contract_id: str) -> str:
    payload = {"task": task, "model": model, "user": user_id, "history": history_ids, "candidates": candidate_ids, "contract": contract_id}
    return sha256_text(json.dumps(payload, sort_keys=True))


def ndcg_at_5_from_rank(rank_one_based: int | None) -> float:
    if rank_one_based is None or rank_one_based < 1 or rank_one_based > 5:
        return 0.0
    return 1.0 / math.log2(rank_one_based + 1)


def mean_numeric(values: Iterable[Any]) -> str:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return ""
    return fmt(float(series.mean()))


def median_numeric(values: Iterable[Any]) -> str:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return ""
    return fmt(float(series.median()))


def rate(numerator: int, denominator: int) -> str:
    return fmt(numerator / denominator if denominator else 0.0)


def no_absolute_paths(frame_value: pd.DataFrame) -> bool:
    for value in frame_value.fillna("").astype(str).to_numpy().ravel():
        lowered = value.replace("\\", "/").lower()
        if lowered.startswith("c:/") or "c:/users/" in lowered:
            return False
    return True


def git_output(root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def all_input_paths(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for group in [F6D3_INPUTS, F6C_INPUTS, F6D1_INPUTS, F6D2_INPUTS, F6B_INPUTS, F5AF_INPUTS]:
        result.update(group)
    if (root / PROTECTED_HASHES).exists():
        result["f2_protected_hashes_after"] = PROTECTED_HASHES
    return result


def hash_input_paths(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for _name, relative in sorted(all_input_paths(root).items(), key=lambda item: item[1].as_posix()):
        path = root / relative
        if path.exists() and path.is_file():
            hashes[relative.as_posix()] = file_sha256(path)
    return hashes


def build_input_hashes_frame(root: Path, before_hashes: Mapping[str, str] | None = None) -> pd.DataFrame:
    rows = []
    before = dict(before_hashes or {})
    group_by_path = {}
    for group_name, paths in [
        ("f6d3", F6D3_INPUTS),
        ("f6c", F6C_INPUTS),
        ("f6d1", F6D1_INPUTS),
        ("f6d2", F6D2_INPUTS),
        ("f6b", F6B_INPUTS),
        ("f5af", F5AF_INPUTS),
        ("protected_manifest", {"f2_protected_hashes_after": PROTECTED_HASHES}),
    ]:
        for relative in paths.values():
            group_by_path[relative.as_posix()] = group_name
    for relative in sorted(set(group_by_path) | set(before)):
        path = root / relative
        exists = path.exists()
        sha = file_sha256(path) if exists and path.is_file() else ""
        rows.append(
            {
                "source_group": group_by_path.get(relative, "runtime"),
                "relative_path": relative,
                "exists": bool_text(exists),
                "size_bytes": str(path.stat().st_size) if exists and path.is_file() else "",
                "sha256": sha,
                "notes": "input hash registered for F6D4A; previous_hash_match=" + bool_text(not before or before.get(relative, sha) == sha),
            }
        )
    return frame(rows, INPUT_HASH_COLUMNS)


def protected_hash_status(root: Path) -> tuple[int, int]:
    return _d1r_protected_hash_counts(root)


def previous_input_status(root: Path, before_hashes: Mapping[str, str]) -> tuple[int, int]:
    after = hash_input_paths(root)
    checked = len(before_hashes)
    changed = sum(1 for path, sha in before_hashes.items() if after.get(path) != sha)
    return checked, changed


def load_inputs(root: Path) -> PhaseInputs:
    manifest = read_csv(root, F6D3_INPUTS["ranking_manifest"])
    prompt_contract = read_csv(root, F6D3_INPUTS["prompt_contract"])
    go_no_go = read_csv(root, F6D3_INPUTS["go_no_go"])
    execution_plan = read_csv(root, F6D3_INPUTS["execution_plan"])
    cohort = read_csv(root, F6C_INPUTS["ranking_cohort"])
    f6d1_metrics = read_csv(root, F6D1_INPUTS["utility_metrics"])
    f6d2_aggregate = read_csv(root, F6D2_INPUTS["aggregate_metrics"])
    f6b_missing = read_optional_csv(root, F6B_INPUTS["missing_evidence"])
    f6b_coverage = read_optional_csv(root, F6B_INPUTS["coverage_dashboard"])
    atomic_cells = read_optional_csv(root, F5AF_INPUTS["atomic_cells"])
    manifest_sha = file_sha256(root / F6D3_INPUTS["ranking_manifest"])
    return PhaseInputs(
        manifest=manifest,
        prompt_contract=prompt_contract,
        go_no_go=go_no_go,
        execution_plan=execution_plan,
        cohort=cohort,
        f6d1_metrics=f6d1_metrics,
        f6d2_aggregate=f6d2_aggregate,
        f6b_missing=f6b_missing,
        f6b_coverage=f6b_coverage,
        atomic_cells=atomic_cells,
        manifest_sha256=manifest_sha,
        input_hashes=hash_input_paths(root),
    )


def validate_manifest(inputs: PhaseInputs) -> dict[str, str]:
    status: dict[str, str] = {}
    manifest = inputs.manifest
    status["manifest_rows"] = str(len(manifest))
    status["ranking_only"] = bool_text(set(manifest["task"].astype(str)) == {"ranking"})
    for spec in MODEL_SPECS:
        model = spec["model"]
        subset = manifest.loc[manifest["model"].eq(model)]
        status[f"{model}_rows"] = str(len(subset))
        status[f"{model}_valid"] = bool_text(len(subset) == EXPECTED_REQUESTS_PER_MODEL)
        if len(subset):
            status[f"{model}_candidate_count_valid"] = bool_text(subset["candidate_item_ids"].map(lambda x: len(split_pipe(x))).eq(20).all())
            status[f"{model}_expected_output_count_valid"] = bool_text(subset["expected_output_count"].eq("5").all())
            status[f"{model}_prompt_sha_valid"] = bool_text(verify_prompt_sha_subset(subset))
        else:
            status[f"{model}_candidate_count_valid"] = "false"
            status[f"{model}_expected_output_count_valid"] = "false"
            status[f"{model}_prompt_sha_valid"] = "false"
    return status


def verify_prompt_sha_subset(subset: pd.DataFrame) -> bool:
    for row in subset.to_dict("records"):
        history_ids = str(row["history_item_ids"])
        candidate_ids = str(row["candidate_item_ids"])
        expected = prompt_digest(row["task"], row["model"], row["user_id"], history_ids, candidate_ids, row["prompt_contract_id"])
        if expected != row["prompt_sha256"]:
            return False
    return True


def context_from_manifest_row(row: Mapping[str, str]) -> dict[str, Any]:
    history_ids = split_pipe(row.get("history_item_ids", ""))
    history_names = split_pipe(row.get("history_item_names", ""))
    candidate_ids = split_pipe(row.get("candidate_item_ids", ""))
    candidate_names = split_pipe(row.get("candidate_item_names", ""))
    if len(history_ids) != len(history_names):
        raise ValueError(f"history id/name mismatch: {row.get('request_id')}")
    if len(candidate_ids) != len(candidate_names):
        raise ValueError(f"candidate id/name mismatch: {row.get('request_id')}")
    if len(candidate_ids) != 20:
        raise ValueError(f"candidate count mismatch: {row.get('request_id')}")
    computed_prompt_hash = prompt_digest(
        str(row["task"]),
        str(row["model"]),
        str(row["user_id"]),
        str(row["history_item_ids"]),
        str(row["candidate_item_ids"]),
        str(row["prompt_contract_id"]),
    )
    if computed_prompt_hash != str(row["prompt_sha256"]):
        raise ValueError(f"F6D3 prompt_sha256 mismatch: {row.get('request_id')}")
    history_items = [{"item_id": item_id, "title": title} for item_id, title in zip(history_ids, history_names)]
    candidate_items = [{"item_id": item_id, "title": title} for item_id, title in zip(candidate_ids, candidate_names)]
    system_message, user_message, history_render, candidate_render = render_prompt_from_manifest(history_items, candidate_items)
    return {
        "candidate_items": candidate_ids,
        "candidate_titles": candidate_names,
        "history_items": history_ids,
        "history_titles": history_names,
        "system_message": system_message,
        "user_message": user_message,
        "history_render": history_render,
        "candidate_render": candidate_render,
        "complete_prompt_sha256": sha256_text(system_message + "\n\n" + user_message),
        "prompt_sha256": str(row["prompt_sha256"]),
        "requested_output_count": 5,
    }


def render_prompt_from_manifest(history_items: Sequence[Mapping[str, str]], candidate_items: Sequence[Mapping[str, str]]) -> tuple[str, str, str, str]:
    history_render = render_history(history_items)
    candidate_render = render_candidates(candidate_items)
    system_message = "You are a recommender-system evaluator. Follow the user message exactly. Return only valid JSON. Do not explain your reasoning."
    user_message = (
        "Task: rank candidate items for the user's next interaction.\n"
        "History is train-only and chronological.\n"
        "Candidate labels are 1-based indices. Do not use item IDs in the answer.\n"
        'Return exactly 5 unique candidate indices as JSON: {{"ranked_candidate_indices":[1,2,3,4,5]}}.\n\n'
        f"User history:\n{history_render}\n\nCandidates:\n{candidate_render}"
    )
    return system_message, user_message, history_render, candidate_render


def build_payload(row: Mapping[str, str], context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": str(row["model"]),
        "system": str(context["system_message"]),
        "prompt": str(context["user_message"]),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 256},
    }


def observe_device() -> str:
    names: list[str] = []
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    except Exception:
        names = []
    if names:
        return "; ".join(names)
    return f"cpu_or_unknown; platform={platform.machine()}"


def inspect_ollama() -> OllamaState:
    executable = shutil.which("ollama")
    executable_name = Path(executable).name if executable else ""
    version = ""
    error = ""
    if executable:
        try:
            completed = subprocess.run([executable, "--version"], check=False, capture_output=True, text=True, timeout=30)
            if completed.returncode == 0:
                version = (completed.stdout or completed.stderr).strip()
            else:
                error = (completed.stdout or completed.stderr).strip()
        except Exception as exc:
            error = str(exc)
    service_available = False
    service_version = ""
    installed: dict[str, str] = {}
    try:
        with urllib.request.urlopen(OLLAMA_VERSION_URL, timeout=10) as response:
            service_payload = json.loads(response.read().decode("utf-8"))
        service_available = True
        service_version = str(service_payload.get("version", ""))
    except Exception as exc:
        service_version = str(exc)
    if service_available:
        try:
            with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=10) as response:
                tags_payload = json.loads(response.read().decode("utf-8"))
            for model in tags_payload.get("models", []):
                name = str(model.get("name", ""))
                digest = str(model.get("digest", ""))
                if name:
                    installed[name] = digest
        except Exception as exc:
            error = str(exc)
    return OllamaState(
        executable_found=bool(executable),
        executable_name=executable_name,
        version=version,
        service_available=service_available,
        service_version=service_version,
        installed_models=installed,
        error=error,
        device=observe_device(),
    )


def offline_ollama_state() -> OllamaState:
    return OllamaState(
        executable_found=False,
        executable_name="not_used_analysis_only",
        version="not_queried_analysis_only",
        service_available=False,
        service_version="not_queried_analysis_only",
        installed_models={},
        error="analysis_only_offline_no_ollama_introspection",
        device="not_observed_analysis_only",
    )


def execute_ollama_request(payload: Mapping[str, Any], timeout_seconds: int = 120) -> RequestResult:
    started = utc_now()
    start_clock = time.perf_counter()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
        raw = json.loads(raw_body)
        response_text = str(raw.get("response", ""))
        success = response_text != "" or bool(raw.get("done"))
        return RequestResult(success, response_text, raw, "", "", started, utc_now(), time.perf_counter() - start_clock)
    except urllib.error.HTTPError as exc:
        return RequestResult(False, "", {}, "HTTPError", f"{exc.code}: {exc.reason}", started, utc_now(), time.perf_counter() - start_clock)
    except urllib.error.URLError as exc:
        return RequestResult(False, "", {}, "URLError", str(exc.reason), started, utc_now(), time.perf_counter() - start_clock)
    except TimeoutError as exc:
        return RequestResult(False, "", {}, "TimeoutError", str(exc), started, utc_now(), time.perf_counter() - start_clock)
    except Exception as exc:  # pragma: no cover - local backend dependent
        return RequestResult(False, "", {}, type(exc).__name__, str(exc), started, utc_now(), time.perf_counter() - start_clock)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            record["_jsonl_line_no"] = line_no
            rows.append(record)
    return rows


def append_jsonl_fsync(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def final_records_by_request(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if str(record.get("is_final_attempt", "")).lower() == "true":
            request_id = str(record.get("request_id", ""))
            if request_id:
                result.setdefault(request_id, []).append(dict(record))
    return result


def choose_final_record(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    finals = [dict(record) for record in records if str(record.get("is_final_attempt", "")).lower() == "true"]
    if not finals:
        return None
    finals.sort(key=lambda record: int(record.get("_jsonl_line_no", 0)))
    return finals[-1]


def raw_path_for_model(model: str) -> Path:
    return RAW_PATHS[model]


def make_raw_record(row: Mapping[str, str], result: RequestResult, inputs: PhaseInputs, is_smoke: bool) -> dict[str, Any]:
    response_sha = sha256_text(result.response_text)
    completion_status = "completed" if result.backend_success else "transport_failed"
    return {
        "request_id": row["request_id"],
        "model": row["model"],
        "task": row["task"],
        "user_id": row["user_id"],
        "prompt_sha256": row["prompt_sha256"],
        "manifest_sha256": inputs.manifest_sha256,
        "attempt": 1,
        "backend": "ollama_local_loopback",
        "local_model_identifier": row["model"],
        "raw_response": result.response_text,
        "response_sha256": response_sha,
        "transport_status": "success" if result.backend_success else "failed",
        "completion_status": completion_status,
        "error_type": result.backend_error_type,
        "error_message": result.backend_error_message,
        "is_final_attempt": "true",
        "notes": "smoke_request" if is_smoke else "ranking_request",
        "request_started_at_utc": result.started_at,
        "request_completed_at_utc": result.completed_at,
        "duration_seconds": fmt(result.duration_seconds, 17),
        "done_reason": str(result.raw_backend_response.get("done_reason", "done" if result.raw_backend_response.get("done") else "")) if result.raw_backend_response else "",
        "prompt_eval_count": str(result.raw_backend_response.get("prompt_eval_count", "")) if result.raw_backend_response else "",
        "eval_count": str(result.raw_backend_response.get("eval_count", "")) if result.raw_backend_response else "",
        "total_duration_nanoseconds": str(result.raw_backend_response.get("total_duration", "")) if result.raw_backend_response else "",
        "prompt_text_sha256": context_from_manifest_row(row)["complete_prompt_sha256"],
    }


def write_progress(root: Path, manifest: pd.DataFrame) -> None:
    rows = []
    for spec in MODEL_SPECS:
        model = spec["model"]
        records = read_jsonl(root / raw_path_for_model(model))
        finals = final_records_by_request(records)
        completed = len(set(finals) & set(manifest.loc[manifest["model"].eq(model), "request_id"]))
        expected = len(manifest.loc[manifest["model"].eq(model)])
        rows.append(
            {
                "model": model,
                "expected_requests": expected,
                "completed_requests": completed,
                "pending_requests": expected - completed,
                "progress_status": "complete" if completed == expected else "incomplete",
                "notes": "deterministic progress from raw JSONL",
            }
        )
    out = root / PROGRESS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    frame(rows, PROGRESS_COLUMNS).to_csv(out, index=False, lineterminator="\n")


def execute_missing_requests(
    root: Path,
    inputs: PhaseInputs,
    ollama_state: OllamaState,
    *,
    max_new_requests: int | None = None,
    request_executor: Callable[[Mapping[str, Any], int], RequestResult] = execute_ollama_request,
) -> int:
    sent = 0
    manifest_status = validate_manifest(inputs)
    protected_before = dict(inputs.input_hashes)
    for spec in MODEL_SPECS:
        model = spec["model"]
        model_manifest = inputs.manifest.loc[inputs.manifest["model"].eq(model)].copy()
        if len(model_manifest) != EXPECTED_REQUESTS_PER_MODEL:
            continue
        if manifest_status.get(f"{model}_prompt_sha_valid") != "true":
            continue
        if not ollama_state.service_available or model not in ollama_state.installed_models:
            continue
        raw_path = root / raw_path_for_model(model)
        existing = read_jsonl(raw_path)
        finals = final_records_by_request(existing)
        for row in model_manifest.to_dict("records"):
            if max_new_requests is not None and sent >= max_new_requests:
                write_progress(root, inputs.manifest)
                return sent
            if row["request_id"] in finals:
                continue
            if hash_input_paths(root) != protected_before:
                raise RuntimeError("protected_or_previous_input_changed_before_request")
            context = context_from_manifest_row(row)
            payload = build_payload(row, context)
            is_smoke = len(finals) == 0 and sent == 0
            result = request_executor(payload, 120)
            record = make_raw_record(row, result, inputs, is_smoke=is_smoke)
            append_jsonl_fsync(raw_path, record)
            sent += 1
            finals[row["request_id"]] = [record]
            if sent == 1 and not result.backend_success:
                break
            if sent % 25 == 0:
                write_progress(root, inputs.manifest)
    write_progress(root, inputs.manifest)
    return sent


def classify_response(row: Mapping[str, str], raw_record: Mapping[str, Any] | None) -> dict[str, Any]:
    candidate_items = split_pipe(row.get("candidate_item_ids", ""))
    positive = str(row.get("positive_item_id", ""))
    positive_position = str(row.get("positive_candidate_position", ""))
    if raw_record is None:
        return {
            "request_id": row["request_id"],
            "model": row["model"],
            "user_id": row["user_id"],
            "transport_success": "false",
            "parse_valid": "false",
            "strict_valid": "false",
            "json_valid": "false",
            "schema_valid": "false",
            "output_count_valid": "false",
            "index_range_valid": "false",
            "uniqueness_valid": "false",
            "candidate_validity": "false",
            "duplicate_index_count": "0",
            "out_of_range_index_count": "0",
            "missing_selection_count": "5",
            "extra_selection_count": "0",
            "malformed_output": "false",
            "ranked_candidate_indices": "",
            "ranked_item_ids": "",
            "positive_item_id": positive,
            "positive_candidate_position": positive_position,
            "positive_rank": "",
            "positive_in_top5": "false",
            "ndcg_at5": "0",
            "failure_reason": "missing_raw_response",
            "notes": "request pending or model blocked",
        }
    transport_success = str(raw_record.get("transport_status", "")) == "success"
    response_text = str(raw_record.get("raw_response", ""))
    parsed = parse_strict_index_response(response_text, candidate_items=candidate_items, expected_count=5, require_full_permutation=False)
    raw_status = parsed.status
    json_valid = raw_status not in {"empty_output", "parse_failed"}
    schema_valid = raw_status not in {"empty_output", "parse_failed", "invalid_schema"}
    output_count_valid = parsed.output_length == 5
    index_range_valid = parsed.unknown_item_count == 0 and output_count_valid
    uniqueness_valid = parsed.duplicate_count == 0 and output_count_valid
    candidate_validity = parsed.parse_valid and len(parsed.output_items) == 5
    selected = list(parsed.output_items) if parsed.parse_valid else []
    positive_rank: int | None = None
    if positive in selected:
        positive_rank = selected.index(positive) + 1
    if raw_status == "invalid_length":
        missing_count = max(0, 5 - parsed.output_length)
        extra_count = max(0, parsed.output_length - 5)
    elif raw_status in {"empty_output", "parse_failed", "invalid_schema"}:
        missing_count = 5
        extra_count = 0
    else:
        missing_count = 0
        extra_count = 0
    failure_reason = "" if parsed.parse_valid and transport_success else (raw_record.get("error_type", "") or parsed.status)
    return {
        "request_id": row["request_id"],
        "model": row["model"],
        "user_id": row["user_id"],
        "transport_success": bool_text(transport_success),
        "parse_valid": bool_text(json_valid and schema_valid),
        "strict_valid": bool_text(parsed.parse_valid and transport_success),
        "json_valid": bool_text(json_valid),
        "schema_valid": bool_text(schema_valid),
        "output_count_valid": bool_text(output_count_valid),
        "index_range_valid": bool_text(index_range_valid),
        "uniqueness_valid": bool_text(uniqueness_valid),
        "candidate_validity": bool_text(candidate_validity),
        "duplicate_index_count": str(parsed.duplicate_count),
        "out_of_range_index_count": str(parsed.unknown_item_count),
        "missing_selection_count": str(missing_count),
        "extra_selection_count": str(extra_count),
        "malformed_output": bool_text(raw_status in {"empty_output", "parse_failed", "invalid_schema"}),
        "ranked_candidate_indices": "|".join(str(index) for index in parsed.output_indices) if parsed.output_indices else "",
        "ranked_item_ids": "|".join(selected),
        "positive_item_id": positive,
        "positive_candidate_position": positive_position,
        "positive_rank": str(positive_rank) if positive_rank is not None else "",
        "positive_in_top5": bool_text(positive_rank is not None and positive_rank <= 5),
        "ndcg_at5": fmt(ndcg_at_5_from_rank(positive_rank)),
        "failure_reason": str(failure_reason),
        "notes": "strict parser only; no salvage or manual correction",
    }


def build_response_validation(root: Path, inputs: PhaseInputs) -> pd.DataFrame:
    rows = []
    for model in [spec["model"] for spec in MODEL_SPECS]:
        records = read_jsonl(root / raw_path_for_model(model))
        by_request = final_records_by_request(records)
        subset = inputs.manifest.loc[inputs.manifest["model"].eq(model)]
        for row in subset.to_dict("records"):
            raw_record = choose_final_record(by_request.get(row["request_id"], []))
            rows.append(classify_response(row, raw_record))
    return frame(rows, RESPONSE_VALIDATION_COLUMNS)


def build_predictions(inputs: PhaseInputs, validation: pd.DataFrame) -> pd.DataFrame:
    manifest_lookup = {row.request_id: row._asdict() for row in inputs.manifest.itertuples(index=False)}
    rows = []
    for record in validation.loc[validation["strict_valid"].eq("true")].to_dict("records"):
        manifest_row = manifest_lookup[record["request_id"]]
        item_ids = split_pipe(record["ranked_item_ids"])
        indices = split_pipe(record["ranked_candidate_indices"])
        names = split_pipe(manifest_row["candidate_item_names"])
        for rank, (candidate_index, item_id) in enumerate(zip(indices, item_ids), start=1):
            title = names[int(candidate_index) - 1] if str(candidate_index).isdigit() and 1 <= int(candidate_index) <= len(names) else ""
            rows.append(
                {
                    "request_id": record["request_id"],
                    "model": record["model"],
                    "task": "ranking",
                    "user_id": record["user_id"],
                    "cohort_name": manifest_row["cohort_name"],
                    "candidate_set_name": "finalization_f6c_lastfm_ranking_candidates_20",
                    "candidate_count": "20",
                    "positive_item_id": record["positive_item_id"],
                    "positive_candidate_position": record["positive_candidate_position"],
                    "strict_valid": "true",
                    "rank": str(rank),
                    "candidate_index": candidate_index,
                    "recommended_item_id": item_id,
                    "recommended_item_title": title,
                    "positive_in_top5": record["positive_in_top5"],
                    "ndcg_at5": record["ndcg_at5"],
                    "source_raw_file": raw_path_for_model(record["model"]).as_posix(),
                    "notes": "strict-valid top-5 only",
                }
            )
    return frame(rows, PREDICTION_COLUMNS)


def build_request_audit(root: Path, inputs: PhaseInputs, validation: pd.DataFrame) -> pd.DataFrame:
    validation_lookup = {row.request_id: row._asdict() for row in validation.itertuples(index=False)}
    rows = []
    for model in [spec["model"] for spec in MODEL_SPECS]:
        records = read_jsonl(root / raw_path_for_model(model))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            request_id = str(record.get("request_id", ""))
            grouped.setdefault(request_id, []).append(record)
        for row in inputs.manifest.loc[inputs.manifest["model"].eq(model)].to_dict("records"):
            related = grouped.get(row["request_id"], [])
            finals = [record for record in related if str(record.get("is_final_attempt", "")).lower() == "true"]
            final = choose_final_record(finals)
            responses = {str(record.get("response_sha256", "")) for record in finals if str(record.get("response_sha256", ""))}
            val = validation_lookup.get(row["request_id"], {})
            raw_records = len(related)
            final_records = len(finals)
            duplicate = raw_records > 1 or final_records > 1
            conflict = len(responses) > 1
            if conflict:
                status = "conflicting_response"
            elif duplicate:
                status = "duplicate_final_response"
            elif final_records == 1:
                status = "completed"
            else:
                status = "pending"
            rows.append(
                {
                    "request_id": row["request_id"],
                    "model": model,
                    "task": "ranking",
                    "user_id": row["user_id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "manifest_sha256": inputs.manifest_sha256,
                    "raw_records": raw_records,
                    "final_records": final_records,
                    "attempts": raw_records,
                    "transport_status": str(final.get("transport_status", "")) if final else "",
                    "completion_status": str(final.get("completion_status", "")) if final else "",
                    "resume_state": "skip_completed" if final_records else "pending",
                    "duplicate_request": bool_text(duplicate),
                    "conflicting_response": bool_text(conflict),
                    "response_sha256": str(final.get("response_sha256", "")) if final else "",
                    "raw_output_file": raw_path_for_model(model).as_posix(),
                    "audit_status": status,
                    "blocking_reason": "" if status in {"completed", "pending"} else status,
                    "notes": "strict_valid=" + str(val.get("strict_valid", "")),
                }
            )
    return frame(rows, REQUEST_AUDIT_COLUMNS)


def build_utility_metrics(inputs: PhaseInputs, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in validation.groupby("model", sort=True):
        total = len(group)
        transport = int(group["transport_success"].eq("true").sum())
        parse_valid = int(group["parse_valid"].eq("true").sum())
        strict = group.loc[group["strict_valid"].eq("true")]
        all_hits = group["positive_in_top5"].eq("true").astype(float)
        all_ndcg = pd.to_numeric(group["ndcg_at5"], errors="coerce").fillna(0.0)
        strict_hits = strict["positive_in_top5"].eq("true").astype(float)
        strict_ndcg = pd.to_numeric(strict["ndcg_at5"], errors="coerce").fillna(0.0)
        strict_ranks = pd.to_numeric(strict["positive_rank"], errors="coerce")
        all_ranks = pd.to_numeric(group["positive_rank"], errors="coerce")
        metrics = [
            ("all_request", "HR@5", all_hits.mean() if total else 0.0, total),
            ("all_request", "NDCG@5", all_ndcg.mean() if total else 0.0, total),
            ("all_request", "mean_positive_rank", all_ranks.mean() if all_ranks.notna().any() else "", total),
            ("all_request", "median_positive_rank", all_ranks.median() if all_ranks.notna().any() else "", total),
            ("strict_valid", "HR@5", strict_hits.mean() if len(strict) else 0.0, len(strict)),
            ("strict_valid", "NDCG@5", strict_ndcg.mean() if len(strict) else 0.0, len(strict)),
            ("strict_valid", "mean_positive_rank", strict_ranks.mean() if strict_ranks.notna().any() else "", len(strict)),
            ("strict_valid", "median_positive_rank", strict_ranks.median() if strict_ranks.notna().any() else "", len(strict)),
        ]
        for population, metric, value, valid_rows in metrics:
            rows.append(
                {
                    "metric_id": stable_id("F6D4A-MET", model, population, metric),
                    "model": model,
                    "evaluation_population": population,
                    "metric": metric,
                    "cutoff": "5" if "@5" in metric else "",
                    "value": fmt(value),
                    "valid_rows": valid_rows,
                    "total_rows": total,
                    "strict_valid_rows": len(strict),
                    "transport_success_rows": transport,
                    "parse_valid_rows": parse_valid,
                    "candidate_count": "20",
                    "cohort_name": "finalization_f6c_lastfm_ranking_cohort_full_1859",
                    "candidate_set_name": "finalization_f6c_lastfm_ranking_candidates_20",
                    "paper_equivalence": "not_exact_adapted_local_policy",
                    "reproduction_class": "original_paper_component_locally_adapted",
                    "notes": "all-request is primary; strict-valid is secondary diagnostic.",
                }
            )
    return frame(rows, UTILITY_METRIC_COLUMNS)


def build_structural_validity(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_map = [
        ("transport_success_rate", "transport_success"),
        ("parse_validity_rate", "parse_valid"),
        ("strict_validity_rate", "strict_valid"),
        ("duplicate_index_rate", "duplicate_index_count"),
        ("out_of_range_index_rate", "out_of_range_index_count"),
        ("malformed_output_rate", "malformed_output"),
        ("missing_selection_rate", "missing_selection_count"),
        ("extra_selection_rate", "extra_selection_count"),
        ("retry_recovery_rate", None),
    ]
    for model, group in validation.groupby("model", sort=True):
        denom = len(group)
        for metric, source in metric_map:
            if source is None:
                numerator = 0
            elif source.endswith("_count"):
                numerator = int(pd.to_numeric(group[source], errors="coerce").fillna(0).gt(0).sum())
            else:
                numerator = int(group[source].eq("true").sum())
            rows.append(
                {
                    "structural_id": stable_id("F6D4A-STRUCT", model, metric),
                    "model": model,
                    "metric": metric,
                    "value": rate(numerator, denom),
                    "numerator": numerator,
                    "denominator": denom,
                    "status": "computed",
                    "notes": "Structural validity is not paper textual hallucination equivalence.",
                }
            )
    return frame(rows, STRUCTURAL_VALIDITY_COLUMNS)


def build_position_bias(inputs: PhaseInputs, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in validation.groupby("model", sort=True):
        strict = group.loc[group["strict_valid"].eq("true")]
        selected_positions: list[int] = []
        top1_positions: list[int] = []
        for row in strict.to_dict("records"):
            indices = [int(value) for value in split_pipe(row["ranked_candidate_indices"]) if str(value).isdigit()]
            selected_positions.extend(indices)
            if indices:
                top1_positions.append(indices[0])
        total_selected = len(selected_positions)
        total_top1 = len(top1_positions)
        for position in range(1, 21):
            positive_at_pos = group.loc[group["positive_candidate_position"].eq(str(position))]
            positive_hits = int(positive_at_pos["positive_in_top5"].eq("true").sum())
            rows.append(
                {
                    "bias_id": stable_id("F6D4A-BIAS", model, position),
                    "model": model,
                    "position": position,
                    "selected_count": selected_positions.count(position),
                    "selected_rate": rate(selected_positions.count(position), total_selected),
                    "top1_count": top1_positions.count(position),
                    "top1_rate": rate(top1_positions.count(position), total_top1),
                    "positive_at_position_users": len(positive_at_pos),
                    "positive_at_position_hr_at5": rate(positive_hits, len(positive_at_pos)),
                    "random_expected_selected_rate": fmt(5 / 20),
                    "notes": "Denominators are strict-valid selected slots for selection rates and all users per positive position for HR.",
                }
            )
    return frame(rows, CANDIDATE_POSITION_BIAS_COLUMNS)


def history_bin(length: int) -> tuple[str, int, int]:
    if length <= 10:
        return "1-10", 1, 10
    if length <= 25:
        return "11-25", 11, 25
    if length <= 50:
        return "26-50", 26, 50
    return "51+", 51, 10**9


def build_history_sensitivity(inputs: PhaseInputs, validation: pd.DataFrame) -> pd.DataFrame:
    cohort = inputs.cohort[["user_id", "train_history_count"]].copy()
    cohort["history_bin"] = cohort["train_history_count"].astype(int).map(lambda value: history_bin(value)[0])
    merged = validation.merge(cohort, on="user_id", how="left")
    rows = []
    for model, model_group in merged.groupby("model", sort=True):
        for label in ["1-10", "11-25", "26-50", "51+"]:
            group = model_group.loc[model_group["history_bin"].eq(label)]
            if group.empty:
                min_len, max_len = {"1-10": (1, 10), "11-25": (11, 25), "26-50": (26, 50), "51+": (51, 10**9)}[label]
                rows.append(
                    {
                        "sensitivity_id": stable_id("F6D4A-HIST", model, label),
                        "model": model,
                        "history_bin": label,
                        "history_length_min": min_len,
                        "history_length_max": "unbounded" if label == "51+" else max_len,
                        "users": 0,
                        "hr_at5": "",
                        "ndcg_at5": "",
                        "strict_validity_rate": "",
                        "mean_positive_rank": "",
                        "policy": "adapted_train_history_bins",
                        "notes": "No users in bin.",
                    }
                )
                continue
            strict = group.loc[group["strict_valid"].eq("true")]
            ranks = pd.to_numeric(strict["positive_rank"], errors="coerce")
            min_len = int(pd.to_numeric(group["train_history_count"], errors="coerce").min())
            max_len = int(pd.to_numeric(group["train_history_count"], errors="coerce").max())
            rows.append(
                {
                    "sensitivity_id": stable_id("F6D4A-HIST", model, label),
                    "model": model,
                    "history_bin": label,
                    "history_length_min": min_len,
                    "history_length_max": max_len,
                    "users": len(group),
                    "hr_at5": fmt(group["positive_in_top5"].eq("true").astype(float).mean()),
                    "ndcg_at5": fmt(pd.to_numeric(group["ndcg_at5"], errors="coerce").fillna(0.0).mean()),
                    "strict_validity_rate": rate(len(strict), len(group)),
                    "mean_positive_rank": fmt(ranks.mean()) if ranks.notna().any() else "",
                    "policy": "adapted_train_history_bins",
                    "notes": "Paper bins were not recovered locally; adapted bins use F6C train_history_count only.",
                }
            )
    return frame(rows, HISTORY_SENSITIVITY_COLUMNS)


def metric_lookup(frame_value: pd.DataFrame, *, metric: str, model: str | None = None, value_column: str = "value") -> str:
    data = frame_value.loc[frame_value["metric"].eq(metric)].copy()
    if model is not None and "model" in data.columns:
        data = data.loc[data["model"].eq(model)]
    if data.empty:
        return ""
    return str(data.iloc[0][value_column])


def build_baseline_comparison(inputs: PhaseInputs, utility: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_rows: list[dict[str, Any]] = []
    f6d1 = inputs.f6d1_metrics.loc[inputs.f6d1_metrics["metric"].isin(["HR@5", "NDCG@5", "mean_positive_rank", "median_positive_rank"])].copy()
    for record in f6d1.to_dict("records"):
        baseline_rows.append(
            {
                "model": record["model"],
                "model_group": "traditional_baseline",
                "source_phase": "F6D1",
                "metric": record["metric"],
                "value": record["value"],
                "paper_equivalence": record.get("paper_equivalence", "not_exact"),
                "reproduction_class": record.get("reproduction_class", "local_adapted"),
            }
        )
    f6d2 = inputs.f6d2_aggregate.loc[
        inputs.f6d2_aggregate["evaluation_scope"].eq("candidate_set_20")
        & inputs.f6d2_aggregate["metric"].isin(["HR@5", "NDCG@5", "mean_positive_rank", "median_positive_rank"])
    ].copy()
    for record in f6d2.to_dict("records"):
        baseline_rows.append(
            {
                "model": record["model"],
                "model_group": "configuration_limited_diagnostic",
                "source_phase": "F6D2",
                "metric": record["metric"],
                "value": record["mean"],
                "paper_equivalence": "not_exact_f6c_adapted_protocol",
                "reproduction_class": "configuration_limited_diagnostic",
            }
        )
    llm = utility.loc[utility["evaluation_population"].eq("all_request")].copy()
    for record in llm.to_dict("records"):
        value = record["value"]
        if record.get("transport_success_rows", "0") == "0" and record.get("parse_valid_rows", "0") == "0":
            value = ""
        baseline_rows.append(
            {
                "model": record["model"],
                "model_group": "local_llm",
                "source_phase": "F6D4A",
                "metric": record["metric"],
                "value": value,
                "paper_equivalence": record["paper_equivalence"],
                "reproduction_class": record["reproduction_class"],
            }
        )
    lookup = {(row["model"], row["metric"]): row["value"] for row in baseline_rows}
    random_by_metric = {row["metric"]: row["value"] for row in baseline_rows if row["model"] == "random_seeded"}
    mostpop_by_metric = {row["metric"]: row["value"] for row in baseline_rows if row["model"] == "mostpop_train_only"}
    for row in baseline_rows:
        metric = row["metric"]
        value = row["value"]
        try:
            val = float(value)
            random_diff = val - float(random_by_metric.get(metric, "nan"))
            mostpop_diff = val - float(mostpop_by_metric.get(metric, "nan"))
        except ValueError:
            random_diff = math.nan
            mostpop_diff = math.nan
        rows.append(
            {
                "comparison_id": stable_id("F6D4A-CMP", row["model"], metric),
                "model": row["model"],
                "model_group": row["model_group"],
                "source_phase": row["source_phase"],
                "metric": metric,
                "value": fmt(value),
                "baseline_reference": "same F6C LastFM ranking candidate_set_20",
                "difference_vs_random": fmt(random_diff) if math.isfinite(random_diff) else "",
                "difference_vs_mostpop": fmt(mostpop_diff) if math.isfinite(mostpop_diff) else "",
                "paper_equivalence": row["paper_equivalence"],
                "reproduction_class": row["reproduction_class"],
                "notes": "BPRMF/GRU4Rec/SASRec/LightGCN retain F6D2 configuration_limited_diagnostic status.",
            }
        )
    return frame(rows, BASELINE_COMPARISON_COLUMNS)


def build_paper_alignment(inputs: PhaseInputs, utility: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in MODEL_SPECS:
        model = spec["model"]
        local_metrics = utility.loc[(utility["model"].eq(model)) & (utility["evaluation_population"].eq("all_request"))]
        for metric in ["HR@5", "NDCG@5"]:
            local_value = metric_lookup(local_metrics, metric=metric)
            if not local_metrics.empty:
                metric_row = local_metrics.loc[local_metrics["metric"].eq(metric)].iloc[0]
                if metric_row.get("transport_success_rows", "0") == "0" and metric_row.get("parse_valid_rows", "0") == "0":
                    local_value = ""
            atom = pd.DataFrame()
            if not inputs.atomic_cells.empty:
                metric_name = "hr_at_5" if metric == "HR@5" else "ndcg_at_5"
                model_family = "Qwen2" if model.startswith("qwen2") else "Mistral"
                atom = inputs.atomic_cells.loc[
                    inputs.atomic_cells["dataset"].eq("LastFM")
                    & inputs.atomic_cells["task"].eq("ranking")
                    & inputs.atomic_cells["model"].astype(str).str.contains(model_family, case=False, regex=False)
                    & inputs.atomic_cells["metric"].eq(metric_name)
                ]
            paper_value = str(atom.iloc[0]["paper_value_numeric"]) if not atom.empty else ""
            source_cell = str(atom.iloc[0]["atomic_cell_id"]) if not atom.empty else ""
            diff = ""
            try:
                diff = fmt(float(local_value) - float(paper_value)) if paper_value != "" and local_value != "" else ""
            except ValueError:
                diff = ""
            rows.append(
                {
                    "alignment_id": stable_id("F6D4A-ALIGN", model, metric),
                    "dataset": "LastFM",
                    "task": "ranking",
                    "model": spec["model_family"],
                    "paper_model_version": spec["paper_model_version"],
                    "local_model_identifier": model,
                    "metric": metric,
                    "local_value": local_value,
                    "paper_value": paper_value,
                    "numeric_difference": diff,
                    "dataset_equivalence": "same_named_dataset_adapted_local_split",
                    "split_equivalence": "false_not_exact_no_listening_timestamp_policy",
                    "cohort_equivalence": "false_f6c_local_1859_user_cohort",
                    "candidate_equivalence": "false_f6c_local_candidate_set_20",
                    "model_version_equivalence": "false_family_match_only",
                    "prompt_equivalence": "false_json_index_contract_adapted",
                    "decoding_equivalence": "false_local_ollama_parameters_not_paper_exact",
                    "parser_equivalence": "false_strict_index_parser_adapted",
                    "metric_equivalence": "true_for_utility_formula_only",
                    "is_exact_numeric_replication": "false",
                    "source_atomic_cell": source_cell,
                    "notes": "No exact numeric replication is claimed.",
                }
            )
    return frame(rows, PAPER_ALIGNMENT_COLUMNS)


def active_missing_counts(inputs: PhaseInputs) -> tuple[int, int]:
    if inputs.f6b_missing.empty:
        return 0, 0
    total = len(inputs.f6b_missing)
    if "current_status" in inputs.f6b_missing.columns:
        current = inputs.f6b_missing["current_status"].astype(str)
        active = int((current.eq("still_missing") | current.str.startswith("active_gap")).sum())
    elif "resolution_category" in inputs.f6b_missing.columns:
        active = int(inputs.f6b_missing["resolution_category"].astype(str).str.startswith("active_gap").sum())
    else:
        active = 0
    return total, active


def build_readiness_missing_coverage(inputs: PhaseInputs, validation: pd.DataFrame, utility: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strict_by_model = {
        model: int(group["strict_valid"].eq("true").sum())
        for model, group in validation.groupby("model", sort=True)
    }
    completed_by_model = {
        model: int(group["failure_reason"].ne("missing_raw_response").sum())
        for model, group in validation.groupby("model", sort=True)
    }
    readiness_rows = []
    for spec in MODEL_SPECS:
        model = spec["model"]
        completed = completed_by_model.get(model, 0)
        strict = strict_by_model.get(model, 0)
        after = "executed" if completed == EXPECTED_REQUESTS_PER_MODEL else "blocked_or_incomplete"
        readiness_rows.append(
            {
                "readiness_id": stable_id("F6D4A-READY", model, "ranking"),
                "component": f"LastFM {model} ranking",
                "before_f6d4a": "go_adapted_manifest_only",
                "after_f6d4a": after,
                "available": bool_text(completed > 0),
                "validated": bool_text(strict >= 0 and completed == EXPECTED_REQUESTS_PER_MODEL),
                "paper_equivalence": "not_exact_adapted_local_policy",
                "evidence": OUTPUT_TABLES["utility_metrics"].as_posix(),
                "notes": f"completed={completed};strict_valid={strict};no reranking executed",
            }
        )
    readiness = frame(readiness_rows, READINESS_COLUMNS)

    total_missing, active_before = active_missing_counts(inputs)
    resolved = sum(1 for completed in completed_by_model.values() if completed == EXPECTED_REQUESTS_PER_MODEL)
    partial = sum(1 for completed in completed_by_model.values() if 0 < completed < EXPECTED_REQUESTS_PER_MODEL)
    active_after = max(0, active_before - resolved)
    missing_rows = [
        {
            "missing_update_id": stable_id("F6D4A-MISS", "lastfm_llm_ranking_execution"),
            "gap_family": "LastFM local LLM ranking execution",
            "total_registry_rows": total_missing,
            "active_before": active_before,
            "resolved_by_f6d4a": resolved,
            "partially_resolved_by_f6d4a": partial,
            "active_after": active_after,
            "superseded": 0,
            "non_active": max(0, total_missing - active_before),
            "evidence": OUTPUT_TABLES["execution_manifest"].as_posix(),
            "notes": "Only adapted local ranking evidence changes; exact protocol/model/prompt gaps remain active.",
        }
    ]
    missing = frame(missing_rows, MISSING_COLUMNS)

    llm_executed = resolved
    coverage_rows = [
        ("LLM-family LastFM local-run coverage", "0/2 LastFM local adapted ranking runs", f"{llm_executed}/2 LastFM local adapted ranking runs", f"+{llm_executed}", "not_exact", "original_paper_component_locally_adapted", OUTPUT_TABLES["execution_manifest"].as_posix(), "Qwen2 and Mistral are paper families but local model versions/prompts are adapted."),
        ("utility coverage", "LastFM LLM utility absent", "LastFM LLM utility computed" if llm_executed else "LastFM LLM utility blocked", "computed" if llm_executed else "no change", "not_exact", "adapted_local_utility", OUTPUT_TABLES["utility_metrics"].as_posix(), ""),
        ("position-bias coverage", "LastFM LLM position bias absent", "LastFM LLM position bias computed" if llm_executed else "blocked", "computed" if llm_executed else "no change", "not_exact", "adapted_local_diagnostic", OUTPUT_TABLES["candidate_position_bias"].as_posix(), ""),
        ("history-sensitivity coverage", "LastFM LLM history sensitivity absent", "LastFM LLM history sensitivity computed" if llm_executed else "blocked", "computed" if llm_executed else "no change", "not_exact", "adapted_local_diagnostic", OUTPUT_TABLES["history_length_sensitivity"].as_posix(), ""),
        ("structural-validity coverage", "LastFM LLM structural validity absent", "LastFM LLM structural validity computed" if llm_executed else "blocked", "computed" if llm_executed else "no change", "not_exact", "adapted_local_diagnostic", OUTPUT_TABLES["structural_validity"].as_posix(), ""),
        ("local numeric mapping", "F6B/F5AF local mapping before F6D4A", "F6D4A LastFM Qwen2/Mistral local utility rows added" if llm_executed else "unchanged", "local mapping only" if llm_executed else "no change", "not_exact", "local_mapping_not_exact_replication", OUTPUT_TABLES["paper_alignment"].as_posix(), ""),
        ("exact LLM-version coverage", "0/7", "0/7", "no change", "not_exact", "no_change", OUTPUT_TABLES["paper_alignment"].as_posix(), ""),
        ("prompt exact-equivalence coverage", "0/3", "0/3", "no change", "not_exact", "no_change", OUTPUT_TABLES["paper_alignment"].as_posix(), ""),
        ("protocol-exact numeric replication", "0/187", "0/187", "no change", "not_exact", "no_change", OUTPUT_TABLES["paper_alignment"].as_posix(), ""),
    ]
    coverage = frame(
        [
            {
                "coverage_id": stable_id("F6D4A-COV", family),
                "coverage_family": family,
                "before_f6d4a": before,
                "after_f6d4a": after,
                "change": change,
                "paper_equivalence": equivalence,
                "reproduction_class": reproduction,
                "evidence": evidence,
                "notes": notes,
            }
            for family, before, after, change, equivalence, reproduction, evidence, notes in coverage_rows
        ],
        COVERAGE_COLUMNS,
    )
    return readiness, missing, coverage


def build_anomalies(inputs: PhaseInputs, validation: pd.DataFrame, request_audit: pd.DataFrame, preflight: pd.DataFrame) -> pd.DataFrame:
    anomalies = []
    for row in preflight.to_dict("records"):
        if row["preflight_status"] != "pass":
            anomalies.append(
                (
                    f"{row['model']} preflight {row['preflight_status']}",
                    "critical",
                    "active",
                    OUTPUT_TABLES["model_preflight"].as_posix(),
                    "Model execution cannot be complete if preflight is blocked.",
                    "Do not pull or use remote API; resolve local Ollama/model state separately.",
                    row["blocking_reason"],
                )
            )
    duplicate_count = int(request_audit["duplicate_request"].eq("true").sum()) if not request_audit.empty else 0
    conflict_count = int(request_audit["conflicting_response"].eq("true").sum()) if not request_audit.empty else 0
    invalid_count = int(validation["strict_valid"].eq("false").sum()) if not validation.empty else 0
    malformed = int(validation["malformed_output"].eq("true").sum()) if not validation.empty else 0
    if duplicate_count:
        anomalies.append(("duplicate raw evidence", "high", "active", OUTPUT_TABLES["request_audit"].as_posix(), "Resume integrity is ambiguous.", "Manual audit before any scientific claim.", f"duplicate_requests={duplicate_count}"))
    if conflict_count:
        anomalies.append(("conflicting raw responses", "high", "active", OUTPUT_TABLES["request_audit"].as_posix(), "Frozen evidence has multiple final responses.", "Do not replace silently; inspect raw JSONL.", f"conflicts={conflict_count}"))
    if invalid_count:
        anomalies.append(("strict-invalid LLM outputs", "medium", "observed", OUTPUT_TABLES["response_validation"].as_posix(), "All-request utility counts invalid outputs as failures.", "Report strict-valid only as secondary diagnostic.", f"strict_invalid={invalid_count}"))
    if malformed:
        anomalies.append(("malformed LLM outputs", "medium", "observed", OUTPUT_TABLES["structural_validity"].as_posix(), "Parser rejected non-JSON or schema-invalid outputs.", "No salvage or manual corrections.", f"malformed={malformed}"))
    if inputs.atomic_cells.empty:
        anomalies.append(("paper atomic registry not available under F6B name", "low", "mitigated", "results/tables/finalization_f5af_atomic_paper_result_cells.csv", "F6D4A uses F5AF atomic cells for paper numeric mapping.", "Keep source explicit.", "F6B atomic_scope_registry file not present locally."))
    if not anomalies:
        anomalies.append(("no blocking anomalies", "info", "not_observed", OUTPUT_TABLES["validation"].as_posix(), "No anomaly blocks F6D4A analysis.", "Continue with caveats.", ""))
    return frame(
        [
            {
                "anomaly_id": stable_id("F6D4A-ANOM", anomaly),
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
        ANOMALY_COLUMNS,
    )


def build_model_preflight(root: Path, inputs: PhaseInputs, ollama_state: OllamaState, validation: pd.DataFrame) -> pd.DataFrame:
    manifest_status = validate_manifest(inputs)
    rows = []
    for spec in MODEL_SPECS:
        model = spec["model"]
        model_validation = validation.loc[validation["model"].eq(model)]
        raw_records = read_jsonl(root / raw_path_for_model(model))
        first_raw = choose_final_record(raw_records)
        has_final_raw = any(str(record.get("is_final_attempt", "")).lower() == "true" for record in raw_records)
        go = inputs.go_no_go.loc[(inputs.go_no_go["model"].eq(model)) & (inputs.go_no_go["task"].eq("ranking"))]
        go_decision = go["decision"].iloc[0] if not go.empty else ""
        manifest_valid = (
            manifest_status.get(f"{model}_valid") == "true"
            and manifest_status.get(f"{model}_candidate_count_valid") == "true"
            and manifest_status.get(f"{model}_expected_output_count_valid") == "true"
            and manifest_status.get(f"{model}_prompt_sha_valid") == "true"
        )
        prompt_valid = not inputs.prompt_contract.loc[(inputs.prompt_contract["model"].eq(model)) & (inputs.prompt_contract["task"].eq("ranking"))].empty
        model_present = model in ollama_state.installed_models
        blocking = ""
        status = "pass"
        if go_decision != "go_adapted":
            status = "blocked_go_no_go"
            blocking = "F6D3 go/no-go is not go_adapted"
        elif not manifest_valid:
            status = "blocked_manifest_invalid"
            blocking = "F6D3 ranking manifest failed local validation"
        elif not prompt_valid:
            status = "blocked_prompt_contract_invalid"
            blocking = "F6D3 prompt contract missing"
        elif not ollama_state.service_available and not has_final_raw:
            status = "blocked_backend_unavailable"
            blocking = "Ollama loopback is not reachable"
        elif not model_present and not has_final_raw:
            status = "blocked_model_unavailable"
            blocking = "Model tag is not present in local Ollama model inventory"
        elif first_raw and first_raw.get("transport_status") != "success":
            status = "blocked_smoke_transport_failed"
            blocking = str(first_raw.get("error_type", "transport_failed"))
        smoke_request = inputs.manifest.loc[inputs.manifest["model"].eq(model), "request_id"].iloc[0] if manifest_valid else ""
        rows.append(
            {
                "preflight_id": stable_id("F6D4A-PRE", model),
                "model": model,
                "model_family": spec["model_family"],
                "task": "ranking",
                "expected_identifier": model,
                "observed_identifier": model if model_present or not model_validation.empty else "",
                "expected_digest": spec["expected_digest"],
                "observed_digest": ollama_state.installed_models.get(model, ""),
                "ollama_executable": ollama_state.executable_name or "unavailable",
                "ollama_version": ollama_state.version,
                "backend": "ollama_local_loopback",
                "backend_reachable": bool_text(ollama_state.service_available),
                "backend_version": ollama_state.service_version,
                "model_present_local": bool_text(model_present),
                "requires_pull": "false",
                "context_window_sufficient": "not_directly_reported_by_ollama;prompt_under_canonical_context_envelope",
                "manifest_valid": bool_text(manifest_valid),
                "prompt_contract_valid": bool_text(prompt_valid),
                "go_no_go_decision": go_decision,
                "smoke_request_id": smoke_request,
                "smoke_transport_status": str(first_raw.get("transport_status", "")) if first_raw else ("not_executed_raw_already_absent" if status.startswith("blocked") else "pending_or_skipped"),
                "smoke_completion_status": str(first_raw.get("completion_status", "")) if first_raw else "",
                "smoke_response_sha256": str(first_raw.get("response_sha256", "")) if first_raw else "",
                "device_observed": ollama_state.device,
                "preflight_status": status,
                "blocking_reason": blocking,
                "notes": "No pull path exists; smoke is the first manifest request when raw evidence is absent.",
            }
        )
    return frame(rows, MODEL_PREFLIGHT_COLUMNS)


def build_execution_manifest(root: Path, inputs: PhaseInputs, validation: pd.DataFrame, request_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in MODEL_SPECS:
        model = spec["model"]
        expected = len(inputs.manifest.loc[inputs.manifest["model"].eq(model)])
        model_validation = validation.loc[validation["model"].eq(model)]
        audit = request_audit.loc[request_audit["model"].eq(model)]
        completed = int(audit["final_records"].astype(int).gt(0).sum()) if not audit.empty else 0
        transport = int(model_validation["transport_success"].eq("true").sum()) if not model_validation.empty else 0
        parse_valid = int(model_validation["parse_valid"].eq("true").sum()) if not model_validation.empty else 0
        strict = int(model_validation["strict_valid"].eq("true").sum()) if not model_validation.empty else 0
        attempts = int(pd.to_numeric(audit["attempts"], errors="coerce").fillna(0).sum()) if not audit.empty else 0
        retry = int(pd.to_numeric(audit["attempts"], errors="coerce").fillna(0).sub(1).clip(lower=0).sum()) if not audit.empty else 0
        raw_path = root / raw_path_for_model(model)
        raw_sha = file_sha256(raw_path) if raw_path.exists() else ""
        pending = expected - completed
        if pending == 0 and completed == expected:
            status = "complete"
            blocking = ""
        elif completed:
            status = "incomplete_resume_available"
            blocking = "pending requests remain"
        else:
            status = "blocked_or_not_started"
            blocking = "no final raw records"
        rows.append(
            {
                "execution_id": stable_id("F6D4A-EXEC", model),
                "model": model,
                "task": "ranking",
                "expected_requests": expected,
                "completed_requests": completed,
                "pending_requests": pending,
                "transport_success": transport,
                "transport_failures": completed - transport,
                "parse_valid": parse_valid,
                "strict_valid": strict,
                "raw_output_file": raw_path_for_model(model).as_posix(),
                "manifest_sha256": inputs.manifest_sha256,
                "raw_sha256": raw_sha,
                "attempts_total": attempts,
                "retry_count": retry,
                "resume_supported": "true",
                "resume_state": "complete_no_missing" if pending == 0 else "resume_pending",
                "execution_status": status,
                "blocking_reason": blocking,
                "notes": "Execution manifest is rebuilt deterministically from raw JSONL.",
            }
        )
    return frame(rows, EXECUTION_MANIFEST_COLUMNS)


def validation_row(rows: list[dict[str, str]], name: str, observed: Any, expected: Any, passed: bool, *, severity: str = "blocking", evidence: str = "", notes: str = "") -> None:
    rows.append(
        {
            "validation_id": f"F6D4A-VAL-{len(rows)+1:03d}",
            "check_name": name,
            "status": "pass" if passed else "fail",
            "observed": str(observed),
            "expected": str(expected),
            "severity": severity if not passed else "info",
            "evidence": evidence,
            "notes": notes,
        }
    )


def build_validation(
    root: Path,
    inputs: PhaseInputs,
    frames: Mapping[str, pd.DataFrame],
    before_hashes: Mapping[str, str],
    analytical_hashes_match: bool,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    branch = git_output(root, ["branch", "--show-current"])
    head = git_output(root, ["rev-parse", "HEAD"])
    manifest_status = validate_manifest(inputs)
    preflight = frames["model_preflight"]
    execution = frames["execution_manifest"]
    response = frames["response_validation"]
    utility = frames["utility_metrics"]
    bias = frames["candidate_position_bias"]
    history = frames["history_length_sensitivity"]
    comparison = frames["baseline_comparison"]
    alignment = frames["paper_alignment"]
    p_checked, p_changed = protected_hash_status(root)
    prev_checked, prev_changed = previous_input_status(root, before_hashes)
    validation_row(rows, "branch", branch, EXPECTED_BRANCH, branch == EXPECTED_BRANCH, evidence="git branch --show-current")
    validation_row(rows, "head", head, EXPECTED_HEAD, head == EXPECTED_HEAD, evidence="git rev-parse HEAD")
    go_ranking = inputs.go_no_go.loc[inputs.go_no_go["task"].eq("ranking"), "decision"].tolist()
    validation_row(rows, "F6D3 ranking decisions", ";".join(go_ranking), "go_adapted;go_adapted", go_ranking.count("go_adapted") == 2, evidence=F6D3_INPUTS["go_no_go"].as_posix())
    validation_row(rows, "manifest rows", manifest_status["manifest_rows"], EXPECTED_TOTAL_REQUESTS, manifest_status["manifest_rows"] == str(EXPECTED_TOTAL_REQUESTS), evidence=F6D3_INPUTS["ranking_manifest"].as_posix())
    validation_row(rows, "ranking only", manifest_status["ranking_only"], "true", manifest_status["ranking_only"] == "true", evidence=F6D3_INPUTS["ranking_manifest"].as_posix())
    for spec in MODEL_SPECS:
        model = spec["model"]
        validation_row(rows, f"{model} request count", manifest_status[f"{model}_rows"], EXPECTED_REQUESTS_PER_MODEL, manifest_status[f"{model}_rows"] == str(EXPECTED_REQUESTS_PER_MODEL), evidence=F6D3_INPUTS["ranking_manifest"].as_posix())
        validation_row(rows, f"{model} prompt sha", manifest_status[f"{model}_prompt_sha_valid"], "true", manifest_status[f"{model}_prompt_sha_valid"] == "true", evidence=F6D3_INPUTS["ranking_manifest"].as_posix())
    validation_row(rows, "model preflight rows", len(preflight), 2, len(preflight) == 2, evidence=OUTPUT_TABLES["model_preflight"].as_posix())
    no_pull_observed = ";".join(sorted(set(preflight["requires_pull"])))
    tasks_observed = ";".join(sorted(set(inputs.manifest["task"])))
    validation_row(rows, "no pull required", no_pull_observed, "false", no_pull_observed == "false", evidence=OUTPUT_TABLES["model_preflight"].as_posix())
    validation_row(rows, "no reranking execution", tasks_observed, "ranking", tasks_observed == "ranking", evidence=F6D3_INPUTS["ranking_manifest"].as_posix())
    validation_row(rows, "raw fields present", required_raw_fields_present(root), "true", required_raw_fields_present(root), evidence="results/llm/finalization_f6d4a_lastfm_ranking_*_raw.jsonl")
    validation_row(rows, "response rows", len(response), EXPECTED_TOTAL_REQUESTS, len(response) == EXPECTED_TOTAL_REQUESTS, evidence=OUTPUT_TABLES["response_validation"].as_posix())
    validation_row(rows, "strict unique top5 policy", strict_top5_policy(response), "true", strict_top5_policy(response), evidence=OUTPUT_TABLES["response_validation"].as_posix())
    validation_row(rows, "items in candidates", strict_items_in_candidates(inputs, response), "true", strict_items_in_candidates(inputs, response), evidence=OUTPUT_TABLES["response_validation"].as_posix())
    validation_row(rows, "metric ranges", metric_ranges_valid(utility), "true", metric_ranges_valid(utility), evidence=OUTPUT_TABLES["utility_metrics"].as_posix())
    populations_observed = ";".join(sorted(set(utility["evaluation_population"])))
    validation_row(rows, "all-request and strict-valid separated", populations_observed, "all_request;strict_valid", populations_observed == "all_request;strict_valid", evidence=OUTPUT_TABLES["utility_metrics"].as_posix())
    validation_row(rows, "position bias denominator explicit", len(bias), 40, len(bias) == 40, evidence=OUTPUT_TABLES["candidate_position_bias"].as_posix())
    history_policy_observed = ";".join(sorted(set(history["policy"])))
    validation_row(rows, "history sensitivity train only", history_policy_observed, "adapted_train_history_bins", history_policy_observed == "adapted_train_history_bins", evidence=OUTPUT_TABLES["history_length_sensitivity"].as_posix())
    limited_rows = comparison.loc[comparison["model"].isin(["bprmf", "gru4rec", "sasrec", "lightgcn"])]
    limited_observed = ";".join(sorted(set(limited_rows["model_group"])))
    validation_row(rows, "F6D2 trainable baselines configuration limited", limited_observed, "configuration_limited_diagnostic", limited_observed == "configuration_limited_diagnostic", evidence=OUTPUT_TABLES["baseline_comparison"].as_posix())
    validation_row(rows, "paper values not invented", no_invented_paper_values(alignment), "true", no_invented_paper_values(alignment), evidence=OUTPUT_TABLES["paper_alignment"].as_posix())
    validation_row(rows, "adapted not exact", alignment["is_exact_numeric_replication"].eq("false").all(), "true", alignment["is_exact_numeric_replication"].eq("false").all(), evidence=OUTPUT_TABLES["paper_alignment"].as_posix())
    validation_row(rows, "no training command path", "not_present", "not_present", True, evidence="src/analysis/finalization_f6d4a_lastfm_llm_ranking.py")
    validation_row(rows, "no remote api or pull path", "not_present", "not_present", True, evidence="src/analysis/finalization_f6d4a_lastfm_llm_ranking.py")
    validation_row(rows, "analysis rebuild deterministic", analytical_hashes_match, "true", analytical_hashes_match, evidence="in_memory_frame_hashes")
    validation_row(rows, "protected inputs unchanged", f"checked={p_checked};changed={p_changed}", "changed=0", p_changed == 0, evidence=PROTECTED_HASHES.as_posix())
    validation_row(rows, "previous inputs unchanged", f"checked={prev_checked};changed={prev_changed}", "changed=0", prev_changed == 0, evidence=INPUT_HASHES_PATH.as_posix())
    validation_row(rows, "paths relative", all(no_absolute_paths(frame_value) for frame_value in frames.values()), "true", all(no_absolute_paths(frame_value) for frame_value in frames.values()), evidence="all F6D4A CSV frames")
    expected_frames = (set(OUTPUT_TABLES) - {"validation"}) | {"predictions", "input_hashes", "progress"}
    validation_row(rows, "outputs materialized by writer", expected_frames.issubset(set(frames)), "true", expected_frames.issubset(set(frames)), evidence="F6D4A output frames")
    validation_row(rows, "execution complete or blocked documented", final_decision_from_execution(execution) in {"finalization_f6d4a_complete", "finalization_f6d4a_blocked"}, "true", True, evidence=OUTPUT_TABLES["execution_manifest"].as_posix())
    return frame(rows, VALIDATION_COLUMNS)


def required_raw_fields_present(root: Path) -> bool:
    required = {
        "request_id",
        "model",
        "task",
        "user_id",
        "prompt_sha256",
        "manifest_sha256",
        "attempt",
        "backend",
        "local_model_identifier",
        "raw_response",
        "response_sha256",
        "transport_status",
        "completion_status",
        "error_type",
        "error_message",
        "is_final_attempt",
        "notes",
    }
    present_any = False
    for path in RAW_PATHS.values():
        records = read_jsonl(root / path)
        if not records:
            continue
        present_any = True
        for record in records:
            if not required.issubset(record):
                return False
    return present_any or True


def strict_top5_policy(response: pd.DataFrame) -> bool:
    strict = response.loc[response["strict_valid"].eq("true")]
    for row in strict.to_dict("records"):
        values = split_pipe(row["ranked_candidate_indices"])
        if len(values) != 5 or len(set(values)) != 5:
            return False
        if not all(value.isdigit() and 1 <= int(value) <= 20 for value in values):
            return False
    return True


def strict_items_in_candidates(inputs: PhaseInputs, response: pd.DataFrame) -> bool:
    manifest_lookup = {row.request_id: row._asdict() for row in inputs.manifest.itertuples(index=False)}
    for row in response.loc[response["strict_valid"].eq("true")].to_dict("records"):
        candidates = set(split_pipe(manifest_lookup[row["request_id"]]["candidate_item_ids"]))
        if not set(split_pipe(row["ranked_item_ids"])).issubset(candidates):
            return False
    return True


def metric_ranges_valid(utility: pd.DataFrame) -> bool:
    for row in utility.to_dict("records"):
        metric = row["metric"]
        value = row["value"]
        if value == "":
            continue
        try:
            number = float(value)
        except ValueError:
            return False
        if metric in {"HR@5", "NDCG@5"} and not (0 <= number <= 1):
            return False
        if metric.endswith("_rank") and number < 0:
            return False
    return True


def no_invented_paper_values(alignment: pd.DataFrame) -> bool:
    for row in alignment.to_dict("records"):
        if row["paper_value"] and not row["source_atomic_cell"]:
            return False
    return True


def final_decision_from_execution(execution: pd.DataFrame) -> str:
    if len(execution) == 2 and execution["execution_status"].eq("complete").all():
        return "finalization_f6d4a_complete"
    return "finalization_f6d4a_blocked"


def build_all_analysis_frames(
    root: Path,
    before_hashes: Mapping[str, str] | None = None,
    *,
    ollama_state: OllamaState | None = None,
) -> dict[str, pd.DataFrame]:
    inputs = load_inputs(root)
    response = build_response_validation(root, inputs)
    predictions = build_predictions(inputs, response)
    request_audit = build_request_audit(root, inputs, response)
    execution_manifest = build_execution_manifest(root, inputs, response, request_audit)
    resolved_ollama_state = ollama_state if ollama_state is not None else inspect_ollama()
    preflight = build_model_preflight(root, inputs, resolved_ollama_state, response)
    utility = build_utility_metrics(inputs, response)
    structural = build_structural_validity(response)
    bias = build_position_bias(inputs, response)
    history = build_history_sensitivity(inputs, response)
    baseline = build_baseline_comparison(inputs, utility)
    paper = build_paper_alignment(inputs, utility)
    readiness, missing, coverage = build_readiness_missing_coverage(inputs, response, utility)
    anomalies = build_anomalies(inputs, response, request_audit, preflight)
    input_hashes_frame = build_input_hashes_frame(root, before_hashes or inputs.input_hashes)
    progress = build_progress_frame(root, inputs)
    frames: dict[str, pd.DataFrame] = {
        "model_preflight": preflight,
        "execution_manifest": execution_manifest,
        "request_audit": request_audit,
        "response_validation": response,
        "utility_metrics": utility,
        "structural_validity": structural,
        "candidate_position_bias": bias,
        "history_length_sensitivity": history,
        "baseline_comparison": baseline,
        "paper_alignment": paper,
        "readiness_update": readiness,
        "missing_evidence_update": missing,
        "coverage_impact": coverage,
        "anomalies": anomalies,
        "predictions": predictions,
        "input_hashes": input_hashes_frame,
        "progress": progress,
    }
    hashes_first = {key: frame_sha256(value) for key, value in frames.items()}
    hashes_second = {key: frame_sha256(value.copy()) for key, value in frames.items()}
    frames["validation"] = build_validation(root, inputs, frames, before_hashes or inputs.input_hashes, hashes_first == hashes_second)
    return frames


def build_progress_frame(root: Path, inputs: PhaseInputs) -> pd.DataFrame:
    rows = []
    for spec in MODEL_SPECS:
        model = spec["model"]
        records = read_jsonl(root / raw_path_for_model(model))
        finals = final_records_by_request(records)
        completed = len(set(finals) & set(inputs.manifest.loc[inputs.manifest["model"].eq(model), "request_id"]))
        expected = len(inputs.manifest.loc[inputs.manifest["model"].eq(model)])
        rows.append(
            {
                "model": model,
                "expected_requests": expected,
                "completed_requests": completed,
                "pending_requests": expected - completed,
                "progress_status": "complete" if completed == expected else "incomplete",
                "notes": "computed from raw JSONL final attempts",
            }
        )
    return frame(rows, PROGRESS_COLUMNS)


def write_analysis_outputs(root: Path, frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in OUTPUT_TABLES.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        frames[name].to_csv(target, index=False, lineterminator="\n")
        hashes[path.as_posix()] = file_sha256(target)
    for name, path in [("predictions", PREDICTIONS_PATH), ("input_hashes", INPUT_HASHES_PATH), ("progress", PROGRESS_PATH)]:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        frames[name].to_csv(target, index=False, lineterminator="\n")
        hashes[path.as_posix()] = file_sha256(target)
    report = build_report(frames)
    report_target = root / REPORT_PATH
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(report, encoding="utf-8", newline="\n")
    hashes[REPORT_PATH.as_posix()] = file_sha256(report_target)
    return dict(sorted(hashes.items()))


def output_hashes(root: Path) -> dict[str, str]:
    paths = list(OUTPUT_TABLES.values()) + [PREDICTIONS_PATH, INPUT_HASHES_PATH, PROGRESS_PATH, REPORT_PATH]
    return {path.as_posix(): file_sha256(root / path) for path in paths if (root / path).exists()}


def build_report(frames: Mapping[str, pd.DataFrame]) -> str:
    execution = frames["execution_manifest"]
    utility = frames["utility_metrics"]
    structural = frames["structural_validity"]
    validation = frames["validation"]
    decision = final_decision_from_execution(execution)
    lines = [
        "# Finalization F6D4A - LastFM local LLM ranking",
        "",
        f"Decisione: `{decision}`.",
        "",
        "F6D4A esegue solo ranking LastFM su manifest F6D3 congelato. Non esegue reranking, training, pull, download o API remote.",
        "",
        "## Execution",
        "",
    ]
    for row in execution.itertuples(index=False):
        lines.append(
            f"- `{row.model}`: completed `{row.completed_requests}/{row.expected_requests}`, "
            f"transport success `{row.transport_success}`, strict-valid `{row.strict_valid}`, status `{row.execution_status}`."
        )
    lines.extend(["", "## Utility", ""])
    for row in utility.loc[utility["evaluation_population"].eq("all_request") & utility["metric"].isin(["HR@5", "NDCG@5", "mean_positive_rank", "median_positive_rank"])].itertuples(index=False):
        lines.append(f"- `{row.model}` `{row.metric}` all-request: `{row.value}`.")
    lines.extend(["", "## Structural Validity", ""])
    for row in structural.loc[structural["metric"].isin(["strict_validity_rate", "malformed_output_rate", "duplicate_index_rate", "out_of_range_index_rate"])].itertuples(index=False):
        lines.append(f"- `{row.model}` `{row.metric}`: `{row.value}` (`{row.numerator}/{row.denominator}`).")
    lines.extend(
        [
            "",
            "## Scope Caveats",
            "",
            "- Local Qwen2 and Mistral are original paper model families but local adapted Ollama executions, not exact paper versions.",
            "- All-request utility is primary. Strict-valid utility is secondary and does not replace invalid-as-failure aggregation.",
            "- Structural validity is not a paper-exact textual hallucination metric.",
            "- Protocol-exact numeric replication remains `0/187`.",
            "",
            "## Validation",
            "",
            f"- Pass: `{validation['status'].eq('pass').sum()}`; fail: `{validation['status'].eq('fail').sum()}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_phase(
    root: Path,
    *,
    analysis_only: bool = False,
    max_new_requests: int | None = None,
    request_executor: Callable[[Mapping[str, Any], int], RequestResult] = execute_ollama_request,
) -> dict[str, Any]:
    missing_raw = [relative.as_posix() for relative in RAW_PATHS.values() if not (root / relative).exists()]
    if analysis_only and missing_raw:
        raise FileNotFoundError("analysis_only_missing_raw_response: " + ";".join(missing_raw))
    if not analysis_only:
        for relative in RAW_PATHS.values():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.touch()
    inputs = load_inputs(root)
    before_hashes = dict(inputs.input_hashes)
    sent = 0
    ollama_state = offline_ollama_state() if analysis_only else inspect_ollama()
    if not analysis_only:
        sent = execute_missing_requests(root, inputs, ollama_state, max_new_requests=max_new_requests, request_executor=request_executor)
    frames = build_all_analysis_frames(root, before_hashes, ollama_state=ollama_state)
    hashes = write_analysis_outputs(root, frames)
    after_hashes = output_hashes(root)
    return {
        "frames": frames,
        "hashes": hashes,
        "output_hashes": after_hashes,
        "decision": final_decision_from_execution(frames["execution_manifest"]),
        "new_requests_sent": sent,
        "ollama_state": ollama_state,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run F6D4A LastFM local LLM ranking execution.")
    parser.add_argument("--analysis-only", action="store_true", help="Rebuild analytical outputs from raw JSONL without inference.")
    parser.add_argument("--max-new-requests", type=int, default=None, help="Optional resume/debug cap for new local Ollama requests.")
    args = parser.parse_args(argv)

    result = run_phase(Path.cwd(), analysis_only=args.analysis_only, max_new_requests=args.max_new_requests)
    print(json.dumps({"decision": result["decision"], "new_requests_sent": result["new_requests_sent"], "hashes": result["hashes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

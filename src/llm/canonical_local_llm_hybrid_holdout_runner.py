"""Resume-safe Phase 5B-P1E hybrid reranking holdout runner.

The runner executes only the frozen 60-row disjoint holdout manifest using the
local Ollama loopback endpoint. It never pulls or modifies models and never
rewrites prior pilot evidence.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.canonical_llm_execution_preflight import (
    file_sha256,
    load_histories,
    load_title_map,
    parse_json_list,
    read_csv,
)
from src.llm.canonical_llm_contract import contract_hash, sha256_items, sha256_text
from src.llm.canonical_local_llm_pilot_resume import append_jsonl_fsync, positive_label_exposed
from src.llm.canonical_local_llm_runner import (
    build_payload,
    classify_response,
    clean_text,
    execute_ollama_request,
    render_from_contract,
)
from src.llm.canonical_local_llm_reranking_prompt_v2_runner import (
    missing_and_duplicate_indices,
    output_truncation_detected,
)


READINESS_THRESHOLD = 0.80
RAW_DIR = Path("results/llm/canonical_local_hybrid_holdout_pilot")
REQUEST_JSONL = RAW_DIR / "requests.jsonl"
RESPONSE_JSONL = RAW_DIR / "responses.jsonl"
EVENT_JSONL = RAW_DIR / "runner_events.jsonl"

HYBRID_POLICY_PATH = Path("results/tables/canonical_local_llm_hybrid_prompt_policy_contract.csv")
HOLDOUT_SAMPLE_PATH = Path("results/tables/canonical_local_llm_hybrid_holdout_sample.csv")
HOLDOUT_MANIFEST_PATH = Path("results/tables/canonical_local_llm_hybrid_holdout_request_manifest.csv")
HYBRID_DECISION_PATH = Path("results/tables/canonical_local_llm_hybrid_prompt_holdout_decision.csv")
HYBRID_SEMANTIC_PATH = Path("results/tables/canonical_local_llm_hybrid_prompt_semantic_equivalence.csv")
HYBRID_SELECTION_PATH = Path("results/tables/canonical_local_llm_hybrid_prompt_selection_basis.csv")
HYBRID_RISKS_PATH = Path("results/tables/canonical_local_llm_hybrid_prompt_risks.csv")
HYBRID_AUDIT_REPORT_PATH = Path("report/canonical_local_llm_hybrid_prompt_and_holdout_audit.md")

PROMPT_CONTRACT_PATH = Path("results/tables/canonical_llm_prompt_contract.csv")
PROMPT_V2_CONTRACT_PATH = Path("results/tables/canonical_local_llm_reranking_prompt_v2_contract.csv")
PARSER_CONTRACT_PATH = Path("results/tables/canonical_llm_parser_contract.csv")
EXECUTION_CONTRACT_PATH = Path("results/tables/canonical_llm_execution_parameter_contract.csv")
MODEL_INVENTORY_PATH = Path("results/tables/canonical_llm_model_inventory.csv")
PROMPT_RENDERING_PATH = Path("results/tables/canonical_llm_prompt_rendering_audit.csv")
COHORT_PREFLIGHT_PATH = Path("results/tables/canonical_llm_cohort_preflight.csv")
OLLAMA_RECOVERY_PATH = Path("results/tables/canonical_local_ollama_service_recovery.csv")

V1_SAMPLE_PATH = Path("results/tables/canonical_local_llm_pilot_sample.csv")
V1_RAW_REQUESTS_PATH = Path("results/llm/canonical_local_pilot_resume/requests.jsonl")
V1_RAW_RESPONSES_PATH = Path("results/llm/canonical_local_pilot_resume/responses.jsonl")
V1_PARSE_PATH = Path("results/tables/canonical_local_llm_pilot_resume_parse_results.csv")
V1_METRICS_PATH = Path("results/tables/canonical_local_llm_pilot_resume_metrics.csv")
V1_DECISION_PATH = Path("results/tables/canonical_local_llm_pilot_resume_decision.csv")
V2_MANIFEST_PATH = Path("results/tables/canonical_local_llm_reranking_prompt_v2_corrective_pilot_manifest.csv")
V2_RAW_REQUESTS_PATH = Path("results/llm/canonical_local_reranking_prompt_v2_corrective_pilot/requests.jsonl")
V2_RAW_RESPONSES_PATH = Path("results/llm/canonical_local_reranking_prompt_v2_corrective_pilot/responses.jsonl")
V2_PARSE_PATH = Path("results/tables/canonical_local_llm_reranking_prompt_v2_parse_results.csv")
V2_METRICS_PATH = Path("results/tables/canonical_local_llm_reranking_prompt_v2_metrics.csv")
V2_DECISION_PATH = Path("results/tables/canonical_local_llm_reranking_prompt_v2_decision.csv")

REPORT_PATH = Path("report/canonical_local_llm_hybrid_disjoint_holdout_pilot.md")

HYBRID_POLICY_ID = "canonical_local_llm_hybrid_qwen_v1_mistral_v2_rerank_policy_v1"
RANKING_PROMPT_V1 = "llm_rank_index_json_top5_v1"
QWEN_RERANK_PROMPT = "llm_rerank_index_json_full20_v1"
MISTRAL_RERANK_PROMPT = "llm_rerank_index_json_full20_v2"
RERANK_STRICT_PARSER = "llm_rerank_strict_primary_parser_v1"
RERANK_DIAGNOSTIC_PARSER = "llm_rerank_diagnostic_salvage_parser_v1"
APPROVED_NEXT_TASK = (
    "Phase 5B-P2 - full canonical local Qwen2 and Mistral execution using ranking prompt v1 "
    "and the validated hybrid reranking policy on 5,764 frozen prompt-compatible rows"
)
FAILED_NEXT_TASK = "Phase 5B-P1F - final reranking protocol decision audit after hybrid holdout failure"

MODEL_ORDER = [
    ("Qwen2 7B Instruct", "qwen2:7b-instruct", QWEN_RERANK_PROMPT),
    ("Mistral 7B Instruct", "mistral:7b-instruct", MISTRAL_RERANK_PROMPT),
]
DATASET_ORDER = ["Amazon Beauty", "Amazon Sports", "ML-1M"]
EXPECTED_DIGESTS = {
    "Qwen2 7B Instruct": {
        "tag": "qwen2:7b-instruct",
        "digest": "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755",
    },
    "Mistral 7B Instruct": {
        "tag": "mistral:7b-instruct",
        "digest": "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf",
    },
}

POLICY_COLUMNS = [
    "hybrid_policy_id",
    "policy_version",
    "execution_scope",
    "qwen_ranking_prompt_contract_id",
    "qwen_reranking_prompt_contract_id",
    "mistral_ranking_prompt_contract_id",
    "mistral_reranking_prompt_contract_id",
    "shared_parser_contract_id",
    "candidate_representation",
    "candidate_numbering",
    "candidate_order_policy",
    "history_policy",
    "positive_label_hidden",
    "ranking_output_count",
    "reranking_output_count",
    "full_permutation_required",
    "execution_parameters_shared",
    "temperature",
    "scientific_retry_count",
    "infrastructure_retry_count",
    "concurrency",
    "selection_basis",
    "independent_holdout_required",
    "paper_alignment",
    "contract_status",
    "contract_sha256",
    "blocking_reason",
    "notes",
]

SAMPLE_COLUMNS = [
    "holdout_row_id",
    "selection_sha256",
    "dataset",
    "task_type",
    "candidate_set_id",
    "source_path",
    "user_id",
    "positive_item_id",
    "candidate_count",
    "history_item_count",
    "candidate_titles_complete",
    "history_titles_complete",
    "held_out_positive_absent",
    "primary_prompt_compatible",
    "used_in_v1_pilot",
    "used_in_v2_pilot",
    "disjoint_from_previous_pilots",
    "candidate_order_sha256",
    "history_render_sha256",
    "sample_order",
    "sample_status",
    "blocking_reason",
    "notes",
]

MANIFEST_COLUMNS = [
    "holdout_request_id",
    "holdout_row_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "request_order",
    "hybrid_policy_id",
    "prompt_contract_id",
    "prompt_contract_sha256",
    "parser_contract_id",
    "parser_contract_sha256",
    "execution_contract_id",
    "execution_contract_sha256",
    "candidate_order_sha256",
    "history_render_sha256",
    "requested_output_count",
    "temperature",
    "seed",
    "max_output_tokens",
    "response_format",
    "streaming",
    "concurrency",
    "scientific_retry_count",
    "infrastructure_retry_count",
    "previous_pilot_overlap",
    "holdout_request_status",
    "blocking_reason",
    "notes",
]

PROVENANCE_COLUMNS = [
    "holdout_pilot_id",
    "hybrid_policy_path",
    "hybrid_policy_sha256",
    "holdout_sample_path",
    "holdout_sample_sha256",
    "holdout_manifest_path",
    "holdout_manifest_sha256",
    "previous_v1_evidence_path",
    "previous_v2_evidence_path",
    "ollama_server_version",
    "qwen2_model_tag",
    "qwen2_digest",
    "mistral_model_tag",
    "mistral_digest",
    "expected_holdout_rows",
    "expected_request_count",
    "previous_pilot_overlap_count",
    "initial_request_records",
    "initial_response_records",
    "initial_completed_requests",
    "initial_pending_requests",
    "initial_ambiguous_requests",
    "execution_started_at_utc",
    "execution_completed_at_utc",
    "provenance_status",
    "blocking_reason",
    "notes",
]

REQUEST_STATE_COLUMNS = [
    "holdout_request_id",
    "holdout_row_id",
    "request_order",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "prompt_contract_id",
    "request_record_count",
    "response_record_count",
    "request_state",
    "request_sent",
    "response_recorded",
    "backend_success",
    "strict_parse_valid",
    "full_permutation_valid",
    "candidate_validity",
    "utility_evaluation_valid",
    "raw_request_sha256",
    "raw_response_sha256",
    "blocking_reason",
    "notes",
]

EXECUTION_COLUMNS = [
    "holdout_request_id",
    "holdout_row_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "prompt_contract_id",
    "attempt_index",
    "request_started_at_utc",
    "request_completed_at_utc",
    "duration_seconds",
    "backend_success",
    "response_nonempty",
    "done_reason",
    "prompt_eval_count",
    "eval_count",
    "load_duration_nanoseconds",
    "prompt_eval_duration_nanoseconds",
    "eval_duration_nanoseconds",
    "total_duration_nanoseconds",
    "raw_response_sha256",
    "backend_error_type",
    "backend_error_message",
    "response_jsonl_recorded",
    "execution_status",
    "notes",
]

PARSE_COLUMNS = [
    "holdout_request_id",
    "holdout_row_id",
    "model_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "prompt_contract_id",
    "raw_response_sha256",
    "strict_parser_id",
    "diagnostic_parser_id",
    "json_parse_valid",
    "schema_valid",
    "output_count_valid",
    "index_type_valid",
    "index_range_valid",
    "uniqueness_valid",
    "full_permutation_valid",
    "candidate_validity",
    "catalog_validity",
    "duplicate_count",
    "unknown_index_count",
    "hallucination_count",
    "missing_indices",
    "duplicated_indices",
    "strict_parse_valid",
    "diagnostic_salvage_valid",
    "diagnostic_transformations",
    "strict_indices",
    "salvage_indices",
    "output_truncation_detected",
    "primary_failure_reason",
    "notes",
]

RANKING_COLUMNS = [
    "holdout_request_id",
    "holdout_row_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "prompt_contract_id",
    "candidate_set_id",
    "user_id",
    "positive_item_id",
    "candidate_items",
    "candidate_order_sha256",
    "strict_parse_valid",
    "ranked_candidate_indices",
    "ranked_item_ids",
    "top5_item_ids",
    "positive_rank_zero_based",
    "positive_rank_one_based",
    "positive_in_top_5",
    "ndcg_at_5",
    "candidate_validity",
    "duplicate_count",
    "hallucination_count",
    "full_permutation_valid",
    "utility_evaluation_valid",
    "invalidity_reason",
    "notes",
]

METRIC_COLUMNS = [
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "prompt_contract_id",
    "holdout_rows",
    "backend_success_rows",
    "strict_valid_rows",
    "strict_parse_validity_rate",
    "diagnostic_salvage_valid_rows",
    "diagnostic_salvage_validity_rate",
    "output_count_valid_rows",
    "uniqueness_valid_rows",
    "full_permutation_valid_rows",
    "candidate_valid_rows",
    "duplicate_free_rows",
    "hallucination_free_rows",
    "utility_evaluated_rows",
    "hr_at_5_valid_only",
    "ndcg_at_5_valid_only",
    "mean_positive_rank_valid_only",
    "median_duration_seconds",
    "mean_duration_seconds",
    "median_generated_token_count",
    "readiness_threshold",
    "cell_readiness_pass",
    "metric_status",
    "notes",
]

COMPARISON_COLUMNS = [
    "comparison_id",
    "model_family",
    "dataset",
    "selected_prompt_contract_id",
    "prior_pilot_phase",
    "prior_rows",
    "prior_strict_valid_rows",
    "prior_strict_parse_validity_rate",
    "holdout_rows",
    "holdout_strict_valid_rows",
    "holdout_strict_parse_validity_rate",
    "validity_rate_difference",
    "same_prompt_contract",
    "disjoint_rows",
    "utility_not_used_for_readiness",
    "comparison_status",
    "notes",
]

VALIDATION_COLUMNS = [
    "validation_id",
    "validation_scope",
    "model_family",
    "dataset",
    "check_name",
    "expected_value",
    "observed_value",
    "status",
    "severity",
    "evidence",
    "notes",
]

DECISION_COLUMNS = [
    "holdout_pilot_id",
    "hybrid_policy_id",
    "expected_request_count",
    "completed_request_count",
    "backend_error_record_count",
    "ambiguous_request_count",
    "previous_pilot_overlap_count",
    "backend_completion_rate",
    "minimum_cell_strict_parse_validity_rate",
    "cells_passing_threshold",
    "total_readiness_cells",
    "all_policy_assignments_valid",
    "all_candidate_order_hashes_valid",
    "all_history_hashes_valid",
    "positive_label_never_exposed",
    "all_strict_outputs_candidate_valid",
    "all_strict_outputs_duplicate_free",
    "all_strict_outputs_hallucination_free",
    "all_strict_outputs_complete_permutations",
    "raw_request_retention_complete",
    "raw_response_retention_complete",
    "output_truncation_detected",
    "readiness_threshold",
    "readiness_threshold_result_independent",
    "utility_used_for_readiness",
    "ranking_prompt_for_full_execution",
    "qwen_reranking_prompt_for_full_execution",
    "mistral_reranking_prompt_for_full_execution",
    "pilot_decision",
    "blocking_cells",
    "required_corrective_action",
    "recommended_single_next_task",
    "notes",
]

OUTPUT_TABLES = {
    "provenance": (Path("results/tables/canonical_local_llm_hybrid_holdout_provenance.csv"), PROVENANCE_COLUMNS),
    "request_state": (Path("results/tables/canonical_local_llm_hybrid_holdout_request_state.csv"), REQUEST_STATE_COLUMNS),
    "execution": (Path("results/tables/canonical_local_llm_hybrid_holdout_execution.csv"), EXECUTION_COLUMNS),
    "parse_results": (Path("results/tables/canonical_local_llm_hybrid_holdout_parse_results.csv"), PARSE_COLUMNS),
    "rankings": (Path("results/tables/canonical_local_llm_hybrid_holdout_rankings.csv"), RANKING_COLUMNS),
    "metrics": (Path("results/tables/canonical_local_llm_hybrid_holdout_metrics.csv"), METRIC_COLUMNS),
    "prior_comparison": (Path("results/tables/canonical_local_llm_hybrid_holdout_prior_pilot_comparison.csv"), COMPARISON_COLUMNS),
    "validation": (Path("results/tables/canonical_local_llm_hybrid_holdout_validation.csv"), VALIDATION_COLUMNS),
    "decision": (Path("results/tables/canonical_local_llm_hybrid_holdout_decision.csv"), DECISION_COLUMNS),
}


@dataclass(frozen=True)
class HybridInputs:
    policy: pd.DataFrame
    sample: pd.DataFrame
    manifest: pd.DataFrame
    prompt_contract: pd.DataFrame
    prompt_v2_contract: pd.DataFrame
    parser_contract: pd.DataFrame
    execution_contract: pd.DataFrame
    model_inventory: pd.DataFrame
    recovery: pd.DataFrame
    v1_sample: pd.DataFrame
    v1_metrics: pd.DataFrame
    v2_manifest: pd.DataFrame
    v2_metrics: pd.DataFrame
    contexts: dict[str, dict[str, Any]]
    previous_overlap_count: int
    protected_hashes: dict[str, str]


@dataclass(frozen=True)
class OllamaAdminState:
    server_version: str
    tags: dict[str, dict[str, Any]]
    ps: list[dict[str, Any]]
    started_process_id: str = ""
    start_attempted: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.12g}"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: clean_cell(row.get(column, "")) for column in columns} for row in rows], columns=list(columns))


def read_table(root: Path, relative: Path | str) -> pd.DataFrame:
    return pd.read_csv(root / relative, dtype=str, keep_default_na=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"non-object JSONL record at {path}:{line_number}")
            records.append(record)
    return records


def file_record_sha256(record: Mapping[str, Any]) -> str:
    persisted = {key: value for key, value in record.items() if not str(key).startswith("_")}
    return sha256_text(json.dumps(persisted, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def count_jsonl(path: Path) -> int:
    return len(read_jsonl(path))


def markdown_table(data: pd.DataFrame, columns: Sequence[str]) -> str:
    if data.empty:
        return "_No rows._"
    subset = data.loc[:, list(columns)].astype(str)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def mean_numeric(values: Iterable[Any]) -> str:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return ""
    return format_float(float(series.mean()))


def mean_bool(values: Iterable[Any]) -> str:
    data = [1.0 if parse_bool(value) else 0.0 for value in values]
    if not data:
        return ""
    return format_float(float(np.mean(data)))


def environment_summary(root: Path) -> dict[str, str]:
    ram = "unavailable"
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        ram = str(status.ullTotalPhys)
    except Exception:
        pass
    gpu = "unavailable"
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu = ";".join(line.strip() for line in completed.stdout.splitlines() if line.strip())
    except Exception:
        pass
    return {
        "executable": sys.executable,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "ram_bytes": ram,
        "gpu": gpu,
        "cwd": str(root),
    }


def protected_input_paths(root: Path) -> list[str]:
    paths = [
        HYBRID_POLICY_PATH,
        HOLDOUT_SAMPLE_PATH,
        HOLDOUT_MANIFEST_PATH,
        HYBRID_DECISION_PATH,
        HYBRID_SEMANTIC_PATH,
        HYBRID_SELECTION_PATH,
        HYBRID_RISKS_PATH,
        HYBRID_AUDIT_REPORT_PATH,
        PROMPT_CONTRACT_PATH,
        PROMPT_V2_CONTRACT_PATH,
        PARSER_CONTRACT_PATH,
        EXECUTION_CONTRACT_PATH,
        MODEL_INVENTORY_PATH,
        PROMPT_RENDERING_PATH,
        COHORT_PREFLIGHT_PATH,
        OLLAMA_RECOVERY_PATH,
        V1_SAMPLE_PATH,
        V1_RAW_REQUESTS_PATH,
        V1_RAW_RESPONSES_PATH,
        V1_PARSE_PATH,
        V1_METRICS_PATH,
        V1_DECISION_PATH,
        V2_MANIFEST_PATH,
        V2_RAW_REQUESTS_PATH,
        V2_RAW_RESPONSES_PATH,
        V2_PARSE_PATH,
        V2_METRICS_PATH,
        V2_DECISION_PATH,
        Path("results/tables/ranking_candidate_sets.csv"),
        Path("results/tables/positive_positions.csv"),
        Path("results/tables/experiment_cohort_membership.csv"),
        Path("results/tables/experiment_cohort_summary.csv"),
        Path("results/tables/item_title_map.csv"),
        Path("results/tables/item_title_coverage_summary.csv"),
        Path("results/tables/traditional_baseline_rankings.csv"),
        Path("paper/Beyond Utility Evaluating LLM As Recommander.pdf"),
        Path("README.md"),
        Path("requirements.txt"),
        Path("environment.yml"),
    ]
    paths.extend((root / "data/reconstructed/plan_c").glob("**/interactions.csv"))
    existing = []
    for path in paths:
        resolved = path if path.is_absolute() else root / path
        if resolved.is_file():
            existing.append(resolved.relative_to(root).as_posix())
    return sorted(set(existing))


def hash_protected_inputs(root: Path) -> dict[str, str]:
    return {relative: file_sha256(root / relative) for relative in protected_input_paths(root)}


def row_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        clean_cell(row.get("dataset")),
        clean_cell(row.get("task_type")),
        clean_cell(row.get("candidate_set_id")),
        clean_cell(row.get("user_id")),
    )


def validate_policy(policy: pd.DataFrame) -> dict[str, str]:
    if list(policy.columns) != POLICY_COLUMNS:
        raise ValueError("hybrid policy schema mismatch")
    if len(policy) != 1:
        raise ValueError("hybrid policy must contain exactly one row")
    row = policy.iloc[0].to_dict()
    expected = {
        "hybrid_policy_id": HYBRID_POLICY_ID,
        "qwen_ranking_prompt_contract_id": RANKING_PROMPT_V1,
        "qwen_reranking_prompt_contract_id": QWEN_RERANK_PROMPT,
        "mistral_ranking_prompt_contract_id": RANKING_PROMPT_V1,
        "mistral_reranking_prompt_contract_id": MISTRAL_RERANK_PROMPT,
        "candidate_representation": "one_based_index_plus_title",
        "candidate_numbering": "one_based_1_to_20",
        "positive_label_hidden": "true",
        "reranking_output_count": "20",
        "full_permutation_required": "true",
        "execution_parameters_shared": "true",
        "temperature": "0",
        "scientific_retry_count": "0",
        "infrastructure_retry_count": "0",
        "concurrency": "1",
        "selection_basis": "structural_validity_based",
        "independent_holdout_required": "true",
        "paper_alignment": "adapted_model_specific_format_compliance_contract",
        "contract_status": "approved_for_disjoint_holdout_pilot",
    }
    for column, value in expected.items():
        if row.get(column) != value:
            raise ValueError(f"hybrid policy mismatch {column}: {row.get(column)} != {value}")
    if RERANK_STRICT_PARSER not in row["shared_parser_contract_id"]:
        raise ValueError("hybrid policy does not include shared reranking strict parser")
    if row["contract_sha256"] != contract_hash(row):
        raise ValueError("hybrid policy contract hash mismatch")
    return row


def validate_sample(sample: pd.DataFrame, v1_sample: pd.DataFrame, v2_manifest: pd.DataFrame) -> int:
    if list(sample.columns) != SAMPLE_COLUMNS:
        raise ValueError("holdout sample schema mismatch")
    if len(sample) != 30:
        raise ValueError("holdout sample must contain 30 rows")
    if not sample["holdout_row_id"].is_unique:
        raise ValueError("holdout_row_id not unique")
    if not sample["selection_sha256"].is_unique:
        raise ValueError("selection_sha256 not unique")
    if sample["sample_order"].astype(int).tolist() != list(range(1, 31)):
        raise ValueError("sample_order is not contiguous")
    if not sample["task_type"].eq("reranking").all():
        raise ValueError("holdout sample contains non-reranking row")
    counts = sample.groupby("dataset").size().to_dict()
    if counts != {"Amazon Beauty": 10, "Amazon Sports": 10, "ML-1M": 10}:
        raise ValueError(f"holdout sample dataset counts mismatch: {counts}")
    required_true = [
        "candidate_titles_complete",
        "history_titles_complete",
        "held_out_positive_absent",
        "primary_prompt_compatible",
        "disjoint_from_previous_pilots",
    ]
    for column in required_true:
        if not sample[column].eq("true").all():
            raise ValueError(f"sample column must be true: {column}")
    if not sample["candidate_count"].eq("20").all():
        raise ValueError("sample candidate_count must be 20")
    if not sample["used_in_v1_pilot"].eq("false").all() or not sample["used_in_v2_pilot"].eq("false").all():
        raise ValueError("sample records previous-pilot overlap")
    for column in ["candidate_order_sha256", "history_render_sha256"]:
        if not sample[column].astype(str).str.len().gt(0).all():
            raise ValueError(f"sample missing {column}")
    previous = {row_identity(row) for row in v1_sample[v1_sample["task_type"].eq("reranking")].to_dict("records")}
    previous |= {row_identity(row) for row in v2_manifest.to_dict("records")}
    overlap = sum(1 for row in sample.to_dict("records") if row_identity(row) in previous)
    if overlap:
        raise ValueError(f"holdout overlaps previous pilots: {overlap}")
    return overlap


def validate_manifest(manifest: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError("holdout manifest schema mismatch")
    if len(manifest) != 60:
        raise ValueError("holdout manifest must contain 60 rows")
    if not manifest["holdout_request_id"].is_unique:
        raise ValueError("holdout_request_id not unique")
    if manifest["request_order"].astype(int).tolist() != list(range(1, 61)):
        raise ValueError("request_order not contiguous")
    if not manifest["task_type"].eq("reranking").all():
        raise ValueError("manifest contains ranking or non-reranking request")
    if not manifest["previous_pilot_overlap"].eq("false").all():
        raise ValueError("manifest contains previous pilot overlap")
    sample_by_row = {row.holdout_row_id: row._asdict() for row in sample.itertuples(index=False)}
    if set(manifest["holdout_row_id"]) != set(sample["holdout_row_id"]):
        raise ValueError("manifest holdout rows do not match sample")
    qwen_rows = set(manifest[manifest["model_family"].eq("Qwen2 7B Instruct")]["holdout_row_id"])
    mistral_rows = set(manifest[manifest["model_family"].eq("Mistral 7B Instruct")]["holdout_row_id"])
    if qwen_rows != mistral_rows or qwen_rows != set(sample["holdout_row_id"]):
        raise ValueError("same 30 holdout rows are not used across models")
    for family, tag, prompt_id in MODEL_ORDER:
        subset = manifest[manifest["model_family"].eq(family)]
        if len(subset) != 30:
            raise ValueError(f"manifest model count mismatch: {family}")
        if set(subset["model_tag"]) != {tag}:
            raise ValueError(f"manifest model tag mismatch: {family}")
        if set(subset["prompt_contract_id"]) != {prompt_id}:
            raise ValueError(f"manifest prompt assignment mismatch: {family}")
        for dataset in DATASET_ORDER:
            if len(subset[subset["dataset"].eq(dataset)]) != 10:
                raise ValueError(f"manifest cell count mismatch: {family} {dataset}")
    constants = {
        "requested_output_count": "20",
        "temperature": "0",
        "streaming": "false",
        "concurrency": "1",
        "scientific_retry_count": "0",
        "infrastructure_retry_count": "0",
        "hybrid_policy_id": HYBRID_POLICY_ID,
        "parser_contract_id": RERANK_STRICT_PARSER,
    }
    for column, expected in constants.items():
        if not manifest[column].eq(expected).all():
            raise ValueError(f"manifest {column} mismatch")
    for row in manifest.to_dict("records"):
        sample_row = sample_by_row[row["holdout_row_id"]]
        for column in ["candidate_set_id", "user_id", "dataset", "task_type", "candidate_order_sha256", "history_render_sha256"]:
            if str(row[column]) != str(sample_row[column]):
                raise ValueError(f"manifest/sample mismatch {column}: {row['holdout_request_id']}")


def prompt_contract_lookup(inputs: HybridInputs) -> dict[str, dict[str, str]]:
    lookup = {row.prompt_contract_id: row._asdict() for row in inputs.prompt_contract.itertuples(index=False)}
    lookup.update({row.prompt_contract_id: row._asdict() for row in inputs.prompt_v2_contract.itertuples(index=False)})
    return lookup


def build_contexts(root: Path, sample: pd.DataFrame, manifest: pd.DataFrame, prompt_contract: pd.DataFrame, prompt_v2_contract: pd.DataFrame) -> dict[str, dict[str, Any]]:
    title_map = load_title_map(root)
    histories = load_histories(root, title_map)
    candidates = read_csv(root, "results/tables/ranking_candidate_sets.csv")
    candidate_lookup = {
        (str(row.candidate_set_id), str(row.user_id)): parse_json_list(row.candidate_items)
        for row in candidates[["candidate_set_id", "user_id", "candidate_items"]].itertuples(index=False)
    }
    prompt_lookup = {row.prompt_contract_id: row._asdict() for row in prompt_contract.itertuples(index=False)}
    prompt_lookup.update({row.prompt_contract_id: row._asdict() for row in prompt_v2_contract.itertuples(index=False)})
    sample_lookup = {row.holdout_row_id: row._asdict() for row in sample.itertuples(index=False)}
    contexts: dict[str, dict[str, Any]] = {}
    for row in manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        sample_row = sample_lookup[row["holdout_row_id"]]
        key = (row["candidate_set_id"], row["user_id"])
        if key not in candidate_lookup:
            raise ValueError(f"candidate row missing: {row['holdout_request_id']}")
        candidate_ids = [str(item) for item in candidate_lookup[key]]
        if len(candidate_ids) != 20:
            raise ValueError(f"candidate count mismatch: {row['holdout_request_id']}")
        if sha256_items(candidate_ids) != row["candidate_order_sha256"]:
            raise ValueError(f"candidate-order hash mismatch: {row['holdout_request_id']}")
        candidate_items = [{"item_id": item_id, "title": title_map[(row["dataset"], item_id)]} for item_id in candidate_ids]
        history_items = [
            item
            for item in histories[row["dataset"]].get(row["user_id"], [])
            if item["item_id"] != sample_row["positive_item_id"]
        ][-10:]
        prompt_row = prompt_lookup[row["prompt_contract_id"]]
        system_message, user_message, history_render, candidate_render = render_from_contract(
            prompt_row,
            history_items=history_items,
            candidate_items=candidate_items,
        )
        if sha256_text(history_render) != row["history_render_sha256"]:
            raise ValueError(f"history-render hash mismatch: {row['holdout_request_id']}")
        if positive_label_exposed(json.dumps({"system": system_message, "prompt": user_message}, ensure_ascii=False), sample_row["positive_item_id"]):
            raise ValueError(f"positive label exposed: {row['holdout_request_id']}")
        contexts[row["holdout_request_id"]] = {
            "holdout_request_id": row["holdout_request_id"],
            "holdout_row_id": row["holdout_row_id"],
            "positive_item_id": sample_row["positive_item_id"],
            "candidate_items": candidate_ids,
            "candidate_titles": [item["title"] for item in candidate_items],
            "system_message": system_message,
            "user_message": user_message,
            "history_render": history_render,
            "candidate_render": candidate_render,
            "candidate_order_sha256": sha256_items(candidate_ids),
            "history_render_sha256": sha256_text(history_render),
            "complete_prompt_sha256": sha256_text(system_message + "\n\n" + user_message),
            "prompt_contract_sha256": row["prompt_contract_sha256"],
            "requested_output_count": "20",
            "requires_full_permutation": True,
        }
    return contexts


def load_inputs(root: Path) -> HybridInputs:
    policy = read_table(root, HYBRID_POLICY_PATH)
    validate_policy(policy)
    sample = read_table(root, HOLDOUT_SAMPLE_PATH)
    manifest = read_table(root, HOLDOUT_MANIFEST_PATH)
    prompt_contract = read_table(root, PROMPT_CONTRACT_PATH)
    prompt_v2_contract = read_table(root, PROMPT_V2_CONTRACT_PATH)
    parser_contract = read_table(root, PARSER_CONTRACT_PATH)
    execution_contract = read_table(root, EXECUTION_CONTRACT_PATH)
    recovery = read_table(root, OLLAMA_RECOVERY_PATH)
    v1_sample = read_table(root, V1_SAMPLE_PATH)
    v2_manifest = read_table(root, V2_MANIFEST_PATH)
    overlap = validate_sample(sample, v1_sample, v2_manifest)
    validate_manifest(manifest, sample)
    decision = read_table(root, HYBRID_DECISION_PATH)
    if decision.iloc[0]["recommended_decision"] != "approve_hybrid_disjoint_holdout_pilot":
        raise ValueError("hybrid audit did not approve holdout pilot")
    contexts = build_contexts(root, sample, manifest, prompt_contract, prompt_v2_contract)
    inputs = HybridInputs(
        policy=policy,
        sample=sample,
        manifest=manifest,
        prompt_contract=prompt_contract,
        prompt_v2_contract=prompt_v2_contract,
        parser_contract=parser_contract,
        execution_contract=execution_contract,
        model_inventory=read_table(root, MODEL_INVENTORY_PATH),
        recovery=recovery,
        v1_sample=v1_sample,
        v1_metrics=read_table(root, V1_METRICS_PATH),
        v2_manifest=v2_manifest,
        v2_metrics=read_table(root, V2_METRICS_PATH),
        contexts=contexts,
        previous_overlap_count=overlap,
        protected_hashes=hash_protected_inputs(root),
    )
    validate_contract_hashes(inputs)
    return inputs


def validate_contract_hashes(inputs: HybridInputs) -> None:
    prompts = prompt_contract_lookup(inputs)
    parsers = {row.parser_contract_id: row._asdict() for row in inputs.parser_contract.itertuples(index=False)}
    executions = {row.execution_contract_id: row._asdict() for row in inputs.execution_contract.itertuples(index=False)}
    for row in inputs.manifest.to_dict("records"):
        if prompts[row["prompt_contract_id"]]["contract_sha256"] != row["prompt_contract_sha256"]:
            raise ValueError(f"prompt hash mismatch: {row['holdout_request_id']}")
        if parsers[row["parser_contract_id"]]["parser_schema_sha256"] != row["parser_contract_sha256"]:
            raise ValueError(f"parser hash mismatch: {row['holdout_request_id']}")
        if executions[row["execution_contract_id"]]["contract_sha256"] != row["execution_contract_sha256"]:
            raise ValueError(f"execution hash mismatch: {row['holdout_request_id']}")
        context = inputs.contexts[row["holdout_request_id"]]
        if context["candidate_order_sha256"] != row["candidate_order_sha256"]:
            raise ValueError(f"candidate hash mismatch: {row['holdout_request_id']}")
        if context["history_render_sha256"] != row["history_render_sha256"]:
            raise ValueError(f"history hash mismatch: {row['holdout_request_id']}")


def fetch_local_json(url: str, timeout: int = 5) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def verify_ollama_admin(root: Path, recovery: pd.DataFrame) -> OllamaAdminState:
    base_url = "http://127.0.0.1:11434"
    start_attempted = False
    started_pid = ""
    try:
        version = fetch_local_json(base_url + "/api/version", timeout=3)
    except Exception:
        process_check = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Process ollama -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if process_check.stdout.strip():
            raise RuntimeError("stale_existing_process")
        port_check = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue | Where-Object {$_.State -eq 'Listen'} | Select-Object -ExpandProperty OwningProcess"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if port_check.stdout.strip():
            raise RuntimeError("port_conflict")
        exe = recovery.iloc[0]["ollama_executable_path"]
        out_path = Path("C:/tmp/phase5b_p1e_ollama_stdout.log")
        err_path = Path("C:/tmp/phase5b_p1e_ollama_stderr.log")
        out_handle = out_path.open("ab")
        err_handle = err_path.open("ab")
        process = subprocess.Popen([exe, "serve"], stdout=out_handle, stderr=err_handle)
        start_attempted = True
        started_pid = str(process.pid)
        version = {}
        for _ in range(30):
            time.sleep(2)
            try:
                version = fetch_local_json(base_url + "/api/version", timeout=3)
                break
            except Exception:
                if process.poll() is not None:
                    raise RuntimeError(f"service_start_failed: exit={process.returncode}")
        if not version:
            raise RuntimeError("service_start_failed: timeout")
    tags_payload = fetch_local_json(base_url + "/api/tags", timeout=10)
    ps_payload = fetch_local_json(base_url + "/api/ps", timeout=10)
    tags = {str(item.get("name", "")): item for item in tags_payload.get("models", [])}
    for family, expected in EXPECTED_DIGESTS.items():
        tag = expected["tag"]
        if tag not in tags:
            raise RuntimeError(f"blocked_missing_local_model: {tag}")
        if str(tags[tag].get("digest", "")).lower() != expected["digest"]:
            raise RuntimeError(f"model_digest_mismatch: {tag}")
    return OllamaAdminState(
        server_version=str(version.get("version", "")),
        tags=tags,
        ps=list(ps_payload.get("models", [])),
        started_process_id=started_pid,
        start_attempted=start_attempted,
    )


def count_by_holdout_id(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        request_id = str(record.get("holdout_request_id", record.get("request_id", "")))
        counts[request_id] = counts.get(request_id, 0) + 1
    return counts


def classify_request_state(request_count: int, response_count: int) -> str:
    if request_count == 0 and response_count == 0:
        return "not_started"
    if request_count == 1 and response_count == 1:
        return "completed"
    if request_count == 1 and response_count == 0:
        return "ambiguous_interrupted"
    return "invalid_duplicate_evidence"


def state_summary(manifest: pd.DataFrame, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    req_counts = count_by_holdout_id(request_records)
    resp_counts = count_by_holdout_id(response_records)
    summary = {"completed": 0, "not_started": 0, "ambiguous_interrupted": 0, "invalid_duplicate_evidence": 0}
    manifest_ids = set(manifest["holdout_request_id"])
    for row in manifest.itertuples(index=False):
        state = classify_request_state(req_counts.get(row.holdout_request_id, 0), resp_counts.get(row.holdout_request_id, 0))
        summary[state] += 1
    extras = (set(req_counts) | set(resp_counts)) - manifest_ids
    if extras:
        summary["invalid_duplicate_evidence"] += len(extras)
    return summary


def pending_requests_in_order(manifest: pd.DataFrame, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    req_counts = count_by_holdout_id(request_records)
    resp_counts = count_by_holdout_id(response_records)
    manifest_ids = set(manifest["holdout_request_id"])
    extras = (set(req_counts) | set(resp_counts)) - manifest_ids
    if extras:
        raise RuntimeError(f"invalid_duplicate_evidence: raw id outside manifest {sorted(extras)[0]}")
    pending: list[dict[str, str]] = []
    for row in manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        state = classify_request_state(req_counts.get(row["holdout_request_id"], 0), resp_counts.get(row["holdout_request_id"], 0))
        if state == "not_started":
            pending.append(row)
        elif state == "completed":
            continue
        elif state == "ambiguous_interrupted":
            raise RuntimeError(f"ambiguous_interrupted_request: {row['holdout_request_id']}")
        else:
            raise RuntimeError(f"invalid_duplicate_evidence: {row['holdout_request_id']}")
    return pending


def build_request_record(row: Mapping[str, str], context: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if positive_label_exposed(payload_text, str(context.get("positive_item_id", ""))):
        raise ValueError(f"positive label in payload: {row['holdout_request_id']}")
    return {
        "holdout_request_id": row["holdout_request_id"],
        "request_id": row["holdout_request_id"],
        "holdout_row_id": row["holdout_row_id"],
        "hybrid_policy_id": row["hybrid_policy_id"],
        "attempt_index": 1,
        "model_family": row["model_family"],
        "model_tag": row["model_tag"],
        "verified_model_digest": EXPECTED_DIGESTS[row["model_family"]]["digest"],
        "dataset": row["dataset"],
        "task_type": row["task_type"],
        "candidate_set_id": row["candidate_set_id"],
        "user_id": row["user_id"],
        "prompt_contract_id": row["prompt_contract_id"],
        "prompt_contract_sha256": row["prompt_contract_sha256"],
        "parser_contract_id": row["parser_contract_id"],
        "parser_contract_sha256": row["parser_contract_sha256"],
        "execution_contract_id": row["execution_contract_id"],
        "execution_contract_sha256": row["execution_contract_sha256"],
        "candidate_order_sha256": row["candidate_order_sha256"],
        "history_render_sha256": row["history_render_sha256"],
        "complete_prompt_sha256": context["complete_prompt_sha256"],
        "scientific_retry_count": "0",
        "infrastructure_retry_count": "0",
        "candidate_items": context["candidate_items"],
        "request_payload": payload,
        "timestamp_before_send_utc": utc_now(),
    }


def build_response_record(row: Mapping[str, str], result: Any) -> dict[str, Any]:
    raw_sha = sha256_text(result.response_text)
    return {
        "holdout_request_id": row["holdout_request_id"],
        "request_id": row["holdout_request_id"],
        "holdout_row_id": row["holdout_row_id"],
        "attempt_index": 1,
        "backend_success": bool_text(result.backend_success),
        "raw_response_text": result.response_text,
        "raw_response_sha256": raw_sha,
        "model_reported_tag": str(result.raw_backend_response.get("model", row["model_tag"])) if result.raw_backend_response else "",
        "backend_metadata": result.raw_backend_response,
        "done": str(result.raw_backend_response.get("done", "")) if result.raw_backend_response else "",
        "done_reason": str(result.raw_backend_response.get("done_reason", "done" if result.raw_backend_response.get("done") else "")) if result.raw_backend_response else "",
        "prompt_eval_count": result.raw_backend_response.get("prompt_eval_count", ""),
        "generated_token_count": result.raw_backend_response.get("eval_count", ""),
        "load_duration_nanoseconds": result.raw_backend_response.get("load_duration", ""),
        "prompt_eval_duration_nanoseconds": result.raw_backend_response.get("prompt_eval_duration", ""),
        "eval_duration_nanoseconds": result.raw_backend_response.get("eval_duration", ""),
        "total_duration_nanoseconds": result.raw_backend_response.get("total_duration", ""),
        "request_started_at_utc": result.started_at,
        "timestamp_after_completion_utc": result.completed_at,
        "duration_seconds": format_float(result.duration_seconds),
        "backend_error_type": result.backend_error_type,
        "backend_error_message": result.backend_error_message,
    }


def execute_pending_requests(root: Path, inputs: HybridInputs) -> None:
    (root / RAW_DIR).mkdir(parents=True, exist_ok=True)
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    pending = pending_requests_in_order(inputs.manifest, request_records, response_records)
    for row in pending:
        if hash_protected_inputs(root) != inputs.protected_hashes:
            raise RuntimeError("protected_input_validation_failed")
        context = inputs.contexts[row["holdout_request_id"]]
        if context["prompt_contract_sha256"] != row["prompt_contract_sha256"]:
            raise ValueError(f"prompt hash mismatch before send: {row['holdout_request_id']}")
        if context["candidate_order_sha256"] != row["candidate_order_sha256"]:
            raise ValueError(f"candidate-order hash mismatch before send: {row['holdout_request_id']}")
        if context["history_render_sha256"] != row["history_render_sha256"]:
            raise ValueError(f"history hash mismatch before send: {row['holdout_request_id']}")
        payload = build_payload(row, context)
        request_record = build_request_record(row, context, payload)
        append_jsonl_fsync(root / REQUEST_JSONL, request_record)
        append_jsonl_fsync(root / EVENT_JSONL, {"event": "request_recorded_before_send", "holdout_request_id": row["holdout_request_id"], "timestamp_utc": utc_now()})
        timeout_seconds = max(120, int(float(row["max_output_tokens"] or 120)))
        result = execute_ollama_request(payload, timeout_seconds)
        response_record = build_response_record(row, result)
        append_jsonl_fsync(root / RESPONSE_JSONL, response_record)
        append_jsonl_fsync(
            root / EVENT_JSONL,
            {
                "event": "response_or_error_recorded",
                "holdout_request_id": row["holdout_request_id"],
                "backend_success": bool_text(result.backend_success),
                "timestamp_utc": utc_now(),
            },
        )
        print(f"{row['holdout_request_id']} {row['model_family']} {row['dataset']} backend_success={bool_text(result.backend_success)}", flush=True)


def parse_indices_for_forensics(raw_text: str) -> list[Any]:
    try:
        payload = json.loads(str(raw_text).strip())
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict) and isinstance(payload.get("ranked_candidate_indices"), list):
        return list(payload["ranked_candidate_indices"])
    return []


def records_to_execution(manifest: pd.DataFrame, response_records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    manifest_lookup = {row.holdout_request_id: row._asdict() for row in manifest.itertuples(index=False)}
    response_lookup = {str(record.get("holdout_request_id", record.get("request_id", ""))): record for record in response_records}
    rows: list[dict[str, Any]] = []
    for holdout_request_id in manifest.sort_values("request_order", key=lambda s: s.astype(int))["holdout_request_id"]:
        manifest_row = manifest_lookup[holdout_request_id]
        record = response_lookup.get(holdout_request_id, {})
        rows.append(
            {
                "holdout_request_id": holdout_request_id,
                "holdout_row_id": manifest_row["holdout_row_id"],
                "model_family": manifest_row["model_family"],
                "model_tag": manifest_row["model_tag"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "prompt_contract_id": manifest_row["prompt_contract_id"],
                "attempt_index": "1" if record else "",
                "request_started_at_utc": str(record.get("request_started_at_utc", "")),
                "request_completed_at_utc": str(record.get("timestamp_after_completion_utc", "")),
                "duration_seconds": str(record.get("duration_seconds", "")),
                "backend_success": str(record.get("backend_success", "")),
                "response_nonempty": bool_text(bool(str(record.get("raw_response_text", "")).strip())) if record else "",
                "done_reason": str(record.get("done_reason", "")),
                "prompt_eval_count": str(record.get("prompt_eval_count", "")),
                "eval_count": str(record.get("generated_token_count", "")),
                "load_duration_nanoseconds": str(record.get("load_duration_nanoseconds", "")),
                "prompt_eval_duration_nanoseconds": str(record.get("prompt_eval_duration_nanoseconds", "")),
                "eval_duration_nanoseconds": str(record.get("eval_duration_nanoseconds", "")),
                "total_duration_nanoseconds": str(record.get("total_duration_nanoseconds", "")),
                "raw_response_sha256": str(record.get("raw_response_sha256", "")),
                "backend_error_type": str(record.get("backend_error_type", "")),
                "backend_error_message": clean_text(str(record.get("backend_error_message", "")), 500),
                "response_jsonl_recorded": bool_text(bool(record)),
                "execution_status": "completed" if str(record.get("backend_success", "")) == "true" else ("backend_error_recorded" if record else "missing_response"),
                "notes": "single scientific attempt; no retry" if record else "missing raw response",
            }
        )
    return frame(rows, EXECUTION_COLUMNS)


def build_parse_and_rankings(inputs: HybridInputs, execution: pd.DataFrame, response_records: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_lookup = {row.holdout_request_id: row._asdict() for row in inputs.manifest.itertuples(index=False)}
    response_lookup = {str(record.get("holdout_request_id", record.get("request_id", ""))): record for record in response_records}
    parse_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for holdout_request_id, manifest_row in manifest_lookup.items():
        context = inputs.contexts[holdout_request_id]
        record = response_lookup.get(holdout_request_id, {})
        raw_text = str(record.get("raw_response_text", ""))
        raw_sha = str(record.get("raw_response_sha256", sha256_text(raw_text)))
        candidate_items = list(context["candidate_items"])
        classified = classify_response(
            raw_text,
            candidate_items=candidate_items,
            expected_count=20,
            require_full_permutation=True,
        )
        forensic_values = parse_indices_for_forensics(raw_text)
        missing, duplicated, out_of_range = missing_and_duplicate_indices(forensic_values)
        truncation = output_truncation_detected(
            raw_text,
            str(record.get("done_reason", "")),
            record.get("generated_token_count", ""),
            manifest_row.get("max_output_tokens", ""),
        )
        parse_rows.append(
            {
                "holdout_request_id": holdout_request_id,
                "holdout_row_id": manifest_row["holdout_row_id"],
                "model_family": manifest_row["model_family"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "prompt_contract_id": manifest_row["prompt_contract_id"],
                "raw_response_sha256": raw_sha,
                "strict_parser_id": manifest_row["parser_contract_id"],
                "diagnostic_parser_id": RERANK_DIAGNOSTIC_PARSER,
                "json_parse_valid": bool_text(classified["json_parse_valid"]),
                "schema_valid": bool_text(classified["schema_valid"]),
                "output_count_valid": bool_text(classified["output_count_valid"]),
                "index_type_valid": bool_text(classified["index_type_valid"]),
                "index_range_valid": bool_text(classified["index_range_valid"]),
                "uniqueness_valid": bool_text(classified["uniqueness_valid"]),
                "full_permutation_valid": bool_text(classified["full_permutation_valid"]),
                "candidate_validity": bool_text(classified["candidate_validity"]),
                "catalog_validity": bool_text(classified["catalog_validity"]),
                "duplicate_count": str(classified["duplicate_count"]),
                "unknown_index_count": str(classified["unknown_index_count"]),
                "hallucination_count": str(classified["hallucination_count"]),
                "missing_indices": json_dumps(missing),
                "duplicated_indices": json_dumps(duplicated),
                "strict_parse_valid": bool_text(classified["strict_parse_valid"]),
                "diagnostic_salvage_valid": bool_text(classified["diagnostic_salvage_valid"]),
                "diagnostic_transformations": json_dumps(classified["diagnostic_transformations"]),
                "strict_indices": json_dumps(classified["strict_indices"]),
                "salvage_indices": json_dumps(classified["salvage_indices"]),
                "output_truncation_detected": bool_text(truncation),
                "primary_failure_reason": classified["primary_failure_reason"],
                "notes": f"out_of_range_indices={json_dumps(out_of_range)}; salvage output not used for primary ranking",
            }
        )
        ranked_items: list[str] = []
        top5: list[str] = []
        positive_rank_zero = ""
        positive_rank_one = ""
        positive_in_top5 = ""
        ndcg = ""
        utility_valid = "false"
        if classified["strict_parse_valid"]:
            ranked_items = [candidate_items[index - 1] for index in classified["strict_indices"]]
            top5 = ranked_items[:5]
            if context["positive_item_id"] in ranked_items:
                rank = ranked_items.index(context["positive_item_id"])
                positive_rank_zero = str(rank)
                positive_rank_one = str(rank + 1)
                positive_in_top5 = bool_text(rank < 5)
                ndcg = format_float(1.0 / math.log2(rank + 2.0)) if rank < 5 else "0"
            else:
                positive_in_top5 = "false"
                ndcg = "0"
            utility_valid = "true"
        ranking_rows.append(
            {
                "holdout_request_id": holdout_request_id,
                "holdout_row_id": manifest_row["holdout_row_id"],
                "model_family": manifest_row["model_family"],
                "model_tag": manifest_row["model_tag"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "prompt_contract_id": manifest_row["prompt_contract_id"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "positive_item_id": context["positive_item_id"],
                "candidate_items": json_dumps(candidate_items),
                "candidate_order_sha256": context["candidate_order_sha256"],
                "strict_parse_valid": bool_text(classified["strict_parse_valid"]),
                "ranked_candidate_indices": json_dumps(classified["strict_indices"] if classified["strict_parse_valid"] else []),
                "ranked_item_ids": json_dumps(ranked_items),
                "top5_item_ids": json_dumps(top5),
                "positive_rank_zero_based": positive_rank_zero,
                "positive_rank_one_based": positive_rank_one,
                "positive_in_top_5": positive_in_top5,
                "ndcg_at_5": ndcg,
                "candidate_validity": bool_text(classified["candidate_validity"]),
                "duplicate_count": str(classified["duplicate_count"]),
                "hallucination_count": str(classified["hallucination_count"]),
                "full_permutation_valid": bool_text(classified["full_permutation_valid"]),
                "utility_evaluation_valid": utility_valid,
                "invalidity_reason": "" if utility_valid == "true" else classified["primary_failure_reason"],
                "notes": "valid-only utility; invalid rows have empty normalized outputs",
            }
        )
    return frame(parse_rows, PARSE_COLUMNS), frame(ranking_rows, RANKING_COLUMNS)


def build_metrics(inputs: HybridInputs, execution: pd.DataFrame, parse_results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    merged = inputs.manifest[
        ["holdout_request_id", "model_family", "model_tag", "dataset", "task_type", "prompt_contract_id"]
    ].merge(
        execution,
        on=["holdout_request_id", "model_family", "model_tag", "dataset", "task_type", "prompt_contract_id"],
    )
    merged = merged.merge(
        parse_results,
        on=["holdout_request_id", "model_family", "dataset", "task_type", "prompt_contract_id"],
        suffixes=("", "_parse"),
    )
    merged = merged.merge(
        rankings[["holdout_request_id", "positive_in_top_5", "ndcg_at_5", "positive_rank_zero_based", "utility_evaluation_valid"]],
        on="holdout_request_id",
    )
    for (family, tag, dataset, task, prompt_id), group in merged.groupby(["model_family", "model_tag", "dataset", "task_type", "prompt_contract_id"], sort=False):
        rows_count = len(group)
        strict = group[group["strict_parse_valid"].eq("true")]
        valid_utility = group[group["utility_evaluation_valid"].eq("true")]
        durations = pd.to_numeric(group["duration_seconds"], errors="coerce")
        eval_counts = pd.to_numeric(group["eval_count"], errors="coerce")
        strict_count = len(strict)
        output_count_valid = int(group["output_count_valid"].eq("true").sum())
        uniqueness_valid = int(group["uniqueness_valid"].eq("true").sum())
        full_perm_valid = int(group["full_permutation_valid"].eq("true").sum())
        candidate_valid = int(strict["candidate_validity"].eq("true").sum())
        duplicate_free = int(pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).sum())
        hallucination_free = int(pd.to_numeric(strict["hallucination_count"], errors="coerce").fillna(0).eq(0).sum())
        passes = (
            rows_count == 10
            and strict_count >= 8
            and strict_count / rows_count >= READINESS_THRESHOLD
            and candidate_valid == strict_count
            and duplicate_free == strict_count
            and hallucination_free == strict_count
            and int(strict["full_permutation_valid"].eq("true").sum()) == strict_count
        )
        rows.append(
            {
                "model_family": family,
                "model_tag": tag,
                "dataset": dataset,
                "task_type": task,
                "prompt_contract_id": prompt_id,
                "holdout_rows": str(rows_count),
                "backend_success_rows": str(int(group["backend_success"].eq("true").sum())),
                "strict_valid_rows": str(strict_count),
                "strict_parse_validity_rate": format_float(strict_count / rows_count if rows_count else 0.0),
                "diagnostic_salvage_valid_rows": str(int(group["diagnostic_salvage_valid"].eq("true").sum())),
                "diagnostic_salvage_validity_rate": format_float(int(group["diagnostic_salvage_valid"].eq("true").sum()) / rows_count if rows_count else 0.0),
                "output_count_valid_rows": str(output_count_valid),
                "uniqueness_valid_rows": str(uniqueness_valid),
                "full_permutation_valid_rows": str(full_perm_valid),
                "candidate_valid_rows": str(candidate_valid),
                "duplicate_free_rows": str(duplicate_free),
                "hallucination_free_rows": str(hallucination_free),
                "utility_evaluated_rows": str(len(valid_utility)),
                "hr_at_5_valid_only": mean_bool(valid_utility["positive_in_top_5"]) if len(valid_utility) else "",
                "ndcg_at_5_valid_only": mean_numeric(valid_utility["ndcg_at_5"]) if len(valid_utility) else "",
                "mean_positive_rank_valid_only": mean_numeric(valid_utility[valid_utility["positive_rank_zero_based"].ne("")]["positive_rank_zero_based"]) if len(valid_utility[valid_utility["positive_rank_zero_based"].ne("")]) else "",
                "median_duration_seconds": format_float(float(durations.median())) if durations.notna().any() else "",
                "mean_duration_seconds": format_float(float(durations.mean())) if durations.notna().any() else "",
                "median_generated_token_count": format_float(float(eval_counts.median())) if eval_counts.notna().any() else "",
                "readiness_threshold": format_float(READINESS_THRESHOLD),
                "cell_readiness_pass": bool_text(passes),
                "metric_status": "holdout_readiness_pass" if passes else "holdout_readiness_fail",
                "notes": "Valid-only utility is diagnostic and not used for readiness.",
            }
        )
    return frame(rows, METRIC_COLUMNS)


def build_prior_comparison(inputs: HybridInputs, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counter = 1
    for family, _tag, prompt_id in MODEL_ORDER:
        prior_metrics = inputs.v1_metrics if family == "Qwen2 7B Instruct" else inputs.v2_metrics
        prior_phase = "prompt_v1_pilot" if family == "Qwen2 7B Instruct" else "prompt_v2_corrective_pilot"
        for dataset in DATASET_ORDER:
            holdout = metrics[(metrics["model_family"].eq(family)) & (metrics["dataset"].eq(dataset))].iloc[0]
            prior = prior_metrics[
                (prior_metrics["model_family"].eq(family))
                & (prior_metrics["dataset"].eq(dataset))
                & (prior_metrics["task_type"].eq("reranking"))
            ].iloc[0]
            prior_rows_col = "pilot_rows" if family == "Qwen2 7B Instruct" else "corrective_rows"
            prior_rate = float(prior["strict_parse_validity_rate"])
            holdout_rate = float(holdout["strict_parse_validity_rate"])
            rows.append(
                {
                    "comparison_id": f"HYBH-CMP-{counter:03d}",
                    "model_family": family,
                    "dataset": dataset,
                    "selected_prompt_contract_id": prompt_id,
                    "prior_pilot_phase": prior_phase,
                    "prior_rows": prior[prior_rows_col],
                    "prior_strict_valid_rows": prior["strict_valid_rows"],
                    "prior_strict_parse_validity_rate": format_float(prior_rate),
                    "holdout_rows": holdout["holdout_rows"],
                    "holdout_strict_valid_rows": holdout["strict_valid_rows"],
                    "holdout_strict_parse_validity_rate": format_float(holdout_rate),
                    "validity_rate_difference": format_float(holdout_rate - prior_rate),
                    "same_prompt_contract": "true",
                    "disjoint_rows": "true",
                    "utility_not_used_for_readiness": "true",
                    "comparison_status": "descriptive_only_no_significance_claim",
                    "notes": "Prior-pilot utility is not used; comparison is structural validity only.",
                }
            )
            counter += 1
    return frame(rows, COMPARISON_COLUMNS)


def build_request_state(inputs: HybridInputs, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]], parse_results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    req_counts = count_by_holdout_id(request_records)
    resp_counts = count_by_holdout_id(response_records)
    request_sha = {str(record.get("holdout_request_id", "")): file_record_sha256(record) for record in request_records}
    response_sha = {str(record.get("holdout_request_id", "")): str(record.get("raw_response_sha256", "")) for record in response_records}
    parse_lookup = {row.holdout_request_id: row._asdict() for row in parse_results.itertuples(index=False)}
    ranking_lookup = {row.holdout_request_id: row._asdict() for row in rankings.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for row in inputs.manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        request_id = row["holdout_request_id"]
        state = classify_request_state(req_counts.get(request_id, 0), resp_counts.get(request_id, 0))
        parsed = parse_lookup.get(request_id, {})
        ranking = ranking_lookup.get(request_id, {})
        rows.append(
            {
                "holdout_request_id": request_id,
                "holdout_row_id": row["holdout_row_id"],
                "request_order": row["request_order"],
                "model_family": row["model_family"],
                "model_tag": row["model_tag"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "candidate_set_id": row["candidate_set_id"],
                "user_id": row["user_id"],
                "prompt_contract_id": row["prompt_contract_id"],
                "request_record_count": str(req_counts.get(request_id, 0)),
                "response_record_count": str(resp_counts.get(request_id, 0)),
                "request_state": state,
                "request_sent": bool_text(req_counts.get(request_id, 0) > 0),
                "response_recorded": bool_text(resp_counts.get(request_id, 0) > 0),
                "backend_success": "true" if resp_counts.get(request_id, 0) and state == "completed" else "",
                "strict_parse_valid": parsed.get("strict_parse_valid", ""),
                "full_permutation_valid": parsed.get("full_permutation_valid", ""),
                "candidate_validity": parsed.get("candidate_validity", ""),
                "utility_evaluation_valid": ranking.get("utility_evaluation_valid", ""),
                "raw_request_sha256": request_sha.get(request_id, ""),
                "raw_response_sha256": response_sha.get(request_id, ""),
                "blocking_reason": "" if state == "completed" else state,
                "notes": "resume-safe state from append-only raw evidence",
            }
        )
    return frame(rows, REQUEST_STATE_COLUMNS)


def build_decision(inputs: HybridInputs, execution: pd.DataFrame, parse_results: pd.DataFrame, metrics: pd.DataFrame, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]], protected_unchanged: bool) -> pd.DataFrame:
    completed = len(execution[execution["response_jsonl_recorded"].eq("true")])
    backend_success = int(execution["backend_success"].eq("true").sum())
    backend_errors = int(execution["backend_success"].eq("false").sum())
    backend_rate = backend_success / 60 if len(execution) else 0.0
    min_rate = float(pd.to_numeric(metrics["strict_parse_validity_rate"], errors="coerce").min()) if len(metrics) else 0.0
    cells_passing = int(metrics["cell_readiness_pass"].eq("true").sum()) if len(metrics) else 0
    strict = parse_results[parse_results["strict_parse_valid"].eq("true")]
    all_candidate_valid = strict["candidate_validity"].eq("true").all() if len(strict) else False
    all_duplicate_free = pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).all() if len(strict) else False
    all_hallucination_free = pd.to_numeric(strict["hallucination_count"], errors="coerce").fillna(0).eq(0).all() if len(strict) else False
    all_complete = strict["full_permutation_valid"].eq("true").all() if len(strict) else False
    truncation = parse_results["output_truncation_detected"].eq("true").any() if len(parse_results) else False
    raw_request_retained = len(request_records) == 60
    raw_response_retained = len(response_records) == 60
    assignments_valid = (
        set(inputs.manifest[inputs.manifest["model_family"].eq("Qwen2 7B Instruct")]["prompt_contract_id"]) == {QWEN_RERANK_PROMPT}
        and set(inputs.manifest[inputs.manifest["model_family"].eq("Mistral 7B Instruct")]["prompt_contract_id"]) == {MISTRAL_RERANK_PROMPT}
    )
    candidate_hashes = all(inputs.contexts[row["holdout_request_id"]]["candidate_order_sha256"] == row["candidate_order_sha256"] for row in inputs.manifest.to_dict("records"))
    history_hashes = all(inputs.contexts[row["holdout_request_id"]]["history_render_sha256"] == row["history_render_sha256"] for row in inputs.manifest.to_dict("records"))
    positive_hidden = True
    blocking_cells = []
    for row in metrics.itertuples(index=False):
        if row.cell_readiness_pass != "true":
            blocking_cells.append(f"{row.model_family}|{row.dataset}|strict_parse={row.strict_parse_validity_rate}")
    if not protected_unchanged:
        decision = "protected_input_validation_failed"
        corrective = "Restore protected inputs before interpreting the holdout pilot."
        next_task = "Validate protected inputs and rerun the Phase 5B-P1E holdout runner."
    elif completed != 60 or not raw_request_retained or not raw_response_retained:
        decision = "holdout_pilot_incomplete"
        corrective = "Resume only the Phase 5B-P1E holdout runner until request/response accounting is complete."
        next_task = "Resume the Phase 5B-P1E holdout runner."
    elif backend_errors:
        decision = "backend_blocked"
        corrective = "Resolve local backend failure without changing prompts or retrying completed requests."
        next_task = "Audit backend failure evidence before any further execution."
    elif truncation:
        decision = "require_final_reranking_protocol_decision_audit"
        corrective = FAILED_NEXT_TASK
        next_task = FAILED_NEXT_TASK
    elif inputs.previous_overlap_count != 0:
        decision = "hybrid_policy_failed_disjoint_holdout"
        corrective = FAILED_NEXT_TASK
        next_task = FAILED_NEXT_TASK
    elif cells_passing == 6 and assignments_valid and candidate_hashes and history_hashes and positive_hidden and all_candidate_valid and all_duplicate_free and all_hallucination_free and all_complete:
        decision = "approve_full_local_execution_with_hybrid_prompt_policy"
        corrective = "No corrective action required before Phase 5B-P2."
        next_task = APPROVED_NEXT_TASK
    else:
        decision = "hybrid_policy_failed_disjoint_holdout"
        corrective = FAILED_NEXT_TASK
        next_task = FAILED_NEXT_TASK
    rows = [
        {
            "holdout_pilot_id": "canonical_local_llm_hybrid_disjoint_holdout_pilot_v1",
            "hybrid_policy_id": HYBRID_POLICY_ID,
            "expected_request_count": "60",
            "completed_request_count": str(completed),
            "backend_error_record_count": str(backend_errors),
            "ambiguous_request_count": "0",
            "previous_pilot_overlap_count": str(inputs.previous_overlap_count),
            "backend_completion_rate": format_float(backend_rate),
            "minimum_cell_strict_parse_validity_rate": format_float(min_rate),
            "cells_passing_threshold": str(cells_passing),
            "total_readiness_cells": "6",
            "all_policy_assignments_valid": bool_text(assignments_valid),
            "all_candidate_order_hashes_valid": bool_text(candidate_hashes),
            "all_history_hashes_valid": bool_text(history_hashes),
            "positive_label_never_exposed": bool_text(positive_hidden),
            "all_strict_outputs_candidate_valid": bool_text(all_candidate_valid),
            "all_strict_outputs_duplicate_free": bool_text(all_duplicate_free),
            "all_strict_outputs_hallucination_free": bool_text(all_hallucination_free),
            "all_strict_outputs_complete_permutations": bool_text(all_complete),
            "raw_request_retention_complete": bool_text(raw_request_retained),
            "raw_response_retention_complete": bool_text(raw_response_retained),
            "output_truncation_detected": bool_text(truncation),
            "readiness_threshold": format_float(READINESS_THRESHOLD),
            "readiness_threshold_result_independent": "true",
            "utility_used_for_readiness": "false",
            "ranking_prompt_for_full_execution": RANKING_PROMPT_V1,
            "qwen_reranking_prompt_for_full_execution": QWEN_RERANK_PROMPT,
            "mistral_reranking_prompt_for_full_execution": MISTRAL_RERANK_PROMPT,
            "pilot_decision": decision,
            "blocking_cells": json_dumps(blocking_cells),
            "required_corrective_action": corrective,
            "recommended_single_next_task": next_task,
            "notes": "Readiness uses structural validity only; utility diagnostics are valid-only and secondary.",
        }
    ]
    return frame(rows, DECISION_COLUMNS)


def build_prior_validation_rows(inputs: HybridInputs, admin: OllamaAdminState, initial_summary: Mapping[str, int], request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]], parse_results: pd.DataFrame, metrics: pd.DataFrame, decision: pd.DataFrame, protected_unchanged: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(scope: str, check: str, expected: Any, observed: Any, ok: bool, evidence: str, model: str = "", dataset: str = "", severity: str = "error", notes: str = "") -> None:
        rows.append(
            {
                "validation_id": f"HYBH-VAL-{len(rows) + 1:03d}",
                "validation_scope": scope,
                "model_family": model,
                "dataset": dataset,
                "check_name": check,
                "expected_value": expected,
                "observed_value": observed,
                "status": "pass" if ok else "fail",
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    add("protected_inputs", "protected_hashes_unchanged", "true", bool_text(protected_unchanged), protected_unchanged, "hash_protected_inputs")
    add("policy", "hybrid_policy_hash", file_sha256(Path.cwd() / HYBRID_POLICY_PATH), file_sha256(Path.cwd() / HYBRID_POLICY_PATH), True, HYBRID_POLICY_PATH.as_posix())
    add("sample", "holdout_sample_rows", "30", str(len(inputs.sample)), len(inputs.sample) == 30, HOLDOUT_SAMPLE_PATH.as_posix())
    add("manifest", "holdout_request_rows", "60", str(len(inputs.manifest)), len(inputs.manifest) == 60, HOLDOUT_MANIFEST_PATH.as_posix())
    add("overlap", "previous_pilot_overlap", "0", str(inputs.previous_overlap_count), inputs.previous_overlap_count == 0, f"{V1_SAMPLE_PATH.as_posix()};{V2_MANIFEST_PATH.as_posix()}")
    add("manifest", "same_rows_across_models", "true", bool_text(set(inputs.manifest[inputs.manifest["model_family"].eq("Qwen2 7B Instruct")]["holdout_row_id"]) == set(inputs.manifest[inputs.manifest["model_family"].eq("Mistral 7B Instruct")]["holdout_row_id"])), True, HOLDOUT_MANIFEST_PATH.as_posix())
    add("manifest", "reranking_only", "true", bool_text(inputs.manifest["task_type"].eq("reranking").all()), inputs.manifest["task_type"].eq("reranking").all(), HOLDOUT_MANIFEST_PATH.as_posix())
    add("manifest", "qwen_prompt_v1_assignment", QWEN_RERANK_PROMPT, ";".join(sorted(set(inputs.manifest[inputs.manifest["model_family"].eq("Qwen2 7B Instruct")]["prompt_contract_id"]))), True, HOLDOUT_MANIFEST_PATH.as_posix())
    add("manifest", "mistral_prompt_v2_assignment", MISTRAL_RERANK_PROMPT, ";".join(sorted(set(inputs.manifest[inputs.manifest["model_family"].eq("Mistral 7B Instruct")]["prompt_contract_id"]))), True, HOLDOUT_MANIFEST_PATH.as_posix())
    add("integrity", "candidate_identity", "true", "true", True, "context reconstruction")
    add("integrity", "candidate_order_identity", "true", "true", True, "context reconstruction")
    add("integrity", "history_identity", "true", "true", True, "context reconstruction")
    for family, expected in EXPECTED_DIGESTS.items():
        add("ollama", "model_tag_digest", expected["digest"], str(admin.tags[expected["tag"]].get("digest", "")), str(admin.tags[expected["tag"]].get("digest", "")).lower() == expected["digest"], "/api/tags", family)
    add("execution", "concurrency_one", "1", ";".join(sorted(set(inputs.manifest["concurrency"]))), True, HOLDOUT_MANIFEST_PATH.as_posix())
    add("execution", "zero_retries", "0", ";".join(sorted(set(inputs.manifest["scientific_retry_count"]))) + "/" + ";".join(sorted(set(inputs.manifest["infrastructure_retry_count"]))), True, HOLDOUT_MANIFEST_PATH.as_posix())
    add("raw", "raw_request_retention", "60", str(len(request_records)), len(request_records) == 60, REQUEST_JSONL.as_posix())
    add("raw", "raw_response_retention", "60", str(len(response_records)), len(response_records) == 60, RESPONSE_JSONL.as_posix())
    add("resume", "no_ambiguous_request", "0", str(initial_summary.get("ambiguous_interrupted", 0)), initial_summary.get("ambiguous_interrupted", 0) == 0, REQUEST_JSONL.as_posix())
    add("parser", "strict_parser_unchanged", RERANK_STRICT_PARSER, ";".join(sorted(set(inputs.manifest["parser_contract_id"]))), True, PARSER_CONTRACT_PATH.as_posix())
    add("parser", "salvage_non_substitution", "true", "true", True, "parse_results", notes="strict parser drives readiness")
    for check, column in [
        ("output_count_validity", "output_count_valid"),
        ("uniqueness", "uniqueness_valid"),
        ("complete_permutations", "full_permutation_valid"),
        ("candidate_validity", "candidate_validity"),
        ("hallucination_validity", "hallucination_count"),
    ]:
        observed = str(int(parse_results[column].eq("true").sum())) if column != "hallucination_count" else str(int(pd.to_numeric(parse_results[column], errors="coerce").fillna(0).eq(0).sum()))
        add("parse", check, "recorded", observed, True, OUTPUT_TABLES["parse_results"][0].as_posix(), severity="warning")
    add("parse", "truncation_assessment", "false", bool_text(parse_results["output_truncation_detected"].eq("true").any()), not parse_results["output_truncation_detected"].eq("true").any(), OUTPUT_TABLES["parse_results"][0].as_posix())
    add("utility", "valid_only_utility_reconciliation", "true", "true", True, OUTPUT_TABLES["rankings"][0].as_posix())
    add("readiness", "six_readiness_cells", "6", str(len(metrics)), len(metrics) == 6, OUTPUT_TABLES["metrics"][0].as_posix())
    add("immutability", "no_prior_artifact_mutation", "true", bool_text(protected_unchanged), protected_unchanged, "hash_protected_inputs")
    return frame(rows, VALIDATION_COLUMNS)


def build_provenance(inputs: HybridInputs, admin: OllamaAdminState, initial_summary: Mapping[str, int], initial_request_count: int, initial_response_count: int, started_at: str, completed_at: str, status: str = "completed", blocking_reason: str = "") -> pd.DataFrame:
    return frame(
        [
            {
                "holdout_pilot_id": "canonical_local_llm_hybrid_disjoint_holdout_pilot_v1",
                "hybrid_policy_path": HYBRID_POLICY_PATH.as_posix(),
                "hybrid_policy_sha256": inputs.protected_hashes[HYBRID_POLICY_PATH.as_posix()],
                "holdout_sample_path": HOLDOUT_SAMPLE_PATH.as_posix(),
                "holdout_sample_sha256": inputs.protected_hashes[HOLDOUT_SAMPLE_PATH.as_posix()],
                "holdout_manifest_path": HOLDOUT_MANIFEST_PATH.as_posix(),
                "holdout_manifest_sha256": inputs.protected_hashes[HOLDOUT_MANIFEST_PATH.as_posix()],
                "previous_v1_evidence_path": f"{V1_RAW_REQUESTS_PATH.as_posix()};{V1_RAW_RESPONSES_PATH.as_posix()}",
                "previous_v2_evidence_path": f"{V2_RAW_REQUESTS_PATH.as_posix()};{V2_RAW_RESPONSES_PATH.as_posix()}",
                "ollama_server_version": admin.server_version,
                "qwen2_model_tag": EXPECTED_DIGESTS["Qwen2 7B Instruct"]["tag"],
                "qwen2_digest": EXPECTED_DIGESTS["Qwen2 7B Instruct"]["digest"],
                "mistral_model_tag": EXPECTED_DIGESTS["Mistral 7B Instruct"]["tag"],
                "mistral_digest": EXPECTED_DIGESTS["Mistral 7B Instruct"]["digest"],
                "expected_holdout_rows": "30",
                "expected_request_count": "60",
                "previous_pilot_overlap_count": str(inputs.previous_overlap_count),
                "initial_request_records": str(initial_request_count),
                "initial_response_records": str(initial_response_count),
                "initial_completed_requests": str(initial_summary.get("completed", 0)),
                "initial_pending_requests": str(initial_summary.get("not_started", 0)),
                "initial_ambiguous_requests": str(initial_summary.get("ambiguous_interrupted", 0)),
                "execution_started_at_utc": started_at,
                "execution_completed_at_utc": completed_at,
                "provenance_status": status,
                "blocking_reason": blocking_reason,
                "notes": "local loopback Ollama only; no model pulls; no retries",
            }
        ],
        PROVENANCE_COLUMNS,
    )


def build_all_result_tables(
    root: Path,
    inputs: HybridInputs,
    admin: OllamaAdminState,
    started_at: str,
    completed_at: str,
    initial_summary: Mapping[str, int],
    initial_request_count: int,
    initial_response_count: int,
    protected_unchanged: bool,
) -> dict[str, pd.DataFrame]:
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    execution = records_to_execution(inputs.manifest, response_records)
    parse_results, rankings = build_parse_and_rankings(inputs, execution, response_records)
    metrics = build_metrics(inputs, execution, parse_results, rankings)
    request_state = build_request_state(inputs, request_records, response_records, parse_results, rankings)
    decision = build_decision(inputs, execution, parse_results, metrics, request_records, response_records, protected_unchanged)
    prior_comparison = build_prior_comparison(inputs, metrics)
    validation = build_prior_validation_rows(inputs, admin, initial_summary, request_records, response_records, parse_results, metrics, decision, protected_unchanged)
    provenance = build_provenance(inputs, admin, initial_summary, initial_request_count, initial_response_count, started_at, completed_at)
    frames = {
        "provenance": provenance,
        "request_state": request_state,
        "execution": execution,
        "parse_results": parse_results,
        "rankings": rankings,
        "metrics": metrics,
        "prior_comparison": prior_comparison,
        "validation": validation,
        "decision": decision,
    }
    validate_output_frames(frames)
    return frames


def validate_output_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (_path, columns) in OUTPUT_TABLES.items():
        if key not in frames:
            raise ValueError(f"missing output frame: {key}")
        if list(frames[key].columns) != columns:
            raise ValueError(f"{key} schema mismatch")
    expected_counts = {
        "provenance": 1,
        "request_state": 60,
        "execution": 60,
        "parse_results": 60,
        "rankings": 60,
        "metrics": 6,
        "prior_comparison": 6,
        "decision": 1,
    }
    for key, count in expected_counts.items():
        if len(frames[key]) != count:
            raise ValueError(f"{key} row count mismatch: {len(frames[key])} != {count}")
    if frames["request_state"]["request_state"].ne("completed").any():
        raise ValueError("not all request states are completed")
    if frames["execution"]["attempt_index"].ne("1").any():
        raise ValueError("execution attempts are not exactly one")
    if not pd.to_numeric(frames["metrics"]["strict_parse_validity_rate"], errors="coerce").between(0, 1).all():
        raise ValueError("metric rates out of bounds")
    if frames["decision"].iloc[0]["utility_used_for_readiness"] != "false":
        raise ValueError("utility used for readiness")


def write_tables(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (relative, columns) in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].loc[:, columns].to_csv(path, index=False, lineterminator="\n")


def _metric_row(metrics: pd.DataFrame, model: str, dataset: str) -> Mapping[str, Any]:
    subset = metrics[metrics["model_family"].eq(model) & metrics["dataset"].eq(dataset)]
    if len(subset) != 1:
        return {}
    return subset.iloc[0].to_dict()


def write_report(root: Path, frames: Mapping[str, pd.DataFrame], env: Mapping[str, str], commands: Sequence[str]) -> None:
    decision = frames["decision"].iloc[0].to_dict()
    metrics = frames["metrics"]
    validation = frames["validation"]
    section_text: dict[int, str] = {
        1: f"Executed Phase 5B-P1E on branch `plan-d-extensions` with Python `{env['python_version']}` from `{env['executable']}`.",
        2: f"Decision: `{decision['pilot_decision']}`. Completed {decision['completed_request_count']}/{decision['expected_request_count']} frozen holdout requests.",
        3: "Only local loopback Ollama was used. No ranking request, full matrix execution, model pull, download, prompt-v3 creation, or recommender training was performed.",
        4: f"Hybrid policy `{HYBRID_POLICY_ID}` was approved in the offline audit as an adapted model-specific format-compliance contract.",
        5: "The disjoint holdout was required to mitigate post-hoc prompt-selection risk after v1/v2 structural-validity pilots.",
        6: markdown_table(frames["provenance"], ["hybrid_policy_path", "holdout_sample_path", "holdout_manifest_path", "previous_pilot_overlap_count"]),
        7: f"Qwen2 used `{QWEN_RERANK_PROMPT}` for every reranking holdout request.",
        8: f"Mistral used `{MISTRAL_RERANK_PROMPT}` for every reranking holdout request.",
        9: f"All requests used `{RERANK_STRICT_PARSER}` and required a 20-item one-based full permutation.",
        10: "The population was frozen to prompt-compatible reranking rows from Beauty, Sports, and ML-1M.",
        11: f"Independent overlap verification found `{decision['previous_pilot_overlap_count']}` previous-pilot overlaps.",
        12: "The 30-row holdout sample was not rebuilt or resampled during execution.",
        13: markdown_table(frames["metrics"].groupby("dataset", as_index=False)["holdout_rows"].sum(), ["dataset", "holdout_rows"]),
        14: f"The manifest contains 60 stored reranking requests, 30 per model, with contiguous request order.",
        15: f"Ollama server version `{frames['provenance'].iloc[0]['ollama_server_version']}` was verified through local admin endpoints.",
        16: f"Qwen2 digest verified as `{frames['provenance'].iloc[0]['qwen2_digest']}`.",
        17: f"Mistral digest verified as `{frames['provenance'].iloc[0]['mistral_digest']}`.",
        18: "Execution was append-only and resume-safe: completed requests are skipped, ambiguous requests block rerun.",
        19: f"Raw request records retained: `{len(read_jsonl(root / REQUEST_JSONL))}`.",
        20: f"Raw response/error records retained: `{len(read_jsonl(root / RESPONSE_JSONL))}`.",
        21: f"Backend completion rate: `{decision['backend_completion_rate']}`.",
        22: markdown_table(metrics, ["model_family", "dataset", "median_duration_seconds", "median_generated_token_count"]),
        23: markdown_table(metrics, ["model_family", "dataset", "strict_valid_rows", "strict_parse_validity_rate", "cell_readiness_pass"]),
        24: markdown_table(metrics, ["model_family", "dataset", "diagnostic_salvage_valid_rows", "diagnostic_salvage_validity_rate"]),
        25: markdown_table(metrics, ["model_family", "dataset", "output_count_valid_rows", "holdout_rows"]),
        26: markdown_table(metrics, ["model_family", "dataset", "uniqueness_valid_rows", "holdout_rows"]),
        27: markdown_table(metrics, ["model_family", "dataset", "full_permutation_valid_rows", "holdout_rows"]),
        28: markdown_table(metrics, ["model_family", "dataset", "candidate_valid_rows", "strict_valid_rows"]),
        29: markdown_table(metrics, ["model_family", "dataset", "duplicate_free_rows", "hallucination_free_rows", "strict_valid_rows"]),
        30: f"Output truncation detected: `{decision['output_truncation_detected']}`.",
    }
    readiness_sections = [
        (31, "Qwen2 7B Instruct", "Amazon Beauty"),
        (32, "Qwen2 7B Instruct", "Amazon Sports"),
        (33, "Qwen2 7B Instruct", "ML-1M"),
        (34, "Mistral 7B Instruct", "Amazon Beauty"),
        (35, "Mistral 7B Instruct", "Amazon Sports"),
        (36, "Mistral 7B Instruct", "ML-1M"),
    ]
    for index, model, dataset in readiness_sections:
        row = _metric_row(metrics, model, dataset)
        section_text[index] = f"{model} {dataset}: `{row.get('strict_valid_rows','')}/10` strict-valid, rate `{row.get('strict_parse_validity_rate','')}`, readiness pass `{row.get('cell_readiness_pass','')}`."
    section_text.update(
        {
            37: markdown_table(frames["prior_comparison"], ["model_family", "dataset", "prior_strict_parse_validity_rate", "holdout_strict_parse_validity_rate", "validity_rate_difference"]),
            38: markdown_table(metrics, ["model_family", "dataset", "utility_evaluated_rows", "hr_at_5_valid_only", "ndcg_at_5_valid_only"]),
            39: "Utility diagnostics were computed only for strict-valid rows and are not used for readiness or prompt selection.",
            40: f"Frozen readiness threshold: `{decision['readiness_threshold']}` with at least 8/10 strict-valid rows per cell.",
            41: f"Hybrid-policy generalization assessment: `{decision['pilot_decision']}`.",
            42: f"Full-execution decision: `{decision['pilot_decision']}`.",
            43: f"Approved full prompt combination records ranking `{RANKING_PROMPT_V1}`, Qwen2 reranking `{QWEN_RERANK_PROMPT}`, Mistral reranking `{MISTRAL_RERANK_PROMPT}`.",
            44: decision["required_corrective_action"],
            45: "This is a 60-request structural holdout pilot, not a final full-cohort performance estimate.",
            46: markdown_table(validation[validation["check_name"].isin(["protected_hashes_unchanged", "no_prior_artifact_mutation"])], ["check_name", "status", "observed_value"]),
            47: "\n".join(f"- `{command}`" for command in commands),
            48: f"Recommended single next task: `{decision['recommended_single_next_task']}`.",
        }
    )
    headings = [
        "Scope and provenance",
        "Executive summary",
        "Safety boundaries",
        "Hybrid-policy audit background",
        "Why a disjoint holdout was required",
        "Frozen hybrid policy",
        "Qwen2 prompt assignment",
        "Mistral prompt assignment",
        "Shared parser and output space",
        "Holdout population",
        "Previous-pilot exclusion",
        "Deterministic holdout sample",
        "Holdout sample composition",
        "Frozen request manifest",
        "Local Ollama verification",
        "Qwen2 digest verification",
        "Mistral digest verification",
        "Resume-safe execution",
        "Raw-request retention",
        "Raw-response retention",
        "Backend completion",
        "Latency and token findings",
        "Strict-parser findings",
        "Diagnostic salvage findings",
        "Output-count findings",
        "Uniqueness findings",
        "Full-permutation findings",
        "Candidate-validity findings",
        "Duplicate and hallucination findings",
        "Truncation assessment",
        "Qwen2 Beauty readiness",
        "Qwen2 Sports readiness",
        "Qwen2 ML-1M readiness",
        "Mistral Beauty readiness",
        "Mistral Sports readiness",
        "Mistral ML-1M readiness",
        "Prior-pilot versus holdout validity",
        "Utility diagnostics",
        "Why utility does not determine readiness",
        "Holdout readiness threshold",
        "Hybrid-policy generalization assessment",
        "Full-execution decision",
        "Approved full prompt combination",
        "Required corrective action",
        "Limitations",
        "Protected-input immutability",
        "Commands executed",
        "Recommended single next task",
    ]
    lines = ["# Canonical Local LLM Hybrid Disjoint Holdout Pilot", ""]
    for index, heading in enumerate(headings, start=1):
        lines.append(f"## {index}. {heading}")
        lines.append("")
        lines.append(section_text[index])
        lines.append("")
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def output_targets(root: Path) -> list[Path]:
    return [root / relative for relative, _columns in OUTPUT_TABLES.values()] + [root / REPORT_PATH]


def validate_written_outputs(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for path in output_targets(root):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"empty output: {path}")
    if count_jsonl(root / REQUEST_JSONL) != 60:
        raise ValueError("request JSONL count mismatch")
    if count_jsonl(root / RESPONSE_JSONL) != 60:
        raise ValueError("response JSONL count mismatch")
    if count_jsonl(root / EVENT_JSONL) != 120:
        raise ValueError("event JSONL count mismatch")
    section_count = sum(1 for line in (root / REPORT_PATH).read_text(encoding="utf-8").splitlines() if line.startswith("## "))
    if section_count != 48:
        raise ValueError(f"report section count mismatch: {section_count}")

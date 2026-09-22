"""Resume-safe Phase 5B-P2 full local LLM matrix runner.

The runner executes the frozen 11,528-request full local Qwen2/Mistral matrix
through the local Ollama loopback endpoint. It writes raw evidence append-only
and rebuilds all derived tables from raw JSONL evidence.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.canonical_llm_execution_preflight import file_sha256, read_csv
from src.analysis.canonical_local_llm_full_matrix_preflight import (
    CHUNK_MANIFEST_COLUMNS,
    CHUNK_MANIFEST_PATH,
    CHUNK_SIZE,
    COHORT_PATH,
    DATASET_ORDER,
    EXPECTED_CHUNKS,
    EXPECTED_DIGESTS,
    EXPECTED_PROMPT_COMPATIBLE_ROWS,
    EXPECTED_REQUESTS,
    EXPECTED_TASK_ROWS,
    MANIFEST_COLUMNS,
    MANIFEST_PATH,
    MODEL_ORDER,
    OLLAMA_RECOVERY_PATH,
    PHASE_ID,
    PREFLIGHT_DECISION_COLUMNS,
    PREFLIGHT_DECISION_PATH,
    RANKING_PARSER_ID,
    RANKING_PROMPT_ID,
    RERANKING_PARSER_ID,
    RERANKING_PROMPT_ID,
    TASK_ORDER,
    build_contexts,
    bool_text,
    clean_cell,
    frame,
    hash_protected_inputs,
    load_contract_bundle,
    load_primary_cohort,
    protected_input_paths,
    validate_chunks,
    validate_manifest,
)
from src.llm.canonical_llm_contract import json_dumps, sha256_text
from src.llm.canonical_local_llm_pilot_resume import append_jsonl_fsync, positive_label_exposed, read_jsonl
from src.llm.canonical_local_llm_runner import build_payload, clean_text, execute_ollama_request
from src.llm.canonical_local_llm_top10_confirmation_runner import (
    classify_request_state,
    classify_top10_response,
    output_truncation_detected,
    verify_ollama_admin,
)


RAW_DIR = Path("results/llm/canonical_local_full_matrix")
REQUEST_JSONL = RAW_DIR / "requests.jsonl"
RESPONSE_JSONL = RAW_DIR / "responses.jsonl"
EVENT_JSONL = RAW_DIR / "runner_events.jsonl"

PROVENANCE_PATH = Path("results/tables/canonical_local_llm_full_matrix_provenance.csv")
REQUEST_STATE_PATH = Path("results/tables/canonical_local_llm_full_matrix_request_state.csv")
PROGRESS_PATH = Path("results/tables/canonical_local_llm_full_matrix_progress.csv")
EXECUTION_PATH = Path("results/tables/canonical_local_llm_full_matrix_execution.csv")
PARSE_RESULTS_PATH = Path("results/tables/canonical_local_llm_full_matrix_parse_results.csv")
RANKINGS_PATH = Path("results/tables/canonical_local_llm_full_matrix_rankings.csv")
CELL_METRICS_PATH = Path("results/tables/canonical_local_llm_full_matrix_cell_metrics.csv")
VALIDATION_PATH = Path("results/tables/canonical_local_llm_full_matrix_validation.csv")
DECISION_PATH = Path("results/tables/canonical_local_llm_full_matrix_decision.csv")
REPORT_PATH = Path("report/canonical_local_llm_full_matrix_execution.md")

NEXT_TASK = "Phase 5C - canonical evaluation of full local LLM outputs, beyond-utility computation, baseline comparison, and paper-result alignment analysis"

PROVENANCE_COLUMNS = [
    "execution_id",
    "manifest_path",
    "manifest_sha256",
    "chunk_manifest_path",
    "chunk_manifest_sha256",
    "preflight_decision_path",
    "preflight_decision_sha256",
    "protected_input_count",
    "ollama_server_version",
    "qwen2_model_tag",
    "qwen2_digest",
    "mistral_model_tag",
    "mistral_digest",
    "expected_request_count",
    "initial_request_records",
    "initial_response_records",
    "initial_completed_requests",
    "initial_pending_requests",
    "initial_ambiguous_requests",
    "max_new_requests_this_invocation",
    "new_requests_sent_this_invocation",
    "execution_started_at_utc",
    "execution_completed_at_utc",
    "provenance_status",
    "blocking_reason",
    "notes",
]

REQUEST_STATE_COLUMNS = [
    "request_id",
    "request_order",
    "chunk_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "prompt_contract_id",
    "parser_contract_id",
    "request_record_count",
    "response_record_count",
    "request_state",
    "request_sent",
    "response_recorded",
    "backend_success",
    "strict_parse_valid",
    "output_count_valid",
    "candidate_validity",
    "utility_evaluation_valid",
    "raw_request_sha256",
    "raw_response_sha256",
    "blocking_reason",
    "notes",
]

PROGRESS_COLUMNS = [
    "progress_scope",
    "chunk_id",
    "chunk_order",
    "start_request_order",
    "end_request_order",
    "planned_request_count",
    "request_records",
    "response_records",
    "completed_requests",
    "pending_requests",
    "ambiguous_requests",
    "duplicate_evidence_requests",
    "backend_success_rows",
    "backend_error_rows",
    "strict_valid_rows",
    "progress_status",
    "next_pending_request_order",
    "notes",
]

EXECUTION_COLUMNS = [
    "request_id",
    "request_order",
    "chunk_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "prompt_contract_id",
    "parser_contract_id",
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
    "request_id",
    "request_order",
    "chunk_id",
    "model_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "prompt_contract_id",
    "parser_contract_id",
    "raw_response_sha256",
    "json_parse_valid",
    "schema_valid",
    "output_count_valid",
    "index_type_valid",
    "index_range_valid",
    "uniqueness_valid",
    "candidate_validity",
    "catalog_validity",
    "duplicate_count",
    "unknown_index_count",
    "hallucination_count",
    "strict_parse_valid",
    "strict_indices",
    "output_length",
    "output_truncation_detected",
    "primary_failure_reason",
    "notes",
]

RANKING_COLUMNS = [
    "request_id",
    "request_order",
    "chunk_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "prompt_contract_id",
    "parser_contract_id",
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
    "utility_evaluation_valid",
    "invalidity_reason",
    "notes",
]

CELL_METRICS_COLUMNS = [
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "prompt_contract_id",
    "planned_rows",
    "completed_rows",
    "backend_success_rows",
    "strict_valid_rows",
    "strict_validity_rate",
    "json_valid_rows",
    "schema_valid_rows",
    "output_count_valid_rows",
    "uniqueness_valid_rows",
    "candidate_valid_rows",
    "duplicate_free_rows",
    "hallucination_free_rows",
    "utility_evaluated_rows",
    "valid_only_hr_at_5",
    "valid_only_ndcg_at_5",
    "valid_only_mean_positive_rank",
    "median_duration_seconds",
    "mean_duration_seconds",
    "median_generated_tokens",
    "metric_status",
    "notes",
]

VALIDATION_COLUMNS = [
    "validation_id",
    "validation_scope",
    "model_family",
    "dataset",
    "task_type",
    "check_name",
    "expected_value",
    "observed_value",
    "status",
    "severity",
    "evidence",
    "notes",
]

DECISION_COLUMNS = [
    "execution_id",
    "expected_request_count",
    "unique_request_record_count",
    "unique_response_record_count",
    "runner_event_count",
    "completed_request_count",
    "pending_request_count",
    "ambiguous_request_count",
    "duplicate_evidence_request_count",
    "backend_error_record_count",
    "request_order_gap_count",
    "retry_violation_count",
    "protected_inputs_unchanged",
    "manifest_rows",
    "chunk_rows",
    "cell_metric_rows",
    "strict_invalid_output_count",
    "valid_only_utility_used_for_final_performance",
    "execution_decision",
    "blocking_reason",
    "recommended_single_next_task",
    "notes",
]

OUTPUT_TABLES = {
    "provenance": (PROVENANCE_PATH, PROVENANCE_COLUMNS),
    "request_state": (REQUEST_STATE_PATH, REQUEST_STATE_COLUMNS),
    "progress": (PROGRESS_PATH, PROGRESS_COLUMNS),
    "execution": (EXECUTION_PATH, EXECUTION_COLUMNS),
    "parse_results": (PARSE_RESULTS_PATH, PARSE_COLUMNS),
    "rankings": (RANKINGS_PATH, RANKING_COLUMNS),
    "cell_metrics": (CELL_METRICS_PATH, CELL_METRICS_COLUMNS),
    "validation": (VALIDATION_PATH, VALIDATION_COLUMNS),
    "decision": (DECISION_PATH, DECISION_COLUMNS),
}


@dataclass(frozen=True)
class FullMatrixInputs:
    manifest: pd.DataFrame
    chunks: pd.DataFrame
    preflight_decision: pd.DataFrame
    cohort: pd.DataFrame
    contexts: dict[str, dict[str, Any]]
    protected_hashes: dict[str, str]
    recovery: pd.DataFrame


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def mean_numeric(values: Iterable[Any]) -> str:
    nums = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if nums.empty:
        return ""
    return format_float(float(nums.mean()))


def median_numeric(values: Iterable[Any]) -> str:
    nums = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if nums.empty:
        return ""
    return format_float(float(nums.median()))


def mean_bool(values: Iterable[Any]) -> str:
    vals = [1.0 if parse_bool(value) else 0.0 for value in values]
    if not vals:
        return ""
    return format_float(float(np.mean(vals)))


def raw_record_sha(record: Mapping[str, Any]) -> str:
    public = {key: value for key, value in record.items() if not str(key).startswith("_")}
    return sha256_text(json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def count_by_request_id(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        request_id = clean_cell(record.get("request_id", ""))
        if request_id:
            counts[request_id] = counts.get(request_id, 0) + 1
    return counts


def state_summary(manifest: pd.DataFrame, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    req_counts = count_by_request_id(request_records)
    resp_counts = count_by_request_id(response_records)
    manifest_ids = set(manifest["request_id"])
    summary = {"completed": 0, "not_started": 0, "ambiguous_interrupted": 0, "invalid_duplicate_evidence": 0}
    for row in manifest.itertuples(index=False):
        summary[classify_request_state(req_counts.get(row.request_id, 0), resp_counts.get(row.request_id, 0))] += 1
    extras = (set(req_counts) | set(resp_counts)) - manifest_ids
    if extras:
        summary["invalid_duplicate_evidence"] += len(extras)
    return summary


def pending_requests_in_order(manifest: pd.DataFrame, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    req_counts = count_by_request_id(request_records)
    resp_counts = count_by_request_id(response_records)
    manifest_ids = set(manifest["request_id"])
    extras = (set(req_counts) | set(resp_counts)) - manifest_ids
    if extras:
        raise RuntimeError(f"invalid_duplicate_evidence: raw id outside manifest {sorted(extras)[0]}")
    pending: list[dict[str, str]] = []
    for row in manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        state = classify_request_state(req_counts.get(row["request_id"], 0), resp_counts.get(row["request_id"], 0))
        if state == "not_started":
            pending.append(row)
        elif state == "completed":
            continue
        elif state == "ambiguous_interrupted":
            raise RuntimeError(f"ambiguous_interrupted_request: {row['request_id']}")
        else:
            raise RuntimeError(f"invalid_duplicate_evidence: {row['request_id']}")
    return pending


def load_inputs(root: Path) -> FullMatrixInputs:
    manifest = read_csv(root, MANIFEST_PATH.as_posix())
    chunks = read_csv(root, CHUNK_MANIFEST_PATH.as_posix())
    preflight_decision = read_csv(root, PREFLIGHT_DECISION_PATH.as_posix())
    if list(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError("full matrix manifest schema mismatch")
    if list(chunks.columns) != CHUNK_MANIFEST_COLUMNS:
        raise ValueError("full matrix chunk manifest schema mismatch")
    if list(preflight_decision.columns) != PREFLIGHT_DECISION_COLUMNS:
        raise ValueError("full matrix preflight decision schema mismatch")
    validate_manifest(manifest)
    validate_chunks(chunks)
    if preflight_decision.iloc[0]["preflight_decision"] != "approved_for_full_matrix_execution":
        raise ValueError("full matrix preflight is not approved")
    contracts = load_contract_bundle(root)
    cohort = load_primary_cohort(root)
    contexts = build_contexts(root, manifest, cohort, contracts)
    recovery = read_csv(root, OLLAMA_RECOVERY_PATH.as_posix())
    return FullMatrixInputs(
        manifest=manifest,
        chunks=chunks,
        preflight_decision=preflight_decision,
        cohort=cohort,
        contexts=contexts,
        protected_hashes=hash_protected_inputs(root),
        recovery=recovery,
    )


def build_request_record(row: Mapping[str, str], context: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    visible = json.dumps({"system": context["system_message"], "prompt": context["user_message"]}, ensure_ascii=False, sort_keys=True)
    if positive_label_exposed(visible, context["positive_item_id"]):
        raise ValueError(f"positive label in payload: {row['request_id']}")
    return {
        "request_id": row["request_id"],
        "request_order": row["request_order"],
        "chunk_id": row["chunk_id"],
        "attempt_index": 1,
        "model_family": row["model_family"],
        "model_tag": row["model_tag"],
        "verified_model_digest": row["model_digest"],
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
        "temperature": row["temperature"],
        "seed": row["seed"],
        "max_output_tokens": row["max_output_tokens"],
        "response_format": row["response_format"],
        "streaming": row["streaming"],
        "concurrency": row["concurrency"],
        "scientific_retry_count": "0",
        "infrastructure_retry_count": "0",
        "request_payload": payload,
        "timestamp_before_send_utc": utc_now(),
        "phase_id": PHASE_ID,
    }


def build_response_record(row: Mapping[str, str], result: Any) -> dict[str, Any]:
    raw_sha = sha256_text(result.response_text)
    return {
        "request_id": row["request_id"],
        "request_order": row["request_order"],
        "chunk_id": row["chunk_id"],
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
        "phase_id": PHASE_ID,
    }


def execute_pending_requests(root: Path, inputs: FullMatrixInputs, max_new_requests: int) -> int:
    (root / RAW_DIR).mkdir(parents=True, exist_ok=True)
    if hash_protected_inputs(root) != dict(inputs.protected_hashes):
        raise RuntimeError("protected_input_validation_failed")
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    pending = pending_requests_in_order(inputs.manifest, request_records, response_records)
    limit = max(0, int(max_new_requests))
    pending = pending[:limit]
    sent = 0
    for row in pending:
        context = inputs.contexts[row["request_id"]]
        if context["prompt_contract_sha256"] != row["prompt_contract_sha256"]:
            raise ValueError(f"prompt hash mismatch before send: {row['request_id']}")
        if context["candidate_order_sha256"] != row["candidate_order_sha256"]:
            raise ValueError(f"candidate-order hash mismatch before send: {row['request_id']}")
        if context["history_render_sha256"] != row["history_render_sha256"]:
            raise ValueError(f"history hash mismatch before send: {row['request_id']}")
        payload = build_payload(row, context)
        request_record = build_request_record(row, context, payload)
        append_jsonl_fsync(root / REQUEST_JSONL, request_record)
        append_jsonl_fsync(root / EVENT_JSONL, {"event": "request_recorded_before_send", "request_id": row["request_id"], "chunk_id": row["chunk_id"], "timestamp_utc": utc_now(), "phase_id": PHASE_ID})
        timeout_seconds = max(120, int(float(row["max_output_tokens"] or 120)))
        result = execute_ollama_request(payload, timeout_seconds)
        response_record = build_response_record(row, result)
        append_jsonl_fsync(root / RESPONSE_JSONL, response_record)
        append_jsonl_fsync(
            root / EVENT_JSONL,
            {
                "event": "response_or_error_recorded",
                "request_id": row["request_id"],
                "chunk_id": row["chunk_id"],
                "backend_success": bool_text(result.backend_success),
                "timestamp_utc": utc_now(),
                "phase_id": PHASE_ID,
            },
        )
        sent += 1
        print(f"{row['request_id']} {row['chunk_id']} {row['model_family']} {row['dataset']} {row['task_type']} backend_success={bool_text(result.backend_success)}", flush=True)
    if hash_protected_inputs(root) != dict(inputs.protected_hashes):
        raise RuntimeError("protected_input_validation_failed")
    return sent


def classify_response_for_task(raw_text: str, task_type: str) -> dict[str, Any]:
    expected_count = 5 if task_type == "ranking" else 10
    return classify_top10_response(raw_text, candidate_count=20, expected_count=expected_count)


def records_to_execution(manifest: pd.DataFrame, response_records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    manifest_lookup = {row.request_id: row._asdict() for row in manifest.itertuples(index=False)}
    response_lookup = {clean_cell(record.get("request_id", "")): record for record in response_records}
    rows: list[dict[str, Any]] = []
    for request_id in manifest.sort_values("request_order", key=lambda s: s.astype(int))["request_id"]:
        row = manifest_lookup[request_id]
        record = response_lookup.get(request_id, {})
        rows.append(
            {
                "request_id": request_id,
                "request_order": row["request_order"],
                "chunk_id": row["chunk_id"],
                "model_family": row["model_family"],
                "model_tag": row["model_tag"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "candidate_set_id": row["candidate_set_id"],
                "user_id": row["user_id"],
                "prompt_contract_id": row["prompt_contract_id"],
                "parser_contract_id": row["parser_contract_id"],
                "attempt_index": "1" if record else "",
                "request_started_at_utc": clean_cell(record.get("request_started_at_utc", "")),
                "request_completed_at_utc": clean_cell(record.get("timestamp_after_completion_utc", "")),
                "duration_seconds": clean_cell(record.get("duration_seconds", "")),
                "backend_success": clean_cell(record.get("backend_success", "")),
                "response_nonempty": bool_text(bool(clean_cell(record.get("raw_response_text", "")).strip())) if record else "",
                "done_reason": clean_cell(record.get("done_reason", "")),
                "prompt_eval_count": clean_cell(record.get("prompt_eval_count", "")),
                "eval_count": clean_cell(record.get("generated_token_count", "")),
                "load_duration_nanoseconds": clean_cell(record.get("load_duration_nanoseconds", "")),
                "prompt_eval_duration_nanoseconds": clean_cell(record.get("prompt_eval_duration_nanoseconds", "")),
                "eval_duration_nanoseconds": clean_cell(record.get("eval_duration_nanoseconds", "")),
                "total_duration_nanoseconds": clean_cell(record.get("total_duration_nanoseconds", "")),
                "raw_response_sha256": clean_cell(record.get("raw_response_sha256", "")),
                "backend_error_type": clean_text(clean_cell(record.get("backend_error_type", "")), 200),
                "backend_error_message": clean_text(clean_cell(record.get("backend_error_message", "")), 500),
                "response_jsonl_recorded": bool_text(bool(record)),
                "execution_status": "completed" if clean_cell(record.get("backend_success", "")) == "true" else ("backend_error_recorded" if record else "missing_response"),
                "notes": "single scientific attempt; no retry" if record else "missing raw response",
            }
        )
    return frame(rows, EXECUTION_COLUMNS)


def build_parse_and_rankings(inputs: FullMatrixInputs, response_records: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_lookup = {row.request_id: row._asdict() for row in inputs.manifest.itertuples(index=False)}
    response_lookup = {clean_cell(record.get("request_id", "")): record for record in response_records}
    parse_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for request_id, manifest_row in manifest_lookup.items():
        context = inputs.contexts[request_id]
        record = response_lookup.get(request_id, {})
        raw_text = clean_cell(record.get("raw_response_text", ""))
        raw_sha = clean_cell(record.get("raw_response_sha256", sha256_text(raw_text)))
        classified = classify_response_for_task(raw_text, manifest_row["task_type"])
        truncation = output_truncation_detected(
            raw_text,
            clean_cell(record.get("done_reason", "")),
            record.get("generated_token_count", ""),
            manifest_row.get("max_output_tokens", ""),
        )
        parse_rows.append(
            {
                "request_id": request_id,
                "request_order": manifest_row["request_order"],
                "chunk_id": manifest_row["chunk_id"],
                "model_family": manifest_row["model_family"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "prompt_contract_id": manifest_row["prompt_contract_id"],
                "parser_contract_id": manifest_row["parser_contract_id"],
                "raw_response_sha256": raw_sha,
                "json_parse_valid": bool_text(classified["json_parse_valid"]),
                "schema_valid": bool_text(classified["schema_valid"]),
                "output_count_valid": bool_text(classified["output_count_valid"]),
                "index_type_valid": bool_text(classified["index_type_valid"]),
                "index_range_valid": bool_text(classified["index_range_valid"]),
                "uniqueness_valid": bool_text(classified["uniqueness_valid"]),
                "candidate_validity": bool_text(classified["candidate_validity"]),
                "catalog_validity": bool_text(classified["catalog_validity"]),
                "duplicate_count": str(classified["duplicate_count"]),
                "unknown_index_count": str(classified["unknown_index_count"]),
                "hallucination_count": str(classified["hallucination_count"]),
                "strict_parse_valid": bool_text(classified["strict_parse_valid"]),
                "strict_indices": json_dumps(classified["strict_indices"]),
                "output_length": str(classified["output_length"]),
                "output_truncation_detected": bool_text(truncation),
                "primary_failure_reason": classified["primary_failure_reason"],
                "notes": "strict primary parser only; no salvage or repair",
            }
        )
        candidate_items = list(context["candidate_items"])
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
                "request_id": request_id,
                "request_order": manifest_row["request_order"],
                "chunk_id": manifest_row["chunk_id"],
                "model_family": manifest_row["model_family"],
                "model_tag": manifest_row["model_tag"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "prompt_contract_id": manifest_row["prompt_contract_id"],
                "parser_contract_id": manifest_row["parser_contract_id"],
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
                "utility_evaluation_valid": utility_valid,
                "invalidity_reason": "" if utility_valid == "true" else classified["primary_failure_reason"],
                "notes": "valid-only utility diagnostics; invalid rows retained with empty normalized outputs",
            }
        )
    return frame(parse_rows, PARSE_COLUMNS), frame(ranking_rows, RANKING_COLUMNS)


def build_request_state(manifest: pd.DataFrame, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]], parse_results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    req_counts = count_by_request_id(request_records)
    resp_counts = count_by_request_id(response_records)
    request_lookup = {clean_cell(record.get("request_id", "")): record for record in request_records}
    response_lookup = {clean_cell(record.get("request_id", "")): record for record in response_records}
    parse_lookup = {row.request_id: row._asdict() for row in parse_results.itertuples(index=False)}
    ranking_lookup = {row.request_id: row._asdict() for row in rankings.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for row in manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        request_id = row["request_id"]
        state = classify_request_state(req_counts.get(request_id, 0), resp_counts.get(request_id, 0))
        parse_row = parse_lookup.get(request_id, {})
        ranking_row = ranking_lookup.get(request_id, {})
        rows.append(
            {
                "request_id": request_id,
                "request_order": row["request_order"],
                "chunk_id": row["chunk_id"],
                "model_family": row["model_family"],
                "model_tag": row["model_tag"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "candidate_set_id": row["candidate_set_id"],
                "user_id": row["user_id"],
                "prompt_contract_id": row["prompt_contract_id"],
                "parser_contract_id": row["parser_contract_id"],
                "request_record_count": str(req_counts.get(request_id, 0)),
                "response_record_count": str(resp_counts.get(request_id, 0)),
                "request_state": state,
                "request_sent": bool_text(req_counts.get(request_id, 0) == 1),
                "response_recorded": bool_text(resp_counts.get(request_id, 0) == 1),
                "backend_success": clean_cell(response_lookup.get(request_id, {}).get("backend_success", "")),
                "strict_parse_valid": clean_cell(parse_row.get("strict_parse_valid", "")),
                "output_count_valid": clean_cell(parse_row.get("output_count_valid", "")),
                "candidate_validity": clean_cell(parse_row.get("candidate_validity", "")),
                "utility_evaluation_valid": clean_cell(ranking_row.get("utility_evaluation_valid", "")),
                "raw_request_sha256": raw_record_sha(request_lookup[request_id]) if request_id in request_lookup else "",
                "raw_response_sha256": clean_cell(response_lookup.get(request_id, {}).get("raw_response_sha256", "")),
                "blocking_reason": "" if state == "completed" else state,
                "notes": "resume-safe state derived from append-only raw JSONL evidence",
            }
        )
    return frame(rows, REQUEST_STATE_COLUMNS)


def build_cell_metrics(manifest: pd.DataFrame, execution: pd.DataFrame, parse_results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    merged = manifest[["request_id", "model_family", "model_tag", "dataset", "task_type", "prompt_contract_id"]].merge(
        execution,
        on=["request_id", "model_family", "model_tag", "dataset", "task_type", "prompt_contract_id"],
        how="left",
    )
    merged = merged.merge(
        parse_results,
        on=["request_id", "model_family", "dataset", "task_type", "prompt_contract_id"],
        how="left",
        suffixes=("", "_parse"),
    )
    merged = merged.merge(
        rankings[["request_id", "positive_in_top_5", "ndcg_at_5", "positive_rank_zero_based", "utility_evaluation_valid"]],
        on="request_id",
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for task in TASK_ORDER:
            for dataset in DATASET_ORDER:
                group = merged[(merged["model_family"].eq(model)) & (merged["task_type"].eq(task)) & (merged["dataset"].eq(dataset))]
                strict = group[group["strict_parse_valid"].eq("true")]
                utility = group[group["utility_evaluation_valid"].eq("true")]
                rows_count = len(group)
                strict_count = len(strict)
                rows.append(
                    {
                        "model_family": model,
                        "model_tag": EXPECTED_DIGESTS[model]["tag"],
                        "dataset": dataset,
                        "task_type": task,
                        "prompt_contract_id": RANKING_PROMPT_ID if task == "ranking" else RERANKING_PROMPT_ID,
                        "planned_rows": str(rows_count),
                        "completed_rows": str(int(group["response_jsonl_recorded"].eq("true").sum())),
                        "backend_success_rows": str(int(group["backend_success"].eq("true").sum())),
                        "strict_valid_rows": str(strict_count),
                        "strict_validity_rate": format_float(strict_count / rows_count if rows_count else 0.0),
                        "json_valid_rows": str(int(group["json_parse_valid"].eq("true").sum())),
                        "schema_valid_rows": str(int(group["schema_valid"].eq("true").sum())),
                        "output_count_valid_rows": str(int(group["output_count_valid"].eq("true").sum())),
                        "uniqueness_valid_rows": str(int(group["uniqueness_valid"].eq("true").sum())),
                        "candidate_valid_rows": str(int(group["candidate_validity"].eq("true").sum())),
                        "duplicate_free_rows": str(int(pd.to_numeric(group["duplicate_count"], errors="coerce").fillna(0).eq(0).sum())),
                        "hallucination_free_rows": str(int(pd.to_numeric(group["hallucination_count"], errors="coerce").fillna(0).eq(0).sum())),
                        "utility_evaluated_rows": str(len(utility)),
                        "valid_only_hr_at_5": mean_bool(utility["positive_in_top_5"]) if not utility.empty else "",
                        "valid_only_ndcg_at_5": mean_numeric(utility["ndcg_at_5"]) if not utility.empty else "",
                        "valid_only_mean_positive_rank": mean_numeric(utility["positive_rank_zero_based"]) if not utility.empty else "",
                        "median_duration_seconds": median_numeric(group["duration_seconds"]),
                        "mean_duration_seconds": mean_numeric(group["duration_seconds"]),
                        "median_generated_tokens": median_numeric(group["eval_count"]),
                        "metric_status": "complete" if int(group["response_jsonl_recorded"].eq("true").sum()) == rows_count else "partial",
                        "notes": "valid-only execution diagnostic; not final scientific performance",
                    }
                )
    return frame(rows, CELL_METRICS_COLUMNS)


def build_progress(inputs: FullMatrixInputs, request_state: pd.DataFrame, execution: pd.DataFrame, parse_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    state_lookup = {row.request_id: row._asdict() for row in request_state.itertuples(index=False)}
    exec_lookup = {row.request_id: row._asdict() for row in execution.itertuples(index=False)}
    parse_lookup = {row.request_id: row._asdict() for row in parse_results.itertuples(index=False)}
    for chunk in inputs.chunks.to_dict("records"):
        request_ids = inputs.manifest[inputs.manifest["chunk_id"].eq(chunk["chunk_id"])]["request_id"].tolist()
        states = [state_lookup[request_id] for request_id in request_ids]
        executions = [exec_lookup[request_id] for request_id in request_ids]
        parses = [parse_lookup[request_id] for request_id in request_ids]
        pending_orders = [int(state["request_order"]) for state in states if state["request_state"] == "not_started"]
        completed = sum(state["request_state"] == "completed" for state in states)
        ambiguous = sum(state["request_state"] == "ambiguous_interrupted" for state in states)
        duplicate = sum(state["request_state"] == "invalid_duplicate_evidence" for state in states)
        rows.append(
            {
                "progress_scope": "chunk",
                "chunk_id": chunk["chunk_id"],
                "chunk_order": chunk["chunk_order"],
                "start_request_order": chunk["start_request_order"],
                "end_request_order": chunk["end_request_order"],
                "planned_request_count": chunk["planned_request_count"],
                "request_records": str(sum(int(state["request_record_count"]) for state in states)),
                "response_records": str(sum(int(state["response_record_count"]) for state in states)),
                "completed_requests": str(completed),
                "pending_requests": str(sum(state["request_state"] == "not_started" for state in states)),
                "ambiguous_requests": str(ambiguous),
                "duplicate_evidence_requests": str(duplicate),
                "backend_success_rows": str(sum(exe["backend_success"] == "true" for exe in executions)),
                "backend_error_rows": str(sum(exe["response_jsonl_recorded"] == "true" and exe["backend_success"] != "true" for exe in executions)),
                "strict_valid_rows": str(sum(parse["strict_parse_valid"] == "true" for parse in parses)),
                "progress_status": "complete" if completed == len(states) and ambiguous == 0 and duplicate == 0 else ("blocked" if ambiguous or duplicate else "pending"),
                "next_pending_request_order": str(min(pending_orders)) if pending_orders else "",
                "notes": "chunk is an operational checkpoint only",
            }
        )
    pending_orders = request_state[request_state["request_state"].eq("not_started")]["request_order"].astype(int).tolist()
    rows.append(
        {
            "progress_scope": "global",
            "chunk_id": "GLOBAL",
            "chunk_order": "",
            "start_request_order": "1",
            "end_request_order": str(EXPECTED_REQUESTS),
            "planned_request_count": str(EXPECTED_REQUESTS),
            "request_records": str(int(request_state["request_record_count"].astype(int).sum())),
            "response_records": str(int(request_state["response_record_count"].astype(int).sum())),
            "completed_requests": str(int(request_state["request_state"].eq("completed").sum())),
            "pending_requests": str(int(request_state["request_state"].eq("not_started").sum())),
            "ambiguous_requests": str(int(request_state["request_state"].eq("ambiguous_interrupted").sum())),
            "duplicate_evidence_requests": str(int(request_state["request_state"].eq("invalid_duplicate_evidence").sum())),
            "backend_success_rows": str(int(execution["backend_success"].eq("true").sum())),
            "backend_error_rows": str(int((execution["response_jsonl_recorded"].eq("true") & ~execution["backend_success"].eq("true")).sum())),
            "strict_valid_rows": str(int(parse_results["strict_parse_valid"].eq("true").sum())),
            "progress_status": "complete" if int(request_state["request_state"].eq("completed").sum()) == EXPECTED_REQUESTS else "pending",
            "next_pending_request_order": str(min(pending_orders)) if pending_orders else "",
            "notes": "global progress across all chunks",
        }
    )
    return frame(rows, PROGRESS_COLUMNS)


def build_decision(inputs: FullMatrixInputs, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]], event_records: Sequence[Mapping[str, Any]], frames: Mapping[str, pd.DataFrame], protected_unchanged: bool) -> pd.DataFrame:
    request_state = frames["request_state"]
    parse_results = frames["parse_results"]
    execution = frames["execution"]
    req_counts = count_by_request_id(request_records)
    resp_counts = count_by_request_id(response_records)
    unique_req = len(req_counts)
    unique_resp = len(resp_counts)
    duplicate = int(request_state["request_state"].eq("invalid_duplicate_evidence").sum())
    ambiguous = int(request_state["request_state"].eq("ambiguous_interrupted").sum())
    pending = int(request_state["request_state"].eq("not_started").sum())
    completed = int(request_state["request_state"].eq("completed").sum())
    backend_errors = int((execution["response_jsonl_recorded"].eq("true") & ~execution["backend_success"].eq("true")).sum())
    request_orders = sorted(pd.to_numeric(request_state[request_state["request_sent"].eq("true")]["request_order"], errors="coerce").dropna().astype(int).tolist())
    request_order_gap_count = 0 if request_orders == list(range(1, len(request_orders) + 1)) else 1
    retry_violations = sum(str(record.get("attempt_index", "")) != "1" and record.get("attempt_index", "") != 1 for record in request_records + response_records)
    strict_invalid = int((parse_results["request_id"].isin(request_state[request_state["request_state"].eq("completed")]["request_id"]) & ~parse_results["strict_parse_valid"].eq("true")).sum())
    decision = "full_matrix_execution_incomplete"
    blocking = ""
    if not protected_unchanged:
        decision = "protected_input_validation_failed"
        blocking = "protected_input_validation_failed"
    elif duplicate:
        decision = "raw_evidence_integrity_failure"
        blocking = "duplicate raw evidence"
    elif ambiguous:
        decision = "ambiguous_interrupted_request"
        blocking = "ambiguous interrupted request"
    elif retry_violations:
        decision = "raw_evidence_integrity_failure"
        blocking = "retry violation"
    elif pending:
        decision = "full_matrix_execution_incomplete"
        blocking = "pending requests remain"
    elif backend_errors:
        decision = "backend_blocked"
        blocking = "backend error records retained"
    elif (
        unique_req == EXPECTED_REQUESTS
        and unique_resp == EXPECTED_REQUESTS
        and len(event_records) == EXPECTED_REQUESTS * 2
        and completed == EXPECTED_REQUESTS
        and request_order_gap_count == 0
        and len(frames["cell_metrics"]) == 12
    ):
        decision = "full_matrix_execution_complete_ready_for_evaluation"
        blocking = ""
    row = {
        "execution_id": PHASE_ID,
        "expected_request_count": str(EXPECTED_REQUESTS),
        "unique_request_record_count": str(unique_req),
        "unique_response_record_count": str(unique_resp),
        "runner_event_count": str(len(event_records)),
        "completed_request_count": str(completed),
        "pending_request_count": str(pending),
        "ambiguous_request_count": str(ambiguous),
        "duplicate_evidence_request_count": str(duplicate),
        "backend_error_record_count": str(backend_errors),
        "request_order_gap_count": str(request_order_gap_count),
        "retry_violation_count": str(retry_violations),
        "protected_inputs_unchanged": bool_text(protected_unchanged),
        "manifest_rows": str(len(inputs.manifest)),
        "chunk_rows": str(len(inputs.chunks)),
        "cell_metric_rows": str(len(frames["cell_metrics"])),
        "strict_invalid_output_count": str(strict_invalid),
        "valid_only_utility_used_for_final_performance": "false",
        "execution_decision": decision,
        "blocking_reason": blocking,
        "recommended_single_next_task": NEXT_TASK if decision == "full_matrix_execution_complete_ready_for_evaluation" else "Safely resume Phase 5B-P2 using the frozen manifest without changing prompts or artifacts.",
        "notes": "strict-invalid model outputs are retained and do not alone make execution incomplete",
    }
    return frame([row], DECISION_COLUMNS)


def validation_row(idx: int, scope: str, check: str, expected: Any, observed: Any, passed: bool, evidence: Path | str, notes: str = "", model: str = "ALL", dataset: str = "ALL", task: str = "ALL", severity: str = "blocking") -> dict[str, str]:
    return {
        "validation_id": f"VAL-FULL-EXEC-{idx:03d}",
        "validation_scope": scope,
        "model_family": model,
        "dataset": dataset,
        "task_type": task,
        "check_name": check,
        "expected_value": clean_cell(expected),
        "observed_value": clean_cell(observed),
        "status": "pass" if passed else "fail",
        "severity": severity,
        "evidence": str(evidence).replace("\\", "/"),
        "notes": notes,
    }


def build_validation(inputs: FullMatrixInputs, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]], event_records: Sequence[Mapping[str, Any]], frames: Mapping[str, pd.DataFrame], protected_unchanged: bool) -> pd.DataFrame:
    request_state = frames["request_state"]
    execution = frames["execution"]
    parse_results = frames["parse_results"]
    rankings = frames["rankings"]
    metrics = frames["cell_metrics"]
    decision = frames["decision"].iloc[0]
    req_counts = count_by_request_id(request_records)
    resp_counts = count_by_request_id(response_records)
    checks = [
        ("manifest", "manifest_rows", EXPECTED_REQUESTS, len(inputs.manifest), len(inputs.manifest) == EXPECTED_REQUESTS, MANIFEST_PATH, ""),
        ("chunks", "chunk_rows", EXPECTED_CHUNKS, len(inputs.chunks), len(inputs.chunks) == EXPECTED_CHUNKS, CHUNK_MANIFEST_PATH, ""),
        ("raw", "raw_request_records", EXPECTED_REQUESTS, len(request_records), len(request_records) == EXPECTED_REQUESTS, REQUEST_JSONL, ""),
        ("raw", "raw_response_records", EXPECTED_REQUESTS, len(response_records), len(response_records) == EXPECTED_REQUESTS, RESPONSE_JSONL, ""),
        ("raw", "runner_events", EXPECTED_REQUESTS * 2, len(event_records), len(event_records) == EXPECTED_REQUESTS * 2, EVENT_JSONL, ""),
        ("raw", "duplicate_request_ids", 0, len(request_records) - len(req_counts), len(request_records) == len(req_counts), REQUEST_JSONL, ""),
        ("raw", "duplicate_response_ids", 0, len(response_records) - len(resp_counts), len(response_records) == len(resp_counts), RESPONSE_JSONL, ""),
        ("state", "request_state_rows", EXPECTED_REQUESTS, len(request_state), len(request_state) == EXPECTED_REQUESTS, REQUEST_STATE_PATH, ""),
        ("state", "completed_requests", EXPECTED_REQUESTS, int(request_state["request_state"].eq("completed").sum()), int(request_state["request_state"].eq("completed").sum()) == EXPECTED_REQUESTS, REQUEST_STATE_PATH, ""),
        ("state", "ambiguous_requests", 0, int(request_state["request_state"].eq("ambiguous_interrupted").sum()), int(request_state["request_state"].eq("ambiguous_interrupted").sum()) == 0, REQUEST_STATE_PATH, ""),
        ("state", "duplicate_evidence_requests", 0, int(request_state["request_state"].eq("invalid_duplicate_evidence").sum()), int(request_state["request_state"].eq("invalid_duplicate_evidence").sum()) == 0, REQUEST_STATE_PATH, ""),
        ("execution", "execution_rows", EXPECTED_REQUESTS, len(execution), len(execution) == EXPECTED_REQUESTS, EXECUTION_PATH, ""),
        ("parse", "parse_rows", EXPECTED_REQUESTS, len(parse_results), len(parse_results) == EXPECTED_REQUESTS, PARSE_RESULTS_PATH, ""),
        ("rankings", "ranking_rows", EXPECTED_REQUESTS, len(rankings), len(rankings) == EXPECTED_REQUESTS, RANKINGS_PATH, ""),
        ("metrics", "cell_metric_rows", 12, len(metrics), len(metrics) == 12, CELL_METRICS_PATH, ""),
        ("execution", "retry_count", 0, decision["retry_violation_count"], decision["retry_violation_count"] == "0", DECISION_PATH, ""),
        ("execution", "request_order_gaps", 0, decision["request_order_gap_count"], decision["request_order_gap_count"] == "0", DECISION_PATH, ""),
        ("protected", "protected_inputs_unchanged", "true", bool_text(protected_unchanged), protected_unchanged, "hash_protected_inputs", ""),
        ("utility", "valid_only_not_final", "false", decision["valid_only_utility_used_for_final_performance"], decision["valid_only_utility_used_for_final_performance"] == "false", DECISION_PATH, ""),
    ]
    return frame([validation_row(idx, *check) for idx, check in enumerate(checks, start=1)], VALIDATION_COLUMNS)


def build_provenance(inputs: FullMatrixInputs, admin: Any, started_at: str, completed_at: str, initial_summary: Mapping[str, int], initial_request_records: int, initial_response_records: int, max_new_requests: int, sent: int) -> pd.DataFrame:
    row = {
        "execution_id": PHASE_ID,
        "manifest_path": MANIFEST_PATH.as_posix(),
        "manifest_sha256": file_sha256(Path.cwd() / MANIFEST_PATH),
        "chunk_manifest_path": CHUNK_MANIFEST_PATH.as_posix(),
        "chunk_manifest_sha256": file_sha256(Path.cwd() / CHUNK_MANIFEST_PATH),
        "preflight_decision_path": PREFLIGHT_DECISION_PATH.as_posix(),
        "preflight_decision_sha256": file_sha256(Path.cwd() / PREFLIGHT_DECISION_PATH),
        "protected_input_count": str(len(inputs.protected_hashes)),
        "ollama_server_version": admin.server_version,
        "qwen2_model_tag": EXPECTED_DIGESTS["Qwen2 7B Instruct"]["tag"],
        "qwen2_digest": EXPECTED_DIGESTS["Qwen2 7B Instruct"]["digest"],
        "mistral_model_tag": EXPECTED_DIGESTS["Mistral 7B Instruct"]["tag"],
        "mistral_digest": EXPECTED_DIGESTS["Mistral 7B Instruct"]["digest"],
        "expected_request_count": str(EXPECTED_REQUESTS),
        "initial_request_records": str(initial_request_records),
        "initial_response_records": str(initial_response_records),
        "initial_completed_requests": str(initial_summary.get("completed", 0)),
        "initial_pending_requests": str(initial_summary.get("not_started", 0)),
        "initial_ambiguous_requests": str(initial_summary.get("ambiguous_interrupted", 0)),
        "max_new_requests_this_invocation": str(max_new_requests),
        "new_requests_sent_this_invocation": str(sent),
        "execution_started_at_utc": started_at,
        "execution_completed_at_utc": completed_at,
        "provenance_status": "completed" if sent >= 0 else "blocked",
        "blocking_reason": "",
        "notes": f"ollama_start_attempted={bool_text(admin.start_attempted)}; started_process_id={admin.started_process_id}",
    }
    return frame([row], PROVENANCE_COLUMNS)


def build_all_result_tables(root: Path, inputs: FullMatrixInputs, admin: Any, started_at: str, completed_at: str, initial_summary: Mapping[str, int], initial_request_records: int, initial_response_records: int, max_new_requests: int, sent: int) -> dict[str, pd.DataFrame]:
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    event_records = read_jsonl(root / EVENT_JSONL)
    protected_unchanged = hash_protected_inputs(root) == dict(inputs.protected_hashes)
    execution = records_to_execution(inputs.manifest, response_records)
    parse_results, rankings = build_parse_and_rankings(inputs, response_records)
    request_state = build_request_state(inputs.manifest, request_records, response_records, parse_results, rankings)
    cell_metrics = build_cell_metrics(inputs.manifest, execution, parse_results, rankings)
    progress = build_progress(inputs, request_state, execution, parse_results)
    frames: dict[str, pd.DataFrame] = {
        "provenance": build_provenance(inputs, admin, started_at, completed_at, initial_summary, initial_request_records, initial_response_records, max_new_requests, sent),
        "request_state": request_state,
        "progress": progress,
        "execution": execution,
        "parse_results": parse_results,
        "rankings": rankings,
        "cell_metrics": cell_metrics,
    }
    frames["decision"] = build_decision(inputs, request_records, response_records, event_records, frames, protected_unchanged)
    frames["validation"] = build_validation(inputs, request_records, response_records, event_records, frames, protected_unchanged)
    validate_frames(frames)
    return frames


def validate_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    for name, (_path, columns) in OUTPUT_TABLES.items():
        if list(frames[name].columns) != columns:
            raise ValueError(f"{name} schema mismatch")
        if frames[name].empty:
            raise ValueError(f"{name} is empty")
    if len(frames["provenance"]) != 1 or len(frames["decision"]) != 1:
        raise ValueError("provenance/decision row count mismatch")
    for name in ["request_state", "execution", "parse_results", "rankings"]:
        if len(frames[name]) != EXPECTED_REQUESTS:
            raise ValueError(f"{name} row count mismatch")
    if len(frames["cell_metrics"]) != 12:
        raise ValueError("cell metric row count mismatch")
    rates = pd.to_numeric(frames["cell_metrics"]["strict_validity_rate"], errors="raise")
    if not rates.between(0, 1).all():
        raise ValueError("strict validity rate out of range")


def write_tables(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for name, (path, columns) in OUTPUT_TABLES.items():
        output = root / path
        output.parent.mkdir(parents=True, exist_ok=True)
        data = frames[name].copy()
        if list(data.columns) != columns:
            raise ValueError(f"{name} schema mismatch")
        data.to_csv(output, index=False, lineterminator="\n")


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
    cpu = platform.processor()
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.stdout.strip():
            gpu = "; ".join(line.strip() for line in completed.stdout.splitlines() if line.strip())
    except Exception:
        pass
    return {
        "executable": sys.executable,
        "python": sys.version.replace("\n", " "),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": cpu,
        "ram_bytes": ram,
        "gpu": gpu,
        "cwd": str(root),
    }


def markdown_table(data: pd.DataFrame, columns: Sequence[str]) -> str:
    if data.empty:
        return "_No rows._"
    subset = data.loc[:, list(columns)].astype(str)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in subset.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_report(root: Path, frames: Mapping[str, pd.DataFrame], commands: Sequence[str]) -> None:
    decision = frames["decision"].iloc[0]
    progress_global = frames["progress"][frames["progress"]["progress_scope"].eq("global")].iloc[0]
    env = environment_summary(root)
    metrics = frames["cell_metrics"]
    ranking_metrics = metrics[metrics["task_type"].eq("ranking")]
    reranking_metrics = metrics[metrics["task_type"].eq("reranking")]
    sections: list[tuple[str, str]] = [
        ("1. Scope and provenance", f"Phase: {PHASE_ID}. This report is generated from full-matrix raw JSONL and derived CSV files."),
        ("2. Environment", "\n".join(f"- {key}: `{value}`" for key, value in env.items())),
        ("3. Protected-input policy", f"Protected inputs recorded: {decision['protected_inputs_unchanged']}."),
        ("4. Full cohort", f"Prompt-compatible rows: {EXPECTED_PROMPT_COMPATIBLE_ROWS}; ranking rows: {EXPECTED_TASK_ROWS}; reranking rows: {EXPECTED_TASK_ROWS}."),
        ("5. Manifest", f"Manifest rows: {decision['manifest_rows']}; chunks: {decision['chunk_rows']}; expected requests: {decision['expected_request_count']}."),
        ("6. Execution progress", markdown_table(frames["progress"], ["progress_scope", "chunk_id", "planned_request_count", "completed_requests", "pending_requests", "progress_status", "next_pending_request_order"])),
        ("7. Raw retention", f"Requests: {decision['unique_request_record_count']}; responses/errors: {decision['unique_response_record_count']}; events: {decision['runner_event_count']}."),
        ("8. Backend reliability", f"Backend error records: {decision['backend_error_record_count']}."),
        ("9. Strict format validity", markdown_table(metrics, ["model_family", "dataset", "task_type", "planned_rows", "completed_rows", "strict_valid_rows", "strict_validity_rate"])),
        ("10. Ranking diagnostics", markdown_table(ranking_metrics, ["model_family", "dataset", "strict_valid_rows", "valid_only_hr_at_5", "valid_only_ndcg_at_5"])),
        ("11. Reranking diagnostics", markdown_table(reranking_metrics, ["model_family", "dataset", "strict_valid_rows", "valid_only_hr_at_5", "valid_only_ndcg_at_5"])),
        ("12. Utility scope", "HR@5 and NDCG@5 are valid-only execution diagnostics in this phase, not final scientific performance."),
        ("13. Validation", markdown_table(frames["validation"], ["validation_scope", "check_name", "expected_value", "observed_value", "status"])),
        ("14. Decision", f"Decision: `{decision['execution_decision']}`. Blocking reason: `{decision['blocking_reason']}`."),
        ("15. Commands", "\n".join(f"- `{command}`" for command in commands)),
        ("16. Next task", decision["recommended_single_next_task"]),
    ]
    lines = ["# Canonical Local LLM Full Matrix Execution", ""]
    for title, body in sections:
        lines.extend([f"## {title}", "", body, ""])
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def validate_output_files(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, (path, columns) in OUTPUT_TABLES.items():
        full = root / path
        if not full.exists():
            raise FileNotFoundError(full)
        data = pd.read_csv(full, dtype=str, keep_default_na=False)
        if list(data.columns) != columns:
            raise ValueError(f"{path} schema mismatch")
        if data.empty:
            raise ValueError(f"{path} is empty")
        hashes[path.as_posix()] = file_sha256(full)
    if not (root / REPORT_PATH).exists():
        raise FileNotFoundError(root / REPORT_PATH)
    hashes[REPORT_PATH.as_posix()] = file_sha256(root / REPORT_PATH)
    return hashes

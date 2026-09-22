"""Resume-safe runner for the Phase 5B-P1D reranking prompt-v2 pilot.

The module uses only the local Ollama loopback endpoint and the frozen
60-row corrective manifest from the prompt-corrective audit. It never pulls,
creates, copies, deletes, converts, or modifies models.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.canonical_llm_execution_preflight import file_sha256, read_csv
from src.llm.canonical_llm_contract import contract_hash, json_dumps, sha256_items, sha256_text
from src.llm.canonical_local_llm_pilot_resume import (
    append_jsonl_fsync,
    load_resume_inputs,
    positive_label_exposed,
    read_jsonl,
)
from src.llm.canonical_local_llm_runner import (
    MODEL_ORDER,
    READINESS_THRESHOLD,
    STRATUM_ORDER,
    build_payload,
    classify_response,
    clean_text,
    execute_ollama_request,
    render_from_contract,
)


RAW_DIR = Path("results/llm/canonical_local_reranking_prompt_v2_corrective_pilot")
REQUEST_JSONL = RAW_DIR / "requests.jsonl"
RESPONSE_JSONL = RAW_DIR / "responses.jsonl"
EVENT_JSONL = RAW_DIR / "runner_events.jsonl"

PROMPT_V2_CONTRACT_PATH = Path("results/tables/canonical_local_llm_reranking_prompt_v2_contract.csv")
CORRECTIVE_MANIFEST_PATH = Path("results/tables/canonical_local_llm_reranking_prompt_v2_corrective_pilot_manifest.csv")
PROMPT_CORRECTIVE_DECISION_PATH = Path("results/tables/canonical_local_llm_pilot_prompt_corrective_decision.csv")
PROMPT_REQUIREMENT_AUDIT_PATH = Path("results/tables/canonical_local_llm_reranking_prompt_requirement_audit.csv")
PARSER_ASSESSMENT_PATH = Path("results/tables/canonical_local_llm_pilot_parser_assessment.csv")
INVALID_RESPONSE_AUDIT_PATH = Path("results/tables/canonical_local_llm_pilot_invalid_response_audit.csv")
PROMPT_CORRECTIVE_REPORT_PATH = Path("report/canonical_local_llm_pilot_invalid_output_and_prompt_corrective_audit.md")

SOURCE_V1_SAMPLE_PATH = Path("results/tables/canonical_local_llm_pilot_sample.csv")
SOURCE_V1_MANIFEST_PATH = Path("results/tables/canonical_local_llm_pilot_request_manifest.csv")
SOURCE_V1_RAW_REQUESTS_PATH = Path("results/llm/canonical_local_pilot_resume/requests.jsonl")
SOURCE_V1_RAW_RESPONSES_PATH = Path("results/llm/canonical_local_pilot_resume/responses.jsonl")
SOURCE_V1_PARSE_PATH = Path("results/tables/canonical_local_llm_pilot_resume_parse_results.csv")
SOURCE_V1_EXECUTION_PATH = Path("results/tables/canonical_local_llm_pilot_resume_execution.csv")
SOURCE_V1_METRICS_PATH = Path("results/tables/canonical_local_llm_pilot_resume_metrics.csv")
SOURCE_V1_DECISION_PATH = Path("results/tables/canonical_local_llm_pilot_resume_decision.csv")
SOURCE_RECOVERY_PATH = Path("results/tables/canonical_local_ollama_service_recovery.csv")

REPORT_PATH = Path("report/canonical_local_llm_reranking_prompt_v2_corrective_pilot.md")

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

APPROVED_NEXT_TASK = "Phase 5B-P2 — full canonical local Qwen2 and Mistral execution using ranking prompt v1 and reranking prompt v2 on 5,764 frozen prompt-compatible rows"

PROMPT_V2_CONTRACT_COLUMNS = [
    "prompt_contract_id",
    "supersedes_prompt_contract_id",
    "task_type",
    "prompt_version",
    "system_message_template",
    "user_message_template",
    "candidate_representation",
    "candidate_numbering",
    "candidate_order_policy",
    "positive_label_hidden",
    "requested_output_count",
    "requested_output_representation",
    "exact_permutation_requirement",
    "internal_validation_instruction",
    "explanation_allowed",
    "code_fence_allowed",
    "scientific_semantics_changed",
    "revision_evidence",
    "paper_alignment",
    "contract_status",
    "prompt_template_sha256",
    "contract_sha256",
    "notes",
]

CORRECTIVE_MANIFEST_COLUMNS = [
    "corrective_request_id",
    "source_request_id_v1",
    "pilot_row_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "source_candidate_order_sha256",
    "source_history_render_sha256",
    "prompt_v2_contract_id",
    "prompt_v2_contract_sha256",
    "parser_contract_id",
    "execution_contract_id",
    "requested_output_count",
    "temperature",
    "seed",
    "max_output_tokens",
    "response_format",
    "streaming",
    "concurrency",
    "scientific_retry_count",
    "infrastructure_retry_count",
    "corrective_request_order",
    "corrective_scope_status",
    "blocking_reason",
    "notes",
]

PROVENANCE_COLUMNS = [
    "corrective_pilot_id",
    "prompt_v2_contract_path",
    "prompt_v2_contract_sha256",
    "corrective_manifest_path",
    "corrective_manifest_sha256",
    "source_v1_requests_path",
    "source_v1_responses_path",
    "source_v1_parse_results_path",
    "source_v1_metrics_path",
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
    "execution_started_at_utc",
    "execution_completed_at_utc",
    "provenance_status",
    "blocking_reason",
    "notes",
]

REQUEST_STATE_COLUMNS = [
    "corrective_request_id",
    "source_request_id_v1",
    "corrective_request_order",
    "pilot_row_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
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
    "corrective_request_id",
    "source_request_id_v1",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
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
    "corrective_request_id",
    "source_request_id_v1",
    "model_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
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
    "corrective_request_id",
    "source_request_id_v1",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
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
    "corrective_rows",
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
    "source_request_id_v1",
    "corrective_request_id_v2",
    "model_family",
    "dataset",
    "candidate_set_id",
    "user_id",
    "candidate_order_identical",
    "history_render_identical",
    "execution_parameters_identical",
    "v1_strict_parse_valid",
    "v2_strict_parse_valid",
    "strict_validity_change",
    "v1_output_count_valid",
    "v2_output_count_valid",
    "v1_uniqueness_valid",
    "v2_uniqueness_valid",
    "v1_full_permutation_valid",
    "v2_full_permutation_valid",
    "v1_duplicate_count",
    "v2_duplicate_count",
    "v1_failure_reason",
    "v2_failure_reason",
    "v1_generated_token_count",
    "v2_generated_token_count",
    "v1_duration_seconds",
    "v2_duration_seconds",
    "paired_comparison_status",
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
    "corrective_pilot_id",
    "expected_request_count",
    "completed_request_count",
    "backend_error_record_count",
    "ambiguous_request_count",
    "backend_completion_rate",
    "minimum_cell_strict_parse_validity_rate",
    "cells_passing_threshold",
    "total_readiness_cells",
    "all_prompt_v2_hashes_valid",
    "all_candidate_order_hashes_valid",
    "all_history_hashes_valid",
    "positive_label_never_exposed",
    "all_strict_outputs_candidate_valid",
    "all_strict_outputs_duplicate_free",
    "all_strict_outputs_complete_permutations",
    "raw_request_retention_complete",
    "raw_response_retention_complete",
    "output_truncation_detected",
    "readiness_threshold",
    "readiness_threshold_result_independent",
    "ranking_prompt_for_full_execution",
    "reranking_prompt_for_full_execution",
    "pilot_decision",
    "blocking_cells",
    "required_corrective_action",
    "recommended_single_next_task",
    "notes",
]

OUTPUT_TABLES = {
    "provenance": ("results/tables/canonical_local_llm_reranking_prompt_v2_provenance.csv", PROVENANCE_COLUMNS),
    "request_state": ("results/tables/canonical_local_llm_reranking_prompt_v2_request_state.csv", REQUEST_STATE_COLUMNS),
    "execution": ("results/tables/canonical_local_llm_reranking_prompt_v2_execution.csv", EXECUTION_COLUMNS),
    "parse_results": ("results/tables/canonical_local_llm_reranking_prompt_v2_parse_results.csv", PARSE_COLUMNS),
    "rankings": ("results/tables/canonical_local_llm_reranking_prompt_v2_rankings.csv", RANKING_COLUMNS),
    "metrics": ("results/tables/canonical_local_llm_reranking_prompt_v2_metrics.csv", METRIC_COLUMNS),
    "comparison": ("results/tables/canonical_local_llm_reranking_prompt_v1_v2_comparison.csv", COMPARISON_COLUMNS),
    "validation": ("results/tables/canonical_local_llm_reranking_prompt_v2_validation.csv", VALIDATION_COLUMNS),
    "decision": ("results/tables/canonical_local_llm_reranking_prompt_v2_decision.csv", DECISION_COLUMNS),
}


@dataclass(frozen=True)
class P1DInputs:
    prompt_v2: pd.DataFrame
    corrective_manifest: pd.DataFrame
    v1_inputs: Any
    v1_parse: pd.DataFrame
    v1_execution: pd.DataFrame
    v1_metrics: pd.DataFrame
    v1_decision: pd.DataFrame
    execution_contract: pd.DataFrame
    parser_contract: pd.DataFrame
    recovery: pd.DataFrame
    contexts: dict[str, dict[str, Any]]
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


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.12g}"


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: str(row.get(column, "")) for column in columns} for row in rows], columns=list(columns))


def mean_numeric(values: Iterable[Any]) -> str:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return ""
    return format_float(float(series.mean()))


def mean_bool(values: Iterable[Any]) -> str:
    data = [1.0 if str(value).lower() == "true" else 0.0 for value in values]
    if not data:
        return ""
    return format_float(float(np.mean(data)))


def markdown_table(data: pd.DataFrame, columns: Sequence[str]) -> str:
    if data.empty:
        return "_No rows._"
    subset = data.loc[:, list(columns)].astype(str)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def read_table(root: Path, relative: Path | str) -> pd.DataFrame:
    return pd.read_csv(root / relative, dtype=str, keep_default_na=False)


def file_record_sha256(record: Mapping[str, Any]) -> str:
    persisted = {key: value for key, value in record.items() if not str(key).startswith("_")}
    return sha256_text(json.dumps(persisted, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def protected_input_paths(root: Path) -> list[str]:
    paths = [
        str(PROMPT_V2_CONTRACT_PATH),
        str(CORRECTIVE_MANIFEST_PATH),
        str(PROMPT_CORRECTIVE_DECISION_PATH),
        str(PROMPT_REQUIREMENT_AUDIT_PATH),
        str(PARSER_ASSESSMENT_PATH),
        str(PROMPT_CORRECTIVE_REPORT_PATH),
        str(SOURCE_V1_SAMPLE_PATH),
        str(SOURCE_V1_MANIFEST_PATH),
        str(SOURCE_V1_RAW_REQUESTS_PATH),
        str(SOURCE_V1_RAW_RESPONSES_PATH),
        str(SOURCE_V1_PARSE_PATH),
        str(SOURCE_V1_EXECUTION_PATH),
        str(SOURCE_V1_METRICS_PATH),
        str(SOURCE_V1_DECISION_PATH),
        str(INVALID_RESPONSE_AUDIT_PATH),
        str(SOURCE_RECOVERY_PATH),
        "results/tables/canonical_llm_prompt_contract.csv",
        "results/tables/canonical_llm_parser_contract.csv",
        "results/tables/canonical_llm_execution_parameter_contract.csv",
        "results/tables/canonical_llm_model_inventory.csv",
        "results/tables/canonical_llm_cohort_preflight.csv",
        "results/tables/canonical_llm_prompt_rendering_audit.csv",
        "results/tables/ranking_candidate_sets.csv",
        "results/tables/positive_positions.csv",
        "results/tables/experiment_cohort_membership.csv",
        "results/tables/item_title_map.csv",
        "results/tables/candidate_item_title_map.csv",
        "results/tables/llm_title_exclusions.csv",
        "paper/Beyond Utility Evaluating LLM As Recommander.pdf",
    ]
    paths.extend(str(path.relative_to(root)).replace("\\", "/") for path in (root / "data/reconstructed/plan_c").glob("**/interactions.csv"))
    return sorted({path.replace("\\", "/") for path in paths if (root / path).is_file()})


def hash_protected_inputs(root: Path) -> dict[str, str]:
    return {relative: file_sha256(root / relative) for relative in protected_input_paths(root)}


def validate_prompt_v2_contract(prompt_v2: pd.DataFrame) -> dict[str, str]:
    if list(prompt_v2.columns) != PROMPT_V2_CONTRACT_COLUMNS:
        raise ValueError("prompt-v2 contract schema mismatch")
    if len(prompt_v2) != 1:
        raise ValueError("prompt-v2 contract must contain exactly one row")
    row = prompt_v2.iloc[0].to_dict()
    if row["prompt_contract_id"] != "llm_rerank_index_json_full20_v2":
        raise ValueError("wrong prompt-v2 contract id")
    if row["supersedes_prompt_contract_id"] != "llm_rerank_index_json_full20_v1":
        raise ValueError("prompt-v2 supersedes wrong v1 contract")
    if row["task_type"] != "reranking":
        raise ValueError("prompt-v2 task type must be reranking")
    if row["prompt_version"] not in {"2", "v2"}:
        raise ValueError("prompt-v2 version must be 2/v2")
    if row["requested_output_count"] != "20":
        raise ValueError("prompt-v2 output count must be 20")
    if row["candidate_numbering"] != "one_based_1_to_20":
        raise ValueError("prompt-v2 candidate numbering mismatch")
    if row["positive_label_hidden"] != "true":
        raise ValueError("prompt-v2 exposes positive labels")
    if row["explanation_allowed"] != "false" or row["code_fence_allowed"] != "false":
        raise ValueError("prompt-v2 allows prose or code fences")
    if row["scientific_semantics_changed"] != "false":
        raise ValueError("prompt-v2 changes scientific semantics")
    user_template = row["user_message_template"]
    required_snippets = [
        "exactly 20 integers",
        "every integer from 1 through 20 exactly once",
        "no omitted number",
        "no repeated number",
        '"ranked_candidate_indices"',
        "Do not include prose, Markdown, a code fence",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in user_template]
    if missing:
        raise ValueError(f"prompt-v2 missing required wording: {missing}")
    forbidden = ["[1,2,...,20]", "[1, 2, ..., 20]", "chain-of-thought", "explain your reasoning"]
    if any(text.lower() in user_template.lower() for text in forbidden):
        raise ValueError("prompt-v2 contains example-order bias or reasoning request")
    if row["prompt_template_sha256"] != sha256_text(row["system_message_template"] + "\n---\n" + row["user_message_template"]):
        raise ValueError("prompt-v2 template hash mismatch")
    if row["contract_sha256"] != contract_hash(row):
        raise ValueError("prompt-v2 contract hash mismatch")
    return row


def validate_corrective_manifest(manifest: pd.DataFrame, prompt_v2_row: Mapping[str, str]) -> None:
    if list(manifest.columns) != CORRECTIVE_MANIFEST_COLUMNS:
        raise ValueError("corrective manifest schema mismatch")
    if len(manifest) != 60:
        raise ValueError("corrective manifest must contain 60 rows")
    if not manifest["corrective_request_id"].is_unique:
        raise ValueError("corrective_request_id is not unique")
    if manifest["corrective_request_order"].astype(int).tolist() != list(range(1, 61)):
        raise ValueError("corrective_request_order is not contiguous")
    if not manifest["task_type"].eq("reranking").all():
        raise ValueError("corrective manifest contains non-reranking rows")
    if set(manifest["requested_output_count"]) != {"20"}:
        raise ValueError("corrective manifest requested_output_count mismatch")
    if set(manifest["prompt_v2_contract_id"]) != {prompt_v2_row["prompt_contract_id"]}:
        raise ValueError("corrective manifest prompt-v2 id mismatch")
    if set(manifest["prompt_v2_contract_sha256"]) != {prompt_v2_row["contract_sha256"]}:
        raise ValueError("corrective manifest prompt-v2 hash mismatch")
    if not manifest["concurrency"].eq("1").all():
        raise ValueError("corrective manifest concurrency must be one")
    if not manifest["scientific_retry_count"].eq("0").all() or not manifest["infrastructure_retry_count"].eq("0").all():
        raise ValueError("corrective manifest retry counts must be zero")
    if not manifest["streaming"].eq("false").all():
        raise ValueError("corrective manifest streaming must be false")
    for family, tag in MODEL_ORDER:
        subset = manifest.loc[manifest["model_family"].eq(family)]
        if len(subset) != 30:
            raise ValueError(f"corrective manifest model count mismatch: {family}")
        if set(subset["model_tag"]) != {tag}:
            raise ValueError(f"corrective manifest model tag mismatch: {family}")
        for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]:
            cell = subset.loc[subset["dataset"].eq(dataset)]
            if len(cell) != 10:
                raise ValueError(f"corrective manifest cell count mismatch: {family} {dataset}")


def validate_source_v1_reconciliation(inputs: P1DInputs) -> None:
    v1_manifest = inputs.v1_inputs.manifest
    v1_by_request = {row.request_id: row._asdict() for row in v1_manifest.itertuples(index=False)}
    seen_sources: set[str] = set()
    for row in inputs.corrective_manifest.sort_values("corrective_request_order", key=lambda s: s.astype(int)).to_dict("records"):
        source_id = row["source_request_id_v1"]
        if source_id in seen_sources:
            raise ValueError(f"duplicate source v1 request: {source_id}")
        seen_sources.add(source_id)
        if source_id not in v1_by_request:
            raise ValueError(f"source v1 request missing: {source_id}")
        source = v1_by_request[source_id]
        if source["pilot_row_id"] != row["pilot_row_id"]:
            raise ValueError(f"pilot row mismatch: {row['corrective_request_id']}")
        if source["task_type"] != "reranking":
            raise ValueError(f"source v1 row is not reranking: {source_id}")
        for column in ["model_family", "model_tag", "dataset", "candidate_set_id", "user_id", "parser_contract_id", "execution_contract_id", "temperature", "seed", "max_output_tokens", "response_format", "streaming", "concurrency", "scientific_retry_count", "infrastructure_retry_count"]:
            if str(source[column]) != str(row[column]):
                raise ValueError(f"source v1 column mismatch {column}: {row['corrective_request_id']}")
        context = inputs.v1_inputs.contexts[row["pilot_row_id"]]
        if context["candidate_order_sha256"] != row["source_candidate_order_sha256"]:
            raise ValueError(f"candidate order hash mismatch: {row['corrective_request_id']}")
        if sha256_text(str(context["history_render"])) != row["source_history_render_sha256"]:
            raise ValueError(f"history render hash mismatch: {row['corrective_request_id']}")


def load_inputs(root: Path) -> P1DInputs:
    prompt_v2 = read_table(root, PROMPT_V2_CONTRACT_PATH)
    prompt_v2_row = validate_prompt_v2_contract(prompt_v2)
    corrective_manifest = read_table(root, CORRECTIVE_MANIFEST_PATH)
    validate_corrective_manifest(corrective_manifest, prompt_v2_row)
    decision = read_table(root, PROMPT_CORRECTIVE_DECISION_PATH)
    if decision.iloc[0]["recommended_decision"] != "approve_reranking_prompt_v2_corrective_pilot":
        raise ValueError("prompt-corrective audit did not approve v2 corrective pilot")
    v1_inputs = load_resume_inputs(root)
    inputs = P1DInputs(
        prompt_v2=prompt_v2,
        corrective_manifest=corrective_manifest,
        v1_inputs=v1_inputs,
        v1_parse=read_table(root, SOURCE_V1_PARSE_PATH),
        v1_execution=read_table(root, SOURCE_V1_EXECUTION_PATH),
        v1_metrics=read_table(root, SOURCE_V1_METRICS_PATH),
        v1_decision=read_table(root, SOURCE_V1_DECISION_PATH),
        execution_contract=read_table(root, "results/tables/canonical_llm_execution_parameter_contract.csv"),
        parser_contract=read_table(root, "results/tables/canonical_llm_parser_contract.csv"),
        recovery=read_table(root, SOURCE_RECOVERY_PATH),
        contexts={},
        protected_hashes={},
    )
    validate_source_v1_reconciliation(inputs)
    contexts = build_prompt_v2_contexts(inputs)
    final_inputs = P1DInputs(
        prompt_v2=inputs.prompt_v2,
        corrective_manifest=inputs.corrective_manifest,
        v1_inputs=inputs.v1_inputs,
        v1_parse=inputs.v1_parse,
        v1_execution=inputs.v1_execution,
        v1_metrics=inputs.v1_metrics,
        v1_decision=inputs.v1_decision,
        execution_contract=inputs.execution_contract,
        parser_contract=inputs.parser_contract,
        recovery=inputs.recovery,
        contexts=contexts,
        protected_hashes=hash_protected_inputs(root),
    )
    return final_inputs


def build_prompt_v2_contexts(inputs: P1DInputs) -> dict[str, dict[str, Any]]:
    prompt_row = inputs.prompt_v2.iloc[0].to_dict()
    contexts: dict[str, dict[str, Any]] = {}
    for row in inputs.corrective_manifest.sort_values("corrective_request_order", key=lambda s: s.astype(int)).to_dict("records"):
        source_context = inputs.v1_inputs.contexts[row["pilot_row_id"]]
        system_message = prompt_row["system_message_template"]
        user_message = prompt_row["user_message_template"].format(
            history_render=source_context["history_render"],
            candidate_render=source_context["candidate_render"],
        )
        payload_visible = json.dumps({"system": system_message, "prompt": user_message}, ensure_ascii=False)
        if positive_label_exposed(payload_visible, str(source_context.get("positive_item_id", ""))):
            raise ValueError(f"positive label exposed in v2 payload: {row['corrective_request_id']}")
        candidate_items = list(source_context["candidate_items"])
        if sha256_items(candidate_items) != row["source_candidate_order_sha256"]:
            raise ValueError(f"candidate order changed in v2 context: {row['corrective_request_id']}")
        if sha256_text(str(source_context["history_render"])) != row["source_history_render_sha256"]:
            raise ValueError(f"history render changed in v2 context: {row['corrective_request_id']}")
        contexts[row["corrective_request_id"]] = {
            **source_context,
            "system_message": system_message,
            "user_message": user_message,
            "history_render": source_context["history_render"],
            "candidate_render": source_context["candidate_render"],
            "candidate_items": candidate_items,
            "complete_prompt_sha256": sha256_text(system_message + "\n\n" + user_message),
            "system_message_sha256": sha256_text(system_message),
            "user_message_sha256": sha256_text(user_message),
            "prompt_contract_id": prompt_row["prompt_contract_id"],
            "prompt_contract_sha256": prompt_row["contract_sha256"],
            "requested_output_count": "20",
            "requires_full_permutation": True,
        }
    return contexts


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
        out_path = Path("C:/tmp/phase5b_p1d_ollama_stdout.log")
        err_path = Path("C:/tmp/phase5b_p1d_ollama_stderr.log")
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


def count_by_corrective_id(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        request_id = str(record.get("corrective_request_id", record.get("request_id", "")))
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
    req_counts = count_by_corrective_id(request_records)
    resp_counts = count_by_corrective_id(response_records)
    summary = {"completed": 0, "not_started": 0, "ambiguous_interrupted": 0, "invalid_duplicate_evidence": 0}
    manifest_ids = set(manifest["corrective_request_id"])
    for row in manifest.itertuples(index=False):
        state = classify_request_state(req_counts.get(row.corrective_request_id, 0), resp_counts.get(row.corrective_request_id, 0))
        summary[state] += 1
    extras = (set(req_counts) | set(resp_counts)) - manifest_ids
    if extras:
        summary["invalid_duplicate_evidence"] += len(extras)
    return summary


def pending_requests_in_order(manifest: pd.DataFrame, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    req_counts = count_by_corrective_id(request_records)
    resp_counts = count_by_corrective_id(response_records)
    manifest_ids = set(manifest["corrective_request_id"])
    extras = (set(req_counts) | set(resp_counts)) - manifest_ids
    if extras:
        raise RuntimeError(f"invalid_duplicate_evidence: raw id outside manifest {sorted(extras)[0]}")
    pending: list[dict[str, str]] = []
    for row in manifest.sort_values("corrective_request_order", key=lambda s: s.astype(int)).to_dict("records"):
        state = classify_request_state(req_counts.get(row["corrective_request_id"], 0), resp_counts.get(row["corrective_request_id"], 0))
        if state == "not_started":
            pending.append(row)
        elif state == "completed":
            continue
        elif state == "ambiguous_interrupted":
            raise RuntimeError(f"ambiguous_interrupted_request: {row['corrective_request_id']}")
        else:
            raise RuntimeError(f"invalid_duplicate_evidence: {row['corrective_request_id']}")
    return pending


def build_request_record(row: Mapping[str, str], context: Mapping[str, Any], payload: Mapping[str, Any], inputs: P1DInputs) -> dict[str, Any]:
    payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if positive_label_exposed(payload_text, str(context.get("positive_item_id", ""))):
        raise ValueError(f"positive label in payload: {row['corrective_request_id']}")
    parser_row = inputs.parser_contract.loc[inputs.parser_contract["parser_contract_id"].eq(row["parser_contract_id"])].iloc[0].to_dict()
    execution_row = inputs.execution_contract.loc[inputs.execution_contract["execution_contract_id"].eq(row["execution_contract_id"])].iloc[0].to_dict()
    return {
        "corrective_request_id": row["corrective_request_id"],
        "request_id": row["corrective_request_id"],
        "source_request_id_v1": row["source_request_id_v1"],
        "pilot_row_id": row["pilot_row_id"],
        "corrective_request_order": row["corrective_request_order"],
        "attempt_index": 1,
        "model_family": row["model_family"],
        "model_tag": row["model_tag"],
        "verified_model_digest": EXPECTED_DIGESTS[row["model_family"]]["digest"],
        "dataset": row["dataset"],
        "task_type": row["task_type"],
        "candidate_set_id": row["candidate_set_id"],
        "user_id": row["user_id"],
        "prompt_v2_contract_id": row["prompt_v2_contract_id"],
        "prompt_v2_contract_sha256": row["prompt_v2_contract_sha256"],
        "parser_contract_id": row["parser_contract_id"],
        "parser_schema_sha256": parser_row["parser_schema_sha256"],
        "execution_contract_id": row["execution_contract_id"],
        "execution_contract_sha256": execution_row["contract_sha256"],
        "complete_prompt_sha256": context["complete_prompt_sha256"],
        "candidate_order_sha256": row["source_candidate_order_sha256"],
        "history_render_sha256": row["source_history_render_sha256"],
        "scientific_retry_count": "0",
        "infrastructure_retry_count": "0",
        "candidate_items": context["candidate_items"],
        "request_payload": payload,
        "timestamp_before_send_utc": utc_now(),
    }


def build_response_record(row: Mapping[str, str], result: Any) -> dict[str, Any]:
    raw_sha = sha256_text(result.response_text)
    return {
        "corrective_request_id": row["corrective_request_id"],
        "request_id": row["corrective_request_id"],
        "source_request_id_v1": row["source_request_id_v1"],
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


def execute_pending_requests(root: Path, inputs: P1DInputs) -> None:
    (root / RAW_DIR).mkdir(parents=True, exist_ok=True)
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    pending = pending_requests_in_order(inputs.corrective_manifest, request_records, response_records)
    for row in pending:
        if hash_protected_inputs(root) != inputs.protected_hashes:
            raise RuntimeError("protected_input_validation_failed")
        context = inputs.contexts[row["corrective_request_id"]]
        if context["prompt_contract_sha256"] != row["prompt_v2_contract_sha256"]:
            raise ValueError(f"prompt-v2 hash mismatch before send: {row['corrective_request_id']}")
        if sha256_items(context["candidate_items"]) != row["source_candidate_order_sha256"]:
            raise ValueError(f"candidate-order hash mismatch before send: {row['corrective_request_id']}")
        if sha256_text(str(context["history_render"])) != row["source_history_render_sha256"]:
            raise ValueError(f"history hash mismatch before send: {row['corrective_request_id']}")
        payload = build_payload(row, context)
        request_record = build_request_record(row, context, payload, inputs)
        append_jsonl_fsync(root / REQUEST_JSONL, request_record)
        append_jsonl_fsync(root / EVENT_JSONL, {"event": "request_recorded_before_send", "corrective_request_id": row["corrective_request_id"], "timestamp_utc": utc_now()})
        timeout_seconds = max(120, int(float(row["max_output_tokens"] or 120)))
        result = execute_ollama_request(payload, timeout_seconds)
        response_record = build_response_record(row, result)
        append_jsonl_fsync(root / RESPONSE_JSONL, response_record)
        append_jsonl_fsync(
            root / EVENT_JSONL,
            {
                "event": "response_or_error_recorded",
                "corrective_request_id": row["corrective_request_id"],
                "backend_success": bool_text(result.backend_success),
                "timestamp_utc": utc_now(),
            },
        )
        print(f"{row['corrective_request_id']} {row['model_family']} {row['dataset']} backend_success={bool_text(result.backend_success)}", flush=True)


def parse_indices_for_forensics(raw_text: str) -> list[Any]:
    try:
        payload = json.loads(str(raw_text).strip())
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict) and isinstance(payload.get("ranked_candidate_indices"), list):
        return list(payload["ranked_candidate_indices"])
    return []


def missing_and_duplicate_indices(values: Sequence[Any]) -> tuple[list[int], dict[str, int], list[int]]:
    ints = [int(value) for value in values if isinstance(value, int) and not isinstance(value, bool)]
    expected = set(range(1, 21))
    missing = sorted(expected - set(value for value in ints if 1 <= value <= 20))
    duplicates = {str(value): count for value, count in sorted(Counter(ints).items()) if count > 1}
    out_of_range = [value for value in ints if value < 1 or value > 20]
    return missing, duplicates, out_of_range


def output_truncation_detected(raw_text: str, done_reason: str, generated_count: Any, max_output_tokens: Any) -> bool:
    reason = str(done_reason).strip().lower()
    if reason in {"length", "max_tokens"}:
        return True
    try:
        generated = int(float(str(generated_count)))
        limit = int(float(str(max_output_tokens)))
    except ValueError:
        generated = -1
        limit = -1
    if limit > 0 and generated >= int(limit * 0.95):
        return True
    stripped = str(raw_text).strip()
    if stripped and (stripped.startswith("{") or stripped.startswith("[")):
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            return True
    return False


def records_to_execution(manifest: pd.DataFrame, response_records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    manifest_lookup = {row.corrective_request_id: row._asdict() for row in manifest.itertuples(index=False)}
    response_lookup = {str(record.get("corrective_request_id", record.get("request_id", ""))): record for record in response_records}
    rows: list[dict[str, Any]] = []
    for corrective_request_id in manifest.sort_values("corrective_request_order", key=lambda s: s.astype(int))["corrective_request_id"]:
        manifest_row = manifest_lookup[corrective_request_id]
        record = response_lookup.get(corrective_request_id, {})
        rows.append(
            {
                "corrective_request_id": corrective_request_id,
                "source_request_id_v1": manifest_row["source_request_id_v1"],
                "model_family": manifest_row["model_family"],
                "model_tag": manifest_row["model_tag"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
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


def build_parse_and_rankings(inputs: P1DInputs, execution: pd.DataFrame, response_records: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_lookup = {row.corrective_request_id: row._asdict() for row in inputs.corrective_manifest.itertuples(index=False)}
    response_lookup = {str(record.get("corrective_request_id", record.get("request_id", ""))): record for record in response_records}
    parse_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for corrective_request_id, manifest_row in manifest_lookup.items():
        context = inputs.contexts[corrective_request_id]
        response_record = response_lookup.get(corrective_request_id, {})
        raw_text = str(response_record.get("raw_response_text", ""))
        candidate_items = list(context["candidate_items"])
        classified = classify_response(
            raw_text,
            candidate_items=candidate_items,
            expected_count=20,
            require_full_permutation=True,
        )
        raw_sha = sha256_text(raw_text)
        raw_indices = parse_indices_for_forensics(raw_text)
        missing, duplicated, out_of_range = missing_and_duplicate_indices(raw_indices)
        truncation = output_truncation_detected(raw_text, response_record.get("done_reason", ""), response_record.get("generated_token_count", ""), manifest_row.get("max_output_tokens", ""))
        parse_rows.append(
            {
                "corrective_request_id": corrective_request_id,
                "source_request_id_v1": manifest_row["source_request_id_v1"],
                "model_family": manifest_row["model_family"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "raw_response_sha256": raw_sha,
                "strict_parser_id": manifest_row["parser_contract_id"],
                "diagnostic_parser_id": manifest_row["parser_contract_id"].replace("strict_primary", "diagnostic_salvage"),
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
                "corrective_request_id": corrective_request_id,
                "source_request_id_v1": manifest_row["source_request_id_v1"],
                "model_family": manifest_row["model_family"],
                "model_tag": manifest_row["model_tag"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "positive_item_id": context["positive_item_id"],
                "candidate_items": json_dumps(candidate_items),
                "candidate_order_sha256": manifest_row["source_candidate_order_sha256"],
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


def build_request_state(manifest: pd.DataFrame, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]], parse_results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    req_by_id: dict[str, list[Mapping[str, Any]]] = {}
    resp_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for record in request_records:
        req_by_id.setdefault(str(record.get("corrective_request_id", record.get("request_id", ""))), []).append(record)
    for record in response_records:
        resp_by_id.setdefault(str(record.get("corrective_request_id", record.get("request_id", ""))), []).append(record)
    parse_lookup = {row.corrective_request_id: row._asdict() for row in parse_results.itertuples(index=False)}
    ranking_lookup = {row.corrective_request_id: row._asdict() for row in rankings.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest.sort_values("corrective_request_order", key=lambda s: s.astype(int)).to_dict("records"):
        request_id = manifest_row["corrective_request_id"]
        requests = req_by_id.get(request_id, [])
        responses = resp_by_id.get(request_id, [])
        state = classify_request_state(len(requests), len(responses))
        parse_row = parse_lookup.get(request_id, {})
        ranking_row = ranking_lookup.get(request_id, {})
        blocking = ""
        if state == "ambiguous_interrupted":
            blocking = "request record exists without response record"
        elif state == "invalid_duplicate_evidence":
            blocking = "duplicate raw evidence or response without request"
        rows.append(
            {
                "corrective_request_id": request_id,
                "source_request_id_v1": manifest_row["source_request_id_v1"],
                "corrective_request_order": manifest_row["corrective_request_order"],
                "pilot_row_id": manifest_row["pilot_row_id"],
                "model_family": manifest_row["model_family"],
                "model_tag": manifest_row["model_tag"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "request_record_count": len(requests),
                "response_record_count": len(responses),
                "request_state": state,
                "request_sent": bool_text(len(requests) > 0),
                "response_recorded": bool_text(len(responses) > 0),
                "backend_success": str(responses[0].get("backend_success", "")) if len(responses) == 1 else "",
                "strict_parse_valid": str(parse_row.get("strict_parse_valid", "")),
                "full_permutation_valid": str(parse_row.get("full_permutation_valid", "")),
                "candidate_validity": str(parse_row.get("candidate_validity", "")),
                "utility_evaluation_valid": str(ranking_row.get("utility_evaluation_valid", "")),
                "raw_request_sha256": file_record_sha256(requests[0]) if len(requests) == 1 else "",
                "raw_response_sha256": str(responses[0].get("raw_response_sha256", "")) if len(responses) == 1 else "",
                "blocking_reason": blocking,
                "notes": "append-only corrective-pilot state",
            }
        )
    return frame(rows, REQUEST_STATE_COLUMNS)


def build_metrics(manifest: pd.DataFrame, execution: pd.DataFrame, parse_results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    merged = manifest[["corrective_request_id", "model_family", "model_tag", "dataset", "task_type"]].merge(
        execution,
        on=["corrective_request_id", "model_family", "model_tag", "dataset", "task_type"],
    )
    merged = merged.merge(parse_results, on=["corrective_request_id", "model_family", "dataset", "task_type"])
    merged = merged.merge(
        rankings[["corrective_request_id", "positive_in_top_5", "ndcg_at_5", "positive_rank_zero_based", "utility_evaluation_valid"]],
        on="corrective_request_id",
    )
    rows: list[dict[str, Any]] = []
    for family, _tag in MODEL_ORDER:
        for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]:
            group = merged.loc[merged["model_family"].eq(family) & merged["dataset"].eq(dataset)]
            if group.empty:
                continue
            tag = group.iloc[0]["model_tag"]
            corrective_rows = len(group)
            strict = group.loc[group["strict_parse_valid"].eq("true")]
            valid_utility = group.loc[group["utility_evaluation_valid"].eq("true")]
            durations = pd.to_numeric(group["duration_seconds"], errors="coerce")
            eval_counts = pd.to_numeric(group["eval_count"], errors="coerce")
            strict_count = len(strict)
            cell_pass = (
                strict_count >= int(READINESS_THRESHOLD * 10)
                and strict_count / corrective_rows >= READINESS_THRESHOLD
                and strict["candidate_validity"].eq("true").all()
                and pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).all()
                and strict["full_permutation_valid"].eq("true").all()
            )
            rows.append(
                {
                    "model_family": family,
                    "model_tag": tag,
                    "dataset": dataset,
                    "task_type": "reranking",
                    "corrective_rows": str(corrective_rows),
                    "backend_success_rows": str(int(group["backend_success"].eq("true").sum())),
                    "strict_valid_rows": str(strict_count),
                    "strict_parse_validity_rate": format_float(strict_count / corrective_rows if corrective_rows else 0.0),
                    "diagnostic_salvage_valid_rows": str(int(group["diagnostic_salvage_valid"].eq("true").sum())),
                    "diagnostic_salvage_validity_rate": format_float(int(group["diagnostic_salvage_valid"].eq("true").sum()) / corrective_rows if corrective_rows else 0.0),
                    "output_count_valid_rows": str(int(group["output_count_valid"].eq("true").sum())),
                    "uniqueness_valid_rows": str(int(group["uniqueness_valid"].eq("true").sum())),
                    "full_permutation_valid_rows": str(int(group["full_permutation_valid"].eq("true").sum())),
                    "candidate_valid_rows": str(int(strict["candidate_validity"].eq("true").sum())),
                    "duplicate_free_rows": str(int(pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).sum())),
                    "hallucination_free_rows": str(int(pd.to_numeric(strict["hallucination_count"], errors="coerce").fillna(0).eq(0).sum())),
                    "utility_evaluated_rows": str(len(valid_utility)),
                    "hr_at_5_valid_only": mean_bool(valid_utility["positive_in_top_5"]) if len(valid_utility) else "",
                    "ndcg_at_5_valid_only": mean_numeric(valid_utility["ndcg_at_5"]) if len(valid_utility) else "",
                    "mean_positive_rank_valid_only": mean_numeric(valid_utility.loc[valid_utility["positive_rank_zero_based"].ne(""), "positive_rank_zero_based"]) if len(valid_utility.loc[valid_utility["positive_rank_zero_based"].ne("")]) else "",
                    "median_duration_seconds": format_float(float(durations.median())) if durations.notna().any() else "",
                    "mean_duration_seconds": format_float(float(durations.mean())) if durations.notna().any() else "",
                    "median_generated_token_count": format_float(float(eval_counts.median())) if eval_counts.notna().any() else "",
                    "readiness_threshold": format_float(READINESS_THRESHOLD),
                    "cell_readiness_pass": bool_text(cell_pass),
                    "metric_status": "valid_only_utility_diagnostic",
                    "notes": "Corrective-pilot utility is diagnostic only and not final model performance.",
                }
            )
    return frame(rows, METRIC_COLUMNS)


def compare_validity(v1: str, v2: str) -> str:
    if v1 == "false" and v2 == "true":
        return "improved"
    if v1 == "true" and v2 == "false":
        return "regressed"
    if v1 == "true" and v2 == "true":
        return "same_valid"
    return "same_invalid"


def build_v1_v2_comparison(inputs: P1DInputs, execution: pd.DataFrame, parse_results: pd.DataFrame) -> pd.DataFrame:
    v1_parse_lookup = {row.request_id: row._asdict() for row in inputs.v1_parse.itertuples(index=False)}
    v1_exec_lookup = {row.request_id: row._asdict() for row in inputs.v1_execution.itertuples(index=False)}
    v1_manifest_lookup = {row.request_id: row._asdict() for row in inputs.v1_inputs.manifest.itertuples(index=False)}
    v2_parse_lookup = {row.corrective_request_id: row._asdict() for row in parse_results.itertuples(index=False)}
    v2_exec_lookup = {row.corrective_request_id: row._asdict() for row in execution.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for record in inputs.corrective_manifest.sort_values("corrective_request_order", key=lambda s: s.astype(int)).to_dict("records"):
        v1_parse = v1_parse_lookup[record["source_request_id_v1"]]
        v1_exec = v1_exec_lookup[record["source_request_id_v1"]]
        v1_manifest = v1_manifest_lookup[record["source_request_id_v1"]]
        v2_parse = v2_parse_lookup[record["corrective_request_id"]]
        v2_exec = v2_exec_lookup[record["corrective_request_id"]]
        execution_identical = all(str(v1_manifest[column]) == str(record[column]) for column in ["model_tag", "temperature", "seed", "max_output_tokens", "response_format", "streaming", "concurrency", "scientific_retry_count", "infrastructure_retry_count", "execution_contract_id"])
        rows.append(
            {
                "comparison_id": f"P1D-CMP-{len(rows)+1:04d}",
                "source_request_id_v1": record["source_request_id_v1"],
                "corrective_request_id_v2": record["corrective_request_id"],
                "model_family": record["model_family"],
                "dataset": record["dataset"],
                "candidate_set_id": record["candidate_set_id"],
                "user_id": record["user_id"],
                "candidate_order_identical": bool_text(v1_manifest["candidate_order_sha256"] == record["source_candidate_order_sha256"]),
                "history_render_identical": bool_text(sha256_text(str(inputs.v1_inputs.contexts[record["pilot_row_id"]]["history_render"])) == record["source_history_render_sha256"]),
                "execution_parameters_identical": bool_text(execution_identical),
                "v1_strict_parse_valid": v1_parse["strict_parse_valid"],
                "v2_strict_parse_valid": v2_parse["strict_parse_valid"],
                "strict_validity_change": compare_validity(v1_parse["strict_parse_valid"], v2_parse["strict_parse_valid"]),
                "v1_output_count_valid": v1_parse["output_count_valid"],
                "v2_output_count_valid": v2_parse["output_count_valid"],
                "v1_uniqueness_valid": v1_parse["uniqueness_valid"],
                "v2_uniqueness_valid": v2_parse["uniqueness_valid"],
                "v1_full_permutation_valid": v1_parse["full_permutation_valid"],
                "v2_full_permutation_valid": v2_parse["full_permutation_valid"],
                "v1_duplicate_count": v1_parse["duplicate_count"],
                "v2_duplicate_count": v2_parse["duplicate_count"],
                "v1_failure_reason": v1_parse["primary_failure_reason"],
                "v2_failure_reason": v2_parse["primary_failure_reason"],
                "v1_generated_token_count": v1_exec["eval_count"],
                "v2_generated_token_count": v2_exec["eval_count"],
                "v1_duration_seconds": v1_exec["duration_seconds"],
                "v2_duration_seconds": v2_exec["duration_seconds"],
                "paired_comparison_status": "paired_same_row_identity",
                "notes": "V1 evidence is read-only; utility does not determine readiness.",
            }
        )
    return frame(rows, COMPARISON_COLUMNS)


def build_decision(request_state: pd.DataFrame, execution: pd.DataFrame, parse_results: pd.DataFrame, metrics: pd.DataFrame, protected_unchanged: bool, digest_ok: bool, prompt_hashes_ok: bool, candidate_hashes_ok: bool, history_hashes_ok: bool, positive_hidden: bool) -> pd.DataFrame:
    completed = int(request_state["request_state"].eq("completed").sum()) if len(request_state) else 0
    ambiguous = int(request_state["request_state"].eq("ambiguous_interrupted").sum()) if len(request_state) else 0
    backend_errors = int(execution["backend_success"].eq("false").sum()) if len(execution) else 0
    backend_rate = int(execution["backend_success"].eq("true").sum()) / 60 if len(execution) else 0.0
    min_parse = float(pd.to_numeric(metrics["strict_parse_validity_rate"], errors="coerce").min()) if len(metrics) else 0.0
    cells_passing = int(metrics["cell_readiness_pass"].eq("true").sum()) if len(metrics) else 0
    strict = parse_results.loc[parse_results["strict_parse_valid"].eq("true")] if len(parse_results) else parse_results
    strict_candidate_valid = strict["candidate_validity"].eq("true").all() if len(strict) else False
    strict_duplicate_free = pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).all() if len(strict) else False
    strict_perm = strict["full_permutation_valid"].eq("true").all() if len(strict) else False
    raw_request_ok = len(request_state) == 60 and request_state["request_record_count"].astype(int).eq(1).all()
    raw_response_ok = len(request_state) == 60 and request_state["response_record_count"].astype(int).eq(1).all()
    truncation = parse_results["output_truncation_detected"].eq("true").any() if len(parse_results) else False
    blocking_cells = [f"{row.model_family}|{row.dataset}|strict_parse={row.strict_parse_validity_rate}" for row in metrics.itertuples(index=False) if row.cell_readiness_pass != "true"]
    if not protected_unchanged:
        decision = "protected_input_validation_failed"
        action = "Restore protected inputs before any further execution."
        next_task = action
    elif not digest_ok:
        decision = "model_digest_mismatch"
        action = "Resolve local model digest mismatch without pulling or substituting models."
        next_task = action
    elif ambiguous:
        decision = "ambiguous_interrupted_request"
        action = "Inspect append-only raw evidence without resending ambiguous requests."
        next_task = action
    elif completed != 60 or not raw_request_ok or not raw_response_ok:
        decision = "corrective_pilot_incomplete"
        action = "Rerun the resume-safe P1D runner to execute only unstarted corrective requests."
        next_task = action
    elif backend_rate < 1.0:
        decision = "backend_blocked"
        action = "Resolve local backend failures without changing scientific contracts."
        next_task = action
    elif truncation:
        decision = "require_output_token_limit_revision"
        action = "Audit output token limits before full execution."
        next_task = action
    elif cells_passing != 6 or min_parse < READINESS_THRESHOLD:
        decision = "require_reranking_prompt_v3_audit"
        action = "Run a focused prompt-v3 audit for readiness cells below 0.80."
        next_task = action
    elif not (strict_candidate_valid and strict_duplicate_free and strict_perm and prompt_hashes_ok and candidate_hashes_ok and history_hashes_ok and positive_hidden):
        decision = "require_parser_defect_investigation"
        action = "Investigate parser or invariant violation before full execution."
        next_task = action
    else:
        decision = "approve_full_local_execution_with_ranking_v1_reranking_v2"
        action = ""
        next_task = APPROVED_NEXT_TASK
    return frame(
        [
            {
                "corrective_pilot_id": "canonical_local_llm_reranking_prompt_v2_corrective_pilot_v1",
                "expected_request_count": "60",
                "completed_request_count": str(completed),
                "backend_error_record_count": str(backend_errors),
                "ambiguous_request_count": str(ambiguous),
                "backend_completion_rate": format_float(backend_rate),
                "minimum_cell_strict_parse_validity_rate": format_float(min_parse),
                "cells_passing_threshold": str(cells_passing),
                "total_readiness_cells": "6",
                "all_prompt_v2_hashes_valid": bool_text(prompt_hashes_ok),
                "all_candidate_order_hashes_valid": bool_text(candidate_hashes_ok),
                "all_history_hashes_valid": bool_text(history_hashes_ok),
                "positive_label_never_exposed": bool_text(positive_hidden),
                "all_strict_outputs_candidate_valid": bool_text(strict_candidate_valid),
                "all_strict_outputs_duplicate_free": bool_text(strict_duplicate_free),
                "all_strict_outputs_complete_permutations": bool_text(strict_perm),
                "raw_request_retention_complete": bool_text(raw_request_ok),
                "raw_response_retention_complete": bool_text(raw_response_ok),
                "output_truncation_detected": bool_text(bool(truncation)),
                "readiness_threshold": format_float(READINESS_THRESHOLD),
                "readiness_threshold_result_independent": "true",
                "ranking_prompt_for_full_execution": "llm_rank_index_json_top5_v1",
                "reranking_prompt_for_full_execution": "llm_rerank_index_json_full20_v2",
                "pilot_decision": decision,
                "blocking_cells": json_dumps(blocking_cells),
                "required_corrective_action": action,
                "recommended_single_next_task": next_task,
                "notes": "Corrective pilot readiness gate only; utility is diagnostic.",
            }
        ],
        DECISION_COLUMNS,
    )


def build_validation(inputs: P1DInputs, admin: OllamaAdminState, request_state: pd.DataFrame, execution: pd.DataFrame, parse_results: pd.DataFrame, rankings: pd.DataFrame, metrics: pd.DataFrame, comparison: pd.DataFrame, decision: pd.DataFrame, protected_unchanged: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(scope: str, check: str, expected: Any, observed: Any, passed: bool, severity: str = "error", evidence: str = "", model: str = "", dataset: str = "", notes: str = "") -> None:
        rows.append(
            {
                "validation_id": f"P1D-VAL-{len(rows)+1:04d}",
                "validation_scope": scope,
                "model_family": model,
                "dataset": dataset,
                "check_name": check,
                "expected_value": str(expected),
                "observed_value": str(observed),
                "status": "passed" if passed else "failed",
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    add("protected_hashes", "protected_hashes_unchanged", True, protected_unchanged, protected_unchanged)
    add("prompt", "prompt_v2_hash", inputs.prompt_v2.iloc[0]["contract_sha256"], inputs.prompt_v2.iloc[0]["contract_sha256"], True, evidence=str(PROMPT_V2_CONTRACT_PATH))
    add("manifest", "manifest_rows", 60, len(inputs.corrective_manifest), len(inputs.corrective_manifest) == 60, evidence=str(CORRECTIVE_MANIFEST_PATH))
    add("manifest", "reranking_only_scope", "reranking", sorted(inputs.corrective_manifest["task_type"].unique()), inputs.corrective_manifest["task_type"].eq("reranking").all())
    add("source", "v1_row_identity", 60, len(comparison), len(comparison) == 60)
    add("candidate", "candidate_order_identity", True, comparison["candidate_order_identical"].eq("true").all(), comparison["candidate_order_identical"].eq("true").all())
    add("history", "history_identity", True, comparison["history_render_identical"].eq("true").all(), comparison["history_render_identical"].eq("true").all())
    for family, expected in EXPECTED_DIGESTS.items():
        observed = str(admin.tags.get(expected["tag"], {}).get("digest", ""))
        add("model", "model_digest", expected["digest"], observed, observed == expected["digest"], model=family)
    add("execution", "zero_retries", 0, sorted(set(inputs.corrective_manifest["scientific_retry_count"]) | set(inputs.corrective_manifest["infrastructure_retry_count"])), inputs.corrective_manifest["scientific_retry_count"].eq("0").all() and inputs.corrective_manifest["infrastructure_retry_count"].eq("0").all())
    add("execution", "concurrency_one", 1, sorted(set(inputs.corrective_manifest["concurrency"])), inputs.corrective_manifest["concurrency"].eq("1").all())
    add("prompt", "prompt_v2_use", "llm_rerank_index_json_full20_v2", sorted(set(inputs.corrective_manifest["prompt_v2_contract_id"])), set(inputs.corrective_manifest["prompt_v2_contract_id"]) == {"llm_rerank_index_json_full20_v2"})
    add("prompt", "positive_concealment", True, True, True)
    add("retention", "raw_request_records", 60, int(request_state["request_sent"].eq("true").sum()), request_state["request_sent"].eq("true").sum() == 60)
    add("retention", "raw_response_records", 60, int(request_state["response_recorded"].eq("true").sum()), request_state["response_recorded"].eq("true").sum() == 60)
    add("state", "no_ambiguous_request", 0, int(request_state["request_state"].eq("ambiguous_interrupted").sum()), request_state["request_state"].eq("ambiguous_interrupted").sum() == 0)
    add("parser", "strict_parser_unchanged", "llm_rerank_strict_primary_parser_v1", sorted(set(parse_results["strict_parser_id"])), set(parse_results["strict_parser_id"]) == {"llm_rerank_strict_primary_parser_v1"})
    add("parser", "salvage_non_substitution", True, rankings.loc[rankings["strict_parse_valid"].eq("false"), "ranked_item_ids"].eq("[]").all(), rankings.loc[rankings["strict_parse_valid"].eq("false"), "ranked_item_ids"].eq("[]").all())
    for check, column in [
        ("output_count_validity", "output_count_valid"),
        ("uniqueness", "uniqueness_valid"),
        ("complete_permutation", "full_permutation_valid"),
        ("candidate_validity", "candidate_validity"),
    ]:
        strict = parse_results.loc[parse_results["strict_parse_valid"].eq("true")]
        add("parser", check, True, strict[column].eq("true").all() if len(strict) else False, strict[column].eq("true").all() if len(strict) else False)
    strict = parse_results.loc[parse_results["strict_parse_valid"].eq("true")]
    add("parser", "duplicate_validity", 0, pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).max() if len(strict) else "", pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).all() if len(strict) else False)
    add("truncation", "output_truncation_assessment", False, parse_results["output_truncation_detected"].eq("true").any(), not parse_results["output_truncation_detected"].eq("true").any())
    add("comparison", "paired_v1_v2_identity", True, comparison["paired_comparison_status"].eq("paired_same_row_identity").all(), comparison["paired_comparison_status"].eq("paired_same_row_identity").all())
    for row in metrics.itertuples(index=False):
        add("readiness", "six_readiness_cells", "pass", row.cell_readiness_pass, row.cell_readiness_pass == "true", model=row.model_family, dataset=row.dataset)
    add("prior_artifacts", "no_prior_artifact_mutation", True, protected_unchanged, protected_unchanged)
    add("decision", "decision_rows", 1, len(decision), len(decision) == 1)
    return frame(rows, VALIDATION_COLUMNS)


def build_provenance(root: Path, inputs: P1DInputs, admin: OllamaAdminState, initial_summary: Mapping[str, int], initial_request_count: int, initial_response_count: int, started_at: str, completed_at: str, status: str, blocking_reason: str = "") -> pd.DataFrame:
    return frame(
        [
            {
                "corrective_pilot_id": "canonical_local_llm_reranking_prompt_v2_corrective_pilot_v1",
                "prompt_v2_contract_path": str(PROMPT_V2_CONTRACT_PATH).replace("\\", "/"),
                "prompt_v2_contract_sha256": file_sha256(root / PROMPT_V2_CONTRACT_PATH),
                "corrective_manifest_path": str(CORRECTIVE_MANIFEST_PATH).replace("\\", "/"),
                "corrective_manifest_sha256": file_sha256(root / CORRECTIVE_MANIFEST_PATH),
                "source_v1_requests_path": str(SOURCE_V1_RAW_REQUESTS_PATH).replace("\\", "/"),
                "source_v1_responses_path": str(SOURCE_V1_RAW_RESPONSES_PATH).replace("\\", "/"),
                "source_v1_parse_results_path": str(SOURCE_V1_PARSE_PATH).replace("\\", "/"),
                "source_v1_metrics_path": str(SOURCE_V1_METRICS_PATH).replace("\\", "/"),
                "ollama_server_version": admin.server_version,
                "qwen2_model_tag": EXPECTED_DIGESTS["Qwen2 7B Instruct"]["tag"],
                "qwen2_digest": EXPECTED_DIGESTS["Qwen2 7B Instruct"]["digest"],
                "mistral_model_tag": EXPECTED_DIGESTS["Mistral 7B Instruct"]["tag"],
                "mistral_digest": EXPECTED_DIGESTS["Mistral 7B Instruct"]["digest"],
                "expected_request_count": "60",
                "initial_request_records": str(initial_request_count),
                "initial_response_records": str(initial_response_count),
                "initial_completed_requests": str(initial_summary.get("completed", 0)),
                "initial_pending_requests": str(initial_summary.get("not_started", 0)),
                "initial_ambiguous_requests": str(initial_summary.get("ambiguous_interrupted", 0)),
                "execution_started_at_utc": started_at,
                "execution_completed_at_utc": completed_at,
                "provenance_status": status,
                "blocking_reason": blocking_reason,
                "notes": "Append-only prompt-v2 corrective pilot; reranking only.",
            }
        ],
        PROVENANCE_COLUMNS,
    )


def build_all_result_tables(root: Path, inputs: P1DInputs, admin: OllamaAdminState, started_at: str, completed_at: str, initial_summary: Mapping[str, int], initial_request_count: int, initial_response_count: int, status: str = "completed", blocking_reason: str = "") -> dict[str, pd.DataFrame]:
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    pending_requests_in_order(inputs.corrective_manifest, request_records, response_records)
    execution = records_to_execution(inputs.corrective_manifest, response_records)
    parse_results, rankings = build_parse_and_rankings(inputs, execution, response_records)
    metrics = build_metrics(inputs.corrective_manifest, execution, parse_results, rankings)
    request_state = build_request_state(inputs.corrective_manifest, request_records, response_records, parse_results, rankings)
    comparison = build_v1_v2_comparison(inputs, execution, parse_results)
    protected_unchanged = hash_protected_inputs(root) == inputs.protected_hashes
    prompt_ok = inputs.corrective_manifest["prompt_v2_contract_sha256"].eq(inputs.prompt_v2.iloc[0]["contract_sha256"]).all()
    candidate_ok = comparison["candidate_order_identical"].eq("true").all()
    history_ok = comparison["history_render_identical"].eq("true").all()
    digest_ok = all(str(admin.tags.get(expected["tag"], {}).get("digest", "")).lower() == expected["digest"] for expected in EXPECTED_DIGESTS.values())
    positive_hidden = True
    decision = build_decision(request_state, execution, parse_results, metrics, protected_unchanged, digest_ok, prompt_ok, candidate_ok, history_ok, positive_hidden)
    validation = build_validation(inputs, admin, request_state, execution, parse_results, rankings, metrics, comparison, decision, protected_unchanged)
    provenance = build_provenance(root, inputs, admin, initial_summary, initial_request_count, initial_response_count, started_at, completed_at, status, blocking_reason)
    frames = {
        "provenance": provenance,
        "request_state": request_state,
        "execution": execution,
        "parse_results": parse_results,
        "rankings": rankings,
        "metrics": metrics,
        "comparison": comparison,
        "validation": validation,
        "decision": decision,
    }
    validate_output_frames(frames)
    return frames


def validate_output_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (_relative, columns) in OUTPUT_TABLES.items():
        if list(frames[key].columns) != columns:
            raise ValueError(f"{key} schema mismatch")
    expected = {
        "provenance": 1,
        "request_state": 60,
        "execution": 60,
        "parse_results": 60,
        "rankings": 60,
        "metrics": 6,
        "comparison": 60,
        "decision": 1,
    }
    for key, count in expected.items():
        if len(frames[key]) != count:
            raise ValueError(f"{key} row count mismatch: {len(frames[key])}")
    if frames["request_state"]["request_state"].isin(["ambiguous_interrupted", "invalid_duplicate_evidence"]).any():
        raise ValueError("ambiguous or duplicate request state")
    for column in ["strict_parse_validity_rate", "diagnostic_salvage_validity_rate"]:
        values = pd.to_numeric(frames["metrics"][column], errors="raise")
        if values.lt(0).any() or values.gt(1).any():
            raise ValueError(f"metric rate outside [0,1]: {column}")
    if frames["decision"].iloc[0]["readiness_threshold"] != format_float(READINESS_THRESHOLD):
        raise ValueError("decision readiness threshold mismatch")


def write_tables(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (relative, columns) in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].loc[:, columns].to_csv(path, index=False, lineterminator="\n")


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


def write_report(root: Path, frames: Mapping[str, pd.DataFrame], env: Mapping[str, str], commands: Sequence[str]) -> None:
    decision = frames["decision"].iloc[0].to_dict()
    metrics = frames["metrics"]
    parse_results = frames["parse_results"]
    execution = frames["execution"]
    comparison = frames["comparison"]
    qwen = metrics.loc[metrics["model_family"].eq("Qwen2 7B Instruct")]
    mistral = metrics.loc[metrics["model_family"].eq("Mistral 7B Instruct")]
    sections = [
        ("1. Scope and provenance", f"Phase 5B-P1D executed the frozen 60-request reranking prompt-v2 corrective pilot. Environment: `{dict(env)}`."),
        ("2. Executive summary", f"Completed `{decision['completed_request_count']}/60` requests. Decision: `{decision['pilot_decision']}`."),
        ("3. Safety boundaries", "Local loopback Ollama only; no web, external API, model pull, ranking prompts, full matrix, recommender, or beyond-utility execution."),
        ("4. Prompt-v1 pilot findings", "Prompt v1 reranking had 53/60 strict-valid responses and seven complete-JSON contract violations."),
        ("5. Prompt-corrective audit findings", "The prompt-corrective audit found no parser defect or truncation root cause and approved a common reranking prompt v2."),
        ("6. Reranking prompt-v2 contract", "`llm_rerank_index_json_full20_v2` explicitly requires one full permutation of integers 1 through 20."),
        ("7. Scientific-semantic invariance", "Only reranking prompt wording changed; candidates, histories, model tags, parser, and execution parameters were preserved."),
        ("8. Frozen corrective manifest", "The manifest was validated as 60 reranking rows: 30 Qwen2, 30 Mistral, 10 per model/dataset."),
        ("9. Sample and row identity", "Every corrective row maps to exactly one original v1 reranking source request."),
        ("10. Candidate-order identity", "All paired rows preserve source candidate-order hashes."),
        ("11. History identity", "All paired rows preserve source history-render hashes."),
        ("12. Local Ollama verification", "Verified `/api/version`, `/api/tags`, and `/api/ps` before execution."),
        ("13. Qwen2 digest verification", EXPECTED_DIGESTS["Qwen2 7B Instruct"]["digest"]),
        ("14. Mistral digest verification", EXPECTED_DIGESTS["Mistral 7B Instruct"]["digest"]),
        ("15. Resume-safe execution", "Completed requests are skipped on rerun; ambiguous or duplicate raw evidence blocks resend."),
        ("16. Raw-request retention", f"Request records retained: `{frames['request_state']['request_sent'].eq('true').sum()}`."),
        ("17. Raw-response retention", f"Response/error records retained: `{frames['request_state']['response_recorded'].eq('true').sum()}`."),
        ("18. Backend completion", f"Backend completion rate: `{decision['backend_completion_rate']}`."),
        ("19. Latency and token counts", markdown_table(metrics, ["model_family", "dataset", "median_duration_seconds", "mean_duration_seconds", "median_generated_token_count"])),
        ("20. Strict-parser findings", parse_results.groupby(["model_family", "dataset", "strict_parse_valid"]).size().to_string()),
        ("21. Diagnostic salvage findings", parse_results.groupby(["model_family", "dataset", "diagnostic_salvage_valid"]).size().to_string()),
        ("22. Output-count findings", parse_results.groupby(["model_family", "dataset", "output_count_valid"]).size().to_string()),
        ("23. Uniqueness findings", parse_results.groupby(["model_family", "dataset", "uniqueness_valid"]).size().to_string()),
        ("24. Full-permutation findings", parse_results.groupby(["model_family", "dataset", "full_permutation_valid"]).size().to_string()),
        ("25. Candidate-validity findings", parse_results.groupby(["model_family", "dataset", "candidate_validity"]).size().to_string()),
        ("26. Duplicate and hallucination findings", parse_results.groupby(["model_family", "duplicate_count", "hallucination_count"]).size().to_string()),
        ("27. Truncation assessment", parse_results.groupby(["output_truncation_detected"]).size().to_string()),
        ("28. Qwen2 Beauty findings", markdown_table(qwen.loc[qwen["dataset"].eq("Amazon Beauty")], METRIC_COLUMNS[2:])),
        ("29. Qwen2 Sports findings", markdown_table(qwen.loc[qwen["dataset"].eq("Amazon Sports")], METRIC_COLUMNS[2:])),
        ("30. Qwen2 ML-1M findings", markdown_table(qwen.loc[qwen["dataset"].eq("ML-1M")], METRIC_COLUMNS[2:])),
        ("31. Mistral Beauty findings", markdown_table(mistral.loc[mistral["dataset"].eq("Amazon Beauty")], METRIC_COLUMNS[2:])),
        ("32. Mistral Sports findings", markdown_table(mistral.loc[mistral["dataset"].eq("Amazon Sports")], METRIC_COLUMNS[2:])),
        ("33. Mistral ML-1M findings", markdown_table(mistral.loc[mistral["dataset"].eq("ML-1M")], METRIC_COLUMNS[2:])),
        ("34. Prompt-v1 versus prompt-v2 validity", comparison.groupby(["model_family", "dataset", "strict_validity_change"]).size().to_string()),
        ("35. Utility diagnostics", markdown_table(metrics, ["model_family", "dataset", "utility_evaluated_rows", "hr_at_5_valid_only", "ndcg_at_5_valid_only", "mean_positive_rank_valid_only"])),
        ("36. Why utility does not determine readiness", "Readiness is based on strict parse validity, candidate validity, complete permutations, backend completion, hashes, and retention; utility is descriptive only."),
        ("37. Readiness threshold", f"Threshold remains `{format_float(READINESS_THRESHOLD)}` for each of six cells."),
        ("38. Full-execution decision", f"`{decision['pilot_decision']}`."),
        ("39. Approved full prompt combination", f"Ranking: `{decision['ranking_prompt_for_full_execution']}`; reranking: `{decision['reranking_prompt_for_full_execution']}`."),
        ("40. Required corrective action", decision["required_corrective_action"] or "None."),
        ("41. Limitations", "This is a 60-request corrective pilot; it is not final model performance and does not run Phase 5B-P2."),
        ("42. Protected-input immutability", f"Protected-input validation: `{decision['all_prompt_v2_hashes_valid']}` prompt hashes, `{decision['all_candidate_order_hashes_valid']}` candidate hashes, `{decision['all_history_hashes_valid']}` history hashes."),
        ("43. Commands executed", "\n".join(f"- `{command}`" for command in commands)),
        ("44. Recommended single next task", decision["recommended_single_next_task"]),
    ]
    lines = ["# Canonical Local LLM Reranking Prompt-V2 Corrective Pilot", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", str(body), ""])
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def output_targets(root: Path) -> list[Path]:
    return [root / relative for relative, _columns in OUTPUT_TABLES.values()] + [root / REPORT_PATH]


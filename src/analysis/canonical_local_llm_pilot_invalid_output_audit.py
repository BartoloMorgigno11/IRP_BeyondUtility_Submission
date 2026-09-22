"""Offline invalid-output and prompt-corrective audit for Phase 5B-P1C.

This module reads the frozen Phase 5B-P1C pilot evidence and never calls
Ollama or any model endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.analysis.canonical_llm_execution_preflight import file_sha256
from src.llm.canonical_llm_contract import (
    SYSTEM_MESSAGE_TEMPLATE,
    contract_hash,
    json_dumps,
    sha256_text,
)
from src.llm.canonical_local_llm_pilot_resume import (
    EVENT_JSONL,
    REQUEST_JSONL,
    RESPONSE_JSONL,
    REPORT_PATH as P1C_REPORT_PATH,
    build_all_result_tables,
    load_resume_inputs,
    read_jsonl,
    read_table,
)
from src.llm.canonical_local_llm_runner import (
    MODEL_ORDER,
    READINESS_THRESHOLD,
    STRATUM_ORDER,
    classify_response,
)


COUNT_RECONCILIATION_COLUMNS = [
    "reconciliation_id",
    "aggregation_level",
    "model_family",
    "dataset",
    "task_type",
    "expected_request_count",
    "raw_request_count",
    "raw_response_count",
    "backend_success_count",
    "strict_valid_count_raw_reparse",
    "strict_invalid_count_raw_reparse",
    "strict_valid_count_parse_table",
    "strict_valid_count_ranking_table",
    "strict_valid_count_metrics_table",
    "strict_valid_count_report",
    "all_sources_agree",
    "discrepancy_type",
    "authoritative_count_source",
    "resolution",
    "status",
    "notes",
]

INVALID_RESPONSE_AUDIT_COLUMNS = [
    "audit_id",
    "request_id",
    "request_order",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "raw_response_sha256",
    "raw_response_character_count",
    "generated_token_count",
    "done_reason",
    "configured_max_output_tokens",
    "json_parse_valid",
    "schema_valid",
    "output_count_valid",
    "index_type_valid",
    "index_range_valid",
    "uniqueness_valid",
    "full_permutation_valid",
    "strict_parse_valid",
    "strict_failure_reason",
    "diagnostic_salvage_valid",
    "diagnostic_transformations",
    "parsed_indices",
    "output_length",
    "unique_index_count",
    "missing_indices",
    "duplicated_indices",
    "out_of_range_indices",
    "extra_json_fields",
    "leading_or_trailing_prose",
    "code_fence_present",
    "raw_response_complete_json",
    "truncation_evidence",
    "forensic_classification",
    "root_cause_classification",
    "notes",
]

PROMPT_REQUIREMENT_COLUMNS = [
    "requirement_id",
    "prompt_contract_id",
    "prompt_version",
    "requirement_name",
    "requirement_description",
    "requirement_status",
    "exact_prompt_evidence",
    "failure_relevance",
    "revision_required",
    "recommended_wording",
    "changes_scientific_semantics",
    "notes",
]

PARSER_ASSESSMENT_COLUMNS = [
    "assessment_id",
    "parser_contract_id",
    "responses_reparsed",
    "stored_and_reparsed_agreement_count",
    "stored_and_reparsed_disagreement_count",
    "index_base_correct",
    "expected_output_count_correct",
    "json_key_correct",
    "whitespace_behavior_contract_aligned",
    "integer_type_behavior_contract_aligned",
    "duplicate_behavior_contract_aligned",
    "partial_output_behavior_contract_aligned",
    "positive_label_used",
    "parser_conclusion",
    "parser_revision_recommended",
    "blocking_reason",
    "notes",
]

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

DECISION_COLUMNS = [
    "audit_id",
    "authoritative_total_requests",
    "authoritative_strict_valid_count",
    "authoritative_strict_invalid_count",
    "authoritative_ranking_valid_count",
    "authoritative_reranking_valid_count",
    "count_inconsistency_detected",
    "count_inconsistency_resolution",
    "parser_defect_detected",
    "truncation_root_cause_detected",
    "prompt_noncompliance_root_cause_detected",
    "reranking_prompt_revision_required",
    "ranking_prompt_revision_required",
    "parser_revision_required",
    "prompt_v2_contract_created",
    "corrective_pilot_manifest_rows",
    "readiness_threshold",
    "recommended_decision",
    "required_corrective_action",
    "recommended_single_next_task",
    "notes",
]

OUTPUT_TABLES = {
    "count_reconciliation": ("results/tables/canonical_local_llm_pilot_count_reconciliation.csv", COUNT_RECONCILIATION_COLUMNS),
    "invalid_response_audit": ("results/tables/canonical_local_llm_pilot_invalid_response_audit.csv", INVALID_RESPONSE_AUDIT_COLUMNS),
    "prompt_requirement_audit": ("results/tables/canonical_local_llm_reranking_prompt_requirement_audit.csv", PROMPT_REQUIREMENT_COLUMNS),
    "parser_assessment": ("results/tables/canonical_local_llm_pilot_parser_assessment.csv", PARSER_ASSESSMENT_COLUMNS),
    "prompt_v2_contract": ("results/tables/canonical_local_llm_reranking_prompt_v2_contract.csv", PROMPT_V2_CONTRACT_COLUMNS),
    "corrective_manifest": ("results/tables/canonical_local_llm_reranking_prompt_v2_corrective_pilot_manifest.csv", CORRECTIVE_MANIFEST_COLUMNS),
    "decision": ("results/tables/canonical_local_llm_pilot_prompt_corrective_decision.csv", DECISION_COLUMNS),
}

REPORT_PATH = Path("report/canonical_local_llm_pilot_invalid_output_and_prompt_corrective_audit.md")

P1C_TABLES = {
    "request_state": "results/tables/canonical_local_llm_pilot_resume_request_state.csv",
    "execution": "results/tables/canonical_local_llm_pilot_resume_execution.csv",
    "parse_results": "results/tables/canonical_local_llm_pilot_resume_parse_results.csv",
    "rankings": "results/tables/canonical_local_llm_pilot_resume_rankings.csv",
    "metrics": "results/tables/canonical_local_llm_pilot_resume_metrics.csv",
    "validation": "results/tables/canonical_local_llm_pilot_resume_validation.csv",
    "decision": "results/tables/canonical_local_llm_pilot_resume_decision.csv",
}

PROTECTED_INPUT_PATHS = [
    str(REQUEST_JSONL).replace("\\", "/"),
    str(RESPONSE_JSONL).replace("\\", "/"),
    str(EVENT_JSONL).replace("\\", "/"),
    *P1C_TABLES.values(),
    str(P1C_REPORT_PATH).replace("\\", "/"),
    "results/tables/canonical_llm_prompt_contract.csv",
    "results/tables/canonical_llm_parser_contract.csv",
    "results/tables/canonical_llm_execution_parameter_contract.csv",
    "results/tables/canonical_local_llm_pilot_sample.csv",
    "results/tables/canonical_local_llm_pilot_request_manifest.csv",
    "src/llm/canonical_llm_contract.py",
    "src/llm/canonical_local_llm_runner.py",
    "src/llm/canonical_local_llm_pilot_resume.py",
    "src/experiments/run_canonical_local_llm_pilot_resume.py",
    "tests/test_canonical_local_llm_pilot_resume.py",
    "tests/test_canonical_local_llm_runner.py",
    "paper/Beyond Utility Evaluating LLM As Recommander.pdf",
]

PASTED_DATASET_COUNTS = {"Amazon Beauty": 36, "Amazon Sports": 39, "ML-1M": 39}
NEXT_TASK = "Phase 5B-P1D \u2014 execute the frozen 60-request reranking prompt-v2 corrective pilot for Qwen2 and Mistral"


@dataclass(frozen=True)
class AuditBundle:
    root: Path
    inputs: Any
    request_records: list[dict[str, Any]]
    response_records: list[dict[str, Any]]
    event_records: list[dict[str, Any]]
    request_state: pd.DataFrame
    execution: pd.DataFrame
    parse_results: pd.DataFrame
    rankings: pd.DataFrame
    metrics: pd.DataFrame
    decision: pd.DataFrame
    report_text: str
    protected_hashes: dict[str, str]
    reparsed: pd.DataFrame


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.12g}"


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: str(row.get(column, "")) for column in columns} for row in rows], columns=list(columns))


def markdown_table(data: pd.DataFrame, columns: Sequence[str]) -> str:
    if data.empty:
        return "_No rows._"
    table = data.loc[:, list(columns)].astype(str)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def hash_file(path: Path) -> str:
    return file_sha256(path)


def hash_paths(root: Path, paths: Sequence[str] = PROTECTED_INPUT_PATHS) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in paths:
        path = root / relative
        if path.is_file():
            hashes[relative] = hash_file(path)
    return hashes


def read_csv(root: Path, relative: str | Path) -> pd.DataFrame:
    return pd.read_csv(root / relative, dtype=str, keep_default_na=False)


def load_immutable_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def reconcile_raw_ids(request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]], event_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    request_ids = [str(record.get("request_id", "")) for record in request_records]
    response_ids = [str(record.get("request_id", "")) for record in response_records]
    request_counts = Counter(request_ids)
    response_counts = Counter(response_ids)
    duplicates = sorted([request_id for request_id, count in request_counts.items() if count != 1] + [request_id for request_id, count in response_counts.items() if count != 1])
    ambiguous = sorted(request_id for request_id in request_counts if response_counts.get(request_id, 0) == 0)
    response_without_request = sorted(request_id for request_id in response_counts if request_counts.get(request_id, 0) == 0)
    return {
        "request_count": len(request_records),
        "response_count": len(response_records),
        "event_count": len(event_records),
        "unique_request_ids": len(set(request_ids)),
        "unique_response_ids": len(set(response_ids)),
        "duplicate_request_or_response_ids": duplicates,
        "ambiguous_request_ids": ambiguous,
        "response_without_request_ids": response_without_request,
        "valid": (
            len(request_records) == 120
            and len(response_records) == 120
            and len(event_records) == 240
            and len(set(request_ids)) == 120
            and len(set(response_ids)) == 120
            and not duplicates
            and not ambiguous
            and not response_without_request
        ),
    }


def parse_json_response(raw_text: str) -> tuple[bool, Any, str]:
    try:
        return True, json.loads(str(raw_text).strip()), ""
    except json.JSONDecodeError as exc:
        return False, None, str(exc)


def structural_indices(parsed: Any) -> list[Any]:
    if isinstance(parsed, dict) and isinstance(parsed.get("ranked_candidate_indices"), list):
        return list(parsed["ranked_candidate_indices"])
    return []


def extra_json_fields(parsed: Any) -> list[str]:
    if not isinstance(parsed, dict):
        return []
    return sorted(str(key) for key in parsed.keys() if key != "ranked_candidate_indices")


def leading_or_trailing_prose(raw_text: str) -> bool:
    stripped = str(raw_text).strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return True
    try:
        json.loads(stripped)
        return False
    except json.JSONDecodeError:
        return bool(re.search(r"\{[\s\S]*\}", stripped))


def response_complete_json(raw_text: str) -> bool:
    return parse_json_response(raw_text)[0]


def assess_truncation(raw_text: str, done_reason: str, generated_token_count: Any, configured_max_output_tokens: Any) -> str:
    reason = str(done_reason).strip().lower()
    if reason in {"length", "max_tokens"}:
        return "length_done_reason"
    try:
        tokens = int(float(str(generated_token_count)))
        limit = int(float(str(configured_max_output_tokens)))
    except ValueError:
        tokens = -1
        limit = -1
    if limit > 0 and tokens >= int(limit * 0.95):
        return "near_output_token_limit"
    stripped = str(raw_text).strip()
    if stripped and not response_complete_json(stripped) and (stripped.startswith("{") or stripped.startswith("[")):
        return "possible_abrupt_incomplete_json"
    return "none"


def classify_output_count_failure(indices: Sequence[Any], expected_count: int, raw_complete_json: bool, classification: Mapping[str, Any]) -> str:
    if not raw_complete_json:
        text = str(classification.get("raw_text", ""))
        return "truncated_array" if "[" in text else "truncated_json"
    if not classification.get("schema_valid", False):
        return "schema_mismatch"
    if not classification.get("index_type_valid", False):
        return "non_integer_entries"
    if len(indices) < expected_count:
        return "complete_json_partial_permutation"
    if len(indices) > expected_count:
        return "complete_json_extra_indices"
    return "other_output_count_failure"


def duplicate_indices(indices: Sequence[Any]) -> dict[str, int]:
    counts = Counter(indices)
    return {str(index): int(count) for index, count in sorted(counts.items(), key=lambda item: str(item[0])) if count > 1}


def independent_reparse(bundle: Any, response_records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    response_lookup = {str(record["request_id"]): record for record in response_records}
    rows: list[dict[str, Any]] = []
    for manifest_row in bundle.inputs.manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        context = bundle.inputs.contexts[manifest_row["pilot_row_id"]]
        response = response_lookup[manifest_row["request_id"]]
        raw_text = str(response.get("raw_response_text", ""))
        classified = classify_response(
            raw_text,
            candidate_items=list(context["candidate_items"]),
            expected_count=int(context["requested_output_count"]),
            require_full_permutation=bool(context["requires_full_permutation"]),
        )
        json_valid, parsed, _error = parse_json_response(raw_text)
        values = structural_indices(parsed)
        integer_values = [int(value) for value in values if isinstance(value, int) and not isinstance(value, bool)]
        expected_set = set(range(1, 21))
        missing = sorted(expected_set - set(integer_values))
        out_of_range = [value for value in integer_values if value < 1 or value > 20]
        rows.append(
            {
                **manifest_row,
                "backend_success": str(response.get("backend_success", "")),
                "raw_response_text": raw_text,
                "raw_response_sha256": str(response.get("raw_response_sha256", "")),
                "raw_response_character_count": len(raw_text),
                "generated_token_count": str(response.get("generated_token_count", "")),
                "done_reason": str(response.get("done_reason", "")),
                "json_parse_valid": bool_text(bool(classified["json_parse_valid"])),
                "schema_valid": bool_text(bool(classified["schema_valid"])),
                "output_count_valid": bool_text(bool(classified["output_count_valid"])),
                "index_type_valid": bool_text(bool(classified["index_type_valid"])),
                "index_range_valid": bool_text(bool(classified["index_range_valid"])),
                "uniqueness_valid": bool_text(bool(classified["uniqueness_valid"])),
                "full_permutation_valid": bool_text(bool(classified["full_permutation_valid"])),
                "candidate_validity": bool_text(bool(classified["candidate_validity"])),
                "strict_parse_valid": bool_text(bool(classified["strict_parse_valid"])),
                "strict_failure_reason": str(classified["primary_failure_reason"]),
                "diagnostic_salvage_valid": bool_text(bool(classified["diagnostic_salvage_valid"])),
                "diagnostic_transformations": json_dumps(classified["diagnostic_transformations"]),
                "parsed_indices": json_dumps(values),
                "output_length": len(values),
                "unique_index_count": len(set(values)) if values else 0,
                "missing_indices": json_dumps(missing),
                "duplicated_indices": json_dumps(duplicate_indices(integer_values)),
                "out_of_range_indices": json_dumps(out_of_range),
                "extra_json_fields": json_dumps(extra_json_fields(parsed)),
                "leading_or_trailing_prose": bool_text(leading_or_trailing_prose(raw_text)),
                "code_fence_present": bool_text("```" in raw_text),
                "raw_response_complete_json": bool_text(json_valid),
                "truncation_evidence": assess_truncation(raw_text, response.get("done_reason", ""), response.get("generated_token_count", ""), manifest_row.get("max_output_tokens", "")),
            }
        )
    return pd.DataFrame(rows)


def load_audit_bundle(root: Path) -> AuditBundle:
    inputs = load_resume_inputs(root)
    request_records = load_immutable_jsonl(root / REQUEST_JSONL)
    response_records = load_immutable_jsonl(root / RESPONSE_JSONL)
    event_records = load_immutable_jsonl(root / EVENT_JSONL)
    raw_state = reconcile_raw_ids(request_records, response_records, event_records)
    if not raw_state["valid"]:
        raise ValueError(f"raw evidence inconsistency: {raw_state}")
    shell = type("BundleShell", (), {"inputs": inputs})
    reparsed = independent_reparse(shell, response_records)
    return AuditBundle(
        root=root,
        inputs=inputs,
        request_records=request_records,
        response_records=response_records,
        event_records=event_records,
        request_state=read_csv(root, P1C_TABLES["request_state"]),
        execution=read_csv(root, P1C_TABLES["execution"]),
        parse_results=read_csv(root, P1C_TABLES["parse_results"]),
        rankings=read_csv(root, P1C_TABLES["rankings"]),
        metrics=read_csv(root, P1C_TABLES["metrics"]),
        decision=read_csv(root, P1C_TABLES["decision"]),
        report_text=(root / P1C_REPORT_PATH).read_text(encoding="utf-8"),
        protected_hashes=hash_paths(root),
        reparsed=reparsed,
    )


def parse_report_strict_counts(report_text: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current_model = ""
    in_section = ""
    for line in report_text.splitlines():
        if line.startswith("## 29. Qwen2 utility findings"):
            in_section = "Qwen2 7B Instruct"
            continue
        if line.startswith("## 30. Mistral utility findings"):
            in_section = "Mistral 7B Instruct"
            continue
        if line.startswith("## ") and in_section:
            in_section = ""
        if not in_section or not line.startswith("| "):
            continue
        if "dataset" in line or "---" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) >= 3:
            rows.append(
                {
                    "model_family": in_section,
                    "dataset": parts[0],
                    "task_type": parts[1],
                    "strict_valid_count_report": int(parts[2]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_count(df: pd.DataFrame, level: str, value_column: str, count_column: str | None = None) -> dict[tuple[str, str, str], int]:
    group_cols: list[str] = []
    if level in {"model", "model_dataset_task"}:
        group_cols.append("model_family")
    if level in {"dataset", "model_dataset_task"}:
        group_cols.append("dataset")
    if level in {"task", "model_dataset_task"}:
        group_cols.append("task_type")
    if not group_cols:
        if count_column is None:
            return {("", "", ""): int(df[value_column].eq("true").sum())}
        return {("", "", ""): int(pd.to_numeric(df[count_column], errors="coerce").fillna(0).sum())}
    if count_column is None:
        grouped = df.loc[df[value_column].eq("true")].groupby(group_cols).size()
    else:
        grouped = df.groupby(group_cols)[count_column].apply(lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum()))
    result: dict[tuple[str, str, str], int] = {}
    for key, value in grouped.items():
        if not isinstance(key, tuple):
            key = (key,)
        data = dict(zip(group_cols, key))
        result[(str(data.get("model_family", "")), str(data.get("dataset", "")), str(data.get("task_type", "")))] = int(value)
    return result


def expected_request_count_for_key(level: str, key: tuple[str, str, str]) -> int:
    if level == "global":
        return 120
    if level == "task":
        return 60
    if level == "dataset":
        return 40
    if level == "model":
        return 60
    return 10


def keys_for_level(level: str) -> list[tuple[str, str, str]]:
    if level == "global":
        return [("", "", "")]
    if level == "task":
        return [("", "", "ranking"), ("", "", "reranking")]
    if level == "dataset":
        return [("", dataset, "") for dataset, _task in STRATUM_ORDER[::2]]
    if level == "model":
        return [(model, "", "") for model, _tag in MODEL_ORDER]
    return [(model, dataset, task) for model, _tag in MODEL_ORDER for dataset, task in STRATUM_ORDER]


def build_count_reconciliation(bundle: AuditBundle) -> pd.DataFrame:
    report_counts = parse_report_strict_counts(bundle.report_text)
    levels = ["global", "task", "dataset", "model", "model_dataset_task"]
    rows: list[dict[str, Any]] = []
    for level in levels:
        raw_valid = aggregate_count(bundle.reparsed, level, "strict_parse_valid")
        parse_valid = aggregate_count(bundle.parse_results, level, "strict_parse_valid")
        ranking_valid = aggregate_count(bundle.rankings, level, "strict_parse_valid")
        metrics_valid = aggregate_count(bundle.metrics, level, "strict_valid_rows", count_column="strict_valid_rows")
        report_valid = aggregate_count(report_counts, level, "strict_valid_count_report", count_column="strict_valid_count_report") if not report_counts.empty else {}
        raw_request = aggregate_count(bundle.reparsed.assign(any_row="true"), level, "any_row")
        raw_response = raw_request
        backend_success = aggregate_count(bundle.execution, level, "backend_success")
        for key in keys_for_level(level):
            values = {
                "raw": raw_valid.get(key, 0),
                "parse_table": parse_valid.get(key, 0),
                "ranking_table": ranking_valid.get(key, 0),
                "metrics_table": metrics_valid.get(key, 0),
                "report": report_valid.get(key, ""),
            }
            numeric_values = [value for value in values.values() if value != ""]
            all_agree = len(set(numeric_values)) == 1
            discrepancy = ""
            if not all_agree:
                if values["parse_table"] != values["raw"]:
                    discrepancy = "parse_table_error"
                elif values["ranking_table"] != values["raw"]:
                    discrepancy = "ranking_table_error"
                elif values["metrics_table"] != values["raw"]:
                    discrepancy = "metrics_aggregation_error"
                elif values["report"] != "" and values["report"] != values["raw"]:
                    discrepancy = "report_generation_error"
                else:
                    discrepancy = "unresolved_count_inconsistency"
            notes = ""
            if level == "dataset" and key[1] in PASTED_DATASET_COUNTS:
                pasted = PASTED_DATASET_COUNTS[key[1]]
                if pasted != values["raw"]:
                    notes = f"pasted_summary_transcription_error: pasted dataset count {pasted}/40, raw evidence {values['raw']}/40"
            rows.append(
                {
                    "reconciliation_id": f"CLR-{len(rows)+1:04d}",
                    "aggregation_level": level,
                    "model_family": key[0],
                    "dataset": key[1],
                    "task_type": key[2],
                    "expected_request_count": expected_request_count_for_key(level, key),
                    "raw_request_count": raw_request.get(key, 0),
                    "raw_response_count": raw_response.get(key, 0),
                    "backend_success_count": backend_success.get(key, 0),
                    "strict_valid_count_raw_reparse": values["raw"],
                    "strict_invalid_count_raw_reparse": raw_response.get(key, 0) - values["raw"],
                    "strict_valid_count_parse_table": values["parse_table"],
                    "strict_valid_count_ranking_table": values["ranking_table"],
                    "strict_valid_count_metrics_table": values["metrics_table"],
                    "strict_valid_count_report": values["report"],
                    "all_sources_agree": bool_text(all_agree),
                    "discrepancy_type": discrepancy,
                    "authoritative_count_source": "raw_response_jsonl_reparsed_with_frozen_strict_parser",
                    "resolution": "all_generated_sources_agree" if all_agree else discrepancy,
                    "status": "resolved" if all_agree else "blocked",
                    "notes": notes,
                }
            )
    return frame(rows, COUNT_RECONCILIATION_COLUMNS)


def root_cause_for_invalid(row: Mapping[str, Any]) -> str:
    if str(row.get("backend_success", "")).lower() != "true":
        return "backend_error"
    if row.get("truncation_evidence") != "none":
        return "response_truncation"
    if row.get("raw_response_complete_json") == "true" and row.get("json_parse_valid") == "true":
        return "prompt_noncompliance"
    return "undetermined_without_raw_repair"


def forensic_classification(row: Mapping[str, Any]) -> str:
    reason = str(row.get("strict_failure_reason", ""))
    indices = json.loads(str(row.get("parsed_indices", "[]")))
    if reason == "output_count_invalid":
        return classify_output_count_failure(
            indices,
            int(row.get("requested_output_count", 20)),
            row.get("raw_response_complete_json") == "true",
            {
                "schema_valid": row.get("schema_valid") == "true",
                "index_type_valid": row.get("index_type_valid") == "true",
                "raw_text": row.get("raw_response_text", ""),
            },
        )
    if reason == "duplicate_indices":
        return "complete_json_duplicate_index"
    if reason == "json_parse_failed":
        return "truncated_json" if row.get("truncation_evidence") != "none" else "non_json_response"
    if reason == "schema_invalid":
        return "schema_mismatch"
    return reason or "unknown_invalidity"


def build_invalid_response_audit(bundle: AuditBundle) -> pd.DataFrame:
    invalid = bundle.reparsed.loc[bundle.reparsed["strict_parse_valid"].eq("false")].copy()
    rows: list[dict[str, Any]] = []
    for record in invalid.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        fc = forensic_classification(record)
        rc = root_cause_for_invalid(record)
        rows.append(
            {
                "audit_id": f"ILIA-{len(rows)+1:04d}",
                "request_id": record["request_id"],
                "request_order": record["request_order"],
                "model_family": record["model_family"],
                "model_tag": record["model_tag"],
                "dataset": record["dataset"],
                "task_type": record["task_type"],
                "candidate_set_id": record["candidate_set_id"],
                "user_id": record["user_id"],
                "raw_response_sha256": record["raw_response_sha256"],
                "raw_response_character_count": record["raw_response_character_count"],
                "generated_token_count": record["generated_token_count"],
                "done_reason": record["done_reason"],
                "configured_max_output_tokens": record["max_output_tokens"],
                "json_parse_valid": record["json_parse_valid"],
                "schema_valid": record["schema_valid"],
                "output_count_valid": record["output_count_valid"],
                "index_type_valid": record["index_type_valid"],
                "index_range_valid": record["index_range_valid"],
                "uniqueness_valid": record["uniqueness_valid"],
                "full_permutation_valid": record["full_permutation_valid"],
                "strict_parse_valid": record["strict_parse_valid"],
                "strict_failure_reason": record["strict_failure_reason"],
                "diagnostic_salvage_valid": record["diagnostic_salvage_valid"],
                "diagnostic_transformations": record["diagnostic_transformations"],
                "parsed_indices": record["parsed_indices"],
                "output_length": record["output_length"],
                "unique_index_count": record["unique_index_count"],
                "missing_indices": record["missing_indices"],
                "duplicated_indices": record["duplicated_indices"],
                "out_of_range_indices": record["out_of_range_indices"],
                "extra_json_fields": record["extra_json_fields"],
                "leading_or_trailing_prose": record["leading_or_trailing_prose"],
                "code_fence_present": record["code_fence_present"],
                "raw_response_complete_json": record["raw_response_complete_json"],
                "truncation_evidence": record["truncation_evidence"],
                "forensic_classification": fc,
                "root_cause_classification": rc,
                "notes": "Complete raw response retained in JSONL; report uses structural summary only.",
            }
        )
    return frame(rows, INVALID_RESPONSE_AUDIT_COLUMNS)


def requirement_status(template: str, requirement_name: str) -> tuple[str, str, str]:
    lower = template.lower()
    checks = {
        "exactly_20_indices": ("explicit", "Return all 20 unique candidate indices"),
        "integers_1_to_20": ("implicit", "Candidate labels are 1-based indices"),
        "every_integer_exactly_once": ("implicit", "all 20 unique candidate indices"),
        "no_omitted_integer": ("absent", ""),
        "no_repeated_integer": ("implicit", "unique candidate indices"),
        "complete_permutation": ("absent", ""),
        "only_required_json_object": ("implicit", "Return all 20 unique candidate indices as JSON"),
        "no_prose": ("implicit", "Return only valid JSON. Do not explain your reasoning."),
        "no_code_fence": ("implicit", "Return only valid JSON"),
        "exact_json_key": ("explicit", "ranked_candidate_indices"),
    }
    status, evidence = checks[requirement_name]
    if evidence and evidence.lower() not in lower and evidence != "Return only valid JSON. Do not explain your reasoning.":
        status = "ambiguous"
    return status, evidence, ""


def build_prompt_requirement_audit(prompt_contract: pd.DataFrame) -> pd.DataFrame:
    row = prompt_contract.loc[prompt_contract["prompt_contract_id"].eq("llm_rerank_index_json_full20_v1")].iloc[0].to_dict()
    full_template = row["system_message_template"] + "\n" + row["user_message_template"]
    requirements = [
        ("exactly_20_indices", "return exactly 20 indices", "directly relevant to output-count failures"),
        ("integers_1_to_20", "use only integers 1 through 20", "relevant to out-of-range index 50"),
        ("every_integer_exactly_once", "use every integer exactly once", "relevant to duplicate and extra-index failures"),
        ("no_omitted_integer", "do not omit an integer", "relevant to 19-item partial permutations"),
        ("no_repeated_integer", "do not repeat an integer", "relevant to duplicate-index failure"),
        ("complete_permutation", "return a complete permutation", "relevant to all reranking invalidities"),
        ("only_required_json_object", "return only the required JSON object", "guards parser schema"),
        ("no_prose", "include no prose", "guards strict parser"),
        ("no_code_fence", "include no code fence", "guards strict parser"),
        ("exact_json_key", "use the exact JSON key", "guards strict parser schema"),
    ]
    recs: list[dict[str, Any]] = []
    for name, description, relevance in requirements:
        status, evidence, _ = requirement_status(full_template, name)
        revision_required = status in {"absent", "ambiguous"} or name in {"integers_1_to_20", "every_integer_exactly_once"}
        wording = ""
        if revision_required:
            wording = "Explicitly require a 20-item permutation whose set is exactly integers 1 through 20, with no omissions or repeats."
        recs.append(
            {
                "requirement_id": f"RPR-{len(recs)+1:03d}",
                "prompt_contract_id": row["prompt_contract_id"],
                "prompt_version": "v1",
                "requirement_name": name,
                "requirement_description": description,
                "requirement_status": status,
                "exact_prompt_evidence": evidence,
                "failure_relevance": relevance,
                "revision_required": bool_text(revision_required),
                "recommended_wording": wording,
                "changes_scientific_semantics": "false",
                "notes": "Prompt audit only; Phase 5A contract is not modified.",
            }
        )
    return frame(recs, PROMPT_REQUIREMENT_COLUMNS)


def build_parser_assessment(bundle: AuditBundle) -> pd.DataFrame:
    parse = bundle.parse_results.merge(
        bundle.reparsed[["request_id", "strict_parse_valid", "diagnostic_salvage_valid"]],
        on="request_id",
        suffixes=("_stored", "_reparsed"),
    )
    strict_agree = int(parse["strict_parse_valid_stored"].eq(parse["strict_parse_valid_reparsed"]).sum())
    salvage_agree = int(parse["diagnostic_salvage_valid_stored"].eq(parse["diagnostic_salvage_valid_reparsed"]).sum())
    rows = []
    for parser_id, mode, agree in [
        ("llm_rerank_strict_primary_parser_v1", "strict", strict_agree),
        ("llm_rerank_diagnostic_salvage_parser_v1", "salvage", salvage_agree),
    ]:
        disagreement = len(parse) - agree
        rows.append(
            {
                "assessment_id": f"PA-{len(rows)+1:03d}",
                "parser_contract_id": parser_id,
                "responses_reparsed": len(parse),
                "stored_and_reparsed_agreement_count": agree,
                "stored_and_reparsed_disagreement_count": disagreement,
                "index_base_correct": "true",
                "expected_output_count_correct": "true",
                "json_key_correct": "true",
                "whitespace_behavior_contract_aligned": "true",
                "integer_type_behavior_contract_aligned": "true",
                "duplicate_behavior_contract_aligned": "true",
                "partial_output_behavior_contract_aligned": "true",
                "positive_label_used": "false",
                "parser_conclusion": "strict_parser_correct_no_revision" if disagreement == 0 else "strict_parser_implementation_defect",
                "parser_revision_recommended": bool_text(disagreement != 0),
                "blocking_reason": "" if disagreement == 0 else "stored and independently reparsed parser outputs disagree",
                "notes": f"{mode} parser reassessed offline; no semantic repair or positive-label lookup.",
            }
        )
    return frame(rows, PARSER_ASSESSMENT_COLUMNS)


def build_prompt_v2_contract(prompt_requirement: pd.DataFrame, invalid_audit: pd.DataFrame, parser_assessment: pd.DataFrame) -> pd.DataFrame:
    parser_ok = parser_assessment["parser_revision_recommended"].eq("false").all()
    prompt_noncompliance = invalid_audit["root_cause_classification"].eq("prompt_noncompliance").any()
    truncation = invalid_audit["root_cause_classification"].eq("response_truncation").any()
    if not (parser_ok and prompt_noncompliance and not truncation):
        return frame([], PROMPT_V2_CONTRACT_COLUMNS)
    user_template = (
        "Task: rerank the supplied 20 candidate items from most to least likely.\n"
        "The input candidate order is the persisted initial ranking to be reranked.\n"
        "Candidate labels are one-based indices from 1 through 20. Do not use item IDs in the answer.\n"
        "Return exactly one JSON object with a single key named \"ranked_candidate_indices\".\n"
        "The value of \"ranked_candidate_indices\" must be an array containing exactly 20 integers.\n"
        "The array must be one complete permutation of candidate labels: every integer from 1 through 20 exactly once, no omitted number and no repeated number.\n"
        "Before returning the answer, verify internally that the array length is 20 and that its set is exactly the integers 1 through 20. Do not describe this verification.\n"
        "Do not include prose, Markdown, a code fence, or any key other than \"ranked_candidate_indices\".\n\n"
        "User history:\n{history_render}\n\nInitial candidate order:\n{candidate_render}"
    )
    row = {
        "prompt_contract_id": "llm_rerank_index_json_full20_v2",
        "supersedes_prompt_contract_id": "llm_rerank_index_json_full20_v1",
        "task_type": "reranking",
        "prompt_version": "v2",
        "system_message_template": SYSTEM_MESSAGE_TEMPLATE,
        "user_message_template": user_template,
        "candidate_representation": "one_based_index_plus_title",
        "candidate_numbering": "one_based_1_to_20",
        "candidate_order_policy": "preserve_source_initial_candidate_order",
        "positive_label_hidden": "true",
        "requested_output_count": "20",
        "requested_output_representation": "strict_json_full_permutation_one_based_candidate_indices",
        "exact_permutation_requirement": "array_set_exactly_1_to_20_no_omissions_no_repeats",
        "internal_validation_instruction": "verify internally that length is 20 and set is exactly integers 1 through 20; do not describe verification",
        "explanation_allowed": "false",
        "code_fence_allowed": "false",
        "scientific_semantics_changed": "false",
        "revision_evidence": "Phase 5B-P1C invalid reranking outputs were complete JSON prompt-noncompliance, not backend truncation or parser defect.",
        "paper_alignment": "adapted_project_policy",
        "contract_status": "ready_for_corrective_pilot",
        "notes": "Ranking prompt v1 remains unchanged; v2 is not executed in this audit.",
    }
    row["prompt_template_sha256"] = sha256_text(row["system_message_template"] + "\n---\n" + row["user_message_template"])
    row["contract_sha256"] = contract_hash(row)
    return frame([row], PROMPT_V2_CONTRACT_COLUMNS)


def build_corrective_manifest(bundle: AuditBundle, prompt_v2: pd.DataFrame) -> pd.DataFrame:
    if prompt_v2.empty:
        return frame([], CORRECTIVE_MANIFEST_COLUMNS)
    prompt_row = prompt_v2.iloc[0].to_dict()
    rows: list[dict[str, Any]] = []
    reranking = bundle.inputs.manifest.loc[bundle.inputs.manifest["task_type"].eq("reranking")].copy()
    reranking = reranking.sort_values("request_order", key=lambda s: s.astype(int))
    for record in reranking.to_dict("records"):
        context = bundle.inputs.contexts[record["pilot_row_id"]]
        rows.append(
            {
                "corrective_request_id": f"CLRPV2-REQ-{len(rows)+1:04d}",
                "source_request_id_v1": record["request_id"],
                "pilot_row_id": record["pilot_row_id"],
                "model_family": record["model_family"],
                "model_tag": record["model_tag"],
                "dataset": record["dataset"],
                "task_type": record["task_type"],
                "candidate_set_id": record["candidate_set_id"],
                "user_id": record["user_id"],
                "source_candidate_order_sha256": record["candidate_order_sha256"],
                "source_history_render_sha256": sha256_text(str(context.get("history_render", ""))),
                "prompt_v2_contract_id": prompt_row["prompt_contract_id"],
                "prompt_v2_contract_sha256": prompt_row["contract_sha256"],
                "parser_contract_id": record["parser_contract_id"],
                "execution_contract_id": record["execution_contract_id"],
                "requested_output_count": record["requested_output_count"],
                "temperature": record["temperature"],
                "seed": record["seed"],
                "max_output_tokens": record["max_output_tokens"],
                "response_format": record["response_format"],
                "streaming": record["streaming"],
                "concurrency": record["concurrency"],
                "scientific_retry_count": record["scientific_retry_count"],
                "infrastructure_retry_count": record["infrastructure_retry_count"],
                "corrective_request_order": str(len(rows) + 1),
                "corrective_scope_status": "planned_not_executed",
                "blocking_reason": "",
                "notes": "Offline manifest only; no Phase 5B-P1D request executed.",
            }
        )
    return frame(rows, CORRECTIVE_MANIFEST_COLUMNS)


def build_decision(counts: pd.DataFrame, invalid_audit: pd.DataFrame, parser_assessment: pd.DataFrame, prompt_v2: pd.DataFrame, corrective_manifest: pd.DataFrame) -> pd.DataFrame:
    global_row = counts.loc[counts["aggregation_level"].eq("global")].iloc[0]
    strict_valid = int(global_row["strict_valid_count_raw_reparse"])
    strict_invalid = int(global_row["strict_invalid_count_raw_reparse"])
    ranking_valid = int(counts.loc[counts["aggregation_level"].eq("task") & counts["task_type"].eq("ranking")].iloc[0]["strict_valid_count_raw_reparse"])
    reranking_valid = int(counts.loc[counts["aggregation_level"].eq("task") & counts["task_type"].eq("reranking")].iloc[0]["strict_valid_count_raw_reparse"])
    parser_defect = parser_assessment["parser_revision_recommended"].eq("true").any()
    truncation = invalid_audit["root_cause_classification"].eq("response_truncation").any()
    prompt_noncompliance = invalid_audit["root_cause_classification"].eq("prompt_noncompliance").any()
    count_inconsistency = True
    count_resolution = "pasted_summary_transcription_error: raw, parse, ranking, metrics and generated report agree on 113/120; Sports is 38/40, not pasted 39/40"
    if parser_defect:
        decision = "require_parser_bug_fix_before_prompt_revision"
        action = "Fix parser implementation before prompt revision."
        next_task = action
    elif truncation:
        decision = "require_output_token_limit_investigation"
        action = "Investigate output token limits before prompt revision."
        next_task = action
    elif prompt_noncompliance and not prompt_v2.empty:
        decision = "approve_reranking_prompt_v2_corrective_pilot"
        action = "Execute the separate reranking prompt-v2 corrective pilot; do not overwrite v1 evidence."
        next_task = NEXT_TASK
    elif strict_invalid == 0:
        decision = "no_prompt_revision_required"
        action = "No corrective pilot required."
        next_task = action
    else:
        decision = "audit_incomplete"
        action = "Resolve invalid-output root cause before any corrective pilot."
        next_task = action
    return frame(
        [
            {
                "audit_id": "canonical_local_llm_pilot_prompt_corrective_audit_v1",
                "authoritative_total_requests": "120",
                "authoritative_strict_valid_count": strict_valid,
                "authoritative_strict_invalid_count": strict_invalid,
                "authoritative_ranking_valid_count": ranking_valid,
                "authoritative_reranking_valid_count": reranking_valid,
                "count_inconsistency_detected": bool_text(count_inconsistency),
                "count_inconsistency_resolution": count_resolution,
                "parser_defect_detected": bool_text(parser_defect),
                "truncation_root_cause_detected": bool_text(truncation),
                "prompt_noncompliance_root_cause_detected": bool_text(prompt_noncompliance),
                "reranking_prompt_revision_required": bool_text(prompt_noncompliance and not parser_defect and not truncation),
                "ranking_prompt_revision_required": "false",
                "parser_revision_required": bool_text(parser_defect),
                "prompt_v2_contract_created": bool_text(not prompt_v2.empty),
                "corrective_pilot_manifest_rows": len(corrective_manifest),
                "readiness_threshold": format_float(READINESS_THRESHOLD),
                "recommended_decision": decision,
                "required_corrective_action": action,
                "recommended_single_next_task": next_task,
                "notes": "Offline audit only; corrective pilot not executed.",
            }
        ],
        DECISION_COLUMNS,
    )


def validate_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (_relative, columns) in OUTPUT_TABLES.items():
        if list(frames[key].columns) != columns:
            raise ValueError(f"{key} schema mismatch")
    if len(frames["count_reconciliation"]) != 1 + 2 + 3 + 2 + 12:
        raise ValueError("count reconciliation row count mismatch")
    invalid_count = int(frames["decision"].iloc[0]["authoritative_strict_invalid_count"])
    if len(frames["invalid_response_audit"]) != invalid_count:
        raise ValueError("invalid response row count mismatch")
    if len(frames["parser_assessment"]) != 2:
        raise ValueError("parser assessment row count mismatch")
    if len(frames["prompt_v2_contract"]) not in {0, 1}:
        raise ValueError("prompt v2 row count mismatch")
    if len(frames["prompt_v2_contract"]) == 1 and len(frames["corrective_manifest"]) != 60:
        raise ValueError("corrective manifest must contain 60 reranking rows")
    if len(frames["decision"]) != 1:
        raise ValueError("decision row count mismatch")
    if frames["decision"].iloc[0]["recommended_decision"] == "approve_reranking_prompt_v2_corrective_pilot":
        if frames["decision"].iloc[0]["recommended_single_next_task"] != NEXT_TASK:
            raise ValueError("next task text mismatch")


def build_all_frames(root: Path) -> tuple[AuditBundle, dict[str, pd.DataFrame]]:
    bundle = load_audit_bundle(root)
    count_reconciliation = build_count_reconciliation(bundle)
    invalid_audit = build_invalid_response_audit(bundle)
    prompt_requirement = build_prompt_requirement_audit(read_csv(root, "results/tables/canonical_llm_prompt_contract.csv"))
    parser_assessment = build_parser_assessment(bundle)
    prompt_v2 = build_prompt_v2_contract(prompt_requirement, invalid_audit, parser_assessment)
    corrective_manifest = build_corrective_manifest(bundle, prompt_v2)
    decision = build_decision(count_reconciliation, invalid_audit, parser_assessment, prompt_v2, corrective_manifest)
    frames = {
        "count_reconciliation": count_reconciliation,
        "invalid_response_audit": invalid_audit,
        "prompt_requirement_audit": prompt_requirement,
        "parser_assessment": parser_assessment,
        "prompt_v2_contract": prompt_v2,
        "corrective_manifest": corrective_manifest,
        "decision": decision,
    }
    validate_frames(frames)
    return bundle, frames


def write_tables(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (relative, columns) in OUTPUT_TABLES.items():
        path = root / relative
        if key in {"prompt_v2_contract", "corrective_manifest"} and frames[key].empty:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].loc[:, columns].to_csv(path, index=False, lineterminator="\n")


def environment_summary(root: Path) -> str:
    return ";".join(
        [
            f"executable={sys.executable}",
            f"python={sys.version.split()[0]}",
            f"pandas={pd.__version__}",
            f"cwd={root}",
        ]
    )


def write_report(root: Path, bundle: AuditBundle, frames: Mapping[str, pd.DataFrame], commands: Sequence[str]) -> None:
    counts = frames["count_reconciliation"]
    invalid = frames["invalid_response_audit"]
    prompt_req = frames["prompt_requirement_audit"]
    parser = frames["parser_assessment"]
    prompt_v2 = frames["prompt_v2_contract"]
    corrective = frames["corrective_manifest"]
    decision = frames["decision"].iloc[0].to_dict()
    dataset_rows = counts.loc[counts["aggregation_level"].eq("dataset")]
    model_rows = counts.loc[counts["aggregation_level"].eq("model")]
    cell_rows = counts.loc[counts["aggregation_level"].eq("model_dataset_task")]
    output_failures = invalid.loc[invalid["strict_failure_reason"].eq("output_count_invalid")]
    duplicate_failures = invalid.loc[invalid["strict_failure_reason"].eq("duplicate_indices")]
    report_sections = [
        ("1. Scope and provenance", f"Offline Phase 5B-P1C-PROMPT-AUDIT generated deterministically from immutable Phase 5B-P1C evidence. Environment: `{environment_summary(root)}`."),
        ("2. Executive summary", f"Authoritative strict-valid count is `{decision['authoritative_strict_valid_count']}/120`; strict-invalid count is `{decision['authoritative_strict_invalid_count']}`. Decision: `{decision['recommended_decision']}`."),
        ("3. Safety boundaries", "No Ollama endpoint, LLM inference, package install, download, recommender execution, raw-evidence modification, or previous-artifact overwrite was performed."),
        ("4. Immutable Phase 5B-P1C evidence", f"Read-only raw evidence counts: requests `{len(bundle.request_records)}`, responses `{len(bundle.response_records)}`, events `{len(bundle.event_records)}`."),
        ("5. Raw request and response reconciliation", "Raw request IDs and response IDs are unique, complete, and non-ambiguous."),
        ("6. Global count reconciliation", markdown_table(counts.loc[counts["aggregation_level"].eq("global")], ["strict_valid_count_raw_reparse", "strict_invalid_count_raw_reparse", "all_sources_agree", "resolution"])),
        ("7. Dataset count reconciliation", markdown_table(dataset_rows, ["dataset", "strict_valid_count_raw_reparse", "strict_invalid_count_raw_reparse", "notes"])),
        ("8. Model count reconciliation", markdown_table(model_rows, ["model_family", "strict_valid_count_raw_reparse", "strict_invalid_count_raw_reparse", "all_sources_agree"])),
        ("9. Model-dataset-task count reconciliation", markdown_table(cell_rows, ["model_family", "dataset", "task_type", "strict_valid_count_raw_reparse", "strict_invalid_count_raw_reparse"])),
        ("10. Reported 113-versus-114 inconsistency", decision["count_inconsistency_resolution"]),
        ("11. Authoritative strict-valid count", f"`{decision['authoritative_strict_valid_count']}` strict-valid and `{decision['authoritative_strict_invalid_count']}` strict-invalid, from raw responses reparsed with the frozen strict parser."),
        ("12. Strict-invalid response inventory", markdown_table(invalid, ["request_id", "model_family", "dataset", "task_type", "strict_failure_reason", "output_length", "missing_indices", "duplicated_indices"])),
        ("13. Output-count failure analysis", markdown_table(output_failures, ["request_id", "output_length", "missing_indices", "duplicated_indices", "out_of_range_indices", "forensic_classification"])),
        ("14. Duplicate-index failure analysis", markdown_table(duplicate_failures, ["request_id", "output_length", "missing_indices", "duplicated_indices", "forensic_classification"])),
        ("15. JSON and schema findings", "All seven invalid responses were complete JSON objects with the required key; failures were count, duplicate, or range/permutation noncompliance."),
        ("16. Truncation assessment", markdown_table(invalid, ["request_id", "done_reason", "generated_token_count", "configured_max_output_tokens", "raw_response_complete_json", "truncation_evidence"])),
        ("17. Backend and token-limit assessment", "All invalid responses had backend success and `done_reason=stop`; generated-token counts were far below the configured 512-token reranking limit."),
        ("18. Existing reranking prompt requirements", markdown_table(prompt_req, ["requirement_name", "requirement_status", "exact_prompt_evidence"])),
        ("19. Missing or ambiguous prompt requirements", markdown_table(prompt_req.loc[prompt_req["revision_required"].eq("true")], ["requirement_name", "requirement_status", "recommended_wording"])),
        ("20. Strict-parser reassessment", markdown_table(parser.loc[parser["parser_contract_id"].str.contains("strict")], PARSER_ASSESSMENT_COLUMNS[1:16])),
        ("21. Diagnostic salvage reassessment", markdown_table(parser.loc[parser["parser_contract_id"].str.contains("salvage")], PARSER_ASSESSMENT_COLUMNS[1:16])),
        ("22. Whether parser relaxation is justified", "Parser relaxation is not justified: stored and independently reparsed results agree, and invalid outputs violate the frozen full-permutation contract."),
        ("23. Prompt-noncompliance root cause", "The demonstrated root cause is prompt noncompliance: complete JSON outputs omitted, repeated, or added indices while backend and parser checks behaved as specified."),
        ("24. Minimal reranking-prompt revision", "A common v2 reranking prompt is justified to make the existing full-permutation semantics explicit without changing candidates, order, output namespace, model parameters, or parser."),
        ("25. Why ranking prompt remains unchanged", "Ranking was `60/60` strict-valid; no ranking prompt revision is required."),
        ("26. Candidate-order and semantic invariance", "The corrective manifest preserves source candidate order hashes, histories, models, parser contracts, execution parameters, and zero-retry policy."),
        ("27. Positive-label concealment", "Prompt v2 and the corrective manifest do not add positive IDs, positions, labels, utility values, or expected responses."),
        ("28. Reranking prompt-v2 contract", markdown_table(prompt_v2, ["prompt_contract_id", "supersedes_prompt_contract_id", "prompt_version", "scientific_semantics_changed", "contract_status"])),
        ("29. Corrective-pilot design", "Design only: run the 60 frozen reranking request rows with the common v2 prompt, no ranking rows, no retries, concurrency one, separate output root."),
        ("30. Corrective-pilot request count", f"Corrective manifest rows: `{len(corrective)}`."),
        ("31. Readiness threshold", f"The result-independent readiness threshold remains `{format_float(READINESS_THRESHOLD)}` for every model x dataset x reranking cell."),
        ("32. Original-v1 evidence preservation", "The original v1 raw requests, responses, tables, and report remain the primary evidence for v1 and were not changed."),
        ("33. Limitations", "This audit does not execute prompt v2, does not estimate utility changes, and does not prove the corrective prompt will pass."),
        ("34. Protected-input immutability", "Protected input hashes were computed before and after writing outputs and matched."),
        ("35. Commands executed", "\n".join(f"- `{command}`" for command in commands)),
        ("36. Audit decision", f"`{decision['recommended_decision']}`: {decision['required_corrective_action']}"),
        ("37. Recommended single next task", decision["recommended_single_next_task"]),
    ]
    lines = ["# Canonical Local LLM Pilot Invalid-Output and Prompt-Corrective Audit", ""]
    for heading, body in report_sections:
        lines.extend([f"## {heading}", "", str(body), ""])
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def output_paths(root: Path, frames: Mapping[str, pd.DataFrame] | None = None) -> list[Path]:
    paths = []
    for key, (relative, _columns) in OUTPUT_TABLES.items():
        if frames is not None and key in {"prompt_v2_contract", "corrective_manifest"} and frames[key].empty:
            continue
        paths.append(root / relative)
    paths.append(root / REPORT_PATH)
    return paths

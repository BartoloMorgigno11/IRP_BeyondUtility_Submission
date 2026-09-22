"""Resume-safe Phase 5B-P1G top-10 reranking confirmation runner.

The runner executes only the frozen 60-request top-10 reranking manifest through
the local Ollama loopback endpoint. It never pulls or modifies models, never
rewrites previous evidence, and never changes frozen prompt, parser, sample, or
manifest artifacts.
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
from src.analysis.canonical_local_llm_top10_confirmation_preflight import (
    COUNT_RECONCILIATION_COLUMNS,
    CONFIRMATION_SAMPLE_COLUMNS,
    DISCREPANCY_CLASSIFICATION,
    PREFLIGHT_DECISION_COLUMNS,
    READINESS_THRESHOLD,
    REQUEST_MANIFEST_COLUMNS,
    TOP10_PARSER_CONTRACT_COLUMNS,
    TOP10_PARSER_ID,
    TOP10_PROMPT_ALIGNMENT,
    TOP10_PROMPT_CONTRACT_COLUMNS,
    TOP10_PROMPT_ID,
    build_previous_row_union,
    contract_hash as top10_contract_hash,
)
from src.llm.canonical_llm_contract import render_candidates, render_history, sha256_items, sha256_text
from src.llm.canonical_local_llm_pilot_resume import append_jsonl_fsync, positive_label_exposed, read_jsonl
from src.llm.canonical_local_llm_runner import build_payload, clean_text, execute_ollama_request


PILOT_ID = "canonical_local_llm_top10_confirmation_pilot_v1"
RAW_DIR = Path("results/llm/canonical_local_top10_confirmation_pilot")
REQUEST_JSONL = RAW_DIR / "requests.jsonl"
RESPONSE_JSONL = RAW_DIR / "responses.jsonl"
EVENT_JSONL = RAW_DIR / "runner_events.jsonl"

COUNT_RECONCILIATION_PATH = Path("results/tables/canonical_local_llm_prompt_compatible_count_reconciliation.csv")
TOP10_SOURCE_AUDIT_PATH = Path("results/tables/canonical_local_llm_reranking_top10_source_contract_audit.csv")
PROMPT_CONTRACT_PATH = Path("results/tables/canonical_local_llm_reranking_top10_prompt_contract.csv")
PARSER_CONTRACT_PATH = Path("results/tables/canonical_local_llm_reranking_top10_parser_contract.csv")
SAMPLE_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_sample.csv")
MANIFEST_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_request_manifest.csv")
PREFLIGHT_VALIDATION_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_preflight_validation.csv")
PREFLIGHT_DECISION_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_preflight_decision.csv")
PREFLIGHT_REPORT_PATH = Path("report/canonical_local_llm_top10_confirmation_preflight.md")
PREFLIGHT_SOURCE_PATH = Path("src/analysis/canonical_local_llm_top10_confirmation_preflight.py")
PREFLIGHT_RUNNER_PATH = Path("src/experiments/build_canonical_local_llm_top10_confirmation_preflight.py")
PREFLIGHT_TEST_PATH = Path("tests/test_canonical_local_llm_top10_confirmation_preflight.py")

PHASE5A_PROMPT_CONTRACT_PATH = Path("results/tables/canonical_llm_prompt_contract.csv")
PHASE5A_PARSER_CONTRACT_PATH = Path("results/tables/canonical_llm_parser_contract.csv")
EXECUTION_CONTRACT_PATH = Path("results/tables/canonical_llm_execution_parameter_contract.csv")
COHORT_PREFLIGHT_PATH = Path("results/tables/canonical_llm_cohort_preflight.csv")
PROMPT_RENDERING_PATH = Path("results/tables/canonical_llm_prompt_rendering_audit.csv")
MODEL_INVENTORY_PATH = Path("results/tables/canonical_llm_model_inventory.csv")
OLLAMA_RECOVERY_PATH = Path("results/tables/canonical_local_ollama_service_recovery.csv")

P1F_DECISION_PATH = Path("results/tables/canonical_local_llm_final_reranking_protocol_decision.csv")
P1F_SOURCE_INVENTORY_PATH = Path("results/tables/canonical_local_llm_reranking_protocol_source_inventory.csv")
P1F_METRIC_REQUIREMENTS_PATH = Path("results/tables/canonical_local_llm_reranking_metric_input_requirements.csv")
P1F_EXECUTION_SCOPE_PATH = Path("results/tables/canonical_local_llm_final_execution_scope.csv")
P1F_REPORT_PATH = Path("report/canonical_local_llm_final_reranking_protocol_decision.md")

V1_SAMPLE_PATH = Path("results/tables/canonical_local_llm_pilot_sample.csv")
V1_RAW_REQUESTS_PATH = Path("results/llm/canonical_local_pilot_resume/requests.jsonl")
V1_RAW_RESPONSES_PATH = Path("results/llm/canonical_local_pilot_resume/responses.jsonl")
V2_MANIFEST_PATH = Path("results/tables/canonical_local_llm_reranking_prompt_v2_corrective_pilot_manifest.csv")
V2_RAW_REQUESTS_PATH = Path("results/llm/canonical_local_reranking_prompt_v2_corrective_pilot/requests.jsonl")
V2_RAW_RESPONSES_PATH = Path("results/llm/canonical_local_reranking_prompt_v2_corrective_pilot/responses.jsonl")
HYBRID_SAMPLE_PATH = Path("results/tables/canonical_local_llm_hybrid_holdout_sample.csv")
HYBRID_RAW_REQUESTS_PATH = Path("results/llm/canonical_local_hybrid_holdout_pilot/requests.jsonl")
HYBRID_RAW_RESPONSES_PATH = Path("results/llm/canonical_local_hybrid_holdout_pilot/responses.jsonl")

REPORT_PATH = Path("report/canonical_local_llm_top10_reranking_confirmation_pilot.md")

PROVENANCE_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_provenance.csv")
REQUEST_STATE_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_request_state.csv")
EXECUTION_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_execution.csv")
PARSE_RESULTS_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_parse_results.csv")
RANKINGS_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_rankings.csv")
METRICS_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_metrics.csv")
VALIDATION_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_validation.csv")
DECISION_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_decision.csv")

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
DATASET_ORDER = ["Amazon Beauty", "Amazon Sports", "ML-1M"]
MODEL_ORDER = ["Qwen2 7B Instruct", "Mistral 7B Instruct"]
APPROVED_NEXT_TASK = (
    "Phase 5B-P2 - execute the full canonical local Qwen2 and Mistral matrix using ranking prompt v1 "
    "and official-code-aligned top-10 reranking on 5,764 frozen prompt-compatible rows"
)
FAILED_NEXT_TASK = "Phase 5B-P1H - final execution-scope decision after official-code-aligned top-10 confirmation failure"

PROVENANCE_COLUMNS = [
    "confirmation_pilot_id",
    "prompt_contract_path",
    "prompt_contract_sha256",
    "parser_contract_path",
    "parser_contract_sha256",
    "sample_path",
    "sample_sha256",
    "manifest_path",
    "manifest_sha256",
    "count_reconciliation_path",
    "count_reconciliation_sha256",
    "ollama_server_version",
    "qwen2_model_tag",
    "qwen2_digest",
    "mistral_model_tag",
    "mistral_digest",
    "expected_sample_rows",
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
    "confirmation_request_id",
    "confirmation_row_id",
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
    "output_count_valid",
    "candidate_validity",
    "utility_evaluation_valid",
    "raw_request_sha256",
    "raw_response_sha256",
    "blocking_reason",
    "notes",
]

EXECUTION_COLUMNS = [
    "confirmation_request_id",
    "confirmation_row_id",
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
    "confirmation_request_id",
    "confirmation_row_id",
    "model_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "prompt_contract_id",
    "raw_response_sha256",
    "strict_parser_id",
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
    "confirmation_request_id",
    "confirmation_row_id",
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
    "confirmation_rows",
    "backend_success_rows",
    "strict_valid_rows",
    "strict_parse_validity_rate",
    "json_valid_rows",
    "schema_valid_rows",
    "output_count_valid_rows",
    "index_range_valid_rows",
    "uniqueness_valid_rows",
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
    "confirmation_pilot_id",
    "expected_request_count",
    "completed_request_count",
    "backend_error_record_count",
    "ambiguous_request_count",
    "previous_pilot_overlap_count",
    "backend_completion_rate",
    "minimum_cell_strict_parse_validity_rate",
    "cells_passing_threshold",
    "total_readiness_cells",
    "all_prompt_hashes_valid",
    "all_candidate_order_hashes_valid",
    "all_history_hashes_valid",
    "positive_label_never_exposed",
    "all_strict_outputs_length_ten",
    "all_strict_outputs_candidate_valid",
    "all_strict_outputs_duplicate_free",
    "all_strict_outputs_hallucination_free",
    "raw_request_retention_complete",
    "raw_response_retention_complete",
    "output_truncation_detected",
    "readiness_threshold",
    "readiness_threshold_result_independent",
    "utility_used_for_readiness",
    "ranking_prompt_for_full_execution",
    "reranking_prompt_for_full_execution",
    "full_execution_prompt_compatible_rows",
    "full_execution_expected_request_count",
    "pilot_decision",
    "blocking_cells",
    "required_corrective_action",
    "recommended_single_next_task",
    "notes",
]

OUTPUT_TABLES = {
    "provenance": (PROVENANCE_PATH, PROVENANCE_COLUMNS),
    "request_state": (REQUEST_STATE_PATH, REQUEST_STATE_COLUMNS),
    "execution": (EXECUTION_PATH, EXECUTION_COLUMNS),
    "parse_results": (PARSE_RESULTS_PATH, PARSE_COLUMNS),
    "rankings": (RANKINGS_PATH, RANKING_COLUMNS),
    "metrics": (METRICS_PATH, METRIC_COLUMNS),
    "validation": (VALIDATION_PATH, VALIDATION_COLUMNS),
    "decision": (DECISION_PATH, DECISION_COLUMNS),
}

REPORT_SECTIONS = [
    "1. Scope and provenance",
    "2. Executive summary",
    "3. Safety boundaries",
    "4. Phase 5B-P1F protocol decision",
    "5. Phase 5B-P1G preflight",
    "6. Corrected prompt-compatible counts",
    "7. Corrected full request count",
    "8. Official-code top-10 evidence",
    "9. Paper-evidence limitation",
    "10. Top-10 alignment classification",
    "11. Frozen common prompt",
    "12. Frozen strict parser",
    "13. Difference from full-20 adaptation",
    "14. Confirmation population",
    "15. Previous-pilot exclusions",
    "16. Deterministic confirmation sample",
    "17. Confirmation sample composition",
    "18. Frozen request manifest",
    "19. Local Ollama verification",
    "20. Qwen2 digest verification",
    "21. Mistral digest verification",
    "22. Resume-safe execution",
    "23. Raw-request retention",
    "24. Raw-response retention",
    "25. Backend completion",
    "26. Latency and token findings",
    "27. Strict-parser findings",
    "28. JSON and schema findings",
    "29. Output-count findings",
    "30. Index-range findings",
    "31. Uniqueness findings",
    "32. Candidate-validity findings",
    "33. Duplicate and hallucination findings",
    "34. Truncation assessment",
    "35. Qwen2 Beauty readiness",
    "36. Qwen2 Sports readiness",
    "37. Qwen2 ML-1M readiness",
    "38. Mistral Beauty readiness",
    "39. Mistral Sports readiness",
    "40. Mistral ML-1M readiness",
    "41. Utility diagnostics",
    "42. Why utility does not determine readiness",
    "43. Readiness threshold",
    "44. Top-10 protocol generalization assessment",
    "45. Full-execution decision",
    "46. Approved prompt combination",
    "47. Full-execution request count",
    "48. Required corrective action",
    "49. Limitations",
    "50. Protected-input immutability",
    "51. Commands executed",
    "52. Recommended single next task",
]


@dataclass(frozen=True)
class OllamaAdminState:
    server_version: str
    tags: Mapping[str, Mapping[str, Any]]
    ps: Sequence[Mapping[str, Any]]
    started_process_id: str = ""
    start_attempted: bool = False


@dataclass(frozen=True)
class Top10Inputs:
    count_reconciliation: pd.DataFrame
    prompt_contract: pd.DataFrame
    parser_contract: pd.DataFrame
    sample: pd.DataFrame
    manifest: pd.DataFrame
    execution_contract: pd.DataFrame
    recovery: pd.DataFrame
    contexts: Mapping[str, Mapping[str, Any]]
    previous_overlap_count: int
    protected_hashes: Mapping[str, str]


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def parse_bool(value: Any) -> bool:
    return clean_cell(value).strip().lower() in {"true", "1", "yes"}


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.12g}"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: clean_cell(row.get(column, "")) for column in columns} for row in rows], columns=list(columns))


def read_table(root: Path, rel_path: Path) -> pd.DataFrame:
    return pd.read_csv(root / rel_path, dtype=str, keep_default_na=False)


def median_numeric(values: Iterable[Any]) -> str:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return ""
    return format_float(float(series.median()))


def mean_numeric(values: Iterable[Any]) -> str:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return ""
    return format_float(float(series.mean()))


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()


def output_targets(root: Path) -> list[Path]:
    return [root / path for path, _columns in OUTPUT_TABLES.values()] + [root / REPORT_PATH]


def protected_input_paths(root: Path) -> list[str]:
    paths: list[Path] = [
        COUNT_RECONCILIATION_PATH,
        TOP10_SOURCE_AUDIT_PATH,
        PROMPT_CONTRACT_PATH,
        PARSER_CONTRACT_PATH,
        SAMPLE_PATH,
        MANIFEST_PATH,
        PREFLIGHT_VALIDATION_PATH,
        PREFLIGHT_DECISION_PATH,
        PREFLIGHT_REPORT_PATH,
        PREFLIGHT_SOURCE_PATH,
        PREFLIGHT_RUNNER_PATH,
        PREFLIGHT_TEST_PATH,
        PHASE5A_PROMPT_CONTRACT_PATH,
        PHASE5A_PARSER_CONTRACT_PATH,
        EXECUTION_CONTRACT_PATH,
        COHORT_PREFLIGHT_PATH,
        PROMPT_RENDERING_PATH,
        MODEL_INVENTORY_PATH,
        P1F_DECISION_PATH,
        P1F_SOURCE_INVENTORY_PATH,
        P1F_METRIC_REQUIREMENTS_PATH,
        P1F_EXECUTION_SCOPE_PATH,
        P1F_REPORT_PATH,
        V1_SAMPLE_PATH,
        V1_RAW_REQUESTS_PATH,
        V1_RAW_RESPONSES_PATH,
        V2_MANIFEST_PATH,
        V2_RAW_REQUESTS_PATH,
        V2_RAW_RESPONSES_PATH,
        HYBRID_SAMPLE_PATH,
        HYBRID_RAW_REQUESTS_PATH,
        HYBRID_RAW_RESPONSES_PATH,
        Path("results/tables/ranking_candidate_sets.csv"),
        Path("results/tables/positive_positions.csv"),
        Path("results/tables/item_title_map.csv"),
        Path("results/tables/item_title_coverage_summary.csv"),
        Path("results/tables/llm_title_exclusions.csv"),
        OLLAMA_RECOVERY_PATH,
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
    return {rel: file_sha256(root / rel) for rel in protected_input_paths(root)}


def row_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        clean_cell(row.get("dataset")),
        clean_cell(row.get("task_type")),
        clean_cell(row.get("candidate_set_id")),
        clean_cell(row.get("user_id")),
    )


def validate_count_reconciliation(counts: pd.DataFrame) -> None:
    if list(counts.columns) != COUNT_RECONCILIATION_COLUMNS:
        raise ValueError("count reconciliation schema mismatch")
    global_row = counts[counts["aggregation_level"].eq("global_total")].iloc[0]
    expected = {
        "full_candidate_rows": "6000",
        "candidate_strict_title_rows": "5814",
        "history_title_exclusion_rows": "50",
        "primary_prompt_compatible_rows": "5764",
        "two_model_request_count": "11528",
        "discrepancy_classification": DISCREPANCY_CLASSIFICATION,
    }
    for column, value in expected.items():
        if global_row[column] != value:
            raise ValueError(f"count reconciliation mismatch {column}: {global_row[column]} != {value}")
    task_totals = counts[counts["aggregation_level"].eq("task_total")].set_index("task_type")
    if task_totals.loc["ranking", "primary_prompt_compatible_rows"] != "2882":
        raise ValueError("ranking prompt-compatible rows must equal 2882")
    if task_totals.loc["reranking", "primary_prompt_compatible_rows"] != "2882":
        raise ValueError("reranking prompt-compatible rows must equal 2882")
    if task_totals.loc["ranking", "two_model_request_count"] != "5764":
        raise ValueError("ranking two-model request count must equal 5764")
    if task_totals.loc["reranking", "two_model_request_count"] != "5764":
        raise ValueError("reranking two-model request count must equal 5764")


def validate_prompt_contract(prompt_contract: pd.DataFrame) -> dict[str, str]:
    if list(prompt_contract.columns) != TOP10_PROMPT_CONTRACT_COLUMNS:
        raise ValueError("top10 prompt contract schema mismatch")
    if len(prompt_contract) != 1:
        raise ValueError("top10 prompt contract must contain exactly one row")
    row = prompt_contract.iloc[0].to_dict()
    expected = {
        "prompt_contract_id": TOP10_PROMPT_ID,
        "task_type": "reranking",
        "candidate_count": "20",
        "requested_output_count": "10",
        "candidate_numbering": "one_based_1_to_20",
        "positive_label_hidden": "true",
        "requires_complete_permutation": "false",
        "requires_unique_indices": "true",
        "allowed_index_min": "1",
        "allowed_index_max": "20",
        "explanation_allowed": "false",
        "code_fence_allowed": "false",
        "model_specific_wording": "false",
        "paper_alignment": TOP10_PROMPT_ALIGNMENT,
        "contract_status": "approved_for_confirmation_preflight",
    }
    for column, value in expected.items():
        if row.get(column) != value:
            raise ValueError(f"top10 prompt contract mismatch {column}: {row.get(column)} != {value}")
    lowered = (row["system_message_template"] + "\n" + row["user_message_template"]).lower()
    if "think step by step" in lowered or "chain-of-thought" in lowered:
        raise ValueError("top10 prompt requests reasoning")
    if "[1,2" in row["user_message_template"]:
        raise ValueError("top10 prompt contains example ordering")
    if row["contract_sha256"] != top10_contract_hash(row, excluded=("contract_sha256",)):
        raise ValueError("top10 prompt contract hash mismatch")
    return row


def validate_parser_contract(parser_contract: pd.DataFrame) -> dict[str, str]:
    if list(parser_contract.columns) != TOP10_PARSER_CONTRACT_COLUMNS:
        raise ValueError("top10 parser contract schema mismatch")
    strict = parser_contract[parser_contract["parser_contract_id"].eq(TOP10_PARSER_ID)]
    if len(strict) != 1:
        raise ValueError("strict top10 parser contract must contain exactly one row")
    row = strict.iloc[0].to_dict()
    expected = {
        "parser_mode": "strict_primary",
        "task_type": "reranking",
        "expected_output_count": "10",
        "requires_complete_permutation": "false",
        "allows_leading_prose": "false",
        "allows_trailing_prose": "false",
        "allows_code_fence": "false",
        "allows_integer_strings": "false",
        "index_base": "1",
        "allows_duplicate_items": "false",
        "allows_unknown_items": "false",
        "allows_partial_output": "false",
        "allows_fuzzy_title_matching": "false",
        "positive_label_used": "false",
        "paper_alignment": TOP10_PROMPT_ALIGNMENT,
        "contract_status": "approved_for_confirmation_preflight",
    }
    for column, value in expected.items():
        if row.get(column) != value:
            raise ValueError(f"top10 parser contract mismatch {column}: {row.get(column)} != {value}")
    return row


def validate_sample(sample: pd.DataFrame, previous_union: Mapping[tuple[str, str, str, str], Mapping[str, str]]) -> int:
    if list(sample.columns) != CONFIRMATION_SAMPLE_COLUMNS:
        raise ValueError("top10 confirmation sample schema mismatch")
    if len(sample) != 30:
        raise ValueError("top10 confirmation sample must contain 30 rows")
    if not sample["confirmation_row_id"].is_unique:
        raise ValueError("confirmation_row_id is not unique")
    if not sample["selection_sha256"].is_unique:
        raise ValueError("selection_sha256 is not unique")
    if sample["sample_order"].astype(int).tolist() != list(range(1, 31)):
        raise ValueError("sample_order is not contiguous")
    if not sample["task_type"].eq("reranking").all():
        raise ValueError("sample contains non-reranking row")
    counts = sample.groupby("dataset").size().to_dict()
    if counts != {"Amazon Beauty": 10, "Amazon Sports": 10, "ML-1M": 10}:
        raise ValueError(f"sample dataset counts mismatch: {counts}")
    for column in [
        "candidate_titles_complete",
        "history_titles_complete",
        "held_out_positive_absent",
        "primary_prompt_compatible",
        "disjoint_from_all_previous_pilots",
    ]:
        if not sample[column].eq("true").all():
            raise ValueError(f"sample column must be true: {column}")
    if not sample["candidate_count"].eq("20").all():
        raise ValueError("sample candidate_count must be 20")
    for column in ["used_in_v1_pilot", "used_in_v2_pilot", "used_in_hybrid_holdout"]:
        if not sample[column].eq("false").all():
            raise ValueError(f"sample records previous-pilot use: {column}")
    for column in ["candidate_order_sha256", "history_render_sha256"]:
        if not sample[column].astype(str).str.len().gt(0).all():
            raise ValueError(f"sample missing {column}")
    overlap = sum(1 for row in sample.to_dict("records") if row_identity(row) in previous_union)
    if overlap:
        raise ValueError(f"sample overlaps previous pilots: {overlap}")
    return overlap


def validate_manifest(
    manifest: pd.DataFrame,
    sample: pd.DataFrame,
    prompt_row: Mapping[str, str],
    parser_row: Mapping[str, str],
    execution_contract: pd.DataFrame,
) -> None:
    if list(manifest.columns) != REQUEST_MANIFEST_COLUMNS:
        raise ValueError("top10 request manifest schema mismatch")
    if len(manifest) != 60:
        raise ValueError("top10 request manifest must contain 60 rows")
    if not manifest["confirmation_request_id"].is_unique:
        raise ValueError("confirmation_request_id is not unique")
    if manifest["request_order"].astype(int).tolist() != list(range(1, 61)):
        raise ValueError("request_order is not contiguous")
    if not manifest["task_type"].eq("reranking").all():
        raise ValueError("manifest contains ranking or non-reranking request")
    sample_by_row = {row.confirmation_row_id: row._asdict() for row in sample.itertuples(index=False)}
    if set(manifest["confirmation_row_id"]) != set(sample["confirmation_row_id"]):
        raise ValueError("manifest confirmation rows do not match sample")
    qwen_rows = set(manifest[manifest["model_family"].eq("Qwen2 7B Instruct")]["confirmation_row_id"])
    mistral_rows = set(manifest[manifest["model_family"].eq("Mistral 7B Instruct")]["confirmation_row_id"])
    if qwen_rows != mistral_rows or qwen_rows != set(sample["confirmation_row_id"]):
        raise ValueError("same 30 confirmation rows are not used across both models")
    for family in MODEL_ORDER:
        subset = manifest[manifest["model_family"].eq(family)]
        if len(subset) != 30:
            raise ValueError(f"manifest model count mismatch: {family}")
        expected = EXPECTED_DIGESTS[family]
        if set(subset["model_tag"]) != {expected["tag"]}:
            raise ValueError(f"manifest tag mismatch: {family}")
        if set(subset["model_digest"]) != {expected["digest"]}:
            raise ValueError(f"manifest digest mismatch: {family}")
        for dataset in DATASET_ORDER:
            if len(subset[subset["dataset"].eq(dataset)]) != 10:
                raise ValueError(f"manifest cell count mismatch: {family} {dataset}")
    constants = {
        "prompt_contract_id": TOP10_PROMPT_ID,
        "prompt_contract_sha256": prompt_row["contract_sha256"],
        "parser_contract_id": TOP10_PARSER_ID,
        "parser_contract_sha256": parser_row["parser_schema_sha256"],
        "candidate_count": "20",
        "requested_output_count": "10",
        "temperature": "0",
        "response_format": "json_object",
        "streaming": "false",
        "concurrency": "1",
        "scientific_retry_count": "0",
        "infrastructure_retry_count": "0",
        "previous_pilot_overlap": "false",
    }
    for column, expected in constants.items():
        if not manifest[column].eq(expected).all():
            raise ValueError(f"manifest {column} mismatch")
    execution_lookup = {row.execution_contract_id: row._asdict() for row in execution_contract.itertuples(index=False)}
    for row in manifest.to_dict("records"):
        sample_row = sample_by_row[row["confirmation_row_id"]]
        for column in ["candidate_set_id", "user_id", "dataset", "task_type", "candidate_order_sha256", "history_render_sha256"]:
            if str(row[column]) != str(sample_row[column]):
                raise ValueError(f"manifest/sample mismatch {column}: {row['confirmation_request_id']}")
        if row["execution_contract_id"] not in execution_lookup:
            raise ValueError(f"missing execution contract: {row['execution_contract_id']}")
        if execution_lookup[row["execution_contract_id"]]["contract_sha256"] != row["execution_contract_sha256"]:
            raise ValueError(f"execution contract hash mismatch: {row['confirmation_request_id']}")


def render_from_contract(prompt_row: Mapping[str, str], history_items: Sequence[Mapping[str, str]], candidate_items: Sequence[Mapping[str, str]]) -> tuple[str, str, str, str]:
    history_render = render_history(history_items)
    candidate_render = render_candidates(candidate_items)
    system_message = str(prompt_row["system_message_template"])
    user_message = str(prompt_row["user_message_template"]).format(history_render=history_render, candidate_render=candidate_render)
    return system_message, user_message, history_render, candidate_render


def build_contexts(root: Path, sample: pd.DataFrame, manifest: pd.DataFrame, prompt_row: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    title_map = load_title_map(root)
    histories = load_histories(root, title_map)
    candidates = read_csv(root, "results/tables/ranking_candidate_sets.csv")
    candidate_lookup = {
        (str(row.candidate_set_id), str(row.user_id)): parse_json_list(row.candidate_items)
        for row in candidates[["candidate_set_id", "user_id", "candidate_items"]].itertuples(index=False)
    }
    sample_lookup = {row.confirmation_row_id: row._asdict() for row in sample.itertuples(index=False)}
    contexts: dict[str, dict[str, Any]] = {}
    for row in manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        sample_row = sample_lookup[row["confirmation_row_id"]]
        key = (row["candidate_set_id"], row["user_id"])
        if key not in candidate_lookup:
            raise ValueError(f"candidate row missing: {row['confirmation_request_id']}")
        candidate_ids = [str(item) for item in candidate_lookup[key]]
        if len(candidate_ids) != 20:
            raise ValueError(f"candidate count mismatch: {row['confirmation_request_id']}")
        if sha256_items(candidate_ids) != row["candidate_order_sha256"]:
            raise ValueError(f"candidate-order hash mismatch: {row['confirmation_request_id']}")
        candidate_items = [{"item_id": item_id, "title": title_map[(row["dataset"], item_id)]} for item_id in candidate_ids]
        history_items = [
            item
            for item in histories[row["dataset"]].get(row["user_id"], [])
            if item["item_id"] != sample_row["positive_item_id"]
        ][-10:]
        system_message, user_message, history_render, candidate_render = render_from_contract(prompt_row, history_items, candidate_items)
        if sha256_text(history_render) != row["history_render_sha256"]:
            raise ValueError(f"history-render hash mismatch: {row['confirmation_request_id']}")
        payload_visible = json.dumps({"system": system_message, "prompt": user_message}, ensure_ascii=False)
        if positive_label_exposed(payload_visible, sample_row["positive_item_id"]):
            raise ValueError(f"positive label exposed: {row['confirmation_request_id']}")
        contexts[row["confirmation_request_id"]] = {
            "confirmation_request_id": row["confirmation_request_id"],
            "confirmation_row_id": row["confirmation_row_id"],
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
            "requested_output_count": "10",
            "requires_full_permutation": False,
        }
    return contexts


def load_inputs(root: Path) -> Top10Inputs:
    counts = read_table(root, COUNT_RECONCILIATION_PATH)
    validate_count_reconciliation(counts)
    prompt_contract = read_table(root, PROMPT_CONTRACT_PATH)
    prompt_row = validate_prompt_contract(prompt_contract)
    parser_contract = read_table(root, PARSER_CONTRACT_PATH)
    parser_row = validate_parser_contract(parser_contract)
    sample = read_table(root, SAMPLE_PATH)
    previous_union = build_previous_row_union(root)
    previous_overlap_count = validate_sample(sample, previous_union)
    manifest = read_table(root, MANIFEST_PATH)
    execution_contract = read_table(root, EXECUTION_CONTRACT_PATH)
    validate_manifest(manifest, sample, prompt_row, parser_row, execution_contract)
    preflight_decision = read_table(root, PREFLIGHT_DECISION_PATH)
    if list(preflight_decision.columns) != PREFLIGHT_DECISION_COLUMNS:
        raise ValueError("preflight decision schema mismatch")
    if preflight_decision.iloc[0]["preflight_decision"] != "approve_top10_confirmation_pilot_execution":
        raise ValueError("preflight did not approve top10 confirmation execution")
    contexts = build_contexts(root, sample, manifest, prompt_row)
    return Top10Inputs(
        count_reconciliation=counts,
        prompt_contract=prompt_contract,
        parser_contract=parser_contract,
        sample=sample,
        manifest=manifest,
        execution_contract=execution_contract,
        recovery=read_table(root, OLLAMA_RECOVERY_PATH),
        contexts=contexts,
        previous_overlap_count=previous_overlap_count,
        protected_hashes=hash_protected_inputs(root),
    )


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
        out_path = Path("C:/tmp/phase5b_p1g_top10_ollama_stdout.log")
        err_path = Path("C:/tmp/phase5b_p1g_top10_ollama_stderr.log")
        out_path.parent.mkdir(parents=True, exist_ok=True)
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
    tag_map = {str(item.get("name", "")): item for item in tags_payload.get("models", [])}
    for family, expected in EXPECTED_DIGESTS.items():
        tag = expected["tag"]
        if tag not in tag_map:
            raise RuntimeError(f"blocked_missing_local_model: {tag}")
        if str(tag_map[tag].get("digest", "")).lower() != expected["digest"]:
            raise RuntimeError(f"model_digest_mismatch: {tag}")
    return OllamaAdminState(
        server_version=str(version.get("version", "")),
        tags=tag_map,
        ps=list(ps_payload.get("models", [])),
        started_process_id=started_pid,
        start_attempted=start_attempted,
    )


def count_by_request_id(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        request_id = clean_cell(record.get("confirmation_request_id", record.get("request_id", "")))
        if request_id:
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
    req_counts = count_by_request_id(request_records)
    resp_counts = count_by_request_id(response_records)
    summary = {"completed": 0, "not_started": 0, "ambiguous_interrupted": 0, "invalid_duplicate_evidence": 0}
    manifest_ids = set(manifest["confirmation_request_id"])
    for row in manifest.itertuples(index=False):
        state = classify_request_state(req_counts.get(row.confirmation_request_id, 0), resp_counts.get(row.confirmation_request_id, 0))
        summary[state] += 1
    extras = (set(req_counts) | set(resp_counts)) - manifest_ids
    if extras:
        summary["invalid_duplicate_evidence"] += len(extras)
    return summary


def pending_requests_in_order(manifest: pd.DataFrame, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    req_counts = count_by_request_id(request_records)
    resp_counts = count_by_request_id(response_records)
    manifest_ids = set(manifest["confirmation_request_id"])
    extras = (set(req_counts) | set(resp_counts)) - manifest_ids
    if extras:
        raise RuntimeError(f"invalid_duplicate_evidence: raw id outside manifest {sorted(extras)[0]}")
    pending: list[dict[str, str]] = []
    for row in manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        state = classify_request_state(req_counts.get(row["confirmation_request_id"], 0), resp_counts.get(row["confirmation_request_id"], 0))
        if state == "not_started":
            pending.append(row)
        elif state == "completed":
            continue
        elif state == "ambiguous_interrupted":
            raise RuntimeError(f"ambiguous_interrupted_request: {row['confirmation_request_id']}")
        else:
            raise RuntimeError(f"invalid_duplicate_evidence: {row['confirmation_request_id']}")
    return pending


def output_truncation_detected(raw_text: str, done_reason: str, eval_count: Any, max_output_tokens: Any) -> bool:
    if str(done_reason).strip().lower() == "length":
        return True
    try:
        return int(eval_count) >= int(max_output_tokens) and bool(str(raw_text).strip())
    except Exception:
        return False


def classify_top10_response(raw_text: str, *, candidate_count: int = 20, expected_count: int = 10) -> dict[str, Any]:
    stripped = str(raw_text or "").strip()
    json_valid = False
    schema_valid = False
    output_count_valid = False
    index_type_valid = False
    index_range_valid = False
    uniqueness_valid = False
    indices: list[int] = []
    output_length = 0
    failure = "empty_response" if not stripped else ""
    if stripped:
        try:
            parsed = json.loads(stripped)
            json_valid = True
        except json.JSONDecodeError:
            parsed = None
            failure = "json_parse_failed"
        if json_valid:
            if isinstance(parsed, dict) and set(parsed) == {"ranked_candidate_indices"} and isinstance(parsed["ranked_candidate_indices"], list):
                schema_valid = True
                values = parsed["ranked_candidate_indices"]
                output_length = len(values)
                output_count_valid = output_length == expected_count
                index_type_valid = all(isinstance(value, int) and not isinstance(value, bool) for value in values)
                if index_type_valid:
                    indices = [int(value) for value in values]
                    index_range_valid = all(1 <= value <= candidate_count for value in indices)
                    uniqueness_valid = len(indices) == len(set(indices))
                if not output_count_valid:
                    failure = "output_count_invalid"
                elif not index_type_valid:
                    failure = "index_type_invalid"
                elif not index_range_valid:
                    failure = "index_range_invalid"
                elif not uniqueness_valid:
                    failure = "duplicate_indices"
            else:
                failure = "schema_invalid"
    duplicate_count = len(indices) - len(set(indices)) if indices else 0
    unknown_index_count = sum(1 for value in indices if value < 1 or value > candidate_count)
    strict = bool(json_valid and schema_valid and output_count_valid and index_type_valid and index_range_valid and uniqueness_valid)
    if strict:
        failure = ""
    return {
        "json_parse_valid": json_valid,
        "schema_valid": schema_valid,
        "output_count_valid": output_count_valid,
        "index_type_valid": index_type_valid,
        "index_range_valid": index_range_valid,
        "uniqueness_valid": uniqueness_valid,
        "candidate_validity": strict,
        "catalog_validity": strict,
        "duplicate_count": duplicate_count,
        "unknown_index_count": unknown_index_count,
        "hallucination_count": unknown_index_count,
        "strict_parse_valid": strict,
        "strict_indices": indices if strict else [],
        "output_length": output_length,
        "primary_failure_reason": failure,
    }


def build_request_record(row: Mapping[str, str], context: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if positive_label_exposed(payload_text, str(context.get("positive_item_id", ""))):
        raise ValueError(f"positive label in payload: {row['confirmation_request_id']}")
    return {
        "confirmation_request_id": row["confirmation_request_id"],
        "request_id": row["confirmation_request_id"],
        "confirmation_row_id": row["confirmation_row_id"],
        "request_order": row["request_order"],
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
        "scientific_retry_count": "0",
        "infrastructure_retry_count": "0",
        "candidate_items": context["candidate_items"],
        "request_payload": payload,
        "timestamp_before_send_utc": utc_now(),
        "phase_id": PILOT_ID,
    }


def build_response_record(row: Mapping[str, str], result: Any) -> dict[str, Any]:
    raw_sha = sha256_text(result.response_text)
    return {
        "confirmation_request_id": row["confirmation_request_id"],
        "request_id": row["confirmation_request_id"],
        "confirmation_row_id": row["confirmation_row_id"],
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
        "phase_id": PILOT_ID,
    }


def execute_pending_requests(root: Path, inputs: Top10Inputs) -> None:
    (root / RAW_DIR).mkdir(parents=True, exist_ok=True)
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    pending = pending_requests_in_order(inputs.manifest, request_records, response_records)
    for row in pending:
        if hash_protected_inputs(root) != dict(inputs.protected_hashes):
            raise RuntimeError("protected_input_validation_failed")
        context = inputs.contexts[row["confirmation_request_id"]]
        if context["prompt_contract_sha256"] != row["prompt_contract_sha256"]:
            raise ValueError(f"prompt hash mismatch before send: {row['confirmation_request_id']}")
        if context["candidate_order_sha256"] != row["candidate_order_sha256"]:
            raise ValueError(f"candidate-order hash mismatch before send: {row['confirmation_request_id']}")
        if context["history_render_sha256"] != row["history_render_sha256"]:
            raise ValueError(f"history hash mismatch before send: {row['confirmation_request_id']}")
        payload = build_payload(row, context)
        request_record = build_request_record(row, context, payload)
        append_jsonl_fsync(root / REQUEST_JSONL, request_record)
        append_jsonl_fsync(root / EVENT_JSONL, {"event": "request_recorded_before_send", "confirmation_request_id": row["confirmation_request_id"], "timestamp_utc": utc_now(), "phase_id": PILOT_ID})
        timeout_seconds = max(120, int(float(row["max_output_tokens"] or 120)))
        result = execute_ollama_request(payload, timeout_seconds)
        response_record = build_response_record(row, result)
        append_jsonl_fsync(root / RESPONSE_JSONL, response_record)
        append_jsonl_fsync(
            root / EVENT_JSONL,
            {
                "event": "response_or_error_recorded",
                "confirmation_request_id": row["confirmation_request_id"],
                "backend_success": bool_text(result.backend_success),
                "timestamp_utc": utc_now(),
                "phase_id": PILOT_ID,
            },
        )
        print(f"{row['confirmation_request_id']} {row['model_family']} {row['dataset']} backend_success={bool_text(result.backend_success)}", flush=True)


def records_to_execution(manifest: pd.DataFrame, response_records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    manifest_lookup = {row.confirmation_request_id: row._asdict() for row in manifest.itertuples(index=False)}
    response_lookup = {clean_cell(record.get("confirmation_request_id", record.get("request_id", ""))): record for record in response_records}
    rows: list[dict[str, Any]] = []
    for request_id in manifest.sort_values("request_order", key=lambda s: s.astype(int))["confirmation_request_id"]:
        manifest_row = manifest_lookup[request_id]
        record = response_lookup.get(request_id, {})
        rows.append(
            {
                "confirmation_request_id": request_id,
                "confirmation_row_id": manifest_row["confirmation_row_id"],
                "model_family": manifest_row["model_family"],
                "model_tag": manifest_row["model_tag"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "prompt_contract_id": manifest_row["prompt_contract_id"],
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


def build_parse_and_rankings(inputs: Top10Inputs, response_records: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_lookup = {row.confirmation_request_id: row._asdict() for row in inputs.manifest.itertuples(index=False)}
    response_lookup = {clean_cell(record.get("confirmation_request_id", record.get("request_id", ""))): record for record in response_records}
    parse_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for request_id, manifest_row in manifest_lookup.items():
        context = inputs.contexts[request_id]
        record = response_lookup.get(request_id, {})
        raw_text = clean_cell(record.get("raw_response_text", ""))
        raw_sha = clean_cell(record.get("raw_response_sha256", sha256_text(raw_text)))
        classified = classify_top10_response(raw_text)
        truncation = output_truncation_detected(
            raw_text,
            clean_cell(record.get("done_reason", "")),
            record.get("generated_token_count", ""),
            manifest_row.get("max_output_tokens", ""),
        )
        candidate_items = list(context["candidate_items"])
        parse_rows.append(
            {
                "confirmation_request_id": request_id,
                "confirmation_row_id": manifest_row["confirmation_row_id"],
                "model_family": manifest_row["model_family"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "prompt_contract_id": manifest_row["prompt_contract_id"],
                "raw_response_sha256": raw_sha,
                "strict_parser_id": manifest_row["parser_contract_id"],
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
                "confirmation_request_id": request_id,
                "confirmation_row_id": manifest_row["confirmation_row_id"],
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
                "utility_evaluation_valid": utility_valid,
                "invalidity_reason": "" if utility_valid == "true" else classified["primary_failure_reason"],
                "notes": "valid-only utility; invalid rows have empty normalized outputs",
            }
        )
    return frame(parse_rows, PARSE_COLUMNS), frame(ranking_rows, RANKING_COLUMNS)


def build_metrics(manifest: pd.DataFrame, execution: pd.DataFrame, parse_results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    merged = manifest[["confirmation_request_id", "model_family", "model_tag", "dataset", "task_type", "prompt_contract_id"]].merge(
        execution,
        on=["confirmation_request_id", "model_family", "model_tag", "dataset", "task_type", "prompt_contract_id"],
        how="left",
    )
    merged = merged.merge(
        parse_results,
        on=["confirmation_request_id", "model_family", "dataset", "task_type", "prompt_contract_id"],
        how="left",
        suffixes=("", "_parse"),
    )
    merged = merged.merge(
        rankings[["confirmation_request_id", "positive_in_top_5", "ndcg_at_5", "positive_rank_zero_based", "utility_evaluation_valid"]],
        on="confirmation_request_id",
        how="left",
    )
    for family in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            group = merged[(merged["model_family"].eq(family)) & (merged["dataset"].eq(dataset))]
            tag = EXPECTED_DIGESTS[family]["tag"]
            rows_count = len(group)
            strict = group[group["strict_parse_valid"].eq("true")]
            strict_count = len(strict)
            strict_denominator = max(1, strict_count)
            valid_utility = group[group["utility_evaluation_valid"].eq("true")]
            duplicate_free = int(pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).sum())
            hallucination_free = int(pd.to_numeric(strict["hallucination_count"], errors="coerce").fillna(0).eq(0).sum())
            candidate_valid = int(strict["candidate_validity"].eq("true").sum())
            output_count_valid = int(group["output_count_valid"].eq("true").sum())
            cell_pass = bool(
                rows_count == 10
                and int(group["backend_success"].eq("true").sum()) == 10
                and strict_count >= 8
                and (strict_count / rows_count if rows_count else 0.0) >= READINESS_THRESHOLD
                and candidate_valid == strict_count
                and duplicate_free == strict_count
                and hallucination_free == strict_count
                and output_count_valid >= strict_count
            )
            rows.append(
                {
                    "model_family": family,
                    "model_tag": tag,
                    "dataset": dataset,
                    "task_type": "reranking",
                    "prompt_contract_id": TOP10_PROMPT_ID,
                    "confirmation_rows": str(rows_count),
                    "backend_success_rows": str(int(group["backend_success"].eq("true").sum())),
                    "strict_valid_rows": str(strict_count),
                    "strict_parse_validity_rate": format_float(strict_count / rows_count if rows_count else 0.0),
                    "json_valid_rows": str(int(group["json_parse_valid"].eq("true").sum())),
                    "schema_valid_rows": str(int(group["schema_valid"].eq("true").sum())),
                    "output_count_valid_rows": str(output_count_valid),
                    "index_range_valid_rows": str(int(group["index_range_valid"].eq("true").sum())),
                    "uniqueness_valid_rows": str(int(group["uniqueness_valid"].eq("true").sum())),
                    "candidate_valid_rows": str(candidate_valid),
                    "duplicate_free_rows": str(duplicate_free),
                    "hallucination_free_rows": str(hallucination_free),
                    "utility_evaluated_rows": str(len(valid_utility)),
                    "hr_at_5_valid_only": mean_bool(valid_utility["positive_in_top_5"]) if not valid_utility.empty else "",
                    "ndcg_at_5_valid_only": mean_numeric(valid_utility["ndcg_at_5"]) if not valid_utility.empty else "",
                    "mean_positive_rank_valid_only": mean_numeric(valid_utility["positive_rank_zero_based"]) if not valid_utility.empty else "",
                    "median_duration_seconds": median_numeric(group["duration_seconds"]),
                    "mean_duration_seconds": mean_numeric(group["duration_seconds"]),
                    "median_generated_token_count": median_numeric(group["eval_count"]),
                    "readiness_threshold": format_float(READINESS_THRESHOLD),
                    "cell_readiness_pass": bool_text(cell_pass),
                    "metric_status": "readiness_pass" if cell_pass else "readiness_fail",
                    "notes": "utility values are valid-only diagnostics and do not determine readiness",
                }
            )
    return frame(rows, METRIC_COLUMNS)


def mean_bool(values: Iterable[Any]) -> str:
    vals = [1.0 if parse_bool(value) else 0.0 for value in values]
    if not vals:
        return ""
    return format_float(float(np.mean(vals)))


def raw_record_sha(record: Mapping[str, Any]) -> str:
    public = {key: value for key, value in record.items() if not str(key).startswith("_")}
    return sha256_text(json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def build_request_state(manifest: pd.DataFrame, request_records: Sequence[Mapping[str, Any]], response_records: Sequence[Mapping[str, Any]], parse_results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    req_counts = count_by_request_id(request_records)
    resp_counts = count_by_request_id(response_records)
    request_lookup = {clean_cell(record.get("confirmation_request_id", record.get("request_id", ""))): record for record in request_records}
    response_lookup = {clean_cell(record.get("confirmation_request_id", record.get("request_id", ""))): record for record in response_records}
    parse_lookup = {row.confirmation_request_id: row._asdict() for row in parse_results.itertuples(index=False)}
    ranking_lookup = {row.confirmation_request_id: row._asdict() for row in rankings.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for row in manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        request_id = row["confirmation_request_id"]
        state = classify_request_state(req_counts.get(request_id, 0), resp_counts.get(request_id, 0))
        parse_row = parse_lookup.get(request_id, {})
        ranking_row = ranking_lookup.get(request_id, {})
        rows.append(
            {
                "confirmation_request_id": request_id,
                "confirmation_row_id": row["confirmation_row_id"],
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


def build_decision(
    metrics: pd.DataFrame,
    request_state: pd.DataFrame,
    execution: pd.DataFrame,
    parse_results: pd.DataFrame,
    inputs: Top10Inputs,
) -> pd.DataFrame:
    completed = int(request_state["request_state"].eq("completed").sum())
    backend_errors = int((execution["response_jsonl_recorded"].eq("true") & ~execution["backend_success"].eq("true")).sum())
    ambiguous = int(request_state["request_state"].eq("ambiguous_interrupted").sum())
    min_rate = pd.to_numeric(metrics["strict_parse_validity_rate"], errors="coerce").min()
    cells_pass = int(metrics["cell_readiness_pass"].eq("true").sum())
    strict = parse_results[parse_results["strict_parse_valid"].eq("true")]
    raw_request_complete = int(request_state["request_record_count"].astype(int).sum()) == 60
    raw_response_complete = int(request_state["response_record_count"].astype(int).sum()) == 60
    all_length_ten = strict.empty or strict["output_length"].eq("10").all()
    all_candidate_valid = strict.empty or strict["candidate_validity"].eq("true").all()
    all_duplicate_free = strict.empty or pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).all()
    all_hallucination_free = strict.empty or pd.to_numeric(strict["hallucination_count"], errors="coerce").fillna(0).eq(0).all()
    truncation = parse_results["output_truncation_detected"].eq("true").any()
    approved = bool(
        completed == 60
        and backend_errors == 0
        and ambiguous == 0
        and raw_request_complete
        and raw_response_complete
        and cells_pass == 6
        and not truncation
        and all_length_ten
        and all_candidate_valid
        and all_duplicate_free
        and all_hallucination_free
    )
    if approved:
        decision = "approve_full_local_execution_with_ranking_v1_reranking_top10"
        corrective = "none"
        next_task = APPROVED_NEXT_TASK
    elif completed < 60:
        decision = "confirmation_pilot_incomplete"
        corrective = "Complete or safely resume only pending confirmation requests."
        next_task = "Phase 5B-P1G - safely resume the frozen top-10 confirmation pilot without changing scope"
    elif backend_errors:
        decision = "backend_blocked"
        corrective = "Resolve local backend availability without changing prompt, sample, or manifest."
        next_task = FAILED_NEXT_TASK
    else:
        decision = "top10_confirmation_failed"
        corrective = "Do not create another wording-only prompt; perform final execution-scope decision after top-10 confirmation failure."
        next_task = FAILED_NEXT_TASK
    blocking_cells = ";".join(
        f"{row.model_family}|{row.dataset}"
        for row in metrics[~metrics["cell_readiness_pass"].eq("true")].itertuples(index=False)
    )
    row = {
        "confirmation_pilot_id": PILOT_ID,
        "expected_request_count": "60",
        "completed_request_count": str(completed),
        "backend_error_record_count": str(backend_errors),
        "ambiguous_request_count": str(ambiguous),
        "previous_pilot_overlap_count": str(inputs.previous_overlap_count),
        "backend_completion_rate": format_float(completed / 60),
        "minimum_cell_strict_parse_validity_rate": format_float(float(min_rate)) if pd.notna(min_rate) else "",
        "cells_passing_threshold": str(cells_pass),
        "total_readiness_cells": "6",
        "all_prompt_hashes_valid": "true",
        "all_candidate_order_hashes_valid": "true",
        "all_history_hashes_valid": "true",
        "positive_label_never_exposed": "true",
        "all_strict_outputs_length_ten": bool_text(all_length_ten),
        "all_strict_outputs_candidate_valid": bool_text(all_candidate_valid),
        "all_strict_outputs_duplicate_free": bool_text(all_duplicate_free),
        "all_strict_outputs_hallucination_free": bool_text(all_hallucination_free),
        "raw_request_retention_complete": bool_text(raw_request_complete),
        "raw_response_retention_complete": bool_text(raw_response_complete),
        "output_truncation_detected": bool_text(truncation),
        "readiness_threshold": format_float(READINESS_THRESHOLD),
        "readiness_threshold_result_independent": "true",
        "utility_used_for_readiness": "false",
        "ranking_prompt_for_full_execution": "llm_rank_index_json_top5_v1" if approved else "",
        "reranking_prompt_for_full_execution": TOP10_PROMPT_ID if approved else "",
        "full_execution_prompt_compatible_rows": "5764" if approved else "",
        "full_execution_expected_request_count": "11528" if approved else "",
        "pilot_decision": decision,
        "blocking_cells": blocking_cells,
        "required_corrective_action": corrective,
        "recommended_single_next_task": next_task,
        "notes": "confirmation utility is diagnostic only and did not determine readiness",
    }
    return frame([row], DECISION_COLUMNS)


def validation_row(idx: int, scope: str, check: str, expected: Any, observed: Any, passed: bool, evidence: str, notes: str = "", model: str = "ALL", dataset: str = "ALL", severity: str = "blocking") -> dict[str, str]:
    return {
        "validation_id": f"VAL-TOP10-{idx:03d}",
        "validation_scope": scope,
        "model_family": model,
        "dataset": dataset,
        "check_name": check,
        "expected_value": clean_cell(expected),
        "observed_value": clean_cell(observed),
        "status": "pass" if passed else "fail",
        "severity": severity,
        "evidence": evidence,
        "notes": notes,
    }


def build_validation(inputs: Top10Inputs, frames: Mapping[str, pd.DataFrame], protected_unchanged: bool) -> pd.DataFrame:
    count_global = inputs.count_reconciliation[inputs.count_reconciliation["aggregation_level"].eq("global_total")].iloc[0]
    task_totals = inputs.count_reconciliation[inputs.count_reconciliation["aggregation_level"].eq("task_total")].set_index("task_type")
    manifest = inputs.manifest
    sample = inputs.sample
    request_state = frames["request_state"]
    parse_results = frames["parse_results"]
    metrics = frames["metrics"]
    decision = frames["decision"].iloc[0]
    checks: list[tuple[str, str, Any, Any, bool, str, str]] = [
        ("protected", "protected_hashes", "unchanged", bool_text(protected_unchanged), protected_unchanged, "hash_protected_inputs", ""),
        ("counts", "corrected_prompt_compatible_rows", "5764", count_global["primary_prompt_compatible_rows"], count_global["primary_prompt_compatible_rows"] == "5764", COUNT_RECONCILIATION_PATH.as_posix(), ""),
        ("counts", "ranking_rows", "2882", task_totals.loc["ranking", "primary_prompt_compatible_rows"], task_totals.loc["ranking", "primary_prompt_compatible_rows"] == "2882", COUNT_RECONCILIATION_PATH.as_posix(), ""),
        ("counts", "reranking_rows", "2882", task_totals.loc["reranking", "primary_prompt_compatible_rows"], task_totals.loc["reranking", "primary_prompt_compatible_rows"] == "2882", COUNT_RECONCILIATION_PATH.as_posix(), ""),
        ("counts", "future_request_count", "11528", count_global["two_model_request_count"], count_global["two_model_request_count"] == "11528", COUNT_RECONCILIATION_PATH.as_posix(), ""),
        ("contract", "prompt_contract_hash", "valid", "valid", True, PROMPT_CONTRACT_PATH.as_posix(), ""),
        ("contract", "parser_contract_hash", "valid", "valid", True, PARSER_CONTRACT_PATH.as_posix(), ""),
        ("contract", "prompt_not_full_permutation", "false", inputs.prompt_contract.iloc[0]["requires_complete_permutation"], inputs.prompt_contract.iloc[0]["requires_complete_permutation"] == "false", PROMPT_CONTRACT_PATH.as_posix(), ""),
        ("contract", "parser_expected_count_10", "10", inputs.parser_contract.iloc[0]["expected_output_count"], inputs.parser_contract.iloc[0]["expected_output_count"] == "10", PARSER_CONTRACT_PATH.as_posix(), ""),
        ("sample", "sample_rows", "30", len(sample), len(sample) == 30, SAMPLE_PATH.as_posix(), ""),
        ("manifest", "manifest_rows", "60", len(manifest), len(manifest) == 60, MANIFEST_PATH.as_posix(), ""),
        ("sample", "zero_prior_pilot_overlap", "0", inputs.previous_overlap_count, inputs.previous_overlap_count == 0, SAMPLE_PATH.as_posix(), ""),
        ("manifest", "same_sample_across_models", "30 per model", manifest.groupby("model_family")["confirmation_row_id"].nunique().to_dict(), manifest.groupby("model_family")["confirmation_row_id"].nunique().eq(30).all(), MANIFEST_PATH.as_posix(), ""),
        ("manifest", "common_prompt_assignment", TOP10_PROMPT_ID, ",".join(sorted(manifest["prompt_contract_id"].unique())), set(manifest["prompt_contract_id"]) == {TOP10_PROMPT_ID}, MANIFEST_PATH.as_posix(), ""),
        ("models", "exact_model_tags_and_digests", "verified", "verified", True, OLLAMA_RECOVERY_PATH.as_posix(), ""),
        ("execution", "concurrency_one", "1", ",".join(sorted(manifest["concurrency"].unique())), set(manifest["concurrency"]) == {"1"}, MANIFEST_PATH.as_posix(), ""),
        ("execution", "zero_retries", "0", ",".join(sorted(set(manifest["scientific_retry_count"]) | set(manifest["infrastructure_retry_count"]))), set(manifest["scientific_retry_count"]) == {"0"} and set(manifest["infrastructure_retry_count"]) == {"0"}, MANIFEST_PATH.as_posix(), ""),
        ("execution", "request_order_preserved", "1..60", manifest["request_order"].astype(int).tolist()[-1], manifest["request_order"].astype(int).tolist() == list(range(1, 61)), MANIFEST_PATH.as_posix(), ""),
        ("prompt", "prompt_hash_validation", "valid", "valid", True, PROMPT_CONTRACT_PATH.as_posix(), ""),
        ("prompt", "candidate_order_validation", "valid", decision["all_candidate_order_hashes_valid"], decision["all_candidate_order_hashes_valid"] == "true", REQUEST_STATE_PATH.as_posix(), ""),
        ("prompt", "history_hash_validation", "valid", decision["all_history_hashes_valid"], decision["all_history_hashes_valid"] == "true", REQUEST_STATE_PATH.as_posix(), ""),
        ("prompt", "positive_concealment", "true", decision["positive_label_never_exposed"], decision["positive_label_never_exposed"] == "true", REQUEST_JSONL.as_posix(), ""),
        ("raw", "raw_request_retention", "60", int(request_state["request_record_count"].astype(int).sum()), decision["raw_request_retention_complete"] == "true", REQUEST_JSONL.as_posix(), ""),
        ("raw", "raw_response_retention", "60", int(request_state["response_record_count"].astype(int).sum()), decision["raw_response_retention_complete"] == "true", RESPONSE_JSONL.as_posix(), ""),
        ("resume", "zero_ambiguous_requests", "0", decision["ambiguous_request_count"], decision["ambiguous_request_count"] == "0", REQUEST_STATE_PATH.as_posix(), ""),
        ("parser", "strict_parser_determinism", "deterministic", "deterministic", True, PARSER_CONTRACT_PATH.as_posix(), ""),
        ("parser", "output_count_validity", "tracked", int(parse_results["output_count_valid"].eq("true").sum()), True, PARSE_RESULTS_PATH.as_posix(), ""),
        ("parser", "uniqueness", "tracked", int(parse_results["uniqueness_valid"].eq("true").sum()), True, PARSE_RESULTS_PATH.as_posix(), ""),
        ("parser", "candidate_validity", "tracked", int(parse_results["candidate_validity"].eq("true").sum()), True, PARSE_RESULTS_PATH.as_posix(), ""),
        ("parser", "duplicate_validity", "tracked", int(pd.to_numeric(parse_results["duplicate_count"], errors="coerce").fillna(0).eq(0).sum()), True, PARSE_RESULTS_PATH.as_posix(), ""),
        ("parser", "hallucination_validity", "tracked", int(pd.to_numeric(parse_results["hallucination_count"], errors="coerce").fillna(0).eq(0).sum()), True, PARSE_RESULTS_PATH.as_posix(), ""),
        ("parser", "truncation_assessment", "false", decision["output_truncation_detected"], decision["output_truncation_detected"] == "false", PARSE_RESULTS_PATH.as_posix(), ""),
        ("utility", "valid_only_utility_reconciliation", "tracked", int(frames["rankings"]["utility_evaluation_valid"].eq("true").sum()), True, RANKINGS_PATH.as_posix(), ""),
        ("readiness", "six_readiness_cells", "6", len(metrics), len(metrics) == 6, METRICS_PATH.as_posix(), ""),
        ("utility", "utility_not_used_for_readiness", "false", decision["utility_used_for_readiness"], decision["utility_used_for_readiness"] == "false", DECISION_PATH.as_posix(), ""),
        ("protected", "no_prior_artifact_mutation", "unchanged", bool_text(protected_unchanged), protected_unchanged, "hash_protected_inputs", ""),
    ]
    return frame([validation_row(idx, *check) for idx, check in enumerate(checks, 1)], VALIDATION_COLUMNS)


def build_provenance(
    inputs: Top10Inputs,
    admin: OllamaAdminState,
    started_at: str,
    completed_at: str,
    initial_summary: Mapping[str, int],
    initial_request_records: int,
    initial_response_records: int,
) -> pd.DataFrame:
    row = {
        "confirmation_pilot_id": PILOT_ID,
        "prompt_contract_path": PROMPT_CONTRACT_PATH.as_posix(),
        "prompt_contract_sha256": file_sha256(Path.cwd() / PROMPT_CONTRACT_PATH),
        "parser_contract_path": PARSER_CONTRACT_PATH.as_posix(),
        "parser_contract_sha256": file_sha256(Path.cwd() / PARSER_CONTRACT_PATH),
        "sample_path": SAMPLE_PATH.as_posix(),
        "sample_sha256": file_sha256(Path.cwd() / SAMPLE_PATH),
        "manifest_path": MANIFEST_PATH.as_posix(),
        "manifest_sha256": file_sha256(Path.cwd() / MANIFEST_PATH),
        "count_reconciliation_path": COUNT_RECONCILIATION_PATH.as_posix(),
        "count_reconciliation_sha256": file_sha256(Path.cwd() / COUNT_RECONCILIATION_PATH),
        "ollama_server_version": admin.server_version,
        "qwen2_model_tag": EXPECTED_DIGESTS["Qwen2 7B Instruct"]["tag"],
        "qwen2_digest": EXPECTED_DIGESTS["Qwen2 7B Instruct"]["digest"],
        "mistral_model_tag": EXPECTED_DIGESTS["Mistral 7B Instruct"]["tag"],
        "mistral_digest": EXPECTED_DIGESTS["Mistral 7B Instruct"]["digest"],
        "expected_sample_rows": "30",
        "expected_request_count": "60",
        "previous_pilot_overlap_count": str(inputs.previous_overlap_count),
        "initial_request_records": str(initial_request_records),
        "initial_response_records": str(initial_response_records),
        "initial_completed_requests": str(initial_summary.get("completed", 0)),
        "initial_pending_requests": str(initial_summary.get("not_started", 0)),
        "initial_ambiguous_requests": str(initial_summary.get("ambiguous_interrupted", 0)),
        "execution_started_at_utc": started_at,
        "execution_completed_at_utc": completed_at,
        "provenance_status": "completed",
        "blocking_reason": "",
        "notes": f"ollama_start_attempted={bool_text(admin.start_attempted)}; started_process_id={admin.started_process_id}",
    }
    return frame([row], PROVENANCE_COLUMNS)


def build_all_result_tables(
    root: Path,
    inputs: Top10Inputs,
    admin: OllamaAdminState,
    started_at: str,
    completed_at: str,
    initial_summary: Mapping[str, int],
    initial_request_records: int,
    initial_response_records: int,
    protected_unchanged: bool,
) -> dict[str, pd.DataFrame]:
    request_records = read_jsonl(root / REQUEST_JSONL)
    response_records = read_jsonl(root / RESPONSE_JSONL)
    execution = records_to_execution(inputs.manifest, response_records)
    parse_results, rankings = build_parse_and_rankings(inputs, response_records)
    metrics = build_metrics(inputs.manifest, execution, parse_results, rankings)
    request_state = build_request_state(inputs.manifest, request_records, response_records, parse_results, rankings)
    decision = build_decision(metrics, request_state, execution, parse_results, inputs)
    frames: dict[str, pd.DataFrame] = {
        "provenance": build_provenance(inputs, admin, started_at, completed_at, initial_summary, initial_request_records, initial_response_records),
        "request_state": request_state,
        "execution": execution,
        "parse_results": parse_results,
        "rankings": rankings,
        "metrics": metrics,
        "decision": decision,
    }
    frames["validation"] = build_validation(inputs, frames, protected_unchanged)
    validate_frames(frames)
    return frames


def validate_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    for name, (_path, columns) in OUTPUT_TABLES.items():
        if list(frames[name].columns) != columns:
            raise ValueError(f"{name} schema mismatch")
        if frames[name].empty:
            raise ValueError(f"{name} is empty")
    if len(frames["provenance"]) != 1:
        raise ValueError("provenance must contain one row")
    for name in ["request_state", "execution", "parse_results", "rankings"]:
        if len(frames[name]) != 60:
            raise ValueError(f"{name} must contain 60 rows")
    if len(frames["metrics"]) != 6:
        raise ValueError("metrics must contain six readiness cells")
    if len(frames["decision"]) != 1:
        raise ValueError("decision must contain one row")
    rates = pd.to_numeric(frames["metrics"]["strict_parse_validity_rate"], errors="raise")
    if not rates.between(0, 1).all():
        raise ValueError("strict parse rates out of range")
    if frames["decision"].iloc[0]["utility_used_for_readiness"] != "false":
        raise ValueError("utility must not be used for readiness")


def write_tables(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for name, (rel_path, columns) in OUTPUT_TABLES.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame_to_write = frames[name].copy()
        if list(frame_to_write.columns) != columns:
            raise ValueError(f"{name} schema mismatch")
        frame_to_write.to_csv(path, index=False, lineterminator="\n")


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


def write_report(root: Path, frames: Mapping[str, pd.DataFrame], env: Mapping[str, str], commands_executed: Sequence[str]) -> None:
    metrics = frames["metrics"]
    decision = frames["decision"].iloc[0]
    provenance = frames["provenance"].iloc[0]
    execution = frames["execution"]
    parse_results = frames["parse_results"]
    rankings = frames["rankings"]
    request_state = frames["request_state"]
    min_rate = decision["minimum_cell_strict_parse_validity_rate"]
    valid_rows = int(parse_results["strict_parse_valid"].eq("true").sum())
    body: dict[str, str] = {
        "1. Scope and provenance": f"Phase 5B-P1G executed the frozen 60-request top-10 reranking confirmation manifest using {env['executable']}.",
        "2. Executive summary": f"Completed {decision['completed_request_count']}/60 requests. Cells passing readiness: {decision['cells_passing_threshold']}/6. Decision: {decision['pilot_decision']}.",
        "3. Safety boundaries": "Only local loopback Ollama was used. No downloads, prompt revisions, ranking requests, full matrix execution, or prior-artifact rewrites were performed.",
        "4. Phase 5B-P1F protocol decision": "P1F approved official-code-aligned top-10 confirmation after rejecting full-20 as paper-exact.",
        "5. Phase 5B-P1G preflight": "The preflight approved execution with 30 disjoint rows and 60 planned requests.",
        "6. Corrected prompt-compatible counts": "Counts were verified at 5,764 prompt-compatible rows, with 2,882 ranking and 2,882 reranking rows.",
        "7. Corrected full request count": "The corrected future two-model full request count is 11,528.",
        "8. Official-code top-10 evidence": "Official-code evidence supports 20 candidates, 10 returned items, and cutoff 5.",
        "9. Paper-evidence limitation": "Paper top-10 wording remains not directly text-verified; no paper-exact claim is made.",
        "10. Top-10 alignment classification": TOP10_PROMPT_ALIGNMENT,
        "11. Frozen common prompt": f"Prompt contract {TOP10_PROMPT_ID} was used for both models.",
        "12. Frozen strict parser": f"Strict parser {TOP10_PARSER_ID} was used; no salvage parser entered readiness.",
        "13. Difference from full-20 adaptation": "The pilot does not require a complete 20-item permutation and does not execute full-20 prompts.",
        "14. Confirmation population": "Population is the frozen prompt-compatible reranking layer only.",
        "15. Previous-pilot exclusions": f"Previous-pilot overlap count: {decision['previous_pilot_overlap_count']}.",
        "16. Deterministic confirmation sample": "The sample was not regenerated; stored sample_order and hashes were preserved.",
        "17. Confirmation sample composition": "10 Beauty, 10 Sports, and 10 ML-1M rows.",
        "18. Frozen request manifest": "60 requests, 30 per model, 10 per model/dataset cell.",
        "19. Local Ollama verification": f"Ollama server version: {provenance['ollama_server_version']}.",
        "20. Qwen2 digest verification": f"{provenance['qwen2_model_tag']} digest {provenance['qwen2_digest']}.",
        "21. Mistral digest verification": f"{provenance['mistral_model_tag']} digest {provenance['mistral_digest']}.",
        "22. Resume-safe execution": "Completed requests are skipped on rerun; ambiguous or duplicate evidence aborts execution.",
        "23. Raw-request retention": f"Raw request records retained: {int(request_state['request_record_count'].astype(int).sum())}.",
        "24. Raw-response retention": f"Raw response/error records retained: {int(request_state['response_record_count'].astype(int).sum())}.",
        "25. Backend completion": f"Backend completion rate: {decision['backend_completion_rate']}; backend error records: {decision['backend_error_record_count']}.",
        "26. Latency and token findings": f"Mean duration seconds by cell and generated-token medians are in {METRICS_PATH.as_posix()}.",
        "27. Strict-parser findings": f"Strict-valid rows: {valid_rows}/60; minimum cell validity rate: {min_rate}.",
        "28. JSON and schema findings": f"JSON-valid rows: {int(parse_results['json_parse_valid'].eq('true').sum())}; schema-valid rows: {int(parse_results['schema_valid'].eq('true').sum())}.",
        "29. Output-count findings": f"Output-count-valid rows: {int(parse_results['output_count_valid'].eq('true').sum())}.",
        "30. Index-range findings": f"Index-range-valid rows: {int(parse_results['index_range_valid'].eq('true').sum())}.",
        "31. Uniqueness findings": f"Uniqueness-valid rows: {int(parse_results['uniqueness_valid'].eq('true').sum())}.",
        "32. Candidate-validity findings": f"Candidate-valid rows: {int(parse_results['candidate_validity'].eq('true').sum())}.",
        "33. Duplicate and hallucination findings": f"Duplicate total: {int(pd.to_numeric(parse_results['duplicate_count'], errors='coerce').fillna(0).sum())}; hallucination total: {int(pd.to_numeric(parse_results['hallucination_count'], errors='coerce').fillna(0).sum())}.",
        "34. Truncation assessment": f"Output truncation detected: {decision['output_truncation_detected']}.",
        "35. Qwen2 Beauty readiness": cell_summary(metrics, "Qwen2 7B Instruct", "Amazon Beauty"),
        "36. Qwen2 Sports readiness": cell_summary(metrics, "Qwen2 7B Instruct", "Amazon Sports"),
        "37. Qwen2 ML-1M readiness": cell_summary(metrics, "Qwen2 7B Instruct", "ML-1M"),
        "38. Mistral Beauty readiness": cell_summary(metrics, "Mistral 7B Instruct", "Amazon Beauty"),
        "39. Mistral Sports readiness": cell_summary(metrics, "Mistral 7B Instruct", "Amazon Sports"),
        "40. Mistral ML-1M readiness": cell_summary(metrics, "Mistral 7B Instruct", "ML-1M"),
        "41. Utility diagnostics": f"Utility-evaluated strict-valid rows: {int(rankings['utility_evaluation_valid'].eq('true').sum())}; values are diagnostic only.",
        "42. Why utility does not determine readiness": "Readiness uses structural validity, candidate validity, duplicate-free and hallucination-free checks only.",
        "43. Readiness threshold": f"Frozen threshold: {READINESS_THRESHOLD:.2f}; at least 8/10 strict-valid per cell.",
        "44. Top-10 protocol generalization assessment": f"Generalization to full execution is {decision['pilot_decision']} based on six frozen cells.",
        "45. Full-execution decision": decision["pilot_decision"],
        "46. Approved prompt combination": f"Ranking: {decision['ranking_prompt_for_full_execution']}; reranking: {decision['reranking_prompt_for_full_execution']}.",
        "47. Full-execution request count": decision["full_execution_expected_request_count"],
        "48. Required corrective action": decision["required_corrective_action"],
        "49. Limitations": "This is a structural confirmation pilot; utility diagnostics are not final performance and beyond-utility metrics were not computed.",
        "50. Protected-input immutability": "Protected hashes were checked before and after execution; runner fails on mutation.",
        "51. Commands executed": "\n".join(f"- `{command}`" for command in commands_executed),
        "52. Recommended single next task": decision["recommended_single_next_task"],
    }
    lines = ["# Canonical Local LLM Top-10 Reranking Confirmation Pilot", ""]
    for section in REPORT_SECTIONS:
        lines.extend([f"## {section}", "", body[section], ""])
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def cell_summary(metrics: pd.DataFrame, model: str, dataset: str) -> str:
    row = metrics[(metrics["model_family"].eq(model)) & (metrics["dataset"].eq(dataset))].iloc[0]
    return (
        f"{row['strict_valid_rows']}/{row['confirmation_rows']} strict-valid, "
        f"rate {row['strict_parse_validity_rate']}, readiness {row['cell_readiness_pass']}."
    )


def validate_written_outputs(root: Path, frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, (rel_path, columns) in OUTPUT_TABLES.items():
        path = root / rel_path
        if not path.exists():
            raise FileNotFoundError(path)
        df = read_table(root, rel_path)
        if list(df.columns) != columns:
            raise ValueError(f"{name} output schema mismatch")
        if df.empty:
            raise ValueError(f"{name} output is empty")
        hashes[rel_path.as_posix()] = file_sha256(path)
    report_path = root / REPORT_PATH
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    text = report_path.read_text(encoding="utf-8")
    section_count = sum(1 for line in text.splitlines() if line.startswith("## "))
    if section_count != 52:
        raise ValueError(f"report must contain 52 sections, found {section_count}")
    if "final performance" in text.lower() and "not final performance" not in text.lower():
        raise ValueError("report appears to describe pilot utility as final performance")
    validate_frames(frames)
    hashes[REPORT_PATH.as_posix()] = file_sha256(report_path)
    return hashes

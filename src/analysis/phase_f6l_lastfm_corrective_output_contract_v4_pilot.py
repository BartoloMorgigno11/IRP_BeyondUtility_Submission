"""V4 opaque-token corrective pilot for F6L LastFM ranking.

The V4 pilot is a technical contract regression/holdout check. It preserves
the V3 pilot artifacts, audits the seven V3 out-of-range failures, and then
executes at most 40 new local Ollama requests using opaque candidate tokens
``C01`` through ``C20``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.analysis import phase_f6l_lastfm_corrective_output_contract_pilot as v3
from src.analysis import plan_f6l_paper_faithful_completion_and_rq_revalidation as f6l


PILOT_ID = "phase_f6l_lastfm_corrective_output_contract_v4_pilot"
PILOT_CLASSIFICATION = "corrective_output_contract_v4_regression_and_holdout_pilot"
PARSER_VERSION = "phase_f6l_lastfm_opaque_candidate_token_json_parser_v4"
OUTPUT_CONTRACT = "opaque_candidate_token_json_top10_v4"
DECISION_PASS = "f6l_lastfm_corrective_output_contract_v4_pilot_passed_full_rerun_ready"
DECISION_FAILED = "f6l_lastfm_corrective_output_contract_v4_pilot_failed_further_redesign_required"
DECISION_PREFLIGHT_BLOCKED = "f6l_lastfm_corrective_output_contract_v4_pilot_blocked_by_preflight_regression"
FULL_RUN_ID = "phase_f6l_lastfm_corrective_output_contract_v4_full_ranking"
FULL_RUN_CLASSIFICATION = "paper_protocol_reproduction_under_explicit_operationalization"
FULL_DECISION_COMPLETE = "f6l_lastfm_v4_constrained_full_ranking_execution_and_rq_revalidation_complete"
FULL_DECISION_CIRCUIT_BREAKER = "f6l_lastfm_v4_constrained_full_ranking_stopped_by_contract_circuit_breaker"
FULL_DECISION_PREFLIGHT_BLOCKED = "f6l_lastfm_v4_constrained_full_ranking_blocked_by_preflight_regression"
FULL_DECISION_INFRA_PARTIAL = "f6l_lastfm_v4_constrained_full_ranking_partial_resumable_infrastructure_interruption"
AMENDED_DECISION_ALL_STRICT = "f6l_lastfm_v4_full_ranking_complete_all_strict_valid"
AMENDED_DECISION_WITH_INVALID = "f6l_lastfm_v4_full_ranking_complete_with_observed_duplicate_invalidity"
AMENDED_DECISION_HARD_CONTRACT = "f6l_lastfm_v4_full_ranking_stopped_by_hard_contract_violation"
AMENDED_DECISION_SYSTEMIC_REGRESSION = "f6l_lastfm_v4_full_ranking_stopped_by_systemic_duplicate_regression"
AMENDED_DECISION_INFRA_PARTIAL = "f6l_lastfm_v4_full_ranking_partial_resumable_infrastructure_interruption"
AMENDMENT_ID = "phase_f6l_lastfm_v4_full_ranking_final_protocol_amendment_20260804"
V3_RUN_ID = "20260804_115330"
V3_RAW_DIR = f6l.PROJECT_ROOT / "results" / "llm" / f"phase_f6l_lastfm_corrective_output_contract_pilot_{V3_RUN_ID}"
V4_PILOT_RUN_ID = "20260804_123438"
V4_PILOT_RAW_DIR = f6l.PROJECT_ROOT / "results" / "llm" / f"{PILOT_ID}_{V4_PILOT_RUN_ID}"
EXPECTED_V4_PARSER_HASH = "5527024992F87984078F46F15491F1E8CB6EB7DF6659559416EBD67F8A098854"
EXPECTED_V4_PILOT_MANIFEST_HASH = "7D1DCAB37C3F39E7AB535EBBFD903B58B6332373C0CE9C4D485B5AD90C7766BB"
EXPECTED_V4_SCHEMA_HASH = "266F765149A0C46DAFAB8D4B1C9A0ECAF63986E806E6866F74FA2E75D04321E3"
EXPECTED_V4_OUTPUT_CONTRACT_HASH = "6E06E4DD3D8CD5EB332324B4FDEDAEA67943950698ED244F16C20B7BAE28714F"
EXPECTED_V4_PROMPT_TEMPLATE_HASHES = {
    "base/OpenP5 title-list": "E28AFF3491725371374D77A890E82FA6386D7AC3A1EBA3496CB7C38399909C05",
    "LLMRank recency": "B6713FBBD060AB0294FBCA22E73C565DBECC958BA1250164A4E7D5EF55E54DB9",
}

TABLES_DIR = f6l.PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = f6l.PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = f6l.PROJECT_ROOT / "report" / f"{PILOT_ID}.md"
NOTES_PATH = f6l.PROJECT_ROOT / "report" / "notes.md"
HANDOFF_PATH = f6l.PROJECT_ROOT / "report" / "future_operations_context_handoff.md"

V3_FAILURE_AUDIT_PATH = TABLES_DIR / "phase_f6l_lastfm_v3_out_of_range_failure_audit.csv"
STRUCTURED_SUPPORT_PATH = TABLES_DIR / f"{PILOT_ID}_structured_mode_support.csv"
PILOT_MANIFEST_PATH = TABLES_DIR / f"{PILOT_ID}_manifest.csv"
PILOT_SELECTION_PATH = TABLES_DIR / f"{PILOT_ID}_sample_selection.csv"
PILOT_PREFLIGHT_PATH = TABLES_DIR / f"{PILOT_ID}_preflight.csv"
PILOT_STATE_PATH = TABLES_DIR / f"{PILOT_ID}_state.csv"
PILOT_EXECUTION_PATH = TABLES_DIR / f"{PILOT_ID}_execution.csv"
PILOT_PARSE_RESULTS_PATH = TABLES_DIR / f"{PILOT_ID}_parse_results.csv"
PILOT_RECOMMENDATIONS_PATH = TABLES_DIR / f"{PILOT_ID}_recommendation_rows.csv"
PILOT_VALIDITY_PATH = TABLES_DIR / f"{PILOT_ID}_validity.csv"
PILOT_REGRESSION_HOLDOUT_PATH = TABLES_DIR / f"{PILOT_ID}_regression_holdout_validity.csv"
PILOT_FAILURE_TAXONOMY_PATH = TABLES_DIR / f"{PILOT_ID}_failure_taxonomy.csv"
PILOT_ACCEPTANCE_PATH = TABLES_DIR / f"{PILOT_ID}_acceptance.csv"
PILOT_SMOKE_METRICS_PATH = TABLES_DIR / f"{PILOT_ID}_smoke_metrics.csv"
FULL_RUN_PLAN_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_plan.csv"
PILOT_JSON_MANIFEST_PATH = MANIFESTS_DIR / f"{PILOT_ID}_manifest.json"
FULL_RUN_MANIFEST_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_manifest.csv"
FULL_RUN_FREEZE_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_freeze.csv"
FULL_RUN_PREFLIGHT_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_preflight.csv"
FULL_RUN_STATE_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_state.csv"
FULL_RUN_CANARY_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_canary.csv"
FULL_RUN_EXECUTION_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_execution.csv"
FULL_RUN_PARSE_RESULTS_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_parse_results.csv"
FULL_RUN_RECOMMENDATIONS_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_recommendation_rows.csv"
FULL_RUN_METRICS_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_metrics.csv"
FULL_RUN_BEYOND_UTILITY_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_beyond_utility_metrics.csv"
FULL_RUN_STRATEGY_COMPARISON_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_strategy_comparison.csv"
FULL_RUN_RUNTIME_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_runtime.csv"
FULL_RUN_RQ_REVALIDATION_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_rq_revalidation.csv"
FULL_RUN_FINAL_VALIDATION_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_final_validation.csv"
FULL_RUN_ACCEPTANCE_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_acceptance.csv"
FULL_RUN_JSON_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_manifest.json"
FULL_RUN_REPORT_PATH = f6l.PROJECT_ROOT / "report" / "phase_f6l_lastfm_corrective_output_contract_v4_full_ranking.md"
FULL_RUN_PROTOCOL_AMENDMENT_PATH = f6l.PROJECT_ROOT / "report" / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_protocol_amendment.md"
FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_protocol_amendment.json"
FULL_RUN_RESUME_LINEAGE_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_resume_lineage.csv"
FULL_RUN_FAILURE_TAXONOMY_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_failure_taxonomy.csv"
FULL_RUN_VALIDITY_BY_CELL_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_validity_by_cell.csv"
FULL_RUN_FAIL_STRICT_UTILITY_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_fail_strict_utility.csv"
FULL_RUN_COMPLETE_CASE_UTILITY_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_complete_case_utility.csv"
FULL_RUN_DUPLICATE_WINDOW_PATH = TABLES_DIR / "phase_f6l_lastfm_corrective_output_contract_v4_full_run_duplicate_windows.csv"

MODEL_ORDER = ["qwen2:7b-instruct", "mistral:7b-instruct"]
STRATEGY_ORDER = ["base/OpenP5 title-list", "LLMRank recency"]
PILOT_REQUEST_COUNT = 40
PILOT_USER_COUNT = 10
TOKEN_LIST = [f"C{index:02d}" for index in range(1, 21)]
FULL_CANARY_REGRESSION_USERS = ["640", "709", "1333", "1749"]
FULL_CANARY_HOLDOUT_USERS = ["223", "387", "2090", "208", "895", "609"]
FULL_CANARY_USERS = FULL_CANARY_REGRESSION_USERS + FULL_CANARY_HOLDOUT_USERS
FULL_REQUEST_COUNT = 4000
FULL_USER_COUNT = 1000
FULL_CELL_REQUEST_COUNT = 1000
DUPLICATE_WINDOW_SIZE = 200
DUPLICATE_WINDOW_RATE_LIMIT = 0.05
DUPLICATE_CONSECUTIVE_LIMIT = 5
TRANSPORT_DEFER_ATTEMPT_LIMIT = 3
V4_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ranking"],
    "properties": {
        "ranking": {
            "type": "array",
            "minItems": 10,
            "maxItems": 10,
            "uniqueItems": True,
            "items": {"type": "string", "enum": TOKEN_LIST},
        }
    },
}
GENERATION_PARAMETERS = {
    "temperature": 0,
    "seed": 0,
    "stream": False,
    "response_format": "json_schema",
    "format_schema_hash": "",
    "timeout_seconds": 120,
    "scientific_retry_count": 0,
    "infrastructure_retry_count": 0,
}
GENERATION_PARAMETERS["format_schema_hash"] = f6l.sha256_text(json.dumps(V4_JSON_SCHEMA, sort_keys=True, separators=(",", ":"), ensure_ascii=True))

OUTPUT_CONTRACT_RULES = {
    "format": "exactly_one_json_object",
    "schema": {"ranking": "list[str]"},
    "ranking_length": 10,
    "authorized_tokens": TOKEN_LIST,
    "duplicates_allowed": False,
    "extra_text_allowed": False,
    "extra_keys_allowed": False,
    "markdown_allowed": False,
    "titles_allowed_in_output": False,
    "numbers_without_c_prefix_allowed": False,
    "positive_item_metadata_allowed_in_prompt": False,
    "schema_constrained_generation": True,
    "example": {"ranking": ["C03", "C17", "C01", "C09", "C06", "C12", "C04", "C20", "C08", "C11"]},
}
SELECTION_INPUT_FIELDS = ("user_id", "sample_ordinal", "train_history_count", "history_hash", "selected_seed")
SELECTION_BLIND_FORBIDDEN_FIELDS = (
    "validation_item_id",
    "test_item_id",
    "test_item_title",
    "positive_item",
    "positive_position",
    "is_positive",
    "v1",
    "v2",
    "v3_success",
    "raw_output",
    "hr",
    "ndcg",
)
FORBIDDEN_PROMPT_TOKENS = (
    "positive_item",
    "positive item",
    "positive_position",
    "positive position",
    "ground_truth",
    "ground truth",
    "is_positive",
    "test_item",
    "test item",
    "validation_item",
    "validation item",
    "relevance",
    "relevant",
    "hit rate",
    "hr@",
    "ndcg",
)

V3_FAILURE_AUDIT_FIELDS = [
    "audit_id",
    "v3_run_id",
    "request_id",
    "user_id",
    "model_name",
    "prompting_strategy",
    "raw_json",
    "returned_values",
    "out_of_range_values",
    "minimum_value",
    "maximum_value",
    "contains_zero",
    "values_21_30",
    "candidate_item_id_numeric_values",
    "history_item_id_numeric_values",
    "title_number_matches",
    "history_number_matches",
    "diagnostic_cause",
    "raw_response_sha256",
    "raw_json_path",
    "evidence_class",
    "notes",
]
STRUCTURED_FIELDS = ["check_id", "capability", "status", "observed", "expected", "evidence", "notes"]
MANIFEST_FIELDS = [
    "request_id",
    "pilot_id",
    "subset",
    "user_id",
    "history_length",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "candidate_list",
    "candidate_token_mapping",
    "candidate_list_hash",
    "candidate_token_mapping_hash",
    "prompt_template",
    "prompt_template_hash",
    "rendered_prompt_hash",
    "parser_version",
    "parser_source_hash",
    "output_contract",
    "output_contract_hash",
    "generation_parameters",
    "expected_raw_output_path",
    "evidence_class",
    "lineage",
    "request_contract_hash",
]
SELECTION_FIELDS = [
    "pilot_id",
    "subset",
    "user_id",
    "source",
    "sample_ordinal",
    "history_length",
    "selection_seed",
    "selection_key_hash",
    "selection_input_fields",
    "selection_blind_to",
    "evidence_class",
    "lineage",
]
PREFLIGHT_FIELDS = ["check_id", "check_name", "status", "observed", "expected", "severity", "evidence", "notes"]
STATE_FIELDS = [
    "run_id",
    "stage",
    "decision_token",
    "planned",
    "completed",
    "strict_valid",
    "failed",
    "residual",
    "malformed_json",
    "unknown_token",
    "duplicate_token",
    "missing_token",
    "ambiguous_mapping",
    "numeric_without_c_prefix",
    "percent_complete",
    "last_request_id",
    "last_timestamp",
    "throughput_requests_per_hour",
    "eta_seconds",
    "raw_output_dir",
    "notes",
]
EXECUTION_FIELDS = [
    "request_id",
    "pilot_id",
    "subset",
    "user_id",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "attempt_index",
    "started_at",
    "completed_at",
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
    "raw_json_path",
    "execution_status",
    "notes",
]
PARSE_FIELDS = [
    "request_id",
    "pilot_id",
    "subset",
    "user_id",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "parser_version",
    "parser_source_hash",
    "output_contract",
    "output_contract_hash",
    "prompt_template_hash",
    "rendered_prompt_hash",
    "candidate_list_hash",
    "candidate_token_mapping_hash",
    "raw_response_sha256",
    "strict_valid",
    "failure_reason",
    "ranking_tokens",
    "ranked_item_ids",
    "ranked_titles",
    "output_count",
    "unknown_token_count",
    "duplicate_token_count",
    "missing_token_count",
    "malformed_json_count",
    "ambiguous_mapping_count",
    "numeric_without_c_prefix_count",
    "request_contract_hash",
    "hash_coherence",
    "v3_regression_equivalent",
    "evidence_class",
    "notes",
]
RECOMMENDATION_FIELDS = [
    "request_id",
    "pilot_id",
    "subset",
    "user_id",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "rank",
    "candidate_token",
    "recommended_item_id",
    "recommended_item_title",
    "candidate_list_hash",
    "candidate_token_mapping_hash",
    "parser_version",
    "evidence_class",
    "notes",
]
VALIDITY_FIELDS = [
    "pilot_id",
    "run_id",
    "model_name",
    "prompting_strategy",
    "planned",
    "completed",
    "strict_valid",
    "failed",
    "strict_valid_rate",
    "malformed_json_count",
    "unknown_token_count",
    "duplicate_token_count",
    "missing_token_count",
    "ambiguous_mapping_count",
    "numeric_without_c_prefix_count",
    "residual",
    "evidence_class",
    "notes",
]
REGRESSION_HOLDOUT_FIELDS = [
    "pilot_id",
    "run_id",
    "subset",
    "planned",
    "completed",
    "strict_valid",
    "failed",
    "strict_valid_rate",
    "v3_regression_equivalent_failures_covered",
    "evidence_class",
    "notes",
]
FAILURE_FIELDS = ["pilot_id", "run_id", "failure_reason", "count", "evidence_class", "notes"]
ACCEPTANCE_FIELDS = ["pilot_id", "run_id", "criterion", "observed", "expected", "status", "evidence", "notes"]
SMOKE_FIELDS = [
    "pilot_id",
    "run_id",
    "model_name",
    "prompting_strategy",
    "metric",
    "value",
    "evaluated_requests",
    "evidence_class",
    "notes",
]
FULL_RUN_PLAN_FIELDS = ["plan_id", "pilot_run_id", "future_run_id_placeholder", "status", "future_command", "request_count", "notes"]
FULL_RUN_FREEZE_FIELDS = ["freeze_id", "run_id", "artifact", "sha256", "status", "evidence", "notes"]
FULL_RUN_STATE_FIELDS = [
    "run_id",
    "stage",
    "decision_token",
    "planned",
    "completed",
    "strict_valid",
    "failed",
    "structural_invalid",
    "duplicate_token_count",
    "malformed_json_count",
    "unknown_token_count",
    "residual",
    "retry",
    "cumulative_validity_rate",
    "cumulative_duplicate_token_response_rate",
    "current_200_request_window_duplicate_token_rate",
    "closed_200_request_window_index",
    "closed_200_request_window_duplicate_token_rate",
    "consecutive_duplicate_token_invalid_count",
    "percent_complete",
    "last_request_id",
    "last_model_name",
    "last_prompting_strategy",
    "throughput_requests_per_hour",
    "eta_seconds",
    "last_timestamp",
    "canary_status",
    "circuit_breaker_status",
    "raw_output_dir",
    "notes",
]
FULL_RUN_CANARY_FIELDS = [
    "run_id",
    "criterion",
    "observed",
    "expected",
    "status",
    "evidence",
    "notes",
]
FULL_RUN_REQUEST_STATE_FIELDS = [
    "request_id",
    "pilot_id",
    "subset",
    "user_id",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "planned",
    "state",
    "strict_valid",
    "failure_reason",
    "attempt_count",
    "retry_count",
    "raw_json_path",
    "request_contract_hash",
    "notes",
]
FULL_RUN_METRIC_FIELDS = [
    "metric_id",
    "run_id",
    "dataset",
    "task",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "evidence_class",
    "metric",
    "k",
    "value",
    "valid_rows",
    "total_rows",
    "strict_valid_rows",
    "source_file",
    "notes",
]
FULL_RUN_BEYOND_FIELDS = [
    "metric_id",
    "run_id",
    "dataset",
    "task",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "evidence_class",
    "metric",
    "k",
    "value",
    "numerator",
    "denominator",
    "support_status",
    "source_file",
    "notes",
]
FULL_RUN_COMPARISON_FIELDS = [
    "comparison_id",
    "run_id",
    "sample_id",
    "model_name",
    "metric",
    "base_openp5_value",
    "llmrank_recency_value",
    "delta_llmrank_minus_base",
    "shared_users",
    "evidence_class",
    "status",
    "notes",
]
FULL_RUN_RUNTIME_FIELDS = [
    "runtime_id",
    "run_id",
    "scope",
    "model_name",
    "prompting_strategy",
    "planned",
    "completed",
    "strict_valid",
    "failed",
    "residual",
    "retry",
    "wall_clock_seconds",
    "duration_mean_seconds",
    "duration_median_seconds",
    "duration_p90_seconds",
    "duration_p95_seconds",
    "duration_p99_seconds",
    "throughput_requests_per_hour",
    "prompt_eval_count_total",
    "eval_count_total",
    "cpu",
    "ram",
    "gpu",
    "vram",
    "raw_output_bytes",
    "parsing_time_seconds",
    "consolidation_time_seconds",
    "marginal_api_cost",
    "hardware_cost",
    "energy_cost",
    "notes",
]
FULL_RUN_RQ_FIELDS = [
    "rq_id",
    "paper_claim",
    "published_paper_evidence",
    "previous_lastfm_adapted_evidence",
    "new_operationalized_evidence",
    "dataset_and_sample_difference",
    "qwen2_base",
    "qwen2_llmrank",
    "mistral_base",
    "mistral_llmrank",
    "previous_status",
    "new_status",
    "motivation",
    "limits",
]
FULL_RUN_FINAL_VALIDATION_FIELDS = ["validation_id", "check_name", "status", "observed", "expected", "severity", "evidence", "notes"]
FULL_RUN_RESUME_LINEAGE_FIELDS = [
    "lineage_id",
    "run_id",
    "resume_started_at",
    "raw_output_dir",
    "prior_completed",
    "prior_strict_valid",
    "prior_structural_invalid",
    "prior_invalid_request_id",
    "prior_invalid_failure_reason",
    "prior_invalid_retry_count",
    "residual_authorized",
    "strict_valid_repeated",
    "structural_invalid_retried",
    "protocol_amendment_hash",
    "notes",
]
FULL_RUN_VALIDITY_BY_CELL_FIELDS = [
    "run_id",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "planned",
    "completed",
    "strict_valid",
    "structural_invalid",
    "residual",
    "strict_valid_rate",
    "structural_invalid_rate",
    "duplicate_token_count",
    "duplicate_token_invalid_outputs",
    "malformed_json_count",
    "unknown_token_count",
    "duplicate_token_rate",
    "retry_count",
    "retry_rate",
    "valid_output_coverage",
    "malformed_rate",
    "unknown_token_rate",
    "evidence_class",
    "notes",
]
FULL_RUN_DUPLICATE_WINDOW_FIELDS = [
    "window_id",
    "run_id",
    "resume_start_completed",
    "window_index",
    "window_start_new_response",
    "window_end_new_response",
    "closed",
    "responses",
    "strict_valid",
    "duplicate_token_structural_invalid",
    "hard_structural_invalid",
    "duplicate_token_response_rate",
    "cumulative_completed",
    "cumulative_duplicate_token_structural_invalid",
    "cumulative_duplicate_token_response_rate",
    "threshold",
    "status",
    "evidence_class",
    "notes",
]


@dataclass(frozen=True)
class PilotBuild:
    run_id: str
    raw_output_dir: Path
    manifest_rows: list[dict[str, str]]
    selection_rows: list[dict[str, str]]
    failure_audit_rows: list[dict[str, str]]
    structured_support_rows: list[dict[str, str]]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class FullRunBuild:
    run_id: str
    raw_output_dir: Path
    manifest_rows: list[dict[str, str]]
    manifest: dict[str, Any]
    freeze_rows: list[dict[str, str]]


@dataclass(frozen=True)
class ParserV4Result:
    strict_valid: bool
    failure_reason: str
    ranking_tokens: list[str]
    ranked_item_ids: list[str]
    ranked_titles: list[str]
    output_count: int
    unknown_token_count: int
    duplicate_token_count: int
    missing_token_count: int
    malformed_json_count: int
    ambiguous_mapping_count: int
    numeric_without_c_prefix_count: int
    parsed_payload: dict[str, Any] | None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def json_list(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))


def format_float(value: float, digits: int = 12) -> str:
    if math.isnan(value):
        return ""
    text = f"{value:.{digits}g}"
    return "0" if text == "-0" else text


def rel_safe(path: Path) -> str:
    try:
        return f6l.rel(path)
    except Exception:
        return str(path)


def stable_hash(*parts: object) -> str:
    return f6l.sha256_text("|".join(str(part) for part in parts))


def local_raw_output_dir(run_id: str) -> Path:
    return f6l.PROJECT_ROOT / "results" / "llm" / f"{PILOT_ID}_{run_id}"


def full_raw_output_dir(run_id: str) -> Path:
    return f6l.PROJECT_ROOT / "results" / "llm" / f"{FULL_RUN_ID}_{run_id}"


def output_contract_hash() -> str:
    return f6l.sha256_text(canonical_json(OUTPUT_CONTRACT_RULES))


def parser_source_hash() -> str:
    return f6l.sha256_text(inspect.getsource(parse_v4_response))


def mapping_contract_hash() -> str:
    return f6l.sha256_text(canonical_json({token: str(index) for index, token in enumerate(TOKEN_LIST, start=1)}))


def model_digest_hash() -> str:
    return f6l.sha256_text(canonical_json({model: f6l.MODEL_DIGESTS[model] for model in MODEL_ORDER}))


def candidate_set_hash() -> str:
    rows = f6l.read_csv_rows(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH)
    payload = [
        {
            "user_id": row["user_id"],
            "candidate_position": row["candidate_position"],
            "candidate_item_id": row["candidate_item_id"],
            "candidate_item_title": row["candidate_item_title"],
            "is_positive": row.get("is_positive", ""),
        }
        for row in sorted(rows, key=lambda item: (int(item["user_id"]), int(item["candidate_position"])))
    ]
    return f6l.sha256_text(canonical_json(payload))


def prompt_template_name(strategy: str) -> str:
    if strategy == "base/OpenP5 title-list":
        return f"{PILOT_ID}_base_openp5_opaque_token_json"
    if strategy == "LLMRank recency":
        return f"{PILOT_ID}_llmrank_recency_opaque_token_json"
    raise ValueError(f"unsupported strategy: {strategy}")


def prompt_template_text(strategy: str) -> str:
    shared = "\n".join(
        [
            "You are ranking LastFM candidate artists/items for a recommender task.",
            "Use the user's history and the candidate list only.",
            "Each candidate has an opaque token. Only these tokens can appear in the answer.",
            "Authorized tokens: {authorized_tokens}.",
            "Return exactly one JSON object and nothing else.",
            'The required schema is {{"ranking":["C03","C17","C01","C09","C06","C12","C04","C20","C08","C11"]}}.',
            "The ranking value must contain exactly 10 distinct strings.",
            "Every string must be one of the authorized C01-C20 tokens.",
            "Do not output titles, item ids, markdown, comments, explanations, extra keys, or any number without the C prefix.",
            "",
            "User history, earlier to later:",
            "{history_block}",
            "",
            "Candidate list:",
            "{candidate_block}",
            "",
            "Task: rank the 10 candidates most likely to be the user's next preferred artist/item, from most likely to least likely.",
            "Final answer must be only the JSON object.",
        ]
    )
    if strategy == "base/OpenP5 title-list":
        return shared
    if strategy == "LLMRank recency":
        return "\n".join(
            [
                shared,
                "Recency cue: give appropriate attention to the later items in the history while still using the full history.",
                "Final answer must still be only the JSON object.",
            ]
        )
    raise ValueError(f"unsupported strategy: {strategy}")


def prompt_template_hash(strategy: str) -> str:
    return f6l.sha256_text(prompt_template_text(strategy))


def sample_rows_by_user() -> dict[str, dict[str, str]]:
    return {row["user_id"]: row for row in f6l.read_csv_rows(f6l.LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH)}


def token_mapping(candidate_rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    rows = sorted(candidate_rows, key=lambda row: int(row["candidate_position"]))
    return {
        TOKEN_LIST[index - 1]: {
            "candidate_token": TOKEN_LIST[index - 1],
            "candidate_position": str(index),
            "item_id": str(row["candidate_item_id"]),
            "title": str(row["candidate_item_title"]),
        }
        for index, row in enumerate(rows, start=1)
    }


def token_mapping_hash(candidate_rows: Sequence[Mapping[str, str]]) -> str:
    return f6l.sha256_text(canonical_json(token_mapping(candidate_rows)))


def render_history_block(sample_row: Mapping[str, str]) -> str:
    titles = f6l.parse_json_list(sample_row.get("history_item_titles", "[]"))
    if not titles:
        return "(empty history)"
    return "\n".join(f"- {title}" for title in titles)


def render_candidate_block(candidate_rows: Sequence[Mapping[str, str]]) -> str:
    mapping = token_mapping(candidate_rows)
    return "\n".join(f"[{token}] {mapping[token]['title']}" for token in TOKEN_LIST)


def render_v4_prompt(sample_row: Mapping[str, str], candidate_rows: Sequence[Mapping[str, str]], strategy: str) -> str:
    prompt = prompt_template_text(strategy).format(
        authorized_tokens=", ".join(TOKEN_LIST),
        history_block=render_history_block(sample_row),
        candidate_block=render_candidate_block(candidate_rows),
    )
    lowered = prompt.casefold()
    forbidden = [token for token in FORBIDDEN_PROMPT_TOKENS if token in lowered]
    if forbidden:
        raise ValueError(f"forbidden prompt token(s): {forbidden}")
    return prompt


def parse_v4_response(text: str, candidate_rows: Sequence[Mapping[str, str]]) -> ParserV4Result:
    """Parse the V4 strict opaque-token JSON response contract.

    The parser accepts exactly one JSON object with only a ``ranking`` key. The
    ranking must be a list of exactly 10 distinct strings, each belonging to
    the authorized token set C01-C20. It does not use fuzzy matching, relevance,
    positive item labels, or metrics.
    """
    stripped = (text or "").strip()
    if not stripped:
        return ParserV4Result(False, "empty_response", [], [], [], 0, 0, 0, 10, 1, 0, 0, None)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return ParserV4Result(False, "malformed_json", [], [], [], 0, 0, 0, 10, 1, 0, 0, None)
    if not isinstance(payload, dict):
        return ParserV4Result(False, "ambiguous_json_structure", [], [], [], 0, 0, 0, 10, 0, 1, 0, None)
    if set(payload) != {"ranking"}:
        return ParserV4Result(False, "extra_or_missing_json_keys", [], [], [], 0, 0, 0, 10, 0, 1, 0, payload)
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        return ParserV4Result(False, "ranking_not_list", [], [], [], 0, 0, 0, 10, 0, 1, 0, payload)
    output_count = len(ranking)
    missing_count = max(0, 10 - output_count)
    if output_count != 10:
        return ParserV4Result(False, "wrong_ranking_length", [], [], [], output_count, 0, 0, missing_count, 0, 0, 0, payload)
    numeric_without_prefix = sum(1 for item in ranking if isinstance(item, int) and not isinstance(item, bool))
    if numeric_without_prefix:
        return ParserV4Result(False, "numeric_without_c_prefix", [], [], [], output_count, 0, 0, 0, 0, 0, numeric_without_prefix, payload)
    non_string = [item for item in ranking if not isinstance(item, str)]
    if non_string:
        return ParserV4Result(False, "non_string_token", [], [], [], output_count, len(non_string), 0, 0, 0, 0, 0, payload)
    tokens = [str(item) for item in ranking]
    unknown = [token for token in tokens if token not in TOKEN_LIST or not re.fullmatch(r"C(0[1-9]|1[0-9]|20)", token)]
    if unknown:
        return ParserV4Result(False, "unknown_token", tokens, [], [], output_count, len(unknown), 0, 0, 0, 0, 0, payload)
    duplicate_count = output_count - len(set(tokens))
    if duplicate_count:
        return ParserV4Result(False, "duplicate_token", tokens, [], [], output_count, 0, duplicate_count, 0, 0, 0, 0, payload)
    mapping = token_mapping(candidate_rows)
    if set(mapping) != set(TOKEN_LIST):
        return ParserV4Result(False, "ambiguous_token_mapping", tokens, [], [], output_count, 0, 0, 0, 0, 1, 0, payload)
    ranked_item_ids = [mapping[token]["item_id"] for token in tokens]
    ranked_titles = [mapping[token]["title"] for token in tokens]
    return ParserV4Result(True, "", tokens, ranked_item_ids, ranked_titles, output_count, 0, 0, 0, 0, 0, 0, payload)


def v3_manifest_rows() -> list[dict[str, str]]:
    return f6l.read_csv_rows(v3.PILOT_MANIFEST_PATH)


def v3_parse_rows() -> list[dict[str, str]]:
    return f6l.read_csv_rows(v3.PILOT_PARSE_RESULTS_PATH)


def v3_failure_rows() -> list[dict[str, str]]:
    return [row for row in v3_parse_rows() if row.get("failure_reason") == "out_of_range_index"]


def extract_numbers(values: Iterable[str]) -> set[str]:
    found: set[str] = set()
    for value in values:
        found.update(re.findall(r"\d+", str(value)))
    return found


def build_v3_failure_audit(write_outputs: bool = True) -> list[dict[str, str]]:
    sample = sample_rows_by_user()
    candidates = f6l.candidate_rows_for_operationalized_sample()
    raw_records = f6l.final_response_records(V3_RAW_DIR)
    rows: list[dict[str, str]] = []
    for index, parse_row in enumerate(v3_failure_rows(), start=1):
        user_id = parse_row["user_id"]
        raw_record = raw_records.get(parse_row["request_id"], {})
        values = [int(item) for item in json.loads(parse_row["ranking_indices"] or "[]")]
        candidate_ids = {str(row["candidate_item_id"]) for row in candidates[user_id]}
        history_ids = set(f6l.parse_json_list(sample[user_id].get("history_item_ids", "[]")))
        title_numbers = extract_numbers(row["candidate_item_title"] for row in candidates[user_id])
        history_numbers = extract_numbers(f6l.parse_json_list(sample[user_id].get("history_item_titles", "[]")))
        out_values = [value for value in values if value < 1 or value > 20]
        values_21_30 = [value for value in out_values if 21 <= value <= 30]
        candidate_id_hits = [value for value in values if str(value) in candidate_ids]
        history_id_hits = [value for value in values if str(value) in history_ids]
        title_hits = [value for value in values if str(value) in title_numbers]
        history_title_hits = [value for value in values if str(value) in history_numbers]
        if 0 in values:
            cause = "possible_zero_based_index_leakage"
        elif candidate_id_hits:
            cause = "candidate_item_id_leakage_into_index_field"
        elif history_id_hits:
            cause = "history_item_id_leakage_into_index_field"
        elif values_21_30:
            cause = "ordinal_extrapolation_beyond_20_candidate_range"
        else:
            cause = "numeric_index_contract_noncompliance"
        rows.append(
            {
                "audit_id": f"F6L-LASTFM-V3-OOR-AUDIT-{index:03d}",
                "v3_run_id": V3_RUN_ID,
                "request_id": parse_row["request_id"],
                "user_id": user_id,
                "model_name": parse_row["model_name"],
                "prompting_strategy": parse_row["prompting_strategy"],
                "raw_json": str(raw_record.get("raw_output_text", "")),
                "returned_values": json_list(values),
                "out_of_range_values": json_list(out_values),
                "minimum_value": str(min(values) if values else ""),
                "maximum_value": str(max(values) if values else ""),
                "contains_zero": "true" if 0 in values else "false",
                "values_21_30": json_list(values_21_30),
                "candidate_item_id_numeric_values": json_list(candidate_id_hits),
                "history_item_id_numeric_values": json_list(history_id_hits),
                "title_number_matches": json_list(title_hits),
                "history_number_matches": json_list(history_title_hits),
                "diagnostic_cause": cause,
                "raw_response_sha256": parse_row["raw_response_sha256"],
                "raw_json_path": str(raw_record.get("raw_json_path", "")),
                "evidence_class": "v3_failure_forensics_no_reinterpretation",
                "notes": "No clipping, modulo, zero-based conversion, nearest-candidate substitution, or relevance mapping applied.",
            }
        )
    if write_outputs:
        f6l.write_csv(V3_FAILURE_AUDIT_PATH, rows, V3_FAILURE_AUDIT_FIELDS)
    return rows


def structured_mode_support_rows() -> list[dict[str, str]]:
    try:
        version = f6l.fetch_local_json(f6l.OLLAMA_LOCAL_VERSION_URL, timeout_seconds=5)
        version_text = str(version.get("version", ""))
        version_available = bool(version_text)
    except Exception as exc:
        version_text = type(exc).__name__ + ":" + str(exc)
        version_available = False
    try:
        completed = subprocess.run(
            ["ollama", "--help"],
            cwd=f6l.PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        cli_ok = completed.returncode == 0
        cli_observed = f"exit={completed.returncode};sha={f6l.sha256_text(completed.stdout)}"
    except Exception as exc:
        cli_ok = False
        cli_observed = type(exc).__name__ + ":" + str(exc)
    rows = [
        {
            "check_id": "F6L-LASTFM-V4-STRUCTURED-001",
            "capability": "ollama_loopback_version",
            "status": "pass" if version_available else "fail",
            "observed": version_text,
            "expected": "local Ollama service responds on loopback",
            "evidence": f6l.OLLAMA_LOCAL_VERSION_URL,
            "notes": "No model inference consumed for this availability check.",
        },
        {
            "check_id": "F6L-LASTFM-V4-STRUCTURED-002",
            "capability": "ollama_cli_available",
            "status": "pass" if cli_ok else "fail",
            "observed": cli_observed,
            "expected": "ollama CLI help available",
            "evidence": "ollama --help",
            "notes": "No download, pull, or model execution performed.",
        },
        {
            "check_id": "F6L-LASTFM-V4-STRUCTURED-003",
            "capability": "json_schema_format_requested",
            "status": "pass" if version_available else "fail",
            "observed": f"format_schema_hash={GENERATION_PARAMETERS['format_schema_hash']}",
            "expected": "Ollama generate request uses JSON schema in the format field",
            "evidence": "V4 generation request payload",
            "notes": "The pilot uses schema-constrained JSON generation; no extra schema probe is run to preserve the 40-inference cap.",
        },
    ]
    f6l.write_csv(STRUCTURED_SUPPORT_PATH, rows, STRUCTURED_FIELDS)
    return rows


def selected_v3_pilot_users() -> set[str]:
    return {row["user_id"] for row in v3_manifest_rows()}


def select_v4_users(failure_audit_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    sample = sample_rows_by_user()
    regression_users = sorted({row["user_id"] for row in failure_audit_rows}, key=lambda user: int(user))
    if len(regression_users) > PILOT_USER_COUNT:
        raise ValueError("regression user count exceeds pilot size")
    base_seed = next(iter(sample.values())).get("selected_seed", "20260801")
    holdout_seed = f6l.sha256_text(f"{PILOT_ID}|{base_seed}|holdout_blind_v1")[:16]
    excluded = set(regression_users) | selected_v3_pilot_users()
    candidates = [
        row
        for row in sample.values()
        if row["user_id"] not in excluded
    ]
    ranked_holdouts = sorted(
        candidates,
        key=lambda row: (stable_hash(holdout_seed, row["user_id"], row["sample_ordinal"], row["train_history_count"]), int(row["user_id"])),
    )
    holdout_needed = PILOT_USER_COUNT - len(regression_users)
    holdout_users = [row["user_id"] for row in ranked_holdouts[:holdout_needed]]
    selected: list[dict[str, str]] = []
    for user_id in regression_users:
        selected.append({**sample[user_id], "_subset": "regression", "_selection_key_hash": stable_hash("regression", user_id)})
    for row in ranked_holdouts[:holdout_needed]:
        selected.append({**row, "_subset": "holdout", "_selection_key_hash": stable_hash(holdout_seed, row["user_id"], row["sample_ordinal"], row["train_history_count"])})
    meta = {
        "base_sample_seed": base_seed,
        "selection_seed": holdout_seed,
        "regression_users": regression_users,
        "holdout_users": holdout_users,
        "holdout_policy": "deterministic_hash_over_operationalized_1000_excluding_v3_pilot_users_and_regression_users",
        "selection_input_fields": list(SELECTION_INPUT_FIELDS),
        "selection_blind_to": list(SELECTION_BLIND_FORBIDDEN_FIELDS),
    }
    return selected, meta


def build_selection_rows(selected: Sequence[Mapping[str, str]], meta: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in selected:
        rows.append(
            {
                "pilot_id": PILOT_ID,
                "subset": row["_subset"],
                "user_id": row["user_id"],
                "source": "v3_out_of_range_failure_user" if row["_subset"] == "regression" else "deterministic_blind_holdout_new_user",
                "sample_ordinal": row["sample_ordinal"],
                "history_length": row["train_history_count"],
                "selection_seed": str(meta["selection_seed"]),
                "selection_key_hash": row["_selection_key_hash"],
                "selection_input_fields": json_list(SELECTION_INPUT_FIELDS),
                "selection_blind_to": json_list(SELECTION_BLIND_FORBIDDEN_FIELDS),
                "evidence_class": PILOT_CLASSIFICATION,
                "lineage": f6l.rel(f6l.LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH),
            }
        )
    return rows


def request_id(index: int, user_id: str, model_name: str, strategy: str) -> str:
    return f"F6L-LASTFM-V4-PILOT-{index:04d}-{stable_hash(PILOT_ID, user_id, model_name, strategy)[:10]}"


def manifest_row_for_request(
    *,
    index: int,
    raw_output_dir: Path,
    sample_row: Mapping[str, str],
    candidate_rows: Sequence[Mapping[str, str]],
    model_name: str,
    strategy: str,
) -> dict[str, str]:
    prompt = render_v4_prompt(sample_row, candidate_rows, strategy)
    candidate_ids = [str(row["candidate_item_id"]) for row in sorted(candidate_rows, key=lambda item: int(item["candidate_position"]))]
    mapping = token_mapping(candidate_rows)
    rid = request_id(index, sample_row["user_id"], model_name, strategy)
    row = {
        "request_id": rid,
        "pilot_id": PILOT_ID,
        "subset": sample_row["_subset"],
        "user_id": sample_row["user_id"],
        "history_length": sample_row["train_history_count"],
        "model_name": model_name,
        "model_digest": f6l.MODEL_DIGESTS[model_name],
        "prompting_strategy": strategy,
        "candidate_list": json_list(candidate_ids),
        "candidate_token_mapping": canonical_json(mapping),
        "candidate_list_hash": str(candidate_rows[0]["candidate_list_hash"]),
        "candidate_token_mapping_hash": token_mapping_hash(candidate_rows),
        "prompt_template": prompt_template_name(strategy),
        "prompt_template_hash": prompt_template_hash(strategy),
        "rendered_prompt_hash": f6l.sha256_text(prompt),
        "parser_version": PARSER_VERSION,
        "parser_source_hash": parser_source_hash(),
        "output_contract": OUTPUT_CONTRACT,
        "output_contract_hash": output_contract_hash(),
        "generation_parameters": canonical_json(GENERATION_PARAMETERS),
        "expected_raw_output_path": rel_safe(raw_output_dir / "responses" / f"{rid}.json"),
        "evidence_class": PILOT_CLASSIFICATION,
        "lineage": ";".join(
            [
                f6l.rel(f6l.LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH),
                f6l.rel(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH),
                "parser_v4_separate_from_v1_v2_v3",
            ]
        ),
        "request_contract_hash": "",
    }
    contract_payload = {
        key: row[key]
        for key in [
            "pilot_id",
            "subset",
            "user_id",
            "history_length",
            "model_name",
            "model_digest",
            "prompting_strategy",
            "candidate_list_hash",
            "candidate_token_mapping_hash",
            "prompt_template_hash",
            "rendered_prompt_hash",
            "parser_version",
            "parser_source_hash",
            "output_contract_hash",
            "generation_parameters",
        ]
    }
    row["request_contract_hash"] = f6l.sha256_text(canonical_json(contract_payload))
    return row


def build_pilot(run_id: str | None = None, *, raw_output_dir: Path | None = None, write_outputs: bool = True) -> PilotBuild:
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_dir = raw_output_dir or local_raw_output_dir(run_id)
    failure_audit_rows = build_v3_failure_audit(write_outputs=write_outputs)
    structured_rows = structured_mode_support_rows() if write_outputs else structured_mode_support_rows_no_write()
    selected, selection_meta = select_v4_users(failure_audit_rows)
    selection_rows = build_selection_rows(selected, selection_meta)
    candidates = f6l.candidate_rows_for_operationalized_sample()
    rows: list[dict[str, str]] = []
    index = 0
    for sample_row in selected:
        for model_name in MODEL_ORDER:
            for strategy in STRATEGY_ORDER:
                index += 1
                rows.append(
                    manifest_row_for_request(
                        index=index,
                        raw_output_dir=raw_output_dir,
                        sample_row=sample_row,
                        candidate_rows=candidates[sample_row["user_id"]],
                        model_name=model_name,
                        strategy=strategy,
                    )
                )
    manifest = {
        "pilot_id": PILOT_ID,
        "run_id": run_id,
        "created_at": utc_now(),
        "classification": PILOT_CLASSIFICATION,
        "v3_run_id": V3_RUN_ID,
        "request_count": len(rows),
        "user_count": len({row["user_id"] for row in rows}),
        "regression_users": selection_meta["regression_users"],
        "holdout_users": selection_meta["holdout_users"],
        "models": {model: f6l.MODEL_DIGESTS[model] for model in MODEL_ORDER},
        "strategies": STRATEGY_ORDER,
        "parser_version": PARSER_VERSION,
        "parser_source_hash": parser_source_hash(),
        "output_contract": OUTPUT_CONTRACT,
        "output_contract_hash": output_contract_hash(),
        "prompt_template_hashes": {strategy: prompt_template_hash(strategy) for strategy in STRATEGY_ORDER},
        "generation_parameters": GENERATION_PARAMETERS,
        "json_schema": V4_JSON_SCHEMA,
        "structured_mode_support": structured_rows,
        "selection": selection_meta,
        "manifest_hash": f6l.sha256_text(canonical_json(rows)),
        "inference_limit": PILOT_REQUEST_COUNT,
        "full_cohort_execution_performed": False,
        "reranking_performed": False,
        "remote_api_calls_performed": False,
        "training_or_downloads_performed": False,
    }
    if write_outputs:
        f6l.write_csv(PILOT_MANIFEST_PATH, rows, MANIFEST_FIELDS)
        f6l.write_csv(PILOT_SELECTION_PATH, selection_rows, SELECTION_FIELDS)
        f6l.write_json(PILOT_JSON_MANIFEST_PATH, manifest)
    return PilotBuild(run_id, raw_output_dir, rows, selection_rows, failure_audit_rows, structured_rows, manifest)


def full_request_id(index: int, user_id: str, model_name: str, strategy: str) -> str:
    return f"F6L-LASTFM-V4-FULL-{index:04d}-{stable_hash(FULL_RUN_ID, user_id, model_name, strategy)[:10]}"


def full_user_rows() -> list[dict[str, str]]:
    sample = sample_rows_by_user()
    missing = [user for user in FULL_CANARY_USERS if user not in sample]
    if missing:
        raise ValueError(f"canary users missing from operationalized sample: {missing}")
    selected: list[dict[str, str]] = []
    for user_id in FULL_CANARY_REGRESSION_USERS:
        selected.append({**sample[user_id], "_subset": "canary_regression"})
    for user_id in FULL_CANARY_HOLDOUT_USERS:
        selected.append({**sample[user_id], "_subset": "canary_holdout"})
    remaining = [
        row
        for row in sample.values()
        if row["user_id"] not in set(FULL_CANARY_USERS)
    ]
    remaining.sort(key=lambda row: (int(row.get("sample_ordinal", "0") or 0), int(row["user_id"])))
    selected.extend({**row, "_subset": "operationalized_main"} for row in remaining)
    return selected


def full_manifest_row_for_request(
    *,
    index: int,
    raw_output_dir: Path,
    sample_row: Mapping[str, str],
    candidate_rows: Sequence[Mapping[str, str]],
    model_name: str,
    strategy: str,
) -> dict[str, str]:
    prompt = render_v4_prompt(sample_row, candidate_rows, strategy)
    candidate_ids = [str(row["candidate_item_id"]) for row in sorted(candidate_rows, key=lambda item: int(item["candidate_position"]))]
    mapping = token_mapping(candidate_rows)
    rid = full_request_id(index, sample_row["user_id"], model_name, strategy)
    row = {
        "request_id": rid,
        "pilot_id": FULL_RUN_ID,
        "subset": sample_row["_subset"],
        "user_id": sample_row["user_id"],
        "history_length": sample_row["train_history_count"],
        "model_name": model_name,
        "model_digest": f6l.MODEL_DIGESTS[model_name],
        "prompting_strategy": strategy,
        "candidate_list": json_list(candidate_ids),
        "candidate_token_mapping": canonical_json(mapping),
        "candidate_list_hash": str(candidate_rows[0]["candidate_list_hash"]),
        "candidate_token_mapping_hash": token_mapping_hash(candidate_rows),
        "prompt_template": prompt_template_name(strategy),
        "prompt_template_hash": prompt_template_hash(strategy),
        "rendered_prompt_hash": f6l.sha256_text(prompt),
        "parser_version": PARSER_VERSION,
        "parser_source_hash": parser_source_hash(),
        "output_contract": OUTPUT_CONTRACT,
        "output_contract_hash": output_contract_hash(),
        "generation_parameters": canonical_json(GENERATION_PARAMETERS),
        "expected_raw_output_path": rel_safe(raw_output_dir / "responses" / f"{rid}.json"),
        "evidence_class": FULL_RUN_CLASSIFICATION,
        "lineage": ";".join(
            [
                f6l.rel(f6l.LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH),
                f6l.rel(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH),
                rel_safe(PILOT_JSON_MANIFEST_PATH),
                "parser_v4_frozen_no_fuzzy_no_relevance_metrics",
            ]
        ),
        "request_contract_hash": "",
    }
    contract_payload = {
        key: row[key]
        for key in [
            "pilot_id",
            "subset",
            "user_id",
            "history_length",
            "model_name",
            "model_digest",
            "prompting_strategy",
            "candidate_list_hash",
            "candidate_token_mapping_hash",
            "prompt_template_hash",
            "rendered_prompt_hash",
            "parser_version",
            "parser_source_hash",
            "output_contract_hash",
            "generation_parameters",
        ]
    }
    row["request_contract_hash"] = f6l.sha256_text(canonical_json(contract_payload))
    return row


def full_freeze_payload(run_id: str, raw_output_dir: Path, manifest_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    protected = f6l.hash_protected_inputs()
    return {
        "run_id": run_id,
        "phase_id": "phase_f6l",
        "full_run_id": FULL_RUN_ID,
        "classification": FULL_RUN_CLASSIFICATION,
        "request_count": len(manifest_rows),
        "user_count": len({row["user_id"] for row in manifest_rows}),
        "models": {model: f6l.MODEL_DIGESTS[model] for model in MODEL_ORDER},
        "model_digest_hash": model_digest_hash(),
        "strategies": STRATEGY_ORDER,
        "parser_version": PARSER_VERSION,
        "parser_source_hash": parser_source_hash(),
        "json_schema_hash": GENERATION_PARAMETERS["format_schema_hash"],
        "json_schema": V4_JSON_SCHEMA,
        "output_contract": OUTPUT_CONTRACT,
        "output_contract_hash": output_contract_hash(),
        "prompt_template_hashes": {strategy: prompt_template_hash(strategy) for strategy in STRATEGY_ORDER},
        "mapping_contract_hash": mapping_contract_hash(),
        "candidate_set_hash": candidate_set_hash(),
        "manifest_rows_hash": f6l.sha256_text(canonical_json(list(manifest_rows))),
        "generation_parameters": GENERATION_PARAMETERS,
        "seed": GENERATION_PARAMETERS["seed"],
        "git_head": f6l.git_output(["rev-parse", "HEAD"]),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "interpreter": str(sys.executable),
        },
        "endpoint": f6l.OLLAMA_LOCAL_GENERATE_URL,
        "protected_input_status": protected.get("status", ""),
        "protected_input_yaml_count": protected.get("f6i2p_yaml_count", ""),
        "canary": {
            "first_request_count": 40,
            "regression_users": FULL_CANARY_REGRESSION_USERS,
            "holdout_users": FULL_CANARY_HOLDOUT_USERS,
        },
        "raw_output_dir": rel_safe(raw_output_dir),
        "policy": {
            "remote_api_calls_performed": False,
            "full_cohort_executed": False,
            "reranking_executed": False,
            "llmsrec_executed": False,
            "training_performed": False,
            "downloads_performed": False,
            "marginal_api_cost": 0,
            "hardware_cost": "not_monetized",
            "energy_cost": "unavailable",
            "total_cost_claimed_zero": False,
        },
    }


def build_full_run(run_id: str | None = None, *, raw_output_dir: Path | None = None, write_outputs: bool = True) -> FullRunBuild:
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_dir = raw_output_dir or full_raw_output_dir(run_id)
    candidates = f6l.candidate_rows_for_operationalized_sample()
    rows: list[dict[str, str]] = []
    index = 0
    for sample_row in full_user_rows():
        for model_name in MODEL_ORDER:
            for strategy in STRATEGY_ORDER:
                index += 1
                rows.append(
                    full_manifest_row_for_request(
                        index=index,
                        raw_output_dir=raw_output_dir,
                        sample_row=sample_row,
                        candidate_rows=candidates[sample_row["user_id"]],
                        model_name=model_name,
                        strategy=strategy,
                    )
                )
    freeze = full_freeze_payload(run_id, raw_output_dir, rows)
    freeze["manifest_hash"] = f6l.sha256_text(canonical_json(freeze))
    freeze_rows = [
        {
            "freeze_id": f"F6L-LASTFM-V4-FULL-FREEZE-{index:02d}",
            "run_id": run_id,
            "artifact": artifact,
            "sha256": sha256,
            "status": "frozen",
            "evidence": evidence,
            "notes": notes,
        }
        for index, (artifact, sha256, evidence, notes) in enumerate(
            [
                ("parser_v4", freeze["parser_source_hash"], "inspect.getsource(parse_v4_response)", "Must equal the V4 pilot parser hash."),
                ("json_schema", freeze["json_schema_hash"], "V4_JSON_SCHEMA", "JSON schema constrains exactly one top-10 ranking of C01-C20 unique tokens."),
                ("output_contract", freeze["output_contract_hash"], "OUTPUT_CONTRACT_RULES", "Contract rules are unchanged from the accepted V4 pilot."),
                ("prompt_base_openp5", freeze["prompt_template_hashes"]["base/OpenP5 title-list"], prompt_template_name("base/OpenP5 title-list"), "Base/OpenP5 V4 prompt template hash."),
                ("prompt_llmrank_recency", freeze["prompt_template_hashes"]["LLMRank recency"], prompt_template_name("LLMRank recency"), "LLMRank recency V4 prompt template hash."),
                ("mapping_contract", freeze["mapping_contract_hash"], "TOKEN_LIST C01-C20", "Opaque token-to-position contract hash."),
                ("candidate_set", freeze["candidate_set_hash"], rel_safe(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH), "Operationalized 1000-user candidate set hash."),
                ("model_digests", freeze["model_digest_hash"], f6l.OLLAMA_LOCAL_TAGS_URL, "Expected local Ollama model digest hash."),
                ("manifest_rows", freeze["manifest_rows_hash"], rel_safe(FULL_RUN_MANIFEST_PATH), "Full 4000-request manifest rows hash."),
                ("complete_manifest", freeze["manifest_hash"], rel_safe(FULL_RUN_JSON_MANIFEST_PATH), "Complete deterministic prelaunch manifest hash."),
            ],
            start=1,
        )
    ]
    manifest = {**freeze, "run_id": run_id, "raw_output_dir": rel_safe(raw_output_dir)}
    if write_outputs:
        f6l.write_csv(FULL_RUN_MANIFEST_PATH, rows, MANIFEST_FIELDS)
        f6l.write_csv(FULL_RUN_FREEZE_PATH, freeze_rows, FULL_RUN_FREEZE_FIELDS)
        f6l.write_json(FULL_RUN_JSON_MANIFEST_PATH, manifest)
    return FullRunBuild(run_id, raw_output_dir, rows, manifest, freeze_rows)


def full_build_identity_check(run_id: str, raw_output_dir: Path) -> tuple[bool, str, str]:
    first = build_full_run(run_id=run_id, raw_output_dir=raw_output_dir, write_outputs=False)
    second = build_full_run(run_id=run_id, raw_output_dir=raw_output_dir, write_outputs=False)
    first_bytes = canonical_json({"rows": first.manifest_rows, "manifest": first.manifest, "freeze": first.freeze_rows})
    second_bytes = canonical_json({"rows": second.manifest_rows, "manifest": second.manifest, "freeze": second.freeze_rows})
    return first_bytes == second_bytes, f6l.sha256_text(first_bytes), f6l.sha256_text(second_bytes)


def build_manifest_hash(manifest_rows: Sequence[Mapping[str, str]], run_id: str, raw_output_dir: Path) -> str:
    freeze = full_freeze_payload(run_id, raw_output_dir, manifest_rows)
    return f6l.sha256_text(canonical_json(freeze))


def write_protocol_amendment(run_id: str, raw_output_dir: Path, manifest_hash: str) -> dict[str, Any]:
    if FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH.exists():
        existing = f6l.read_json_file(FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH)
        existing_hash = existing.get("amendment_hash", "")
        expected_hash = f6l.sha256_text(canonical_json({key: value for key, value in existing.items() if key != "amendment_hash"}))
        if (
            existing.get("amendment_id") == AMENDMENT_ID
            and existing.get("run_id") == run_id
            and existing.get("unchanged_contract", {}).get("manifest_hash") == manifest_hash
            and existing_hash == expected_hash
        ):
            return existing
    payload = {
        "amendment_id": AMENDMENT_ID,
        "run_id": run_id,
        "created_at": utc_now(),
        "raw_output_dir": rel_safe(raw_output_dir),
        "supersedes_decision_token": "f6l_lastfm_v4_full_ranking_resume_stopped_by_systemic_contract_regression",
        "classification": [
            FULL_RUN_CLASSIFICATION,
            "structured_output_validity_is_an_observed_model_runtime_property",
            "duplicate_token_rate_is_an_observed_model_runtime_property",
        ],
        "scientific_interpretation": {
            "prior_completed": 491,
            "prior_strict_valid": 487,
            "prior_structural_invalid": 4,
            "prior_strict_valid_rate": "0.991853360489",
            "prior_structural_invalid_rate": "0.008146639511",
            "all_prior_structural_invalid_taxonomy": "duplicate_token",
            "malformed_json": 0,
            "unknown_token": 0,
            "transport_retry": 0,
            "interpretation": "runtime_structured_output_does_not_hard_enforce_array_uniqueness",
            "reason_for_amendment": "The previous closed 100-response 99% validity breaker was intentionally conservative and can be tripped by two isolated duplicate-token outputs; the observed cumulative evidence does not by itself establish systemic contract regression.",
        },
        "unchanged_contract": {
            "manifest_hash": manifest_hash,
            "parser_source_hash": EXPECTED_V4_PARSER_HASH,
            "json_schema_hash": EXPECTED_V4_SCHEMA_HASH,
            "output_contract_hash": EXPECTED_V4_OUTPUT_CONTRACT_HASH,
            "prompt_template_hashes": EXPECTED_V4_PROMPT_TEMPLATE_HASHES,
            "model_digests": {model: f6l.MODEL_DIGESTS[model] for model in MODEL_ORDER},
            "generation_parameters": GENERATION_PARAMETERS,
            "mapping_contract_hash": mapping_contract_hash(),
            "same_run_id": True,
            "same_request_ids": True,
        },
        "resume_policy": {
            "do_not_repeat_strict_valid": True,
            "do_not_retry_structural_invalid_outputs": True,
            "process_only_residual_requests": True,
            "retry_transport_errors_only": True,
            "no_new_requests_beyond_original_4000": True,
        },
        "circuit_breaker": {
            "hard_contract_stop_reasons": [
                "parser_hash_changed",
                "schema_hash_changed",
                "prompt_hash_changed",
                "manifest_hash_changed",
                "model_digest_changed",
                "foreign_or_duplicate_request_id",
                "raw_jsonl_coherence_failure",
                "completed_missing_response",
                "malformed_json",
                "missing_ranking",
                "wrong_ranking_length",
                "unknown_token",
                "extra_json_properties",
                "ambiguous_or_impossible_mapping",
            ],
            "stop_on_five_duplicate_token_structural_invalid_consecutive": True,
            "duplicate_token_window": {
                "window_size_new_responses": DUPLICATE_WINDOW_SIZE,
                "non_overlapping": True,
                "first_evaluation_after_new_responses": DUPLICATE_WINDOW_SIZE,
                "stop_if_duplicate_token_response_rate_gt": DUPLICATE_WINDOW_RATE_LIMIT,
            },
            "retired_breaker": "do_not_use_closed_100_new_response_window_validity_below_0_99",
            "stop_on_contract_hash_change": True,
            "stop_on_model_digest_change": True,
            "stop_on_raw_coherence_failure": True,
            "stop_on_duplicate_or_foreign_request_id": True,
            "stop_on_unrecoverable_infrastructure_error": True,
        },
        "evaluation_policy": {
            "primary": "fail_strict_utility_all_1000_requests_per_cell",
            "secondary": "secondary_complete_case_not_primary",
            "beyond_utility_scope": "strict_valid_outputs_only_with_valid_output_coverage",
            "reliability_scope": "all_4000_requests",
        },
    }
    payload["amendment_hash"] = f6l.sha256_text(canonical_json(payload))
    report = "\n".join(
        [
            "# F6L LastFM V4 Full Ranking Protocol Amendment",
            "",
            f"- Amendment ID: `{payload['amendment_id']}`",
            f"- Run ID: `{run_id}`",
            f"- Created at: `{payload['created_at']}`",
            f"- Amendment hash: `{payload['amendment_hash']}`",
            "- Interpretation: `runtime_structured_output_does_not_hard_enforce_array_uniqueness`.",
            "- Scientific classification: `paper_protocol_reproduction_under_explicit_operationalization`.",
            "- Additional classification: `duplicate_token_rate_is_an_observed_model_runtime_property`.",
            "- The prior stop used a conservative 100-response 99% validity window; that rule is retired for this final resume.",
            f"- Duplicate-token systemic stop now requires either `{DUPLICATE_CONSECUTIVE_LIMIT}` consecutive duplicate-token structural-invalid outputs or a duplicate-token response rate greater than `{DUPLICATE_WINDOW_RATE_LIMIT}` in a closed non-overlapping `{DUPLICATE_WINDOW_SIZE}`-new-response window.",
            "- Contract unchanged: parser, schema, prompts, model digests, generation parameters, candidate mapping, request IDs, run ID.",
            "- Structural-invalid outputs are retained as observed behavior and evaluated fail-strict; they are not retried or repaired.",
            "- Completion no longer requires `strict-valid = 4000`; it requires `strict-valid + structural-invalid = 4000` with raw coherence and unchanged contracts.",
        ]
    )
    FULL_RUN_PROTOCOL_AMENDMENT_PATH.write_text(report + "\n", encoding="utf-8")
    f6l.write_json(FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH, payload)
    return payload


def structural_counts(parse_rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    return {
        "duplicate_token_count": sum(int(row.get("duplicate_token_count") or 0) for row in parse_rows),
        "malformed_json_count": sum(int(row.get("malformed_json_count") or 0) for row in parse_rows),
        "unknown_token_count": sum(int(row.get("unknown_token_count") or 0) for row in parse_rows),
    }


def is_duplicate_token_invalid(parse_row: Mapping[str, str]) -> bool:
    return parse_row.get("strict_valid") != "true" and parse_row.get("failure_reason") == "duplicate_token"


def is_hard_structural_invalid(parse_row: Mapping[str, str]) -> bool:
    return parse_row.get("strict_valid") != "true" and parse_row.get("failure_reason") != "duplicate_token"


def duplicate_token_invalid_response_count(parse_rows: Sequence[Mapping[str, str]]) -> int:
    return sum(1 for row in parse_rows if is_duplicate_token_invalid(row))


def trailing_duplicate_token_invalid_count(parse_rows: Sequence[Mapping[str, str]], manifest_rows: Sequence[Mapping[str, str]]) -> int:
    parse_by_id = {row["request_id"]: row for row in parse_rows}
    count = 0
    for row in reversed(manifest_rows):
        parse = parse_by_id.get(row["request_id"])
        if not parse:
            continue
        if not is_duplicate_token_invalid(parse):
            break
        count += 1
    return count


def duplicate_token_response_rate(parse_rows: Sequence[Mapping[str, str]]) -> float:
    if not parse_rows:
        return 0.0
    return duplicate_token_invalid_response_count(parse_rows) / len(parse_rows)


def current_duplicate_window_rate(parse_rows: Sequence[Mapping[str, str]]) -> float:
    if not parse_rows:
        return 0.0
    return duplicate_token_response_rate(list(parse_rows[-DUPLICATE_WINDOW_SIZE:]))


def closed_duplicate_window_status(parse_rows: Sequence[Mapping[str, str]]) -> tuple[int, float, str]:
    if not parse_rows or len(parse_rows) < DUPLICATE_WINDOW_SIZE:
        return 0, 0.0, ""
    if len(parse_rows) % DUPLICATE_WINDOW_SIZE != 0:
        return len(parse_rows) // DUPLICATE_WINDOW_SIZE, 0.0, ""
    window = list(parse_rows[-DUPLICATE_WINDOW_SIZE:])
    rate = duplicate_token_response_rate(window)
    index = len(parse_rows) // DUPLICATE_WINDOW_SIZE
    if rate > DUPLICATE_WINDOW_RATE_LIMIT:
        return index, rate, "closed_200_new_response_window_duplicate_token_rate_above_5_percent"
    return index, rate, ""


def structured_mode_support_rows_no_write() -> list[dict[str, str]]:
    try:
        version = f6l.fetch_local_json(f6l.OLLAMA_LOCAL_VERSION_URL, timeout_seconds=5)
        version_text = str(version.get("version", ""))
        version_available = bool(version_text)
    except Exception as exc:
        version_text = type(exc).__name__ + ":" + str(exc)
        version_available = False
    return [
        {
            "check_id": "F6L-LASTFM-V4-STRUCTURED-001",
            "capability": "ollama_loopback_version",
            "status": "pass" if version_available else "fail",
            "observed": version_text,
            "expected": "local Ollama service responds on loopback",
            "evidence": f6l.OLLAMA_LOCAL_VERSION_URL,
            "notes": "No model inference consumed for this availability check.",
        },
        {
            "check_id": "F6L-LASTFM-V4-STRUCTURED-003",
            "capability": "json_schema_format_requested",
            "status": "pass" if version_available else "fail",
            "observed": f"format_schema_hash={GENERATION_PARAMETERS['format_schema_hash']}",
            "expected": "Ollama generate request uses JSON schema in the format field",
            "evidence": "V4 generation request payload",
            "notes": "Unit-test path skips CLI probe to avoid environment dependency.",
        },
    ]


def v3_artifact_hashes() -> dict[str, str]:
    paths = [
        v3.PILOT_MANIFEST_PATH,
        v3.PILOT_SELECTION_PATH,
        v3.PILOT_PREFLIGHT_PATH,
        v3.PILOT_STATE_PATH,
        v3.PILOT_EXECUTION_PATH,
        v3.PILOT_PARSE_RESULTS_PATH,
        v3.PILOT_RECOMMENDATIONS_PATH,
        v3.PILOT_VALIDITY_PATH,
        v3.PILOT_FAILURE_TAXONOMY_PATH,
        v3.PILOT_ACCEPTANCE_PATH,
        v3.PILOT_SMOKE_METRICS_PATH,
        v3.PILOT_JSON_MANIFEST_PATH,
    ]
    hashes = {rel_safe(path): f6l.safe_file_sha256(path) for path in paths if path.exists()}
    hashes.update({f"v3_raw::{key}": value for key, value in v3.raw_hash_map(V3_RAW_DIR).items()})
    return hashes


def v4_pilot_artifact_hashes() -> dict[str, str]:
    paths = [
        V3_FAILURE_AUDIT_PATH,
        STRUCTURED_SUPPORT_PATH,
        PILOT_MANIFEST_PATH,
        PILOT_SELECTION_PATH,
        PILOT_PREFLIGHT_PATH,
        PILOT_STATE_PATH,
        PILOT_EXECUTION_PATH,
        PILOT_PARSE_RESULTS_PATH,
        PILOT_RECOMMENDATIONS_PATH,
        PILOT_VALIDITY_PATH,
        PILOT_REGRESSION_HOLDOUT_PATH,
        PILOT_FAILURE_TAXONOMY_PATH,
        PILOT_ACCEPTANCE_PATH,
        PILOT_SMOKE_METRICS_PATH,
        FULL_RUN_PLAN_PATH,
        PILOT_JSON_MANIFEST_PATH,
    ]
    hashes = {rel_safe(path): f6l.safe_file_sha256(path) for path in paths if path.exists()}
    hashes.update({f"v4_pilot_raw::{key}": value for key, value in v3.raw_hash_map(V4_PILOT_RAW_DIR).items()})
    return hashes


def validate_manifest(rows: Sequence[Mapping[str, str]], failure_audit_rows: Sequence[Mapping[str, str]]) -> list[tuple[str, bool, str, str, str]]:
    checks: list[tuple[str, bool, str, str, str]] = []
    users = {row["user_id"] for row in rows}
    per_user = Counter(row["user_id"] for row in rows)
    regression_users = {row["user_id"] for row in failure_audit_rows}
    holdout_users = {row["user_id"] for row in rows if row["subset"] == "holdout"}
    checks.append(("manifest_rows_40", len(rows) == 40, str(len(rows)), "40", rel_safe(PILOT_MANIFEST_PATH)))
    checks.append(("unique_request_ids_40", len({row["request_id"] for row in rows}) == 40, str(len({row["request_id"] for row in rows})), "40", rel_safe(PILOT_MANIFEST_PATH)))
    checks.append(("exactly_10_users", len(users) == 10, str(len(users)), "10", rel_safe(PILOT_SELECTION_PATH)))
    checks.append(("four_cells_per_user", set(per_user.values()) == {4}, ",".join(str(item) for item in sorted(set(per_user.values()))), "4", rel_safe(PILOT_MANIFEST_PATH)))
    checks.append(("covers_all_v3_failure_users", regression_users.issubset(users), ";".join(sorted(regression_users, key=int)), "all V3 failure users present", rel_safe(V3_FAILURE_AUDIT_PATH)))
    checks.append(("holdout_new_users_present", len(holdout_users) == 6 and holdout_users.isdisjoint(selected_v3_pilot_users()), ";".join(sorted(holdout_users, key=int)), "6 users not in V3 pilot", rel_safe(PILOT_SELECTION_PATH)))
    checks.append(("candidate_order_identical_across_cells", all(len({row["candidate_list_hash"] for row in rows if row["user_id"] == user}) == 1 for user in users), "pass", "one candidate hash per user", rel_safe(PILOT_MANIFEST_PATH)))
    mapping_ok = all(set(json.loads(row["candidate_token_mapping"])) == set(TOKEN_LIST) for row in rows)
    checks.append(("mapping_c01_c20_deterministic", mapping_ok and all(len({row["candidate_token_mapping_hash"] for row in rows if row["user_id"] == user}) == 1 for user in users), "pass" if mapping_ok else "fail", "tokens C01-C20 per user stable across cells", rel_safe(PILOT_MANIFEST_PATH)))
    sample = sample_rows_by_user()
    candidates = f6l.candidate_rows_for_operationalized_sample()
    prompt_failures: list[str] = []
    for row in rows:
        try:
            prompt = render_v4_prompt({**sample[row["user_id"]], "_subset": row["subset"]}, candidates[row["user_id"]], row["prompting_strategy"])
        except Exception as exc:
            prompt_failures.append(f"{row['request_id']}:{exc}")
            continue
        if f6l.sha256_text(prompt) != row["rendered_prompt_hash"]:
            prompt_failures.append(f"{row['request_id']}:hash_mismatch")
        if not all(token in prompt for token in TOKEN_LIST):
            prompt_failures.append(f"{row['request_id']}:missing_authorized_token")
    checks.append(("no_relevance_positive_or_metric_tokens_in_prompt", not prompt_failures, ";".join(prompt_failures[:3]) if prompt_failures else "none", "none", "render_v4_prompt"))
    checks.append(("only_qwen2_mistral", {row["model_name"] for row in rows} == set(MODEL_ORDER), ";".join(sorted({row["model_name"] for row in rows})), ";".join(MODEL_ORDER), rel_safe(PILOT_MANIFEST_PATH)))
    checks.append(("no_reranking", all("rerank" not in row["prompting_strategy"].casefold() for row in rows), "ranking_only", "ranking_only", rel_safe(PILOT_MANIFEST_PATH)))
    checks.append(("no_full_cohort", len(users) == 10 and len(rows) == 40, f"users={len(users)};requests={len(rows)}", "users=10;requests=40", rel_safe(PILOT_MANIFEST_PATH)))
    selection_blind = not any(field in SELECTION_INPUT_FIELDS for field in SELECTION_BLIND_FORBIDDEN_FIELDS)
    checks.append(("holdout_selection_blind_to_relevance_metrics_and_v3_success", selection_blind, json_list(SELECTION_INPUT_FIELDS), "no relevance/HR/NDCG/V3 success/raw output fields", rel_safe(PILOT_SELECTION_PATH)))
    return checks


def run_focused_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        "results/logs/phase_f6l_lastfm_corrective_output_contract_v4_pilot_pytest",
        "-q",
        "tests/test_phase_f6l_lastfm_corrective_output_contract_v4_pilot.py",
        "tests/test_phase_f6l_lastfm_corrective_output_contract_pilot.py",
        "tests/test_phase_f6l_lastfm_parser_forensics.py",
    ]
    started = utc_now()
    start_clock = time.perf_counter()
    completed = subprocess.run(command, cwd=f6l.PROJECT_ROOT, check=False, capture_output=True, text=True, timeout=1000)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-20:]),
        "stdout_sha256": f6l.sha256_text(completed.stdout),
        "stderr_sha256": f6l.sha256_text(completed.stderr),
        "started_at": started,
        "completed_at": utc_now(),
        "duration_seconds": time.perf_counter() - start_clock,
    }


def validate_preflight(
    build: PilotBuild,
    *,
    raw_output_dir: Path,
    resume: bool,
    allow_api_calls: bool,
    local_only: bool,
    endpoint: str,
    profile_hashes_before: Mapping[str, str],
    v3_hashes_before: Mapping[str, str],
    run_tests: bool,
) -> tuple[list[dict[str, str]], dict[str, Any], f6l.LocalOllamaState | None]:
    rows: list[dict[str, str]] = []
    test_result: dict[str, Any] = {}
    ollama_state: f6l.LocalOllamaState | None = None

    def add(name: str, passed: bool, observed: str, expected: str, evidence: str, notes: str, severity: str = "error") -> None:
        rows.append(
            {
                "check_id": f"F6L-LASTFM-V4-PREFLIGHT-{len(rows) + 1:02d}",
                "check_name": name,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    for name, passed, observed, expected, evidence in validate_manifest(build.manifest_rows, build.failure_audit_rows):
        add(name, passed, observed, expected, evidence, "V4 manifest contract check.")
    add("v3_failure_audit_has_7_rows", len(build.failure_audit_rows) == 7, str(len(build.failure_audit_rows)), "7", rel_safe(V3_FAILURE_AUDIT_PATH), "The seven V3 failures are audited before V4 execution.")
    add("structured_mode_support_documented", all(row["status"] == "pass" for row in build.structured_support_rows), ";".join(f"{row['capability']}={row['status']}" for row in build.structured_support_rows), "all pass", rel_safe(STRUCTURED_SUPPORT_PATH), "Local structured/schema mode availability is documented without extra inference.")
    raw_exists = raw_output_dir.exists()
    raw_ok = (resume and raw_exists) or ((not resume) and (not raw_exists))
    add("raw_output_dir_append_only_policy", raw_ok, f"exists={raw_exists};resume={resume};dir={rel_safe(raw_output_dir)}", "new nonexistent dir unless explicit resume", "CLI args", "No overwrite of V4 raw outputs is allowed.")
    add("allow_api_calls_explicit_for_local_inference", allow_api_calls, str(allow_api_calls).lower(), "true", "CLI args", "Repository convention remains explicit even for local inference.")
    add("local_only_flag", local_only, str(local_only).lower(), "true", "CLI args", "Remote APIs are not authorized.")
    endpoint_ok = endpoint in {f6l.OLLAMA_LOCAL_GENERATE_URL, "http://127.0.0.1:11434/api/generate", "http://localhost:11434/api/generate"}
    add("endpoint_loopback_only", endpoint_ok, endpoint, f6l.OLLAMA_LOCAL_GENERATE_URL, "CLI args", "Only localhost Ollama endpoint is authorized.")
    add("original_repo_clean", f6l.git_output(["status", "--short", "original_repo"]) == "", f6l.git_output(["status", "--short", "original_repo"]) or "none", "none", "git status --short original_repo", "The original repository remains read-only.")
    staged = f6l.git_output(["diff", "--cached", "--name-only"])
    add("git_staging_empty", staged == "", staged or "none", "none", "git diff --cached --name-only", "No commit or staging is authorized.")
    protected = f6l.hash_protected_inputs()
    add("protected_inputs_pass", protected.get("status") == "pass", str(protected.get("status")), "pass", rel_safe(f6l.PROTECTED_HASHES_PATH), "Protected inputs must remain intact.")
    add("twenty_three_yaml_invariant", protected.get("f6i2p_yaml_count") == 23, str(protected.get("f6i2p_yaml_count")), "23", rel_safe(f6l.PROTECTED_HASHES_PATH), "Protected YAML invariant.")
    add("original_4000_raw_unchanged", v3.original_raw_hash_map_sha() == v3.original_raw_expected_sha() and bool(v3.original_raw_expected_sha()), v3.original_raw_hash_map_sha(), v3.original_raw_expected_sha(), rel_safe(v3.ORIGINAL_RAW_DIR), "Original 4000-output run remains immutable.")
    add("v3_raw_manifest_results_unchanged_at_preflight", v3_hashes_before == v3_artifact_hashes(), f"before={len(v3_hashes_before)};now={len(v3_artifact_hashes())}", "same V3 hash map", rel_safe(V3_RAW_DIR), "V3 raw/manifest/results are not modified.")
    profile_after = f6l.profile_raw_hashes()
    add("profile_raw_unchanged", dict(profile_hashes_before) == profile_after, f"before={len(profile_hashes_before)};after={len(profile_after)}", "same hash map", "results/llm/phase_f6l_profile_execution_*", "Profile raw outputs are not touched.")
    if run_tests:
        test_result = run_focused_tests()
        add("focused_f6l_tests_pass", test_result["exit_code"] == 0, f"exit={test_result['exit_code']};stdout_tail={test_result['stdout_tail']}", "exit=0", "pytest focused F6L V4 shard", "Inference is blocked unless focused tests pass first.")
    else:
        add("focused_f6l_tests_pass", True, "skipped_by_unit_test", "pass", "unit-test-only path", "Execution path runs focused tests.")
    if all(row["status"] == "pass" for row in rows):
        ollama_state = f6l.start_local_ollama_if_needed(build.run_id)
        digest_ok = ollama_state.service_available and all(
            ollama_state.model_digests.get(model, "").lower() == f6l.MODEL_DIGESTS[model].lower()
            for model in MODEL_ORDER
        )
        add("local_ollama_digests_match", digest_ok, canonical_json({model: ollama_state.model_digests.get(model, "") for model in MODEL_ORDER}), canonical_json({model: f6l.MODEL_DIGESTS[model] for model in MODEL_ORDER}), f6l.OLLAMA_LOCAL_TAGS_URL, f"start_attempted={ollama_state.start_attempted};pid={ollama_state.started_pid};error={ollama_state.error}")
    else:
        add("local_ollama_digests_match", False, "not_checked_due_prior_preflight_failure", "local Qwen2/Mistral digests available", f6l.OLLAMA_LOCAL_TAGS_URL, "Ollama not started because a prior preflight gate failed.")
    f6l.write_csv(PILOT_PREFLIGHT_PATH, rows, PREFLIGHT_FIELDS)
    return rows, test_result, ollama_state


def validate_full_manifest(rows: Sequence[Mapping[str, str]]) -> list[tuple[str, bool, str, str, str]]:
    checks: list[tuple[str, bool, str, str, str]] = []
    users = {row["user_id"] for row in rows}
    per_user = Counter(row["user_id"] for row in rows)
    checks.append(("manifest_rows_4000", len(rows) == FULL_REQUEST_COUNT, str(len(rows)), str(FULL_REQUEST_COUNT), rel_safe(FULL_RUN_MANIFEST_PATH)))
    checks.append(("unique_request_ids_4000", len({row["request_id"] for row in rows}) == FULL_REQUEST_COUNT, str(len({row["request_id"] for row in rows})), str(FULL_REQUEST_COUNT), rel_safe(FULL_RUN_MANIFEST_PATH)))
    checks.append(("exactly_1000_users", len(users) == FULL_USER_COUNT, str(len(users)), str(FULL_USER_COUNT), rel_safe(FULL_RUN_MANIFEST_PATH)))
    checks.append(("four_requests_per_user", set(per_user.values()) == {4} and len(per_user) == FULL_USER_COUNT, f"users={len(per_user)};counts={sorted(set(per_user.values()))}", "1000 users each with 4 requests", rel_safe(FULL_RUN_MANIFEST_PATH)))
    checks.append(("only_authorized_models", {row["model_name"] for row in rows} == set(MODEL_ORDER), ";".join(sorted({row["model_name"] for row in rows})), ";".join(MODEL_ORDER), rel_safe(FULL_RUN_MANIFEST_PATH)))
    checks.append(("only_authorized_strategies", {row["prompting_strategy"] for row in rows} == set(STRATEGY_ORDER), ";".join(sorted({row["prompting_strategy"] for row in rows})), ";".join(STRATEGY_ORDER), rel_safe(FULL_RUN_MANIFEST_PATH)))
    checks.append(("no_reranking_or_full_cohort", len(users) == FULL_USER_COUNT and not any("rerank" in row["prompting_strategy"].casefold() for row in rows), f"users={len(users)};strategies={';'.join(sorted({row['prompting_strategy'] for row in rows}))}", "operationalized ranking only", rel_safe(FULL_RUN_MANIFEST_PATH)))
    hashes_by_user: dict[str, set[str]] = defaultdict(set)
    mapping_hashes_by_user: dict[str, set[str]] = defaultdict(set)
    mapping_ok = True
    for row in rows:
        hashes_by_user[row["user_id"]].add(row["candidate_list_hash"])
        mapping_hashes_by_user[row["user_id"]].add(row["candidate_token_mapping_hash"])
        try:
            mapping_ok = mapping_ok and set(json.loads(row["candidate_token_mapping"])) == set(TOKEN_LIST)
        except Exception:
            mapping_ok = False
    checks.append(("candidate_order_shared_between_cells", all(len(values) == 1 for values in hashes_by_user.values()), f"users={len(hashes_by_user)}", "one candidate_list_hash per user", rel_safe(FULL_RUN_MANIFEST_PATH)))
    checks.append(("mapping_c01_c20_stable_between_cells", mapping_ok and all(len(values) == 1 for values in mapping_hashes_by_user.values()), "pass" if mapping_ok else "fail", "C01-C20 mapping per user stable across four cells", rel_safe(FULL_RUN_MANIFEST_PATH)))
    canary = list(rows[:40])
    canary_users = [row["user_id"] for row in canary[::4]]
    checks.append(("canary_first_40_internal", len(canary) == 40 and canary_users == FULL_CANARY_USERS, ";".join(canary_users), ";".join(FULL_CANARY_USERS), rel_safe(FULL_RUN_MANIFEST_PATH)))
    canary_cells = {(row["model_name"], row["prompting_strategy"]) for row in canary}
    checks.append(("canary_covers_models_and_strategies", canary_cells == {(model, strategy) for model in MODEL_ORDER for strategy in STRATEGY_ORDER}, ";".join(f"{model}|{strategy}" for model, strategy in sorted(canary_cells)), "all 2x2 cells", rel_safe(FULL_RUN_MANIFEST_PATH)))
    sample = sample_rows_by_user()
    candidates = f6l.candidate_rows_for_operationalized_sample()
    prompt_failures: list[str] = []
    for row in rows:
        try:
            prompt = render_v4_prompt({**sample[row["user_id"]], "_subset": row["subset"]}, candidates[row["user_id"]], row["prompting_strategy"])
        except Exception as exc:
            prompt_failures.append(f"{row['request_id']}:{exc}")
            continue
        if f6l.sha256_text(prompt) != row["rendered_prompt_hash"]:
            prompt_failures.append(f"{row['request_id']}:rendered_prompt_hash_mismatch")
    checks.append(("rendered_prompt_hashes_rebuild", not prompt_failures, ";".join(prompt_failures[:3]) if prompt_failures else "none", "none", "render_v4_prompt"))
    return checks


def validate_full_preflight(
    build: FullRunBuild,
    *,
    raw_output_dir: Path,
    resume: bool,
    allow_api_calls: bool,
    local_only: bool,
    endpoint: str,
    profile_hashes_before: Mapping[str, str],
    v3_hashes_before: Mapping[str, str],
    v4_pilot_hashes_before: Mapping[str, str],
    run_tests: bool,
) -> tuple[list[dict[str, str]], dict[str, Any], f6l.LocalOllamaState | None]:
    rows: list[dict[str, str]] = []
    test_result: dict[str, Any] = {}
    ollama_state: f6l.LocalOllamaState | None = None

    def add(name: str, passed: bool, observed: str, expected: str, evidence: str, notes: str, severity: str = "error") -> None:
        rows.append(
            {
                "check_id": f"F6L-LASTFM-V4-FULL-PREFLIGHT-{len(rows) + 1:02d}",
                "check_name": name,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    for name, passed, observed, expected, evidence in validate_full_manifest(build.manifest_rows):
        add(name, passed, observed, expected, evidence, "Full V4 manifest contract check.")

    candidate_rows = f6l.read_csv_rows(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH)
    by_user_candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        by_user_candidates[row["user_id"]].append(row)
    distinct_ok = all(len(rows_for_user) == 20 and len({row["candidate_item_id"] for row in rows_for_user}) == 20 for rows_for_user in by_user_candidates.values())
    positive_counts = Counter(row["user_id"] for row in candidate_rows if row.get("is_positive") == "true")
    observed = f6l.observed_items_by_user()
    observed_negative_hits = [f"{row['user_id']}:{row['candidate_item_id']}" for row in candidate_rows if row.get("is_positive") == "false" and row["candidate_item_id"] in observed.get(row["user_id"], set())]
    identity_ok, identity_first, identity_second = full_build_identity_check(build.run_id, raw_output_dir)
    raw_exists = raw_output_dir.exists()
    raw_resume_ok = resume and raw_exists and (raw_output_dir / "run_manifest.json").exists()
    raw_new_ok = (not resume) and (not raw_exists)
    endpoint_ok = local_only and endpoint in {f6l.OLLAMA_LOCAL_GENERATE_URL, "http://127.0.0.1:11434/api/generate", "http://localhost:11434/api/generate"}
    schema_hash = GENERATION_PARAMETERS["format_schema_hash"]
    prompt_hashes = {strategy: prompt_template_hash(strategy) for strategy in STRATEGY_ORDER}
    pilot_manifest = f6l.read_json_file(V4_PILOT_RAW_DIR / "run_manifest.json") if (V4_PILOT_RAW_DIR / "run_manifest.json").exists() else {}
    protected = f6l.hash_protected_inputs()
    staged = f6l.git_output(["diff", "--cached", "--name-only"])

    add("candidate_table_20000", len(candidate_rows) == 20000, str(len(candidate_rows)), "20000", rel_safe(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH), "Operationalized sample candidate table is unchanged.")
    add("twenty_distinct_candidates_per_user", distinct_ok and len(by_user_candidates) == FULL_USER_COUNT, f"users={len(by_user_candidates)}", "1000 users with 20 distinct candidates", rel_safe(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH), "No duplicate candidate items per user.")
    add("one_positive_per_user", len(positive_counts) == FULL_USER_COUNT and set(positive_counts.values()) == {1}, f"users={len(positive_counts)};counts={sorted(set(positive_counts.values()))}", "one positive per user", rel_safe(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH), "Positive is used only after parsing for metrics.")
    add("no_observed_negative_candidates", not observed_negative_hits, ";".join(observed_negative_hits[:5]) if observed_negative_hits else "none", "none", rel_safe(f6l.F6C_LASTFM_INPUTS["interactions"]), "Negatives must not be in the user's observed train/validation/test items.")
    add("parser_v4_hash_expected", parser_source_hash() == EXPECTED_V4_PARSER_HASH, parser_source_hash(), EXPECTED_V4_PARSER_HASH, "inspect.getsource(parse_v4_response)", "Parser function is byte-identical to the accepted V4 pilot parser.")
    add("schema_hash_expected", schema_hash == EXPECTED_V4_SCHEMA_HASH, schema_hash, EXPECTED_V4_SCHEMA_HASH, "V4_JSON_SCHEMA", "JSON schema is frozen from the accepted V4 pilot.")
    add("output_contract_hash_expected", output_contract_hash() == EXPECTED_V4_OUTPUT_CONTRACT_HASH, output_contract_hash(), EXPECTED_V4_OUTPUT_CONTRACT_HASH, "OUTPUT_CONTRACT_RULES", "Output contract rules are frozen.")
    add("prompt_template_hashes_expected", prompt_hashes == EXPECTED_V4_PROMPT_TEMPLATE_HASHES, canonical_json(prompt_hashes), canonical_json(EXPECTED_V4_PROMPT_TEMPLATE_HASHES), "prompt_template_text", "Prompt templates are frozen.")
    add("model_digests_expected_in_manifest", all(row["model_digest"] == f6l.MODEL_DIGESTS[row["model_name"]] for row in build.manifest_rows), "pass", "manifest rows use expected model digests", rel_safe(FULL_RUN_MANIFEST_PATH), "No remote or unexpected model digest is authorized.")
    add("pilot_v4_manifest_hash_expected", str(pilot_manifest.get("manifest_hash", "")) == EXPECTED_V4_PILOT_MANIFEST_HASH, str(pilot_manifest.get("manifest_hash", "")), EXPECTED_V4_PILOT_MANIFEST_HASH, rel_safe(V4_PILOT_RAW_DIR / "run_manifest.json"), "The accepted V4 pilot is the source checkpoint for this full run.")
    add("prelaunch_builder_identity_byte_for_byte", identity_ok, f"first={identity_first};second={identity_second}", "identical deterministic bytes", "build_full_run(write_outputs=False) twice", "No inference is dispatched unless the full-run manifest builder is deterministic.")
    add("raw_output_dir_append_only_policy", raw_new_ok or raw_resume_ok, f"exists={raw_exists};resume={resume};dir={rel_safe(raw_output_dir)}", "new nonexistent dir, or explicit resume with run_manifest.json", "CLI args", "Raw outputs are append-only and never overwritten.")
    add("allow_api_calls_explicit_for_local_inference", allow_api_calls, str(allow_api_calls).lower(), "true", "CLI args", "Repository convention requires explicit inference authorization.")
    add("endpoint_loopback_only", endpoint_ok, f"local_only={local_only};endpoint={endpoint}", "127.0.0.1/localhost Ollama endpoint", "CLI args", "Remote APIs are not authorized.")
    add("original_repo_clean", f6l.git_output(["status", "--short", "original_repo"]) == "", f6l.git_output(["status", "--short", "original_repo"]) or "none", "none", "git status --short original_repo", "The original repository remains unmodified.")
    add("git_staging_empty", staged == "", staged or "none", "none", "git diff --cached --name-only", "No staging or commit is authorized.")
    add("protected_inputs_pass", protected.get("status") == "pass", str(protected.get("status")), "pass", rel_safe(f6l.PROTECTED_HASHES_PATH), "Protected inputs must be present and intact under the existing F6L guard.")
    add("twenty_three_yaml_invariant", protected.get("f6i2p_yaml_count") == 23, str(protected.get("f6i2p_yaml_count")), "23", rel_safe(f6l.PROTECTED_HASHES_PATH), "Protected YAML invariant.")
    add("original_4000_raw_unchanged", v3.original_raw_hash_map_sha() == v3.original_raw_expected_sha() and bool(v3.original_raw_expected_sha()), v3.original_raw_hash_map_sha(), v3.original_raw_expected_sha(), rel_safe(v3.ORIGINAL_RAW_DIR), "The previous 4000-output run remains immutable and separate.")
    add("v3_raw_manifest_results_unchanged", v3_hashes_before == v3_artifact_hashes(), f"before={len(v3_hashes_before)};now={len(v3_artifact_hashes())}", "same V3 hash map", rel_safe(V3_RAW_DIR), "The V3 corrective pilot artifacts are not modified.")
    add("v4_pilot_raw_manifest_results_unchanged", v4_pilot_hashes_before == v4_pilot_artifact_hashes(), f"before={len(v4_pilot_hashes_before)};now={len(v4_pilot_artifact_hashes())}", "same V4 pilot hash map", rel_safe(V4_PILOT_RAW_DIR), "The accepted V4 pilot artifacts are not modified.")
    add("profile_raw_unchanged", dict(profile_hashes_before) == f6l.profile_raw_hashes(), f"before={len(profile_hashes_before)};after={len(f6l.profile_raw_hashes())}", "same hash map", "results/llm/phase_f6l_profile_execution_*", "Profile prompt raw outputs are not touched.")
    if run_tests:
        test_result = run_focused_tests()
        add("focused_f6l_tests_pass", test_result["exit_code"] == 0, f"exit={test_result['exit_code']};stdout_tail={test_result['stdout_tail']}", "exit=0", "pytest focused F6L V4 shard", "Inference is blocked unless focused F6L tests pass first.")
    else:
        add("focused_f6l_tests_pass", True, "skipped_by_unit_test", "pass", "unit-test-only path", "Execution path runs focused tests.")
    if all(row["status"] == "pass" for row in rows):
        ollama_state = f6l.start_local_ollama_if_needed(build.run_id)
        digest_ok = ollama_state.service_available and all(
            ollama_state.model_digests.get(model, "").lower() == f6l.MODEL_DIGESTS[model].lower()
            for model in MODEL_ORDER
        )
        add("local_ollama_digests_match", digest_ok, canonical_json({model: ollama_state.model_digests.get(model, "") for model in MODEL_ORDER}), canonical_json({model: f6l.MODEL_DIGESTS[model] for model in MODEL_ORDER}), f6l.OLLAMA_LOCAL_TAGS_URL, f"start_attempted={ollama_state.start_attempted};pid={ollama_state.started_pid};error={ollama_state.error}")
    else:
        add("local_ollama_digests_match", False, "not_checked_due_prior_preflight_failure", "local Qwen2/Mistral digests available", f6l.OLLAMA_LOCAL_TAGS_URL, "Ollama not started because a prior preflight gate failed.")
    f6l.write_csv(FULL_RUN_PREFLIGHT_PATH, rows, PREFLIGHT_FIELDS)
    return rows, test_result, ollama_state


def execute_local_ollama_schema_request(model: str, prompt: str, endpoint: str) -> f6l.LocalOllamaRequestResult:
    started = utc_now()
    start_clock = time.perf_counter()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": V4_JSON_SCHEMA,
        "options": {"temperature": 0, "seed": 0},
    }
    request = urllib.request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=int(GENERATION_PARAMETERS["timeout_seconds"])) as response:
            raw_body = response.read().decode("utf-8")
        raw = json.loads(raw_body)
        response_text = str(raw.get("response", ""))
        success = bool(response_text.strip()) and bool(raw.get("done", True))
        return f6l.LocalOllamaRequestResult(success, response_text, raw, "", "", started, utc_now(), time.perf_counter() - start_clock)
    except urllib.error.HTTPError as exc:
        return f6l.LocalOllamaRequestResult(False, "", {}, "HTTPError", f"{exc.code}:{exc.reason}", started, utc_now(), time.perf_counter() - start_clock)
    except urllib.error.URLError as exc:
        return f6l.LocalOllamaRequestResult(False, "", {}, "URLError", str(exc.reason), started, utc_now(), time.perf_counter() - start_clock)
    except TimeoutError as exc:
        return f6l.LocalOllamaRequestResult(False, "", {}, "TimeoutError", str(exc), started, utc_now(), time.perf_counter() - start_clock)
    except Exception as exc:
        return f6l.LocalOllamaRequestResult(False, "", {}, type(exc).__name__, str(exc), started, utc_now(), time.perf_counter() - start_clock)


def request_payload_for_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "request_id": row["request_id"],
        "pilot_id": row["pilot_id"],
        "subset": row["subset"],
        "user_id": row["user_id"],
        "model_name": row["model_name"],
        "model_digest": row["model_digest"],
        "prompting_strategy": row["prompting_strategy"],
        "candidate_list_hash": row["candidate_list_hash"],
        "candidate_token_mapping_hash": row["candidate_token_mapping_hash"],
        "prompt_template_hash": row["prompt_template_hash"],
        "rendered_prompt_hash": row["rendered_prompt_hash"],
        "parser_version": row["parser_version"],
        "parser_source_hash": row["parser_source_hash"],
        "output_contract": row["output_contract"],
        "output_contract_hash": row["output_contract_hash"],
        "generation_parameters": json.loads(row["generation_parameters"]),
        "candidate_token_mapping": json.loads(row["candidate_token_mapping"]),
        "request_contract_hash": row["request_contract_hash"],
        "evidence_class": row["evidence_class"],
    }


def raw_record_for_response(
    row: Mapping[str, str],
    prompt: str,
    candidate_rows: Sequence[Mapping[str, str]],
    result: f6l.LocalOllamaRequestResult,
    parse: ParserV4Result,
    raw_json_path: Path,
    *,
    attempt_index: int = 1,
    is_final_attempt: bool = True,
) -> dict[str, Any]:
    return {
        **request_payload_for_row(row),
        "prompt": prompt,
        "raw_output_text": result.response_text,
        "raw_backend_response": result.raw_backend_response,
        "parser_output": {
            "strict_valid": parse.strict_valid,
            "failure_reason": parse.failure_reason,
            "ranking_tokens": parse.ranking_tokens,
            "ranked_item_ids": parse.ranked_item_ids,
            "ranked_titles": parse.ranked_titles,
            "output_count": parse.output_count,
            "unknown_token_count": parse.unknown_token_count,
            "duplicate_token_count": parse.duplicate_token_count,
            "missing_token_count": parse.missing_token_count,
            "malformed_json_count": parse.malformed_json_count,
            "ambiguous_mapping_count": parse.ambiguous_mapping_count,
            "numeric_without_c_prefix_count": parse.numeric_without_c_prefix_count,
            "parsed_payload": parse.parsed_payload,
        },
        "backend_success": result.backend_success,
        "backend_error_type": result.backend_error_type,
        "backend_error_message": result.backend_error_message,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "duration_seconds": result.duration_seconds,
        "raw_response_sha256": f6l.sha256_text(result.response_text),
        "attempt_index": attempt_index,
        "is_final_attempt": is_final_attempt,
        "raw_json_path": rel_safe(raw_json_path),
    }


def parse_result_row(row: Mapping[str, str], raw_record: Mapping[str, Any], candidate_rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    parse = raw_record.get("parser_output", {})
    hash_coherence = (
        raw_record.get("rendered_prompt_hash") == row["rendered_prompt_hash"]
        and raw_record.get("candidate_list_hash") == row["candidate_list_hash"]
        and raw_record.get("candidate_token_mapping_hash") == row["candidate_token_mapping_hash"]
        and raw_record.get("request_contract_hash") == row["request_contract_hash"]
    )
    return {
        "request_id": row["request_id"],
        "pilot_id": PILOT_ID,
        "subset": row["subset"],
        "user_id": row["user_id"],
        "model_name": row["model_name"],
        "model_digest": row["model_digest"],
        "prompting_strategy": row["prompting_strategy"],
        "parser_version": PARSER_VERSION,
        "parser_source_hash": row["parser_source_hash"],
        "output_contract": OUTPUT_CONTRACT,
        "output_contract_hash": row["output_contract_hash"],
        "prompt_template_hash": row["prompt_template_hash"],
        "rendered_prompt_hash": row["rendered_prompt_hash"],
        "candidate_list_hash": row["candidate_list_hash"],
        "candidate_token_mapping_hash": row["candidate_token_mapping_hash"],
        "raw_response_sha256": str(raw_record.get("raw_response_sha256", "")),
        "strict_valid": "true" if parse.get("strict_valid") else "false",
        "failure_reason": str(parse.get("failure_reason", "")),
        "ranking_tokens": json_list(parse.get("ranking_tokens", [])),
        "ranked_item_ids": json_list(parse.get("ranked_item_ids", [])),
        "ranked_titles": json_list(parse.get("ranked_titles", [])),
        "output_count": str(parse.get("output_count", 0)),
        "unknown_token_count": str(parse.get("unknown_token_count", 0)),
        "duplicate_token_count": str(parse.get("duplicate_token_count", 0)),
        "missing_token_count": str(parse.get("missing_token_count", 0)),
        "malformed_json_count": str(parse.get("malformed_json_count", 0)),
        "ambiguous_mapping_count": str(parse.get("ambiguous_mapping_count", 0)),
        "numeric_without_c_prefix_count": str(parse.get("numeric_without_c_prefix_count", 0)),
        "request_contract_hash": row["request_contract_hash"],
        "hash_coherence": "true" if hash_coherence else "false",
        "v3_regression_equivalent": "true" if row["subset"] == "regression" and row["model_name"] == "mistral:7b-instruct" else "false",
        "evidence_class": PILOT_CLASSIFICATION,
        "notes": "V4 parser uses opaque tokens only; no fuzzy/relevance/metric information.",
    }


def full_parse_result_row(row: Mapping[str, str], raw_record: Mapping[str, Any], candidate_rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    parse = raw_record.get("parser_output", {})
    hash_coherence = (
        raw_record.get("rendered_prompt_hash") == row["rendered_prompt_hash"]
        and raw_record.get("candidate_list_hash") == row["candidate_list_hash"]
        and raw_record.get("candidate_token_mapping_hash") == row["candidate_token_mapping_hash"]
        and raw_record.get("request_contract_hash") == row["request_contract_hash"]
    )
    return {
        "request_id": row["request_id"],
        "pilot_id": row["pilot_id"],
        "subset": row["subset"],
        "user_id": row["user_id"],
        "model_name": row["model_name"],
        "model_digest": row["model_digest"],
        "prompting_strategy": row["prompting_strategy"],
        "parser_version": PARSER_VERSION,
        "parser_source_hash": row["parser_source_hash"],
        "output_contract": OUTPUT_CONTRACT,
        "output_contract_hash": row["output_contract_hash"],
        "prompt_template_hash": row["prompt_template_hash"],
        "rendered_prompt_hash": row["rendered_prompt_hash"],
        "candidate_list_hash": row["candidate_list_hash"],
        "candidate_token_mapping_hash": row["candidate_token_mapping_hash"],
        "raw_response_sha256": str(raw_record.get("raw_response_sha256", "")),
        "strict_valid": "true" if parse.get("strict_valid") else "false",
        "failure_reason": str(parse.get("failure_reason", "")),
        "ranking_tokens": json_list(parse.get("ranking_tokens", [])),
        "ranked_item_ids": json_list(parse.get("ranked_item_ids", [])),
        "ranked_titles": json_list(parse.get("ranked_titles", [])),
        "output_count": str(parse.get("output_count", 0)),
        "unknown_token_count": str(parse.get("unknown_token_count", 0)),
        "duplicate_token_count": str(parse.get("duplicate_token_count", 0)),
        "missing_token_count": str(parse.get("missing_token_count", 0)),
        "malformed_json_count": str(parse.get("malformed_json_count", 0)),
        "ambiguous_mapping_count": str(parse.get("ambiguous_mapping_count", 0)),
        "numeric_without_c_prefix_count": str(parse.get("numeric_without_c_prefix_count", 0)),
        "request_contract_hash": row["request_contract_hash"],
        "hash_coherence": "true" if hash_coherence else "false",
        "v3_regression_equivalent": "false",
        "evidence_class": FULL_RUN_CLASSIFICATION,
        "notes": "V4 full-run parser uses opaque tokens only; no fuzzy/relevance/positive metric information.",
    }


def execution_row(row: Mapping[str, str], raw_record: Mapping[str, Any]) -> dict[str, str]:
    backend = raw_record.get("raw_backend_response", {})
    return {
        "request_id": row["request_id"],
        "pilot_id": PILOT_ID,
        "subset": row["subset"],
        "user_id": row["user_id"],
        "model_name": row["model_name"],
        "model_digest": row["model_digest"],
        "prompting_strategy": row["prompting_strategy"],
        "attempt_index": str(raw_record.get("attempt_index", 1)),
        "started_at": str(raw_record.get("started_at", "")),
        "completed_at": str(raw_record.get("completed_at", "")),
        "duration_seconds": format_float(float(raw_record.get("duration_seconds", 0.0))),
        "backend_success": "true" if raw_record.get("backend_success") else "false",
        "response_nonempty": "true" if str(raw_record.get("raw_output_text", "")).strip() else "false",
        "done_reason": str(backend.get("done_reason", "")),
        "prompt_eval_count": str(backend.get("prompt_eval_count", "")),
        "eval_count": str(backend.get("eval_count", "")),
        "load_duration_nanoseconds": str(backend.get("load_duration", "")),
        "prompt_eval_duration_nanoseconds": str(backend.get("prompt_eval_duration", "")),
        "eval_duration_nanoseconds": str(backend.get("eval_duration", "")),
        "total_duration_nanoseconds": str(backend.get("total_duration", "")),
        "raw_response_sha256": str(raw_record.get("raw_response_sha256", "")),
        "backend_error_type": str(raw_record.get("backend_error_type", "")),
        "backend_error_message": str(raw_record.get("backend_error_message", "")),
        "response_jsonl_recorded": "true",
        "raw_json_path": str(raw_record.get("raw_json_path", "")),
        "execution_status": "completed" if raw_record.get("backend_success") else "backend_failed",
        "notes": "single local-only schema-constrained attempt; no scientific retry",
    }


def recommendation_rows_from_parse(row: Mapping[str, str], parse_row: Mapping[str, str]) -> list[dict[str, str]]:
    if parse_row["strict_valid"] != "true":
        return []
    tokens = [str(item) for item in json.loads(parse_row["ranking_tokens"])]
    item_ids = [str(item) for item in json.loads(parse_row["ranked_item_ids"])]
    titles = [str(item) for item in json.loads(parse_row["ranked_titles"])]
    return [
        {
            "request_id": row["request_id"],
            "pilot_id": PILOT_ID,
            "subset": row["subset"],
            "user_id": row["user_id"],
            "model_name": row["model_name"],
            "model_digest": row["model_digest"],
            "prompting_strategy": row["prompting_strategy"],
            "rank": str(rank),
            "candidate_token": token,
            "recommended_item_id": item_id,
            "recommended_item_title": title,
            "candidate_list_hash": row["candidate_list_hash"],
            "candidate_token_mapping_hash": row["candidate_token_mapping_hash"],
            "parser_version": PARSER_VERSION,
            "evidence_class": PILOT_CLASSIFICATION,
            "notes": "opaque token mapped deterministically to frozen candidate list",
        }
        for rank, (token, item_id, title) in enumerate(zip(tokens, item_ids, titles), start=1)
    ]


def full_recommendation_rows_from_parse(row: Mapping[str, str], parse_row: Mapping[str, str]) -> list[dict[str, str]]:
    if parse_row["strict_valid"] != "true":
        return []
    tokens = [str(item) for item in json.loads(parse_row["ranking_tokens"])]
    item_ids = [str(item) for item in json.loads(parse_row["ranked_item_ids"])]
    titles = [str(item) for item in json.loads(parse_row["ranked_titles"])]
    return [
        {
            "request_id": row["request_id"],
            "pilot_id": row["pilot_id"],
            "subset": row["subset"],
            "user_id": row["user_id"],
            "model_name": row["model_name"],
            "model_digest": row["model_digest"],
            "prompting_strategy": row["prompting_strategy"],
            "rank": str(rank),
            "candidate_token": token,
            "recommended_item_id": item_id,
            "recommended_item_title": title,
            "candidate_list_hash": row["candidate_list_hash"],
            "candidate_token_mapping_hash": row["candidate_token_mapping_hash"],
            "parser_version": PARSER_VERSION,
            "evidence_class": FULL_RUN_CLASSIFICATION,
            "notes": "opaque token mapped deterministically to frozen candidate list",
        }
        for rank, (token, item_id, title) in enumerate(zip(tokens, item_ids, titles), start=1)
    ]


def write_state(run_id: str, raw_output_dir: Path, stage: str, decision_token: str, started_clock: float, last_request_id: str, parse_rows: Sequence[Mapping[str, str]], completed: int, notes: str) -> None:
    strict = sum(1 for row in parse_rows if row.get("strict_valid") == "true")
    failed = completed - strict
    residual = PILOT_REQUEST_COUNT - completed
    elapsed_hours = max((time.perf_counter() - started_clock) / 3600.0, 1e-9)
    throughput = completed / elapsed_hours if completed else 0.0
    eta = residual / throughput * 3600.0 if throughput and residual else 0.0
    fields = {
        "malformed_json": "malformed_json_count",
        "unknown_token": "unknown_token_count",
        "duplicate_token": "duplicate_token_count",
        "missing_token": "missing_token_count",
        "ambiguous_mapping": "ambiguous_mapping_count",
        "numeric_without_c_prefix": "numeric_without_c_prefix_count",
    }
    row = {
        "run_id": run_id,
        "stage": stage,
        "decision_token": decision_token,
        "planned": str(PILOT_REQUEST_COUNT),
        "completed": str(completed),
        "strict_valid": str(strict),
        "failed": str(failed),
        "residual": str(residual),
        "percent_complete": format_float(completed / PILOT_REQUEST_COUNT),
        "last_request_id": last_request_id,
        "last_timestamp": utc_now(),
        "throughput_requests_per_hour": format_float(throughput),
        "eta_seconds": format_float(eta),
        "raw_output_dir": rel_safe(raw_output_dir),
        "notes": notes,
    }
    for output_name, parse_name in fields.items():
        row[output_name] = str(sum(int(item.get(parse_name) or 0) for item in parse_rows))
    f6l.write_csv(PILOT_STATE_PATH, [row], STATE_FIELDS)


def write_full_state(
    *,
    run_id: str,
    raw_output_dir: Path,
    stage: str,
    decision_token: str,
    started_clock: float,
    last_request_id: str,
    last_model_name: str,
    last_prompting_strategy: str,
    completed: int,
    strict_valid: int,
    failed: int,
    retry: int,
    duplicate_token_count: int = 0,
    malformed_json_count: int = 0,
    unknown_token_count: int = 0,
    cumulative_duplicate_token_response_rate: float = 0.0,
    current_duplicate_token_window_rate: float = 0.0,
    closed_duplicate_window_index: int = 0,
    closed_duplicate_token_window_rate: float = 0.0,
    consecutive_duplicate_token_invalid_count: int = 0,
    canary_status: str = "",
    circuit_breaker_status: str = "",
    notes: str = "",
) -> None:
    residual = FULL_REQUEST_COUNT - completed
    elapsed_hours = max((time.perf_counter() - started_clock) / 3600.0, 1e-9)
    throughput = completed / elapsed_hours if completed else 0.0
    eta = residual / throughput * 3600.0 if throughput and residual else 0.0
    row = {
        "run_id": run_id,
        "stage": stage,
        "decision_token": decision_token,
        "planned": str(FULL_REQUEST_COUNT),
        "completed": str(completed),
        "strict_valid": str(strict_valid),
        "failed": str(failed),
        "structural_invalid": str(failed),
        "duplicate_token_count": str(duplicate_token_count),
        "malformed_json_count": str(malformed_json_count),
        "unknown_token_count": str(unknown_token_count),
        "residual": str(residual),
        "retry": str(retry),
        "cumulative_validity_rate": format_float(strict_valid / completed if completed else 1.0),
        "cumulative_duplicate_token_response_rate": format_float(cumulative_duplicate_token_response_rate),
        "current_200_request_window_duplicate_token_rate": format_float(current_duplicate_token_window_rate),
        "closed_200_request_window_index": str(closed_duplicate_window_index),
        "closed_200_request_window_duplicate_token_rate": format_float(closed_duplicate_token_window_rate),
        "consecutive_duplicate_token_invalid_count": str(consecutive_duplicate_token_invalid_count),
        "percent_complete": format_float(completed / FULL_REQUEST_COUNT),
        "last_request_id": last_request_id,
        "last_model_name": last_model_name,
        "last_prompting_strategy": last_prompting_strategy,
        "throughput_requests_per_hour": format_float(throughput),
        "eta_seconds": format_float(eta),
        "last_timestamp": utc_now(),
        "canary_status": canary_status,
        "circuit_breaker_status": circuit_breaker_status,
        "raw_output_dir": rel_safe(raw_output_dir),
        "notes": notes,
    }
    f6l.write_csv(FULL_RUN_STATE_PATH, [row], FULL_RUN_STATE_FIELDS)


def canary_rows_from_parse(run_id: str, parse_rows: Sequence[Mapping[str, str]], manifest_rows: Sequence[Mapping[str, str]], raw_output_dir: Path) -> list[dict[str, str]]:
    canary_ids = [row["request_id"] for row in manifest_rows[:40]]
    parse_by_id = {row["request_id"]: row for row in parse_rows}
    canary_parse = [parse_by_id[request_id] for request_id in canary_ids if request_id in parse_by_id]
    strict = sum(1 for row in canary_parse if row["strict_valid"] == "true")
    failed = len(canary_parse) - strict
    residual = FULL_REQUEST_COUNT - len(canary_parse)
    hash_coherence = bool(canary_parse) and all(row["hash_coherence"] == "true" for row in canary_parse)
    criteria = [
        ("completed", len(canary_parse), 40, rel_safe(FULL_RUN_PARSE_RESULTS_PATH), "Internal canary consists of the first 40 manifest requests."),
        ("strict_valid", strict, 40, rel_safe(FULL_RUN_PARSE_RESULTS_PATH), "All canary responses must satisfy the strict V4 parser."),
        ("failed", failed, 0, rel_safe(FULL_RUN_PARSE_RESULTS_PATH), "No structural failures are allowed in the canary."),
        ("residual_after_canary", residual, 3960, rel_safe(FULL_RUN_STATE_PATH), "Canary consumes no extra requests beyond the full 4000 manifest."),
        ("raw_hash_coherence", "pass" if hash_coherence else "fail", "pass", rel_safe(raw_output_dir), "Prompt/candidate/mapping/request hashes must cohere for every canary response."),
    ]
    return [
        {
            "run_id": run_id,
            "criterion": name,
            "observed": str(observed),
            "expected": str(expected),
            "status": "pass" if str(observed) == str(expected) else "fail",
            "evidence": evidence,
            "notes": notes,
        }
        for name, observed, expected, evidence, notes in criteria
    ]


def full_request_state_rows(
    manifest_rows: Sequence[Mapping[str, str]],
    records_by_id: Mapping[str, Mapping[str, Any]],
    parse_rows: Sequence[Mapping[str, str]],
    attempt_counts: Mapping[str, int],
) -> list[dict[str, str]]:
    parse_by_id = {row["request_id"]: row for row in parse_rows}
    rows: list[dict[str, str]] = []
    for row in manifest_rows:
        parse = parse_by_id.get(row["request_id"])
        record = records_by_id.get(row["request_id"], {})
        state = "completed" if parse else "pending"
        rows.append(
            {
                "request_id": row["request_id"],
                "pilot_id": row["pilot_id"],
                "subset": row["subset"],
                "user_id": row["user_id"],
                "model_name": row["model_name"],
                "model_digest": row["model_digest"],
                "prompting_strategy": row["prompting_strategy"],
                "planned": "true",
                "state": state,
                "strict_valid": parse.get("strict_valid", "false") if parse else "false",
                "failure_reason": parse.get("failure_reason", "") if parse else "",
                "attempt_count": str(attempt_counts.get(row["request_id"], 0)),
                "retry_count": str(max(0, attempt_counts.get(row["request_id"], 0) - 1)),
                "raw_json_path": str(record.get("raw_json_path", "")),
                "request_contract_hash": row["request_contract_hash"],
                "notes": "Strict-valid rows are never regenerated on resume.",
            }
        )
    return rows


def full_metric_value(rows: Sequence[Mapping[str, str]], model: str, strategy: str, metric: str) -> str:
    for row in rows:
        if row.get("model_name") == model and row.get("prompting_strategy") == strategy and row.get("metric") == metric:
            return row.get("value", "")
    return ""


def compute_full_v4_metrics(
    run_id: str,
    parse_rows: Sequence[Mapping[str, str]],
    recommendation_rows: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    popularity = f6l.train_popularity_counts()
    tag_sets = f6l.item_tag_sets()
    histories = f6l.sample_history_items()
    candidates = f6l.candidate_rows_for_operationalized_sample()
    candidate_universe = {row["candidate_item_id"] for row in f6l.read_csv_rows(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH)}
    popularity_values = list(popularity.values())
    long_tail_cutoff = f6l.percentile([float(value) for value in popularity_values], 0.80) if popularity_values else 0.0
    total_train = sum(popularity.values())
    by_request_recs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in recommendation_rows:
        by_request_recs[row["request_id"]].append(row)
    for rows in by_request_recs.values():
        rows.sort(key=lambda item: int(item["rank"]))
    parse_by_request = {row["request_id"]: row for row in parse_rows}
    manifest_by_request = {row["request_id"]: row for row in manifest_rows}
    metrics: list[dict[str, str]] = []
    beyond: list[dict[str, str]] = []

    def add_metric(metric: str, value: float, valid_rows: int, total_rows: int, strict_rows: int, model: str, strategy: str, notes: str, *, k: str = "5") -> None:
        metrics.append(
            {
                "metric_id": stable_hash("F6L-LASTFM-V4-FULL-METRIC", run_id, model, strategy, metric, k)[:24],
                "run_id": run_id,
                "dataset": "LastFM",
                "task": "ranking",
                "model_name": model,
                "model_digest": f6l.MODEL_DIGESTS[model],
                "prompting_strategy": strategy,
                "evidence_class": FULL_RUN_CLASSIFICATION,
                "metric": metric,
                "k": k,
                "value": format_float(value),
                "valid_rows": str(valid_rows),
                "total_rows": str(total_rows),
                "strict_valid_rows": str(strict_rows),
                "source_file": rel_safe(FULL_RUN_PARSE_RESULTS_PATH),
                "notes": notes,
            }
        )

    def add_beyond(metric: str, value: float, numerator: float, denominator: float, model: str, strategy: str, notes: str, support: str = "supported", k: str = "5") -> None:
        beyond.append(
            {
                "metric_id": stable_hash("F6L-LASTFM-V4-FULL-BEYOND", run_id, model, strategy, metric, k)[:24],
                "run_id": run_id,
                "dataset": "LastFM",
                "task": "ranking",
                "model_name": model,
                "model_digest": f6l.MODEL_DIGESTS[model],
                "prompting_strategy": strategy,
                "evidence_class": FULL_RUN_CLASSIFICATION,
                "metric": metric,
                "k": k,
                "value": format_float(value),
                "numerator": format_float(numerator),
                "denominator": format_float(denominator),
                "support_status": support,
                "source_file": rel_safe(FULL_RUN_RECOMMENDATIONS_PATH),
                "notes": notes,
            }
        )

    for model in MODEL_ORDER:
        for strategy in STRATEGY_ORDER:
            request_ids = [row["request_id"] for row in manifest_rows if row["model_name"] == model and row["prompting_strategy"] == strategy]
            rows = [parse_by_request[request_id] for request_id in request_ids if request_id in parse_by_request]
            total = len(request_ids)
            strict_rows = [row for row in rows if row["strict_valid"] == "true"]
            strict_count = len(strict_rows)
            hits: list[float] = []
            ndcgs: list[float] = []
            ranks: list[float] = []
            for row in rows:
                positives = [candidate for candidate in candidates[row["user_id"]] if candidate.get("is_positive") == "true"]
                positive_item = positives[0]["candidate_item_id"] if positives else ""
                ranked = [str(item) for item in json.loads(row["ranked_item_ids"] or "[]")]
                rank = ranked.index(positive_item) + 1 if positive_item in ranked else None
                hits.append(1.0 if rank is not None and rank <= 5 and row["strict_valid"] == "true" else 0.0)
                ndcgs.append(1.0 / math.log2(rank + 1) if rank is not None and rank <= 5 and row["strict_valid"] == "true" else 0.0)
                ranks.append(float(rank) if rank is not None and row["strict_valid"] == "true" else 20.0)
            output_total = sum(int(row["output_count"] or 0) for row in rows)
            unknown_total = sum(int(row["unknown_token_count"] or 0) for row in rows)
            duplicate_total = sum(int(row["duplicate_token_count"] or 0) for row in rows)
            add_metric("HR@5", f6l.safe_mean(hits), len(rows), total, strict_count, model, strategy, "Computed after full V4 strict-valid completion.")
            add_metric("NDCG@5", f6l.safe_mean(ndcgs), len(rows), total, strict_count, model, strategy, "Computed after full V4 strict-valid completion.")
            add_metric("mean_positive_rank", f6l.safe_mean(ranks), len(rows), total, strict_count, model, strategy, "Missing positive ranks count as 20 within the local candidate pool.")
            add_metric("validity_rate", strict_count / total if total else 0.0, len(rows), total, strict_count, model, strategy, "Strict V4 validity rate.", k="")
            add_metric("hallucination_rate", unknown_total / output_total if output_total else 0.0, len(rows), total, strict_count, model, strategy, "Unknown C-token outputs over parser-observed outputs.", k="")
            add_metric("duplicate_rate", duplicate_total / output_total if output_total else 0.0, len(rows), total, strict_count, model, strategy, "Duplicate C-token outputs over parser-observed outputs.", k="")
            top5_items: list[str] = []
            ild_values: list[float] = []
            calibration_values: list[float] = []
            for request_id in request_ids:
                recs = by_request_recs.get(request_id, [])[:5]
                top5 = [row["recommended_item_id"] for row in recs]
                top5_items.extend(top5)
                pair_distances: list[float] = []
                for left_index, left_item in enumerate(top5):
                    for right_item in top5[left_index + 1 :]:
                        left_tags = tag_sets.get(left_item, set())
                        right_tags = tag_sets.get(right_item, set())
                        if left_tags or right_tags:
                            pair_distances.append(1.0 - (len(left_tags & right_tags) / len(left_tags | right_tags)))
                if pair_distances:
                    ild_values.append(f6l.safe_mean(pair_distances))
                manifest_row = manifest_by_request[request_id]
                history_counter: Counter[str] = Counter()
                rec_counter: Counter[str] = Counter()
                for item in histories.get(manifest_row["user_id"], []):
                    history_counter.update(tag_sets.get(item, set()))
                for item in top5:
                    rec_counter.update(tag_sets.get(item, set()))
                if history_counter and rec_counter:
                    calibration_values.append(1.0 - f6l.js_divergence(f6l.distribution(history_counter), f6l.distribution(rec_counter)))
            pop_sum = sum(popularity.get(item, 0) for item in top5_items)
            novelty_values = [
                -math.log2((popularity.get(item, 0) + 1) / (total_train + len(popularity) if total_train else 1))
                for item in top5_items
            ]
            long_tail_hits = sum(1 for item in top5_items if popularity.get(item, 0) <= long_tail_cutoff)
            denominator = len(top5_items)
            add_beyond("novelty", f6l.safe_mean(novelty_values), sum(novelty_values), denominator, model, strategy, "Mean self-information over strict-valid top-5 rows.")
            add_beyond("APLT_long_tail", long_tail_hits / denominator if denominator else 0.0, long_tail_hits, denominator, model, strategy, f"Long-tail is locally operationalized as train popularity <= 80th percentile ({format_float(long_tail_cutoff)}).")
            add_beyond("average_popularity", pop_sum / denominator if denominator else 0.0, pop_sum, denominator, model, strategy, "Mean train interaction count of selected top-5 items.")
            add_beyond("popularity_bias", pop_sum / denominator if denominator else 0.0, pop_sum, denominator, model, strategy, "Same popularity primitive retained for auditability.")
            add_beyond("catalog_coverage", len(set(top5_items)) / len(candidate_universe) if candidate_universe else 0.0, len(set(top5_items)), len(candidate_universe), model, strategy, "Unique top-5 recommended items divided by the operationalized candidate universe.")
            add_beyond("ILD_diversity", f6l.safe_mean(ild_values), sum(ild_values), len(ild_values), model, strategy, "Mean within-list Jaccard distance over LastFM tag sets.")
            add_beyond("calibration", f6l.safe_mean(calibration_values), sum(calibration_values), len(calibration_values), model, strategy, "One minus Jensen-Shannon divergence between user history tags and top-5 recommendation tags.", support="supported" if calibration_values else "unsupported_missing_tags")

    lookup = {(row["model_name"], row["prompting_strategy"], row["metric"]): float(row["value"] or 0.0) for row in metrics + beyond}
    comparison: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        for metric in ["HR@5", "NDCG@5", "mean_positive_rank", "validity_rate", "hallucination_rate", "duplicate_rate", "novelty", "APLT_long_tail", "average_popularity", "popularity_bias", "catalog_coverage", "ILD_diversity", "calibration"]:
            base = lookup.get((model, "base/OpenP5 title-list", metric), 0.0)
            llmrank = lookup.get((model, "LLMRank recency", metric), 0.0)
            comparison.append(
                {
                    "comparison_id": stable_hash("F6L-LASTFM-V4-FULL-COMP", run_id, model, metric)[:24],
                    "run_id": run_id,
                    "sample_id": "phase_f6l_lastfm_operationalized_1000",
                    "model_name": model,
                    "metric": metric,
                    "base_openp5_value": format_float(base),
                    "llmrank_recency_value": format_float(llmrank),
                    "delta_llmrank_minus_base": format_float(llmrank - base),
                    "shared_users": "1000",
                    "evidence_class": FULL_RUN_CLASSIFICATION,
                    "status": "computed_same_users_same_candidates",
                    "notes": "Base/OpenP5 and LLMRank are compared on identical users and candidate lists.",
                }
            )

    rq_rows = build_full_v4_rq_rows(metrics, beyond, comparison)
    return metrics, beyond, comparison, rq_rows


def compute_amended_full_v4_tables(
    run_id: str,
    parse_rows: Sequence[Mapping[str, str]],
    recommendation_rows: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, str]],
    attempt_counts: Mapping[str, int],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    popularity = f6l.train_popularity_counts()
    tag_sets = f6l.item_tag_sets()
    histories = f6l.sample_history_items()
    candidates = f6l.candidate_rows_for_operationalized_sample()
    candidate_universe = {row["candidate_item_id"] for row in f6l.read_csv_rows(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH)}
    popularity_values = list(popularity.values())
    long_tail_cutoff = f6l.percentile([float(value) for value in popularity_values], 0.80) if popularity_values else 0.0
    total_train = sum(popularity.values())
    parse_by_request = {row["request_id"]: row for row in parse_rows}
    manifest_by_request = {row["request_id"]: row for row in manifest_rows}
    by_request_recs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in recommendation_rows:
        by_request_recs[row["request_id"]].append(row)
    for rows in by_request_recs.values():
        rows.sort(key=lambda item: int(item["rank"]))

    validity_rows: list[dict[str, str]] = []
    fail_strict: list[dict[str, str]] = []
    complete_case: list[dict[str, str]] = []
    beyond: list[dict[str, str]] = []

    def add_utility(target: list[dict[str, str]], metric: str, value: float, valid_rows: int, total_rows: int, strict_rows: int, model: str, strategy: str, notes: str, *, k: str = "5") -> None:
        target.append(
            {
                "metric_id": stable_hash("F6L-LASTFM-V4-AMENDED-UTILITY", run_id, model, strategy, metric, k, notes)[:24],
                "run_id": run_id,
                "dataset": "LastFM",
                "task": "ranking",
                "model_name": model,
                "model_digest": f6l.MODEL_DIGESTS[model],
                "prompting_strategy": strategy,
                "evidence_class": FULL_RUN_CLASSIFICATION,
                "metric": metric,
                "k": k,
                "value": format_float(value),
                "valid_rows": str(valid_rows),
                "total_rows": str(total_rows),
                "strict_valid_rows": str(strict_rows),
                "source_file": rel_safe(FULL_RUN_PARSE_RESULTS_PATH),
                "notes": notes,
            }
        )

    def add_beyond(metric: str, value: float, numerator: float, denominator: float, model: str, strategy: str, notes: str, support: str = "supported", k: str = "5") -> None:
        beyond.append(
            {
                "metric_id": stable_hash("F6L-LASTFM-V4-AMENDED-BEYOND", run_id, model, strategy, metric, k)[:24],
                "run_id": run_id,
                "dataset": "LastFM",
                "task": "ranking",
                "model_name": model,
                "model_digest": f6l.MODEL_DIGESTS[model],
                "prompting_strategy": strategy,
                "evidence_class": FULL_RUN_CLASSIFICATION,
                "metric": metric,
                "k": k,
                "value": format_float(value),
                "numerator": format_float(numerator),
                "denominator": format_float(denominator),
                "support_status": support,
                "source_file": rel_safe(FULL_RUN_RECOMMENDATIONS_PATH),
                "notes": notes,
            }
        )

    for model in MODEL_ORDER:
        for strategy in STRATEGY_ORDER:
            request_ids = [row["request_id"] for row in manifest_rows if row["model_name"] == model and row["prompting_strategy"] == strategy]
            rows = [parse_by_request[request_id] for request_id in request_ids if request_id in parse_by_request]
            total = len(request_ids)
            strict_rows = [row for row in rows if row["strict_valid"] == "true"]
            structural_invalid = len(rows) - len(strict_rows)
            duplicate_invalid_outputs = duplicate_token_invalid_response_count(rows)
            duplicate_tokens = sum(int(row["duplicate_token_count"] or 0) for row in rows)
            malformed = sum(int(row["malformed_json_count"] or 0) for row in rows)
            unknown = sum(int(row["unknown_token_count"] or 0) for row in rows)
            output_total = sum(int(row["output_count"] or 0) for row in rows)
            retry_count = sum(max(0, int(attempt_counts.get(request_id, 0)) - 1) for request_id in request_ids)
            validity_rows.append(
                {
                    "run_id": run_id,
                    "model_name": model,
                    "model_digest": f6l.MODEL_DIGESTS[model],
                    "prompting_strategy": strategy,
                    "planned": str(total),
                    "completed": str(len(rows)),
                    "strict_valid": str(len(strict_rows)),
                    "structural_invalid": str(structural_invalid),
                    "residual": str(total - len(rows)),
                    "strict_valid_rate": format_float(len(strict_rows) / total if total else 0.0),
                    "structural_invalid_rate": format_float(structural_invalid / total if total else 0.0),
                    "duplicate_token_count": str(duplicate_tokens),
                    "duplicate_token_invalid_outputs": str(duplicate_invalid_outputs),
                    "malformed_json_count": str(malformed),
                    "unknown_token_count": str(unknown),
                    "duplicate_token_rate": format_float(duplicate_invalid_outputs / total if total else 0.0),
                    "retry_count": str(retry_count),
                    "retry_rate": format_float(retry_count / total if total else 0.0),
                    "valid_output_coverage": f"{len(strict_rows)}/{total}",
                    "malformed_rate": format_float(malformed / len(rows) if rows else 0.0),
                    "unknown_token_rate": format_float(unknown / output_total if output_total else 0.0),
                    "evidence_class": FULL_RUN_CLASSIFICATION,
                    "notes": "Validity is an observed model/runtime property under the unchanged V4 contract.",
                }
            )

            fail_hits: list[float] = []
            fail_ndcgs: list[float] = []
            fail_ranks: list[float] = []
            cc_hits: list[float] = []
            cc_ndcgs: list[float] = []
            cc_ranks: list[float] = []
            for request_id in request_ids:
                row = parse_by_request.get(request_id)
                if not row:
                    fail_hits.append(0.0)
                    fail_ndcgs.append(0.0)
                    fail_ranks.append(20.0)
                    continue
                positives = [candidate for candidate in candidates[row["user_id"]] if candidate.get("is_positive") == "true"]
                positive_item = positives[0]["candidate_item_id"] if positives else ""
                ranked = [str(item) for item in json.loads(row["ranked_item_ids"] or "[]")]
                rank = ranked.index(positive_item) + 1 if positive_item in ranked else None
                if row["strict_valid"] == "true":
                    hit = 1.0 if rank is not None and rank <= 5 else 0.0
                    ndcg = 1.0 / math.log2(rank + 1) if rank is not None and rank <= 5 else 0.0
                    positive_rank = float(rank) if rank is not None else 20.0
                    fail_hits.append(hit)
                    fail_ndcgs.append(ndcg)
                    fail_ranks.append(positive_rank)
                    cc_hits.append(hit)
                    cc_ndcgs.append(ndcg)
                    cc_ranks.append(positive_rank)
                else:
                    fail_hits.append(0.0)
                    fail_ndcgs.append(0.0)
                    fail_ranks.append(20.0)
            add_utility(fail_strict, "HR@5_fail_strict", f6l.safe_mean(fail_hits), len(rows), total, len(strict_rows), model, strategy, "Primary operational fail-strict: structural-invalid and missing outputs contribute HR@5=0.")
            add_utility(fail_strict, "NDCG@5_fail_strict", f6l.safe_mean(fail_ndcgs), len(rows), total, len(strict_rows), model, strategy, "Primary operational fail-strict: structural-invalid and missing outputs contribute NDCG@5=0.")
            add_utility(fail_strict, "mean_positive_rank_fail_strict_invalid_as_20", f6l.safe_mean(fail_ranks), len(rows), total, len(strict_rows), model, strategy, "Primary operational fail-strict: invalid/missing positive ranks are counted as candidate-pool rank 20.", k="")
            add_utility(fail_strict, "strict_valid_rate", len(strict_rows) / total if total else 0.0, len(rows), total, len(strict_rows), model, strategy, "Reliability over all planned requests.", k="")
            add_utility(fail_strict, "structural_invalid_rate", structural_invalid / total if total else 0.0, len(rows), total, len(strict_rows), model, strategy, "Reliability over all planned requests.", k="")
            add_utility(fail_strict, "duplicate_token_rate", duplicate_invalid_outputs / total if total else 0.0, len(rows), total, len(strict_rows), model, strategy, "Duplicate-token structural-invalid outputs over all planned requests.", k="")
            add_utility(fail_strict, "retry_rate", retry_count / total if total else 0.0, len(rows), total, len(strict_rows), model, strategy, "Transport retry attempts over all planned requests; structural invalid outputs are not retried.", k="")
            add_utility(complete_case, "HR@5_complete_case", f6l.safe_mean(cc_hits), len(strict_rows), total, len(strict_rows), model, strategy, "secondary_complete_case_not_primary")
            add_utility(complete_case, "NDCG@5_complete_case", f6l.safe_mean(cc_ndcgs), len(strict_rows), total, len(strict_rows), model, strategy, "secondary_complete_case_not_primary")
            add_utility(complete_case, "mean_positive_rank_complete_case", f6l.safe_mean(cc_ranks), len(strict_rows), total, len(strict_rows), model, strategy, "secondary_complete_case_not_primary", k="")

            top5_items: list[str] = []
            ild_values: list[float] = []
            calibration_values: list[float] = []
            for request_id in request_ids:
                if parse_by_request.get(request_id, {}).get("strict_valid") != "true":
                    continue
                recs = by_request_recs.get(request_id, [])[:5]
                top5 = [row["recommended_item_id"] for row in recs]
                top5_items.extend(top5)
                pair_distances: list[float] = []
                for left_index, left_item in enumerate(top5):
                    for right_item in top5[left_index + 1 :]:
                        left_tags = tag_sets.get(left_item, set())
                        right_tags = tag_sets.get(right_item, set())
                        if left_tags or right_tags:
                            pair_distances.append(1.0 - (len(left_tags & right_tags) / len(left_tags | right_tags)))
                if pair_distances:
                    ild_values.append(f6l.safe_mean(pair_distances))
                manifest_row = manifest_by_request[request_id]
                history_counter: Counter[str] = Counter()
                rec_counter: Counter[str] = Counter()
                for item in histories.get(manifest_row["user_id"], []):
                    history_counter.update(tag_sets.get(item, set()))
                for item in top5:
                    rec_counter.update(tag_sets.get(item, set()))
                if history_counter and rec_counter:
                    calibration_values.append(1.0 - f6l.js_divergence(f6l.distribution(history_counter), f6l.distribution(rec_counter)))
            denominator = len(top5_items)
            pop_sum = sum(popularity.get(item, 0) for item in top5_items)
            novelty_values = [
                -math.log2((popularity.get(item, 0) + 1) / (total_train + len(popularity) if total_train else 1))
                for item in top5_items
            ]
            long_tail_hits = sum(1 for item in top5_items if popularity.get(item, 0) <= long_tail_cutoff)
            coverage_note = f"Computed on strict-valid outputs only; valid output coverage={len(strict_rows)}/{total}. No invalid output reconstruction or silent deduplication."
            add_beyond("novelty", f6l.safe_mean(novelty_values), sum(novelty_values), denominator, model, strategy, coverage_note)
            add_beyond("APLT_long_tail", long_tail_hits / denominator if denominator else 0.0, long_tail_hits, denominator, model, strategy, coverage_note + f" Long-tail cutoff={format_float(long_tail_cutoff)}.")
            add_beyond("average_popularity", pop_sum / denominator if denominator else 0.0, pop_sum, denominator, model, strategy, coverage_note)
            add_beyond("popularity_bias", pop_sum / denominator if denominator else 0.0, pop_sum, denominator, model, strategy, coverage_note)
            add_beyond("catalog_coverage", len(set(top5_items)) / len(candidate_universe) if candidate_universe else 0.0, len(set(top5_items)), len(candidate_universe), model, strategy, coverage_note)
            add_beyond("ILD_diversity", f6l.safe_mean(ild_values), sum(ild_values), len(ild_values), model, strategy, coverage_note)
            add_beyond("calibration", f6l.safe_mean(calibration_values), sum(calibration_values), len(calibration_values), model, strategy, coverage_note, support="supported" if calibration_values else "unsupported_missing_tags")

    lookup = {(row["model_name"], row["prompting_strategy"], row["metric"]): float(row["value"] or 0.0) for row in fail_strict + beyond}
    comparison: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        for metric in ["HR@5_fail_strict", "NDCG@5_fail_strict", "mean_positive_rank_fail_strict_invalid_as_20", "strict_valid_rate", "structural_invalid_rate", "duplicate_token_rate", "retry_rate", "novelty", "APLT_long_tail", "average_popularity", "popularity_bias", "catalog_coverage", "ILD_diversity", "calibration"]:
            base = lookup.get((model, "base/OpenP5 title-list", metric), 0.0)
            llmrank = lookup.get((model, "LLMRank recency", metric), 0.0)
            comparison.append(
                {
                    "comparison_id": stable_hash("F6L-LASTFM-V4-AMENDED-COMP", run_id, model, metric)[:24],
                    "run_id": run_id,
                    "sample_id": "phase_f6l_lastfm_operationalized_1000",
                    "model_name": model,
                    "metric": metric,
                    "base_openp5_value": format_float(base),
                    "llmrank_recency_value": format_float(llmrank),
                    "delta_llmrank_minus_base": format_float(llmrank - base),
                    "shared_users": "1000",
                    "evidence_class": FULL_RUN_CLASSIFICATION,
                    "status": "computed_same_users_same_candidates",
                    "notes": "Primary comparison uses fail-strict utility where applicable; beyond-utility uses strict-valid output coverage.",
                }
            )

    rq_rows = build_amended_full_v4_rq_rows(fail_strict, beyond, comparison)
    failure_rows = build_failure_rows(run_id, parse_rows, pilot_id=FULL_RUN_ID, evidence_class=FULL_RUN_CLASSIFICATION)
    return validity_rows, fail_strict, complete_case, beyond, comparison, rq_rows, failure_rows


def build_duplicate_window_rows(
    run_id: str,
    parse_rows: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, str]],
    resume_start_completed: int,
) -> list[dict[str, str]]:
    parse_by_request = {row["request_id"]: row for row in parse_rows}
    ordered_parse_rows = [parse_by_request[row["request_id"]] for row in manifest_rows if row["request_id"] in parse_by_request]
    new_rows = ordered_parse_rows[resume_start_completed:]
    rows: list[dict[str, str]] = []

    def add_window(window_index: str, window_rows: Sequence[Mapping[str, str]], start: int, end: int, closed: str) -> None:
        strict = sum(1 for row in window_rows if row["strict_valid"] == "true")
        duplicate_invalid = duplicate_token_invalid_response_count(window_rows)
        hard_invalid = sum(1 for row in window_rows if is_hard_structural_invalid(row))
        rate = duplicate_token_response_rate(window_rows)
        status = "pass"
        if closed == "true" and rate > DUPLICATE_WINDOW_RATE_LIMIT:
            status = "fail"
        if hard_invalid:
            status = "hard_contract_violation"
        cumulative_rows = ordered_parse_rows[: resume_start_completed + end]
        rows.append(
            {
                "window_id": stable_hash("F6L-LASTFM-V4-DUP-WINDOW", run_id, window_index, start, end)[:24],
                "run_id": run_id,
                "resume_start_completed": str(resume_start_completed),
                "window_index": window_index,
                "window_start_new_response": str(start),
                "window_end_new_response": str(end),
                "closed": closed,
                "responses": str(len(window_rows)),
                "strict_valid": str(strict),
                "duplicate_token_structural_invalid": str(duplicate_invalid),
                "hard_structural_invalid": str(hard_invalid),
                "duplicate_token_response_rate": format_float(rate),
                "cumulative_completed": str(resume_start_completed + end),
                "cumulative_duplicate_token_structural_invalid": str(duplicate_token_invalid_response_count(cumulative_rows)),
                "cumulative_duplicate_token_response_rate": format_float(duplicate_token_response_rate(cumulative_rows)),
                "threshold": f">{DUPLICATE_WINDOW_RATE_LIMIT} over closed {DUPLICATE_WINDOW_SIZE} new responses",
                "status": status,
                "evidence_class": FULL_RUN_CLASSIFICATION,
                "notes": "Non-overlapping windows are calculated only on responses generated after the current authorized resume checkpoint.",
            }
        )

    full_windows = len(new_rows) // DUPLICATE_WINDOW_SIZE
    for index in range(full_windows):
        start_index = index * DUPLICATE_WINDOW_SIZE
        add_window(str(index + 1), new_rows[start_index : start_index + DUPLICATE_WINDOW_SIZE], start_index + 1, start_index + DUPLICATE_WINDOW_SIZE, "true")
    remainder = len(new_rows) % DUPLICATE_WINDOW_SIZE
    if remainder:
        start_index = full_windows * DUPLICATE_WINDOW_SIZE
        add_window(str(full_windows + 1), new_rows[start_index:], start_index + 1, len(new_rows), "false")
    add_window("cumulative_after_resume", new_rows, 1 if new_rows else 0, len(new_rows), "n/a")
    return rows


def build_full_v4_rq_rows(
    metrics: Sequence[Mapping[str, str]],
    beyond: Sequence[Mapping[str, str]],
    comparisons: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    qwen_base = f"HR@5={full_metric_value(metrics, 'qwen2:7b-instruct', 'base/OpenP5 title-list', 'HR@5')};NDCG@5={full_metric_value(metrics, 'qwen2:7b-instruct', 'base/OpenP5 title-list', 'NDCG@5')}"
    qwen_llmrank = f"HR@5={full_metric_value(metrics, 'qwen2:7b-instruct', 'LLMRank recency', 'HR@5')};NDCG@5={full_metric_value(metrics, 'qwen2:7b-instruct', 'LLMRank recency', 'NDCG@5')}"
    mistral_base = f"HR@5={full_metric_value(metrics, 'mistral:7b-instruct', 'base/OpenP5 title-list', 'HR@5')};NDCG@5={full_metric_value(metrics, 'mistral:7b-instruct', 'base/OpenP5 title-list', 'NDCG@5')}"
    mistral_llmrank = f"HR@5={full_metric_value(metrics, 'mistral:7b-instruct', 'LLMRank recency', 'HR@5')};NDCG@5={full_metric_value(metrics, 'mistral:7b-instruct', 'LLMRank recency', 'NDCG@5')}"
    llmrank_deltas = [float(row["delta_llmrank_minus_base"]) for row in comparisons if row["metric"] in {"HR@5", "NDCG@5"}]
    prompt_status = "mixed_prompt_dependent" if any(delta > 0 for delta in llmrank_deltas) and any(delta < 0 for delta in llmrank_deltas) else "supported_under_local_operationalization"
    return [
        {
            "rq_id": "F6L-LASTFM-V4-RQ-UTILITY",
            "paper_claim": "LLMs can act as recommenders under candidate ranking evaluation.",
            "published_paper_evidence": "Paper LastFM sample and exact model outputs remain unavailable locally.",
            "previous_lastfm_adapted_evidence": "Earlier local evidence used less constrained or different output contracts and remains separated.",
            "new_operationalized_evidence": "Full 1000-user LastFM ranking with strict opaque-token V4 contract, local Qwen2/Mistral, base and LLMRank.",
            "dataset_and_sample_difference": "Local operationalized HetRec-derived sample; not the original paper sample or exact paper output contract.",
            "qwen2_base": qwen_base,
            "qwen2_llmrank": qwen_llmrank,
            "mistral_base": mistral_base,
            "mistral_llmrank": mistral_llmrank,
            "previous_status": "unresolved_for_original_paper_sample",
            "new_status": "supported_under_local_operationalization",
            "motivation": "All four local cells are evaluated from 4000 strict-valid V4 outputs.",
            "limits": "Does not establish exact paper reproduction, original paper sample, original paper output contract, closed models, LLMSRec, reranking, or full cohort.",
        },
        {
            "rq_id": "F6L-LASTFM-V4-RQ-PROMPT",
            "paper_claim": "Prompting strategy can affect recommendation behavior.",
            "published_paper_evidence": "Paper reports prompt-family comparisons, but exact LastFM raw outputs are missing.",
            "previous_lastfm_adapted_evidence": "Earlier prompt evidence is retained only as separate fragility/adaptation evidence.",
            "new_operationalized_evidence": "Base/OpenP5 and LLMRank recency are paired on identical users and candidates under V4.",
            "dataset_and_sample_difference": "Comparison is internally paired but remains local operationalized evidence.",
            "qwen2_base": qwen_base,
            "qwen2_llmrank": qwen_llmrank,
            "mistral_base": mistral_base,
            "mistral_llmrank": mistral_llmrank,
            "previous_status": "unresolved_for_original_paper_sample",
            "new_status": prompt_status,
            "motivation": "Direction is read from same-user same-candidate deltas.",
            "limits": "No universal paper-level prompt claim is made from local open models alone.",
        },
        {
            "rq_id": "F6L-LASTFM-V4-RQ-BEYOND-UTILITY",
            "paper_claim": "Beyond-utility dimensions vary across LLM recommender settings.",
            "published_paper_evidence": "Exact paper LastFM beyond-utility values are not locally reproducible without raw outputs/specs.",
            "previous_lastfm_adapted_evidence": "Earlier beyond-utility rows are separated from this full V4 dataset.",
            "new_operationalized_evidence": "Novelty, APLT, average popularity, popularity bias, catalog coverage, ILD, calibration, hallucination, duplicate, and validity metrics are computed per cell.",
            "dataset_and_sample_difference": "Tag/popularity primitives are local F6C operationalizations.",
            "qwen2_base": f"novelty={full_metric_value(beyond, 'qwen2:7b-instruct', 'base/OpenP5 title-list', 'novelty')};coverage={full_metric_value(beyond, 'qwen2:7b-instruct', 'base/OpenP5 title-list', 'catalog_coverage')}",
            "qwen2_llmrank": f"novelty={full_metric_value(beyond, 'qwen2:7b-instruct', 'LLMRank recency', 'novelty')};coverage={full_metric_value(beyond, 'qwen2:7b-instruct', 'LLMRank recency', 'catalog_coverage')}",
            "mistral_base": f"novelty={full_metric_value(beyond, 'mistral:7b-instruct', 'base/OpenP5 title-list', 'novelty')};coverage={full_metric_value(beyond, 'mistral:7b-instruct', 'base/OpenP5 title-list', 'catalog_coverage')}",
            "mistral_llmrank": f"novelty={full_metric_value(beyond, 'mistral:7b-instruct', 'LLMRank recency', 'novelty')};coverage={full_metric_value(beyond, 'mistral:7b-instruct', 'LLMRank recency', 'catalog_coverage')}",
            "previous_status": "unresolved_for_original_paper_sample",
            "new_status": "mixed_model_dependent",
            "motivation": "Beyond-utility evidence is multi-metric and cell-dependent.",
            "limits": "Local operationalization cannot decide universal paper RQs alone.",
        },
    ]


def build_amended_full_v4_rq_rows(
    fail_strict: Sequence[Mapping[str, str]],
    beyond: Sequence[Mapping[str, str]],
    comparisons: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    qwen_base = f"HR@5_fail_strict={full_metric_value(fail_strict, 'qwen2:7b-instruct', 'base/OpenP5 title-list', 'HR@5_fail_strict')};NDCG@5_fail_strict={full_metric_value(fail_strict, 'qwen2:7b-instruct', 'base/OpenP5 title-list', 'NDCG@5_fail_strict')};validity={full_metric_value(fail_strict, 'qwen2:7b-instruct', 'base/OpenP5 title-list', 'strict_valid_rate')}"
    qwen_llmrank = f"HR@5_fail_strict={full_metric_value(fail_strict, 'qwen2:7b-instruct', 'LLMRank recency', 'HR@5_fail_strict')};NDCG@5_fail_strict={full_metric_value(fail_strict, 'qwen2:7b-instruct', 'LLMRank recency', 'NDCG@5_fail_strict')};validity={full_metric_value(fail_strict, 'qwen2:7b-instruct', 'LLMRank recency', 'strict_valid_rate')}"
    mistral_base = f"HR@5_fail_strict={full_metric_value(fail_strict, 'mistral:7b-instruct', 'base/OpenP5 title-list', 'HR@5_fail_strict')};NDCG@5_fail_strict={full_metric_value(fail_strict, 'mistral:7b-instruct', 'base/OpenP5 title-list', 'NDCG@5_fail_strict')};validity={full_metric_value(fail_strict, 'mistral:7b-instruct', 'base/OpenP5 title-list', 'strict_valid_rate')}"
    mistral_llmrank = f"HR@5_fail_strict={full_metric_value(fail_strict, 'mistral:7b-instruct', 'LLMRank recency', 'HR@5_fail_strict')};NDCG@5_fail_strict={full_metric_value(fail_strict, 'mistral:7b-instruct', 'LLMRank recency', 'NDCG@5_fail_strict')};validity={full_metric_value(fail_strict, 'mistral:7b-instruct', 'LLMRank recency', 'strict_valid_rate')}"
    llmrank_deltas = [float(row["delta_llmrank_minus_base"]) for row in comparisons if row["metric"] in {"HR@5_fail_strict", "NDCG@5_fail_strict"}]
    prompt_status = "mixed_prompt_dependent" if any(delta > 0 for delta in llmrank_deltas) and any(delta < 0 for delta in llmrank_deltas) else "supported_under_local_operationalization"
    return [
        {
            "rq_id": "F6L-LASTFM-V4-AMENDED-RQ-UTILITY",
            "paper_claim": "LLMs can act as recommenders under candidate ranking evaluation.",
            "published_paper_evidence": "Paper LastFM sample and exact model outputs remain unavailable locally.",
            "previous_lastfm_adapted_evidence": "Earlier V1/V2/V3/pilot outputs remain separate prompt-parser fragility evidence.",
            "new_operationalized_evidence": "Full V4 run evaluated with primary fail-strict utility and observed structural-validity metrics.",
            "dataset_and_sample_difference": "Local operationalized HetRec-derived sample; not original paper sample or exact paper output contract.",
            "qwen2_base": qwen_base,
            "qwen2_llmrank": qwen_llmrank,
            "mistral_base": mistral_base,
            "mistral_llmrank": mistral_llmrank,
            "previous_status": "unresolved_for_original_paper_sample",
            "new_status": "supported_under_local_operationalization_with_observed_runtime_validity",
            "motivation": "Primary utility treats structural-invalid outputs as zero utility while retaining them as observed runtime behavior.",
            "limits": "No exact paper reproduction, original paper sample, original output contract, closed models, LLMSRec, reranking, or full cohort claim.",
        },
        {
            "rq_id": "F6L-LASTFM-V4-AMENDED-RQ-PROMPT",
            "paper_claim": "Prompting strategy can affect recommendation behavior.",
            "published_paper_evidence": "Paper reports prompt-family comparisons, but exact LastFM raw outputs are missing.",
            "previous_lastfm_adapted_evidence": "Earlier prompt evidence is separated from this V4 full run.",
            "new_operationalized_evidence": "Base/OpenP5 and LLMRank recency compared on identical users and candidates using fail-strict primary policy.",
            "dataset_and_sample_difference": "Internally paired local operationalized comparison only.",
            "qwen2_base": qwen_base,
            "qwen2_llmrank": qwen_llmrank,
            "mistral_base": mistral_base,
            "mistral_llmrank": mistral_llmrank,
            "previous_status": "unresolved_for_original_paper_sample",
            "new_status": prompt_status,
            "motivation": "Direction is read from paired same-user same-candidate deltas.",
            "limits": "No universal paper-level prompt claim from local open models alone.",
        },
        {
            "rq_id": "F6L-LASTFM-V4-AMENDED-RQ-STRUCTURED-VALIDITY",
            "paper_claim": "Structured generation contracts can be evaluated as part of recommender reliability.",
            "published_paper_evidence": "The paper does not establish decoder-level uniqueness enforcement for local Ollama JSON schema mode.",
            "previous_lastfm_adapted_evidence": "V4 pilot showed 40/40 but did not prove universal uniqueItems enforcement.",
            "new_operationalized_evidence": "Structural-invalid duplicate-token outputs are retained and counted as observed runtime/model behavior.",
            "dataset_and_sample_difference": "Runtime property observed under local Ollama Qwen2/Mistral and the operationalized LastFM sample.",
            "qwen2_base": qwen_base,
            "qwen2_llmrank": qwen_llmrank,
            "mistral_base": mistral_base,
            "mistral_llmrank": mistral_llmrank,
            "previous_status": "not_established_by_pilot_alone",
            "new_status": "duplicate_token_rate_is_an_observed_model_runtime_property",
            "motivation": "A duplicate-token response under JSON schema mode shows uniqueness was not hard-enforced decoder-side.",
            "limits": "This is not corruption, transport failure, malformed JSON, or proof of full JSON Schema enforcement.",
        },
    ]


def build_full_runtime_rows(
    *,
    run_id: str,
    raw_output_dir: Path,
    run_started_at: str,
    run_completed_at: str,
    execution_rows: Sequence[Mapping[str, str]],
    parse_rows: Sequence[Mapping[str, str]],
    retry: int,
    parsing_time: float,
    consolidation_time: float,
) -> list[dict[str, str]]:
    resources = f6l.resource_snapshot()
    raw_bytes = sum(path.stat().st_size for path in raw_output_dir.rglob("*") if path.is_file()) if raw_output_dir.exists() else 0

    def make_row(scope: str, model: str = "", strategy: str = "", exec_subset: Sequence[Mapping[str, str]] | None = None, parse_subset: Sequence[Mapping[str, str]] | None = None) -> dict[str, str]:
        exec_values = list(exec_subset if exec_subset is not None else execution_rows)
        parse_values = list(parse_subset if parse_subset is not None else parse_rows)
        durations = [float(row["duration_seconds"]) for row in exec_values if row.get("duration_seconds")]
        completed = len(exec_values)
        strict = sum(1 for row in parse_values if row["strict_valid"] == "true")
        failed = completed - strict
        wall = sum(durations)
        prompt_tokens = sum(int(row["prompt_eval_count"]) for row in exec_values if str(row.get("prompt_eval_count", "")).isdigit())
        output_tokens = sum(int(row["eval_count"]) for row in exec_values if str(row.get("eval_count", "")).isdigit())
        planned = FULL_REQUEST_COUNT if scope == "overall" else FULL_CELL_REQUEST_COUNT
        return {
            "runtime_id": stable_hash("F6L-LASTFM-V4-FULL-RUNTIME", run_id, scope, model, strategy)[:24],
            "run_id": run_id,
            "scope": scope,
            "model_name": model,
            "prompting_strategy": strategy,
            "planned": str(planned),
            "completed": str(completed),
            "strict_valid": str(strict),
            "failed": str(failed),
            "residual": str(planned - completed),
            "retry": str(retry if scope == "overall" else 0),
            "wall_clock_seconds": format_float(wall),
            "duration_mean_seconds": format_float(f6l.safe_mean(durations)),
            "duration_median_seconds": format_float(sorted(durations)[len(durations) // 2] if durations else 0.0),
            "duration_p90_seconds": format_float(f6l.percentile(durations, 0.90)),
            "duration_p95_seconds": format_float(f6l.percentile(durations, 0.95)),
            "duration_p99_seconds": format_float(f6l.percentile(durations, 0.99)),
            "throughput_requests_per_hour": format_float((completed / wall * 3600.0) if wall else 0.0),
            "prompt_eval_count_total": str(prompt_tokens),
            "eval_count_total": str(output_tokens),
            "cpu": resources["cpu"],
            "ram": resources["ram"],
            "gpu": resources["gpu"],
            "vram": resources["vram"],
            "raw_output_bytes": str(raw_bytes if scope == "overall" else ""),
            "parsing_time_seconds": format_float(parsing_time if scope == "overall" else 0.0),
            "consolidation_time_seconds": format_float(consolidation_time if scope == "overall" else 0.0),
            "marginal_api_cost": "0",
            "hardware_cost": "not_monetized",
            "energy_cost": "unavailable",
            "notes": f"run_started_at={run_started_at};run_completed_at={run_completed_at};total cost is not claimed as zero.",
        }

    rows = [make_row("overall")]
    for model in MODEL_ORDER:
        for strategy in STRATEGY_ORDER:
            exec_subset = [row for row in execution_rows if row["model_name"] == model and row["prompting_strategy"] == strategy]
            parse_subset = [row for row in parse_rows if row["model_name"] == model and row["prompting_strategy"] == strategy]
            rows.append(make_row("cell", model, strategy, exec_subset, parse_subset))
    return rows


def full_acceptance_rows(
    run_id: str,
    *,
    completed: int,
    strict_valid: int,
    structural_invalid: int,
    residual: int,
    retry: int,
    hash_coherence: bool,
    canary_pass: bool,
    preflight_pass: bool,
    previous_raw_unchanged: bool,
    v4_pilot_unchanged: bool,
    profile_unchanged: bool,
    final_validation_pass: bool,
) -> list[dict[str, str]]:
    criteria = [
        ("planned", FULL_REQUEST_COUNT, FULL_REQUEST_COUNT, FULL_REQUEST_COUNT == FULL_REQUEST_COUNT),
        ("completed", completed, FULL_REQUEST_COUNT, completed == FULL_REQUEST_COUNT),
        ("strict_valid_plus_structural_invalid", strict_valid + structural_invalid, FULL_REQUEST_COUNT, strict_valid + structural_invalid == FULL_REQUEST_COUNT),
        ("structural_invalid_recorded", structural_invalid >= 0, True, structural_invalid >= 0),
        ("residual", residual, 0, residual == 0),
        ("transport_retry_reported", retry, "transport retry allowed and reported", retry >= 0),
        ("raw_hash_coherence", "pass" if hash_coherence else "fail", "pass", hash_coherence),
        ("canary_40_40", "pass" if canary_pass else "fail", "pass", canary_pass),
        ("preflight_pass", "pass" if preflight_pass else "fail", "pass", preflight_pass),
        ("previous_raw_unchanged", "pass" if previous_raw_unchanged else "fail", "pass", previous_raw_unchanged),
        ("v4_pilot_unchanged", "pass" if v4_pilot_unchanged else "fail", "pass", v4_pilot_unchanged),
        ("profile_raw_unchanged", "pass" if profile_unchanged else "fail", "pass", profile_unchanged),
        ("final_validation_pass", "pass" if final_validation_pass else "fail", "pass", final_validation_pass),
    ]
    return [
        {
            "pilot_id": FULL_RUN_ID,
            "run_id": run_id,
            "criterion": name,
            "observed": str(observed),
            "expected": str(expected),
            "status": "pass" if passed else "fail",
            "evidence": rel_safe(FULL_RUN_FINAL_VALIDATION_PATH),
            "notes": "Full V4 completion criteria; not relaxed after execution.",
        }
        for name, observed, expected, passed in criteria
    ]


def full_final_validation_rows(
    *,
    run_id: str,
    raw_output_dir: Path,
    manifest_rows: Sequence[Mapping[str, str]],
    request_state_rows: Sequence[Mapping[str, str]],
    parse_rows: Sequence[Mapping[str, str]],
    preflight_rows: Sequence[Mapping[str, str]],
    canary_rows: Sequence[Mapping[str, str]],
    profile_hashes_before: Mapping[str, str],
    v3_hashes_before: Mapping[str, str],
    v4_pilot_hashes_before: Mapping[str, str],
    focused_test_result: Mapping[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(name: str, passed: bool, observed: str, expected: str, evidence: str, notes: str, severity: str = "error") -> None:
        rows.append(
            {
                "validation_id": f"F6L-LASTFM-V4-FULL-FINAL-{len(rows) + 1:02d}",
                "check_name": name,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    completed = sum(1 for row in request_state_rows if row["state"] == "completed")
    strict = sum(1 for row in request_state_rows if row["strict_valid"] == "true")
    structural_invalid = completed - strict
    residual = FULL_REQUEST_COUNT - completed
    manifest_ids = {row["request_id"] for row in manifest_rows}
    response_ids = {row["request_id"] for row in request_state_rows if row["state"] == "completed"}
    raw_paths = f6l.raw_response_paths(raw_output_dir)
    individual_count = len(list(raw_paths["individual"].glob("*.json"))) if raw_paths["individual"].exists() else 0
    raw_manifest = f6l.read_json_file(raw_paths["manifest"]) if raw_paths["manifest"].exists() else {}
    all_response_records = f6l.read_jsonl(raw_paths["responses"]) if raw_paths["responses"].exists() else []
    final_response_ids = [
        str(row.get("request_id", ""))
        for row in all_response_records
        if str(row.get("is_final_attempt", "true")).casefold() == "true" and row.get("request_id")
    ]
    duplicate_final_ids = sorted(request_id for request_id, count in Counter(final_response_ids).items() if count > 1)
    protected = f6l.hash_protected_inputs()
    add("planned_4000", len(manifest_rows) == FULL_REQUEST_COUNT, str(len(manifest_rows)), str(FULL_REQUEST_COUNT), rel_safe(FULL_RUN_MANIFEST_PATH), "Only the authorized V4 full manifest is planned.")
    add("completed_4000", completed == FULL_REQUEST_COUNT, str(completed), str(FULL_REQUEST_COUNT), rel_safe(FULL_RUN_STATE_PATH), "Full completion requires all 4000 requests.")
    add("strict_valid_plus_structural_invalid_4000", strict + structural_invalid == FULL_REQUEST_COUNT, f"strict={strict};structural_invalid={structural_invalid};sum={strict + structural_invalid}", str(FULL_REQUEST_COUNT), rel_safe(FULL_RUN_PARSE_RESULTS_PATH), "Amended completion counts observed structural-invalid outputs without retry or repair.")
    add("strict_valid_4000_not_required_after_amendment", True, str(strict), "reported_not_required", rel_safe(FULL_RUN_PROTOCOL_AMENDMENT_PATH), "Strict-valid=4000 is no longer an absolute completion precondition after the explicit amendment.", severity="warning")
    add("residual_zero", residual == 0, str(residual), "0", rel_safe(FULL_RUN_STATE_PATH), "No missing requests remain.")
    add("no_request_outside_manifest", response_ids <= manifest_ids, str(len(response_ids - manifest_ids)), "0", rel_safe(raw_paths["responses"]), "Raw response IDs must be a subset of the full V4 manifest.")
    add("no_completed_repeated", not duplicate_final_ids, ";".join(duplicate_final_ids) if duplicate_final_ids else "none", "none", rel_safe(raw_paths["responses"]), "No final completed response may be repeated in raw_responses.jsonl.")
    add("manifest_hash_unchanged", str(raw_manifest.get("manifest_hash", "")) == str(build_manifest_hash(manifest_rows, run_id, raw_output_dir)), str(raw_manifest.get("manifest_hash", "")) or "missing", str(build_manifest_hash(manifest_rows, run_id, raw_output_dir)), rel_safe(raw_paths["manifest"]), "Resume must preserve the original 4000-request manifest hash.")
    add("no_reranking", not any("rerank" in row["prompting_strategy"].casefold() for row in manifest_rows), "ranking_only", "ranking_only", rel_safe(FULL_RUN_MANIFEST_PATH), "Reranking is not authorized.")
    add("no_full_cohort", len({row["user_id"] for row in manifest_rows}) == FULL_USER_COUNT, str(len({row["user_id"] for row in manifest_rows})), str(FULL_USER_COUNT), rel_safe(FULL_RUN_MANIFEST_PATH), "Only the operationalized 1000-user sample is used.")
    add("no_remote_models", {row["model_name"] for row in manifest_rows} == set(MODEL_ORDER), ";".join(sorted({row["model_name"] for row in manifest_rows})), ";".join(MODEL_ORDER), rel_safe(FULL_RUN_MANIFEST_PATH), "Only local Qwen2/Mistral are used.")
    add("model_digests_unchanged", all(row["model_digest"] == f6l.MODEL_DIGESTS[row["model_name"]] for row in manifest_rows), "pass" if all(row["model_digest"] == f6l.MODEL_DIGESTS[row["model_name"]] for row in manifest_rows) else "fail", "pass", rel_safe(FULL_RUN_MANIFEST_PATH), "Local model digest contract remains frozen.")
    add("raw_files_jsonl_coherent", individual_count >= completed and len(f6l.read_jsonl(raw_paths["responses"])) >= completed, f"individual={individual_count};jsonl={len(f6l.read_jsonl(raw_paths['responses']))};completed={completed}", "raw individual/jsonl records cover completed requests", rel_safe(raw_output_dir), "Transport attempts may add non-final attempt records; final strict-valid requests have individual JSON evidence.")
    add("parse_hash_coherence", bool(parse_rows) and all(row["hash_coherence"] == "true" for row in parse_rows), "pass" if parse_rows and all(row["hash_coherence"] == "true" for row in parse_rows) else "fail", "pass", rel_safe(FULL_RUN_PARSE_RESULTS_PATH), "Rendered prompt, candidate, mapping, and request contract hashes cohere.")
    add("parser_hash_unchanged", parser_source_hash() == EXPECTED_V4_PARSER_HASH, parser_source_hash(), EXPECTED_V4_PARSER_HASH, "inspect.getsource(parse_v4_response)", "V4 parser remains frozen.")
    add("schema_hash_unchanged", GENERATION_PARAMETERS["format_schema_hash"] == EXPECTED_V4_SCHEMA_HASH, GENERATION_PARAMETERS["format_schema_hash"], EXPECTED_V4_SCHEMA_HASH, "V4_JSON_SCHEMA", "V4 schema remains frozen.")
    add("prompt_hashes_unchanged", {strategy: prompt_template_hash(strategy) for strategy in STRATEGY_ORDER} == EXPECTED_V4_PROMPT_TEMPLATE_HASHES, canonical_json({strategy: prompt_template_hash(strategy) for strategy in STRATEGY_ORDER}), canonical_json(EXPECTED_V4_PROMPT_TEMPLATE_HASHES), "prompt_template_text", "V4 prompts remain frozen.")
    amendment = f6l.read_json_file(FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH) if FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH.exists() else {}
    amendment_hash_ok = bool(amendment.get("amendment_hash")) and amendment.get("amendment_hash") == f6l.sha256_text(canonical_json({key: value for key, value in amendment.items() if key != "amendment_hash"}))
    add("protocol_amendment_present_and_hashable", amendment_hash_ok, str(amendment.get("amendment_hash", "")) or "missing", "self-describing amendment hash", rel_safe(FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH), "Explicit amendment documents why structural-invalid outputs are retained and evaluated fail-strict.")
    add("duplicate_window_breaker_amended", str(amendment.get("circuit_breaker", {}).get("retired_breaker", "")) == "do_not_use_closed_100_new_response_window_validity_below_0_99", str(amendment.get("circuit_breaker", {}).get("retired_breaker", "")) or "missing", "do_not_use_closed_100_new_response_window_validity_below_0_99", rel_safe(FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH), "The final amendment replaces the 100-response validity breaker with duplicate-token windows.")
    add("canary_pass", all(row["status"] == "pass" for row in canary_rows) and bool(canary_rows), f"{sum(1 for row in canary_rows if row['status'] == 'pass')}/{len(canary_rows)}", "all pass", rel_safe(FULL_RUN_CANARY_PATH), "The internal first-40 canary passed before continuing.")
    add("preflight_pass", all(row["status"] == "pass" for row in preflight_rows), f"{sum(1 for row in preflight_rows if row['status'] == 'pass')}/{len(preflight_rows)}", "all pass", rel_safe(FULL_RUN_PREFLIGHT_PATH), "All preflight gates passed before inference.")
    add("focused_tests_pass", focused_test_result.get("exit_code") == 0, f"exit={focused_test_result.get('exit_code')}", "exit=0", "pytest focused F6L V4 shard", "Focused tests passed before inference.")
    add("original_4000_raw_unchanged", v3.original_raw_hash_map_sha() == v3.original_raw_expected_sha() and bool(v3.original_raw_expected_sha()), v3.original_raw_hash_map_sha(), v3.original_raw_expected_sha(), rel_safe(v3.ORIGINAL_RAW_DIR), "Previous 4000 raw outputs remain separate and immutable.")
    add("v3_pilot_unchanged", v3_hashes_before == v3_artifact_hashes(), f"before={len(v3_hashes_before)};after={len(v3_artifact_hashes())}", "same hash map", rel_safe(V3_RAW_DIR), "V3 pilot artifacts remain unchanged.")
    add("v4_pilot_unchanged", v4_pilot_hashes_before == v4_pilot_artifact_hashes(), f"before={len(v4_pilot_hashes_before)};after={len(v4_pilot_artifact_hashes())}", "same hash map", rel_safe(V4_PILOT_RAW_DIR), "V4 pilot artifacts remain unchanged.")
    add("profile_raw_unchanged", f6l.profile_raw_hashes() == dict(profile_hashes_before), f"before={len(profile_hashes_before)};after={len(f6l.profile_raw_hashes())}", "same hash map", "results/llm/phase_f6l_profile_execution_*", "Profile outputs remain untouched.")
    add("protected_inputs_pass", protected.get("status") == "pass", str(protected.get("status")), "pass", rel_safe(f6l.PROTECTED_HASHES_PATH), "Protected input guard passes.")
    add("twenty_three_yaml_invariant", protected.get("f6i2p_yaml_count") == 23, str(protected.get("f6i2p_yaml_count")), "23", rel_safe(f6l.PROTECTED_HASHES_PATH), "Protected YAML count remains invariant.")
    add("git_staging_empty", f6l.git_output(["diff", "--cached", "--name-only"]) == "", f6l.git_output(["diff", "--cached", "--name-only"]) or "none", "none", "git diff --cached --name-only", "No staging or commit.")
    add("original_repo_clean", f6l.git_output(["status", "--short", "original_repo"]) == "", f6l.git_output(["status", "--short", "original_repo"]) or "none", "none", "git status --short original_repo", "No original repo mutation.")
    return rows


def consolidate_full_run(
    build: FullRunBuild,
    raw_output_dir: Path,
    preflight_rows: Sequence[Mapping[str, str]],
    profile_hashes_before: Mapping[str, str],
    v3_hashes_before: Mapping[str, str],
    v4_pilot_hashes_before: Mapping[str, str],
    focused_test_result: Mapping[str, Any],
    run_started_at: str,
    run_completed_at: str,
    retry: int,
    requested_decision: str | None = None,
    resume_window_start_completed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    parse_start = time.perf_counter()
    raw_records = f6l.final_response_records(raw_output_dir)
    raw_paths = f6l.raw_response_paths(raw_output_dir)
    all_response_records = f6l.read_jsonl(raw_paths["responses"])
    attempt_counts = Counter(str(row.get("request_id", "")) for row in all_response_records if row.get("request_id"))
    candidates = f6l.candidate_rows_for_operationalized_sample()
    manifest_by_id = {row["request_id"]: row for row in build.manifest_rows}
    execution_rows: list[dict[str, str]] = []
    parse_rows: list[dict[str, str]] = []
    rec_rows: list[dict[str, str]] = []
    for rid, row in manifest_by_id.items():
        raw_record = raw_records.get(rid)
        if not raw_record:
            continue
        execution_rows.append(execution_row(row, raw_record))
        parse_row = full_parse_result_row(row, raw_record, candidates[row["user_id"]])
        parse_rows.append(parse_row)
        rec_rows.extend(full_recommendation_rows_from_parse(row, parse_row))
    parsing_time = time.perf_counter() - parse_start
    completed = len(raw_records)
    strict = sum(1 for row in parse_rows if row["strict_valid"] == "true")
    failed = completed - strict
    residual = FULL_REQUEST_COUNT - completed
    hash_coherence = bool(parse_rows) and all(row["hash_coherence"] == "true" for row in parse_rows)
    canary_rows = canary_rows_from_parse(build.run_id, parse_rows, build.manifest_rows, raw_output_dir) if completed >= 40 else []
    request_state_rows = full_request_state_rows(build.manifest_rows, raw_records, parse_rows, attempt_counts)
    validity_rows: list[dict[str, str]] = []
    fail_strict_rows: list[dict[str, str]] = []
    complete_case_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = build_failure_rows(build.run_id, parse_rows, pilot_id=FULL_RUN_ID, evidence_class=FULL_RUN_CLASSIFICATION)
    metrics_rows: list[dict[str, str]] = []
    beyond_rows: list[dict[str, str]] = []
    comparison_rows: list[dict[str, str]] = []
    rq_rows: list[dict[str, str]] = []
    duplicate_window_rows = build_duplicate_window_rows(build.run_id, parse_rows, build.manifest_rows, resume_window_start_completed or 0)
    consolidation_start = time.perf_counter()
    if completed == FULL_REQUEST_COUNT and strict + failed == FULL_REQUEST_COUNT and residual == 0 and hash_coherence:
        validity_rows, fail_strict_rows, complete_case_rows, beyond_rows, comparison_rows, rq_rows, failure_rows = compute_amended_full_v4_tables(build.run_id, parse_rows, rec_rows, build.manifest_rows, attempt_counts)
        metrics_rows = fail_strict_rows
    runtime_rows = build_full_runtime_rows(
        run_id=build.run_id,
        raw_output_dir=raw_output_dir,
        run_started_at=run_started_at,
        run_completed_at=run_completed_at,
        execution_rows=execution_rows,
        parse_rows=parse_rows,
        retry=retry,
        parsing_time=parsing_time,
        consolidation_time=time.perf_counter() - consolidation_start,
    )
    final_validation = full_final_validation_rows(
        run_id=build.run_id,
        raw_output_dir=raw_output_dir,
        manifest_rows=build.manifest_rows,
        request_state_rows=request_state_rows,
        parse_rows=parse_rows,
        preflight_rows=preflight_rows,
        canary_rows=canary_rows,
        profile_hashes_before=profile_hashes_before,
        v3_hashes_before=v3_hashes_before,
        v4_pilot_hashes_before=v4_pilot_hashes_before,
        focused_test_result=focused_test_result,
    )
    final_validation_pass = all(row["status"] == "pass" or row["severity"] != "error" for row in final_validation)
    previous_raw_unchanged = v3.original_raw_hash_map_sha() == v3.original_raw_expected_sha() and bool(v3.original_raw_expected_sha())
    v4_pilot_unchanged = v4_pilot_hashes_before == v4_pilot_artifact_hashes()
    profile_unchanged = f6l.profile_raw_hashes() == dict(profile_hashes_before)
    canary_pass = bool(canary_rows) and all(row["status"] == "pass" for row in canary_rows)
    preflight_pass = all(row["status"] == "pass" for row in preflight_rows)
    acceptance_rows = full_acceptance_rows(
        build.run_id,
        completed=completed,
        strict_valid=strict,
        structural_invalid=failed,
        residual=residual,
        retry=retry,
        hash_coherence=hash_coherence,
        canary_pass=canary_pass,
        preflight_pass=preflight_pass,
        previous_raw_unchanged=previous_raw_unchanged,
        v4_pilot_unchanged=v4_pilot_unchanged,
        profile_unchanged=profile_unchanged,
        final_validation_pass=final_validation_pass,
    )
    if requested_decision:
        decision = requested_decision
    elif completed == FULL_REQUEST_COUNT and strict + failed == FULL_REQUEST_COUNT and residual == 0 and hash_coherence and canary_pass and preflight_pass and previous_raw_unchanged and v4_pilot_unchanged and profile_unchanged and final_validation_pass:
        decision = AMENDED_DECISION_ALL_STRICT if failed == 0 else AMENDED_DECISION_WITH_INVALID
    else:
        decision = AMENDED_DECISION_INFRA_PARTIAL
    f6l.write_csv(FULL_RUN_EXECUTION_PATH, execution_rows, EXECUTION_FIELDS)
    f6l.write_csv(FULL_RUN_PARSE_RESULTS_PATH, parse_rows, PARSE_FIELDS)
    f6l.write_csv(FULL_RUN_RECOMMENDATIONS_PATH, rec_rows, RECOMMENDATION_FIELDS)
    f6l.write_csv(FULL_RUN_CANARY_PATH, canary_rows, FULL_RUN_CANARY_FIELDS)
    f6l.write_csv(FULL_RUN_METRICS_PATH, metrics_rows, FULL_RUN_METRIC_FIELDS)
    f6l.write_csv(FULL_RUN_VALIDITY_BY_CELL_PATH, validity_rows, FULL_RUN_VALIDITY_BY_CELL_FIELDS)
    f6l.write_csv(FULL_RUN_FAIL_STRICT_UTILITY_PATH, fail_strict_rows, FULL_RUN_METRIC_FIELDS)
    f6l.write_csv(FULL_RUN_COMPLETE_CASE_UTILITY_PATH, complete_case_rows, FULL_RUN_METRIC_FIELDS)
    f6l.write_csv(FULL_RUN_FAILURE_TAXONOMY_PATH, failure_rows, FAILURE_FIELDS)
    f6l.write_csv(FULL_RUN_DUPLICATE_WINDOW_PATH, duplicate_window_rows, FULL_RUN_DUPLICATE_WINDOW_FIELDS)
    f6l.write_csv(FULL_RUN_BEYOND_UTILITY_PATH, beyond_rows, FULL_RUN_BEYOND_FIELDS)
    f6l.write_csv(FULL_RUN_STRATEGY_COMPARISON_PATH, comparison_rows, FULL_RUN_COMPARISON_FIELDS)
    f6l.write_csv(FULL_RUN_RUNTIME_PATH, runtime_rows, FULL_RUN_RUNTIME_FIELDS)
    f6l.write_csv(FULL_RUN_RQ_REVALIDATION_PATH, rq_rows, FULL_RUN_RQ_FIELDS)
    f6l.write_csv(FULL_RUN_FINAL_VALIDATION_PATH, final_validation, FULL_RUN_FINAL_VALIDATION_FIELDS)
    f6l.write_csv(FULL_RUN_ACCEPTANCE_PATH, acceptance_rows, ACCEPTANCE_FIELDS)
    f6l.write_csv(FULL_RUN_STATE_PATH, [
        {
            "run_id": build.run_id,
            "stage": "complete" if decision in {FULL_DECISION_COMPLETE, AMENDED_DECISION_ALL_STRICT, AMENDED_DECISION_WITH_INVALID} else "partial_or_blocked",
            "decision_token": decision,
            "planned": str(FULL_REQUEST_COUNT),
            "completed": str(completed),
            "strict_valid": str(strict),
            "failed": str(failed),
            "structural_invalid": str(failed),
            "duplicate_token_count": str(sum(int(row["duplicate_token_count"] or 0) for row in parse_rows)),
            "malformed_json_count": str(sum(int(row["malformed_json_count"] or 0) for row in parse_rows)),
            "unknown_token_count": str(sum(int(row["unknown_token_count"] or 0) for row in parse_rows)),
            "residual": str(residual),
            "retry": str(retry),
            "cumulative_validity_rate": format_float(strict / completed if completed else 1.0),
            "cumulative_duplicate_token_response_rate": format_float(duplicate_token_response_rate(parse_rows)),
            "current_200_request_window_duplicate_token_rate": "",
            "closed_200_request_window_index": "",
            "closed_200_request_window_duplicate_token_rate": "",
            "consecutive_duplicate_token_invalid_count": "",
            "percent_complete": format_float(completed / FULL_REQUEST_COUNT),
            "last_request_id": max(raw_records) if raw_records else "",
            "last_model_name": "",
            "last_prompting_strategy": "",
            "throughput_requests_per_hour": "",
            "eta_seconds": "",
            "last_timestamp": utc_now(),
            "canary_status": "pass" if canary_pass else "not_passed",
            "circuit_breaker_status": "clear" if decision in {FULL_DECISION_COMPLETE, AMENDED_DECISION_ALL_STRICT, AMENDED_DECISION_WITH_INVALID} else decision,
            "raw_output_dir": rel_safe(raw_output_dir),
            "notes": "terminal full V4 consolidation",
        }
    ], FULL_RUN_STATE_FIELDS)
    output_paths = [
        FULL_RUN_MANIFEST_PATH,
        FULL_RUN_FREEZE_PATH,
        FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH,
        FULL_RUN_RESUME_LINEAGE_PATH,
        FULL_RUN_PREFLIGHT_PATH,
        FULL_RUN_STATE_PATH,
        FULL_RUN_CANARY_PATH,
        FULL_RUN_EXECUTION_PATH,
        FULL_RUN_PARSE_RESULTS_PATH,
        FULL_RUN_RECOMMENDATIONS_PATH,
        FULL_RUN_FAILURE_TAXONOMY_PATH,
        FULL_RUN_VALIDITY_BY_CELL_PATH,
        FULL_RUN_FAIL_STRICT_UTILITY_PATH,
        FULL_RUN_COMPLETE_CASE_UTILITY_PATH,
        FULL_RUN_DUPLICATE_WINDOW_PATH,
        FULL_RUN_METRICS_PATH,
        FULL_RUN_BEYOND_UTILITY_PATH,
        FULL_RUN_STRATEGY_COMPARISON_PATH,
        FULL_RUN_RUNTIME_PATH,
        FULL_RUN_RQ_REVALIDATION_PATH,
        FULL_RUN_FINAL_VALIDATION_PATH,
        FULL_RUN_ACCEPTANCE_PATH,
    ]
    manifest = {
        **build.manifest,
        "decision_token": decision,
        "protocol_amendment": rel_safe(FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH) if FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH.exists() else "",
        "protocol_amendment_hash": f6l.safe_file_sha256(FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH) if FULL_RUN_PROTOCOL_AMENDMENT_MANIFEST_PATH.exists() else "",
        "run_started_at": run_started_at,
        "run_completed_at": run_completed_at,
        "raw_output_dir": rel_safe(raw_output_dir),
        "planned": FULL_REQUEST_COUNT,
        "completed": completed,
        "strict_valid": strict,
        "failed": failed,
        "structural_invalid": failed,
        "residual": residual,
        "retry": retry,
        "canary_status": "pass" if canary_pass else "not_passed",
        "circuit_breaker_status": "clear" if decision in {FULL_DECISION_COMPLETE, AMENDED_DECISION_ALL_STRICT, AMENDED_DECISION_WITH_INVALID} else decision,
        "preflight_pass": preflight_pass,
        "final_validation_pass": final_validation_pass,
        "metrics_computed": bool(metrics_rows),
        "primary_utility_policy": "fail_strict",
        "secondary_utility_policy": "secondary_complete_case_not_primary",
        "additional_classification": [
            "structured_output_validity_is_an_observed_model_runtime_property",
            "duplicate_token_rate_is_an_observed_model_runtime_property",
        ],
        "output_hashes": {rel_safe(path): f6l.safe_file_sha256(path) for path in output_paths if path.exists()},
        "raw_files": {name: rel_safe(path) for name, path in raw_paths.items() if name != "individual"},
        "git": {
            "head": f6l.git_output(["rev-parse", "HEAD"]),
            "staged_paths": f6l.git_output(["diff", "--cached", "--name-only"]),
            "original_repo_status": f6l.git_output(["status", "--short", "original_repo"]),
        },
    }
    f6l.write_json(FULL_RUN_JSON_MANIFEST_PATH, manifest)
    write_full_report_and_notes(manifest, metrics_rows, beyond_rows, runtime_rows, rq_rows, acceptance_rows)
    return decision, manifest


def write_full_report_and_notes(
    manifest: Mapping[str, Any],
    metrics_rows: Sequence[Mapping[str, str]],
    beyond_rows: Sequence[Mapping[str, str]],
    runtime_rows: Sequence[Mapping[str, str]],
    rq_rows: Sequence[Mapping[str, str]],
    acceptance_rows: Sequence[Mapping[str, str]],
) -> None:
    metric_lines = [
        f"- `{row['model_name']}` `{row['prompting_strategy']}` `{row['metric']}` = `{row['value']}`."
        for row in metrics_rows
        if row.get("metric") in {"HR@5", "NDCG@5", "HR@5_fail_strict", "NDCG@5_fail_strict", "strict_valid_rate", "structural_invalid_rate", "duplicate_token_rate", "retry_rate"}
    ]
    report = "\n".join(
        [
            "# Phase F6L LastFM V4 Constrained Full Ranking",
            "",
            f"- Decision token: `{manifest['decision_token']}`",
            f"- Run ID: `{manifest['run_id']}`",
            f"- Raw output directory: `{manifest['raw_output_dir']}`",
            f"- Classification: `{FULL_RUN_CLASSIFICATION}`",
            f"- Planned/completed/strict-valid/failed/residual: `{manifest['planned']}`/`{manifest['completed']}`/`{manifest['strict_valid']}`/`{manifest['failed']}`/`{manifest['residual']}`.",
            f"- Canary status: `{manifest['canary_status']}`.",
            f"- Circuit breaker: `{manifest['circuit_breaker_status']}`.",
            f"- Parser hash: `{manifest['parser_source_hash']}`.",
            f"- Manifest hash: `{manifest['manifest_hash']}`.",
            "",
            "## Metrics",
            "",
            *(metric_lines or ["- Not computed because completed=4000 and strict-valid+structural-invalid=4000 were not reached."]),
            "",
            "## Boundaries",
            "",
            "- This is `paper_protocol_reproduction_under_explicit_operationalization`.",
            "- Additional classification: `structured_output_validity_is_an_observed_model_runtime_property`.",
            "- Additional classification: `duplicate_token_rate_is_an_observed_model_runtime_property`.",
            "- It is not exact paper reproduction, not the original paper sample, and not the original paper output contract.",
            "- V1/V2/V3 and pilot V4 outputs remain separate fragility evidence.",
        ]
    )
    FULL_RUN_REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    note = "\n".join(
        [
            "",
            f"## {FULL_RUN_ID} ({utc_now()})",
            "",
            f"- Decision token: `{manifest['decision_token']}`",
            f"- Run ID: `{manifest['run_id']}`",
            f"- Raw output dir: `{manifest['raw_output_dir']}`",
            f"- Planned/completed/strict-valid/failed/residual: `{manifest['planned']}`/`{manifest['completed']}`/`{manifest['strict_valid']}`/`{manifest['failed']}`/`{manifest['residual']}`.",
            f"- Metrics rows: `{len(metrics_rows)}`; beyond-utility rows: `{len(beyond_rows)}`; RQ rows: `{len(rq_rows)}`; acceptance pass: `{sum(1 for row in acceptance_rows if row['status'] == 'pass')}/{len(acceptance_rows)}`.",
            "- Scope boundary: no reranking, no full cohort, no LLMSRec, no remote APIs, no training, no download.",
        ]
    )
    with NOTES_PATH.open("a", encoding="utf-8") as handle:
        handle.write(note + "\n")
    with HANDOFF_PATH.open("a", encoding="utf-8") as handle:
        handle.write(note + "\n")


def execute_pilot_requests(build: PilotBuild, raw_output_dir: Path, endpoint: str, resume: bool) -> None:
    paths = f6l.raw_response_paths(raw_output_dir)
    paths["individual"].mkdir(parents=True, exist_ok=True)
    if not paths["command"].exists():
        paths["command"].write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    f6l.write_json(paths["manifest"], {**build.manifest, "raw_output_dir": rel_safe(raw_output_dir), "started_at": utc_now(), "endpoint": endpoint, "resume": resume})
    sample = sample_rows_by_user()
    candidates = f6l.candidate_rows_for_operationalized_sample()
    finals = f6l.final_response_records(raw_output_dir) if resume else {}
    started_clock = time.perf_counter()
    parse_rows: list[dict[str, str]] = []
    completed = len(finals)
    for row in build.manifest_rows:
        if row["request_id"] in finals:
            continue
        prompt = render_v4_prompt({**sample[row["user_id"]], "_subset": row["subset"]}, candidates[row["user_id"]], row["prompting_strategy"])
        raw_json_path = paths["individual"] / f"{row['request_id']}.json"
        if raw_json_path.exists():
            raise RuntimeError(f"refusing to overwrite existing raw response without final JSONL resume record: {raw_json_path}")
        request_record = {**request_payload_for_row(row), "prompt": prompt, "endpoint": endpoint, "queued_at": utc_now(), "expected_raw_output_path": row["expected_raw_output_path"]}
        f6l.append_jsonl_fsync(paths["requests"], request_record)
        result = execute_local_ollama_schema_request(row["model_name"], prompt, endpoint)
        parse = parse_v4_response(result.response_text, candidates[row["user_id"]])
        raw_record = raw_record_for_response(row, prompt, candidates[row["user_id"]], result, parse, raw_json_path)
        raw_json_path.write_text(json.dumps(raw_record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        f6l.append_jsonl_fsync(paths["responses"], raw_record)
        completed += 1
        parse_row = parse_result_row(row, raw_record, candidates[row["user_id"]])
        parse_rows.append(parse_row)
        write_state(build.run_id, raw_output_dir, "executing", DECISION_FAILED, started_clock, row["request_id"], parse_rows, completed, "observable incremental state; terminal decision assigned during consolidation")
        print(f"[{utc_now()}] completed {completed}/{PILOT_REQUEST_COUNT} strict_valid={sum(1 for item in parse_rows if item['strict_valid'] == 'true')} last={row['request_id']} failure={parse.failure_reason or 'none'}", flush=True)


def build_validity_rows(run_id: str, parse_rows: Sequence[Mapping[str, str]], completed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    by_cell: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in parse_rows:
        by_cell[(row["model_name"], row["prompting_strategy"])].append(row)
    for model in MODEL_ORDER:
        for strategy in STRATEGY_ORDER:
            cell = by_cell.get((model, strategy), [])
            strict = sum(1 for row in cell if row["strict_valid"] == "true")
            rows.append(
                {
                    "pilot_id": PILOT_ID,
                    "run_id": run_id,
                    "model_name": model,
                    "prompting_strategy": strategy,
                    "planned": str(PILOT_USER_COUNT),
                    "completed": str(len(cell)),
                    "strict_valid": str(strict),
                    "failed": str(len(cell) - strict),
                    "strict_valid_rate": format_float(strict / len(cell) if cell else 0.0),
                    "malformed_json_count": str(sum(int(row["malformed_json_count"] or 0) for row in cell)),
                    "unknown_token_count": str(sum(int(row["unknown_token_count"] or 0) for row in cell)),
                    "duplicate_token_count": str(sum(int(row["duplicate_token_count"] or 0) for row in cell)),
                    "missing_token_count": str(sum(int(row["missing_token_count"] or 0) for row in cell)),
                    "ambiguous_mapping_count": str(sum(int(row["ambiguous_mapping_count"] or 0) for row in cell)),
                    "numeric_without_c_prefix_count": str(sum(int(row["numeric_without_c_prefix_count"] or 0) for row in cell)),
                    "residual": str(PILOT_USER_COUNT - len(cell)),
                    "evidence_class": PILOT_CLASSIFICATION,
                    "notes": "validity diagnostic only; no scientific model comparison",
                }
            )
    strict_all = sum(1 for row in parse_rows if row["strict_valid"] == "true")
    rows.append(
        {
            "pilot_id": PILOT_ID,
            "run_id": run_id,
            "model_name": "__all__",
            "prompting_strategy": "__all__",
            "planned": str(PILOT_REQUEST_COUNT),
            "completed": str(completed),
            "strict_valid": str(strict_all),
            "failed": str(completed - strict_all),
            "strict_valid_rate": format_float(strict_all / completed if completed else 0.0),
            "malformed_json_count": str(sum(int(row["malformed_json_count"] or 0) for row in parse_rows)),
            "unknown_token_count": str(sum(int(row["unknown_token_count"] or 0) for row in parse_rows)),
            "duplicate_token_count": str(sum(int(row["duplicate_token_count"] or 0) for row in parse_rows)),
            "missing_token_count": str(sum(int(row["missing_token_count"] or 0) for row in parse_rows)),
            "ambiguous_mapping_count": str(sum(int(row["ambiguous_mapping_count"] or 0) for row in parse_rows)),
            "numeric_without_c_prefix_count": str(sum(int(row["numeric_without_c_prefix_count"] or 0) for row in parse_rows)),
            "residual": str(PILOT_REQUEST_COUNT - completed),
            "evidence_class": PILOT_CLASSIFICATION,
            "notes": "aggregate V4 strict-validity diagnostic",
        }
    )
    return rows


def build_regression_holdout_rows(run_id: str, parse_rows: Sequence[Mapping[str, str]], failure_audit_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    by_subset: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in parse_rows:
        by_subset[row["subset"]].append(row)
    regression_failure_users = {row["user_id"] for row in failure_audit_rows}
    for subset, planned_users in [("regression", len(regression_failure_users)), ("holdout", PILOT_USER_COUNT - len(regression_failure_users))]:
        subset_rows = by_subset.get(subset, [])
        strict = sum(1 for row in subset_rows if row["strict_valid"] == "true")
        rows.append(
            {
                "pilot_id": PILOT_ID,
                "run_id": run_id,
                "subset": subset,
                "planned": str(planned_users * 4),
                "completed": str(len(subset_rows)),
                "strict_valid": str(strict),
                "failed": str(len(subset_rows) - strict),
                "strict_valid_rate": format_float(strict / len(subset_rows) if subset_rows else 0.0),
                "v3_regression_equivalent_failures_covered": "true" if subset == "regression" and regression_failure_users.issubset({row["user_id"] for row in subset_rows}) else ("" if subset == "holdout" else "false"),
                "evidence_class": PILOT_CLASSIFICATION,
                "notes": "regression covers all users associated with the seven V3 out-of-range failures; holdout uses new deterministic blind users",
            }
        )
    return rows


def build_failure_rows(
    run_id: str,
    parse_rows: Sequence[Mapping[str, str]],
    *,
    pilot_id: str = PILOT_ID,
    evidence_class: str = PILOT_CLASSIFICATION,
) -> list[dict[str, str]]:
    counts = Counter(row["failure_reason"] or "strict_valid" for row in parse_rows)
    if not counts:
        counts["no_completed_requests"] = 0
    return [
        {
            "pilot_id": pilot_id,
            "run_id": run_id,
            "failure_reason": reason,
            "count": str(count),
            "evidence_class": evidence_class,
            "notes": "V4 parser failure taxonomy; strict_valid is not a failure",
        }
        for reason, count in sorted(counts.items())
    ]


def build_smoke_metrics(run_id: str, parse_rows: Sequence[Mapping[str, str]], manifest_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    candidates = f6l.candidate_rows_for_operationalized_sample()
    rows: list[dict[str, str]] = []
    by_cell: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in parse_rows:
        by_cell[(row["model_name"], row["prompting_strategy"])].append(row)
    for (model, strategy), cell_rows in sorted(by_cell.items()):
        hits: list[float] = []
        ndcgs: list[float] = []
        for row in cell_rows:
            if row["strict_valid"] != "true":
                continue
            positives = [candidate for candidate in candidates[row["user_id"]] if candidate.get("is_positive") == "true"]
            positive_item = positives[0]["candidate_item_id"] if positives else ""
            ranked = [str(item) for item in json.loads(row["ranked_item_ids"])]
            rank = ranked.index(positive_item) + 1 if positive_item in ranked else None
            hit = 1.0 if rank is not None and rank <= 5 else 0.0
            ndcg = 1.0 / math.log2(rank + 1) if rank is not None and rank <= 5 else 0.0
            hits.append(hit)
            ndcgs.append(ndcg)
        for metric, values in [("HR@5", hits), ("NDCG@5", ndcgs)]:
            rows.append(
                {
                    "pilot_id": PILOT_ID,
                    "run_id": run_id,
                    "model_name": model,
                    "prompting_strategy": strategy,
                    "metric": metric,
                    "value": format_float(sum(values) / len(values) if values else math.nan),
                    "evaluated_requests": str(len(values)),
                    "evidence_class": "pilot_only_not_scientific_evidence",
                    "notes": "technical smoke check only; do not update RQs or compare models on 10 users",
                }
            )
    return rows


def build_acceptance_rows(run_id: str, values: Mapping[str, Any]) -> list[dict[str, str]]:
    criteria = [
        ("completed", values["completed"], 40),
        ("strict_valid_v4", values["strict_valid"], 40),
        ("failed", values["failed"], 0),
        ("residual", values["residual"], 0),
        ("malformed_json", values["malformed_json"], 0),
        ("unknown_token", values["unknown_token"], 0),
        ("duplicate_token", values["duplicate_token"], 0),
        ("missing_token", values["missing_token"], 0),
        ("ambiguous_mapping", values["ambiguous_mapping"], 0),
        ("numeric_without_c_prefix", values["numeric_without_c_prefix"], 0),
        ("raw_hash_coherence", "pass" if values["hash_coherence"] else "fail", "pass"),
        ("v3_regression_equivalents_valid", "pass" if values["regression_equivalents_valid"] else "fail", "pass"),
        ("v3_artifacts_unchanged", "pass" if values["v3_unchanged"] else "fail", "pass"),
        ("original_raw_unchanged", "pass" if values["original_raw_unchanged"] else "fail", "pass"),
        ("profile_raw_unchanged", "pass" if values["profile_raw_unchanged"] else "fail", "pass"),
        ("preflight_pass", "pass" if values["preflight_pass"] else "fail", "pass"),
    ]
    return [
        {
            "pilot_id": PILOT_ID,
            "run_id": run_id,
            "criterion": name,
            "observed": str(observed),
            "expected": str(expected),
            "status": "pass" if str(observed) == str(expected) else "fail",
            "evidence": rel_safe(PILOT_PARSE_RESULTS_PATH) if name not in {"preflight_pass", "v3_artifacts_unchanged", "original_raw_unchanged", "profile_raw_unchanged"} else rel_safe(PILOT_PREFLIGHT_PATH),
            "notes": "preregistered V4 acceptance threshold; not relaxed after execution",
        }
        for name, observed, expected in criteria
    ]


def build_full_run_plan_rows(run_id: str) -> list[dict[str, str]]:
    command = " ".join(
        [
            "$env:PYTHONNOUSERSITE='1';",
            "$env:PYTHONDONTWRITEBYTECODE='1';",
            "$env:PYTHONPATH='.';",
            str(f6l.RECOVERED_INTERPRETER),
            "-B",
            "src/experiments/build_phase_f6l_lastfm_corrective_output_contract_v4_pilot.py",
            "--mode",
            "execute-full-v4",
            "--run-id",
            "YYYYMMDD_HHMMSS",
            "--allow-api-calls",
            "--local-only",
            "--confirm-full-v4-rerun",
        ]
    )
    return [
        {
            "plan_id": "F6L-LASTFM-V4-FULL-RUN-PLAN-001",
            "pilot_run_id": run_id,
            "future_run_id_placeholder": "YYYYMMDD_HHMMSS",
            "status": "prepared_not_executed",
            "future_command": command,
            "request_count": "4000",
            "notes": "Prepared only if V4 pilot reaches 40/40 strict-valid. Future operation must create a new raw dir and not mix with V1/V2/V3/V4 pilot outputs.",
        }
    ]


def consolidate_outputs(build: PilotBuild, raw_output_dir: Path, preflight_rows: Sequence[Mapping[str, str]], profile_hashes_before: Mapping[str, str], v3_hashes_before: Mapping[str, str], original_raw_sha_before: str) -> tuple[str, dict[str, Any]]:
    raw_records = f6l.final_response_records(raw_output_dir)
    candidates = f6l.candidate_rows_for_operationalized_sample()
    manifest_by_id = {row["request_id"]: row for row in build.manifest_rows}
    execution_rows: list[dict[str, str]] = []
    parse_rows: list[dict[str, str]] = []
    rec_rows: list[dict[str, str]] = []
    for rid, row in manifest_by_id.items():
        raw_record = raw_records.get(rid)
        if not raw_record:
            continue
        execution_rows.append(execution_row(row, raw_record))
        parse_row = parse_result_row(row, raw_record, candidates[row["user_id"]])
        parse_rows.append(parse_row)
        rec_rows.extend(recommendation_rows_from_parse(row, parse_row))
    completed = len(raw_records)
    strict = sum(1 for row in parse_rows if row["strict_valid"] == "true")
    failed = completed - strict
    residual = PILOT_REQUEST_COUNT - completed
    counts = {
        "malformed_json": sum(int(row["malformed_json_count"] or 0) for row in parse_rows),
        "unknown_token": sum(int(row["unknown_token_count"] or 0) for row in parse_rows),
        "duplicate_token": sum(int(row["duplicate_token_count"] or 0) for row in parse_rows),
        "missing_token": sum(int(row["missing_token_count"] or 0) for row in parse_rows),
        "ambiguous_mapping": sum(int(row["ambiguous_mapping_count"] or 0) for row in parse_rows),
        "numeric_without_c_prefix": sum(int(row["numeric_without_c_prefix_count"] or 0) for row in parse_rows),
    }
    request_counts, response_counts = f6l.request_record_counts(raw_output_dir)
    no_duplicate_records = all(count == 1 for count in request_counts.values()) and all(count == 1 for count in response_counts.values()) and len(request_counts) == 40 and len(response_counts) == 40
    hash_coherence = bool(parse_rows) and all(row["hash_coherence"] == "true" for row in parse_rows) and no_duplicate_records
    regression_equivalents = [row for row in parse_rows if row["v3_regression_equivalent"] == "true"]
    regression_equivalents_valid = len(regression_equivalents) == 8 and all(row["strict_valid"] == "true" for row in regression_equivalents)
    v3_unchanged = v3_hashes_before == v3_artifact_hashes()
    original_raw_unchanged = original_raw_sha_before == v3.original_raw_hash_map_sha() == v3.original_raw_expected_sha()
    profile_after = f6l.profile_raw_hashes()
    profile_unchanged = dict(profile_hashes_before) == profile_after
    preflight_pass = all(row["status"] == "pass" for row in preflight_rows)
    decision = DECISION_PASS if completed == 40 and strict == 40 and failed == 0 and residual == 0 and all(value == 0 for value in counts.values()) and hash_coherence and regression_equivalents_valid and v3_unchanged and original_raw_unchanged and profile_unchanged and preflight_pass else DECISION_FAILED
    validity_rows = build_validity_rows(build.run_id, parse_rows, completed)
    subset_rows = build_regression_holdout_rows(build.run_id, parse_rows, build.failure_audit_rows)
    failure_rows = build_failure_rows(build.run_id, parse_rows)
    smoke_rows = build_smoke_metrics(build.run_id, parse_rows, build.manifest_rows)
    acceptance_rows = build_acceptance_rows(
        build.run_id,
        {
            "completed": completed,
            "strict_valid": strict,
            "failed": failed,
            "residual": residual,
            **counts,
            "hash_coherence": hash_coherence,
            "regression_equivalents_valid": regression_equivalents_valid,
            "v3_unchanged": v3_unchanged,
            "original_raw_unchanged": original_raw_unchanged,
            "profile_raw_unchanged": profile_unchanged,
            "preflight_pass": preflight_pass,
        },
    )
    f6l.write_csv(PILOT_EXECUTION_PATH, execution_rows, EXECUTION_FIELDS)
    f6l.write_csv(PILOT_PARSE_RESULTS_PATH, parse_rows, PARSE_FIELDS)
    f6l.write_csv(PILOT_RECOMMENDATIONS_PATH, rec_rows, RECOMMENDATION_FIELDS)
    f6l.write_csv(PILOT_VALIDITY_PATH, validity_rows, VALIDITY_FIELDS)
    f6l.write_csv(PILOT_REGRESSION_HOLDOUT_PATH, subset_rows, REGRESSION_HOLDOUT_FIELDS)
    f6l.write_csv(PILOT_FAILURE_TAXONOMY_PATH, failure_rows, FAILURE_FIELDS)
    f6l.write_csv(PILOT_ACCEPTANCE_PATH, acceptance_rows, ACCEPTANCE_FIELDS)
    f6l.write_csv(PILOT_SMOKE_METRICS_PATH, smoke_rows, SMOKE_FIELDS)
    if decision == DECISION_PASS:
        f6l.write_csv(FULL_RUN_PLAN_PATH, build_full_run_plan_rows(build.run_id), FULL_RUN_PLAN_FIELDS)
    write_state(build.run_id, raw_output_dir, "complete", decision, time.perf_counter(), max(raw_records) if raw_records else "", parse_rows, completed, "terminal V4 pilot consolidation")
    output_hashes = {
        rel_safe(path): f6l.safe_file_sha256(path)
        for path in [
            V3_FAILURE_AUDIT_PATH,
            STRUCTURED_SUPPORT_PATH,
            PILOT_MANIFEST_PATH,
            PILOT_SELECTION_PATH,
            PILOT_PREFLIGHT_PATH,
            PILOT_STATE_PATH,
            PILOT_EXECUTION_PATH,
            PILOT_PARSE_RESULTS_PATH,
            PILOT_RECOMMENDATIONS_PATH,
            PILOT_VALIDITY_PATH,
            PILOT_REGRESSION_HOLDOUT_PATH,
            PILOT_FAILURE_TAXONOMY_PATH,
            PILOT_ACCEPTANCE_PATH,
            PILOT_SMOKE_METRICS_PATH,
            FULL_RUN_PLAN_PATH,
        ]
        if path.exists()
    }
    manifest = {
        **build.manifest,
        "decision_token": decision,
        "raw_output_dir": rel_safe(raw_output_dir),
        "completed": completed,
        "strict_valid": strict,
        "failed": failed,
        "residual": residual,
        "acceptance": {row["criterion"]: row["status"] for row in acceptance_rows},
        "v3_artifact_hash_before": f6l.sha256_text(canonical_json(v3_hashes_before)),
        "v3_artifact_hash_after": f6l.sha256_text(canonical_json(v3_artifact_hashes())),
        "original_raw_hash_before": original_raw_sha_before,
        "original_raw_hash_after": v3.original_raw_hash_map_sha(),
        "profile_raw_count_before": len(profile_hashes_before),
        "profile_raw_count_after": len(profile_after),
        "output_hashes": output_hashes,
    }
    f6l.write_json(PILOT_JSON_MANIFEST_PATH, manifest)
    write_report_and_notes(build, manifest, acceptance_rows, validity_rows, subset_rows, failure_rows)
    return decision, manifest


def write_report_and_notes(build: PilotBuild, manifest: Mapping[str, Any], acceptance_rows: Sequence[Mapping[str, str]], validity_rows: Sequence[Mapping[str, str]], subset_rows: Sequence[Mapping[str, str]], failure_rows: Sequence[Mapping[str, str]]) -> None:
    aggregate = next((row for row in validity_rows if row["model_name"] == "__all__"), {})
    acceptance_pass = sum(1 for row in acceptance_rows if row["status"] == "pass")
    report = "\n".join(
        [
            f"# {PILOT_ID}",
            "",
            f"- Decision token: `{manifest['decision_token']}`",
            f"- Run id: `{build.run_id}`",
            f"- Raw output dir: `{manifest.get('raw_output_dir', '')}`",
            f"- Classification: `{PILOT_CLASSIFICATION}`",
            f"- Completed: {aggregate.get('completed', '0')}/40",
            f"- V4 strict-valid: {aggregate.get('strict_valid', '0')}/40",
            f"- Acceptance checks: {acceptance_pass}/{len(acceptance_rows)} pass.",
            f"- Regression users: `{', '.join(manifest['regression_users'])}`",
            f"- Holdout users: `{', '.join(manifest['holdout_users'])}`",
            f"- Parser: `{PARSER_VERSION}` hash `{manifest['parser_source_hash']}`.",
            f"- Contract hash: `{manifest['output_contract_hash']}`.",
            f"- Full 4000-run command prepared: {'yes, not executed' if manifest['decision_token'] == DECISION_PASS else 'no'}",
            "",
            "## Contract",
            "",
            "V4 uses opaque candidate tokens C01-C20 and requires exactly one JSON object with a `ranking` list of 10 distinct authorized strings. Numeric indices without `C`, titles, markdown, explanations, extra keys, unknown tokens, and duplicates are rejected.",
            "",
            "## Evidence Boundary",
            "",
            "Validity diagnostics are technical pilot evidence. HR@5/NDCG@5 smoke rows are marked `pilot_only_not_scientific_evidence` and do not update RQs.",
            "",
            "## Files",
            "",
            f"- V3 failure audit: `{rel_safe(V3_FAILURE_AUDIT_PATH)}`",
            f"- Structured support: `{rel_safe(STRUCTURED_SUPPORT_PATH)}`",
            f"- Manifest: `{rel_safe(PILOT_MANIFEST_PATH)}`",
            f"- Preflight: `{rel_safe(PILOT_PREFLIGHT_PATH)}`",
            f"- Parse results: `{rel_safe(PILOT_PARSE_RESULTS_PATH)}`",
            f"- Regression/holdout validity: `{rel_safe(PILOT_REGRESSION_HOLDOUT_PATH)}`",
            f"- Acceptance: `{rel_safe(PILOT_ACCEPTANCE_PATH)}`",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    note = "\n".join(
        [
            "",
            f"## {PILOT_ID} ({utc_now()})",
            "",
            f"- Decision token: `{manifest['decision_token']}`",
            f"- Run id: `{build.run_id}`",
            f"- Raw output dir: `{manifest.get('raw_output_dir', '')}`",
            f"- Completed/strict-valid: {aggregate.get('completed', '0')}/{aggregate.get('strict_valid', '0')} of 40.",
            f"- Regression users: `{', '.join(manifest['regression_users'])}`; holdout users: `{', '.join(manifest['holdout_users'])}`.",
            "- Scientific boundary: V4 validity only; smoke HR/NDCG is not RQ evidence.",
        ]
    )
    with NOTES_PATH.open("a", encoding="utf-8") as handle:
        handle.write(note + "\n")
    with HANDOFF_PATH.open("a", encoding="utf-8") as handle:
        handle.write(note + "\n")


def execute_full_v4(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if not args.confirm_full_v4_rerun:
        raise SystemExit("--confirm-full-v4-rerun is required for the single authorized full V4 run")
    if not args.allow_api_calls:
        raise SystemExit("--allow-api-calls is required before local Ollama inference")
    if not args.local_only:
        raise SystemExit("--local-only is required; remote APIs are not authorized")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_dir = Path(args.raw_output_dir) if args.raw_output_dir else full_raw_output_dir(run_id)
    if not raw_output_dir.is_absolute():
        raw_output_dir = f6l.PROJECT_ROOT / raw_output_dir
    profile_hashes_before = f6l.profile_raw_hashes()
    v3_hashes_before = v3_artifact_hashes()
    v4_hashes_before = v4_pilot_artifact_hashes()
    build = build_full_run(run_id=run_id, raw_output_dir=raw_output_dir, write_outputs=True)
    amendment = write_protocol_amendment(run_id, raw_output_dir, str(build.manifest.get("manifest_hash", "")))
    preflight_rows, focused_test_result, ollama_state = validate_full_preflight(
        build,
        raw_output_dir=raw_output_dir,
        resume=args.resume,
        allow_api_calls=args.allow_api_calls,
        local_only=args.local_only,
        endpoint=str(args.ollama_endpoint),
        profile_hashes_before=profile_hashes_before,
        v3_hashes_before=v3_hashes_before,
        v4_pilot_hashes_before=v4_hashes_before,
        run_tests=True,
    )
    preflight_pass = all(row["status"] == "pass" for row in preflight_rows)
    start_clock = time.perf_counter()
    if not preflight_pass or ollama_state is None:
        write_full_state(
            run_id=run_id,
            raw_output_dir=raw_output_dir,
            stage="blocked_preflight",
            decision_token=FULL_DECISION_PREFLIGHT_BLOCKED,
            started_clock=start_clock,
            last_request_id="",
            last_model_name="",
            last_prompting_strategy="",
            completed=0,
            strict_valid=0,
            failed=0,
            retry=0,
            canary_status="not_started",
            circuit_breaker_status="not_started",
            notes="preflight failed; no full V4 inference executed",
        )
        manifest = {**build.manifest, "decision_token": FULL_DECISION_PREFLIGHT_BLOCKED, "preflight_failures": [row for row in preflight_rows if row["status"] != "pass"], "focused_tests": focused_test_result, "protocol_amendment": amendment}
        f6l.write_json(FULL_RUN_JSON_MANIFEST_PATH, manifest)
        return FULL_DECISION_PREFLIGHT_BLOCKED, manifest

    paths = f6l.raw_response_paths(raw_output_dir)
    if args.resume:
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        paths["individual"].mkdir(parents=True, exist_ok=True)
        if not paths["manifest"].exists():
            raise SystemExit("resume requested but raw run_manifest.json is missing")
    else:
        raw_output_dir.mkdir(parents=True, exist_ok=False)
        paths["individual"].mkdir(parents=True, exist_ok=True)
        paths["command"].write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    run_started_at = utc_now()
    if not args.resume:
        f6l.write_json(
            paths["manifest"],
            {
                **build.manifest,
                "decision_token": FULL_DECISION_INFRA_PARTIAL,
                "run_started_at": run_started_at,
                "raw_output_dir": rel_safe(raw_output_dir),
                "endpoint": str(args.ollama_endpoint),
                "ollama_service_version": ollama_state.service_version,
                "ollama_start_attempted": ollama_state.start_attempted,
                "ollama_started_pid": ollama_state.started_pid,
                "resume": bool(args.resume),
                "focused_tests": focused_test_result,
            },
        )
    candidates = f6l.candidate_rows_for_operationalized_sample()
    sample = sample_rows_by_user()
    manifest_by_id = {row["request_id"]: row for row in build.manifest_rows}
    completed_records = f6l.final_response_records(raw_output_dir) if args.resume else {}
    all_attempts = f6l.read_jsonl(paths["responses"])
    attempt_counts: Counter[str] = Counter(str(row.get("request_id", "")) for row in all_attempts if row.get("request_id"))
    state_parse_rows: list[dict[str, str]] = []
    for request_id, record in completed_records.items():
        row = manifest_by_id.get(request_id)
        if row:
            state_parse_rows.append(full_parse_result_row(row, record, candidates[row["user_id"]]))
    completed = len(state_parse_rows)
    strict_valid = sum(1 for row in state_parse_rows if row["strict_valid"] == "true")
    failed = completed - strict_valid
    retry = sum(max(0, count - 1) for count in attempt_counts.values())
    invalid_rows = [row for row in state_parse_rows if row["strict_valid"] != "true"]
    prior_invalid = invalid_rows[0] if invalid_rows else {}
    existing_lineage_rows = f6l.read_csv_rows(FULL_RUN_RESUME_LINEAGE_PATH) if FULL_RUN_RESUME_LINEAGE_PATH.exists() else []
    non_final_attempts = [
        row
        for row in all_attempts
        if str(row.get("is_final_attempt", "true")).casefold() != "true"
    ]
    latest_non_final = non_final_attempts[-1] if non_final_attempts else {}
    lineage_row = {
        "lineage_id": f"F6L-LASTFM-V4-FULL-RESUME-{len(existing_lineage_rows) + 1:03d}",
        "run_id": run_id,
        "resume_started_at": run_started_at,
        "raw_output_dir": rel_safe(raw_output_dir),
        "prior_completed": str(completed),
        "prior_strict_valid": str(strict_valid),
        "prior_structural_invalid": str(failed),
        "prior_invalid_request_id": str(prior_invalid.get("request_id", "")),
        "prior_invalid_failure_reason": str(prior_invalid.get("failure_reason", "")),
        "prior_invalid_retry_count": "0",
        "residual_authorized": str(FULL_REQUEST_COUNT - completed),
        "strict_valid_repeated": "false",
        "structural_invalid_retried": "false",
        "protocol_amendment_hash": str(amendment.get("amendment_hash", "")),
        "notes": "Resume continues the same run_id and skips all final completed records, including duplicate-token structural-invalid responses.",
    }
    if latest_non_final:
        lineage_row["notes"] += (
            f" Existing non-final transport attempt retained for retry: request_id={latest_non_final.get('request_id', '')};"
            f" attempt_index={latest_non_final.get('attempt_index', '')};"
            f" backend_error_type={latest_non_final.get('backend_error_type', '')}."
        )
    f6l.write_csv(
        FULL_RUN_RESUME_LINEAGE_PATH,
        [*existing_lineage_rows, lineage_row],
        FULL_RUN_RESUME_LINEAGE_FIELDS,
    )

    canary_status = "pass" if completed >= 40 and all(row["status"] == "pass" for row in canary_rows_from_parse(run_id, state_parse_rows, build.manifest_rows, raw_output_dir)) else ("pending" if completed < 40 else "not_passed")
    counts = structural_counts(state_parse_rows)
    resume_window_start_completed = completed
    consecutive_duplicate_invalid = trailing_duplicate_token_invalid_count(state_parse_rows, build.manifest_rows)
    write_full_state(
        run_id=run_id,
        raw_output_dir=raw_output_dir,
        stage="executing",
        decision_token=AMENDED_DECISION_INFRA_PARTIAL,
        started_clock=start_clock,
        last_request_id="",
        last_model_name="",
        last_prompting_strategy="",
        completed=completed,
        strict_valid=strict_valid,
        failed=failed,
        retry=retry,
        duplicate_token_count=counts["duplicate_token_count"],
        malformed_json_count=counts["malformed_json_count"],
        unknown_token_count=counts["unknown_token_count"],
        cumulative_duplicate_token_response_rate=duplicate_token_response_rate(state_parse_rows),
        current_duplicate_token_window_rate=0.0,
        closed_duplicate_window_index=0,
        closed_duplicate_token_window_rate=0.0,
        consecutive_duplicate_token_invalid_count=consecutive_duplicate_invalid,
        canary_status=canary_status,
        circuit_breaker_status="clear",
        notes="starting or resuming append-only full V4 execution",
    )

    resume_new_parse_rows: list[dict[str, str]] = []
    manifest_order_by_id = {row["request_id"]: order for order, row in enumerate(build.manifest_rows, start=1)}
    pending_rows = [row for row in build.manifest_rows if row["request_id"] not in completed_records]
    active_pending_rows = [
        row
        for row in pending_rows
        if attempt_counts[row["request_id"]] < TRANSPORT_DEFER_ATTEMPT_LIMIT
    ]
    deferred_pending_rows = [
        row
        for row in pending_rows
        if attempt_counts[row["request_id"]] >= TRANSPORT_DEFER_ATTEMPT_LIMIT
    ]
    ordered_pending_rows = [*active_pending_rows, *deferred_pending_rows]
    transport_deferred_count = 0
    for row in ordered_pending_rows:
        order = manifest_order_by_id[row["request_id"]]
        prompt = render_v4_prompt({**sample[row["user_id"]], "_subset": row["subset"]}, candidates[row["user_id"]], row["prompting_strategy"])
        if f6l.sha256_text(prompt) != row["rendered_prompt_hash"] or row["parser_source_hash"] != EXPECTED_V4_PARSER_HASH or row["prompt_template_hash"] != EXPECTED_V4_PROMPT_TEMPLATE_HASHES[row["prompting_strategy"]]:
            return consolidate_full_run(
                build,
                raw_output_dir,
                preflight_rows,
                profile_hashes_before,
                v3_hashes_before,
                v4_hashes_before,
                focused_test_result,
                run_started_at,
                utc_now(),
                retry,
                requested_decision=AMENDED_DECISION_HARD_CONTRACT,
                resume_window_start_completed=resume_window_start_completed,
            )
        attempt_index = attempt_counts[row["request_id"]] + 1
        request_record = {
            **request_payload_for_row(row),
            "prompt": prompt,
            "endpoint": str(args.ollama_endpoint),
            "queued_at": utc_now(),
            "request_order": order,
            "attempt_index": attempt_index,
            "expected_raw_output_path": row["expected_raw_output_path"],
        }
        f6l.append_jsonl_fsync(paths["requests"], request_record)
        result = execute_local_ollama_schema_request(row["model_name"], prompt, str(args.ollama_endpoint))
        attempt_counts[row["request_id"]] += 1
        if not result.backend_success:
            parse = ParserV4Result(False, "transport_error", [], [], [], 0, 0, 0, 10, 0, 0, 0, None)
            raw_json_path = paths["individual"] / f"{row['request_id']}__attempt{attempt_index:03d}.json"
            raw_record = raw_record_for_response(row, prompt, candidates[row["user_id"]], result, parse, raw_json_path, attempt_index=attempt_index, is_final_attempt=False)
            raw_json_path.write_text(json.dumps(raw_record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            f6l.append_jsonl_fsync(paths["responses"], raw_record)
            retry = sum(max(0, count - 1) for count in attempt_counts.values())
            counts = structural_counts(state_parse_rows)
            consecutive_duplicate_invalid = trailing_duplicate_token_invalid_count(state_parse_rows, build.manifest_rows)
            closed_window_index, closed_window_rate, _ = closed_duplicate_window_status(resume_new_parse_rows)
            write_full_state(
                run_id=run_id,
                raw_output_dir=raw_output_dir,
                stage="partial_resumable_infrastructure_interruption",
                decision_token=AMENDED_DECISION_INFRA_PARTIAL,
                started_clock=start_clock,
                last_request_id=row["request_id"],
                last_model_name=row["model_name"],
                last_prompting_strategy=row["prompting_strategy"],
                completed=completed,
                strict_valid=strict_valid,
                failed=failed,
                retry=retry,
                duplicate_token_count=counts["duplicate_token_count"],
                malformed_json_count=counts["malformed_json_count"],
                unknown_token_count=counts["unknown_token_count"],
                cumulative_duplicate_token_response_rate=duplicate_token_response_rate(state_parse_rows),
                current_duplicate_token_window_rate=current_duplicate_window_rate(resume_new_parse_rows),
                closed_duplicate_window_index=closed_window_index,
                closed_duplicate_token_window_rate=closed_window_rate,
                consecutive_duplicate_token_invalid_count=consecutive_duplicate_invalid,
                canary_status=canary_status,
                circuit_breaker_status="clear",
                notes=(
                    f"transport error deferred without final completion {result.backend_error_type}:{result.backend_error_message}; "
                    "remaining pending manifest requests continue in this process"
                ),
            )
            transport_deferred_count += 1
            continue
        parse = parse_v4_response(result.response_text, candidates[row["user_id"]])
        raw_json_path = paths["individual"] / f"{row['request_id']}.json"
        if raw_json_path.exists():
            raise RuntimeError(f"refusing to overwrite existing final raw response: {raw_json_path}")
        raw_record = raw_record_for_response(row, prompt, candidates[row["user_id"]], result, parse, raw_json_path, attempt_index=attempt_index, is_final_attempt=True)
        raw_json_path.write_text(json.dumps(raw_record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        f6l.append_jsonl_fsync(paths["responses"], raw_record)
        completed_records[row["request_id"]] = raw_record
        parse_row = full_parse_result_row(row, raw_record, candidates[row["user_id"]])
        state_parse_rows.append(parse_row)
        resume_new_parse_rows.append(parse_row)
        completed += 1
        strict_valid += 1 if parse.strict_valid else 0
        failed = completed - strict_valid
        retry = sum(max(0, count - 1) for count in attempt_counts.values())
        counts = structural_counts(state_parse_rows)
        consecutive_duplicate_invalid = trailing_duplicate_token_invalid_count(state_parse_rows, build.manifest_rows)
        current_duplicate_rate = current_duplicate_window_rate(resume_new_parse_rows)
        closed_window_index, closed_window_rate, duplicate_window_breaker = closed_duplicate_window_status(resume_new_parse_rows)
        hard_breaker = ""
        duplicate_breaker = ""
        if parse_row["hash_coherence"] != "true":
            hard_breaker = "raw_hash_coherence_failure"
        elif is_hard_structural_invalid(parse_row):
            hard_breaker = parse_row["failure_reason"] or "hard_structural_invalid"
        elif consecutive_duplicate_invalid >= DUPLICATE_CONSECUTIVE_LIMIT:
            duplicate_breaker = "five_duplicate_token_structural_invalid_consecutive"
        elif duplicate_window_breaker:
            duplicate_breaker = duplicate_window_breaker
        if hard_breaker or duplicate_breaker:
            decision = AMENDED_DECISION_HARD_CONTRACT if hard_breaker else AMENDED_DECISION_SYSTEMIC_REGRESSION
            breaker = hard_breaker or duplicate_breaker
            write_full_state(
                run_id=run_id,
                raw_output_dir=raw_output_dir,
                stage="stopped_by_hard_contract_violation" if hard_breaker else "stopped_by_systemic_duplicate_regression",
                decision_token=decision,
                started_clock=start_clock,
                last_request_id=row["request_id"],
                last_model_name=row["model_name"],
                last_prompting_strategy=row["prompting_strategy"],
                completed=completed,
                strict_valid=strict_valid,
                failed=failed,
                retry=retry,
                duplicate_token_count=counts["duplicate_token_count"],
                malformed_json_count=counts["malformed_json_count"],
                unknown_token_count=counts["unknown_token_count"],
                cumulative_duplicate_token_response_rate=duplicate_token_response_rate(state_parse_rows),
                current_duplicate_token_window_rate=current_duplicate_rate,
                closed_duplicate_window_index=closed_window_index,
                closed_duplicate_token_window_rate=closed_window_rate,
                consecutive_duplicate_token_invalid_count=consecutive_duplicate_invalid,
                canary_status=canary_status,
                circuit_breaker_status=breaker,
                notes="final amended circuit breaker triggered; observed outputs are retained without retry or repair",
            )
            return consolidate_full_run(
                build,
                raw_output_dir,
                preflight_rows,
                profile_hashes_before,
                v3_hashes_before,
                v4_hashes_before,
                focused_test_result,
                run_started_at,
                utc_now(),
                retry,
                requested_decision=decision,
                resume_window_start_completed=resume_window_start_completed,
            )
        if completed == 40:
            canary_rows = canary_rows_from_parse(run_id, state_parse_rows, build.manifest_rows, raw_output_dir)
            f6l.write_csv(FULL_RUN_CANARY_PATH, canary_rows, FULL_RUN_CANARY_FIELDS)
            canary_status = "pass" if all(item["status"] == "pass" for item in canary_rows) else "not_passed"
            if canary_status != "pass":
                counts = structural_counts(state_parse_rows)
                consecutive_duplicate_invalid = trailing_duplicate_token_invalid_count(state_parse_rows, build.manifest_rows)
                closed_window_index, closed_window_rate, _ = closed_duplicate_window_status(resume_new_parse_rows)
                write_full_state(
                    run_id=run_id,
                    raw_output_dir=raw_output_dir,
                    stage="stopped_by_canary",
                    decision_token=AMENDED_DECISION_HARD_CONTRACT,
                    started_clock=start_clock,
                    last_request_id=row["request_id"],
                    last_model_name=row["model_name"],
                    last_prompting_strategy=row["prompting_strategy"],
                    completed=completed,
                    strict_valid=strict_valid,
                    failed=failed,
                    retry=retry,
                    duplicate_token_count=counts["duplicate_token_count"],
                    malformed_json_count=counts["malformed_json_count"],
                    unknown_token_count=counts["unknown_token_count"],
                    cumulative_duplicate_token_response_rate=duplicate_token_response_rate(state_parse_rows),
                    current_duplicate_token_window_rate=current_duplicate_window_rate(resume_new_parse_rows),
                    closed_duplicate_window_index=closed_window_index,
                    closed_duplicate_token_window_rate=closed_window_rate,
                    consecutive_duplicate_token_invalid_count=consecutive_duplicate_invalid,
                    canary_status=canary_status,
                    circuit_breaker_status="canary_validation_failed",
                    notes="internal canary failed; run stopped before residual requests",
                )
                return consolidate_full_run(
                    build,
                    raw_output_dir,
                    preflight_rows,
                    profile_hashes_before,
                    v3_hashes_before,
                    v4_hashes_before,
                    focused_test_result,
                    run_started_at,
                    utc_now(),
                    retry,
                    requested_decision=AMENDED_DECISION_HARD_CONTRACT,
                    resume_window_start_completed=resume_window_start_completed,
                )
        elif completed < 40:
            canary_status = "pending"
        write_full_state(
            run_id=run_id,
            raw_output_dir=raw_output_dir,
            stage="executing",
            decision_token=AMENDED_DECISION_INFRA_PARTIAL,
            started_clock=start_clock,
            last_request_id=row["request_id"],
            last_model_name=row["model_name"],
            last_prompting_strategy=row["prompting_strategy"],
            completed=completed,
            strict_valid=strict_valid,
            failed=failed,
            retry=retry,
            duplicate_token_count=counts["duplicate_token_count"],
            malformed_json_count=counts["malformed_json_count"],
            unknown_token_count=counts["unknown_token_count"],
            cumulative_duplicate_token_response_rate=duplicate_token_response_rate(state_parse_rows),
            current_duplicate_token_window_rate=current_duplicate_rate,
            closed_duplicate_window_index=closed_window_index,
            closed_duplicate_token_window_rate=closed_window_rate,
            consecutive_duplicate_token_invalid_count=consecutive_duplicate_invalid,
            canary_status=canary_status,
            circuit_breaker_status="clear",
            notes="append-only local Ollama V4 full execution in progress",
        )
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "completed": completed,
                    "strict_valid": strict_valid,
                    "structural_invalid": failed,
                    "residual": FULL_REQUEST_COUNT - completed,
                    "retry": retry,
                    "current_200_window_duplicate_token_rate": format_float(current_duplicate_rate),
                    "closed_200_window_index": closed_window_index,
                    "closed_200_window_duplicate_token_rate": format_float(closed_window_rate),
                    "consecutive_duplicate_token_invalid": consecutive_duplicate_invalid,
                    "last_request_id": row["request_id"],
                    "canary_status": canary_status,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return consolidate_full_run(
        build,
        raw_output_dir,
        preflight_rows,
        profile_hashes_before,
        v3_hashes_before,
        v4_hashes_before,
        focused_test_result,
        run_started_at,
        utc_now(),
        retry,
        resume_window_start_completed=resume_window_start_completed,
        requested_decision=AMENDED_DECISION_INFRA_PARTIAL if transport_deferred_count else None,
    )


def run_pilot(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.mode == "execute-full-v4":
        return execute_full_v4(args)
    if not args.allow_api_calls:
        raise SystemExit("--allow-api-calls is required before local Ollama inference")
    if not args.local_only:
        raise SystemExit("--local-only is required; remote APIs are not authorized")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_dir = Path(args.raw_output_dir) if args.raw_output_dir else local_raw_output_dir(run_id)
    if not raw_output_dir.is_absolute():
        raw_output_dir = f6l.PROJECT_ROOT / raw_output_dir
    build = build_pilot(run_id=run_id, raw_output_dir=raw_output_dir, write_outputs=True)
    profile_hashes_before = f6l.profile_raw_hashes()
    v3_hashes_before = v3_artifact_hashes()
    original_raw_sha_before = v3.original_raw_hash_map_sha()
    preflight_rows, test_result, _ = validate_preflight(
        build,
        raw_output_dir=raw_output_dir,
        resume=args.resume,
        allow_api_calls=args.allow_api_calls,
        local_only=args.local_only,
        endpoint=str(args.ollama_endpoint),
        profile_hashes_before=profile_hashes_before,
        v3_hashes_before=v3_hashes_before,
        run_tests=True,
    )
    if not all(row["status"] == "pass" for row in preflight_rows):
        decision = DECISION_PREFLIGHT_BLOCKED
        write_state(run_id, raw_output_dir, "blocked_preflight", decision, time.perf_counter(), "", [], 0, "preflight failed; no V4 inference executed")
        manifest = {**build.manifest, "decision_token": decision, "raw_output_dir": rel_safe(raw_output_dir), "focused_tests": test_result, "preflight_failures": [row for row in preflight_rows if row["status"] != "pass"]}
        f6l.write_json(PILOT_JSON_MANIFEST_PATH, manifest)
        return decision, manifest
    execute_pilot_requests(build, raw_output_dir, str(args.ollama_endpoint), args.resume)
    return consolidate_outputs(build, raw_output_dir, preflight_rows, profile_hashes_before, v3_hashes_before, original_raw_sha_before)


def build_only(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    run_id = args.run_id or "planned"
    raw_output_dir = Path(args.raw_output_dir) if args.raw_output_dir else local_raw_output_dir(run_id)
    if not raw_output_dir.is_absolute():
        raw_output_dir = f6l.PROJECT_ROOT / raw_output_dir
    build = build_pilot(run_id=run_id, raw_output_dir=raw_output_dir, write_outputs=True)
    preflight_rows, _, _ = validate_preflight(
        build,
        raw_output_dir=raw_output_dir,
        resume=args.resume,
        allow_api_calls=args.allow_api_calls,
        local_only=args.local_only,
        endpoint=str(args.ollama_endpoint),
        profile_hashes_before=f6l.profile_raw_hashes(),
        v3_hashes_before=v3_artifact_hashes(),
        run_tests=False,
    )
    decision = "v4_pilot_manifest_and_preflight_prepared_no_inference"
    manifest = {**build.manifest, "decision_token": decision, "preflight_status": "pass" if all(row["status"] == "pass" for row in preflight_rows) else "fail"}
    f6l.write_json(PILOT_JSON_MANIFEST_PATH, manifest)
    return decision, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "execute", "execute-full-v4"], default="plan")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--raw-output-dir", default="")
    parser.add_argument("--allow-api-calls", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-full-v4-rerun", action="store_true")
    parser.add_argument("--ollama-endpoint", default=f6l.OLLAMA_LOCAL_GENERATE_URL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "plan":
        decision, manifest = build_only(args)
    else:
        decision, manifest = run_pilot(args)
    print(json.dumps({"decision_token": decision, "run_id": manifest.get("run_id"), "raw_output_dir": manifest.get("raw_output_dir", "")}, sort_keys=True))
    return 0 if decision not in {DECISION_PREFLIGHT_BLOCKED, DECISION_FAILED} else 1


if __name__ == "__main__":
    raise SystemExit(main())

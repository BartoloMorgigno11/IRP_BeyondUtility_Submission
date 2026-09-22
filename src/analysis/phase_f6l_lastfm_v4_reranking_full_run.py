"""F6L LastFM V4 reranking full-run contract and local runner.

This module executes a separate 1000-user LastFM reranking run. It reuses the
accepted V4 opaque-token parser/schema from the F6L ranking run, but freezes a
new reranking manifest, prompt contract, candidate source, raw directory, and
result tables. It never consumes ranking V4 outputs as reranking inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.analysis import phase_f6l_lastfm_corrective_output_contract_v4_pilot as v4
from src.analysis import plan_f6l_paper_faithful_completion_and_rq_revalidation as f6l


PHASE_ID = "phase_f6l"
FULL_RUN_ID = "phase_f6l_lastfm_v4_reranking_full_run"
RUN_DESCRIPTION = "LastFM sample reranking reproduction under explicit local operationalization"
CLASSIFICATION = "paper_protocol_reproduction_under_explicit_operationalization"
DECISION_PREFLIGHT_PASSED = "f6l_lastfm_v4_reranking_preflight_passed_ready_to_launch"
DECISION_PREFLIGHT_BLOCKED = "f6l_lastfm_v4_reranking_blocked_by_preflight_regression"
DECISION_PARTIAL = "f6l_lastfm_v4_reranking_partial_resumable_infrastructure_interruption"
DECISION_COMPLETE_ALL_STRICT = "f6l_lastfm_v4_reranking_complete_all_strict_valid"
DECISION_COMPLETE_WITH_INVALID = "f6l_lastfm_v4_reranking_complete_with_observed_duplicate_invalidity"
DECISION_HARD_CONTRACT = "f6l_lastfm_v4_reranking_stopped_by_hard_contract_violation"
DECISION_SYSTEMIC_DUPLICATE = "f6l_lastfm_v4_reranking_stopped_by_systemic_duplicate_regression"

TABLES_DIR = f6l.PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = f6l.PROJECT_ROOT / "results" / "manifests"
LOGS_DIR = f6l.PROJECT_ROOT / "results" / "logs"
REPORT_PATH = f6l.PROJECT_ROOT / "report" / "phase_f6l_lastfm_v4_reranking_full_run.md"

MANIFEST_PATH = TABLES_DIR / f"{FULL_RUN_ID}_manifest.csv"
FREEZE_PATH = TABLES_DIR / f"{FULL_RUN_ID}_freeze.csv"
PREFLIGHT_PATH = TABLES_DIR / f"{FULL_RUN_ID}_preflight.csv"
STATE_PATH = TABLES_DIR / f"{FULL_RUN_ID}_state.csv"
CANARY_PATH = TABLES_DIR / f"{FULL_RUN_ID}_canary.csv"
EXECUTION_PATH = TABLES_DIR / f"{FULL_RUN_ID}_execution.csv"
PARSE_RESULTS_PATH = TABLES_DIR / f"{FULL_RUN_ID}_parse_results.csv"
RECOMMENDATIONS_PATH = TABLES_DIR / f"{FULL_RUN_ID}_recommendation_rows.csv"
METRICS_PATH = TABLES_DIR / f"{FULL_RUN_ID}_metrics.csv"
BEYOND_UTILITY_PATH = TABLES_DIR / f"{FULL_RUN_ID}_beyond_utility_metrics.csv"
COMPARISON_PATH = TABLES_DIR / f"{FULL_RUN_ID}_strategy_comparison.csv"
FINAL_VALIDATION_PATH = TABLES_DIR / f"{FULL_RUN_ID}_final_validation.csv"
ACCEPTANCE_PATH = TABLES_DIR / f"{FULL_RUN_ID}_acceptance.csv"
FAILURE_TAXONOMY_PATH = TABLES_DIR / f"{FULL_RUN_ID}_failure_taxonomy.csv"
DUPLICATE_WINDOWS_PATH = TABLES_DIR / f"{FULL_RUN_ID}_duplicate_windows.csv"
RUNTIME_PATH = TABLES_DIR / f"{FULL_RUN_ID}_runtime.csv"
RANKING_BASELINE_AUDIT_PATH = TABLES_DIR / f"{FULL_RUN_ID}_ranking_baseline_audit.csv"
RANKING_BASELINE_COMPARISON_PATH = TABLES_DIR / f"{FULL_RUN_ID}_ranking_baseline_comparison.csv"
JSON_MANIFEST_PATH = MANIFESTS_DIR / f"{FULL_RUN_ID}_manifest.json"

RANKING_BASELINE_RUN_ID = "20260804_131959"
RANKING_BASELINE_DECISION_TOKEN = "f6l_lastfm_v4_ranking_metrics_consolidated_and_frozen_for_reranking"
RANKING_BASELINE_FREEZE_HASH = "D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC"
RANKING_BASELINE_PATHS = {
    "metrics": TABLES_DIR / "phase_f6l_lastfm_v4_ranking_freeze_metrics.csv",
    "comparisons": TABLES_DIR / "phase_f6l_lastfm_v4_ranking_freeze_comparisons.csv",
    "statistics": TABLES_DIR / "phase_f6l_lastfm_v4_ranking_freeze_statistical_tests.csv",
    "summary": MANIFESTS_DIR / "phase_f6l_lastfm_v4_ranking_metrics_freeze_summary.json",
    "hash_text": MANIFESTS_DIR / "phase_f6l_lastfm_v4_ranking_metrics_freeze.sha256",
    "report": f6l.PROJECT_ROOT / "report" / "phase_f6l_lastfm_v4_ranking_metrics_freeze.md",
}
RANKING_BASELINE_EXPECTED_HASHES = {
    "metrics": "2CC69C5611EEB44F0DC94B60EF699C5A688E022EC778597DA4070224268DE884",
    "comparisons": "673ECB5046FA38DB2ADAF510FC1C4EB84BA778CA1D19CECAB64BEA8ADC49A6EF",
    "statistics": "C2F15E9713AE37C8C11BE80BBCC0814F4FF24E979F54D10FAB6946AC7E5E7C04",
    "summary": "5AEC146291E96B911533D4C86566E0EDC0976F7FD0DA8F4E62D923E9DC2CD028",
    "hash_text": "734E1AB3779916A781DD8B70077874C773D862F0922A5B34D361CB9CE4E41B83",
    "report": "5BBBA61AE09CA0DD55C93B389867E293E08216B9CAC8BA894EDC5047D8150F3D",
}
RANKING_BASELINE_EXPECTED_METRICS = {
    ("qwen2:7b-instruct", "base/OpenP5 title-list", "HR@5_fail_strict"): "0.458",
    ("qwen2:7b-instruct", "base/OpenP5 title-list", "NDCG@5_fail_strict"): "0.323647507835",
    ("qwen2:7b-instruct", "base/OpenP5 title-list", "mean_positive_rank_fail_strict_invalid_as_20"): "9.712",
    ("qwen2:7b-instruct", "base/OpenP5 title-list", "strict_valid_rate"): "1",
    ("qwen2:7b-instruct", "base/OpenP5 title-list", "duplicate_token_rate"): "0",
    ("qwen2:7b-instruct", "LLMRank recency", "HR@5_fail_strict"): "0.419",
    ("qwen2:7b-instruct", "LLMRank recency", "NDCG@5_fail_strict"): "0.308222730898",
    ("qwen2:7b-instruct", "LLMRank recency", "mean_positive_rank_fail_strict_invalid_as_20"): "10.107",
    ("qwen2:7b-instruct", "LLMRank recency", "strict_valid_rate"): "1",
    ("qwen2:7b-instruct", "LLMRank recency", "duplicate_token_rate"): "0",
    ("mistral:7b-instruct", "base/OpenP5 title-list", "HR@5_fail_strict"): "0.307",
    ("mistral:7b-instruct", "base/OpenP5 title-list", "NDCG@5_fail_strict"): "0.223073398036",
    ("mistral:7b-instruct", "base/OpenP5 title-list", "mean_positive_rank_fail_strict_invalid_as_20"): "12.202",
    ("mistral:7b-instruct", "base/OpenP5 title-list", "strict_valid_rate"): "0.983",
    ("mistral:7b-instruct", "base/OpenP5 title-list", "duplicate_token_rate"): "0.017",
    ("mistral:7b-instruct", "LLMRank recency", "HR@5_fail_strict"): "0.319",
    ("mistral:7b-instruct", "LLMRank recency", "NDCG@5_fail_strict"): "0.232876217620",
    ("mistral:7b-instruct", "LLMRank recency", "mean_positive_rank_fail_strict_invalid_as_20"): "12.056",
    ("mistral:7b-instruct", "LLMRank recency", "strict_valid_rate"): "0.967",
    ("mistral:7b-instruct", "LLMRank recency", "duplicate_token_rate"): "0.033",
}
RANKING_TO_RERANKING_STRATEGY = {
    "base/OpenP5-derived reranking": "base/OpenP5 title-list",
    "LLMRank recency-focused reranking": "LLMRank recency",
}

MODEL_ORDER = ["qwen2:7b-instruct", "mistral:7b-instruct"]
STRATEGY_ORDER = ["base/OpenP5-derived reranking", "LLMRank recency-focused reranking"]
TOKEN_LIST = v4.TOKEN_LIST
REQUEST_COUNT = 4000
USER_COUNT = 1000
CELL_REQUEST_COUNT = 1000
CANARY_REGRESSION_USERS = ["640", "709", "1333", "1749"]
CANARY_HOLDOUT_USERS = ["223", "387", "2090", "208", "895", "609"]
CANARY_USERS = CANARY_REGRESSION_USERS + CANARY_HOLDOUT_USERS
TRANSPORT_DEFER_ATTEMPT_LIMIT = v4.TRANSPORT_DEFER_ATTEMPT_LIMIT
DUPLICATE_WINDOW_SIZE = v4.DUPLICATE_WINDOW_SIZE
DUPLICATE_WINDOW_RATE_LIMIT = v4.DUPLICATE_WINDOW_RATE_LIMIT
DUPLICATE_CONSECUTIVE_LIMIT = v4.DUPLICATE_CONSECUTIVE_LIMIT

EXPECTED_V4_PARSER_HASH = v4.EXPECTED_V4_PARSER_HASH
EXPECTED_V4_SCHEMA_HASH = v4.EXPECTED_V4_SCHEMA_HASH
EXPECTED_V4_OUTPUT_CONTRACT_HASH = v4.EXPECTED_V4_OUTPUT_CONTRACT_HASH

GENERATION_PARAMETERS = dict(v4.GENERATION_PARAMETERS)
GENERATION_PARAMETERS["scientific_retry_count"] = 0
GENERATION_PARAMETERS["infrastructure_retry_count"] = 0

FORBIDDEN_PROMPT_TOKENS = v4.FORBIDDEN_PROMPT_TOKENS

MANIFEST_FIELDS = [
    "request_id",
    "phase_id",
    "full_run_id",
    "run_id",
    "task",
    "sample_id",
    "subset",
    "user_id",
    "history_length",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "candidate_pool_source",
    "candidate_pool_hash",
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
    "positive_item_id",
    "positive_candidate_position",
    "positive_present",
    "expected_raw_output_path",
    "evidence_class",
    "lineage",
    "request_contract_hash",
]
FREEZE_FIELDS = ["freeze_id", "run_id", "artifact", "sha256", "status", "evidence", "notes"]
PREFLIGHT_FIELDS = ["check_id", "check_name", "status", "observed", "expected", "severity", "evidence", "notes"]
STATE_FIELDS = [
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
CANARY_FIELDS = ["run_id", "criterion", "observed", "expected", "status", "evidence", "notes"]
EXECUTION_FIELDS = [
    "request_id",
    "full_run_id",
    "run_id",
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
    "full_run_id",
    "run_id",
    "task",
    "sample_id",
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
    "positive_item_id",
    "positive_candidate_position",
    "positive_present",
    "positive_rank",
    "request_contract_hash",
    "hash_coherence",
    "evidence_class",
    "notes",
]
RECOMMENDATION_FIELDS = [
    "request_id",
    "full_run_id",
    "run_id",
    "task",
    "sample_id",
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
METRIC_FIELDS = [
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
    "positive_present_rows",
    "source_file",
    "notes",
]
BEYOND_FIELDS = [
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
COMPARISON_FIELDS = [
    "comparison_id",
    "run_id",
    "sample_id",
    "model_name",
    "metric",
    "base_openp5_value",
    "llmrank_recency_value",
    "delta_llmrank_minus_base",
    "shared_users",
    "positive_present_users",
    "evidence_class",
    "status",
    "notes",
]
FINAL_VALIDATION_FIELDS = ["validation_id", "check_name", "status", "observed", "expected", "severity", "evidence", "notes"]
ACCEPTANCE_FIELDS = ["criterion", "status", "observed", "expected", "evidence", "notes"]
FAILURE_TAXONOMY_FIELDS = [
    "failure_reason",
    "count",
    "requests",
    "rate_over_completed",
    "policy",
    "evidence_class",
    "notes",
]
DUPLICATE_WINDOW_FIELDS = v4.FULL_RUN_DUPLICATE_WINDOW_FIELDS
RUNTIME_FIELDS = [
    "run_id",
    "started_at",
    "completed_at",
    "duration_seconds",
    "planned",
    "completed",
    "strict_valid",
    "structural_invalid",
    "retry",
    "throughput_requests_per_hour",
    "raw_output_dir",
    "notes",
]
RANKING_BASELINE_AUDIT_FIELDS = ["audit_id", "check_name", "status", "observed", "expected", "evidence", "notes"]
RANKING_BASELINE_COMPARISON_FIELDS = [
    "comparison_id",
    "run_id",
    "ranking_run_id",
    "dataset",
    "model_name",
    "ranking_strategy",
    "reranking_strategy",
    "metric",
    "ranking_fail_strict_value",
    "reranking_fail_strict_value",
    "delta_reranking_minus_ranking",
    "ranking_freeze_hash",
    "ranking_source_file",
    "reranking_source_file",
    "status",
    "notes",
]


@dataclass(frozen=True)
class RerankingBuild:
    run_id: str
    raw_output_dir: Path
    manifest_rows: list[dict[str, str]]
    manifest: dict[str, Any]
    freeze_rows: list[dict[str, str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return v4.canonical_json(value)


def json_list(values: Iterable[Any]) -> str:
    return v4.json_list(values)


def format_float(value: float, digits: int = 12) -> str:
    return v4.format_float(value, digits=digits)


def stable_hash(*parts: object) -> str:
    return v4.stable_hash(*parts)


def rel_safe(path: Path) -> str:
    return v4.rel_safe(path)


def decimal_equal(left: str, right: str) -> bool:
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, ValueError):
        return str(left) == str(right)


def decimal_delta(right: str, left: str) -> str:
    try:
        return format_float(float(Decimal(str(right)) - Decimal(str(left))))
    except (InvalidOperation, ValueError):
        return ""


def ranking_baseline_summary() -> dict[str, Any]:
    return f6l.read_json_file(RANKING_BASELINE_PATHS["summary"])


def ranking_baseline_metric_lookup() -> dict[tuple[str, str, str], str]:
    rows = f6l.read_csv_rows(RANKING_BASELINE_PATHS["metrics"])
    return {(row["model_name"], row["prompting_strategy"], row["metric"]): row.get("value", "") for row in rows}


def ranking_baseline_audit_rows(write_outputs: bool = True) -> list[dict[str, str]]:
    summary = ranking_baseline_summary()
    hash_text = RANKING_BASELINE_PATHS["hash_text"].read_text(encoding="utf-8") if RANKING_BASELINE_PATHS["hash_text"].exists() else ""
    metric_lookup = ranking_baseline_metric_lookup()
    rows: list[dict[str, str]] = []

    def add(name: str, passed: bool, observed: str, expected: str, evidence: str, notes: str) -> None:
        rows.append(
            {
                "audit_id": f"F6L-LASTFM-V4-RERANK-BASELINE-{len(rows) + 1:02d}",
                "check_name": name,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "evidence": evidence,
                "notes": notes,
            }
        )

    summary_freeze = str(summary.get("freeze_bundle_sha256", ""))
    hash_file_freeze = hash_text.split()[0] if hash_text.strip() else ""
    add(
        "freeze_hash_identical",
        summary_freeze == RANKING_BASELINE_FREEZE_HASH and hash_file_freeze == RANKING_BASELINE_FREEZE_HASH,
        f"summary={summary_freeze};sha256_file={hash_file_freeze}",
        RANKING_BASELINE_FREEZE_HASH,
        f"{rel_safe(RANKING_BASELINE_PATHS['summary'])};{rel_safe(RANKING_BASELINE_PATHS['hash_text'])}",
        "Ranking baseline freeze hash must remain exactly unchanged.",
    )
    add(
        "decision_token_identical",
        str(summary.get("decision_token", "")) == RANKING_BASELINE_DECISION_TOKEN,
        str(summary.get("decision_token", "")),
        RANKING_BASELINE_DECISION_TOKEN,
        rel_safe(RANKING_BASELINE_PATHS["summary"]),
        "Only the frozen ranking metrics decision is accepted before reranking.",
    )
    for artifact, path in RANKING_BASELINE_PATHS.items():
        expected = RANKING_BASELINE_EXPECTED_HASHES[artifact]
        actual = f6l.safe_file_sha256(path)
        add(
            f"artifact_byte_invariant_{artifact}",
            actual == expected,
            actual,
            expected,
            rel_safe(path),
            "Listed ranking baseline artifact must remain byte-identical.",
        )
    metric_failures = []
    for key, expected in RANKING_BASELINE_EXPECTED_METRICS.items():
        actual = metric_lookup.get(key, "")
        if not decimal_equal(actual, expected):
            metric_failures.append(f"{key[0]}|{key[1]}|{key[2]}:{actual}!={expected}")
    add(
        "fail_strict_metric_values_identical",
        not metric_failures,
        "none" if not metric_failures else ";".join(metric_failures[:5]),
        "all authoritative fail-strict ranking values",
        rel_safe(RANKING_BASELINE_PATHS["metrics"]),
        "Ranking fail-strict values are read from the frozen CSV and never recomputed.",
    )
    add(
        "ranking_values_not_regenerated_or_recomputed",
        True,
        "read_only_frozen_artifacts",
        "no ranking regeneration or recalculation",
        rel_safe(RANKING_BASELINE_PATHS["metrics"]),
        "Reranking code consumes frozen values only for downstream comparison tables.",
    )
    add(
        "complete_case_ranking_secondary_only",
        True,
        "secondary_only",
        "primary comparisons use fail-strict",
        rel_safe(RANKING_BASELINE_PATHS["comparisons"]),
        "Complete-case ranking remains a secondary analysis and is not used as the primary baseline.",
    )
    add(
        "historical_pre_existing_gap_classified",
        True,
        "historical_pre_existing_gap",
        "historical_pre_existing_gap",
        "tests/test_phase_f6l_lastfm_execution_contracts.py",
        "The historical missing-local-evidence gap is not auto-blocking unless reranking source evidence depends on it.",
    )
    if write_outputs:
        f6l.write_csv(RANKING_BASELINE_AUDIT_PATH, rows, RANKING_BASELINE_AUDIT_FIELDS)
    return rows


def ranking_baseline_passes() -> bool:
    return all(row["status"] == "pass" for row in ranking_baseline_audit_rows(write_outputs=False))


def local_raw_output_dir(run_id: str) -> Path:
    return f6l.PROJECT_ROOT / "results" / "llm" / f"{FULL_RUN_ID}_{run_id}"


def output_contract_hash() -> str:
    return v4.output_contract_hash()


def parser_source_hash() -> str:
    return v4.parser_source_hash()


def schema_hash() -> str:
    return str(GENERATION_PARAMETERS["format_schema_hash"])


def model_digest_hash() -> str:
    return f6l.sha256_text(canonical_json({model: f6l.MODEL_DIGESTS[model] for model in MODEL_ORDER}))


def prompt_template_name(strategy: str) -> str:
    if strategy == "base/OpenP5-derived reranking":
        return f"{FULL_RUN_ID}_base_openp5_derived_opaque_token_json"
    if strategy == "LLMRank recency-focused reranking":
        return f"{FULL_RUN_ID}_llmrank_recency_focused_opaque_token_json"
    raise ValueError(f"unsupported strategy: {strategy}")


def prompt_template_text(strategy: str) -> str:
    shared = "\n".join(
        [
            "You are reranking LastFM candidate artists/items for a recommender task.",
            "Use the user's history and the supplied initial candidate order only.",
            "The candidate list is a frozen 20-item initial ranking reconstructed from local BPRMF, GRU4Rec, SASRec, and LightGCN evidence.",
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
            "Initial candidate order:",
            "{candidate_block}",
            "",
            "Task: rerank the supplied candidates and return the 10 candidates most likely to match the user's next preferred artist/item, from most likely to least likely.",
            "Final answer must be only the JSON object.",
        ]
    )
    if strategy == "base/OpenP5-derived reranking":
        return shared
    if strategy == "LLMRank recency-focused reranking":
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


@lru_cache(maxsize=1)
def sample_rows_by_user() -> dict[str, dict[str, str]]:
    return {row["user_id"]: row for row in f6l.read_csv_rows(f6l.LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH)}


@lru_cache(maxsize=1)
def reranking_pool_rows_by_user() -> dict[str, list[dict[str, str]]]:
    by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in f6l.read_csv_rows(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH):
        by_user[row["user_id"]].append(row)
    for rows in by_user.values():
        rows.sort(key=lambda item: int(item["candidate_position"]))
    return dict(by_user)


def candidate_set_hash() -> str:
    rows = f6l.read_csv_rows(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH)
    payload = [
        {
            "user_id": row["user_id"],
            "candidate_position": row["candidate_position"],
            "candidate_item_id": row["candidate_item_id"],
            "candidate_item_title": row["candidate_item_title"],
            "is_positive": row.get("is_positive", ""),
            "positive_item": row.get("positive_item", ""),
            "positive_position": row.get("positive_position", ""),
            "pool_hash": row.get("pool_hash", ""),
            "source_models": row.get("source_models", ""),
            "construction_policy": row.get("construction_policy", ""),
            "evidence_class": row.get("evidence_class", ""),
        }
        for row in sorted(rows, key=lambda item: (int(item["user_id"]), int(item["candidate_position"])))
    ]
    return f6l.sha256_text(canonical_json(payload))


def token_mapping(candidate_rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    return v4.token_mapping(candidate_rows)


def token_mapping_hash(candidate_rows: Sequence[Mapping[str, str]]) -> str:
    return v4.token_mapping_hash(candidate_rows)


def render_history_block(sample_row: Mapping[str, str]) -> str:
    return v4.render_history_block(sample_row)


def render_candidate_block(candidate_rows: Sequence[Mapping[str, str]]) -> str:
    return v4.render_candidate_block(candidate_rows)


def render_prompt(sample_row: Mapping[str, str], candidate_rows: Sequence[Mapping[str, str]], strategy: str) -> str:
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


def positive_metadata(candidate_rows: Sequence[Mapping[str, str]]) -> tuple[str, str, str]:
    positives = [row for row in candidate_rows if str(row.get("is_positive", "")).lower() == "true"]
    if not positives:
        return "", "", "false"
    row = positives[0]
    return str(row["candidate_item_id"]), str(row["candidate_position"]), "true"


def positive_coverage() -> dict[str, Any]:
    by_user = reranking_pool_rows_by_user()
    positive_counts = Counter(
        user_id
        for user_id, rows in by_user.items()
        if sum(1 for row in rows if str(row.get("is_positive", "")).lower() == "true") == 1
    )
    present = len(positive_counts)
    return {
        "users": len(by_user),
        "positive_present_users": present,
        "positive_absent_users": len(by_user) - present,
        "positive_present_rate": present / len(by_user) if by_user else 0.0,
    }


def full_user_rows() -> list[dict[str, str]]:
    sample = sample_rows_by_user()
    missing = [user for user in CANARY_USERS if user not in sample]
    if missing:
        raise ValueError(f"canary users missing from operationalized sample: {missing}")
    selected: list[dict[str, str]] = []
    for user_id in CANARY_REGRESSION_USERS:
        selected.append({**sample[user_id], "_subset": "canary_regression"})
    for user_id in CANARY_HOLDOUT_USERS:
        selected.append({**sample[user_id], "_subset": "canary_holdout"})
    remaining = [row for row in sample.values() if row["user_id"] not in set(CANARY_USERS)]
    remaining.sort(key=lambda row: (int(row.get("sample_ordinal", "0") or 0), int(row["user_id"])))
    selected.extend({**row, "_subset": "operationalized_main"} for row in remaining)
    return selected


def request_id(index: int, user_id: str, model_name: str, strategy: str) -> str:
    suffix = stable_hash(FULL_RUN_ID, user_id, model_name, strategy)[:10]
    return f"F6L-LASTFM-V4-RERANK-FULL-{index:04d}-{suffix}"


def manifest_row_for_request(
    *,
    index: int,
    run_id: str,
    raw_output_dir: Path,
    sample_row: Mapping[str, str],
    candidate_rows: Sequence[Mapping[str, str]],
    model_name: str,
    strategy: str,
) -> dict[str, str]:
    prompt = render_prompt(sample_row, candidate_rows, strategy)
    mapping = token_mapping(candidate_rows)
    candidate_ids = [str(row["candidate_item_id"]) for row in sorted(candidate_rows, key=lambda item: int(item["candidate_position"]))]
    pool_hashes = {str(row.get("pool_hash", "")) for row in candidate_rows if row.get("pool_hash")}
    candidate_pool_hash = sorted(pool_hashes)[0] if len(pool_hashes) == 1 else f6l.sha256_text(canonical_json(candidate_ids))
    positive_item, positive_position, positive_present = positive_metadata(candidate_rows)
    rid = request_id(index, sample_row["user_id"], model_name, strategy)
    row = {
        "request_id": rid,
        "phase_id": PHASE_ID,
        "full_run_id": FULL_RUN_ID,
        "run_id": run_id,
        "task": "reranking",
        "sample_id": "phase_f6l_lastfm_operationalized_1000",
        "subset": sample_row["_subset"],
        "user_id": sample_row["user_id"],
        "history_length": sample_row.get("train_history_count", ""),
        "model_name": model_name,
        "model_digest": f6l.MODEL_DIGESTS[model_name],
        "prompting_strategy": strategy,
        "candidate_pool_source": rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH),
        "candidate_pool_hash": candidate_pool_hash,
        "candidate_list": json_list(candidate_ids),
        "candidate_token_mapping": canonical_json(mapping),
        "candidate_list_hash": candidate_pool_hash,
        "candidate_token_mapping_hash": token_mapping_hash(candidate_rows),
        "prompt_template": prompt_template_name(strategy),
        "prompt_template_hash": prompt_template_hash(strategy),
        "rendered_prompt_hash": f6l.sha256_text(prompt),
        "parser_version": v4.PARSER_VERSION,
        "parser_source_hash": parser_source_hash(),
        "output_contract": v4.OUTPUT_CONTRACT,
        "output_contract_hash": output_contract_hash(),
        "generation_parameters": canonical_json(GENERATION_PARAMETERS),
        "positive_item_id": positive_item,
        "positive_candidate_position": positive_position,
        "positive_present": positive_present,
        "expected_raw_output_path": rel_safe(raw_output_dir / "responses" / f"{rid}.json"),
        "evidence_class": CLASSIFICATION,
        "lineage": ";".join(
            [
                rel_safe(f6l.LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH),
                rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH),
                "f6g1_reconstructed_adapted_round_robin_union_20",
                "parser_v4_frozen_no_fuzzy_no_relevance_metrics",
            ]
        ),
        "request_contract_hash": "",
    }
    contract_payload = {
        key: row[key]
        for key in [
            "full_run_id",
            "task",
            "sample_id",
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


def build_manifest_rows(run_id: str, raw_output_dir: Path) -> list[dict[str, str]]:
    candidates = reranking_pool_rows_by_user()
    rows: list[dict[str, str]] = []
    index = 0
    for sample_row in full_user_rows():
        user_candidates = candidates.get(sample_row["user_id"], [])
        if not user_candidates:
            raise ValueError(f"missing reranking candidates for user {sample_row['user_id']}")
        for model_name in MODEL_ORDER:
            for strategy in STRATEGY_ORDER:
                index += 1
                rows.append(
                    manifest_row_for_request(
                        index=index,
                        run_id=run_id,
                        raw_output_dir=raw_output_dir,
                        sample_row=sample_row,
                        candidate_rows=user_candidates,
                        model_name=model_name,
                        strategy=strategy,
                    )
                )
    return rows


def full_freeze_payload(run_id: str, raw_output_dir: Path, manifest_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    coverage = positive_coverage()
    ranking_manifest_hash = f6l.safe_file_sha256(v4.FULL_RUN_MANIFEST_PATH)
    baseline_rows = ranking_baseline_audit_rows(write_outputs=False)
    return {
        "run_id": run_id,
        "phase_id": PHASE_ID,
        "full_run_id": FULL_RUN_ID,
        "description": RUN_DESCRIPTION,
        "classification": CLASSIFICATION,
        "request_count": len(manifest_rows),
        "user_count": len({row["user_id"] for row in manifest_rows}),
        "models": {model: f6l.MODEL_DIGESTS[model] for model in MODEL_ORDER},
        "model_digest_hash": model_digest_hash(),
        "strategies": STRATEGY_ORDER,
        "task": "reranking",
        "source_pool": rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH),
        "source_pool_file_sha256": f6l.safe_file_sha256(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH),
        "source_pool_content_hash": candidate_set_hash(),
        "excluded_superseded_artifacts": {
            "full_cohort_reranking_pool": {
                "path": rel_safe(f6l.LASTFM_FULL_COHORT_RERANKING_POOL_PATH),
                "sha256": f6l.safe_file_sha256(f6l.LASTFM_FULL_COHORT_RERANKING_POOL_PATH),
                "reason": "full_cohort_extension_1859_users_7436_requests_not_authorized",
            },
            "f6d3_planned_manifest": {
                "path": "results/tables/finalization_f6d3_lastfm_reranking_llm_manifest.csv",
                "sha256": f6l.safe_file_sha256(TABLES_DIR / "finalization_f6d3_lastfm_reranking_llm_manifest.csv"),
                "reason": "planned_not_executed_full_cohort_index_contract_superseded",
            },
            "ranking_v4_manifest": {
                "path": rel_safe(v4.FULL_RUN_MANIFEST_PATH),
                "sha256": ranking_manifest_hash,
                "reason": "ranking_output_contract_and_candidate_pool_not_used_as_reranking_pool",
            },
        },
        "candidate_pool_definition": "20-item per-user reranking pool reconstructed by local BPRMF/GRU4Rec/SASRec/LightGCN round-robin union, with original candidate order preserved and C01-C20 assigned by candidate_position.",
        "positive_coverage": coverage,
        "ranking_baseline": {
            "run_id": RANKING_BASELINE_RUN_ID,
            "decision_token": RANKING_BASELINE_DECISION_TOKEN,
            "freeze_hash": RANKING_BASELINE_FREEZE_HASH,
            "audit_status": "pass" if all(row["status"] == "pass" for row in baseline_rows) else "fail",
            "audit_path": rel_safe(RANKING_BASELINE_AUDIT_PATH),
            "primary_policy": "fail_strict_values_only_for_ranking_vs_reranking",
            "complete_case_policy": "secondary_only",
        },
        "historical_pre_existing_gap": {
            "test": "tests/test_phase_f6l_lastfm_execution_contracts.py",
            "decision_token": "f6l_lastfm_execution_contracts_blocked_by_missing_local_evidence",
            "classification": "historical_pre_existing_gap",
            "auto_blocking": False,
            "blocking_condition": "only_if_reranking_audit_requires_the_missing_evidence_for_cohort_candidate_pool_positive_or_provenance",
        },
        "parser_version": v4.PARSER_VERSION,
        "parser_source_hash": parser_source_hash(),
        "json_schema_hash": schema_hash(),
        "json_schema": v4.V4_JSON_SCHEMA,
        "output_contract": v4.OUTPUT_CONTRACT,
        "output_contract_hash": output_contract_hash(),
        "prompt_template_hashes": {strategy: prompt_template_hash(strategy) for strategy in STRATEGY_ORDER},
        "mapping_contract_hash": v4.mapping_contract_hash(),
        "manifest_rows_hash": f6l.sha256_text(canonical_json(list(manifest_rows))),
        "generation_parameters": GENERATION_PARAMETERS,
        "endpoint": f6l.OLLAMA_LOCAL_GENERATE_URL,
        "raw_output_dir": rel_safe(raw_output_dir),
        "canary": {
            "first_request_count": 40,
            "regression_users": CANARY_REGRESSION_USERS,
            "holdout_users": CANARY_HOLDOUT_USERS,
        },
        "circuit_breaker": {
            "stop_on_hard_contract_or_provenance_failure": True,
            "stop_on_hard_structural_failure": True,
            "duplicate_token_policy": {
                "classification": "duplicate_token_rate_is_an_observed_model_runtime_property",
                "no_retry_no_repair_no_dedup": True,
                "stop_on_consecutive_duplicate_token_structural_invalid": DUPLICATE_CONSECUTIVE_LIMIT,
                "closed_window_size_new_responses": DUPLICATE_WINDOW_SIZE,
                "stop_if_closed_window_rate_gt": DUPLICATE_WINDOW_RATE_LIMIT,
            },
        },
        "git": {
            "head": f6l.git_output(["rev-parse", "HEAD"]),
            "original_repo_status": f6l.git_output(["status", "--short", "original_repo"]),
            "staged_paths": f6l.git_output(["diff", "--cached", "--name-only"]),
        },
    }


def build_full_run(run_id: str | None = None, *, raw_output_dir: Path | None = None, write_outputs: bool = True) -> RerankingBuild:
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_dir = raw_output_dir or local_raw_output_dir(run_id)
    rows = build_manifest_rows(run_id, raw_output_dir)
    freeze = full_freeze_payload(run_id, raw_output_dir, rows)
    manifest_hash = f6l.sha256_text(canonical_json(freeze))
    manifest = {**freeze, "manifest_hash": manifest_hash}
    freeze_rows = [
        {
            "freeze_id": f"F6L-LASTFM-V4-RERANK-FREEZE-{index:02d}",
            "run_id": run_id,
            "artifact": artifact,
            "sha256": sha256,
            "status": "frozen",
            "evidence": evidence,
            "notes": notes,
        }
        for index, (artifact, sha256, evidence, notes) in enumerate(
            [
                ("source_pool_file", manifest["source_pool_file_sha256"], manifest["source_pool"], "Canonical operationalized 1000-user reranking pool."),
                ("source_pool_content", manifest["source_pool_content_hash"], manifest["source_pool"], "Canonical content hash over user/candidate/source/positive fields."),
                ("parser_v4", manifest["parser_source_hash"], "inspect.getsource(parse_v4_response)", "Must equal accepted V4 parser hash."),
                ("json_schema", manifest["json_schema_hash"], "V4_JSON_SCHEMA", "Same opaque-token JSON schema as V4 ranking."),
                ("output_contract", manifest["output_contract_hash"], "OUTPUT_CONTRACT_RULES", "Same strict C01-C20 top-10 output contract."),
                ("prompt_base_openp5_reranking", manifest["prompt_template_hashes"][STRATEGY_ORDER[0]], prompt_template_name(STRATEGY_ORDER[0]), "Base/OpenP5-derived reranking prompt template hash."),
                ("prompt_llmrank_recency_reranking", manifest["prompt_template_hashes"][STRATEGY_ORDER[1]], prompt_template_name(STRATEGY_ORDER[1]), "LLMRank recency-focused reranking prompt template hash."),
                ("mapping_contract", manifest["mapping_contract_hash"], "TOKEN_LIST C01-C20", "Opaque token-to-position contract hash."),
                ("model_digests", manifest["model_digest_hash"], f6l.OLLAMA_LOCAL_TAGS_URL, "Expected local Ollama model digest hash."),
                ("manifest_rows", manifest["manifest_rows_hash"], rel_safe(MANIFEST_PATH), "Full 4000-request reranking manifest rows hash."),
                ("complete_manifest", manifest["manifest_hash"], rel_safe(JSON_MANIFEST_PATH), "Complete deterministic prelaunch manifest hash."),
            ],
            start=1,
        )
    ]
    if write_outputs:
        f6l.write_csv(MANIFEST_PATH, rows, MANIFEST_FIELDS)
        f6l.write_csv(FREEZE_PATH, freeze_rows, FREEZE_FIELDS)
        f6l.write_json(JSON_MANIFEST_PATH, manifest)
    return RerankingBuild(run_id, raw_output_dir, rows, manifest, freeze_rows)


def build_identity_check(run_id: str, raw_output_dir: Path) -> tuple[bool, str, str]:
    first = build_full_run(run_id=run_id, raw_output_dir=raw_output_dir, write_outputs=False)
    second = build_full_run(run_id=run_id, raw_output_dir=raw_output_dir, write_outputs=False)
    first_bytes = canonical_json({"rows": first.manifest_rows, "manifest": first.manifest, "freeze": first.freeze_rows})
    second_bytes = canonical_json({"rows": second.manifest_rows, "manifest": second.manifest, "freeze": second.freeze_rows})
    return first_bytes == second_bytes, f6l.sha256_text(first_bytes), f6l.sha256_text(second_bytes)


def run_focused_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        "results/logs/phase_f6l_lastfm_v4_reranking_pytest",
        "-q",
        "tests/test_phase_f6l_lastfm_v4_reranking_full_run.py",
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


def _pass(status: bool) -> str:
    return "pass" if status else "fail"


def validate_preflight(
    build: RerankingBuild,
    *,
    raw_output_dir: Path,
    resume: bool,
    allow_api_calls: bool,
    local_only: bool,
    endpoint: str,
    run_tests: bool,
    check_runtime: bool,
    max_prompt_checks: int | None = None,
    check_identity: bool = True,
) -> tuple[list[dict[str, str]], dict[str, Any], f6l.LocalOllamaState | None]:
    rows: list[dict[str, str]] = []
    test_result: dict[str, Any] = {}
    ollama_state: f6l.LocalOllamaState | None = None

    def add(name: str, passed: bool, observed: str, expected: str, evidence: str, notes: str, severity: str = "error") -> None:
        rows.append(
            {
                "check_id": f"F6L-LASTFM-V4-RERANK-PREFLIGHT-{len(rows) + 1:02d}",
                "check_name": name,
                "status": _pass(passed),
                "observed": observed,
                "expected": expected,
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    pool_rows = f6l.read_csv_rows(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH)
    by_user = reranking_pool_rows_by_user()
    readiness = f6l.read_csv_rows(f6l.LASTFM_RERANKING_READINESS_PATH)
    readiness_by_scope = {row["scope"]: row for row in readiness}
    full_pool_rows = f6l.read_csv_rows(f6l.LASTFM_FULL_COHORT_RERANKING_POOL_PATH)
    ranking_users = {row["user_id"] for row in f6l.read_csv_rows(v4.FULL_RUN_MANIFEST_PATH)}
    rerank_users = set(by_user)
    ranking_candidate_hash = f6l.safe_file_sha256(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH)
    rerank_pool_hash = f6l.safe_file_sha256(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH)
    baseline_audit = ranking_baseline_audit_rows(write_outputs=True)
    baseline_ok = all(row["status"] == "pass" for row in baseline_audit)
    positions_ok = all([row["candidate_position"] for row in pool_rows]) and all(
        sorted(int(row["candidate_position"]) for row in user_rows) == list(range(1, 21))
        for user_rows in by_user.values()
    )
    distinct_ok = all(len(user_rows) == 20 and len({row["candidate_item_id"] for row in user_rows}) == 20 for user_rows in by_user.values())
    scopes = {row.get("cohort_scope", "") for row in pool_rows}
    sample_ids = {row.get("sample_id", "") for row in pool_rows}
    evidence_classes = {row.get("evidence_class", "") for row in pool_rows}
    policies = {row.get("construction_policy", "") for row in pool_rows}
    lineage_ok = all("phase_f6g1_pool_item_provenance.csv" in row.get("lineage", "") for row in pool_rows)
    positive_counts = {
        user_id: sum(1 for row in user_rows if str(row.get("is_positive", "")).lower() == "true")
        for user_id, user_rows in by_user.items()
    }
    positive_present = sum(1 for count in positive_counts.values() if count == 1)
    positive_position_bad = [
        f"{row['user_id']}:{row['candidate_position']}"
        for row in pool_rows
        if str(row.get("is_positive", "")).lower() == "true"
        and str(row.get("positive_position", "")) != str(row.get("candidate_position", ""))
    ]
    missing_titles = [f"{row['user_id']}:{row['candidate_position']}" for row in pool_rows if not row.get("candidate_item_title")]
    per_user_manifest = Counter(row["user_id"] for row in build.manifest_rows)
    candidate_hashes_by_user: dict[str, set[str]] = defaultdict(set)
    mapping_hashes_by_user: dict[str, set[str]] = defaultdict(set)
    mapping_ok = True
    for row in build.manifest_rows:
        candidate_hashes_by_user[row["user_id"]].add(row["candidate_list_hash"])
        mapping_hashes_by_user[row["user_id"]].add(row["candidate_token_mapping_hash"])
        try:
            mapping_ok = mapping_ok and set(json.loads(row["candidate_token_mapping"])) == set(TOKEN_LIST)
        except Exception:
            mapping_ok = False
    canary_users = [row["user_id"] for row in build.manifest_rows[:40:4]]
    prompt_hash_failures: list[str] = []
    forbidden_prompt_failures: list[str] = []
    sample = sample_rows_by_user()
    candidates = reranking_pool_rows_by_user()
    prompt_rows = build.manifest_rows if max_prompt_checks is None else build.manifest_rows[:max_prompt_checks]
    for row in prompt_rows:
        try:
            prompt = render_prompt({**sample[row["user_id"]], "_subset": row["subset"]}, candidates[row["user_id"]], row["prompting_strategy"])
            if f6l.sha256_text(prompt) != row["rendered_prompt_hash"]:
                prompt_hash_failures.append(f"{row['request_id']}:hash_mismatch")
            lowered = prompt.casefold()
            forbidden = [token for token in FORBIDDEN_PROMPT_TOKENS if token in lowered]
            if forbidden:
                forbidden_prompt_failures.append(f"{row['request_id']}:{';'.join(forbidden)}")
        except Exception as exc:
            prompt_hash_failures.append(f"{row['request_id']}:{type(exc).__name__}")
            forbidden_prompt_failures.append(f"{row['request_id']}:{type(exc).__name__}")
    if check_identity:
        identity_ok, identity_first, identity_second = build_identity_check(build.run_id, raw_output_dir)
    else:
        identity_ok, identity_first, identity_second = True, "skipped_by_unit_test", "skipped_by_unit_test"
    raw_exists = raw_output_dir.exists()
    raw_ok = ((not resume) and (not raw_exists)) or (resume and raw_exists and (raw_output_dir / "run_manifest.json").exists())

    add("canonical_source_operationalized_pool", f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH.exists(), rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "existing canonical operationalized reranking pool", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "Canonical source must be explicit before inference.")
    add("superseded_full_cohort_excluded", len(full_pool_rows) == 37180 and len({row['user_id'] for row in full_pool_rows}) == 1859, f"rows={len(full_pool_rows)};users={len({row['user_id'] for row in full_pool_rows})};excluded=true", "37180 rows / 1859 users excluded", rel_safe(f6l.LASTFM_FULL_COHORT_RERANKING_POOL_PATH), "Full-cohort extension is not consumed.")
    op_ready = readiness_by_scope.get("operationalized_sample", {})
    add("readiness_operationalized_authorized", op_ready.get("status") == "ready_after_explicit_authorization" and op_ready.get("future_request_count") == "4000", f"status={op_ready.get('status','')};future={op_ready.get('future_request_count','')}", "ready_after_explicit_authorization;future=4000", rel_safe(f6l.LASTFM_RERANKING_READINESS_PATH), "User explicitly authorized the operationalized 1000-user reranking run.")
    paper_exact = readiness_by_scope.get("paper_exact", {})
    add("paper_exact_blocked_not_used", paper_exact.get("status", "").startswith("blocked"), paper_exact.get("status", ""), "blocked_missing_original_specification", rel_safe(f6l.LASTFM_RERANKING_READINESS_PATH), "No paper-exact LastFM reranking claim is made.")
    add("pool_rows_20000", len(pool_rows) == 20000, str(len(pool_rows)), "20000", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "1000 users x 20 reranking candidates.")
    add("pool_users_1000", len(by_user) == 1000, str(len(by_user)), "1000", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "Only the operationalized sample is authorized.")
    add("pool_twenty_candidates_per_user", set(len(rows_for_user) for rows_for_user in by_user.values()) == {20}, str(sorted(set(len(rows_for_user) for rows_for_user in by_user.values()))), "[20]", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "Every user has exactly 20 candidates.")
    add("pool_distinct_candidates_per_user", distinct_ok, "pass" if distinct_ok else "fail", "20 unique item ids per user", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "No candidate deduplication is performed at prompt time.")
    add("pool_positions_1_to_20", positions_ok, "pass" if positions_ok else "fail", "positions 1..20 per user", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "C01-C20 mapping follows candidate_position.")
    add("pool_scope_operationalized_only", scopes == {"operationalized"}, ";".join(sorted(scopes)), "operationalized", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "No full-cohort rows enter the manifest.")
    add("pool_sample_id_operationalized_1000_only", sample_ids == {"phase_f6l_lastfm_operationalized_1000"}, ";".join(sorted(sample_ids)), "phase_f6l_lastfm_operationalized_1000", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "Sample id is frozen.")
    add("pool_evidence_class_expected", evidence_classes == {CLASSIFICATION}, ";".join(sorted(evidence_classes)), CLASSIFICATION, rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "Evidence class is not elevated beyond operationalization.")
    add("pool_lineage_expected", lineage_ok, "phase_f6g1_pool_item_provenance.csv" if lineage_ok else "missing", "phase_f6g1_pool_item_provenance.csv", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "Reranking pool lineage must point to reconstructed source-model provenance.")
    add("pool_construction_policy_expected", policies == {"phase_f6g1_reconstructed_adapted_round_robin_union_20"}, ";".join(sorted(policies)), "phase_f6g1_reconstructed_adapted_round_robin_union_20", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "Candidate pool definition is frozen.")
    add("same_1000_users_as_ranking_v4_manifest", rerank_users == ranking_users and len(rerank_users) == 1000, f"rerank={len(rerank_users)};ranking_v4={len(ranking_users)};same={rerank_users == ranking_users}", "same 1000 users", rel_safe(v4.FULL_RUN_MANIFEST_PATH), "The cohort is shared, but candidates and raw outputs are not reused.")
    add("ranking_baseline_frozen_and_candidate_pool_not_used", baseline_ok and ranking_candidate_hash and rerank_pool_hash and ranking_candidate_hash != rerank_pool_hash, f"baseline_ok={baseline_ok};ranking_candidates={ranking_candidate_hash};reranking_pool={rerank_pool_hash}", "baseline_ok=true;different source hashes", f"{rel_safe(RANKING_BASELINE_AUDIT_PATH)};{rel_safe(f6l.LASTFM_OPERATIONALIZED_CANDIDATES_PATH)}", "Ranking baseline is byte-frozen and ranking V4 candidate rows are excluded from reranking input.")
    add("positive_coverage_reported", positive_present == 314 and len(by_user) == 1000, f"positive_present={positive_present};positive_absent={len(by_user)-positive_present};rate={format_float(positive_present/len(by_user) if by_user else 0.0)}", "reported 314/1000, not required as 1000/1000", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "Positive absence remains in the operational denominator.", severity="info")
    add("at_most_one_positive_per_user", all(count in {0, 1} for count in positive_counts.values()), str(sorted(set(positive_counts.values()))), "0 or 1", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "Multiple positives would make utility mapping ambiguous.")
    add("positive_position_consistent_when_present", not positive_position_bad, ";".join(positive_position_bad[:5]) if positive_position_bad else "none", "none", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "Positive metadata is used only after parsing.")
    add("candidate_titles_present", not missing_titles, ";".join(missing_titles[:5]) if missing_titles else "none", "none", rel_safe(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "Prompt rendering needs candidate titles.")
    add("manifest_rows_4000", len(build.manifest_rows) == REQUEST_COUNT, str(len(build.manifest_rows)), str(REQUEST_COUNT), rel_safe(MANIFEST_PATH), "1000 users x 2 models x 2 strategies.")
    add("manifest_unique_request_ids_4000", len({row['request_id'] for row in build.manifest_rows}) == REQUEST_COUNT, str(len({row['request_id'] for row in build.manifest_rows})), str(REQUEST_COUNT), rel_safe(MANIFEST_PATH), "Request ids are deterministic and unique.")
    add("manifest_four_requests_per_user", len(per_user_manifest) == USER_COUNT and set(per_user_manifest.values()) == {4}, f"users={len(per_user_manifest)};counts={sorted(set(per_user_manifest.values()))}", "1000 users each with 4 requests", rel_safe(MANIFEST_PATH), "Both model x strategy cells share users.")
    add("only_authorized_models", {row["model_name"] for row in build.manifest_rows} == set(MODEL_ORDER), ";".join(sorted({row["model_name"] for row in build.manifest_rows})), ";".join(MODEL_ORDER), rel_safe(MANIFEST_PATH), "Only local Qwen2/Mistral cells are authorized.")
    add("only_authorized_reranking_strategies", {row["prompting_strategy"] for row in build.manifest_rows} == set(STRATEGY_ORDER), ";".join(sorted({row["prompting_strategy"] for row in build.manifest_rows})), ";".join(STRATEGY_ORDER), rel_safe(MANIFEST_PATH), "Only base/OpenP5-derived and LLMRank recency-focused reranking are authorized.")
    add("no_llmsrec_or_extra_strategy", not any("llmsrec" in row["prompting_strategy"].casefold() for row in build.manifest_rows), "none", "none", rel_safe(MANIFEST_PATH), "LLMSRec remains excluded because the demonstration contract is not recovered.")
    add("candidate_order_shared_between_cells", all(len(value) == 1 for value in candidate_hashes_by_user.values()) and len(candidate_hashes_by_user) == USER_COUNT, f"users={len(candidate_hashes_by_user)}", "one candidate_list_hash per user", rel_safe(MANIFEST_PATH), "Candidate order is identical across four cells for each user.")
    add("token_mapping_c01_c20_stable", mapping_ok and all(len(value) == 1 for value in mapping_hashes_by_user.values()), "pass" if mapping_ok else "fail", "C01-C20 mapping stable per user", rel_safe(MANIFEST_PATH), "Opaque tokens are stable across model/strategy cells.")
    add("canary_first_40_internal", canary_users == CANARY_USERS and len(build.manifest_rows[:40]) == 40, ";".join(canary_users), ";".join(CANARY_USERS), rel_safe(MANIFEST_PATH), "First 40 rows are internal canary rows, not extra requests.")
    add("prompt_hashes_rebuild", not prompt_hash_failures, ";".join(prompt_hash_failures[:3]) if prompt_hash_failures else "none", "none", "render_prompt", "Every prompt hash must rebuild from frozen source.")
    add("prompt_no_forbidden_tokens", not forbidden_prompt_failures, ";".join(forbidden_prompt_failures[:3]) if forbidden_prompt_failures else "none", "no positive/metric labels in prompt", "render_prompt", "Prompt is blind to positive and metric labels.")
    add("parser_v4_hash_expected", parser_source_hash() == EXPECTED_V4_PARSER_HASH, parser_source_hash(), EXPECTED_V4_PARSER_HASH, "inspect.getsource(parse_v4_response)", "Parser is byte-identical to accepted V4.")
    add("schema_hash_expected", schema_hash() == EXPECTED_V4_SCHEMA_HASH, schema_hash(), EXPECTED_V4_SCHEMA_HASH, "V4_JSON_SCHEMA", "JSON schema is unchanged from accepted V4.")
    add("output_contract_hash_expected", output_contract_hash() == EXPECTED_V4_OUTPUT_CONTRACT_HASH, output_contract_hash(), EXPECTED_V4_OUTPUT_CONTRACT_HASH, "OUTPUT_CONTRACT_RULES", "Strict output contract is unchanged.")
    add("generation_parameters_frozen", GENERATION_PARAMETERS == v4.GENERATION_PARAMETERS, canonical_json(GENERATION_PARAMETERS), canonical_json(v4.GENERATION_PARAMETERS), "GENERATION_PARAMETERS", "Temperature, seed, stream, schema, and retry policy are frozen.")
    add("model_digests_expected_in_manifest", all(row["model_digest"] == f6l.MODEL_DIGESTS[row["model_name"]] for row in build.manifest_rows), "pass", "manifest rows use expected model digests", rel_safe(MANIFEST_PATH), "No unexpected model digest is authorized.")
    add("raw_output_dir_append_only_policy", raw_ok, f"exists={raw_exists};resume={resume};dir={rel_safe(raw_output_dir)}", "new nonexistent dir, or explicit resume with run_manifest.json", "CLI args", "Raw outputs are append-only.")

    test_ok = True
    if run_tests:
        test_result = run_focused_tests()
        test_ok = test_result["exit_code"] == 0
    if check_runtime:
        protected = f6l.hash_protected_inputs()
        staged = f6l.git_output(["diff", "--cached", "--name-only"])
        original_status = f6l.git_output(["status", "--short", "original_repo"])
    else:
        protected = {"status": "pass"}
        staged = ""
        original_status = ""
    endpoint_ok = local_only and endpoint in {f6l.OLLAMA_LOCAL_GENERATE_URL, "http://127.0.0.1:11434/api/generate", "http://localhost:11434/api/generate"}
    runtime_ok = allow_api_calls and endpoint_ok and original_status == "" and staged == "" and protected.get("status") == "pass" and test_ok
    ollama_observed = "not_checked"
    if check_runtime and all(row["status"] == "pass" for row in rows) and runtime_ok:
        ollama_state = f6l.start_local_ollama_if_needed(build.run_id)
        digest_ok = ollama_state.service_available and all(
            ollama_state.model_digests.get(model, "").lower() == f6l.MODEL_DIGESTS[model].lower()
            for model in MODEL_ORDER
        )
        runtime_ok = runtime_ok and digest_ok
        ollama_observed = canonical_json({model: ollama_state.model_digests.get(model, "") for model in MODEL_ORDER})
    elif not check_runtime:
        ollama_observed = "skipped_by_unit_test"
    observed_guard = canonical_json(
        {
            "allow_api_calls": allow_api_calls,
            "endpoint_ok": endpoint_ok,
            "original_repo_status": original_status or "none",
            "staged_paths": staged or "none",
            "protected_inputs": protected.get("status", ""),
            "focused_tests": "pass" if test_ok else f"fail:{test_result.get('exit_code')}",
            "ollama_digests": ollama_observed,
        }
    )
    add("runtime_repository_and_test_guardrails", runtime_ok, observed_guard, "allow=true;loopback=true;original_repo_clean;staging_empty;protected_pass;tests_pass;digests_match", "CLI args;git;protected hashes;pytest;Ollama tags", "Final gate before local inference.")

    f6l.write_csv(PREFLIGHT_PATH, rows, PREFLIGHT_FIELDS)
    return rows, test_result, ollama_state


def request_payload_for_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "request_id": row["request_id"],
        "full_run_id": row["full_run_id"],
        "run_id": row["run_id"],
        "task": row["task"],
        "sample_id": row["sample_id"],
        "subset": row["subset"],
        "user_id": row["user_id"],
        "model_name": row["model_name"],
        "model_digest": row["model_digest"],
        "prompting_strategy": row["prompting_strategy"],
        "candidate_pool_hash": row["candidate_pool_hash"],
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
    result: f6l.LocalOllamaRequestResult,
    parse: v4.ParserV4Result,
    raw_json_path: Path,
    *,
    attempt_index: int,
    is_final_attempt: bool,
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


def parse_result_row(row: Mapping[str, str], raw_record: Mapping[str, Any]) -> dict[str, str]:
    parse = raw_record.get("parser_output", {})
    hash_coherence = (
        raw_record.get("rendered_prompt_hash") == row["rendered_prompt_hash"]
        and raw_record.get("candidate_list_hash") == row["candidate_list_hash"]
        and raw_record.get("candidate_token_mapping_hash") == row["candidate_token_mapping_hash"]
        and raw_record.get("request_contract_hash") == row["request_contract_hash"]
    )
    ranked = [str(item) for item in parse.get("ranked_item_ids", [])]
    positive_item = row.get("positive_item_id", "")
    positive_rank = str(ranked.index(positive_item) + 1) if positive_item and positive_item in ranked else ""
    return {
        "request_id": row["request_id"],
        "full_run_id": row["full_run_id"],
        "run_id": row["run_id"],
        "task": "reranking",
        "sample_id": row["sample_id"],
        "subset": row["subset"],
        "user_id": row["user_id"],
        "model_name": row["model_name"],
        "model_digest": row["model_digest"],
        "prompting_strategy": row["prompting_strategy"],
        "parser_version": v4.PARSER_VERSION,
        "parser_source_hash": row["parser_source_hash"],
        "output_contract": row["output_contract"],
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
        "positive_item_id": row.get("positive_item_id", ""),
        "positive_candidate_position": row.get("positive_candidate_position", ""),
        "positive_present": row.get("positive_present", ""),
        "positive_rank": positive_rank,
        "request_contract_hash": row["request_contract_hash"],
        "hash_coherence": "true" if hash_coherence else "false",
        "evidence_class": CLASSIFICATION,
        "notes": "V4 reranking parser uses opaque tokens only; positive metadata is applied only after parsing.",
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
            "full_run_id": FULL_RUN_ID,
            "run_id": row["run_id"],
            "task": "reranking",
            "sample_id": row["sample_id"],
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
            "parser_version": v4.PARSER_VERSION,
            "evidence_class": CLASSIFICATION,
            "notes": "opaque token mapped deterministically to the frozen reranking candidate list",
        }
        for rank, (token, item_id, title) in enumerate(zip(tokens, item_ids, titles), start=1)
    ]


def execution_row(row: Mapping[str, str], raw_record: Mapping[str, Any]) -> dict[str, str]:
    backend = raw_record.get("raw_backend_response", {})
    return {
        "request_id": row["request_id"],
        "full_run_id": FULL_RUN_ID,
        "run_id": row["run_id"],
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


def structural_counts(parse_rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    return v4.structural_counts(parse_rows)


def is_duplicate_token_invalid(parse_row: Mapping[str, str]) -> bool:
    return v4.is_duplicate_token_invalid(parse_row)


def is_hard_structural_invalid(parse_row: Mapping[str, str]) -> bool:
    return v4.is_hard_structural_invalid(parse_row)


def duplicate_token_invalid_response_count(parse_rows: Sequence[Mapping[str, str]]) -> int:
    return v4.duplicate_token_invalid_response_count(parse_rows)


def duplicate_token_response_rate(parse_rows: Sequence[Mapping[str, str]]) -> float:
    return v4.duplicate_token_response_rate(parse_rows)


def current_duplicate_window_rate(parse_rows: Sequence[Mapping[str, str]]) -> float:
    return v4.current_duplicate_window_rate(parse_rows)


def closed_duplicate_window_status(parse_rows: Sequence[Mapping[str, str]]) -> tuple[int, float, str]:
    return v4.closed_duplicate_window_status(parse_rows)


def trailing_duplicate_token_invalid_count(parse_rows: Sequence[Mapping[str, str]], manifest_rows: Sequence[Mapping[str, str]]) -> int:
    return v4.trailing_duplicate_token_invalid_count(parse_rows, manifest_rows)


def write_state(
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
    residual = REQUEST_COUNT - completed
    elapsed_hours = max((time.perf_counter() - started_clock) / 3600.0, 1e-9)
    throughput = completed / elapsed_hours if completed else 0.0
    eta = residual / throughput * 3600.0 if throughput and residual else 0.0
    row = {
        "run_id": run_id,
        "stage": stage,
        "decision_token": decision_token,
        "planned": str(REQUEST_COUNT),
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
        "percent_complete": format_float(completed / REQUEST_COUNT),
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
    f6l.write_csv(STATE_PATH, [row], STATE_FIELDS)


def canary_rows_from_parse(run_id: str, parse_rows: Sequence[Mapping[str, str]], manifest_rows: Sequence[Mapping[str, str]], raw_output_dir: Path) -> list[dict[str, str]]:
    canary_ids = [row["request_id"] for row in manifest_rows[:40]]
    parse_by_id = {row["request_id"]: row for row in parse_rows}
    canary_parse = [parse_by_id[request_id] for request_id in canary_ids if request_id in parse_by_id]
    hard_failures = sum(1 for row in canary_parse if row["hash_coherence"] != "true" or is_hard_structural_invalid(row))
    duplicate_failures = duplicate_token_invalid_response_count(canary_parse)
    criteria = [
        ("completed_first_40", len(canary_parse), 40, rel_safe(PARSE_RESULTS_PATH), "Internal canary consists of the first 40 manifest requests."),
        ("hard_structural_or_hash_failures", hard_failures, 0, rel_safe(PARSE_RESULTS_PATH), "Canary blocks only hard structural/provenance failures; duplicate-token invalidity is tracked separately."),
        ("duplicate_token_invalids_observed", duplicate_failures, "reported", rel_safe(PARSE_RESULTS_PATH), "Duplicate-token invalidity is retained fail-strict and does not trigger canary stop unless systemic."),
        ("residual_after_canary", REQUEST_COUNT - len(canary_parse), 3960, rel_safe(STATE_PATH), "Canary consumes no extra requests beyond the 4000 manifest."),
    ]
    rows = []
    for name, observed, expected, evidence, notes in criteria:
        status = "pass" if expected == "reported" or str(observed) == str(expected) else "fail"
        rows.append(
            {
                "run_id": run_id,
                "criterion": name,
                "observed": str(observed),
                "expected": str(expected),
                "status": status,
                "evidence": evidence,
                "notes": notes,
            }
        )
    return rows


def build_duplicate_window_rows(
    run_id: str,
    parse_rows: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, str]],
    resume_start_completed: int,
) -> list[dict[str, str]]:
    rows = v4.build_duplicate_window_rows(run_id, parse_rows, manifest_rows, resume_start_completed)
    for row in rows:
        row["window_id"] = stable_hash("F6L-LASTFM-V4-RERANK-DUP-WINDOW", run_id, row["window_index"], row["window_start_new_response"], row["window_end_new_response"])[:24]
        row["evidence_class"] = CLASSIFICATION
        row["notes"] = "Reranking duplicate-token windows use the final F6L V4 amended policy."
    return rows


def request_state_rows(
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
                "full_run_id": FULL_RUN_ID,
                "run_id": row["run_id"],
                "subset": row["subset"],
                "user_id": row["user_id"],
                "model_name": row["model_name"],
                "prompting_strategy": row["prompting_strategy"],
                "planned": "true",
                "state": state,
                "strict_valid": parse.get("strict_valid", "false") if parse else "false",
                "failure_reason": parse.get("failure_reason", "") if parse else "",
                "attempt_count": str(attempt_counts.get(row["request_id"], 0)),
                "retry_count": str(max(0, attempt_counts.get(row["request_id"], 0) - 1)),
                "raw_json_path": str(record.get("raw_json_path", "")),
                "request_contract_hash": row["request_contract_hash"],
                "notes": "Strict-valid and structural-invalid final rows are never regenerated on resume.",
            }
        )
    return rows


def compute_metrics(
    run_id: str,
    parse_rows: Sequence[Mapping[str, str]],
    recommendation_rows: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, str]],
    attempt_counts: Mapping[str, int],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    popularity = f6l.train_popularity_counts()
    tag_sets = f6l.item_tag_sets()
    histories = f6l.sample_history_items()
    candidates = reranking_pool_rows_by_user()
    candidate_universe = {row["candidate_item_id"] for row in f6l.read_csv_rows(f6l.LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH)}
    popularity_values = list(popularity.values())
    long_tail_cutoff = f6l.percentile([float(value) for value in popularity_values], 0.80) if popularity_values else 0.0
    total_train = sum(popularity.values())
    parse_by_request = {row["request_id"]: row for row in parse_rows}
    by_request_recs: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in recommendation_rows:
        by_request_recs[row["request_id"]].append(row)
    for rows in by_request_recs.values():
        rows.sort(key=lambda item: int(item["rank"]))
    metrics: list[dict[str, str]] = []
    beyond: list[dict[str, str]] = []

    def add_metric(metric: str, value: float, valid_rows: int, total_rows: int, strict_rows: int, positive_rows: int, model: str, strategy: str, notes: str, *, k: str = "5") -> None:
        metrics.append(
            {
                "metric_id": stable_hash("F6L-LASTFM-V4-RERANK-METRIC", run_id, model, strategy, metric, k)[:24],
                "run_id": run_id,
                "dataset": "LastFM",
                "task": "reranking",
                "model_name": model,
                "model_digest": f6l.MODEL_DIGESTS[model],
                "prompting_strategy": strategy,
                "evidence_class": CLASSIFICATION,
                "metric": metric,
                "k": k,
                "value": format_float(value),
                "valid_rows": str(valid_rows),
                "total_rows": str(total_rows),
                "strict_valid_rows": str(strict_rows),
                "positive_present_rows": str(positive_rows),
                "source_file": rel_safe(PARSE_RESULTS_PATH),
                "notes": notes,
            }
        )

    def add_beyond(metric: str, value: float, numerator: float, denominator: float, model: str, strategy: str, notes: str, support: str = "supported", k: str = "5") -> None:
        beyond.append(
            {
                "metric_id": stable_hash("F6L-LASTFM-V4-RERANK-BEYOND", run_id, model, strategy, metric, k)[:24],
                "run_id": run_id,
                "dataset": "LastFM",
                "task": "reranking",
                "model_name": model,
                "model_digest": f6l.MODEL_DIGESTS[model],
                "prompting_strategy": strategy,
                "evidence_class": CLASSIFICATION,
                "metric": metric,
                "k": k,
                "value": format_float(value),
                "numerator": format_float(numerator),
                "denominator": format_float(denominator),
                "support_status": support,
                "source_file": rel_safe(RECOMMENDATIONS_PATH),
                "notes": notes,
            }
        )

    for model in MODEL_ORDER:
        for strategy in STRATEGY_ORDER:
            request_ids = [row["request_id"] for row in manifest_rows if row["model_name"] == model and row["prompting_strategy"] == strategy]
            rows = [parse_by_request[request_id] for request_id in request_ids if request_id in parse_by_request]
            total = len(request_ids)
            strict_rows = [row for row in rows if row["strict_valid"] == "true"]
            positive_rows = [row for row in rows if row.get("positive_present") == "true"]
            hits: list[float] = []
            ndcgs: list[float] = []
            ranks: list[float] = []
            conditional_hits: list[float] = []
            conditional_ndcgs: list[float] = []
            for row in rows:
                rank = int(row["positive_rank"]) if row.get("positive_rank") else None
                strict = row["strict_valid"] == "true"
                positive = row.get("positive_present") == "true"
                hits.append(1.0 if strict and positive and rank is not None and rank <= 5 else 0.0)
                ndcgs.append(1.0 / math.log2(rank + 1) if strict and positive and rank is not None and rank <= 5 else 0.0)
                ranks.append(float(rank) if strict and positive and rank is not None else 20.0)
                if positive:
                    conditional_hits.append(1.0 if strict and rank is not None and rank <= 5 else 0.0)
                    conditional_ndcgs.append(1.0 / math.log2(rank + 1) if strict and rank is not None and rank <= 5 else 0.0)
            structural_invalid = len(rows) - len(strict_rows)
            duplicate_invalid = duplicate_token_invalid_response_count(rows)
            malformed = sum(int(row["malformed_json_count"] or 0) for row in rows)
            unknown = sum(int(row["unknown_token_count"] or 0) for row in rows)
            duplicate_tokens = sum(int(row["duplicate_token_count"] or 0) for row in rows)
            output_total = sum(int(row["output_count"] or 0) for row in rows)
            retry_count = sum(max(0, int(attempt_counts.get(request_id, 0)) - 1) for request_id in request_ids)
            add_metric("HR@5_fail_strict", f6l.safe_mean(hits), len(rows), total, len(strict_rows), len(positive_rows), model, strategy, "Fail-strict denominator includes all 1000 users; positive-absent users count as misses.")
            add_metric("NDCG@5_fail_strict", f6l.safe_mean(ndcgs), len(rows), total, len(strict_rows), len(positive_rows), model, strategy, "Fail-strict denominator includes all 1000 users; positive-absent users count as zero.")
            add_metric("mean_positive_rank_fail_strict_invalid_as_20", f6l.safe_mean(ranks), len(rows), total, len(strict_rows), len(positive_rows), model, strategy, "Missing, absent, or invalid positive ranks count as 20.")
            add_metric("HR@5_positive_present_only", f6l.safe_mean(conditional_hits), len(positive_rows), total, len(strict_rows), len(positive_rows), model, strategy, "Diagnostic conditional utility over positive-present users only.")
            add_metric("NDCG@5_positive_present_only", f6l.safe_mean(conditional_ndcgs), len(positive_rows), total, len(strict_rows), len(positive_rows), model, strategy, "Diagnostic conditional utility over positive-present users only.")
            add_metric("positive_present_rate", len(positive_rows) / total if total else 0.0, len(rows), total, len(strict_rows), len(positive_rows), model, strategy, "Coverage of positive item inside the reranking candidate pool.", k="")
            add_metric("strict_valid_rate", len(strict_rows) / total if total else 0.0, len(rows), total, len(strict_rows), len(positive_rows), model, strategy, "Strict V4 validity over planned requests.", k="")
            add_metric("structural_invalid_rate", structural_invalid / total if total else 0.0, len(rows), total, len(strict_rows), len(positive_rows), model, strategy, "All structural-invalid final outputs over planned requests.", k="")
            add_metric("duplicate_token_rate", duplicate_invalid / total if total else 0.0, len(rows), total, len(strict_rows), len(positive_rows), model, strategy, "Duplicate-token structural-invalid outputs over planned requests.", k="")
            add_metric("hallucination_rate", unknown / output_total if output_total else 0.0, len(rows), total, len(strict_rows), len(positive_rows), model, strategy, "Unknown C-token outputs over parser-observed outputs.", k="")
            add_metric("malformed_json_rate", malformed / total if total else 0.0, len(rows), total, len(strict_rows), len(positive_rows), model, strategy, "Malformed JSON final outputs over planned requests.", k="")
            add_metric("retry_rate", retry_count / total if total else 0.0, len(rows), total, len(strict_rows), len(positive_rows), model, strategy, "Transport retries only; no scientific retry.", k="")

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
                user_id = parse_by_request.get(request_id, {}).get("user_id", "")
                history_counter: Counter[str] = Counter()
                rec_counter: Counter[str] = Counter()
                for item in histories.get(user_id, []):
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
            coverage_note = f"Computed on strict-valid top-5 reranking outputs only; valid output coverage={len(strict_rows)}/{total}."
            add_beyond("novelty", f6l.safe_mean(novelty_values), sum(novelty_values), denominator, model, strategy, coverage_note)
            add_beyond("APLT_long_tail", long_tail_hits / denominator if denominator else 0.0, long_tail_hits, denominator, model, strategy, coverage_note + f" Long-tail cutoff={format_float(long_tail_cutoff)}.")
            add_beyond("average_popularity", pop_sum / denominator if denominator else 0.0, pop_sum, denominator, model, strategy, coverage_note)
            add_beyond("popularity_bias", pop_sum / denominator if denominator else 0.0, pop_sum, denominator, model, strategy, coverage_note)
            add_beyond("catalog_coverage", len(set(top5_items)) / len(candidate_universe) if candidate_universe else 0.0, len(set(top5_items)), len(candidate_universe), model, strategy, coverage_note)
            add_beyond("ILD_diversity", f6l.safe_mean(ild_values), sum(ild_values), len(ild_values), model, strategy, coverage_note)
            add_beyond("calibration", f6l.safe_mean(calibration_values), sum(calibration_values), len(calibration_values), model, strategy, coverage_note, support="supported" if calibration_values else "unsupported_missing_tags")

    lookup = {(row["model_name"], row["prompting_strategy"], row["metric"]): float(row["value"] or 0.0) for row in metrics + beyond}
    comparisons: list[dict[str, str]] = []
    coverage = positive_coverage()
    for model in MODEL_ORDER:
        for metric in [
            "HR@5_fail_strict",
            "NDCG@5_fail_strict",
            "mean_positive_rank_fail_strict_invalid_as_20",
            "HR@5_positive_present_only",
            "NDCG@5_positive_present_only",
            "strict_valid_rate",
            "structural_invalid_rate",
            "duplicate_token_rate",
            "retry_rate",
            "novelty",
            "APLT_long_tail",
            "average_popularity",
            "popularity_bias",
            "catalog_coverage",
            "ILD_diversity",
            "calibration",
        ]:
            base = lookup.get((model, STRATEGY_ORDER[0], metric), 0.0)
            llmrank = lookup.get((model, STRATEGY_ORDER[1], metric), 0.0)
            comparisons.append(
                {
                    "comparison_id": stable_hash("F6L-LASTFM-V4-RERANK-COMP", run_id, model, metric)[:24],
                    "run_id": run_id,
                    "sample_id": "phase_f6l_lastfm_operationalized_1000",
                    "model_name": model,
                    "metric": metric,
                    "base_openp5_value": format_float(base),
                    "llmrank_recency_value": format_float(llmrank),
                    "delta_llmrank_minus_base": format_float(llmrank - base),
                    "shared_users": str(USER_COUNT),
                    "positive_present_users": str(coverage["positive_present_users"]),
                    "evidence_class": CLASSIFICATION,
                    "status": "computed_same_users_same_reranking_candidates",
                    "notes": "Base/OpenP5-derived and LLMRank recency-focused reranking are compared on identical users and candidate lists.",
                }
            )
    return metrics, beyond, comparisons


def ranking_baseline_comparison_rows(run_id: str, reranking_metrics: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    baseline = ranking_baseline_metric_lookup()
    rerank_lookup = {
        (row["model_name"], row["prompting_strategy"], row["metric"]): row.get("value", "")
        for row in reranking_metrics
    }
    rows: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        for reranking_strategy, ranking_strategy in RANKING_TO_RERANKING_STRATEGY.items():
            for metric in [
                "HR@5_fail_strict",
                "NDCG@5_fail_strict",
                "mean_positive_rank_fail_strict_invalid_as_20",
                "strict_valid_rate",
                "duplicate_token_rate",
            ]:
                ranking_value = baseline.get((model, ranking_strategy, metric), "")
                reranking_value = rerank_lookup.get((model, reranking_strategy, metric), "")
                rows.append(
                    {
                        "comparison_id": stable_hash("F6L-LASTFM-V4-RERANK-VS-RANK", run_id, model, reranking_strategy, metric)[:24],
                        "run_id": run_id,
                        "ranking_run_id": RANKING_BASELINE_RUN_ID,
                        "dataset": "LastFM",
                        "model_name": model,
                        "ranking_strategy": ranking_strategy,
                        "reranking_strategy": reranking_strategy,
                        "metric": metric,
                        "ranking_fail_strict_value": ranking_value,
                        "reranking_fail_strict_value": reranking_value,
                        "delta_reranking_minus_ranking": decimal_delta(reranking_value, ranking_value) if ranking_value and reranking_value else "",
                        "ranking_freeze_hash": RANKING_BASELINE_FREEZE_HASH,
                        "ranking_source_file": rel_safe(RANKING_BASELINE_PATHS["metrics"]),
                        "reranking_source_file": rel_safe(METRICS_PATH),
                        "status": "computed_from_frozen_fail_strict_ranking_baseline" if ranking_value and reranking_value else "pending_reranking_metric",
                        "notes": "Ranking value is read from the frozen fail-strict baseline; complete-case ranking is secondary only.",
                    }
                )
    return rows


def build_failure_rows(parse_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    completed = len(parse_rows)
    by_reason = Counter(row["failure_reason"] for row in parse_rows if row["strict_valid"] != "true")
    if not by_reason:
        return [
            {
                "failure_reason": "none",
                "count": "0",
                "requests": "",
                "rate_over_completed": "0",
                "policy": "no structural invalid final outputs observed",
                "evidence_class": CLASSIFICATION,
                "notes": "All completed final responses are strict-valid.",
            }
        ]
    rows = []
    for reason, count in sorted(by_reason.items()):
        request_ids = [row["request_id"] for row in parse_rows if row["failure_reason"] == reason]
        policy = "observed_runtime_property_no_retry_no_repair_no_dedup" if reason == "duplicate_token" else "hard_structural_invalid_stop"
        rows.append(
            {
                "failure_reason": reason,
                "count": str(count),
                "requests": ";".join(request_ids[:20]),
                "rate_over_completed": format_float(count / completed if completed else 0.0),
                "policy": policy,
                "evidence_class": CLASSIFICATION,
                "notes": "Request list is truncated to first 20 ids for readability.",
            }
        )
    return rows


def final_validation_rows(
    build: RerankingBuild,
    parse_rows: Sequence[Mapping[str, str]],
    recommendation_rows: Sequence[Mapping[str, str]],
    attempt_counts: Mapping[str, int],
    decision: str,
    raw_output_dir: Path,
) -> list[dict[str, str]]:
    completed = len(parse_rows)
    strict = sum(1 for row in parse_rows if row["strict_valid"] == "true")
    failed = completed - strict
    residual = REQUEST_COUNT - completed
    hard_invalid = sum(1 for row in parse_rows if is_hard_structural_invalid(row) or row["hash_coherence"] != "true")
    retry = sum(max(0, count - 1) for count in attempt_counts.values())
    response_records = f6l.read_jsonl(f6l.raw_response_paths(raw_output_dir)["responses"])
    final_ids = [str(row.get("request_id", "")) for row in response_records if str(row.get("is_final_attempt", "true")).lower() == "true"]
    duplicate_final_ids = [request_id for request_id, count in Counter(final_ids).items() if count > 1]

    checks = [
        ("planned_4000", len(build.manifest_rows), REQUEST_COUNT, rel_safe(MANIFEST_PATH), "Only the authorized 4000-row reranking manifest is planned."),
        ("completed_plus_residual_4000", completed + residual, REQUEST_COUNT, rel_safe(STATE_PATH), "Accounting is closed."),
        ("strict_plus_failed_equals_completed", strict + failed, completed, rel_safe(PARSE_RESULTS_PATH), "Strict-valid plus structural-invalid equals completed final responses."),
        ("hard_invalid_zero_for_complete", hard_invalid, 0, rel_safe(PARSE_RESULTS_PATH), "Hard structural/provenance failures stop execution."),
        ("no_duplicate_final_response_ids", len(duplicate_final_ids), 0, rel_safe(raw_output_dir), "Final response ids must be unique."),
        ("recommendations_match_strict_valid_top10", len(recommendation_rows), strict * 10, rel_safe(RECOMMENDATIONS_PATH), "Each strict-valid response yields 10 recommendation rows."),
        ("parser_hash_unchanged", parser_source_hash(), EXPECTED_V4_PARSER_HASH, "inspect.getsource(parse_v4_response)", "Parser hash remains frozen."),
        ("schema_hash_unchanged", schema_hash(), EXPECTED_V4_SCHEMA_HASH, "V4_JSON_SCHEMA", "Schema hash remains frozen."),
        ("output_contract_hash_unchanged", output_contract_hash(), EXPECTED_V4_OUTPUT_CONTRACT_HASH, "OUTPUT_CONTRACT_RULES", "Output contract hash remains frozen."),
        ("retry_transport_reported", retry >= 0, True, rel_safe(EXECUTION_PATH), "Retries are transport-only and reported."),
        ("decision_token_allowed", decision in {DECISION_COMPLETE_ALL_STRICT, DECISION_COMPLETE_WITH_INVALID, DECISION_PARTIAL, DECISION_HARD_CONTRACT, DECISION_SYSTEMIC_DUPLICATE}, True, rel_safe(JSON_MANIFEST_PATH), "Decision token is from the reranking V4 vocabulary."),
    ]
    return [
        {
            "validation_id": f"F6L-LASTFM-V4-RERANK-FINAL-{index:02d}",
            "check_name": name,
            "status": "pass" if str(observed) == str(expected) else "fail",
            "observed": str(observed),
            "expected": str(expected),
            "severity": "error",
            "evidence": evidence,
            "notes": notes,
        }
        for index, (name, observed, expected, evidence, notes) in enumerate(checks, start=1)
    ]


def acceptance_rows(final_validation: Sequence[Mapping[str, str]], decision: str, parse_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    duplicate_invalid = duplicate_token_invalid_response_count(parse_rows)
    hard_invalid = sum(1 for row in parse_rows if is_hard_structural_invalid(row) or row["hash_coherence"] != "true")
    criteria = [
        ("final_validation_pass", all(row["status"] == "pass" for row in final_validation), "true", rel_safe(FINAL_VALIDATION_PATH), "All final validation rows pass."),
        ("classification_correct", CLASSIFICATION, CLASSIFICATION, rel_safe(JSON_MANIFEST_PATH), "Classification remains operationalized, not paper-exact."),
        ("duplicate_semantics_fail_strict", duplicate_invalid >= 0, "reported", rel_safe(FAILURE_TAXONOMY_PATH), "Duplicate-token invalidity is observed, retained, and not repaired."),
        ("hard_invalid_absent_or_stopped", hard_invalid == 0 or decision == DECISION_HARD_CONTRACT, "true", rel_safe(PARSE_RESULTS_PATH), "Hard invalidity either absent or responsible for the stop token."),
    ]
    return [
        {
            "criterion": name,
            "status": "pass" if expected == "reported" or str(observed).lower() == str(expected).lower() else "fail",
            "observed": str(observed),
            "expected": str(expected),
            "evidence": evidence,
            "notes": notes,
        }
        for name, observed, expected, evidence, notes in criteria
    ]


def write_report(manifest: Mapping[str, Any], decision: str, completed: int, strict: int, failed: int, retry: int) -> None:
    coverage = manifest.get("positive_coverage", {})
    lines = [
        "# F6L LastFM V4 Reranking Full Run",
        "",
        f"Decision token: `{decision}`.",
        "",
        f"Description: {RUN_DESCRIPTION}.",
        f"Classification: `{CLASSIFICATION}`.",
        "",
        "## Source",
        "",
        f"- Canonical pool: `{manifest.get('source_pool', '')}`",
        f"- Pool file SHA256: `{manifest.get('source_pool_file_sha256', '')}`",
        f"- Pool content hash: `{manifest.get('source_pool_content_hash', '')}`",
        f"- Candidate definition: {manifest.get('candidate_pool_definition', '')}",
        f"- Positive coverage: `{coverage.get('positive_present_users', '')}/{coverage.get('users', '')}` users; absent `{coverage.get('positive_absent_users', '')}`.",
        f"- Ranking baseline freeze hash: `{RANKING_BASELINE_FREEZE_HASH}`.",
        f"- Ranking baseline audit: `{rel_safe(RANKING_BASELINE_AUDIT_PATH)}`.",
        "",
        "## Exclusions",
        "",
        "- Full-cohort 1859-user reranking pool is excluded.",
        "- F6D3 planned full-cohort reranking manifest is excluded.",
        "- Ranking V4 raw outputs and candidate rows are not used as reranking inputs.",
        "- Ranking-vs-reranking comparisons use frozen fail-strict ranking values only; complete-case ranking is secondary.",
        "",
        "## Execution",
        "",
        f"- Run id: `{manifest.get('run_id', '')}`",
        f"- Raw output dir: `{manifest.get('raw_output_dir', '')}`",
        f"- Planned: `{REQUEST_COUNT}`; completed: `{completed}`; strict-valid: `{strict}`; structural-invalid: `{failed}`; retry: `{retry}`.",
        f"- Parser hash: `{manifest.get('parser_source_hash', '')}`",
        f"- Schema hash: `{manifest.get('json_schema_hash', '')}`",
        f"- Manifest hash: `{manifest.get('manifest_hash', '')}`",
        "",
        "No paper-exact LastFM reranking claim is made.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def consolidate(
    build: RerankingBuild,
    raw_output_dir: Path,
    preflight_rows: Sequence[Mapping[str, str]],
    focused_test_result: Mapping[str, Any],
    started_at: str,
    completed_at: str,
    retry: int,
    *,
    requested_decision: str | None = None,
    resume_start_completed: int = 0,
) -> tuple[str, dict[str, Any]]:
    paths = f6l.raw_response_paths(raw_output_dir)
    records_by_id = f6l.final_response_records(raw_output_dir)
    all_records = f6l.read_jsonl(paths["responses"])
    request_records = f6l.read_jsonl(paths["requests"])
    attempt_counts: Counter[str] = Counter(str(row.get("request_id", "")) for row in all_records if row.get("request_id"))
    manifest_by_id = {row["request_id"]: row for row in build.manifest_rows}
    parse_rows = [
        parse_result_row(manifest_by_id[request_id], record)
        for request_id, record in records_by_id.items()
        if request_id in manifest_by_id
    ]
    order_by_request = {row["request_id"]: index for index, row in enumerate(build.manifest_rows)}
    parse_rows.sort(key=lambda row: order_by_request[row["request_id"]])
    recommendation_rows = [
        rec
        for parse_row in parse_rows
        for rec in recommendation_rows_from_parse(manifest_by_id[parse_row["request_id"]], parse_row)
    ]
    execution_rows = [
        execution_row(manifest_by_id[str(record.get("request_id", ""))], record)
        for record in all_records
        if str(record.get("request_id", "")) in manifest_by_id
    ]
    metrics, beyond, comparisons = compute_metrics(build.run_id, parse_rows, recommendation_rows, build.manifest_rows, attempt_counts)
    ranking_comparisons = ranking_baseline_comparison_rows(build.run_id, metrics)
    duplicate_windows = build_duplicate_window_rows(build.run_id, parse_rows, build.manifest_rows, resume_start_completed)
    failure_rows = build_failure_rows(parse_rows)
    completed = len(parse_rows)
    strict = sum(1 for row in parse_rows if row["strict_valid"] == "true")
    failed = completed - strict
    residual = REQUEST_COUNT - completed
    if requested_decision:
        decision = requested_decision
    elif residual:
        decision = DECISION_PARTIAL
    elif failed:
        decision = DECISION_COMPLETE_WITH_INVALID
    else:
        decision = DECISION_COMPLETE_ALL_STRICT
    final_validation = final_validation_rows(build, parse_rows, recommendation_rows, attempt_counts, decision, raw_output_dir)
    acceptance = acceptance_rows(final_validation, decision, parse_rows)
    duration_seconds = max(0.0, datetime.fromisoformat(completed_at.replace("Z", "+00:00")).timestamp() - datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp())
    runtime_rows = [
        {
            "run_id": build.run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": format_float(duration_seconds),
            "planned": str(REQUEST_COUNT),
            "completed": str(completed),
            "strict_valid": str(strict),
            "structural_invalid": str(failed),
            "retry": str(sum(max(0, count - 1) for count in attempt_counts.values())),
            "throughput_requests_per_hour": format_float(completed / max(duration_seconds / 3600.0, 1e-9) if completed else 0.0),
            "raw_output_dir": rel_safe(raw_output_dir),
            "notes": "Runtime is wall-clock process duration for this local reranking run.",
        }
    ]
    f6l.write_csv(EXECUTION_PATH, execution_rows, EXECUTION_FIELDS)
    f6l.write_csv(PARSE_RESULTS_PATH, parse_rows, PARSE_FIELDS)
    f6l.write_csv(RECOMMENDATIONS_PATH, recommendation_rows, RECOMMENDATION_FIELDS)
    f6l.write_csv(METRICS_PATH, metrics, METRIC_FIELDS)
    f6l.write_csv(BEYOND_UTILITY_PATH, beyond, BEYOND_FIELDS)
    f6l.write_csv(COMPARISON_PATH, comparisons, COMPARISON_FIELDS)
    f6l.write_csv(RANKING_BASELINE_COMPARISON_PATH, ranking_comparisons, RANKING_BASELINE_COMPARISON_FIELDS)
    f6l.write_csv(DUPLICATE_WINDOWS_PATH, duplicate_windows, DUPLICATE_WINDOW_FIELDS)
    f6l.write_csv(FAILURE_TAXONOMY_PATH, failure_rows, FAILURE_TAXONOMY_FIELDS)
    f6l.write_csv(RUNTIME_PATH, runtime_rows, RUNTIME_FIELDS)
    f6l.write_csv(FINAL_VALIDATION_PATH, final_validation, FINAL_VALIDATION_FIELDS)
    f6l.write_csv(ACCEPTANCE_PATH, acceptance, ACCEPTANCE_FIELDS)
    canary_rows = canary_rows_from_parse(build.run_id, parse_rows, build.manifest_rows, raw_output_dir)
    f6l.write_csv(CANARY_PATH, canary_rows, CANARY_FIELDS)
    manifest = {
        **build.manifest,
        "decision_token": decision,
        "completed": completed,
        "strict_valid": strict,
        "failed": failed,
        "residual": residual,
        "retry": sum(max(0, count - 1) for count in attempt_counts.values()),
        "canary_status": "pass" if all(row["status"] == "pass" for row in canary_rows[:2]) else "not_passed",
        "circuit_breaker_status": "clear" if decision in {DECISION_COMPLETE_ALL_STRICT, DECISION_COMPLETE_WITH_INVALID, DECISION_PARTIAL} else decision,
        "preflight_pass": all(row["status"] == "pass" for row in preflight_rows),
        "focused_tests": focused_test_result,
        "output_hashes": {
            rel_safe(path): f6l.safe_file_sha256(path)
            for path in [
                MANIFEST_PATH,
                FREEZE_PATH,
                PREFLIGHT_PATH,
                STATE_PATH,
                CANARY_PATH,
                EXECUTION_PATH,
                PARSE_RESULTS_PATH,
                RECOMMENDATIONS_PATH,
                METRICS_PATH,
                BEYOND_UTILITY_PATH,
                COMPARISON_PATH,
                RANKING_BASELINE_AUDIT_PATH,
                RANKING_BASELINE_COMPARISON_PATH,
                DUPLICATE_WINDOWS_PATH,
                FAILURE_TAXONOMY_PATH,
                RUNTIME_PATH,
                FINAL_VALIDATION_PATH,
                ACCEPTANCE_PATH,
                REPORT_PATH,
            ]
            if path.exists()
        },
        "raw_counts": {
            "request_records": len(request_records),
            "response_records": len(all_records),
            "final_response_records": len(records_by_id),
        },
    }
    f6l.write_json(JSON_MANIFEST_PATH, manifest)
    f6l.write_json(paths["manifest"], {**manifest, "completed_at": completed_at})
    write_report(manifest, decision, completed, strict, failed, sum(max(0, count - 1) for count in attempt_counts.values()))
    counts = structural_counts(parse_rows)
    closed_windows = [row for row in duplicate_windows if row.get("closed") == "true"]
    state_window = next((row for row in closed_windows if row.get("status") == "fail"), closed_windows[-1] if closed_windows else {})
    state_window_index = int(str(state_window.get("window_index", "0"))) if str(state_window.get("window_index", "0")).isdigit() else 0
    state_window_rate = float(str(state_window.get("duplicate_token_response_rate", "0") or "0"))
    write_state(
        run_id=build.run_id,
        raw_output_dir=raw_output_dir,
        stage="complete" if residual == 0 else "partial_resumable",
        decision_token=decision,
        started_clock=time.perf_counter() - duration_seconds,
        last_request_id=parse_rows[-1]["request_id"] if parse_rows else "",
        last_model_name=parse_rows[-1]["model_name"] if parse_rows else "",
        last_prompting_strategy=parse_rows[-1]["prompting_strategy"] if parse_rows else "",
        completed=completed,
        strict_valid=strict,
        failed=failed,
        retry=sum(max(0, count - 1) for count in attempt_counts.values()),
        duplicate_token_count=counts["duplicate_token_count"],
        malformed_json_count=counts["malformed_json_count"],
        unknown_token_count=counts["unknown_token_count"],
        cumulative_duplicate_token_response_rate=duplicate_token_response_rate(parse_rows),
        current_duplicate_token_window_rate=0.0,
        closed_duplicate_window_index=state_window_index,
        closed_duplicate_token_window_rate=state_window_rate,
        consecutive_duplicate_token_invalid_count=trailing_duplicate_token_invalid_count(parse_rows, build.manifest_rows),
        canary_status=manifest["canary_status"],
        circuit_breaker_status=manifest["circuit_breaker_status"],
        notes="consolidated reranking V4 outputs",
    )
    return decision, manifest


def execute_full_run(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if not args.allow_api_calls:
        raise SystemExit("--allow-api-calls is required before local Ollama inference")
    if not args.local_only:
        raise SystemExit("--local-only is required; remote endpoints are not authorized")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_dir = Path(args.raw_output_dir) if args.raw_output_dir else local_raw_output_dir(run_id)
    if not raw_output_dir.is_absolute():
        raw_output_dir = (f6l.PROJECT_ROOT / raw_output_dir).resolve()
    build = build_full_run(run_id=run_id, raw_output_dir=raw_output_dir, write_outputs=True)
    preflight_rows, focused_test_result, ollama_state = validate_preflight(
        build,
        raw_output_dir=raw_output_dir,
        resume=args.resume,
        allow_api_calls=args.allow_api_calls,
        local_only=args.local_only,
        endpoint=str(args.ollama_endpoint),
        run_tests=not args.skip_focused_tests,
        check_runtime=True,
    )
    preflight_pass = all(row["status"] == "pass" for row in preflight_rows)
    start_clock = time.perf_counter()
    if args.preflight_only:
        decision = DECISION_PREFLIGHT_PASSED if preflight_pass else DECISION_PREFLIGHT_BLOCKED
        write_state(
            run_id=run_id,
            raw_output_dir=raw_output_dir,
            stage="preflight_passed" if preflight_pass else "blocked_preflight",
            decision_token=decision,
            started_clock=start_clock,
            last_request_id="",
            last_model_name="",
            last_prompting_strategy="",
            completed=0,
            strict_valid=0,
            failed=0,
            retry=0,
            canary_status="not_started",
            circuit_breaker_status="clear" if preflight_pass else "preflight_failed",
            notes="preflight-only; no local inference executed",
        )
        manifest = {**build.manifest, "decision_token": decision, "preflight_pass": preflight_pass, "focused_tests": focused_test_result}
        f6l.write_json(JSON_MANIFEST_PATH, manifest)
        write_report(manifest, decision, 0, 0, 0, 0)
        return decision, manifest
    if not preflight_pass or ollama_state is None:
        write_state(
            run_id=run_id,
            raw_output_dir=raw_output_dir,
            stage="blocked_preflight",
            decision_token=DECISION_PREFLIGHT_BLOCKED,
            started_clock=start_clock,
            last_request_id="",
            last_model_name="",
            last_prompting_strategy="",
            completed=0,
            strict_valid=0,
            failed=0,
            retry=0,
            canary_status="not_started",
            circuit_breaker_status="preflight_failed",
            notes="preflight failed; no reranking inference executed",
        )
        manifest = {**build.manifest, "decision_token": DECISION_PREFLIGHT_BLOCKED, "preflight_failures": [row for row in preflight_rows if row["status"] != "pass"], "focused_tests": focused_test_result}
        f6l.write_json(JSON_MANIFEST_PATH, manifest)
        write_report(manifest, DECISION_PREFLIGHT_BLOCKED, 0, 0, 0, 0)
        return DECISION_PREFLIGHT_BLOCKED, manifest

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
                "decision_token": DECISION_PARTIAL,
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
    sample = sample_rows_by_user()
    candidates = reranking_pool_rows_by_user()
    manifest_by_id = {row["request_id"]: row for row in build.manifest_rows}
    completed_records = f6l.final_response_records(raw_output_dir) if args.resume else {}
    all_attempts = f6l.read_jsonl(paths["responses"])
    attempt_counts: Counter[str] = Counter(str(row.get("request_id", "")) for row in all_attempts if row.get("request_id"))
    state_parse_rows = [
        parse_result_row(manifest_by_id[request_id], record)
        for request_id, record in completed_records.items()
        if request_id in manifest_by_id
    ]
    order_by_request = {row["request_id"]: index for index, row in enumerate(build.manifest_rows)}
    state_parse_rows.sort(key=lambda row: order_by_request[row["request_id"]])
    completed = len(state_parse_rows)
    strict_valid = sum(1 for row in state_parse_rows if row["strict_valid"] == "true")
    failed = completed - strict_valid
    retry = sum(max(0, count - 1) for count in attempt_counts.values())
    resume_start_completed = completed
    canary_status = "pass" if completed >= 40 and all(row["status"] == "pass" for row in canary_rows_from_parse(run_id, state_parse_rows, build.manifest_rows, raw_output_dir)[:2]) else ("pending" if completed < 40 else "not_passed")
    counts = structural_counts(state_parse_rows)
    write_state(
        run_id=run_id,
        raw_output_dir=raw_output_dir,
        stage="executing",
        decision_token=DECISION_PARTIAL,
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
        consecutive_duplicate_token_invalid_count=trailing_duplicate_token_invalid_count(state_parse_rows, build.manifest_rows),
        canary_status=canary_status,
        circuit_breaker_status="clear",
        notes="starting or resuming append-only reranking V4 execution",
    )
    resume_new_parse_rows: list[dict[str, str]] = []
    pending_rows = [row for row in build.manifest_rows if row["request_id"] not in completed_records]
    active_pending_rows = [row for row in pending_rows if attempt_counts[row["request_id"]] < TRANSPORT_DEFER_ATTEMPT_LIMIT]
    deferred_pending_rows = [row for row in pending_rows if attempt_counts[row["request_id"]] >= TRANSPORT_DEFER_ATTEMPT_LIMIT]
    transport_deferred_count = 0
    for row in [*active_pending_rows, *deferred_pending_rows]:
        prompt = render_prompt({**sample[row["user_id"]], "_subset": row["subset"]}, candidates[row["user_id"]], row["prompting_strategy"])
        hard_contract_mismatch = (
            f6l.sha256_text(prompt) != row["rendered_prompt_hash"]
            or row["parser_source_hash"] != EXPECTED_V4_PARSER_HASH
            or row["output_contract_hash"] != EXPECTED_V4_OUTPUT_CONTRACT_HASH
            or row["prompt_template_hash"] != prompt_template_hash(row["prompting_strategy"])
        )
        if hard_contract_mismatch:
            return consolidate(
                build,
                raw_output_dir,
                preflight_rows,
                focused_test_result,
                run_started_at,
                utc_now(),
                retry,
                requested_decision=DECISION_HARD_CONTRACT,
                resume_start_completed=resume_start_completed,
            )
        attempt_index = attempt_counts[row["request_id"]] + 1
        request_record = {
            **request_payload_for_row(row),
            "prompt": prompt,
            "endpoint": str(args.ollama_endpoint),
            "queued_at": utc_now(),
            "request_order": str(order_by_request[row["request_id"]] + 1),
            "attempt_index": attempt_index,
            "expected_raw_output_path": row["expected_raw_output_path"],
        }
        f6l.append_jsonl_fsync(paths["requests"], request_record)
        result = v4.execute_local_ollama_schema_request(row["model_name"], prompt, str(args.ollama_endpoint))
        attempt_counts[row["request_id"]] += 1
        if not result.backend_success:
            parse = v4.ParserV4Result(False, "transport_error", [], [], [], 0, 0, 0, 10, 0, 0, 0, None)
            raw_json_path = paths["individual"] / f"{row['request_id']}__attempt{attempt_index:03d}.json"
            raw_record = raw_record_for_response(row, prompt, result, parse, raw_json_path, attempt_index=attempt_index, is_final_attempt=False)
            raw_json_path.write_text(json.dumps(raw_record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            f6l.append_jsonl_fsync(paths["responses"], raw_record)
            retry = sum(max(0, count - 1) for count in attempt_counts.values())
            counts = structural_counts(state_parse_rows)
            closed_index, closed_rate, _ = closed_duplicate_window_status(resume_new_parse_rows)
            write_state(
                run_id=run_id,
                raw_output_dir=raw_output_dir,
                stage="partial_resumable_infrastructure_interruption",
                decision_token=DECISION_PARTIAL,
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
                closed_duplicate_window_index=closed_index,
                closed_duplicate_token_window_rate=closed_rate,
                consecutive_duplicate_token_invalid_count=trailing_duplicate_token_invalid_count(state_parse_rows, build.manifest_rows),
                canary_status=canary_status,
                circuit_breaker_status="clear",
                notes=f"transport error deferred without final completion {result.backend_error_type}:{result.backend_error_message}",
            )
            transport_deferred_count += 1
            continue
        parse = v4.parse_v4_response(result.response_text, candidates[row["user_id"]])
        raw_json_path = paths["individual"] / f"{row['request_id']}.json"
        if raw_json_path.exists():
            raise RuntimeError(f"refusing to overwrite existing final raw response: {raw_json_path}")
        raw_record = raw_record_for_response(row, prompt, result, parse, raw_json_path, attempt_index=attempt_index, is_final_attempt=True)
        raw_json_path.write_text(json.dumps(raw_record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        f6l.append_jsonl_fsync(paths["responses"], raw_record)
        completed_records[row["request_id"]] = raw_record
        parse_row = parse_result_row(row, raw_record)
        state_parse_rows.append(parse_row)
        resume_new_parse_rows.append(parse_row)
        completed += 1
        strict_valid += 1 if parse.strict_valid else 0
        failed = completed - strict_valid
        retry = sum(max(0, count - 1) for count in attempt_counts.values())
        counts = structural_counts(state_parse_rows)
        consecutive_duplicate_invalid = trailing_duplicate_token_invalid_count(state_parse_rows, build.manifest_rows)
        current_rate = current_duplicate_window_rate(resume_new_parse_rows)
        closed_index, closed_rate, duplicate_window_breaker = closed_duplicate_window_status(resume_new_parse_rows)
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
        if completed == 40:
            canary_rows = canary_rows_from_parse(run_id, state_parse_rows, build.manifest_rows, raw_output_dir)
            f6l.write_csv(CANARY_PATH, canary_rows, CANARY_FIELDS)
            if any(item["status"] != "pass" for item in canary_rows[:2]):
                hard_breaker = hard_breaker or "canary_hard_validation_failed"
            canary_status = "pass" if all(item["status"] == "pass" for item in canary_rows[:2]) else "not_passed"
        elif completed < 40:
            canary_status = "pending"
        if hard_breaker or duplicate_breaker:
            decision = DECISION_HARD_CONTRACT if hard_breaker else DECISION_SYSTEMIC_DUPLICATE
            breaker = hard_breaker or duplicate_breaker
            write_state(
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
                current_duplicate_token_window_rate=current_rate,
                closed_duplicate_window_index=closed_index,
                closed_duplicate_token_window_rate=closed_rate,
                consecutive_duplicate_token_invalid_count=consecutive_duplicate_invalid,
                canary_status=canary_status,
                circuit_breaker_status=breaker,
                notes="reranking V4 amended circuit breaker triggered",
            )
            return consolidate(
                build,
                raw_output_dir,
                preflight_rows,
                focused_test_result,
                run_started_at,
                utc_now(),
                retry,
                requested_decision=decision,
                resume_start_completed=resume_start_completed,
            )
        write_state(
            run_id=run_id,
            raw_output_dir=raw_output_dir,
            stage="executing",
            decision_token=DECISION_PARTIAL,
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
            current_duplicate_token_window_rate=current_rate,
            closed_duplicate_window_index=closed_index,
            closed_duplicate_token_window_rate=closed_rate,
            consecutive_duplicate_token_invalid_count=consecutive_duplicate_invalid,
            canary_status=canary_status,
            circuit_breaker_status="clear",
            notes="append-only local Ollama V4 reranking execution in progress",
        )
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "completed": completed,
                    "strict_valid": strict_valid,
                    "structural_invalid": failed,
                    "residual": REQUEST_COUNT - completed,
                    "retry": retry,
                    "current_200_window_duplicate_token_rate": format_float(current_rate),
                    "closed_200_window_index": closed_index,
                    "closed_200_window_duplicate_token_rate": format_float(closed_rate),
                    "consecutive_duplicate_token_invalid": consecutive_duplicate_invalid,
                    "last_request_id": row["request_id"],
                    "canary_status": canary_status,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return consolidate(
        build,
        raw_output_dir,
        preflight_rows,
        focused_test_result,
        run_started_at,
        utc_now(),
        retry,
        requested_decision=DECISION_PARTIAL if transport_deferred_count else None,
        resume_start_completed=resume_start_completed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or preflight the F6L LastFM V4 reranking full run.")
    parser.add_argument("--run-id", default="", help="Run id, e.g. 20260805_120000.")
    parser.add_argument("--raw-output-dir", default="", help="Raw output directory under results/llm.")
    parser.add_argument("--allow-api-calls", action="store_true", help="Required before local Ollama inference.")
    parser.add_argument("--local-only", action="store_true", help="Require loopback-only Ollama endpoint.")
    parser.add_argument("--ollama-endpoint", default=f6l.OLLAMA_LOCAL_GENERATE_URL, help="Local Ollama /api/generate endpoint.")
    parser.add_argument("--preflight-only", action="store_true", help="Build manifests and run gates without inference.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing append-only raw run.")
    parser.add_argument("--skip-focused-tests", action="store_true", help="Only for controlled test paths; execution normally runs focused tests.")
    args = parser.parse_args(argv)
    decision, manifest = execute_full_run(args)
    print(
        json.dumps(
            {
                "decision_token": decision,
                "run_id": manifest.get("run_id", ""),
                "manifest_hash": manifest.get("manifest_hash", ""),
                "raw_output_dir": manifest.get("raw_output_dir", ""),
                "completed": manifest.get("completed", 0),
                "strict_valid": manifest.get("strict_valid", 0),
                "failed": manifest.get("failed", 0),
                "residual": manifest.get("residual", REQUEST_COUNT),
            },
            sort_keys=True,
        )
    )
    return 0 if decision != DECISION_PREFLIGHT_BLOCKED else 1


if __name__ == "__main__":
    raise SystemExit(main())

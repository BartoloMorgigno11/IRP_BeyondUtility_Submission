"""Execute the F6L LastFM V4.1 reranking continuation.

This runner consumes only the frozen V4.1 continuation manifest. It writes a
new append-only raw directory for continuation attempts and never overwrites the
original V4 raw evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.analysis import phase_f6l_lastfm_v4_reranking_continuation_v4_1 as freeze
from src.analysis import phase_f6l_lastfm_v4_reranking_full_run as rerank
from src.analysis import phase_f6l_lastfm_v4_reranking_post_breaker as post
from src.analysis import phase_f6l_lastfm_corrective_output_contract_v4_pilot as v4
from src.analysis import plan_f6l_paper_faithful_completion_and_rq_revalidation as f6l


EXECUTION_ID = "phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution"
BASE_RUN_ID = "20260806_034027"
CONTINUATION_ID = freeze.CONTINUATION_ID

DECISION_BLOCKED_PREFLIGHT = "f6l_lastfm_v4_reranking_continuation_v4_1_blocked_by_precondition_failure"
DECISION_HARD_STOP = "f6l_lastfm_v4_reranking_continuation_v4_1_stopped_by_hard_contract_regression"
DECISION_DUP_STOP = "f6l_lastfm_v4_reranking_continuation_v4_1_stopped_by_duplicate_regression"
DECISION_COMPLETE = "f6l_lastfm_v4_reranking_continuation_v4_1_complete"
DECISION_PARTIAL_TRANSPORT = "f6l_lastfm_v4_reranking_continuation_v4_1_partial_resumable_transport_timeout"
DECISION_EXECUTING = "f6l_lastfm_v4_reranking_continuation_v4_1_executing"

TABLES_DIR = f6l.PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = f6l.PROJECT_ROOT / "results" / "manifests"
REPORT_DIR = f6l.PROJECT_ROOT / "report"

PREFLIGHT_PATH = TABLES_DIR / f"{EXECUTION_ID}_preflight.csv"
STATE_PATH = TABLES_DIR / f"{EXECUTION_ID}_state.csv"
ATTEMPT_LOG_PATH = TABLES_DIR / f"{EXECUTION_ID}_attempt_log.csv"
PARSE_RESULTS_PATH = TABLES_DIR / f"{EXECUTION_ID}_parse_results.csv"
AGGREGATE_PARSE_RESULTS_PATH = TABLES_DIR / f"{EXECUTION_ID}_aggregate_parse_results.csv"
RECOMMENDATIONS_PATH = TABLES_DIR / f"{EXECUTION_ID}_recommendation_rows.csv"
AGGREGATE_RECOMMENDATIONS_PATH = TABLES_DIR / f"{EXECUTION_ID}_aggregate_recommendation_rows.csv"
FAILURE_TAXONOMY_PATH = TABLES_DIR / f"{EXECUTION_ID}_failure_taxonomy.csv"
DUPLICATE_WINDOWS_PATH = TABLES_DIR / f"{EXECUTION_ID}_duplicate_windows.csv"
VALIDITY_PATH = TABLES_DIR / f"{EXECUTION_ID}_validity.csv"
FAIL_STRICT_METRICS_PATH = TABLES_DIR / f"{EXECUTION_ID}_fail_strict_metrics.csv"
COMPLETE_CASE_METRICS_PATH = TABLES_DIR / f"{EXECUTION_ID}_complete_case_metrics.csv"
BEYOND_UTILITY_PATH = TABLES_DIR / f"{EXECUTION_ID}_beyond_utility_metrics.csv"
RANKING_COMPARISON_PATH = TABLES_DIR / f"{EXECUTION_ID}_ranking_vs_reranking_comparison.csv"
STATISTICAL_TESTS_PATH = TABLES_DIR / f"{EXECUTION_ID}_statistical_tests.csv"
FINAL_VALIDATION_PATH = TABLES_DIR / f"{EXECUTION_ID}_final_validation.csv"
RUNTIME_PATH = TABLES_DIR / f"{EXECUTION_ID}_runtime.csv"
SUMMARY_PATH = MANIFESTS_DIR / f"{EXECUTION_ID}_summary.json"
HASHES_PATH = MANIFESTS_DIR / f"{EXECUTION_ID}_hashes.json"
REPORT_PATH = REPORT_DIR / f"{EXECUTION_ID}_report.md"

PREFLIGHT_FIELDS = ["check_id", "check_name", "status", "observed", "expected", "severity", "evidence", "notes"]
STATE_FIELDS = [
    "execution_run_id",
    "continuation_id",
    "stage",
    "decision_token",
    "planned",
    "completed_final_responses",
    "residual",
    "strict_valid",
    "structural_invalid",
    "duplicate_token_invalid",
    "non_final_transport_timeout",
    "retry_attempt_records",
    "canary_status",
    "current_cell",
    "current_cell_closed_window_duplicate_rate",
    "current_cell_cumulative_duplicate_rate",
    "breaker_state",
    "last_canonical_request_id",
    "last_timestamp",
    "raw_output_dir",
    "notes",
]
VALIDITY_FIELDS = ["run_id", "model_name", "prompting_strategy", "planned", "final_responses", "strict_valid", "structural_invalid", "duplicate_token_invalid", "validity_rate", "duplicate_token_rate", "notes"]
DUPLICATE_WINDOW_FIELDS = [
    "run_id",
    "model_name",
    "prompting_strategy",
    "window_index",
    "window_start_cell_final",
    "window_end_cell_final",
    "closed",
    "responses",
    "duplicate_token_invalid",
    "duplicate_token_rate",
    "cumulative_responses",
    "cumulative_duplicate_token_invalid",
    "cumulative_duplicate_token_rate",
    "status",
    "breaker_rule",
    "notes",
]
FINAL_VALIDATION_FIELDS = ["check_name", "status", "observed", "expected", "evidence", "notes"]
STAT_FIELDS = ["test_name", "model_name", "prompting_strategy", "metric", "observed", "expected", "p_value", "status", "notes"]
RUNTIME_FIELDS = ["execution_run_id", "started_at", "completed_at", "duration_seconds", "planned", "continuation_final_responses", "aggregate_final_responses", "strict_valid", "structural_invalid", "retry_attempt_records", "raw_output_dir", "notes"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return rerank.rel_safe(path)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def format_float(value: float) -> str:
    return rerank.format_float(value)


def local_raw_output_dir(run_id: str) -> Path:
    return f6l.PROJECT_ROOT / "results" / "llm" / f"{EXECUTION_ID}_{run_id}"


def read_csv(path: Path) -> list[dict[str, str]]:
    return f6l.read_csv_rows(path)


def continuation_manifest_rows() -> list[dict[str, str]]:
    return read_csv(freeze.MANIFEST_PATH)


def base_manifest_by_id() -> dict[str, dict[str, str]]:
    return {row["request_id"]: row for row in read_csv(rerank.MANIFEST_PATH)}


def execution_manifest_rows(execution_run_id: str, raw_output_dir: Path) -> list[dict[str, str]]:
    base = base_manifest_by_id()
    rows = []
    for item in continuation_manifest_rows():
        row = dict(base[item["canonical_request_id"]])
        row["run_id"] = execution_run_id
        row["full_run_id"] = EXECUTION_ID
        row["expected_raw_output_path"] = rel(raw_output_dir / "responses" / f"{row['request_id']}.json")
        row["continuation_id"] = CONTINUATION_ID
        row["continuation_class"] = item["continuation_class"]
        row["prior_attempt_count"] = item["prior_attempt_count"]
        row["next_attempt_index"] = item["next_attempt_index"]
        row["prior_timeout_raw_json_path"] = item["prior_timeout_raw_json_path"]
        row["prior_timeout_raw_json_sha256"] = item["prior_timeout_raw_json_sha256"]
        rows.append(row)
    return rows


def original_parse_rows() -> list[dict[str, str]]:
    return read_csv(rerank.PARSE_RESULTS_PATH)


def original_recommendation_rows() -> list[dict[str, str]]:
    return read_csv(rerank.RECOMMENDATIONS_PATH)


def original_final_ids() -> set[str]:
    return {row["request_id"] for row in original_parse_rows()}


def raw_file_manifest_hash(path: Path) -> str:
    return freeze.raw_file_manifest_hash(path)


def run_focused_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        "results/logs/phase_f6l_lastfm_v4_1_execution_pytest",
        "-q",
        "tests/test_phase_f6l_lastfm_v4_reranking_continuation_v4_1.py",
        "tests/test_phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution.py",
    ]
    started = utc_now()
    start_clock = time.perf_counter()
    completed = subprocess.run(command, cwd=f6l.PROJECT_ROOT, check=False, capture_output=True, text=True, timeout=1000)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-20:]),
        "started_at": started,
        "completed_at": utc_now(),
        "duration_seconds": time.perf_counter() - start_clock,
    }


def validate_preflight(
    *,
    execution_run_id: str,
    raw_output_dir: Path,
    resume: bool,
    allow_api_calls: bool,
    local_only: bool,
    endpoint: str,
    run_tests: bool,
    check_runtime: bool,
) -> tuple[list[dict[str, str]], dict[str, Any], f6l.LocalOllamaState | None]:
    rows: list[dict[str, str]] = []
    test_result: dict[str, Any] = {}
    ollama_state: f6l.LocalOllamaState | None = None

    def add(name: str, passed: bool, observed: str, expected: str, evidence: str, notes: str, severity: str = "error") -> None:
        rows.append(
            {
                "check_id": f"F6L-LASTFM-V41-EXEC-PREFLIGHT-{len(rows) + 1:02d}",
                "check_name": name,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    continuation = continuation_manifest_rows()
    classes = Counter(row["continuation_class"] for row in continuation)
    ids = [row["canonical_request_id"] for row in continuation]
    final_ids = original_final_ids()
    policy_rows = read_csv(freeze.POLICY_AUDIT_PATH)
    post_summary = f6l.read_json_file(post.SUMMARY_PATH)
    freeze_summary = f6l.read_json_file(freeze.SUMMARY_PATH)
    raw_hash = raw_file_manifest_hash(post.RAW_OUTPUT_DIR)
    baseline_rows = rerank.ranking_baseline_audit_rows(write_outputs=False)
    exec_rows = execution_manifest_rows(execution_run_id, raw_output_dir)
    base_rows = base_manifest_by_id()
    schema_ok = all(row["schema_hash"] == rerank.EXPECTED_V4_SCHEMA_HASH for row in continuation)
    prompt_ok = all(row["rendered_prompt_hash"] == base_rows[row["canonical_request_id"]]["rendered_prompt_hash"] for row in continuation)
    provenance_ok = all(row["candidate_provenance"] == base_rows[row["canonical_request_id"]]["lineage"] for row in continuation)
    contamination = any(any(token in row["candidate_pool_source"].casefold() for token in ("v1", "v2", "v3", "pilot")) for row in continuation)
    raw_exists = raw_output_dir.exists()
    raw_ok = ((not resume) and (not raw_exists)) or (resume and raw_exists and (raw_output_dir / "run_manifest.json").exists())

    add("continuation_manifest_rows_1600", len(continuation) == 1600, str(len(continuation)), "1600", rel(freeze.MANIFEST_PATH), "Only frozen V4.1 continuation rows are executable.")
    add("canonical_request_ids_unique", len(ids) == len(set(ids)) == 1600, f"rows={len(ids)};unique={len(set(ids))}", "1600 unique", rel(freeze.MANIFEST_PATH), "Idempotency is by canonical request id.")
    add("timeout_retry_rows_48", classes["retry_after_non_final_transport_timeout"] == 48, str(classes["retry_after_non_final_transport_timeout"]), "48", rel(freeze.MANIFEST_PATH), "Prior non-final timeouts remain pending.")
    add("never_attempted_rows_1552", classes["first_attempt_never_attempted"] == 1552, str(classes["first_attempt_never_attempted"]), "1552", rel(freeze.MANIFEST_PATH), "Never-attempted rows are included.")
    add("prior_final_response_false_all", {row["prior_final_response_exists"] for row in continuation} == {"false"}, str(Counter(row["prior_final_response_exists"] for row in continuation)), "all false", rel(freeze.MANIFEST_PATH), "No row already has a final response.")
    add("no_pending_request_has_original_final_response", not (set(ids) & final_ids), str(len(set(ids) & final_ids)), "0", rel(rerank.PARSE_RESULTS_PATH), "Existing 2400 final responses are excluded.")
    add("pending_final_response_set_exact_and_exhaustive", len(continuation) == 1600 and not (set(ids) & final_ids), f"rows={len(continuation)};overlap_final={len(set(ids) & final_ids)}", "1600 rows;0 final-response overlap", rel(freeze.MANIFEST_PATH), "V4.1 target is all and only canonical requests without final response.")
    add("baseline_ranking_freeze_hash_invariant", rerank.RANKING_BASELINE_FREEZE_HASH == "D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC" and all(row["status"] == "pass" for row in baseline_rows), rerank.RANKING_BASELINE_FREEZE_HASH, "D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC", rel(rerank.RANKING_BASELINE_PATHS["hash_text"]), "Frozen ranking baseline is audited, not recomputed.")
    add("partial_raw_artifact_hash_invariant", raw_hash == post_summary["integrity"]["partial_raw_artifact_manifest_sha256"] == freeze_summary["raw_hash_before"], raw_hash, post_summary["integrity"]["partial_raw_artifact_manifest_sha256"], rel(post.RAW_OUTPUT_DIR), "Original V4 raw evidence is byte-invariant.")
    add("parser_hash_invariant", all(row["parser_source_hash"] == rerank.EXPECTED_V4_PARSER_HASH for row in continuation), rerank.EXPECTED_V4_PARSER_HASH, rerank.EXPECTED_V4_PARSER_HASH, rel(freeze.MANIFEST_PATH), "Parser remains V4.")
    add("schema_hash_invariant", schema_ok, rerank.EXPECTED_V4_SCHEMA_HASH, rerank.EXPECTED_V4_SCHEMA_HASH, rel(freeze.MANIFEST_PATH), "JSON schema remains V4.")
    add("prompt_contract_invariant", prompt_ok, "prompt hashes match base manifest", "match", rel(freeze.MANIFEST_PATH), "No prompt mutation is allowed.")
    add("candidate_provenance_integral", provenance_ok, "candidate provenance matches base manifest lineage", "match", rel(freeze.MANIFEST_PATH), "Candidate pool provenance remains canonical.")
    original_status = f6l.git_output(["status", "--short", "original_repo"]) if check_runtime else ""
    add("original_repo_clean", original_status == "", original_status or "clean", "clean", "git status --short original_repo", "original_repo is read-only.")
    add("previous_policy_audit_valid", policy_rows and all(row["status"] == "pass" for row in policy_rows), str(Counter(row["status"] for row in policy_rows)), "all pass", rel(freeze.POLICY_AUDIT_PATH), "V4.1 freeze policy audit remains valid.")
    add("no_v1_v2_v3_pilot_contamination", not contamination, "none" if not contamination else "contamination", "none", rel(freeze.MANIFEST_PATH), "No superseded source enters execution.")
    add("raw_output_dir_append_only", raw_ok, f"exists={raw_exists};resume={resume};dir={rel(raw_output_dir)}", "new dir or explicit resume with run_manifest.json", "CLI args", "Continuation uses a new raw directory, never the original V4 raw dir.")
    add("execution_manifest_matches_frozen_ids", [row["request_id"] for row in exec_rows] == ids, "same canonical id order", "same", rel(freeze.MANIFEST_PATH), "Execution rows are derived from frozen continuation rows.")

    test_ok = True
    if run_tests:
        test_result = run_focused_tests()
        test_ok = test_result["exit_code"] == 0
    endpoint_ok = local_only and endpoint in {f6l.OLLAMA_LOCAL_GENERATE_URL, "http://127.0.0.1:11434/api/generate", "http://localhost:11434/api/generate"}
    staged = f6l.git_output(["diff", "--cached", "--name-only"]) if check_runtime else ""
    runtime_ok = allow_api_calls and endpoint_ok and staged == "" and original_status == "" and test_ok
    ollama_observed = "not_checked"
    if check_runtime and all(row["status"] == "pass" for row in rows) and runtime_ok:
        ollama_state = f6l.start_local_ollama_if_needed(execution_run_id)
        digest_ok = ollama_state.service_available and all(
            ollama_state.model_digests.get(model, "").lower() == f6l.MODEL_DIGESTS[model].lower()
            for model in rerank.MODEL_ORDER
        )
        runtime_ok = runtime_ok and digest_ok
        ollama_observed = canonical_json({model: ollama_state.model_digests.get(model, "") for model in rerank.MODEL_ORDER})
    elif not check_runtime:
        ollama_observed = "skipped"
    add(
        "runtime_guardrails",
        runtime_ok,
        canonical_json({"allow_api_calls": allow_api_calls, "endpoint_ok": endpoint_ok, "staged": staged or "none", "focused_tests": test_ok, "ollama": ollama_observed}),
        "allow=true;loopback=true;tests_pass;digests_match",
        "CLI args;pytest;Ollama tags",
        "Final gate before local continuation inference.",
    )
    f6l.write_csv(PREFLIGHT_PATH, rows, PREFLIGHT_FIELDS)
    return rows, test_result, ollama_state


def request_payload_for_row(row: Mapping[str, str]) -> dict[str, Any]:
    payload = rerank.request_payload_for_row(row)
    payload.update(
        {
            "continuation_id": CONTINUATION_ID,
            "base_run_id": BASE_RUN_ID,
            "continuation_class": row.get("continuation_class", ""),
            "prior_attempt_count": row.get("prior_attempt_count", ""),
        }
    )
    return payload


def raw_record_for_response(
    row: Mapping[str, str],
    prompt: str,
    result: f6l.LocalOllamaRequestResult,
    parse: v4.ParserV4Result,
    raw_json_path: Path,
    *,
    attempt_index: int,
    continuation_attempt_number: int,
    is_final_attempt: bool,
) -> dict[str, Any]:
    raw = rerank.raw_record_for_response(row, prompt, result, parse, raw_json_path, attempt_index=attempt_index, is_final_attempt=is_final_attempt)
    raw.update(
        {
            "continuation_id": CONTINUATION_ID,
            "base_run_id": BASE_RUN_ID,
            "continuation_execution": True,
            "continuation_class": row.get("continuation_class", ""),
            "continuation_attempt_number": continuation_attempt_number,
            "prior_attempt_count": row.get("prior_attempt_count", ""),
            "prior_timeout_raw_json_path": row.get("prior_timeout_raw_json_path", ""),
            "prior_timeout_raw_json_sha256": row.get("prior_timeout_raw_json_sha256", ""),
        }
    )
    return raw


def parse_result_row(row: Mapping[str, str], raw_record: Mapping[str, Any]) -> dict[str, str]:
    parsed = rerank.parse_result_row(row, raw_record)
    parsed["full_run_id"] = EXECUTION_ID
    parsed["run_id"] = row["run_id"]
    parsed["notes"] = "V4.1 continuation parser result; duplicate_token remains fail-strict structural-invalid."
    return parsed


def recommendation_rows_from_parse(row: Mapping[str, str], parse_row: Mapping[str, str]) -> list[dict[str, str]]:
    recs = rerank.recommendation_rows_from_parse(row, parse_row)
    for rec in recs:
        rec["full_run_id"] = EXECUTION_ID
        rec["run_id"] = row["run_id"]
        rec["notes"] = "V4.1 continuation recommendation row derived from a strict-valid final response."
    return recs


def execution_row(row: Mapping[str, str], raw_record: Mapping[str, Any]) -> dict[str, str]:
    out = rerank.execution_row(row, raw_record)
    out["full_run_id"] = EXECUTION_ID
    out["run_id"] = row["run_id"]
    out["notes"] = f"V4.1 append-only continuation attempt; continuation_attempt_number={raw_record.get('continuation_attempt_number','')}"
    return out


def aggregate_cell_rows(continuation_parse_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    combined = [dict(row) for row in original_parse_rows()] + [dict(row) for row in continuation_parse_rows]
    order = {row["request_id"]: index for index, row in enumerate(read_csv(rerank.MANIFEST_PATH), start=1)}
    combined.sort(key=lambda row: order[row["request_id"]])
    return combined


def cell_breaker_state(parse_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], str, str, float, float]:
    rows: list[dict[str, str]] = []
    breaker = ""
    breaker_cell = ""
    current_window_rate = 0.0
    current_cumulative_rate = 0.0
    for model in rerank.MODEL_ORDER:
        for strategy in rerank.STRATEGY_ORDER:
            cell = [row for row in parse_rows if row["model_name"] == model and row["prompting_strategy"] == strategy]
            cumulative_dup = 0
            window_rates: list[float] = []
            closed_rows = []
            for index, start in enumerate(range(0, len(cell), rerank.DUPLICATE_WINDOW_SIZE), start=1):
                window = cell[start : start + rerank.DUPLICATE_WINDOW_SIZE]
                if len(window) < rerank.DUPLICATE_WINDOW_SIZE:
                    continue
                dup = sum(1 for row in window if row["failure_reason"] == "duplicate_token")
                cumulative_dup = sum(1 for row in cell[: start + len(window)] if row["failure_reason"] == "duplicate_token")
                rate = dup / len(window)
                cumulative_rate = cumulative_dup / (start + len(window))
                window_rates.append(rate)
                status = "fail" if rate > 0.05 else "pass"
                closed_rows.append(
                    {
                        "run_id": "",
                        "model_name": model,
                        "prompting_strategy": strategy,
                        "window_index": str(index),
                        "window_start_cell_final": str(start + 1),
                        "window_end_cell_final": str(start + len(window)),
                        "closed": "true",
                        "responses": str(len(window)),
                        "duplicate_token_invalid": str(dup),
                        "duplicate_token_rate": format_float(rate),
                        "cumulative_responses": str(start + len(window)),
                        "cumulative_duplicate_token_invalid": str(cumulative_dup),
                        "cumulative_duplicate_token_rate": format_float(cumulative_rate),
                        "status": status,
                        "breaker_rule": "same_cell_closed_200_duplicate_rate_gt_0.05",
                        "notes": "V4.1 duplicate-token monitoring by model/strategy cell over aggregate canonical final responses.",
                    }
                )
            rows.extend(closed_rows)
            total_dup = sum(1 for row in cell if row["failure_reason"] == "duplicate_token")
            total_rate = total_dup / len(cell) if cell else 0.0
            if cell:
                current_cumulative_rate = total_rate
                current_window_rate = window_rates[-1] if window_rates else 0.0
            if len(window_rates) >= 2 and window_rates[-1] > 0.05 and window_rates[-2] > 0.05 and not breaker:
                breaker = "two_consecutive_same_cell_closed_200_windows_duplicate_rate_above_0.05"
                breaker_cell = f"{model}|{strategy}"
            if len(cell) >= 400 and total_rate > 0.05 and not breaker:
                breaker = "same_cell_cumulative_duplicate_rate_above_0.05_after_400_final_responses"
                breaker_cell = f"{model}|{strategy}"
    return rows, breaker, breaker_cell, current_window_rate, current_cumulative_rate


def write_state(
    *,
    execution_run_id: str,
    raw_output_dir: Path,
    stage: str,
    decision_token: str,
    manifest_rows: Sequence[Mapping[str, str]],
    parse_rows: Sequence[Mapping[str, str]],
    attempt_counts: Mapping[str, int],
    canary_status: str,
    current_cell: str,
    current_window_rate: float,
    current_cumulative_rate: float,
    breaker_state: str,
    last_request_id: str,
    notes: str,
) -> None:
    strict = sum(1 for row in parse_rows if row["strict_valid"] == "true")
    invalid = len(parse_rows) - strict
    duplicate = sum(1 for row in parse_rows if row["failure_reason"] == "duplicate_token")
    non_final_timeout = sum(max(0, count) for count in attempt_counts.values()) - len(parse_rows)
    row = {
        "execution_run_id": execution_run_id,
        "continuation_id": CONTINUATION_ID,
        "stage": stage,
        "decision_token": decision_token,
        "planned": str(len(manifest_rows)),
        "completed_final_responses": str(len(parse_rows)),
        "residual": str(len(manifest_rows) - len(parse_rows)),
        "strict_valid": str(strict),
        "structural_invalid": str(invalid),
        "duplicate_token_invalid": str(duplicate),
        "non_final_transport_timeout": str(non_final_timeout),
        "retry_attempt_records": str(sum(max(0, count - 1) for count in attempt_counts.values())),
        "canary_status": canary_status,
        "current_cell": current_cell,
        "current_cell_closed_window_duplicate_rate": format_float(current_window_rate),
        "current_cell_cumulative_duplicate_rate": format_float(current_cumulative_rate),
        "breaker_state": breaker_state,
        "last_canonical_request_id": last_request_id,
        "last_timestamp": utc_now(),
        "raw_output_dir": rel(raw_output_dir),
        "notes": notes,
    }
    f6l.write_csv(STATE_PATH, [row], STATE_FIELDS)


def build_failure_rows(parse_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return rerank.build_failure_rows(parse_rows)


def split_metrics(metric_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    fail_metrics = []
    complete_metrics = []
    validity = []
    for row in metric_rows:
        if row["metric"] in {"HR@5_fail_strict", "NDCG@5_fail_strict", "mean_positive_rank_fail_strict_invalid_as_20"}:
            fail_metrics.append(dict(row))
        elif row["metric"] in {"HR@5_positive_present_only", "NDCG@5_positive_present_only"}:
            complete_metrics.append(dict(row))
        elif row["metric"] in {"strict_valid_rate", "structural_invalid_rate", "duplicate_token_rate", "retry_rate", "malformed_json_rate", "hallucination_rate", "positive_present_rate"}:
            validity.append(dict(row))
    return fail_metrics, complete_metrics, validity


def final_validation_rows(
    *,
    decision: str,
    manifest_rows: Sequence[Mapping[str, str]],
    continuation_parse_rows: Sequence[Mapping[str, str]],
    aggregate_parse_rows: Sequence[Mapping[str, str]],
    raw_hash_before: str,
    raw_hash_after: str,
) -> list[dict[str, str]]:
    continuation_ids = {row["request_id"] for row in continuation_parse_rows}
    aggregate_ids = [row["request_id"] for row in aggregate_parse_rows]
    original_ids = {row["request_id"] for row in original_parse_rows()}

    checks = [
        ("continuation_manifest_1600", len(manifest_rows) == 1600, str(len(manifest_rows)), "1600", rel(freeze.MANIFEST_PATH), "Frozen continuation manifest size."),
        ("continuation_final_responses_1600_if_success", decision != DECISION_COMPLETE or len(continuation_parse_rows) == 1600, str(len(continuation_parse_rows)), "1600 on success", rel(PARSE_RESULTS_PATH), "Success requires all continuation rows final."),
        ("original_final_responses_2400", len(original_ids) == 2400, str(len(original_ids)), "2400", rel(rerank.PARSE_RESULTS_PATH), "Original final responses are reused by provenance only."),
        ("aggregate_unique_ids", len(aggregate_ids) == len(set(aggregate_ids)), f"rows={len(aggregate_ids)};unique={len(set(aggregate_ids))}", "unique", rel(AGGREGATE_PARSE_RESULTS_PATH), "Exactly one final response per canonical request."),
        ("aggregate_4000_if_success", decision != DECISION_COMPLETE or len(aggregate_parse_rows) == 4000, str(len(aggregate_parse_rows)), "4000 on success", rel(AGGREGATE_PARSE_RESULTS_PATH), "Success requires 2400+1600 final rows."),
        ("no_overlap_original_continuation", not (original_ids & continuation_ids), str(len(original_ids & continuation_ids)), "0", rel(PARSE_RESULTS_PATH), "No existing final response is rerun."),
        ("raw_original_preserved", raw_hash_before == raw_hash_after, raw_hash_after, raw_hash_before, rel(post.RAW_OUTPUT_DIR), "Original V4 raw directory is byte-preserved."),
        ("parser_schema_invariant", all(row["parser_source_hash"] == rerank.EXPECTED_V4_PARSER_HASH for row in continuation_parse_rows), rerank.EXPECTED_V4_PARSER_HASH, rerank.EXPECTED_V4_PARSER_HASH, rel(PARSE_RESULTS_PATH), "Parser is unchanged."),
        ("ranking_freeze_hash_invariant", rerank.RANKING_BASELINE_FREEZE_HASH == "D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC", rerank.RANKING_BASELINE_FREEZE_HASH, "D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC", rel(rerank.RANKING_BASELINE_PATHS["hash_text"]), "Ranking baseline is frozen."),
        ("original_repo_clean", f6l.git_output(["status", "--short", "original_repo"]) == "", f6l.git_output(["status", "--short", "original_repo"]) or "clean", "clean", "git status --short original_repo", "No original_repo modification."),
    ]
    return [
        {
            "check_name": name,
            "status": "pass" if passed else "fail",
            "observed": observed,
            "expected": expected,
            "evidence": evidence,
            "notes": notes,
        }
        for name, passed, observed, expected, evidence, notes in checks
    ]


def statistical_rows(run_id: str, ranking_comparisons: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in ranking_comparisons:
        if row["metric"] not in {"HR@5_fail_strict", "NDCG@5_fail_strict"}:
            continue
        rows.append(
            {
                "test_name": "descriptive_delta_no_paired_test_in_runner",
                "model_name": row["model_name"],
                "prompting_strategy": row["reranking_strategy"],
                "metric": row["metric"],
                "observed": row["delta_reranking_minus_ranking"],
                "expected": "reranking minus frozen ranking",
                "p_value": "",
                "status": "descriptive",
                "notes": "Paired tests require final per-user contrast extraction; denominators are preserved in metric tables.",
            }
        )
    return rows


def consolidate(
    *,
    execution_run_id: str,
    raw_output_dir: Path,
    manifest_rows: Sequence[Mapping[str, str]],
    preflight_rows: Sequence[Mapping[str, str]],
    focused_test_result: Mapping[str, Any],
    started_at: str,
    decision: str | None = None,
) -> tuple[str, dict[str, Any]]:
    paths = f6l.raw_response_paths(raw_output_dir)
    final_records = f6l.final_response_records(raw_output_dir)
    all_records = f6l.read_jsonl(paths["responses"])
    attempt_counts = Counter(str(row.get("request_id", "")) for row in all_records if row.get("request_id"))
    manifest_by_id = {row["request_id"]: row for row in manifest_rows}
    continuation_parse = [
        parse_result_row(manifest_by_id[request_id], record)
        for request_id, record in final_records.items()
        if request_id in manifest_by_id
    ]
    order = {row["request_id"]: index for index, row in enumerate(manifest_rows)}
    continuation_parse.sort(key=lambda row: order[row["request_id"]])
    continuation_recs = [
        rec
        for row in continuation_parse
        for rec in recommendation_rows_from_parse(manifest_by_id[row["request_id"]], row)
    ]
    aggregate_parse = aggregate_cell_rows(continuation_parse)
    aggregate_recs = original_recommendation_rows() + continuation_recs
    aggregate_attempt_counts = Counter({row["request_id"]: 1 for row in original_parse_rows()})
    aggregate_attempt_counts.update(attempt_counts)
    metric_rows, beyond_rows, strategy_comparisons = rerank.compute_metrics(execution_run_id, aggregate_parse, aggregate_recs, read_csv(rerank.MANIFEST_PATH), aggregate_attempt_counts)
    fail_metrics, complete_metrics, validity_metric_rows = split_metrics(metric_rows)
    ranking_comparisons = rerank.ranking_baseline_comparison_rows(execution_run_id, metric_rows)
    duplicate_rows, dup_breaker, dup_cell, _, _ = cell_breaker_state(aggregate_parse)
    for row in duplicate_rows:
        row["run_id"] = execution_run_id
    failure_rows = build_failure_rows(continuation_parse)
    execution_rows = [execution_row(manifest_by_id[str(record.get("request_id", ""))], record) for record in all_records if str(record.get("request_id", "")) in manifest_by_id]
    completed = len(continuation_parse)
    strict = sum(1 for row in continuation_parse if row["strict_valid"] == "true")
    invalid = completed - strict
    raw_hash_before = f6l.read_json_file(freeze.SUMMARY_PATH)["raw_hash_before"]
    raw_hash_after = raw_file_manifest_hash(post.RAW_OUTPUT_DIR)
    if decision is None:
        if completed == len(manifest_rows):
            decision = DECISION_COMPLETE
        else:
            decision = DECISION_PARTIAL_TRANSPORT
    validation = final_validation_rows(
        decision=decision,
        manifest_rows=manifest_rows,
        continuation_parse_rows=continuation_parse,
        aggregate_parse_rows=aggregate_parse,
        raw_hash_before=raw_hash_before,
        raw_hash_after=raw_hash_after,
    )
    completed_at = utc_now()
    duration = max(0.0, datetime.fromisoformat(completed_at.replace("Z", "+00:00")).timestamp() - datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp())
    runtime_rows = [
        {
            "execution_run_id": execution_run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": format_float(duration),
            "planned": str(len(manifest_rows)),
            "continuation_final_responses": str(completed),
            "aggregate_final_responses": str(len(aggregate_parse)),
            "strict_valid": str(strict),
            "structural_invalid": str(invalid),
            "retry_attempt_records": str(sum(max(0, count - 1) for count in attempt_counts.values())),
            "raw_output_dir": rel(raw_output_dir),
            "notes": "Runtime for V4.1 continuation process only.",
        }
    ]
    f6l.write_csv(ATTEMPT_LOG_PATH, execution_rows, rerank.EXECUTION_FIELDS)
    f6l.write_csv(PARSE_RESULTS_PATH, continuation_parse, rerank.PARSE_FIELDS)
    f6l.write_csv(AGGREGATE_PARSE_RESULTS_PATH, aggregate_parse, rerank.PARSE_FIELDS)
    f6l.write_csv(RECOMMENDATIONS_PATH, continuation_recs, rerank.RECOMMENDATION_FIELDS)
    f6l.write_csv(AGGREGATE_RECOMMENDATIONS_PATH, aggregate_recs, rerank.RECOMMENDATION_FIELDS)
    f6l.write_csv(FAILURE_TAXONOMY_PATH, failure_rows, rerank.FAILURE_TAXONOMY_FIELDS)
    f6l.write_csv(DUPLICATE_WINDOWS_PATH, duplicate_rows, DUPLICATE_WINDOW_FIELDS)
    f6l.write_csv(VALIDITY_PATH, validity_metric_rows, rerank.METRIC_FIELDS)
    f6l.write_csv(FAIL_STRICT_METRICS_PATH, fail_metrics, rerank.METRIC_FIELDS)
    f6l.write_csv(COMPLETE_CASE_METRICS_PATH, complete_metrics, rerank.METRIC_FIELDS)
    f6l.write_csv(BEYOND_UTILITY_PATH, beyond_rows, rerank.BEYOND_FIELDS)
    f6l.write_csv(RANKING_COMPARISON_PATH, ranking_comparisons, rerank.RANKING_BASELINE_COMPARISON_FIELDS)
    f6l.write_csv(STATISTICAL_TESTS_PATH, statistical_rows(execution_run_id, ranking_comparisons), STAT_FIELDS)
    f6l.write_csv(FINAL_VALIDATION_PATH, validation, FINAL_VALIDATION_FIELDS)
    f6l.write_csv(RUNTIME_PATH, runtime_rows, RUNTIME_FIELDS)
    output_paths = [
        PREFLIGHT_PATH,
        STATE_PATH,
        ATTEMPT_LOG_PATH,
        PARSE_RESULTS_PATH,
        AGGREGATE_PARSE_RESULTS_PATH,
        FAILURE_TAXONOMY_PATH,
        DUPLICATE_WINDOWS_PATH,
        VALIDITY_PATH,
        FAIL_STRICT_METRICS_PATH,
        COMPLETE_CASE_METRICS_PATH,
        BEYOND_UTILITY_PATH,
        RANKING_COMPARISON_PATH,
        STATISTICAL_TESTS_PATH,
        FINAL_VALIDATION_PATH,
        RUNTIME_PATH,
        REPORT_PATH,
    ]
    summary = {
        "execution_run_id": execution_run_id,
        "continuation_id": CONTINUATION_ID,
        "decision_token": decision,
        "planned": len(manifest_rows),
        "continuation_final_responses": completed,
        "original_final_responses": len(original_parse_rows()),
        "aggregate_final_responses": len(aggregate_parse),
        "strict_valid": strict,
        "structural_invalid": invalid,
        "duplicate_token_invalid": sum(1 for row in continuation_parse if row["failure_reason"] == "duplicate_token"),
        "raw_output_dir": rel(raw_output_dir),
        "preflight_pass": all(row["status"] == "pass" for row in preflight_rows),
        "focused_tests": focused_test_result,
        "raw_original_hash_preserved": raw_hash_before == raw_hash_after,
        "duplicate_breaker_state": dup_breaker,
        "duplicate_breaker_cell": dup_cell,
        "output_paths": {path.name: rel(path) for path in output_paths},
    }
    f6l.write_json(SUMMARY_PATH, summary)
    f6l.write_json(paths["manifest"], {**summary, "completed_at": completed_at})
    write_report(summary)
    hashes = {
        "original_raw_hash_before": raw_hash_before,
        "original_raw_hash_after": raw_hash_after,
        "original_raw_preserved": raw_hash_before == raw_hash_after,
        "continuation_raw_hash": raw_file_manifest_hash(raw_output_dir) if raw_output_dir.exists() else "",
        "output_artifacts": {rel(path): f6l.safe_file_sha256(path) for path in [*output_paths, SUMMARY_PATH] if path.exists()},
    }
    f6l.write_json(HASHES_PATH, hashes)
    return decision, summary


def write_report(summary: Mapping[str, Any]) -> None:
    lines = [
        "# F6L LastFM V4.1 Reranking Continuation Execution",
        "",
        f"Decision token: `{summary['decision_token']}`.",
        "",
        f"- Execution run id: `{summary['execution_run_id']}`.",
        f"- Continuation id: `{CONTINUATION_ID}`.",
        f"- Raw output dir: `{summary['raw_output_dir']}`.",
        f"- Continuation final responses: `{summary['continuation_final_responses']}/1600`.",
        f"- Aggregate canonical final responses: `{summary['aggregate_final_responses']}/4000`.",
        f"- Strict-valid: `{summary['strict_valid']}`; structural-invalid: `{summary['structural_invalid']}`; duplicate-token invalid: `{summary['duplicate_token_invalid']}`.",
        f"- Original raw preserved: `{str(summary['raw_original_hash_preserved']).lower()}`.",
        "",
        "No raw output from the original V4 run is copied or overwritten.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if not args.allow_api_calls:
        raise SystemExit("--allow-api-calls is required for the explicitly authorized local continuation inference")
    if not args.local_only:
        raise SystemExit("--local-only is required")
    execution_run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_dir = Path(args.raw_output_dir) if args.raw_output_dir else local_raw_output_dir(execution_run_id)
    if not raw_output_dir.is_absolute():
        raw_output_dir = (f6l.PROJECT_ROOT / raw_output_dir).resolve()
    manifest_rows = execution_manifest_rows(execution_run_id, raw_output_dir)
    preflight_rows, focused_tests, ollama_state = validate_preflight(
        execution_run_id=execution_run_id,
        raw_output_dir=raw_output_dir,
        resume=args.resume,
        allow_api_calls=args.allow_api_calls,
        local_only=args.local_only,
        endpoint=str(args.ollama_endpoint),
        run_tests=not args.skip_focused_tests,
        check_runtime=True,
    )
    preflight_pass = all(row["status"] == "pass" for row in preflight_rows)
    if args.preflight_only or not preflight_pass or ollama_state is None:
        decision = DECISION_EXECUTING if preflight_pass else DECISION_BLOCKED_PREFLIGHT
        if args.preflight_only and preflight_pass:
            decision = freeze.DECISION_READY
        write_state(
            execution_run_id=execution_run_id,
            raw_output_dir=raw_output_dir,
            stage="preflight_passed" if preflight_pass else "blocked_preflight",
            decision_token=decision,
            manifest_rows=manifest_rows,
            parse_rows=[],
            attempt_counts={},
            canary_status="not_started",
            current_cell="",
            current_window_rate=0.0,
            current_cumulative_rate=0.0,
            breaker_state="clear" if preflight_pass else "precondition_failure",
            last_request_id="",
            notes="preflight-only; no inference executed" if args.preflight_only else "preflight failed; no inference executed",
        )
        summary = {"execution_run_id": execution_run_id, "continuation_id": CONTINUATION_ID, "decision_token": decision, "preflight_pass": preflight_pass, "raw_output_dir": rel(raw_output_dir)}
        f6l.write_json(SUMMARY_PATH, summary)
        return decision, summary

    paths = f6l.raw_response_paths(raw_output_dir)
    if args.resume:
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        paths["individual"].mkdir(parents=True, exist_ok=True)
        if not paths["manifest"].exists():
            raise SystemExit("resume requested but run_manifest.json is missing")
    else:
        raw_output_dir.mkdir(parents=True, exist_ok=False)
        paths["individual"].mkdir(parents=True, exist_ok=True)
        paths["command"].write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    started_at = utc_now()
    if not args.resume:
        f6l.write_json(
            paths["manifest"],
            {
                "execution_run_id": execution_run_id,
                "continuation_id": CONTINUATION_ID,
                "decision_token": DECISION_EXECUTING,
                "started_at": started_at,
                "raw_output_dir": rel(raw_output_dir),
                "base_run_id": BASE_RUN_ID,
                "ollama_service_version": ollama_state.service_version,
                "ollama_start_attempted": ollama_state.start_attempted,
                "ollama_started_pid": ollama_state.started_pid,
                "focused_tests": focused_tests,
            },
        )
    sample = rerank.sample_rows_by_user()
    candidates = rerank.reranking_pool_rows_by_user()
    manifest_by_id = {row["request_id"]: row for row in manifest_rows}
    final_records = f6l.final_response_records(raw_output_dir)
    all_records = f6l.read_jsonl(paths["responses"])
    attempt_counts: Counter[str] = Counter(str(row.get("request_id", "")) for row in all_records if row.get("request_id"))
    state_parse_rows = [
        parse_result_row(manifest_by_id[request_id], record)
        for request_id, record in final_records.items()
        if request_id in manifest_by_id
    ]
    order = {row["request_id"]: index for index, row in enumerate(manifest_rows)}
    state_parse_rows.sort(key=lambda row: order[row["request_id"]])
    write_state(
        execution_run_id=execution_run_id,
        raw_output_dir=raw_output_dir,
        stage="executing",
        decision_token=DECISION_EXECUTING,
        manifest_rows=manifest_rows,
        parse_rows=state_parse_rows,
        attempt_counts=attempt_counts,
        canary_status="pass" if len(state_parse_rows) >= 40 else "pending",
        current_cell="",
        current_window_rate=0.0,
        current_cumulative_rate=0.0,
        breaker_state="clear",
        last_request_id=state_parse_rows[-1]["request_id"] if state_parse_rows else "",
        notes="starting or resuming V4.1 continuation",
    )

    for row in manifest_rows:
        if row["request_id"] in final_records:
            continue
        prompt = rerank.render_prompt({**sample[row["user_id"]], "_subset": row["subset"]}, candidates[row["user_id"]], row["prompting_strategy"])
        hard_contract_mismatch = (
            f6l.sha256_text(prompt) != row["rendered_prompt_hash"]
            or row["parser_source_hash"] != rerank.EXPECTED_V4_PARSER_HASH
            or row["output_contract_hash"] != rerank.EXPECTED_V4_OUTPUT_CONTRACT_HASH
            or row["prompt_template_hash"] != rerank.prompt_template_hash(row["prompting_strategy"])
        )
        if hard_contract_mismatch:
            return consolidate(execution_run_id=execution_run_id, raw_output_dir=raw_output_dir, manifest_rows=manifest_rows, preflight_rows=preflight_rows, focused_test_result=focused_tests, started_at=started_at, decision=DECISION_HARD_STOP)
        local_attempt_number = attempt_counts[row["request_id"]] + 1
        canonical_attempt_index = int(row.get("prior_attempt_count", "0") or 0) + local_attempt_number
        request_record = {
            **request_payload_for_row(row),
            "prompt": prompt,
            "endpoint": str(args.ollama_endpoint),
            "queued_at": utc_now(),
            "continuation_attempt_number": local_attempt_number,
            "attempt_index": canonical_attempt_index,
            "expected_raw_output_path": rel(paths["individual"] / f"{row['request_id']}.json"),
        }
        f6l.append_jsonl_fsync(paths["requests"], request_record)
        result = v4.execute_local_ollama_schema_request(row["model_name"], prompt, str(args.ollama_endpoint))
        attempt_counts[row["request_id"]] += 1
        if not result.backend_success:
            parse = v4.ParserV4Result(False, "transport_error", [], [], [], 0, 0, 0, 10, 0, 0, 0, None)
            raw_json_path = paths["individual"] / f"{row['request_id']}__continuation_attempt{canonical_attempt_index:03d}.json"
            raw_record = raw_record_for_response(row, prompt, result, parse, raw_json_path, attempt_index=canonical_attempt_index, continuation_attempt_number=local_attempt_number, is_final_attempt=False)
            raw_json_path.write_text(json.dumps(raw_record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            f6l.append_jsonl_fsync(paths["responses"], raw_record)
            write_state(
                execution_run_id=execution_run_id,
                raw_output_dir=raw_output_dir,
                stage="executing_with_non_final_transport_timeout",
                decision_token=DECISION_EXECUTING,
                manifest_rows=manifest_rows,
                parse_rows=state_parse_rows,
                attempt_counts=attempt_counts,
                canary_status="pass" if len(state_parse_rows) >= 40 else "pending",
                current_cell=f"{row['model_name']}|{row['prompting_strategy']}",
                current_window_rate=0.0,
                current_cumulative_rate=0.0,
                breaker_state="clear",
                last_request_id=row["request_id"],
                notes=f"non-final transport timeout preserved append-only: {result.backend_error_type}",
            )
            continue
        raw_json_path = paths["individual"] / f"{row['request_id']}.json"
        if raw_json_path.exists():
            raise RuntimeError(f"refusing to overwrite existing final continuation raw response: {raw_json_path}")
        parse = v4.parse_v4_response(result.response_text, candidates[row["user_id"]])
        raw_record = raw_record_for_response(row, prompt, result, parse, raw_json_path, attempt_index=canonical_attempt_index, continuation_attempt_number=local_attempt_number, is_final_attempt=True)
        raw_json_path.write_text(json.dumps(raw_record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        f6l.append_jsonl_fsync(paths["responses"], raw_record)
        final_records[row["request_id"]] = raw_record
        parse_row = parse_result_row(row, raw_record)
        state_parse_rows.append(parse_row)
        state_parse_rows.sort(key=lambda item: order[item["request_id"]])
        aggregate_parse = aggregate_cell_rows(state_parse_rows)
        duplicate_windows, dup_breaker, dup_cell, current_window_rate, current_cumulative_rate = cell_breaker_state(aggregate_parse)
        hard_breaker = ""
        if parse_row["hash_coherence"] != "true":
            hard_breaker = "raw_hash_coherence_failure"
        elif rerank.is_hard_structural_invalid(parse_row):
            hard_breaker = parse_row["failure_reason"] or "hard_structural_invalid"
        canary_status = "pending"
        if len(state_parse_rows) >= 40:
            first_40 = state_parse_rows[:40]
            canary_hard = any(row["hash_coherence"] != "true" or rerank.is_hard_structural_invalid(row) for row in first_40)
            canary_status = "not_passed" if canary_hard else "pass"
            if canary_hard:
                hard_breaker = hard_breaker or "canary_hard_contract_failure"
        if hard_breaker or dup_breaker:
            decision = DECISION_HARD_STOP if hard_breaker else DECISION_DUP_STOP
            breaker_state = hard_breaker or dup_breaker
            write_state(
                execution_run_id=execution_run_id,
                raw_output_dir=raw_output_dir,
                stage="stopped",
                decision_token=decision,
                manifest_rows=manifest_rows,
                parse_rows=state_parse_rows,
                attempt_counts=attempt_counts,
                canary_status=canary_status,
                current_cell=dup_cell or f"{row['model_name']}|{row['prompting_strategy']}",
                current_window_rate=current_window_rate,
                current_cumulative_rate=current_cumulative_rate,
                breaker_state=breaker_state,
                last_request_id=row["request_id"],
                notes="V4.1 breaker triggered",
            )
            return consolidate(execution_run_id=execution_run_id, raw_output_dir=raw_output_dir, manifest_rows=manifest_rows, preflight_rows=preflight_rows, focused_test_result=focused_tests, started_at=started_at, decision=decision)
        write_state(
            execution_run_id=execution_run_id,
            raw_output_dir=raw_output_dir,
            stage="executing",
            decision_token=DECISION_EXECUTING,
            manifest_rows=manifest_rows,
            parse_rows=state_parse_rows,
            attempt_counts=attempt_counts,
            canary_status=canary_status,
            current_cell=f"{row['model_name']}|{row['prompting_strategy']}",
            current_window_rate=current_window_rate,
            current_cumulative_rate=current_cumulative_rate,
            breaker_state="clear",
            last_request_id=row["request_id"],
            notes="V4.1 continuation running",
        )
    return consolidate(execution_run_id=execution_run_id, raw_output_dir=raw_output_dir, manifest_rows=manifest_rows, preflight_rows=preflight_rows, focused_test_result=focused_tests, started_at=started_at)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or preflight the F6L LastFM V4.1 reranking continuation.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--raw-output-dir", default="")
    parser.add_argument("--allow-api-calls", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--ollama-endpoint", default=f6l.OLLAMA_LOCAL_GENERATE_URL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-focused-tests", action="store_true")
    args = parser.parse_args()
    decision, summary = execute(args)
    print(canonical_json({"decision_token": decision, "execution_run_id": summary.get("execution_run_id", args.run_id), "raw_output_dir": summary.get("raw_output_dir", ""), "continuation_final_responses": summary.get("continuation_final_responses", 0)}))


if __name__ == "__main__":
    main()

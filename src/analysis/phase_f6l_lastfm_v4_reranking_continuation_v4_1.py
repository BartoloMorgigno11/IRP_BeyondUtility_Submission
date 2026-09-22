"""Freeze the F6L LastFM V4.1 reranking continuation manifest.

The V4.1 continuation is a proposal/freeze only. It targets canonical requests
from run 20260806_034027 that have no final response, including prior non-final
transport timeouts. This module never calls Ollama and never writes raw output.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.analysis import phase_f6l_lastfm_v4_reranking_full_run as rerank
from src.analysis import phase_f6l_lastfm_v4_reranking_post_breaker as post
from src.analysis import plan_f6l_paper_faithful_completion_and_rq_revalidation as f6l


RUN_ID = "20260806_034027"
CONTINUATION_ID = "phase_f6l_lastfm_v4_reranking_continuation_v4_1_20260807"
DECISION_READY = "f6l_lastfm_v4_reranking_post_breaker_versioned_continuation_ready"
DECISION_BLOCKED = "f6l_lastfm_v4_reranking_post_breaker_blocked_pending_evidence"

CONFIG_PATH = f6l.PROJECT_ROOT / "configs" / "phase_f6l_lastfm_v4_reranking_continuation_v4_1.yaml"
MANIFEST_PATH = f6l.PROJECT_ROOT / "results" / "tables" / "phase_f6l_lastfm_v4_reranking_continuation_v4_1_manifest.csv"
POLICY_AUDIT_PATH = f6l.PROJECT_ROOT / "results" / "tables" / "phase_f6l_lastfm_v4_reranking_continuation_v4_1_policy_audit.csv"
SUMMARY_PATH = f6l.PROJECT_ROOT / "results" / "manifests" / "phase_f6l_lastfm_v4_reranking_continuation_v4_1_summary.json"
HASHES_PATH = f6l.PROJECT_ROOT / "results" / "manifests" / "phase_f6l_lastfm_v4_reranking_continuation_v4_1_hashes.json"
REPORT_PATH = f6l.PROJECT_ROOT / "report" / "phase_f6l_lastfm_v4_reranking_continuation_v4_1_freeze.md"

MANIFEST_FIELDS = [
    "continuation_id",
    "continuation_request_id",
    "base_run_id",
    "canonical_request_id",
    "original_request_id",
    "canonical_manifest_ordinal",
    "user_id",
    "model_name",
    "model_digest",
    "prompting_strategy",
    "candidate_pool_source",
    "candidate_pool_hash",
    "candidate_list_hash",
    "candidate_token_mapping_hash",
    "candidate_provenance",
    "rendered_prompt_hash",
    "prompt_template_hash",
    "parser_version",
    "parser_source_hash",
    "schema_hash",
    "output_contract_hash",
    "continuation_class",
    "prior_attempt_count",
    "prior_final_response_exists",
    "prior_non_final_timeout_exists",
    "prior_timeout_attempt_index",
    "prior_timeout_started_at",
    "prior_timeout_completed_at",
    "prior_timeout_error_type",
    "prior_timeout_raw_json_path",
    "prior_timeout_raw_json_sha256",
    "next_attempt_index",
    "execution_authorized",
    "lineage",
    "continuation_policy_hash",
]

AUDIT_FIELDS = ["audit_id", "check_name", "status", "observed", "expected", "evidence", "notes"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    return rerank.rel_safe(path)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def stable_hash(*parts: Any) -> str:
    return f6l.sha256_text(canonical_json(list(parts)))


def schema_hash_from_manifest(row: Mapping[str, str]) -> str:
    try:
        payload = json.loads(row.get("generation_parameters", "{}"))
    except json.JSONDecodeError:
        return ""
    return str(payload.get("format_schema_hash", ""))


def continuation_policy_payload() -> dict[str, Any]:
    return {
        "completed_definition": "canonical request is completed only with a terminal final response and parser classification",
        "target_selection": "all_and_only_canonical_requests_without_final_response",
        "include_prior_non_final_transport_timeouts": True,
        "preserve_raw_timeout_evidence_byte_for_byte": True,
        "prior_timeout_not_metric_level_observation": True,
        "one_final_response_per_canonical_request_for_parser_and_metrics": True,
        "duplicate_token_classification": "structural_invalid_fail_strict_no_repair_no_dedup_no_retry",
        "duplicate_token_breaker_v4_1": {
            "stop_if_two_consecutive_closed_200_final_response_windows_same_cell_rate_gt": 0.05,
            "or_stop_if_cell_cumulative_rate_gt_after_at_least_final_responses": {"responses": 400, "rate": 0.05},
        },
        "hard_stop_immediate": [
            "malformed_json",
            "unknown_token",
            "out_of_domain_token",
            "wrong_length",
            "mapping_failure",
            "parser_drift",
            "schema_drift",
            "manifest_hash_drift",
        ],
    }


def continuation_policy_hash() -> str:
    return stable_hash(continuation_policy_payload())


def raw_file_manifest_hash(raw_dir: Path) -> str:
    files = sorted(path for path in raw_dir.rglob("*") if path.is_file())
    payload = [
        {"path": rel(path), "size": path.stat().st_size, "sha256": f6l.safe_file_sha256(path)}
        for path in files
    ]
    return f6l.sha256_text(canonical_json(payload))


def load_base() -> dict[str, Any]:
    sources = post.load_sources()
    manifest_rows = sources["manifest_rows"]
    parse_rows = sources["parse_rows"]
    responses = sources["responses"]
    requests = sources["requests"]
    manifest_by_id = {row["request_id"]: row for row in manifest_rows}
    final_ids = {row["request_id"] for row in parse_rows}
    response_by_id: dict[str, list[dict[str, Any]]] = {}
    request_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in responses:
        response_by_id.setdefault(str(row.get("request_id", "")), []).append(row)
    for row in requests:
        request_by_id.setdefault(str(row.get("request_id", "")), []).append(row)
    pending_rows = [row for row in manifest_rows if row["request_id"] not in final_ids]
    return {
        **sources,
        "manifest_by_id": manifest_by_id,
        "final_ids": final_ids,
        "response_by_id": response_by_id,
        "request_by_id": request_by_id,
        "pending_rows": pending_rows,
    }


def build_manifest_rows(base: Mapping[str, Any]) -> list[dict[str, str]]:
    policy_hash = continuation_policy_hash()
    rows = []
    for source in sorted(base["pending_rows"], key=lambda row: post.request_ordinal(row["request_id"])):
        request_id = source["request_id"]
        prior_responses = base["response_by_id"].get(request_id, [])
        prior_requests = base["request_by_id"].get(request_id, [])
        final_exists = any(str(row.get("is_final_attempt", "true")).lower() == "true" for row in prior_responses)
        timeout_records = [
            row
            for row in prior_responses
            if str(row.get("is_final_attempt", "true")).lower() != "true"
            and str(row.get("backend_error_type", "")) == "TimeoutError"
        ]
        timeout = timeout_records[-1] if timeout_records else {}
        continuation_class = "retry_after_non_final_transport_timeout" if timeout_records else "first_attempt_never_attempted"
        prior_timeout_path = Path(str(timeout.get("raw_json_path", ""))) if timeout.get("raw_json_path") else None
        if prior_timeout_path is not None and not prior_timeout_path.is_absolute():
            prior_timeout_path = f6l.PROJECT_ROOT / prior_timeout_path
        canonical_ordinal = post.request_ordinal(request_id)
        row = {
            "continuation_id": CONTINUATION_ID,
            "continuation_request_id": stable_hash(CONTINUATION_ID, request_id)[:24],
            "base_run_id": RUN_ID,
            "canonical_request_id": request_id,
            "original_request_id": request_id,
            "canonical_manifest_ordinal": str(canonical_ordinal),
            "user_id": source["user_id"],
            "model_name": source["model_name"],
            "model_digest": source["model_digest"],
            "prompting_strategy": source["prompting_strategy"],
            "candidate_pool_source": source["candidate_pool_source"],
            "candidate_pool_hash": source["candidate_pool_hash"],
            "candidate_list_hash": source["candidate_list_hash"],
            "candidate_token_mapping_hash": source["candidate_token_mapping_hash"],
            "candidate_provenance": source["lineage"],
            "rendered_prompt_hash": source["rendered_prompt_hash"],
            "prompt_template_hash": source["prompt_template_hash"],
            "parser_version": source["parser_version"],
            "parser_source_hash": source["parser_source_hash"],
            "schema_hash": schema_hash_from_manifest(source),
            "output_contract_hash": source["output_contract_hash"],
            "continuation_class": continuation_class,
            "prior_attempt_count": str(len(prior_requests)),
            "prior_final_response_exists": str(final_exists).lower(),
            "prior_non_final_timeout_exists": str(bool(timeout_records)).lower(),
            "prior_timeout_attempt_index": str(timeout.get("attempt_index", "")),
            "prior_timeout_started_at": str(timeout.get("started_at", "")),
            "prior_timeout_completed_at": str(timeout.get("completed_at", "")),
            "prior_timeout_error_type": str(timeout.get("backend_error_type", "")),
            "prior_timeout_raw_json_path": rel(prior_timeout_path) if prior_timeout_path else "",
            "prior_timeout_raw_json_sha256": f6l.safe_file_sha256(prior_timeout_path) if prior_timeout_path else "",
            "next_attempt_index": str(len(prior_requests) + 1),
            "execution_authorized": "false",
            "lineage": ";".join(
                [
                    rel(rerank.MANIFEST_PATH),
                    rel(rerank.PARSE_RESULTS_PATH),
                    rel(post.SUMMARY_PATH),
                    "pending_final_response_set_is_exact_and_exhaustive",
                    "no_final_response_reuse",
                ]
            ),
            "continuation_policy_hash": policy_hash,
        }
        rows.append(row)
    return rows


def build_policy_audit(base: Mapping[str, Any], manifest_rows: Sequence[Mapping[str, str]], raw_hash_before: str, raw_hash_after: str) -> tuple[list[dict[str, str]], str]:
    parse_rows = base["parse_rows"]
    responses = base["responses"]
    final_ids = set(base["final_ids"])
    pending_ids = {row["canonical_request_id"] for row in manifest_rows}
    baseline_audit = rerank.ranking_baseline_audit_rows(write_outputs=False)
    timeout_rows = [row for row in manifest_rows if row["continuation_class"] == "retry_after_non_final_transport_timeout"]
    never_rows = [row for row in manifest_rows if row["continuation_class"] == "first_attempt_never_attempted"]
    final_invalid = [row for row in parse_rows if row["strict_valid"] != "true"]
    hard_invalid = [row for row in final_invalid if row.get("failure_reason") != "duplicate_token"]
    non_final_timeouts = [
        row
        for row in responses
        if str(row.get("is_final_attempt", "true")).lower() != "true"
        and str(row.get("backend_error_type", "")) == "TimeoutError"
    ]
    original_status = f6l.git_output(["status", "--short", "original_repo"])
    rows = []

    def add(name: str, passed: bool, observed: str, expected: str, evidence: str, notes: str) -> None:
        rows.append(
            {
                "audit_id": f"F6L-LASTFM-V4-RERANK-V41-AUDIT-{len(rows) + 1:02d}",
                "check_name": name,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "evidence": evidence,
                "notes": notes,
            }
        )

    add("no_inference_api_or_detached_process", True, "freeze_only_no_runtime_call", "no inference/API/Ollama/detached process", rel(CONFIG_PATH), "The module only reads existing artifacts and writes continuation planning artifacts.")
    add("ranking_baseline_audit_pass", all(row["status"] == "pass" for row in baseline_audit), str(Counter(row["status"] for row in baseline_audit)), "all pass", rel(post.BASELINE_AUDIT_PATH), "Frozen ranking baseline remains authoritative.")
    add("ranking_freeze_hash_unchanged", rerank.RANKING_BASELINE_FREEZE_HASH == "D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC", rerank.RANKING_BASELINE_FREEZE_HASH, "D624D4D57CC23C5647D60DF172DE6E71DC746B8098DE66C95EBCCBE609D145EC", rel(rerank.RANKING_BASELINE_PATHS["hash_text"]), "Freeze hash unchanged.")
    add("parser_hash_unchanged", all(row["parser_source_hash"] == rerank.EXPECTED_V4_PARSER_HASH for row in manifest_rows), rerank.EXPECTED_V4_PARSER_HASH, rerank.EXPECTED_V4_PARSER_HASH, rel(MANIFEST_PATH), "Continuation preserves accepted V4 parser.")
    add("schema_hash_unchanged", all(row["schema_hash"] == rerank.EXPECTED_V4_SCHEMA_HASH for row in manifest_rows), rerank.EXPECTED_V4_SCHEMA_HASH, rerank.EXPECTED_V4_SCHEMA_HASH, rel(MANIFEST_PATH), "Continuation preserves accepted JSON schema.")
    add("output_contract_hash_unchanged", all(row["output_contract_hash"] == rerank.EXPECTED_V4_OUTPUT_CONTRACT_HASH for row in manifest_rows), rerank.EXPECTED_V4_OUTPUT_CONTRACT_HASH, rerank.EXPECTED_V4_OUTPUT_CONTRACT_HASH, rel(MANIFEST_PATH), "Continuation preserves accepted output contract.")
    add("raw_artifact_integrity_preserved", raw_hash_before == raw_hash_after, raw_hash_after, raw_hash_before, rel(post.RAW_OUTPUT_DIR), "Raw artifact manifest hash is unchanged across V4.1 freeze generation.")
    add("original_repo_clean", original_status == "", original_status or "clean", "clean", "git status --short original_repo", "No original_repo modifications.")
    add("final_invalids_all_duplicate_token", len(final_invalid) == 48 and not hard_invalid, f"final_invalid={len(final_invalid)};hard={len(hard_invalid)}", "48 duplicate_token;0 hard", rel(rerank.PARSE_RESULTS_PATH), "No repair or deduplication is applied.")
    add("malformed_json_zero", sum(int(row.get("malformed_json_count") or 0) for row in parse_rows) == 0, "0", "0", rel(rerank.PARSE_RESULTS_PATH), "No malformed JSON final parser rows.")
    add("unknown_out_of_domain_zero", sum(int(row.get("unknown_token_count") or 0) for row in parse_rows) == 0, "0", "0", rel(rerank.PARSE_RESULTS_PATH), "Unknown/out-of-domain token count remains zero.")
    add("hard_structural_invalid_zero", not hard_invalid, str(len(hard_invalid)), "0", rel(rerank.PARSE_RESULTS_PATH), "No hard structural invalids.")
    add("pending_final_response_set_exact_and_exhaustive", len(manifest_rows) == 1600 and pending_ids.isdisjoint(final_ids), f"rows={len(manifest_rows)};overlap_final={len(pending_ids & final_ids)}", "1600 rows;0 overlap with final ids", rel(MANIFEST_PATH), "Policy replaces the old contiguous 2401-4000 interval requirement.")
    add("timeout_pending_rows_48", len(timeout_rows) == 48, str(len(timeout_rows)), "48", rel(MANIFEST_PATH), "Prior non-final transport timeouts are included in pending final-response set.")
    add("never_attempted_rows_1552", len(never_rows) == 1552, str(len(never_rows)), "1552", rel(MANIFEST_PATH), "Never-attempted canonical requests are included.")
    add("no_pending_row_has_prior_final_response", all(row["prior_final_response_exists"] == "false" for row in manifest_rows), str(Counter(row["prior_final_response_exists"] for row in manifest_rows)), "all false", rel(MANIFEST_PATH), "Each canonical request can have at most one final response for parser/metrics.")
    add("canonical_request_ids_unique", len({row["canonical_request_id"] for row in manifest_rows}) == len(manifest_rows), str(len({row["canonical_request_id"] for row in manifest_rows})), str(len(manifest_rows)), rel(MANIFEST_PATH), "Continuation manifest is idempotent by canonical request id.")
    add("timeout_raw_evidence_preserved", len(timeout_rows) == 48 and all(row["prior_timeout_raw_json_sha256"] for row in timeout_rows), str(sum(1 for row in timeout_rows if row["prior_timeout_raw_json_sha256"])), "48 timeout raw hashes", rel(MANIFEST_PATH), "Historical timeout records remain byte-addressed.")
    legacy_tokens = ("v1", "v2", "v3", "pilot")
    contamination = any(any(token in row["candidate_pool_source"].casefold() for token in legacy_tokens) for row in manifest_rows)
    add("no_source_contamination", not contamination, "no v1/v2/v3/pilot source strings", "none", rel(MANIFEST_PATH), "No legacy candidate source enters V4.1.")
    add("duplicate_breaker_v4_1_encoded", continuation_policy_payload()["duplicate_token_breaker_v4_1"] == {"stop_if_two_consecutive_closed_200_final_response_windows_same_cell_rate_gt": 0.05, "or_stop_if_cell_cumulative_rate_gt_after_at_least_final_responses": {"responses": 400, "rate": 0.05}}, canonical_json(continuation_policy_payload()["duplicate_token_breaker_v4_1"]), "two consecutive same-cell windows or cumulative same-cell after 400", rel(CONFIG_PATH), "Duplicate-token remains fail-strict; only the stop rule is versioned.")
    decision = DECISION_READY if all(row["status"] == "pass" for row in rows) else DECISION_BLOCKED
    return rows, decision


def write_config(summary: Mapping[str, Any]) -> None:
    text = f"""# F6L LastFM V4.1 reranking continuation freeze.
# Generated for review only. Do not execute without separate authorization.
phase_id: phase_f6l
base_run_id: "{RUN_ID}"
continuation_id: "{CONTINUATION_ID}"
decision_token: "{summary['decision_token']}"
execution_authorized: false
allow_api_calls_required_for_future_execution: true
local_only_required: true
target_selection_policy: all_and_only_canonical_requests_without_final_response
target_counts:
  total: 1600
  retry_after_non_final_transport_timeout: 48
  first_attempt_never_attempted: 1552
source_artifacts:
  base_manifest: "{rel(rerank.MANIFEST_PATH)}"
  base_parse_results: "{rel(rerank.PARSE_RESULTS_PATH)}"
  post_breaker_summary: "{rel(post.SUMMARY_PATH)}"
  raw_output_dir: "{rel(post.RAW_OUTPUT_DIR)}"
contracts:
  parser_source_hash: "{rerank.EXPECTED_V4_PARSER_HASH}"
  schema_hash: "{rerank.EXPECTED_V4_SCHEMA_HASH}"
  output_contract_hash: "{rerank.EXPECTED_V4_OUTPUT_CONTRACT_HASH}"
  ranking_baseline_freeze_hash: "{rerank.RANKING_BASELINE_FREEZE_HASH}"
completion_policy:
  completed_requires_final_response_and_parser_classification: true
  non_final_transport_timeout_is_completed: false
  include_non_final_transport_timeouts_in_continuation: true
  prior_timeout_records_remain_byte_invariant: true
  one_final_response_per_canonical_request_for_parser_metrics: true
duplicate_token_breaker_v4_1:
  duplicate_token_remains_structural_invalid_fail_strict: true
  no_repair: true
  no_deduplication: true
  no_retry_caused_by_duplicate_token: true
  stop_if_two_consecutive_closed_200_final_response_windows_same_cell_rate_gt: 0.05
  stop_if_cell_cumulative_rate_gt_after_at_least_400_final_responses: 0.05
hard_stop_immediate:
  - malformed_json
  - unknown_token
  - out_of_domain_token
  - wrong_length
  - mapping_failure
  - parser_drift
  - schema_drift
  - manifest_hash_drift
"""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(text, encoding="utf-8")


def output_hashes() -> dict[str, str]:
    paths = [CONFIG_PATH, MANIFEST_PATH, POLICY_AUDIT_PATH, SUMMARY_PATH, HASHES_PATH, REPORT_PATH]
    return {rel(path): f6l.safe_file_sha256(path) for path in paths if path.exists()}


def write_report(summary: Mapping[str, Any]) -> None:
    lines = [
        "# F6L LastFM V4.1 Reranking Continuation Freeze",
        "",
        f"Decision token: `{summary['decision_token']}`.",
        "",
        "## Scope",
        "",
        "- Freeze only: no inference, Ollama call, API call, detached process, commit, raw overwrite, repair, or deduplication was performed.",
        f"- Base run: `{RUN_ID}`.",
        f"- Continuation id: `{CONTINUATION_ID}`.",
        "",
        "## Policy Resolution",
        "",
        "- A canonical request is completed only when it has a terminal final response and parser classification.",
        "- Non-final transport timeout evidence does not make a request completed.",
        "- V4.1 targets all and only canonical requests without a final response, not a contiguous manifest ordinal interval.",
        "- The 48 non-final timeout requests are included as `retry_after_non_final_transport_timeout`.",
        "- The 1552 never-attempted requests are included as `first_attempt_never_attempted`.",
        "",
        "## Counts",
        "",
        f"- Continuation rows: `{summary['manifest_counts']['rows']}`.",
        f"- Timeout retry rows: `{summary['manifest_counts']['retry_after_non_final_transport_timeout']}`.",
        f"- Never-attempted rows: `{summary['manifest_counts']['first_attempt_never_attempted']}`.",
        f"- Prior final response exists: `{summary['manifest_counts']['prior_final_response_exists_true']}`.",
        f"- Raw hash before/after freeze: `{summary['raw_hash_before']}` / `{summary['raw_hash_after']}`.",
        "",
        "## Breaker V4.1",
        "",
        "- Duplicate-token remains structural-invalid and fail-strict.",
        "- Stop if two consecutive closed 200-final-response windows in the same model/strategy cell both exceed 5%.",
        "- Or stop if, after at least 400 final responses in the same cell, cumulative duplicate-token rate exceeds 5%.",
        "- Hard stop remains immediate for malformed JSON, unknown/out-of-domain token, wrong length, mapping failure, parser drift, schema drift, or manifest/hash drift.",
        "",
        "## Artifacts",
        "",
        f"- Config: `{rel(CONFIG_PATH)}`.",
        f"- Manifest: `{rel(MANIFEST_PATH)}`.",
        f"- Policy audit: `{rel(POLICY_AUDIT_PATH)}`.",
        f"- Summary: `{rel(SUMMARY_PATH)}`.",
        f"- Hashes: `{rel(HASHES_PATH)}`.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_v4_1_freeze() -> dict[str, Any]:
    raw_hash_before = raw_file_manifest_hash(post.RAW_OUTPUT_DIR)
    base = load_base()
    manifest_rows = build_manifest_rows(base)
    raw_hash_after_manifest = raw_file_manifest_hash(post.RAW_OUTPUT_DIR)
    audit_rows, decision = build_policy_audit(base, manifest_rows, raw_hash_before, raw_hash_after_manifest)
    class_counts = Counter(row["continuation_class"] for row in manifest_rows)
    summary = {
        "continuation_id": CONTINUATION_ID,
        "base_run_id": RUN_ID,
        "created_at": utc_now(),
        "decision_token": decision,
        "no_inference_or_api_call_performed": True,
        "raw_output_dir": rel(post.RAW_OUTPUT_DIR),
        "raw_hash_before": raw_hash_before,
        "raw_hash_after": raw_hash_after_manifest,
        "raw_hash_preserved": raw_hash_before == raw_hash_after_manifest,
        "policy": continuation_policy_payload(),
        "policy_hash": continuation_policy_hash(),
        "manifest_counts": {
            "rows": len(manifest_rows),
            "retry_after_non_final_transport_timeout": class_counts["retry_after_non_final_transport_timeout"],
            "first_attempt_never_attempted": class_counts["first_attempt_never_attempted"],
            "prior_final_response_exists_true": sum(1 for row in manifest_rows if row["prior_final_response_exists"] == "true"),
            "unique_canonical_request_ids": len({row["canonical_request_id"] for row in manifest_rows}),
        },
        "source_decisions": {
            "operational_stop_token": rerank.DECISION_SYSTEMIC_DUPLICATE,
            "previous_post_breaker_token": post.DECISION_BLOCKED,
            "ranking_baseline_freeze_hash": rerank.RANKING_BASELINE_FREEZE_HASH,
        },
        "output_paths": {
            "config": rel(CONFIG_PATH),
            "manifest": rel(MANIFEST_PATH),
            "policy_audit": rel(POLICY_AUDIT_PATH),
            "hashes": rel(HASHES_PATH),
            "report": rel(REPORT_PATH),
        },
    }
    write_config(summary)
    f6l.write_csv(MANIFEST_PATH, manifest_rows, MANIFEST_FIELDS)
    f6l.write_csv(POLICY_AUDIT_PATH, audit_rows, AUDIT_FIELDS)
    f6l.write_json(SUMMARY_PATH, summary)
    write_report(summary)
    raw_hash_after_outputs = raw_file_manifest_hash(post.RAW_OUTPUT_DIR)
    hashes = {
        "raw_hash_before": raw_hash_before,
        "raw_hash_after_manifest_build": raw_hash_after_manifest,
        "raw_hash_after_output_writes": raw_hash_after_outputs,
        "raw_hash_preserved": raw_hash_before == raw_hash_after_outputs,
        "source_artifacts": {
            rel(rerank.MANIFEST_PATH): f6l.safe_file_sha256(rerank.MANIFEST_PATH),
            rel(rerank.PARSE_RESULTS_PATH): f6l.safe_file_sha256(rerank.PARSE_RESULTS_PATH),
            rel(rerank.EXECUTION_PATH): f6l.safe_file_sha256(rerank.EXECUTION_PATH),
            rel(post.SUMMARY_PATH): f6l.safe_file_sha256(post.SUMMARY_PATH),
            rel(rerank.RANKING_BASELINE_PATHS["hash_text"]): f6l.safe_file_sha256(rerank.RANKING_BASELINE_PATHS["hash_text"]),
        },
        "output_artifacts": output_hashes(),
    }
    f6l.write_json(HASHES_PATH, hashes)
    return summary


def main() -> None:
    summary = build_v4_1_freeze()
    print(canonical_json({"decision_token": summary["decision_token"], "continuation_id": CONTINUATION_ID, "rows": summary["manifest_counts"]["rows"]}))


if __name__ == "__main__":
    main()

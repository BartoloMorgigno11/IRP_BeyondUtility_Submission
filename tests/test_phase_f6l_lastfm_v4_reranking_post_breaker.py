"""Focused tests for F6L LastFM V4 reranking post-breaker consolidation."""

from __future__ import annotations

import csv
import json

import pytest

from src.analysis import phase_f6l_lastfm_v4_reranking_post_breaker as post


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_summary() -> dict:
    return json.loads(post.SUMMARY_PATH.read_text(encoding="utf-8"))


pytestmark = [pytest.mark.cleanup_historical, pytest.mark.phase_snapshot]


def test_integrity_counts_and_exact_residual_set() -> None:
    summary = read_summary()
    integrity = summary["integrity"]
    assert integrity["strict_valid"] + integrity["structural_invalid"] == integrity["final_completed"] == 2400
    assert integrity["duplicate_token_invalid"] == 48
    assert integrity["pending_final_response_count"] == 1600
    assert integrity["pending_final_response_ordinals_are_2401_4000"] is False
    assert "2449-4000" in integrity["pending_manifest_ordinal_ranges"]
    assert integrity["non_final_transport_timeout_count"] == 48


def test_completed_ordinals_and_request_ids_are_unique() -> None:
    parse_rows = read_csv(post.rerank.PARSE_RESULTS_PATH)
    request_ids = [row["request_id"] for row in parse_rows]
    ordinals = [post.request_ordinal(request_id) for request_id in request_ids]
    assert len(request_ids) == len(set(request_ids)) == 2400
    assert len(ordinals) == len(set(ordinals)) == 2400


def test_failure_taxonomy_classifies_all_final_invalids_as_duplicate_token() -> None:
    rows = {row["failure_class"]: row for row in read_csv(post.FAILURE_TAXONOMY_PATH)}
    assert rows["duplicate_token"]["count"] == "48"
    assert rows["malformed_json"]["count"] == "0"
    assert rows["unknown_token"]["count"] == "0"
    assert rows["out_of_domain_token"]["count"] == "0"
    assert rows["wrong_length"]["count"] == "0"
    assert rows["mapping_failure"]["count"] == "0"
    assert rows["other_structural_error"]["count"] == "0"
    assert rows["transport_non_final_timeout"]["count"] == "48"


def test_window_12_is_first_and_only_breaker_failure() -> None:
    rows = read_csv(post.WINDOW_ANALYSIS_PATH)
    failures = [row for row in rows if row["status"] == "fail"]
    assert [row["window_index"] for row in failures] == ["12"]
    window_12 = failures[0]
    assert window_12["duplicate_token_invalid"] == "11"
    assert window_12["duplicate_rate"] == "0.055"
    assert window_12["response_ordinal_start"] == "2201"
    assert window_12["response_ordinal_end"] == "2400"


def test_baseline_audit_and_original_repo_guardrails() -> None:
    summary = read_summary()
    assert summary["integrity"]["baseline_audit_pass"] is True
    assert summary["integrity"]["baseline_freeze_hash"] == post.rerank.RANKING_BASELINE_FREEZE_HASH
    assert summary["integrity"]["original_repo_status"] == ""
    baseline_rows = read_csv(post.BASELINE_AUDIT_PATH)
    assert baseline_rows
    assert all(row["status"] == "pass" for row in baseline_rows)


def test_no_inference_policy_and_no_legacy_source_contamination() -> None:
    summary = read_summary()
    assert summary["no_inference_or_api_call_performed"] is True
    assert summary["integrity"]["no_v1_v2_v3_pilot_source_contamination"] is True
    assert summary["decision_token"] == post.DECISION_BLOCKED
    assert summary["continuation_v4_1_proposal_generated"] is False


def test_raw_hash_manifest_matches_current_raw_state() -> None:
    summary = read_summary()
    current_hash = post.source_hashes(post.load_sources())["partial_raw_artifact_manifest_sha256"]
    assert current_hash == summary["integrity"]["partial_raw_artifact_manifest_sha256"]


def test_metrics_denominators_and_interim_partial_labels() -> None:
    coverage = read_csv(post.CELL_COVERAGE_PATH)
    assert {row["coverage_status"] for row in coverage} == {"INTERIM_PARTIAL"}
    completed_by_cell = {
        (row["model_name"], row["prompting_strategy"]): row["final_completed_requests"]
        for row in coverage
    }
    metrics = read_csv(post.METRICS_PATH)
    primary = [row for row in metrics if row["metric_group"] == "primary_fail_strict_completed_final"]
    assert primary
    assert {row["coverage_status"] for row in primary} == {"INTERIM_PARTIAL"}
    for row in primary:
        key = (row["model_name"], row["prompting_strategy"])
        assert row["denominator"] == completed_by_cell[key]
        assert row["planned_denominator"] == "1000"

"""Focused tests for the F6L LastFM V4.1 continuation freeze."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.analysis import phase_f6l_lastfm_v4_reranking_continuation_v4_1 as v41


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_summary() -> dict:
    return json.loads(v41.SUMMARY_PATH.read_text(encoding="utf-8"))


pytestmark = [pytest.mark.cleanup_historical, pytest.mark.phase_snapshot]


def test_continuation_manifest_counts_and_classes() -> None:
    rows = read_csv(v41.MANIFEST_PATH)
    classes = {name: sum(1 for row in rows if row["continuation_class"] == name) for name in {row["continuation_class"] for row in rows}}
    assert len(rows) == 1600
    assert classes["retry_after_non_final_transport_timeout"] == 48
    assert classes["first_attempt_never_attempted"] == 1552


def test_no_manifest_row_has_prior_final_response_and_ids_are_unique() -> None:
    rows = read_csv(v41.MANIFEST_PATH)
    ids = [row["canonical_request_id"] for row in rows]
    assert len(ids) == len(set(ids)) == 1600
    assert {row["prior_final_response_exists"] for row in rows} == {"false"}
    assert {row["execution_authorized"] for row in rows} == {"false"}


def test_timeout_raw_evidence_is_preserved_and_addressed() -> None:
    rows = [row for row in read_csv(v41.MANIFEST_PATH) if row["continuation_class"] == "retry_after_non_final_transport_timeout"]
    assert len(rows) == 48
    for row in rows:
        path = v41.f6l.PROJECT_ROOT / row["prior_timeout_raw_json_path"]
        assert path.exists()
        assert v41.f6l.safe_file_sha256(path) == row["prior_timeout_raw_json_sha256"]
        assert row["prior_attempt_count"] == "1"
        assert row["next_attempt_index"] == "2"
        assert row["prior_timeout_error_type"] == "TimeoutError"


def test_never_attempted_rows_have_no_prior_timeout_evidence() -> None:
    rows = [row for row in read_csv(v41.MANIFEST_PATH) if row["continuation_class"] == "first_attempt_never_attempted"]
    assert len(rows) == 1552
    assert {row["prior_attempt_count"] for row in rows} == {"0"}
    assert {row["next_attempt_index"] for row in rows} == {"1"}
    assert {row["prior_timeout_raw_json_sha256"] for row in rows} == {""}


def test_policy_audit_all_pass_and_decision_ready() -> None:
    summary = read_summary()
    audit_rows = read_csv(v41.POLICY_AUDIT_PATH)
    assert summary["decision_token"] == v41.DECISION_READY
    assert audit_rows
    assert all(row["status"] == "pass" for row in audit_rows)


def test_baseline_parser_schema_and_original_repo_guardrails() -> None:
    rows = read_csv(v41.MANIFEST_PATH)
    summary = read_summary()
    assert summary["source_decisions"]["ranking_baseline_freeze_hash"] == v41.rerank.RANKING_BASELINE_FREEZE_HASH
    assert {row["parser_source_hash"] for row in rows} == {v41.rerank.EXPECTED_V4_PARSER_HASH}
    assert {row["schema_hash"] for row in rows} == {v41.rerank.EXPECTED_V4_SCHEMA_HASH}
    assert v41.f6l.git_output(["status", "--short", "original_repo"]) == ""


def test_raw_outputs_not_overwritten_by_freeze() -> None:
    summary = read_summary()
    hashes = json.loads(v41.HASHES_PATH.read_text(encoding="utf-8"))
    current = v41.raw_file_manifest_hash(v41.post.RAW_OUTPUT_DIR)
    assert summary["raw_hash_preserved"] is True
    assert hashes["raw_hash_preserved"] is True
    assert current == summary["raw_hash_before"] == summary["raw_hash_after"]


def test_zero_inference_api_policy_and_v41_breaker_encoded() -> None:
    summary = read_summary()
    config_text = v41.CONFIG_PATH.read_text(encoding="utf-8")
    assert summary["no_inference_or_api_call_performed"] is True
    assert "execution_authorized: false" in config_text
    assert "stop_if_two_consecutive_closed_200_final_response_windows_same_cell_rate_gt: 0.05" in config_text
    assert "stop_if_cell_cumulative_rate_gt_after_at_least_400_final_responses: 0.05" in config_text

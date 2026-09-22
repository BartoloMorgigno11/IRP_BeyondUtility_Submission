"""Focused tests for the F6L LastFM V4.1 continuation runner."""

from __future__ import annotations

import csv

import pytest

from src.analysis import phase_f6l_lastfm_v4_reranking_continuation_v4_1_execution as runner


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_execution_manifest_targets_only_frozen_pending_requests() -> None:
    rows = runner.execution_manifest_rows("unit", runner.local_raw_output_dir("unit"))
    frozen = read_csv(runner.freeze.MANIFEST_PATH)
    assert len(rows) == 1600
    assert [row["request_id"] for row in rows] == [row["canonical_request_id"] for row in frozen]
    assert {row["run_id"] for row in rows} == {"unit"}
    assert {row["full_run_id"] for row in rows} == {runner.EXECUTION_ID}
    assert {row["parser_source_hash"] for row in rows} == {runner.rerank.EXPECTED_V4_PARSER_HASH}


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_preflight_unit_mode_passes_without_runtime_call() -> None:
    rows, _, state = runner.validate_preflight(
        execution_run_id="unit",
        raw_output_dir=runner.local_raw_output_dir("unit"),
        resume=False,
        allow_api_calls=True,
        local_only=True,
        endpoint="http://127.0.0.1:11434/api/generate",
        run_tests=False,
        check_runtime=False,
    )
    assert state is None
    assert len(rows) >= 18
    assert all(row["status"] == "pass" for row in rows)
    assert any(row["check_name"] == "pending_final_response_set_exact_and_exhaustive" for row in rows)


def test_cell_breaker_uses_same_cell_cumulative_rule() -> None:
    rows = []
    for index in range(400):
        rows.append(
            {
                "request_id": f"r{index}",
                "model_name": "mistral:7b-instruct",
                "prompting_strategy": "LLMRank recency-focused reranking",
                "failure_reason": "duplicate_token" if index < 21 else "",
            }
        )
    windows, breaker, cell, _, cumulative = runner.cell_breaker_state(rows)
    assert breaker == "same_cell_cumulative_duplicate_rate_above_0.05_after_400_final_responses"
    assert cell == "mistral:7b-instruct|LLMRank recency-focused reranking"
    assert cumulative > 0.05
    assert len(windows) == 2


def test_cell_breaker_uses_two_consecutive_window_rule() -> None:
    rows = []
    for index in range(400):
        rows.append(
            {
                "request_id": f"r{index}",
                "model_name": "mistral:7b-instruct",
                "prompting_strategy": "base/OpenP5-derived reranking",
                "failure_reason": "duplicate_token" if index < 11 or 200 <= index < 211 else "",
            }
        )
    _, breaker, cell, _, _ = runner.cell_breaker_state(rows)
    assert breaker == "two_consecutive_same_cell_closed_200_windows_duplicate_rate_above_0.05"
    assert cell == "mistral:7b-instruct|base/OpenP5-derived reranking"


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_original_final_ids_are_excluded_from_continuation() -> None:
    continuation_ids = {row["canonical_request_id"] for row in read_csv(runner.freeze.MANIFEST_PATH)}
    original_ids = runner.original_final_ids()
    assert len(original_ids) == 2400
    assert len(continuation_ids) == 1600
    assert not (continuation_ids & original_ids)

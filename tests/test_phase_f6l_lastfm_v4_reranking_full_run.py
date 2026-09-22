"""Focused tests for the F6L LastFM V4 reranking full run."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from src.analysis import phase_f6l_lastfm_v4_reranking_full_run as rerank


def toy_candidates() -> list[dict[str, str]]:
    return [
        {
            "candidate_position": str(index),
            "candidate_item_id": str(1000 + index),
            "candidate_item_title": f"Artist {index}",
            "pool_hash": "HASH",
            "is_positive": "true" if index == 3 else "false",
        }
        for index in range(1, 21)
    ]


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_reranking_pool_shape_and_positive_coverage() -> None:
    rows_by_user = rerank.reranking_pool_rows_by_user()
    coverage = rerank.positive_coverage()
    assert len(rows_by_user) == 1000
    assert {len(rows) for rows in rows_by_user.values()} == {20}
    assert all(len({row["candidate_item_id"] for row in rows}) == 20 for rows in rows_by_user.values())
    assert coverage["positive_present_users"] == 314
    assert coverage["positive_absent_users"] == 686


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_full_reranking_manifest_has_4000_requests_and_four_cells() -> None:
    build = rerank.build_full_run(run_id="unit", raw_output_dir=rerank.local_raw_output_dir("unit"), write_outputs=False)
    rows = build.manifest_rows
    assert len(rows) == 4000
    assert len({row["request_id"] for row in rows}) == 4000
    assert len({row["user_id"] for row in rows}) == 1000
    assert set(Counter(row["user_id"] for row in rows).values()) == {4}
    assert {row["model_name"] for row in rows} == set(rerank.MODEL_ORDER)
    assert {row["prompting_strategy"] for row in rows} == set(rerank.STRATEGY_ORDER)
    assert {row["task"] for row in rows} == {"reranking"}
    assert all(row["parser_source_hash"] == rerank.EXPECTED_V4_PARSER_HASH for row in rows)


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_manifest_uses_reranking_pool_not_ranking_candidates() -> None:
    build = rerank.build_full_run(run_id="unit", raw_output_dir=rerank.local_raw_output_dir("unit"), write_outputs=False)
    assert build.manifest["source_pool"].endswith("phase_f6l_lastfm_operationalized_reranking_pool.csv")
    assert "phase_f6l_lastfm_operationalized_candidates.csv" not in build.manifest["source_pool"]
    assert build.manifest["excluded_superseded_artifacts"]["ranking_v4_manifest"]["reason"]
    assert build.manifest["classification"] == rerank.CLASSIFICATION
    assert build.manifest["ranking_baseline"]["freeze_hash"] == rerank.RANKING_BASELINE_FREEZE_HASH
    assert build.manifest["ranking_baseline"]["primary_policy"] == "fail_strict_values_only_for_ranking_vs_reranking"
    assert build.manifest["historical_pre_existing_gap"]["classification"] == "historical_pre_existing_gap"


def test_reranking_prompt_contains_tokens_and_no_forbidden_labels() -> None:
    sample = {"history_item_titles": json.dumps(["A", "B"])}
    prompt = rerank.render_prompt(sample, toy_candidates(), "base/OpenP5-derived reranking")
    assert "Initial candidate order:" in prompt
    assert "rerank the supplied candidates" in prompt
    assert "[C01] Artist 1" in prompt
    assert all(token in prompt for token in rerank.TOKEN_LIST)
    lowered = prompt.casefold()
    assert "positive" not in lowered
    assert "ground truth" not in lowered
    assert "ndcg" not in lowered
    assert "hr@" not in lowered


def test_llmrank_prompt_adds_recency_cue() -> None:
    sample = {"history_item_titles": json.dumps(["A", "B"])}
    prompt = rerank.render_prompt(sample, toy_candidates(), "LLMRank recency-focused reranking")
    assert "Recency cue" in prompt
    assert "Final answer must still be only the JSON object." in prompt


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_v4_parser_schema_and_output_contract_hashes_are_reused() -> None:
    assert rerank.parser_source_hash() == rerank.EXPECTED_V4_PARSER_HASH
    assert rerank.schema_hash() == rerank.EXPECTED_V4_SCHEMA_HASH
    assert rerank.output_contract_hash() == rerank.EXPECTED_V4_OUTPUT_CONTRACT_HASH


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_preflight_has_38_checks_in_unit_mode() -> None:
    build = rerank.build_full_run(run_id="unit", raw_output_dir=rerank.local_raw_output_dir("unit"), write_outputs=False)
    rows, _, _ = rerank.validate_preflight(
        build,
        raw_output_dir=rerank.local_raw_output_dir("unit"),
        resume=False,
        allow_api_calls=True,
        local_only=True,
        endpoint="http://127.0.0.1:11434/api/generate",
        run_tests=False,
        check_runtime=False,
        max_prompt_checks=40,
        check_identity=False,
    )
    assert len(rows) == 38
    assert all(row["status"] == "pass" for row in rows)
    assert any(row["check_name"] == "no_llmsrec_or_extra_strategy" for row in rows)
    assert any(row["check_name"] == "ranking_baseline_frozen_and_candidate_pool_not_used" for row in rows)


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_ranking_baseline_audit_matches_authoritative_freeze() -> None:
    rows = rerank.ranking_baseline_audit_rows(write_outputs=False)
    assert rows
    assert all(row["status"] == "pass" for row in rows)
    assert any(row["check_name"] == "freeze_hash_identical" for row in rows)
    assert any(row["check_name"] == "fail_strict_metric_values_identical" for row in rows)
    assert any(row["check_name"] == "historical_pre_existing_gap_classified" for row in rows)


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_ranking_baseline_comparison_uses_frozen_fail_strict_values() -> None:
    reranking_metrics = [
        {
            "model_name": "qwen2:7b-instruct",
            "prompting_strategy": "base/OpenP5-derived reranking",
            "metric": "HR@5_fail_strict",
            "value": "0.500",
        }
    ]
    rows = rerank.ranking_baseline_comparison_rows("unit", reranking_metrics)
    row = next(
        item
        for item in rows
        if item["model_name"] == "qwen2:7b-instruct"
        and item["reranking_strategy"] == "base/OpenP5-derived reranking"
        and item["metric"] == "HR@5_fail_strict"
    )
    assert row["ranking_fail_strict_value"] == "0.458"
    assert row["reranking_fail_strict_value"] == "0.500"
    assert row["ranking_freeze_hash"] == rerank.RANKING_BASELINE_FREEZE_HASH
    assert "complete-case ranking is secondary" in row["notes"]


def test_duplicate_window_policy_is_inherited_from_final_v4_ranking() -> None:
    strict = {"strict_valid": "true", "failure_reason": ""}
    duplicate = {"strict_valid": "false", "failure_reason": "duplicate_token"}
    assert rerank.closed_duplicate_window_status([strict] * 199) == (0, 0.0, "")
    assert rerank.closed_duplicate_window_status([strict] * 190 + [duplicate] * 10) == (1, 0.05, "")
    assert rerank.closed_duplicate_window_status([strict] * 189 + [duplicate] * 11)[2] == "closed_200_new_response_window_duplicate_token_rate_above_5_percent"

"""Focused tests for the F6L LastFM corrective V3 output-contract pilot."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.analysis import phase_f6l_lastfm_corrective_output_contract_pilot as pilot


def toy_candidates() -> list[dict[str, str]]:
    return [
        {
            "candidate_position": str(index),
            "candidate_item_id": str(1000 + index),
            "candidate_item_title": f"Artist {index}",
            "candidate_list_hash": "HASH",
        }
        for index in range(1, 21)
    ]


def test_pilot_manifest_has_exact_40_requests_10_users_and_four_cells() -> None:
    build = pilot.build_pilot(run_id="unit", raw_output_dir=pilot.local_raw_output_dir("unit"), write_outputs=False)
    rows = build.manifest_rows
    assert len(rows) == 40
    assert len({row["request_id"] for row in rows}) == 40
    assert len({row["user_id"] for row in rows}) == 10
    assert set(Counter(row["user_id"] for row in rows).values()) == {4}
    assert {row["model_name"] for row in rows} == {"qwen2:7b-instruct", "mistral:7b-instruct"}
    assert {row["prompting_strategy"] for row in rows} == {"base/OpenP5 title-list", "LLMRank recency"}


def test_history_stratification_is_4_3_3_users_and_selection_is_blind() -> None:
    build = pilot.build_pilot(run_id="unit", raw_output_dir=pilot.local_raw_output_dir("unit"), write_outputs=False)
    per_stratum = Counter(row["history_stratum"] for row in build.selection_rows)
    assert per_stratum == {"short": 4, "medium": 3, "long": 3}
    forbidden = set(pilot.SELECTION_BLIND_FORBIDDEN_FIELDS)
    assert not forbidden.intersection(pilot.SELECTION_INPUT_FIELDS)
    assert all("positive" not in row["selection_input_fields"] for row in build.selection_rows)
    assert all("ndcg" not in row["selection_input_fields"].casefold() for row in build.selection_rows)


def test_manifest_is_deterministic_and_candidate_list_shared_across_cells() -> None:
    first = pilot.build_pilot(run_id="unit", raw_output_dir=pilot.local_raw_output_dir("unit"), write_outputs=False)
    second = pilot.build_pilot(run_id="unit", raw_output_dir=pilot.local_raw_output_dir("unit"), write_outputs=False)
    assert pilot.canonical_json(first.manifest_rows) == pilot.canonical_json(second.manifest_rows)
    for user_id in {row["user_id"] for row in first.manifest_rows}:
        user_rows = [row for row in first.manifest_rows if row["user_id"] == user_id]
        assert len({row["candidate_list_hash"] for row in user_rows}) == 1
        assert len({row["candidate_index_mapping"] for row in user_rows}) == 1


def test_prompt_contract_excludes_positive_labels_and_metric_tokens() -> None:
    build = pilot.build_pilot(run_id="unit", raw_output_dir=pilot.local_raw_output_dir("unit"), write_outputs=False)
    checks = pilot.validate_pilot_manifest(build.manifest_rows)
    by_name = {name: passed for name, passed, *_ in checks}
    assert by_name["no_positive_or_metric_tokens_in_prompt"]
    assert by_name["no_full_cohort"]
    assert by_name["no_reranking"]
    assert by_name["no_llmsrec"]


def test_v3_parser_accepts_only_strict_json_index_contract() -> None:
    result = pilot.parse_v3_response('{"ranking":[1,2,3,4,5,6,7,8,9,10]}', toy_candidates())
    assert result.strict_valid
    assert result.ranking_indices == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert result.ranked_item_ids[:2] == ["1001", "1002"]


def test_v3_parser_rejects_extra_text_extra_keys_duplicates_and_range() -> None:
    candidates = toy_candidates()
    assert pilot.parse_v3_response('Here: {"ranking":[1,2,3,4,5,6,7,8,9,10]}', candidates).failure_reason == "malformed_json"
    assert pilot.parse_v3_response('{"ranking":[1,2,3,4,5,6,7,8,9,10],"note":"x"}', candidates).failure_reason == "extra_or_missing_json_keys"
    assert pilot.parse_v3_response('{"ranking":[1,1,3,4,5,6,7,8,9,10]}', candidates).failure_reason == "duplicate_index"
    assert pilot.parse_v3_response('{"ranking":[1,2,3,4,5,6,7,8,9,21]}', candidates).failure_reason == "out_of_range_index"
    assert pilot.parse_v3_response('{"ranking":[1,2,3,4,5,6,7,8,9,"10"]}', candidates).failure_reason == "non_integer_index"
    assert pilot.parse_v3_response('{"ranking":[1,2,3]}', candidates).failure_reason == "wrong_ranking_length"


def test_mapping_hash_and_request_hash_are_stable() -> None:
    candidates = toy_candidates()
    assert pilot.candidate_index_mapping_hash(candidates) == pilot.candidate_index_mapping_hash(candidates)
    sample = {
        "user_id": "u1",
        "train_history_count": "3",
        "_history_stratum": "short",
        "history_item_titles": json.dumps(["A", "B", "C"]),
    }
    row1 = pilot.manifest_row_for_request(
        index=1,
        run_id="unit",
        raw_output_dir=Path("results/llm/unit"),
        sample_row=sample,
        candidate_rows=candidates,
        model_name="qwen2:7b-instruct",
        strategy="base/OpenP5 title-list",
    )
    row2 = pilot.manifest_row_for_request(
        index=1,
        run_id="unit",
        raw_output_dir=Path("results/llm/unit"),
        sample_row=sample,
        candidate_rows=candidates,
        model_name="qwen2:7b-instruct",
        strategy="base/OpenP5 title-list",
    )
    assert row1["request_contract_hash"] == row2["request_contract_hash"]
    assert row1["rendered_prompt_hash"] == row2["rendered_prompt_hash"]


def test_original_raw_hash_marker_remains_available_for_preflight() -> None:
    expected = pilot.original_raw_expected_sha()
    assert expected
    assert pilot.original_raw_hash_map_sha() == expected

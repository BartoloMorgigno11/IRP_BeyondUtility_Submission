"""Focused tests for the F6L LastFM corrective V4 output-contract pilot."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.analysis import phase_f6l_lastfm_corrective_output_contract_v4_pilot as pilot


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


def test_v3_failure_audit_has_exact_seven_rows_and_expected_users() -> None:
    rows = pilot.build_v3_failure_audit(write_outputs=False)
    assert len(rows) == 7
    assert {row["user_id"] for row in rows} == {"640", "709", "1333", "1749"}
    assert all(row["diagnostic_cause"] for row in rows)
    assert all(row["notes"].startswith("No clipping") for row in rows)


def test_v4_manifest_has_40_requests_and_covers_regression_holdout_users() -> None:
    build = pilot.build_pilot(run_id="unit", raw_output_dir=pilot.local_raw_output_dir("unit"), write_outputs=False)
    rows = build.manifest_rows
    assert len(rows) == 40
    assert len({row["request_id"] for row in rows}) == 40
    assert len({row["user_id"] for row in rows}) == 10
    assert set(Counter(row["user_id"] for row in rows).values()) == {4}
    assert {row["subset"] for row in rows} == {"regression", "holdout"}
    assert set(build.manifest["regression_users"]) == {"640", "709", "1333", "1749"}
    assert len(build.manifest["holdout_users"]) == 6
    assert set(build.manifest["holdout_users"]).isdisjoint(pilot.selected_v3_pilot_users())


def test_v4_mapping_tokens_are_c01_c20_and_stable_across_cells() -> None:
    build = pilot.build_pilot(run_id="unit", raw_output_dir=pilot.local_raw_output_dir("unit"), write_outputs=False)
    for row in build.manifest_rows:
        mapping = json.loads(row["candidate_token_mapping"])
        assert set(mapping) == set(pilot.TOKEN_LIST)
    for user_id in {row["user_id"] for row in build.manifest_rows}:
        user_rows = [row for row in build.manifest_rows if row["user_id"] == user_id]
        assert len({row["candidate_token_mapping_hash"] for row in user_rows}) == 1
        assert len({row["candidate_list_hash"] for row in user_rows}) == 1


def test_v4_selection_is_blind_to_relevance_metrics_and_v3_success() -> None:
    build = pilot.build_pilot(run_id="unit", raw_output_dir=pilot.local_raw_output_dir("unit"), write_outputs=False)
    forbidden = set(pilot.SELECTION_BLIND_FORBIDDEN_FIELDS)
    assert not forbidden.intersection(pilot.SELECTION_INPUT_FIELDS)
    assert all("ndcg" not in row["selection_input_fields"].casefold() for row in build.selection_rows)
    assert all("v3_success" not in row["selection_input_fields"] for row in build.selection_rows)


def test_v4_prompt_contains_authorized_tokens_and_no_forbidden_labels() -> None:
    sample = {"history_item_titles": json.dumps(["A", "B"]), "_subset": "holdout"}
    prompt = pilot.render_v4_prompt(sample, toy_candidates(), "base/OpenP5 title-list")
    assert all(token in prompt for token in pilot.TOKEN_LIST)
    assert "[C01] Artist 1" in prompt
    lowered = prompt.casefold()
    assert "positive" not in lowered
    assert "ndcg" not in lowered
    assert "relevance" not in lowered


def test_v4_parser_accepts_strict_opaque_token_json() -> None:
    result = pilot.parse_v4_response('{"ranking":["C01","C02","C03","C04","C05","C06","C07","C08","C09","C10"]}', toy_candidates())
    assert result.strict_valid
    assert result.ranking_tokens[:2] == ["C01", "C02"]
    assert result.ranked_item_ids[:2] == ["1001", "1002"]


def test_v4_parser_rejects_bad_structures_and_tokens() -> None:
    candidates = toy_candidates()
    assert pilot.parse_v4_response('text {"ranking":["C01","C02"]}', candidates).failure_reason == "malformed_json"
    assert pilot.parse_v4_response('{"ranking":["C01","C02"],"note":"x"}', candidates).failure_reason == "extra_or_missing_json_keys"
    assert pilot.parse_v4_response('{"ranking":[1,2,3,4,5,6,7,8,9,10]}', candidates).failure_reason == "numeric_without_c_prefix"
    assert pilot.parse_v4_response('{"ranking":["C01","C02","C03","C04","C05","C06","C07","C08","C09","C21"]}', candidates).failure_reason == "unknown_token"
    assert pilot.parse_v4_response('{"ranking":["C01","C01","C03","C04","C05","C06","C07","C08","C09","C10"]}', candidates).failure_reason == "duplicate_token"
    assert pilot.parse_v4_response('{"ranking":["C01","C02"]}', candidates).failure_reason == "wrong_ranking_length"


def test_v4_manifest_is_deterministic_and_hashes_stable() -> None:
    first = pilot.build_pilot(run_id="unit", raw_output_dir=pilot.local_raw_output_dir("unit"), write_outputs=False)
    second = pilot.build_pilot(run_id="unit", raw_output_dir=pilot.local_raw_output_dir("unit"), write_outputs=False)
    assert pilot.canonical_json(first.manifest_rows) == pilot.canonical_json(second.manifest_rows)
    assert first.manifest["manifest_hash"] == second.manifest["manifest_hash"]


def test_v3_artifact_hashes_are_available_for_immutability_gate() -> None:
    hashes = pilot.v3_artifact_hashes()
    assert hashes
    assert any(key.startswith("v3_raw::") for key in hashes)


def test_full_v4_manifest_has_4000_requests_and_same_four_cells_per_user() -> None:
    build = pilot.build_full_run(run_id="unit", raw_output_dir=pilot.full_raw_output_dir("unit"), write_outputs=False)
    rows = build.manifest_rows
    assert len(rows) == 4000
    assert len({row["request_id"] for row in rows}) == 4000
    assert len({row["user_id"] for row in rows}) == 1000
    assert set(Counter(row["user_id"] for row in rows).values()) == {4}
    assert {row["model_name"] for row in rows} == set(pilot.MODEL_ORDER)
    assert {row["prompting_strategy"] for row in rows} == set(pilot.STRATEGY_ORDER)
    assert all(row["parser_source_hash"] == pilot.EXPECTED_V4_PARSER_HASH for row in rows)


def test_full_v4_manifest_places_internal_canary_first_without_extra_requests() -> None:
    build = pilot.build_full_run(run_id="unit", raw_output_dir=pilot.full_raw_output_dir("unit"), write_outputs=False)
    first_40 = build.manifest_rows[:40]
    assert [row["user_id"] for row in first_40[::4]] == pilot.FULL_CANARY_USERS
    assert {row["subset"] for row in first_40[:16]} == {"canary_regression"}
    assert {row["subset"] for row in first_40[16:]} == {"canary_holdout"}
    for offset in range(0, 40, 4):
        cell_rows = first_40[offset : offset + 4]
        assert {(row["model_name"], row["prompting_strategy"]) for row in cell_rows} == {
            (model, strategy) for model in pilot.MODEL_ORDER for strategy in pilot.STRATEGY_ORDER
        }


def test_full_v4_freeze_hashes_match_authorized_pilot_contract() -> None:
    build = pilot.build_full_run(run_id="unit", raw_output_dir=pilot.full_raw_output_dir("unit"), write_outputs=False)
    assert build.manifest["parser_source_hash"] == pilot.EXPECTED_V4_PARSER_HASH
    assert build.manifest["json_schema_hash"] == pilot.EXPECTED_V4_SCHEMA_HASH
    assert build.manifest["output_contract_hash"] == pilot.EXPECTED_V4_OUTPUT_CONTRACT_HASH
    assert build.manifest["prompt_template_hashes"] == pilot.EXPECTED_V4_PROMPT_TEMPLATE_HASHES
    assert build.manifest["request_count"] == 4000


def test_full_v4_prelaunch_builder_is_byte_identical() -> None:
    ok, first_hash, second_hash = pilot.full_build_identity_check("unit", pilot.full_raw_output_dir("unit"))
    assert ok
    assert first_hash == second_hash


def test_final_amended_duplicate_window_breaker_uses_closed_200_response_windows() -> None:
    strict = {"strict_valid": "true", "failure_reason": ""}
    duplicate = {"strict_valid": "false", "failure_reason": "duplicate_token"}
    assert pilot.closed_duplicate_window_status([strict] * 199) == (0, 0.0, "")
    assert pilot.closed_duplicate_window_status([strict] * 190 + [duplicate] * 10) == (1, 0.05, "")
    assert pilot.closed_duplicate_window_status([strict] * 189 + [duplicate] * 11) == (
        1,
        0.055,
        "closed_200_new_response_window_duplicate_token_rate_above_5_percent",
    )


def test_final_amended_breaker_distinguishes_duplicate_from_hard_structural_invalid() -> None:
    duplicate = {"strict_valid": "false", "failure_reason": "duplicate_token"}
    malformed = {"strict_valid": "false", "failure_reason": "malformed_json"}
    assert pilot.is_duplicate_token_invalid(duplicate)
    assert not pilot.is_hard_structural_invalid(duplicate)
    assert pilot.is_hard_structural_invalid(malformed)
    assert pilot.AMENDED_DECISION_WITH_INVALID == "f6l_lastfm_v4_full_ranking_complete_with_observed_duplicate_invalidity"
    assert pilot.AMENDED_DECISION_SYSTEMIC_REGRESSION == "f6l_lastfm_v4_full_ranking_stopped_by_systemic_duplicate_regression"

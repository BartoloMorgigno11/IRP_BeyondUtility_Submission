import json

import pandas as pd
import pytest

from src.analysis import canonical_llm_execution_preflight as preflight
from src.llm.canonical_llm_contract import (
    contract_hash,
    estimate_token_envelope,
    parse_diagnostic_salvage_response,
    parse_strict_index_response,
    parser_contract_rows,
    prompt_contract_rows,
    render_candidates,
    render_history,
    render_prompt,
    request_count,
    validate_one_recommended_policy_per_category,
)


def test_prompt_registry_has_unique_ids_and_stable_hashes():
    rows = prompt_contract_rows()
    ids = [row["prompt_contract_id"] for row in rows]

    assert len(ids) == len(set(ids))
    for row in rows:
        assert row["prompt_template_sha256"]
        assert row["contract_sha256"] == contract_hash(row)
        assert row["contract_status"] in {"ready_adapted", "blocked_ambiguous", "diagnostic_only"}


def test_blocked_profile_prompt_cannot_be_execution_ready():
    profile = [row for row in prompt_contract_rows() if row["task_type"] == "profile_generation"][0]

    assert profile["contract_status"] == "blocked_ambiguous"
    assert profile["blocking_reason"]


def test_parser_contract_modes_are_distinct():
    rows = parser_contract_rows()
    modes = {(row["task_type"], row["parser_mode"]) for row in rows}

    assert ("ranking", "strict_primary_parser") in modes
    assert ("ranking", "diagnostic_salvage_parser") in modes
    strict = [row for row in rows if row["parser_mode"] == "strict_primary_parser"]
    assert all(row["allows_fuzzy_title_matching"] == "false" for row in strict)
    assert all(row["allows_code_fence"] == "false" for row in strict)


def test_history_rendering_preserves_order_truncates_and_rejects_missing_title():
    history = [{"item_id": f"i{i}", "title": f"Title {i}"} for i in range(12)]

    rendered = render_history(history, max_items=3)

    assert rendered.splitlines() == ["1. Title 9", "2. Title 10", "3. Title 11"]
    with pytest.raises(ValueError, match="missing"):
        render_history([{"item_id": "001", "title": ""}])


def test_load_histories_uses_train_only_and_chronology(tmp_path, monkeypatch):
    root = tmp_path
    interactions_path = root / "interactions.csv"
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "test", "split": "test", "timestamp": "1"},
            {"user_id": "u1", "item_id": "train2", "split": "train", "timestamp": "2"},
            {"user_id": "u1", "item_id": "train1", "split": "train", "timestamp": "1"},
        ]
    ).to_csv(interactions_path, index=False)
    monkeypatch.setattr(preflight, "DATASET_INTERACTIONS", {"Synthetic": "interactions.csv"})

    histories = preflight.load_histories(
        root,
        {
            ("Synthetic", "train1"): "Train One",
            ("Synthetic", "train2"): "Train Two",
            ("Synthetic", "test"): "Test",
        },
    )

    assert [item["item_id"] for item in histories["Synthetic"]["u1"]] == ["train1", "train2"]


def test_candidate_rendering_requires_20_preserves_order_and_duplicate_titles():
    candidates = [{"item_id": f"i{i}", "title": "Same" if i < 2 else f"Title {i}"} for i in range(20)]

    rendered = render_candidates(candidates)

    assert rendered.splitlines()[0] == "1. Same"
    assert rendered.splitlines()[1] == "2. Same"
    assert rendered.splitlines()[-1] == "20. Title 19"
    with pytest.raises(ValueError, match="20"):
        render_candidates(candidates[:19])
    bad = list(candidates)
    bad[0] = {"item_id": "i0", "title": ""}
    with pytest.raises(ValueError, match="missing"):
        render_candidates(bad)


def test_render_prompt_hides_positive_label_and_uses_indices():
    history = [{"item_id": f"h{i}", "title": f"History {i}"} for i in range(10)]
    candidates = [{"item_id": f"c{i}", "title": f"Candidate {i}"} for i in range(20)]

    system_message, user_message, _, _ = render_prompt(
        task_type="ranking",
        history_items=history,
        candidate_items=candidates,
    )

    assert "positive_item_id" not in user_message.lower()
    assert "ranked_candidate_indices" in user_message
    assert "1. Candidate 0" in user_message
    assert system_message


def test_strict_parser_accepts_valid_one_based_json_indices():
    result = parse_strict_index_response(
        '{"ranked_candidate_indices":[1,2,5]}',
        candidate_items=["a", "b", "c", "d", "e"],
        expected_count=3,
    )

    assert result.parse_valid
    assert result.output_items == ("a", "b", "e")


def test_strict_parser_rejects_leading_prose_code_fences_duplicates_unknown_and_lengths():
    candidates = ["a", "b", "c", "d", "e"]

    assert not parse_strict_index_response("Here: {\"ranked_candidate_indices\":[1,2,3]}", candidate_items=candidates, expected_count=3).parse_valid
    assert not parse_strict_index_response("```json\n{\"ranked_candidate_indices\":[1,2,3]}\n```", candidate_items=candidates, expected_count=3).parse_valid
    assert parse_strict_index_response('{"ranked_candidate_indices":[1,1,2]}', candidate_items=candidates, expected_count=3).status == "duplicate_indices"
    assert parse_strict_index_response('{"ranked_candidate_indices":[0,1,2]}', candidate_items=candidates, expected_count=3).status == "unknown_indices"
    assert parse_strict_index_response('{"ranked_candidate_indices":[1,2]}', candidate_items=candidates, expected_count=3).status == "invalid_length"
    assert parse_strict_index_response('{"ranked_candidate_indices":[1,2,3,4]}', candidate_items=candidates, expected_count=3).status == "invalid_length"
    assert parse_strict_index_response('{"ranked_candidate_indices":["1","2","3"]}', candidate_items=candidates, expected_count=3).status == "invalid_schema"


def test_strict_parser_rejects_incomplete_full_permutation():
    result = parse_strict_index_response(
        '{"ranked_candidate_indices":[1,2,3,4,5]}',
        candidate_items=["a", "b", "c", "d", "e"],
        expected_count=5,
        require_full_permutation=True,
    )
    assert result.parse_valid

    missing = parse_strict_index_response(
        '{"ranked_candidate_indices":[1,2,3,4,4]}',
        candidate_items=["a", "b", "c", "d", "e"],
        expected_count=5,
        require_full_permutation=True,
    )
    assert not missing.parse_valid


def test_parser_does_not_accept_positive_label_assisted_repair():
    response = '{"positive_item_id":"c","ranked_candidate_indices":[1,2]}'

    result = parse_strict_index_response(response, candidate_items=["a", "b", "c"], expected_count=3)

    assert not result.parse_valid
    assert result.output_items == ()


def test_salvage_parser_records_transformations_without_overwriting_strict_result():
    text = 'Answer:\n```json\n{"ranked_candidate_indices":["1","2","3"]}\n```'

    strict = parse_strict_index_response(text, candidate_items=["a", "b", "c"], expected_count=3)
    salvage = parse_diagnostic_salvage_response(text, candidate_items=["a", "b", "c"], expected_count=3)

    assert not strict.parse_valid
    assert salvage.parse_valid
    assert set(salvage.transformations) == {"extracted_json_object", "converted_integer_strings"}


def test_request_count_multipliers_and_profile_workflow():
    base, total = request_count(
        10,
        repetitions_per_row=3,
        position_permutations_per_row=2,
        history_variants_per_row=4,
        profile_generation_requests_per_row=1,
        recommendation_requests_per_row=1,
    )

    assert base == 240
    assert total == 250
    with pytest.raises(ValueError):
        request_count(1, repetitions_per_row=0)


def test_token_estimates_are_deterministic_envelopes():
    first = estimate_token_envelope("one two three")
    second = estimate_token_envelope("one two three")

    assert first == second
    character_count, whitespace_count, token_min, token_max = first
    assert character_count == 13
    assert whitespace_count == 3
    assert 1 <= token_min <= token_max


def test_policy_validation_requires_one_recommendation_and_forbids_positive_or_result_dependent():
    rows = [
        {"ambiguity_category": "parser", "recommendation_status": "recommended", "uses_positive_label": "false", "result_dependent": "false"},
        {"ambiguity_category": "parser", "recommendation_status": "allowed_not_recommended", "uses_positive_label": "false", "result_dependent": "false"},
    ]
    validate_one_recommended_policy_per_category(rows)

    with pytest.raises(ValueError):
        validate_one_recommended_policy_per_category(
            [{"ambiguity_category": "parser", "recommendation_status": "recommended", "uses_positive_label": "true", "result_dependent": "false"}]
        )
    with pytest.raises(ValueError):
        validate_one_recommended_policy_per_category(
            [{"ambiguity_category": "parser", "recommendation_status": "recommended", "uses_positive_label": "false", "result_dependent": "true"}]
        )
    with pytest.raises(ValueError):
        validate_one_recommended_policy_per_category(
            [
                {"ambiguity_category": "parser", "recommendation_status": "recommended", "uses_positive_label": "false", "result_dependent": "false"},
                {"ambiguity_category": "parser", "recommendation_status": "recommended", "uses_positive_label": "false", "result_dependent": "false"},
            ]
        )


def test_json_outputs_from_parser_are_serializable():
    result = parse_strict_index_response(
        json.dumps({"ranked_candidate_indices": [1, 2, 3]}),
        candidate_items=["001", "002", "003"],
        expected_count=3,
    )

    assert json.dumps(list(result.output_items)) == '["001", "002", "003"]'

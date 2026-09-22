from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from src.baselines.plan_e_gru4rec_adapter import (
    CandidateValidationError,
    ChronologyValidationError,
    MappingValidationError,
    PaddingValidationError,
    ScoreAlignmentError,
    align_scores_to_candidate_ids,
    audit_index_space,
    audit_plan_e2_gru4rec_candidate_row,
    build_gru4rec_model_index,
    build_training_histories,
    ensure_gru4rec_tokens_within_cardinality,
    filter_train_interactions,
    gru4rec_exclusion_reasons,
    json_array,
    load_raw_to_internal_mapping,
    make_prefix_target_pairs,
    pad_history_left,
    padding_mask_from_tokens,
    parse_candidate_items,
    prepare_candidate_tensor_payload,
    safe_raw_id,
    sequence_length_from_tokens,
    sort_training_interactions,
    validate_mapping_direction,
    validate_padding_index_semantics,
    validate_raw_to_internal_pairs,
)


def test_raw_ids_remain_strings_and_leading_zeros_are_preserved() -> None:
    assert safe_raw_id("00123") == "00123"
    assert safe_raw_id(123) == "123"
    assert json.loads(json_array(["001", "A02"])) == ["001", "A02"]


def test_safe_raw_id_rejects_null_like_values() -> None:
    with pytest.raises(MappingValidationError):
        safe_raw_id("nan")
    with pytest.raises(MappingValidationError):
        safe_raw_id("  ")


def test_timestamp_sorting_and_train_only_filtering_exclude_validation_and_test() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1"],
            "item_id": ["heldout", "i3", "i1", "i2"],
            "split": ["test", "train", "train", "validation"],
            "timestamp": ["4", "3", "1", "2"],
        }
    )
    histories, chronology, ordered = build_training_histories(interactions)
    assert histories["u1"] == ("i1", "i3")
    assert "heldout" not in histories["u1"]
    assert list(ordered["item_id"]) == ["i1", "i3"]
    assert chronology.timestamp_status == "present_numeric"


def test_equal_timestamp_tie_breaking_is_deterministic_and_detected() -> None:
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i2", "i1", "i3"],
            "split": ["train", "train", "train"],
            "timestamp": ["5", "5", "6"],
        }
    )
    ordered, chronology = sort_training_interactions(filter_train_interactions(train))
    assert list(ordered["item_id"]) == ["i2", "i1", "i3"]
    assert chronology.equal_timestamp_tie_rule == "input_order_after_repository_csv_read"
    assert chronology.chronology_ambiguous_user_count == 1
    assert "u1" in chronology.ambiguous_user_ids


def test_missing_timestamp_fails_when_chronology_is_required() -> None:
    train = pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"], "split": ["train"]})
    with pytest.raises(ChronologyValidationError):
        sort_training_interactions(filter_train_interactions(train), require_timestamp=True)


def test_sequence_construction_counts_zero_one_two_event_users() -> None:
    histories = {"u1": (), "u2": ("i1",), "u3": ("i1", "i2")}
    pairs = make_prefix_target_pairs(histories)
    assert [pair["user_id"] for pair in pairs] == ["u3"]
    assert pairs[0]["prefix_item_ids"] == ("i1",)
    assert pairs[0]["target_item_id"] == "i2"


def test_maximum_length_truncation_padding_direction_mask_and_length() -> None:
    item_index = {"i1": 0, "i2": 1, "i3": 2, "i4": 3}
    tokens = pad_history_left(["i1", "i2", "i3", "i4"], item_index, max_sequence_length=3)
    assert tokens == (2, 3, 4)
    padded = pad_history_left(["i2", "i4"], item_index, max_sequence_length=5)
    assert padded == (0, 0, 0, 2, 4)
    assert sequence_length_from_tokens(padded) == 2
    assert padding_mask_from_tokens(padded) == (False, False, False, True, True)


def test_mapping_validation_duplicate_ids_direction_and_inverse(tmp_path) -> None:
    with pytest.raises(MappingValidationError):
        validate_raw_to_internal_pairs([("u1", 1), ("u1", 2)], entity_type="user")
    with pytest.raises(MappingValidationError):
        validate_raw_to_internal_pairs([("u1", 1), ("u2", 1)], entity_type="user")
    with pytest.raises(MappingValidationError):
        validate_mapping_direction("raw_id")
    path = tmp_path / "datamaps.json"
    path.write_text(
        json.dumps(
            {
                "user2id": {"001": "7"},
                "id2user": {"7": "001"},
                "item2id": {"i1": "0"},
                "id2item": {"0": "i1"},
            }
        ),
        encoding="utf-8",
    )
    mapping = load_raw_to_internal_mapping(path, entity_type="user")
    assert mapping.raw_to_internal["001"] == 7
    audit = audit_index_space(load_raw_to_internal_mapping(path, entity_type="item"))
    assert "valid_source_entity" in audit.index_zero_semantics


def test_padding_index_semantics_offset_and_out_of_range_tokens() -> None:
    item_index = {"i0": 0, "i1": 1}
    validate_padding_index_semantics(item_index)
    assert pad_history_left(["i0", "i1"], item_index, max_sequence_length=2) == (1, 2)
    with pytest.raises(PaddingValidationError):
        validate_padding_index_semantics(item_index, padding_index=1)
    with pytest.raises(PaddingValidationError):
        ensure_gru4rec_tokens_within_cardinality([0, 1, 3], model_cardinality=3)


def test_unknown_ids_receive_no_fallback() -> None:
    with pytest.raises(CandidateValidationError):
        pad_history_left(["missing"], {"i1": 0}, max_sequence_length=2)
    with pytest.raises(CandidateValidationError):
        prepare_candidate_tensor_payload(
            user_id="u1",
            history_item_ids=["i1"],
            candidate_item_ids=["i1", "missing"],
            item_model_index={"i1": 0},
            expected_candidate_count=None,
        )


def test_candidate_order_membership_and_positive_position_are_preserved_without_sorting() -> None:
    candidates = ["i02", "i01", "i03"] + [f"i{x:02d}" for x in range(4, 21)]
    item_index = {item: idx for idx, item in enumerate(candidates)}
    payload = prepare_candidate_tensor_payload(
        user_id="u1",
        history_item_ids=["i02", "i01"],
        candidate_item_ids=candidates,
        item_model_index=item_index,
        max_sequence_length=4,
    )
    assert list(payload.raw_candidate_item_ids) == candidates
    assert list(payload.candidate_item_tokens) == list(range(1, 21))
    assert payload.sequence_tokens == (0, 0, 1, 2)


def test_malformed_candidate_lists_fail() -> None:
    with pytest.raises(CandidateValidationError):
        parse_candidate_items('{"not": "a list"}')
    with pytest.raises(CandidateValidationError):
        parse_candidate_items(json.dumps(["i1", "i1"] + [f"i{x}" for x in range(2, 20)]))


def test_fully_compatible_row_and_train_unseen_candidates_are_classified() -> None:
    row, users, items, user_mapping, item_mapping, item_index = _compatibility_fixture()
    audited = audit_plan_e2_gru4rec_candidate_row(
        row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={"u1": ("i00", "i01", "i02")},
        train_seen_items=items,
        item_model_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert audited["compatibility_status"] == "sequentially_compatible"
    assert audited["sequentially_compatible"]
    assert audited["training_pair_available"]
    assert json.loads(audited["train_unseen_candidate_item_ids"]) == []

    row_with_unseen = dict(row)
    candidates = json.loads(row_with_unseen["candidate_items"])
    candidates[-1] = "i99"
    row_with_unseen["candidate_items"] = json.dumps(candidates)
    item_mapping_unseen = validate_raw_to_internal_pairs(
        [(item, idx + 1) for idx, item in enumerate(candidates)],
        entity_type="item",
    )
    audited_unseen = audit_plan_e2_gru4rec_candidate_row(
        row_with_unseen,
        user_mapping=user_mapping,
        item_mapping=item_mapping_unseen,
        train_histories={"u1": ("i00", "i01", "i02")},
        train_seen_items=items,
        item_model_index=item_index,
        eligible_full=True,
        eligible_matched=False,
    )
    assert audited_unseen["compatibility_status"] == "candidate_train_unseen"
    assert "candidate_item_not_seen_in_train" in audited_unseen["exclusion_reasons"]
    assert json.loads(audited_unseen["train_unseen_candidate_item_ids"]) == ["i99"]


def test_missing_history_insufficient_history_mapping_failure_and_temporal_ambiguity_statuses() -> None:
    row, _, items, user_mapping, item_mapping, item_index = _compatibility_fixture()
    missing_history = audit_plan_e2_gru4rec_candidate_row(
        row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={},
        train_seen_items=items,
        item_model_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert missing_history["compatibility_status"] == "history_unavailable"

    one_event = audit_plan_e2_gru4rec_candidate_row(
        row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={"u1": ("i00",)},
        train_seen_items=items,
        item_model_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert one_event["compatibility_status"] == "insufficient_training_history"

    bad_mapping = dict(row)
    candidates = json.loads(bad_mapping["candidate_items"])
    candidates[-1] = "missing"
    bad_mapping["candidate_items"] = json.dumps(candidates)
    mapping_failure = audit_plan_e2_gru4rec_candidate_row(
        bad_mapping,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={"u1": ("i00", "i01")},
        train_seen_items=items,
        item_model_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert mapping_failure["compatibility_status"] == "mapping_incompatible"

    temporal = audit_plan_e2_gru4rec_candidate_row(
        row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={"u1": ("i00", "i01")},
        train_seen_items=items,
        item_model_index=item_index,
        eligible_full=True,
        eligible_matched=True,
        chronology_verified=False,
        temporal_protocol_unverifiable=True,
    )
    assert temporal["compatibility_status"] == "temporal_protocol_unverifiable"


def test_deterministic_exclusion_reasons_and_json_serialization() -> None:
    reasons = gru4rec_exclusion_reasons(
        user_mapped=False,
        positive_item_mapped=False,
        unmapped_candidate_count=2,
        history_available=False,
        training_pair_available=False,
        positive_item_seen=False,
        train_unseen_candidate_count=3,
        padding_index_compatible=False,
        chronology_verified=False,
        temporal_protocol_unverifiable=True,
    )
    assert reasons[:3] == ["user_unmapped", "positive_item_unmapped", "candidate_item_unmapped"]
    assert json.loads(json_array(["002", "001"])) == ["002", "001"]


def test_score_vector_alignment_requires_finite_values_and_keeps_order() -> None:
    aligned = align_scores_to_candidate_ids(["i2", "i1"], [0.2, 0.9])
    assert [row["candidate_item_id"] for row in aligned] == ["i2", "i1"]
    assert [row["candidate_position_zero_based"] for row in aligned] == [0, 1]
    with pytest.raises(ScoreAlignmentError):
        align_scores_to_candidate_ids(["i1"], [0.1, 0.2])
    with pytest.raises(ScoreAlignmentError):
        align_scores_to_candidate_ids(["i1"], [math.inf])


def test_model_index_uses_train_seen_items_and_natural_sort() -> None:
    index = build_gru4rec_model_index(["10", "2", "001"])
    assert list(index) == ["001", "2", "10"]


def test_synthetic_gru4rec_forward_pass_without_optimizer_backward_or_checkpoint() -> None:
    import torch

    from src.baselines.plan_d_gru4rec import GRU4RecModel

    model = GRU4RecModel(num_items=3, embedding_dim=4, hidden_dim=4, num_layers=1, dropout=0.0)
    sequence = torch.tensor([[0, 1, 2]], dtype=torch.long)
    candidates = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        scores = model.score_candidates(sequence, candidates)
    assert tuple(scores.shape) == (1, 3)

    if torch.cuda.is_available():
        cuda_model = GRU4RecModel(num_items=3, embedding_dim=4, hidden_dim=4, num_layers=1, dropout=0.0).to("cuda")
        cuda_sequence = sequence.to("cuda")
        cuda_candidates = candidates.to("cuda")
        with torch.no_grad():
            cuda_scores = cuda_model.score_candidates(cuda_sequence, cuda_candidates)
        assert tuple(cuda_scores.shape) == (1, 3)


def _compatibility_fixture():
    candidates = [f"i{x:02d}" for x in range(20)]
    row = {
        "candidate_set_id": "synthetic_plan_e2",
        "source_path": "synthetic.csv",
        "dataset": "Synthetic",
        "task_type": "ranking",
        "user_id": "u1",
        "positive_item_id": "i00",
        "candidate_items": json.dumps(candidates),
    }
    users = {"u1"}
    items = set(candidates)
    user_mapping = validate_raw_to_internal_pairs([("u1", 1)], entity_type="user")
    item_mapping = validate_raw_to_internal_pairs(
        [(item, idx + 1) for idx, item in enumerate(candidates)],
        entity_type="item",
    )
    item_index = build_gru4rec_model_index(items)
    return row, users, items, user_mapping, item_mapping, item_index

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from src.baselines.plan_e_bprmf_adapter import (
    CandidateValidationError,
    MappingValidationError,
    ScoreAlignmentError,
    align_scores_to_candidate_ids,
    audit_index_space,
    audit_plan_e2_candidate_row,
    build_existing_model_index,
    compute_train_seen_entities,
    ensure_indices_within_cardinality,
    filter_train_interactions,
    json_array,
    load_raw_to_internal_mapping,
    parse_candidate_items,
    prepare_candidate_tensor_payload,
    safe_raw_id,
    validate_mapping_direction,
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


def test_raw_to_internal_mapping_works_and_preserves_ids() -> None:
    mapping = validate_raw_to_internal_pairs(
        [("001", "7"), ("ABC", 9)],
        entity_type="item",
        source_path="synthetic",
    )
    assert mapping.raw_to_internal["001"] == 7
    assert mapping.internal_to_raw[7] == "001"
    assert mapping.internal_to_raw[9] == "ABC"


def test_duplicate_raw_mapping_ids_fail() -> None:
    with pytest.raises(MappingValidationError):
        validate_raw_to_internal_pairs([("u1", 1), ("u1", 2)], entity_type="user")


def test_duplicate_internal_mapping_ids_fail() -> None:
    with pytest.raises(MappingValidationError):
        validate_raw_to_internal_pairs([("u1", 1), ("u2", 1)], entity_type="user")


def test_ambiguous_mapping_direction_fails() -> None:
    with pytest.raises(MappingValidationError):
        validate_mapping_direction("raw_id")


def test_datamap_loader_requires_explicit_inverse(tmp_path) -> None:
    path = tmp_path / "datamaps.json"
    path.write_text(
        json.dumps(
            {
                "user2id": {"u1": "1", "u2": "2"},
                "id2user": {"1": "u1", "2": "u2"},
                "item2id": {"i1": "1"},
                "id2item": {"1": "i1"},
            }
        ),
        encoding="utf-8",
    )
    mapping = load_raw_to_internal_mapping(path, entity_type="user")
    assert mapping.raw_to_internal == {"u1": 1, "u2": 2}


def test_out_of_range_internal_indices_fail() -> None:
    with pytest.raises(MappingValidationError):
        ensure_indices_within_cardinality([0, 3], cardinality=3, entity_type="item")


def test_non_contiguous_index_spaces_are_detected_and_zero_semantics_recorded() -> None:
    mapping = validate_raw_to_internal_pairs([("i1", 1), ("i3", 3)], entity_type="item")
    audit = audit_index_space(mapping)
    assert not audit.contiguous_index_space
    assert audit.missing_internal_index_count == 1
    assert "absent_from_datamap" in audit.index_zero_semantics


def test_only_train_interactions_count_as_seen() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "item_id": ["i1", "i2", "i3"],
            "split": ["train", "validation", "test"],
        }
    )
    train = filter_train_interactions(interactions)
    users, items = compute_train_seen_entities(train)
    assert users == {"u1"}
    assert items == {"i1"}


def test_candidate_order_and_raw_to_internal_alignment_are_preserved() -> None:
    candidates = ["i02", "i01", "i03"] + [f"i{x:02d}" for x in range(4, 21)]
    model_index = {item: idx for idx, item in enumerate(candidates)}
    payload = prepare_candidate_tensor_payload(
        user_id="u1",
        candidate_item_ids=candidates,
        user_model_index={"u1": 0},
        item_model_index=model_index,
    )
    assert list(payload.raw_candidate_item_ids) == candidates
    assert list(payload.item_indices) == list(range(20))


def test_malformed_candidate_lists_fail() -> None:
    with pytest.raises(CandidateValidationError):
        parse_candidate_items('{"not": "a list"}')
    with pytest.raises(CandidateValidationError):
        parse_candidate_items(json.dumps(["i1", "i1"] + [f"i{x}" for x in range(2, 20)]))


def test_unknown_ids_do_not_receive_fallback_indices() -> None:
    with pytest.raises(CandidateValidationError):
        prepare_candidate_tensor_payload(
            user_id="u1",
            candidate_item_ids=["i1", "missing"],
            user_model_index={"u1": 0},
            item_model_index={"i1": 0},
        )


def test_strict_train_known_compatible_row() -> None:
    row, users, items, user_mapping, item_mapping, user_index, item_index = _compatibility_fixture()
    audited = audit_plan_e2_candidate_row(
        row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_seen_users=users,
        train_seen_items=items,
        user_model_index=user_index,
        item_model_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert audited["compatibility_status"] == "strict_train_known"
    assert audited["strict_train_known_compatible"]
    assert not audited["contains_untrained_embedding_risk"]
    assert json.loads(audited["train_unseen_candidate_item_ids"]) == []


def test_index_compatible_row_with_unseen_candidate_risk() -> None:
    row, users, items, user_mapping, item_mapping, user_index, item_index = _compatibility_fixture()
    candidates = json.loads(row["candidate_items"])
    candidates[-1] = "i99"
    row["candidate_items"] = json.dumps(candidates)
    item_mapping = validate_raw_to_internal_pairs(
        [(item, idx + 1) for idx, item in enumerate(candidates)],
        entity_type="item",
    )
    item_index = build_existing_model_index(train_seen_ids=items, plan_e2_ids=set(candidates))
    audited = audit_plan_e2_candidate_row(
        row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_seen_users=users,
        train_seen_items=items,
        user_model_index=user_index,
        item_model_index=item_index,
        eligible_full=True,
        eligible_matched=False,
    )
    assert audited["compatibility_status"] == "index_compatible_with_untrained_embedding_risk"
    assert audited["existing_model_index_compatible"]
    assert not audited["strict_train_known_compatible"]
    assert audited["contains_untrained_embedding_risk"]
    assert json.loads(audited["train_unseen_candidate_item_ids"]) == ["i99"]


def test_mapping_incompatible_row_and_user_not_seen_statuses_are_explicit() -> None:
    row, users, items, user_mapping, item_mapping, user_index, item_index = _compatibility_fixture()
    missing_candidate_row = dict(row)
    candidates = json.loads(missing_candidate_row["candidate_items"])
    candidates[-1] = "missing"
    missing_candidate_row["candidate_items"] = json.dumps(candidates)
    audited = audit_plan_e2_candidate_row(
        missing_candidate_row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_seen_users=users,
        train_seen_items=items,
        user_model_index=user_index,
        item_model_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert audited["compatibility_status"] == "mapping_incompatible"
    assert "candidate_item_unmapped" in audited["exclusion_reasons"]

    user_not_seen_row = dict(row)
    user_not_seen_row["user_id"] = "u2"
    user_mapping = validate_raw_to_internal_pairs([("u1", 1), ("u2", 2)], entity_type="user")
    user_index = build_existing_model_index(train_seen_ids=users, plan_e2_ids={"u2"})
    audited_user = audit_plan_e2_candidate_row(
        user_not_seen_row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_seen_users=users,
        train_seen_items=items,
        user_model_index=user_index,
        item_model_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert audited_user["compatibility_status"] == "user_not_seen_in_train"
    assert "user_not_seen_in_train" in audited_user["exclusion_reasons"]


def test_score_alignment_requires_exact_finite_vector_and_does_not_sort() -> None:
    aligned = align_scores_to_candidate_ids(["i2", "i1"], [0.2, 0.9])
    assert [row["candidate_item_id"] for row in aligned] == ["i2", "i1"]
    assert [row["candidate_position_zero_based"] for row in aligned] == [0, 1]
    with pytest.raises(ScoreAlignmentError):
        align_scores_to_candidate_ids(["i1"], [0.1, 0.2])
    with pytest.raises(ScoreAlignmentError):
        align_scores_to_candidate_ids(["i1"], [math.inf])


def test_synthetic_bprmf_forward_pass_without_optimizer_or_backward() -> None:
    import torch

    from src.baselines.plan_d_bprmf import BPRMFModel

    model = BPRMFModel(num_users=2, num_items=3, embedding_dim=4)
    users = torch.tensor([0, 1], dtype=torch.long)
    positives = torch.tensor([0, 1], dtype=torch.long)
    negatives = torch.tensor([2, 2], dtype=torch.long)
    positive_scores, negative_scores = model(users, positives, negatives)
    assert tuple(positive_scores.shape) == (2,)
    assert tuple(negative_scores.shape) == (2,)

    if torch.cuda.is_available():
        cuda_model = BPRMFModel(num_users=2, num_items=3, embedding_dim=4).to("cuda")
        cuda_users = users.to("cuda")
        cuda_positives = positives.to("cuda")
        cuda_negatives = negatives.to("cuda")
        cuda_positive, cuda_negative = cuda_model(cuda_users, cuda_positives, cuda_negatives)
        assert tuple(cuda_positive.shape) == (2,)
        assert tuple(cuda_negative.shape) == (2,)


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
    user_index = build_existing_model_index(train_seen_ids=users, plan_e2_ids={"u1"})
    item_index = build_existing_model_index(train_seen_ids=items, plan_e2_ids=set(candidates))
    return row, users, items, user_mapping, item_mapping, user_index, item_index

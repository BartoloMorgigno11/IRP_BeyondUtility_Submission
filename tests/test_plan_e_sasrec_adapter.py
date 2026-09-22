from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from src.baselines.plan_d_sasrec import SASRecModel
from src.baselines.plan_e_sasrec_adapter import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
    CandidateValidationError,
    ChronologyValidationError,
    EntityMapping,
    MappingValidationError,
    MaskValidationError,
    PaddingValidationError,
    ScoreAlignmentError,
    TargetLeakageError,
    audit_plan_e2_sasrec_candidate_row,
    build_sasrec_token_index,
    build_training_histories,
    causal_attention_mask,
    final_non_padding_position,
    freeze_source_row_index,
    make_sasrec_training_examples,
    pad_history_left,
    padding_key_mask_from_tokens,
    parse_candidate_items,
    position_indices_for_sequence,
    prepare_candidate_tensor_payload,
    safe_raw_id,
    sort_chronological_train_interactions,
    validate_causal_mask,
    validate_mapping_direction,
    validate_padding_index_semantics,
    validate_padding_mask,
    validate_raw_to_internal_pairs,
    validate_score_vector_alignment,
)


def test_chronology_freezes_source_row_index_and_sorts_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "item_id": ["B", "A", "C", "D"],
            "split": ["train", "train", "train", "train"],
            "timestamp": ["2", "1", "2", "1"],
        }
    )
    frozen = freeze_source_row_index(frame)
    ordered, audit = sort_chronological_train_interactions(frozen)
    assert ordered.loc[ordered["user_id"] == "u1", "item_id"].tolist() == ["A", "B", "C"]
    assert ordered.loc[ordered["user_id"] == "u1", "source_row_index"].tolist() == [1, 0, 2]
    assert audit.equal_timestamp_group_count == 1
    assert audit.equal_timestamp_tie_rule == "frozen_persisted_source_row_order"


def test_chronology_rejects_missing_or_malformed_timestamps() -> None:
    missing = pd.DataFrame({"user_id": ["u1"], "item_id": ["A"], "split": ["train"]})
    malformed = pd.DataFrame({"user_id": ["u1"], "item_id": ["A"], "split": ["train"], "timestamp": ["bad"]})
    with pytest.raises(ChronologyValidationError, match="missing timestamp"):
        sort_chronological_train_interactions(freeze_source_row_index(missing))
    with pytest.raises(ChronologyValidationError, match="non-numeric"):
        build_training_histories(malformed)


def test_training_histories_use_train_split_only_and_exclude_validation_test() -> None:
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1"],
            "item_id": ["A", "B", "V", "T"],
            "split": ["train", "train", "validation", "test"],
            "timestamp": ["1", "2", "3", "4"],
        }
    )
    audit = build_training_histories(frame, max_sequence_length=3)
    assert audit.histories["u1"] == ("A", "B")
    assert audit.train_rows == 2
    assert audit.validation_rows == 1
    assert audit.test_rows == 1


def test_training_objective_lengths_and_shifted_targets() -> None:
    assert make_sasrec_training_examples({"u0": []}) == []
    assert make_sasrec_training_examples({"u1": ["A"]}) == []
    examples = make_sasrec_training_examples({"u2": ["A", "B", "C"]}, max_sequence_length=2)
    assert [row["prefix_item_ids"] for row in examples] == [("A",), ("A", "B")]
    assert [row["target_item_id"] for row in examples] == ["B", "C"]
    assert all(row["target_item_id"] not in row["prefix_item_ids"][-1:] for row in examples)


def test_padding_offset_truncation_and_positions() -> None:
    index = build_sasrec_token_index(["001", "A", "B", "C"])
    validate_padding_index_semantics(index)
    tokens = pad_history_left(["001", "A", "B", "C"], index, max_sequence_length=3)
    assert tokens == (2, 3, 4)
    assert 0 not in tokens
    padded = pad_history_left(["001"], index, max_sequence_length=4)
    assert padded == (0, 0, 0, 1)
    assert position_indices_for_sequence(padded, max_sequence_length=4) == (0, 1, 2, 3)
    with pytest.raises(PaddingValidationError):
        position_indices_for_sequence((0, 1, 2), max_sequence_length=2)


def test_causal_and_padding_masks_have_plan_d_semantics() -> None:
    mask = causal_attention_mask(4)
    validate_causal_mask(mask)
    assert mask[0][1] is True
    assert mask[0][3] is True
    assert mask[2][1] is False
    assert mask[3][3] is False
    with pytest.raises(MaskValidationError):
        validate_causal_mask(((False, False), (True, False)))
    sequence = (0, 0, 3, 4)
    padding = padding_key_mask_from_tokens(sequence)
    assert padding == (True, True, False, False)
    validate_padding_mask(sequence, padding)
    with pytest.raises(MaskValidationError):
        validate_padding_mask(sequence, (False, False, True, True))


def test_left_padded_final_non_padding_position_and_fully_padded_failure() -> None:
    assert final_non_padding_position((0, 0, 1, 2)) == 3
    with pytest.raises(PaddingValidationError):
        final_non_padding_position((0, 0, 0))


def test_mapping_preserves_string_ids_and_rejects_duplicates() -> None:
    mapping = validate_raw_to_internal_pairs([("001", 1), ("1", 2)], entity_type="item")
    assert safe_raw_id("001") == "001"
    assert mapping.raw_to_internal["001"] == 1
    assert validate_mapping_direction("item2id") == "raw_to_internal"
    with pytest.raises(MappingValidationError, match="duplicate raw"):
        validate_raw_to_internal_pairs([("A", 1), ("A", 2)], entity_type="item")
    with pytest.raises(MappingValidationError, match="duplicate internal"):
        validate_raw_to_internal_pairs([("A", 1), ("B", 1)], entity_type="item")
    with pytest.raises(MappingValidationError):
        validate_mapping_direction("items")


def test_candidate_payload_preserves_order_and_rejects_unknowns() -> None:
    index = build_sasrec_token_index(["A", "B", "C", "D"])
    payload = prepare_candidate_tensor_payload(
        user_id="u1",
        history_item_ids=["A", "B"],
        candidate_item_ids=["D", "B", "C"],
        item_token_index=index,
        max_sequence_length=4,
        expected_candidate_count=3,
    )
    assert payload.raw_candidate_item_ids == ("D", "B", "C")
    assert payload.candidate_item_tokens == (4, 2, 3)
    assert payload.padding_key_mask == (True, True, False, False)
    with pytest.raises(CandidateValidationError, match="expected 3"):
        prepare_candidate_tensor_payload(
            user_id="u1",
            history_item_ids=["A"],
            candidate_item_ids=["A", "B"],
            item_token_index=index,
            expected_candidate_count=3,
        )
    with pytest.raises(CandidateValidationError, match="not in SASRec token index"):
        prepare_candidate_tensor_payload(
            user_id="u1",
            history_item_ids=["A"],
            candidate_item_ids=["Z"],
            item_token_index=index,
            expected_candidate_count=1,
        )


def test_score_vector_alignment_requires_one_finite_score_per_candidate() -> None:
    assert validate_score_vector_alignment(["A", "B"], [0.1, 0.2]) == (0.1, 0.2)
    with pytest.raises(ScoreAlignmentError, match="lengths differ"):
        validate_score_vector_alignment(["A"], [0.1, 0.2])
    with pytest.raises(ScoreAlignmentError, match="not finite"):
        validate_score_vector_alignment(["A"], [float("nan")])


def test_json_candidate_parsing_preserves_raw_ids() -> None:
    assert parse_candidate_items(json.dumps(["001", 2]), expected_count=2) == ["001", "2"]
    with pytest.raises(CandidateValidationError):
        parse_candidate_items('{"not": "array"}')


def test_row_compatibility_statuses_are_deterministic() -> None:
    user_mapping = EntityMapping("user", {"u1": 1}, {1: "u1"}, "test")
    item_mapping = EntityMapping("item", {f"i{i}": i for i in range(1, 22)}, {i: f"i{i}" for i in range(1, 22)}, "test")
    train_seen = {f"i{i}" for i in range(1, 21)}
    item_index = build_sasrec_token_index(train_seen)
    row = _candidate_row(user="u1", positive="i20", candidates=[f"i{i}" for i in range(1, 21)])
    compatible = audit_plan_e2_sasrec_candidate_row(
        row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={"u1": ["i1", "i2", "i3"]},
        train_seen_items=train_seen,
        item_token_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert compatible["compatibility_status"] == "sasrec_sequentially_compatible"
    assert compatible["sasrec_sequentially_compatible"] is True

    no_history = audit_plan_e2_sasrec_candidate_row(
        row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={},
        train_seen_items=train_seen,
        item_token_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert no_history["compatibility_status"] == "history_unavailable"

    no_objective = audit_plan_e2_sasrec_candidate_row(
        row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={"u1": ["i1"]},
        train_seen_items=train_seen,
        item_token_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert no_objective["compatibility_status"] == "training_objective_unavailable"

    unseen = audit_plan_e2_sasrec_candidate_row(
        _candidate_row(user="u1", positive="i21", candidates=[*(f"i{i}" for i in range(1, 20)), "i21"]),
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={"u1": ["i1", "i2"]},
        train_seen_items=train_seen,
        item_token_index=item_index,
        eligible_full=True,
        eligible_matched=False,
    )
    assert unseen["compatibility_status"] == "candidate_train_unseen"
    assert json.loads(unseen["train_unseen_candidate_item_ids"]) == ["i21"]


def test_row_compatibility_mapping_mask_and_leakage_failures() -> None:
    user_mapping = EntityMapping("user", {"u1": 1}, {1: "u1"}, "test")
    item_ids = [f"I{i:02d}" for i in range(1, 21)]
    item_mapping = EntityMapping(
        "item",
        {item_id: index for index, item_id in enumerate(item_ids, start=1)},
        {index: item_id for index, item_id in enumerate(item_ids, start=1)},
        "test",
    )
    item_index = build_sasrec_token_index(item_ids)
    row = _candidate_row(user="u1", positive="I20", candidates=item_ids)
    mapped = audit_plan_e2_sasrec_candidate_row(
        row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={"u1": ["I01", "I02"]},
        train_seen_items=set(item_ids),
        item_token_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert mapped["compatibility_status"] == "sasrec_sequentially_compatible"
    bad_candidates = [*item_ids[:-1], "Z99"]
    bad_mapping = audit_plan_e2_sasrec_candidate_row(
        _candidate_row(user="u1", positive="I19", candidates=bad_candidates),
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={"u1": ["I01", "I02"]},
        train_seen_items=set(item_ids),
        item_token_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert bad_mapping["compatibility_status"] == "mapping_incompatible"
    bad_mask = audit_plan_e2_sasrec_candidate_row(
        row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={"u1": ["I01", "I02"]},
        train_seen_items=set(item_ids),
        item_token_index=item_index,
        eligible_full=True,
        eligible_matched=True,
        causal_mask_verified=False,
    )
    assert bad_mask["compatibility_status"] == "causal_mask_unverified"
    leakage = audit_plan_e2_sasrec_candidate_row(
        row,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        train_histories={"u1": ["I01", "I20"]},
        train_seen_items=set(item_ids),
        item_token_index=item_index,
        eligible_full=True,
        eligible_matched=True,
    )
    assert leakage["compatibility_status"] == "target_leakage_detected"


def test_tiny_plan_d_sasrec_forward_only_cpu_and_optional_cuda() -> None:
    _run_synthetic_forward_probe(torch.device("cpu"))
    if torch.cuda.is_available():
        _run_synthetic_forward_probe(torch.device("cuda"))


def _run_synthetic_forward_probe(device: torch.device) -> None:
    model = SASRecModel(
        num_items=8,
        embedding_dim=4,
        max_seq_len=5,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    ).to(device)
    model.eval()
    sequence = torch.tensor([[0, 0, 1, 2, 3]], dtype=torch.long, device=device)
    positives = torch.tensor([4], dtype=torch.long, device=device)
    negatives = torch.tensor([5], dtype=torch.long, device=device)
    candidates = torch.tensor([[4, 5, 6]], dtype=torch.long, device=device)
    with torch.no_grad():
        positive_scores, negative_scores = model(sequence, positives, negatives)
        candidate_scores = model.score_candidates(sequence, candidates)
    assert positive_scores.shape == (1,)
    assert negative_scores.shape == (1,)
    assert candidate_scores.shape == (1, 3)


def _candidate_row(user: str, positive: str, candidates: list[str]) -> dict[str, str]:
    return {
        "candidate_set_id": "synthetic_candidates",
        "source_path": "synthetic.csv",
        "source_plan": "synthetic",
        "dataset": "Synthetic",
        "task_type": "ranking",
        "user_id": user,
        "positive_item_id": positive,
        "candidate_items": json.dumps(candidates),
    }

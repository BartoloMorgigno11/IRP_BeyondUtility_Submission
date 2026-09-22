from __future__ import annotations

import pandas as pd
import pytest

from src.data.schemas import (
    validate_candidates,
    validate_ground_truth,
    validate_interactions,
    validate_recommendations,
)


def test_validate_interactions_normalizes_columns_and_preserves_extra_columns() -> None:
    raw = pd.DataFrame(
        {
            " User_ID ": ["u1"],
            "ITEM_ID": ["i1"],
            " split ": ["test"],
            "rating": [5],
            "source": ["synthetic"],
        }
    )

    normalized = validate_interactions(raw)

    assert list(normalized.columns) == [
        "user_id",
        "item_id",
        "split",
        "rating",
        "source",
    ]
    assert " User_ID " in raw.columns
    assert normalized.to_dict("records") == [
        {
            "user_id": "u1",
            "item_id": "i1",
            "split": "test",
            "rating": 5,
            "source": "synthetic",
        }
    ]


def test_validate_interactions_rejects_missing_required_columns() -> None:
    raw = pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"]})

    with pytest.raises(ValueError, match="interactions.*split"):
        validate_interactions(raw)


def test_validate_candidates_rejects_missing_candidate_position() -> None:
    raw = pd.DataFrame({"user_id": ["u1"], "candidate_item_id": ["i1"]})

    with pytest.raises(ValueError, match="candidates.*candidate_position"):
        validate_candidates(raw)


def test_validate_candidates_accepts_empty_frame_with_required_columns() -> None:
    raw = pd.DataFrame(columns=["user_id", "candidate_item_id", "candidate_position"])

    normalized = validate_candidates(raw)

    assert list(normalized.columns) == [
        "user_id",
        "candidate_item_id",
        "candidate_position",
    ]
    assert normalized.empty


def test_validate_recommendations_accepts_item_text_without_item_id() -> None:
    raw = pd.DataFrame(
        {
            "user_id": ["u1"],
            "model_name": ["gpt-test"],
            "dataset": ["beauty"],
            "rank": [1],
            "recommended_item_text": ["Item title"],
            "run_id": ["run-001"],
            "raw_output": ["Item title"],
        }
    )

    normalized = validate_recommendations(raw)

    assert list(normalized.columns) == [
        "user_id",
        "model_name",
        "dataset",
        "rank",
        "run_id",
        "recommended_item_text",
        "raw_output",
    ]


def test_validate_recommendations_requires_recommendation_value() -> None:
    raw = pd.DataFrame(
        {
            "user_id": ["u1"],
            "model_name": ["gpt-test"],
            "dataset": ["beauty"],
            "rank": [1],
            "run_id": ["run-001"],
        }
    )

    with pytest.raises(ValueError, match="at least one"):
        validate_recommendations(raw)


def test_validate_recommendations_rejects_rows_without_item_id_or_text() -> None:
    raw = pd.DataFrame(
        {
            "user_id": ["u1"],
            "model_name": ["gpt-test"],
            "dataset": ["beauty"],
            "rank": [1],
            "run_id": ["run-001"],
            "recommended_item_id": [None],
            "recommended_item_text": [" "],
        }
    )

    with pytest.raises(ValueError, match="without any recommendation value"):
        validate_recommendations(raw)


def test_validate_ground_truth_normalizes_and_orders_columns() -> None:
    raw = pd.DataFrame(
        {
            "split": ["test"],
            " Ground_Truth_Item_ID ": ["i1"],
            "user_id": ["u1"],
            "dataset": ["sports"],
            "source": ["synthetic"],
        }
    )

    normalized = validate_ground_truth(raw)

    assert list(normalized.columns) == [
        "user_id",
        "ground_truth_item_id",
        "dataset",
        "split",
        "source",
    ]


def test_validators_reject_duplicate_column_names_after_normalization() -> None:
    raw = pd.DataFrame(
        [["u1", "u2", "i1", "test"]],
        columns=["user_id", " User_ID ", "item_id", "split"],
    )

    with pytest.raises(ValueError, match="duplicate column"):
        validate_interactions(raw)

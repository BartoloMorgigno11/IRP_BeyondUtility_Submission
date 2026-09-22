from __future__ import annotations

from math import log2

import pandas as pd
import pytest

from src.evaluation.utility_metrics import (
    evaluate_utility_at_k,
    hit_rate_at_k,
    ndcg_at_k,
)


def test_hit_rate_supports_multiple_relevant_items_and_missing_recommendations() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u2", "u3"],
            "recommended_item_id": ["i1", "x", "i4", "ignored"],
            "rank": [1, 1, 2, 1],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u2", "u4"],
            "ground_truth_item_id": ["i1", "i3", "i4", "i9"],
        }
    )

    assert hit_rate_at_k(recommendations, ground_truth, k=2) == pytest.approx(2 / 3)


def test_ndcg_supports_multiple_relevant_items_with_exact_expected_value() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "recommended_item_id": ["x", "b", "a"],
            "rank": [1, 2, 3],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "ground_truth_item_id": ["a", "b"],
        }
    )

    expected_dcg = (1 / log2(3)) + (1 / log2(4))
    expected_idcg = 1 + (1 / log2(3))

    assert ndcg_at_k(recommendations, ground_truth, k=3) == pytest.approx(
        expected_dcg / expected_idcg
    )


def test_rank_column_controls_recommendation_order_when_present() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "recommended_item_id": ["miss", "hit"],
            "rank": [2, 1],
        }
    )
    ground_truth = pd.DataFrame(
        {"user_id": ["u1"], "ground_truth_item_id": ["hit"]}
    )

    assert hit_rate_at_k(recommendations, ground_truth, k=1) == 1.0
    assert ndcg_at_k(recommendations, ground_truth, k=1) == 1.0


def test_input_order_is_used_when_rank_column_is_absent() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "recommended_item_id": ["miss", "hit"],
        }
    )
    ground_truth = pd.DataFrame(
        {"user_id": ["u1"], "ground_truth_item_id": ["hit"]}
    )

    assert hit_rate_at_k(recommendations, ground_truth, k=1) == 0.0
    assert ndcg_at_k(recommendations, ground_truth, k=1) == 0.0


def test_empty_recommendations_score_zero_for_users_with_ground_truth() -> None:
    recommendations = pd.DataFrame()
    ground_truth = pd.DataFrame(
        {"user_id": ["u1"], "ground_truth_item_id": ["i1"]}
    )

    assert hit_rate_at_k(recommendations, ground_truth, k=5) == 0.0
    assert ndcg_at_k(recommendations, ground_truth, k=5) == 0.0


def test_users_without_ground_truth_are_excluded_from_denominator() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "recommended_item_id": ["hit", "miss"],
            "rank": [1, 1],
        }
    )
    ground_truth = pd.DataFrame(
        {"user_id": ["u1"], "ground_truth_item_id": ["hit"]}
    )

    assert hit_rate_at_k(recommendations, ground_truth, k=1) == 1.0
    assert ndcg_at_k(recommendations, ground_truth, k=1) == 1.0


def test_duplicate_recommendations_keep_rank_positions_without_repeated_gain() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "recommended_item_id": ["a", "a", "b"],
            "rank": [1, 2, 3],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "ground_truth_item_id": ["a", "b"],
        }
    )

    expected_dcg = 1 + (1 / log2(4))
    expected_idcg = 1 + (1 / log2(3))

    assert ndcg_at_k(recommendations, ground_truth, k=3) == pytest.approx(
        expected_dcg / expected_idcg
    )


def test_recommendation_text_column_is_supported_when_item_id_is_absent() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1"],
            "recommended_item_text": ["Exact Title"],
            "rank": [1],
        }
    )
    ground_truth = pd.DataFrame(
        {"user_id": ["u1"], "ground_truth_item_id": ["Exact Title"]}
    )

    assert hit_rate_at_k(recommendations, ground_truth, k=1) == 1.0
    assert ndcg_at_k(recommendations, ground_truth, k=1) == 1.0


def test_evaluate_utility_at_k_returns_compact_summary() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "recommended_item_id": ["a", "x"],
            "rank": [1, 1],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "ground_truth_item_id": ["a", "b"],
        }
    )

    summary = evaluate_utility_at_k(recommendations, ground_truth, k=1)

    assert summary == {
        "k": 1,
        "user_count": 2,
        "hit_rate": 0.5,
        "ndcg": 0.5,
    }


def test_invalid_k_raises_clear_error() -> None:
    recommendations = pd.DataFrame()
    ground_truth = pd.DataFrame()

    with pytest.raises(ValueError, match="positive integer"):
        hit_rate_at_k(recommendations, ground_truth, k=0)

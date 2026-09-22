from __future__ import annotations

from math import log2

import pandas as pd
import pytest

from src.analysis.plan_d_beyond_utility import (
    average_popularity_at_k,
    build_item_feature_sets,
    candidate_validity_rate_at_k,
    catalog_coverage_at_k,
    catalog_item_set,
    duplicate_rate_at_k,
    intra_list_diversity_at_k,
    item_coverage_at_k,
    long_tail_rate_at_k,
    novelty_at_k,
    training_popularity,
)


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["A", "B", "C", "D", "E"],
            "categories": [
                "Action | Sci-Fi",
                "Drama",
                "Action",
                "Sports",
                "Sports | Fitness",
            ],
        }
    )


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "item_id": ["A", "A", "A", "B", "D"],
            "split": ["train", "train", "train", "train", "train"],
        }
    )


def _recommendations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1", "u2", "u2", "u2"],
            "recommended_item_id": ["A", "B", "B", "C", "E", "D", "E", "A"],
            "rank": [1, 2, 3, 4, 5, 1, 2, 3],
        }
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u2", "u2"],
            "candidate_item_id": ["A", "B", "C", "E", "D", "E"],
        }
    )


def test_coverage_metrics_count_unique_top5_items() -> None:
    recommendations = _recommendations()
    catalog = _catalog()

    assert item_coverage_at_k(recommendations, 5) == 5
    assert catalog_coverage_at_k(recommendations, catalog, 5) == pytest.approx(1.0)


def test_novelty_uses_documented_smoothing_formula() -> None:
    recommendations = _recommendations()
    popularity = training_popularity(_interactions())
    catalog = _catalog()

    actual = novelty_at_k(
        recommendations,
        popularity,
        train_interaction_count=int(popularity.sum()),
        catalog_size=len(catalog),
        k=5,
    )

    counts = [3, 1, 1, 0, 0, 1, 0, 3]
    expected = sum(-log2((count + 1) / 10) for count in counts) / len(counts)
    assert actual == pytest.approx(expected)


def test_average_popularity_and_long_tail_use_training_popularity() -> None:
    recommendations = _recommendations()
    popularity = training_popularity(_interactions())

    assert average_popularity_at_k(recommendations, popularity, 5) == pytest.approx(9 / 8)
    assert long_tail_rate_at_k(
        recommendations,
        popularity,
        catalog_item_set(_catalog()),
        5,
    ) == pytest.approx(3 / 8)


def test_duplicate_rate_is_averaged_per_user() -> None:
    assert duplicate_rate_at_k(_recommendations(), 5) == pytest.approx(0.1)


def test_candidate_validity_uses_user_specific_candidate_sets() -> None:
    assert candidate_validity_rate_at_k(
        _recommendations(),
        candidates=_candidates(),
        catalog=_catalog(),
        k=5,
    ) == pytest.approx(7 / 8)


def test_intra_list_diversity_uses_pairwise_jaccard_dissimilarity() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "recommended_item_id": ["A", "B", "C"],
            "rank": [1, 2, 3],
        }
    )
    features = build_item_feature_sets(
        pd.DataFrame(
            {
                "item_id": ["A", "B", "C"],
                "categories": ["x", "x | y", "z"],
            }
        ),
        dataset="ml1m",
    )

    assert intra_list_diversity_at_k(recommendations, features, 5) == pytest.approx(2.5 / 3)

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from src.evaluation.hallucination import (
    candidate_hallucination_rate,
    catalog_hallucination_rate,
    hallucination_rate,
)
from src.evaluation.output_parser import (
    MATCH_EXACT,
    MATCH_FUZZY,
    MATCH_ITEM_ID,
    MATCH_NORMALIZED,
    MATCH_UNMATCHED,
)


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["A1", "B2", "C3"],
            "item_title": ["Alpha Cream", "Beta Brush", "Gamma Mask"],
        }
    )


def test_hallucination_rate_distinguishes_match_types() -> None:
    recommendations = pd.DataFrame(
        {
            "match_type": [
                MATCH_ITEM_ID,
                MATCH_EXACT,
                MATCH_NORMALIZED,
                MATCH_FUZZY,
                MATCH_UNMATCHED,
            ]
        }
    )

    summary = hallucination_rate(recommendations)

    assert summary == {
        "total": 5,
        "hallucinated": 1,
        "hallucination_rate": 0.2,
        "strict_hallucinated": 2,
        "strict_hallucination_rate": 0.4,
        "item_id_matches": 1,
        "exact_matches": 1,
        "exact_or_item_id_matches": 2,
        "normalized_matches": 1,
        "fuzzy_matches": 1,
        "unmatched": 1,
    }


def test_fuzzy_match_is_never_counted_as_exact() -> None:
    recommendations = pd.DataFrame(
        {
            "recommended_item_id": ["A1"],
            "match_type": [MATCH_FUZZY],
        }
    )

    summary = hallucination_rate(recommendations)

    assert summary["exact_matches"] == 0
    assert summary["exact_or_item_id_matches"] == 0
    assert summary["fuzzy_matches"] == 1
    assert summary["hallucination_rate"] == 0.0
    assert summary["strict_hallucination_rate"] == 1.0


def test_catalog_hallucination_rate_matches_id_exact_normalized_and_unmatched() -> None:
    recommendations = pd.DataFrame(
        {
            "recommended_item_id": ["A1", None, None, None],
            "recommended_item_text": [
                "wrong title text",
                "Beta Brush",
                "gamma-mask!",
                "Unknown Product",
            ],
        }
    )

    summary = catalog_hallucination_rate(recommendations, _catalog())

    assert summary["total"] == 4
    assert summary["item_id_matches"] == 1
    assert summary["exact_matches"] == 1
    assert summary["normalized_matches"] == 1
    assert summary["fuzzy_matches"] == 0
    assert summary["unmatched"] == 1
    assert summary["hallucination_rate"] == 0.25


def test_catalog_hallucination_rate_does_not_use_fuzzy_by_default() -> None:
    recommendations = pd.DataFrame(
        {
            "recommended_item_text": ["Alfa Creme"],
        }
    )

    summary = catalog_hallucination_rate(recommendations, _catalog())

    assert summary["unmatched"] == 1
    assert summary["fuzzy_matches"] == 0
    assert summary["hallucination_rate"] == 1.0


def test_catalog_fuzzy_matching_is_optional_and_separate() -> None:
    pytest.importorskip("rapidfuzz")
    recommendations = pd.DataFrame(
        {
            "recommended_item_text": ["Alfa Creme"],
        }
    )

    summary = catalog_hallucination_rate(
        recommendations,
        _catalog(),
        enable_fuzzy=True,
        fuzzy_threshold=70,
    )

    assert summary["exact_matches"] == 0
    assert summary["normalized_matches"] == 0
    assert summary["fuzzy_matches"] == 1
    assert summary["unmatched"] == 0
    assert summary["hallucination_rate"] == 0.0
    assert summary["strict_hallucination_rate"] == 1.0


def test_enabling_fuzzy_without_rapidfuzz_fails_explicitly() -> None:
    if importlib.util.find_spec("rapidfuzz") is not None:
        pytest.skip("rapidfuzz is installed")

    recommendations = pd.DataFrame(
        {
            "recommended_item_text": ["Alfa Creme"],
        }
    )

    with pytest.raises(ImportError, match="rapidfuzz is required"):
        catalog_hallucination_rate(
            recommendations,
            _catalog(),
            enable_fuzzy=True,
            fuzzy_threshold=70,
        )


def test_candidate_hallucination_rate_uses_user_specific_candidates() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "recommended_item_id": ["A1", "C3", None],
            "recommended_item_text": ["Alpha Cream", "Gamma Mask", "gamma-mask!"],
            "rank": [1, 2, 1],
        }
    )
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "candidate_item_id": ["A1", "B2", "C3"],
            "candidate_item_title": ["Alpha Cream", "Beta Brush", "Gamma Mask"],
        }
    )

    summary = candidate_hallucination_rate(recommendations, candidates)

    assert summary["total"] == 3
    assert summary["item_id_matches"] == 1
    assert summary["normalized_matches"] == 1
    assert summary["unmatched"] == 1
    assert summary["hallucination_rate"] == pytest.approx(1 / 3)


def test_candidate_hallucination_rate_honors_top_k_per_user() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "recommended_item_id": ["A1", "Z9"],
            "rank": [1, 2],
        }
    )
    candidates = pd.DataFrame(
        {
            "user_id": ["u1"],
            "candidate_item_id": ["A1"],
        }
    )

    summary = candidate_hallucination_rate(recommendations, candidates, k=1)

    assert summary["total"] == 1
    assert summary["item_id_matches"] == 1
    assert summary["unmatched"] == 0
    assert summary["hallucination_rate"] == 0.0


def test_unknown_match_type_raises_clear_error() -> None:
    recommendations = pd.DataFrame({"match_type": ["probably"]})

    with pytest.raises(ValueError, match="unknown match_type"):
        hallucination_rate(recommendations)

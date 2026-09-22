from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from src.evaluation.output_parser import (
    MATCH_EXACT,
    MATCH_FUZZY,
    MATCH_ITEM_ID,
    MATCH_NORMALIZED,
    MATCH_UNMATCHED,
    normalize_recommendation_text,
    parse_llm_output,
    parse_recommendation_items,
)


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["SKU-001", "SKU-002", "SKU-003"],
            "item_title": ["Alpha Cream", "Beta Brush", "Gamma Mask"],
        }
    )


def test_normalize_recommendation_text_removes_rank_punctuation_and_spacing() -> None:
    assert normalize_recommendation_text("  2)  The Alpha-Cream!!! ") == (
        "the alpha cream"
    )


def test_parse_numbered_and_bullet_lists_with_explanatory_text() -> None:
    output = """
    Here are the strongest choices:
    1. "Alpha Cream" because it matches the user history.
    2) Beta Brush
    - Gamma Mask
    Hope this helps.
    """

    assert parse_recommendation_items(output) == [
        "Alpha Cream",
        "Beta Brush",
        "Gamma Mask",
    ]


def test_parse_comma_separated_output_after_header() -> None:
    output = "Recommendations: Alpha Cream, Beta Brush, Gamma Mask."

    assert parse_recommendation_items(output) == [
        "Alpha Cream",
        "Beta Brush",
        "Gamma Mask",
    ]


def test_parse_quoted_titles_from_sentence() -> None:
    output = 'I would recommend "Alpha Cream", "Beta Brush", and "Gamma Mask".'

    assert parse_recommendation_items(output) == [
        "Alpha Cream",
        "Beta Brush",
        "Gamma Mask",
    ]


def test_parse_stringified_list_with_commas_and_apostrophes() -> None:
    output = (
        "['Nautica Blue Eau De Toilette Spray for Men, 3.4 fluid ounce', "
        "'Reb'L Fleur by Rihanna, 3.4 Ounce', "
        "\"L'Oreal Paris EverPure Conditioner, 8.5 Fluid Ounce\"]"
    )

    assert parse_recommendation_items(output) == [
        "Nautica Blue Eau De Toilette Spray for Men, 3.4 fluid ounce",
        "Reb'L Fleur by Rihanna, 3.4 Ounce",
        "L'Oreal Paris EverPure Conditioner, 8.5 Fluid Ounce",
    ]


def test_parse_llm_output_returns_ranked_exact_matches() -> None:
    parsed = parse_llm_output(
        '1. "Alpha Cream"\n2. "Beta Brush"',
        user_id="u1",
        dataset="beauty",
        model_name="gpt-test",
        run_id="run-001",
        catalog=_catalog(),
    )

    assert parsed[["rank", "recommended_item_id", "match_type"]].to_dict(
        "records"
    ) == [
        {"rank": 1, "recommended_item_id": "SKU-001", "match_type": MATCH_EXACT},
        {"rank": 2, "recommended_item_id": "SKU-002", "match_type": MATCH_EXACT},
    ]


def test_normalized_match_is_distinct_from_exact_match() -> None:
    parsed = parse_llm_output(
        "1. alpha cream!",
        user_id="u1",
        dataset="beauty",
        model_name="gpt-test",
        run_id="run-001",
        catalog=_catalog(),
    )

    assert parsed.loc[0, "recommended_item_id"] == "SKU-001"
    assert parsed.loc[0, "match_type"] == MATCH_NORMALIZED


def test_item_ids_in_output_are_used_when_present() -> None:
    parsed = parse_llm_output(
        "1. [SKU-002] Brush option with imperfect title text",
        user_id="u1",
        dataset="beauty",
        model_name="gpt-test",
        run_id="run-001",
        catalog=_catalog(),
    )

    assert parsed.loc[0, "recommended_item_id"] == "SKU-002"
    assert parsed.loc[0, "match_type"] == MATCH_ITEM_ID


def test_unmatched_text_is_not_silently_converted_to_valid_item() -> None:
    parsed = parse_llm_output(
        "1. Unknown Product",
        user_id="u1",
        dataset="beauty",
        model_name="gpt-test",
        run_id="run-001",
        catalog=_catalog(),
    )

    assert pd.isna(parsed.loc[0, "recommended_item_id"])
    assert parsed.loc[0, "recommended_item_text"] == "Unknown Product"
    assert parsed.loc[0, "match_type"] == MATCH_UNMATCHED


def test_no_catalog_keeps_parsed_text_as_unmatched_recommendation() -> None:
    parsed = parse_llm_output(
        "1. Alpha Cream",
        user_id="u1",
        dataset="beauty",
        model_name="gpt-test",
        run_id="run-001",
    )

    assert pd.isna(parsed.loc[0, "recommended_item_id"])
    assert parsed.loc[0, "recommended_item_text"] == "Alpha Cream"
    assert parsed.loc[0, "match_type"] == MATCH_UNMATCHED


def test_empty_output_returns_empty_recommendation_dataframe() -> None:
    parsed = parse_llm_output(
        "   ",
        user_id="u1",
        dataset="beauty",
        model_name="gpt-test",
        run_id="run-001",
        catalog=_catalog(),
    )

    assert parsed.empty
    assert list(parsed.columns) == [
        "user_id",
        "model_name",
        "dataset",
        "rank",
        "recommended_item_id",
        "recommended_item_text",
        "run_id",
        "raw_output",
        "match_type",
        "match_score",
        "catalog_item_title",
    ]


def test_fuzzy_matching_is_optional_and_explicit() -> None:
    pytest.importorskip("rapidfuzz")

    without_fuzzy = parse_llm_output(
        "1. Alfa Creme",
        user_id="u1",
        dataset="beauty",
        model_name="gpt-test",
        run_id="run-001",
        catalog=_catalog(),
        enable_fuzzy=False,
    )
    with_fuzzy = parse_llm_output(
        "1. Alfa Creme",
        user_id="u1",
        dataset="beauty",
        model_name="gpt-test",
        run_id="run-001",
        catalog=_catalog(),
        enable_fuzzy=True,
        fuzzy_threshold=70,
    )

    assert without_fuzzy.loc[0, "match_type"] == MATCH_UNMATCHED
    assert with_fuzzy.loc[0, "recommended_item_id"] == "SKU-001"
    assert with_fuzzy.loc[0, "match_type"] == MATCH_FUZZY


def test_enabling_fuzzy_without_rapidfuzz_fails_explicitly() -> None:
    if importlib.util.find_spec("rapidfuzz") is not None:
        pytest.skip("rapidfuzz is installed")

    with pytest.raises(ImportError, match="rapidfuzz is required"):
        parse_llm_output(
            "1. Alfa Creme",
            user_id="u1",
            dataset="beauty",
            model_name="gpt-test",
            run_id="run-001",
            catalog=_catalog(),
            enable_fuzzy=True,
            fuzzy_threshold=70,
        )

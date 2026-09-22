from __future__ import annotations

import inspect
import json
import math

import pytest

from src.baselines.plan_e_non_neural_baselines import (
    aggregate_metrics,
    decode_candidate_items,
    duplicate_count,
    hr_at_k,
    is_ranking_valid,
    json_list,
    make_ranking_output_row,
    ndcg_at_k,
    positive_rank_zero_based,
    rank_with_scores,
    score_lookup_candidate_row,
    score_mostpop_candidate_row,
    score_random_candidate_row,
    stable_random_score,
    validate_bm25_variant_label,
    compute_train_popularity,
)


def test_random_same_row_and_seed_produce_identical_ranking() -> None:
    items = ["a", "b", "c", "d"]
    first = score_random_candidate_row(items, 42, "Beauty", "ranking", "set1", "user1")
    second = score_random_candidate_row(items, 42, "Beauty", "ranking", "set1", "user1")
    assert first.ranked_items == second.ranked_items
    assert first.ranked_scores == second.ranked_scores


def test_random_different_stable_identity_can_change_permutation() -> None:
    items = ["a", "b", "c", "d", "e", "f"]
    base = score_random_candidate_row(items, 42, "Beauty", "ranking", "set1", "user1").ranked_items
    found_different = False
    for user_index in range(2, 40):
        other = score_random_candidate_row(items, 42, "Beauty", "ranking", "set1", f"user{user_index}").ranked_items
        if other != base:
            found_different = True
            break
    assert found_different


def test_random_uses_sha256_not_builtin_hash() -> None:
    source = inspect.getsource(stable_random_score)
    assert "hashlib.sha256" in source
    assert "hash" not in stable_random_score.__code__.co_names


def test_random_positive_label_does_not_affect_ranking() -> None:
    items = ["a", "b", "c", "d"]
    scored = score_random_candidate_row(items, 42, "Beauty", "ranking", "set1", "user1")
    row_a = make_ranking_output_row(
        run_id="run",
        baseline_id="random_seeded",
        dataset="Beauty",
        task_type="ranking",
        candidate_set_id="set1",
        source_path="source.csv",
        user_id="user1",
        positive_item_id="a",
        candidate_items=items,
        scored=scored,
        eligible_full_cohort=True,
        eligible_matched_cohort=True,
        notes="",
    )
    row_b = make_ranking_output_row(
        run_id="run",
        baseline_id="random_seeded",
        dataset="Beauty",
        task_type="ranking",
        candidate_set_id="set1",
        source_path="source.csv",
        user_id="user1",
        positive_item_id="b",
        candidate_items=items,
        scored=scored,
        eligible_full_cohort=True,
        eligible_matched_cohort=True,
        notes="",
    )
    assert row_a["ranked_items"] == row_b["ranked_items"]
    assert set(json.loads(row_a["ranked_items"])) == set(items)


def test_random_candidate_membership_preserved_and_repeat_ordering() -> None:
    items = ["001", "010", "2"]
    scored = score_random_candidate_row(items, 7, "ML-1M", "reranking", "set2", "u")
    assert set(scored.ranked_items) == set(items)
    assert scored.ranked_items == score_random_candidate_row(items, 7, "ML-1M", "reranking", "set2", "u").ranked_items


def test_mostpop_higher_training_popularity_ranks_first_and_ignores_test() -> None:
    rows = [
        {"item_id": "b", "split": "train"},
        {"item_id": "b", "split": "train"},
        {"item_id": "a", "split": "train"},
        {"item_id": "a", "split": "test"},
        {"item_id": "a", "split": "test"},
    ]
    counts = compute_train_popularity(rows)
    scored = score_mostpop_candidate_row(["a", "b", "c"], counts)
    assert scored.ranked_items[0] == "b"
    assert scored.candidate_scores == [1.0, 2.0, 0.0]


def test_mostpop_missing_popularity_and_ties_are_deterministic() -> None:
    scored = score_mostpop_candidate_row(["z", "a", "m"], {})
    assert scored.candidate_scores == [0.0, 0.0, 0.0]
    assert scored.ranked_items == ["a", "m", "z"]
    assert set(scored.ranked_items) == {"z", "a", "m"}


def test_bm25_variant_label_rules() -> None:
    validate_bm25_variant_label("bm25_like_e3b", "reconstructed_BM25_like")
    with pytest.raises(ValueError):
        validate_bm25_variant_label("bm25_exact_official", "reconstructed_BM25_like")
    with pytest.raises(ValueError):
        validate_bm25_variant_label("bm25_like_e3b", "exact_official")


def test_bm25_lookup_alignment_missing_behavior_and_ties() -> None:
    items = ["c", "a", "b"]
    scored = score_lookup_candidate_row(items, {"a": 1.0, "b": 1.0}, missing_score_value=0.0)
    assert scored.candidate_scores == [0.0, 1.0, 1.0]
    assert scored.missing_score_count == 1
    assert scored.ranked_items == ["a", "b", "c"]
    assert set(scored.ranked_items) == set(items)


@pytest.mark.parametrize(
    ("rank", "expected"),
    [(0, 1.0), (4, 1.0), (5, 0.0), (None, 0.0)],
)
def test_hr_at_5(rank: int | None, expected: float) -> None:
    assert hr_at_k(rank, 5) == expected


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, 1.0),
        (1, 1 / math.log2(3)),
        (4, 1 / math.log2(6)),
        (5, 0.0),
        (None, 0.0),
    ],
)
def test_ndcg_at_5(rank: int | None, expected: float) -> None:
    assert ndcg_at_k(rank, 5) == pytest.approx(expected)


def test_positive_rank_duplicate_and_validation_detection() -> None:
    assert positive_rank_zero_based(["a", "b", "c"], "b") == 1
    assert positive_rank_zero_based(["a", "b", "c"], "x") is None
    assert duplicate_count(["a", "b", "a"]) == 1
    assert is_ranking_valid(["a", "b"], ["a", "b"], [1.0, 0.0], "a")
    assert not is_ranking_valid(["a", "b"], ["a", "c"], [1.0, 0.0], "a")
    assert not is_ranking_valid(["a", "b"], ["a", "b"], [float("inf"), 0.0], "a")


def test_full_and_matched_filtering_use_same_ranking_rows() -> None:
    items = ["a", "b", "c", "d", "e", "f"]
    scored = score_lookup_candidate_row(items, {"a": 6, "b": 5, "c": 4, "d": 3, "e": 2, "f": 1})
    row_matched = make_ranking_output_row(
        run_id="run",
        baseline_id="bm25_like_e3b",
        dataset="Beauty",
        task_type="ranking",
        candidate_set_id="set",
        source_path="source",
        user_id="u1",
        positive_item_id="a",
        candidate_items=items,
        scored=scored,
        eligible_full_cohort=True,
        eligible_matched_cohort=True,
        notes="",
    )
    row_unmatched = row_matched | {"user_id": "u2", "eligible_matched_cohort": "false", "positive_rank_zero_based": "5", "positive_rank_one_based": "6", "positive_in_top_5": "false", "ndcg_at_5": "0"}
    full = aggregate_metrics(
        [row_matched, row_unmatched],
        cohort_id="plan_e2_full",
        run_id="run",
        baseline_id="bm25_like_e3b",
        dataset="Beauty",
        task_type="ranking",
        candidate_set_id="set",
        paper_alignment="adapted",
        notes="",
    )
    matched = aggregate_metrics(
        [row_matched, row_unmatched],
        cohort_id="plan_e2_matched_title_complete",
        run_id="run",
        baseline_id="bm25_like_e3b",
        dataset="Beauty",
        task_type="ranking",
        candidate_set_id="set",
        paper_alignment="adapted",
        notes="",
    )
    assert full["evaluated_rows"] == "2"
    assert matched["evaluated_rows"] == "1"
    assert float(matched["hr_at_5"]) == 1.0


def test_empty_cohort_failure() -> None:
    with pytest.raises(ValueError):
        aggregate_metrics(
            [],
            cohort_id="plan_e2_full",
            run_id="run",
            baseline_id="random_seeded",
            dataset="Beauty",
            task_type="ranking",
            candidate_set_id="set",
            paper_alignment="paper_like_reconstructed",
            notes="",
        )


def test_json_serialization_alignment_and_string_ids() -> None:
    values = ["001", "2", "10"]
    encoded = json_list(values)
    assert decode_candidate_items(encoded) == values
    assert all(isinstance(item, str) for item in decode_candidate_items(encoded))
    ranked, scores = rank_with_scores(values, [0.0, 0.0, 0.0])
    assert ranked == ["001", "10", "2"]
    assert scores == [0.0, 0.0, 0.0]

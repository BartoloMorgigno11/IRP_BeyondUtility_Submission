from __future__ import annotations

from math import log2

import pytest

from src.metrics.canonical_metric_contract import (
    build_long_tail_set,
    build_metric_registry,
    candidate_validity_rate_at_k,
    catalog_coverage_ratio,
    contract_sha256,
    cosine_distance,
    duplicate_rate_at_k,
    exact_row_keys,
    first_positive_rank_zero_based,
    hr_at_k_from_rank,
    intra_list_diversity_jaccard,
    item_coverage_count,
    jaccard_distance,
    long_tail_rate,
    macro_mean,
    ndcg_at_k_from_rank,
    novelty_self_information,
    popularity_bias_difference,
    positive_occurrence_count,
    rank_zero_to_one_based,
    same_candidate_lists,
    same_row_keys,
    self_information_from_probability,
    utility_from_ranks,
    validate_cutoff,
    validate_metric_registry,
    validate_one_positive,
    average_popularity,
)


def test_registry_metric_ids_are_unique_and_hashes_are_stable() -> None:
    registry = build_metric_registry()
    validate_metric_registry(registry)
    metric_ids = [contract.metric_id for contract in registry]

    assert len(metric_ids) == len(set(metric_ids))
    assert contract_sha256(registry[0]) == contract_sha256(registry[0])


def test_registry_ranges_directionality_and_cutoffs_are_valid() -> None:
    registry = build_metric_registry()

    assert {contract.cutoff for contract in registry}.issubset({"5", "all"})
    assert {contract.higher_is_better for contract in registry}.issubset({"true", "false", "not_applicable"})
    assert all(contract.aggregation_level for contract in registry)


def test_blocked_contracts_are_not_marked_ready() -> None:
    blocked = [contract for contract in build_metric_registry() if contract.contract_status.startswith("blocked")]

    assert blocked
    assert all(contract.blocking_reason for contract in blocked)


def test_cutoff_validation() -> None:
    validate_cutoff(5)
    with pytest.raises(ValueError):
        validate_cutoff(0)
    with pytest.raises(ValueError):
        validate_cutoff(True)  # type: ignore[arg-type]


def test_hr_at_5_rank_boundaries() -> None:
    assert hr_at_k_from_rank(0, 5) == 1.0
    assert hr_at_k_from_rank(4, 5) == 1.0
    assert hr_at_k_from_rank(5, 5) == 0.0
    assert hr_at_k_from_rank(None, 5) == 0.0


def test_ndcg_at_5_rank_boundaries_and_one_based_conversion() -> None:
    assert rank_zero_to_one_based(0) == 1
    assert ndcg_at_k_from_rank(0, 5) == 1.0
    assert ndcg_at_k_from_rank(4, 5) == pytest.approx(1 / log2(6))
    assert ndcg_at_k_from_rank(5, 5) == 0.0


def test_empty_cohort_handling() -> None:
    with pytest.raises(ValueError):
        macro_mean([])
    with pytest.raises(ValueError):
        utility_from_ranks([])


def test_positive_rank_present_missing_and_duplicates() -> None:
    ranked = ["a", "p", "b", "p"]

    assert first_positive_rank_zero_based(ranked, "p") == 1
    assert first_positive_rank_zero_based(ranked, "missing") is None
    assert positive_occurrence_count(ranked, "p") == 2
    with pytest.raises(ValueError):
        validate_one_positive(ranked, "p")


def test_one_positive_per_row_validation() -> None:
    validate_one_positive(["a", "p", "b"], "p")
    with pytest.raises(ValueError):
        validate_one_positive(["a", "b"], "p")


def test_utility_from_ranks_macro_mean() -> None:
    values = utility_from_ranks([0, 4, 5], 5)

    assert values["hr_at_5"] == pytest.approx(2 / 3)
    assert values["ndcg_at_5"] == pytest.approx((1 + (1 / log2(6)) + 0) / 3)


def test_novelty_self_information_uses_log_base_two() -> None:
    assert self_information_from_probability(0.25) == 2.0
    assert novelty_self_information(25, 100) == 2.0


def test_zero_popularity_is_rejected_without_smoothing() -> None:
    with pytest.raises(ValueError):
        novelty_self_information(0, 100)


def test_coverage_ratios_use_unique_items_not_mean_per_row() -> None:
    recommended = ["a", "a", "b", "c"]
    universe = ["a", "b", "c", "d"]

    assert catalog_coverage_ratio(recommended, universe) == pytest.approx(0.75)
    assert item_coverage_count(recommended) == 3


def test_candidate_union_and_train_catalog_denominators_are_distinct() -> None:
    recommended = ["a", "b"]

    assert catalog_coverage_ratio(recommended, ["a", "b"]) == 1.0
    assert catalog_coverage_ratio(recommended, ["a", "b", "c", "d"]) == 0.5


def test_long_tail_threshold_and_tie_boundary() -> None:
    tail, threshold = build_long_tail_set({"a": 1, "b": 2, "c": 2}, threshold=2)

    assert threshold == 2
    assert tail == {"a"}
    assert long_tail_rate(["a", "b", "a", "c"], tail) == pytest.approx(0.5)


def test_long_tail_rejects_undefined_threshold_without_counts() -> None:
    with pytest.raises(ValueError):
        build_long_tail_set({})


def test_result_dependent_threshold_policy_is_not_encoded_in_formula() -> None:
    tail, threshold = build_long_tail_set({"a": 1, "b": 10}, quantile=0.8)

    assert threshold == pytest.approx(8.2)
    assert tail == {"a"}


def test_average_popularity_and_popularity_bias_are_distinct() -> None:
    counts = {"a": 10, "b": 2, "c": 1}

    assert average_popularity(["a", "b"], counts) == 6.0
    assert popularity_bias_difference(["a", "b"], ["c"], counts) == 5.0


def test_zero_history_popularity_bias_rejected() -> None:
    with pytest.raises(ValueError):
        popularity_bias_difference(["a"], [], {"a": 1})


def test_ild_pair_count_for_five_items_jaccard_example() -> None:
    items = ["a", "b", "c", "d", "e"]
    features = {
        "a": {"x"},
        "b": {"x", "y"},
        "c": {"z"},
        "d": {"z", "q"},
        "e": {"r"},
    }

    score = intra_list_diversity_jaccard(items, features)

    assert 0.0 <= score <= 1.0


def test_jaccard_duplicate_missing_and_empty_features() -> None:
    features = {"a": {"x"}, "b": {"x"}}

    assert intra_list_diversity_jaccard(["a", "a"], features) == 0.0
    assert jaccard_distance({"x"}, {"x", "y"}) == 0.5
    with pytest.raises(ValueError):
        intra_list_diversity_jaccard(["a", "missing"], features)
    with pytest.raises(ValueError):
        jaccard_distance(set(), set())


def test_cosine_distance_zero_vector_rejected() -> None:
    assert cosine_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cosine_distance([0, 0], [1, 0])


def test_duplicate_rate_fixed_top5_denominator() -> None:
    assert duplicate_rate_at_k(["a", "b", "c", "d", "e"], 5) == 0.0
    assert duplicate_rate_at_k(["a", "b", "a", "d", "e"], 5) == 0.2
    assert duplicate_rate_at_k(["a", "b"], 5) == 0.6


def test_candidate_validity_rate_fixed_top5_denominator() -> None:
    candidates = ["a", "b", "c", "d", "e"]

    assert candidate_validity_rate_at_k(["a", "b", "c", "d", "e"], candidates, k=5) == 1.0
    assert candidate_validity_rate_at_k(["a", "x", "c"], candidates, k=5) == 0.4


def test_same_cohort_row_keys_and_candidate_lists() -> None:
    left = [{"dataset": "d", "task": "ranking", "user": "u1"}]
    right = [{"dataset": "d", "task": "ranking", "user": "u1"}]
    other = [{"dataset": "d", "task": "reranking", "user": "u1"}]
    columns = ["dataset", "task", "user"]

    assert exact_row_keys(left, columns) == {("d", "ranking", "u1")}
    assert same_row_keys(left, right, columns)
    assert not same_row_keys(left, other, columns)
    assert same_candidate_lists({("u1",): ["a", "b"]}, {("u1",): ["a", "b"]})
    assert not same_candidate_lists({("u1",): ["a", "b"]}, {("u1",): ["b", "a"]})


def test_dataset_task_cohort_separation_in_row_keys() -> None:
    ranking = [{"dataset": "d", "task": "ranking", "cohort": "full", "user": "u"}]
    matched = [{"dataset": "d", "task": "ranking", "cohort": "matched", "user": "u"}]

    assert not same_row_keys(ranking, matched, ["dataset", "task", "cohort", "user"])

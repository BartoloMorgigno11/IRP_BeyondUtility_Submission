import json
import math

import pandas as pd
import pytest

from src.metrics.canonical_metric_computation import (
    TOP_K,
    bool_text,
    candidate_validity_rate,
    contract_payload_hash,
    coverage_ratio,
    descending_popularity_rank_items,
    duplicate_flags,
    duplicate_rate,
    finite_scores,
    format_float,
    hr_from_rank,
    json_dumps,
    long_tail_member,
    mean_float,
    median_float,
    ndcg_from_rank,
    parse_float_array,
    parse_json_array,
    popularity_quantiles,
    quantile_threshold,
    require_known_metric,
    require_no_blocked_substitution,
    row_metric_contributions,
    self_information_bits,
    sha256_items,
    train_item_statistics,
)


def test_json_array_parsing_preserves_string_item_ids_and_leading_zeros():
    assert parse_json_array('["001", 2, "A03"]') == ["001", "2", "A03"]


def test_parse_float_array_and_finite_score_validation():
    scores = parse_float_array("[1, 2.5, -3]")
    assert scores == [1.0, 2.5, -3.0]
    assert finite_scores(scores)
    assert not finite_scores([1.0, math.inf])


def test_utility_metrics_from_positive_rank():
    assert hr_from_rank(0) == 1.0
    assert hr_from_rank(4) == 1.0
    assert hr_from_rank(5) == 0.0
    assert ndcg_from_rank(0) == 1.0
    assert ndcg_from_rank(4) == pytest.approx(1.0 / math.log2(6.0))
    assert ndcg_from_rank(5) == 0.0


def test_self_information_uses_unsmoothed_train_probability():
    assert self_information_bits(2, 8) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="zero train count"):
        self_information_bits(0, 8)


def test_train_item_statistics_uses_only_train_split():
    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "001", "split": "train"},
            {"user_id": "u1", "item_id": "001", "split": "train"},
            {"user_id": "u2", "item_id": "002", "split": "train"},
            {"user_id": "u3", "item_id": "001", "split": "test"},
        ]
    )

    item_counts, unique_users, total = train_item_statistics(interactions)

    assert item_counts == {"001": 2, "002": 1}
    assert unique_users == {"001": 1, "002": 1}
    assert total == 3


def test_popularity_rank_quantile_and_threshold_are_deterministic():
    counts = {"b": 5, "a": 5, "c": 1, "d": 2}

    assert descending_popularity_rank_items(counts) == {"a": 1, "b": 2, "d": 3, "c": 4}
    assert popularity_quantiles(counts)["c"] == pytest.approx(0.25)
    assert quantile_threshold(counts, 0.5) == pytest.approx(3.5)
    assert long_tail_member(2, 3.5)
    assert not long_tail_member(5, 3.5)


def test_duplicate_flags_and_rate_are_structural():
    assert duplicate_flags(["a", "b", "a", "c", "b"]) == [False, False, True, False, True]
    assert duplicate_rate(["a", "b", "a", "c", "b"]) == pytest.approx(2 / TOP_K)
    with pytest.raises(ValueError, match="expected 5"):
        duplicate_rate(["a", "b"])


def test_candidate_validity_requires_candidate_and_train_membership():
    top_items = ["a", "b", "c", "d", "e"]
    candidates = ["a", "b", "c", "d", "x"]
    train_seen = ["a", "b", "c", "z"]

    assert candidate_validity_rate(top_items, candidates, train_seen) == pytest.approx(3 / TOP_K)


def test_row_metric_contributions_include_top5_evidence():
    contributions = row_metric_contributions(
        positive_rank_zero_based=2,
        top_items=["a", "b", "c", "d", "e"],
        candidate_items=["a", "b", "c", "d", "e", "f"],
        train_counts={"a": 10, "b": 5, "c": 4, "d": 2, "e": 1},
        unique_user_counts={"a": 9, "b": 5, "c": 4, "d": 2, "e": 1},
        total_train_interactions=22,
        long_tail_threshold=4.1,
    )

    assert contributions["hr_at_5"] == 1.0
    assert contributions["ndcg_at_5"] == pytest.approx(1.0 / math.log2(4.0))
    assert contributions["long_tail_rate_at_5"] == pytest.approx(3 / TOP_K)
    assert contributions["average_popularity_at_5"] == pytest.approx(22 / TOP_K)
    assert contributions["duplicate_rate_at_5"] == 0.0
    assert contributions["candidate_validity_rate"] == 1.0
    assert contributions["top5_train_popularity_counts"] == [10, 5, 4, 2, 1]
    assert contributions["top5_long_tail_membership"] == [False, False, True, True, True]


def test_row_metric_contributions_reject_unseen_top_items_without_smoothing():
    with pytest.raises(ValueError, match="positive train popularity"):
        row_metric_contributions(
            positive_rank_zero_based=0,
            top_items=["a", "b", "c", "d", "e"],
            candidate_items=["a", "b", "c", "d", "e"],
            train_counts={"a": 1, "b": 1, "c": 1, "d": 1},
            unique_user_counts={"a": 1, "b": 1, "c": 1, "d": 1},
            total_train_interactions=4,
            long_tail_threshold=2.0,
        )


def test_coverage_ratio_counts_valid_unique_items_only():
    numerator, denominator, value = coverage_ratio(["a", "a", "x"], ["a", "b"])

    assert numerator == 1
    assert denominator == 2
    assert value == pytest.approx(0.5)


def test_aggregation_helpers_reject_empty_inputs():
    assert mean_float(["1", 2]) == pytest.approx(1.5)
    assert median_float(["1", 5, 2]) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="empty"):
        mean_float([])
    with pytest.raises(ValueError, match="empty"):
        median_float([])


def test_serialization_helpers_are_stable():
    assert json_dumps({"b": 2, "a": ["001"]}) == '{"a":["001"],"b":2}'
    assert json.loads(json_dumps({"title": "A"})) == {"title": "A"}
    assert sha256_items(["b", "a", "a"]) == sha256_items(["a", "b", "a"])
    assert format_float(1.0) == "1"
    assert bool_text(True) == "true"
    assert bool_text(False) == "false"


def test_contract_payload_hash_ignores_existing_hash_column():
    row = {"metric_id": "hr_at_5", "contract_sha256": "old", "cutoff": 5}
    changed_hash = {"metric_id": "hr_at_5", "contract_sha256": "new", "cutoff": 5}

    assert contract_payload_hash(row) == contract_payload_hash(changed_hash)


def test_blocked_metric_guardrails_prevent_proxy_computation():
    require_known_metric("hr_at_5")
    require_known_metric("popularity_bias_at_5")
    require_no_blocked_substitution("hr_at_5")
    with pytest.raises(ValueError, match="blocked metric cannot be computed"):
        require_no_blocked_substitution("popularity_bias_at_5")
    with pytest.raises(ValueError, match="unknown canonical metric"):
        require_known_metric("unsupported_metric")

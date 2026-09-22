from __future__ import annotations

import pandas as pd

from src.metrics.canonical_local_llm_beyond_utility import (
    aggregate_rows,
    build_train_context,
    prompt_candidate_union,
)


def _train_items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "Amazon Beauty",
                "item_id": "i1",
                "train_interaction_count": "1",
                "self_information_bits": "4",
                "long_tail_threshold": "3",
                "long_tail_member": "true",
            },
            {
                "dataset": "Amazon Beauty",
                "item_id": "i2",
                "train_interaction_count": "3",
                "self_information_bits": "2",
                "long_tail_threshold": "3",
                "long_tail_member": "false",
            },
            {
                "dataset": "Amazon Beauty",
                "item_id": "i3",
                "train_interaction_count": "5",
                "self_information_bits": "1",
                "long_tail_threshold": "3",
                "long_tail_member": "false",
            },
            {
                "dataset": "Amazon Beauty",
                "item_id": "i4",
                "train_interaction_count": "2",
                "self_information_bits": "3",
                "long_tail_threshold": "3",
                "long_tail_member": "true",
            },
            {
                "dataset": "Amazon Beauty",
                "item_id": "i5",
                "train_interaction_count": "4",
                "self_information_bits": "1.5",
                "long_tail_threshold": "3",
                "long_tail_member": "false",
            },
        ]
    )


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "Amazon Beauty",
                "task_type": "ranking",
                "candidate_set_id": "cand",
                "user_id": "u1",
                "candidate_items": '["i1","i2","i3","i4","i5","i6"]',
                "top5_item_ids": '["i1","i2","i3","i4","i5"]',
            }
        ]
    )


def test_novelty_average_popularity_and_long_tail_use_train_counts_without_smoothing():
    train = build_train_context(_train_items())
    unions = {("Amazon Beauty", "ranking"): {"i1", "i2", "i3", "i4", "i5", "i6"}}
    metrics = aggregate_rows(
        _rows(),
        dataset="Amazon Beauty",
        task_type="ranking",
        planned_rows=1,
        strict_valid_rows=1,
        train=train,
        candidate_unions=unions,
    )
    assert metrics["novelty_at_5"]["metric_value"] == "2.2999999999999998"
    assert metrics["average_popularity_at_5"]["metric_value"] == "3"
    assert metrics["long_tail_rate_at_5"]["metric_value"] == "0.40000000000000002"


def test_catalog_and_item_coverage_use_distinct_denominators():
    train = build_train_context(_train_items())
    unions = {("Amazon Beauty", "ranking"): {"i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10"}}
    metrics = aggregate_rows(
        _rows(),
        dataset="Amazon Beauty",
        task_type="ranking",
        planned_rows=1,
        strict_valid_rows=1,
        train=train,
        candidate_unions=unions,
    )
    assert metrics["catalog_coverage_at_5"]["metric_numerator"] == "5"
    assert metrics["catalog_coverage_at_5"]["metric_denominator"] == "5"
    assert metrics["item_coverage_at_5"]["metric_numerator"] == "5"
    assert metrics["item_coverage_at_5"]["metric_denominator"] == "10"


def test_missing_popularity_reduces_popularity_metric_coverage_not_candidate_union():
    train = build_train_context(_train_items())
    rows = _rows()
    rows.loc[0, "top5_item_ids"] = '["i1","i2","i3","i4","missing"]'
    unions = {("Amazon Beauty", "ranking"): {"i1", "i2", "i3", "i4", "missing"}}
    metrics = aggregate_rows(
        rows,
        dataset="Amazon Beauty",
        task_type="ranking",
        planned_rows=1,
        strict_valid_rows=1,
        train=train,
        candidate_unions=unions,
    )
    assert metrics["novelty_at_5"]["metric_status"] == "blocked_missing_popularity_support"
    assert metrics["item_coverage_at_5"]["metric_value"] == "1"


def test_prompt_candidate_union_deduplicates_prompt_rows_not_model_rows():
    rankings = pd.concat([_rows(), _rows()], ignore_index=True)
    rankings["model_family"] = ["Qwen2 7B Instruct", "Mistral 7B Instruct"]
    union = prompt_candidate_union(rankings)
    assert union[("Amazon Beauty", "ranking")] == {"i1", "i2", "i3", "i4", "i5", "i6"}

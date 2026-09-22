from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.canonical_local_llm_baseline_alignment import (
    build_baseline_cohort_alignment,
    paired_bootstrap_delta_ci,
)


def _llm_rows(candidate_items: str = '["i1","i2","i3","i4","i5"]') -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_family": "Qwen2 7B Instruct",
                "dataset": "Amazon Beauty",
                "task_type": "ranking",
                "candidate_set_id": "cand",
                "user_id": "u1",
                "positive_item_id": "i1",
                "candidate_items": candidate_items,
                "strict_parse_valid": "true",
                "positive_in_top_5": "true",
                "ndcg_at_5": "1",
                "top5_item_ids": '["i1","i2","i3","i4","i5"]',
            }
        ]
    )


def _baseline_rows(candidate_items: str = '["i1","i2","i3","i4","i5"]') -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "baseline_id": "random_seeded",
                "dataset": "Amazon Beauty",
                "task_type": "ranking",
                "candidate_set_id": "cand",
                "user_id": "u1",
                "positive_item_id": "i1",
                "candidate_items": candidate_items,
                "ranked_items": '["i2","i1","i3","i4","i5"]',
                "positive_in_top_5": "true",
                "ndcg_at_5": "0.6309297535714575",
            }
        ]
    )


def test_baseline_alignment_detects_exact_same_cohort():
    alignment = build_baseline_cohort_alignment(_llm_rows(), _baseline_rows())
    row = alignment[
        alignment["baseline_id"].eq("random_seeded")
        & alignment["dataset"].eq("Amazon Beauty")
        & alignment["task_type"].eq("ranking")
    ].iloc[0]
    assert row["exact_same_cohort"] == "true"
    assert row["alignment_status"] == "exact_same_prompt_compatible_cohort"


def test_baseline_alignment_detects_candidate_mismatch():
    alignment = build_baseline_cohort_alignment(_llm_rows(), _baseline_rows('["i1","i3","i2","i4","i5"]'))
    row = alignment[
        alignment["baseline_id"].eq("random_seeded")
        & alignment["dataset"].eq("Amazon Beauty")
        & alignment["task_type"].eq("ranking")
    ].iloc[0]
    assert row["alignment_status"] == "candidate_mismatch"
    assert row["candidate_order_match_rows"] == "0"


def test_baseline_alignment_detects_partial_overlap():
    llm = pd.concat(
        [
            _llm_rows(),
            pd.DataFrame(
                [
                    {
                        "model_family": "Qwen2 7B Instruct",
                        "dataset": "Amazon Beauty",
                        "task_type": "ranking",
                        "candidate_set_id": "cand",
                        "user_id": "u2",
                        "positive_item_id": "i2",
                        "candidate_items": '["i1","i2","i3","i4","i5"]',
                        "strict_parse_valid": "true",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    alignment = build_baseline_cohort_alignment(llm, _baseline_rows())
    row = alignment[
        alignment["baseline_id"].eq("random_seeded")
        & alignment["dataset"].eq("Amazon Beauty")
        & alignment["task_type"].eq("ranking")
    ].iloc[0]
    assert row["alignment_status"] == "partial_baseline_coverage"
    assert row["missing_baseline_rows"] == "1"


def test_paired_bootstrap_is_deterministic_and_requires_pairs():
    low1, high1 = paired_bootstrap_delta_ci([1, 0, 1], [0, 0, 1], seed=42, resamples=100)
    low2, high2 = paired_bootstrap_delta_ci([1, 0, 1], [0, 0, 1], seed=42, resamples=100)
    assert (low1, high1) == (low2, high2)
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap_delta_ci([1, 0], [1], resamples=10)

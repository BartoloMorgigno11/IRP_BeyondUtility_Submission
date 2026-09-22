from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis.plan_e_gru4rec_baseline_consolidation import (
    BPRMF_BASELINE_ID,
    FULL_COHORT_ID,
    GRU4REC_BASELINE_ID,
    MATCHED_COHORT_ID,
    build_paired_comparisons,
    build_same_cohort_metrics,
    cohort_keys_from_gru4rec_rankings,
    filter_rankings_to_cohort,
    metrics_from_rankings,
    paired_positive_rank_summary,
    parse_json_array,
    require_exact_row_key_equality,
    validate_comparison_frame,
    validate_metric_ranges,
    verify_candidate_list_equality,
)


def test_exact_row_key_filtering_and_no_source_mutation() -> None:
    frame = _rankings("random_seeded", users=("u1", "u2", "u3"))
    original = frame.copy(deep=True)
    filtered = filter_rankings_to_cohort(frame, {("D1", "ranking", "cs_rank", "u2")})
    assert filtered["user_id"].tolist() == ["u2"]
    pd.testing.assert_frame_equal(frame, original)


def test_exact_gru_bprmf_row_equality_and_failures() -> None:
    gru = _gru_rankings()
    bprmf = _rankings(BPRMF_BASELINE_ID, users=("u1", "u2"))
    require_exact_row_key_equality(gru, bprmf)
    with pytest.raises(ValueError, match="row-key sets differ"):
        require_exact_row_key_equality(gru, _rankings(BPRMF_BASELINE_ID, users=("u1", "u3")))


def test_full_and_matched_cohort_filtering() -> None:
    gru = _gru_rankings()
    full_keys = cohort_keys_from_gru4rec_rankings(gru, FULL_COHORT_ID)
    matched_keys = cohort_keys_from_gru4rec_rankings(gru, MATCHED_COHORT_ID)
    assert len(full_keys) == 2
    assert matched_keys == [("D1", "ranking", "cs_rank", "u1")]


def test_missing_extra_and_candidate_list_mismatch_fail() -> None:
    frame = _rankings("random_seeded", users=("u1",))
    with pytest.raises(ValueError, match="missing"):
        filter_rankings_to_cohort(frame, {("D1", "ranking", "cs_rank", "u1"), ("D1", "ranking", "cs_rank", "u2")})
    with pytest.raises(ValueError, match="extra"):
        filter_rankings_to_cohort(
            _rankings("random_seeded", users=("u1", "u2")),
            {("D1", "ranking", "cs_rank", "u1")},
            require_no_extras=True,
        )
    left = _rankings("a", users=("u1",))
    right = _rankings("b", users=("u1",))
    right.loc[0, "candidate_items"] = json.dumps(["i9", "i2", "i3"])
    with pytest.raises(ValueError, match="candidate-list mismatch"):
        verify_candidate_list_equality(left, right)


def test_metrics_recomputed_from_filtered_rows_and_ranges_validate() -> None:
    rows = _rankings("random_seeded", users=("u1", "u2"))
    metrics = metrics_from_rankings(
        rows,
        baseline_id="random_seeded",
        cohort_id=FULL_COHORT_ID,
        ranking_source_path="rankings.csv",
        cohort_membership_source_path="cohort.csv",
        paper_alignment="adapted",
        notes="synthetic",
    )
    assert metrics["evaluated_rows"] == 2
    assert metrics["hr_at_5"] == 1.0
    assert metrics["ndcg_at_5"] == pytest.approx((1.0 + 0.5) / 2.0)
    validate_metric_ranges(pd.DataFrame([metrics]))


def test_paired_positive_rank_counts_and_win_rates() -> None:
    gru = _rankings(GRU4REC_BASELINE_ID, users=("u1", "u2", "u3"), ranks=(0, 2, 1))
    comp = _rankings("random_seeded", users=("u1", "u2", "u3"), ranks=(1, 2, 0))
    paired = paired_positive_rank_summary(gru, comp)
    assert paired["gru4rec_better_positive_rank_rows"] == 1
    assert paired["equal_positive_rank_rows"] == 1
    assert paired["comparator_better_positive_rank_rows"] == 1
    assert paired["gru4rec_positive_rank_win_rate"] == pytest.approx(1 / 3)


def test_build_same_cohort_metrics_preserves_dataset_task_and_cohort_separation() -> None:
    gru = pd.concat(
        [
            _gru_rankings(dataset="D1", task="ranking", candidate_set="cs_rank"),
            _gru_rankings(dataset="D1", task="reranking", candidate_set="cs_rerank"),
            _gru_rankings(dataset="D2", task="ranking", candidate_set="cs2_rank"),
        ],
        ignore_index=True,
    )
    bprmf = pd.concat(
        [
            _rankings(BPRMF_BASELINE_ID, dataset="D1", task="ranking", candidate_set="cs_rank"),
            _rankings(BPRMF_BASELINE_ID, dataset="D1", task="reranking", candidate_set="cs_rerank"),
            _rankings(BPRMF_BASELINE_ID, dataset="D2", task="ranking", candidate_set="cs2_rank"),
        ],
        ignore_index=True,
    )
    trad_frames = []
    for baseline in ("random_seeded", "mostpop_train_only", "bm25_like_e3b"):
        trad_frames.extend(
            [
                _rankings(baseline, dataset="D1", task="ranking", candidate_set="cs_rank"),
                _rankings(baseline, dataset="D1", task="reranking", candidate_set="cs_rerank"),
                _rankings(baseline, dataset="D2", task="ranking", candidate_set="cs2_rank"),
            ]
        )
    metrics, filtered = build_same_cohort_metrics(
        traditional_rankings=pd.concat(trad_frames, ignore_index=True),
        bprmf_rankings=bprmf,
        gru4rec_rankings=gru,
        cohort_source_path_by_dataset={"D1": "cohort1.csv", "D2": "cohort2.csv"},
    )
    assert set(metrics["dataset"]) == {"D1", "D2"}
    assert set(metrics["task_type"]) == {"ranking", "reranking"}
    assert set(metrics["cohort_id"]) == {FULL_COHORT_ID, MATCHED_COHORT_ID}
    assert (GRU4REC_BASELINE_ID, "D1", "ranking", "cs_rank", FULL_COHORT_ID) in filtered


def test_build_paired_comparisons_is_deterministic_and_valid() -> None:
    gru = _gru_rankings()
    bprmf = _rankings(BPRMF_BASELINE_ID)
    trad = pd.concat(
        [_rankings(baseline) for baseline in ("random_seeded", "mostpop_train_only", "bm25_like_e3b")],
        ignore_index=True,
    )
    metrics, filtered = build_same_cohort_metrics(
        traditional_rankings=trad,
        bprmf_rankings=bprmf,
        gru4rec_rankings=gru,
        cohort_source_path_by_dataset={"D1": "cohort.csv"},
    )
    first = build_paired_comparisons(metrics, filtered)
    second = build_paired_comparisons(metrics.sample(frac=1.0, random_state=1).reset_index(drop=True), filtered)
    pd.testing.assert_frame_equal(first, second)
    validate_comparison_frame(first)
    assert len(first) == 8


def test_json_candidate_parsing_and_empty_cohort_failure() -> None:
    assert parse_json_array('["001", 2]') == ["001", "2"]
    with pytest.raises(ValueError):
        parse_json_array('{"not": "array"}')
    with pytest.raises(ValueError, match="empty cohort"):
        filter_rankings_to_cohort(_rankings("random_seeded"), set())


def _gru_rankings(dataset: str = "D1", task: str = "ranking", candidate_set: str = "cs_rank") -> pd.DataFrame:
    frame = _rankings(GRU4REC_BASELINE_ID, dataset=dataset, task=task, candidate_set=candidate_set, users=("u1", "u2"))
    frame["eligible_gru4rec_strict_full"] = ["true", "true"]
    frame["eligible_gru4rec_strict_matched"] = ["true", "false"]
    frame["ranking_source_path"] = "gru.csv"
    return frame


def _rankings(
    baseline: str,
    *,
    dataset: str = "D1",
    task: str = "ranking",
    candidate_set: str = "cs_rank",
    users: tuple[str, ...] = ("u1", "u2"),
    ranks: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    ranks = ranks or tuple(range(len(users)))
    rows = []
    for user, rank in zip(users, ranks):
        candidates = ["i1", "i2", "i3"]
        ranked = [candidates[rank], *[item for item in candidates if item != candidates[rank]]]
        rows.append(
            {
                "baseline_id": baseline,
                "dataset": dataset,
                "task_type": task,
                "candidate_set_id": candidate_set,
                "source_path": "source.csv",
                "user_id": user,
                "positive_item_id": candidates[rank],
                "candidate_items": json.dumps(candidates),
                "candidate_scores": json.dumps([0.3, 0.2, 0.1]),
                "ranked_items": json.dumps(ranked),
                "ranked_scores": json.dumps([1.0, 0.5, 0.25]),
                "positive_rank_zero_based": rank,
                "positive_rank_one_based": rank + 1,
                "positive_in_top_5": "true",
                "ndcg_at_5": 1.0 / (rank + 1),
                "ranking_valid": "true",
                "duplicate_ranked_item_count": 0,
                "missing_score_count": 0,
                "ranking_source_path": f"{baseline}.csv",
            }
        )
    return pd.DataFrame(rows)

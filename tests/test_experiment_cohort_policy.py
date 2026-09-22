from __future__ import annotations

import json

import pytest

from src.analysis.experiment_cohort_policy import (
    aggregate_candidate_item_usage,
    calculate_membership,
    candidate_only_title_map_rows,
    classify_task_type,
    cohort_summary_rows,
    exclusion_reasons,
    membership_row,
    sort_membership_rows,
    title_exclusion_rows,
)


DEFAULT_CANDIDATES = ["p", *[f"n{i}" for i in range(19)]]


def test_valid_plan_e2_twenty_candidate_full_cohort_eligibility() -> None:
    row = _candidate_row()
    membership = calculate_membership(row, _title_map())

    assert membership["eligible_plan_e2_paper_like_full"] is True


def test_plan_e2_all_titles_enters_strict_and_matched() -> None:
    row = _candidate_row()
    membership = calculate_membership(row, _title_map())

    assert membership["eligible_plan_e2_llm_strict_title"] is True
    assert membership["eligible_plan_e2_matched_comparison"] is True


def test_plan_e2_missing_negative_title_stays_full_but_exits_strict_and_matched() -> None:
    row = _candidate_row()
    membership = calculate_membership(row, _title_map(DEFAULT_CANDIDATES[:-1]))

    assert membership["eligible_plan_e2_paper_like_full"] is True
    assert membership["eligible_plan_e2_llm_strict_title"] is False
    assert membership["eligible_plan_e2_matched_comparison"] is False
    assert membership["missing_title_item_ids"] == ["n18"]


def test_plan_e2_missing_positive_title_exits_strict_and_matched() -> None:
    row = _candidate_row()
    membership = calculate_membership(row, _title_map(DEFAULT_CANDIDATES[1:]))

    assert membership["positive_title_available"] is False
    assert membership["eligible_plan_e2_paper_like_full"] is True
    assert membership["eligible_plan_e2_llm_strict_title"] is False
    assert membership["eligible_plan_e2_matched_comparison"] is False


def test_matched_membership_equals_strict_membership() -> None:
    rows = [
        membership_row(_candidate_row(user_id="u1"), _title_map(["p", "a", "b"])),
        membership_row(_candidate_row(user_id="u2"), _title_map(["p", "a"])),
    ]

    assert [row["eligible_plan_e2_llm_strict_title"] for row in rows] == [
        row["eligible_plan_e2_matched_comparison"] for row in rows
    ]


def test_plan_d_row_remains_diagnostic_only() -> None:
    row = _candidate_row(
        candidate_set_id="plan_d_qwen2_ml1m_ranking_diagnostic_openp5_base",
        source_plan="Plan D",
        source_path="results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_manifest.csv",
    )
    membership = calculate_membership(row, _title_map())

    assert membership["eligible_plan_d_diagnostic_20_candidate"] is True
    assert membership["eligible_plan_e2_paper_like_full"] is False


def test_plan_c_one_hundred_candidate_row_remains_legacy_only() -> None:
    row = _candidate_row(
        candidate_set_id="plan_c_ml1m_reconstructed_100_candidate_pool",
        source_plan="Plan C",
        source_path="data/reconstructed/plan_c/ml1m/candidates.csv",
        candidate_items=["p", *[f"n{i}" for i in range(99)]],
    )
    title_map = _title_map(["p", *[f"n{i}" for i in range(99)]])
    membership = calculate_membership(row, title_map)

    assert membership["eligible_legacy_reconstructed_100_candidate"] is True
    assert membership["eligible_plan_e2_paper_like_full"] is False


def test_twenty_candidates_alone_do_not_infer_plan_e2_membership() -> None:
    row = _candidate_row(source_plan="Plan C")
    membership = calculate_membership(row, _title_map())

    assert membership["eligible_plan_e2_paper_like_full"] is False


def test_invalid_structural_row_enters_no_cohort() -> None:
    row = _candidate_row(row_validation_status="invalid")
    membership = calculate_membership(row, _title_map())

    assert not membership["eligible_plan_e2_paper_like_full"]
    assert not membership["eligible_plan_d_diagnostic_20_candidate"]
    assert not membership["eligible_legacy_reconstructed_100_candidate"]


def test_task_type_ranking_classification() -> None:
    assert classify_task_type("plan_e_beauty_1000user_ranking_manifest", "", "Plan E2") == "ranking"


def test_task_type_reranking_classification() -> None:
    assert classify_task_type("plan_e_beauty_1000user_reranking_manifest", "", "Plan E2") == "reranking"


def test_unknown_task_type() -> None:
    assert classify_task_type("unlabeled_manifest", "unknown.csv", "Plan E2") == "unknown"


def test_missing_title_json_serialization() -> None:
    row = membership_row(_candidate_row(), _title_map(["p"]))

    assert json.loads(row["missing_title_item_ids"]) == DEFAULT_CANDIDATES[1:]


def test_deterministic_exclusion_reasons() -> None:
    row = _candidate_row()
    membership = calculate_membership(row, _title_map(DEFAULT_CANDIDATES[1:]))

    assert exclusion_reasons(row, membership) == ["missing_positive_title", "missing_candidate_titles"]


def test_deterministic_output_ordering() -> None:
    rows = [
        membership_row(_candidate_row(user_id="10"), _title_map()),
        membership_row(_candidate_row(user_id="2"), _title_map()),
    ]

    assert [row["user_id"] for row in sort_membership_rows(rows)] == ["2", "10"]


def test_candidate_only_title_map_projection() -> None:
    rows = [
        _candidate_row(candidate_items=["p", "a", "b"]),
        _candidate_row(user_id="u2", candidate_items=["p", "c", "d"]),
    ]
    usage = aggregate_candidate_item_usage(rows)
    projected = candidate_only_title_map_rows(usage, _title_map(["p", "a", "b", "c", "d"]))

    assert len(projected) == 5
    assert {row["item_id"] for row in projected} == {"p", "a", "b", "c", "d"}
    assert all(row["llm_usable"] == "true" for row in projected)


def test_candidate_and_positive_occurrence_aggregation() -> None:
    rows = [
        _candidate_row(candidate_items=["p", "a", "b"]),
        _candidate_row(user_id="u2", candidate_items=["p", "c", "d"]),
    ]
    usage = aggregate_candidate_item_usage(rows)

    assert usage[("synthetic", "p")]["candidate_occurrence_count"] == 2
    assert usage[("synthetic", "p")]["positive_occurrence_count"] == 2


def test_cohort_summary_counts_and_position_distribution() -> None:
    rows = [
        membership_row(
            _candidate_row(user_id="u1", computed_positive_position_zero_based="0"),
            _title_map(),
        ),
        membership_row(
            _candidate_row(user_id="u2", computed_positive_position_zero_based="2"),
            _title_map(DEFAULT_CANDIDATES[:-1]),
        ),
    ]

    summary = cohort_summary_rows(rows)
    full = _summary_row(summary, "plan_e2_paper_like_full")
    strict = _summary_row(summary, "plan_e2_llm_strict_title")

    assert full["eligible_rows"] == 2
    assert strict["eligible_rows"] == 1
    assert json.loads(full["positive_position_distribution"]) == {"0": 1, "2": 1}


def test_eligibility_rate_calculation() -> None:
    rows = [
        membership_row(_candidate_row(user_id="u1"), _title_map()),
        membership_row(_candidate_row(user_id="u2"), _title_map(DEFAULT_CANDIDATES[:-1])),
    ]

    strict = _summary_row(cohort_summary_rows(rows), "plan_e2_llm_strict_title")

    assert strict["eligibility_rate"] == 0.5


def test_title_exclusion_rows_include_only_title_policy_exclusions() -> None:
    rows = [
        membership_row(_candidate_row(user_id="u1"), _title_map()),
        membership_row(_candidate_row(user_id="u2"), _title_map(DEFAULT_CANDIDATES[:-1])),
    ]

    exclusions = title_exclusion_rows(rows)

    assert len(exclusions) == 1
    assert exclusions[0]["exclusion_id"] == "LTE-000001"
    assert exclusions[0]["policy_action"] == "exclude_from_llm_and_matched_cohorts_keep_in_full_baseline_cohort"


def test_empty_title_map_failure() -> None:
    usage = aggregate_candidate_item_usage([_candidate_row()])

    with pytest.raises(ValueError):
        candidate_only_title_map_rows(usage, {})


def test_missing_candidate_to_title_referential_integrity_failure() -> None:
    usage = aggregate_candidate_item_usage([_candidate_row()])

    with pytest.raises(ValueError):
        candidate_only_title_map_rows(usage, _title_map(["p", "a"]))


def _candidate_row(
    *,
    candidate_set_id: str = "plan_e_beauty_1000user_ranking_manifest",
    source_plan: str = "Plan E2",
    source_path: str = "results/tables/plan_e_beauty_1000user_ranking_manifest.csv",
    dataset: str = "synthetic",
    user_id: str = "u1",
    positive_item_id: str = "p",
    candidate_items: list[str] | None = None,
    stored_positive_position: str = "1",
    computed_positive_position_zero_based: str = "0",
    row_validation_status: str = "valid",
) -> dict[str, str]:
    items = candidate_items or DEFAULT_CANDIDATES
    return {
        "candidate_set_id": candidate_set_id,
        "source_path": source_path,
        "source_plan": source_plan,
        "dataset": dataset,
        "user_id": user_id,
        "positive_item_id": positive_item_id,
        "candidate_items": json.dumps(items),
        "candidate_count": str(len(items)),
        "negative_count": str(len(items) - 1),
        "stored_positive_position": stored_positive_position,
        "stored_position_base": "one_based",
        "computed_positive_position_zero_based": computed_positive_position_zero_based,
        "computed_positive_position_one_based": str(int(computed_positive_position_zero_based) + 1),
        "positive_occurrence_count": "1",
        "candidate_items_unique": "true",
        "positive_in_candidates": "true",
        "user_mapping_valid": "true",
        "positive_item_mapping_valid": "true",
        "all_candidate_items_mapping_valid": "true",
        "all_candidate_titles_available": "true",
        "ground_truth_match": "true",
        "negative_history_exclusion_valid": "true",
        "row_validation_status": row_validation_status,
        "validation_issues": "",
    }


def _title_map(resolved_items: list[str] | None = None) -> dict[tuple[str, str], dict[str, str]]:
    resolved_items = resolved_items if resolved_items is not None else DEFAULT_CANDIDATES
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for item_id in resolved_items:
        rows[("synthetic", item_id)] = {
            "dataset": "synthetic",
            "item_id": item_id,
            "canonical_title": f"title {item_id}",
            "title_status": "resolved_single_source",
            "selected_source_path": "catalog.csv",
        }
    return rows


def _summary_row(rows: list[dict[str, object]], cohort_id: str) -> dict[str, object]:
    matches = [row for row in rows if row["cohort_id"] == cohort_id]
    assert len(matches) == 1
    return matches[0]

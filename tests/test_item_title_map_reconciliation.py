from __future__ import annotations

import json

from src.analysis.item_title_map_reconciliation import (
    ITEM_TITLE_COVERAGE_COLUMNS,
    TitleSourceRecord,
    aggregate_candidate_usage,
    alternate_titles_json,
    cleanup_title,
    comparison_normalize_title,
    coverage_summary_by_candidate_set,
    generate_issue_rows,
    is_missing_title,
    preserve_item_id,
    reconcile_title_map,
)


def candidate_row(
    *,
    dataset: str = "Demo",
    candidate_set_id: str = "set_a",
    user_id: str = "u1",
    positive_item_id: str = "001",
    items: list[str] | None = None,
) -> dict[str, str]:
    item_list = items or ["001", "002", "003"]
    return {
        "candidate_set_id": candidate_set_id,
        "source_path": "tmp/candidates.csv",
        "source_plan": "test",
        "dataset": dataset,
        "user_id": user_id,
        "positive_item_id": positive_item_id,
        "candidate_items": json.dumps(item_list),
        "candidate_count": str(len(item_list)),
        "negative_count": str(max(0, len(item_list) - 1)),
        "stored_positive_position": "1",
        "stored_position_base": "one_based",
        "computed_positive_position_zero_based": "0",
        "computed_positive_position_one_based": "1",
        "positive_occurrence_count": "1",
        "candidate_items_unique": "true",
        "positive_in_candidates": "true",
        "user_mapping_valid": "true",
        "positive_item_mapping_valid": "true",
        "all_candidate_items_mapping_valid": "true",
        "all_candidate_titles_available": "true",
        "ground_truth_match": "true",
        "negative_history_exclusion_valid": "true",
        "row_validation_status": "valid",
        "validation_issues": "",
    }


def source_record(
    item_id: str,
    title: str,
    *,
    dataset: str = "Demo",
    priority: int = 10,
    source_path: str = "source_a.csv",
    mapping_status: str = "verified",
) -> TitleSourceRecord:
    return TitleSourceRecord(
        dataset=dataset,
        item_id=item_id,
        title=cleanup_title(title),
        source_path=source_path,
        source_field="title",
        source_origin="test",
        source_priority=priority,
        mapping_status=mapping_status,
    )


def map_by_item(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["item_id"]): row for row in rows}


def test_preserve_item_ids_as_strings_and_leading_zeros() -> None:
    assert preserve_item_id("00123") == "00123"
    assert preserve_item_id(123) == "123"
    assert preserve_item_id(" 0007 ") == "0007"


def test_safe_whitespace_cleanup_and_json_string_decode() -> None:
    assert cleanup_title("  Alpha   Beta\nGamma\t ") == "Alpha Beta Gamma"
    assert cleanup_title('"Alpha Beta"') == "Alpha Beta"


def test_comparison_normalized_agreement_keeps_display_title() -> None:
    assert comparison_normalize_title(" Alpha   Beta ") == "Alpha Beta"
    assert cleanup_title("Alpha   Beta") == "Alpha Beta"


def test_single_source_resolution() -> None:
    usage = aggregate_candidate_usage([candidate_row(items=["001"])])
    rows, issues = reconcile_title_map([source_record("001", "Only Title")], usage)
    row = map_by_item(rows)["001"]
    assert row["title_status"] == "resolved_single_source"
    assert row["canonical_title"] == "Only Title"
    assert issues == []


def test_multiple_source_exact_agreement() -> None:
    usage = aggregate_candidate_usage([candidate_row(items=["001"])])
    rows, _ = reconcile_title_map(
        [
            source_record("001", "Same Title", priority=10),
            source_record("001", "Same Title", priority=20, source_path="source_b.csv"),
        ],
        usage,
    )
    row = map_by_item(rows)["001"]
    assert row["title_status"] == "resolved_agreement"
    assert row["title_conflict_count"] == 0


def test_deterministic_priority_resolution_for_conflict() -> None:
    usage = aggregate_candidate_usage([candidate_row(items=["001"])])
    rows, issues = reconcile_title_map(
        [
            source_record("001", "Lower Priority Title", priority=20, source_path="b.csv"),
            source_record("001", "Preferred Title", priority=10, source_path="a.csv"),
        ],
        usage,
    )
    row = map_by_item(rows)["001"]
    assert row["title_status"] == "resolved_by_priority"
    assert row["canonical_title"] == "Preferred Title"
    assert row["title_conflict_count"] == 1
    alternates = json.loads(str(row["alternate_titles"]))
    assert alternates[0]["title"] == "Lower Priority Title"
    assert any(issue["issue_type"] == "conflicting_titles" for issue in issues)


def test_missing_title_and_null_like_handling() -> None:
    assert is_missing_title("")
    assert is_missing_title(" none ")
    assert is_missing_title("NaN")
    usage = aggregate_candidate_usage([candidate_row(items=["001"])])
    rows, issues = reconcile_title_map([source_record("001", "none")], usage)
    row = map_by_item(rows)["001"]
    assert row["title_status"] == "missing_title"
    assert row["canonical_title"] == ""
    assert any(issue["issue_type"] == "missing_title" for issue in issues)


def test_duplicate_identical_source_records_generate_warning_issue() -> None:
    usage = aggregate_candidate_usage([candidate_row(items=["001"])])
    rows, issues = reconcile_title_map(
        [
            source_record("001", "Same Title"),
            source_record("001", "Same Title"),
        ],
        usage,
    )
    assert map_by_item(rows)["001"]["title_status"] == "resolved_agreement"
    assert any(issue["issue_type"] == "duplicate_source_record" for issue in issues)


def test_malformed_source_record_issue_generation() -> None:
    issues = generate_issue_rows(
        [],
        malformed_records=[
            type(
                "Malformed",
                (),
                {
                    "dataset": "Demo",
                    "item_id": "",
                    "source_path": "bad.csv",
                    "source_field": "title",
                    "issue_type": "malformed_source_record",
                    "evidence": "missing item_id",
                    "notes": "synthetic",
                },
            )()
        ],
    )
    assert issues[0]["issue_type"] == "malformed_source_record"
    assert issues[0]["severity"] == "warning"


def test_unverifiable_mapping_status() -> None:
    usage = aggregate_candidate_usage([candidate_row(items=["001"])])
    rows, issues = reconcile_title_map(
        [source_record("001", "Title", mapping_status="unverifiable")],
        usage,
    )
    row = map_by_item(rows)["001"]
    assert row["title_status"] == "unverifiable_mapping"
    assert row["canonical_title"] == ""
    assert any(issue["issue_type"] == "unverifiable_id_mapping" for issue in issues)


def test_candidate_and_positive_occurrence_aggregation() -> None:
    usage = aggregate_candidate_usage(
        [
            candidate_row(candidate_set_id="a", positive_item_id="001", items=["001", "002"]),
            candidate_row(candidate_set_id="b", positive_item_id="002", items=["002", "003"]),
        ]
    )
    assert usage[("Demo", "002")].candidate_occurrence_count == 2
    assert usage[("Demo", "002")].positive_occurrence_count == 1
    assert usage[("Demo", "002")].candidate_set_ids == {"a", "b"}


def test_alternate_title_json_serialization_is_array() -> None:
    payload = alternate_titles_json(
        [
            {
                "title": "B",
                "source_path": "b.csv",
                "source_field": "title",
                "source_origin": "test",
                "source_priority": 20,
            }
        ]
    )
    decoded = json.loads(payload)
    assert isinstance(decoded, list)
    assert decoded[0]["title"] == "B"


def test_title_coverage_calculation() -> None:
    candidate_rows = [
        candidate_row(candidate_set_id="a", positive_item_id="001", items=["001", "002"]),
        candidate_row(candidate_set_id="a", user_id="u2", positive_item_id="002", items=["002", "003"]),
    ]
    usage = aggregate_candidate_usage(candidate_rows)
    map_rows, _ = reconcile_title_map(
        [source_record("001", "One"), source_record("002", "Two")],
        usage,
        include_source_only_items=False,
    )
    coverage = coverage_summary_by_candidate_set(candidate_rows, map_rows)
    assert list(coverage[0]) == ITEM_TITLE_COVERAGE_COLUMNS
    assert coverage[0]["candidate_row_count"] == 2
    assert coverage[0]["unique_item_count"] == 3
    assert coverage[0]["titled_unique_item_count"] == 2
    assert coverage[0]["rows_with_all_titles"] == 1
    assert coverage[0]["rows_with_missing_titles"] == 1
    assert coverage[0]["positive_title_coverage"] == 1.0


def test_deterministic_output_ordering_numeric_then_textual_ids() -> None:
    usage = aggregate_candidate_usage(
        [
            candidate_row(positive_item_id="10", items=["10", "2", "A1"]),
        ]
    )
    rows, _ = reconcile_title_map(
        [
            source_record("A1", "A"),
            source_record("10", "Ten"),
            source_record("2", "Two"),
        ],
        usage,
        include_source_only_items=False,
    )
    assert [row["item_id"] for row in rows] == ["2", "10", "A1"]

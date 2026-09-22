from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.dataset_candidate_set_validation import (
    CandidateSetValidationError,
    DatasetContext,
    ListCandidateArtifact,
    RowCandidateArtifact,
    canonicalize_list_candidate_artifact,
    canonicalize_row_candidate_artifact,
    derive_positive_positions,
    parse_candidate_items,
    validate_candidate_row,
)


def test_valid_20_candidate_row() -> None:
    row = _validate(candidate_items=[str(i) for i in range(20)], positive_item_id="7", stored_positive_position=7)

    assert row["candidate_count"] == 20
    assert row["negative_count"] == 19
    assert row["row_validation_status"] == "valid"
    assert row["stored_position_base"] == "zero_based"


def test_valid_100_candidate_row() -> None:
    row = _validate(
        candidate_items=[str(i) for i in range(100)],
        positive_item_id="99",
        stored_positive_position=100,
        expected_candidate_count=100,
    )

    assert row["candidate_count"] == 100
    assert row["negative_count"] == 99
    assert row["row_validation_status"] == "valid"
    assert row["stored_position_base"] == "one_based"


def test_positive_missing_from_candidates() -> None:
    row = _validate(candidate_items=["a", "b", "c"], positive_item_id="z", stored_positive_position="")

    assert row["positive_in_candidates"] == "false"
    assert row["row_validation_status"] == "invalid"
    assert "positive_occurrence_count_0" in row["validation_issues"]


def test_positive_appearing_twice() -> None:
    row = _validate(candidate_items=["a", "p", "b", "p"], positive_item_id="p", stored_positive_position=1)

    assert row["positive_occurrence_count"] == 2
    assert row["candidate_items_unique"] == "false"
    assert row["row_validation_status"] == "invalid"


def test_duplicate_negative_candidate() -> None:
    row = _validate(candidate_items=["p", "a", "a"], positive_item_id="p", stored_positive_position=1)

    assert row["candidate_items_unique"] == "false"
    assert "duplicate_candidate_items" in row["validation_issues"]


def test_incorrect_stored_positive_position() -> None:
    row = _validate(candidate_items=["p", "a", "b"], positive_item_id="p", stored_positive_position=3)

    assert row["stored_position_base"] == "unknown"
    assert row["row_validation_status"] == "invalid"
    assert "stored_positive_position_mismatch" in row["validation_issues"]


def test_zero_based_stored_position() -> None:
    row = _validate(candidate_items=["a", "p", "b"], positive_item_id="p", stored_positive_position=1)

    assert row["stored_position_base"] == "zero_based"
    assert row["computed_positive_position_zero_based"] == "1"


def test_one_based_stored_position() -> None:
    row = _validate(candidate_items=["a", "p", "b"], positive_item_id="p", stored_positive_position=2)

    assert row["stored_position_base"] == "one_based"
    assert row["computed_positive_position_one_based"] == "2"


def test_unknown_position_base() -> None:
    row = _validate(candidate_items=["a", "p", "b"], positive_item_id="p", stored_positive_position=9)

    assert row["stored_position_base"] == "unknown"
    assert row["row_validation_status"] == "invalid"


def test_malformed_candidate_json_list() -> None:
    with pytest.raises(CandidateSetValidationError):
        parse_candidate_items("[not-json")


def test_empty_user_id() -> None:
    row = validate_candidate_row(
        candidate_set_id="set",
        source_path="source.csv",
        source_plan="test",
        dataset="synthetic",
        user_id="",
        positive_item_id="p",
        candidate_items=["p", "a"],
        stored_positive_position=1,
        expected_candidate_count=2,
        context=_context(),
    )

    assert row["row_validation_status"] == "invalid"
    assert "empty_user_id" in row["validation_issues"]


def test_mapping_validation() -> None:
    context = _context(user_ids={"u1"}, item_ids={"p", "a"})
    valid = _validate(candidate_items=["p", "a"], positive_item_id="p", stored_positive_position=1, context=context, user_id="u1", expected_candidate_count=2)
    invalid = _validate(candidate_items=["p", "missing"], positive_item_id="p", stored_positive_position=1, context=context, user_id="u1", expected_candidate_count=2)

    assert valid["all_candidate_items_mapping_valid"] == "true"
    assert invalid["all_candidate_items_mapping_valid"] == "false"
    assert invalid["row_validation_status"] == "invalid"


def test_title_coverage_calculation() -> None:
    context = _context(user_ids={"u1"}, item_ids={"p", "a"}, title_by_item={"p": "Positive", "a": ""})
    row = _validate(candidate_items=["p", "a"], positive_item_id="p", stored_positive_position=1, context=context, user_id="u1", expected_candidate_count=2)

    assert row["all_candidate_titles_available"] == "false"
    assert row["row_validation_status"] == "valid_with_warnings"
    assert "candidate_title_mapping_incomplete" in row["validation_issues"]


def test_deterministic_ordering_from_row_candidate_artifact(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {"user_id": "u1", "candidate_item_id": "b", "candidate_position": "2", "is_ground_truth": "false"},
            {"user_id": "u1", "candidate_item_id": "p", "candidate_position": "1", "is_ground_truth": "true"},
            {"user_id": "u1", "candidate_item_id": "a", "candidate_position": "3", "is_ground_truth": "false"},
        ]
    ).to_csv(path, index=False)
    artifact = RowCandidateArtifact(
        candidate_set_id="synthetic",
        source_path=Path("candidates.csv"),
        source_plan="test",
        dataset="synthetic",
        context_key="synthetic",
        expected_candidate_count=3,
        paper_alignment="test",
        row_format="row_per_candidate",
    )

    frame = canonicalize_row_candidate_artifact(artifact, project_root=tmp_path, context=_context(user_ids={"u1"}, item_ids={"p", "a", "b"}))

    assert parse_candidate_items(frame.iloc[0]["candidate_items"]) == ["p", "b", "a"]


def test_list_candidate_artifact_uses_suffix_for_unique_candidate_set_ids(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"user_id": "u1", "positive_item_id": "p", "variant": "positive_first", "position": "1", "candidate_ids": '["p", "a"]'},
            {"user_id": "u1", "positive_item_id": "p", "variant": "positive_last", "position": "2", "candidate_ids": '["a", "p"]'},
        ]
    ).to_csv(path, index=False)
    artifact = ListCandidateArtifact(
        candidate_set_id_prefix="synthetic",
        source_path=Path("manifest.csv"),
        source_plan="test",
        dataset="synthetic",
        context_key="synthetic",
        expected_candidate_count=2,
        paper_alignment="test",
        row_format="row_per_user_json_list",
        stored_position_column="position",
        candidate_set_suffix_column="variant",
    )

    frame = canonicalize_list_candidate_artifact(artifact, project_root=tmp_path, context=_context(user_ids={"u1"}, item_ids={"p", "a"}))

    assert list(frame["candidate_set_id"]) == ["synthetic_positive_first", "synthetic_positive_last"]
    assert frame["row_validation_status"].tolist() == ["valid", "valid"]


def test_derivation_of_positive_position_output() -> None:
    candidate_frame = pd.DataFrame(
        [
            _validate(candidate_items=["a", "p"], positive_item_id="p", stored_positive_position=2, expected_candidate_count=2),
            _validate(candidate_items=["p", "a"], positive_item_id="p", stored_positive_position="", expected_candidate_count=2, user_id="u2"),
        ]
    )

    positions = derive_positive_positions(candidate_frame)

    assert len(positions) == len(candidate_frame)
    assert positions.iloc[0]["position_matches_stored"] == "true"
    assert positions.iloc[0]["position_validation_status"] == "valid"
    assert positions.iloc[1]["position_matches_stored"] == "not_verifiable_from_current_artifacts"


def _validate(
    *,
    candidate_items: list[str],
    positive_item_id: str,
    stored_positive_position: str | int | None,
    expected_candidate_count: int | None = 20,
    context: DatasetContext | None = None,
    user_id: str = "u1",
) -> dict[str, str | int]:
    return validate_candidate_row(
        candidate_set_id="set",
        source_path="source.csv",
        source_plan="test",
        dataset="synthetic",
        user_id=user_id,
        positive_item_id=positive_item_id,
        candidate_items=candidate_items,
        stored_positive_position=stored_positive_position,
        expected_candidate_count=expected_candidate_count,
        context=context
        if context is not None
        else _context(user_ids={user_id}, item_ids=set(candidate_items), ground_truth_by_user={user_id: positive_item_id}),
    )


def _context(
    *,
    user_ids: set[str] | None = None,
    item_ids: set[str] | None = None,
    title_by_item: dict[str, str] | None = None,
    ground_truth_by_user: dict[str, str] | None = None,
) -> DatasetContext:
    user_ids = user_ids or {"u1"}
    item_ids = item_ids or {"p", "a", "b", *{str(i) for i in range(100)}}
    return DatasetContext(
        dataset="synthetic",
        user_ids=frozenset(user_ids),
        item_ids=frozenset(item_ids),
        title_by_item=title_by_item or {item: f"title {item}" for item in item_ids},
        ground_truth_by_user=ground_truth_by_user if ground_truth_by_user is not None else {user: "p" for user in user_ids if user != "u2"},
        history_by_user={user: frozenset({"h1", "h2"}) for user in user_ids},
    )

from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis.phase_f6i1_repository_retention_audit import (
    DELETE_COLUMNS,
    RETENTION_CLASSES,
    FileRecord,
    blocking_gate_for,
    build_classification_rows,
    build_edges,
    category_for,
    classify_record,
    duplicate_groups,
    future_dependencies,
    is_protected_path,
    normalize_path,
    source_for_pyc,
    unique_scientific_information,
)


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
]


def record(path: str, text: str = "", sha: str = "A" * 64, size: int = 1) -> FileRecord:
    return FileRecord(
        path=normalize_path(path),
        absolute_path=Path(path),
        size_bytes=size,
        sha256=sha,
        tracked_state="untracked",
        category=category_for(path),
        extension=Path(path).suffix.lower().lstrip(".") or "no_extension",
        is_directory_unit=path.endswith("/"),
        inspection_status="test",
        text_sample=text,
    )


def test_protected_paths_cover_phase_exclusions() -> None:
    assert is_protected_path("original_repo/EvaLLMasRecommender/README.md")
    assert is_protected_path("paper/2411.00331v1.pdf")
    assert is_protected_path("data/raw/.gitkeep")
    assert is_protected_path("results/llm/run/responses.jsonl")
    assert is_protected_path("configs/beauty.yaml")
    assert not is_protected_path("src/evaluation/__pycache__/output_parser.cpython-312.pyc")


def test_cache_without_source_fails_closed() -> None:
    pyc = record("src/analysis/__pycache__/missing.cpython-312.pyc")
    classification, reason, replacement = classify_record(pyc, {"incoming": {}, "outgoing": {}})
    assert classification == "manual_review"
    assert reason == "no_retained_source_for_regeneration"
    assert replacement == ""
    assert blocking_gate_for(reason) == "gate_9_canonical_replacement_or_regeneration_path"


def test_source_and_test_classes_are_retained() -> None:
    source = record("src/analysis/x.py")
    test = record("tests/test_x.py")
    assert classify_record(source, {"incoming": {}, "outgoing": {}})[0] == "keep_canonical_builder"
    assert classify_record(test, {"incoming": {}, "outgoing": {}})[0] == "keep_canonical_test"


def test_future_dependency_detects_roadmap_terms() -> None:
    row = record("results/tables/x.csv", "Qwen2 reranking top-20 candidate position hallucination diversity")
    dependencies = {dependency for dependency, _evidence in future_dependencies(row)}
    assert "local open LLMs" in dependencies
    assert "ranking and reranking" in dependencies
    assert "top-20 candidate generation" in dependencies
    assert "hallucination" in dependencies
    assert "diversity" in dependencies


def test_unique_scientific_information_blocks_deletion_terms() -> None:
    row = record("report/x.md", "The seed and prompt policy are required for metric interpretation.")
    value = unique_scientific_information(row)
    assert value.startswith("yes:")
    assert "report" in value


def test_duplicate_groups_keep_semantic_paths_by_default() -> None:
    first = record("data/processed/.gitkeep", sha="B" * 64, size=0)
    second = record("data/external/plan_c/.gitkeep", sha="B" * 64, size=0)
    rows = duplicate_groups([first, second], {first.path: "keep_future_dependency", second.path: "keep_future_dependency"})
    assert len(rows) == 2
    assert {row["delete_eligible"] for row in rows} == {"false"}


def test_build_classification_rows_delete_candidates_have_required_columns() -> None:
    records = [record("src/analysis/x.py", "from src.analysis import y"), record("src/analysis/y.py", "")]
    edges = build_edges(records)
    classification, delete_rows, manual_rows, by_path = build_classification_rows(records, edges)
    assert classification
    assert not delete_rows
    assert not manual_rows
    assert set(by_path.values()).issubset(RETENTION_CLASSES)
    assert DELETE_COLUMNS[0] == "path"


def test_source_for_pyc_maps_cache_to_source_when_present() -> None:
    # This repository source exists, so the cache mapping can be validated without
    # creating a pyc file in the test.
    source = source_for_pyc("src/analysis/__pycache__/phase_f6i1_repository_retention_audit.cpython-312.pyc")
    assert source == "src/analysis/phase_f6i1_repository_retention_audit.py"

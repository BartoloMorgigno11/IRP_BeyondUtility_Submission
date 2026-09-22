"""Focused checks for F6I2I path normalization consolidation."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from src.analysis.plan_f6i2i_path_normalization_consolidation_execution import (
    APPROVED_INTERPRETER,
    CONFIG_PATH,
    DECISION_COMPLETE,
    F6I2G_DECISION_TOKEN,
    F6I2H_DECISION_TOKEN,
    MEMBER_FILES,
    PHASE_ID,
    PROJECT_ROOT,
    build,
    semantic_digest,
)
from src.audit.common_paths import (
    path_as_posix,
    relative_to_root,
    resolved_relative_to_root,
    resolved_relative_to_root_or_absolute,
)


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
    pytest.mark.refactor_regression,
]


@lru_cache(maxsize=1)
def built_result():
    return build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)


def test_runtime_and_authoritative_contracts() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    result = built_result()
    manifest = result.manifests["consolidation_manifest"]
    assert PHASE_ID == "phase_f6i2i"
    assert manifest["previous_phase_decision_token"] == F6I2H_DECISION_TOKEN
    assert result.decision_token == DECISION_COMPLETE
    assert len(manifest["member_files"]) == 7
    assert manifest["selected_group"] == "path_normalization"
    assert F6I2G_DECISION_TOKEN


def test_common_paths_api_preserves_observed_semantics() -> None:
    inside = PROJECT_ROOT / "results" / "tables" / "missing_path_fixture.csv"
    outside = PROJECT_ROOT.parent / "outside_missing_path_fixture.csv"
    assert resolved_relative_to_root(inside, PROJECT_ROOT) == "results/tables/missing_path_fixture.csv"
    assert relative_to_root(inside, PROJECT_ROOT) == "results/tables/missing_path_fixture.csv"
    assert resolved_relative_to_root_or_absolute(inside, PROJECT_ROOT) == "results/tables/missing_path_fixture.csv"
    assert resolved_relative_to_root_or_absolute(outside, PROJECT_ROOT) == str(outside.resolve())
    assert path_as_posix(Path("archive\\phase_reports\\report.md")) == "archive/phase_reports/report.md"
    with pytest.raises(ValueError):
        relative_to_root(Path("relative.txt"), PROJECT_ROOT)
    with pytest.raises(ValueError):
        resolved_relative_to_root(outside, PROJECT_ROOT)


def test_windows_fixture_matrix_and_baseline_are_complete() -> None:
    result = built_result()
    baseline = result.tables["function_equivalence"]
    fixture_names = {row["fixture_name"] for row in baseline}
    assert len(baseline) == 266
    assert {
        "absolute_windows_path",
        "mixed_separators",
        "trailing_separator",
        "uppercase_drive",
        "lowercase_drive",
        "unicode_filename",
        "archive_path",
        "nonexistent_inside_root",
    } <= fixture_names
    assert all(row["status"] == "pass" for row in baseline)


def test_wrappers_retained_delegate_and_remove_duplicate_bodies() -> None:
    result = built_result()
    wrappers = result.tables["wrapper_compatibility"]
    assert len(wrappers) == len(MEMBER_FILES)
    assert all(row["signature_preserved"] == "true" for row in wrappers)
    assert all(row["wrapper_retained"] == "true" for row in wrappers)
    assert all(row["delegates_to_canonical"] == "true" for row in wrappers)
    assert all(row["duplicate_normalization_body_removed"] == "true" for row in wrappers)
    assert all(row["status"] == "pass" for row in wrappers)


def test_function_and_output_equivalence() -> None:
    result = built_result()
    assert len(result.tables["function_equivalence"]) == 266
    assert all(row["status"] == "pass" for row in result.tables["function_equivalence"])
    assert len(result.tables["output_hash_comparison"]) == 266
    assert all(row["output_mismatch"] == "false" for row in result.tables["output_hash_comparison"])
    assert all(row["path_spelling_changed"] == "false" for row in result.tables["output_hash_comparison"])


def test_dependency_future_and_protected_validation() -> None:
    result = built_result()
    graph = result.tables["dependency_graph_comparison"]
    assert graph[0]["scope_definition"] == graph[1]["scope_definition"]
    assert graph[1]["common_paths_importer_count"] == "7"
    assert graph[1]["circular_imports"] == ""
    assert all(row["status"] == "pass" for row in graph)
    assert all(row["status"] == "pass" for row in result.tables["future_dependency_validation"])
    assert result.protected_hashes["status"] == "pass"
    assert result.protected_hashes["changed"] == []
    assert result.protected_hashes["missing"] == []
    assert all(row["status"] == "pass" for row in result.tables["validation"])


def test_builder_classification_and_wrapper_removal_are_conservative() -> None:
    result = built_result()
    classes = {row["classification"] for row in result.tables["builder_equivalence"]}
    assert "unsafe_requires_training" in classes
    assert "frozen_output_equivalence" in classes
    assert {row["wrapper_removal_status"] for row in result.tables["wrapper_removal_readiness"]} == {
        "retain_phase_provenance"
    }
    assert result.manifests["consolidation_manifest"]["counts"]["files_deleted"] == 0
    assert result.manifests["consolidation_manifest"]["counts"]["file_reduction_claimed"] == 0


def test_in_memory_build_is_deterministic() -> None:
    first = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    second = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    assert semantic_digest(first) == semantic_digest(second)

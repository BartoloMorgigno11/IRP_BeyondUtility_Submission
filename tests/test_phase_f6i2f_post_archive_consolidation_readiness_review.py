"""Focused checks for F6I2F post-archive consolidation readiness review."""

from __future__ import annotations

import os
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from src.analysis.plan_f6i2f_post_archive_consolidation_readiness_review import (
    APPROVED_INTERPRETER,
    CONFIG_PATH,
    DECISION_COMPLETE,
    EXPECTED_F6I2A_PYTHON_COUNT,
    EXPECTED_F6I2E_REPORTED_AST_COUNT,
    EXPECTED_MOVES,
    EXPECTED_TRACKED_DELETIONS_FROM_F6I2E,
    EXPECTED_UNTRACKED_MOVES,
    F6I2E_DECISION_TOKEN,
    PHASE_ID,
    PROJECT_ROOT,
    build,
    load_inputs,
    semantic_digest,
)


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
    pytest.mark.migration_regression,
]


@lru_cache(maxsize=1)
def built_result():
    return build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)


def test_runtime_and_f6i2e_authority() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    inputs = load_inputs(CONFIG_PATH)
    assert PHASE_ID == "phase_f6i2f"
    assert inputs.manifests["f6i2e_migration_manifest"]["decision_token"] == F6I2E_DECISION_TOKEN
    assert inputs.manifests["f6i2e_migration_manifest"]["counts"]["files_moved"] == EXPECTED_MOVES
    assert inputs.manifests["f6i2e_migration_manifest"]["counts"]["bytes_migrated"] == 303523


def test_archive_move_reconciliation_explains_git_state() -> None:
    result = built_result()
    rows = result.tables["archive_move_reconciliation"]
    classes = Counter(row["move_classification"] for row in rows)
    assert len(rows) == EXPECTED_MOVES
    assert classes["tracked_source_to_untracked_archive"] == EXPECTED_TRACKED_DELETIONS_FROM_F6I2E
    assert classes["untracked_source_to_untracked_archive"] == EXPECTED_UNTRACKED_MOVES
    assert all(row["byte_identity"] == "true" for row in rows)
    assert all(row["archive_index_entry"] == "true" for row in rows)
    assert all(row["rollback_entry"] == "true" for row in rows)
    assert all(row["original_path_currently_absent"] == "true" for row in rows)
    assert all(row["archive_path_currently_present"] == "true" for row in rows)


def test_python_counts_keep_active_source_separate_from_broad_scan() -> None:
    result = built_result()
    count_by_scope = {row["scope"]: int(row["count"]) for row in result.tables["python_count_reconciliation"]}
    current_src_count = sum(1 for path in (PROJECT_ROOT / "src").rglob("*.py") if path.is_file())
    assert count_by_scope["f6i2a_source_file_audit_table"] == EXPECTED_F6I2A_PYTHON_COUNT
    assert count_by_scope["f6i2e_reported_ast_parse_active_retained_python"] == EXPECTED_F6I2E_REPORTED_AST_COUNT
    assert count_by_scope["authoritative_active_local_python_for_consolidation"] == current_src_count
    assert count_by_scope["environment_dependency_python"] > count_by_scope["authoritative_active_local_python_for_consolidation"]
    excluded_scopes = {
        row["path_pattern"]
        for row in result.tables["source_scope_contract"]
        if row["included_in_semantic_consolidation"] == "false"
    }
    assert ".venv_audit_recovery" in excluded_scopes
    assert "archive" in excluded_scopes
    assert "original_repo" in excluded_scopes
    assert "results/environment" in excluded_scopes


def test_cleanup_test_lifecycle_and_current_suite_plan() -> None:
    result = built_result()
    rows = result.tables["test_lifecycle_audit"]
    f6i2b_rows = [row for row in rows if row["phase"] == "F6I2B"]
    assert f6i2b_rows
    assert all(row["lifecycle_classification"] == "migration_superseded_invariant" for row in f6i2b_rows)
    current_rows = [row for row in rows if row["lifecycle_classification"] == "current_state_invariant"]
    assert {row["phase"] for row in current_rows} == {"F6I2E", "F6I2F"}
    suite_checks = {row["check_name"]: row["treatment"] for row in result.tables["current_validation_suite_plan"]}
    assert suite_checks["old_path_absent"] == "include"
    assert suite_checks["archive_hash_identity"] == "include"
    assert suite_checks["f6i2b_retained_path_assertion"] == "exclude_current_state"


def test_dependency_graph_has_active_nodes_and_distinct_edge_types() -> None:
    result = built_result()
    nodes = result.tables["dependency_graph_nodes"]
    edges = result.tables["dependency_graph_edges"]
    assert nodes
    assert edges
    assert all(not row["path"].startswith((".venv", "archive/", "original_repo/", "results/environment/")) for row in nodes)
    edge_types = {row["edge_type"] for row in edges}
    assert "active_import" in edge_types
    assert "test_target" in edge_types
    assert "archive_reference" in edge_types
    assert "future_roadmap_dependency" in edge_types
    assert "historical_reference" in edge_types


def test_merge_group_priority_and_first_contract() -> None:
    result = built_result()
    merge_rows = result.tables["python_merge_group_revalidation"]
    priority_rows = result.tables["python_merge_priority"]
    selected = [row for row in priority_rows if row["selected_for_first_bundle"] == "true"]
    assert len(merge_rows) == 9
    assert len(selected) == 1
    assert selected[0]["merge_group_id"] == "PYMERGE-F6I2A-002"
    assert selected[0]["group_name"] == "deterministic_csv_io"
    ready = [row for row in merge_rows if row["current_status"] == "ready_for_consolidation"]
    assert [row["group_name"] for row in ready] == ["deterministic_csv_io"]
    contract = result.tables["first_python_consolidation_contract"]
    assert len(contract) == 1
    assert contract[0]["target_shared_module"] == "src/audit/common_io.py"
    assert contract[0]["compatibility_wrappers_required"] == "true"
    assert contract[0]["implementation_in_f6i2f"] == "false"


def test_config_group_overlap_and_validation() -> None:
    result = built_result()
    config_rows = result.tables["config_group_revalidation"]
    assert len(config_rows) == 6
    assert sum(int(row["f6i2c_redirected_overlap_count"]) for row in config_rows) == 9
    phase_archive = [row for row in config_rows if row["group_name"] == "phase_specific_archive"][0]
    assert phase_archive["f6i2c_redirected_overlap_count"] == "9"
    assert all(row["status"] == "pass" for row in result.tables["validation"])
    assert result.protected_hashes["status"] == "pass"
    assert result.decision_token == DECISION_COMPLETE


def test_in_memory_build_is_deterministic() -> None:
    first = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    second = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    assert semantic_digest(first) == semantic_digest(second)

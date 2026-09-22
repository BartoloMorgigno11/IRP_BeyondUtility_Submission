"""Focused checks for F6I2E substantive archive migration execution."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from src.analysis.plan_f6i2e_substantive_archive_migration_execution import (
    APPROVED_INTERPRETER,
    CONFIG_PATH,
    EXPECTED_ACTIVE_REFERENCE_ROWS,
    EXPECTED_APPROVED,
    EXPECTED_HEAD,
    F6I2D_DECISION_TOKEN,
    PHASE_ID,
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
    return build(
        CONFIG_PATH,
        write_outputs_enabled=False,
        execute_migration_enabled=False,
        update_current_state=False,
    )


def test_runtime_and_authoritative_f6i2d_counts() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    inputs = load_inputs(CONFIG_PATH)
    assert PHASE_ID == "phase_f6i2e"
    assert inputs.f6i2d_migration_manifest["decision_token"] == F6I2D_DECISION_TOKEN
    assert inputs.f6i2d_migration_manifest["final_git_state"]["head"] == EXPECTED_HEAD
    assert inputs.f6i2d_migration_manifest["counts"]["archive_ready_for_migration"] == EXPECTED_APPROVED
    assert inputs.f6i2d_migration_manifest["counts"]["active_reference_rewrite_rows"] == EXPECTED_ACTIVE_REFERENCE_ROWS
    assert inputs.f6i2d_migration_manifest["counts"]["compatibility_stub_or_alias_required"] == 0


def test_preflight_uses_exact_approved_set_only() -> None:
    result = built_result()
    inputs = load_inputs(CONFIG_PATH)
    approved_paths = {row["original_path"] for row in inputs.f6i2d_archive_ready}
    preflight_paths = {row["original_path"] for row in result.tables["candidate_preflight"]}
    assert len(result.tables["candidate_preflight"]) == EXPECTED_APPROVED
    assert preflight_paths == approved_paths
    assert result.tables["migration_rejected"] == []
    assert all(row["preflight_status"].startswith("pass") for row in result.tables["candidate_preflight"])


def test_migration_actions_are_restart_safe_and_literal() -> None:
    result = built_result()
    actions = result.tables["migration_actions"]
    assert len(actions) == EXPECTED_APPROVED
    assert {row["action_status"] for row in actions} <= {"not_executed", "moved", "already_migrated"}
    assert all(row["archive_path"].startswith("archive/") for row in actions)
    assert all(row["validation_status"] in {"dry_run_only", "pass"} for row in actions)
    assert sum(int(row["approved_size_bytes"]) for row in actions) == 303523


def test_reference_rewrite_rows_and_duplicate_groups_are_represented() -> None:
    result = built_result()
    actions = result.tables["reference_rewrite_actions"]
    validation = result.tables["reference_rewrite_validation"]
    assert len(actions) == EXPECTED_ACTIVE_REFERENCE_ROWS
    assert len(validation) == 6
    assert sum(int(row["approved_logical_rows"]) for row in validation) == EXPECTED_ACTIVE_REFERENCE_ROWS
    assert {row["referencer_path"] for row in actions} == {"PROJECT_SPEC.md", "CODEX_PROMPTS.md"}
    assert all(row["validation_status"] in {"dry_run_only", "pass"} for row in actions)


def test_archive_index_and_rollback_schema() -> None:
    result = built_result()
    archive_index = result.manifests["archive_index"]["entries"]
    assert len(archive_index) == EXPECTED_APPROVED
    assert len(result.tables["rollback_plan"]) == EXPECTED_APPROVED
    assert all(row["archived_path"].startswith("archive/") for row in archive_index)
    assert all(row["rollback_order"] for row in result.tables["rollback_plan"])


def test_discovery_validation_and_protected_inputs() -> None:
    result = built_result()
    assert all(row["status"] == "pass" for row in result.tables["validation"])
    assert all(row["status"] == "pass" for row in result.tables["python_discovery_validation"])
    assert all(row["status"] == "pass" for row in result.tables["test_discovery_validation"])
    assert all(row["status"] == "pass" for row in result.tables["config_discovery_validation"])
    assert result.protected_hashes["status"] == "pass"
    assert result.protected_hashes["changed"] == []
    assert result.protected_hashes["missing"] == []


def test_in_memory_build_is_deterministic() -> None:
    first = build(CONFIG_PATH, write_outputs_enabled=False, execute_migration_enabled=False, update_current_state=False)
    second = build(CONFIG_PATH, write_outputs_enabled=False, execute_migration_enabled=False, update_current_state=False)
    assert semantic_digest(first) == semantic_digest(second)

"""Focused checks for F6I2C archive verification and migration design."""

from __future__ import annotations

import os
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from src.analysis.plan_f6i2c_archive_bundle_verification_and_migration_design import (
    APPROVED_INTERPRETER,
    ARCHIVE_PLANNED_CLASSES,
    CONFIG_PATH,
    EXPECTED_HEAD,
    PHASE_ID,
    build,
    load_inputs,
    semantic_digest,
)


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def built_result():
    return build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)


def test_phase_runtime_and_authoritative_inputs() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    inputs = load_inputs(CONFIG_PATH)
    paths = [row["path"] for row in inputs.f6i2a_archive_bundle]
    assert PHASE_ID == "phase_f6i2c"
    assert len(paths) == 179
    assert len(set(paths)) == 179
    assert inputs.f6i2b_manifest["decision_token"] == "f6i2b_low_risk_deletion_complete"
    assert inputs.f6i2b_manifest["summary"]["deleted_count"] == 8
    assert inputs.f6i2b_manifest["summary"]["bytes_removed"] == 0


def test_preflight_represents_changed_and_missing_candidates() -> None:
    result = built_result()
    preflight = result.tables["candidate_preflight"]
    assert len(preflight) == 179
    assert Counter(row["preflight_class"] for row in preflight) == {
        "archive_preflight_unchanged_since_f6i2a": 169,
        "archive_preflight_changed_since_f6i2a": 10,
    }
    changed = {row["path"] for row in preflight if row["preflight_class"] == "archive_preflight_changed_since_f6i2a"}
    assert {
        "report/notes.md",
        "report/phase_f6i1_repository_retention_audit_and_safe_cleanup_batch1.md",
        "results/logs/phase_f6e4b_background_stderr.log",
        "results/logs/phase_f6e4b_resume_stdout.log",
    } <= changed
    assert sum(row["f6i2b_assumption_effect"] == "deleted_by_f6i2b_low_risk_action" for row in preflight) == 8
    assert all(row["manual_review_required"] == "true" for row in preflight if row["preflight_class"] == "archive_preflight_changed_since_f6i2a")


def test_decision_classes_are_complete_and_fail_closed() -> None:
    result = built_result()
    decisions = result.tables["archive_decisions"]
    assert len(decisions) == 179
    assert len({row["path"] for row in decisions}) == 179
    assert Counter(row["current_decision"] for row in decisions) == {
        "archive_ready_after_content_indexing": 12,
        "archive_ready_after_current_summary_migration": 6,
        "archive_rejected_changed_since_f6i2a": 10,
        "keep_current_future_dependency": 12,
        "keep_current_operational": 130,
        "redirect_to_config_consolidation": 9,
    }
    immediate_ready = [row for row in decisions if row["archive_gate_status"] == "passes_immediate_gate"]
    assert immediate_ready == []
    assert all(row["current_decision"] in ARCHIVE_PLANNED_CLASSES for row in result.tables["archive_ready"])


def test_archive_mapping_reference_rewrite_and_collisions() -> None:
    result = built_result()
    mapping = result.tables["archive_path_mapping"]
    ready = result.tables["archive_ready"]
    rewrites = result.tables["reference_rewrite_plan"]
    collisions = result.tables["target_path_collisions"]
    assert len(mapping) == 18
    assert len(ready) == 18
    assert len(rewrites) == 711
    assert collisions == [
        {
            "target_archive_path": "all_planned_targets",
            "collision_count": "0",
            "original_paths": "",
            "status": "pass_collision_free",
        }
    ]
    assert all(row["target_archive_path"].startswith("archive/") for row in mapping)
    assert all(row["rollback_instructions"].startswith("restore ") for row in mapping)


def test_future_dependency_current_summary_and_cache_separation() -> None:
    result = built_result()
    assert len(result.tables["future_dependency_recheck"]) == 179 * 18
    assert len(result.tables["current_summary_migration_plan"]) == 6
    assert len(result.tables["cache_cleanup_bundle"]) == 751
    assert all(row["substantive_archive_candidate"] == "false" for row in result.tables["cache_cleanup_bundle"])
    assert all(row["required_before_archive"] == "true" for row in result.tables["current_summary_migration_plan"])


def test_validation_manifest_and_protected_hashes() -> None:
    result = built_result()
    assert all(row["status"] == "pass" for row in result.tables["validation"])
    assert result.manifest["decision_token"] == "f6i2c_archive_verification_complete_partial_ready"
    assert result.manifest["runtime"]["not_exact_original_runtime"] is True
    assert result.manifest["git_state"]["head"] == EXPECTED_HEAD
    assert result.manifest["counts"]["target_path_collision_count"] == 0
    assert result.protected_hashes["changed"] == []
    assert result.protected_hashes["missing"] == []


def test_forbidden_actions_and_next_phase_contract() -> None:
    result = built_result()
    assert not any(result.manifest["forbidden_actions"].values())
    assert result.manifest["next_migration_phase"] == "phase_f6i2d_archive_migration_execution_after_explicit_approval"
    assert result.manifest["starting_state_preserved"]["f6i1"]["relevant_inventory_rows"] == 1948
    assert result.manifest["starting_state_preserved"]["f6i2a"]["archive_candidates"] == 179
    assert result.manifest["starting_state_preserved"]["f6i2b"]["candidates_deleted"] == 8


def test_in_memory_build_is_deterministic() -> None:
    first = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    second = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    assert semantic_digest(first) == semantic_digest(second)

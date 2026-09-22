"""Focused checks for F6I2D archive prerequisite closure and cache cleanup."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from src.analysis.plan_f6i2d_archive_prerequisite_closure_and_technical_cache_cleanup import (
    APPROVED_INTERPRETER,
    CONFIG_PATH,
    DECISION_TOKEN_COMPLETE,
    EXPECTED_CACHE_CANDIDATES,
    EXPECTED_CONTENT_CANDIDATES,
    EXPECTED_HEAD,
    EXPECTED_SUMMARY_CANDIDATES,
    NEXT_PHASE,
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
    return build(
        CONFIG_PATH,
        write_outputs_enabled=False,
        execute_cache_cleanup_enabled=False,
        update_current_state=False,
    )


def test_runtime_and_authoritative_input_counts() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    inputs = load_inputs(CONFIG_PATH)
    assert PHASE_ID == "phase_f6i2d"
    assert inputs.f6i2c_manifest["decision_token"] == "f6i2c_archive_verification_complete_partial_ready"
    assert inputs.f6i2c_manifest["git_state"]["head"] == EXPECTED_HEAD
    assert len(inputs.f6i2c_archive_ready) == 18
    assert len(
        [row for row in inputs.f6i2c_archive_ready if row["current_decision"] == "archive_ready_after_content_indexing"]
    ) == EXPECTED_CONTENT_CANDIDATES
    assert len(
        [
            row
            for row in inputs.f6i2c_archive_ready
            if row["current_decision"] == "archive_ready_after_current_summary_migration"
        ]
    ) == EXPECTED_SUMMARY_CANDIDATES
    assert len(inputs.f6i2c_cache_bundle) == EXPECTED_CACHE_CANDIDATES


def test_content_index_and_current_fact_gates_pass() -> None:
    result = built_result()
    assert len(result.tables["content_index_candidates"]) == EXPECTED_CONTENT_CANDIDATES
    assert len(result.tables["content_index"]) == EXPECTED_CONTENT_CANDIDATES
    assert all(row["status"] == "pass" for row in result.tables["content_index_validation"])
    assert len(result.tables["current_fact_inventory"]) == EXPECTED_SUMMARY_CANDIDATES
    assert len(result.tables["current_fact_migration"]) == EXPECTED_SUMMARY_CANDIDATES
    assert all(row["status"] == "pass" for row in result.tables["current_fact_validation"])
    assert all("not exact original" not in row["current_wording"].lower() for row in result.tables["current_fact_inventory"])


def test_cache_cleanup_preflight_is_exact_and_restart_safe() -> None:
    result = built_result()
    inputs = load_inputs(CONFIG_PATH)
    approved_paths = {row["path"] for row in inputs.f6i2c_cache_bundle}
    preflight_paths = {row["path"] for row in result.tables["cache_candidate_preflight"]}
    assert preflight_paths == approved_paths
    assert len(result.tables["cache_candidate_preflight"]) == EXPECTED_CACHE_CANDIDATES
    assert result.tables["cache_cleanup_rejected"] == []
    assert all(row["preflight_status"] == "approved_for_exact_cache_cleanup" for row in result.tables["cache_candidate_preflight"])
    assert all(".venv_audit_recovery" not in row["path"] for row in result.tables["cache_candidate_preflight"])
    assert {row["action_status"] for row in result.tables["cache_cleanup_actions"]} <= {
        "not_executed",
        "deleted",
        "already_absent",
    }


def test_final_archive_gate_and_migration_manifest() -> None:
    result = built_result()
    assert len(result.tables["final_archive_gate"]) == 18
    assert {row["final_state"] for row in result.tables["final_archive_gate"]} == {
        "archive_ready_for_f6i2e_with_reference_rewrite"
    }
    assert len(result.tables["archive_ready_for_migration"]) == 18
    assert result.tables["archive_not_ready"] == []
    assert result.summary["decision_token"] == DECISION_TOKEN_COMPLETE
    assert result.summary["next_phase"] == NEXT_PHASE
    assert result.manifests["migration"]["next_archive_migration_phase"] == NEXT_PHASE


def test_reference_and_compatibility_closure_are_concrete() -> None:
    result = built_result()
    refs = result.tables["reference_rewrite_closure"]
    assert len(refs) == 711
    assert sum(row["active_reference_must_change"] == "true" for row in refs) == 15
    assert len({row["original_path"] for row in refs if row["active_reference_must_change"] == "true"}) == 5
    assert all(row["closure_status"].startswith("pass_") for row in refs)
    compatibility = result.tables["compatibility_closure"]
    assert len(compatibility) == 18
    assert all(row["closure_status"] == "pass" for row in compatibility)
    assert all(row["stub_or_import_alias_required"] == "false" for row in compatibility)


def test_validation_and_protected_hashes_pass() -> None:
    result = built_result()
    assert all(row["status"] == "pass" for row in result.tables["validation"])
    assert result.protected_hashes["status"] == "pass"
    assert result.protected_hashes["changed"] == []
    assert result.protected_hashes["missing"] == []
    assert result.summary["target_collisions"] == 0
    assert result.manifests["migration"]["forbidden_actions"]["substantive_candidate_deleted"] is False
    assert result.manifests["migration"]["forbidden_actions"]["training_executed"] is False


def test_in_memory_build_is_deterministic() -> None:
    first = build(
        CONFIG_PATH,
        write_outputs_enabled=False,
        execute_cache_cleanup_enabled=False,
        update_current_state=False,
    )
    second = build(
        CONFIG_PATH,
        write_outputs_enabled=False,
        execute_cache_cleanup_enabled=False,
        update_current_state=False,
    )
    assert semantic_digest(first) == semantic_digest(second)

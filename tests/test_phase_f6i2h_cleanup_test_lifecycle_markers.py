"""Focused checks for F6I2H cleanup test lifecycle markers."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from src.analysis.plan_f6i2h_cleanup_test_lifecycle_markers import (
    APPROVED_INTERPRETER,
    CONFIG_PATH,
    DECISION_COMPLETE,
    F6I2F_DECISION_TOKEN,
    F6I2G_DECISION_TOKEN,
    NEXT_HELPER_GROUP,
    PHASE_ID,
    build,
    semantic_digest,
)


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
    pytest.mark.migration_regression,
    pytest.mark.refactor_regression,
]


@lru_cache(maxsize=1)
def built_result():
    return build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)


def test_runtime_and_authoritative_inputs() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    result = built_result()
    manifest = result.manifests["test_lifecycle_manifest"]
    assert PHASE_ID == "phase_f6i2h"
    assert manifest["previous_phase_decision_tokens"]["phase_f6i2f"] == F6I2F_DECISION_TOKEN
    assert manifest["previous_phase_decision_tokens"]["phase_f6i2g"] == F6I2G_DECISION_TOKEN
    assert result.decision_token == DECISION_COMPLETE


def test_lifecycle_classification_and_suites_are_disjoint() -> None:
    result = built_result()
    lifecycle_rows = result.tables["test_lifecycle_classification"]
    assert lifecycle_rows
    assert not [row for row in lifecycle_rows if row["primary_lifecycle"] == "unresolved"]
    current = {row["nodeid"] for row in result.tables["current_cleanup_suite"]}
    historical = {row["nodeid"] for row in result.tables["historical_cleanup_suite"]}
    assert current
    assert historical
    assert not (current & historical)
    assert len(current) == 31
    assert len(historical) == 35


def test_markers_are_registered_and_assigned() -> None:
    result = built_result()
    assert all(row["status"] == "pass" for row in result.tables["marker_validation"])
    assert all(row["status"] == "pass" for row in result.tables["test_marker_assignment"])
    marker_names = {
        row["marker"]
        for row in result.tables["marker_validation"]
        if row["check_name"].startswith("registered:")
    }
    assert {
        "cleanup_current",
        "cleanup_historical",
        "phase_snapshot",
        "superseded_invariant",
        "migration_regression",
        "refactor_regression",
        "manual_offline",
        "unsafe_external",
    } <= marker_names


def test_superseded_f6i2b_and_migration_meta_validation() -> None:
    result = built_result()
    superseded = result.tables["superseded_invariants"]
    migration = result.tables["migration_supersession_validation"]
    assert len(superseded) == 2
    assert {row["classification"] for row in superseded} == {"superseded_by_authorized_migration"}
    assert all(row["affected_path_count"] == "18" for row in superseded)
    assert len(migration) == 18
    assert all(row["old_path_absent"] == "true" for row in migration)
    assert all(row["archive_path_present"] == "true" for row in migration)
    assert all(row["archive_hash_matches_approved"] == "true" for row in migration)
    assert all(row["status"] == "pass" for row in migration)


def test_f6i2g_refactor_regression_validation() -> None:
    result = built_result()
    rows = result.tables["refactor_regression_validation"]
    assert rows
    assert all(row["status"] == "pass" for row in rows)
    checks = {row["check_name"] for row in rows}
    assert "canonical_module_imports_without_side_effects" in checks
    assert "approved_wrappers_delegate_to_canonical" in checks
    assert "representative_output_bytes_stable" in checks
    assert "protected_scientific_outputs_unchanged" in checks


def test_discovery_safety_and_helper_readiness() -> None:
    result = built_result()
    assert all(row["status"] == "pass" for row in result.tables["test_discovery_validation"])
    helper_rows = result.tables["helper_group_readiness"]
    assert len(helper_rows) == 5
    assert {row["decision"] for row in helper_rows} == {"ready_for_consolidation"}
    selected = [row for row in result.tables["next_helper_bundle"] if row["selected"] == "true"]
    assert len(selected) == 1
    assert selected[0]["group_name"] == NEXT_HELPER_GROUP


def test_validation_and_protected_hashes_pass() -> None:
    result = built_result()
    assert all(row["status"] == "pass" for row in result.tables["validation"])
    assert result.protected_hashes["status"] == "pass"
    assert result.protected_hashes["changed"] == []
    assert result.protected_hashes["missing"] == []
    assert result.manifests["current_cleanup_suite_manifest"]["test_count"] == 31


def test_in_memory_build_is_deterministic() -> None:
    first = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    second = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    assert semantic_digest(first) == semantic_digest(second)

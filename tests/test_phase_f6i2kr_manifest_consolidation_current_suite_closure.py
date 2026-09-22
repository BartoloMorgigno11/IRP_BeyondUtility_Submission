"""Focused checks for F6I2KR current-suite closure."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from src.analysis import plan_f6i2j_git_state_capture_consolidation_execution as f6i2j
from src.analysis.plan_f6i2kr_manifest_consolidation_current_suite_closure import (
    APPROVED_INTERPRETER,
    CONFIG_PATH,
    DECISION_COMPLETE,
    NEXT_HELPER_GROUP,
    PHASE_ID,
    PROJECT_ROOT,
    build,
    semantic_digest,
)


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
    pytest.mark.refactor_regression,
]


@lru_cache(maxsize=1)
def built_result():
    return build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)


def test_runtime_phase_and_decision() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    result = built_result()
    assert PHASE_ID == "phase_f6i2kr"
    assert result.decision_token == DECISION_COMPLETE
    assert result.manifests["current_guard_manifest"]["next_helper_group"] == NEXT_HELPER_GROUP


def test_three_failures_are_classified_as_stale_guard() -> None:
    rows = built_result().tables["failure_classification"]
    assert len(rows) == 3
    assert {row["classification"] for row in rows} == {
        "authorized_new_phase_artifact_not_allowed_by_stale_guard"
    }
    assert {row["genuine_regression"] for row in rows} == {"false"}
    assert {row["stale_guard"] for row in rows} == {"true"}


def test_authorized_delta_reconciliation_is_exact_f6i2k_outputs() -> None:
    rows = built_result().tables["authorized_delta_reconciliation"]
    assert len(rows) == 25
    assert all(row["originating_phase"] == "phase_f6i2k" for row in rows)
    assert all(row["status"] == "pass" for row in rows)
    assert {row["protected_label"] for row in rows} == {"scientific_tables", "scientific_manifests"}


def test_current_guard_contract_is_manifest_derived_and_not_broad() -> None:
    rows = built_result().tables["current_guard_contract"]
    phase_ids = {row["phase_id"] for row in rows}
    assert "phase_f6i2k" in phase_ids
    assert "phase_f6i2kr" in phase_ids
    assert all(row["phase_id"].startswith("phase_f6i2") or row["phase_id"] in {"phase_f6j", "phase_f6l"} for row in rows)
    assert all(row["permitted_table_prefix"] in {f"results/tables/{row['phase_id']}_"} for row in rows)
    assert all(row["permitted_manifest_prefix"] in {f"results/manifests/{row['phase_id']}_"} for row in rows)
    assert all(row["protected_input_exclusion"] == "phase_artifacts_only" for row in rows)


def test_f6i2k_manifest_regression_recheck_passes() -> None:
    rows = built_result().tables["manifest_regression_recheck"]
    assert all(row["status"] == "pass" for row in rows)
    checks = {row["check_name"]: row for row in rows}
    assert checks["f6i2k_function_equivalence"]["observed"] == "9/9"
    assert checks["f6i2k_schema_equivalence"]["observed"] == "8/8"
    assert checks["f6i2k_output_mismatches"]["observed"] == "0"


def test_f6i2j_current_guard_now_allows_registered_phase_artifacts() -> None:
    protected = f6i2j.protected_hashes()
    assert protected["status"] == "pass"
    assert protected["changed"] == []
    assert protected["missing"] == []
    assert "phase_f6i2k" in protected["current_guard_phase_artifact_sources"]


def test_git_status_scope_reconciliation_records_recursive_difference() -> None:
    rows = built_result().tables["git_status_scope_reconciliation"]
    by_name = {row["command_name"]: row for row in rows}
    assert int(by_name["recursive_all_untracked"]["row_count"]) > int(by_name["default_normal_untracked"]["row_count"])
    assert "27331/27370" in by_name["recursive_all_untracked"]["explanation"]
    counts = built_result().tables["git_status_authoritative_counts"]
    assert {row["authoritative_command"] for row in counts} == {
        "git status --porcelain=v1 --untracked-files=normal"
    }


def test_validation_protected_inputs_and_historical_mapping_pass() -> None:
    result = built_result()
    assert all(row["status"] == "pass" for row in result.tables["validation"])
    assert result.manifests["protected_input_hashes"]["status"] == "pass"
    assert all(row["status"] == "pass" for row in result.tables["historical_current_guard_mapping"])


def test_in_memory_build_is_deterministic() -> None:
    first = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    second = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    assert semantic_digest(first) == semantic_digest(second)

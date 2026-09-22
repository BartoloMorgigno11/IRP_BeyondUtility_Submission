"""Focused checks for F6I2P remaining YAML and cleanup closure."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from src.analysis import plan_f6i2j_git_state_capture_consolidation_execution as f6i2j
from src.analysis.plan_f6i2p_remaining_yaml_and_repository_cleanup_closure import (
    APPROVED_INTERPRETER,
    CONFIG_PATH,
    DECISION_SUCCESS,
    NEXT_PHASE,
    PHASE_ID,
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
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    result = built_result()
    assert PHASE_ID == "phase_f6i2p"
    assert result.decision_token == DECISION_SUCCESS
    assert result.manifests["cleanup_closure_manifest"]["next_phase"] == NEXT_PHASE


def test_two_current_suite_failures_are_classified_as_stale_assumptions() -> None:
    rows = built_result().tables["current_suite_failure_closure"]
    assert len(rows) == 2
    assert {row["classification"] for row in rows} == {"stale_current_phase_assumption"}
    assert {row["genuine_regression"] for row in rows} == {"false"}
    assert {row["protected_input_change"] for row in rows} == {"false"}
    assert all(row["status"] == "pass" for row in rows)


def test_remaining_yaml_groups_are_terminal_without_bprmf_or_unresolved() -> None:
    rows = built_result().tables["remaining_yaml_terminal_decisions"]
    states = {row["group_name"]: row["terminal_status"] for row in rows}
    assert states == {
        "base_dataset_or_cost": "retained_future_dependency",
        "model_gru4rec_overrides": "no_change_retained_distinct",
        "model_lightgcn_overrides": "no_change_retained_distinct",
        "model_sasrec_overrides": "no_change_retained_distinct",
        "phase_specific_archive": "retained_historical_evidence",
    }
    assert "model_bprmf_overrides" not in states
    assert all(row["physical_consolidation_performed"] == "false" for row in rows)
    assert all(row["member_yaml_modified"] == "false" for row in rows)
    assert all(row["new_direct_removal_opportunity"] == "false" for row in rows)


def test_resolved_config_equivalence_has_zero_mismatches() -> None:
    rows = built_result().tables["resolved_config_equivalence"]
    assert len(rows) == 23
    assert all(row["status"] == "pass" for row in rows)
    assert all(row["pre_member_sha256"] == row["post_member_sha256"] for row in rows)
    assert all(row["pre_resolved_sha256"] == row["post_resolved_sha256"] for row in rows)


def test_repository_cleanup_closes_with_retain_decision() -> None:
    rows = built_result().tables["repository_cleanup_closure"]
    decision = rows[-1]
    assert decision["cleanup_area"] == "repository_cleanup_decision"
    assert decision["terminal_classification"] == "terminal_retain_current_compatibility_architecture"
    assert all(row["new_direct_removal_opportunity"] == "false" for row in rows)
    manifest = built_result().manifests["cleanup_closure_manifest"]
    assert manifest["wrapper_conclusion"]["wrappers"] == 551
    assert manifest["wrapper_conclusion"]["safe_removal_bundles"] == 0


def test_checkpoint_plan_is_exact_path_only_and_not_staged() -> None:
    rows = built_result().tables["checkpoint_staging_plan"]
    assert rows
    assert all(row["wildcard_staging_allowed"] == "false" for row in rows)
    assert all(row["staging_performed"] == "false" for row in rows)
    assert any(row["path"] == "wildcard_staging_prohibited" for row in rows)
    manifest = built_result().manifests["checkpoint_staging_manifest"]
    assert manifest["staging_performed"] is False
    assert manifest["commit_performed"] is False
    assert manifest["staged_paths"] == []


def test_protected_inputs_and_validation_pass() -> None:
    result = built_result()
    protected = result.manifests["protected_input_hashes"]
    assert protected["status"] == "pass"
    assert protected["changed"] == []
    assert protected["missing"] == []
    assert all(row["status"] == "pass" for row in result.tables["validation"])


def test_current_guard_terminal_registry_includes_f6i2p_decision() -> None:
    assert f6i2j.current_guard_terminal_phase("phase_f6i2p", DECISION_SUCCESS)


def test_in_memory_build_is_deterministic() -> None:
    first = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    second = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    assert semantic_digest(first) == semantic_digest(second)

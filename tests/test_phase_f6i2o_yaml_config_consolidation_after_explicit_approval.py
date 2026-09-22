"""Focused tests for F6I2O YAML config consolidation audit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from src.analysis import plan_f6i2o_yaml_config_consolidation_after_explicit_approval as f6i2o
from src.analysis import plan_f6i2j_git_state_capture_consolidation_execution as f6i2j
from src.analysis import plan_f6i2k_manifest_generation_consolidation_execution as f6i2k
from src.analysis import plan_f6i2l_validation_table_creation_consolidation_execution as f6i2l


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
    pytest.mark.refactor_regression,
]


def test_runtime_contract_uses_approved_recovered_interpreter() -> None:
    assert Path(sys.executable).resolve() == f6i2o.APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"


def test_exact_group_membership_from_f6i2n_and_f6i2a() -> None:
    result = f6i2o.build(write_outputs_enabled=False)
    members = [row["config_path"] for row in result.tables["config_group_members"]]
    assert tuple(members) == f6i2o.MEMBER_CONFIGS
    assert {row["group_name"] for row in result.tables["config_group_members"]} == {"model_bprmf_overrides"}
    assert result.manifests["consolidation_manifest"]["selected_group"] == "model_bprmf_overrides"


def test_required_tables_and_manifests_have_declared_schemas() -> None:
    result = f6i2o.build(write_outputs_enabled=False)
    assert set(result.tables) == set(f6i2o.REQUIRED_TABLES)
    assert set(result.manifests) == set(f6i2o.REQUIRED_MANIFESTS)
    for name, rows in result.tables.items():
        assert rows, name
        expected = set(f6i2o.TABLE_COLUMNS[name])
        assert all(set(row) <= expected for row in rows), name
        assert expected.issuperset(rows[0]), name


def test_no_change_strategy_preserves_member_hashes_and_blocks_physical_factoring() -> None:
    result = f6i2o.build(write_outputs_enabled=False)
    selected = [row for row in result.tables["consolidation_strategy"] if row["selected"] == "true"]
    assert len(selected) == 1
    assert selected[0]["strategy_id"] == "D"
    assert selected[0]["decision_if_selected"] == f6i2o.DECISION_NO_CHANGE
    assert {row["change_applied"] for row in result.tables["config_actions"] if row["target_path"] in f6i2o.MEMBER_CONFIGS} == {"false"}
    assert all(row["byte_identical"] == "true" for row in result.tables["historical_config_validation"])


def test_resolved_config_equivalence_is_exact_for_all_members() -> None:
    result = f6i2o.build(write_outputs_enabled=False)
    equivalence = result.tables["resolved_config_equivalence"]
    assert len(equivalence) == len(f6i2o.MEMBER_CONFIGS)
    assert {row["status"] for row in equivalence} == {"pass"}
    assert {row["equivalence_classification"] for row in equivalence} == {"exact_same_document_no_change"}


def test_duplicate_detection_and_loader_contract_are_conservative() -> None:
    result = f6i2o.build(write_outputs_enabled=False)
    syntax_rows = [row for row in result.tables["duplicate_key_analysis"] if row["scope"] == "within_yaml"]
    assert len(syntax_rows) == len(f6i2o.MEMBER_CONFIGS)
    assert {row["classification"] for row in syntax_rows} == {"no_duplicate_yaml_keys"}
    loaders = result.tables["loader_contract_inventory"]
    assert any(row["override_support"] == "none" for row in loaders)
    assert all(row["env_interpolation_support"] in {"none", "not_applicable"} for row in loaders)


def test_output_equivalence_does_not_execute_training() -> None:
    result = f6i2o.build(write_outputs_enabled=False)
    assert result.tables["output_equivalence"]
    assert {row["execution_performed"] for row in result.tables["output_equivalence"]} == {"false"}
    assert result.manifests["consolidation_manifest"]["training_performed"] is False
    assert result.manifests["consolidation_manifest"]["llm_inference_performed"] is False


def test_remaining_readiness_selects_one_next_yaml_group() -> None:
    result = f6i2o.build(write_outputs_enabled=False)
    selected = [row for row in result.tables["remaining_yaml_group_readiness"] if row["selected_next_yaml_group"] == "true"]
    assert len(selected) == 1
    assert selected[0]["group_name"] == f6i2o.NEXT_YAML_GROUP


def test_build_is_semantically_deterministic() -> None:
    first = f6i2o.build(write_outputs_enabled=False)
    second = f6i2o.build(write_outputs_enabled=False)
    assert f6i2o.semantic_digest(first) == f6i2o.semantic_digest(second)


def test_decision_token_is_no_change_retained_distinct() -> None:
    result = f6i2o.build(write_outputs_enabled=False)
    assert result.decision_token == f6i2o.DECISION_NO_CHANGE
    assert result.validation_passed


def test_no_change_decision_is_registered_as_terminal_current_guard_phase() -> None:
    phase_id = f6i2o.PHASE_ID
    decision = f6i2o.DECISION_NO_CHANGE
    assert f6i2j.current_guard_terminal_phase(phase_id, decision)
    assert f6i2k.current_guard_terminal_later_phase(phase_id, decision)
    assert f6i2l.current_guard_terminal_later_phase(phase_id, decision)

from __future__ import annotations

import importlib
import os
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from src.analysis.plan_f6i2n_repository_cleanup_after_explicit_approval import (
    APPROVED_INTERPRETER,
    CANONICAL_MODULES,
    CONFIG_PATH,
    DECISION_CURRENT_SUITE_FAILED,
    DECISION_NO_SAFE_REMOVAL,
    NEXT_PHASE,
    PHASE_ID,
    build,
    has_canonical_cycle,
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
    result = built_result()
    manifest = result.manifests["wrapper_cleanup_readiness_manifest"]
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert result.decision_token in {DECISION_NO_SAFE_REMOVAL, DECISION_CURRENT_SUITE_FAILED}
    assert manifest["phase_id"] == PHASE_ID
    assert manifest["runtime"]["runtime_classification"] == "defensible_repaired_clone_exact_package_versions"
    assert manifest["not_exact_original_runtime"] is True


def test_six_canonical_modules_present_importable_and_clean() -> None:
    result = built_result()
    rows = result.tables["canonical_module_inventory"]
    assert [row["path"] for row in rows] == [module["path"] for module in CANONICAL_MODULES]
    assert {row["status"] for row in rows} == {"pass"}
    for module in CANONICAL_MODULES:
        imported = importlib.import_module(module["module"])
        assert imported is not None
    assert {row["imports_phase_specific_builders"] for row in rows} == {"false"}
    assert {row["absorbed_scientific_protocol"] for row in rows} == {"false"}


def test_canonical_dependency_graph_has_no_cycle() -> None:
    result = built_result()
    rows = result.tables["canonical_dependency_graph"]
    assert rows
    assert {row["status"] for row in rows} == {"pass"}
    assert not has_canonical_cycle(rows)


def test_wrapper_inventory_completeness_and_group_counts() -> None:
    result = built_result()
    rows = result.tables["wrapper_inventory"]
    assert len(rows) == 551
    assert {row["status"] for row in rows} == {"pass"}
    assert Counter(row["consolidation_group"] for row in rows) == {
        "deterministic_csv_io": 33,
        "git_state_capture": 63,
        "hashing_helpers": 285,
        "manifest_generation": 156,
        "path_normalization": 7,
        "validation_table_creation": 7,
    }
    assert Counter(row["wrapper_type"] for row in rows) == {
        "direct_delegating_wrapper": 78,
        "partial_delegating_wrapper": 8,
        "phase_specific_adapter": 463,
        "schema_adapter": 2,
    }


def test_wrapper_target_resolution_and_consumer_graph() -> None:
    result = built_result()
    inventory = result.tables["wrapper_inventory"]
    edges = result.tables["wrapper_consumer_edges"]
    assert all(row["target_resolution_status"] == "resolved" for row in inventory)
    assert len(edges) == 18480
    assert {row["status"] for row in edges} == {"pass"}
    assert all(not row["consumer_file"].startswith("archive/") for row in edges)
    assert all(not row["consumer_file"].startswith(".venv") for row in edges)
    assert any(row["active_executable_reference"] == "true" for row in edges)
    assert any(row["edge_type"] == "dynamic_or_string_reference" for row in edges)


def test_wrapper_classification_counts_and_dynamic_references() -> None:
    result = built_result()
    rows = result.tables["wrapper_classification"]
    assert Counter(row["primary_status"] for row in rows) == {
        "retain_active_internal_consumers": 44,
        "retain_cli_or_entrypoint": 433,
        "retain_historical_schema": 1,
        "retain_signature_adapter": 73,
    }
    assert sum(1 for row in rows if int(row["active_consumer_count"]) > 0) == 440
    assert sum(1 for row in rows if int(row["dynamic_reference_count"]) > 0) == 458
    assert all(row["primary_status"] != "unresolved" for row in rows)
    dynamic_rows = result.tables["dynamic_reference_audit"]
    assert {row["status"] for row in dynamic_rows} == {"pass"}
    assert all(
        (row["classification"] == "resolved_dynamic_or_string_reference") == (int(row["dynamic_reference_count"]) > 0)
        for row in dynamic_rows
    )


def test_function_file_and_archive_removal_gates_are_closed() -> None:
    result = built_result()
    assert all(row["function_level_removal_status"] == "retention_required" for row in result.tables["function_removal_readiness"])
    assert all(row["file_level_removal_status"] == "not_candidate" for row in result.tables["file_removal_readiness"])
    assert all(row["candidate_status"] == "not_selected" for row in result.tables["consumer_migration_contracts"])
    assert not any(row["primary_status"] == "candidate_for_archive_after_migration" for row in result.tables["wrapper_classification"])
    bundle = result.tables["first_removal_bundle"][0]
    assert bundle["safe_removal_bundle_exists"] == "false"
    assert bundle["selected"] == "false"
    assert bundle["selected_alternative"] == "YAML/config consolidation"
    assert bundle["next_phase"] == NEXT_PHASE


def test_wrapper_test_coverage_counts() -> None:
    result = built_result()
    assert Counter(row["coverage_classification"] for row in result.tables["wrapper_test_coverage"]) == {
        "historical_only": 209,
        "indirectly_covered": 84,
        "partially_covered": 258,
    }


def test_residual_duplication_audit_is_classified() -> None:
    result = built_result()
    rows = result.tables["residual_duplication_audit"]
    assert len(rows) == 936
    assert Counter(row["classification"] for row in rows) == {
        "approved_wrapper": 555,
        "incompatible_distinct_contract": 140,
        "phase_specific_adapter": 213,
        "scientific_logic": 28,
    }
    assert not any(row["classification"] == "missed_safe_consolidation_candidate" for row in rows)


def test_yaml_config_readiness_reconciles_prior_evidence() -> None:
    result = built_result()
    rows = result.tables["config_readiness"]
    selected = [row for row in rows if row["selected_next_yaml_group"] == "true"]
    assert len(rows) == 6
    assert len(selected) == 1
    assert selected[0]["group_name"] == "model_bprmf_overrides"
    assert selected[0]["readiness"] == "safe_first_yaml_group"
    assert {row["f6i2a_config_audit_rows"] for row in rows} == {"27"}
    assert {row["f6i2a_merge_groups"] for row in rows} == {"6"}
    assert {row["f6i2c_redirected_configurations"] for row in rows} == {"9"}


def test_dependency_protected_and_git_validation_pass() -> None:
    result = built_result()
    assert {row["status"] for row in result.tables["dependency_graph_validation"]} == {"pass"}
    assert result.manifests["protected_input_hashes"]["status"] == "pass"
    git_guard = result.manifests["git_metadata_guard"]
    assert git_guard["status"] == "pass"
    assert git_guard["checks"]["branch_unchanged"]
    assert git_guard["checks"]["head_unchanged"]
    assert git_guard["checks"]["index_unchanged"]
    assert git_guard["checks"]["staged_paths_unchanged"]
    assert git_guard["checks"]["no_new_staged_paths_introduced_by_f6i2n"]


def test_current_suite_failure_is_recorded_without_masking_wrapper_readiness() -> None:
    result = built_result()
    suite_rows = {row["suite"]: row for row in result.tables["current_suite_validation"]}
    validation_rows = {row["check_name"]: row for row in result.tables["validation"]}
    if result.decision_token == DECISION_CURRENT_SUITE_FAILED:
        assert suite_rows["current_cleanup_suite"]["status"] == "fail"
        assert suite_rows["f6i2m_hashing_regression"]["status"] == "fail"
        assert validation_rows["current_suite"]["status"] == "fail"
        assert result.tables["first_removal_bundle"][0]["safe_removal_bundle_exists"] == "false"
        assert result.tables["next_actions"][0]["recommendation"] == "stop_and_investigate_current_suite_failure"
    else:
        assert {row["status"] for row in result.tables["current_suite_validation"]} == {"pass"}
        assert {row["status"] for row in result.tables["validation"]} == {"pass"}


def test_deterministic_in_memory_outputs() -> None:
    first = built_result()
    second = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    assert semantic_digest(first) == semantic_digest(second)

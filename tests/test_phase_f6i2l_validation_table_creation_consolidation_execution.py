"""Focused checks for F6I2L validation-table consolidation."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from src.audit import common_validation
from src.analysis.plan_f6i2l_validation_table_creation_consolidation_execution import (
    APPROVED_INTERPRETER,
    CANONICAL_APIS,
    CONFIG_PATH,
    DECISION_COMPLETE,
    MIGRATED_WRAPPERS,
    NEXT_HELPER_GROUP,
    OUTPUT_TABLES,
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
    manifest = result.manifests["consolidation_manifest"]
    assert PHASE_ID == "phase_f6i2l"
    assert manifest["previous_phase_decision_token"] == "f6i2kr_current_suite_closure_complete"
    assert manifest["selected_group"] == "validation_table_creation"
    assert manifest["runtime"]["runtime_classification"] == "defensible_repaired_clone_exact_package_versions"
    assert manifest["not_exact_original_runtime"] is True
    assert result.decision_token == DECISION_COMPLETE


def test_common_validation_api_contract_and_import_safety(tmp_path: Path) -> None:
    assert set(CANONICAL_APIS) >= {
        "status_from_outcome",
        "make_validation_row",
        "build_validation_table",
        "summarize_validation_statuses",
        "validation_passed",
        "write_validation_table",
    }
    assert common_validation.status_from_outcome(True) == "pass"
    assert common_validation.status_from_outcome(False, fail_status="failed") == "failed"
    assert common_validation.status_from_outcome(None, none_status="not_applicable") == "not_applicable"
    with pytest.raises(ValueError):
        common_validation.status_from_outcome(None)

    row = common_validation.make_validation_row(
        columns=["check_id", "status", "details"],
        values={"status": "warning", "check_id": "c1"},
    )
    assert list(row) == ["check_id", "status", "details"]
    assert row["details"] == ""
    rows = common_validation.build_validation_table([row], columns=["check_id", "status"])
    assert rows == [{"check_id": "c1", "status": "warning"}]
    assert common_validation.summarize_validation_statuses(rows)["warning"] == 1
    assert common_validation.validation_passed([], allow_empty=True)
    with pytest.raises(ValueError):
        common_validation.validation_passed(rows)

    path = tmp_path / "validation.csv"
    common_validation.write_validation_table(path, rows, ["check_id", "status"], lineterminator="\n")
    assert path.read_text(encoding="utf-8") == "check_id,status\nc1,warning\n"


def test_migrated_wrappers_have_exact_function_schema_and_output_equivalence() -> None:
    result = built_result()
    assert len(MIGRATED_WRAPPERS) == 5
    assert all(row["status"] == "pass" for row in result.tables["function_equivalence"])
    assert all(row["status"] == "pass" for row in result.tables["schema_equivalence"])
    assert all(row["output_mismatch"] == "false" for row in result.tables["output_hash_comparison"])
    assert {row["function_name"] for row in result.tables["function_equivalence"]} == {
        "validation_row",
        "blocked_validation_row",
        "_validation_row",
    }


def test_status_vocabulary_preserves_distinct_semantics() -> None:
    rows = built_result().tables["status_vocabulary"]
    by_status = {row["status_value"]: row for row in rows}
    assert set(by_status) == {"pass", "fail", "passed", "failed", "blocked", "not_applicable"}
    assert by_status["blocked"]["counted_as_failure"] == "true"
    assert by_status["not_applicable"]["counted_as_passing"] == "true"
    assert by_status["fail"]["counted_as_passing"] == "false"


def test_schema_families_and_failure_policy_are_not_globalized() -> None:
    result = built_result()
    families = {row["schema_family"] for row in result.tables["schema_family_inventory"]}
    assert "metric_blocked_validation" in families
    assert "llm_top10_preflight_validation" in families
    assert all(row["status"] == "pass" for row in result.tables["failure_policy_equivalence"])
    policies = {row["check_name"]: row for row in result.tables["failure_policy_equivalence"]}
    assert policies["decision_tokens_phase_specific"]["observed"] == "true"


def test_canonical_module_does_not_embed_hashing_or_phase_logic() -> None:
    text = (PROJECT_ROOT / "src/audit/common_validation.py").read_text(encoding="utf-8")
    assert "hashlib" not in text
    assert "sha256" not in text.lower()
    assert "phase_f6" not in text
    assert "Qwen2" not in text
    assert "Amazon Beauty" not in text


def test_dependency_graph_future_and_wrapper_readiness_pass() -> None:
    result = built_result()
    graph = result.tables["dependency_graph_comparison"][0]
    assert graph["status"] == "pass"
    assert graph["circular_imports"] == ""
    assert int(graph["after_common_validation_importers"]) >= 4
    assert all(row["status"] == "pass" for row in result.tables["future_dependency_validation"])
    assert all(row["expected_file_reduction_now"] == "0" for row in result.tables["wrapper_removal_readiness"])


def test_validation_protected_inputs_and_git_metadata_pass() -> None:
    result = built_result()
    assert set(result.tables) == set(OUTPUT_TABLES)
    assert all(row["status"] == "pass" for row in result.tables["validation"])
    assert result.protected_hashes["status"] == "pass"
    assert result.protected_hashes["changed"] == []
    assert result.git_metadata_guard["status"] == "pass"
    assert result.manifests["consolidation_manifest"]["next_helper_group"] == NEXT_HELPER_GROUP


def test_source_modification_scope_is_narrow() -> None:
    rows = built_result().tables["source_modifications"]
    modified = {row["source_file"] for row in rows}
    assert "src/audit/common_validation.py" in modified
    assert "src/llm/canonical_local_llm_full_matrix_runner.py" not in modified
    assert "src/llm/canonical_local_llm_top10_confirmation_runner.py" not in modified
    assert all(row["status"] == "pass" for row in rows)


def test_in_memory_build_is_deterministic() -> None:
    first = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    second = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    assert semantic_digest(first) == semantic_digest(second)

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from src.analysis.plan_f6i2nr_git_protected_hash_current_suite_closure import (
    APPROVED_INTERPRETER,
    CONFIG_PATH,
    DECISION_COMPLETE,
    EXPECTED_BRANCH,
    EXPECTED_HEAD,
    NEXT_PHASE,
    PHASE_ID,
    RUNTIME_CLASSIFICATION,
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
    return build(
        CONFIG_PATH,
        write_outputs_enabled=False,
        update_current_state=False,
        strict_current_suite_results=False,
    )


def test_runtime_phase_and_decision() -> None:
    result = built_result()
    manifest = result.manifests["current_git_guard_manifest"]
    runtime = manifest["runtime"]
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert result.decision_token == DECISION_COMPLETE
    assert manifest["phase_id"] == PHASE_ID
    assert runtime["runtime_classification"] == RUNTIME_CLASSIFICATION
    assert runtime["not_exact_original_runtime"] is True


def test_original_f6i2m_failures_are_inventoried() -> None:
    rows = built_result().tables["original_failure_inventory"]
    assert [row["test_function"] for row in rows] == [
        "test_runtime_phase_and_decision",
        "test_real_hash_and_protected_guard_equivalence",
        "test_validation_and_git_metadata_pass",
    ]
    assert {row["status"] for row in rows} == {"pass"}
    assert {row["subject"] for row in rows} == {"historical F6I2M recursive .git protected hash"}


def test_git_tree_diff_records_aggregate_mismatch_and_inventory_limit() -> None:
    rows = built_result().tables["git_tree_file_diff"]
    aggregate = rows[0]
    assert aggregate["git_relative_path"] == "__historical_f6i2m_recursive_git_digest__"
    assert aggregate["historical_sha256"] != aggregate["current_sha256"]
    assert aggregate["change_class"] == "historical_inventory_incomplete"
    assert aggregate["possible_producing_command"] == "not attributable from aggregate digest alone"
    assert any(row["git_relative_path"] == "index" for row in rows)
    assert {row["status"] for row in rows} == {"pass"}


def test_semantic_git_guard_protects_current_repository_identity() -> None:
    guard = built_result().manifests["git_metadata_guard"]
    assert guard["status"] == "pass"
    assert guard["after"]["branch"] == EXPECTED_BRANCH
    assert guard["after"]["head"] == EXPECTED_HEAD
    assert guard["before"]["index_sha256"] == guard["after"]["index_sha256"]
    assert guard["before"]["refs_sha256"] == guard["after"]["refs_sha256"]
    assert guard["before"]["config_sha256"] == guard["after"]["config_sha256"]
    assert guard["before"]["head_file_sha256"] == guard["after"]["head_file_sha256"]
    assert guard["after"]["locks"] == []
    assert all(guard["checks"].values())


def test_staged_state_is_derived_from_git_index_not_trimmed_porcelain() -> None:
    result = built_result()
    guard = result.manifests["git_metadata_guard"]
    rows = {row["check_name"]: row for row in result.tables["staged_state_validation"]}
    assert guard["before"]["staged"]["staged_paths"] == guard["after"]["staged"]["staged_paths"]
    assert rows["staged_path_set_unchanged"]["status"] == "pass"
    assert rows["staged_content_identity_unchanged"]["status"] == "pass"
    assert rows["codex_prompts_index_matches_head"]["status"] == "pass"
    assert rows["f6i2n_reported_staged_paths_preserved_as_historical_note"]["status"] == "pass"


def test_git_metadata_classification_has_no_unknown_stable_failure() -> None:
    rows = built_result().tables["git_metadata_classification"]
    stable_rows = [row for row in rows if row["stability_classification"] == "stable_and_protected"]
    assert stable_rows
    assert {row["status"] for row in stable_rows} == {"pass"}
    assert any(row["git_metadata_component"] == "loose_objects" for row in rows)
    assert any(row["change_class_if_changed"] == "historical_inventory_incomplete" for row in rows)


def test_f6i2n_readiness_revalidation_adopts_only_green_evidence() -> None:
    rows = {row["check_name"]: row for row in built_result().tables["f6i2n_readiness_revalidation"]}
    assert rows["wrappers_inventoried"]["observed"] == "551"
    assert rows["unresolved_dynamic_references"]["observed"] == "0"
    assert rows["function_removal_candidates"]["observed"] == "0"
    assert rows["file_removal_candidates"]["observed"] == "0"
    assert rows["archive_candidates"]["observed"] == "0"
    assert rows["safe_removal_bundles"]["observed"] == "0"
    assert rows["selected_config_group"]["observed"] == "model_bprmf_overrides"
    assert rows["determinism_mismatch_count"]["observed"] == "0"
    assert rows["protected_inputs"]["observed"] == "pass"
    assert {row["status"] for row in rows.values()} == {"pass"}


def test_historical_and_current_guard_roles_are_separated() -> None:
    rows = built_result().tables["historical_current_guard_mapping"]
    roles = {row["current_guard_role"] for row in rows}
    assert "not a current-suite guard after F6I2NR" in roles
    assert "current-suite authority" in roles
    assert {row["status"] for row in rows} == {"pass"}


def test_current_suite_and_validation_are_green() -> None:
    result = built_result()
    assert {row["status"] for row in result.tables["current_suite_results"]} == {"pass"}
    assert {row["status"] for row in result.tables["validation"]} == {"pass"}
    assert result.tables["next_actions"][0]["recommendation"] == NEXT_PHASE


def test_protected_inputs_and_runtime_provenance() -> None:
    protected = built_result().manifests["protected_input_hashes"]
    assert protected["status"] == "pass"
    assert protected["changed"] == []
    assert protected["runtime"]["runtime_classification"] == RUNTIME_CLASSIFICATION
    assert protected["runtime"]["not_exact_original_runtime"] is True


def test_deterministic_in_memory_build() -> None:
    first = built_result()
    second = build(
        CONFIG_PATH,
        write_outputs_enabled=False,
        update_current_state=False,
        strict_current_suite_results=False,
    )
    assert semantic_digest(first) == semantic_digest(second)

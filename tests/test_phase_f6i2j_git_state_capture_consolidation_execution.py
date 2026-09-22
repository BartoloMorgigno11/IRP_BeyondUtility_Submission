"""Focused checks for F6I2J Git state-capture consolidation."""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from src.analysis.plan_f6i2j_git_state_capture_consolidation_execution import (
    APPROVED_INTERPRETER,
    CONFIG_PATH,
    DECISION_COMPLETE,
    F6I2I_DECISION_TOKEN,
    PHASE_ID,
    PROJECT_ROOT,
    build,
    semantic_digest,
)
from src.audit import common_git


pytestmark = [
    pytest.mark.cleanup_historical,
    pytest.mark.phase_snapshot,
    pytest.mark.refactor_regression,
]


@lru_cache(maxsize=1)
def built_result():
    return build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)


def test_runtime_and_authoritative_contracts() -> None:
    assert Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    result = built_result()
    manifest = result.manifests["consolidation_manifest"]
    assert PHASE_ID == "phase_f6i2j"
    assert manifest["previous_phase_decision_token"] == F6I2I_DECISION_TOKEN
    assert manifest["selected_group"] == "git_state_capture"
    assert manifest["runtime"]["runtime_classification"] == "defensible_repaired_clone_exact_package_versions"
    assert manifest["not_exact_original_runtime"] is True
    assert result.decision_token == DECISION_COMPLETE


def test_common_git_enforces_read_only_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    assert common_git.normalize_git_args(["git", "status", "--porcelain=v1"]) == ["status", "--porcelain=v1"]
    assert common_git.normalize_git_args(["branch", "--show-current"]) == ["branch", "--show-current"]
    with pytest.raises(common_git.GitReadOnlyError):
        common_git.normalize_git_args(["add", "file.txt"])
    with pytest.raises(common_git.GitReadOnlyError):
        common_git.normalize_git_args(["branch", "-D", "old"])
    with pytest.raises(common_git.GitReadOnlyError):
        common_git.normalize_git_args(["config", "user.name", "bad"])
    with pytest.raises(common_git.GitReadOnlyError):
        common_git.normalize_git_args(["diff", "--output=written.txt"])

    calls = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    completed = common_git.run_git_readonly(["status", "--porcelain=v1"], PROJECT_ROOT, check=False)
    assert completed.stdout == "ok\n"
    assert calls[0][0] == ["git", "status", "--porcelain=v1"]
    assert calls[0][1]["cwd"] == PROJECT_ROOT
    assert calls[0][1]["shell"] is False


def test_common_git_parsers_preserve_legacy_status_semantics() -> None:
    text = " M tracked.txt\nA  staged.txt\n D deleted.txt\n?? untracked.txt\nR  old.txt -> new.txt\n"
    lines = common_git.git_status_lines(text)
    assert lines == [
        " M tracked.txt",
        "A  staged.txt",
        " D deleted.txt",
        "?? untracked.txt",
        "R  old.txt -> new.txt",
    ]
    assert common_git.git_status_path(lines[0]) == "tracked.txt"
    assert common_git.git_status_counts(lines) == {
        "modified": 1,
        "added": 1,
        "deleted": 1,
        "untracked": 1,
        "other": 1,
    }


def test_fixture_matrix_covers_required_git_states() -> None:
    result = built_result()
    fixtures = {row["fixture_name"]: row for row in result.tables["git_fixture_matrix"]}
    assert {
        "clean_repository",
        "modified_tracked_file",
        "staged_file",
        "untracked_file",
        "deleted_tracked_file",
        "staged_rename",
        "ignored_file",
        "unborn_repository",
        "detached_head",
        "paths_with_spaces_and_unicode",
        "outside_repository",
        "missing_git_mocked",
    } <= set(fixtures)
    assert fixtures["outside_repository"]["capture_status"] == "pass"
    assert fixtures["missing_git_mocked"]["setup_scope"] == "mock_only_no_project_mutation"
    assert all(row["capture_status"] == "pass" for row in fixtures.values())


def test_wrappers_retained_delegate_and_remove_duplicate_git_bodies() -> None:
    result = built_result()
    wrappers = result.tables["wrapper_compatibility"]
    assert len(wrappers) == result.manifests["consolidation_manifest"]["counts"]["non_test_wrappers"]
    assert len(wrappers) == 63
    assert all(row["signature_preserved"] == "true" for row in wrappers)
    assert all(row["wrapper_retained"] == "true" for row in wrappers)
    assert all(row["delegates_to_canonical"] == "true" for row in wrappers)
    assert all(row["duplicate_git_body_removed"] == "true" for row in wrappers)
    assert all(row["status"] == "pass" for row in wrappers)
    assert result.manifests["consolidation_manifest"]["counts"]["test_definitions_not_migrated"] == 6


def test_function_equivalence_output_hashes_and_builders_are_classified() -> None:
    result = built_result()
    fixture_count = len(result.tables["git_fixture_matrix"])
    wrapper_count = len(result.tables["wrapper_compatibility"])
    assert len(result.tables["function_equivalence"]) == fixture_count * wrapper_count
    assert all(row["status"] == "pass" for row in result.tables["function_equivalence"])
    assert all(row["output_mismatch"] == "false" for row in result.tables["output_hash_comparison"])
    classes = {row["classification"] for row in result.tables["builder_equivalence"]}
    assert "unsafe_requires_training" in classes
    assert "unsafe_requires_inference" in classes
    assert "validation_by_frozen_output" in classes


def test_git_metadata_protected_inputs_and_dependency_graph_pass() -> None:
    result = built_result()
    assert result.manifests["git_metadata_guard"]["status"] == "pass"
    assert all(row["status"] == "pass" for row in result.tables["project_git_immutability"])
    assert result.protected_hashes["status"] == "pass"
    assert result.protected_hashes["changed"] == []
    assert result.protected_hashes["missing"] == []
    graph = result.tables["dependency_graph_comparison"]
    assert graph[-1]["status"] == "pass"
    assert int(graph[-1]["common_git_importer_count"]) >= 40
    assert graph[-1]["circular_imports"] == ""


def test_current_project_read_only_capture_does_not_change_git_metadata() -> None:
    git_dir = PROJECT_ROOT / ".git"
    before_index = git_dir.joinpath("index").read_bytes()
    before_head = common_git.git_stdout_no_check(["rev-parse", "HEAD"], PROJECT_ROOT)
    status = common_git.git_stdout_no_check(["status", "--porcelain=v1"], PROJECT_ROOT, strip=False)
    after_head = common_git.git_stdout_no_check(["rev-parse", "HEAD"], PROJECT_ROOT)
    after_index = git_dir.joinpath("index").read_bytes()
    assert before_head == after_head
    assert before_index == after_index
    assert isinstance(status, str)


def test_builder_outputs_and_future_readiness_are_conservative() -> None:
    result = built_result()
    assert all(row["status"] == "pass" for row in result.tables["validation"])
    assert all(row["status"] == "pass" for row in result.tables["future_dependency_validation"])
    assert {row["wrapper_removal_status"] for row in result.tables["wrapper_removal_readiness"]} == {
        "retain_phase_provenance"
    }
    assert result.manifests["consolidation_manifest"]["counts"]["files_deleted"] == 0
    assert result.manifests["consolidation_manifest"]["next_helper_group"] == "manifest_generation"


def test_in_memory_build_is_deterministic() -> None:
    first = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    second = build(CONFIG_PATH, write_outputs_enabled=False, update_current_state=False)
    assert semantic_digest(first) == semantic_digest(second)

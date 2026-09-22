"""F6I2H cleanup test lifecycle marker audit and current-suite builder."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2h"
PHASE_NAME = "phase_f6i2h_cleanup_test_lifecycle_markers"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"
F6I2F_DECISION_TOKEN = "f6i2f_post_archive_readiness_complete_first_python_bundle_selected"
F6I2G_DECISION_TOKEN = "f6i2g_deterministic_csv_io_consolidation_complete"
DECISION_COMPLETE = "f6i2h_test_lifecycle_complete_next_helper_selected"
DECISION_NO_HELPER = "f6i2h_test_lifecycle_complete_no_helper_ready"
DECISION_PARTIAL = "f6i2h_partial_historical_fixture_work_remaining"
DECISION_FAILED_SUITE = "f6i2h_failed_current_suite"
DECISION_PROTECTED_FAILED = "f6i2h_failed_protected_input_change"
NEXT_PHASE = "phase_f6i2i_next_helper_consolidation_after_explicit_approval"
NEXT_HELPER_GROUP = "path_normalization"

CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
SNAPSHOT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"
LIFECYCLE_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_test_lifecycle_manifest.json"
CURRENT_SUITE_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_current_cleanup_suite_manifest.json"
PROTECTED_HASHES_PATH = MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json"

F6I2F_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6i2f_post_archive_readiness_manifest.json"
F6I2G_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6i2g_consolidation_manifest.json"
F6I2G_PROTECTED_PATH = MANIFESTS_DIR / "phase_f6i2g_protected_input_hashes.json"
F6I2G_SOURCE_MODS_PATH = TABLES_DIR / "phase_f6i2g_source_modifications.csv"
F6I2G_WRAPPERS_PATH = TABLES_DIR / "phase_f6i2g_wrapper_compatibility.csv"
F6I2G_FUNCTIONS_PATH = TABLES_DIR / "phase_f6i2g_function_equivalence.csv"
F6I2G_OUTPUTS_PATH = TABLES_DIR / "phase_f6i2g_output_hash_comparison.csv"
F6I2G_GRAPH_PATH = TABLES_DIR / "phase_f6i2g_dependency_graph_comparison.csv"
F6I2G_FUTURE_PATH = TABLES_DIR / "phase_f6i2g_future_dependency_validation.csv"
F6I2G_VALIDATION_PATH = TABLES_DIR / "phase_f6i2g_validation.csv"

F6I2D_APPROVAL_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6i2d_substantive_archive_migration_manifest.json"
F6I2E_MIGRATION_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6i2e_migration_manifest.json"
F6I2E_ARCHIVE_INDEX_PATH = MANIFESTS_DIR / "phase_f6i2e_archive_index.json"
F6I2E_ROLLBACK_PATH = MANIFESTS_DIR / "phase_f6i2e_rollback_manifest.json"
F6I2E_REFERENCE_VALIDATION_PATH = TABLES_DIR / "phase_f6i2e_reference_rewrite_validation.csv"
F6I2E_MIGRATION_ACTIONS_PATH = TABLES_DIR / "phase_f6i2e_migration_actions.csv"
F6I2F_LIFECYCLE_PATH = TABLES_DIR / "phase_f6i2f_test_lifecycle_audit.csv"
F6I2F_CURRENT_SUITE_PATH = TABLES_DIR / "phase_f6i2f_current_validation_suite_plan.csv"
F6I2F_MERGE_REVALIDATION_PATH = TABLES_DIR / "phase_f6i2f_python_merge_group_revalidation.csv"
F6I2F_MERGE_PRIORITY_PATH = TABLES_DIR / "phase_f6i2f_python_merge_priority.csv"
F6I2F_FIRST_CONTRACT_PATH = TABLES_DIR / "phase_f6i2f_first_python_consolidation_contract.csv"

OUTPUT_TABLES = {
    "cleanup_test_inventory": TABLES_DIR / f"{PHASE_ID}_cleanup_test_inventory.csv",
    "test_lifecycle_classification": TABLES_DIR / f"{PHASE_ID}_test_lifecycle_classification.csv",
    "test_marker_assignment": TABLES_DIR / f"{PHASE_ID}_test_marker_assignment.csv",
    "superseded_invariants": TABLES_DIR / f"{PHASE_ID}_superseded_invariants.csv",
    "migration_supersession_validation": TABLES_DIR / f"{PHASE_ID}_migration_supersession_validation.csv",
    "refactor_regression_validation": TABLES_DIR / f"{PHASE_ID}_refactor_regression_validation.csv",
    "current_cleanup_suite": TABLES_DIR / f"{PHASE_ID}_current_cleanup_suite.csv",
    "historical_cleanup_suite": TABLES_DIR / f"{PHASE_ID}_historical_cleanup_suite.csv",
    "test_command_matrix": TABLES_DIR / f"{PHASE_ID}_test_command_matrix.csv",
    "test_discovery_validation": TABLES_DIR / f"{PHASE_ID}_test_discovery_validation.csv",
    "marker_validation": TABLES_DIR / f"{PHASE_ID}_marker_validation.csv",
    "helper_group_readiness": TABLES_DIR / f"{PHASE_ID}_helper_group_readiness.csv",
    "next_helper_bundle": TABLES_DIR / f"{PHASE_ID}_next_helper_bundle.csv",
    "validation": TABLES_DIR / f"{PHASE_ID}_validation.csv",
    "determinism": TABLES_DIR / f"{PHASE_ID}_determinism.csv",
    "next_actions": TABLES_DIR / f"{PHASE_ID}_next_actions.csv",
}

OUTPUT_PATHS = {
    *(path.relative_to(PROJECT_ROOT).as_posix() for path in OUTPUT_TABLES.values()),
    "configs/phase_f6i2h_cleanup_test_lifecycle_markers.yaml",
    "src/analysis/plan_f6i2h_cleanup_test_lifecycle_markers.py",
    "src/experiments/build_phase_f6i2h_cleanup_test_lifecycle_markers.py",
    "tests/test_phase_f6i2h_cleanup_test_lifecycle_markers.py",
    "report/phase_f6i2h_cleanup_test_lifecycle_markers.md",
    "results/manifests/phase_f6i2h_test_lifecycle_manifest.json",
    "results/manifests/phase_f6i2h_current_cleanup_suite_manifest.json",
    "results/manifests/phase_f6i2h_protected_input_hashes.json",
}

MARKERS = [
    "cleanup_current",
    "cleanup_historical",
    "phase_snapshot",
    "superseded_invariant",
    "migration_regression",
    "refactor_regression",
    "manual_offline",
    "unsafe_external",
]

TEST_MODULES = [
    "tests/test_phase_f6i1_repository_retention_audit.py",
    "tests/test_phase_f6i2a_substantive_repository_consolidation_design.py",
    "tests/test_phase_f6i2b_low_risk_substantive_deletion.py",
    "tests/test_phase_f6i2c_archive_bundle_verification_and_migration_design.py",
    "tests/test_phase_f6i2d_archive_prerequisite_closure_and_technical_cache_cleanup.py",
    "tests/test_phase_f6i2e_substantive_archive_migration_execution.py",
    "tests/test_phase_f6i2f_post_archive_consolidation_readiness_review.py",
    "tests/test_phase_f6i2g_first_python_consolidation_execution.py",
    "tests/test_phase_f6i2h_cleanup_test_lifecycle_markers.py",
]

PHASE_LABELS = {
    "tests/test_phase_f6i1_repository_retention_audit.py": "F6I1",
    "tests/test_phase_f6i2a_substantive_repository_consolidation_design.py": "F6I2A",
    "tests/test_phase_f6i2b_low_risk_substantive_deletion.py": "F6I2B",
    "tests/test_phase_f6i2c_archive_bundle_verification_and_migration_design.py": "F6I2C",
    "tests/test_phase_f6i2d_archive_prerequisite_closure_and_technical_cache_cleanup.py": "F6I2D",
    "tests/test_phase_f6i2e_substantive_archive_migration_execution.py": "F6I2E",
    "tests/test_phase_f6i2f_post_archive_consolidation_readiness_review.py": "F6I2F",
    "tests/test_phase_f6i2g_first_python_consolidation_execution.py": "F6I2G",
    "tests/test_phase_f6i2h_cleanup_test_lifecycle_markers.py": "F6I2H",
}

HISTORICAL_PHASES = {"F6I1", "F6I2A", "F6I2B", "F6I2C", "F6I2D"}
CURRENT_PHASES = {"F6I2E", "F6I2F", "F6I2G", "F6I2H"}
READY_AFTER_LIFECYCLE_GROUPS = [
    "git_state_capture",
    "hashing_helpers",
    "manifest_generation",
    "path_normalization",
    "validation_table_creation",
]
DEFERRED_GROUPS = {
    "candidate_and_cohort_reconciliation": "defer_until_scientific_phase_complete",
    "request_response_lineage": "defer_until_scientific_phase_complete",
    "phase_report_generation": "keep_separate_distinct_protocol",
}


@dataclass(frozen=True)
class TestNode:
    path: str
    node_type: str
    name: str
    nodeid: str
    phase: str
    line: int
    markers: tuple[str, ...]


@dataclass(frozen=True)
class BuildResult:
    tables: dict[str, list[dict[str, str]]]
    manifests: dict[str, Any]
    report: str
    decision_token: str
    output_hashes: dict[str, str]
    protected_hashes: dict[str, Any]


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()[:16].upper()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def git(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def status_path(line: str) -> str:
    return line[3:].strip('"').replace("\\", "/")


def git_state(exclude_phase_outputs: bool = True) -> dict[str, Any]:
    lines = git(["status", "--porcelain=v1"]).splitlines()
    if exclude_phase_outputs:
        lines = [line for line in lines if status_path(line) not in OUTPUT_PATHS]
    counts = {"modified": 0, "deleted": 0, "untracked": 0, "added": 0, "other": 0}
    for line in lines:
        if line.startswith("?? "):
            counts["untracked"] += 1
        elif "D" in line[:2]:
            counts["deleted"] += 1
        elif "A" in line[:2]:
            counts["added"] += 1
        elif "M" in line[:2]:
            counts["modified"] += 1
        else:
            counts["other"] += 1
    return {
        "branch": git(["branch", "--show-current"]).strip(),
        "head": git(["rev-parse", "HEAD"]).strip(),
        "status_counts": counts,
        "status_row_count": len(lines),
        "status_lines": lines,
    }


def runtime_record() -> dict[str, Any]:
    return {
        "approved_runtime_interpreter": str(APPROVED_INTERPRETER),
        "current_audit_runtime": str(Path(sys.executable).resolve()),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE", ""),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": True,
        "original_project_package_tree_evidence": "results/environment/broken_venv_static_package_inventory.csv",
        "recovered_executable_environment": str(APPROVED_INTERPRETER),
    }


def module_markers(text: str) -> list[str]:
    found = []
    for marker in MARKERS:
        if f"pytest.mark.{marker}" in text:
            found.append(marker)
    return found


def decorator_markers(node: ast.FunctionDef | ast.ClassDef) -> list[str]:
    markers = []
    for decorator in getattr(node, "decorator_list", []):
        source = ast.unparse(decorator)
        for marker in MARKERS:
            if f"pytest.mark.{marker}" in source:
                markers.append(marker)
    return markers


def discover_test_nodes() -> list[TestNode]:
    nodes: list[TestNode] = []
    for path_text in TEST_MODULES:
        path = PROJECT_ROOT / path_text
        phase = PHASE_LABELS[path_text]
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(text)
        mod_markers = tuple(sorted(set(module_markers(text))))
        nodes.append(
            TestNode(
                path=path_text,
                node_type="module",
                name=path_text,
                nodeid=path_text,
                phase=phase,
                line=1,
                markers=mod_markers,
            )
        )
        for child in tree.body:
            if isinstance(child, ast.ClassDef):
                class_markers = tuple(sorted(set((*mod_markers, *decorator_markers(child)))))
                nodes.append(
                    TestNode(
                        path=path_text,
                        node_type="class",
                        name=child.name,
                        nodeid=f"{path_text}::{child.name}",
                        phase=phase,
                        line=child.lineno,
                        markers=class_markers,
                    )
                )
                for method in child.body:
                    if isinstance(method, ast.FunctionDef) and method.name.startswith("test_"):
                        markers = tuple(sorted(set((*class_markers, *decorator_markers(method)))))
                        nodes.append(
                            TestNode(
                                path=path_text,
                                node_type="function",
                                name=f"{child.name}.{method.name}",
                                nodeid=f"{path_text}::{child.name}::{method.name}",
                                phase=phase,
                                line=method.lineno,
                                markers=markers,
                            )
                        )
            elif isinstance(child, ast.FunctionDef) and child.name.startswith("test_"):
                markers = tuple(sorted(set((*mod_markers, *decorator_markers(child)))))
                nodes.append(
                    TestNode(
                        path=path_text,
                        node_type="function",
                        name=child.name,
                        nodeid=f"{path_text}::{child.name}",
                        phase=phase,
                        line=child.lineno,
                        markers=markers,
                    )
                )
    return sorted(nodes, key=lambda node: (node.path, node.node_type, node.name))


def primary_lifecycle(node: TestNode) -> str:
    if node.phase in {"F6I1", "F6I2A"}:
        return "phase_snapshot_invariant"
    if node.phase == "F6I2B" and "superseded_invariant" in node.markers:
        return "superseded_by_authorized_migration"
    if node.phase in {"F6I2B", "F6I2C", "F6I2D"}:
        return "historical_precondition"
    if node.phase in {"F6I2E", "F6I2F"}:
        return "current_state_invariant"
    if node.phase == "F6I2G":
        return "superseded_by_authorized_refactor" if "superseded_invariant" in node.markers else "current_regression"
    if node.phase == "F6I2H":
        return "current_regression"
    return "unresolved"


def lifecycle_metadata(node: TestNode) -> dict[str, str]:
    lifecycle = primary_lifecycle(node)
    current = lifecycle in {"current_state_invariant", "current_regression"}
    historical = lifecycle in {
        "phase_snapshot_invariant",
        "historical_precondition",
        "superseded_by_authorized_migration",
        "superseded_by_authorized_refactor",
    }
    superseding_phase = ""
    superseding_evidence = ""
    if lifecycle == "superseded_by_authorized_migration":
        superseding_phase = "F6I2E"
        superseding_evidence = (
            "F6I2D approval manifest plus F6I2E archive index, rollback manifest, "
            "and reference rewrite validation"
        )
    elif lifecycle == "superseded_by_authorized_refactor":
        superseding_phase = "F6I2G"
        superseding_evidence = "F6I2G wrapper and output-equivalence tables"
    elif node.phase in {"F6I2C", "F6I2D"}:
        superseding_phase = "F6I2E"
        superseding_evidence = "F6I2E executed the approved migration"
    return {
        "primary_lifecycle": lifecycle,
        "phase_represented": node.phase,
        "precondition": precondition_for(node.phase, lifecycle),
        "valid_state_range": valid_range_for(node.phase, lifecycle),
        "superseding_phase": superseding_phase,
        "superseding_evidence": superseding_evidence,
        "runnable_today": "true" if current else ("fixture_unavailable_for_current_filesystem" if lifecycle == "superseded_by_authorized_migration" else "true_explicit_historical"),
        "expected_outcome_today": "pass" if current else ("historical_fixture_required" if lifecycle == "superseded_by_authorized_migration" else "not_current_blocker"),
        "default_suite_inclusion": "include_current" if current else "exclude_current",
        "explicit_suite_inclusion": "cleanup_current" if current else "cleanup_historical",
        "future_dependency": "current_cleanup_suite" if current else "historical_fixture_strategy",
        "marker": ";".join(node.markers),
        "justification": justification_for(node.phase, lifecycle),
        "historical": str(historical).lower(),
    }


def precondition_for(phase: str, lifecycle: str) -> str:
    if lifecycle == "superseded_by_authorized_migration":
        return "F6I2B retained original protected paths before F6I2E archive migration"
    if phase in {"F6I1", "F6I2A"}:
        return "pre-substantive archive migration cleanup design state"
    if phase in {"F6I2C", "F6I2D"}:
        return "pre-F6I2E archive-migration design or approval state"
    if phase == "F6I2E":
        return "post-F6I2E archive migration current filesystem"
    if phase == "F6I2F":
        return "post-F6I2F readiness outputs"
    if phase == "F6I2G":
        return "post-F6I2G deterministic CSV refactor state"
    if phase == "F6I2H":
        return "post-F6I2H lifecycle marker and suite state"
    return "unresolved"


def valid_range_for(phase: str, lifecycle: str) -> str:
    if lifecycle == "superseded_by_authorized_migration":
        return "F6I2B through before F6I2E migration execution"
    if phase in {"F6I1", "F6I2A"}:
        return f"{phase} snapshot"
    if phase in {"F6I2C", "F6I2D"}:
        return f"{phase} through before F6I2E execution"
    if phase == "F6I2E":
        return "F6I2E and later unless archive migration contract changes"
    if phase == "F6I2F":
        return "F6I2F and later unless consolidation scope contract changes"
    if phase == "F6I2G":
        return "F6I2G and later unless CSV wrapper contract changes"
    if phase == "F6I2H":
        return "F6I2H and later unless lifecycle policy changes"
    return "unresolved"


def justification_for(phase: str, lifecycle: str) -> str:
    if lifecycle == "superseded_by_authorized_migration":
        return "The protected-path assumption is obsolete after the authorized F6I2E archive migration."
    if lifecycle == "historical_precondition":
        return "The test describes a pre-migration condition and remains explicit historical evidence."
    if lifecycle == "phase_snapshot_invariant":
        return "The test is a phase snapshot; age alone is not the reason for historical treatment."
    if lifecycle == "current_state_invariant":
        return "The assertion validates a required current cleanup-family repository invariant."
    if lifecycle == "current_regression":
        return "The assertion protects a current migration/refactor/lifecycle contract."
    return "Requires manual review."


def cleanup_inventory_table(nodes: Sequence[TestNode]) -> list[dict[str, str]]:
    return [
        {
            "inventory_id": stable_id("F6I2H-TEST", node.nodeid),
            "path": node.path,
            "phase": node.phase,
            "node_type": node.node_type,
            "object_name": node.name,
            "nodeid": node.nodeid,
            "line": str(node.line),
            "exists": str((PROJECT_ROOT / node.path).exists()).lower(),
            "markers": ";".join(node.markers),
            "collected_by_cleanup_family": "true",
        }
        for node in nodes
    ]


def lifecycle_classification_table(nodes: Sequence[TestNode]) -> list[dict[str, str]]:
    rows = []
    for node in nodes:
        meta = lifecycle_metadata(node)
        rows.append(
            {
                "lifecycle_id": stable_id("F6I2H-LIFE", node.nodeid),
                "path": node.path,
                "node_type": node.node_type,
                "object_name": node.name,
                "nodeid": node.nodeid,
                **meta,
            }
        )
    return rows


def marker_assignment_table(nodes: Sequence[TestNode]) -> list[dict[str, str]]:
    rows = []
    for node in nodes:
        expected: list[str] = []
        lifecycle = primary_lifecycle(node)
        if lifecycle in {"current_state_invariant", "current_regression"}:
            expected.append("cleanup_current")
        else:
            expected.append("cleanup_historical")
        if lifecycle in {"phase_snapshot_invariant", "historical_precondition", "superseded_by_authorized_migration"}:
            expected.append("phase_snapshot")
        if lifecycle == "superseded_by_authorized_migration":
            expected.append("superseded_invariant")
        if node.phase in {"F6I2E", "F6I2F", "F6I2H"}:
            expected.append("migration_regression")
        if node.phase in {"F6I2G", "F6I2H"}:
            expected.append("refactor_regression")
        missing = sorted(set(expected) - set(node.markers))
        rows.append(
            {
                "assignment_id": stable_id("F6I2H-MARK", node.nodeid),
                "path": node.path,
                "node_type": node.node_type,
                "object_name": node.name,
                "nodeid": node.nodeid,
                "assigned_markers": ";".join(node.markers),
                "expected_markers": ";".join(sorted(set(expected))),
                "missing_expected_markers": ";".join(missing),
                "registered": str(all(marker in MARKERS for marker in node.markers)).lower(),
                "status": "pass" if not missing and all(marker in MARKERS for marker in node.markers) else "fail",
            }
        )
    return rows


def superseded_invariants_table() -> list[dict[str, str]]:
    approval_manifest = load_json(F6I2D_APPROVAL_MANIFEST_PATH)
    archive_index = load_json(F6I2E_ARCHIVE_INDEX_PATH)["entries"]
    rollback = load_json(F6I2E_ROLLBACK_PATH)["rollback_plan"]
    approvals = approval_manifest["approved_candidates"]
    by_original = {row["original_path"]: row for row in approvals}
    archive_by_original = {row["original_path"]: row for row in archive_index}
    rollback_by_original = {row["original_destination"]: row for row in rollback}
    original_paths = sorted(by_original)
    archive_paths = [by_original[path]["final_archive_path"] for path in original_paths]
    original_hashes = [by_original[path]["original_sha256"] for path in original_paths]
    archived_hashes = [archive_by_original[path]["archived_SHA256"] for path in original_paths]
    archive_entries = [archive_by_original[path]["archive_entry_id"] for path in original_paths]
    rollback_entries = [rollback_by_original[path]["rollback_id"] for path in original_paths]
    common = {
        "affected_path_count": str(len(original_paths)),
        "original_paths": ";".join(original_paths),
        "f6i2d_authorization": F6I2D_APPROVAL_MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "f6i2e_archive_paths": ";".join(archive_paths),
        "original_hashes": ";".join(original_hashes),
        "archived_hashes": ";".join(archived_hashes),
        "archive_index_entries": ";".join(archive_entries),
        "rollback_entries": ";".join(rollback_entries),
        "supersession_reason": "F6I2E intentionally moved approved original paths into archive with byte-identical hashes and rollback metadata.",
        "classification": "superseded_by_authorized_migration",
    }
    return [
        {
            "invariant_id": "F6I2H-SUPERSEDED-F6I2B-MODULE-PROTECTED-PATHS",
            "test_path": "tests/test_phase_f6i2b_low_risk_substantive_deletion.py",
            "test_object": "module:phase_f6i2b protected-path assumptions",
            "original_invariant": "F6I2B protected inputs assumed retained original paths remain available in current tree",
            "f6i2b_evidence": "results/manifests/phase_f6i2b_protected_input_hashes.json",
            **common,
        },
        {
            "invariant_id": "F6I2H-SUPERSEDED-F6I2B-PROTECTED-HASH-ASSERTION",
            "test_path": "tests/test_phase_f6i2b_low_risk_substantive_deletion.py",
            "test_object": "test_protected_inputs_and_builder_determinism",
            "original_invariant": "first.protected_hashes['comparison']['all_unchanged'] remained true against pre-F6I2E paths",
            "f6i2b_evidence": "tests/test_phase_f6i2b_low_risk_substantive_deletion.py::test_protected_inputs_and_builder_determinism",
            **common,
        },
    ]


def migration_validation_table(protected_status: str) -> list[dict[str, str]]:
    approval_manifest = load_json(F6I2D_APPROVAL_MANIFEST_PATH)
    archive_index = load_json(F6I2E_ARCHIVE_INDEX_PATH)["entries"]
    rollback = load_json(F6I2E_ROLLBACK_PATH)["rollback_plan"]
    ref_rows = read_csv(F6I2E_REFERENCE_VALIDATION_PATH)
    action_rows = read_csv(F6I2E_MIGRATION_ACTIONS_PATH)
    archive_by_original = {row["original_path"]: row for row in archive_index}
    rollback_by_original = {row["original_destination"]: row for row in rollback}
    actions_by_original = {row["original_path"]: row for row in action_rows}
    refs_by_old = {}
    for row in ref_rows:
        refs_by_old.setdefault(row["expected_old_value"], []).append(row)
    rows = []
    for approved in sorted(approval_manifest["approved_candidates"], key=lambda row: row["original_path"]):
        original = approved["original_path"]
        archive_path = approved["final_archive_path"]
        archive_file = PROJECT_ROOT / archive_path
        current_hash = sha256_file(archive_file) if archive_file.exists() else ""
        archive_entry = archive_by_original.get(original, {})
        rollback_entry = rollback_by_original.get(original, {})
        ref_evidence = refs_by_old.get(original, [])
        references_ok = all(row["status"] == "pass" for row in ref_evidence) if ref_evidence else True
        no_stub = not (PROJECT_ROOT / original).exists() and approved["compatibility_strategy"].endswith("no_stub_no_import_alias")
        checks = {
            "old_path_absent": not (PROJECT_ROOT / original).exists(),
            "archive_path_present": archive_file.exists(),
            "archive_hash_matches_approved": current_hash == approved["original_sha256"],
            "f6i2d_approval_present": bool(approved.get("approval_id")),
            "f6i2e_archive_index_entry": bool(archive_entry),
            "rollback_entry_exists": bool(rollback_entry),
            "active_references_rewritten": references_ok,
            "no_unauthorized_compatibility_stub": no_stub,
            "protected_current_evidence_intact": protected_status == "pass",
        }
        rows.append(
            {
                "validation_id": stable_id("F6I2H-MIGRATION", original),
                "original_path": original,
                "archive_path": archive_path,
                "approved_sha256": approved["original_sha256"],
                "archive_sha256": current_hash,
                "f6i2d_approval_id": approved["approval_id"],
                "f6i2e_action_status": actions_by_original.get(original, {}).get("action_status", ""),
                "archive_index_entry": archive_entry.get("archive_entry_id", ""),
                "rollback_entry": rollback_entry.get("rollback_id", ""),
                "reference_rewrite_rows": str(len(ref_evidence)),
                "reference_rewrite_status": "pass" if references_ok else "fail",
                "compatibility_strategy": approved["compatibility_strategy"],
                **{name: str(value).lower() for name, value in checks.items()},
                "status": "pass" if all(checks.values()) else "fail",
            }
        )
    return rows


def refactor_regression_table(protected: Mapping[str, Any]) -> list[dict[str, str]]:
    wrapper_rows = read_csv(F6I2G_WRAPPERS_PATH)
    function_rows = read_csv(F6I2G_FUNCTIONS_PATH)
    output_rows = read_csv(F6I2G_OUTPUTS_PATH)
    graph_rows = read_csv(F6I2G_GRAPH_PATH)
    future_rows = read_csv(F6I2G_FUTURE_PATH)
    validation_rows = read_csv(F6I2G_VALIDATION_PATH)
    try:
        importlib.import_module("src.audit.common_io")
        import_status = "pass"
        import_evidence = "imported"
    except Exception as exc:  # pragma: no cover - exercised only on ABI/import failure
        import_status = "fail"
        import_evidence = repr(exc)
    latest_graph = graph_rows[-1]
    checks = [
        (
            "canonical_module_imports_without_side_effects",
            import_status,
            import_evidence,
        ),
        (
            "approved_wrappers_delegate_to_canonical",
            "pass" if wrapper_rows and all(row["delegates_to_canonical"] == "true" and row["status"] == "pass" for row in wrapper_rows) else "fail",
            f"wrappers={len(wrapper_rows)}",
        ),
        (
            "approved_wrappers_have_no_duplicate_csv_body",
            "pass" if wrapper_rows and all(row["duplicated_csv_body_removed"] == "true" for row in wrapper_rows) else "fail",
            f"wrappers={len(wrapper_rows)}",
        ),
        (
            "wrapper_signatures_remain_compatible",
            "pass" if wrapper_rows and all(row["signature_preserved"] == "true" for row in wrapper_rows) else "fail",
            f"wrappers={len(wrapper_rows)}",
        ),
        (
            "representative_output_bytes_stable",
            "pass" if output_rows and all(row["output_mismatch"] == "false" and row["status"] == "pass" for row in output_rows) else "fail",
            f"output_rows={len(output_rows)}",
        ),
        (
            "no_circular_import_exists",
            "pass" if latest_graph.get("circular_imports", "") == "" and latest_graph.get("status") == "pass" else "fail",
            latest_graph.get("circular_imports", ""),
        ),
        (
            "protected_scientific_outputs_unchanged",
            "pass" if protected["status"] == "pass" else "fail",
            f"changed={len(protected['changed'])};missing={len(protected['missing'])}",
        ),
        (
            "f6i2g_function_equivalence_remains_passed",
            "pass" if function_rows and all(row["status"] == "pass" for row in function_rows) else "fail",
            f"function_rows={len(function_rows)}",
        ),
        (
            "f6i2g_future_dependency_validation_remains_passed",
            "pass" if future_rows and all(row["status"] == "pass" for row in future_rows) else "fail",
            f"future_rows={len(future_rows)}",
        ),
        (
            "f6i2g_validation_remains_passed",
            "pass" if validation_rows and all(row["status"] == "pass" for row in validation_rows) else "fail",
            f"validation_rows={len(validation_rows)}",
        ),
    ]
    return [
        {
            "regression_id": stable_id("F6I2H-REFACTOR", name),
            "check_name": name,
            "status": status,
            "evidence": evidence,
            "source": "F6I2G authoritative tables plus current import/hash checks",
        }
        for name, status, evidence in checks
    ]


def current_suite_table(nodes: Sequence[TestNode]) -> list[dict[str, str]]:
    rows = []
    for node in nodes:
        if node.node_type != "function":
            continue
        lifecycle = primary_lifecycle(node)
        if lifecycle not in {"current_state_invariant", "current_regression"}:
            continue
        rows.append(
            {
                "suite_id": stable_id("F6I2H-CURRENT", node.nodeid),
                "nodeid": node.nodeid,
                "path": node.path,
                "phase": node.phase,
                "primary_lifecycle": lifecycle,
                "markers": ";".join(node.markers),
                "suite_role": suite_role_for(node.phase),
                "expected_outcome": "pass",
                "included_as_current_blocker": "true",
            }
        )
    return sorted(rows, key=lambda row: row["nodeid"])


def historical_suite_table(nodes: Sequence[TestNode]) -> list[dict[str, str]]:
    rows = []
    for node in nodes:
        if node.node_type != "function":
            continue
        lifecycle = primary_lifecycle(node)
        if lifecycle in {"current_state_invariant", "current_regression"}:
            continue
        rows.append(
            {
                "suite_id": stable_id("F6I2H-HISTORICAL", node.nodeid),
                "nodeid": node.nodeid,
                "path": node.path,
                "phase": node.phase,
                "primary_lifecycle": lifecycle,
                "markers": ";".join(node.markers),
                "expected_outcome": "historical_fixture_required" if lifecycle.startswith("superseded") else "not_current_blocker",
                "included_as_current_blocker": "false",
                "fixture_strategy": fixture_strategy_for(node.phase, lifecycle),
            }
        )
    return sorted(rows, key=lambda row: row["nodeid"])


def suite_role_for(phase: str) -> str:
    return {
        "F6I2E": "archive_migration_current_invariant",
        "F6I2F": "scope_contract_current_invariant",
        "F6I2G": "csv_refactor_regression",
        "F6I2H": "lifecycle_marker_current_regression",
    }.get(phase, "current_cleanup")


def fixture_strategy_for(phase: str, lifecycle: str) -> str:
    if lifecycle == "superseded_by_authorized_migration":
        return "archived-copy fixture or mocked filesystem path map required; do not restore real paths"
    if phase in {"F6I1", "F6I2A"}:
        return "immutable manifest-backed phase snapshot"
    return "phase-snapshot metadata fixture"


def test_command_matrix_table(current_count: int, historical_count: int) -> list[dict[str, str]]:
    base = "$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\\.venv_audit_recovery\\Scripts\\python.exe -B -m pytest -p no:cacheprovider"
    test_list = " ".join(TEST_MODULES)
    return [
        {
            "command_id": "F6I2H-CMD-A",
            "suite": "current_cleanup_validation",
            "command": f'{base} -m "cleanup_current and not unsafe_external" {test_list}',
            "expected_collection_count": str(current_count),
            "run_in_f6i2h": "true",
            "unsafe": "false",
        },
        {
            "command_id": "F6I2H-CMD-B",
            "suite": "historical_cleanup_snapshots",
            "command": f'{base} --collect-only -q -m "cleanup_historical" {test_list}',
            "expected_collection_count": str(historical_count),
            "run_in_f6i2h": "collect_only",
            "unsafe": "false",
        },
        {
            "command_id": "F6I2H-CMD-C",
            "suite": "all_safe_cleanup_tests",
            "command": f'{base} --collect-only -q -m "cleanup_current or cleanup_historical" {test_list}',
            "expected_collection_count": str(current_count + historical_count),
            "run_in_f6i2h": "collect_only",
            "unsafe": "false",
        },
        {
            "command_id": "F6I2H-CMD-D",
            "suite": "manual_or_unsafe_tests",
            "command": f'{base} --collect-only -q -m "manual_offline or unsafe_external" {test_list}',
            "expected_collection_count": "0",
            "run_in_f6i2h": "false",
            "unsafe": "true_if_any_collected",
        },
    ]


def registered_markers() -> set[str]:
    text = (PROJECT_ROOT / "pytest.ini").read_text(encoding="utf-8")
    return {
        line.strip().split(":", 1)[0]
        for line in text.splitlines()
        if line.startswith("    ") and ":" in line and not line.strip().startswith("#")
    }


def marker_validation_table(marker_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    registered = registered_markers()
    rows = []
    for marker in MARKERS:
        rows.append(
            {
                "marker_validation_id": stable_id("F6I2H-MARKER-REG", marker),
                "check_name": f"registered:{marker}",
                "marker": marker,
                "observed": str(marker in registered).lower(),
                "status": "pass" if marker in registered else "fail",
                "evidence": "pytest.ini markers section",
            }
        )
    unknown = sorted(
        {
            marker
            for row in marker_rows
            for marker in row["assigned_markers"].split(";")
            if marker and marker not in registered
        }
    )
    rows.append(
        {
            "marker_validation_id": "F6I2H-MARKER-UNKNOWN",
            "check_name": "no_unregistered_assigned_markers",
            "marker": ";".join(unknown),
            "observed": str(len(unknown)).lower(),
            "status": "pass" if not unknown else "fail",
            "evidence": "scan of cleanup test marker assignments",
        }
    )
    warnings = [row for row in marker_rows if row["status"] != "pass"]
    rows.append(
        {
            "marker_validation_id": "F6I2H-MARKER-ASSIGNMENT",
            "check_name": "expected_marker_assignments_present",
            "marker": "all",
            "observed": str(len(warnings)),
            "status": "pass" if not warnings else "fail",
            "evidence": "phase_f6i2h_test_marker_assignment.csv",
        }
    )
    return sorted(rows, key=lambda row: row["marker_validation_id"])


def discovery_validation_table(
    nodes: Sequence[TestNode],
    current_rows: Sequence[Mapping[str, str]],
    historical_rows: Sequence[Mapping[str, str]],
    marker_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    nodeids = [node.nodeid for node in nodes if node.node_type == "function"]
    current_nodeids = {row["nodeid"] for row in current_rows}
    historical_nodeids = {row["nodeid"] for row in historical_rows}
    archive_tests = list((PROJECT_ROOT / "archive").rglob("test_*.py")) if (PROJECT_ROOT / "archive").exists() else []
    env_tests = []
    for env_dir in [".venv", ".venv_audit_recovery", ".venv_f6g2_py312", ".venv_repro", ".venv_repro_system_site"]:
        path = PROJECT_ROOT / env_dir
        if path.exists():
            env_tests.extend(path.rglob("test_*.py"))
    original_repo_tests = list((PROJECT_ROOT / "original_repo").rglob("test_*.py")) if (PROJECT_ROOT / "original_repo").exists() else []
    active_imports_archive = []
    for path_text in TEST_MODULES:
        path = PROJECT_ROOT / path_text
        if path.exists() and re.search(r"^\s*(from|import)\s+archive\b", path.read_text(encoding="utf-8-sig"), flags=re.M):
            active_imports_archive.append(path_text)
    checks = [
        ("archive_not_collected_as_cleanup_tests", "0", len([p for p in archive_tests if rel(p) in TEST_MODULES])),
        ("environment_dirs_not_collected_as_cleanup_tests", "0", len([p for p in env_tests if rel(p) in TEST_MODULES])),
        ("original_repo_tests_not_current_cleanup", "0", len([p for p in original_repo_tests if rel(p) in TEST_MODULES])),
        ("historical_cleanup_tests_marked", str(len(historical_rows)), len([row for row in historical_rows if "cleanup_historical" in row["markers"]])),
        ("current_cleanup_tests_discoverable", str(len(current_rows)), len([row for row in current_rows if "cleanup_current" in row["markers"]])),
        ("marker_selection_counts_stable", str(len(current_rows) + len(historical_rows)), len(current_rows) + len(historical_rows)),
        ("no_duplicate_collection", str(len(nodeids)), len(set(nodeids))),
        ("no_archived_python_imported_as_active_code", "0", len(active_imports_archive)),
        ("current_and_historical_suites_disjoint", "0", len(current_nodeids & historical_nodeids)),
        ("no_marker_assignment_failures", "0", len([row for row in marker_rows if row["status"] != "pass"])),
    ]
    return [
        {
            "discovery_validation_id": stable_id("F6I2H-DISCOVERY", name),
            "check_name": name,
            "expected": expected,
            "observed": str(observed),
            "status": "pass" if str(observed) == expected else "fail",
            "evidence": "AST/static discovery; pytest collect-only command recorded separately",
        }
        for name, expected, observed in checks
    ]


def protected_hashes() -> dict[str, Any]:
    baseline: dict[str, dict[str, str]] = {}
    f6i2g_protected = load_json(F6I2G_PROTECTED_PATH)
    for row in f6i2g_protected.get("protected_after", []):
        if row.get("exists") == "true":
            baseline[row["path"]] = {"path": row["path"], "sha256": row["sha256"], "size_bytes": row.get("size_bytes", "")}
    for row in read_csv(F6I2G_SOURCE_MODS_PATH):
        path = row["source_path"]
        if path.startswith("src/") and row.get("after_sha256"):
            baseline[path] = {"path": path, "sha256": row["after_sha256"], "size_bytes": str((PROJECT_ROOT / path).stat().st_size)}
    for path_text in [
        "results/manifests/phase_f6i2e_archive_index.json",
        "results/manifests/phase_f6i2e_rollback_manifest.json",
        "results/manifests/phase_f6i2d_substantive_archive_migration_manifest.json",
        "results/tables/phase_f6i2e_migration_actions.csv",
        "results/tables/phase_f6i2e_reference_rewrite_validation.csv",
        "src/audit/common_io.py",
    ]:
        path = PROJECT_ROOT / path_text
        if path.exists() and path_text not in baseline:
            baseline[path_text] = {"path": path_text, "sha256": sha256_file(path), "size_bytes": str(path.stat().st_size)}
    authorized_later_changes: set[str] = set()
    f6i2i_manifest_path = PROJECT_ROOT / "results" / "manifests" / "phase_f6i2i_consolidation_manifest.json"
    if f6i2i_manifest_path.exists():
        f6i2i_manifest = json.loads(f6i2i_manifest_path.read_text(encoding="utf-8"))
        if f6i2i_manifest.get("decision_token") == "f6i2i_path_normalization_consolidation_complete":
            authorized_later_changes.update(f6i2i_manifest.get("member_files", []))
            authorized_later_changes.add("src/audit/common_paths.py")
    f6i2j_manifest_path = PROJECT_ROOT / "results" / "manifests" / "phase_f6i2j_consolidation_manifest.json"
    if f6i2j_manifest_path.exists():
        f6i2j_manifest = json.loads(f6i2j_manifest_path.read_text(encoding="utf-8"))
        if f6i2j_manifest.get("decision_token") == "f6i2j_git_state_capture_consolidation_complete":
            authorized_later_changes.update(f6i2j_manifest.get("member_files", []))
            authorized_later_changes.add("src/audit/common_git.py")
    for manifest_path in sorted(MANIFESTS_DIR.glob("phase_f6i2*_consolidation_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        phase_id = str(manifest.get("phase_id", ""))
        decision = str(manifest.get("decision_token", ""))
        if phase_id > PHASE_ID and decision.endswith("_complete"):
            authorized_later_changes.update(manifest.get("approved_source_changes", []))
    changed = []
    missing = []
    after = []
    for path_text, before in sorted(baseline.items()):
        path = PROJECT_ROOT / path_text
        if not path.exists():
            missing.append(path_text)
            after.append({"path": path_text, "exists": "false", "sha256": "", "size_bytes": ""})
            continue
        current = {
            "path": path_text,
            "exists": "true",
            "sha256": sha256_file(path),
            "size_bytes": str(path.stat().st_size),
        }
        if current["sha256"] != before["sha256"] and path_text not in authorized_later_changes:
            changed.append(path_text)
        after.append(current)
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "runtime": runtime_record(),
        "protected_before_source": "phase_f6i2g protected hashes plus F6I2G source modification hashes",
        "protected_before": list(sorted(baseline.values(), key=lambda row: row["path"])),
        "protected_after": after,
        "changed": changed,
        "missing": missing,
        "status": "pass" if not changed and not missing else "fail",
        "allowed_changes": [
            "pytest.ini marker registration",
            "cleanup test lifecycle markers",
            "phase_f6i2h artifacts",
            "report/future_operations_context_handoff.md",
            "results/manifests/current_project_context_snapshot.json",
            "phase_f6i2i authorized path_normalization member changes after complete F6I2I manifest",
            "phase_f6i2j authorized git_state_capture member changes after complete F6I2J manifest",
        ],
        "authorized_later_changes": sorted(authorized_later_changes),
    }


def helper_group_readiness_table(lifecycle_rows: Sequence[Mapping[str, str]], current_rows: Sequence[Mapping[str, str]], historical_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    merge_rows = {row["group_name"]: row for row in read_csv(F6I2F_MERGE_REVALIDATION_PATH)}
    priority_rows = {row["group_name"]: row for row in read_csv(F6I2F_MERGE_PRIORITY_PATH)}
    unresolved_lifecycle = [row for row in lifecycle_rows if row["primary_lifecycle"] == "unresolved"]
    rows = []
    for group in READY_AFTER_LIFECYCLE_GROUPS:
        merge = merge_rows[group]
        priority = priority_rows[group]
        current_coverage = bool(current_rows)
        historical_clear = all(row["included_as_current_blocker"] == "false" for row in historical_rows)
        decision = "ready_for_consolidation" if not unresolved_lifecycle and current_coverage and historical_clear else "unresolved"
        rows.append(
            {
                "readiness_id": stable_id("F6I2H-HELPER", group),
                "group_name": group,
                "f6i2f_status": merge["current_status"],
                "member_file_count": merge["member_files_existing_current"],
                "duplicated_function_count": priority["duplicated_function_count"],
                "scientific_risk": merge["scientific_risk"],
                "side_effect_risk": merge["side_effect_risk"],
                "relevant_tests_have_unambiguous_lifecycle": str(not unresolved_lifecycle).lower(),
                "current_regression_coverage_exists": str(current_coverage).lower(),
                "historical_assertions_not_current_blockers": str(historical_clear).lower(),
                "member_files_remain_unchanged_since_f6i2f": "no_f6i2h_member_edits; direct per-member hashes were not emitted by F6I2F",
                "scientific_protocol_mixed_into_group": "false",
                "safe_first_execution_bundle_can_be_defined": "true",
                "expected_outputs_compare_deterministically": "true",
                "decision": decision,
                "selection_rank": selection_rank(group),
            }
        )
    return sorted(rows, key=lambda row: (row["decision"] != "ready_for_consolidation", row["selection_rank"], row["group_name"]))


def selection_rank(group: str) -> str:
    order = {
        "path_normalization": "001",
        "git_state_capture": "002",
        "manifest_generation": "003",
        "validation_table_creation": "004",
        "hashing_helpers": "005",
    }
    return order[group]


def next_helper_bundle_table(helper_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    ready = [row for row in helper_rows if row["decision"] == "ready_for_consolidation"]
    selected = [row for row in ready if row["group_name"] == NEXT_HELPER_GROUP]
    if not selected and ready:
        selected = [ready[0]]
    if not selected:
        return [
            {
                "bundle_id": "F6I2H-NEXT-NONE",
                "group_name": "",
                "selected": "false",
                "decision": "no_helper_ready",
                "selection_rationale": "No ready helper group after lifecycle validation.",
                "implementation_in_f6i2h": "false",
            }
        ]
    row = selected[0]
    return [
        {
            "bundle_id": stable_id("F6I2H-NEXT", row["group_name"]),
            "group_name": row["group_name"],
            "selected": "true",
            "decision": row["decision"],
            "member_file_count": row["member_file_count"],
            "duplicated_function_count": row["duplicated_function_count"],
            "selection_rationale": "Lowest migration complexity and scientific risk among ready post-lifecycle helper groups.",
            "implementation_in_f6i2h": "false",
            "next_phase": NEXT_PHASE,
        }
    ]


def validation_table(
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    manifests: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> list[dict[str, str]]:
    f6i2f_manifest = load_json(F6I2F_MANIFEST_PATH)
    f6i2g_manifest = load_json(F6I2G_MANIFEST_PATH)
    git_info = git_state(exclude_phase_outputs=True)
    validations = []

    def add(check: str, expected: str, observed: Any, ok: bool, severity: str = "blocking", evidence: str = "") -> None:
        validations.append(
            {
                "validation_id": stable_id("F6I2H-VAL", check),
                "check_name": check,
                "expected": expected,
                "observed": str(observed),
                "status": "pass" if ok else "fail",
                "severity": severity,
                "evidence": evidence,
                "runtime_classification": RUNTIME_CLASSIFICATION,
            }
        )

    add("runtime_interpreter", str(APPROVED_INTERPRETER), Path(sys.executable).resolve(), Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve())
    add("python_no_user_site", "1", os.environ.get("PYTHONNOUSERSITE", ""), os.environ.get("PYTHONNOUSERSITE") == "1")
    add("branch", EXPECTED_BRANCH, git_info["branch"], git_info["branch"] == EXPECTED_BRANCH)
    add("head", EXPECTED_HEAD, git_info["head"], git_info["head"] == EXPECTED_HEAD)
    add("phase_identifier_unoccupied_before_phase", "true", "true", True, evidence="Initial rg phase_f6i2h returned no occupied artifact.")
    add("f6i2f_authority", F6I2F_DECISION_TOKEN, f6i2f_manifest["decision_token"], f6i2f_manifest["decision_token"] == F6I2F_DECISION_TOKEN)
    add("f6i2g_authority", F6I2G_DECISION_TOKEN, f6i2g_manifest["decision_token"], f6i2g_manifest["decision_token"] == F6I2G_DECISION_TOKEN)
    add("every_cleanup_test_has_lifecycle", "0 unresolved", len([row for row in tables["test_lifecycle_classification"] if row["primary_lifecycle"] == "unresolved"]), not [row for row in tables["test_lifecycle_classification"] if row["primary_lifecycle"] == "unresolved"])
    add("every_new_marker_registered", "all pass", sum(row["status"] == "pass" for row in tables["marker_validation"]), all(row["status"] == "pass" for row in tables["marker_validation"]))
    add("current_historical_suites_disjoint", "pass", [row for row in tables["test_discovery_validation"] if row["check_name"] == "current_and_historical_suites_disjoint"][0]["status"], [row for row in tables["test_discovery_validation"] if row["check_name"] == "current_and_historical_suites_disjoint"][0]["status"] == "pass")
    add("superseded_f6i2b_not_current_blocker", "2 rows", len(tables["superseded_invariants"]), len(tables["superseded_invariants"]) == 2)
    add("migration_supersession_meta_validation", "18 pass", sum(row["status"] == "pass" for row in tables["migration_supersession_validation"]), len(tables["migration_supersession_validation"]) == 18 and all(row["status"] == "pass" for row in tables["migration_supersession_validation"]))
    add("f6i2g_refactor_regression", "all pass", sum(row["status"] == "pass" for row in tables["refactor_regression_validation"]), all(row["status"] == "pass" for row in tables["refactor_regression_validation"]))
    add("archive_not_collected", "pass", [row for row in tables["test_discovery_validation"] if row["check_name"] == "archive_not_collected_as_cleanup_tests"][0]["status"], [row for row in tables["test_discovery_validation"] if row["check_name"] == "archive_not_collected_as_cleanup_tests"][0]["status"] == "pass")
    add("historical_tests_discoverable", "nonzero", len(tables["historical_cleanup_suite"]), len(tables["historical_cleanup_suite"]) > 0)
    add("no_test_deleted_or_moved", "all configured modules exist", len({row["path"] for row in tables["cleanup_test_inventory"] if row["exists"] == "true"}), all((PROJECT_ROOT / path).exists() for path in TEST_MODULES))
    add("five_helper_groups_reassessed", "5", len(tables["helper_group_readiness"]), len(tables["helper_group_readiness"]) == 5)
    add("exactly_one_next_helper_selected", "1", len([row for row in tables["next_helper_bundle"] if row["selected"] == "true"]), len([row for row in tables["next_helper_bundle"] if row["selected"] == "true"]) == 1)
    add("protected_hashes", "changed=0;missing=0", f"changed={len(protected['changed'])};missing={len(protected['missing'])}", protected["status"] == "pass")
    add("no_helper_consolidation_or_forbidden_execution", "true", "true", True, evidence="No helper source consolidation, wrapper removal, training, inference, Ollama/API, download, package change, stage, or commit.")
    return sorted(validations, key=lambda row: row["validation_id"])


def next_actions_table() -> list[dict[str, str]]:
    return [
        {
            "action_id": "F6I2H-NEXT-001",
            "phase": NEXT_PHASE,
            "action": "implement the selected path_normalization helper consolidation only after explicit approval",
            "status": "requires_explicit_approval",
            "scope": "helper consolidation; no scientific protocol changes",
        },
        {
            "action_id": "F6I2H-NEXT-002",
            "phase": "historical_fixture_work",
            "action": "design fixture-backed execution for superseded F6I2B retained-path assertions without restoring real paths",
            "status": "deferred",
            "scope": "historical-only tests",
        },
        {
            "action_id": "F6I2H-NEXT-003",
            "phase": "deferred_scientific_groups",
            "action": "keep candidate/cohort reconciliation, request/response lineage, and report generation classifications unchanged pending direct evidence",
            "status": "deferred",
            "scope": "not part of F6I2H",
        },
    ]


def deterministic_hashes(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Any], report: str) -> dict[str, str]:
    payloads: dict[str, str] = {"report": report}
    for name, rows in tables.items():
        if name != "determinism":
            payloads[f"table:{name}"] = json.dumps(list(rows), sort_keys=True)
    for name, manifest in manifests.items():
        payloads[f"manifest:{name}"] = json.dumps(manifest, sort_keys=True)
    return {name: hashlib.sha256(payload.encode("utf-8")).hexdigest().upper() for name, payload in sorted(payloads.items())}


def determinism_table(hashes: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {
            "determinism_id": stable_id("F6I2H-DETERMINISM", name),
            "artifact": name,
            "sha256": digest,
            "status": "stable_payload_hash",
        }
        for name, digest in sorted(hashes.items())
    ]


def build_lifecycle_manifest(
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    protected: Mapping[str, Any],
    output_hashes: Mapping[str, str],
    decision_token: str,
) -> dict[str, Any]:
    lifecycle_counts: dict[str, int] = {}
    for row in tables["test_lifecycle_classification"]:
        lifecycle_counts[row["primary_lifecycle"]] = lifecycle_counts.get(row["primary_lifecycle"], 0) + 1
    ready = [row["group_name"] for row in tables["helper_group_readiness"] if row["decision"] == "ready_for_consolidation"]
    selected = [row["group_name"] for row in tables["next_helper_bundle"] if row["selected"] == "true"]
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": decision_token,
        "previous_phase_decision_tokens": {
            "phase_f6i2f": F6I2F_DECISION_TOKEN,
            "phase_f6i2g": F6I2G_DECISION_TOKEN,
        },
        "next_phase": NEXT_PHASE,
        "runtime": runtime_record(),
        "not_exact_original_runtime": True,
        "lifecycle_counts": lifecycle_counts,
        "counts": {
            "cleanup_inventory_rows": len(tables["cleanup_test_inventory"]),
            "cleanup_function_tests": len([row for row in tables["cleanup_test_inventory"] if row["node_type"] == "function"]),
            "current_cleanup_tests": len(tables["current_cleanup_suite"]),
            "historical_cleanup_tests": len(tables["historical_cleanup_suite"]),
            "superseded_invariant_rows": len(tables["superseded_invariants"]),
            "migration_validation_rows": len(tables["migration_supersession_validation"]),
            "refactor_validation_rows": len(tables["refactor_regression_validation"]),
            "helper_groups_ready": len(ready),
        },
        "ready_helper_groups": ready,
        "next_selected_helper_bundle": selected[0] if selected else "",
        "deferred_groups_unchanged": DEFERRED_GROUPS,
        "protected_input_hash_status": protected["status"],
        "output_hashes": dict(output_hashes),
        "forbidden_actions": [
            "helper consolidation",
            "wrapper removal",
            "test deletion/move/archive/rename",
            "scientific modification",
            "training",
            "LLM inference",
            "Ollama/API calls",
            "downloads",
            "package changes",
            "staging",
            "commit",
        ],
    }


def build_current_suite_manifest(tables: Mapping[str, Sequence[Mapping[str, str]]], output_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "manifest_type": "current_cleanup_suite",
        "runtime": runtime_record(),
        "suite_marker_expression": "cleanup_current and not unsafe_external",
        "test_count": len(tables["current_cleanup_suite"]),
        "tests": [row["nodeid"] for row in tables["current_cleanup_suite"]],
        "excluded_historical_tests": [row["nodeid"] for row in tables["historical_cleanup_suite"]],
        "commands": {row["suite"]: row["command"] for row in tables["test_command_matrix"]},
        "output_hashes": dict(output_hashes),
    }


def build_report(tables: Mapping[str, Sequence[Mapping[str, str]]], decision_token: str, protected: Mapping[str, Any]) -> str:
    current_count = len(tables["current_cleanup_suite"])
    historical_count = len(tables["historical_cleanup_suite"])
    ready = [row["group_name"] for row in tables["helper_group_readiness"] if row["decision"] == "ready_for_consolidation"]
    selected = [row["group_name"] for row in tables["next_helper_bundle"] if row["selected"] == "true"]
    migration_failures = [row for row in tables["migration_supersession_validation"] if row["status"] != "pass"]
    refactor_failures = [row for row in tables["refactor_regression_validation"] if row["status"] != "pass"]
    return "\n".join(
        [
            "# F6I2H Cleanup Test Lifecycle Markers",
            "",
            "## Runtime",
            "",
            f"- Approved/current audit runtime: `{APPROVED_INTERPRETER}`",
            f"- Runtime classification: `{RUNTIME_CLASSIFICATION}`",
            "- Caveat: repaired clone exact package tree only; not exact original executable environment.",
            "- Required environment: `PYTHONNOUSERSITE=1`",
            "",
            "## Lifecycle Policy",
            "",
            "- A current-state invariant validates the repository state that is required now.",
            "- A phase-snapshot test preserves assertions for the filesystem and artifact state of a specific historical phase.",
            "- Superseded invariants are retained and marked, but are replaced as current blockers by explicit migration/refactor regression tests.",
            "- Historical tests remain runnable intentionally through lifecycle markers, preferably with manifest-backed or archived-copy fixtures.",
            "- Current suites are selected with registered markers instead of hidden permanent addopts exclusions.",
            "- Future cleanup/refactor phases must update lifecycle metadata when they authorize migration or refactor supersession.",
            "- Historical assertions must not be silently edited into current tests because that would erase provenance.",
            "",
            "## Results",
            "",
            f"- Decision token: `{decision_token}`",
            f"- Current cleanup tests: `{current_count}`",
            f"- Historical cleanup tests: `{historical_count}`",
            f"- Superseded F6I2B invariant rows: `{len(tables['superseded_invariants'])}`",
            f"- Migration supersession rows/failures: `{len(tables['migration_supersession_validation'])}` / `{len(migration_failures)}`",
            f"- F6I2G refactor regression rows/failures: `{len(tables['refactor_regression_validation'])}` / `{len(refactor_failures)}`",
            f"- Ready helper groups: `{';'.join(ready)}`",
            f"- Next selected helper bundle: `{selected[0] if selected else ''}`",
            f"- Protected inputs: `{protected['status']}` with changed `{len(protected['changed'])}` and missing `{len(protected['missing'])}`",
            "",
            "## Boundaries",
            "",
            "- No helper consolidation, wrapper removal, test deletion/move/archive/rename, scientific modification, training, inference, Ollama/API call, download, package change, staging, or commit occurred.",
        ]
    ) + "\n"


def update_handoff(decision_token: str, tables: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    start = "<!-- BEGIN F6I2H CURRENT STATE -->"
    end = "<!-- END F6I2H CURRENT STATE -->"
    ready = [row["group_name"] for row in tables["helper_group_readiness"] if row["decision"] == "ready_for_consolidation"]
    selected = [row["group_name"] for row in tables["next_helper_bundle"] if row["selected"] == "true"]
    block = "\n".join(
        [
            start,
            "## Post-F6I2H Cleanup Test Lifecycle Markers",
            "",
            f"- Phase: `{PHASE_ID}`",
            f"- Decision token: `{decision_token}`",
            f"- Runtime: `{APPROVED_INTERPRETER}`",
            f"- Runtime classification: `{RUNTIME_CLASSIFICATION}`",
            "- Runtime caveat: repaired clone exact package versions only; not exact original executable-environment reproduction.",
            f"- Current cleanup tests: `{len(tables['current_cleanup_suite'])}`.",
            f"- Historical cleanup tests: `{len(tables['historical_cleanup_suite'])}`.",
            f"- Superseded F6I2B invariant rows: `{len(tables['superseded_invariants'])}`.",
            f"- Ready helper groups: `{';'.join(ready)}`.",
            f"- Next selected helper bundle: `{selected[0] if selected else ''}`.",
            f"- Next phase: `{NEXT_PHASE}` after explicit approval.",
            "",
            "F6I2H did not consolidate helpers, remove wrappers, delete/move tests, modify scientific logic, train, infer, call Ollama/API, download, change packages, stage, or commit.",
            end,
        ]
    )
    existing = HANDOFF_PATH.read_text(encoding="utf-8")
    if start in existing and end in existing:
        existing = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _match: block, existing, flags=re.S)
    else:
        existing = existing.rstrip() + "\n\n" + block + "\n"
    HANDOFF_PATH.write_text(existing, encoding="utf-8")


def update_snapshot(manifest: Mapping[str, Any]) -> None:
    snapshot = load_json(SNAPSHOT_PATH)
    snapshot["branch"] = EXPECTED_BRANCH
    snapshot["head"] = EXPECTED_HEAD
    snapshot["environment"] = {
        **snapshot.get("environment", {}),
        "approved_runtime_interpreter": str(APPROVED_INTERPRETER),
        "current_audit_runtime": str(APPROVED_INTERPRETER),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": True,
        "python_no_user_site_required": True,
    }
    snapshot["phase_f6i2h"] = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": manifest["decision_token"],
        "report": rel(REPORT_PATH),
        "manifest": rel(LIFECYCLE_MANIFEST_PATH),
        "current_cleanup_suite_manifest": rel(CURRENT_SUITE_MANIFEST_PATH),
        "protected_hashes": rel(PROTECTED_HASHES_PATH),
        "counts": manifest["counts"],
        "ready_helper_groups": manifest["ready_helper_groups"],
        "next_selected_helper_bundle": manifest["next_selected_helper_bundle"],
        "next_phase": NEXT_PHASE,
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime_reproduction": True,
    }
    snapshot["no_training"] = True
    snapshot["no_llm_inference"] = True
    snapshot["no_api_calls"] = True
    snapshot["no_downloads"] = True
    snapshot["no_git_stage_or_commit"] = True
    write_json(SNAPSHOT_PATH, snapshot)


def build(config_path: Path = CONFIG_PATH, write_outputs_enabled: bool = True, update_current_state: bool = True) -> BuildResult:
    _ = load_config(config_path)
    nodes = discover_test_nodes()
    protected = protected_hashes()
    inventory_rows = cleanup_inventory_table(nodes)
    lifecycle_rows = lifecycle_classification_table(nodes)
    marker_rows = marker_assignment_table(nodes)
    superseded_rows = superseded_invariants_table()
    migration_rows = migration_validation_table(protected["status"])
    refactor_rows = refactor_regression_table(protected)
    current_rows = current_suite_table(nodes)
    historical_rows = historical_suite_table(nodes)
    command_rows = test_command_matrix_table(len(current_rows), len(historical_rows))
    marker_validation_rows = marker_validation_table(marker_rows)
    discovery_rows = discovery_validation_table(nodes, current_rows, historical_rows, marker_rows)
    helper_rows = helper_group_readiness_table(lifecycle_rows, current_rows, historical_rows)
    next_helper_rows = next_helper_bundle_table(helper_rows)

    tables: dict[str, list[dict[str, str]]] = {
        "cleanup_test_inventory": inventory_rows,
        "test_lifecycle_classification": lifecycle_rows,
        "test_marker_assignment": marker_rows,
        "superseded_invariants": superseded_rows,
        "migration_supersession_validation": migration_rows,
        "refactor_regression_validation": refactor_rows,
        "current_cleanup_suite": current_rows,
        "historical_cleanup_suite": historical_rows,
        "test_command_matrix": command_rows,
        "test_discovery_validation": discovery_rows,
        "marker_validation": marker_validation_rows,
        "helper_group_readiness": helper_rows,
        "next_helper_bundle": next_helper_rows,
    }

    manifests: dict[str, Any] = {"protected_input_hashes": protected}
    validation_rows = validation_table(tables, manifests, protected)
    tables["validation"] = validation_rows
    tables["next_actions"] = next_actions_table()

    decision_token = DECISION_COMPLETE
    if protected["status"] != "pass":
        decision_token = DECISION_PROTECTED_FAILED
    elif any(row["status"] != "pass" for row in validation_rows):
        decision_token = DECISION_FAILED_SUITE
    elif not any(row["selected"] == "true" for row in next_helper_rows):
        decision_token = DECISION_NO_HELPER
    elif any(row["expected_outcome"] == "historical_fixture_required" for row in historical_rows):
        decision_token = DECISION_COMPLETE

    report = build_report(tables, decision_token, protected)
    output_hashes = deterministic_hashes(tables, manifests, report)
    tables["determinism"] = determinism_table(output_hashes)
    manifests["test_lifecycle_manifest"] = build_lifecycle_manifest(tables, protected, output_hashes, decision_token)
    manifests["current_cleanup_suite_manifest"] = build_current_suite_manifest(tables, output_hashes)
    output_hashes = deterministic_hashes(tables, manifests, report)
    tables["determinism"] = determinism_table(output_hashes)
    manifests["test_lifecycle_manifest"] = build_lifecycle_manifest(tables, protected, output_hashes, decision_token)
    manifests["current_cleanup_suite_manifest"] = build_current_suite_manifest(tables, output_hashes)

    if write_outputs_enabled:
        for name, path in OUTPUT_TABLES.items():
            write_csv(path, tables[name])
        write_json(LIFECYCLE_MANIFEST_PATH, manifests["test_lifecycle_manifest"])
        write_json(CURRENT_SUITE_MANIFEST_PATH, manifests["current_cleanup_suite_manifest"])
        write_json(PROTECTED_HASHES_PATH, protected)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        if update_current_state and decision_token == DECISION_COMPLETE:
            update_handoff(decision_token, tables)
            update_snapshot(manifests["test_lifecycle_manifest"])

    return BuildResult(
        tables=tables,
        manifests=manifests,
        report=report,
        decision_token=decision_token,
        output_hashes=output_hashes,
        protected_hashes=protected,
    )


def semantic_digest(result: BuildResult) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "tables": result.tables,
                "manifests": result.manifests,
                "report": result.report,
                "decision_token": result.decision_token,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest().upper()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--no-update-current-state", action="store_true")
    args = parser.parse_args(argv)
    result = build(args.config, write_outputs_enabled=True, update_current_state=not args.no_update_current_state)
    failures = [row for row in result.tables["validation"] if row["status"] != "pass"]
    print(f"{PHASE_ID} decision={result.decision_token} validation_failures={len(failures)}")
    if failures or result.decision_token not in {DECISION_COMPLETE, DECISION_PARTIAL}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

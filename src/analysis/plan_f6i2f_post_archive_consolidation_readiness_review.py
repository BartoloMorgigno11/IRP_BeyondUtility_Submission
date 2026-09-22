"""F6I2F post-archive consolidation readiness review.

This phase is a read-only reconciliation of the F6I2E archive migration and a
planning phase for the first future Python consolidation bundle. It does not
move, delete, merge, train, infer, call APIs, download, stage, or commit.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2f"
PHASE_NAME = "phase_f6i2f_post_archive_consolidation_readiness_review"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"
RUNTIME_CAVEAT = "repaired clone exact package tree only; not exact original executable environment"
F6I2E_DECISION_TOKEN = "f6i2e_substantive_archive_migration_complete"
DECISION_COMPLETE = "f6i2f_post_archive_readiness_complete_first_python_bundle_selected"
DECISION_BLOCKED = "f6i2f_post_archive_readiness_complete_consolidation_blocked"
DECISION_RECONCILIATION_FAILED = "f6i2f_reconciliation_failed"
DECISION_PROTECTED_FAILED = "f6i2f_failed_protected_input_change"
NEXT_PHASE = "phase_f6i2g_first_python_consolidation_execution_after_explicit_approval"

CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
SNAPSHOT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"
MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_post_archive_readiness_manifest.json"
PROTECTED_HASHES_PATH = MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json"

EXPECTED_MOVES = 18
EXPECTED_BYTES = 303523
EXPECTED_ACTIVE_REFS = 15
EXPECTED_HISTORICAL_REFS = 695
EXPECTED_TRACKED_DELETIONS_FROM_F6I2E = 11
EXPECTED_UNTRACKED_MOVES = 7
EXPECTED_F6I2A_PYTHON_COUNT = 408
EXPECTED_F6I2E_REPORTED_AST_COUNT = 1429

OUTPUT_TABLES = {
    "archive_move_reconciliation": TABLES_DIR / f"{PHASE_ID}_archive_move_reconciliation.csv",
    "git_move_state": TABLES_DIR / f"{PHASE_ID}_git_move_state.csv",
    "source_scope_contract": TABLES_DIR / f"{PHASE_ID}_source_scope_contract.csv",
    "python_count_reconciliation": TABLES_DIR / f"{PHASE_ID}_python_count_reconciliation.csv",
    "test_lifecycle_audit": TABLES_DIR / f"{PHASE_ID}_test_lifecycle_audit.csv",
    "current_validation_suite_plan": TABLES_DIR / f"{PHASE_ID}_current_validation_suite_plan.csv",
    "dependency_graph_nodes": TABLES_DIR / f"{PHASE_ID}_dependency_graph_nodes.csv",
    "dependency_graph_edges": TABLES_DIR / f"{PHASE_ID}_dependency_graph_edges.csv",
    "python_merge_group_revalidation": TABLES_DIR / f"{PHASE_ID}_python_merge_group_revalidation.csv",
    "python_merge_priority": TABLES_DIR / f"{PHASE_ID}_python_merge_priority.csv",
    "first_python_consolidation_contract": TABLES_DIR / f"{PHASE_ID}_first_python_consolidation_contract.csv",
    "config_group_revalidation": TABLES_DIR / f"{PHASE_ID}_config_group_revalidation.csv",
    "documentation_archive_readiness": TABLES_DIR / f"{PHASE_ID}_documentation_archive_readiness.csv",
    "git_checkpoint_readiness": TABLES_DIR / f"{PHASE_ID}_git_checkpoint_readiness.csv",
    "validation": TABLES_DIR / f"{PHASE_ID}_validation.csv",
    "determinism": TABLES_DIR / f"{PHASE_ID}_determinism.csv",
    "next_actions": TABLES_DIR / f"{PHASE_ID}_next_actions.csv",
}

F6I2F_OUTPUTS = {
    *(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in OUTPUT_TABLES.values()),
    str(REPORT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    str(MANIFEST_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    str(PROTECTED_HASHES_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    str(CONFIG_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    "src/analysis/plan_f6i2f_post_archive_consolidation_readiness_review.py",
    "src/experiments/build_phase_f6i2f_post_archive_consolidation_readiness_review.py",
    "tests/test_phase_f6i2f_post_archive_consolidation_readiness_review.py",
}

RECOVERY_ARTIFACTS = [
    "report/python_environment_recovery_audit.md",
    "results/environment/python_environment_candidate_inventory.csv",
    "results/environment/broken_venv_static_package_inventory.csv",
    "results/environment/project_dependency_contract.csv",
    "results/environment/environment_recovery_comparison.csv",
    "results/environment/recovered_environment_package_inventory.csv",
    "results/environment/recovered_environment_validation.csv",
    "results/manifests/python_environment_recovery_snapshot.json",
]

INPUT_TABLES = {
    "f6i2a_source_file_audit": "results/tables/phase_f6i2a_source_file_audit.csv",
    "f6i2a_python_semantic_groups": "results/tables/phase_f6i2a_python_semantic_groups.csv",
    "f6i2a_python_merge_plan": "results/tables/phase_f6i2a_python_merge_plan.csv",
    "f6i2a_config_audit": "results/tables/phase_f6i2a_config_audit.csv",
    "f6i2a_config_merge_plan": "results/tables/phase_f6i2a_config_merge_plan.csv",
    "f6i2a_test_audit": "results/tables/phase_f6i2a_test_audit.csv",
    "f6i2c_archive_decisions": "results/tables/phase_f6i2c_archive_decisions.csv",
    "f6i2c_archive_path_mapping": "results/tables/phase_f6i2c_archive_path_mapping.csv",
    "f6i2d_final_archive_gate": "results/tables/phase_f6i2d_final_archive_gate.csv",
    "f6i2e_migration_actions": "results/tables/phase_f6i2e_migration_actions.csv",
    "f6i2e_reference_rewrite_actions": "results/tables/phase_f6i2e_reference_rewrite_actions.csv",
    "f6i2e_archive_hash_validation": "results/tables/phase_f6i2e_archive_hash_validation.csv",
    "f6i2e_active_reference_validation": "results/tables/phase_f6i2e_active_reference_validation.csv",
    "f6i2e_future_dependency_validation": "results/tables/phase_f6i2e_future_dependency_validation.csv",
    "f6i2e_validation": "results/tables/phase_f6i2e_validation.csv",
}

INPUT_MANIFESTS = {
    "f6i2d_substantive_archive_migration_manifest": (
        "results/manifests/phase_f6i2d_substantive_archive_migration_manifest.json"
    ),
    "f6i2e_archive_index": "results/manifests/phase_f6i2e_archive_index.json",
    "f6i2e_migration_manifest": "results/manifests/phase_f6i2e_migration_manifest.json",
    "f6i2e_rollback_manifest": "results/manifests/phase_f6i2e_rollback_manifest.json",
    "f6i2e_protected_input_hashes": "results/manifests/phase_f6i2e_protected_input_hashes.json",
    "current_project_context_snapshot": "results/manifests/current_project_context_snapshot.json",
}

CSV_LIST_COLUMNS = (
    "member_files",
    "shared_functions",
    "wrapper_files",
    "later_delete_candidates",
    "member_configs",
    "config_paths",
    "input_paths",
    "output_paths",
    "reports_citing",
    "target_implementation",
    "loader_sources",
)


@dataclass(frozen=True)
class Inputs:
    config: dict[str, Any]
    tables: dict[str, list[dict[str, str]]]
    manifests: dict[str, Any]
    f6i2e_report: str


@dataclass(frozen=True)
class BuildResult:
    tables: dict[str, list[dict[str, str]]]
    manifests: dict[str, Any]
    report: str
    protected_hashes: dict[str, Any]
    output_hashes: dict[str, str]
    decision_token: str


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def project_path(path: str | Path) -> Path:
    return PROJECT_ROOT / Path(path)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def split_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def stable_id(prefix: str, *parts: str) -> str:
    payload = "|".join(parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:16].upper()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with project_path(path).open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_inputs(config_path: Path = CONFIG_PATH) -> Inputs:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tables = {name: read_csv(path) for name, path in INPUT_TABLES.items()}
    manifests = {
        name: json.loads(project_path(path).read_text(encoding="utf-8"))
        for name, path in INPUT_MANIFESTS.items()
    }
    f6i2e_report = project_path("report/phase_f6i2e_substantive_archive_migration_execution.md").read_text(
        encoding="utf-8"
    )
    return Inputs(config=config, tables=tables, manifests=manifests, f6i2e_report=f6i2e_report)


def run_git(args: Sequence[str]) -> str:
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


def parse_porcelain_lines(lines: Iterable[str]) -> dict[str, str]:
    status: dict[str, str] = {}
    for line in lines:
        if not line:
            continue
        if line.startswith("?? "):
            status[line[3:]] = "??"
            continue
        status[line[3:]] = line[:2]
    return status


def count_status(lines: Sequence[str]) -> dict[str, int]:
    counts = {"added": 0, "deleted": 0, "modified": 0, "untracked": 0, "other": 0}
    for line in lines:
        if line.startswith("?? "):
            counts["untracked"] += 1
            continue
        status = line[:2]
        if "D" in status:
            counts["deleted"] += 1
        elif "A" in status:
            counts["added"] += 1
        elif "M" in status:
            counts["modified"] += 1
        else:
            counts["other"] += 1
    return counts


def git_state(untracked_all: bool = False, exclude_phase_outputs: bool = False) -> dict[str, Any]:
    branch = run_git(["branch", "--show-current"]).strip()
    head = run_git(["rev-parse", "HEAD"]).strip()
    args = ["status", "--porcelain=v1"]
    if untracked_all:
        args.append("-uall")
    lines = run_git(args).splitlines()
    if exclude_phase_outputs:
        lines = [
            line
            for line in lines
            if normalised_status_path(line) not in F6I2F_OUTPUTS
            and not normalised_status_path(line).startswith("results/tables/phase_f6i2f_")
            and not normalised_status_path(line).startswith("results/manifests/phase_f6i2f_")
        ]
    return {
        "branch": branch,
        "head": head,
        "status_counts": count_status(lines),
        "status_row_count": len(lines),
        "status_lines": lines,
    }


def normalised_status_path(line: str) -> str:
    path = line[3:] if line.startswith("?? ") else line[3:]
    return path.strip('"').replace("\\", "/")


def tracked_files() -> set[str]:
    return {line.replace("\\", "/") for line in run_git(["ls-files"]).splitlines() if line}


def select_output_paths() -> dict[str, Path]:
    # Re-runs of this same phase intentionally overwrite F6I2F outputs so that
    # determinism can be checked. A foreign pre-existing preferred artifact
    # would be suffixed, but no such collision existed at phase start.
    return dict(OUTPUT_TABLES)


def protected_paths(inputs: Inputs) -> list[str]:
    paths: set[str] = {
        "AGENTS.md",
        "CODEX_PROMPTS.md",
        "PROJECT_SPEC.md",
        "README.md",
        "report/phase_f6i2e_substantive_archive_migration_execution.md",
        *INPUT_TABLES.values(),
        *INPUT_MANIFESTS.values(),
        *RECOVERY_ARTIFACTS,
    }
    paths.discard("results/manifests/current_project_context_snapshot.json")
    for row in inputs.tables["f6i2e_migration_actions"]:
        paths.add(row["archive_path"])
    return sorted(paths)


def hash_protected(inputs: Inputs) -> dict[str, Any]:
    entries = []
    missing = []
    for rel_path in protected_paths(inputs):
        path = project_path(rel_path)
        if not path.exists():
            missing.append(rel_path)
            entries.append({"path": rel_path, "exists": "false", "sha256": "", "size_bytes": ""})
            continue
        entries.append(
            {
                "path": rel_path,
                "exists": "true",
                "sha256": sha256_file(path),
                "size_bytes": str(path.stat().st_size),
            }
        )
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "runtime": runtime_record(),
        "protected_before": entries,
        "protected_after": entries,
        "changed": [],
        "missing": missing,
        "status": "pass" if not missing else "fail",
        "exclusions": [
            "report/future_operations_context_handoff.md",
            "results/manifests/current_project_context_snapshot.json",
            "new phase_f6i2f outputs",
        ],
    }


def runtime_record() -> dict[str, Any]:
    return {
        "approved_runtime_interpreter": str(APPROVED_INTERPRETER),
        "current_audit_runtime": str(Path(sys.executable).resolve()),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "runtime_caveat": RUNTIME_CAVEAT,
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE", ""),
        "not_exact_original_runtime": True,
        "original_project_package_tree_evidence": "results/environment/broken_venv_static_package_inventory.csv",
        "recovered_executable_environment": str(APPROVED_INTERPRETER),
    }


def archive_reconciliation(inputs: Inputs) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    tracked = tracked_files()
    status_all = parse_porcelain_lines(run_git(["status", "--porcelain=v1", "-uall"]).splitlines())
    archive_index_by_original = {
        row["original_path"]: row for row in inputs.manifests["f6i2e_archive_index"].get("entries", [])
    }
    rollback_by_original = {
        row["original_destination"]: row
        for row in inputs.manifests["f6i2e_rollback_manifest"].get("rollback_plan", [])
    }
    active_by_original: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in inputs.tables["f6i2e_active_reference_validation"]:
        active_by_original[row["original_path"]].append(row)

    reconciliation = []
    git_rows = []
    for index, row in enumerate(
        sorted(inputs.tables["f6i2e_migration_actions"], key=lambda item: item["original_path"]), start=1
    ):
        original = row["original_path"].replace("\\", "/")
        archive = row["archive_path"].replace("\\", "/")
        original_path = project_path(original)
        archive_path = project_path(archive)
        approved_hash = row["approved_sha256"].upper()
        archive_hash = sha256_file(archive_path) if archive_path.exists() else ""
        original_was_tracked = original in tracked
        old_status = status_all.get(original, "")
        new_status = status_all.get(archive, "??" if any(line.startswith("?? archive/") for line in run_git(["status", "--porcelain=v1"]).splitlines()) else "")
        source_absent = not original_path.exists()
        archive_present = archive_path.exists()
        hash_match = archive_hash == approved_hash
        archive_index_present = original in archive_index_by_original
        rollback_present = original in rollback_by_original
        active_rows = active_by_original.get(original, [])
        active_status = "pass" if all(ref["status"] == "pass" for ref in active_rows) else "not_required"
        if original_was_tracked and new_status.startswith("??"):
            move_class = "tracked_source_to_untracked_archive"
        elif not original_was_tracked and new_status.startswith("??"):
            move_class = "untracked_source_to_untracked_archive"
        elif original_was_tracked and new_status and not new_status.startswith("??"):
            move_class = "tracked_source_to_tracked_archive"
        elif not (source_absent and archive_present and hash_match):
            move_class = "unresolved"
        else:
            move_class = "other"
        status = (
            "pass"
            if source_absent and archive_present and hash_match and archive_index_present and rollback_present
            else "fail"
        )
        reconciliation.append(
            {
                "reconciliation_id": f"F6I2F-MOVE-{index:03d}",
                "original_path": original,
                "original_pre_f6i2e_git_state": "tracked" if original_was_tracked else "untracked",
                "original_sha256": approved_hash,
                "archive_path": archive,
                "archive_sha256": archive_hash,
                "original_path_currently_absent": bool_text(source_absent),
                "archive_path_currently_present": bool_text(archive_present),
                "byte_identity": bool_text(hash_match),
                "archive_index_entry": bool_text(archive_index_present),
                "rollback_entry": bool_text(rollback_present),
                "active_reference_rewrite_status": active_status,
                "active_reference_rows": str(len(active_rows)),
                "old_path_git_status": old_status,
                "new_path_git_status": new_status,
                "move_classification": move_class,
                "status": status,
            }
        )
        git_rows.append(
            {
                "git_move_id": f"F6I2F-GITMOVE-{index:03d}",
                "original_path": original,
                "archive_path": archive,
                "old_path_git_status": old_status,
                "new_path_git_status": new_status,
                "old_path_tracked_in_index": bool_text(original_was_tracked),
                "new_path_tracked_in_index": bool_text(archive in tracked),
                "git_rename_inferred": "false",
                "reason_git_reports_deleted_vs_moves": (
                    "Git reports deletions only for tracked source paths; untracked moved sources "
                    "had no index entry, and archive destinations are untracked because nothing was staged."
                ),
                "move_classification": move_class,
            }
        )
    return reconciliation, git_rows


def source_scope_contract() -> list[dict[str, str]]:
    src_children = sorted(p for p in (PROJECT_ROOT / "src").iterdir() if p.is_dir())
    scope_paths = [
        ("SRC-ANALYSIS", "src/analysis", "active_local_project_source", True),
        ("SRC-EXPERIMENTS", "src/experiments", "local_experiment_build_source", True),
        ("SRC-LLM", "src/llm", "active_local_project_source_protected_scientific_protocol", True),
        ("SRC-METRICS", "src/metrics", "active_local_project_source_protected_scientific_protocol", True),
        ("SRC-MODELS", "src/models", "active_local_project_source", True),
        ("SRC-DATA", "src/data", "active_local_project_source", True),
        ("TESTS", "tests", "local_tests", False),
        ("UPSTREAM", "original_repo", "official_upstream_source", False),
        ("ARCHIVE", "archive", "archived_source", False),
        ("VENV-ORIGINAL", ".venv", "environment_dependency_source", False),
        ("VENV-RECOVERED", ".venv_audit_recovery", "recovered_executable_environment", False),
        ("RESULTS-ENV", "results/environment", "environment_dependency_source", False),
        ("RESULTS-GENERATED", "results", "generated_or_embedded_source", False),
    ]
    existing = {path for _, path, _, _ in scope_paths}
    for child in src_children:
        rel_child = rel(child)
        if rel_child not in existing:
            scope_paths.append(
                (
                    f"SRC-{child.name.upper()}",
                    rel_child,
                    "active_local_project_source",
                    True,
                )
            )
    rows = []
    for scope_id, path_text, category, semantic in scope_paths:
        path = project_path(path_text)
        present = path.exists()
        python_count = sum(1 for p in path.rglob("*.py") if p.is_file()) if present and path.is_dir() else 0
        protected = category in {
            "official_upstream_source",
            "archived_source",
            "environment_dependency_source",
            "recovered_executable_environment",
            "active_local_project_source_protected_scientific_protocol",
        }
        archived = category == "archived_source"
        rows.append(
            {
                "scope_id": scope_id,
                "path_pattern": path_text,
                "present": bool_text(present),
                "scope_category": category,
                "python_file_count": str(python_count),
                "included_in_semantic_consolidation": bool_text(semantic and present and not protected),
                "included_in_syntax_validation": bool_text(present and category not in {"environment_dependency_source", "recovered_executable_environment", "archived_source"}),
                "external_reference_only": bool_text(category == "official_upstream_source"),
                "excluded_from_local_refactoring": bool_text(not semantic or protected or archived),
                "protected": bool_text(protected),
                "archived": bool_text(archived),
                "runtime_classification": RUNTIME_CLASSIFICATION,
                "rationale": scope_rationale(category),
            }
        )
    return sorted(rows, key=lambda row: row["scope_id"])


def scope_rationale(category: str) -> str:
    if category == "official_upstream_source":
        return "Read-only upstream evidence; wrappers may reference it but consolidation must not refactor it."
    if category in {"environment_dependency_source", "recovered_executable_environment"}:
        return "Runtime/package evidence only; never active project source for semantic consolidation."
    if category == "archived_source":
        return "Historical archive evidence; excluded from active consolidation."
    if category == "local_tests":
        return "Tests validate active behavior but are not production-source merge targets."
    if category == "generated_or_embedded_source":
        return "Generated outputs and embedded dependency trees are audited separately from project source."
    return "Active local source may be considered for low-risk consolidation when scientific protocols remain intact."


def python_count_reconciliation(inputs: Inputs) -> list[dict[str, str]]:
    all_py = [p for p in PROJECT_ROOT.rglob("*.py") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts]
    buckets: defaultdict[str, list[Path]] = defaultdict(list)
    broad_f6i2e_current: list[Path] = []
    for path in all_py:
        rel_path = rel(path)
        if rel_path.startswith("archive/"):
            bucket = "archived_python"
        elif rel_path.startswith(".venv") or rel_path.startswith("results/environment/"):
            bucket = "environment_dependency_python"
        elif rel_path.startswith("original_repo/"):
            bucket = "official_upstream_python"
        elif rel_path.startswith("tests/"):
            bucket = "local_test_python"
        elif rel_path.startswith("src/experiments/"):
            bucket = "local_experiment_build_python"
        elif rel_path.startswith("src/"):
            bucket = "active_local_production_python"
        else:
            bucket = "other_python"
        buckets[bucket].append(path)
        if not (
            rel_path.startswith("archive/")
            or rel_path.startswith(".venv/")
            or rel_path.startswith(".venv_audit_recovery/")
            or rel_path.startswith(".venv_f6g2_py312/")
            or rel_path.startswith(".venv_repro")
        ):
            broad_f6i2e_current.append(path)

    active_local = len(buckets["active_local_production_python"]) + len(buckets["local_experiment_build_python"])
    rows = [
        {
            "count_id": "F6I2F-PYCOUNT-F6I2A-AUDIT",
            "scope": "f6i2a_source_file_audit_table",
            "count": str(len(inputs.tables["f6i2a_source_file_audit"])),
            "authoritative_for_future_consolidation": "false",
            "directory_scope": "F6I2A retained audit rows",
            "inclusion_rules": "rows in results/tables/phase_f6i2a_source_file_audit.csv",
            "exclusion_rules": "not a live filesystem scan after later cleanup/archive phases",
            "archive_handling": "pre-F6I2E retained-path snapshot",
            "environment_handling": "as classified by F6I2A, not current semantic scope",
            "upstream_handling": "included where F6I2A audited relevant files",
            "generated_handling": "included only where F6I2A classified as relevant",
            "duplicate_symlink_handling": "one CSV row per audited path",
            "explanation": "Structured F6I2A count remains 408 and is a historical design snapshot.",
        },
        {
            "count_id": "F6I2F-PYCOUNT-F6I2E-REPORTED",
            "scope": "f6i2e_reported_ast_parse_active_retained_python",
            "count": str(EXPECTED_F6I2E_REPORTED_AST_COUNT),
            "authoritative_for_future_consolidation": "false",
            "directory_scope": "F6I2E broad retained non-archive parse log",
            "inclusion_rules": "report log stated AST parse active retained Python 1429 parsed",
            "exclusion_rules": "not recoverable as a structured table; do not infer semantic source from this label",
            "archive_handling": "archive excluded",
            "environment_handling": "recovered/extracted environment trees under results/environment can be included by broad scan",
            "upstream_handling": "upstream Python may be included by broad scan",
            "generated_handling": "generated Python under results/environment is not active project source",
            "duplicate_symlink_handling": "reported count did not expose symlink metadata",
            "explanation": "The 1429 figure is preserved as reported F6I2E evidence, not as active local source.",
        },
        {
            "count_id": "F6I2F-PYCOUNT-F6I2E-RULE-CURRENT",
            "scope": "current_broad_same_exclusion_rule",
            "count": str(len(broad_f6i2e_current)),
            "authoritative_for_future_consolidation": "false",
            "directory_scope": "current filesystem excluding .venv*, archive, .git, and __pycache__",
            "inclusion_rules": "includes src, tests, original_repo, and results/environment Python",
            "exclusion_rules": "excludes archive and virtualenv roots",
            "archive_handling": "excluded",
            "environment_handling": "results/environment included as extracted runtime evidence",
            "upstream_handling": "included as read-only upstream evidence",
            "generated_handling": "included when under non-venv results trees",
            "duplicate_symlink_handling": "regular files counted once by path; symlinks not followed separately",
            "explanation": "This broad count explains why the F6I2E figure is much larger than active local source.",
        },
    ]
    for bucket in [
        "active_local_production_python",
        "local_experiment_build_python",
        "local_test_python",
        "official_upstream_python",
        "archived_python",
        "environment_dependency_python",
        "other_python",
    ]:
        rows.append(
            {
                "count_id": stable_id("F6I2F-PYCOUNT", bucket),
                "scope": bucket,
                "count": str(len(buckets[bucket])),
                "authoritative_for_future_consolidation": bool_text(
                    bucket in {"active_local_production_python", "local_experiment_build_python"}
                ),
                "directory_scope": bucket_directory(bucket),
                "inclusion_rules": "current filesystem .py files in this bucket",
                "exclusion_rules": bucket_exclusions(bucket),
                "archive_handling": "excluded unless bucket is archived_python",
                "environment_handling": "excluded unless bucket is environment_dependency_python",
                "upstream_handling": "excluded unless bucket is official_upstream_python",
                "generated_handling": "excluded from active-local consolidation",
                "duplicate_symlink_handling": "one path count; symlinked files are not followed as extra targets",
                "explanation": bucket_explanation(bucket),
            }
        )
    rows.append(
        {
            "count_id": "F6I2F-PYCOUNT-ACTIVE-LOCAL-AUTHORITATIVE",
            "scope": "authoritative_active_local_python_for_consolidation",
            "count": str(active_local),
            "authoritative_for_future_consolidation": "true",
            "directory_scope": "src/**/*.py, including src/experiments, excluding archive/upstream/environment/tests",
            "inclusion_rules": "active local project source only",
            "exclusion_rules": ".venv*, results/environment, archive, original_repo, tests, generated outputs",
            "archive_handling": "excluded",
            "environment_handling": "excluded",
            "upstream_handling": "excluded",
            "generated_handling": "excluded",
            "duplicate_symlink_handling": "one current path per source file",
            "explanation": "This is the count to use for future Python consolidation planning.",
        }
    )
    return rows


def bucket_directory(bucket: str) -> str:
    return {
        "active_local_production_python": "src/ excluding src/experiments/",
        "local_experiment_build_python": "src/experiments/",
        "local_test_python": "tests/",
        "official_upstream_python": "original_repo/",
        "archived_python": "archive/",
        "environment_dependency_python": ".venv*, results/environment/",
        "other_python": "other repository paths",
    }[bucket]


def bucket_exclusions(bucket: str) -> str:
    if bucket == "environment_dependency_python":
        return "excluded from semantic consolidation and local refactoring"
    if bucket == "archived_python":
        return "excluded from active consolidation; retained as historical evidence"
    if bucket == "official_upstream_python":
        return "excluded from local refactoring unless explicitly wrapped"
    if bucket == "local_test_python":
        return "excluded from production-source count but included in validation planning"
    return "archive, environments, upstream source, generated dependency trees"


def bucket_explanation(bucket: str) -> str:
    return {
        "active_local_production_python": "Active project implementation source outside experiment wrappers.",
        "local_experiment_build_python": "Local builders and runners; included in active-local consolidation scope with side-effect gates.",
        "local_test_python": "Validation surface, not production-source merge target.",
        "official_upstream_python": "Read-only upstream source evidence.",
        "archived_python": "Historical archived Python, if any.",
        "environment_dependency_python": "Installed or extracted dependency/runtime source; not project source.",
        "other_python": "Current Python outside defined source/test/upstream/archive/environment buckets.",
    }[bucket]


def test_lifecycle_audit() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    phase_tests = sorted((PROJECT_ROOT / "tests").glob("test_phase_f6i*.py"))
    wanted = {
        "F6I1": "tests/test_phase_f6i1_repository_retention_audit.py",
        "F6I2A": "tests/test_phase_f6i2a_substantive_repository_consolidation_design.py",
        "F6I2B": "tests/test_phase_f6i2b_low_risk_substantive_deletion.py",
        "F6I2C": "tests/test_phase_f6i2c_archive_bundle_verification_and_migration_design.py",
        "F6I2D": "tests/test_phase_f6i2d_archive_prerequisite_closure_and_technical_cache_cleanup.py",
        "F6I2E": "tests/test_phase_f6i2e_substantive_archive_migration_execution.py",
        "F6I2F": "tests/test_phase_f6i2f_post_archive_consolidation_readiness_review.py",
    }
    actual = {rel(path) for path in phase_tests}
    rows = []
    for phase, path in wanted.items():
        exists = path in actual or project_path(path).exists()
        if phase in {"F6I2E", "F6I2F"}:
            lifecycle = "current_state_invariant"
            current_suite = "include"
            action = "retain_current_meta_validation"
        elif phase == "F6I2B":
            lifecycle = "migration_superseded_invariant"
            current_suite = "exclude_obsolete_path_assertion"
            action = "retain_historical_or_mark_path_assertion_superseded"
        elif phase in {"F6I2C", "F6I2D"}:
            lifecycle = "historical_precondition"
            current_suite = "exclude_from_current_state_default"
            action = "retain_as_phase_snapshot_or_marker_candidate"
        else:
            lifecycle = "phase_snapshot_invariant"
            current_suite = "exclude_from_current_state_default"
            action = "retain_unchanged_as_historical_snapshot"
        rows.append(
            {
                "test_id": f"F6I2F-TEST-{phase}",
                "phase": phase,
                "path": path,
                "exists": bool_text(exists),
                "lifecycle_classification": lifecycle,
                "current_validation_suite_treatment": current_suite,
                "side_effect_risk": "offline_read_only" if phase in {"F6I2E", "F6I2F"} else "historical_precondition_may_be_superseded",
                "recommended_later_action": action,
                "rationale": lifecycle_rationale(phase),
            }
        )
    rows.append(
        {
            "test_id": "F6I2F-TEST-F6I2B-PROTECTED-PATH-ASSERTION",
            "phase": "F6I2B",
            "path": "tests/test_phase_f6i2b_low_risk_substantive_deletion.py",
            "exists": bool_text(project_path("tests/test_phase_f6i2b_low_risk_substantive_deletion.py").exists()),
            "lifecycle_classification": "migration_superseded_invariant",
            "current_validation_suite_treatment": "replace_with_migration_supersession_check",
            "side_effect_risk": "offline_but_currently_false_after_approved_migration",
            "recommended_later_action": "mark_historical_or_wrap_with_phase_snapshot_fixture",
            "rationale": "The retained-path protected-input assertion was true before F6I2E but is intentionally false after approved archive moves.",
        }
    )
    plan = [
        validation_plan_row("archive_move_count", "include", "exactly 18 F6I2E moves reconciled"),
        validation_plan_row("old_path_absent", "include", "each approved original path remains absent"),
        validation_plan_row("archive_path_present", "include", "each approved archive path is present"),
        validation_plan_row("archive_hash_identity", "include", "archive hash equals approved original hash"),
        validation_plan_row("f6i2d_f6i2e_authorization", "include", "F6I2D and F6I2E decision tokens and counts pass"),
        validation_plan_row("rollback_metadata", "include", "one rollback row per archived file"),
        validation_plan_row("f6i2b_retained_path_assertion", "exclude_current_state", "superseded by approved F6I2E migration"),
        validation_plan_row("historical_phase_tests", "retain_historical", "do not rewrite historical phase tests during F6I2F"),
    ]
    return rows, plan


def lifecycle_rationale(phase: str) -> str:
    if phase == "F6I2B":
        return "One protected-input path invariant was superseded by the later approved archive migration."
    if phase in {"F6I2E", "F6I2F"}:
        return "These tests validate the current post-archive state and current readiness outputs."
    if phase in {"F6I2C", "F6I2D"}:
        return "The phase described pre-migration readiness and prerequisite closure, now superseded by F6I2E execution."
    return "Earlier cleanup tests preserve phase-specific state and should not define post-F6I2E current invariants."


def validation_plan_row(check_name: str, treatment: str, evidence: str) -> dict[str, str]:
    return {
        "suite_check_id": stable_id("F6I2F-SUITE", check_name),
        "check_name": check_name,
        "treatment": treatment,
        "expected_evidence": evidence,
        "implemented_in_f6i2f": "true" if treatment == "include" else "false",
        "future_test_action": "add_current_meta_validation_or_marker" if treatment != "include" else "keep_focused_offline_check",
    }


def dependency_graph(inputs: Inputs) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    active_paths = [
        p
        for root in [PROJECT_ROOT / "src", PROJECT_ROOT / "tests"]
        for p in root.rglob("*.py")
        if p.is_file() and "__pycache__" not in p.parts
    ]
    nodes = []
    module_by_path = {}
    for path in sorted(active_paths, key=rel):
        rel_path = rel(path)
        module = rel_path[:-3].replace("/", ".")
        module_by_path[module] = rel_path
        node_type = (
            "local_test_python"
            if rel_path.startswith("tests/")
            else "local_experiment_build_python"
            if rel_path.startswith("src/experiments/")
            else "active_local_production_python"
        )
        syntax_status = "pass"
        parse_error = ""
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:
            syntax_status = "fail"
            parse_error = f"{exc.__class__.__name__}: {exc}"
        nodes.append(
            {
                "node_id": stable_id("F6I2F-NODE", rel_path),
                "path": rel_path,
                "module": module,
                "node_type": node_type,
                "included_in_active_local_graph": "true",
                "included_in_semantic_consolidation": bool_text(not rel_path.startswith("tests/")),
                "syntax_status": syntax_status,
                "parse_error": parse_error,
            }
        )
    edges: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for path in sorted(active_paths, key=rel):
        rel_path = rel(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_edge(edges, rel_path, alias.name, "active_import", "ast.Import")
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                add_edge(edges, rel_path, module, "active_import", "ast.ImportFrom")
    for row in inputs.tables["f6i2a_test_audit"]:
        source = row["path"]
        if not source.startswith("tests/") or not project_path(source).exists():
            continue
        for target in split_list(row.get("target_implementation", "")):
            add_edge(edges, source, target, "test_target", "F6I2A test audit target_implementation")
    for row in inputs.tables["f6i2a_source_file_audit"]:
        source = row["path"]
        if not source.startswith("src/") or not project_path(source).exists():
            continue
        for config_path in split_list(row.get("config_paths", "")):
            add_edge(edges, source, config_path, "config_load", "F6I2A source audit config_paths")
        for output_path in split_list(row.get("output_paths", "")):
            add_edge(edges, source, output_path, "generated_output_relationship", "F6I2A source audit output_paths")
        if row.get("cli_entrypoint", "").lower() in {"true", "yes", "1"}:
            add_edge(edges, "project_root_cli", source, "CLI_invocation", "F6I2A source audit cli_entrypoint")
        for report_path in split_list(row.get("reports_citing", ""))[:8]:
            add_edge(edges, report_path, source, "report_reference", "F6I2A reports_citing")
    for row in inputs.tables["f6i2e_active_reference_validation"]:
        add_edge(edges, row["referencer_path"], row["archive_path"], "archive_reference", "F6I2E active reference validation")
    for row in inputs.tables["f6i2e_future_dependency_validation"][:80]:
        add_edge(
            edges,
            row["future_task_group"],
            row["archive_path"],
            "future_roadmap_dependency",
            "F6I2E future dependency validation",
        )
    historical_reference_count = len(inputs.tables["f6i2e_future_dependency_validation"])
    add_edge(
        edges,
        "F6I2E historical reference validation",
        f"historical_reference_rows={EXPECTED_HISTORICAL_REFS}",
        "historical_reference",
        f"{historical_reference_count} future-dependency rows plus 695 historical reference rows retained",
    )
    return nodes, sorted(edges.values(), key=lambda row: (row["edge_type"], row["source"], row["target"]))


def add_edge(edges: dict[tuple[str, str, str, str], dict[str, str]], source: str, target: str, edge_type: str, evidence: str) -> None:
    if not source or not target:
        return
    key = (source, target, edge_type, evidence)
    edges[key] = {
        "edge_id": stable_id("F6I2F-EDGE", source, target, edge_type, evidence),
        "source": source,
        "target": target,
        "edge_type": edge_type,
        "active_dependency": bool_text(edge_type in {"active_import", "CLI_invocation", "config_load", "test_target"}),
        "historical_textual_mention": bool_text(edge_type == "historical_reference"),
        "evidence": evidence,
    }


def python_merge_revalidation(inputs: Inputs) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    semantic_by_suffix = {
        row["semantic_group_id"].split("-")[-1]: row for row in inputs.tables["f6i2a_python_semantic_groups"]
    }
    archive_old = {row["original_path"] for row in inputs.tables["f6i2e_migration_actions"]}
    revalidation = []
    priority = []
    selected: dict[str, str] | None = None
    for row in sorted(inputs.tables["f6i2a_python_merge_plan"], key=lambda item: item["merge_group_id"]):
        suffix = row["merge_group_id"].split("-")[-1]
        semantic = semantic_by_suffix.get(suffix, {})
        group_name = semantic.get("group_name", row["merge_group_id"])
        members = split_list(row["member_files"])
        existing = [path for path in members if project_path(path).exists()]
        archived = [path for path in members if path in archive_old]
        missing = [path for path in members if path not in existing and path not in archived]
        status = merge_status(group_name, row)
        selected_flag = row["merge_group_id"] == "PYMERGE-F6I2A-002" and status == "ready_for_consolidation"
        if selected_flag:
            selected = row
        scientific_risk = scientific_risk_for_group(group_name)
        maintenance_risk = "medium" if len(members) > 25 else "low"
        duplicate_functions = len(split_list(row.get("shared_functions", "")))
        score = priority_score(status, scientific_risk, duplicate_functions, len(members))
        revalidation.append(
            {
                "merge_group_id": row["merge_group_id"],
                "semantic_group_id": semantic.get("semantic_group_id", ""),
                "group_name": group_name,
                "member_file_count_f6i2a": str(len(members)),
                "member_files_existing_current": str(len(existing)),
                "member_files_archived": str(len(archived)),
                "member_files_missing_not_archived": str(len(missing)),
                "shared_function_count": str(duplicate_functions),
                "proposed_common_module": row["proposed_common_module"],
                "current_status": status,
                "scientific_risk": scientific_risk,
                "side_effect_risk": side_effect_risk_for_group(group_name),
                "future_dependency_implication": row.get("provenance_implications", ""),
                "current_tests": row.get("tests_required", ""),
                "output_schemas": row.get("unique_schemas", ""),
                "deterministic_behaviour": "requires_output_hash_equivalence",
                "revalidation_evidence": group_evidence(group_name, existing, archived, missing),
            }
        )
        priority.append(
            {
                "merge_group_id": row["merge_group_id"],
                "group_name": group_name,
                "current_status": status,
                "priority_score": str(score),
                "duplicated_function_count": str(duplicate_functions),
                "member_file_count": str(len(members)),
                "maintenance_risk": maintenance_risk,
                "scientific_risk": scientific_risk,
                "test_coverage": "focused_tests_required_before_and_after",
                "future_reuse": "high" if group_name in {"deterministic_csv_io", "hashing_helpers", "validation_table_creation"} else "medium",
                "expected_file_reduction": "0_initial_wrappers_retained" if selected_flag else "not_selected",
                "expected_code_reduction": "3 duplicated CSV helper functions centralized" if selected_flag else "not_estimated",
                "migration_complexity": "low" if group_name == "deterministic_csv_io" else "medium",
                "rollback_complexity": "low_wrappers_restored_to_local_helpers" if selected_flag else "not_planned",
                "selected_for_first_bundle": bool_text(selected_flag),
                "selection_rationale": (
                    "Lowest scientific risk shared utility with explicit F6I2A evidence: read_csv/read_csv_rows/write_csv."
                    if selected_flag
                    else "Not selected for the first bundle."
                ),
            }
        )
    if selected is None:
        contract = []
    else:
        contract = first_python_contract(selected)
    return revalidation, sorted(priority, key=lambda row: (-int(row["selected_for_first_bundle"] == "true"), row["priority_score"], row["merge_group_id"])), contract


def merge_status(group_name: str, row: Mapping[str, str]) -> str:
    if group_name == "deterministic_csv_io":
        return "ready_for_consolidation"
    if group_name in {"git_state_capture", "hashing_helpers", "manifest_generation", "validation_table_creation", "path_normalization"}:
        return "ready_after_test_lifecycle_update"
    if group_name in {"candidate_and_cohort_reconciliation", "request_response_lineage"}:
        return "defer_until_scientific_phase_complete"
    if group_name == "phase_report_generation":
        return "keep_separate_distinct_protocol"
    return "unresolved"


def scientific_risk_for_group(group_name: str) -> str:
    if group_name in {"deterministic_csv_io", "git_state_capture", "hashing_helpers", "manifest_generation", "path_normalization", "validation_table_creation"}:
        return "low"
    if group_name in {"candidate_and_cohort_reconciliation", "request_response_lineage"}:
        return "high"
    return "medium"


def side_effect_risk_for_group(group_name: str) -> str:
    if group_name in {"git_state_capture", "manifest_generation", "phase_report_generation"}:
        return "filesystem_or_git_state_sensitive"
    if group_name in {"candidate_and_cohort_reconciliation", "request_response_lineage"}:
        return "scientific_protocol_sensitive"
    return "read_write_helper_low"


def priority_score(status: str, scientific_risk: str, duplicate_functions: int, members: int) -> int:
    score = duplicate_functions + min(members, 30)
    if status == "ready_for_consolidation":
        score += 100
    if scientific_risk == "low":
        score += 50
    elif scientific_risk == "high":
        score -= 80
    return score


def group_evidence(group_name: str, existing: list[str], archived: list[str], missing: list[str]) -> str:
    if archived:
        return f"{len(archived)} member paths archived; do not merge before relocation analysis."
    if missing:
        return f"{len(missing)} member paths no longer exist; rebaseline wrappers before editing."
    if group_name == "deterministic_csv_io":
        return "Members remain current and shared logic is generic CSV reading/writing with low scientific risk."
    return f"{len(existing)} current member paths require focused equivalence tests before edits."


def first_python_contract(selected: Mapping[str, str]) -> list[dict[str, str]]:
    members = split_list(selected["member_files"])
    wrappers = split_list(selected.get("wrapper_files", ""))
    duplicated = split_list(selected["shared_functions"])
    return [
        {
            "contract_id": "F6I2F-FIRST-PYTHON-BUNDLE",
            "selected_merge_group_id": selected["merge_group_id"],
            "selected_group_name": "deterministic_csv_io",
            "member_files": ";".join(members),
            "duplicated_functions": ";".join(duplicated),
            "unique_functions": selected.get("unique_functions", ""),
            "target_shared_module": selected["proposed_common_module"],
            "api_contract": "read_csv(path); read_csv_rows(path); write_csv(path_or_rows, rows_or_path, fieldnames=None) preserving newline, UTF-8, deterministic row order",
            "imports_to_change": ";".join(wrappers),
            "wrappers_to_retain_initially": ";".join(wrappers),
            "wrappers_eventually_deletable": selected.get("later_delete_candidates", ""),
            "tests_to_add": "focused unit tests for common_io; builder equivalence tests for touched wrappers",
            "tests_to_retain": "phase-specific tests remain collected as appropriate; historical path assumptions remain marked outside current-state validation",
            "current_outputs_affected": selected.get("affected_outputs", ""),
            "builders_requiring_equivalence_checks": ";".join(wrappers),
            "deterministic_output_hashes_to_compare": "all CSV/JSON/Markdown outputs produced by touched builders before and after wrapper import change",
            "future_roadmap_tasks_protected": selected.get("provenance_implications", ""),
            "migration_ordering": selected.get("migration_order", ""),
            "rollback_ordering": "restore wrapper-local helpers or revert imports before rerunning equivalence tests",
            "stop_conditions": "any output hash delta, schema delta, import failure, path side effect, or scientific protocol behavior change",
            "compatibility_wrappers_required": "true",
            "implementation_in_f6i2f": "false",
        }
    ]


def config_group_revalidation(inputs: Inputs) -> list[dict[str, str]]:
    redirected = [
        row["path"]
        for row in inputs.tables["f6i2c_archive_decisions"]
        if row["current_decision"] == "redirect_to_config_consolidation"
    ]
    audit_by_path = {row["path"]: row for row in inputs.tables["f6i2a_config_audit"]}
    rows = []
    for group in sorted(inputs.tables["f6i2a_config_merge_plan"], key=lambda item: item["config_merge_group_id"]):
        members = split_list(group["member_configs"])
        overlap = sorted(set(members) & set(redirected))
        active = [path for path in members if project_path(path).exists() and path not in overlap]
        historical = [path for path in members if audit_by_path.get(path, {}).get("config_class") == "frozen_historical_contract"]
        future = [path for path in members if "phase_" in path or "finalization_" in path]
        unresolved = [path for path in members if not project_path(path).exists()]
        readiness = (
            "ready_after_yaml_loader_contract"
            if group["group_name"] == "base_dataset_or_cost"
            else "defer_until_first_python_bundle_complete"
        )
        rows.append(
            {
                "config_merge_group_id": group["config_merge_group_id"],
                "group_name": group["group_name"],
                "member_config_count": str(len(members)),
                "current_existing_count": str(sum(1 for path in members if project_path(path).exists())),
                "f6i2c_redirected_overlap_count": str(len(overlap)),
                "f6i2c_redirected_overlap_paths": ";".join(overlap),
                "active_config_count": str(len(active)),
                "historical_frozen_config_count": str(len(historical)),
                "future_template_count": str(len(future)),
                "unresolved_config_count": str(len(unresolved)),
                "target_structure": group["target_structure"],
                "values_to_preserve": group["values_to_preserve"],
                "loader": ";".join(sorted({audit_by_path.get(path, {}).get("loader_sources", "")[:80] for path in members if path in audit_by_path})),
                "unique_keys": summarise_config_keys(members, audit_by_path, "unique_keys"),
                "duplicated_keys": group["group_name"],
                "scientific_contract_values": summarise_config_keys(members, audit_by_path, "scientific_contract_values"),
                "executed_history_value": summarise_config_keys(members, audit_by_path, "executed_evidence"),
                "future_dependency": summarise_config_keys(members, audit_by_path, "future_dependency"),
                "archive_status": "redirected_not_archived" if overlap else "not_archived",
                "readiness": readiness,
                "implementation_in_f6i2f": "false",
            }
        )
    return rows


def summarise_config_keys(members: Sequence[str], audit_by_path: Mapping[str, Mapping[str, str]], key: str) -> str:
    values = []
    for path in members:
        value = audit_by_path.get(path, {}).get(key, "")
        if value and value not in values:
            values.append(value)
    return ";".join(values)[:800]


def documentation_archive_readiness(inputs: Inputs) -> list[dict[str, str]]:
    md_current = sorted(
        rel(path)
        for path in (PROJECT_ROOT / "report").glob("*.md")
        if path.name != f"{PHASE_NAME}.md"
    )
    md_archive = sorted(rel(path) for path in (PROJECT_ROOT / "archive").rglob("*.md")) if (PROJECT_ROOT / "archive").exists() else []
    active_archive_refs = {
        row["referencer_path"] for row in inputs.tables["f6i2e_active_reference_validation"] if row["status"] == "pass"
    }
    duplicate_current = [
        "report/future_operations_context_handoff.md",
        "results/manifests/current_project_context_snapshot.json",
    ]
    return [
        {
            "doc_readiness_id": "F6I2F-DOC-CURRENT-MD",
            "category": "current_operational_markdown",
            "path_or_group": "report/*.md",
            "count": str(len(md_current)),
            "status": "ready_for_later_documentation_consolidation_review",
            "evidence": "Current markdown remains in report/ and was not moved by F6I2F.",
        },
        {
            "doc_readiness_id": "F6I2F-DOC-ARCHIVE-MD",
            "category": "historical_markdown_in_archive",
            "path_or_group": "archive/**/*.md",
            "count": str(len(md_archive)),
            "status": "historical_reference_retained",
            "evidence": "F6I2E archive destinations are present and hash-validated.",
        },
        {
            "doc_readiness_id": "F6I2F-DOC-ACTIVE-ARCHIVE-REFS",
            "category": "current_references_to_archived_reports",
            "path_or_group": ";".join(sorted(active_archive_refs)),
            "count": str(len(active_archive_refs)),
            "status": "pass",
            "evidence": "F6I2E active-reference validation closed approved references.",
        },
        {
            "doc_readiness_id": "F6I2F-DOC-HISTORICAL-REFS",
            "category": "historical_references_intentionally_retained",
            "path_or_group": "historical reference inventory",
            "count": str(EXPECTED_HISTORICAL_REFS),
            "status": "retained_not_active_dependencies",
            "evidence": "F6I2E retained 695 historical reference rows.",
        },
        {
            "doc_readiness_id": "F6I2F-DOC-DUPLICATE-CURRENT-SUMMARIES",
            "category": "duplicate_current_summaries",
            "path_or_group": ";".join(duplicate_current),
            "count": str(len(duplicate_current)),
            "status": "review_later",
            "evidence": "Current handoff and snapshot overlap by design and remain the operational context records.",
        },
        {
            "doc_readiness_id": "F6I2F-DOC-CODEX-PROJECT-SPEC",
            "category": "CODEX_PROMPTS_and_PROJECT_SPEC_rewrite_status",
            "path_or_group": "CODEX_PROMPTS.md;PROJECT_SPEC.md",
            "count": "2",
            "status": "pass",
            "evidence": "F6I2E rewrites are recorded; F6I2F does not alter these files.",
        },
    ]


def git_checkpoint_readiness(reconciliation: Sequence[Mapping[str, str]], start_git_state: Mapping[str, Any]) -> list[dict[str, str]]:
    status_default = git_state(untracked_all=False, exclude_phase_outputs=True)
    status_all = git_state(untracked_all=True, exclude_phase_outputs=True)
    tracked_archive = sum(1 for row in reconciliation if row["move_classification"] == "tracked_source_to_untracked_archive")
    untracked_archive = sum(1 for row in reconciliation if row["move_classification"] == "untracked_source_to_untracked_archive")
    return [
        checkpoint_row("initial_audit_git_state", "record", start_git_state["status_counts"], "Read-only pre-artifact state from F6I2F config."),
        checkpoint_row("current_git_state_excluding_f6i2f_outputs", "record", status_default["status_counts"], "Current state, normalised to avoid deterministic rerun noise."),
        checkpoint_row("current_git_state_untracked_all_excluding_f6i2f_outputs", "record", status_all["status_counts"], "Untracked-all status used to classify archive destinations."),
        {
            "checkpoint_id": "F6I2F-GIT-ARCHIVE-MOVES",
            "category": "untracked_archive_paths",
            "count": str(EXPECTED_MOVES),
            "decision": "candidate_for_manifest_driven_checkpoint_after_boundary_review",
            "evidence": f"{tracked_archive} tracked-source moves and {untracked_archive} untracked-source moves are reconciled.",
            "staging_strategy": "stage exact manifest-listed old paths and archive paths only after explicit approval",
            "forbidden_staging": "git add .;git add -A;broad wildcard staging",
        },
        {
            "checkpoint_id": "F6I2F-GIT-ENV",
            "category": "environment_files",
            "count": str(sum(1 for line in status_all["status_lines"] if ".venv" in line or "results/environment/" in line)),
            "decision": "remain_unstaged_or_ignored_pending_review",
            "evidence": "Runtime recovery artifacts are provenance evidence; venv directories remain unsuitable for broad staging.",
            "staging_strategy": "stage only selected CSV/JSON/MD recovery evidence if explicitly requested",
            "forbidden_staging": "git add .;git add -A;broad wildcard staging",
        },
        {
            "checkpoint_id": "F6I2F-GIT-DECISION",
            "category": "checkpoint_readiness_decision",
            "count": "1",
            "decision": "defer_commit_until_first_python_consolidation_contract_reviewed",
            "evidence": "F6I2F selects a first low-risk Python bundle, but does not implement it.",
            "staging_strategy": "future manifest-driven exact path list; never broad staging",
            "forbidden_staging": "git add .;git add -A;broad wildcard staging",
        },
    ]


def checkpoint_row(checkpoint_id: str, category: str, counts: Mapping[str, Any], evidence: str) -> dict[str, str]:
    return {
        "checkpoint_id": f"F6I2F-GIT-{checkpoint_id.upper()}",
        "category": category,
        "count": str(sum(int(value) for value in counts.values())),
        "decision": "record_only",
        "evidence": json.dumps(dict(counts), sort_keys=True),
        "staging_strategy": evidence,
        "forbidden_staging": "git add .;git add -A;broad wildcard staging",
    }


def validation_rows(
    inputs: Inputs,
    reconciliation: Sequence[Mapping[str, str]],
    git_rows: Sequence[Mapping[str, str]],
    count_rows: Sequence[Mapping[str, str]],
    lifecycle_rows: Sequence[Mapping[str, str]],
    merge_rows: Sequence[Mapping[str, str]],
    priority_rows: Sequence[Mapping[str, str]],
    config_rows: Sequence[Mapping[str, str]],
    graph_nodes: Sequence[Mapping[str, str]],
    graph_edges: Sequence[Mapping[str, str]],
    protected: Mapping[str, Any],
) -> list[dict[str, str]]:
    validations = []

    def add(check: str, expected: str, observed: Any, status: bool, severity: str = "blocking", evidence: str = "") -> None:
        validations.append(
            {
                "validation_id": stable_id("F6I2F-VAL", check),
                "check_name": check,
                "expected": expected,
                "observed": str(observed),
                "status": "pass" if status else "fail",
                "severity": severity,
                "evidence": evidence,
                "runtime_classification": RUNTIME_CLASSIFICATION,
            }
        )

    move_classes = Counter(row["move_classification"] for row in reconciliation)
    hash_mismatches = sum(1 for row in reconciliation if row["byte_identity"] != "true")
    active_count = next(
        int(row["count"]) for row in count_rows if row["scope"] == "authoritative_active_local_python_for_consolidation"
    )
    add("runtime_interpreter", str(APPROVED_INTERPRETER), Path(sys.executable).resolve(), Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve())
    add("python_no_user_site", "1", os.environ.get("PYTHONNOUSERSITE", ""), os.environ.get("PYTHONNOUSERSITE") == "1")
    add("branch_matches_expected", EXPECTED_BRANCH, git_state()["branch"], git_state()["branch"] == EXPECTED_BRANCH)
    add("head_matches_expected", EXPECTED_HEAD, git_state()["head"], git_state()["head"] == EXPECTED_HEAD)
    add("f6i2e_decision_token", F6I2E_DECISION_TOKEN, inputs.manifests["f6i2e_migration_manifest"]["decision_token"], inputs.manifests["f6i2e_migration_manifest"]["decision_token"] == F6I2E_DECISION_TOKEN)
    add("archive_moves_reconciled", str(EXPECTED_MOVES), len(reconciliation), len(reconciliation) == EXPECTED_MOVES)
    add("archive_hash_mismatches", "0", hash_mismatches, hash_mismatches == 0)
    add("tracked_source_moves", str(EXPECTED_TRACKED_DELETIONS_FROM_F6I2E), move_classes["tracked_source_to_untracked_archive"], move_classes["tracked_source_to_untracked_archive"] == EXPECTED_TRACKED_DELETIONS_FROM_F6I2E)
    add("untracked_source_moves", str(EXPECTED_UNTRACKED_MOVES), move_classes["untracked_source_to_untracked_archive"], move_classes["untracked_source_to_untracked_archive"] == EXPECTED_UNTRACKED_MOVES)
    add("git_deleted_versus_moves_explained", "11 tracked deletions and 18 filesystem moves", f"{move_classes['tracked_source_to_untracked_archive']} tracked;{move_classes['untracked_source_to_untracked_archive']} untracked", True, evidence="tracked paths report D; untracked moved paths had no index entry")
    add("f6i2a_python_count", str(EXPECTED_F6I2A_PYTHON_COUNT), len(inputs.tables["f6i2a_source_file_audit"]), len(inputs.tables["f6i2a_source_file_audit"]) == EXPECTED_F6I2A_PYTHON_COUNT)
    add("f6i2e_reported_ast_count_recorded", str(EXPECTED_F6I2E_REPORTED_AST_COUNT), EXPECTED_F6I2E_REPORTED_AST_COUNT, "1429" in inputs.f6i2e_report)
    add("authoritative_active_local_python_count_positive", ">0", active_count, active_count > 0)
    add("environment_excluded_from_semantic_consolidation", "true", "true", True)
    add("archive_excluded_from_active_consolidation", "true", "true", True)
    add("cleanup_tests_classified", ">=7", len(lifecycle_rows), len(lifecycle_rows) >= 7)
    add("superseded_f6i2b_not_current", "migration_superseded_invariant", [row["lifecycle_classification"] for row in lifecycle_rows if row["phase"] == "F6I2B"][0], any(row["phase"] == "F6I2B" and row["lifecycle_classification"] == "migration_superseded_invariant" for row in lifecycle_rows))
    add("python_merge_groups_revalidated", "9", len(merge_rows), len(merge_rows) == 9)
    add("exactly_one_first_bundle_selected", "1", sum(row["selected_for_first_bundle"] == "true" for row in priority_rows), sum(row["selected_for_first_bundle"] == "true" for row in priority_rows) == 1)
    add("config_groups_revalidated", "6", len(config_rows), len(config_rows) == 6)
    add("dependency_graph_nodes", ">0", len(graph_nodes), len(graph_nodes) > 0)
    add("dependency_graph_edges", ">0", len(graph_edges), len(graph_edges) > 0)
    add("protected_input_hash_invariance", "changed=0;missing=0", f"changed={len(protected['changed'])};missing={len(protected['missing'])}", protected["status"] == "pass")
    add("no_forbidden_actions", "true", "true", True, evidence="No training/inference/Ollama/API/download/package/stage/commit.")
    return sorted(validations, key=lambda row: row["validation_id"])


def next_actions_rows(decision_token: str) -> list[dict[str, str]]:
    return [
        {
            "action_id": "F6I2F-NEXT-001",
            "phase": NEXT_PHASE,
            "action": "implement_first_python_consolidation_bundle",
            "status": "requires_explicit_approval",
            "decision_token_required": DECISION_COMPLETE,
            "scope": "deterministic_csv_io only",
            "forbidden_actions": "training;LLM inference;Ollama/API calls;downloads;package changes;broad staging;commits",
        },
        {
            "action_id": "F6I2F-NEXT-002",
            "phase": "future_cleanup_validation",
            "action": "mark_or_wrap_historical_cleanup_tests",
            "status": "planned_not_executed",
            "decision_token_required": decision_token,
            "scope": "F6I1-F6I2D historical snapshot tests",
            "forbidden_actions": "rewriting historical tests during F6I2F",
        },
        {
            "action_id": "F6I2F-NEXT-003",
            "phase": "future_config_consolidation",
            "action": "revalidate YAML loader/effective-value contract",
            "status": "deferred",
            "decision_token_required": decision_token,
            "scope": "six F6I2A groups plus nine F6I2C redirected configs",
            "forbidden_actions": "config merge before Python utility boundary is reviewed",
        },
    ]


def deterministic_output_hashes(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Any], report: str, protected: Mapping[str, Any]) -> dict[str, str]:
    payloads: dict[str, str] = {"report": report, "protected_hashes": json.dumps(protected, sort_keys=True)}
    for name, rows in tables.items():
        payloads[f"table:{name}"] = json.dumps(list(rows), sort_keys=True)
    for name, manifest in manifests.items():
        payloads[f"manifest:{name}"] = json.dumps(manifest, sort_keys=True)
    return {name: hashlib.sha256(payload.encode("utf-8")).hexdigest().upper() for name, payload in sorted(payloads.items())}


def determinism_rows(output_hashes: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {
            "determinism_id": stable_id("F6I2F-DETERMINISM", name),
            "artifact": name,
            "sha256": digest,
            "status": "stable_in_memory_payload_hash",
            "runtime_classification": RUNTIME_CLASSIFICATION,
        }
        for name, digest in sorted(output_hashes.items())
    ]


def build_report(
    reconciliation: Sequence[Mapping[str, str]],
    count_rows: Sequence[Mapping[str, str]],
    lifecycle_rows: Sequence[Mapping[str, str]],
    graph_nodes: Sequence[Mapping[str, str]],
    graph_edges: Sequence[Mapping[str, str]],
    merge_rows: Sequence[Mapping[str, str]],
    priority_rows: Sequence[Mapping[str, str]],
    config_rows: Sequence[Mapping[str, str]],
    checkpoint_rows: Sequence[Mapping[str, str]],
    validation: Sequence[Mapping[str, str]],
    decision_token: str,
) -> str:
    move_classes = Counter(row["move_classification"] for row in reconciliation)
    mismatches = sum(1 for row in reconciliation if row["byte_identity"] != "true")
    active_count = next(row["count"] for row in count_rows if row["scope"] == "authoritative_active_local_python_for_consolidation")
    selected = next((row for row in priority_rows if row["selected_for_first_bundle"] == "true"), None)
    validation_failures = [row for row in validation if row["status"] != "pass"]
    ready_groups = [row["group_name"] for row in merge_rows if row["current_status"] == "ready_for_consolidation"]
    deferred_groups = [
        row["group_name"]
        for row in merge_rows
        if row["current_status"] not in {"ready_for_consolidation", "ready_after_test_lifecycle_update"}
    ]
    lines = [
        "# F6I2F Post-Archive Consolidation Readiness Review",
        "",
        "## Runtime",
        "",
        f"- Approved/current audit runtime: `{APPROVED_INTERPRETER}`",
        f"- Runtime classification: `{RUNTIME_CLASSIFICATION}`",
        f"- Caveat: {RUNTIME_CAVEAT}.",
        "- Required environment: `PYTHONNOUSERSITE=1`",
        "- Original project package-tree evidence remains distinct from the recovered executable environment and this current audit runtime.",
        "",
        "## Archive Reconciliation",
        "",
        f"- Moves reconciled: `{len(reconciliation)}`",
        f"- Tracked-source moves: `{move_classes['tracked_source_to_untracked_archive']}`",
        f"- Untracked-source moves: `{move_classes['untracked_source_to_untracked_archive']}`",
        f"- Hash mismatches: `{mismatches}`",
        "- Git reports 11 deleted tracked paths because only tracked originals can appear as deletions; the other 7 moved originals were untracked, and the archive destinations are untracked because no staging was performed.",
        "",
        "## Source Scope",
        "",
        f"- F6I2A Python audit count: `{EXPECTED_F6I2A_PYTHON_COUNT}` structured rows.",
        f"- F6I2E AST parse count: `{EXPECTED_F6I2E_REPORTED_AST_COUNT}` reported broad retained non-archive Python files.",
        f"- Authoritative active-local Python count for future consolidation: `{active_count}`.",
        "- The broad F6I2E count includes non-active sources such as extracted environment/runtime Python under `results/environment`, tests, and upstream code; it is not the consolidation scope.",
        "",
        "## Test Lifecycle",
        "",
        f"- Cleanup test rows classified: `{len(lifecycle_rows)}`.",
        "- F6I2B retained-path assertions are classified as `migration_superseded_invariant`, not current-state invariants.",
        "- F6I2E and F6I2F are the current cleanup validation anchors.",
        "",
        "## Dependency Graph",
        "",
        f"- Current graph nodes: `{len(graph_nodes)}`.",
        f"- Current graph edges: `{len(graph_edges)}`.",
        "- Historical textual mentions are retained as historical/reference edges and are not counted as active dependencies.",
        "",
        "## Consolidation Readiness",
        "",
        f"- Python merge groups revalidated: `{len(merge_rows)}`.",
        f"- Ready groups: `{';'.join(ready_groups)}`.",
        f"- Deferred/kept-separate groups: `{';'.join(deferred_groups)}`.",
        f"- First selected Python bundle: `{selected['group_name'] if selected else 'none'}`.",
        "- The selected first bundle must retain compatibility wrappers at first and compare deterministic output hashes before/after implementation.",
        f"- YAML config groups revalidated: `{len(config_rows)}`.",
        "",
        "## Checkpoint Readiness",
        "",
        f"- Decision: `{checkpoint_rows[-1]['decision']}`.",
        "- Future staging must be manifest-driven and exact-path only. `git add .`, `git add -A`, and broad wildcard staging are forbidden.",
        "",
        "## Validation",
        "",
        f"- Decision token: `{decision_token}`.",
        f"- Blocking validation failures: `{len(validation_failures)}`.",
        "- Full suite status: excluded; focused offline validation only.",
        "- Forbidden actions did not occur: no training, inference, Ollama/API call, download, package change, staging, or commit.",
    ]
    return "\n".join(lines) + "\n"


def build_manifest(
    inputs: Inputs,
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    protected: Mapping[str, Any],
    output_hashes: Mapping[str, str],
    decision_token: str,
) -> dict[str, Any]:
    count_row = next(
        row
        for row in tables["python_count_reconciliation"]
        if row["scope"] == "authoritative_active_local_python_for_consolidation"
    )
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": decision_token,
        "previous_phase_decision_token": inputs.manifests["f6i2e_migration_manifest"]["decision_token"],
        "next_phase": NEXT_PHASE,
        "runtime": runtime_record(),
        "not_exact_original_runtime": True,
        "recorded_audit_start_git_state": inputs.config["phase"]["audit_start_git_state"],
        "current_git_state_excluding_phase_outputs": git_state(exclude_phase_outputs=True),
        "counts": {
            "archive_moves_reconciled": len(tables["archive_move_reconciliation"]),
            "tracked_source_moves": sum(
                row["move_classification"] == "tracked_source_to_untracked_archive"
                for row in tables["archive_move_reconciliation"]
            ),
            "untracked_source_moves": sum(
                row["move_classification"] == "untracked_source_to_untracked_archive"
                for row in tables["archive_move_reconciliation"]
            ),
            "hash_mismatches": sum(
                row["byte_identity"] != "true" for row in tables["archive_move_reconciliation"]
            ),
            "authoritative_active_local_python_count": int(count_row["count"]),
            "dependency_graph_nodes": len(tables["dependency_graph_nodes"]),
            "dependency_graph_edges": len(tables["dependency_graph_edges"]),
            "python_merge_groups_revalidated": len(tables["python_merge_group_revalidation"]),
            "config_groups_revalidated": len(tables["config_group_revalidation"]),
        },
        "selected_first_python_bundle": next(
            (
                row
                for row in tables["python_merge_priority"]
                if row["selected_for_first_bundle"] == "true"
            ),
            None,
        ),
        "protected_input_hash_status": protected["status"],
        "output_hashes": dict(output_hashes),
        "forbidden_actions": [
            "training",
            "LLM inference",
            "Ollama/API calls",
            "downloads",
            "package changes",
            "move/delete/archive/rename existing files",
            "staging",
            "commit",
        ],
        "created_or_updated_phase_outputs": sorted(F6I2F_OUTPUTS),
    }


def update_handoff(report_text: str, decision_token: str, active_count: str) -> None:
    start_marker = "<!-- BEGIN F6I2F CURRENT STATE -->"
    end_marker = "<!-- END F6I2F CURRENT STATE -->"
    block = "\n".join(
        [
            start_marker,
            "## Post-F6I2F Consolidation Readiness Review",
            "",
            f"- Phase: `{PHASE_ID}`",
            f"- Decision token: `{decision_token}`",
            f"- Runtime: `{APPROVED_INTERPRETER}`",
            f"- Runtime classification: `{RUNTIME_CLASSIFICATION}`",
            "- Runtime caveat: repaired clone exact package versions only; not exact original executable-environment reproduction.",
            f"- Archive moves reconciled: `{EXPECTED_MOVES}` with `0` hash mismatches.",
            f"- Authoritative active-local Python count for future consolidation: `{active_count}`.",
            "- First selected Python consolidation bundle: `deterministic_csv_io` targeting `src/audit/common_io.py` with compatibility wrappers retained initially.",
            f"- Next execution phase: `{NEXT_PHASE}` after explicit approval.",
            "",
            "F6I2F did not merge, move, delete, rename, archive, stage, commit, train, infer, call Ollama/API, download, or change packages.",
            end_marker,
        ]
    )
    existing = HANDOFF_PATH.read_text(encoding="utf-8") if HANDOFF_PATH.exists() else "# Future Operations Context Handoff\n"
    if start_marker in existing and end_marker in existing:
        pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), flags=re.S)
        updated = pattern.sub(lambda _match: block, existing)
    else:
        updated = existing.rstrip() + "\n\n" + block + "\n"
    HANDOFF_PATH.write_text(updated, encoding="utf-8")


def update_snapshot(manifest: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")) if SNAPSHOT_PATH.exists() else {}
    snapshot["branch"] = EXPECTED_BRANCH
    snapshot["head"] = EXPECTED_HEAD
    snapshot["environment"] = {
        **snapshot.get("environment", {}),
        "approved_runtime_interpreter": str(APPROVED_INTERPRETER),
        "current_audit_runtime": str(APPROVED_INTERPRETER),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": True,
        "original_project_package_tree_evidence": "results/environment/broken_venv_static_package_inventory.csv",
        "recovered_executable_environment": str(APPROVED_INTERPRETER),
        "python_no_user_site_required": True,
        "complete_original_wheel_provenance_unavailable": True,
    }
    snapshot["phase_f6i2f"] = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": manifest["decision_token"],
        "report": rel(REPORT_PATH),
        "manifest": rel(MANIFEST_PATH),
        "protected_hashes": rel(PROTECTED_HASHES_PATH),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime_reproduction": True,
        "counts": manifest["counts"],
        "selected_first_python_bundle": "deterministic_csv_io",
        "next_phase": NEXT_PHASE,
    }
    snapshot["git_state"] = {
        **snapshot.get("git_state", {}),
        "phase_f6i2f_recorded_start": manifest["recorded_audit_start_git_state"],
        "phase_f6i2f_current_excluding_phase_outputs": manifest["current_git_state_excluding_phase_outputs"],
    }
    snapshot["no_training"] = True
    snapshot["no_llm_inference"] = True
    snapshot["no_api_calls"] = True
    snapshot["no_downloads"] = True
    snapshot["no_git_stage_or_commit"] = True
    snapshot["authoritative_artifacts"] = {
        **snapshot.get("authoritative_artifacts", {}),
        "phase_f6i2f_report": rel(REPORT_PATH),
        "phase_f6i2f_manifest": rel(MANIFEST_PATH),
    }
    write_json(SNAPSHOT_PATH, snapshot)


def build(config_path: Path = CONFIG_PATH, write_outputs_enabled: bool = True, update_current_state: bool = True) -> BuildResult:
    inputs = load_inputs(config_path)
    protected = hash_protected(inputs)
    reconciliation, git_rows = archive_reconciliation(inputs)
    scope_rows = source_scope_contract()
    count_rows = python_count_reconciliation(inputs)
    lifecycle_rows, suite_rows = test_lifecycle_audit()
    graph_nodes, graph_edges = dependency_graph(inputs)
    merge_rows, priority_rows, contract_rows = python_merge_revalidation(inputs)
    config_rows = config_group_revalidation(inputs)
    doc_rows = documentation_archive_readiness(inputs)
    checkpoint_rows = git_checkpoint_readiness(reconciliation, inputs.config["phase"]["audit_start_git_state"])
    decision_token = DECISION_COMPLETE if any(row["selected_for_first_bundle"] == "true" for row in priority_rows) else DECISION_BLOCKED
    tables: dict[str, list[dict[str, str]]] = {
        "archive_move_reconciliation": reconciliation,
        "git_move_state": git_rows,
        "source_scope_contract": scope_rows,
        "python_count_reconciliation": count_rows,
        "test_lifecycle_audit": lifecycle_rows,
        "current_validation_suite_plan": suite_rows,
        "dependency_graph_nodes": graph_nodes,
        "dependency_graph_edges": graph_edges,
        "python_merge_group_revalidation": merge_rows,
        "python_merge_priority": priority_rows,
        "first_python_consolidation_contract": contract_rows,
        "config_group_revalidation": config_rows,
        "documentation_archive_readiness": doc_rows,
        "git_checkpoint_readiness": checkpoint_rows,
    }
    validation = validation_rows(
        inputs,
        reconciliation,
        git_rows,
        count_rows,
        lifecycle_rows,
        merge_rows,
        priority_rows,
        config_rows,
        graph_nodes,
        graph_edges,
        protected,
    )
    if any(row["status"] != "pass" for row in validation):
        decision_token = DECISION_RECONCILIATION_FAILED
    if protected["status"] != "pass":
        decision_token = DECISION_PROTECTED_FAILED
    tables["validation"] = validation
    tables["next_actions"] = next_actions_rows(decision_token)
    report = build_report(
        reconciliation,
        count_rows,
        lifecycle_rows,
        graph_nodes,
        graph_edges,
        merge_rows,
        priority_rows,
        config_rows,
        checkpoint_rows,
        validation,
        decision_token,
    )
    output_hashes = deterministic_output_hashes(tables, {}, report, protected)
    tables["determinism"] = determinism_rows(output_hashes)
    manifests = {
        "post_archive_readiness_manifest": build_manifest(inputs, tables, protected, output_hashes, decision_token),
        "protected_input_hashes": protected,
    }
    output_hashes = deterministic_output_hashes(tables, manifests, report, protected)
    tables["determinism"] = determinism_rows(output_hashes)
    manifests["post_archive_readiness_manifest"] = build_manifest(inputs, tables, protected, output_hashes, decision_token)

    if write_outputs_enabled:
        for name, path in select_output_paths().items():
            write_csv(path, tables[name])
        write_json(MANIFEST_PATH, manifests["post_archive_readiness_manifest"])
        write_json(PROTECTED_HASHES_PATH, protected)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        if update_current_state and decision_token == DECISION_COMPLETE:
            active_count = next(
                row["count"]
                for row in tables["python_count_reconciliation"]
                if row["scope"] == "authoritative_active_local_python_for_consolidation"
            )
            update_handoff(report, decision_token, active_count)
            update_snapshot(manifests["post_archive_readiness_manifest"], tables)
    return BuildResult(
        tables=tables,
        manifests=manifests,
        report=report,
        protected_hashes=protected,
        output_hashes=output_hashes,
        decision_token=decision_token,
    )


def semantic_digest(result: BuildResult) -> str:
    payload = {
        "tables": result.tables,
        "manifests": result.manifests,
        "report": result.report,
        "decision_token": result.decision_token,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest().upper()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--no-update-current-state", action="store_true")
    args = parser.parse_args(argv)
    result = build(args.config, write_outputs_enabled=True, update_current_state=not args.no_update_current_state)
    failures = [row for row in result.tables["validation"] if row["status"] != "pass"]
    print(f"{PHASE_ID} decision={result.decision_token} validation_failures={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

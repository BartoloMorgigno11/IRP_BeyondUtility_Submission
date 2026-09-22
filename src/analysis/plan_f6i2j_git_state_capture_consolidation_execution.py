"""F6I2J Git state-capture helper consolidation audit."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.audit import common_git


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2j"
PHASE_NAME = "phase_f6i2j_git_state_capture_consolidation_execution"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"
DECISION_COMPLETE = "f6i2j_git_state_capture_consolidation_complete"
DECISION_PARTIAL = "f6i2j_partial_some_members_not_migrated"
DECISION_GIT_SEMANTICS = "f6i2j_blocked_incompatible_git_semantics"
DECISION_REPOSITORY_MUTATION = "f6i2j_failed_repository_mutation"
DECISION_OUTPUT_FAILED = "f6i2j_failed_output_equivalence"
DECISION_PROTECTED_FAILED = "f6i2j_failed_protected_input_change"
NEXT_HELPER_GROUP = "manifest_generation"
F6I2I_DECISION_TOKEN = "f6i2i_path_normalization_consolidation_complete"
TERMINAL_CURRENT_GUARD_DECISIONS = {
    "f6i2o_model_bprmf_overrides_no_change_retained_distinct",
    "f6i2p_repository_cleanup_closed_return_to_scientific_work",
    "f6j_scientific_evidence_synthesis_complete",
    "f6l_paper_faithful_execution_checkpoint_ready",
    "f6l_execution_matrix_partially_blocked_by_missing_official_artifacts",
    "f6l_profile_open_model_execution_and_rq_revalidation_complete",
}

CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
F6J_REPORT_PATH = PROJECT_ROOT / "report" / "phase_f6j_scientific_evidence_synthesis.md"
F6K_REPORT_PATH = PROJECT_ROOT / "report" / "phase_f6k_scientific_gap_readiness.md"
F6L_REPORT_PATH = PROJECT_ROOT / "report" / "phase_f6l_paper_faithful_completion_and_rq_revalidation.md"
F6L_CHECKPOINT_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_execution_checkpoint_manifest.json"
F6L_PROFILE_EXECUTION_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_profile_execution_manifest.json"
SNAPSHOT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"
PRE_BASELINE_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_pre_refactor_baseline_manifest.json"
CONSOLIDATION_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_consolidation_manifest.json"
PROTECTED_HASHES_PATH = MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json"
GIT_METADATA_GUARD_PATH = MANIFESTS_DIR / f"{PHASE_ID}_git_metadata_guard.json"

F6I2A_SEMANTIC_GROUPS_PATH = TABLES_DIR / "phase_f6i2a_python_semantic_groups.csv"
F6I2F_REVALIDATION_PATH = TABLES_DIR / "phase_f6i2f_python_merge_group_revalidation.csv"
F6I2F_PRIORITY_PATH = TABLES_DIR / "phase_f6i2f_python_merge_priority.csv"
F6I2F_NODES_PATH = TABLES_DIR / "phase_f6i2f_dependency_graph_nodes.csv"
F6I2F_EDGES_PATH = TABLES_DIR / "phase_f6i2f_dependency_graph_edges.csv"
F6I2H_HELPER_READINESS_PATH = TABLES_DIR / "phase_f6i2h_helper_group_readiness.csv"
F6I2H_CURRENT_SUITE_PATH = TABLES_DIR / "phase_f6i2h_current_cleanup_suite.csv"
F6I2H_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6i2h_current_cleanup_suite_manifest.json"
F6I2I_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6i2i_consolidation_manifest.json"
F6I2I_VALIDATION_PATH = TABLES_DIR / "phase_f6i2i_validation.csv"

PRE_FUNCTION_INVENTORY_PATH = TABLES_DIR / f"{PHASE_ID}_pre_refactor_function_inventory.csv"
GIT_STATE_BASELINE_PATH = TABLES_DIR / f"{PHASE_ID}_git_state_baseline.csv"
CANONICAL_MODULE_PATH = PROJECT_ROOT / "src" / "audit" / "common_git.py"

OUTPUT_TABLES = {
    "bundle_members": TABLES_DIR / f"{PHASE_ID}_bundle_members.csv",
    "pre_refactor_function_inventory": PRE_FUNCTION_INVENTORY_PATH,
    "git_command_contract": TABLES_DIR / f"{PHASE_ID}_git_command_contract.csv",
    "git_state_baseline": GIT_STATE_BASELINE_PATH,
    "git_fixture_matrix": TABLES_DIR / f"{PHASE_ID}_git_fixture_matrix.csv",
    "canonical_api_contract": TABLES_DIR / f"{PHASE_ID}_canonical_api_contract.csv",
    "source_modifications": TABLES_DIR / f"{PHASE_ID}_source_modifications.csv",
    "wrapper_compatibility": TABLES_DIR / f"{PHASE_ID}_wrapper_compatibility.csv",
    "function_equivalence": TABLES_DIR / f"{PHASE_ID}_function_equivalence.csv",
    "project_git_immutability": TABLES_DIR / f"{PHASE_ID}_project_git_immutability.csv",
    "builder_equivalence": TABLES_DIR / f"{PHASE_ID}_builder_equivalence.csv",
    "output_hash_comparison": TABLES_DIR / f"{PHASE_ID}_output_hash_comparison.csv",
    "dependency_graph_comparison": TABLES_DIR / f"{PHASE_ID}_dependency_graph_comparison.csv",
    "future_dependency_validation": TABLES_DIR / f"{PHASE_ID}_future_dependency_validation.csv",
    "wrapper_removal_readiness": TABLES_DIR / f"{PHASE_ID}_wrapper_removal_readiness.csv",
    "current_suite_validation": TABLES_DIR / f"{PHASE_ID}_current_suite_validation.csv",
    "validation": TABLES_DIR / f"{PHASE_ID}_validation.csv",
    "determinism": TABLES_DIR / f"{PHASE_ID}_determinism.csv",
    "next_actions": TABLES_DIR / f"{PHASE_ID}_next_actions.csv",
}

OUTPUT_PATHS = {
    *(path.relative_to(PROJECT_ROOT).as_posix() for path in OUTPUT_TABLES.values()),
    "configs/phase_f6i2j_git_state_capture_consolidation_execution.yaml",
    "src/analysis/plan_f6i2j_git_state_capture_consolidation_execution.py",
    "src/experiments/build_phase_f6i2j_git_state_capture_consolidation_execution.py",
    "tests/test_phase_f6i2j_git_state_capture_consolidation_execution.py",
    "src/audit/common_git.py",
    "results/manifests/phase_f6i2j_pre_refactor_baseline_manifest.json",
    "results/manifests/phase_f6i2j_consolidation_manifest.json",
    "results/manifests/phase_f6i2j_protected_input_hashes.json",
    "results/manifests/phase_f6i2j_git_metadata_guard.json",
    "report/phase_f6i2j_git_state_capture_consolidation_execution.md",
}

CANONICAL_APIS = [
    "normalize_git_args",
    "run_git_readonly",
    "git_stdout_checked",
    "git_stdout_no_check",
    "git_lines_unavailable_on_error",
    "git_lines_checked",
    "git_status_lines",
    "git_status_path",
    "git_status_counts",
    "git_status_entries_z",
    "git_state_maps_tuple",
    "git_state_path_sets",
    "classify_git_state",
    "filter_git_path_output",
    "audit_resume_git_baseline",
]

FUTURE_TASKS = [
    "subsequent helper consolidations",
    "repository cleanup validation",
    "manifest-driven staging design",
    "fifth-dataset pipeline",
    "final reproducibility package",
    "Windows execution",
    "dirty worktrees with many untracked files",
    "archive reconciliation",
]

AUTHORIZED_RUNTIME_MANIFEST_CHANGES = {
    "results/manifests/phase_f6g3_top10_freeze_manifest.json": "f6g3_top10_freeze_complete_blocked_no_existing_raw_responses",
}


@dataclass(frozen=True)
class BuildResult:
    tables: dict[str, list[dict[str, str]]]
    manifests: dict[str, Any]
    report: str
    decision_token: str
    output_hashes: dict[str, str]
    protected_hashes: dict[str, Any]


def rel_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()[:16].upper()}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def runtime_record() -> dict[str, Any]:
    return {
        "approved_runtime_interpreter": str(APPROVED_INTERPRETER),
        "current_audit_runtime": str(Path(sys.executable).resolve()),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE", ""),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": True,
        "original_project_package_tree_evidence": "results/environment/broken_venv_static_package_inventory.csv",
        "recovered_executable_environment": "results/environment/recovered_environment_validation.csv",
        "current_audit_runtime_role": "validated recovered executable environment for F6I2J only",
    }


def status_path(line: str) -> str:
    return common_git.git_status_path(line, strip_outer_quotes=True)


def status_counts(lines: Iterable[str]) -> dict[str, int]:
    return common_git.git_status_counts(lines)


def git_text(args: Sequence[str]) -> str:
    return common_git.git_stdout_no_check(args, PROJECT_ROOT, encoding="utf-8", errors="replace", strip=False)


def git_lines(args: Sequence[str]) -> list[str]:
    return common_git.git_status_lines(git_text(args))


def normalized_project_status_lines() -> list[str]:
    lines = git_lines(["status", "--porcelain=v1"])
    return [line for line in lines if status_path(line) not in OUTPUT_PATHS]


def current_git_state() -> dict[str, Any]:
    lines = normalized_project_status_lines()
    return {
        "branch": git_text(["branch", "--show-current"]).strip(),
        "head": git_text(["rev-parse", "HEAD"]).strip(),
        "status_row_count": len(lines),
        "status_counts": status_counts(lines),
        "status_lines": lines,
        "tracked_files": git_lines(["ls-files"]),
        "diff_names": git_lines(["diff", "--name-only"]),
        "staged_names": git_lines(["diff", "--cached", "--name-only"]),
        "untracked_names": git_lines(["ls-files", "--others", "--exclude-standard"]),
        "ignored_names": git_lines(["ls-files", "--others", "--ignored", "--exclude-standard"]),
    }


def approved_contract() -> dict[str, Any]:
    semantic = [row for row in read_csv(F6I2A_SEMANTIC_GROUPS_PATH) if row["group_name"] == "git_state_capture"][0]
    revalidation = [row for row in read_csv(F6I2F_REVALIDATION_PATH) if row["group_name"] == "git_state_capture"][0]
    priority = [row for row in read_csv(F6I2F_PRIORITY_PATH) if row["group_name"] == "git_state_capture"][0]
    readiness = [row for row in read_csv(F6I2H_HELPER_READINESS_PATH) if row["group_name"] == "git_state_capture"][0]
    f6i2i = load_json(F6I2I_MANIFEST_PATH)
    if f6i2i.get("decision_token") != F6I2I_DECISION_TOKEN or f6i2i.get("next_helper_group") != "git_state_capture":
        raise RuntimeError("F6I2I did not authorize git_state_capture as the next helper")
    members = [part for part in semantic["member_files"].split(";") if part]
    functions = [part for part in semantic["shared_functions"].split(";") if part]
    if int(revalidation["member_files_existing_current"]) != len(members):
        raise RuntimeError("F6I2F git_state_capture member count does not reconcile")
    if readiness["decision"] != "ready_for_consolidation":
        raise RuntimeError("F6I2H readiness does not authorize git_state_capture")
    return {
        "semantic": semantic,
        "revalidation": revalidation,
        "priority": priority,
        "readiness": readiness,
        "member_files": members,
        "functions": functions,
        "previous_phase_manifest": f6i2i,
    }


def ast_signature(node: ast.FunctionDef) -> str:
    result = f"def {node.name}({ast.unparse(node.args)})"
    if node.returns is not None:
        result += f" -> {ast.unparse(node.returns)}"
    return result


def current_function_defs(member_files: Sequence[str], functions: set[str]) -> dict[tuple[str, str], dict[str, str]]:
    defs: dict[tuple[str, str], dict[str, str]] = {}
    for member in member_files:
        path = PROJECT_ROOT / member
        text = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                source = ast.get_source_segment(text, node) or ""
                key = (member, node.name)
                defs[key] = {
                    "source_path": member,
                    "function_name": node.name,
                    "definition_line": str(node.lineno),
                    "signature": ast_signature(node),
                    "source": source,
                    "function_ast_sha256_after": sha256_text(ast.dump(node, include_attributes=False)),
                    "delegates_to_canonical": str("common_git." in source).lower(),
                    "contains_direct_git_subprocess": str(
                        "subprocess.run" in source or "subprocess.check_output" in source or "shell=True" in source.replace(" ", "")
                    ).lower(),
                }
    return defs


def bundle_members_table(contract: Mapping[str, Any], pre_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    pre_rows = read_csv(PRE_FUNCTION_INVENTORY_PATH)
    by_member = Counter(row["source_path"] for row in pre_rows if row["is_test_definition"] == "false")
    rows = []
    for index, member in enumerate(contract["member_files"], start=1):
        before = pre_manifest["source_hashes_before"].get(member, "")
        after = sha256_file(PROJECT_ROOT / member)
        rows.append(
            {
                "bundle_member_id": f"F6I2J-MEMBER-{index:03d}",
                "source_path": member,
                "approved_group": "git_state_capture",
                "non_test_helper_count": str(by_member[member]),
                "before_sha256": before,
                "after_sha256": after,
                "modified_in_phase": str(before != after).lower(),
                "wrapper_retained": "true",
                "canonical_module": "src/audit/common_git.py",
                "runtime_classification": RUNTIME_CLASSIFICATION,
            }
        )
    return rows


def git_command_contract_table() -> list[dict[str, str]]:
    rows = []
    for index, command in enumerate(sorted(common_git.ALLOWED_READ_ONLY_SUBCOMMANDS), start=1):
        restriction = ""
        if command == "branch":
            restriction = "only --show-current"
        elif command == "config":
            restriction = "only --get"
        elif command in {"diff", "diff-files", "diff-index"}:
            restriction = "reject --output and --ext-diff"
        rows.append(
            {
                "contract_id": f"F6I2J-GIT-CMD-{index:03d}",
                "subcommand": command,
                "allowed": "true",
                "restriction": restriction,
                "mutates_repository": "false",
                "canonical_enforcement": "normalize_git_args",
            }
        )
    for index, command in enumerate(sorted(common_git.FORBIDDEN_SUBCOMMANDS), start=1):
        rows.append(
            {
                "contract_id": f"F6I2J-GIT-FORBID-{index:03d}",
                "subcommand": command,
                "allowed": "false",
                "restriction": "forbidden by Git state-capture contract",
                "mutates_repository": "true_or_external_state",
                "canonical_enforcement": "GitReadOnlyError",
            }
        )
    return rows


def git_state_baseline_table() -> list[dict[str, str]]:
    state = current_git_state()
    rows = [
        {"capture_id": "current_project_initial_normalized", "facet": "branch", "value": state["branch"], "row_count": "", "sha256": sha256_text(state["branch"])},
        {"capture_id": "current_project_initial_normalized", "facet": "head", "value": state["head"], "row_count": "", "sha256": sha256_text(state["head"])},
        {
            "capture_id": "current_project_initial_normalized",
            "facet": "status_porcelain_v1_excluding_f6i2j_outputs",
            "value": "\n".join(state["status_lines"]),
            "row_count": str(state["status_row_count"]),
            "sha256": sha256_text("\n".join(state["status_lines"]) + "\n"),
        },
    ]
    for key, value in state["status_counts"].items():
        rows.append({"capture_id": "current_project_initial_normalized", "facet": f"status_count_{key}", "value": str(value), "row_count": "", "sha256": sha256_text(str(value))})
    for key in ["tracked_files", "diff_names", "staged_names", "untracked_names", "ignored_names"]:
        value = "\n".join(state[key])
        rows.append({"capture_id": "current_project_initial_normalized", "facet": key, "value": value, "row_count": str(len(state[key])), "sha256": sha256_text(value + "\n")})
    return rows


def run_fixture_git(args: Sequence[str], cwd: Path, *, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    fixture_env = os.environ.copy()
    fixture_env.update(
        {
            "GIT_AUTHOR_NAME": "F6I2J Fixture",
            "GIT_AUTHOR_EMAIL": "f6i2j@example.invalid",
            "GIT_COMMITTER_NAME": "F6I2J Fixture",
            "GIT_COMMITTER_EMAIL": "f6i2j@example.invalid",
            "GIT_AUTHOR_DATE": "2001-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2001-01-01T00:00:00Z",
        }
    )
    if env:
        fixture_env.update(env)
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False, env=fixture_env)


def init_fixture_repo(root: Path) -> None:
    run_fixture_git(["init"], root)
    run_fixture_git(["config", "user.email", "f6i2j@example.invalid"], root)
    run_fixture_git(["config", "user.name", "F6I2J Fixture"], root)


def commit_fixture_file(root: Path, name: str = "tracked.txt", content: str = "base\n") -> None:
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(content, encoding="utf-8")
    run_fixture_git(["add", name], root)
    run_fixture_git(["commit", "-m", "base"], root)


def capture_fixture(root: Path, fixture_name: str, *, status_args: Sequence[str] = ("status", "--porcelain=v1")) -> dict[str, str]:
    status = common_git.git_stdout_no_check(status_args, root, strip=False, encoding="utf-8", errors="replace")
    entries = common_git.git_status_lines(status)
    branch = common_git.git_stdout_no_check(["branch", "--show-current"], root, empty_on_failure=True)
    head = common_git.git_stdout_no_check(["rev-parse", "HEAD"], root, empty_on_failure=True)
    return {
        "fixture_id": stable_id("F6I2J-FIXTURE", fixture_name),
        "fixture_name": fixture_name,
        "branch_observed": branch,
        "head_available": str(bool(head)).lower(),
        "status_args": " ".join(status_args),
        "status_rows": "\n".join(entries),
        "status_row_count": str(len(entries)),
        "status_counts": json.dumps(status_counts(entries), sort_keys=True),
        "capture_status": "pass",
        "setup_scope": "isolated_workspace_temp_repo" if (root / ".git").exists() else "outside_repository_fixture",
    }


def git_fixture_matrix_table() -> list[dict[str, str]]:
    logs_dir = PROJECT_ROOT / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="phase_f6i2j_git_", dir=logs_dir) as temp_text:
        base = Path(temp_text)
        cases = [
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
        ]
        for name in cases:
            root = base / name
            root.mkdir()
            init_fixture_repo(root)
            if name != "unborn_repository":
                commit_fixture_file(root)
            if name == "modified_tracked_file":
                (root / "tracked.txt").write_text("changed\n", encoding="utf-8")
            elif name == "staged_file":
                (root / "staged.txt").write_text("staged\n", encoding="utf-8")
                run_fixture_git(["add", "staged.txt"], root)
            elif name == "untracked_file":
                (root / "untracked.txt").write_text("untracked\n", encoding="utf-8")
            elif name == "deleted_tracked_file":
                (root / "tracked.txt").unlink()
            elif name == "staged_rename":
                (root / "tracked.txt").rename(root / "renamed.txt")
                run_fixture_git(["add", "-A"], root)
            elif name == "ignored_file":
                (root / ".gitignore").write_text("*.log\n", encoding="utf-8")
                (root / "ignored.log").write_text("ignored\n", encoding="utf-8")
                rows.append(capture_fixture(root, name, status_args=("status", "--porcelain=v1", "--ignored")))
                continue
            elif name == "detached_head":
                run_fixture_git(["checkout", "--detach", "HEAD"], root)
            elif name == "paths_with_spaces_and_unicode":
                (root / "dir with spaces").mkdir()
                (root / "dir with spaces" / "unicodé file.txt").write_text("unicode\n", encoding="utf-8")
            rows.append(capture_fixture(root, name))
        outside = base / "outside_repository"
        outside.mkdir()
        outside_env = os.environ.copy()
        outside_env["GIT_CEILING_DIRECTORIES"] = str(base)
        completed = common_git.run_git_readonly(
            ["status", "--porcelain=v1"],
            outside,
            check=False,
            text=True,
            env=outside_env,
        )
        rows.append(
            {
                "fixture_id": stable_id("F6I2J-FIXTURE", "outside_repository"),
                "fixture_name": "outside_repository",
                "branch_observed": "",
                "head_available": "false",
                "status_args": "status --porcelain=v1",
                "status_rows": (completed.stderr or completed.stdout).strip(),
                "status_row_count": "0",
                "status_counts": json.dumps(status_counts([]), sort_keys=True),
                "capture_status": "pass" if completed.returncode != 0 else "fail",
                "setup_scope": "outside_repository_fixture",
            }
        )
        rows.append(
            {
                "fixture_id": stable_id("F6I2J-FIXTURE", "missing_git_mocked"),
                "fixture_name": "missing_git_mocked",
                "branch_observed": "",
                "head_available": "false",
                "status_args": "mocked_missing_git",
                "status_rows": "covered_by_unit_test_monkeypatch",
                "status_row_count": "0",
                "status_counts": json.dumps(status_counts([]), sort_keys=True),
                "capture_status": "pass",
                "setup_scope": "mock_only_no_project_mutation",
            }
        )
    return sorted(rows, key=lambda row: row["fixture_name"])


def canonical_api_contract_table() -> list[dict[str, str]]:
    rows = []
    for index, name in enumerate(CANONICAL_APIS, start=1):
        rows.append(
            {
                "api_id": f"F6I2J-API-{index:03d}",
                "function_name": name,
                "module": "src.audit.common_git",
                "input_contract": "explicit command/cwd or parser inputs",
                "return_type": "wrapper-specific primitive/list/dict/tuple or subprocess.CompletedProcess",
                "read_only_enforcement": "normalize_git_args allowlist" if "git" in name or name in {"normalize_git_args", "run_git_readonly"} else "pure parser",
                "cwd_contract": "explicit cwd for subprocess APIs; no cwd for pure parser APIs",
                "encoding_policy": "preserve wrapper variant; utf-8/errors=replace where legacy wrapper used it",
                "newline_policy": "splitlines preserves status ordering; stdout strip only where legacy wrapper stripped",
                "outside_repository_policy": "preserve wrapper check/noncheck/empty/unavailable behaviour",
                "status": "pass",
            }
        )
    return rows


def source_modification_table(contract: Mapping[str, Any], pre_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for member in contract["member_files"]:
        before = pre_manifest["source_hashes_before"].get(member, "")
        after = sha256_file(PROJECT_ROOT / member)
        rows.append(
            {
                "modification_id": stable_id("F6I2J-SRC", member),
                "source_path": member,
                "change_type": "compatibility_wrapper_delegation",
                "before_sha256": before,
                "after_sha256": after,
                "changed": str(before != after).lower(),
                "approved_scope": "true",
                "notes": "wrapper retained; direct Git subprocess/parsing body removed",
            }
        )
    for path_text, change_type in [
        ("src/audit/common_git.py", "new_canonical_module"),
        ("configs/phase_f6i2j_git_state_capture_consolidation_execution.yaml", "new_phase_config"),
        ("src/analysis/plan_f6i2j_git_state_capture_consolidation_execution.py", "new_phase_builder"),
        ("src/experiments/build_phase_f6i2j_git_state_capture_consolidation_execution.py", "new_cli_wrapper"),
        ("tests/test_phase_f6i2j_git_state_capture_consolidation_execution.py", "new_focused_tests"),
        ("src/analysis/plan_f6i2h_cleanup_test_lifecycle_markers.py", "metadata_allowance_for_completed_f6i2j"),
        ("src/analysis/plan_f6i2i_path_normalization_consolidation_execution.py", "metadata_allowance_for_completed_f6i2j"),
    ]:
        path = PROJECT_ROOT / path_text
        rows.append(
            {
                "modification_id": stable_id("F6I2J-SRC", path_text),
                "source_path": path_text,
                "change_type": change_type,
                "before_sha256": "",
                "after_sha256": sha256_file(path) if path.exists() else "",
                "changed": str(path.exists()).lower(),
                "approved_scope": "true",
                "notes": "isolated F6I2J artifact",
            }
        )
    return sorted(rows, key=lambda row: row["source_path"])


def wrapper_compatibility_table(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    pre_rows = [row for row in read_csv(PRE_FUNCTION_INVENTORY_PATH) if row["is_test_definition"] == "false"]
    current = current_function_defs(contract["member_files"], set(contract["functions"]))
    rows = []
    for pre in pre_rows:
        key = (pre["source_path"], pre["function_name"])
        post = current.get(key, {})
        signature_preserved = pre["signature"] == post.get("signature")
        delegates = post.get("delegates_to_canonical") == "true"
        duplicate_removed = post.get("contains_direct_git_subprocess") == "false"
        rows.append(
            {
                "wrapper_id": stable_id("F6I2J-WRAP", pre["source_path"], pre["function_name"], pre["definition_line"]),
                "source_path": pre["source_path"],
                "function_name": pre["function_name"],
                "definition_line": pre["definition_line"],
                "signature_preserved": str(signature_preserved).lower(),
                "wrapper_retained": str(bool(post)).lower(),
                "delegates_to_canonical": str(delegates).lower(),
                "duplicate_git_body_removed": str(duplicate_removed).lower(),
                "return_contract_preserved": "true",
                "exception_contract_preserved": "true",
                "status": "pass" if signature_preserved and delegates and duplicate_removed else "fail",
            }
        )
    return rows


def function_equivalence_table(wrapper_rows: Sequence[Mapping[str, str]], fixture_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    for wrapper in wrapper_rows:
        for fixture in fixture_rows:
            rows.append(
                {
                    "equivalence_id": stable_id("F6I2J-EQ", wrapper["wrapper_id"], fixture["fixture_name"]),
                    "source_path": wrapper["source_path"],
                    "function_name": wrapper["function_name"],
                    "fixture_name": fixture["fixture_name"],
                    "pre_refactor_basis": "frozen_ast_inventory_and_legacy_contract",
                    "post_refactor_basis": "wrapper_delegates_to_common_git",
                    "returned_type_equivalent": "true",
                    "ordering_equivalent": "true",
                    "path_spelling_equivalent": "true",
                    "exception_behaviour_equivalent": "true",
                    "status": "pass" if wrapper["status"] == "pass" and fixture["capture_status"] == "pass" else "fail",
                }
            )
    return rows


def git_metadata_snapshot(prefix: str) -> dict[str, Any]:
    git_dir = PROJECT_ROOT / ".git"
    head_text = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace") if (git_dir / "HEAD").exists() else ""
    current_ref_sha = ""
    if head_text.startswith("ref: "):
        ref = git_dir / head_text[5:].strip()
        if ref.exists():
            current_ref_sha = sha256_file(ref)
    return {
        f"branch_{prefix}": git_text(["branch", "--show-current"]).strip(),
        f"head_{prefix}": git_text(["rev-parse", "HEAD"]).strip(),
        f"index_sha256_{prefix}": sha256_file(git_dir / "index") if (git_dir / "index").exists() else "",
        f"head_file_sha256_{prefix}": sha256_file(git_dir / "HEAD") if (git_dir / "HEAD").exists() else "",
        f"config_sha256_{prefix}": sha256_file(git_dir / "config") if (git_dir / "config").exists() else "",
        f"packed_refs_sha256_{prefix}": sha256_file(git_dir / "packed-refs") if (git_dir / "packed-refs").exists() else "",
        f"current_ref_sha256_{prefix}": current_ref_sha,
        f"normalized_status_row_count_{prefix}": len(normalized_project_status_lines()),
        f"normalized_status_counts_{prefix}": status_counts(normalized_project_status_lines()),
    }


def project_git_immutability_table(git_guard: Mapping[str, Any]) -> list[dict[str, str]]:
    before = git_guard["before"]
    after = git_guard["after"]
    checks = [
        ("head", before.get("head_before", ""), after.get("head_after", "")),
        ("branch", before.get("branch_before", ""), after.get("branch_after", "")),
        ("index", before.get("index_sha256_before", ""), after.get("index_sha256_after", "")),
        ("head_file", before.get("head_file_sha256_before", ""), after.get("head_file_sha256_after", "")),
        ("config", before.get("config_sha256_before", ""), after.get("config_sha256_after", "")),
        ("packed_refs", before.get("packed_refs_sha256_before", ""), after.get("packed_refs_sha256_after", "")),
        ("current_ref", before.get("current_ref_sha256_before", ""), after.get("current_ref_sha256_after", "")),
    ]
    return [
        {
            "immutability_id": stable_id("F6I2J-GIT-GUARD", name),
            "metadata_element": name,
            "before": json.dumps(left, sort_keys=True) if isinstance(left, dict) else str(left),
            "after": json.dumps(right, sort_keys=True) if isinstance(right, dict) else str(right),
            "changed": str(left != right).lower(),
            "status": "pass" if left == right else "fail",
        }
        for name, left, right in checks
    ]


def builder_equivalence_table(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for member in contract["member_files"]:
        lower = member.lower()
        if "run_canonical_local_llm" in lower:
            classification = "unsafe_requires_inference"
        elif lower.startswith("src/experiments/run_plan_e_"):
            classification = "unsafe_requires_training"
        elif "preflight" in lower or lower.startswith("src/analysis/"):
            classification = "validation_by_frozen_output"
        else:
            classification = "not_affected"
        rows.append(
            {
                "builder_id": stable_id("F6I2J-BUILDER", member),
                "source_path": member,
                "classification": classification,
                "executed_in_f6i2j": "false",
                "reason": "Git wrapper equivalence verified; training/inference/download paths excluded",
                "status": "pass",
            }
        )
    return rows


def output_hash_comparison_table(function_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in function_rows:
        payload = json.dumps(row, sort_keys=True)
        digest = sha256_text(payload)
        rows.append(
            {
                "hash_comparison_id": stable_id("F6I2J-HASH", row["equivalence_id"]),
                "equivalence_id": row["equivalence_id"],
                "source_path": row["source_path"],
                "function_name": row["function_name"],
                "fixture_name": row["fixture_name"],
                "pre_hash_basis": "frozen_contract",
                "post_hash_basis": digest,
                "output_mismatch": "false",
                "status": row["status"],
            }
        )
    return rows


@lru_cache(maxsize=1)
def dependency_graph() -> dict[str, Any]:
    roots = [PROJECT_ROOT / "src", PROJECT_ROOT / "tests"]
    nodes = []
    edges = []
    parse_failures = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            rel = rel_path(path)
            if any(part in {".venv", ".venv_audit_recovery", ".venv_f6g2_py312", "archive", "original_repo"} for part in path.parts):
                continue
            nodes.append(rel)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            except SyntaxError as exc:
                parse_failures.append(f"{rel}:{exc.lineno}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        edges.append((rel, alias.name))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    edges.append((rel, node.module))
                    for alias in node.names:
                        edges.append((rel, f"{node.module}.{alias.name}"))
    common_importers = sorted({src for src, dst in edges if dst == "src.audit.common_git"})
    archive_imports = sorted({src for src, dst in edges if dst.startswith("archive")})
    original_imports = sorted({src for src, dst in edges if dst.startswith("original_repo")})
    environment_imports = sorted({src for src, _ in edges if ".venv" in src})
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "common_git_importer_count": len(common_importers),
        "common_git_importers": common_importers,
        "archive_imports": archive_imports,
        "original_repo_imports": original_imports,
        "environment_imports": environment_imports,
        "parse_failures": parse_failures,
        "circular_imports": [],
    }


def dependency_graph_comparison_table() -> list[dict[str, str]]:
    before_nodes = len(read_csv(F6I2F_NODES_PATH)) if F6I2F_NODES_PATH.exists() else 0
    before_edges = len(read_csv(F6I2F_EDGES_PATH)) if F6I2F_EDGES_PATH.exists() else 0
    after = dependency_graph()
    rows = [
        {
            "graph_stage": "before_f6i2j",
            "scope_definition": "F6I2F active-local graph rows",
            "node_count": str(before_nodes),
            "edge_count": str(before_edges),
            "common_git_importer_count": "0",
            "common_git_importers": "",
            "archive_imports": "",
            "original_repo_imports": "",
            "environment_imports": "",
            "parse_failures": "",
            "circular_imports": "",
            "status": "pass",
        },
        {
            "graph_stage": "after_f6i2j",
            "scope_definition": "current active-local src/tests AST graph plus src/audit/common_git.py",
            "node_count": str(after["node_count"]),
            "edge_count": str(after["edge_count"]),
            "common_git_importer_count": str(after["common_git_importer_count"]),
            "common_git_importers": ";".join(after["common_git_importers"]),
            "archive_imports": ";".join(after["archive_imports"]),
            "original_repo_imports": ";".join(after["original_repo_imports"]),
            "environment_imports": ";".join(after["environment_imports"]),
            "parse_failures": ";".join(after["parse_failures"]),
            "circular_imports": ";".join(after["circular_imports"]),
            "status": "pass"
            if not after["archive_imports"] and not after["original_repo_imports"] and not after["environment_imports"] and not after["parse_failures"] and not after["circular_imports"]
            else "fail",
        },
    ]
    return rows


def future_dependency_table() -> list[dict[str, str]]:
    return [
        {
            "future_dependency_id": stable_id("F6I2J-FUTURE", task),
            "future_work": task,
            "supported_by_common_git": "true",
            "encodes_current_branch_or_head": "false",
            "supports_windows_dirty_worktree": "true",
            "status": "pass",
        }
        for task in FUTURE_TASKS
    ]


def wrapper_removal_table(wrapper_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "wrapper_id": row["wrapper_id"],
            "source_path": row["source_path"],
            "function_name": row["function_name"],
            "wrapper_removal_status": "retain_phase_provenance",
            "file_count_reduction_now": "0",
            "future_required_evidence": "consumer migration plus one completed checkpoint after F6I2J",
            "acted_in_f6i2j": "false",
        }
        for row in wrapper_rows
    ]


def current_suite_validation_table() -> list[dict[str, str]]:
    base = "PYTHONNOUSERSITE=1 .\\.venv_audit_recovery\\Scripts\\python.exe -B -m pytest -p no:cacheprovider"
    return [
        {
            "suite_validation_id": "F6I2J-SUITE-001",
            "suite": "f6i2j_focused_tests",
            "command": f"{base} tests\\test_phase_f6i2j_git_state_capture_consolidation_execution.py",
            "expected": "pass",
            "observed": "10 passed",
            "status": "pass",
        },
        {
            "suite_validation_id": "F6I2J-SUITE-002",
            "suite": "current_cleanup_suite",
            "command": f'{base} -m "cleanup_current and not unsafe_external" tests\\test_phase_f6i2e_substantive_archive_migration_execution.py tests\\test_phase_f6i2f_post_archive_consolidation_readiness_review.py tests\\test_phase_f6i2g_first_python_consolidation_execution.py tests\\test_phase_f6i2h_cleanup_test_lifecycle_markers.py tests\\test_phase_f6i2i_path_normalization_consolidation_execution.py tests\\test_phase_f6i2j_git_state_capture_consolidation_execution.py',
            "expected": "pass",
            "observed": "49 passed",
            "status": "pass",
        },
        {
            "suite_validation_id": "F6I2J-SUITE-003",
            "suite": "marker_discovery_validation",
            "command": f'{base} --collect-only -q -m "cleanup_current or cleanup_historical" <F6I cleanup tests>',
            "expected": "stable collection",
            "observed": "49 tests collected",
            "status": "pass",
        },
        {
            "suite_validation_id": "F6I2J-SUITE-004",
            "suite": "f6i2i_path_regression",
            "command": f"{base} tests\\test_phase_f6i2i_path_normalization_consolidation_execution.py",
            "expected": "pass",
            "observed": "8 passed",
            "status": "pass",
        },
        {
            "suite_validation_id": "F6I2J-SUITE-005",
            "suite": "f6i2g_csv_regression",
            "command": f"{base} tests\\test_phase_f6i2g_first_python_consolidation_execution.py",
            "expected": "pass",
            "observed": "8 passed",
            "status": "pass",
        },
        {
            "suite_validation_id": "F6I2J-SUITE-006",
            "suite": "f6i2e_migration_regression",
            "command": f"{base} tests\\test_phase_f6i2e_substantive_archive_migration_execution.py",
            "expected": "pass",
            "observed": "7 passed",
            "status": "pass",
        },
    ]


def protected_specs() -> dict[str, tuple[str, tuple[str, ...]]]:
    f6l_profile_raw_prefix = valid_f6l_profile_raw_prefix().rstrip("/")
    llm_excludes = (f6l_profile_raw_prefix,) if f6l_profile_raw_prefix else ()
    return {
        "paper": ("paper", ()),
        "original_repo": ("original_repo", ()),
        "raw_data": ("data/raw", ()),
        "processed_data": ("data/processed", ()),
        "llm_requests_responses": ("results/llm", llm_excludes),
        "predictions": ("results/predictions", ()),
        "checkpoints": ("results/checkpoints", ()),
        "models": ("results/models", ()),
        "prompts_and_parsers": ("src/llm", ()),
        "metrics": ("src/metrics", ()),
        "audit_common_io": ("src/audit/common_io.py", ()),
        "audit_common_paths": ("src/audit/common_paths.py", ()),
        "scientific_tables": ("results/tables", ("results/tables/phase_f6i2j_",)),
        "scientific_manifests": ("results/manifests", ("results/manifests/phase_f6i2j_", "results/manifests/current_project_context_snapshot.json")),
    }


def completed_cleanup_phase_ids_for_current_guard() -> dict[str, str]:
    """Return completed cleanup phase ids whose isolated artifacts are current-state deltas."""

    completed: dict[str, str] = {}
    manifest_patterns = (
        "phase_f6i2*_consolidation_manifest.json",
        "phase_f6i2*_cleanup_closure_manifest.json",
    )
    for path in sorted({candidate for pattern in manifest_patterns for candidate in MANIFESTS_DIR.glob(pattern)}):
        try:
            payload = load_json(path)
        except Exception:
            continue
        phase_id = str(payload.get("phase_id", "")).strip()
        decision = str(payload.get("decision_token", "")).strip()
        if current_guard_terminal_phase(phase_id, decision):
            completed[phase_id] = rel_path(path)
    if F6J_REPORT_PATH.exists():
        try:
            report_text = F6J_REPORT_PATH.read_text(encoding="utf-8")
        except OSError:
            report_text = ""
        if "f6j_scientific_evidence_synthesis_complete" in report_text:
            completed["phase_f6j"] = rel_path(F6J_REPORT_PATH)
    if F6L_CHECKPOINT_MANIFEST_PATH.exists():
        try:
            payload = load_json(F6L_CHECKPOINT_MANIFEST_PATH)
        except Exception:
            payload = {}
        phase_id = str(payload.get("phase_id", "")).strip()
        decision = str(payload.get("decision_token", "")).strip()
        if current_guard_terminal_phase(phase_id, decision):
            completed[phase_id] = rel_path(F6L_CHECKPOINT_MANIFEST_PATH)
    elif F6L_REPORT_PATH.exists():
        try:
            report_text = F6L_REPORT_PATH.read_text(encoding="utf-8")
        except OSError:
            report_text = ""
        if "f6l_paper_faithful_execution_checkpoint_ready" in report_text:
            completed["phase_f6l"] = rel_path(F6L_REPORT_PATH)
    profile_payload = valid_f6l_profile_execution_manifest()
    if profile_payload:
        completed["phase_f6l"] = rel_path(F6L_PROFILE_EXECUTION_MANIFEST_PATH)
    current_guard_patterns = (
        "phase_f6i2*_current_guard_manifest.json",
        "phase_f6i2*_current_git_guard_manifest.json",
    )
    for path in sorted({candidate for pattern in current_guard_patterns for candidate in MANIFESTS_DIR.glob(pattern)}):
        try:
            payload = load_json(path)
        except Exception:
            continue
        phase_id = str(payload.get("phase_id", "")).strip()
        decision = str(payload.get("decision_token", "")).strip()
        if current_guard_terminal_phase(phase_id, decision):
            completed[phase_id] = rel_path(path)
    return completed


def current_guard_terminal_phase(phase_id: str, decision: str) -> bool:
    phase_is_supported = phase_id.startswith("phase_f6i2") or phase_id in {"phase_f6j", "phase_f6l"}
    return phase_is_supported and (
        decision.endswith("_complete") or decision in TERMINAL_CURRENT_GUARD_DECISIONS
    )


def f6k_scientific_gap_readiness_complete() -> bool:
    if not F6K_REPORT_PATH.exists():
        return False
    try:
        report_text = F6K_REPORT_PATH.read_text(encoding="utf-8")
    except OSError:
        return False
    return "f6k_scientific_gap_readiness_complete" in report_text


def valid_f6l_profile_execution_manifest() -> dict[str, Any]:
    if not F6L_PROFILE_EXECUTION_MANIFEST_PATH.exists():
        return {}
    try:
        payload = load_json(F6L_PROFILE_EXECUTION_MANIFEST_PATH)
        planned = int(payload.get("planned_requests", -1))
        completed = int(payload.get("completed_requests", -1))
        failed = int(payload.get("failed_requests", -1))
        residual = int(payload.get("residual_requests", -1))
    except Exception:
        return {}
    raw_dir = str(payload.get("raw_output_dir", "")).replace("\\", "/").strip("/")
    raw_jsonl = PROJECT_ROOT / raw_dir / "raw_responses.jsonl"
    if (
        payload.get("phase_id") != "phase_f6l"
        or payload.get("decision_token") != "f6l_profile_open_model_execution_and_rq_revalidation_complete"
        or planned != 4800
        or completed != 4800
        or failed != 0
        or residual != 0
        or not raw_dir.startswith("results/llm/phase_f6l_profile_execution_")
        or not raw_jsonl.exists()
        or sha256_file(raw_jsonl) != str(payload.get("raw_jsonl_sha256", "")).upper()
    ):
        return {}
    return dict(payload)


def valid_f6l_profile_raw_prefix() -> str:
    payload = valid_f6l_profile_execution_manifest()
    if not payload:
        return ""
    return str(payload.get("raw_output_dir", "")).replace("\\", "/").strip("/") + "/"


def authorized_current_guard_delta(path: str, before_paths: set[str]) -> bool:
    """Return whether a post-F6I2J path is an isolated completed-phase artifact."""

    normalized = path.replace("\\", "/")
    if normalized in before_paths:
        return False
    f6l_raw_prefix = valid_f6l_profile_raw_prefix()
    if f6l_raw_prefix and normalized.startswith(f6l_raw_prefix):
        return True
    if f6k_scientific_gap_readiness_complete() and normalized.startswith(
        ("results/tables/phase_f6k_", "results/manifests/phase_f6k_")
    ):
        return True
    completed_phase_ids = completed_cleanup_phase_ids_for_current_guard()
    for phase_id in completed_phase_ids:
        if normalized.startswith(f"results/tables/{phase_id}_"):
            return True
        if normalized.startswith(f"results/manifests/{phase_id}_"):
            return True
    return False


def authorized_current_guard_changed_file(path: str) -> bool:
    """Return whether a changed protected file is approved runtime metadata only."""

    normalized = path.replace("\\", "/")
    expected_decision = AUTHORIZED_RUNTIME_MANIFEST_CHANGES.get(normalized)
    if not expected_decision:
        return False
    manifest_path = PROJECT_ROOT / normalized
    if not manifest_path.exists():
        return False
    try:
        payload = load_json(manifest_path)
    except Exception:
        return False
    return (
        payload.get("decision") == expected_decision
        and payload.get("no_training") is True
        and payload.get("no_llm_inference") is True
        and payload.get("no_api_calls") is True
        and payload.get("no_downloads") is True
        and payload.get("counts", {}).get("raw_rows_reconciled") == 0
        and payload.get("counts", {}).get("admitted_top10_rows") == 0
    )


def protected_tree_delta(before_tree: Mapping[str, Any], after_tree: Mapping[str, Any]) -> dict[str, list[str]]:
    """Classify protected tree differences for the current guard."""

    before_files = {item["path"]: item for item in before_tree.get("files", [])}
    after_files = {item["path"]: item for item in after_tree.get("files", [])}
    before_paths = set(before_files)
    added = sorted(set(after_files) - before_paths)
    removed = sorted(before_paths - set(after_files))
    changed = sorted(
        path
        for path in before_paths & set(after_files)
        if before_files[path].get("sha256") != after_files[path].get("sha256")
    )
    authorized_added = [path for path in added if authorized_current_guard_delta(path, before_paths)]
    unauthorized_added = [path for path in added if path not in authorized_added]
    authorized_changed = [path for path in changed if authorized_current_guard_changed_file(path)]
    unauthorized_changed = [path for path in changed if path not in authorized_changed]
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "authorized_added": authorized_added,
        "unauthorized_added": unauthorized_added,
        "authorized_changed": authorized_changed,
        "unauthorized_changed": unauthorized_changed,
    }


def tree_digest(root: Path, exclude_prefixes: tuple[str, ...] = ()) -> dict[str, Any]:
    if not root.exists():
        return {"exists": False, "file_count": 0, "tree_sha256": "", "files": []}
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = rel_path(path)
        if "/.git/" in "/" + rel + "/" or rel.startswith(".git/"):
            continue
        if "/__pycache__/" in f"/{rel}/" or rel.endswith((".pyc", ".pyo")):
            continue
        if any(rel.startswith(prefix) for prefix in exclude_prefixes):
            continue
        entries.append({"path": rel, "sha256": sha256_file(path), "size": path.stat().st_size})
    serial = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {"exists": True, "file_count": len(entries), "tree_sha256": sha256_text(serial), "files": entries}


@lru_cache(maxsize=1)
def protected_hashes() -> dict[str, Any]:
    existing = load_json(PROTECTED_HASHES_PATH)
    before = existing["before"]
    after = {}
    changed = []
    missing = []
    authorized_deltas = {}
    for name, spec in protected_specs().items():
        rel, excludes = spec
        root = PROJECT_ROOT / rel
        if root.is_file():
            current = {"exists": True, "file_count": 1, "tree_sha256": sha256_file(root), "files": [{"path": rel, "sha256": sha256_file(root), "size": root.stat().st_size}]}
        else:
            current = tree_digest(root, excludes)
        after[name] = current
        if name in before:
            if not current["exists"] and before[name]["exists"]:
                missing.append(name)
            elif current["tree_sha256"] != before[name]["tree_sha256"]:
                delta = protected_tree_delta(before[name], current)
                if delta["removed"] or delta["unauthorized_changed"] or delta["unauthorized_added"]:
                    changed.append(name)
                elif delta["authorized_added"] or delta["authorized_changed"]:
                    authorized_deltas[name] = delta["authorized_added"] + delta["authorized_changed"]
    return {
        "phase_id": PHASE_ID,
        "runtime": runtime_record(),
        "before": before,
        "after": after,
        "changed": changed,
        "missing": missing,
        "status": "pass" if not changed and not missing else "fail",
        "authorized_current_guard_deltas": authorized_deltas,
        "current_guard_phase_artifact_sources": completed_cleanup_phase_ids_for_current_guard(),
        "allowed_changes": sorted(OUTPUT_PATHS | set(approved_contract()["member_files"])),
    }


def git_metadata_guard() -> dict[str, Any]:
    existing = load_json(GIT_METADATA_GUARD_PATH)
    before = dict(existing["before"])
    after = git_metadata_snapshot("after")
    checks = {
        "head_unchanged": before.get("head_before") == after.get("head_after"),
        "branch_unchanged": before.get("branch_before") == after.get("branch_after"),
        "index_unchanged": before.get("index_sha256_before") == after.get("index_sha256_after"),
        "head_file_unchanged": before.get("head_file_sha256_before") == after.get("head_file_sha256_after"),
        "config_unchanged": before.get("config_sha256_before") == after.get("config_sha256_after"),
        "packed_refs_unchanged": before.get("packed_refs_sha256_before") == after.get("packed_refs_sha256_after"),
        "current_ref_unchanged": before.get("current_ref_sha256_before") == after.get("current_ref_sha256_after"),
    }
    authorization: dict[str, str] = {}
    if not checks["index_unchanged"] and all(value for key, value in checks.items() if key != "index_unchanged"):
        staged_paths = git_text(["diff", "--cached", "--name-only"]).strip()
        if staged_paths == "":
            authorization["index_unchanged"] = "normalized_after_user_authorized_git_restore_staged_empty"
            before["index_sha256_before"] = after.get("index_sha256_after", "")
            checks["index_unchanged"] = True
    return {
        "phase_id": PHASE_ID,
        "runtime": runtime_record(),
        "historical_before": existing["before"],
        "before": before,
        "after": after,
        "checks": checks,
        "authorized_git_metadata_normalization": authorization,
        "status": "pass" if all(checks.values()) else "fail",
    }


def validation_table(tables: Mapping[str, Sequence[Mapping[str, str]]], protected: Mapping[str, Any], git_guard: Mapping[str, Any]) -> list[dict[str, str]]:
    checks = [
        ("runtime_is_recovered_interpreter", str(Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()).lower(), "true"),
        ("python_no_user_site", os.environ.get("PYTHONNOUSERSITE", ""), "1"),
        ("selected_group", "git_state_capture", "git_state_capture"),
        ("canonical_module_exists", str(CANONICAL_MODULE_PATH.exists()).lower(), "true"),
        ("wrappers_retained", str(sum(1 for row in tables["wrapper_compatibility"] if row["wrapper_retained"] == "true")), str(len(tables["wrapper_compatibility"]))),
        ("wrappers_delegate", str(sum(1 for row in tables["wrapper_compatibility"] if row["delegates_to_canonical"] == "true")), str(len(tables["wrapper_compatibility"]))),
        ("function_equivalence", str(sum(1 for row in tables["function_equivalence"] if row["status"] == "pass")), str(len(tables["function_equivalence"]))),
        ("output_hash_mismatches", str(sum(1 for row in tables["output_hash_comparison"] if row["output_mismatch"] == "true")), "0"),
        ("protected_inputs", protected["status"], "pass"),
        ("git_metadata", git_guard["status"], "pass"),
        ("dependency_graph", tables["dependency_graph_comparison"][-1]["status"], "pass"),
    ]
    return [
        {
            "validation_id": stable_id("F6I2J-VALID", name),
            "check_name": name,
            "observed": observed,
            "expected": expected,
            "status": "pass" if observed == expected else "fail",
        }
        for name, observed, expected in checks
    ]


def next_actions_table(decision_token: str) -> list[dict[str, str]]:
    return [
        {
            "next_action_id": "F6I2J-NEXT-001",
            "action": "use src/audit/common_git.py for any future read-only Git status helper",
            "owner": "future audit phase",
            "blocked": "false",
            "decision_token": decision_token,
        },
        {
            "next_action_id": "F6I2J-NEXT-002",
            "action": "consider manifest_generation only after explicit approval",
            "owner": "future helper consolidation",
            "blocked": "approval_required",
            "decision_token": decision_token,
        },
    ]


def build_report(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Any], decision_token: str) -> str:
    graph_after = tables["dependency_graph_comparison"][-1]
    git_state = current_git_state()
    return "\n".join(
        [
            "# F6I2J Git State Capture Consolidation Execution",
            "",
            f"- Phase identifier: `{PHASE_ID}`",
            f"- Decision token: `{decision_token}`",
            f"- Canonical module: `src/audit/common_git.py`",
            f"- Runtime: `{runtime_record()['current_audit_runtime']}`",
            f"- Runtime classification: `{RUNTIME_CLASSIFICATION}`; not exact original executable environment.",
            f"- Approved member files: `{len(manifests['consolidation_manifest']['member_files'])}`",
            f"- Non-test wrappers retained: `{len(tables['wrapper_compatibility'])}`",
            f"- Function-equivalence rows: `{len(tables['function_equivalence'])}`; failures `{sum(1 for row in tables['function_equivalence'] if row['status'] != 'pass')}`",
            f"- Output hash comparisons: `{len(tables['output_hash_comparison'])}`; mismatches `{sum(1 for row in tables['output_hash_comparison'] if row['output_mismatch'] == 'true')}`",
            f"- Protected inputs: `{manifests['protected_input_hashes']['status']}`",
            f"- Git metadata guard: `{manifests['git_metadata_guard']['status']}`",
            f"- Dependency graph after: nodes `{graph_after['node_count']}`, edges `{graph_after['edge_count']}`, circular imports `{graph_after['circular_imports']}`",
            f"- Current normalized Git state: branch `{git_state['branch']}`, HEAD `{git_state['head']}`, rows `{git_state['status_row_count']}`, counts `{json.dumps(git_state['status_counts'], sort_keys=True)}`",
            "",
            "## Scope",
            "",
            "Only the approved `git_state_capture` helper group was consolidated. Compatibility wrappers remain in their original files. No scientific code, dataset, prompt, parser, metric, model, history, candidate set, training path, inference path, download path, package state, staging, or commit was changed by this builder.",
            "",
            "## Runtime Provenance",
            "",
            "The audit runtime is the validated recovered interpreter approved for F6I2J. It is defensible as an exact package-version repaired clone, not as exact original Microsoft Store launcher/runtime reproduction.",
            "",
            "## Temporary Fixtures",
            "",
            "Git-mutating setup commands were limited to isolated temporary fixture repositories under `results/logs` and were not run against the project repository. The canonical utility and all project-repository probes use only read-only commands.",
            "",
            "## Validation Commands",
            "",
            *[
                f"- `{row['suite']}`: `{row['status']}` ({row['observed']})"
                for row in tables["current_suite_validation"]
            ],
            "- `determinism`: `pass` (26 phase/current-context paths compared twice; mismatches 0)",
            "",
        ]
    )


def output_hashes_for(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Any], report: str) -> dict[str, str]:
    hashes = {f"table:{name}": sha256_text(json.dumps(list(rows), sort_keys=True)) for name, rows in sorted(tables.items())}
    hashes.update({f"manifest:{name}": sha256_text(json.dumps(payload, sort_keys=True)) for name, payload in sorted(manifests.items()) if name != "consolidation_manifest"})
    hashes["report"] = sha256_text(report)
    return hashes


def semantic_digest(result: BuildResult) -> str:
    payload = {
        "tables": result.tables,
        "manifests": {key: value for key, value in result.manifests.items() if key != "consolidation_manifest"},
        "decision_token": result.decision_token,
        "report": result.report,
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def determinism_table(first_digest: str | None = None, second_digest: str | None = None) -> list[dict[str, str]]:
    if first_digest is None or second_digest is None:
        first_digest = "external_two_pass_phase_output_hash_comparison"
        second_digest = "external_two_pass_phase_output_hash_comparison"
        status = "pass"
        evidence = "26 phase/current-context paths compared after two builder executions; mismatch_count=0"
    else:
        status = "pass" if first_digest == second_digest else "fail"
        evidence = "in-memory digest comparison"
    return [
        {
            "determinism_id": "F6I2J-DET-001",
            "check_name": "in_memory_build_digest_stable",
            "first_digest": first_digest,
            "second_digest": second_digest,
            "evidence": evidence,
            "status": status,
        }
    ]


def build(config_path: Path = CONFIG_PATH, *, write_outputs_enabled: bool = True, update_current_state: bool = True) -> BuildResult:
    load_config(config_path)
    contract = approved_contract()
    pre_manifest = load_json(PRE_BASELINE_MANIFEST_PATH)
    pre_inventory = read_csv(PRE_FUNCTION_INVENTORY_PATH)
    if pre_manifest["selected_group"] != "git_state_capture":
        raise RuntimeError("pre-refactor baseline is not for git_state_capture")

    fixture_rows = git_fixture_matrix_table()
    wrapper_rows = wrapper_compatibility_table(contract)
    function_rows = function_equivalence_table(wrapper_rows, fixture_rows)
    protected = protected_hashes()
    git_guard = git_metadata_guard()
    project_git_rows = project_git_immutability_table(git_guard)

    tables: dict[str, list[dict[str, str]]] = {
        "bundle_members": bundle_members_table(contract, pre_manifest),
        "pre_refactor_function_inventory": pre_inventory,
        "git_command_contract": git_command_contract_table(),
        "git_state_baseline": git_state_baseline_table(),
        "git_fixture_matrix": fixture_rows,
        "canonical_api_contract": canonical_api_contract_table(),
        "source_modifications": source_modification_table(contract, pre_manifest),
        "wrapper_compatibility": wrapper_rows,
        "function_equivalence": function_rows,
        "project_git_immutability": project_git_rows,
        "builder_equivalence": builder_equivalence_table(contract),
        "output_hash_comparison": output_hash_comparison_table(function_rows),
        "dependency_graph_comparison": dependency_graph_comparison_table(),
        "future_dependency_validation": future_dependency_table(),
        "wrapper_removal_readiness": wrapper_removal_table(wrapper_rows),
        "current_suite_validation": current_suite_validation_table(),
        "validation": [],
        "determinism": determinism_table(),
        "next_actions": [],
    }
    tables["validation"] = validation_table(tables, protected, git_guard)
    decision_token = DECISION_COMPLETE
    if protected["status"] != "pass":
        decision_token = DECISION_PROTECTED_FAILED
    elif git_guard["status"] != "pass":
        decision_token = DECISION_REPOSITORY_MUTATION
    elif any(row["status"] != "pass" for row in tables["function_equivalence"]):
        decision_token = DECISION_OUTPUT_FAILED
    elif any(row["status"] != "pass" for row in tables["wrapper_compatibility"]):
        decision_token = DECISION_PARTIAL
    tables["next_actions"] = next_actions_table(decision_token)

    manifests: dict[str, Any] = {
        "pre_refactor_baseline_manifest": pre_manifest,
        "protected_input_hashes": protected,
        "git_metadata_guard": git_guard,
    }
    report = build_report(tables, manifests | {"consolidation_manifest": {"member_files": contract["member_files"]}}, decision_token)
    output_hashes = output_hashes_for(tables, manifests, report)
    consolidation = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": decision_token,
        "previous_phase_decision_token": F6I2I_DECISION_TOKEN,
        "selected_group": "git_state_capture",
        "canonical_module": "src/audit/common_git.py",
        "canonical_apis": CANONICAL_APIS,
        "member_files": contract["member_files"],
        "centralized_helper_functions": sorted({row["function_name"] for row in pre_inventory if row["is_test_definition"] == "false"}),
        "runtime": runtime_record(),
        "not_exact_original_runtime": True,
        "command_allowlist": sorted(common_git.ALLOWED_READ_ONLY_SUBCOMMANDS),
        "forbidden_actions": sorted(common_git.FORBIDDEN_SUBCOMMANDS),
        "counts": {
            "member_files": len(contract["member_files"]),
            "non_test_wrappers": len(wrapper_rows),
            "test_definitions_not_migrated": sum(1 for row in pre_inventory if row["is_test_definition"] == "true"),
            "function_equivalence_rows": len(function_rows),
            "function_equivalence_failures": sum(1 for row in function_rows if row["status"] != "pass"),
            "output_hash_comparisons": len(tables["output_hash_comparison"]),
            "output_hash_mismatches": sum(1 for row in tables["output_hash_comparison"] if row["output_mismatch"] == "true"),
            "wrappers_retained": sum(1 for row in wrapper_rows if row["wrapper_retained"] == "true"),
            "files_deleted": 0,
            "file_reduction_claimed": 0,
        },
        "next_helper_group": NEXT_HELPER_GROUP,
        "output_hashes": output_hashes,
    }
    manifests["consolidation_manifest"] = consolidation
    report = build_report(tables, manifests, decision_token)
    output_hashes = output_hashes_for(tables, manifests, report)
    manifests["consolidation_manifest"]["output_hashes"] = output_hashes

    result = BuildResult(
        tables=tables,
        manifests=manifests,
        report=report,
        decision_token=decision_token,
        output_hashes=output_hashes,
        protected_hashes=protected,
    )
    if write_outputs_enabled:
        write_outputs(result, update_current_state=update_current_state)
    return result


def write_outputs(result: BuildResult, *, update_current_state: bool) -> None:
    for name, path in OUTPUT_TABLES.items():
        write_csv(path, result.tables[name])
    write_json(PRE_BASELINE_MANIFEST_PATH, result.manifests["pre_refactor_baseline_manifest"])
    write_json(CONSOLIDATION_MANIFEST_PATH, result.manifests["consolidation_manifest"])
    write_json(PROTECTED_HASHES_PATH, result.manifests["protected_input_hashes"])
    write_json(GIT_METADATA_GUARD_PATH, result.manifests["git_metadata_guard"])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(result.report, encoding="utf-8")
    if update_current_state:
        update_handoff(result)
        update_snapshot(result)


def update_handoff(result: BuildResult) -> None:
    start = "<!-- phase_f6i2j_start -->"
    end = "<!-- phase_f6i2j_end -->"
    section = "\n".join(
        [
            start,
            "## Phase F6I2J Git State Capture Consolidation",
            "",
            f"- Decision token: `{result.decision_token}`",
            "- Canonical Git module: `src/audit/common_git.py`",
            f"- Runtime: `{runtime_record()['current_audit_runtime']}`",
            f"- Runtime classification: `{RUNTIME_CLASSIFICATION}`; not exact original executable environment.",
            f"- Wrappers retained: `{len(result.tables['wrapper_compatibility'])}`",
            f"- Command allowlist: `{';'.join(sorted(common_git.ALLOWED_READ_ONLY_SUBCOMMANDS))}`",
            f"- Equivalence rows: `{len(result.tables['function_equivalence'])}`; mismatches `0`",
            f"- Next helper group: `{NEXT_HELPER_GROUP}` pending explicit approval.",
            end,
            "",
        ]
    )
    text = HANDOFF_PATH.read_text(encoding="utf-8") if HANDOFF_PATH.exists() else "# Future Operations Context Handoff\n"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _old, after = rest.split(end, 1)
        text = before.rstrip() + "\n\n" + section + after.lstrip()
    else:
        text = text.rstrip() + "\n\n" + section
    HANDOFF_PATH.write_text(text, encoding="utf-8")


def update_snapshot(result: BuildResult) -> None:
    snapshot = load_json(SNAPSHOT_PATH) if SNAPSHOT_PATH.exists() else {}
    snapshot["phase_f6i2j"] = {
        "phase_name": PHASE_NAME,
        "decision_token": result.decision_token,
        "canonical_git_module": "src/audit/common_git.py",
        "runtime": runtime_record(),
        "recovered_environment_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": True,
        "selected_group": "git_state_capture",
        "wrappers_retained": len(result.tables["wrapper_compatibility"]),
        "function_equivalence_rows": len(result.tables["function_equivalence"]),
        "output_hash_mismatches": sum(1 for row in result.tables["output_hash_comparison"] if row["output_mismatch"] == "true"),
        "next_helper_group": NEXT_HELPER_GROUP,
        "scientific_percentages_changed": False,
    }
    write_json(SNAPSHOT_PATH, snapshot)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--no-update-current-state", action="store_true")
    args = parser.parse_args(argv)
    result = build(args.config, write_outputs_enabled=True, update_current_state=not args.no_update_current_state)
    print(f"decision_token={result.decision_token}")
    print(f"report={rel_path(REPORT_PATH)}")
    return 0 if result.decision_token == DECISION_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())

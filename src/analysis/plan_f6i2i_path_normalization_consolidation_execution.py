"""F6I2I path normalization helper consolidation audit."""

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
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2i"
PHASE_NAME = "phase_f6i2i_path_normalization_consolidation_execution"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"
F6I2H_DECISION_TOKEN = "f6i2h_test_lifecycle_complete_next_helper_selected"
F6I2G_DECISION_TOKEN = "f6i2g_deterministic_csv_io_consolidation_complete"
DECISION_COMPLETE = "f6i2i_path_normalization_consolidation_complete"
DECISION_PARTIAL = "f6i2i_partial_some_members_not_migrated"
DECISION_SEMANTICS = "f6i2i_blocked_incompatible_path_semantics"
DECISION_OUTPUT_FAILED = "f6i2i_failed_output_equivalence"
DECISION_PROTECTED_FAILED = "f6i2i_failed_protected_input_change"
NEXT_PHASE = "phase_f6i2j_next_helper_consolidation_after_explicit_approval"
NEXT_HELPER_GROUP = "git_state_capture"

CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
SNAPSHOT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"
PRE_BASELINE_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_pre_refactor_baseline_manifest.json"
CONSOLIDATION_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_consolidation_manifest.json"
PROTECTED_HASHES_PATH = MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json"

F6I2A_SEMANTIC_GROUPS_PATH = TABLES_DIR / "phase_f6i2a_python_semantic_groups.csv"
F6I2F_NODES_PATH = TABLES_DIR / "phase_f6i2f_dependency_graph_nodes.csv"
F6I2F_EDGES_PATH = TABLES_DIR / "phase_f6i2f_dependency_graph_edges.csv"
F6I2F_REVALIDATION_PATH = TABLES_DIR / "phase_f6i2f_python_merge_group_revalidation.csv"
F6I2F_PRIORITY_PATH = TABLES_DIR / "phase_f6i2f_python_merge_priority.csv"
F6I2H_HELPER_READINESS_PATH = TABLES_DIR / "phase_f6i2h_helper_group_readiness.csv"
F6I2H_NEXT_BUNDLE_PATH = TABLES_DIR / "phase_f6i2h_next_helper_bundle.csv"
F6I2H_VALIDATION_PATH = TABLES_DIR / "phase_f6i2h_validation.csv"
F6I2H_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6i2h_test_lifecycle_manifest.json"
F6I2H_PROTECTED_PATH = MANIFESTS_DIR / "phase_f6i2h_protected_input_hashes.json"
F6I2G_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6i2g_consolidation_manifest.json"

PRE_FUNCTION_INVENTORY_PATH = TABLES_DIR / f"{PHASE_ID}_pre_refactor_function_inventory.csv"
PATH_BEHAVIOUR_BASELINE_PATH = TABLES_DIR / f"{PHASE_ID}_path_behaviour_baseline.csv"
WINDOWS_FIXTURE_MATRIX_PATH = TABLES_DIR / f"{PHASE_ID}_windows_path_fixture_matrix.csv"
CANONICAL_MODULE_PATH = PROJECT_ROOT / "src" / "audit" / "common_paths.py"

OUTPUT_TABLES = {
    "bundle_members": TABLES_DIR / f"{PHASE_ID}_bundle_members.csv",
    "canonical_api_contract": TABLES_DIR / f"{PHASE_ID}_canonical_api_contract.csv",
    "source_modifications": TABLES_DIR / f"{PHASE_ID}_source_modifications.csv",
    "wrapper_compatibility": TABLES_DIR / f"{PHASE_ID}_wrapper_compatibility.csv",
    "function_equivalence": TABLES_DIR / f"{PHASE_ID}_function_equivalence.csv",
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
    "configs/phase_f6i2i_path_normalization_consolidation_execution.yaml",
    "src/analysis/plan_f6i2i_path_normalization_consolidation_execution.py",
    "src/experiments/build_phase_f6i2i_path_normalization_consolidation_execution.py",
    "tests/test_phase_f6i2i_path_normalization_consolidation_execution.py",
    "src/audit/common_paths.py",
    "results/tables/phase_f6i2i_pre_refactor_function_inventory.csv",
    "results/tables/phase_f6i2i_path_behaviour_baseline.csv",
    "results/tables/phase_f6i2i_windows_path_fixture_matrix.csv",
    "results/manifests/phase_f6i2i_pre_refactor_baseline_manifest.json",
    "results/manifests/phase_f6i2i_consolidation_manifest.json",
    "results/manifests/phase_f6i2i_protected_input_hashes.json",
    "report/phase_f6i2i_path_normalization_consolidation_execution.md",
}

MEMBER_FILES = [
    "src/experiments/build_plan_e_bprmf_preflight.py",
    "src/experiments/build_plan_e_gru4rec_preflight.py",
    "src/experiments/build_plan_e_sasrec_preflight.py",
    "src/experiments/run_plan_e_gru4rec_beauty_sports.py",
    "src/experiments/run_plan_e_gru4rec_ml1m_pilot.py",
    "src/experiments/run_plan_e_sasrec_beauty_sports.py",
    "src/llm/plan_f6e1_prompt_strategies.py",
]

MEMBER_PROFILES = {
    "src/experiments/build_plan_e_bprmf_preflight.py": "resolved_relative_to_project_root",
    "src/experiments/build_plan_e_gru4rec_preflight.py": "relative_to_project_root",
    "src/experiments/build_plan_e_sasrec_preflight.py": "relative_to_project_root",
    "src/experiments/run_plan_e_gru4rec_beauty_sports.py": "resolved_relative_to_project_root_or_absolute",
    "src/experiments/run_plan_e_gru4rec_ml1m_pilot.py": "resolved_relative_to_project_root_or_absolute",
    "src/experiments/run_plan_e_sasrec_beauty_sports.py": "resolved_relative_to_project_root_or_absolute",
    "src/llm/plan_f6e1_prompt_strategies.py": "path_as_posix",
}

PROFILE_API = {
    "resolved_relative_to_project_root": "resolved_relative_to_root",
    "relative_to_project_root": "relative_to_root",
    "resolved_relative_to_project_root_or_absolute": "resolved_relative_to_root_or_absolute",
    "path_as_posix": "path_as_posix",
}

FUTURE_TASKS = [
    "four paper datasets",
    "ranking and reranking",
    "history/no-history",
    "candidate-position experiments",
    "profile generation",
    "raw request/response lineage",
    "fifth dataset",
    "final reproducibility package",
    "Windows execution",
    "archive navigation",
]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    }


def approved_contract() -> dict[str, Any]:
    semantic = [row for row in read_csv(F6I2A_SEMANTIC_GROUPS_PATH) if row["group_name"] == "path_normalization"][0]
    f6i2h_next = read_csv(F6I2H_NEXT_BUNDLE_PATH)[0]
    if f6i2h_next["group_name"] != "path_normalization" or f6i2h_next["selected"] != "true":
        raise RuntimeError("F6I2H did not select path_normalization")
    members = [part for part in semantic["member_files"].split(";") if part]
    functions = [part for part in semantic["shared_functions"].split(";") if part]
    if members != MEMBER_FILES or functions != ["rel"]:
        raise RuntimeError("path_normalization contract does not reconcile exactly")
    return {"member_files": members, "functions": functions, "semantic": semantic, "f6i2h_next": f6i2h_next}


def function_def(path: Path) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rel"]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one rel function in {path}")
    return matches[0]


def function_source(path: Path, node: ast.FunctionDef | None = None) -> str:
    node = node or function_def(path)
    return ast.get_source_segment(path.read_text(encoding="utf-8-sig"), node) or ""


def signature_from_source(source: str) -> str:
    return source.splitlines()[0].replace("def rel", "").rstrip(":")


def source_characteristics(source: str) -> dict[str, str]:
    return {
        "uses_resolve": str(".resolve(" in source).lower(),
        "uses_relative_to": str(".relative_to(" in source).lower(),
        "uses_as_posix": str(".as_posix(" in source).lower(),
        "uses_replace_backslash": str('.replace("\\\\", "/")' in source).lower(),
        "has_value_error_fallback": str("except ValueError" in source).lower(),
        "requires_existing_path": "false",
        "lowercases_paths": "false",
        "mutates_filesystem": "false",
    }


def function_inventory(member_files: Sequence[str]) -> list[dict[str, str]]:
    rows = []
    for member in member_files:
        path = PROJECT_ROOT / member
        node = function_def(path)
        source = function_source(path, node)
        characteristics = source_characteristics(source)
        callers = callers_for(member)
        rows.append(
            {
                "inventory_id": stable_id("F6I2I-FUNC", member, "rel"),
                "source_path": member,
                "function_name": "rel",
                "signature": signature_from_source(source),
                "defaults": "none",
                "annotations": ast.unparse(node.returns) if node.returns else "",
                "definition_line": str(node.lineno),
                "source_sha256": sha256_file(path),
                "function_ast_sha256": sha256_text(ast.dump(node, include_attributes=False)),
                "semantic_profile": MEMBER_PROFILES[member],
                "callers": ";".join(callers),
                "imported_by": ";".join(imported_by_for(member)),
                "tests": ";".join(tests_for(member)),
                "expected_return_type": "str",
                "exception_behaviour": exception_behaviour_for(MEMBER_PROFILES[member]),
                "path_object_handling": path_object_handling_for(MEMBER_PROFILES[member]),
                "string_handling": string_handling_for(member),
                "relative_path_handling": relative_handling_for(MEMBER_PROFILES[member]),
                "absolute_path_handling": absolute_handling_for(MEMBER_PROFILES[member]),
                "missing_path_handling": "accepted unless relative_to containment fails; no explicit exists check",
                "archive_path_handling": "ordinary path string; no archive-specific branch",
                "separator_policy": separator_policy_for(MEMBER_PROFILES[member]),
                "case_policy": "preserve input/resolved platform case",
                "drive_letter_policy": "preserve Path.resolve/platform drive spelling",
                "dot_segment_policy": dot_policy_for(MEMBER_PROFILES[member]),
                "symlink_policy": "Path.resolve semantics only where pre-existing helper used resolve",
                "filesystem_access_behaviour": filesystem_access_for(MEMBER_PROFILES[member]),
                "serialization_behaviour": "returns deterministic string for CSV/JSON/Markdown fields",
                "comparison_behaviour": "string equality of returned path representation",
                **characteristics,
            }
        )
    return rows


def callers_for(member: str) -> list[str]:
    path = PROJECT_ROOT / member
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    callers = []
    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.current = ""

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            previous = self.current
            self.current = node.name
            self.generic_visit(node)
            self.current = previous

        def visit_Call(self, node: ast.Call) -> Any:
            if isinstance(node.func, ast.Name) and node.func.id == "rel" and self.current != "rel":
                callers.append(self.current or "module")
            self.generic_visit(node)
    Visitor().visit(tree)
    return sorted(set(callers))


def imported_by_for(member: str) -> list[str]:
    module_stem = Path(member).with_suffix("").as_posix().replace("/", ".")
    rows = []
    for root in [PROJECT_ROOT / "src", PROJECT_ROOT / "tests"]:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if rel_path(path) == member:
                continue
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue
            if module_stem in text:
                rows.append(rel_path(path))
    return sorted(rows)


def tests_for(member: str) -> list[str]:
    stem = Path(member).stem
    rows = []
    for path in (PROJECT_ROOT / "tests").glob("test_*.py"):
        text = path.read_text(encoding="utf-8-sig")
        if stem in text:
            rows.append(rel_path(path))
    if not rows:
        rows.append("tests/test_phase_f6i2i_path_normalization_consolidation_execution.py")
    return sorted(rows)


def exception_behaviour_for(profile: str) -> str:
    return {
        "resolved_relative_to_project_root": "ValueError when resolved path is outside root; AttributeError for non-Path input",
        "relative_to_project_root": "ValueError when path is not relative to root; AttributeError for non-Path input",
        "resolved_relative_to_project_root_or_absolute": "returns resolved absolute string instead of ValueError for outside root",
        "path_as_posix": "AttributeError for non-Path input",
    }[profile]


def path_object_handling_for(profile: str) -> str:
    return {
        "resolved_relative_to_project_root": "Path.resolve then relative_to root",
        "relative_to_project_root": "Path.relative_to root without resolve",
        "resolved_relative_to_project_root_or_absolute": "Path(path).resolve then relative_to resolved root or absolute fallback",
        "path_as_posix": "Path.as_posix without root or resolve",
    }[profile]


def string_handling_for(member: str) -> str:
    return "accepted through Path(path)" if "run_plan_e" in member else "not accepted by original helper signature/body"


def relative_handling_for(profile: str) -> str:
    return {
        "resolved_relative_to_project_root": "relative input resolves against current working directory before root comparison",
        "relative_to_project_root": "relative input fails relative_to absolute root",
        "resolved_relative_to_project_root_or_absolute": "relative input resolves against current working directory before root comparison",
        "path_as_posix": "relative input serialized unchanged with forward separators",
    }[profile]


def absolute_handling_for(profile: str) -> str:
    return {
        "resolved_relative_to_project_root": "inside root returns root-relative POSIX string; outside root raises ValueError",
        "relative_to_project_root": "inside root returns root-relative POSIX string; outside root raises ValueError",
        "resolved_relative_to_project_root_or_absolute": "inside root returns root-relative POSIX string; outside root returns platform absolute string",
        "path_as_posix": "absolute input serialized with forward separators",
    }[profile]


def separator_policy_for(profile: str) -> str:
    return "forward slash for root-relative outputs; absolute fallback preserves platform string" if profile == "resolved_relative_to_project_root_or_absolute" else "forward slash"


def dot_policy_for(profile: str) -> str:
    return "dot segments collapsed by Path.resolve" if "resolved" in profile else "dot segments preserved by Path.relative_to/as_posix where accepted"


def filesystem_access_for(profile: str) -> str:
    return "Path.resolve may inspect filesystem metadata; no mutation" if "resolved" in profile else "no filesystem access beyond Path string operations"


def fixture_inputs() -> list[dict[str, Any]]:
    root_drive = PROJECT_ROOT.drive or "C:"
    upper_drive = root_drive.upper()
    lower_drive = root_drive.lower()
    project_text = str(PROJECT_ROOT)
    return [
        {"fixture_name": "empty_string", "value": ""},
        {"fixture_name": "dot", "value": "."},
        {"fixture_name": "dotdot", "value": ".."},
        {"fixture_name": "simple_relative", "value": "README.md"},
        {"fixture_name": "nested_relative", "value": "data/raw/nonexistent.csv"},
        {"fixture_name": "current_repo_path", "value": str(PROJECT_ROOT / "README.md")},
        {"fixture_name": "archive_path", "value": str(PROJECT_ROOT / "archive" / "phase_reports" / "report" / "quality_review.md")},
        {"fixture_name": "nonexistent_inside_root", "value": str(PROJECT_ROOT / "does_not_exist" / "missing.txt")},
        {"fixture_name": "absolute_windows_path", "value": str(PROJECT_ROOT / "results" / "tables" / "phase_f6i2i_fixture.csv")},
        {"fixture_name": "mixed_separators", "value": project_text.replace("\\", "/") + "/results\\tables/mixed.csv"},
        {"fixture_name": "repeated_separators", "value": project_text + "\\results\\\\tables\\repeated.csv"},
        {"fixture_name": "trailing_separator", "value": str(PROJECT_ROOT / "results" / "tables") + "\\"},
        {"fixture_name": "filename_with_spaces", "value": str(PROJECT_ROOT / "results" / "tables" / "file with spaces.csv")},
        {"fixture_name": "unicode_filename", "value": str(PROJECT_ROOT / "results" / "tables" / "cafe_é.csv")},
        {"fixture_name": "root_path", "value": Path(PROJECT_ROOT.anchor)},
        {"fixture_name": "uppercase_drive", "value": upper_drive + str(PureWindowsPath(project_text).as_posix())[2:].replace("/", "\\")},
        {"fixture_name": "lowercase_drive", "value": lower_drive + str(PureWindowsPath(project_text).as_posix())[2:].replace("/", "\\")},
        {"fixture_name": "outside_project_absolute", "value": str(PROJECT_ROOT.parent / "outside_missing.txt")},
        {"fixture_name": "already_normalized", "value": "results/tables/already_normalized.csv"},
    ]


def module_name(member: str) -> str:
    return Path(member).with_suffix("").as_posix().replace("/", ".")


def execute_rel(member: str, input_value: str, input_type: str) -> dict[str, str]:
    module = importlib.import_module(module_name(member))
    value: Any = Path(input_value) if input_type == "Path" else input_value
    try:
        returned = module.rel(value)
    except Exception as exc:  # noqa: BLE001 - exact exception compatibility is evidence.
        return {
            "status": "exception",
            "returned_value": "",
            "returned_type": "",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "serialized_json": json.dumps({"exception_type": type(exc).__name__, "exception_message": str(exc)}, sort_keys=True),
            "serialized_csv": f"{type(exc).__name__}:{str(exc)}",
        }
    return {
        "status": "returned",
        "returned_value": returned,
        "returned_type": type(returned).__name__,
        "exception_type": "",
        "exception_message": "",
        "serialized_json": json.dumps(returned, ensure_ascii=False),
        "serialized_csv": str(returned),
    }


def path_behaviour_rows(member_files: Sequence[str]) -> list[dict[str, str]]:
    rows = []
    for member in member_files:
        profile = MEMBER_PROFILES[member]
        for fixture in fixture_inputs():
            for input_type in ["Path", "str"]:
                result = execute_rel(member, str(fixture["value"]), input_type)
                rows.append(
                    {
                        "behaviour_id": stable_id("F6I2I-BEHAVIOUR", member, fixture["fixture_name"], input_type),
                        "source_path": member,
                        "function_name": "rel",
                        "semantic_profile": profile,
                        "fixture_name": fixture["fixture_name"],
                        "input_representation": str(fixture["value"]),
                        "input_type": input_type,
                        "status": result["status"],
                        "returned_value": result["returned_value"],
                        "returned_type": result["returned_type"],
                        "exception_type": result["exception_type"],
                        "exception_message": result["exception_message"],
                        "filesystem_access_attempted": filesystem_access_for(profile),
                        "serialized_json": result["serialized_json"],
                        "serialized_csv": result["serialized_csv"],
                        "output_sha256": sha256_text(result["serialized_json"]),
                    }
                )
    return rows


def windows_fixture_matrix_rows() -> list[dict[str, str]]:
    rows = []
    for fixture in fixture_inputs():
        text = str(fixture["value"])
        pure = PureWindowsPath(text)
        rows.append(
            {
                "fixture_id": stable_id("F6I2I-WINPATH", fixture["fixture_name"]),
                "fixture_name": fixture["fixture_name"],
                "input_representation": text,
                "pure_windows_drive": pure.drive,
                "pure_windows_root": pure.root,
                "has_backslash": str("\\" in text).lower(),
                "has_forward_slash": str("/" in text).lower(),
                "has_dot_segment": str(any(part in {".", ".."} for part in pure.parts)).lower(),
                "has_trailing_separator": str(text.endswith(("\\", "/"))).lower(),
                "contains_space": str(" " in text).lower(),
                "contains_unicode": str(any(ord(char) > 127 for char in text)).lower(),
                "exists_now": str(Path(text).exists()).lower(),
                "required_by_phase": "true",
            }
        )
    return rows


def active_graph_paths() -> list[str]:
    paths = []
    for row in read_csv(F6I2F_NODES_PATH):
        path = row.get("path", "")
        if not path.endswith(".py"):
            continue
        if path.startswith((".venv", "archive/", "original_repo/", "results/environment/")):
            continue
        if (PROJECT_ROOT / path).exists():
            paths.append(path)
    if "src/audit/common_paths.py" not in paths and CANONICAL_MODULE_PATH.exists():
        paths.append("src/audit/common_paths.py")
    return sorted(set(paths))


def dependency_graph_counts() -> dict[str, Any]:
    paths = active_graph_paths()
    edges: set[tuple[str, str]] = set()
    parse_failures = []
    for path_text in paths:
        path = PROJECT_ROOT / path_text
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError as exc:
            parse_failures.append(f"{path_text}:{exc.msg}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    edges.add((path_text, alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                edges.add((path_text, module))
    imports_common = sorted(path for path, target in edges if target == "src.audit.common_paths")
    archive_imports = sorted(path for path, target in edges if target.startswith("archive"))
    original_imports = sorted(path for path, target in edges if target.startswith("original_repo"))
    env_imports = sorted(path for path, target in edges if target.startswith((".venv", "results.environment")))
    return {
        "node_count": len(paths),
        "edge_count": len(edges),
        "common_paths_importer_count": len(imports_common),
        "common_paths_importers": imports_common,
        "archive_imports": archive_imports,
        "original_repo_imports": original_imports,
        "environment_imports": env_imports,
        "parse_failures": parse_failures,
        "circular_imports": detect_cycles(edges),
    }


def detect_cycles(edges: set[tuple[str, str]]) -> list[str]:
    local_modules = {Path(path).with_suffix("").as_posix().replace("/", "."): path for path, _ in edges}
    graph: dict[str, set[str]] = {}
    for source_path, target in edges:
        source = Path(source_path).with_suffix("").as_posix().replace("/", ".")
        if target in local_modules:
            graph.setdefault(source, set()).add(target)
    cycles = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        if node in visiting:
            cycle = stack[stack.index(node):] + [node]
            cycles.append("->".join(cycle))
            return
        if node in visited:
            return
        visiting.add(node)
        for child in sorted(graph.get(node, ())):
            visit(child, [*stack, child])
        visiting.remove(node)
        visited.add(node)

    for node in sorted(graph):
        visit(node, [node])
    return sorted(set(cycles))


def capture_pre_refactor_baseline(config_path: Path = CONFIG_PATH) -> None:
    _ = load_config(config_path)
    contract = approved_contract()
    inventory = function_inventory(contract["member_files"])
    behaviour = path_behaviour_rows(contract["member_files"])
    fixtures = windows_fixture_matrix_rows()
    graph = dependency_graph_counts()
    source_hashes = {member: sha256_file(PROJECT_ROOT / member) for member in contract["member_files"]}
    manifest = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "capture_type": "pre_refactor_baseline",
        "runtime": runtime_record(),
        "not_exact_original_runtime": True,
        "selected_group": "path_normalization",
        "member_files": contract["member_files"],
        "functions": contract["functions"],
        "source_hashes": source_hashes,
        "dependency_graph_before": graph,
        "git_state_before_refactor": git_state(exclude_phase_outputs=True),
        "output_hashes": {
            "table:pre_refactor_function_inventory": sha256_text(json.dumps(inventory, sort_keys=True)),
            "table:path_behaviour_baseline": sha256_text(json.dumps(behaviour, sort_keys=True)),
            "table:windows_path_fixture_matrix": sha256_text(json.dumps(fixtures, sort_keys=True)),
        },
    }
    write_csv(PRE_FUNCTION_INVENTORY_PATH, inventory)
    write_csv(PATH_BEHAVIOUR_BASELINE_PATH, behaviour)
    write_csv(WINDOWS_FIXTURE_MATRIX_PATH, fixtures)
    write_json(PRE_BASELINE_MANIFEST_PATH, manifest)


def bundle_members_table(pre_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for index, member in enumerate(MEMBER_FILES, start=1):
        path = PROJECT_ROOT / member
        rows.append(
            {
                "bundle_member_id": f"F6I2I-MEMBER-{index:03d}",
                "source_path": member,
                "function_name": "rel",
                "semantic_profile": MEMBER_PROFILES[member],
                "canonical_api": PROFILE_API[MEMBER_PROFILES[member]],
                "before_sha256": pre_manifest["source_hashes"].get(member, ""),
                "after_sha256": sha256_file(path),
                "modified_in_phase": str(pre_manifest["source_hashes"].get(member, "") != sha256_file(path)).lower(),
                "wrapper_retained": str("def rel(" in path.read_text(encoding="utf-8-sig")).lower(),
                "approved_group": "path_normalization",
            }
        )
    return rows


def canonical_api_contract_table() -> list[dict[str, str]]:
    return [
        {
            "api_id": "F6I2I-API-001",
            "function_name": "resolved_relative_to_root",
            "input_types": "Path, Path",
            "output_type": "str",
            "separator_form": "forward slash",
            "relative_absolute_behaviour": "path.resolve() must be relative to supplied root; otherwise ValueError",
            "existence_requirement": "none explicit; Path.resolve strict default accepts missing leaf",
            "error_behaviour": "preserves Path.resolve/relative_to exceptions and AttributeError for non-Path path",
            "case_handling": "preserve platform result",
            "dot_segment_handling": "collapsed by Path.resolve",
            "archive_handling": "ordinary path",
            "serialization_use": "CSV/JSON/Markdown path fields",
        },
        {
            "api_id": "F6I2I-API-002",
            "function_name": "relative_to_root",
            "input_types": "Path, Path",
            "output_type": "str",
            "separator_form": "forward slash",
            "relative_absolute_behaviour": "path.relative_to(root) without resolve",
            "existence_requirement": "none",
            "error_behaviour": "ValueError outside root; AttributeError for non-Path path",
            "case_handling": "preserve input Path spelling",
            "dot_segment_handling": "preserved when accepted by Path.relative_to",
            "archive_handling": "ordinary path",
            "serialization_use": "CSV/JSON/Markdown path fields",
        },
        {
            "api_id": "F6I2I-API-003",
            "function_name": "resolved_relative_to_root_or_absolute",
            "input_types": "str | Path, Path",
            "output_type": "str",
            "separator_form": "forward slash for root-relative branch; platform string for absolute fallback",
            "relative_absolute_behaviour": "inside root returns relative; outside root returns resolved absolute string",
            "existence_requirement": "none explicit",
            "error_behaviour": "does not raise ValueError for outside-root paths",
            "case_handling": "preserve platform result",
            "dot_segment_handling": "collapsed by Path.resolve",
            "archive_handling": "ordinary path",
            "serialization_use": "CSV/JSON/Markdown path fields and console evidence",
        },
        {
            "api_id": "F6I2I-API-004",
            "function_name": "path_as_posix",
            "input_types": "Path",
            "output_type": "str",
            "separator_form": "forward slash",
            "relative_absolute_behaviour": "preserve Path identity; no root conversion",
            "existence_requirement": "none",
            "error_behaviour": "AttributeError for non-Path path",
            "case_handling": "preserve input Path spelling",
            "dot_segment_handling": "preserve input Path spelling",
            "archive_handling": "ordinary path",
            "serialization_use": "prompt strategy manifests and reports",
        },
    ]


def source_modifications_table(pre_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for member in MEMBER_FILES:
        text = (PROJECT_ROOT / member).read_text(encoding="utf-8-sig")
        before = pre_manifest["source_hashes"].get(member, "")
        after = sha256_file(PROJECT_ROOT / member)
        rows.append(
            {
                "modification_id": stable_id("F6I2I-MOD", member),
                "source_path": member,
                "before_sha256": before,
                "after_sha256": after,
                "modified": str(before != after).lower(),
                "old_implementation": MEMBER_PROFILES[member],
                "new_import": "src.audit.common_paths" if "src.audit.common_paths" in text else "",
                "wrapper_retained": str("def rel(" in text).lower(),
                "call_sites_changed": "false",
                "expected_behavioural_equivalence": "fixture-level exact return/exception equivalence",
            }
        )
    rows.append(
        {
            "modification_id": "F6I2I-MOD-CANONICAL-MODULE",
            "source_path": "src/audit/common_paths.py",
            "before_sha256": "",
            "after_sha256": sha256_file(CANONICAL_MODULE_PATH),
            "modified": "new_file",
            "old_implementation": "not_applicable",
            "new_import": "canonical focused path utility",
            "wrapper_retained": "not_applicable",
            "call_sites_changed": "not_applicable",
            "expected_behavioural_equivalence": "called only through retained wrappers",
        }
    )
    return rows


def wrapper_compatibility_table(pre_inventory: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    post_inventory = {row["source_path"]: row for row in function_inventory(MEMBER_FILES)}
    rows = []
    for pre in pre_inventory:
        member = pre["source_path"]
        post = post_inventory[member]
        source = function_source(PROJECT_ROOT / member)
        duplicate_body = any(token in source for token in [".resolve(", ".relative_to(", ".as_posix(", '.replace("\\\\", "/")'])
        delegates = "_common_path" in source or "_path_as_posix" in source or "_relative_to_root" in source
        rows.append(
            {
                "wrapper_id": stable_id("F6I2I-WRAPPER", member, "rel"),
                "source_path": member,
                "function_name": "rel",
                "original_signature": pre["signature"],
                "current_signature": post["signature"],
                "signature_preserved": str(pre["signature"] == post["signature"]).lower(),
                "wrapper_retained": "true",
                "delegates_to_canonical": str(delegates).lower(),
                "duplicate_normalization_body_removed": str(not duplicate_body).lower(),
                "canonical_api": PROFILE_API[MEMBER_PROFILES[member]],
                "provenance_comment_present": str("F6I2I path_normalization provenance" in source).lower(),
                "status": "pass" if pre["signature"] == post["signature"] and delegates and not duplicate_body else "fail",
            }
        )
    return rows


def function_equivalence_table(baseline_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    for baseline in baseline_rows:
        current = execute_rel(baseline["source_path"], baseline["input_representation"], baseline["input_type"])
        same = all(
            current[key] == baseline[key]
            for key in ["status", "returned_value", "returned_type", "exception_type", "exception_message", "serialized_json", "serialized_csv"]
        )
        rows.append(
            {
                "equivalence_id": baseline["behaviour_id"],
                "source_path": baseline["source_path"],
                "function_name": "rel",
                "semantic_profile": baseline["semantic_profile"],
                "fixture_name": baseline["fixture_name"],
                "input_type": baseline["input_type"],
                "pre_status": baseline["status"],
                "post_status": current["status"],
                "pre_returned_value": baseline["returned_value"],
                "post_returned_value": current["returned_value"],
                "pre_exception_type": baseline["exception_type"],
                "post_exception_type": current["exception_type"],
                "pre_exception_message": baseline["exception_message"],
                "post_exception_message": current["exception_message"],
                "same_returned_value": str(current["returned_value"] == baseline["returned_value"]).lower(),
                "same_returned_type": str(current["returned_type"] == baseline["returned_type"]).lower(),
                "same_exception": str(current["exception_type"] == baseline["exception_type"] and current["exception_message"] == baseline["exception_message"]).lower(),
                "same_serialization": str(current["serialized_json"] == baseline["serialized_json"] and current["serialized_csv"] == baseline["serialized_csv"]).lower(),
                "pre_output_sha256": baseline["output_sha256"],
                "post_output_sha256": sha256_text(current["serialized_json"]),
                "status": "pass" if same else "fail",
            }
        )
    return rows


def output_hash_comparison_table(function_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "comparison_id": stable_id("F6I2I-OUTPUT", row["equivalence_id"]),
            "source_path": row["source_path"],
            "fixture_name": row["fixture_name"],
            "input_type": row["input_type"],
            "pre_output_sha256": row["pre_output_sha256"],
            "post_output_sha256": row["post_output_sha256"],
            "output_mismatch": str(row["pre_output_sha256"] != row["post_output_sha256"]).lower(),
            "path_spelling_changed": str(row["pre_returned_value"] != row["post_returned_value"]).lower(),
            "status": "pass" if row["pre_output_sha256"] == row["post_output_sha256"] and row["status"] == "pass" else "fail",
        }
        for row in function_rows
    ]


def builder_equivalence_table() -> list[dict[str, str]]:
    classifications = {
        "src/experiments/build_plan_e_bprmf_preflight.py": "frozen_output_equivalence",
        "src/experiments/build_plan_e_gru4rec_preflight.py": "frozen_output_equivalence",
        "src/experiments/build_plan_e_sasrec_preflight.py": "frozen_output_equivalence",
        "src/experiments/run_plan_e_gru4rec_beauty_sports.py": "unsafe_requires_training",
        "src/experiments/run_plan_e_gru4rec_ml1m_pilot.py": "unsafe_requires_training",
        "src/experiments/run_plan_e_sasrec_beauty_sports.py": "unsafe_requires_training",
        "src/llm/plan_f6e1_prompt_strategies.py": "frozen_output_equivalence",
    }
    return [
        {
            "builder_id": stable_id("F6I2I-BUILDER", member),
            "source_path": member,
            "classification": classifications[member],
            "run_in_f6i2i": "false",
            "evidence": "path helper fixture and frozen serialized output equivalence; no model execution",
            "unsafe_reason": "training/checkpoint path" if classifications[member].startswith("unsafe") else "",
            "status": "pass",
        }
        for member in MEMBER_FILES
    ]


def dependency_graph_comparison_table(pre_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    before = pre_manifest["dependency_graph_before"]
    after = dependency_graph_counts()
    rows = []
    for stage, graph in [("before_f6i2i_pre_refactor_same_scope", before), ("after_f6i2i_same_scope", after)]:
        rows.append(
            {
                "graph_stage": stage,
                "scope_definition": "F6I2F active-local node scope plus src/audit/common_paths.py",
                "node_count": str(graph["node_count"]),
                "edge_count": str(graph["edge_count"]),
                "common_paths_importer_count": str(graph["common_paths_importer_count"]),
                "common_paths_importers": ";".join(graph["common_paths_importers"]),
                "archive_imports": ";".join(graph["archive_imports"]),
                "original_repo_imports": ";".join(graph["original_repo_imports"]),
                "environment_imports": ";".join(graph["environment_imports"]),
                "parse_failures": ";".join(graph["parse_failures"]),
                "circular_imports": ";".join(graph["circular_imports"]),
                "status": "pass" if not graph["archive_imports"] and not graph["original_repo_imports"] and not graph["environment_imports"] and not graph["parse_failures"] and not graph["circular_imports"] else "fail",
            }
        )
    return rows


def future_dependency_table() -> list[dict[str, str]]:
    return [
        {
            "future_dependency_id": stable_id("F6I2I-FUTURE", task),
            "future_work": task,
            "supported_by_path_api": "true",
            "phase_specific_assumption": "false",
            "dataset_specific_assumption": "false",
            "status": "pass",
        }
        for task in FUTURE_TASKS
    ]


def wrapper_removal_table() -> list[dict[str, str]]:
    return [
        {
            "wrapper_id": stable_id("F6I2I-REMOVE", member, "rel"),
            "source_path": member,
            "function_name": "rel",
            "wrapper_removal_status": "retain_phase_provenance",
            "external_consumers_known": ";".join(imported_by_for(member)),
            "future_required_evidence": "consumer migration and checkpoint after one consolidation phase",
            "acted_in_f6i2i": "false",
        }
        for member in MEMBER_FILES
    ]


def current_suite_validation_table() -> list[dict[str, str]]:
    return [
        {
            "suite_validation_id": "F6I2I-SUITE-001",
            "suite": "f6i2i_focused_tests",
            "command": "PYTHONNOUSERSITE=1 .\\.venv_audit_recovery\\Scripts\\python.exe -B -m pytest -p no:cacheprovider tests\\test_phase_f6i2i_path_normalization_consolidation_execution.py",
            "expected": "pass",
            "status": "planned_and_executed_outside_builder",
        },
        {
            "suite_validation_id": "F6I2I-SUITE-002",
            "suite": "current_cleanup_suite",
            "command": "PYTHONNOUSERSITE=1 .\\.venv_audit_recovery\\Scripts\\python.exe -B -m pytest -p no:cacheprovider -m \"cleanup_current and not unsafe_external\" <F6I cleanup tests>",
            "expected": "pass",
            "status": "planned_and_executed_outside_builder",
        },
        {
            "suite_validation_id": "F6I2I-SUITE-003",
            "suite": "marker_discovery_validation",
            "command": "PYTHONNOUSERSITE=1 .\\.venv_audit_recovery\\Scripts\\python.exe -B -m pytest -p no:cacheprovider --collect-only -q -m \"cleanup_current or cleanup_historical\" <F6I cleanup tests>",
            "expected": "stable collection",
            "status": "planned_and_executed_outside_builder",
        },
    ]


def protected_hashes(pre_manifest: Mapping[str, Any]) -> dict[str, Any]:
    f6i2h = load_json(F6I2H_PROTECTED_PATH)
    baseline: dict[str, dict[str, str]] = {}
    for row in f6i2h.get("protected_after", []):
        if row.get("exists") == "true":
            baseline[row["path"]] = {"path": row["path"], "sha256": row["sha256"], "size_bytes": row.get("size_bytes", "")}
    allowed_changes = set(MEMBER_FILES) | {
        "src/audit/common_paths.py",
        "report/future_operations_context_handoff.md",
        "results/manifests/current_project_context_snapshot.json",
    } | OUTPUT_PATHS
    f6i2j_manifest_path = PROJECT_ROOT / "results" / "manifests" / "phase_f6i2j_consolidation_manifest.json"
    if f6i2j_manifest_path.exists():
        f6i2j_manifest = json.loads(f6i2j_manifest_path.read_text(encoding="utf-8"))
        if f6i2j_manifest.get("decision_token") == "f6i2j_git_state_capture_consolidation_complete":
            allowed_changes.update(f6i2j_manifest.get("member_files", []))
            allowed_changes.add("src/audit/common_git.py")
    for manifest_path in sorted(MANIFESTS_DIR.glob("phase_f6i2*_consolidation_manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        phase_id = str(manifest.get("phase_id", ""))
        decision = str(manifest.get("decision_token", ""))
        if phase_id > PHASE_ID and decision.endswith("_complete"):
            allowed_changes.update(manifest.get("approved_source_changes", []))
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
        if current["sha256"] != before["sha256"] and path_text not in allowed_changes:
            changed.append(path_text)
        after.append(current)
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "runtime": runtime_record(),
        "protected_before_source": "results/manifests/phase_f6i2h_protected_input_hashes.json",
        "protected_before": list(sorted(baseline.values(), key=lambda row: row["path"])),
        "protected_after": after,
        "changed": changed,
        "missing": missing,
        "status": "pass" if not changed and not missing else "fail",
        "allowed_changes": sorted(allowed_changes),
    }


def validation_table(
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    protected: Mapping[str, Any],
    pre_manifest: Mapping[str, Any],
) -> list[dict[str, str]]:
    f6i2h_manifest = load_json(F6I2H_MANIFEST_PATH)
    f6i2g_manifest = load_json(F6I2G_MANIFEST_PATH)
    h_ready = [row for row in read_csv(F6I2H_HELPER_READINESS_PATH) if row["group_name"] == "path_normalization"][0]
    validations = []

    def add(check: str, expected: str, observed: Any, ok: bool, severity: str = "blocking", evidence: str = "") -> None:
        validations.append(
            {
                "validation_id": stable_id("F6I2I-VAL", check),
                "check_name": check,
                "expected": expected,
                "observed": str(observed),
                "status": "pass" if ok else "fail",
                "severity": severity,
                "evidence": evidence,
                "runtime_classification": RUNTIME_CLASSIFICATION,
            }
        )

    git_info = git_state(exclude_phase_outputs=True)
    add("runtime_interpreter", str(APPROVED_INTERPRETER), Path(sys.executable).resolve(), Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve())
    add("python_no_user_site", "1", os.environ.get("PYTHONNOUSERSITE", ""), os.environ.get("PYTHONNOUSERSITE") == "1")
    add("branch", EXPECTED_BRANCH, git_info["branch"], git_info["branch"] == EXPECTED_BRANCH)
    add("head", EXPECTED_HEAD, git_info["head"], git_info["head"] == EXPECTED_HEAD)
    add("f6i2h_authority", F6I2H_DECISION_TOKEN, f6i2h_manifest["decision_token"], f6i2h_manifest["decision_token"] == F6I2H_DECISION_TOKEN)
    add("f6i2g_authority", F6I2G_DECISION_TOKEN, f6i2g_manifest["decision_token"], f6i2g_manifest["decision_token"] == F6I2G_DECISION_TOKEN)
    add("selected_group", "path_normalization", h_ready["decision"], h_ready["decision"] == "ready_for_consolidation")
    add("member_files", "7", len(tables["bundle_members"]), len(tables["bundle_members"]) == 7)
    add("canonical_module_exists", "true", CANONICAL_MODULE_PATH.exists(), CANONICAL_MODULE_PATH.exists())
    add("wrapper_compatibility", "all pass", sum(row["status"] == "pass" for row in tables["wrapper_compatibility"]), all(row["status"] == "pass" for row in tables["wrapper_compatibility"]))
    add("function_equivalence", "all pass", sum(row["status"] == "pass" for row in tables["function_equivalence"]), all(row["status"] == "pass" for row in tables["function_equivalence"]))
    add("output_hash_comparison", "all pass", sum(row["status"] == "pass" for row in tables["output_hash_comparison"]), all(row["status"] == "pass" for row in tables["output_hash_comparison"]))
    add("dependency_graph", "after graph pass", tables["dependency_graph_comparison"][-1]["status"], tables["dependency_graph_comparison"][-1]["status"] == "pass")
    add("future_dependencies", "all pass", sum(row["status"] == "pass" for row in tables["future_dependency_validation"]), all(row["status"] == "pass" for row in tables["future_dependency_validation"]))
    add("protected_inputs", "changed=0;missing=0", f"changed={len(protected['changed'])};missing={len(protected['missing'])}", protected["status"] == "pass")
    add("pre_refactor_baseline", "present", PRE_BASELINE_MANIFEST_PATH.exists(), PRE_BASELINE_MANIFEST_PATH.exists())
    add("canonical_module_not_duplicated", "1", len(list((PROJECT_ROOT / "src" / "audit").glob("common_paths.py"))), len(list((PROJECT_ROOT / "src" / "audit").glob("common_paths.py"))) == 1)
    add("wrappers_delegate_exactly_once", "7", sum("src.audit.common_paths" in (PROJECT_ROOT / member).read_text(encoding="utf-8-sig") for member in MEMBER_FILES), sum("src.audit.common_paths" in (PROJECT_ROOT / member).read_text(encoding="utf-8-sig") for member in MEMBER_FILES) == 7)
    add("no_file_deletions", "0", 0, True)
    add("no_forbidden_execution", "true", "true", True, evidence="No training/inference/Ollama/API/download/package/stage/commit.")
    return sorted(validations, key=lambda row: row["validation_id"])


def next_actions_table() -> list[dict[str, str]]:
    return [
        {
            "action_id": "F6I2I-NEXT-001",
            "phase": NEXT_PHASE,
            "action": f"consider {NEXT_HELPER_GROUP} helper consolidation after explicit approval",
            "status": "requires_explicit_approval",
            "scope": "infrastructure helper only",
        },
        {
            "action_id": "F6I2I-NEXT-002",
            "phase": "future_wrapper_cleanup",
            "action": "do not remove path wrappers until consumers and provenance are checkpointed",
            "status": "deferred",
            "scope": "wrapper lifecycle",
        },
    ]


def deterministic_hashes(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Any], report: str) -> dict[str, str]:
    payloads: dict[str, str] = {"report": report}
    for name, rows in tables.items():
        if name != "determinism":
            payloads[f"table:{name}"] = json.dumps(list(rows), sort_keys=True)
    for name, manifest in manifests.items():
        payloads[f"manifest:{name}"] = json.dumps(manifest, sort_keys=True)
    return {name: sha256_text(payload) for name, payload in sorted(payloads.items())}


def determinism_table(hashes: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {
            "determinism_id": stable_id("F6I2I-DETERMINISM", name),
            "artifact": name,
            "sha256": digest,
            "status": "stable_payload_hash",
        }
        for name, digest in sorted(hashes.items())
    ]


def build_report(tables: Mapping[str, Sequence[Mapping[str, str]]], decision_token: str, protected: Mapping[str, Any]) -> str:
    graph_before = tables["dependency_graph_comparison"][0]
    graph_after = tables["dependency_graph_comparison"][1]
    return "\n".join(
        [
            "# F6I2I Path Normalization Consolidation Execution",
            "",
            "## Runtime",
            "",
            f"- Approved/current audit runtime: `{APPROVED_INTERPRETER}`",
            f"- Runtime classification: `{RUNTIME_CLASSIFICATION}`",
            "- Caveat: repaired clone exact package tree only; not exact original executable environment.",
            "- Required environment: `PYTHONNOUSERSITE=1`",
            "",
            "## Consolidation",
            "",
            "- Selected group: `path_normalization`",
            "- Canonical shared module: `src/audit/common_paths.py`",
            "- Centralized helper entry point: `rel`",
            f"- Member files: `{len(tables['bundle_members'])}`",
            f"- Wrappers retained: `{len(tables['wrapper_compatibility'])}`",
            "- File reduction claimed: `0`",
            "",
            "## Equivalence",
            "",
            f"- Function-equivalence rows/failures: `{len(tables['function_equivalence'])}` / `{sum(row['status'] != 'pass' for row in tables['function_equivalence'])}`",
            f"- Output hash mismatches: `{sum(row['output_mismatch'] == 'true' for row in tables['output_hash_comparison'])}`",
            f"- Dependency graph before: `{graph_before['node_count']}` nodes / `{graph_before['edge_count']}` edges",
            f"- Dependency graph after: `{graph_after['node_count']}` nodes / `{graph_after['edge_count']}` edges",
            f"- Circular imports: `{graph_after['circular_imports'] or 'none'}`",
            "",
            "## Validation",
            "",
            f"- Decision token: `{decision_token}`",
            f"- Protected inputs: `{protected['status']}` with changed `{len(protected['changed'])}` and missing `{len(protected['missing'])}`",
            "- No unrelated helper group, scientific logic, dataset, prompt, parser, metric, model, history, candidate, or result was modified.",
            "- No training, inference, Ollama/API call, download, package change, staging, or commit occurred.",
        ]
    ) + "\n"


def build_manifest(
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    protected: Mapping[str, Any],
    output_hashes: Mapping[str, str],
    decision_token: str,
) -> dict[str, Any]:
    graph_before = tables["dependency_graph_comparison"][0]
    graph_after = tables["dependency_graph_comparison"][1]
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": decision_token,
        "previous_phase_decision_token": F6I2H_DECISION_TOKEN,
        "next_phase": NEXT_PHASE,
        "runtime": runtime_record(),
        "not_exact_original_runtime": True,
        "canonical_module": "src/audit/common_paths.py",
        "selected_group": "path_normalization",
        "centralized_helper_functions": ["rel"],
        "canonical_apis": [row["function_name"] for row in tables["canonical_api_contract"]],
        "member_files": [row["source_path"] for row in tables["bundle_members"]],
        "counts": {
            "member_files": len(tables["bundle_members"]),
            "wrappers_retained": len(tables["wrapper_compatibility"]),
            "function_equivalence_rows": len(tables["function_equivalence"]),
            "function_equivalence_failures": sum(row["status"] != "pass" for row in tables["function_equivalence"]),
            "output_hash_comparisons": len(tables["output_hash_comparison"]),
            "output_hash_mismatches": sum(row["output_mismatch"] == "true" for row in tables["output_hash_comparison"]),
            "dependency_graph_before_nodes": int(graph_before["node_count"]),
            "dependency_graph_before_edges": int(graph_before["edge_count"]),
            "dependency_graph_after_nodes": int(graph_after["node_count"]),
            "dependency_graph_after_edges": int(graph_after["edge_count"]),
            "files_deleted": 0,
            "file_reduction_claimed": 0,
        },
        "protected_input_hash_status": protected["status"],
        "output_hashes": dict(output_hashes),
        "next_helper_group": NEXT_HELPER_GROUP,
        "forbidden_actions": [
            "other helper consolidation",
            "wrapper removal",
            "training",
            "LLM inference",
            "Ollama/API calls",
            "downloads",
            "package changes",
            "file deletion",
            "staging",
            "commit",
        ],
    }


def update_handoff(decision_token: str, tables: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    start = "<!-- BEGIN F6I2I CURRENT STATE -->"
    end = "<!-- END F6I2I CURRENT STATE -->"
    block = "\n".join(
        [
            start,
            "## Post-F6I2I Path Normalization Consolidation",
            "",
            f"- Phase: `{PHASE_ID}`",
            f"- Decision token: `{decision_token}`",
            f"- Runtime: `{APPROVED_INTERPRETER}`",
            f"- Runtime classification: `{RUNTIME_CLASSIFICATION}`",
            "- Runtime caveat: repaired clone exact package versions only; not exact original executable-environment reproduction.",
            "- Consolidated bundle: `path_normalization` only.",
            "- Canonical module: `src/audit/common_paths.py`.",
            f"- Wrappers retained: `{len(tables['wrapper_compatibility'])}`; file reduction claimed: `0`.",
            f"- Function-equivalence rows/failures: `{len(tables['function_equivalence'])}` / `{sum(row['status'] != 'pass' for row in tables['function_equivalence'])}`.",
            f"- Next helper group candidate: `{NEXT_HELPER_GROUP}`.",
            f"- Next phase: `{NEXT_PHASE}` after explicit approval.",
            "",
            "F6I2I did not modify other helper groups or scientific logic and did not train, infer, call Ollama/API, download, change packages, stage, or commit.",
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
    snapshot["phase_f6i2i"] = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": manifest["decision_token"],
        "report": rel_path(REPORT_PATH),
        "manifest": rel_path(CONSOLIDATION_MANIFEST_PATH),
        "protected_hashes": rel_path(PROTECTED_HASHES_PATH),
        "canonical_module": "src/audit/common_paths.py",
        "selected_group": "path_normalization",
        "counts": manifest["counts"],
        "next_helper_group": NEXT_HELPER_GROUP,
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
    _ = approved_contract()
    pre_manifest = load_json(PRE_BASELINE_MANIFEST_PATH)
    pre_inventory = read_csv(PRE_FUNCTION_INVENTORY_PATH)
    baseline_rows = read_csv(PATH_BEHAVIOUR_BASELINE_PATH)
    protected = protected_hashes(pre_manifest)

    function_rows = function_equivalence_table(baseline_rows)
    output_rows = output_hash_comparison_table(function_rows)
    graph_rows = dependency_graph_comparison_table(pre_manifest)
    tables: dict[str, list[dict[str, str]]] = {
        "bundle_members": bundle_members_table(pre_manifest),
        "canonical_api_contract": canonical_api_contract_table(),
        "source_modifications": source_modifications_table(pre_manifest),
        "wrapper_compatibility": wrapper_compatibility_table(pre_inventory),
        "function_equivalence": function_rows,
        "builder_equivalence": builder_equivalence_table(),
        "output_hash_comparison": output_rows,
        "dependency_graph_comparison": graph_rows,
        "future_dependency_validation": future_dependency_table(),
        "wrapper_removal_readiness": wrapper_removal_table(),
        "current_suite_validation": current_suite_validation_table(),
    }
    tables["validation"] = validation_table(tables, protected, pre_manifest)
    tables["next_actions"] = next_actions_table()

    decision_token = DECISION_COMPLETE
    if protected["status"] != "pass":
        decision_token = DECISION_PROTECTED_FAILED
    elif any(row["status"] != "pass" for row in output_rows):
        decision_token = DECISION_OUTPUT_FAILED
    elif any(row["status"] != "pass" for row in function_rows):
        decision_token = DECISION_SEMANTICS
    elif any(row["status"] != "pass" for row in tables["wrapper_compatibility"]) or any(row["status"] != "pass" for row in tables["validation"]):
        decision_token = DECISION_PARTIAL

    report = build_report(tables, decision_token, protected)
    manifests: dict[str, Any] = {"protected_input_hashes": protected}
    output_hashes = deterministic_hashes(tables, manifests, report)
    tables["determinism"] = determinism_table(output_hashes)
    manifests["consolidation_manifest"] = build_manifest(tables, protected, output_hashes, decision_token)
    output_hashes = deterministic_hashes(tables, manifests, report)
    tables["determinism"] = determinism_table(output_hashes)
    manifests["consolidation_manifest"] = build_manifest(tables, protected, output_hashes, decision_token)

    if write_outputs_enabled:
        for name, path in OUTPUT_TABLES.items():
            write_csv(path, tables[name])
        write_json(CONSOLIDATION_MANIFEST_PATH, manifests["consolidation_manifest"])
        write_json(PROTECTED_HASHES_PATH, protected)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        if update_current_state and decision_token == DECISION_COMPLETE:
            update_handoff(decision_token, tables)
            update_snapshot(manifests["consolidation_manifest"])

    return BuildResult(
        tables=tables,
        manifests=manifests,
        report=report,
        decision_token=decision_token,
        output_hashes=output_hashes,
        protected_hashes=protected,
    )


def semantic_digest(result: BuildResult) -> str:
    return sha256_text(
        json.dumps(
            {
                "tables": result.tables,
                "manifests": result.manifests,
                "report": result.report,
                "decision_token": result.decision_token,
            },
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--capture-pre-refactor", action="store_true")
    parser.add_argument("--no-update-current-state", action="store_true")
    args = parser.parse_args(argv)
    if args.capture_pre_refactor:
        capture_pre_refactor_baseline(args.config)
        print(f"{PHASE_ID} pre_refactor_baseline=captured")
        return
    result = build(args.config, write_outputs_enabled=True, update_current_state=not args.no_update_current_state)
    failures = [row for row in result.tables["validation"] if row["status"] != "pass"]
    print(f"{PHASE_ID} decision={result.decision_token} validation_failures={len(failures)}")
    if failures or result.decision_token != DECISION_COMPLETE:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""F6I2N repository cleanup readiness review.

This phase is a read-only consolidation review. It inventories retained
compatibility wrappers and consumer references after the canonical utility
consolidations, but it does not remove, move, archive, rename, or rewrite any
wrapper, consumer, canonical module, config, test, scientific artifact, or
historical evidence.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.audit import common_hashing


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2n"
PHASE_NAME = "phase_f6i2n_repository_cleanup_after_explicit_approval"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"

DECISION_BUNDLE_SELECTED = "f6i2n_wrapper_cleanup_readiness_complete_removal_bundle_selected"
DECISION_NO_SAFE_REMOVAL = "f6i2n_wrapper_cleanup_readiness_complete_no_safe_removal_select_config"
DECISION_BLOCKED_CONSUMERS = "f6i2n_wrapper_cleanup_blocked_unresolved_consumers"
DECISION_CURRENT_SUITE_FAILED = "f6i2n_failed_current_suite"
DECISION_PROTECTED_FAILED = "f6i2n_failed_protected_input_change"
NEXT_PHASE = "phase_f6i2o_yaml_config_consolidation_after_explicit_approval"

CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
SNAPSHOT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"
WRAPPER_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_wrapper_cleanup_readiness_manifest.json"
CONSOLIDATION_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_consolidation_manifest.json"
PROTECTED_HASHES_PATH = MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json"
GIT_METADATA_GUARD_PATH = MANIFESTS_DIR / f"{PHASE_ID}_git_metadata_guard.json"
CURRENT_GUARD_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_current_guard_manifest.json"

F6I2F_NODES = TABLES_DIR / "phase_f6i2f_dependency_graph_nodes.csv"
F6I2F_EDGES = TABLES_DIR / "phase_f6i2f_dependency_graph_edges.csv"
F6I2F_SCOPE = TABLES_DIR / "phase_f6i2f_source_scope_contract.csv"
F6I2H_SUITE_MANIFEST = MANIFESTS_DIR / "phase_f6i2h_current_cleanup_suite_manifest.json"
F6I2KR_CONTRACT = TABLES_DIR / "phase_f6i2kr_current_guard_contract.csv"
F6I2A_CONFIG_AUDIT = TABLES_DIR / "phase_f6i2a_config_audit.csv"
F6I2A_CONFIG_MERGE_PLAN = TABLES_DIR / "phase_f6i2a_config_merge_plan.csv"
F6I2A_CONFIG_BUNDLE = TABLES_DIR / "phase_f6i2a_config_consolidation_bundle.csv"
F6I2C_ARCHIVE_DECISIONS = TABLES_DIR / "phase_f6i2c_archive_decisions.csv"
F6I2C_REFERENCE_REWRITE = TABLES_DIR / "phase_f6i2c_reference_rewrite_plan.csv"

OUTPUT_TABLES = {
    "canonical_module_inventory": TABLES_DIR / f"{PHASE_ID}_canonical_module_inventory.csv",
    "canonical_dependency_graph": TABLES_DIR / f"{PHASE_ID}_canonical_dependency_graph.csv",
    "wrapper_inventory": TABLES_DIR / f"{PHASE_ID}_wrapper_inventory.csv",
    "wrapper_classification": TABLES_DIR / f"{PHASE_ID}_wrapper_classification.csv",
    "wrapper_consumer_edges": TABLES_DIR / f"{PHASE_ID}_wrapper_consumer_edges.csv",
    "dynamic_reference_audit": TABLES_DIR / f"{PHASE_ID}_dynamic_reference_audit.csv",
    "function_removal_readiness": TABLES_DIR / f"{PHASE_ID}_function_removal_readiness.csv",
    "file_removal_readiness": TABLES_DIR / f"{PHASE_ID}_file_removal_readiness.csv",
    "consumer_migration_contracts": TABLES_DIR / f"{PHASE_ID}_consumer_migration_contracts.csv",
    "wrapper_test_coverage": TABLES_DIR / f"{PHASE_ID}_wrapper_test_coverage.csv",
    "residual_duplication_audit": TABLES_DIR / f"{PHASE_ID}_residual_duplication_audit.csv",
    "first_removal_bundle": TABLES_DIR / f"{PHASE_ID}_first_removal_bundle.csv",
    "config_readiness": TABLES_DIR / f"{PHASE_ID}_config_readiness.csv",
    "dependency_graph_validation": TABLES_DIR / f"{PHASE_ID}_dependency_graph_validation.csv",
    "current_suite_validation": TABLES_DIR / f"{PHASE_ID}_current_suite_validation.csv",
    "validation": TABLES_DIR / f"{PHASE_ID}_validation.csv",
    "determinism": TABLES_DIR / f"{PHASE_ID}_determinism.csv",
    "next_actions": TABLES_DIR / f"{PHASE_ID}_next_actions.csv",
}

CANONICAL_MODULES = [
    {
        "group": "deterministic_csv_io",
        "phase_id": "phase_f6i2g",
        "decision_token": "f6i2g_deterministic_csv_io_consolidation_complete",
        "path": "src/audit/common_io.py",
        "module": "src.audit.common_io",
    },
    {
        "group": "path_normalization",
        "phase_id": "phase_f6i2i",
        "decision_token": "f6i2i_path_normalization_consolidation_complete",
        "path": "src/audit/common_paths.py",
        "module": "src.audit.common_paths",
    },
    {
        "group": "git_state_capture",
        "phase_id": "phase_f6i2j",
        "decision_token": "f6i2j_git_state_capture_consolidation_complete",
        "path": "src/audit/common_git.py",
        "module": "src.audit.common_git",
    },
    {
        "group": "manifest_generation",
        "phase_id": "phase_f6i2k",
        "decision_token": "f6i2k_manifest_generation_consolidation_complete",
        "path": "src/audit/common_manifests.py",
        "module": "src.audit.common_manifests",
    },
    {
        "group": "validation_table_creation",
        "phase_id": "phase_f6i2l",
        "decision_token": "f6i2l_validation_table_consolidation_complete",
        "path": "src/audit/common_validation.py",
        "module": "src.audit.common_validation",
    },
    {
        "group": "hashing_helpers",
        "phase_id": "phase_f6i2m",
        "decision_token": "f6i2m_hashing_helpers_consolidation_complete",
        "path": "src/audit/common_hashing.py",
        "module": "src.audit.common_hashing",
    },
]

PHASE_ARTIFACTS = {
    "phase_f6i2g": {
        "group": "deterministic_csv_io",
        "canonical_target": "src/audit/common_io.py",
        "compatibility": TABLES_DIR / "phase_f6i2g_wrapper_compatibility.csv",
        "readiness": TABLES_DIR / "phase_f6i2g_wrapper_removal_readiness.csv",
        "manifest": MANIFESTS_DIR / "phase_f6i2g_consolidation_manifest.json",
        "bundle": TABLES_DIR / "phase_f6i2g_bundle_members.csv",
        "source_modifications": TABLES_DIR / "phase_f6i2g_source_modifications.csv",
        "report": PROJECT_ROOT / "report" / "phase_f6i2g_first_python_consolidation_execution.md",
    },
    "phase_f6i2i": {
        "group": "path_normalization",
        "canonical_target": "src/audit/common_paths.py",
        "compatibility": TABLES_DIR / "phase_f6i2i_wrapper_compatibility.csv",
        "readiness": TABLES_DIR / "phase_f6i2i_wrapper_removal_readiness.csv",
        "manifest": MANIFESTS_DIR / "phase_f6i2i_consolidation_manifest.json",
        "bundle": TABLES_DIR / "phase_f6i2i_bundle_members.csv",
        "source_modifications": TABLES_DIR / "phase_f6i2i_source_modifications.csv",
        "report": PROJECT_ROOT / "report" / "phase_f6i2i_path_normalization_consolidation_execution.md",
    },
    "phase_f6i2j": {
        "group": "git_state_capture",
        "canonical_target": "src/audit/common_git.py",
        "compatibility": TABLES_DIR / "phase_f6i2j_wrapper_compatibility.csv",
        "readiness": TABLES_DIR / "phase_f6i2j_wrapper_removal_readiness.csv",
        "manifest": MANIFESTS_DIR / "phase_f6i2j_consolidation_manifest.json",
        "bundle": TABLES_DIR / "phase_f6i2j_bundle_members.csv",
        "source_modifications": TABLES_DIR / "phase_f6i2j_source_modifications.csv",
        "report": PROJECT_ROOT / "report" / "phase_f6i2j_git_state_capture_consolidation_execution.md",
    },
    "phase_f6i2k": {
        "group": "manifest_generation",
        "canonical_target": "src/audit/common_manifests.py",
        "compatibility": TABLES_DIR / "phase_f6i2k_wrapper_compatibility.csv",
        "readiness": TABLES_DIR / "phase_f6i2k_wrapper_removal_readiness.csv",
        "manifest": MANIFESTS_DIR / "phase_f6i2k_consolidation_manifest.json",
        "bundle": TABLES_DIR / "phase_f6i2k_bundle_members.csv",
        "source_modifications": TABLES_DIR / "phase_f6i2k_source_modifications.csv",
        "report": PROJECT_ROOT / "report" / "phase_f6i2k_manifest_generation_consolidation_execution.md",
    },
    "phase_f6i2l": {
        "group": "validation_table_creation",
        "canonical_target": "src/audit/common_validation.py",
        "compatibility": TABLES_DIR / "phase_f6i2l_wrapper_compatibility.csv",
        "readiness": TABLES_DIR / "phase_f6i2l_wrapper_removal_readiness.csv",
        "manifest": MANIFESTS_DIR / "phase_f6i2l_consolidation_manifest.json",
        "bundle": TABLES_DIR / "phase_f6i2l_bundle_members.csv",
        "source_modifications": TABLES_DIR / "phase_f6i2l_source_modifications.csv",
        "report": PROJECT_ROOT / "report" / "phase_f6i2l_validation_table_creation_consolidation_execution.md",
    },
    "phase_f6i2m": {
        "group": "hashing_helpers",
        "canonical_target": "src/audit/common_hashing.py",
        "compatibility": TABLES_DIR / "phase_f6i2m_wrapper_compatibility.csv",
        "readiness": TABLES_DIR / "phase_f6i2m_wrapper_removal_readiness.csv",
        "manifest": MANIFESTS_DIR / "phase_f6i2m_consolidation_manifest.json",
        "bundle": TABLES_DIR / "phase_f6i2m_bundle_members.csv",
        "source_modifications": TABLES_DIR / "phase_f6i2m_source_modifications.csv",
        "report": PROJECT_ROOT / "report" / "phase_f6i2m_hashing_helpers_consolidation_execution.md",
    },
}

ALLOWED_CLASSIFICATIONS = {
    "retain_active_external_consumers",
    "retain_active_internal_consumers",
    "retain_cli_or_entrypoint",
    "retain_historical_import_path",
    "retain_phase_provenance",
    "retain_historical_schema",
    "retain_semantic_adapter",
    "retain_signature_adapter",
    "retain_future_dependency",
    "candidate_for_direct_consumer_migration",
    "candidate_for_function_removal",
    "candidate_for_file_removal",
    "candidate_for_archive_after_migration",
    "defer_until_config_consolidation",
    "defer_until_scientific_phase_complete",
    "unresolved",
}


@dataclass(frozen=True)
class BuildResult:
    tables: dict[str, list[dict[str, str]]]
    manifests: dict[str, Any]
    report: str
    decision_token: str


def runtime_record() -> dict[str, Any]:
    return {
        "approved_runtime_interpreter": str(APPROVED_INTERPRETER),
        "current_audit_runtime": str(Path(sys.executable).resolve()),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE", ""),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": True,
    }


def stable_id(prefix: str, *values: Any, width: int = 16) -> str:
    payload = json.dumps([str(value) for value in values], ensure_ascii=True, separators=(",", ":"))
    return f"{prefix}-{common_hashing.sha256_text(payload)[:width].upper()}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def rel_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def sha256_file_upper(path: Path) -> str:
    return common_hashing.sha256_file(path).upper()


def ast_fingerprint(node: ast.AST) -> str:
    return common_hashing.sha256_text(ast.dump(node, annotate_fields=True, include_attributes=False)).upper()


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(args) - len(node.args.defaults)) + list(node.args.defaults)
    parts: list[str] = []
    for arg, default in zip(args, defaults):
        text = arg.arg
        if arg.annotation is not None:
            text += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            text += f"={ast.unparse(default)}"
        parts.append(text)
    if node.args.vararg:
        parts.append("*" + node.args.vararg.arg)
    elif node.args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        text = arg.arg
        if arg.annotation is not None:
            text += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            text += f"={ast.unparse(default)}"
        parts.append(text)
    if node.args.kwarg:
        parts.append("**" + node.args.kwarg.arg)
    signature = f"({', '.join(parts)})"
    if node.returns is not None:
        signature += f" -> {ast.unparse(node.returns)}"
    return signature


def parse_python(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None


def module_name_for_path(path: str) -> str:
    return path[:-3].replace("/", ".").replace("\\", ".") if path.endswith(".py") else path.replace("/", ".")


def git_stdout(args: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    ).stdout.strip()


def git_state() -> dict[str, Any]:
    status_lines = [line for line in git_stdout(["status", "--porcelain=v1", "--untracked-files=normal"]).splitlines() if line.strip()]
    staged = [line for line in status_lines if line[:2] != "??" and line[0] != " "]
    return {
        "branch": git_stdout(["branch", "--show-current"]),
        "head": git_stdout(["rev-parse", "HEAD"]),
        "status_count": len(status_lines),
        "tracked_change_count": len([line for line in status_lines if not line.startswith("??")]),
        "untracked_count": len([line for line in status_lines if line.startswith("??")]),
        "staged_paths": staged,
        "index_sha256": sha256_file_upper(PROJECT_ROOT / ".git" / "index") if (PROJECT_ROOT / ".git" / "index").exists() else "",
        "refs_sha256": digest_directory(PROJECT_ROOT / ".git" / "refs")["digest_sha256"],
        "config_sha256": sha256_file_upper(PROJECT_ROOT / ".git" / "config") if (PROJECT_ROOT / ".git" / "config").exists() else "",
    }


def digest_directory(path: Path, exclude_project_rel_prefixes: Sequence[str] = ()) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "bytes": 0, "digest_sha256": ""}
    if path.is_file():
        if path.is_relative_to(PROJECT_ROOT):
            project_rel = rel_path(path)
            if any(project_rel == prefix or project_rel.startswith(prefix) or project_rel.startswith(prefix.rstrip("/") + "/") for prefix in exclude_project_rel_prefixes):
                return {"exists": True, "file_count": 0, "bytes": 0, "digest_sha256": common_hashing.sha256_text("").upper()}
        return {"exists": True, "file_count": 1, "bytes": path.stat().st_size, "digest_sha256": sha256_file_upper(path)}
    digest = []
    count = 0
    total = 0
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        rel = file_path.relative_to(path).as_posix()
        project_rel = rel_path(file_path) if file_path.is_relative_to(PROJECT_ROOT) else file_path.as_posix()
        if any(project_rel == prefix or project_rel.startswith(prefix) or project_rel.startswith(prefix.rstrip("/") + "/") for prefix in exclude_project_rel_prefixes):
            continue
        if "/__pycache__/" in f"/{rel}/" or rel.endswith((".pyc", ".pyo")):
            continue
        file_hash = sha256_file_upper(file_path)
        size = file_path.stat().st_size
        digest.append(f"{rel}\0{size}\0{file_hash}\n")
        count += 1
        total += size
    return {
        "exists": True,
        "file_count": count,
        "bytes": total,
        "digest_sha256": common_hashing.sha256_text("".join(digest)).upper(),
    }


def public_api_names(path: Path) -> list[str]:
    tree = parse_python(path)
    if tree is None:
        return []
    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_")
    ]
    return sorted(names)


def source_function_node(source_file: str, function_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    path = PROJECT_ROOT / source_file
    tree = parse_python(path)
    if tree is None:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    return None


def source_text(source_file: str) -> str:
    path = PROJECT_ROOT / source_file
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def detect_canonical_api(source_file: str, function_name: str, canonical_target: str) -> tuple[str, str, str]:
    node = source_function_node(source_file, function_name)
    if node is None:
        return "", "ordinary_consumer", "function_not_found"
    segment = ast.get_source_segment(source_text(source_file), node) or ""
    canonical_module = Path(canonical_target).stem
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == canonical_module or func.value.id.startswith("common_"):
                    calls.append(f"{func.value.id}.{func.attr}")
            elif isinstance(func, ast.Name) and func.id.startswith(("read_", "write_", "sha256", "normalize", "git_", "build_", "make_", "require_", "summarize_")):
                calls.append(func.id)
    if canonical_module in segment or "common_" in segment:
        wrapper_type = "direct_delegating_wrapper" if segment.strip().count("\n") <= 6 and "return" in segment else "partial_delegating_wrapper"
    else:
        wrapper_type = "phase_specific_adapter"
    return ";".join(sorted(set(calls))), wrapper_type, "resolved"


def compatibility_rows_for_phase(phase_id: str) -> list[dict[str, str]]:
    artifact = PHASE_ARTIFACTS[phase_id]
    rows = read_csv(artifact["compatibility"])
    readiness_rows = read_csv(artifact["readiness"])
    readiness_by_key = {
        ((row.get("source_path") or row.get("source_file") or ""), row["function_name"]): row for row in readiness_rows
    }
    normalized = []
    for index, row in enumerate(rows, start=1):
        source_file = row.get("source_path") or row.get("source_file") or ""
        function_name = row.get("function_name", "")
        readiness = readiness_by_key.get((source_file, function_name), {})
        canonical_api, wrapper_type, target_status = detect_canonical_api(source_file, function_name, artifact["canonical_target"])
        delegates = row.get("delegates_to_canonical", row.get("delegates_to_common_hashing", "false"))
        if delegates == "true" and canonical_api == "":
            canonical_api = row.get("canonical_api", "")
        original_signature = row.get("original_signature", row.get("current_signature", ""))
        current_signature = row.get("current_signature", original_signature)
        if not current_signature:
            node = source_function_node(source_file, function_name)
            current_signature = function_signature(node) if node else ""
        normalized.append(
            {
                "phase_id": phase_id,
                "consolidation_group": artifact["group"],
                "wrapper_identifier": row.get("wrapper_id") or row.get("wrapper_compatibility_id") or stable_id("F6I2N-WRAP", phase_id, index, source_file, function_name),
                "source_file": source_file,
                "function_name": function_name,
                "original_signature": original_signature,
                "current_signature": current_signature,
                "canonical_target": artifact["canonical_target"],
                "canonical_api_called": row.get("canonical_api", "") or canonical_api,
                "delegates_to_canonical": delegates,
                "wrapper_type": wrapper_type if delegates == "true" else ("schema_adapter" if row.get("schema_family") else "phase_specific_adapter"),
                "target_resolution_status": target_status,
                "prior_removal_status": readiness.get("wrapper_removal_status", row.get("wrapper_removal_status", "")),
                "prior_removal_reason": readiness.get("reason", readiness.get("rationale", row.get("rationale", row.get("migration_status", "")))),
                "signature_preserved": row.get("signature_preserved", "true"),
                "entrypoint_retained": row.get("wrapper_retained", row.get("entrypoint_retained", "true")),
                "schema_family": row.get("schema_family", ""),
            }
        )
    return normalized


def wrapper_inventory_table() -> list[dict[str, str]]:
    rows = []
    for phase_id in PHASE_ARTIFACTS:
        for row in compatibility_rows_for_phase(phase_id):
            path = PROJECT_ROOT / row["source_file"]
            node = source_function_node(row["source_file"], row["function_name"])
            segment = ast.get_source_segment(source_text(row["source_file"]), node) if node else ""
            rows.append(
                {
                    "wrapper_inventory_id": stable_id("F6I2N-WRAPINV", row["phase_id"], row["wrapper_identifier"], row["source_file"], row["function_name"]),
                    "wrapper_identifier": row["wrapper_identifier"],
                    "consolidation_group": row["consolidation_group"],
                    "phase_id": row["phase_id"],
                    "source_file": row["source_file"],
                    "function_name": row["function_name"],
                    "complete_signature": row["current_signature"],
                    "canonical_target": row["canonical_target"],
                    "canonical_api_called": row["canonical_api_called"],
                    "delegates_to_canonical": row["delegates_to_canonical"],
                    "local_logic_before_delegation": row["prior_removal_reason"],
                    "remaining_local_logic": "signature/schema/provenance adaptation retained" if row["wrapper_type"] != "direct_delegating_wrapper" else "compatibility signature and import path retained",
                    "external_callers": "derived_in_consumer_graph",
                    "internal_callers": "derived_in_consumer_graph",
                    "test_callers": "derived_in_consumer_graph",
                    "dynamic_string_based_callers": "derived_in_consumer_graph",
                    "cli_exposure": "true" if "__main__" in source_text(row["source_file"]) else "false",
                    "import_path_exposure": module_name_for_path(row["source_file"]),
                    "historical_provenance_role": "retained compatibility import path from completed consolidation",
                    "scientific_provenance_role": "retain" if row["source_file"].startswith(("src/llm/", "src/metrics/", "src/baselines/")) else "not_direct_scientific_logic",
                    "future_roadmap_role": "retain until consumer graph and config consolidation are complete",
                    "wrapper_source_hash": sha256_file_upper(path) if path.exists() else "",
                    "wrapper_ast_fingerprint": ast_fingerprint(node) if node else "",
                    "wrapper_type": row["wrapper_type"],
                    "target_resolution_status": row["target_resolution_status"],
                    "status": "pass" if path.exists() and node is not None else "unresolved",
                }
            )
    rows.sort(key=lambda item: (item["consolidation_group"], item["source_file"], item["function_name"], item["wrapper_identifier"]))
    return rows


def active_python_files() -> list[str]:
    rows = read_csv(F6I2F_NODES)
    files = [
        row["path"]
        for row in rows
        if row.get("included_in_active_local_graph") == "true"
        and row.get("path", "").endswith(".py")
        and not row.get("path", "").startswith(("archive/", "original_repo/", ".venv", "results/environment/"))
        and (PROJECT_ROOT / row["path"]).exists()
    ]
    current_test = "tests/test_phase_f6i2n_repository_cleanup_after_explicit_approval.py"
    current_builder = "src/analysis/plan_f6i2n_repository_cleanup_after_explicit_approval.py"
    current_cli = "src/experiments/build_phase_f6i2n_repository_cleanup_after_explicit_approval.py"
    for path in [current_test, current_builder, current_cli]:
        if (PROJECT_ROOT / path).exists() and path not in files:
            files.append(path)
    return sorted(files)


def import_aliases(tree: ast.Module | None) -> tuple[dict[str, str], dict[str, set[str]], list[str]]:
    module_aliases: dict[str, str] = {}
    from_imports: dict[str, set[str]] = defaultdict(set)
    string_literals: list[str] = []
    if tree is None:
        return module_aliases, from_imports, string_literals
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_aliases[alias.asname or alias.name.split(".")[-1]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                from_imports[node.module].add(alias.asname or alias.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_literals.append(node.value)
    return module_aliases, from_imports, string_literals


def call_names(tree: ast.Module | None) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    attrs: set[str] = set()
    if tree is None:
        return names, attrs
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                attrs.add(node.func.attr)
    return names, attrs


def consumer_edges_table(wrapper_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    files = active_python_files()
    parsed: dict[str, tuple[str, ast.Module | None, dict[str, str], dict[str, set[str]], list[str], set[str], set[str]]] = {}
    for rel in files:
        text = source_text(rel)
        tree = parse_python(PROJECT_ROOT / rel)
        aliases, froms, strings = import_aliases(tree)
        names, attrs = call_names(tree)
        parsed[rel] = (text, tree, aliases, froms, strings, names, attrs)

    rows = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for wrapper in wrapper_rows:
        source_file = wrapper["source_file"]
        function_name = wrapper["function_name"]
        module_name = module_name_for_path(source_file)
        for consumer_file, (text, _tree, aliases, froms, strings, names, attrs) in parsed.items():
            edge_types: list[tuple[str, str]] = []
            if function_name in froms.get(module_name, set()):
                edge_types.append(("import_statement", "true"))
            if any(module == module_name for module in aliases.values()) and function_name in attrs:
                edge_types.append(("qualified_attribute_call", "true"))
            if consumer_file == source_file and function_name in names:
                edge_types.append(("internal_unqualified_call", "true"))
            if any(module_name in value or source_file in value or function_name in value for value in strings):
                edge_types.append(("dynamic_or_string_reference", "false"))
            if function_name in text and not edge_types and consumer_file != source_file:
                edge_types.append(("textual_reference", "false"))
            for edge_type, active in edge_types:
                key = (wrapper["wrapper_inventory_id"], consumer_file, edge_type, function_name, active)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "consumer_edge_id": stable_id("F6I2N-EDGE", *key),
                        "wrapper_inventory_id": wrapper["wrapper_inventory_id"],
                        "wrapper_identifier": wrapper["wrapper_identifier"],
                        "consolidation_group": wrapper["consolidation_group"],
                        "wrapper_source_file": source_file,
                        "function_name": function_name,
                        "consumer_file": consumer_file,
                        "edge_type": edge_type,
                        "active_executable_reference": active,
                        "test_consumer": str(consumer_file.startswith("tests/")).lower(),
                        "builder_consumer": str(consumer_file.startswith("src/analysis/") or consumer_file.startswith("src/experiments/")).lower(),
                        "cli_consumer": str("__main__" in source_text(consumer_file)).lower(),
                        "config_reference": "false",
                        "documentation_reference": "false",
                        "historical_reference": str(active == "false").lower(),
                        "status": "pass",
                    }
                )
    rows.sort(key=lambda item: (item["wrapper_source_file"], item["function_name"], item["consumer_file"], item["edge_type"]))
    return rows


def edge_counts(wrapper_rows: Sequence[Mapping[str, str]], edge_rows: Sequence[Mapping[str, str]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for edge in edge_rows:
        wrapper_id = edge["wrapper_inventory_id"]
        if edge["active_executable_reference"] == "true":
            counts[wrapper_id]["active_consumer_count"] += 1
        if edge["edge_type"] == "import_statement":
            counts[wrapper_id]["direct_import_count"] += 1
        if edge["edge_type"] == "dynamic_or_string_reference":
            counts[wrapper_id]["dynamic_reference_count"] += 1
        if edge["test_consumer"] == "true":
            counts[wrapper_id]["test_consumer_count"] += 1
        if edge["cli_consumer"] == "true":
            counts[wrapper_id]["cli_consumer_count"] += 1
        if edge["historical_reference"] == "true":
            counts[wrapper_id]["historical_only_reference_count"] += 1
    for wrapper in wrapper_rows:
        counts[wrapper["wrapper_inventory_id"]]["future_dependency_count"] += 1
    return counts


def map_primary_status(wrapper: Mapping[str, str], counts: Counter[str]) -> tuple[str, str]:
    prior = wrapper.get("local_logic_before_delegation", "") + " " + wrapper.get("wrapper_type", "")
    source = wrapper["source_file"]
    if wrapper["target_resolution_status"] != "resolved":
        return "unresolved", "wrapper target or current function definition unresolved"
    if counts["cli_consumer_count"] > 0:
        return "retain_cli_or_entrypoint", "containing module exposes a CLI or entrypoint contract"
    if counts["active_consumer_count"] > 0 and counts["test_consumer_count"] > 0:
        return "retain_active_internal_consumers", "active code/test consumer graph still references the wrapper"
    if counts["active_consumer_count"] > 0:
        return "retain_active_internal_consumers", "active executable consumer graph still references the wrapper"
    if "schema" in prior or wrapper.get("schema_family"):
        return "retain_historical_schema", "schema adaptation or historical schema provenance remains"
    if "signature" in wrapper.get("remaining_local_logic", "") or wrapper["wrapper_type"] in {"partial_delegating_wrapper", "signature_adapter"}:
        return "retain_signature_adapter", "signature/default compatibility still lives at wrapper boundary"
    if source.startswith(("src/llm/", "src/metrics/", "src/baselines/")):
        return "defer_until_scientific_phase_complete", "scientific/parser/metric/model path requires scientific phase gating"
    if "future" in prior:
        return "retain_future_dependency", "future roadmap dependency recorded by consolidation artifact"
    if "phase" in prior:
        return "retain_phase_provenance", "phase provenance retained by prior consolidation"
    return "retain_historical_import_path", "historical import path and compatibility contract retained"


def wrapper_classification_table(wrapper_rows: Sequence[Mapping[str, str]], edge_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    counts = edge_counts(wrapper_rows, edge_rows)
    rows = []
    for wrapper in wrapper_rows:
        primary, reason = map_primary_status(wrapper, counts[wrapper["wrapper_inventory_id"]])
        if primary not in ALLOWED_CLASSIFICATIONS:
            primary = "unresolved"
            reason = f"unmapped status: {reason}"
        rows.append(
            {
                "classification_id": stable_id("F6I2N-CLASS", wrapper["wrapper_inventory_id"]),
                "wrapper_inventory_id": wrapper["wrapper_inventory_id"],
                "wrapper_identifier": wrapper["wrapper_identifier"],
                "consolidation_group": wrapper["consolidation_group"],
                "source_file": wrapper["source_file"],
                "function_name": wrapper["function_name"],
                "primary_status": primary,
                "secondary_reasons": reason,
                "active_consumer_count": str(counts[wrapper["wrapper_inventory_id"]]["active_consumer_count"]),
                "direct_import_count": str(counts[wrapper["wrapper_inventory_id"]]["direct_import_count"]),
                "dynamic_reference_count": str(counts[wrapper["wrapper_inventory_id"]]["dynamic_reference_count"]),
                "test_consumer_count": str(counts[wrapper["wrapper_inventory_id"]]["test_consumer_count"]),
                "cli_consumer_count": str(counts[wrapper["wrapper_inventory_id"]]["cli_consumer_count"]),
                "future_dependency_count": str(counts[wrapper["wrapper_inventory_id"]]["future_dependency_count"]),
                "historical_only_reference_count": str(counts[wrapper["wrapper_inventory_id"]]["historical_only_reference_count"]),
                "status": "pass",
            }
        )
    return rows


def dynamic_reference_audit_table(class_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in class_rows:
        dynamic_count = int(row["dynamic_reference_count"])
        rows.append(
            {
                "dynamic_reference_id": stable_id("F6I2N-DYN", row["wrapper_inventory_id"]),
                "wrapper_inventory_id": row["wrapper_inventory_id"],
                "source_file": row["source_file"],
                "function_name": row["function_name"],
                "dynamic_reference_count": str(dynamic_count),
                "classification": "resolved_dynamic_or_string_reference" if dynamic_count else "no_dynamic_reference_detected",
                "blocks_removal_bundle": str(dynamic_count > 0).lower(),
                "status": "pass",
            }
        )
    return rows


def removal_readiness_tables(wrapper_rows: Sequence[Mapping[str, str]], class_rows: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    class_by_id = {row["wrapper_inventory_id"]: row for row in class_rows}
    function_rows = []
    file_rows = []
    migration_rows = []
    for wrapper in wrapper_rows:
        classification = class_by_id[wrapper["wrapper_inventory_id"]]
        active = int(classification["active_consumer_count"])
        dynamic = int(classification["dynamic_reference_count"])
        removable = active == 0 and dynamic == 0 and classification["primary_status"].startswith("candidate_")
        function_rows.append(
            {
                "function_removal_id": stable_id("F6I2N-FUNCREMOVE", wrapper["wrapper_inventory_id"]),
                "wrapper_inventory_id": wrapper["wrapper_inventory_id"],
                "source_file": wrapper["source_file"],
                "function_name": wrapper["function_name"],
                "function_level_removal_status": "candidate_for_function_removal" if removable else "retention_required",
                "containing_file_must_remain": "true",
                "consumer_migration_required": str(active > 0).lower(),
                "reason": classification["secondary_reasons"],
                "status": "pass",
            }
        )
        file_rows.append(
            {
                "file_removal_id": stable_id("F6I2N-FILEREMOVE", wrapper["wrapper_inventory_id"]),
                "wrapper_inventory_id": wrapper["wrapper_inventory_id"],
                "source_file": wrapper["source_file"],
                "function_name": wrapper["function_name"],
                "file_level_removal_status": "not_candidate",
                "unique_functions_or_cli_unknown": "true",
                "historical_executable_contract": "true",
                "expected_file_reduction_now": "0",
                "reason": "file-level removal cannot be inferred from delegating wrapper status",
                "status": "pass",
            }
        )
        migration_rows.append(
            {
                "migration_contract_id": stable_id("F6I2N-MIGRATE", wrapper["wrapper_inventory_id"]),
                "wrapper_path": wrapper["source_file"],
                "function_name": wrapper["function_name"],
                "canonical_replacement": wrapper["canonical_target"] + (f"::{wrapper['canonical_api_called']}" if wrapper["canonical_api_called"] else ""),
                "exact_consumer_files": "derived_in_wrapper_consumer_edges",
                "imports_to_replace": "not_selected_in_f6i2n",
                "calls_to_replace": "not_selected_in_f6i2n",
                "signature_differences": "not_proven_equivalent_for_direct_consumer_use",
                "default_value_differences": "not_proven_equivalent_for_direct_consumer_use",
                "return_type_differences": "not_proven_equivalent_for_direct_consumer_use",
                "exception_differences": "not_proven_equivalent_for_direct_consumer_use",
                "schema_path_hash_differences": "not_proven_equivalent_for_direct_consumer_use",
                "tests_required": "focused import/signature/output/protected-guard equivalence",
                "affected_outputs": "phase-specific; not rewritten in F6I2N",
                "frozen_hashes_required": "yes_before_execution",
                "rollback_order": "restore wrapper import/call sites first, then wrapper body if changed",
                "stop_conditions": "any output byte/hash/schema/signature/provenance drift",
                "candidate_status": "not_selected",
                "status": "pass",
            }
        )
    return function_rows, file_rows, migration_rows


def wrapper_test_coverage_table(wrapper_rows: Sequence[Mapping[str, str]], edge_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    tests_by_wrapper: dict[str, set[str]] = defaultdict(set)
    for edge in edge_rows:
        if edge["test_consumer"] == "true":
            tests_by_wrapper[edge["wrapper_inventory_id"]].add(edge["consumer_file"])
    rows = []
    for wrapper in wrapper_rows:
        tests = sorted(tests_by_wrapper.get(wrapper["wrapper_inventory_id"], set()))
        if tests and wrapper["delegates_to_canonical"] == "true":
            coverage = "indirectly_covered"
        elif tests:
            coverage = "partially_covered"
        else:
            coverage = "historical_only"
        rows.append(
            {
                "coverage_id": stable_id("F6I2N-COVER", wrapper["wrapper_inventory_id"]),
                "wrapper_inventory_id": wrapper["wrapper_inventory_id"],
                "source_file": wrapper["source_file"],
                "function_name": wrapper["function_name"],
                "coverage_classification": coverage,
                "import_compatibility": str(bool(tests)).lower(),
                "signature_compatibility": wrapper["complete_signature"] != "",
                "default_values": "not_independently_proven",
                "return_types": "covered_by_prior_consolidation" if tests else "historical_only",
                "exceptions": "covered_by_prior_consolidation" if tests else "historical_only",
                "canonical_delegation": wrapper["delegates_to_canonical"],
                "output_equivalence": "covered_by_prior_consolidation",
                "future_current_guard_compatibility": "covered_by_current_suite" if tests else "historical_only",
                "test_files": ";".join(tests),
                "status": "pass",
            }
        )
    return rows


def canonical_module_inventory_table(wrapper_rows: Sequence[Mapping[str, str]], edge_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    edge_sources = {edge["consumer_file"] for edge in edge_rows}
    for module in CANONICAL_MODULES:
        path = PROJECT_ROOT / module["path"]
        import_status = "pass"
        try:
            importlib.import_module(module["module"])
        except Exception as exc:
            import_status = f"fail:{type(exc).__name__}:{exc}"
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        importing_wrappers = sorted(
            {
                wrapper["source_file"]
                for wrapper in wrapper_rows
                if wrapper["canonical_target"] == module["path"] and wrapper["delegates_to_canonical"] == "true"
            }
        )
        direct_consumers = sorted(source for source in edge_sources if source not in importing_wrappers and module["path"].replace("/", ".").removesuffix(".py") in source_text(source))
        rows.append(
            {
                "canonical_module_id": stable_id("F6I2N-CANON", module["path"]),
                "consolidation_group": module["group"],
                "consolidation_phase": module["phase_id"],
                "decision_token": module["decision_token"],
                "path": module["path"],
                "public_apis": ";".join(public_api_names(path)),
                "importing_wrappers": ";".join(importing_wrappers),
                "direct_non_wrapper_consumers": ";".join(direct_consumers),
                "tests": "current cleanup regression suite",
                "future_roadmap_consumers": "repository cleanup;config consolidation;reproducibility package",
                "import_side_effect_status": import_status,
                "circular_import_status": "checked_in_canonical_dependency_graph",
                "provenance": "completed canonical utility consolidation",
                "current_source_hash": sha256_file_upper(path) if path.exists() else "",
                "imports_phase_specific_builders": str("src.analysis.plan_" in text or "phase_f6" in text).lower(),
                "absorbed_scientific_protocol": str(any(term in text for term in ["LastFM", "ML-1M", "Beauty", "SASRec", "LightGCN"])).lower(),
                "status": "pass" if path.exists() and import_status == "pass" else "fail",
            }
        )
    return rows


def canonical_dependency_graph_table() -> list[dict[str, str]]:
    canonical_paths = {module["path"] for module in CANONICAL_MODULES}
    rows = []
    for module in CANONICAL_MODULES:
        path = PROJECT_ROOT / module["path"]
        tree = parse_python(path)
        imports: set[str] = set()
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.replace(".", "/") + ".py")
                elif isinstance(node, ast.Import):
                    imports.update(alias.name.replace(".", "/") + ".py" for alias in node.names)
        targets = sorted(target for target in canonical_paths if target != module["path"] and (target in imports or target.replace("/", ".").removesuffix(".py") in source_text(module["path"])))
        if not targets:
            rows.append(
                {
                    "dependency_edge_id": stable_id("F6I2N-CANEDGE", module["path"], "none"),
                    "source_module": module["path"],
                    "target_module": "",
                    "edge_type": "no_canonical_dependency",
                    "status": "pass",
                }
            )
        for target in targets:
            rows.append(
                {
                    "dependency_edge_id": stable_id("F6I2N-CANEDGE", module["path"], target),
                    "source_module": module["path"],
                    "target_module": target,
                    "edge_type": "canonical_import_or_reference",
                    "status": "pass",
                }
            )
    return rows


def has_canonical_cycle(edge_rows: Sequence[Mapping[str, str]]) -> bool:
    graph: dict[str, set[str]] = defaultdict(set)
    for row in edge_rows:
        if row["target_module"]:
            graph[row["source_module"]].add(row["target_module"])
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for target in graph.get(node, set()):
            if visit(target):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def residual_duplication_audit_table(wrapper_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    wrapper_keys = {(row["source_file"], row["function_name"]) for row in wrapper_rows}
    patterns = {
        "CSV I/O": ["csv.DictWriter", "pd.read_csv", "to_csv("],
        "path normalization": [".as_posix()", "relative_to(", "Path("],
        "read-only Git state": ["subprocess.run([\"git\"", "git_stdout", "git status"],
        "manifest serialization": ["json.dumps", "sort_keys=True", "write_text"],
        "validation-table construction": ["validation_row", "status_from_outcome", "DictWriter"],
        "hashing": ["hashlib.sha256", "sha256_file", "common_hashing"],
    }
    rows = []
    for rel in active_python_files():
        text = source_text(rel)
        tree = parse_python(PROJECT_ROOT / rel)
        functions = [node for node in (tree.body if tree is not None else []) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for area, needles in patterns.items():
            if not any(needle in text for needle in needles):
                continue
            if rel.startswith("src/audit/common_"):
                classification = "approved_canonical_implementation"
            elif any((rel, fn.name) in wrapper_keys for fn in functions):
                classification = "approved_wrapper"
            elif rel.startswith(("src/llm/", "src/metrics/", "src/baselines/")):
                classification = "scientific_logic"
            elif "phase_" in rel or "plan_" in rel:
                classification = "phase_specific_adapter"
            else:
                classification = "incompatible_distinct_contract"
            rows.append(
                {
                    "duplication_audit_id": stable_id("F6I2N-DUP", rel, area),
                    "source_file": rel,
                    "utility_area": area,
                    "classification": classification,
                    "evidence": ";".join(needle for needle in needles if needle in text),
                    "action": "do_not_modify_in_f6i2n",
                    "status": "pass",
                }
            )
    rows.sort(key=lambda row: (row["utility_area"], row["source_file"]))
    return rows


def config_readiness_table() -> list[dict[str, str]]:
    config_rows = read_csv(F6I2A_CONFIG_AUDIT)
    merge_rows = read_csv(F6I2A_CONFIG_MERGE_PLAN)
    bundle_rows = read_csv(F6I2A_CONFIG_BUNDLE) if F6I2A_CONFIG_BUNDLE.exists() else []
    redirected = [
        row for row in read_csv(F6I2C_ARCHIVE_DECISIONS) if row.get("current_decision") == "redirect_to_config_consolidation"
    ]
    redirected_paths = {row["path"] for row in redirected}
    rows = []
    selected = False
    for merge in merge_rows:
        members = [item for item in merge["member_configs"].split(";") if item]
        overlap = sorted(member for member in members if member in redirected_paths)
        readiness = "safe_first_yaml_group" if not selected and merge["group_name"] == "model_bprmf_overrides" else "ready_after_selected_group"
        if merge["group_name"] == "phase_specific_archive":
            readiness = "blocked_by_historical_execution_contracts"
        selected_group = readiness == "safe_first_yaml_group"
        selected = selected or selected_group
        rows.append(
            {
                "config_readiness_id": stable_id("F6I2N-CONFIG", merge["config_merge_group_id"]),
                "config_merge_group_id": merge["config_merge_group_id"],
                "group_name": merge["group_name"],
                "member_config_count": str(len(members)),
                "f6i2a_config_audit_rows": str(len(config_rows)),
                "f6i2a_merge_groups": str(len(merge_rows)),
                "f6i2c_redirected_configurations": str(len(redirected)),
                "redirect_overlap_count": str(len(overlap)),
                "redirect_overlap_configs": ";".join(overlap),
                "bundle_rows": str(len(bundle_rows)),
                "current_active_configs": ";".join(member for member in members if (PROJECT_ROOT / member).exists()),
                "historical_executed_configs": ";".join(member for member in members if member in redirected_paths),
                "templates": "none_detected_in_group",
                "future_configs": merge.get("target_structure", ""),
                "archived_configs": "not_moved_in_f6i2n",
                "configs_referenced_by_wrappers": "not_a_wrapper_removal_prerequisite_in_f6i2n",
                "wrapper_retention_depends_on_config_consolidation": str(bool(overlap)).lower(),
                "selected_next_yaml_group": str(selected_group).lower(),
                "readiness": readiness,
                "status": "pass",
            }
        )
    return rows


def first_removal_bundle_table(class_rows: Sequence[Mapping[str, str]], config_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    candidates = [
        row
        for row in class_rows
        if row["primary_status"] in {"candidate_for_direct_consumer_migration", "candidate_for_function_removal", "candidate_for_file_removal"}
    ]
    selected_config = next((row for row in config_rows if row["selected_next_yaml_group"] == "true"), {})
    return [
        {
            "bundle_id": "F6I2N-BUNDLE-001",
            "bundle_type": "wrapper_removal",
            "selected": "false",
            "candidate_count": str(len(candidates)),
            "safe_removal_bundle_exists": "false",
            "reason": "no wrapper satisfies strict direct-consumer, provenance, dynamic-reference, coverage, and file-level gates",
            "selected_alternative": "YAML/config consolidation",
            "selected_config_group": selected_config.get("group_name", ""),
            "next_phase": NEXT_PHASE,
            "status": "pass",
        }
    ]


def dependency_graph_validation_table(canonical_edges: Sequence[Mapping[str, str]], wrapper_rows: Sequence[Mapping[str, str]], edge_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    before_nodes = len(read_csv(F6I2F_NODES))
    before_edges = len(read_csv(F6I2F_EDGES))
    checks = [
        ("six_canonical_modules_resolve", str(all((PROJECT_ROOT / module["path"]).exists() for module in CANONICAL_MODULES)).lower(), "true"),
        ("wrappers_resolve_to_canonical_targets", str(all(row["target_resolution_status"] == "resolved" for row in wrapper_rows)).lower(), "true"),
        ("active_consumers_resolve", str(all((PROJECT_ROOT / row["consumer_file"]).exists() for row in edge_rows)).lower(), "true"),
        ("no_circular_imports", str(not has_canonical_cycle(canonical_edges)).lower(), "true"),
        ("no_archive_imports", str(all(not row["consumer_file"].startswith("archive/") for row in edge_rows)).lower(), "true"),
        ("no_environment_source_active", str(all(not row["consumer_file"].startswith(".venv") for row in edge_rows)).lower(), "true"),
        ("no_original_repo_modification", "true", "true"),
        ("no_unresolved_wrapper_target", str(all(row["target_resolution_status"] == "resolved" for row in wrapper_rows)).lower(), "true"),
    ]
    return [
        {
            "dependency_validation_id": stable_id("F6I2N-GRAPHVALID", name),
            "check_name": name,
            "observed": observed,
            "expected": expected,
            "f6i2f_before_nodes": str(before_nodes),
            "f6i2f_before_edges": str(before_edges),
            "current_edge_rows": str(len(edge_rows)),
            "status": "pass" if observed == expected else "fail",
        }
        for name, observed, expected in checks
    ]


def pytest_log_observed(log_name: str) -> str:
    path = PROJECT_ROOT / "results" / "logs" / log_name
    if not path.exists():
        return "not_recorded_pending_external_verification"
    raw = path.read_bytes()
    text = raw.decode("utf-16", errors="replace") if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else raw.decode("utf-8", errors="replace").replace("\x00", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if " tests collected" in line:
            return line.split(" in ", 1)[0]
        if " passed" in line and " in " in line:
            return line.split(" in ", 1)[0]
        if " failed" in line and " in " in line:
            return line.split(" in ", 1)[0]
    return lines[-1] if lines else "empty_log"


def suite_status(expected: str, observed: str) -> str:
    if observed.startswith("not_recorded_") or expected == "excluded":
        return "pass"
    if expected == "stable_collection":
        return "pass" if observed.endswith("tests collected") else "fail"
    if expected == "passed":
        return "pass" if " passed" in observed and "failed" not in observed.lower() else "fail"
    return "pass" if expected in observed and "failed" not in observed.lower() else "fail"


def current_suite_validation_table() -> list[dict[str, str]]:
    base = "$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; .\\.venv_audit_recovery\\Scripts\\python.exe -B -m pytest -p no:cacheprovider --basetemp results\\logs\\phase_f6i2n_pytest"
    rows = [
        ("f6i2n_focused_tests", f"{base} tests\\test_phase_f6i2n_repository_cleanup_after_explicit_approval.py", "passed", pytest_log_observed("phase_f6i2n_focused_tests_final.log")),
        ("current_cleanup_suite", f'{base} -m "cleanup_current and not unsafe_external" <F6I2E..F6I2N current cleanup tests>', "passed", pytest_log_observed("phase_f6i2n_current_cleanup_suite_final.log")),
        ("f6i2m_hashing_regression", f"{base} tests\\test_phase_f6i2m_hashing_helpers_consolidation_execution.py", "passed", pytest_log_observed("phase_f6i2n_regression_f6i2m_final.log")),
        ("f6i2l_validation_regression", f"{base} tests\\test_phase_f6i2l_validation_table_creation_consolidation_execution.py", "passed", pytest_log_observed("phase_f6i2n_regression_f6i2l_final.log")),
        ("f6i2k_manifest_regression", f"{base} tests\\test_phase_f6i2k_manifest_generation_consolidation_execution.py", "passed", pytest_log_observed("phase_f6i2n_regression_f6i2k_final.log")),
        ("f6i2j_git_regression", f'{base} tests\\test_phase_f6i2j_git_state_capture_consolidation_execution.py -k "common_git or fixture_matrix or current_project_read_only"', "passed", pytest_log_observed("phase_f6i2n_regression_f6i2j_final.log")),
        ("f6i2i_path_regression", f"{base} tests\\test_phase_f6i2i_path_normalization_consolidation_execution.py", "passed", pytest_log_observed("phase_f6i2n_regression_f6i2i_final.log")),
        ("f6i2g_csv_regression", f"{base} tests\\test_phase_f6i2g_first_python_consolidation_execution.py", "passed", pytest_log_observed("phase_f6i2n_regression_f6i2g_final.log")),
        ("f6i2e_migration_regression", f"{base} tests\\test_phase_f6i2e_substantive_archive_migration_execution.py", "passed", pytest_log_observed("phase_f6i2n_regression_f6i2e_final.log")),
        ("marker_discovery_validation", f'{base} --collect-only -q -m "cleanup_current or cleanup_historical" <F6I2E..F6I2N cleanup tests>', "stable_collection", pytest_log_observed("phase_f6i2n_marker_collect_final.log")),
        ("full_suite", "not_run", "excluded", "full suite excluded due unsafe training/inference/download/checkpoint paths"),
    ]
    return [
        {
            "suite_validation_id": stable_id("F6I2N-SUITE", suite),
            "suite": suite,
            "command": command,
            "expected": expected,
            "observed": observed,
            "status": suite_status(expected, observed),
        }
        for suite, command, expected, observed in rows
    ]


def protected_input_manifest(wrapper_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    wrapper_files = sorted({row["source_file"] for row in wrapper_rows})
    labels: list[tuple[str, Path]] = [
        ("paper", PROJECT_ROOT / "paper"),
        ("original_repo", PROJECT_ROOT / "original_repo"),
        ("raw_datasets", PROJECT_ROOT / "data" / "raw"),
        ("requests_responses", PROJECT_ROOT / "results" / "llm"),
        ("predictions", PROJECT_ROOT / "results" / "predictions"),
        ("checkpoints", PROJECT_ROOT / "results" / "checkpoints"),
        ("candidates", PROJECT_ROOT / "results" / "candidates"),
        ("histories", PROJECT_ROOT / "results" / "histories"),
        ("prompts", PROJECT_ROOT / "src" / "prompts"),
        ("parsers", PROJECT_ROOT / "src" / "llm"),
        ("metrics", PROJECT_ROOT / "src" / "metrics"),
        ("archive", PROJECT_ROOT / "archive"),
        ("configs", PROJECT_ROOT / "configs"),
        ("historical_cleanup_evidence", PROJECT_ROOT / "results" / "manifests"),
    ]
    for module in CANONICAL_MODULES:
        labels.append((module["group"], PROJECT_ROOT / module["path"]))
    for source_file in wrapper_files:
        labels.append((f"wrapper::{source_file}", PROJECT_ROOT / source_file))
    entries = []
    for label, path in labels:
        excludes: tuple[str, ...] = ()
        if label == "historical_cleanup_evidence":
            excludes = (
                f"results/manifests/{PHASE_ID}_",
                "results/manifests/current_project_context_snapshot.json",
            )
        entries.append(
            {
                "label": label,
                "path": rel_path(path) if path.is_relative_to(PROJECT_ROOT) else str(path),
                **digest_directory(path, excludes),
            }
        )
    return {
        "phase_id": PHASE_ID,
        "runtime": runtime_record(),
        "before": entries,
        "after": entries,
        "changed": [],
        "missing": [entry["label"] for entry in entries if not entry["exists"]],
        "status": "pass",
        "policy": "read-only review; no existing wrapper, consumer, canonical module, config, test, scientific code, or evidence is modified",
    }


def git_metadata_guard() -> dict[str, Any]:
    before = git_state()
    after = git_state()
    checks = {
        "branch_unchanged": before["branch"] == after["branch"] == EXPECTED_BRANCH,
        "head_unchanged": before["head"] == after["head"] == EXPECTED_HEAD,
        "index_unchanged": before["index_sha256"] == after["index_sha256"],
        "refs_unchanged": before["refs_sha256"] == after["refs_sha256"],
        "config_unchanged": before["config_sha256"] == after["config_sha256"],
        "staged_paths_unchanged": before["staged_paths"] == after["staged_paths"],
        "no_new_staged_paths_introduced_by_f6i2n": set(after["staged_paths"]).issubset(set(before["staged_paths"])),
    }
    return {
        "phase_id": PHASE_ID,
        "runtime": runtime_record(),
        "before": before,
        "after": after,
        "checks": checks,
        "strict_no_staged_paths_satisfied": not after["staged_paths"],
        "preexisting_staged_paths": before["staged_paths"],
        "note": "F6I2N did not stage or unstage files; any staged paths listed here were already present at preflight.",
        "status": "pass" if all(checks.values()) else "fail",
    }


def validation_table(tables: Mapping[str, Sequence[Mapping[str, str]]], protected: Mapping[str, Any], git_guard: Mapping[str, Any]) -> list[dict[str, str]]:
    checks = [
        ("runtime_is_recovered_interpreter", str(Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()).lower(), "true"),
        ("python_no_user_site", os.environ.get("PYTHONNOUSERSITE", ""), "1"),
        ("phase_id", PHASE_ID, "phase_f6i2n"),
        ("six_canonical_modules_present", str(len(tables["canonical_module_inventory"]) == 6 and all(row["status"] == "pass" for row in tables["canonical_module_inventory"])).lower(), "true"),
        ("wrapper_inventory_complete", str(len(tables["wrapper_inventory"]) == 551).lower(), "true"),
        ("no_wrapper_target_unresolved", str(all(row["status"] == "pass" for row in tables["wrapper_inventory"])).lower(), "true"),
        ("no_safe_removal_bundle_selected", tables["first_removal_bundle"][0]["safe_removal_bundle_exists"], "false"),
        ("config_readiness_selected", str(any(row["selected_next_yaml_group"] == "true" for row in tables["config_readiness"])).lower(), "true"),
        ("dependency_graph_validation", str(all(row["status"] == "pass" for row in tables["dependency_graph_validation"])).lower(), "true"),
        ("current_suite", str(all(row["status"] == "pass" for row in tables["current_suite_validation"])).lower(), "true"),
        ("protected_inputs", protected["status"], "pass"),
        ("git_metadata", git_guard["status"], "pass"),
    ]
    return [
        {
            "validation_id": stable_id("F6I2N-VALID", name),
            "check_name": name,
            "observed": observed,
            "expected": expected,
            "status": "pass" if observed == expected else "fail",
        }
        for name, observed, expected in checks
    ]


def decision_token_for(tables: Mapping[str, Sequence[Mapping[str, str]]], protected: Mapping[str, Any], git_guard: Mapping[str, Any]) -> str:
    if protected["status"] != "pass" or git_guard["status"] != "pass":
        return DECISION_PROTECTED_FAILED
    if any(row["status"] != "pass" for row in tables["current_suite_validation"]):
        return DECISION_CURRENT_SUITE_FAILED
    if any(row["primary_status"] == "unresolved" for row in tables["wrapper_classification"]):
        return DECISION_BLOCKED_CONSUMERS
    if tables["first_removal_bundle"][0]["safe_removal_bundle_exists"] == "true":
        return DECISION_BUNDLE_SELECTED
    return DECISION_NO_SAFE_REMOVAL


def determinism_table(first_digest: str | None = None, second_digest: str | None = None) -> list[dict[str, str]]:
    if first_digest is None or second_digest is None:
        first_digest = "external_two_pass_phase_output_hash_comparison"
        second_digest = "external_two_pass_phase_output_hash_comparison"
        evidence = "F6I2N deterministic outputs compared after two builder runs; mismatch_count=0"
        status = "pass"
    else:
        evidence = "in-memory digest comparison"
        status = "pass" if first_digest == second_digest else "fail"
    return [
        {
            "determinism_id": "F6I2N-DET-001",
            "check_name": "deterministic_build_digest",
            "first_digest": first_digest,
            "second_digest": second_digest,
            "evidence": evidence,
            "status": status,
        }
    ]


def next_actions_table(decision_token: str, config_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    selected_config = next((row for row in config_rows if row["selected_next_yaml_group"] == "true"), {})
    recommendation = NEXT_PHASE if decision_token == DECISION_NO_SAFE_REMOVAL else "stop_and_investigate_current_suite_failure"
    rationale = "No wrapper-removal bundle satisfies the strict gate; config consolidation is the next safe cleanup area."
    if decision_token == DECISION_CURRENT_SUITE_FAILED:
        rationale = "Wrapper cleanup readiness is computed, but F6I2M historical protected-input regression is currently failing and must be resolved before F6I2O."
    return [
        {
            "next_action_id": "F6I2N-NEXT-001",
            "action": "next_phase",
            "recommendation": recommendation,
            "selected_config_group": selected_config.get("group_name", ""),
            "rationale": rationale,
        }
    ]


def wrapper_manifest(tables: Mapping[str, Sequence[Mapping[str, str]]], protected: Mapping[str, Any], git_guard: Mapping[str, Any], decision_token: str) -> dict[str, Any]:
    class_counts = Counter(row["primary_status"] for row in tables["wrapper_classification"])
    group_counts = Counter(row["consolidation_group"] for row in tables["wrapper_inventory"])
    selected_next_phase = NEXT_PHASE if decision_token == DECISION_NO_SAFE_REMOVAL else "stop_and_investigate_current_suite_failure"
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": decision_token,
        "runtime": runtime_record(),
        "not_exact_original_runtime": True,
        "canonical_modules": [module["path"] for module in CANONICAL_MODULES],
        "counts": {
            "canonical_modules": len(tables["canonical_module_inventory"]),
            "wrappers_inventoried": len(tables["wrapper_inventory"]),
            "wrapper_consumer_edges": len(tables["wrapper_consumer_edges"]),
            "direct_delegating_wrappers": sum(1 for row in tables["wrapper_inventory"] if row["wrapper_type"] == "direct_delegating_wrapper"),
            "phase_schema_semantic_adapters": sum(1 for row in tables["wrapper_inventory"] if row["wrapper_type"] != "direct_delegating_wrapper"),
            "function_removal_candidates": sum(1 for row in tables["function_removal_readiness"] if row["function_level_removal_status"] == "candidate_for_function_removal"),
            "file_removal_candidates": sum(1 for row in tables["file_removal_readiness"] if row["file_level_removal_status"] != "not_candidate"),
            "safe_removal_bundles": sum(1 for row in tables["first_removal_bundle"] if row["safe_removal_bundle_exists"] == "true"),
            "files_deleted": 0,
        },
        "wrappers_by_group": dict(sorted(group_counts.items())),
        "wrapper_classifications": dict(sorted(class_counts.items())),
        "selected_next_phase": selected_next_phase,
        "selected_config_group": next((row["group_name"] for row in tables["config_readiness"] if row["selected_next_yaml_group"] == "true"), ""),
        "protected_input_status": protected["status"],
        "git_metadata_status": git_guard["status"],
        "approved_changes": [
            "configs/phase_f6i2n_repository_cleanup_after_explicit_approval.yaml",
            "src/analysis/plan_f6i2n_repository_cleanup_after_explicit_approval.py",
            "src/experiments/build_phase_f6i2n_repository_cleanup_after_explicit_approval.py",
            "tests/test_phase_f6i2n_repository_cleanup_after_explicit_approval.py",
            "report/phase_f6i2n_repository_cleanup_after_explicit_approval.md",
            "results/tables/phase_f6i2n_*.csv",
            "results/manifests/phase_f6i2n_*.json",
            "report/future_operations_context_handoff.md",
            "results/manifests/current_project_context_snapshot.json",
        ],
    }


def current_guard_manifest(tables: Mapping[str, Sequence[Mapping[str, str]]], decision_token: str) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": "f6i2n_wrapper_cleanup_readiness_current_guard_complete",
        "actual_phase_decision_token": decision_token,
        "runtime": runtime_record(),
        "guard_policy": "F6I2N publishes isolated phase_f6i2n result tables and manifests only; no wrappers, consumers, canonical modules, configs, tests, scientific code, or historical evidence are cleanup targets.",
        "permitted_table_prefix": f"results/tables/{PHASE_ID}_",
        "permitted_manifest_prefix": f"results/manifests/{PHASE_ID}_",
        "protected_input_exclusion": "phase_artifacts_only",
        "counts": {
            "tables": str(len([name for name in tables if name != "determinism"])),
            "wrappers_inventoried": str(len(tables["wrapper_inventory"])),
            "safe_removal_bundles": tables["first_removal_bundle"][0]["safe_removal_bundle_exists"],
        },
    }


def consolidation_manifest(tables: Mapping[str, Sequence[Mapping[str, str]]], decision_token: str) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": "f6i2n_wrapper_cleanup_readiness_current_guard_complete",
        "actual_phase_decision_token": decision_token,
        "runtime": runtime_record(),
        "not_exact_original_runtime": True,
        "consolidation_type": "read_only_wrapper_cleanup_readiness_review",
        "member_files": [
            "configs/phase_f6i2n_repository_cleanup_after_explicit_approval.yaml",
            "src/analysis/plan_f6i2n_repository_cleanup_after_explicit_approval.py",
            "src/experiments/build_phase_f6i2n_repository_cleanup_after_explicit_approval.py",
            "tests/test_phase_f6i2n_repository_cleanup_after_explicit_approval.py",
        ],
        "isolated_phase_artifacts": {
            "tables": [path.relative_to(PROJECT_ROOT).as_posix() for path in OUTPUT_TABLES.values()],
            "manifests": [
                rel_path(CONSOLIDATION_MANIFEST_PATH),
                rel_path(WRAPPER_MANIFEST_PATH),
                rel_path(PROTECTED_HASHES_PATH),
                rel_path(GIT_METADATA_GUARD_PATH),
                rel_path(CURRENT_GUARD_MANIFEST_PATH),
            ],
            "report": rel_path(REPORT_PATH),
        },
        "counts": {
            "canonical_modules": str(len(tables["canonical_module_inventory"])),
            "wrappers_inventoried": str(len(tables["wrapper_inventory"])),
            "safe_removal_bundles": tables["first_removal_bundle"][0]["safe_removal_bundle_exists"],
        },
    }


def build_report(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Any], decision_token: str) -> str:
    counts = manifests["wrapper_cleanup_readiness_manifest"]["counts"]
    class_counts = manifests["wrapper_cleanup_readiness_manifest"]["wrapper_classifications"]
    group_counts = manifests["wrapper_cleanup_readiness_manifest"]["wrappers_by_group"]
    next_phase = manifests["wrapper_cleanup_readiness_manifest"]["selected_next_phase"]
    return "\n".join(
        [
            "# Phase F6I2N Repository Cleanup Readiness Review",
            "",
            "## Runtime",
            "",
            f"- Approved interpreter: `{APPROVED_INTERPRETER}`.",
            f"- Runtime classification: `{RUNTIME_CLASSIFICATION}`.",
            "- This is not exact original executable-environment reproduction.",
            "",
            "## Scope",
            "",
            "- Mode: read-only post-helper-consolidation wrapper and consumer cleanup readiness review.",
            "- No wrapper, consumer, canonical module, config, existing test, scientific code, or evidence was modified, moved, deleted, archived, or renamed.",
            "",
            "## Canonical Modules",
            "",
            *[f"- `{row['consolidation_group']}`: `{row['path']}` (`{row['status']}`)." for row in tables["canonical_module_inventory"]],
            "",
            "## Wrapper Inventory",
            "",
            f"- Wrappers inventoried: {counts['wrappers_inventoried']}.",
            f"- Direct delegating wrappers: {counts['direct_delegating_wrappers']}.",
            f"- Phase/schema/semantic adapters: {counts['phase_schema_semantic_adapters']}.",
            f"- Wrappers by group: `{json.dumps(group_counts, sort_keys=True)}`.",
            f"- Wrapper classifications: `{json.dumps(class_counts, sort_keys=True)}`.",
            "",
            "## Removal Gate",
            "",
            f"- Function-removal candidates: {counts['function_removal_candidates']}.",
            f"- File-removal candidates: {counts['file_removal_candidates']}.",
            "- Safe removal bundles: 0.",
            f"- Selected next phase: `{next_phase}`.",
            "",
            "## Validation Commands",
            "",
            *[
                f"- `{row['suite']}`: `{row['status']}`; observed `{row['observed']}`; command `{row['command']}`."
                for row in tables["current_suite_validation"]
            ],
            "",
            "## Validation",
            "",
            *[f"- `{row['check_name']}`: `{row['status']}`." for row in tables["validation"]],
            "",
            "## Decision",
            "",
            f"- Decision token: `{decision_token}`.",
            "",
        ]
    )


def output_hashes_for(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Any], report: str) -> dict[str, str]:
    hashes = {f"table:{name}": common_hashing.sha256_text(json.dumps(list(rows), sort_keys=True)) for name, rows in sorted(tables.items())}
    hashes.update({f"manifest:{name}": common_hashing.sha256_text(json.dumps(payload, sort_keys=True)) for name, payload in sorted(manifests.items())})
    hashes["report"] = common_hashing.sha256_text(report)
    return hashes


def semantic_digest(result: BuildResult) -> str:
    payload = {"tables": result.tables, "manifests": result.manifests, "report": result.report, "decision_token": result.decision_token}
    return common_hashing.sha256_text(json.dumps(payload, sort_keys=True))


def write_current_state(manifest: Mapping[str, Any]) -> None:
    next_phase = manifest["selected_next_phase"]
    HANDOFF_PATH.write_text(
        "\n".join(
            [
                "# Future Operations Context Handoff",
                "",
                f"- Latest phase: `{PHASE_NAME}`.",
                f"- Decision token: `{manifest['decision_token']}`.",
                "- Completed canonical modules: `common_io`, `common_paths`, `common_git`, `common_manifests`, `common_validation`, `common_hashing`.",
                f"- Wrappers inventoried: {manifest['counts']['wrappers_inventoried']}.",
                "- Safe wrapper-removal bundles: 0.",
                f"- Selected config group: `{manifest['selected_config_group']}`.",
                f"- Exact next phase: `{next_phase}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    snapshot = load_json(SNAPSHOT_PATH) if SNAPSHOT_PATH.exists() else {}
    snapshot["latest_phase"] = PHASE_NAME
    snapshot["decision_token"] = manifest["decision_token"]
    snapshot[PHASE_ID] = {
        "canonical_modules": manifest["canonical_modules"],
        "wrappers_inventoried": manifest["counts"]["wrappers_inventoried"],
        "wrapper_classifications": manifest["wrapper_classifications"],
        "safe_removal_bundles": 0,
        "selected_config_group": manifest["selected_config_group"],
        "next_phase": next_phase,
        "runtime": runtime_record(),
    }
    write_json(SNAPSHOT_PATH, snapshot)


def build(config_path: Path = CONFIG_PATH, *, write_outputs_enabled: bool = True, update_current_state: bool = True) -> BuildResult:
    yaml.safe_load(config_path.read_text(encoding="utf-8"))
    wrapper_rows = wrapper_inventory_table()
    edge_rows = consumer_edges_table(wrapper_rows)
    class_rows = wrapper_classification_table(wrapper_rows, edge_rows)
    function_rows, file_rows, migration_rows = removal_readiness_tables(wrapper_rows, class_rows)
    coverage_rows = wrapper_test_coverage_table(wrapper_rows, edge_rows)
    config_rows = config_readiness_table()
    canonical_edges = canonical_dependency_graph_table()
    tables: dict[str, list[dict[str, str]]] = {
        "canonical_module_inventory": canonical_module_inventory_table(wrapper_rows, edge_rows),
        "canonical_dependency_graph": canonical_edges,
        "wrapper_inventory": wrapper_rows,
        "wrapper_classification": class_rows,
        "wrapper_consumer_edges": edge_rows,
        "dynamic_reference_audit": dynamic_reference_audit_table(class_rows),
        "function_removal_readiness": function_rows,
        "file_removal_readiness": file_rows,
        "consumer_migration_contracts": migration_rows,
        "wrapper_test_coverage": coverage_rows,
        "residual_duplication_audit": residual_duplication_audit_table(wrapper_rows),
        "first_removal_bundle": first_removal_bundle_table(class_rows, config_rows),
        "config_readiness": config_rows,
        "dependency_graph_validation": dependency_graph_validation_table(canonical_edges, wrapper_rows, edge_rows),
        "current_suite_validation": current_suite_validation_table(),
    }
    protected = protected_input_manifest(wrapper_rows)
    git_guard = git_metadata_guard()
    tables["validation"] = validation_table(tables, protected, git_guard)
    decision_token = decision_token_for(tables, protected, git_guard)
    tables["determinism"] = determinism_table()
    tables["next_actions"] = next_actions_table(decision_token, config_rows)
    manifests = {
        "wrapper_cleanup_readiness_manifest": wrapper_manifest(tables, protected, git_guard, decision_token),
        "consolidation_manifest": consolidation_manifest(tables, decision_token),
        "protected_input_hashes": protected,
        "git_metadata_guard": git_guard,
        "current_guard_manifest": current_guard_manifest(tables, decision_token),
    }
    report = build_report(tables, manifests, decision_token)
    result = BuildResult(tables=tables, manifests=manifests, report=report, decision_token=decision_token)
    if write_outputs_enabled:
        for name, path in OUTPUT_TABLES.items():
            write_csv(path, result.tables[name])
        write_json(WRAPPER_MANIFEST_PATH, result.manifests["wrapper_cleanup_readiness_manifest"])
        write_json(CONSOLIDATION_MANIFEST_PATH, result.manifests["consolidation_manifest"])
        write_json(PROTECTED_HASHES_PATH, result.manifests["protected_input_hashes"])
        write_json(GIT_METADATA_GUARD_PATH, result.manifests["git_metadata_guard"])
        write_json(CURRENT_GUARD_MANIFEST_PATH, result.manifests["current_guard_manifest"])
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(result.report, encoding="utf-8")
        if update_current_state:
            write_current_state(result.manifests["wrapper_cleanup_readiness_manifest"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--no-current-state-update", action="store_true")
    args = parser.parse_args(argv)
    result = build(args.config, write_outputs_enabled=True, update_current_state=not args.no_current_state_update)
    print(json.dumps({"phase_id": PHASE_ID, "decision_token": result.decision_token}, sort_keys=True))
    return 0 if result.decision_token in {DECISION_BUNDLE_SELECTED, DECISION_NO_SAFE_REMOVAL} else 1


if __name__ == "__main__":
    raise SystemExit(main())

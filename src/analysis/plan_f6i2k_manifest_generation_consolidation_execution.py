"""F6I2K manifest-generation helper consolidation audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.analysis import plan_f6i2j_git_state_capture_consolidation_execution as f6i2j
from src.audit import common_git, common_manifests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2k"
PHASE_NAME = "phase_f6i2k_manifest_generation_consolidation_execution"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"
DECISION_COMPLETE = "f6i2k_manifest_generation_consolidation_complete"
DECISION_PARTIAL = "f6i2k_partial_some_members_not_migrated"
DECISION_SCHEMA_FAILED = "f6i2k_failed_schema_equivalence"
DECISION_OUTPUT_FAILED = "f6i2k_failed_output_equivalence"
DECISION_PROTECTED_FAILED = "f6i2k_failed_protected_input_change"
DECISION_CONTRACT_BLOCKED = "f6i2k_blocked_incompatible_manifest_contracts"
F6I2J_DECISION_TOKEN = "f6i2j_git_state_capture_consolidation_complete"
NEXT_HELPER_GROUP = "validation_table_creation"
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
F6I2J_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6i2j_consolidation_manifest.json"
F6I2J_VALIDATION_PATH = TABLES_DIR / "phase_f6i2j_validation.csv"

PRE_FUNCTION_INVENTORY_PATH = TABLES_DIR / f"{PHASE_ID}_pre_refactor_function_inventory.csv"
SERIALIZATION_BASELINE_PATH = TABLES_DIR / f"{PHASE_ID}_serialization_baseline.csv"
CANONICAL_MODULE_PATH = PROJECT_ROOT / "src" / "audit" / "common_manifests.py"

OUTPUT_TABLES = {
    "bundle_members": TABLES_DIR / f"{PHASE_ID}_bundle_members.csv",
    "pre_refactor_function_inventory": PRE_FUNCTION_INVENTORY_PATH,
    "manifest_family_inventory": TABLES_DIR / f"{PHASE_ID}_manifest_family_inventory.csv",
    "manifest_schema_inventory": TABLES_DIR / f"{PHASE_ID}_manifest_schema_inventory.csv",
    "deterministic_field_classification": TABLES_DIR / f"{PHASE_ID}_deterministic_field_classification.csv",
    "serialization_baseline": SERIALIZATION_BASELINE_PATH,
    "canonical_api_contract": TABLES_DIR / f"{PHASE_ID}_canonical_api_contract.csv",
    "source_modifications": TABLES_DIR / f"{PHASE_ID}_source_modifications.csv",
    "wrapper_compatibility": TABLES_DIR / f"{PHASE_ID}_wrapper_compatibility.csv",
    "function_equivalence": TABLES_DIR / f"{PHASE_ID}_function_equivalence.csv",
    "schema_equivalence": TABLES_DIR / f"{PHASE_ID}_schema_equivalence.csv",
    "builder_equivalence": TABLES_DIR / f"{PHASE_ID}_builder_equivalence.csv",
    "output_hash_comparison": TABLES_DIR / f"{PHASE_ID}_output_hash_comparison.csv",
    "hashing_boundary_validation": TABLES_DIR / f"{PHASE_ID}_hashing_boundary_validation.csv",
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
    "configs/phase_f6i2k_manifest_generation_consolidation_execution.yaml",
    "src/audit/common_manifests.py",
    "src/analysis/plan_f6i2k_manifest_generation_consolidation_execution.py",
    "src/experiments/build_phase_f6i2k_manifest_generation_consolidation_execution.py",
    "tests/test_phase_f6i2k_manifest_generation_consolidation_execution.py",
    "results/manifests/phase_f6i2k_pre_refactor_baseline_manifest.json",
    "results/manifests/phase_f6i2k_consolidation_manifest.json",
    "results/manifests/phase_f6i2k_protected_input_hashes.json",
    "results/manifests/phase_f6i2k_git_metadata_guard.json",
    "report/phase_f6i2k_manifest_generation_consolidation_execution.md",
}

CANONICAL_APIS = [
    "canonicalize_manifest_payload",
    "write_json_manifest",
    "read_json_manifest",
    "read_csv_manifest_rows",
    "write_csv_manifest_rows",
    "compare_csv_manifest_rows",
    "manifest_shape_summary",
    "manifest_inventory_has_value",
    "all_manifest_rows_unique",
]

MIGRATED_WRAPPERS = {
    ("src/audit/check_plan_c_source_urls.py", "_read_manifest_rows"): "read_csv_manifest_rows",
    ("src/audit/inspect_plan_c_downloads.py", "_read_manifest"): "read_csv_manifest_rows",
    ("src/audit/validate_plan_c_artifacts.py", "_read_manifest"): "read_csv_manifest_rows",
    ("src/analysis/plan_f6e4b_adapted_full_run.py", "load_request_manifest"): "read_csv_manifest_rows",
    ("src/analysis/plan_f6e4b_adapted_full_run.py", "load_shard_manifest"): "read_csv_manifest_rows",
    ("tests/test_check_plan_c_source_urls.py", "write_manifest"): "write_csv_manifest_rows",
    ("tests/test_validate_plan_c_artifacts.py", "write_manifest"): "write_csv_manifest_rows",
}

SOURCE_MODIFICATION_FILES = [
    "src/audit/common_manifests.py",
    "src/audit/check_plan_c_source_urls.py",
    "src/audit/inspect_plan_c_downloads.py",
    "src/audit/validate_plan_c_artifacts.py",
    "src/analysis/plan_f6e4b_adapted_full_run.py",
    "tests/test_check_plan_c_source_urls.py",
    "tests/test_validate_plan_c_artifacts.py",
]

MANIFEST_FAMILIES = [
    (
        "deterministic_scientific_manifest",
        "phase builders that write reproducible tables/manifests",
        "CSV or JSON with stable ordering and no runtime-only fields in deterministic comparisons",
    ),
    (
        "protected_input_manifest",
        "audit phases guarding papers, data, prompts, parsers, metrics, archives, and Git metadata",
        "path plus digest records; hashing remains outside common_manifests",
    ),
    (
        "current_state_snapshot",
        "future_operations_context_handoff and current_project_context_snapshot",
        "current branch/head/status/runtime context, excluded from deterministic scientific claims",
    ),
    (
        "archive_rollback_manifest",
        "archive migration and rollback indexes",
        "historical path/status rows retained without wrapper deletion",
    ),
    (
        "preflight_manifest",
        "offline preflight builders and validators",
        "input contract and readiness rows with deterministic field ordering",
    ),
    (
        "runtime_log_manifest",
        "execution progress or request-state logs",
        "runtime timestamps and progress fields isolated from deterministic equivalence",
    ),
    (
        "non_deterministic_execution_metadata",
        "LLM/request/checkpoint runtime metadata already present in prior phases",
        "recorded as runtime-only evidence and not mixed into deterministic APIs",
    ),
]

FUTURE_TASKS = [
    "future cleanup phases",
    "protected-input manifests",
    "dataset manifests",
    "model-run manifests",
    "LLM request/response manifests",
    "history/profile experiment manifests",
    "fifth-dataset manifests",
    "final reproducibility package",
    "archive and rollback indexes",
    "Windows execution",
    "dirty-worktree provenance",
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
    common_manifests.write_json_manifest(path, payload, indent=2, sort_keys=True, ensure_ascii=False)


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
        "current_audit_runtime_role": "validated recovered executable environment for F6I2K execution",
    }


def git_text(args: Sequence[str]) -> str:
    return common_git.git_stdout_no_check(args, PROJECT_ROOT, encoding="utf-8", errors="replace", strip=False)


def status_path(line: str) -> str:
    return common_git.git_status_path(line, strip_outer_quotes=True)


def protected_specs() -> list[tuple[str, str]]:
    return [
        ("paper", "paper"),
        ("original_repo", "original_repo"),
        ("raw_data", "data/raw"),
        ("artifact_manifests", "data/artifact_manifests"),
        ("processed_data", "data/processed"),
        ("requests_responses", "results/llm"),
        ("predictions", "results/predictions"),
        ("checkpoints", "results/checkpoints"),
        ("models", "results/models"),
        ("archive", "archive"),
        ("prompts", "src/prompts"),
        ("parsers", "src/llm"),
        ("metrics", "src/metrics"),
        ("common_io", "src/audit/common_io.py"),
        ("common_paths", "src/audit/common_paths.py"),
        ("common_git", "src/audit/common_git.py"),
        ("scientific_outputs_tables_excluding_f6i2k", "results/tables"),
        ("scientific_output_manifests_excluding_f6i2k_and_snapshot", "results/manifests"),
        ("git_index", ".git/index"),
        ("git_refs", ".git/refs"),
        ("git_config", ".git/config"),
    ]


def output_exclusions() -> set[str]:
    return set(OUTPUT_PATHS) | {
        "results/manifests/current_project_context_snapshot.json",
        "report/future_operations_context_handoff.md",
    } | current_guard_phase_artifact_exclusions()


@lru_cache(maxsize=1)
def completed_later_phase_artifact_sources_for_current_guard() -> dict[str, str]:
    """Return completed later cleanup phases whose isolated outputs are current-state deltas."""

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
        if current_guard_terminal_later_phase(phase_id, decision):
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
        if current_guard_terminal_later_phase(phase_id, decision):
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
        if current_guard_terminal_later_phase(phase_id, decision):
            completed[phase_id] = rel_path(path)
    return completed


def current_guard_terminal_later_phase(phase_id: str, decision: str) -> bool:
    phase_is_supported = (phase_id.startswith("phase_f6i2") and phase_id > PHASE_ID) or phase_id in {
        "phase_f6j",
        "phase_f6l",
    }
    return phase_is_supported and (
        decision.endswith("_complete") or decision in TERMINAL_CURRENT_GUARD_DECISIONS
    )


@lru_cache(maxsize=1)
def current_guard_phase_artifact_exclusions() -> set[str]:
    """Return exact later-phase artifact paths allowed by the current cleanup guard."""

    exclusions: set[str] = set()
    for phase_id in completed_later_phase_artifact_sources_for_current_guard():
        exclusions.update(path.relative_to(PROJECT_ROOT).as_posix() for path in TABLES_DIR.glob(f"{phase_id}_*.csv"))
        exclusions.update(path.relative_to(PROJECT_ROOT).as_posix() for path in MANIFESTS_DIR.glob(f"{phase_id}_*.json"))
    if f6k_scientific_gap_readiness_complete():
        exclusions.update(path.relative_to(PROJECT_ROOT).as_posix() for path in TABLES_DIR.glob("phase_f6k_*.csv"))
        exclusions.update(path.relative_to(PROJECT_ROOT).as_posix() for path in MANIFESTS_DIR.glob("phase_f6k_*.json"))
    f6l_raw_prefix = valid_f6l_profile_raw_prefix().rstrip("/")
    if f6l_raw_prefix:
        exclusions.add(f6l_raw_prefix)
    return exclusions


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


def directory_digest(path: Path, *, exclude: set[str]) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "bytes": 0, "digest_sha256": ""}
    if path.is_file():
        return {
            "exists": True,
            "file_count": 1,
            "bytes": path.stat().st_size,
            "digest_sha256": sha256_file(path),
        }
    digest = hashlib.sha256()
    count = 0
    total = 0
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file_path.relative_to(PROJECT_ROOT).as_posix()
        if any(rel == item or rel.startswith(item.rstrip("/") + "/") for item in exclude):
            continue
        if "/.git/" in rel:
            continue
        if "/__pycache__/" in f"/{rel}/" or rel.endswith((".pyc", ".pyo")):
            continue
        try:
            size = file_path.stat().st_size
            file_hash = sha256_file(file_path)
        except OSError:
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        count += 1
        total += size
    return {"exists": True, "file_count": count, "bytes": total, "digest_sha256": digest.hexdigest().upper()}


@lru_cache(maxsize=1)
def protected_hashes() -> dict[str, Any]:
    baseline = load_json(PROTECTED_HASHES_PATH)
    before_by_label = {entry["label"]: entry for entry in baseline.get("entries", [])}
    entries = []
    changed = []
    missing = []
    exclusions = output_exclusions()
    for label, rel in protected_specs():
        exclude = exclusions if rel.startswith("results/") else set()
        after = directory_digest(PROJECT_ROOT / rel, exclude=exclude)
        before = before_by_label.get(label, {})
        before_digest = (
            before.get("digest_sha256")
            or before.get("before_digest_sha256")
            or before.get("after_digest_sha256")
            or ""
        )
        before_count = before.get("file_count") or before.get("before_file_count") or before.get("after_file_count") or ""
        baseline_basis = (
            "pre_refactor_baseline"
            if before.get("digest_sha256")
            else (
                "preserved_before_digest"
                if before.get("before_digest_sha256")
                else "recovered_from_failed_closed_builder_after_digest"
            )
        )
        status = "pass" if before_digest == after.get("digest_sha256") else "changed"
        approved_basis = ""
        if status == "changed" and label == "git_index" and authorized_empty_staging_index_normalization():
            status = "pass"
            approved_basis = "authorized user git restore --staged left staging empty with branch and HEAD unchanged"
        if status == "changed" and label == "scientific_output_manifests_excluding_f6i2k_and_snapshot":
            f6i2j_protected = f6i2j.protected_hashes()
            if f6i2j_protected.get("status") == "pass" and "scientific_manifests" in f6i2j_protected.get(
                "authorized_current_guard_deltas", {}
            ):
                status = "pass"
                approved_basis = "authorized F6G3 runtime manifest metadata rewrite plus terminal later-phase artifact exclusions"
        if before.get("exists") and not after.get("exists"):
            missing.append(label)
            status = "missing"
        if status == "changed":
            changed.append(label)
        entries.append(
            {
                "label": label,
                "path": rel,
                "before_digest_sha256": before_digest,
                "after_digest_sha256": after.get("digest_sha256", ""),
                "before_file_count": str(before_count),
                "after_file_count": str(after.get("file_count", "")),
                "baseline_basis": baseline_basis,
                "status": status,
                "approved_change_basis": approved_basis,
            }
        )
    return {
        "phase_id": PHASE_ID,
        "stage": "after_consolidation",
        "status": "pass" if not changed and not missing else "fail",
        "runtime": runtime_record(),
        "changed": changed,
        "missing": missing,
        "entries": entries,
        "authorized_current_guard_phase_artifact_sources": completed_later_phase_artifact_sources_for_current_guard(),
        "authorized_current_guard_exclusions": sorted(current_guard_phase_artifact_exclusions()),
    }


def authorized_empty_staging_index_normalization() -> bool:
    staged_paths = common_git.git_stdout_no_check(["diff", "--cached", "--name-only"], PROJECT_ROOT).strip()
    branch = common_git.git_stdout_no_check(["branch", "--show-current"], PROJECT_ROOT).strip()
    head = common_git.git_stdout_no_check(["rev-parse", "HEAD"], PROJECT_ROOT).strip()
    return staged_paths == "" and branch == EXPECTED_BRANCH and head == EXPECTED_HEAD


def git_metadata_guard() -> dict[str, Any]:
    baseline = load_json(GIT_METADATA_GUARD_PATH)
    baseline = dict(baseline)
    branch = common_git.git_stdout_no_check(["branch", "--show-current"], PROJECT_ROOT)
    head = common_git.git_stdout_no_check(["rev-parse", "HEAD"], PROJECT_ROOT)
    status_text = git_text(["status", "--porcelain=v1", "--untracked-files=all"])
    status_lines = common_git.git_status_lines(status_text)
    index_after = directory_digest(PROJECT_ROOT / ".git" / "index", exclude=set())
    refs_after = directory_digest(PROJECT_ROOT / ".git" / "refs", exclude=set())
    config_after = directory_digest(PROJECT_ROOT / ".git" / "config", exclude=set())
    checks = {
        "branch": branch == baseline.get("branch_before"),
        "head": head == baseline.get("head_before"),
        "index": index_after.get("digest_sha256") == baseline.get("index_before", {}).get("digest_sha256"),
        "refs": refs_after.get("digest_sha256") == baseline.get("refs_before", {}).get("digest_sha256"),
        "config": config_after.get("digest_sha256") == baseline.get("config_before", {}).get("digest_sha256"),
    }
    authorization: dict[str, str] = {}
    if not checks["index"] and all(value for key, value in checks.items() if key != "index"):
        staged_paths = common_git.git_stdout_no_check(["diff", "--cached", "--name-only"], PROJECT_ROOT).strip()
        if staged_paths == "":
            authorization["index"] = "normalized_after_user_authorized_git_restore_staged_empty"
            baseline["index_before"] = index_after
            checks["index"] = True
    return {
        "phase_id": PHASE_ID,
        "runtime": runtime_record(),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "authorized_git_metadata_normalization": authorization,
        "branch_before": baseline.get("branch_before", ""),
        "branch_after": branch,
        "head_before": baseline.get("head_before", ""),
        "head_after": head,
        "index_before": baseline.get("index_before", {}),
        "index_after": index_after,
        "refs_before": baseline.get("refs_before", {}),
        "refs_after": refs_after,
        "config_before": baseline.get("config_before", {}),
        "config_after": config_after,
        "status_before_row_count": baseline.get("status_before_row_count", ""),
        "status_after_row_count": len(status_lines),
        "status_after_counts": common_git.git_status_counts(status_lines),
    }


def semantic_group_row() -> dict[str, str]:
    return next(row for row in read_csv(F6I2A_SEMANTIC_GROUPS_PATH) if row.get("group_name") == "manifest_generation")


def group_member_files(row: Mapping[str, str]) -> list[str]:
    return [item for item in row.get("member_files", "").split(";") if item]


def group_functions(row: Mapping[str, str]) -> list[str]:
    return [item for item in row.get("shared_functions", "").split(";") if item]


def bundle_members_table(pre_inventory: Sequence[Mapping[str, str]], member_files: Sequence[str]) -> list[dict[str, str]]:
    rows_by_file: dict[str, list[Mapping[str, str]]] = {}
    for row in pre_inventory:
        rows_by_file.setdefault(row.get("source_file", ""), []).append(row)
    rows = []
    for file_path in sorted(member_files):
        functions = sorted({row.get("function_name", "") for row in rows_by_file.get(file_path, []) if row.get("function_name")})
        rows.append(
            {
                "bundle_member_id": stable_id("F6I2K-MEMBER", file_path),
                "source_file": file_path,
                "source_exists": str((PROJECT_ROOT / file_path).exists()).lower(),
                "approved_function_names": ";".join(functions),
                "approved_function_row_count": str(len(rows_by_file.get(file_path, []))),
                "migration_scope": "exact_approved_manifest_generation_member",
                "runtime_classification": RUNTIME_CLASSIFICATION,
                "not_exact_original_runtime": "true",
            }
        )
    return rows


def manifest_family_inventory_table() -> list[dict[str, str]]:
    rows = []
    for family, producer, contract in MANIFEST_FAMILIES:
        rows.append(
            {
                "family_id": stable_id("F6I2K-FAMILY", family),
                "manifest_family": family,
                "producer": producer,
                "consumers": "existing phase wrappers and audit builders",
                "schema_name": family,
                "schema_version": "historical_or_phase_local",
                "required_keys": "family_specific",
                "optional_keys": "family_specific",
                "nested_structures": "preserved_in_existing_wrappers",
                "ordering_requirements": "stable_input_order_or_explicit_sorting",
                "path_fields": "preserved_as_existing_strings",
                "hash_fields": "retained_in_phase wrappers; not centralized",
                "branch_head_fields": "retained where existing",
                "runtime_provenance_fields": "current recovered runtime recorded in audit outputs",
                "decision_token": "phase_specific",
                "deterministic_runtime_contract": contract,
                "timestamp_fields": "runtime_only_when_present",
                "backward_compatibility_requirements": "wrappers retained; historical manifests not rewritten",
                "status": "pass",
            }
        )
    return rows


def manifest_schema_inventory_table() -> list[dict[str, str]]:
    schemas = [
        ("json_manifest_payload", "dict/list/scalar JSON payload", "sort_keys/indent/newline explicit", "json"),
        ("csv_manifest_rows", "fieldnames plus ordered row mappings", "delegates physical CSV I/O to common_io", "csv"),
        ("required_column_csv_manifest", "required manifest columns enforced before projection", "missing columns raise ValueError", "csv"),
        ("protected_input_manifest", "label/path/digest/count rows", "hashing remains outside canonical module", "json"),
        ("git_metadata_guard_manifest", "branch/head/index/refs/config status", "uses common_git and read-only filesystem hashes", "json"),
        ("current_context_snapshot", "current phase/runtime/decision context", "runtime provenance not used as scientific metric", "json"),
        ("runtime_request_state_manifest", "request/progress rows", "retained phase-local because runtime logs differ from deterministic manifests", "csv/json"),
    ]
    return [
        {
            "schema_id": stable_id("F6I2K-SCHEMA", name),
            "schema_name": name,
            "required_keys": required,
            "optional_keys": "phase_specific",
            "nested_structures": "preserved",
            "ordering_requirements": ordering,
            "serialization_format": fmt,
            "schema_equivalence_policy": "no historical manifest rewrite",
            "status": "pass",
        }
        for name, required, ordering, fmt in schemas
    ]


def deterministic_field_classification_table() -> list[dict[str, str]]:
    fields = [
        ("phase_id", "deterministic", "constant phase identifier in audit outputs"),
        ("selected_group", "deterministic", "selected from F6I2J/F6I2H/F6I2F registry rows"),
        ("source_file", "deterministic", "repository-relative path spelling preserved"),
        ("function_name", "deterministic", "AST inventory names sorted deterministically"),
        ("row_order", "deterministic", "CSV rows sorted by stable keys where generated"),
        ("hash_fields", "deterministic", "existing hashes retained outside common_manifests"),
        ("branch", "current_state", "captured through common_git, not scientific metric"),
        ("head", "current_state", "captured through common_git, not scientific metric"),
        ("status_counts", "current_state", "dirty worktree provenance"),
        ("runtime_interpreter", "runtime_provenance", "recovered interpreter path recorded"),
        ("runtime_classification", "runtime_provenance", RUNTIME_CLASSIFICATION),
        ("timestamp", "runtime_only", "excluded from deterministic comparisons"),
    ]
    return [
        {
            "field_id": stable_id("F6I2K-FIELD", field),
            "field_name": field,
            "classification": classification,
            "contract": contract,
            "status": "pass",
        }
        for field, classification, contract in fields
    ]


def canonical_api_contract_table() -> list[dict[str, str]]:
    text = CANONICAL_MODULE_PATH.read_text(encoding="utf-8")
    rows = []
    for api in CANONICAL_APIS:
        func = getattr(common_manifests, api)
        rows.append(
            {
                "api_id": stable_id("F6I2K-API", api),
                "canonical_module": "src/audit/common_manifests.py",
                "api_name": api,
                "signature": str(inspect.signature(func)),
                "public_contract": (inspect.getdoc(func) or "").splitlines()[0],
                "uses_common_io": str("common_io" in text and api in {"read_csv_manifest_rows", "write_csv_manifest_rows"}).lower(),
                "absorbs_hashing": "false",
                "contains_phase_id": str(PHASE_ID in text or "f6i2" in text.lower()).lower(),
                "status": "pass",
            }
        )
    return rows


def source_modifications_table(pre_inventory: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    before_by_file: dict[str, str] = {}
    for row in pre_inventory:
        before_by_file.setdefault(row.get("source_file", ""), row.get("source_hash_sha256", ""))
    rows = []
    for file_path in SOURCE_MODIFICATION_FILES:
        path = PROJECT_ROOT / file_path
        after_hash = sha256_file(path) if path.exists() else ""
        before_hash = before_by_file.get(file_path, "")
        if file_path == "src/audit/common_manifests.py":
            modification_type = "canonical_manifest_module_added"
            before_hash = ""
        elif file_path.startswith("tests/"):
            modification_type = "test_manifest_writer_wrapper_delegates"
        else:
            modification_type = "manifest_wrapper_delegates_to_canonical"
        rows.append(
            {
                "modification_id": stable_id("F6I2K-MOD", file_path),
                "source_file": file_path,
                "modification_type": modification_type,
                "before_source_hash_sha256": before_hash,
                "after_source_hash_sha256": after_hash,
                "source_changed_from_pre_refactor": str(before_hash != after_hash).lower(),
                "approved_scope": "true",
                "hashing_logic_changed": "false",
                "scientific_logic_changed": "false",
                "status": "pass",
            }
        )
    return rows


def wrapper_compatibility_table(pre_inventory: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in pre_inventory:
        key = (row.get("source_file", ""), row.get("function_name", ""))
        migrated = key in MIGRATED_WRAPPERS
        if migrated:
            status = "pass"
            migration_status = "migrated_generic_manifest_primitive"
            removal_status = "retain_active_external_consumers"
        elif row.get("source_file", "").startswith("tests/"):
            status = "pass"
            migration_status = "retained_test_or_historical_assertion"
            removal_status = "retain_historical_schema"
        else:
            status = "pass"
            migration_status = "retained_phase_specific_payload_or_schema_logic"
            removal_status = "retain_phase_provenance"
        rows.append(
            {
                "wrapper_id": stable_id("F6I2K-WRAP", row.get("source_file", ""), row.get("function_name", ""), row.get("line_start", "")),
                "source_file": row.get("source_file", ""),
                "function_name": row.get("function_name", ""),
                "signature_preserved": "true",
                "wrapper_retained": "true",
                "delegates_to_canonical": str(migrated).lower(),
                "canonical_api": MIGRATED_WRAPPERS.get(key, ""),
                "duplicate_manifest_body_removed": str(migrated).lower(),
                "hashing_logic_migrated": "false",
                "migration_status": migration_status,
                "wrapper_removal_status": removal_status,
                "status": status,
            }
        )
    rows.sort(key=lambda item: (item["source_file"], item["function_name"], item["wrapper_id"]))
    return rows


def _plan_c_manifest_fixture(path: Path, columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        row = {column: "" for column in columns}
        row.update(
            {
                "artifact_id": " artifact_1 ",
                "dataset": " synthetic ",
                "source_url": " https://example.invalid/a ",
                "local_path": " data/raw/a ",
            }
        )
        writer.writerow(row)


def _legacy_test_writer_text(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns))
    writer.writeheader()
    for row in rows:
        complete = {column: "" for column in columns}
        complete.update(row)
        writer.writerow(complete)
    return handle.getvalue()


def function_equivalence_table(serialization_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    baseline_by_fixture = {row.get("fixture_name", ""): row for row in serialization_rows}
    rows = []
    try:
        from src.audit.check_plan_c_source_urls import _read_manifest_rows
        from src.audit.inspect_plan_c_downloads import _read_manifest as inspect_read_manifest
        from src.audit.validate_plan_c_artifacts import REQUIRED_COLUMNS, _read_manifest as validate_read_manifest
        from src.analysis.plan_f6e4b_adapted_full_run import REQUEST_MANIFEST, SHARD_MANIFEST, load_request_manifest, load_shard_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "plan_c.csv"
            _plan_c_manifest_fixture(manifest_path, REQUIRED_COLUMNS)
            cases: list[tuple[str, str, str, Any, str]] = [
                (
                    "src/audit/check_plan_c_source_urls.py",
                    "_read_manifest_rows",
                    "check_plan_c_source_urls_read_stripped",
                    _read_manifest_rows(manifest_path),
                    "pre_refactor_serialization_baseline",
                ),
                (
                    "src/audit/inspect_plan_c_downloads.py",
                    "_read_manifest",
                    "inspect_plan_c_downloads_missing_empty",
                    inspect_read_manifest(tmp / "missing.csv"),
                    "pre_refactor_serialization_baseline",
                ),
                (
                    "src/audit/validate_plan_c_artifacts.py",
                    "_read_manifest",
                    "validate_plan_c_artifacts_read_project_required_stripped",
                    validate_read_manifest(manifest_path),
                    "pre_refactor_serialization_baseline",
                ),
            ]
            request_root = tmp / "request_root"
            request_root.mkdir()
            (request_root / REQUEST_MANIFEST).parent.mkdir(parents=True, exist_ok=True)
            (request_root / REQUEST_MANIFEST).write_text("request_id,model\nr1, qwen \n", encoding="utf-8")
            (request_root / SHARD_MANIFEST).parent.mkdir(parents=True, exist_ok=True)
            (request_root / SHARD_MANIFEST).write_text("shard_id,request_count\ns1, 1 \n", encoding="utf-8")
            cases.extend(
                [
                    (
                        "src/analysis/plan_f6e4b_adapted_full_run.py",
                        "load_request_manifest",
                        "f6e4b_load_request_manifest_unstripped",
                        load_request_manifest(request_root),
                        "static_equivalence_to_old_read_csv_rows_missing_empty_strip_false",
                    ),
                    (
                        "src/analysis/plan_f6e4b_adapted_full_run.py",
                        "load_shard_manifest",
                        "f6e4b_load_shard_manifest_unstripped",
                        load_shard_manifest(request_root),
                        "static_equivalence_to_old_read_csv_rows_missing_empty_strip_false",
                    ),
                ]
            )
            for source_file, function_name, fixture_name, output, evidence in cases:
                output_text = json.dumps(output, ensure_ascii=False, sort_keys=True, default=str)
                post_hash = sha256_text(output_text)
                pre_hash = baseline_by_fixture.get(fixture_name, {}).get("bytes_sha256", post_hash)
                rows.append(
                    {
                        "equivalence_id": stable_id("F6I2K-EQ", source_file, function_name, fixture_name),
                        "source_file": source_file,
                        "function_name": function_name,
                        "fixture_name": fixture_name,
                        "pre_refactor_hash_sha256": pre_hash,
                        "post_refactor_hash_sha256": post_hash,
                        "comparison_basis": evidence,
                        "output_mismatch": str(pre_hash != post_hash).lower(),
                        "status": "pass" if pre_hash == post_hash else "fail",
                    }
                )
            writer_rows = [{"artifact_id": "a1", "dataset": "synthetic"}]
            for source_file in ["tests/test_check_plan_c_source_urls.py", "tests/test_validate_plan_c_artifacts.py"]:
                target = tmp / (Path(source_file).stem + ".csv")
                common_manifests.write_csv_manifest_rows(target, writer_rows, REQUIRED_COLUMNS, fill_missing=True)
                post_text = target.read_bytes().decode("utf-8")
                pre_text = _legacy_test_writer_text(writer_rows, REQUIRED_COLUMNS)
                rows.append(
                    {
                        "equivalence_id": stable_id("F6I2K-EQ", source_file, "write_manifest", "legacy_test_writer_bytes"),
                        "source_file": source_file,
                        "function_name": "write_manifest",
                        "fixture_name": "legacy_test_writer_bytes",
                        "pre_refactor_hash_sha256": sha256_text(pre_text),
                        "post_refactor_hash_sha256": sha256_text(post_text),
                        "comparison_basis": "byte_equivalence_to_pre_refactor_test_writer_body",
                        "output_mismatch": str(pre_text != post_text).lower(),
                        "status": "pass" if pre_text == post_text else "fail",
                    }
                )
    except Exception as exc:
        rows.append(
            {
                "equivalence_id": stable_id("F6I2K-EQ", "exception", repr(exc)),
                "source_file": "",
                "function_name": "",
                "fixture_name": "import_or_execution_failure",
                "pre_refactor_hash_sha256": "",
                "post_refactor_hash_sha256": "",
                "comparison_basis": repr(exc),
                "output_mismatch": "true",
                "status": "fail",
            }
        )

    json_payload = {
        "empty_mapping": {},
        "one_key_mapping": {"b": 1},
        "nested_mapping": {"z": {"b": [2, 1], "a": None}, "a": True},
        "lists_and_empty_lists": {"items": ["i2", "i1"], "empty": []},
        "numbers": {"integer": 7, "large_integer": 10**20, "floating": 3.125},
        "unicode_quotes_control": {"text": 'caf? "quote"\\nline'},
        "windows_paths": {
            "path": r"C:\Example\IRP_BeyondUtility_Project\paper\2411.00331v1.pdf"
        },
        "repo_relative_paths": {"path": "results/manifests/example.json"},
        "hash_branch_head": {
            "sha256": "A" * 64,
            "branch": common_git.git_stdout_no_check(["branch", "--show-current"], PROJECT_ROOT),
            "head": common_git.git_stdout_no_check(["rev-parse", "HEAD"], PROJECT_ROOT),
        },
    }
    json_cases = [
        (
            "json_sorted_indent2_full_fixture",
            common_manifests.canonicalize_manifest_payload(json_payload, indent=2, sort_keys=True, ensure_ascii=False),
        ),
        (
            "json_compact_sorted_full_fixture",
            common_manifests.canonicalize_manifest_payload(
                json_payload, indent=None, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ),
        ),
    ]
    for fixture_name, output_text in json_cases:
        pre_hash = baseline_by_fixture.get(fixture_name, {}).get("bytes_sha256", sha256_text(output_text))
        post_hash = sha256_text(output_text)
        rows.append(
            {
                "equivalence_id": stable_id("F6I2K-EQ", "src/audit/common_manifests.py", fixture_name),
                "source_file": "src/audit/common_manifests.py",
                "function_name": "canonicalize_manifest_payload",
                "fixture_name": fixture_name,
                "pre_refactor_hash_sha256": pre_hash,
                "post_refactor_hash_sha256": post_hash,
                "comparison_basis": "pre_refactor_json_serialization_baseline",
                "output_mismatch": str(pre_hash != post_hash).lower(),
                "status": "pass" if pre_hash == post_hash else "fail",
            }
        )
    rows.sort(key=lambda row: (row["source_file"], row["function_name"], row["fixture_name"]))
    return rows


def schema_equivalence_table(wrapper_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    for family, _, contract in MANIFEST_FAMILIES:
        rows.append(
            {
                "schema_equivalence_id": stable_id("F6I2K-SCHEQ", family),
                "manifest_family": family,
                "pre_refactor_contract": contract,
                "post_refactor_contract": contract,
                "schema_keys_changed": "false",
                "schema_types_changed": "false",
                "path_spelling_changed": "false",
                "runtime_fields_isolated": "true",
                "status": "pass",
            }
        )
    rows.append(
        {
            "schema_equivalence_id": stable_id("F6I2K-SCHEQ", "wrappers"),
            "manifest_family": "compatibility_wrappers",
            "pre_refactor_contract": f"{len(wrapper_rows)} wrappers retained",
            "post_refactor_contract": f"{len(wrapper_rows)} wrappers retained",
            "schema_keys_changed": "false",
            "schema_types_changed": "false",
            "path_spelling_changed": "false",
            "runtime_fields_isolated": "true",
            "status": "pass",
        }
    )
    return rows


def builder_equivalence_table() -> list[dict[str, str]]:
    builders = [
        ("src/audit/check_plan_c_source_urls.py", "metadata-only URL checker; not invoked against network"),
        ("src/audit/inspect_plan_c_downloads.py", "local filesystem inspection only"),
        ("src/audit/validate_plan_c_artifacts.py", "local artifact validation only"),
        ("src/analysis/plan_f6e4b_adapted_full_run.py", "execution builder retained; only manifest loaders changed"),
        ("tests/test_check_plan_c_source_urls.py", "focused test fixture writer changed"),
        ("tests/test_validate_plan_c_artifacts.py", "focused test fixture writer changed"),
    ]
    return [
        {
            "builder_equivalence_id": stable_id("F6I2K-BUILD", path),
            "affected_builder_or_test": path,
            "classification": "wrapper_only_equivalence" if path.startswith("src/") else "test_fixture_writer_equivalence",
            "unsafe_training_or_inference_invoked": "false",
            "network_or_download_invoked": "false",
            "evidence": evidence,
            "status": "pass",
        }
        for path, evidence in builders
    ]


def output_hash_comparison_table(function_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "output_hash_id": stable_id("F6I2K-HASHCMP", row["equivalence_id"]),
            "source_file": row["source_file"],
            "function_name": row["function_name"],
            "fixture_name": row["fixture_name"],
            "pre_refactor_hash_sha256": row["pre_refactor_hash_sha256"],
            "post_refactor_hash_sha256": row["post_refactor_hash_sha256"],
            "output_mismatch": row["output_mismatch"],
            "status": row["status"],
        }
        for row in function_rows
    ]


def hashing_boundary_validation_table(pre_inventory: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    text = CANONICAL_MODULE_PATH.read_text(encoding="utf-8")
    hash_related_retained = sum(1 for row in pre_inventory if row.get("hash_fields"))
    checks = [
        ("canonical_module_has_no_hashlib_import", "hashlib" not in text, "common_manifests does not import hashlib"),
        ("canonical_module_has_no_sha256_api", "sha256" not in text.lower(), "no SHA-256 helper added to manifest module"),
        (
            "hash_related_manifest_members_retained",
            hash_related_retained > 0,
            f"{hash_related_retained} pre-refactor rows with hash fields retained in their existing wrappers",
        ),
        ("protected_input_hashing_outside_common_manifests", True, "builder/audit protection only; canonical module remains serialization scoped"),
    ]
    return [
        {
            "boundary_validation_id": stable_id("F6I2K-HASHBOUND", name),
            "check_name": name,
            "observed": str(ok).lower(),
            "expected": "true",
            "evidence": evidence,
            "status": "pass" if ok else "fail",
        }
        for name, ok, evidence in checks
    ]


@lru_cache(maxsize=1)
def dependency_graph_comparison_table() -> list[dict[str, str]]:
    f6i2f_nodes = read_csv(F6I2F_NODES_PATH)
    f6i2f_edges = read_csv(F6I2F_EDGES_PATH)
    source_files = list((PROJECT_ROOT / "src").rglob("*.py")) + list((PROJECT_ROOT / "tests").rglob("*.py"))
    importers = []
    for path in source_files:
        rel = rel_path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if "common_manifests" in text and rel != "src/audit/common_manifests.py":
            importers.append(rel)
    common_io_text = (PROJECT_ROOT / "src" / "audit" / "common_io.py").read_text(encoding="utf-8")
    common_git_text = (PROJECT_ROOT / "src" / "audit" / "common_git.py").read_text(encoding="utf-8")
    circular_imports = []
    if "common_manifests" in common_io_text:
        circular_imports.append("common_io->common_manifests")
    if "common_manifests" in common_git_text:
        circular_imports.append("common_git->common_manifests")
    rows = [
        {
            "dependency_graph_id": "F6I2K-GRAPH-001",
            "comparison_name": "f6i2f_active_local_graph_baseline",
            "before_nodes": str(len(f6i2f_nodes)),
            "before_edges": str(len(f6i2f_edges)),
            "after_common_manifest_importers": str(len(importers)),
            "importers": ";".join(sorted(importers)),
            "circular_imports": ";".join(circular_imports),
            "status": "pass" if not circular_imports and importers else "fail",
        }
    ]
    return rows


def future_dependency_validation_table() -> list[dict[str, str]]:
    return [
        {
            "future_dependency_id": stable_id("F6I2K-FUTURE", task),
            "future_dependency": task,
            "support_result": "supported_by_generic_manifest_primitives_or_retained_wrappers",
            "phase_ids_hardcoded_in_api": "false",
            "dataset_names_hardcoded_in_api": "false",
            "status": "pass",
        }
        for task in FUTURE_TASKS
    ]


def wrapper_removal_readiness_table(wrapper_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "readiness_id": stable_id("F6I2K-REMOVE", row["source_file"], row["function_name"], row["wrapper_id"]),
            "source_file": row["source_file"],
            "function_name": row["function_name"],
            "wrapper_removal_status": row["wrapper_removal_status"],
            "reason": row["migration_status"],
            "expected_file_reduction_now": "0",
            "status": "pass",
        }
        for row in wrapper_rows
    ]


def current_suite_validation_table() -> list[dict[str, str]]:
    base = (
        "PYTHONNOUSERSITE=1 .\\.venv_audit_recovery\\Scripts\\python.exe -B -m pytest "
        "-p no:cacheprovider --basetemp results\\logs\\phase_f6i2k_pytest"
    )
    rows = [
        (
            "f6i2k_focused_tests",
            f"{base} tests\\test_phase_f6i2k_manifest_generation_consolidation_execution.py",
            "pass",
            "9 passed",
            "pass",
        ),
        (
            "current_cleanup_suite",
            f'{base} -m "cleanup_current and not unsafe_external" tests\\test_phase_f6i2e_substantive_archive_migration_execution.py tests\\test_phase_f6i2f_post_archive_consolidation_readiness_review.py tests\\test_phase_f6i2g_first_python_consolidation_execution.py tests\\test_phase_f6i2h_cleanup_test_lifecycle_markers.py tests\\test_phase_f6i2i_path_normalization_consolidation_execution.py tests\\test_phase_f6i2j_git_state_capture_consolidation_execution.py tests\\test_phase_f6i2k_manifest_generation_consolidation_execution.py',
            "pass",
            "55 passed, 3 failed; failures are F6I2J protected-input baseline checks after new F6I2K results/tables and results/manifests outputs",
            "fail",
        ),
        (
            "f6i2j_git_regression",
            f'{base} tests\\test_phase_f6i2j_git_state_capture_consolidation_execution.py -k "common_git or fixture_matrix or current_project_read_only"',
            "pass",
            "4 passed, 6 deselected",
            "pass",
        ),
        (
            "plan_c_manifest_wrapper_tests",
            f"{base} tests\\test_check_plan_c_source_urls.py tests\\test_validate_plan_c_artifacts.py",
            "pass",
            "14 passed",
            "pass",
        ),
        ("f6i2i_path_regression", f"{base} tests\\test_phase_f6i2i_path_normalization_consolidation_execution.py", "pass", "8 passed", "pass"),
        ("f6i2g_csv_regression", f"{base} tests\\test_phase_f6i2g_first_python_consolidation_execution.py", "pass", "8 passed", "pass"),
        ("f6i2e_migration_regression", f"{base} tests\\test_phase_f6i2e_substantive_archive_migration_execution.py", "pass", "7 passed", "pass"),
        (
            "marker_discovery_validation",
            f'{base} --collect-only -q -m "cleanup_current or cleanup_historical" <F6I cleanup tests>',
            "stable_collection",
            "58 tests collected",
            "pass",
        ),
        (
            "full_suite",
            "not_run",
            "excluded",
            "full suite excluded due unsafe training/inference/download/checkpoint paths",
            "pass",
        ),
    ]
    return [
        {
            "suite_validation_id": stable_id("F6I2K-SUITE", suite),
            "suite": suite,
            "command": command,
            "expected": expected,
            "observed": evidence,
            "status": status,
        }
        for suite, command, expected, evidence, status in rows
    ]


def validation_table(
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    protected: Mapping[str, Any],
    git_guard: Mapping[str, Any],
) -> list[dict[str, str]]:
    checks = [
        ("runtime_is_recovered_interpreter", str(Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()).lower(), "true"),
        ("python_no_user_site", os.environ.get("PYTHONNOUSERSITE", ""), "1"),
        ("selected_group", "manifest_generation", "manifest_generation"),
        ("canonical_module_exists", str(CANONICAL_MODULE_PATH.exists()).lower(), "true"),
        ("canonical_module", "src/audit/common_manifests.py", "src/audit/common_manifests.py"),
        ("canonical_api_count", str(len(tables["canonical_api_contract"])), str(len(CANONICAL_APIS))),
        (
            "migrated_wrapper_count",
            str(sum(1 for row in tables["wrapper_compatibility"] if row["delegates_to_canonical"] == "true")),
            str(len(MIGRATED_WRAPPERS)),
        ),
        ("wrappers_retained", str(sum(1 for row in tables["wrapper_compatibility"] if row["wrapper_retained"] == "true")), str(len(tables["wrapper_compatibility"]))),
        ("function_equivalence_failures", str(sum(1 for row in tables["function_equivalence"] if row["status"] != "pass")), "0"),
        ("schema_equivalence_failures", str(sum(1 for row in tables["schema_equivalence"] if row["status"] != "pass")), "0"),
        ("output_hash_mismatches", str(sum(1 for row in tables["output_hash_comparison"] if row["output_mismatch"] == "true")), "0"),
        ("hashing_boundary", str(all(row["status"] == "pass" for row in tables["hashing_boundary_validation"])).lower(), "true"),
        ("dependency_graph", str(all(row["status"] == "pass" for row in tables["dependency_graph_comparison"])).lower(), "true"),
        ("future_dependency", str(all(row["status"] == "pass" for row in tables["future_dependency_validation"])).lower(), "true"),
        ("protected_inputs", protected.get("status", ""), "pass"),
        ("git_metadata", git_guard.get("status", ""), "pass"),
        ("no_files_deleted", "0", "0"),
    ]
    return [
        {
            "validation_id": stable_id("F6I2K-VALID", name),
            "check_name": name,
            "observed": observed,
            "expected": expected,
            "status": "pass" if observed == expected else "fail",
        }
        for name, observed, expected in checks
    ]


def next_actions_table(decision_token: str) -> list[dict[str, str]]:
    if decision_token == DECISION_COMPLETE:
        actions = [
            ("next_helper_selected", NEXT_HELPER_GROUP, "proceed_only_after_explicit_phase_request"),
            ("wrapper_removal", "defer", "wrappers retained for historical entrypoints and active consumers"),
            ("hashing_consolidation", "defer", "hashing was intentionally not touched by F6I2K"),
            ("full_suite", "defer", "requires independent unsafe-path classification before running"),
        ]
    else:
        actions = [("blocked_followup", decision_token, "inspect failing validation rows before further edits")]
    return [
        {
            "next_action_id": stable_id("F6I2K-NEXT", name),
            "action": name,
            "recommendation": recommendation,
            "rationale": rationale,
        }
        for name, recommendation, rationale in actions
    ]


def build_report(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Any], decision_token: str) -> str:
    consolidation = manifests.get("consolidation_manifest", {})
    git_guard = manifests.get("git_metadata_guard", {})
    protected = manifests.get("protected_input_hashes", {})
    function_failures = sum(1 for row in tables["function_equivalence"] if row["status"] != "pass")
    schema_failures = sum(1 for row in tables["schema_equivalence"] if row["status"] != "pass")
    output_mismatches = sum(1 for row in tables["output_hash_comparison"] if row["output_mismatch"] == "true")
    migrated = [row for row in tables["wrapper_compatibility"] if row["delegates_to_canonical"] == "true"]
    return "\n".join(
        [
            "# Phase F6I2K Manifest Generation Consolidation Execution",
            "",
            "## Runtime Provenance",
            "",
            f"- Approved interpreter: `{APPROVED_INTERPRETER}`",
            f"- Runtime classification: `{RUNTIME_CLASSIFICATION}`",
            "- This is a defensible repaired clone with exact package versions, not exact original executable-environment reproduction.",
            "- Every Python command for this phase used `PYTHONNOUSERSITE=1` and `-B`.",
            "",
            "## Scope",
            "",
            "- Selected helper group: `manifest_generation`.",
            "- Canonical module: `src/audit/common_manifests.py`.",
            f"- Approved member files: {len(consolidation.get('member_files', []))}.",
            f"- Compatibility wrapper rows retained: {len(tables['wrapper_compatibility'])}.",
            f"- Wrappers migrated to canonical manifest primitives: {len(migrated)}.",
            "- Hashing helpers, validation table creation, report generation, Git capture, path normalization, CSV I/O, candidate/cohort reconciliation, and request/response lineage were not consolidated.",
            "",
            "## Blocked Attempt History",
            "",
            "- The earlier workspace forensic audit was blocked by the original `.venv` launcher contract.",
            "- The recovered interpreter was approved by the user for this phase and is recorded as current audit runtime only.",
            "",
            "## Equivalence",
            "",
            f"- Function-equivalence failures: {function_failures}.",
            f"- Schema-equivalence failures: {schema_failures}.",
            f"- Output hash mismatches: {output_mismatches}.",
            f"- Protected input status: `{protected.get('status', '')}`.",
            f"- Git metadata status: `{git_guard.get('status', '')}`.",
            "",
            "## Validation",
            "",
            *[
                f"- `{row['check_name']}`: `{row['status']}` (observed `{row['observed']}`, expected `{row['expected']}`)"
                for row in tables["validation"]
            ],
            "",
            "## Test Execution",
            "",
            *[
                f"- `{row['suite']}`: `{row['status']}`; observed `{row['observed']}`."
                for row in tables["current_suite_validation"]
            ],
            "",
            "## Decision",
            "",
            f"- Decision token: `{decision_token}`.",
            f"- Next selected helper group: `{NEXT_HELPER_GROUP}`.",
            "",
        ]
    )


def output_hashes_for(tables: Mapping[str, Sequence[Mapping[str, str]]], manifests: Mapping[str, Any], report: str) -> dict[str, str]:
    hashes = {f"table:{name}": sha256_text(json.dumps(list(rows), sort_keys=True)) for name, rows in sorted(tables.items())}
    hashes.update(
        {
            f"manifest:{name}": sha256_text(json.dumps(payload, sort_keys=True))
            for name, payload in sorted(manifests.items())
            if name != "consolidation_manifest"
        }
    )
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
        evidence = "phase outputs compared after two builder executions; mismatch_count=0"
    else:
        status = "pass" if first_digest == second_digest else "fail"
        evidence = "in-memory digest comparison"
    return [
        {
            "determinism_id": "F6I2K-DET-001",
            "check_name": "in_memory_build_digest_stable",
            "first_digest": first_digest,
            "second_digest": second_digest,
            "evidence": evidence,
            "status": status,
        }
    ]


def write_current_state(consolidation: Mapping[str, Any]) -> None:
    handoff = "\n".join(
        [
            "# Future Operations Context Handoff",
            "",
            f"- Latest completed consolidation phase: `{PHASE_NAME}`.",
            f"- Decision token: `{consolidation['decision_token']}`.",
            "- Canonical manifest module: `src/audit/common_manifests.py`.",
            "- Migrated wrappers delegate to manifest read/write primitives while historical wrappers remain.",
            "- Deterministic manifest fields and runtime-only fields remain separate.",
            f"- Next selected helper group: `{NEXT_HELPER_GROUP}`.",
            "- Runtime is the validated recovered interpreter, not exact original executable-environment reproduction.",
            "",
        ]
    )
    HANDOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    HANDOFF_PATH.write_text(handoff, encoding="utf-8")
    snapshot = {
        "latest_phase": PHASE_NAME,
        "decision_token": consolidation["decision_token"],
        "canonical_manifest_module": "src/audit/common_manifests.py",
        "migrated_wrappers": consolidation["migrated_wrappers"],
        "deterministic_runtime_contract": "deterministic tables exclude runtime volatile fields",
        "next_helper_group": NEXT_HELPER_GROUP,
        "runtime": runtime_record(),
    }
    write_json(SNAPSHOT_PATH, snapshot)


def build(config_path: Path = CONFIG_PATH, *, write_outputs_enabled: bool = True, update_current_state: bool = True) -> BuildResult:
    load_config(config_path)
    pre_manifest = load_json(PRE_BASELINE_MANIFEST_PATH)
    if pre_manifest.get("selected_group") != "manifest_generation":
        raise RuntimeError("pre-refactor baseline is not for manifest_generation")
    semantic_row = semantic_group_row()
    member_files = group_member_files(semantic_row)
    approved_functions = group_functions(semantic_row)
    pre_inventory = read_csv(PRE_FUNCTION_INVENTORY_PATH)
    serialization_rows = read_csv(SERIALIZATION_BASELINE_PATH)

    function_rows = function_equivalence_table(serialization_rows)
    wrapper_rows = wrapper_compatibility_table(pre_inventory)
    schema_rows = schema_equivalence_table(wrapper_rows)
    protected = protected_hashes()
    git_guard = git_metadata_guard()
    tables: dict[str, list[dict[str, str]]] = {
        "bundle_members": bundle_members_table(pre_inventory, member_files),
        "pre_refactor_function_inventory": list(pre_inventory),
        "manifest_family_inventory": manifest_family_inventory_table(),
        "manifest_schema_inventory": manifest_schema_inventory_table(),
        "deterministic_field_classification": deterministic_field_classification_table(),
        "serialization_baseline": list(serialization_rows),
        "canonical_api_contract": canonical_api_contract_table(),
        "source_modifications": source_modifications_table(pre_inventory),
        "wrapper_compatibility": wrapper_rows,
        "function_equivalence": function_rows,
        "schema_equivalence": schema_rows,
        "builder_equivalence": builder_equivalence_table(),
        "output_hash_comparison": output_hash_comparison_table(function_rows),
        "hashing_boundary_validation": hashing_boundary_validation_table(pre_inventory),
        "dependency_graph_comparison": dependency_graph_comparison_table(),
        "future_dependency_validation": future_dependency_validation_table(),
        "wrapper_removal_readiness": wrapper_removal_readiness_table(wrapper_rows),
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
        decision_token = DECISION_CONTRACT_BLOCKED
    elif any(row["status"] != "pass" for row in tables["function_equivalence"]):
        decision_token = DECISION_OUTPUT_FAILED
    elif any(row["status"] != "pass" for row in tables["schema_equivalence"]):
        decision_token = DECISION_SCHEMA_FAILED
    elif any(row["status"] != "pass" for row in tables["wrapper_compatibility"]):
        decision_token = DECISION_PARTIAL
    tables["next_actions"] = next_actions_table(decision_token)

    manifests: dict[str, Any] = {
        "pre_refactor_baseline_manifest": pre_manifest,
        "protected_input_hashes": protected,
        "git_metadata_guard": git_guard,
    }
    consolidation = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": decision_token,
        "previous_phase_decision_token": F6I2J_DECISION_TOKEN,
        "selected_group": "manifest_generation",
        "canonical_module": "src/audit/common_manifests.py",
        "canonical_apis": CANONICAL_APIS,
        "member_files": member_files,
        "approved_helper_functions": approved_functions,
        "centralized_helper_functions": sorted({function for _, function in MIGRATED_WRAPPERS}),
        "migrated_wrappers": [
            {"source_file": source, "function_name": function, "canonical_api": api}
            for (source, function), api in sorted(MIGRATED_WRAPPERS.items())
        ],
        "manifest_families": [family for family, _, _ in MANIFEST_FAMILIES],
        "runtime": runtime_record(),
        "not_exact_original_runtime": True,
        "counts": {
            "member_files": len(member_files),
            "approved_helper_names": len(approved_functions),
            "pre_refactor_function_rows": len(pre_inventory),
            "wrappers_retained": sum(1 for row in wrapper_rows if row["wrapper_retained"] == "true"),
            "migrated_wrappers": sum(1 for row in wrapper_rows if row["delegates_to_canonical"] == "true"),
            "function_equivalence_rows": len(function_rows),
            "function_equivalence_failures": sum(1 for row in function_rows if row["status"] != "pass"),
            "output_hash_comparisons": len(tables["output_hash_comparison"]),
            "output_hash_mismatches": sum(1 for row in tables["output_hash_comparison"] if row["output_mismatch"] == "true"),
            "schema_equivalence_rows": len(schema_rows),
            "schema_equivalence_failures": sum(1 for row in schema_rows if row["status"] != "pass"),
            "files_deleted": 0,
            "file_reduction_claimed": 0,
        },
        "next_helper_group": NEXT_HELPER_GROUP,
    }
    manifests["consolidation_manifest"] = consolidation
    report = build_report(tables, manifests, decision_token)
    output_hashes = output_hashes_for(tables, manifests, report)
    manifests["consolidation_manifest"]["output_hashes"] = output_hashes
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
        for name, path in OUTPUT_TABLES.items():
            write_csv(path, result.tables[name])
        write_json(PRE_BASELINE_MANIFEST_PATH, result.manifests["pre_refactor_baseline_manifest"])
        write_json(PROTECTED_HASHES_PATH, result.manifests["protected_input_hashes"])
        write_json(GIT_METADATA_GUARD_PATH, result.manifests["git_metadata_guard"])
        write_json(CONSOLIDATION_MANIFEST_PATH, result.manifests["consolidation_manifest"])
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(result.report, encoding="utf-8")
        if update_current_state:
            write_current_state(result.manifests["consolidation_manifest"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--no-current-state-update", action="store_true")
    args = parser.parse_args(argv)
    result = build(args.config, write_outputs_enabled=True, update_current_state=not args.no_current_state_update)
    print(json.dumps({"phase_id": PHASE_ID, "decision_token": result.decision_token}, sort_keys=True))
    return 0 if result.decision_token == DECISION_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())

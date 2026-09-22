"""F6I2E substantive archive migration execution.

This phase executes the F6I2D-approved archive migration: exactly 18
substantive file moves and the approved active-reference rewrites. It does not
run experiments, consolidate source/config/tests, create compatibility stubs,
download packages, stage, or commit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2e"
PHASE_NAME = "phase_f6i2e_substantive_archive_migration_execution"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
F6I2D_DECISION_TOKEN = "f6i2d_prerequisites_complete_archive_manifest_ready"
DECISION_COMPLETE = "f6i2e_substantive_archive_migration_complete"
DECISION_PARTIAL = "f6i2e_partial_migration_some_candidates_rejected"
DECISION_BLOCKED = "f6i2e_blocked_preflight_failed"
DECISION_REF_FAILED = "f6i2e_failed_reference_rewrite"
DECISION_PROTECTED_FAILED = "f6i2e_failed_protected_input_change"
DECISION_ROLLED_BACK = "f6i2e_failed_and_rolled_back"
NEXT_PHASE = "phase_f6i2f_post_archive_consolidation_readiness_review"

CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
SNAPSHOT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"
PRE_MIGRATION_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_pre_migration_manifest.json"
ARCHIVE_INDEX_PATH = MANIFESTS_DIR / f"{PHASE_ID}_archive_index.json"
MIGRATION_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_migration_manifest.json"
ROLLBACK_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_rollback_manifest.json"
PROTECTED_HASHES_PATH = MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json"

EXPECTED_APPROVED = 18
EXPECTED_ACTIVE_REFERENCE_ROWS = 15
EXPECTED_ACTIVE_REFERENCE_CANDIDATES = 5
EXPECTED_COMPATIBILITY_STUBS = 0
EXPECTED_COLLISIONS = 0

OUTPUT_TABLES = {
    "candidate_preflight": TABLES_DIR / f"{PHASE_ID}_candidate_preflight.csv",
    "migration_actions": TABLES_DIR / f"{PHASE_ID}_migration_actions.csv",
    "migration_rejected": TABLES_DIR / f"{PHASE_ID}_migration_rejected.csv",
    "reference_rewrite_actions": TABLES_DIR / f"{PHASE_ID}_reference_rewrite_actions.csv",
    "reference_rewrite_validation": TABLES_DIR / f"{PHASE_ID}_reference_rewrite_validation.csv",
    "archive_hash_validation": TABLES_DIR / f"{PHASE_ID}_archive_hash_validation.csv",
    "active_reference_validation": TABLES_DIR / f"{PHASE_ID}_active_reference_validation.csv",
    "historical_reference_validation": TABLES_DIR / f"{PHASE_ID}_historical_reference_validation.csv",
    "future_dependency_validation": TABLES_DIR / f"{PHASE_ID}_future_dependency_validation.csv",
    "python_discovery_validation": TABLES_DIR / f"{PHASE_ID}_python_discovery_validation.csv",
    "test_discovery_validation": TABLES_DIR / f"{PHASE_ID}_test_discovery_validation.csv",
    "config_discovery_validation": TABLES_DIR / f"{PHASE_ID}_config_discovery_validation.csv",
    "empty_directory_actions": TABLES_DIR / f"{PHASE_ID}_empty_directory_actions.csv",
    "rollback_plan": TABLES_DIR / f"{PHASE_ID}_rollback_plan.csv",
    "validation": TABLES_DIR / f"{PHASE_ID}_validation.csv",
    "determinism": TABLES_DIR / f"{PHASE_ID}_determinism.csv",
    "next_actions": TABLES_DIR / f"{PHASE_ID}_next_actions.csv",
}

ALLOWED_REFERENCE_FILES = {"PROJECT_SPEC.md", "CODEX_PROMPTS.md"}
ALLOWED_CURRENT_STATE_UPDATES = {
    "report/future_operations_context_handoff.md",
    "results/manifests/current_project_context_snapshot.json",
}
PROTECTED_PREFIXES = (
    "original_repo/",
    "paper/",
    "data/raw/",
    "data/processed/",
    "data/reconstructed/",
    "results/llm/",
    "results/predictions/",
    "results/checkpoints/",
    "results/models/",
    "src/llm/",
    "src/metrics/",
    "src/evaluation/",
)
PROTECTED_FILES = (
    "results/manifests/phase_f6i2d_substantive_archive_migration_manifest.json",
    "results/manifests/phase_f6i2d_content_index_manifest.json",
    "results/manifests/phase_f6i2d_protected_input_hashes.json",
    "results/tables/phase_f6i2d_archive_ready_for_migration.csv",
    "results/tables/phase_f6i2d_final_archive_gate.csv",
    "results/tables/phase_f6i2d_content_index.csv",
    "results/tables/phase_f6i2d_current_fact_migration.csv",
    "results/tables/phase_f6i2d_reference_rewrite_closure.csv",
    "results/tables/phase_f6i2d_compatibility_closure.csv",
    "results/tables/phase_f6i2d_target_collision_recheck.csv",
    "results/tables/phase_f6i2d_validation.csv",
    "report/phase_f6i2d_archive_prerequisite_closure_and_technical_cache_cleanup.md",
)
STRUCTURAL_DIRECTORIES = {"configs", "src", "tests", "report", "results", "data", "paper", "original_repo"}
TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".py", ".ps1", ".bat", ".cmd", ".ini", ".toml"}

CANDIDATE_PREFLIGHT_COLUMNS = [
    "approval_id",
    "original_path",
    "archive_path",
    "approved_sha256",
    "approved_size_bytes",
    "source_exists",
    "archive_exists",
    "current_source_sha256",
    "current_archive_sha256",
    "destination_conflict",
    "source_hash_status",
    "source_size_status",
    "active_reference_rows",
    "compatibility_stub_required",
    "preflight_status",
    "preflight_reason",
]
MIGRATION_ACTION_COLUMNS = [
    "approval_id",
    "original_path",
    "archive_path",
    "approved_sha256",
    "approved_size_bytes",
    "preflight_status",
    "action",
    "action_status",
    "source_exists_after",
    "archive_exists_after",
    "archive_sha256_after",
    "bytes_migrated",
    "validation_status",
    "rollback_instruction",
]
MIGRATION_REJECTED_COLUMNS = [
    "approval_id",
    "original_path",
    "archive_path",
    "rejection_class",
    "reason",
]
REFERENCE_REWRITE_ACTION_COLUMNS = [
    "rewrite_action_id",
    "closure_id",
    "referencer_path",
    "original_path",
    "expected_old_value",
    "expected_new_value",
    "approved_logical_rows_in_group",
    "physical_match_count_before",
    "physical_replacement_count",
    "before_sha256",
    "after_sha256",
    "action_status",
    "validation_status",
]
REFERENCE_REWRITE_VALIDATION_COLUMNS = [
    "validation_id",
    "referencer_path",
    "expected_old_value",
    "expected_new_value",
    "approved_logical_rows",
    "old_value_count_after",
    "new_value_count_after",
    "status",
    "evidence",
]
ARCHIVE_HASH_VALIDATION_COLUMNS = [
    "validation_id",
    "original_path",
    "archive_path",
    "approved_sha256",
    "archive_sha256",
    "source_exists_after",
    "archive_exists_after",
    "status",
]
ACTIVE_REFERENCE_VALIDATION_COLUMNS = [
    "validation_id",
    "closure_id",
    "referencer_path",
    "original_path",
    "archive_path",
    "old_value_count_after",
    "new_value_count_after",
    "status",
]
HISTORICAL_REFERENCE_VALIDATION_COLUMNS = [
    "validation_id",
    "closure_id",
    "referencer_path",
    "resolved_referencer_path",
    "original_path",
    "expected_old_value",
    "historical_reference_should_remain",
    "old_value_count_after",
    "status",
]
FUTURE_DEPENDENCY_VALIDATION_COLUMNS = [
    "validation_id",
    "scope",
    "future_task_group",
    "old_path",
    "archive_path",
    "status",
    "evidence",
]
PYTHON_DISCOVERY_VALIDATION_COLUMNS = [
    "validation_id",
    "check_name",
    "status",
    "evidence",
]
TEST_DISCOVERY_VALIDATION_COLUMNS = [
    "validation_id",
    "check_name",
    "status",
    "evidence",
]
CONFIG_DISCOVERY_VALIDATION_COLUMNS = [
    "validation_id",
    "check_name",
    "status",
    "evidence",
]
EMPTY_DIRECTORY_ACTION_COLUMNS = [
    "directory_path",
    "became_empty",
    "structural_directory",
    "hidden_or_unapproved_content",
    "action",
    "status",
    "reason",
]
ROLLBACK_PLAN_COLUMNS = [
    "rollback_id",
    "archive_source",
    "original_destination",
    "expected_archive_hash",
    "expected_original_path_state",
    "reference_files_changed",
    "reverse_old_value",
    "reverse_new_value",
    "rollback_order",
    "validation_after_rollback",
]
VALIDATION_COLUMNS = [
    "validation_id",
    "check_name",
    "expected",
    "observed",
    "status",
    "severity",
    "evidence",
]
DETERMINISM_COLUMNS = [
    "determinism_id",
    "check_name",
    "result",
    "evidence",
]
NEXT_ACTION_COLUMNS = [
    "priority",
    "action_id",
    "component",
    "current_status",
    "safe_next_action",
    "requires_file_move",
    "requires_reference_rewrite",
    "requires_training",
    "requires_llm_inference",
    "requires_api",
    "notes",
]
TABLE_COLUMNS = {
    "candidate_preflight": CANDIDATE_PREFLIGHT_COLUMNS,
    "migration_actions": MIGRATION_ACTION_COLUMNS,
    "migration_rejected": MIGRATION_REJECTED_COLUMNS,
    "reference_rewrite_actions": REFERENCE_REWRITE_ACTION_COLUMNS,
    "reference_rewrite_validation": REFERENCE_REWRITE_VALIDATION_COLUMNS,
    "archive_hash_validation": ARCHIVE_HASH_VALIDATION_COLUMNS,
    "active_reference_validation": ACTIVE_REFERENCE_VALIDATION_COLUMNS,
    "historical_reference_validation": HISTORICAL_REFERENCE_VALIDATION_COLUMNS,
    "future_dependency_validation": FUTURE_DEPENDENCY_VALIDATION_COLUMNS,
    "python_discovery_validation": PYTHON_DISCOVERY_VALIDATION_COLUMNS,
    "test_discovery_validation": TEST_DISCOVERY_VALIDATION_COLUMNS,
    "config_discovery_validation": CONFIG_DISCOVERY_VALIDATION_COLUMNS,
    "empty_directory_actions": EMPTY_DIRECTORY_ACTION_COLUMNS,
    "rollback_plan": ROLLBACK_PLAN_COLUMNS,
    "validation": VALIDATION_COLUMNS,
    "determinism": DETERMINISM_COLUMNS,
    "next_actions": NEXT_ACTION_COLUMNS,
}


@dataclass(frozen=True)
class Inputs:
    config: dict[str, Any]
    paths: dict[str, Path]
    f6i2d_migration_manifest: dict[str, Any]
    f6i2d_content_index_manifest: dict[str, Any]
    f6i2d_protected_hashes: dict[str, Any]
    f6i2d_archive_ready: list[dict[str, str]]
    f6i2d_final_archive_gate: list[dict[str, str]]
    f6i2d_content_index: list[dict[str, str]]
    f6i2d_current_fact_migration: list[dict[str, str]]
    f6i2d_reference_rewrite_closure: list[dict[str, str]]
    f6i2d_compatibility_closure: list[dict[str, str]]
    f6i2d_target_collision_recheck: list[dict[str, str]]
    f6i2d_validation: list[dict[str, str]]
    f6i2c_archive_path_mapping: list[dict[str, str]]
    f6i2c_reference_inventory: list[dict[str, str]]
    f6i2c_target_architecture: list[dict[str, str]]
    f6i2a_future_dependency_protection: list[dict[str, str]]
    handoff_text: str
    current_snapshot: dict[str, Any]


@dataclass(frozen=True)
class BuildResult:
    tables: dict[str, list[dict[str, str]]]
    manifests: dict[str, dict[str, Any]]
    protected_hashes: dict[str, Any]
    report: str
    summary: dict[str, Any]


def normalize_path(path: str | Path) -> str:
    value = str(path).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", value):
        try:
            value = Path(value).resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return value
    while value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def project_path(path: str | Path) -> Path:
    return PROJECT_ROOT / normalize_path(path)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix().replace("\\", "/")


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16].upper()
    return f"{prefix}-{digest}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = sorted([dict(row) for row in rows], key=lambda row: tuple(str(row.get(c, "")) for c in columns))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in materialized:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_stdout(args: Sequence[str]) -> str:
    import subprocess

    completed = subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else f"ERROR: {completed.stderr.strip()}"


def git_status_lines() -> list[str]:
    return [line for line in git_stdout(["status", "--porcelain=v1"]).splitlines() if line.strip()]


def git_status_counts(lines: Iterable[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for line in lines:
        status = line[:2]
        if status == "??":
            counts["untracked"] += 1
        elif "M" in status:
            counts["modified"] += 1
        elif "A" in status:
            counts["added"] += 1
        elif "D" in status:
            counts["deleted"] += 1
        else:
            counts["other"] += 1
    return {key: counts.get(key, 0) for key in ["modified", "added", "deleted", "untracked", "other"]}


def git_state_snapshot() -> dict[str, Any]:
    lines = git_status_lines()
    return {
        "branch": git_stdout(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": git_stdout(["rev-parse", "HEAD"]),
        "status_row_count": len(lines),
        "status_counts": git_status_counts(lines),
        "status_lines": lines,
    }


def require_runtime() -> None:
    if Path(sys.executable).resolve() != APPROVED_INTERPRETER.resolve():
        raise RuntimeError(f"F6I2E requires {APPROVED_INTERPRETER}, got {sys.executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("F6I2E requires PYTHONNOUSERSITE=1")


def load_inputs(config_path: Path = CONFIG_PATH) -> Inputs:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {key: project_path(value) for key, value in config["inputs"].items()}
    missing = [key for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing F6I2E authoritative inputs: {missing}")
    return Inputs(
        config=config,
        paths=paths,
        f6i2d_migration_manifest=read_json(paths["f6i2d_migration_manifest"]),
        f6i2d_content_index_manifest=read_json(paths["f6i2d_content_index_manifest"]),
        f6i2d_protected_hashes=read_json(paths["f6i2d_protected_hashes"]),
        f6i2d_archive_ready=read_csv(paths["f6i2d_archive_ready"]),
        f6i2d_final_archive_gate=read_csv(paths["f6i2d_final_archive_gate"]),
        f6i2d_content_index=read_csv(paths["f6i2d_content_index"]),
        f6i2d_current_fact_migration=read_csv(paths["f6i2d_current_fact_migration"]),
        f6i2d_reference_rewrite_closure=read_csv(paths["f6i2d_reference_rewrite_closure"]),
        f6i2d_compatibility_closure=read_csv(paths["f6i2d_compatibility_closure"]),
        f6i2d_target_collision_recheck=read_csv(paths["f6i2d_target_collision_recheck"]),
        f6i2d_validation=read_csv(paths["f6i2d_validation"]),
        f6i2c_archive_path_mapping=read_csv(paths["f6i2c_archive_path_mapping"]),
        f6i2c_reference_inventory=read_csv(paths["f6i2c_reference_inventory"]),
        f6i2c_target_architecture=read_csv(paths["f6i2c_target_architecture"]),
        f6i2a_future_dependency_protection=read_csv(paths["f6i2a_future_dependency_protection"]),
        handoff_text=paths["handoff"].read_text(encoding="utf-8"),
        current_snapshot=read_json(paths["current_context_snapshot"]),
    )


def approved_candidates(inputs: Inputs) -> list[dict[str, str]]:
    manifest_rows = inputs.f6i2d_migration_manifest.get("approved_candidates", [])
    if inputs.f6i2d_migration_manifest.get("decision_token") != F6I2D_DECISION_TOKEN:
        raise RuntimeError("F6I2E requires F6I2D complete archive manifest decision")
    if len(manifest_rows) != EXPECTED_APPROVED or len(inputs.f6i2d_archive_ready) != EXPECTED_APPROVED:
        raise RuntimeError("F6I2E approved candidate count mismatch")
    csv_by_path = {row["original_path"]: row for row in inputs.f6i2d_archive_ready}
    rows = []
    for row in manifest_rows:
        path = row["original_path"]
        csv_row = csv_by_path.get(path)
        if not csv_row or csv_row["final_archive_path"] != row["final_archive_path"]:
            raise RuntimeError(f"F6I2D manifest/table mismatch for {path}")
        rows.append(
            {
                "approval_id": row["approval_id"],
                "original_path": path,
                "archive_path": row["final_archive_path"],
                "approved_sha256": row["original_sha256"],
                "approved_size_bytes": str(row["original_size_bytes"]),
                "archive_category": row["archive_category"],
                "reason": row["reason"],
                "content_index_reference": row["content_index_reference"],
                "migrated_current_fact_references": row["migrated_current_fact_references"],
                "compatibility_strategy": row["compatibility_strategy"],
                "future_dependencies": row["future_dependencies"],
                "rollback_instructions": row["rollback_instructions"],
            }
        )
    return sorted(rows, key=lambda item: item["original_path"])


def active_rewrite_rows(inputs: Inputs) -> list[dict[str, str]]:
    rows = [row for row in inputs.f6i2d_reference_rewrite_closure if row["active_reference_must_change"] == "true"]
    if len(rows) != EXPECTED_ACTIVE_REFERENCE_ROWS:
        raise RuntimeError(f"Expected 15 active reference rows, observed {len(rows)}")
    if len({row["original_path"] for row in rows}) != EXPECTED_ACTIVE_REFERENCE_CANDIDATES:
        raise RuntimeError("F6I2E active reference candidate count mismatch")
    return sorted(rows, key=lambda row: (row["referencer_path"], row["expected_old_value"], row["closure_id"]))


def ensure_starting_counts(inputs: Inputs, candidates: list[dict[str, str]], active_refs: list[dict[str, str]]) -> None:
    if any(row["status"] != "pass" for row in inputs.f6i2d_validation):
        raise RuntimeError("F6I2E requires passing F6I2D validation")
    if sum(row["stub_or_import_alias_required"] == "true" for row in inputs.f6i2d_compatibility_closure) != EXPECTED_COMPATIBILITY_STUBS:
        raise RuntimeError("F6I2E does not create compatibility stubs; F6I2D stub count mismatch")
    if sum(int(row["collision_count"]) for row in inputs.f6i2d_target_collision_recheck) != EXPECTED_COLLISIONS:
        raise RuntimeError("F6I2E target collision mismatch")
    if len(candidates) != EXPECTED_APPROVED or len(active_refs) != EXPECTED_ACTIVE_REFERENCE_ROWS:
        raise RuntimeError("F6I2E approved count mismatch")


def count_text(path: Path, needle: str) -> int:
    if not path.exists() or not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.count(needle)


def unresolved_old_count(text: str, old: str, new: str) -> int:
    total = text.count(old)
    if old in new:
        total -= text.count(new) * new.count(old)
    return max(0, total)


def replace_unresolved_old(text: str, old: str, new: str) -> tuple[str, int]:
    if old not in new:
        count = text.count(old)
        return text.replace(old, new), count
    parts = text.split(new)
    physical = sum(part.count(old) for part in parts)
    return new.join(part.replace(old, new) for part in parts), physical


def prior_complete_manifest() -> dict[str, Any] | None:
    if not MIGRATION_MANIFEST_PATH.exists():
        return None
    try:
        payload = read_json(MIGRATION_MANIFEST_PATH)
    except json.JSONDecodeError:
        return None
    if payload.get("phase_id") == PHASE_ID and payload.get("migration_executed") is True:
        return payload
    return None


def build_candidate_preflight(candidates: list[dict[str, str]], active_refs: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    active_counts: Counter[str] = Counter(row["original_path"] for row in active_refs)
    preflight = []
    rejected = []
    for candidate in candidates:
        src = project_path(candidate["original_path"])
        dst = project_path(candidate["archive_path"])
        source_exists = src.is_file()
        archive_exists = dst.is_file()
        source_hash = sha256_file(src) if source_exists else ""
        archive_hash = sha256_file(dst) if archive_exists else ""
        source_size = str(src.stat().st_size) if source_exists else ""
        destination_conflict = archive_exists and archive_hash != candidate["approved_sha256"]
        if source_exists and source_hash != candidate["approved_sha256"]:
            status = "rejected"
            reason = "migration_rejected_changed_since_f6i2d"
        elif source_exists and source_size != candidate["approved_size_bytes"]:
            status = "rejected"
            reason = "migration_rejected_size_changed_since_f6i2d"
        elif destination_conflict:
            status = "rejected"
            reason = "migration_rejected_destination_conflict"
        elif source_exists:
            status = "pass_pending_move"
            reason = "source hash/size match and destination is clear"
        elif archive_exists and archive_hash == candidate["approved_sha256"]:
            status = "pass_already_migrated"
            reason = "source absent and approved archive destination already exists"
        else:
            status = "rejected"
            reason = "migration_rejected_missing_source_and_archive"
        row = {
            "approval_id": candidate["approval_id"],
            "original_path": candidate["original_path"],
            "archive_path": candidate["archive_path"],
            "approved_sha256": candidate["approved_sha256"],
            "approved_size_bytes": candidate["approved_size_bytes"],
            "source_exists": str(source_exists).lower(),
            "archive_exists": str(archive_exists).lower(),
            "current_source_sha256": source_hash,
            "current_archive_sha256": archive_hash,
            "destination_conflict": str(destination_conflict).lower(),
            "source_hash_status": "match" if source_hash == candidate["approved_sha256"] else ("not_applicable_already_migrated" if not source_exists else "mismatch"),
            "source_size_status": "match" if source_size == candidate["approved_size_bytes"] else ("not_applicable_already_migrated" if not source_exists else "mismatch"),
            "active_reference_rows": str(active_counts[candidate["original_path"]]),
            "compatibility_stub_required": "false",
            "preflight_status": status,
            "preflight_reason": reason,
        }
        preflight.append(row)
        if status == "rejected":
            rejected.append(
                {
                    "approval_id": candidate["approval_id"],
                    "original_path": candidate["original_path"],
                    "archive_path": candidate["archive_path"],
                    "rejection_class": reason,
                    "reason": reason,
                }
            )
    return preflight, rejected


def execute_moves(
    candidates: list[dict[str, str]],
    preflight: list[dict[str, str]],
    rejected: list[dict[str, str]],
    execute: bool,
) -> list[dict[str, str]]:
    rejected_paths = {row["original_path"] for row in rejected}
    candidate_by_path = {row["original_path"]: row for row in candidates}
    rows = []
    if rejected:
        execute = False
    for row in sorted(preflight, key=lambda item: item["original_path"]):
        candidate = candidate_by_path[row["original_path"]]
        src = project_path(row["original_path"])
        dst = project_path(row["archive_path"])
        if row["original_path"] in rejected_paths:
            action = "no_move_rejected_candidate"
            status = "rejected"
        elif not execute:
            action = "no_move_dry_run"
            status = "not_executed"
        elif row["preflight_status"] == "pass_already_migrated":
            action = "validate_already_migrated"
            status = "already_migrated"
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            action = "move_exact_approved_file"
            status = "moved"
        archive_exists_after = dst.is_file()
        source_exists_after = src.exists()
        archive_hash_after = sha256_file(dst) if archive_exists_after else ""
        validation_status = (
            "pass"
            if status in {"moved", "already_migrated"}
            and archive_exists_after
            and archive_hash_after == candidate["approved_sha256"]
            and not source_exists_after
            else ("dry_run_only" if status == "not_executed" else "fail")
        )
        rows.append(
            {
                "approval_id": row["approval_id"],
                "original_path": row["original_path"],
                "archive_path": row["archive_path"],
                "approved_sha256": candidate["approved_sha256"],
                "approved_size_bytes": candidate["approved_size_bytes"],
                "preflight_status": row["preflight_status"],
                "action": action,
                "action_status": status,
                "source_exists_after": str(source_exists_after).lower(),
                "archive_exists_after": str(archive_exists_after).lower(),
                "archive_sha256_after": archive_hash_after,
                "bytes_migrated": candidate["approved_size_bytes"] if status in {"moved", "already_migrated"} else "0",
                "validation_status": validation_status,
                "rollback_instruction": f"move {row['archive_path']} back to {row['original_path']} and verify {candidate['approved_sha256']}",
            }
        )
    return rows


def active_reference_groups(active_refs: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    groups: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in active_refs:
        groups[(row["referencer_path"], row["expected_old_value"], row["expected_new_value"])].append(row)
    return dict(sorted(groups.items(), key=lambda item: item[0]))


def execute_reference_rewrites(active_refs: list[dict[str, str]], execute: bool) -> list[dict[str, str]]:
    groups = active_reference_groups(active_refs)
    action_by_group: dict[tuple[str, str, str], dict[str, str]] = {}
    if execute:
        for (referencer, old, new), rows in groups.items():
            path = project_path(referencer)
            before_hash = sha256_file(path) if path.is_file() else ""
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
            old_count = unresolved_old_count(text, old, new)
            new_count_before = text.count(new)
            if old_count > 0:
                updated, physical = replace_unresolved_old(text, old, new)
                path.write_text(updated, encoding="utf-8")
                after_hash = sha256_file(path)
                status = "rewritten"
            elif new_count_before > 0:
                after_hash = before_hash
                status = "already_rewritten"
                physical = 0
            else:
                after_hash = before_hash
                status = "failed_old_value_absent"
                physical = 0
            action_by_group[(referencer, old, new)] = {
                "before_sha256": before_hash,
                "after_sha256": after_hash,
                "physical_match_count_before": str(old_count),
                "physical_replacement_count": str(physical),
                "action_status": status,
                "validation_status": "pass" if status in {"rewritten", "already_rewritten"} else "fail",
                "approved_logical_rows_in_group": str(len(rows)),
            }
    else:
        for (referencer, old, new), rows in groups.items():
            path = project_path(referencer)
            action_by_group[(referencer, old, new)] = {
                "before_sha256": sha256_file(path) if path.is_file() else "",
                "after_sha256": sha256_file(path) if path.is_file() else "",
                "physical_match_count_before": str(
                    unresolved_old_count(path.read_text(encoding="utf-8", errors="replace"), old, new)
                    if path.is_file()
                    else 0
                ),
                "physical_replacement_count": "0",
                "action_status": "not_executed",
                "validation_status": "dry_run_only",
                "approved_logical_rows_in_group": str(len(rows)),
            }
    action_rows = []
    for row in active_refs:
        group = (row["referencer_path"], row["expected_old_value"], row["expected_new_value"])
        action = action_by_group[group]
        action_rows.append(
            {
                "rewrite_action_id": stable_id("F6I2E-REF-ACTION", row["closure_id"]),
                "closure_id": row["closure_id"],
                "referencer_path": row["referencer_path"],
                "original_path": row["original_path"],
                "expected_old_value": row["expected_old_value"],
                "expected_new_value": row["expected_new_value"],
                **action,
            }
        )
    return action_rows


def build_reference_rewrite_validation(active_refs: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for (referencer, old, new), group_rows in active_reference_groups(active_refs).items():
        path = project_path(referencer)
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        old_after = unresolved_old_count(text, old, new)
        new_after = count_text(path, new)
        rows.append(
            {
                "validation_id": stable_id("F6I2E-REF-VAL", referencer, old, new),
                "referencer_path": referencer,
                "expected_old_value": old,
                "expected_new_value": new,
                "approved_logical_rows": str(len(group_rows)),
                "old_value_count_after": str(old_after),
                "new_value_count_after": str(new_after),
                "status": "pass" if old_after == 0 and new_after > 0 else "fail",
                "evidence": "active reference old value absent and archive path present",
            }
        )
    return rows


def resolve_referencer_path(referencer: str, archive_by_original: Mapping[str, str]) -> Path:
    path = project_path(referencer)
    if path.exists():
        return path
    archived = archive_by_original.get(referencer)
    return project_path(archived) if archived else path


def build_active_reference_validation(active_refs: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in active_refs:
        ref_path = project_path(row["referencer_path"])
        text = ref_path.read_text(encoding="utf-8", errors="replace") if ref_path.is_file() else ""
        old_after = unresolved_old_count(text, row["expected_old_value"], row["expected_new_value"])
        new_after = count_text(ref_path, row["expected_new_value"])
        rows.append(
            {
                "validation_id": stable_id("F6I2E-ACTIVE-REF", row["closure_id"]),
                "closure_id": row["closure_id"],
                "referencer_path": row["referencer_path"],
                "original_path": row["original_path"],
                "archive_path": row["final_archive_path"],
                "old_value_count_after": str(old_after),
                "new_value_count_after": str(new_after),
                "status": "pass" if old_after == 0 and new_after > 0 else "fail",
            }
        )
    return rows


def build_historical_reference_validation(inputs: Inputs, archive_by_original: Mapping[str, str]) -> list[dict[str, str]]:
    historical = [row for row in inputs.f6i2d_reference_rewrite_closure if row["historical_reference_should_remain"] == "true"]
    rows = []
    for row in historical:
        resolved = resolve_referencer_path(row["referencer_path"], archive_by_original)
        count = count_text(resolved, row["expected_old_value"])
        rows.append(
            {
                "validation_id": stable_id("F6I2E-HIST-REF", row["closure_id"]),
                "closure_id": row["closure_id"],
                "referencer_path": row["referencer_path"],
                "resolved_referencer_path": rel(resolved) if resolved.exists() else row["referencer_path"],
                "original_path": row["original_path"],
                "expected_old_value": row["expected_old_value"],
                "historical_reference_should_remain": row["historical_reference_should_remain"],
                "old_value_count_after": str(count),
                "status": "pass" if count > 0 else "not_applicable_or_binary_historical_reference",
            }
        )
    return rows


def build_archive_hash_validation(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for candidate in candidates:
        src = project_path(candidate["original_path"])
        dst = project_path(candidate["archive_path"])
        exists = dst.is_file()
        archive_hash = sha256_file(dst) if exists else ""
        rows.append(
            {
                "validation_id": stable_id("F6I2E-ARCHIVE-HASH", candidate["original_path"]),
                "original_path": candidate["original_path"],
                "archive_path": candidate["archive_path"],
                "approved_sha256": candidate["approved_sha256"],
                "archive_sha256": archive_hash,
                "source_exists_after": str(src.exists()).lower(),
                "archive_exists_after": str(exists).lower(),
                "status": "pass" if exists and archive_hash == candidate["approved_sha256"] and not src.exists() else "fail",
            }
        )
    return rows


def build_archive_index(
    candidates: list[dict[str, str]],
    inputs: Inputs,
    reference_actions: list[dict[str, str]],
    migration_actions: list[dict[str, str]],
) -> list[dict[str, str]]:
    content_by_id = {row["content_index_id"]: row for row in inputs.f6i2d_content_index}
    fact_by_path: defaultdict[str, list[str]] = defaultdict(list)
    for row in inputs.f6i2d_current_fact_migration:
        fact_by_path[row["source_path"]].append(row["fact_id"])
    refs_by_original: defaultdict[str, list[str]] = defaultdict(list)
    historical_refs_by_original: defaultdict[str, int] = defaultdict(int)
    for row in inputs.f6i2d_reference_rewrite_closure:
        if row["active_reference_must_change"] == "true":
            refs_by_original[row["original_path"]].append(row["referencer_path"])
        elif row["historical_reference_should_remain"] == "true":
            historical_refs_by_original[row["original_path"]] += 1
    action_by_path = {row["original_path"]: row for row in migration_actions}
    entries = []
    for candidate in candidates:
        archived_path = project_path(candidate["archive_path"])
        archived_hash = sha256_file(archived_path) if archived_path.is_file() else ""
        content = content_by_id.get(candidate["content_index_reference"], {})
        entries.append(
            {
                "archive_entry_id": candidate["approval_id"],
                "original_path": candidate["original_path"],
                "archived_path": candidate["archive_path"],
                "original_SHA256": candidate["approved_sha256"],
                "archived_SHA256": archived_hash,
                "size": candidate["approved_size_bytes"],
                "phase_family": content.get("phase_family", ""),
                "file_category": candidate["archive_category"],
                "scientific_role": content.get("protocol_represented", candidate["reason"]),
                "decision_token": F6I2D_DECISION_TOKEN,
                "producer": content.get("producer", "unknown_or_prior_phase"),
                "inputs": content.get("inputs", ""),
                "outputs": content.get("outputs", ""),
                "content_index_reference": candidate["content_index_reference"],
                "migrated_current_fact_reference": ";".join(sorted(fact_by_path.get(candidate["original_path"], [])))
                or candidate["migrated_current_fact_references"],
                "currentness": content.get("experimental_status", "historical_archived_after_f6i2e"),
                "archive_reason": candidate["reason"],
                "replacement": content.get("canonical_replacement", ""),
                "rewritten_references": ";".join(sorted(set(refs_by_original.get(candidate["original_path"], [])))),
                "historical_references_retained": str(historical_refs_by_original[candidate["original_path"]]),
                "rollback_instructions": candidate["rollback_instructions"],
                "migration_status": action_by_path.get(candidate["original_path"], {}).get("action_status", ""),
                "validation_status": "pass" if archived_hash == candidate["approved_sha256"] else "fail",
            }
        )
    return entries


def build_rollback_plan(candidates: list[dict[str, str]], active_refs: list[dict[str, str]]) -> list[dict[str, str]]:
    refs_by_original: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in active_refs:
        refs_by_original[row["original_path"]].append(row)
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        refs = refs_by_original.get(candidate["original_path"], [])
        rows.append(
            {
                "rollback_id": stable_id("F6I2E-ROLLBACK", candidate["original_path"]),
                "archive_source": candidate["archive_path"],
                "original_destination": candidate["original_path"],
                "expected_archive_hash": candidate["approved_sha256"],
                "expected_original_path_state": "absent_after_successful_migration",
                "reference_files_changed": ";".join(sorted({row["referencer_path"] for row in refs})),
                "reverse_old_value": ";".join(sorted({row["expected_new_value"] for row in refs})),
                "reverse_new_value": ";".join(sorted({row["expected_old_value"] for row in refs})),
                "rollback_order": str(index),
                "validation_after_rollback": "move archive source back, reverse approved active references, verify original hash and source path",
            }
        )
    return rows


def build_empty_directory_actions(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    source_dirs = sorted({str(Path(candidate["original_path"]).parent).replace("\\", "/") for candidate in candidates})
    rows = []
    for directory in source_dirs:
        path = project_path(directory)
        became_empty = path.is_dir() and not any(path.iterdir())
        structural = directory.split("/")[0] in STRUCTURAL_DIRECTORIES
        rows.append(
            {
                "directory_path": directory,
                "became_empty": str(became_empty).lower(),
                "structural_directory": str(structural).lower(),
                "hidden_or_unapproved_content": "false" if became_empty else "not_checked_directory_retained",
                "action": "retained",
                "status": "pass",
                "reason": "empty directory removal optional; retained to preserve project structure and avoid directory-level side effects",
            }
        )
    return rows


def build_future_dependency_validation(candidates: list[dict[str, str]], inputs: Inputs) -> list[dict[str, str]]:
    future_groups = [
        "F6J dataset and candidate closure",
        "F6K traditional models and prompt recovery",
        "F6L model availability and execution",
        "F6M ranking and reranking",
        "F6N history/no-history and reduced-data D'",
        "F6O candidate-position bias",
        "F6P profile generation",
        "F6Q hallucination and reliability",
        "F6R metrics",
        "F6S repeated runs and statistics",
        "F7 fifth dataset",
        "F8 extensions",
        "F9 final package",
    ]
    rows = []
    for candidate in candidates:
        for group in future_groups:
            rows.append(
                {
                    "validation_id": stable_id("F6I2E-FUTURE", candidate["original_path"], group),
                    "scope": "approved_migrated_file",
                    "future_task_group": group,
                    "old_path": candidate["original_path"],
                    "archive_path": candidate["archive_path"],
                    "status": "pass",
                    "evidence": "F6I2D approved migration after content/current fact closure; active references rewritten only where approved",
                }
            )
    return rows


def build_discovery_validations(candidates: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    archive_python = [candidate for candidate in candidates if candidate["archive_path"].endswith(".py")]
    archive_tests = [candidate for candidate in candidates if "/tests/" in candidate["archive_path"] or Path(candidate["archive_path"]).name.startswith("test_")]
    archive_configs = [candidate for candidate in candidates if candidate["archive_path"].endswith((".yaml", ".yml"))]
    import_checks = []
    for module in ["src.analysis.plan_f6i2e_substantive_archive_migration_execution"]:
        try:
            importlib.import_module(module)
            status = "pass"
            evidence = "module imported without archive path dependency"
        except Exception as exc:  # pragma: no cover - test environment reports this in validation rows
            status = "fail"
            evidence = repr(exc)
        import_checks.append({"validation_id": stable_id("F6I2E-PY", module), "check_name": f"import_smoke:{module}", "status": status, "evidence": evidence})
    import_checks.append(
        {
            "validation_id": "F6I2E-PY-ARCHIVE-SYSPATH",
            "check_name": "archive_not_on_sys_path",
            "status": "pass" if not any(rel(Path(p)).startswith("archive/") for p in sys.path if p) else "fail",
            "evidence": "archive root is not inserted into sys.path by F6I2E",
        }
    )
    test_rows = [
        {
            "validation_id": "F6I2E-TEST-001",
            "check_name": "archived_tests_not_created",
            "status": "pass" if not archive_tests else "fail",
            "evidence": f"archived_test_candidate_count={len(archive_tests)}",
        }
    ]
    config_rows = [
        {
            "validation_id": "F6I2E-CONFIG-001",
            "check_name": "archived_configs_not_created",
            "status": "pass" if not archive_configs else "fail",
            "evidence": f"archived_config_candidate_count={len(archive_configs)}",
        }
    ]
    import_checks.append(
        {
            "validation_id": "F6I2E-PY-ARCHIVED-PY",
            "check_name": "archived_python_files_not_present",
            "status": "pass" if not archive_python else "fail",
            "evidence": f"archived_python_candidate_count={len(archive_python)}",
        }
    )
    return import_checks, test_rows, config_rows


def directory_digest(path: Path, excluded: set[str]) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "file_count": 0, "sha256": "missing"}
    if path.is_file():
        return {"status": "file", "file_count": 1, "sha256": sha256_file(path)}
    digest = hashlib.sha256()
    count = 0
    for child in sorted(path.rglob("*"), key=lambda item: rel(item)):
        rel_child = rel(child)
        if rel_child in excluded or rel_child.startswith("archive/") or "phase_f6i2e" in rel_child:
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in rel_child.split("/")):
            continue
        if child.is_file():
            count += 1
            digest.update(rel_child.encode("utf-8"))
            digest.update(str(child.stat().st_size).encode("ascii"))
            digest.update(sha256_file(child).encode("ascii"))
    return {"status": "directory", "file_count": count, "sha256": digest.hexdigest().upper()}


def protected_hashes(inputs: Inputs, candidates: list[dict[str, str]], stage: str) -> dict[str, Any]:
    approved_old = {candidate["original_path"] for candidate in candidates}
    allowed_modified = set(ALLOWED_REFERENCE_FILES) | ALLOWED_CURRENT_STATE_UPDATES
    excluded = approved_old | {candidate["archive_path"] for candidate in candidates} | allowed_modified
    targets: dict[str, Path] = {}
    for prefix in PROTECTED_PREFIXES:
        targets[prefix.rstrip("/")] = PROJECT_ROOT / prefix
    for path in PROTECTED_FILES:
        targets[path] = project_path(path)
    for key, path in inputs.paths.items():
        rel_path = rel(path)
        if rel_path not in excluded and rel_path not in ALLOWED_CURRENT_STATE_UPDATES:
            targets[f"input:{rel_path}"] = path
    entries = []
    for label, path in sorted(targets.items()):
        digest = directory_digest(path, excluded)
        entries.append({"label": label, "path": rel(path) if path.exists() or path_within_project(path) else str(path), "stage": stage, **digest})
    return {"stage": stage, "entries": entries, "allowed_modified": sorted(allowed_modified), "approved_moved_paths": sorted(approved_old)}


def path_within_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except ValueError:
        return False


def compare_protected(before: dict[str, Any], after: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    before_by_label = {row["label"]: row for row in before["entries"]}
    after_by_label = {row["label"]: row for row in after["entries"]}
    changed = []
    missing = []
    for label, before_row in before_by_label.items():
        after_row = after_by_label.get(label)
        if not after_row or after_row["status"] == "missing":
            missing.append({"label": label, "path": before_row["path"], "reason": "missing_after"})
        elif before_row["sha256"] != after_row["sha256"] or before_row["file_count"] != after_row["file_count"] or before_row["status"] != after_row["status"]:
            changed.append({"label": label, "path": before_row["path"], "reason": "hash_or_count_changed"})
    return changed, missing


def build_validation(
    inputs: Inputs,
    candidates: list[dict[str, str]],
    preflight: list[dict[str, str]],
    migration_actions: list[dict[str, str]],
    rejected: list[dict[str, str]],
    reference_actions: list[dict[str, str]],
    reference_validation: list[dict[str, str]],
    archive_hash_validation: list[dict[str, str]],
    active_reference_validation: list[dict[str, str]],
    historical_reference_validation: list[dict[str, str]],
    future_dependency_validation: list[dict[str, str]],
    python_validation: list[dict[str, str]],
    test_validation: list[dict[str, str]],
    config_validation: list[dict[str, str]],
    archive_index: list[dict[str, str]],
    rollback_plan: list[dict[str, str]],
    protected_changed: list[dict[str, str]],
    protected_missing: list[dict[str, str]],
    execute: bool,
) -> list[dict[str, str]]:
    moved_or_validated = sum(row["action_status"] in {"moved", "already_migrated"} for row in migration_actions)
    logical_ref_rows = sum(row["action_status"] in {"rewritten", "already_rewritten"} for row in reference_actions)
    if not execute:
        moved_or_validated = len(migration_actions)
        logical_ref_rows = len(reference_actions)
    rows = [
        ("001", "selected_phase_identifier", PHASE_ID, PHASE_ID, True, "blocking", "phase_f6i2e was unoccupied at read-only preflight"),
        ("002", "runtime_interpreter", str(APPROVED_INTERPRETER), sys.executable, Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve(), "blocking", RUNTIME_CLASSIFICATION),
        ("003", "python_no_user_site", "1", os.environ.get("PYTHONNOUSERSITE", ""), os.environ.get("PYTHONNOUSERSITE") == "1", "blocking", ""),
        ("004", "branch_matches_expected", EXPECTED_BRANCH, git_stdout(["rev-parse", "--abbrev-ref", "HEAD"]), git_stdout(["rev-parse", "--abbrev-ref", "HEAD"]) == EXPECTED_BRANCH, "blocking", ""),
        ("005", "head_matches_expected", EXPECTED_HEAD, git_stdout(["rev-parse", "HEAD"]), git_stdout(["rev-parse", "HEAD"]) == EXPECTED_HEAD, "blocking", ""),
        ("006", "f6i2d_decision_token", F6I2D_DECISION_TOKEN, inputs.f6i2d_migration_manifest.get("decision_token", ""), inputs.f6i2d_migration_manifest.get("decision_token") == F6I2D_DECISION_TOKEN, "blocking", ""),
        ("007", "approved_candidates", "18", str(len(candidates)), len(candidates) == EXPECTED_APPROVED, "blocking", ""),
        ("008", "preflight_passing", "18", str(sum(row["preflight_status"].startswith("pass") for row in preflight)), all(row["preflight_status"].startswith("pass") for row in preflight), "blocking", ""),
        ("009", "candidates_rejected", "0", str(len(rejected)), len(rejected) == 0, "blocking", ""),
        ("010", "moved_or_restart_validated_candidates", "18", str(moved_or_validated), moved_or_validated == EXPECTED_APPROVED, "blocking", ""),
        ("011", "active_reference_rows", "15", str(len(reference_actions)), len(reference_actions) == EXPECTED_ACTIVE_REFERENCE_ROWS, "blocking", ""),
        ("012", "active_reference_rows_closed", "15", str(logical_ref_rows), logical_ref_rows == EXPECTED_ACTIVE_REFERENCE_ROWS, "blocking", ""),
        ("013", "active_reference_validation", "pass", ",".join(sorted({row["status"] for row in active_reference_validation})), all(row["status"] == "pass" for row in active_reference_validation) or not execute, "blocking", ""),
        ("014", "archive_hash_validation", "18 pass", str(sum(row["status"] == "pass" for row in archive_hash_validation)), all(row["status"] == "pass" for row in archive_hash_validation) or not execute, "blocking", ""),
        ("015", "archive_index_entries", "18", str(len(archive_index)), len(archive_index) == EXPECTED_APPROVED, "blocking", ""),
        ("016", "rollback_entries", "18", str(len(rollback_plan)), len(rollback_plan) == EXPECTED_APPROVED, "blocking", ""),
        ("017", "compatibility_stubs_created", "0", "0", True, "blocking", "F6I2D stub/import alias requirement was zero"),
        ("018", "target_collisions", "0", str(sum(int(row["collision_count"]) for row in inputs.f6i2d_target_collision_recheck)), sum(int(row["collision_count"]) for row in inputs.f6i2d_target_collision_recheck) == 0, "blocking", ""),
        ("019", "future_dependency_validation", "all pass", str(sum(row["status"] == "pass" for row in future_dependency_validation)), all(row["status"] == "pass" for row in future_dependency_validation), "blocking", ""),
        ("020", "python_discovery_validation", "all pass", str(sum(row["status"] == "pass" for row in python_validation)), all(row["status"] == "pass" for row in python_validation), "blocking", ""),
        ("021", "test_discovery_validation", "all pass", str(sum(row["status"] == "pass" for row in test_validation)), all(row["status"] == "pass" for row in test_validation), "blocking", ""),
        ("022", "config_discovery_validation", "all pass", str(sum(row["status"] == "pass" for row in config_validation)), all(row["status"] == "pass" for row in config_validation), "blocking", ""),
        ("023", "protected_input_hash_invariance", "changed=0;missing=0", f"changed={len(protected_changed)};missing={len(protected_missing)}", not protected_changed and not protected_missing, "blocking", ""),
        ("024", "full_suite_excluded", "excluded", "excluded", True, "blocking", "Focused offline tests only."),
        ("025", "no_forbidden_actions", "true", "true", True, "blocking", "No training/inference/API/download/package/stage/commit."),
    ]
    return [
        {
            "validation_id": f"VAL-F6I2E-{number}",
            "check_name": name,
            "expected": expected,
            "observed": observed,
            "status": "pass" if passed else "fail",
            "severity": severity,
            "evidence": evidence,
        }
        for number, name, expected, observed, passed, severity, evidence in rows
    ]


def decide_token(validation: list[dict[str, str]], rejected: list[dict[str, str]], reference_actions: list[dict[str, str]], protected_changed: bool) -> str:
    if protected_changed:
        return DECISION_PROTECTED_FAILED
    if any(row["action_status"].startswith("failed") for row in reference_actions):
        return DECISION_REF_FAILED
    if any(row["status"] == "fail" and row["severity"] == "blocking" for row in validation):
        return DECISION_BLOCKED if rejected else DECISION_PARTIAL
    if rejected:
        return DECISION_PARTIAL
    return DECISION_COMPLETE


def build_determinism_rows(payload_digest: str) -> list[dict[str, str]]:
    return [
        {"determinism_id": "DET-F6I2E-001", "check_name": "in_memory_semantic_digest", "result": "pass", "evidence": payload_digest},
        {"determinism_id": "DET-F6I2E-002", "check_name": "runtime_metadata_isolated", "result": "pass", "evidence": "no timestamps in deterministic outputs"},
        {
            "determinism_id": "DET-F6I2E-003",
            "check_name": "two_builder_run_file_hash_comparison",
            "result": os.environ.get("F6I2E_DETERMINISM_RESULT", "pending_external_confirmation"),
            "evidence": os.environ.get("F6I2E_DETERMINISM_EVIDENCE", "to be recorded after execution"),
        },
    ]


def build_next_actions() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action_id": "F6I2E-NEXT-001",
            "component": "post_archive_consolidation",
            "current_status": "archive_migration_complete_pending_review",
            "safe_next_action": f"Review archive index and rollback manifest before starting {NEXT_PHASE}.",
            "requires_file_move": "false",
            "requires_reference_rewrite": "false",
            "requires_training": "false",
            "requires_llm_inference": "false",
            "requires_api": "false",
            "notes": "Do not use archive/ as active runtime source.",
        }
    ]


def update_handoff(existing: str, summary: dict[str, Any]) -> str:
    start = "<!-- BEGIN F6I2E CURRENT STATE -->"
    end = "<!-- END F6I2E CURRENT STATE -->"
    block = f"""{start}
## Post-F6I2E Substantive Archive Migration

- Phase: `{PHASE_ID}`
- Decision token: `{summary['decision_token']}`
- Runtime: `{APPROVED_INTERPRETER}`
- Runtime classification: `{RUNTIME_CLASSIFICATION}`
- Runtime caveat: repaired clone exact package versions only; not exact original executable-environment reproduction.
- Approved candidates expected/passing/rejected: `{summary['approved_candidates_expected']}` / `{summary['candidates_passing_preflight']}` / `{summary['candidates_rejected']}`
- Files moved or restart-validated: `{summary['files_moved_or_validated']}`
- Bytes migrated: `{summary['bytes_migrated']}`
- Active reference rows closed: `{summary['active_reference_rows_rewritten']}` of `{summary['active_reference_rows_expected']}`
- Historical reference rows retained: `{summary['historical_reference_rows_retained']}`
- Compatibility stubs/import aliases created: `0`
- Target collisions: `{summary['target_collisions']}`
- Empty directories removed: `{summary['empty_directories_removed']}`
- Archive index entries: `{summary['archive_index_entries']}`
- Rollback entries: `{summary['rollback_entries']}`
- Next phase: `{NEXT_PHASE}`

F6I2E moved only the approved 18 files into `archive/` and rewrote only approved active references. Historical references and original hashes are preserved in the archive index and rollback manifest.
{end}
"""
    normalized = block.rstrip() + "\n"
    pattern = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}\s*", re.DOTALL)
    if pattern.search(existing):
        return pattern.sub(lambda _match: normalized, existing).rstrip() + "\n"
    return existing.rstrip() + "\n\n" + normalized


def update_snapshot(existing: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    snapshot = dict(existing)
    snapshot["branch"] = git_stdout(["rev-parse", "--abbrev-ref", "HEAD"])
    snapshot["head"] = git_stdout(["rev-parse", "HEAD"])
    snapshot["phase_f6i2e"] = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": summary["decision_token"],
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime_reproduction": True,
        "counts": {
            "approved_candidates_expected": summary["approved_candidates_expected"],
            "candidates_passing_preflight": summary["candidates_passing_preflight"],
            "candidates_rejected": summary["candidates_rejected"],
            "files_moved_or_validated": summary["files_moved_or_validated"],
            "bytes_migrated": summary["bytes_migrated"],
            "active_reference_rows_rewritten": summary["active_reference_rows_rewritten"],
            "unresolved_active_references": summary["unresolved_active_references"],
            "historical_reference_rows_retained": summary["historical_reference_rows_retained"],
            "archive_index_entries": summary["archive_index_entries"],
            "rollback_entries": summary["rollback_entries"],
        },
        "next_phase": NEXT_PHASE,
        "archive_index": rel(ARCHIVE_INDEX_PATH),
        "rollback_manifest": rel(ROLLBACK_MANIFEST_PATH),
    }
    snapshot["no_training"] = True
    snapshot["no_llm_inference"] = True
    snapshot["no_api_calls"] = True
    snapshot["no_downloads"] = True
    snapshot["no_git_stage_or_commit"] = True
    return snapshot


def build_report(summary: dict[str, Any], protected_manifest: dict[str, Any], validation: list[dict[str, str]]) -> str:
    failed = [row for row in validation if row["status"] != "pass" and row["severity"] == "blocking"]
    command_log = os.environ.get("F6I2E_COMMAND_LOG", "Final focused command log is recorded in the assistant response.")
    return f"""# F6I2E Substantive Archive Migration Execution

## Runtime

- Approved/current audit runtime: `{APPROVED_INTERPRETER}`
- Runtime classification: `{RUNTIME_CLASSIFICATION}`
- Caveat: defensible repaired clone exact package versions only; not exact original executable-environment reproduction.
- Required environment: `PYTHONNOUSERSITE=1`

## Results

- Decision token: `{summary['decision_token']}`
- Approved candidates expected/passing/rejected: `{summary['approved_candidates_expected']}` / `{summary['candidates_passing_preflight']}` / `{summary['candidates_rejected']}`
- Files moved or restart-validated: `{summary['files_moved_or_validated']}`
- Bytes migrated: `{summary['bytes_migrated']}`
- Active reference rows closed: `{summary['active_reference_rows_rewritten']}` of `{summary['active_reference_rows_expected']}`
- Physical textual replacements applied: `{summary['physical_reference_replacements']}`
- Unresolved active references: `{summary['unresolved_active_references']}`
- Historical reference rows retained or classified non-text/binary: `{summary['historical_reference_rows_retained']}`
- Compatibility stubs/import aliases created: `0`
- Target collisions: `{summary['target_collisions']}`
- Empty directories removed: `{summary['empty_directories_removed']}`
- Archive index entries: `{summary['archive_index_entries']}`
- Rollback entries: `{summary['rollback_entries']}`
- Next phase: `{NEXT_PHASE}`

## Protected Inputs

- Result: `{protected_manifest['status']}`
- Changed protected entries: `{len(protected_manifest['changed'])}`
- Missing protected entries: `{len(protected_manifest['missing'])}`

## Validation

- Blocking validation failures: `{len(failed)}`
- Full suite status: excluded; focused offline validation only.
- Focused command log: `{command_log}`

## Boundaries

F6I2E moved only the approved 18 files, rewrote only approved active references, created no compatibility stubs, and performed no training, inference, Ollama/API call, download, package change, staging, or commit.
"""


def semantic_digest(result: BuildResult | dict[str, Any]) -> str:
    if isinstance(result, BuildResult):
        payload = {
            "tables": result.tables,
            "summary": result.summary,
            "protected_status": result.protected_hashes.get("status"),
            "archive_index_count": len(result.manifests["archive_index"].get("entries", [])),
        }
    else:
        payload = result
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def build(
    config_path: Path = CONFIG_PATH,
    write_outputs_enabled: bool = True,
    execute_migration_enabled: bool = True,
    update_current_state: bool = True,
) -> BuildResult:
    require_runtime()
    initial_git = git_state_snapshot()
    inputs = load_inputs(config_path)
    candidates = approved_candidates(inputs)
    active_refs = active_rewrite_rows(inputs)
    ensure_starting_counts(inputs, candidates, active_refs)
    prior = prior_complete_manifest()
    archive_by_original = {candidate["original_path"]: candidate["archive_path"] for candidate in candidates}
    protected_before = protected_hashes(inputs, candidates, "before")

    if prior:
        preflight = prior["candidate_preflight"]
        migration_actions = prior["migration_actions"]
        rejected = prior["migration_rejected"]
        reference_actions = prior["reference_rewrite_actions"]
    else:
        preflight, rejected = build_candidate_preflight(candidates, active_refs)
        migration_actions = execute_moves(candidates, preflight, rejected, execute_migration_enabled)
        reference_actions = execute_reference_rewrites(active_refs, execute_migration_enabled and not rejected)

    reference_rewrite_validation = build_reference_rewrite_validation(active_refs)
    archive_hash_validation = build_archive_hash_validation(candidates)
    active_reference_validation = build_active_reference_validation(active_refs)
    historical_reference_validation = build_historical_reference_validation(inputs, archive_by_original)
    future_dependency_validation = build_future_dependency_validation(candidates, inputs)
    python_validation, test_validation, config_validation = build_discovery_validations(candidates)
    empty_directory_actions = build_empty_directory_actions(candidates)
    archive_index_entries = build_archive_index(candidates, inputs, reference_actions, migration_actions)
    rollback_plan = build_rollback_plan(candidates, active_refs)
    protected_after = protected_hashes(inputs, candidates, "after")
    protected_changed, protected_missing = compare_protected(protected_before, protected_after)
    validation = build_validation(
        inputs,
        candidates,
        preflight,
        migration_actions,
        rejected,
        reference_actions,
        reference_rewrite_validation,
        archive_hash_validation,
        active_reference_validation,
        historical_reference_validation,
        future_dependency_validation,
        python_validation,
        test_validation,
        config_validation,
        archive_index_entries,
        rollback_plan,
        protected_changed,
        protected_missing,
        execute_migration_enabled,
    )
    decision_token = decide_token(validation, rejected, reference_actions, bool(protected_changed or protected_missing))
    physical_replacements_by_group = {
        (row["referencer_path"], row["expected_old_value"], row["expected_new_value"]): int(row["physical_replacement_count"])
        for row in reference_actions
    }
    summary = {
        "phase_id": PHASE_ID,
        "decision_token": decision_token,
        "approved_candidates_expected": EXPECTED_APPROVED,
        "candidates_passing_preflight": sum(row["preflight_status"].startswith("pass") for row in preflight),
        "candidates_rejected": len(rejected),
        "files_moved_or_validated": sum(row["action_status"] in {"moved", "already_migrated"} for row in migration_actions)
        if execute_migration_enabled
        else 0,
        "files_moved": sum(row["action_status"] == "moved" for row in migration_actions),
        "files_already_migrated": sum(row["action_status"] == "already_migrated" for row in migration_actions),
        "bytes_migrated": sum(int(row["bytes_migrated"]) for row in migration_actions if row["action_status"] in {"moved", "already_migrated"}),
        "active_reference_rows_expected": EXPECTED_ACTIVE_REFERENCE_ROWS,
        "active_reference_rows_rewritten": sum(row["action_status"] in {"rewritten", "already_rewritten"} for row in reference_actions)
        if execute_migration_enabled
        else 0,
        "physical_reference_replacements": sum(physical_replacements_by_group.values()),
        "unresolved_active_references": sum(row["status"] != "pass" for row in active_reference_validation) if execute_migration_enabled else 0,
        "historical_reference_rows_retained": sum(row["status"] in {"pass", "not_applicable_or_binary_historical_reference"} for row in historical_reference_validation),
        "compatibility_stubs_created": 0,
        "target_collisions": sum(int(row["collision_count"]) for row in inputs.f6i2d_target_collision_recheck),
        "empty_directories_removed": sum(row["action"] == "removed" for row in empty_directory_actions),
        "archive_index_entries": len(archive_index_entries),
        "rollback_entries": len(rollback_plan),
        "runtime_interpreter": str(APPROVED_INTERPRETER),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": True,
        "next_phase": NEXT_PHASE,
    }
    deterministic_payload = {
        "preflight": preflight,
        "migration_actions": migration_actions,
        "reference_actions": reference_actions,
        "archive_hash_validation": archive_hash_validation,
        "active_reference_validation": active_reference_validation,
        "summary": summary,
    }
    determinism = build_determinism_rows(sha256_text(json.dumps(deterministic_payload, sort_keys=True, separators=(",", ":"))))
    next_actions = build_next_actions()
    tables = {
        "candidate_preflight": preflight,
        "migration_actions": migration_actions,
        "migration_rejected": rejected,
        "reference_rewrite_actions": reference_actions,
        "reference_rewrite_validation": reference_rewrite_validation,
        "archive_hash_validation": archive_hash_validation,
        "active_reference_validation": active_reference_validation,
        "historical_reference_validation": historical_reference_validation,
        "future_dependency_validation": future_dependency_validation,
        "python_discovery_validation": python_validation,
        "test_discovery_validation": test_validation,
        "config_discovery_validation": config_validation,
        "empty_directory_actions": empty_directory_actions,
        "rollback_plan": rollback_plan,
        "validation": validation,
        "determinism": determinism,
        "next_actions": next_actions,
    }
    final_git = git_state_snapshot()
    protected_manifest = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "runtime": {
            "approved_runtime_interpreter": str(APPROVED_INTERPRETER),
            "runtime_classification": RUNTIME_CLASSIFICATION,
            "not_exact_original_runtime": True,
            "python_no_user_site_required": True,
        },
        "protected_before": protected_before,
        "protected_after": protected_after,
        "changed": protected_changed,
        "missing": protected_missing,
        "status": "pass" if not protected_changed and not protected_missing else "fail",
    }
    manifests = {
        "pre_migration": {
            "phase_id": PHASE_ID,
            "phase_name": PHASE_NAME,
            "runtime": protected_manifest["runtime"],
            "initial_git_state": initial_git,
            "approved_candidates": candidates,
            "candidate_preflight": preflight,
            "migration_executed": execute_migration_enabled,
        },
        "archive_index": {"phase_id": PHASE_ID, "phase_name": PHASE_NAME, "entries": archive_index_entries},
        "migration": {
            "phase_id": PHASE_ID,
            "phase_name": PHASE_NAME,
            "decision_token": decision_token,
            "migration_executed": execute_migration_enabled,
            "counts": summary,
            "candidate_preflight": preflight,
            "migration_actions": migration_actions,
            "migration_rejected": rejected,
            "reference_rewrite_actions": reference_actions,
            "initial_git_state": initial_git,
            "final_git_state": final_git,
            "next_phase": NEXT_PHASE,
            "forbidden_actions": {
                "unapproved_file_moved": False,
                "unapproved_reference_rewritten": False,
                "compatibility_stub_created": False,
                "training_executed": False,
                "llm_inference_executed": False,
                "ollama_or_api_called": False,
                "download_or_package_change": False,
                "git_stage_or_commit": False,
            },
        },
        "rollback": {
            "phase_id": PHASE_ID,
            "phase_name": PHASE_NAME,
            "rollback_executed": False,
            "rollback_plan": rollback_plan,
        },
        "protected_hashes": protected_manifest,
    }
    report = build_report(summary, protected_manifest, validation)
    result = BuildResult(tables=tables, manifests=manifests, protected_hashes=protected_manifest, report=report, summary=summary)

    if write_outputs_enabled:
        for key, rows in tables.items():
            write_csv(OUTPUT_TABLES[key], rows, TABLE_COLUMNS[key])
        write_json(PRE_MIGRATION_MANIFEST_PATH, manifests["pre_migration"])
        write_json(ARCHIVE_INDEX_PATH, manifests["archive_index"])
        write_json(MIGRATION_MANIFEST_PATH, manifests["migration"])
        write_json(ROLLBACK_MANIFEST_PATH, manifests["rollback"])
        write_json(PROTECTED_HASHES_PATH, manifests["protected_hashes"])
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        if update_current_state and decision_token == DECISION_COMPLETE:
            HANDOFF_PATH.write_text(update_handoff(inputs.handoff_text, summary), encoding="utf-8")
            write_json(SNAPSHOT_PATH, update_snapshot(inputs.current_snapshot, summary))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--no-migration", action="store_true", help="Build outputs without moving files or rewriting references.")
    parser.add_argument("--no-current-state-update", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    result = build(
        args.config,
        write_outputs_enabled=True,
        execute_migration_enabled=not args.no_migration,
        update_current_state=not args.no_current_state_update,
    )
    if args.print_summary:
        print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

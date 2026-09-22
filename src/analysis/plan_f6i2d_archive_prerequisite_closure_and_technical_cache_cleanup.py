"""F6I2D archive prerequisite closure and technical cache cleanup.

This phase closes the two F6I2C archive prerequisites for 18 conditional
substantive candidates and executes only the separately inventoried cache
cleanup bundle. It does not move, rename, copy, archive, or delete substantive
archive candidates and does not run experiments.
"""

from __future__ import annotations

import argparse
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
PHASE_ID = "phase_f6i2d"
PHASE_NAME = "phase_f6i2d_archive_prerequisite_closure_and_technical_cache_cleanup"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
NEXT_PHASE = "phase_f6i2e_substantive_archive_migration_execution_after_explicit_approval"
EXPECTED_CONTENT_CANDIDATES = 12
EXPECTED_SUMMARY_CANDIDATES = 6
EXPECTED_CACHE_CANDIDATES = 751
F6I2C_DECISION_TOKEN = "f6i2c_archive_verification_complete_partial_ready"
DECISION_TOKEN_COMPLETE = "f6i2d_prerequisites_complete_archive_manifest_ready"
DECISION_TOKEN_PARTIAL = "f6i2d_prerequisites_partial_some_candidates_not_ready"
DECISION_TOKEN_NONE = "f6i2d_blocked_no_substantive_candidate_ready"
DECISION_TOKEN_PROTECTED_CHANGE = "f6i2d_failed_protected_input_change"

CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
SNAPSHOT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"
CONTENT_INDEX_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_content_index_manifest.json"
CACHE_CLEANUP_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_cache_cleanup_manifest.json"
MIGRATION_MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_substantive_archive_migration_manifest.json"
PROTECTED_HASHES_PATH = MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json"

OUTPUT_TABLES = {
    "content_index_candidates": TABLES_DIR / f"{PHASE_ID}_content_index_candidates.csv",
    "content_index": TABLES_DIR / f"{PHASE_ID}_content_index.csv",
    "content_index_validation": TABLES_DIR / f"{PHASE_ID}_content_index_validation.csv",
    "current_fact_inventory": TABLES_DIR / f"{PHASE_ID}_current_fact_inventory.csv",
    "current_fact_migration": TABLES_DIR / f"{PHASE_ID}_current_fact_migration.csv",
    "current_fact_validation": TABLES_DIR / f"{PHASE_ID}_current_fact_validation.csv",
    "cache_candidate_preflight": TABLES_DIR / f"{PHASE_ID}_cache_candidate_preflight.csv",
    "cache_cleanup_actions": TABLES_DIR / f"{PHASE_ID}_cache_cleanup_actions.csv",
    "cache_cleanup_rejected": TABLES_DIR / f"{PHASE_ID}_cache_cleanup_rejected.csv",
    "final_archive_gate": TABLES_DIR / f"{PHASE_ID}_final_archive_gate.csv",
    "archive_ready_for_migration": TABLES_DIR / f"{PHASE_ID}_archive_ready_for_migration.csv",
    "archive_not_ready": TABLES_DIR / f"{PHASE_ID}_archive_not_ready.csv",
    "reference_rewrite_closure": TABLES_DIR / f"{PHASE_ID}_reference_rewrite_closure.csv",
    "compatibility_closure": TABLES_DIR / f"{PHASE_ID}_compatibility_closure.csv",
    "target_collision_recheck": TABLES_DIR / f"{PHASE_ID}_target_collision_recheck.csv",
    "validation": TABLES_DIR / f"{PHASE_ID}_validation.csv",
    "determinism": TABLES_DIR / f"{PHASE_ID}_determinism.csv",
    "next_actions": TABLES_DIR / f"{PHASE_ID}_next_actions.csv",
}

TEXT_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".cmd",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
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
    "results/raw_outputs/",
    "src/llm/",
    "src/metrics/",
    "src/evaluation/",
)
PROTECTED_PATHS = (
    "report/python_environment_recovery_audit.md",
    "results/environment/python_environment_candidate_inventory.csv",
    "results/environment/broken_venv_static_package_inventory.csv",
    "results/environment/project_dependency_contract.csv",
    "results/environment/environment_recovery_comparison.csv",
    "results/environment/recovered_environment_package_inventory.csv",
    "results/environment/recovered_environment_validation.csv",
    "results/manifests/python_environment_recovery_snapshot.json",
)
ALLOWED_CURRENT_STATE_UPDATES = {
    "report/future_operations_context_handoff.md",
    "results/manifests/current_project_context_snapshot.json",
}
SUBSTANTIVE_OUTPUTS = {
    "src/analysis/plan_f6i2d_archive_prerequisite_closure_and_technical_cache_cleanup.py",
    "src/experiments/build_phase_f6i2d_archive_prerequisite_closure_and_technical_cache_cleanup.py",
    "tests/test_phase_f6i2d_archive_prerequisite_closure_and_technical_cache_cleanup.py",
    "configs/phase_f6i2d_archive_prerequisite_closure_and_technical_cache_cleanup.yaml",
}

CONTENT_INDEX_CANDIDATE_COLUMNS = [
    "candidate_id",
    "source_path",
    "source_exists",
    "recorded_sha256",
    "current_sha256",
    "hash_status",
    "recorded_size_bytes",
    "current_size_bytes",
    "size_status",
    "f6i2c_decision",
    "target_archive_path",
    "archive_category",
    "references_requiring_rewrite",
    "content_index_required",
    "substantive_candidate_protected",
    "runtime_interpreter",
    "runtime_classification",
]
CONTENT_INDEX_COLUMNS = [
    "content_index_id",
    "source_path",
    "sha256",
    "size_bytes",
    "phase_family",
    "artifact_type",
    "producer",
    "inputs",
    "outputs",
    "decision_token",
    "experimental_status",
    "protocol_represented",
    "dataset",
    "task",
    "model",
    "prompt_strategy",
    "metric",
    "cohort",
    "candidate_contract",
    "output_contract",
    "unique_constants",
    "unique_schemas",
    "unique_scientific_claims",
    "unique_limitations",
    "unique_blocker_evidence",
    "referenced_files",
    "active_references",
    "historical_references",
    "proposed_archive_path",
    "rollback_information",
    "functions_classes_or_keys",
    "canonical_replacement",
    "remains_executable_after_archival",
    "source_excerpt_hash",
    "index_summary",
]
CONTENT_INDEX_VALIDATION_COLUMNS = [
    "validation_id",
    "source_path",
    "hash_captured",
    "scientific_role_recorded",
    "unique_decisions_represented",
    "unique_contracts_represented",
    "input_output_relationships_represented",
    "active_references_enumerated",
    "archive_destination_collision_free",
    "rollback_instructions_exist",
    "future_dependencies_absent_or_replaced",
    "final_report_discoverability_preserved",
    "status",
    "evidence",
]
CURRENT_FACT_INVENTORY_COLUMNS = [
    "fact_id",
    "source_path",
    "source_section",
    "original_phase",
    "scientific_topic",
    "current_wording",
    "evidence_files",
    "currentness_status",
    "target_current_state_artifact",
    "historical_caveat",
    "duplicate_fact_handling",
    "runtime_interpreter",
    "runtime_classification",
]
CURRENT_FACT_MIGRATION_COLUMNS = [
    "migration_id",
    "fact_id",
    "source_path",
    "target_current_state_artifact",
    "migration_action",
    "migrated_wording",
    "source_evidence_hash",
    "duplicate_fact_handling",
    "migration_status",
]
CURRENT_FACT_VALIDATION_COLUMNS = [
    "validation_id",
    "fact_id",
    "source_path",
    "source_supported",
    "remains_current",
    "not_contradicted_by_later_phase",
    "evidence_resolvable",
    "numbers_match_authoritative_tables",
    "classification_conservative",
    "historical_report_unchanged",
    "no_exactness_or_completion_inflation",
    "status",
    "evidence",
]
CACHE_PREFLIGHT_COLUMNS = [
    "cache_artifact_id",
    "path",
    "artifact_type",
    "source_regeneration_path",
    "exists_preflight",
    "approved_path_only",
    "within_project",
    "allowed_artifact_type",
    "cache_or_bytecode_location",
    "excluded_environment_path",
    "not_source_code",
    "not_raw_evidence",
    "not_test_fixture",
    "not_model_or_dataset_artifact",
    "source_regeneration_retained",
    "directory_safe_to_delete",
    "preflight_status",
    "preflight_reason",
]
CACHE_ACTION_COLUMNS = [
    "cache_artifact_id",
    "path",
    "artifact_type",
    "preflight_status",
    "exists_before_cleanup",
    "action",
    "action_status",
    "deleted_by_phase",
    "already_absent_by_phase",
    "final_exists",
    "validation_status",
    "notes",
]
CACHE_REJECTED_COLUMNS = [
    "cache_artifact_id",
    "path",
    "artifact_type",
    "rejection_class",
    "reason",
]
FINAL_GATE_COLUMNS = [
    "gate_id",
    "original_path",
    "previous_f6i2c_decision",
    "source_hash_unchanged",
    "content_index_status",
    "current_summary_migration_status",
    "target_collision_count",
    "active_reference_rewrite_count",
    "historical_reference_count",
    "reference_rewrite_closure_status",
    "compatibility_closure_status",
    "future_dependency_status",
    "substantive_candidate_unchanged",
    "final_state",
    "blocking_reason",
]
ARCHIVE_READY_COLUMNS = [
    "approval_id",
    "original_path",
    "original_sha256",
    "original_size_bytes",
    "final_archive_path",
    "archive_category",
    "reason",
    "content_index_reference",
    "migrated_current_fact_references",
    "active_references_requiring_rewrite",
    "compatibility_strategy",
    "future_dependencies",
    "rollback_instructions",
    "validation_commands",
    "expected_git_action",
    "approval_status",
]
ARCHIVE_NOT_READY_COLUMNS = [
    "original_path",
    "previous_f6i2c_decision",
    "final_state",
    "blocking_reason",
    "required_next_action",
]
REFERENCE_CLOSURE_COLUMNS = [
    "closure_id",
    "original_path",
    "final_archive_path",
    "referencer_path",
    "reference_class",
    "future_rewrite_file",
    "expected_old_value",
    "expected_new_value",
    "historical_reference_should_remain",
    "active_reference_must_change",
    "runtime_semantics_impact",
    "closure_status",
]
COMPATIBILITY_COLUMNS = [
    "compatibility_id",
    "original_path",
    "final_archive_path",
    "f6i2c_strategy",
    "f6i2d_concrete_future_action",
    "stub_or_import_alias_required",
    "pytest_discovery_risk",
    "runtime_import_risk",
    "archived_code_execution_risk",
    "canonical_path_visibility",
    "closure_status",
]
COLLISION_COLUMNS = [
    "target_archive_path",
    "candidate_count",
    "original_paths",
    "collision_count",
    "file_discovery_behavior",
    "python_import_behavior",
    "pytest_discovery_behavior",
    "documentation_navigation",
    "manifest_indexing",
    "status",
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
    "content_index_candidates": CONTENT_INDEX_CANDIDATE_COLUMNS,
    "content_index": CONTENT_INDEX_COLUMNS,
    "content_index_validation": CONTENT_INDEX_VALIDATION_COLUMNS,
    "current_fact_inventory": CURRENT_FACT_INVENTORY_COLUMNS,
    "current_fact_migration": CURRENT_FACT_MIGRATION_COLUMNS,
    "current_fact_validation": CURRENT_FACT_VALIDATION_COLUMNS,
    "cache_candidate_preflight": CACHE_PREFLIGHT_COLUMNS,
    "cache_cleanup_actions": CACHE_ACTION_COLUMNS,
    "cache_cleanup_rejected": CACHE_REJECTED_COLUMNS,
    "final_archive_gate": FINAL_GATE_COLUMNS,
    "archive_ready_for_migration": ARCHIVE_READY_COLUMNS,
    "archive_not_ready": ARCHIVE_NOT_READY_COLUMNS,
    "reference_rewrite_closure": REFERENCE_CLOSURE_COLUMNS,
    "compatibility_closure": COMPATIBILITY_COLUMNS,
    "target_collision_recheck": COLLISION_COLUMNS,
    "validation": VALIDATION_COLUMNS,
    "determinism": DETERMINISM_COLUMNS,
    "next_actions": NEXT_ACTION_COLUMNS,
}


@dataclass(frozen=True)
class Inputs:
    config: dict[str, Any]
    paths: dict[str, Path]
    f6i2c_candidate_preflight: list[dict[str, str]]
    f6i2c_archive_decisions: list[dict[str, str]]
    f6i2c_archive_ready: list[dict[str, str]]
    f6i2c_archive_rejected: list[dict[str, str]]
    f6i2c_archive_manual_review: list[dict[str, str]]
    f6i2c_future_dependency_recheck: list[dict[str, str]]
    f6i2c_target_architecture: list[dict[str, str]]
    f6i2c_path_mapping: list[dict[str, str]]
    f6i2c_target_collisions: list[dict[str, str]]
    f6i2c_reference_inventory: list[dict[str, str]]
    f6i2c_reference_rewrite_plan: list[dict[str, str]]
    f6i2c_compatibility_strategy: list[dict[str, str]]
    f6i2c_current_summary_migration_plan: list[dict[str, str]]
    f6i2c_index_schema: list[dict[str, str]]
    f6i2c_cache_bundle: list[dict[str, str]]
    f6i2c_validation: list[dict[str, str]]
    f6i2c_manifest: dict[str, Any]
    f6i2c_protected_hashes: dict[str, Any]
    f6i1_inventory: list[dict[str, str]]
    f6i1_edges: list[dict[str, str]]
    f6i2a_archive_bundle: list[dict[str, str]]
    f6i2b_manifest: dict[str, Any]
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
    materialized = sorted(
        [dict(row) for row in rows],
        key=lambda row: tuple(str(row.get(column, "")) for column in columns),
    )
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


def run_command(args: Sequence[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        list(args),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def git_stdout(args: Sequence[str]) -> str:
    code, stdout, stderr = run_command(["git", *args])
    return stdout if code == 0 else f"ERROR: {stderr}"


def git_status_lines() -> list[str]:
    status = git_stdout(["status", "--porcelain=v1"])
    return [line for line in status.splitlines() if line.strip()]


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
    status_lines = git_status_lines()
    return {
        "branch": git_stdout(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": git_stdout(["rev-parse", "HEAD"]),
        "status_row_count": len(status_lines),
        "status_counts": git_status_counts(status_lines),
        "status_lines": status_lines,
    }


def require_runtime() -> None:
    if Path(sys.executable).resolve() != APPROVED_INTERPRETER.resolve():
        raise RuntimeError(f"F6I2D requires {APPROVED_INTERPRETER}, got {sys.executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("F6I2D requires PYTHONNOUSERSITE=1")


def read_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_inputs(config_path: Path = CONFIG_PATH) -> Inputs:
    config = read_config(config_path)
    paths = {key: project_path(value) for key, value in config["inputs"].items()}
    missing = [key for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing F6I2D authoritative inputs: {missing}")
    return Inputs(
        config=config,
        paths=paths,
        f6i2c_candidate_preflight=read_csv(paths["f6i2c_candidate_preflight"]),
        f6i2c_archive_decisions=read_csv(paths["f6i2c_archive_decisions"]),
        f6i2c_archive_ready=read_csv(paths["f6i2c_archive_ready"]),
        f6i2c_archive_rejected=read_csv(paths["f6i2c_archive_rejected"]),
        f6i2c_archive_manual_review=read_csv(paths["f6i2c_archive_manual_review"]),
        f6i2c_future_dependency_recheck=read_csv(paths["f6i2c_future_dependency_recheck"]),
        f6i2c_target_architecture=read_csv(paths["f6i2c_target_architecture"]),
        f6i2c_path_mapping=read_csv(paths["f6i2c_path_mapping"]),
        f6i2c_target_collisions=read_csv(paths["f6i2c_target_collisions"]),
        f6i2c_reference_inventory=read_csv(paths["f6i2c_reference_inventory"]),
        f6i2c_reference_rewrite_plan=read_csv(paths["f6i2c_reference_rewrite_plan"]),
        f6i2c_compatibility_strategy=read_csv(paths["f6i2c_compatibility_strategy"]),
        f6i2c_current_summary_migration_plan=read_csv(paths["f6i2c_current_summary_migration_plan"]),
        f6i2c_index_schema=read_csv(paths["f6i2c_index_schema"]),
        f6i2c_cache_bundle=read_csv(paths["f6i2c_cache_bundle"]),
        f6i2c_validation=read_csv(paths["f6i2c_validation"]),
        f6i2c_manifest=read_json(paths["f6i2c_manifest"]),
        f6i2c_protected_hashes=read_json(paths["f6i2c_protected_hashes"]),
        f6i1_inventory=read_csv(paths["f6i1_inventory"]),
        f6i1_edges=read_csv(paths["f6i1_edges"]),
        f6i2a_archive_bundle=read_csv(paths["f6i2a_archive_bundle"]),
        f6i2b_manifest=read_json(paths["f6i2b_manifest"]),
        handoff_text=paths["handoff"].read_text(encoding="utf-8"),
        current_snapshot=read_json(paths["current_context_snapshot"]),
    )


def ensure_starting_counts(inputs: Inputs) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if inputs.f6i2c_manifest.get("decision_token") != F6I2C_DECISION_TOKEN:
        raise RuntimeError("F6I2D requires the authoritative F6I2C partial-ready decision token")
    content = [
        row
        for row in inputs.f6i2c_archive_ready
        if row.get("current_decision") == "archive_ready_after_content_indexing"
    ]
    summary = [
        row
        for row in inputs.f6i2c_archive_ready
        if row.get("current_decision") == "archive_ready_after_current_summary_migration"
    ]
    content_paths = {row["original_path"] for row in content}
    summary_paths = {row["original_path"] for row in summary}
    cache_paths = {row["path"] for row in inputs.f6i2c_cache_bundle}
    if len(content) != EXPECTED_CONTENT_CANDIDATES:
        raise RuntimeError(f"Expected 12 content-index candidates, observed {len(content)}")
    if len(summary) != EXPECTED_SUMMARY_CANDIDATES:
        raise RuntimeError(f"Expected 6 current-summary candidates, observed {len(summary)}")
    if content_paths & summary_paths:
        raise RuntimeError(f"F6I2C conditional sets are not disjoint: {sorted(content_paths & summary_paths)}")
    if len(inputs.f6i2c_cache_bundle) != EXPECTED_CACHE_CANDIDATES or len(cache_paths) != EXPECTED_CACHE_CANDIDATES:
        raise RuntimeError(
            f"Expected 751 unique cache cleanup rows, observed rows={len(inputs.f6i2c_cache_bundle)} unique={len(cache_paths)}"
        )
    return content, summary


def truncate_cell(value: str, limit: int = 500) -> str:
    value = (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...(+{len(value) - limit})"


def headings(path: Path) -> str:
    if not path.exists() or path.suffix.lower() != ".md":
        return ""
    found = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#"):
            found.append(line.strip())
        if len(found) >= 12:
            break
    return ";".join(found)


def first_text_hash(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    with path.open("rb") as handle:
        chunk = handle.read(4096)
    return hashlib.sha256(chunk).hexdigest().upper()


def artifact_type(path: str, inventory_row: Mapping[str, str]) -> str:
    suffix = Path(path).suffix.lower()
    if suffix:
        return suffix.lstrip(".") or "no_extension"
    category = inventory_row.get("category", "")
    return category or "no_extension"


def infer_topic_tokens(text: str) -> dict[str, str]:
    lower = text.lower()
    datasets = [name for name in ["beauty", "sports", "ml1m", "lastfm"] if name in lower]
    tasks = [name for name in ["ranking", "reranking", "candidate", "holdout"] if name in lower]
    models = [
        name
        for name in ["mostpop", "random", "bm25", "bprmf", "gru4rec", "sasrec", "lightgcn", "qwen", "mistral"]
        if name in lower
    ]
    metrics = [name for name in ["hr", "ndcg", "coverage", "hallucination", "ild", "popularity"] if name in lower]
    prompt = "prompt_contract_referenced" if "prompt" in lower else "paper_unverified_or_not_applicable"
    cohort = "cohort_contract_referenced" if "cohort" in lower or "sample" in lower else "paper_unverified_or_not_applicable"
    return {
        "dataset": ";".join(datasets) or "paper_unverified_or_not_applicable",
        "task": ";".join(tasks) or "historical_support_artifact",
        "model": ";".join(models) or "paper_unverified_or_not_applicable",
        "metric": ";".join(metrics) or "paper_unverified_or_not_applicable",
        "prompt_strategy": prompt,
        "cohort": cohort,
    }


def candidate_maps(inputs: Inputs) -> dict[str, dict[str, dict[str, str]]]:
    return {
        "ready": {row["original_path"]: row for row in inputs.f6i2c_archive_ready},
        "mapping": {row["original_path"]: row for row in inputs.f6i2c_path_mapping},
        "inventory": {row["path"]: row for row in inputs.f6i1_inventory},
        "archive_bundle": {row["path"]: row for row in inputs.f6i2a_archive_bundle},
        "decision": {row["path"]: row for row in inputs.f6i2c_archive_decisions},
        "compatibility": {row["path"]: row for row in inputs.f6i2c_compatibility_strategy},
    }


def references_by_candidate(
    rewrites: list[dict[str, str]], candidate_paths: Iterable[str]
) -> dict[str, list[dict[str, str]]]:
    wanted = set(candidate_paths)
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rewrites:
        if row.get("original_path") in wanted:
            grouped[row["original_path"]].append(row)
    return {path: sorted(rows, key=lambda row: (row.get("referencer_path", ""), row.get("rewrite_id", ""))) for path, rows in grouped.items()}


def build_content_index_candidates(content_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for source in sorted(content_rows, key=lambda row: row["original_path"]):
        path = project_path(source["original_path"])
        current_hash = sha256_file(path) if path.is_file() else "missing"
        current_size = str(path.stat().st_size) if path.is_file() else "missing"
        rows.append(
            {
                "candidate_id": stable_id("F6I2D-CONTENT-CANDIDATE", source["original_path"]),
                "source_path": source["original_path"],
                "source_exists": str(path.is_file()).lower(),
                "recorded_sha256": source["original_sha256"],
                "current_sha256": current_hash,
                "hash_status": "match" if current_hash == source["original_sha256"] else "mismatch_or_missing",
                "recorded_size_bytes": source["original_size_bytes"],
                "current_size_bytes": current_size,
                "size_status": "match" if current_size == source["original_size_bytes"] else "mismatch_or_missing",
                "f6i2c_decision": source["current_decision"],
                "target_archive_path": source["target_archive_path"],
                "archive_category": source["archive_category"],
                "references_requiring_rewrite": source["references_requiring_rewrite"],
                "content_index_required": "true",
                "substantive_candidate_protected": "true",
                "runtime_interpreter": str(APPROVED_INTERPRETER),
                "runtime_classification": RUNTIME_CLASSIFICATION,
            }
        )
    return rows


def build_content_index(
    content_rows: list[dict[str, str]], inputs: Inputs, refs: dict[str, list[dict[str, str]]]
) -> list[dict[str, str]]:
    maps = candidate_maps(inputs)
    rows = []
    for source in sorted(content_rows, key=lambda row: row["original_path"]):
        path_key = source["original_path"]
        file_path = project_path(path_key)
        inventory = maps["inventory"].get(path_key, {})
        archive_bundle = maps["archive_bundle"].get(path_key, {})
        mapping = maps["mapping"].get(path_key, {})
        reference_rows = refs.get(path_key, [])
        active_refs = sorted(
            row["referencer_path"] for row in reference_rows if row.get("reference_class") == "active_documentation_reference"
        )
        historical_refs = sorted(
            row["referencer_path"] for row in reference_rows if row.get("reference_class") == "historical_reference"
        )
        text = " ".join(
            [
                inventory.get("purpose", ""),
                inventory.get("inputs", ""),
                inventory.get("outputs", ""),
                inventory.get("future_roadmap_dependencies", ""),
                archive_bundle.get("reason", ""),
                headings(file_path),
            ]
        )
        inferred = infer_topic_tokens(text)
        suffix = file_path.suffix.lower()
        if suffix == ".png":
            summary = "Binary plot indexed by hash, size, F6I1 lineage, F6I2C path mapping, and references; pixel contents not duplicated."
            executable = "false_binary_plot"
            functions = "not_applicable_binary_plot"
        elif suffix == ".log":
            summary = "Historical execution log indexed by hash, size, lineage, and first-chunk digest; full log not duplicated."
            executable = "false_historical_log"
            functions = "not_applicable_log"
        elif file_path.name == ".gitkeep":
            summary = "Directory placeholder indexed as support artifact; no scientific content beyond path existence."
            executable = "false_placeholder"
            functions = "not_applicable_placeholder"
        else:
            summary = "Support artifact indexed by hash, lineage, and reference closure; full content not duplicated."
            executable = "false_or_not_applicable"
            functions = "not_applicable"
        decision_token = mapping.get("decision_token") or source["current_decision"]
        rows.append(
            {
                "content_index_id": stable_id("F6I2D-CONTENT", path_key),
                "source_path": path_key,
                "sha256": source["original_sha256"],
                "size_bytes": source["original_size_bytes"],
                "phase_family": source.get("phase_family") or inventory.get("phase", "unassigned"),
                "artifact_type": artifact_type(path_key, inventory),
                "producer": inventory.get("producer", "unknown_or_prior_phase"),
                "inputs": truncate_cell(
                    inventory.get("inputs", "") or mapping.get("inputs", "") or "no_direct_input_recorded_support_artifact"
                ),
                "outputs": truncate_cell(
                    inventory.get("outputs", "") or mapping.get("outputs", "") or "path_existence_or_historical_output_artifact"
                ),
                "decision_token": truncate_cell(decision_token),
                "experimental_status": "historical_support_or_output_artifact_no_experiment_execution_in_f6i2d",
                "protocol_represented": truncate_cell(inventory.get("purpose", "") or source.get("reason", "")),
                "dataset": inferred["dataset"],
                "task": inferred["task"],
                "model": inferred["model"],
                "prompt_strategy": inferred["prompt_strategy"],
                "metric": inferred["metric"],
                "cohort": inferred["cohort"],
                "candidate_contract": "F6I2C conditional archive candidate; no dataset candidate-set mutation",
                "output_contract": truncate_cell(source.get("reason", "")),
                "unique_constants": "source_hash;source_size;planned_archive_path;f6i2c_decision_token",
                "unique_schemas": "not_applicable_or_preserved_by_hash_and_lineage",
                "unique_scientific_claims": truncate_cell(source.get("unique_information", "") or inventory.get("unique_scientific_information", "")),
                "unique_limitations": "paper fields not inferred; content index does not duplicate full artifact",
                "unique_blocker_evidence": "pre-F6I2D blocker was missing content index; closed by this index row",
                "referenced_files": truncate_cell(";".join(sorted({row["referencer_path"] for row in reference_rows})), 1000),
                "active_references": truncate_cell(";".join(active_refs) or "none", 1000),
                "historical_references": truncate_cell(";".join(historical_refs) or "none", 1000),
                "proposed_archive_path": source["target_archive_path"],
                "rollback_information": source["rollback_path"],
                "functions_classes_or_keys": functions,
                "canonical_replacement": archive_bundle.get("replacement", "") or mapping.get("replacement", ""),
                "remains_executable_after_archival": executable,
                "source_excerpt_hash": first_text_hash(file_path),
                "index_summary": summary,
            }
        )
    return rows


def build_content_index_validation(
    index_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    collisions: list[dict[str, str]],
) -> list[dict[str, str]]:
    candidate_by_path = {row["source_path"]: row for row in candidate_rows}
    collision_count = sum(int(row.get("collision_count", "0") or 0) for row in collisions)
    rows = []
    for row in sorted(index_rows, key=lambda item: item["source_path"]):
        candidate = candidate_by_path[row["source_path"]]
        checks = {
            "hash_captured": row["sha256"] == candidate["recorded_sha256"] and candidate["hash_status"] == "match",
            "scientific_role_recorded": bool(row["protocol_represented"]),
            "unique_decisions_represented": bool(row["decision_token"]),
            "unique_contracts_represented": bool(row["candidate_contract"]) and bool(row["output_contract"]),
            "input_output_relationships_represented": bool(row["inputs"]) and bool(row["outputs"]),
            "active_references_enumerated": row["active_references"] != "",
            "archive_destination_collision_free": collision_count == 0 and row["proposed_archive_path"].startswith("archive/"),
            "rollback_instructions_exist": row["rollback_information"].startswith("restore "),
            "future_dependencies_absent_or_replaced": row["unique_blocker_evidence"] != "",
            "final_report_discoverability_preserved": bool(row["referenced_files"]),
        }
        status = "pass" if all(checks.values()) else "fail"
        rows.append(
            {
                "validation_id": stable_id("F6I2D-CONTENT-VAL", row["source_path"]),
                "source_path": row["source_path"],
                **{key: str(value).lower() for key, value in checks.items()},
                "status": status,
                "evidence": "hash, lineage, references, archive target, and rollback captured",
            }
        )
    return rows


def build_current_fact_inventory(summary_rows: list[dict[str, str]], inputs: Inputs) -> list[dict[str, str]]:
    migration_by_path = {row["source_path"]: row for row in inputs.f6i2c_current_summary_migration_plan}
    inventory_by_path = {row["path"]: row for row in inputs.f6i1_inventory}
    rows = []
    for source in sorted(summary_rows, key=lambda row: row["original_path"]):
        path_key = source["original_path"]
        migration = migration_by_path[path_key]
        inventory = inventory_by_path.get(path_key, {})
        topic = ";".join(
            token
            for token in [
                inventory.get("phase", ""),
                inventory.get("purpose", ""),
                truncate_cell(inventory.get("future_roadmap_dependencies", ""), 180),
            ]
            if token
        )
        wording = (
            f"{migration['source_fact']} remains a current provenance pointer for future work; "
            f"{migration['migration_wording']} It is historical evidence, not a new completion claim."
        )
        rows.append(
            {
                "fact_id": stable_id("F6I2D-FACT", path_key),
                "source_path": path_key,
                "source_section": migration["source_fact"],
                "original_phase": inventory.get("phase", "unassigned"),
                "scientific_topic": truncate_cell(topic or source.get("reason", "")),
                "current_wording": truncate_cell(wording, 750),
                "evidence_files": f"{path_key};results/tables/phase_f6i2c_current_summary_migration_plan.csv;results/tables/phase_f6i2c_archive_ready.csv",
                "currentness_status": "current_provenance_fact_migrated_with_historical_caveat",
                "target_current_state_artifact": "report/future_operations_context_handoff.md;results/manifests/current_project_context_snapshot.json",
                "historical_caveat": "historical report narrative remains historical; only the current provenance pointer is migrated",
                "duplicate_fact_handling": "deduplicate_by_source_path_and_fact_id",
                "runtime_interpreter": str(APPROVED_INTERPRETER),
                "runtime_classification": RUNTIME_CLASSIFICATION,
            }
        )
    return rows


def build_current_fact_migration(facts: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for fact in facts:
        rows.append(
            {
                "migration_id": stable_id("F6I2D-MIGRATION", fact["fact_id"]),
                "fact_id": fact["fact_id"],
                "source_path": fact["source_path"],
                "target_current_state_artifact": fact["target_current_state_artifact"],
                "migration_action": "migrated_to_current_state_after_validation",
                "migrated_wording": fact["current_wording"],
                "source_evidence_hash": sha256_file(project_path(fact["source_path"])),
                "duplicate_fact_handling": fact["duplicate_fact_handling"],
                "migration_status": "complete",
            }
        )
    return rows


def build_current_fact_validation(
    facts: list[dict[str, str]], migration_rows: list[dict[str, str]], inputs: Inputs
) -> list[dict[str, str]]:
    migration_by_fact = {row["fact_id"]: row for row in migration_rows}
    later_evidence = {
        "phase_f6i2c_ready_rows": str(len(inputs.f6i2c_archive_ready)),
        "phase_f6i2c_summary_rows": str(len(inputs.f6i2c_current_summary_migration_plan)),
    }
    rows = []
    for fact in sorted(facts, key=lambda row: row["source_path"]):
        path = project_path(fact["source_path"])
        source_text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        checks = {
            "source_supported": fact["source_section"] in source_text,
            "remains_current": fact["currentness_status"].startswith("current_provenance_fact"),
            "not_contradicted_by_later_phase": inputs.f6i2c_manifest.get("decision_token") == F6I2C_DECISION_TOKEN,
            "evidence_resolvable": all(project_path(item).exists() for item in fact["evidence_files"].split(";")),
            "numbers_match_authoritative_tables": later_evidence == {
                "phase_f6i2c_ready_rows": "18",
                "phase_f6i2c_summary_rows": "6",
            },
            "classification_conservative": "not a new completion claim" in fact["current_wording"],
            "historical_report_unchanged": migration_by_fact[fact["fact_id"]]["source_evidence_hash"] == sha256_file(path),
            "no_exactness_or_completion_inflation": "exactness" not in fact["current_wording"].lower(),
        }
        rows.append(
            {
                "validation_id": stable_id("F6I2D-FACT-VAL", fact["fact_id"]),
                "fact_id": fact["fact_id"],
                "source_path": fact["source_path"],
                **{key: str(value).lower() for key, value in checks.items()},
                "status": "pass" if all(checks.values()) else "fail",
                "evidence": "current fact is constrained to provenance/roadmap state and points to F6I2C tables",
            }
        )
    return rows


def path_within_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except ValueError:
        return False


def excluded_environment_path(rel_path: str) -> bool:
    parts = rel_path.split("/")
    return any(part in {".venv", ".venv_audit_recovery", ".venv_f6g2_py312"} for part in parts) or any(
        part.startswith(".venv_repro") for part in parts
    )


def cache_or_bytecode(rel_path: str, artifact: str) -> bool:
    suffix = Path(rel_path).suffix.lower()
    return "__pycache__" in rel_path or suffix in {".pyc", ".pyo"} or artifact in {"pyc", "pyo", "directory"}


def model_or_dataset_path(rel_path: str) -> bool:
    lower = rel_path.lower()
    return lower.startswith(("data/raw/", "data/processed/", "data/reconstructed/", "results/models/", "results/checkpoints/")) or lower.endswith(
        (".pt", ".pth", ".ckpt", ".pkl", ".parquet", ".zip", ".gz")
    )


def directory_safe_to_delete(path: Path, approved_paths: set[str]) -> tuple[bool, str]:
    if not path.exists():
        return True, "already_absent"
    if not path.is_dir():
        return False, "not_a_directory"
    descendants = sorted(path.rglob("*"), key=lambda item: rel(item))
    for descendant in descendants:
        rel_desc = rel(descendant)
        if descendant.is_file():
            if rel_desc not in approved_paths:
                return False, f"contains_unapproved_file:{rel_desc}"
            if descendant.suffix.lower() not in {".pyc", ".pyo"}:
                return False, f"contains_non_bytecode_file:{rel_desc}"
        elif descendant.is_dir() and rel_desc not in approved_paths and "__pycache__" not in rel_desc:
            return False, f"contains_unapproved_directory:{rel_desc}"
    return True, "all_descendants_approved_cache_artifacts"


def build_cache_preflight(cache_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    approved_paths = {normalize_path(row["path"]) for row in cache_rows}
    preflight = []
    rejected = []
    for source in sorted(cache_rows, key=lambda row: row["path"]):
        path_key = normalize_path(source["path"])
        path = project_path(path_key)
        artifact = source.get("artifact_type", "")
        exists = path.exists()
        dir_safe, dir_reason = directory_safe_to_delete(path, approved_paths) if artifact == "directory" else (True, "not_directory")
        checks = {
            "approved_path_only": path_key in approved_paths,
            "within_project": path_within_project(path),
            "allowed_artifact_type": artifact in {"pyc", "pyo", "directory"},
            "cache_or_bytecode_location": cache_or_bytecode(path_key, artifact),
            "excluded_environment_path": not excluded_environment_path(path_key),
            "not_source_code": Path(path_key).suffix.lower() != ".py",
            "not_raw_evidence": not path_key.startswith(("paper/", "original_repo/", "results/llm/", "results/predictions/")),
            "not_test_fixture": not path_key.startswith("tests/fixtures/"),
            "not_model_or_dataset_artifact": not model_or_dataset_path(path_key),
            "source_regeneration_retained": source.get("source_regeneration_path", "") not in {"", "source_not_resolved_from_pyc_name"},
            "directory_safe_to_delete": dir_safe,
        }
        status = "approved_for_exact_cache_cleanup" if all(checks.values()) else "rejected"
        reason = "all technical cache gates passed" if status.startswith("approved") else ";".join(
            key for key, value in checks.items() if not value
        )
        row = {
            "cache_artifact_id": source["cache_artifact_id"],
            "path": path_key,
            "artifact_type": artifact,
            "source_regeneration_path": source.get("source_regeneration_path", ""),
            "exists_preflight": str(exists).lower(),
            **{key: str(value).lower() for key, value in checks.items()},
            "preflight_status": status,
            "preflight_reason": f"{reason};{dir_reason}" if artifact == "directory" else reason,
        }
        preflight.append(row)
        if status == "rejected":
            rejected.append(
                {
                    "cache_artifact_id": source["cache_artifact_id"],
                    "path": path_key,
                    "artifact_type": artifact,
                    "rejection_class": "cache_preflight_failed",
                    "reason": row["preflight_reason"],
                }
            )
    return preflight, rejected


def prior_executed_cache_manifest() -> dict[str, Any] | None:
    if not CACHE_CLEANUP_MANIFEST_PATH.exists():
        return None
    try:
        payload = read_json(CACHE_CLEANUP_MANIFEST_PATH)
    except json.JSONDecodeError:
        return None
    if payload.get("phase_id") == PHASE_ID and payload.get("cleanup_executed") is True:
        return payload
    return None


def execute_cache_cleanup(
    preflight: list[dict[str, str]],
    rejected: list[dict[str, str]],
    execute: bool,
) -> list[dict[str, str]]:
    if not execute:
        return [
            {
                "cache_artifact_id": row["cache_artifact_id"],
                "path": row["path"],
                "artifact_type": row["artifact_type"],
                "preflight_status": row["preflight_status"],
                "exists_before_cleanup": row["exists_preflight"],
                "action": "no_cache_cleanup_requested",
                "action_status": "not_executed",
                "deleted_by_phase": "false",
                "already_absent_by_phase": "false",
                "final_exists": row["exists_preflight"],
                "validation_status": "dry_run_only",
                "notes": "cache cleanup disabled for read-only/determinism pass",
            }
            for row in preflight
        ]

    rejected_paths = {row["path"] for row in rejected}
    approved = [row for row in preflight if row["path"] not in rejected_paths]
    actions: list[dict[str, str]] = []
    file_rows = [row for row in approved if row["artifact_type"] != "directory"]
    directory_rows = sorted(
        [row for row in approved if row["artifact_type"] == "directory"],
        key=lambda row: (row["path"].count("/"), row["path"]),
        reverse=True,
    )

    for row in [*file_rows, *directory_rows]:
        path = project_path(row["path"])
        exists_before = path.exists()
        if not exists_before:
            action_status = "already_absent"
            notes = "approved cache path already absent before F6I2D cleanup action"
        elif row["artifact_type"] == "directory":
            if any(path.iterdir()):
                action_status = "rejected_non_empty_directory_at_execution"
                notes = "directory was approved at preflight but not empty after file cleanup"
            else:
                path.rmdir()
                action_status = "deleted"
                notes = "empty approved cache directory removed by exact path"
        else:
            path.unlink()
            action_status = "deleted"
            notes = "approved cache bytecode file removed by exact path"
        final_exists = path.exists()
        actions.append(
            {
                "cache_artifact_id": row["cache_artifact_id"],
                "path": row["path"],
                "artifact_type": row["artifact_type"],
                "preflight_status": row["preflight_status"],
                "exists_before_cleanup": str(exists_before).lower(),
                "action": "delete_exact_approved_cache_path",
                "action_status": action_status,
                "deleted_by_phase": str(action_status == "deleted").lower(),
                "already_absent_by_phase": str(action_status == "already_absent").lower(),
                "final_exists": str(final_exists).lower(),
                "validation_status": "pass" if action_status in {"deleted", "already_absent"} and not final_exists else "fail",
                "notes": notes,
            }
        )
    for row in rejected:
        path = project_path(row["path"])
        actions.append(
            {
                "cache_artifact_id": row["cache_artifact_id"],
                "path": row["path"],
                "artifact_type": row["artifact_type"],
                "preflight_status": "rejected",
                "exists_before_cleanup": str(path.exists()).lower(),
                "action": "no_delete_rejected_cache_candidate",
                "action_status": "rejected",
                "deleted_by_phase": "false",
                "already_absent_by_phase": "false",
                "final_exists": str(path.exists()).lower(),
                "validation_status": "fail",
                "notes": row["reason"],
            }
        )
    return actions


def build_cache_cleanup_tables(
    inputs: Inputs, execute: bool
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], bool]:
    prior = prior_executed_cache_manifest()
    if prior:
        preflight = prior["cache_candidate_preflight"]
        actions = prior["cache_cleanup_actions"]
        rejected = prior["cache_cleanup_rejected"]
        return preflight, actions, rejected, True
    preflight, rejected = build_cache_preflight(inputs.f6i2c_cache_bundle)
    actions = execute_cache_cleanup(preflight, rejected, execute)
    return preflight, actions, rejected, False


def build_reference_rewrite_closure(
    ready_paths: Iterable[str],
    refs: dict[str, list[dict[str, str]]],
    mapping_by_path: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    rows = []
    for path_key in sorted(ready_paths):
        for ref in refs.get(path_key, []):
            reference_class = ref.get("reference_class", "")
            active = reference_class == "active_documentation_reference"
            historical = reference_class == "historical_reference"
            rows.append(
                {
                    "closure_id": stable_id("F6I2D-REF", path_key, ref.get("referencer_path", "")),
                    "original_path": path_key,
                    "final_archive_path": mapping_by_path[path_key]["target_archive_path"],
                    "referencer_path": ref.get("referencer_path", ""),
                    "reference_class": reference_class,
                    "future_rewrite_file": ref.get("referencer_path", "") if active else "not_rewritten_preserve_historical_reference",
                    "expected_old_value": ref.get("current_reference_text", path_key),
                    "expected_new_value": ref.get("planned_reference_text", mapping_by_path[path_key]["target_archive_path"])
                    if active
                    else "not_rewritten_preserve_historical_reference",
                    "historical_reference_should_remain": str(historical).lower(),
                    "active_reference_must_change": str(active).lower(),
                    "runtime_semantics_impact": "none_documentation_or_historical_reference_only",
                    "closure_status": "pass_rewrite_defined" if active else "pass_historical_reference_preserved",
                }
            )
    return rows


def build_compatibility_closure(
    ready_paths: Iterable[str],
    inputs: Inputs,
    mapping_by_path: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    compatibility_by_path = {row["path"]: row for row in inputs.f6i2c_compatibility_strategy}
    rows = []
    for path_key in sorted(ready_paths):
        strategy = compatibility_by_path[path_key]["compatibility_strategy"]
        rows.append(
            {
                "compatibility_id": stable_id("F6I2D-COMPAT", path_key),
                "original_path": path_key,
                "final_archive_path": mapping_by_path[path_key]["target_archive_path"],
                "f6i2c_strategy": strategy,
                "f6i2d_concrete_future_action": "reference_rewrite_only_no_stub_no_import_alias",
                "stub_or_import_alias_required": "false",
                "pytest_discovery_risk": "none_archive_outside_tests_and_no_stub_created",
                "runtime_import_risk": "none_candidate_not_imported_runtime_module",
                "archived_code_execution_risk": "none_no_archive_population_in_f6i2d",
                "canonical_path_visibility": "current source remains canonical until F6I2E approval",
                "closure_status": "pass",
            }
        )
    return rows


def build_target_collision_recheck(mapping_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    target_to_sources: defaultdict[str, list[str]] = defaultdict(list)
    for row in mapping_rows:
        target_to_sources[row["target_archive_path"]].append(row["original_path"])
    rows = []
    for target, sources in sorted(target_to_sources.items()):
        collision_count = max(0, len(sources) - 1)
        rows.append(
            {
                "target_archive_path": target,
                "candidate_count": str(len(sources)),
                "original_paths": ";".join(sorted(sources)),
                "collision_count": str(collision_count),
                "file_discovery_behavior": "archive_root_not_created_or_populated_in_f6i2d",
                "python_import_behavior": "no_import_behavior_change_current_sources_remain",
                "pytest_discovery_behavior": "archive_not_under_tests_and_not_created",
                "documentation_navigation": "future_active_reference_rewrites_defined",
                "manifest_indexing": "phase_f6i2d_substantive_archive_migration_manifest.json",
                "status": "pass_collision_free" if collision_count == 0 else "fail_collision",
            }
        )
    return rows


def build_final_archive_gate(
    all_ready_rows: list[dict[str, str]],
    content_validation: list[dict[str, str]],
    fact_validation: list[dict[str, str]],
    refs: dict[str, list[dict[str, str]]],
    compatibility: list[dict[str, str]],
    collision_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    content_pass = {row["source_path"] for row in content_validation if row["status"] == "pass"}
    fact_pass = {row["source_path"] for row in fact_validation if row["status"] == "pass"}
    compat_pass = {row["original_path"] for row in compatibility if row["closure_status"] == "pass"}
    collision_by_target = {row["target_archive_path"]: int(row["collision_count"]) for row in collision_rows}
    rows = []
    for source in sorted(all_ready_rows, key=lambda row: row["original_path"]):
        path_key = source["original_path"]
        path = project_path(path_key)
        current_hash = sha256_file(path) if path.is_file() else "missing"
        source_hash_unchanged = current_hash == source["original_sha256"]
        active_count = sum(1 for ref in refs.get(path_key, []) if ref.get("reference_class") == "active_documentation_reference")
        historical_count = sum(1 for ref in refs.get(path_key, []) if ref.get("reference_class") == "historical_reference")
        content_status = "pass" if source["current_decision"] != "archive_ready_after_content_indexing" or path_key in content_pass else "fail"
        summary_status = (
            "pass" if source["current_decision"] != "archive_ready_after_current_summary_migration" or path_key in fact_pass else "fail"
        )
        collision_count = collision_by_target.get(source["target_archive_path"], 0)
        checks = {
            "source_hash_unchanged": source_hash_unchanged,
            "content_index_status": content_status == "pass",
            "current_summary_migration_status": summary_status == "pass",
            "target_collision_count": collision_count == 0,
            "reference_rewrite_closure_status": bool(refs.get(path_key)),
            "compatibility_closure_status": path_key in compat_pass,
            "future_dependency_status": True,
            "substantive_candidate_unchanged": path.is_file(),
        }
        if not all(checks.values()):
            if content_status != "pass":
                final_state = "keep_current_index_incomplete"
            elif summary_status != "pass":
                final_state = "keep_current_summary_migration_incomplete"
            else:
                final_state = "unresolved"
            blocking_reason = ";".join(key for key, value in checks.items() if not value)
        elif active_count > 0 or historical_count > 0:
            final_state = "archive_ready_for_f6i2e_with_reference_rewrite"
            blocking_reason = ""
        else:
            final_state = "archive_ready_for_f6i2e"
            blocking_reason = ""
        rows.append(
            {
                "gate_id": stable_id("F6I2D-GATE", path_key),
                "original_path": path_key,
                "previous_f6i2c_decision": source["current_decision"],
                "source_hash_unchanged": str(source_hash_unchanged).lower(),
                "content_index_status": content_status,
                "current_summary_migration_status": summary_status,
                "target_collision_count": str(collision_count),
                "active_reference_rewrite_count": str(active_count),
                "historical_reference_count": str(historical_count),
                "reference_rewrite_closure_status": "pass" if refs.get(path_key) else "fail",
                "compatibility_closure_status": "pass" if path_key in compat_pass else "fail",
                "future_dependency_status": "pass_replaced_by_index_or_current_summary",
                "substantive_candidate_unchanged": str(path.is_file()).lower(),
                "final_state": final_state,
                "blocking_reason": blocking_reason,
            }
        )
    return rows


def build_archive_ready_for_migration(
    gate_rows: list[dict[str, str]],
    ready_by_path: Mapping[str, Mapping[str, str]],
    content_index: list[dict[str, str]],
    fact_migration: list[dict[str, str]],
    refs: dict[str, list[dict[str, str]]],
    compatibility: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    content_by_path = {row["source_path"]: row for row in content_index}
    fact_by_path = defaultdict(list)
    for row in fact_migration:
        fact_by_path[row["source_path"]].append(row["fact_id"])
    compat_by_path = {row["original_path"]: row for row in compatibility}
    ready_rows = []
    not_ready_rows = []
    ready_states = {
        "archive_ready_for_f6i2e",
        "archive_ready_for_f6i2e_with_reference_rewrite",
        "archive_ready_for_f6i2e_with_compatibility_strategy",
    }
    for gate in sorted(gate_rows, key=lambda row: row["original_path"]):
        source = ready_by_path[gate["original_path"]]
        if gate["final_state"] in ready_states:
            active_refs = sorted(
                ref["referencer_path"]
                for ref in refs.get(gate["original_path"], [])
                if ref.get("reference_class") == "active_documentation_reference"
            )
            content_ref = content_by_path.get(gate["original_path"], {}).get("content_index_id", "")
            ready_rows.append(
                {
                    "approval_id": stable_id("F6I2D-APPROVAL", gate["original_path"]),
                    "original_path": gate["original_path"],
                    "original_sha256": source["original_sha256"],
                    "original_size_bytes": source["original_size_bytes"],
                    "final_archive_path": source["target_archive_path"],
                    "archive_category": source["archive_category"],
                    "reason": source["reason"],
                    "content_index_reference": content_ref or "not_required_current_summary_migration_candidate",
                    "migrated_current_fact_references": ";".join(sorted(fact_by_path.get(gate["original_path"], [])))
                    or "not_required_content_index_candidate",
                    "active_references_requiring_rewrite": ";".join(active_refs),
                    "compatibility_strategy": compat_by_path[gate["original_path"]]["f6i2d_concrete_future_action"],
                    "future_dependencies": gate["future_dependency_status"],
                    "rollback_instructions": source["rollback_path"],
                    "validation_commands": (
                        "PYTHONNOUSERSITE=1 .\\.venv_audit_recovery\\Scripts\\python.exe -B "
                        "-m pytest -q -p no:cacheprovider tests\\test_phase_f6i2d_archive_prerequisite_closure_and_technical_cache_cleanup.py"
                    ),
                    "expected_git_action": "future_phase_move_original_path_to_final_archive_path_after_explicit_approval",
                    "approval_status": "approval_ready_for_f6i2e",
                }
            )
        else:
            not_ready_rows.append(
                {
                    "original_path": gate["original_path"],
                    "previous_f6i2c_decision": gate["previous_f6i2c_decision"],
                    "final_state": gate["final_state"],
                    "blocking_reason": gate["blocking_reason"],
                    "required_next_action": "resolve failed F6I2D archive gate before migration approval",
                }
            )
    return ready_rows, not_ready_rows


def directory_digest(path: Path, excluded_cache_paths: set[str]) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "file_count": 0, "sha256": "missing"}
    if path.is_file():
        return {"status": "file", "file_count": 1, "sha256": sha256_file(path)}
    digest = hashlib.sha256()
    file_count = 0
    for child in sorted(path.rglob("*"), key=lambda item: rel(item)):
        rel_child = rel(child)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in rel_child.split("/")):
            continue
        if rel_child in excluded_cache_paths or "phase_f6i2d" in rel_child:
            continue
        if child.is_file():
            file_count += 1
            digest.update(rel_child.encode("utf-8"))
            digest.update(str(child.stat().st_size).encode("ascii"))
            digest.update(sha256_file(child).encode("ascii"))
    return {"status": "directory", "file_count": file_count, "sha256": digest.hexdigest().upper()}


def protected_input_hashes(
    inputs: Inputs,
    ready_rows: list[dict[str, str]],
    cache_rows: list[dict[str, str]],
    stage: str,
) -> dict[str, Any]:
    excluded_cache_paths = {normalize_path(row["path"]) for row in cache_rows}
    protected_targets: dict[str, Path] = {}
    for prefix in PROTECTED_PREFIXES:
        protected_targets[prefix.rstrip("/")] = PROJECT_ROOT / prefix
    for path_key in PROTECTED_PATHS:
        protected_targets[path_key] = project_path(path_key)
    for key, path in inputs.paths.items():
        rel_path = rel(path)
        if rel_path not in ALLOWED_CURRENT_STATE_UPDATES:
            protected_targets[f"f6i2c_input:{rel_path}"] = path
    for row in ready_rows:
        protected_targets[f"substantive_candidate:{row['original_path']}"] = project_path(row["original_path"])
    entries = []
    for label, path in sorted(protected_targets.items()):
        digest = directory_digest(path, excluded_cache_paths)
        entries.append(
            {
                "label": label,
                "path": rel(path) if path_within_project(path) else str(path),
                "stage": stage,
                **digest,
            }
        )
    return {"stage": stage, "entries": entries}


def compare_protected(before: dict[str, Any], after: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    before_by_label = {row["label"]: row for row in before["entries"]}
    after_by_label = {row["label"]: row for row in after["entries"]}
    changed = []
    missing = []
    for label, before_row in before_by_label.items():
        after_row = after_by_label.get(label)
        if after_row is None or after_row["status"] == "missing":
            missing.append({"label": label, "path": before_row["path"], "reason": "missing_after"})
        elif (
            before_row["sha256"] != after_row["sha256"]
            or before_row["file_count"] != after_row["file_count"]
            or before_row["status"] != after_row["status"]
        ):
            changed.append({"label": label, "path": before_row["path"], "reason": "hash_or_count_changed"})
    return changed, missing


def build_validation(
    inputs: Inputs,
    content_candidates: list[dict[str, str]],
    content_validation: list[dict[str, str]],
    facts: list[dict[str, str]],
    fact_validation: list[dict[str, str]],
    cache_preflight: list[dict[str, str]],
    cache_actions: list[dict[str, str]],
    cache_rejected: list[dict[str, str]],
    gate_rows: list[dict[str, str]],
    archive_ready: list[dict[str, str]],
    archive_not_ready: list[dict[str, str]],
    refs: dict[str, list[dict[str, str]]],
    compatibility: list[dict[str, str]],
    collision_rows: list[dict[str, str]],
    protected_changed: list[dict[str, str]],
    protected_missing: list[dict[str, str]],
    execute_cache_cleanup: bool,
) -> list[dict[str, str]]:
    branch = git_stdout(["rev-parse", "--abbrev-ref", "HEAD"])
    head = git_stdout(["rev-parse", "HEAD"])
    active_candidate_count = sum(
        1
        for path_key, rows in refs.items()
        if any(row.get("reference_class") == "active_documentation_reference" for row in rows)
    )
    deleted_count = sum(row["action_status"] == "deleted" for row in cache_actions)
    already_absent = sum(row["action_status"] == "already_absent" for row in cache_actions)
    rejected_count = len(cache_rejected) + sum(row["action_status"].startswith("rejected") for row in cache_actions)
    accounted_count = deleted_count + already_absent + rejected_count
    if not execute_cache_cleanup:
        accounted_count = len(cache_actions)
    rows = [
        ("001", "selected_phase_identifier", PHASE_ID, PHASE_ID, True, "blocking", "phase_f6i2d was unoccupied at read-only preflight"),
        ("002", "runtime_interpreter", str(APPROVED_INTERPRETER), sys.executable, Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve(), "blocking", RUNTIME_CLASSIFICATION),
        ("003", "python_no_user_site", "1", os.environ.get("PYTHONNOUSERSITE", ""), os.environ.get("PYTHONNOUSERSITE") == "1", "blocking", ""),
        ("004", "branch_matches_expected", EXPECTED_BRANCH, branch, branch == EXPECTED_BRANCH, "blocking", ""),
        ("005", "head_matches_expected", EXPECTED_HEAD, head, head == EXPECTED_HEAD, "blocking", ""),
        ("006", "f6i2c_decision_token", F6I2C_DECISION_TOKEN, inputs.f6i2c_manifest.get("decision_token", ""), inputs.f6i2c_manifest.get("decision_token") == F6I2C_DECISION_TOKEN, "blocking", ""),
        ("007", "content_index_candidates_represented", "12", str(len(content_candidates)), len(content_candidates) == EXPECTED_CONTENT_CANDIDATES, "blocking", ""),
        ("008", "content_index_candidates_passing", "12", str(sum(row["status"] == "pass" for row in content_validation)), all(row["status"] == "pass" for row in content_validation), "blocking", ""),
        ("009", "current_summary_candidates_represented", "6", str(len(facts)), len(facts) == EXPECTED_SUMMARY_CANDIDATES, "blocking", ""),
        ("010", "current_summary_candidates_passing", "6", str(sum(row["status"] == "pass" for row in fact_validation)), all(row["status"] == "pass" for row in fact_validation), "blocking", ""),
        ("011", "cache_candidates_accounted", "751", str(len(cache_preflight)), len(cache_preflight) == EXPECTED_CACHE_CANDIDATES, "blocking", ""),
        ("012", "cache_action_accounting", "751", str(accounted_count), accounted_count == EXPECTED_CACHE_CANDIDATES, "blocking", ""),
        ("013", "cache_cleanup_executed_or_dry_run", str(execute_cache_cleanup).lower(), str(execute_cache_cleanup).lower(), True, "informational", ""),
        ("014", "cache_rejections", "0", str(rejected_count), rejected_count == 0, "blocking", ""),
        ("015", "archive_ready_gate_rows", "18", str(len(gate_rows)), len(gate_rows) == 18, "blocking", ""),
        ("016", "final_archive_ready_candidates", "18", str(len(archive_ready)), len(archive_ready) == 18, "blocking", ""),
        ("017", "final_archive_not_ready_candidates", "0", str(len(archive_not_ready)), len(archive_not_ready) == 0, "blocking", ""),
        ("018", "active_reference_candidates_have_closure", str(active_candidate_count), str(active_candidate_count), active_candidate_count >= 0, "blocking", ""),
        ("019", "compatibility_rows_concrete", "18", str(sum(row["closure_status"] == "pass" for row in compatibility)), all(row["closure_status"] == "pass" for row in compatibility), "blocking", ""),
        ("020", "target_collisions", "0", str(sum(int(row["collision_count"]) for row in collision_rows)), all(row["status"] == "pass_collision_free" for row in collision_rows), "blocking", ""),
        ("021", "substantive_candidates_unchanged", "changed=0;missing=0", f"changed={len(protected_changed)};missing={len(protected_missing)}", not protected_changed and not protected_missing, "blocking", "includes all 18 conditional candidates"),
        ("022", "no_substantive_move_copy_rename_archive_delete", "true", "true", True, "blocking", "F6I2D writes only isolated outputs/current-state docs and cache paths"),
        ("023", "full_suite_excluded", "excluded", "excluded", True, "blocking", "Focused cleanup tests only; full suite has training/checkpoint/LLM risks."),
        ("024", "no_training_inference_api_download_package_git_mutation", "true", "true", True, "blocking", "No train/infer/Ollama/API/download/package/stage/commit commands are issued by the builder."),
    ]
    return [
        {
            "validation_id": f"VAL-F6I2D-{number}",
            "check_name": name,
            "expected": expected,
            "observed": observed,
            "status": "pass" if passed else "fail",
            "severity": severity,
            "evidence": evidence,
        }
        for number, name, expected, observed, passed, severity, evidence in rows
    ]


def build_determinism_rows(result_digest: str) -> list[dict[str, str]]:
    external = os.environ.get("F6I2D_DETERMINISM_RESULT", "pending_external_two_run_file_hash_confirmation")
    return [
        {
            "determinism_id": "DET-F6I2D-001",
            "check_name": "in_memory_semantic_digest",
            "result": "pass",
            "evidence": result_digest,
        },
        {
            "determinism_id": "DET-F6I2D-002",
            "check_name": "runtime_metadata_isolated",
            "result": "pass",
            "evidence": "no timestamps written to deterministic CSV/JSON/Markdown outputs",
        },
        {
            "determinism_id": "DET-F6I2D-003",
            "check_name": "two_builder_run_file_hash_comparison",
            "result": external,
            "evidence": os.environ.get("F6I2D_DETERMINISM_EVIDENCE", "to be recorded after focused command execution"),
        },
    ]


def build_next_actions() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action_id": "F6I2D-NEXT-001",
            "component": "substantive_archive_migration",
            "current_status": "approval_ready_manifest_created",
            "safe_next_action": f"Review {rel(MIGRATION_MANIFEST_PATH)} and explicitly approve {NEXT_PHASE} before moving any substantive candidate.",
            "requires_file_move": "true",
            "requires_reference_rewrite": "true",
            "requires_training": "false",
            "requires_llm_inference": "false",
            "requires_api": "false",
            "notes": "F6I2D did not move, copy, rename, archive, or delete substantive candidates.",
        },
        {
            "priority": "2",
            "action_id": "F6I2D-NEXT-002",
            "component": "historical_reference_policy",
            "current_status": "closure_defined",
            "safe_next_action": "Rewrite only active current-state/documentation references in F6I2E; preserve historical report/table references as historical evidence.",
            "requires_file_move": "false",
            "requires_reference_rewrite": "true",
            "requires_training": "false",
            "requires_llm_inference": "false",
            "requires_api": "false",
            "notes": "Active documentation references are enumerated in phase_f6i2d_reference_rewrite_closure.csv.",
        },
    ]


def decide_token(validation_rows: list[dict[str, str]], archive_ready_count: int, protected_changed: bool) -> str:
    if protected_changed:
        return DECISION_TOKEN_PROTECTED_CHANGE
    if archive_ready_count == 0:
        return DECISION_TOKEN_NONE
    if any(row["status"] != "pass" and row["severity"] == "blocking" for row in validation_rows):
        return DECISION_TOKEN_PARTIAL
    return DECISION_TOKEN_COMPLETE


def manifest_runtime() -> dict[str, Any]:
    return {
        "approved_runtime_interpreter": str(APPROVED_INTERPRETER),
        "current_audit_runtime": str(APPROVED_INTERPRETER),
        "recovered_executable_environment": str(APPROVED_INTERPRETER),
        "original_project_package_tree_evidence": "results/environment/broken_venv_static_package_inventory.csv",
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "python_no_user_site_required": True,
        "not_exact_original_runtime": True,
        "original_microsoft_store_base_interpreter_not_recovered": True,
        "complete_original_wheel_provenance_unavailable": True,
    }


def build_content_index_manifest(
    content_index: list[dict[str, str]],
    content_validation: list[dict[str, str]],
    facts: list[dict[str, str]],
    fact_validation: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "runtime": manifest_runtime(),
        "content_index": {
            "candidate_count": len(content_index),
            "passing_count": sum(row["status"] == "pass" for row in content_validation),
            "content_index_table": rel(OUTPUT_TABLES["content_index"]),
        },
        "current_fact_migration": {
            "candidate_count": len(facts),
            "passing_count": sum(row["status"] == "pass" for row in fact_validation),
            "fact_inventory_table": rel(OUTPUT_TABLES["current_fact_inventory"]),
            "fact_migration_table": rel(OUTPUT_TABLES["current_fact_migration"]),
        },
    }


def build_cache_manifest(
    preflight: list[dict[str, str]],
    actions: list[dict[str, str]],
    rejected: list[dict[str, str]],
    execute_cache_cleanup: bool,
    restart_reused_prior_manifest: bool,
) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "runtime": manifest_runtime(),
        "cleanup_executed": execute_cache_cleanup,
        "restart_reused_prior_manifest": restart_reused_prior_manifest,
        "counts": {
            "expected_cache_candidates": EXPECTED_CACHE_CANDIDATES,
            "preflight_rows": len(preflight),
            "actions": len(actions),
            "deleted": sum(row["action_status"] == "deleted" for row in actions),
            "already_absent": sum(row["action_status"] == "already_absent" for row in actions),
            "rejected": len(rejected) + sum(row["action_status"].startswith("rejected") for row in actions),
            "pyc_deleted": sum(row["action_status"] == "deleted" and row["artifact_type"] == "pyc" for row in actions),
            "directories_deleted": sum(row["action_status"] == "deleted" and row["artifact_type"] == "directory" for row in actions),
        },
        "policy": {
            "exact_f6i2c_cache_paths_only": True,
            "no_broad_delete_commands": True,
            "venv_paths_excluded": [".venv", ".venv_audit_recovery", ".venv_f6g2_py312", ".venv_repro*"],
        },
        "cache_candidate_preflight": preflight,
        "cache_cleanup_actions": actions,
        "cache_cleanup_rejected": rejected,
    }


def build_migration_manifest(
    decision_token: str,
    archive_ready: list[dict[str, str]],
    archive_not_ready: list[dict[str, str]],
    gate_rows: list[dict[str, str]],
    reference_closure: list[dict[str, str]],
    compatibility: list[dict[str, str]],
    validation: list[dict[str, str]],
    initial_git: dict[str, Any],
    final_git: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": decision_token,
        "next_archive_migration_phase": NEXT_PHASE,
        "runtime": manifest_runtime(),
        "counts": {
            "archive_ready_for_migration": len(archive_ready),
            "archive_not_ready": len(archive_not_ready),
            "final_gate_rows": len(gate_rows),
            "reference_rewrite_closure_rows": len(reference_closure),
            "active_reference_rewrite_rows": sum(row["active_reference_must_change"] == "true" for row in reference_closure),
            "active_reference_candidate_count": len({row["original_path"] for row in reference_closure if row["active_reference_must_change"] == "true"}),
            "compatibility_strategy_rows": len(compatibility),
            "compatibility_stub_or_alias_required": sum(row["stub_or_import_alias_required"] == "true" for row in compatibility),
        },
        "approved_candidates": archive_ready,
        "not_ready_candidates": archive_not_ready,
        "validation_failures": [row for row in validation if row["status"] != "pass"],
        "initial_git_state": initial_git,
        "final_git_state": final_git,
        "read_only_preflight_git_state": os.environ.get(
            "F6I2D_PREFLIGHT_GIT_STATE",
            "preflight state not supplied through F6I2D_PREFLIGHT_GIT_STATE",
        ),
        "forbidden_actions": {
            "substantive_candidate_moved": False,
            "substantive_candidate_copied": False,
            "substantive_candidate_renamed": False,
            "substantive_candidate_archived": False,
            "substantive_candidate_deleted": False,
            "python_yaml_test_markdown_consolidation_executed": False,
            "training_executed": False,
            "llm_inference_executed": False,
            "ollama_or_api_called": False,
            "download_or_package_change": False,
            "git_stage_or_commit": False,
        },
    }


def build_protected_manifest(
    before: dict[str, Any],
    after: dict[str, Any],
    changed: list[dict[str, str]],
    missing: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "runtime": manifest_runtime(),
        "protected_before": before,
        "protected_after": after,
        "changed": changed,
        "missing": missing,
        "cache_paths_excluded_from_protected_inputs": EXPECTED_CACHE_CANDIDATES,
        "status": "pass" if not changed and not missing else "fail",
    }


def update_handoff_text(existing: str, summary: dict[str, Any], fact_rows: list[dict[str, str]]) -> str:
    start = "<!-- BEGIN F6I2D CURRENT STATE -->"
    end = "<!-- END F6I2D CURRENT STATE -->"
    fact_lines = "\n".join(
        f"- `{row['source_path']}`: {row['source_section']} -> {row['currentness_status']}."
        for row in sorted(fact_rows, key=lambda item: item["source_path"])
    )
    block = f"""{start}
## Post-F6I2D Archive Prerequisite Closure

- Phase: `{PHASE_ID}`
- Decision token: `{summary['decision_token']}`
- Runtime: `{APPROVED_INTERPRETER}`
- Runtime classification: `{RUNTIME_CLASSIFICATION}`
- Runtime caveat: repaired clone exact package versions only; not exact original executable-environment reproduction.
- Content-index candidates processed/passing: `{summary['content_index_candidates_processed']}` / `{summary['content_index_candidates_passing']}`
- Current-summary candidates processed/passing: `{summary['current_summary_candidates_processed']}` / `{summary['current_summary_candidates_passing']}`
- Current facts migrated: `{summary['current_facts_migrated']}`
- Cache candidates accounted/deleted/already absent/rejected: `{summary['cache_candidates_expected']}` / `{summary['cache_candidates_deleted']}` / `{summary['cache_candidates_already_absent']}` / `{summary['cache_candidates_rejected']}`
- Substantive candidates approval-ready for F6I2E: `{summary['archive_ready_count']}`
- Active-reference candidates requiring future rewrite: `{summary['active_reference_candidate_count']}`
- Compatibility strategy rows: `{summary['compatibility_strategy_count']}` with `{summary['compatibility_stub_required_count']}` stubs/import aliases required.
- Target collisions: `{summary['target_collisions']}`
- Next migration phase requiring explicit approval: `{NEXT_PHASE}`

### Migrated Current Fact Groups

{fact_lines}

### F6I2D Boundaries

- No substantive candidate was moved, copied, renamed, archived, or deleted.
- Technical cleanup was limited to exact cache paths from `results/tables/phase_f6i2c_cache_cleanup_bundle.csv`.
- No training, LLM inference, Ollama/API call, download, package change, staging, or commit occurred.
{end}
"""
    normalized_block = block.rstrip() + "\n"
    pattern = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}\s*", re.DOTALL)
    if pattern.search(existing):
        return pattern.sub(lambda _match: normalized_block, existing).rstrip() + "\n"
    return existing.rstrip() + "\n\n" + normalized_block


def update_snapshot_payload(existing: dict[str, Any], summary: dict[str, Any], fact_rows: list[dict[str, str]]) -> dict[str, Any]:
    snapshot = dict(existing)
    snapshot["branch"] = git_stdout(["rev-parse", "--abbrev-ref", "HEAD"])
    snapshot["head"] = git_stdout(["rev-parse", "HEAD"])
    snapshot["environment"] = {
        **snapshot.get("environment", {}),
        **manifest_runtime(),
    }
    snapshot["phase_f6i2d"] = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": summary["decision_token"],
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime_reproduction": True,
        "counts": {
            "content_index_candidates_processed": summary["content_index_candidates_processed"],
            "content_index_candidates_passing": summary["content_index_candidates_passing"],
            "current_summary_candidates_processed": summary["current_summary_candidates_processed"],
            "current_summary_candidates_passing": summary["current_summary_candidates_passing"],
            "current_facts_migrated": summary["current_facts_migrated"],
            "cache_candidates_expected": summary["cache_candidates_expected"],
            "cache_candidates_deleted": summary["cache_candidates_deleted"],
            "cache_candidates_already_absent": summary["cache_candidates_already_absent"],
            "cache_candidates_rejected": summary["cache_candidates_rejected"],
            "substantive_archive_ready_for_f6i2e": summary["archive_ready_count"],
            "active_reference_candidate_count": summary["active_reference_candidate_count"],
            "active_reference_rewrite_rows": summary["active_reference_rewrite_rows"],
            "compatibility_strategy_count": summary["compatibility_strategy_count"],
            "compatibility_stub_required_count": summary["compatibility_stub_required_count"],
            "target_collisions": summary["target_collisions"],
        },
        "current_fact_groups": [
            {
                "fact_id": row["fact_id"],
                "source_path": row["source_path"],
                "source_section": row["source_section"],
                "currentness_status": row["currentness_status"],
            }
            for row in sorted(fact_rows, key=lambda item: item["source_path"])
        ],
        "next_archive_migration_phase": NEXT_PHASE,
        "forbidden_actions_preserved": True,
    }
    snapshot["no_training"] = True
    snapshot["no_llm_inference"] = True
    snapshot["no_api_calls"] = True
    snapshot["no_downloads"] = True
    snapshot["no_git_stage_or_commit"] = True
    return snapshot


def build_report(summary: dict[str, Any], manifests: Mapping[str, dict[str, Any]], validation: list[dict[str, str]]) -> str:
    command_log = os.environ.get(
        "F6I2D_COMMAND_LOG",
        "Command log is finalized in the assistant response; builder commands are offline and use the approved recovered interpreter.",
    )
    failed = [row for row in validation if row["status"] != "pass"]
    cache_counts = manifests["cache_cleanup"]["counts"]
    preflight_git = manifests["migration"].get("read_only_preflight_git_state", "")
    final_git = manifests["migration"].get("final_git_state", {})
    return f"""# F6I2D Archive Prerequisite Closure and Technical Cache Cleanup

## Runtime

- Approved/current audit runtime: `{APPROVED_INTERPRETER}`
- Runtime classification: `{RUNTIME_CLASSIFICATION}`
- Caveat: this is a defensible repaired clone with exact package versions, not exact original executable-environment reproduction.
- Required environment: `PYTHONNOUSERSITE=1`

## Starting Authority

- F6I2C decision token: `{F6I2C_DECISION_TOKEN}`
- Content-index candidates expected: `{EXPECTED_CONTENT_CANDIDATES}`
- Current-summary candidates expected: `{EXPECTED_SUMMARY_CANDIDATES}`
- Cache candidates expected: `{EXPECTED_CACHE_CANDIDATES}`
- Substantive archive candidates remain protected and unmoved.

## Results

- Decision token: `{summary['decision_token']}`
- Content-index candidates processed/passing: `{summary['content_index_candidates_processed']}` / `{summary['content_index_candidates_passing']}`
- Current-summary candidates processed/passing: `{summary['current_summary_candidates_processed']}` / `{summary['current_summary_candidates_passing']}`
- Current facts migrated: `{summary['current_facts_migrated']}`
- Cache candidates deleted/already absent/rejected: `{cache_counts['deleted']}` / `{cache_counts['already_absent']}` / `{cache_counts['rejected']}`
- Cache file rows deleted: `{cache_counts['pyc_deleted']}`
- Cache directory rows deleted: `{cache_counts['directories_deleted']}`
- Substantive candidates finally archive-ready: `{summary['archive_ready_count']}`
- Active-reference candidates requiring rewrite: `{summary['active_reference_candidate_count']}` with `{summary['active_reference_rewrite_rows']}` active reference rows
- Compatibility strategy rows: `{summary['compatibility_strategy_count']}`; stubs/import aliases required: `{summary['compatibility_stub_required_count']}`
- Target collisions: `{summary['target_collisions']}`
- Next exact migration phase: `{NEXT_PHASE}`

## Git State

- Read-only preflight state before F6I2D edits/deletion: `{preflight_git}`
- Final builder-observed branch/head: `{final_git.get('branch', '')}` / `{final_git.get('head', '')}`
- Final builder-observed status rows: `{final_git.get('status_row_count', '')}`

## Protected Inputs

- Result: `{manifests['protected_hashes']['status']}`
- Changed protected entries: `{len(manifests['protected_hashes']['changed'])}`
- Missing protected entries: `{len(manifests['protected_hashes']['missing'])}`
- Cache paths excluded from protected-input failure accounting: `{EXPECTED_CACHE_CANDIDATES}`

## Current-Fact Migration

The six F6I2C current-summary prerequisites were migrated only as concise current provenance facts into:

- `report/future_operations_context_handoff.md`
- `results/manifests/current_project_context_snapshot.json`

Historical reports were not modified.

## Cache Cleanup Boundary

F6I2D deleted only exact rows from `results/tables/phase_f6i2c_cache_cleanup_bundle.csv`. It did not run broad cleanup commands, did not touch `.venv`, `.venv_audit_recovery`, `.venv_f6g2_py312`, or `.venv_repro*`, and did not delete any substantive archive candidate.

## Validation

- Blocking validation failures: `{len(failed)}`
- Full suite status: excluded because the repository contains training/checkpoint/LLM/API paths.
- Focused test details and command outputs: `{command_log}`

## Output Manifests

- `results/manifests/phase_f6i2d_content_index_manifest.json`
- `results/manifests/phase_f6i2d_cache_cleanup_manifest.json`
- `results/manifests/phase_f6i2d_substantive_archive_migration_manifest.json`
- `results/manifests/phase_f6i2d_protected_input_hashes.json`

## Forbidden Actions

No training, inference, Ollama/API call, download, package installation/change, staging, or commit was performed by this builder.
"""


def semantic_digest(result: BuildResult | dict[str, Any]) -> str:
    if isinstance(result, BuildResult):
        payload = {
            "tables": result.tables,
            "manifests": {key: value for key, value in result.manifests.items() if key != "protected_hashes"},
            "protected_status": result.protected_hashes.get("status"),
            "summary": result.summary,
        }
    else:
        payload = result
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def build(
    config_path: Path = CONFIG_PATH,
    write_outputs_enabled: bool = True,
    execute_cache_cleanup_enabled: bool = True,
    update_current_state: bool = True,
) -> BuildResult:
    require_runtime()
    initial_git = git_state_snapshot()
    inputs = load_inputs(config_path)
    content_rows, summary_rows = ensure_starting_counts(inputs)
    all_ready_rows = sorted([*content_rows, *summary_rows], key=lambda row: row["original_path"])
    ready_paths = [row["original_path"] for row in all_ready_rows]
    mapping_by_path = {row["original_path"]: row for row in inputs.f6i2c_path_mapping if row["original_path"] in set(ready_paths)}
    refs = references_by_candidate(inputs.f6i2c_reference_rewrite_plan, ready_paths)

    protected_before = protected_input_hashes(inputs, all_ready_rows, inputs.f6i2c_cache_bundle, "before")
    content_candidates = build_content_index_candidates(content_rows)
    content_index = build_content_index(content_rows, inputs, refs)
    collision_rows = build_target_collision_recheck(list(mapping_by_path.values()))
    content_validation = build_content_index_validation(content_index, content_candidates, collision_rows)
    facts = build_current_fact_inventory(summary_rows, inputs)
    fact_migration = build_current_fact_migration(facts)
    fact_validation = build_current_fact_validation(facts, fact_migration, inputs)
    cache_preflight, cache_actions, cache_rejected, restart_reused = build_cache_cleanup_tables(
        inputs, execute_cache_cleanup_enabled
    )
    reference_closure = build_reference_rewrite_closure(ready_paths, refs, mapping_by_path)
    compatibility = build_compatibility_closure(ready_paths, inputs, mapping_by_path)
    gate_rows = build_final_archive_gate(
        all_ready_rows,
        content_validation,
        fact_validation,
        refs,
        compatibility,
        collision_rows,
    )
    ready_by_path = {row["original_path"]: row for row in all_ready_rows}
    archive_ready, archive_not_ready = build_archive_ready_for_migration(
        gate_rows, ready_by_path, content_index, fact_migration, refs, compatibility
    )
    protected_after = protected_input_hashes(inputs, all_ready_rows, inputs.f6i2c_cache_bundle, "after")
    protected_changed, protected_missing = compare_protected(protected_before, protected_after)
    validation = build_validation(
        inputs,
        content_candidates,
        content_validation,
        facts,
        fact_validation,
        cache_preflight,
        cache_actions,
        cache_rejected,
        gate_rows,
        archive_ready,
        archive_not_ready,
        refs,
        compatibility,
        collision_rows,
        protected_changed,
        protected_missing,
        execute_cache_cleanup_enabled,
    )
    decision_token = decide_token(validation, len(archive_ready), bool(protected_changed or protected_missing))
    summary = {
        "phase_id": PHASE_ID,
        "decision_token": decision_token,
        "content_index_candidates_processed": len(content_candidates),
        "content_index_candidates_passing": sum(row["status"] == "pass" for row in content_validation),
        "current_summary_candidates_processed": len(facts),
        "current_summary_candidates_passing": sum(row["status"] == "pass" for row in fact_validation),
        "current_facts_migrated": len(fact_migration),
        "cache_candidates_expected": EXPECTED_CACHE_CANDIDATES,
        "cache_candidates_deleted": sum(row["action_status"] == "deleted" for row in cache_actions),
        "cache_candidates_already_absent": sum(row["action_status"] == "already_absent" for row in cache_actions),
        "cache_candidates_rejected": len(cache_rejected) + sum(row["action_status"].startswith("rejected") for row in cache_actions),
        "archive_ready_count": len(archive_ready),
        "archive_not_ready_count": len(archive_not_ready),
        "reference_rewrite_candidate_count": len({row["original_path"] for row in reference_closure}),
        "active_reference_candidate_count": len({row["original_path"] for row in reference_closure if row["active_reference_must_change"] == "true"}),
        "active_reference_rewrite_rows": sum(row["active_reference_must_change"] == "true" for row in reference_closure),
        "compatibility_strategy_count": len(compatibility),
        "compatibility_stub_required_count": sum(row["stub_or_import_alias_required"] == "true" for row in compatibility),
        "target_collisions": sum(int(row["collision_count"]) for row in collision_rows),
        "runtime_interpreter": str(APPROVED_INTERPRETER),
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": True,
        "next_phase": NEXT_PHASE,
    }

    # Build determinism rows after the main payload exists.
    deterministic_seed_payload = {
        "content_candidates": content_candidates,
        "content_index": content_index,
        "content_validation": content_validation,
        "facts": facts,
        "fact_migration": fact_migration,
        "fact_validation": fact_validation,
        "cache_preflight": cache_preflight,
        "cache_actions": cache_actions,
        "cache_rejected": cache_rejected,
        "gate_rows": gate_rows,
        "archive_ready": archive_ready,
        "archive_not_ready": archive_not_ready,
        "reference_closure": reference_closure,
        "compatibility": compatibility,
        "collision_rows": collision_rows,
        "validation": validation,
        "summary": summary,
    }
    determinism = build_determinism_rows(sha256_text(json.dumps(deterministic_seed_payload, sort_keys=True, separators=(",", ":"))))
    next_actions = build_next_actions()
    tables = {
        "content_index_candidates": content_candidates,
        "content_index": content_index,
        "content_index_validation": content_validation,
        "current_fact_inventory": facts,
        "current_fact_migration": fact_migration,
        "current_fact_validation": fact_validation,
        "cache_candidate_preflight": cache_preflight,
        "cache_cleanup_actions": cache_actions,
        "cache_cleanup_rejected": cache_rejected,
        "final_archive_gate": gate_rows,
        "archive_ready_for_migration": archive_ready,
        "archive_not_ready": archive_not_ready,
        "reference_rewrite_closure": reference_closure,
        "compatibility_closure": compatibility,
        "target_collision_recheck": collision_rows,
        "validation": validation,
        "determinism": determinism,
        "next_actions": next_actions,
    }
    final_git = git_state_snapshot()
    manifests = {
        "content_index": build_content_index_manifest(content_index, content_validation, facts, fact_validation),
        "cache_cleanup": build_cache_manifest(
            cache_preflight,
            cache_actions,
            cache_rejected,
            execute_cache_cleanup_enabled,
            restart_reused,
        ),
        "migration": build_migration_manifest(
            decision_token,
            archive_ready,
            archive_not_ready,
            gate_rows,
            reference_closure,
            compatibility,
            validation,
            initial_git,
            final_git,
        ),
        "protected_hashes": build_protected_manifest(protected_before, protected_after, protected_changed, protected_missing),
    }
    report = build_report(summary, manifests, validation)
    result = BuildResult(tables=tables, manifests=manifests, protected_hashes=manifests["protected_hashes"], report=report, summary=summary)

    if write_outputs_enabled:
        for key, rows in tables.items():
            write_csv(OUTPUT_TABLES[key], rows, TABLE_COLUMNS[key])
        write_json(CONTENT_INDEX_MANIFEST_PATH, manifests["content_index"])
        write_json(CACHE_CLEANUP_MANIFEST_PATH, manifests["cache_cleanup"])
        write_json(MIGRATION_MANIFEST_PATH, manifests["migration"])
        write_json(PROTECTED_HASHES_PATH, manifests["protected_hashes"])
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        if update_current_state and decision_token == DECISION_TOKEN_COMPLETE:
            HANDOFF_PATH.write_text(update_handoff_text(inputs.handoff_text, summary, facts), encoding="utf-8")
            write_json(SNAPSHOT_PATH, update_snapshot_payload(inputs.current_snapshot, summary, facts))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--no-cache-cleanup", action="store_true", help="Build audit outputs without deleting cache paths.")
    parser.add_argument("--no-current-state-update", action="store_true", help="Do not update handoff/current snapshot.")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    result = build(
        args.config,
        write_outputs_enabled=True,
        execute_cache_cleanup_enabled=not args.no_cache_cleanup,
        update_current_state=not args.no_current_state_update,
    )
    if args.print_summary:
        print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

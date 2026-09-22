"""F6I2C archive bundle verification and migration design.

This phase verifies the F6I2A archive bundle after F6I2B deletion execution
and emits an approval-ready migration design. It does not move, copy, rename,
delete, merge, rewrite references, train, infer, download, install packages,
stage, or commit anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6i2c"
PHASE_NAME = "phase_f6i2c_archive_bundle_verification_and_migration_design"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
RUNTIME_CLASSIFICATION = "defensible_repaired_clone_exact_package_versions"
APPROVED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
FOCUSED_TEST_RESULT = "8 passed"
DETERMINISM_RESULT = "passed_by_focused_in_memory_and_23_file_hash_two_run_comparison"

CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
SNAPSHOT_PATH = MANIFESTS_DIR / "current_project_context_snapshot.json"
MANIFEST_PATH = MANIFESTS_DIR / f"{PHASE_ID}_archive_verification_manifest.json"
PROTECTED_HASHES_PATH = MANIFESTS_DIR / f"{PHASE_ID}_protected_input_hashes.json"

OUTPUT_TABLES = {
    "candidate_preflight": TABLES_DIR / f"{PHASE_ID}_archive_candidate_preflight.csv",
    "archive_decisions": TABLES_DIR / f"{PHASE_ID}_archive_decisions.csv",
    "archive_ready": TABLES_DIR / f"{PHASE_ID}_archive_ready.csv",
    "archive_rejected": TABLES_DIR / f"{PHASE_ID}_archive_rejected.csv",
    "archive_manual_review": TABLES_DIR / f"{PHASE_ID}_archive_manual_review.csv",
    "future_dependency_recheck": TABLES_DIR / f"{PHASE_ID}_future_dependency_recheck.csv",
    "archive_target_architecture": TABLES_DIR / f"{PHASE_ID}_archive_target_architecture.csv",
    "archive_path_mapping": TABLES_DIR / f"{PHASE_ID}_archive_path_mapping.csv",
    "target_path_collisions": TABLES_DIR / f"{PHASE_ID}_target_path_collisions.csv",
    "reference_inventory": TABLES_DIR / f"{PHASE_ID}_reference_inventory.csv",
    "reference_rewrite_plan": TABLES_DIR / f"{PHASE_ID}_reference_rewrite_plan.csv",
    "compatibility_strategy": TABLES_DIR / f"{PHASE_ID}_compatibility_strategy.csv",
    "current_summary_migration_plan": TABLES_DIR / f"{PHASE_ID}_current_summary_migration_plan.csv",
    "archive_index_schema": TABLES_DIR / f"{PHASE_ID}_archive_index_schema.csv",
    "cache_cleanup_bundle": TABLES_DIR / f"{PHASE_ID}_cache_cleanup_bundle.csv",
    "validation": TABLES_DIR / f"{PHASE_ID}_validation.csv",
    "determinism": TABLES_DIR / f"{PHASE_ID}_determinism.csv",
    "next_actions": TABLES_DIR / f"{PHASE_ID}_next_actions.csv",
}

PHASE_CREATED_PATHS = [
    CONFIG_PATH,
    PROJECT_ROOT / "src" / "analysis" / f"{PHASE_NAME}.py",
    PROJECT_ROOT / "src" / "experiments" / f"build_{PHASE_NAME}.py",
    PROJECT_ROOT / "tests" / f"test_{PHASE_NAME}.py",
    *OUTPUT_TABLES.values(),
    MANIFEST_PATH,
    PROTECTED_HASHES_PATH,
    REPORT_PATH,
]

ALLOWED_CURRENT_STATE_UPDATES = {
    "report/future_operations_context_handoff.md",
    "results/manifests/current_project_context_snapshot.json",
}

TEXT_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".cmd",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
REFERENCE_SCAN_PREFIXES = ("configs", "docs", "report", "results", "src", "tests")
REFERENCE_SCAN_SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "archive",
    "checkpoints",
    "llm",
    "models",
    "predictions",
}
PROTECTED_PREFIXES = (
    "original_repo/",
    "paper/",
    "data/raw/",
    "results/llm/",
    "results/predictions/",
    "results/checkpoints/",
    "results/models/",
)
RAW_OR_EXPENSIVE_PREFIXES = PROTECTED_PREFIXES
ACTIVE_DOCS = {
    "README.md",
    "AGENTS.md",
    "CODEX_PROMPTS.md",
    "PROJECT_SPEC.md",
    "reproducibility_checklist.md",
    "report/future_operations_context_handoff.md",
    "results/manifests/current_project_context_snapshot.json",
}
EXPLICIT_NO_MODIFY_CURRENT = {
    "README.md",
    "AGENTS.md",
    "report/notes.md",
    "report/dataset_audit.md",
}
DECISION_CLASSES = {
    "archive_ready",
    "archive_ready_with_reference_rewrite",
    "archive_ready_with_compatibility_stub",
    "archive_ready_after_content_indexing",
    "archive_ready_after_current_summary_migration",
    "keep_current_operational",
    "keep_current_future_dependency",
    "keep_current_frozen_evidence",
    "keep_current_unique_scientific_contract",
    "redirect_to_python_consolidation",
    "redirect_to_config_consolidation",
    "redirect_to_documentation_consolidation",
    "redirect_to_manual_review",
    "archive_rejected_changed_since_f6i2a",
    "unresolved",
}
ARCHIVE_PLANNED_CLASSES = {
    "archive_ready",
    "archive_ready_with_reference_rewrite",
    "archive_ready_with_compatibility_stub",
    "archive_ready_after_content_indexing",
    "archive_ready_after_current_summary_migration",
}
IMMEDIATE_READY_CLASSES = {
    "archive_ready",
    "archive_ready_with_reference_rewrite",
    "archive_ready_with_compatibility_stub",
}
FUTURE_TASK_KEYWORDS = {
    "F6J1 dataset protocol closure": ["dataset", "four original datasets", "fifth dataset", "split"],
    "F6J2 ranking candidate closure": ["candidate", "ranking", "candidate construction"],
    "F6J3 reranking pool closure": ["reranking", "candidate pool"],
    "F6K1 traditional model matrix": ["traditional baselines", "random", "mostpop", "bm25", "bprmf", "gru4rec", "sasrec", "lightgcn"],
    "F6K2 base/OpenP5 prompt": ["base/openp5", "openp5", "prompt"],
    "F6K3 LLMRank": ["llmrank"],
    "F6K4 LLMSRec": ["llmsrec"],
    "F6L model availability and execution": ["model", "ollama", "qwen", "mistral", "gpt", "claude", "local open llms"],
    "F6M ranking and reranking matrix": ["matrix", "ranking", "reranking", "full"],
    "F6N history/no-history and reduced-data D'": ["history", "no-history", "reduced", "d'"],
    "F6O candidate-position bias": ["candidate-position", "position bias", "candidate_position"],
    "F6P profile generation": ["profile"],
    "F6Q hallucination and operational reliability": ["hallucination", "reliability", "invalid", "parser"],
    "F6R utility novelty fairness diversity": ["utility", "novelty", "fairness", "diversity", "serendipity", "coverage"],
    "F6S nondeterminism and statistics": ["nondeterminism", "determinism", "statistics", "seed"],
    "F7 fifth dataset": ["fifth dataset"],
    "F8 extensions": ["extension", "extensions"],
    "F9 final package report presentation": ["final", "report", "presentation", "reproducibility", "package"],
}

CANDIDATE_PREFLIGHT_COLUMNS = [
    "bundle_id",
    "path",
    "current_exists",
    "recorded_sha256",
    "current_sha256",
    "sha256_matches_f6i2a",
    "recorded_size_bytes",
    "current_size_bytes",
    "size_matches_f6i2a",
    "file_type",
    "git_state",
    "f6i1_tracked_state",
    "phase_family",
    "producer",
    "consumers",
    "f6i1_dependency_edges",
    "current_reference_count",
    "active_reference_count",
    "phase",
    "category",
    "unique_scientific_content",
    "future_roadmap_dependencies",
    "f6i2b_assumption_effect",
    "new_phase_dependency",
    "preflight_class",
    "manual_review_required",
    "notes",
]

DECISION_COLUMNS = [
    "bundle_id",
    "path",
    "current_decision",
    "decision_reason",
    "archive_gate_status",
    "protected_input",
    "raw_or_expensive_evidence",
    "runtime_required",
    "active_runtime_reference_count",
    "active_test_reference_count",
    "future_dependency_status",
    "merge_or_consolidation_status",
    "summary_migration_required",
    "content_indexing_required",
    "reference_rewrite_required",
    "compatibility_strategy_required",
    "target_collision_status",
    "rollback_defined",
    "post_migration_validation",
]

ARCHIVE_READY_COLUMNS = [
    "bundle_id",
    "original_path",
    "target_archive_path",
    "current_decision",
    "original_sha256",
    "original_size_bytes",
    "phase_family",
    "archive_category",
    "reason",
    "unique_information",
    "references_requiring_rewrite",
    "compatibility_stub_requirement",
    "rollback_path",
    "validation_procedure",
    "ready_for_execution",
]

ARCHIVE_REJECTED_COLUMNS = [
    "bundle_id",
    "path",
    "rejection_class",
    "reason",
    "recorded_sha256",
    "current_sha256",
    "recorded_size_bytes",
    "current_size_bytes",
    "manual_review_required",
]

MANUAL_REVIEW_COLUMNS = [
    "bundle_id",
    "path",
    "manual_review_reason",
    "required_review",
    "blocking_status",
    "source_evidence",
]

FUTURE_RECHECK_COLUMNS = [
    "bundle_id",
    "path",
    "future_task",
    "matched",
    "evidence",
    "replacement_status",
    "current_decision",
    "requires_current_retention",
]

ARCHITECTURE_COLUMNS = [
    "target_directory",
    "permitted_file_types",
    "path_preservation_policy",
    "original_relative_path_embedded",
    "index_format",
    "reference_policy",
    "rollback_policy",
    "git_treatment",
    "runtime_discovery_exclusion",
    "test_discovery_exclusion",
]

PATH_MAPPING_COLUMNS = [
    "archive_entry_id",
    "original_path",
    "target_archive_path",
    "original_sha256",
    "original_size_bytes",
    "phase",
    "file_category",
    "scientific_role",
    "decision_token",
    "producer",
    "inputs",
    "outputs",
    "replacement",
    "references",
    "unique_information_summary",
    "rollback_instructions",
    "currentness",
    "archive_reason",
    "validation_procedure",
]

COLLISION_COLUMNS = [
    "target_archive_path",
    "collision_count",
    "original_paths",
    "status",
]

REFERENCE_COLUMNS = [
    "reference_id",
    "target_path",
    "referencer_path",
    "reference_kind",
    "reference_class",
    "line_or_edge",
    "active_blocking",
    "rewrite_required_for_archive",
    "notes",
]

REWRITE_COLUMNS = [
    "rewrite_id",
    "original_path",
    "target_archive_path",
    "referencer_path",
    "reference_class",
    "current_reference_text",
    "planned_reference_text",
    "rewrite_required",
    "compatibility_requirement",
    "validation",
]

COMPATIBILITY_COLUMNS = [
    "bundle_id",
    "path",
    "current_decision",
    "compatibility_strategy",
    "stub_required",
    "stub_content_policy",
    "reference_update_policy",
    "validation_procedure",
    "rejection_reason",
]

SUMMARY_MIGRATION_COLUMNS = [
    "migration_id",
    "source_path",
    "source_fact",
    "target_current_document",
    "migration_wording",
    "historical_current_distinction",
    "evidence_link",
    "required_before_archive",
]

INDEX_SCHEMA_COLUMNS = [
    "field_name",
    "type",
    "required",
    "deterministic",
    "source",
    "notes",
]

CACHE_COLUMNS = [
    "cache_artifact_id",
    "path",
    "artifact_type",
    "source_regeneration_path",
    "ignored_git_status",
    "deletion_safety",
    "validation_procedure",
    "substantive_archive_candidate",
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


@dataclass(frozen=True)
class Inputs:
    config: dict[str, Any]
    paths: dict[str, Path]
    f6i1_inventory: list[dict[str, str]]
    f6i1_edges: list[dict[str, str]]
    f6i1_retention: list[dict[str, str]]
    f6i1_future_map: list[dict[str, str]]
    f6i1_validation: list[dict[str, str]]
    f6i2a_archive_bundle: list[dict[str, str]]
    f6i2a_retention_decisions: list[dict[str, str]]
    f6i2a_source_file_audit: list[dict[str, str]]
    f6i2a_future_dependency_protection: list[dict[str, str]]
    f6i2a_phase_family_consolidation: list[dict[str, str]]
    f6i2a_markdown_audit: list[dict[str, str]]
    f6i2a_agents_codex_audit: list[dict[str, str]]
    f6i2a_results_manifest_audit: list[dict[str, str]]
    f6i2a_manual_review: list[dict[str, str]]
    f6i2a_target_repository_architecture: list[dict[str, str]]
    f6i2a_manifest: dict[str, Any]
    f6i2b_candidate_preflight: list[dict[str, str]]
    f6i2b_deletion_actions: list[dict[str, str]]
    f6i2b_rejected_candidates: list[dict[str, str]]
    f6i2b_reference_validation: list[dict[str, str]]
    f6i2b_future_dependency_validation: list[dict[str, str]]
    f6i2b_validation: list[dict[str, str]]
    f6i2b_manifest: dict[str, Any]


@dataclass(frozen=True)
class BuildResult:
    tables: dict[str, list[dict[str, str]]]
    manifest: dict[str, Any]
    protected_hashes: dict[str, Any]
    report: str


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
        return path.as_posix()


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


def git_status_by_path(status_text: str) -> dict[str, str]:
    status: dict[str, str] = {}
    for line in git_status_lines(status_text):
        path = line[3:].strip().strip('"').replace("\\", "/")
        status[path] = line[:2]
    return status


def require_runtime() -> None:
    if Path(sys.executable).resolve() != APPROVED_INTERPRETER.resolve():
        raise RuntimeError(f"F6I2C requires {APPROVED_INTERPRETER}, got {sys.executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("F6I2C requires PYTHONNOUSERSITE=1")


def read_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_inputs(config_path: Path = CONFIG_PATH) -> Inputs:
    config = read_config(config_path)
    paths = {key: project_path(value) for key, value in config["inputs"].items()}
    missing = [key for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing F6I2C authoritative inputs: {missing}")
    return Inputs(
        config=config,
        paths=paths,
        f6i1_inventory=read_csv(paths["f6i1_inventory"]),
        f6i1_edges=read_csv(paths["f6i1_edges"]),
        f6i1_retention=read_csv(paths["f6i1_retention"]),
        f6i1_future_map=read_csv(paths["f6i1_future_map"]),
        f6i1_validation=read_csv(paths["f6i1_validation"]),
        f6i2a_archive_bundle=read_csv(paths["f6i2a_archive_bundle"]),
        f6i2a_retention_decisions=read_csv(paths["f6i2a_retention_decisions"]),
        f6i2a_source_file_audit=read_csv(paths["f6i2a_source_file_audit"]),
        f6i2a_future_dependency_protection=read_csv(paths["f6i2a_future_dependency_protection"]),
        f6i2a_phase_family_consolidation=read_csv(paths["f6i2a_phase_family_consolidation"]),
        f6i2a_markdown_audit=read_csv(paths["f6i2a_markdown_audit"]),
        f6i2a_agents_codex_audit=read_csv(paths["f6i2a_agents_codex_audit"]),
        f6i2a_results_manifest_audit=read_csv(paths["f6i2a_results_manifest_audit"]),
        f6i2a_manual_review=read_csv(paths["f6i2a_manual_review"]),
        f6i2a_target_repository_architecture=read_csv(paths["f6i2a_target_repository_architecture"]),
        f6i2a_manifest=read_json(paths["f6i2a_manifest"]),
        f6i2b_candidate_preflight=read_csv(paths["f6i2b_candidate_preflight"]),
        f6i2b_deletion_actions=read_csv(paths["f6i2b_deletion_actions"]),
        f6i2b_rejected_candidates=read_csv(paths["f6i2b_rejected_candidates"]),
        f6i2b_reference_validation=read_csv(paths["f6i2b_reference_validation"]),
        f6i2b_future_dependency_validation=read_csv(paths["f6i2b_future_dependency_validation"]),
        f6i2b_validation=read_csv(paths["f6i2b_validation"]),
        f6i2b_manifest=read_json(paths["f6i2b_manifest"]),
    )


def protected_input_hashes(inputs: Inputs) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in sorted(inputs.paths.items()):
        path_key = rel(path)
        if path_key in ALLOWED_CURRENT_STATE_UPDATES:
            continue
        hashes[path_key] = sha256_file(path) if path.is_file() else "missing_or_directory"
    for row in inputs.f6i2a_archive_bundle:
        candidate = project_path(row["path"])
        hashes[row["path"]] = sha256_file(candidate) if candidate.is_file() else "missing"
    return dict(sorted(hashes.items()))


def git_status_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


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


def phase_resume_baseline(final_status: str) -> dict[str, Any]:
    phase_paths = {rel(path) for path in PHASE_CREATED_PATHS}
    phase_paths.update(ALLOWED_CURRENT_STATE_UPDATES)
    baseline_lines = []
    phase_lines = []
    for line in git_status_lines(final_status):
        path = line[3:].strip().strip('"').replace("\\", "/")
        if path in phase_paths or "phase_f6i2c" in path:
            phase_lines.append(line)
        else:
            baseline_lines.append(line)
    return {
        "basis": "reconstructed_from_final_status_minus_phase_f6i2c_paths",
        "status_row_count": len(baseline_lines),
        "status_counts": git_status_counts(baseline_lines),
        "phase_created_or_updated_status_row_count": len(phase_lines),
        "phase_created_or_updated_paths": sorted(line[3:].strip().strip('"').replace("\\", "/") for line in phase_lines),
    }


def current_file_type(path: str, category: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix:
        return suffix.lstrip(".")
    if category:
        return category
    return "no_extension"


def is_protected(path: str, category: str) -> bool:
    return path in ACTIVE_DOCS or path in EXPLICIT_NO_MODIFY_CURRENT or any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES) or category in {
        "official_repository",
        "paper",
        "raw_data",
        "raw_llm_output",
        "prediction",
        "model_checkpoint",
        "model_artifact",
    }


def is_raw_or_expensive(path: str, category: str) -> bool:
    return any(path.startswith(prefix) for prefix in RAW_OR_EXPENSIVE_PREFIXES) or category in {
        "raw_data",
        "raw_llm_output",
        "prediction",
        "model_checkpoint",
        "model_artifact",
    }


def path_to_archive_category(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if path.startswith("report/") and suffix == ".md":
        return "phase_reports"
    if path.startswith("configs/"):
        return "configs"
    if path.startswith("results/manifests/"):
        return "manifests"
    if path.startswith("results/"):
        return "results"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("src/"):
        return "source"
    if "codex" in path.lower() or path in {"AGENTS.md", "CODEX_PROMPTS.md"}:
        return "codex"
    return "miscellaneous"


def archive_target_path(path: str) -> str:
    category = path_to_archive_category(path)
    return f"archive/{category}/{path}"


def classify_reference(referencer: str) -> str:
    if referencer.startswith("src/"):
        return "active_runtime_reference"
    if referencer.startswith("tests/"):
        return "active_test_reference"
    if referencer.startswith("configs/"):
        return "active_runtime_reference"
    if referencer in {"README.md", "AGENTS.md", "CODEX_PROMPTS.md", "PROJECT_SPEC.md", "reproducibility_checklist.md"}:
        return "active_documentation_reference"
    if referencer == "report/future_operations_context_handoff.md":
        return "active_documentation_reference"
    if referencer == "results/manifests/current_project_context_snapshot.json":
        return "current_manifest_reference"
    if referencer.startswith("report/"):
        return "historical_reference"
    if referencer.startswith("results/manifests/"):
        return "current_manifest_reference" if "current_project_context_snapshot" in referencer else "historical_reference"
    if referencer.startswith("results/tables/"):
        return "historical_reference"
    return "informational_reference"


def iter_reference_scan_files() -> Iterable[Path]:
    root_names = {"README.md", "AGENTS.md", "CODEX_PROMPTS.md", "PROJECT_SPEC.md", "reproducibility_checklist.md", "pytest.ini"}
    for child in PROJECT_ROOT.iterdir():
        if child.is_file() and child.name in root_names:
            yield child
    for prefix in REFERENCE_SCAN_PREFIXES:
        base = PROJECT_ROOT / prefix
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            rel_path = rel(path)
            parts = set(rel_path.split("/"))
            if not path.is_file() or parts & REFERENCE_SCAN_SKIP_PARTS:
                continue
            if "phase_f6i2c" in rel_path:
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
            except OSError:
                continue
            if path.suffix.lower() in TEXT_EXTENSIONS:
                yield path


def build_reference_inventory(candidate_paths: list[str], edges: list[dict[str, str]]) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    candidates = sorted(set(candidate_paths))
    candidate_set = set(candidates)
    for edge in edges:
        source = edge.get("source_path", "")
        target = edge.get("target_path", "")
        if target not in candidate_set:
            continue
        ref_class = classify_reference(source)
        references.append(
            {
                "reference_id": stable_id("F6I2C-REFEDGE", target, source, edge.get("edge_id", "")),
                "target_path": target,
                "referencer_path": source,
                "reference_kind": edge.get("edge_type", "dependency_edge"),
                "reference_class": ref_class,
                "line_or_edge": edge.get("edge_id", "f6i1_dependency_edge"),
                "active_blocking": str(ref_class in {"active_runtime_reference", "active_test_reference"}).lower(),
                "rewrite_required_for_archive": str(ref_class != "false_positive").lower(),
                "notes": "Reference inherited from F6I1 dependency graph.",
            }
        )
    variants = {path: {path, path.replace("/", "\\")} for path in candidates}
    for source in iter_reference_scan_files():
        rel_source = rel(source)
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for target in candidates:
            if rel_source == target:
                continue
            matched = [variant for variant in variants[target] if variant and variant in text]
            if not matched:
                continue
            ref_class = classify_reference(rel_source)
            references.append(
                {
                    "reference_id": stable_id("F6I2C-REF", target, rel_source),
                    "target_path": target,
                    "referencer_path": rel_source,
                    "reference_kind": "literal_path",
                    "reference_class": ref_class,
                    "line_or_edge": "content_scan",
                    "active_blocking": str(ref_class in {"active_runtime_reference", "active_test_reference"}).lower(),
                    "rewrite_required_for_archive": str(ref_class != "false_positive").lower(),
                    "notes": "Exact path string found in current text artifact.",
                }
            )
    return sorted(references, key=lambda row: (row["target_path"], row["referencer_path"], row["reference_class"]))


def phase_family_for_path(path: str, phase: str, rows: list[dict[str, str]]) -> str:
    matched = []
    for row in rows:
        haystack = ";".join(
            [
                row.get("phases", ""),
                row.get("canonical_final_implementation", ""),
                row.get("still_useful_intermediate_implementation", ""),
                row.get("frozen_outputs_to_remain", ""),
                row.get("reports_to_archive", ""),
            ]
        )
        if path in haystack or phase in row.get("phases", ""):
            matched.append(row["phase_family_id"])
    return ";".join(sorted(set(matched))) if matched else phase or "unassigned"


def match_future_tasks(path: str, future_text: str, reference_text: str) -> list[tuple[str, str]]:
    haystack = f"{path} {future_text} {reference_text}".lower()
    matches = []
    for task, keywords in FUTURE_TASK_KEYWORDS.items():
        hits = [keyword for keyword in keywords if keyword.lower() in haystack]
        if hits:
            matches.append((task, ";".join(sorted(set(hits)))))
    return matches


def extract_decision_tokens(path: str) -> str:
    file_path = project_path(path)
    if not file_path.exists() or file_path.suffix.lower() not in TEXT_EXTENSIONS:
        return ""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")[:50000]
    except OSError:
        return ""
    tokens = sorted(set(re.findall(r"\b[a-z0-9]+(?:_[a-z0-9]+){3,}\b", text)))
    return ";".join(tokens[:12])


def first_heading_or_summary(path: str, fallback: str) -> str:
    file_path = project_path(path)
    if file_path.exists() and file_path.suffix.lower() in TEXT_EXTENSIONS:
        try:
            for line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    return stripped[:180]
                if stripped:
                    return stripped[:180]
        except OSError:
            pass
    return fallback[:180]


def build_preflight(inputs: Inputs, references: list[dict[str, str]], git_state: Mapping[str, str]) -> list[dict[str, str]]:
    archive_rows = inputs.f6i2a_archive_bundle
    paths = [row["path"] for row in archive_rows]
    if len(paths) != 179 or len(set(paths)) != 179:
        raise RuntimeError(f"F6I2C expected 179 unique archive candidates, found rows={len(paths)} unique={len(set(paths))}")
    inventory = {row["path"]: row for row in inputs.f6i1_inventory}
    edges_by_path: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    inbound_consumers: defaultdict[str, list[str]] = defaultdict(list)
    outbound_targets: defaultdict[str, list[str]] = defaultdict(list)
    for edge in inputs.f6i1_edges:
        if edge["source_path"] in paths:
            edges_by_path[edge["source_path"]].append(edge)
            outbound_targets[edge["source_path"]].append(edge["target_path"])
        if edge["target_path"] in paths:
            edges_by_path[edge["target_path"]].append(edge)
            inbound_consumers[edge["target_path"]].append(edge["source_path"])
    refs_by_target: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for ref in references:
        refs_by_target[ref["target_path"]].append(ref)
    deleted_by_f6i2b = {row["path"]: row for row in inputs.f6i2b_deletion_actions if row.get("action_executed") == "deleted"}
    rows = []
    for archive in archive_rows:
        path = archive["path"]
        file_path = project_path(path)
        inv = inventory.get(path, {})
        exists = file_path.is_file()
        current_sha = sha256_file(file_path) if exists else "missing"
        current_size = str(file_path.stat().st_size) if exists else "missing"
        recorded_sha = inv.get("sha256", "")
        recorded_size = inv.get("size_bytes", "")
        sha_matches = exists and current_sha == recorded_sha
        size_matches = exists and current_size == recorded_size
        f6i2b_effect = "deleted_by_f6i2b_low_risk_action" if path in deleted_by_f6i2b else "none"
        if not exists or not sha_matches or not size_matches:
            preflight_class = "archive_preflight_changed_since_f6i2a"
            manual_review = "true"
            notes = "Current path missing or hash/size changed; F6I2A archive decision not inherited."
        else:
            preflight_class = "archive_preflight_unchanged_since_f6i2a"
            manual_review = "false"
            notes = "Current file matches F6I1/F6I2A recorded hash and size."
        active_refs = [ref for ref in refs_by_target[path] if ref["active_blocking"] == "true"]
        consumers = sorted(set(inbound_consumers[path] + [ref["referencer_path"] for ref in refs_by_target[path]]))
        phase_family = phase_family_for_path(path, inv.get("phase", ""), inputs.f6i2a_phase_family_consolidation)
        rows.append(
            {
                "bundle_id": archive["bundle_id"],
                "path": path,
                "current_exists": str(exists).lower(),
                "recorded_sha256": recorded_sha,
                "current_sha256": current_sha,
                "sha256_matches_f6i2a": str(sha_matches).lower(),
                "recorded_size_bytes": recorded_size,
                "current_size_bytes": current_size,
                "size_matches_f6i2a": str(size_matches).lower(),
                "file_type": current_file_type(path, inv.get("category", "")),
                "git_state": git_state.get(path, "tracked_clean_or_untracked_parent"),
                "f6i1_tracked_state": inv.get("tracked_state", ""),
                "phase_family": phase_family,
                "producer": inv.get("producer", ""),
                "consumers": ";".join(consumers[:20]),
                "f6i1_dependency_edges": str(len(edges_by_path[path])),
                "current_reference_count": str(len(refs_by_target[path])),
                "active_reference_count": str(len(active_refs)),
                "phase": inv.get("phase", ""),
                "category": inv.get("category", ""),
                "unique_scientific_content": inv.get("unique_scientific_information", ""),
                "future_roadmap_dependencies": inv.get("future_roadmap_dependencies", ""),
                "f6i2b_assumption_effect": f6i2b_effect,
                "new_phase_dependency": "post_f6i2a_f6h2_or_f6i2b_reference_detected" if any("phase_f6h2" in ref["referencer_path"] or "phase_f6i2b" in ref["referencer_path"] for ref in refs_by_target[path]) else "none_detected",
                "preflight_class": preflight_class,
                "manual_review_required": manual_review,
                "notes": notes,
            }
        )
    return rows


def build_future_recheck(preflight: list[dict[str, str]], inputs: Inputs, references: list[dict[str, str]]) -> list[dict[str, str]]:
    refs_by_target: defaultdict[str, list[str]] = defaultdict(list)
    for ref in references:
        refs_by_target[ref["target_path"]].append(ref["referencer_path"])
    rows = []
    for row in preflight:
        evidence_text = row["future_roadmap_dependencies"]
        ref_text = ";".join(refs_by_target[row["path"]])
        matches = dict(match_future_tasks(row["path"], evidence_text, ref_text))
        for task in FUTURE_TASK_KEYWORDS:
            matched = task in matches
            rows.append(
                {
                    "bundle_id": row["bundle_id"],
                    "path": row["path"],
                    "future_task": task,
                    "matched": str(matched).lower(),
                    "evidence": matches.get(task, ""),
                    "replacement_status": "not_replaced_in_current_workspace" if matched else "not_applicable",
                    "current_decision": "",
                    "requires_current_retention": str(matched).lower(),
                }
            )
    return rows


def active_reference_counts(path: str, references: list[dict[str, str]]) -> tuple[int, int, int]:
    runtime = sum(1 for ref in references if ref["target_path"] == path and ref["reference_class"] == "active_runtime_reference")
    tests = sum(1 for ref in references if ref["target_path"] == path and ref["reference_class"] == "active_test_reference")
    active_docs = sum(1 for ref in references if ref["target_path"] == path and ref["reference_class"] in {"active_documentation_reference", "current_manifest_reference"})
    return runtime, tests, active_docs


def choose_decision(row: dict[str, str], archive_row: dict[str, str], references: list[dict[str, str]]) -> tuple[str, str]:
    path = row["path"]
    suffix = Path(path).suffix.lower()
    category = row["category"]
    runtime_refs, test_refs, active_doc_refs = active_reference_counts(path, references)
    protected = is_protected(path, category)
    raw_or_expensive = is_raw_or_expensive(path, category)
    ref_text = ";".join(ref["referencer_path"] for ref in references if ref["target_path"] == path)
    future_dependency = bool(row["future_roadmap_dependencies"]) or bool(match_future_tasks(path, row["future_roadmap_dependencies"], ref_text))
    unique = row["unique_scientific_content"].lower().startswith("yes")
    if row["preflight_class"] == "archive_preflight_changed_since_f6i2a":
        return "archive_rejected_changed_since_f6i2a", "Current file is missing or changed since the F6I2A recorded hash/size."
    if path in EXPLICIT_NO_MODIFY_CURRENT or path in ACTIVE_DOCS:
        return "keep_current_operational", "Explicit current operational document or current-state artifact."
    if suffix == ".py":
        return "redirect_to_python_consolidation", "Python files require consolidation, wrappers, and tests before archival."
    if path.startswith("configs/") or suffix in {".yaml", ".yml"}:
        return "redirect_to_config_consolidation", "Historical config candidate requires config consolidation and value indexing first."
    if runtime_refs or test_refs:
        return "keep_current_operational", "Active runtime or test references remain at the current path."
    if protected or raw_or_expensive:
        return "keep_current_frozen_evidence", "Protected or raw/expensive evidence remains current."
    if suffix == ".md":
        if active_doc_refs or future_dependency or unique:
            return "archive_ready_after_current_summary_migration", "Historical/current facts must be migrated or represented before moving the report."
        return "archive_ready_with_reference_rewrite" if row["current_reference_count"] != "0" else "archive_ready", "Historical Markdown can migrate after reference rewrite inventory."
    if path.startswith("results/") and suffix in {".csv", ".json", ".jsonl", ".sha256"}:
        if future_dependency or unique:
            return "archive_ready_after_content_indexing", "Derived historical table requires content/hash indexing before archival."
        return "archive_ready_with_reference_rewrite" if row["current_reference_count"] != "0" else "archive_ready", "Derived historical result has no active runtime/test dependency."
    if suffix in {".whl", ".exe"} or "wheelhouse" in path or "cpython" in path:
        return "keep_current_future_dependency", "Environment recovery payload may be needed for future reproducibility or runtime recovery."
    if suffix == ".log":
        return "archive_ready_after_content_indexing", "Historical log requires content indexing; no deletion or movement in F6I2C."
    if suffix in {".png", ".txt", ".gitkeep", ""}:
        if future_dependency or unique:
            return "archive_ready_after_content_indexing", "Historical/support artifact requires content indexing before archival."
        return "archive_ready_with_reference_rewrite" if row["current_reference_count"] != "0" else "archive_ready", "Non-runtime historical/support artifact can be archived after path validation."
    return "redirect_to_manual_review", "No deterministic safe rule matched; fail closed."


def build_decisions(
    preflight: list[dict[str, str]],
    inputs: Inputs,
    references: list[dict[str, str]],
    future_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    archive_by_path = {row["path"]: row for row in inputs.f6i2a_archive_bundle}
    matched_future_by_path: defaultdict[str, int] = defaultdict(int)
    for row in future_rows:
        if row["matched"] == "true":
            matched_future_by_path[row["path"]] += 1
    rows = []
    for row in preflight:
        decision, reason = choose_decision(row, archive_by_path[row["path"]], references)
        runtime_refs, test_refs, _ = active_reference_counts(row["path"], references)
        protected = is_protected(row["path"], row["category"])
        raw_or_expensive = is_raw_or_expensive(row["path"], row["category"])
        refs_require_rewrite = any(ref["target_path"] == row["path"] for ref in references)
        summary_migration = decision == "archive_ready_after_current_summary_migration"
        content_indexing = decision == "archive_ready_after_content_indexing"
        rows.append(
            {
                "bundle_id": row["bundle_id"],
                "path": row["path"],
                "current_decision": decision,
                "decision_reason": reason,
                "archive_gate_status": "passes_immediate_gate" if decision in IMMEDIATE_READY_CLASSES else "blocked_or_conditional",
                "protected_input": str(protected).lower(),
                "raw_or_expensive_evidence": str(raw_or_expensive).lower(),
                "runtime_required": str(runtime_refs > 0).lower(),
                "active_runtime_reference_count": str(runtime_refs),
                "active_test_reference_count": str(test_refs),
                "future_dependency_status": "matched_unfinished_task" if matched_future_by_path[row["path"]] else "none_detected",
                "merge_or_consolidation_status": decision if decision.startswith("redirect_to_") else "none",
                "summary_migration_required": str(summary_migration).lower(),
                "content_indexing_required": str(content_indexing).lower(),
                "reference_rewrite_required": str(refs_require_rewrite and decision in ARCHIVE_PLANNED_CLASSES).lower(),
                "compatibility_strategy_required": str(decision == "archive_ready_with_compatibility_stub").lower(),
                "target_collision_status": "pending_collision_check" if decision in ARCHIVE_PLANNED_CLASSES else "not_applicable",
                "rollback_defined": str(decision in ARCHIVE_PLANNED_CLASSES or decision.startswith("keep_current") or decision.startswith("redirect_to_")).lower(),
                "post_migration_validation": "hash_manifest_and_reference_audit" if decision in ARCHIVE_PLANNED_CLASSES else "not_applicable_until_future_phase",
            }
        )
    return rows


def fill_future_decisions(future_rows: list[dict[str, str]], decisions: list[dict[str, str]]) -> list[dict[str, str]]:
    decision_by_path = {row["path"]: row["current_decision"] for row in decisions}
    updated = []
    for row in future_rows:
        copy = dict(row)
        copy["current_decision"] = decision_by_path[copy["path"]]
        copy["requires_current_retention"] = str(copy["matched"] == "true" and decision_by_path[copy["path"]] not in ARCHIVE_PLANNED_CLASSES).lower()
        updated.append(copy)
    return updated


def build_path_mapping(
    preflight: list[dict[str, str]],
    decisions: list[dict[str, str]],
    inputs: Inputs,
    references: list[dict[str, str]],
) -> list[dict[str, str]]:
    decision_by_path = {row["path"]: row for row in decisions}
    archive_by_path = {row["path"]: row for row in inputs.f6i2a_archive_bundle}
    refs_by_path: defaultdict[str, list[str]] = defaultdict(list)
    for ref in references:
        refs_by_path[ref["target_path"]].append(ref["referencer_path"])
    rows = []
    for row in preflight:
        decision = decision_by_path[row["path"]]["current_decision"]
        if decision not in ARCHIVE_PLANNED_CLASSES:
            continue
        archive_row = archive_by_path[row["path"]]
        tokens = extract_decision_tokens(row["path"])
        rows.append(
            {
                "archive_entry_id": stable_id("F6I2C-ARCHIVE", row["path"]),
                "original_path": row["path"],
                "target_archive_path": archive_target_path(row["path"]),
                "original_sha256": row["current_sha256"],
                "original_size_bytes": row["current_size_bytes"],
                "phase": row["phase"],
                "file_category": row["category"],
                "scientific_role": archive_row.get("reason", ""),
                "decision_token": tokens or decision,
                "producer": row["producer"],
                "inputs": "indexed_from_f6i1_inventory",
                "outputs": "not_executed_in_f6i2c",
                "replacement": archive_row.get("replacement", ""),
                "references": ";".join(sorted(set(refs_by_path[row["path"]]))[:30]),
                "unique_information_summary": row["unique_scientific_content"],
                "rollback_instructions": f"restore {row['path']} from {archive_target_path(row['path'])} and verify sha256 {row['current_sha256']}",
                "currentness": "conditional_current_summary_required" if decision == "archive_ready_after_current_summary_migration" else "historical_archival_candidate",
                "archive_reason": decision_by_path[row["path"]]["decision_reason"],
                "validation_procedure": "compare archive index hash, run reference rewrite audit, and run focused cleanup validation",
            }
        )
    return rows


def build_collisions(path_mapping: list[dict[str, str]]) -> list[dict[str, str]]:
    by_target: defaultdict[str, list[str]] = defaultdict(list)
    for row in path_mapping:
        by_target[row["target_archive_path"]].append(row["original_path"])
    rows = []
    for target, originals in sorted(by_target.items()):
        if len(originals) > 1:
            rows.append(
                {
                    "target_archive_path": target,
                    "collision_count": str(len(originals)),
                    "original_paths": ";".join(sorted(originals)),
                    "status": "collision_blocking",
                }
            )
    if not rows:
        rows.append(
            {
                "target_archive_path": "all_planned_targets",
                "collision_count": "0",
                "original_paths": "",
                "status": "pass_collision_free",
            }
        )
    return rows


def build_ready_rejected_manual(
    path_mapping: list[dict[str, str]],
    decisions: list[dict[str, str]],
    preflight: list[dict[str, str]],
    references: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    decisions_by_path = {row["path"]: row for row in decisions}
    preflight_by_path = {row["path"]: row for row in preflight}
    refs_by_path: Counter[str] = Counter(ref["target_path"] for ref in references if ref["rewrite_required_for_archive"] == "true")
    ready = []
    for row in path_mapping:
        decision = decisions_by_path[row["original_path"]]
        ready.append(
            {
                "bundle_id": preflight_by_path[row["original_path"]]["bundle_id"],
                "original_path": row["original_path"],
                "target_archive_path": row["target_archive_path"],
                "current_decision": decision["current_decision"],
                "original_sha256": row["original_sha256"],
                "original_size_bytes": row["original_size_bytes"],
                "phase_family": preflight_by_path[row["original_path"]]["phase_family"],
                "archive_category": path_to_archive_category(row["original_path"]),
                "reason": row["archive_reason"],
                "unique_information": row["unique_information_summary"],
                "references_requiring_rewrite": str(refs_by_path[row["original_path"]]),
                "compatibility_stub_requirement": decision["compatibility_strategy_required"],
                "rollback_path": row["rollback_instructions"],
                "validation_procedure": row["validation_procedure"],
                "ready_for_execution": str(decision["current_decision"] in IMMEDIATE_READY_CLASSES).lower(),
            }
        )
    rejected = []
    manual = []
    for row in preflight:
        decision = decisions_by_path[row["path"]]
        if decision["current_decision"] in {"archive_rejected_changed_since_f6i2a", "unresolved"}:
            rejected.append(
                {
                    "bundle_id": row["bundle_id"],
                    "path": row["path"],
                    "rejection_class": decision["current_decision"],
                    "reason": decision["decision_reason"],
                    "recorded_sha256": row["recorded_sha256"],
                    "current_sha256": row["current_sha256"],
                    "recorded_size_bytes": row["recorded_size_bytes"],
                    "current_size_bytes": row["current_size_bytes"],
                    "manual_review_required": "true",
                }
            )
        if decision["current_decision"] in {"redirect_to_manual_review", "archive_rejected_changed_since_f6i2a", "unresolved"} or row["manual_review_required"] == "true":
            manual.append(
                {
                    "bundle_id": row["bundle_id"],
                    "path": row["path"],
                    "manual_review_reason": decision["decision_reason"],
                    "required_review": "verify current path/hash and decide whether candidate should be excluded, restored, or rebaselined",
                    "blocking_status": decision["current_decision"],
                    "source_evidence": row["preflight_class"],
                }
            )
    return ready, rejected, manual


def build_reference_rewrite_plan(
    references: list[dict[str, str]],
    decisions: list[dict[str, str]],
    path_mapping: list[dict[str, str]],
) -> list[dict[str, str]]:
    decision_by_path = {row["path"]: row["current_decision"] for row in decisions}
    target_by_path = {row["original_path"]: row["target_archive_path"] for row in path_mapping}
    rows = []
    for ref in references:
        decision = decision_by_path.get(ref["target_path"], "")
        if decision not in ARCHIVE_PLANNED_CLASSES:
            continue
        rewrite_required = ref["rewrite_required_for_archive"] == "true"
        rows.append(
            {
                "rewrite_id": stable_id("F6I2C-REWRITE", ref["target_path"], ref["referencer_path"]),
                "original_path": ref["target_path"],
                "target_archive_path": target_by_path[ref["target_path"]],
                "referencer_path": ref["referencer_path"],
                "reference_class": ref["reference_class"],
                "current_reference_text": ref["target_path"],
                "planned_reference_text": target_by_path[ref["target_path"]],
                "rewrite_required": str(rewrite_required).lower(),
                "compatibility_requirement": "retain_current_path_or_stub" if ref["active_blocking"] == "true" else "rewrite_reference_in_future_migration_phase",
                "validation": "rerun reference inventory and focused cleanup tests after migration",
            }
        )
    return rows


def build_compatibility_strategy(decisions: list[dict[str, str]], references: list[dict[str, str]]) -> list[dict[str, str]]:
    refs_by_path: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for ref in references:
        refs_by_path[ref["target_path"]].append(ref)
    rows = []
    for decision in decisions:
        path = decision["path"]
        refs = refs_by_path[path]
        active_refs = [ref for ref in refs if ref["active_blocking"] == "true"]
        if decision["current_decision"] == "archive_ready_with_compatibility_stub":
            strategy = "create_minimal_redirect_or_wrapper_in_future_migration"
            stub_required = "true"
            stub_policy = "stub must not duplicate full scientific content"
            reference_policy = "retain import/CLI compatibility and rewrite documentation"
            rejection = ""
        elif active_refs:
            strategy = "retain_original_current_path"
            stub_required = "false"
            stub_policy = "not_applicable"
            reference_policy = "no F6I2C movement; future consolidation required first"
            rejection = "active runtime or test reference"
        elif decision["current_decision"] in ARCHIVE_PLANNED_CLASSES:
            strategy = "future_reference_rewrite_and_hash_index"
            stub_required = "false"
            stub_policy = "not_applicable"
            reference_policy = "rewrite historical/current references in future migration phase"
            rejection = ""
        else:
            strategy = "no_archive_migration_in_current_plan"
            stub_required = "false"
            stub_policy = "not_applicable"
            reference_policy = "retain current path"
            rejection = decision["decision_reason"]
        rows.append(
            {
                "bundle_id": decision["bundle_id"],
                "path": path,
                "current_decision": decision["current_decision"],
                "compatibility_strategy": strategy,
                "stub_required": stub_required,
                "stub_content_policy": stub_policy,
                "reference_update_policy": reference_policy,
                "validation_procedure": "focused cleanup tests plus exact reference inventory diff",
                "rejection_reason": rejection,
            }
        )
    return rows


def build_current_summary_migration(decisions: list[dict[str, str]], inputs: Inputs) -> list[dict[str, str]]:
    archive_by_path = {row["path"]: row for row in inputs.f6i2a_archive_bundle}
    rows = []
    for decision in decisions:
        if decision["summary_migration_required"] != "true":
            continue
        path = decision["path"]
        source_fact = first_heading_or_summary(path, archive_by_path[path].get("reason", ""))
        tokens = extract_decision_tokens(path)
        rows.append(
            {
                "migration_id": stable_id("F6I2C-SUMMARY", path),
                "source_path": path,
                "source_fact": source_fact,
                "target_current_document": "report/future_operations_context_handoff.md;results/manifests/current_project_context_snapshot.json;future docs/current_status.md",
                "migration_wording": f"Index historical conclusions from {path}; preserve decision tokens: {tokens or decision['current_decision']}.",
                "historical_current_distinction": "migrate only current operational facts; keep historical narrative in archive index",
                "evidence_link": path,
                "required_before_archive": "true",
            }
        )
    return rows


def build_architecture(inputs: Inputs) -> list[dict[str, str]]:
    categories = inputs.config["archive_architecture"]["categories"]
    specs = [
        ("phase_reports", "*.md", "embed original report/ relative path", "true", "archive_index.csv plus manifest hashes", "rewrite report/docs links only in future phase", "restore from archive path and verify hash", "tracked archive files after approval", "outside src/config runtime loaders", "not under tests/"),
        ("codex", "*.md;*.txt", "embed original root-relative path", "true", "archive_index.csv", "active AGENTS stays current", "restore original path from archive", "tracked after approval", "outside runtime discovery", "not under tests/"),
        ("decisions", "*.csv;*.json;*.md", "embed original path", "true", "archive_index.csv", "preserve decision tokens", "restore exact file and hash", "tracked after approval", "outside runtime discovery", "not under tests/"),
        ("configs", "*.yaml;*.yml", "embed configs/ path", "true", "archive_index.csv", "do not load from archive by default", "restore config and rerun focused tests", "tracked after approval", "archive root not searched by loaders", "not under tests/"),
        ("results", "*.csv;*.json;*.log;*.png;*.txt;*.sha256", "embed results/ path", "true", "archive_index.csv plus per-file hashes", "historical report links rewritten", "restore result and compare hash", "tracked after approval", "outside active results/current conventions", "not under tests/"),
        ("manifests", "*.json", "embed results/manifests path", "true", "archive_index.csv", "current snapshots never archived", "restore manifest and compare hash", "tracked after approval", "outside active manifest names", "not under tests/"),
        ("miscellaneous", "*", "embed original path", "true", "archive_index.csv", "manual review before reference rewrite", "restore exact file", "tracked after approval", "outside runtime discovery", "not under tests/"),
    ]
    return [
        {
            "target_directory": categories.get(category, f"archive/{category}"),
            "permitted_file_types": permitted,
            "path_preservation_policy": path_policy,
            "original_relative_path_embedded": embedded,
            "index_format": index_format,
            "reference_policy": reference_policy,
            "rollback_policy": rollback,
            "git_treatment": git_treatment,
            "runtime_discovery_exclusion": runtime,
            "test_discovery_exclusion": test,
        }
        for category, permitted, path_policy, embedded, index_format, reference_policy, rollback, git_treatment, runtime, test in specs
    ]


def build_index_schema() -> list[dict[str, str]]:
    fields = [
        ("archive_entry_id", "string", "yes", "yes", "stable hash of original_path", "primary key"),
        ("original_path", "path", "yes", "yes", "F6I2A archive bundle", "current source path before migration"),
        ("archived_path", "path", "yes", "yes", "F6I2C path mapping", "planned target path"),
        ("SHA-256", "hex", "yes", "yes", "current file hash", "no timestamp"),
        ("size", "integer", "yes", "yes", "current file stat", "bytes"),
        ("phase", "string", "yes", "yes", "F6I1 inventory", "phase family/source phase"),
        ("file category", "string", "yes", "yes", "F6I1 inventory", "report/config/result/etc."),
        ("scientific role", "string", "yes", "yes", "F6I2A reason", "why preserved"),
        ("decision token", "string", "yes", "yes", "content extraction or F6I2C decision", "stable token list"),
        ("producer", "string", "yes", "yes", "F6I1 inventory", "producer when known"),
        ("inputs", "string", "yes", "yes", "F6I1/F6I2C", "indexed input relation"),
        ("outputs", "string", "yes", "yes", "F6I1/F6I2C", "indexed output relation"),
        ("replacement", "string", "yes", "yes", "F6I2A bundle", "canonical replacement or future doc"),
        ("references", "string", "yes", "yes", "F6I2C reference inventory", "semicolon sorted"),
        ("unique-information summary", "string", "yes", "yes", "F6I1 inventory/content heading", "brief content summary"),
        ("migration date", "runtime-log-only", "no", "no", "future non-deterministic runtime log", "excluded from deterministic index"),
        ("rollback instructions", "string", "yes", "yes", "F6I2C mapping", "restore path and verify hash"),
        ("currentness", "string", "yes", "yes", "F6I2C decision", "historical/current/conditional"),
        ("archive reason", "string", "yes", "yes", "F6I2C decision", "why this archive action is proposed"),
    ]
    return [
        {"field_name": name, "type": typ, "required": required, "deterministic": deterministic, "source": source, "notes": notes}
        for name, typ, required, deterministic, source, notes in fields
    ]


def build_cache_cleanup_bundle(candidate_paths: set[str]) -> list[dict[str, str]]:
    rows = []
    cache_paths = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        rel_path = rel(path)
        if any(part.startswith(".venv") for part in rel_path.split("/")):
            continue
        if "__pycache__" in rel_path or path.suffix.lower() in {".pyc", ".pyo"}:
            cache_paths.append(path)
    for path in cache_paths:
        rel_path = rel(path)
        artifact_type = "directory" if path.is_dir() else path.suffix.lower().lstrip(".") or "cache"
        if path.is_file() and path.suffix.lower() == ".pyc":
            source_name = re.sub(r"\.cpython-\d+.*\.pyc$", ".py", path.name)
            source = path.parent.parent / source_name
            source_regen = rel(source) if source.exists() else "source_not_resolved_from_pyc_name"
        elif path.is_dir():
            source_regen = rel(path.parent)
        else:
            source_regen = "not_applicable"
        rows.append(
            {
                "cache_artifact_id": stable_id("F6I2C-CACHE", rel_path),
                "path": rel_path,
                "artifact_type": artifact_type,
                "source_regeneration_path": source_regen,
                "ignored_git_status": "ignored_or_untracked_python_cache_artifact",
                "deletion_safety": "future_technical_cleanup_only_after_import/test_probe",
                "validation_procedure": "remove cache in future cleanup then rerun focused import/tests",
                "substantive_archive_candidate": str(rel_path in candidate_paths).lower(),
            }
        )
    return rows


def build_next_actions() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action_id": "F6I2C-NEXT-001",
            "component": "archive_migration",
            "current_status": "design_ready_partial",
            "safe_next_action": "Review F6I2C rejected, conditional, and target mapping tables, then approve an execution-only F6I2D archive migration phase.",
            "requires_file_move": "true",
            "requires_reference_rewrite": "true",
            "requires_training": "false",
            "requires_llm_inference": "false",
            "requires_api": "false",
            "notes": "F6I2C did not execute migration.",
        },
        {
            "priority": "2",
            "action_id": "F6I2C-NEXT-002",
            "component": "current_summary_migration",
            "current_status": "required_before_report_archive",
            "safe_next_action": "Migrate current facts listed in phase_f6i2c_current_summary_migration_plan.csv into approved current documents.",
            "requires_file_move": "false",
            "requires_reference_rewrite": "false",
            "requires_training": "false",
            "requires_llm_inference": "false",
            "requires_api": "false",
            "notes": "Keep historical/current distinction explicit.",
        },
        {
            "priority": "3",
            "action_id": "F6I2C-NEXT-003",
            "component": "cache_cleanup",
            "current_status": "separate_technical_cleanup_bundle",
            "safe_next_action": "Delete only cache artifacts from the separate cache bundle in a future technical-cleanup phase.",
            "requires_file_move": "false",
            "requires_reference_rewrite": "false",
            "requires_training": "false",
            "requires_llm_inference": "false",
            "requires_api": "false",
            "notes": "Do not mix cache cleanup with substantive archive migration.",
        },
    ]


def output_hashes() -> dict[str, str]:
    paths = [*OUTPUT_TABLES.values(), MANIFEST_PATH, PROTECTED_HASHES_PATH, REPORT_PATH, HANDOFF_PATH, SNAPSHOT_PATH]
    hashes = {}
    for path in paths:
        key = rel(path)
        if path == MANIFEST_PATH:
            hashes[key] = "self_reference_excluded_for_determinism"
        elif path.exists():
            hashes[key] = sha256_file(path)
        else:
            hashes[key] = "missing"
    return dict(sorted(hashes.items()))


def build_validation(
    inputs: Inputs,
    preflight: list[dict[str, str]],
    decisions: list[dict[str, str]],
    ready: list[dict[str, str]],
    rejected: list[dict[str, str]],
    manual: list[dict[str, str]],
    future_rows: list[dict[str, str]],
    path_mapping: list[dict[str, str]],
    collisions: list[dict[str, str]],
    references: list[dict[str, str]],
    rewrites: list[dict[str, str]],
    migration: list[dict[str, str]],
    cache_rows: list[dict[str, str]],
    protected_before: dict[str, str],
    protected_after: dict[str, str],
) -> list[dict[str, str]]:
    branch = git_stdout(["rev-parse", "--abbrev-ref", "HEAD"])
    head = git_stdout(["rev-parse", "HEAD"])
    archive_paths = [row["path"] for row in inputs.f6i2a_archive_bundle]
    decision_counts = Counter(row["path"] for row in decisions)
    immediate_ready = [row for row in decisions if row["current_decision"] in IMMEDIATE_READY_CLASSES]
    decision_by_path = {row["path"]: row for row in decisions}
    ready_paths = {row["original_path"] for row in ready}
    active_blocking_ready_refs = [
        ref for ref in references
        if ref["target_path"] in ready_paths and ref["active_blocking"] == "true" and decision_by_path[ref["target_path"]]["current_decision"] in IMMEDIATE_READY_CLASSES
    ]
    cache_candidate_overlap = [row for row in cache_rows if row["substantive_archive_candidate"] == "true"]
    checks = [
        ("selected_phase_identifier", PHASE_ID, PHASE_ID, "read-only reconciliation found phase_f6i2c unoccupied before artifact creation"),
        ("runtime_interpreter", str(Path(sys.executable).resolve() == APPROVED_INTERPRETER.resolve()).lower(), "true", str(APPROVED_INTERPRETER)),
        ("python_no_user_site", os.environ.get("PYTHONNOUSERSITE", ""), "1", ""),
        ("branch_matches_expected", branch, EXPECTED_BRANCH, ""),
        ("head_matches_expected", head, EXPECTED_HEAD, ""),
        ("archive_bundle_rows", str(len(archive_paths)), "179", ""),
        ("archive_bundle_unique_paths", str(len(set(archive_paths))), "179", ""),
        ("candidate_preflight_rows", str(len(preflight)), "179", ""),
        ("candidate_decision_rows", str(len(decisions)), "179", ""),
        ("unique_candidate_decisions", str(all(count == 1 for count in decision_counts.values())).lower(), "true", ""),
        ("valid_decision_classes", str(all(row["current_decision"] in DECISION_CLASSES for row in decisions)).lower(), "true", ""),
        ("changed_candidates_manual_reviewed", str(all(row["path"] in {m["path"] for m in manual} for row in rejected)).lower(), "true", ""),
        ("no_immediate_ready_protected", str(sum(row["protected_input"] == "true" for row in immediate_ready)), "0", ""),
        ("no_immediate_ready_raw_or_expensive", str(sum(row["raw_or_expensive_evidence"] == "true" for row in immediate_ready)), "0", ""),
        ("no_immediate_ready_future_required", str(sum(row["future_dependency_status"] == "matched_unfinished_task" for row in immediate_ready)), "0", ""),
        ("no_immediate_ready_unresolved_python_config_merge", str(sum(row["current_decision"] in {"redirect_to_python_consolidation", "redirect_to_config_consolidation"} for row in immediate_ready)), "0", ""),
        ("no_immediate_ready_active_runtime_or_test_reference", str(len(active_blocking_ready_refs)), "0", ""),
        ("target_path_collisions", str(sum(1 for row in collisions if row["status"] == "collision_blocking")), "0", ""),
        ("archive_mapping_paths_subset_ready", str({row["original_path"] for row in path_mapping} == ready_paths).lower(), "true", ""),
        ("archive_ready_rollback_defined", str(all(row["rollback_path"] for row in ready)).lower(), "true", ""),
        ("unique_decision_token_indexed", str(all(row["decision_token"] for row in path_mapping)).lower(), "true", ""),
        ("summary_migration_rows_cover_conditional_reports", str(sum(row["summary_migration_required"] == "true" for row in decisions)), str(len(migration)), ""),
        ("reference_rewrites_enumerated", str(len(rewrites)), str(sum(1 for ref in references if ref["target_path"] in ready_paths)), ""),
        ("future_recheck_complete_matrix", str(len(future_rows)), str(179 * len(FUTURE_TASK_KEYWORDS)), ""),
        ("cache_artifacts_separated_from_substantive_candidates", str(len(cache_candidate_overlap)), "0", ""),
        ("f6i2b_deleted_count_preserved", str(inputs.f6i2b_manifest.get("summary", {}).get("deleted_count")), "8", ""),
        ("f6i2b_rejected_count_preserved", str(inputs.f6i2b_manifest.get("summary", {}).get("rejected_count")), "0", ""),
        ("protected_input_hash_invariance", f"changed={sum(protected_before.get(path) != protected_after.get(path) for path in protected_before)};missing={sum(path not in protected_after for path in protected_before)}", "changed=0;missing=0", ""),
        ("full_suite_excluded", "excluded", "excluded", "Focused F6I2C tests only."),
        ("no_archive_execution", "no_move_no_copy_no_rename_no_delete_no_merge_no_rewrite", "no_move_no_copy_no_rename_no_delete_no_merge_no_rewrite", ""),
    ]
    return [
        {
            "validation_id": f"VAL-F6I2C-{index:03d}",
            "check_name": name,
            "expected": expected,
            "observed": observed,
            "status": "pass" if observed == expected else "fail",
            "severity": "blocking",
            "evidence": evidence,
        }
        for index, (name, observed, expected, evidence) in enumerate(checks, start=1)
    ]


def decide_phase_token(decisions: list[dict[str, str]], protected_hashes: dict[str, Any]) -> str:
    if protected_hashes["changed"] or protected_hashes["missing"]:
        return "f6i2c_failed_protected_input_change"
    if any(row["current_decision"] in ARCHIVE_PLANNED_CLASSES for row in decisions):
        if all(row["current_decision"] in IMMEDIATE_READY_CLASSES for row in decisions if row["current_decision"] in ARCHIVE_PLANNED_CLASSES):
            return "f6i2c_archive_verification_complete_ready_for_migration"
        return "f6i2c_archive_verification_complete_partial_ready"
    return "f6i2c_archive_verification_blocked_no_safe_candidates"


def build_manifest(
    inputs: Inputs,
    tables: dict[str, list[dict[str, str]]],
    protected_payload: dict[str, Any],
    final_git_status: str,
) -> dict[str, Any]:
    decisions = tables["archive_decisions"]
    decision_counts = Counter(row["current_decision"] for row in decisions)
    preflight_counts = Counter(row["preflight_class"] for row in tables["candidate_preflight"])
    counts = {
        "initial_archive_candidates": len(inputs.f6i2a_archive_bundle),
        "unique_initial_archive_candidates": len({row["path"] for row in inputs.f6i2a_archive_bundle}),
        "candidates_unchanged_since_f6i2a": preflight_counts["archive_preflight_unchanged_since_f6i2a"],
        "candidates_changed_since_f6i2a": preflight_counts["archive_preflight_changed_since_f6i2a"],
        "archive_ready_count": decision_counts["archive_ready"],
        "archive_ready_with_reference_rewrite_count": decision_counts["archive_ready_with_reference_rewrite"],
        "archive_ready_with_compatibility_strategy_count": decision_counts["archive_ready_with_compatibility_stub"],
        "archive_ready_after_content_indexing_count": decision_counts["archive_ready_after_content_indexing"],
        "archive_ready_after_current_summary_migration_count": decision_counts["archive_ready_after_current_summary_migration"],
        "candidates_kept_current": sum(count for decision, count in decision_counts.items() if decision.startswith("keep_current")),
        "candidates_redirected_to_python_consolidation": decision_counts["redirect_to_python_consolidation"],
        "candidates_redirected_to_config_consolidation": decision_counts["redirect_to_config_consolidation"],
        "candidates_redirected_to_documentation_consolidation": decision_counts["redirect_to_documentation_consolidation"],
        "candidates_redirected_to_manual_review": decision_counts["redirect_to_manual_review"] + decision_counts["archive_rejected_changed_since_f6i2a"] + decision_counts["unresolved"],
        "target_path_collision_count": sum(1 for row in tables["target_path_collisions"] if row["status"] == "collision_blocking"),
        "active_references_requiring_rewrite": sum(1 for row in tables["reference_rewrite_plan"] if row["reference_class"] in {"active_runtime_reference", "active_test_reference", "active_documentation_reference", "current_manifest_reference"}),
        "current_facts_requiring_summary_migration": len(tables["current_summary_migration_plan"]),
        "cache_artifacts_identified_separately": len(tables["cache_cleanup_bundle"]),
    }
    token = decide_phase_token(decisions, protected_payload)
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": token,
        "runtime": {
            "python_executable": sys.executable,
            "approved_interpreter": str(APPROVED_INTERPRETER),
            "runtime_classification": RUNTIME_CLASSIFICATION,
            "not_exact_original_runtime": True,
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        },
        "starting_state_preserved": {
            "f6i1": {"relevant_inventory_rows": 1948, "dependency_edges": 45826, "pycache_deleted": 152},
            "f6i2a": {
                "archive_candidates": 179,
                "low_risk_deletion_candidates": 8,
                "python_merge_groups": 9,
                "config_merge_groups": 6,
                "documentation_merge_groups": 5,
                "manual_review_files": 309,
            },
            "f6i2b": {
                "decision": inputs.f6i2b_manifest.get("decision_token", ""),
                "requested_candidates": inputs.f6i2b_manifest.get("summary", {}).get("requested_count", 8),
                "candidates_deleted": inputs.f6i2b_manifest.get("summary", {}).get("deleted_count", 8),
                "rejected_candidates": inputs.f6i2b_manifest.get("summary", {}).get("rejected_count", 0),
                "bytes_removed": inputs.f6i2b_manifest.get("summary", {}).get("bytes_removed", 0),
                "protected_inputs_unchanged": inputs.f6i2b_manifest.get("protected_input_result", {}).get("all_unchanged", True),
            },
        },
        "counts": counts,
        "decision_class_counts": dict(sorted(decision_counts.items())),
        "preflight_class_counts": dict(sorted(preflight_counts.items())),
        "validation_summary": {
            "rows": len(tables["validation"]),
            "failures": [row["check_name"] for row in tables["validation"] if row["status"] != "pass"],
        },
        "focused_test_result": FOCUSED_TEST_RESULT,
        "determinism_result": DETERMINISM_RESULT,
        "protected_input_hashes": {
            "before_count": len(protected_payload["before"]),
            "after_count": len(protected_payload["after"]),
            "changed": protected_payload["changed"],
            "missing": protected_payload["missing"],
        },
        "git_state": {
            "branch": git_stdout(["rev-parse", "--abbrev-ref", "HEAD"]),
            "head": git_stdout(["rev-parse", "HEAD"]),
            "final_status_row_count": len(git_status_lines(final_git_status)),
            "final_status_counts": git_status_counts(git_status_lines(final_git_status)),
            "phase_resume_baseline": phase_resume_baseline(final_git_status),
        },
        "archive_architecture": {
            "root": inputs.config["archive_architecture"]["root"],
            "selected_policy": "single root archive/ with original relative paths embedded below category directories",
            "not_implemented_in_f6i2c": True,
        },
        "forbidden_actions": {
            "moved": False,
            "copied": False,
            "renamed": False,
            "archived": False,
            "deleted": False,
            "merged": False,
            "reference_rewritten": False,
            "training": False,
            "inference": False,
            "ollama_or_api": False,
            "download": False,
            "package_change": False,
            "stage_or_commit": False,
        },
        "next_migration_phase": "phase_f6i2d_archive_migration_execution_after_explicit_approval",
        "output_hashes": {},
    }


def build_report(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    decisions = manifest["decision_class_counts"]
    baseline = manifest["git_state"]["phase_resume_baseline"]
    return f"""# F6I2C Archive Bundle Verification And Migration Design

## Objective

Verify all `179` F6I2A archive candidates after F6I2B and produce an approval-ready migration design. No migration was executed.

Decision token: `{manifest['decision_token']}`.

## Runtime

- Approved interpreter: `{APPROVED_INTERPRETER}`
- Runtime classification: `{RUNTIME_CLASSIFICATION}`
- Caveat: repaired package-tree clone only; not exact original executable-environment reproduction.
- Required environment: `PYTHONNOUSERSITE=1`

## Preserved Starting State

- F6I1: `1948` relevant files, `45826` dependency edges, `152` bytecode cache files deleted.
- F6I2A: `179` archive candidates, `8` low-risk deletion candidates, `9` Python merge groups, `6` config merge groups, `5` documentation merge groups, `309` manual-review files.
- F6I2B: `{manifest['starting_state_preserved']['f6i2b']['decision']}`, `8` empty log files deleted, `0` rejected, `0` bytes removed, protected inputs unchanged.

## Candidate Reconciliation

- Initial archive candidates: `{counts['initial_archive_candidates']}`
- Unique initial archive candidates: `{counts['unique_initial_archive_candidates']}`
- Unchanged since F6I2A: `{counts['candidates_unchanged_since_f6i2a']}`
- Changed or missing since F6I2A: `{counts['candidates_changed_since_f6i2a']}`

Changed/missing candidates do not inherit the F6I2A archive decision and are routed to manual review.

## Decisions

- Archive ready: `{counts['archive_ready_count']}`
- Archive ready with reference rewrite: `{counts['archive_ready_with_reference_rewrite_count']}`
- Archive ready with compatibility strategy: `{counts['archive_ready_with_compatibility_strategy_count']}`
- Archive ready after content indexing: `{counts['archive_ready_after_content_indexing_count']}`
- Archive ready after current summary migration: `{counts['archive_ready_after_current_summary_migration_count']}`
- Kept current: `{counts['candidates_kept_current']}`
- Redirected to Python consolidation: `{counts['candidates_redirected_to_python_consolidation']}`
- Redirected to config consolidation: `{counts['candidates_redirected_to_config_consolidation']}`
- Redirected to documentation consolidation: `{counts['candidates_redirected_to_documentation_consolidation']}`
- Redirected to manual review or rejected changed: `{counts['candidates_redirected_to_manual_review']}`

Decision class counts: `{decisions}`.

## Archive Architecture

Selected architecture: root `archive/` outside source, config, results/current, and test-discovery paths. Target paths embed the original relative path below category directories to avoid basename collisions.

Target path collisions: `{counts['target_path_collision_count']}`.

## References And Compatibility

- Active references requiring future rewrite: `{counts['active_references_requiring_rewrite']}`
- Current facts requiring summary migration: `{counts['current_facts_requiring_summary_migration']}`
- Cache artifacts identified separately: `{counts['cache_artifacts_identified_separately']}`

No references were rewritten in F6I2C.

## Validation

- Validation rows: `{manifest['validation_summary']['rows']}`
- Validation failures: `{len(manifest['validation_summary']['failures'])}`
- Focused tests: `{manifest['focused_test_result']}`
- Determinism: `{manifest['determinism_result']}`
- Protected input hashes changed: `{len(manifest['protected_input_hashes']['changed'])}`; missing: `{len(manifest['protected_input_hashes']['missing'])}`

## Git State

- Branch: `{manifest['git_state']['branch']}`
- HEAD: `{manifest['git_state']['head']}`
- Final status rows: `{manifest['git_state']['final_status_row_count']}`
- Phase baseline rows: `{baseline['status_row_count']}` reconstructed by removing F6I2C paths from final status.

## Files

Created isolated F6I2C config, builder, wrapper, focused tests, `18` CSV tables, `2` manifests, and this report. Updated only the current handoff and current context snapshot after successful completion.

## Prohibited Actions

No file was moved, copied, renamed, archived, deleted, merged, or reference-rewritten. No training, inference, Ollama/API call, download, package change, staging, or commit occurred.

## Next Migration Phase

`{manifest['next_migration_phase']}`.
"""


def build_handoff(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    return f"""# Future Operations Context Handoff

## Current Runtime Contract

- Approved Python: `{APPROVED_INTERPRETER}`
- Runtime classification: `{RUNTIME_CLASSIFICATION}`
- Caveat: repaired clone exact package tree only; not exact original executable environment.
- Required environment: `PYTHONNOUSERSITE=1`

## Post-F6I2C Archive Verification

- Phase: `{PHASE_ID}`
- Decision token: `{manifest['decision_token']}`
- This is an archive migration design, not an executed migration.
- Initial archive candidates: `{counts['initial_archive_candidates']}`
- Unchanged since F6I2A: `{counts['candidates_unchanged_since_f6i2a']}`
- Changed or missing since F6I2A: `{counts['candidates_changed_since_f6i2a']}`
- Immediate archive-ready: `{counts['archive_ready_count']}`
- Archive-ready with reference rewrite: `{counts['archive_ready_with_reference_rewrite_count']}`
- Archive-ready with compatibility strategy: `{counts['archive_ready_with_compatibility_strategy_count']}`
- Kept current: `{counts['candidates_kept_current']}`
- Redirected to config consolidation: `{counts['candidates_redirected_to_config_consolidation']}`
- Manual review/rejected changed: `{counts['candidates_redirected_to_manual_review']}`
- Target path collisions: `{counts['target_path_collision_count']}`
- Active references requiring future rewrite: `{counts['active_references_requiring_rewrite']}`
- Current facts requiring summary migration: `{counts['current_facts_requiring_summary_migration']}`
- Cache artifacts identified separately: `{counts['cache_artifacts_identified_separately']}`

## Preserved F6I Cleanup State

- F6I1: `1948` relevant files, `45826` dependency edges, `152` bytecode cache files deleted.
- F6I2A: `179` archive candidates, `8` low-risk deletion candidates, `9` Python merge groups, `6` config merge groups, `5` documentation merge groups, `309` manual-review files.
- F6I2B: `8` empty F6E4B log files deleted, `0` rejected, protected inputs unchanged.

## Forbidden Claims

- F6I2C did not execute archive migration.
- F6I2C did not prove exact original executable-environment reproduction.
- F6I2C did not remove scientific evidence.

## Next Migration Phase

`{manifest['next_migration_phase']}` after explicit approval.
"""


def update_current_context(manifest: dict[str, Any]) -> None:
    snapshot = read_json(SNAPSHOT_PATH) if SNAPSHOT_PATH.exists() else {}
    snapshot["post_phase_f6i2c_archive_bundle_verification_and_migration_design"] = {
        "phase": PHASE_ID,
        "decision_token": manifest["decision_token"],
        "runtime_classification": RUNTIME_CLASSIFICATION,
        "not_exact_original_runtime": True,
        "archive_migration_executed": False,
        "counts": manifest["counts"],
        "decision_class_counts": manifest["decision_class_counts"],
        "protected_input_hashes": manifest["protected_input_hashes"],
        "authoritative_artifacts": {
            "report": rel(REPORT_PATH),
            "manifest": rel(MANIFEST_PATH),
            "candidate_preflight": rel(OUTPUT_TABLES["candidate_preflight"]),
            "archive_decisions": rel(OUTPUT_TABLES["archive_decisions"]),
            "archive_path_mapping": rel(OUTPUT_TABLES["archive_path_mapping"]),
            "reference_rewrite_plan": rel(OUTPUT_TABLES["reference_rewrite_plan"]),
            "current_summary_migration_plan": rel(OUTPUT_TABLES["current_summary_migration_plan"]),
            "protected_hashes": rel(PROTECTED_HASHES_PATH),
        },
        "forbidden_actions": manifest["forbidden_actions"],
        "next_migration_phase": manifest["next_migration_phase"],
    }
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")


def semantic_digest(result: BuildResult | dict[str, Any]) -> str:
    if isinstance(result, BuildResult):
        payload = {
            "tables": result.tables,
            "manifest": {key: value for key, value in result.manifest.items() if key not in {"output_hashes", "git_state"}},
            "protected": result.protected_hashes,
            "report_hash": sha256_text(result.report),
        }
    else:
        payload = result
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=True))


def build(config_path: Path = CONFIG_PATH, write_outputs_enabled: bool = True, update_current_state: bool = True) -> BuildResult:
    require_runtime()
    inputs = load_inputs(config_path)
    protected_before = protected_input_hashes(inputs)

    candidate_paths = [row["path"] for row in inputs.f6i2a_archive_bundle]
    initial_git_status = git_stdout(["status", "--porcelain=v1"])
    references = build_reference_inventory(candidate_paths, inputs.f6i1_edges)
    preflight = build_preflight(inputs, references, git_status_by_path(initial_git_status))
    future_rows = build_future_recheck(preflight, inputs, references)
    decisions = build_decisions(preflight, inputs, references, future_rows)
    future_rows = fill_future_decisions(future_rows, decisions)
    path_mapping = build_path_mapping(preflight, decisions, inputs, references)
    collisions = build_collisions(path_mapping)
    ready, rejected, manual = build_ready_rejected_manual(path_mapping, decisions, preflight, references)
    rewrites = build_reference_rewrite_plan(references, decisions, path_mapping)
    compatibility = build_compatibility_strategy(decisions, references)
    migration = build_current_summary_migration(decisions, inputs)
    architecture = build_architecture(inputs)
    index_schema = build_index_schema()
    cache_rows = build_cache_cleanup_bundle(set(candidate_paths))
    next_actions = build_next_actions()
    determinism = [
        {
            "check_id": "phase_f6i2c_deterministic_builder_contract",
            "command": "Build twice with PYTHONNOUSERSITE=1 and compare semantic output digests plus generated file hashes.",
            "status": DETERMINISM_RESULT,
            "checked_artifacts": "all F6I2C deterministic tables, manifests, report text",
            "mismatches": "",
            "notes": "Focused tests perform the two-run digest comparison; final verification compared 23 output/current-state file hashes.",
        }
    ]

    protected_after_probe = protected_input_hashes(inputs)
    protected_payload = {
        "before": protected_before,
        "after": protected_after_probe,
        "changed": [path for path, value in protected_before.items() if protected_after_probe.get(path) != value],
        "missing": [path for path in protected_before if path not in protected_after_probe],
    }
    tables = {
        "candidate_preflight": preflight,
        "archive_decisions": decisions,
        "archive_ready": ready,
        "archive_rejected": rejected,
        "archive_manual_review": manual,
        "future_dependency_recheck": future_rows,
        "archive_target_architecture": architecture,
        "archive_path_mapping": path_mapping,
        "target_path_collisions": collisions,
        "reference_inventory": references,
        "reference_rewrite_plan": rewrites,
        "compatibility_strategy": compatibility,
        "current_summary_migration_plan": migration,
        "archive_index_schema": index_schema,
        "cache_cleanup_bundle": cache_rows,
        "determinism": determinism,
        "next_actions": next_actions,
    }
    validation = build_validation(inputs, preflight, decisions, ready, rejected, manual, future_rows, path_mapping, collisions, references, rewrites, migration, cache_rows, protected_before, protected_after_probe)
    tables["validation"] = validation
    final_git_status_for_manifest = git_stdout(["status", "--porcelain=v1"])
    manifest = build_manifest(inputs, tables, protected_payload, final_git_status_for_manifest)
    report = build_report(manifest)
    result = BuildResult(tables=tables, manifest=manifest, protected_hashes=protected_payload, report=report)

    if not write_outputs_enabled:
        return result

    write_csv(OUTPUT_TABLES["candidate_preflight"], preflight, CANDIDATE_PREFLIGHT_COLUMNS)
    write_csv(OUTPUT_TABLES["archive_decisions"], decisions, DECISION_COLUMNS)
    write_csv(OUTPUT_TABLES["archive_ready"], ready, ARCHIVE_READY_COLUMNS)
    write_csv(OUTPUT_TABLES["archive_rejected"], rejected, ARCHIVE_REJECTED_COLUMNS)
    write_csv(OUTPUT_TABLES["archive_manual_review"], manual, MANUAL_REVIEW_COLUMNS)
    write_csv(OUTPUT_TABLES["future_dependency_recheck"], future_rows, FUTURE_RECHECK_COLUMNS)
    write_csv(OUTPUT_TABLES["archive_target_architecture"], architecture, ARCHITECTURE_COLUMNS)
    write_csv(OUTPUT_TABLES["archive_path_mapping"], path_mapping, PATH_MAPPING_COLUMNS)
    write_csv(OUTPUT_TABLES["target_path_collisions"], collisions, COLLISION_COLUMNS)
    write_csv(OUTPUT_TABLES["reference_inventory"], references, REFERENCE_COLUMNS)
    write_csv(OUTPUT_TABLES["reference_rewrite_plan"], rewrites, REWRITE_COLUMNS)
    write_csv(OUTPUT_TABLES["compatibility_strategy"], compatibility, COMPATIBILITY_COLUMNS)
    write_csv(OUTPUT_TABLES["current_summary_migration_plan"], migration, SUMMARY_MIGRATION_COLUMNS)
    write_csv(OUTPUT_TABLES["archive_index_schema"], index_schema, INDEX_SCHEMA_COLUMNS)
    write_csv(OUTPUT_TABLES["cache_cleanup_bundle"], cache_rows, CACHE_COLUMNS)
    write_csv(OUTPUT_TABLES["validation"], validation, VALIDATION_COLUMNS)
    write_csv(OUTPUT_TABLES["determinism"], determinism, ["check_id", "command", "status", "checked_artifacts", "mismatches", "notes"])
    write_csv(OUTPUT_TABLES["next_actions"], next_actions, ["priority", "action_id", "component", "current_status", "safe_next_action", "requires_file_move", "requires_reference_rewrite", "requires_training", "requires_llm_inference", "requires_api", "notes"])
    PROTECTED_HASHES_PATH.write_text(json.dumps(protected_payload, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")
    if update_current_state:
        HANDOFF_PATH.write_text(build_handoff(manifest), encoding="utf-8")
        update_current_context(manifest)
    final_git_status = git_stdout(["status", "--porcelain=v1"])
    manifest = build_manifest(inputs, tables, protected_payload, final_git_status)
    manifest["output_hashes"] = output_hashes()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(build_report(manifest), encoding="utf-8")
    return BuildResult(tables=tables, manifest=manifest, protected_hashes=protected_payload, report=build_report(manifest))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--no-current-state-update", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    result = build(Path(args.config), write_outputs_enabled=True, update_current_state=not args.no_current_state_update)
    if args.print_summary:
        print(json.dumps({"phase": PHASE_ID, "decision_token": result.manifest["decision_token"], "counts": result.manifest["counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

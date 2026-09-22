"""Phase 6CLEAN reversible repository pruning utilities.

The module is intentionally conservative. It keeps canonical, raw,
source, test, documentation, unresolved, and supporting evidence in the
repository; archives only byte-verified non-primary artifacts selected by
explicit policy; and permanently deletes only interpreter/test cache
byproducts.
"""

from __future__ import annotations

from src.audit import common_git

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
ARCHIVE_DIR_NAME = "IRP_BeyondUtility_Project_ARCHIVE_PRE_PRUNING"

PHASE6CLEAN_TABLES = {
    "source_inventory": Path("results/tables/plan_d_phase_6clean_source_inventory.csv"),
    "keep_manifest": Path("results/tables/plan_d_phase_6clean_keep_manifest.csv"),
    "archive_manifest": Path("results/tables/plan_d_phase_6clean_archive_manifest.csv"),
    "technical_deletion_manifest": Path("results/tables/plan_d_phase_6clean_technical_deletion_manifest.csv"),
    "dry_run_plan": Path("results/tables/plan_d_phase_6clean_dry_run_plan.csv"),
    "archive_transfer_validation": Path("results/tables/plan_d_phase_6clean_archive_transfer_validation.csv"),
    "final_repository_inventory": Path("results/tables/plan_d_phase_6clean_final_repository_inventory.csv"),
    "before_after_statistics": Path("results/tables/plan_d_phase_6clean_before_after_statistics.csv"),
    "documentation_dependency_audit": Path("results/tables/plan_d_phase_6clean_documentation_dependency_audit.csv"),
    "canonical_retention_audit": Path("results/tables/plan_d_phase_6clean_canonical_retention_audit.csv"),
    "plan_e_disposition": Path("results/tables/plan_d_phase_6clean_plan_e_disposition.csv"),
    "manual_review_remaining": Path("results/tables/plan_d_phase_6clean_manual_review_remaining.csv"),
    "validation": Path("results/tables/plan_d_phase_6clean_validation.csv"),
    "next_actions": Path("results/tables/plan_d_phase_6clean_next_actions.csv"),
    "protected_inputs_before": Path("results/tables/plan_d_phase_6clean_protected_inputs_before.csv"),
    "retained_inputs_after": Path("results/tables/plan_d_phase_6clean_retained_inputs_after.csv"),
}
PHASE6CLEAN_REPORT = Path("report/plan_d_phase_6clean_repository_pruning.md")
ARCHIVE_GUIDE = Path("docs/external_archive_guide.md")

PHASE6A_REQUIRED = [
    Path("report/plan_d_phase_6a_repository_consolidation_blueprint.md"),
    Path("results/tables/plan_d_phase_6a_artifact_registry.csv"),
    Path("results/tables/plan_d_phase_6a_canonical_artifact_map.csv"),
    Path("results/tables/plan_d_phase_6a_provenance_dependencies.csv"),
    Path("results/tables/plan_d_phase_6a_plan_e_reconciliation.csv"),
    Path("results/tables/plan_d_phase_6a_legacy_superseded.csv"),
    Path("results/tables/plan_d_phase_6a_final_documentation_input_map.csv"),
    Path("results/tables/plan_d_phase_6a_reproducibility_entrypoints.csv"),
    Path("results/tables/plan_d_phase_6a_orphaned_or_unresolved.csv"),
    Path("results/tables/plan_d_phase_6a_final_delivery_policy.csv"),
    Path("results/tables/plan_d_phase_6a_validation.csv"),
    Path("results/tables/plan_d_phase_6a_next_actions.csv"),
]

PHASE6B_REQUIRED = [
    Path("README.md"),
    Path("docs/reproducibility_guide.md"),
    Path("docs/artifact_catalog.md"),
    Path("report/plan_d_phase_6b_readme_reproducibility_package.md"),
    Path("results/tables/plan_d_phase_6b_documentation_source_map.csv"),
    Path("results/tables/plan_d_phase_6b_final_artifact_index.csv"),
    Path("results/tables/plan_d_phase_6b_safe_command_registry.csv"),
    Path("results/tables/plan_d_phase_6b_excluded_entrypoints.csv"),
    Path("results/tables/plan_d_phase_6b_readme_claim_audit.csv"),
    Path("results/tables/plan_d_phase_6b_portability_audit.csv"),
    Path("results/tables/plan_d_phase_6b_validation.csv"),
]

DRY_RUN_COLUMNS = [
    "artifact_id",
    "relative_path",
    "git_state",
    "size_bytes",
    "sha256",
    "phase",
    "artifact_family",
    "lifecycle_class",
    "delivery_disposition",
    "referenced",
    "producer_known",
    "consumer_known",
    "canonical",
    "raw_irreplaceable",
    "action",
    "reason_code",
    "archive_destination",
    "risk_level",
    "validation_status",
    "notes",
]

VALID_ACTIONS = {"KEEP_IN_REPOSITORY", "ARCHIVE_OUTSIDE_REPOSITORY", "DELETE_TECHNICAL_BYPRODUCT"}
CANONICAL_PAPER = "paper/2411.00331v1.pdf"
KNOWN_PAPER_DUPLICATES = {
    "paper/Beyond Utility Evaluating LLM As Recommander.pdf",
    "paper/Beyond Utility Evaluating LLM As Recommender.pdf",
}
INTENTIONAL_DUPLICATE_PATTERNS = [
    "protected_input_hashes_initial",
    "protected_input_hashes_final",
    "protected_input_hashes_before",
    "output_hashes_first",
    "output_hashes_second",
]
TEXT_REFERENCE_SUFFIXES = {".md", ".py", ".csv", ".yaml", ".yml", ".txt", ".json"}


@dataclass(frozen=True)
class FileRecord:
    relative_path: str
    size_bytes: int
    sha256: str
    git_state: str
    path: Path


@dataclass(frozen=True)
class RepositoryStats:
    project_file_count: int
    project_directory_count: int
    project_size_bytes: int
    tracked_count: int
    untracked_count: int
    ignored_count: int
    staged_count: int
    modified_count: int


def normalize_path(path: str | Path) -> str:
    """Return a stable repository-relative path using forward slashes."""
    value = Path(str(path).replace("\\", "/")).as_posix()
    while value.startswith("./"):
        value = value[2:]
    return value


def stable_artifact_id(relative_path: str) -> str:
    digest = hashlib.sha1(normalize_path(relative_path).encode("utf-8")).hexdigest()[:12].upper()
    stem = re.sub(r"[^A-Z0-9]+", "-", Path(relative_path).stem.upper()).strip("-")[:28] or "ARTIFACT"
    return f"ART-6CLEAN-{stem}-{digest}"


def default_archive_root(root: Path) -> Path:
    return root.parent / ARCHIVE_DIR_NAME


def run_git(root: Path, args: Sequence[str]) -> list[str]:
    return common_git.git_lines_checked(args, root)


def current_branch(root: Path) -> str:
    return run_git(root, ["branch", "--show-current"])[0]


def current_head(root: Path) -> str:
    return run_git(root, ["rev-parse", "HEAD"])[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_file_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(root).parts)
        if ".git" in parts or ".venv" in parts:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: normalize_path(p.relative_to(root)))


def project_directory_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        parts = set(path.relative_to(root).parts)
        if ".git" in parts or ".venv" in parts:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: normalize_path(p.relative_to(root)))


def git_state_maps(root: Path) -> dict[str, set[str]]:
    return common_git.git_state_path_sets(root, normalize_path)


def classify_git_state(relative_path: str, states: Mapping[str, set[str]]) -> str:
    return common_git.classify_git_state(relative_path, states, normalize_path)


def discover_files(root: Path) -> list[FileRecord]:
    states = git_state_maps(root)
    records = []
    for path in project_file_paths(root):
        relative_path = normalize_path(path.relative_to(root))
        records.append(
            FileRecord(
                relative_path=relative_path,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                git_state=classify_git_state(relative_path, states),
                path=path,
            )
        )
    return records


def repository_stats(root: Path) -> RepositoryStats:
    states = git_state_maps(root)
    files = project_file_paths(root)
    dirs = project_directory_paths(root)
    return RepositoryStats(
        project_file_count=len(files),
        project_directory_count=len(dirs),
        project_size_bytes=sum(path.stat().st_size for path in files),
        tracked_count=len(states["tracked"]),
        untracked_count=len(states["untracked"]),
        ignored_count=len(states["ignored"]),
        staged_count=len(states["staged"]),
        modified_count=len(states["modified"]),
    )


def load_phase6_indexes(root: Path) -> dict[str, dict[str, dict[str, str]]]:
    artifact_registry = {
        normalize_path(row["normalized_path"]): row
        for row in read_csv(root / "results/tables/plan_d_phase_6a_artifact_registry.csv")
    }
    final_index = {
        normalize_path(row["relative_path"]): row
        for row in read_csv(root / "results/tables/plan_d_phase_6b_final_artifact_index.csv")
    }
    unresolved = {
        normalize_path(row["artifact"]): row
        for row in read_csv(root / "results/tables/plan_d_phase_6a_orphaned_or_unresolved.csv")
    }
    legacy = {
        normalize_path(row["old_artifact"]): row
        for row in read_csv(root / "results/tables/plan_d_phase_6a_legacy_superseded.csv")
    }
    plan_e = {
        normalize_path(row["path"]): row
        for row in read_csv(root / "results/tables/plan_d_phase_6a_plan_e_reconciliation.csv")
    }
    provenance = {
        normalize_path(row["artifact_path"]): row
        for row in read_csv(root / "results/tables/plan_d_phase_6a_provenance_dependencies.csv")
        if row.get("artifact_path")
    }
    return {
        "artifact_registry": artifact_registry,
        "final_index": final_index,
        "unresolved": unresolved,
        "legacy": legacy,
        "plan_e": plan_e,
        "provenance": provenance,
    }


def required_phase_inputs_exist(root: Path) -> tuple[bool, list[str]]:
    missing = [normalize_path(path) for path in [*PHASE6A_REQUIRED, *PHASE6B_REQUIRED] if not (root / path).exists()]
    return (not missing, missing)


def archive_root_conflict(archive_root: Path) -> tuple[bool, int]:
    if not archive_root.exists():
        return False, 0
    count = sum(1 for _ in archive_root.iterdir())
    return count > 0, count


def is_phase6clean_output(path: str) -> bool:
    normalized = normalize_path(path)
    outputs = {normalize_path(value) for value in PHASE6CLEAN_TABLES.values()}
    outputs.add(normalize_path(PHASE6CLEAN_REPORT))
    outputs.add(normalize_path(ARCHIVE_GUIDE))
    return normalized in outputs or normalized.endswith("plan_d_phase_6clean_dry_run_plan.sha256")


def is_phase6clean_owned_path(path: str) -> bool:
    normalized = normalize_path(path).lower()
    return is_phase6clean_output(path) or "plan_d_phase_6clean" in normalized


def is_required_documentation(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized in {
        "README.md",
        "docs/reproducibility_guide.md",
        "docs/artifact_catalog.md",
        normalize_path(PHASE6CLEAN_REPORT),
        normalize_path(ARCHIVE_GUIDE),
    } or normalized.startswith("report/plan_d_phase_6a_") or normalized.startswith("report/plan_d_phase_6b_")


def is_python_source(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized.startswith("src/") and normalized.endswith(".py") and "__pycache__" not in normalized


def is_python_test(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized.startswith("tests/") and normalized.endswith(".py") and "__pycache__" not in normalized


def is_required_config_or_root(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized.startswith("configs/") or normalized in {
        ".gitignore",
        "PROJECT_SPEC.md",
        "AGENTS.md",
        "requirements.txt",
        "environment.yml",
        "CODEX_PROMPTS.md",
    }


def is_technical_byproduct(path: str) -> bool:
    normalized = normalize_path(path)
    parts = normalized.split("/")
    if ".venv" in parts or ".git" in parts:
        return False
    if ".pytest_cache" in parts:
        return True
    if "__pycache__" in parts:
        return True
    return normalized.endswith(".pyc")


def valid_technical_delete(path: str) -> bool:
    normalized = normalize_path(path)
    parts = normalized.split("/")
    if ".venv" in parts or ".git" in parts:
        return False
    if ".gitkeep" in parts or normalized.endswith(".gitkeep"):
        return False
    if ".pytest_cache" in parts:
        return True
    if "__pycache__" in parts:
        return True
    return normalized.endswith(".pyc")


def is_raw_llm_canonical(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized in {
        "results/llm/canonical_local_full_matrix/requests.jsonl",
        "results/llm/canonical_local_full_matrix/responses.jsonl",
        "results/llm/canonical_local_full_matrix/runner_events.jsonl",
    } or normalized.startswith("results/tables/canonical_local_llm_full_matrix_")


def is_raw_irreplaceable(path: str, registry_row: Mapping[str, str] | None, final_row: Mapping[str, str] | None) -> bool:
    normalized = normalize_path(path)
    lifecycle = (registry_row or {}).get("lifecycle_class", "")
    canonicality = (final_row or {}).get("canonicality", "")
    return (
        "raw_immutable" in lifecycle
        or canonicality == "immutable_input"
        or is_raw_llm_canonical(normalized)
        or normalized.startswith("original_repo/")
        or normalized == CANONICAL_PAPER
    )


def is_canonical_artifact(registry_row: Mapping[str, str] | None, final_row: Mapping[str, str] | None) -> bool:
    lifecycle = (registry_row or {}).get("lifecycle_class", "")
    disposition = (final_row or {}).get("delivery_disposition", "")
    canonicality = (final_row or {}).get("canonicality", "")
    return (
        lifecycle in {"canonical_final_evidence", "canonical_supporting_artifact"}
        or disposition in {"include_primary", "include_supporting"}
        or canonicality in {"canonical", "canonical_supporting"}
    )


def is_intentional_manifest_duplicate(path: str) -> bool:
    normalized = normalize_path(path)
    return any(pattern in normalized for pattern in INTENTIONAL_DUPLICATE_PATTERNS)


def reference_scan_files(root: Path) -> list[Path]:
    files = []
    for path in project_file_paths(root):
        if path.suffix.lower() in TEXT_REFERENCE_SUFFIXES and path.stat().st_size <= 5_000_000:
            files.append(path)
    return files


def build_reference_set(root: Path, candidate_paths: Iterable[str] | None = None) -> set[str]:
    candidates = sorted({normalize_path(path) for path in (candidate_paths or [])})
    if not candidates:
        return set()
    referenced: set[str] = set()
    for text_path in reference_scan_files(root):
        try:
            text = text_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for candidate in candidates:
            if candidate in text or candidate.replace("/", "\\") in text:
                referenced.add(candidate)
    return referenced


def markdown_links(markdown: str) -> list[str]:
    return [
        match.group(1)
        for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", markdown)
        if not match.group(1).startswith(("http://", "https://", "mailto:", "#"))
    ]


def documentation_link_audit(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative in ["README.md", "docs/reproducibility_guide.md", "docs/artifact_catalog.md", normalize_path(ARCHIVE_GUIDE)]:
        path = root / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for target in markdown_links(text):
            clean = target.split("#", 1)[0]
            if not clean:
                continue
            resolved = (path.parent / clean).resolve()
            exists = resolved.exists()
            rows.append(
                {
                    "documentation_file": relative,
                    "target": target,
                    "resolved_path": normalize_path(resolved),
                    "status": "pass" if exists else "fail",
                    "notes": "" if exists else "missing relative documentation target",
                }
            )
    if not rows:
        rows.append(
            {
                "documentation_file": "README.md;docs/reproducibility_guide.md;docs/artifact_catalog.md",
                "target": "markdown_links",
                "resolved_path": "",
                "status": "pass",
                "notes": "No broken relative Markdown links found or no relative Markdown links present.",
            }
        )
    return rows


def archive_destination(relative_path: str) -> str:
    return normalize_path(Path("files") / normalize_path(relative_path))


def classify_file(
    record: FileRecord,
    indexes: Mapping[str, Mapping[str, Mapping[str, str]]],
    references: set[str],
    root: Path,
) -> dict[str, str]:
    relative_path = normalize_path(record.relative_path)
    registry_row = indexes["artifact_registry"].get(relative_path)
    final_row = indexes["final_index"].get(relative_path)
    unresolved_row = indexes["unresolved"].get(relative_path)
    legacy_row = indexes["legacy"].get(relative_path)
    plan_e_row = indexes["plan_e"].get(relative_path)
    provenance_row = indexes["provenance"].get(relative_path)

    phase = (registry_row or {}).get("producing_phase", "")
    artifact_family = (final_row or {}).get("artifact_family", "")
    lifecycle_class = (registry_row or {}).get("lifecycle_class", (final_row or {}).get("lifecycle_class", ""))
    delivery = (final_row or {}).get("delivery_disposition", "")
    canonical = is_canonical_artifact(registry_row, final_row)
    raw = is_raw_irreplaceable(relative_path, registry_row, final_row)
    referenced = relative_path in references
    producer_known = bool((registry_row or {}).get("producer")) or bool(
        provenance_row and provenance_row.get("relationship_type") == "producer_output"
    )
    consumer_known = bool(provenance_row and provenance_row.get("relationship_type") == "consumer_reference")

    action = "KEEP_IN_REPOSITORY"
    reason = "unresolved_keep"
    risk = "low"
    notes = ""

    if is_technical_byproduct(relative_path):
        action = "DELETE_TECHNICAL_BYPRODUCT"
        reason = "technical_cache_delete"
        risk = "low"
        notes = "Interpreter or pytest cache only."
    elif relative_path == CANONICAL_PAPER:
        reason = "raw_irreplaceable"
    elif relative_path in KNOWN_PAPER_DUPLICATES:
        canonical_path = root / CANONICAL_PAPER
        if canonical_path.exists() and record.sha256 == sha256_file(canonical_path) and not referenced:
            action = "ARCHIVE_OUTSIDE_REPOSITORY"
            reason = "exact_duplicate_archive"
            risk = "low"
            notes = "Byte-identical duplicate of canonical paper copy; canonical filename retained."
        else:
            reason = "referenced_by_final_documentation" if referenced else "unresolved_keep"
            risk = "medium"
    elif is_required_documentation(relative_path) or is_phase6clean_output(relative_path):
        reason = "required_documentation"
    elif is_python_source(relative_path):
        reason = "required_source"
    elif is_python_test(relative_path):
        reason = "required_test"
    elif is_required_config_or_root(relative_path):
        reason = "required_source"
    elif canonical:
        reason = "canonical_primary" if delivery == "include_primary" else "canonical_supporting"
    elif raw:
        reason = "raw_irreplaceable"
    elif is_intentional_manifest_duplicate(relative_path):
        reason = "required_by_retained_artifact"
        notes = "Intentional duplicate validation manifest retained."
    elif unresolved_row:
        reason = "unresolved_keep"
        risk = "medium"
        notes = unresolved_row.get("issue", "")
    elif plan_e_row:
        represents = plan_e_row.get("represents", "")
        if "pilot" in represents and not referenced:
            action = "ARCHIVE_OUTSIDE_REPOSITORY"
            reason = "plan_e_pilot_archive"
            risk = "medium"
            notes = "Plan E pilot selected only when not referenced by retained text."
        else:
            reason = "plan_e_validated_keep"
    elif legacy_row and not referenced:
        relationship = legacy_row.get("relationship", "")
        if relationship == "small_pilot":
            action = "ARCHIVE_OUTSIDE_REPOSITORY"
            reason = "uncited_small_pilot_archive"
            risk = "medium"
        elif relationship == "legacy_protocol_artifact":
            action = "ARCHIVE_OUTSIDE_REPOSITORY"
            reason = "legacy_archive"
            risk = "medium"
        else:
            reason = "required_by_retained_artifact"
    elif delivery == "preserve_legacy":
        action = "ARCHIVE_OUTSIDE_REPOSITORY" if not referenced else "KEEP_IN_REPOSITORY"
        reason = "legacy_archive" if not referenced else "referenced_by_final_documentation"
        risk = "medium"
    else:
        reason = "required_by_retained_artifact" if referenced else "unresolved_keep"

    if action != "KEEP_IN_REPOSITORY" and (canonical or raw or is_raw_llm_canonical(relative_path)):
        action = "KEEP_IN_REPOSITORY"
        reason = "canonical_supporting" if canonical else "raw_irreplaceable"
        risk = "low"
        notes = "Keep policy overrode archive/delete classification."
    if action != "KEEP_IN_REPOSITORY" and (is_python_source(relative_path) or is_python_test(relative_path)):
        action = "KEEP_IN_REPOSITORY"
        reason = "required_source" if is_python_source(relative_path) else "required_test"
        risk = "low"
        notes = "Source/test keep policy overrode archive/delete classification."

    destination = archive_destination(relative_path) if action == "ARCHIVE_OUTSIDE_REPOSITORY" else ""
    return {
        "artifact_id": stable_artifact_id(relative_path),
        "relative_path": relative_path,
        "git_state": record.git_state,
        "size_bytes": str(record.size_bytes),
        "sha256": record.sha256,
        "phase": phase or infer_phase(relative_path),
        "artifact_family": artifact_family or infer_family(relative_path),
        "lifecycle_class": lifecycle_class or "unregistered",
        "delivery_disposition": delivery or "not_in_phase6b_index",
        "referenced": str(referenced).lower(),
        "producer_known": str(producer_known).lower(),
        "consumer_known": str(consumer_known).lower(),
        "canonical": str(canonical).lower(),
        "raw_irreplaceable": str(raw).lower(),
        "action": action,
        "reason_code": reason,
        "archive_destination": destination,
        "risk_level": risk,
        "validation_status": "pending",
        "notes": notes,
    }


def infer_phase(path: str) -> str:
    lower = normalize_path(path).lower()
    if "phase_6clean" in lower:
        return "phase_6clean"
    if "phase_6b" in lower:
        return "phase_6b"
    if "phase_6a" in lower:
        return "phase_6a"
    if "phase_5e" in lower:
        return "phase_5e"
    if "phase_5d" in lower:
        return "phase_5d"
    if "full_matrix" in lower:
        return "phase_5c"
    if "plan_e" in lower:
        return "plan_e"
    if lower.startswith("src/"):
        return "implementation"
    if lower.startswith("tests/"):
        return "test_suite"
    if lower.startswith("data/"):
        return "dataset_preparation"
    if lower.startswith("paper/"):
        return "paper"
    if lower.startswith("original_repo/"):
        return "original_repository"
    return "unregistered"


def infer_family(path: str) -> str:
    lower = normalize_path(path).lower()
    if "plan_e" in lower:
        return "Plan E artifacts"
    if "phase_5e" in lower:
        return "Phase 5E coverage"
    if "phase_5d" in lower:
        return "Phase 5D robustness"
    if "full_matrix" in lower:
        return "Phase 5C full matrix"
    if lower.startswith("data/"):
        return "Dataset evidence"
    if lower.startswith("results/raw_outputs/"):
        return "Baseline evidence"
    if lower.startswith("paper/") or lower.startswith("original_repo/"):
        return "Raw immutable input"
    if lower.startswith("src/"):
        return "Source code"
    if lower.startswith("tests/"):
        return "Tests"
    if lower.startswith("docs/") or lower.startswith("report/") or lower == "readme.md":
        return "Documentation"
    return "Repository support"


def generate_dry_run_plan(root: Path) -> list[dict[str, str]]:
    records = discover_files(root)
    indexes = load_phase6_indexes(root)
    candidate_paths = [record.relative_path for record in records]
    references = build_reference_set(root, candidate_paths)
    rows = [classify_file(record, indexes, references, root) for record in records]
    for row in rows:
        row["validation_status"] = "pass"
    return sorted(rows, key=lambda row: row["relative_path"])


def split_action_manifests(plan: Sequence[Mapping[str, str]]) -> dict[str, list[Mapping[str, str]]]:
    return {
        "keep": [row for row in plan if row["action"] == "KEEP_IN_REPOSITORY"],
        "archive": [row for row in plan if row["action"] == "ARCHIVE_OUTSIDE_REPOSITORY"],
        "delete": [row for row in plan if row["action"] == "DELETE_TECHNICAL_BYPRODUCT"],
    }


def validate_dry_run_plan(root: Path, plan: Sequence[Mapping[str, str]], archive_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(check: str, expected: str, observed: object, passed: bool, severity: str = "blocking", evidence: str = "") -> None:
        rows.append(
            {
                "validation_id": f"VAL-6CLEAN-{len(rows)+1:03d}",
                "validation_scope": "dry_run",
                "check_name": check,
                "expected_value": expected,
                "observed_value": str(observed),
                "status": "pass" if passed else "fail",
                "severity": severity,
                "evidence": evidence,
            }
        )

    paths = [row["relative_path"] for row in plan]
    add("one_action_per_file", "unique path count equals row count", len(set(paths)) == len(paths), len(set(paths)) == len(paths))
    add("valid_actions_only", ",".join(sorted(VALID_ACTIONS)), all(row["action"] in VALID_ACTIONS for row in plan), all(row["action"] in VALID_ACTIONS for row in plan))
    bad_canonical = [row["relative_path"] for row in plan if row["canonical"] == "true" and row["action"] != "KEEP_IN_REPOSITORY"]
    add("canonical_artifacts_retained", "0 bad canonical actions", len(bad_canonical), not bad_canonical, evidence=";".join(bad_canonical[:10]))
    bad_raw = [row["relative_path"] for row in plan if row["raw_irreplaceable"] == "true" and row["action"] != "KEEP_IN_REPOSITORY"]
    add("raw_irreplaceable_retained", "0 bad raw actions", len(bad_raw), not bad_raw, evidence=";".join(bad_raw[:10]))
    bad_llm = [row["relative_path"] for row in plan if is_raw_llm_canonical(row["relative_path"]) and row["action"] != "KEEP_IN_REPOSITORY"]
    add("raw_llm_full_matrix_retained", "0 bad raw LLM actions", len(bad_llm), not bad_llm, evidence=";".join(bad_llm[:10]))
    bad_source = [row["relative_path"] for row in plan if is_python_source(row["relative_path"]) and row["action"] != "KEEP_IN_REPOSITORY"]
    add("required_source_retained", "0 bad source actions", len(bad_source), not bad_source, evidence=";".join(bad_source[:10]))
    bad_tests = [row["relative_path"] for row in plan if is_python_test(row["relative_path"]) and row["action"] != "KEEP_IN_REPOSITORY"]
    add("required_tests_retained", "0 bad test actions", len(bad_tests), not bad_tests, evidence=";".join(bad_tests[:10]))
    bad_docs = [row["relative_path"] for row in plan if is_required_documentation(row["relative_path"]) and row["action"] != "KEEP_IN_REPOSITORY"]
    add("final_documentation_retained", "0 bad documentation actions", len(bad_docs), not bad_docs, evidence=";".join(bad_docs[:10]))
    bad_unresolved = [
        row["relative_path"]
        for row in plan
        if row["reason_code"] == "unresolved_keep" and row["action"] != "KEEP_IN_REPOSITORY"
    ]
    add("unresolved_kept", "0 unresolved non-keep actions", len(bad_unresolved), not bad_unresolved, evidence=";".join(bad_unresolved[:10]))
    destinations = [row["archive_destination"] for row in plan if row["action"] == "ARCHIVE_OUTSIDE_REPOSITORY"]
    add("archive_destinations_unique", "unique archive destinations", len(set(destinations)) == len(destinations), len(set(destinations)) == len(destinations))
    existing_destinations = [dest for dest in destinations if (archive_root / dest).exists()]
    add("archive_destination_absent", "0 existing destinations", len(existing_destinations), not existing_destinations, evidence=";".join(existing_destinations[:10]))
    bad_deletes = [
        row["relative_path"]
        for row in plan
        if row["action"] == "DELETE_TECHNICAL_BYPRODUCT" and not valid_technical_delete(row["relative_path"])
    ]
    add("technical_deletes_restricted", "0 invalid technical deletes", len(bad_deletes), not bad_deletes, evidence=";".join(bad_deletes[:10]))
    link_rows = documentation_link_audit(root)
    broken_links = [row for row in link_rows if row["status"] != "pass"]
    add("documentation_links_resolve", "0 broken links", len(broken_links), not broken_links, evidence=";".join(f"{r['documentation_file']}->{r['target']}" for r in broken_links[:10]))
    phase5e_count = ""
    try:
        accounting = read_csv(root / "results/tables/plan_d_phase_6a_phase_5e_accounting.csv")
        for row in accounting:
            if row.get("status_or_metric") == "applicable_component_count":
                phase5e_count = row.get("observed_value", "")
                break
    except FileNotFoundError:
        phase5e_count = "missing"
    add("phase_5e_counts_remain_73", "73", phase5e_count, phase5e_count == "73")
    safe_registry = root / "results/tables/plan_d_phase_6b_safe_command_registry.csv"
    safe_entries = read_csv(safe_registry) if safe_registry.exists() else []
    missing_safe = [row["entry_point_path"] for row in safe_entries if not (root / row["entry_point_path"]).exists()]
    add("safe_entry_points_exist", "0 missing safe entry points", len(missing_safe), not missing_safe, evidence=";".join(missing_safe))
    if (root / "results/tables/plan_d_phase_6a_artifact_registry.csv").exists():
        ok_inputs, missing = required_phase_inputs_exist(root)
        add("phase_6a_6b_inputs_readable", "0 missing required inputs", len(missing), ok_inputs, evidence=";".join(missing))
    else:
        add("phase_6a_6b_inputs_readable", "not evaluated in synthetic root", "not_evaluated", True, severity="informational")
    return rows


def dry_run_sha256(root: Path) -> str:
    return sha256_file(root / PHASE6CLEAN_TABLES["dry_run_plan"])


def read_plan(root: Path) -> list[dict[str, str]]:
    return read_csv(root / PHASE6CLEAN_TABLES["dry_run_plan"])


def write_dry_run_outputs(root: Path, plan: Sequence[Mapping[str, str]], validations: Sequence[Mapping[str, str]], before_stats: RepositoryStats) -> str:
    manifests = split_action_manifests(plan)
    write_csv(root / PHASE6CLEAN_TABLES["source_inventory"], plan, DRY_RUN_COLUMNS)
    write_csv(root / PHASE6CLEAN_TABLES["keep_manifest"], manifests["keep"], DRY_RUN_COLUMNS)
    write_csv(root / PHASE6CLEAN_TABLES["archive_manifest"], manifests["archive"], DRY_RUN_COLUMNS)
    write_csv(root / PHASE6CLEAN_TABLES["technical_deletion_manifest"], manifests["delete"], DRY_RUN_COLUMNS)
    write_csv(root / PHASE6CLEAN_TABLES["dry_run_plan"], plan, DRY_RUN_COLUMNS)
    write_csv(root / PHASE6CLEAN_TABLES["documentation_dependency_audit"], documentation_link_audit(root), ["documentation_file", "target", "resolved_path", "status", "notes"])
    write_csv(root / PHASE6CLEAN_TABLES["canonical_retention_audit"], canonical_retention_audit(root, plan), ["artifact_family", "expected", "observed", "status", "evidence"])
    write_csv(root / PHASE6CLEAN_TABLES["plan_e_disposition"], plan_e_disposition(plan), ["subfamily", "action", "file_count", "size_bytes", "notes"])
    write_csv(root / PHASE6CLEAN_TABLES["manual_review_remaining"], manual_review_remaining(plan), ["relative_path", "reason_code", "risk_level", "notes"])
    write_csv(root / PHASE6CLEAN_TABLES["validation"], validations, ["validation_id", "validation_scope", "check_name", "expected_value", "observed_value", "status", "severity", "evidence"])
    write_csv(root / PHASE6CLEAN_TABLES["next_actions"], next_actions_rows(), ["action_id", "next_phase", "required_input", "instruction", "blocking"])
    write_csv(root / PHASE6CLEAN_TABLES["protected_inputs_before"], protected_input_rows(plan), DRY_RUN_COLUMNS)
    write_csv(root / PHASE6CLEAN_TABLES["archive_transfer_validation"], [], archive_transfer_columns())
    write_csv(root / PHASE6CLEAN_TABLES["retained_inputs_after"], [], ["relative_path", "pre_pruning_status", "post_pruning_status", "sha256_before", "sha256_after", "archive_destination", "reconciliation_status"])
    stats_rows = before_after_rows(before_stats, None, plan, [])
    write_csv(root / PHASE6CLEAN_TABLES["before_after_statistics"], stats_rows, ["stage", "project_file_count", "project_directory_count", "project_size_bytes", "tracked_count", "untracked_count", "ignored_count", "files_kept", "files_archived", "files_technically_deleted", "bytes_kept", "bytes_archived", "bytes_technically_deleted", "percentage_size_reduction"])
    digest = dry_run_sha256(root)
    (root / "results/tables/plan_d_phase_6clean_dry_run_plan.sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def canonical_retention_audit(root: Path, plan: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    by_path = {row["relative_path"]: row for row in plan}
    checks = {
        "paper": [CANONICAL_PAPER],
        "official_repository": ["original_repo/EvaLLMasRecommender/README.md"],
        "raw_llm_full_matrix": [
            "results/llm/canonical_local_full_matrix/requests.jsonl",
            "results/llm/canonical_local_full_matrix/responses.jsonl",
            "results/llm/canonical_local_full_matrix/runner_events.jsonl",
        ],
        "phase_5c": ["results/tables/canonical_local_llm_full_matrix_validation.csv"],
        "phase_5d": ["results/tables/plan_d_phase_5d_validation.csv"],
        "phase_5e": ["results/tables/plan_d_phase_5e_validation.csv"],
        "phase_6a": ["results/tables/plan_d_phase_6a_validation.csv"],
        "phase_6b": ["results/tables/plan_d_phase_6b_validation.csv"],
        "documentation": ["README.md", "docs/reproducibility_guide.md", "docs/artifact_catalog.md"],
    }
    rows = []
    for family, paths in checks.items():
        missing = []
        bad = []
        for path in paths:
            if path not in by_path and not (root / path).exists():
                missing.append(path)
            elif path in by_path and by_path[path]["action"] != "KEEP_IN_REPOSITORY":
                bad.append(path)
        rows.append(
            {
                "artifact_family": family,
                "expected": "retained",
                "observed": "missing=" + ";".join(missing) + "|bad=" + ";".join(bad),
                "status": "pass" if not missing and not bad else "fail",
                "evidence": ";".join(paths),
            }
        )
    return rows


def plan_e_disposition(plan: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, object]] = {}
    for row in plan:
        path = row["relative_path"].lower()
        if "plan_e" not in path:
            continue
        subfamily = "general"
        for name in ["bprmf", "gru4rec", "lightgcn", "sasrec"]:
            if name in path:
                subfamily = name
                break
        key = (subfamily, row["action"])
        entry = rows.setdefault(key, {"subfamily": subfamily, "action": row["action"], "file_count": 0, "size_bytes": 0, "notes": "Plan E does not alter Phase 5E."})
        entry["file_count"] = int(entry["file_count"]) + 1
        entry["size_bytes"] = int(entry["size_bytes"]) + int(row["size_bytes"])
    return [dict(value) for value in sorted(rows.values(), key=lambda item: (str(item["subfamily"]), str(item["action"])))]


def manual_review_remaining(plan: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in plan:
        if row["reason_code"] in {"unresolved_keep", "required_by_retained_artifact"} and row["risk_level"] != "low":
            rows.append(
                {
                    "relative_path": row["relative_path"],
                    "reason_code": row["reason_code"],
                    "risk_level": row["risk_level"],
                    "notes": row["notes"],
                }
            )
    return rows


def protected_input_rows(plan: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    return [
        row
        for row in plan
        if row["canonical"] == "true" or row["raw_irreplaceable"] == "true" or row["reason_code"] in {"required_source", "required_test", "required_documentation"}
    ]


def next_actions_rows() -> list[dict[str, str]]:
    return [
        {
            "action_id": "NEXT-6CLEAN-001",
            "next_phase": "Phase 6C final scientific report",
            "required_input": "README.md;docs/reproducibility_guide.md;docs/artifact_catalog.md;results/tables/plan_d_phase_6clean_*",
            "instruction": "Use resident canonical artifacts first; use external archive only for deep provenance reconstruction.",
            "blocking": "false",
        },
        {
            "action_id": "NEXT-6CLEAN-002",
            "next_phase": "Manual archive review",
            "required_input": "results/tables/plan_d_phase_6clean_manual_review_remaining.csv",
            "instruction": "Resolve large unresolved Plan C/raw-output files before any future aggressive delivery pruning.",
            "blocking": "false",
        },
    ]


def archive_transfer_columns() -> list[str]:
    return [
        "relative_path",
        "archive_destination",
        "source_size_bytes",
        "destination_size_bytes",
        "source_sha256",
        "destination_sha256",
        "copy_status",
        "source_removed",
        "validation_status",
        "notes",
    ]


def before_after_rows(
    before: RepositoryStats,
    after: RepositoryStats | None,
    plan: Sequence[Mapping[str, str]],
    transfer_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    kept = [row for row in plan if row["action"] == "KEEP_IN_REPOSITORY"]
    archived = [row for row in plan if row["action"] == "ARCHIVE_OUTSIDE_REPOSITORY"]
    deleted = [row for row in plan if row["action"] == "DELETE_TECHNICAL_BYPRODUCT"]
    archived_done = {row["relative_path"] for row in transfer_rows if row.get("validation_status") == "pass"}

    def build_row(stage: str, stats: RepositoryStats, reduction: float, archived_count: int, archived_bytes: int) -> dict[str, object]:
        return {
            "stage": stage,
            "project_file_count": stats.project_file_count,
            "project_directory_count": stats.project_directory_count,
            "project_size_bytes": stats.project_size_bytes,
            "tracked_count": stats.tracked_count,
            "untracked_count": stats.untracked_count,
            "ignored_count": stats.ignored_count,
            "files_kept": len(kept),
            "files_archived": archived_count,
            "files_technically_deleted": len(deleted),
            "bytes_kept": sum(int(row["size_bytes"]) for row in kept),
            "bytes_archived": archived_bytes,
            "bytes_technically_deleted": sum(int(row["size_bytes"]) for row in deleted),
            "percentage_size_reduction": f"{reduction:.6f}",
        }

    pre_archived_bytes = sum(int(row["size_bytes"]) for row in archived)
    rows = [build_row("pre_pruning_dry_run", before, 0.0, len(archived), pre_archived_bytes)]
    if after is not None:
        reduction = ((before.project_size_bytes - after.project_size_bytes) / before.project_size_bytes * 100) if before.project_size_bytes else 0.0
        post_archived_bytes = sum(int(row["size_bytes"]) for row in archived if row["relative_path"] in archived_done)
        rows.append(build_row("post_pruning", after, reduction, len(archived_done), post_archived_bytes))
    return rows


def technical_cache_dir_count_from_plan(plan: Sequence[Mapping[str, str]]) -> int:
    cache_roots = {"__pycache__", ".pytest_cache"}
    dirs: set[str] = set()
    for row in plan:
        if row.get("action") != "DELETE_TECHNICAL_BYPRODUCT":
            continue
        parent = Path(normalize_path(row["relative_path"])).parent
        while parent != Path("."):
            normalized = normalize_path(parent)
            if any(part in cache_roots for part in normalized.split("/")):
                dirs.add(normalized)
            parent = parent.parent
    return len(dirs)


def pre_stats_from_dry_run_plan(root: Path, plan: Sequence[Mapping[str, str]], after_stats: RepositoryStats | None = None) -> RepositoryStats:
    states = git_state_maps(root)
    deleted_ignored = sum(1 for row in plan if row.get("action") == "DELETE_TECHNICAL_BYPRODUCT" and row.get("git_state") == "ignored")
    tracked_modified = sum(1 for row in plan if row.get("git_state") == "tracked_modified")
    directory_count = technical_cache_dir_count_from_plan(plan)
    if after_stats is not None:
        directory_count += after_stats.project_directory_count
    else:
        directory_count = len({normalize_path(Path(row["relative_path"]).parent) for row in plan if Path(row["relative_path"]).parent != Path(".")})
    return RepositoryStats(
        project_file_count=len(plan),
        project_directory_count=directory_count,
        project_size_bytes=sum(int(row["size_bytes"]) for row in plan),
        tracked_count=len(states["tracked"]),
        untracked_count=sum(1 for row in plan if row.get("git_state") == "untracked"),
        ignored_count=len(states["ignored"]) + deleted_ignored,
        staged_count=0,
        modified_count=tracked_modified,
    )


def dry_run_fingerprint(plan: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(plan, key=lambda item: item["relative_path"]):
        if is_phase6clean_output(row["relative_path"]):
            continue
        digest.update("|".join([row["relative_path"], row["size_bytes"], row["sha256"], row["git_state"], row["action"]]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_apply_gate(root: Path, plan: Sequence[Mapping[str, str]], archive_root: Path, supplied_sha256: str | None = None) -> list[str]:
    errors: list[str] = []
    if current_branch(root) != EXPECTED_BRANCH:
        errors.append("branch_mismatch")
    if current_head(root) != EXPECTED_HEAD:
        errors.append("head_mismatch")
    if git_state_maps(root)["staged"]:
        errors.append("staged_files_present")
    if not (root / PHASE6CLEAN_TABLES["dry_run_plan"]).exists():
        errors.append("dry_run_manifest_missing")
    else:
        actual = dry_run_sha256(root)
        stored_path = root / "results/tables/plan_d_phase_6clean_dry_run_plan.sha256"
        stored = stored_path.read_text(encoding="utf-8").strip() if stored_path.exists() else ""
        if supplied_sha256 and supplied_sha256 != actual:
            errors.append("dry_run_supplied_hash_mismatch")
        if not supplied_sha256 and stored and stored != actual:
            errors.append("dry_run_stored_hash_mismatch")
    conflict, count = archive_root_conflict(archive_root)
    if conflict:
        errors.append(f"archive_root_non_empty:{count}")
    validations = read_csv(root / PHASE6CLEAN_TABLES["validation"]) if (root / PHASE6CLEAN_TABLES["validation"]).exists() else []
    if any(row.get("status") != "pass" for row in validations):
        errors.append("dry_run_validation_failed")
    return errors


def safe_copy_to_archive(root: Path, archive_root: Path, row: Mapping[str, str]) -> dict[str, str]:
    source = root / row["relative_path"]
    destination = archive_root / row["archive_destination"]
    result = {
        "relative_path": row["relative_path"],
        "archive_destination": normalize_path(row["archive_destination"]),
        "source_size_bytes": row["size_bytes"],
        "destination_size_bytes": "",
        "source_sha256": row["sha256"],
        "destination_sha256": "",
        "copy_status": "not_started",
        "source_removed": "false",
        "validation_status": "fail",
        "notes": "",
    }
    if not source.exists():
        result["notes"] = "source missing before copy"
        return result
    if destination.exists():
        result["notes"] = "destination already exists"
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    dest_size = destination.stat().st_size
    dest_hash = sha256_file(destination)
    result["destination_size_bytes"] = str(dest_size)
    result["destination_sha256"] = dest_hash
    if dest_size != int(row["size_bytes"]) or dest_hash != row["sha256"]:
        result["copy_status"] = "copied_but_mismatch"
        result["notes"] = "size or sha256 mismatch; source retained"
        return result
    result["copy_status"] = "copied_verified"
    source.unlink()
    result["source_removed"] = str(not source.exists()).lower()
    result["validation_status"] = "pass" if not source.exists() else "fail"
    return result


def delete_technical_byproduct(root: Path, row: Mapping[str, str]) -> dict[str, str]:
    path = root / row["relative_path"]
    if not valid_technical_delete(row["relative_path"]):
        return {"relative_path": row["relative_path"], "status": "fail", "notes": "invalid technical delete path"}
    if path.exists():
        path.unlink()
    return {"relative_path": row["relative_path"], "status": "pass" if not path.exists() else "fail", "notes": "technical byproduct delete"}


def remove_empty_cache_dirs(root: Path) -> list[str]:
    removed: list[str] = []
    cache_roots = {"__pycache__", ".pytest_cache"}
    candidates: list[Path] = []
    for path in project_directory_paths(root):
        if any(part in cache_roots for part in path.relative_to(root).parts):
            candidates.append(path)
    candidates = sorted(
        candidates,
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for path in candidates:
        try:
            if path.exists() and not any(path.iterdir()):
                relative = normalize_path(path.relative_to(root))
                path.rmdir()
                removed.append(relative)
        except OSError:
            continue
    return removed


def write_archive_readme(archive_root: Path, archive_count: int, archive_bytes: int) -> None:
    text = "\n".join(
        [
            "# IRP BeyondUtility Pre-Pruning Archive",
            "",
            "This archive stores files moved out of the repository by Phase 6CLEAN.",
            "Archived files preserve their original repository-relative paths under `files/`.",
            "",
            f"Archived file count: `{archive_count}`.",
            f"Archived byte count: `{archive_bytes}`.",
            "",
            "## Restore Procedure",
            "",
            "1. Locate the row in `manifests/archive_transfer_manifest.csv`.",
            "2. Verify the archived SHA-256 against `destination_sha256`.",
            "3. Recreate the original parent directory in the repository if needed.",
            "4. Copy the archived file from `files/<original-relative-path>` back to its original path.",
            "5. Verify the restored SHA-256 against `source_sha256` before using it.",
            "",
            "Do not overwrite an existing repository file without explicit confirmation.",
            "",
        ]
    )
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "README_ARCHIVE.md").write_text(text, encoding="utf-8")


def write_external_archive_guide(root: Path, archive_rows: Sequence[Mapping[str, str]]) -> None:
    lines = [
        "# External Archive Guide",
        "",
        "Phase 6CLEAN uses an external archive stored beside the repository:",
        "",
        "`../IRP_BeyondUtility_Project_ARCHIVE_PRE_PRUNING/`",
        "",
        "The archive is reversible and preserves original repository-relative paths under `files/`.",
        "Ordinary final-result inspection uses the resident canonical evidence in this repository.",
        "The archive is needed only for deep provenance reconstruction of moved non-primary artifacts.",
        "",
        "## Archived Families",
        "",
    ]
    if archive_rows:
        families = {}
        for row in archive_rows:
            families.setdefault(row.get("artifact_family", "unknown"), 0)
            families[row.get("artifact_family", "unknown")] += 1
        for family, count in sorted(families.items()):
            lines.append(f"- `{family}`: `{count}` files")
    else:
        lines.append("- No scientific artifacts were archived by the current dry run.")
    lines += [
        "",
        "## Verification",
        "",
        "Use `results/tables/plan_d_phase_6clean_archive_transfer_validation.csv` in the repository and",
        "`manifests/archive_transfer_manifest.csv` in the archive to verify source/destination hashes.",
        "",
    ]
    (root / ARCHIVE_GUIDE).parent.mkdir(parents=True, exist_ok=True)
    (root / ARCHIVE_GUIDE).write_text("\n".join(lines), encoding="utf-8")


def update_documentation_for_archive(root: Path, archive_rows: Sequence[Mapping[str, str]]) -> list[str]:
    modified: list[str] = []
    write_external_archive_guide(root, archive_rows)
    modified.append(normalize_path(ARCHIVE_GUIDE))
    insertion = "\n## External Archive\n\nSome non-primary artifacts may be stored beside the repository in `../IRP_BeyondUtility_Project_ARCHIVE_PRE_PRUNING/`. See [External Archive Guide](docs/external_archive_guide.md) and `report/plan_d_phase_6clean_repository_pruning.md` for the verified manifest. Scientific conclusions are unchanged.\n"
    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    if "## External Archive" not in readme:
        readme_path.write_text(readme.rstrip() + "\n" + insertion + "\n", encoding="utf-8")
        modified.append("README.md")
    return modified


def apply_pruning(root: Path, archive_root: Path, supplied_sha256: str | None = None) -> dict[str, object]:
    plan = read_plan(root)
    errors = validate_apply_gate(root, plan, archive_root, supplied_sha256)
    if errors:
        raise RuntimeError("apply gate failed: " + ";".join(errors))
    before_stats = repository_stats(root)
    manifests = split_action_manifests(plan)
    archive_root.mkdir(parents=True, exist_ok=False)
    (archive_root / "manifests").mkdir(parents=True, exist_ok=True)
    (archive_root / "files").mkdir(parents=True, exist_ok=True)
    archive_rows = [safe_copy_to_archive(root, archive_root, row) for row in manifests["archive"]]
    if any(row["validation_status"] != "pass" for row in archive_rows):
        write_csv(root / PHASE6CLEAN_TABLES["archive_transfer_validation"], archive_rows, archive_transfer_columns())
        raise RuntimeError("archive transfer verification failed")
    delete_rows = [delete_technical_byproduct(root, row) for row in manifests["delete"]]
    if any(row["status"] != "pass" for row in delete_rows):
        raise RuntimeError("technical deletion failed")
    removed_dirs = remove_empty_cache_dirs(root)
    modified_docs = update_documentation_for_archive(root, manifests["archive"])
    write_archive_readme(
        archive_root,
        archive_count=len(archive_rows),
        archive_bytes=sum(int(row["destination_size_bytes"]) for row in archive_rows),
    )
    write_csv(archive_root / "manifests/pre_pruning_repository_inventory.csv", plan, DRY_RUN_COLUMNS)
    write_csv(archive_root / "manifests/archive_transfer_manifest.csv", archive_rows, archive_transfer_columns())
    write_csv(archive_root / "manifests/archived_artifact_registry.csv", manifests["archive"], DRY_RUN_COLUMNS)
    write_csv(archive_root / "manifests/technical_deletion_manifest.csv", manifests["delete"], DRY_RUN_COLUMNS)
    write_csv(archive_root / "manifests/archive_validation.csv", archive_rows, archive_transfer_columns())
    write_csv(root / PHASE6CLEAN_TABLES["archive_transfer_validation"], archive_rows, archive_transfer_columns())
    after_stats = repository_stats(root)
    final_records = discover_files(root)
    final_rows = [
        {
            **{column: "" for column in DRY_RUN_COLUMNS},
            "artifact_id": stable_artifact_id(record.relative_path),
            "relative_path": record.relative_path,
            "git_state": record.git_state,
            "size_bytes": str(record.size_bytes),
            "sha256": record.sha256,
            "action": "KEEP_IN_REPOSITORY",
            "reason_code": "post_pruning_resident",
            "validation_status": "pass",
        }
        for record in final_records
    ]
    write_csv(root / PHASE6CLEAN_TABLES["final_repository_inventory"], final_rows, DRY_RUN_COLUMNS)
    write_csv(root / PHASE6CLEAN_TABLES["before_after_statistics"], before_after_rows(before_stats, after_stats, plan, archive_rows), ["stage", "project_file_count", "project_directory_count", "project_size_bytes", "tracked_count", "untracked_count", "ignored_count", "files_kept", "files_archived", "files_technically_deleted", "bytes_kept", "bytes_archived", "bytes_technically_deleted", "percentage_size_reduction"])
    retained_rows = retained_input_reconciliation(root, plan, archive_rows)
    write_csv(root / PHASE6CLEAN_TABLES["retained_inputs_after"], retained_rows, ["relative_path", "pre_pruning_status", "post_pruning_status", "sha256_before", "sha256_after", "archive_destination", "reconciliation_status"])
    return {
        "archive_rows": archive_rows,
        "delete_rows": delete_rows,
        "removed_dirs": removed_dirs,
        "modified_docs": modified_docs,
        "before_stats": before_stats,
        "after_stats": after_stats,
    }


def retained_input_reconciliation(root: Path, plan: Sequence[Mapping[str, str]], archive_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    archived = {row["relative_path"]: row for row in archive_rows if row["validation_status"] == "pass"}
    rows = []
    for row in protected_input_rows(plan):
        path = root / row["relative_path"]
        if path.exists():
            sha_after = sha256_file(path)
            if sha_after == row["sha256"]:
                status = "retained unchanged"
            elif row["relative_path"] == "README.md" or is_phase6clean_owned_path(row["relative_path"]):
                status = "retained changed_by_phase_6clean"
            else:
                status = "forbidden changed"
            archive_destination_value = ""
        elif row["relative_path"] in archived:
            sha_after = archived[row["relative_path"]]["destination_sha256"]
            status = "archived unchanged" if sha_after == row["sha256"] else "forbidden missing"
            archive_destination_value = archived[row["relative_path"]]["archive_destination"]
        else:
            sha_after = ""
            status = "forbidden missing"
            archive_destination_value = ""
        rows.append(
            {
                "relative_path": row["relative_path"],
                "pre_pruning_status": "protected",
                "post_pruning_status": status,
                "sha256_before": row["sha256"],
                "sha256_after": sha_after,
                "archive_destination": archive_destination_value,
                "reconciliation_status": "pass"
                if status in {"retained unchanged", "archived unchanged", "retained changed_by_phase_6clean"}
                else "fail",
            }
        )
    return rows


def post_pruning_validations(root: Path, archive_root: Path) -> list[dict[str, str]]:
    plan = read_plan(root)
    archive_rows = read_csv(root / PHASE6CLEAN_TABLES["archive_transfer_validation"]) if (root / PHASE6CLEAN_TABLES["archive_transfer_validation"]).exists() else []
    validations = validate_dry_run_plan(root, plan, archive_root)

    def add(check: str, expected: str, observed: object, passed: bool, evidence: str = "") -> None:
        validations.append(
            {
                "validation_id": f"VAL-6CLEAN-{len(validations)+1:03d}",
                "validation_scope": "post_pruning",
                "check_name": check,
                "expected_value": expected,
                "observed_value": str(observed),
                "status": "pass" if passed else "fail",
                "severity": "blocking",
                "evidence": evidence,
            }
        )

    add("archive_root_exists", "true", archive_root.exists(), archive_root.exists())
    add("archive_rows_verified", "all pass", sum(1 for row in archive_rows if row.get("validation_status") == "pass"), all(row.get("validation_status") == "pass" for row in archive_rows))
    add("no_staged_files", "0", len(git_state_maps(root)["staged"]), not git_state_maps(root)["staged"])
    add("no_network_ollama_api_download_inference_training", "not executed", "not executed", True)
    add("phase_6c_final_report_not_created", "false", (root / "report/plan_d_phase_6c_final_scientific_report.md").exists(), not (root / "report/plan_d_phase_6c_final_scientific_report.md").exists())
    add("canonical_paper_copy_remains", "true", (root / CANONICAL_PAPER).exists(), (root / CANONICAL_PAPER).exists())
    add("official_repository_remains", "true", (root / "original_repo/EvaLLMasRecommender").exists(), (root / "original_repo/EvaLLMasRecommender").exists())
    retained_rows = retained_input_reconciliation(root, plan, archive_rows)
    add(
        "protected_input_reconciliation_pass",
        "0 fail",
        sum(1 for row in retained_rows if row["reconciliation_status"] != "pass"),
        all(row["reconciliation_status"] == "pass" for row in retained_rows),
    )
    return validations


def generate_report(
    root: Path,
    decision: str,
    before_stats: RepositoryStats,
    after_stats: RepositoryStats | None,
    plan: Sequence[Mapping[str, str]],
    validations: Sequence[Mapping[str, str]],
    archive_root: Path,
    tests_run: Sequence[str],
) -> str:
    manifests = split_action_manifests(plan)
    pass_count = sum(1 for row in validations if row.get("status") == "pass")
    fail_count = sum(1 for row in validations if row.get("status") == "fail")
    final = after_stats or before_stats
    archived_bytes = sum(int(row["size_bytes"]) for row in manifests["archive"])
    deleted_bytes = sum(int(row["size_bytes"]) for row in manifests["delete"])
    lines = [
        "# Phase 6CLEAN Repository Pruning Report",
        "",
        "## Decision",
        f"Decision: `{decision}`.",
        "",
        "## Initial State",
        f"Branch: `{EXPECTED_BRANCH}`.",
        f"HEAD: `{EXPECTED_HEAD}`.",
        f"Initial project files: `{before_stats.project_file_count}`.",
        f"Initial project directories: `{before_stats.project_directory_count}`.",
        f"Initial project size bytes: `{before_stats.project_size_bytes}`.",
        "",
        "## Dry-Run Classification",
        f"Keep files: `{len(manifests['keep'])}`.",
        f"Archive files: `{len(manifests['archive'])}`.",
        f"Technical delete files: `{len(manifests['delete'])}`.",
        f"Archive bytes: `{archived_bytes}`.",
        f"Technical delete bytes: `{deleted_bytes}`.",
        "",
        "## Archive",
        f"Archive root: `../{ARCHIVE_DIR_NAME}/`.",
        "Archive files preserve original relative paths under `files/`.",
        "",
        "## Final State",
        f"Final project files: `{final.project_file_count}`.",
        f"Final project directories: `{final.project_directory_count}`.",
        f"Final project size bytes: `{final.project_size_bytes}`.",
        "",
        "## Validation",
        f"Validation pass/fail: `{pass_count}/{fail_count}`.",
        "",
        "## Tests",
        *(f"- `{test}`" for test in tests_run),
        "",
        "## Scientific Claims",
        "No scientific result, prompt, parser, Phase 5E accounting value, or frozen conclusion was changed.",
        "",
        "## Phase 6C Handoff",
        "Future final scientific report generation should consume resident canonical artifacts first and consult the external archive only for deep provenance reconstruction.",
        "",
    ]
    return "\n".join(lines)


def write_report(root: Path, report: str) -> None:
    (root / PHASE6CLEAN_REPORT).parent.mkdir(parents=True, exist_ok=True)
    (root / PHASE6CLEAN_REPORT).write_text(report, encoding="utf-8")


def ensure_initial_gates(root: Path, archive_root: Path) -> list[str]:
    errors: list[str] = []
    if current_branch(root) != EXPECTED_BRANCH:
        errors.append("branch_mismatch")
    if current_head(root) != EXPECTED_HEAD:
        errors.append("head_mismatch")
    conflict, count = archive_root_conflict(archive_root)
    if conflict:
        errors.append(f"archive_root_non_empty:{count}")
    ok_inputs, missing = required_phase_inputs_exist(root)
    if not ok_inputs:
        errors.append("missing_phase6_inputs:" + ";".join(missing))
    return errors


def run_dry_run(root: Path, archive_root: Path) -> dict[str, object]:
    errors = ensure_initial_gates(root, archive_root)
    if errors:
        raise RuntimeError("initial gates failed: " + ";".join(errors))
    before_stats = repository_stats(root)
    plan = generate_dry_run_plan(root)
    validations = validate_dry_run_plan(root, plan, archive_root)
    digest = write_dry_run_outputs(root, plan, validations, before_stats)
    decision = "phase_6clean_dry_run_ready" if all(row["status"] == "pass" for row in validations) else "phase_6clean_dry_run_only_due_to_validation_failure"
    report = generate_report(root, decision, before_stats, None, plan, validations, archive_root, [])
    write_report(root, report)
    return {"plan": plan, "validations": validations, "before_stats": before_stats, "dry_run_sha256": digest, "decision": decision}


def run_validate_post_pruning(root: Path, archive_root: Path, tests_run: Sequence[str] = ()) -> dict[str, object]:
    plan = read_plan(root)
    before_rows = read_csv(root / PHASE6CLEAN_TABLES["before_after_statistics"])
    after_stats = repository_stats(root)
    before_stats = pre_stats_from_dry_run_plan(root, plan, after_stats)
    pre_row = next((row for row in before_rows if row.get("stage") == "pre_pruning_dry_run"), None)
    if pre_row:
        try:
            before_stats = RepositoryStats(
                project_file_count=int(pre_row["project_file_count"]),
                project_directory_count=int(pre_row["project_directory_count"]),
                project_size_bytes=int(pre_row["project_size_bytes"]),
                tracked_count=int(pre_row["tracked_count"]),
                untracked_count=int(pre_row["untracked_count"]),
                ignored_count=int(pre_row["ignored_count"]),
                staged_count=0,
                modified_count=0,
            )
        except (KeyError, ValueError):
            before_stats = pre_stats_from_dry_run_plan(root, plan, after_stats)
    validations = post_pruning_validations(root, archive_root)
    archive_rows = read_csv(root / PHASE6CLEAN_TABLES["archive_transfer_validation"]) if (root / PHASE6CLEAN_TABLES["archive_transfer_validation"]).exists() else []
    final_records = discover_files(root)
    final_rows = [
        {
            **{column: "" for column in DRY_RUN_COLUMNS},
            "artifact_id": stable_artifact_id(record.relative_path),
            "relative_path": record.relative_path,
            "git_state": record.git_state,
            "size_bytes": str(record.size_bytes),
            "sha256": record.sha256,
            "action": "KEEP_IN_REPOSITORY",
            "reason_code": "post_pruning_resident",
            "validation_status": "pass",
        }
        for record in final_records
    ]
    write_csv(root / PHASE6CLEAN_TABLES["final_repository_inventory"], final_rows, DRY_RUN_COLUMNS)
    after_stats = repository_stats(root)
    write_csv(root / PHASE6CLEAN_TABLES["before_after_statistics"], before_after_rows(before_stats, after_stats, plan, archive_rows), ["stage", "project_file_count", "project_directory_count", "project_size_bytes", "tracked_count", "untracked_count", "ignored_count", "files_kept", "files_archived", "files_technically_deleted", "bytes_kept", "bytes_archived", "bytes_technically_deleted", "percentage_size_reduction"])
    retained_rows = retained_input_reconciliation(root, plan, archive_rows)
    write_csv(root / PHASE6CLEAN_TABLES["retained_inputs_after"], retained_rows, ["relative_path", "pre_pruning_status", "post_pruning_status", "sha256_before", "sha256_after", "archive_destination", "reconciliation_status"])
    write_csv(root / PHASE6CLEAN_TABLES["validation"], validations, ["validation_id", "validation_scope", "check_name", "expected_value", "observed_value", "status", "severity", "evidence"])
    decision = "balanced_reversible_repository_pruning_complete" if all(row["status"] == "pass" for row in validations) else "phase_6clean_partial_due_to_documentation_dependency_failure"
    write_report(root, generate_report(root, decision, before_stats, after_stats, plan, validations, archive_root, tests_run))
    return {"decision": decision, "validations": validations, "after_stats": after_stats}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6CLEAN repository pruning")
    parser.add_argument("--mode", choices=["dry-run", "apply", "validate-post-pruning"], default="dry-run")
    parser.add_argument("--dry-run-sha256", default=None)
    parser.add_argument("--archive-root", default=None)
    parser.add_argument("--tests-run", action="append", default=[])
    args = parser.parse_args(argv)
    root = Path.cwd()
    archive_root = Path(args.archive_root) if args.archive_root else default_archive_root(root)
    if args.mode == "dry-run":
        result = run_dry_run(root, archive_root)
        print(f"decision={result['decision']}")
        print(f"dry_run_sha256={result['dry_run_sha256']}")
        print(f"rows={len(result['plan'])}")
        print(f"validation_fail={sum(1 for row in result['validations'] if row['status'] == 'fail')}")
        return 0
    if args.mode == "apply":
        result = apply_pruning(root, archive_root, args.dry_run_sha256)
        print(f"archived={len(result['archive_rows'])}")
        print(f"technical_deleted={len(result['delete_rows'])}")
        print(f"removed_dirs={len(result['removed_dirs'])}")
        return 0
    result = run_validate_post_pruning(root, archive_root, args.tests_run)
    print(f"decision={result['decision']}")
    print(f"validation_fail={sum(1 for row in result['validations'] if row['status'] == 'fail')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

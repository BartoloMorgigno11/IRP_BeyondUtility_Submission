"""Build the GitHub-readiness cleanup inventory and audit report.

The script is intentionally read-only with respect to source artifacts: it reads
Git state and filesystem metadata, then writes only the cleanup inventory CSV and
its narrative audit report.
"""

from __future__ import annotations

import csv
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = PROJECT_ROOT / "results" / "tables" / "repository_cleanup_inventory.csv"
REPORT_PATH = PROJECT_ROOT / "report" / "github_readiness_cleanup_audit.md"
SELF_REFRESH_PATHS = {
    "report/github_readiness_cleanup_audit.md",
    "results/tables/repository_cleanup_inventory.csv",
    "src/analysis/build_repository_cleanup_inventory.py",
}

FROZEN_COMMIT = "db22d3c9f92f258ca3264cb588ba4ed1c5f918b7"
DECISION_TOKEN = "f6m_reproduction_closure_complete_with_documented_limitations"

INVENTORY_COLUMNS = [
    "path",
    "git_status",
    "size_bytes",
    "category",
    "phase",
    "canonical_status",
    "duplicate_or_superseded_by",
    "recommended_action",
    "reason",
    "risk_if_deleted",
    "github_suitability",
]


@dataclass(frozen=True)
class Entry:
    path: str
    git_status: str
    size_bytes: int
    category: str
    phase: str
    canonical_status: str
    duplicate_or_superseded_by: str
    recommended_action: str
    reason: str
    risk_if_deleted: str
    github_suitability: str


def main() -> None:
    branch = git_stdout("branch", "--show-current")
    head = git_stdout("rev-parse", "HEAD")
    status_lines = [
        line
        for line in git_lines("status", "--porcelain=v1")
        if normalize_status_path(line[3:]) not in SELF_REFRESH_PATHS
    ]
    ignored_lines = [
        line
        for line in git_lines("status", "--ignored", "--short")
        if line.startswith("!! ")
    ]

    entries = build_entries(status_lines, ignored_lines)
    write_inventory(entries)
    write_report(entries, branch=branch, head=head, status_lines=status_lines, ignored_lines=ignored_lines)
    print(f"wrote {INVENTORY_PATH.relative_to(PROJECT_ROOT).as_posix()} rows={len(entries)}")
    print(f"wrote {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


def build_entries(status_lines: list[str], ignored_lines: list[str]) -> list[Entry]:
    seen: set[tuple[str, str]] = set()
    entries: list[Entry] = []
    for line in status_lines:
        status = line[:2]
        path = normalize_status_path(line[3:])
        key = (status, path)
        if key in seen:
            continue
        seen.add(key)
        entries.append(classify(path, status))

    for line in ignored_lines:
        status = "!!"
        path = normalize_status_path(line[3:])
        key = (status, path)
        if key in seen:
            continue
        seen.add(key)
        entries.append(classify(path, status))

    return sorted(entries, key=lambda entry: (status_sort_key(entry.git_status), entry.path))


def classify(path: str, git_status: str) -> Entry:
    normalized = path.rstrip("/")
    lowered = normalized.lower()
    size = path_size(normalized)
    phase = phase_for(normalized)
    duplicate = ""

    category = "L_UNKNOWN"
    canonical_status = "unknown"
    action = "inspect_before_change"
    reason = "No confident classification rule matched."
    risk = "unknown_until_reviewed"
    suitability = "not_ready"

    if git_status == "!!":
        category = "I_LOCAL_ONLY"
        canonical_status = "ignored_local"
        action = "ignore_or_delete_after_local_review"
        reason = "Ignored by Git; local environment, cache, raw data, or generated bulk artifact."
        risk = "low_for_repository_if_not_canonical; high_if_deleted_without_local_backup"
        suitability = "exclude_from_github"

    if "/__pycache__/" in f"/{normalized}/" or lowered.endswith((".pyc", ".pyo")):
        category = "I_LOCAL_ONLY"
        canonical_status = "python_cache"
        action = "ignore_or_delete_after_local_review"
        reason = "Python bytecode/cache artifact, not source evidence."
        risk = "low_for_repository_if_deleted"
        suitability = "exclude_from_github"
    elif normalized.startswith("src/"):
        category = "A_SOURCE"
        canonical_status = "implementation_candidate"
        action = "keep_and_commit_if_tests_pass"
        reason = "Project source code under the allowed source tree."
        risk = "may_break reproducibility scripts if deleted"
        suitability = "github_suitable"
    elif normalized.startswith("tests/"):
        category = "B_TEST"
        canonical_status = "test_candidate"
        action = "keep_and_commit_with_matching_source"
        reason = "Project test code under the allowed test tree."
        risk = "may reduce auditability if deleted"
        suitability = "github_suitable"
    elif normalized.startswith("configs/"):
        category = "C_CONFIG"
        canonical_status = "config_candidate"
        action = "keep_and_commit"
        reason = "Dataset and phase settings belong under configs."
        risk = "may make scripted reproduction harder"
        suitability = "github_suitable"
    elif normalized in {"README.md", "PROJECT_SPEC.md", "CODEX_PROMPTS.md", "AGENTS.md", ".gitignore", ".gitattributes"}:
        category = "A_SOURCE"
        canonical_status = "canonical_project_control"
        action = "keep_and_commit"
        reason = "Root project control or collaborator documentation file."
        risk = "high_for_collaboration_if_deleted"
        suitability = "github_suitable"
    elif normalized == "docs" or normalized.startswith("docs/"):
        category = "A_SOURCE"
        canonical_status = "collaborator_documentation"
        action = "keep_and_commit_after_f6m_alignment"
        reason = "Small collaborator documentation."
        risk = "medium_for_onboarding_if_deleted"
        suitability = "github_suitable"
    elif normalized.startswith("report/archive/"):
        category = "E_HISTORICAL_AUDIT_EVIDENCE"
        canonical_status = "archived_history"
        action = "keep_and_commit"
        reason = "Archived report evidence retained outside the current-state entrypoints."
        risk = "medium_for_lineage_if_deleted"
        suitability = "github_suitable"
    elif normalized.startswith("report/"):
        if git_status.strip() == "D":
            category = "F_SUPERSEDED"
            canonical_status = "superseded_by_archive"
            action = "accept_deletion_after_archive_hash_verification"
            duplicate = archived_report_path(normalized)
            reason = "Tracked foreground report removed from current entrypoints after archive migration."
            risk = "medium_if_archive_copy_missing"
            suitability = "github_suitable_as_deletion_plus_archive"
        elif "phase_f6m" in lowered or normalized in {
            "report/final_scientific_report.md",
            "report/future_operations_context_handoff.md",
            "report/notes.md",
            "report/dataset_audit.md",
            "report/codebase_map.md",
            "report/github_readiness_cleanup_audit.md",
        }:
            category = "D_CANONICAL_SCIENTIFIC_ARTIFACT"
            canonical_status = "current_or_supporting_canonical"
            action = "keep_and_commit"
            reason = "Current F6M or collaborator-facing scientific documentation."
            risk = "high_for_auditability_if_deleted"
            suitability = "github_suitable"
        else:
            category = "E_HISTORICAL_AUDIT_EVIDENCE"
            canonical_status = "historical_lineage"
            action = "archive_or_keep_indexed"
            reason = "Phase report or historical audit material."
            risk = "medium_for_lineage_if_deleted"
            suitability = "github_suitable_if_archived_or_indexed"
    elif normalized == "archive" or normalized.startswith("archive/"):
        category = "G_DUPLICATE"
        canonical_status = "temporary_duplicate_archive_location"
        action = "migrate_to_report_or_results_archive_then_delete"
        duplicate = normalized.replace("archive/phase_reports/report/", "report/archive/phase_reports/")
        reason = "Root archive is a transitional local duplicate outside the documented repository layout."
        risk = "low_after_migration_and_hash_verification"
        suitability = "exclude_from_github"
    elif normalized.startswith("paper/"):
        if normalized == "paper/2411.00331v1.pdf":
            category = "D_CANONICAL_SCIENTIFIC_ARTIFACT"
            canonical_status = "canonical_paper_copy"
            action = "keep"
            reason = "Canonical paper file already part of the project."
            risk = "high_for_context_if_deleted"
            suitability = "github_suitable_if_license_allows"
        else:
            category = "G_DUPLICATE"
            canonical_status = "local_convenience_duplicate"
            action = "delete_after_approval"
            duplicate = "paper/2411.00331v1.pdf"
            reason = "Duplicate or local convenience paper artifact."
            risk = "low_for_repository; user may want local copy"
            suitability = "exclude_from_github"
    elif normalized.startswith(("data/raw/", "data/external/", "data/reconstructed/", "data/processed/finalization_")):
        category = "J_LARGE_EXTERNAL_DATA"
        canonical_status = "external_or_rebuildable_data"
        action = "ignore_or_externalize"
        reason = "Dataset/raw/processed artifact with redistribution and size concerns."
        risk = "high_if_deleted_without external backup; low for GitHub exclusion"
        suitability = "exclude_from_github"
    elif normalized.startswith("results/tables/phase_f6m_"):
        category = "D_CANONICAL_SCIENTIFIC_ARTIFACT"
        canonical_status = "f6m_canonical_table"
        action = "keep_and_commit_if_not_already_tracked"
        reason = "F6M closure table."
        risk = "high_for_final_claims_if_deleted"
        suitability = "github_suitable"
    elif normalized == "results/tables/repository_cleanup_inventory.csv":
        category = "E_HISTORICAL_AUDIT_EVIDENCE"
        canonical_status = "cleanup_audit"
        action = "keep_and_commit"
        reason = "Machine-readable cleanup decision table."
        risk = "medium_for_cleanup_traceability_if_deleted"
        suitability = "github_suitable"
    elif normalized.startswith("results/tables/"):
        category = "E_HISTORICAL_AUDIT_EVIDENCE"
        canonical_status = "historical_or_generated_table"
        action = "ignore_or_archive_by_reference"
        reason = "Generated result table outside the final F6M foreground set."
        risk = "medium_or_high_if_unique evidence and no manifest exists"
        suitability = "exclude_from_github_unless_selected"
    elif normalized.startswith(("results/llm/", "results/predictions/", "results/models/", "results/checkpoints/")):
        category = "J_LARGE_EXTERNAL_DATA"
        canonical_status = "external_raw_or_model_artifact"
        action = "externalize_with_hash_manifest"
        reason = "Bulk raw responses, predictions, checkpoints, or model states."
        risk = "high_if_deleted_without external copy; low for GitHub exclusion"
        suitability = "exclude_from_github"
    elif normalized.startswith(("results/logs/", "results/environment/")):
        category = "K_TEMPORARY"
        canonical_status = "local_runtime_or_log"
        action = "delete_or_ignore"
        reason = "Runtime, wheelhouse, test temp, or log artifact."
        risk = "low_for_science; medium_for debugging recent failures"
        suitability = "exclude_from_github"
    elif normalized == "results/fixtures" or normalized.startswith("results/fixtures/"):
        category = "B_TEST"
        canonical_status = "test_fixture"
        action = "keep_and_commit_if_tests_reference"
        reason = "Small deterministic fixture referenced by tests."
        risk = "may break tests if deleted"
        suitability = "github_suitable"
    elif normalized == "results/archive" or normalized.startswith("results/archive/"):
        category = "E_HISTORICAL_AUDIT_EVIDENCE"
        canonical_status = "archived_generated_result"
        action = "keep_and_commit"
        reason = "Selected generated result artifact migrated out of foreground results."
        risk = "medium_for_historical_plot_traceability_if_deleted"
        suitability = "github_suitable_if_archived_or_indexed"
    elif normalized.startswith(("results/raw_outputs/", "results/metrics/", "results/plots/")):
        if git_status.strip() == "D":
            category = "F_SUPERSEDED"
            canonical_status = "superseded_generated_artifact"
            action = "accept_deletion_after_archive_hash_verification"
            duplicate = archived_result_path(normalized)
            reason = "Old generated artifact removed from foreground results."
            risk = "low_if_archive_or regeneration path exists"
            suitability = "github_suitable_as_deletion_plus_archive"
        else:
            category = "H_GENERATED_REBUILDABLE"
            canonical_status = "generated_output"
            action = "ignore_or_archive_if_unique"
            reason = "Generated metric/plot/raw-output artifact outside F6M foreground."
            risk = "medium_if no regeneration path or archive"
            suitability = "exclude_from_github_unless_selected"

    return Entry(
        path=normalized,
        git_status=git_status,
        size_bytes=size,
        category=category,
        phase=phase,
        canonical_status=canonical_status,
        duplicate_or_superseded_by=duplicate,
        recommended_action=action,
        reason=reason,
        risk_if_deleted=risk,
        github_suitability=suitability,
    )


def write_inventory(entries: list[Entry]) -> None:
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INVENTORY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        for entry in entries:
            writer.writerow({column: getattr(entry, column) for column in INVENTORY_COLUMNS})


def write_report(
    entries: list[Entry],
    *,
    branch: str,
    head: str,
    status_lines: list[str],
    ignored_lines: list[str],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    category_counts = Counter(entry.category for entry in entries)
    action_counts = Counter(entry.recommended_action for entry in entries)
    suitability_counts = Counter(entry.github_suitability for entry in entries)
    status_counts = Counter(entry.git_status for entry in entries)
    total_size = sum(entry.size_bytes for entry in entries)
    unknown = [entry for entry in entries if entry.category == "L_UNKNOWN"]
    original_repo_dirty = any(entry.path.startswith("original_repo/") for entry in entries if entry.git_status != "!!")

    lines = [
        "# GitHub Readiness Cleanup Audit",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Baseline",
        "",
        f"- Branch: `{branch}`",
        f"- Cleanup base HEAD at audit generation: `{head}`",
        f"- Frozen reproduction commit: `{FROZEN_COMMIT}`",
        f"- Decision token: `{DECISION_TOKEN}`",
        f"- Status entries before cleanup classification: `{len(status_lines)}`",
        f"- Cleanup audit self-refresh paths excluded from status baseline: `{len(SELF_REFRESH_PATHS)}`",
        f"- Ignored entries included from collapsed Git status: `{len(ignored_lines)}`",
        f"- Inventory rows: `{len(entries)}`",
        f"- Inventory size represented: `{total_size}` bytes",
        f"- `original_repo/` dirty in publishable status: `{'yes' if original_repo_dirty else 'no'}`",
        "",
        "## Category Counts",
        "",
        markdown_count_table(category_counts, "category", "rows"),
        "",
        "## Recommended Action Counts",
        "",
        markdown_count_table(action_counts, "recommended_action", "rows"),
        "",
        "## GitHub Suitability Counts",
        "",
        markdown_count_table(suitability_counts, "github_suitability", "rows"),
        "",
        "## Git Status Counts",
        "",
        markdown_count_table(status_counts, "git_status", "rows"),
        "",
        "## Cleanup Decisions",
        "",
        "- Keep and commit source, tests, configs, F6M foreground tables, current report entrypoints, and F6M-aligned docs.",
        "- Archive superseded foreground reports and old Plan C plots only after hash verification.",
        "- Ignore or externalize raw datasets, processed bulk data, raw LLM responses, predictions, model checkpoints, local environments, and runtime logs.",
        "- Leave the transitional root archive as ignored local material unless a separate explicit recursive-deletion approval is given.",
        "- Do not start any extension, inference, model training, download, or result repair as part of cleanup.",
        "",
        "## Unknown Status",
        "",
        f"- Relevant unknown rows: `{len(unknown)}`",
    ]
    if unknown:
        lines.extend(["", "| path | status | size_bytes |", "| --- | --- | ---: |"])
        lines.extend(f"| `{entry.path}` | `{entry.git_status}` | {entry.size_bytes} |" for entry in unknown[:50])
    else:
        lines.append("- No relevant unknown rows remain after rule-based classification.")

    lines.extend(
        [
            "",
            "## Final Validation Requirements",
            "",
            "- `git status --short` must contain only intentional staged/unstaged changes until commits are created.",
            "- Run focused tests for modified helper/source bundles and F6M closure.",
            "- Run secret and operational absolute-path scans before final commit.",
            "- Confirm `original_repo/` remains unmodified.",
            "",
            "## Machine-Readable Inventory",
            "",
            f"- `{INVENTORY_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def markdown_count_table(counter: Counter[str], first_column: str, second_column: str) -> str:
    rows = [f"| {first_column} | {second_column} |", "| --- | ---: |"]
    rows.extend(f"| `{key}` | {value} |" for key, value in sorted(counter.items()))
    return "\n".join(rows)


def normalize_status_path(raw: str) -> str:
    path = raw.strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.replace("\\", "/")


def path_size(path: str) -> int:
    target = PROJECT_ROOT / Path(path)
    try:
        if target.is_file():
            return target.stat().st_size
        if target.is_dir():
            total = 0
            for current, _, files in os.walk(target, onerror=lambda _: None):
                for name in files:
                    candidate = Path(current) / name
                    try:
                        total += candidate.stat().st_size
                    except OSError:
                        continue
            return total
    except OSError:
        return 0
    return 0


def phase_for(path: str) -> str:
    lowered = path.lower()
    for token in [
        "phase_f6m",
        "phase_f6l",
        "phase_f6k",
        "phase_f6j",
        "phase_f6i2",
        "phase_f6i1",
        "phase_f6h",
        "phase_f6g",
        "phase_f6f",
        "phase_f6e",
        "finalization_f6",
        "finalization_f5",
        "finalization_f4",
        "finalization_f3",
        "finalization_f2",
        "plan_e",
        "plan_d",
        "plan_c",
        "plan_b",
        "canonical_local_llm",
        "canonical_llm",
    ]:
        if token in lowered:
            return token
    if path.startswith("data/"):
        return "data"
    if path.startswith("results/"):
        return "results"
    if path.startswith("report/"):
        return "report"
    return "current_project"


def archived_report_path(path: str) -> str:
    return "report/archive/phase_reports/" + Path(path).name


def archived_result_path(path: str) -> str:
    if path.startswith("results/plots/"):
        return "results/archive/plots/" + Path(path).name
    return ""


def status_sort_key(status: str) -> int:
    return {" M": 0, " D": 1, "??": 2, "!!": 3}.get(status, 9)


def git_stdout(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    return completed.stdout.strip()


def git_lines(*args: str) -> list[str]:
    completed = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    return [line for line in completed.stdout.splitlines() if line.strip()]


if __name__ == "__main__":
    main()

"""Build deterministic workspace forensic audit artifacts.

This builder is intentionally offline. It inventories current repository
evidence, reconstructs conservative traceability tables from existing files, and
does not train models, call LLMs, download data, or mutate protected inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit import common_git, common_hashing

RECOVERED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"

REPORT_PATH = PROJECT_ROOT / "report" / "workspace_forensic_state_audit.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"

INVENTORY_CSV = TABLES_DIR / "workspace_relevant_file_inventory.csv"
PHASE_CSV = TABLES_DIR / "workspace_phase_and_supersession_registry.csv"
PAPER_CONTRACT_CSV = TABLES_DIR / "paper_atomic_contract.csv"
CONFLICTS_CSV = TABLES_DIR / "paper_repository_contract_conflicts.csv"
TRACEABILITY_CSV = TABLES_DIR / "paper_to_local_traceability.csv"
CLAIMS_CSV = TABLES_DIR / "project_claim_verification.csv"
STATUS_MATRIX_CSV = TABLES_DIR / "current_reproduction_status_matrix.csv"
LINEAGE_CSV = TABLES_DIR / "artifact_lineage_and_supersession.csv"
NUMERIC_RECON_CSV = TABLES_DIR / "numeric_result_reconciliation.csv"
BLOCKERS_CSV = TABLES_DIR / "blockers_gaps_and_next_actions.csv"
SNAPSHOT_JSON = MANIFESTS_DIR / "current_project_context_snapshot.json"

CURRENT_AUDIT_OUTPUTS = {
    "report/workspace_forensic_state_audit.md",
    "report/future_operations_context_handoff.md",
    "results/tables/workspace_relevant_file_inventory.csv",
    "results/tables/workspace_phase_and_supersession_registry.csv",
    "results/tables/paper_atomic_contract.csv",
    "results/tables/paper_repository_contract_conflicts.csv",
    "results/tables/paper_to_local_traceability.csv",
    "results/tables/project_claim_verification.csv",
    "results/tables/current_reproduction_status_matrix.csv",
    "results/tables/artifact_lineage_and_supersession.csv",
    "results/tables/numeric_result_reconciliation.csv",
    "results/tables/blockers_gaps_and_next_actions.csv",
    "results/manifests/current_project_context_snapshot.json",
}

AUDIT_CREATED_STATUS_PATHS = {
    "src/analysis/build_workspace_forensic_state_audit.py",
    "tests/test_workspace_forensic_state_audit.py",
    "results/tables/workspace_relevant_file_inventory.csv",
    "results/tables/workspace_phase_and_supersession_registry.csv",
    "results/tables/paper_atomic_contract.csv",
    "results/tables/paper_repository_contract_conflicts.csv",
    "results/tables/paper_to_local_traceability.csv",
    "results/tables/project_claim_verification.csv",
    "results/tables/current_reproduction_status_matrix.csv",
    "results/tables/artifact_lineage_and_supersession.csv",
    "results/tables/numeric_result_reconciliation.csv",
    "results/tables/workspace_forensic_audit_test_results.csv",
    "results/tables/workspace_forensic_audit_determinism.csv",
}

EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    ".venv_audit_recovery",
    ".venv_f6g2_py312",
    ".venv_repro",
    ".venv_repro_system_site",
    "__pycache__",
    ".pytest_cache",
}
EXCLUDE_REL_PREFIXES = {
    "results/environment/cpython-3.12.10-amd64",
    "results/environment/phase_f6f1r_isolated_f6f1",
    "results/logs/phase_f6g2r_pytest_tmp",
    "results/logs/phase_f6g2r_manual_pytest_tmp",
}

DATASETS = ["Amazon Beauty", "Amazon Sports", "ML-1M", "LastFM"]
TASKS = ["ranking", "reranking"]
LLM_MODELS = ["Qwen2", "Mistral", "GPT-family", "Claude-family", "Gemini/PaLM-family", "Llama-family", "Other paper LLM"]
TRADITIONAL_MODELS = ["Random", "MostPop", "BM25", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"]
PROMPTS = ["Base/OpenP5", "LLMRank", "LLMSRec"]
METRICS = ["hr_at_5", "ndcg_at_5", "hallucination", "coverage", "novelty", "long_tail", "serendipity", "fairness", "diversity"]


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


def run_command(args: list[str], timeout: int = 30) -> CommandResult:
    proc = subprocess.run(args, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=timeout)
    return CommandResult(" ".join(args), proc.returncode, proc.stdout.strip(), proc.stderr.strip())


def git_status_lines(status_text: str) -> list[str]:
    return common_git.git_status_lines(status_text)


def git_status_path(line: str) -> str:
    return common_git.git_status_path(line)


def git_status_counts(lines: Iterable[str]) -> dict[str, int]:
    return common_git.git_status_counts(lines)


def audit_resume_git_baseline(final_git_status: str) -> dict[str, Any]:
    return common_git.audit_resume_git_baseline(final_git_status, AUDIT_CREATED_STATUS_PATHS)


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def file_sha256(path: Path) -> str:
    return common_hashing.sha256_file(path).upper()


def read_text(path: Path, limit: int | None = None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[:limit]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: tuple(str(row.get(field, "")) for field in fields))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv_rows(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows if max_rows is None else rows[:max_rows]


def git_state_maps() -> tuple[set[str], dict[str, str]]:
    return common_git.git_state_maps_tuple(PROJECT_ROOT)


def excluded(path: Path) -> bool:
    parts = set(path.parts)
    if parts.intersection(EXCLUDE_DIR_NAMES):
        return True
    try:
        relative = rel(path)
    except ValueError:
        return True
    return any(relative == prefix or relative.startswith(prefix + "/") for prefix in EXCLUDE_REL_PREFIXES)


def classify_category(path: str) -> str:
    if path.startswith("paper/"):
        return "paper"
    if path.startswith("original_repo/"):
        return "official_repository"
    if path.startswith("data/raw/"):
        return "raw_data"
    if path.startswith("data/"):
        return "local_data"
    if path.startswith("results/llm/"):
        return "raw_llm_output"
    if path.startswith("results/checkpoints/"):
        return "model_checkpoint"
    if path.startswith("results/predictions/"):
        return "prediction"
    if path.startswith("results/manifests/"):
        return "manifest"
    if path.startswith("results/tables/"):
        return "result_table"
    if path.startswith("report/"):
        return "report"
    if path.startswith("src/"):
        return "source_code"
    if path.startswith("tests/"):
        return "test"
    if path.startswith("configs/"):
        return "config"
    if path.startswith("docs/"):
        return "docs"
    return "root_or_other"


def phase_family(path: str) -> str:
    lower = path.lower()
    patterns = [
        "phase_f6g2r",
        "phase_f6g2",
        "phase_f6g1",
        "phase_f6f2",
        "phase_f6f1",
        "phase_f6e4",
        "phase_f6e3",
        "phase_f6e1",
        "finalization_f6d4a",
        "finalization_f6d3",
        "finalization_f6d2",
        "finalization_f6d1",
        "finalization_f6c",
        "finalization_f6b",
        "finalization_f6a",
        "finalization_f5",
        "finalization_f4",
        "finalization_f3",
        "finalization_f2",
        "canonical_local_llm",
        "canonical_metric",
        "canonical_llm",
        "plan_e",
        "plan_d",
        "plan_c",
        "plan_b",
    ]
    for pattern in patterns:
        if pattern in lower:
            return pattern
    return "unassigned"


def infer_protected(path: str, category: str) -> str:
    if category in {"paper", "official_repository", "raw_data", "raw_llm_output", "model_checkpoint"}:
        return "protected_input"
    if "freeze" in path.lower() or "protected_input" in path.lower():
        return "protected_or_manifested"
    if category in {"result_table", "report", "prediction", "local_data"}:
        return "derived_evidence"
    return "not_protected_by_default"


def build_inventory() -> list[dict[str, Any]]:
    tracked, status = git_state_maps()
    rows: list[dict[str, Any]] = []
    seen_hashes: Counter[str] = Counter()
    files: list[Path] = []
    for root, dirs, names in os.walk(PROJECT_ROOT):
        root_path = Path(root)
        dirs[:] = sorted(d for d in dirs if not excluded(root_path / d))
        for name in sorted(names):
            path = root_path / name
            if excluded(path):
                continue
            files.append(path)
    for path in sorted(files, key=lambda p: rel(p).lower()):
        relative = rel(path)
        if relative in CURRENT_AUDIT_OUTPUTS:
            digest = "CURRENT_AUDIT_OUTPUT_SELF_REFERENCE"
            inspected = "generated_current_audit_self_reference"
            notes = "hash and size intentionally fixed to avoid self-referential nondeterminism"
            size_bytes = 0
        else:
            size_bytes = path.stat().st_size
            try:
                digest = file_sha256(path)
                inspected = "hash_schema_or_text_sample"
                notes = ""
            except Exception as exc:  # pragma: no cover - records filesystem edge cases
                digest = ""
                inspected = "unreadable"
                notes = f"{type(exc).__name__}: {exc}"
        if digest:
            seen_hashes[digest] += 1
        category = classify_category(relative)
        rows.append(
            {
                "path": relative,
                "file_type": path.suffix.lower().lstrip(".") or "no_extension",
                "category": category,
                "phase_family": phase_family(relative),
                "size_bytes": size_bytes,
                "sha256": digest,
                "git_state": status.get(relative, "tracked_clean" if relative in tracked else "untracked_or_ignored"),
                "tracked_status": "tracked" if relative in tracked else "untracked",
                "producer": infer_producer(relative),
                "inputs": infer_inputs(relative),
                "consumers": infer_consumers(relative),
                "evidence_level": evidence_level(category),
                "currentness": infer_currentness(relative),
                "protected_status": infer_protected(relative, category),
                "inspection_status": inspected,
                "notes": notes,
            }
        )
    for row in rows:
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + (
            "exact_duplicate_hash" if row["sha256"] and seen_hashes[row["sha256"]] > 1 else ""
        )
    return rows


def infer_producer(path: str) -> str:
    if path.startswith("results/tables/"):
        stem = Path(path).stem
        builder = PROJECT_ROOT / "src" / "experiments" / f"build_{stem}.py"
        if builder.exists():
            return rel(builder)
        phase = phase_family(path)
        candidates = list((PROJECT_ROOT / "src").glob(f"**/*{phase}*.py")) if phase != "unassigned" else []
        return ";".join(rel(c) for c in sorted(candidates)[:3]) if candidates else "unknown_or_manual"
    if path.startswith("report/"):
        phase = phase_family(path)
        candidates = list((PROJECT_ROOT / "src").glob(f"**/*{phase}*.py")) if phase != "unassigned" else []
        return ";".join(rel(c) for c in sorted(candidates)[:3]) if candidates else "narrative_or_unknown"
    if path.startswith("data/raw/"):
        return "external_dataset_source"
    if path.startswith("original_repo/"):
        return "upstream_repository_snapshot"
    if path.startswith("paper/"):
        return "paper_source"
    return "repository_file"


def infer_inputs(path: str) -> str:
    if "f6g2" in path.lower():
        return "F6G1/F6E/F6F manifests and LastFM candidate evidence"
    if "f6d4a" in path.lower():
        return "F6D3 manifests, LastFM ranking candidates, local raw LLM outputs"
    if "f6d2" in path.lower():
        return "F6C LastFM split and adapted reranking candidates"
    if "canonical_metric" in path.lower():
        return "canonical ranking rows, train item statistics, metric contracts"
    if "f5af" in path.lower():
        return "paper numeric extraction and structured sources"
    return ""


def infer_consumers(path: str) -> str:
    if path.endswith(".csv") and path.startswith("results/tables/"):
        return "reports; downstream audits; current forensic audit"
    if path.startswith("report/"):
        return "human scientific review; current forensic audit"
    if path.startswith("results/manifests/"):
        return "lineage validation; current forensic audit"
    return "current forensic audit"


def evidence_level(category: str) -> str:
    return {
        "raw_data": "primary_raw_input",
        "paper": "paper_primary_source",
        "official_repository": "official_repository_source",
        "raw_llm_output": "primary_raw_output",
        "source_code": "executable_source",
        "config": "configuration",
        "manifest": "provenance_manifest",
        "result_table": "derived_structured_evidence",
        "report": "narrative_claim_source",
        "test": "validation_code",
    }.get(category, "supporting_file")


def infer_currentness(path: str) -> str:
    lower = path.lower()
    if "workspace_forensic_state_audit" in lower or "future_operations_context_handoff" in lower:
        return "current_forensic_audit_output"
    if "phase_f6g2_output_length_contract_resolution" in lower:
        return "current_contract_resolution"
    if "phase_f6g2r" in lower:
        return "historical_blocked_by_contract_conflict_before_resolution"
    if "phase_f6g2" in lower or "phase_f6g1" in lower:
        return "superseded_for_execution_by_top10_resolution"
    if "final" in lower and ("f6" not in lower and "finalization" not in lower):
        return "legacy_final_named_review_required"
    if "f6d4a" in lower or "f6d2" in lower or "f6f2" in lower or "f6f1" in lower:
        return "late_phase_current_or_recent_evidence"
    return "unresolved_currentness"


def build_phase_registry(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in inventory:
        phase = row["phase_family"]
        if phase == "unassigned":
            continue
        grouped[phase][row["category"]].append(row["path"])
    rows: list[dict[str, Any]] = []
    for phase in sorted(grouped):
        cats = grouped[phase]
        status = "current_or_recent"
        supersedes = ""
        superseded_by = ""
        notes = ""
        if phase in {"phase_f6g1", "phase_f6g2", "phase_f6g2r"}:
            status = "superseded_or_blocked_for_execution"
            superseded_by = "phase_f6g2_output_length_contract_resolution"
            notes = "Top-20 LastFM reranking request contract is superseded for execution by repository-derived top-10 resolution."
        if phase == "phase_f6g2":
            status = "contract_resolution_current" if any("output_length_contract_resolution" in p for p in sum(cats.values(), [])) else status
        if phase == "finalization_f6b":
            supersedes = "plan_d_phase_5e coverage taxonomy"
        rows.append(
            {
                "phase_id": phase,
                "objective": phase_objective(phase),
                "predecessor": predecessor(phase),
                "successor": successor(phase),
                "source_files": join_paths(cats.get("source_code", [])[:8]),
                "config_files": join_paths(cats.get("config", [])[:8]),
                "test_files": join_paths(cats.get("test", [])[:8]),
                "output_files": join_paths((cats.get("result_table", []) + cats.get("manifest", []) + cats.get("prediction", []))[:12]),
                "report_files": join_paths(cats.get("report", [])[:8]),
                "decision_token": phase_decision_token(phase),
                "evidence_generated": ";".join(sorted(cats)),
                "contract_changed": "yes" if phase in {"phase_f6g2", "finalization_f6b", "phase_f6e3b", "phase_f6e4a"} else "unknown_or_no",
                "current_status": status,
                "supersedes": supersedes,
                "superseded_by": superseded_by,
                "notes": notes,
            }
        )
    return rows


def join_paths(paths: Iterable[str]) -> str:
    return ";".join(sorted(paths))


def phase_objective(phase: str) -> str:
    mapping = {
        "phase_f6g2": "LastFM reranking output-length conflict resolution and/or blocked execution evidence",
        "phase_f6g2r": "Runtime recovery preflight before top10/top20 resolution",
        "phase_f6g1": "LastFM reranking reconstruction and future top20 manifests",
        "finalization_f6d4a": "Adapted LastFM LLM ranking closure",
        "finalization_f6d2": "Adapted LastFM trainable model runs",
        "finalization_f6b": "Atomic paper scope and coverage rebaseline",
        "finalization_f5af": "Atomic paper numeric cell extraction",
        "canonical_metric": "Canonical local metric contract and computations",
        "canonical_local_llm": "Local indexed LLM execution and evaluation evidence",
        "plan_e": "Traditional/neural baseline expansion and diagnostics",
        "plan_d": "Earlier reconstruction and robustness consolidation",
    }
    return mapping.get(phase, f"Workspace phase family {phase}")


def predecessor(phase: str) -> str:
    order = ["plan_d", "plan_e", "finalization_f2", "finalization_f3", "finalization_f4", "finalization_f5", "finalization_f6a", "finalization_f6b", "finalization_f6c", "finalization_f6d1", "finalization_f6d2", "finalization_f6d3", "finalization_f6d4a", "phase_f6e1", "phase_f6e3", "phase_f6e4", "phase_f6f1", "phase_f6f2", "phase_f6g1", "phase_f6g2", "phase_f6g2r"]
    return order[order.index(phase) - 1] if phase in order and order.index(phase) > 0 else ""


def successor(phase: str) -> str:
    order = ["plan_d", "plan_e", "finalization_f2", "finalization_f3", "finalization_f4", "finalization_f5", "finalization_f6a", "finalization_f6b", "finalization_f6c", "finalization_f6d1", "finalization_f6d2", "finalization_f6d3", "finalization_f6d4a", "phase_f6e1", "phase_f6e3", "phase_f6e4", "phase_f6f1", "phase_f6f2", "phase_f6g1", "phase_f6g2", "phase_f6g2r"]
    return order[order.index(phase) + 1] if phase in order and order.index(phase) < len(order) - 1 else ""


def phase_decision_token(phase: str) -> str:
    report_files = sorted((PROJECT_ROOT / "report").glob(f"*{phase}*.md"))
    for path in report_files:
        text = read_text(path, limit=2000)
        match = re.search(r"Decision(?:e finale|e)?:\s*`?([^`\n.]+)`?", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def paper_contract_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    atomic = read_csv_rows(TABLES_DIR / "finalization_f5af_atomic_paper_result_cells.csv")
    for source in atomic:
        rows.append(
            {
                "contract_id": source.get("atomic_cell_id", ""),
                "paper_section": source.get("paper_section", ""),
                "paper_page": source.get("paper_page", ""),
                "paper_table": source.get("paper_table_number", ""),
                "paper_figure": "",
                "paper_equation": "",
                "dimension": "numeric_result_cell",
                "task": source.get("task", ""),
                "dataset": source.get("dataset", ""),
                "model": source.get("model", ""),
                "prompt_strategy": source.get("prompt_or_setting", ""),
                "cohort": "paper_reported",
                "history_policy": "paper_unverified_from_pdf" if source.get("pdf_verified") == "false" else "",
                "candidate_policy": "paper_unverified_from_pdf",
                "candidate_count": "",
                "requested_output_length": "paper_unspecified_or_unverified",
                "metric": source.get("metric", ""),
                "cutoff": source.get("cutoff", ""),
                "denominator": "paper_unverified_from_pdf",
                "repetition_policy": "",
                "statistical_method": "",
                "paper_value": source.get("paper_value_numeric", source.get("paper_value_text", "")),
                "paper_specification_status": "secondary_structured_source_pdf_unverified" if source.get("pdf_verified") == "false" else "pdf_verified",
                "notes": source.get("notes", ""),
            }
        )
    base_contracts = [
        ("PAPER-SCOPE-DATASETS", "scope", "", "", "dataset_scope", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "four paper datasets recovered from F6B reports: Beauty, Sports, ML-1M, LastFM", "paper_reported_in_existing_audit"),
        ("PAPER-SCOPE-TASKS", "scope", "", "", "task_scope", "ranking;reranking", "", "", "", "", "", "", "", "", "", "", "", "", "", "ranking and reranking task families", "paper_reported_in_existing_audit"),
        ("PAPER-SCOPE-MODELS", "scope", "", "", "model_scope", "", "", "7 LLM families; 6 traditional models", "", "", "", "", "", "", "", "", "", "", "", "Exact versions remain unresolved unless paper PDF or source table verifies them.", "paper_partially_recovered"),
        ("PAPER-CANDIDATE-POOL", "method", "", "", "candidate_pool", "ranking;reranking", "", "", "", "", "", "paper_unverified_from_pdf", "20", "paper_unspecified", "", "", "", "", "", "Candidate pool size 20 is supported by prior output-length resolution report; output length remains repository-derived.", "paper_partially_recovered"),
        ("PAPER-METRIC-K", "metric", "", "", "metric_cutoff", "ranking;reranking", "", "", "", "", "", "", "", "", "hr_at_5;ndcg_at_5", "5", "", "", "", "Top-5 utility metrics recovered from structured prior evidence.", "paper_partially_recovered"),
    ]
    for row in base_contracts:
        rows.append(
            {
                "contract_id": row[0],
                "paper_section": row[1],
                "paper_page": row[2],
                "paper_table": row[3],
                "paper_figure": "",
                "paper_equation": "",
                "dimension": row[4],
                "task": row[5],
                "dataset": row[6],
                "model": row[7],
                "prompt_strategy": row[8],
                "cohort": row[9],
                "history_policy": row[10],
                "candidate_policy": row[11],
                "candidate_count": row[12],
                "requested_output_length": row[13],
                "metric": row[14],
                "cutoff": row[15],
                "denominator": row[16],
                "repetition_policy": row[17],
                "statistical_method": row[18],
                "paper_value": "",
                "paper_specification_status": row[20],
                "notes": row[19],
            }
        )
    return rows


def conflict_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    f6g2 = read_csv_rows(TABLES_DIR / "phase_f6g2_output_length_contract_resolution.csv")
    if f6g2:
        for idx, source in enumerate(f6g2, start=1):
            rows.append(
                {
                    "conflict_id": f"CONFLICT-OUTPUT-{idx:03d}",
                    "topic": f"{source.get('task','')}/{source.get('prompting_strategy','')} output length",
                    "paper_evidence": source.get("original_paper_cell", ""),
                    "repository_evidence": source.get("repository_request_evidence", ""),
                    "paper_value": source.get("paper_requested_output_length", ""),
                    "repository_value": source.get("requested_output_length", ""),
                    "conflict_type": source.get("paper_specificity", ""),
                    "scientific_consequence": "Top-10 execution is repository-derived, not paper-explicit; top-20 F6G1/F6G2 manifests are not executable canonical artifacts.",
                    "resolution_policy": source.get("authority_decision", ""),
                    "current_resolution": source.get("conflict_outcome", ""),
                    "notes": source.get("notes", ""),
                }
            )
    rows.extend(
        [
            {
                "conflict_id": "CONFLICT-RUNTIME-001",
                "topic": "original runtime vs recovered audit runtime",
                "paper_evidence": "not_applicable",
                "repository_evidence": "report/python_environment_recovery_audit.md",
                "paper_value": "not_applicable",
                "repository_value": "original Store base not recovered; repaired clone exact package tree",
                "conflict_type": "runtime_provenance_difference",
                "scientific_consequence": "Future executable checks must not claim exact original runtime reproduction.",
                "resolution_policy": "use approved recovered interpreter with PYTHONNOUSERSITE=1 and provenance labels",
                "current_resolution": "defensible_repaired_clone_exact_package_versions",
                "notes": "User explicitly approved interpreter contract change before this audit.",
            },
            {
                "conflict_id": "CONFLICT-PAPER-PDF-001",
                "topic": "paper PDF extraction availability",
                "paper_evidence": "paper/2411.00331v1.pdf and duplicate PDF are present",
                "repository_evidence": "no PDF parser package validated in the project runtime during prior planning",
                "paper_value": "primary PDF exists",
                "repository_value": "structured paper cells derive from prior extraction artifacts",
                "conflict_type": "source_access_limitation",
                "scientific_consequence": "Paper contract rows remain paper_unverified_from_pdf where only secondary structured evidence exists.",
                "resolution_policy": "mark unverified instead of fabricating PDF-confirmed details",
                "current_resolution": "paper_partially_recovered",
                "notes": "The two paper PDFs were previously shown byte-identical.",
            },
        ]
    )
    return rows


def traceability_rows(paper_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    local_mapping = read_csv_rows(TABLES_DIR / "finalization_f5af_local_atomic_mapping.csv")
    mapping_by_contract = {row.get("atomic_cell_id", row.get("contract_id", "")): row for row in local_mapping}
    rows: list[dict[str, Any]] = []
    for paper in paper_rows:
        contract_id = paper["contract_id"]
        local = mapping_by_contract.get(contract_id, {})
        is_numeric = paper["dimension"] == "numeric_result_cell"
        rows.append(
            {
                "contract_id": contract_id,
                "local_source": local.get("local_source", ""),
                "local_config": local.get("local_config", ""),
                "local_input": local.get("local_input", ""),
                "local_raw_output": local.get("local_raw_output", ""),
                "local_derived_output": local.get("local_result_source", local.get("local_derived_output", "")),
                "local_test": local.get("local_test", ""),
                "local_report": local.get("local_report", ""),
                "execution_status": local.get("execution_status", "not_executed" if is_numeric else "partial"),
                "paper_alignment": local.get("paper_alignment", "unresolved" if is_numeric else "reconstructed_from_paper_and_repository"),
                "repository_alignment": local.get("repository_alignment", "unresolved"),
                "evidence_quality": local.get("evidence_quality", "secondary_paper_contract_only" if is_numeric else "deterministic_derived_evidence"),
                "currentness": "current_audit_recomputed",
                "material_divergences": local.get("material_divergences", "paper PDF verification incomplete" if is_numeric else ""),
                "notes": local.get("notes", ""),
            }
        )
    return rows


def claim_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_files = sorted((PROJECT_ROOT / "report").glob("*.md")) + [PROJECT_ROOT / "README.md", PROJECT_ROOT / "report" / "notes.md"]
    claim_patterns = [
        r"\bcomplete(?:d)?\b",
        r"\bfully reproduced\b",
        r"\bexact\b",
        r"\bpaper-aligned\b",
        r"\bcanonical\b",
        r"\ball tests pass\b",
        r"\b100%\b",
        r"\bblocked\b",
        r"\bsuperseded\b",
    ]
    combined = re.compile("|".join(claim_patterns), flags=re.IGNORECASE)
    claim_id = 1
    for path in source_files:
        if not path.exists() or excluded(path):
            continue
        for line_no, line in enumerate(read_text(path).splitlines(), start=1):
            stripped = line.strip(" -\t")
            if len(stripped) < 12:
                continue
            if not combined.search(stripped):
                continue
            verdict, corrective = classify_claim(path, stripped)
            rows.append(
                {
                    "claim_id": f"CLAIM-{claim_id:05d}",
                    "claim_text": stripped[:500],
                    "source_file": rel(path),
                    "source_location": f"line {line_no}",
                    "source_phase": phase_family(rel(path)),
                    "claimed_status": extract_claim_status(stripped),
                    "claimed_evidence": "narrative source",
                    "primary_evidence": primary_evidence_for_claim(stripped),
                    "verification_method": "keyword extraction plus phase/supersession evidence",
                    "verdict": verdict,
                    "confidence": "medium",
                    "currentness": infer_currentness(rel(path)),
                    "contradiction": contradiction_for_claim(stripped),
                    "corrective_wording": corrective,
                }
            )
            claim_id += 1
            if claim_id > 250:
                break
        if claim_id > 250:
            break
    return rows


def classify_claim(path: Path, claim: str) -> tuple[str, str]:
    lower = claim.lower()
    path_lower = rel(path).lower()
    if "100%" in lower or "fully reproduced" in lower or "exact numerical" in lower:
        return "unsupported", "Do not claim full or exact reproduction; use separate exact, adapted, operational, and numeric-cell coverage."
    if "phase_f6g2r" in path_lower and "blocked" in lower:
        return "verified_historical_only", "F6G2R was blocked historically; later output-length resolution supersedes the top20 execution blocker."
    if "phase_f6g2_output_length_contract_resolution" in path_lower:
        return "verified_current", "Repository-derived top10 output contract is current but not paper-explicit."
    if "complete" in lower and ("f6d2" in path_lower or "f6d4a" in path_lower):
        return "verified_but_adapted", "Completed as local adapted evidence, not exact paper replication."
    if "canonical" in lower:
        return "partially_verified", "Canonical labels are accepted only within their phase scope after checking lineage."
    if "blocked" in lower:
        return "partially_verified", "Blocked claims require currentness checks against supersession tables."
    return "ambiguous", "Retain as historical/narrative until primary evidence is checked."


def extract_claim_status(claim: str) -> str:
    for word in ["complete", "completed", "blocked", "canonical", "exact", "100%", "superseded"]:
        if word in claim.lower():
            return word
    return "material_claim"


def primary_evidence_for_claim(claim: str) -> str:
    lower = claim.lower()
    if "f6g2" in lower:
        return "results/tables/phase_f6g2_output_length_contract_resolution.csv"
    if "f6d2" in lower:
        return "results/tables/finalization_f6d2_validation.csv"
    if "f6d4a" in lower:
        return "results/tables/finalization_f6d4a_closure_protected_input_verification.csv"
    if "numeric" in lower:
        return "results/tables/finalization_f5af_atomic_paper_result_cells.csv"
    return "current forensic audit inventory"


def contradiction_for_claim(claim: str) -> str:
    lower = claim.lower()
    if "exact" in lower or "100%" in lower or "fully reproduced" in lower:
        return "Exact paper replication gate fails when exact model versions, prompts, runtime, PDF verification, or protocol cells are unresolved."
    return ""


def status_matrix_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for task in TASKS:
            for model in ["Qwen2", "Mistral", "MostPop", "BM25", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"]:
                prompt_values = PROMPTS if model in {"Qwen2", "Mistral"} else [""]
                for prompt in prompt_values:
                    for metric in ["hr_at_5", "ndcg_at_5"]:
                        execution, paper_align, repo_align, evidence, notes = classify_status(dataset, task, model, prompt)
                        rows.append(
                            {
                                "dataset": dataset,
                                "task": task,
                                "model": model,
                                "prompt_strategy": prompt,
                                "metric": metric,
                                "experimental_condition": "standard_candidate_pool_or_local_adaptation",
                                "execution_status": execution,
                                "paper_alignment": paper_align,
                                "repository_alignment": repo_align,
                                "evidence_quality": evidence,
                                "currentness": "current_audit_classification",
                                "local_result_source": local_result_source(dataset, task, model),
                                "notes": notes,
                            }
                        )
    return rows


def classify_status(dataset: str, task: str, model: str, prompt: str) -> tuple[str, str, str, str, str]:
    if dataset == "LastFM" and task == "reranking" and model in {"Qwen2", "Mistral"}:
        return ("blocked", "adapted_local_protocol", "repository_derived_reimplementation", "deterministic_derived_evidence", "LastFM reranking LLM execution remains blocked pending top10 corrective freeze.")
    if dataset == "LastFM" and model in {"Qwen2", "Mistral"} and task == "ranking":
        return ("completed", "local_extension_or_adapted_paper_dataset", "not_applicable", "raw_and_recomputed", "F6D4A adapted local LastFM LLM ranking evidence.")
    if dataset == "LastFM" and model in {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}:
        return ("completed", "adapted_local_protocol", "repository_derived_reimplementation", "raw_and_recomputed", "F6D2 adapted multi-seed trainable model evidence.")
    if model in {"Qwen2", "Mistral"} and dataset in {"ML-1M", "Amazon Beauty", "Amazon Sports"}:
        return ("completed", "adapted_local_protocol", "repository_derived_reimplementation", "raw_and_recomputed", "Local indexed/open-weight model substitute, not exact paper model/prompt.")
    if model in {"MostPop", "BM25", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"}:
        return ("completed" if dataset != "LastFM" or task == "ranking" else "partial", "adapted_local_protocol", "functionally_equivalent_or_reimplementation", "deterministic_derived_evidence", "Traditional baseline evidence is local/reconstructed unless exact official run is proven.")
    return ("not_executed", "unresolved", "unresolved", "missing_evidence", "")


def local_result_source(dataset: str, task: str, model: str) -> str:
    if dataset == "LastFM" and model in {"Qwen2", "Mistral"}:
        return "results/tables/finalization_f6d4a_utility_metrics.csv"
    if dataset == "LastFM" and model in {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}:
        return "results/tables/finalization_f6d2_aggregate_utility_metrics.csv"
    if model in {"Qwen2", "Mistral"}:
        return "results/tables/canonical_local_llm_utility_metrics.csv"
    return "results/tables/traditional_baselines_metrics.csv"


def lineage_rows(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in inventory:
        if row["category"] not in {"result_table", "manifest", "prediction", "raw_llm_output", "local_data", "report"}:
            continue
        path = row["path"]
        rows.append(
            {
                "artifact_path": path,
                "artifact_hash": row["sha256"],
                "producer": row["producer"],
                "source_inputs": row["inputs"],
                "protocol_id": protocol_id(path),
                "cohort_id": cohort_id(path),
                "candidate_contract": candidate_contract(path),
                "prompt_contract": prompt_contract(path),
                "metric_contract": metric_contract(path),
                "status": row["currentness"],
                "supersedes": supersedes(path),
                "superseded_by": superseded_by(path),
                "canonical_for": canonical_for(path),
                "notes": row["notes"],
            }
        )
    return rows


def protocol_id(path: str) -> str:
    lower = path.lower()
    if "top10" in lower or "f6e3b" in lower or "f6e4" in lower:
        return "repository_derived_top10_adapter"
    if "f6g1" in lower or "f6g2" in lower:
        return "lastfm_reranking_contract_resolution_or_superseded_top20"
    if "f6d2" in lower:
        return "lastfm_adapted_trainable_models"
    if "f6d4a" in lower:
        return "lastfm_adapted_llm_ranking"
    return phase_family(path)


def cohort_id(path: str) -> str:
    lower = path.lower()
    if "lastfm" in lower:
        return "lastfm_f6c_or_adapted"
    if "ml1m" in lower:
        return "ml1m_local"
    if "beauty" in lower:
        return "beauty_local"
    if "sports" in lower:
        return "sports_local"
    return ""


def candidate_contract(path: str) -> str:
    lower = path.lower()
    if "candidate" in lower or "ranking" in lower or "reranking" in lower:
        return "candidate_pool_20_when_available; top10 output if LLM adapter"
    return ""


def prompt_contract(path: str) -> str:
    lower = path.lower()
    if "prompt" in lower:
        return "prompt_contract_evidence"
    if "llm" in lower:
        return "local_llm_prompt_or_parser_contract"
    return ""


def metric_contract(path: str) -> str:
    lower = path.lower()
    if "metric" in lower or "utility" in lower:
        return "top5_metric_contract_or_beyond_utility_variant"
    return ""


def supersedes(path: str) -> str:
    if "phase_f6g2_output_length_contract_resolution" in path:
        return "phase_f6g1_top20_request_contract; phase_f6g2r_blocked_by_contract_conflict"
    if "finalization_f6b" in path:
        return "plan_d_phase_5e coverage taxonomy"
    return ""


def superseded_by(path: str) -> str:
    lower = path.lower()
    if "phase_f6g1" in lower or ("phase_f6g2r" in lower and "runtime_recovery" in lower):
        return "report/phase_f6g2_output_length_contract_resolution.md"
    if "plan_d_phase_5e" in lower:
        return "report/finalization_f6b_atomic_scope_registry.md"
    return ""


def canonical_for(path: str) -> str:
    lower = path.lower()
    if "phase_f6g2_output_length_contract_resolution" in lower:
        return "LLM output length contract"
    if "python_environment_recovery" in lower:
        return "current audit runtime provenance"
    if "workspace_forensic" in lower:
        return "current forensic audit"
    if "f6d4a" in lower:
        return "adapted LastFM LLM ranking"
    if "f6d2" in lower:
        return "adapted LastFM trainable models"
    return ""


def numeric_reconciliation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    local_mapping = read_csv_rows(TABLES_DIR / "finalization_f5af_local_atomic_mapping.csv")
    if local_mapping:
        for idx, row in enumerate(local_mapping[:200], start=1):
            paper_val = row.get("paper_value_numeric", row.get("paper_value", ""))
            local_val = row.get("local_value_numeric", row.get("local_value", ""))
            diff = numeric_diff(paper_val, local_val)
            rows.append(
                {
                    "result_id": row.get("atomic_cell_id", f"LOCAL-MAP-{idx:03d}"),
                    "source_report": row.get("source_report", "results/tables/finalization_f5af_local_atomic_mapping.csv"),
                    "source_table": row.get("paper_table_number", ""),
                    "raw_source": row.get("local_result_source", ""),
                    "metric": row.get("metric", ""),
                    "reported_value": paper_val,
                    "recomputed_value": local_val,
                    "absolute_difference": diff,
                    "relative_difference": "",
                    "cohort_match": row.get("cohort_match", "unresolved"),
                    "protocol_match": row.get("protocol_match", "unresolved"),
                    "verification_status": row.get("verification_status", "mapped_from_existing_local_atomic_mapping"),
                    "notes": row.get("notes", ""),
                }
            )
    else:
        atomic = read_csv_rows(TABLES_DIR / "finalization_f5af_atomic_paper_result_cells.csv")
        for idx, row in enumerate(atomic[:187], start=1):
            rows.append(
                {
                    "result_id": row.get("atomic_cell_id", f"PAPER-CELL-{idx:03d}"),
                    "source_report": "results/tables/finalization_f5af_atomic_paper_result_cells.csv",
                    "source_table": row.get("paper_table_number", ""),
                    "raw_source": "paper structured extraction",
                    "metric": row.get("metric", ""),
                    "reported_value": row.get("paper_value_numeric", ""),
                    "recomputed_value": "",
                    "absolute_difference": "",
                    "relative_difference": "",
                    "cohort_match": "not_recomputed",
                    "protocol_match": "not_recomputed",
                    "verification_status": "paper_cell_extracted_not_locally_recomputed_in_current_audit",
                    "notes": "Current audit did not rerun metrics; paper PDF verification remains as recorded.",
                }
            )
    return rows


def numeric_diff(a: str, b: str) -> str:
    try:
        return f"{abs(float(a) - float(b)):.17g}"
    except Exception:
        return ""


def blockers_rows() -> list[dict[str, Any]]:
    return [
        {
            "priority": "critical",
            "component": "paper_exact_runtime",
            "current_status": "blocked_for_exact_original_runtime_claim",
            "blocking_condition": "Original Microsoft Store base interpreter was not recovered; current runtime is repaired clone.",
            "blocker_evidence": "report/python_environment_recovery_audit.md; results/manifests/python_environment_recovery_snapshot.json",
            "paid_dependency": "no",
            "technical_feasibility": "repaired_clone_usable_for_audit",
            "scientific_importance": "critical",
            "required_inputs": "original base interpreter provenance or explicit repaired-runtime wording",
            "estimated_compute": "none",
            "safe_next_action": "Use approved recovered interpreter but do not claim exact original runtime reproduction.",
            "dependency": "none",
            "claim_if_completed": "runtime provenance is defensible for audit execution",
            "notes": "User approved recovered interpreter for this audit.",
        },
        {
            "priority": "critical",
            "component": "LastFM LLM reranking",
            "current_status": "blocked_pending_corrective_top10_freeze",
            "blocking_condition": "F6G1/F6G2 top20 request manifests are superseded by repository-derived top10 contract.",
            "blocker_evidence": "report/phase_f6g2_output_length_contract_resolution.md; results/tables/phase_f6g2_output_length_contract_resolution.csv",
            "paid_dependency": "no",
            "technical_feasibility": "offline_freeze_needed_before_inference",
            "scientific_importance": "high",
            "required_inputs": "new top10 request/shard manifests preserving F6G1 evidence",
            "estimated_compute": "low for freeze; inference still forbidden in this audit",
            "safe_next_action": "Build corrective offline top10 LastFM reranking freeze in a future phase.",
            "dependency": "output-length contract resolution",
            "claim_if_completed": "LastFM reranking can become executable under repository-derived adapted contract",
            "notes": "No inference was run.",
        },
        {
            "priority": "high",
            "component": "paper PDF primary extraction",
            "current_status": "partial",
            "blocking_condition": "Current audit relies on existing structured extraction for many paper cells; PDF visual verification is incomplete.",
            "blocker_evidence": "results/tables/finalization_f5af_atomic_paper_result_cells.csv",
            "paid_dependency": "no",
            "technical_feasibility": "manual_or_tooled_pdf_review",
            "scientific_importance": "high",
            "required_inputs": "PDF parser or manual table verification",
            "estimated_compute": "low",
            "safe_next_action": "Verify F5AF paper numeric cells directly against local PDF.",
            "dependency": "paper/2411.00331v1.pdf",
            "claim_if_completed": "paper numeric-cell evidence can be marked PDF-verified",
            "notes": "Do not fabricate PDF verification.",
        },
        {
            "priority": "high",
            "component": "exact paper LLM model versions",
            "current_status": "blocked_or_unavailable",
            "blocking_condition": "Local open-weight model substitutes are not exact paper endpoints/versions.",
            "blocker_evidence": "canonical/local LLM reports and model inventories",
            "paid_dependency": "possibly",
            "technical_feasibility": "depends_on_retired_or_paid_endpoints",
            "scientific_importance": "high",
            "required_inputs": "exact model endpoint/version access and paper prompt contract",
            "estimated_compute": "unknown",
            "safe_next_action": "Keep local LLM evidence classified as adapted.",
            "dependency": "model availability",
            "claim_if_completed": "selected exact model cells could be rerun if access exists",
            "notes": "No API calls were made.",
        },
    ]


def progress_measures(status_rows: list[dict[str, Any]], paper_rows: list[dict[str, Any]], numeric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    exact_components = [r for r in status_rows if r["paper_alignment"] == "exact_paper_replication"]
    adapted = [r for r in status_rows if "adapted" in r["paper_alignment"] or "local_extension" in r["paper_alignment"]]
    executed = [r for r in status_rows if r["execution_status"] == "completed"]
    numeric_exact = [r for r in numeric_rows if r["verification_status"] in {"protocol_exact_verified", "exact_match"}]
    return {
        "exact_paper_component_coverage": {"numerator": len(exact_components), "denominator": len(status_rows), "percentage": pct(len(exact_components), len(status_rows)), "rule": "status rows classified exact_paper_replication"},
        "adapted_or_reconstructed_component_coverage": {"numerator": len(adapted), "denominator": len(status_rows), "percentage": pct(len(adapted), len(status_rows)), "rule": "status rows classified adapted or local extension"},
        "operational_executable_coverage": {"numerator": len(executed), "denominator": len(status_rows), "percentage": pct(len(executed), len(status_rows)), "rule": "status rows completed"},
        "paper_numeric_cell_exact_coverage": {"numerator": len(numeric_exact), "denominator": len([r for r in paper_rows if r['dimension'] == 'numeric_result_cell']), "percentage": pct(len(numeric_exact), len([r for r in paper_rows if r['dimension'] == 'numeric_result_cell'])), "rule": "numeric cells protocol-exact verified in current evidence"},
        "paper_contract_rows_recovered": {"numerator": len(paper_rows), "denominator": len(paper_rows), "percentage": "100.000000", "rule": "rows represented in paper_atomic_contract.csv; not all PDF-verified"},
    }


def pct(num: int, den: int) -> str:
    return "" if den == 0 else f"{(num / den) * 100:.6f}"


def environment_snapshot() -> dict[str, Any]:
    recovery = json.loads(read_text(PROJECT_ROOT / "results" / "manifests" / "python_environment_recovery_snapshot.json"))
    return {
        "approved_runtime_interpreter": str(RECOVERED_INTERPRETER),
        "runtime_classification": recovery.get("environment_equivalence_verdict", ""),
        "not_exact_original_runtime": True,
        "python_version_check": sys.version,
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "python_no_user_site_required": True,
        "recovery_artifacts": [
            "report/python_environment_recovery_audit.md",
            "results/environment/python_environment_candidate_inventory.csv",
            "results/environment/broken_venv_static_package_inventory.csv",
            "results/environment/project_dependency_contract.csv",
            "results/environment/environment_recovery_comparison.csv",
            "results/environment/recovered_environment_package_inventory.csv",
            "results/environment/recovered_environment_validation.csv",
            "results/manifests/python_environment_recovery_snapshot.json",
        ],
    }


def build_reports(
    inventory: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    paper_rows: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    numeric_rows: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    progress: dict[str, dict[str, Any]],
    initial_git_status: str,
    final_git_status: str,
) -> tuple[str, str]:
    env = environment_snapshot()
    category_counts = Counter(row["category"] for row in inventory)
    current_reports = [r for r in inventory if r["currentness"] == "current_contract_resolution"]
    stale_claims = [r for r in claims if r["verdict"] in {"unsupported", "stale_superseded", "verified_historical_only"}]
    paper_numeric_count = len([r for r in paper_rows if r["dimension"] == "numeric_result_cell"])
    exact_numeric = progress["paper_numeric_cell_exact_coverage"]
    adapted = progress["adapted_or_reconstructed_component_coverage"]
    exact = progress["exact_paper_component_coverage"]
    operational = progress["operational_executable_coverage"]
    test_results = read_csv_rows(TABLES_DIR / "workspace_forensic_audit_test_results.csv")
    determinism_results = read_csv_rows(TABLES_DIR / "workspace_forensic_audit_determinism.csv")
    test_summary = test_results[0] if test_results else {}
    protected_hash_summary = next((row for row in test_results if row.get("test_scope") == "protected_input_hash_invariance"), {})
    determinism_summary = determinism_results[0] if determinism_results else {}
    initial_lines = git_status_lines(initial_git_status)
    final_lines = git_status_lines(final_git_status)
    initial_counts = git_status_counts(initial_lines)
    final_counts = git_status_counts(final_lines)
    resume_git = audit_resume_git_baseline(final_git_status)
    resume_counts = resume_git["status_counts"]

    report = f"""# Workspace Forensic State Audit

## 1. Executive summary

Decision: `forensic_audit_completed_with_conservative_classification`.

This audit restarted from the beginning after explicit approval to use the recovered interpreter. It recomputed inventories, phase/supersession records, paper-to-local traceability, claim checks, status matrices, lineage, numeric reconciliation placeholders, blockers, and a context snapshot from current workspace evidence.

The maximum defensible statement is conservative: the project contains substantial adapted/reconstructed local evidence, but this audit found `0` exact paper-replication status rows under the strict gate because exact model versions, exact prompts, exact runtime, PDF verification, and several protocol factors remain unresolved or adapted.

## 2. Audit scope and limitations

- No training, LLM inference, Ollama invocation, API calls, package installs, downloads, staging, or commits were performed.
- The original `.venv` was not modified.
- Current audit runtime: `{env['approved_runtime_interpreter']}`.
- Runtime classification: `{env['runtime_classification']}`.
- Runtime caveat: this is not exact original executable-environment reproduction.
- Paper PDF extraction remains partial; many paper rows derive from existing structured artifacts and are marked accordingly.

## 3. Git and environment state

- Branch: `{run_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).stdout}`
- HEAD: `{run_command(['git', 'rev-parse', 'HEAD']).stdout}`
- Audit-resume baseline status rows: `{resume_git['status_row_count']}` reconstructed from final status minus `{resume_git['audit_created_status_row_count']}` new complete-audit paths.
- Audit-resume baseline counts: modified `{resume_counts['modified']}`, added `{resume_counts['added']}`, deleted `{resume_counts['deleted']}`, untracked `{resume_counts['untracked']}`, other `{resume_counts['other']}`.
- Initial status rows captured before this builder execution: `{len(initial_lines)}`.
- Initial-before-builder counts: modified `{initial_counts['modified']}`, added `{initial_counts['added']}`, deleted `{initial_counts['deleted']}`, untracked `{initial_counts['untracked']}`, other `{initial_counts['other']}`.
- Final status rows captured by builder: `{len(final_lines)}`.
- Final counts: modified `{final_counts['modified']}`, added `{final_counts['added']}`, deleted `{final_counts['deleted']}`, untracked `{final_counts['untracked']}`, other `{final_counts['other']}`.
- Python executable: `{sys.executable}`
- Python version: `{sys.version.splitlines()[0]}`
- Package-tree recovery evidence: `results/manifests/python_environment_recovery_snapshot.json`

## 4. Repository architecture

Relevant file inventory rows: `{len(inventory)}`.

Category counts:

{markdown_counts(category_counts)}

## 5. Source-authority hierarchy

The audit used this hierarchy: raw inputs and raw outputs; executable source/configs; frozen manifests and hashes; validation outputs; derived CSV/JSON; reports; README and notes. Narrative claims were not trusted without cross-reference.

## 6. Relevant file inventory

Full inventory: `results/tables/workspace_relevant_file_inventory.csv`.

The inventory excludes `.git`, virtualenv package trees, Python runtime package trees, cache directories, and pytest temporary logs, while retaining scientific data, reports, manifests, tables, source, tests, raw LLM outputs, and checkpoints.

## 7. Phase and supersession reconstruction

Phase registry rows: `{len(phase_rows)}` in `results/tables/workspace_phase_and_supersession_registry.csv`.

Key supersession: F6G1/F6G2/F6G2R top-20 LastFM reranking execution artifacts are not current executable contracts after `phase_f6g2_output_length_contract_resolution`; the current executable rule is repository-derived top-10 output over 20 candidates with top-5 metrics.

## 8. Paper contract

Paper atomic contract rows: `{len(paper_rows)}`. Numeric paper cells represented: `{paper_numeric_count}`.

Rows sourced from F5AF remain marked `secondary_structured_source_pdf_unverified` where the prior extraction did not directly verify PDF pages. The paper contract is separated from official repository semantics.

## 9. Official repository contract

The official repository snapshot under `original_repo/EvaLLMasRecommender` remains protected. Normal nested Git commands are blocked by dubious ownership unless Git config is changed, so this audit uses source-tree evidence and direct `.git` metadata only where available.

Recovered official executable semantics from prior and current evidence include candidate pool size 20, parser/evaluator top-10 behavior for executable LLM outputs, and top-5 metrics. LLMRank template wording remains a repository-internal conflict because template text can ask for all 20 while parser/evaluator behavior consumes top 10.

## 10. Paper-versus-repository conflicts

Conflict rows: `{len(conflicts)}` in `results/tables/paper_repository_contract_conflicts.csv`.

Material conflicts include paper-unspecified output length versus repository-derived top-10 executable behavior, runtime provenance differences, and incomplete direct PDF verification.

## 11. Actual local pipeline

Pipeline evidence is reconstructed through data files, configs, source, manifests, raw outputs, parser/result tables, and reports. The local pipeline is adapted/reconstructed, not exact paper execution. LastFM is a paper dataset locally reconstructed/adapted, not a fifth-dataset extension.

## 12. Dataset and cohort audit

Dataset/task/model status is in `results/tables/current_reproduction_status_matrix.csv`. LastFM ranking has adapted local evidence; LastFM reranking LLM execution remains blocked until a corrective top-10 offline freeze exists.

## 13. Candidate-set audit

Candidate evidence is represented in inventory and lineage tables. The current candidate/output contract is: 20-item candidate pools where available; repository-derived 10-item LLM returned list; metric cutoff `K=5`. Positive inclusion and candidate-order invariants are not recomputed by costly experiment reruns in this audit.

## 14. Model audit

The status matrix separates open-weight local substitutes (`Qwen2`, `Mistral`) from exact paper LLM endpoints. Exact paper LLM-version coverage is not claimed. Traditional baselines are local or repository-derived reimplementations unless exact official outputs are traceably matched.

## 15. Prompt and parser audit

Prompt/parser contracts are traced through canonical LLM tables, F6E prompt recovery artifacts, F6E3B/F6E4 adapted indexed prompts, and F6G2 output-length resolution. The local JSON/index prompt is classified as adapted and must not be called exact paper prompting.

## 16. Output-length contract audit

Current resolution: `D_repository_derived_top10_paper_unspecified`. F6G1/F6G2 full-20 request manifests are preserved but superseded for execution.

## 17. Metric audit

Metric evidence is traced through `canonical_metric_*`, F5B-F5E, F6F1, and local utility tables. Unsupported or unresolved paper-exact metrics remain blocked or adapted rather than zero-filled.

## 18. Experimental evidence audit

Evidence exists for adapted local LLM and baseline runs, especially canonical local LLM, F6D2 LastFM trainable models, and F6D4A LastFM LLM ranking. No new experimental evidence was generated by this audit.

## 19. Numerical reconciliation

Numeric reconciliation rows: `{len(numeric_rows)}` in `results/tables/numeric_result_reconciliation.csv`.

Exact paper numeric-cell coverage is `{exact_numeric['numerator']}/{exact_numeric['denominator']} = {exact_numeric['percentage']}%` under the current strict gate. Existing local mappings are preserved as evidence but not upgraded to exact paper reproduction.

## 20. Claim-verification findings

Claim verification rows: `{len(claims)}` in `results/tables/project_claim_verification.csv`.

High-risk claims include any wording that implies full reproduction, exact numerical reproduction, exact prompt reproduction, or 100% paper coverage. Stale historical blocker records are preserved but superseded where later evidence resolves them.

## 21. Exact reproduction status

Exact paper-component coverage: `{exact['numerator']}/{exact['denominator']} = {exact['percentage']}%`.

This is intentionally strict. Completed adapted work does not count as exact paper replication.

## 22. Reconstructed/adapted status

Adapted/reconstructed component coverage: `{adapted['numerator']}/{adapted['denominator']} = {adapted['percentage']}%`.

## 23. Local extension status

The university-extension contract is partially supported by added metrics and local adapted analyses. A true fifth dataset beyond the paper is not verified because LastFM is a paper dataset.

## 24. University-requirement status

The project has documented code, saved logs/results, paper-vs-local comparison artifacts, limitations, and report-ready evidence. Extension claims must distinguish added metrics/local analyses from exact paper reproduction.

## 25. Supported scientific claims

- The project is not a complete exact numerical replication of the paper.
- The current executable LLM output-length contract is repository-derived top-10, not paper-explicit.
- Local Qwen2/Mistral evidence is adapted open-weight/local evidence, not exact paper model execution.
- The recovered runtime is scientifically defensible for audit execution but not exact original runtime reproduction.

## 26. Unsupported or overstated claims

- 100% paper reproduction.
- Exact prompt reproduction for local JSON/index prompts.
- Exact model reproduction for local substitutes.
- Full LastFM reranking LLM execution after F6G2R without a corrective top-10 freeze.
- Paper-exact fairness/diversity/serendipity where formula inputs and denominators remain unresolved.

## 27. Stale and superseded reports

Reports with `final`, `canonical`, or earlier blocker labels are not automatically current. The lineage table records superseded artifacts, especially Plan D/F6B coverage taxonomy and F6G1/F6G2R top-20 reranking artifacts.

## 28. Genuine blockers

Blockers are listed in `results/tables/blockers_gaps_and_next_actions.csv`. The top blocker is exact paper-runtime provenance; the top executable next work package is a corrective LastFM reranking top-10 freeze, not inference.

## 29. Unresolved ambiguities

- Direct PDF verification of all paper cells.
- Exact paper LLM versions and endpoint availability.
- Full official repository provenance for nested repo cleanliness without changing Git safe-directory config.
- Some denominator/formula details for appendix metrics.

## 30. Multiple progress percentages

{markdown_progress(progress)}

## 31. Maximum defensible completion statement

The project has a substantial adapted/reconstructed reproducibility package, but exact paper replication remains `0%` under the strict atomic matrix used here. Operational/adapted evidence is meaningfully broader than exact evidence and must be reported separately.

## 32. Threats to validity

This audit uses available local artifacts and does not rerun costly experiments. Paper PDF extraction is incomplete. Git state is dirty and many artifacts are untracked. Runtime is a repaired package-tree clone rather than the original Store base.

## 33. Recommended single next work package

Create a corrective offline LastFM reranking top-10 freeze that preserves F6G1 evidence but regenerates request/shard manifests with `candidate_pool_size=20`, `requested_output_length=10`, `parser_expected_length=10`, and `metric_cutoff_K=5`. Do not run inference in that work package.

## 34. Files created

Current audit outputs are the report, handoff, 10 CSV tables, one JSON snapshot, builder, and focused tests.

## 35. Commands executed

Commands are summarized in the final response and in the JSON snapshot. Python commands used only `{env['approved_runtime_interpreter']}` with `PYTHONNOUSERSITE=1`.

## 36. Test results

Focused audit tests: `{test_summary.get('status', 'not_recorded')}`.

- Command: `{test_summary.get('command', '')}`
- Passed: `{test_summary.get('passed', '')}`
- Failed: `{test_summary.get('failed', '')}`
- Warnings: `{test_summary.get('warnings', '')}`
- Duration seconds: `{test_summary.get('duration_seconds', '')}`
- Excluded tests: `{test_summary.get('excluded_tests', 'full_project_suite')}`
- Exclusion reason: `{test_summary.get('reason', 'Full suite is outside this audit and may exercise training/checkpoint/writer paths.')}`

## 37. Determinism results

Determinism status: `{determinism_summary.get('status', 'not_recorded')}`.

- Command: `{determinism_summary.get('command', '')}`
- Checked artifacts: `{determinism_summary.get('checked_artifacts', '')}`
- Mismatches: `{determinism_summary.get('mismatches', '')}`
- Notes: `{determinism_summary.get('notes', '')}`

## 38. Protected-input verification

Protected inputs were not intentionally modified. Current audit artifacts are isolated. Existing recovery artifacts are runtime provenance, not scientific experimental outputs.

- Protected hash-invariance status: `{protected_hash_summary.get('status', 'not_recorded')}`
- Protected/manifested rows compared: `{protected_hash_summary.get('passed', '')}`
- Protected hash check command: `{protected_hash_summary.get('command', '')}`

## 39. Initial and final Git status

The audit-resume baseline was rechecked before implementation for branch and HEAD. The exact live raw status listing was not frozen before artifact generation; this report therefore records a deterministic reconstructed baseline from final status by removing only the new complete-audit paths. Initial and final status are dirty. No staging or commit was performed.

Audit-created status entries:

{chr(10).join(f"- `{path}`" for path in resume_git['audit_created_paths'])}
"""

    handoff = f"""# Future Operations Context Handoff

## Verified Current Context

- Root: `{PROJECT_ROOT}`
- Branch: `{run_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).stdout}`
- HEAD: `{run_command(['git', 'rev-parse', 'HEAD']).stdout}`
- Current audit runtime: `{env['approved_runtime_interpreter']}`
- Runtime classification: `{env['runtime_classification']}`
- Runtime caveat: repaired clone exact package tree, not exact original Store-base environment.
- Final Git status rows: `{len(final_lines)}`; reconstructed audit-resume baseline rows: `{resume_git['status_row_count']}`.

## Source Hierarchy

Use raw inputs/outputs first, then executable source/config, frozen manifests/hashes, validation outputs, derived CSV/JSON, reports, then README/notes.

## Canonical Current Artifacts

- Runtime provenance: `report/python_environment_recovery_audit.md`
- Output-length contract: `report/phase_f6g2_output_length_contract_resolution.md`
- Paper atomic cells: `results/tables/finalization_f5af_atomic_paper_result_cells.csv`
- Current forensic audit tables: `results/tables/workspace_*`, `paper_*`, `project_claim_verification.csv`, `current_reproduction_status_matrix.csv`, `artifact_lineage_and_supersession.csv`, `numeric_result_reconciliation.csv`, `blockers_gaps_and_next_actions.csv`
- Current context snapshot: `results/manifests/current_project_context_snapshot.json`

## Coverage Snapshot

- Exact paper-component coverage: `{exact['numerator']}/{exact['denominator']} = {exact['percentage']}%`
- Adapted/reconstructed coverage: `{adapted['numerator']}/{adapted['denominator']} = {adapted['percentage']}%`
- Operational executable coverage: `{operational['numerator']}/{operational['denominator']} = {operational['percentage']}%`
- Exact paper numeric-cell coverage: `{exact_numeric['numerator']}/{exact_numeric['denominator']} = {exact_numeric['percentage']}%`

## Protected Paths

Keep `original_repo/`, `paper/`, `data/raw/`, existing candidate sets, cohorts, raw LLM outputs, checkpoints, prompt/parser/metric contracts, and prior phase outputs read-only unless a future task explicitly authorizes otherwise.

## Superseded Or Risky Artifacts

- F6G1/F6G2 top-20 LastFM reranking request manifests are superseded for execution.
- F6G2R contract-blocked report is historical after output-length resolution.
- Any `final` or `canonical` file must still be checked against lineage before use.

## Allowed Claims

- The project contains adapted/reconstructed local evidence.
- The current LLM output-length contract is repository-derived top-10 and paper-unspecified.
- The recovered audit runtime is defensible but not exact original runtime.

## Forbidden Claims

- 100% exact reproduction.
- Exact prompt/model/runtime reproduction for local substitutes.
- Complete LastFM LLM reranking execution.
- Paper-exact unsupported metrics with unresolved denominators.

## Recommended Next Work Package

Build a corrective offline LastFM reranking top-10 freeze. It should not run inference.
"""
    return report, handoff


def markdown_counts(counter: Counter[str]) -> str:
    return "\n".join(f"- `{key}`: `{value}`" for key, value in sorted(counter.items()))


def markdown_progress(progress: dict[str, dict[str, Any]]) -> str:
    lines = []
    for name, item in sorted(progress.items()):
        lines.append(f"- `{name}`: `{item['numerator']}/{item['denominator']}` = `{item['percentage']}%`; rule: {item['rule']}")
    return "\n".join(lines)


def build_snapshot(
    inventory: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    paper_rows: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    numeric_rows: list[dict[str, Any]],
    blockers: list[dict[str, Any]],
    progress: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    git_branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout
    git_head = run_command(["git", "rev-parse", "HEAD"]).stdout
    git_status = run_command(["git", "status", "--porcelain=v1"]).stdout.splitlines()
    git_status_text = "\n".join(git_status)
    return {
        "snapshot_status": "complete_forensic_audit_conservative",
        "project_root": str(PROJECT_ROOT),
        "branch": git_branch,
        "head": git_head,
        "git_status_row_count": len(git_status),
        "git_state": {
            "branch": git_branch,
            "head": git_head,
            "current_status_row_count": len(git_status),
            "current_status_counts": git_status_counts(git_status),
            "audit_resume_baseline": audit_resume_git_baseline(git_status_text),
        },
        "environment": environment_snapshot(),
        "authoritative_artifacts": {
            "forensic_report": rel(REPORT_PATH),
            "handoff": rel(HANDOFF_PATH),
            "runtime_recovery_report": "report/python_environment_recovery_audit.md",
            "output_length_resolution": "report/phase_f6g2_output_length_contract_resolution.md",
        },
        "protected_inputs": ["original_repo/", "paper/", "data/raw/", "results/llm/", "results/checkpoints/"],
        "counts": {
            "inventory_rows": len(inventory),
            "phase_rows": len(phase_rows),
            "paper_contract_rows": len(paper_rows),
            "conflict_rows": len(conflicts),
            "claim_rows": len(claims),
            "status_rows": len(status_rows),
            "numeric_reconciliation_rows": len(numeric_rows),
            "blocker_rows": len(blockers),
        },
        "progress": progress,
        "supported_claims": [
            "substantial adapted/reconstructed local evidence exists",
            "exact paper replication is not established",
            "repository-derived top10 output length is current executable contract",
        ],
        "forbidden_claims": [
            "100% exact reproduction",
            "exact original runtime reproduction",
            "exact local JSON/index prompt reproduction",
            "complete LastFM LLM reranking execution",
        ],
        "unresolved_conflicts": [row["conflict_id"] for row in conflicts if row["current_resolution"] not in {"D_repository_derived_top10_paper_unspecified", "defensible_repaired_clone_exact_package_versions"}],
        "blockers": [row["component"] for row in blockers],
        "recommended_next_work_package": "Corrective offline LastFM reranking top10 freeze; no inference.",
        "no_training": True,
        "no_llm_inference": True,
        "no_api_calls": True,
        "no_downloads": True,
        "no_git_stage_or_commit": True,
    }


def output_hashes() -> dict[str, str]:
    paths = [
        INVENTORY_CSV,
        PHASE_CSV,
        PAPER_CONTRACT_CSV,
        CONFLICTS_CSV,
        TRACEABILITY_CSV,
        CLAIMS_CSV,
        STATUS_MATRIX_CSV,
        LINEAGE_CSV,
        NUMERIC_RECON_CSV,
        BLOCKERS_CSV,
        SNAPSHOT_JSON,
        REPORT_PATH,
        HANDOFF_PATH,
    ]
    return {rel(path): file_sha256(path) for path in paths if path.exists()}


def build_all() -> dict[str, Any]:
    if Path(sys.executable).resolve() != RECOVERED_INTERPRETER.resolve():
        raise RuntimeError(f"forensic audit must run with recovered interpreter: {RECOVERED_INTERPRETER}; got {sys.executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 is required")

    initial_git_status = run_command(["git", "status", "--porcelain=v1"]).stdout
    inventory = build_inventory()
    phase_rows = build_phase_registry(inventory)
    paper_rows = paper_contract_rows()
    conflicts = conflict_rows()
    traceability = traceability_rows(paper_rows)
    claims = claim_rows()
    status_rows = status_matrix_rows()
    lineage = lineage_rows(inventory)
    numeric_rows = numeric_reconciliation_rows()
    blockers = blockers_rows()
    progress = progress_measures(status_rows, paper_rows, numeric_rows)

    write_csv(
        INVENTORY_CSV,
        inventory,
        [
            "path",
            "file_type",
            "category",
            "phase_family",
            "size_bytes",
            "sha256",
            "git_state",
            "tracked_status",
            "producer",
            "inputs",
            "consumers",
            "evidence_level",
            "currentness",
            "protected_status",
            "inspection_status",
            "notes",
        ],
    )
    write_csv(
        PHASE_CSV,
        phase_rows,
        [
            "phase_id",
            "objective",
            "predecessor",
            "successor",
            "source_files",
            "config_files",
            "test_files",
            "output_files",
            "report_files",
            "decision_token",
            "evidence_generated",
            "contract_changed",
            "current_status",
            "supersedes",
            "superseded_by",
            "notes",
        ],
    )
    write_csv(
        PAPER_CONTRACT_CSV,
        paper_rows,
        [
            "contract_id",
            "paper_section",
            "paper_page",
            "paper_table",
            "paper_figure",
            "paper_equation",
            "dimension",
            "task",
            "dataset",
            "model",
            "prompt_strategy",
            "cohort",
            "history_policy",
            "candidate_policy",
            "candidate_count",
            "requested_output_length",
            "metric",
            "cutoff",
            "denominator",
            "repetition_policy",
            "statistical_method",
            "paper_value",
            "paper_specification_status",
            "notes",
        ],
    )
    write_csv(CONFLICTS_CSV, conflicts, ["conflict_id", "topic", "paper_evidence", "repository_evidence", "paper_value", "repository_value", "conflict_type", "scientific_consequence", "resolution_policy", "current_resolution", "notes"])
    write_csv(TRACEABILITY_CSV, traceability, ["contract_id", "local_source", "local_config", "local_input", "local_raw_output", "local_derived_output", "local_test", "local_report", "execution_status", "paper_alignment", "repository_alignment", "evidence_quality", "currentness", "material_divergences", "notes"])
    write_csv(CLAIMS_CSV, claims, ["claim_id", "claim_text", "source_file", "source_location", "source_phase", "claimed_status", "claimed_evidence", "primary_evidence", "verification_method", "verdict", "confidence", "currentness", "contradiction", "corrective_wording"])
    write_csv(STATUS_MATRIX_CSV, status_rows, ["dataset", "task", "model", "prompt_strategy", "metric", "experimental_condition", "execution_status", "paper_alignment", "repository_alignment", "evidence_quality", "currentness", "local_result_source", "notes"])
    write_csv(LINEAGE_CSV, lineage, ["artifact_path", "artifact_hash", "producer", "source_inputs", "protocol_id", "cohort_id", "candidate_contract", "prompt_contract", "metric_contract", "status", "supersedes", "superseded_by", "canonical_for", "notes"])
    write_csv(NUMERIC_RECON_CSV, numeric_rows, ["result_id", "source_report", "source_table", "raw_source", "metric", "reported_value", "recomputed_value", "absolute_difference", "relative_difference", "cohort_match", "protocol_match", "verification_status", "notes"])
    write_csv(BLOCKERS_CSV, blockers, ["priority", "component", "current_status", "blocking_condition", "blocker_evidence", "paid_dependency", "technical_feasibility", "scientific_importance", "required_inputs", "estimated_compute", "safe_next_action", "dependency", "claim_if_completed", "notes"])

    final_git_status = run_command(["git", "status", "--porcelain=v1"]).stdout
    report, handoff = build_reports(inventory, phase_rows, paper_rows, conflicts, claims, status_rows, numeric_rows, blockers, progress, initial_git_status, final_git_status)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    HANDOFF_PATH.write_text(handoff, encoding="utf-8")

    snapshot = build_snapshot(inventory, phase_rows, paper_rows, conflicts, claims, status_rows, numeric_rows, blockers, progress)
    snapshot["output_hashes_pre_snapshot"] = {}
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_JSON.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    snapshot["output_hashes"] = output_hashes()
    SNAPSHOT_JSON.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()
    snapshot = build_all()
    if args.print_summary:
        print(json.dumps({"snapshot_status": snapshot["snapshot_status"], "counts": snapshot["counts"], "progress": snapshot["progress"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

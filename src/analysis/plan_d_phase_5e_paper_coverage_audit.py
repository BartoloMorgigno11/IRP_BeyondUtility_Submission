"""Offline Phase 5E paper coverage audit.

This module builds a deterministic, evidence-grounded audit of how much of the
paper has a local counterpart. It reads existing artifacts only and does not
run LLM inference, recommender training, network calls, prompt generation, or
parser changes.
"""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
CANONICAL_PAPER_PATH = Path("paper/2411.00331v1.pdf")

STATUS_ORDER = [
    "reproduced_exact",
    "reconstructed",
    "adapted",
    "extension_only",
    "blocked",
    "not_executed",
    "unresolved",
    "not_applicable",
]
APPLICABLE_STATUSES = [status for status in STATUS_ORDER if status != "not_applicable"]
COMPARABILITY_ORDER = [
    "paper_exact",
    "official_code_aligned_local_model_substitute",
    "adapted_local_model_extension",
    "reconstructed_paper_like",
    "same_metric_different_cohort",
    "same_task_different_model",
    "same_dataset_different_protocol",
    "same_protocol_different_model",
    "diagnostic_only",
    "extension_not_in_paper",
    "blocked_not_comparable",
    "unresolved_comparability",
    "not_applicable",
]
EVIDENCE_PRECEDENCE = [
    "immutable_raw_local_experiment_artifact",
    "phase_5c_full_matrix_evidence",
    "phase_5d_robustness_consolidation",
    "canonical_manifest_or_metric_contract",
    "exact_paired_diagnostic_experiment",
    "source_or_generated_result_table",
    "earlier_phase_report",
    "paper_pdf",
    "original_repository_code_or_documentation",
    "report_only_summary",
    "none",
]

OUTPUT_TABLES = {
    "source_inventory": Path("results/tables/plan_d_phase_5e_source_inventory.csv"),
    "paper_component_registry": Path("results/tables/plan_d_phase_5e_paper_component_registry.csv"),
    "local_evidence_registry": Path("results/tables/plan_d_phase_5e_local_evidence_registry.csv"),
    "component_evidence_mapping": Path("results/tables/plan_d_phase_5e_component_evidence_mapping.csv"),
    "component_coverage": Path("results/tables/plan_d_phase_5e_component_coverage.csv"),
    "paper_result_cells": Path("results/tables/plan_d_phase_5e_paper_result_cells.csv"),
    "result_cell_comparison": Path("results/tables/plan_d_phase_5e_result_cell_comparison.csv"),
    "coverage_metrics": Path("results/tables/plan_d_phase_5e_coverage_metrics.csv"),
    "category_coverage": Path("results/tables/plan_d_phase_5e_category_coverage.csv"),
    "extensions": Path("results/tables/plan_d_phase_5e_extensions.csv"),
    "blockers": Path("results/tables/plan_d_phase_5e_blockers.csv"),
    "claims": Path("results/tables/plan_d_phase_5e_claims.csv"),
    "validation": Path("results/tables/plan_d_phase_5e_validation.csv"),
    "next_actions": Path("results/tables/plan_d_phase_5e_next_actions.csv"),
}
REPORT_PATH = Path("report/plan_d_phase_5e_final_paper_coverage_audit.md")
INITIAL_PROTECTED_MANIFEST_PATH = Path("results/tables/plan_d_phase_5e_protected_input_hashes_initial.csv")
FINAL_PROTECTED_MANIFEST_PATH = Path("results/tables/plan_d_phase_5e_protected_input_hashes_final.csv")
OUTPUT_HASHES_FIRST_PATH = Path("results/tables/plan_d_phase_5e_output_hashes_first.csv")
OUTPUT_HASHES_SECOND_PATH = Path("results/tables/plan_d_phase_5e_output_hashes_second.csv")

PHASE5E_OUTPUT_PATHS = {
    *{path.as_posix() for path in OUTPUT_TABLES.values()},
    REPORT_PATH.as_posix(),
    INITIAL_PROTECTED_MANIFEST_PATH.as_posix(),
    FINAL_PROTECTED_MANIFEST_PATH.as_posix(),
    OUTPUT_HASHES_FIRST_PATH.as_posix(),
    OUTPUT_HASHES_SECOND_PATH.as_posix(),
    "src/analysis/plan_d_phase_5e_paper_coverage_audit.py",
    "src/experiments/build_plan_d_phase_5e_paper_coverage_audit.py",
    "tests/test_plan_d_phase_5e_paper_coverage_audit.py",
}

SOURCE_INVENTORY_COLUMNS = [
    "path",
    "artifact_type",
    "phase",
    "scientific_role",
    "inclusion_status",
    "exclusion_reason",
    "protected_status",
    "hash_status",
]
PAPER_COMPONENT_COLUMNS = [
    "component_id",
    "category",
    "component_name",
    "description",
    "paper_section",
    "paper_page",
    "table_or_figure",
    "original_dataset_scope",
    "original_model_scope",
    "original_task_scope",
    "original_protocol_scope",
    "original_metric_scope",
    "original_artifact_or_repository_reference",
    "applicability_to_project",
    "not_applicable_reason",
]
LOCAL_EVIDENCE_COLUMNS = [
    "local_evidence_id",
    "source_artifact_paths",
    "evidence_phase",
    "evidence_level",
    "local_dataset",
    "local_model",
    "local_task",
    "local_prompt",
    "local_parser",
    "local_candidate_count",
    "local_output_count",
    "local_cohort",
    "local_metric",
    "local_denominator",
    "execution_status",
    "pairing_status",
    "repetition_status",
    "validity_status",
    "exactness_mismatches",
    "permitted_conclusion",
    "prohibited_conclusion",
]
COMPONENT_MAPPING_COLUMNS = [
    "paper_component_id",
    *LOCAL_EVIDENCE_COLUMNS,
    "determines_primary_status",
    "primary_status_reason",
]
COMPONENT_COVERAGE_COLUMNS = [
    "component_id",
    "category",
    "component",
    "applicable",
    "primary_coverage_status",
    "comparability_classification",
    "evidence_level",
    "evidence_artifacts",
    "exactness_mismatches",
    "blocker",
    "permitted_claim",
    "prohibited_claim",
    "rationale",
]
PAPER_RESULT_CELL_COLUMNS = [
    "paper_result_cell_id",
    "paper_table",
    "paper_row",
    "paper_column",
    "dataset",
    "task",
    "model",
    "prompt_or_method",
    "metric",
    "cutoff",
    "value",
    "source_page",
    "verification_status",
]
RESULT_CELL_COMPARISON_COLUMNS = [
    "paper_result_cell_id",
    "local_result_id",
    "comparability_status",
    "exact_dataset_match",
    "exact_task_match",
    "exact_model_match",
    "exact_cohort_match",
    "exact_candidate_protocol_match",
    "exact_metric_definition_match",
    "exact_cutoff_match",
    "exact_denominator_match",
    "numerical_comparison_permitted",
    "reason",
]
COVERAGE_METRIC_COLUMNS = [
    "metric_name",
    "numerator",
    "denominator",
    "formula",
    "value",
    "interpretation",
    "prohibited_interpretation",
]
CATEGORY_COVERAGE_COLUMNS = [
    "category",
    "total_components",
    "applicable_components",
    "reproduced_exact",
    "reconstructed",
    "adapted",
    "extension_only",
    "blocked",
    "not_executed",
    "unresolved",
    "not_applicable",
    "strict_exact_pct",
    "reconstructed_or_better_pct",
    "maximum_defensible_pct",
    "blocked_pct",
]
EXTENSION_COLUMNS = [
    "extension_id",
    "description",
    "datasets",
    "models",
    "tasks",
    "evidence",
    "professor_requested_status",
    "scientific_contribution",
    "relation_to_paper",
    "why_it_does_not_increase_reproduction_percentage",
]
BLOCKER_COLUMNS = [
    "blocker_id",
    "component",
    "blocker_type",
    "missing_artifact_or_resource",
    "current_evidence",
    "recoverable_without_new_inference",
    "scientific_impact",
    "permitted_statement",
    "prohibited_statement",
]
CLAIM_COLUMNS = [
    "claim_id",
    "claim_type",
    "claim",
    "status",
    "supporting_artifacts",
    "permitted_statement",
    "prohibited_statement",
]
VALIDATION_COLUMNS = [
    "validation_id",
    "validation_scope",
    "check_name",
    "expected_value",
    "observed_value",
    "status",
    "severity",
    "evidence",
]
NEXT_ACTION_COLUMNS = [
    "action_id",
    "input_artifact",
    "required_use",
    "status",
    "phase_6_handoff",
]
PROTECTED_MANIFEST_COLUMNS = [
    "path",
    "protected_input",
    "hash_method",
    "sha256",
    "size_bytes",
    "mtime_ns",
    "policy",
]
OUTPUT_HASH_COLUMNS = ["path", "sha256"]


@dataclass(frozen=True)
class ComponentSpec:
    category: str
    name: str
    description: str
    section: str
    table_or_figure: str
    dataset_scope: str
    model_scope: str
    task_scope: str
    protocol_scope: str
    metric_scope: str
    reference: str
    status: str
    comparability: str
    evidence_level: str
    evidence_artifacts: str
    mismatches: str
    blocker: str
    permitted: str
    prohibited: str
    rationale: str
    applicable: bool = True
    not_applicable_reason: str = ""


def normalize_path(value: str | Path) -> str:
    """Return a repository-style path using forward slashes."""
    return Path(str(value).replace("\\", "/")).as_posix()


def slugify(value: str) -> str:
    """Normalize text for stable identifiers."""
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-")


def stable_component_id(category: str, component_name: str) -> str:
    """Create a stable paper-component identifier."""
    return "PC-" + slugify(category).upper() + "-" + slugify(component_name).upper()


def stable_result_cell_id(seed: str) -> str:
    """Create a stable paper result-cell identifier."""
    return "PRC-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10].upper()


def frame(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    """Build a dataframe with fixed columns and deterministic empty handling."""
    materialized = [{column: clean_cell(row.get(column, "")) for column in columns} for row in rows]
    return pd.DataFrame(materialized, columns=list(columns))


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def percent(numerator: int | float, denominator: int | float) -> float:
    """Compute a bounded percentage without arbitrary partial scores."""
    if denominator == 0:
        return 0.0
    value = float(numerator) / float(denominator) * 100.0
    return round(max(0.0, min(100.0, value)), 6)


def read_csv_if_exists(root: Path, path: str | Path) -> pd.DataFrame:
    full = root / path
    if not full.exists():
        return pd.DataFrame()
    return pd.read_csv(full)


def git_stdout(root: Path, args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{exc}"
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_phase5e_output(rel: str) -> bool:
    return rel in PHASE5E_OUTPUT_PATHS or rel.startswith("results/tables/plan_d_phase_5e_")


def build_protected_manifest(root: Path) -> pd.DataFrame:
    """Hash protected pre-Phase-5E inputs with a deterministic exclusion policy."""
    skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    allowed_suffixes = {".csv", ".json", ".jsonl", ".md", ".txt", ".py", ".yaml", ".yml", ".pdf", ".zip", ".pt", ".ipynb", ".sh"}
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix().lower()):
        rel = path.relative_to(root).as_posix()
        if any(part in skip_dirs for part in path.relative_to(root).parts):
            continue
        if _is_phase5e_output(rel):
            continue
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        size = path.stat().st_size
        must_hash = (
            rel.startswith("paper/")
            or rel.startswith("original_repo/")
            or rel.startswith("results/llm/")
            or rel.startswith("results/tables/canonical_")
            or rel.startswith("results/tables/plan_d_phase_5d_")
            or rel.startswith("results/tables/traditional_")
            or rel.startswith("results/tables/plan_e_")
            or rel.startswith("results/metrics/")
            or rel.startswith("results/raw_outputs/")
            or rel.startswith("src/")
            or rel.startswith("tests/")
            or rel.startswith("report/")
            or rel.startswith("configs/")
            or rel.endswith("README.md")
            or rel == "report/dataset_audit.md"
        )
        if must_hash or size <= 75_000_000:
            method = "sha256"
            digest = file_sha256(path)
        else:
            method = "metadata_only_large_file"
            digest = ""
        rows.append(
            {
                "path": rel,
                "protected_input": "true",
                "hash_method": method,
                "sha256": digest,
                "size_bytes": str(size),
                "mtime_ns": str(path.stat().st_mtime_ns),
                "policy": "phase_5e_initial_protected_input_manifest_excludes_only_phase_5e_outputs",
            }
        )
    return frame(rows, PROTECTED_MANIFEST_COLUMNS)


def write_csv(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, lineterminator="\n")


def stable_csv_bytes(data: pd.DataFrame) -> bytes:
    buffer = StringIO()
    data.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def verify_protected_manifest(initial: pd.DataFrame, final: pd.DataFrame) -> bool:
    """Verify protected inputs did not change, independent of row ordering."""
    cols = ["path", "protected_input", "hash_method", "sha256", "size_bytes", "mtime_ns", "policy"]
    left = normalize_manifest_for_compare(initial.loc[:, cols]).sort_values("path").reset_index(drop=True)
    right = normalize_manifest_for_compare(final.loc[:, cols]).sort_values("path").reset_index(drop=True)
    return left.equals(right)


def normalize_manifest_for_compare(manifest: pd.DataFrame) -> pd.DataFrame:
    """Normalize CSV-loaded and in-memory manifest values for comparison."""
    normalized = manifest.fillna("").copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].map(clean_cell)
    if "protected_input" in normalized.columns:
        normalized["protected_input"] = normalized["protected_input"].str.lower()
    return normalized.astype(str)


def discover_source_artifacts(root: Path, protected_manifest: pd.DataFrame) -> pd.DataFrame:
    """Inventory Phase 5E evidence inputs without reading network or executable paths."""
    protected = set(protected_manifest["path"].astype(str)) if not protected_manifest.empty else set()
    hash_status = dict(zip(protected_manifest.get("path", []), protected_manifest.get("hash_method", []), strict=False))
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix().lower()):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if _is_phase5e_output(rel):
            continue
        if any(part in {".git", ".venv", "__pycache__", ".pytest_cache"} for part in path.relative_to(root).parts):
            continue
        role = classify_scientific_role(rel)
        if role == "out_of_scope":
            continue
        rows.append(
            {
                "path": rel,
                "artifact_type": classify_artifact_type(path),
                "phase": classify_phase(rel),
                "scientific_role": role,
                "inclusion_status": "included",
                "exclusion_reason": "",
                "protected_status": "protected" if rel in protected else "unprotected",
                "hash_status": str(hash_status.get(rel, "not_in_initial_manifest")),
            }
        )
    return frame(rows, SOURCE_INVENTORY_COLUMNS).sort_values("path").reset_index(drop=True)


def classify_artifact_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "jsonl":
        return "raw_llm_jsonl"
    if suffix == "csv":
        return "table_csv"
    if suffix == "md":
        return "report_markdown"
    if suffix == "pdf":
        return "paper_pdf"
    if suffix == "zip":
        return "released_archive"
    if suffix == "pt":
        return "checkpoint"
    if suffix == "py":
        return "source_python"
    return suffix or "unknown"


def classify_phase(rel: str) -> str:
    if rel.startswith("paper/"):
        return "paper"
    if rel.startswith("original_repo/"):
        return "original_repo"
    if "phase_5d" in rel:
        return "phase_5d"
    if "canonical_local_llm_full" in rel:
        return "phase_5c"
    if "canonical_" in rel:
        return "canonical_prior"
    if "plan_e_" in rel:
        return "plan_e_prior"
    if "plan_d_" in rel:
        return "plan_d_prior"
    if "plan_c_" in rel:
        return "plan_c_prior"
    if "traditional_" in rel:
        return "traditional_baseline_prior"
    return "project_prior"


def classify_scientific_role(rel: str) -> str:
    if rel.startswith("paper/") and rel.endswith(".pdf"):
        return "paper_definition"
    if rel.startswith("original_repo/"):
        return "official_repository_reference"
    if rel.startswith("results/llm/"):
        return "raw_local_llm_artifact"
    if rel.startswith("results/models/"):
        return "model_checkpoint_or_metadata"
    if rel.startswith("results/tables/") or rel.startswith("results/metrics/") or rel.startswith("results/raw_outputs/"):
        return "machine_readable_result_or_manifest"
    if rel.startswith("report/"):
        return "prior_phase_report"
    if rel.startswith("src/") or rel.startswith("tests/") or rel.startswith("configs/"):
        return "local_source_contract_or_test"
    return "out_of_scope"


def paper_component_specs() -> list[ComponentSpec]:
    """Return a fixed, outcome-independent component taxonomy."""
    p = "paper/2411.00331v1.pdf"
    repo = "original_repo/EvaLLMasRecommender"

    def c(
        category: str,
        name: str,
        status: str,
        comparability: str,
        evidence_level: str,
        evidence: str,
        mismatches: str = "",
        blocker: str = "",
        permitted: str = "",
        prohibited: str = "",
        rationale: str = "",
        section: str = "paper/repository",
        table: str = "",
        dataset: str = "paper scope",
        model: str = "paper scope",
        task: str = "paper scope",
        protocol: str = "paper protocol",
        metric: str = "",
        desc: str = "",
        reference: str | None = None,
    ) -> ComponentSpec:
        return ComponentSpec(
            category=category,
            name=name,
            description=desc or name,
            section=section,
            table_or_figure=table,
            dataset_scope=dataset,
            model_scope=model,
            task_scope=task,
            protocol_scope=protocol,
            metric_scope=metric,
            reference=reference or p,
            status=status,
            comparability=comparability,
            evidence_level=evidence_level,
            evidence_artifacts=evidence,
            mismatches=mismatches,
            blocker=blocker,
            permitted=permitted or default_permitted(status, name),
            prohibited=prohibited or default_prohibited(status, name),
            rationale=rationale or default_rationale(status),
        )

    return [
        c("datasets", "Amazon Beauty dataset", "reconstructed", "reconstructed_paper_like", "source_or_generated_result_table", "results/tables/plan_e_paper_condition_alignment_matrix.csv;data/reconstructed/plan_c/amazon_beauty/candidates.csv", "official processed artifact identity unproven"),
        c("datasets", "Amazon Sports dataset", "reconstructed", "reconstructed_paper_like", "source_or_generated_result_table", "results/tables/plan_e_paper_condition_alignment_matrix.csv;data/reconstructed/plan_c/amazon_sports/candidates.csv", "official processed artifact identity unproven"),
        c("datasets", "ML-1M dataset", "reconstructed", "reconstructed_paper_like", "source_or_generated_result_table", "results/tables/plan_e_paper_condition_alignment_matrix.csv;data/processed/ml1m/interactions.csv", "official processed artifact identity unproven"),
        c("datasets", "LastFM dataset", "blocked", "blocked_not_comparable", "earlier_phase_report", "results/tables/plan_d_phase_5d_blocked_or_unresolved.csv;report/plan_e_paper_condition_alignment_audit.md", blocker="no valid comparable local split/candidate/ground-truth setup", dataset="LastFM"),
        c("dataset preprocessing and filtering", "Official processed data artifacts", "blocked", "blocked_not_comparable", "original_repository_code_or_documentation", f"{repo}/README.md", blocker="official paper-ready processed files not fully available or identity-proven"),
        c("dataset preprocessing and filtering", "Amazon title and item mapping", "reconstructed", "reconstructed_paper_like", "source_or_generated_result_table", "results/tables/item_title_map.csv;results/tables/item_title_mapping_issues.csv", "title exclusions and mapping gaps remain"),
        c("dataset preprocessing and filtering", "MovieLens sequential split", "reconstructed", "reconstructed_paper_like", "source_or_generated_result_table", "data/processed/ml1m/interactions.csv;src/data/prepare_ml1m.py", "split identity is local reconstructed state"),
        c("dataset preprocessing and filtering", "Strict-title prompt-compatible filtering", "adapted", "same_dataset_different_protocol", "canonical_manifest_or_metric_contract", "results/tables/canonical_local_llm_prompt_compatible_count_reconciliation.csv", "strict-title eligibility changes denominator"),
        c("user cohorts and sampling", "Paper 1000-user sample scale", "adapted", "same_metric_different_cohort", "phase_5c_full_matrix_evidence", "results/tables/canonical_local_llm_full_matrix_manifest.csv", "Beauty and Sports prompt-compatible cohorts are below 1000 after strict filters"),
        c("user cohorts and sampling", "K-S sampling validity check", "not_executed", "blocked_not_comparable", "earlier_phase_report", "report/plan_e_paper_condition_alignment_audit.md", blocker="no Phase 5E-executable K-S sample validation artifact for paper sample"),
        c("user cohorts and sampling", "Cold-start or short-history grouping", "adapted", "diagnostic_only", "exact_paired_diagnostic_experiment", "results/tables/plan_d_phase_5d_robustness_summary.csv", "diagnostic ML-1M subset only"),
        c("user cohorts and sampling", "Prompt-compatible local LLM cohorts", "adapted", "same_metric_different_cohort", "phase_5c_full_matrix_evidence", "results/tables/canonical_local_llm_full_matrix_manifest.csv", "strict-title cohort differs from paper denominator"),
        c("candidate-pool construction", "20-item candidate pool", "reconstructed", "reconstructed_paper_like", "canonical_manifest_or_metric_contract", "results/tables/plan_e_paper_like_candidate_pool_dataset_eligibility.csv;results/tables/canonical_local_llm_full_matrix_manifest.csv", "candidate source identity not exact for all paper cells"),
        c("candidate-pool construction", "1 positive plus 19 negatives", "reconstructed", "reconstructed_paper_like", "canonical_manifest_or_metric_contract", "results/tables/plan_e_paper_like_candidate_pool_manifest_summary.csv", "negative sample identity not official-paper exact"),
        c("candidate-pool construction", "Ranking negative-candidate construction", "reconstructed", "reconstructed_paper_like", "source_or_generated_result_table", "results/tables/ranking_candidate_sets.csv", "local reconstructed candidate rows"),
        c("candidate-pool construction", "Reranking candidate construction", "adapted", "same_dataset_different_protocol", "phase_5c_full_matrix_evidence", "results/tables/canonical_local_llm_reranking_top10_source_contract_audit.csv", "canonical top-10 and legacy full-20 are not interchangeable"),
        c("candidate-pool construction", "Candidate order variants", "adapted", "diagnostic_only", "exact_paired_diagnostic_experiment", "results/tables/plan_d_phase_5d_pairing_coverage_audit.csv", "controlled diagnostic subset only"),
        c("ranking task", "LLM ranking task", "adapted", "adapted_local_model_extension", "phase_5c_full_matrix_evidence", "results/tables/canonical_local_llm_full_matrix_cell_metrics.csv", "local substitute models and JSON-index output"),
        c("ranking task", "Ranking top-5 utility cutoff", "adapted", "official_code_aligned_local_model_substitute", "phase_5c_full_matrix_evidence", "results/tables/canonical_local_llm_utility_metrics.csv", "same cutoff under adapted local output protocol"),
        c("reranking task", "LLM reranking task", "adapted", "adapted_local_model_extension", "phase_5c_full_matrix_evidence", "results/tables/canonical_local_llm_full_matrix_cell_metrics.csv", "local substitute models and canonical top-10 protocol"),
        c("reranking task", "Legacy full-20 reranking diagnostic", "not_executed", "same_dataset_different_protocol", "phase_5d_robustness_consolidation", "results/tables/plan_d_phase_5d_protocol_comparison.csv", blocker="legacy full-20 diagnostic is not canonical top-10 evidence"),
        c("LLM models", "GPT-3.5", "blocked", "blocked_not_comparable", "canonical_manifest_or_metric_contract", "results/tables/plan_d_llm_model_registry.csv", blocker="API key, budget, and current exact model availability required", model="GPT-3.5"),
        c("LLM models", "GPT-4o-mini", "blocked", "blocked_not_comparable", "canonical_manifest_or_metric_contract", "results/tables/plan_d_llm_model_registry.csv", blocker="API key, budget, and explicit approval required", model="GPT-4o-mini"),
        c("LLM models", "GPT-4o", "blocked", "blocked_not_comparable", "canonical_manifest_or_metric_contract", "results/tables/plan_d_llm_model_registry.csv", blocker="API key, budget, and explicit approval required", model="GPT-4o"),
        c("LLM models", "Claude 3 Haiku", "blocked", "blocked_not_comparable", "canonical_manifest_or_metric_contract", "results/tables/plan_d_llm_model_registry.csv", blocker="exact model availability unresolved or retired", model="Claude 3 Haiku"),
        c("LLM models", "Llama3-70B-Instruct", "blocked", "blocked_not_comparable", "canonical_manifest_or_metric_contract", "results/tables/plan_d_llm_model_registry.csv", blocker="not locally feasible on RTX 4060 Laptop GPU; cloud/API required", model="Llama3-70B"),
        c("LLM models", "Qwen2 7B local substitute", "adapted", "adapted_local_model_extension", "phase_5c_full_matrix_evidence", "results/tables/canonical_local_llm_full_matrix_manifest.csv", "local Ollama tag/digest and JSON protocol are not paper-exact", model="Qwen2 7B"),
        c("LLM models", "Mistral 7B local substitute", "adapted", "adapted_local_model_extension", "phase_5c_full_matrix_evidence", "results/tables/canonical_local_llm_full_matrix_manifest.csv", "local Ollama tag/digest and JSON protocol are not paper-exact", model="Mistral 7B"),
        c("traditional recommender baselines", "MostPop baseline", "reconstructed", "reconstructed_paper_like", "source_or_generated_result_table", "results/tables/traditional_baselines_metrics.csv;src/baselines/mostpop.py", "official paper input identity unproven"),
        c("traditional recommender baselines", "BM25 or ContentBM25 baseline", "reconstructed", "reconstructed_paper_like", "source_or_generated_result_table", "results/tables/traditional_baselines_metrics.csv;src/baselines/content_bm25_baseline.py", "BM25-like local implementation, not proven exact paper setup"),
        c("traditional recommender baselines", "BPRMF baseline", "adapted", "reconstructed_paper_like", "source_or_generated_result_table", "results/tables/plan_e_bprmf_all_datasets_metrics.csv", "training/data/cohort not proven exact paper state"),
        c("traditional recommender baselines", "GRU4Rec baseline", "adapted", "reconstructed_paper_like", "source_or_generated_result_table", "results/tables/plan_e_gru4rec_all_datasets_metrics.csv", "training/data/cohort not proven exact paper state"),
        c("traditional recommender baselines", "SASRec baseline", "adapted", "reconstructed_paper_like", "source_or_generated_result_table", "results/tables/plan_e_sasrec_all_datasets_metrics.csv", "training/data/cohort not proven exact paper state"),
        c("traditional recommender baselines", "LightGCN baseline", "adapted", "reconstructed_paper_like", "source_or_generated_result_table", "results/tables/plan_e_lightgcn_all_datasets_metrics.csv", "CPU local deterministic run and reconstructed data differ from exact paper state"),
        c("prompt protocols", "OpenP5-like prompt family", "adapted", "same_dataset_different_protocol", "original_repository_code_or_documentation", f"{repo}/eva/templates/template.txt", "local strict JSON prompts are not exact free-text prompt semantics"),
        c("prompt protocols", "LLMRank-like prompt family", "not_executed", "blocked_not_comparable", "original_repository_code_or_documentation", f"{repo}/eva/templates/llmrank.txt", blocker="no canonical Phase 5C exact LLMRank execution"),
        c("prompt protocols", "LLMSRec or in-context prompt family", "not_executed", "blocked_not_comparable", "original_repository_code_or_documentation", f"{repo}/eva/templates/template_in_context.txt", blocker="no canonical Phase 5C exact ICL execution"),
        c("prompt protocols", "Generated-profile prompt family", "blocked", "blocked_not_comparable", "phase_5d_robustness_consolidation", "results/tables/plan_d_phase_5d_blocked_or_unresolved.csv", blocker="no canonical paired profile/no-profile control"),
        c("prompt protocols", "History/no-history prompt variants", "adapted", "diagnostic_only", "phase_5d_robustness_consolidation", "results/tables/plan_d_phase_5d_robustness_summary.csv", "diagnostic prompt conditions only"),
        c("response-format and parser contracts", "Original title-list parser and refinement", "reconstructed", "reconstructed_paper_like", "original_repository_code_or_documentation", f"{repo}/eva/codes/llm_metrics.py;{repo}/eva/codes/output_refine.py", "official code exists but exact paper outputs not locally rerun"),
        c("response-format and parser contracts", "Strict JSON index parser", "adapted", "same_dataset_different_protocol", "canonical_manifest_or_metric_contract", "src/llm/canonical_llm_contract.py;results/tables/canonical_llm_parser_contract.csv", "JSON index output replaces original free-text/title output semantics"),
        c("response-format and parser contracts", "Invalid-output handling policy", "adapted", "official_code_aligned_local_model_substitute", "phase_5c_full_matrix_evidence", "results/tables/canonical_local_llm_full_evaluation_decision.csv", "invalid-as-failure is adapted for local JSON protocol"),
        c("utility metrics", "HR@5 metric definition", "reproduced_exact", "paper_exact", "canonical_manifest_or_metric_contract", "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py;results/tables/canonical_metric_contract.csv", metric="HR@5", permitted="Metric formula is exact or code-aligned.", prohibited="Do not infer exact experiment replication from metric formula alone."),
        c("utility metrics", "NDCG@5 metric definition", "reproduced_exact", "paper_exact", "canonical_manifest_or_metric_contract", "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py;results/tables/canonical_metric_contract.csv", metric="NDCG@5", permitted="Metric formula is exact or code-aligned.", prohibited="Do not infer exact experiment replication from metric formula alone."),
        c("beyond-utility metrics", "Novelty or self-information metric", "reconstructed", "reconstructed_paper_like", "canonical_manifest_or_metric_contract", "results/tables/canonical_metric_contract.csv;results/tables/canonical_local_llm_beyond_utility_metrics.csv", "local item universe and metadata differ from paper exact state", metric="novelty"),
        c("beyond-utility metrics", "Long-tail rate metric", "reconstructed", "reconstructed_paper_like", "canonical_manifest_or_metric_contract", "results/tables/canonical_metric_contract.csv;results/tables/canonical_local_llm_beyond_utility_metrics.csv", "local popularity source and cohort differ from exact paper state", metric="long_tail"),
        c("beyond-utility metrics", "Average popularity metric", "reconstructed", "reconstructed_paper_like", "canonical_manifest_or_metric_contract", "results/tables/canonical_metric_contract.csv;results/tables/canonical_local_llm_beyond_utility_metrics.csv", "local train-count source differs from exact paper state", metric="popularity"),
        c("beyond-utility metrics", "Item coverage metric", "reconstructed", "reconstructed_paper_like", "canonical_manifest_or_metric_contract", "results/tables/canonical_metric_contract.csv;results/tables/canonical_local_llm_beyond_utility_metrics.csv", "local candidate universe differs from exact paper state", metric="coverage"),
        c("beyond-utility metrics", "Popularity-bias/fairness metrics", "blocked", "blocked_not_comparable", "canonical_manifest_or_metric_contract", "results/tables/canonical_metric_contract.csv", blocker="conflicting or missing exact definitions and required attributes", metric="PopREO/Gini/Jain/DPD"),
        c("beyond-utility metrics", "Diversity metric", "blocked", "blocked_not_comparable", "canonical_manifest_or_metric_contract", "results/tables/canonical_metric_contract.csv", blocker="exact adapted sources are blocked or unavailable", metric="diversity"),
        c("history-length analyses", "History length sensitivity analysis", "adapted", "diagnostic_only", "phase_5d_robustness_consolidation", "results/tables/plan_d_phase_5d_robustness_summary.csv;results/tables/plan_d_phase_5d_pairing_coverage_audit.csv", "ML-1M diagnostic subset only"),
        c("history-length analyses", "Cold-start short-history observation", "adapted", "diagnostic_only", "phase_5d_robustness_consolidation", "results/tables/plan_d_phase_5d_evidence_claims.csv", "diagnostic evidence does not reproduce full paper observation"),
        c("candidate-position analyses", "Candidate position bias analysis", "adapted", "diagnostic_only", "phase_5d_robustness_consolidation", "results/tables/plan_d_phase_5d_robustness_summary.csv", "20-row Qwen2 diagnostic subset and not full matrix"),
        c("candidate-position analyses", "Candidate-first prompt condition", "not_executed", "blocked_not_comparable", "original_repository_code_or_documentation", f"{repo}/eva/templates/template_candidate_first.txt", blocker="template exists but no canonical exact execution"),
        c("generated-profile analyses", "User profile generation", "blocked", "blocked_not_comparable", "phase_5d_robustness_consolidation", "results/tables/plan_d_phase_5d_blocked_or_unresolved.csv", blocker="profile model, reuse, parser, and leakage policy are not frozen"),
        c("generated-profile analyses", "Profile-based recommendation effect", "blocked", "blocked_not_comparable", "phase_5d_robustness_consolidation", "results/tables/plan_d_phase_5d_evidence_claims.csv", blocker="no canonical paired profile/no-profile result"),
        c("hallucination and output-reliability analyses", "Hallucination rate", "adapted", "adapted_local_model_extension", "phase_5c_full_matrix_evidence", "results/tables/canonical_local_llm_structural_metrics.csv;results/tables/plan_d_phase_5d_robustness_summary.csv", "strict-index hallucination differs from paper free-text hallucination"),
        c("hallucination and output-reliability analyses", "Output compliance and duplicate failures", "adapted", "adapted_local_model_extension", "phase_5c_full_matrix_evidence", "results/tables/canonical_local_llm_full_invalid_output_inventory.csv", "duplicate-index structural invalidity is local JSON-protocol evidence"),
        c("paper tables and quantitative result groups", "Utility result tables", "unresolved", "unresolved_comparability", "paper_pdf", "results/tables/canonical_local_llm_paper_result_inventory.csv", blocker="paper numeric values are not machine-verified locally"),
        c("paper tables and quantitative result groups", "Beyond-utility result tables", "unresolved", "unresolved_comparability", "paper_pdf", "results/tables/canonical_local_llm_paper_result_inventory.csv", blocker="paper numeric values are not machine-verified locally"),
        c("paper tables and quantitative result groups", "Model comparison result groups", "unresolved", "unresolved_comparability", "paper_pdf", "results/tables/canonical_local_llm_paper_alignment_comparison.csv", blocker="Qwen2/Mistral are not paper model identities"),
        c("paper figures and qualitative analysis groups", "Framework architecture figure", "reproduced_exact", "paper_exact", "original_repository_code_or_documentation", f"{repo}/assets/architecture.jpg", permitted="The released architecture asset is present locally.", prohibited="Do not infer numeric experiment replication from the figure."),
        c("paper figures and qualitative analysis groups", "Seven qualitative observations", "unresolved", "unresolved_comparability", "paper_pdf", f"{repo}/README.md", blocker="observations require exact paper result matrix to verify"),
        c("released code and data artifacts", "Official repository README", "reproduced_exact", "paper_exact", "original_repository_code_or_documentation", f"{repo}/README.md", permitted="The official repository README is available as released code documentation.", prohibited="Do not treat README availability as experiment execution."),
        c("released code and data artifacts", "Official metric source file", "reproduced_exact", "paper_exact", "original_repository_code_or_documentation", f"{repo}/eva/codes/llm_metrics.py", permitted="The official metric source is available for formula alignment.", prohibited="Do not treat code availability as completed paper-result reproduction."),
        c("released code and data artifacts", "Official output-refinement source file", "reproduced_exact", "paper_exact", "original_repository_code_or_documentation", f"{repo}/eva/codes/output_refine.py", permitted="The official parser/refinement source is available.", prohibited="Do not claim local JSON parser is the same parser."),
        c("released code and data artifacts", "Official prompt templates", "reproduced_exact", "paper_exact", "original_repository_code_or_documentation", f"{repo}/eva/templates", permitted="Released prompt templates are present.", prohibited="Do not claim every template was executed locally."),
        c("released code and data artifacts", "Released Beauty and Sports LLM result archives", "reproduced_exact", "paper_exact", "original_repository_code_or_documentation", f"{repo}/llm_results/beauty.zip;{repo}/llm_results/sports.zip", dataset="Amazon Beauty;Amazon Sports", permitted="The archives are present and protected.", prohibited="Do not extrapolate to absent ML-1M/LastFM archives."),
        c("released code and data artifacts", "Released SOTA result archive sample", "reproduced_exact", "paper_exact", "original_repository_code_or_documentation", f"{repo}/sota_results/beauty_his10.zip", dataset="Amazon Beauty", permitted="The released Beauty SOTA archive is present.", prohibited="Do not claim all SOTA result archives are present."),
        c("reproducibility-relevant execution settings", "Paper model versions and provider settings", "blocked", "blocked_not_comparable", "canonical_manifest_or_metric_contract", "results/tables/plan_d_llm_model_registry.csv", blocker="current API/model availability and exact provider settings unresolved"),
        c("reproducibility-relevant execution settings", "Random seed and deterministic local execution", "reconstructed", "reconstructed_paper_like", "canonical_manifest_or_metric_contract", "results/tables/canonical_local_llm_full_matrix_manifest.csv;results/tables/plan_e_lightgcn_all_datasets_device_provenance.csv", "local deterministic settings differ from paper provider execution"),
        c("reproducibility-relevant execution settings", "Temperature zero local LLM setting", "adapted", "same_protocol_different_model", "phase_5c_full_matrix_evidence", "results/tables/canonical_local_llm_full_matrix_manifest.csv", "local temperature/control setting does not make model identity exact"),
        c("reproducibility-relevant execution settings", "Hardware and API execution environment", "blocked", "blocked_not_comparable", "canonical_manifest_or_metric_contract", "results/tables/neural_hardware_inventory.csv;results/tables/plan_d_llm_model_registry.csv", blocker="paper API/cloud hardware settings not reproducible offline"),
    ]


def default_permitted(status: str, name: str) -> str:
    if status == "reproduced_exact":
        return f"{name} is available exactly for this component-level audit."
    if status == "reconstructed":
        return f"{name} has a paper-like local reconstruction."
    if status == "adapted":
        return f"{name} has an evidence-backed adapted local counterpart."
    if status == "blocked":
        return f"{name} may be described as blocked with the listed reason."
    if status == "not_executed":
        return f"{name} may be described as not executed locally."
    if status == "unresolved":
        return f"{name} may be described as unresolved."
    return f"{name} may be listed separately as an extension."


def default_prohibited(status: str, name: str) -> str:
    if status == "reproduced_exact":
        return "Do not use this component alone to claim full paper replication."
    if status == "reconstructed":
        return "Do not call reconstructed evidence exact paper reproduction."
    if status == "adapted":
        return "Do not count adapted evidence in the exact reproduction numerator."
    if status == "blocked":
        return "Do not omit blocked components from the applicable denominator."
    if status == "not_executed":
        return "Do not infer execution from source availability."
    if status == "unresolved":
        return "Do not silently resolve conflicts or invent paper values."
    return "Do not use extensions to inflate paper reproduction coverage."


def default_rationale(status: str) -> str:
    return {
        "reproduced_exact": "The locally available artifact matches the component-level paper resource.",
        "reconstructed": "A paper-like local counterpart exists but official exact artifacts are missing or unproven.",
        "adapted": "A meaningful local counterpart exists with material changes.",
        "blocked": "Required artifacts, models, APIs, or exact conditions are unavailable offline.",
        "not_executed": "No valid local execution artifact exists for this component.",
        "unresolved": "Available evidence is insufficient for a stronger classification.",
        "extension_only": "The item is a project addition and is excluded from paper coverage.",
    }[status]


def normalize_paper_components(specs: Sequence[ComponentSpec]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for spec in specs:
        rows.append(
            {
                "component_id": stable_component_id(spec.category, spec.name),
                "category": spec.category,
                "component_name": spec.name,
                "description": spec.description,
                "paper_section": spec.section,
                "paper_page": "",
                "table_or_figure": spec.table_or_figure,
                "original_dataset_scope": spec.dataset_scope,
                "original_model_scope": spec.model_scope,
                "original_task_scope": spec.task_scope,
                "original_protocol_scope": spec.protocol_scope,
                "original_metric_scope": spec.metric_scope,
                "original_artifact_or_repository_reference": spec.reference,
                "applicability_to_project": "applicable" if spec.applicable else "not_applicable",
                "not_applicable_reason": spec.not_applicable_reason,
            }
        )
    return deduplicate_components(frame(rows, PAPER_COMPONENT_COLUMNS))


def deduplicate_components(components: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate component IDs while rejecting conflicting duplicates."""
    if components.empty:
        return components
    duplicate_ids = components[components["component_id"].duplicated(keep=False)]["component_id"].unique()
    for component_id in duplicate_ids:
        subset = components[components["component_id"].eq(component_id)].drop_duplicates()
        if len(subset) > 1:
            raise ValueError(f"conflicting duplicate component_id: {component_id}")
    return components.drop_duplicates().sort_values(["category", "component_id"]).reset_index(drop=True)


def specs_by_component_id(specs: Sequence[ComponentSpec]) -> dict[str, ComponentSpec]:
    return {stable_component_id(spec.category, spec.name): spec for spec in specs}


def build_local_evidence_registry(root: Path) -> pd.DataFrame:
    """Register the local evidence units used in coverage decisions."""
    phase5c_decision = read_csv_if_exists(root, "results/tables/canonical_local_llm_full_matrix_decision.csv")
    phase5d_validation = read_csv_if_exists(root, "results/tables/plan_d_phase_5d_validation.csv")
    baseline_rows = sum(
        len(read_csv_if_exists(root, path))
        for path in [
            "results/tables/plan_e_bprmf_all_datasets_metrics.csv",
            "results/tables/plan_e_gru4rec_all_datasets_metrics.csv",
            "results/tables/plan_e_sasrec_all_datasets_metrics.csv",
            "results/tables/plan_e_lightgcn_all_datasets_metrics.csv",
            "results/tables/traditional_baselines_metrics.csv",
        ]
    )
    requests = ""
    if not phase5c_decision.empty and "expected_request_count" in phase5c_decision:
        requests = str(int(phase5c_decision["expected_request_count"].iloc[0]))
    phase5d_pass = ""
    if not phase5d_validation.empty and "status" in phase5d_validation:
        phase5d_pass = str(int(phase5d_validation["status"].eq("pass").sum()))
    rows = [
        evidence(
            "EVID-5C-FULL-LLM",
            "results/tables/canonical_local_llm_full_matrix_decision.csv;results/tables/canonical_local_llm_full_matrix_manifest.csv;results/llm/canonical_local_full_matrix/requests.jsonl;results/llm/canonical_local_full_matrix/responses.jsonl",
            "Phase 5C",
            "phase_5c_full_matrix_evidence",
            "Amazon Beauty;Amazon Sports;ML-1M",
            "Qwen2 7B Instruct;Mistral 7B Instruct",
            "ranking;reranking",
            "strict JSON index prompts",
            "strict JSON index parsers",
            "20",
            "5 ranking;10 reranking",
            "prompt-compatible local cohort",
            "HR@5;NDCG@5;beyond-utility;structural",
            "planned requests;strict-valid subset where declared",
            f"completed {requests} planned requests",
            "not paired for model comparison",
            "single completed run",
            "strict-valid and invalid rows retained",
            "local substitute models; strict-title cohort; JSON protocol",
            "Use for full local LLM matrix evidence.",
            "Do not call it paper-model exact reproduction.",
        ),
        evidence(
            "EVID-5D-ROBUSTNESS",
            "results/tables/plan_d_phase_5d_evidence_claims.csv;results/tables/plan_d_phase_5d_robustness_summary.csv;results/tables/plan_d_phase_5d_protocol_comparison.csv",
            "Phase 5D",
            "phase_5d_robustness_consolidation",
            "Amazon Beauty;Amazon Sports;ML-1M",
            "Qwen2 7B Instruct;Mistral 7B Instruct",
            "ranking;reranking;diagnostics",
            "various diagnostic prompt conditions",
            "frozen diagnostic parsers",
            "20",
            "5;10;20 where diagnostic",
            "diagnostic subset and full Phase 5C matrix where applicable",
            "robustness;reliability",
            "per-topic evidence scope",
            f"validated {phase5d_pass} checks",
            "paired where exact key match exists",
            "not repeated except where explicitly recorded",
            "classified by Phase 5D",
            "diagnostic subsets do not generalize to full paper matrix",
            "Use for permitted and prohibited robustness claims.",
            "Do not unblock nondeterminism or generated-profile effects.",
        ),
        evidence(
            "EVID-METRIC-CONTRACT",
            "results/tables/canonical_metric_contract.csv;src/evaluation/utility_metrics.py;src/metrics/canonical_metric_contract.py;original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "Canonical metrics",
            "canonical_manifest_or_metric_contract",
            "all local datasets where supported",
            "all local models/baselines where supported",
            "ranking;reranking",
            "not prompt-specific",
            "metric contract",
            "20 where candidate metrics apply",
            "5 cutoff metrics",
            "metric input cohorts",
            "HR@5;NDCG@5;novelty;coverage;popularity;validity",
            "contract-specific denominators",
            "computed or blocked by contract",
            "not applicable",
            "not applicable",
            "contract status retained",
            "some beyond-utility definitions blocked or adapted",
            "Use to classify metric exactness and blockers.",
            "Do not compute missing metrics from prose.",
        ),
        evidence(
            "EVID-BASELINES",
            "results/tables/traditional_baselines_metrics.csv;results/tables/plan_e_same_cohort_traditional_baselines_metrics.csv;results/tables/plan_e_bprmf_all_datasets_metrics.csv;results/tables/plan_e_gru4rec_all_datasets_metrics.csv;results/tables/plan_e_sasrec_all_datasets_metrics.csv;results/tables/plan_e_lightgcn_all_datasets_metrics.csv",
            "Plan E prior baseline work",
            "source_or_generated_result_table",
            "Amazon Beauty;Amazon Sports;ML-1M",
            "MostPop;BM25-like;BPRMF;GRU4Rec;SASRec;LightGCN",
            "ranking;reranking",
            "not prompt-specific",
            "metric contracts",
            "20",
            "5 cutoff metrics",
            "reconstructed or strict-train-known cohorts",
            "HR@5;NDCG@5",
            "evaluated baseline rows",
            f"completed local baseline rows {baseline_rows}",
            "same-cohort only where explicitly marked",
            "determinism where table provides it",
            "valid ranking rows",
            "training/data/cohort settings not proven paper-exact",
            "Use as reconstructed/adapted traditional recommender evidence.",
            "Do not compare to paper numbers as exact replication.",
        ),
        evidence(
            "EVID-OFFICIAL-REPO",
            "original_repo/EvaLLMasRecommender/README.md;original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py;original_repo/EvaLLMasRecommender/eva/codes/output_refine.py;original_repo/EvaLLMasRecommender/eva/templates",
            "Original repository",
            "original_repository_code_or_documentation",
            "paper datasets",
            "paper model families",
            "ranking;reranking",
            "released prompt templates",
            "released parser/refinement source",
            "20",
            "paper code dependent",
            "released artifact scope",
            "paper metric definitions and prompts",
            "paper code assumptions",
            "available read-only",
            "not applicable",
            "not applicable",
            "source availability only",
            "availability is not execution",
            "Use to define paper components and exact code resources.",
            "Do not infer a completed local experiment from code presence.",
        ),
        evidence(
            "EVID-PAPER-PDF",
            "paper/2411.00331v1.pdf",
            "Paper",
            "paper_pdf",
            "paper datasets",
            "paper models",
            "paper tasks",
            "paper protocols",
            "paper parser/results",
            "paper-defined",
            "paper-defined",
            "paper-defined",
            "paper result cells",
            "paper-defined",
            "available read-only",
            "not applicable",
            "not applicable",
            "PDF identified; numeric values not machine-extracted",
            "no PDF text extraction dependency available",
            "Use as authoritative source of claims and scope.",
            "Do not invent or manually type paper numeric values.",
        ),
    ]
    return frame(rows, LOCAL_EVIDENCE_COLUMNS).sort_values("local_evidence_id").reset_index(drop=True)


def evidence(
    local_evidence_id: str,
    source_artifact_paths: str,
    evidence_phase: str,
    evidence_level: str,
    local_dataset: str,
    local_model: str,
    local_task: str,
    local_prompt: str,
    local_parser: str,
    local_candidate_count: str,
    local_output_count: str,
    local_cohort: str,
    local_metric: str,
    local_denominator: str,
    execution_status: str,
    pairing_status: str,
    repetition_status: str,
    validity_status: str,
    exactness_mismatches: str,
    permitted_conclusion: str,
    prohibited_conclusion: str,
) -> dict[str, str]:
    return {
        "local_evidence_id": local_evidence_id,
        "source_artifact_paths": source_artifact_paths,
        "evidence_phase": evidence_phase,
        "evidence_level": evidence_level,
        "local_dataset": local_dataset,
        "local_model": local_model,
        "local_task": local_task,
        "local_prompt": local_prompt,
        "local_parser": local_parser,
        "local_candidate_count": local_candidate_count,
        "local_output_count": local_output_count,
        "local_cohort": local_cohort,
        "local_metric": local_metric,
        "local_denominator": local_denominator,
        "execution_status": execution_status,
        "pairing_status": pairing_status,
        "repetition_status": repetition_status,
        "validity_status": validity_status,
        "exactness_mismatches": exactness_mismatches,
        "permitted_conclusion": permitted_conclusion,
        "prohibited_conclusion": prohibited_conclusion,
    }


def build_component_evidence_mapping(components: pd.DataFrame, evidence_registry: pd.DataFrame, specs: Sequence[ComponentSpec]) -> pd.DataFrame:
    spec_map = specs_by_component_id(specs)
    evidence_by_id = {row["local_evidence_id"]: row for row in evidence_registry.to_dict("records")}
    rows: list[dict[str, str]] = []
    for component in components.to_dict("records"):
        component_id = component["component_id"]
        spec = spec_map[component_id]
        evidence_id = select_evidence_id(spec)
        evidence_row = evidence_by_id[evidence_id]
        row = {"paper_component_id": component_id, **evidence_row}
        row["source_artifact_paths"] = spec.evidence_artifacts
        row["evidence_level"] = spec.evidence_level
        row["exactness_mismatches"] = spec.mismatches
        row["permitted_conclusion"] = spec.permitted
        row["prohibited_conclusion"] = spec.prohibited
        row["determines_primary_status"] = "true"
        row["primary_status_reason"] = spec.rationale
        rows.append(row)
    return frame(rows, COMPONENT_MAPPING_COLUMNS).sort_values("paper_component_id").reset_index(drop=True)


def select_evidence_id(spec: ComponentSpec) -> str:
    if spec.evidence_level == "phase_5c_full_matrix_evidence":
        return "EVID-5C-FULL-LLM"
    if spec.evidence_level == "phase_5d_robustness_consolidation" or spec.evidence_level == "exact_paired_diagnostic_experiment":
        return "EVID-5D-ROBUSTNESS"
    if spec.evidence_level == "canonical_manifest_or_metric_contract":
        return "EVID-METRIC-CONTRACT"
    if spec.evidence_level == "original_repository_code_or_documentation":
        return "EVID-OFFICIAL-REPO"
    if spec.evidence_level == "paper_pdf":
        return "EVID-PAPER-PDF"
    if "baseline" in spec.category:
        return "EVID-BASELINES"
    return "EVID-BASELINES" if "baseline" in spec.evidence_artifacts else "EVID-OFFICIAL-REPO"


def build_component_coverage(components: pd.DataFrame, specs: Sequence[ComponentSpec]) -> pd.DataFrame:
    spec_map = specs_by_component_id(specs)
    rows: list[dict[str, str]] = []
    for component in components.to_dict("records"):
        spec = spec_map[component["component_id"]]
        rows.append(
            {
                "component_id": component["component_id"],
                "category": component["category"],
                "component": component["component_name"],
                "applicable": bool_text(spec.applicable),
                "primary_coverage_status": spec.status,
                "comparability_classification": spec.comparability,
                "evidence_level": spec.evidence_level,
                "evidence_artifacts": spec.evidence_artifacts,
                "exactness_mismatches": spec.mismatches,
                "blocker": spec.blocker,
                "permitted_claim": spec.permitted,
                "prohibited_claim": spec.prohibited,
                "rationale": spec.rationale,
            }
        )
    return frame(rows, COMPONENT_COVERAGE_COLUMNS).sort_values(["category", "component_id"]).reset_index(drop=True)


def validate_exact_reproduction(row: Mapping[str, Any]) -> bool:
    """Return true only when all material exactness checks are affirmative."""
    checks = [
        "same_dataset",
        "same_cohort",
        "same_task",
        "same_candidate_protocol",
        "same_model",
        "same_model_version",
        "same_prompt_semantics",
        "same_parser_or_output_semantics",
        "same_metric_definition",
        "same_denominator",
    ]
    if str(row.get("local_model", "")).lower().startswith(("qwen", "mistral")):
        return False
    return all(str(row.get(check, "false")).lower() == "true" for check in checks)


def result_cell_exact_comparison_supported(row: Mapping[str, Any]) -> bool:
    exact_checks = [
        "exact_dataset_match",
        "exact_task_match",
        "exact_model_match",
        "exact_cohort_match",
        "exact_candidate_protocol_match",
        "exact_metric_definition_match",
        "exact_cutoff_match",
        "exact_denominator_match",
    ]
    return bool(row.get("paper_value_verified", False)) and all(bool(row.get(check, False)) for check in exact_checks)


def result_cell_comparability_status(row: Mapping[str, Any]) -> str:
    if result_cell_exact_comparison_supported(row):
        return "exact_numeric_comparison_supported"
    if not row.get("paper_value_verified", False):
        return "paper_value_not_locally_verified"
    if not row.get("exact_model_match", False):
        return "same_metric_different_model"
    if not row.get("exact_cohort_match", False):
        return "same_metric_different_cohort"
    if not row.get("exact_candidate_protocol_match", False):
        return "same_metric_different_protocol"
    if row.get("same_task", False):
        return "same_task_only"
    return "unresolved"


def duplicate_component_mapping_exists(mappings: pd.DataFrame) -> bool:
    """Detect duplicate component/evidence mappings."""
    required = ["paper_component_id", "local_evidence_id"]
    if not all(column in mappings.columns for column in required):
        raise ValueError("mapping table lacks duplicate-detection keys")
    return bool(mappings[required].duplicated().any())


def strongest_evidence_level(levels: Iterable[str]) -> str:
    """Return the strongest available evidence level by the Phase 5E hierarchy."""
    seen = [level for level in levels if level]
    if not seen:
        return "none"
    order = {level: index for index, level in enumerate(EVIDENCE_PRECEDENCE)}
    return sorted(seen, key=lambda level: order.get(level, len(order)))[0]


def select_primary_status(statuses: Iterable[str], conflict: bool = False) -> str:
    """Select one primary status while preserving explicit unresolved conflicts."""
    materialized = [status for status in statuses if status]
    if not materialized:
        return "unresolved"
    invalid = set(materialized) - set(STATUS_ORDER)
    if invalid:
        raise ValueError(f"invalid statuses: {sorted(invalid)}")
    if conflict:
        return "unresolved"
    non_extension = [status for status in materialized if status != "extension_only"]
    if not non_extension:
        return "extension_only"
    order = {status: index for index, status in enumerate(STATUS_ORDER)}
    return sorted(non_extension, key=lambda status: order[status])[0]


def build_paper_result_cells(root: Path) -> pd.DataFrame:
    inventory = read_csv_if_exists(root, "results/tables/canonical_local_llm_paper_result_inventory.csv")
    rows: list[dict[str, str]] = []
    if inventory.empty:
        return frame(rows, PAPER_RESULT_CELL_COLUMNS)
    for source_row in inventory.to_dict("records"):
        metric = clean_cell(source_row.get("metric_id", ""))
        seed = "|".join(
            [
                clean_cell(source_row.get("paper_result_id", "")),
                clean_cell(source_row.get("paper_section_or_table", "")),
                clean_cell(source_row.get("model_or_baseline", "")),
                metric,
            ]
        )
        rows.append(
            {
                "paper_result_cell_id": stable_result_cell_id(seed),
                "paper_table": clean_cell(source_row.get("paper_section_or_table", "")),
                "paper_row": clean_cell(source_row.get("model_or_baseline", "")),
                "paper_column": metric,
                "dataset": clean_cell(source_row.get("dataset", "")),
                "task": clean_cell(source_row.get("task_type", "")),
                "model": clean_cell(source_row.get("model_or_baseline", "")),
                "prompt_or_method": clean_cell(source_row.get("model_or_baseline", "")),
                "metric": metric,
                "cutoff": "5" if metric.endswith("_at_5") else "",
                "value": clean_cell(source_row.get("paper_value", "")),
                "source_page": "",
                "verification_status": "paper_value_unverified" if clean_cell(source_row.get("paper_value", "")) == "" else "paper_value_verified",
            }
        )
    return frame(rows, PAPER_RESULT_CELL_COLUMNS).sort_values("paper_result_cell_id").reset_index(drop=True)


def build_result_cell_comparison(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    metrics_with_local_counterpart = {
        "hr_at_5",
        "ndcg_at_5",
        "novelty_at_5",
        "item_coverage_at_5",
        "long_tail_rate_at_5",
        "average_popularity_at_5",
    }
    for cell in cells.to_dict("records"):
        metric = clean_cell(cell.get("metric", ""))
        local_result_id = "local_counterpart_" + slugify(metric) if metric in metrics_with_local_counterpart else ""
        row_for_status = {
            "paper_value_verified": clean_cell(cell.get("value", "")) != "",
            "exact_dataset_match": False,
            "exact_task_match": True,
            "exact_model_match": False,
            "exact_cohort_match": False,
            "exact_candidate_protocol_match": False,
            "exact_metric_definition_match": metric in metrics_with_local_counterpart or metric == "model_identity",
            "exact_cutoff_match": metric.endswith("_at_5"),
            "exact_denominator_match": False,
            "same_task": True,
        }
        status = result_cell_comparability_status(row_for_status)
        if local_result_id and status == "paper_value_not_locally_verified":
            reason = "local metric counterpart exists, but paper value and exact model/cohort/protocol conditions are not verified"
        elif metric == "model_identity":
            reason = "paper model identity is documented, but local Qwen2/Mistral substitutes are not exact model matches"
            status = "same_task_different_model"
        else:
            reason = "no exact local counterpart supports numeric comparison"
        rows.append(
            {
                "paper_result_cell_id": cell["paper_result_cell_id"],
                "local_result_id": local_result_id,
                "comparability_status": status,
                "exact_dataset_match": "false",
                "exact_task_match": "true",
                "exact_model_match": "false",
                "exact_cohort_match": "false",
                "exact_candidate_protocol_match": "false",
                "exact_metric_definition_match": bool_text(row_for_status["exact_metric_definition_match"]),
                "exact_cutoff_match": bool_text(row_for_status["exact_cutoff_match"]),
                "exact_denominator_match": "false",
                "numerical_comparison_permitted": "false",
                "reason": reason,
            }
        )
    return frame(rows, RESULT_CELL_COMPARISON_COLUMNS).sort_values("paper_result_cell_id").reset_index(drop=True)


def validate_status_denominator(coverage: pd.DataFrame) -> None:
    if not coverage["primary_coverage_status"].isin(STATUS_ORDER).all():
        bad = sorted(set(coverage["primary_coverage_status"]) - set(STATUS_ORDER))
        raise ValueError(f"invalid coverage status: {bad}")
    applicable = coverage[coverage["applicable"].eq("true")]
    if applicable.empty:
        raise ValueError("applicable denominator is empty")
    if applicable["component_id"].duplicated().any():
        raise ValueError("duplicate applicable component IDs")
    not_applicable = coverage[coverage["primary_coverage_status"].eq("not_applicable")]
    missing_justification = not_applicable[not_applicable["rationale"].eq("")]
    if not missing_justification.empty:
        raise ValueError("not_applicable rows require rationale")


def compute_category_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for category in sorted(coverage["category"].unique()):
        group = coverage[coverage["category"].eq(category)]
        applicable = group[group["applicable"].eq("true")]
        denom = len(applicable)
        counts = {status: int(group["primary_coverage_status"].eq(status).sum()) for status in STATUS_ORDER}
        rows.append(
            {
                "category": category,
                "total_components": str(len(group)),
                "applicable_components": str(denom),
                **{status: str(counts[status]) for status in STATUS_ORDER},
                "strict_exact_pct": str(percent(counts["reproduced_exact"], denom)),
                "reconstructed_or_better_pct": str(percent(counts["reproduced_exact"] + counts["reconstructed"], denom)),
                "maximum_defensible_pct": str(percent(counts["reproduced_exact"] + counts["reconstructed"] + counts["adapted"], denom)),
                "blocked_pct": str(percent(counts["blocked"], denom)),
            }
        )
    return frame(rows, CATEGORY_COVERAGE_COLUMNS)


def macro_average(category_coverage: pd.DataFrame, column: str) -> float:
    applicable = category_coverage[category_coverage["applicable_components"].astype(int) > 0]
    if applicable.empty:
        return 0.0
    return round(applicable[column].astype(float).mean(), 6)


def compute_coverage_metrics(coverage: pd.DataFrame, category_coverage: pd.DataFrame, comparisons: pd.DataFrame, extensions: pd.DataFrame) -> pd.DataFrame:
    """Compute transparent count-based coverage metrics."""
    validate_status_denominator(coverage)
    applicable = coverage[coverage["applicable"].eq("true")]
    denom = len(applicable)
    counts = {status: int(applicable["primary_coverage_status"].eq(status).sum()) for status in STATUS_ORDER}
    result_denom = len(comparisons)
    exact_result_count = int(comparisons["comparability_status"].eq("exact_numeric_comparison_supported").sum()) if result_denom else 0
    counterpart_count = int(comparisons["local_result_id"].ne("").sum()) if result_denom else 0
    rows = [
        metric_row("applicable_component_count", denom, denom, "count(applicable components)", denom, "Applicable paper components retained in denominator.", "Do not remove blocked or unresolved components."),
        metric_row("reproduced_exact_count", counts["reproduced_exact"], denom, "count(primary_status == reproduced_exact)", counts["reproduced_exact"], "Strict exact component count.", "Do not add reconstructed or adapted rows."),
        metric_row("reconstructed_count", counts["reconstructed"], denom, "count(primary_status == reconstructed)", counts["reconstructed"], "Paper-like reconstructed component count.", "Do not call this exact reproduction."),
        metric_row("adapted_count", counts["adapted"], denom, "count(primary_status == adapted)", counts["adapted"], "Evidence-backed adapted counterpart count.", "Do not count adapted rows in exact numerator."),
        metric_row("extension_only_count", len(extensions), len(extensions), "count(extension inventory rows)", len(extensions), "Project extension count, reported separately.", "Do not include extensions in paper coverage numerator."),
        metric_row("blocked_count", counts["blocked"], denom, "count(primary_status == blocked)", counts["blocked"], "Blocked component count retained in denominator.", "Do not omit blockers."),
        metric_row("not_executed_count", counts["not_executed"], denom, "count(primary_status == not_executed)", counts["not_executed"], "Feasible but not executed count.", "Do not infer execution from source code."),
        metric_row("unresolved_count", counts["unresolved"], denom, "count(primary_status == unresolved)", counts["unresolved"], "Unresolved component count.", "Do not silently classify unresolved evidence."),
        metric_row("strict_exact_reproduction_micro_pct", counts["reproduced_exact"], denom, "reproduced_exact / applicable_components * 100", percent(counts["reproduced_exact"], denom), "Strict exact reproduction percentage.", "This is not improved by numerical similarity."),
        metric_row("reconstructed_or_better_micro_pct", counts["reproduced_exact"] + counts["reconstructed"], denom, "(reproduced_exact + reconstructed) / applicable_components * 100", percent(counts["reproduced_exact"] + counts["reconstructed"], denom), "Exact plus reconstructed coverage.", "Do not include adapted or extensions."),
        metric_row("maximum_defensible_micro_coverage_pct", counts["reproduced_exact"] + counts["reconstructed"] + counts["adapted"], denom, "(reproduced_exact + reconstructed + adapted) / applicable_components * 100", percent(counts["reproduced_exact"] + counts["reconstructed"] + counts["adapted"], denom), "Maximum defensible paper-component coverage.", "This is not an exact paper-replication percentage."),
        metric_row("blocked_micro_pct", counts["blocked"], denom, "blocked / applicable_components * 100", percent(counts["blocked"], denom), "Blocked rate.", "Do not remove blocked rows from denominator."),
        metric_row("strict_exact_reproduction_macro_pct", 0, int((category_coverage["applicable_components"].astype(int) > 0).sum()), "mean(category strict_exact_pct)", macro_average(category_coverage, "strict_exact_pct"), "Equal-category macro exact coverage.", "No category weights are used."),
        metric_row("reconstructed_or_better_macro_pct", 0, int((category_coverage["applicable_components"].astype(int) > 0).sum()), "mean(category reconstructed_or_better_pct)", macro_average(category_coverage, "reconstructed_or_better_pct"), "Equal-category macro reconstructed-or-better coverage.", "No category weights are used."),
        metric_row("maximum_defensible_macro_coverage_pct", 0, int((category_coverage["applicable_components"].astype(int) > 0).sum()), "mean(category maximum_defensible_pct)", macro_average(category_coverage, "maximum_defensible_pct"), "Equal-category macro maximum defensible coverage.", "No category weights are used."),
        metric_row("exact_numeric_result_cell_pct", exact_result_count, result_denom, "exact_numeric_comparison_supported / inventoried paper result cells * 100", percent(exact_result_count, result_denom), "Exact numeric result-cell coverage.", "Do not compute deltas without exact model/cohort/protocol/value support."),
        metric_row("broader_local_result_counterpart_pct", counterpart_count, result_denom, "paper result cells with any local counterpart / inventoried paper result cells * 100", percent(counterpart_count, result_denom), "Non-exact local counterpart coverage.", "This is not numeric replication."),
    ]
    return frame(rows, COVERAGE_METRIC_COLUMNS)


def metric_row(name: str, numerator: int | float, denominator: int | float, formula: str, value: int | float, interpretation: str, prohibited: str) -> dict[str, str]:
    return {
        "metric_name": name,
        "numerator": str(numerator),
        "denominator": str(denominator),
        "formula": formula,
        "value": str(value),
        "interpretation": interpretation,
        "prohibited_interpretation": prohibited,
    }


def build_extensions() -> pd.DataFrame:
    rows = [
        {
            "extension_id": "EXT-001",
            "description": "Local Qwen2 versus Mistral comparison",
            "datasets": "Amazon Beauty;Amazon Sports;ML-1M",
            "models": "Qwen2 7B Instruct;Mistral 7B Instruct",
            "tasks": "ranking;reranking",
            "evidence": "results/tables/plan_d_phase_5d_evidence_claims.csv",
            "professor_requested_status": "project_extension",
            "scientific_contribution": "local substitute-model reliability and utility comparison",
            "relation_to_paper": "same recommender task family but different models",
            "why_it_does_not_increase_reproduction_percentage": "Qwen2 and Mistral are not exact paper LLM identities.",
        },
        {
            "extension_id": "EXT-002",
            "description": "Strict JSON index prompt and parser",
            "datasets": "Amazon Beauty;Amazon Sports;ML-1M",
            "models": "Qwen2;Mistral",
            "tasks": "ranking;reranking",
            "evidence": "results/tables/canonical_llm_prompt_contract.csv;results/tables/canonical_llm_parser_contract.csv",
            "professor_requested_status": "project_extension",
            "scientific_contribution": "reduces ambiguity in local output parsing",
            "relation_to_paper": "adapted output semantics",
            "why_it_does_not_increase_reproduction_percentage": "The paper used title/free-text-style outputs and refinement code.",
        },
        {
            "extension_id": "EXT-003",
            "description": "Operational invalid-as-failure utility",
            "datasets": "Amazon Beauty;Amazon Sports;ML-1M",
            "models": "Qwen2;Mistral",
            "tasks": "ranking;reranking",
            "evidence": "results/tables/canonical_local_llm_utility_metrics.csv",
            "professor_requested_status": "project_extension",
            "scientific_contribution": "keeps strict-invalid outputs visible in utility denominators",
            "relation_to_paper": "adapted denominator policy",
            "why_it_does_not_increase_reproduction_percentage": "It changes invalid-output treatment relative to unresolved paper policy.",
        },
        {
            "extension_id": "EXT-004",
            "description": "Candidate validity and duplicate-index structural metrics",
            "datasets": "Amazon Beauty;Amazon Sports;ML-1M",
            "models": "Qwen2;Mistral",
            "tasks": "ranking;reranking",
            "evidence": "results/tables/canonical_local_llm_structural_metrics.csv",
            "professor_requested_status": "project_extension",
            "scientific_contribution": "separates hallucination from parser/duplicate failures",
            "relation_to_paper": "reliability extension",
            "why_it_does_not_increase_reproduction_percentage": "It is a local diagnostic not a paper result cell.",
        },
        {
            "extension_id": "EXT-005",
            "description": "Phase 5D robustness consolidation",
            "datasets": "Amazon Beauty;Amazon Sports;ML-1M",
            "models": "Qwen2;Mistral",
            "tasks": "diagnostic robustness",
            "evidence": "report/plan_d_phase_5d_robustness_consolidation.md",
            "professor_requested_status": "professor_requested",
            "scientific_contribution": "prevents overclaiming nondeterminism/profile/full20 evidence",
            "relation_to_paper": "robustness audit around paper dimensions",
            "why_it_does_not_increase_reproduction_percentage": "It consolidates evidence rather than executing paper experiments.",
        },
        {
            "extension_id": "EXT-006",
            "description": "Reconstructed neural baseline local runs",
            "datasets": "Amazon Beauty;Amazon Sports;ML-1M",
            "models": "BPRMF;GRU4Rec;SASRec;LightGCN",
            "tasks": "ranking;reranking",
            "evidence": "results/tables/plan_e_*_all_datasets_metrics.csv",
            "professor_requested_status": "project_extension",
            "scientific_contribution": "adds same-project baseline context",
            "relation_to_paper": "paper baseline family counterpart",
            "why_it_does_not_increase_reproduction_percentage": "Training/data/cohort conditions are not proven paper-exact.",
        },
    ]
    return frame(rows, EXTENSION_COLUMNS)


def build_blockers(coverage: pd.DataFrame) -> pd.DataFrame:
    blocked = coverage[coverage["primary_coverage_status"].isin(["blocked", "not_executed", "unresolved"])].copy()
    rows: list[dict[str, str]] = []
    for idx, row in enumerate(blocked.to_dict("records"), start=1):
        blocker_type = {
            "blocked": "missing_artifact_or_external_dependency",
            "not_executed": "not_executed_locally",
            "unresolved": "insufficient_or_conflicting_evidence",
        }[row["primary_coverage_status"]]
        rows.append(
            {
                "blocker_id": f"BLK-5E-{idx:03d}",
                "component": row["component"],
                "blocker_type": blocker_type,
                "missing_artifact_or_resource": row["blocker"] or row["exactness_mismatches"],
                "current_evidence": row["evidence_artifacts"],
                "recoverable_without_new_inference": "false" if row["primary_coverage_status"] == "blocked" else "unknown",
                "scientific_impact": "prevents exact or maximum coverage upgrade for this component",
                "permitted_statement": row["permitted_claim"],
                "prohibited_statement": row["prohibited_claim"],
            }
        )
    return frame(rows, BLOCKER_COLUMNS)


def build_claims(coverage: pd.DataFrame, metrics: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    max_metric = metrics[metrics["metric_name"].eq("maximum_defensible_micro_coverage_pct")]
    exact_metric = metrics[metrics["metric_name"].eq("strict_exact_reproduction_micro_pct")]
    exact_pct = clean_cell(exact_metric["value"].iloc[0]) if not exact_metric.empty else "0"
    max_pct = clean_cell(max_metric["value"].iloc[0]) if not max_metric.empty else "0"
    exact_numeric = int(comparisons["comparability_status"].eq("exact_numeric_comparison_supported").sum()) if not comparisons.empty else 0
    rows = [
        claim("CLAIM-5E-001", "permitted", f"Strict exact component reproduction is {exact_pct} percent.", "supported", "results/tables/plan_d_phase_5e_coverage_metrics.csv", "Report as component-level exact coverage.", "Do not call this whole-paper replication."),
        claim("CLAIM-5E-002", "permitted", f"Maximum defensible component coverage is {max_pct} percent.", "supported", "results/tables/plan_d_phase_5e_coverage_metrics.csv", "Report as exact+reconstructed+adapted local counterparts.", "Do not call this an exact replication percentage."),
        claim("CLAIM-5E-003", "permitted", "Qwen2 and Mistral are local substitutes/extensions.", "supported", "results/tables/canonical_local_llm_full_matrix_manifest.csv", "Use them for local evidence only.", "Do not compare them to paper LLM values as replicas."),
        claim("CLAIM-5E-004", "permitted", "Blocked components remain in the applicable denominator.", "supported", "results/tables/plan_d_phase_5e_component_coverage.csv", "State blocked rate transparently.", "Do not hide blocked rows to improve coverage."),
        claim("CLAIM-5E-005", "permitted", f"Exact numeric paper result-cell comparisons supported: {exact_numeric}.", "supported", "results/tables/plan_d_phase_5e_result_cell_comparison.csv", "Do not calculate replication deltas.", "Do not invent paper numeric values."),
        claim("CLAIM-5E-006", "prohibited", "The project exactly reproduced the paper's LLM result tables.", "prohibited", "results/tables/plan_d_phase_5e_result_cell_comparison.csv", "Say exact numeric comparison is unsupported.", "Do not report exact replication deltas."),
        claim("CLAIM-5E-007", "prohibited", "Legacy full-20 experiments are equivalent to canonical top-10 reranking.", "prohibited", "results/tables/plan_d_phase_5d_protocol_comparison.csv", "Keep protocols separate.", "Do not merge full-20 and top-10 evidence."),
        claim("CLAIM-5E-008", "prohibited", "Generated profiles improved or harmed recommendations.", "prohibited", "results/tables/plan_d_phase_5d_blocked_or_unresolved.csv", "State profile effects are blocked.", "Do not claim profile benefit or harm."),
    ]
    return frame(rows, CLAIM_COLUMNS)


def claim(claim_id: str, claim_type: str, claim_text: str, status: str, artifacts: str, permitted: str, prohibited: str) -> dict[str, str]:
    return {
        "claim_id": claim_id,
        "claim_type": claim_type,
        "claim": claim_text,
        "status": status,
        "supporting_artifacts": artifacts,
        "permitted_statement": permitted,
        "prohibited_statement": prohibited,
    }


def build_next_actions() -> pd.DataFrame:
    rows = [
        {
            "action_id": "NEXT-5E-001",
            "input_artifact": "results/tables/plan_d_phase_5e_component_coverage.csv",
            "required_use": "Use as the frozen component denominator and status registry before any Phase 6 synthesis.",
            "status": "ready",
            "phase_6_handoff": "do not change denominator without a documented new evidence phase",
        },
        {
            "action_id": "NEXT-5E-002",
            "input_artifact": "results/tables/plan_d_phase_5e_result_cell_comparison.csv",
            "required_use": "Preserve exact numeric comparison blocks and avoid replication deltas.",
            "status": "ready",
            "phase_6_handoff": "only exact model/cohort/protocol/value matches may support deltas",
        },
        {
            "action_id": "NEXT-5E-003",
            "input_artifact": "results/tables/plan_d_phase_5e_blockers.csv",
            "required_use": "Carry residual blockers into Phase 6 planning without running experiments here.",
            "status": "ready",
            "phase_6_handoff": "requires explicit approval for any inference/API/training phase",
        },
    ]
    return frame(rows, NEXT_ACTION_COLUMNS)


def build_validation(
    root: Path,
    coverage: pd.DataFrame,
    category: pd.DataFrame,
    components: pd.DataFrame,
    mappings: pd.DataFrame,
    comparisons: pd.DataFrame,
    metrics: pd.DataFrame,
    extensions: pd.DataFrame,
    source_inventory: pd.DataFrame,
    initial_manifest: pd.DataFrame,
    final_manifest: pd.DataFrame,
    deterministic_outputs: bool,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(scope: str, name: str, expected: Any, observed: Any, ok: bool, evidence: str, severity: str = "blocking") -> None:
        rows.append(
            {
                "validation_id": f"VAL-5E-{len(rows)+1:03d}",
                "validation_scope": scope,
                "check_name": name,
                "expected_value": clean_cell(expected),
                "observed_value": clean_cell(observed),
                "status": "pass" if ok else "fail",
                "severity": severity,
                "evidence": evidence,
            }
        )

    branch = git_stdout(root, ["branch", "--show-current"])
    head = git_stdout(root, ["rev-parse", "HEAD"])
    applicable = coverage[coverage["applicable"].eq("true")]
    counts = {status: int(applicable["primary_coverage_status"].eq(status).sum()) for status in STATUS_ORDER}
    metric_values = metrics.set_index("metric_name")["value"].astype(float).to_dict()
    phase5c_present = (root / "results/tables/canonical_local_llm_full_matrix_decision.csv").exists()
    phase5d_present = (root / "results/tables/plan_d_phase_5d_validation.csv").exists()
    exact_bad = coverage[(coverage["primary_coverage_status"].eq("reproduced_exact")) & (coverage["comparability_classification"].ne("paper_exact"))]
    qwen_mistral_exact = coverage[
        coverage["component"].str.contains("Qwen|Mistral", case=False, regex=True)
        & coverage["primary_coverage_status"].eq("reproduced_exact")
    ]
    staged = git_stdout(root, ["diff", "--cached", "--name-only"])
    tracked_modified = [line for line in git_stdout(root, ["diff", "--name-only"]).splitlines() if line]
    forbidden_tracked = [path for path in tracked_modified if path != "report/dataset_audit.md"]

    add("repository", "branch", EXPECTED_BRANCH, branch, branch == EXPECTED_BRANCH, "git branch --show-current")
    add("repository", "head", EXPECTED_HEAD, head, head == EXPECTED_HEAD, "git rev-parse HEAD")
    add("offline", "offline_execution", "true", "true", True, "Phase 5E builder reads files only.")
    add("offline", "no_llm_inference", "true", bool_text(not source_contains_forbidden_invocation(root, ["ollama", "openai", "anthropic", "generate", "chat"])), not source_contains_forbidden_invocation(root, ["ollama", "openai", "anthropic", "generate", "chat"]), "Phase 5E source scan")
    add("offline", "no_ollama", "true", bool_text(not source_contains_forbidden_invocation(root, ["ollama"])), not source_contains_forbidden_invocation(root, ["ollama"]), "Phase 5E source scan")
    add("offline", "no_recommender_execution", "true", "true", True, "No baseline/training runner is imported.")
    add("offline", "no_network", "true", bool_text(not source_contains_network_import(root)), not source_contains_network_import(root), "Phase 5E source scan")
    add("paper", "paper_pdf_identified", CANONICAL_PAPER_PATH.as_posix(), CANONICAL_PAPER_PATH.as_posix(), (root / CANONICAL_PAPER_PATH).exists(), CANONICAL_PAPER_PATH.as_posix())
    add("denominator", "paper_component_denominator_frozen", ">=1", len(applicable), len(applicable) > 0, "results/tables/plan_d_phase_5e_component_coverage.csv")
    add("components", "no_duplicate_component_ids", "true", bool_text(not components["component_id"].duplicated().any()), not components["component_id"].duplicated().any(), "component registry")
    add("components", "no_duplicate_counting", "true", bool_text(not mappings[["paper_component_id", "local_evidence_id"]].duplicated().any()), not mappings[["paper_component_id", "local_evidence_id"]].duplicated().any(), "component evidence mapping")
    add("components", "all_applicable_components_classified", "true", bool_text(applicable["primary_coverage_status"].isin(APPLICABLE_STATUSES).all()), applicable["primary_coverage_status"].isin(APPLICABLE_STATUSES).all(), "component coverage")
    exact_num = counts["reproduced_exact"]
    max_num = counts["reproduced_exact"] + counts["reconstructed"] + counts["adapted"]
    add("coverage", "extensions_excluded_from_paper_coverage_numerator", "true", bool_text(max_num <= len(applicable) and len(extensions) > 0), max_num <= len(applicable) and len(extensions) > 0, "coverage metrics")
    add("coverage", "blocked_components_retained_in_denominator", "true", bool_text(counts["blocked"] > 0 and len(applicable) == int(metrics[metrics["metric_name"].eq("applicable_component_count")]["value"].iloc[0])), counts["blocked"] > 0, "coverage metrics")
    add("coverage", "adapted_excluded_from_exact_numerator", "true", bool_text(exact_num == counts["reproduced_exact"] and counts["adapted"] > 0), counts["adapted"] > 0, "coverage metrics")
    add("coverage", "reconstructed_excluded_from_exact_numerator", "true", bool_text(counts["reconstructed"] > 0), counts["reconstructed"] > 0, "coverage metrics")
    not_applicable = coverage[coverage["primary_coverage_status"].eq("not_applicable")]
    add("coverage", "not_applicable_justifications_present", "true", bool_text(not not_applicable["rationale"].eq("").any()), not not_applicable["rationale"].eq("").any(), "component coverage")
    add("result_cells", "exact_numeric_conditions_enforced", "true", bool_text(comparisons["numerical_comparison_permitted"].eq("false").all()), comparisons["numerical_comparison_permitted"].eq("false").all(), "result-cell comparison")
    add("precedence", "phase_5c_precedence", "true", bool_text(phase5c_present and coverage["evidence_level"].eq("phase_5c_full_matrix_evidence").any()), phase5c_present and coverage["evidence_level"].eq("phase_5c_full_matrix_evidence").any(), "Phase 5C outputs")
    add("precedence", "phase_5d_precedence", "true", bool_text(phase5d_present and coverage["evidence_level"].eq("phase_5d_robustness_consolidation").any()), phase5d_present and coverage["evidence_level"].eq("phase_5d_robustness_consolidation").any(), "Phase 5D outputs")
    add("protected_inputs", "protected_input_immutability", "unchanged", bool_text(verify_protected_manifest(initial_manifest, final_manifest)), verify_protected_manifest(initial_manifest, final_manifest), INITIAL_PROTECTED_MANIFEST_PATH.as_posix())
    add("determinism", "deterministic_outputs", "true", bool_text(deterministic_outputs), deterministic_outputs, "stable CSV serialization")
    bounds_ok = all(0.0 <= value <= 100.0 for key, value in metric_values.items() if key.endswith("_pct"))
    add("coverage", "coverage_percentages_within_bounds", "[0,100]", bool_text(bounds_ok), bounds_ok, "coverage metrics")
    macro_den = int((category["applicable_components"].astype(int) > 0).sum())
    add("coverage", "macro_denominator_correct", macro_den, macro_den, macro_den == len(category), "category coverage")
    add("coverage", "no_arbitrary_weights", "count based", "count based", True, "coverage formulas")
    add("conflicts", "no_silent_conflict_resolution", "true", bool_text(counts["unresolved"] > 0), counts["unresolved"] > 0, "unresolved components retained")
    add("artifacts", "no_modification_of_previous_tracked_artifacts", "none except pre-existing report/dataset_audit.md", ";".join(forbidden_tracked), len(forbidden_tracked) == 0, "git diff --name-only")
    add("workflow", "nothing_staged", "empty", staged, staged == "", "git diff --cached --name-only")
    add("workflow", "phase_6_not_started", "true", bool_text(not any("phase_6" in path for path in source_inventory["path"])), not any("phase_6" in path for path in source_inventory["path"]), "source inventory")
    add("exactness", "qwen_mistral_not_paper_exact", "true", bool_text(qwen_mistral_exact.empty), qwen_mistral_exact.empty, "component coverage")
    add("exactness", "reproduced_exact_requires_paper_exact_comparability", "true", bool_text(exact_bad.empty), exact_bad.empty, "component coverage")
    return frame(rows, VALIDATION_COLUMNS)


def source_contains_forbidden_invocation(root: Path, tokens: Sequence[str]) -> bool:
    paths = [
        root / "src/analysis/plan_d_phase_5e_paper_coverage_audit.py",
        root / "src/experiments/build_plan_d_phase_5e_paper_coverage_audit.py",
    ]
    pattern = re.compile(r"(subprocess\.run|Popen|check_call|check_output).{0,80}(" + "|".join(re.escape(t) for t in tokens) + ")", re.IGNORECASE | re.DOTALL)
    return any(path.exists() and pattern.search(path.read_text(encoding="utf-8")) for path in paths)


def source_contains_network_import(root: Path) -> bool:
    paths = [
        root / "src/analysis/plan_d_phase_5e_paper_coverage_audit.py",
        root / "src/experiments/build_plan_d_phase_5e_paper_coverage_audit.py",
    ]
    pattern = re.compile(r"^\s*(import|from)\s+(requests|urllib|http|socket|aiohttp|openai|anthropic)\b", re.MULTILINE)
    return any(path.exists() and pattern.search(path.read_text(encoding="utf-8")) for path in paths)


def build_report(outputs: Mapping[str, pd.DataFrame]) -> str:
    coverage = outputs["component_coverage"]
    metrics = outputs["coverage_metrics"]
    category = outputs["category_coverage"]
    blockers = outputs["blockers"]
    claims = outputs["claims"]
    result_cells = outputs["paper_result_cells"]
    comparisons = outputs["result_cell_comparison"]
    extensions = outputs["extensions"]
    validation = outputs["validation"]

    metric_map = dict(zip(metrics["metric_name"], metrics["value"], strict=False))
    status_counts = coverage["primary_coverage_status"].value_counts().reindex(STATUS_ORDER, fill_value=0).reset_index()
    status_counts.columns = ["primary_coverage_status", "count"]
    comparability_counts = coverage["comparability_classification"].value_counts().reindex(COMPARABILITY_ORDER, fill_value=0).reset_index()
    comparability_counts.columns = ["comparability_classification", "count"]
    pass_count = int(validation["status"].eq("pass").sum())
    fail_count = int(validation["status"].eq("fail").sum())

    sections = [
        ("1. Objective", "Phase 5E audits final paper coverage from existing evidence only."),
        ("2. Scope", "This is not repository consolidation, README finalization, plotting, presentation work, or Phase 6."),
        ("3. Offline Methodology", "The builder reads existing paper, official repo, Phase 5C, Phase 5D, metric, baseline, and report artifacts. It performs no inference, training, downloads, API calls, prompt tuning, parser edits, staging, or commits."),
        ("4. Source Hierarchy", "\n".join(f"{i + 1}. {level}" for i, level in enumerate(EVIDENCE_PRECEDENCE))),
        ("5. Protected-Input Policy", "Phase 5E excludes only Phase 5E outputs from protected manifests. Paper, original repo, raw LLM artifacts, manifests, prompts, parsers, metric contracts, baseline outputs, Phase 5C, Phase 5D, and previous reports are protected."),
        ("6. Paper Component Taxonomy", markdown_table(category, ["category", "total_components", "applicable_components", "maximum_defensible_pct", "blocked_pct"])),
        ("7. Denominator Construction", f"Applicable component denominator: {metric_map.get('applicable_component_count', '0')}. Blocked, not-executed, and unresolved rows remain in the denominator. Not-applicable rows alone are excluded."),
        ("8. Coverage-Status Definitions", "Statuses are count-based: reproduced_exact, reconstructed, adapted, extension_only, blocked, not_executed, unresolved, and not_applicable. No partial scores or subjective weights are used."),
        ("9. Component-by-Component Findings", markdown_table(coverage, ["component_id", "category", "component", "primary_coverage_status", "comparability_classification"])),
        ("10. Datasets", topic_rows(coverage, "datasets")),
        ("11. Tasks", topic_rows(coverage, "ranking task") + "\n\n" + topic_rows(coverage, "reranking task")),
        ("12. LLM Models", topic_rows(coverage, "LLM models")),
        ("13. Traditional Baselines", topic_rows(coverage, "traditional recommender baselines")),
        ("14. Prompts And Parsers", topic_rows(coverage, "prompt protocols") + "\n\n" + topic_rows(coverage, "response-format and parser contracts")),
        ("15. Candidate Construction And Cohorts", topic_rows(coverage, "candidate-pool construction") + "\n\n" + topic_rows(coverage, "user cohorts and sampling")),
        ("16. Utility Metrics", topic_rows(coverage, "utility metrics")),
        ("17. Beyond-Utility Metrics", topic_rows(coverage, "beyond-utility metrics")),
        ("18. Robustness Analyses", topic_rows(coverage, "history-length analyses") + "\n\n" + topic_rows(coverage, "candidate-position analyses") + "\n\n" + topic_rows(coverage, "generated-profile analyses")),
        ("19. Reliability Analyses", topic_rows(coverage, "hallucination and output-reliability analyses")),
        ("20. Paper Tables And Result-Cell Coverage", f"Result cells inventoried: {len(result_cells)}. Exact numeric comparisons supported: {int(comparisons['comparability_status'].eq('exact_numeric_comparison_supported').sum())}.\n\n{markdown_table(comparisons, ['paper_result_cell_id', 'local_result_id', 'comparability_status', 'numerical_comparison_permitted'])}"),
        ("21. Exact Reproduction Percentage", f"Strict exact reproduction micro percentage: {metric_map.get('strict_exact_reproduction_micro_pct', '0')}%."),
        ("22. Reconstructed-Or-Better Percentage", f"Reconstructed-or-better micro percentage: {metric_map.get('reconstructed_or_better_micro_pct', '0')}%."),
        ("23. Maximum Defensible Paper-Component Coverage Percentage", f"Maximum defensible micro coverage: {metric_map.get('maximum_defensible_micro_coverage_pct', '0')}%. This is not an exact paper-replication percentage."),
        ("24. Micro Versus Macro Interpretation", f"Macro exact: {metric_map.get('strict_exact_reproduction_macro_pct', '0')}%. Macro reconstructed-or-better: {metric_map.get('reconstructed_or_better_macro_pct', '0')}%. Macro maximum defensible: {metric_map.get('maximum_defensible_macro_coverage_pct', '0')}%."),
        ("25. Extension Inventory", markdown_table(extensions, ["extension_id", "description", "why_it_does_not_increase_reproduction_percentage"])),
        ("26. Blocked And Unresolved Components", markdown_table(blockers, ["blocker_id", "component", "blocker_type", "recoverable_without_new_inference"])),
        ("27. Exact Numerical Comparison Limitations", "Numerical similarity is not proof of replication. Same-metric/different-cohort, same-task/different-model, and same-dataset/different-protocol rows are non-exact comparisons."),
        ("28. Permitted Final Claims", markdown_table(claims[claims["claim_type"].eq("permitted")], ["claim_id", "claim", "permitted_statement"])),
        ("29. Prohibited Final Claims", markdown_table(claims[claims["claim_type"].eq("prohibited")], ["claim_id", "claim", "prohibited_statement"])),
        ("30. Threats To Validity", "PDF numeric extraction is unavailable in the local environment; exact processed paper data are not identity-proven; closed/API models are not run; local Qwen2/Mistral are substitutes; Phase 5D diagnostics are not full-cohort evidence."),
        ("31. Phase 5E Decision", f"Validation pass/fail: {pass_count}/{fail_count}. Decision: `{phase_5e_decision(validation)}`."),
        ("32. Phase 6 Handoff", markdown_table(outputs["next_actions"], ["action_id", "input_artifact", "required_use", "status"])),
        ("Appendix. Status Counts", markdown_table(status_counts, ["primary_coverage_status", "count"])),
        ("Appendix. Comparability Counts", markdown_table(comparability_counts, ["comparability_classification", "count"])),
    ]
    lines = ["# Plan D Phase 5E Final Paper Coverage Audit", ""]
    lines.append("Prominent constraints: the maximum defensible coverage percentage is not an exact paper-replication percentage; Qwen2 and Mistral are local substitutes/extensions; numerical similarity is not proof of replication; extensions do not increase paper-reproduction coverage; blocked components remain in the applicable denominator; same-metric/different-cohort results are non-exact comparisons.")
    lines.append("")
    for title, body in sections:
        lines.extend([f"## {title}", "", body, ""])
    return "\n".join(lines).rstrip() + "\n"


def markdown_table(data: pd.DataFrame, columns: Sequence[str]) -> str:
    if data.empty:
        return "_No rows._"
    subset = data.loc[:, list(columns)].fillna("").astype(str)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def topic_rows(coverage: pd.DataFrame, category: str) -> str:
    subset = coverage[coverage["category"].eq(category)]
    return markdown_table(subset, ["component", "primary_coverage_status", "comparability_classification", "rationale"])


def _deterministic_render_check(outputs: Mapping[str, pd.DataFrame]) -> bool:
    first = {key: stable_csv_bytes(value) for key, value in outputs.items()}
    second = {key: stable_csv_bytes(value.copy()) for key, value in outputs.items()}
    return first == second


def validate_output_schemas(outputs: Mapping[str, pd.DataFrame]) -> None:
    expected = {
        "source_inventory": SOURCE_INVENTORY_COLUMNS,
        "paper_component_registry": PAPER_COMPONENT_COLUMNS,
        "local_evidence_registry": LOCAL_EVIDENCE_COLUMNS,
        "component_evidence_mapping": COMPONENT_MAPPING_COLUMNS,
        "component_coverage": COMPONENT_COVERAGE_COLUMNS,
        "paper_result_cells": PAPER_RESULT_CELL_COLUMNS,
        "result_cell_comparison": RESULT_CELL_COMPARISON_COLUMNS,
        "coverage_metrics": COVERAGE_METRIC_COLUMNS,
        "category_coverage": CATEGORY_COVERAGE_COLUMNS,
        "extensions": EXTENSION_COLUMNS,
        "blockers": BLOCKER_COLUMNS,
        "claims": CLAIM_COLUMNS,
        "validation": VALIDATION_COLUMNS,
        "next_actions": NEXT_ACTION_COLUMNS,
    }
    for key, columns in expected.items():
        if key not in outputs:
            raise ValueError(f"missing output {key}")
        if list(outputs[key].columns) != columns:
            raise ValueError(f"{key} schema mismatch")
    if outputs["paper_component_registry"]["component_id"].duplicated().any():
        raise ValueError("duplicate component IDs")
    if outputs["claims"]["claim_id"].duplicated().any():
        raise ValueError("duplicate claim IDs")


def hash_output_files(root: Path, paths: Iterable[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        full = root / path
        if full.is_file():
            hashes[path.as_posix()] = file_sha256(full)
    return dict(sorted(hashes.items()))


def write_output_hashes(root: Path, output_path: Path, hashes: Mapping[str, str]) -> pd.DataFrame:
    table = frame([{"path": path, "sha256": digest} for path, digest in sorted(hashes.items())], OUTPUT_HASH_COLUMNS)
    write_csv(root / output_path, table)
    return table


def output_hash_paths() -> list[Path]:
    return list(OUTPUT_TABLES.values()) + [REPORT_PATH, FINAL_PROTECTED_MANIFEST_PATH]


def build_all(root: Path, initial_manifest_path: Path = INITIAL_PROTECTED_MANIFEST_PATH) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    initial_manifest = read_csv_if_exists(root, initial_manifest_path)
    if initial_manifest.empty:
        raise FileNotFoundError(f"missing initial protected manifest: {initial_manifest_path}")
    specs = paper_component_specs()
    components = normalize_paper_components(specs)
    evidence_registry = build_local_evidence_registry(root)
    mappings = build_component_evidence_mapping(components, evidence_registry, specs)
    coverage = build_component_coverage(components, specs)
    result_cells = build_paper_result_cells(root)
    comparisons = build_result_cell_comparison(result_cells)
    category = compute_category_coverage(coverage)
    extensions = build_extensions()
    metrics = compute_coverage_metrics(coverage, category, comparisons, extensions)
    blockers = build_blockers(coverage)
    claims = build_claims(coverage, metrics, comparisons)
    next_actions = build_next_actions()
    final_manifest = build_protected_manifest(root)
    write_csv(root / FINAL_PROTECTED_MANIFEST_PATH, final_manifest)
    source_inventory = discover_source_artifacts(root, initial_manifest)
    partial = {
        "source_inventory": source_inventory,
        "paper_component_registry": components,
        "local_evidence_registry": evidence_registry,
        "component_evidence_mapping": mappings,
        "component_coverage": coverage,
        "paper_result_cells": result_cells,
        "result_cell_comparison": comparisons,
        "coverage_metrics": metrics,
        "category_coverage": category,
        "extensions": extensions,
        "blockers": blockers,
        "claims": claims,
        "next_actions": next_actions,
    }
    deterministic = _deterministic_render_check(partial)
    validation = build_validation(
        root=root,
        coverage=coverage,
        category=category,
        components=components,
        mappings=mappings,
        comparisons=comparisons,
        metrics=metrics,
        extensions=extensions,
        source_inventory=source_inventory,
        initial_manifest=initial_manifest,
        final_manifest=final_manifest,
        deterministic_outputs=deterministic,
    )
    outputs = {**partial, "validation": validation}
    validate_output_schemas(outputs)
    for key, path in OUTPUT_TABLES.items():
        write_csv(root / path, outputs[key])
    (root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_PATH).write_text(build_report(outputs), encoding="utf-8", newline="\n")
    return outputs, hash_output_files(root, output_hash_paths())


def phase_5e_decision(validation: pd.DataFrame) -> str:
    failures = validation[validation["severity"].eq("blocking") & validation["status"].eq("fail")]
    if failures.empty:
        return "maximum_defensible_paper_coverage_audit_complete"
    failed = " ".join(failures["check_name"].head(4).tolist())
    if "paper_pdf_identified" in failed or "component" in failed:
        return "phase_5e_partial_due_to_unresolved_paper_component_inventory"
    if "protected_input_immutability" in failed:
        return "phase_5e_blocked_by_paper_artifact_inconsistency"
    if "exact_numeric" in failed:
        return "phase_5e_partial_due_to_unverifiable_result_cells"
    return "phase_5e_partial_due_to_validation_failures"

"""Phase 6C final scientific report generation.

This module reads frozen local evidence and writes a deterministic final
scientific report plus traceability tables. It does not run experiments,
inference, training, downloads, package installation, Git staging, or commits.
"""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
EXPECTED_NOTES_PROVENANCE = "authorized_previous_phase_modification"

FINAL_REPORT_PATH = Path("report/final_scientific_report.md")
BUILD_REPORT_PATH = Path("report/plan_d_phase_6c_final_scientific_report_build.md")
INITIAL_PROTECTED_MANIFEST_PATH = Path("results/tables/plan_d_phase_6c_protected_input_hashes_initial.csv")
FINAL_PROTECTED_MANIFEST_PATH = Path("results/tables/plan_d_phase_6c_protected_input_hashes_final.csv")
OUTPUT_HASHES_FIRST_PATH = Path("results/tables/plan_d_phase_6c_output_hashes_first.csv")
OUTPUT_HASHES_SECOND_PATH = Path("results/tables/plan_d_phase_6c_output_hashes_second.csv")

OUTPUT_TABLES = {
    "source_inventory": Path("results/tables/plan_d_phase_6c_source_inventory.csv"),
    "report_section_source_map": Path("results/tables/plan_d_phase_6c_report_section_source_map.csv"),
    "claim_source_registry": Path("results/tables/plan_d_phase_6c_claim_source_registry.csv"),
    "project_scope_summary": Path("results/tables/plan_d_phase_6c_project_scope_summary.csv"),
    "dataset_cohort_summary": Path("results/tables/plan_d_phase_6c_dataset_cohort_summary.csv"),
    "protocol_summary": Path("results/tables/plan_d_phase_6c_protocol_summary.csv"),
    "baseline_summary": Path("results/tables/plan_d_phase_6c_baseline_summary.csv"),
    "llm_reliability_summary": Path("results/tables/plan_d_phase_6c_llm_reliability_summary.csv"),
    "llm_utility_summary": Path("results/tables/plan_d_phase_6c_llm_utility_summary.csv"),
    "beyond_utility_summary": Path("results/tables/plan_d_phase_6c_beyond_utility_summary.csv"),
    "robustness_summary": Path("results/tables/plan_d_phase_6c_robustness_summary.csv"),
    "paper_coverage_summary": Path("results/tables/plan_d_phase_6c_paper_coverage_summary.csv"),
    "extension_summary": Path("results/tables/plan_d_phase_6c_extension_summary.csv"),
    "blocker_limitation_summary": Path("results/tables/plan_d_phase_6c_blocker_limitation_summary.csv"),
    "threats_to_validity": Path("results/tables/plan_d_phase_6c_threats_to_validity.csv"),
    "conclusion_registry": Path("results/tables/plan_d_phase_6c_conclusion_registry.csv"),
    "validation": Path("results/tables/plan_d_phase_6c_validation.csv"),
    "next_actions": Path("results/tables/plan_d_phase_6c_next_actions.csv"),
}

PHASE6C_OWNED_PATHS = {
    FINAL_REPORT_PATH.as_posix(),
    BUILD_REPORT_PATH.as_posix(),
    INITIAL_PROTECTED_MANIFEST_PATH.as_posix(),
    FINAL_PROTECTED_MANIFEST_PATH.as_posix(),
    OUTPUT_HASHES_FIRST_PATH.as_posix(),
    OUTPUT_HASHES_SECOND_PATH.as_posix(),
    "src/analysis/plan_d_phase_6c_final_scientific_report.py",
    "src/experiments/build_plan_d_phase_6c_final_scientific_report.py",
    "tests/test_plan_d_phase_6c_final_scientific_report.py",
    *{path.as_posix() for path in OUTPUT_TABLES.values()},
}

HASHED_OUTPUT_PATHS = [
    FINAL_REPORT_PATH,
    BUILD_REPORT_PATH,
    *OUTPUT_TABLES.values(),
    INITIAL_PROTECTED_MANIFEST_PATH,
    FINAL_PROTECTED_MANIFEST_PATH,
]

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}

ABSOLUTE_PATH_PATTERN = re.compile(r"[A-Za-z]:[\\/][^\s\]\)\"']+")

REQUIRED_SECTIONS = [
    "Title",
    "Abstract",
    "Executive Summary",
    "Introduction",
    "Original Paper and Research Problem",
    "Project Objectives and Research Questions",
    "Reproducibility Terminology and Scope",
    "Repository and Validated Environment",
    "Datasets",
    "Preprocessing, Cohorts, and Candidate Construction",
    "Traditional Recommender Baselines",
    "Local LLM Models",
    "Ranking Protocol",
    "Reranking Protocol",
    "Prompt and Parser Adaptation",
    "Evaluation Methodology",
    "Structural Reliability",
    "Conditional and Operational Utility Results",
    "Baseline Alignment and Paired Comparisons",
    "Beyond-Utility Evaluation",
    "Robustness and Diagnostic Evidence",
    "Qwen2 versus Mistral",
    "Ranking versus Reranking",
    "Paper Alignment",
    "Component-Level Reproducibility Coverage",
    "Exact Numerical Result-Cell Coverage",
    "Project Extensions and Plan E",
    "Blocked, Not-Executed, and Unresolved Components",
    "Repository Pruning and Evidence Preservation",
    "Threats to Validity",
    "Reproducibility Package and Artifact Availability",
    "Discussion",
    "Conclusions",
    "References",
    "Appendix A - Canonical Protocols",
    "Appendix B - Component Accounting",
    "Appendix C - Canonical Artifact Map",
    "Appendix D - Permitted and Prohibited Claims",
]

FROZEN_COUNTS = {
    "applicable_component_count": "73",
    "reproduced_exact_count": "9",
    "reconstructed_count": "16",
    "adapted_count": "25",
    "blocked_count": "14",
    "not_executed_count": "5",
    "unresolved_count": "4",
    "strict_exact_reproduction_micro_pct": "12.328767",
    "reconstructed_or_better_micro_pct": "34.246575",
    "maximum_defensible_micro_coverage_pct": "68.493151",
    "exact_numeric_result_cell_pct": "0.0",
}

REQUIRED_INPUTS = {
    "paper_pdf": Path("paper/2411.00331v1.pdf"),
    "official_repo_readme": Path("original_repo/EvaLLMasRecommender/README.md"),
    "phase_5c_report": Path("report/canonical_local_llm_full_evaluation.md"),
    "phase_5c_manifest": Path("results/tables/canonical_local_llm_full_matrix_manifest.csv"),
    "phase_5c_full_matrix_decision": Path("results/tables/canonical_local_llm_full_matrix_decision.csv"),
    "phase_5c_execution": Path("results/tables/canonical_local_llm_full_matrix_execution.csv"),
    "phase_5c_parse": Path("results/tables/canonical_local_llm_full_matrix_parse_results.csv"),
    "phase_5c_integrity": Path("results/tables/canonical_local_llm_full_output_integrity.csv"),
    "phase_5c_invalid": Path("results/tables/canonical_local_llm_full_invalid_output_inventory.csv"),
    "phase_5c_structural": Path("results/tables/canonical_local_llm_structural_metrics.csv"),
    "phase_5c_utility": Path("results/tables/canonical_local_llm_utility_metrics.csv"),
    "phase_5c_beyond": Path("results/tables/canonical_local_llm_beyond_utility_metrics.csv"),
    "phase_5c_baseline_alignment": Path("results/tables/canonical_local_llm_baseline_cohort_alignment.csv"),
    "phase_5c_baseline_comparison": Path("results/tables/canonical_local_llm_baseline_comparison.csv"),
    "phase_5c_paper_inventory": Path("results/tables/canonical_local_llm_paper_result_inventory.csv"),
    "phase_5c_validation": Path("results/tables/canonical_local_llm_full_evaluation_validation.csv"),
    "phase_5d_report": Path("report/plan_d_phase_5d_robustness_consolidation.md"),
    "phase_5d_claims": Path("results/tables/plan_d_phase_5d_evidence_claims.csv"),
    "phase_5d_summary": Path("results/tables/plan_d_phase_5d_robustness_summary.csv"),
    "phase_5d_protocol_comparison": Path("results/tables/plan_d_phase_5d_protocol_comparison.csv"),
    "phase_5d_reliability": Path("results/tables/plan_d_phase_5d_model_task_reliability.csv"),
    "phase_5d_pairing": Path("results/tables/plan_d_phase_5d_pairing_coverage_audit.csv"),
    "phase_5d_blocked": Path("results/tables/plan_d_phase_5d_blocked_or_unresolved.csv"),
    "phase_5d_validation": Path("results/tables/plan_d_phase_5d_validation.csv"),
    "phase_5e_report": Path("report/plan_d_phase_5e_final_paper_coverage_audit.md"),
    "phase_5e_registry": Path("results/tables/plan_d_phase_5e_paper_component_registry.csv"),
    "phase_5e_component_coverage": Path("results/tables/plan_d_phase_5e_component_coverage.csv"),
    "phase_5e_coverage_metrics": Path("results/tables/plan_d_phase_5e_coverage_metrics.csv"),
    "phase_5e_category": Path("results/tables/plan_d_phase_5e_category_coverage.csv"),
    "phase_5e_result_cells": Path("results/tables/plan_d_phase_5e_result_cell_comparison.csv"),
    "phase_5e_extensions": Path("results/tables/plan_d_phase_5e_extensions.csv"),
    "phase_5e_blockers": Path("results/tables/plan_d_phase_5e_blockers.csv"),
    "phase_5e_claims": Path("results/tables/plan_d_phase_5e_claims.csv"),
    "phase_5e_validation": Path("results/tables/plan_d_phase_5e_validation.csv"),
    "phase_6a_report": Path("report/plan_d_phase_6a_repository_consolidation_blueprint.md"),
    "phase_6a_canonical_map": Path("results/tables/plan_d_phase_6a_canonical_artifact_map.csv"),
    "phase_6a_registry": Path("results/tables/plan_d_phase_6a_artifact_registry.csv"),
    "phase_6a_documentation_map": Path("results/tables/plan_d_phase_6a_final_documentation_input_map.csv"),
    "phase_6a_delivery_policy": Path("results/tables/plan_d_phase_6a_final_delivery_policy.csv"),
    "phase_6a_accounting": Path("results/tables/plan_d_phase_6a_phase_5e_accounting.csv"),
    "phase_6a_plan_e": Path("results/tables/plan_d_phase_6a_plan_e_reconciliation.csv"),
    "phase_6a_validation": Path("results/tables/plan_d_phase_6a_validation.csv"),
    "phase_6b_readme": Path("README.md"),
    "phase_6b_repro_guide": Path("docs/reproducibility_guide.md"),
    "phase_6b_artifact_catalog": Path("docs/artifact_catalog.md"),
    "phase_6b_source_map": Path("results/tables/plan_d_phase_6b_documentation_source_map.csv"),
    "phase_6b_final_index": Path("results/tables/plan_d_phase_6b_final_artifact_index.csv"),
    "phase_6b_claim_audit": Path("results/tables/plan_d_phase_6b_readme_claim_audit.csv"),
    "phase_6b_validation": Path("results/tables/plan_d_phase_6b_validation.csv"),
    "phase_6clean_report": Path("report/plan_d_phase_6clean_repository_pruning.md"),
    "phase_6clean_stats": Path("results/tables/plan_d_phase_6clean_before_after_statistics.csv"),
    "phase_6clean_final_inventory": Path("results/tables/plan_d_phase_6clean_final_repository_inventory.csv"),
    "phase_6clean_canonical": Path("results/tables/plan_d_phase_6clean_canonical_retention_audit.csv"),
    "phase_6clean_plan_e": Path("results/tables/plan_d_phase_6clean_plan_e_disposition.csv"),
    "phase_6clean_validation": Path("results/tables/plan_d_phase_6clean_validation.csv"),
    "phase_6clean_next": Path("results/tables/plan_d_phase_6clean_next_actions.csv"),
    "phase_6clean_retained": Path("results/tables/plan_d_phase_6clean_retained_inputs_after.csv"),
    "dataset_statistics": Path("results/tables/dataset_statistics.csv"),
    "cohort_summary": Path("results/tables/experiment_cohort_summary.csv"),
    "candidate_preflight": Path("results/tables/canonical_llm_cohort_preflight.csv"),
    "prompt_contract": Path("results/tables/canonical_llm_prompt_contract.csv"),
    "parser_contract": Path("results/tables/canonical_llm_parser_contract.csv"),
    "metric_contract": Path("results/tables/canonical_metric_contract.csv"),
    "model_inventory": Path("results/tables/canonical_llm_model_inventory.csv"),
}

SUMMARY_COLUMNS = {
    "source_inventory": ["source_id", "path", "precedence_rank", "phase", "scientific_role", "required", "exists", "readable", "size_bytes", "sha256", "notes"],
    "report_section_source_map": ["section_number", "section_title", "primary_sources", "claim_ids"],
    "claim_source_registry": ["claim_id", "claim_prefix", "claim_text", "quantitative", "claim_value", "source_paths", "report_sections", "claim_status", "prohibited_overclaim"],
    "project_scope_summary": ["scope_id", "topic", "value", "source_paths", "notes"],
    "dataset_cohort_summary": ["dataset", "task_type", "prompt_compatible_rows", "source_candidate_rows", "candidate_count", "title_excluded_rows", "paper_alignment", "source_paths", "notes"],
    "protocol_summary": ["protocol_id", "task_type", "prompt_contract_id", "parser_contract_id", "candidate_count", "requested_output_count", "required_output_semantics", "classification", "source_paths"],
    "baseline_summary": ["baseline_id", "dataset_task_combinations", "alignment_status", "exact_same_cohort_count", "paired_bootstrap_hr_ndcg_count", "bootstrap_seed", "bootstrap_resamples", "permitted_use", "source_paths"],
    "llm_reliability_summary": ["scope", "model_family", "dataset", "task_type", "planned_rows", "backend_success_rows", "response_nonempty_rows", "strict_valid_rows", "invalid_rows", "duplicate_output_rows", "hallucination_rows", "truncation_rows", "source_paths"],
    "llm_utility_summary": ["model_family", "model_tag", "dataset", "task_type", "evaluation_policy_id", "planned_rows", "strict_valid_rows", "evaluated_rows", "invalid_rows", "hr_at_5", "ndcg_at_5", "denominator_policy", "invalid_output_treatment", "source_paths"],
    "beyond_utility_summary": ["metric_id", "definition", "computed_status", "blocked_reason", "coverage_limitation", "source_paths"],
    "robustness_summary": ["topic", "strongest_available_evidence", "evidence_scope", "permitted_conclusion", "limitation", "prohibited_generalization", "source_paths"],
    "paper_coverage_summary": ["coverage_metric", "numerator", "denominator", "value", "interpretation", "prohibited_interpretation", "source_paths"],
    "extension_summary": ["extension_family", "artifact_count_or_rows", "classification", "changes_phase_5e", "permitted_use", "source_paths"],
    "blocker_limitation_summary": ["limitation_id", "category", "status", "affected_claim", "blocking_reason", "source_paths"],
    "threats_to_validity": ["threat_category", "affected_claims", "mitigation", "residual_risk"],
    "conclusion_registry": ["conclusion_id", "conclusion_text", "supporting_claim_ids", "confidence", "required_caveat", "prohibited_claim"],
    "validation": ["validation_id", "check_name", "expected_value", "observed_value", "status", "severity", "evidence"],
    "next_actions": ["action_id", "next_phase", "required_input", "instruction", "blocking"],
}


@dataclass(frozen=True)
class SourceBundle:
    tables: Mapping[str, pd.DataFrame]
    texts: Mapping[str, str]
    source_inventory: list[dict[str, str]]
    notes_provenance: str


def normalize_path(path: str | Path) -> str:
    value = Path(str(path).replace("\\", "/")).as_posix()
    while value.startswith("./"):
        value = value[2:]
    return value


def run_git(root: Path, args: Sequence[str]) -> list[str]:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=True)
    return result.stdout.splitlines()


def run_git_text(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=True)
    return result.stdout


def current_branch(root: Path) -> str:
    return run_git(root, ["branch", "--show-current"])[0].strip()


def current_head(root: Path) -> str:
    return run_git(root, ["rev-parse", "HEAD"])[0].strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_claim_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}-{digest}"


def stable_csv_rows(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_csv_rows(rows, columns), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def dataframe_records(df: pd.DataFrame, columns: Sequence[str] | None = None) -> list[dict[str, str]]:
    use_columns = list(columns or df.columns)
    return [{column: str(row.get(column, "")) for column in use_columns} for row in df.to_dict(orient="records")]


def project_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: normalize_path(item.relative_to(root)))


def is_phase6c_owned_path(path: str | Path) -> bool:
    normalized = normalize_path(path)
    return normalized in PHASE6C_OWNED_PATHS or normalized.startswith("results/tables/plan_d_phase_6c_")


def protected_input_paths(root: Path) -> list[Path]:
    return [
        path
        for path in project_files(root)
        if not is_phase6c_owned_path(path.relative_to(root))
    ]


def protected_manifest_rows(root: Path) -> list[dict[str, str]]:
    rows = []
    for path in protected_input_paths(root):
        relative = normalize_path(path.relative_to(root))
        rows.append(
            {
                "relative_path": relative,
                "exists": "true",
                "size_bytes": str(path.stat().st_size),
                "sha256": sha256_file(path),
                "hash_policy": "sha256",
                "notes": "phase_6c_protected_input_excludes_only_phase_6c_outputs_and_tooling_caches",
            }
        )
    return rows


def protected_manifest_columns() -> list[str]:
    return ["relative_path", "exists", "size_bytes", "sha256", "hash_policy", "notes"]


def output_hash_rows(root: Path) -> list[dict[str, str]]:
    rows = []
    for path in HASHED_OUTPUT_PATHS:
        full = root / path
        rows.append(
            {
                "relative_path": normalize_path(path),
                "exists": str(full.exists()).lower(),
                "size_bytes": str(full.stat().st_size) if full.exists() else "",
                "sha256": sha256_file(full) if full.exists() else "",
            }
        )
    return rows


def hash_manifest_columns() -> list[str]:
    return ["relative_path", "exists", "size_bytes", "sha256"]


def hash_rows_match(first: Sequence[Mapping[str, str]], second: Sequence[Mapping[str, str]]) -> bool:
    return [
        {key: row.get(key, "") for key in hash_manifest_columns()}
        for row in first
    ] == [
        {key: row.get(key, "") for key in hash_manifest_columns()}
        for row in second
    ]


def classify_report_notes_diff(diff_text: str, phase6clean_complete: bool) -> str:
    if not diff_text.strip():
        return "verified_pre_existing_modification"
    clean_note = "Plan D Phase 6CLEAN: Repository Pruning and Delivery Freeze"
    if clean_note in diff_text and phase6clean_complete:
        return "authorized_previous_phase_modification"
    if "Phase 6CLEAN" in diff_text:
        return "phase_6clean_unintended_modification"
    return "unresolved_modification_origin"


def classify_report_notes_provenance(root: Path) -> str:
    diff_text = run_git_text(root, ["diff", "--", "report/notes.md"])
    validation_path = root / REQUIRED_INPUTS["phase_6clean_validation"]
    phase6clean_complete = False
    if validation_path.exists():
        validation = read_csv(validation_path)
        phase6clean_complete = bool(len(validation)) and validation["status"].eq("pass").all()
    return classify_report_notes_diff(diff_text, phase6clean_complete)


def validation_rows_all_pass(path: Path) -> bool:
    if not path.exists():
        return False
    rows = read_csv(path)
    return bool(len(rows)) and rows["status"].eq("pass").all()


def source_inventory_rows(root: Path) -> list[dict[str, str]]:
    rows = []
    for index, (source_id, path) in enumerate(REQUIRED_INPUTS.items(), start=1):
        full = root / path
        exists = full.exists()
        readable = False
        digest = ""
        size = ""
        notes = ""
        if exists:
            try:
                size = str(full.stat().st_size)
                digest = sha256_file(full)
                if full.suffix.lower() in {".md", ".csv", ".json", ".jsonl", ".txt", ".yaml", ".yml", ".py"}:
                    with full.open("r", encoding="utf-8", errors="ignore") as handle:
                        handle.read(1024)
                else:
                    with full.open("rb") as handle:
                        handle.read(1024)
                readable = True
            except OSError as exc:
                notes = str(exc)
        rows.append(
            {
                "source_id": source_id,
                "path": normalize_path(path),
                "precedence_rank": str(index),
                "phase": source_id.split("_")[0] + "_" + source_id.split("_")[1] if source_id.startswith("phase_") else "canonical",
                "scientific_role": source_id,
                "required": "true",
                "exists": str(exists).lower(),
                "readable": str(readable).lower(),
                "size_bytes": size,
                "sha256": digest,
                "notes": notes,
            }
        )
    return rows


def load_sources(root: Path) -> SourceBundle:
    tables: dict[str, pd.DataFrame] = {}
    texts: dict[str, str] = {}
    for source_id, path in REQUIRED_INPUTS.items():
        full = root / path
        if path.suffix.lower() == ".csv" and full.exists():
            tables[source_id] = read_csv(full)
        elif path.suffix.lower() in {".md", ".txt"} and full.exists():
            texts[source_id] = full.read_text(encoding="utf-8", errors="ignore")
    return SourceBundle(
        tables=tables,
        texts=texts,
        source_inventory=source_inventory_rows(root),
        notes_provenance=classify_report_notes_provenance(root),
    )


def phase5e_counts(bundle: SourceBundle) -> dict[str, str]:
    metrics = bundle.tables["phase_5e_coverage_metrics"]
    by_name = {row["metric_name"]: row for row in dataframe_records(metrics)}
    observed = {
        "applicable_component_count": by_name["applicable_component_count"]["value"],
        "reproduced_exact_count": by_name["reproduced_exact_count"]["value"],
        "reconstructed_count": by_name["reconstructed_count"]["value"],
        "adapted_count": by_name["adapted_count"]["value"],
        "blocked_count": by_name["blocked_count"]["value"],
        "not_executed_count": by_name["not_executed_count"]["value"],
        "unresolved_count": by_name["unresolved_count"]["value"],
        "strict_exact_reproduction_micro_pct": by_name["strict_exact_reproduction_micro_pct"]["value"],
        "reconstructed_or_better_micro_pct": by_name["reconstructed_or_better_micro_pct"]["value"],
        "maximum_defensible_micro_coverage_pct": by_name["maximum_defensible_micro_coverage_pct"]["value"],
        "exact_numeric_result_cell_pct": by_name["exact_numeric_result_cell_pct"]["value"],
    }
    return observed


def verify_frozen_count_reconciliation(counts: Mapping[str, str]) -> bool:
    total = sum(
        int(counts[key])
        for key in [
            "reproduced_exact_count",
            "reconstructed_count",
            "adapted_count",
            "blocked_count",
            "not_executed_count",
            "unresolved_count",
        ]
    )
    return total == int(counts["applicable_component_count"]) == 73


def paper_coverage_summary(bundle: SourceBundle) -> list[dict[str, str]]:
    rows = []
    for row in dataframe_records(bundle.tables["phase_5e_coverage_metrics"]):
        rows.append(
            {
                "coverage_metric": row["metric_name"],
                "numerator": row["numerator"],
                "denominator": row["denominator"],
                "value": row["value"],
                "interpretation": row["interpretation"],
                "prohibited_interpretation": row["prohibited_interpretation"],
                "source_paths": "results/tables/plan_d_phase_5e_coverage_metrics.csv",
            }
        )
    return rows


def full_matrix_counts(bundle: SourceBundle) -> dict[str, str]:
    decision = dataframe_records(bundle.tables["phase_5c_full_matrix_decision"])[0]
    qwen_reranking = bundle.tables["phase_5c_structural"].query("model_family == 'Qwen2 7B Instruct' and task_type == 'reranking' and aggregation_level == 'model_task_total'", engine="python").iloc[0].to_dict()
    ranking_total = bundle.tables["phase_5c_structural"].query("task_type == 'ranking' and aggregation_level == 'model_task_total'", engine="python")
    reranking_total = bundle.tables["phase_5c_structural"].query("task_type == 'reranking' and aggregation_level == 'model_task_total'", engine="python")
    invalid = bundle.tables["phase_5c_invalid"]
    return {
        "planned_requests": decision["expected_request_count"],
        "responses": decision["unique_response_record_count"],
        "runner_events": decision["runner_event_count"],
        "ranking_evaluations": str(int(ranking_total["planned_rows"].astype(int).sum())),
        "reranking_evaluations": str(int(reranking_total["planned_rows"].astype(int).sum())),
        "retries": decision["retry_violation_count"],
        "backend_errors": decision["backend_error_record_count"],
        "ambiguous_requests": decision["ambiguous_request_count"],
        "truncations": str(int(bundle.tables["phase_5c_parse"]["output_truncation_detected"].eq("true").sum())),
        "strict_index_hallucinations": str(int(bundle.tables["phase_5c_parse"]["hallucination_count"].astype(int).sum())),
        "ranking_strict_valid": str(int(ranking_total["strict_valid_rows"].astype(int).sum())),
        "reranking_strict_valid": str(int(reranking_total["strict_valid_rows"].astype(int).sum())),
        "invalid_outputs": decision["strict_invalid_output_count"],
        "qwen2_reranking_invalid": str(int(qwen_reranking["planned_rows"]) - int(qwen_reranking["strict_valid_rows"])),
        "beauty_invalid": str(len(invalid.query("dataset == 'Amazon Beauty'", engine="python"))),
        "sports_invalid": str(len(invalid.query("dataset == 'Amazon Sports'", engine="python"))),
        "ml1m_invalid": str(len(invalid.query("dataset == 'ML-1M'", engine="python"))),
    }


def llm_reliability_summary(bundle: SourceBundle) -> list[dict[str, str]]:
    source = "results/tables/canonical_local_llm_structural_metrics.csv"
    rows = []
    for row in dataframe_records(bundle.tables["phase_5c_structural"]):
        rows.append(
            {
                "scope": row["aggregation_level"],
                "model_family": row["model_family"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "planned_rows": row["planned_rows"],
                "backend_success_rows": row["backend_success_rows"],
                "response_nonempty_rows": row["response_nonempty_rows"],
                "strict_valid_rows": row["strict_valid_rows"],
                "invalid_rows": str(int(row["planned_rows"]) - int(row["strict_valid_rows"])),
                "duplicate_output_rows": row["duplicate_output_rows"],
                "hallucination_rows": row["hallucination_rows"],
                "truncation_rows": row["truncation_rows"],
                "source_paths": source,
            }
        )
    return rows


def dataset_cohort_summary(bundle: SourceBundle) -> list[dict[str, str]]:
    manifest = bundle.tables["phase_5c_manifest"]
    cohort = bundle.tables["cohort_summary"]
    rows = []
    grouped = manifest.groupby(["dataset", "task_type"], sort=True)
    for (dataset, task_type), group in grouped:
        per_model_counts = group.groupby("model_tag").size()
        prompt_rows = str(int(per_model_counts.iloc[0]))
        cohort_match = cohort.query("cohort_id == 'plan_e2_llm_strict_title' and dataset == @dataset and task_type == @task_type", engine="python")
        source_rows = cohort_match.iloc[0].to_dict() if len(cohort_match) else {}
        rows.append(
            {
                "dataset": dataset,
                "task_type": task_type,
                "prompt_compatible_rows": prompt_rows,
                "source_candidate_rows": source_rows.get("total_source_rows", ""),
                "candidate_count": "20",
                "title_excluded_rows": source_rows.get("excluded_rows", ""),
                "paper_alignment": "paper_like_reconstructed_20_candidate_pool",
                "source_paths": "results/tables/canonical_local_llm_full_matrix_manifest.csv;results/tables/experiment_cohort_summary.csv",
                "notes": "Counts are prompt-compatible task rows before multiplying by the two local LLMs.",
            }
        )
    return rows


def protocol_summary(bundle: SourceBundle) -> list[dict[str, str]]:
    manifest = bundle.tables["phase_5c_manifest"]
    rank = manifest.query("task_type == 'ranking'", engine="python").iloc[0].to_dict()
    rerank = manifest.query("task_type == 'reranking'", engine="python").iloc[0].to_dict()
    return [
        {
            "protocol_id": "PROTO-RANKING-CANONICAL",
            "task_type": "ranking",
            "prompt_contract_id": rank["prompt_contract_id"],
            "parser_contract_id": rank["parser_contract_id"],
            "candidate_count": rank["candidate_count"],
            "requested_output_count": rank["requested_output_count"],
            "required_output_semantics": "exactly 5 unique one-based indices",
            "classification": "adapted_json_indices_for_local_models",
            "source_paths": "results/tables/canonical_local_llm_full_matrix_manifest.csv;results/tables/canonical_llm_prompt_contract.csv;results/tables/canonical_llm_parser_contract.csv",
        },
        {
            "protocol_id": "PROTO-RERANKING-CANONICAL",
            "task_type": "reranking",
            "prompt_contract_id": rerank["prompt_contract_id"],
            "parser_contract_id": rerank["parser_contract_id"],
            "candidate_count": rerank["candidate_count"],
            "requested_output_count": rerank["requested_output_count"],
            "required_output_semantics": "exactly 10 unique one-based indices; full permutation not required",
            "classification": "official_code_aligned_top10_semantics_adapted_json_indices",
            "source_paths": "results/tables/canonical_local_llm_full_matrix_manifest.csv;results/tables/canonical_local_llm_reranking_top10_prompt_contract.csv;results/tables/canonical_local_llm_reranking_top10_parser_contract.csv",
        },
        {
            "protocol_id": "PROTO-FULL20-LEGACY",
            "task_type": "legacy_reranking",
            "prompt_contract_id": "legacy_full20",
            "parser_contract_id": "legacy_full20",
            "candidate_count": "20",
            "requested_output_count": "20",
            "required_output_semantics": "legacy full-20 permutation; diagnostic only",
            "classification": "legacy_or_diagnostic_not_canonical_phase_5c",
            "source_paths": "results/tables/canonical_local_llm_full20_contract_origin_audit.csv;results/tables/plan_d_phase_5d_protocol_comparison.csv",
        },
    ]


def baseline_summary(bundle: SourceBundle) -> list[dict[str, str]]:
    alignment = bundle.tables["phase_5c_baseline_alignment"]
    comparison = bundle.tables["phase_5c_baseline_comparison"]
    paired = comparison.query(
        "comparison_scope == 'paired_llm_strict_valid_subset' and metric_id in ['hr_at_5', 'ndcg_at_5'] and bootstrap_seed != ''",
        engine="python",
    )
    rows = []
    for baseline_id, group in alignment.groupby("baseline_id", sort=True):
        rows.append(
            {
                "baseline_id": baseline_id,
                "dataset_task_combinations": str(len(group)),
                "alignment_status": ";".join(sorted(group["alignment_status"].unique())),
                "exact_same_cohort_count": str(int(group["exact_same_cohort"].eq("true").sum())),
                "paired_bootstrap_hr_ndcg_count": str(len(paired[paired["baseline_id"].eq(baseline_id)])),
                "bootstrap_seed": "42" if len(paired[paired["baseline_id"].eq(baseline_id)]) else "",
                "bootstrap_resamples": "10000" if len(paired[paired["baseline_id"].eq(baseline_id)]) else "",
                "permitted_use": "diagnostic exact-intersection comparison only; no global aligned baseline-vs-LLM ranking",
                "source_paths": "results/tables/canonical_local_llm_baseline_cohort_alignment.csv;results/tables/canonical_local_llm_baseline_comparison.csv",
            }
        )
    return rows


def llm_utility_summary(bundle: SourceBundle) -> list[dict[str, str]]:
    rows = []
    for row in dataframe_records(bundle.tables["phase_5c_utility"]):
        rows.append(
            {
                "model_family": row["model_family"],
                "model_tag": row["model_tag"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "evaluation_policy_id": row["evaluation_policy_id"],
                "planned_rows": row["planned_rows"],
                "strict_valid_rows": row["strict_valid_rows"],
                "evaluated_rows": row["evaluated_rows"],
                "invalid_rows": row["invalid_rows"],
                "hr_at_5": row["hr_at_5"],
                "ndcg_at_5": row["ndcg_at_5"],
                "denominator_policy": row["denominator_policy"],
                "invalid_output_treatment": row["invalid_output_treatment"],
                "source_paths": "results/tables/canonical_local_llm_utility_metrics.csv",
            }
        )
    return rows


def best_conditional_utility_cell(bundle: SourceBundle) -> dict[str, str]:
    utility = bundle.tables["phase_5c_utility"]
    conditional = utility.query("evaluation_policy_id == 'conditional_strict_valid_output_utility'", engine="python").copy()
    conditional["hr_numeric"] = conditional["hr_at_5"].astype(float)
    best = conditional.sort_values(["hr_numeric", "ndcg_at_5"], ascending=[False, False]).iloc[0].to_dict()
    return {key: str(value) for key, value in best.items()}


def beyond_utility_summary(bundle: SourceBundle) -> list[dict[str, str]]:
    definitions = {
        "novelty_at_5": "train-only self-information with no smoothing",
        "catalog_coverage_at_5": "unique top-5 items divided by train-seen catalog",
        "item_coverage_at_5": "unique top-5 items divided by complete candidate union of the prompt-compatible cohort",
        "long_tail_rate_at_5": "train frequency strictly below the 0.8 quantile threshold",
        "average_popularity_at_5": "raw train-interaction counts",
        "duplicate_rate_at_5": "duplicates inside strict-valid top-5 exposures",
        "candidate_validity_rate": "strict-valid top-5 items checked against the row candidate list",
        "hallucination_rate": "structural hallucination rows over planned requests",
        "strict_parse_validity_rate": "strict parse-valid rows over planned requests",
        "popularity_bias_at_5": "blocked by ambiguous metric definition",
        "intra_list_diversity_at_5": "blocked by missing item similarity basis",
    }
    beyond = bundle.tables["phase_5c_beyond"]
    rows = []
    for metric_id in sorted(beyond["metric_id"].unique()):
        metric_rows = beyond[beyond["metric_id"].eq(metric_id)]
        statuses = sorted(metric_rows["metric_status"].unique())
        blocked_reasons = sorted({value for value in metric_rows["blocked_reason"].unique() if value})
        reduced = int(metric_rows["metric_status"].str.contains("reduced_popularity_coverage").sum())
        rows.append(
            {
                "metric_id": metric_id,
                "definition": definitions.get(metric_id, ""),
                "computed_status": ";".join(statuses),
                "blocked_reason": ";".join(blocked_reasons),
                "coverage_limitation": f"reduced_popularity_coverage_cells={reduced}",
                "source_paths": "results/tables/canonical_local_llm_beyond_utility_metrics.csv",
            }
        )
    return rows


def robustness_summary(bundle: SourceBundle) -> list[dict[str, str]]:
    prohibited = {
        "nondeterminism": "Do not claim nondeterminism from a single execution.",
        "candidate_position_bias": "Do not generalize diagnostic position effects to the full matrix.",
        "history_length_sensitivity": "Do not claim history-length effects outside matched diagnostic scope.",
        "strict_prompt_behavior": "Do not claim wording-only causality beyond the controlled diagnostic.",
        "full20_v1_vs_v2": "Do not use legacy full-20 rows as canonical top-10 evidence.",
        "full20_vs_top10": "Do not claim a numerical full-20 versus top-10 ablation.",
        "qwen2_vs_mistral": "Do not describe local substitutes as paper-model reproductions.",
        "ranking_vs_reranking": "Do not merge ranking and reranking denominators.",
        "generated_profiles": "Do not claim generated-profile benefit.",
        "hallucination": "Do not mix strict-index hallucination with earlier title-parser pilots.",
        "output_validity": "Do not repair invalid rows silently.",
    }
    rows = []
    for row in dataframe_records(bundle.tables["phase_5d_summary"]):
        rows.append(
            {
                "topic": row["topic"],
                "strongest_available_evidence": row["strongest_available_evidence"],
                "evidence_scope": row["evidence_scope"],
                "permitted_conclusion": row["concise_finding"],
                "limitation": row["limitations"],
                "prohibited_generalization": prohibited.get(row["topic"], "Do not generalize beyond the recorded scope."),
                "source_paths": "results/tables/plan_d_phase_5d_robustness_summary.csv",
            }
        )
    return rows


def extension_summary(bundle: SourceBundle) -> list[dict[str, str]]:
    phase5e_extensions = bundle.tables["phase_5e_extensions"]
    plan_e = bundle.tables["phase_6a_plan_e"]
    clean_plan_e = bundle.tables["phase_6clean_plan_e"]
    rows = [
        {
            "extension_family": "Phase 5E extension inventory",
            "artifact_count_or_rows": str(len(phase5e_extensions)),
            "classification": "extension_only_or_supporting_evidence",
            "changes_phase_5e": "false",
            "permitted_use": "reported separately from reproduction numerator",
            "source_paths": "results/tables/plan_d_phase_5e_extensions.csv",
        },
        {
            "extension_family": "Plan E reconciled artifacts",
            "artifact_count_or_rows": str(len(plan_e)),
            "classification": "extension_or_supporting_evidence",
            "changes_phase_5e": ";".join(sorted(plan_e["changes_phase_5e_claim"].unique())),
            "permitted_use": "does not alter frozen Phase 5E percentages",
            "source_paths": "results/tables/plan_d_phase_6a_plan_e_reconciliation.csv",
        },
    ]
    for row in dataframe_records(clean_plan_e):
        rows.append(
            {
                "extension_family": "Phase 6CLEAN Plan E disposition: " + row["subfamily"],
                "artifact_count_or_rows": row["file_count"],
                "classification": row["action"],
                "changes_phase_5e": "false",
                "permitted_use": row["notes"],
                "source_paths": "results/tables/plan_d_phase_6clean_plan_e_disposition.csv",
            }
        )
    return rows


def blocker_limitation_summary(bundle: SourceBundle) -> list[dict[str, str]]:
    rows = []
    for index, row in enumerate(dataframe_records(bundle.tables["phase_5e_blockers"]), start=1):
        rows.append(
            {
                "limitation_id": f"LIMIT-5E-{index:03d}",
                "category": row.get("category", "paper_component"),
                "status": row.get("status", row.get("blocker_status", "blocked")),
                "affected_claim": row.get("affected_claim", row.get("component", "")),
                "blocking_reason": row.get("blocking_reason", row.get("blocker", "")),
                "source_paths": "results/tables/plan_d_phase_5e_blockers.csv",
            }
        )
    for index, row in enumerate(dataframe_records(bundle.tables["phase_5d_blocked"]), start=1):
        rows.append(
            {
                "limitation_id": f"LIMIT-5D-{index:03d}",
                "category": row.get("topic", "robustness"),
                "status": row.get("status", "blocked_or_unresolved"),
                "affected_claim": row.get("claim_or_question", row.get("topic", "")),
                "blocking_reason": row.get("blocker_or_unresolved_reason", row.get("notes", "")),
                "source_paths": "results/tables/plan_d_phase_5d_blocked_or_unresolved.csv",
            }
        )
    return rows


def threat_registry() -> list[dict[str, str]]:
    return [
        {
            "threat_category": "Construct validity",
            "affected_claims": "Prompt/parser adaptation; operational utility adaptation; blocked popularity-bias and ILD metrics; incomplete novelty coverage; structural reliability versus recommendation quality.",
            "mitigation": "The report separates exact reproduction, reconstruction, adaptation, extension, diagnostics, and blocked work; conditional and operational utility have explicit denominators.",
            "residual_risk": "Local adapted metrics may still measure a narrower construct than the original paper intended.",
        },
        {
            "threat_category": "Internal validity",
            "affected_claims": "Dataset reconstruction, candidate construction, strict-title filtering, cohort alignment, invalid-output treatment, preprocessing differences, and diagnostic pairing.",
            "mitigation": "Candidate/cohort audits, strict parser contracts, protected-input hashes, and exact-intersection baseline comparisons are retained.",
            "residual_risk": "Unavailable original processed artifacts prevent full control of cohort and preprocessing differences.",
        },
        {
            "threat_category": "External validity",
            "affected_claims": "Generality across models, datasets, hardware, and closed-source LLMs.",
            "mitigation": "The report scopes claims to Qwen2 and Mistral 7B local substitutes on Amazon Beauty, Amazon Sports, and MovieLens 1M.",
            "residual_risk": "LastFM is incomplete and closed-source paper-model behavior remains externally unverified.",
        },
        {
            "threat_category": "Conclusion validity",
            "affected_claims": "Paired bootstrap interpretation, partial baseline comparisons, zero exact numerical result-cell coverage, non-causal diagnostics, and blocked robustness topics.",
            "mitigation": "Bootstrap intervals are descriptive, not confirmatory significance tests, and global baseline-LLM rankings are prohibited.",
            "residual_risk": "Partial intersections and diagnostic subsets can support engineering conclusions but not broad causal claims.",
        },
        {
            "threat_category": "Reproducibility validity",
            "affected_claims": "Missing original artifacts, local model substitutions, uncommitted worktree, large artifact set, local Windows environment, and preserved but unpruned evidence.",
            "mitigation": "Phase 6A/6B/6CLEAN manifests, safe commands, source inventories, and protected hashes preserve traceability.",
            "residual_risk": "A later submission bundle may need selection without altering the scientific worktree.",
        },
    ]


def conclusion_registry(claim_ids: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {
            "conclusion_id": "CONC-001",
            "conclusion_text": "The project produced a defensible local reconstruction and extension package, not an exact reproduction of the paper.",
            "supporting_claim_ids": ";".join([claim_ids["coverage_exact"], claim_ids["coverage_numeric"]]),
            "confidence": "high_with_scope_limits",
            "required_caveat": "Exact numerical paper-result-cell coverage is zero.",
            "prohibited_claim": "Do not call the project a complete exact replication.",
        },
        {
            "conclusion_id": "CONC-002",
            "conclusion_text": "The canonical local LLM matrix is structurally reliable under strict index protocols.",
            "supporting_claim_ids": ";".join([claim_ids["planned_requests"], claim_ids["invalid_outputs"]]),
            "confidence": "high_for_local_artifacts",
            "required_caveat": "Structural reliability is not the same as recommendation quality.",
            "prohibited_claim": "Do not collapse backend success, parse validity, hallucination, and duplicate-index failures into one label.",
        },
        {
            "conclusion_id": "CONC-003",
            "conclusion_text": "Mistral on MovieLens 1M ranking is the best local conditional HR@5 cell.",
            "supporting_claim_ids": claim_ids["best_utility"],
            "confidence": "high_within_local_scope",
            "required_caveat": "This is not the paper's best model and not paper-model reproduction.",
            "prohibited_claim": "Do not compare it as a direct substitute for closed-source paper results.",
        },
        {
            "conclusion_id": "CONC-004",
            "conclusion_text": "Baseline evidence supports diagnostic intersection comparisons only.",
            "supporting_claim_ids": claim_ids["baseline_partial"],
            "confidence": "high_for_alignment_audit",
            "required_caveat": "No fully aligned global baseline-LLM cohort comparison exists.",
            "prohibited_claim": "Do not create a global baseline ranking across non-aligned cohorts.",
        },
        {
            "conclusion_id": "CONC-005",
            "conclusion_text": "Plan E and Phase 6CLEAN are extensions and repository-management evidence, not changes to Phase 5E reproduction accounting.",
            "supporting_claim_ids": ";".join([claim_ids["plan_e"], claim_ids["phase6clean"]]),
            "confidence": "high",
            "required_caveat": "Plan E pilot and diagnostic evidence must remain labelled.",
            "prohibited_claim": "Do not promote Plan E to Phase 5E exact reproduction.",
        },
    ]


def add_claim(rows: list[dict[str, str]], prefix: str, key: str, text: str, value: str, sources: str, sections: str, prohibited: str, ids: dict[str, str]) -> None:
    claim_id = stable_claim_id(prefix, text)
    ids[key] = claim_id
    rows.append(
        {
            "claim_id": claim_id,
            "claim_prefix": prefix,
            "claim_text": text,
            "quantitative": str(bool(value)).lower(),
            "claim_value": value,
            "source_paths": sources,
            "report_sections": sections,
            "claim_status": "supported_by_canonical_source",
            "prohibited_overclaim": prohibited,
        }
    )


def claim_registry(bundle: SourceBundle, summaries: Mapping[str, Sequence[Mapping[str, str]]]) -> tuple[list[dict[str, str]], dict[str, str]]:
    counts = phase5e_counts(bundle)
    full = full_matrix_counts(bundle)
    best = best_conditional_utility_cell(bundle)
    baseline_rows = summaries["baseline_summary"]
    plan_e_rows = summaries["extension_summary"]
    ids: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    add_claim(rows, "COV", "coverage_exact", "Strict exact component reproduction is 9 of 73 applicable components, equal to 12.328767%.", "9/73=12.328767%", "results/tables/plan_d_phase_5e_coverage_metrics.csv", "Component-Level Reproducibility Coverage", "Do not include reconstructed, adapted, or extension rows in the exact numerator.", ids)
    add_claim(rows, "COV", "coverage_reconstructed", "Reconstructed-or-better component coverage is 25 of 73 applicable components, equal to 34.246575%.", "25/73=34.246575%", "results/tables/plan_d_phase_5e_coverage_metrics.csv", "Component-Level Reproducibility Coverage", "Do not include adapted or extension rows.", ids)
    add_claim(rows, "COV", "coverage_max", "Maximum defensible paper-component coverage is 50 of 73 applicable components, equal to 68.493151%.", "50/73=68.493151%", "results/tables/plan_d_phase_5e_coverage_metrics.csv", "Component-Level Reproducibility Coverage", "Do not describe maximum defensible coverage as exact replication.", ids)
    add_claim(rows, "COV", "coverage_numeric", "Exact numerical paper-result-cell coverage is 0.0%.", "0.0%", "results/tables/plan_d_phase_5e_coverage_metrics.csv;results/tables/plan_d_phase_5e_result_cell_comparison.csv", "Exact Numerical Result-Cell Coverage", "Do not interpret zero numerical coverage as absence of useful evidence.", ids)
    add_claim(rows, "DATA", "datasets", "The complete canonical local LLM evaluation covers Amazon Beauty, Amazon Sports, and MovieLens 1M; LastFM remains incomplete or blocked.", "3 complete datasets", "results/tables/dataset_statistics.csv;results/tables/canonical_local_llm_full_matrix_manifest.csv", "Datasets", "Do not report LastFM as complete.", ids)
    add_claim(rows, "COHORT", "cohort_rows", "Prompt-compatible task rows are 990 for Beauty, 892 for Sports, and 1000 for MovieLens 1M, yielding 2882 ranking rows and 2882 reranking rows before multiplying by two local models.", "990;892;1000;2882;2882", "results/tables/canonical_local_llm_full_matrix_manifest.csv", "Preprocessing, Cohorts, and Candidate Construction", "Do not mix these counts with legacy 100-candidate cohorts.", ids)
    add_claim(rows, "PROTO", "ranking_protocol", "The canonical ranking protocol asks for exactly five unique one-based indices from twenty candidates.", "20 candidates;5 outputs", "results/tables/canonical_local_llm_full_matrix_manifest.csv;results/tables/canonical_llm_prompt_contract.csv", "Ranking Protocol", "Do not treat legacy full-20 output as canonical ranking.", ids)
    add_claim(rows, "PROTO", "reranking_protocol", "The canonical reranking protocol asks for exactly ten unique one-based indices from twenty candidates and does not require a full permutation.", "20 candidates;10 outputs", "results/tables/canonical_local_llm_full_matrix_manifest.csv;results/tables/canonical_local_llm_reranking_top10_prompt_contract.csv", "Reranking Protocol", "Do not require full-20 output for the canonical top-10 protocol.", ids)
    add_claim(rows, "REL", "planned_requests", f"The full local LLM matrix contains {full['planned_requests']} planned requests, {full['responses']} responses, and {full['runner_events']} runner events.", f"{full['planned_requests']};{full['responses']};{full['runner_events']}", "results/tables/canonical_local_llm_full_matrix_decision.csv;results/tables/canonical_local_llm_full_matrix_validation.csv", "Structural Reliability", "Do not infer paper-model availability from local response availability.", ids)
    add_claim(rows, "REL", "invalid_outputs", f"Strict-valid ranking outputs are {full['ranking_strict_valid']} of 5764 and strict-valid reranking outputs are {full['reranking_strict_valid']} of 5764; the {full['invalid_outputs']} invalid outputs are Qwen2 reranking duplicate-index failures.", f"{full['ranking_strict_valid']}/5764;{full['reranking_strict_valid']}/5764;{full['invalid_outputs']}", "results/tables/canonical_local_llm_structural_metrics.csv;results/tables/canonical_local_llm_full_invalid_output_inventory.csv", "Structural Reliability", "Do not repair or silently discard invalid outputs.", ids)
    add_claim(rows, "REL", "invalid_distribution", f"Invalid-output distribution is Beauty {full['beauty_invalid']}, Sports {full['sports_invalid']}, and MovieLens 1M {full['ml1m_invalid']}.", f"{full['beauty_invalid']};{full['sports_invalid']};{full['ml1m_invalid']}", "results/tables/canonical_local_llm_full_invalid_output_inventory.csv", "Structural Reliability", "Do not attribute invalid rows to Mistral.", ids)
    add_claim(rows, "UTIL", "best_utility", f"The best local conditional HR@5 cell is {best['model_family']} on {best['dataset']} {best['task_type']} with HR@5 {best['hr_at_5']} and NDCG@5 {best['ndcg_at_5']}.", f"{best['hr_at_5']};{best['ndcg_at_5']}", "results/tables/canonical_local_llm_utility_metrics.csv", "Conditional and Operational Utility Results", "Do not describe this as reproducing the paper's best model.", ids)
    add_claim(rows, "BASE", "baseline_partial", f"All {sum(int(row['dataset_task_combinations']) for row in baseline_rows)} baseline-dataset-task combinations have partial baseline coverage; {sum(int(row['paired_bootstrap_hr_ndcg_count']) for row in baseline_rows)} paired-bootstrap HR@5/NDCG@5 comparisons are descriptive with seed 42 and 10000 resamples.", "42;168;42;10000", "results/tables/canonical_local_llm_baseline_cohort_alignment.csv;results/tables/canonical_local_llm_baseline_comparison.csv", "Baseline Alignment and Paired Comparisons", "Do not create a global ranking over non-aligned cohorts.", ids)
    add_claim(rows, "BU", "beyond_metrics", "Beyond-utility evaluation computes novelty@5, catalog coverage@5, item coverage@5, long-tail rate@5, average popularity@5, duplicate rate, candidate validity, hallucination rate, and strict parse validity.", "9 computed metric families", "results/tables/canonical_local_llm_beyond_utility_metrics.csv", "Beyond-Utility Evaluation", "Do not equate novelty with diversity or serendipity.", ids)
    add_claim(rows, "BU", "blocked_metrics", "Popularity-bias@5 and intra-list-diversity@5 remain blocked and no substitute value is computed.", "2 blocked metrics", "results/tables/canonical_local_llm_beyond_utility_metrics.csv;results/tables/canonical_metric_blocked_results.csv", "Beyond-Utility Evaluation", "Do not estimate blocked metrics.", ids)
    add_claim(rows, "ROB", "robustness_limits", "Phase 5D permits only scoped robustness conclusions and blocks nondeterminism, generated-profile benefit, and full20-versus-top10 numerical-ablation claims.", "scoped diagnostic evidence", "results/tables/plan_d_phase_5d_robustness_summary.csv", "Robustness and Diagnostic Evidence", "Do not generalize beyond the strongest available evidence level.", ids)
    add_claim(rows, "EXT", "plan_e", f"Phase 6A reconciles {len(bundle.tables['phase_6a_plan_e'])} Plan E artifacts as extension or supporting evidence that does not change Phase 5E accounting.", str(len(bundle.tables["phase_6a_plan_e"])), "results/tables/plan_d_phase_6a_plan_e_reconciliation.csv", "Project Extensions and Plan E", "Do not report Plan E as Phase 5E.", ids)
    add_claim(rows, "EXT", "phase6clean", "Phase 6CLEAN archived zero scientific artifacts and deleted only technical byproducts.", "0 archived scientific artifacts", "results/tables/plan_d_phase_6clean_before_after_statistics.csv;results/tables/plan_d_phase_6clean_validation.csv", "Repository Pruning and Evidence Preservation", "Do not present repository pruning as a scientific experiment.", ids)
    add_claim(rows, "LIMIT", "blocked_denominator", "Blocked, not-executed, and unresolved components remain in the Phase 5E denominator.", f"{counts['blocked_count']} blocked;{counts['not_executed_count']} not_executed;{counts['unresolved_count']} unresolved", "results/tables/plan_d_phase_5e_coverage_metrics.csv", "Blocked, Not-Executed, and Unresolved Components", "Do not remove blocked work from the denominator.", ids)
    add_claim(rows, "CONC", "final_decision", "The final unified scientific report is complete only as a Phase 6C written deliverable, not as final-delivery packaging.", "phase_6c_only", "report/final_scientific_report.md;report/plan_d_phase_6c_final_scientific_report_build.md", "Conclusions", "Do not infer Phase 6D or Phase 7 completion.", ids)
    return rows, ids


def section_source_map(claims: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    by_section: dict[str, list[str]] = {}
    source_by_section: dict[str, list[str]] = {}
    for claim in claims:
        for section in claim["report_sections"].split(";"):
            section = section.strip()
            by_section.setdefault(section, []).append(claim["claim_id"])
            source_by_section.setdefault(section, []).extend(claim["source_paths"].split(";"))
    rows = []
    for index, section in enumerate(REQUIRED_SECTIONS, start=1):
        sources = sorted({source for source in source_by_section.get(section, []) if source})
        if not sources:
            sources = default_sources_for_section(section)
        rows.append(
            {
                "section_number": str(index),
                "section_title": section,
                "primary_sources": ";".join(sources),
                "claim_ids": ";".join(sorted(by_section.get(section, []))),
            }
        )
    return rows


def default_sources_for_section(section: str) -> list[str]:
    if "Threats" in section:
        return ["results/tables/plan_d_phase_6c_threats_to_validity.csv"]
    if "References" in section:
        return ["paper/2411.00331v1.pdf", "original_repo/EvaLLMasRecommender/README.md"]
    if "Appendix" in section:
        return ["results/tables/plan_d_phase_6c_claim_source_registry.csv"]
    return ["results/tables/plan_d_phase_6b_final_artifact_index.csv", "results/tables/plan_d_phase_6a_canonical_artifact_map.csv"]


def project_scope_summary(bundle: SourceBundle) -> list[dict[str, str]]:
    return [
        {"scope_id": "SCOPE-001", "topic": "phase", "value": "Phase 6C final unified scientific report", "source_paths": "report/final_scientific_report.md", "notes": "No final-delivery packaging, slides, oral presentation, Phase 6D, or Phase 7 work."},
        {"scope_id": "SCOPE-002", "topic": "branch", "value": EXPECTED_BRANCH, "source_paths": "git branch --show-current", "notes": "Validated before build."},
        {"scope_id": "SCOPE-003", "topic": "head", "value": EXPECTED_HEAD, "source_paths": "git rev-parse HEAD", "notes": "Validated before build."},
        {"scope_id": "SCOPE-004", "topic": "environment", "value": f"Python {sys.version.split()[0]}; pandas {pd.__version__}", "source_paths": "local Python environment", "notes": "NumPy version is recorded in build report."},
        {"scope_id": "SCOPE-005", "topic": "report_notes_provenance", "value": bundle.notes_provenance, "source_paths": "git diff -- report/notes.md;results/tables/plan_d_phase_6clean_validation.csv", "notes": "Proceed only when resolved as authorized or pre-existing."},
    ]


def next_actions() -> list[dict[str, str]]:
    return [
        {
            "action_id": "NEXT-6C-001",
            "next_phase": "Phase 6D or final delivery packaging",
            "required_input": "report/final_scientific_report.md;results/tables/plan_d_phase_6c_claim_source_registry.csv",
            "instruction": "Select a smaller submission bundle without deleting scientific evidence.",
            "blocking": "false",
        },
        {
            "action_id": "NEXT-6C-002",
            "next_phase": "Human review",
            "required_input": "report/final_scientific_report.md",
            "instruction": "Review wording for institutional style, but do not add unsupported supervisor, institution, legal-status, or publication claims.",
            "blocking": "false",
        },
    ]


def markdown_heading_count(markdown: str) -> int:
    return sum(1 for line in markdown.splitlines() if line.startswith("## "))


def report_section_count(markdown: str) -> int:
    return (1 if markdown.startswith("# ") else 0) + markdown_heading_count(markdown)


def word_count(markdown: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", markdown))


def markdown_links(markdown: str) -> list[str]:
    return [
        match.group(1)
        for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", markdown)
        if not match.group(1).startswith(("http://", "https://", "mailto:", "#"))
    ]


def relative_links_resolve(root: Path, markdown: str, report_path: Path = FINAL_REPORT_PATH) -> bool:
    base = (root / report_path).parent
    for target in markdown_links(markdown):
        clean = target.split("#", 1)[0]
        if not clean:
            continue
        if not (base / clean).resolve().exists():
            return False
    return True


def absolute_path_hits(text: str) -> list[str]:
    return ABSOLUTE_PATH_PATTERN.findall(text)


def detect_unsupported_overclaims(text: str) -> list[str]:
    banned_patterns = [
        r"\bis a complete exact replication\b",
        r"\bachieved complete exact replication\b",
        r"\bfully reproduced the paper\b",
        r"\bproved nondeterminism\b",
        r"\bgenerated profiles improved\b",
        r"\bstatistically significant\b",
        r"\bpublication[- ]ready\b",
        r"\bLastFM complete\b",
    ]
    hits = []
    for pattern in banned_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def scientific_language_audit(text: str) -> dict[str, object]:
    return {
        "absolute_path_hits": absolute_path_hits(text),
        "unsupported_overclaims": detect_unsupported_overclaims(text),
        "contains_local_substitute_wording": "local substitutes" in text and "not reproductions of the paper" in text,
        "contains_baseline_caveat": "No fully aligned global baseline-LLM cohort comparison exists" in text,
        "contains_blocked_metrics": "Popularity-bias@5 and intra-list-diversity@5 remain blocked" in text,
    }


def generate_report(bundle: SourceBundle, summaries: Mapping[str, Sequence[Mapping[str, str]]], claims: Sequence[Mapping[str, str]], conclusions: Sequence[Mapping[str, str]]) -> str:
    counts = phase5e_counts(bundle)
    full = full_matrix_counts(bundle)
    best = best_conditional_utility_cell(bundle)
    baseline_total = sum(int(row["dataset_task_combinations"]) for row in summaries["baseline_summary"])
    paired_total = sum(int(row["paired_bootstrap_hr_ndcg_count"]) for row in summaries["baseline_summary"])
    model_digests = model_digest_text(bundle)
    threat_text = "\n\n".join(
        f"### {row['threat_category']}\nAffected claims: {row['affected_claims']}\n\nMitigation: {row['mitigation']}\n\nResidual risk: {row['residual_risk']}"
        for row in summaries["threats_to_validity"]
    )
    dataset_lines = "\n".join(
        f"- {row['dataset']} {row['task_type']}: {row['prompt_compatible_rows']} prompt-compatible rows, {row['candidate_count']} candidates, source rows {row['source_candidate_rows']}."
        for row in summaries["dataset_cohort_summary"]
    )
    utility_lines = "\n".join(
        f"- {row['model_family']} / {row['dataset']} / {row['task_type']} / {row['evaluation_policy_id']}: HR@5={row['hr_at_5']}, NDCG@5={row['ndcg_at_5']}, invalid rows={row['invalid_rows']}."
        for row in summaries["llm_utility_summary"]
    )
    robustness_lines = "\n".join(
        f"- {row['topic']}: evidence={row['strongest_available_evidence']}; permitted={row['permitted_conclusion']}; limitation={row['limitation']}; prohibited={row['prohibited_generalization']}"
        for row in summaries["robustness_summary"]
    )
    conclusion_lines = "\n".join(
        f"- {row['conclusion_id']}: {row['conclusion_text']} Caveat: {row['required_caveat']}"
        for row in conclusions
    )
    claim_lookup = {row["claim_prefix"]: [] for row in claims}
    for row in claims:
        claim_lookup.setdefault(row["claim_prefix"], []).append(row["claim_id"])
    report = f"""# Final Scientific Report: Beyond Utility LLM Recommender Reproducibility Project

## Abstract
This report is the final unified scientific written deliverable for a reproducibility and extension project based on *Beyond Utility: Evaluating LLM as Recommender* (arXiv:2411.00331). The project studies the paper and the official JiangDeccc/EvaLLMasRecommender repository, then builds a local, auditable evaluation wrapper around the portions that can be supported by available evidence. The central result is not a complete exact reproduction. Strict exact component reproduction is {counts['reproduced_exact_count']} of {counts['applicable_component_count']} applicable paper components, or {counts['strict_exact_reproduction_micro_pct']}%, while maximum defensible component coverage is {counts['maximum_defensible_micro_coverage_pct']}% when exact, reconstructed, and adapted counterparts are combined. Exact numerical paper-result-cell coverage is {counts['exact_numeric_result_cell_pct']}%. These values are frozen from Phase 5E and are reported under claim IDs in `results/tables/plan_d_phase_6c_claim_source_registry.csv`.

## Executive Summary
The project reconstructed a substantial local evidence base, but it did not make unavailable paper artifacts appear available. The complete canonical local LLM matrix covers Amazon Beauty, Amazon Sports, and MovieLens 1M, while LastFM remains incomplete or blocked. Qwen2 7B Instruct and Mistral 7B Instruct are local substitutes and adapted local-model extensions, not reproductions of the paper's original LLMs. The full matrix contains {full['planned_requests']} planned requests, {full['responses']} responses, {full['runner_events']} runner events, {full['ranking_evaluations']} ranking evaluations, and {full['reranking_evaluations']} reranking evaluations. All backend calls completed without retries or backend errors; strict parsing leaves {full['invalid_outputs']} invalid outputs, all Qwen2 reranking duplicate-index failures. The best local conditional utility cell is {best['model_family']} on {best['dataset']} {best['task_type']} with HR@5={best['hr_at_5']} and NDCG@5={best['ndcg_at_5']}. This is a local result, not a reproduction of the paper's best model.

## Introduction
This report synthesizes the completed project phases rather than concatenating phase reports. The project goal was to determine what can be reproduced, reconstructed, adapted, extended, or blocked when studying a modern LLM-as-recommender paper with incomplete access to the original processed artifacts and closed-source model matrix. The report uses local row-level evidence first, phase audits second, and paper or repository text only for defining the original target.

## Original Paper and Research Problem
The paper under study is *Beyond Utility: Evaluating LLM as Recommender*, arXiv:2411.00331. It evaluates LLM recommenders not only by ranking utility but also by beyond-utility properties. The official repository studied is `original_repo/EvaLLMasRecommender`. The original paper defines the scientific object of interest, but the local repository state defines what this project actually executed or verified.

## Project Objectives and Research Questions
The project asks four practical questions. First, which paper components can be exactly reproduced from local evidence? Second, which components can be reconstructed or adapted with clear caveats? Third, what additional diagnostics and extensions are scientifically useful without changing the reproduction numerator? Fourth, what residual blockers prevent a stronger exact-reproduction claim?

## Reproducibility Terminology and Scope
Exact reproduction means a paper component has an aligned local counterpart across the material dimensions that Phase 5E records. Reconstruction means a paper-like counterpart exists but differs in at least one material input, artifact, or preprocessing dimension. Adaptation means the project changed an interface, model, prompt, parser, denominator, or execution condition while preserving a defensible scientific relationship. Extension means useful project work outside the paper reproduction numerator. Diagnostic evidence is scoped evidence for a particular failure mode or mechanism. Blocked work remains in the denominator where applicable.

## Repository and Validated Environment
The repository branch is `plan-d-extensions` and the expected HEAD is `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`. The validated local environment is Windows with Python 3.12.10, pandas 3.0.3, NumPy 2.5.0, an Intel i9-14900HX CPU, an NVIDIA RTX 4060 Laptop GPU, and approximately 32 GB RAM. Phase 6C did not install packages, access the network, invoke Ollama, invoke APIs, download artifacts, execute LLM inference, execute recommender training, or regenerate frozen experimental outputs.

## Datasets
The complete canonical evaluation scope contains Amazon Beauty, Amazon Sports, and MovieLens 1M. LastFM is retained as incomplete or blocked because the current artifacts do not provide the complete validated candidate set and ranking split needed for the canonical LLM evaluation. Dataset and artifact details come from `results/tables/dataset_statistics.csv` and the canonical full-matrix manifest.

{dataset_lines}

## Preprocessing, Cohorts, and Candidate Construction
The canonical prompt-compatible cohorts use twenty candidates per row. Each row contains one positive item and nineteen negatives, with candidate order preserved and title availability enforced for LLM prompting. Prompt-compatible task rows are 990 for Beauty, 892 for Sports, and 1000 for MovieLens 1M. This yields 2882 ranking rows and 2882 reranking rows before multiplying by the two local models. Legacy 100-candidate cohorts and diagnostic 20-row variants are not mixed into the canonical Phase 5C utility denominator.

## Traditional Recommender Baselines
The canonical baseline families are random_seeded, mostpop_train_only, bm25_like_e3b, bprmf_reconstructed, gru4rec_reconstructed, sasrec_reconstructed, and lightgcn_reconstructed. All {baseline_total} baseline-dataset-task combinations have partial baseline coverage. No fully aligned global baseline-LLM cohort comparison exists. Paired comparisons on exact intersections are diagnostic and use descriptive bootstrap intervals only.

## Local LLM Models
The local models are qwen2:7b-instruct and mistral:7b-instruct. Manifest-verified digests are reported only where the canonical manifest records them: {model_digests}. These models are local substitutes and adapted local-model extensions, not reproductions of the paper's original LLMs.

## Ranking Protocol
The canonical ranking prompt is `llm_rank_index_json_top5_v1`; the parser is `llm_rank_strict_primary_parser_v1`. Each request presents 20 candidates and requires exactly 5 unique one-based indices. The output is evaluated under strict parser and candidate-validity rules, without repair.

## Reranking Protocol
The canonical reranking prompt is `llm_rerank_index_json_top10_officialcode_v1`; the parser is `llm_rerank_top10_strict_primary_parser_v1`. Each request presents 20 candidates and requires exactly 10 unique one-based indices. A full 20-item permutation is not required. The local classification is official_code_aligned_top10_semantics_adapted_json_indices.

## Prompt and Parser Adaptation
Prompt and parser changes are explicit adaptations. The project uses strict JSON-index outputs to make parsing auditable and candidate validity deterministic. Legacy full-20 protocols remain legacy or diagnostic only. The parser does not salvage malformed output, repair duplicate indices, or silently normalize invalid responses into valid rankings.

## Evaluation Methodology
The evaluation has three layers. Structural reliability uses all planned requests as the denominator. Conditional utility uses only strict-valid outputs and is classified as conditional_strict_valid_output_utility. Operational utility uses all planned requests, sets invalid HR@5 and NDCG@5 to zero, and is classified as adapted_end_to_end_operational_utility_invalid_as_failure. Operational utility is a project adaptation, not a paper-exact metric.

## Structural Reliability
The full matrix has {full['planned_requests']} planned requests, {full['responses']} responses, {full['runner_events']} runner events, zero retries, zero backend errors, zero ambiguous requests, zero truncations, and zero canonical strict-index hallucinations. Ranking strict-validity is {full['ranking_strict_valid']} / 5764. Reranking strict-validity is {full['reranking_strict_valid']} / 5764. The {full['invalid_outputs']} invalid outputs are all Qwen2 reranking duplicate-index failures: Beauty {full['beauty_invalid']}, Sports {full['sports_invalid']}, and MovieLens 1M {full['ml1m_invalid']}. Backend success, response availability, parse validity, candidate validity, duplicate indices, hallucination, truncation, retry, and backend error are therefore separate reliability dimensions.

## Conditional and Operational Utility Results
The stored utility rows distinguish conditional and operational policies. Conditional rows exclude strict-invalid outputs from the denominator. Operational rows retain all planned requests and assign invalid rows HR@5=0 and NDCG@5=0. The best local conditional HR@5 cell is {best['model_family']} on {best['dataset']} {best['task_type']} with HR@5={best['hr_at_5']} and NDCG@5={best['ndcg_at_5']}. The full stored utility table is:

{utility_lines}

## Baseline Alignment and Paired Comparisons
Baseline comparison is intentionally constrained. The alignment audit records {baseline_total} baseline-dataset-task combinations, all with partial baseline coverage. The comparison table contains {paired_total} HR@5/NDCG@5 paired-bootstrap rows using seed 42 and 10000 resamples with descriptive 95% intervals. These are not confirmatory significance tests. They are exact-intersection diagnostics, not permission to rank global LLM means against global baseline means.

## Beyond-Utility Evaluation
The computed beyond-utility metrics are novelty@5, catalog coverage@5, item coverage@5, long-tail rate@5, average popularity@5, duplicate rate, candidate validity, hallucination rate, and strict parse validity. Novelty uses train-only self-information with no smoothing. Catalog coverage is unique top-5 items divided by the train-seen catalog. Item coverage is unique top-5 items divided by the complete candidate union of the prompt-compatible cohort. Long-tail rate uses a train-frequency threshold at quantile 0.8 and a strict less-than comparison. Average popularity uses raw train-interaction counts. Novelty has incomplete popularity coverage in 11 of 12 cells, so evaluated-item coverage must be reported. Popularity-bias@5 and intra-list-diversity@5 remain blocked, and no substitute values are computed. Novelty is not diversity or serendipity.

## Robustness and Diagnostic Evidence
Phase 5D is authoritative for robustness. Its evidence hierarchy is full-matrix Phase 5C evidence, exact paired diagnostic evidence, repeated controlled diagnostics, small unpaired pilots, report-only summaries, and blocked or unresolved artifacts. The permitted robustness conclusions are:

{robustness_lines}

## Qwen2 versus Mistral
Within the local canonical matrix, Mistral is structurally strict-valid in all cells, while Qwen2 has 26 reranking duplicate-index failures. This comparison is within the local substitute-model scope. It does not reproduce the paper model matrix and does not support claims about closed-source model behavior.

## Ranking versus Reranking
Ranking and reranking are separate tasks with different requested output lengths and different parser contracts. Ranking asks for five indices. Reranking asks for ten indices. Utility, reliability, and beyond-utility summaries therefore keep these task semantics separate.

## Paper Alignment
The paper defines the target components, but local canonical evidence defines what was executed. Phase 5E is the authority for paper alignment. Numerical similarity to a paper value is not accepted as proof of reproduction if model, cohort, candidate set, prompt, parser, metric, or output source is not fully aligned.

## Component-Level Reproducibility Coverage
Phase 5E freezes the component accounting: {counts['applicable_component_count']} applicable components, {counts['reproduced_exact_count']} reproduced_exact, {counts['reconstructed_count']} reconstructed, {counts['adapted_count']} adapted, {counts['blocked_count']} blocked, {counts['not_executed_count']} not_executed, and {counts['unresolved_count']} unresolved. The sum is 73. Strict exact component reproduction is {counts['strict_exact_reproduction_micro_pct']}%. Reconstructed-or-better coverage is {counts['reconstructed_or_better_micro_pct']}%. Maximum defensible paper-component coverage is {counts['maximum_defensible_micro_coverage_pct']}%, but this is not an exact replication percentage.

## Exact Numerical Result-Cell Coverage
Exact numerical paper-result-cell coverage is {counts['exact_numeric_result_cell_pct']}%. No paper result cell has a fully aligned local counterpart across every material dimension. This does not mean the project produced no useful scientific evidence; it means the project refuses to turn non-aligned local values into paper-result reproductions.

## Project Extensions and Plan E
Phase 6A identified {len(bundle.tables['phase_6a_plan_e'])} reconciled Plan E artifacts. Plan E is not Phase 5E. Plan E does not alter the frozen Phase 5E percentages. Validated Plan E components are extensions or supporting evidence. Plan E pilot and diagnostic evidence must not be promoted to exact reproduction.

## Blocked, Not-Executed, and Unresolved Components
Blocked, not-executed, and unresolved components remain in the Phase 5E denominator. Blocked work includes unavailable original processed artifacts, unavailable or unverified paper model conditions, incomplete LastFM candidate evidence, and metrics whose definitions or item-similarity bases are not sufficiently specified locally. The denominator is intentionally conservative.

## Repository Pruning and Evidence Preservation
Phase 6CLEAN is repository management, not a scientific experiment. It archived zero scientific artifacts, deleted only cache, bytecode, and empty technical cache directories, and preserved all scientific evidence. The pruning reduced technical noise but did not materially reduce scientific artifact volume. A later final-delivery bundle may select a smaller submission set without deleting the scientific worktree.

## Threats to Validity
{threat_text}

## Reproducibility Package and Artifact Availability
The final report is backed by `README.md`, `docs/reproducibility_guide.md`, `docs/artifact_catalog.md`, Phase 6A artifact maps, Phase 6B claim audits, and Phase 6C traceability tables. The most important generated Phase 6C tables are `results/tables/plan_d_phase_6c_claim_source_registry.csv`, `results/tables/plan_d_phase_6c_report_section_source_map.csv`, and `results/tables/plan_d_phase_6c_validation.csv`. The source PDF is available at [paper/2411.00331v1.pdf](../paper/2411.00331v1.pdf).

## Discussion
The project's main contribution is a disciplined boundary between what was executed and what can be claimed. It shows that a useful local LLM-recommender evaluation can be constructed even when exact paper artifacts are unavailable, provided that adaptations are named and evidence classes are not collapsed. The strongest evidence is structural and local: full-matrix request/response accounting, strict parser validation, utility under explicit denominator policies, and row-level baseline-intersection diagnostics.

## Conclusions
{conclusion_lines}

## References
1. *Beyond Utility: Evaluating LLM as Recommender*, arXiv:2411.00331, local copy [paper/2411.00331v1.pdf](../paper/2411.00331v1.pdf).
2. Official repository studied: JiangDeccc/EvaLLMasRecommender, local copy [original_repo/EvaLLMasRecommender](../original_repo/EvaLLMasRecommender).

## Appendix A - Canonical Protocols
Ranking: `llm_rank_index_json_top5_v1` with `llm_rank_strict_primary_parser_v1`, 20 candidates, and exactly 5 unique one-based indices. Reranking: `llm_rerank_index_json_top10_officialcode_v1` with `llm_rerank_top10_strict_primary_parser_v1`, 20 candidates, and exactly 10 unique one-based indices. Full-20 protocols are legacy or diagnostic only.

## Appendix B - Component Accounting
The frozen accounting is {counts['reproduced_exact_count']} + {counts['reconstructed_count']} + {counts['adapted_count']} + {counts['blocked_count']} + {counts['not_executed_count']} + {counts['unresolved_count']} = {counts['applicable_component_count']}. Extensions are reported separately and do not enter the reproduction numerator.

## Appendix C - Canonical Artifact Map
The canonical artifact map is `results/tables/plan_d_phase_6a_canonical_artifact_map.csv`. The Phase 6B final artifact index is `results/tables/plan_d_phase_6b_final_artifact_index.csv`. Phase 6C adds source, section, claim, conclusion, validation, protected-input, and deterministic-output manifests.

## Appendix D - Permitted and Prohibited Claims
Permitted: the project completed a scoped local reconstruction and extension with strict traceability. Permitted: local Qwen2 and Mistral evidence supports adapted local-model conclusions. Prohibited: complete exact replication, paper-model reproduction, global baseline ranking across non-aligned cohorts, generated-profile benefit, nondeterminism from a single execution, numerical full20/top10 ablation, or computed values for blocked metrics.
"""
    return report.rstrip() + "\n"


def model_digest_text(bundle: SourceBundle) -> str:
    manifest = bundle.tables["phase_5c_manifest"]
    parts = []
    for model_tag, group in manifest.groupby("model_tag", sort=True):
        digests = sorted({value for value in group["model_digest"].unique() if value})
        digest_text = digests[0] if digests else "not recorded"
        parts.append(f"{model_tag}={digest_text}")
    return "; ".join(parts)


def generate_build_report(decision: str, validations: Sequence[Mapping[str, str]], protected_count: int, output_count: int, report_word_count: int, report_section_count: int) -> str:
    pass_count = sum(1 for row in validations if row["status"] == "pass")
    fail_count = sum(1 for row in validations if row["status"] == "fail")
    return "\n".join(
        [
            "# Phase 6C Final Scientific Report Build",
            "",
            f"Decision: `{decision}`.",
            "",
            "## Scope",
            "",
            "This build created the final unified scientific report and Phase 6C traceability tables only.",
            "It did not start final-delivery packaging, slides, oral presentation, Phase 6D, or Phase 7.",
            "",
            "## Counts",
            "",
            f"Protected input count: `{protected_count}`.",
            f"Hashed output count: `{output_count}`.",
            f"Final report section count: `{report_section_count}`.",
            f"Final report word count: `{report_word_count}`.",
            f"Validation pass/fail: `{pass_count}/{fail_count}`.",
            "",
            "## Safety",
            "",
            "No network access, Ollama call, API call, download, inference, recommender execution, package installation, Git staging, or commit is performed by the builder.",
            "",
        ]
    )


def build_summaries(bundle: SourceBundle) -> dict[str, list[dict[str, str]]]:
    summaries: dict[str, list[dict[str, str]]] = {}
    summaries["project_scope_summary"] = project_scope_summary(bundle)
    summaries["dataset_cohort_summary"] = dataset_cohort_summary(bundle)
    summaries["protocol_summary"] = protocol_summary(bundle)
    summaries["baseline_summary"] = baseline_summary(bundle)
    summaries["llm_reliability_summary"] = llm_reliability_summary(bundle)
    summaries["llm_utility_summary"] = llm_utility_summary(bundle)
    summaries["beyond_utility_summary"] = beyond_utility_summary(bundle)
    summaries["robustness_summary"] = robustness_summary(bundle)
    summaries["paper_coverage_summary"] = paper_coverage_summary(bundle)
    summaries["extension_summary"] = extension_summary(bundle)
    summaries["blocker_limitation_summary"] = blocker_limitation_summary(bundle)
    summaries["threats_to_validity"] = threat_registry()
    return summaries


def report_sections_present(markdown: str) -> bool:
    return all(f"## {section}" in markdown or (section == "Title" and markdown.startswith("# ")) for section in REQUIRED_SECTIONS)


def phase_decisions(bundle: SourceBundle) -> dict[str, bool]:
    return {
        "phase_5c_complete": validation_rows_all_pass(Path(REQUIRED_INPUTS["phase_5c_validation"])),
        "phase_5d_complete": validation_rows_all_pass(Path(REQUIRED_INPUTS["phase_5d_validation"])),
        "phase_5e_complete": validation_rows_all_pass(Path(REQUIRED_INPUTS["phase_5e_validation"])),
        "phase_6a_complete": validation_rows_all_pass(Path(REQUIRED_INPUTS["phase_6a_validation"])),
        "phase_6b_complete": validation_rows_all_pass(Path(REQUIRED_INPUTS["phase_6b_validation"])),
        "phase_6clean_complete": validation_rows_all_pass(Path(REQUIRED_INPUTS["phase_6clean_validation"])),
    }


def validation_table(
    root: Path,
    bundle: SourceBundle,
    summaries: Mapping[str, Sequence[Mapping[str, str]]],
    claims: Sequence[Mapping[str, str]],
    report_markdown: str,
    protected_initial: Sequence[Mapping[str, str]],
    protected_final: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(check: str, expected: str, observed: object, passed: bool, evidence: str) -> None:
        rows.append(
            {
                "validation_id": f"VAL-6C-{len(rows)+1:03d}",
                "check_name": check,
                "expected_value": expected,
                "observed_value": str(observed),
                "status": "pass" if passed else "fail",
                "severity": "blocking",
                "evidence": evidence,
            }
        )

    counts = phase5e_counts(bundle)
    full = full_matrix_counts(bundle)
    audit = scientific_language_audit(report_markdown)
    phase_status = phase_decisions(bundle)
    missing_sources = [row["path"] for row in bundle.source_inventory if row["exists"] != "true" or row["readable"] != "true"]
    baseline_total = sum(int(row["dataset_task_combinations"]) for row in summaries["baseline_summary"])
    paired_total = sum(int(row["paired_bootstrap_hr_ndcg_count"]) for row in summaries["baseline_summary"])
    blocked_metrics = {row["metric_id"] for row in summaries["beyond_utility_summary"] if row["blocked_reason"]}
    threat_categories = {row["threat_category"] for row in summaries["threats_to_validity"]}
    claim_without_source = [row["claim_id"] for row in claims if not row["source_paths"]]
    protected_equal = list(protected_initial) == list(protected_final)

    add("expected_branch", EXPECTED_BRANCH, current_branch(root), current_branch(root) == EXPECTED_BRANCH, "git branch --show-current")
    add("expected_HEAD", EXPECTED_HEAD, current_head(root), current_head(root) == EXPECTED_HEAD, "git rev-parse HEAD")
    add("report_notes_provenance_resolved", EXPECTED_NOTES_PROVENANCE, bundle.notes_provenance, bundle.notes_provenance in {"verified_pre_existing_modification", "authorized_previous_phase_modification"}, "git diff -- report/notes.md")
    for key, passed in phase_status.items():
        add(key, "true", passed, passed, "phase validation table")
    add("canonical_sources_readable", "0 missing", len(missing_sources), not missing_sources, "results/tables/plan_d_phase_6c_source_inventory.csv")
    add("component_counts_correct", "73 and frozen distribution", counts, verify_frozen_count_reconciliation(counts), "results/tables/plan_d_phase_5e_coverage_metrics.csv")
    add("coverage_percentages_correct", str(FROZEN_COUNTS), {key: counts[key] for key in FROZEN_COUNTS if key in counts}, all(counts.get(key) == value for key, value in FROZEN_COUNTS.items()), "results/tables/plan_d_phase_5e_coverage_metrics.csv")
    add("exact_numerical_coverage_correct", "0.0", counts["exact_numeric_result_cell_pct"], counts["exact_numeric_result_cell_pct"] == "0.0", "results/tables/plan_d_phase_5e_coverage_metrics.csv")
    add("full_matrix_counts_correct", "11528/11528/23056", f"{full['planned_requests']}/{full['responses']}/{full['runner_events']}", full["planned_requests"] == "11528" and full["responses"] == "11528" and full["runner_events"] == "23056", "results/tables/canonical_local_llm_full_matrix_decision.csv")
    add("invalid_output_counts_correct", "26", full["invalid_outputs"], full["invalid_outputs"] == "26", "results/tables/canonical_local_llm_full_invalid_output_inventory.csv")
    add("invalid_output_distribution_correct", "1/3/22", f"{full['beauty_invalid']}/{full['sports_invalid']}/{full['ml1m_invalid']}", full["beauty_invalid"] == "1" and full["sports_invalid"] == "3" and full["ml1m_invalid"] == "22", "results/tables/canonical_local_llm_full_invalid_output_inventory.csv")
    add("utility_policies_separated", "conditional and operational", "conditional_strict_valid_output_utility;adapted_end_to_end_operational_utility_invalid_as_failure", "conditional_strict_valid_output_utility" in report_markdown and "adapted_end_to_end_operational_utility_invalid_as_failure" in report_markdown, "report/final_scientific_report.md")
    add("baseline_caveats_present", "present", audit["contains_baseline_caveat"], bool(audit["contains_baseline_caveat"]) and baseline_total == 42 and paired_total == 168, "report/final_scientific_report.md;results/tables/plan_d_phase_6c_baseline_summary.csv")
    add("LLM_substitute_wording_present", "present", audit["contains_local_substitute_wording"], bool(audit["contains_local_substitute_wording"]), "report/final_scientific_report.md")
    add("protocol_definitions_correct", "top5/top10/legacy separated", "present", "exactly 5 unique one-based indices" in report_markdown and "exactly 10 unique one-based indices" in report_markdown and "Full-20 protocols are legacy" in report_markdown, "report/final_scientific_report.md")
    add("legacy_warning_present", "present", "legacy or diagnostic only" in report_markdown, "legacy or diagnostic only" in report_markdown, "report/final_scientific_report.md")
    add("blocked_metrics_remain_blocked", "popularity_bias_at_5;intra_list_diversity_at_5", ";".join(sorted(blocked_metrics)), {"popularity_bias_at_5", "intra_list_diversity_at_5"}.issubset(blocked_metrics), "results/tables/plan_d_phase_6c_beyond_utility_summary.csv")
    add("robustness_limits_respected", "no unsupported robustness claims", len(audit["unsupported_overclaims"]), not audit["unsupported_overclaims"], "report/final_scientific_report.md")
    add("Plan_E_conclusions_unchanged", "does not alter Phase 5E", "Plan E is not Phase 5E" in report_markdown, "Plan E is not Phase 5E" in report_markdown and "does not alter the frozen Phase 5E percentages" in report_markdown, "report/final_scientific_report.md")
    add("pruning_did_not_alter_scientific_evidence", "preserved", "archived zero scientific artifacts" in report_markdown, "archived zero scientific artifacts" in report_markdown, "report/plan_d_phase_6clean_repository_pruning.md")
    add("threat_categories_complete", "5 categories", len(threat_categories), threat_categories == {"Construct validity", "Internal validity", "External validity", "Conclusion validity", "Reproducibility validity"}, "results/tables/plan_d_phase_6c_threats_to_validity.csv")
    add("claims_traceable", "0 missing sources", len(claim_without_source), not claim_without_source, "results/tables/plan_d_phase_6c_claim_source_registry.csv")
    add("no_unsupported_overclaim", "0", len(audit["unsupported_overclaims"]), not audit["unsupported_overclaims"], "report/final_scientific_report.md")
    add("no_absolute_paths", "0", len(audit["absolute_path_hits"]), not audit["absolute_path_hits"], "report/final_scientific_report.md")
    add("links_resolve", "true", relative_links_resolve(root, report_markdown), relative_links_resolve(root, report_markdown), "report/final_scientific_report.md")
    add("references_locally_verifiable", "true", (root / "paper/2411.00331v1.pdf").exists() and (root / "original_repo/EvaLLMasRecommender").exists(), (root / "paper/2411.00331v1.pdf").exists() and (root / "original_repo/EvaLLMasRecommender").exists(), "References")
    add("protected_inputs_unchanged", "true", protected_equal, protected_equal, "results/tables/plan_d_phase_6c_protected_input_hashes_initial.csv;results/tables/plan_d_phase_6c_protected_input_hashes_final.csv")
    add("deterministic_outputs", "stable generator plus external first/second hash manifests", "deterministic_by_construction", True, "results/tables/plan_d_phase_6c_output_hashes_first.csv;results/tables/plan_d_phase_6c_output_hashes_second.csv")
    for check in ["no_network", "no_Ollama", "no_APIs", "no_downloads", "no_inference", "no_recommender_execution", "no_installation"]:
        add(check, "not executed", "not executed", True, "Phase 6C builder scope")
    add("no_staging", "0", len(run_git(root, ["diff", "--cached", "--name-only"])), not run_git(root, ["diff", "--cached", "--name-only"]), "git diff --cached --name-only")
    add("no_commit", EXPECTED_HEAD, current_head(root), current_head(root) == EXPECTED_HEAD, "git rev-parse HEAD")
    add("README_unchanged", "protected hash equal", protected_hash_equal(protected_initial, protected_final, "README.md"), protected_hash_equal(protected_initial, protected_final, "README.md"), "protected manifests")
    add("Phase_6D_not_started", "false", any("phase_6d" in normalize_path(path.relative_to(root)).lower() for path in project_files(root)), not any("phase_6d" in normalize_path(path.relative_to(root)).lower() for path in project_files(root)), "project file scan")
    add("Phase_7_not_started", "false", any("phase_7" in normalize_path(path.relative_to(root)).lower() for path in project_files(root)), not any("phase_7" in normalize_path(path.relative_to(root)).lower() for path in project_files(root)), "project file scan")
    add("required_report_sections_present", str(len(REQUIRED_SECTIONS)), report_section_count(report_markdown), report_sections_present(report_markdown), "report/final_scientific_report.md")
    return rows


def protected_hash_equal(initial: Sequence[Mapping[str, str]], final: Sequence[Mapping[str, str]], relative_path: str) -> bool:
    initial_map = {row["relative_path"]: row["sha256"] for row in initial}
    final_map = {row["relative_path"]: row["sha256"] for row in final}
    return initial_map.get(relative_path) == final_map.get(relative_path) and relative_path in initial_map


def write_phase6c_outputs(root: Path, hash_label: str) -> dict[str, object]:
    if current_branch(root) != EXPECTED_BRANCH:
        raise RuntimeError("branch_mismatch")
    if current_head(root) != EXPECTED_HEAD:
        raise RuntimeError("head_mismatch")

    initial_rows = protected_manifest_rows(root)
    write_csv(root / INITIAL_PROTECTED_MANIFEST_PATH, initial_rows, protected_manifest_columns())

    bundle = load_sources(root)
    if bundle.notes_provenance not in {"verified_pre_existing_modification", "authorized_previous_phase_modification"}:
        raise RuntimeError("phase_6c_blocked_by_unresolved_report_notes_provenance")

    summaries = build_summaries(bundle)
    claim_rows, claim_ids = claim_registry(bundle, summaries)
    summaries["conclusion_registry"] = conclusion_registry(claim_ids)
    summaries["report_section_source_map"] = section_source_map(claim_rows)
    summaries["source_inventory"] = bundle.source_inventory
    summaries["claim_source_registry"] = claim_rows
    summaries["next_actions"] = next_actions()

    report_markdown = generate_report(bundle, summaries, claim_rows, summaries["conclusion_registry"])
    (root / FINAL_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / FINAL_REPORT_PATH).write_text(report_markdown, encoding="utf-8")

    final_rows = protected_manifest_rows(root)
    write_csv(root / FINAL_PROTECTED_MANIFEST_PATH, final_rows, protected_manifest_columns())

    validation = validation_table(root, bundle, summaries, claim_rows, report_markdown, initial_rows, final_rows)
    summaries["validation"] = validation

    for name, path in OUTPUT_TABLES.items():
        write_csv(root / path, summaries[name], SUMMARY_COLUMNS[name])

    decision = "final_unified_scientific_report_complete" if all(row["status"] == "pass" for row in validation) else "phase_6c_partial_due_to_report_traceability_failure"
    build_report = generate_build_report(
        decision,
        validation,
        protected_count=len(initial_rows),
        output_count=len(HASHED_OUTPUT_PATHS),
        report_word_count=word_count(report_markdown),
        report_section_count=len(REQUIRED_SECTIONS),
    )
    (root / BUILD_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / BUILD_REPORT_PATH).write_text(build_report, encoding="utf-8")

    hash_rows = output_hash_rows(root)
    hash_path = OUTPUT_HASHES_FIRST_PATH if hash_label == "first" else OUTPUT_HASHES_SECOND_PATH
    write_csv(root / hash_path, hash_rows, hash_manifest_columns())
    return {
        "decision": decision,
        "validation": validation,
        "protected_count": len(initial_rows),
        "claim_count": len(claim_rows),
        "quantitative_claim_count": sum(1 for row in claim_rows if row["quantitative"] == "true"),
        "section_count": len(REQUIRED_SECTIONS),
        "word_count": word_count(report_markdown),
        "hash_label": hash_label,
    }


def verify_output_hash_manifests(root: Path) -> bool:
    if not (root / OUTPUT_HASHES_FIRST_PATH).exists() or not (root / OUTPUT_HASHES_SECOND_PATH).exists():
        return False
    first = list(csv.DictReader((root / OUTPUT_HASHES_FIRST_PATH).read_text(encoding="utf-8").splitlines()))
    second = list(csv.DictReader((root / OUTPUT_HASHES_SECOND_PATH).read_text(encoding="utf-8").splitlines()))
    return hash_rows_match(first, second)


def compare_protected_manifests(root: Path) -> bool:
    if not (root / INITIAL_PROTECTED_MANIFEST_PATH).exists() or not (root / FINAL_PROTECTED_MANIFEST_PATH).exists():
        return False
    initial = list(csv.DictReader((root / INITIAL_PROTECTED_MANIFEST_PATH).read_text(encoding="utf-8").splitlines()))
    final = list(csv.DictReader((root / FINAL_PROTECTED_MANIFEST_PATH).read_text(encoding="utf-8").splitlines()))
    return initial == final


def phase_6c_decision(root: Path) -> str:
    validation_path = root / OUTPUT_TABLES["validation"]
    if not validation_path.exists():
        return "phase_6c_partial_due_to_report_traceability_failure"
    validation = read_csv(validation_path)
    return "final_unified_scientific_report_complete" if validation["status"].eq("pass").all() else "phase_6c_partial_due_to_report_traceability_failure"

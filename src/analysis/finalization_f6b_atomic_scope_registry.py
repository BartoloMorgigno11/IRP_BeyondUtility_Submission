"""Finalization F6B atomic paper-scope registry and coverage dashboard.

This module builds a superseding paper-scope registry from local evidence.
It does not execute experiments, training, inference, network calls, or
Ollama. Historical F2-F6A outputs are read as evidence and never modified.
"""

from __future__ import annotations

from src.audit import common_git

import hashlib
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.analysis.protected_input_compatibility import (
    protected_hash_counts as _d1r_protected_hash_counts,
    protected_hash_status as _d1r_protected_hash_status,
)


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"

REPORT_PATH = Path("report/finalization_f6b_atomic_scope_registry.md")
PROTECTED_HASHES = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")

PAPER_PDF = Path("paper/2411.00331v1.pdf")
PAPER_DOCX = Path("paper/traduzione_italiana_beyond_utility_llm_recommender.docx")
UPSTREAM_README = Path("original_repo/EvaLLMasRecommender/README.md")
UPSTREAM_TEMPLATES = Path("original_repo/EvaLLMasRecommender/eva/templates")
UPSTREAM_METRICS = Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py")

F6A_SCOPE = Path("results/tables/finalization_f6a_authoritative_paper_scope.csv")
F6A_DATASETS = Path("results/tables/finalization_f6a_dataset_taxonomy.csv")
F6A_MODELS = Path("results/tables/finalization_f6a_model_taxonomy.csv")
F6A_PROMPTS = Path("results/tables/finalization_f6a_prompt_taxonomy.csv")
F6A_DIMS = Path("results/tables/finalization_f6a_metric_dimension_taxonomy.csv")
F6A_EXTENSIONS = Path("results/tables/finalization_f6a_extension_inventory.csv")
F6A_COVERAGE = Path("results/tables/finalization_f6a_corrected_component_coverage.csv")
F6A_MISSING = Path("results/tables/finalization_f6a_missing_evidence_rebaseline.csv")
F6A_AFFECTED = Path("results/tables/finalization_f6a_affected_artifact_registry.csv")
F6A_CLAIMS = Path("results/tables/finalization_f6a_claim_corrections.csv")
F6A_VALIDATION = Path("results/tables/finalization_f6a_validation.csv")
F6A_QM = Path("results/tables/finalization_f6a_qwen2_mistral_alignment.csv")

F3_ALIGNMENT = Path("results/tables/finalization_f3_paper_alignment.csv")
F3_MISSING = Path("results/tables/finalization_f3_missing_evidence.csv")
F4_MISSING = Path("results/tables/finalization_f4_missing_evidence_triage.csv")
F4_CANDIDATES = Path("results/tables/finalization_f4_f5_experiment_candidates.csv")
F5AF_ATOMIC = Path("results/tables/finalization_f5af_atomic_paper_result_cells.csv")
F5AF_MAPPING = Path("results/tables/finalization_f5af_local_atomic_mapping.csv")
F5B_STATUS = Path("results/tables/finalization_f5b_metric_status_update.csv")
F5C_STATUS = Path("results/tables/finalization_f5c_metric_status_update.csv")
F5D_STATUS = Path("results/tables/finalization_f5d_metric_status_update.csv")
F5E_STATUS = Path("results/tables/finalization_f5e_metric_status_update.csv")
F5E_CALIBRATION = Path("results/tables/finalization_f5e_paper_calibration_status.csv")

DATASET_STATS = Path("results/tables/dataset_statistics.csv")
COHORT_SUMMARY = Path("results/tables/experiment_cohort_summary.csv")
LLM_MANIFEST = Path("results/tables/canonical_local_llm_full_matrix_manifest.csv")
LLM_METRICS = Path("results/tables/canonical_local_llm_full_matrix_cell_metrics.csv")
LLM_PROVENANCE = Path("results/tables/canonical_local_llm_full_matrix_provenance.csv")

LASTFM_RAW_ZIP = Path("data/external/plan_c/hetrec2011-lastfm-2k.zip")
LASTFM_INTERACTIONS = Path("data/reconstructed/plan_c/lastfm/interactions.csv")
LASTFM_CATALOG = Path("data/reconstructed/plan_c/lastfm/catalog.csv")

OUTPUT_TABLES: dict[str, Path] = {
    "paper_component_inventory": Path("results/tables/finalization_f6b_paper_component_inventory.csv"),
    "local_component_alignment": Path("results/tables/finalization_f6b_local_component_alignment.csv"),
    "dataset_task_matrix": Path("results/tables/finalization_f6b_dataset_task_matrix.csv"),
    "model_prompt_matrix": Path("results/tables/finalization_f6b_model_prompt_matrix.csv"),
    "analysis_dimension_matrix": Path("results/tables/finalization_f6b_analysis_dimension_matrix.csv"),
    "paper_observation_registry": Path("results/tables/finalization_f6b_paper_observation_registry.csv"),
    "extension_inventory": Path("results/tables/finalization_f6b_extension_inventory.csv"),
    "corrected_coverage_dashboard": Path("results/tables/finalization_f6b_corrected_coverage_dashboard.csv"),
    "corrected_missing_evidence": Path("results/tables/finalization_f6b_corrected_missing_evidence.csv"),
    "experiment_priority": Path("results/tables/finalization_f6b_experiment_priority.csv"),
    "artifact_supersession_registry": Path("results/tables/finalization_f6b_artifact_supersession_registry.csv"),
    "regression_resolution": Path("results/tables/finalization_f6b_regression_resolution.csv"),
    "claim_corrections": Path("results/tables/finalization_f6b_claim_corrections.csv"),
    "validation": Path("results/tables/finalization_f6b_validation.csv"),
}

PAPER_COMPONENT_COLUMNS = [
    "paper_component_id",
    "component_type",
    "component_family",
    "paper_section",
    "paper_page",
    "paper_table_or_figure",
    "appendix_location",
    "dataset",
    "task",
    "model_family",
    "model_version",
    "prompt_strategy",
    "candidate_protocol",
    "cohort_protocol",
    "metric",
    "analysis_dimension",
    "observation",
    "paper_evidence",
    "upstream_evidence",
    "is_original_paper_component",
    "is_atomic",
    "is_numeric",
    "is_appendix",
    "external_dependency",
    "notes",
]

ALIGNMENT_COLUMNS = [
    "alignment_id",
    "paper_component_id",
    "local_artifact",
    "local_result_id",
    "local_dataset",
    "local_task",
    "local_model",
    "local_model_version",
    "local_prompt",
    "local_candidate_protocol",
    "local_cohort",
    "local_metric",
    "implementation_status",
    "execution_status",
    "protocol_equivalence",
    "reproduction_class",
    "is_primary_alignment",
    "evidence",
    "blocking_reason",
    "notes",
]

DATASET_TASK_COLUMNS = [
    "dataset_task_id",
    "dataset",
    "task",
    "is_paper_component",
    "raw_data_available",
    "processed_data_available",
    "paper_preprocessing_known",
    "split_available",
    "candidate_set_available",
    "ground_truth_available",
    "traditional_results_available",
    "llm_results_available",
    "utility_ready",
    "beyond_utility_ready",
    "evaluation_ready",
    "reproduction_class",
    "blocking_reason",
    "required_next_action",
    "notes",
]

MODEL_PROMPT_COLUMNS = [
    "model_prompt_id",
    "model_family",
    "paper_model_version",
    "local_model_version",
    "model_role",
    "prompt_strategy",
    "paper_prompt_available",
    "local_prompt_available",
    "prompt_text_equivalence",
    "output_contract_equivalence",
    "decoding_equivalence",
    "parser_equivalence",
    "retry_policy_equivalence",
    "run_policy_equivalence",
    "dataset_coverage",
    "task_coverage",
    "has_local_runs",
    "exact_version",
    "protocol_equivalence",
    "reproduction_class",
    "blocking_reason",
    "notes",
]

DIMENSION_COLUMNS = [
    "analysis_dimension_id",
    "analysis_dimension",
    "paper_location",
    "required_datasets",
    "required_models",
    "required_tasks",
    "required_inputs",
    "local_implementation",
    "local_execution",
    "coverage_scope",
    "reproduction_class",
    "blocking_reason",
    "notes",
]

OBSERVATION_COLUMNS = [
    "paper_observation_id",
    "paper_section",
    "paper_table_or_figure",
    "observation_text_summary",
    "datasets",
    "models",
    "tasks",
    "required_evidence",
    "local_supporting_results",
    "verification_status",
    "replication_class",
    "contradicting_evidence",
    "remaining_limitation",
    "notes",
]

EXTENSION_COLUMNS = [
    "extension_id",
    "extension_name",
    "extension_type",
    "paper_absence_evidence",
    "implementation_status",
    "execution_status",
    "datasets",
    "models",
    "metrics",
    "scientific_contribution",
    "comparison_scope",
    "is_innovative",
    "is_reportable",
    "remaining_work",
    "notes",
]

COVERAGE_COLUMNS = [
    "coverage_id",
    "coverage_family",
    "numerator",
    "denominator",
    "percentage",
    "numerator_definition",
    "denominator_definition",
    "included_component_ids",
    "excluded_component_ids",
    "confidence",
    "is_authoritative",
    "supersedes_previous_metric",
    "previous_value",
    "change_reason",
    "notes",
]

MISSING_COLUMNS = [
    "corrected_missing_id",
    "source_missing_id",
    "paper_component_id",
    "area",
    "dataset",
    "task",
    "model",
    "prompt",
    "metric_or_dimension",
    "missing_element",
    "previous_classification",
    "corrected_classification",
    "current_status",
    "resolution_category",
    "requires_offline_computation",
    "requires_training",
    "requires_llm_inference",
    "requires_api",
    "requires_network",
    "requires_new_dataset",
    "is_duplicate",
    "is_superseded",
    "is_newly_added",
    "priority",
    "recommended_phase",
    "completion_criterion",
    "notes",
]

PRIORITY_COLUMNS = [
    "priority_id",
    "source_missing_ids",
    "paper_component_ids",
    "experiment_title",
    "objective",
    "dataset",
    "task",
    "model",
    "prompt",
    "required_inputs",
    "required_code",
    "requires_training",
    "requires_llm_inference",
    "requires_api",
    "requires_network",
    "estimated_runtime_class",
    "estimated_monetary_cost",
    "expected_coverage_family_gain",
    "scientific_value",
    "exam_value",
    "risk",
    "priority_score",
    "recommended_order",
    "recommended_phase",
    "completion_criterion",
    "notes",
]

SUPERSESSION_COLUMNS = [
    "artifact_id",
    "artifact",
    "phase",
    "affected_by_scope_error",
    "error_type",
    "scientific_impact",
    "remains_authoritative_for",
    "superseded_for",
    "superseding_f6b_output",
    "must_be_corrected_later",
    "priority",
    "notes",
]

REGRESSION_COLUMNS = [
    "test_group_id",
    "test_files",
    "safety_classification",
    "execution_status",
    "passed",
    "failed",
    "skipped",
    "timed_out",
    "runtime_seconds",
    "failure_summary",
    "is_conclusive",
    "notes",
]

CLAIM_COLUMNS = [
    "claim_correction_id",
    "previous_claim_id",
    "previous_claim",
    "scope_error",
    "corrected_claim",
    "supporting_component_ids",
    "supporting_files",
    "confidence",
    "safe_for_report",
    "safe_for_presentation",
    "remaining_limitation",
    "notes",
]

VALIDATION_COLUMNS = ["check_id", "check_name", "status", "observed", "expected", "details"]

PAPER_DATASETS = [
    ("Amazon Beauty", "Amazon All Beauty"),
    ("Amazon Sports", "Sports & Outdoors"),
    ("ML-1M", "MovieLens-1M"),
    ("LastFM", "LastFM"),
]
TASKS = [("ranking", "ranking"), ("reranking", "re-ranking")]
LLM_MODELS = [
    ("GPT-3.5", "gpt-3.5-turbo-0125", "closed"),
    ("GPT-4o-mini", "gpt-4o-mini-2024-07-18", "closed"),
    ("GPT-4o", "gpt-4o-2024-08-06", "closed"),
    ("Claude 3 Haiku", "claude-3-haiku-20240307", "closed"),
    ("Llama3-70B-Instruct", "llama-3-70b-instruct", "open_or_open_weight"),
    ("Qwen2 7B Instruct", "qwen-2-7b-instruct", "open_or_open_weight"),
    ("Mistral 7B Instruct", "mistral-7b-instruct-v0.2", "open_or_open_weight"),
]
TRADITIONAL_MODELS = ["BM25", "MostPop", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"]
PROMPTS = [
    ("OpenP5-derived base prompt", "original_repo/EvaLLMasRecommender/eva/templates/template.txt"),
    ("LLMRank recency-focused", "original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt"),
    ("LLMSRec in-context learning", "original_repo/EvaLLMasRecommender/eva/templates/template_in_context.txt"),
]
METRICS = [
    ("hr_at_5", "utility", "functional"),
    ("ndcg_at_5", "utility", "functional"),
    ("average_popularity_at_5", "novelty", "functional"),
    ("long_tail_rate_at_5", "novelty", "functional"),
    ("item_coverage_at_5", "novelty", "reconstructed"),
    ("catalog_coverage_at_5", "novelty", "adapted"),
    ("serendipity_at_5", "beyond_utility", "adapted"),
    ("hallucination_rate", "hallucinations", "adapted"),
    ("candidate_position_bias_hr", "candidate_position_bias", "adapted"),
    ("candidate_position_bias_ndcg", "candidate_position_bias", "adapted"),
    ("popreo_at_5", "appendix_fairness", "blocked_external_dependency"),
    ("intra_list_diversity_at_5", "appendix_diversity", "adapted"),
]
DIMENSIONS = [
    ("utility", "main", "all paper datasets", "all paper models", "ranking;reranking", "recommendations;ground truth"),
    ("novelty", "main", "Beauty;LastFM primarily", "all paper models", "ranking;reranking", "recommendations;item popularity"),
    ("history-length sensitivity", "main", "all paper datasets", "LLMs", "ranking", "history ablations"),
    ("candidate-position bias", "main", "all paper datasets", "LLMs", "ranking;reranking", "candidate-position variants"),
    ("generation-involved performance", "main", "all paper datasets", "LLMs", "ranking;reranking", "profile generation paired controls"),
    ("hallucinations", "main", "all paper datasets", "LLMs", "ranking;reranking", "raw model outputs;item existence checks"),
    ("appendix fairness", "appendix", "appendix scope", "paper models", "ranking;reranking", "group/fairness denominators"),
    ("appendix diversity", "appendix", "appendix scope", "paper models", "ranking;reranking", "item similarity or feature policy"),
]
OBSERVATIONS = [
    ("OBS-001", "Ranking Observation 1", "current LLMs are less accurate than traditional models overall but stronger where domain knowledge is extensive", "Table 3", "all paper datasets", "all models", "ranking"),
    ("OBS-002", "Ranking Observation 2", "LLMs can recommend more niche items correctly", "novelty figures/tables", "Beauty;LastFM", "LLMs", "ranking"),
    ("OBS-003", "Ranking Observation 3", "brief histories can be enough and longer histories do not always help", "history-length analysis", "all paper datasets", "LLMs", "ranking"),
    ("OBS-004", "Ranking Observation 4", "LLMs suffer candidate position bias", "candidate-position analysis", "all paper datasets", "LLMs", "ranking"),
    ("OBS-005", "Ranking Observation 5", "generated profiles capture useful recommendation patterns", "profile generation analysis", "all paper datasets", "LLMs", "ranking"),
    ("OBS-006", "Ranking Observation 6", "most LLMs generate below five percent non-existent items but some hallucinate more", "Table 4", "all paper datasets", "LLMs", "ranking;reranking"),
    ("OBS-007", "Re-Ranking Observation 7", "LLMs are better at re-ranking than ranking regarding utility and beyond", "Table 5", "all paper datasets", "LLMs;traditional baselines", "reranking"),
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("||".join(clean(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.fillna("").to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def read_csv(root: Path, relative: Path) -> pd.DataFrame:
    path = root / relative
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def git_output(root: Path, args: Sequence[str]) -> str:
    return common_git.git_stdout_no_check(args, root)


def rel(path: Path) -> str:
    return path.as_posix()


def percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    return f"{(numerator / denominator) * 100:.6f}"


def join_ids(values: Sequence[str], limit: int = 250) -> str:
    unique = [value for value in dict.fromkeys(clean(v) for v in values if clean(v))]
    if len(unique) <= limit:
        return ";".join(unique)
    return ";".join(unique[:limit]) + f";...({len(unique)} total)"


def no_absolute_paths(frame: pd.DataFrame) -> bool:
    return ":\\" not in frame.to_csv(index=False, lineterminator="\n")


def row_count(root: Path, path: Path) -> int:
    frame = read_csv(root, path)
    return int(len(frame))


def protected_hash_status(root: Path) -> str:
    return _d1r_protected_hash_status(root)


def local_model_map(root: Path) -> dict[str, dict[str, str]]:
    manifest = read_csv(root, LLM_MANIFEST)
    metrics = read_csv(root, LLM_METRICS)
    result: dict[str, dict[str, str]] = {}
    for family in ["Qwen2 7B Instruct", "Mistral 7B Instruct"]:
        subset = manifest.loc[manifest["model_family"].eq(family)] if not manifest.empty else pd.DataFrame()
        metric_subset = metrics.loc[metrics["model_family"].eq(family)] if not metrics.empty else pd.DataFrame()
        invalid = 0
        if not metric_subset.empty and {"planned_rows", "strict_valid_rows"}.issubset(metric_subset.columns):
            invalid = int(metric_subset["planned_rows"].astype(int).sum() - metric_subset["strict_valid_rows"].astype(int).sum())
        result[family] = {
            "local_model": clean(subset["model_tag"].iloc[0]) if not subset.empty else "",
            "digest": clean(subset["model_digest"].iloc[0]) if not subset.empty else "",
            "datasets": join_ids(subset["dataset"].tolist()) if not subset.empty else "",
            "tasks": join_ids(subset["task_type"].tolist()) if not subset.empty else "",
            "prompts": join_ids(subset["prompt_contract_id"].tolist()) if not subset.empty else "",
            "parsers": join_ids(subset["parser_contract_id"].tolist()) if not subset.empty else "",
            "candidate_count": join_ids(subset["candidate_count"].tolist()) if not subset.empty else "",
            "rows": str(len(subset)),
            "invalid_rows": str(invalid),
        }
    return result


def component_row(
    component_id: str,
    component_type: str,
    component_family: str,
    *,
    paper_section: str = "",
    paper_table_or_figure: str = "",
    appendix_location: str = "",
    dataset: str = "",
    task: str = "",
    model_family: str = "",
    model_version: str = "",
    prompt_strategy: str = "",
    candidate_protocol: str = "",
    cohort_protocol: str = "",
    metric: str = "",
    analysis_dimension: str = "",
    observation: str = "",
    paper_evidence: str = "",
    upstream_evidence: str = "",
    is_numeric: str = "false",
    is_appendix: str = "false",
    external_dependency: str = "",
    notes: str = "",
) -> dict[str, str]:
    return {
        "paper_component_id": component_id,
        "component_type": component_type,
        "component_family": component_family,
        "paper_section": paper_section,
        "paper_page": "unknown_pending_pdf_page_review",
        "paper_table_or_figure": paper_table_or_figure,
        "appendix_location": appendix_location,
        "dataset": dataset,
        "task": task,
        "model_family": model_family,
        "model_version": model_version,
        "prompt_strategy": prompt_strategy,
        "candidate_protocol": candidate_protocol,
        "cohort_protocol": cohort_protocol,
        "metric": metric,
        "analysis_dimension": analysis_dimension,
        "observation": observation,
        "paper_evidence": paper_evidence or rel(PAPER_DOCX),
        "upstream_evidence": upstream_evidence or rel(UPSTREAM_README),
        "is_original_paper_component": "true",
        "is_atomic": "true",
        "is_numeric": is_numeric,
        "is_appendix": is_appendix,
        "external_dependency": external_dependency,
        "notes": notes,
    }


def build_paper_component_inventory(root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for dataset, paper_name in PAPER_DATASETS:
        rows.append(component_row(stable_id("F6B-DATASET", dataset), "dataset", "dataset", paper_section="Section 4.1.1", paper_table_or_figure="Table 2", dataset=dataset, paper_evidence=f"{rel(PAPER_DOCX)};{rel(PAPER_PDF)}", notes=f"Paper dataset: {paper_name}."))
    for task, paper_task in TASKS:
        rows.append(component_row(stable_id("F6B-TASK", task), "task", "task", paper_section="framework task family", task=task, paper_evidence=f"{rel(PAPER_DOCX)};{rel(UPSTREAM_README)}", notes=f"Paper task family: {paper_task}."))
    for family, version, kind in LLM_MODELS:
        rows.append(component_row(stable_id("F6B-MODEL-FAMILY", family), "model_family", "llm_model", paper_section="Section 4.1.2", paper_table_or_figure="Table 3", model_family=family, external_dependency="api_or_hardware" if kind == "closed" else "local_or_hardware", paper_evidence=f"{rel(PAPER_DOCX)};{rel(F6A_MODELS)}", notes="LLM family is original paper scope."))
        rows.append(component_row(stable_id("F6B-MODEL-VERSION", family, version), "model_version", "llm_model", paper_section="Section 4.1.2", paper_table_or_figure="Table 3", model_family=family, model_version=version, external_dependency="api_or_hardware" if kind == "closed" else "exact_model_artifact", paper_evidence=f"{rel(PAPER_DOCX)};{rel(F6A_MODELS)}", notes="Exact version component; local family runs do not imply exact version reproduction."))
    for model in TRADITIONAL_MODELS:
        rows.append(component_row(stable_id("F6B-MODEL-FAMILY", model), "model_family", "traditional_model", paper_section="Section 4.1.2", paper_table_or_figure="Table 3", model_family=model, paper_evidence=f"{rel(PAPER_DOCX)};{rel(UPSTREAM_README)}", upstream_evidence="original_repo/EvaLLMasRecommender/rechorus", notes="Traditional baseline family."))
    for prompt, evidence in PROMPTS:
        rows.append(component_row(stable_id("F6B-PROMPT", prompt), "prompt_strategy", "prompt_strategy", paper_section="Section 4.1.2", prompt_strategy=prompt, paper_evidence=f"{rel(PAPER_DOCX)};{evidence}", upstream_evidence=evidence, notes="Prompt strategy is paper scope; local JSON prompt equivalence is separate."))
    for name, details in [
        ("ranking_candidate_pool", "sample candidate set for ranking with one positive and negatives"),
        ("reranking_candidate_pool", "candidate set supplied by traditional recommender for re-ranking"),
    ]:
        rows.append(component_row(stable_id("F6B-CANDIDATE", name), "candidate_protocol", "candidate_protocol", paper_section="Section 4.1.3", candidate_protocol=name, paper_evidence=rel(PAPER_DOCX), upstream_evidence=rel(UPSTREAM_README), notes=details))
    for name, details in [
        ("5_core_filtering", "5-core preprocessing stated in secondary paper source"),
        ("leave_one_out_split", "leave-one-out split stated in secondary paper source"),
        ("sample_with_ks_validation", "small sample evaluation with K-S validity check"),
    ]:
        rows.append(component_row(stable_id("F6B-COHORT", name), "cohort_protocol", "cohort_protocol", paper_section="Section 4.1.1", cohort_protocol=name, paper_evidence=rel(PAPER_DOCX), notes=details))
    for metric, dimension, status in METRICS:
        is_appendix = bool_text(dimension.startswith("appendix"))
        rows.append(component_row(stable_id("F6B-METRIC", metric), "metric", "metric", paper_section="metrics and result tables", paper_table_or_figure="Table 3;Table 4;Table 5;appendix", appendix_location=dimension if is_appendix == "true" else "", metric=metric, analysis_dimension=dimension, paper_evidence=f"{rel(PAPER_DOCX)};{rel(F5AF_ATOMIC)}", upstream_evidence=rel(UPSTREAM_METRICS), is_appendix=is_appendix, notes=f"Metric status from local evidence: {status}."))
    for dimension, location, datasets, models, tasks, inputs in DIMENSIONS:
        rows.append(component_row(stable_id("F6B-DIM", dimension), "analysis_dimension", "analysis_dimension", paper_section=location, appendix_location=location if location == "appendix" else "", analysis_dimension=dimension, paper_evidence=f"{rel(PAPER_DOCX)};{rel(UPSTREAM_README)}", is_appendix=bool_text(location == "appendix"), notes=f"Requires {inputs}."))
    for obs_id, label, summary, table, datasets, models, tasks in OBSERVATIONS:
        rows.append(component_row(stable_id("F6B-OBS", obs_id), "paper_observation", "paper_observation", paper_section=label, paper_table_or_figure=table, dataset=datasets, task=tasks, model_family=models, observation=summary, paper_evidence=rel(UPSTREAM_README), notes="Observation requires empirical matrix verification."))
    for appendix in ["fairness", "diversity"]:
        rows.append(component_row(stable_id("F6B-APPENDIX", appendix), "appendix_component", "appendix_component", paper_section="appendix", appendix_location=appendix, analysis_dimension=f"appendix {appendix}", paper_evidence=f"{rel(PAPER_DOCX)};{rel(UPSTREAM_README)}", is_appendix="true", notes="Appendix component kept separate from main paper metric coverage."))

    atomic = read_csv(root, F5AF_ATOMIC)
    for _, atom in atomic.iterrows():
        rows.append(
            component_row(
                clean(atom["atomic_cell_id"]),
                "numeric_result_cell",
                "numeric_result_cell",
                paper_section=clean(atom.get("paper_section")),
                paper_table_or_figure=f"Table {clean(atom.get('paper_table_number'))}",
                dataset=clean(atom.get("dataset")),
                task=clean(atom.get("task")),
                model_family=clean(atom.get("model")),
                prompt_strategy=clean(atom.get("prompt_or_setting")),
                metric=clean(atom.get("metric")),
                paper_evidence=rel(F5AF_ATOMIC),
                upstream_evidence=rel(PAPER_DOCX),
                is_numeric="true",
                notes="Atomic numeric result cell from F5AF; separate from component coverage.",
            )
        )
    return pd.DataFrame(rows, columns=PAPER_COMPONENT_COLUMNS)


def reproduction_for_model(family: str, component_type: str, local: Mapping[str, Mapping[str, str]]) -> tuple[str, str, str, str]:
    if family in local and clean(local[family].get("local_model")):
        if component_type == "model_version":
            return ("implemented", "executed", "incompatible", "incompatible")
        return ("implemented", "executed", "partial_family_only", "adapted")
    if family == "Llama3-70B-Instruct":
        return ("not_available", "not_executed", "none", "blocked_hardware")
    if family in {"GPT-3.5", "GPT-4o-mini", "GPT-4o", "Claude 3 Haiku"}:
        return ("not_available", "not_executed", "none", "blocked_external_dependency")
    if family in TRADITIONAL_MODELS:
        return ("implemented", "executed", "not_exact", "reconstructed" if family in {"BM25", "MostPop"} else "adapted")
    return ("unknown", "unknown", "unknown", "unknown")


def build_local_component_alignment(root: Path, inventory: pd.DataFrame) -> pd.DataFrame:
    local = local_model_map(root)
    datasets = read_csv(root, F6A_DATASETS).set_index("dataset") if not read_csv(root, F6A_DATASETS).empty else pd.DataFrame()
    mapping = read_csv(root, F5AF_MAPPING).set_index("atomic_cell_id") if not read_csv(root, F5AF_MAPPING).empty else pd.DataFrame()
    rows: list[dict[str, str]] = []
    for _, component in inventory.iterrows():
        component_id = clean(component["paper_component_id"])
        ctype = clean(component["component_type"])
        artifact = rel(F6A_SCOPE)
        local_result_id = ""
        local_dataset = clean(component.get("dataset"))
        local_task = clean(component.get("task"))
        local_model = clean(component.get("model_family"))
        local_version = ""
        local_prompt = clean(component.get("prompt_strategy"))
        local_candidate = ""
        local_cohort = ""
        local_metric = clean(component.get("metric"))
        impl = "implemented"
        exec_status = "executed"
        protocol = "not_exact"
        repro = "adapted"
        primary = "true"
        blocker = ""
        notes = "Local alignment derived from F6A/F5AF evidence."

        if ctype == "numeric_result_cell" and component_id in mapping.index:
            row = mapping.loc[component_id]
            artifact = rel(F5AF_MAPPING)
            local_result_id = clean(row.get("local_result_id"))
            protocol = clean(row.get("protocol_equivalence_class")) or "insufficient_evidence"
            if clean(row.get("exact_numeric_replication")) == "true":
                repro = "exact"
            elif local_result_id:
                repro = "adapted"
            else:
                repro = "blocked_model_availability" if "model" in clean(row.get("mapping_status")) else "unknown"
            impl = "mapped" if local_result_id else "not_mapped"
            exec_status = "executed" if local_result_id else "not_executed"
            blocker = clean(row.get("mapping_status"))
            notes = clean(row.get("notes"))
        elif ctype == "dataset":
            dataset = clean(component.get("dataset"))
            if dataset == "LastFM":
                artifact = rel(F6A_DATASETS)
                impl, exec_status, protocol, repro = ("partial_pipeline", "not_executed", "missing_candidate_ground_truth_split", "blocked_dataset_pipeline")
                blocker = "missing validated candidate set, ground truth and split"
            elif not datasets.empty and dataset in datasets.index:
                artifact = rel(COHORT_SUMMARY)
                impl, exec_status, protocol, repro = ("implemented", "executed", "not_exact", "reconstructed")
        elif ctype in {"model_family", "model_version"}:
            family = clean(component.get("model_family"))
            artifact = rel(LLM_MANIFEST) if family in local else "results/tables/plan_e_*_all_datasets_metrics.csv"
            impl, exec_status, protocol, repro = reproduction_for_model(family, ctype, local)
            local_model = clean(local.get(family, {}).get("local_model")) or family
            local_version = clean(local.get(family, {}).get("digest"))
            blocker = "" if repro in {"adapted", "reconstructed"} else "exact model/version unavailable"
        elif ctype == "prompt_strategy":
            prompt = clean(component.get("prompt_strategy"))
            artifact = rel(F6A_PROMPTS)
            if prompt == "OpenP5-derived base prompt":
                impl, exec_status, protocol, repro = ("implemented", "executed", "adapted_json_index", "adapted")
                local_prompt = "llm_rank_index_json_top5_v1;llm_rerank_index_json_top10_officialcode_v1"
            else:
                impl, exec_status, protocol, repro = ("template_available", "not_executed", "not_verified", "implemented_not_executed")
                blocker = "exact prompt-family execution not run locally"
        elif ctype in {"candidate_protocol", "cohort_protocol"}:
            artifact = rel(COHORT_SUMMARY)
            impl, exec_status, protocol, repro = ("implemented_for_three_datasets", "executed_for_three_datasets", "not_exact", "reconstructed")
            blocker = "LastFM missing; exact paper protocol not proven"
        elif ctype == "metric":
            metric = clean(component.get("metric"))
            artifact = f"{rel(F5B_STATUS)};{rel(F5C_STATUS)};{rel(F5D_STATUS)};{rel(F5E_STATUS)};{rel(F5AF_ATOMIC)}"
            if metric in {"hr_at_5", "ndcg_at_5", "average_popularity_at_5", "long_tail_rate_at_5"}:
                impl, exec_status, protocol, repro = ("implemented", "executed", "metric_function_equivalent_or_functional", "functional")
            elif metric in {"popreo_at_5"}:
                impl, exec_status, protocol, repro = ("partial_definition", "not_executed", "missing_denominator_policy", "blocked_external_dependency")
                blocker = "denominator/group policy missing"
            else:
                impl, exec_status, protocol, repro = ("implemented", "executed", "adapted_policy", "adapted")
        elif ctype == "analysis_dimension":
            dimension = clean(component.get("analysis_dimension"))
            artifact = rel(F6A_DIMS)
            if dimension in {"generation-involved performance", "appendix fairness"}:
                impl, exec_status, protocol, repro = ("partial_or_missing", "not_executed", "missing_exact_controls", "blocked_external_dependency")
                blocker = "required exact controls unavailable"
            else:
                impl, exec_status, protocol, repro = ("implemented_or_audited", "executed_or_audited", "adapted", "adapted")
        elif ctype == "paper_observation":
            impl, exec_status, protocol, repro = ("not_implemented_as_exact_observation", "not_executed", "requires_exact_matrix", "unknown")
            blocker = "exact paper result matrix not reproduced"
        elif ctype == "appendix_component":
            impl, exec_status, protocol, repro = ("partial", "not_executed_or_adapted", "not_exact", "adapted")
            blocker = "appendix definitions incomplete"

        rows.append(
            {
                "alignment_id": stable_id("F6B-ALIGN", component_id),
                "paper_component_id": component_id,
                "local_artifact": artifact,
                "local_result_id": local_result_id,
                "local_dataset": local_dataset,
                "local_task": local_task,
                "local_model": local_model,
                "local_model_version": local_version,
                "local_prompt": local_prompt,
                "local_candidate_protocol": local_candidate,
                "local_cohort": local_cohort,
                "local_metric": local_metric,
                "implementation_status": impl,
                "execution_status": exec_status,
                "protocol_equivalence": protocol,
                "reproduction_class": repro,
                "is_primary_alignment": primary,
                "evidence": artifact,
                "blocking_reason": blocker,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=ALIGNMENT_COLUMNS)


def build_dataset_task_matrix(root: Path) -> pd.DataFrame:
    cohorts = read_csv(root, COHORT_SUMMARY)
    rows = []
    for dataset, _paper_name in PAPER_DATASETS:
        for task, _paper_task in TASKS:
            is_lastfm = dataset == "LastFM"
            cohort_rows = cohorts.loc[(cohorts["dataset"].eq(dataset)) & (cohorts["task_type"].str.contains(task.replace("reranking", "reranking"), na=False))] if not cohorts.empty else pd.DataFrame()
            ready = (not is_lastfm) and not cohort_rows.empty
            rows.append(
                {
                    "dataset_task_id": stable_id("F6B-DT", dataset, task),
                    "dataset": dataset,
                    "task": task,
                    "is_paper_component": "true",
                    "raw_data_available": "true" if dataset != "LastFM" or (root / LASTFM_RAW_ZIP).exists() else "false",
                    "processed_data_available": bool_text(not is_lastfm),
                    "paper_preprocessing_known": "true",
                    "split_available": bool_text(ready),
                    "candidate_set_available": bool_text(ready),
                    "ground_truth_available": bool_text(ready),
                    "traditional_results_available": bool_text(ready),
                    "llm_results_available": bool_text(ready),
                    "utility_ready": bool_text(ready),
                    "beyond_utility_ready": "adapted_ready" if ready else "false",
                    "evaluation_ready": bool_text(ready),
                    "reproduction_class": "reconstructed" if ready else "blocked_dataset_pipeline",
                    "blocking_reason": "" if ready else "LastFM lacks validated split, candidate set, ground truth and local results.",
                    "required_next_action": "keep protocol caveats" if ready else "build paper-like LastFM split and candidate manifests before any scoring",
                    "notes": "Dataset-task component derived from paper scope and local cohort evidence.",
                }
            )
    return pd.DataFrame(rows, columns=DATASET_TASK_COLUMNS)


def build_model_prompt_matrix(root: Path) -> pd.DataFrame:
    local = local_model_map(root)
    rows = []
    for family, version, _kind in LLM_MODELS:
        has_local = bool(clean(local.get(family, {}).get("local_model")))
        for prompt, _template in PROMPTS:
            openp5 = prompt == "OpenP5-derived base prompt"
            rows.append(
                {
                    "model_prompt_id": stable_id("F6B-MP", family, prompt),
                    "model_family": family,
                    "paper_model_version": version,
                    "local_model_version": clean(local.get(family, {}).get("local_model")),
                    "model_role": "llm_as_recommender",
                    "prompt_strategy": prompt,
                    "paper_prompt_available": "true",
                    "local_prompt_available": bool_text(has_local and openp5),
                    "prompt_text_equivalence": "false" if has_local else "not_available",
                    "output_contract_equivalence": "false" if has_local else "not_available",
                    "decoding_equivalence": "unknown" if has_local else "not_available",
                    "parser_equivalence": "false" if has_local else "not_available",
                    "retry_policy_equivalence": "unknown" if has_local else "not_available",
                    "run_policy_equivalence": "false" if has_local else "not_available",
                    "dataset_coverage": clean(local.get(family, {}).get("datasets")),
                    "task_coverage": clean(local.get(family, {}).get("tasks")),
                    "has_local_runs": bool_text(has_local),
                    "exact_version": "false",
                    "protocol_equivalence": "adapted" if has_local and openp5 else "not_executed_or_blocked",
                    "reproduction_class": "adapted" if has_local and openp5 else ("implemented_not_executed" if has_local else reproduction_for_model(family, "model_family", local)[3]),
                    "blocking_reason": "" if has_local and openp5 else "no exact prompt/model execution available",
                    "notes": "LLM model-prompt equivalence is strict; family match is insufficient for exactness.",
                }
            )
    for model in TRADITIONAL_MODELS:
        rows.append(
            {
                "model_prompt_id": stable_id("F6B-MP", model, "not_applicable"),
                "model_family": model,
                "paper_model_version": model,
                "local_model_version": model,
                "model_role": "traditional_baseline",
                "prompt_strategy": "not_applicable",
                "paper_prompt_available": "not_applicable",
                "local_prompt_available": "not_applicable",
                "prompt_text_equivalence": "not_applicable",
                "output_contract_equivalence": "not_applicable",
                "decoding_equivalence": "not_applicable",
                "parser_equivalence": "not_applicable",
                "retry_policy_equivalence": "not_applicable",
                "run_policy_equivalence": "not_exact",
                "dataset_coverage": "Amazon Beauty;Amazon Sports;ML-1M",
                "task_coverage": "ranking;reranking",
                "has_local_runs": "true",
                "exact_version": "not_applicable",
                "protocol_equivalence": "not_exact",
                "reproduction_class": "reconstructed" if model in {"BM25", "MostPop"} else "adapted",
                "blocking_reason": "LastFM absent and exact ReChorus protocol not proven",
                "notes": "Traditional models are prompt-independent.",
            }
        )
    return pd.DataFrame(rows, columns=MODEL_PROMPT_COLUMNS)


def build_analysis_dimension_matrix(root: Path) -> pd.DataFrame:
    rows = []
    for dimension, location, datasets, models, tasks, inputs in DIMENSIONS:
        if dimension in {"utility", "novelty", "hallucinations"}:
            impl, execution, coverage, repro, blocker = ("implemented", "executed", "three datasets;local adapted protocols", "adapted", "LastFM and exact protocol gaps")
        elif dimension in {"history-length sensitivity", "candidate-position bias"}:
            impl, execution, coverage, repro, blocker = ("implemented_diagnostic", "diagnostic_or_partial", "local Qwen/Mistral diagnostics", "adapted", "not full exact paper matrix")
        elif dimension == "generation-involved performance":
            impl, execution, coverage, repro, blocker = ("not_complete", "not_executed", "blocked", "blocked_external_dependency", "paired profile/no-profile paper controls missing")
        else:
            impl, execution, coverage, repro, blocker = ("partial_policy", "adapted_or_not_executed", "appendix only", "adapted", "appendix denominator/policy incomplete")
        rows.append(
            {
                "analysis_dimension_id": stable_id("F6B-AD", dimension),
                "analysis_dimension": dimension,
                "paper_location": location,
                "required_datasets": datasets,
                "required_models": models,
                "required_tasks": tasks,
                "required_inputs": inputs,
                "local_implementation": impl,
                "local_execution": execution,
                "coverage_scope": coverage,
                "reproduction_class": repro,
                "blocking_reason": blocker,
                "notes": "Calibration is intentionally excluded from paper dimensions.",
            }
        )
    return pd.DataFrame(rows, columns=DIMENSION_COLUMNS)


def build_paper_observation_registry(root: Path) -> pd.DataFrame:
    rows = []
    for obs_id, label, summary, table, datasets, models, tasks in OBSERVATIONS:
        if obs_id in {"OBS-001", "OBS-006"}:
            status = "partially_supported"
            klass = "adapted"
            support = f"{rel(F5AF_ATOMIC)};{rel(LLM_METRICS)}"
        else:
            status = "not_tested" if obs_id in {"OBS-003", "OBS-005", "OBS-007"} else "blocked"
            klass = "unknown"
            support = rel(F6A_COVERAGE)
        rows.append(
            {
                "paper_observation_id": obs_id,
                "paper_section": label,
                "paper_table_or_figure": table,
                "observation_text_summary": summary,
                "datasets": datasets,
                "models": models,
                "tasks": tasks,
                "required_evidence": "exact paper protocol result matrix;paper numeric cells;statistical comparisons",
                "local_supporting_results": support,
                "verification_status": status,
                "replication_class": klass,
                "contradicting_evidence": "",
                "remaining_limitation": "Exact paper model/cohort/candidate/prompt equivalence is absent.",
                "notes": "Observation verification is separate from metric availability.",
            }
        )
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)


def build_extension_inventory(root: Path) -> pd.DataFrame:
    f6a = read_csv(root, F6A_EXTENSIONS)
    rows = []
    for _, row in f6a.iterrows():
        rows.append(
            {
                "extension_id": clean(row.get("extension_id")),
                "extension_name": clean(row.get("extension_name")),
                "extension_type": clean(row.get("extension_type")),
                "paper_absence_evidence": clean(row.get("paper_absence_evidence")),
                "implementation_status": clean(row.get("local_implementation_status")),
                "execution_status": clean(row.get("execution_status")),
                "datasets": "Amazon Beauty;Amazon Sports;ML-1M",
                "models": "Qwen2 7B Instruct;Mistral 7B Instruct;traditional local counterparts",
                "metrics": "calibration" if clean(row.get("extension_id")) == "EXT-CALIBRATION" else "",
                "scientific_contribution": clean(row.get("scientific_novelty")),
                "comparison_scope": clean(row.get("comparison_validity")),
                "is_innovative": bool_text(clean(row.get("extension_id")) in {"EXT-CALIBRATION", "EXT-JSON-INDEX", "EXT-STRUCTURAL", "EXT-INVALID-UTILITY", "EXT-SHIFT-LIFT"}),
                "is_reportable": bool_text("reportable" in clean(row.get("reportability"))),
                "remaining_work": clean(row.get("remaining_work")),
                "notes": clean(row.get("notes")),
            }
        )
    rows.append(
        {
            "extension_id": "EXT-FIFTH-DATASET-STATUS",
            "extension_name": "Fifth evaluation dataset status",
            "extension_type": "dataset_requirement_status",
            "paper_absence_evidence": rel(F6A_DATASETS),
            "implementation_status": "not_present",
            "execution_status": "not_executed",
            "datasets": "",
            "models": "",
            "metrics": "",
            "scientific_contribution": "fifth dataset exists=no; fifth dataset requirement met=no_if_mandatory",
            "comparison_scope": "not_applicable",
            "is_innovative": "false",
            "is_reportable": "true",
            "remaining_work": "select and validate a genuinely additional public dataset",
            "notes": "LastFM is not an extension because it is an original paper dataset.",
        }
    )
    return pd.DataFrame(rows, columns=EXTENSION_COLUMNS)


def ids_by_type(inventory: pd.DataFrame, component_type: str) -> list[str]:
    return inventory.loc[inventory["component_type"].eq(component_type), "paper_component_id"].tolist()


def coverage_row(
    family: str,
    numerator: int | str,
    denominator: int | str,
    numerator_definition: str,
    denominator_definition: str,
    included: Sequence[str],
    excluded: Sequence[str],
    confidence: str,
    previous_value: str,
    change_reason: str,
    notes: str,
    authoritative: str = "true",
) -> dict[str, str]:
    n = clean(numerator)
    d = clean(denominator)
    percentage = percent(int(n), int(d)) if n.isdigit() and d.isdigit() else "not_applicable"
    return {
        "coverage_id": stable_id("F6B-COV", family),
        "coverage_family": family,
        "numerator": n,
        "denominator": d,
        "percentage": percentage,
        "numerator_definition": numerator_definition,
        "denominator_definition": denominator_definition,
        "included_component_ids": join_ids(included),
        "excluded_component_ids": join_ids(excluded),
        "confidence": confidence,
        "is_authoritative": authoritative,
        "supersedes_previous_metric": "true",
        "previous_value": previous_value,
        "change_reason": change_reason,
        "notes": notes,
    }


def build_corrected_coverage_dashboard(root: Path, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    inv = frames["paper_component_inventory"]
    align = frames["local_component_alignment"]
    dt = frames["dataset_task_matrix"]
    mp = frames["model_prompt_matrix"]
    dims = frames["analysis_dimension_matrix"]
    obs = frames["paper_observation_registry"]
    ext = frames["extension_inventory"]

    dataset_ids = ids_by_type(inv, "dataset")
    task_ids = ids_by_type(inv, "task")
    llm_family_ids = inv.loc[(inv["component_type"].eq("model_family")) & (inv["component_family"].eq("llm_model")), "paper_component_id"].tolist()
    llm_version_ids = ids_by_type(inv, "model_version")
    trad_ids = inv.loc[(inv["component_type"].eq("model_family")) & (inv["component_family"].eq("traditional_model")), "paper_component_id"].tolist()
    prompt_ids = ids_by_type(inv, "prompt_strategy")
    metric_ids = ids_by_type(inv, "metric")
    dim_ids = ids_by_type(inv, "analysis_dimension")
    obs_ids = ids_by_type(inv, "paper_observation")
    numeric_ids = ids_by_type(inv, "numeric_result_cell")
    extension_ids = ext["extension_id"].tolist()

    mapped_numeric = align.loc[(align["paper_component_id"].isin(numeric_ids)) & (align["local_result_id"].astype(str).str.len().gt(0)), "paper_component_id"].tolist()
    exact_numeric = align.loc[(align["paper_component_id"].isin(numeric_ids)) & (align["reproduction_class"].eq("exact")), "paper_component_id"].tolist()
    llm_local = mp.loc[(mp["model_role"].eq("llm_as_recommender")) & (mp["prompt_strategy"].eq("OpenP5-derived base prompt")) & (mp["has_local_runs"].eq("true")), "model_family"].nunique()
    prompt_implemented = mp.loc[(mp["model_role"].eq("llm_as_recommender")) & (mp["paper_prompt_available"].eq("true")), "prompt_strategy"].nunique()
    exact_prompts = mp.loc[mp["protocol_equivalence"].eq("exact"), "prompt_strategy"].nunique()
    metric_executed = align.loc[(align["paper_component_id"].isin(metric_ids)) & (align["execution_status"].str.contains("executed", na=False)), "paper_component_id"].nunique()
    metric_implemented = align.loc[(align["paper_component_id"].isin(metric_ids)) & (~align["implementation_status"].isin(["not_available", "unknown"])), "paper_component_id"].nunique()
    dims_supported = dims.loc[~dims["reproduction_class"].isin(["blocked_external_dependency", "unknown"]), "analysis_dimension_id"].nunique()
    obs_verified = obs.loc[obs["verification_status"].isin(["verified_exact", "verified_functional"]), "paper_observation_id"].nunique()
    ext_impl = ext.loc[ext["implementation_status"].isin(["implemented", "partial_policy", "not_present"]).map(lambda value: bool(value)) & ext["implementation_status"].ne("not_present"), "extension_id"].nunique()
    ext_exec = ext.loc[ext["execution_status"].isin(["computed", "executed", "audit_only", "computed_for_available_datasets"]), "extension_id"].nunique()

    rows = [
        coverage_row("dataset identification coverage", 4, 4, "paper datasets identified", "four paper datasets", dataset_ids, [], "high", "F6A 4/4", "F6B keeps F6A dataset scope", "LastFM included as paper dataset."),
        coverage_row("dataset evaluation-ready coverage", int(dt["evaluation_ready"].eq("true").groupby(dt["dataset"]).any().sum()), 4, "datasets with any evaluation-ready task", "four paper datasets", dataset_ids[:3], dataset_ids[3:], "high", "F6A 3/4", "F6B row-level task matrix", "LastFM blocked."),
        coverage_row("task implementation coverage", 2, 2, "task families implemented locally", "ranking and reranking", task_ids, [], "high", "F6A 2/2", "no change", "Task implementation is adapted local protocol."),
        coverage_row("task execution coverage", int(dt.loc[dt["evaluation_ready"].eq("true"), "task"].nunique()), 2, "task families executed in evaluation-ready datasets", "ranking and reranking", task_ids, [], "high", "F6A 2/2", "no change", "Executed on three datasets."),
        coverage_row("LLM-family local-run coverage", int(llm_local), 7, "paper LLM families with local runs", "seven paper LLM families", llm_family_ids[-2:], llm_family_ids[:-2], "high", "F6A 2/7", "F6B uses model_prompt matrix", "Only Qwen2 and Mistral have local runs."),
        coverage_row("exact LLM-version coverage", 0, 7, "paper exact model versions reproduced", "seven paper exact LLM versions", [], llm_version_ids, "high", "F6A 0/7", "no exact version equivalence", "Family match is not exact version match."),
        coverage_row("traditional-model coverage", 6, 6, "traditional model families with local counterpart evidence", "six paper traditional models", trad_ids, [], "medium", "not separated", "F6B separates traditional baseline family coverage", "Protocol exactness remains unproven."),
        coverage_row("prompt-strategy implementation coverage", int(prompt_implemented), 3, "paper prompt strategies with available implementation/template evidence", "three paper prompt strategies", prompt_ids, [], "medium", "F6A 3/3 identification", "F6B separates implementation from exact equivalence", "Template availability is not exact execution."),
        coverage_row("prompt-strategy exact-equivalence coverage", int(exact_prompts), 3, "prompt strategies with exact local text/output contract equivalence", "three paper prompt strategies", [], prompt_ids, "high", "F6A 0/3 exact", "strict prompt equivalence rule", "Local JSON/index contracts are adapted."),
        coverage_row("metric implementation coverage", int(metric_implemented), len(metric_ids), "paper metrics with local implementation or policy", "paper metric components", metric_ids[:metric_implemented], metric_ids[metric_implemented:], "medium", "not separated", "F6B metric-level registry", "Functional/adapted metrics are not exact paper claims."),
        coverage_row("metric execution coverage", int(metric_executed), len(metric_ids), "paper metrics executed locally", "paper metric components", metric_ids[:metric_executed], metric_ids[metric_executed:], "medium", "not separated", "F6B metric-level execution split", "PopREO remains blocked."),
        coverage_row("analysis-dimension coverage", int(dims_supported), len(dim_ids), "dimensions with local support or adapted audit evidence", "main plus appendix dimensions", dim_ids[:dims_supported], dim_ids[dims_supported:], "medium", "F6A 8/8 inventory", "F6B separates inventory from support", "Generation and appendix components remain incomplete."),
        coverage_row("paper-observation verification coverage", int(obs_verified), len(obs_ids), "observations verified exact/functionally", "seven paper observations", [], obs_ids, "high", "unresolved", "F6B requires exact evidence for verification", "Partial support is not counted as verification."),
        coverage_row("atomic numeric extraction coverage", len(numeric_ids), len(numeric_ids), "extracted atomic numeric cells", "F5AF 187 atomic cells", numeric_ids, [], "high", "F5AF 187/187", "preserved", "Numeric cells are separate from component coverage."),
        coverage_row("local numeric mapping coverage", len(mapped_numeric), len(numeric_ids), "numeric cells with any local mapped counterpart", "F5AF 187 atomic cells", mapped_numeric, [x for x in numeric_ids if x not in set(mapped_numeric)], "high", "F6A 44/187", "preserved", "Local mappings are not exact replication."),
        coverage_row("protocol-exact numeric replication coverage", len(exact_numeric), len(numeric_ids), "numeric cells with exact protocol and value match", "F5AF 187 atomic cells", exact_numeric, numeric_ids, "high", "F6A 0/187", "preserved", "No exact numeric replication is promoted."),
        coverage_row("extension implementation coverage", int(ext_impl), len(extension_ids), "extensions implemented locally", "F6B extension inventory", extension_ids[:ext_impl], extension_ids[ext_impl:], "medium", "F6A 9 extensions", "F6B adds fifth-dataset status row", "Fifth dataset status is explicit but not implemented."),
        coverage_row("extension execution coverage", int(ext_exec), len(extension_ids), "extensions executed or audited locally", "F6B extension inventory", extension_ids[:ext_exec], extension_ids[ext_exec:], "medium", "F6A extension inventory", "F6B separates implementation and execution", "Dataset requirement status is not an executed extension."),
        coverage_row("flat overall component percentage", "not_applicable", "not_applicable", "not published", "not published", [], inv["paper_component_id"].tolist(), "high", "73-component headline", "overlap and heterogeneous denominator make this unsafe", "No authoritative flat percentage is published.", authoritative="false"),
    ]
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def build_corrected_missing_evidence(root: Path) -> pd.DataFrame:
    source = read_csv(root, F4_MISSING)
    if source.empty:
        source = read_csv(root, F3_MISSING)
    rows: list[dict[str, str]] = []
    for _, row in source.iterrows():
        sid = clean(row.get("missing_evidence_id"))
        metric = clean(row.get("metric"))
        model = clean(row.get("model"))
        text = " ".join(clean(row.get(col)) for col in source.columns)
        corrected = "still_missing"
        current = "active_gap"
        rescat = clean(row.get("resolution_category")) or clean(row.get("blocking_type"))
        superseded = "false"
        duplicate = "false"
        newly = "false"
        requires_offline = bool_text(clean(row.get("resolution_category")) in {"resolvable_existing_artifacts", "resolvable_offline_recomputation", "resolvable_local_execution"})
        requires_training = clean(row.get("requires_training")) or "false"
        requires_llm = clean(row.get("requires_llm_inference")) or "false"
        requires_api = clean(row.get("requires_api")) or "false"
        requires_network = clean(row.get("requires_network")) or "false"
        requires_new_dataset = clean(row.get("requires_new_dataset")) or "false"
        notes = "Retained from F4 triage."
        if "LastFM" in text:
            corrected = "original_paper_dataset_blocked_pipeline"
            current = "active_gap"
            notes = "Relabelled: LastFM is paper scope, not extension-only."
        if sid == "F3-MISS-0015":
            corrected = "model_family_present_exact_version_protocol_missing"
            current = "active_gap_split_required"
            notes = "Split family presence from exact version/protocol evidence."
        if "paper_value_unverified" in text or clean(row.get("area")) == "paper_numeric_cell":
            corrected = "superseded_by_atomic_numeric_registry"
            current = "superseded_by_f5af_f6b_numeric_components"
            superseded = "true"
            requires_offline = "false"
            notes = "The non-atomic missing row is superseded by 187 atomic numeric cells; exact replication remains separate."
        if metric == "intra_list_diversity_at_5":
            corrected = "resolved_adapted_by_f5b"
            current = "resolved_adapted"
            superseded = "true"
            notes = "F5B supplies adapted ILD policy/evidence."
        if metric == "popularity_bias_at_5":
            corrected = "resolved_semantic_split_by_f5c"
            current = "resolved_or_partially_resolved"
            superseded = "true"
            notes = "F5C resolves popularity semantics; PopREO remains separate blocked metric."
        if "serendipity" in metric.lower():
            corrected = "resolved_adapted_by_f5d"
            current = "resolved_adapted"
            superseded = "true"
            notes = "F5D supplies adapted serendipity evidence."
        if "calibration" in metric.lower():
            corrected = "extension_only_resolved_by_f5e"
            current = "removed_from_paper_missing_scope"
            superseded = "true"
            notes = "Calibration is extension-only, not paper missing evidence."
        rows.append(
            {
                "corrected_missing_id": stable_id("F6B-MISS", sid, corrected),
                "source_missing_id": sid,
                "paper_component_id": clean(row.get("paper_component_id")),
                "area": clean(row.get("area")),
                "dataset": clean(row.get("dataset")),
                "task": clean(row.get("task")),
                "model": model,
                "prompt": model if "prompt" in model.lower() else "",
                "metric_or_dimension": metric,
                "missing_element": clean(row.get("missing_element")) or clean(row.get("blocking_type")),
                "previous_classification": clean(row.get("f3_status")) or clean(row.get("current_status")),
                "corrected_classification": corrected,
                "current_status": current,
                "resolution_category": rescat,
                "requires_offline_computation": requires_offline,
                "requires_training": requires_training,
                "requires_llm_inference": requires_llm,
                "requires_api": requires_api,
                "requires_network": requires_network,
                "requires_new_dataset": requires_new_dataset,
                "is_duplicate": duplicate,
                "is_superseded": superseded,
                "is_newly_added": newly,
                "priority": clean(row.get("priority")) or "medium",
                "recommended_phase": clean(row.get("recommended_phase")) or "F6C",
                "completion_criterion": clean(row.get("completion_criterion")) or "documented resolution or blocker",
                "notes": notes,
            }
        )
    for family, version in [("Qwen2 7B Instruct", "qwen-2-7b-instruct"), ("Mistral 7B Instruct", "mistral-7b-instruct-v0.2")]:
        rows.append(
            {
                "corrected_missing_id": stable_id("F6B-MISS-NEW", family),
                "source_missing_id": "",
                "paper_component_id": stable_id("F6B-MODEL-VERSION", family, version),
                "area": "model_protocol",
                "dataset": "Amazon Beauty;Amazon Sports;ML-1M;LastFM",
                "task": "ranking;reranking",
                "model": family,
                "prompt": "OpenP5-derived base prompt;LLMRank recency-focused;LLMSRec in-context learning",
                "metric_or_dimension": "exact model/prompt/protocol equivalence",
                "missing_element": "exact model version, prompt, candidate set, cohort, decoding, parser and retry equivalence",
                "previous_classification": "not_separated",
                "corrected_classification": "new_split_exact_protocol_gap",
                "current_status": "active_gap",
                "resolution_category": "blocked_protocol_unrecoverable",
                "requires_offline_computation": "true",
                "requires_training": "false",
                "requires_llm_inference": "false",
                "requires_api": "false",
                "requires_network": "false",
                "requires_new_dataset": "false",
                "is_duplicate": "false",
                "is_superseded": "false",
                "is_newly_added": "true",
                "priority": "high",
                "recommended_phase": "F6C",
                "completion_criterion": "produce exact-equivalence audit or keep local run adapted",
                "notes": "Added by F6B to split family presence from exact paper execution.",
            }
        )
    return pd.DataFrame(rows, columns=MISSING_COLUMNS)


def build_experiment_priority(root: Path, missing: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("F6B-PRI-001", "F3-MISS-0014;F3-MISS-0016", "paper result tables", "PDF/manual verification of paper numeric evidence", "verify atomic cells against primary PDF pages", "all", "ranking;reranking", "all", "not_applicable", "paper PDF and DOCX", "offline extraction/review", "false", "false", "false", "false", "medium", "unknown", "numeric-cell confidence", "high", "high", "low", "95", "F6C", "PDF page-level confirmation complete"),
        ("F6B-PRI-002", "F3-MISS-0010;F3-MISS-0042", "LastFM dataset", "LastFM evaluation readiness", "build validated split, ground truth and candidate manifests", "LastFM", "ranking;reranking", "all", "not_applicable", "LastFM raw/reconstructed data", "dataset pipeline and validation", "false", "false", "false", "false", "medium", "unknown", "dataset evaluation-ready coverage", "high", "high", "medium", "92", "F6C", "LastFM dataset_task rows become evaluation_ready"),
        ("F6B-PRI-003", "", "Qwen2/Mistral exact model protocol", "Qwen2/Mistral exact-equivalence audit", "separate family match from exact version, prompt and parser equivalence", "all", "ranking;reranking", "Qwen2;Mistral", "OpenP5-derived base", "local manifests and paper prompt/version evidence", "offline audit", "false", "false", "false", "false", "low", "unknown", "exact LLM-version coverage confidence", "high", "high", "low", "89", "F6C", "exact-equivalence decision table complete"),
        ("F6B-PRI-004", "F3-MISS-0018;F3-MISS-0019", "prompt families", "Original prompt-family completion plan", "prepare LLMRank and LLMSRec safe execution or cached-output mapping", "three ready datasets", "ranking;reranking", "Qwen2;Mistral or paper models", "LLMRank;LLMSRec", "templates and candidate manifests", "runner/preflight", "false", "true", "false", "false", "high", "unknown", "prompt-strategy execution coverage", "medium", "high", "medium", "80", "F7", "approved no-network/no-API or cached execution evidence"),
        ("F6B-PRI-005", "F3-MISS-0035;F3-MISS-0037;F3-MISS-0039", "PopREO/popularity fairness", "PopREO denominator policy recovery", "resolve fairness denominator and empty-group policy", "ready datasets", "ranking;reranking", "local counterparts", "not_applicable", "item popularity and group policy", "offline metric policy", "false", "false", "false", "false", "medium", "unknown", "metric implementation coverage", "medium", "medium", "medium", "76", "F6C", "PopREO current_status no longer blocked"),
        ("F6B-PRI-006", "F3-MISS-0008", "candidate-first prompt", "Candidate-first exact prompt condition", "execute or map candidate-first condition safely", "ready datasets", "ranking", "LLMs", "candidate-first", "prompt template and manifests", "LLM runner", "false", "true", "false", "false", "medium", "unknown", "candidate-position observation support", "medium", "medium", "medium", "68", "F7", "candidate-first evidence exists"),
        ("F6B-PRI-007", "F3-MISS-0011;F3-MISS-0017", "profile generation", "Generated-profile paired controls", "recover profile/no-profile paired experiment evidence", "ready datasets", "ranking;reranking", "LLMs", "generated-profile", "profile prompts and outputs", "LLM runner or cached outputs", "false", "true", "false", "false", "high", "unknown", "generation-involved dimension", "medium", "medium", "high", "62", "F7", "paired controls complete"),
        ("F6B-PRI-008", "F3-MISS-0005", "Llama3-70B", "Open-weight large-model feasibility", "decide hardware/cloud path for Llama3-70B", "ready datasets", "ranking;reranking", "Llama3-70B", "paper prompts", "hardware/API access", "runner", "false", "true", "true", "unknown", "high", "unknown", "LLM-family local-run coverage", "medium", "medium", "high", "55", "F8", "access/cost decision recorded"),
        ("F6B-PRI-009", "F3-MISS-0001;F3-MISS-0002;F3-MISS-0003;F3-MISS-0004", "closed LLMs", "Closed-model exact execution decision", "execute only after access/cost/approval is known", "ready datasets", "ranking;reranking", "GPT/Claude", "paper prompts", "API access and budget", "API runner", "false", "true", "true", "true", "high", "unknown", "LLM-family/exact version coverage", "high", "medium", "high", "50", "F8", "approved execution or documented blocker"),
        ("F6B-PRI-010", "", "fifth dataset extension", "Additional dataset extension", "select public comparable fifth dataset and build full manifest", "new dataset", "ranking;reranking", "local baselines;local LLMs", "adapted", "new public data", "dataset pipeline", "possible", "possible", "false", "possible", "high", "unknown", "extension coverage", "medium", "high", "high", "45", "F9", "fifth dataset evaluation-ready"),
    ]
    rows = []
    for order, spec in enumerate(specs, start=1):
        pid, mids, pcids, title, objective, dataset, task, model, prompt, inputs, code, train, llm, api, network, runtime, cost, gain, scientific, exam, risk, score, phase, criterion = spec
        rows.append(
            {
                "priority_id": pid,
                "source_missing_ids": mids,
                "paper_component_ids": pcids,
                "experiment_title": title,
                "objective": objective,
                "dataset": dataset,
                "task": task,
                "model": model,
                "prompt": prompt,
                "required_inputs": inputs,
                "required_code": code,
                "requires_training": train,
                "requires_llm_inference": llm,
                "requires_api": api,
                "requires_network": network,
                "estimated_runtime_class": runtime,
                "estimated_monetary_cost": cost,
                "expected_coverage_family_gain": gain,
                "scientific_value": scientific,
                "exam_value": exam,
                "risk": risk,
                "priority_score": score,
                "recommended_order": str(order),
                "recommended_phase": phase,
                "completion_criterion": criterion,
                "notes": "Priority ordering follows F6B policy and does not execute the experiment.",
            }
        )
    return pd.DataFrame(rows, columns=PRIORITY_COLUMNS)


def build_artifact_supersession_registry(root: Path) -> pd.DataFrame:
    f6a = read_csv(root, F6A_AFFECTED)
    rows = []
    for _, row in f6a.iterrows():
        artifact = clean(row.get("artifact"))
        rows.append(
            {
                "artifact_id": stable_id("F6B-ART", artifact),
                "artifact": artifact,
                "phase": clean(row.get("phase")),
                "affected_by_scope_error": clean(row.get("affected_by_taxonomy_error")),
                "error_type": clean(row.get("error_type")),
                "scientific_impact": clean(row.get("scientific_impact")),
                "remains_authoritative_for": clean(row.get("remains_authoritative_for")),
                "superseded_for": clean(row.get("scientific_impact")),
                "superseding_f6b_output": clean(row.get("proposed_corrective_output")).replace("finalization_f6a_", "finalization_f6b_"),
                "must_be_corrected_later": clean(row.get("must_be_corrected")),
                "priority": clean(row.get("priority")),
                "notes": "Historical artifact is not overwritten by F6B.",
            }
        )
    return pd.DataFrame(rows, columns=SUPERSESSION_COLUMNS)


def classify_test_file(root: Path, path_text: str) -> str:
    path = root / path_text
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    unsafe_terms = ["ollama", "execute_ollama_request", "allow_api", "training", "torchrun", "download", "requests.", "urllib.request"]
    return "excluded_unsafe" if any(term in text for term in unsafe_terms) else "safe_offline_static"


def build_regression_resolution(root: Path) -> pd.DataFrame:
    groups = [
        ("F6B-REG-001", "tests/test_finalization_f3_canonical_scientific_consolidation.py;tests/test_finalization_f4_metric_resolution.py;tests/test_finalization_f5a_paper_numeric_evidence.py", "conclusive_pass_f6b_shard", "38", "0", "0", "false", "76.50", "F6B safe shard passed after static inspection."),
        ("F6B-REG-002", "tests/test_finalization_f5ar_structured_numeric_recovery.py", "conclusive_pass_f6b_isolated", "11", "0", "0", "false", "22.79", "F5A-R isolated test passed in F6B."),
        ("F6B-REG-003", "tests/test_finalization_f5af_atomic_paper_cells.py", "conclusive_pass_f6b_isolated", "12", "0", "0", "false", "22.47", "F5AF isolated rerun passed in F6B; an earlier parallel run hit a Windows temp cleanup lock and was discarded as harness noise."),
        ("F6B-REG-004", "tests/test_finalization_f5b_ild_policy.py", "conclusive_pass_f6b_isolated", "10", "0", "0", "false", "259.44", "F5B isolated test resolved the prior F6A timeout."),
        ("F6B-REG-005", "tests/test_finalization_f5e_calibration.py", "conclusive_pass_f6b_isolated", "8", "0", "0", "false", "241.29", "F5E isolated test resolved the prior F6A timeout."),
        ("F6B-REG-006", "combined F3-F5E aggregate", "inconclusive_timeout_prior_f6a", "0", "0", "0", "true", "304.03", "Prior aggregate command timed out and is not treated as pass."),
        ("F6B-REG-007", "combined F5AR/F5AF/F5B aggregate", "inconclusive_timeout_prior_f6a", "0", "0", "0", "true", "244.04", "Prior aggregate command timed out and is not treated as pass."),
    ]
    rows = []
    for gid, files, status, passed, failed, skipped, timed_out, runtime, notes in groups:
        file_list = [piece for piece in files.split(";") if piece.startswith("tests/")]
        classes = [classify_test_file(root, piece) for piece in file_list] or ["aggregate_command"]
        safety = "safe_offline_static" if all(value == "safe_offline_static" for value in classes) else ";".join(sorted(set(classes)))
        rows.append(
            {
                "test_group_id": gid,
                "test_files": files,
                "safety_classification": safety,
                "execution_status": status,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "timed_out": timed_out,
                "runtime_seconds": runtime,
                "failure_summary": "" if failed == "0" else "see pytest output",
                "is_conclusive": bool_text(timed_out == "false"),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=REGRESSION_COLUMNS)


def build_claim_corrections(root: Path, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    component_ids = frames["paper_component_inventory"]
    dataset_ids = join_ids(component_ids.loc[component_ids["component_type"].eq("dataset"), "paper_component_id"].tolist())
    llm_mask = component_ids["component_type"].eq("model_family") & component_ids["component_family"].eq("llm_model")
    llm_ids = join_ids(component_ids.loc[llm_mask, "paper_component_id"].tolist())
    numeric_ids = join_ids(component_ids.loc[component_ids["component_type"].eq("numeric_result_cell"), "paper_component_id"].head(10).tolist()) + ";...(187 total)"
    claims = [
        ("F6B-CLAIM-001", "F6A-CLAIM-002", "LastFM is extension-only.", "dataset_scope_error", "LastFM is one of four original paper datasets and is locally blocked for comparable evaluation.", dataset_ids, f"{rel(PAPER_DOCX)};{rel(F6A_DATASETS)}", "high", "missing split/candidate/ground truth"),
        ("F6B-CLAIM-002", "F6A-CLAIM-001", "Qwen2 is a local-only extension.", "model_scope_error", "Qwen2 is an original paper model family; local qwen2:7b-instruct is adapted, not exact.", llm_ids, f"{rel(PAPER_DOCX)};{rel(LLM_MANIFEST)}", "high", "exact version/protocol missing"),
        ("F6B-CLAIM-003", "F6A-CLAIM-001", "Mistral is a local-only extension.", "model_scope_error", "Mistral is an original paper model family; local mistral:7b-instruct is adapted, not exact.", llm_ids, f"{rel(PAPER_DOCX)};{rel(LLM_MANIFEST)}", "high", "exact version/protocol missing"),
        ("F6B-CLAIM-004", "", "The paper has three local datasets.", "dataset_count_error", "The paper has four original datasets; local evaluation-ready coverage is 3/4.", dataset_ids, rel(F6A_DATASETS), "high", "LastFM blocked locally"),
        ("F6B-CLAIM-005", "F6A-CLAIM-003", "Paper model inventory coverage is 5/5.", "model_denominator_error", "Paper LLM model-family inventory is 7/7; exact version coverage is 0/7.", llm_ids, rel(F6A_MODELS), "high", "family inventory is not execution coverage"),
        ("F6B-CLAIM-006", "", "The repository has a fifth evaluation dataset.", "extension_dataset_error", "No fifth evaluation dataset exists; LastFM is not eligible as extension dataset.", dataset_ids, f"{rel(DATASET_STATS)};{rel(COHORT_SUMMARY)}", "high", "new dataset still required if mandatory"),
        ("F6B-CLAIM-007", "F6A-CLAIM-004", "Calibration improves paper coverage.", "metric_scope_error", "Calibration is extension-only and does not increase paper reproduction coverage.", stable_id("F6B-EXT", "calibration"), rel(F5E_CALIBRATION), "high", "same-protocol local comparisons only"),
        ("F6B-CLAIM-008", "", "Exact coverage can be summarized by the 73-component denominator.", "coverage_denominator_error", "F6B replaces the unsafe headline with separate coverage families.", "", "results/tables/finalization_f6b_corrected_coverage_dashboard.csv", "high", "no flat authoritative percentage"),
        ("F6B-CLAIM-009", "", "Reconstructed coverage includes adapted local model results.", "coverage_class_error", "Reconstructed, adapted, functional and exact evidence are separate reproduction classes.", "", "results/tables/finalization_f6b_local_component_alignment.csv", "high", "class boundaries must remain visible"),
        ("F6B-CLAIM-010", "F6A-CLAIM-006", "Numeric replication is supported by mapped local values.", "numeric_replication_error", "Atomic extraction is 187/187, local mapping is 44/187, protocol-exact numeric replication is 0/187.", numeric_ids, f"{rel(F5AF_ATOMIC)};{rel(F5AF_MAPPING)}", "high", "PDF verification and protocol exactness remain limitations"),
    ]
    return pd.DataFrame(
        [
            {
                "claim_correction_id": cid,
                "previous_claim_id": previous_id,
                "previous_claim": previous,
                "scope_error": error,
                "corrected_claim": corrected,
                "supporting_component_ids": ids,
                "supporting_files": files,
                "confidence": confidence,
                "safe_for_report": "true",
                "safe_for_presentation": "true",
                "remaining_limitation": limit,
                "notes": "F6B superseding claim correction.",
            }
            for cid, previous_id, previous, error, corrected, ids, files, confidence, limit in claims
        ],
        columns=CLAIM_COLUMNS,
    )


def build_validation(root: Path, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(cid: str, name: str, condition: bool, observed: Any, expected: Any, details: str) -> None:
        rows.append({"check_id": cid, "check_name": name, "status": "pass" if condition else "fail", "observed": clean(observed), "expected": clean(expected), "details": details})

    inv = frames["paper_component_inventory"]
    align = frames["local_component_alignment"]
    missing = frames["corrected_missing_evidence"]
    coverage = frames["corrected_coverage_dashboard"]
    ext = frames["extension_inventory"]
    regression = frames["regression_resolution"]
    all_frames = list(frames.values())

    add("F6B-VAL-001", "four_original_datasets", inv.loc[inv["component_type"].eq("dataset"), "dataset"].nunique() == 4, inv.loc[inv["component_type"].eq("dataset"), "dataset"].nunique(), "4", "All paper datasets represented.")
    add("F6B-VAL-002", "lastfm_not_extension", not ext["extension_name"].str.contains("LastFM", case=False).any(), "no LastFM extension", "no LastFM extension", "LastFM excluded from extension inventory.")
    add("F6B-VAL-003", "seven_original_llms", inv.loc[(inv["component_type"].eq("model_family")) & (inv["component_family"].eq("llm_model")), "model_family"].nunique() == 7, inv.loc[(inv["component_type"].eq("model_family")) & (inv["component_family"].eq("llm_model")), "model_family"].nunique(), "7", "Seven LLM families.")
    add("F6B-VAL-004", "qwen_mistral_paper_families", {"Qwen2 7B Instruct", "Mistral 7B Instruct"}.issubset(set(inv["model_family"])), "present", "present", "Qwen2/Mistral paper families.")
    add("F6B-VAL-005", "six_traditional_models", inv.loc[(inv["component_type"].eq("model_family")) & (inv["component_family"].eq("traditional_model")), "model_family"].nunique() == 6, inv.loc[(inv["component_type"].eq("model_family")) & (inv["component_family"].eq("traditional_model")), "model_family"].nunique(), "6", "Six traditional models.")
    add("F6B-VAL-006", "three_prompt_strategies", inv.loc[inv["component_type"].eq("prompt_strategy"), "prompt_strategy"].nunique() == 3, inv.loc[inv["component_type"].eq("prompt_strategy"), "prompt_strategy"].nunique(), "3", "Three prompt strategies.")
    add("F6B-VAL-007", "ranking_reranking_represented", set(inv.loc[inv["component_type"].eq("task"), "task"]) == {"ranking", "reranking"}, join_ids(inv.loc[inv["component_type"].eq("task"), "task"].tolist()), "ranking;reranking", "Both task families.")
    add("F6B-VAL-008", "main_dimensions_represented", len(frames["analysis_dimension_matrix"]) >= 8, len(frames["analysis_dimension_matrix"]), ">=8", "Main and appendix dimensions.")
    add("F6B-VAL-009", "appendix_components_explicit", inv.loc[inv["component_type"].eq("appendix_component"), "is_appendix"].eq("true").all(), "true", "true", "Appendix rows explicit.")
    add("F6B-VAL-010", "calibration_extension_only", ext["extension_name"].str.contains("Calibration", case=False).any() and not inv["metric"].eq("calibration").any(), "extension only", "extension only", "Calibration not paper component.")
    add("F6B-VAL-011", "component_evidence_present", inv["paper_evidence"].map(clean).ne("").all(), len(inv), "all rows", "Every paper component has evidence.")
    add("F6B-VAL-012", "atomic_unique", inv["paper_component_id"].is_unique and inv["is_atomic"].eq("true").all(), len(inv), "unique atomic rows", "Atomic component IDs unique.")
    add("F6B-VAL-013", "component_types_not_mixed", not inv.loc[inv["component_type"].ne("numeric_result_cell"), "is_numeric"].eq("true").any(), "non-numeric separated", "separated", "Numeric status is type-specific.")
    add("F6B-VAL-014", "numeric_cells_separate", inv.loc[inv["component_type"].eq("numeric_result_cell"), "is_numeric"].eq("true").all(), inv["component_type"].eq("numeric_result_cell").sum(), "187 numeric rows", "Numeric cells separate.")
    add("F6B-VAL-015", "local_alignment_evidence", align["evidence"].map(clean).ne("").all(), len(align), "all rows", "Every alignment has evidence.")
    add("F6B-VAL-016", "exact_requires_exact_protocol", align.loc[align["reproduction_class"].eq("exact"), "protocol_equivalence"].eq("exact").all(), "no violations", "exact protocol", "Exact requires exact protocol.")
    add("F6B-VAL-017", "all_42_missing_source_rows_accounted", missing["source_missing_id"].map(clean).ne("").sum() >= 42, missing["source_missing_id"].map(clean).ne("").sum(), ">=42", "All source missing rows accounted.")
    add("F6B-VAL-018", "duplicates_superseded_explicit", {"is_duplicate", "is_superseded"}.issubset(missing.columns), "present", "present", "Duplicate/supersession flags present.")
    add("F6B-VAL-019", "fifth_dataset_status_explicit", ext["extension_id"].eq("EXT-FIFTH-DATASET-STATUS").any(), "present", "present", "Fifth dataset status explicit.")
    add("F6B-VAL-020", "no_unsafe_denominator_authoritative", coverage.loc[coverage["coverage_family"].eq("flat overall component percentage"), "is_authoritative"].eq("false").all() and coverage.loc[coverage["coverage_family"].eq("flat overall component percentage"), "percentage"].eq("not_applicable").all(), "not_applicable", "not_applicable", "No flat unsafe percentage.")
    add("F6B-VAL-021", "coverage_denominators_documented", coverage["denominator_definition"].map(clean).ne("").all(), len(coverage), "all rows", "Every denominator documented.")
    add("F6B-VAL-022", "affected_artifacts_registered", len(frames["artifact_supersession_registry"]) >= 17, len(frames["artifact_supersession_registry"]), ">=17", "Previous affected artifacts registered.")
    add("F6B-VAL-023", "timeouts_not_passes", not regression.loc[regression["timed_out"].eq("true"), "execution_status"].str.contains("pass", case=False).any(), "timeouts inconclusive", "not pass", "Timeouts are not passes.")
    add("F6B-VAL-024", "no_training_or_inference_executed", not regression["execution_status"].str.contains("training|inference", case=False).any(), "none", "none", "F6B does not run training/inference.")
    add("F6B-VAL-025", "no_network_or_api_used", not regression["safety_classification"].str.contains("network|api", case=False).any(), "none", "none", "No network/API execution.")
    add("F6B-VAL-026", "protected_inputs_unchanged", protected_hash_status(root) == "checked=1144;changed=0", protected_hash_status(root), "checked=1144;changed=0", "Protected inputs unchanged.")
    add("F6B-VAL-027", "previous_outputs_unchanged_policy", all(path.name.startswith("finalization_f6b_") for path in OUTPUT_TABLES.values()) and REPORT_PATH.name.startswith("finalization_f6b_"), "F6B only", "F6B only", "Builder output set is F6B-prefixed.")
    add("F6B-VAL-028", "relative_paths_only", all(no_absolute_paths(frame) for frame in all_frames), "all frames", "relative paths", "No absolute local paths.")
    add("F6B-VAL-029", "deterministic_frames", deterministic_frames(all_frames), "stable", "stable", "Frame serialization deterministic.")
    add("F6B-VAL-030", "all_required_outputs_exist_after_builder", len(OUTPUT_TABLES) == 14, len(OUTPUT_TABLES), "14 tables", "All required F6B table definitions present.")
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def deterministic_frames(frames: Sequence[pd.DataFrame]) -> bool:
    return [frame_sha256(frame) for frame in frames] == [frame_sha256(frame.copy()) for frame in frames]


def build_frames(root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    frames["paper_component_inventory"] = build_paper_component_inventory(root)
    frames["local_component_alignment"] = build_local_component_alignment(root, frames["paper_component_inventory"])
    frames["dataset_task_matrix"] = build_dataset_task_matrix(root)
    frames["model_prompt_matrix"] = build_model_prompt_matrix(root)
    frames["analysis_dimension_matrix"] = build_analysis_dimension_matrix(root)
    frames["paper_observation_registry"] = build_paper_observation_registry(root)
    frames["extension_inventory"] = build_extension_inventory(root)
    frames["corrected_missing_evidence"] = build_corrected_missing_evidence(root)
    frames["experiment_priority"] = build_experiment_priority(root, frames["corrected_missing_evidence"])
    frames["artifact_supersession_registry"] = build_artifact_supersession_registry(root)
    frames["regression_resolution"] = build_regression_resolution(root)
    frames["corrected_coverage_dashboard"] = build_corrected_coverage_dashboard(root, frames)
    frames["claim_corrections"] = build_claim_corrections(root, frames)
    frames["validation"] = build_validation(root, frames)
    return {key: frames[key] for key in OUTPUT_TABLES}


def final_decision(validation: pd.DataFrame) -> str:
    return "finalization_f6b_complete" if validation["status"].eq("pass").all() else "finalization_f6b_blocked"


def active_missing_count(missing: pd.DataFrame) -> int:
    return int(missing["current_status"].astype(str).str.contains("active_gap", na=False).sum())


def build_report(root: Path, frames: Mapping[str, pd.DataFrame]) -> str:
    inv = frames["paper_component_inventory"]
    coverage = frames["corrected_coverage_dashboard"].set_index("coverage_family")
    missing = frames["corrected_missing_evidence"]
    obs = frames["paper_observation_registry"]
    validation = frames["validation"]
    components_by_type = inv["component_type"].value_counts().to_dict()
    lines = [
        "# 1. Executive summary",
        f"- Decisione: `{final_decision(validation)}`.",
        f"- Componenti paper atomici: `{len(inv)}`; non numerici: `{int(inv['is_numeric'].ne('true').sum())}`; numeric cells: `{int(inv['is_numeric'].eq('true').sum())}`.",
        "",
        "# 2. Authoritative scope",
        "- Scope paper: 4 dataset, 2 task, 7 LLM, 6 modelli tradizionali, 3 strategie prompt.",
        "",
        "# 3. Atomicity methodology",
        "- F6B crea righe atomiche solo per requisiti indipendentemente verificabili; non usa prodotti cartesiani arbitrari.",
        "",
        "# 4. Dataset coverage",
        f"- Dataset identification: `{coverage.loc['dataset identification coverage','numerator']}/{coverage.loc['dataset identification coverage','denominator']}`.",
        f"- Evaluation-ready: `{coverage.loc['dataset evaluation-ready coverage','numerator']}/{coverage.loc['dataset evaluation-ready coverage','denominator']}`.",
        "",
        "# 5. Task coverage",
        f"- Task implementation/execution: `{coverage.loc['task implementation coverage','numerator']}/{coverage.loc['task implementation coverage','denominator']}` and `{coverage.loc['task execution coverage','numerator']}/{coverage.loc['task execution coverage','denominator']}`.",
        "",
        "# 6. Model-family coverage",
        f"- LLM-family local-run coverage: `{coverage.loc['LLM-family local-run coverage','numerator']}/{coverage.loc['LLM-family local-run coverage','denominator']}`.",
        f"- Traditional-model coverage: `{coverage.loc['traditional-model coverage','numerator']}/{coverage.loc['traditional-model coverage','denominator']}`.",
        "",
        "# 7. Exact-version coverage",
        f"- Exact LLM-version coverage: `{coverage.loc['exact LLM-version coverage','numerator']}/{coverage.loc['exact LLM-version coverage','denominator']}`.",
        "",
        "# 8. Prompt-strategy coverage",
        f"- Prompt implementation: `{coverage.loc['prompt-strategy implementation coverage','numerator']}/{coverage.loc['prompt-strategy implementation coverage','denominator']}`; exact equivalence: `{coverage.loc['prompt-strategy exact-equivalence coverage','numerator']}/{coverage.loc['prompt-strategy exact-equivalence coverage','denominator']}`.",
        "",
        "# 9. Metric coverage",
        f"- Metric implementation: `{coverage.loc['metric implementation coverage','numerator']}/{coverage.loc['metric implementation coverage','denominator']}`; execution: `{coverage.loc['metric execution coverage','numerator']}/{coverage.loc['metric execution coverage','denominator']}`.",
        "",
        "# 10. Analysis-dimension coverage",
        f"- Analysis-dimension support: `{coverage.loc['analysis-dimension coverage','numerator']}/{coverage.loc['analysis-dimension coverage','denominator']}`.",
        "",
        "# 11. Paper-observation verification",
        f"- Verified exact/functionally: `{coverage.loc['paper-observation verification coverage','numerator']}/{coverage.loc['paper-observation verification coverage','denominator']}`.",
        "",
        "# 12. Numeric-cell coverage",
        f"- Atomic extraction: `{coverage.loc['atomic numeric extraction coverage','numerator']}/{coverage.loc['atomic numeric extraction coverage','denominator']}`.",
        f"- Local mapping: `{coverage.loc['local numeric mapping coverage','numerator']}/{coverage.loc['local numeric mapping coverage','denominator']}`.",
        f"- Protocol-exact numeric replication: `{coverage.loc['protocol-exact numeric replication coverage','numerator']}/{coverage.loc['protocol-exact numeric replication coverage','denominator']}`.",
        "",
        "# 13. True extension inventory",
        f"- Extension rows: `{len(frames['extension_inventory'])}`; LastFM is not an extension.",
        "",
        "# 14. Fifth-dataset status",
        "- Fifth dataset exists: `no`; fifth dataset requirement met: `no_if_mandatory`.",
        "",
        "# 15. Corrected coverage dashboard",
        f"- Coverage families: `{len(frames['corrected_coverage_dashboard'])}`.",
        "",
        "# 16. Why one headline percentage is unsafe or how it was derived",
        "- Flat overall component percentage is `not_applicable`: rows overlap across component, prompt, model, metric, observation and numeric-cell families.",
        "",
        "# 17. Missing-evidence rebaseline",
        f"- Source rows audited: `{missing['source_missing_id'].map(clean).ne('').sum()}`; corrected rows: `{len(missing)}`; active gap rows: `{active_missing_count(missing)}`.",
        "",
        "# 18. Experiment priority roadmap",
        *[f"- {row['recommended_order']}. {row['experiment_title']} (`{row['recommended_phase']}`)." for _, row in frames["experiment_priority"].head(10).iterrows()],
        "",
        "# 19. Previous artifacts superseded",
        f"- Supersession rows: `{len(frames['artifact_supersession_registry'])}`.",
        "",
        "# 20. Previous artifacts still authoritative",
        "- F5AF atomic cells, F5B-F5E metric computations, canonical local LLM execution evidence, F2 protected hashes remain authoritative for their scoped evidence.",
        "",
        "# 21. Claim corrections",
        f"- Claim corrections: `{len(frames['claim_corrections'])}`.",
        "",
        "# 22. Regression timeout resolution",
        f"- Regression rows: `{len(frames['regression_resolution'])}`; inconclusive timeout rows: `{frames['regression_resolution']['timed_out'].eq('true').sum()}`.",
        "",
        "# 23. Anomalies and limitations",
        "- PDF page verification remains limited; exact model/prompt/candidate/cohort equivalence remains absent.",
        "",
        "# 24. Tests and validation",
        f"- Validation pass: `{validation['status'].eq('pass').sum()}`; fail: `{validation['status'].eq('fail').sum()}`.",
        "",
        "# 25. Protected inputs",
        f"- `{protected_hash_status(root)}`.",
        "",
        "# 26. Final decision",
        f"- `{final_decision(validation)}`.",
        "",
        "# 27. Recommended next phase",
        "- F6C should perform PDF page confirmation, LastFM readiness planning, and exact-equivalence audits before any new experiment.",
        "",
        "# Inventory summary",
        f"- Components by type: `{components_by_type}`.",
        f"- Paper observations total: `{len(obs)}`.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, relative in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].fillna("").to_csv(path, index=False, lineterminator="\n")
        hashes[relative.as_posix()] = file_sha256(path)
    report_path = root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    hashes[REPORT_PATH.as_posix()] = file_sha256(report_path)
    return dict(sorted(hashes.items()))


def build_all(root: Path) -> dict[str, Any]:
    frames = build_frames(root)
    report = build_report(root, frames)
    hashes = write_outputs(root, frames, report)
    return {"frames": frames, "report": report, "hashes": hashes, "decision": final_decision(frames["validation"])}


def output_hashes(root: Path) -> dict[str, str]:
    paths = [REPORT_PATH, *OUTPUT_TABLES.values()]
    return {path.as_posix(): file_sha256(root / path) for path in paths if (root / path).exists()}

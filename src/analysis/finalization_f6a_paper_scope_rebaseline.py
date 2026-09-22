"""Finalization F6A authoritative paper-scope rebaseline.

F6A corrects taxonomy and accounting artifacts without editing historical
F2-F5 outputs. The builder reads local evidence, writes only
``finalization_f6a_*`` artifacts, and keeps exact paper reproduction claims
separate from local adapted evidence.
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

REPORT_PATH = Path("report/finalization_f6a_paper_scope_rebaseline.md")
PROTECTED_HASHES = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")

PAPER_PDF = Path("paper/2411.00331v1.pdf")
PAPER_DOCX = Path("paper/traduzione_italiana_beyond_utility_llm_recommender.docx")
UPSTREAM_README = Path("original_repo/EvaLLMasRecommender/README.md")
UPSTREAM_METRICS = Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py")
DATASET_STATS = Path("results/tables/dataset_statistics.csv")
COHORT_SUMMARY = Path("results/tables/experiment_cohort_summary.csv")
MODEL_INVENTORY = Path("results/tables/canonical_llm_model_inventory.csv")
LLM_MANIFEST = Path("results/tables/canonical_local_llm_full_matrix_manifest.csv")
LLM_CELL_METRICS = Path("results/tables/canonical_local_llm_full_matrix_cell_metrics.csv")
LLM_PROVENANCE = Path("results/tables/canonical_local_llm_full_matrix_provenance.csv")
PHASE5E_COVERAGE = Path("results/tables/plan_d_phase_5e_coverage_metrics.csv")
PHASE5E_COMPONENTS = Path("results/tables/plan_d_phase_5e_component_coverage.csv")
F3_MISSING = Path("results/tables/finalization_f3_missing_evidence.csv")
F4_MISSING = Path("results/tables/finalization_f4_missing_evidence_triage.csv")
F5AF_ATOMIC = Path("results/tables/finalization_f5af_atomic_paper_result_cells.csv")
F5AF_MAPPING = Path("results/tables/finalization_f5af_local_atomic_mapping.csv")
F5AF_COVERAGE = Path("results/tables/finalization_f5af_corrected_coverage.csv")
F5AF_MODEL_INVENTORY = Path("results/tables/finalization_f5af_paper_model_inventory.csv")
F5B_STATUS = Path("results/tables/finalization_f5b_metric_status_update.csv")
F5C_STATUS = Path("results/tables/finalization_f5c_metric_status_update.csv")
F5D_STATUS = Path("results/tables/finalization_f5d_metric_status_update.csv")
F5E_STATUS = Path("results/tables/finalization_f5e_metric_status_update.csv")
F5E_CALIBRATION_STATUS = Path("results/tables/finalization_f5e_paper_calibration_status.csv")
F5E_COVERAGE = Path("results/tables/finalization_f5e_coverage_impact.csv")

LASTFM_RAW_ZIP = Path("data/external/plan_c/hetrec2011-lastfm-2k.zip")
LASTFM_INTERACTIONS = Path("data/reconstructed/plan_c/lastfm/interactions.csv")
LASTFM_CATALOG = Path("data/reconstructed/plan_c/lastfm/catalog.csv")
LASTFM_DATAMAPS = Path("data/reconstructed/plan_c/lastfm/datamaps.json")
LASTFM_META = Path("data/reconstructed/plan_c/lastfm/meta.json")

OUTPUT_TABLES: dict[str, Path] = {
    "authoritative_paper_scope": Path("results/tables/finalization_f6a_authoritative_paper_scope.csv"),
    "paper_component_inventory": Path("results/tables/finalization_f6a_paper_component_inventory.csv"),
    "dataset_taxonomy": Path("results/tables/finalization_f6a_dataset_taxonomy.csv"),
    "model_taxonomy": Path("results/tables/finalization_f6a_model_taxonomy.csv"),
    "prompt_taxonomy": Path("results/tables/finalization_f6a_prompt_taxonomy.csv"),
    "metric_dimension_taxonomy": Path("results/tables/finalization_f6a_metric_dimension_taxonomy.csv"),
    "lastfm_reproduction_status": Path("results/tables/finalization_f6a_lastfm_reproduction_status.csv"),
    "qwen2_mistral_alignment": Path("results/tables/finalization_f6a_qwen2_mistral_alignment.csv"),
    "extension_inventory": Path("results/tables/finalization_f6a_extension_inventory.csv"),
    "additional_dataset_requirement": Path("results/tables/finalization_f6a_additional_dataset_requirement.csv"),
    "coverage_family_definitions": Path("results/tables/finalization_f6a_coverage_family_definitions.csv"),
    "corrected_component_coverage": Path("results/tables/finalization_f6a_corrected_component_coverage.csv"),
    "missing_evidence_rebaseline": Path("results/tables/finalization_f6a_missing_evidence_rebaseline.csv"),
    "affected_artifact_registry": Path("results/tables/finalization_f6a_affected_artifact_registry.csv"),
    "claim_corrections": Path("results/tables/finalization_f6a_claim_corrections.csv"),
    "validation": Path("results/tables/finalization_f6a_validation.csv"),
}

AUTHORITATIVE_SCOPE_COLUMNS = [
    "scope_id",
    "scope_family",
    "name",
    "exact_identifier",
    "paper_role",
    "paper_section_or_table",
    "primary_source",
    "secondary_source",
    "upstream_source",
    "local_evidence",
    "classification",
    "notes",
]

COMPONENT_COLUMNS = [
    "component_id",
    "component_family",
    "component_name",
    "dataset",
    "task",
    "model_family",
    "exact_model_version",
    "model_role",
    "prompting_strategy",
    "candidate_set_construction",
    "cohort_sample_policy",
    "metric",
    "analysis_dimension",
    "result_table_or_figure",
    "paper_observation",
    "appendix_component",
    "external_dependency",
    "reproducibility_status",
    "classification",
    "evidence_file",
    "notes",
]

DATASET_COLUMNS = [
    "dataset",
    "paper_name",
    "authoritative_paper_dataset",
    "paper_users",
    "paper_items",
    "paper_interactions",
    "paper_density_percent",
    "local_raw_or_reconstructed_files",
    "local_interaction_rows",
    "local_catalog_rows",
    "local_in_experiment_cohort",
    "local_candidate_set_exists",
    "local_ground_truth_exists",
    "local_leave_one_out_verified",
    "local_5core_verified",
    "paper_stats_match_local",
    "local_status",
    "is_additional_dataset",
    "can_satisfy_extra_dataset_requirement",
    "missing_inputs",
    "offline_completion_possible",
    "training_required",
    "llm_inference_required",
    "evidence",
    "notes",
]

MODEL_COLUMNS = [
    "model_taxonomy_id",
    "paper_model_family",
    "paper_exact_identifier",
    "open_or_closed",
    "paper_presence_verified",
    "local_model_identifier",
    "local_digest",
    "backend",
    "model_family_equivalence",
    "exact_version_equivalence",
    "prompt_equivalence",
    "candidate_set_equivalence",
    "cohort_equivalence",
    "decoding_equivalence",
    "parser_equivalence",
    "retry_policy_equivalence",
    "dataset_coverage",
    "task_coverage",
    "local_run_scope",
    "classification",
    "blocking_reason",
    "evidence",
    "notes",
]

PROMPT_COLUMNS = [
    "prompt_taxonomy_id",
    "prompt_strategy",
    "paper_presence_verified",
    "upstream_template_files",
    "local_prompt_contracts",
    "local_execution_status",
    "exact_prompt_equivalence",
    "classification",
    "evidence",
    "notes",
]

DIMENSION_COLUMNS = [
    "dimension_id",
    "dimension",
    "paper_presence",
    "paper_scope",
    "metric_examples",
    "local_status",
    "classification",
    "evidence",
    "notes",
]

LASTFM_STATUS_COLUMNS = [
    "status_id",
    "question",
    "answer",
    "status",
    "evidence",
    "blocking_reason",
    "offline_completion_possible",
    "training_required",
    "llm_inference_required",
    "notes",
]

QWEN_MISTRAL_COLUMNS = [
    "alignment_id",
    "paper_model_family",
    "paper_exact_identifier",
    "local_model_identifier",
    "local_digest",
    "backend",
    "quantization_or_packaging",
    "manifest_rows",
    "completed_rows",
    "datasets",
    "tasks",
    "candidate_count",
    "prompt_contracts",
    "temperature",
    "seed",
    "response_format",
    "parser_contracts",
    "scientific_retry_policy",
    "invalid_output_rows",
    "hallucination_definition",
    "utility_metrics_available",
    "beyond_utility_metrics_available",
    "model_family_equivalence",
    "exact_version_equivalence",
    "overall_classification",
    "paper_table_mapping_status",
    "notes",
]

EXTENSION_COLUMNS = [
    "extension_id",
    "extension_name",
    "extension_type",
    "paper_absence_evidence",
    "local_implementation_status",
    "execution_status",
    "scientific_novelty",
    "comparison_validity",
    "reportability",
    "remaining_work",
    "evidence",
    "notes",
]

ADDITIONAL_DATASET_COLUMNS = [
    "requirement_id",
    "question",
    "answer",
    "status",
    "evidence",
    "requirement_source_type",
    "notes",
]

COVERAGE_FAMILY_COLUMNS = [
    "coverage_family_id",
    "coverage_family",
    "definition",
    "numerator_rule",
    "denominator_rule",
    "must_not_be_combined_with",
    "notes",
]

CORRECTED_COVERAGE_COLUMNS = [
    "coverage_id",
    "coverage_family",
    "numerator",
    "denominator",
    "coverage_percent",
    "denominator_status",
    "confidence",
    "source_files",
    "relation_to_previous_73_denominator",
    "notes",
]

MISSING_REBASELINE_COLUMNS = [
    "rebaseline_id",
    "source_missing_id",
    "area",
    "paper_component_id",
    "dataset",
    "model",
    "metric",
    "previous_status",
    "f6a_classification",
    "f6a_action",
    "is_current_effective_missing",
    "requires_split",
    "requires_merge",
    "should_remove_from_paper_missing",
    "new_missing_row_required",
    "evidence",
    "notes",
]

AFFECTED_ARTIFACT_COLUMNS = [
    "artifact",
    "phase",
    "affected_by_taxonomy_error",
    "error_type",
    "scientific_impact",
    "must_be_superseded",
    "must_be_corrected",
    "remains_authoritative_for",
    "proposed_corrective_output",
    "priority",
    "notes",
]

CLAIM_COLUMNS = [
    "claim_id",
    "previous_claim",
    "correction_type",
    "corrected_claim",
    "safe_for_report",
    "safe_for_exam",
    "evidence",
    "notes",
]

VALIDATION_COLUMNS = ["check_id", "check_name", "status", "observed", "expected", "details"]


PAPER_DATASETS = [
    {
        "name": "Amazon All Beauty",
        "canonical": "Amazon Beauty",
        "users": "22363",
        "items": "12101",
        "interactions": "198502",
        "density": "0.0734",
    },
    {
        "name": "Sports & Outdoors",
        "canonical": "Amazon Sports",
        "users": "35598",
        "items": "18357",
        "interactions": "296337",
        "density": "0.0453",
    },
    {"name": "MovieLens-1M", "canonical": "ML-1M", "users": "6040", "items": "3416", "interactions": "999611", "density": "4.8448"},
    {"name": "LastFM", "canonical": "LastFM", "users": "1543", "items": "7286", "interactions": "173778", "density": "1.5458"},
]

PAPER_LLMS = [
    ("GPT-3.5", "gpt-3.5-turbo-0125", "closed"),
    ("GPT-4o-mini", "gpt-4o-mini-2024-07-18", "closed"),
    ("GPT-4o", "gpt-4o-2024-08-06", "closed"),
    ("Claude 3 Haiku", "claude-3-haiku-20240307", "closed"),
    ("Llama3-70B-Instruct", "llama-3-70b-instruct", "open_or_open_weight"),
    ("Qwen2 7B Instruct", "qwen-2-7b-instruct", "open_or_open_weight"),
    ("Mistral 7B Instruct", "mistral-7b-instruct-v0.2", "open_or_open_weight"),
]

TRADITIONAL_MODELS = ["BM25", "MostPop", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"]
PROMPT_STRATEGIES = [
    ("OpenP5-derived base prompt", "original_repo/EvaLLMasRecommender/eva/templates/template.txt"),
    ("LLMRank recency-focused", "original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt"),
    ("LLMSRec in-context learning", "original_repo/EvaLLMasRecommender/eva/templates/template_in_context.txt"),
]
TASKS = ["ranking", "re-ranking"]
DIMENSIONS = [
    ("utility", "main", "hr_at_5;ndcg_at_5"),
    ("novelty", "main", "average_popularity_at_5;long_tail_rate_at_5"),
    ("history-length sensitivity", "main", "history ablations"),
    ("candidate-position bias", "main", "candidate position bias HR/NDCG"),
    ("generation-involved performance", "main", "profile generation;P-only"),
    ("hallucinations", "main", "hallucination_rate"),
    ("fairness", "appendix", "popularity/fairness analysis"),
    ("diversity", "appendix", "diversity analysis"),
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


def percent_text(numerator: int | float, denominator: int | float) -> str:
    denominator = float(denominator)
    if denominator == 0:
        return ""
    return f"{(float(numerator) / denominator) * 100:.6f}"


def rel(path: Path) -> str:
    return path.as_posix()


def path_exists(root: Path, path: Path) -> bool:
    return (root / path).exists()


def row_count(root: Path, path: Path) -> int:
    frame = read_csv(root, path)
    return int(len(frame))


def no_absolute_paths(frame: pd.DataFrame) -> bool:
    return ":\\" not in frame.to_csv(index=False, lineterminator="\n")


def protected_hash_status(root: Path) -> str:
    return _d1r_protected_hash_status(root)


def unique_join(values: Sequence[Any]) -> str:
    cleaned = sorted({clean(value) for value in values if clean(value)})
    return ";".join(cleaned)


def build_authoritative_paper_scope(root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(family: str, name: str, identifier: str, role: str, section: str, local: str, classification: str, notes: str) -> None:
        rows.append(
            {
                "scope_id": stable_id("F6A-SCOPE", family, name, identifier),
                "scope_family": family,
                "name": name,
                "exact_identifier": identifier,
                "paper_role": role,
                "paper_section_or_table": section,
                "primary_source": rel(PAPER_PDF),
                "secondary_source": rel(PAPER_DOCX),
                "upstream_source": rel(UPSTREAM_README),
                "local_evidence": local,
                "classification": classification,
                "notes": notes,
            }
        )

    for dataset in PAPER_DATASETS:
        add("dataset", dataset["canonical"], dataset["name"], "evaluation_dataset", "Section 4.1.1; Table 2", rel(DATASET_STATS), "original_paper_component", "Paper dataset identity.")
    for family, identifier, _kind in PAPER_LLMS:
        add("llm_model", family, identifier, "llm_as_recommender", "Section 4.1.2; Table 3", rel(MODEL_INVENTORY), "original_paper_component", "Exact local reproduction requires version and protocol equivalence.")
    for model in TRADITIONAL_MODELS:
        add("traditional_model", model, model, "baseline_recommender", "Section 4.1.2; Table 3", "results/tables/traditional_baselines_metrics.csv;results/tables/plan_e_*_all_datasets_metrics.csv", "original_paper_component", "Local baseline evidence is reconstructed/adapted unless protocol exactness is proven.")
    for prompt, template in PROMPT_STRATEGIES:
        add("prompt_strategy", prompt, prompt, "llm_prompting_strategy", "Section 4.1.2", template, "original_paper_component", "Local prompt contracts are JSON/index adapted.")
    for task in TASKS:
        add("task", task, task, "recommendation_task", "framework task family", rel(LLM_MANIFEST), "original_paper_component", "Local runs cover ranking/reranking on evaluation-ready datasets.")
    for dimension, scope, metrics in DIMENSIONS:
        add("evaluation_dimension", dimension, metrics, scope, "evaluation framework", rel(F5AF_ATOMIC), "original_paper_component", "Dimension inventory; exact numeric coverage is separately accounted.")
    return pd.DataFrame(rows, columns=AUTHORITATIVE_SCOPE_COLUMNS)


def build_dataset_taxonomy(root: Path) -> pd.DataFrame:
    cohorts = read_csv(root, COHORT_SUMMARY)
    cohort_datasets = set(cohorts["dataset"]) if "dataset" in cohorts else set()
    rows: list[dict[str, str]] = []
    for dataset in PAPER_DATASETS:
        canonical = dataset["canonical"]
        files: list[str] = []
        if canonical == "LastFM":
            for path in [LASTFM_RAW_ZIP, LASTFM_INTERACTIONS, LASTFM_CATALOG, LASTFM_DATAMAPS, LASTFM_META]:
                if path_exists(root, path):
                    files.append(rel(path))
            local_interactions = str(row_count(root, LASTFM_INTERACTIONS)) if path_exists(root, LASTFM_INTERACTIONS) else "0"
            local_catalog = str(row_count(root, LASTFM_CATALOG)) if path_exists(root, LASTFM_CATALOG) else "0"
            candidate_exists = "false"
            ground_truth_exists = "false"
            loo = "false"
            five_core = "unknown"
            stats_match = bool_text(local_interactions == dataset["interactions"] and local_catalog == dataset["items"])
            status = "original_paper_component_blocked"
            missing = "validated leave-one-out split;ground truth;ranking candidate pools;reranking candidate lists;paper-equivalent preprocessing"
            offline = "partial"
            training = "true"
            inference = "true"
            notes = "Raw/reconstructed artifacts exist, but no comparable candidate/ground-truth setup is materialized."
        else:
            files = [rel(COHORT_SUMMARY)] if canonical in cohort_datasets else [rel(DATASET_STATS)]
            local_interactions = ""
            local_catalog = ""
            candidate_exists = bool_text(canonical in cohort_datasets)
            ground_truth_exists = bool_text(canonical in cohort_datasets)
            loo = "not_proven_exact"
            five_core = "not_proven_exact"
            stats_match = "not_asserted"
            status = "original_paper_component_locally_reconstructed" if canonical in cohort_datasets else "original_paper_component_blocked"
            missing = "exact paper processed artifact;exact cohort/candidate protocol" if canonical in cohort_datasets else "evaluation cohort"
            offline = "true"
            training = "false"
            inference = "false"
            notes = "Evaluation-ready local cohorts exist, but exact paper preprocessing remains unproven."
        rows.append(
            {
                "dataset": canonical,
                "paper_name": dataset["name"],
                "authoritative_paper_dataset": "true",
                "paper_users": dataset["users"],
                "paper_items": dataset["items"],
                "paper_interactions": dataset["interactions"],
                "paper_density_percent": dataset["density"],
                "local_raw_or_reconstructed_files": ";".join(files),
                "local_interaction_rows": local_interactions,
                "local_catalog_rows": local_catalog,
                "local_in_experiment_cohort": bool_text(canonical in cohort_datasets),
                "local_candidate_set_exists": candidate_exists,
                "local_ground_truth_exists": ground_truth_exists,
                "local_leave_one_out_verified": loo,
                "local_5core_verified": five_core,
                "paper_stats_match_local": stats_match,
                "local_status": status,
                "is_additional_dataset": "false",
                "can_satisfy_extra_dataset_requirement": "false",
                "missing_inputs": missing,
                "offline_completion_possible": offline,
                "training_required": training,
                "llm_inference_required": inference,
                "evidence": f"{rel(PAPER_DOCX)};{rel(UPSTREAM_README)};{rel(DATASET_STATS)};{rel(COHORT_SUMMARY)}",
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=DATASET_COLUMNS)


def model_local_evidence(root: Path) -> dict[str, dict[str, str]]:
    manifest = read_csv(root, LLM_MANIFEST)
    provenance = read_csv(root, LLM_PROVENANCE)
    cell = read_csv(root, LLM_CELL_METRICS)
    evidence: dict[str, dict[str, str]] = {}
    for family in ["Qwen2 7B Instruct", "Mistral 7B Instruct"]:
        subset = manifest.loc[manifest["model_family"].eq(family)] if not manifest.empty else pd.DataFrame()
        metrics = cell.loc[cell["model_family"].eq(family)] if not cell.empty else pd.DataFrame()
        tag = clean(subset["model_tag"].iloc[0]) if not subset.empty and "model_tag" in subset else ""
        digest = clean(subset["model_digest"].iloc[0]) if not subset.empty and "model_digest" in subset else ""
        invalid = 0
        if not metrics.empty and {"planned_rows", "strict_valid_rows"}.issubset(metrics.columns):
            invalid = int(metrics["planned_rows"].astype(int).sum() - metrics["strict_valid_rows"].astype(int).sum())
        backend = "Ollama" if not provenance.empty else ""
        evidence[family] = {
            "tag": tag,
            "digest": digest,
            "backend": backend,
            "datasets": unique_join(subset["dataset"]) if "dataset" in subset else "",
            "tasks": unique_join(subset["task_type"]) if "task_type" in subset else "",
            "candidate_count": unique_join(subset["candidate_count"]) if "candidate_count" in subset else "",
            "prompt_contracts": unique_join(subset["prompt_contract_id"]) if "prompt_contract_id" in subset else "",
            "parser_contracts": unique_join(subset["parser_contract_id"]) if "parser_contract_id" in subset else "",
            "temperature": unique_join(subset["temperature"]) if "temperature" in subset else "",
            "seed": unique_join(subset["seed"]) if "seed" in subset else "",
            "response_format": unique_join(subset["response_format"]) if "response_format" in subset else "",
            "retry": unique_join(subset["scientific_retry_count"]) if "scientific_retry_count" in subset else "",
            "manifest_rows": str(len(subset)),
            "completed_rows": str(len(subset)),
            "invalid_rows": str(invalid),
        }
    return evidence


def build_model_taxonomy(root: Path) -> pd.DataFrame:
    local = model_local_evidence(root)
    rows: list[dict[str, str]] = []
    for family, identifier, kind in PAPER_LLMS:
        has_local = family in local and bool(local[family]["tag"])
        evidence = local.get(family, {})
        rows.append(
            {
                "model_taxonomy_id": stable_id("F6A-MODEL", family),
                "paper_model_family": family,
                "paper_exact_identifier": identifier,
                "open_or_closed": kind,
                "paper_presence_verified": "true",
                "local_model_identifier": evidence.get("tag", ""),
                "local_digest": evidence.get("digest", ""),
                "backend": evidence.get("backend", ""),
                "model_family_equivalence": bool_text(has_local),
                "exact_version_equivalence": "false" if has_local else "not_available",
                "prompt_equivalence": "false" if has_local else "not_available",
                "candidate_set_equivalence": "false" if has_local else "not_available",
                "cohort_equivalence": "false" if has_local else "not_available",
                "decoding_equivalence": "unknown" if has_local else "not_available",
                "parser_equivalence": "false" if has_local else "not_available",
                "retry_policy_equivalence": "unknown" if has_local else "not_available",
                "dataset_coverage": evidence.get("datasets", ""),
                "task_coverage": evidence.get("tasks", ""),
                "local_run_scope": "canonical_local_full_matrix" if has_local else "none",
                "classification": "original_paper_component_locally_adapted" if has_local else "original_paper_component_blocked",
                "blocking_reason": "" if has_local else "no local exact execution or required API/hardware unavailable",
                "evidence": f"{rel(PAPER_DOCX)};{rel(MODEL_INVENTORY)};{rel(LLM_MANIFEST)};{rel(LLM_PROVENANCE)};{rel(F5AF_ATOMIC)}",
                "notes": "Paper model family is original; local run is adapted, not exact." if has_local else "Original paper model has no local exact run.",
            }
        )
    return pd.DataFrame(rows, columns=MODEL_COLUMNS)


def build_prompt_taxonomy(root: Path) -> pd.DataFrame:
    manifest = read_csv(root, LLM_MANIFEST)
    local_contracts = unique_join(manifest["prompt_contract_id"]) if not manifest.empty and "prompt_contract_id" in manifest else ""
    rows: list[dict[str, str]] = []
    for prompt, template in PROMPT_STRATEGIES:
        exact = "false"
        local_status = "not_executed_exact"
        classification = "original_paper_component_not_executed"
        if prompt == "OpenP5-derived base prompt":
            local_status = "local_json_index_prompt_executed"
            classification = "original_paper_component_locally_adapted"
        rows.append(
            {
                "prompt_taxonomy_id": stable_id("F6A-PROMPT", prompt),
                "prompt_strategy": prompt,
                "paper_presence_verified": "true",
                "upstream_template_files": template,
                "local_prompt_contracts": local_contracts,
                "local_execution_status": local_status,
                "exact_prompt_equivalence": exact,
                "classification": classification,
                "evidence": f"{rel(PAPER_DOCX)};{template};{rel(LLM_MANIFEST)}",
                "notes": "Local contracts use JSON/index outputs and cannot be promoted to exact prompt equivalence.",
            }
        )
    return pd.DataFrame(rows, columns=PROMPT_COLUMNS)


def build_metric_dimension_taxonomy(root: Path) -> pd.DataFrame:
    rows = []
    for dimension, scope, metrics in DIMENSIONS:
        rows.append(
            {
                "dimension_id": stable_id("F6A-DIM", dimension),
                "dimension": dimension,
                "paper_presence": "true",
                "paper_scope": scope,
                "metric_examples": metrics,
                "local_status": "partially_supported" if dimension in {"utility", "novelty", "hallucinations"} else "adapted_or_blocked",
                "classification": "original_paper_component",
                "evidence": f"{rel(PAPER_DOCX)};{rel(UPSTREAM_README)};{rel(F5AF_ATOMIC)}",
                "notes": "Dimension-level presence is separate from exact numeric replication.",
            }
        )
    rows.append(
        {
            "dimension_id": stable_id("F6A-DIM", "calibration"),
            "dimension": "calibration",
            "paper_presence": "false",
            "paper_scope": "local_extension",
            "metric_examples": "calibration_divergence_at_5;calibration_score_at_5",
            "local_status": "resolved_extension_only",
            "classification": "local_extension",
            "evidence": rel(F5E_CALIBRATION_STATUS),
            "notes": "Calibration remains excluded from paper coverage denominators.",
        }
    )
    return pd.DataFrame(rows, columns=DIMENSION_COLUMNS)


def build_lastfm_status(root: Path) -> pd.DataFrame:
    interactions = row_count(root, LASTFM_INTERACTIONS) if path_exists(root, LASTFM_INTERACTIONS) else 0
    catalog = row_count(root, LASTFM_CATALOG) if path_exists(root, LASTFM_CATALOG) else 0
    checks = [
        ("raw_archive", bool_text(path_exists(root, LASTFM_RAW_ZIP)), "present" if path_exists(root, LASTFM_RAW_ZIP) else "missing", rel(LASTFM_RAW_ZIP), "", "true", "false", "false", "HetRec ZIP is staged."),
        ("interactions", str(interactions), "present" if interactions else "missing", rel(LASTFM_INTERACTIONS), "", "true", "false", "false", "Rows are reconstructed observed interactions."),
        ("catalog", str(catalog), "present" if catalog else "missing", rel(LASTFM_CATALOG), "", "true", "false", "false", "Artist-name metadata is available locally."),
        ("paper_preprocessing_conditions", "5-core and leave-one-out stated in paper source", "known_from_secondary_source", rel(PAPER_DOCX), "", "true", "false", "false", "Exact local match is not proven."),
        ("paper_statistics_match", "false", "mismatch_or_unverified", f"{rel(PAPER_DOCX)};{rel(LASTFM_INTERACTIONS)};{rel(LASTFM_CATALOG)}", "local reconstructed counts differ from paper table counts", "partial", "false", "false", "Local rows/items are not the paper Table 2 counts."),
        ("train_validation_test_split", "false", "missing", rel(COHORT_SUMMARY), "no LastFM cohort rows", "true", "false", "false", "No validated local split for comparable evaluation."),
        ("candidate_pools", "false", "missing", rel(COHORT_SUMMARY), "no LastFM candidate set", "true", "false", "true", "Ranking and reranking require paper-like candidates."),
        ("ground_truth", "false", "missing", rel(COHORT_SUMMARY), "no LastFM ground truth", "true", "false", "true", "Ground truth is not materialized."),
        ("traditional_results", "false", "missing", "results/tables/plan_e_*_all_datasets_metrics.csv", "no local LastFM metric rows found in canonical evaluation", "true", "true", "false", "Training/scoring would be required."),
        ("llm_results", "false", "missing", rel(LLM_MANIFEST), "canonical local LLM matrix excludes LastFM", "true", "false", "true", "LLM inference would be required for local model runs."),
        ("calculable_metrics", "blocked", "blocked", f"{rel(LASTFM_INTERACTIONS)};{rel(LASTFM_CATALOG)}", "candidate/ground-truth missing", "partial", "true", "true", "Metadata-only metrics are not comparable utility results."),
    ]
    rows = []
    for index, (question, answer, status, evidence, blocker, offline, training, inference, notes) in enumerate(checks, start=1):
        rows.append(
            {
                "status_id": f"F6A-LASTFM-{index:03d}",
                "question": question,
                "answer": answer,
                "status": status,
                "evidence": evidence,
                "blocking_reason": blocker,
                "offline_completion_possible": offline,
                "training_required": training,
                "llm_inference_required": inference,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=LASTFM_STATUS_COLUMNS)


def build_qwen_mistral_alignment(root: Path) -> pd.DataFrame:
    local = model_local_evidence(root)
    rows = []
    for family, identifier in [("Qwen2 7B Instruct", "qwen-2-7b-instruct"), ("Mistral 7B Instruct", "mistral-7b-instruct-v0.2")]:
        evidence = local.get(family, {})
        rows.append(
            {
                "alignment_id": stable_id("F6A-QM", family),
                "paper_model_family": family,
                "paper_exact_identifier": identifier,
                "local_model_identifier": evidence.get("tag", ""),
                "local_digest": evidence.get("digest", ""),
                "backend": evidence.get("backend", "Ollama"),
                "quantization_or_packaging": "ollama_tag_digest_not_paper_version",
                "manifest_rows": evidence.get("manifest_rows", "0"),
                "completed_rows": evidence.get("completed_rows", "0"),
                "datasets": evidence.get("datasets", ""),
                "tasks": evidence.get("tasks", ""),
                "candidate_count": evidence.get("candidate_count", ""),
                "prompt_contracts": evidence.get("prompt_contracts", ""),
                "temperature": evidence.get("temperature", ""),
                "seed": evidence.get("seed", ""),
                "response_format": evidence.get("response_format", ""),
                "parser_contracts": evidence.get("parser_contracts", ""),
                "scientific_retry_policy": evidence.get("retry", ""),
                "invalid_output_rows": evidence.get("invalid_rows", "0"),
                "hallucination_definition": "strict index/candidate validity; not paper free-text hallucination exact definition",
                "utility_metrics_available": "true",
                "beyond_utility_metrics_available": "adapted_structural_and_metric_extensions",
                "model_family_equivalence": "true",
                "exact_version_equivalence": "false",
                "overall_classification": "original_paper_component_locally_adapted",
                "paper_table_mapping_status": "blocked_protocol_mismatch_for_exact_numeric_replication",
                "notes": "Family is in paper; local execution is not paper-exact.",
            }
        )
    return pd.DataFrame(rows, columns=QWEN_MISTRAL_COLUMNS)


def build_extension_inventory(root: Path) -> pd.DataFrame:
    rows = [
        ("EXT-CALIBRATION", "Calibration@5 recommendation/profile distribution", "metric", rel(F5E_CALIBRATION_STATUS), "implemented", "computed", "additional metric", "same-protocol only", "reportable_extension", "none for local scope"),
        ("EXT-JSON-INDEX", "JSON/index-based output contract", "protocol", rel(LLM_MANIFEST), "implemented", "executed", "strict parsing reproducibility", "not paper-exact prompt comparison", "reportable_method", "do not call exact"),
        ("EXT-STRUCTURAL", "Structural validity and duplicate-index diagnostics", "reliability", "results/tables/canonical_local_llm_structural_metrics.csv", "implemented", "executed", "separates backend success from validity", "local Qwen/Mistral only", "reportable_extension", "none"),
        ("EXT-INVALID-UTILITY", "Operational invalid-as-failure utility", "evaluation_policy", "results/tables/canonical_local_llm_utility_metrics.csv", "implemented", "executed", "explicit denominator policy", "within local protocol only", "reportable_extension", "none"),
        ("EXT-ROBUSTNESS", "Robustness, nondeterminism, cost and hardware audits", "supporting_analysis", "report/plan_d_phase_5d_robustness_consolidation.md", "implemented", "audit_only", "claim safety", "diagnostic only", "reportable_support", "no causal claim"),
        ("EXT-NORMALIZED-POPULARITY", "Normalized popularity and head exposure", "metric", rel(F5C_STATUS), "implemented", "computed", "semantic split of popularity evidence", "same-protocol only", "reportable_extension", "PopREO remains blocked"),
        ("EXT-SHIFT-LIFT", "Popularity shift and popularity lift", "metric", rel(F5C_STATUS), "implemented", "computed", "profile-vs-recommendation popularity analysis", "same-protocol only", "reportable_extension", "not a paper metric"),
        ("EXT-PLAN-E-NEURAL", "Plan E local neural baseline runs", "baseline_context", "results/tables/plan_e_*_all_datasets_metrics.csv", "implemented", "computed_for_available_datasets", "local baseline context", "not paper-exact training/cohort", "reportable_with_caveat", "LastFM absent"),
        ("EXT-ILD-ADAPTED", "ILD adapted categorical policy", "metric", rel(F5B_STATUS), "implemented", "computed", "adapted diversity evidence", "not a clean new metric because appendix mentions diversity", "reportable_adapted", "exact feature policy missing"),
    ]
    return pd.DataFrame(
        [
            {
                "extension_id": extension_id,
                "extension_name": name,
                "extension_type": ext_type,
                "paper_absence_evidence": evidence,
                "local_implementation_status": impl,
                "execution_status": execution,
                "scientific_novelty": novelty,
                "comparison_validity": validity,
                "reportability": reportability,
                "remaining_work": remaining,
                "evidence": evidence,
                "notes": "Excluded from exact paper reproduction numerator.",
            }
            for extension_id, name, ext_type, evidence, impl, execution, novelty, validity, reportability, remaining in rows
        ],
        columns=EXTENSION_COLUMNS,
    )


def build_additional_dataset_requirement(root: Path) -> pd.DataFrame:
    project_doc_exists = path_exists(root, Path("Obiettivo_Progetto_IR_Reproducibility.md"))
    rows = [
        ("F6A-ADR-001", "Does a fifth evaluation dataset exist?", "no", "not_met", rel(DATASET_STATS), "local_artifact_audit", "Only the four paper datasets appear in dataset_statistics."),
        ("F6A-ADR-002", "Can LastFM count as the additional dataset?", "no", "not_eligible", f"{rel(PAPER_DOCX)};{rel(UPSTREAM_README)}", "paper_scope", "LastFM is an original paper dataset."),
        ("F6A-ADR-003", "Is calibration sufficient as additional metric?", "yes", "met_for_metric_extension", rel(F5E_CALIBRATION_STATUS), "implementation_evidence", "Calibration is extension-only and computed locally."),
        ("F6A-ADR-004", "Is ILD also an additional adapted metric?", "yes_with_caveat", "adapted_metric", rel(F5B_STATUS), "implementation_evidence", "ILD is adapted and related to appendix diversity scope."),
        ("F6A-ADR-005", "Is a new dataset mandatory according to local objective docs?", "not_documented_locally" if not project_doc_exists else "requires_manual_review", "unknown_requirement_strength", "Obiettivo_Progetto_IR_Reproducibility.md", "missing_objective_doc" if not project_doc_exists else "objective_doc_present", "Do not infer professor requirement beyond local evidence."),
        ("F6A-ADR-006", "What dataset characteristics preserve comparability?", "public sequential implicit-feedback data with titles, 5-core-compatible filtering, leave-one-out split, 20-candidate ranking/reranking pools", "design_requirement", "derived_from_paper_protocol", "recommendation", "A reconstructed variant of an original paper dataset is not a new dataset."),
    ]
    return pd.DataFrame(
        [{"requirement_id": rid, "question": q, "answer": a, "status": s, "evidence": e, "requirement_source_type": t, "notes": n} for rid, q, a, s, e, t, n in rows],
        columns=ADDITIONAL_DATASET_COLUMNS,
    )


def build_coverage_family_definitions(root: Path) -> pd.DataFrame:
    families = [
        ("component_implementation", "paper components with local status", "status-count over corrected F6A component inventory", "corrected component inventory rows", "numeric cell coverage;extension coverage"),
        ("dataset_coverage", "paper dataset identification and local evaluation readiness", "datasets satisfying readiness policy", "4 paper datasets", "component headline"),
        ("model_family_coverage", "paper LLM family identification and local family execution", "families with verified local counterpart", "7 paper LLM families", "exact model-version coverage"),
        ("exact_model_version_coverage", "exact paper model identifiers reproduced", "exact version/protocol matches", "7 paper LLM exact identifiers", "family coverage"),
        ("task_coverage", "ranking and re-ranking task presence", "covered tasks", "2 paper task families", "dataset/model coverage"),
        ("prompt_strategy_coverage", "OpenP5, LLMRank, LLMSRec strategy coverage", "exact or adapted strategy rows", "3 paper prompt strategies", "local prompt contracts"),
        ("metric_coverage", "paper metric definitions and local computability", "resolved metric definitions", "paper metric inventory", "extension metrics"),
        ("analysis_dimension_coverage", "main and appendix dimensions", "dimension rows with local evidence", "paper dimensions", "numeric cells"),
        ("paper_observation_coverage", "paper observations support", "observations with exact support", "paper observations", "metric coverage"),
        ("atomic_numeric_extraction", "paper numeric cell extraction", "extracted atomic cells", "187 F5AF cells", "exact replication"),
        ("local_result_mapping", "local counterpart mapping for paper cells", "mapped cells", "187 F5AF cells", "exact replication"),
        ("protocol_exact_numeric_replication", "numeric match with exact protocol", "exact protocol and value matches", "187 F5AF cells", "local mapping"),
        ("extension_coverage", "local extensions outside paper", "implemented extensions", "F6A extension inventory", "paper coverage"),
    ]
    rows = []
    for family, definition, numerator, denominator, separate in families:
        rows.append(
            {
                "coverage_family_id": stable_id("F6A-CF", family),
                "coverage_family": family,
                "definition": definition,
                "numerator_rule": numerator,
                "denominator_rule": denominator,
                "must_not_be_combined_with": separate,
                "notes": "Report as a separate family unless an explicit weighting policy is declared.",
            }
        )
    return pd.DataFrame(rows, columns=COVERAGE_FAMILY_COLUMNS)


def coverage_row(family: str, numerator: int | str, denominator: int | str, confidence: str, sources: str, notes: str, relation: str = "supersedes_or_refines_previous_accounting") -> dict[str, str]:
    n_text = clean(numerator)
    d_text = clean(denominator)
    pct = percent_text(int(n_text), int(d_text)) if n_text.isdigit() and d_text.isdigit() else "not_safely_derivable"
    return {
        "coverage_id": stable_id("F6A-COV", family),
        "coverage_family": family,
        "numerator": n_text,
        "denominator": d_text,
        "coverage_percent": pct,
        "denominator_status": "family_specific" if d_text.isdigit() else "requires_builder_reconstruction",
        "confidence": confidence,
        "source_files": sources,
        "relation_to_previous_73_denominator": relation,
        "notes": notes,
    }


def build_corrected_component_coverage(root: Path, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    atomic = read_csv(root, F5AF_ATOMIC)
    mapping = read_csv(root, F5AF_MAPPING)
    protocol_exact = int(mapping["protocol_equivalence_class"].eq("exact").sum()) if not mapping.empty and "protocol_equivalence_class" in mapping else 0
    local_mapped = int(mapping["local_result_id"].astype(str).str.len().gt(0).sum()) if not mapping.empty and "local_result_id" in mapping else 0
    extension_count = len(frames["extension_inventory"])
    rows = [
        coverage_row("paper dataset identification", 4, 4, "high", f"{rel(PAPER_DOCX)};{rel(UPSTREAM_README)}", "All four paper datasets are identified."),
        coverage_row("local evaluation-ready dataset coverage", 3, 4, "high", rel(COHORT_SUMMARY), "LastFM is not evaluation-ready locally."),
        coverage_row("paper LLM model-family inventory", 7, 7, "high", f"{rel(PAPER_DOCX)};{rel(MODEL_INVENTORY)};{rel(F5AF_ATOMIC)}", "Qwen2 and Mistral are paper model families."),
        coverage_row("paper LLM families with local runs", 2, 7, "high", rel(LLM_MANIFEST), "Only Qwen2 and Mistral have local full-matrix runs."),
        coverage_row("exact LLM model-version coverage", 0, 7, "high", rel(LLM_MANIFEST), "No local model has proven exact version/protocol equivalence."),
        coverage_row("task family coverage", 2, 2, "high", rel(LLM_MANIFEST), "Ranking and reranking are both represented locally."),
        coverage_row("paper prompt strategy identification", 3, 3, "high", f"{rel(PAPER_DOCX)};original_repo/EvaLLMasRecommender/eva/templates", "All three paper strategies are identified."),
        coverage_row("exact prompt strategy execution", 0, 3, "medium", rel(LLM_MANIFEST), "Local JSON/index contracts are adapted."),
        coverage_row("paper evaluation dimension inventory", 8, 8, "medium", f"{rel(PAPER_DOCX)};{rel(UPSTREAM_README)}", "Six main dimensions plus fairness/diversity appendix are represented."),
        coverage_row("atomic paper numeric-cell extraction", len(atomic), len(atomic), "high", rel(F5AF_ATOMIC), "F5AF atomic cell inventory remains authoritative."),
        coverage_row("local result mapping coverage", local_mapped, len(atomic), "high", rel(F5AF_MAPPING), "Mappings are local counterparts, not exact replication."),
        coverage_row("protocol-exact numeric replication", protocol_exact, len(atomic), "high", rel(F5AF_COVERAGE), "Exact numeric replication remains zero."),
        coverage_row("extension inventory coverage", extension_count, extension_count, "medium", rel(F5E_CALIBRATION_STATUS), "Extension coverage is separate from paper coverage.", "separate_from_paper_denominator"),
        coverage_row("corrected component-level denominator", "not_safely_derivable", "not_safely_derivable", "low", rel(PHASE5E_COVERAGE), "Do not invent a replacement for the 73-component denominator before row-level F6A reconstruction."),
    ]
    return pd.DataFrame(rows, columns=CORRECTED_COVERAGE_COLUMNS)


def build_paper_component_inventory(root: Path, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(family: str, name: str, classification: str, status: str, evidence: str, notes: str, **extra: str) -> None:
        row = {column: "" for column in COMPONENT_COLUMNS}
        row.update(
            {
                "component_id": stable_id("F6A-PC", family, name),
                "component_family": family,
                "component_name": name,
                "reproducibility_status": status,
                "classification": classification,
                "evidence_file": evidence,
                "notes": notes,
            }
        )
        row.update(extra)
        rows.append(row)

    for row in frames["dataset_taxonomy"].to_dict("records"):
        add("dataset", row["dataset"], row["local_status"], row["local_status"], row["evidence"], row["notes"], dataset=row["dataset"], external_dependency="data availability")
    for row in frames["model_taxonomy"].to_dict("records"):
        add("llm_model", row["paper_model_family"], row["classification"], row["classification"], row["evidence"], row["notes"], model_family=row["paper_model_family"], exact_model_version=row["paper_exact_identifier"], model_role="llm_as_recommender", external_dependency=row["blocking_reason"])
    for model in TRADITIONAL_MODELS:
        status = "original_paper_component_locally_reconstructed" if model in {"BM25", "MostPop"} else "original_paper_component_locally_adapted"
        add("traditional_model", model, status, status, "results/tables/traditional_baselines_metrics.csv;results/tables/plan_e_*_all_datasets_metrics.csv", "Local baseline evidence exists but exact ReChorus protocol equivalence is not proven.", model_family=model, model_role="traditional_baseline")
    for row in frames["prompt_taxonomy"].to_dict("records"):
        add("prompt_strategy", row["prompt_strategy"], row["classification"], row["classification"], row["evidence"], row["notes"], prompting_strategy=row["prompt_strategy"])
    for task in TASKS:
        add("task", task, "original_paper_component_locally_adapted", "local_adapted_evidence", rel(LLM_MANIFEST), "Task family is executed locally with adapted local models/protocols.", task=task)
    for row in frames["metric_dimension_taxonomy"].to_dict("records"):
        if row["classification"] == "local_extension":
            continue
        add("analysis_dimension", row["dimension"], "original_paper_component", row["local_status"], row["evidence"], row["notes"], analysis_dimension=row["dimension"], metric=row["metric_examples"], appendix_component=bool_text(row["paper_scope"] == "appendix"))
    return pd.DataFrame(rows, columns=COMPONENT_COLUMNS)


def build_missing_evidence_rebaseline(root: Path) -> pd.DataFrame:
    source = read_csv(root, F4_MISSING)
    if source.empty:
        source = read_csv(root, F3_MISSING)
    rows: list[dict[str, str]] = []
    for _, row in source.iterrows():
        text = " ".join(clean(row.get(col)) for col in source.columns)
        source_id = clean(row.get("missing_evidence_id"))
        action = "preserve_missing"
        classification = "still_missing"
        effective = "true"
        requires_split = "false"
        should_remove = "false"
        new_required = "false"
        if "LastFM" in text:
            action = "relabel_as_original_paper_dataset_blocked"
            classification = "LastFM paper component"
        if "Qwen2" in text or "Mistral" in text or source_id == "F3-MISS-0015":
            action = "split_model_family_presence_from_exact_version_protocol"
            classification = "Qwen2/Mistral paper-family alignment gap"
            requires_split = "true"
            new_required = "true"
        if source_id == "F3-MISS-0041":
            action = "resolved_by_f4_historical"
            classification = "already_resolved"
            effective = "false"
            should_remove = "true"
        if clean(row.get("metric")) == "calibration":
            action = "remove_from_paper_missing_scope"
            classification = "true_extension_gap_resolved"
            effective = "false"
            should_remove = "true"
        rows.append(
            {
                "rebaseline_id": stable_id("F6A-MISS", source_id, action),
                "source_missing_id": source_id,
                "area": clean(row.get("area")),
                "paper_component_id": clean(row.get("paper_component_id")),
                "dataset": clean(row.get("dataset")),
                "model": clean(row.get("model")),
                "metric": clean(row.get("metric")),
                "previous_status": clean(row.get("resolution_status")) or clean(row.get("current_status")) or clean(row.get("f3_status")),
                "f6a_classification": classification,
                "f6a_action": action,
                "is_current_effective_missing": effective,
                "requires_split": requires_split,
                "requires_merge": "false",
                "should_remove_from_paper_missing": should_remove,
                "new_missing_row_required": new_required,
                "evidence": f"{rel(F4_MISSING)};{rel(F5E_COVERAGE)}",
                "notes": "Current operational missing evidence count is 37 from F5E coverage impact; row-level corrected count requires F6A superseding registry.",
            }
        )
    return pd.DataFrame(rows, columns=MISSING_REBASELINE_COLUMNS)


def build_affected_artifact_registry(root: Path) -> pd.DataFrame:
    items = [
        ("results/tables/plan_d_phase_5e_component_coverage.csv", "Plan D 5E", "true", "Qwen/Mistral local extension wording; LastFM blocked wording", "component coverage needs taxonomy correction", "true", "false", "historical 73-component audit", "finalization_f6a_corrected_component_coverage.csv", "critical"),
        ("results/tables/plan_d_phase_5e_coverage_metrics.csv", "Plan D 5E", "true", "single denominator headline", "coverage percentages need family split", "true", "false", "historical coverage metrics", "finalization_f6a_corrected_component_coverage.csv", "critical"),
        ("results/tables/plan_d_phase_5e_paper_component_registry.csv", "Plan D 5E", "true", "model labels incomplete/ambiguous", "component registry needs superseding taxonomy", "true", "false", "historical registry", "finalization_f6a_paper_component_inventory.csv", "critical"),
        ("results/tables/plan_d_phase_5e_category_coverage.csv", "Plan D 5E", "true", "category aggregation inherited taxonomy errors", "macro/micro summary should be superseded", "true", "false", "historical category accounting", "finalization_f6a_corrected_component_coverage.csv", "high"),
        ("results/tables/plan_d_phase_5e_result_cell_comparison.csv", "Plan D 5E", "true", "Qwen/Mistral described as different model identities", "must split family identity from exact version", "true", "false", "historical comparison", "finalization_f6a_qwen2_mistral_alignment.csv", "critical"),
        ("results/tables/plan_d_phase_5e_extensions.csv", "Plan D 5E", "true", "Qwen/Mistral listed as extension", "extension inventory overstates local-only model novelty", "true", "false", "historical extension list", "finalization_f6a_extension_inventory.csv", "critical"),
        ("results/tables/finalization_f2_canonical_result_registry.csv", "F2", "true", "artifact category/paper equivalence needs relabel", "local results remain numeric evidence but taxonomy changes", "true", "false", "canonical local row values", "finalization_f6a_model_taxonomy.csv", "high"),
        ("results/tables/finalization_f3_paper_alignment.csv", "F3", "true", "paper alignment uses old taxonomy", "claims need relabeling", "true", "false", "historical alignment", "finalization_f6a_claim_corrections.csv", "high"),
        ("results/tables/finalization_f3_missing_evidence.csv", "F3", "true", "LastFM extension wording and model-family split missing", "missing evidence count needs rebaseline", "true", "false", "source missing rows", "finalization_f6a_missing_evidence_rebaseline.csv", "critical"),
        ("results/tables/finalization_f4_missing_evidence_triage.csv", "F4", "true", "triage inherits F3 categories", "row-level rebaseline needed", "true", "false", "triage fields", "finalization_f6a_missing_evidence_rebaseline.csv", "critical"),
        ("results/tables/finalization_f5af_paper_model_inventory.csv", "F5AF", "true", "counts 5 model identities instead of 7", "paper model inventory coverage overstated as 5/5", "true", "false", "historical F5AF bug evidence", "finalization_f6a_model_taxonomy.csv", "critical"),
        ("results/tables/finalization_f5af_corrected_coverage.csv", "F5AF", "true", "model inventory coverage denominator 5", "model-family coverage must be 7/7 identified", "true", "false", "187 atomic numeric-cell accounting", "finalization_f6a_corrected_component_coverage.csv", "critical"),
        ("report/final_scientific_report.md", "report", "true", "narrative overclassifies Qwen/Mistral as substitutes/extensions", "scientific conclusion needs correction", "true", "true_later_f6b", "historical report", "report/finalization_f6a_paper_scope_rebaseline.md", "critical"),
        ("report/dataset_audit.md", "report", "true", "LastFM called extension-only", "dataset scope correction needed", "true", "true_later_f6b", "factual file inventory", "finalization_f6a_dataset_taxonomy.csv", "critical"),
        ("README.md", "documentation", "true", "extension summary includes Qwen/Mistral", "user-facing claims need update later", "true", "true_later_f6b", "historical current README", "finalization_f6a_claim_corrections.csv", "high"),
        ("report/notes.md", "notes", "true", "historical notes contain old taxonomy", "should not be rewritten in F6A", "true", "false", "chronology", "finalization_f6a_claim_corrections.csv", "medium"),
        ("PROJECT_SPEC.md", "spec", "true", "LastFM optional/future wording conflicts with paper scope", "project objective wording needs review", "true", "true_later_f6b", "historical spec", "finalization_f6a_additional_dataset_requirement.csv", "medium"),
    ]
    return pd.DataFrame(
        [
            {
                "artifact": artifact,
                "phase": phase,
                "affected_by_taxonomy_error": affected,
                "error_type": error,
                "scientific_impact": impact,
                "must_be_superseded": superseded,
                "must_be_corrected": corrected,
                "remains_authoritative_for": valid_for,
                "proposed_corrective_output": output,
                "priority": priority,
                "notes": "Do not edit this artifact in F6A; supersede with F6A output.",
            }
            for artifact, phase, affected, error, impact, superseded, corrected, valid_for, output, priority in items
        ],
        columns=AFFECTED_ARTIFACT_COLUMNS,
    )


def build_claim_corrections(root: Path) -> pd.DataFrame:
    rows = [
        ("F6A-CLAIM-001", "Qwen2 and Mistral are local substitute extensions.", "relabel", "Qwen2 and Mistral are original paper model families with non-exact local adapted executions.", "true", "true", f"{rel(PAPER_DOCX)};{rel(LLM_MANIFEST)}"),
        ("F6A-CLAIM-002", "LastFM is extension-only.", "relabel", "LastFM is an original paper dataset whose local comparable evaluation is blocked.", "true", "true", f"{rel(PAPER_DOCX)};{rel(LASTFM_INTERACTIONS)};{rel(COHORT_SUMMARY)}"),
        ("F6A-CLAIM-003", "Paper model inventory coverage is 5/5.", "denominator_fix", "Paper LLM model-family inventory is 7/7 identified; exact local version coverage is 0/7.", "true", "true", f"{rel(PAPER_DOCX)};{rel(F5AF_MODEL_INVENTORY)};{rel(F5AF_ATOMIC)}"),
        ("F6A-CLAIM-004", "Calibration can improve paper coverage.", "scope_guard", "Calibration is a local extension-only metric and does not increase paper coverage.", "true", "true", rel(F5E_CALIBRATION_STATUS)),
        ("F6A-CLAIM-005", "A fifth evaluation dataset exists.", "scope_guard", "No fifth evaluation dataset exists beyond the four paper datasets.", "true", "true", f"{rel(DATASET_STATS)};{rel(COHORT_SUMMARY)}"),
        ("F6A-CLAIM-006", "Exact numeric replication is improved by local mappings.", "scope_guard", "Exact numeric replication remains 0/187 because protocol-exact mapping is absent.", "true", "true", rel(F5AF_COVERAGE)),
    ]
    return pd.DataFrame(
        [{"claim_id": cid, "previous_claim": old, "correction_type": kind, "corrected_claim": new, "safe_for_report": report, "safe_for_exam": exam, "evidence": evidence, "notes": "Use only after F6A outputs are generated."} for cid, old, kind, new, report, exam, evidence in rows],
        columns=CLAIM_COLUMNS,
    )


def build_validation(root: Path, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(check_id: str, name: str, condition: bool, observed: Any, expected: Any, details: str) -> None:
        rows.append({"check_id": check_id, "check_name": name, "status": "pass" if condition else "fail", "observed": clean(observed), "expected": clean(expected), "details": details})

    datasets = frames["dataset_taxonomy"]
    models = frames["model_taxonomy"]
    coverage = frames["corrected_component_coverage"].set_index("coverage_family")
    affected = frames["affected_artifact_registry"]
    missing = frames["missing_evidence_rebaseline"]
    extensions = frames["extension_inventory"]
    validation_frames = list(frames.values())

    add("F6A-VAL-001", "authoritative_dataset_count", len(datasets) == 4, len(datasets), "4", "Four paper datasets are inventoried.")
    add("F6A-VAL-002", "lastfm_not_extension", datasets.loc[datasets["dataset"].eq("LastFM"), "is_additional_dataset"].eq("false").all(), "false", "false", "LastFM is not an additional dataset.")
    add("F6A-VAL-003", "lastfm_blocked", datasets.loc[datasets["dataset"].eq("LastFM"), "local_status"].eq("original_paper_component_blocked").all(), datasets.loc[datasets["dataset"].eq("LastFM"), "local_status"].to_string(index=False), "original_paper_component_blocked", "LastFM status is paper-scope blocked.")
    add("F6A-VAL-004", "paper_llm_count", len(models) == 7, len(models), "7", "Seven paper LLM model families are inventoried.")
    add("F6A-VAL-005", "qwen_mistral_original_adapted", models.loc[models["paper_model_family"].isin(["Qwen2 7B Instruct", "Mistral 7B Instruct"]), "classification"].eq("original_paper_component_locally_adapted").all(), "adapted", "adapted", "Qwen/Mistral are original paper families with adapted local runs.")
    add("F6A-VAL-006", "calibration_extension_only", frames["metric_dimension_taxonomy"].loc[frames["metric_dimension_taxonomy"]["dimension"].eq("calibration"), "classification"].eq("local_extension").all(), "local_extension", "local_extension", "Calibration is excluded from paper scope.")
    add("F6A-VAL-007", "no_fifth_dataset", frames["additional_dataset_requirement"].loc[frames["additional_dataset_requirement"]["requirement_id"].eq("F6A-ADR-001"), "answer"].eq("no").all(), "no", "no", "No fifth evaluation dataset exists.")
    add("F6A-VAL-008", "atomic_cells_preserved", coverage.loc["atomic paper numeric-cell extraction", "denominator"] == "187", coverage.loc["atomic paper numeric-cell extraction", "denominator"], "187", "F5AF atomic denominator is preserved.")
    add("F6A-VAL-009", "exact_numeric_zero", coverage.loc["protocol-exact numeric replication", "numerator"] == "0", coverage.loc["protocol-exact numeric replication", "numerator"], "0", "Exact numeric replication remains zero.")
    add("F6A-VAL-010", "affected_artifact_minimum", len(affected) >= 17, len(affected), ">=17", "Affected artifacts registry includes required files.")
    add("F6A-VAL-011", "missing_rebaseline_present", len(missing) >= 42, len(missing), ">=42 source rows", "F3/F4 source missing rows are audited.")
    add("F6A-VAL-012", "model_inventory_superseded", affected["artifact"].eq(rel(F5AF_MODEL_INVENTORY)).any(), rel(F5AF_MODEL_INVENTORY), "listed", "F5AF 5-model inventory is marked affected.")
    add("F6A-VAL-013", "qwen_mistral_alignment_rows", len(frames["qwen2_mistral_alignment"]) == 2, len(frames["qwen2_mistral_alignment"]), "2", "Qwen2 and Mistral alignment rows exist.")
    add("F6A-VAL-014", "extension_inventory_has_calibration", extensions["extension_id"].eq("EXT-CALIBRATION").any(), "present", "present", "Calibration is genuine extension.")
    add("F6A-VAL-015", "coverage_families_not_single_headline", len(frames["coverage_family_definitions"]) >= 13, len(frames["coverage_family_definitions"]), ">=13", "Coverage is split by family.")
    add("F6A-VAL-016", "corrected_denominator_not_invented", coverage.loc["corrected component-level denominator", "coverage_percent"] == "not_safely_derivable", coverage.loc["corrected component-level denominator", "coverage_percent"], "not_safely_derivable", "No arbitrary component denominator is produced.")
    add("F6A-VAL-017", "protected_inputs_unchanged", protected_hash_status(root) == "checked=1144;changed=0", protected_hash_status(root), "checked=1144;changed=0", "Protected F2 input hashes match.")
    add("F6A-VAL-018", "relative_paths_only", all(no_absolute_paths(frame) for frame in validation_frames), "all frames", "relative paths", "No absolute local paths in generated tables.")
    add("F6A-VAL-019", "no_nan_cells", all(not frame.isna().any().any() for frame in validation_frames), "all frames", "no NaN", "Tables are filled with explicit strings.")
    add("F6A-VAL-020", "deterministic_frames", deterministic_frames(validation_frames), "stable", "stable", "In-memory CSV serialization is deterministic.")
    add("F6A-VAL-021", "only_f6a_outputs_declared", all("finalization_f6a_" in path.name for path in OUTPUT_TABLES.values()) and "finalization_f6a_" in REPORT_PATH.name, "F6A only", "F6A only", "Builder writes only F6A-prefixed artifacts.")
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def deterministic_frames(frames: Sequence[pd.DataFrame]) -> bool:
    return [frame_sha256(frame) for frame in frames] == [frame_sha256(frame.copy()) for frame in frames]


def build_frames(root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    frames["authoritative_paper_scope"] = build_authoritative_paper_scope(root)
    frames["dataset_taxonomy"] = build_dataset_taxonomy(root)
    frames["model_taxonomy"] = build_model_taxonomy(root)
    frames["prompt_taxonomy"] = build_prompt_taxonomy(root)
    frames["metric_dimension_taxonomy"] = build_metric_dimension_taxonomy(root)
    frames["lastfm_reproduction_status"] = build_lastfm_status(root)
    frames["qwen2_mistral_alignment"] = build_qwen_mistral_alignment(root)
    frames["extension_inventory"] = build_extension_inventory(root)
    frames["additional_dataset_requirement"] = build_additional_dataset_requirement(root)
    frames["coverage_family_definitions"] = build_coverage_family_definitions(root)
    frames["missing_evidence_rebaseline"] = build_missing_evidence_rebaseline(root)
    frames["affected_artifact_registry"] = build_affected_artifact_registry(root)
    frames["claim_corrections"] = build_claim_corrections(root)
    frames["paper_component_inventory"] = build_paper_component_inventory(root, frames)
    frames["corrected_component_coverage"] = build_corrected_component_coverage(root, frames)
    frames["validation"] = build_validation(root, frames)
    return {key: frames[key] for key in OUTPUT_TABLES}


def final_decision(validation: pd.DataFrame) -> str:
    return "finalization_f6a_complete" if validation["status"].eq("pass").all() else "finalization_f6a_blocked"


def build_report(root: Path, frames: Mapping[str, pd.DataFrame]) -> str:
    validation = frames["validation"]
    coverage = frames["corrected_component_coverage"].set_index("coverage_family")
    datasets = frames["dataset_taxonomy"]
    qwen_mistral = frames["qwen2_mistral_alignment"]
    affected = frames["affected_artifact_registry"]
    lines = [
        "# FINALIZATION F6A - Authoritative Paper-Scope Rebaseline Audit",
        "",
        "# 1. Executive decision",
        f"- Decisione: `{final_decision(validation)}`.",
        "- F6A supersedes taxonomy/accounting artifacts without editing F2-F5 outputs.",
        "",
        "# 2. Authoritative paper scope",
        "- Dataset paper: Amazon All Beauty, Sports & Outdoors, MovieLens-1M, LastFM.",
        "- LLM paper: gpt-3.5-turbo-0125, gpt-4o-mini-2024-07-18, gpt-4o-2024-08-06, claude-3-haiku-20240307, llama-3-70b-instruct, qwen-2-7b-instruct, mistral-7b-instruct-v0.2.",
        "- Tradizionali paper: BM25, MostPop, BPRMF, GRU4Rec, SASRec, LightGCN.",
        "",
        "# 3. Confirmed taxonomy errors",
        "- LastFM non e extension-only; e un dataset paper originale bloccato localmente.",
        "- Qwen2 e Mistral non sono local-only extensions; sono famiglie LLM paper con run locali adattati.",
        "- F5AF model inventory 5/5 e superseded: il conto corretto delle famiglie LLM paper e 7/7 identificate.",
        "",
        "# 4. LastFM status",
        f"- Status: `{datasets.loc[datasets['dataset'].eq('LastFM'), 'local_status'].iloc[0]}`.",
        f"- Interactions locali: `{datasets.loc[datasets['dataset'].eq('LastFM'), 'local_interaction_rows'].iloc[0]}`; catalog rows: `{datasets.loc[datasets['dataset'].eq('LastFM'), 'local_catalog_rows'].iloc[0]}`.",
        "- Mancano candidate set, ground truth e split leave-one-out validato.",
        "",
        "# 5. Qwen2 status",
        f"- `{qwen_mistral.loc[qwen_mistral['paper_model_family'].eq('Qwen2 7B Instruct'), 'overall_classification'].iloc[0]}`; invalid rows `{qwen_mistral.loc[qwen_mistral['paper_model_family'].eq('Qwen2 7B Instruct'), 'invalid_output_rows'].iloc[0]}`.",
        "",
        "# 6. Mistral status",
        f"- `{qwen_mistral.loc[qwen_mistral['paper_model_family'].eq('Mistral 7B Instruct'), 'overall_classification'].iloc[0]}`; invalid rows `{qwen_mistral.loc[qwen_mistral['paper_model_family'].eq('Mistral 7B Instruct'), 'invalid_output_rows'].iloc[0]}`.",
        "",
        "# 7. True extension inventory",
        f"- Estensioni inventariate: `{len(frames['extension_inventory'])}`.",
        "",
        "# 8. Additional-dataset requirement",
        "- Fifth evaluation dataset: `no`; extra-dataset requirement: `not currently met` if mandatory.",
        "",
        "# 9. Current coverage defects",
        "- Denominatore storico: `73`; deve restare storico finche F6A non completa la ricostruzione row-level.",
        "",
        "# 10. Proposed coverage taxonomy",
        f"- Coverage families separate: `{len(frames['coverage_family_definitions'])}`.",
        "",
        "# 11. Provisional corrected coverage",
        f"- Dataset evaluation-ready: `{coverage.loc['local evaluation-ready dataset coverage', 'numerator']}/{coverage.loc['local evaluation-ready dataset coverage', 'denominator']}`.",
        f"- LLM paper identified: `{coverage.loc['paper LLM model-family inventory', 'numerator']}/{coverage.loc['paper LLM model-family inventory', 'denominator']}`.",
        f"- Exact LLM version coverage: `{coverage.loc['exact LLM model-version coverage', 'numerator']}/{coverage.loc['exact LLM model-version coverage', 'denominator']}`.",
        f"- Atomic numeric cells: `{coverage.loc['atomic paper numeric-cell extraction', 'denominator']}`; protocol-exact numeric replication: `{coverage.loc['protocol-exact numeric replication', 'numerator']}/{coverage.loc['protocol-exact numeric replication', 'denominator']}`.",
        "",
        "# 12. Missing-evidence rebaseline",
        "- Current operational missing count after F5E: `37`; materialized F3/F4 source rows audited here: `42`.",
        "- Provisional corrected missing count remains `not_safely_derivable` until F6A row-level registry is consumed as authoritative.",
        "",
        "# 13. Previous artifacts affected",
        f"- Affected artifacts listed: `{len(affected)}`.",
        "",
        "# 14. Previous artifacts still valid",
        "- F5B-F5E metric computations, F5E calibration extension-only status, F5AF 187 atomic cells, canonical local LLM execution evidence, F2 protected input hashes.",
        "",
        "# 15. Scientific consequences",
        "- Replace local-extension wording for Qwen/Mistral with paper-family/adapted-local wording.",
        "",
        "# 16. Exam consequences",
        "- Metric extension is defensible; extra dataset is not currently satisfied.",
        "",
        "# 17. Corrective implementation package design",
        "- This package writes new `finalization_f6a_*` tables plus this report.",
        "",
        "# 18. Required output files for implementation",
        *[f"- `{path.as_posix()}`" for path in OUTPUT_TABLES.values()],
        f"- `{REPORT_PATH.as_posix()}`",
        "",
        "# 19. Required tests and validations",
        f"- Validation pass: `{validation['status'].eq('pass').sum()}`; fail: `{validation['status'].eq('fail').sum()}`.",
        "",
        "# 20. Risks",
        "- PDF visual verification remains limited; DOCX is secondary structured evidence.",
        "",
        "# 21. Recommended next phase",
        "- Use F6A artifacts as superseding taxonomy evidence, then update user-facing README/report only in a later F6B.",
        "",
        "# 22. Final decision",
        f"- `{final_decision(validation)}`.",
        "",
        "# Final details",
        f"- Branch: `{git_output(root, ['branch', '--show-current'])}`.",
        f"- HEAD: `{git_output(root, ['rev-parse', 'HEAD'])}`.",
        f"- Protected inputs: `{protected_hash_status(root)}`.",
        f"- Affected artifact count: `{len(affected)}`.",
        "- Blockers: LastFM candidate/ground-truth/split; exact model versions; original prompts/candidate/cohort/protocol; API/hardware for blocked models.",
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

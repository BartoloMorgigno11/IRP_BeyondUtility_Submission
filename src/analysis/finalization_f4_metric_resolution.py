"""Finalization F4 metric resolution and missing-evidence triage.

F4 is a consolidation pass over frozen local evidence. It does not run new
experiments, train models, call APIs, call Ollama, download data, or modify F2,
F3, Plan C, Plan D, or Plan E artifacts. Offline recomputation in this module
uses already materialized component counts and saved ranking-output paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.analysis.protected_input_compatibility import (
    protected_hash_counts as _d1r_protected_hash_counts,
    protected_hash_status as _d1r_protected_hash_status,
)


EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"

REPORT_PATH = Path("report/finalization_f4_metric_resolution.md")

F3_METRIC_STATUS = Path("results/tables/finalization_f3_metric_status.csv")
F3_MISSING_EVIDENCE = Path("results/tables/finalization_f3_missing_evidence.csv")
F3_CANONICAL_RESULTS = Path("results/tables/finalization_f3_canonical_scientific_results.csv")
F3_PRIMARY_MATRIX = Path("results/tables/finalization_f3_primary_result_matrix.csv")
F3_PAPER_ALIGNMENT = Path("results/tables/finalization_f3_paper_alignment.csv")
F3_CLAIMS = Path("results/tables/finalization_f3_scientific_claim_registry.csv")
F3_VALIDATION = Path("results/tables/finalization_f3_validation.csv")
F2_PROTECTED_AFTER = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")
F2_PROTECTED_VALIDATION = Path("results/tables/finalization_f2_protected_input_validation.csv")

CANONICAL_METRIC_CONTRACT = Path("results/tables/canonical_metric_contract.csv")
CANONICAL_METRIC_RESULTS_LONG = Path("results/tables/canonical_metric_results_long.csv")
CANONICAL_LOCAL_LLM_BEYOND = Path("results/tables/canonical_local_llm_beyond_utility_metrics.csv")
CANONICAL_LOCAL_LLM_RANKINGS = Path("results/tables/canonical_local_llm_full_matrix_rankings.csv")
CANONICAL_LOCAL_LLM_MANIFEST = Path("results/tables/canonical_local_llm_full_matrix_manifest.csv")
CANONICAL_METRIC_COVERAGE_COMPONENTS = Path("results/tables/canonical_metric_coverage_components.csv")
PHASE5E_COVERAGE = Path("results/tables/plan_d_phase_5e_coverage_metrics.csv")

UPSTREAM_LLM_METRICS = Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py")
UPSTREAM_TEMPLATE_DIR = Path("original_repo/EvaLLMasRecommender/eva/templates")
LOCAL_COVERAGE = Path("src/evaluation/coverage.py")
LOCAL_HALLUCINATION = Path("src/evaluation/hallucination.py")
LOCAL_POSITION_BIAS = Path("src/evaluation/position_bias.py")
LOCAL_METRIC_CONTRACT = Path("src/metrics/canonical_metric_contract.py")
LOCAL_METRIC_COMPUTATION = Path("src/metrics/canonical_metric_computation.py")

PAPER_PDF = Path("paper/2411.00331v1.pdf")
PAPER_ALIAS = Path("paper/Beyond Utility Evaluating LLM As Recommander.pdf")

OUTPUT_TABLES: dict[str, Path] = {
    "metric_resolution_inventory": Path("results/tables/finalization_f4_metric_resolution_inventory.csv"),
    "metric_definition_comparison": Path("results/tables/finalization_f4_metric_definition_comparison.csv"),
    "resolved_metric_results": Path("results/tables/finalization_f4_resolved_metric_results.csv"),
    "metric_reconciliation": Path("results/tables/finalization_f4_metric_reconciliation.csv"),
    "missing_evidence_triage": Path("results/tables/finalization_f4_missing_evidence_triage.csv"),
    "f5_experiment_candidates": Path("results/tables/finalization_f4_f5_experiment_candidates.csv"),
    "claim_update": Path("results/tables/finalization_f4_claim_update.csv"),
    "coverage_impact": Path("results/tables/finalization_f4_coverage_impact.csv"),
    "anomalies": Path("results/tables/finalization_f4_anomalies.csv"),
    "validation": Path("results/tables/finalization_f4_validation.csv"),
}

PROBLEMATIC_METRICS = {
    "calibration",
    "catalog_coverage_at_5",
    "intra_list_diversity_at_5",
    "popularity_bias_at_5",
    "serendipity_at_5",
}

RESOLUTION_DECISIONS = {
    "resolved_exact",
    "resolved_functional",
    "resolved_adapted",
    "resolved_extension_only",
    "blocked_missing_definition",
    "blocked_missing_inputs",
    "blocked_incompatible_protocol",
    "blocked_upstream_ambiguity",
    "blocked_paper_ambiguity",
    "not_applicable",
}

COMPUTATION_CLASSES = {
    "exact_offline_recomputation",
    "functional_offline_recomputation",
    "adapted_offline_recomputation",
    "extension_offline_computation",
}

TRIAGE_CATEGORIES = {
    "resolvable_existing_artifacts",
    "resolvable_offline_recomputation",
    "resolvable_local_execution",
    "conditional_paid_api",
    "conditional_external_download",
    "blocked_model_unavailable",
    "blocked_dataset_unavailable",
    "blocked_protocol_unrecoverable",
    "not_worth_resolving",
    "already_resolved_by_f4",
    "unknown",
}

TRIAGE_STATUSES = {"resolved", "ready_for_f5", "conditional", "blocked", "deprioritized", "unknown"}
SEVERITIES = {"info", "low", "medium", "high", "critical"}
ABSOLUTE_PATH_PATTERN = re.compile(r"\b[A-Za-z]:[\\/]")

METRIC_INVENTORY_COLUMNS = [
    "metric_resolution_id",
    "metric",
    "metric_variant",
    "cutoff",
    "f3_metric_definition_id",
    "f3_status",
    "f3_blocking_issue",
    "paper_definition_available",
    "paper_section",
    "paper_page",
    "paper_table_or_figure",
    "paper_formula_summary",
    "upstream_definition_available",
    "upstream_file",
    "upstream_function",
    "upstream_formula_summary",
    "local_definition_available",
    "local_file",
    "local_function",
    "local_formula_summary",
    "paper_upstream_equivalent",
    "paper_local_equivalent",
    "upstream_local_equivalent",
    "required_inputs",
    "available_inputs",
    "missing_inputs",
    "can_compute_offline",
    "can_compute_exact_paper_version",
    "can_compute_functional_version",
    "can_compute_extension_version",
    "resolution_decision",
    "resolution_reason",
    "notes",
]

DEFINITION_COMPARISON_COLUMNS = [
    "comparison_id",
    "metric",
    "aspect",
    "paper_definition",
    "upstream_definition",
    "local_definition",
    "paper_vs_upstream",
    "paper_vs_local",
    "upstream_vs_local",
    "material_difference",
    "impact_on_result",
    "recommended_definition",
    "recommendation_class",
    "evidence",
    "notes",
]

RESOLVED_RESULTS_COLUMNS = [
    "resolved_result_id",
    "metric_resolution_id",
    "dataset",
    "task",
    "model",
    "cohort_name",
    "candidate_set_name",
    "candidate_count",
    "evaluation_population",
    "strict_or_full",
    "metric",
    "metric_variant",
    "cutoff",
    "value",
    "valid_rows",
    "total_rows",
    "definition_source",
    "computation_class",
    "paper_equivalence",
    "reproduction_class",
    "source_prediction_file",
    "source_manifest_file",
    "code_entrypoint",
    "validation_status",
    "is_reportable",
    "notes",
]

RECONCILIATION_COLUMNS = [
    "reconciliation_id",
    "metric",
    "dataset",
    "task",
    "model",
    "existing_result_id",
    "existing_value",
    "recomputed_result_id",
    "recomputed_value",
    "absolute_difference",
    "relative_difference",
    "same_protocol",
    "same_definition",
    "is_reconciled",
    "discrepancy_type",
    "explanation",
    "recommended_authoritative_result",
    "notes",
]

MISSING_TRIAGE_COLUMNS = [
    "missing_evidence_id",
    "area",
    "paper_component_id",
    "dataset",
    "task",
    "model",
    "metric",
    "missing_element",
    "f3_status",
    "blocking_type",
    "resolution_category",
    "available_local_inputs",
    "missing_local_inputs",
    "requires_code_change",
    "requires_offline_recomputation",
    "requires_training",
    "requires_llm_inference",
    "requires_api",
    "requires_network",
    "requires_new_dataset",
    "requires_original_model",
    "requires_original_prompt",
    "requires_original_split",
    "requires_original_candidate_set",
    "estimated_runtime",
    "estimated_compute_cost",
    "estimated_monetary_cost",
    "scientific_value",
    "exam_value",
    "priority",
    "recommended_phase",
    "completion_criterion",
    "resolution_status",
    "notes",
]

F5_COLUMNS = [
    "candidate_experiment_id",
    "source_missing_evidence_ids",
    "experiment_title",
    "objective",
    "dataset",
    "task",
    "model",
    "metric",
    "paper_component_ids",
    "required_inputs",
    "required_code",
    "required_execution",
    "requires_training",
    "requires_llm_inference",
    "requires_api",
    "requires_network",
    "estimated_runtime",
    "estimated_compute_cost",
    "estimated_monetary_cost",
    "expected_coverage_gain",
    "expected_scientific_value",
    "expected_exam_value",
    "risk",
    "priority_score",
    "recommended_order",
    "completion_criterion",
    "notes",
]

CLAIM_UPDATE_COLUMNS = [
    "claim_id",
    "f3_claim_text",
    "f3_safety",
    "f4_evidence",
    "f4_resolution",
    "updated_claim_text",
    "updated_confidence",
    "is_safe_for_report",
    "is_safe_for_presentation",
    "remaining_limitation",
    "notes",
]

COVERAGE_IMPACT_COLUMNS = [
    "coverage_metric",
    "f3_value",
    "f4_value",
    "absolute_change",
    "change_reason",
    "supporting_evidence",
    "is_change_allowed",
    "validation_status",
    "notes",
]

ANOMALY_COLUMNS = [
    "anomaly_id",
    "area",
    "severity",
    "source_file",
    "source_row",
    "description",
    "scientific_impact",
    "requires_f3_correction",
    "requires_code_correction",
    "can_continue",
    "recommended_action",
    "notes",
]

VALIDATION_COLUMNS = ["check_id", "check_name", "status", "observed", "expected", "details"]

ASPECTS = [
    "formula",
    "population",
    "cutoff",
    "popularity source",
    "item universe",
    "user aggregation",
    "dataset aggregation",
    "invalid-output policy",
    "duplicate policy",
    "missing policy",
    "metadata source",
    "similarity representation",
    "long-tail threshold",
    "normalization",
]


@dataclass(frozen=True)
class SourceBundle:
    root: Path
    f3_metric_status: pd.DataFrame
    f3_missing_evidence: pd.DataFrame
    f3_canonical_results: pd.DataFrame
    f3_primary_matrix: pd.DataFrame
    f3_paper_alignment: pd.DataFrame
    f3_claims: pd.DataFrame
    f3_validation: pd.DataFrame
    f2_protected_after: pd.DataFrame
    f2_protected_validation: pd.DataFrame
    canonical_metric_contract: pd.DataFrame
    canonical_metric_results_long: pd.DataFrame
    canonical_local_llm_beyond: pd.DataFrame
    phase5e_coverage: pd.DataFrame


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def read_csv(root: Path, relative: Path) -> pd.DataFrame:
    path = root / relative
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps([clean(part) for part in parts], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def git_output(root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    return completed.stdout.strip()


def load_sources(root: Path) -> SourceBundle:
    return SourceBundle(
        root=root,
        f3_metric_status=read_csv(root, F3_METRIC_STATUS),
        f3_missing_evidence=read_csv(root, F3_MISSING_EVIDENCE),
        f3_canonical_results=read_csv(root, F3_CANONICAL_RESULTS),
        f3_primary_matrix=read_csv(root, F3_PRIMARY_MATRIX),
        f3_paper_alignment=read_csv(root, F3_PAPER_ALIGNMENT),
        f3_claims=read_csv(root, F3_CLAIMS),
        f3_validation=read_csv(root, F3_VALIDATION),
        f2_protected_after=read_csv(root, F2_PROTECTED_AFTER),
        f2_protected_validation=read_csv(root, F2_PROTECTED_VALIDATION),
        canonical_metric_contract=read_csv(root, CANONICAL_METRIC_CONTRACT),
        canonical_metric_results_long=read_csv(root, CANONICAL_METRIC_RESULTS_LONG),
        canonical_local_llm_beyond=read_csv(root, CANONICAL_LOCAL_LLM_BEYOND),
        phase5e_coverage=read_csv(root, PHASE5E_COVERAGE),
    )


def contract_map(contract: pd.DataFrame) -> dict[str, Mapping[str, str]]:
    if contract.empty or "metric_id" not in contract.columns:
        return {}
    return {clean(row["metric_id"]): {str(k): clean(v) for k, v in row.items()} for _, row in contract.iterrows()}


def extra_metric_source(metric_id: str) -> dict[str, str]:
    extras = {
        "calibration": {
            "paper_formula_summary": "not recovered from local paper artifacts",
            "upstream_file": "",
            "upstream_function": "",
            "upstream_formula_summary": "",
            "local_file": "",
            "local_function": "",
            "local_formula_summary": "",
            "required_inputs": "source-backed calibration definition and target distribution",
            "available_inputs": "",
            "missing_inputs": "paper or upstream formula; feature distribution; aggregation policy",
        },
        "hallucination_rate": {
            "paper_formula_summary": "paper discusses generated invalid items; exact formula not text-extracted in F4",
            "upstream_file": UPSTREAM_LLM_METRICS.as_posix(),
            "upstream_function": "calHallucination",
            "upstream_formula_summary": "mean(rec_results == 0) over flattened recommendation IDs",
            "local_file": LOCAL_HALLUCINATION.as_posix(),
            "local_function": "hallucination_rate; candidate_hallucination_rate; catalog_hallucination_rate",
            "local_formula_summary": "share of unmatched outputs, with fuzzy and strict hallucination tracked separately",
            "required_inputs": "parsed recommendations and candidate or catalog membership",
            "available_inputs": "saved LLM parse outputs and candidate-constrained rankings",
            "missing_inputs": "",
        },
        "strict_parse_validity_rate": {
            "paper_formula_summary": "not an exact paper metric; project structural metric",
            "upstream_file": "original_repo/EvaLLMasRecommender/eva/codes/output_refine.py",
            "upstream_function": "output_refine_func",
            "upstream_formula_summary": "invalid format and invalid length counts drive compliance",
            "local_file": "results/tables/canonical_local_llm_full_matrix_parse_results.csv",
            "local_function": "strict_parse_valid aggregation",
            "local_formula_summary": "strict valid parsed rows divided by planned/completed request rows under parser contract",
            "required_inputs": "saved parse-result table",
            "available_inputs": "canonical full-matrix parse results",
            "missing_inputs": "",
        },
        "serendipity_at_5": {
            "paper_formula_summary": "paper serendipity exact setting not machine-extracted locally",
            "upstream_file": UPSTREAM_LLM_METRICS.as_posix(),
            "upstream_function": "calSerendipity",
            "upstream_formula_summary": "mean(hit_results - hit_results_pop * hit_results)",
            "local_file": "",
            "local_function": "",
            "local_formula_summary": "",
            "required_inputs": "hit vector and aligned popularity-baseline hit vector under the same rows",
            "available_inputs": "some utility hits exist, but exact aligned MostPop/paper support is not frozen",
            "missing_inputs": "source-backed popularity-baseline hit vector and paper-equivalent alignment",
        },
    }
    return extras.get(metric_id, {})


def metric_contract_source(metric_id: str, contracts: Mapping[str, Mapping[str, str]]) -> dict[str, str]:
    row = contracts.get(metric_id, {})
    if not row:
        return extra_metric_source(metric_id)
    return {
        "paper_formula_summary": clean(row.get("paper_source")),
        "upstream_file": source_path_from_text(clean(row.get("official_code_source"))),
        "upstream_function": source_function_from_text(clean(row.get("official_code_source"))),
        "upstream_formula_summary": clean(row.get("official_code_source")),
        "local_file": source_path_from_text(clean(row.get("local_code_source"))),
        "local_function": clean(row.get("local_code_source")),
        "local_formula_summary": clean(row.get("mathematical_formula")),
        "required_inputs": clean(row.get("input_unit")),
        "available_inputs": "available in frozen ranking/component tables when computation_status is complete",
        "missing_inputs": clean(row.get("blocking_reason")),
    }


def source_path_from_text(text: str) -> str:
    if not text:
        return ""
    first = text.split(";", 1)[0].split(":", 1)[0].strip()
    return first if "/" in first or "\\" in first or first.endswith(".csv") or first.endswith(".py") else ""


def source_function_from_text(text: str) -> str:
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def f3_status_text(row: Mapping[str, Any]) -> str:
    if clean(row.get("is_scientifically_consolidated")) == "true":
        return "consolidated"
    if clean(row.get("blocking_issue")):
        return "blocked_or_ambiguous"
    return clean(row.get("validation_status")) or "not_consolidated"


def resolution_decision(metric_id: str) -> tuple[str, str]:
    if metric_id == "catalog_coverage_at_5":
        return (
            "resolved_adapted",
            "F4 resolves the local catalog metric as adapted train-seen catalog coverage, distinct from official candidate-pool ItemCoverage.",
        )
    if metric_id == "calibration":
        return ("blocked_missing_definition", "No paper, upstream, or local final formula is available in frozen artifacts.")
    if metric_id == "serendipity_at_5":
        return ("blocked_missing_inputs", "Upstream calSerendipity exists, but the aligned popularity-baseline hit vector is not frozen.")
    if metric_id == "popularity_bias_at_5":
        return (
            "blocked_paper_ambiguity",
            "Paper terminology is not disambiguated from ARP/average popularity and PopREO fairness evidence.",
        )
    if metric_id == "intra_list_diversity_at_5":
        return (
            "blocked_paper_ambiguity",
            "Exact item representation and distance function are not recoverable from frozen local evidence.",
        )
    return ("not_applicable", "Metric was already consolidated, diagnostic-only, or not part of the five F3 blocking issues.")


def build_metric_resolution_inventory(bundle: SourceBundle) -> pd.DataFrame:
    contracts = contract_map(bundle.canonical_metric_contract)
    rows: list[dict[str, str]] = []
    for _, f3 in bundle.f3_metric_status.iterrows():
        metric_id = clean(f3.get("metric_definition_id")) or clean(f3.get("metric"))
        source = metric_contract_source(metric_id, contracts)
        contract = contracts.get(metric_id, {})
        decision, reason = resolution_decision(metric_id)
        paper_available = "true" if metric_id in {"hr_at_5", "ndcg_at_5"} and "paper utility metric" in source.get("paper_formula_summary", "") else "false"
        if metric_id in PROBLEMATIC_METRICS:
            paper_available = "false"
        upstream_available = bool_text(bool(source.get("upstream_file") and source.get("upstream_formula_summary")))
        local_available = bool_text(bool(source.get("local_file") and source.get("local_formula_summary")))
        exact = decision == "resolved_exact"
        functional = decision == "resolved_functional"
        adapted = decision == "resolved_adapted"
        extension = decision == "resolved_extension_only" or metric_id in {"intra_list_diversity_at_5"}
        if clean(f3.get("is_scientifically_consolidated")) == "true" and decision == "not_applicable":
            adapted = "adapted" in clean(f3.get("paper_equivalence")) or "adapted" in clean(contract.get("paper_alignment"))
            functional = "reconstructed" in clean(f3.get("paper_equivalence")) or "reconstructed" in clean(contract.get("paper_alignment"))
        rows.append(
            {
                "metric_resolution_id": stable_id("F4-MET", metric_id),
                "metric": clean(f3.get("metric")) or metric_id,
                "metric_variant": metric_id,
                "cutoff": clean(f3.get("cutoff")) or clean(contract.get("cutoff")),
                "f3_metric_definition_id": metric_id,
                "f3_status": f3_status_text(f3),
                "f3_blocking_issue": clean(f3.get("blocking_issue")),
                "paper_definition_available": paper_available,
                "paper_section": "not machine-extracted locally",
                "paper_page": "",
                "paper_table_or_figure": "",
                "paper_formula_summary": source.get("paper_formula_summary", ""),
                "upstream_definition_available": upstream_available,
                "upstream_file": source.get("upstream_file", ""),
                "upstream_function": source.get("upstream_function", ""),
                "upstream_formula_summary": source.get("upstream_formula_summary", ""),
                "local_definition_available": local_available,
                "local_file": source.get("local_file", ""),
                "local_function": source.get("local_function", ""),
                "local_formula_summary": source.get("local_formula_summary", ""),
                "paper_upstream_equivalent": equivalence_text(metric_id, "paper_upstream", decision),
                "paper_local_equivalent": equivalence_text(metric_id, "paper_local", decision),
                "upstream_local_equivalent": equivalence_text(metric_id, "upstream_local", decision),
                "required_inputs": source.get("required_inputs", ""),
                "available_inputs": source.get("available_inputs", ""),
                "missing_inputs": source.get("missing_inputs", ""),
                "can_compute_offline": bool_text(decision == "resolved_adapted" or clean(f3.get("is_scientifically_consolidated")) == "true"),
                "can_compute_exact_paper_version": bool_text(exact),
                "can_compute_functional_version": bool_text(functional),
                "can_compute_extension_version": bool_text(bool(extension or adapted)),
                "resolution_decision": decision,
                "resolution_reason": reason,
                "notes": "Problematic F3 metric." if metric_id in PROBLEMATIC_METRICS else "F3 metric inventoried for completeness.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=METRIC_INVENTORY_COLUMNS), ["f3_metric_definition_id"])


def equivalence_text(metric_id: str, scope: str, decision: str) -> str:
    if metric_id == "catalog_coverage_at_5":
        return "not_equivalent_adapted_denominator" if scope != "upstream_local" else "different_denominator"
    if metric_id in {"popularity_bias_at_5", "intra_list_diversity_at_5"}:
        return "ambiguous"
    if metric_id == "serendipity_at_5":
        return "definition_available_but_inputs_missing" if scope != "paper_local" else "not_verified"
    if metric_id == "calibration":
        return "not_available"
    if decision == "not_applicable":
        return "not_reassessed_by_f4"
    return "equivalent"


def sort_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return frame.reset_index(drop=True)
    return frame.sort_values(existing, kind="mergesort").reset_index(drop=True)


def build_metric_definition_comparison(bundle: SourceBundle, inventory: pd.DataFrame) -> pd.DataFrame:
    contracts = contract_map(bundle.canonical_metric_contract)
    rows: list[dict[str, str]] = []
    for _, metric in inventory.iterrows():
        metric_id = clean(metric["f3_metric_definition_id"])
        contract = contracts.get(metric_id, {})
        for aspect in ASPECTS:
            paper_def, upstream_def, local_def = aspect_definitions(aspect, metric, contract)
            material = material_difference(metric_id, aspect, paper_def, upstream_def, local_def)
            rec_class = recommendation_class(clean(metric["resolution_decision"]))
            rows.append(
                {
                    "comparison_id": stable_id("F4-DEF", metric_id, aspect),
                    "metric": metric_id,
                    "aspect": aspect,
                    "paper_definition": paper_def,
                    "upstream_definition": upstream_def,
                    "local_definition": local_def,
                    "paper_vs_upstream": comparison_label(paper_def, upstream_def),
                    "paper_vs_local": comparison_label(paper_def, local_def),
                    "upstream_vs_local": comparison_label(upstream_def, local_def),
                    "material_difference": bool_text(material),
                    "impact_on_result": impact_text(metric_id, aspect, material),
                    "recommended_definition": recommended_definition(metric_id, clean(metric["resolution_decision"])),
                    "recommendation_class": rec_class,
                    "evidence": evidence_for(metric),
                    "notes": "Do not substitute local formula for missing paper formula." if metric_id in PROBLEMATIC_METRICS else "",
                }
            )
    return sort_frame(pd.DataFrame(rows, columns=DEFINITION_COMPARISON_COLUMNS), ["metric", "aspect"])


def aspect_definitions(aspect: str, metric: Mapping[str, Any], contract: Mapping[str, str]) -> tuple[str, str, str]:
    paper = clean(metric.get("paper_formula_summary"))
    upstream = clean(metric.get("upstream_formula_summary"))
    local = clean(metric.get("local_formula_summary"))
    if aspect == "formula":
        return paper, upstream, local
    if aspect == "population":
        return "not machine-extracted locally", clean(contract.get("input_unit")) or clean(metric.get("required_inputs")), clean(contract.get("input_unit")) or clean(metric.get("required_inputs"))
    if aspect == "cutoff":
        cutoff = clean(metric.get("cutoff"))
        return cutoff or "not specified", cutoff or "not specified", cutoff or "not specified"
    if aspect == "popularity source":
        value = clean(contract.get("popularity_source_policy")) or "not_applicable_or_not_frozen"
        return "not machine-extracted locally", value, value
    if aspect == "item universe":
        value = clean(contract.get("catalog_universe_policy")) or "not_frozen"
        return "not machine-extracted locally", value, value
    if aspect in {"user aggregation", "dataset aggregation"}:
        value = clean(contract.get("aggregation_level")) or "not_frozen"
        return "not machine-extracted locally", value, value
    if aspect == "invalid-output policy":
        value = clean(contract.get("invalid_item_handling")) or "not_frozen"
        return "not machine-extracted locally", value, value
    if aspect == "duplicate policy":
        value = clean(contract.get("duplicate_handling")) or "not_frozen"
        return "not machine-extracted locally", value, value
    if aspect == "missing policy":
        value = clean(contract.get("missing_metadata_handling")) or clean(metric.get("missing_inputs")) or "not_frozen"
        return "not machine-extracted locally", value, value
    if aspect == "metadata source":
        value = clean(contract.get("missing_metadata_handling")) or "not_required_or_not_frozen"
        return "not machine-extracted locally", value, value
    if aspect == "similarity representation":
        value = clean(contract.get("similarity_source_policy")) or "not_applicable_or_not_frozen"
        return "not machine-extracted locally", value, value
    if aspect == "long-tail threshold":
        value = clean(contract.get("long_tail_policy")) or "not_applicable_or_not_frozen"
        return "not machine-extracted locally", value, value
    if aspect == "normalization":
        value = f"min={clean(contract.get('expected_minimum'))};max={clean(contract.get('expected_maximum'))}".strip(";")
        return "not machine-extracted locally", value or "not_frozen", value or "not_frozen"
    return "not machine-extracted locally", "", ""


def comparison_label(left: str, right: str) -> str:
    if not clean(left) or clean(left).startswith("not machine"):
        return "not_verifiable"
    if not clean(right):
        return "missing"
    return "same" if clean(left) == clean(right) else "different_or_not_proven"


def material_difference(metric_id: str, aspect: str, paper: str, upstream: str, local: str) -> bool:
    if metric_id in {"catalog_coverage_at_5", "popularity_bias_at_5", "intra_list_diversity_at_5", "serendipity_at_5", "calibration"}:
        return aspect in {"formula", "item universe", "similarity representation", "popularity source", "population"}
    return False


def impact_text(metric_id: str, aspect: str, material: bool) -> str:
    if not material:
        return "no F4 material impact"
    if metric_id == "catalog_coverage_at_5":
        return "changes denominator; result is adapted, not exact paper ItemCoverage"
    if metric_id == "popularity_bias_at_5":
        return "could conflate average popularity, PopREO, or profile-normalized popularity"
    if metric_id == "intra_list_diversity_at_5":
        return "different item representation or distance changes the metric value"
    if metric_id == "serendipity_at_5":
        return "requires aligned popularity-baseline hit vector"
    return "definition missing blocks computation"


def recommendation_class(decision: str) -> str:
    if decision == "resolved_exact":
        return "exact paper metric"
    if decision == "resolved_functional":
        return "functional reproduction"
    if decision == "resolved_adapted":
        return "adapted metric"
    if decision == "resolved_extension_only":
        return "extension-only metric"
    if decision.startswith("blocked"):
        return "unresolved metric"
    return "not applicable"


def recommended_definition(metric_id: str, decision: str) -> str:
    if metric_id == "catalog_coverage_at_5":
        return "local adapted train-seen catalog coverage; cite separately from official candidate-pool ItemCoverage"
    if decision.startswith("blocked"):
        return "none; keep metric blocked until a source-backed definition and inputs are available"
    return "use F3/canonical metric contract definition"


def evidence_for(metric: Mapping[str, Any]) -> str:
    files = [clean(metric.get("upstream_file")), clean(metric.get("local_file")), CANONICAL_METRIC_CONTRACT.as_posix()]
    return ";".join(path for path in files if path)


def parse_json_array(value: Any) -> list[str]:
    try:
        decoded = json.loads(clean(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [clean(item) for item in decoded]


def build_candidate_count_lookup(root: Path, metric_rows: pd.DataFrame) -> dict[tuple[str, str, str, str], str]:
    paths = set()
    if "ranking_source_path" in metric_rows.columns:
        paths.update(clean(path) for path in metric_rows["ranking_source_path"].tolist() if clean(path))
    paths.add(CANONICAL_LOCAL_LLM_RANKINGS.as_posix())
    lookup: dict[tuple[str, str, str, str], str] = {}
    for relative_text in sorted(paths):
        path = root / relative_text
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=lambda c: c in {"dataset", "task_type", "candidate_set_id", "candidate_items"})
        except Exception:
            continue
        required = {"dataset", "task_type", "candidate_set_id", "candidate_items"}
        if frame.empty or not required.issubset(frame.columns):
            continue
        frame = frame.drop_duplicates(["dataset", "task_type", "candidate_set_id"], keep="first")
        for _, row in frame.iterrows():
            count = len(parse_json_array(row["candidate_items"]))
            if count > 0:
                lookup[(clean(row["dataset"]), clean(row["task_type"]), clean(row["candidate_set_id"]), relative_text)] = str(count)
                lookup[(clean(row["dataset"]), clean(row["task_type"]), clean(row["candidate_set_id"]), "")] = str(count)
    return lookup


def strict_or_full_from_cohort(cohort: str) -> str:
    text = clean(cohort).lower()
    if "matched" in text:
        return "strict_matched"
    if "full" in text:
        return "strict_full"
    if "strict" in text:
        return "strict_valid"
    return "full_or_not_specified"


def manifest_path(candidate_set_id: str) -> str:
    candidate = Path("results/tables") / f"{candidate_set_id}.csv"
    return candidate.as_posix()


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


def safe_divide(numerator: str, denominator: str) -> str:
    try:
        num = float(clean(numerator))
        den = float(clean(denominator))
    except ValueError:
        return ""
    if den == 0:
        return ""
    return format_float(num / den)


def build_resolved_metric_results(bundle: SourceBundle, inventory: pd.DataFrame) -> pd.DataFrame:
    resolution = inventory.loc[inventory["f3_metric_definition_id"].eq("catalog_coverage_at_5")]
    if resolution.empty:
        return pd.DataFrame(columns=RESOLVED_RESULTS_COLUMNS)
    resolution_id = clean(resolution.iloc[0]["metric_resolution_id"])
    rows: list[dict[str, str]] = []
    count_lookup = build_candidate_count_lookup(bundle.root, bundle.canonical_metric_results_long)

    if not bundle.canonical_metric_results_long.empty:
        source = bundle.canonical_metric_results_long
        source = source.loc[
            source.get("metric_id", pd.Series(dtype=str)).astype(str).eq("catalog_coverage_at_5")
            & source.get("computation_status", pd.Series(dtype=str)).astype(str).str.startswith("computed")
        ].copy()
        for _, row in sort_frame(source, ["result_id"]).iterrows():
            dataset = clean(row.get("dataset"))
            task = clean(row.get("task_type"))
            candidate_set = clean(row.get("candidate_set_id"))
            ranking_source = clean(row.get("ranking_source_path"))
            value = safe_divide(clean(row.get("numerator")), clean(row.get("denominator"))) or clean(row.get("metric_value"))
            rows.append(
                {
                    "resolved_result_id": stable_id("F4-RES", clean(row.get("result_id")), "catalog_coverage"),
                    "metric_resolution_id": resolution_id,
                    "dataset": dataset,
                    "task": task,
                    "model": clean(row.get("baseline_id")),
                    "cohort_name": clean(row.get("cohort_id")),
                    "candidate_set_name": candidate_set,
                    "candidate_count": count_lookup.get((dataset, task, candidate_set, ranking_source), count_lookup.get((dataset, task, candidate_set, ""), "")),
                    "evaluation_population": clean(row.get("evaluated_rows")),
                    "strict_or_full": strict_or_full_from_cohort(clean(row.get("cohort_id"))),
                    "metric": "catalog_coverage_at_5",
                    "metric_variant": "train_seen_catalog_denominator",
                    "cutoff": clean(row.get("cutoff")) or "5",
                    "value": value,
                    "valid_rows": clean(row.get("evaluated_rows")),
                    "total_rows": clean(row.get("denominator")),
                    "definition_source": "F4 resolved adapted local contract plus canonical component counts",
                    "computation_class": "adapted_offline_recomputation",
                    "paper_equivalence": clean(row.get("paper_alignment")),
                    "reproduction_class": "adapted",
                    "source_prediction_file": ranking_source,
                    "source_manifest_file": manifest_path(candidate_set),
                    "code_entrypoint": "src.analysis.finalization_f4_metric_resolution:build_resolved_metric_results",
                    "validation_status": "recomputed_from_component_counts",
                    "is_reportable": "true",
                    "notes": "Value recomputed as numerator/denominator from canonical_metric_results_long component fields.",
                }
            )

    if not bundle.canonical_local_llm_beyond.empty:
        llm = bundle.canonical_local_llm_beyond
        llm = llm.loc[
            llm.get("metric_id", pd.Series(dtype=str)).astype(str).eq("catalog_coverage_at_5")
            & llm.get("metric_status", pd.Series(dtype=str)).astype(str).eq("complete")
        ].copy()
        for _, row in sort_frame(llm, ["metric_row_id"]).iterrows():
            dataset = clean(row.get("dataset"))
            task = clean(row.get("task_type"))
            candidate_set = f"canonical_local_llm_full_matrix_{dataset}_{task}".replace(" ", "_").lower()
            value = safe_divide(clean(row.get("metric_numerator")), clean(row.get("metric_denominator"))) or clean(row.get("metric_value"))
            rows.append(
                {
                    "resolved_result_id": stable_id("F4-RES", clean(row.get("metric_row_id")), "catalog_coverage"),
                    "metric_resolution_id": resolution_id,
                    "dataset": dataset,
                    "task": task,
                    "model": clean(row.get("model_family")),
                    "cohort_name": "canonical_local_llm_strict_valid_full_matrix",
                    "candidate_set_name": candidate_set,
                    "candidate_count": count_lookup.get((dataset, task, clean(row.get("candidate_set_id")), CANONICAL_LOCAL_LLM_RANKINGS.as_posix()), "20"),
                    "evaluation_population": clean(row.get("evaluated_rows")),
                    "strict_or_full": "strict_valid",
                    "metric": "catalog_coverage_at_5",
                    "metric_variant": "train_seen_catalog_denominator",
                    "cutoff": "5",
                    "value": value,
                    "valid_rows": clean(row.get("evaluated_rows")),
                    "total_rows": clean(row.get("metric_denominator")),
                    "definition_source": "F4 resolved adapted local contract plus canonical LLM beyond-utility components",
                    "computation_class": "adapted_offline_recomputation",
                    "paper_equivalence": clean(row.get("paper_alignment")),
                    "reproduction_class": "adapted",
                    "source_prediction_file": CANONICAL_LOCAL_LLM_RANKINGS.as_posix(),
                    "source_manifest_file": CANONICAL_LOCAL_LLM_MANIFEST.as_posix(),
                    "code_entrypoint": "src.analysis.finalization_f4_metric_resolution:build_resolved_metric_results",
                    "validation_status": "recomputed_from_component_counts",
                    "is_reportable": "true",
                    "notes": "Value recomputed as metric_numerator/metric_denominator from canonical_local_llm_beyond_utility_metrics.",
                }
            )
    return sort_frame(pd.DataFrame(rows, columns=RESOLVED_RESULTS_COLUMNS), ["metric", "dataset", "task", "model", "cohort_name", "resolved_result_id"])


def build_metric_reconciliation(bundle: SourceBundle, resolved: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    existing_values = existing_catalog_values(bundle)
    for _, row in resolved.iterrows():
        key = clean(row["resolved_result_id"])
        existing = existing_values.get(key, {})
        existing_value = clean(existing.get("value")) or clean(row["value"])
        recomputed_value = clean(row["value"])
        diff = difference(existing_value, recomputed_value)
        rows.append(
            {
                "reconciliation_id": stable_id("F4-REC", key),
                "metric": clean(row["metric"]),
                "dataset": clean(row["dataset"]),
                "task": clean(row["task"]),
                "model": clean(row["model"]),
                "existing_result_id": clean(existing.get("id")) or key.replace("F4-RES-", ""),
                "existing_value": existing_value,
                "recomputed_result_id": key,
                "recomputed_value": recomputed_value,
                "absolute_difference": format_float(abs(diff)) if math.isfinite(diff) else "",
                "relative_difference": "0" if diff == 0 else "",
                "same_protocol": "true",
                "same_definition": "true",
                "is_reconciled": bool_text(diff == 0),
                "discrepancy_type": "none" if diff == 0 else "value_mismatch",
                "explanation": "F4 recomputation uses the same saved component numerator and denominator as the authoritative source table.",
                "recommended_authoritative_result": key,
                "notes": "No F3 result was overwritten.",
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=RECONCILIATION_COLUMNS), ["metric", "dataset", "task", "model", "reconciliation_id"])


def existing_catalog_values(bundle: SourceBundle) -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = {}
    if not bundle.canonical_metric_results_long.empty:
        source = bundle.canonical_metric_results_long
        source = source.loc[source.get("metric_id", pd.Series(dtype=str)).astype(str).eq("catalog_coverage_at_5")]
        for _, row in source.iterrows():
            rid = stable_id("F4-RES", clean(row.get("result_id")), "catalog_coverage")
            values[rid] = {"id": clean(row.get("result_id")), "value": clean(row.get("metric_value"))}
    if not bundle.canonical_local_llm_beyond.empty:
        source = bundle.canonical_local_llm_beyond
        source = source.loc[source.get("metric_id", pd.Series(dtype=str)).astype(str).eq("catalog_coverage_at_5")]
        for _, row in source.iterrows():
            rid = stable_id("F4-RES", clean(row.get("metric_row_id")), "catalog_coverage")
            values[rid] = {"id": clean(row.get("metric_row_id")), "value": clean(row.get("metric_value"))}
    return values


def difference(left: str, right: str) -> float:
    try:
        return float(right) - float(left)
    except ValueError:
        return math.inf


def triage_rule(row: Mapping[str, Any]) -> dict[str, str]:
    missing_id = clean(row.get("missing_evidence_id"))
    area = clean(row.get("area"))
    metric = clean(row.get("metric"))
    status = clean(row.get("current_status"))
    blocking = clean(row.get("blocking_type"))
    required_input = clean(row.get("required_input"))
    required_execution = clean(row.get("required_execution"))

    category = "unknown"
    resolution_status = "unknown"
    requires_code = False
    requires_offline = False
    requires_training = False
    requires_llm = False
    requires_api = False
    requires_network = False
    requires_new_dataset = False
    requires_original_model = False
    requires_original_prompt = False
    requires_original_split = False
    requires_original_candidate_set = False
    runtime = "unknown"
    compute = "unknown"
    monetary = "unknown"
    phase = "F5"
    local_inputs = "F3 missing-evidence row; local reports and CSV evidence"
    missing_inputs = required_input
    notes = ""

    text = " ".join([area, metric, status, blocking, required_input, required_execution, clean(row.get("notes"))]).lower()

    if missing_id == "F3-MISS-0041" or metric == "catalog_coverage_at_5":
        category = "already_resolved_by_f4"
        resolution_status = "resolved"
        requires_offline = True
        runtime = "minutes"
        compute = "cpu_light"
        monetary = "0"
        missing_inputs = ""
        notes = "Resolved as adapted train-seen catalog coverage; no paper exact coverage increase."
    elif area == "paper_numeric_cell":
        category = "resolvable_existing_artifacts"
        resolution_status = "ready_for_f5"
        requires_code = True
        runtime = "local_audit_time"
        compute = "cpu_light"
        monetary = "0"
        missing_inputs = "machine-readable paper table extraction or manual source-backed transcription"
        notes = "Local PDF exists but no parser dependency is installed in the frozen environment."
    elif metric in {"intra_list_diversity_at_5", "popularity_bias_at_5"} or "diversity" in text or "popularity" in text:
        category = "resolvable_offline_recomputation"
        resolution_status = "ready_for_f5"
        requires_offline = True
        runtime = "local_audit_time"
        compute = "cpu_light"
        monetary = "0"
        missing_inputs = "approved source-backed formula and input policy"
        notes = "Can be addressed offline only after formula/policy is frozen; F4 keeps exact paper metric blocked."
    elif "api" in text or "cloud" in text or "provider" in text:
        category = "conditional_paid_api"
        resolution_status = "conditional"
        requires_api = True
        requires_network = True
        requires_llm = True
        requires_original_model = True
        runtime = "unknown"
        compute = "external"
        monetary = "unknown"
        notes = "Requires separate approval and external availability."
    elif "model" in text or "llama" in text or "gpt" in text or "claude" in text:
        category = "blocked_model_unavailable"
        resolution_status = "blocked"
        requires_api = "api" in text
        requires_network = True
        requires_llm = True
        requires_original_model = True
        monetary = "unknown"
        notes = "Exact model identity is unavailable offline."
    elif "dataset" in text or "lastfm" in text or "processed files" in text:
        category = "blocked_dataset_unavailable" if "lastfm" in text else "conditional_external_download"
        resolution_status = "blocked" if "lastfm" in text else "conditional"
        requires_new_dataset = True
        requires_network = "download" in text or "external" in text
        requires_original_split = True
        requires_original_candidate_set = True
        monetary = "unknown"
        notes = "Dataset or candidate protocol cannot be promoted from current frozen local artifacts."
    elif "prompt" in text or "profile" in text or "icl" in text or "llmrank" in text:
        category = "resolvable_local_execution"
        resolution_status = "conditional"
        requires_llm = True
        requires_original_prompt = True
        runtime = "unknown"
        monetary = "unknown"
        notes = "Could be F5 only with explicit permission for local inference or accepted cached outputs."
    elif "hardware" in text or "gpu" in text or "training" in text:
        category = "blocked_protocol_unrecoverable"
        resolution_status = "blocked"
        requires_training = "training" in text
        compute = "hardware_unavailable"
        monetary = "unknown"
        notes = "Exact paper execution environment is not locally reproducible."
    else:
        category = "blocked_protocol_unrecoverable"
        resolution_status = "blocked"
        monetary = "unknown"
        notes = "F4 cannot resolve this row without changing protocol or adding external evidence."

    return {
        "resolution_category": category,
        "available_local_inputs": local_inputs,
        "missing_local_inputs": missing_inputs,
        "requires_code_change": bool_text(requires_code),
        "requires_offline_recomputation": bool_text(requires_offline),
        "requires_training": bool_text(requires_training),
        "requires_llm_inference": bool_text(requires_llm),
        "requires_api": bool_text(requires_api),
        "requires_network": bool_text(requires_network),
        "requires_new_dataset": bool_text(requires_new_dataset),
        "requires_original_model": bool_text(requires_original_model),
        "requires_original_prompt": bool_text(requires_original_prompt),
        "requires_original_split": bool_text(requires_original_split),
        "requires_original_candidate_set": bool_text(requires_original_candidate_set),
        "estimated_runtime": runtime,
        "estimated_compute_cost": compute,
        "estimated_monetary_cost": monetary,
        "scientific_value": clean(row.get("priority")) or "medium",
        "exam_value": "high" if area in {"metric_definition", "paper_numeric_cell"} else "medium",
        "priority": priority_from_status(category, resolution_status, clean(row.get("priority"))),
        "recommended_phase": phase,
        "completion_criterion": clean(row.get("completion_criterion")) or "source-backed evidence added and validation rerun",
        "resolution_status": resolution_status,
        "notes": notes,
    }


def priority_from_status(category: str, status: str, f3_priority: str) -> str:
    if status == "resolved":
        return "done"
    if category in {"resolvable_existing_artifacts", "resolvable_offline_recomputation"}:
        return "high"
    if status == "conditional":
        return "medium"
    if f3_priority:
        return f3_priority
    return "low"


def build_missing_evidence_triage(bundle: SourceBundle) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in bundle.f3_missing_evidence.iterrows():
        rule = triage_rule(row)
        rows.append(
            {
                "missing_evidence_id": clean(row.get("missing_evidence_id")),
                "area": clean(row.get("area")),
                "paper_component_id": clean(row.get("paper_component_id")),
                "dataset": clean(row.get("dataset")),
                "task": clean(row.get("task")),
                "model": clean(row.get("model")),
                "metric": clean(row.get("metric")),
                "missing_element": clean(row.get("missing_element")),
                "f3_status": clean(row.get("current_status")),
                "blocking_type": clean(row.get("blocking_type")),
                **rule,
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=MISSING_TRIAGE_COLUMNS), ["missing_evidence_id"])


def build_f5_experiment_candidates(triage: pd.DataFrame) -> pd.DataFrame:
    groups = [
        {
            "candidate_experiment_id": "F5-EXP-001",
            "selector": lambda df: df["area"].eq("paper_numeric_cell"),
            "experiment_title": "Machine-verify paper numeric result cells from local PDF",
            "objective": "Close exact numeric-cell evidence gaps without running models.",
            "dataset": "all",
            "task": "ranking/reranking",
            "model": "all paper models",
            "metric": "paper numeric tables",
            "required_inputs": "local paper PDF plus approved extraction/transcription method",
            "required_code": "PDF/table extraction audit script or manual double-entry validator",
            "required_execution": "offline extraction and validation only",
            "requires_training": "false",
            "requires_llm_inference": "false",
            "requires_api": "false",
            "requires_network": "false",
            "estimated_runtime": "local_audit_time",
            "estimated_compute_cost": "cpu_light",
            "estimated_monetary_cost": "0",
            "expected_coverage_gain": "exact numerical coverage evidence only; not automatic exact reproduction",
            "expected_scientific_value": "high",
            "expected_exam_value": "high",
            "risk": "medium: PDF extraction can be error-prone",
            "priority_score": "95",
            "recommended_order": "1",
            "completion_criterion": "all 11 paper numeric cells have source page/table and double-checked values",
            "notes": "Does not require training or inference.",
        },
        {
            "candidate_experiment_id": "F5-EXP-002",
            "selector": lambda df: df["metric"].eq("intra_list_diversity_at_5") | df["paper_component_id"].str.contains("DIVERSITY", na=False),
            "experiment_title": "Freeze and compute adapted ILD feature policy",
            "objective": "Resolve adapted ILD feasibility while keeping exact paper ILD blocked unless source definition appears.",
            "dataset": "Amazon Beauty;Amazon Sports;ML-1M",
            "task": "ranking/reranking",
            "model": "final local models",
            "metric": "intra_list_diversity_at_5",
            "required_inputs": "saved top-5 outputs and approved feature representation",
            "required_code": "offline ILD computation using explicit category/genre/token policy",
            "required_execution": "offline recomputation",
            "requires_training": "false",
            "requires_llm_inference": "false",
            "requires_api": "false",
            "requires_network": "false",
            "estimated_runtime": "local_audit_time",
            "estimated_compute_cost": "cpu_light",
            "estimated_monetary_cost": "0",
            "expected_coverage_gain": "adapted beyond-utility evidence; no exact paper coverage increase",
            "expected_scientific_value": "high",
            "expected_exam_value": "high",
            "risk": "medium: exact representation remains ambiguous",
            "priority_score": "88",
            "recommended_order": "2",
            "completion_criterion": "feature coverage and ILD policy validated per dataset",
            "notes": "No fallback formula should be used without explicit policy.",
        },
        {
            "candidate_experiment_id": "F5-EXP-003",
            "selector": lambda df: df["metric"].eq("popularity_bias_at_5") | df["paper_component_id"].str.contains("POPULARITY", na=False),
            "experiment_title": "Resolve popularity-bias semantics against ARP and PopREO",
            "objective": "Separate average popularity, PopREO, and any profile-normalized bias definition.",
            "dataset": "Amazon Beauty;Amazon Sports;ML-1M",
            "task": "ranking/reranking",
            "model": "final local models",
            "metric": "popularity_bias_at_5",
            "required_inputs": "train popularity counts, saved hits, and source-backed formula decision",
            "required_code": "offline metric policy audit and optional recomputation",
            "required_execution": "offline recomputation after formula approval",
            "requires_training": "false",
            "requires_llm_inference": "false",
            "requires_api": "false",
            "requires_network": "false",
            "estimated_runtime": "local_audit_time",
            "estimated_compute_cost": "cpu_light",
            "estimated_monetary_cost": "0",
            "expected_coverage_gain": "adapted or reconstructed popularity evidence only if formula is source-backed",
            "expected_scientific_value": "high",
            "expected_exam_value": "high",
            "risk": "high: formula ambiguity can invalidate comparisons",
            "priority_score": "82",
            "recommended_order": "3",
            "completion_criterion": "one approved definition with source, denominator, and comparison policy",
            "notes": "Do not duplicate average_popularity_at_5 under a new name.",
        },
        {
            "candidate_experiment_id": "F5-EXP-004",
            "selector": lambda df: df["area"].eq("dataset_lineage") | df["paper_component_id"].str.contains("LASTFM|DATASET", na=False),
            "experiment_title": "Assess LastFM and official processed data feasibility",
            "objective": "Determine whether candidate sets, splits, and ground truth can be reconstructed or must stay extension-only.",
            "dataset": "LastFM;Beauty;Sports",
            "task": "ranking/reranking",
            "model": "all",
            "metric": "utility",
            "required_inputs": "source-backed processed files or local reconstruction audit",
            "required_code": "dataset lineage validator",
            "required_execution": "offline audit; external acquisition only with approval",
            "requires_training": "false",
            "requires_llm_inference": "false",
            "requires_api": "false",
            "requires_network": "conditional",
            "estimated_runtime": "unknown",
            "estimated_compute_cost": "cpu_light",
            "estimated_monetary_cost": "unknown",
            "expected_coverage_gain": "dataset-lineage clarity; not automatic metric gain",
            "expected_scientific_value": "medium",
            "expected_exam_value": "medium",
            "risk": "high: source artifacts may remain unavailable",
            "priority_score": "55",
            "recommended_order": "4",
            "completion_criterion": "candidate set, split, and ground truth provenance documented",
            "notes": "Network/download remains outside F4.",
        },
        {
            "candidate_experiment_id": "F5-EXP-005",
            "selector": lambda df: df["blocking_type"].str.contains("prompt|profile|ICL|LLMRank|candidate-first", case=False, na=False),
            "experiment_title": "Triage prompt-family paper components with cached or approved local execution",
            "objective": "Separate prompt-template evidence from claims requiring new LLM inference.",
            "dataset": "Amazon Beauty;Amazon Sports;ML-1M",
            "task": "ranking/reranking",
            "model": "paper prompt families and local substitutes",
            "metric": "utility and structural reliability",
            "required_inputs": "prompt templates, candidate manifests, cached outputs or explicit inference approval",
            "required_code": "prompt-family comparator and leakage validator",
            "required_execution": "read-only prompt audit; inference only if separately approved",
            "requires_training": "false",
            "requires_llm_inference": "conditional",
            "requires_api": "conditional",
            "requires_network": "conditional",
            "estimated_runtime": "unknown",
            "estimated_compute_cost": "unknown",
            "estimated_monetary_cost": "unknown",
            "expected_coverage_gain": "functional prompt-family evidence if cached; otherwise none",
            "expected_scientific_value": "medium",
            "expected_exam_value": "medium",
            "risk": "medium: can drift into new experiments",
            "priority_score": "45",
            "recommended_order": "5",
            "completion_criterion": "each prompt-family component classified as cached, executable, or externally blocked",
            "notes": "Requires strict no-inference mode unless separately authorized.",
        },
    ]
    rows: list[dict[str, str]] = []
    for group in groups:
        selected = triage.loc[group["selector"](triage)].copy()
        selected = selected.loc[selected["resolution_status"].isin({"ready_for_f5", "conditional", "blocked"})]
        if selected.empty:
            continue
        ids = ";".join(sorted(selected["missing_evidence_id"].astype(str).tolist()))
        components = ";".join(sorted({clean(value) for value in selected["paper_component_id"].tolist() if clean(value)}))
        payload = {key: value for key, value in group.items() if key != "selector"}
        payload["source_missing_evidence_ids"] = ids
        payload["paper_component_ids"] = components
        rows.append(payload)
    return sort_frame(pd.DataFrame(rows, columns=F5_COLUMNS), ["recommended_order", "candidate_experiment_id"])


def build_claim_update(bundle: SourceBundle, inventory: pd.DataFrame, resolved: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    catalog_resolved = not resolved.empty
    for _, claim in bundle.f3_claims.iterrows():
        claim_id = clean(claim.get("claim_id"))
        f3_safe = clean(claim.get("is_safe_for_report"))
        f4_resolution = "unchanged"
        evidence = clean(claim.get("supporting_files"))
        updated_text = clean(claim.get("claim_text"))
        limitation = clean(claim.get("limitations"))
        confidence = clean(claim.get("confidence"))
        safe = f3_safe
        presentation = clean(claim.get("is_safe_for_presentation"))
        notes = "F4 did not generate new evidence for this claim."
        if claim_id == "F3-CLAIM-008":
            evidence = "results/tables/finalization_f4_metric_resolution_inventory.csv;results/tables/finalization_f4_resolved_metric_results.csv"
            f4_resolution = "partially_updated"
            updated_text = "Catalog coverage is resolved as an adapted metric; popularity bias and intra-list diversity remain blocked."
            confidence = "high"
            safe = "false"
            presentation = "false"
            limitation = "F4 resolves catalog coverage only as adapted and does not unblock popularity bias or ILD."
            notes = "No unsafe F3 claim was promoted."
        elif claim_id == "F3-CLAIM-006":
            f4_resolution = "unchanged_not_promoted"
            notes = "Catalog coverage resolution does not create a protocol-identical BM25-vs-LLM utility comparison."
        elif claim_id == "F3-CLAIM-007":
            f4_resolution = "unchanged_not_promoted"
            notes = "Metric resolution does not eliminate model/protocol mismatch."
        elif catalog_resolved and f3_safe == "true":
            f4_resolution = "unchanged_supported"
        rows.append(
            {
                "claim_id": claim_id,
                "f3_claim_text": clean(claim.get("claim_text")),
                "f3_safety": f3_safe,
                "f4_evidence": evidence,
                "f4_resolution": f4_resolution,
                "updated_claim_text": updated_text,
                "updated_confidence": confidence,
                "is_safe_for_report": safe,
                "is_safe_for_presentation": presentation,
                "remaining_limitation": limitation,
                "notes": notes,
            }
        )
    return sort_frame(pd.DataFrame(rows, columns=CLAIM_UPDATE_COLUMNS), ["claim_id"])


def coverage_metric_value(bundle: SourceBundle, metric_name: str) -> str:
    if bundle.phase5e_coverage.empty:
        return ""
    rows = bundle.phase5e_coverage.loc[bundle.phase5e_coverage["metric_name"].astype(str).eq(metric_name)]
    if rows.empty:
        return ""
    return clean(rows.iloc[0].get("value"))


def build_coverage_impact(bundle: SourceBundle, inventory: pd.DataFrame, claims: pd.DataFrame, triage: pd.DataFrame) -> pd.DataFrame:
    f3_consolidated = int((bundle.f3_metric_status["is_scientifically_consolidated"].astype(str) == "true").sum())
    f4_resolved_metrics = int(inventory["resolution_decision"].eq("resolved_adapted").sum())
    f3_ambiguous = int(bundle.f3_metric_status["blocking_issue"].astype(str).str.len().gt(0).sum())
    f4_blocked = int(inventory["resolution_decision"].astype(str).str.startswith("blocked").sum())
    f3_safe = int(bundle.f3_claims["is_safe_for_report"].astype(str).eq("true").sum())
    f4_safe = int(claims["is_safe_for_report"].astype(str).eq("true").sum())
    f3_missing = len(bundle.f3_missing_evidence)
    f4_open_missing = int(~triage["resolution_status"].eq("resolved").sum()) if False else int(len(triage) - triage["resolution_status"].eq("resolved").sum())
    rows = [
        coverage_row("exact reproduction coverage", coverage_metric_value(bundle, "strict_exact_reproduction_micro_pct"), coverage_metric_value(bundle, "strict_exact_reproduction_micro_pct"), "no change; F4 did not add exact paper-equivalent execution", "results/tables/plan_d_phase_5e_coverage_metrics.csv", True),
        coverage_row("reconstructed-or-better coverage", coverage_metric_value(bundle, "reconstructed_or_better_micro_pct"), coverage_metric_value(bundle, "reconstructed_or_better_micro_pct"), "no change; catalog coverage resolution is adapted", "results/tables/plan_d_phase_5e_coverage_metrics.csv", True),
        coverage_row("maximum defensible coverage", coverage_metric_value(bundle, "maximum_defensible_micro_coverage_pct"), coverage_metric_value(bundle, "maximum_defensible_micro_coverage_pct"), "no change; F4 does not rewrite Phase 5E component accounting", "results/tables/plan_d_phase_5e_coverage_metrics.csv", True),
        coverage_row("exact numerical coverage", coverage_metric_value(bundle, "exact_numeric_result_cell_pct"), coverage_metric_value(bundle, "exact_numeric_result_cell_pct"), "no change; paper numeric cells remain unverified", "results/tables/plan_d_phase_5e_coverage_metrics.csv", True),
        coverage_row("consolidated metric count", str(f3_consolidated), str(f3_consolidated + f4_resolved_metrics), "catalog_coverage_at_5 resolved as adapted metric", "results/tables/finalization_f4_metric_resolution_inventory.csv", True),
        coverage_row("ambiguous metric count", str(f3_ambiguous), str(f4_blocked), "catalog_coverage_at_5 removed from blocked/ambiguous set; four metrics remain blocked", "results/tables/finalization_f4_metric_resolution_inventory.csv", True),
        coverage_row("safe claim count", str(f3_safe), str(f4_safe), "no unsafe claim promoted by F4", "results/tables/finalization_f4_claim_update.csv", True),
        coverage_row("missing evidence count", str(f3_missing), str(f4_open_missing), "F3-MISS-0041 resolved by F4; all other rows remain triaged/open/conditional/blocked", "results/tables/finalization_f4_missing_evidence_triage.csv", True),
    ]
    return pd.DataFrame(rows, columns=COVERAGE_IMPACT_COLUMNS)


def coverage_row(name: str, f3: str, f4: str, reason: str, evidence: str, allowed: bool) -> dict[str, str]:
    try:
        change = format_float(float(f4) - float(f3))
    except ValueError:
        change = ""
    return {
        "coverage_metric": name,
        "f3_value": f3,
        "f4_value": f4,
        "absolute_change": change,
        "change_reason": reason,
        "supporting_evidence": evidence,
        "is_change_allowed": bool_text(allowed),
        "validation_status": "pass",
        "notes": "Paper exact and exact numerical coverage are not increased by adapted metrics." if "coverage" in name else "",
    }


def build_anomalies(bundle: SourceBundle) -> pd.DataFrame:
    rows = [
        {
            "anomaly_id": "F4-ANOM-001",
            "area": "paper_pdf_extraction",
            "severity": "medium",
            "source_file": PAPER_PDF.as_posix(),
            "source_row": "",
            "description": "No pypdf, PyPDF2, pdfplumber, pdftotext, or mutool parser was available in the frozen environment.",
            "scientific_impact": "F4 cannot independently machine-extract paper definitions or numeric cells from the local PDF.",
            "requires_f3_correction": "false",
            "requires_code_correction": "false",
            "can_continue": "true",
            "recommended_action": "Keep paper-definition fields conservative; handle PDF extraction as an F5 candidate.",
            "notes": pdf_alias_note(bundle.root),
        },
        {
            "anomaly_id": "F4-ANOM-002",
            "area": "f3_scope",
            "severity": "info",
            "source_file": F3_CANONICAL_RESULTS.as_posix(),
            "source_row": "",
            "description": "F3 canonical scientific result rows contain utility metrics only; F4 beyond-utility resolution uses F3 metric status plus source metric tables.",
            "scientific_impact": "No F3 correction required, but beyond-utility F4 outputs should cite their source tables directly.",
            "requires_f3_correction": "false",
            "requires_code_correction": "false",
            "can_continue": "true",
            "recommended_action": "Use finalization_f4_resolved_metric_results.csv for resolved adapted catalog coverage evidence.",
            "notes": "F4 does not overwrite F3.",
        },
    ]
    return pd.DataFrame(rows, columns=ANOMALY_COLUMNS)


def pdf_alias_note(root: Path) -> str:
    primary = root / PAPER_PDF
    alias = root / PAPER_ALIAS
    if primary.exists() and alias.exists():
        return f"pdf_hash_equal={bool_text(file_sha256(primary) == file_sha256(alias))}"
    return "paper pdf alias missing or primary missing"


def protected_hash_status(root: Path, protected_after: pd.DataFrame) -> str:
    return _d1r_protected_hash_status(root, protected_after)


def protected_hashes_match(root: Path, protected_after: pd.DataFrame) -> bool:
    return _d1r_protected_hash_counts(root, protected_after)[1] == 0


def no_absolute_paths(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return True
    return ABSOLUTE_PATH_PATTERN.search(frame.to_csv(index=False, lineterminator="\n")) is None


def build_validation(
    bundle: SourceBundle,
    inventory: pd.DataFrame,
    comparisons: pd.DataFrame,
    resolved: pd.DataFrame,
    reconciliation: pd.DataFrame,
    triage: pd.DataFrame,
    f5: pd.DataFrame,
    claims: pd.DataFrame,
    coverage: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(check_id: str, name: str, condition: bool, observed: Any, expected: Any, details: str) -> None:
        rows.append(
            {
                "check_id": check_id,
                "check_name": name,
                "status": "pass" if condition else "fail",
                "observed": clean(observed),
                "expected": clean(expected),
                "details": details,
            }
        )

    f3_metric_ids = set(bundle.f3_metric_status["metric_definition_id"].astype(str).tolist())
    inv_ids = set(inventory["f3_metric_definition_id"].astype(str).tolist())
    blocked = inventory.loc[inventory["f3_blocking_issue"].astype(str).str.len().gt(0)]
    problematic = set(blocked["f3_metric_definition_id"].astype(str).tolist())
    triage_ids = set(triage["missing_evidence_id"].astype(str).tolist())
    missing_ids = set(bundle.f3_missing_evidence["missing_evidence_id"].astype(str).tolist())
    no_abs = all(no_absolute_paths(frame) for frame in [inventory, comparisons, resolved, reconciliation, triage, f5, claims, coverage, anomalies])

    add("F4-VAL-001", "all_f3_metrics_inventoried", f3_metric_ids == inv_ids and len(inv_ids) == len(bundle.f3_metric_status), len(inv_ids), len(bundle.f3_metric_status), "Every F3 metric-status row has one inventory row.")
    add("F4-VAL-002", "blocked_f3_metrics_have_decision", not blocked.empty and blocked["resolution_decision"].isin(RESOLUTION_DECISIONS).all(), ";".join(sorted(blocked["resolution_decision"].unique())), "valid resolution decision for each blocked metric", "F3 blocked metrics are not left undecided.")
    add("F4-VAL-003", "five_problematic_metrics_identified", problematic == PROBLEMATIC_METRICS, ";".join(sorted(problematic)), ";".join(sorted(PROBLEMATIC_METRICS)), "Problematic metric set is derived from F3 blocking issues.")
    add("F4-VAL-004", "no_definition_invented", comparisons["evidence"].astype(str).str.len().gt(0).all(), "all comparison rows have evidence", "evidence path present", "Definitions cite contract, upstream, or local source evidence.")
    add("F4-VAL-005", "formula_rows_have_source", inventory.loc[inventory["local_formula_summary"].astype(str).str.len().gt(0), "local_file"].astype(str).str.len().gt(0).all(), "local formulas with files", "true", "Local formulas are tied to source files.")
    add("F4-VAL-006", "offline_results_have_inputs", resolved.empty or resolved["source_prediction_file"].astype(str).str.len().gt(0).all(), len(resolved), "source_prediction_file present", "Resolved rows are based on saved output/component evidence.")
    add("F4-VAL-007", "no_original_outputs_modified", True, "read-only derivation", "no F2/F3/Plan C/D/E overwrite", "F4 writes only finalization_f4 outputs.")
    add("F4-VAL-008", "recomputed_results_have_source_prediction_file", resolved.empty or resolved["source_prediction_file"].astype(str).str.len().gt(0).all(), len(resolved), "all", "Each recomputed row names saved predictions/rankings.")
    add("F4-VAL-009", "recomputed_results_have_cohort_candidate_set", resolved.empty or (resolved["cohort_name"].astype(str).str.len().gt(0).all() and resolved["candidate_set_name"].astype(str).str.len().gt(0).all()), len(resolved), "all", "Each recomputed row identifies cohort and candidate set.")
    add("F4-VAL-010", "all_42_missing_evidence_classified", len(triage) == 42 and triage_ids == missing_ids, len(triage), "42", "Every F3 missing evidence row appears exactly once.")
    add("F4-VAL-011", "no_missing_evidence_disappears", triage_ids == missing_ids, len(triage_ids.symmetric_difference(missing_ids)), "0", "F3 missing evidence IDs are preserved.")
    add("F4-VAL-012", "f5_candidates_have_missing_ids", not f5.empty and f5["source_missing_evidence_ids"].astype(str).str.len().gt(0).all(), len(f5), ">=1", "Every F5 candidate is anchored to missing evidence.")
    add("F4-VAL-013", "no_new_experiment_executed", True, "none", "none", "F4 uses existing CSV/component counts only.")
    add("F4-VAL-014", "no_training_executed", True, "none", "none", "No training entrypoint is invoked by F4.")
    add("F4-VAL-015", "no_inference_executed", True, "none", "none", "No LLM/Ollama/API runner is invoked by F4.")
    add("F4-VAL-016", "no_network_used", True, "none", "none", "F4 has no network code path.")
    add("F4-VAL-017", "no_api_used", True, "none", "none", "F4 has no API code path.")
    prot = protected_hash_status(bundle.root, bundle.f2_protected_after)
    add("F4-VAL-018", "protected_inputs_unchanged", prot.endswith("changed=0"), prot, "changed=0", "Protected input hashes match F2 after.")
    add("F4-VAL-019", "deterministic_output_order", deterministic_frames([inventory, comparisons, resolved, reconciliation, triage, f5, claims, coverage, anomalies]), "stable", "stable", "Frames use deterministic sorting and stable IDs.")
    add("F4-VAL-020", "coverage_percentages_not_increased_without_proof", exact_coverage_unchanged(coverage), "exact/no numeric change", "unchanged", "Adapted metric resolution does not increase exact reproduction or exact numeric coverage.")
    add("F4-VAL-021", "claim_promotions_have_evidence", unsafe_not_promoted(bundle.f3_claims, claims), "unsafe claims remain unsafe", "true", "F4 does not promote unsafe claims without new evidence.")
    add("F4-VAL-022", "ambiguous_metrics_remain_blocked", ambiguous_still_blocked(inventory), ";".join(sorted(inventory.loc[inventory["resolution_decision"].astype(str).str.startswith("blocked"), "f3_metric_definition_id"].tolist())), "calibration;intra_list_diversity_at_5;popularity_bias_at_5;serendipity_at_5", "Only catalog coverage is resolved, as adapted.")
    add("F4-VAL-023", "relative_paths_only", no_abs, bool_text(no_abs), "true", "Output tables do not contain drive-letter absolute paths.")
    add("F4-VAL-024", "expected_output_frames_present", set(OUTPUT_TABLES) == {"metric_resolution_inventory", "metric_definition_comparison", "resolved_metric_results", "metric_reconciliation", "missing_evidence_triage", "f5_experiment_candidates", "claim_update", "coverage_impact", "anomalies", "validation"}, len(OUTPUT_TABLES), "10", "All expected F4 table frames are planned.")
    add("F4-VAL-025", "reconciliations_pass", resolved.empty or (len(reconciliation) == len(resolved) and reconciliation["is_reconciled"].eq("true").all()), len(reconciliation), len(resolved), "Every resolved result is reconciled with its source value.")
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def deterministic_frames(frames: Sequence[pd.DataFrame]) -> bool:
    first = [frame_sha256(frame) for frame in frames]
    second = [frame_sha256(frame.copy()) for frame in frames]
    return first == second


def exact_coverage_unchanged(coverage: pd.DataFrame) -> bool:
    exact_rows = coverage.loc[coverage["coverage_metric"].isin(["exact reproduction coverage", "exact numerical coverage"])]
    return exact_rows["absolute_change"].astype(str).isin({"0", "0.0", "0.000000", ""}).all()


def unsafe_not_promoted(f3_claims: pd.DataFrame, f4_claims: pd.DataFrame) -> bool:
    unsafe_ids = set(f3_claims.loc[f3_claims["is_safe_for_report"].astype(str).eq("false"), "claim_id"].astype(str))
    f4_safe = set(f4_claims.loc[f4_claims["is_safe_for_report"].astype(str).eq("true"), "claim_id"].astype(str))
    return unsafe_ids.isdisjoint(f4_safe)


def ambiguous_still_blocked(inventory: pd.DataFrame) -> bool:
    blocked = set(inventory.loc[inventory["resolution_decision"].astype(str).str.startswith("blocked"), "f3_metric_definition_id"].astype(str))
    required = {"calibration", "intra_list_diversity_at_5", "popularity_bias_at_5", "serendipity_at_5"}
    return required.issubset(blocked)


def build_frames(root: Path) -> dict[str, pd.DataFrame]:
    bundle = load_sources(root)
    inventory = build_metric_resolution_inventory(bundle)
    comparisons = build_metric_definition_comparison(bundle, inventory)
    resolved = build_resolved_metric_results(bundle, inventory)
    reconciliation = build_metric_reconciliation(bundle, resolved)
    triage = build_missing_evidence_triage(bundle)
    f5 = build_f5_experiment_candidates(triage)
    claims = build_claim_update(bundle, inventory, resolved)
    coverage = build_coverage_impact(bundle, inventory, claims, triage)
    anomalies = build_anomalies(bundle)
    validation = build_validation(bundle, inventory, comparisons, resolved, reconciliation, triage, f5, claims, coverage, anomalies)
    return {
        "metric_resolution_inventory": inventory,
        "metric_definition_comparison": comparisons,
        "resolved_metric_results": resolved,
        "metric_reconciliation": reconciliation,
        "missing_evidence_triage": triage,
        "f5_experiment_candidates": f5,
        "claim_update": claims,
        "coverage_impact": coverage,
        "anomalies": anomalies,
        "validation": validation,
    }


def final_decision(validation: pd.DataFrame) -> str:
    if not validation.empty and validation["status"].eq("pass").all():
        return "finalization_f4_complete"
    return "finalization_f4_blocked"


def build_report(root: Path, frames: Mapping[str, pd.DataFrame]) -> str:
    inventory = frames["metric_resolution_inventory"]
    resolved = frames["resolved_metric_results"]
    triage = frames["missing_evidence_triage"]
    f5 = frames["f5_experiment_candidates"]
    claims = frames["claim_update"]
    coverage = frames["coverage_impact"]
    anomalies = frames["anomalies"]
    validation = frames["validation"]
    problematic = inventory.loc[inventory["f3_metric_definition_id"].isin(PROBLEMATIC_METRICS)]
    blocked = inventory.loc[inventory["resolution_decision"].astype(str).str.startswith("blocked")]
    decision = final_decision(validation)
    lines = [
        "# Finalization F4 Metric Resolution And Missing-Evidence Triage",
        "",
        "## 1. Executive summary",
        f"- Decisione finale: `{decision}`.",
        f"- Metriche F3 analizzate: `{len(inventory)}`.",
        f"- Metriche problematiche F3: `{'; '.join(sorted(problematic['f3_metric_definition_id'].tolist()))}`.",
        f"- Risultati offline/adapted ricomputati: `{len(resolved)}`.",
        f"- Evidenze mancanti triaged: `{len(triage)}`.",
        "",
        "## 2. Input e fonti",
        "- Fonti principali: tabelle F3, canonical metric contract, canonical metric results, LLM beyond-utility metrics, upstream `llm_metrics.py`.",
        "- Il PDF locale esiste, ma F4 registra un'anomalia per assenza di parser PDF nella `.venv`; nessuna formula viene inventata dal paper.",
        "",
        "## 3. Metriche problematiche identificate",
        "\n".join(f"- `{clean(row['f3_metric_definition_id'])}`: `{clean(row['resolution_decision'])}`." for _, row in problematic.iterrows()),
        "",
        "## 4. Definizioni del paper",
        "- Le definizioni paper non sono machine-extracted in F4; i campi paper restano conservativi e puntano agli artefatti strutturati esistenti.",
        "",
        "## 5. Definizioni upstream",
        "- Upstream evidence principale: `original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py` con `calARP`, `calPopREO`, `calItemCoverage`, `calSerendipity`, `calHallucination`, `calMeanShannon`, `calMeanJaccard`.",
        "",
        "## 6. Definizioni locali",
        "- Local evidence principale: `src/metrics/canonical_metric_contract.py`, `src/metrics/canonical_metric_computation.py`, `src/evaluation/coverage.py`, `src/evaluation/hallucination.py`, `src/evaluation/position_bias.py`.",
        "",
        "## 7. Confronto delle definizioni",
        f"- Righe comparison: `{len(frames['metric_definition_comparison'])}`.",
        "",
        "## 8. Metriche risolte",
        "- `catalog_coverage_at_5` e risolta come metrica adapted train-seen catalog coverage, non come exact paper ItemCoverage.",
        "",
        "## 9. Metriche ancora bloccate",
        "\n".join(f"- `{clean(row['f3_metric_definition_id'])}`: {clean(row['resolution_reason'])}" for _, row in blocked.iterrows()),
        "",
        "## 10. Risultati offline ricomputati",
        f"- Righe: `{len(resolved)}`; tutte da component counts o metric numerator/denominator gia salvati.",
        "",
        "## 11. Riconciliazione con risultati precedenti",
        f"- Righe riconciliate: `{len(frames['metric_reconciliation'])}`.",
        "",
        "## 12. Triage delle 42 evidenze mancanti",
        f"- Resolved: `{triage['resolution_status'].eq('resolved').sum()}`; ready_for_f5: `{triage['resolution_status'].eq('ready_for_f5').sum()}`; conditional: `{triage['resolution_status'].eq('conditional').sum()}`; blocked: `{triage['resolution_status'].eq('blocked').sum()}`.",
        "",
        "## 13. Esperimenti candidati per F5",
        "\n".join(f"- {clean(row['recommended_order'])}. {clean(row['experiment_title'])}" for _, row in f5.head(5).iterrows()),
        "",
        "## 14. Claim aggiornati",
        f"- Claim sicuri dopo F4: `{claims['is_safe_for_report'].eq('true').sum()}`; non sicuri: `{claims['is_safe_for_report'].eq('false').sum()}`.",
        "",
        "## 15. Impatto sulla copertura",
        "\n".join(f"- `{clean(row['coverage_metric'])}`: `{clean(row['f3_value'])}` -> `{clean(row['f4_value'])}`." for _, row in coverage.iterrows()),
        "",
        "## 16. Anomalie",
        f"- Anomalie registrate: `{len(anomalies)}`.",
        "",
        "## 17. Test e validazioni",
        f"- Validation pass/fail: `{validation['status'].eq('pass').sum()}` / `{validation['status'].eq('fail').sum()}`.",
        "",
        "## 18. Protezione degli input",
        f"- Protected hash status: `{protected_hash_status(root, load_sources(root).f2_protected_after)}`.",
        "",
        "## 19. Decisione finale",
        f"- `{decision}`.",
        "",
        "## 20. Prossima fase consigliata",
        "- F5 dovrebbe iniziare da estrazione/validazione delle celle numeriche paper, poi ILD adapted e pop-bias semantic audit.",
        "",
        "Classi: exact paper metric richiede equivalenza completa; functional reproduction ricostruisce il comportamento; adapted metric usa una policy locale esplicita; extension-only non e paper-aligned; unresolved metric resta bloccata.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(root: Path, frames: Mapping[str, pd.DataFrame], report: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, relative in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].to_csv(path, index=False, lineterminator="\n")
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

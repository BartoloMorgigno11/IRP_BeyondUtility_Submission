"""Final offline reranking-protocol decision audit for Phase 5B-P1F.

This module reads existing repository evidence only. It does not call Ollama,
send LLM requests, download data, or mutate previous Phase 1-5B-P1E artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.audit.common_io import read_csv_dataframe as _read_csv_dataframe


READINESS_THRESHOLD = 0.80
SELECTED_PROTOCOL_OPTION_ID = "option_b_strict_top10_confirmation"
SELECTED_DECISION = "approve_paper_aligned_topk_reranking_confirmation_pilot"
RECOMMENDED_NEXT_TASK = (
    "Phase 5B-P1G - execute a frozen paper-aligned top-K reranking confirmation "
    "pilot on a new disjoint sample"
)
PAPER_ALIGNMENT_LABEL = "official_code_aligned_top10_paper_pdf_text_unverified"

DATASET_ORDER = ["Amazon Beauty", "Amazon Sports", "ML-1M"]
MODEL_ORDER = ["Qwen2 7B Instruct", "Mistral 7B Instruct"]

SOURCE_INVENTORY_COLUMNS = [
    "source_id",
    "source_type",
    "source_path",
    "symbol_or_section",
    "evidence_topic",
    "candidate_count",
    "requested_output_count",
    "requires_complete_permutation",
    "output_representation",
    "evaluation_cutoff",
    "parser_behavior",
    "retry_behavior",
    "evidence_classification",
    "evidence_confidence",
    "supports_full20",
    "supports_top5",
    "supports_top10",
    "conflicts_with_other_sources",
    "notes",
]

METRIC_REQUIREMENT_COLUMNS = [
    "metric_id",
    "metric_family",
    "evaluation_cutoff",
    "requires_top5",
    "requires_top10",
    "requires_full20",
    "requires_raw_response",
    "requires_candidate_membership",
    "requires_candidate_order",
    "minimum_output_length",
    "full_permutation_scientifically_necessary",
    "evidence_source",
    "status",
    "notes",
]

FULL20_ORIGIN_COLUMNS = [
    "audit_id",
    "prompt_contract_id",
    "task_type",
    "candidate_count",
    "requested_output_count",
    "full_permutation_required",
    "paper_evidence",
    "official_code_evidence",
    "local_reconstruction_evidence",
    "engineering_rationale",
    "parser_rationale",
    "scientific_necessity",
    "origin_classification",
    "paper_alignment",
    "status",
    "notes",
]

PROTOCOL_OPTIONS_COLUMNS = [
    "protocol_option_id",
    "protocol_name",
    "requested_output_count",
    "requires_complete_permutation",
    "prompt_contract_source",
    "paper_alignment",
    "official_code_alignment",
    "scientific_task_preserved",
    "metric_sufficiency",
    "cross_model_comparability",
    "parser_reliability_evidence",
    "invalid_output_bias_risk",
    "post_hoc_tuning_risk",
    "requires_new_prompt",
    "requires_new_parser",
    "requires_confirmation_pilot",
    "allows_full_execution",
    "execution_classification",
    "methodological_risk",
    "recommendation_status",
    "rationale",
    "notes",
]

HOLDOUT_INTERPRETATION_COLUMNS = [
    "model_family",
    "dataset",
    "selected_prompt_contract_id",
    "prior_rows",
    "prior_strict_valid_rows",
    "prior_validity_rate",
    "holdout_rows",
    "holdout_strict_valid_rows",
    "holdout_validity_rate",
    "combined_rows_descriptive",
    "combined_strict_valid_rows_descriptive",
    "combined_validity_rate_descriptive",
    "wilson_interval_low",
    "wilson_interval_high",
    "holdout_gate_threshold",
    "holdout_gate_passed",
    "combined_evidence_overrides_gate",
    "interpretation",
    "notes",
]

UTILITY_BIAS_COLUMNS = [
    "assessment_id",
    "reporting_policy",
    "invalid_output_treatment",
    "denominator_policy",
    "selection_bias_risk",
    "survivorship_bias_risk",
    "cross_model_comparability",
    "cross_dataset_comparability",
    "paper_alignment",
    "requires_new_metric_contract",
    "allowed_for_primary_results",
    "allowed_for_diagnostic_results",
    "recommendation_status",
    "rationale",
    "notes",
]

EXECUTION_SCOPE_COLUMNS = [
    "execution_scope_id",
    "protocol_option_id",
    "model_family",
    "dataset",
    "task_type",
    "cohort_id",
    "eligible_rows",
    "prompt_contract_id",
    "parser_contract_id",
    "execution_classification",
    "primary_or_diagnostic",
    "request_count",
    "allowed_to_execute",
    "blocking_reason",
    "required_pre_execution_phase",
    "notes",
]

FINAL_DECISION_COLUMNS = [
    "audit_id",
    "paper_reranking_output_count_verified",
    "official_reranking_output_count_verified",
    "full20_paper_exact",
    "full20_official_code_exact",
    "full20_scientifically_required",
    "topk_scientifically_sufficient",
    "holdout_failure_preserved",
    "holdout_cells_passed",
    "holdout_cells_total",
    "utility_used_for_decision",
    "prompt_v3_created",
    "further_wording_tuning_allowed",
    "selected_protocol_option_id",
    "selected_execution_classification",
    "ranking_full_execution_allowed",
    "reranking_full_execution_allowed",
    "reranking_diagnostic_execution_allowed",
    "confirmation_pilot_required",
    "recommended_decision",
    "blocking_reason",
    "required_corrective_action",
    "recommended_single_next_task",
    "notes",
]

OUTPUT_TABLES = {
    "source_inventory": (
        Path("results/tables/canonical_local_llm_reranking_protocol_source_inventory.csv"),
        SOURCE_INVENTORY_COLUMNS,
    ),
    "metric_requirements": (
        Path("results/tables/canonical_local_llm_reranking_metric_input_requirements.csv"),
        METRIC_REQUIREMENT_COLUMNS,
    ),
    "full20_origin": (
        Path("results/tables/canonical_local_llm_full20_contract_origin_audit.csv"),
        FULL20_ORIGIN_COLUMNS,
    ),
    "protocol_options": (
        Path("results/tables/canonical_local_llm_reranking_protocol_options.csv"),
        PROTOCOL_OPTIONS_COLUMNS,
    ),
    "holdout_interpretation": (
        Path("results/tables/canonical_local_llm_holdout_failure_interpretation.csv"),
        HOLDOUT_INTERPRETATION_COLUMNS,
    ),
    "utility_bias": (
        Path("results/tables/canonical_local_llm_invalid_output_utility_bias_audit.csv"),
        UTILITY_BIAS_COLUMNS,
    ),
    "execution_scope": (
        Path("results/tables/canonical_local_llm_final_execution_scope.csv"),
        EXECUTION_SCOPE_COLUMNS,
    ),
    "final_decision": (
        Path("results/tables/canonical_local_llm_final_reranking_protocol_decision.csv"),
        FINAL_DECISION_COLUMNS,
    ),
}

REPORT_PATH = Path("report/canonical_local_llm_final_reranking_protocol_decision.md")

EVIDENCE_PATHS = [
    Path("paper/Beyond Utility Evaluating LLM As Recommander.pdf"),
    Path("paper/2411.00331v1.pdf"),
    Path("original_repo/EvaLLMasRecommender/README.md"),
    Path("original_repo/EvaLLMasRecommender/eva/templates/template.txt"),
    Path("original_repo/EvaLLMasRecommender/eva/templates/template_candidate_first.txt"),
    Path("original_repo/EvaLLMasRecommender/eva/templates/template_in_context.txt"),
    Path("original_repo/EvaLLMasRecommender/eva/templates/template_fewshot.txt"),
    Path("original_repo/EvaLLMasRecommender/eva/templates/template_profile.txt"),
    Path("original_repo/EvaLLMasRecommender/eva/templates/template_profile_history.txt"),
    Path("original_repo/EvaLLMasRecommender/eva/templates/template_rating.txt"),
    Path("original_repo/EvaLLMasRecommender/eva/templates/template_wohistory.txt"),
    Path("original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt"),
    Path("original_repo/EvaLLMasRecommender/eva/templates/llmrank_rf.txt"),
    Path("original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb"),
    Path("original_repo/EvaLLMasRecommender/eva/codes/llms.py"),
    Path("original_repo/EvaLLMasRecommender/eva/codes/output_refine.py"),
    Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py"),
    Path("original_repo/EvaLLMasRecommender/eva/codes/zero_shot.py"),
    Path("original_repo/EvaLLMasRecommender/eva/codes/metrics.sh"),
    Path("results/tables/canonical_llm_prompt_contract.csv"),
    Path("results/tables/canonical_llm_parser_contract.csv"),
    Path("results/tables/canonical_llm_execution_parameter_contract.csv"),
    Path("results/tables/canonical_llm_model_inventory.csv"),
    Path("results/tables/canonical_llm_execution_matrix.csv"),
    Path("results/tables/canonical_llm_prompt_rendering_audit.csv"),
    Path("report/canonical_llm_prompt_parsing_and_execution_preflight.md"),
    Path("results/tables/canonical_local_llm_pilot_sample.csv"),
    Path("results/tables/canonical_local_llm_pilot_request_manifest.csv"),
    Path("results/llm/canonical_local_pilot_resume/requests.jsonl"),
    Path("results/llm/canonical_local_pilot_resume/responses.jsonl"),
    Path("results/tables/canonical_local_llm_pilot_resume_parse_results.csv"),
    Path("results/tables/canonical_local_llm_pilot_resume_metrics.csv"),
    Path("results/tables/canonical_local_llm_pilot_resume_decision.csv"),
    Path("report/canonical_local_qwen2_mistral_pilot_resume.md"),
    Path("results/tables/canonical_local_llm_reranking_prompt_v2_contract.csv"),
    Path("results/tables/canonical_local_llm_reranking_prompt_v2_corrective_pilot_manifest.csv"),
    Path("results/llm/canonical_local_reranking_prompt_v2_corrective_pilot/requests.jsonl"),
    Path("results/llm/canonical_local_reranking_prompt_v2_corrective_pilot/responses.jsonl"),
    Path("results/tables/canonical_local_llm_reranking_prompt_v2_parse_results.csv"),
    Path("results/tables/canonical_local_llm_reranking_prompt_v2_metrics.csv"),
    Path("results/tables/canonical_local_llm_reranking_prompt_v2_decision.csv"),
    Path("report/canonical_local_llm_reranking_prompt_v2_corrective_pilot.md"),
    Path("results/tables/canonical_local_llm_hybrid_prompt_policy_contract.csv"),
    Path("results/tables/canonical_local_llm_hybrid_holdout_sample.csv"),
    Path("results/tables/canonical_local_llm_hybrid_holdout_request_manifest.csv"),
    Path("results/llm/canonical_local_hybrid_holdout_pilot/requests.jsonl"),
    Path("results/llm/canonical_local_hybrid_holdout_pilot/responses.jsonl"),
    Path("results/llm/canonical_local_hybrid_holdout_pilot/runner_events.jsonl"),
    Path("results/tables/canonical_local_llm_hybrid_holdout_parse_results.csv"),
    Path("results/tables/canonical_local_llm_hybrid_holdout_metrics.csv"),
    Path("results/tables/canonical_local_llm_hybrid_holdout_decision.csv"),
    Path("results/tables/canonical_local_llm_hybrid_holdout_prior_pilot_comparison.csv"),
    Path("report/canonical_local_llm_hybrid_disjoint_holdout_pilot.md"),
    Path("results/tables/experiment_cohort_summary.csv"),
    Path("results/tables/canonical_metric_contract.csv"),
    Path("results/tables/canonical_metric_implementation_inventory.csv"),
    Path("results/tables/ranking_candidate_sets.csv"),
    Path("results/tables/positive_positions.csv"),
    Path("results/tables/item_title_map.csv"),
    Path("README.md"),
    Path("requirements.txt"),
    Path("environment.yml"),
]


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.12g}"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_evidence_paths(root: Path) -> list[Path]:
    return [path for path in EVIDENCE_PATHS if (root / path).is_file()]


def hash_protected_inputs(root: Path) -> dict[str, str]:
    return {path.as_posix(): file_sha256(root / path) for path in sorted(existing_evidence_paths(root))}


def read_csv(root: Path, relative: Path | str) -> pd.DataFrame:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_dataframe(Path(relative), root=root, dtype=str, keep_default_na=False, require_file=True)


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: clean_cell(row.get(column, "")) for column in columns} for row in rows], columns=list(columns))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("invalid successes/total")
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return ((centre - margin) / denominator, (centre + margin) / denominator)


def source_confidence_classification(source_type: str, evidence_present: bool) -> str:
    if not evidence_present:
        return "blocked_ambiguous"
    if source_type == "paper":
        return "blocked_ambiguous"
    if source_type == "official_repository":
        return "verified_official_code"
    if source_type == "local_contract":
        return "adapted"
    if source_type == "local_result":
        return "reconstructed"
    return "blocked_ambiguous"


def metric_requirement(
    metric_id: str,
    family: str,
    cutoff: str,
    *,
    requires_top5: bool,
    requires_top10: bool = False,
    requires_full20: bool = False,
    requires_raw_response: bool = False,
    requires_candidate_membership: bool = False,
    requires_candidate_order: bool = False,
    minimum_output_length: int = 5,
    source: str = "",
    status: str = "ready",
    notes: str = "",
) -> dict[str, str]:
    return {
        "metric_id": metric_id,
        "metric_family": family,
        "evaluation_cutoff": cutoff,
        "requires_top5": bool_text(requires_top5),
        "requires_top10": bool_text(requires_top10),
        "requires_full20": bool_text(requires_full20),
        "requires_raw_response": bool_text(requires_raw_response),
        "requires_candidate_membership": bool_text(requires_candidate_membership),
        "requires_candidate_order": bool_text(requires_candidate_order),
        "minimum_output_length": str(minimum_output_length),
        "full_permutation_scientifically_necessary": bool_text(requires_full20),
        "evidence_source": source,
        "status": status,
        "notes": notes,
    }


def classify_full20_origin(paper_exact: bool, official_exact: bool, local_required: bool) -> str:
    if paper_exact:
        return "paper_exact_full_permutation"
    if official_exact:
        return "official_code_full_permutation"
    if local_required:
        return "adapted_full_permutation"
    return "unsupported_full_permutation"


def holdout_gate_passed(strict_valid_rows: int, holdout_rows: int, threshold: float = READINESS_THRESHOLD) -> bool:
    if holdout_rows <= 0:
        return False
    return (strict_valid_rows / holdout_rows) >= threshold


def preserve_holdout_failure(rows: pd.DataFrame) -> None:
    failing = rows[
        rows["model_family"].eq("Mistral 7B Instruct")
        & rows["dataset"].eq("Amazon Sports")
        & rows["holdout_gate_passed"].eq("false")
    ]
    if len(failing) != 1:
        raise ValueError("Mistral Sports holdout failure was not preserved")
    if rows["combined_evidence_overrides_gate"].ne("false").any():
        raise ValueError("combined evidence may not override the holdout gate")


def assess_reporting_policy(policy: str) -> dict[str, str]:
    if policy == "strict-valid-only utility":
        return {
            "selection_bias_risk": "high",
            "survivorship_bias_risk": "high",
            "cross_model_comparability": "weak_model_dependent_denominators",
            "allowed_for_primary_results": "false",
            "allowed_for_diagnostic_results": "true",
        }
    if policy == "invalid-as-zero utility":
        return {
            "selection_bias_risk": "medium",
            "survivorship_bias_risk": "low",
            "cross_model_comparability": "requires_explicit_operational_metric_contract",
            "allowed_for_primary_results": "false",
            "allowed_for_diagnostic_results": "false",
        }
    return {
        "selection_bias_risk": "low",
        "survivorship_bias_risk": "low",
        "cross_model_comparability": "clear_when_validity_and_utility_are_separate",
        "allowed_for_primary_results": "true",
        "allowed_for_diagnostic_results": "true",
    }


def final_decision_from_evidence(
    *,
    official_topk_supported: bool,
    full20_scientifically_required: bool,
    holdout_failed: bool,
    utility_used: bool,
    prompt_v3_created: bool,
) -> str:
    if utility_used or prompt_v3_created:
        return "audit_incomplete"
    if official_topk_supported and not full20_scientifically_required and holdout_failed:
        return SELECTED_DECISION
    if holdout_failed:
        return "approve_ranking_only_full_execution_reranking_blocked"
    return "audit_incomplete"


def build_source_inventory(root: Path) -> pd.DataFrame:
    rows = [
        {
            "source_id": "SRC-001",
            "source_type": "paper",
            "source_path": "paper/Beyond Utility Evaluating LLM As Recommander.pdf",
            "symbol_or_section": "local_pdf_file",
            "evidence_topic": "paper reranking output semantics",
            "candidate_count": "blocked_unextracted",
            "requested_output_count": "blocked_unextracted",
            "requires_complete_permutation": "blocked_ambiguous",
            "output_representation": "blocked_unextracted",
            "evaluation_cutoff": "blocked_unextracted",
            "parser_behavior": "not_applicable",
            "retry_behavior": "not_applicable",
            "evidence_classification": "blocked_ambiguous",
            "evidence_confidence": "medium",
            "supports_full20": "false",
            "supports_top5": "false",
            "supports_top10": "false",
            "conflicts_with_other_sources": "false",
            "notes": "PDF is hashed, but text extraction is not locally available; no paper-exact output-count claim is made.",
        },
        {
            "source_id": "SRC-002",
            "source_type": "official_repository",
            "source_path": "original_repo/EvaLLMasRecommender/eva/templates/template.txt",
            "symbol_or_section": "zero-shot templates",
            "evidence_topic": "ranking-style candidate recommendation output count",
            "candidate_count": "20",
            "requested_output_count": "10",
            "requires_complete_permutation": "false",
            "output_representation": "item_titles_free_text_list",
            "evaluation_cutoff": "5",
            "parser_behavior": "official parser expects title list",
            "retry_behavior": "handled by zero_shot.evaluate for invalid outputs",
            "evidence_classification": "verified_official_code",
            "evidence_confidence": "high",
            "supports_full20": "false",
            "supports_top5": "true",
            "supports_top10": "true",
            "conflicts_with_other_sources": "false",
            "notes": "Templates repeatedly request selecting top 10 items from candidate lists.",
        },
        {
            "source_id": "SRC-003",
            "source_type": "official_repository",
            "source_path": "original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt",
            "symbol_or_section": "llmrank reranking template",
            "evidence_topic": "reranking prompt wording",
            "candidate_count": "20",
            "requested_output_count": "ambiguous_rank_20",
            "requires_complete_permutation": "ambiguous",
            "output_representation": "natural_language_title_ranking",
            "evaluation_cutoff": "5",
            "parser_behavior": "paired with official top10 parser/evaluator in code",
            "retry_behavior": "not specified in template",
            "evidence_classification": "verified_official_code",
            "evidence_confidence": "medium",
            "supports_full20": "true",
            "supports_top5": "false",
            "supports_top10": "false",
            "conflicts_with_other_sources": "true",
            "notes": "The template says rank 20 candidates, but official system prompts and parsers require 10 outputs.",
        },
        {
            "source_id": "SRC-004",
            "source_type": "official_repository",
            "source_path": "original_repo/EvaLLMasRecommender/eva/codes/llms.py",
            "symbol_or_section": "callChatGPT/callChatGLM system messages",
            "evidence_topic": "LLM response contract",
            "candidate_count": "20",
            "requested_output_count": "10",
            "requires_complete_permutation": "false",
            "output_representation": "python_list_of_item_titles",
            "evaluation_cutoff": "5",
            "parser_behavior": "expects a list of titles",
            "retry_behavior": "no retry here; zero_shot.evaluate retries invalid responses",
            "evidence_classification": "verified_official_code",
            "evidence_confidence": "high",
            "supports_full20": "false",
            "supports_top5": "true",
            "supports_top10": "true",
            "conflicts_with_other_sources": "true",
            "notes": "System message says choose 10 items and sort from highest to lowest priority.",
        },
        {
            "source_id": "SRC-005",
            "source_type": "official_repository",
            "source_path": "original_repo/EvaLLMasRecommender/eva/codes/output_refine.py",
            "symbol_or_section": "output_refine_func(..., target_len)",
            "evidence_topic": "output parsing and list completion",
            "candidate_count": "20",
            "requested_output_count": "10",
            "requires_complete_permutation": "false",
            "output_representation": "python/json-like title list",
            "evaluation_cutoff": "5",
            "parser_behavior": "invalid length is truncated or padded for target_len",
            "retry_behavior": "invalid indices returned to caller for retry",
            "evidence_classification": "verified_official_code",
            "evidence_confidence": "high",
            "supports_full20": "false",
            "supports_top5": "true",
            "supports_top10": "true",
            "conflicts_with_other_sources": "true",
            "notes": "zero_shot.evaluate calls this with target_len=10.",
        },
        {
            "source_id": "SRC-006",
            "source_type": "official_repository",
            "source_path": "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
            "symbol_or_section": "read_llm_results; calHR; calNDCG; metric_len",
            "evidence_topic": "evaluation cutoff and expected LLM list length",
            "candidate_count": "20",
            "requested_output_count": "10",
            "requires_complete_permutation": "false",
            "output_representation": "mapped title list to item ids",
            "evaluation_cutoff": "5",
            "parser_behavior": "raises on malformed list, then length-checks for 10",
            "retry_behavior": "not in metric script",
            "evidence_classification": "verified_official_code",
            "evidence_confidence": "high",
            "supports_full20": "false",
            "supports_top5": "true",
            "supports_top10": "true",
            "conflicts_with_other_sources": "true",
            "notes": "Default metric_len is 5; read_llm_results checks len(results[i]) != 10.",
        },
        {
            "source_id": "SRC-007",
            "source_type": "local_contract",
            "source_path": "results/tables/canonical_llm_prompt_contract.csv",
            "symbol_or_section": "llm_rerank_index_json_full20_v1",
            "evidence_topic": "local full20 reranking prompt v1",
            "candidate_count": "20",
            "requested_output_count": "20",
            "requires_complete_permutation": "true",
            "output_representation": "strict_json_one_based_indices",
            "evaluation_cutoff": "5",
            "parser_behavior": "strict local parser expects full permutation",
            "retry_behavior": "no scientific retry",
            "evidence_classification": "adapted",
            "evidence_confidence": "high",
            "supports_full20": "true",
            "supports_top5": "false",
            "supports_top10": "false",
            "conflicts_with_other_sources": "true",
            "notes": "Local Phase 5A contract already labels this adapted_project_policy.",
        },
        {
            "source_id": "SRC-008",
            "source_type": "local_contract",
            "source_path": "results/tables/canonical_local_llm_reranking_prompt_v2_contract.csv",
            "symbol_or_section": "llm_rerank_index_json_full20_v2",
            "evidence_topic": "local full20 reranking prompt v2",
            "candidate_count": "20",
            "requested_output_count": "20",
            "requires_complete_permutation": "true",
            "output_representation": "strict_json_one_based_indices",
            "evaluation_cutoff": "5",
            "parser_behavior": "same strict parser as v1",
            "retry_behavior": "no scientific retry",
            "evidence_classification": "adapted",
            "evidence_confidence": "high",
            "supports_full20": "true",
            "supports_top5": "false",
            "supports_top10": "false",
            "conflicts_with_other_sources": "true",
            "notes": "Prompt v2 changed wording only; it did not create official-code exact evidence for full20.",
        },
        {
            "source_id": "SRC-009",
            "source_type": "local_result",
            "source_path": "results/tables/canonical_local_llm_hybrid_holdout_decision.csv",
            "symbol_or_section": "hybrid_policy_failed_disjoint_holdout",
            "evidence_topic": "full20 reliability gate",
            "candidate_count": "20",
            "requested_output_count": "20",
            "requires_complete_permutation": "true",
            "output_representation": "strict_json_one_based_indices",
            "evaluation_cutoff": "5",
            "parser_behavior": "strict full20 parser determines readiness",
            "retry_behavior": "no retry",
            "evidence_classification": "reconstructed",
            "evidence_confidence": "high",
            "supports_full20": "true",
            "supports_top5": "false",
            "supports_top10": "false",
            "conflicts_with_other_sources": "false",
            "notes": "Readiness gate failed at Mistral Sports 7/10; this diagnoses full20 reliability only.",
        },
    ]
    for row in rows:
        path = root / row["source_path"]
        if path.is_file():
            row["notes"] = f"{row['notes']} sha256={file_sha256(path)}"
    return frame(rows, SOURCE_INVENTORY_COLUMNS)


def build_metric_requirements(root: Path) -> pd.DataFrame:
    source = "results/tables/canonical_metric_contract.csv;original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py"
    rows = [
        metric_requirement("hr_at_5", "utility", "5", requires_top5=True, requires_candidate_membership=True, source=source, notes="Positive hit at cutoff 5 only needs the first five parsed recommendations."),
        metric_requirement("ndcg_at_5", "utility", "5", requires_top5=True, requires_candidate_membership=True, source=source, notes="With one positive item, NDCG@5 uses positive rank only when rank <= 5."),
        metric_requirement("novelty_at_5", "beyond_utility", "5", requires_top5=True, source=source, notes="Official self-information uses results[:metric_len]."),
        metric_requirement("catalog_coverage_at_5", "beyond_utility", "5", requires_top5=True, source=source, notes="Local catalog coverage is top-5 exposure over a catalog universe."),
        metric_requirement("item_coverage_at_5", "beyond_utility", "5", requires_top5=True, source=source, notes="Official ItemCoverage uses unique top-k recommended IDs over candidate pool."),
        metric_requirement("long_tail_rate_at_5", "beyond_utility", "5", requires_top5=True, source=source, notes="Official ACLT inspects results_cutoff = x[:metric_len]."),
        metric_requirement("average_popularity_at_5", "beyond_utility", "5", requires_top5=True, source=source, notes="Official ARP inspects top-k recommendation positions."),
        metric_requirement("popularity_bias_at_5", "beyond_utility", "5", requires_top5=True, source=source, status="blocked_ambiguous_definition", notes="Definition remains ambiguous, but full20 is not the blocker."),
        metric_requirement("intra_list_diversity_at_5", "beyond_utility", "5", requires_top5=True, source=source, status="blocked_ambiguous_definition", notes="Feature/similarity source is ambiguous; a top-5 list is sufficient once definition is frozen."),
        metric_requirement("duplicate_rate_at_5", "structural_quality", "5", requires_top5=True, requires_candidate_membership=True, source=source, notes="Duplicate detection at top-5 needs only the returned top-5 positions."),
        metric_requirement("candidate_validity_rate", "structural_quality", "5", requires_top5=True, requires_candidate_membership=True, source=source, notes="Candidate-validity at top-5 checks returned items against row candidates."),
        metric_requirement("hallucination_rate_at_5", "structural_quality", "5", requires_top5=True, requires_candidate_membership=True, source=source, notes="Official hallucination maps unmatched outputs to ID 0; top-5 version needs top-5 only."),
        metric_requirement("strict_parse_validity_rate", "structural_quality", "contract", requires_top5=False, requires_top10=True, minimum_output_length=10, source="results/tables/canonical_llm_parser_contract.csv", notes="Under a strict top-10 contract this checks exactly ten valid indices."),
        metric_requirement("operational_success_rate", "structural_quality", "contract", requires_top5=False, requires_top10=True, requires_raw_response=True, minimum_output_length=10, source="results/tables/canonical_local_llm_hybrid_holdout_execution.csv", notes="Backend success plus parse success is contract-specific, not a paper utility metric."),
        metric_requirement("position_bias_top5_diagnostic", "diagnostic", "5", requires_top5=True, requires_candidate_order=True, source="results/tables/positive_positions.csv", notes="Bias in exposed top-5 positions needs candidate order and top-5 outputs."),
        metric_requirement("position_bias_full20_diagnostic", "diagnostic", "20", requires_top5=False, requires_full20=True, requires_candidate_order=True, minimum_output_length=20, source="results/tables/positive_positions.csv", status="diagnostic_only", notes="Full-rank position-shift diagnostics require a complete permutation, but this is not needed for HR@5/NDCG@5."),
    ]
    return frame(rows, METRIC_REQUIREMENT_COLUMNS)


def build_full20_origin(root: Path) -> pd.DataFrame:
    rows = []
    for idx, contract_id in enumerate(["llm_rerank_index_json_full20_v1", "llm_rerank_index_json_full20_v2"], start=1):
        rows.append(
            {
                "audit_id": f"FULL20-{idx:03d}",
                "prompt_contract_id": contract_id,
                "task_type": "reranking",
                "candidate_count": "20",
                "requested_output_count": "20",
                "full_permutation_required": "true",
                "paper_evidence": "blocked: local paper PDF text not extracted",
                "official_code_evidence": "official llms.py/output_refine.py/llm_metrics.py support top10 title lists, not full20 strict JSON",
                "local_reconstruction_evidence": "Phase 5A prompt/parser contracts request strict JSON full permutation",
                "engineering_rationale": "machine-checkable candidate-index output and full reranking diagnostics",
                "parser_rationale": "strict parser convenience and complete-permutation structural validation",
                "scientific_necessity": "not_required_for_hr_ndcg_or_top5_beyond_utility",
                "origin_classification": classify_full20_origin(False, False, True),
                "paper_alignment": "adapted_not_paper_exact",
                "status": "diagnostic_only_after_failed_holdout",
                "notes": "Full20 may remain useful as diagnostic evidence, but cannot support a canonical full-reranking readiness claim after P1E.",
            }
        )
    return frame(rows, FULL20_ORIGIN_COLUMNS)


def build_protocol_options(root: Path) -> pd.DataFrame:
    rows = [
        {
            "protocol_option_id": "option_a_full20_strict_permutation",
            "protocol_name": "Full 20-item strict permutation",
            "requested_output_count": "20",
            "requires_complete_permutation": "true",
            "prompt_contract_source": "Phase 5A/P1E adapted local contracts",
            "paper_alignment": "adapted_not_paper_exact",
            "official_code_alignment": "weak_conflicts_with_top10_parser",
            "scientific_task_preserved": "partially",
            "metric_sufficiency": "more_than_required_for_top5_metrics",
            "cross_model_comparability": "weak_after_failed_holdout",
            "parser_reliability_evidence": "holdout 53/60 strict valid; Mistral Sports 7/10 failed",
            "invalid_output_bias_risk": "high",
            "post_hoc_tuning_risk": "high_if_tuned_again",
            "requires_new_prompt": "false",
            "requires_new_parser": "false",
            "requires_confirmation_pilot": "false",
            "allows_full_execution": "false",
            "execution_classification": "blocked_for_primary",
            "methodological_risk": "high",
            "recommendation_status": "not_recommended",
            "rationale": "Failed the predeclared holdout gate and is not necessary for top-5 metrics.",
            "notes": "Can be retained only as a diagnostic option if explicitly labeled.",
        },
        {
            "protocol_option_id": SELECTED_PROTOCOL_OPTION_ID,
            "protocol_name": "Strict top-10 reranking confirmation pilot",
            "requested_output_count": "10",
            "requires_complete_permutation": "false",
            "prompt_contract_source": "official llms.py/output_refine.py/llm_metrics.py top10 behavior",
            "paper_alignment": PAPER_ALIGNMENT_LABEL,
            "official_code_alignment": "strong_verified_top10",
            "scientific_task_preserved": "true",
            "metric_sufficiency": "sufficient_for_hr_ndcg_and_top5_beyond_utility",
            "cross_model_comparability": "requires_frozen_common_contract_and_new_disjoint_pilot",
            "parser_reliability_evidence": "not_yet_piloted_under_new_contract",
            "invalid_output_bias_risk": "medium_until_confirmed",
            "post_hoc_tuning_risk": "low_if_justified_by_official_code_not_pass_rates",
            "requires_new_prompt": "true",
            "requires_new_parser": "true",
            "requires_confirmation_pilot": "true",
            "allows_full_execution": "false",
            "execution_classification": "confirmation_pilot_required",
            "methodological_risk": "medium",
            "recommendation_status": "recommended",
            "rationale": "Official code repeatedly requires 10-item LLM outputs while top-5 metrics need only top-5.",
            "notes": "This is a contract change justified by primary local official-code evidence, not by prompt pass rates.",
        },
        {
            "protocol_option_id": "option_c_strict_top5_output",
            "protocol_name": "Strict top-5 reranking output",
            "requested_output_count": "5",
            "requires_complete_permutation": "false",
            "prompt_contract_source": "metric sufficiency only",
            "paper_alignment": "metric_sufficient_but_official_code_requests_top10",
            "official_code_alignment": "weak",
            "scientific_task_preserved": "partially",
            "metric_sufficiency": "sufficient_for_hr_ndcg_at_5",
            "cross_model_comparability": "possible_with_new_contract",
            "parser_reliability_evidence": "not_piloted",
            "invalid_output_bias_risk": "medium",
            "post_hoc_tuning_risk": "medium",
            "requires_new_prompt": "true",
            "requires_new_parser": "true",
            "requires_confirmation_pilot": "true",
            "allows_full_execution": "false",
            "execution_classification": "not_selected",
            "methodological_risk": "medium",
            "recommendation_status": "not_recommended",
            "rationale": "Top-5 is enough for top-5 metrics, but official executable code points to top-10.",
            "notes": "Would unnecessarily diverge from official parser behavior.",
        },
        {
            "protocol_option_id": "option_d_ranking_only_reranking_blocked",
            "protocol_name": "Ranking-only full execution",
            "requested_output_count": "5",
            "requires_complete_permutation": "false",
            "prompt_contract_source": "existing Phase 5A ranking contract",
            "paper_alignment": "adapted_ranking_only",
            "official_code_alignment": "adapted_strict_json",
            "scientific_task_preserved": "ranking_task_only",
            "metric_sufficiency": "sufficient_for_ranking_top5_metrics",
            "cross_model_comparability": "stronger_than_failed_reranking",
            "parser_reliability_evidence": "ranking prompt was not failed by P1E",
            "invalid_output_bias_risk": "lower_for_ranking_contract",
            "post_hoc_tuning_risk": "low",
            "requires_new_prompt": "false",
            "requires_new_parser": "false",
            "requires_confirmation_pilot": "false",
            "allows_full_execution": "true",
            "execution_classification": "available_fallback_not_selected",
            "methodological_risk": "low_for_ranking_medium_for_missing_reranking",
            "recommendation_status": "not_recommended",
            "rationale": "Fallback if top-K reranking evidence were insufficient; official top10 evidence makes reranking confirmation preferable.",
            "notes": "May be used later if top-K confirmation also fails.",
        },
        {
            "protocol_option_id": "option_e_full20_diagnostic_only",
            "protocol_name": "Full-20 diagnostic-only execution",
            "requested_output_count": "20",
            "requires_complete_permutation": "true",
            "prompt_contract_source": "Phase 5A/P1E adapted full20 contracts",
            "paper_alignment": "adapted_diagnostic_only",
            "official_code_alignment": "weak",
            "scientific_task_preserved": "diagnostic_only",
            "metric_sufficiency": "sufficient_but_overconstrained",
            "cross_model_comparability": "weak_due_invalid_denominators",
            "parser_reliability_evidence": "failed holdout gate",
            "invalid_output_bias_risk": "high",
            "post_hoc_tuning_risk": "low_if_no_tuning",
            "requires_new_prompt": "false",
            "requires_new_parser": "false",
            "requires_confirmation_pilot": "false",
            "allows_full_execution": "false",
            "execution_classification": "diagnostic_only_not_selected",
            "methodological_risk": "medium",
            "recommendation_status": "not_recommended",
            "rationale": "A diagnostic can be defensible but should not precede resolving the official top-K contract.",
            "notes": "Full20 diagnostic remains possible later with explicit invalid-output accounting.",
        },
        {
            "protocol_option_id": "option_f_structured_decoding_adaptation",
            "protocol_name": "Structured decoding adaptation",
            "requested_output_count": "contract_dependent",
            "requires_complete_permutation": "contract_dependent",
            "prompt_contract_source": "local support not documented in current evidence",
            "paper_alignment": "adapted_constrained_decoding",
            "official_code_alignment": "not_verified",
            "scientific_task_preserved": "unresolved",
            "metric_sufficiency": "unresolved",
            "cross_model_comparability": "unresolved",
            "parser_reliability_evidence": "no local evidence inspected that uniqueness can be enforced",
            "invalid_output_bias_risk": "unknown",
            "post_hoc_tuning_risk": "medium",
            "requires_new_prompt": "possibly",
            "requires_new_parser": "possibly",
            "requires_confirmation_pilot": "true",
            "allows_full_execution": "false",
            "execution_classification": "offline_audit_required_if_pursued",
            "methodological_risk": "medium",
            "recommendation_status": "not_recommended",
            "rationale": "No local evidence currently demonstrates a valid constrained-decoding mechanism.",
            "notes": "Do not assume JSON Schema can enforce uniqueness without separate local evidence.",
        },
    ]
    return frame(rows, PROTOCOL_OPTIONS_COLUMNS)


def build_holdout_interpretation(root: Path) -> pd.DataFrame:
    comparison = read_csv(root, "results/tables/canonical_local_llm_hybrid_holdout_prior_pilot_comparison.csv")
    rows = []
    for record in comparison.to_dict("records"):
        prior_rows = int(record["prior_rows"])
        prior_valid = int(record["prior_strict_valid_rows"])
        holdout_rows = int(record["holdout_rows"])
        holdout_valid = int(record["holdout_strict_valid_rows"])
        combined_rows = prior_rows + holdout_rows
        combined_valid = prior_valid + holdout_valid
        low, high = wilson_interval(holdout_valid, holdout_rows)
        gate_passed = holdout_gate_passed(holdout_valid, holdout_rows)
        interpretation = "holdout_gate_passed_descriptive_reliability_only" if gate_passed else "holdout_gate_failed_preserved"
        rows.append(
            {
                "model_family": record["model_family"],
                "dataset": record["dataset"],
                "selected_prompt_contract_id": record["selected_prompt_contract_id"],
                "prior_rows": str(prior_rows),
                "prior_strict_valid_rows": str(prior_valid),
                "prior_validity_rate": format_float(prior_valid / prior_rows),
                "holdout_rows": str(holdout_rows),
                "holdout_strict_valid_rows": str(holdout_valid),
                "holdout_validity_rate": format_float(holdout_valid / holdout_rows),
                "combined_rows_descriptive": str(combined_rows),
                "combined_strict_valid_rows_descriptive": str(combined_valid),
                "combined_validity_rate_descriptive": format_float(combined_valid / combined_rows),
                "wilson_interval_low": format_float(low),
                "wilson_interval_high": format_float(high),
                "holdout_gate_threshold": format_float(READINESS_THRESHOLD),
                "holdout_gate_passed": bool_text(gate_passed),
                "combined_evidence_overrides_gate": "false",
                "interpretation": interpretation,
                "notes": "Combined counts estimate format reliability only and do not override the predeclared disjoint-holdout gate.",
            }
        )
    return frame(rows, HOLDOUT_INTERPRETATION_COLUMNS)


def build_utility_bias() -> pd.DataFrame:
    policies = [
        (
            "BIAS-001",
            "strict-valid-only utility",
            "drop invalid rows from utility denominator",
            "model/dataset-specific valid rows only",
            "not_paper_exact",
            "false",
            "diagnostic_only",
            "Useful only with explicit validity-rate companion metrics.",
        ),
        (
            "BIAS-002",
            "invalid-as-zero utility",
            "score invalid outputs as utility failure",
            "all requested rows",
            "new_operational_metric_not_current_contract",
            "true",
            "not_recommended",
            "Would require a new metric contract; this audit does not modify Phase 4B/5A contracts.",
        ),
        (
            "BIAS-003",
            "separate validity plus valid-only utility",
            "report validity and valid-only utility as separate layers",
            "valid-only utility plus all-row validity denominator",
            "adapted_diagnostic_reporting",
            "false",
            "recommended_for_diagnostics",
            "Separates format reliability from conditional quality without using utility to choose protocol.",
        ),
        (
            "BIAS-004",
            "ranking-only primary results",
            "reranking invalidity not applicable",
            "all ranking rows under ranking parser",
            "adapted_ranking_only",
            "false",
            "allowed_if_selected",
            "Clean fallback when reranking remains blocked.",
        ),
        (
            "BIAS-005",
            "full-permutation diagnostic results",
            "preserve invalid outputs; utility only conditional",
            "all requested rows for validity; strict-valid rows for conditional utility",
            "adapted_diagnostic_only",
            "false",
            "allowed_diagnostic_only",
            "Cannot be primary after the failed full20 holdout gate.",
        ),
    ]
    rows = []
    for assessment_id, policy, treatment, denominator, alignment, requires_contract, status, rationale in policies:
        risk = assess_reporting_policy(policy)
        rows.append(
            {
                "assessment_id": assessment_id,
                "reporting_policy": policy,
                "invalid_output_treatment": treatment,
                "denominator_policy": denominator,
                "selection_bias_risk": risk["selection_bias_risk"],
                "survivorship_bias_risk": risk["survivorship_bias_risk"],
                "cross_model_comparability": risk["cross_model_comparability"],
                "cross_dataset_comparability": "clear_only_when_denominators_are_explicit",
                "paper_alignment": alignment,
                "requires_new_metric_contract": requires_contract,
                "allowed_for_primary_results": risk["allowed_for_primary_results"] if status != "allowed_if_selected" else "true",
                "allowed_for_diagnostic_results": risk["allowed_for_diagnostic_results"],
                "recommendation_status": status,
                "rationale": rationale,
                "notes": "Utility must not determine protocol selection.",
            }
        )
    return frame(rows, UTILITY_BIAS_COLUMNS)


def cohort_eligible_rows(root: Path) -> dict[tuple[str, str], int]:
    summary = read_csv(root, "results/tables/experiment_cohort_summary.csv")
    subset = summary[summary["cohort_id"].eq("plan_e2_llm_strict_title")]
    return {(row["dataset"], row["task_type"]): int(row["eligible_rows"]) for row in subset.to_dict("records")}


def calculate_request_counts(root: Path) -> dict[str, int]:
    eligible = cohort_eligible_rows(root)
    ranking_rows = sum(eligible[(dataset, "ranking")] for dataset in DATASET_ORDER)
    reranking_rows = sum(eligible[(dataset, "reranking")] for dataset in DATASET_ORDER)
    model_count = len(MODEL_ORDER)
    return {
        "ranking_only_full_request_count": ranking_rows * model_count,
        "full_ranking_plus_reranking_request_count": (ranking_rows + reranking_rows) * model_count,
        "topk_confirmation_pilot_request_count": 10 * len(DATASET_ORDER) * model_count,
        "diagnostic_full_permutation_request_count": reranking_rows * model_count,
    }


def build_execution_scope(root: Path) -> pd.DataFrame:
    eligible = cohort_eligible_rows(root)
    rows = []
    index = 1
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            rows.append(
                {
                    "execution_scope_id": f"SCOPE-{index:03d}",
                    "protocol_option_id": SELECTED_PROTOCOL_OPTION_ID,
                    "model_family": model,
                    "dataset": dataset,
                    "task_type": "ranking",
                    "cohort_id": "plan_e2_llm_strict_title",
                    "eligible_rows": str(eligible[(dataset, "ranking")]),
                    "prompt_contract_id": "llm_rank_index_json_top5_v1",
                    "parser_contract_id": "llm_rank_strict_primary_parser_v1",
                    "execution_classification": "deferred_full_execution",
                    "primary_or_diagnostic": "primary_after_separate_full_execution_approval",
                    "request_count": "0",
                    "allowed_to_execute": "false",
                    "blocking_reason": "selected next phase is top-K reranking confirmation pilot, not full matrix execution",
                    "required_pre_execution_phase": RECOMMENDED_NEXT_TASK,
                    "notes": "Ranking remains technically separable, but this selected protocol freezes a reranking confirmation task first.",
                }
            )
            index += 1
            rows.append(
                {
                    "execution_scope_id": f"SCOPE-{index:03d}",
                    "protocol_option_id": SELECTED_PROTOCOL_OPTION_ID,
                    "model_family": model,
                    "dataset": dataset,
                    "task_type": "reranking",
                    "cohort_id": "plan_e2_llm_strict_title",
                    "eligible_rows": str(eligible[(dataset, "reranking")]),
                    "prompt_contract_id": "to_be_frozen_top10_reranking_contract",
                    "parser_contract_id": "to_be_frozen_top10_strict_parser",
                    "execution_classification": "confirmation_pilot",
                    "primary_or_diagnostic": "primary_protocol_confirmation",
                    "request_count": "10",
                    "allowed_to_execute": "true",
                    "blocking_reason": "",
                    "required_pre_execution_phase": RECOMMENDED_NEXT_TASK,
                    "notes": "Exactly 10 new disjoint rows per dataset/model are prospective for the next phase; this audit executes none.",
                }
            )
            index += 1
    return frame(rows, EXECUTION_SCOPE_COLUMNS)


def build_final_decision(root: Path, holdout: pd.DataFrame) -> pd.DataFrame:
    holdout_failed = holdout["holdout_gate_passed"].eq("false").any()
    decision = final_decision_from_evidence(
        official_topk_supported=True,
        full20_scientifically_required=False,
        holdout_failed=holdout_failed,
        utility_used=False,
        prompt_v3_created=False,
    )
    if decision != SELECTED_DECISION:
        raise ValueError(f"unexpected final decision {decision}")
    rows = [
        {
            "audit_id": "canonical_local_llm_final_reranking_protocol_decision_v1",
            "paper_reranking_output_count_verified": "false",
            "official_reranking_output_count_verified": "true",
            "full20_paper_exact": "false",
            "full20_official_code_exact": "false",
            "full20_scientifically_required": "false",
            "topk_scientifically_sufficient": "true",
            "holdout_failure_preserved": "true",
            "holdout_cells_passed": "5",
            "holdout_cells_total": "6",
            "utility_used_for_decision": "false",
            "prompt_v3_created": "false",
            "further_wording_tuning_allowed": "false",
            "selected_protocol_option_id": SELECTED_PROTOCOL_OPTION_ID,
            "selected_execution_classification": "top10_reranking_confirmation_pilot_required",
            "ranking_full_execution_allowed": "false",
            "reranking_full_execution_allowed": "false",
            "reranking_diagnostic_execution_allowed": "false",
            "confirmation_pilot_required": "true",
            "recommended_decision": decision,
            "blocking_reason": "full20 reranking failed the predeclared holdout gate and is not official-code exact",
            "required_corrective_action": "Freeze a top-10 strict reranking contract from official-code evidence and validate it on a new disjoint confirmation pilot.",
            "recommended_single_next_task": RECOMMENDED_NEXT_TASK,
            "notes": "No full canonical execution is approved by this audit.",
        }
    ]
    return frame(rows, FINAL_DECISION_COLUMNS)


def build_all_frames(root: Path) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    before_hashes = hash_protected_inputs(root)
    source_inventory = build_source_inventory(root)
    metric_requirements = build_metric_requirements(root)
    full20_origin = build_full20_origin(root)
    protocol_options = build_protocol_options(root)
    holdout_interpretation = build_holdout_interpretation(root)
    utility_bias = build_utility_bias()
    execution_scope = build_execution_scope(root)
    final_decision = build_final_decision(root, holdout_interpretation)
    frames = {
        "source_inventory": source_inventory,
        "metric_requirements": metric_requirements,
        "full20_origin": full20_origin,
        "protocol_options": protocol_options,
        "holdout_interpretation": holdout_interpretation,
        "utility_bias": utility_bias,
        "execution_scope": execution_scope,
        "final_decision": final_decision,
    }
    validate_frames(frames)
    after_hashes = hash_protected_inputs(root)
    if before_hashes != after_hashes:
        raise RuntimeError("protected input hashes changed while building frames")
    return before_hashes, frames


def validate_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (_path, columns) in OUTPUT_TABLES.items():
        if key not in frames:
            raise ValueError(f"missing frame {key}")
        if list(frames[key].columns) != columns:
            raise ValueError(f"{key} schema mismatch")
    if len(frames["holdout_interpretation"]) != 6:
        raise ValueError("holdout interpretation must contain six cells")
    preserve_holdout_failure(frames["holdout_interpretation"])
    if frames["protocol_options"]["recommendation_status"].eq("recommended").sum() != 1:
        raise ValueError("exactly one protocol option must be recommended")
    decision = frames["final_decision"].iloc[0]
    if decision["recommended_decision"] != SELECTED_DECISION:
        raise ValueError("wrong final decision")
    if decision["utility_used_for_decision"] != "false":
        raise ValueError("utility may not influence the decision")
    if decision["prompt_v3_created"] != "false":
        raise ValueError("prompt v3 may not be created")
    if decision["further_wording_tuning_allowed"] != "false":
        raise ValueError("further wording tuning must be frozen false")
    if frames["full20_origin"]["origin_classification"].ne("adapted_full_permutation").any():
        raise ValueError("full20 origin must remain adapted")
    if frames["metric_requirements"].query("metric_id == 'hr_at_5'").iloc[0]["requires_full20"] != "false":
        raise ValueError("HR@5 must not require full20")
    if frames["metric_requirements"].query("metric_id == 'ndcg_at_5'").iloc[0]["requires_full20"] != "false":
        raise ValueError("NDCG@5 must not require full20")


def write_tables(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (relative, columns) in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].loc[:, columns].to_csv(path, index=False, lineterminator="\n")


def markdown_table(data: pd.DataFrame, columns: Sequence[str]) -> str:
    if data.empty:
        return "_No rows._"
    subset = data.loc[:, list(columns)].astype(str)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in subset.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(root: Path, frames: Mapping[str, pd.DataFrame], protected_hashes: Mapping[str, str], commands: Sequence[str]) -> None:
    decision = frames["final_decision"].iloc[0].to_dict()
    counts = calculate_request_counts(root)
    metric = frames["metric_requirements"]
    holdout = frames["holdout_interpretation"]
    options = frames["protocol_options"]
    source = frames["source_inventory"]
    full20 = frames["full20_origin"]
    source_top10 = source[source["supports_top10"].eq("true")]
    source_full20 = source[source["supports_full20"].eq("true")]
    section_text = {
        1: f"Offline Phase 5B-P1F audit on branch `plan-d-extensions`, commit `02a20b8b509b8cf834a8766dd1cf1f751ae204bf`. Protected evidence files hashed: `{len(protected_hashes)}`.",
        2: f"Decision: `{decision['recommended_decision']}`. The failed full20 holdout is preserved, and no full execution is approved.",
        3: "This audit invokes no Ollama endpoint, creates no request/response evidence, and does not create prompt v3.",
        4: markdown_table(full20, ["prompt_contract_id", "origin_classification", "scientific_necessity", "status"]),
        5: "Prompt v1 selected Qwen2 full20 reranking only as an adapted local format contract.",
        6: "Prompt v2 selected Mistral full20 reranking only as an adapted local format contract.",
        7: markdown_table(holdout, ["model_family", "dataset", "holdout_strict_valid_rows", "holdout_rows", "holdout_gate_passed"]),
        8: "Mistral Sports remains failed at 7/10. Combined pilot+holdout evidence is descriptive and never overrides the gate.",
        9: "The local paper PDF is hashed but not locally text-extracted, so no paper-exact output-count claim is made.",
        10: markdown_table(source[["official_repository" in value for value in source["source_type"]]], ["source_path", "requested_output_count", "requires_complete_permutation", "supports_top10"]),
        11: "The frozen canonical candidate sets use 20 candidates per row for Plan E2 LLM-compatible ranking and reranking.",
        12: "Official executable code supports 10 returned recommendations; local full20 contracts require 20 but are adapted.",
        13: "Top-K semantics preserve the recommendation task because HR@5/NDCG@5 and beyond-utility@5 consume the first five positions.",
        14: "Official parsing accepts title lists and checks length 10; local Phase 5A parsing accepts strict JSON one-based indices.",
        15: "The official and canonical metric cutoff is 5 for HR/NDCG and top-5 beyond-utility calculations.",
        16: markdown_table(metric, ["metric_id", "evaluation_cutoff", "requires_top5", "requires_top10", "requires_full20"]),
        17: "HR@5 requires only the positive item's presence in the first five returned recommendations.",
        18: "NDCG@5 requires only the positive item's one-based rank when that rank is at most 5.",
        19: "Beyond-utility metrics frozen at @5 use top-5 exposures; only full-rank position diagnostics require full20.",
        20: "The full20 requirement originated in local reconstruction, parser convenience, and diagnostic engineering.",
        21: f"Full20 classification: `{full20.iloc[0]['origin_classification']}`; paper alignment: `{full20.iloc[0]['paper_alignment']}`.",
        22: "Full20 reliability is insufficient for primary execution because one disjoint holdout cell failed the 0.80 threshold.",
        23: "Prompt tuning history includes v1, v2, hybrid selection, and disjoint holdout; more wording tuning would be post-hoc.",
        24: "No further wording-only prompt revision may be approved using existing pilot or holdout outcomes.",
        25: "Prompt v3 is rejected in this phase; a new contract must be justified by paper/official code, not pass rates.",
        26: markdown_table(holdout, ["model_family", "dataset", "combined_validity_rate_descriptive", "wilson_interval_low", "wilson_interval_high"]),
        27: "Combined validity counts estimate format reliability only and are not a readiness decision.",
        28: markdown_table(frames["utility_bias"], ["reporting_policy", "selection_bias_risk", "allowed_for_primary_results", "allowed_for_diagnostic_results"]),
        29: "Valid-only utility has model- and dataset-dependent denominators, so it is diagnostic only.",
        30: "Ranking-only remains a fallback but is not the selected path because official-code top10 evidence supports a reranking confirmation pilot.",
        31: "Strict top-5 is metric-sufficient but weaker than top10 against official-code evidence.",
        32: "Strict top-10 is the selected top-K reranking contract to validate next.",
        33: "Full20 can only be diagnostic after the failed holdout and is not selected for full execution.",
        34: "Structured decoding is not selected because local support is not documented in the audited evidence.",
        35: markdown_table(options, ["protocol_option_id", "requested_output_count", "execution_classification", "recommendation_status"]),
        36: f"Selected protocol: `{SELECTED_PROTOCOL_OPTION_ID}`.",
        37: f"Paper-alignment label: `{PAPER_ALIGNMENT_LABEL}`.",
        38: "Primary execution is limited to a future top10 reranking confirmation pilot; no full matrix is approved.",
        39: markdown_table(frames["execution_scope"], ["model_family", "dataset", "task_type", "request_count", "allowed_to_execute"]),
        40: f"Request-count implications: ranking-only full `{counts['ranking_only_full_request_count']}`, ranking+reranking full `{counts['full_ranking_plus_reranking_request_count']}`, top-K confirmation `{counts['topk_confirmation_pilot_request_count']}`, full20 diagnostic reranking `{counts['diagnostic_full_permutation_request_count']}`.",
        41: "Blocked components: full20 primary reranking and full canonical matrix execution.",
        42: "Required confirmation phase is a new disjoint top10 reranking confirmation pilot.",
        43: "Future reports must separate parse validity, operational success, and conditional utility.",
        44: "Limitations: paper PDF text remains unverified locally, and the top10 strict JSON contract has not yet been piloted.",
        45: "Protected hashes were computed before and after generation; the runner fails if evidence hashes change.",
        46: "\n".join(f"- `{command}`" for command in commands),
        47: markdown_table(frames["final_decision"], ["recommended_decision", "selected_protocol_option_id", "confirmation_pilot_required", "utility_used_for_decision"]),
        48: f"Recommended single next task: `{decision['recommended_single_next_task']}`.",
    }
    headings = [
        "Scope and provenance",
        "Executive summary",
        "Safety boundaries",
        "Phase 5A reranking contract",
        "Prompt-v1 pilot evidence",
        "Prompt-v2 corrective-pilot evidence",
        "Hybrid holdout evidence",
        "Preservation of the failed holdout gate",
        "Paper reranking task evidence",
        "Official repository reranking evidence",
        "Candidate-set size",
        "Requested output count",
        "Top-K versus full-permutation semantics",
        "Parser assumptions",
        "Evaluation cutoff",
        "Metric input requirements",
        "Whether HR@5 requires full permutation",
        "Whether NDCG@5 requires full permutation",
        "Whether beyond-utility metrics require full permutation",
        "Origin of the local full-20 contract",
        "Full-20 paper-alignment classification",
        "Full-permutation reliability findings",
        "Prompt-tuning history",
        "No-more-wording-tuning policy",
        "Prompt-v3 rejection or necessity",
        "Holdout statistical interpretation",
        "Combined evidence limitations",
        "Invalid-output selection bias",
        "Valid-only utility limitations",
        "Ranking-only protocol option",
        "Top-5 reranking option",
        "Top-10 reranking option",
        "Full-20 diagnostic option",
        "Structured-decoding option",
        "Protocol comparison",
        "Selected final protocol",
        "Paper-alignment label",
        "Primary versus diagnostic execution",
        "Full execution scope",
        "Request-count implications",
        "Blocked components",
        "Required confirmation phase",
        "Reporting requirements",
        "Limitations",
        "Protected-input immutability",
        "Commands executed",
        "Final audit decision",
        "Recommended single next task",
    ]
    lines = ["# Final Canonical Local LLM Reranking Protocol Decision", ""]
    for index, heading in enumerate(headings, start=1):
        lines.append(f"## {index}. {heading}")
        lines.append("")
        lines.append(section_text[index])
        lines.append("")
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def output_paths(root: Path) -> list[Path]:
    return [root / path for path, _columns in OUTPUT_TABLES.values()] + [root / REPORT_PATH]


def validate_written_outputs(root: Path, frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, (relative, columns) in OUTPUT_TABLES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"empty output {path}")
        data = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(data.columns) != columns:
            raise ValueError(f"{relative} schema mismatch")
        hashes[relative.as_posix()] = file_sha256(path)
    report = root / REPORT_PATH
    if not report.is_file() or report.stat().st_size == 0:
        raise FileNotFoundError(report)
    section_count = sum(1 for line in report.read_text(encoding="utf-8").splitlines() if line.startswith("## "))
    if section_count != 48:
        raise ValueError(f"report section count mismatch: {section_count}")
    hashes[REPORT_PATH.as_posix()] = file_sha256(report)
    if frames["final_decision"].iloc[0]["prompt_v3_created"] != "false":
        raise ValueError("prompt v3 flag changed")
    if frames["final_decision"].iloc[0]["utility_used_for_decision"] != "false":
        raise ValueError("utility flag changed")
    return hashes

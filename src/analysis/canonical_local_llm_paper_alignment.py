"""Paper-result inventory and paper-alignment classification for Phase 5C."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.analysis.canonical_local_llm_baseline_alignment import BASELINES
from src.analysis.canonical_local_llm_full_evaluation import DATASET_ORDER, MODEL_ORDER, TASK_ORDER, clean_cell, frame


PAPER_RESULT_INVENTORY_COLUMNS = [
    "paper_result_id",
    "paper_section_or_table",
    "model_or_baseline",
    "dataset",
    "task_type",
    "metric_id",
    "paper_value",
    "aggregation_policy",
    "seed_count",
    "candidate_count",
    "requested_output_count",
    "evidence_source",
    "evidence_classification",
    "evidence_confidence",
    "local_comparability_status",
    "blocking_reason",
    "notes",
]

PAPER_ALIGNMENT_COLUMNS = [
    "alignment_id",
    "local_result_type",
    "local_model_or_baseline",
    "paper_model_or_baseline",
    "dataset",
    "task_type",
    "metric_id",
    "local_value",
    "paper_value",
    "absolute_difference",
    "relative_difference",
    "same_model",
    "same_dataset_artifact",
    "same_candidate_protocol",
    "same_output_protocol",
    "same_metric_definition",
    "comparison_classification",
    "agreement_classification",
    "claim_allowed",
    "blocking_reason",
    "notes",
]

PAPER_MODELS = ["GPT-3.5", "GPT-4o-mini", "GPT-4o", "Claude 3 Haiku", "Llama3-70B-Instruct"]
PAPER_METRICS = ["hr_at_5", "ndcg_at_5", "novelty_at_5", "item_coverage_at_5", "long_tail_rate_at_5", "average_popularity_at_5"]


def build_paper_result_inventory() -> pd.DataFrame:
    """Inventory only source-traceable paper-result evidence available locally.

    The project has local paper and official-code evidence for metric identities
    and candidate counts, but no machine-verified numeric paper table. Numeric
    values therefore remain blank by design.
    """

    rows: list[dict[str, Any]] = []
    counter = 1
    for metric_id in PAPER_METRICS:
        rows.append(
            {
                "paper_result_id": f"PAPERINV-{counter:04d}",
                "paper_section_or_table": "paper metric/result tables not machine-extracted locally",
                "model_or_baseline": "paper LLM and baseline families",
                "dataset": "Amazon Beauty;Amazon Sports;ML-1M;LastFM",
                "task_type": "ranking;reranking",
                "metric_id": metric_id,
                "paper_value": "",
                "aggregation_policy": "unverified in local machine-readable form",
                "seed_count": "",
                "candidate_count": "20 where supported by official/local protocol evidence",
                "requested_output_count": "5 for utility cutoff; official reranking code reads top-k prefixes",
                "evidence_source": "paper/Beyond Utility Evaluating LLM As Recommander.pdf;original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py;report/reproducibility_audit_current_state.md",
                "evidence_classification": "verified_metric_code_but_numeric_value_blocked",
                "evidence_confidence": "medium",
                "local_comparability_status": "blocked_paper_value_unverified",
                "blocking_reason": "no verified machine-readable local paper numeric value",
                "notes": "numeric paper value intentionally blank; no value was inferred from prose",
            }
        )
        counter += 1
    for model in PAPER_MODELS:
        rows.append(
            {
                "paper_result_id": f"PAPERINV-{counter:04d}",
                "paper_section_or_table": "paper LLM model inventory",
                "model_or_baseline": model,
                "dataset": "Amazon Beauty;Amazon Sports;ML-1M;LastFM",
                "task_type": "ranking;reranking",
                "metric_id": "model_identity",
                "paper_value": "",
                "aggregation_policy": "",
                "seed_count": "",
                "candidate_count": "",
                "requested_output_count": "",
                "evidence_source": "report/canonical_llm_prompt_parsing_and_execution_preflight.md;report/reproducibility_audit_current_state.md",
                "evidence_classification": "locally_documented_model_family",
                "evidence_confidence": "medium",
                "local_comparability_status": "not_directly_comparable_to_qwen2_or_mistral",
                "blocking_reason": "paper model not executed in Phase 5C and local Qwen2/Mistral are substitutes/extensions",
                "notes": "recorded to prevent exact-reproduction claims for local substitute models",
            }
        )
        counter += 1
    return frame(rows, PAPER_RESULT_INVENTORY_COLUMNS)


def _paper_value_for(_inventory: pd.DataFrame, _metric_id: str) -> str:
    return ""


def _classification_for_local(local_type: str, same_model: bool) -> str:
    if local_type.startswith("local_llm"):
        return "adapted_local_model_extension"
    if same_model:
        return "reconstructed_paper_like"
    return "same_metric_different_cohort"


def _agreement(paper_value: str, same_model: bool, same_protocol: bool) -> str:
    if paper_value == "":
        return "paper_value_unverified"
    if not same_model or not same_protocol:
        return "not_directly_comparable"
    return "materially_different"


def build_paper_alignment_comparison(
    utility: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counter = 1
    utility_rows = utility[utility["evaluation_policy_id"].eq("adapted_end_to_end_operational_utility_invalid_as_failure")]
    for row in utility_rows.to_dict("records"):
        for metric_id, value_col in (("hr_at_5", "hr_at_5"), ("ndcg_at_5", "ndcg_at_5")):
            paper_value = _paper_value_for(inventory, metric_id)
            rows.append(
                {
                    "alignment_id": f"PAPERALIGN-{counter:05d}",
                    "local_result_type": "local_llm_operational_utility",
                    "local_model_or_baseline": row["model_family"],
                    "paper_model_or_baseline": "paper closed/API or larger LLM family",
                    "dataset": row["dataset"],
                    "task_type": row["task_type"],
                    "metric_id": metric_id,
                    "local_value": row[value_col],
                    "paper_value": paper_value,
                    "absolute_difference": "",
                    "relative_difference": "",
                    "same_model": "false",
                    "same_dataset_artifact": "false",
                    "same_candidate_protocol": "false",
                    "same_output_protocol": "false",
                    "same_metric_definition": "true",
                    "comparison_classification": "adapted_local_model_extension",
                    "agreement_classification": _agreement(paper_value, False, False),
                    "claim_allowed": "false",
                    "blocking_reason": "local model, dataset artifacts, prompt/output protocol, and paper numeric values are not exact",
                    "notes": "no exact numerical-replication claim is supported for Qwen2 or Mistral",
                }
            )
            counter += 1
    paired = baseline_metrics[
        baseline_metrics["comparison_scope"].eq("paired_llm_strict_valid_subset")
        & baseline_metrics["metric_id"].isin(["hr_at_5", "ndcg_at_5"])
        & baseline_metrics["metric_status"].eq("complete")
    ]
    for row in paired.to_dict("records"):
        baseline = row["baseline_id"]
        same_model = baseline in BASELINES and baseline != "bm25_like_e3b"
        paper_value = _paper_value_for(inventory, row["metric_id"])
        rows.append(
            {
                "alignment_id": f"PAPERALIGN-{counter:05d}",
                "local_result_type": "baseline_paired_strict_valid_subset",
                "local_model_or_baseline": baseline,
                "paper_model_or_baseline": baseline if same_model else "paper BM25 identity unresolved",
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "metric_id": row["metric_id"],
                "local_value": row["metric_value"],
                "paper_value": paper_value,
                "absolute_difference": "",
                "relative_difference": "",
                "same_model": "true" if same_model else "false",
                "same_dataset_artifact": "false",
                "same_candidate_protocol": "false",
                "same_output_protocol": "false",
                "same_metric_definition": "true",
                "comparison_classification": _classification_for_local("baseline", same_model),
                "agreement_classification": _agreement(paper_value, same_model, False),
                "claim_allowed": "false",
                "blocking_reason": "paired strict-valid subset differs from paper dataset/candidate/training protocol or lacks verified paper value",
                "notes": "baseline value is useful for local comparison, not exact paper replication",
            }
        )
        counter += 1
    if not rows:
        rows.append(
            {
                "alignment_id": "PAPERALIGN-00001",
                "local_result_type": "none",
                "local_model_or_baseline": "",
                "paper_model_or_baseline": "",
                "dataset": "",
                "task_type": "",
                "metric_id": "",
                "local_value": "",
                "paper_value": "",
                "absolute_difference": "",
                "relative_difference": "",
                "same_model": "false",
                "same_dataset_artifact": "false",
                "same_candidate_protocol": "false",
                "same_output_protocol": "false",
                "same_metric_definition": "false",
                "comparison_classification": "blocked_not_comparable",
                "agreement_classification": "paper_value_unverified",
                "claim_allowed": "false",
                "blocking_reason": "no local comparison rows available",
                "notes": "",
            }
        )
    return frame(rows, PAPER_ALIGNMENT_COLUMNS)


def reject_unsupported_exact_claim(row: Mapping[str, Any]) -> None:
    equality_fields = [
        "same_model",
        "same_dataset_artifact",
        "same_candidate_protocol",
        "same_output_protocol",
        "same_metric_definition",
    ]
    if clean_cell(row.get("claim_allowed", "")) == "true" and any(clean_cell(row.get(field, "")) != "true" for field in equality_fields):
        raise ValueError("exact claim is unsupported when any protocol equality field is false or unresolved")

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.canonical_local_llm_paper_alignment import (
    build_paper_alignment_comparison,
    build_paper_result_inventory,
    reject_unsupported_exact_claim,
)


def test_paper_inventory_does_not_fabricate_numeric_values():
    inventory = build_paper_result_inventory()
    assert not inventory.empty
    assert inventory["paper_value"].eq("").all()
    assert inventory["evidence_source"].ne("").all()


def test_local_llm_alignment_is_not_directly_comparable_to_paper_models():
    utility = pd.DataFrame(
        [
            {
                "evaluation_policy_id": "adapted_end_to_end_operational_utility_invalid_as_failure",
                "model_family": "Qwen2 7B Instruct",
                "dataset": "Amazon Beauty",
                "task_type": "ranking",
                "hr_at_5": "0.5",
                "ndcg_at_5": "0.4",
            }
        ]
    )
    baseline_metrics = pd.DataFrame(
        columns=[
            "comparison_scope",
            "metric_id",
            "metric_status",
            "baseline_id",
            "comparison_model_family",
            "dataset",
            "task_type",
            "metric_value",
        ]
    )
    alignment = build_paper_alignment_comparison(utility, baseline_metrics, build_paper_result_inventory())
    row = alignment[alignment["local_result_type"].eq("local_llm_operational_utility")].iloc[0]
    assert row["same_model"] == "false"
    assert row["agreement_classification"] == "paper_value_unverified"
    assert row["claim_allowed"] == "false"


def test_rejects_unsupported_exact_claim_when_protocol_field_false():
    with pytest.raises(ValueError, match="unsupported"):
        reject_unsupported_exact_claim(
            {
                "claim_allowed": "true",
                "same_model": "true",
                "same_dataset_artifact": "false",
                "same_candidate_protocol": "true",
                "same_output_protocol": "true",
                "same_metric_definition": "true",
            }
        )


def test_baseline_alignment_rows_are_reconstructed_or_unverified_not_exact_claims():
    utility = pd.DataFrame(
        [
            {
                "evaluation_policy_id": "adapted_end_to_end_operational_utility_invalid_as_failure",
                "model_family": "Mistral 7B Instruct",
                "dataset": "ML-1M",
                "task_type": "ranking",
                "hr_at_5": "0.4",
                "ndcg_at_5": "0.3",
            }
        ]
    )
    baseline_metrics = pd.DataFrame(
        [
            {
                "comparison_scope": "paired_llm_strict_valid_subset",
                "metric_id": "hr_at_5",
                "metric_status": "complete",
                "baseline_id": "mostpop_train_only",
                "comparison_model_family": "Mistral 7B Instruct",
                "dataset": "ML-1M",
                "task_type": "ranking",
                "metric_value": "0.2",
            }
        ]
    )
    alignment = build_paper_alignment_comparison(utility, baseline_metrics, build_paper_result_inventory())
    baseline = alignment[alignment["local_result_type"].eq("baseline_paired_strict_valid_subset")].iloc[0]
    assert baseline["comparison_classification"] == "reconstructed_paper_like"
    assert baseline["claim_allowed"] == "false"

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.canonical_local_llm_full_evaluation import (
    build_invalid_inventory,
    build_policy_contract,
    build_utility_metrics,
    reparse_raw_responses,
)


def _manifest(task: str = "ranking") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "request_id": "REQ-1",
                "request_order": "1",
                "chunk_id": "CHUNK-1",
                "model_family": "Qwen2 7B Instruct",
                "model_tag": "qwen2:7b-instruct",
                "dataset": "Amazon Beauty",
                "task_type": task,
                "candidate_set_id": "cand",
                "user_id": "u1",
                "max_output_tokens": "256",
            }
        ]
    )


def _execution() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "request_id": "REQ-1",
                "backend_success": "true",
                "done_reason": "stop",
                "eval_count": "8",
            }
        ]
    )


def test_reparse_agrees_with_stored_parse_and_rejects_mismatch():
    manifest = _manifest()
    responses = [
        {
            "request_id": "REQ-1",
            "raw_response_text": '{"ranked_candidate_indices":[1,2,3,4,5]}',
            "raw_response_sha256": "f9e1a8bcb443b4f775c1b810773f506b27c3ce8920ec9212c7bb3132346a9900",
            "done_reason": "stop",
            "generated_token_count": "8",
        }
    ]
    stored = pd.DataFrame(
        [
            {
                "request_id": "REQ-1",
                "raw_response_sha256": "f9e1a8bcb443b4f775c1b810773f506b27c3ce8920ec9212c7bb3132346a9900",
                "json_parse_valid": "true",
                "schema_valid": "true",
                "output_count_valid": "true",
                "index_type_valid": "true",
                "index_range_valid": "true",
                "uniqueness_valid": "true",
                "candidate_validity": "true",
                "catalog_validity": "true",
                "duplicate_count": "0",
                "unknown_index_count": "0",
                "hallucination_count": "0",
                "strict_parse_valid": "true",
                "output_length": "5",
                "output_truncation_detected": "false",
                "primary_failure_reason": "",
            }
        ]
    )
    integrity = reparse_raw_responses(manifest, _execution(), stored, responses)
    assert integrity.loc[0, "stored_reparse_agreement"] == "true"
    stored.loc[0, "duplicate_count"] = "1"
    integrity = reparse_raw_responses(manifest, _execution(), stored, responses)
    assert integrity.loc[0, "stored_reparse_agreement"] == "false"


def test_invalid_inventory_detects_duplicate_without_repair_or_raw_text_leakage():
    manifest = _manifest("reranking")
    responses = [
        {
            "request_id": "REQ-1",
            "raw_response_text": '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,1]}',
            "raw_response_sha256": "hash",
            "done_reason": "stop",
            "generated_token_count": "12",
        }
    ]
    stored = pd.DataFrame(
        [
            {
                "request_id": "REQ-1",
                "raw_response_sha256": "hash",
                "json_parse_valid": "true",
                "schema_valid": "true",
                "output_count_valid": "true",
                "index_type_valid": "true",
                "index_range_valid": "true",
                "uniqueness_valid": "false",
                "candidate_validity": "false",
                "catalog_validity": "false",
                "duplicate_count": "1",
                "unknown_index_count": "0",
                "hallucination_count": "0",
                "strict_parse_valid": "false",
                "output_length": "10",
                "output_truncation_detected": "false",
                "primary_failure_reason": "duplicate_indices",
            }
        ]
    )
    integrity = reparse_raw_responses(manifest, _execution(), stored, responses)
    inventory = build_invalid_inventory(integrity, _execution(), responses)
    assert inventory.loc[0, "duplicated_indices"] == "[1]"
    assert inventory.loc[0, "primary_failure_reason"] == "duplicate_indices"
    assert "ranked_candidate_indices" not in inventory.to_csv(index=False)
    assert inventory.loc[0, "failure_category_count"] == "1"


def test_utility_policies_keep_valid_only_conditional_and_invalid_operational_zero():
    rankings = pd.DataFrame(
        [
            {
                "model_family": "Qwen2 7B Instruct",
                "model_tag": "qwen2:7b-instruct",
                "dataset": "Amazon Beauty",
                "task_type": "ranking",
                "strict_parse_valid": "true",
                "positive_in_top_5": "true",
                "ndcg_at_5": "1",
                "positive_rank_zero_based": "0",
            },
            {
                "model_family": "Qwen2 7B Instruct",
                "model_tag": "qwen2:7b-instruct",
                "dataset": "Amazon Beauty",
                "task_type": "ranking",
                "strict_parse_valid": "false",
                "positive_in_top_5": "",
                "ndcg_at_5": "",
                "positive_rank_zero_based": "",
            },
        ]
    )
    for model in ["Qwen2 7B Instruct", "Mistral 7B Instruct"]:
        for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]:
            for task in ["ranking", "reranking"]:
                if not ((model, dataset, task) == ("Qwen2 7B Instruct", "Amazon Beauty", "ranking")):
                    rankings = pd.concat(
                        [
                            rankings,
                            pd.DataFrame(
                                [
                                    {
                                        "model_family": model,
                                        "model_tag": "tag",
                                        "dataset": dataset,
                                        "task_type": task,
                                        "strict_parse_valid": "true",
                                        "positive_in_top_5": "false",
                                        "ndcg_at_5": "0",
                                        "positive_rank_zero_based": "10",
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
    metrics = build_utility_metrics(rankings)
    conditional = metrics[
        metrics["model_family"].eq("Qwen2 7B Instruct")
        & metrics["dataset"].eq("Amazon Beauty")
        & metrics["task_type"].eq("ranking")
        & metrics["evaluation_policy_id"].eq("conditional_strict_valid_output_utility")
    ].iloc[0]
    operational = metrics[
        metrics["model_family"].eq("Qwen2 7B Instruct")
        & metrics["dataset"].eq("Amazon Beauty")
        & metrics["task_type"].eq("ranking")
        & metrics["evaluation_policy_id"].eq("adapted_end_to_end_operational_utility_invalid_as_failure")
    ].iloc[0]
    assert conditional["hr_at_5"] == "1"
    assert operational["hr_at_5"] == "0.5"
    assert operational["mean_positive_rank"] == ""


def test_policy_contract_prevents_unconditional_valid_only_label():
    policy = build_policy_contract()
    row = policy[policy["policy_id"].eq("conditional_strict_valid_output_utility")].iloc[0]
    assert row["primary_or_diagnostic"] == "secondary_diagnostic"
    assert row["denominator_policy"] == "strict-valid outputs only"
    assert row["allowed_for_paper_comparison"] == "false"

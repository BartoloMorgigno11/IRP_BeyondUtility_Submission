from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.cost_metrics import (
    aggregate_cost_metrics_by_model,
    compute_total_tokens,
    estimate_cost_per_1000_users,
    estimate_query_cost,
    summarize_latency_by_model,
    summarize_tokens_by_model,
)


def test_total_token_computation_from_prompt_and_completion_tokens() -> None:
    assert compute_total_tokens(prompt_tokens=100, completion_tokens=25) == 125


def test_total_tokens_already_provided_are_supported() -> None:
    assert compute_total_tokens(
        prompt_tokens=100,
        completion_tokens=25,
        total_tokens=200,
    ) == 200


def test_cost_per_query_with_synthetic_prices() -> None:
    price_record = {
        "prompt_per_1k_tokens": 0.01,
        "completion_per_1k_tokens": 0.03,
    }

    value = estimate_query_cost(
        prompt_tokens=1000,
        completion_tokens=500,
        price_record=price_record,
    )

    assert value == pytest.approx(0.025)


def test_cost_per_1000_users_with_synthetic_prices() -> None:
    price_record = {
        "prompt_per_1k_tokens": 0.01,
        "completion_per_1k_tokens": 0.03,
    }

    value = estimate_cost_per_1000_users(
        prompt_tokens=1000,
        completion_tokens=500,
        price_record=price_record,
    )

    assert value == pytest.approx(25.0)


def test_missing_prices_are_handled_gracefully() -> None:
    assert estimate_query_cost(
        prompt_tokens=1000,
        completion_tokens=500,
        price_record={"prompt_per_1k_tokens": None, "completion_per_1k_tokens": None},
    ) is None


def test_missing_token_counts_are_handled_gracefully() -> None:
    records = pd.DataFrame(
        {
            "model_name": ["m1", "m1"],
            "prompt_tokens": [None, ""],
            "completion_tokens": [None, ""],
        }
    )

    summary = summarize_tokens_by_model(records)

    assert compute_total_tokens(prompt_tokens=100, completion_tokens=None) is None
    assert estimate_query_cost(
        prompt_tokens=100,
        completion_tokens=None,
        price_record={"prompt_per_1k_tokens": 0.01, "completion_per_1k_tokens": 0.02},
    ) is None
    assert summary.loc[0, "token_status"] == "unavailable"
    assert pd.isna(summary.loc[0, "total_tokens"])


def test_missing_latency_is_handled_gracefully() -> None:
    records = pd.DataFrame({"model_name": ["m1", "m1"]})

    summary = summarize_latency_by_model(records)

    assert summary.loc[0, "latency_status"] == "unavailable"
    assert pd.isna(summary.loc[0, "mean_latency_seconds"])


def test_token_and_latency_aggregation_by_model() -> None:
    records = pd.DataFrame(
        {
            "model_name": ["m1", "m1", "m2"],
            "prompt_tokens": [100, 200, None],
            "completion_tokens": [50, 100, None],
            "total_tokens": [None, None, 1000],
            "latency_seconds": [1.0, 2.0, None],
        }
    )

    token_summary = summarize_tokens_by_model(records)
    latency_summary = summarize_latency_by_model(records)

    m1_tokens = token_summary.loc[token_summary["model_name"] == "m1"].iloc[0]
    m2_tokens = token_summary.loc[token_summary["model_name"] == "m2"].iloc[0]
    m1_latency = latency_summary.loc[latency_summary["model_name"] == "m1"].iloc[0]
    m2_latency = latency_summary.loc[latency_summary["model_name"] == "m2"].iloc[0]

    assert m1_tokens["prompt_tokens"] == 300
    assert m1_tokens["completion_tokens"] == 150
    assert m1_tokens["total_tokens"] == 450
    assert m1_tokens["average_total_tokens_per_query"] == pytest.approx(225.0)
    assert m1_tokens["token_status"] == "available"
    assert m2_tokens["total_tokens"] == 1000
    assert m2_tokens["token_status"] == "available"
    assert m1_latency["mean_latency_seconds"] == pytest.approx(1.5)
    assert m1_latency["latency_status"] == "available"
    assert m2_latency["latency_status"] == "unavailable"


def test_cost_aggregation_by_model_with_synthetic_prices() -> None:
    records = pd.DataFrame(
        {
            "model_name": ["m1", "m1", "m2"],
            "prompt_tokens": [100, 200, None],
            "completion_tokens": [50, 100, None],
            "total_tokens": [None, None, 1000],
        }
    )
    pricing = {
        "models": {
            "m1": {
                "prompt_per_1k_tokens": 0.01,
                "completion_per_1k_tokens": 0.02,
            },
            "m2": {
                "total_per_1k_tokens": 0.005,
            },
        }
    }

    summary = aggregate_cost_metrics_by_model(records, pricing)
    m1 = summary.loc[summary["model_name"] == "m1"].iloc[0]
    m2 = summary.loc[summary["model_name"] == "m2"].iloc[0]

    assert m1["estimated_cost_total"] == pytest.approx(0.006)
    assert m1["estimated_cost_per_query"] == pytest.approx(0.003)
    assert m1["estimated_cost_per_1000_users"] == pytest.approx(3.0)
    assert m1["cost_status"] == "available"
    assert m2["estimated_cost_total"] == pytest.approx(0.005)
    assert m2["estimated_cost_per_1000_users"] == pytest.approx(5.0)


def test_missing_prices_make_cost_unavailable_but_do_not_block_token_summary() -> None:
    records = pd.DataFrame(
        {
            "model_name": ["m1"],
            "prompt_tokens": [100],
            "completion_tokens": [50],
        }
    )

    cost_summary = aggregate_cost_metrics_by_model(records, pricing=None)
    token_summary = summarize_tokens_by_model(records)

    assert cost_summary.loc[0, "cost_status"] == "unavailable"
    assert pd.isna(cost_summary.loc[0, "estimated_cost_per_query"])
    assert token_summary.loc[0, "total_tokens"] == 150
    assert token_summary.loc[0, "token_status"] == "available"


def test_cost_metrics_have_no_external_api_dependency() -> None:
    import src.evaluation.cost_metrics as cost_metrics

    assert "requests" not in cost_metrics.__dict__
    assert "openai" not in cost_metrics.__dict__
    assert estimate_query_cost(
        total_tokens=1000,
        price_record={"total_per_1k_tokens": 0.01},
    ) == pytest.approx(0.01)

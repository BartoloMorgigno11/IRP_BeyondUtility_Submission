from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.extension import f1bx_execution_stage1 as f1bx


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def outputs() -> dict[str, object]:
    return f1bx.generate_f1bx(ROOT)


def test_f1bx_executes_all_deterministic_baselines(outputs: dict[str, object]) -> None:
    execution = outputs["baseline_execution"]
    recommendations = outputs["baseline_recommendations"]

    assert len(execution) == 15
    assert execution["status"].eq("COMPUTED").all()
    assert set(execution["dataset"]) == set(f1bx.ACTIVE_DATASETS)
    assert set(execution["method"]) == set(f1bx.BASELINE_METHODS)
    assert execution["recommendation_output_hash"].astype(str).str.len().eq(64).all()
    assert len(recommendations) == int((execution["cohort_N"].astype(int) * 20).sum())


def test_f1bx_baseline_metrics_are_materialized(outputs: dict[str, object]) -> None:
    utility = outputs["baseline_utility_results"]
    beyond = outputs["baseline_beyond_utility_results"]

    assert len(utility) == 15 * len(f1bx.UTILITY_METRICS)
    assert utility["status"].eq("COMPUTED").all()
    assert set(utility["metric"]) == set(f1bx.UTILITY_METRICS)
    assert len(beyond) == 15 * len(f1bx.BEYOND_METRICS)
    assert set(beyond["metric"]) == set(f1bx.BEYOND_METRICS)
    assert beyond.loc[beyond["metric"].isin(["popreo_at_5", "dpd_at_5"]), "status"].isin({"COMPUTED", "DEGENERATE_PROTOCOL_INPUT"}).all()


def test_f1bx_prevents_target_and_test_leakage(outputs: dict[str, object]) -> None:
    validation = outputs["validation"]
    tuning_runs = outputs["foodcom_tuning_runs"]
    integrity = outputs["integrity"]

    assert validation["status"].eq("PASS").all()
    assert integrity["target_leakage_count"] == 0
    assert tuning_runs["final_test_metric_computed"].eq("false").all()
    assert tuning_runs["training_executed"].eq("false").all()


def test_f1bx_foodcom_tuning_is_explicitly_blocked_without_winners(outputs: dict[str, object]) -> None:
    tuning = outputs["foodcom_tuning_runs"]
    selected = outputs["foodcom_selected_configs"]
    manifest = outputs["execution_manifest"]

    assert len(tuning) == 40
    assert tuning["seed"].astype(int).eq(42).all()
    assert tuning["terminal"].eq("true").all()
    assert tuning["status"].eq("BLOCKED_NOT_EXECUTED_MISSING_VALIDATION_CANDIDATE_CONTRACT").all()
    assert len(selected) == 4
    assert set(selected["model"]) == set(f1bx.TRAINABLE_MODELS)
    assert selected["selected"].eq("false").all()
    assert manifest["decision_token"] == f1bx.DECISION_BLOCKED
    assert manifest["foodcom_winners_selected"] == 0


def test_f1bx_keeps_forbidden_counters_zero(outputs: dict[str, object]) -> None:
    manifest = outputs["execution_manifest"]
    integrity = outputs["integrity"]

    assert manifest["llm_inference_calls_executed"] == 0
    assert manifest["ollama_recommendation_calls_executed"] == 0
    assert manifest["external_api_calls_executed"] == 0
    assert manifest["downloads_executed"] == 0
    assert manifest["network_calls_executed"] == 0
    assert manifest["final_training_runs_executed"] == 0
    assert manifest["original_four_final_runs_executed"] == 0
    assert manifest["foodcom_final_five_seed_runs_executed"] == 0
    assert integrity["active_datasets"] == list(f1bx.ACTIVE_DATASETS)
    assert integrity["active_video_games_rows"] == 0
    assert integrity["f0_budget_unchanged"] is True


def test_f1bx_foodcom_bm25_uses_tags_only(outputs: dict[str, object]) -> None:
    config = f1bx.load_config(ROOT, f1bx.CONFIG_PATH)
    foodcom = next(row for row in config["datasets"] if row["name"] == "foodcom")
    report = outputs["report"]

    assert foodcom["bm25_text_source"] == "RAW_recipes.tags"
    assert config["baselines"]["BM25-like/content"]["foodcom_text_policy"] == "RAW_recipes.tags"
    assert "RAW_recipes.tags" in report

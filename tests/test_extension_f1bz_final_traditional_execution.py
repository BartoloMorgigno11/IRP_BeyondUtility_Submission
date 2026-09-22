from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "tables" / name, dtype=str, keep_default_na=False)


def manifest(name: str) -> dict:
    return json.loads((ROOT / "results" / "manifests" / name).read_text(encoding="utf-8"))


def test_f1bz_100_final_runs_complete() -> None:
    runs = table("f1bz_final_run_manifest.csv")

    assert len(runs) == 100
    assert runs["training_status"].eq("COMPLETED").all()
    assert runs["retry_count"].astype(int).sum() == 0
    assert runs.groupby(["dataset", "method"]).size().eq(5).all()
    assert set(runs["seed"].astype(int)) == {42, 43, 44, 45, 46}


def test_f1bz_frozen_config_identity_and_foodcom_winners() -> None:
    runs = table("f1bz_final_run_manifest.csv")
    winners = table("f1by_foodcom_selected_configs.csv").set_index("model")

    original_four = runs.loc[~runs["dataset"].eq("foodcom")]
    assert original_four["classification"].eq("PAPER_HYPERPARAMETER_ALIGNED_LOCAL_IMPLEMENTATION").all()
    assert original_four["paper_seed_identity_equivalent"].eq("false").all()

    foodcom = runs.loc[runs["dataset"].eq("foodcom")]
    assert foodcom["classification"].eq("ORIGINAL_EXTENSION").all()
    for _, row in foodcom.iterrows():
        winner = winners.loc[row["method"]]
        assert row["config_id"] == winner["selected_config_id"]
        assert row["config_hash"] == winner["config_hash"]
    assert foodcom["final_test_targets_used_in_tuning"].eq("false").all()


def test_f1bz_metric_and_traditional_matrix_completion() -> None:
    utility = table("f1bz_per_seed_utility_results.csv")
    beyond = table("f1bz_per_seed_beyond_utility_results.csv")
    aggregate_utility = table("f1bz_aggregate_utility_results.csv")
    aggregate_beyond = table("f1bz_aggregate_beyond_utility_results.csv")
    traditional = table("f1bz_traditional_35row_completeness.csv")
    training_curves = table("f1bz_training_curves.csv")

    assert len(utility) == 300
    assert utility["status"].eq("COMPUTED").all()
    assert len(beyond) == 1400
    assert set(beyond["status"]).issubset({"COMPUTED", "DEGENERATE_PROTOCOL_INPUT", "NOT_APPLICABLE"})
    assert len(aggregate_utility) == 60
    assert len(aggregate_beyond) == 280
    assert len(traditional) == 35
    assert traditional["final_status"].eq("COMPLETED").all()
    assert len(training_curves) >= 100
    assert training_curves["run_id"].nunique() == 100
    assert training_curves["curve_status"].isin(
        {
            "OBSERVED_DURING_CURRENT_EXECUTION",
            "FINAL_EPOCH_FROM_RUN_MANIFEST_AFTER_RESUME",
            "RUN_COMPLETION_FROM_RECOVERED_ARTIFACTS",
        }
    ).all()


def test_f1bz_reranking_readiness_and_candidate_recall() -> None:
    first_stage = table("f1bz_first_stage_rankings.csv")
    pools = table("f1bz_reranking_pools.csv")
    recall = table("f1bz_candidate_recall.csv")
    readiness = table("f1bz_reranking_readiness.csv")

    assert len(first_stage) == 97640
    assert len(pools) == 4882
    assert pools["ranking_size"].astype(int).eq(20).all()
    assert len(recall) == 5
    assert recall["status"].eq("COMPUTED").all()
    assert len(readiness) == 30
    assert readiness["readiness_status"].eq("READY_TO_INFER").all()
    assert readiness["future_requests_unlocked"].astype(int).sum() == 29292


def test_f1bz_no_llm_and_integrity_manifest() -> None:
    payload = manifest("f1bz_final_traditional_execution.json")
    no_llm = manifest("f1bz_no_llm_inference.json")

    assert payload["decision_token"] == "F1BZ_FINAL_TRADITIONAL_EXECUTION_VALIDATED"
    assert payload["completed_runs"] == 100
    assert payload["failed_runs"] == 0
    assert payload["integrity"]["active_video_games_rows"] == 0
    assert payload["integrity"]["f0_budget_unchanged"] is True
    assert payload["integrity"]["global_metric_blockers"] == 0
    assert no_llm["llm_inference_calls_executed"] == 0
    assert no_llm["ollama_recommendation_calls_executed"] == 0
    assert no_llm["external_api_calls_executed"] == 0
    assert no_llm["network_calls_executed"] == 0
    assert no_llm["downloads_executed"] == 0

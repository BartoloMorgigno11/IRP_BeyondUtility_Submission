from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "tables" / name, dtype=str, keep_default_na=False)


def read_manifest(name: str) -> dict:
    import json

    return json.loads((ROOT / "results" / "manifests" / name).read_text(encoding="utf-8"))


def test_f1by_executes_40_frozen_configs_only() -> None:
    execution = read_table("f1by_foodcom_tuning_execution.csv")

    assert len(execution) == 40
    assert execution["dataset"].eq("foodcom").all()
    assert execution["seed"].astype(int).eq(42).all()
    assert execution["training_status"].eq("COMPLETED").all()
    assert execution.groupby("model").size().to_dict() == {"BPRMF": 16, "GRU4Rec": 12, "LightGCN": 4, "SASRec": 8}


def test_f1by_metrics_are_validation_only() -> None:
    metrics = read_table("f1by_foodcom_tuning_metrics.csv")
    execution = read_table("f1by_foodcom_tuning_execution.csv")

    assert set(metrics["metric"]) == {"validation_hr_at_5", "validation_ndcg_at_5"}
    assert len(metrics) == 80
    assert metrics["final_test_metric"].eq("false").all()
    assert execution["final_test_metrics_computed"].eq("false").all()
    assert execution["final_test_candidates_scored"].eq("false").all()
    assert execution["final_test_targets_used"].eq("false").all()


def test_f1by_selects_exactly_four_deterministic_winners() -> None:
    ranking = read_table("f1by_foodcom_config_ranking.csv")
    selected = read_table("f1by_foodcom_selected_configs.csv")

    assert len(ranking) == 40
    assert len(selected) == 4
    assert selected["selected"].eq("true").all()
    assert selected["final_test_metrics_available_at_selection"].eq("false").all()
    for model, group in ranking.groupby("model"):
        ordered = group.sort_values(["selection_rank"], key=lambda column: column.astype(int))
        assert selected.loc[selected["model"].eq(model), "selected_config_id"].iloc[0] == ordered.iloc[0]["config_id"]


def test_f1by_leakage_and_preservation_audits_pass() -> None:
    leakage = read_table("f1by_test_leakage_audit.csv")
    validation = read_table("f1by_validation.csv")
    manifest = read_manifest("f1by_foodcom_tuning_execution.json")

    assert leakage["status"].eq("PASS").all()
    assert validation["status"].eq("PASS").all()
    assert manifest["integrity"]["baseline_rows"] == 15
    assert manifest["integrity"]["baseline_recommendation_rows"] == 292920
    assert manifest["integrity"]["baseline_hashes_before_after_unchanged"] is True
    assert manifest["integrity"]["baseline_hashes_match_f1bx_manifest"] is True
    assert manifest["llm_inference_calls_executed"] == 0
    assert manifest["network_calls_executed"] == 0
    assert manifest["downloads_executed"] == 0
    assert manifest["original_four_final_runs_executed"] == 0
    assert manifest["foodcom_final_five_seed_runs_executed"] == 0


def test_f1by_integrity_constraints_hold() -> None:
    manifest = read_manifest("f1by_foodcom_winner_selection.json")

    assert manifest["decision_token"] == "F1BY_FOODCOM_WINNERS_SELECTED"
    assert manifest["integrity"]["active_datasets"] == ["beauty", "sports", "ml1m", "lastfm", "foodcom"]
    assert manifest["integrity"]["active_video_games_rows"] == 0
    assert manifest["integrity"]["f0_budget_unchanged"] is True
    assert manifest["integrity"]["global_metric_blockers"] == 0

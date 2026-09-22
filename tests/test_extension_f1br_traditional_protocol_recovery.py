from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.extension import f1br_traditional_protocol_recovery as f1br


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def outputs() -> dict[str, object]:
    return f1br.generate_f1br(ROOT)


def test_f1br_validation_resolves_protocol_without_training(outputs: dict[str, object]) -> None:
    validation = outputs["validation"]
    manifest = outputs["manifest"]

    assert validation["status"].eq("PASS").all()
    assert manifest["decision_token"] == f1br.DECISION_TOKEN
    assert manifest["traditional_training_runs_executed"] == 0
    assert manifest["deterministic_baseline_runs_executed"] == 0
    assert manifest["llm_inference_calls_executed"] == 0
    assert manifest["downloads_executed"] == 0
    assert manifest["network_calls_executed"] == 0


def test_rechorus_gitlink_and_local_default_limits_are_explicit(outputs: dict[str, object]) -> None:
    lineage = outputs["rechorus_lineage"]

    assert lineage.loc[lineage["aspect"].eq("exact_rechorus_gitlink"), "recovered_value"].iloc[0] == f1br.RECHORUS_GITLINK
    assert "NOT_RECOVERED_LOCALLY" in set(lineage["status"])
    assert lineage.loc[lineage["aspect"].eq("local_project_adapter_smoke"), "status"].iloc[0] == "PASS"


def test_paper_hyperparameters_and_seed_policy_are_frozen(outputs: dict[str, object]) -> None:
    hyper = outputs["paper_hyperparameters"]
    seeds = outputs["seed_policy"]

    assert len(hyper) == 20
    assert set(hyper["method"]) == {"BM25", *f1br.TRAINABLE_MODELS}
    assert hyper.loc[(hyper["dataset"].eq("beauty")) & (hyper["method"].eq("BPRMF")), "embedding"].iloc[0] == "64"
    assert hyper.loc[(hyper["dataset"].eq("lastfm")) & (hyper["method"].eq("LightGCN")), "layers"].iloc[0] == "4"
    assert seeds["final_seeds"].eq("[42,43,44,45,46]").all()
    assert seeds["evidence_classification"].eq("LOCAL_REPRODUCIBLE_FIVE_SEED_POLICY").all()
    assert seeds["paper_seed_identity_equivalent"].eq("false").all()
    assert seeds["claimed_as_author_seed_identity"].eq("false").all()


def test_foodcom_search_is_validation_only_and_counted_separately(outputs: dict[str, object]) -> None:
    search = outputs["foodcom_search_space"]
    tuning = outputs["foodcom_tuning_contract"]
    budget = {row["budget_item"]: int(row["run_count"]) for row in outputs["training_run_budget"].to_dict("records")}

    assert len(search) == 40
    assert search["uses_final_test_metrics"].eq("false").all()
    assert tuning["test_metric_tuning_allowed"].eq("false").all()
    assert budget["ORIGINAL_FOUR_FINAL_RUNS"] == 80
    assert budget["FOODCOM_FINAL_RUNS"] == 20
    assert budget["FOODCOM_TUNING_RUNS"] == 40
    assert budget["TOTAL_TRAINING_RUNS"] == 140


def test_readiness_targets_are_met_and_video_games_absent(outputs: dict[str, object]) -> None:
    baseline = outputs["baseline_execution_readiness"]
    trainable = outputs["trainable_execution_readiness"]
    reranking = outputs["reranking_source_protocol"]
    integrity = outputs["integrity"]

    assert len(baseline) == 15
    assert baseline["status"].eq("READY_TO_EXECUTE").all()
    assert len(trainable) == 20
    assert trainable["status"].eq("READY_TO_TRAIN").all()
    assert len(reranking) == 5
    assert reranking["status"].eq("ADAPTED_POLICY_FROZEN").all()
    assert integrity["active_datasets"] == list(f1br.ACTIVE_DATASETS)
    assert integrity["active_video_games_rows"] == 0


def test_f1b_planning_updates_replace_ambiguity_statuses(outputs: dict[str, object]) -> None:
    updates = outputs["f1b_updates"]
    reuse = updates["reuse_audit"]
    plan = updates["training_plan"]
    readiness = updates["result_cell_readiness"]

    assert reuse.loc[reuse["method"].isin(f1br.BASELINE_MODELS), "final_f1b_status"].eq("READY_TO_EXECUTE").all()
    assert reuse.loc[reuse["method"].isin(f1br.TRAINABLE_MODELS), "final_f1b_status"].eq("READY_TO_TRAIN").all()
    assert plan["resolved_seed"].astype(int).isin(f1br.FINAL_SEEDS).all()
    assert plan["training_plan_status"].eq("READY_TO_TRAIN").all()
    assert not readiness["f1b_status"].eq("BLOCKED_EXECUTION").any()
    assert set(readiness["f1b_status"]).issubset({"READY_FROM_NEW_RAW", "READY_AFTER_TRAINING_RAW", "DEGENERATE_PROTOCOL_INPUT"})

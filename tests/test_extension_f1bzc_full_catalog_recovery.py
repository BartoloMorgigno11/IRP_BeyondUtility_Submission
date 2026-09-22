from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "tables" / name, dtype=str, keep_default_na=False)


def manifest(name: str) -> dict:
    return json.loads((ROOT / "results" / "manifests" / name).read_text(encoding="utf-8"))


def test_f1bzc_manifest_blocks_instead_of_fabricating_full_catalog_recovery() -> None:
    payload = manifest("f1bzc_full_catalog_recovery.json")
    no_llm = manifest("f1bzc_no_llm_no_training.json")

    assert payload["decision_token"] == "F1BZC_RERANKING_SOURCE_BLOCKED"
    assert payload["ready_full_catalog_cells"] == 4
    assert payload["blocked_full_catalog_cells"] == 16
    assert payload["single_model_dominated_pools"] == 1
    assert payload["future_requests_unlocked"] == 0
    assert payload["frozen_max_future_requests"] == 29292
    assert payload["f1bz_evidence_hashes_unchanged"] is True
    assert no_llm["training_runs_executed_in_f1bzc"] == 0
    assert no_llm["checkpoint_parameter_updates"] == 0
    assert no_llm["llm_inference_calls_executed"] == 0
    assert no_llm["network_calls_executed"] == 0


def test_f1bzc_source_seed_policy_is_outcome_independent() -> None:
    policy = table("f1bzc_source_seed_policy.csv")

    assert len(policy) == 20
    assert policy["uses_final_test_metric"].eq("false").all()
    assert policy["selected_seed"].eq("ALL_FIVE_SEED_MEAN").all()
    assert policy["fallback_used"].eq("false").all()
    assert set(policy["model"]) == {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}


def test_f1bzc_adapter_audit_recovered_only_scientifically_ready_cells() -> None:
    adapter = table("f1bzc_full_catalog_adapter_audit.csv")
    execution = table("f1bzc_full_catalog_execution.csv")

    assert len(adapter) == 20
    assert adapter["checkpoint_hash_identity"].eq("true").all()
    assert adapter["force_positive"].eq("false").all()
    assert adapter["target_membership_used_for_ranking"].eq("false").all()
    assert adapter.loc[adapter["dataset"].eq("lastfm"), "adapter_status"].eq("FULL_CATALOG_REGENERATION_READY").all()
    assert adapter.loc[~adapter["dataset"].eq("lastfm"), "adapter_status"].eq(
        "BLOCKED_SCIENTIFIC_ITEM_UNIVERSE_COVERAGE"
    ).all()
    assert adapter.loc[adapter["dataset"].eq("lastfm"), "candidate20_overlap_smoke_status"].eq("PASS").all()
    assert adapter.loc[adapter["dataset"].eq("lastfm"), "deterministic_repeat_smoke_status"].eq("PASS").all()

    assert len(execution) == 20
    assert execution.loc[execution["dataset"].eq("lastfm"), "status"].eq("CORRECTED_F1BZC").all()
    assert execution.loc[execution["dataset"].eq("lastfm"), "catalog_size"].astype(int).gt(20).all()
    assert execution.loc[execution["dataset"].eq("lastfm"), "candidate_recall_tautological"].eq("false").all()
    assert execution.loc[~execution["dataset"].eq("lastfm"), "status"].eq(
        "BLOCKED_SCIENTIFIC_ITEM_UNIVERSE_COVERAGE"
    ).all()


def test_f1bzc_full_catalog_predictions_are_not_candidate20_or_target_forced() -> None:
    execution = table("f1bzc_full_catalog_execution.csv")
    for path in execution.loc[execution["dataset"].eq("lastfm"), "prediction_path"]:
        predictions = pd.read_csv(ROOT / path, dtype=str, keep_default_na=False)
        assert len(predictions) == 20000
        assert predictions["eligible_for_recommendation"].eq("true").all()
        assert predictions["was_seen_in_train"].eq("false").all()
        assert predictions["force_positive"].eq("false").all()
        assert predictions["ranker_uses_positive_signal"].eq("false").all()
        assert predictions["catalog_size"].astype(int).min() > 20
        assert predictions.groupby("user_id").size().eq(20).all()


def test_f1bzc_pool_contribution_defect_is_explicit() -> None:
    contributions = table("f1bzc_reranking_source_contributions.csv")
    pool_validation = table("f1bzc_reranking_pool_validation.csv")

    assert len(contributions) == 4000
    assert contributions["single_model_filled_pool"].eq("true").all()
    totals = contributions.groupby("source_model")["contribution_count"].apply(lambda s: s.astype(int).sum()).to_dict()
    assert totals["BPRMF"] == 20000
    assert totals["GRU4Rec"] == 0
    assert totals["SASRec"] == 0
    assert totals["LightGCN"] == 0
    assert "BLOCKED" in set(pool_validation["status"])
    assert pool_validation.loc[pool_validation["check_name"].eq("multi_model_contribution"), "status"].iloc[0] == "BLOCKED"


def test_f1bzc_candidate_recall_and_readiness_are_not_authoritative() -> None:
    recall = table("f1bzc_candidate_recall.csv")
    readiness = table("f1bzc_reranking_readiness.csv")

    lastfm = recall.loc[recall["dataset"].eq("lastfm")].iloc[0]
    assert lastfm["candidate_recall_tautological"] == "false"
    assert lastfm["authoritative"] == "false"
    assert lastfm["status"] == "BLOCKED_ADAPTED_POOL_SINGLE_MODEL_DOMINANCE"
    assert recall.loc[~recall["dataset"].eq("lastfm"), "status"].eq(
        "BLOCKED_SCIENTIFIC_ITEM_UNIVERSE_COVERAGE"
    ).all()

    assert len(readiness) == 30
    assert readiness["future_requests_unlocked"].astype(int).sum() == 0
    assert "READY_TO_INFER" not in set(readiness["readiness_status"])


def test_f1bzc_foodcom_diagnostics_and_superseded_registry() -> None:
    diagnostics = table("f1bzc_foodcom_warm_cold_diagnostics.csv")
    superseded = table("f1bzc_superseded_artifact_registry.csv")
    validity = table("f1bzc_result_validity_classification.csv")
    validation = table("f1bzc_validation.csv")

    assert {"ALL_USERS", "WARM_USERS", "TRUE_ZERO_HISTORY_USERS"}.issubset(set(diagnostics["scope"]))
    cold = diagnostics.loc[
        (diagnostics["scope"].eq("TRUE_ZERO_HISTORY_USERS"))
        & diagnostics["metric_family"].eq("full_catalog_candidate_recall")
    ].iloc[0]
    assert cold["user_N"] == "727"
    assert cold["status"] == "BLOCKED_SCIENTIFIC_ITEM_UNIVERSE_COVERAGE"
    assert superseded["superseded_status"].eq("SUPERSEDED_RERANKING_SOURCE_DEFECT").all()
    assert superseded["authoritative_for_downstream"].eq("false").all()
    assert {"VALID_AS_IS", "SUPERSEDED_RERANKING_SOURCE_DEFECT", "CORRECTED_F1BZC"}.issubset(
        set(validity["classification"])
    )
    assert validation["status"].eq("PASS").all()

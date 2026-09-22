from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.extension.f1bzq_authoritative_reranking_pool_materialization import (
    DECISION_VALIDATED,
    POOL_STATUS_COMPUTED,
    SOURCE_POLICY,
)
from src.extension.f1bzp_reranking_protocol_resolution import ADAPTED_ALGORITHM, ADAPTED_POLICY


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_USERS = {"beauty": 990, "sports": 892, "ml1m": 1000, "lastfm": 1000, "foodcom": 1000}
EXPECTED_MODELS = {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "tables" / name, dtype=str, keep_default_na=False)


def manifest(name: str) -> dict:
    return json.loads((ROOT / "results" / "manifests" / name).read_text(encoding="utf-8"))


def test_f1bzq_manifests_close_without_training_or_llm() -> None:
    pool = manifest("f1bzq_authoritative_pool_manifest.json")
    source = manifest("f1bzq_source_ranking_manifest.json")
    no_training = manifest("f1bzq_no_training_no_llm.json")

    assert pool["decision_token"] == DECISION_VALIDATED
    assert source["decision_token"] == DECISION_VALIDATED
    assert pool["precondition_worktree_clean_at_phase_start"] is True
    assert pool["source_cells_terminal"] == 20
    assert pool["source_cells_computed"] == 20
    assert pool["user_model_source_cases"] == 19528
    assert pool["source_ranking_rows"] == 390560
    assert pool["authoritative_pool_users"] == sum(EXPECTED_USERS.values())
    assert pool["pool_candidate_rows"] == sum(EXPECTED_USERS.values()) * 20
    assert pool["ready_reranking_rows"] == 30
    assert pool["ready_llm_requests"] == 29292
    assert pool["checkpoint_hashes_changed"] == 0
    assert no_training["training_runs_executed_in_f1bzq"] == 0
    assert no_training["checkpoint_parameter_updates"] == 0
    assert no_training["llm_inference_calls_executed"] == 0
    assert no_training["network_calls_executed"] == 0
    assert no_training["downloads_executed"] == 0
    assert no_training["commit_created"] == 0


def test_source_execution_covers_all_dataset_model_cells() -> None:
    execution = table("f1bzq_source_execution.csv")
    recall = table("f1bzq_source_recall.csv")

    assert len(execution) == 20
    assert set(execution["dataset"]) == set(EXPECTED_USERS)
    assert set(execution["model"]) == EXPECTED_MODELS
    assert execution["source_policy"].eq(SOURCE_POLICY).all()
    assert execution["status"].eq("COMPUTED").all()
    assert execution["terminal"].eq("true").all()
    assert execution["source_depth"].eq("20").all()
    assert execution["force_positive"].eq("false").all()
    assert execution["target_identity_used_for_ranking"].eq("false").all()
    assert execution["checkpoint_hash_identity"].eq("true").all()
    assert execution["retrievable_catalog_N"].astype(int).eq(execution["checkpoint_item_vocab_N"].astype(int)).all()
    assert execution["source_artifact_rows"].astype(int).sum() == 390560
    assert execution["user_model_cases"].astype(int).sum() == 19528

    assert len(recall) == 20
    assert recall["candidate_recall_tautological"].eq("false").all()
    assert recall["force_positive"].eq("false").all()
    assert recall["target_identity_used_for_ranking"].eq("false").all()


def test_pool_contributions_and_candidate_recall_are_non_tautological() -> None:
    contributions = table("f1bzq_pool_source_contributions.csv")
    recall = table("f1bzq_candidate_recall.csv")

    assert len(contributions) == sum(EXPECTED_USERS.values())
    assert contributions["status"].eq(POOL_STATUS_COMPUTED).all()
    assert contributions["first_model_20_0_0_0"].eq("false").all()
    contribution_cols = ["BPRMF_contribution", "GRU4Rec_contribution", "SASRec_contribution", "LightGCN_contribution"]
    assert contributions[contribution_cols].astype(int).sum(axis=1).eq(20).all()
    assert contributions["pool_size"].astype(int).eq(20).all()
    assert contributions["unique_count"].astype(int).eq(20).all()

    assert len(recall) == 5
    assert recall["status"].eq("COMPUTED").all()
    assert recall["candidate_recall_tautological"].eq("false").all()
    assert recall["force_positive"].eq("false").all()
    assert recall["target_identity_used_for_ranking"].eq("false").all()
    assert recall.set_index("dataset")["evaluation_N"].astype(int).to_dict() == EXPECTED_USERS


def test_foodcom_warm_cold_and_readiness_are_preserved() -> None:
    warm_cold = table("f1bzq_foodcom_warm_cold_recall.csv").set_index("scope")
    readiness = table("f1bzq_reranking_readiness.csv")
    requests = table("f1bzq_reranking_request_manifest.csv")

    assert warm_cold.loc["ALL_USERS", "evaluation_N"] == "1000"
    assert warm_cold.loc["WARM_USERS", "evaluation_N"] == "273"
    assert warm_cold.loc["TRUE_ZERO_HISTORY_USERS", "evaluation_N"] == "727"
    assert warm_cold["candidate_recall_tautological"].eq("false").all()
    assert warm_cold["force_positive"].eq("false").all()

    assert len(readiness) == 30
    assert readiness["readiness_status"].eq("READY_TO_INFER").all()
    assert readiness["future_requests_unlocked"].astype(int).sum() == 29292
    assert readiness["llm_inference_executed"].eq("false").all()
    assert readiness["training_executed_in_f1bzq"].eq("false").all()
    assert requests["candidate_order_source"].eq("f1bzq_authoritative_pool_hash").all()
    assert requests["ready_for_llm_inference"].eq("true").all()


def test_pool_validation_and_final_validation_pass() -> None:
    pool_validation = table("f1bzq_pool_validation.csv")
    validation = table("f1bzq_validation.csv")
    supersession = table("f1bzq_supersession_validation.csv")

    assert pool_validation["status"].eq("PASS").all()
    assert validation["status"].eq("PASS").all()
    assert supersession["superseded_status"].eq("SUPERSEDED_RERANKING_SOURCE_DEFECT").all()
    assert supersession["authoritative_for_downstream"].eq("false").all()


def test_authoritative_source_and_pool_artifacts_exist() -> None:
    source_manifest = manifest("f1bzq_source_ranking_manifest.json")
    pool_manifest = manifest("f1bzq_authoritative_pool_manifest.json")
    candidate_manifest = table("f1bzq_reranking_candidate_manifest.csv")

    assert len(source_manifest["source_artifacts"]) == 20
    assert len(pool_manifest["pool_artifacts"]) == 5
    for rel_path in source_manifest["source_artifacts"]:
        path = ROOT / rel_path
        assert path.exists()
        artifact = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert artifact.groupby("user_id").size().eq(20).all()
        assert artifact["force_positive"].eq("false").all()
        assert artifact["target_identity_used_for_ranking"].eq("false").all()

    assert len(candidate_manifest) == 5
    for row in candidate_manifest.to_dict("records"):
        path = ROOT / row["candidate_artifact_path"]
        assert path.exists()
        artifact = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert len(artifact) == int(row["candidate_rows"])
        assert artifact.groupby("user_id").size().eq(20).all()
        assert artifact["status"].eq(POOL_STATUS_COMPUTED).all()
        assert artifact["candidate_title"].str.strip().ne("").all()
        assert artifact["target_identity_used_for_ranking"].eq("false").all()


def test_adapted_policy_identity_is_preserved() -> None:
    contributions = table("f1bzq_pool_source_contributions.csv")
    candidate_manifest = table("f1bzq_reranking_candidate_manifest.csv")

    assert manifest("f1bzq_authoritative_pool_manifest.json")["pool_policy"] == ADAPTED_POLICY
    assert manifest("f1bzq_authoritative_pool_manifest.json")["pool_algorithm"] == ADAPTED_ALGORITHM
    assert candidate_manifest["candidate_order_immutable_for_llm"].eq("true").all()
    assert contributions["anti_domination_status"].eq("PASS_REPLAYED_BALANCED_POLICY").all()

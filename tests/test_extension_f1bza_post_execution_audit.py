from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "tables" / name, dtype=str, keep_default_na=False)


def manifest(name: str) -> dict:
    return json.loads((ROOT / "results" / "manifests" / name).read_text(encoding="utf-8"))


def test_f1bza_manifest_records_correction_required_decision() -> None:
    payload = manifest("f1bza_post_execution_audit.json")

    assert payload["precondition_state"] == "F1BZ_FINAL_TRADITIONAL_EXECUTION_VALIDATED"
    assert payload["decision_token"] == "F1BZA_CORRECTION_REQUIRED"
    assert payload["foodcom_history_decision"] == "FOODCOM_HISTORY_VALID"
    assert payload["reranking_pool_decision"] == "RERANKING_POOL_SOURCE_DEFECT"
    assert payload["f1br_protocol_defect_discovered_post_execution"] is True
    assert payload["no_training"] is True
    assert payload["no_llm_inference"] is True
    assert payload["no_network"] is True
    assert payload["f1bz_evidence_hash_counts"]["checkpoints"] == 100
    assert payload["f1bz_evidence_hash_counts"]["raw_recommendations"] == 100


def test_f1bza_foodcom_zero_history_is_true_input_property() -> None:
    alignment = table("f1bza_foodcom_history_alignment.csv")
    summary = table("f1bza_foodcom_zero_history_summary.csv")

    assert len(alignment) == 1000
    assert alignment["mapping_success"].eq("true").all()
    assert alignment["target_mapping_success"].eq("true").all()
    assert alignment["candidate_set_mapping_success"].eq("true").all()
    assert alignment["expected_matches_actual"].eq("true").all()
    assert alignment["zero_history_flag"].eq("true").sum() == 727
    assert alignment["zero_history_reason"].eq("TRUE_ZERO_HISTORY_RAW_SINGLE_INTERACTION_TARGET_ONLY").sum() == 727
    assert alignment["zero_history_reason"].eq("MAPPING_OR_PIPELINE_ZERO_HISTORY").sum() == 0
    assert summary["decision"].eq("FOODCOM_HISTORY_VALID").all()


def test_f1bza_model_input_mapping_audit_covers_all_foodcom_models() -> None:
    mapping = table("f1bza_model_input_mapping_audit.csv")

    assert len(mapping) == 4000
    assert set(mapping["model"]) == {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}
    assert mapping.groupby("model").size().eq(1000).all()
    assert mapping["user_mapping_success"].eq("true").all()
    assert mapping["target_mapping_success"].eq("true").all()
    assert mapping["candidate_set_mapping_success"].eq("true").all()
    assert mapping.groupby("model")["zero_history_flag"].apply(lambda s: s.eq("true").sum()).eq(727).all()


def test_f1bza_reranking_source_and_recall_are_tautological() -> None:
    source = table("f1bza_first_stage_source_audit.csv")
    tautology = table("f1bza_candidate_recall_tautology_audit.csv")

    assert len(source) == 20
    assert source["source_candidate_universe"].eq("FROZEN_RANKING_CANDIDATE20").all()
    assert source["source_universe_size"].astype(int).eq(20).all()
    assert source["full_catalog_scoring"].eq("false").all()
    assert source["positive_forced_into_source_universe"].eq("true").all()
    assert source["protocol_class"].eq("RERANKING_POOL_SOURCE_DEFECT").all()

    assert len(tautology) == 4882
    assert tautology["positive_in_ranking_candidate20"].eq("true").all()
    assert tautology["positive_in_final_reranking_pool"].eq("true").all()
    assert tautology["candidate_recall_tautological"].eq("true").all()
    assert tautology["scientific_status"].eq("RERANKING_POOL_SOURCE_DEFECT").all()


def test_f1bza_full_catalog_readiness_and_lastfm_lineage() -> None:
    readiness = table("f1bza_full_catalog_regeneration_readiness.csv")
    lineage = table("f1bza_lastfm_reranking_lineage_comparison.csv")

    assert len(readiness) == 20
    assert readiness["checkpoint_load_success_count"].astype(int).sum() == 100
    assert readiness["scoring_possible_over_model_universe"].eq("true").all()
    assert readiness.loc[readiness["dataset"].eq("lastfm"), "target_conclusion"].eq(
        "FULL_CATALOG_REGENERATION_READY"
    ).all()
    assert readiness.loc[~readiness["dataset"].eq("lastfm"), "target_conclusion"].eq("BLOCKED").all()

    assert set(lineage["status"]) == {
        "HISTORICAL_NON_TAUTOLOGICAL_REFERENCE",
        "RERANKING_POOL_SOURCE_DEFECT",
    }
    assert lineage.loc[lineage["lineage"].eq("current_f1bz_lastfm"), "candidate_recall_at_20"].iloc[0] == "1"


def test_f1bza_validity_and_integrity_gates_pass() -> None:
    validity = table("f1bza_result_validity_classification.csv")
    validation = table("f1bza_validation.csv")
    payload = manifest("f1bza_post_execution_audit.json")

    assert {"VALID_AS_IS", "INVALID", "REGENERATE_FROM_EXISTING_CHECKPOINT", "RETRAIN_REQUIRED"}.issubset(
        set(validity["classification"])
    )
    assert validation["status"].eq("PASS").all()
    assert payload["protected_integrity"]["active_video_games_rows"] == 0
    assert payload["protected_integrity"]["global_metric_blockers"] == 0
    assert payload["protected_integrity"]["f0_budget_unchanged"] is True
    assert payload["protected_integrity"]["foodcom_raw_hashes_unchanged"] is True
    assert payload["protected_integrity"]["h0_hashes_unchanged"] is True
    assert payload["protected_integrity"]["original_repo_status"] == ""
    assert payload["protected_integrity"]["paper_status"] == ""

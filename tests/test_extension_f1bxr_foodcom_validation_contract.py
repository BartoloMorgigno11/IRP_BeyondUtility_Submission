from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from src.extension import f1bxr_foodcom_validation_contract as f1bxr


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def outputs() -> dict[str, object]:
    return f1bxr.generate_f1bxr(ROOT)


def test_f1bxr_materializes_nested_temporal_split(outputs: dict[str, object]) -> None:
    split = outputs["nested_split_audit"]

    assert len(split) == 1000
    assert split["nested_split_status"].eq("NESTED_TEMPORAL_SPLIT_READY").all()
    assert split["tuning_training_prefix_excludes_validation_target"].eq("true").all()
    assert split["tuning_training_prefix_excludes_final_test_target"].eq("true").all()
    assert (split["training_history_length"].astype(int) + 2 == split["total_interactions"].astype(int)).all()
    validation_dates = pd.to_datetime(split["validation_target_date"])
    final_dates = pd.to_datetime(split["final_test_target_date"])
    same_day_ordered = split["validation_target_input_order"].astype(int) < split["final_test_target_input_order"].astype(int)
    assert ((validation_dates < final_dates) | ((validation_dates == final_dates) & same_day_ordered)).all()


def test_f1bxr_validation_cohort_is_1000_and_disjoint(outputs: dict[str, object]) -> None:
    cohort = outputs["validation_cohort"]
    final_contract = outputs["final_test_contract"]
    eligibility = outputs["validation_eligibility"]

    assert cohort["user_id"].nunique() == 1000
    assert len(set(cohort["user_id"]) & set(final_contract["user_ids"])) == 0
    assert f1bxr.eligibility_count_for(eligibility, "users_supporting_candidate_construction") >= 1000


def test_f1bxr_candidate_contract_has_one_positive_and_19_unseen(outputs: dict[str, object]) -> None:
    candidates = outputs["validation_candidates"]
    audit = outputs["validation_candidate_audit"]

    assert len(candidates) == 20000
    assert candidates.groupby("validation_sample_id")["candidate_item_id"].nunique().eq(20).all()
    assert candidates.groupby("validation_sample_id")["is_validation_positive"].apply(lambda value: (value == "true").sum()).eq(1).all()
    assert audit["positive_count"].astype(int).eq(1).all()
    assert audit["negative_count"].astype(int).eq(19).all()
    assert audit["prefix_overlap_count"].astype(int).eq(0).all()
    assert audit["duplicate_candidate_count"].astype(int).eq(0).all()


def test_f1bxr_sampling_and_ordering_are_deterministic_without_test_target() -> None:
    signature = inspect.signature(f1bxr.generate_validation_candidate_items)
    assert "final_test_target" not in signature.parameters
    assert "test_target" not in signature.parameters

    universe = [str(index) for index in range(1, 200)]
    first = f1bxr.generate_validation_candidate_items(
        user_id="user-1",
        validation_target="50",
        tuning_training_prefix=["1", "2", "3"],
        item_universe=universe,
    )
    second = f1bxr.generate_validation_candidate_items(
        user_id="user-1",
        validation_target="50",
        tuning_training_prefix=["1", "2", "3"],
        item_universe=universe,
    )

    assert first == second
    candidates = first[0]
    assert len(candidates) == 20
    assert len(set(candidates)) == 20
    assert "50" in candidates
    assert not (set(candidates) & {"1", "2", "3"})


def test_f1bxr_leakage_audit_passes(outputs: dict[str, object]) -> None:
    leakage = outputs["test_leakage_audit"]
    readiness = outputs["tuning_readiness"]

    assert leakage["status"].eq("PASS").all()
    assert readiness["final_test_candidates_input_to_tuning"].eq("false").all()
    assert readiness["uses_final_test_metrics"].eq("false").all()


def test_f1bxr_transitions_40_configs_to_ready_to_tune(outputs: dict[str, object]) -> None:
    readiness = outputs["tuning_readiness"]

    assert len(readiness) == 40
    assert readiness["status"].eq("READY_TO_TUNE").all()
    assert readiness["seed"].astype(int).eq(42).all()
    assert readiness["selection_metric"].eq("validation_ndcg_at_5").all()
    assert set(readiness["model"]) == set(f1bxr.TRAINABLE_MODELS)
    assert readiness["training_executed"].eq("false").all()
    assert readiness["final_training_executed"].eq("false").all()


def test_f1bxr_preserves_f1bx_baseline_evidence(outputs: dict[str, object]) -> None:
    preservation = outputs["baseline_preservation"]

    assert preservation["f1bx_decision_token"] == "F1BX_TUNING_AND_BASELINES_BLOCKED"
    assert preservation["baseline_rows"] == 15
    assert preservation["baseline_computed_rows"] == 15
    assert preservation["baseline_recommendation_rows"] == 292920
    assert preservation["hashes_unchanged"] is True


def test_f1bxr_integrity_and_validation(outputs: dict[str, object]) -> None:
    manifest = outputs["validation_contract_manifest"]
    validation = outputs["validation"]

    assert manifest["decision_token"] == f1bxr.DECISION_RESOLVED
    assert manifest["training_runs_executed"] == 0
    assert manifest["final_training_runs_executed"] == 0
    assert manifest["llm_inference_calls_executed"] == 0
    assert manifest["integrity"]["active_video_games_rows"] == 0
    assert manifest["integrity"]["global_metric_blockers"] == 0
    assert manifest["integrity"]["f0_budget_unchanged"] is True
    assert validation["status"].eq("PASS").all()

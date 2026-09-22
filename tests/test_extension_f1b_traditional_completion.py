from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.extension import f1b_traditional_completion as f1b


ROOT = Path(__file__).resolve().parents[1]


def test_f1b_materializes_explicit_blocked_gate_without_training() -> None:
    outputs = f1b.generate_f1b(ROOT)
    validation = outputs["validation"]
    manifest = outputs["training_manifest"]

    assert outputs["decision_token"] == f1b.DECISION_BLOCKED
    assert validation["status"].eq("PASS").all()
    assert manifest["blocked_before_training"] is True
    assert manifest["completion_validated"] is False
    assert manifest["executed_training_runs"] == 0
    assert manifest["llm_inference_calls_executed"] == 0
    assert manifest["downloads_executed"] == 0
    assert manifest["network_calls_executed"] == 0


def test_f1b_covers_the_frozen_35_by_17_traditional_surface() -> None:
    outputs = f1b.generate_f1b(ROOT)
    reuse = outputs["reuse_audit"]
    readiness = outputs["result_cell_readiness"]
    completeness = outputs["five_dataset_completeness"]

    assert len(reuse) == 35
    assert len(completeness) == 35
    assert len(readiness) == 595
    assert set(reuse["dataset"]) == set(f1b.ACTIVE_DATASETS)
    assert set(reuse["method"]) == set(f1b.METHODS)
    assert not readiness.astype(str).apply(
        lambda column: column.str.contains("video_games|video games|videogames", case=False, regex=True)
    ).any().any()
    assert readiness["f1b_status"].isin(f1b.RESULT_STATUS_ALLOWED).all()


def test_f1b_training_plan_records_100_seed_slots_but_executes_none() -> None:
    outputs = f1b.generate_f1b(ROOT)
    plan = outputs["training_plan"]
    results = outputs["training_results"]

    assert len(plan) == 5 * 4 * 5
    assert len(results) == len(plan)
    assert set(plan["method"]) == set(f1b.TRAINABLE_METHODS)
    assert set(pd.to_numeric(plan["seed_slot"]).astype(int)) == {1, 2, 3, 4, 5}
    assert plan["resolved_seed"].eq("").all()
    assert plan["training_plan_status"].eq("BLOCKED_BEFORE_TRAINING").all()
    assert results["training_executed"].astype(str).str.lower().eq("false").all()


def test_f1b_preserves_f1arr_degenerate_traditional_cells() -> None:
    outputs = f1b.generate_f1b(ROOT)
    readiness = outputs["result_cell_readiness"]
    degenerate = readiness.loc[readiness["f1b_status"].eq("DEGENERATE_PROTOCOL_INPUT")]

    assert len(degenerate) == outputs["degenerate_traditional_cell_count"]
    assert len(degenerate) == 7
    assert set(degenerate["dataset"]) == {"foodcom"}
    assert set(degenerate["metric"]) == {"popreo_at_5"}


def test_f1b_keeps_reranking_locked_until_candidate_recall_exists() -> None:
    outputs = f1b.generate_f1b(ROOT)
    candidate_recall = outputs["candidate_recall"]
    reranking = outputs["reranking_readiness"]

    assert len(candidate_recall) == 30
    assert candidate_recall["status"].eq("BLOCKED_FIRST_STAGE_NOT_MATERIALIZED").all()
    assert pd.to_numeric(reranking["future_requests_unlocked"]).sum() == 0
    assert reranking["readiness_status"].eq("BLOCKED_FIRST_STAGE_NOT_MATERIALIZED").all()

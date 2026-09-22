from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.extension.f1bzp_reranking_protocol_resolution import (
    ADAPTED_ALGORITHM,
    ADAPTED_POLICY,
    DECISION_ADAPTED_FROZEN,
    EXACT_UNRECOVERED,
    F1BR_DEFECT,
    MODEL_READY,
    TARGET_OOV_RECALL_MISS,
    TARGET_REPRESENTABLE,
    balanced_round_robin_unique_pool,
)


ROOT = Path(__file__).resolve().parents[1]


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "tables" / name, dtype=str, keep_default_na=False)


def manifest() -> dict:
    return json.loads((ROOT / "results/manifests/f1bzp_reranking_protocol_resolution.json").read_text(encoding="utf-8"))


def test_f1bzp_manifest_freezes_adapted_protocol_without_execution() -> None:
    payload = manifest()

    assert payload["decision_token"] == DECISION_ADAPTED_FROZEN
    assert payload["model_representable_ready_cells"] == 20
    assert payload["prior_f1bzc_raw_catalog_blockers_reassessed"] == 16
    assert payload["checkpoint_retraining_required_cells"] == 0
    assert payload["authoritative_pools_generated"] == 0
    assert payload["traditional_scoring_executed_in_f1bzp"] == 0
    assert payload["training_runs_executed_in_f1bzp"] == 0
    assert payload["llm_inference_calls_executed"] == 0
    assert payload["network_calls_executed"] == 0
    assert payload["future_reranking_requests_unlocked"] == 0
    assert payload["requested_source_depth"] == 20
    assert payload["actual_max_depth_used"] == "PROTOCOL_ONLY_NOT_EXECUTED"


def test_checkpoint_vocabulary_intersection_makes_all_cells_scoring_ready() -> None:
    audit = table("f1bzp_checkpoint_vocab_audit.csv")

    assert len(audit) == 20
    assert audit["model_representable_status"].eq(MODEL_READY).all()
    assert audit["checkpoint_shape_status"].eq("PASS").all()
    assert audit["representative_checkpoint_hash_matches_manifest"].eq("true").all()
    assert audit["can_score_eligible_catalog_intersection"].eq("true").all()
    assert audit["requires_scoring_raw_catalog_oov_items"].eq("false").all()
    assert audit["checkpoint_retraining_required"].eq("false").all()
    assert audit["eligible_and_representable_N"].astype(int).eq(audit["checkpoint_item_vocab_N"].astype(int)).all()
    assert int(audit["f1bzc_raw_catalog_blocker_reassessed"].eq("RESOLVED_BY_MODEL_REPRESENTABLE_POLICY").sum()) == 16


def test_target_oov_policy_is_recall_miss_not_forced() -> None:
    targets = table("f1bzp_target_representability.csv")

    assert len(targets) == (990 + 892 + 1000 + 1000 + 1000) * 4
    assert targets["target_forced"].eq("false").all()
    assert set(targets["status"]).issubset({TARGET_REPRESENTABLE, TARGET_OOV_RECALL_MISS})
    assert targets.loc[targets["positive_representable"].ne("true"), "status"].eq(TARGET_OOV_RECALL_MISS).all()
    assert targets["eligible_candidate_count"].astype(int).ge(20).all()


def test_paper_evidence_and_top5x4_are_not_overclaimed() -> None:
    evidence = table("f1bzp_paper_pool_evidence.csv")
    top5x4 = table("f1bzp_top5x4_hypothesis_audit.csv")
    policy = table("f1bzp_pool_policy_resolution.csv")

    exact = policy.loc[policy["policy_id"].eq("PAPER_REPOSITORY_EXACT")].iloc[0]
    assert exact["resolution_status"] == EXACT_UNRECOVERED
    assert exact["paper_exact"] == "false"
    assert "PAPER_REPOSITORY_RERANKING_POOL_RECOVERED" not in set(evidence["classification"])
    assert top5x4["classification"].eq("PLAUSIBLE_BUT_UNRECOVERED").all()
    assert "SUPPORTED" not in set(top5x4["classification"])


def test_current_f1br_pool_defect_is_explicit_and_superseded() -> None:
    policy = table("f1bzp_pool_policy_resolution.csv")
    old = policy.loc[policy["policy_id"].eq("F1BR_ADAPTED_ORDERED_DEDUP_TOP20_UNION_V1")].iloc[0]
    adapted = policy.loc[policy["policy_id"].eq(ADAPTED_POLICY)].iloc[0]

    assert old["resolution_status"] == "SUPERSEDED_POOL_COMPONENT_ONLY"
    assert F1BR_DEFECT in old["reason"]
    assert adapted["algorithm_id"] == ADAPTED_ALGORITHM
    assert adapted["resolution_status"] == "FROZEN"
    assert adapted["paper_exact"] == "false"
    assert adapted["force_positive"] == "false"
    assert adapted["target_identity_used"] == "false"
    assert adapted["candidate_order_immutable_for_llm"] == "true"
    assert adapted["source_order_preservation"] == "true"
    assert adapted["final_pool_size"] == "20"


def test_balanced_round_robin_prevents_first_model_domination() -> None:
    sources = {
        "BPRMF": [f"b{index}" for index in range(1, 21)],
        "GRU4Rec": [f"g{index}" for index in range(1, 21)],
        "SASRec": [f"s{index}" for index in range(1, 21)],
        "LightGCN": [f"l{index}" for index in range(1, 21)],
    }

    pool, lineage = balanced_round_robin_unique_pool(sources)

    assert len(pool) == 20
    assert len(set(pool)) == 20
    assert [row["source_model"] for row in lineage[:4]] == ["BPRMF", "GRU4Rec", "SASRec", "LightGCN"]
    assert [row["source_rank"] for row in lineage[:4]] == [1, 1, 1, 1]
    assert lineage[4]["source_model"] == "BPRMF"
    assert lineage[4]["source_rank"] == 2
    assert max(row["source_rank"] for row in lineage) == 5


def test_balanced_round_robin_deduplication_refill_is_deterministic() -> None:
    sources = {
        "BPRMF": ["x", "b2", "b3", "b4", "b5", "b6"],
        "GRU4Rec": ["x", "g2", "b3", "g4", "g5", "g6"],
        "SASRec": ["s1", "b2", "s3", "s4", "s5", "s6"],
        "LightGCN": ["l1", "l2", "x", "l4", "l5", "l6"],
    }

    first_pool, first_lineage = balanced_round_robin_unique_pool(sources, pool_size=12, source_depth=6)
    second_pool, second_lineage = balanced_round_robin_unique_pool(sources, pool_size=12, source_depth=6)

    assert first_pool == second_pool
    assert first_lineage == second_lineage
    assert len(first_pool) == 12
    assert len(set(first_pool)) == 12
    assert max(row["source_rank"] for row in first_lineage) > 3


def test_corrective_scope_has_no_retraining_blocker_and_no_video_games() -> None:
    scope = table("f1bzp_corrective_scope.csv")
    readiness = table("f1bzp_representable_catalog_readiness.csv")
    payload = manifest()

    assert len(scope) == 20
    assert set(scope["dataset"]) == {"beauty", "sports", "ml1m", "lastfm", "foodcom"}
    assert "video_games" not in set(scope["dataset"])
    assert scope["CHECKPOINT_VALID"].eq("true").all()
    assert scope["REPRESENTABLE_CATALOG_SCORING_READY"].eq("true").all()
    assert scope["POOL_POLICY_READY"].eq("true").all()
    assert scope["FULL_CATALOG_REGENERATION_REQUIRED"].eq("true").all()
    assert scope["REAL_BLOCKER"].eq("NONE").all()
    assert scope["checkpoint_retraining_required"].eq("false").all()
    assert readiness["checkpoint_retraining_required"].eq("false").all()
    assert payload["active_datasets"] == ["beauty", "sports", "ml1m", "lastfm", "foodcom"]


def test_budget_and_validation_gate_are_closed() -> None:
    payload = manifest()
    validation = table("f1bzp_validation.csv")
    budget = payload["request_budget"]

    assert budget["CORE_MANDATORY"] == 29292
    assert budget["REQUIRED_SENSITIVITY"] == 21000
    assert budget["MANDATORY_TOTAL"] == 50292
    assert budget["TRAINING_DEPENDENT"] == 29292
    assert budget["CONDITIONAL"] == 9000
    assert budget["OPTIONAL"] == 12000
    assert budget["CONCEPTUAL_ALL_APPLICABLE"] == 100584
    assert validation["status"].eq("PASS").all()

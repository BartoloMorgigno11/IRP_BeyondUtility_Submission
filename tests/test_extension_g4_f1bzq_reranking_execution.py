from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.extension import g4_f1bzq_reranking_execution as g4


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g4_f1bzq_reranking_execution.yaml"


@lru_cache(maxsize=1)
def _plan() -> dict:
    return g4.run_g4(ROOT, CONFIG, execute=False, resume=False, max_new_calls=None)


def test_g4_config_freezes_f1bzq_only_zero_reuse_protocol() -> None:
    config = g4.load_config(ROOT, CONFIG)

    assert config["phase"]["id"] == g4.PHASE_ID
    assert config["phase"]["scientific_status"] == g4.SCIENTIFIC_STATUS
    assert config["phase"]["expected_base_checkpoint"] == g4.SCIENTIFIC_BASE_CHECKPOINT
    assert config["task"]["id"] == "reranking"
    assert config["task"]["history_contract"] == "visible_last_10_or_available"
    assert config["task"]["candidate_pool_source"] == "F1BZQ_ONLY"
    assert config["task"]["candidate_pool_algorithm"] == "BALANCED_RANK_ROUND_ROBIN_UNIQUE_20"
    assert config["execution"]["response_reuse"] == 0
    assert config["accounting"]["logical_requests"] == 29_292
    assert config["accounting"]["new_ollama_calls"] == 29_292
    assert config["accounting"]["cells"] == 30
    assert config["execution"]["allow_local_ollama_inference"] is False
    assert config["execution"]["allow_local_ollama_metadata"] is False
    assert config["execution"]["allow_f1br_f1bzc_pool_use"] is False
    assert config["execution"]["allow_f6m_response_reuse"] is False
    g4.validate_execution_config(config)


def test_g4_request_universe_is_full_f1bzq_reranking_matrix() -> None:
    request_universe = _plan()["request_universe"]

    assert len(request_universe) == 29_292
    assert request_universe["canonical_request_id"].is_unique
    assert request_universe.groupby(["dataset", "model", "prompt_strategy"]).ngroups == 30
    assert request_universe["task"].eq("reranking").all()
    assert request_universe["scientific_status"].eq("ADAPTED_EXTENSION_NOT_PAPER_EXACT").all()
    assert request_universe["candidate_source_protocol"].eq(g4.CANDIDATE_SOURCE_PROTOCOL).all()
    assert request_universe["candidate_pool_algorithm"].eq("BALANCED_RANK_ROUND_ROBIN_UNIQUE_20").all()
    assert request_universe["candidate_count"].astype(int).eq(20).all()
    assert request_universe["requested_output_length"].astype(int).eq(10).all()
    assert request_universe["parser_expected_length"].astype(int).eq(10).all()
    assert request_universe["metric_k"].astype(int).eq(5).all()
    assert request_universe["response_reuse_allowed"].eq("false").all()
    assert request_universe["planned_new_ollama_call"].eq("true").all()

    dataset_users = request_universe.drop_duplicates(["dataset", "user_id"]).groupby("dataset").size().to_dict()
    assert dataset_users == {"beauty": 990, "sports": 892, "ml1m": 1000, "lastfm": 1000, "foodcom": 1000}


def test_g4_hashes_prompt_history_model_and_pool_contracts_are_frozen() -> None:
    request_universe = _plan()["request_universe"]
    candidate_hashes = _plan()["candidate_hashes"]
    model_digests = _plan()["model_digests"]

    assert len(candidate_hashes) == 4_882
    assert request_universe["pool_hash"].str.len().eq(64).all()
    assert request_universe["candidate_hash"].str.len().eq(64).all()
    assert request_universe["prompt_hash"].str.len().eq(64).all()
    assert request_universe["payload_hash"].str.len().eq(64).all()
    assert request_universe["history_hash"].str.len().eq(64).all()
    assert request_universe["history_contract"].eq("visible_last_10_or_available").all()
    assert request_universe["history_length"].astype(int).le(10).all()
    assert request_universe["candidate_source"].str.contains("f1bzq_reranking_pools", regex=False).all()
    assert not request_universe["candidate_source"].str.contains("f1bzc", regex=False).any()
    assert request_universe["force_positive"].eq("false").all()
    assert request_universe["target_identity_used_for_ranking"].eq("false").all()
    assert model_digests["frozen_digest"].str.len().eq(64).all()


def test_g4_metric_and_statistics_contract_exclude_blocked_or_out_of_scope_metrics() -> None:
    metric_contract = _plan()["metric_contract"]
    statistics = _plan()["statistical_comparisons"]
    candidate_recall = _plan()["candidate_recall"]

    included = set(metric_contract.loc[metric_contract["included_in_g4"].eq("true"), "metric_id"])
    excluded = set(metric_contract.loc[metric_contract["included_in_g4"].eq("false"), "metric_id"])

    assert {"hr_at_5", "ndcg_at_5", "mrr", "candidate_recall_at_20", "conditional_hr_at_5", "conditional_ndcg_at_5"} <= included
    assert {"canddif_hr_at_5", "canddif_ndcg_at_5", "popreo_at_5", "dpd_at_5"} <= excluded
    assert metric_contract.loc[metric_contract["metric_family"].eq("beyond_utility_g1_g2_compatible"), "included_in_g4"].eq("true").all()
    assert candidate_recall.set_index("dataset")["evaluation_N"].astype(int).to_dict() == {
        "beauty": 990,
        "sports": 892,
        "ml1m": 1000,
        "lastfm": 1000,
        "foodcom": 1000,
    }
    assert candidate_recall["denominator"].eq("all_users").all()
    assert statistics["p_value_method"].eq("NOT_FROZEN_DO_NOT_COMPUTE").all()
    assert statistics["holm"].eq("false").all()
    assert statistics.loc[statistics["comparison_family"].eq("g1_vs_g4_pipeline_descriptive"), "inferential_test_allowed"].eq("false").all()


def test_g4_zero_call_dry_run_accounting_and_validation_pass() -> None:
    plan = _plan()
    accounting = plan["dry_run_accounting"].iloc[0]
    reuse = plan["response_reuse_policy"]
    validation = plan["validation"]

    assert int(accounting["logical_requests"]) == 29_292
    assert int(accounting["exact_response_reuse"]) == 0
    assert int(accounting["future_ollama_calls"]) == 29_292
    assert int(accounting["cells"]) == 30
    assert int(accounting["existing_terminal_g4_evidence_before"]) == 0
    assert int(accounting["llm_inference_calls_executed_in_this_invocation"]) == 0
    assert reuse["reuse_allowed"].eq("false").all()
    assert reuse["exact_reused_rows"].astype(int).sum() == 0
    assert validation["status"].eq("PASS").all()


def test_g4_cli_requires_explicit_allow_api_calls_for_execution(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = g4.main(["--config", str(CONFIG), "--execute", "--max-new-calls", "0", "--no-write"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--allow-api-calls" in captured.out


def test_g4_writes_freeze_manifests_after_dry_run() -> None:
    outputs = g4.run_g4(ROOT, CONFIG, execute=False, resume=False, max_new_calls=None)
    g4.write_outputs(ROOT, outputs)

    protocol = json.loads((ROOT / "results" / "manifests" / "g4_f1bzq_reranking_protocol_freeze.json").read_text(encoding="utf-8"))
    universe = pd.read_csv(ROOT / "results" / "tables" / "g4_f1bzq_reranking_request_universe.csv", dtype=str, keep_default_na=False)

    assert protocol["decision_token"] == g4.DECISION_VALIDATED
    assert protocol["logical_requests"] == 29_292
    assert protocol["response_reuse"] == 0
    assert protocol["future_ollama_calls"] == 29_292
    assert protocol["llm_inference_calls_executed_in_this_invocation"] == 0
    assert len(universe) == 29_292

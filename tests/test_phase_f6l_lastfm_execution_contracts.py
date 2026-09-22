"""Focused checks for F6L LastFM execution contracts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from src.analysis import plan_f6l_paper_faithful_completion_and_rq_revalidation as f6l


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml"

pytestmark = [pytest.mark.cleanup_historical, pytest.mark.phase_snapshot]


@pytest.fixture(scope="module")
def lastfm_contracts() -> f6l.LastfmExecutionContractsResult:
    return f6l.build_lastfm_execution_contracts(CONFIG, write_outputs_enabled=False)


def test_lastfm_contracts_decision_counts_and_hierarchy(lastfm_contracts: f6l.LastfmExecutionContractsResult) -> None:
    manifest = lastfm_contracts.manifest

    assert lastfm_contracts.decision_token == f6l.DECISION_LASTFM_CONTRACTS_READY
    assert manifest["evidence_hierarchy"] == {
        "exact_paper_replica": "blocked_missing_original_specification",
        "operationalized_1000_user_sample": "paper_protocol_reproduction_under_explicit_operationalization",
        "full_1859_local_cohort": "original_project_extension_full_cohort",
    }
    assert manifest["counts"]["operationalized_sample_users"] == 1000
    assert manifest["counts"]["operationalized_candidate_rows"] == 20000
    assert manifest["counts"]["operationalized_ranking_requests"] == 4000
    assert manifest["counts"]["full_cohort_ranking_requests"] == 7436
    assert manifest["policy"]["inference_performed"] is False
    assert manifest["policy"]["training_performed"] is False


def test_operationalized_ks_seed_and_holm_gate(lastfm_contracts: f6l.LastfmExecutionContractsResult) -> None:
    seed_rows = lastfm_contracts.tables["operationalized_seed_search"]
    ks_rows = lastfm_contracts.tables["operationalized_ks_results"]

    assert len(seed_rows) == 1
    assert seed_rows[0]["seed"] == "20260801"
    assert seed_rows[0]["accepted"] == "true"
    assert len(ks_rows) == 8
    assert {row["metric"] for row in ks_rows} == {"HR@5", "NDCG@5"}
    assert {row["model_name"] for row in ks_rows} == {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}
    assert all(float(row["p_value_holm"]) >= 0.05 for row in ks_rows)
    assert all(row["significant_after_holm"] == "false" for row in ks_rows)


def test_operationalized_candidates_are_fixed_shared_and_valid(lastfm_contracts: f6l.LastfmExecutionContractsResult) -> None:
    candidates = lastfm_contracts.tables["operationalized_candidates"]
    manifest = lastfm_contracts.tables["operationalized_ranking_request_manifest"]

    by_user = Counter(row["user_id"] for row in candidates)
    positives = Counter(row["user_id"] for row in candidates if row["is_positive"] == "true")
    assert len(by_user) == 1000
    assert set(by_user.values()) == {20}
    assert len(positives) == 1000
    assert set(positives.values()) == {1}

    hashes_by_user: dict[str, set[str]] = {}
    manifest_counts = Counter(row["user_id"] for row in manifest)
    for row in manifest:
        hashes_by_user.setdefault(row["user_id"], set()).add(row["candidate_list_hash"])
    assert set(manifest_counts.values()) == {4}
    assert all(len(value) == 1 for value in hashes_by_user.values())


def test_ranking_manifests_exclude_llmsrec_and_closed_models(lastfm_contracts: f6l.LastfmExecutionContractsResult) -> None:
    rows = (
        lastfm_contracts.tables["operationalized_ranking_request_manifest"]
        + lastfm_contracts.tables["full_cohort_ranking_request_manifest"]
    )
    assert {row["model_name"] for row in rows} == {"qwen2:7b-instruct", "mistral:7b-instruct"}
    assert {row["prompting_strategy"] for row in rows} == {"base/OpenP5 title-list", "LLMRank recency"}
    assert not any("LLMSRec" in row["prompting_strategy"] for row in rows)
    assert all(row["output_length"] == "10" for row in rows)
    assert all(row["rendered_prompt_hash"] and row["request_contract_hash"] for row in rows)


def test_reranking_readiness_keeps_paper_exact_blocked(lastfm_contracts: f6l.LastfmExecutionContractsResult) -> None:
    readiness = lastfm_contracts.tables["reranking_readiness"]
    by_scope = {row["scope"]: row for row in readiness if row["scope"] in {"paper_exact", "operationalized_sample", "full_cohort_extension"}}

    assert by_scope["paper_exact"]["status"] == "blocked_missing_original_specification"
    assert by_scope["paper_exact"]["future_request_count"] == "0"
    assert by_scope["operationalized_sample"]["future_request_count"] == "4000"
    assert by_scope["full_cohort_extension"]["future_request_count"] == "7436"
    assert len(lastfm_contracts.tables["operationalized_reranking_pool"]) == 20000
    assert len(lastfm_contracts.tables["full_cohort_reranking_pool"]) == 37180


def test_validation_and_manifest_flags_are_terminal(lastfm_contracts: f6l.LastfmExecutionContractsResult) -> None:
    validation = lastfm_contracts.tables["validation"]
    manifest = lastfm_contracts.manifest

    assert len(validation) == 23
    assert all(row["status"] == "pass" for row in validation)
    assert manifest["deterministic_in_memory_digest_match"] is True
    assert manifest["git"]["staged_paths"] == ""
    assert manifest["git"]["original_repo_status"] == ""
    assert manifest["f6i2p_yaml_count"] == 23
    json.dumps(manifest, sort_keys=True)

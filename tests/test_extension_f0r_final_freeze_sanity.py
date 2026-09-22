from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.extension import f0r_final_freeze_sanity as f0r


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _outputs() -> dict:
    return f0r.generate_f0r(ROOT)


def test_f0r_validation_passes_without_rerunning_f0_or_execution() -> None:
    outputs = _outputs()
    manifest = outputs["manifest"]

    assert outputs["validation"]["passed"], outputs["validation"]["errors"]
    assert manifest["head"] == f0r.EXPECTED_HEAD
    assert manifest["f0_rerun_executed"] == 0
    assert manifest["llm_inference_calls_executed"] == 0
    assert manifest["training_runs_executed"] == 0
    assert manifest["downloads_executed"] == 0
    assert manifest["network_calls_executed"] == 0
    assert manifest["commit_created"] == 0


def test_primary_breakdown_proves_dataset_n_and_request_identity() -> None:
    primary = _outputs()["primary_breakdown"]

    assert primary["dataset"].tolist() == ["beauty", "sports", "ml1m", "lastfm", "foodcom"]
    assert primary["N"].astype(int).tolist() == [990, 892, 1000, 1000, 1000]
    assert int(primary["N"].sum()) == 4_882
    assert int(primary["total_primary_requests"].sum()) == 29_292
    assert primary["model_count"].astype(int).eq(3).all()
    assert primary["prompt_count"].astype(int).eq(2).all()
    assert primary["status"].eq("PASS").all()

    foodcom = primary.loc[primary["dataset"].eq("foodcom")].iloc[0]
    assert foodcom["user_or_cohort_hash"] == "53c6bd4969102154776f6a67489fc402fa2af9608f77b0ea5512fcca0bce9717"
    assert foodcom["candidate_set_hash_where_applicable"] == "f946195b02da768691c417ea31bf42893b06c31fc0e1ea5488247263c523347b"
    assert foodcom["hash_status"] == "EXACT_USER_AND_CANDIDATE_HASHES_MATERIALIZED"

    original_four = primary.loc[primary["dataset"].ne("foodcom")]
    assert original_four["hash_status"].eq("COHORT_IDENTITY_HASH_ONLY").all()
    assert original_four["candidate_set_hash_where_applicable"].eq("PENDING_USER_LEVEL_REQUEST_MANIFEST").all()


def test_reranking_breakdown_is_training_dependent_only() -> None:
    reranking = _outputs()["reranking_breakdown"]
    reconciliation = _outputs()["reconciliation"]

    assert len(reranking) == 5
    assert int(reranking["requests"].sum()) == 29_292
    assert reranking["included_in_mandatory_total"].eq("false").all()
    assert reranking["included_only_in_training_dependent"].eq("true").all()
    assert reranking["training_prerequisite"].eq("true").all()
    assert reranking["candidate_recall_prerequisite"].eq("true").all()
    assert f0r.reconciliation_value(reconciliation, "MANDATORY_TOTAL") == 50_292
    assert f0r.reconciliation_value(reconciliation, "TRAINING_DEPENDENT") == 29_292
    assert f0r.reconciliation_value(reconciliation, "MANDATORY_PLUS_TRAINING_DEPENDENT") == 79_584


def test_sensitivity_profile_and_optional_breakdowns_have_exact_counts() -> None:
    outputs = _outputs()
    sensitivity = outputs["sensitivity_breakdown"]
    profile = outputs["profile_breakdown"]
    optional = outputs["optional_breakdown"]

    history = sensitivity.loc[sensitivity["family"].eq("history")]
    canddif = sensitivity.loc[sensitivity["family"].eq("candidate_order_canddif")]
    assert len(history) == 5 * 3 * 6
    assert int(history["requests"].sum()) == 18_000
    assert set(history["history_length"].astype(int)) == {0, 1, 2, 5, 10, 20}
    assert len(canddif) == 15
    assert int(canddif["requests"].sum()) == 3_000
    assert canddif["required_for_science"].eq("true").all()

    assert len(profile) == 60
    history_only = profile.loc[profile["profile_condition"].eq("HISTORY_ONLY")]
    conditional = profile.loc[profile["profile_condition"].ne("HISTORY_ONLY")]
    assert len(history_only) == 15
    assert history_only["requests"].astype(int).eq(0).all()
    assert history_only["reuses_history_l20_outputs"].eq("true").all()
    assert len(conditional) == 45
    assert int(conditional["requests"].sum()) == 9_000
    assert conditional["requires_new_requests"].eq("true").all()

    assert len(optional) == 60
    assert set(optional["pool_size"].astype(int)) == {10, 50}
    assert int(optional["requests"].sum()) == 12_000
    assert optional["included_in_mandatory"].eq("false").all()
    assert optional["included_in_conditional"].eq("false").all()


def test_request_total_reconciliation_has_no_double_counting() -> None:
    reconciliation = _outputs()["reconciliation"]
    values = {row["budget_item"]: int(row["computed_request_count"]) for row in reconciliation.to_dict("records")}

    assert reconciliation["status"].eq("PASS").all()
    assert values == {
        "CORE_MANDATORY": 29_292,
        "REQUIRED_SENSITIVITY": 21_000,
        "MANDATORY_TOTAL": 50_292,
        "TRAINING_DEPENDENT": 29_292,
        "MANDATORY_PLUS_TRAINING_DEPENDENT": 79_584,
        "CONDITIONAL": 9_000,
        "FULL_PLANNED_IF_CONDITIONAL": 88_584,
        "OPTIONAL": 12_000,
        "CONCEPTUAL_ALL_APPLICABLE": 100_584,
        "EXISTING_REUSABLE": 0,
        "CANARY_INCREMENTAL": 0,
    }


def test_canary_and_exact_runtime_are_written_to_updated_inference_contract() -> None:
    outputs = _outputs()
    contract = outputs["updated_inference_contract"]
    runtime = {row["key"]: row["value"] for row in outputs["runtime_contract"].to_dict("records")}

    assert contract["canary_contract"]["canary_request_source"] == "canonical_final_request_ids"
    assert contract["canary_contract"]["incremental_canary_request_count"] == 0
    assert contract["canary_contract"]["noncanonical_diagnostic_canary_request_count"] == 0
    assert contract["canary_contract"]["canary_outputs_remain_final_evidence_if_protocol_passes"] is True

    preflight = contract["runtime_preflight"]
    assert "python_required" not in preflight
    assert preflight["python_interpreter_path"] == runtime["python_interpreter_path"]
    assert preflight["python_version"] == runtime["python_version"]
    assert preflight["environment_classification"] in {"project_local_venv_audit_recovery", "project_local_virtualenv"}
    assert preflight["ollama_client_version"] == "0.32.15"
    assert preflight["qwen2_7b_instruct_manifest_digest"].startswith("dd314f039b9d")
    assert preflight["mistral_7b_instruct_manifest_digest"].startswith("6577803aa9a0")
    assert preflight["gemma2_9b_manifest_digest"].startswith("ff02c3702f32")
    assert str(preflight["gpu"]).strip()
    assert str(preflight["ram_total_gb"]).strip()
    assert str(preflight["torch_installed"]).strip()

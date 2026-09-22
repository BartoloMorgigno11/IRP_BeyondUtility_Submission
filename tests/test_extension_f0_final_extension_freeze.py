from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.extension import f0_final_extension_freeze as f0


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _outputs() -> dict:
    return f0.generate_f0(ROOT)


def test_f0_precondition_and_local_model_manifests_are_verified() -> None:
    outputs = _outputs()
    manifest = outputs["manifest"]
    models = outputs["model_matrix"]

    assert outputs["validation"]["passed"], outputs["validation"]["errors"]
    assert manifest["head"] == f0.EXPECTED_HEAD
    assert manifest["head_matches_precondition"] is True
    assert manifest["llm_inference_calls_executed"] == 0
    assert manifest["training_runs_executed"] == 0
    assert manifest["downloads_executed"] == 0
    assert manifest["network_calls_executed"] == 0

    assert models["model_name"].tolist() == [
        "qwen2:7b-instruct",
        "mistral:7b-instruct",
        "gemma2:9b",
    ]
    assert models["ollama_manifest_present"].eq("true").all()
    prefixes = dict(zip(models["model_name"], models["ollama_manifest_digest_prefix"]))
    assert prefixes == {
        "qwen2:7b-instruct": "dd314f039b9d",
        "mistral:7b-instruct": "6577803aa9a0",
        "gemma2:9b": "ff02c3702f32",
    }


def test_final_dataset_universe_and_primary_protocol_are_exact() -> None:
    outputs = _outputs()
    datasets = outputs["dataset_matrix"]
    raw = outputs["raw_universe"]
    cells = outputs["result_universe"]
    primary = outputs["primary_matrix"]

    assert datasets["dataset"].tolist() == ["beauty", "sports", "ml1m", "lastfm", "foodcom"]
    assert datasets["active"].eq("true").all()
    assert set(raw.loc[raw["dataset"].ne("GLOBAL"), "dataset"]) == set(f0.DATASETS)
    assert set(cells.loc[cells["dataset"].ne("GLOBAL"), "dataset"]) == set(f0.DATASETS)
    assert not raw.astype(str).apply(lambda column: column.str.contains("video_games|video games|videogames", case=False, regex=True)).any().any()
    assert not cells.astype(str).apply(lambda column: column.str.contains("video_games|video games|videogames", case=False, regex=True)).any().any()

    assert len(primary) == 5 * 3 * 2
    assert set(primary["prompt"]) == {"BASE", "LLMRANK"}
    assert primary["candidate_pool_size"].astype(int).eq(20).all()
    assert primary["candidate_composition"].eq("1_heldout_positive_19_unseen_negatives").all()
    assert primary["candidate_indexing"].eq("1..20").all()
    assert primary["requested_output_length"].astype(int).eq(10).all()
    assert primary["parser_expected_length"].astype(int).eq(10).all()
    assert primary["metric_k"].astype(int).eq(5).all()
    assert primary["raw_status"].eq("TO_INFER").all()
    assert primary["request_count"].astype(int).sum() == 29_292


def test_history_matrix_is_paired_base_only_with_exact_request_count() -> None:
    history = _outputs()["history_matrix"]

    assert len(history) == 5 * 3 * 6
    assert set(history["dataset"]) == set(f0.DATASETS)
    assert set(history["model_name"]) == set(f0.LOCAL_LLMS)
    assert history["prompt"].eq("BASE").all()
    assert set(history["history_length"].astype(int)) == {0, 1, 2, 5, 10, 20}
    assert history["cohort_size"].astype(int).eq(200).all()
    assert history["request_count"].astype(int).sum() == 18_000
    assert history.loc[history["history_length"].astype(int).eq(0), "l0_policy"].eq("explicit_empty_history").all()
    assert history["paired_identity"].str.contains("same_user_positive_candidate_set_order").all()
    assert history["reuse_decision"].eq("NO_REUSE_R5C_TOP5_NOT_PROTOCOL_IDENTICAL").all()


def test_raw_and_result_universes_have_controlled_statuses_and_unique_identities() -> None:
    outputs = _outputs()
    raw = outputs["raw_universe"]
    cells = outputs["result_universe"]

    assert len(raw) == 323
    assert len(cells) == 8938
    assert raw["status"].isin(f0.RAW_STATUSES).all()
    assert cells["status"].isin(f0.RESULT_STATUSES).all()
    assert not raw.astype(str).apply(lambda column: column.str.contains("UNKNOWN", case=False, na=False)).any().any()
    assert not cells.astype(str).apply(lambda column: column.str.contains("UNKNOWN", case=False, na=False)).any().any()
    assert raw["raw_experiment_id"].is_unique
    assert raw["request_identity_hash"].is_unique
    assert cells["result_cell_id"].is_unique
    assert set(cells["raw_experiment_id"]).issubset(set(raw["raw_experiment_id"]))
    assert pd.to_numeric(cells["derived_metric_extra_llm_requests"]).sum() == 0


def test_request_budget_is_exact_and_does_not_promote_historical_reuse() -> None:
    budget = _outputs()["request_budget"]
    values = {row["budget_item"]: int(row["request_count"]) for row in budget.to_dict("records")}

    assert values["EXISTING_REUSABLE_REQUESTS"] == 0
    assert values["CORE_MANDATORY_NOW_REQUESTS"] == 29_292
    assert values["REQUIRED_SENSITIVITY_REQUESTS"] == 21_000
    assert values["MANDATORY_RAW_REQUESTS_TOTAL"] == 50_292
    assert values["TRAINING_DEPENDENT_REQUESTS"] == 29_292
    assert values["CONDITIONAL_REQUESTS"] == 9_000
    assert values["OPTIONAL_REQUESTS"] == 12_000
    assert values["FULL_PLANNED_IF_CONDITIONAL_REQUESTS_TOTAL"] == 88_584
    assert values["CONCEPTUAL_ALL_APPLICABLE_REQUESTS_TOTAL"] == 100_584
    assert values["PREVIOUS_D0B_ESTIMATE"] == 44_928
    assert values["MANDATORY_MINUS_PREVIOUS_D0B_ESTIMATE"] == 5_364


def test_reranking_traditional_sensitivity_and_profile_rows_are_separated() -> None:
    outputs = _outputs()
    raw = outputs["raw_universe"]
    reranking = outputs["reranking_matrix"]
    traditional = outputs["traditional_matrix"]
    sensitivity = outputs["sensitivity_matrix"]
    profile = outputs["profile_matrix"]

    assert len(reranking) == 30
    assert reranking["raw_status"].eq("TRAINING_DEPENDENT").all()
    assert reranking["dependency"].str.contains("candidate_recall_audit").all()
    assert reranking["positive_present_candidate_recall_required"].eq("true").all()

    assert len(traditional) == 35
    assert set(traditional["method"]) == set(f0.TRADITIONAL_METHODS)
    assert traditional.loc[traditional["requires_training"].eq("true"), "raw_status"].eq("TRAINING_DEPENDENT").all()
    assert traditional.loc[traditional["requires_training"].eq("false"), "raw_status"].eq("OFFLINE_ONLY").all()
    assert traditional["requires_llm_inference"].eq("false").all()
    assert traditional["request_count"].astype(int).eq(0).all()

    canddif = sensitivity.loc[sensitivity["sensitivity_family"].eq("CandDif / candidate order")]
    pools = sensitivity.loc[sensitivity["sensitivity_family"].eq("candidate_pool_size")]
    assert len(canddif) == 15
    assert canddif["budget_class"].eq("REQUIRED_SENSITIVITY").all()
    assert canddif["request_count"].astype(int).sum() == 3_000
    assert len(pools) == 60
    assert set(pools["condition"]) == {"POOL_10", "POOL_50"}
    assert pools["budget_class"].eq("OPTIONAL").all()

    assert len(profile) == 60
    assert profile.loc[profile["profile_condition"].eq("HISTORY_ONLY"), "request_count"].astype(int).eq(0).all()
    assert profile.loc[profile["profile_condition"].ne("HISTORY_ONLY"), "status"].eq("CONDITIONAL").all()
    assert set(raw["budget_class"]).issuperset({"CORE_MANDATORY_NOW", "REQUIRED_SENSITIVITY", "TRAINING_DEPENDENT", "CONDITIONAL", "OPTIONAL"})


def test_metric_inventory_blockers_and_foodcom_feature_policy_are_explicit() -> None:
    outputs = _outputs()
    metrics = outputs["metric_matrix"]
    blockers = outputs["global_blockers"]
    policy = outputs["foodcom_feature_policy"].iloc[0]

    assert {"hr_at_5", "ndcg_at_5", "popreo_at_5", "dpd_at_5", "ks_sample_representativeness"}.issubset(set(metrics["metric_id"]))
    metric_status = dict(zip(metrics["metric_id"], metrics["implementation_status"]))
    assert metric_status["popreo_at_5"] == "BLOCKED_GLOBAL_PROTOCOL"
    assert metric_status["dpd_at_5"] == "BLOCKED_GLOBAL_PROTOCOL"
    assert metric_status["ks_sample_representativeness"] == "IMPLEMENTATION_TO_COMPLETE"
    assert metrics["adds_llm_requests"].astype(int).eq(0).all()

    blocker_status = dict(zip(blockers["metric_or_family"], blockers["status"]))
    assert blocker_status["PopREO@5"] == "BLOCKED_GLOBAL_PROTOCOL"
    assert blocker_status["DPD@5"] == "BLOCKED_GLOBAL_PROTOCOL"
    assert blocker_status["K-S sample representativeness"] == "IMPLEMENTATION_TO_COMPLETE"
    assert blockers.loc[blockers["metric_or_family"].eq("K-S sample representativeness"), "missing_code_only"].eq("true").all()

    assert policy["primary_source_columns"] == "RAW_recipes.tags"
    assert policy["secondary_source_columns"] == "RAW_recipes.ingredients"
    assert policy["title_source_column"] == "RAW_recipes.name"
    assert policy["single_missing_name_policy"] == "do_not_synthesize"
    assert float(policy["primary_nonempty_coverage"]) > 0.99
    assert float(policy["secondary_nonempty_coverage"]) == 1.0


def test_execution_dag_inference_contract_and_runtime_contract_are_frozen() -> None:
    outputs = _outputs()
    dag = outputs["execution_dag"]
    contract = outputs["inference_contract"]

    assert f0.dag_is_acyclic(dag)
    assert set(dag["node_id"]) == {f"F0-DAG-{letter}" for letter in "ABCDEFGHIJK"}
    assert dag.loc[dag["node_id"].eq("F0-DAG-D"), "estimated_raw_llm_requests"].astype(int).iloc[0] == 29_292
    assert dag.loc[dag["node_id"].eq("F0-DAG-E"), "estimated_raw_llm_requests"].astype(int).iloc[0] == 18_000
    assert dag.loc[dag["node_id"].eq("F0-DAG-F"), "estimated_raw_llm_requests"].astype(int).iloc[0] == 3_000

    invalidity = contract["invalidity_policy"]
    assert invalidity["structural_policy"] == "preserve_raw_response_strict_valid_false_no_semantic_repair_no_retry"
    assert invalidity["backend_retry_count"] == 2
    assert invalidity["backend_retry_backoff_seconds"] == [5, 15]
    assert contract["raw_output_no_overwrite"] is True
    assert "candidate_hash" in contract["resume_skip_requires_exact_fields"]
    assert contract["canary_contract"]["minimum_evaluated_requests_before_gate"] == 30
    assert contract["canary_contract"]["action"] == "PAUSE_NO_PROMPT_OR_PARSER_CHANGE_WITHOUT_REFREEZE"
    assert contract["network_calls_executed"] == 0
    assert contract["llm_inference_calls_executed"] == 0
    assert contract["training_runs_executed"] == 0
    assert contract["downloads_executed"] == 0

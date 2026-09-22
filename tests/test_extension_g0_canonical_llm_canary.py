from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.extension.e0_parser import parse_extension_top10_response
from src.extension.e0_protocol import (
    ACTIVE_EXTENSION_DATASETS,
    FUTURE_MODELS,
    PROMPT_STRATEGIES,
    REQUESTED_OUTPUT_LENGTH,
)
from src.extension.g0_canonical_llm_canary import DECISION_VALIDATED, G0_RAW_PROTOCOL


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
MANIFESTS = ROOT / "results" / "manifests"
RAW_DIR = ROOT / "results" / "llm" / "extension_canonical" / "g0_canary"


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, dtype=str, keep_default_na=False)


def manifest(name: str) -> dict:
    return json.loads((MANIFESTS / name).read_text(encoding="utf-8"))


def request_rows() -> pd.DataFrame:
    return table("g0_canary_request_matrix.csv")


def parse_rows() -> pd.DataFrame:
    return table("g0_canary_parse_validation.csv")


def test_g0_manifests_validate_without_incremental_budget_or_training() -> None:
    request_manifest = manifest("g0_canary_request_manifest.json")
    execution_manifest = manifest("g0_canary_execution_manifest.json")

    assert request_manifest["decision_token"] == DECISION_VALIDATED
    assert execution_manifest["decision_token"] == DECISION_VALIDATED
    assert request_manifest["planned_requests"] == 72
    assert execution_manifest["terminal_requests"] == 72
    assert execution_manifest["backend_successful_requests"] == 72
    assert execution_manifest["executed_canonical_requests"] == 72
    assert execution_manifest["incremental_experiment_requests"] == 0
    assert execution_manifest["resume_run_llm_calls"] == 0
    assert execution_manifest["training_runs_executed"] == 0
    assert execution_manifest["tuning_runs_executed"] == 0
    assert execution_manifest["downloads_executed"] == 0
    assert execution_manifest["external_api_calls_executed"] == 0
    assert execution_manifest["f1bzq_pool_hashes_unchanged"] is True
    assert execution_manifest["validation_passed"] is True
    assert len(request_manifest["canonical_request_ids"]) == 72
    assert len(set(request_manifest["canonical_request_ids"])) == 72


def test_g0_request_matrix_crosses_users_models_prompts_and_tasks() -> None:
    matrix = request_rows()
    selection = table("g0_canary_selection.csv")

    assert len(matrix) == 72
    assert matrix["canonical_request_id"].is_unique
    assert set(matrix["dataset"]) == set(ACTIVE_EXTENSION_DATASETS)
    assert set(matrix["model"]) == set(FUTURE_MODELS)
    assert set(matrix["prompt_strategy"]) == set(PROMPT_STRATEGIES)
    assert set(matrix["task"]) == {"ranking", "reranking"}
    assert matrix.groupby(["dataset", "user_id", "task", "prompt_strategy"]).size().eq(len(FUTURE_MODELS)).all()
    assert matrix["candidate_count"].astype(int).eq(20).all()
    assert matrix["requested_output_length"].astype(int).eq(REQUESTED_OUTPUT_LENGTH).all()
    assert matrix["candidate_hash"].str.len().eq(64).all()
    assert matrix["prompt_hash"].str.len().eq(64).all()
    assert matrix["raw_response_protocol"].eq(G0_RAW_PROTOCOL).all()
    assert matrix["raw_response_path"].str.startswith("results/llm/extension_canonical/g0_canary/responses/").all()

    assert len(selection) == 6
    assert selection["user_id"].is_unique
    assert selection["selection_used_recommendation_success"].eq("false").all()
    assert selection["selection_used_candidate_recall"].eq("false").all()


def test_g0_ranking_and_reranking_candidate_contracts_are_frozen() -> None:
    matrix = request_rows()
    ranking = matrix[matrix["task"].eq("ranking")]
    reranking = matrix[matrix["task"].eq("reranking")]

    assert len(ranking) == 36
    assert ranking["candidate_source_protocol"].eq("F0_FROZEN_RANKING_CANDIDATES").all()
    assert ranking["candidate_source"].str.contains("f1bzc", case=False).sum() == 0

    assert len(reranking) == 36
    assert reranking["candidate_source_protocol"].eq("F1BZQ_AUTHORITATIVE_RERANKING_POOL").all()
    assert reranking["candidate_source"].str.contains("f1bzq", case=False).all()
    assert reranking["candidate_source"].str.contains("f1bzc", case=False).sum() == 0
    assert reranking["reranking_pool_hash"].str.len().eq(64).all()
    assert reranking.groupby(["dataset", "user_id"])["candidate_hash"].nunique().eq(1).all()
    assert reranking.groupby(["dataset", "user_id"])["reranking_pool_hash"].nunique().eq(1).all()


def test_g0_foodcom_warm_and_true_zero_history_are_explicit() -> None:
    selection = table("g0_canary_selection.csv")
    food = selection[selection["dataset"].eq("foodcom")].set_index("logical_user_group")

    assert set(food.index) == {"FOODCOM_WARM", "FOODCOM_TRUE_ZERO_HISTORY"}
    assert int(food.loc["FOODCOM_WARM", "visible_history_length"]) > 0
    assert int(food.loc["FOODCOM_WARM", "full_history_length"]) > 0
    assert food.loc["FOODCOM_WARM", "history_representation"] == "numbered_title_list"
    assert food.loc["FOODCOM_TRUE_ZERO_HISTORY", "visible_history_length"] == "0"
    assert food.loc["FOODCOM_TRUE_ZERO_HISTORY", "full_history_length"] == "0"
    assert food.loc["FOODCOM_TRUE_ZERO_HISTORY", "history_representation"] == "(no visible user history)"


def test_g0_execution_and_resume_are_idempotent() -> None:
    execution = table("g0_canary_execution.csv")
    resume = table("g0_canary_resume_validation.csv").iloc[0]

    assert len(execution) == 72
    assert execution["canonical_request_id"].is_unique
    assert execution["terminal"].eq("true").all()
    assert execution["backend_success"].eq("true").all()
    assert execution["raw_response_recorded_before_parse"].eq("true").all()
    assert execution["raw_response_hash"].str.len().eq(64).all()
    assert execution["response_record_count"].eq("1").all()
    assert execution["request_record_count"].eq("1").all()

    assert resume["resume_mode"] == "true"
    assert resume["completed_canonical_requests"] == "72"
    assert resume["resume_run_llm_calls"] == "0"
    assert resume["raw_response_hashes_unchanged"] == "true"
    assert resume["duplicate_request_rows"] == "0"
    assert resume["duplicate_response_rows"] == "0"
    assert resume["status"] == "PASS"


def test_g0_parse_validation_preserves_invalid_outputs_without_retry() -> None:
    parse = parse_rows()
    request_level = parse[parse["canary_request_id"].ne("AGGREGATE")]
    aggregate = parse[parse["canary_request_id"].eq("AGGREGATE")]
    matrix = request_rows()
    expected_aggregates = matrix.groupby(["model", "task", "prompt_strategy", "dataset"]).ngroups

    assert len(request_level) == 72
    assert len(aggregate) == expected_aggregates
    assert request_level["status"].eq("COMPUTED").all()
    assert request_level["invalid_output_retried"].eq("false").all()
    assert set(request_level["strict_parse_valid"]) <= {"true", "false"}
    assert request_level["raw_response_hash"].str.len().eq(64).all()
    assert aggregate["status"].eq("COMPUTED").all()
    assert aggregate["invalid_output_retried"].eq("false").all()

    invalid = request_level[request_level["strict_parse_valid"].eq("false")]
    assert invalid["structurally_invalid_preserved"].eq("true").all()


def test_g0_runtime_validation_and_raw_artifacts_exist() -> None:
    runtime = table("g0_canary_runtime.csv")
    validation = table("g0_canary_validation.csv")
    matrix = request_rows()

    assert not runtime.empty
    assert {"model_task", "model_task_prompt_dataset"} <= set(runtime["runtime_scope"])
    assert pd.to_numeric(runtime["mean_seconds"], errors="coerce").notna().all()
    assert validation["status"].eq("PASS").all()
    for rel_path in matrix["raw_response_path"]:
        path = ROOT / rel_path
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["raw_response_recorded_before_parse"] is True
        assert payload["canonical_request_id"] in set(matrix["canonical_request_id"])
    assert (RAW_DIR / "requests.jsonl").exists()
    assert (RAW_DIR / "responses.jsonl").exists()
    assert (RAW_DIR / "events.jsonl").exists()


def test_g0_strict_parser_rejects_unrepaired_structural_errors() -> None:
    candidates = [f"item-{index}" for index in range(1, 21)]
    valid = parse_extension_top10_response(
        '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}',
        candidate_items=candidates,
    )
    assert valid.parse_valid is True

    assert parse_extension_top10_response(
        'Here is the answer: {"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}',
        candidate_items=candidates,
    ).parse_valid is False
    assert parse_extension_top10_response(
        '{"ranked_candidate_indices":[1,1,2,3,4,5,6,7,8,9]}',
        candidate_items=candidates,
    ).parse_valid is False
    assert parse_extension_top10_response(
        '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,21]}',
        candidate_items=candidates,
    ).parse_valid is False
    assert parse_extension_top10_response(
        '{"ranked_candidate_indices":[1,2,3,4,5]}',
        candidate_items=candidates,
    ).parse_valid is False

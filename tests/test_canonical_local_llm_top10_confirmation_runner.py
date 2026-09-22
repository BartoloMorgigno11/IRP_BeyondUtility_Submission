from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.analysis.canonical_local_llm_top10_confirmation_preflight import (
    build_top10_parser_contract,
    build_top10_prompt_contract,
)
from src.llm.canonical_local_llm_top10_confirmation_runner import (
    DECISION_COLUMNS,
    EXECUTION_COLUMNS,
    EXPECTED_DIGESTS,
    MANIFEST_PATH,
    METRIC_COLUMNS,
    MODEL_ORDER,
    PARSE_COLUMNS,
    RANKING_COLUMNS,
    READINESS_THRESHOLD,
    REQUEST_MANIFEST_COLUMNS,
    REQUEST_STATE_COLUMNS,
    SAMPLE_PATH,
    TOP10_PARSER_ID,
    TOP10_PROMPT_ID,
    append_jsonl_fsync,
    build_decision,
    build_metrics,
    build_request_record,
    classify_request_state,
    classify_top10_response,
    frame,
    pending_requests_in_order,
    read_jsonl,
    validate_manifest,
    validate_parser_contract,
    validate_prompt_contract,
    validate_sample,
)


def _sample(overlap: bool = False, task_type: str = "reranking") -> pd.DataFrame:
    rows = []
    order = 1
    for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]:
        for idx in range(10):
            rows.append(
                {
                    "confirmation_row_id": f"CLT10-ROW-{order:04d}",
                    "selection_sha256": f"sel-{order:04d}",
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": f"{dataset}-cs",
                    "source_path": "synthetic.csv",
                    "user_id": f"user-{order:04d}",
                    "positive_item_id": "item-1",
                    "candidate_count": "20",
                    "history_item_count": "10",
                    "candidate_titles_complete": "true",
                    "history_titles_complete": "true",
                    "held_out_positive_absent": "true",
                    "primary_prompt_compatible": "true",
                    "used_in_v1_pilot": "true" if overlap and order == 1 else "false",
                    "used_in_v2_pilot": "false",
                    "used_in_hybrid_holdout": "false",
                    "disjoint_from_all_previous_pilots": "false" if overlap and order == 1 else "true",
                    "candidate_order_sha256": f"cand-{order:04d}",
                    "history_render_sha256": f"hist-{order:04d}",
                    "sample_order": str(order),
                    "sample_status": "selected_disjoint_not_executed",
                    "blocking_reason": "",
                    "notes": "test",
                }
            )
            order += 1
    return frame(rows, [
        "confirmation_row_id",
        "selection_sha256",
        "dataset",
        "task_type",
        "candidate_set_id",
        "source_path",
        "user_id",
        "positive_item_id",
        "candidate_count",
        "history_item_count",
        "candidate_titles_complete",
        "history_titles_complete",
        "held_out_positive_absent",
        "primary_prompt_compatible",
        "used_in_v1_pilot",
        "used_in_v2_pilot",
        "used_in_hybrid_holdout",
        "disjoint_from_all_previous_pilots",
        "candidate_order_sha256",
        "history_render_sha256",
        "sample_order",
        "sample_status",
        "blocking_reason",
        "notes",
    ])


def _execution_contract() -> pd.DataFrame:
    rows = []
    for model, contract_id in [
        ("Qwen2 7B Instruct", "exec_qwen2_7b_instruct_reranking_v1"),
        ("Mistral 7B Instruct", "exec_mistral_7b_instruct_reranking_v1"),
    ]:
        rows.append(
            {
                "execution_contract_id": contract_id,
                "model_family": model,
                "task_type": "reranking",
                "temperature": "0",
                "top_p": "",
                "top_k": "",
                "seed": "",
                "max_output_tokens": "512",
                "context_limit": "",
                "stop_sequences": "",
                "response_format": "json_object",
                "streaming": "false",
                "concurrency": "1",
                "timeout_seconds": "120",
                "infrastructure_retry_count": "0",
                "scientific_retry_count": "0",
                "request_order_policy": "deterministic",
                "raw_response_retention": "required",
                "parameter_source": "test",
                "paper_alignment": "adapted",
                "contract_status": "ready",
                "blocking_reason": "",
                "contract_sha256": f"{contract_id}-hash",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _manifest(sample: pd.DataFrame | None = None, prompt_hash: str = "prompt-hash") -> pd.DataFrame:
    sample = _sample() if sample is None else sample
    parser_hash = build_top10_parser_contract().iloc[0]["parser_schema_sha256"]
    rows = []
    order = 1
    for model in MODEL_ORDER:
        for sample_row in sample.to_dict("records"):
            exec_id = "exec_qwen2_7b_instruct_reranking_v1" if model.startswith("Qwen") else "exec_mistral_7b_instruct_reranking_v1"
            rows.append(
                {
                    "confirmation_request_id": f"CLT10-REQ-{order:04d}",
                    "confirmation_row_id": sample_row["confirmation_row_id"],
                    "model_family": model,
                    "model_tag": EXPECTED_DIGESTS[model]["tag"],
                    "model_digest": EXPECTED_DIGESTS[model]["digest"],
                    "dataset": sample_row["dataset"],
                    "task_type": sample_row["task_type"],
                    "candidate_set_id": sample_row["candidate_set_id"],
                    "user_id": sample_row["user_id"],
                    "request_order": str(order),
                    "prompt_contract_id": TOP10_PROMPT_ID,
                    "prompt_contract_sha256": prompt_hash,
                    "parser_contract_id": TOP10_PARSER_ID,
                    "parser_contract_sha256": parser_hash,
                    "execution_contract_id": exec_id,
                    "execution_contract_sha256": f"{exec_id}-hash",
                    "candidate_order_sha256": sample_row["candidate_order_sha256"],
                    "history_render_sha256": sample_row["history_render_sha256"],
                    "candidate_count": "20",
                    "requested_output_count": "10",
                    "temperature": "0",
                    "seed": "",
                    "max_output_tokens": "512",
                    "response_format": "json_object",
                    "streaming": "false",
                    "concurrency": "1",
                    "scientific_retry_count": "0",
                    "infrastructure_retry_count": "0",
                    "previous_pilot_overlap": "false",
                    "request_status": "planned_not_executed",
                    "blocking_reason": "",
                    "notes": "test",
                }
            )
            order += 1
    return frame(rows, REQUEST_MANIFEST_COLUMNS)


def test_prompt_contract_validation_accepts_valid_and_rejects_full_permutation_model_wording_and_hash():
    prompt = build_top10_prompt_contract()
    row = validate_prompt_contract(prompt)
    assert row["prompt_contract_id"] == TOP10_PROMPT_ID
    bad = prompt.copy()
    bad.loc[0, "requires_complete_permutation"] = "true"
    with pytest.raises(ValueError, match="requires_complete_permutation"):
        validate_prompt_contract(bad)
    bad = prompt.copy()
    bad.loc[0, "model_specific_wording"] = "true"
    with pytest.raises(ValueError, match="model_specific_wording"):
        validate_prompt_contract(bad)
    bad = prompt.copy()
    bad.loc[0, "contract_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        validate_prompt_contract(bad)


def test_parser_rejects_invalid_top10_forms_and_does_not_use_positive_label():
    parser = validate_parser_contract(build_top10_parser_contract())
    assert parser["positive_label_used"] == "false"
    assert classify_top10_response('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}')["strict_parse_valid"] is True
    assert classify_top10_response('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9]}')["primary_failure_reason"] == "output_count_invalid"
    assert classify_top10_response('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11]}')["primary_failure_reason"] == "output_count_invalid"
    assert classify_top10_response('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}')["primary_failure_reason"] == "output_count_invalid"
    assert classify_top10_response('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,9]}')["primary_failure_reason"] == "duplicate_indices"
    assert classify_top10_response('{"ranked_candidate_indices":[0,2,3,4,5,6,7,8,9,10]}')["primary_failure_reason"] == "index_range_invalid"
    assert classify_top10_response('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,21]}')["primary_failure_reason"] == "index_range_invalid"
    assert classify_top10_response('{"ranked_candidate_indices":[1,2]')["primary_failure_reason"] == "json_parse_failed"
    assert classify_top10_response('Here: {\"ranked_candidate_indices\":[1,2,3,4,5,6,7,8,9,10]}')["primary_failure_reason"] == "json_parse_failed"
    assert classify_top10_response('```json\\n{\"ranked_candidate_indices\":[1,2,3,4,5,6,7,8,9,10]}\\n```')["primary_failure_reason"] == "json_parse_failed"


def test_sample_and_manifest_validation_require_frozen_disjoint_reranking_contract():
    sample = _sample()
    assert validate_sample(sample, {}) == 0
    with pytest.raises(ValueError, match="previous-pilot|disjoint"):
        validate_sample(_sample(overlap=True), {})
    previous = {("Amazon Beauty", "reranking", "Amazon Beauty-cs", "user-0001"): {}}
    with pytest.raises(ValueError, match="overlaps"):
        validate_sample(sample, previous)
    with pytest.raises(ValueError, match="non-reranking"):
        validate_sample(_sample(task_type="ranking"), {})

    prompt = build_top10_prompt_contract()
    prompt_row = validate_prompt_contract(prompt)
    manifest = _manifest(sample, prompt_row["contract_sha256"])
    validate_manifest(manifest, sample, prompt_row, validate_parser_contract(build_top10_parser_contract()), _execution_contract())
    bad = manifest.copy()
    bad.loc[0, "requested_output_count"] = "20"
    with pytest.raises(ValueError, match="requested_output_count"):
        validate_manifest(bad, sample, prompt_row, validate_parser_contract(build_top10_parser_contract()), _execution_contract())
    bad = manifest.copy()
    bad.loc[0, "concurrency"] = "2"
    with pytest.raises(ValueError, match="concurrency"):
        validate_manifest(bad, sample, prompt_row, validate_parser_contract(build_top10_parser_contract()), _execution_contract())
    bad = manifest.copy()
    bad.loc[0, "scientific_retry_count"] = "1"
    with pytest.raises(ValueError, match="scientific_retry_count"):
        validate_manifest(bad, sample, prompt_row, validate_parser_contract(build_top10_parser_contract()), _execution_contract())


def test_resume_state_pending_completed_ambiguous_duplicate_append_only_and_payload_retention(tmp_path):
    assert classify_request_state(0, 0) == "not_started"
    assert classify_request_state(1, 1) == "completed"
    assert classify_request_state(1, 0) == "ambiguous_interrupted"
    assert classify_request_state(2, 1) == "invalid_duplicate_evidence"

    manifest = _manifest(_sample(), build_top10_prompt_contract().iloc[0]["contract_sha256"]).head(2).copy()
    assert len(pending_requests_in_order(manifest, [], [])) == 2
    completed = [{"confirmation_request_id": "CLT10-REQ-0001"}]
    assert [row["confirmation_request_id"] for row in pending_requests_in_order(manifest, completed, completed)] == ["CLT10-REQ-0002"]
    with pytest.raises(RuntimeError, match="ambiguous"):
        pending_requests_in_order(manifest, completed, [])
    with pytest.raises(RuntimeError, match="invalid_duplicate"):
        pending_requests_in_order(manifest, completed * 2, completed)

    path = tmp_path / "records.jsonl"
    append_jsonl_fsync(path, {"confirmation_request_id": "a"})
    append_jsonl_fsync(path, {"confirmation_request_id": "b"})
    assert [row["confirmation_request_id"] for row in read_jsonl(path)] == ["a", "b"]

    context = {
        "positive_item_id": "item-20",
        "candidate_items": [f"item-{idx}" for idx in range(1, 21)],
        "complete_prompt_sha256": "prompt",
    }
    payload = {"model": "qwen2:7b-instruct", "prompt": "no positive id here"}
    record = build_request_record(manifest.iloc[0].to_dict(), context, payload)
    assert record["attempt_index"] == 1
    assert record["request_payload"] == payload
    assert record["scientific_retry_count"] == "0"


def _frames_for_decision(strict_per_cell: int = 8, candidate_invalid: bool = False, duplicate: bool = False, truncation: bool = False):
    manifest = _manifest(_sample(), build_top10_prompt_contract().iloc[0]["contract_sha256"])
    exec_rows = []
    parse_rows = []
    ranking_rows = []
    cell_counts: dict[tuple[str, str], int] = {}
    for row in manifest.to_dict("records"):
        key = (row["model_family"], row["dataset"])
        seen = cell_counts.get(key, 0)
        is_strict = seen < strict_per_cell
        cell_counts[key] = seen + 1
        candidate_ok = is_strict and not (candidate_invalid and row["confirmation_request_id"] == "CLT10-REQ-0001")
        duplicate_count = 1 if duplicate and row["confirmation_request_id"] == "CLT10-REQ-0001" else 0
        exec_rows.append(
            {
                **{col: row.get(col, "") for col in EXECUTION_COLUMNS},
                "attempt_index": "1",
                "duration_seconds": "1",
                "backend_success": "true",
                "response_nonempty": "true",
                "eval_count": "12",
                "raw_response_sha256": f"sha-{row['confirmation_request_id']}",
                "response_jsonl_recorded": "true",
                "execution_status": "completed",
            }
        )
        parse_rows.append(
            {
                **{col: row.get(col, "") for col in PARSE_COLUMNS},
                "raw_response_sha256": f"sha-{row['confirmation_request_id']}",
                "strict_parser_id": TOP10_PARSER_ID,
                "json_parse_valid": "true" if is_strict else "false",
                "schema_valid": "true" if is_strict else "false",
                "output_count_valid": "true" if is_strict else "false",
                "index_type_valid": "true" if is_strict else "false",
                "index_range_valid": "true" if is_strict else "false",
                "uniqueness_valid": "true" if is_strict and not duplicate_count else "false",
                "candidate_validity": "true" if candidate_ok and not duplicate_count else "false",
                "catalog_validity": "true" if candidate_ok and not duplicate_count else "false",
                "duplicate_count": str(duplicate_count),
                "unknown_index_count": "0",
                "hallucination_count": "0",
                "strict_parse_valid": "true" if is_strict else "false",
                "strict_indices": "[1,2,3,4,5,6,7,8,9,10]" if is_strict else "[]",
                "output_length": "10" if is_strict else "9",
                "output_truncation_detected": "true" if truncation and row["confirmation_request_id"] == "CLT10-REQ-0001" else "false",
                "primary_failure_reason": "" if is_strict else "output_count_invalid",
            }
        )
        ranking_rows.append(
            {
                **{col: row.get(col, "") for col in RANKING_COLUMNS},
                "positive_item_id": "item-1",
                "candidate_items": "[\"item-1\"]",
                "strict_parse_valid": "true" if is_strict else "false",
                "ranked_candidate_indices": "[1,2,3,4,5,6,7,8,9,10]" if is_strict else "[]",
                "ranked_item_ids": "[\"item-1\"]" if is_strict else "[]",
                "top5_item_ids": "[\"item-1\"]" if is_strict else "[]",
                "positive_rank_zero_based": "0" if is_strict else "",
                "positive_in_top_5": "true" if is_strict else "",
                "ndcg_at_5": "1" if is_strict else "",
                "candidate_validity": "true" if candidate_ok and not duplicate_count else "false",
                "duplicate_count": str(duplicate_count),
                "hallucination_count": "0",
                "utility_evaluation_valid": "true" if is_strict else "false",
            }
        )
    execution = frame(exec_rows, EXECUTION_COLUMNS)
    parse = frame(parse_rows, PARSE_COLUMNS)
    rankings = frame(ranking_rows, RANKING_COLUMNS)
    metrics = build_metrics(manifest, execution, parse, rankings)
    request_state = frame(
        [
            {
                **{col: row.get(col, "") for col in REQUEST_STATE_COLUMNS},
                "request_record_count": "1",
                "response_record_count": "1",
                "request_state": "completed",
            }
            for row in manifest.to_dict("records")
        ],
        REQUEST_STATE_COLUMNS,
    )
    return manifest, execution, parse, rankings, metrics, request_state


def test_metrics_and_decision_approve_at_eight_per_cell_and_utility_does_not_set_readiness():
    manifest, execution, parse, rankings, metrics, request_state = _frames_for_decision(strict_per_cell=8)
    assert len(metrics) == 6
    assert metrics["strict_parse_validity_rate"].eq(str(READINESS_THRESHOLD)).all()
    assert metrics["cell_readiness_pass"].eq("true").all()
    decision = build_decision(metrics, request_state, execution, parse, SimpleNamespace(previous_overlap_count=0)).iloc[0]
    assert decision["pilot_decision"] == "approve_full_local_execution_with_ranking_v1_reranking_top10"
    assert decision["full_execution_expected_request_count"] == "11528"
    assert decision["utility_used_for_readiness"] == "false"


def test_decision_fails_for_one_seven_of_ten_cell_candidate_invalid_duplicate_or_truncation():
    for kwargs in [
        {"strict_per_cell": 7},
        {"strict_per_cell": 8, "candidate_invalid": True},
        {"strict_per_cell": 8, "duplicate": True},
        {"strict_per_cell": 8, "truncation": True},
    ]:
        _manifest_df, execution, parse, _rankings, metrics, request_state = _frames_for_decision(**kwargs)
        decision = build_decision(metrics, request_state, execution, parse, SimpleNamespace(previous_overlap_count=0)).iloc[0]
        assert decision["pilot_decision"] != "approve_full_local_execution_with_ranking_v1_reranking_top10"
        assert decision["readiness_threshold"] == str(READINESS_THRESHOLD)

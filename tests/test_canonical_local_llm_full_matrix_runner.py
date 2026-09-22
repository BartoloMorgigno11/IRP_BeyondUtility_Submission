from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.analysis.canonical_local_llm_full_matrix_preflight import MANIFEST_COLUMNS, frame
from src.llm import canonical_local_llm_full_matrix_runner as runner
from src.llm.canonical_local_llm_full_matrix_runner import (
    CELL_METRICS_COLUMNS,
    FullMatrixInputs,
    append_jsonl_fsync,
    build_cell_metrics,
    build_decision,
    classify_response_for_task,
    execute_pending_requests,
    pending_requests_in_order,
    read_jsonl,
    state_summary,
)


def _manifest_row(request_id: str = "CLFULL-REQ-000001", order: int = 1, task: str = "ranking") -> dict[str, str]:
    model = "Qwen2 7B Instruct"
    return {
        "request_id": request_id,
        "request_order": str(order),
        "chunk_id": "CLFULL-CHUNK-001",
        "model_family": model,
        "model_tag": "qwen2:7b-instruct",
        "model_digest": "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755",
        "dataset": "Amazon Beauty",
        "task_type": task,
        "candidate_set_id": "synthetic-candidates",
        "user_id": "user-1",
        "source_path": "synthetic.csv",
        "prompt_contract_id": "llm_rank_index_json_top5_v1" if task == "ranking" else "llm_rerank_index_json_top10_officialcode_v1",
        "prompt_contract_sha256": "prompt-hash",
        "parser_contract_id": "llm_rank_strict_primary_parser_v1" if task == "ranking" else "llm_rerank_top10_strict_primary_parser_v1",
        "parser_contract_sha256": "parser-hash",
        "execution_contract_id": "exec-qwen-ranking",
        "execution_contract_sha256": "exec-hash",
        "candidate_order_sha256": "candidate-hash",
        "history_render_sha256": "history-hash",
        "candidate_count": "20",
        "requested_output_count": "5" if task == "ranking" else "10",
        "temperature": "0",
        "seed": "",
        "max_output_tokens": "256" if task == "ranking" else "512",
        "response_format": "json_object",
        "streaming": "false",
        "concurrency": "1",
        "scientific_retry_count": "0",
        "infrastructure_retry_count": "0",
        "primary_prompt_compatible": "true",
        "manifest_status": "planned_not_executed",
        "blocking_reason": "",
        "notes": "synthetic",
    }


def _inputs(manifest: pd.DataFrame) -> FullMatrixInputs:
    contexts = {}
    for row in manifest.to_dict("records"):
        contexts[row["request_id"]] = {
            "request_id": row["request_id"],
            "positive_item_id": "item-1",
            "candidate_items": [f"item-{idx}" for idx in range(1, 21)],
            "candidate_titles": [f"Title {idx}" for idx in range(1, 21)],
            "system_message": "Return only JSON.",
            "user_message": "History\nCandidates",
            "history_render": "1. History",
            "candidate_render": "1. Title 1",
            "candidate_order_sha256": "candidate-hash",
            "history_render_sha256": "history-hash",
            "complete_prompt_sha256": "complete-hash",
            "prompt_contract_sha256": row["prompt_contract_sha256"],
            "requested_output_count": row["requested_output_count"],
            "requires_full_permutation": False,
        }
    return FullMatrixInputs(
        manifest=manifest,
        chunks=pd.DataFrame(),
        preflight_decision=pd.DataFrame(),
        cohort=pd.DataFrame(),
        contexts=contexts,
        protected_hashes={},
        recovery=pd.DataFrame(),
    )


def test_pending_state_skips_completed_and_blocks_ambiguous_or_duplicate(tmp_path):
    manifest = frame([_manifest_row("CLFULL-REQ-000001", 1), _manifest_row("CLFULL-REQ-000002", 2)], MANIFEST_COLUMNS)
    assert state_summary(manifest, [], [])["not_started"] == 2
    completed_req = [{"request_id": "CLFULL-REQ-000001"}]
    completed_resp = [{"request_id": "CLFULL-REQ-000001"}]
    pending = pending_requests_in_order(manifest, completed_req, completed_resp)
    assert [row["request_id"] for row in pending] == ["CLFULL-REQ-000002"]
    with pytest.raises(RuntimeError, match="ambiguous"):
        pending_requests_in_order(manifest, [{"request_id": "CLFULL-REQ-000001"}], [])
    with pytest.raises(RuntimeError, match="invalid_duplicate"):
        pending_requests_in_order(manifest, [{"request_id": "CLFULL-REQ-000001"}, {"request_id": "CLFULL-REQ-000001"}], [{"request_id": "CLFULL-REQ-000001"}])


def test_execute_pending_respects_max_new_requests_and_append_only(tmp_path, monkeypatch):
    manifest = frame([_manifest_row("CLFULL-REQ-000001", 1), _manifest_row("CLFULL-REQ-000002", 2)], MANIFEST_COLUMNS)
    inputs = _inputs(manifest)

    def fake_execute(_payload, _timeout):
        return SimpleNamespace(
            backend_success=True,
            response_text='{"ranked_candidate_indices":[1,2,3,4,5]}',
            raw_backend_response={"done": True, "eval_count": 5},
            backend_error_type="",
            backend_error_message="",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:01Z",
            duration_seconds=1.0,
        )

    monkeypatch.setattr(runner, "execute_ollama_request", fake_execute)
    sent = execute_pending_requests(tmp_path, inputs, 1)
    assert sent == 1
    assert len(read_jsonl(tmp_path / runner.REQUEST_JSONL)) == 1
    assert len(read_jsonl(tmp_path / runner.RESPONSE_JSONL)) == 1
    assert len(read_jsonl(tmp_path / runner.EVENT_JSONL)) == 2
    sent = execute_pending_requests(tmp_path, inputs, 1)
    assert sent == 1
    assert len(read_jsonl(tmp_path / runner.REQUEST_JSONL)) == 2
    assert len(read_jsonl(tmp_path / runner.RESPONSE_JSONL)) == 2


def test_strict_task_parsers_accept_valid_outputs_and_reject_invalid_forms():
    assert classify_response_for_task('{"ranked_candidate_indices":[1,2,3,4,5]}', "ranking")["strict_parse_valid"] is True
    assert classify_response_for_task('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}', "reranking")["strict_parse_valid"] is True
    assert classify_response_for_task('{"ranked_candidate_indices":[1,2,3,4]}', "ranking")["primary_failure_reason"] == "output_count_invalid"
    assert classify_response_for_task('{"ranked_candidate_indices":[1,2,3,4,4]}', "ranking")["primary_failure_reason"] == "duplicate_indices"
    assert classify_response_for_task('{"ranked_candidate_indices":[1,2,3,4,21]}', "ranking")["primary_failure_reason"] == "index_range_invalid"
    assert classify_response_for_task('Here: {"ranked_candidate_indices":[1,2,3,4,5]}', "ranking")["primary_failure_reason"] == "json_parse_failed"
    assert classify_response_for_task('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}', "reranking")["primary_failure_reason"] == "output_count_invalid"


def _metric_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_rows = []
    execution_rows = []
    parse_rows = []
    ranking_rows = []
    order = 1
    for model in runner.MODEL_ORDER:
        for task in runner.TASK_ORDER:
            for dataset in runner.DATASET_ORDER:
                row = _manifest_row(f"CLFULL-REQ-{order:06d}", order, task)
                row["model_family"] = model
                row["model_tag"] = runner.EXPECTED_DIGESTS[model]["tag"]
                row["dataset"] = dataset
                manifest_rows.append(row)
                execution_rows.append({
                    "request_id": row["request_id"],
                    "model_family": model,
                    "model_tag": row["model_tag"],
                    "dataset": dataset,
                    "task_type": task,
                    "prompt_contract_id": row["prompt_contract_id"],
                    "backend_success": "true",
                    "response_jsonl_recorded": "true",
                    "duration_seconds": "1",
                    "eval_count": "5",
                })
                parse_rows.append({
                    "request_id": row["request_id"],
                    "model_family": model,
                    "dataset": dataset,
                    "task_type": task,
                    "prompt_contract_id": row["prompt_contract_id"],
                    "strict_parse_valid": "true",
                    "json_parse_valid": "true",
                    "schema_valid": "true",
                    "output_count_valid": "true",
                    "uniqueness_valid": "true",
                    "candidate_validity": "true",
                    "duplicate_count": "0",
                    "hallucination_count": "0",
                })
                ranking_rows.append({
                    "request_id": row["request_id"],
                    "positive_in_top_5": "true",
                    "ndcg_at_5": "1",
                    "positive_rank_zero_based": "0",
                    "utility_evaluation_valid": "true",
                })
                order += 1
    return (
        frame(manifest_rows, MANIFEST_COLUMNS),
        pd.DataFrame(execution_rows),
        pd.DataFrame(parse_rows),
        pd.DataFrame(ranking_rows),
    )


def test_cell_metrics_has_12_cells_and_valid_only_utility():
    manifest, execution, parse_results, rankings = _metric_frames()
    metrics = build_cell_metrics(manifest, execution, parse_results, rankings)
    assert list(metrics.columns) == CELL_METRICS_COLUMNS
    assert len(metrics) == 12
    assert metrics["valid_only_hr_at_5"].eq("1").all()
    assert metrics["valid_only_ndcg_at_5"].eq("1").all()


def test_decision_complete_requires_accounted_records_and_ignores_strict_invalid_for_completeness(monkeypatch):
    monkeypatch.setattr(runner, "EXPECTED_REQUESTS", 2)
    inputs = _inputs(frame([_manifest_row("CLFULL-REQ-000001", 1), _manifest_row("CLFULL-REQ-000002", 2)], MANIFEST_COLUMNS))
    request_records = [{"request_id": "CLFULL-REQ-000001", "attempt_index": 1}, {"request_id": "CLFULL-REQ-000002", "attempt_index": 1}]
    response_records = [{"request_id": "CLFULL-REQ-000001", "attempt_index": 1}, {"request_id": "CLFULL-REQ-000002", "attempt_index": 1}]
    event_records = [{"request_id": "CLFULL-REQ-000001"}, {"request_id": "CLFULL-REQ-000001"}, {"request_id": "CLFULL-REQ-000002"}, {"request_id": "CLFULL-REQ-000002"}]
    frames = {
        "request_state": pd.DataFrame({
            "request_id": ["CLFULL-REQ-000001", "CLFULL-REQ-000002"],
            "request_state": ["completed", "completed"],
            "request_sent": ["true", "true"],
            "request_order": ["1", "2"],
        }),
        "parse_results": pd.DataFrame({"request_id": ["CLFULL-REQ-000001", "CLFULL-REQ-000002"], "strict_parse_valid": ["false", "true"]}),
        "execution": pd.DataFrame({"response_jsonl_recorded": ["true", "true"], "backend_success": ["true", "true"]}),
        "cell_metrics": pd.DataFrame([{"x": i} for i in range(12)]),
    }
    decision = build_decision(inputs, request_records, response_records, event_records, frames, protected_unchanged=True)
    assert decision.iloc[0]["execution_decision"] == "full_matrix_execution_complete_ready_for_evaluation"
    assert decision.iloc[0]["strict_invalid_output_count"] == "1"
    incomplete = build_decision(inputs, request_records, response_records[:1], event_records[:2], frames, protected_unchanged=True)
    assert incomplete.iloc[0]["execution_decision"] == "full_matrix_execution_incomplete"

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.llm import canonical_local_llm_pilot_resume as resume
from src.llm import canonical_local_llm_runner as runner
from src.llm.canonical_llm_contract import (
    EXECUTION_PARAMETER_COLUMNS,
    contract_hash,
    json_dumps,
    parser_contract_rows,
    prompt_contract_rows,
    sha256_items,
    sha256_text,
)


def _execution_rows() -> list[dict[str, str]]:
    rows = []
    for family, _tag in runner.MODEL_ORDER:
        for task_type, max_tokens in [("ranking", "256"), ("reranking", "512")]:
            row = {
                "execution_contract_id": f"exec_{family.lower().replace(' ', '_')}_{task_type}_v1",
                "model_family": family,
                "task_type": task_type,
                "temperature": "0",
                "top_p": "",
                "top_k": "",
                "seed": "",
                "max_output_tokens": max_tokens,
                "context_limit": "",
                "stop_sequences": "[]",
                "response_format": "json_object",
                "streaming": "false",
                "concurrency": "1",
                "timeout_seconds": "120",
                "infrastructure_retry_count": "0",
                "scientific_retry_count": "0",
                "request_order_policy": "deterministic_dataset_task_user_order",
                "raw_response_retention": "required",
                "parameter_source": "synthetic test",
                "paper_alignment": "adapted_project_policy",
                "contract_status": "ready_adapted",
                "blocking_reason": "",
                "notes": "test",
            }
            row["contract_sha256"] = contract_hash(row)
            rows.append({column: row.get(column, "") for column in EXECUTION_PARAMETER_COLUMNS})
    return rows


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(
        prompt_contract=pd.DataFrame(prompt_contract_rows()),
        parser_contract=pd.DataFrame(parser_contract_rows()),
        execution_contract=pd.DataFrame(_execution_rows()),
    )


def _ready_prompt_by_task(bundle: SimpleNamespace) -> dict[str, dict[str, str]]:
    return {
        row["task_type"]: row
        for row in bundle.prompt_contract.to_dict("records")
        if row["contract_status"] == "ready_adapted"
    }


def _strict_parser_by_task(bundle: SimpleNamespace) -> dict[str, dict[str, str]]:
    return {
        row["task_type"]: row
        for row in bundle.parser_contract.to_dict("records")
        if row["parser_mode"] == "strict_primary_parser"
    }


def _exec_by_cell(bundle: SimpleNamespace) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["model_family"], row["task_type"]): row
        for row in bundle.execution_contract.to_dict("records")
    }


def _sample_frame() -> pd.DataFrame:
    rows = []
    order = 1
    for dataset, task_type in runner.STRATUM_ORDER:
        prefix = dataset.lower().replace(" ", "_").replace("-", "_")
        for index in range(10):
            candidate_set_id = f"{prefix}_{task_type}_cs{index:02d}"
            user_id = f"{prefix}_{task_type}_u{index:02d}"
            positive_item_id = f"{prefix}_{task_type}_item00"
            rows.append(
                {
                    "pilot_row_id": f"CLLP-{order:04d}",
                    "selection_sha256": runner.selection_key(dataset, task_type, candidate_set_id, user_id),
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": candidate_set_id,
                    "source_path": "synthetic.csv",
                    "user_id": user_id,
                    "positive_item_id": positive_item_id,
                    "candidate_count": "20",
                    "history_item_count": "10",
                    "candidate_titles_complete": "true",
                    "history_titles_complete": "true",
                    "held_out_positive_absent": "true",
                    "primary_prompt_compatible": "true",
                    "prompt_contract_id": "prompt",
                    "parser_contract_id": "parser",
                    "candidate_order_sha256": "a" * 64,
                    "system_message_sha256": "b" * 64,
                    "user_message_sha256": "c" * 64,
                    "complete_prompt_sha256": "d" * 64,
                    "sample_order": str(order),
                    "sample_status": "selected",
                    "blocking_reason": "",
                    "notes": "synthetic",
                }
            )
            order += 1
    return pd.DataFrame(rows, columns=runner.SAMPLE_COLUMNS)


def _manifest_frame(sample: pd.DataFrame, bundle: SimpleNamespace) -> pd.DataFrame:
    prompt_by_task = _ready_prompt_by_task(bundle)
    parser_by_task = _strict_parser_by_task(bundle)
    exec_by_cell = _exec_by_cell(bundle)
    rows = []
    order = 1
    for family, tag in runner.MODEL_ORDER:
        for sample_row in sample.sort_values("sample_order", key=lambda s: s.astype(int)).to_dict("records"):
            task_type = sample_row["task_type"]
            prompt = prompt_by_task[task_type]
            parser = parser_by_task[task_type]
            execution = exec_by_cell[(family, task_type)]
            rows.append(
                {
                    "request_id": f"CLLP-REQ-{order:04d}",
                    "pilot_row_id": sample_row["pilot_row_id"],
                    "model_family": family,
                    "model_tag": tag,
                    "model_manifest_or_digest": "",
                    "dataset": sample_row["dataset"],
                    "task_type": task_type,
                    "candidate_set_id": sample_row["candidate_set_id"],
                    "user_id": sample_row["user_id"],
                    "request_order": str(order),
                    "prompt_contract_id": prompt["prompt_contract_id"],
                    "prompt_contract_sha256": prompt["contract_sha256"],
                    "parser_contract_id": parser["parser_contract_id"],
                    "parser_schema_sha256": parser["parser_schema_sha256"],
                    "execution_contract_id": execution["execution_contract_id"],
                    "execution_contract_sha256": execution["contract_sha256"],
                    "complete_prompt_sha256": sample_row["complete_prompt_sha256"],
                    "candidate_order_sha256": sample_row["candidate_order_sha256"],
                    "requested_output_count": "5" if task_type == "ranking" else "20",
                    "temperature": "0",
                    "seed": "",
                    "max_output_tokens": "256" if task_type == "ranking" else "512",
                    "response_format": "json_object",
                    "streaming": "false",
                    "concurrency": "1",
                    "scientific_retry_count": "0",
                    "infrastructure_retry_count": "0",
                    "request_jsonl_path": "results/llm/canonical_local_pilot/requests.jsonl",
                    "response_jsonl_path": "results/llm/canonical_local_pilot/responses.jsonl",
                    "request_status": "blocked_pre_execution",
                    "blocking_reason": "backend_blocked",
                    "notes": "synthetic",
                }
            )
            order += 1
    return pd.DataFrame(rows, columns=runner.REQUEST_MANIFEST_COLUMNS)


def _contexts(sample: pd.DataFrame) -> dict[str, dict[str, object]]:
    contexts = {}
    for row in sample.to_dict("records"):
        candidate_items = [f"{row['pilot_row_id']}_item{i:02d}" for i in range(20)]
        positive = candidate_items[0]
        contexts[row["pilot_row_id"]] = {
            **row,
            "candidate_items": candidate_items,
            "positive_item_id": positive,
            "candidate_order_sha256": sha256_items(candidate_items),
            "requested_output_count": "5" if row["task_type"] == "ranking" else "20",
            "requires_full_permutation": row["task_type"] == "reranking",
            "system_message": "system",
            "user_message": "rank candidates",
        }
    return contexts


def _response_text(task_type: str, *, valid: bool = True) -> str:
    if not valid:
        return "not json"
    if task_type == "ranking":
        return '{"ranked_candidate_indices":[1,2,3,4,5]}'
    return '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}'


def _write_raw_records(root: Path, manifest: pd.DataFrame, *, first_invalid: bool = False) -> None:
    request_path = root / resume.REQUEST_JSONL
    response_path = root / resume.RESPONSE_JSONL
    request_path.parent.mkdir(parents=True, exist_ok=True)
    requests = []
    responses = []
    for index, row in enumerate(manifest.to_dict("records")):
        requests.append({"request_id": row["request_id"], "attempt_index": 1, "request_order": row["request_order"]})
        valid = not (first_invalid and index == 0)
        text = _response_text(row["task_type"], valid=valid)
        responses.append(
            {
                "request_id": row["request_id"],
                "attempt_index": 1,
                "backend_success": "true",
                "raw_response_text": text,
                "raw_response_sha256": sha256_text(text),
                "model_reported_tag": row["model_tag"],
                "backend_metadata": {"done": True, "eval_count": 5},
                "done": "True",
                "done_reason": "done",
                "prompt_eval_count": "10",
                "generated_token_count": "5",
                "load_duration_nanoseconds": "1",
                "prompt_eval_duration_nanoseconds": "2",
                "eval_duration_nanoseconds": "3",
                "total_duration_nanoseconds": "4",
                "request_started_at_utc": "2026-01-01T00:00:00.000Z",
                "timestamp_after_completion_utc": "2026-01-01T00:00:01.000Z",
                "duration_seconds": "1",
                "backend_error_type": "",
                "backend_error_message": "",
            }
        )
    request_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in requests), encoding="utf-8")
    response_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in responses), encoding="utf-8")


def _inputs(sample: pd.DataFrame, manifest: pd.DataFrame) -> resume.ResumeInputs:
    return resume.ResumeInputs(sample, manifest, pd.DataFrame(), pd.DataFrame(), _contexts(sample), {})


def test_frozen_sample_validation_accepts_cardinality_and_rejects_hash_or_order_mismatch():
    sample = _sample_frame()
    resume.validate_frozen_sample(sample)

    with pytest.raises(ValueError, match="row count"):
        resume.validate_frozen_sample(sample.iloc[:-1].copy())

    corrupted_hash = sample.copy()
    corrupted_hash.loc[0, "selection_sha256"] = "bad"
    with pytest.raises(ValueError, match="selection hash"):
        resume.validate_frozen_sample(corrupted_hash)

    corrupted_order = sample.copy()
    corrupted_order.loc[0, "sample_order"] = "2"
    with pytest.raises(ValueError, match="sample_order"):
        resume.validate_frozen_sample(corrupted_order)


def test_frozen_manifest_validation_accepts_cardinality_and_rejects_order_or_model_tag_mismatch():
    sample = _sample_frame()
    bundle = _bundle()
    manifest = _manifest_frame(sample, bundle)
    resume.validate_frozen_manifest(manifest, sample, bundle)

    with pytest.raises(ValueError, match="row count"):
        resume.validate_frozen_manifest(manifest.iloc[:-1].copy(), sample, bundle)

    bad_tag = manifest.copy()
    bad_tag.loc[0, "model_tag"] = "qwen2:wrong"
    with pytest.raises(ValueError, match="model tag"):
        resume.validate_frozen_manifest(bad_tag, sample, bundle)

    bad_order = manifest.copy()
    bad_order.loc[0, "request_order"] = "2"
    with pytest.raises(ValueError, match="request_order"):
        resume.validate_frozen_manifest(bad_order, sample, bundle)


def test_resume_state_classification_pending_completed_ambiguous_and_duplicate_evidence():
    sample = _sample_frame()
    manifest = _manifest_frame(sample, _bundle())

    assert resume.state_summary(manifest, [], [])["not_started"] == 120
    pending = resume.pending_requests_in_order(manifest, [], [])
    assert pending[0]["request_id"] == "CLLP-REQ-0001"
    assert len(pending) == 120

    request = [{"request_id": "CLLP-REQ-0001"}]
    response = [{"request_id": "CLLP-REQ-0001"}]
    pending_after_complete = resume.pending_requests_in_order(manifest, request, response)
    assert pending_after_complete[0]["request_id"] == "CLLP-REQ-0002"

    with pytest.raises(RuntimeError, match="ambiguous_interrupted_request"):
        resume.pending_requests_in_order(manifest, request, [])
    with pytest.raises(RuntimeError, match="invalid_duplicate_evidence"):
        resume.pending_requests_in_order(manifest, [], response)
    with pytest.raises(RuntimeError, match="invalid_duplicate_evidence"):
        resume.pending_requests_in_order(manifest, request * 2, response)
    with pytest.raises(RuntimeError, match="invalid_duplicate_evidence"):
        resume.pending_requests_in_order(manifest, [{"request_id": "EXTRA"}], [])


def test_append_jsonl_fsync_is_utf8_append_only_and_does_not_truncate(tmp_path):
    path = tmp_path / "records.jsonl"
    resume.append_jsonl_fsync(path, {"request_id": "r1", "text": "caf\u00e9"})
    first_bytes = path.read_bytes()
    resume.append_jsonl_fsync(path, {"request_id": "r2", "text": "ok"})

    assert path.read_bytes().startswith(first_bytes)
    assert len(resume.read_jsonl(path)) == 2
    assert resume.read_jsonl(path)[0]["text"] == "caf\u00e9"


def test_request_record_preserves_payload_and_rejects_positive_exposure():
    row = _manifest_frame(_sample_frame(), _bundle()).iloc[0].to_dict()
    context = {"positive_item_id": "POSITIVE", "system_message": "system", "user_message": "prompt"}
    payload = {"model": row["model_tag"], "system": "system", "prompt": "rank only", "stream": False, "format": "json"}

    record = resume.build_request_record(row, context, payload, "digest")

    assert record["request_payload"] == payload
    assert record["scientific_retry_count"] == "0"
    assert record["infrastructure_retry_count"] == "0"
    with pytest.raises(ValueError, match="positive label"):
        resume.build_request_record(row, context, {**payload, "prompt": "POSITIVE"}, "digest")
    assert not resume.positive_label_exposed("Candidate 1. Movie from 1986", "86")
    assert resume.positive_label_exposed("Use the positive_item_id key", "86")


def test_strict_and_salvage_parsing_reuse_contract_without_substitution():
    candidates = [f"item{i}" for i in range(1, 21)]
    valid_ranking = runner.classify_response(
        '{"ranked_candidate_indices":[1,2,3,4,5]}',
        candidate_items=candidates,
        expected_count=5,
        require_full_permutation=False,
    )
    valid_reranking = runner.classify_response(
        '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}',
        candidate_items=candidates,
        expected_count=20,
        require_full_permutation=True,
    )
    invalid = runner.classify_response("not json", candidate_items=candidates, expected_count=5, require_full_permutation=False)
    duplicate = runner.classify_response('{"ranked_candidate_indices":[1,1,2,3,4]}', candidate_items=candidates, expected_count=5, require_full_permutation=False)
    partial = runner.classify_response('{"ranked_candidate_indices":[1,2,3,4,5]}', candidate_items=candidates, expected_count=20, require_full_permutation=True)
    salvage = runner.classify_response(
        'Answer:\n```json\n{"ranked_candidate_indices":["1","2","3","4","5"]}\n```',
        candidate_items=candidates,
        expected_count=5,
        require_full_permutation=False,
    )

    assert valid_ranking["strict_parse_valid"] is True
    assert valid_reranking["full_permutation_valid"] is True
    assert invalid["primary_failure_reason"] == "json_parse_failed"
    assert duplicate["primary_failure_reason"] == "duplicate_indices"
    assert partial["primary_failure_reason"] == "output_count_invalid"
    assert salvage["strict_parse_valid"] is False
    assert salvage["diagnostic_salvage_valid"] is True


def test_build_result_tables_valid_only_metrics_and_no_invalid_zero_fabrication(tmp_path):
    sample = _sample_frame()
    manifest = _manifest_frame(sample, _bundle())
    _write_raw_records(tmp_path, manifest, first_invalid=True)

    frames = resume.build_all_result_tables(tmp_path, _inputs(sample, manifest))

    assert len(frames["execution"]) == 120
    assert len(frames["parse_results"]) == 120
    assert len(frames["rankings"]) == 120
    assert len(frames["metrics"]) == 12
    invalid_ranking = frames["rankings"].loc[frames["rankings"]["request_id"].eq("CLLP-REQ-0001")].iloc[0]
    assert invalid_ranking["utility_evaluation_valid"] == "false"
    assert invalid_ranking["ndcg_at_5"] == ""
    cell = frames["metrics"].loc[
        frames["metrics"]["model_family"].eq("Qwen2 7B Instruct")
        & frames["metrics"]["dataset"].eq("Amazon Beauty")
        & frames["metrics"]["task_type"].eq("ranking")
    ].iloc[0]
    assert cell["pilot_rows"] == "10"
    assert cell["strict_valid_rows"] == "9"
    assert cell["utility_evaluated_rows"] == "9"
    assert cell["hr_at_5_valid_only"] == "1"


def test_decision_gates_threshold_ambiguity_candidate_validity_and_permutation(tmp_path):
    sample = _sample_frame()
    manifest = _manifest_frame(sample, _bundle())
    _write_raw_records(tmp_path, manifest, first_invalid=True)
    frames = resume.build_all_result_tables(tmp_path, _inputs(sample, manifest))

    approved = resume.build_decision(**frames, protected_unchanged=True, digest_ok=True)

    assert approved.iloc[0]["pilot_decision"] == "approve_full_local_execution"
    assert approved.iloc[0]["readiness_threshold"] == "0.80000000000000004"
    assert approved.iloc[0]["recommended_single_next_task"].startswith("Phase 5B-P2")

    ambiguous_state = frames["request_state"].copy()
    ambiguous_state.loc[0, "request_state"] = "ambiguous_interrupted"
    assert resume.build_decision(ambiguous_state, frames["execution"], frames["parse_results"], frames["rankings"], frames["metrics"], True, True).iloc[0]["pilot_decision"] == "ambiguous_interrupted_request"

    missing_response_state = frames["request_state"].copy()
    missing_response_state.loc[0, "response_record_count"] = "0"
    assert resume.build_decision(missing_response_state, frames["execution"], frames["parse_results"], frames["rankings"], frames["metrics"], True, True).iloc[0]["pilot_decision"] == "pilot_incomplete"

    candidate_invalid = frames["parse_results"].copy()
    candidate_invalid.loc[candidate_invalid["strict_parse_valid"].eq("true").idxmax(), "candidate_validity"] = "false"
    assert resume.build_decision(frames["request_state"], frames["execution"], candidate_invalid, frames["rankings"], frames["metrics"], True, True).iloc[0]["pilot_decision"] == "require_parser_contract_revision"

    permutation_invalid = frames["parse_results"].copy()
    rerank_index = permutation_invalid.loc[
        permutation_invalid["strict_parse_valid"].eq("true") & permutation_invalid["task_type"].eq("reranking")
    ].index[0]
    permutation_invalid.loc[rerank_index, "full_permutation_valid"] = "false"
    assert resume.build_decision(frames["request_state"], frames["execution"], permutation_invalid, frames["rankings"], frames["metrics"], True, True).iloc[0]["pilot_decision"] == "require_parser_contract_revision"

    low_threshold = frames["metrics"].copy()
    low_threshold.loc[0, "strict_parse_validity_rate"] = "0.79"
    assert resume.build_decision(frames["request_state"], frames["execution"], frames["parse_results"], frames["rankings"], low_threshold, True, True).iloc[0]["pilot_decision"] == "require_prompt_contract_revision"

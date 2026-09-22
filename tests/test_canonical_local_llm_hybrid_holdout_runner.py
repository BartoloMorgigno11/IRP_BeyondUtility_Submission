from types import SimpleNamespace

import pandas as pd
import pytest

from src.llm.canonical_llm_contract import contract_hash
from src.llm.canonical_local_llm_hybrid_holdout_runner import (
    DECISION_COLUMNS,
    EXECUTION_COLUMNS,
    HOLDOUT_SAMPLE_PATH,
    HYBRID_POLICY_ID,
    MANIFEST_COLUMNS,
    METRIC_COLUMNS,
    MISTRAL_RERANK_PROMPT,
    PARSE_COLUMNS,
    POLICY_COLUMNS,
    QWEN_RERANK_PROMPT,
    RANKING_COLUMNS,
    RANKING_PROMPT_V1,
    READINESS_THRESHOLD,
    RERANK_STRICT_PARSER,
    SAMPLE_COLUMNS,
    append_jsonl_fsync,
    build_decision,
    build_metrics,
    build_parse_and_rankings,
    classify_request_state,
    frame,
    pending_requests_in_order,
    read_jsonl,
    validate_manifest,
    validate_policy,
    validate_sample,
)


VALID_20 = '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}'
SHORT_19 = '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]}'
DUPLICATE = '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,19]}'
OUT_OF_RANGE = '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21]}'
MALFORMED = '{"ranked_candidate_indices":[1,2]'


def _policy(**overrides):
    row = {
        "hybrid_policy_id": HYBRID_POLICY_ID,
        "policy_version": "hybrid_prompt_policy_v1",
        "execution_scope": "disjoint_holdout_pilot_only",
        "qwen_ranking_prompt_contract_id": RANKING_PROMPT_V1,
        "qwen_reranking_prompt_contract_id": QWEN_RERANK_PROMPT,
        "mistral_ranking_prompt_contract_id": RANKING_PROMPT_V1,
        "mistral_reranking_prompt_contract_id": MISTRAL_RERANK_PROMPT,
        "shared_parser_contract_id": "llm_rank_strict_primary_parser_v1;llm_rerank_strict_primary_parser_v1",
        "candidate_representation": "one_based_index_plus_title",
        "candidate_numbering": "one_based_1_to_20",
        "candidate_order_policy": "preserve_source_candidate_order_for_ranking_and_initial_candidate_order_for_reranking",
        "history_policy": "train_split_only_chronological_last_10_exclude_heldout_positive",
        "positive_label_hidden": "true",
        "ranking_output_count": "5",
        "reranking_output_count": "20",
        "full_permutation_required": "true",
        "execution_parameters_shared": "true",
        "temperature": "0",
        "scientific_retry_count": "0",
        "infrastructure_retry_count": "0",
        "concurrency": "1",
        "selection_basis": "structural_validity_based",
        "independent_holdout_required": "true",
        "paper_alignment": "adapted_model_specific_format_compliance_contract",
        "contract_status": "approved_for_disjoint_holdout_pilot",
        "blocking_reason": "",
        "notes": "test",
    }
    row.update(overrides)
    if "contract_sha256" not in row:
        row["contract_sha256"] = contract_hash(row)
    return frame([row], POLICY_COLUMNS)


def _sample(overlap=False, task="reranking"):
    rows = []
    order = 1
    for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]:
        for index in range(10):
            rows.append(
                {
                    "holdout_row_id": f"ROW-{order:04d}",
                    "selection_sha256": f"sel-{order:04d}",
                    "dataset": dataset,
                    "task_type": task,
                    "candidate_set_id": f"{dataset}-cs",
                    "source_path": f"{dataset}.csv",
                    "user_id": f"user-{order:04d}",
                    "positive_item_id": f"pos-{order:04d}",
                    "candidate_count": "20",
                    "history_item_count": "5",
                    "candidate_titles_complete": "true",
                    "history_titles_complete": "true",
                    "held_out_positive_absent": "true",
                    "primary_prompt_compatible": "true",
                    "used_in_v1_pilot": "false",
                    "used_in_v2_pilot": "false",
                    "disjoint_from_previous_pilots": "true",
                    "candidate_order_sha256": f"cand-{order:04d}",
                    "history_render_sha256": f"hist-{order:04d}",
                    "sample_order": str(order),
                    "sample_status": "selected_frozen_not_executed",
                    "blocking_reason": "",
                    "notes": "test",
                }
            )
            order += 1
    if overlap:
        rows[0]["used_in_v1_pilot"] = "true"
        rows[0]["disjoint_from_previous_pilots"] = "false"
    return frame(rows, SAMPLE_COLUMNS)


def _manifest(sample=None, wrong_qwen=False, wrong_mistral=False, ranking=False):
    sample = sample if sample is not None else _sample()
    rows = []
    order = 1
    assignments = [
        ("Qwen2 7B Instruct", "qwen2:7b-instruct", MISTRAL_RERANK_PROMPT if wrong_qwen else QWEN_RERANK_PROMPT),
        ("Mistral 7B Instruct", "mistral:7b-instruct", QWEN_RERANK_PROMPT if wrong_mistral else MISTRAL_RERANK_PROMPT),
    ]
    for model, tag, prompt in assignments:
        for sample_row in sample.to_dict("records"):
            rows.append(
                {
                    "holdout_request_id": f"REQ-{order:04d}",
                    "holdout_row_id": sample_row["holdout_row_id"],
                    "model_family": model,
                    "model_tag": tag,
                    "dataset": sample_row["dataset"],
                    "task_type": "ranking" if ranking and order == 1 else sample_row["task_type"],
                    "candidate_set_id": sample_row["candidate_set_id"],
                    "user_id": sample_row["user_id"],
                    "request_order": str(order),
                    "hybrid_policy_id": HYBRID_POLICY_ID,
                    "prompt_contract_id": prompt,
                    "prompt_contract_sha256": f"prompt-{prompt}",
                    "parser_contract_id": RERANK_STRICT_PARSER,
                    "parser_contract_sha256": "parser-hash",
                    "execution_contract_id": "exec-qwen" if model.startswith("Qwen") else "exec-mistral",
                    "execution_contract_sha256": "exec-hash",
                    "candidate_order_sha256": sample_row["candidate_order_sha256"],
                    "history_render_sha256": sample_row["history_render_sha256"],
                    "requested_output_count": "20",
                    "temperature": "0",
                    "seed": "",
                    "max_output_tokens": "512",
                    "response_format": "json_object",
                    "streaming": "false",
                    "concurrency": "1",
                    "scientific_retry_count": "0",
                    "infrastructure_retry_count": "0",
                    "previous_pilot_overlap": "false",
                    "holdout_request_status": "frozen_not_executed",
                    "blocking_reason": "",
                    "notes": "test",
                }
            )
            order += 1
    return frame(rows, MANIFEST_COLUMNS)


def _previous_empty():
    return pd.DataFrame(columns=["dataset", "task_type", "candidate_set_id", "user_id"])


def test_policy_validation_accepts_valid_and_rejects_wrong_prompts_paper_exact_and_hash():
    assert validate_policy(_policy())["hybrid_policy_id"] == HYBRID_POLICY_ID
    with pytest.raises(ValueError, match="qwen_reranking"):
        validate_policy(_policy(qwen_reranking_prompt_contract_id=MISTRAL_RERANK_PROMPT))
    with pytest.raises(ValueError, match="mistral_reranking"):
        validate_policy(_policy(mistral_reranking_prompt_contract_id=QWEN_RERANK_PROMPT))
    with pytest.raises(ValueError, match="paper_alignment"):
        validate_policy(_policy(paper_alignment="paper_exact"))
    bad = _policy()
    bad.loc[0, "contract_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        validate_policy(bad)


def test_sample_validation_requires_30_unique_reranking_rows_ten_per_dataset_and_no_overlap():
    sample = _sample()
    assert validate_sample(sample, _previous_empty(), _previous_empty()) == 0
    with pytest.raises(ValueError, match="reranking"):
        validate_sample(_sample(task="ranking"), _previous_empty(), _previous_empty())
    with pytest.raises(ValueError, match="disjoint|overlap|previous-pilot"):
        validate_sample(_sample(overlap=True), _previous_empty(), _previous_empty())
    previous = pd.DataFrame([sample.iloc[0].to_dict()])
    with pytest.raises(ValueError, match="overlaps"):
        validate_sample(sample, previous, _previous_empty())


def test_manifest_validation_requires_60_requests_same_rows_assignments_order_and_no_ranking():
    sample = _sample()
    validate_manifest(_manifest(sample), sample)
    with pytest.raises(ValueError, match="prompt assignment"):
        validate_manifest(_manifest(sample, wrong_qwen=True), sample)
    with pytest.raises(ValueError, match="prompt assignment"):
        validate_manifest(_manifest(sample, wrong_mistral=True), sample)
    with pytest.raises(ValueError, match="ranking|non-reranking"):
        validate_manifest(_manifest(sample, ranking=True), sample)
    bad = _manifest(sample)
    bad.loc[0, "request_order"] = "99"
    with pytest.raises(ValueError, match="request_order"):
        validate_manifest(bad, sample)


def test_resume_state_pending_completed_ambiguous_duplicate_and_append_only(tmp_path):
    assert classify_request_state(0, 0) == "not_started"
    assert classify_request_state(1, 1) == "completed"
    assert classify_request_state(1, 0) == "ambiguous_interrupted"
    assert classify_request_state(2, 1) == "invalid_duplicate_evidence"

    manifest = _manifest(_sample()).head(2).copy()
    assert len(pending_requests_in_order(manifest, [], [])) == 2
    completed = [{"holdout_request_id": "REQ-0001"}]
    assert [row["holdout_request_id"] for row in pending_requests_in_order(manifest, completed, completed)] == ["REQ-0002"]
    with pytest.raises(RuntimeError, match="ambiguous"):
        pending_requests_in_order(manifest, completed, [])
    with pytest.raises(RuntimeError, match="invalid_duplicate"):
        pending_requests_in_order(manifest, completed * 2, completed)

    path = tmp_path / "records.jsonl"
    append_jsonl_fsync(path, {"holdout_request_id": "a"})
    append_jsonl_fsync(path, {"holdout_request_id": "b"})
    assert [row["holdout_request_id"] for row in read_jsonl(path)] == ["a", "b"]


def _inputs_for_outputs(raw_texts):
    sample = _sample().head(len(raw_texts)).copy()
    sample["sample_order"] = [str(i + 1) for i in range(len(sample))]
    manifest = _manifest(sample).head(len(raw_texts)).copy()
    contexts = {}
    for row in manifest.to_dict("records"):
        contexts[row["holdout_request_id"]] = {
            "positive_item_id": "item-1",
            "candidate_items": [f"item-{i}" for i in range(1, 21)],
            "candidate_order_sha256": row["candidate_order_sha256"],
            "history_render_sha256": row["history_render_sha256"],
        }
    return SimpleNamespace(manifest=manifest, contexts=contexts, previous_overlap_count=0)


def _execution_and_responses(inputs, raw_texts):
    responses = []
    exec_rows = []
    for row, text in zip(inputs.manifest.to_dict("records"), raw_texts):
        responses.append(
            {
                "holdout_request_id": row["holdout_request_id"],
                "raw_response_text": text,
                "raw_response_sha256": "sha",
                "done_reason": "stop",
                "generated_token_count": "20",
            }
        )
        exec_rows.append(
            {
                **{column: row.get(column, "") for column in EXECUTION_COLUMNS},
                "holdout_request_id": row["holdout_request_id"],
                "holdout_row_id": row["holdout_row_id"],
                "model_family": row["model_family"],
                "model_tag": row["model_tag"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "prompt_contract_id": row["prompt_contract_id"],
                "attempt_index": "1",
                "duration_seconds": "1",
                "backend_success": "true",
                "response_nonempty": "true",
                "eval_count": "20",
                "response_jsonl_recorded": "true",
            }
        )
    return frame(exec_rows, EXECUTION_COLUMNS), responses


def test_parsing_taxonomy_and_salvage_non_substitution_for_valid_short_duplicate_range_and_malformed():
    inputs = _inputs_for_outputs([VALID_20, SHORT_19, DUPLICATE, OUT_OF_RANGE, MALFORMED])
    execution, responses = _execution_and_responses(inputs, [VALID_20, SHORT_19, DUPLICATE, OUT_OF_RANGE, MALFORMED])
    parse_results, rankings = build_parse_and_rankings(inputs, execution, responses)
    assert parse_results.iloc[0]["strict_parse_valid"] == "true"
    assert parse_results.iloc[1]["primary_failure_reason"] == "output_count_invalid"
    assert parse_results.iloc[2]["primary_failure_reason"] == "duplicate_indices"
    assert parse_results.iloc[3]["primary_failure_reason"] == "index_range_invalid"
    assert parse_results.iloc[4]["primary_failure_reason"] == "json_parse_failed"
    assert rankings.iloc[0]["utility_evaluation_valid"] == "true"
    assert rankings.iloc[1]["utility_evaluation_valid"] == "false"
    assert rankings.iloc[1]["ranked_item_ids"] == "[]"


def _six_cell_inputs(valid_counts):
    rows = []
    contexts = {}
    order = 1
    for model, tag, prompt in [
        ("Qwen2 7B Instruct", "qwen2:7b-instruct", QWEN_RERANK_PROMPT),
        ("Mistral 7B Instruct", "mistral:7b-instruct", MISTRAL_RERANK_PROMPT),
    ]:
        for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]:
            for index in range(10):
                rows.append(
                    {
                        "holdout_request_id": f"REQ-{order:04d}",
                        "holdout_row_id": f"ROW-{order:04d}",
                        "model_family": model,
                        "model_tag": tag,
                        "dataset": dataset,
                        "task_type": "reranking",
                        "candidate_set_id": f"{dataset}-cs",
                        "user_id": f"user-{order:04d}",
                        "request_order": str(order),
                        "prompt_contract_id": prompt,
                        "parser_contract_id": RERANK_STRICT_PARSER,
                        "max_output_tokens": "512",
                        "candidate_order_sha256": f"cand-{order}",
                        "history_render_sha256": f"hist-{order}",
                    }
                )
                contexts[f"REQ-{order:04d}"] = {
                    "positive_item_id": "item-1",
                    "candidate_items": [f"item-{i}" for i in range(1, 21)],
                    "candidate_order_sha256": f"cand-{order}",
                    "history_render_sha256": f"hist-{order}",
                }
                order += 1
    manifest = pd.DataFrame(rows)
    inputs = SimpleNamespace(manifest=manifest, contexts=contexts, previous_overlap_count=0)
    responses = []
    exec_rows = []
    for row in rows:
        cell_key = (row["model_family"], row["dataset"])
        cell_index = len([r for r in responses if r.get("cell_key") == cell_key])
        text = VALID_20 if cell_index < valid_counts.get(cell_key, 10) else SHORT_19
        responses.append({"holdout_request_id": row["holdout_request_id"], "raw_response_text": text, "raw_response_sha256": "sha", "done_reason": "stop", "generated_token_count": "20", "cell_key": cell_key})
        exec_rows.append({**{column: "" for column in EXECUTION_COLUMNS}, **row, "attempt_index": "1", "duration_seconds": "1", "backend_success": "true", "response_nonempty": "true", "eval_count": "20", "response_jsonl_recorded": "true"})
    execution = frame(exec_rows, EXECUTION_COLUMNS)
    parse_results, rankings = build_parse_and_rankings(inputs, execution, responses)
    metrics = build_metrics(inputs, execution, parse_results, rankings)
    return inputs, execution, parse_results, rankings, metrics, responses


def test_metrics_and_decision_approve_only_when_all_six_cells_reach_eight_of_ten():
    inputs, execution, parse_results, rankings, metrics, responses = _six_cell_inputs({})
    assert len(metrics) == 6
    assert metrics["cell_readiness_pass"].eq("true").all()
    decision = build_decision(inputs, execution, parse_results, metrics, [{}] * 60, responses, True)
    assert decision.iloc[0]["pilot_decision"] == "approve_full_local_execution_with_hybrid_prompt_policy"
    assert decision.iloc[0]["utility_used_for_readiness"] == "false"
    assert decision.iloc[0]["readiness_threshold"] == f"{READINESS_THRESHOLD:.12g}"

    fail_counts = {("Qwen2 7B Instruct", "Amazon Beauty"): 7}
    inputs, execution, parse_results, rankings, metrics, responses = _six_cell_inputs(fail_counts)
    decision = build_decision(inputs, execution, parse_results, metrics, [{}] * 60, responses, True)
    assert decision.iloc[0]["pilot_decision"] == "hybrid_policy_failed_disjoint_holdout"
    assert "Qwen2 7B Instruct|Amazon Beauty" in decision.iloc[0]["blocking_cells"]


def test_decision_fails_on_overlap_candidate_invalid_incomplete_permutation_and_truncation():
    inputs, execution, parse_results, rankings, metrics, responses = _six_cell_inputs({})
    inputs_overlap = SimpleNamespace(manifest=inputs.manifest, contexts=inputs.contexts, previous_overlap_count=1)
    decision = build_decision(inputs_overlap, execution, parse_results, metrics, [{}] * 60, responses, True)
    assert decision.iloc[0]["pilot_decision"] == "hybrid_policy_failed_disjoint_holdout"
    parse_bad = parse_results.copy()
    parse_bad.loc[0, "candidate_validity"] = "false"
    metrics_bad = metrics.copy()
    metrics_bad.loc[0, "candidate_valid_rows"] = "9"
    metrics_bad.loc[0, "cell_readiness_pass"] = "false"
    decision = build_decision(inputs, execution, parse_bad, metrics_bad, [{}] * 60, responses, True)
    assert decision.iloc[0]["pilot_decision"] == "hybrid_policy_failed_disjoint_holdout"
    parse_trunc = parse_results.copy()
    parse_trunc.loc[0, "output_truncation_detected"] = "true"
    decision = build_decision(inputs, execution, parse_trunc, metrics, [{}] * 60, responses, True)
    assert decision.iloc[0]["pilot_decision"] == "require_final_reranking_protocol_decision_audit"

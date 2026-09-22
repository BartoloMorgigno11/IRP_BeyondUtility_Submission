from types import SimpleNamespace

import pandas as pd
import pytest

from src.llm.canonical_llm_contract import SYSTEM_MESSAGE_TEMPLATE, contract_hash, json_dumps, sha256_items, sha256_text
from src.llm.canonical_local_llm_reranking_prompt_v2_runner import (
    APPROVED_NEXT_TASK,
    CORRECTIVE_MANIFEST_COLUMNS,
    PROMPT_V2_CONTRACT_COLUMNS,
    P1DInputs,
    append_jsonl_fsync,
    build_decision,
    build_metrics,
    build_prompt_v2_contexts,
    build_v1_v2_comparison,
    classify_request_state,
    compare_validity,
    missing_and_duplicate_indices,
    output_truncation_detected,
    pending_requests_in_order,
    read_jsonl,
    validate_corrective_manifest,
    validate_prompt_v2_contract,
    validate_source_v1_reconciliation,
)
from src.llm.canonical_local_llm_runner import classify_response


def _prompt_v2_row(**overrides):
    row = {
        "prompt_contract_id": "llm_rerank_index_json_full20_v2",
        "supersedes_prompt_contract_id": "llm_rerank_index_json_full20_v1",
        "task_type": "reranking",
        "prompt_version": "v2",
        "system_message_template": SYSTEM_MESSAGE_TEMPLATE,
        "user_message_template": (
            "Task: rerank the supplied 20 candidate items from most to least likely.\n"
            "The input candidate order is the persisted initial ranking to be reranked.\n"
            "Candidate labels are one-based indices from 1 through 20. Do not use item IDs in the answer.\n"
            "Return exactly one JSON object with a single key named \"ranked_candidate_indices\".\n"
            "The value of \"ranked_candidate_indices\" must be an array containing exactly 20 integers.\n"
            "The array must be one complete permutation of candidate labels: every integer from 1 through 20 exactly once, no omitted number and no repeated number.\n"
            "Before returning the answer, verify internally that the array length is 20 and that its set is exactly the integers 1 through 20. Do not describe this verification.\n"
            "Do not include prose, Markdown, a code fence, or any key other than \"ranked_candidate_indices\".\n\n"
            "User history:\n{history_render}\n\nInitial candidate order:\n{candidate_render}"
        ),
        "candidate_representation": "one_based_index_plus_title",
        "candidate_numbering": "one_based_1_to_20",
        "candidate_order_policy": "preserve_source_initial_candidate_order",
        "positive_label_hidden": "true",
        "requested_output_count": "20",
        "requested_output_representation": "strict_json_full_permutation_one_based_candidate_indices",
        "exact_permutation_requirement": "array_set_exactly_1_to_20_no_omissions_no_repeats",
        "internal_validation_instruction": "verify internally that length is 20 and set is exactly integers 1 through 20; do not describe verification",
        "explanation_allowed": "false",
        "code_fence_allowed": "false",
        "scientific_semantics_changed": "false",
        "revision_evidence": "test",
        "paper_alignment": "adapted_project_policy",
        "contract_status": "ready_for_corrective_pilot",
        "notes": "test",
    }
    row.update(overrides)
    row["prompt_template_sha256"] = sha256_text(row["system_message_template"] + "\n---\n" + row["user_message_template"])
    row["contract_sha256"] = contract_hash(row)
    return row


def _prompt_v2_frame(**overrides):
    return pd.DataFrame([_prompt_v2_row(**overrides)], columns=PROMPT_V2_CONTRACT_COLUMNS)


def _manifest(rows=60, include_ranking=False):
    prompt = _prompt_v2_row()
    data = []
    order = 1
    for family, tag in [("Qwen2 7B Instruct", "qwen2:7b-instruct"), ("Mistral 7B Instruct", "mistral:7b-instruct")]:
        for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]:
            for index in range(10):
                task_type = "ranking" if include_ranking and order == 1 else "reranking"
                data.append(
                    {
                        "corrective_request_id": f"CLRPV2-REQ-{order:04d}",
                        "source_request_id_v1": f"CLLP-REQ-{order:04d}",
                        "pilot_row_id": f"CLLP-{order:04d}",
                        "model_family": family,
                        "model_tag": tag,
                        "dataset": dataset,
                        "task_type": task_type,
                        "candidate_set_id": f"cs-{order:04d}",
                        "user_id": f"user-{order:04d}",
                        "source_candidate_order_sha256": sha256_items([f"item-{order}-{i}" for i in range(20)]),
                        "source_history_render_sha256": sha256_text(f"history-{order}"),
                        "prompt_v2_contract_id": prompt["prompt_contract_id"],
                        "prompt_v2_contract_sha256": prompt["contract_sha256"],
                        "parser_contract_id": "llm_rerank_strict_primary_parser_v1",
                        "execution_contract_id": f"exec_{family.lower().replace(' ', '_')}_reranking_v1",
                        "requested_output_count": "20",
                        "temperature": "0",
                        "seed": "",
                        "max_output_tokens": "512",
                        "response_format": "json_object",
                        "streaming": "false",
                        "concurrency": "1",
                        "scientific_retry_count": "0",
                        "infrastructure_retry_count": "0",
                        "corrective_request_order": str(order),
                        "corrective_scope_status": "planned_not_executed",
                        "blocking_reason": "",
                        "notes": "test",
                    }
                )
                order += 1
                if len(data) == rows:
                    return pd.DataFrame(data, columns=CORRECTIVE_MANIFEST_COLUMNS)
    return pd.DataFrame(data, columns=CORRECTIVE_MANIFEST_COLUMNS)


def _source_inputs(manifest):
    v1_rows = []
    contexts = {}
    for row in manifest.to_dict("records"):
        candidate_items = [f"item-{int(row['corrective_request_order'])}-{i}" for i in range(20)]
        history = f"history-{int(row['corrective_request_order'])}"
        contexts[row["pilot_row_id"]] = {
            "candidate_items": candidate_items,
            "candidate_titles": [f"Title {i}" for i in range(20)],
            "candidate_render": "\n".join(f"{i+1}. Title {i}" for i in range(20)),
            "history_render": history,
            "positive_item_id": candidate_items[0],
            "candidate_order_sha256": row["source_candidate_order_sha256"],
        }
        v1_rows.append(
            {
                "request_id": row["source_request_id_v1"],
                "pilot_row_id": row["pilot_row_id"],
                "model_family": row["model_family"],
                "model_tag": row["model_tag"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "candidate_set_id": row["candidate_set_id"],
                "user_id": row["user_id"],
                "candidate_order_sha256": row["source_candidate_order_sha256"],
                "parser_contract_id": row["parser_contract_id"],
                "execution_contract_id": row["execution_contract_id"],
                "temperature": row["temperature"],
                "seed": row["seed"],
                "max_output_tokens": row["max_output_tokens"],
                "response_format": row["response_format"],
                "streaming": row["streaming"],
                "concurrency": row["concurrency"],
                "scientific_retry_count": row["scientific_retry_count"],
                "infrastructure_retry_count": row["infrastructure_retry_count"],
            }
        )
    return SimpleNamespace(manifest=pd.DataFrame(v1_rows), contexts=contexts)


def test_prompt_v2_contract_validates_and_rejects_hash_semantic_example_and_count_defects():
    row = validate_prompt_v2_contract(_prompt_v2_frame())
    assert row["prompt_contract_id"] == "llm_rerank_index_json_full20_v2"

    bad_hash = _prompt_v2_frame()
    bad_hash.loc[0, "contract_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        validate_prompt_v2_contract(bad_hash)

    with pytest.raises(ValueError, match="scientific"):
        validate_prompt_v2_contract(_prompt_v2_frame(scientific_semantics_changed="true"))

    with pytest.raises(ValueError, match="example-order"):
        validate_prompt_v2_contract(_prompt_v2_frame(user_message_template=_prompt_v2_row()["user_message_template"] + "\n[1,2,...,20]"))

    with pytest.raises(ValueError, match="20"):
        validate_prompt_v2_contract(_prompt_v2_frame(requested_output_count="5"))


def test_manifest_validation_requires_sixty_reranking_rows_ten_per_model_dataset_and_model_tags():
    prompt = validate_prompt_v2_contract(_prompt_v2_frame())
    validate_corrective_manifest(_manifest(), prompt)

    with pytest.raises(ValueError, match="60"):
        validate_corrective_manifest(_manifest(rows=59), prompt)
    with pytest.raises(ValueError, match="non-reranking"):
        validate_corrective_manifest(_manifest(include_ranking=True), prompt)
    bad_tag = _manifest()
    bad_tag.loc[0, "model_tag"] = "wrong"
    with pytest.raises(ValueError, match="model tag"):
        validate_corrective_manifest(bad_tag, prompt)


def test_source_v1_reconciliation_and_prompt_context_preserve_history_candidates_and_hide_positive():
    manifest = _manifest()
    source = _source_inputs(manifest)
    inputs = P1DInputs(
        prompt_v2=_prompt_v2_frame(),
        corrective_manifest=manifest,
        v1_inputs=source,
        v1_parse=pd.DataFrame(),
        v1_execution=pd.DataFrame(),
        v1_metrics=pd.DataFrame(),
        v1_decision=pd.DataFrame(),
        execution_contract=pd.DataFrame(),
        parser_contract=pd.DataFrame(),
        recovery=pd.DataFrame(),
        contexts={},
        protected_hashes={},
    )
    validate_source_v1_reconciliation(inputs)
    contexts = build_prompt_v2_contexts(inputs)
    first = contexts["CLRPV2-REQ-0001"]
    assert sha256_items(first["candidate_items"]) == manifest.iloc[0]["source_candidate_order_sha256"]
    assert sha256_text(first["history_render"]) == manifest.iloc[0]["source_history_render_sha256"]
    assert "ranked_candidate_indices" in first["user_message"]
    assert "positive_item_id" not in json_dumps({"system": first["system_message"], "prompt": first["user_message"]}).lower()
    assert "previous output" not in first["user_message"].lower()


def test_resume_state_pending_completed_ambiguous_duplicate_and_append_only(tmp_path):
    manifest = _manifest(rows=2)
    assert classify_request_state(0, 0) == "not_started"
    assert classify_request_state(1, 1) == "completed"
    assert classify_request_state(1, 0) == "ambiguous_interrupted"
    assert classify_request_state(2, 1) == "invalid_duplicate_evidence"

    assert len(pending_requests_in_order(manifest, [], [])) == 2
    completed_request = [{"corrective_request_id": "CLRPV2-REQ-0001"}]
    completed_response = [{"corrective_request_id": "CLRPV2-REQ-0001"}]
    assert [row["corrective_request_id"] for row in pending_requests_in_order(manifest, completed_request, completed_response)] == ["CLRPV2-REQ-0002"]
    with pytest.raises(RuntimeError, match="ambiguous"):
        pending_requests_in_order(manifest, completed_request, [])
    with pytest.raises(RuntimeError, match="invalid_duplicate"):
        pending_requests_in_order(manifest, completed_request * 2, completed_response)

    path = tmp_path / "records.jsonl"
    append_jsonl_fsync(path, {"corrective_request_id": "a"})
    append_jsonl_fsync(path, {"corrective_request_id": "b"})
    assert [row["corrective_request_id"] for row in read_jsonl(path)] == ["a", "b"]


def test_parser_invalidity_taxonomy_and_salvage_non_substitution_cases():
    candidates = [f"item{i}" for i in range(20)]
    cases = {
        "valid": '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}',
        "partial": '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]}',
        "long": '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,1]}',
        "duplicate": '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,19]}',
        "out_of_range": '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,50]}',
        "malformed": '{"ranked_candidate_indices":[1,2]',
    }
    parsed = {name: classify_response(text, candidate_items=candidates, expected_count=20, require_full_permutation=True) for name, text in cases.items()}
    assert parsed["valid"]["strict_parse_valid"] is True
    assert parsed["partial"]["primary_failure_reason"] == "output_count_invalid"
    assert parsed["long"]["primary_failure_reason"] == "output_count_invalid"
    assert parsed["duplicate"]["primary_failure_reason"] == "duplicate_indices"
    assert parsed["out_of_range"]["primary_failure_reason"] == "index_range_invalid"
    assert parsed["malformed"]["primary_failure_reason"] == "json_parse_failed"
    missing, duplicates, out_of_range = missing_and_duplicate_indices([1, 2, 2, 50])
    assert 3 in missing and duplicates == {"2": 2} and out_of_range == [50]
    assert output_truncation_detected('{"ranked_candidate_indices":[1,2]}', "stop", "10", "512") is False
    assert output_truncation_detected('{"ranked_candidate_indices":[1,2]', "stop", "10", "512") is True


def _frames_for_decision(cell_valid_counts):
    manifest_rows = []
    execution_rows = []
    parse_rows = []
    ranking_rows = []
    order = 1
    for family in ["Qwen2 7B Instruct", "Mistral 7B Instruct"]:
        for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]:
            valid_count = cell_valid_counts[(family, dataset)]
            for i in range(10):
                request_id = f"CLRPV2-REQ-{order:04d}"
                manifest_rows.append({"corrective_request_id": request_id, "model_family": family, "model_tag": "tag", "dataset": dataset, "task_type": "reranking"})
                execution_rows.append({"corrective_request_id": request_id, "model_family": family, "model_tag": "tag", "dataset": dataset, "task_type": "reranking", "duration_seconds": "1", "eval_count": "50", "backend_success": "true"})
                valid = i < valid_count
                parse_rows.append(
                    {
                        "corrective_request_id": request_id,
                        "model_family": family,
                        "dataset": dataset,
                        "task_type": "reranking",
                        "strict_parse_valid": "true" if valid else "false",
                        "diagnostic_salvage_valid": "false",
                        "output_count_valid": "true" if valid else "false",
                        "uniqueness_valid": "true" if valid else "false",
                        "full_permutation_valid": "true" if valid else "false",
                        "candidate_validity": "true" if valid else "false",
                        "duplicate_count": "0",
                        "hallucination_count": "0",
                        "output_truncation_detected": "false",
                    }
                )
                ranking_rows.append(
                    {
                        "corrective_request_id": request_id,
                        "positive_in_top_5": "true" if valid else "",
                        "ndcg_at_5": "1" if valid else "",
                        "positive_rank_zero_based": "0" if valid else "",
                        "utility_evaluation_valid": "true" if valid else "false",
                    }
                )
                order += 1
    manifest = pd.DataFrame(manifest_rows)
    execution = pd.DataFrame(execution_rows)
    parse = pd.DataFrame(parse_rows)
    rankings = pd.DataFrame(ranking_rows)
    metrics = build_metrics(manifest, execution, parse, rankings)
    request_state = pd.DataFrame(
        [
            {"request_state": "completed", "request_record_count": "1", "response_record_count": "1"}
            for _ in range(60)
        ]
    )
    return request_state, execution, parse, metrics


def test_decision_passes_with_six_cells_at_least_eight_and_blocks_on_seven_candidate_invalid_or_truncation():
    pass_counts = {(family, dataset): 8 for family in ["Qwen2 7B Instruct", "Mistral 7B Instruct"] for dataset in ["Amazon Beauty", "Amazon Sports", "ML-1M"]}
    request_state, execution, parse, metrics = _frames_for_decision(pass_counts)
    decision = build_decision(request_state, execution, parse, metrics, True, True, True, True, True, True).iloc[0]
    assert decision["pilot_decision"] == "approve_full_local_execution_with_ranking_v1_reranking_v2"
    assert decision["recommended_single_next_task"] == APPROVED_NEXT_TASK

    fail_counts = dict(pass_counts)
    fail_counts[("Mistral 7B Instruct", "Amazon Beauty")] = 7
    request_state, execution, parse, metrics = _frames_for_decision(fail_counts)
    decision = build_decision(request_state, execution, parse, metrics, True, True, True, True, True, True).iloc[0]
    assert decision["pilot_decision"] == "require_reranking_prompt_v3_audit"

    request_state, execution, parse, metrics = _frames_for_decision(pass_counts)
    parse.loc[parse["strict_parse_valid"].eq("true").idxmax(), "candidate_validity"] = "false"
    decision = build_decision(request_state, execution, parse, metrics, True, True, True, True, True, True).iloc[0]
    assert decision["pilot_decision"] == "require_parser_defect_investigation"

    parse["candidate_validity"] = parse["strict_parse_valid"]
    parse["output_truncation_detected"] = "false"
    parse.loc[0, "output_truncation_detected"] = "true"
    decision = build_decision(request_state, execution, parse, metrics, True, True, True, True, True, True).iloc[0]
    assert decision["pilot_decision"] == "require_output_token_limit_revision"


def test_v1_v2_validity_change_labels():
    assert compare_validity("false", "true") == "improved"
    assert compare_validity("true", "false") == "regressed"
    assert compare_validity("true", "true") == "same_valid"
    assert compare_validity("false", "false") == "same_invalid"

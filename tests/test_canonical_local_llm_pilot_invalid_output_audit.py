import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.analysis.canonical_local_llm_pilot_invalid_output_audit import (
    AuditBundle,
    build_corrective_manifest,
    build_count_reconciliation,
    build_decision,
    build_parser_assessment,
    build_prompt_requirement_audit,
    build_prompt_v2_contract,
    classify_output_count_failure,
    duplicate_indices,
    extra_json_fields,
    leading_or_trailing_prose,
    parse_json_response,
    reconcile_raw_ids,
    assess_truncation,
)
from src.llm.canonical_llm_contract import prompt_contract_rows, sha256_text
from src.llm.canonical_local_llm_runner import MODEL_ORDER, STRATUM_ORDER, classify_response


def _strict_counts_report(counts: dict[tuple[str, str, str], int]) -> str:
    lines = [
        "## 29. Qwen2 utility findings",
        "",
        "| dataset | task_type | utility_evaluated_rows | hr_at_5_valid_only | ndcg_at_5_valid_only |",
        "| --- | --- | --- | --- | --- |",
    ]
    for dataset, task_type in STRATUM_ORDER:
        lines.append(f"| {dataset} | {task_type} | {counts[('Qwen2 7B Instruct', dataset, task_type)]} | 0 | 0 |")
    lines.extend(
        [
            "",
            "## 30. Mistral utility findings",
            "",
            "| dataset | task_type | utility_evaluated_rows | hr_at_5_valid_only | ndcg_at_5_valid_only |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for dataset, task_type in STRATUM_ORDER:
        lines.append(f"| {dataset} | {task_type} | {counts[('Mistral 7B Instruct', dataset, task_type)]} | 0 | 0 |")
    lines.append("")
    lines.append("## 31. Next section")
    return "\n".join(lines)


def _count_bundle(*, report_sports_transcription_error: bool = False) -> AuditBundle:
    invalid_by_cell = {
        ("Qwen2 7B Instruct", "Amazon Sports", "reranking"): 1,
        ("Qwen2 7B Instruct", "ML-1M", "reranking"): 1,
        ("Mistral 7B Instruct", "Amazon Beauty", "reranking"): 4,
        ("Mistral 7B Instruct", "Amazon Sports", "reranking"): 1,
    }
    rows = []
    request_order = 1
    for model_family, model_tag in MODEL_ORDER:
        for dataset, task_type in STRATUM_ORDER:
            invalid = invalid_by_cell.get((model_family, dataset, task_type), 0)
            for index in range(10):
                rows.append(
                    {
                        "request_id": f"REQ-{request_order:04d}",
                        "request_order": str(request_order),
                        "model_family": model_family,
                        "model_tag": model_tag,
                        "dataset": dataset,
                        "task_type": task_type,
                        "strict_parse_valid": "false" if index < invalid else "true",
                        "diagnostic_salvage_valid": "false",
                    }
                )
                request_order += 1
    reparsed = pd.DataFrame(rows)
    valid = reparsed.loc[reparsed["strict_parse_valid"].eq("true")]
    metrics = (
        valid.groupby(["model_family", "dataset", "task_type"])
        .size()
        .reset_index(name="strict_valid_rows")
    )
    report_counts = {
        (model, dataset, task): int(group["strict_parse_valid"].eq("true").sum())
        for (model, dataset, task), group in reparsed.groupby(["model_family", "dataset", "task_type"])
    }
    if report_sports_transcription_error:
        report_counts[("Mistral 7B Instruct", "Amazon Sports", "reranking")] += 1
    return AuditBundle(
        root=Path("."),
        inputs=SimpleNamespace(),
        request_records=[],
        response_records=[],
        event_records=[],
        request_state=pd.DataFrame(),
        execution=reparsed.assign(backend_success="true"),
        parse_results=reparsed.copy(),
        rankings=valid.copy(),
        metrics=metrics,
        decision=pd.DataFrame(),
        report_text=_strict_counts_report(report_counts),
        protected_hashes={},
        reparsed=reparsed,
    )


def test_reconcile_raw_ids_requires_unique_complete_120_request_response_evidence():
    requests = [{"request_id": f"R{i:03d}"} for i in range(120)]
    responses = [{"request_id": f"R{i:03d}"} for i in range(120)]
    events = [{"request_id": f"R{i:03d}", "event": event} for i in range(120) for event in ("start", "finish")]

    state = reconcile_raw_ids(requests, responses, events)
    assert state["valid"] is True

    duplicated = reconcile_raw_ids(requests + [{"request_id": "R000"}], responses, events)
    assert duplicated["valid"] is False
    assert "R000" in duplicated["duplicate_request_or_response_ids"]


def test_count_reconciliation_selects_raw_evidence_and_detects_113_vs_114_transcription_error():
    bundle = _count_bundle()
    counts = build_count_reconciliation(bundle)
    global_row = counts.loc[counts["aggregation_level"].eq("global")].iloc[0]
    sports_row = counts.loc[counts["aggregation_level"].eq("dataset") & counts["dataset"].eq("Amazon Sports")].iloc[0]

    assert global_row["strict_valid_count_raw_reparse"] == "113"
    assert global_row["all_sources_agree"] == "true"
    assert sports_row["strict_valid_count_raw_reparse"] == "38"
    assert "pasted_summary_transcription_error" in sports_row["notes"]

    mismatched = build_count_reconciliation(_count_bundle(report_sports_transcription_error=True))
    mismatched_global = mismatched.loc[mismatched["aggregation_level"].eq("global")].iloc[0]
    mismatched_sports = mismatched.loc[mismatched["aggregation_level"].eq("dataset") & mismatched["dataset"].eq("Amazon Sports")].iloc[0]
    assert mismatched_global["strict_valid_count_report"] == "114"
    assert mismatched_global["discrepancy_type"] == "report_generation_error"
    assert mismatched_sports["strict_valid_count_report"] == "39"


def test_invalid_output_forensics_cover_short_long_duplicate_prose_fence_and_extra_fields():
    short = {"schema_valid": True, "index_type_valid": True, "raw_text": '{"ranked_candidate_indices":[1,2]}'}
    long = {"schema_valid": True, "index_type_valid": True, "raw_text": '{"ranked_candidate_indices":[1,2,3,4]}'}
    schema = {"schema_valid": False, "index_type_valid": False, "raw_text": '{"other":[1,2]}'}

    assert classify_output_count_failure([1, 2], 3, True, short) == "complete_json_partial_permutation"
    assert classify_output_count_failure([1, 2, 3, 4], 3, True, long) == "complete_json_extra_indices"
    assert classify_output_count_failure([], 3, True, schema) == "schema_mismatch"
    assert classify_output_count_failure([], 3, False, {"raw_text": '{"ranked_candidate_indices":[1,2]'}) == "truncated_array"
    assert duplicate_indices([1, 2, 2, 3, 3, 3]) == {"2": 2, "3": 3}
    valid_json, parsed, _ = parse_json_response('{"ranked_candidate_indices":[1,2],"extra":true}')
    assert valid_json is True
    assert extra_json_fields(parsed) == ["extra"]
    assert leading_or_trailing_prose('Answer: {"ranked_candidate_indices":[1,2]}') is True
    assert leading_or_trailing_prose('```json\n{"ranked_candidate_indices":[1,2]}\n```') is True


def test_truncation_assessment_does_not_treat_clean_short_json_as_truncated():
    assert assess_truncation('{"ranked_candidate_indices":[1,2]}', "stop", "5", "512") == "none"
    assert assess_truncation('{"ranked_candidate_indices":[1,2]}', "length", "512", "512") == "length_done_reason"
    assert assess_truncation('{"ranked_candidate_indices":[1,2]}', "stop", "500", "512") == "near_output_token_limit"
    assert assess_truncation('{"ranked_candidate_indices":[1,2]', "stop", "5", "512") == "possible_abrupt_incomplete_json"


def test_strict_parser_rejects_duplicates_and_partials_without_positive_or_salvage_repair():
    candidates = [f"item{i}" for i in range(20)]
    duplicate = '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,19]}'
    partial = '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]}'

    dup_result = classify_response(duplicate, candidate_items=candidates, expected_count=20, require_full_permutation=True)
    partial_result = classify_response(partial, candidate_items=candidates, expected_count=20, require_full_permutation=True)

    assert dup_result["strict_parse_valid"] is False
    assert dup_result["primary_failure_reason"] == "duplicate_indices"
    assert partial_result["strict_parse_valid"] is False
    assert partial_result["primary_failure_reason"] == "output_count_invalid"
    assert dup_result["diagnostic_salvage_valid"] is False
    assert partial_result["diagnostic_salvage_valid"] is False


def test_parser_assessment_flags_disagreement_as_defect_and_records_no_positive_label_use():
    bundle = SimpleNamespace(
        parse_results=pd.DataFrame(
            [
                {"request_id": "r1", "strict_parse_valid": "true", "diagnostic_salvage_valid": "false"},
                {"request_id": "r2", "strict_parse_valid": "true", "diagnostic_salvage_valid": "false"},
            ]
        ),
        reparsed=pd.DataFrame(
            [
                {"request_id": "r1", "strict_parse_valid": "true", "diagnostic_salvage_valid": "false"},
                {"request_id": "r2", "strict_parse_valid": "false", "diagnostic_salvage_valid": "false"},
            ]
        ),
    )

    assessment = build_parser_assessment(bundle)
    strict = assessment.loc[assessment["parser_contract_id"].str.contains("strict")].iloc[0]
    assert strict["stored_and_reparsed_disagreement_count"] == "1"
    assert strict["parser_revision_recommended"] == "true"
    assert strict["positive_label_used"] == "false"


def test_prompt_audit_and_v2_contract_are_minimal_without_example_order_bias():
    prompt_audit = build_prompt_requirement_audit(pd.DataFrame(prompt_contract_rows()))
    assert prompt_audit.loc[prompt_audit["requirement_name"].eq("exactly_20_indices"), "requirement_status"].iloc[0] == "explicit"
    assert prompt_audit.loc[prompt_audit["requirement_name"].eq("integers_1_to_20"), "requirement_status"].iloc[0] == "implicit"
    assert prompt_audit.loc[prompt_audit["requirement_name"].eq("no_omitted_integer"), "requirement_status"].iloc[0] == "absent"

    parser_ok = pd.DataFrame(
        [
            {"parser_revision_recommended": "false", "parser_contract_id": "strict"},
            {"parser_revision_recommended": "false", "parser_contract_id": "salvage"},
        ]
    )
    invalid = pd.DataFrame([{"root_cause_classification": "prompt_noncompliance"}])
    prompt_v2 = build_prompt_v2_contract(prompt_audit, invalid, parser_ok)

    assert len(prompt_v2) == 1
    row = prompt_v2.iloc[0]
    assert row["scientific_semantics_changed"] == "false"
    assert "[1,2,...,20]" not in row["user_message_template"]
    assert "verify internally" in row["internal_validation_instruction"]


def test_corrective_manifest_keeps_only_same_60_reranking_rows_with_no_retries():
    manifest_rows = []
    contexts = {}
    order = 1
    for index in range(62):
        task_type = "ranking" if index in {0, 61} else "reranking"
        pilot_row_id = f"P{index:03d}"
        contexts[pilot_row_id] = {"history_render": f"history {index}"}
        manifest_rows.append(
            {
                "request_id": f"REQ-{index:04d}",
                "pilot_row_id": pilot_row_id,
                "model_family": "Qwen2 7B Instruct" if index < 31 else "Mistral 7B Instruct",
                "model_tag": "model",
                "dataset": "Amazon Beauty",
                "task_type": task_type,
                "candidate_set_id": f"C{index:03d}",
                "user_id": f"U{index:03d}",
                "candidate_order_sha256": f"{index:064x}"[-64:],
                "parser_contract_id": "parser",
                "execution_contract_id": "execution",
                "requested_output_count": "20" if task_type == "reranking" else "5",
                "temperature": "0",
                "seed": "",
                "max_output_tokens": "512",
                "response_format": "json_object",
                "streaming": "false",
                "concurrency": "1",
                "scientific_retry_count": "0",
                "infrastructure_retry_count": "0",
                "request_order": str(order),
            }
        )
        order += 1
    bundle = SimpleNamespace(inputs=SimpleNamespace(manifest=pd.DataFrame(manifest_rows), contexts=contexts))
    prompt_v2 = pd.DataFrame([{"prompt_contract_id": "llm_rerank_index_json_full20_v2", "contract_sha256": "a" * 64}])

    corrective = build_corrective_manifest(bundle, prompt_v2)

    assert len(corrective) == 60
    assert corrective["task_type"].eq("reranking").all()
    assert corrective["corrective_request_order"].tolist() == [str(i) for i in range(1, 61)]
    assert corrective["scientific_retry_count"].eq("0").all()
    assert corrective["infrastructure_retry_count"].eq("0").all()
    assert "REQ-0000" not in set(corrective["source_request_id_v1"])
    first = corrective.iloc[0]
    source = manifest_rows[1]
    assert first["source_candidate_order_sha256"] == source["candidate_order_sha256"]
    assert first["source_history_render_sha256"] == sha256_text(contexts[source["pilot_row_id"]]["history_render"])


def test_decision_approves_prompt_v2_only_when_parser_ok_and_truncation_absent():
    counts = build_count_reconciliation(_count_bundle())
    invalid = pd.DataFrame(
        [
            {"root_cause_classification": "prompt_noncompliance"},
            {"root_cause_classification": "prompt_noncompliance"},
        ]
    )
    parser_ok = pd.DataFrame([{"parser_revision_recommended": "false"}, {"parser_revision_recommended": "false"}])
    prompt_v2 = pd.DataFrame([{"prompt_contract_id": "v2"}])
    corrective = pd.DataFrame([{"row": str(i)} for i in range(60)])

    decision = build_decision(counts, invalid, parser_ok, prompt_v2, corrective).iloc[0]

    assert decision["recommended_decision"] == "approve_reranking_prompt_v2_corrective_pilot"
    assert decision["ranking_prompt_revision_required"] == "false"
    assert "Sports is 38/40" in decision["count_inconsistency_resolution"]

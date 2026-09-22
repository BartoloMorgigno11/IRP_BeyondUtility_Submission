import pandas as pd
import pytest

from src.analysis.canonical_local_llm_hybrid_prompt_holdout_audit import (
    DATASET_ORDER,
    DECISION_COLUMNS,
    EXPECTED_STRICT_COUNTS,
    HOLDOUT_REQUEST_COLUMNS,
    HOLDOUT_SAMPLE_COLUMNS,
    HYBRID_POLICY_ID,
    RANKING_V1_CONTRACT_ID,
    READINESS_THRESHOLD,
    RERANKING_PARSER_ID,
    RERANKING_V1_CONTRACT_ID,
    RERANKING_V2_CONTRACT_ID,
    SELECTION_BASIS_COLUMNS,
    SEMANTIC_EQUIVALENCE_COLUMNS,
    EvidenceBundle,
    build_decision,
    build_holdout_request_manifest,
    build_hybrid_policy_contract,
    build_prompt_selection_basis,
    build_semantic_equivalence,
    deterministic_selection_key,
    frame,
    reconstruct_v1_v2_cells,
    select_disjoint_holdout_sample,
)
from src.llm.canonical_llm_contract import contract_hash
from src.llm.canonical_local_llm_runner import classify_response


VALID_20 = '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}'
INVALID_19 = '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]}'


def _empty_bundle(tables=None, jsonl=None):
    return EvidenceBundle(tables=tables or {}, jsonl=jsonl or {}, protected_hashes={})


def _classification_row(model, dataset, prompt_version, index, strict_valid):
    text = VALID_20 if strict_valid else INVALID_19
    parsed = classify_response(
        text,
        candidate_items=[str(i) for i in range(1, 21)],
        expected_count=20,
        require_full_permutation=True,
    )
    dataset_slug = dataset.replace(" ", "-").replace("&", "and")
    model_slug = model.replace(" ", "-")
    request_id = f"{prompt_version.upper()}-{model_slug}-{dataset_slug}-{index:02d}"
    request = {
        "request_id": request_id,
        "corrective_request_id": request_id,
        "model_family": model,
        "dataset": dataset,
        "task_type": "reranking",
        "candidate_set_id": f"{dataset}-rerank",
        "user_id": f"user-{prompt_version}-{model}-{dataset}-{index}",
        "candidate_items": [str(i) for i in range(1, 21)],
    }
    response = {
        "request_id": request_id,
        "corrective_request_id": request_id,
        "backend_success": "true",
        "raw_response_text": text,
    }
    parse = {
        "model_family": model,
        "dataset": dataset,
        "task_type": "reranking",
        "strict_parse_valid": str(parsed["strict_parse_valid"]).lower(),
        "output_count_valid": str(parsed["output_count_valid"]).lower(),
        "uniqueness_valid": str(parsed["uniqueness_valid"]).lower(),
        "full_permutation_valid": str(parsed["full_permutation_valid"]).lower(),
        "candidate_validity": str(parsed["candidate_validity"]).lower(),
        "duplicate_count": str(parsed["duplicate_count"]),
        "hallucination_count": str(parsed["hallucination_count"]),
    }
    return request, response, parse, parsed


def _reconciliation_bundle():
    tables = {}
    jsonl = {"v1_requests": [], "v1_responses": [], "v2_requests": [], "v2_responses": []}
    for version in ["v1", "v2"]:
        parse_rows = []
        metric_rows = []
        for model in ["Qwen2 7B Instruct", "Mistral 7B Instruct"]:
            for dataset in DATASET_ORDER:
                expected = EXPECTED_STRICT_COUNTS[(version, model, dataset)]
                parsed_rows = []
                for index in range(10):
                    request, response, parse, parsed = _classification_row(model, dataset, version, index, index < expected)
                    jsonl[f"{version}_requests"].append(request)
                    jsonl[f"{version}_responses"].append(response)
                    parse_rows.append(parse)
                    parsed_rows.append(parsed)
                metric = {
                    "model_family": model,
                    "model_tag": "tag",
                    "dataset": dataset,
                    "task_type": "reranking",
                    "backend_success_rows": "10",
                    "strict_valid_rows": str(expected),
                    "candidate_valid_rows": str(expected),
                    "duplicate_free_rows": str(expected),
                    "hallucination_free_rows": str(expected),
                }
                if version == "v1":
                    metric.update(
                        {
                            "pilot_rows": "10",
                            "complete_permutation_rows": str(sum(bool(row["full_permutation_valid"]) for row in parsed_rows)),
                        }
                    )
                else:
                    metric.update(
                        {
                            "corrective_rows": "10",
                            "output_count_valid_rows": str(sum(bool(row["output_count_valid"]) for row in parsed_rows)),
                            "uniqueness_valid_rows": str(sum(bool(row["uniqueness_valid"]) for row in parsed_rows)),
                            "full_permutation_valid_rows": str(sum(bool(row["full_permutation_valid"]) for row in parsed_rows)),
                        }
                    )
                metric_rows.append(metric)
        tables[f"{version}_parse"] = pd.DataFrame(parse_rows)
        tables[f"{version}_metrics"] = pd.DataFrame(metric_rows)
    return _empty_bundle(tables=tables, jsonl=jsonl)


def test_evidence_reconciliation_accepts_valid_12_cell_evidence_and_reconstructs_minimums():
    cells = reconstruct_v1_v2_cells(_reconciliation_bundle())
    assert len(cells) == 12
    assert cells["all_sources_agree"].eq("true").all()
    qwen_v1 = cells[(cells["model_family"].eq("Qwen2 7B Instruct")) & cells["prompt_version"].eq("v1")]
    assert qwen_v1["strict_parse_validity_rate"].astype(float).min() == pytest.approx(0.9)
    qwen_v2 = cells[(cells["model_family"].eq("Qwen2 7B Instruct")) & cells["prompt_version"].eq("v2")]
    assert qwen_v2["strict_parse_validity_rate"].astype(float).min() == pytest.approx(0.4)


def test_evidence_reconciliation_rejects_cell_count_mismatch_and_raw_table_disagreement():
    bundle = _reconciliation_bundle()
    bundle.tables["v2_metrics"].loc[0, "strict_valid_rows"] = "999"
    with pytest.raises(ValueError, match="reconciliation failed"):
        reconstruct_v1_v2_cells(bundle)

    bundle = _reconciliation_bundle()
    bundle.tables["v1_parse"].loc[0, "strict_parse_valid"] = "false"
    with pytest.raises(ValueError, match="reconciliation failed"):
        reconstruct_v1_v2_cells(bundle)


def test_selection_basis_accepts_structural_only_and_rejects_utility_dependent_decisions():
    cells = reconstruct_v1_v2_cells(_reconciliation_bundle())
    selection = build_prompt_selection_basis(cells)
    assert len(selection) == 2
    assert selection["selection_basis_classification"].eq("structural_validity_based").all()
    assert selection["hr_used_for_selection"].eq("false").all()

    semantic = _semantic_ok()
    policy = frame([{"hybrid_policy_id": HYBRID_POLICY_ID}], ["hybrid_policy_id"])
    sample = frame([{"disjoint_from_previous_pilots": "true"}] * 30, HOLDOUT_SAMPLE_COLUMNS)
    manifest = frame([{"previous_pilot_overlap": "false"}] * 60, HOLDOUT_REQUEST_COLUMNS)
    selection_hr = selection.copy()
    selection_hr.loc[0, "hr_used_for_selection"] = "true"
    decision = build_decision(cells, selection_hr, semantic, policy, sample, manifest)
    assert decision.iloc[0]["recommended_decision"] == "reject_hybrid_policy_as_noncomparable"
    selection_ndcg = selection.copy()
    selection_ndcg.loc[0, "ndcg_used_for_selection"] = "true"
    decision = build_decision(cells, selection_ndcg, semantic, policy, sample, manifest)
    assert decision.iloc[0]["recommended_decision"] == "reject_hybrid_policy_as_noncomparable"
    selection_latency = selection.copy()
    selection_latency.loc[0, "latency_used_for_selection"] = "true"
    decision = build_decision(cells, selection_latency, semantic, policy, sample, manifest)
    assert decision.iloc[0]["recommended_decision"] == "reject_hybrid_policy_as_noncomparable"


def _contract_bundle(**overrides):
    prompt_v1 = {
        "prompt_contract_id": RERANKING_V1_CONTRACT_ID,
        "task_type": "reranking",
        "candidate_representation": "one_based_index_plus_title",
        "candidate_numbering": "one_based_1_to_20",
        "candidate_order_policy": "preserve_source_initial_candidate_order",
        "requested_output_count": "20",
        "requested_output_representation": "strict_json_full_permutation_one_based_candidate_indices",
        "explanation_allowed": "false",
        "code_fence_allowed": "false",
        "positive_label_hidden": "true",
        "history_source_policy": "train_split_only_exclude_heldout_positive",
        "history_item_representation": "numbered_title_only_from_item_title_map",
        "contract_sha256": "v1hash",
    }
    prompt_v2 = {
        "prompt_contract_id": RERANKING_V2_CONTRACT_ID,
        "task_type": "reranking",
        "candidate_representation": "one_based_index_plus_title",
        "candidate_numbering": "one_based_1_to_20",
        "candidate_order_policy": "preserve_source_initial_candidate_order",
        "requested_output_count": "20",
        "requested_output_representation": "strict_json_full_permutation_one_based_candidate_indices",
        "exact_permutation_requirement": "array_set_exactly_1_to_20_no_omissions_no_repeats",
        "internal_validation_instruction": "verify internally",
        "explanation_allowed": "false",
        "code_fence_allowed": "false",
        "positive_label_hidden": "true",
        "scientific_semantics_changed": "false",
        "contract_sha256": "v2hash",
    }
    prompt_v2.update(overrides.pop("prompt_v2", {}))
    parser = {
        "parser_contract_id": RERANKING_PARSER_ID,
        "expected_identifier_namespace": "one_based_candidate_index",
        "requires_full_permutation": "true",
        "parser_schema_sha256": "parserhash",
    }
    comparison = {
        "candidate_order_identical": overrides.pop("candidate_order_identical", "true"),
        "history_render_identical": overrides.pop("history_render_identical", "true"),
        "execution_parameters_identical": overrides.pop("execution_parameters_identical", "true"),
    }
    return _empty_bundle(
        tables={
            "prompt_contract": pd.DataFrame([prompt_v1, {"prompt_contract_id": RANKING_V1_CONTRACT_ID, "contract_sha256": "rankhash"}]),
            "v2_contract": pd.DataFrame([prompt_v2]),
            "parser_contract": pd.DataFrame([parser]),
            "v2_comparison": pd.DataFrame([comparison]),
        }
    )


def _semantic_ok():
    return build_semantic_equivalence(_contract_bundle())


def test_semantic_equivalence_accepts_format_reinforcement_and_rejects_material_changes():
    semantic = _semantic_ok()
    assert semantic["hybrid_policy_compatible"].eq("true").all()
    assert "format_compliance_reinforcement" in set(semantic["difference_classification"])

    assert build_semantic_equivalence(_contract_bundle(prompt_v2={"candidate_representation": "item_id"}))[
        "candidate_information_changed"
    ].eq("true").any()
    assert build_semantic_equivalence(_contract_bundle(history_render_identical="false"))[
        "history_information_changed"
    ].eq("true").any()
    assert build_semantic_equivalence(_contract_bundle(prompt_v2={"requested_output_count": "5"}))[
        "output_space_changed"
    ].eq("true").any()
    assert build_semantic_equivalence(_contract_bundle(execution_parameters_identical="false"))[
        "execution_parameter_changed"
    ].eq("true").any()

    semantic_parser_changed = semantic.copy()
    semantic_parser_changed.loc[0, "parser_changed"] = "true"
    semantic_parser_changed.loc[0, "hybrid_policy_compatible"] = "false"
    assert semantic_parser_changed["hybrid_policy_compatible"].eq("false").any()


def test_hybrid_policy_assigns_qwen_v1_mistral_v2_and_requires_adapted_classification():
    cells = reconstruct_v1_v2_cells(_reconciliation_bundle())
    selection = build_prompt_selection_basis(cells)
    policy = build_hybrid_policy_contract(_contract_bundle(), selection, _semantic_ok())
    assert len(policy) == 1
    row = policy.iloc[0]
    assert row["qwen_ranking_prompt_contract_id"] == RANKING_V1_CONTRACT_ID
    assert row["qwen_reranking_prompt_contract_id"] == RERANKING_V1_CONTRACT_ID
    assert row["mistral_reranking_prompt_contract_id"] == RERANKING_V2_CONTRACT_ID
    assert row["paper_alignment"] == "adapted_model_specific_format_compliance_contract"
    assert row["paper_alignment"] != "paper_exact"


def _holdout_bundle():
    preflight_rows = []
    rendering_rows = []
    v1_rows = []
    v2_rows = []
    for dataset in DATASET_ORDER:
        for index in range(12):
            row = {
                "dataset": dataset,
                "task_type": "reranking",
                "candidate_set_id": f"{dataset}-rerank",
                "source_path": f"source/{dataset}.csv",
                "user_id": f"user-{index:02d}",
                "positive_item_id": f"pos-{index:02d}",
                "candidate_count": "20",
                "history_item_count": "5",
                "candidate_titles_complete": "true",
                "history_titles_complete": "true",
                "held_out_positive_absent_from_history": "true",
                "primary_prompt_compatible": "true",
            }
            preflight_rows.append(row)
            rendering_rows.append(
                {
                    **row,
                    "prompt_contract_id": RERANKING_V1_CONTRACT_ID,
                    "candidate_order_sha256": f"candidate-{dataset}-{index}",
                    "history_render_sha256": f"history-{dataset}-{index}",
                }
            )
            if index < 2:
                v1_rows.append(row)
                v2_rows.append(row)
    tables = {
        "cohort_preflight": pd.DataFrame(preflight_rows),
        "prompt_rendering": pd.DataFrame(rendering_rows),
        "v1_sample": pd.DataFrame(v1_rows),
        "v2_manifest": pd.DataFrame(v2_rows),
        "prompt_contract": pd.DataFrame(
            [
                {"prompt_contract_id": RERANKING_V1_CONTRACT_ID, "contract_sha256": "v1hash"},
                {"prompt_contract_id": RANKING_V1_CONTRACT_ID, "contract_sha256": "rankhash"},
            ]
        ),
        "v2_contract": pd.DataFrame([{"prompt_contract_id": RERANKING_V2_CONTRACT_ID, "contract_sha256": "v2hash"}]),
        "parser_contract": pd.DataFrame(
            [{"parser_contract_id": RERANKING_PARSER_ID, "parser_schema_sha256": "parserhash"}]
        ),
        "execution_contract": pd.DataFrame(
            [
                {
                    "model_family": "Qwen2 7B Instruct",
                    "task_type": "reranking",
                    "execution_contract_id": "exec_qwen",
                    "contract_sha256": "execq",
                    "temperature": "0",
                    "seed": "",
                    "max_output_tokens": "512",
                    "response_format": "json_object",
                    "streaming": "false",
                    "concurrency": "1",
                    "scientific_retry_count": "0",
                    "infrastructure_retry_count": "0",
                },
                {
                    "model_family": "Mistral 7B Instruct",
                    "task_type": "reranking",
                    "execution_contract_id": "exec_mistral",
                    "contract_sha256": "execm",
                    "temperature": "0",
                    "seed": "",
                    "max_output_tokens": "512",
                    "response_format": "json_object",
                    "streaming": "false",
                    "concurrency": "1",
                    "scientific_retry_count": "0",
                    "infrastructure_retry_count": "0",
                },
            ]
        ),
    }
    return _empty_bundle(tables=tables)


def test_holdout_selection_is_deterministic_disjoint_and_result_independent():
    assert deterministic_selection_key("D", "reranking", "C", "001") == deterministic_selection_key(
        "D", "reranking", "C", "001"
    )
    bundle = _holdout_bundle()
    sample = select_disjoint_holdout_sample(bundle, approved=True)
    assert len(sample) == 30
    assert sample["dataset"].value_counts().to_dict() == {"Amazon Beauty": 10, "Amazon Sports": 10, "ML-1M": 10}
    assert sample["holdout_row_id"].is_unique
    assert sample["task_type"].eq("reranking").all()
    assert sample["used_in_v1_pilot"].eq("false").all()
    assert sample["used_in_v2_pilot"].eq("false").all()
    assert sample["disjoint_from_previous_pilots"].eq("true").all()
    assert "stored_positive_position" not in sample.columns


def test_holdout_manifest_has_60_requests_same_rows_across_models_and_no_overlap():
    bundle = _holdout_bundle()
    sample = select_disjoint_holdout_sample(bundle, approved=True)
    policy = frame([{"hybrid_policy_id": HYBRID_POLICY_ID}], ["hybrid_policy_id"])
    manifest = build_holdout_request_manifest(bundle, sample, policy)
    assert len(manifest) == 60
    assert manifest["task_type"].eq("reranking").all()
    qwen = manifest[manifest["model_family"].eq("Qwen2 7B Instruct")]
    mistral = manifest[manifest["model_family"].eq("Mistral 7B Instruct")]
    assert set(qwen["holdout_row_id"]) == set(mistral["holdout_row_id"]) == set(sample["holdout_row_id"])
    assert set(qwen["prompt_contract_id"]) == {RERANKING_V1_CONTRACT_ID}
    assert set(mistral["prompt_contract_id"]) == {RERANKING_V2_CONTRACT_ID}
    assert manifest["previous_pilot_overlap"].eq("false").all()


def test_decision_approves_only_without_utility_and_requires_holdout_or_prompt_v3_on_semantic_failure():
    cells = reconstruct_v1_v2_cells(_reconciliation_bundle())
    selection = build_prompt_selection_basis(cells)
    semantic = _semantic_ok()
    policy = frame([{"hybrid_policy_id": HYBRID_POLICY_ID}], ["hybrid_policy_id"])
    sample = frame([{"disjoint_from_previous_pilots": "true"}] * 30, HOLDOUT_SAMPLE_COLUMNS)
    manifest = frame([{"previous_pilot_overlap": "false"}] * 60, HOLDOUT_REQUEST_COLUMNS)
    decision = build_decision(cells, selection, semantic, policy, sample, manifest)
    row = decision.iloc[0]
    assert row["recommended_decision"] == "approve_hybrid_disjoint_holdout_pilot"
    assert row["holdout_validation_required"] == "true"
    assert row["readiness_threshold"] == f"{READINESS_THRESHOLD:.17g}"

    semantic_bad = semantic.copy()
    semantic_bad.loc[0, "hybrid_policy_compatible"] = "false"
    decision_bad = build_decision(cells, selection, semantic_bad, policy, sample, manifest)
    assert decision_bad.iloc[0]["recommended_decision"] == "require_common_prompt_v3_audit"

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.canonical_local_llm_top10_confirmation_preflight import (
    DATASET_ORDER,
    DISCREPANCY_CLASSIFICATION,
    MODEL_ORDER,
    PREFLIGHT_DECISION,
    TOP10_PARSER_ID,
    TOP10_PROMPT_ID,
    build_confirmation_sample,
    build_preflight_decision,
    build_request_manifest,
    build_top10_parser_contract,
    build_top10_prompt_contract,
    build_top10_source_audit,
    classify_p1f_discrepancy,
    parse_top10_strict_index_response,
    reconcile_prompt_compatible_counts,
    selection_key,
)


def _cohort_and_rendering_for_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, str]] = []
    render_rows: list[dict[str, str]] = []
    counts = {
        ("Amazon Beauty", "ranking"): (1000, 994, 990),
        ("Amazon Beauty", "reranking"): (1000, 994, 990),
        ("Amazon Sports", "ranking"): (1000, 913, 892),
        ("Amazon Sports", "reranking"): (1000, 913, 892),
        ("ML-1M", "ranking"): (1000, 1000, 1000),
        ("ML-1M", "reranking"): (1000, 1000, 1000),
    }
    for (dataset, task_type), (full_rows, strict_rows, primary_rows) in counts.items():
        for idx in range(full_rows):
            rows.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": f"{dataset}-{task_type}",
                    "source_path": "synthetic.csv",
                    "user_id": f"{idx:04d}",
                    "positive_item_id": f"item-{idx}",
                    "candidate_count": "20",
                    "candidate_titles_complete": str(idx < strict_rows).lower(),
                    "history_titles_complete": str(idx < primary_rows).lower(),
                    "held_out_positive_absent_from_history": "true",
                    "primary_prompt_compatible": str(idx < primary_rows).lower(),
                    "eligible_full_candidate_layer": str(idx < full_rows).lower(),
                    "eligible_strict_title_primary": str(idx < strict_rows).lower(),
                }
            )
        for idx in range(strict_rows):
            render_rows.append(
                {
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": f"{dataset}-{task_type}",
                    "user_id": f"{idx:04d}",
                    "rendering_valid": str(idx < primary_rows).lower(),
                    "history_item_count": "10",
                    "candidate_order_sha256": f"cand-{idx}",
                    "history_render_sha256": f"hist-{idx}",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(render_rows)


def _sample_fixture() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str, str], dict[str, str]]]:
    rows: list[dict[str, str]] = []
    render_rows: list[dict[str, str]] = []
    previous: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for dataset in DATASET_ORDER:
        for idx in range(16):
            candidate_set_id = f"{dataset}-rerank"
            user_id = f"user-{idx:02d}"
            row = {
                "dataset": dataset,
                "task_type": "reranking",
                "candidate_set_id": candidate_set_id,
                "source_path": "synthetic_candidates.csv",
                "user_id": user_id,
                "positive_item_id": f"pos-{idx}",
                "candidate_count": "20",
                "candidate_titles_complete": "true",
                "history_titles_complete": "true",
                "held_out_positive_absent_from_history": "true",
                "primary_prompt_compatible": "true",
                "eligible_full_candidate_layer": "true",
                "eligible_strict_title_primary": "true",
            }
            rows.append(row)
            render_rows.append(
                {
                    "dataset": dataset,
                    "task_type": "reranking",
                    "candidate_set_id": candidate_set_id,
                    "user_id": user_id,
                    "rendering_valid": "true",
                    "history_item_count": "10",
                    "candidate_order_sha256": f"candhash-{dataset}-{idx}",
                    "history_render_sha256": f"histhash-{dataset}-{idx}",
                }
            )
        sorted_identities = sorted(
            [
                (dataset, "reranking", f"{dataset}-rerank", f"user-{idx:02d}")
                for idx in range(16)
            ],
            key=lambda item: (selection_key(*item), item[2], item[3]),
        )
        for identity in sorted_identities[:2]:
            previous[identity] = {
                "used_in_v1_pilot": "true",
                "used_in_v2_pilot": "false",
                "used_in_hybrid_holdout": "false",
            }
    return pd.DataFrame(rows), pd.DataFrame(render_rows), previous


def _write_manifest_support_files(root: Path) -> None:
    table_dir = root / "results/tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "execution_contract_id": "exec_qwen2_7b_instruct_reranking_v1",
                "model_family": "Qwen2 7B Instruct",
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
                "parameter_source": "synthetic",
                "paper_alignment": "adapted",
                "contract_status": "ready",
                "blocking_reason": "",
                "contract_sha256": "qwen-contract",
                "notes": "",
            },
            {
                "execution_contract_id": "exec_mistral_7b_instruct_reranking_v1",
                "model_family": "Mistral 7B Instruct",
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
                "parameter_source": "synthetic",
                "paper_alignment": "adapted",
                "contract_status": "ready",
                "blocking_reason": "",
                "contract_sha256": "mistral-contract",
                "notes": "",
            },
        ]
    ).to_csv(table_dir / "canonical_llm_execution_parameter_contract.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_tag": "qwen2:7b-instruct",
                "model_digest": "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755",
            },
            {
                "model_tag": "mistral:7b-instruct",
                "model_digest": "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf",
            },
        ]
    ).to_csv(table_dir / "canonical_local_ollama_service_recovery.csv", index=False)


def test_count_reconciliation_authoritative_totals_and_discrepancy() -> None:
    cohort, rendering = _cohort_and_rendering_for_counts()
    result = reconcile_prompt_compatible_counts(cohort, rendering)
    global_row = result[result["aggregation_level"].eq("global_total")].iloc[0]
    ranking = result[
        (result["aggregation_level"].eq("task_total")) & (result["task_type"].eq("ranking"))
    ].iloc[0]
    reranking = result[
        (result["aggregation_level"].eq("task_total")) & (result["task_type"].eq("reranking"))
    ].iloc[0]
    assert int(global_row["full_candidate_rows"]) == 6000
    assert int(global_row["candidate_strict_title_rows"]) == 5814
    assert int(global_row["primary_prompt_compatible_rows"]) == 5764
    assert int(ranking["primary_prompt_compatible_rows"]) == 2882
    assert int(reranking["primary_prompt_compatible_rows"]) == 2882
    assert int(ranking["two_model_request_count"]) == 5764
    assert int(reranking["two_model_request_count"]) == 5764
    assert ranking["count_difference"] == "-50"
    assert reranking["count_difference"] == "-50"
    assert global_row["count_difference"] == "-100"
    assert global_row["discrepancy_classification"] == DISCREPANCY_CLASSIFICATION
    assert int(global_row["history_title_exclusion_rows"]) == 50
    assert classify_p1f_discrepancy(5814, 5814, 5764) == DISCREPANCY_CLASSIFICATION


def test_top10_source_evidence_accepts_official_10_and_rejects_paper_exact(tmp_path: Path) -> None:
    audit = build_top10_source_audit(tmp_path)
    assert audit["requested_output_count"].eq("10").any()
    paper = audit[audit["source_type"].eq("paper_pdf")].iloc[0]
    assert paper["evidence_classification"] == "blocked_ambiguous"
    assert paper["supports_top10"] == "false"
    assert not audit["paper_alignment"].eq("paper_exact").any()
    full20 = audit[audit["supports_full20"].eq("true")]
    assert set(full20["evidence_classification"]) == {"conflicting_prompt_variant"}


def test_prompt_contract_freezes_common_top10_json_index_semantics() -> None:
    prompt = build_top10_prompt_contract().iloc[0]
    assert prompt["prompt_contract_id"] == TOP10_PROMPT_ID
    assert prompt["candidate_count"] == "20"
    assert prompt["requested_output_count"] == "10"
    assert prompt["requires_complete_permutation"] == "false"
    assert prompt["model_specific_wording"] == "false"
    assert prompt["candidate_numbering"] == "one_based_1_to_20"
    assert prompt["positive_label_hidden"] == "true"
    assert "[1,2" not in prompt["user_message_template"]
    assert "Do not return all 20 candidates" in prompt["user_message_template"]


def test_strict_parser_accepts_only_ten_unique_one_based_indices() -> None:
    valid = parse_top10_strict_index_response('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}')
    assert valid.parse_valid is True
    assert valid.output_length == 10
    assert parse_top10_strict_index_response('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9]}').status == "invalid_output_length"
    assert parse_top10_strict_index_response('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11]}').status == "invalid_output_length"
    assert parse_top10_strict_index_response('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,9]}').status == "duplicate_index"
    assert parse_top10_strict_index_response('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,21]}').status == "out_of_range_index"
    full20 = parse_top10_strict_index_response('{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}')
    assert full20.parse_valid is False
    contract = build_top10_parser_contract().iloc[0]
    assert contract["parser_contract_id"] == TOP10_PARSER_ID
    assert contract["requires_complete_permutation"] == "false"
    assert contract["positive_label_used"] == "false"


def test_deterministic_sample_excludes_previous_rows_and_has_zero_overlap() -> None:
    cohort, rendering, previous = _sample_fixture()
    sample_a = build_confirmation_sample(cohort, rendering, previous)
    sample_b = build_confirmation_sample(cohort, rendering, previous)
    assert sample_a.equals(sample_b)
    assert len(sample_a) == 30
    assert sample_a.groupby("dataset").size().to_dict() == {
        "Amazon Beauty": 10,
        "Amazon Sports": 10,
        "ML-1M": 10,
    }
    identities = {
        (row.dataset, row.task_type, row.candidate_set_id, row.user_id)
        for row in sample_a.itertuples(index=False)
    }
    assert identities.isdisjoint(previous)
    assert sample_a["disjoint_from_all_previous_pilots"].eq("true").all()
    assert sample_a["sample_status"].eq("selected_disjoint_not_executed").all()


def test_manifest_has_sixty_requests_same_rows_common_prompt_and_zero_retries(tmp_path: Path) -> None:
    _write_manifest_support_files(tmp_path)
    cohort, rendering, previous = _sample_fixture()
    sample = build_confirmation_sample(cohort, rendering, previous)
    prompt = build_top10_prompt_contract()
    parser = build_top10_parser_contract()
    manifest = build_request_manifest(tmp_path, sample, prompt, parser)
    assert len(manifest) == 60
    assert manifest.groupby("model_family")["confirmation_row_id"].nunique().to_dict() == {
        "Mistral 7B Instruct": 30,
        "Qwen2 7B Instruct": 30,
    }
    assert (
        manifest[manifest["model_family"].eq("Qwen2 7B Instruct")]["model_digest"].iloc[0]
        == "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755"
    )
    assert (
        manifest[manifest["model_family"].eq("Mistral 7B Instruct")]["model_digest"].iloc[0]
        == "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf"
    )
    assert set(manifest["prompt_contract_id"]) == {TOP10_PROMPT_ID}
    assert set(manifest["parser_contract_id"]) == {TOP10_PARSER_ID}
    assert set(manifest["scientific_retry_count"]) == {"0"}
    assert set(manifest["infrastructure_retry_count"]) == {"0"}
    assert set(manifest["concurrency"]) == {"1"}
    assert list(manifest["model_family"].drop_duplicates()) == MODEL_ORDER
    assert manifest["request_status"].eq("planned_not_executed").all()


def test_decision_approves_only_after_reconciled_counts_and_zero_overlap(tmp_path: Path) -> None:
    _write_manifest_support_files(tmp_path)
    cohort, rendering = _cohort_and_rendering_for_counts()
    count = reconcile_prompt_compatible_counts(cohort, rendering)
    sample_cohort, sample_rendering, previous = _sample_fixture()
    sample = build_confirmation_sample(sample_cohort, sample_rendering, previous)
    prompt = build_top10_prompt_contract()
    parser = build_top10_parser_contract()
    manifest = build_request_manifest(tmp_path, sample, prompt, parser)
    frames = {
        "count_reconciliation": count,
        "top10_source_audit": build_top10_source_audit(tmp_path),
        "top10_prompt_contract": prompt,
        "top10_parser_contract": parser,
        "confirmation_sample": sample,
        "request_manifest": manifest,
    }
    decision = build_preflight_decision(frames).iloc[0]
    assert decision["preflight_decision"] == PREFLIGHT_DECISION
    assert decision["previous_pilot_overlap_count"] == "0"
    assert decision["readiness_threshold"] == "0.80"
    bad_sample = sample.copy()
    bad_sample.loc[bad_sample.index[0], "disjoint_from_all_previous_pilots"] = "false"
    frames["confirmation_sample"] = bad_sample
    frames["request_manifest"] = manifest
    assert build_preflight_decision(frames).iloc[0]["preflight_decision"] == "preflight_incomplete"

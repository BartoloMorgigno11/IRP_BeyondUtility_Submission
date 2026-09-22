from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.canonical_local_llm_final_reranking_protocol_audit import (
    FINAL_DECISION_COLUMNS,
    PAPER_ALIGNMENT_LABEL,
    SELECTED_DECISION,
    SELECTED_PROTOCOL_OPTION_ID,
    assess_reporting_policy,
    build_execution_scope,
    build_final_decision,
    build_full20_origin,
    build_holdout_interpretation,
    build_metric_requirements,
    build_protocol_options,
    build_source_inventory,
    calculate_request_counts,
    classify_full20_origin,
    final_decision_from_evidence,
    holdout_gate_passed,
    preserve_holdout_failure,
    source_confidence_classification,
    validate_frames,
    wilson_interval,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _holdout_rows() -> list[dict[str, object]]:
    rows = []
    cells = [
        ("Qwen2 7B Instruct", "Amazon Beauty", "llm_rerank_index_json_full20_v1", 10, 10, 10, 8),
        ("Qwen2 7B Instruct", "Amazon Sports", "llm_rerank_index_json_full20_v1", 10, 9, 10, 10),
        ("Qwen2 7B Instruct", "ML-1M", "llm_rerank_index_json_full20_v1", 10, 9, 10, 10),
        ("Mistral 7B Instruct", "Amazon Beauty", "llm_rerank_index_json_full20_v2", 10, 10, 10, 9),
        ("Mistral 7B Instruct", "Amazon Sports", "llm_rerank_index_json_full20_v2", 10, 9, 10, 7),
        ("Mistral 7B Instruct", "ML-1M", "llm_rerank_index_json_full20_v2", 10, 9, 10, 9),
    ]
    for idx, (model, dataset, prompt, prior_rows, prior_valid, holdout_rows, holdout_valid) in enumerate(cells, 1):
        rows.append(
            {
                "comparison_id": f"CMP-{idx:03d}",
                "model_family": model,
                "dataset": dataset,
                "selected_prompt_contract_id": prompt,
                "prior_pilot_phase": "pilot",
                "prior_rows": prior_rows,
                "prior_strict_valid_rows": prior_valid,
                "prior_strict_parse_validity_rate": prior_valid / prior_rows,
                "holdout_rows": holdout_rows,
                "holdout_strict_valid_rows": holdout_valid,
                "holdout_strict_parse_validity_rate": holdout_valid / holdout_rows,
                "validity_rate_difference": "",
                "same_prompt_contract": "true",
                "disjoint_rows": "true",
                "utility_not_used_for_readiness": "true",
                "comparison_status": "descriptive_only",
                "notes": "",
            }
        )
    return rows


def _cohort_rows() -> list[dict[str, object]]:
    rows = []
    for dataset, eligible in [("Amazon Beauty", 994), ("Amazon Sports", 913), ("ML-1M", 1000)]:
        for task_type in ["ranking", "reranking"]:
            rows.append(
                {
                    "cohort_id": "plan_e2_llm_strict_title",
                    "dataset": dataset,
                    "task_type": task_type,
                    "candidate_set_id": f"{dataset}_{task_type}",
                    "source_plan": "Plan E2",
                    "source_path": "",
                    "total_source_rows": 1000,
                    "eligible_rows": eligible,
                    "excluded_rows": 1000 - eligible,
                    "eligibility_rate": eligible / 1000,
                    "rows_missing_any_title": 0,
                    "rows_missing_positive_title": 0,
                    "unique_missing_title_items": 0,
                    "positive_position_min_zero_based": 0,
                    "positive_position_max_zero_based": 19,
                    "positive_position_mean_zero_based": 10,
                    "positive_position_distribution": "{}",
                    "notes": "",
                }
            )
    return rows


def test_source_classification_blocks_missing_paper_evidence() -> None:
    assert source_confidence_classification("paper", False) == "blocked_ambiguous"
    assert source_confidence_classification("official_repository", True) == "verified_official_code"


def test_source_inventory_does_not_fabricate_paper_exact_claim(tmp_path: Path) -> None:
    inventory = build_source_inventory(tmp_path)
    paper = inventory[inventory["source_type"].eq("paper")].iloc[0]
    assert paper["evidence_classification"] == "blocked_ambiguous"
    assert paper["supports_full20"] == "false"
    official_top10 = inventory[inventory["supports_top10"].eq("true")]
    assert not official_top10.empty


def test_metric_requirements_hr_ndcg_and_coverage_need_top5_not_full20(tmp_path: Path) -> None:
    metrics = build_metric_requirements(tmp_path)
    hr = metrics[metrics["metric_id"].eq("hr_at_5")].iloc[0]
    ndcg = metrics[metrics["metric_id"].eq("ndcg_at_5")].iloc[0]
    coverage = metrics[metrics["metric_id"].eq("item_coverage_at_5")].iloc[0]
    assert hr["requires_top5"] == "true"
    assert hr["requires_full20"] == "false"
    assert ndcg["requires_top5"] == "true"
    assert ndcg["requires_full20"] == "false"
    assert coverage["requires_top5"] == "true"
    assert coverage["requires_full20"] == "false"


def test_metric_requirements_identify_full_permutation_only_diagnostic(tmp_path: Path) -> None:
    metrics = build_metric_requirements(tmp_path)
    full_rank = metrics[metrics["metric_id"].eq("position_bias_full20_diagnostic")].iloc[0]
    assert full_rank["requires_full20"] == "true"
    assert full_rank["full_permutation_scientifically_necessary"] == "true"
    assert full_rank["status"] == "diagnostic_only"


def test_contract_origin_classifications() -> None:
    assert classify_full20_origin(True, False, False) == "paper_exact_full_permutation"
    assert classify_full20_origin(False, True, False) == "official_code_full_permutation"
    assert classify_full20_origin(False, False, True) == "adapted_full_permutation"
    assert classify_full20_origin(False, False, False) == "unsupported_full_permutation"


def test_full20_origin_is_adapted_not_paper_exact(tmp_path: Path) -> None:
    origin = build_full20_origin(tmp_path)
    assert set(origin["origin_classification"]) == {"adapted_full_permutation"}
    assert origin["paper_alignment"].eq("adapted_not_paper_exact").all()


def test_protocol_options_have_exactly_one_recommendation(tmp_path: Path) -> None:
    options = build_protocol_options(tmp_path)
    recommended = options[options["recommendation_status"].eq("recommended")]
    assert len(recommended) == 1
    assert recommended.iloc[0]["protocol_option_id"] == SELECTED_PROTOCOL_OPTION_ID
    assert recommended.iloc[0]["official_code_alignment"] == "strong_verified_top10"


def test_topk_cannot_be_approved_without_official_or_paper_evidence() -> None:
    decision = final_decision_from_evidence(
        official_topk_supported=False,
        full20_scientifically_required=False,
        holdout_failed=True,
        utility_used=False,
        prompt_v3_created=False,
    )
    assert decision != SELECTED_DECISION


def test_full20_cannot_be_marked_paper_exact_without_evidence() -> None:
    origin = classify_full20_origin(paper_exact=False, official_exact=False, local_required=True)
    assert origin == "adapted_full_permutation"
    assert origin != "paper_exact_full_permutation"


def test_protocol_comparison_includes_ranking_diagnostic_and_structured_options(tmp_path: Path) -> None:
    options = build_protocol_options(tmp_path)
    ids = set(options["protocol_option_id"])
    assert "option_d_ranking_only_reranking_blocked" in ids
    assert "option_e_full20_diagnostic_only" in ids
    structured = options[options["protocol_option_id"].eq("option_f_structured_decoding_adaptation")].iloc[0]
    assert structured["paper_alignment"] == "adapted_constrained_decoding"
    assert structured["recommendation_status"] == "not_recommended"


def test_failed_holdout_remains_failed_and_combined_does_not_override(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "results/tables/canonical_local_llm_hybrid_holdout_prior_pilot_comparison.csv",
        _holdout_rows(),
    )
    holdout = build_holdout_interpretation(tmp_path)
    failing = holdout[holdout["model_family"].eq("Mistral 7B Instruct") & holdout["dataset"].eq("Amazon Sports")].iloc[0]
    assert failing["holdout_validity_rate"] == "0.7"
    assert failing["holdout_gate_passed"] == "false"
    assert failing["combined_evidence_overrides_gate"] == "false"
    preserve_holdout_failure(holdout)


def test_wilson_interval_is_ordered() -> None:
    low, high = wilson_interval(7, 10)
    assert 0 <= low <= 0.7 <= high <= 1


def test_utility_cannot_alter_readiness_or_decision() -> None:
    assert holdout_gate_passed(7, 10) is False
    decision = final_decision_from_evidence(
        official_topk_supported=True,
        full20_scientifically_required=False,
        holdout_failed=True,
        utility_used=True,
        prompt_v3_created=False,
    )
    assert decision == "audit_incomplete"


def test_bias_policies_flag_valid_only_and_invalid_as_zero() -> None:
    valid_only = assess_reporting_policy("strict-valid-only utility")
    invalid_zero = assess_reporting_policy("invalid-as-zero utility")
    separate = assess_reporting_policy("separate validity plus valid-only utility")
    assert valid_only["selection_bias_risk"] == "high"
    assert valid_only["allowed_for_primary_results"] == "false"
    assert invalid_zero["cross_model_comparability"] == "requires_explicit_operational_metric_contract"
    assert separate["allowed_for_diagnostic_results"] == "true"


def test_final_decision_row_freezes_no_prompt_v3_and_next_task(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "results/tables/canonical_local_llm_hybrid_holdout_prior_pilot_comparison.csv",
        _holdout_rows(),
    )
    holdout = build_holdout_interpretation(tmp_path)
    decision = build_final_decision(tmp_path, holdout)
    assert list(decision.columns) == FINAL_DECISION_COLUMNS
    row = decision.iloc[0]
    assert row["recommended_decision"] == SELECTED_DECISION
    assert row["prompt_v3_created"] == "false"
    assert row["further_wording_tuning_allowed"] == "false"
    assert "Phase 5B-P1G" in row["recommended_single_next_task"]


def test_request_count_calculation_and_execution_scope(tmp_path: Path) -> None:
    _write_csv(tmp_path / "results/tables/experiment_cohort_summary.csv", _cohort_rows())
    counts = calculate_request_counts(tmp_path)
    assert counts["ranking_only_full_request_count"] == (994 + 913 + 1000) * 2
    assert counts["full_ranking_plus_reranking_request_count"] == (994 + 913 + 1000) * 4
    assert counts["topk_confirmation_pilot_request_count"] == 60
    scope = build_execution_scope(tmp_path)
    assert len(scope) == 12
    assert scope[scope["task_type"].eq("reranking")]["request_count"].astype(int).sum() == 60


def test_validate_frames_accepts_synthetic_complete_audit(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "results/tables/canonical_local_llm_hybrid_holdout_prior_pilot_comparison.csv",
        _holdout_rows(),
    )
    _write_csv(tmp_path / "results/tables/experiment_cohort_summary.csv", _cohort_rows())
    holdout = build_holdout_interpretation(tmp_path)
    frames = {
        "source_inventory": build_source_inventory(tmp_path),
        "metric_requirements": build_metric_requirements(tmp_path),
        "full20_origin": build_full20_origin(tmp_path),
        "protocol_options": build_protocol_options(tmp_path),
        "holdout_interpretation": holdout,
        "utility_bias": pd.DataFrame(),
        "execution_scope": build_execution_scope(tmp_path),
        "final_decision": build_final_decision(tmp_path, holdout),
    }
    from src.analysis.canonical_local_llm_final_reranking_protocol_audit import build_utility_bias

    frames["utility_bias"] = build_utility_bias()
    validate_frames(frames)


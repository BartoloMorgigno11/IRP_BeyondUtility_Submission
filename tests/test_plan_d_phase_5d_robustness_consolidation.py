from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_d_phase_5d_robustness_consolidation import (
    BLOCKED_COLUMNS,
    EVIDENCE_CLAIM_COLUMNS,
    EXPERIMENT_REGISTRY_COLUMNS,
    MODEL_TASK_RELIABILITY_COLUMNS,
    NEXT_ACTION_COLUMNS,
    PAIRING_AUDIT_COLUMNS,
    PROTOCOL_COMPARISON_COLUMNS,
    ROBUSTNESS_SUMMARY_COLUMNS,
    SOURCE_INVENTORY_COLUMNS,
    VALIDATION_COLUMNS,
    assign_evidence_level,
    build_model_task_reliability,
    build_protected_input_manifest,
    classify_protocol_status,
    frame,
    generated_profile_claim_allowed,
    markdown_table,
    normalize_repo_path,
    output_validity_category,
    render_report,
    select_strongest_evidence,
    stable_csv_bytes,
    validate_history_length_control,
    validate_nondeterminism_repetition,
    validate_pairing,
    validate_position_bias_control,
    verify_protected_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_evidence_hierarchy_assignment_full_matrix_precedes_diagnostic():
    assert assign_evidence_level(scope_classification="full_matrix", paired=False, repeated=False) == "phase_5c_full_matrix_evidence"
    assert assign_evidence_level(scope_classification="diagnostic_subset", paired=True, repeated=False) == "exact_paired_diagnostic_experiment"
    assert assign_evidence_level(scope_classification="diagnostic_subset", paired=True, repeated=True) == "repeated_controlled_diagnostic"


def test_canonical_and_legacy_protocol_classification():
    assert classify_protocol_status(prompt_identifier="llm_rank_index_json_top5_v1") == "canonical_ranking_top5"
    assert classify_protocol_status(prompt_identifier="llm_rerank_index_json_top10_officialcode_v1") == "canonical_top10"
    assert classify_protocol_status(prompt_identifier="llm_rerank_index_json_full20_v1") == "legacy_full20_v1"
    assert classify_protocol_status(prompt_identifier="llm_rerank_index_json_full20_v2") == "legacy_full20_v2"


def test_pilot_cannot_be_labeled_full_matrix():
    level = assign_evidence_level(scope_classification="pilot", paired=False, repeated=False)
    assert level == "small_unpaired_pilot"
    assert level != "phase_5c_full_matrix_evidence"


def test_unmatched_rows_are_not_paired():
    left = pd.DataFrame([{"user_id": "u1"}, {"user_id": "u2"}])
    right = pd.DataFrame([{"user_id": "u1"}, {"user_id": "u3"}])
    result = validate_pairing(left, right, ["user_id"])
    assert result["paired_analysis_permitted"] == "false"
    assert result["left_only_count"] == 1
    assert result["right_only_count"] == 1


def test_paired_key_validation_allows_exact_pairs():
    left = pd.DataFrame([{"user_id": "u1"}, {"user_id": "u2"}])
    right = pd.DataFrame([{"user_id": "u2"}, {"user_id": "u1"}])
    result = validate_pairing(left, right, ["user_id"])
    assert result["paired_analysis_permitted"] == "true"
    assert result["matched_pair_count"] == 2


def test_duplicate_paired_keys_are_detected():
    left = pd.DataFrame([{"user_id": "u1"}, {"user_id": "u1"}])
    right = pd.DataFrame([{"user_id": "u1"}])
    result = validate_pairing(left, right, ["user_id"])
    assert result["paired_analysis_permitted"] == "false"
    assert result["duplicate_key_count"] == 1


def test_nondeterminism_requires_repeated_runs():
    one_run = pd.DataFrame([{"model": "qwen", "prompt": "p1", "row": "1", "run_id": "a"}])
    repeated = pd.DataFrame(
        [
            {"model": "qwen", "prompt": "p1", "row": "1", "run_id": "a"},
            {"model": "qwen", "prompt": "p1", "row": "1", "run_id": "b"},
        ]
    )
    assert not validate_nondeterminism_repetition(one_run, ["model", "prompt", "row"], "run_id")
    assert validate_nondeterminism_repetition(repeated, ["model", "prompt", "row"], "run_id")


def test_position_bias_control_validation():
    rows = pd.DataFrame(
        [
            {"original_instance_id": "i1", "variant": "reverse", "original_positive_position": "2", "variant_positive_position": "19", "candidate_order_signature": "a", "candidate_pool_size": "20"},
            {"original_instance_id": "i1", "variant": "first", "original_positive_position": "2", "variant_positive_position": "1", "candidate_order_signature": "b", "candidate_pool_size": "20"},
        ]
    )
    assert validate_position_bias_control(rows)


def test_history_length_control_validation():
    rows = pd.DataFrame(
        [
            {"model": "qwen", "user_id": "u1", "history_setting": "short", "history_length_used": "5", "candidate_ids": "[1,2]", "positive_item_id": "1"},
            {"model": "qwen", "user_id": "u1", "history_setting": "long", "history_length_used": "20", "candidate_ids": "[1,2]", "positive_item_id": "1"},
        ]
    )
    assert validate_history_length_control(rows)


def test_generated_profile_claim_requires_control():
    assert not generated_profile_claim_allowed()
    controls = pd.DataFrame([{"profile_condition": "profile", "control_condition": "no_profile", "paired": "true"}])
    assert generated_profile_claim_allowed(controls)


def test_blocked_experiment_classification_uses_blocked_level():
    assert assign_evidence_level(scope_classification="blocked", paired=False, repeated=False, blocked=True) == "blocked_or_unresolved_artifact"


def test_output_validity_categories_are_separated():
    assert output_validity_category({"primary_failure_reason": "duplicate_indices", "strict_parse_valid": "false"}) == "duplicate_indices"
    assert output_validity_category({"hallucination_count": "1", "strict_parse_valid": "false"}) == "hallucination"
    assert output_validity_category({"backend_success": "false"}) == "backend_error"
    assert output_validity_category({"strict_parse_valid": "true"}) == "strict_valid"


def test_phase_5c_precedence_over_weaker_evidence():
    strongest = select_strongest_evidence(["small_unpaired_pilot", "phase_5c_full_matrix_evidence", "report_only_summary"])
    assert strongest == "phase_5c_full_matrix_evidence"


def test_stable_sorting_and_deterministic_csv_generation():
    rows = pd.DataFrame([{"id": "b", "value": "2"}, {"id": "a", "value": "1"}])
    left = stable_csv_bytes(rows, sort_columns=["id"])
    right = stable_csv_bytes(rows.iloc[::-1].reset_index(drop=True), sort_columns=["id"])
    assert left == right


def test_windows_path_normalization():
    raw = r"C:\Example\IRP_BeyondUtility_Project\results\tables\x.csv"
    assert normalize_repo_path(raw) == "results/tables/x.csv"


def test_protected_input_hash_verification(tmp_path: Path):
    (tmp_path / "report").mkdir()
    (tmp_path / "report" / "x.md").write_text("same\n", encoding="utf-8")
    first = build_protected_input_manifest(tmp_path, Path("results/tables/plan_d_phase_5d_protected_input_hashes_initial.csv"))
    second = build_protected_input_manifest(tmp_path, Path("results/tables/plan_d_phase_5d_protected_input_hashes_final.csv"))
    assert verify_protected_manifest(first, second)
    (tmp_path / "report" / "x.md").write_text("changed\n", encoding="utf-8")
    third = build_protected_input_manifest(tmp_path, Path("results/tables/plan_d_phase_5d_protected_input_hashes_final.csv"))
    assert not verify_protected_manifest(first, third)


def test_real_phase_5c_reconciles_expected_26_qwen_reranking_duplicate_failures():
    path = PROJECT_ROOT / "results/tables/canonical_local_llm_full_invalid_output_inventory.csv"
    if not path.is_file():
        pytest.skip("Phase 5C invalid-output inventory is not present")
    reliability = build_model_task_reliability(PROJECT_ROOT)
    assert int(pd.to_numeric(reliability["invalid_count"]).sum()) == 26
    assert int(pd.to_numeric(reliability["duplicate_index_count"]).sum()) == 26
    invalid = reliability[reliability["invalid_count"].ne("0")]
    assert set(invalid["model"]) == {"Qwen2 7B Instruct"}
    assert set(invalid["task"]) == {"reranking"}
    counts = dict(zip(invalid["dataset"], invalid["invalid_count"], strict=True))
    assert counts == {"Amazon Beauty": "1", "Amazon Sports": "3", "ML-1M": "22"}


def test_no_silent_repair_of_invalid_outputs():
    row = {"primary_failure_reason": "duplicate_indices", "uniqueness_valid": "false", "strict_parse_valid": "false"}
    assert output_validity_category(row) == "duplicate_indices"


def test_experiment_and_claim_ids_are_unique_in_synthetic_tables():
    experiments = frame([{"experiment_id": "EXP-001"}, {"experiment_id": "EXP-002"}], EXPERIMENT_REGISTRY_COLUMNS)
    claims = frame([{"claim_id": "CLAIM-001"}, {"claim_id": "CLAIM-002"}], EVIDENCE_CLAIM_COLUMNS)
    assert not experiments["experiment_id"].duplicated().any()
    assert not claims["claim_id"].duplicated().any()


def test_report_generation_from_normalized_data():
    outputs = _minimal_outputs()
    report = render_report(outputs)
    assert "# Plan D Phase 5D Robustness Consolidation" in report
    assert "## 21. Phase 5D Decision" in report
    assert "canonical_existing_robustness_evidence_consolidation_complete" in report
    assert "| evidence_level | count |" in report


def test_markdown_table_escapes_pipes():
    table = pd.DataFrame([{"a": "x|y"}])
    assert "x\\|y" in markdown_table(table, ["a"])


def _minimal_outputs() -> dict[str, pd.DataFrame]:
    robustness_rows = []
    for topic in [
        "nondeterminism",
        "candidate_position_bias",
        "history_length_sensitivity",
        "strict_prompt_behavior",
        "full20_v1_vs_v2",
        "full20_vs_top10",
        "qwen2_vs_mistral",
        "ranking_vs_reranking",
        "generated_profiles",
        "hallucination",
        "output_validity",
    ]:
        robustness_rows.append(
            {
                "topic": topic,
                "strongest_available_evidence": "phase_5c_full_matrix_evidence",
                "evidence_scope": "scope",
                "paired": "false",
                "repeated": "false",
                "sample_or_cohort_size": "1",
                "result_availability": "permitted",
                "generalizability": "within_scope",
                "canonical_relevance": "canonical",
                "concise_finding": "finding",
                "limitations": "limitations",
            }
        )
    return {
        "source_inventory": frame([{"path": "x"}], SOURCE_INVENTORY_COLUMNS),
        "experiment_registry": frame([{"experiment_id": "EXP-001", "scope_classification": "full_matrix"}], EXPERIMENT_REGISTRY_COLUMNS),
        "evidence_claims": frame(
            [
                {
                    "topic": "output_validity",
                    "claim_id": "CLAIM-001",
                    "evidence_level": "phase_5c_full_matrix_evidence",
                    "claim_status": "permitted",
                    "permitted_interpretation": "permitted",
                    "prohibited_interpretation": "prohibited",
                    "limitations": "limits",
                }
            ],
            EVIDENCE_CLAIM_COLUMNS,
        ),
        "robustness_summary": frame(robustness_rows, ROBUSTNESS_SUMMARY_COLUMNS),
        "protocol_comparison": frame(
            [
                {
                    "protocol_label": "canonical",
                    "candidate_count": "20",
                    "output_count": "10",
                    "permutation_requirement": "false",
                    "canonical_status": "canonical_top10",
                    "comparability_status": "canonical",
                }
            ],
            PROTOCOL_COMPARISON_COLUMNS,
        ),
        "model_task_reliability": frame(
            [
                {
                    "model": "Qwen2",
                    "dataset": "ML-1M",
                    "task": "ranking",
                    "planned_count": "1",
                    "strict_valid_count": "1",
                    "invalid_count": "0",
                    "duplicate_index_count": "0",
                    "hallucination_count": "0",
                    "validity_rate": "1",
                }
            ],
            MODEL_TASK_RELIABILITY_COLUMNS,
        ),
        "pairing_coverage_audit": frame(
            [{"comparison_identifier": "cmp", "matched_pair_count": "1", "paired_analysis_permitted": "true", "reason": "ok"}],
            PAIRING_AUDIT_COLUMNS,
        ),
        "blocked_or_unresolved": frame(
            [{"topic": "nondeterminism", "blocker_type": "not_executed", "status": "blocked", "recoverable_without_new_inference": "false", "prohibited_statement": "do not claim"}],
            BLOCKED_COLUMNS,
        ),
        "validation": frame(
            [{"status": "pass", "severity": "blocking", "check_name": "ok"}],
            VALIDATION_COLUMNS,
        ),
        "next_actions": frame(
            [{"action_id": "NEXT", "input_artifact": "x", "required_use": "use", "status": "ready"}],
            NEXT_ACTION_COLUMNS,
        ),
    }

from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis.plan_d_phase_6c_final_scientific_report import (
    BUILD_REPORT_PATH,
    FINAL_REPORT_PATH,
    FROZEN_COUNTS,
    HASHED_OUTPUT_PATHS,
    PHASE6C_OWNED_PATHS,
    REQUIRED_SECTIONS,
    SourceBundle,
    absolute_path_hits,
    baseline_summary,
    beyond_utility_summary,
    build_summaries,
    classify_report_notes_diff,
    claim_registry,
    conclusion_registry,
    dataset_cohort_summary,
    detect_unsupported_overclaims,
    full_matrix_counts,
    generate_report,
    hash_rows_match,
    llm_utility_summary,
    markdown_links,
    phase5e_counts,
    phase_decisions,
    protected_manifest_columns,
    protocol_summary,
    relative_links_resolve,
    report_sections_present,
    robustness_summary,
    scientific_language_audit,
    section_source_map,
    stable_claim_id,
    stable_csv_rows,
    verify_frozen_count_reconciliation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def bundle() -> SourceBundle:
    from src.analysis.plan_d_phase_6c_final_scientific_report import load_sources

    return load_sources(PROJECT_ROOT)


@pytest.fixture(scope="session")
def context(bundle: SourceBundle):
    summaries = build_summaries(bundle)
    claims, claim_ids = claim_registry(bundle, summaries)
    summaries["conclusion_registry"] = conclusion_registry(claim_ids)
    summaries["report_section_source_map"] = section_source_map(claims)
    report = generate_report(bundle, summaries, claims, summaries["conclusion_registry"])
    return summaries, claims, report


def test_phase_decision_validation_for_prior_phases(bundle: SourceBundle) -> None:
    decisions = phase_decisions(bundle)
    assert decisions == {
        "phase_5c_complete": True,
        "phase_5d_complete": True,
        "phase_5e_complete": True,
        "phase_6a_complete": True,
        "phase_6b_complete": True,
        "phase_6clean_complete": True,
    }


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_report_notes_provenance_gate(bundle: SourceBundle) -> None:
    assert bundle.notes_provenance == "authorized_previous_phase_modification"
    assert classify_report_notes_diff("", True) == "verified_pre_existing_modification"
    assert classify_report_notes_diff("+ Plan D Phase 6CLEAN: Repository Pruning and Delivery Freeze", True) == "authorized_previous_phase_modification"
    assert classify_report_notes_diff("+ unrelated", True) == "unresolved_modification_origin"


def test_73_component_accounting(bundle: SourceBundle) -> None:
    counts = phase5e_counts(bundle)
    assert verify_frozen_count_reconciliation(counts)
    assert int(counts["reproduced_exact_count"]) + int(counts["reconstructed_count"]) + int(counts["adapted_count"]) + int(counts["blocked_count"]) + int(counts["not_executed_count"]) + int(counts["unresolved_count"]) == 73


def test_every_coverage_percentage(bundle: SourceBundle) -> None:
    counts = phase5e_counts(bundle)
    for key, value in FROZEN_COUNTS.items():
        assert counts[key] == value


def test_exact_numerical_result_cell_coverage(bundle: SourceBundle) -> None:
    assert phase5e_counts(bundle)["exact_numeric_result_cell_pct"] == "0.0"


def test_full_matrix_counts(bundle: SourceBundle) -> None:
    counts = full_matrix_counts(bundle)
    assert counts["planned_requests"] == "11528"
    assert counts["responses"] == "11528"
    assert counts["runner_events"] == "23056"
    assert counts["ranking_evaluations"] == "5764"
    assert counts["reranking_evaluations"] == "5764"


def test_26_invalid_output_reconciliation(bundle: SourceBundle) -> None:
    counts = full_matrix_counts(bundle)
    assert counts["invalid_outputs"] == "26"
    assert counts["qwen2_reranking_invalid"] == "26"


def test_invalid_output_dataset_distribution(bundle: SourceBundle) -> None:
    counts = full_matrix_counts(bundle)
    assert (counts["beauty_invalid"], counts["sports_invalid"], counts["ml1m_invalid"]) == ("1", "3", "22")


def test_conditional_versus_operational_utility(context) -> None:
    summaries, _, report = context
    policies = {row["evaluation_policy_id"] for row in summaries["llm_utility_summary"]}
    assert "conditional_strict_valid_output_utility" in policies
    assert "adapted_end_to_end_operational_utility_invalid_as_failure" in policies
    assert "Operational utility is a project adaptation" in report


def test_baseline_cohort_caveats(context) -> None:
    summaries, _, report = context
    assert sum(int(row["dataset_task_combinations"]) for row in summaries["baseline_summary"]) == 42
    assert "No fully aligned global baseline-LLM cohort comparison exists" in report


def test_bootstrap_descriptive_wording(context) -> None:
    summaries, _, report = context
    assert sum(int(row["paired_bootstrap_hr_ndcg_count"]) for row in summaries["baseline_summary"]) == 168
    assert "descriptive 95% intervals" in report
    assert "confirmatory significance tests" in report


def test_qwen2_mistral_substitute_wording(context) -> None:
    _, _, report = context
    assert "local substitutes" in report
    assert "not reproductions of the paper" in report


def test_ranking_reranking_distinction(context) -> None:
    _, _, report = context
    assert "Ranking and reranking are separate tasks" in report


def test_top5_top10_distinction(bundle: SourceBundle) -> None:
    rows = protocol_summary(bundle)
    rank = next(row for row in rows if row["task_type"] == "ranking")
    rerank = next(row for row in rows if row["task_type"] == "reranking")
    assert rank["requested_output_count"] == "5"
    assert rerank["requested_output_count"] == "10"


def test_full20_legacy_classification(bundle: SourceBundle) -> None:
    rows = protocol_summary(bundle)
    legacy = next(row for row in rows if row["protocol_id"] == "PROTO-FULL20-LEGACY")
    assert "legacy_or_diagnostic" in legacy["classification"]


def test_beyond_utility_definitions(bundle: SourceBundle) -> None:
    rows = beyond_utility_summary(bundle)
    novelty = next(row for row in rows if row["metric_id"] == "novelty_at_5")
    assert "no smoothing" in novelty["definition"]
    item = next(row for row in rows if row["metric_id"] == "item_coverage_at_5")
    assert "complete candidate union" in item["definition"]


def test_blocked_popularity_bias_enforcement(bundle: SourceBundle) -> None:
    rows = beyond_utility_summary(bundle)
    blocked = next(row for row in rows if row["metric_id"] == "popularity_bias_at_5")
    assert blocked["blocked_reason"]


def test_blocked_ild_enforcement(bundle: SourceBundle) -> None:
    rows = beyond_utility_summary(bundle)
    blocked = next(row for row in rows if row["metric_id"] == "intra_list_diversity_at_5")
    assert blocked["blocked_reason"]


def test_unsupported_nondeterminism_rejection(context) -> None:
    _, _, report = context
    assert "Do not claim nondeterminism from a single execution" in report
    assert not detect_unsupported_overclaims(report)


def test_unsupported_profile_effect_rejection(context) -> None:
    _, _, report = context
    assert "Do not claim generated-profile benefit" in report


def test_unsupported_full20_top10_numerical_ablation_rejection(context) -> None:
    _, _, report = context
    assert "Do not claim a numerical full-20 versus top-10 ablation" in report


def test_plan_e_not_changing_phase_5e(context, bundle: SourceBundle) -> None:
    _, _, report = context
    assert len(bundle.tables["phase_6a_plan_e"]) == 285
    assert "Plan E does not alter the frozen Phase 5E percentages" in report


def test_phase_6clean_not_changing_scientific_evidence(context) -> None:
    _, _, report = context
    assert "archived zero scientific artifacts" in report
    assert "not a scientific experiment" in report


def test_complete_threat_categories(context) -> None:
    summaries, _, _ = context
    assert {row["threat_category"] for row in summaries["threats_to_validity"]} == {
        "Construct validity",
        "Internal validity",
        "External validity",
        "Conclusion validity",
        "Reproducibility validity",
    }


def test_claim_traceability(context) -> None:
    _, claims, _ = context
    assert claims
    assert all(row["claim_id"] and row["source_paths"] for row in claims)


def test_required_report_sections(context) -> None:
    _, _, report = context
    assert report_sections_present(report)
    assert len(REQUIRED_SECTIONS) == 38


def test_repository_relative_links(context) -> None:
    _, _, report = context
    assert markdown_links(report)
    assert relative_links_resolve(PROJECT_ROOT, report)


def test_absolute_path_rejection(context) -> None:
    _, _, report = context
    assert not absolute_path_hits(report)
    assert absolute_path_hits(r"C:\\Users\\x\\file.txt")


def test_reference_non_invention(context) -> None:
    _, _, report = context
    assert "arXiv:2411.00331" in report
    assert "supervisor" not in report.lower()
    assert "license" not in report.lower()


def test_deterministic_markdown(context) -> None:
    _, _, report_one = context
    _, _, report_two = context
    assert report_one == report_two


def test_deterministic_csvs() -> None:
    rows = [{"b": "2", "a": "1"}]
    assert stable_csv_rows(rows, ["a", "b"]) == stable_csv_rows(rows, ["a", "b"])


def test_protected_input_immutability_helpers() -> None:
    assert "relative_path" in protected_manifest_columns()
    assert "README.md" not in PHASE6C_OWNED_PATHS


def test_no_mutation_of_input_tables(bundle: SourceBundle) -> None:
    before_columns = list(bundle.tables["phase_5c_utility"].columns)
    _ = llm_utility_summary(bundle)
    assert list(bundle.tables["phase_5c_utility"].columns) == before_columns


def test_final_report_build_report_separation() -> None:
    assert FINAL_REPORT_PATH != BUILD_REPORT_PATH


def test_phase_6d_not_started_static() -> None:
    assert not any("phase_6d" in path.lower() for path in PHASE6C_OWNED_PATHS)
    assert not any("phase_7" in path.lower() for path in PHASE6C_OWNED_PATHS)


def test_stable_claim_id() -> None:
    assert stable_claim_id("DATA", "x") == stable_claim_id("DATA", "x")
    assert stable_claim_id("DATA", "x").startswith("DATA-")


def test_scientific_language_audit(context) -> None:
    _, _, report = context
    audit = scientific_language_audit(report)
    assert audit["contains_local_substitute_wording"]
    assert audit["contains_baseline_caveat"]
    assert audit["contains_blocked_metrics"]


def test_output_hash_match_helper() -> None:
    rows = [{"relative_path": "a", "exists": "true", "size_bytes": "1", "sha256": "abc"}]
    assert hash_rows_match(rows, rows)
    assert not hash_rows_match(rows, [{"relative_path": "a", "exists": "true", "size_bytes": "1", "sha256": "def"}])


def test_hashed_outputs_exclude_self_referential_hash_manifests() -> None:
    paths = {path.as_posix() for path in HASHED_OUTPUT_PATHS}
    assert "results/tables/plan_d_phase_6c_output_hashes_first.csv" not in paths
    assert "results/tables/plan_d_phase_6c_output_hashes_second.csv" not in paths

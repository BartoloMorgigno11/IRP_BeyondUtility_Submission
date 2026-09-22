from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.plan_d_phase_5e_paper_coverage_audit import (
    CATEGORY_COVERAGE_COLUMNS,
    COMPONENT_COVERAGE_COLUMNS,
    COMPONENT_MAPPING_COLUMNS,
    OUTPUT_TABLES,
    RESULT_CELL_COMPARISON_COLUMNS,
    build_component_coverage,
    build_component_evidence_mapping,
    build_extensions,
    build_local_evidence_registry,
    build_report,
    compute_category_coverage,
    compute_coverage_metrics,
    deduplicate_components,
    duplicate_component_mapping_exists,
    frame,
    macro_average,
    normalize_paper_components,
    normalize_path,
    paper_component_specs,
    percent,
    result_cell_comparability_status,
    result_cell_exact_comparison_supported,
    select_primary_status,
    stable_component_id,
    stable_csv_bytes,
    stable_result_cell_id,
    strongest_evidence_level,
    validate_exact_reproduction,
    validate_status_denominator,
    verify_protected_manifest,
)


def coverage_row(component_id: str, category: str, status: str, comparability: str = "reconstructed_paper_like") -> dict[str, str]:
    return {
        "component_id": component_id,
        "category": category,
        "component": component_id,
        "applicable": "true",
        "primary_coverage_status": status,
        "comparability_classification": comparability,
        "evidence_level": "source_or_generated_result_table",
        "evidence_artifacts": "artifact.csv",
        "exactness_mismatches": "",
        "blocker": "",
        "permitted_claim": "permitted",
        "prohibited_claim": "prohibited",
        "rationale": "rationale",
    }


def comparison_row(cell_id: str, status: str = "paper_value_not_locally_verified", local_id: str = "local") -> dict[str, str]:
    return {
        "paper_result_cell_id": cell_id,
        "local_result_id": local_id,
        "comparability_status": status,
        "exact_dataset_match": "false",
        "exact_task_match": "true",
        "exact_model_match": "false",
        "exact_cohort_match": "false",
        "exact_candidate_protocol_match": "false",
        "exact_metric_definition_match": "true",
        "exact_cutoff_match": "true",
        "exact_denominator_match": "false",
        "numerical_comparison_permitted": "false",
        "reason": "non-exact",
    }


def test_stable_paper_component_id_generation() -> None:
    assert stable_component_id("LLM models", "Qwen2 7B local substitute") == "PC-LLM-MODELS-QWEN2-7B-LOCAL-SUBSTITUTE"


def test_result_cell_id_is_stable() -> None:
    assert stable_result_cell_id("table|row|metric") == stable_result_cell_id("table|row|metric")
    assert stable_result_cell_id("table|row|metric") != stable_result_cell_id("table|row|other")


def test_component_deduplication_allows_identical_rows() -> None:
    row = {
        "component_id": "PC-A",
        "category": "datasets",
        "component_name": "A",
        "description": "A",
        "paper_section": "s",
        "paper_page": "",
        "table_or_figure": "",
        "original_dataset_scope": "",
        "original_model_scope": "",
        "original_task_scope": "",
        "original_protocol_scope": "",
        "original_metric_scope": "",
        "original_artifact_or_repository_reference": "",
        "applicability_to_project": "applicable",
        "not_applicable_reason": "",
    }
    deduped = deduplicate_components(pd.DataFrame([row, row]))
    assert len(deduped) == 1


def test_component_deduplication_rejects_conflicts() -> None:
    row_a = {
        "component_id": "PC-A",
        "category": "datasets",
        "component_name": "A",
        "description": "A",
        "paper_section": "s",
        "paper_page": "",
        "table_or_figure": "",
        "original_dataset_scope": "",
        "original_model_scope": "",
        "original_task_scope": "",
        "original_protocol_scope": "",
        "original_metric_scope": "",
        "original_artifact_or_repository_reference": "",
        "applicability_to_project": "applicable",
        "not_applicable_reason": "",
    }
    row_b = dict(row_a, description="different")
    with pytest.raises(ValueError):
        deduplicate_components(pd.DataFrame([row_a, row_b]))


def test_denominator_keeps_blocked_components() -> None:
    coverage = frame(
        [
            coverage_row("A", "cat", "reproduced_exact", "paper_exact"),
            coverage_row("B", "cat", "blocked", "blocked_not_comparable"),
        ],
        COMPONENT_COVERAGE_COLUMNS,
    )
    category = compute_category_coverage(coverage)
    comparisons = frame([comparison_row("R")], RESULT_CELL_COMPARISON_COLUMNS)
    metrics = compute_coverage_metrics(coverage, category, comparisons, build_extensions().head(1))
    assert metrics.set_index("metric_name").loc["applicable_component_count", "value"] == "2"
    assert metrics.set_index("metric_name").loc["blocked_micro_pct", "value"] == "50.0"


def test_not_applicable_requires_a_rationale() -> None:
    row = coverage_row("A", "cat", "not_applicable", "not_applicable")
    row["applicable"] = "false"
    row["rationale"] = ""
    with pytest.raises(ValueError):
        validate_status_denominator(frame([row], COMPONENT_COVERAGE_COLUMNS))


def test_extensions_are_excluded_from_reproduction_numerator() -> None:
    coverage = frame(
        [
            coverage_row("A", "cat", "reproduced_exact", "paper_exact"),
            coverage_row("B", "cat", "adapted", "adapted_local_model_extension"),
        ],
        COMPONENT_COVERAGE_COLUMNS,
    )
    category = compute_category_coverage(coverage)
    comparisons = frame([comparison_row("R")], RESULT_CELL_COMPARISON_COLUMNS)
    metrics = compute_coverage_metrics(coverage, category, comparisons, build_extensions())
    metric_map = metrics.set_index("metric_name")["value"].to_dict()
    assert metric_map["extension_only_count"] == str(len(build_extensions()))
    assert metric_map["strict_exact_reproduction_micro_pct"] == "50.0"


def test_adapted_components_excluded_from_exact_reproduction() -> None:
    coverage = frame([coverage_row("A", "cat", "adapted")], COMPONENT_COVERAGE_COLUMNS)
    category = compute_category_coverage(coverage)
    metrics = compute_coverage_metrics(coverage, category, frame([comparison_row("R")], RESULT_CELL_COMPARISON_COLUMNS), build_extensions())
    assert metrics.set_index("metric_name").loc["strict_exact_reproduction_micro_pct", "value"] == "0.0"


def test_reconstructed_components_excluded_from_exact_reproduction() -> None:
    coverage = frame([coverage_row("A", "cat", "reconstructed")], COMPONENT_COVERAGE_COLUMNS)
    category = compute_category_coverage(coverage)
    metrics = compute_coverage_metrics(coverage, category, frame([comparison_row("R")], RESULT_CELL_COMPARISON_COLUMNS), build_extensions())
    assert metrics.set_index("metric_name").loc["strict_exact_reproduction_micro_pct", "value"] == "0.0"
    assert metrics.set_index("metric_name").loc["reconstructed_or_better_micro_pct", "value"] == "100.0"


def test_reproduced_exact_requires_all_material_exactness_checks() -> None:
    row = {
        "same_dataset": "true",
        "same_cohort": "true",
        "same_task": "true",
        "same_candidate_protocol": "true",
        "same_model": "true",
        "same_model_version": "true",
        "same_prompt_semantics": "true",
        "same_parser_or_output_semantics": "true",
        "same_metric_definition": "true",
        "same_denominator": "true",
        "local_model": "official",
    }
    assert validate_exact_reproduction(row)
    assert not validate_exact_reproduction({**row, "same_model_version": "false"})


def test_qwen2_and_mistral_are_not_paper_exact() -> None:
    exact = {
        "same_dataset": "true",
        "same_cohort": "true",
        "same_task": "true",
        "same_candidate_protocol": "true",
        "same_model": "true",
        "same_model_version": "true",
        "same_prompt_semantics": "true",
        "same_parser_or_output_semantics": "true",
        "same_metric_definition": "true",
        "same_denominator": "true",
    }
    assert not validate_exact_reproduction({**exact, "local_model": "Qwen2 7B Instruct"})
    assert not validate_exact_reproduction({**exact, "local_model": "Mistral 7B Instruct"})


def test_same_metric_different_cohort_blocks_numeric_replication() -> None:
    status = result_cell_comparability_status(
        {
            "paper_value_verified": True,
            "exact_dataset_match": True,
            "exact_task_match": True,
            "exact_model_match": True,
            "exact_cohort_match": False,
            "exact_candidate_protocol_match": True,
            "exact_metric_definition_match": True,
            "exact_cutoff_match": True,
            "exact_denominator_match": True,
        }
    )
    assert status == "same_metric_different_cohort"


def test_same_task_different_model_blocks_numeric_replication() -> None:
    status = result_cell_comparability_status(
        {
            "paper_value_verified": True,
            "exact_dataset_match": True,
            "exact_task_match": True,
            "exact_model_match": False,
            "exact_cohort_match": True,
            "exact_candidate_protocol_match": True,
            "exact_metric_definition_match": True,
            "exact_cutoff_match": True,
            "exact_denominator_match": True,
            "same_task": True,
        }
    )
    assert status == "same_metric_different_model"


def test_legacy_full20_not_equated_with_canonical_top10() -> None:
    specs = paper_component_specs()
    legacy = [spec for spec in specs if spec.name == "Legacy full-20 reranking diagnostic"][0]
    assert legacy.status == "not_executed"
    assert legacy.comparability == "same_dataset_different_protocol"


def test_result_cell_exact_comparison_validation() -> None:
    row = {
        "paper_value_verified": True,
        "exact_dataset_match": True,
        "exact_task_match": True,
        "exact_model_match": True,
        "exact_cohort_match": True,
        "exact_candidate_protocol_match": True,
        "exact_metric_definition_match": True,
        "exact_cutoff_match": True,
        "exact_denominator_match": True,
    }
    assert result_cell_exact_comparison_supported(row)


def test_arbitrary_partial_status_scoring_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_status_denominator(frame([coverage_row("A", "cat", "partial_0_5")], COMPONENT_COVERAGE_COLUMNS))


def test_micro_percentage_calculation() -> None:
    assert percent(2, 4) == 50.0
    assert percent(0, 0) == 0.0


def test_macro_category_percentage_calculation() -> None:
    coverage = frame(
        [
            coverage_row("A", "cat1", "reproduced_exact", "paper_exact"),
            coverage_row("B", "cat2", "blocked", "blocked_not_comparable"),
        ],
        COMPONENT_COVERAGE_COLUMNS,
    )
    category = compute_category_coverage(coverage)
    assert macro_average(category, "strict_exact_pct") == 50.0


def test_empty_category_handling_excludes_zero_applicable() -> None:
    category = frame(
        [
            {
                "category": "empty",
                "total_components": "1",
                "applicable_components": "0",
                "reproduced_exact": "0",
                "reconstructed": "0",
                "adapted": "0",
                "extension_only": "0",
                "blocked": "0",
                "not_executed": "0",
                "unresolved": "0",
                "not_applicable": "1",
                "strict_exact_pct": "0.0",
                "reconstructed_or_better_pct": "0.0",
                "maximum_defensible_pct": "0.0",
                "blocked_pct": "0.0",
            },
            {
                "category": "full",
                "total_components": "1",
                "applicable_components": "1",
                "reproduced_exact": "1",
                "reconstructed": "0",
                "adapted": "0",
                "extension_only": "0",
                "blocked": "0",
                "not_executed": "0",
                "unresolved": "0",
                "not_applicable": "0",
                "strict_exact_pct": "100.0",
                "reconstructed_or_better_pct": "100.0",
                "maximum_defensible_pct": "100.0",
                "blocked_pct": "0.0",
            },
        ],
        CATEGORY_COVERAGE_COLUMNS,
    )
    assert macro_average(category, "strict_exact_pct") == 100.0


def test_percentage_bounds() -> None:
    assert percent(10, 1) == 100.0
    assert percent(-1, 1) == 0.0


def test_duplicate_component_mapping_detection() -> None:
    mapping = frame(
        [
            {"paper_component_id": "PC-A", "local_evidence_id": "E1"},
            {"paper_component_id": "PC-A", "local_evidence_id": "E1"},
        ],
        COMPONENT_MAPPING_COLUMNS,
    )
    assert duplicate_component_mapping_exists(mapping)


def test_primary_status_selection_from_multiple_evidence_mappings() -> None:
    assert select_primary_status(["adapted", "reconstructed"]) == "reconstructed"
    assert select_primary_status(["extension_only", "blocked"]) == "blocked"


def test_phase_5c_precedence_over_weaker_evidence() -> None:
    assert strongest_evidence_level(["earlier_phase_report", "phase_5c_full_matrix_evidence"]) == "phase_5c_full_matrix_evidence"


def test_phase_5d_precedence_for_robustness_classification() -> None:
    assert strongest_evidence_level(["report_only_summary", "phase_5d_robustness_consolidation"]) == "phase_5d_robustness_consolidation"


def test_extension_only_evidence_does_not_upgrade_paper_component() -> None:
    assert select_primary_status(["extension_only", "not_executed"]) == "not_executed"


def test_unresolved_conflicts_remain_unresolved() -> None:
    assert select_primary_status(["reconstructed", "blocked"], conflict=True) == "unresolved"


def test_deterministic_sorting_and_serialization() -> None:
    data = pd.DataFrame([{"b": "2", "a": "1"}], columns=["a", "b"])
    assert stable_csv_bytes(data) == stable_csv_bytes(data.copy())


def test_windows_path_normalization() -> None:
    assert normalize_path(r"results\\tables\\x.csv") == "results/tables/x.csv"


def test_protected_input_hash_verification() -> None:
    initial = pd.DataFrame(
        [{"path": "a", "protected_input": "true", "hash_method": "sha256", "sha256": "abc", "size_bytes": "1", "mtime_ns": "2", "policy": "p"}]
    )
    final = initial.copy()
    assert verify_protected_manifest(initial, final)
    changed = final.copy()
    changed.loc[0, "sha256"] = "def"
    assert not verify_protected_manifest(initial, changed)


def test_no_silent_mutation_of_input_data() -> None:
    coverage = frame([coverage_row("A", "cat", "reconstructed")], COMPONENT_COVERAGE_COLUMNS)
    before = coverage.copy(deep=True)
    category = compute_category_coverage(coverage)
    compute_coverage_metrics(coverage, category, frame([comparison_row("R")], RESULT_CELL_COMPARISON_COLUMNS), build_extensions())
    pd.testing.assert_frame_equal(coverage, before)


def test_report_generation_from_normalized_tables() -> None:
    root = Path(".")
    specs = paper_component_specs()[:3]
    components = normalize_paper_components(specs)
    evidence = build_local_evidence_registry(root)
    coverage = build_component_coverage(components, specs)
    mappings = build_component_evidence_mapping(components, evidence, specs)
    comparisons = frame([comparison_row("R")], RESULT_CELL_COMPARISON_COLUMNS)
    category = compute_category_coverage(coverage)
    extensions = build_extensions()
    metrics = compute_coverage_metrics(coverage, category, comparisons, extensions)
    outputs = {
        "component_coverage": coverage,
        "coverage_metrics": metrics,
        "category_coverage": category,
        "blockers": frame([], ["blocker_id", "component", "blocker_type", "missing_artifact_or_resource", "current_evidence", "recoverable_without_new_inference", "scientific_impact", "permitted_statement", "prohibited_statement"]),
        "claims": frame([], ["claim_id", "claim_type", "claim", "status", "supporting_artifacts", "permitted_statement", "prohibited_statement"]),
        "paper_result_cells": frame([], ["paper_result_cell_id", "paper_table", "paper_row", "paper_column", "dataset", "task", "model", "prompt_or_method", "metric", "cutoff", "value", "source_page", "verification_status"]),
        "result_cell_comparison": comparisons,
        "extensions": extensions,
        "validation": frame([{"validation_id": "V", "validation_scope": "s", "check_name": "c", "expected_value": "x", "observed_value": "x", "status": "pass", "severity": "blocking", "evidence": "e"}], ["validation_id", "validation_scope", "check_name", "expected_value", "observed_value", "status", "severity", "evidence"]),
        "next_actions": frame([], ["action_id", "input_artifact", "required_use", "status", "phase_6_handoff"]),
        "component_evidence_mapping": mappings,
    }
    report = build_report(outputs)
    assert "Maximum Defensible Paper-Component Coverage" in report


def test_real_phase_5c_and_phase_5d_schemas_are_readable() -> None:
    root = Path(".")
    phase5c = pd.read_csv(root / "results/tables/canonical_local_llm_full_matrix_decision.csv")
    phase5d = pd.read_csv(root / "results/tables/plan_d_phase_5d_validation.csv")
    assert {"expected_request_count", "execution_decision"}.issubset(phase5c.columns)
    assert {"validation_id", "status"}.issubset(phase5d.columns)


def test_output_table_names_are_phase_5e_specific() -> None:
    assert OUTPUT_TABLES
    assert all(path.name.startswith("plan_d_phase_5e_") for path in OUTPUT_TABLES.values())

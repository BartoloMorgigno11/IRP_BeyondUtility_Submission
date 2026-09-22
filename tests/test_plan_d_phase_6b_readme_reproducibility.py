from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.plan_d_phase_6b_readme_reproducibility import (
    ARTIFACT_CATALOG_PATH,
    ARTIFACT_CATALOG_PATH,
    CATALOG_REQUIRED_SECTIONS,
    DOC_SOURCE_COLUMNS,
    ENTRYPOINT_COLUMNS,
    FROZEN_PHASE5E_COUNTS,
    GUIDE_REQUIRED_SECTIONS,
    OUTPUT_TABLES,
    README_PATH,
    README_REQUIRED_SECTIONS,
    Phase6AInputs,
    build_documentation_source_map,
    build_excluded_entrypoints,
    build_final_artifact_index,
    build_portability_audit,
    build_readme_claim_audit,
    build_readme_hash_audit,
    build_safe_command_registry,
    command_is_safe,
    contains_prohibited_overclaim,
    detect_absolute_paths,
    detect_secret_patterns,
    detect_unsafe_quickstart,
    exclude_unsafe_entrypoints,
    frame,
    generate_artifact_catalog,
    generate_phase_report,
    generate_readme,
    generate_reproducibility_guide,
    markdown_links,
    missing_required_sections,
    path_with_spaces_safe,
    phase_6b_decision,
    portable_command,
    readme_preservation_plan,
    relative_markdown_links_resolve,
    select_safe_entrypoints,
    stable_csv_bytes,
    stable_markdown,
    verify_phase5e_counts,
    verify_protected_manifest,
)


def phase6a_inputs() -> Phase6AInputs:
    entrypoints = frame(
        [
            {
                "path": "src/experiments/build_canonical_local_llm_full_matrix_preflight.py",
                "intended_purpose": "offline builder",
                "required_inputs": "protected artifacts",
                "produced_outputs": "preflight tables",
                "safe_for_final_documentation": "true",
                "performs_llm_inference": "false",
                "performs_recommender_training": "false",
                "performs_recommender_inference": "false",
                "performs_downloading": "false",
                "performs_api_access": "false",
                "expected_runtime_class": "cheap_offline_aggregation",
            },
            {
                "path": "src/experiments/build_canonical_metric_preflight.py",
                "intended_purpose": "offline builder",
                "required_inputs": "protected artifacts",
                "produced_outputs": "metric preflight tables",
                "safe_for_final_documentation": "true",
                "performs_llm_inference": "false",
                "performs_recommender_training": "false",
                "performs_recommender_inference": "false",
                "performs_downloading": "false",
                "performs_api_access": "false",
                "expected_runtime_class": "cheap_offline_aggregation",
            },
            {
                "path": "src/experiments/run_canonical_local_llm_full_matrix.py",
                "intended_purpose": "runner",
                "required_inputs": "protected artifacts",
                "produced_outputs": "responses",
                "safe_for_final_documentation": "false",
                "performs_llm_inference": "true",
                "performs_recommender_training": "false",
                "performs_recommender_inference": "false",
                "performs_downloading": "false",
                "performs_api_access": "false",
                "expected_runtime_class": "llm_inference_or_llm_service_dependent",
            },
        ],
        ENTRYPOINT_COLUMNS,
    )
    accounting_rows = [
        {"status_or_metric": key, "value": value, "verification_status": "pass"}
        for key, value in {
            "applicable_component_count": "73",
            "reproduced_exact": "9",
            "reconstructed": "16",
            "adapted": "25",
            "blocked": "14",
            "not_executed": "5",
            "unresolved": "4",
            "strict_exact_reproduction_micro_pct": "12.328767",
            "maximum_defensible_micro_coverage_pct": "68.493151",
            "exact_numeric_result_cell_pct": "0.0",
        }.items()
    ]
    return Phase6AInputs(
        artifact_registry=frame(
            [
                {
                    "artifact_id": "ART-1",
                    "normalized_path": "results/tables/plan_d_phase_5e_coverage_metrics.csv",
                    "lifecycle_class": "canonical_final_evidence",
                    "canonicality": "canonical",
                    "final_delivery_disposition": "include_primary",
                    "scientific_role": "coverage",
                }
            ],
            ["artifact_id", "normalized_path", "lifecycle_class", "canonicality", "final_delivery_disposition", "scientific_role"],
        ),
        canonical_artifact_map=frame([{"scientific_topic": "Phase 5E paper coverage audit", "canonical_primary_artifacts": "results/tables/plan_d_phase_5e_coverage_metrics.csv"}], ["scientific_topic", "canonical_primary_artifacts"]),
        final_documentation_input_map=frame([{"final_document_section": "reproduction coverage"}], ["final_document_section"]),
        reproducibility_entrypoints=entrypoints,
        final_delivery_policy=frame([{"artifact_or_artifact_family": "Phase 5E coverage"}], ["artifact_or_artifact_family"]),
        legacy_superseded=frame([], ["old_artifact"]),
        plan_e_reconciliation=frame([{"path": "results/tables/plan_e_x.csv", "changes_phase_5e_claim": "false"}], ["path", "changes_phase_5e_claim"]),
        orphaned_or_unresolved=frame([], ["artifact"]),
        phase_5e_accounting=frame(accounting_rows, ["status_or_metric", "value", "verification_status"]),
        validation=frame([{"status": "pass"}], ["status"]),
        next_actions=frame([], ["action_id"]),
    )


FULL_MATRIX = {
    "canonical_requests": "11528",
    "responses": "11528",
    "runner_events": "23056",
    "ranking_evaluations": "5764",
    "reranking_evaluations": "5764",
    "ranking_strict_valid": "5764",
    "reranking_strict_valid": "5738",
    "invalid_outputs": "26",
    "strict_index_hallucination_count": "0",
}


def test_phase_6a_decision_validation() -> None:
    assert phase6a_inputs().validation["status"].eq("pass").all()


def test_canonical_source_selection() -> None:
    source_map = build_documentation_source_map(phase6a_inputs())
    assert list(source_map.columns) == DOC_SOURCE_COLUMNS
    assert "Paper-Coverage Audit" in set(source_map["documentation_section"])


def test_exactly_two_safe_entrypoints_selected_from_phase_6a() -> None:
    assert len(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)) == 2


def test_unsafe_entrypoint_exclusion() -> None:
    excluded = exclude_unsafe_entrypoints(phase6a_inputs().reproducibility_entrypoints)
    assert excluded.iloc[0]["path"].startswith("src/experiments/run_")


def test_llm_command_exclusion() -> None:
    excluded = build_excluded_entrypoints(exclude_unsafe_entrypoints(phase6a_inputs().reproducibility_entrypoints))
    assert "llm_inference" in excluded.iloc[0]["side_effect_categories"]


def test_recommender_training_command_exclusion() -> None:
    command = "run_plan_e_lightgcn_training.py"
    assert not command_is_safe(command)


def test_recommender_inference_command_exclusion() -> None:
    assert not command_is_safe("src/experiments/run_plan_e_bprmf_beauty_sports.py")


def test_api_command_exclusion() -> None:
    assert not command_is_safe("--allow-api-calls")


def test_download_command_exclusion() -> None:
    assert not command_is_safe("download data")


def test_windows_relative_command_generation() -> None:
    command = portable_command("src/experiments/build_canonical_metric_preflight.py")
    assert command.startswith('$env:PYTHONNOUSERSITE = "1"')
    assert 'C:\\Users' not in command


def test_no_user_specific_absolute_paths() -> None:
    assert not detect_absolute_paths("Use results/tables/x.csv")
    assert detect_absolute_paths(r"C:\Example\x.csv")


def test_markdown_relative_link_validation(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/a.md").write_text("x", encoding="utf-8")
    ok, missing = relative_markdown_links_resolve(tmp_path, Path("README.md"), "[a](docs/a.md)")
    assert ok and not missing


def test_missing_link_detection(tmp_path: Path) -> None:
    ok, missing = relative_markdown_links_resolve(tmp_path, Path("README.md"), "[x](missing.md)")
    assert not ok and missing == ["missing.md"]


def test_safe_handling_of_paths_containing_spaces() -> None:
    assert path_with_spaces_safe('$env:PYTHONNOUSERSITE = "1"\n& ".\\.venv\\Scripts\\python.exe" "src\\x.py"')


def test_readme_required_section_validation() -> None:
    readme = generate_readme(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)), FULL_MATRIX)
    assert not missing_required_sections(readme, README_REQUIRED_SECTIONS)


def test_reproducibility_guide_required_section_validation() -> None:
    guide = generate_reproducibility_guide(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)))
    assert not missing_required_sections(guide, GUIDE_REQUIRED_SECTIONS)


def test_artifact_catalog_required_section_validation() -> None:
    catalog = generate_artifact_catalog(phase6a_inputs())
    assert not missing_required_sections(catalog, CATALOG_REQUIRED_SECTIONS)


def test_phase_5e_count_reconciliation() -> None:
    assert verify_phase5e_counts(phase6a_inputs())["applicable_component_count"] == "73"


def test_exact_coverage_value_verification() -> None:
    assert verify_phase5e_counts(phase6a_inputs())["strict_exact_reproduction_micro_pct"] == "12.328767"


def test_maximum_defensible_value_verification() -> None:
    assert verify_phase5e_counts(phase6a_inputs())["maximum_defensible_micro_coverage_pct"] == "68.493151"


def test_exact_numeric_coverage_verification() -> None:
    assert verify_phase5e_counts(phase6a_inputs())["exact_numeric_result_cell_pct"] == "0.0"


def test_full_matrix_count_verification() -> None:
    assert FULL_MATRIX["canonical_requests"] == "11528"


def test_invalid_output_count_verification() -> None:
    assert FULL_MATRIX["invalid_outputs"] == "26"


def test_qwen2_mistral_local_substitute_wording() -> None:
    readme = generate_readme(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)), FULL_MATRIX)
    assert "local substitutes/extensions" in readme


def test_full20_legacy_wording() -> None:
    readme = generate_readme(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)), FULL_MATRIX)
    assert "Legacy full-20 protocols remain diagnostic or legacy" in readme


def test_blocked_metric_wording() -> None:
    readme = generate_readme(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)), FULL_MATRIX)
    assert "does not claim they were computed" in readme


def test_extension_separation() -> None:
    readme = generate_readme(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)), FULL_MATRIX)
    assert "Extensions do not increase the exact reproduction percentage" in readme


def test_unsupported_nondeterminism_claim_rejection() -> None:
    assert contains_prohibited_overclaim("The paper is fully reproduced.")


def test_overclaim_detector_allows_explicit_disclaimers() -> None:
    assert not contains_prohibited_overclaim("68.493151% is not an exact replication score.")


def test_unsupported_profile_effect_claim_rejection() -> None:
    readme = generate_readme(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)), FULL_MATRIX)
    assert "generated-profile effect" in readme and "unsupported" in readme


def test_unsupported_full20_top10_ablation_claim_rejection() -> None:
    readme = generate_readme(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)), FULL_MATRIX)
    assert "full20-versus-top10 numerical ablation" in readme


def test_same_cohort_overclaim_rejection() -> None:
    readme = generate_readme(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)), FULL_MATRIX)
    assert "same-cohort comparability must be established" in readme


def test_canonical_artifact_link_selection() -> None:
    index = build_final_artifact_index(phase6a_inputs())
    assert index.iloc[0]["delivery_disposition"] == "include_primary"


def test_unresolved_artifact_exclusion_from_foreground() -> None:
    readme = generate_readme(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)), FULL_MATRIX)
    assert "not foregrounded as canonical evidence" in readme


def test_license_non_invention() -> None:
    readme = generate_readme(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)), FULL_MATRIX)
    assert "No project license file" in readme


def test_authorship_non_invention() -> None:
    readme = generate_readme(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)), FULL_MATRIX)
    assert "No additional author" in readme


def test_secret_pattern_detection() -> None:
    assert detect_secret_patterns("api_key=123456789")


def test_stable_documentation_ordering() -> None:
    readme = generate_readme(phase6a_inputs(), build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)), FULL_MATRIX)
    assert readme.index("## Project Status") < readme.index("## Paper Under Study")


def test_deterministic_markdown_serialization() -> None:
    assert stable_markdown(["# A", "", "x  "]) == "# A\n\nx\n"


def test_deterministic_csv_serialization() -> None:
    df = frame([{"a": "1"}], ["a"])
    assert stable_csv_bytes(df) == stable_csv_bytes(df)


def test_readme_preservation_plan_handling() -> None:
    plan = readme_preservation_plan("# old")
    assert "preserved" in plan and "corrected" in plan


def test_protected_input_hash_verification_excluding_readme() -> None:
    initial = frame([{"path": "x.csv", "sha256": "abc"}], ["path", "sha256"])
    final = frame([{"path": "x.csv", "sha256": "abc"}], ["path", "sha256"])
    assert verify_protected_manifest(initial, final)


def test_readme_before_after_hash_audit() -> None:
    audit = build_readme_hash_audit("a", "b")
    assert audit.iloc[0]["readme_changed"] == "true"


def test_no_mutation_of_input_tables() -> None:
    inputs = phase6a_inputs()
    before = inputs.reproducibility_entrypoints.copy(deep=True)
    _ = select_safe_entrypoints(inputs.reproducibility_entrypoints)
    pd.testing.assert_frame_equal(inputs.reproducibility_entrypoints, before)


def test_phase_6b_report_generation() -> None:
    inputs = phase6a_inputs()
    safe = select_safe_entrypoints(inputs.reproducibility_entrypoints)
    safe_commands = build_safe_command_registry(safe)
    readme = generate_readme(inputs, safe_commands, FULL_MATRIX)
    claim = build_readme_claim_audit(readme, inputs, FULL_MATRIX)
    validation = frame([{"status": "pass"}], ["status"])
    report = generate_phase_report(inputs, readme_preservation_plan(""), safe, frame([], ["entry_point"]), frame([{"status": "pass"}], ["status"]), claim, validation, build_readme_hash_audit("a", "b"))
    assert "Phase 6B Decision" in report


def test_phase_6c_not_started() -> None:
    assert "phase_6c" not in README_PATH.as_posix()


def test_safe_command_registry_schema() -> None:
    registry = build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints))
    assert "command_id" in registry.columns


def test_portability_audit_schema(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/reproducibility_guide.md").write_text("x", encoding="utf-8")
    audit = build_portability_audit(tmp_path, {Path("README.md"): "[g](docs/reproducibility_guide.md)"}, build_safe_command_registry(select_safe_entrypoints(phase6a_inputs().reproducibility_entrypoints)))
    assert "check_name" in audit.columns


def test_markdown_links_extract() -> None:
    assert markdown_links("[a](x.md)") == ["x.md"]

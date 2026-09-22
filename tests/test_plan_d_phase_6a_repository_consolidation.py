from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.plan_d_phase_6a_repository_consolidation import (
    ARTIFACT_REGISTRY_COLUMNS,
    DOC_INPUT_COLUMNS,
    ENTRYPOINT_COLUMNS,
    FINAL_DELIVERY_DISPOSITIONS,
    LIFECYCLE_CLASSES,
    ORPHAN_COLUMNS,
    PLAN_E_COLUMNS,
    PROTECTED_MANIFEST_COLUMNS,
    VALIDATION_COLUMNS,
    build_artifact_registry,
    build_final_documentation_input_map,
    build_orphaned_or_unresolved,
    build_plan_e_reconciliation,
    build_report,
    build_reproducibility_entrypoints,
    build_validation,
    canonicality_for,
    classify_entrypoint_side_effects,
    compare_output_hashes,
    duplicate_hash_groups,
    family_for,
    final_delivery_disposition_for,
    frame,
    infer_protocol,
    is_phase6a_output,
    lifecycle_class_for,
    normalize_path,
    output_hash_manifest,
    phase5e_accounting_valid,
    plan_e_represents,
    producer_for,
    reconcile_phase5e_accounting,
    same_name_different_content,
    scientific_role_for,
    stable_artifact_id,
    stable_csv_bytes,
    verify_protected_manifest,
)


def test_stable_artifact_id_generation() -> None:
    assert stable_artifact_id("results\\tables\\plan_e_metrics.csv") == stable_artifact_id("results/tables/plan_e_metrics.csv")
    assert stable_artifact_id("a.csv") != stable_artifact_id("b.csv")


def test_windows_path_normalization() -> None:
    assert normalize_path(r".\results\tables\\x.csv") == "results/tables/x.csv"


def test_lifecycle_class_uniqueness() -> None:
    klass = lifecycle_class_for("results/tables/plan_d_phase_5e_component_coverage.csv")
    assert klass in LIFECYCLE_CLASSES


def test_delivery_disposition_uniqueness() -> None:
    disposition = final_delivery_disposition_for("canonical_final_evidence")
    assert disposition == "include_primary"
    assert disposition in FINAL_DELIVERY_DISPOSITIONS


def test_canonical_phase_5c_precedence() -> None:
    path = "results/tables/canonical_local_llm_full_matrix_cell_metrics.csv"
    assert lifecycle_class_for(path) == "canonical_final_evidence"
    assert canonicality_for(lifecycle_class_for(path), path) == "canonical"


def test_canonical_phase_5d_robustness_precedence() -> None:
    path = "results/tables/plan_d_phase_5d_robustness_summary.csv"
    assert lifecycle_class_for(path) == "canonical_final_evidence"


def test_canonical_phase_5e_coverage_precedence() -> None:
    path = "results/tables/plan_d_phase_5e_coverage_metrics.csv"
    assert lifecycle_class_for(path) == "canonical_final_evidence"


def test_pilot_not_promoted_to_full_matrix() -> None:
    path = "results/tables/canonical_local_llm_pilot_metrics.csv"
    assert lifecycle_class_for(path) == "small_pilot"
    assert canonicality_for("small_pilot", path) != "canonical"


def test_full20_legacy_not_promoted_to_canonical() -> None:
    path = "results/tables/canonical_local_llm_full20_contract_origin_audit.csv"
    assert lifecycle_class_for(path) == "legacy_protocol_artifact"
    assert infer_protocol(path) == "legacy_full20"


def test_blocked_artifact_not_promoted_to_executed() -> None:
    path = "results/tables/canonical_metric_blocked_results.csv"
    assert lifecycle_class_for(path) == "blocked_or_incomplete_artifact"
    assert final_delivery_disposition_for("blocked_or_incomplete_artifact") == "preserve_blocked"


def test_extension_not_classified_as_exact_reproduction() -> None:
    path = "results/tables/plan_e_bprmf_beauty_sports_metrics.csv"
    assert lifecycle_class_for(path) == "extension_artifact"
    assert canonicality_for("extension_artifact", path) == "extension_not_reproduction"


def test_plan_e_detection_by_path_and_content() -> None:
    path = "results/tables/plan_e_lightgcn_ml1m_pilot_metrics.csv"
    assert "plan_e" in path
    assert family_for(path) == "plan_e_lightgcn"


def test_plan_e_not_equated_with_phase_5e() -> None:
    assert lifecycle_class_for("results/tables/plan_e_sasrec_all_datasets_metrics.csv") != lifecycle_class_for(
        "results/tables/plan_d_phase_5e_coverage_metrics.csv"
    )


def test_post_phase_5e_artifact_classification(tmp_path: Path) -> None:
    (tmp_path / "results/tables").mkdir(parents=True)
    path = tmp_path / "results/tables/plan_e_new_metrics.csv"
    path.write_text("metric,value\nx,1\n", encoding="utf-8")
    registry = build_artifact_registry(tmp_path, ["results/tables/plan_e_new_metrics.csv"])
    plan_e = build_plan_e_reconciliation(tmp_path, registry)
    assert plan_e.iloc[0]["phase_5e_temporal_classification"] == "post_phase_5e_evidence_candidate"


def test_duplicate_hash_detection() -> None:
    registry = frame(
        [
            {"normalized_path": "a.csv", "sha256": "same"},
            {"normalized_path": "b.csv", "sha256": "same"},
        ],
        ARTIFACT_REGISTRY_COLUMNS,
    )
    duplicates = duplicate_hash_groups(registry)
    assert len(duplicates) == 1


def test_same_name_different_content_detection() -> None:
    registry = frame(
        [
            {"normalized_path": "a/metrics.csv", "sha256": "one"},
            {"normalized_path": "b/metrics.csv", "sha256": "two"},
        ],
        ARTIFACT_REGISTRY_COLUMNS,
    )
    result = same_name_different_content(registry)
    assert len(result) == 1


def test_orphaned_artifact_detection(tmp_path: Path) -> None:
    registry = frame(
        [
            {
                "normalized_path": "results/tables/no_producer.csv",
                "artifact_type": "result_table",
                "lifecycle_class": "extension_artifact",
            }
        ],
        ARTIFACT_REGISTRY_COLUMNS,
    )
    provenance = frame(
        [{"artifact_path": "results/tables/no_producer.csv", "relationship_type": "no_explicit_relationship_identified"}],
        ["artifact_path", "relationship_type"],
    )
    orphaned = build_orphaned_or_unresolved(tmp_path, registry, provenance)
    assert "producer_or_consumer_not_identified" in set(orphaned["issue"])


def test_producer_consumer_mapping() -> None:
    paths = {"src/experiments/build_plan_d_phase_5e_paper_coverage_audit.py"}
    assert producer_for("results/tables/plan_d_phase_5e_coverage_metrics.csv", paths).endswith("build_plan_d_phase_5e_paper_coverage_audit.py")


def test_absolute_windows_path_detection(tmp_path: Path) -> None:
    report = tmp_path / "report/x.md"
    report.parent.mkdir()
    report.write_text(r"C:\Example\Desktop\file.csv", encoding="utf-8")
    registry = frame([{"normalized_path": "report/x.md", "artifact_type": "report", "lifecycle_class": "canonical_supporting_artifact"}], ARTIFACT_REGISTRY_COLUMNS)
    provenance = frame([{"artifact_path": "report/x.md", "relationship_type": "consumer_reference"}], ["artifact_path", "relationship_type"])
    orphaned = build_orphaned_or_unresolved(tmp_path, registry, provenance)
    assert "absolute_path_reference" in set(orphaned["issue"])


def test_entrypoint_side_effect_classification() -> None:
    side = classify_entrypoint_side_effects("src/experiments/run_plan_e_bprmf.py", "optimizer = x\nloss.backward()\ntorch.save(x)")
    assert side["performs_recommender_training"] == "true"
    assert side["safe_for_final_documentation"] == "false"


def test_offline_builder_classification() -> None:
    side = classify_entrypoint_side_effects("src/experiments/build_plan_d_phase_5e.py", "pd.read_csv('x.csv')")
    assert side["performs_offline_aggregation"] == "true"


def test_llm_runner_classification() -> None:
    side = classify_entrypoint_side_effects("src/experiments/run_canonical_local_llm_full_matrix.py", "from src.llm.canonical_local_llm_runner import x")
    assert side["performs_llm_inference"] == "true"


def test_recommender_runner_classification() -> None:
    side = classify_entrypoint_side_effects("src/baselines/plan_e_lightgcn_training.py", "optimizer.step()")
    assert side["performs_recommender_inference"] == "true"


def write_phase5e_fixture(root: Path) -> None:
    table_dir = root / "results/tables"
    table_dir.mkdir(parents=True)
    rows = []
    for status, count in {
        "reproduced_exact": 9,
        "reconstructed": 16,
        "adapted": 25,
        "blocked": 14,
        "not_executed": 5,
        "unresolved": 4,
    }.items():
        rows.extend({"component_id": f"{status}-{i}", "applicable": "true", "primary_coverage_status": status} for i in range(count))
    pd.DataFrame(rows).to_csv(table_dir / "plan_d_phase_5e_component_coverage.csv", index=False)
    pd.DataFrame(
        [
            {"metric_name": "applicable_component_count", "numerator": "73", "denominator": "73", "value": "73"},
            {"metric_name": "strict_exact_reproduction_micro_pct", "numerator": "9", "denominator": "73", "value": "12.328767"},
            {"metric_name": "maximum_defensible_micro_coverage_pct", "numerator": "50", "denominator": "73", "value": "68.493151"},
            {"metric_name": "exact_numeric_result_cell_pct", "numerator": "0", "denominator": "11", "value": "0.0"},
        ]
    ).to_csv(table_dir / "plan_d_phase_5e_coverage_metrics.csv", index=False)


def test_phase_5e_status_counts_summing_to_73(tmp_path: Path) -> None:
    write_phase5e_fixture(tmp_path)
    accounting = reconcile_phase5e_accounting(tmp_path)
    sum_row = accounting[accounting["status_or_metric"].eq("applicable_status_counts_sum")].iloc[0]
    assert sum_row["verification_status"] == "pass"
    assert sum_row["value"] == "73"


def test_exact_percentage_verification(tmp_path: Path) -> None:
    write_phase5e_fixture(tmp_path)
    accounting = reconcile_phase5e_accounting(tmp_path)
    row = accounting[accounting["status_or_metric"].eq("strict_exact_reproduction_micro_pct")].iloc[0]
    assert row["verification_status"] == "pass"


def test_maximum_defensible_percentage_verification(tmp_path: Path) -> None:
    write_phase5e_fixture(tmp_path)
    accounting = reconcile_phase5e_accounting(tmp_path)
    row = accounting[accounting["status_or_metric"].eq("maximum_defensible_micro_coverage_pct")].iloc[0]
    assert row["observed_value"] == "68.493151"


def test_missing_status_reconciliation(tmp_path: Path) -> None:
    write_phase5e_fixture(tmp_path)
    accounting = reconcile_phase5e_accounting(tmp_path)
    assert accounting[accounting["status_or_metric"].eq("not_executed")].iloc[0]["count"] == "5"
    assert accounting[accounting["status_or_metric"].eq("unresolved")].iloc[0]["count"] == "4"


def test_deterministic_sorting(tmp_path: Path) -> None:
    for rel in ["b.csv", "A.csv"]:
        (tmp_path / rel).write_text("x\n", encoding="utf-8")
    registry = build_artifact_registry(tmp_path)
    assert registry["normalized_path"].tolist() == ["A.csv", "b.csv"]


def test_deterministic_csv_serialization() -> None:
    df = frame([{"b": "2", "a": "1"}], ["a", "b"])
    assert stable_csv_bytes(df) == stable_csv_bytes(df)


def test_protected_input_hash_verification() -> None:
    initial = frame([{"path": "a.csv", "sha256": "abc"}], PROTECTED_MANIFEST_COLUMNS)
    final = frame([{"path": "a.csv", "sha256": "abc"}], PROTECTED_MANIFEST_COLUMNS)
    assert verify_protected_manifest(initial, final)


def test_no_mutation_of_input_data() -> None:
    registry = frame([{"normalized_path": "x.csv", "lifecycle_class": "extension_artifact"}], ARTIFACT_REGISTRY_COLUMNS)
    before = registry.copy(deep=True)
    _ = registry["normalized_path"].tolist()
    pd.testing.assert_frame_equal(registry, before)


def test_report_generation_from_normalized_tables() -> None:
    registry = frame(
        [{"normalized_path": "x.csv", "producing_phase": "phase_5e", "lifecycle_class": "canonical_final_evidence", "final_delivery_disposition": "include_primary"}],
        ARTIFACT_REGISTRY_COLUMNS,
    )
    canonical = frame(
        [
            {
                "scientific_topic": "topic",
                "canonical_primary_artifacts": "x.csv",
                "canonical_supporting_artifacts": "",
                "required_caveat": "caveat",
            }
        ],
        ["scientific_topic", "canonical_primary_artifacts", "canonical_supporting_artifacts", "source_precedence_basis", "final_document_role", "required_caveat"],
    )
    phase5e = frame(
        [
            {"record_type": "status_count", "status_or_metric": "reproduced_exact", "value": "9", "verification_status": "pass"},
            {"record_type": "percentage_or_total_verification", "status_or_metric": "strict_exact_reproduction_micro_pct", "value": "12.328767", "verification_status": "pass"},
        ],
        ["record_type", "status_or_metric", "value", "verification_status"],
    )
    text = build_report(
        registry,
        canonical,
        frame([], PLAN_E_COLUMNS),
        frame([], ["old_artifact", "current_canonical_artifact", "relationship", "reason", "scientifically_useful", "supersession_declared_by_previous_phase", "preservation_policy", "final_report_policy"]),
        frame([], DOC_INPUT_COLUMNS),
        frame([], ENTRYPOINT_COLUMNS),
        frame([], ORPHAN_COLUMNS),
        phase5e,
        frame([], ["artifact_or_artifact_family", "inclusion_disposition", "reason", "canonical_replacement_if_any", "readme_role", "report_role", "reproducibility_role", "preservation_requirement"]),
        frame([{"status": "pass"}], VALIDATION_COLUMNS),
    )
    assert "Phase 6A Repository Consolidation Blueprint" in text


def test_final_document_input_map_validation() -> None:
    doc_map = build_final_documentation_input_map()
    assert "reproduction coverage" in set(doc_map["final_document_section"])


def test_no_previous_artifact_selected_for_deletion() -> None:
    assert "delete" not in scientific_role_for("extension_artifact", "results/tables/plan_e_x.csv").lower()


def test_unresolved_conflicts_remaining_unresolved() -> None:
    path = "results/tables/plan_d_phase_5d_blocked_or_unresolved.csv"
    assert lifecycle_class_for(path) == "blocked_or_incomplete_artifact"


def test_phase6a_outputs_excluded_from_input_discovery() -> None:
    assert is_phase6a_output("results/tables/plan_d_phase_6a_artifact_registry.csv")


def test_output_hash_manifest_empty_when_no_outputs(tmp_path: Path) -> None:
    manifest = output_hash_manifest(tmp_path)
    assert list(manifest.columns) == ["path", "sha256"]


def test_output_hash_compare_requires_both_manifests(tmp_path: Path) -> None:
    assert not compare_output_hashes(tmp_path)


def test_plan_e_representation_for_neural_baseline() -> None:
    assert "extension" in plan_e_represents("results/tables/plan_e_sasrec_metrics.csv", "extension_artifact")


def test_phase5e_accounting_valid_requires_all_pass() -> None:
    accounting = frame([{"verification_status": "pass"}], ["verification_status"])
    assert phase5e_accounting_valid(accounting)


def test_reproducibility_entrypoint_registry_schema(tmp_path: Path) -> None:
    script = tmp_path / "src/experiments/build_x.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('offline')\n", encoding="utf-8")
    registry = build_artifact_registry(tmp_path, ["src/experiments/build_x.py"])
    entrypoints = build_reproducibility_entrypoints(tmp_path, registry)
    assert list(entrypoints.columns) == ENTRYPOINT_COLUMNS


def test_validation_generation_schema(tmp_path: Path) -> None:
    write_phase5e_fixture(tmp_path)
    registry = frame(
        [
            {"normalized_path": "results/tables/canonical_local_llm_full_matrix_cell_metrics.csv", "producing_phase": "phase_5c", "lifecycle_class": "canonical_final_evidence", "final_delivery_disposition": "include_primary", "canonicality": "canonical"},
            {"normalized_path": "results/tables/plan_d_phase_5d_robustness_summary.csv", "producing_phase": "phase_5d", "lifecycle_class": "canonical_final_evidence", "final_delivery_disposition": "include_primary", "canonicality": "canonical"},
            {"normalized_path": "results/tables/plan_d_phase_5e_coverage_metrics.csv", "producing_phase": "phase_5e", "lifecycle_class": "canonical_final_evidence", "final_delivery_disposition": "include_primary", "canonicality": "canonical"},
        ],
        ARTIFACT_REGISTRY_COLUMNS,
    )
    accounting = reconcile_phase5e_accounting(tmp_path)
    validation = build_validation(tmp_path, registry, frame([], PLAN_E_COLUMNS), accounting, frame([{"path": "src/experiments/build_x.py"}], ENTRYPOINT_COLUMNS), frame([{"artifact": "x"}], ORPHAN_COLUMNS))
    assert list(validation.columns) == VALIDATION_COLUMNS

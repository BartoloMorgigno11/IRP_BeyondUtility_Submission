from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f5d_serendipity import (
    ANOMALY_COLUMNS,
    CLAIM_COLUMNS,
    COMPARISON_COLUMNS,
    COMPONENT_COLUMNS,
    COVERAGE_COLUMNS,
    DEFINITION_COMPARISON_COLUMNS,
    METRIC_INVENTORY_COLUMNS,
    METRIC_STATUS_COLUMNS,
    OUTPUT_TABLES,
    PREDICTION_INVENTORY_COLUMNS,
    PREREQUISITE_COLUMNS,
    PRIMITIVE_POLICY_COLUMNS,
    RECONCILIATION_COLUMNS,
    RELEVANCE_POLICY_COLUMNS,
    RESULT_COLUMNS,
    SERENDIPITY_POLICY_COLUMNS,
    SENSITIVITY_COLUMNS,
    TRADEOFF_COLUMNS,
    UNEXPECTEDNESS_POLICY_COLUMNS,
    USER_SUMMARY_COLUMNS,
    VALIDATION_COLUMNS,
    PrimitiveRow,
    binary_relevance_at_k,
    build_frames,
    build_report,
    frame_sha256,
    macro,
    no_absolute_paths,
    output_hashes,
    parse_item_list,
    primitive_list_overlap,
    primitive_unexpectedness,
    protected_hash_status,
    serendipity_value,
    user_serendipity_record,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return build_frames(PROJECT_ROOT)


def test_parsing_relevance_unexpectedness_and_formula_cases() -> None:
    assert parse_item_list('["1","2.0","A"]') == ["1", "2", "A"]
    assert binary_relevance_at_k(["a", "b"], "b") == 1
    assert binary_relevance_at_k(["a", "b"], "c") == 0
    assert primitive_unexpectedness(0) == 1
    assert primitive_unexpectedness(1) == 0
    assert serendipity_value(1, 0) == 1
    assert serendipity_value(1, 1) == 0
    assert serendipity_value(0, 0) == 0
    assert macro([1, 0, 1]) == pytest.approx(2 / 3)


def test_primitive_overlap_history_edge_and_row_diagnostics() -> None:
    primitive = PrimitiveRow(("a", "b", "c", "d", "e"), 0)
    assert primitive_list_overlap(["a", "x", "y", "z", "w"], primitive.items) == 0.2
    assert primitive_list_overlap(["x", "y"], primitive.items) == 0.0
    row = user_serendipity_record(["p", "p", "x"], "p", primitive, user_id="u1", candidate_items={"p", "x"})
    assert row.relevance == 1
    assert row.serendipity == 1
    assert row.duplicate_count == 1
    assert row.invalid_item_count == 2
    expected = user_serendipity_record(["p"], "p", PrimitiveRow(("p",), 1), candidate_items={"p"})
    assert expected.serendipity == 0


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_f5d_output_schemas() -> None:
    built = frames()
    assert list(built["serendipity_metric_inventory"].columns) == METRIC_INVENTORY_COLUMNS
    assert list(built["serendipity_definition_comparison"].columns) == DEFINITION_COMPARISON_COLUMNS
    assert list(built["serendipity_prerequisite_inventory"].columns) == PREREQUISITE_COLUMNS
    assert list(built["primitive_recommender_policy"].columns) == PRIMITIVE_POLICY_COLUMNS
    assert list(built["relevance_policy"].columns) == RELEVANCE_POLICY_COLUMNS
    assert list(built["unexpectedness_policy"].columns) == UNEXPECTEDNESS_POLICY_COLUMNS
    assert list(built["serendipity_policy"].columns) == SERENDIPITY_POLICY_COLUMNS
    assert list(built["prediction_source_inventory"].columns) == PREDICTION_INVENTORY_COLUMNS
    assert list(built["serendipity_components"].columns) == COMPONENT_COLUMNS
    assert list(built["serendipity_results"].columns) == RESULT_COLUMNS
    assert list(built["serendipity_user_summary"].columns) == USER_SUMMARY_COLUMNS
    assert list(built["serendipity_sensitivity"].columns) == SENSITIVITY_COLUMNS
    assert list(built["valid_serendipity_comparisons"].columns) == COMPARISON_COLUMNS
    assert list(built["serendipity_reconciliation"].columns) == RECONCILIATION_COLUMNS
    assert list(built["metric_status_update"].columns) == METRIC_STATUS_COLUMNS
    assert list(built["claim_update"].columns) == CLAIM_COLUMNS
    assert list(built["metric_tradeoffs"].columns) == TRADEOFF_COLUMNS
    assert list(built["coverage_impact"].columns) == COVERAGE_COLUMNS
    assert list(built["anomalies"].columns) == ANOMALY_COLUMNS
    assert list(built["validation"].columns) == VALIDATION_COLUMNS


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_definition_policy_source_inventory_and_status() -> None:
    built = frames()
    metric_inventory = built["serendipity_metric_inventory"]
    primitive = built["primitive_recommender_policy"]
    relevance = built["relevance_policy"]
    unexpectedness = built["unexpectedness_policy"]
    prediction = built["prediction_source_inventory"]
    status = built["metric_status_update"].iloc[0]
    assert metric_inventory["formula_summary"].str.contains("hit", case=False).any()
    assert primitive["primitive_model"].isin(["mostpop_train_only", "random_seeded"]).all()
    assert relevance["relevance_type"].eq("binary").all()
    assert unexpectedness["unexpectedness_formula"].eq("1 - primitive_hit_at_5").all()
    assert prediction["can_compute_serendipity"].eq("true").sum() > 0
    assert status["f5d_status"] == "resolved_adapted"
    assert status["feature_policy_status"] == "not_applicable"


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_results_components_sensitivity_comparisons_and_reconciliation() -> None:
    built = frames()
    results = built["serendipity_results"]
    components = built["serendipity_components"]
    sensitivity = built["serendipity_sensitivity"]
    comparisons = built["valid_serendipity_comparisons"]
    reconciliation = built["serendipity_reconciliation"]
    assert results["value"].astype(float).between(0.0, 1.0).all()
    assert not results.loc[results["reproduction_class"].eq("adapted"), "computation_class"].str.contains("exact").any()
    assert set(components["component"]).issuperset({"relevance_rate", "unexpectedness", "primitive_list_overlap", "final_serendipity"})
    assert not results.loc[results["metric_variant"].str.contains("sensitivity"), "is_reportable"].eq("true").any()
    assert len(sensitivity) > 0
    assert comparisons.empty or comparisons["is_valid_comparison"].eq("true").all()
    assert reconciliation["existing_metric_name"].astype(str).str.contains("serendip", case=False).any()


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_user_summary_tradeoffs_coverage_anomalies_and_validation() -> None:
    built = frames()
    summary = built["serendipity_user_summary"]
    tradeoffs = built["metric_tradeoffs"]
    coverage = built["coverage_impact"].set_index("coverage_metric")
    anomalies = built["anomalies"]
    validation = built["validation"]
    assert {"zero_serendipity_user_rate", "at_least_one_serendipitous_item_rate", "no_relevant_top5_item_rate"}.issubset(set(summary["summary_key"]))
    assert tradeoffs.empty or tradeoffs["is_same_protocol"].eq("true").all()
    assert coverage.loc["exact reproduction coverage", "absolute_change"] == "0"
    assert coverage.loc["exact numeric replication coverage", "absolute_change"] == "0"
    assert anomalies["severity"].isin(["info", "low", "medium", "high", "critical"]).all()
    assert validation["status"].eq("pass").all()


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_determinism_idempotence_input_invariance_and_relative_paths(tmp_path: Path) -> None:
    built = frames()
    first = {key: frame_sha256(value) for key, value in built.items()}
    second = {key: frame_sha256(value) for key, value in build_frames(PROJECT_ROOT).items()}
    assert first == second
    report = build_report(PROJECT_ROOT, built)
    write_outputs(tmp_path, built, report)
    first_hashes = output_hashes(tmp_path)
    write_outputs(tmp_path, built, report)
    second_hashes = output_hashes(tmp_path)
    assert first_hashes == second_hashes
    assert set(first_hashes) == {"report/finalization_f5d_serendipity.md", *(path.as_posix() for path in OUTPUT_TABLES.values())}
    assert all(no_absolute_paths(frame) for frame in built.values())
    assert all(not frame.isna().any().any() for frame in built.values())
    assert protected_hash_status(PROJECT_ROOT) == "checked=1144;changed=0"

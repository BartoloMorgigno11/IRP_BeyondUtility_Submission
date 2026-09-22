from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f5b_ild_policy import (
    ANOMALY_COLUMNS,
    CLAIM_UPDATE_COLUMNS,
    COMPARISON_COLUMNS,
    COVERAGE_IMPACT_COLUMNS,
    FEATURE_COVERAGE_COLUMNS,
    FEATURE_INVENTORY_COLUMNS,
    ILD_POLICY_COLUMNS,
    ILD_RESULTS_COLUMNS,
    OUTPUT_TABLES,
    PREDICTION_INVENTORY_COLUMNS,
    RECONCILIATION_COLUMNS,
    REPORT_PATH,
    SENSITIVITY_COLUMNS,
    USER_SUMMARY_COLUMNS,
    VALIDATION_COLUMNS,
    build_frames,
    build_report,
    cosine_dissimilarity,
    frame_sha256,
    ild_for_user,
    jaccard_dissimilarity,
    macro_ild,
    no_absolute_paths,
    output_hashes,
    parse_feature_tokens,
    parse_item_list,
    protected_hash_status,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return build_frames(PROJECT_ROOT)


def test_feature_parsing_and_item_list_parsing() -> None:
    assert parse_feature_tokens("Beauty | Skin Care | Face") == ("beauty", "face", "skin care")
    assert parse_feature_tokens('["Action", ["Sci-Fi", "Action"]]') == ("action", "sci-fi")
    assert parse_feature_tokens("") == ()
    assert parse_item_list('["i1","i2","i3"]') == ["i1", "i2", "i3"]


def test_cosine_and_jaccard_similarity_cases() -> None:
    assert cosine_dissimilarity({"a"}, {"a"}) == 0.0
    assert jaccard_dissimilarity({"a"}, {"a"}) == 0.0
    assert cosine_dissimilarity({"a"}, {"b"}) == 1.0
    assert jaccard_dissimilarity({"a"}, {"b"}) == 1.0


def test_duplicate_disjoint_identical_missing_and_invalid_cases() -> None:
    features = {"i1": frozenset({"a"}), "i2": frozenset({"a"}), "i3": frozenset({"b"}), "i4": frozenset()}
    assert ild_for_user(["i1", "i1"], features).ild_value == 0.0
    assert ild_for_user(["i1", "i2"], features).ild_value == 0.0
    assert ild_for_user(["i1", "i3"], features).ild_value == 1.0
    missing = ild_for_user(["i1", "i4"], features)
    assert missing.ild_value is None
    assert missing.missing_feature_item_count == 1
    invalid = ild_for_user(["i1", "unknown"], features)
    assert invalid.ild_value is None
    assert invalid.invalid_item_count == 1
    assert ild_for_user(["i1"], features).validation_status == "blocked_less_than_two_feature_items"


def test_ild_formula_unordered_pairs_and_macro_aggregation() -> None:
    features = {"a": frozenset({"x"}), "b": frozenset({"y"}), "c": frozenset({"x"})}
    row = ild_for_user(["a", "b", "c"], features, similarity_function="jaccard")
    assert row.pair_count == 3
    assert row.valid_item_count == 3
    assert row.ild_value == 2.0 / 3.0
    assert macro_ild([row, ild_for_user(["a", "c"], features, similarity_function="jaccard")]) == 1.0 / 3.0


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_f5b_output_schemas() -> None:
    built = frames()
    assert list(built["item_feature_source_inventory"].columns) == FEATURE_INVENTORY_COLUMNS
    assert list(built["ild_policy"].columns) == ILD_POLICY_COLUMNS
    assert list(built["prediction_source_inventory"].columns) == PREDICTION_INVENTORY_COLUMNS
    assert list(built["item_feature_coverage"].columns) == FEATURE_COVERAGE_COLUMNS
    assert list(built["ild_results"].columns) == ILD_RESULTS_COLUMNS
    assert list(built["ild_user_summary"].columns) == USER_SUMMARY_COLUMNS
    assert list(built["ild_sensitivity"].columns) == SENSITIVITY_COLUMNS
    assert list(built["valid_ild_comparisons"].columns) == COMPARISON_COLUMNS
    assert list(built["ild_reconciliation"].columns) == RECONCILIATION_COLUMNS
    assert list(built["metric_status_update"].columns)[0] == "metric_definition_id"
    assert list(built["claim_update"].columns) == CLAIM_UPDATE_COLUMNS
    assert list(built["coverage_impact"].columns) == COVERAGE_IMPACT_COLUMNS
    assert list(built["anomalies"].columns) == ANOMALY_COLUMNS
    assert list(built["validation"].columns) == VALIDATION_COLUMNS


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_policy_and_source_inventory_content() -> None:
    built = frames()
    policy = built["ild_policy"]
    features = built["item_feature_source_inventory"]
    predictions = built["prediction_source_inventory"]
    assert policy["is_primary_policy"].eq("true").sum() == 3
    assert policy["is_sensitivity_policy"].eq("true").sum() == 3
    assert set(policy.loc[policy["is_primary_policy"].eq("true"), "similarity_function"]) == {"cosine"}
    assert {"genre", "category_path", "title", "description", "artist"}.issubset(set(features["feature_type"]))
    assert predictions["can_compute_ild"].eq("true").sum() > 0
    assert not predictions.loc[predictions["is_pilot"].eq("true"), "is_primary_source"].eq("true").any()


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_results_sensitivity_comparisons_reconciliation_and_metric_status() -> None:
    built = frames()
    results = built["ild_results"]
    sensitivity = built["ild_sensitivity"]
    comparisons = built["valid_ild_comparisons"]
    reconciliation = built["ild_reconciliation"]
    status = built["metric_status_update"]
    assert results["similarity_function"].isin({"cosine", "jaccard"}).all()
    assert results.loc[results["ild_mean"].astype(str).str.len().gt(0), "ild_mean"].astype(float).between(0.0, 1.0).all()
    assert not results.loc[results["similarity_function"].eq("jaccard"), "is_primary_result"].eq("true").any()
    assert len(sensitivity) > 0
    assert comparisons["is_valid_comparison"].eq("true").all()
    assert reconciliation["is_reconciled"].isin({"true", "false"}).all()
    assert status.iloc[0]["f5b_status"] == "resolved_adapted"
    assert status.iloc[0]["paper_equivalence"] == "adapted_not_exact"


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_coverage_impact_exact_metrics_do_not_increase_and_claims_are_safe() -> None:
    built = frames()
    coverage = built["coverage_impact"].set_index("coverage_metric")
    claims = built["claim_update"]
    assert coverage.loc["exact reproduction coverage", "absolute_change"] == "0"
    assert coverage.loc["exact numeric replication coverage", "absolute_change"] == "0"
    assert int(coverage.loc["consolidated metric count", "f5b_value"]) == int(
        float(coverage.loc["consolidated metric count", "previous_value"])
    ) + 1
    assert claims["is_safe_for_report"].eq("true").sum() >= 1


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_feature_coverage_valid_user_rate_and_validation() -> None:
    built = frames()
    feature_coverage = built["item_feature_coverage"]
    validation = built["validation"]
    assert feature_coverage["feature_coverage"].astype(float).between(0.0, 1.0).all()
    nonempty_user_rates = feature_coverage.loc[feature_coverage["valid_user_rate"].astype(str).str.len().gt(0), "valid_user_rate"]
    assert nonempty_user_rates.astype(float).between(0.0, 1.0).all()
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
    assert set(first_hashes) == {REPORT_PATH.as_posix(), *(path.as_posix() for path in OUTPUT_TABLES.values())}
    assert all(no_absolute_paths(frame) for frame in built.values())
    assert all(not frame.isna().any().any() for frame in built.values())
    assert protected_hash_status(PROJECT_ROOT) == "checked=1144;changed=0"

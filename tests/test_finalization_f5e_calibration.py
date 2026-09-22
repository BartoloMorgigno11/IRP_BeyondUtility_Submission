from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f5e_calibration import (
    ANOMALY_COLUMNS,
    CALIBRATION_POLICY_COLUMNS,
    CLAIM_COLUMNS,
    COMPARISON_COLUMNS,
    COVERAGE_COLUMNS,
    COVERAGE_IMPACT_COLUMNS,
    DEFINITION_COMPARISON_COLUMNS,
    METRIC_INVENTORY_COLUMNS,
    METRIC_STATUS_COLUMNS,
    OUTPUT_TABLES,
    PAPER_STATUS_COLUMNS,
    PREDICTION_INVENTORY_COLUMNS,
    PREREQUISITE_COLUMNS,
    RECOMMENDATION_POLICY_COLUMNS,
    RECONCILIATION_COLUMNS,
    RESULT_COLUMNS,
    SENSITIVITY_COLUMNS,
    TRADEOFF_COLUMNS,
    USER_PROFILE_POLICY_COLUMNS,
    USER_SUMMARY_COLUMNS,
    VALIDATION_COLUMNS,
    FeatureIndex,
    UserProfile,
    build_frames,
    build_report,
    calibration_score,
    feature_distribution,
    frame_sha256,
    hellinger_distance,
    jensen_shannon_divergence,
    no_absolute_paths,
    output_hashes,
    parse_feature_tokens,
    parse_item_list,
    protected_hash_status,
    recommendation_distribution,
    user_calibration_record,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return build_frames(PROJECT_ROOT)


def synthetic_index() -> FeatureIndex:
    return FeatureIndex(
        dataset="Synthetic",
        feature_policy_id="synthetic",
        feature_source_id="synthetic",
        feature_file="synthetic.csv",
        feature_type="category",
        item_features={"i1": ("a",), "i2": ("a", "b"), "i3": ("c",)},
        all_item_ids=frozenset({"i1", "i2", "i3", "missing_feature"}),
        vocabulary=frozenset({"a", "b", "c"}),
    )


def test_feature_distribution_fractional_multilabel_and_normalization() -> None:
    assert parse_item_list('["1","2.0","A"]') == ["1", "2", "A"]
    assert parse_feature_tokens("Beauty | Skin Care | Face") == ("beauty", "face", "skin care")
    distribution = feature_distribution(["i2", "i1"], synthetic_index().item_features)
    assert distribution == pytest.approx({"a": 0.75, "b": 0.25})
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_jsd_score_hellinger_identical_disjoint_partial_zero_support() -> None:
    left = {"a": 1.0}
    assert jensen_shannon_divergence(left, left) == pytest.approx(0.0)
    assert calibration_score(0.0) == pytest.approx(1.0)
    assert hellinger_distance(left, left) == pytest.approx(0.0)
    assert jensen_shannon_divergence({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    assert hellinger_distance({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    partial = jensen_shannon_divergence({"a": 0.5, "b": 0.5}, {"a": 1.0})
    assert 0.0 < partial < 1.0


def test_recommendation_distribution_missing_invalid_duplicate_and_empty_history() -> None:
    index = synthetic_index()
    rec_dist, total, with_features, invalid, missing = recommendation_distribution(["i1", "i1", "bad", "missing_feature"], index)
    assert rec_dist == pytest.approx({"a": 1.0})
    assert total == 5
    assert with_features == 2
    assert invalid == 2
    assert missing == 1
    valid = user_calibration_record("u1", ["i1", "i2"], UserProfile({"a": 1.0}, 1, 1), index)
    assert valid.validation_status == "valid"
    assert valid.jsd is not None and 0.0 <= valid.jsd <= 1.0
    empty = user_calibration_record("u2", ["i1"], UserProfile({}, 1, 0), index)
    assert empty.jsd is None
    assert empty.validation_status == "blocked_empty_profile_features"


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_f5e_output_schemas() -> None:
    built = frames()
    assert list(built["calibration_metric_inventory"].columns) == METRIC_INVENTORY_COLUMNS
    assert list(built["calibration_definition_comparison"].columns) == DEFINITION_COMPARISON_COLUMNS
    assert list(built["paper_calibration_status"].columns) == PAPER_STATUS_COLUMNS
    assert list(built["calibration_prerequisite_inventory"].columns) == PREREQUISITE_COLUMNS
    assert list(built["user_profile_policy"].columns) == USER_PROFILE_POLICY_COLUMNS
    assert list(built["recommendation_distribution_policy"].columns) == RECOMMENDATION_POLICY_COLUMNS
    assert list(built["calibration_policy"].columns) == CALIBRATION_POLICY_COLUMNS
    assert list(built["prediction_source_inventory"].columns) == PREDICTION_INVENTORY_COLUMNS
    assert list(built["calibration_coverage"].columns) == COVERAGE_COLUMNS
    assert list(built["calibration_results"].columns) == RESULT_COLUMNS
    assert list(built["calibration_user_summary"].columns) == USER_SUMMARY_COLUMNS
    assert list(built["calibration_sensitivity"].columns) == SENSITIVITY_COLUMNS
    assert list(built["valid_calibration_comparisons"].columns) == COMPARISON_COLUMNS
    assert list(built["calibration_reconciliation"].columns) == RECONCILIATION_COLUMNS
    assert list(built["metric_status_update"].columns) == METRIC_STATUS_COLUMNS
    assert list(built["claim_update"].columns) == CLAIM_COLUMNS
    assert list(built["metric_tradeoffs"].columns) == TRADEOFF_COLUMNS
    assert list(built["coverage_impact"].columns) == COVERAGE_IMPACT_COLUMNS
    assert list(built["anomalies"].columns) == ANOMALY_COLUMNS
    assert list(built["validation"].columns) == VALIDATION_COLUMNS


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_paper_status_policy_inventory_and_prediction_sources() -> None:
    built = frames()
    paper = built["paper_calibration_status"].iloc[0]
    metric_inventory = built["calibration_metric_inventory"]
    user_policy = built["user_profile_policy"]
    rec_policy = built["recommendation_distribution_policy"]
    cal_policy = built["calibration_policy"]
    prediction = built["prediction_source_inventory"]
    assert paper["classification"] in {"extension_only_metric", "paper_ambiguous_target"}
    assert metric_inventory["metric_variant"].str.contains("probability", case=False).any()
    assert user_policy.loc[user_policy["is_primary_policy"].eq("true"), "history_scope"].eq("train_only").all()
    assert rec_policy.loc[rec_policy["is_primary_policy"].eq("true"), "rank_weighting"].eq("uniform").all()
    assert cal_policy["distance_or_divergence"].str.contains("Jensen-Shannon|Hellinger|not_applicable", regex=True).all()
    assert prediction["can_compute_calibration"].eq("true").sum() > 0
    assert "LastFM" in set(prediction.loc[prediction["can_compute_calibration"].eq("false"), "dataset"])


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_results_coverage_summary_sensitivity_and_comparisons() -> None:
    built = frames()
    prediction = built["prediction_source_inventory"]
    results = built["calibration_results"]
    coverage = built["calibration_coverage"]
    summary = built["calibration_user_summary"]
    sensitivity = built["calibration_sensitivity"]
    comparisons = built["valid_calibration_comparisons"]
    computable = prediction["can_compute_calibration"].eq("true").sum()
    assert len(results) == computable * 2
    assert results["value"].astype(float).between(0.0, 1.0).all()
    assert results["paper_equivalence"].eq("not_paper_metric").all()
    assert coverage["valid_user_rate"].replace("", "0").astype(float).between(0.0, 1.0).all()
    assert {"perfect_alignment_rate", "high_divergence_rate", "history_length_1_4"}.issubset(set(summary["summary_key"]))
    assert len(sensitivity) == computable
    assert sensitivity["sensitivity_axis"].str.contains("Hellinger").all()
    assert comparisons.empty or comparisons["is_valid_comparison"].eq("true").all()


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_reconciliation_status_claims_tradeoffs_coverage_anomalies_validation() -> None:
    built = frames()
    reconciliation = built["calibration_reconciliation"]
    status = built["metric_status_update"].iloc[0]
    claims = built["claim_update"]
    tradeoffs = built["metric_tradeoffs"]
    coverage_impact = built["coverage_impact"].set_index("coverage_metric")
    anomalies = built["anomalies"]
    validation = built["validation"]
    assert reconciliation["existing_metric_name"].str.contains("calibration").any()
    assert status["f5e_status"] == "resolved_extension_only"
    assert claims["is_safe_for_report"].eq("true").sum() >= 5
    assert tradeoffs.empty or tradeoffs["is_same_protocol"].eq("true").all()
    assert coverage_impact.loc["exact reproduction coverage", "absolute_change"] == "0"
    assert coverage_impact.loc["exact numeric replication coverage", "absolute_change"] == "0"
    assert coverage_impact.loc["ambiguous metric count", "f5e_value"] == "0"
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
    assert set(first_hashes) == {"report/finalization_f5e_calibration.md", *(path.as_posix() for path in OUTPUT_TABLES.values())}
    assert all(no_absolute_paths(frame) for frame in built.values())
    assert all(not frame.isna().any().any() for frame in built.values())
    assert protected_hash_status(PROJECT_ROOT) == "checked=1144;changed=0"


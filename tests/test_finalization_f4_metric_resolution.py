from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.finalization_f4_metric_resolution import (
    ANOMALY_COLUMNS,
    CLAIM_UPDATE_COLUMNS,
    COVERAGE_IMPACT_COLUMNS,
    DEFINITION_COMPARISON_COLUMNS,
    F5_COLUMNS,
    METRIC_INVENTORY_COLUMNS,
    MISSING_TRIAGE_COLUMNS,
    PROBLEMATIC_METRICS,
    RECONCILIATION_COLUMNS,
    RESOLVED_RESULTS_COLUMNS,
    RESOLUTION_DECISIONS,
    TRIAGE_CATEGORIES,
    TRIAGE_STATUSES,
    VALIDATION_COLUMNS,
    build_frames,
    final_decision,
    frame_sha256,
    no_absolute_paths,
    output_hashes,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.cleanup_historical, pytest.mark.phase_snapshot]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return build_frames(PROJECT_ROOT)


def test_f4_output_schemas() -> None:
    built = frames()
    assert list(built["metric_resolution_inventory"].columns) == METRIC_INVENTORY_COLUMNS
    assert list(built["metric_definition_comparison"].columns) == DEFINITION_COMPARISON_COLUMNS
    assert list(built["resolved_metric_results"].columns) == RESOLVED_RESULTS_COLUMNS
    assert list(built["metric_reconciliation"].columns) == RECONCILIATION_COLUMNS
    assert list(built["missing_evidence_triage"].columns) == MISSING_TRIAGE_COLUMNS
    assert list(built["f5_experiment_candidates"].columns) == F5_COLUMNS
    assert list(built["claim_update"].columns) == CLAIM_UPDATE_COLUMNS
    assert list(built["coverage_impact"].columns) == COVERAGE_IMPACT_COLUMNS
    assert list(built["anomalies"].columns) == ANOMALY_COLUMNS
    assert list(built["validation"].columns) == VALIDATION_COLUMNS


def test_f4_is_deterministic_in_memory() -> None:
    first = {key: frame_sha256(value) for key, value in frames().items()}
    second = {key: frame_sha256(value) for key, value in build_frames(PROJECT_ROOT).items()}
    assert first == second


def test_f4_output_hashes_are_idempotent_in_tmpdir(tmp_path: Path) -> None:
    built = frames()
    write_outputs(tmp_path, built, "temporary f4 report\n")
    first = output_hashes(tmp_path)
    write_outputs(tmp_path, built, "temporary f4 report\n")
    second = output_hashes(tmp_path)
    assert first == second


def test_problematic_metrics_are_identified_and_decided() -> None:
    inventory = frames()["metric_resolution_inventory"]
    problematic = set(inventory.loc[inventory["f3_blocking_issue"].str.len().gt(0), "f3_metric_definition_id"])
    assert problematic == PROBLEMATIC_METRICS
    decisions = dict(zip(inventory["f3_metric_definition_id"], inventory["resolution_decision"], strict=False))
    assert decisions["catalog_coverage_at_5"] == "resolved_adapted"
    assert decisions["calibration"] == "blocked_missing_definition"
    assert decisions["popularity_bias_at_5"].startswith("blocked")
    assert decisions["intra_list_diversity_at_5"].startswith("blocked")
    assert decisions["serendipity_at_5"] == "blocked_missing_inputs"


def test_all_f3_metrics_and_missing_evidence_are_covered() -> None:
    built = frames()
    assert len(built["metric_resolution_inventory"]) == 17
    assert len(built["missing_evidence_triage"]) == 42
    assert built["metric_resolution_inventory"]["f3_metric_definition_id"].is_unique
    assert built["missing_evidence_triage"]["missing_evidence_id"].is_unique


def test_enums_are_valid() -> None:
    built = frames()
    assert set(built["metric_resolution_inventory"]["resolution_decision"]).issubset(RESOLUTION_DECISIONS)
    assert set(built["missing_evidence_triage"]["resolution_category"]).issubset(TRIAGE_CATEGORIES)
    assert set(built["missing_evidence_triage"]["resolution_status"]).issubset(TRIAGE_STATUSES)
    assert set(built["resolved_metric_results"]["computation_class"]).issubset(
        {"exact_offline_recomputation", "functional_offline_recomputation", "adapted_offline_recomputation", "extension_offline_computation"}
    )


def test_formula_provenance_and_no_unsourced_formula() -> None:
    inventory = frames()["metric_resolution_inventory"]
    sourced = inventory[inventory["local_formula_summary"].str.len().gt(0)]
    assert not sourced.empty
    assert sourced["local_file"].str.len().gt(0).all()
    comparisons = frames()["metric_definition_comparison"]
    assert comparisons["evidence"].str.len().gt(0).all()
    assert (comparisons["recommended_definition"] != "").all()


def test_offline_computation_policy_and_reconciliation() -> None:
    built = frames()
    resolved = built["resolved_metric_results"]
    reconciliation = built["metric_reconciliation"]
    assert len(resolved) == 96
    assert resolved["metric"].eq("catalog_coverage_at_5").all()
    assert resolved["computation_class"].eq("adapted_offline_recomputation").all()
    assert resolved["source_prediction_file"].str.len().gt(0).all()
    assert resolved["source_manifest_file"].str.len().gt(0).all()
    assert len(reconciliation) == len(resolved)
    assert reconciliation["is_reconciled"].eq("true").all()


def test_claim_update_and_coverage_impact_are_conservative() -> None:
    built = frames()
    claims = built["claim_update"]
    assert len(claims) == 9
    assert claims["is_safe_for_report"].eq("true").sum() == 6
    assert claims["is_safe_for_report"].eq("false").sum() == 3
    claim_8 = claims.loc[claims["claim_id"].eq("F3-CLAIM-008")].iloc[0]
    assert claim_8["is_safe_for_report"] == "false"
    coverage = built["coverage_impact"]
    exact = coverage.loc[coverage["coverage_metric"].eq("exact reproduction coverage")].iloc[0]
    numeric = coverage.loc[coverage["coverage_metric"].eq("exact numerical coverage")].iloc[0]
    assert exact["absolute_change"] in {"0", "0.0", ""}
    assert numeric["absolute_change"] in {"0", "0.0", ""}
    consolidated = coverage.loc[coverage["coverage_metric"].eq("consolidated metric count")].iloc[0]
    assert consolidated["f3_value"] == "6"
    assert consolidated["f4_value"] == "7"


def test_anomalies_and_validation() -> None:
    built = frames()
    anomalies = built["anomalies"]
    assert len(anomalies) >= 1
    assert set(anomalies["severity"]).issubset({"info", "low", "medium", "high", "critical"})
    assert anomalies["can_continue"].eq("true").all()
    validation = built["validation"]
    assert validation["status"].eq("pass").all()
    assert final_decision(validation) == "finalization_f4_complete"


def test_missing_triage_counts_and_f5_candidates() -> None:
    triage = frames()["missing_evidence_triage"]
    assert triage["resolution_status"].eq("resolved").sum() == 1
    assert triage["resolution_status"].eq("ready_for_f5").sum() >= 10
    assert triage["resolution_status"].eq("conditional").sum() >= 1
    assert triage["resolution_status"].eq("blocked").sum() >= 1
    f5 = frames()["f5_experiment_candidates"]
    assert not f5.empty
    assert f5["source_missing_evidence_ids"].str.len().gt(0).all()
    assert f5["recommended_order"].astype(int).is_monotonic_increasing


def test_protected_inputs_and_relative_paths() -> None:
    validation = frames()["validation"]
    protected = validation.loc[validation["check_name"].eq("protected_inputs_unchanged")].iloc[0]
    assert protected["status"] == "pass"
    assert "checked=1144" in protected["observed"]
    assert "changed=0" in protected["observed"]
    for frame in frames().values():
        assert no_absolute_paths(frame)
        assert not frame.isna().any().any()

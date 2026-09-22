"""Tests for the F6L LastFM V4 ranking metrics freeze consolidation."""

from __future__ import annotations

from src.analysis import phase_f6l_lastfm_v4_ranking_metrics_freeze as freeze


def test_fail_strict_table_uses_authoritative_denominators() -> None:
    fail_rows = freeze.read_csv(freeze.SOURCE_PATHS["fail_strict"])
    validity_rows = freeze.read_csv(freeze.SOURCE_PATHS["validity"])

    table = freeze.build_fail_strict_table(fail_rows, validity_rows)

    assert len(table) == 4
    assert [row["total_users"] for row in table] == ["1000", "1000", "1000", "1000"]
    assert [row["strict_valid"] for row in table] == ["1000", "1000", "983", "967"]
    assert [row["structural_invalid"] for row in table] == ["0", "0", "17", "33"]
    assert [row["hit_count_at_5"] for row in table] == ["458", "419", "307", "319"]
    assert all(row["MRR"] == "not_calculated" for row in table)


def test_audit_passes_for_authoritative_v4_run() -> None:
    fail_rows = freeze.read_csv(freeze.SOURCE_PATHS["fail_strict"])
    complete_rows = freeze.read_csv(freeze.SOURCE_PATHS["complete_case"])
    validity_rows = freeze.read_csv(freeze.SOURCE_PATHS["validity"])
    taxonomy_rows = freeze.read_csv(freeze.SOURCE_PATHS["failure_taxonomy"])
    final_validation_rows = freeze.read_csv(freeze.SOURCE_PATHS["final_validation"])
    state_rows = freeze.read_csv(freeze.TABLES_DIR / f"{freeze.SOURCE_PREFIX}_state.csv")
    stats_rows = [
        {"metric_delta_matches_fail_strict_artifact": "pass"},
    ]

    audit_rows = freeze.build_audit_table(
        state_rows,
        validity_rows,
        fail_rows,
        complete_rows,
        taxonomy_rows,
        final_validation_rows,
        stats_rows,
    )

    assert len(audit_rows) == 16
    assert all(row["status"] == "pass" for row in audit_rows)


def test_comparison_table_contains_required_cross_model_delta() -> None:
    comparisons = freeze.build_comparison_table(
        freeze.read_csv(freeze.SOURCE_PATHS["fail_strict"]),
        freeze.read_csv(freeze.SOURCE_PATHS["complete_case"]),
        freeze.read_csv(freeze.SOURCE_PATHS["validity"]),
        freeze.read_csv(freeze.SOURCE_PATHS["beyond"]),
    )

    target = [
        row
        for row in comparisons
        if row["comparison"] == "mistral_base_minus_qwen2_base"
        and row["metric"] == "HR@5_fail_strict"
    ]

    assert target
    assert target[0]["delta"] == "-0.151"


def test_freeze_constants_prevent_reranking_authorization_drift() -> None:
    assert freeze.RUN_ID == "20260804_131959"
    assert freeze.PASS_TOKEN == "f6l_lastfm_v4_ranking_metrics_consolidated_and_frozen_for_reranking"
    assert freeze.FROZEN_STATE == "ranking_metrics_frozen_before_reranking"

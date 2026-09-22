from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest
import pandas as pd

from src.analysis.finalization_f5c_popularity_bias import (
    BIAS_RESOLUTION_COLUMNS,
    CLAIM_COLUMNS,
    COMPARISON_COLUMNS,
    COVERAGE_COLUMNS,
    DEFINITION_COMPARISON_COLUMNS,
    GROUP_EXPOSURE_COLUMNS,
    ITEM_POLICY_COLUMNS,
    ITEM_STAT_COLUMNS,
    METRIC_INVENTORY_COLUMNS,
    METRIC_STATUS_COLUMNS,
    OUTPUT_TABLES,
    POPULARITY_RESULT_COLUMNS,
    PREDICTION_INVENTORY_COLUMNS,
    RECONCILIATION_COLUMNS,
    REPORT_PATH,
    SHIFT_SUMMARY_COLUMNS,
    VALIDATION_COLUMNS,
    PopularityIndex,
    build_frames,
    build_report,
    compute_user_popularity,
    frame_sha256,
    gini,
    no_absolute_paths,
    output_hashes,
    parse_item_list,
    popreo_from_hits,
    protected_hash_status,
    write_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def frames() -> dict[str, pd.DataFrame]:
    return build_frames(PROJECT_ROOT)


def synthetic_index() -> PopularityIndex:
    counts = {"i1": 10, "i2": 2, "i3": 1}
    return PopularityIndex(
        dataset="Toy",
        policy_id="toy_policy",
        interactions_path=Path("toy/interactions.csv"),
        catalog_path=Path("toy/catalog.csv"),
        source_sha256="source",
        catalog_sha256="catalog",
        counts=counts,
        normalized_counts={"i1": 1.0, "i2": 0.2, "i3": 0.1},
        profile_mean={"u1": 6.0},
        catalog_items=frozenset({"i1", "i2", "i3", "cold"}),
        train_users=1,
        interaction_rows=3,
        train_rows=3,
        threshold=5.0,
        q20=1.4,
        q80=5.0,
        max_count=10,
        head_items=frozenset({"i1"}),
        tail_items=frozenset({"i2", "i3"}),
    )


def test_parsing_gini_and_popreo_guard() -> None:
    assert parse_item_list('["1","2.0","B003"]') == ["1", "2", "B003"]
    assert parse_item_list("a;b;c") == ["a", "b", "c"]
    assert gini([1, 1, 1]) == 0.0
    with pytest.raises(ValueError, match="zero-exposure"):
        popreo_from_hits([1, 0], ["i1", "i2"], {"i1": 1, "i2": 2}, [-1, 0, 1, 2, 3, 4])


def test_user_popularity_metrics_do_not_conflate_bias_alias() -> None:
    row = compute_user_popularity(["i1", "i2", "cold", "missing"], synthetic_index(), user_id="u1")
    assert row.raw_mean == 3.0
    assert row.normalized_mean == 0.3
    assert row.head_rate == 0.25
    assert row.tail_rate == 0.25
    assert row.shift == -3.0
    assert row.lift == 0.5
    assert row.missing_popularity_count == 2
    assert row.invalid_item_count == 1


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_f5c_output_schemas() -> None:
    built = frames()
    assert list(built["popularity_metric_inventory"].columns) == METRIC_INVENTORY_COLUMNS
    assert list(built["popularity_definition_comparison"].columns) == DEFINITION_COMPARISON_COLUMNS
    assert list(built["item_popularity_policy"].columns) == ITEM_POLICY_COLUMNS
    assert list(built["item_popularity_statistics"].columns) == ITEM_STAT_COLUMNS
    assert list(built["prediction_source_inventory"].columns) == PREDICTION_INVENTORY_COLUMNS
    assert list(built["popularity_results"].columns) == POPULARITY_RESULT_COLUMNS
    assert list(built["item_group_exposure"].columns) == GROUP_EXPOSURE_COLUMNS
    assert list(built["user_popularity_shift_summary"].columns) == SHIFT_SUMMARY_COLUMNS
    assert list(built["valid_popularity_comparisons"].columns) == COMPARISON_COLUMNS
    assert list(built["popularity_reconciliation"].columns) == RECONCILIATION_COLUMNS
    assert list(built["popularity_bias_resolution"].columns) == BIAS_RESOLUTION_COLUMNS
    assert list(built["metric_status_update"].columns) == METRIC_STATUS_COLUMNS
    assert list(built["claim_update"].columns) == CLAIM_COLUMNS
    assert list(built["coverage_impact"].columns) == COVERAGE_COLUMNS
    assert list(built["validation"].columns) == VALIDATION_COLUMNS


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_metric_policy_and_prediction_inventory_content() -> None:
    built = frames()
    metrics = built["popularity_metric_inventory"]
    policy = built["item_popularity_policy"]
    predictions = built["prediction_source_inventory"]
    assert {"arp_at_5", "aplt_at_5", "popreo_at_5", "popularity_bias_at_5"}.issubset(set(metrics["metric_definition_id"]))
    assert metrics.set_index("metric_definition_id").loc["popularity_bias_at_5", "f5c_resolution"] == "superseded_alias"
    assert metrics.set_index("metric_definition_id").loc["popreo_at_5", "f5c_resolution"] == "blocked_missing_denominator_policy"
    assert policy["policy_status"].eq("active").sum() == 3
    assert policy.set_index("dataset").loc["LastFM", "policy_status"] == "blocked"
    assert policy.loc[policy["policy_status"].eq("active"), "mid_rule"].str.contains("not defined").all()
    assert predictions["can_compute_arp"].eq("true").sum() > 0
    assert predictions["can_compute_popreo"].eq("true").sum() == 0


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_results_reconciliation_metric_status_and_coverage() -> None:
    built = frames()
    results = built["popularity_results"]
    reconciliation = built["popularity_reconciliation"]
    status = built["metric_status_update"].set_index("metric_definition_id")
    coverage = built["coverage_impact"].set_index("coverage_metric")
    assert not results["metric_definition_id"].eq("popularity_bias_at_5").any()
    assert not results["metric_definition_id"].eq("popreo_at_5").any()
    arp = results.loc[results["metric_definition_id"].eq("arp_at_5"), ["prediction_source_id", "value"]]
    avg = results.loc[results["metric_definition_id"].eq("average_recommendation_popularity_at_5"), ["prediction_source_id", "value"]]
    merged = arp.merge(avg, on="prediction_source_id", suffixes=("_arp", "_avg"))
    assert not merged.empty
    assert (merged["value_arp"] == merged["value_avg"]).all()
    assert results.loc[results["metric_definition_id"].isin(["aplt_at_5", "head_exposure_at_5"]), "value"].astype(float).between(0.0, 1.0).all()
    assert reconciliation["existing_metric"].astype(str).str.contains("popularity_bias|Popularity Bias", case=False, regex=True).any()
    assert status.loc["popularity_bias_at_5", "f5c_status"] == "superseded_alias"
    assert status.loc["popreo_at_5", "f5c_status"] == "blocked_missing_denominator_policy"
    assert coverage.loc["exact reproduction coverage", "absolute_change"] == "0"
    assert coverage.loc["exact numeric replication coverage", "absolute_change"] == "0"


@pytest.mark.cleanup_historical
@pytest.mark.phase_snapshot
def test_group_exposure_shift_summary_comparisons_and_validation() -> None:
    built = frames()
    exposure = built["item_group_exposure"]
    shift = built["user_popularity_shift_summary"]
    comparisons = built["valid_popularity_comparisons"]
    validation = built["validation"]
    assert set(exposure["group_name"]).issuperset({"head", "tail", "cold_or_unseen"})
    assert exposure["exposure_rate"].replace("", "0").astype(float).between(0.0, 1.0).all()
    assert set(shift["profile_group"]).issubset({"low_profile_popularity", "mid_profile_popularity", "high_profile_popularity"})
    assert comparisons.empty or comparisons["is_valid_comparison"].eq("true").all()
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

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_global_extended_neural_baselines_summary.csv"
COVERAGE_PATH = PROJECT_ROOT / "results/tables/plan_d_global_extended_dataset_model_coverage.csv"
COMPARISON_PATH = PROJECT_ROOT / "results/tables/plan_d_global_neural_diagnostic_vs_extended.csv"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_global_extended_neural_baselines.md"

UTILITY_DATASETS = {"ml1m", "amazon_beauty", "amazon_sports"}
COVERAGE_DATASETS = {"ml1m", "amazon_beauty", "amazon_sports", "lastfm"}
METHODS = {"bprmf", "lightgcn", "sasrec", "gru4rec"}
EXPECTED_SUMMARY_PAIRS = {(dataset, method) for dataset in UTILITY_DATASETS for method in METHODS}
EXPECTED_COVERAGE_PAIRS = {(dataset, method) for dataset in COVERAGE_DATASETS for method in METHODS}

SUMMARY_COLUMNS = {
    "plan",
    "dataset",
    "scope",
    "method",
    "hr_at_5",
    "ndcg_at_5",
    "final_train_loss",
    "best_train_loss",
    "num_users_evaluated",
    "num_recommendations_evaluated",
    "num_train_interactions",
    "num_items",
    "num_users",
    "runtime_seconds",
    "device",
    "epochs_completed",
    "best_epoch",
    "seed",
    "config_hash",
    "metrics_path",
    "recommendations_path",
    "dataset_method_hr_rank",
    "dataset_method_ndcg_rank",
    "global_hr_rank",
    "global_ndcg_rank",
    "notes",
}

COVERAGE_COLUMNS = {
    "dataset",
    "method",
    "diagnostic_subset_available",
    "extended_subset_available",
    "full_available",
    "extended_metrics_path",
    "diagnostic_metrics_path",
    "status",
    "notes",
}

COMPARISON_COLUMNS = {
    "dataset",
    "method",
    "metric",
    "diagnostic_value",
    "extended_value",
    "absolute_delta",
    "relative_delta",
    "diagnostic_num_users_evaluated",
    "extended_num_users_evaluated",
    "diagnostic_num_train_interactions",
    "extended_num_train_interactions",
    "diagnostic_epochs_completed",
    "extended_epochs_completed",
    "notes",
}


def test_global_extended_summary_artifacts_exist() -> None:
    assert SUMMARY_PATH.is_file()
    assert COVERAGE_PATH.is_file()
    assert COMPARISON_PATH.is_file()
    assert REPORT_PATH.is_file()


def test_global_extended_summary_has_required_columns_and_pairs() -> None:
    summary = pd.read_csv(SUMMARY_PATH, dtype=str, keep_default_na=False)

    assert SUMMARY_COLUMNS.issubset(summary.columns)
    assert len(summary) == 12
    assert set(zip(summary["dataset"], summary["method"])) == EXPECTED_SUMMARY_PAIRS
    assert "lastfm" not in set(summary["dataset"])
    assert summary["plan"].eq("Plan D").all()
    assert summary["scope"].eq("extended_subset").all()


def test_global_coverage_has_required_columns_and_lastfm_rows() -> None:
    coverage = pd.read_csv(COVERAGE_PATH, dtype=str, keep_default_na=False)

    assert COVERAGE_COLUMNS.issubset(coverage.columns)
    assert len(coverage) == 16
    assert set(zip(coverage["dataset"], coverage["method"])) == EXPECTED_COVERAGE_PAIRS

    lastfm = coverage.loc[coverage["dataset"] == "lastfm"]
    assert len(lastfm) == 4
    for _, row in lastfm.iterrows():
        extended_path = PROJECT_ROOT / f"results/metrics/plan_d_lastfm_{row['method']}_extended_subset_metrics.csv"
        full_path = PROJECT_ROOT / f"results/metrics/plan_d_lastfm_{row['method']}_metrics.csv"
        if not extended_path.is_file() and not full_path.is_file():
            assert row["status"] == "extension_only_no_valid_utility_evaluation"
            assert "excluded from HR@5/NDCG@5 ranking" in row["notes"]


def test_global_coverage_marks_utility_extended_metrics_available() -> None:
    coverage = pd.read_csv(COVERAGE_PATH, dtype=str, keep_default_na=False)
    utility = coverage.loc[coverage["dataset"].isin(UTILITY_DATASETS)]

    assert len(utility) == 12
    assert utility["extended_subset_available"].eq("true").all()
    assert utility["extended_metrics_path"].ne("").all()


def test_global_diagnostic_vs_extended_has_required_columns_and_no_lastfm() -> None:
    comparison = pd.read_csv(COMPARISON_PATH, dtype=str, keep_default_na=False)

    assert COMPARISON_COLUMNS.issubset(comparison.columns)
    assert not comparison.empty
    assert "lastfm" not in set(comparison["dataset"])

    expected_pairs_with_diagnostics = {
        pair
        for pair in EXPECTED_SUMMARY_PAIRS
        if (PROJECT_ROOT / f"results/metrics/plan_d_{pair[0]}_{pair[1]}_diagnostic_subset_metrics.csv").is_file()
    }
    comparison_pairs = set(zip(comparison["dataset"], comparison["method"]))
    assert expected_pairs_with_diagnostics.issubset(comparison_pairs)
    if expected_pairs_with_diagnostics == EXPECTED_SUMMARY_PAIRS:
        assert len(comparison) == 48


def test_global_summary_metric_values_are_numeric_and_ranks_are_unique() -> None:
    summary = pd.read_csv(SUMMARY_PATH, dtype=str, keep_default_na=False)

    hr = pd.to_numeric(summary["hr_at_5"], errors="coerce")
    ndcg = pd.to_numeric(summary["ndcg_at_5"], errors="coerce")
    assert hr.notna().all()
    assert ndcg.notna().all()

    for dataset, rows in summary.groupby("dataset"):
        assert dataset in UTILITY_DATASETS
        for column in ("dataset_method_hr_rank", "dataset_method_ndcg_rank"):
            ranks = pd.to_numeric(rows[column], errors="coerce")
            assert ranks.notna().all()
            assert (ranks.astype(int) > 0).all()
            assert set(ranks.astype(int)) == {1, 2, 3, 4}
            assert ranks.astype(int).is_unique

    for column in ("global_hr_rank", "global_ndcg_rank"):
        ranks = pd.to_numeric(summary[column], errors="coerce")
        assert ranks.notna().all()
        assert (ranks.astype(int) > 0).all()
        assert set(ranks.astype(int)) == set(range(1, 13))
        assert ranks.astype(int).is_unique


def test_global_summary_values_match_source_metric_csvs_exactly() -> None:
    summary = pd.read_csv(SUMMARY_PATH, dtype=str, keep_default_na=False)

    for dataset, method in EXPECTED_SUMMARY_PAIRS:
        row = summary.loc[(summary["dataset"] == dataset) & (summary["method"] == method)].iloc[0]
        source_path = PROJECT_ROOT / f"results/metrics/plan_d_{dataset}_{method}_extended_subset_metrics.csv"
        source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
        values = dict(zip(source["metric"], source["value"]))

        assert row["hr_at_5"] == values["hit_rate"]
        assert row["ndcg_at_5"] == values["ndcg"]
        assert row["final_train_loss"] == values["final_train_loss"]
        assert row["best_train_loss"] == values["best_train_loss"]
        assert row["metrics_path"] == source_path.relative_to(PROJECT_ROOT).as_posix()


def test_global_report_mentions_lastfm_and_exclusion_rationale() -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")

    assert "LastFM" in text
    assert "excluded from HR@5/NDCG@5" in text
    assert "No fake LastFM utility metrics were generated" in text
    assert "Plan D Global Extended Neural Baseline Summary" in text

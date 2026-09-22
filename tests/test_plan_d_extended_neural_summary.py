from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_ml1m_extended_neural_baselines_summary.csv"
COMPARISON_PATH = PROJECT_ROOT / "results/tables/plan_d_ml1m_neural_diagnostic_vs_extended.csv"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_ml1m_extended_neural_baselines.md"
METHODS = {"bprmf", "lightgcn", "sasrec", "gru4rec"}

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
    "hr_rank",
    "ndcg_rank",
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


def test_extended_neural_summary_artifacts_exist() -> None:
    assert SUMMARY_PATH.is_file()
    assert COMPARISON_PATH.is_file()
    assert REPORT_PATH.is_file()


def test_extended_neural_summary_has_required_columns_and_methods() -> None:
    summary = pd.read_csv(SUMMARY_PATH, dtype=str, keep_default_na=False)

    assert SUMMARY_COLUMNS.issubset(summary.columns)
    assert set(summary["method"]) == METHODS
    assert len(summary) == 4
    assert summary["plan"].eq("Plan D").all()
    assert summary["dataset"].eq("ml1m").all()
    assert summary["scope"].eq("extended_subset").all()


def test_diagnostic_vs_extended_has_required_columns() -> None:
    comparison = pd.read_csv(COMPARISON_PATH, dtype=str, keep_default_na=False)

    assert COMPARISON_COLUMNS.issubset(comparison.columns)
    assert not comparison.empty
    assert set(comparison["method"]).issubset(METHODS)
    assert "gru4rec" in set(comparison["method"])
    assert len(comparison) == 16


def test_hr_ndcg_values_are_numeric_and_ranks_are_positive_integers() -> None:
    summary = pd.read_csv(SUMMARY_PATH, dtype=str, keep_default_na=False)

    hr = pd.to_numeric(summary["hr_at_5"], errors="coerce")
    ndcg = pd.to_numeric(summary["ndcg_at_5"], errors="coerce")
    assert hr.notna().all()
    assert ndcg.notna().all()

    for column in ("hr_rank", "ndcg_rank"):
        ranks = pd.to_numeric(summary[column], errors="coerce")
        assert ranks.notna().all()
        assert (ranks.astype(int) > 0).all()
        assert set(ranks.astype(int)) == {1, 2, 3, 4}
        assert ranks.astype(int).is_unique


def test_summary_values_match_source_metric_csvs_exactly() -> None:
    summary = pd.read_csv(SUMMARY_PATH, dtype=str, keep_default_na=False)

    for method in METHODS:
        row = summary.loc[summary["method"] == method].iloc[0]
        source_path = PROJECT_ROOT / f"results/metrics/plan_d_ml1m_{method}_extended_subset_metrics.csv"
        source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
        values = dict(zip(source["metric"], source["value"]))

        assert row["hr_at_5"] == values["hit_rate"]
        assert row["ndcg_at_5"] == values["ndcg"]
        assert row["final_train_loss"] == values["final_train_loss"]
        assert row["best_train_loss"] == values["best_train_loss"]
        assert row["metrics_path"] == source_path.relative_to(PROJECT_ROOT).as_posix()

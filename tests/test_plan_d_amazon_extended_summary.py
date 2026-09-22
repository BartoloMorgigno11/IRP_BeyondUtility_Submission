from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "results/tables/plan_d_amazon_bprmf_lightgcn_extended_summary.csv"
COMPARISON_PATH = PROJECT_ROOT / "results/tables/plan_d_amazon_bprmf_lightgcn_diagnostic_vs_extended.csv"
REPORT_PATH = PROJECT_ROOT / "report/plan_d_amazon_bprmf_lightgcn_extended_summary.md"
DATASET_METHODS = {
    ("amazon_beauty", "bprmf"),
    ("amazon_beauty", "lightgcn"),
    ("amazon_sports", "bprmf"),
    ("amazon_sports", "lightgcn"),
}

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


def test_amazon_extended_summary_artifacts_exist() -> None:
    assert SUMMARY_PATH.is_file()
    assert COMPARISON_PATH.is_file()
    assert REPORT_PATH.is_file()


def test_amazon_extended_summary_has_required_columns_and_pairs() -> None:
    summary = pd.read_csv(SUMMARY_PATH, dtype=str, keep_default_na=False)

    assert SUMMARY_COLUMNS.issubset(summary.columns)
    assert set(zip(summary["dataset"], summary["method"])) == DATASET_METHODS
    assert len(summary) == 4
    assert summary["plan"].eq("Plan D").all()
    assert summary["scope"].eq("extended_subset").all()


def test_amazon_diagnostic_vs_extended_has_required_columns_and_pairs() -> None:
    comparison = pd.read_csv(COMPARISON_PATH, dtype=str, keep_default_na=False)

    assert COMPARISON_COLUMNS.issubset(comparison.columns)
    assert not comparison.empty

    diagnostic_pairs = {
        pair
        for pair in DATASET_METHODS
        if (PROJECT_ROOT / f"results/metrics/plan_d_{pair[0]}_{pair[1]}_diagnostic_subset_metrics.csv").is_file()
    }
    comparison_pairs = set(zip(comparison["dataset"], comparison["method"]))
    assert diagnostic_pairs.issubset(comparison_pairs)
    if diagnostic_pairs == DATASET_METHODS:
        assert len(comparison) == 16


def test_amazon_hr_ndcg_values_are_numeric_and_ranks_are_per_dataset() -> None:
    summary = pd.read_csv(SUMMARY_PATH, dtype=str, keep_default_na=False)

    hr = pd.to_numeric(summary["hr_at_5"], errors="coerce")
    ndcg = pd.to_numeric(summary["ndcg_at_5"], errors="coerce")
    assert hr.notna().all()
    assert ndcg.notna().all()

    for dataset, rows in summary.groupby("dataset"):
        assert dataset in {"amazon_beauty", "amazon_sports"}
        for column in ("dataset_method_hr_rank", "dataset_method_ndcg_rank"):
            ranks = pd.to_numeric(rows[column], errors="coerce")
            assert ranks.notna().all()
            assert (ranks.astype(int) > 0).all()
            assert set(ranks.astype(int)) == {1, 2}
            assert ranks.astype(int).is_unique


def test_amazon_summary_values_match_source_metric_csvs_exactly() -> None:
    summary = pd.read_csv(SUMMARY_PATH, dtype=str, keep_default_na=False)

    for dataset, method in DATASET_METHODS:
        row = summary.loc[(summary["dataset"] == dataset) & (summary["method"] == method)].iloc[0]
        source_path = PROJECT_ROOT / f"results/metrics/plan_d_{dataset}_{method}_extended_subset_metrics.csv"
        source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
        values = dict(zip(source["metric"], source["value"]))

        assert row["hr_at_5"] == values["hit_rate"]
        assert row["ndcg_at_5"] == values["ndcg"]
        assert row["final_train_loss"] == values["final_train_loss"]
        assert row["best_train_loss"] == values["best_train_loss"]
        assert row["metrics_path"] == source_path.relative_to(PROJECT_ROOT).as_posix()


def test_amazon_report_mentions_scope_and_methods() -> None:
    text = REPORT_PATH.read_text(encoding="utf-8")

    assert "Plan D Amazon Extended BPRMF/LightGCN Summary" in text
    assert "extended_subset" in text
    assert "BPRMF" in text
    assert "LightGCN" in text

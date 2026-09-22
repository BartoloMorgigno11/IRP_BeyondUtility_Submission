from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.summarize_plan_c_final_results import (
    build_final_results_overview,
    summarize_plan_c_final_results,
)


def _write_plan_c_inputs(root: Path) -> None:
    tables = root / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "dataset": ["ml1m", "amazon_beauty", "lastfm"],
            "reconstruction_status": [
                "reconstructed_from_public_raw",
                "reconstructed_from_public_raw",
                "extension_only",
            ],
            "exact_original_artifact_recovered": ["no", "no", "no"],
            "candidate_item_universe": ["interaction_items", "interaction_items", ""],
        }
    ).to_csv(tables / "plan_c_reconstructed_dataset_summary.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["ml1m", "amazon_beauty"],
            "baseline": ["random", "content_bm25"],
            "hr_at_5": [0.1, 0.2],
            "ndcg_at_5": [0.05, 0.12],
            "catalog_coverage_at_5": [0.4, 0.3],
            "evaluated_users": [10, 5],
            "recommendation_rows": [1000, 500],
            "diagnostic_subset": [False, True],
            "max_users": [pd.NA, 5],
            "run_scope": ["full", "diagnostic_max_users_5"],
            "status": ["available", "available"],
            "reconstruction_status": [
                "reconstructed_from_public_raw",
                "reconstructed_from_public_raw",
            ],
            "notes": ["real full", "real diagnostic"],
        }
    ).to_csv(tables / "plan_c_baseline_comparison_summary.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["lastfm"],
            "baseline": ["all"],
            "status": ["blocked"],
            "reason": ["No ground truth/candidates."],
            "diagnostic_subset": [False],
            "max_users": [pd.NA],
            "run_scope": ["full"],
            "reconstruction_status": ["extension_only"],
        }
    ).to_csv(tables / "plan_c_baseline_blockers.csv", index=False)


def test_build_final_results_overview_preserves_scope_and_blockers() -> None:
    reconstruction = pd.DataFrame(
        {
            "dataset": ["ml1m", "lastfm"],
            "reconstruction_status": ["reconstructed_from_public_raw", "extension_only"],
            "exact_original_artifact_recovered": ["no", "no"],
            "candidate_item_universe": ["interaction_items", ""],
        }
    )
    comparison = pd.DataFrame(
        {
            "dataset": ["ml1m"],
            "baseline": ["random"],
            "hr_at_5": [0.1],
            "ndcg_at_5": [0.05],
            "catalog_coverage_at_5": [0.4],
            "evaluated_users": [10],
            "diagnostic_subset": [False],
            "max_users": [pd.NA],
            "run_scope": ["full"],
            "status": ["available"],
        }
    )
    blockers = pd.DataFrame(
        {
            "dataset": ["lastfm"],
            "baseline": ["all"],
            "status": ["blocked"],
            "reason": ["No ground truth/candidates."],
            "diagnostic_subset": [False],
            "run_scope": ["full"],
        }
    )

    overview = build_final_results_overview(reconstruction, comparison, blockers)

    assert set(overview["dataset"]) == {"ml1m", "lastfm"}
    ml1m = overview.loc[overview["dataset"] == "ml1m"].iloc[0]
    lastfm = overview.loc[overview["dataset"] == "lastfm"].iloc[0]
    assert ml1m["run_scope"] == "full"
    assert ml1m["candidate_item_universe"] == "interaction_items"
    assert lastfm["status"] == "blocked"
    assert pd.isna(lastfm["hr_at_5"])


def test_summarizer_writes_final_tables_and_plots(tmp_path: Path) -> None:
    _write_plan_c_inputs(tmp_path)

    outputs = summarize_plan_c_final_results(project_root=tmp_path)

    tables = tmp_path / "results/tables"
    assert (tables / "plan_c_final_results_overview.csv").is_file()
    assert (tables / "plan_c_claims_and_limitations.csv").is_file()
    assert (tables / "plan_c_final_experiment_inventory.csv").is_file()
    overview = pd.read_csv(tables / "plan_c_final_results_overview.csv")
    assert set(overview["dataset"]) == {"ml1m", "amazon_beauty", "lastfm"}
    diagnostic = overview.loc[overview["run_scope"] == "diagnostic_max_users_5"].iloc[0]
    assert bool(diagnostic["diagnostic_subset"]) is True
    assert int(diagnostic["max_users"]) == 5
    assert outputs["plot_paths"]


def test_missing_source_tables_produce_diagnostic_overview(tmp_path: Path) -> None:
    outputs = summarize_plan_c_final_results(project_root=tmp_path)

    overview = outputs["overview"]
    assert len(overview) == 1
    assert overview.loc[0, "status"] == "missing_source_tables"
    assert pd.isna(overview.loc[0, "hr_at_5"])

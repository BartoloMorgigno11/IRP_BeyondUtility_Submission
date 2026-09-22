from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.plan_d_beyond_utility import PlanDRunSpec, run_plan_d_beyond_utility


def _run_spec() -> PlanDRunSpec:
    return PlanDRunSpec(
        dataset="ml1m",
        method="random",
        scope="full",
        recommendation_filename="plan_c_ml1m_random_recommendations.csv",
    )


def _write_reconstructed_inputs(root: Path) -> None:
    base = root / "data/reconstructed/plan_c/ml1m"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "item_id": ["A", "A", "B"],
            "split": ["train", "train", "train"],
        }
    ).to_csv(base / "interactions.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "candidate_item_id": ["A", "B", "C", "A", "D"],
        }
    ).to_csv(base / "candidates.csv", index=False)
    pd.DataFrame(
        {
            "item_id": ["A", "B", "C", "D"],
            "categories": ["Action", "Drama", "Action | Comedy", "Sports"],
        }
    ).to_csv(base / "catalog.csv", index=False)


def _write_plan_c_recommendations(root: Path) -> Path:
    raw_outputs = root / "results/raw_outputs"
    raw_outputs.mkdir(parents=True, exist_ok=True)
    path = raw_outputs / "plan_c_ml1m_random_recommendations.csv"
    pd.DataFrame(
        {
            "dataset": ["ml1m", "ml1m", "ml1m", "ml1m"],
            "baseline": ["random", "random", "random", "random"],
            "user_id": ["u1", "u1", "u2", "u2"],
            "recommended_item_id": ["A", "B", "D", "A"],
            "rank": [1, 2, 1, 2],
            "in_candidate_set": [True, True, True, True],
        }
    ).to_csv(path, index=False)
    return path


def test_missing_inputs_write_status_without_empty_metrics_csv(tmp_path: Path) -> None:
    result = run_plan_d_beyond_utility(project_root=tmp_path, run_specs=[_run_spec()])

    metrics_path = tmp_path / "results/metrics/plan_d_beyond_utility_metrics.csv"
    status_path = tmp_path / "results/tables/plan_d_beyond_utility_dataset_status.csv"

    assert result["metrics"].empty
    assert result["metrics_path"] is None
    assert not metrics_path.exists()
    assert status_path.is_file()
    assert result["status"].loc[0, "status"] == "MISSING"
    assert bool(result["status"].loc[0, "metrics_written"]) is False


def test_runner_writes_plan_d_outputs_without_overwriting_plan_c_outputs(tmp_path: Path) -> None:
    _write_reconstructed_inputs(tmp_path)
    recommendation_path = _write_plan_c_recommendations(tmp_path)
    original_recommendation_text = recommendation_path.read_text(encoding="utf-8")

    plan_c_metric_path = tmp_path / "results/metrics/plan_c_ml1m_random_metrics.csv"
    plan_c_metric_path.parent.mkdir(parents=True, exist_ok=True)
    plan_c_metric_path.write_text("plan_c_metric_file\nkeep\n", encoding="utf-8")

    result = run_plan_d_beyond_utility(project_root=tmp_path, run_specs=[_run_spec()])

    metrics_path = tmp_path / "results/metrics/plan_d_beyond_utility_metrics.csv"
    comparison_path = tmp_path / "results/tables/plan_d_beyond_utility_comparison.csv"
    status_path = tmp_path / "results/tables/plan_d_beyond_utility_dataset_status.csv"

    assert metrics_path.is_file()
    assert comparison_path.is_file()
    assert status_path.is_file()
    assert result["status"].loc[0, "status"] == "AVAILABLE"
    assert result["status"].loc[0, "scope"] == "full"
    assert set(result["metrics"]["metric"]).issuperset(
        {
            "catalog_coverage_at_5",
            "novelty_at_5",
            "long_tail_rate_at_5",
            "candidate_validity_rate",
            "duplicate_rate_at_5",
        }
    )
    assert plan_c_metric_path.read_text(encoding="utf-8") == "plan_c_metric_file\nkeep\n"
    assert recommendation_path.read_text(encoding="utf-8") == original_recommendation_text

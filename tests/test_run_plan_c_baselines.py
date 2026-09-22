from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.experiments.run_plan_c_baselines import (
    build_plan_c_comparison,
    load_real_plan_c_metric_rows,
    parse_args,
    rank_content_bm25_candidates,
    rank_mostpop_candidates,
    rank_random_candidates,
    run_plan_c_baselines,
    _dataset_paths,
    _load_dataset,
)


def _write_reconstructed_dataset(root: Path, dataset: str = "ml1m") -> Path:
    base = root / "data/reconstructed/plan_c" / dataset
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["A", "P", "A", "B"],
            "split": ["train", "train", "train", "train"],
            "timestamp": [1, 2, 1, 2],
        }
    ).to_csv(base / "interactions.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "ground_truth_item_id": ["C", "D"],
            "dataset": [dataset, dataset],
            "split": ["test", "test"],
        }
    ).to_csv(base / "ground_truth.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2"],
            "candidate_item_id": ["C", "A", "B", "E", "F", "D", "A", "B", "C", "E"],
            "candidate_position": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "is_ground_truth": [True, False, False, False, False, True, False, False, False, False],
        }
    ).to_csv(base / "candidates.csv", index=False)
    pd.DataFrame(
        {
            "item_id": ["A", "B", "C", "D", "E", "F", "P"],
            "title": [
                "Space Training",
                "Romance Training",
                "Space Positive",
                "Romance Positive",
                "Quiet Drama",
                "Sparse Mystery",
                "Space Profile",
            ],
            "genres": ["Sci-Fi", "Romance", "Sci-Fi", "Romance", "Drama", "Mystery", "Sci-Fi"],
            "normalized_text": [
                "space training sci fi",
                "romance training",
                "space positive sci fi",
                "romance positive",
                "quiet drama",
                "sparse mystery",
                "space profile sci fi",
            ],
        }
    ).to_csv(base / "catalog.csv", index=False)
    return base


def _load_temp_dataset(root: Path, dataset: str = "ml1m", max_users: int | None = None):
    paths = _dataset_paths(root / "data/reconstructed/plan_c", dataset)
    return _load_dataset(
        dataset,
        paths,
        max_users=max_users,
        diagnostic_subset=max_users is not None,
    )


def test_cli_defaults_select_plan_c_datasets_and_baselines() -> None:
    args = parse_args([])
    assert args.datasets == ["amazon_beauty", "amazon_sports", "ml1m"]
    assert args.baselines == ["random", "mostpop", "content_bm25"]


def test_random_baseline_is_deterministic(tmp_path: Path) -> None:
    _write_reconstructed_dataset(tmp_path)
    dataset = _load_temp_dataset(tmp_path)

    first = rank_random_candidates(dataset, seed=42)
    second = rank_random_candidates(dataset, seed=42)
    changed = rank_random_candidates(dataset, seed=99)

    assert first[["user_id", "recommended_item_id", "rank", "score"]].equals(
        second[["user_id", "recommended_item_id", "rank", "score"]]
    )
    assert not first[["user_id", "recommended_item_id", "rank"]].equals(
        changed[["user_id", "recommended_item_id", "rank"]]
    )


def test_mostpop_uses_training_popularity_and_tie_breaks_by_item_id(tmp_path: Path) -> None:
    _write_reconstructed_dataset(tmp_path)
    dataset = _load_temp_dataset(tmp_path)

    recommendations = rank_mostpop_candidates(dataset)
    u1 = recommendations.loc[recommendations["user_id"] == "u1"].sort_values("rank")
    u2 = recommendations.loc[recommendations["user_id"] == "u2"].sort_values("rank")

    assert u1["recommended_item_id"].tolist()[:3] == ["A", "B", "C"]
    assert u2["recommended_item_id"].tolist()[:4] == ["A", "B", "C", "D"]
    assert u1["rank"].tolist() == [1, 2, 3, 4, 5]
    assert u2["rank"].tolist() == [1, 2, 3, 4, 5]


def test_content_bm25_ranks_only_candidate_items_and_is_deterministic(tmp_path: Path) -> None:
    _write_reconstructed_dataset(tmp_path)
    dataset = _load_temp_dataset(tmp_path)

    first = rank_content_bm25_candidates(dataset, timestamp="2026-01-01T00:00:00+00:00")
    second = rank_content_bm25_candidates(dataset, timestamp="2026-01-01T00:00:00+00:00")
    candidate_pairs = set(zip(dataset.candidates["user_id"], dataset.candidates["candidate_item_id"]))
    recommendation_pairs = set(zip(first["user_id"], first["recommended_item_id"]))

    assert recommendation_pairs == candidate_pairs
    assert first[["user_id", "recommended_item_id", "rank", "score"]].equals(
        second[["user_id", "recommended_item_id", "rank", "score"]]
    )
    for _, group in first.groupby("user_id"):
        assert group.sort_values("rank")["rank"].tolist() == [1, 2, 3, 4, 5]


def test_runner_writes_real_metrics_and_comparison(tmp_path: Path) -> None:
    _write_reconstructed_dataset(tmp_path)

    result = run_plan_c_baselines(
        datasets=["ml1m"],
        baselines=["mostpop"],
        project_root=tmp_path,
    )

    metric_path = tmp_path / "results/metrics/plan_c_ml1m_mostpop_metrics.csv"
    recommendation_path = tmp_path / "results/raw_outputs/plan_c_ml1m_mostpop_recommendations.csv"
    assert metric_path.is_file()
    assert recommendation_path.is_file()
    metrics = pd.read_csv(metric_path)
    assert set(metrics["metric"]) == {"hit_rate", "ndcg", "catalog_coverage"}
    assert metrics.loc[metrics["metric"] == "hit_rate", "value"].iloc[0] == pytest.approx(1.0)
    assert metrics["diagnostic_subset"].eq(False).all()
    assert metrics["max_users"].isna().all()
    assert metrics["run_scope"].eq("full").all()
    assert not result["comparison"].empty
    assert result["comparison"]["run_scope"].eq("full").all()
    diagnostics = result["candidate_diagnostics"]
    assert diagnostics.loc[0, "candidate_items_not_in_interaction_universe_count"] == 3
    assert diagnostics.loc[0, "candidate_items_not_in_interaction_universe_rate"] == pytest.approx(0.3)
    assert diagnostics.loc[0, "zero_train_pop_candidate_rate"] == pytest.approx(0.6)


def test_missing_inputs_create_blockers_without_metric_csv(tmp_path: Path) -> None:
    result = run_plan_c_baselines(
        datasets=["amazon_beauty"],
        baselines=["random"],
        project_root=tmp_path,
    )

    assert not (tmp_path / "results/metrics/plan_c_amazon_beauty_random_metrics.csv").exists()
    assert result["blockers"].loc[0, "dataset"] == "amazon_beauty"
    assert result["blockers"].loc[0, "status"] == "blocked"


def test_lastfm_is_blocked_without_ground_truth_or_candidates(tmp_path: Path) -> None:
    base = tmp_path / "data/reconstructed/plan_c/lastfm"
    base.mkdir(parents=True)
    pd.DataFrame(
        {"user_id": ["u1"], "item_id": ["a1"], "split": ["train"]}
    ).to_csv(base / "interactions.csv", index=False)
    pd.DataFrame({"item_id": ["a1"], "normalized_text": ["artist one"]}).to_csv(
        base / "catalog.csv", index=False
    )

    result = run_plan_c_baselines(
        datasets=["lastfm"],
        baselines=["random"],
        project_root=tmp_path,
    )

    assert result["blockers"].loc[0, "dataset"] == "lastfm"
    assert result["blockers"].loc[0, "reconstruction_status"] == "extension_only"
    assert result["candidate_diagnostics"].empty
    assert not (tmp_path / "results/metrics/plan_c_lastfm_random_metrics.csv").exists()


def test_max_users_marks_diagnostic_subset(tmp_path: Path) -> None:
    _write_reconstructed_dataset(tmp_path)

    result = run_plan_c_baselines(
        datasets=["ml1m"],
        baselines=["random"],
        max_users=1,
        project_root=tmp_path,
    )

    full_metric_path = tmp_path / "results/metrics/plan_c_ml1m_random_metrics.csv"
    diagnostic_metric_path = (
        tmp_path / "results/metrics/plan_c_ml1m_random_diagnostic_max_users_1_metrics.csv"
    )
    full_recommendations_path = tmp_path / "results/raw_outputs/plan_c_ml1m_random_recommendations.csv"
    diagnostic_recommendations_path = (
        tmp_path / "results/raw_outputs/plan_c_ml1m_random_diagnostic_max_users_1_recommendations.csv"
    )

    assert not full_metric_path.exists()
    assert not full_recommendations_path.exists()
    assert diagnostic_metric_path.is_file()
    assert diagnostic_recommendations_path.is_file()

    metrics = pd.read_csv(diagnostic_metric_path)
    recommendations = pd.read_csv(diagnostic_recommendations_path)
    assert metrics["diagnostic_subset"].eq(True).all()
    assert metrics["max_users"].eq(1).all()
    assert metrics["run_scope"].eq("diagnostic_max_users_1").all()
    assert recommendations["diagnostic_subset"].eq(True).all()
    assert recommendations["max_users"].eq(1).all()
    assert recommendations["run_scope"].eq("diagnostic_max_users_1").all()
    assert recommendations["user_id"].nunique() == 1
    assert bool(result["dataset_status"].loc[0, "diagnostic_subset"]) is True
    assert int(result["dataset_status"].loc[0, "max_users"]) == 1
    assert result["dataset_status"].loc[0, "run_scope"] == "diagnostic_max_users_1"
    assert result["comparison"].loc[0, "run_scope"] == "diagnostic_max_users_1"


def test_plan_b_metric_file_is_not_overwritten(tmp_path: Path) -> None:
    _write_reconstructed_dataset(tmp_path)
    plan_b_metric = tmp_path / "results/metrics/ml1m_random_metrics.csv"
    plan_b_metric.parent.mkdir(parents=True, exist_ok=True)
    plan_b_metric.write_text("plan_b_metric_file\nkeep\n", encoding="utf-8")

    run_plan_c_baselines(
        datasets=["ml1m"],
        baselines=["random"],
        project_root=tmp_path,
    )

    assert plan_b_metric.read_text(encoding="utf-8") == "plan_b_metric_file\nkeep\n"
    assert (tmp_path / "results/metrics/plan_c_ml1m_random_metrics.csv").is_file()


def test_metric_loader_uses_only_real_plan_c_rows(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "results/metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "dataset": ["ml1m"],
            "baseline": ["random"],
            "k": [5],
            "metric": ["hit_rate"],
            "value": [0.5],
            "evaluated_users": [2],
            "recommendation_rows": [10],
            "diagnostic_subset": [False],
            "max_users": [pd.NA],
            "run_scope": ["full"],
            "reconstruction_status": ["reconstructed_from_public_raw"],
            "status": ["computed"],
            "notes": ["real"],
            "timestamp": ["2026-01-01T00:00:00+00:00"],
        }
    ).to_csv(metrics_dir / "plan_c_ml1m_random_metrics.csv", index=False)
    (metrics_dir / "ml1m_random_metrics.csv").write_text("not,plan,c\n", encoding="utf-8")
    (metrics_dir / "plan_c_empty_metrics.csv").write_text("", encoding="utf-8")

    summary = load_real_plan_c_metric_rows(metrics_dir, k=5)
    comparison = build_plan_c_comparison(summary, k=5)

    assert len(summary) == 1
    assert comparison.loc[0, "dataset"] == "ml1m"
    assert comparison.loc[0, "hr_at_5"] == pytest.approx(0.5)
    assert comparison.loc[0, "run_scope"] == "full"


def test_metric_loader_preserves_full_and_diagnostic_rows(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "results/metrics"
    metrics_dir.mkdir(parents=True)
    rows = []
    for run_scope, diagnostic_subset, max_users, value in (
        ("full", False, pd.NA, 0.5),
        ("diagnostic_max_users_1", True, 1, 1.0),
    ):
        rows.append(
            {
                "dataset": "ml1m",
                "baseline": "random",
                "k": 5,
                "metric": "hit_rate",
                "value": value,
                "evaluated_users": 2 if not diagnostic_subset else 1,
                "recommendation_rows": 10 if not diagnostic_subset else 5,
                "diagnostic_subset": diagnostic_subset,
                "max_users": max_users,
                "run_scope": run_scope,
                "reconstruction_status": "reconstructed_from_public_raw",
                "status": "computed",
                "notes": "real",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        )
    pd.DataFrame([rows[0]]).to_csv(metrics_dir / "plan_c_ml1m_random_metrics.csv", index=False)
    pd.DataFrame([rows[1]]).to_csv(
        metrics_dir / "plan_c_ml1m_random_diagnostic_max_users_1_metrics.csv",
        index=False,
    )

    summary = load_real_plan_c_metric_rows(metrics_dir, k=5)
    comparison = build_plan_c_comparison(summary, k=5)

    assert set(comparison["run_scope"]) == {"full", "diagnostic_max_users_1"}
    diagnostic = comparison.loc[comparison["run_scope"] == "diagnostic_max_users_1"].iloc[0]
    assert bool(diagnostic["diagnostic_subset"]) is True
    assert int(diagnostic["max_users"]) == 1


def test_plan_c_raw_outputs_are_ignored_by_gitignore() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "results/raw_outputs/plan_c_*_recommendations.csv" in gitignore
    assert "results/raw_outputs/plan_c_*diagnostic_max_users_*_recommendations.csv" in gitignore

from __future__ import annotations

from math import log2
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.baselines.ml1m_mostpop_baseline import (
    build_baseline_comparison,
    compute_train_popularity,
    mostpop_metric_rows,
    rank_candidates_mostpop,
    run_mostpop_candidate_baseline,
)


def _write_project_markers(root: Path) -> None:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")


def _minimal_config(
    *,
    interactions_path: str | None = "data/processed/ml1m/interactions.csv",
    candidates_path: str | None = "data/processed/ml1m/candidates.csv",
    ground_truth_path: str | None = "data/processed/ml1m/ground_truth.csv",
) -> dict:
    return {
        "dataset": {
            "name": "ml1m",
            "raw_dir": "data/raw/ml1m",
            "processed_dir": "data/processed/ml1m",
            "sample_dir": "data/samples/ml1m",
            "interactions_path": interactions_path,
            "catalog_path": "data/processed/ml1m/catalog.csv",
            "candidates_path": candidates_path,
            "ground_truth_path": ground_truth_path,
            "llm_outputs_path": None,
            "baseline_outputs_path": None,
        },
        "paths": {
            "raw_dir": "data/raw/ml1m",
            "processed_dir": "data/processed/ml1m",
            "sample_dir": "data/samples/ml1m",
            "interactions_path": interactions_path,
            "catalog_path": "data/processed/ml1m/catalog.csv",
            "candidates_path": candidates_path,
            "ground_truth_path": ground_truth_path,
            "llm_outputs_path": None,
            "baseline_outputs_path": None,
        },
        "original_repo": {
            "root": "original_repo/EvaLLMasRecommender",
            "data_dir": None,
            "llm_results_dir": "original_repo/EvaLLMasRecommender/llm_results",
            "sota_results_dir": "original_repo/EvaLLMasRecommender/sota_results",
        },
        "experiment": {
            "task": "ranking",
            "k_values": [5],
            "seed": 42,
            "sample_size": None,
            "allow_api_calls": False,
        },
        "api": {
            "allow_api_calls": False,
            "provider": None,
            "model": None,
            "api_key_env_var": None,
        },
        "models": {
            "llm_outputs": [],
            "baselines": ["mostpop", "random"],
            "baseline_outputs": [],
        },
        "metrics": {
            "utility": ["hr", "ndcg"],
            "beyond_utility": ["coverage"],
            "matching_protocols": ["exact"],
        },
        "output": {
            "metrics_dir": "results/metrics",
            "tables_dir": "results/tables",
            "plots_dir": "results/plots",
            "raw_outputs_dir": "results/raw_outputs",
            "dataset_audit_path": "results/tables/ml1m_dataset_audit.csv",
            "metrics_path": "results/metrics/ml1m_reconstruction_metrics.csv",
            "summary_path": "results/tables/ml1m_reconstruction_summary.csv",
            "analysis_path": "report/results_analysis.md",
        },
    }


def _write_config(root: Path, config: dict) -> Path:
    config_path = root / "configs/ml1m.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _write_canonical_inputs(root: Path) -> None:
    processed = root / "data/processed/ml1m"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u6", "u6"],
            "item_id": ["A", "A", "B", "C", "Z", "D", "D"],
            "split": ["train", "train", "train", "train", "test", "train", "test"],
        }
    ).to_csv(processed / "interactions.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2"],
            "candidate_item_id": ["A", "B", "C", "D", "Z", "A", "B", "C", "D", "Z"],
            "candidate_position": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "is_ground_truth": [False, True, False, False, False, True, False, False, False, False],
        }
    ).to_csv(processed / "candidates.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "ground_truth_item_id": ["B", "A"],
            "dataset": ["ml1m", "ml1m"],
            "split": ["test", "test"],
        }
    ).to_csv(processed / "ground_truth.csv", index=False)


def test_popularity_uses_only_training_interactions() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "item_id": ["A", "A", "Z", "Z"],
            "split": ["train", "train", "test", "test"],
        }
    )

    popularity = compute_train_popularity(interactions)

    assert popularity.to_dict("records") == [{"item_id": "A", "score": 2}]
    assert "Z" not in popularity["item_id"].tolist()


def test_mostpop_ranks_by_popularity_and_ties_by_item_id() -> None:
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1"],
            "candidate_item_id": ["10", "2", "7", "9"],
            "candidate_position": [1, 2, 3, 4],
            "is_ground_truth": ["False", "False", "True", "False"],
        }
    )
    popularity = pd.DataFrame(
        {"item_id": ["10", "2"], "score": [3, 3]}
    )

    recommendations = rank_candidates_mostpop(
        candidates,
        popularity,
        dataset="ml1m",
        run_id="run",
        config_path="configs/ml1m.yaml",
        interactions_path="data/processed/ml1m/interactions.csv",
        candidates_path="data/processed/ml1m/candidates.csv",
        ground_truth_path="data/processed/ml1m/ground_truth.csv",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    assert recommendations.sort_values("rank")["recommended_item_id"].tolist() == [
        "2",
        "10",
        "7",
        "9",
    ]
    assert recommendations.sort_values("rank")["score"].tolist() == [3, 3, 0, 0]
    assert set(recommendations["is_positive"]) == {True, False}


def test_ranked_output_uses_only_candidates_and_consecutive_ranks(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    _write_canonical_inputs(tmp_path)
    config_path = _write_config(tmp_path, _minimal_config())

    recommendations, _, _, _, _, _, _, _, _ = run_mostpop_candidate_baseline(
        config_path,
        project_root=tmp_path,
    )
    candidates = pd.read_csv(tmp_path / "data/processed/ml1m/candidates.csv", dtype=str)
    candidate_pairs = set(zip(candidates["user_id"], candidates["candidate_item_id"]))
    recommendation_pairs = set(zip(recommendations["user_id"], recommendations["recommended_item_id"]))

    assert recommendation_pairs == candidate_pairs
    for _, group in recommendations.groupby("user_id"):
        assert group.sort_values("rank")["rank"].tolist() == list(range(1, len(group) + 1))


def test_hr_and_ndcg_are_computed_from_controlled_recommendations() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "recommended_item_id": ["A", "B", "C", "D"],
            "rank": [1, 2, 1, 2],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "ground_truth_item_id": ["A", "D"],
        }
    )

    metrics = mostpop_metric_rows(
        recommendations,
        ground_truth,
        dataset="ml1m",
        k=5,
        run_id="controlled",
        config_path="configs/ml1m.yaml",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    hit_rate = metrics.loc[metrics["metric"] == "hit_rate", "value"].iloc[0]
    ndcg = metrics.loc[metrics["metric"] == "ndcg", "value"].iloc[0]
    assert hit_rate == pytest.approx(1.0)
    assert ndcg == pytest.approx((1.0 + 1.0 / log2(3)) / 2)


def test_missing_inputs_write_summary_without_metrics_or_recommendations(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    config_path = _write_config(tmp_path, _minimal_config())

    recommendations, metrics, summary, comparison, recommendations_path, metrics_path, summary_path, comparison_path, metrics_written = (
        run_mostpop_candidate_baseline(config_path, project_root=tmp_path)
    )

    assert recommendations.empty
    assert metrics.empty
    assert metrics_written is False
    assert summary.loc[0, "status"] == "missing_input"
    assert "interactions_path" in summary.loc[0, "notes"]
    assert summary_path.is_file()
    assert comparison_path.is_file()
    assert set(comparison["status"]) == {"missing"}
    assert not recommendations_path.exists()
    assert not metrics_path.exists()


def test_comparison_uses_existing_real_metrics_without_inventing_missing_values(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    metrics_dir = tmp_path / "results/metrics"
    comparison_path = tmp_path / "results/tables/ml1m_baseline_comparison_summary.csv"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "dataset": ["ml1m"],
            "model": ["random"],
            "model_name": ["random"],
            "metric": ["hit_rate"],
            "k": [5],
            "value": [0.25],
            "num_users": [2],
            "num_recommendation_rows": [10],
            "status": ["computed"],
        }
    ).to_csv(metrics_dir / "ml1m_random_metrics.csv", index=False)

    comparison = build_baseline_comparison(
        dataset="ml1m",
        metrics_dir=metrics_dir,
        comparison_path=comparison_path,
        k=5,
        timestamp="2026-01-01T00:00:00+00:00",
        root=tmp_path,
    )

    random_hr = comparison.loc[
        (comparison["model"] == "random")
        & (comparison["metric"] == "hit_rate")
    ].iloc[0]
    random_ndcg = comparison.loc[
        (comparison["model"] == "random")
        & (comparison["metric"] == "ndcg")
    ].iloc[0]

    assert random_hr["status"] == "available"
    assert random_hr["value"] == pytest.approx(0.25)
    assert random_ndcg["status"] == "missing"
    assert pd.isna(random_ndcg["value"])
    assert comparison_path.is_file()

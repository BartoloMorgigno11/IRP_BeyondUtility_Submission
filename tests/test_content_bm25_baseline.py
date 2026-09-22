from __future__ import annotations

from collections import Counter
from math import log2
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.baselines.content_bm25_baseline import (
    bm25_score,
    build_baseline_comparison,
    build_bm25_index,
    build_user_profiles,
    content_metric_rows,
    rank_candidates_content_bm25,
    run_content_bm25_baseline,
    tokenize_text,
)


def _write_project_markers(root: Path) -> None:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")


def _minimal_config(
    *,
    interactions_path: str | None = "data/processed/ml1m/interactions.csv",
    candidates_path: str | None = "data/processed/ml1m/candidates.csv",
    ground_truth_path: str | None = "data/processed/ml1m/ground_truth.csv",
    catalog_path: str | None = "data/processed/ml1m/catalog.csv",
) -> dict:
    return {
        "dataset": {
            "name": "ml1m",
            "raw_dir": "data/raw/ml1m",
            "processed_dir": "data/processed/ml1m",
            "sample_dir": "data/samples/ml1m",
            "interactions_path": interactions_path,
            "catalog_path": catalog_path,
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
            "catalog_path": catalog_path,
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
            "user_id": ["u1", "u1", "u2", "u3"],
            "item_id": ["A", "H", "B", "Z"],
            "split": ["train", "test", "train", "test"],
        }
    ).to_csv(processed / "interactions.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "candidate_item_id": ["H", "C", "D", "A", "H", "D"],
            "candidate_position": [1, 2, 3, 1, 2, 3],
            "is_ground_truth": [True, False, False, False, True, False],
        }
    ).to_csv(processed / "candidates.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "ground_truth_item_id": ["H", "H"],
            "dataset": ["ml1m", "ml1m"],
            "split": ["test", "test"],
        }
    ).to_csv(processed / "ground_truth.csv", index=False)
    pd.DataFrame(
        {
            "item_id": ["A", "B", "C", "D", "H", "Z"],
            "title": [
                "Space Action",
                "Romantic Comedy",
                "Space Adventure",
                "Quiet Drama",
                "Hidden Space",
                "Leakage Movie",
            ],
            "genres": [
                "Action|Sci-Fi",
                "Comedy|Romance",
                "Adventure|Sci-Fi",
                "Drama",
                "Sci-Fi",
                "Noir",
            ],
            "normalized_text": [
                "space action sci fi",
                "romantic comedy romance",
                "space adventure sci fi",
                "quiet drama",
                "hidden space sci fi",
                "leakage noir",
            ],
        }
    ).to_csv(processed / "catalog.csv", index=False)


def test_tokenizer_is_deterministic() -> None:
    assert tokenize_text("Toy Story (1995) Children's|Comedy") == [
        "toy",
        "story",
        "1995",
        "children",
        "s",
        "comedy",
    ]


def test_user_profiles_use_only_training_interactions() -> None:
    catalog = pd.DataFrame(
        {
            "item_id": ["A", "Z"],
            "normalized_text": ["space action", "leakage hidden"],
        }
    )
    index = build_bm25_index(catalog)
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["A", "Z"],
            "split": ["train", "test"],
        }
    )

    profiles = build_user_profiles(interactions, index.doc_terms)

    assert profiles["u1"]["space"] == 1
    assert profiles["u1"]["action"] == 1
    assert "leakage" not in profiles["u1"]
    assert "hidden" not in profiles["u1"]


def test_bm25_scores_matching_content_above_unmatched_content() -> None:
    catalog = pd.DataFrame(
        {
            "item_id": ["A", "B"],
            "normalized_text": ["space action", "romance comedy"],
        }
    )
    index = build_bm25_index(catalog)
    profile = Counter({"space": 1, "action": 1})

    matched = bm25_score(profile, index.doc_terms["A"], index.doc_lengths["A"], index)
    unmatched = bm25_score(profile, index.doc_terms["B"], index.doc_lengths["B"], index)

    assert matched > 0
    assert unmatched == 0


def test_content_ranking_uses_candidates_consecutive_ranks_and_tie_breaks_by_item_id() -> None:
    catalog = pd.DataFrame(
        {
            "item_id": ["2", "10", "30"],
            "normalized_text": ["same token", "same token", "different"],
        }
    )
    index = build_bm25_index(catalog)
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "candidate_item_id": ["30", "10", "2"],
            "candidate_position": [1, 2, 3],
            "is_ground_truth": ["False", "False", "True"],
        }
    )
    recommendations = rank_candidates_content_bm25(
        candidates,
        profiles={"u1": Counter({"same": 1, "token": 1})},
        index=index,
        dataset="ml1m",
        run_id="run",
        config_path="configs/ml1m.yaml",
        interactions_path="data/processed/ml1m/interactions.csv",
        candidates_path="data/processed/ml1m/candidates.csv",
        ground_truth_path="data/processed/ml1m/ground_truth.csv",
        catalog_path="data/processed/ml1m/catalog.csv",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    assert recommendations.sort_values("rank")["recommended_item_id"].tolist() == [
        "2",
        "10",
        "30",
    ]
    assert recommendations.sort_values("rank")["rank"].tolist() == [1, 2, 3]
    assert set(recommendations["is_positive"]) == {True, False}


def test_content_baseline_is_deterministic(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    _write_canonical_inputs(tmp_path)
    config_path = _write_config(tmp_path, _minimal_config())

    first, _, _, _, _, _, _, _, _ = run_content_bm25_baseline(config_path, project_root=tmp_path)
    second, _, _, _, _, _, _, _, _ = run_content_bm25_baseline(config_path, project_root=tmp_path)

    assert first[["user_id", "recommended_item_id", "score", "rank"]].to_dict("records") == second[
        ["user_id", "recommended_item_id", "score", "rank"]
    ].to_dict("records")


def test_hr_and_ndcg_are_computed_from_controlled_recommendations() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "recommended_item_id": ["A", "X", "X", "B"],
            "rank": [1, 2, 1, 2],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "ground_truth_item_id": ["A", "B"],
        }
    )

    metrics = content_metric_rows(
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
        run_content_bm25_baseline(config_path, project_root=tmp_path)
    )

    assert recommendations.empty
    assert metrics.empty
    assert metrics_written is False
    assert summary.loc[0, "status"] == "missing_input"
    assert "interactions_path" in summary.loc[0, "notes"]
    assert summary_path.is_file()
    assert comparison_path.is_file()
    assert not recommendations_path.exists()
    assert not metrics_path.exists()


def test_comparison_includes_content_without_inventing_missing_values(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    metrics_dir = tmp_path / "results/metrics"
    comparison_path = tmp_path / "results/tables/ml1m_baseline_comparison_summary.csv"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "dataset": ["ml1m"],
            "model": ["content_bm25"],
            "model_name": ["content_bm25"],
            "metric": ["hit_rate"],
            "k": [5],
            "value": [0.75],
            "num_users": [2],
            "num_recommendation_rows": [10],
            "status": ["computed"],
        }
    ).to_csv(metrics_dir / "ml1m_content_bm25_metrics.csv", index=False)

    comparison = build_baseline_comparison(
        dataset="ml1m",
        metrics_dir=metrics_dir,
        comparison_path=comparison_path,
        k=5,
        timestamp="2026-01-01T00:00:00+00:00",
        root=tmp_path,
    )

    content_hr = comparison.loc[
        (comparison["model"] == "content_bm25")
        & (comparison["metric"] == "hit_rate")
    ].iloc[0]
    random_hr = comparison.loc[
        (comparison["model"] == "random")
        & (comparison["metric"] == "hit_rate")
    ].iloc[0]

    assert content_hr["status"] == "available"
    assert content_hr["value"] == pytest.approx(0.75)
    assert random_hr["status"] == "missing"
    assert pd.isna(random_hr["value"])
    assert comparison_path.is_file()

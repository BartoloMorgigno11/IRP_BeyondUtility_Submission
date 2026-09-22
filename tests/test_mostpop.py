from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.baselines.mostpop import MostPopError, compute_global_popularity, run_mostpop


def _write_project_markers(root: Path) -> None:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")


def _minimal_config(interactions_path: str | None) -> dict:
    return {
        "dataset": {
            "name": "demo",
            "raw_dir": "data/raw/demo",
            "processed_dir": "data/processed/demo",
            "sample_dir": "data/samples/demo",
            "interactions_path": interactions_path,
            "catalog_path": None,
            "candidates_path": None,
            "ground_truth_path": None,
            "llm_outputs_path": "original_repo/EvaLLMasRecommender/llm_results/demo.zip",
            "baseline_outputs_path": None,
        },
        "paths": {
            "raw_dir": "data/raw/demo",
            "processed_dir": "data/processed/demo",
            "sample_dir": "data/samples/demo",
            "interactions_path": interactions_path,
            "catalog_path": None,
            "candidates_path": None,
            "ground_truth_path": None,
            "llm_outputs_path": "original_repo/EvaLLMasRecommender/llm_results/demo.zip",
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
            "k_values": [2],
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
            "llm_outputs": [
                {
                    "model_name": "gpt-test",
                    "source_file": "original_repo/EvaLLMasRecommender/llm_results/demo.zip",
                    "archive_member": "demo/results.json",
                    "run_id": "synthetic-run",
                }
            ],
            "baselines": ["mostpop"],
            "baseline_outputs": [],
        },
        "metrics": {
            "utility": ["hr", "ndcg"],
            "beyond_utility": ["hallucination", "coverage"],
            "matching_protocols": ["exact", "normalized", "fuzzy"],
        },
        "output": {
            "metrics_dir": "results/metrics",
            "tables_dir": "results/tables",
            "plots_dir": "results/plots",
            "raw_outputs_dir": "results/raw_outputs",
            "dataset_audit_path": "results/tables/demo_dataset_audit.csv",
            "metrics_path": "results/metrics/demo_reproduction_metrics.csv",
            "summary_path": "results/tables/demo_reproduction_summary.csv",
            "analysis_path": "report/results_analysis.md",
        },
    }


def test_compute_global_popularity_uses_train_interaction_counts() -> None:
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "item_id": ["B", "A", "B", "C"],
            "split": ["train", "train", "train", "train"],
        }
    )

    popularity = compute_global_popularity(train)

    assert popularity.to_dict("records") == [
        {"item_id": "B", "train_popularity": 2},
        {"item_id": "A", "train_popularity": 1},
        {"item_id": "C", "train_popularity": 1},
    ]


def test_run_mostpop_writes_recommendations_and_summary(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    interactions_path = "data/processed/demo/interactions.csv"
    interactions_file = tmp_path / interactions_path
    interactions_file.parent.mkdir(parents=True)
    interactions_file.write_text(
        "\n".join(
            [
                "user_id,item_id,split",
                "u1,A,train",
                "u1,B,train",
                "u2,A,train",
                "u3,B,train",
                "u3,C,train",
                "u1,Z,test",
                "u2,Z,test",
                "u4,Z,test",
                "u5,Z,test",
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "configs" / "demo.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(_minimal_config(interactions_path)),
        encoding="utf-8",
    )

    recommendations, summary, rec_path, summary_path = run_mostpop(
        config_path,
        project_root=tmp_path,
    )

    assert rec_path == tmp_path / "results/raw_outputs/demo_mostpop_recommendations.csv"
    assert summary_path == tmp_path / "results/tables/demo_mostpop_summary.csv"
    assert rec_path.is_file()
    assert summary_path.is_file()
    assert recommendations[["user_id", "rank", "recommended_item_id"]].to_dict(
        "records"
    ) == [
        {"user_id": "u1", "rank": 1, "recommended_item_id": "C"},
        {"user_id": "u2", "rank": 1, "recommended_item_id": "B"},
        {"user_id": "u2", "rank": 2, "recommended_item_id": "C"},
        {"user_id": "u4", "rank": 1, "recommended_item_id": "A"},
        {"user_id": "u4", "rank": 2, "recommended_item_id": "B"},
        {"user_id": "u5", "rank": 1, "recommended_item_id": "A"},
        {"user_id": "u5", "rank": 2, "recommended_item_id": "B"},
    ]
    assert "Z" not in recommendations["recommended_item_id"].tolist()
    assert int(summary.loc[0, "train_interactions"]) == 5
    assert int(summary.loc[0, "test_interactions"]) == 4
    assert int(summary.loc[0, "test_users"]) == 4
    assert int(summary.loc[0, "unique_train_items"]) == 3
    assert int(summary.loc[0, "recommendations_written"]) == 7
    assert int(summary.loc[0, "users_with_short_recommendations"]) == 1


def test_run_mostpop_fails_clearly_when_interactions_path_is_missing(
    tmp_path: Path,
) -> None:
    _write_project_markers(tmp_path)
    config_path = tmp_path / "configs" / "demo.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(_minimal_config(None)),
        encoding="utf-8",
    )

    with pytest.raises(MostPopError, match="interactions_path is not configured"):
        run_mostpop(config_path, project_root=tmp_path)

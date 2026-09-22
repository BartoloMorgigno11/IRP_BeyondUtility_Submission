from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.experiments import run_ml1m_reconstruction as runner


def test_comparison_loader_uses_only_real_available_rows(tmp_path: Path) -> None:
    comparison_path = tmp_path / "results/tables/ml1m_baseline_comparison_summary.csv"
    comparison_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset": "ml1m",
                "model": "random",
                "metric": "hit_rate",
                "k": 5,
                "value": 0.25,
                "status": "available",
                "source_file": "results/metrics/ml1m_random_metrics.csv",
                "notes": "real",
                "timestamp": "2026-06-29T00:00:00+00:00",
            },
            {
                "dataset": "ml1m",
                "model": "mostpop",
                "metric": "hit_rate",
                "k": 5,
                "value": "",
                "status": "missing",
                "source_file": "results/metrics/ml1m_mostpop_metrics.csv",
                "notes": "missing",
                "timestamp": "2026-06-29T00:00:00+00:00",
            },
            {
                "dataset": "other",
                "model": "random",
                "metric": "hit_rate",
                "k": 5,
                "value": 0.75,
                "status": "available",
                "source_file": "other.csv",
                "notes": "wrong dataset",
                "timestamp": "2026-06-29T00:00:00+00:00",
            },
        ]
    ).to_csv(comparison_path, index=False)

    rows = runner.load_real_comparison_rows(comparison_path, dataset="ml1m", k=5)

    assert len(rows) == 1
    assert rows.loc[0, "model"] == "random"
    assert rows.loc[0, "value"] == 0.25


def test_missing_metric_files_do_not_produce_fake_comparison_values(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "results/metrics"
    comparison_path = tmp_path / "results/tables/ml1m_baseline_comparison_summary.csv"
    metrics_dir.mkdir(parents=True)

    comparison = runner.build_baseline_comparison(
        dataset="ml1m",
        metrics_dir=metrics_dir,
        comparison_path=comparison_path,
        k=5,
        timestamp="2026-06-29T00:00:00+00:00",
        root=tmp_path,
    )
    real_rows = runner.load_real_comparison_rows(comparison_path, dataset="ml1m", k=5)

    assert comparison_path.is_file()
    assert set(comparison["status"]) == {"missing"}
    assert real_rows.empty


def test_plot_functions_skip_gracefully_when_no_real_rows(tmp_path: Path) -> None:
    plot_summary, plot_paths = runner.generate_reconstruction_plots(
        pd.DataFrame(),
        plots_dir=tmp_path / "results/plots",
        dataset="ml1m",
        k=5,
        timestamp="2026-06-29T00:00:00+00:00",
        root=tmp_path,
    )

    assert plot_paths == []
    assert set(plot_summary["status"]) == {"skipped"}
    assert not list((tmp_path / "results/plots").glob("*.png"))


def test_runner_reports_missing_inputs_without_raw_and_keeps_original_repo(tmp_path: Path) -> None:
    config_path = _write_project(tmp_path)
    sentinel = tmp_path / "original_repo/EvaLLMasRecommender/sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    beauty_config = tmp_path / "configs/beauty.yaml"
    sports_config = tmp_path / "configs/sports.yaml"
    beauty_config.write_text("dataset:\n  name: beauty\n", encoding="utf-8")
    sports_config.write_text("dataset:\n  name: sports\n", encoding="utf-8")

    summary, comparison, plot_summary, summary_path, comparison_path, plot_paths = (
        runner.run_ml1m_reconstruction(config_path, project_root=tmp_path)
    )

    assert summary_path.is_file()
    assert comparison_path.is_file()
    assert comparison["status"].eq("missing").all()
    assert plot_paths == []
    assert set(plot_summary["status"]) == {"skipped"}
    assert "missing" in set(summary["status"])
    assert "missing_raw" in set(summary["status"])
    assert "blocked" in set(summary["status"])
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert beauty_config.read_text(encoding="utf-8") == "dataset:\n  name: beauty\n"
    assert sports_config.read_text(encoding="utf-8") == "dataset:\n  name: sports\n"
    assert not list((tmp_path / "results/metrics").glob("*.csv"))


def test_runner_uses_processed_inputs_without_raw_and_generates_real_outputs(tmp_path: Path) -> None:
    config_path = _write_project(tmp_path)
    _write_synthetic_canonical_inputs(tmp_path)

    summary, comparison, plot_summary, summary_path, comparison_path, plot_paths = (
        runner.run_ml1m_reconstruction(config_path, project_root=tmp_path)
    )

    preprocessing = summary.loc[summary["stage"] == "preprocessing"].iloc[0]
    assert preprocessing["status"] == "skipped"
    assert summary_path.is_file()
    assert comparison_path.is_file()
    assert len(runner.load_real_comparison_rows(comparison_path, dataset="ml1m", k=5)) == 6
    assert set(comparison["status"]) == {"available"}
    assert len(plot_paths) == 3
    assert (tmp_path / "results/plots/ml1m_baseline_hr_at_5.png").is_file()
    assert (tmp_path / "results/plots/ml1m_baseline_ndcg_at_5.png").is_file()
    assert (tmp_path / "results/plots/ml1m_baseline_comparison.png").is_file()
    assert set(plot_summary["status"]) == {"generated"}
    assert (tmp_path / "results/metrics/ml1m_random_metrics.csv").is_file()
    assert (tmp_path / "results/metrics/ml1m_mostpop_metrics.csv").is_file()
    assert (tmp_path / "results/metrics/ml1m_content_bm25_metrics.csv").is_file()


def _write_project(root: Path) -> Path:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (root / "configs").mkdir(parents=True)
    (root / "report").mkdir(parents=True)
    (root / "original_repo/EvaLLMasRecommender").mkdir(parents=True)
    config_path = root / "configs/ml1m.yaml"
    config_path.write_text(yaml.safe_dump(_minimal_config(), sort_keys=False), encoding="utf-8")
    return config_path


def _minimal_config() -> dict:
    return {
        "dataset": {
            "name": "ml1m",
            "raw_dir": "data/raw/ml1m",
            "processed_dir": "data/processed/ml1m",
            "sample_dir": "data/samples/ml1m",
            "interactions_path": "data/processed/ml1m/interactions.csv",
            "catalog_path": "data/processed/ml1m/catalog.csv",
            "candidates_path": "data/processed/ml1m/candidates.csv",
            "ground_truth_path": "data/processed/ml1m/ground_truth.csv",
            "llm_outputs_path": None,
            "baseline_outputs_path": None,
        },
        "paths": {
            "raw_dir": "data/raw/ml1m",
            "processed_dir": "data/processed/ml1m",
            "sample_dir": "data/samples/ml1m",
            "interactions_path": "data/processed/ml1m/interactions.csv",
            "catalog_path": "data/processed/ml1m/catalog.csv",
            "candidates_path": "data/processed/ml1m/candidates.csv",
            "ground_truth_path": "data/processed/ml1m/ground_truth.csv",
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
        "preprocessing": {
            "min_interactions": 2,
            "negative_samples_per_user": 2,
            "candidate_ordering": "deterministic_shuffle",
        },
        "api": {
            "allow_api_calls": False,
            "provider": None,
            "model": None,
            "api_key_env_var": None,
        },
        "models": {
            "llm_outputs": [],
            "baselines": ["mostpop", "random", "content_bm25"],
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
            "preparation_summary_path": "results/tables/ml1m_dataset_preparation_summary.csv",
            "metrics_path": "results/metrics/ml1m_reconstruction_metrics.csv",
            "summary_path": "results/tables/ml1m_reconstruction_summary.csv",
            "analysis_path": "report/results_analysis.md",
        },
    }


def _write_synthetic_canonical_inputs(root: Path) -> None:
    processed = root / "data/processed/ml1m"
    processed.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["A", "B", "A"],
            "split": ["train", "train", "train"],
            "timestamp": [1, 2, 1],
            "rating": [5, 4, 5],
        }
    ).to_csv(processed / "interactions.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "ground_truth_item_id": ["C", "B"],
            "dataset": ["ml1m", "ml1m"],
            "split": ["test", "test"],
        }
    ).to_csv(processed / "ground_truth.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "candidate_item_id": ["C", "D", "E", "B", "D", "E"],
            "candidate_position": [1, 2, 3, 1, 2, 3],
            "is_ground_truth": [True, False, False, True, False, False],
            "candidate_item_title": ["Space Hit", "Drama", "Comedy", "Action Hit", "Drama", "Comedy"],
        }
    ).to_csv(processed / "candidates.csv", index=False)
    pd.DataFrame(
        {
            "item_id": ["A", "B", "C", "D", "E"],
            "title": ["Space Action", "Action Sequel", "Space Hit", "Quiet Drama", "Light Comedy"],
            "genres": ["Action|Sci-Fi", "Action", "Sci-Fi", "Drama", "Comedy"],
            "normalized_text": [
                "space action sci fi",
                "action sequel",
                "space hit sci fi",
                "quiet drama",
                "light comedy",
            ],
        }
    ).to_csv(processed / "catalog.csv", index=False)

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.experiments import make_plots


def test_plot_is_generated_when_valid_metric_csv_exists(tmp_path: Path) -> None:
    config_path = _write_project(tmp_path)
    metrics_dir = tmp_path / "results/metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset": "demo",
                "model_name": "model-a",
                "metric": "hit_rate",
                "k": 5,
                "value": 0.75,
                "run_id": "run-a",
                "metric_context": "utility:test",
            }
        ]
    ).to_csv(metrics_dir / "demo_reproduction_metrics.csv", index=False)

    summary, summary_path, _ = make_plots.run_make_plots(config_path, project_root=tmp_path)

    assert summary_path.is_file()
    assert (tmp_path / "results/plots/demo_hr_at_5.png").is_file()
    generated = summary.loc[summary["plot_name"] == "hr_at_5"].iloc[0]
    assert generated["status"] == "generated"
    assert generated["output_path"] == "results\\plots\\demo_hr_at_5.png" or generated["output_path"] == "results/plots/demo_hr_at_5.png"


def test_plot_is_skipped_when_metric_is_missing(tmp_path: Path) -> None:
    config_path = _write_project(tmp_path)
    (tmp_path / "results/metrics").mkdir(parents=True)

    summary, _, _ = make_plots.run_make_plots(config_path, project_root=tmp_path)

    hr_row = summary.loc[summary["plot_name"] == "hr_at_5"].iloc[0]
    assert hr_row["status"] == "skipped"
    assert not (tmp_path / "results/plots/demo_hr_at_5.png").exists()


def test_no_crash_with_empty_metrics_directory(tmp_path: Path) -> None:
    config_path = _write_project(tmp_path)
    (tmp_path / "results/metrics").mkdir(parents=True)

    summary, summary_path, analysis_path = make_plots.run_make_plots(
        config_path,
        project_root=tmp_path,
    )

    assert summary_path.is_file()
    assert analysis_path.is_file()
    assert set(summary["status"]) == {"skipped"}


def test_deterministic_output_file_names(tmp_path: Path) -> None:
    config_path = _write_project(tmp_path)
    metrics_dir = tmp_path / "results/metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset": "demo",
                "model_name": "model-a",
                "metric": "ndcg",
                "k": 5,
                "value": 0.5,
            }
        ]
    ).to_csv(metrics_dir / "demo_reproduction_metrics.csv", index=False)

    summary, _, _ = make_plots.run_make_plots(config_path, project_root=tmp_path)

    ndcg_plot = tmp_path / "results/plots/demo_ndcg_at_5.png"
    assert ndcg_plot.is_file()
    ndcg_row = summary.loc[summary["plot_name"] == "ndcg_at_5"].iloc[0]
    assert ndcg_row["output_path"].endswith("demo_ndcg_at_5.png")


def test_readiness_summary_does_not_create_fake_metric_plot(tmp_path: Path) -> None:
    config_path = _write_project(tmp_path)
    tables_dir = tmp_path / "results/tables"
    tables_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset": "demo", "phase": "reproduction", "step": "ground_truth", "status": "blocked"},
            {"dataset": "demo", "phase": "coverage", "step": "catalog", "status": "not_configured"},
            {"dataset": "demo", "phase": "llm_outputs", "step": "llm", "status": "ready"},
        ]
    ).to_csv(tables_dir / "demo_final_summary.csv", index=False)

    summary, _, _ = make_plots.run_make_plots(config_path, project_root=tmp_path)

    assert not (tmp_path / "results/plots/demo_hr_at_5.png").exists()
    assert not (tmp_path / "results/plots/demo_ndcg_at_5.png").exists()
    assert (tmp_path / "results/plots/demo_readiness_status_counts.png").is_file()
    readiness = summary.loc[summary["plot_name"] == "readiness_status_counts"].iloc[0]
    assert readiness["status"] == "generated"
    generated_metric_plots = summary.loc[
        (summary["status"] == "generated")
        & (summary["plot_name"] != "readiness_status_counts")
    ]
    assert generated_metric_plots.empty


def _write_project(root: Path) -> Path:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (root / "configs").mkdir(parents=True)
    (root / "report").mkdir(parents=True)
    (root / "original_repo/EvaLLMasRecommender").mkdir(parents=True)
    config_path = root / "configs/demo.yaml"
    config_path.write_text(yaml.safe_dump(_minimal_config()), encoding="utf-8")
    return config_path


def _minimal_config() -> dict:
    return {
        "dataset": {
            "name": "demo",
            "raw_dir": "data/raw/demo",
            "processed_dir": "data/processed/demo",
            "sample_dir": "data/samples/demo",
            "interactions_path": None,
            "catalog_path": None,
            "candidates_path": None,
            "ground_truth_path": None,
            "llm_outputs_path": None,
            "baseline_outputs_path": None,
        },
        "paths": {
            "raw_dir": "data/raw/demo",
            "processed_dir": "data/processed/demo",
            "sample_dir": "data/samples/demo",
            "interactions_path": None,
            "catalog_path": None,
            "candidates_path": None,
            "ground_truth_path": None,
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

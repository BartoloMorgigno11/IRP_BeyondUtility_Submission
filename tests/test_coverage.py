from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.evaluation.coverage import catalog_coverage_at_k, coverage_by_model
from src.experiments.run_extension_coverage import run_coverage_extension


def test_catalog_coverage_is_one_when_all_catalog_items_are_recommended() -> None:
    catalog = pd.DataFrame({"item_id": ["A", "B", "C"]})
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "recommended_item_id": ["A", "B", "C"],
            "rank": [1, 2, 1],
        }
    )

    assert catalog_coverage_at_k(recommendations, catalog, k=2) == 1.0


def test_catalog_coverage_is_zero_when_no_recommendations_are_present() -> None:
    catalog = pd.DataFrame({"item_id": ["A", "B"]})

    assert catalog_coverage_at_k(pd.DataFrame(), catalog, k=5) == 0.0


def test_catalog_coverage_partial_catalog() -> None:
    catalog = pd.DataFrame({"item_id": ["A", "B", "C", "D"]})
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "recommended_item_id": ["A", "C"],
            "rank": [1, 1],
        }
    )

    assert catalog_coverage_at_k(recommendations, catalog, k=5) == pytest.approx(0.5)


def test_top_k_cutoff_changes_catalog_coverage() -> None:
    catalog = pd.DataFrame({"item_id": ["A", "B"]})
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "recommended_item_id": ["A", "B"],
            "rank": [1, 2],
        }
    )

    assert catalog_coverage_at_k(recommendations, catalog, k=1) == pytest.approx(0.5)
    assert catalog_coverage_at_k(recommendations, catalog, k=2) == pytest.approx(1.0)


def test_duplicate_recommendations_do_not_inflate_coverage() -> None:
    catalog = pd.DataFrame({"item_id": ["A", "B", "C"]})
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "recommended_item_id": ["A", "A", "A"],
            "rank": [1, 2, 1],
        }
    )

    assert catalog_coverage_at_k(recommendations, catalog, k=5) == pytest.approx(1 / 3)


def test_multiple_users_are_aggregated_for_coverage() -> None:
    catalog = pd.DataFrame({"item_id": ["A", "B", "C", "D"]})
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u2"],
            "recommended_item_id": ["A", "B", "C"],
            "rank": [1, 1, 2],
        }
    )

    assert catalog_coverage_at_k(recommendations, catalog, k=2) == pytest.approx(0.75)


def test_empty_catalog_returns_zero_by_documented_policy() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1"],
            "recommended_item_id": ["A"],
            "rank": [1],
        }
    )

    assert catalog_coverage_at_k(recommendations, pd.DataFrame({"item_id": []}), k=5) == 0.0


def test_catalog_coverage_supports_string_titles_when_requested() -> None:
    catalog = pd.DataFrame({"item_title": ["Alpha", "Beta", "Gamma"]})
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "recommended_item_text": ["Alpha", "Gamma"],
            "rank": [1, 1],
        }
    )

    assert catalog_coverage_at_k(
        recommendations,
        catalog,
        k=5,
        recommendation_col="recommended_item_text",
        catalog_col="item_title",
    ) == pytest.approx(2 / 3)


def test_coverage_by_model_aggregates_per_model_and_run() -> None:
    catalog = pd.DataFrame({"item_id": ["A", "B", "C"]})
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "model_name": ["m1", "m2", "m2"],
            "run_id": ["r1", "r2", "r2"],
            "recommended_item_id": ["A", "B", "C"],
            "rank": [1, 1, 1],
        }
    )

    result = coverage_by_model(recommendations, catalog, k=5)

    assert result[["model_name", "run_id", "value"]].to_dict("records") == [
        {"model_name": "m1", "run_id": "r1", "value": pytest.approx(1 / 3)},
        {"model_name": "m2", "run_id": "r2", "value": pytest.approx(2 / 3)},
    ]


def test_coverage_runner_writes_metrics_only_with_valid_catalog(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    catalog_path = "data/processed/demo/catalog.csv"
    catalog_file = tmp_path / catalog_path
    catalog_file.parent.mkdir(parents=True)
    catalog_file.write_text("item_id,item_title\nA,Alpha\nB,Beta\nC,Gamma\nD,Delta\n", encoding="utf-8")

    raw_outputs = tmp_path / "results/raw_outputs"
    raw_outputs.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "model_name": ["gpt-test", "gpt-test", "gpt-test"],
            "dataset": ["demo", "demo", "demo"],
            "rank": [1, 2, 1],
            "recommended_item_id": ["A", "B", "C"],
            "run_id": ["coverage-run", "coverage-run", "coverage-run"],
            "source_file": ["llm.csv", "llm.csv", "llm.csv"],
        }
    ).to_csv(raw_outputs / "demo_llm_outputs_normalized.csv", index=False)

    config_path = _write_config(
        tmp_path,
        _minimal_config(catalog_path=catalog_path),
    )

    metrics, summary, metrics_path, summary_path, analysis_path, blocked = (
        run_coverage_extension(config_path, project_root=tmp_path)
    )

    assert not blocked
    assert metrics_path.is_file()
    assert summary_path.is_file()
    assert analysis_path.is_file()
    assert len(metrics) == 1
    assert metrics.loc[0, "metric"] == "catalog_coverage"
    assert metrics.loc[0, "value"] == pytest.approx(0.75)
    assert metrics.loc[0, "catalog_size"] == 4
    assert metrics.loc[0, "unique_recommended_items"] == 3
    assert "ready" in set(summary["status"])


def test_coverage_runner_writes_diagnostic_when_catalog_path_is_missing(
    tmp_path: Path,
) -> None:
    _write_project_markers(tmp_path)
    raw_outputs = tmp_path / "results/raw_outputs"
    raw_outputs.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_id": ["u1"],
            "model_name": ["gpt-test"],
            "dataset": ["demo"],
            "rank": [1],
            "recommended_item_id": ["A"],
            "run_id": ["coverage-run"],
            "source_file": ["llm.csv"],
        }
    ).to_csv(raw_outputs / "demo_llm_outputs_normalized.csv", index=False)
    config_path = _write_config(tmp_path, _minimal_config(catalog_path=None))

    metrics, summary, metrics_path, summary_path, analysis_path, blocked = (
        run_coverage_extension(config_path, project_root=tmp_path)
    )

    assert blocked
    assert metrics.empty
    assert not metrics_path.exists()
    assert summary_path.is_file()
    assert analysis_path.is_file()
    assert summary.loc[summary["check"] == "catalog_path", "status"].iloc[0] == "not_configured"


def _write_project_markers(root: Path) -> None:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")


def _write_config(root: Path, config: dict) -> Path:
    config_path = root / "configs/demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _minimal_config(*, catalog_path: str | None) -> dict:
    return {
        "dataset": {
            "name": "demo",
            "raw_dir": "data/raw/demo",
            "processed_dir": "data/processed/demo",
            "sample_dir": "data/samples/demo",
            "interactions_path": None,
            "catalog_path": catalog_path,
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
            "catalog_path": catalog_path,
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
            "llm_outputs": [
                {
                    "model_name": "gpt-test",
                    "source_file": "original_repo/EvaLLMasRecommender/llm_results/demo.zip",
                    "archive_member": "demo/results.json",
                    "run_id": "coverage-run",
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

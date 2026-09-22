from __future__ import annotations

from math import log2
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.experiments.run_reproduction import main, run_reproduction


def _write_project_markers(root: Path) -> None:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")


def _minimal_config(
    *,
    ground_truth_path: str | None,
    catalog_path: str | None = None,
    candidates_path: str | None = None,
) -> dict:
    return {
        "dataset": {
            "name": "demo",
            "raw_dir": "data/raw/demo",
            "processed_dir": "data/processed/demo",
            "sample_dir": "data/samples/demo",
            "interactions_path": None,
            "catalog_path": catalog_path,
            "candidates_path": candidates_path,
            "ground_truth_path": ground_truth_path,
            "llm_outputs_path": "original_repo/EvaLLMasRecommender/llm_results/demo.zip",
            "baseline_outputs_path": None,
        },
        "paths": {
            "raw_dir": "data/raw/demo",
            "processed_dir": "data/processed/demo",
            "sample_dir": "data/samples/demo",
            "interactions_path": None,
            "catalog_path": catalog_path,
            "candidates_path": candidates_path,
            "ground_truth_path": ground_truth_path,
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
                    "run_id": "synthetic-llm-run",
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


def _write_config(root: Path, config: dict) -> Path:
    config_path = root / "configs" / "demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_run_reproduction_computes_utility_and_llm_hallucination(
    tmp_path: Path,
) -> None:
    _write_project_markers(tmp_path)
    ground_truth_path = "data/processed/demo/ground_truth.csv"
    catalog_path = "data/processed/demo/catalog.csv"

    ground_truth_file = tmp_path / ground_truth_path
    ground_truth_file.parent.mkdir(parents=True)
    ground_truth_file.write_text(
        "user_id,ground_truth_item_id\nu1,A\nu2,B\n",
        encoding="utf-8",
    )

    catalog_file = tmp_path / catalog_path
    catalog_file.write_text(
        "item_id,item_title\nA,Alpha\nB,Beta\n",
        encoding="utf-8",
    )

    raw_outputs = tmp_path / "results" / "raw_outputs"
    raw_outputs.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "model_name": ["gpt-test"] * 4,
            "dataset": ["demo"] * 4,
            "rank": [1, 2, 1, 2],
            "recommended_item_id": ["A", "X", "X", "B"],
            "recommended_item_text": ["Alpha", "Unknown", "Unknown", "Beta"],
            "run_id": ["synthetic-llm-run"] * 4,
            "source_file": ["llm.csv"] * 4,
            "match_type": ["item_id", "unmatched", "unmatched", "item_id"],
        }
    ).to_csv(raw_outputs / "demo_llm_outputs_normalized.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "model_name": ["BPRMF", "BPRMF"],
            "dataset": ["demo", "demo"],
            "rank": [1, 1],
            "recommended_item_id": ["A", "B"],
            "run_id": ["synthetic-baseline-run", "synthetic-baseline-run"],
            "source_file": ["baseline.csv", "baseline.csv"],
        }
    ).to_csv(raw_outputs / "demo_baseline_outputs_normalized.csv", index=False)

    config_path = _write_config(
        tmp_path,
        _minimal_config(
            ground_truth_path=ground_truth_path,
            catalog_path=catalog_path,
        ),
    )

    metrics, summary, metrics_path, summary_path, analysis_path, blocked = (
        run_reproduction(config_path, project_root=tmp_path)
    )

    assert not blocked
    assert metrics_path == tmp_path / "results/metrics/demo_reproduction_metrics.csv"
    assert summary_path == tmp_path / "results/tables/demo_reproduction_summary.csv"
    assert analysis_path == tmp_path / "report/results_analysis.md"
    assert metrics_path.is_file()
    assert summary_path.is_file()
    assert analysis_path.is_file()
    assert {
        "run_id",
        "dataset",
        "task",
        "model_name",
        "metric",
        "k",
        "value",
        "config_path",
        "timestamp",
    }.issubset(metrics.columns)

    llm_hr = _metric_value(metrics, "gpt-test", "hit_rate", "utility:llm")
    llm_ndcg = _metric_value(metrics, "gpt-test", "ndcg", "utility:llm")
    baseline_ndcg = _metric_value(metrics, "BPRMF", "ndcg", "utility:traditional_baseline")
    hallucination = _metric_value(
        metrics,
        "gpt-test",
        "hallucination_rate",
        "catalog_hallucination",
    )

    assert llm_hr == pytest.approx(1.0)
    assert llm_ndcg == pytest.approx((1 + (1 / log2(3))) / 2)
    assert baseline_ndcg == pytest.approx(1.0)
    assert hallucination == pytest.approx(0.5)
    assert "ready" in set(summary["status"])


def test_run_reproduction_writes_diagnostics_when_inputs_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project_markers(tmp_path)
    config_path = _write_config(
        tmp_path,
        _minimal_config(ground_truth_path=None),
    )

    raw_outputs = tmp_path / "results" / "raw_outputs"
    raw_outputs.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_id": ["u1"],
            "model_name": ["gpt-test"],
            "dataset": ["demo"],
            "rank": [1],
            "recommended_item_id": [pd.NA],
            "recommended_item_text": ["Unknown Item"],
            "run_id": ["synthetic-llm-run"],
            "source_file": ["llm.csv"],
        }
    ).to_csv(raw_outputs / "demo_llm_outputs_normalized.csv", index=False)

    metrics_path = tmp_path / "results/metrics/demo_reproduction_metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "run_id,dataset,task,model_name,metric,k,value,config_path,timestamp,source_file,metric_context\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config_path.relative_to(tmp_path))]) == 0

    summary_path = tmp_path / "results/tables/demo_reproduction_summary.csv"
    analysis_path = tmp_path / "report/results_analysis.md"
    summary = pd.read_csv(summary_path)

    metrics, _, _, _, _, blocked = run_reproduction(config_path, project_root=tmp_path)

    assert blocked
    assert metrics.empty
    assert not metrics_path.exists()
    assert summary_path.is_file()
    assert analysis_path.is_file()
    assert "blocked" in set(summary["status"])
    assert "ground_truth_path" in set(summary["check"])
    llm = summary.loc[summary["check"] == "llm_recommendations"].iloc[0]
    assert llm["status"] == "ready"
    assert int(llm["row_count"]) == 1
    assert "hallucination_reference" in set(summary["check"])
    hallucination = summary.loc[summary["check"] == "hallucination_reference"].iloc[0]
    assert hallucination["status"] == "not_configured"
    assert "No metric values were computed." in analysis_path.read_text(encoding="utf-8")
    assert "Metrics CSV: `not written`" in analysis_path.read_text(encoding="utf-8")


def test_run_reproduction_skips_empty_recommendation_rows_without_blocking_file(
    tmp_path: Path,
) -> None:
    _write_project_markers(tmp_path)
    config_path = _write_config(
        tmp_path,
        _minimal_config(ground_truth_path=None),
    )

    raw_outputs = tmp_path / "results" / "raw_outputs"
    raw_outputs.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "model_name": ["gpt-test", "gpt-test"],
            "dataset": ["demo", "demo"],
            "rank": [1, 1],
            "recommended_item_id": [pd.NA, pd.NA],
            "recommended_item_text": ["Text-only Recommendation", pd.NA],
            "run_id": ["synthetic-llm-run", "synthetic-llm-run"],
            "source_file": ["llm.csv", "llm.csv"],
        }
    ).to_csv(raw_outputs / "demo_llm_outputs_normalized.csv", index=False)

    metrics, summary, metrics_path, _, _, blocked = run_reproduction(
        config_path,
        project_root=tmp_path,
    )

    llm = summary.loc[summary["check"] == "llm_recommendations"].iloc[0]
    assert blocked
    assert metrics.empty
    assert not metrics_path.exists()
    assert llm["status"] == "ready_with_warnings"
    assert int(llm["row_count"]) == 1
    assert "Skipped 1 row(s)" in llm["message"]
    assert "recommendation_inputs" not in set(summary["check"])


def test_run_reproduction_blocks_completely_empty_recommendation_file(
    tmp_path: Path,
) -> None:
    _write_project_markers(tmp_path)
    config_path = _write_config(
        tmp_path,
        _minimal_config(ground_truth_path=None),
    )

    raw_outputs = tmp_path / "results" / "raw_outputs"
    raw_outputs.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_id": ["u1"],
            "model_name": ["gpt-test"],
            "dataset": ["demo"],
            "rank": [1],
            "recommended_item_id": [pd.NA],
            "recommended_item_text": [pd.NA],
            "run_id": ["synthetic-llm-run"],
            "source_file": ["llm.csv"],
        }
    ).to_csv(raw_outputs / "demo_llm_outputs_normalized.csv", index=False)

    metrics, summary, metrics_path, _, _, blocked = run_reproduction(
        config_path,
        project_root=tmp_path,
    )

    llm = summary.loc[summary["check"] == "llm_recommendations"].iloc[0]
    assert blocked
    assert metrics.empty
    assert not metrics_path.exists()
    assert llm["status"] == "blocked"
    assert "no rows with a non-empty" in llm["message"]
    assert "recommendation_inputs" in set(summary["check"])


def _metric_value(
    metrics: pd.DataFrame,
    model_name: str,
    metric_name: str,
    metric_context: str,
) -> float:
    matches = metrics.loc[
        (metrics["model_name"] == model_name)
        & (metrics["metric"] == metric_name)
        & (metrics["metric_context"] == metric_context),
        "value",
    ]
    assert len(matches) == 1
    return float(matches.iloc[0])

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.experiments import run_pipeline as pipeline
from src.utils.config import ConfigError


def test_pipeline_writes_final_summary_and_records_blocked_steps(
    tmp_path: Path,
) -> None:
    config_path = _write_project(tmp_path)
    sentinel = tmp_path / "original_repo/EvaLLMasRecommender/sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    metrics, summary, metrics_path, summary_path, analysis_path, blocked = pipeline.run_pipeline(
        config_path,
        project_root=tmp_path,
    )

    assert blocked
    assert metrics.empty
    assert summary_path.is_file()
    assert analysis_path.is_file()
    assert not metrics_path.exists()
    assert {"blocked", "not_configured"}.intersection(set(summary["status"]))
    history_row = summary.loc[summary["phase"] == "history_length"].iloc[0]
    assert str(history_row["path"]).endswith("demo_history_length_summary.csv")
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_pipeline_main_exits_successfully_with_diagnostic_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    exit_code = pipeline.main(["--config", "configs/demo.yaml"])

    assert exit_code == 0
    assert (tmp_path / "results/tables/demo_final_summary.csv").is_file()
    assert not (tmp_path / "results/metrics/demo_final_results.csv").exists()


def test_no_final_metrics_csv_is_written_when_no_real_metrics_exist(
    tmp_path: Path,
) -> None:
    config_path = _write_project(tmp_path)

    metrics, _, metrics_path, _, _, _ = pipeline.run_pipeline(
        config_path,
        project_root=tmp_path,
    )

    assert metrics.empty
    assert not metrics_path.exists()


def test_final_metrics_csv_is_written_only_when_real_metric_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_project(tmp_path)

    def fake_run_reproduction(config_path: Path, *, project_root: Path | None = None):
        metrics = pd.DataFrame(
            [
                {
                    "run_id": "synthetic-run",
                    "dataset": "demo",
                    "task": "ranking",
                    "model_name": "demo-model",
                    "metric": "synthetic_metric",
                    "k": 5,
                    "value": 0.5,
                    "config_path": str(config_path),
                    "timestamp": "2026-06-27T00:00:00+00:00",
                    "source_file": "synthetic.csv",
                    "metric_context": "test",
                }
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "dataset": "demo",
                    "status": "ready",
                    "check": "synthetic_ground_truth",
                    "path": "synthetic.csv",
                    "message": "Synthetic metric was supplied by the test.",
                    "row_count": 1,
                    "timestamp": "2026-06-27T00:00:00+00:00",
                }
            ]
        )
        root = Path(project_root) if project_root is not None else tmp_path
        return (
            metrics,
            summary,
            root / "results/metrics/demo_reproduction_metrics.csv",
            root / "results/tables/demo_reproduction_summary.csv",
            root / "report/results_analysis.md",
            False,
        )

    monkeypatch.setattr(pipeline, "run_reproduction", fake_run_reproduction)

    metrics, summary, metrics_path, summary_path, _, blocked = pipeline.run_pipeline(
        config_path,
        project_root=tmp_path,
    )

    written = pd.read_csv(metrics_path)
    assert not blocked
    assert not metrics.empty
    assert metrics_path.is_file()
    assert summary_path.is_file()
    assert written.loc[0, "metric"] == "synthetic_metric"
    assert written.loc[0, "final_source_phase"] == "reproduction"
    assert "completed" in set(summary["status"])


def test_optional_missing_extension_runners_are_handled_gracefully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_project(tmp_path)
    monkeypatch.setattr(pipeline, "run_coverage_extension", None)
    monkeypatch.setattr(pipeline, "run_position_bias", None)
    monkeypatch.setattr(pipeline, "run_history_length", None)

    _, summary, _, _, _, _ = pipeline.run_pipeline(config_path, project_root=tmp_path)

    skipped = summary.loc[summary["status"] == "skipped", "phase"].tolist()
    assert "coverage" in skipped
    assert "position_bias" in skipped
    assert "history_length" in skipped


def test_api_calls_are_disabled_by_default_and_rejected_when_enabled(
    tmp_path: Path,
) -> None:
    config_path = _write_project(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["api"]["allow_api_calls"] is False
    assert config["experiment"]["allow_api_calls"] is False

    config["api"]["allow_api_calls"] = True
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError, match="api.allow_api_calls"):
        pipeline.run_pipeline(config_path, project_root=tmp_path)


def test_invalid_configs_produce_clear_errors(tmp_path: Path) -> None:
    config_path = _write_project(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.pop("metrics")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError, match="Missing required top-level section"):
        pipeline.run_pipeline(config_path, project_root=tmp_path)


def _write_project(root: Path) -> Path:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (root / "report").mkdir(parents=True)
    (root / "report/codebase_map.md").write_text("# map\n", encoding="utf-8")
    (root / "original_repo/EvaLLMasRecommender").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
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

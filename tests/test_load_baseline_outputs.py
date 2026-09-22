from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.baselines.load_baseline_outputs import load_baseline_outputs


def _write_project_markers(root: Path) -> None:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")


def _minimal_config(baseline_outputs_path: str | None) -> dict:
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
            "llm_outputs_path": "original_repo/EvaLLMasRecommender/llm_results/demo.zip",
            "baseline_outputs_path": baseline_outputs_path,
        },
        "paths": {
            "raw_dir": "data/raw/demo",
            "processed_dir": "data/processed/demo",
            "sample_dir": "data/samples/demo",
            "interactions_path": None,
            "catalog_path": None,
            "candidates_path": None,
            "ground_truth_path": None,
            "llm_outputs_path": "original_repo/EvaLLMasRecommender/llm_results/demo.zip",
            "baseline_outputs_path": baseline_outputs_path,
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


def _write_config(root: Path, config: dict) -> Path:
    config_path = root / "configs" / "demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_loads_synthetic_baseline_csv_without_interactions_path(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    baseline_path = "data/baselines/demo/rec-SYN-history10-seed0.csv"
    baseline_file = tmp_path / baseline_path
    baseline_file.parent.mkdir(parents=True)
    baseline_file.write_text(
        "user_id\trec_items\nu1\t['C', 'A', 'B']\nu2\t['B', 'A']\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, _minimal_config(baseline_path))

    normalized, summary, normalized_path, summary_path = load_baseline_outputs(
        config_path,
        project_root=tmp_path,
    )

    assert normalized_path == tmp_path / "results/raw_outputs/demo_baseline_outputs_normalized.csv"
    assert summary_path == tmp_path / "results/tables/demo_baseline_outputs_summary.csv"
    assert normalized_path.is_file()
    assert summary_path.is_file()
    assert len(normalized) == 5
    found = summary[summary["status"] == "found"]
    assert len(found) == 1
    assert int(found.iloc[0]["recommendations_normalized"]) == 5


def test_preserves_recommendation_order_from_rec_items(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    baseline_path = "data/baselines/demo/rec-ORDER-history10-seed0.csv"
    baseline_file = tmp_path / baseline_path
    baseline_file.parent.mkdir(parents=True)
    baseline_file.write_text(
        "user_id\trec_items\nu1\t['item-3', 'item-1', 'item-2']\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, _minimal_config(baseline_path))

    normalized, _, _, _ = load_baseline_outputs(config_path, project_root=tmp_path)

    assert normalized[["rank", "recommended_item_id"]].to_dict("records") == [
        {"rank": 1, "recommended_item_id": "item-3"},
        {"rank": 2, "recommended_item_id": "item-1"},
        {"rank": 3, "recommended_item_id": "item-2"},
    ]


def test_writes_diagnostic_summary_when_no_files_are_configured_or_found(
    tmp_path: Path,
) -> None:
    _write_project_markers(tmp_path)
    config_path = _write_config(tmp_path, _minimal_config(None))

    normalized, summary, normalized_path, summary_path = load_baseline_outputs(
        config_path,
        project_root=tmp_path,
    )

    assert normalized.empty
    assert not normalized_path.exists()
    assert summary_path.is_file()
    assert {"not_configured", "missing"}.issubset(set(summary["status"]))
    assert "original_repo_search" in summary["source_kind"].tolist()


def test_loader_does_not_depend_on_real_original_repo(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    baseline_path = "data/baselines/demo/rec-NOREPO-history10-seed0.csv"
    baseline_file = tmp_path / baseline_path
    baseline_file.parent.mkdir(parents=True)
    baseline_file.write_text("user_id,rec_items\nu1,\"[1, 2]\"\n", encoding="utf-8")
    config_path = _write_config(tmp_path, _minimal_config(baseline_path))

    normalized, summary, _, _ = load_baseline_outputs(config_path, project_root=tmp_path)

    assert not (tmp_path / "original_repo").exists()
    assert normalized["recommended_item_id"].tolist() == [1, 2]
    assert "found" in summary["status"].tolist()


def test_loader_does_not_require_interactions_path(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    baseline_path = "data/baselines/demo/rec-NOINTERACTIONS-history10-seed0.csv"
    baseline_file = tmp_path / baseline_path
    baseline_file.parent.mkdir(parents=True)
    baseline_file.write_text("user_id,rec_items\nu1,\"['A']\"\n", encoding="utf-8")
    config = _minimal_config(baseline_path)
    config["paths"]["interactions_path"] = None
    config["dataset"]["interactions_path"] = None
    config_path = _write_config(tmp_path, config)

    normalized, _, _, _ = load_baseline_outputs(config_path, project_root=tmp_path)

    assert normalized[["user_id", "rank", "recommended_item_id"]].to_dict(
        "records"
    ) == [{"user_id": "u1", "rank": 1, "recommended_item_id": "A"}]

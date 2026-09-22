from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from src.experiments.load_llm_outputs import load_llm_outputs


def _write_project_markers(root: Path) -> None:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")


def _minimal_config(
    *,
    source_file: str,
    archive_member: str,
    catalog_path: str | None = None,
) -> dict:
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
            "llm_outputs_path": source_file,
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
            "llm_outputs_path": source_file,
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
                    "source_file": source_file,
                    "archive_member": archive_member,
                    "prompt_variant": "history10_candidate20",
                    "candidate_ordering": "shuffle",
                    "history_length": 10,
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


def test_load_llm_outputs_parses_configured_zip_and_writes_outputs(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    source_file = "original_repo/EvaLLMasRecommender/llm_results/demo.zip"
    archive_member = "demo/results.json"
    zip_path = tmp_path / source_file
    zip_path.parent.mkdir(parents=True)
    catalog_path = "data/processed/demo/catalog.csv"
    catalog_file = tmp_path / catalog_path
    catalog_file.parent.mkdir(parents=True)
    catalog_file.write_text(
        "item_id,item_title\nA1,Alpha Cream\nB2,Beta Brush\nC3,Gamma Mask\n",
        encoding="utf-8",
    )
    records = [
        {
            "user": "u1",
            "target": "Alpha Cream",
            "cand_list": "['A1', 'B2']",
            "template": 2,
            "result": "1. Alpha Cream\n2. Unknown Product",
        },
        {
            "user": "u2",
            "target": "Gamma Mask",
            "cand_list": "['B2', 'C3']",
            "template": 3,
            "result": "['Beta Brush', 'Gamma Mask']",
        },
    ]
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(archive_member, json.dumps(records))

    config_path = tmp_path / "configs" / "demo.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            _minimal_config(
                source_file=source_file,
                archive_member=archive_member,
                catalog_path=catalog_path,
            )
        ),
        encoding="utf-8",
    )

    normalized, summary, normalized_path, summary_path = load_llm_outputs(
        config_path,
        project_root=tmp_path,
        log_unmatched_limit=0,
    )

    assert normalized_path == tmp_path / "results/raw_outputs/demo_llm_outputs_normalized.csv"
    assert summary_path == tmp_path / "results/tables/demo_llm_parsing_summary.csv"
    assert normalized_path.is_file()
    assert summary_path.is_file()
    assert len(normalized) == 4
    assert normalized["recommended_item_text"].tolist() == [
        "Alpha Cream",
        "Unknown Product",
        "Beta Brush",
        "Gamma Mask",
    ]
    assert normalized["match_type"].tolist() == [
        "exact",
        "unmatched",
        "exact",
        "exact",
    ]
    assert summary.loc[0, "status"] == "parsed"
    assert int(summary.loc[0, "records_total"]) == 2
    assert int(summary.loc[0, "parsed_recommendations"]) == 4
    assert int(summary.loc[0, "exact_matches"]) == 3
    assert int(summary.loc[0, "unmatched"]) == 1

    saved = pd.read_csv(normalized_path)
    assert len(saved) == 4


def test_load_llm_outputs_writes_diagnostic_when_source_is_missing(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    source_file = "original_repo/EvaLLMasRecommender/llm_results/missing.zip"
    archive_member = "demo/results.json"
    config_path = tmp_path / "configs" / "demo.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(_minimal_config(source_file=source_file, archive_member=archive_member)),
        encoding="utf-8",
    )

    normalized, summary, normalized_path, summary_path = load_llm_outputs(
        config_path,
        project_root=tmp_path,
        log_unmatched_limit=0,
    )

    assert normalized.empty
    assert normalized_path.is_file()
    assert summary_path.is_file()
    assert summary.loc[0, "status"] == "source_file_missing"
    assert "does not exist" in summary.loc[0, "message"]

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    ConfigError,
    find_project_root,
    load_yaml_config,
    resolve_configured_paths,
    resolve_project_path,
    validate_config,
)


def _minimal_config(dataset_name: str = "demo") -> dict:
    return {
        "dataset": {
            "name": dataset_name,
            "raw_dir": f"data/raw/{dataset_name}",
            "processed_dir": f"data/processed/{dataset_name}",
            "sample_dir": f"data/samples/{dataset_name}",
            "interactions_path": None,
            "catalog_path": None,
            "candidates_path": None,
            "ground_truth_path": None,
            "llm_outputs_path": f"original_repo/EvaLLMasRecommender/llm_results/{dataset_name}.zip",
            "baseline_outputs_path": None,
        },
        "paths": {
            "raw_dir": f"data/raw/{dataset_name}",
            "processed_dir": f"data/processed/{dataset_name}",
            "sample_dir": f"data/samples/{dataset_name}",
            "interactions_path": None,
            "catalog_path": None,
            "candidates_path": None,
            "ground_truth_path": None,
            "llm_outputs_path": f"original_repo/EvaLLMasRecommender/llm_results/{dataset_name}.zip",
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
                    "model_name": "gpt3.5",
                    "source_file": f"original_repo/EvaLLMasRecommender/llm_results/{dataset_name}.zip",
                    "archive_member": f"{dataset_name}/results.json",
                    "run_id": f"{dataset_name}_run",
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
            "dataset_audit_path": f"results/tables/{dataset_name}_dataset_audit.csv",
            "metrics_path": f"results/metrics/{dataset_name}_reproduction_metrics.csv",
            "summary_path": f"results/tables/{dataset_name}_reproduction_summary.csv",
            "analysis_path": "report/results_analysis.md",
        },
    }


def test_find_project_root_from_nested_directory(tmp_path: Path) -> None:
    (tmp_path / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    nested = tmp_path / "src" / "utils"
    nested.mkdir(parents=True)

    assert find_project_root(nested) == tmp_path


def test_load_yaml_config_from_project_relative_path(tmp_path: Path) -> None:
    (tmp_path / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    config_path = tmp_path / "configs" / "demo.yaml"
    config_path.write_text(yaml.safe_dump(_minimal_config()), encoding="utf-8")

    config = load_yaml_config("configs/demo.yaml", project_root=tmp_path)

    assert config["dataset"]["name"] == "demo"


def test_repository_configs_validate() -> None:
    root = find_project_root()

    beauty = load_yaml_config("configs/beauty.yaml", project_root=root)
    sports = load_yaml_config("configs/sports.yaml", project_root=root)
    ml1m = load_yaml_config("configs/ml1m.yaml", project_root=root)

    assert beauty["dataset"]["name"] == "beauty"
    assert sports["dataset"]["name"] == "sports"
    assert ml1m["dataset"]["name"] == "ml1m"


def test_validate_config_reports_missing_top_level_section() -> None:
    config = _minimal_config()
    config.pop("metrics")

    with pytest.raises(ConfigError, match="Missing required top-level section"):
        validate_config(config)


def test_validate_config_rejects_api_enabled_by_default() -> None:
    config = _minimal_config()
    config["api"]["allow_api_calls"] = True

    with pytest.raises(ConfigError, match="api.allow_api_calls must remain false"):
        validate_config(config)


def test_validate_config_accepts_optional_top_level_seed_and_outputs() -> None:
    config = _minimal_config()
    config["seed"] = 42
    config["baselines"] = ["mostpop"]
    config["llm_outputs"] = config["models"]["llm_outputs"]

    validate_config(config)


def test_validate_config_rejects_absolute_configured_path(tmp_path: Path) -> None:
    config = _minimal_config()
    config["paths"]["raw_dir"] = str(tmp_path / "data")

    with pytest.raises(ConfigError, match="must not be an absolute path"):
        validate_config(config)


def test_resolve_configured_paths_keeps_null_and_resolves_relative_paths(tmp_path: Path) -> None:
    config = _minimal_config()

    resolved = resolve_configured_paths(config, project_root=tmp_path)

    assert resolved["paths.raw_dir"] == tmp_path / "data" / "raw" / "demo"
    assert resolved["paths.interactions_path"] is None
    assert resolved["models.llm_outputs[0].source_file"] == (
        tmp_path / "original_repo" / "EvaLLMasRecommender" / "llm_results" / "demo.zip"
    )


def test_resolve_project_path_rejects_escaping_project_root(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="escapes the project root"):
        resolve_project_path("../outside", tmp_path)

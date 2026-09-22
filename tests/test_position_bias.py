from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.evaluation.position_bias import (
    position_bias_summary,
    selected_candidate_position_distribution,
)
from src.experiments.run_position_bias import run_position_bias
from src.prompts.candidate_ordering import apply_candidate_ordering


def test_reversed_candidate_ordering_works() -> None:
    ordered = apply_candidate_ordering(_candidate_frame(), "reversed")

    assert ordered["candidate_item_id"].tolist() == ["C", "B", "A"]
    assert ordered["candidate_position"].tolist() == [1, 2, 3]
    assert ordered["original_candidate_position"].tolist() == [3, 2, 1]
    assert ordered["candidate_item_title"].tolist() == ["Gamma", "Beta", "Alpha"]


def test_random_candidate_ordering_is_deterministic_with_seed() -> None:
    first = apply_candidate_ordering(_candidate_frame(), "random", seed=123)
    second = apply_candidate_ordering(_candidate_frame(), "random", seed=123)

    assert first["candidate_item_id"].tolist() == second["candidate_item_id"].tolist()
    assert first["candidate_position"].tolist() == [1, 2, 3]
    assert second["candidate_position"].tolist() == [1, 2, 3]


def test_positive_at_start_places_known_positive_first() -> None:
    ordered = apply_candidate_ordering(_candidate_frame(), "positive_at_start")

    assert ordered["candidate_item_id"].tolist()[0] == "B"
    assert ordered.loc[ordered["candidate_item_id"] == "B", "candidate_position"].iloc[0] == 1


def test_positive_at_end_places_known_positive_last() -> None:
    ordered = apply_candidate_ordering(_candidate_frame(), "positive_at_end")

    assert ordered["candidate_item_id"].tolist()[-1] == "B"
    assert ordered.loc[ordered["candidate_item_id"] == "B", "candidate_position"].iloc[0] == 3


def test_selected_candidate_position_distribution_is_computed_correctly() -> None:
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "candidate_item_id": ["A", "B", "C", "D"],
            "candidate_item_title": ["Alpha", "Beta", "Gamma", "Delta"],
            "candidate_position": [1, 2, 1, 2],
        }
    )
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "recommended_item_id": ["B", "A", "D"],
            "rank": [1, 2, 1],
        }
    )

    result = selected_candidate_position_distribution(recommendations, candidates, k=1)

    assert result["position_counts"] == {2: 2}
    assert result["position_distribution"] == {2: pytest.approx(1.0)}
    assert result["matched_recommendations"] == 2
    assert result["missing_recommendations"] == 0


def test_duplicate_recommendations_do_not_inflate_position_counts() -> None:
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_item_id": ["A", "B"],
            "candidate_position": [1, 2],
        }
    )
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "recommended_item_id": ["B", "B"],
            "rank": [1, 2],
        }
    )

    result = selected_candidate_position_distribution(recommendations, candidates, k=5)

    assert result["position_counts"] == {2: 1}
    assert result["matched_recommendations"] == 1
    assert result["duplicate_recommendations"] == 1


def test_missing_candidates_are_reported_explicitly() -> None:
    candidates = pd.DataFrame(
        {
            "user_id": ["u1"],
            "candidate_item_id": ["A"],
            "candidate_position": [1],
        }
    )
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1"],
            "recommended_item_id": ["missing-item"],
            "rank": [1],
        }
    )

    result = selected_candidate_position_distribution(recommendations, candidates, k=5)
    summary = position_bias_summary(
        recommendations.assign(model_name="demo-model", run_id="demo-run"),
        candidates,
        k=5,
    )

    assert result["position_counts"] == {}
    assert result["matched_recommendations"] == 0
    assert result["missing_recommendations"] == 1
    assert pd.isna(summary.loc[0, "candidate_position"])
    assert summary.loc[0, "missing_recommendations"] == 1


def test_position_bias_summary_groups_by_model_and_run() -> None:
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "candidate_item_id": ["A", "B", "C", "D"],
            "candidate_position": [1, 2, 1, 2],
        }
    )
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "model_name": ["m1", "m2"],
            "run_id": ["r1", "r2"],
            "recommended_item_id": ["A", "D"],
            "rank": [1, 1],
        }
    )

    summary = position_bias_summary(recommendations, candidates, k=5)

    assert summary[["model_name", "run_id", "candidate_position", "fraction"]].to_dict("records") == [
        {"model_name": "m1", "run_id": "r1", "candidate_position": 1, "fraction": pytest.approx(1.0)},
        {"model_name": "m2", "run_id": "r2", "candidate_position": 2, "fraction": pytest.approx(1.0)},
    ]


def test_position_bias_runner_writes_diagnostic_when_inputs_unavailable(
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
            "recommended_item_text": ["Alpha"],
            "run_id": ["position-run"],
            "source_file": ["llm.csv"],
        }
    ).to_csv(raw_outputs / "demo_llm_outputs_normalized.csv", index=False)
    config_path = _write_config(tmp_path, _minimal_config(candidates_path=None))

    metrics, summary, metrics_path, summary_path, analysis_path, blocked = (
        run_position_bias(config_path, project_root=tmp_path)
    )

    assert blocked
    assert metrics.empty
    assert not metrics_path.exists()
    assert summary_path.is_file()
    assert analysis_path.is_file()
    assert summary.loc[summary["check"] == "candidates_path", "status"].iloc[0] == "not_configured"


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "candidate_item_id": ["A", "B", "C"],
            "candidate_item_title": ["Alpha", "Beta", "Gamma"],
            "candidate_position": [1, 2, 3],
            "is_ground_truth": [False, True, False],
        }
    )


def _write_project_markers(root: Path) -> None:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")


def _write_config(root: Path, config: dict) -> Path:
    config_path = root / "configs/demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _minimal_config(*, candidates_path: str | None) -> dict:
    return {
        "dataset": {
            "name": "demo",
            "raw_dir": "data/raw/demo",
            "processed_dir": "data/processed/demo",
            "sample_dir": "data/samples/demo",
            "interactions_path": None,
            "catalog_path": None,
            "candidates_path": candidates_path,
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
            "candidates_path": candidates_path,
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
                    "run_id": "position-run",
                }
            ],
            "baselines": ["mostpop"],
            "baseline_outputs": [],
        },
        "metrics": {
            "utility": ["hr", "ndcg"],
            "beyond_utility": ["hallucination", "coverage", "position_bias"],
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

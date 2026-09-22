from __future__ import annotations

from math import log2
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.baselines.random_baseline import (
    random_metric_rows,
    rank_candidates_random,
    run_random_baseline,
)


def _write_project_markers(root: Path) -> None:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")


def _minimal_config(*, seed: int = 42, candidates_path: str | None = None, ground_truth_path: str | None = None) -> dict:
    candidates_path = candidates_path or "data/processed/ml1m/candidates.csv"
    ground_truth_path = ground_truth_path or "data/processed/ml1m/ground_truth.csv"
    return {
        "dataset": {
            "name": "ml1m",
            "raw_dir": "data/raw/ml1m",
            "processed_dir": "data/processed/ml1m",
            "sample_dir": "data/samples/ml1m",
            "interactions_path": "data/processed/ml1m/interactions.csv",
            "catalog_path": "data/processed/ml1m/catalog.csv",
            "candidates_path": candidates_path,
            "ground_truth_path": ground_truth_path,
            "llm_outputs_path": None,
            "baseline_outputs_path": None,
        },
        "paths": {
            "raw_dir": "data/raw/ml1m",
            "processed_dir": "data/processed/ml1m",
            "sample_dir": "data/samples/ml1m",
            "interactions_path": "data/processed/ml1m/interactions.csv",
            "catalog_path": "data/processed/ml1m/catalog.csv",
            "candidates_path": candidates_path,
            "ground_truth_path": ground_truth_path,
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
            "seed": seed,
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
            "baselines": ["mostpop", "random"],
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
            "metrics_path": "results/metrics/ml1m_reconstruction_metrics.csv",
            "summary_path": "results/tables/ml1m_reconstruction_summary.csv",
            "analysis_path": "report/results_analysis.md",
        },
    }


def _write_config(root: Path, config: dict) -> Path:
    config_path = root / "configs/ml1m.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _write_canonical_inputs(root: Path) -> None:
    processed = root / "data/processed/ml1m"
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2", "u2"],
            "candidate_item_id": ["A", "B", "C", "D", "E", "F", "A", "B", "C", "D", "E", "F"],
            "candidate_position": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "is_ground_truth": [True, False, False, False, False, False, False, True, False, False, False, False],
            "candidate_item_title": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"] * 2,
        }
    ).to_csv(processed / "candidates.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "ground_truth_item_id": ["A", "B"],
            "dataset": ["ml1m", "ml1m"],
            "split": ["test", "test"],
        }
    ).to_csv(processed / "ground_truth.csv", index=False)


def _rank_order(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        user_id: group.sort_values("rank")["recommended_item_id"].tolist()
        for user_id, group in frame.groupby("user_id", sort=False)
    }


def test_random_ranking_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    _write_canonical_inputs(tmp_path)
    config_path = _write_config(tmp_path, _minimal_config(seed=7))

    first, _, _, _, _, _, _ = run_random_baseline(config_path, project_root=tmp_path)
    second, _, _, _, _, _, _ = run_random_baseline(config_path, project_root=tmp_path)

    assert _rank_order(first) == _rank_order(second)
    assert first["score"].tolist() == second["score"].tolist()


def test_changing_seed_can_change_ranking(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    _write_canonical_inputs(tmp_path)
    config_path = _write_config(tmp_path, _minimal_config(seed=7))
    first, _, _, _, _, _, _ = run_random_baseline(config_path, project_root=tmp_path)

    config_path.write_text(yaml.safe_dump(_minimal_config(seed=99), sort_keys=False), encoding="utf-8")
    second, _, _, _, _, _, _ = run_random_baseline(config_path, project_root=tmp_path)

    assert _rank_order(first) != _rank_order(second)


def test_ranked_output_uses_only_candidates_and_consecutive_ranks(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    _write_canonical_inputs(tmp_path)
    config_path = _write_config(tmp_path, _minimal_config(seed=7))

    recommendations, _, _, _, _, _, _ = run_random_baseline(config_path, project_root=tmp_path)
    candidates = pd.read_csv(tmp_path / "data/processed/ml1m/candidates.csv", dtype=str)
    candidate_pairs = set(zip(candidates["user_id"], candidates["candidate_item_id"]))
    recommendation_pairs = set(zip(recommendations["user_id"], recommendations["recommended_item_id"]))

    assert recommendation_pairs == candidate_pairs
    for _, group in recommendations.groupby("user_id"):
        assert group.sort_values("rank")["rank"].tolist() == list(range(1, len(group) + 1))


def test_hr_and_ndcg_are_computed_from_controlled_recommendations() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "recommended_item_id": ["A", "X", "X", "B"],
            "rank": [1, 2, 1, 2],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "ground_truth_item_id": ["A", "B"],
        }
    )

    metrics = random_metric_rows(
        recommendations,
        ground_truth,
        dataset="ml1m",
        k=5,
        run_id="controlled",
        config_path="configs/ml1m.yaml",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    hit_rate = metrics.loc[metrics["metric"] == "hit_rate", "value"].iloc[0]
    ndcg = metrics.loc[metrics["metric"] == "ndcg", "value"].iloc[0]
    assert hit_rate == pytest.approx(1.0)
    assert ndcg == pytest.approx((1.0 + 1.0 / log2(3)) / 2)


def test_missing_inputs_write_summary_without_metrics_or_recommendations(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    config_path = _write_config(tmp_path, _minimal_config())

    recommendations, metrics, summary, recommendations_path, metrics_path, summary_path, metrics_written = (
        run_random_baseline(config_path, project_root=tmp_path)
    )

    assert recommendations.empty
    assert metrics.empty
    assert metrics_written is False
    assert summary.loc[0, "status"] == "missing_input"
    assert "candidates_path" in summary.loc[0, "notes"]
    assert summary_path.is_file()
    assert not recommendations_path.exists()
    assert not metrics_path.exists()


def test_rank_candidates_random_preserves_candidate_position_and_positive_flag() -> None:
    candidates = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "candidate_item_id": ["A", "B"],
            "candidate_position": [1, 2],
            "is_ground_truth": ["True", "False"],
        }
    )

    recommendations = rank_candidates_random(
        candidates,
        dataset="ml1m",
        seed=1,
        run_id="run",
        config_path="configs/ml1m.yaml",
        candidates_path="data/processed/ml1m/candidates.csv",
        ground_truth_path="data/processed/ml1m/ground_truth.csv",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    assert set(recommendations["candidate_position"]) == {1, 2}
    assert set(recommendations["is_positive"]) == {True, False}

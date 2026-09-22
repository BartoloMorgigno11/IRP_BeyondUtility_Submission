from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.prepare_ml1m import prepare_ml1m


def _write_project_markers(root: Path) -> None:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")


def _minimal_config(negative_samples_per_user: int = 2) -> dict:
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
            "seed": 123,
            "sample_size": None,
            "allow_api_calls": False,
        },
        "preprocessing": {
            "min_interactions": 2,
            "negative_samples_per_user": negative_samples_per_user,
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
            "preparation_summary_path": "results/tables/ml1m_dataset_preparation_summary.csv",
            "metrics_path": "results/metrics/ml1m_reconstruction_metrics.csv",
            "summary_path": "results/tables/ml1m_reconstruction_summary.csv",
            "analysis_path": "report/results_analysis.md",
        },
    }


def _write_config(root: Path, config: dict | None = None) -> Path:
    config_path = root / "configs" / "ml1m.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config or _minimal_config(), sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _write_raw_ml1m(root: Path) -> None:
    raw_dir = root / "data" / "raw" / "ml1m"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "ratings.dat").write_text(
        "\n".join(
            [
                "1::10::5::100",
                "1::20::4::200",
                "1::30::3::150",
                "2::20::5::50",
                "2::40::4::75",
                "3::50::2::99",
                "4::60::4::10",
                "4::70::5::20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "movies.dat").write_text(
        "\n".join(
            [
                "10::Movie 10 (2000)::Drama",
                "20::Movie 20 (2000)::Comedy",
                "30::Movie 30 (2000)::Action",
                "40::Movie 40 (2000)::Drama",
                "50::Movie 50 (2000)::Comedy",
                "60::Movie 60 (2000)::Drama",
                "70::Movie 70 (2000)::Action",
                "80::Movie 80 (2000)::Comedy",
                "90::Movie 90 (2000)::Drama",
                "100::Movie 100 (2000)::Action",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _prepared_project(tmp_path: Path) -> tuple[Path, Path]:
    _write_project_markers(tmp_path)
    config_path = _write_config(tmp_path)
    _write_raw_ml1m(tmp_path)
    return tmp_path, config_path


def test_leave_one_out_split_is_chronological(tmp_path: Path) -> None:
    root, config_path = _prepared_project(tmp_path)

    result = prepare_ml1m(config_path, project_root=root)

    assert result.processed is True
    interactions = pd.read_csv(root / "data/processed/ml1m/interactions.csv", dtype=str)
    ground_truth = pd.read_csv(root / "data/processed/ml1m/ground_truth.csv", dtype=str)

    assert ground_truth.set_index("user_id")["ground_truth_item_id"].to_dict() == {
        "1": "20",
        "2": "40",
        "4": "70",
    }
    assert interactions.loc[interactions["user_id"] == "1", "item_id"].tolist() == [
        "10",
        "30",
    ]
    assert set(interactions["split"]) == {"train"}


def test_users_with_fewer_than_two_interactions_are_excluded(tmp_path: Path) -> None:
    root, config_path = _prepared_project(tmp_path)

    prepare_ml1m(config_path, project_root=root)

    interactions = pd.read_csv(root / "data/processed/ml1m/interactions.csv", dtype=str)
    ground_truth = pd.read_csv(root / "data/processed/ml1m/ground_truth.csv", dtype=str)
    candidates = pd.read_csv(root / "data/processed/ml1m/candidates.csv", dtype=str)

    assert "3" not in set(interactions["user_id"])
    assert "3" not in set(ground_truth["user_id"])
    assert "3" not in set(candidates["user_id"])


def test_candidates_have_one_positive_and_unseen_negatives(tmp_path: Path) -> None:
    root, config_path = _prepared_project(tmp_path)

    prepare_ml1m(config_path, project_root=root)

    candidates = pd.read_csv(root / "data/processed/ml1m/candidates.csv")
    full_history = {
        "1": {"10", "20", "30"},
        "2": {"20", "40"},
        "4": {"60", "70"},
    }

    per_user_positive_counts = candidates.groupby("user_id")["is_ground_truth"].sum()
    assert per_user_positive_counts.to_dict() == {1: 1, 2: 1, 4: 1}
    assert candidates.groupby("user_id").size().to_dict() == {1: 3, 2: 3, 4: 3}

    for _, row in candidates.iterrows():
        user_id = str(row["user_id"])
        item_id = str(row["candidate_item_id"])
        if bool(row["is_ground_truth"]):
            assert item_id in full_history[user_id]
        else:
            assert item_id not in full_history[user_id]


def test_candidate_sampling_is_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    root, config_path = _prepared_project(tmp_path)

    prepare_ml1m(config_path, project_root=root)
    first = (root / "data/processed/ml1m/candidates.csv").read_text(encoding="utf-8")
    prepare_ml1m(config_path, project_root=root)
    second = (root / "data/processed/ml1m/candidates.csv").read_text(encoding="utf-8")

    assert first == second


def test_missing_required_raw_files_writes_only_diagnostic_summary(tmp_path: Path) -> None:
    _write_project_markers(tmp_path)
    config_path = _write_config(tmp_path)
    raw_dir = tmp_path / "data/raw/ml1m"
    raw_dir.mkdir(parents=True)
    (raw_dir / "ratings.dat").write_text("1::10::5::100\n", encoding="utf-8")

    result = prepare_ml1m(config_path, project_root=tmp_path)

    assert result.processed is False
    assert result.summary_path.is_file()
    assert result.summary.loc[0, "status"] == "missing_raw"
    assert "movies.dat" in result.summary.loc[0, "missing_files"]
    assert not (tmp_path / "data/processed/ml1m/interactions.csv").exists()
    assert not (tmp_path / "data/processed/ml1m/ground_truth.csv").exists()
    assert not (tmp_path / "data/processed/ml1m/candidates.csv").exists()
    assert not (tmp_path / "data/processed/ml1m/catalog.csv").exists()

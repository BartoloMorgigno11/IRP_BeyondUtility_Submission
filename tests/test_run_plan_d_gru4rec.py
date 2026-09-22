from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.experiments.run_plan_d_gru4rec as runner
from src.baselines.plan_d_gru4rec import GRU4RecConfig, output_stem, run_plan_d_gru4rec


def test_output_stems_are_plan_d_prefixed() -> None:
    assert output_stem("ml1m", "full") == "plan_d_ml1m_gru4rec"
    assert output_stem("ml1m", "diagnostic_subset") == "plan_d_ml1m_gru4rec_diagnostic_subset"
    assert output_stem("ml1m", "extended_subset") == "plan_d_ml1m_gru4rec_extended_subset"


def test_cli_accepts_all_scopes_and_subset_caps() -> None:
    diagnostic_args = runner.parse_args(
        [
            "--dataset",
            "ml1m",
            "--scope",
            "diagnostic_subset",
            "--max-users",
            "100",
            "--max-train-interactions",
            "5000",
        ]
    )
    extended_args = runner.parse_args(
        [
            "--dataset",
            "ml1m",
            "--scope",
            "extended_subset",
            "--max-users",
            "1000",
            "--max-train-interactions",
            "100000",
        ]
    )
    full_args = runner.parse_args(["--dataset", "ml1m", "--scope", "full"])

    assert diagnostic_args.scope == "diagnostic_subset"
    assert diagnostic_args.max_users == 100
    assert diagnostic_args.max_train_interactions == 5000
    assert extended_args.scope == "extended_subset"
    assert extended_args.max_users == 1000
    assert extended_args.max_train_interactions == 100000
    assert full_args.scope == "full"


def test_cli_diagnostic_mode_runs_on_tiny_reconstructed_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_tiny_plan_c_dataset(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    exit_code = runner.main(
        [
            "--dataset",
            "ml1m",
            "--epochs",
            "1",
            "--max-users",
            "2",
            "--max-train-interactions",
            "4",
            "--scope",
            "diagnostic_subset",
            "--skip-plots",
            "--device",
            "cpu",
            "--embedding-dim",
            "4",
            "--hidden-dim",
            "4",
            "--num-layers",
            "1",
            "--max-seq-len",
            "3",
        ]
    )

    metrics_path = tmp_path / "results/metrics/plan_d_ml1m_gru4rec_diagnostic_subset_metrics.csv"
    recommendations_path = tmp_path / "results/raw_outputs/plan_d_ml1m_gru4rec_diagnostic_subset_recommendations.csv"
    summary_path = tmp_path / "results/tables/plan_d_gru4rec_training_summary.csv"

    assert exit_code == 0
    assert metrics_path.is_file()
    assert recommendations_path.is_file()
    assert summary_path.is_file()
    metrics = pd.read_csv(metrics_path)
    assert {"hit_rate", "ndcg", "final_train_loss", "best_train_loss"}.issubset(set(metrics["metric"]))
    assert metrics["method"].eq("gru4rec").all()
    assert metrics["scope"].eq("diagnostic_subset").all()
    assert {"embedding_dim", "hidden_dim", "num_layers", "dropout", "max_seq_len"}.issubset(metrics.columns)
    recommendations = pd.read_csv(recommendations_path)
    assert recommendations["method"].eq("gru4rec").all()
    assert recommendations["scope"].eq("diagnostic_subset").all()


def test_cli_extended_subset_uses_separate_outputs_on_tiny_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_tiny_plan_c_dataset(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    exit_code = runner.main(
        [
            "--dataset",
            "ml1m",
            "--epochs",
            "1",
            "--max-users",
            "2",
            "--max-train-interactions",
            "4",
            "--scope",
            "extended_subset",
            "--skip-plots",
            "--device",
            "cpu",
            "--embedding-dim",
            "4",
            "--hidden-dim",
            "4",
            "--num-layers",
            "1",
            "--max-seq-len",
            "3",
        ]
    )

    metrics_path = tmp_path / "results/metrics/plan_d_ml1m_gru4rec_extended_subset_metrics.csv"
    recommendations_path = tmp_path / "results/raw_outputs/plan_d_ml1m_gru4rec_extended_subset_recommendations.csv"
    diagnostic_metrics_path = tmp_path / "results/metrics/plan_d_ml1m_gru4rec_diagnostic_subset_metrics.csv"
    full_metrics_path = tmp_path / "results/metrics/plan_d_ml1m_gru4rec_metrics.csv"

    assert exit_code == 0
    assert metrics_path.is_file()
    assert recommendations_path.is_file()
    assert not diagnostic_metrics_path.exists()
    assert not full_metrics_path.exists()
    metrics = pd.read_csv(metrics_path)
    assert metrics["scope"].eq("extended_subset").all()
    recommendations = pd.read_csv(recommendations_path)
    assert recommendations["scope"].eq("extended_subset").all()


def test_full_scope_rejects_caps_before_outputs_are_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full scope cannot be used with diagnostic max_users"):
        run_plan_d_gru4rec(
            config=GRU4RecConfig(
                dataset="ml1m",
                scope="full",
                epochs=1,
                max_users=2,
                max_train_interactions=4,
                requested_device="cpu",
            ),
            project_root=tmp_path,
            skip_plots=True,
        )

    assert not (tmp_path / "results/metrics/plan_d_ml1m_gru4rec_metrics.csv").exists()


def test_runner_does_not_overwrite_plan_c_outputs(tmp_path: Path) -> None:
    _write_tiny_plan_c_dataset(tmp_path)
    plan_c_path = tmp_path / "results/metrics/plan_c_ml1m_random_metrics.csv"
    plan_c_path.parent.mkdir(parents=True, exist_ok=True)
    plan_c_path.write_text("plan_c_metric_file\nkeep\n", encoding="utf-8")

    run_plan_d_gru4rec(
        config=GRU4RecConfig(
            dataset="ml1m",
            scope="diagnostic_subset",
            epochs=1,
            batch_size=2,
            embedding_dim=4,
            hidden_dim=4,
            num_layers=1,
            dropout=0.0,
            max_seq_len=3,
            seed=9,
            max_users=2,
            max_train_interactions=4,
            requested_device="cpu",
        ),
        project_root=tmp_path,
        skip_plots=True,
    )

    assert plan_c_path.read_text(encoding="utf-8") == "plan_c_metric_file\nkeep\n"
    assert not (tmp_path / "results/metrics/plan_c_ml1m_gru4rec_metrics.csv").exists()
    assert (tmp_path / "results/metrics/plan_d_ml1m_gru4rec_diagnostic_subset_metrics.csv").is_file()


def _write_tiny_plan_c_dataset(root: Path) -> None:
    base = root / "data/reconstructed/plan_c/ml1m"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3"],
            "item_id": ["A", "B", "B", "C", "D"],
            "split": ["train", "train", "train", "train", "train"],
            "timestamp": [1, 2, 1, 2, 1],
        }
    ).to_csv(base / "interactions.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "ground_truth_item_id": ["C", "A", "B"],
            "dataset": ["ml1m", "ml1m", "ml1m"],
            "split": ["test", "test", "test"],
        }
    ).to_csv(base / "ground_truth.csv", index=False)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3", "u3"],
            "candidate_item_id": ["A", "B", "C", "A", "B", "C", "B", "C", "D"],
            "candidate_position": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "is_ground_truth": [False, False, True, True, False, False, True, False, False],
        }
    ).to_csv(base / "candidates.csv", index=False)

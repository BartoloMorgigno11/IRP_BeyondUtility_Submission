from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.experiments.run_history_length import run_history_length
from src.prompts.history_length import (
    create_history_length_variants,
    normalize_history_length,
    truncate_user_histories,
)


def test_length_zero_returns_empty_history() -> None:
    result = truncate_user_histories(_history_frame(), 0)

    assert result.empty
    assert "history_position" in result.columns


def test_length_one_returns_most_recent_timestamped_item() -> None:
    result = truncate_user_histories(_history_frame(), 1)

    assert _items_for_user(result, "u1") == ["C"]
    assert _items_for_user(result, "u2") == ["Y"]


def test_length_one_returns_latest_ranked_item_when_timestamp_missing() -> None:
    histories = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["B", "C", "A"],
            "rank": [2, 3, 1],
        }
    )

    result = truncate_user_histories(histories, 1)

    assert _items_for_user(result, "u1") == ["C"]


def test_length_two_truncates_correctly() -> None:
    result = truncate_user_histories(_history_frame(), 2)

    assert _items_for_user(result, "u1") == ["B", "C"]
    assert _items_for_user(result, "u2") == ["X", "Y"]


def test_length_five_truncates_correctly() -> None:
    result = truncate_user_histories(_history_frame(), 5)

    assert _items_for_user(result, "u1") == ["A", "B", "C"]
    assert _items_for_user(result, "u2") == ["X", "Y"]


def test_length_ten_truncates_correctly() -> None:
    result = truncate_user_histories(_history_frame(), 10)

    assert _items_for_user(result, "u1") == ["A", "B", "C"]
    assert _items_for_user(result, "u2") == ["X", "Y"]


def test_full_preserves_all_available_history() -> None:
    result = truncate_user_histories(_history_frame(), "full")

    assert _items_for_user(result, "u1") == ["A", "B", "C"]
    assert _items_for_user(result, "u2") == ["X", "Y"]
    assert result.loc[result["user_id"] == "u1", "history_position"].tolist() == [1, 2, 3]


def test_users_with_shorter_histories_are_handled_correctly() -> None:
    result = truncate_user_histories(_history_frame(), 10)

    assert result.loc[result["user_id"] == "u2", "actual_history_length"].tolist() == [2, 2]
    assert _items_for_user(result, "u2") == ["X", "Y"]


def test_empty_histories_are_handled_correctly() -> None:
    empty = pd.DataFrame(columns=["user_id", "item_id", "timestamp"])

    truncated = truncate_user_histories(empty, "full")
    variants = create_history_length_variants(empty)

    assert truncated.empty
    assert variants.empty


def test_ordering_is_deterministic() -> None:
    histories = _history_frame()

    first = truncate_user_histories(histories, 2)
    second = truncate_user_histories(histories, 2)

    assert first[["user_id", "item_id", "history_position"]].to_dict("records") == second[
        ["user_id", "item_id", "history_position"]
    ].to_dict("records")


def test_prompt_input_variant_creation_is_reproducible() -> None:
    first = create_history_length_variants(_history_frame())
    second = create_history_length_variants(_history_frame())

    assert first.to_dict("records") == second.to_dict("records")
    u1_rows = first.loc[first["user_id"] == "u1"]
    assert u1_rows["history_length"].tolist() == [0, 1, 2, 5, 10, "full"]
    assert u1_rows["history_item_ids"].tolist() == [
        [],
        ["C"],
        ["B", "C"],
        ["A", "B", "C"],
        ["A", "B", "C"],
        ["A", "B", "C"],
    ]


def test_invalid_history_length_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="history length"):
        normalize_history_length("recent")


def test_history_length_runner_writes_diagnostic_when_history_unavailable(
    tmp_path: Path,
) -> None:
    _write_project_markers(tmp_path)
    config_path = _write_config(tmp_path, _minimal_config(interactions_path=None))

    metrics, summary, prompts_path, metrics_path, summary_path, analysis_path, blocked = (
        run_history_length(config_path, project_root=tmp_path)
    )

    assert blocked
    assert metrics.empty
    assert not prompts_path.exists()
    assert not metrics_path.exists()
    assert summary_path.is_file()
    assert analysis_path.is_file()
    assert summary.loc[summary["check"] == "history_data", "status"].iloc[0] == "not_configured"


def test_history_length_runner_writes_prompt_variants_from_synthetic_interactions(
    tmp_path: Path,
) -> None:
    _write_project_markers(tmp_path)
    interactions_path = "data/processed/demo/interactions.csv"
    interactions_file = tmp_path / interactions_path
    interactions_file.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["A", "B", "C"],
            "split": ["train", "train", "test"],
            "timestamp": [1, 2, 3],
        }
    ).to_csv(interactions_file, index=False)
    config_path = _write_config(tmp_path, _minimal_config(interactions_path=interactions_path))

    metrics, summary, prompts_path, metrics_path, summary_path, analysis_path, blocked = (
        run_history_length(config_path, project_root=tmp_path)
    )

    prompts = pd.read_csv(prompts_path)
    full_row = prompts.loc[prompts["history_length"].astype(str) == "full"].iloc[0]

    assert blocked
    assert metrics.empty
    assert prompts_path.is_file()
    assert not metrics_path.exists()
    assert summary_path.is_file()
    assert analysis_path.is_file()
    assert len(prompts) == 6
    assert full_row["history_item_ids"] == '["A", "B"]'
    assert "ready" in set(summary["status"])


def _history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "item_id": ["C", "A", "B", "Y", "X"],
            "timestamp": [3, 1, 2, 2, 1],
        }
    )


def _items_for_user(frame: pd.DataFrame, user_id: str) -> list[str]:
    return frame.loc[frame["user_id"] == user_id, "item_id"].tolist()


def _write_project_markers(root: Path) -> None:
    (root / "PROJECT_SPEC.md").write_text("# spec\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")


def _write_config(root: Path, config: dict) -> Path:
    config_path = root / "configs/demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _minimal_config(*, interactions_path: str | None) -> dict:
    return {
        "dataset": {
            "name": "demo",
            "raw_dir": "data/raw/demo",
            "processed_dir": "data/processed/demo",
            "sample_dir": "data/samples/demo",
            "interactions_path": interactions_path,
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
            "interactions_path": interactions_path,
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
            "llm_outputs": [
                {
                    "model_name": "gpt-test",
                    "source_file": "original_repo/EvaLLMasRecommender/llm_results/demo.zip",
                    "archive_member": "demo/results.json",
                    "run_id": "history-run",
                }
            ],
            "baselines": ["mostpop"],
            "baseline_outputs": [],
        },
        "metrics": {
            "utility": ["hr", "ndcg"],
            "beyond_utility": ["hallucination", "coverage", "history_length"],
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

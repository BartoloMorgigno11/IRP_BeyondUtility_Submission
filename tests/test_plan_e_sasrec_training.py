import ast
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.baselines.plan_e_sasrec_training as training
from src.baselines.plan_d_sasrec import SASRecModel
from src.baselines.plan_e_sasrec_adapter import final_non_padding_position
from src.baselines.plan_e_sasrec_training import (
    FiniteMaskedSASRecModel,
    SASRecPilotConfig,
    SASRecPilotSafetyError,
    aggregate_metrics,
    build_chronology,
    build_item_token_map_rows,
    build_model_indices,
    build_sequence_summary_rows,
    build_training_arrays,
    checkpoint_payload,
    finite_left_padded_attention_mask,
    load_interactions_with_source_rows,
    score_candidate_rows,
    set_deterministic_execution,
    state_dict_sha256,
    train_one_run,
    write_checkpoint_atomic,
)


def _config(device: str = "cpu", epochs: int = 1) -> SASRecPilotConfig:
    return SASRecPilotConfig(
        experiment_id="test_SASRec",
        model_id="test_model",
        dataset="ML-1M",
        seed=123,
        embedding_dim=4,
        hidden_size=4,
        num_attention_heads=1,
        num_layers=1,
        dropout=0.0,
        max_sequence_length=3,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=2,
        epochs=epochs,
        device=device,
        evaluation_k=5,
        max_run_minutes=1.0,
    )


def _write_interactions(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "interactions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _synthetic_chronology(tmp_path: Path):
    path = _write_interactions(
        tmp_path,
        [
            {"user_id": "u1", "item_id": "A", "split": "train", "timestamp": 2},
            {"user_id": "u1", "item_id": "B", "split": "train", "timestamp": 1},
            {"user_id": "u1", "item_id": "C", "split": "train", "timestamp": 1},
            {"user_id": "u2", "item_id": "A", "split": "train", "timestamp": 1},
            {"user_id": "u2", "item_id": "D", "split": "train", "timestamp": 2},
            {"user_id": "u1", "item_id": "V", "split": "validation", "timestamp": 3},
            {"user_id": "u2", "item_id": "T", "split": "test", "timestamp": 3},
        ],
    )
    return build_chronology(load_interactions_with_source_rows(path), max_sequence_length=3)


def test_seed_initial_state_and_no_builtin_hash_use() -> None:
    settings = set_deterministic_execution(7)
    assert settings["torch_use_deterministic_algorithms"] is True

    set_deterministic_execution(7)
    first = FiniteMaskedSASRecModel(num_items=5, embedding_dim=4, max_seq_len=3, num_heads=1, num_layers=1, dropout=0.0)
    first_hash = state_dict_sha256(first.state_dict())
    set_deterministic_execution(7)
    second = FiniteMaskedSASRecModel(num_items=5, embedding_dim=4, max_seq_len=3, num_heads=1, num_layers=1, dropout=0.0)
    assert state_dict_sha256(second.state_dict()) == first_hash

    tree = ast.parse(inspect.getsource(training))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash" for node in ast.walk(tree))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_seed_smoke_is_deterministic() -> None:
    set_deterministic_execution(99)
    first = torch.rand(3, device="cuda")
    set_deterministic_execution(99)
    second = torch.rand(3, device="cuda")
    assert torch.equal(first, second)


def test_finite_mask_blocks_padding_keys_without_nan() -> None:
    sequences = torch.tensor([[0, 0, 1], [0, 2, 3]], dtype=torch.long)
    mask = finite_left_padded_attention_mask(sequences, num_heads=1)

    assert mask.shape == (2, 3, 3)
    assert mask[0, 2, 0] is True or bool(mask[0, 2, 0])
    assert mask[0, 2, 1] is True or bool(mask[0, 2, 1])
    assert not bool(mask[0, 0, 0])

    model = FiniteMaskedSASRecModel(num_items=5, embedding_dim=4, max_seq_len=3, num_heads=1, num_layers=1, dropout=0.0)
    with torch.no_grad():
        scores = model.score_candidates(sequences, torch.tensor([[1, 2], [3, 4]], dtype=torch.long))
    assert torch.isfinite(scores).all()


def test_chronology_preserves_source_order_and_train_only(tmp_path: Path) -> None:
    chronology = _synthetic_chronology(tmp_path)

    assert chronology.train_rows == 5
    assert chronology.validation_rows == 1
    assert chronology.test_rows == 1
    assert chronology.train_histories["u1"] == ("B", "C", "A")
    assert chronology.history_source_rows["u1"] == (1, 2, 0)
    assert chronology.equal_timestamp_group_counts["u1"] == 1
    assert chronology.equal_timestamp_event_counts["u1"] == 2
    assert chronology.chronology_sha256 == build_chronology(
        load_interactions_with_source_rows(tmp_path / "interactions.csv"), max_sequence_length=3
    ).chronology_sha256


def test_chronology_hash_changes_when_equal_timestamp_source_order_changes(tmp_path: Path) -> None:
    first = _write_interactions(
        tmp_path,
        [
            {"user_id": "u1", "item_id": "B", "split": "train", "timestamp": 1},
            {"user_id": "u1", "item_id": "C", "split": "train", "timestamp": 1},
        ],
    )
    second = tmp_path / "interactions_reordered.csv"
    pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "C", "split": "train", "timestamp": 1},
            {"user_id": "u1", "item_id": "B", "split": "train", "timestamp": 1},
        ]
    ).to_csv(second, index=False)
    first_hash = build_chronology(load_interactions_with_source_rows(first), max_sequence_length=3).chronology_sha256
    second_hash = build_chronology(load_interactions_with_source_rows(second), max_sequence_length=3).chronology_sha256
    assert first_hash != second_hash


def test_training_objective_padding_and_empty_pair_failure(tmp_path: Path) -> None:
    chronology = _synthetic_chronology(tmp_path)
    user_index, item_index = build_model_indices(chronology)
    arrays = build_training_arrays(chronology, user_index=user_index, item_index=item_index, max_sequence_length=2)

    assert arrays.training_example_count == chronology.train_rows - chronology.unique_train_users
    assert arrays.sequences.shape == (3, 2)
    assert (arrays.positive_item_indices + 1 > 0).all()
    assert 0 not in arrays.positive_item_indices + 1
    assert arrays.sequences.tolist()[1] == [item_index["B"] + 1, item_index["C"] + 1]
    assert arrays.positive_item_indices[1] == item_index["A"]
    assert arrays.positive_item_indices[1] + 1 not in arrays.sequences[1].tolist()

    one_event_path = _write_interactions(
        tmp_path,
        [
            {"user_id": "u1", "item_id": "A", "split": "train", "timestamp": 1},
            {"user_id": "u2", "item_id": "B", "split": "train", "timestamp": 1},
        ],
    )
    one_event = build_chronology(load_interactions_with_source_rows(one_event_path), max_sequence_length=2)
    with pytest.raises(SASRecPilotSafetyError, match="training examples are empty"):
        build_training_arrays(one_event, user_index=build_model_indices(one_event)[0], item_index=build_model_indices(one_event)[1], max_sequence_length=2)


def test_sequence_summary_and_item_token_map(tmp_path: Path) -> None:
    chronology = _synthetic_chronology(tmp_path)
    item_index = {str(item_id): item_id - 1 for item_id in range(1, 21)}
    candidates = [str(item_id) for item_id in range(1, 21)]
    pilot_rows = [
        {
            "user_id": "u1",
            "candidate_items": json.dumps(candidates),
            "positive_item_id": "3",
            "included_in_pilot": True,
            "task_type": "ranking",
        }
    ]

    token_rows = build_item_token_map_rows(
        item_index=item_index,
        item_datamap={str(item_id): item_id - 1 for item_id in range(1, 21)},
        pilot_rows=pilot_rows,
        mapping_source_path="synthetic/datamaps.json",
    )
    token_ids = [row["sasrec_token_id"] for row in token_rows]
    assert min(token_ids) == 1
    assert 0 not in token_ids
    assert any(row["raw_item_id"] == "3" and row["positive_occurrence_count"] == 1 for row in token_rows)

    sequence_rows = build_sequence_summary_rows(chronology=chronology, pilot_cohort_rows=pilot_rows, max_sequence_length=2)
    user1 = next(row for row in sequence_rows if row["user_id"] == "u1")
    assert user1["effective_history_length"] == 2
    assert user1["history_truncated"] is True


def test_tiny_cpu_training_is_deterministic_and_checkpoint_is_restricted(tmp_path: Path) -> None:
    chronology = _synthetic_chronology(tmp_path)
    user_index, item_index = build_model_indices(chronology)
    arrays = build_training_arrays(chronology, user_index=user_index, item_index=item_index, max_sequence_length=3)
    config = _config(epochs=2)

    first = train_one_run(config=config, arrays=arrays, num_items=len(item_index), execution_role="primary")
    second = train_one_run(config=config, arrays=arrays, num_items=len(item_index), execution_role="repeat")

    assert first.initial_state_hash == second.initial_state_hash
    assert first.final_state_hash == second.final_state_hash
    assert [row["mean_training_loss"] for row in first.history] == [row["mean_training_loss"] for row in second.history]
    assert all(math.isfinite(row["mean_training_loss"]) for row in first.history)

    payload = checkpoint_payload(model=first.model, config={"seed": config.seed}, metadata={"run": first.run_id})
    assert {"model_state_dict", "config", "metadata"} <= set(payload)
    approved_root = tmp_path / "approved"
    checkpoint_path = approved_root / "model_state.pt"
    write_checkpoint_atomic(payload, checkpoint_path, approved_root=approved_root)
    assert checkpoint_path.exists()
    with pytest.raises(SASRecPilotSafetyError, match="outside approved root"):
        write_checkpoint_atomic(payload, tmp_path / "outside.pt", approved_root=approved_root)


def test_training_nan_loss_fails_before_backward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chronology = _synthetic_chronology(tmp_path)
    user_index, item_index = build_model_indices(chronology)
    arrays = build_training_arrays(chronology, user_index=user_index, item_index=item_index, max_sequence_length=3)

    def nan_loss(_positive: torch.Tensor, _negative: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(training, "bpr_loss", nan_loss)
    with pytest.raises(SASRecPilotSafetyError, match="non-finite SASRec loss"):
        train_one_run(config=_config(), arrays=arrays, num_items=len(item_index), execution_role="primary")


def test_scoring_preserves_candidates_and_uses_deterministic_tie_breaking() -> None:
    config = _config()
    item_ids = [str(i) for i in range(1, 24)]
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    model = FiniteMaskedSASRecModel(num_items=len(item_index), embedding_dim=4, max_seq_len=3, num_heads=1, num_layers=1, dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    candidates = ["10", "2", "1", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "4", "5", "6", "7"]
    rows = [
        {
            "task_type": "ranking",
            "candidate_set_id": "synthetic",
            "source_path": "synthetic.csv",
            "user_id": "u1",
            "positive_item_id": "4",
            "raw_train_history_length": 3,
            "effective_history_length": 3,
            "history_truncated": False,
            "candidate_items": json.dumps(candidates),
        }
    ]
    scored = score_candidate_rows(
        model,
        rows,
        train_histories={"u1": ("1", "2", "3")},
        item_index=item_index,
        config=config,
        device=torch.device("cpu"),
        training_run_id="run",
    )
    row = scored[0]
    assert json.loads(row["candidate_items"]) == candidates
    assert len(json.loads(row["candidate_scores"])) == 20
    assert set(json.loads(row["ranked_items"])) == set(candidates)
    assert json.loads(row["ranked_items"]) == sorted(candidates)
    assert row["positive_rank_zero_based"] == sorted(candidates).index("4")
    assert row["missing_score_count"] == 0


def test_scoring_rejects_positive_in_history() -> None:
    config = _config()
    item_index = {str(i): index for index, i in enumerate(range(1, 24))}
    model = FiniteMaskedSASRecModel(num_items=len(item_index), embedding_dim=4, max_seq_len=3, num_heads=1, num_layers=1, dropout=0.0)
    candidates = [str(i) for i in range(4, 24)]
    rows = [
        {
            "task_type": "ranking",
            "candidate_set_id": "synthetic",
            "source_path": "synthetic.csv",
            "user_id": "u1",
            "positive_item_id": "4",
            "raw_train_history_length": 3,
            "effective_history_length": 3,
            "history_truncated": False,
            "candidate_items": json.dumps(candidates),
        }
    ]
    with pytest.raises(SASRecPilotSafetyError, match="held-out positive"):
        score_candidate_rows(
            model,
            rows,
            train_histories={"u1": ("1", "2", "4")},
            item_index=item_index,
            config=config,
            device=torch.device("cpu"),
            training_run_id="run",
        )


def test_metrics_hr_ndcg_positive_rank_and_truncation_rate() -> None:
    config = _config()
    rows = []
    for task in ("ranking", "reranking"):
        for rank, truncated in [(0, True), (6, False)]:
            ranked = [str(i) for i in range(20)]
            positive = ranked[rank]
            rows.append(
                {
                    "experiment_id": config.experiment_id,
                    "training_run_id": "run",
                    "model_id": config.model_id,
                    "dataset": "ML-1M",
                    "task_type": task,
                    "candidate_set_id": f"synthetic_{task}",
                    "source_path": "synthetic.csv",
                    "user_id": f"{task}_{rank}",
                    "positive_item_id": positive,
                    "raw_train_history_length": 60 if truncated else 4,
                    "effective_history_length": 3,
                    "history_truncated": truncated,
                    "candidate_items": json.dumps(ranked),
                    "candidate_scores": json.dumps([float(20 - i) for i in range(20)]),
                    "ranked_items": json.dumps(ranked),
                    "ranked_scores": json.dumps([float(20 - i) for i in range(20)]),
                    "positive_rank_zero_based": rank,
                    "positive_rank_one_based": rank + 1,
                    "positive_in_top_5": rank < 5,
                    "ndcg_at_5": 1.0 if rank == 0 else 0.0,
                    "ranking_valid": True,
                    "duplicate_ranked_item_count": 0,
                    "missing_score_count": 0,
                    "pilot_full_cohort_id": "plan_e2_sasrec_strict_causal_full",
                    "pilot_matched_cohort_id": "plan_e2_sasrec_strict_causal_matched",
                    "eligible_pilot_full": True,
                    "eligible_pilot_matched": True,
                    "notes": "",
                }
            )

    metrics = aggregate_metrics(rows, config=config, training_run_id="run")
    assert len(metrics) == 4
    first = next(row for row in metrics if row["task_type"] == "ranking" and row["cohort_id"].endswith("full"))
    assert first["evaluated_rows"] == 2
    assert first["hr_at_5"] == 0.5
    assert first["ndcg_at_5"] == 0.5
    assert first["median_positive_rank_zero_based"] == 3.0
    assert first["history_truncation_rate"] == 0.5


def test_last_non_padding_position_uses_final_valid_token() -> None:
    assert final_non_padding_position([0, 0, 2, 3]) == 3
    assert final_non_padding_position([0, 5, 0, 0]) == 1
    with pytest.raises(Exception, match="fully padded"):
        final_non_padding_position([0, 0, 0, 0])




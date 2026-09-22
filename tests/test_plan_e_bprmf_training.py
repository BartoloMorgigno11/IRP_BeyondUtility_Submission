from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.baselines.plan_e_bprmf_training import (
    PilotConfig,
    PilotSafetyError,
    aggregate_metrics,
    build_model_indices,
    checkpoint_payload,
    load_train_interactions,
    prepare_training_arrays,
    sample_negatives,
    score_candidate_rows,
    set_deterministic_execution,
    state_dict_sha256,
    train_one_run,
    validate_negative_samples,
    write_checkpoint_atomic,
)


def _config(device: str = "cpu", epochs: int = 2) -> PilotConfig:
    return PilotConfig(
        experiment_id="synthetic",
        model_id="bprmf_synthetic",
        dataset="Synthetic",
        seed=7,
        embedding_dim=4,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=2,
        epochs=epochs,
        device=device,
        evaluation_k=5,
        max_run_minutes=1,
    )


def _train_frame() -> pd.DataFrame:
    users = ["u1", "u1", "u2", "u2", "u3"]
    items = ["i1", "i2", "i2", "i3", "i1"]
    for idx in range(20):
        users.append("u1")
        items.append(f"c{idx:02d}")
    return pd.DataFrame(
        {
            "user_id": users,
            "item_id": items,
            "split": ["train"] * len(users),
        }
    )


def _pilot_rows() -> list[dict[str, str]]:
    candidates = ["i1", "i2", "i3"] + [f"ix{x}" for x in range(17)]
    return [
        {
            "candidate_set_id": "synthetic_rank",
            "source_path": "synthetic.csv",
            "dataset": "Synthetic",
            "task_type": "ranking",
            "user_id": "u1",
            "positive_item_id": "i1",
            "candidate_items": json.dumps(candidates),
            "eligible_plan_e2_matched": "true",
        }
    ]


def test_seed_application_produces_identical_initial_state() -> None:
    from src.baselines.plan_d_bprmf import BPRMFModel

    set_deterministic_execution(123)
    first = BPRMFModel(2, 3, 4)
    first_hash = state_dict_sha256(first.state_dict())
    set_deterministic_execution(123)
    second = BPRMFModel(2, 3, 4)
    assert first_hash == state_dict_sha256(second.state_dict())


def test_identical_cpu_training_data_produces_identical_final_state() -> None:
    train = _train_frame()
    pilot = [{"user_id": "u1", "positive_item_id": "c00", "candidate_items": json.dumps([f"c{x:02d}" for x in range(20)])}]
    user_index, item_index, _, _ = build_model_indices(train, pilot)
    arrays = prepare_training_arrays(train, user_index=user_index, item_index=item_index)
    first = train_one_run(config=_config(), arrays=arrays, num_users=len(user_index), num_items=len(item_index), execution_role="primary")
    second = train_one_run(config=_config(), arrays=arrays, num_users=len(user_index), num_items=len(item_index), execution_role="same_seed_repeat")
    assert first.initial_state_hash == second.initial_state_hash
    assert first.final_state_hash == second.final_state_hash


def test_cuda_deterministic_smoke_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    train = _train_frame()
    pilot = [{"user_id": "u1", "positive_item_id": "c00", "candidate_items": json.dumps([f"c{x:02d}" for x in range(20)])}]
    user_index, item_index, _, _ = build_model_indices(train, pilot)
    arrays = prepare_training_arrays(train, user_index=user_index, item_index=item_index)
    result = train_one_run(
        config=_config(device="cuda", epochs=1),
        arrays=arrays,
        num_users=len(user_index),
        num_items=len(item_index),
        execution_role="primary",
    )
    assert result.history[0]["finite_loss"] is True


def test_load_train_interactions_accepts_only_train_split(tmp_path: Path) -> None:
    path = tmp_path / "interactions.csv"
    pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "item_id": ["i1", "i2", "i3"],
            "split": ["train", "validation", "test"],
        }
    ).to_csv(path, index=False)
    train, summary = load_train_interactions(path)
    assert train["split"].tolist() == ["train"]
    assert summary["training_interaction_count"] == 1


def test_empty_training_frame_fails(tmp_path: Path) -> None:
    path = tmp_path / "interactions.csv"
    pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"], "split": ["test"]}).to_csv(path, index=False)
    with pytest.raises(PilotSafetyError):
        load_train_interactions(path)


def test_negative_sampling_contract_is_seeded_and_safe() -> None:
    users = np.array([0, 0, 1, 1], dtype=np.int64)
    positives = np.array([0, 1, 1, 2], dtype=np.int64)
    universe = np.array([0, 1, 2, 3], dtype=np.int64)
    user_pos = {0: {0, 1}, 1: {1, 2}}
    first = sample_negatives(users, item_universe=universe, user_positive_items=user_pos, rng=np.random.default_rng(5))
    second = sample_negatives(users, item_universe=universe, user_positive_items=user_pos, rng=np.random.default_rng(5))
    assert first.tolist() == second.tolist()
    validate_negative_samples(users, positives, first, item_universe=universe, user_positive_items=user_pos)


def test_tiny_training_produces_finite_loss_and_expected_shapes() -> None:
    train = _train_frame()
    pilot = [{"user_id": "u1", "positive_item_id": "c00", "candidate_items": json.dumps([f"c{x:02d}" for x in range(20)])}]
    user_index, item_index, _, _ = build_model_indices(train, pilot)
    arrays = prepare_training_arrays(train, user_index=user_index, item_index=item_index)
    result = train_one_run(config=_config(epochs=1), arrays=arrays, num_users=len(user_index), num_items=len(item_index), execution_role="primary")
    assert result.history[0]["batch_count"] == int(np.ceil(len(arrays.user_indices) / _config(epochs=1).batch_size))
    assert np.isfinite(result.history[0]["mean_total_loss"])


def test_scoring_preserves_candidate_order_and_tie_breaking() -> None:
    from src.baselines.plan_d_bprmf import BPRMFModel

    candidates = ["i2", "i1", "i3"] + [f"ix{x}" for x in range(17)]
    user_index = {"u1": 0}
    item_index = {item: idx for idx, item in enumerate(candidates)}
    model = BPRMFModel(num_users=1, num_items=len(item_index), embedding_dim=2)
    with torch.no_grad():
        model.user_embedding.weight[:] = torch.tensor([[1.0, 0.0]])
        model.item_embedding.weight[:] = torch.zeros_like(model.item_embedding.weight)
    rows = score_candidate_rows(
        model,
        [
            {
                "candidate_set_id": "synthetic_rank",
                "source_path": "synthetic.csv",
                "task_type": "ranking",
                "user_id": "u1",
                "positive_item_id": "i1",
                "candidate_items": json.dumps(candidates),
                "eligible_plan_e2_matched": "true",
            }
        ],
        user_index=user_index,
        item_index=item_index,
        config=_config(),
        device=torch.device("cpu"),
    )
    assert json.loads(rows[0]["candidate_items"]) == candidates
    assert json.loads(rows[0]["ranked_items"])[:3] == ["i1", "i2", "i3"]
    assert rows[0]["duplicate_ranked_item_count"] == 0


def test_metric_aggregation_full_and_matched() -> None:
    row = {
        "experiment_id": "synthetic",
        "training_run_id": "synthetic_primary",
        "model_id": "m",
        "dataset": "ML-1M",
        "task_type": "ranking",
        "candidate_set_id": "cs",
        "positive_rank_zero_based": 0,
        "positive_in_top_5": True,
        "ndcg_at_5": 1.0,
        "ranking_valid": True,
        "duplicate_ranked_item_count": 0,
        "missing_score_count": 0,
        "eligible_pilot_full": True,
        "eligible_pilot_matched": True,
    }
    metrics = aggregate_metrics([row], config=_config())
    assert len(metrics) == 2
    assert all(metric["hr_at_5"] == 1.0 for metric in metrics)


def test_checkpoint_atomic_write_and_path_restriction(tmp_path: Path) -> None:
    from src.baselines.plan_d_bprmf import BPRMFModel

    model = BPRMFModel(1, 1, 2)
    payload = checkpoint_payload(model=model, config={"x": 1}, metadata={"m": 2})
    approved = tmp_path / "approved"
    target = approved / "model_state.pt"
    write_checkpoint_atomic(payload, target, approved_root=approved)
    assert target.exists()
    with pytest.raises(PilotSafetyError):
        write_checkpoint_atomic(payload, tmp_path / "outside.pt", approved_root=approved)

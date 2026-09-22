from __future__ import annotations

from math import log2

import numpy as np
import pandas as pd
import pytest
import torch

from src.baselines.plan_d_bprmf import (
    BPRMFConfig,
    BPRMFModel,
    DeterministicNegativeSampler,
    PreparedBPRData,
    TrainingResult,
    bpr_loss,
    build_metric_rows,
    rank_bprmf_candidates,
    select_device,
    set_deterministic_seeds,
)


def test_bpr_loss_is_finite_on_tiny_batch() -> None:
    set_deterministic_seeds(7)
    model = BPRMFModel(num_users=2, num_items=3, embedding_dim=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    users = torch.tensor([0, 1], dtype=torch.long)
    positives = torch.tensor([0, 1], dtype=torch.long)
    negatives = torch.tensor([2, 2], dtype=torch.long)

    positive_scores, negative_scores = model(users, positives, negatives)
    initial = bpr_loss(positive_scores, negative_scores)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        positive_scores, negative_scores = model(users, positives, negatives)
        loss = bpr_loss(positive_scores, negative_scores)
        loss.backward()
        optimizer.step()

    positive_scores, negative_scores = model(users, positives, negatives)
    final = bpr_loss(positive_scores, negative_scores)

    assert torch.isfinite(initial)
    assert torch.isfinite(final)


def test_negative_sampler_never_samples_user_positives_and_is_repeatable() -> None:
    positives = {0: {0, 1}, 1: {2}}
    users = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    first = DeterministicNegativeSampler(
        item_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        user_positive_items=positives,
        seed=123,
    ).sample(users)
    second = DeterministicNegativeSampler(
        item_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        user_positive_items=positives,
        seed=123,
    ).sample(users)

    assert first.tolist() == second.tolist()
    for user_idx, item_idx in zip(users, first):
        assert int(item_idx) not in positives[int(user_idx)]


def test_candidate_ranking_returns_exact_top_k_items() -> None:
    data = _prepared_data()
    config = BPRMFConfig(dataset="ml1m", scope="diagnostic_subset", epochs=1, k=3)
    model = BPRMFModel(num_users=1, num_items=4, embedding_dim=2)
    with torch.no_grad():
        model.user_embedding.weight[:] = torch.tensor([[1.0, 0.0]])
        model.item_embedding.weight[:] = torch.tensor(
            [
                [0.2, 0.0],
                [0.9, 0.0],
                [0.5, 0.0],
                [0.7, 0.0],
            ]
        )

    recommendations = rank_bprmf_candidates(model, data, config=config, device=torch.device("cpu"))

    assert recommendations["recommended_item_id"].tolist() == ["B", "D", "C"]
    assert recommendations["rank"].tolist() == [1, 2, 3]


def test_metric_rows_compute_hr_and_ndcg_on_toy_recommendations() -> None:
    data = _prepared_data_two_users()
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "recommended_item_id": ["A", "B", "A", "B"],
            "rank": [1, 2, 1, 2],
        }
    )
    training = TrainingResult(
        model=BPRMFModel(num_users=2, num_items=2, embedding_dim=2),
        epochs_completed=1,
        best_epoch=1,
        final_train_loss=0.5,
        best_train_loss=0.5,
        device="cpu",
    )
    metrics = build_metric_rows(
        recommendations,
        data,
        config=BPRMFConfig(dataset="ml1m", scope="diagnostic_subset", epochs=1),
        training=training,
        runtime_seconds=1.0,
        config_hash="abc123",
        recommendations_path="results/raw_outputs/plan_d_ml1m_bprmf_diagnostic_subset_recommendations.csv",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    values = dict(zip(metrics["metric"], metrics["value"]))
    assert values["hit_rate"] == pytest.approx(1.0)
    assert values["ndcg"] == pytest.approx((1.0 + 1.0 / log2(3)) / 2)


def test_device_selection_falls_back_to_cpu_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    assert select_device("cuda").type == "cpu"


def _prepared_data() -> PreparedBPRData:
    return PreparedBPRData(
        dataset="ml1m",
        scope="diagnostic_subset",
        train_user_indices=np.asarray([0], dtype=np.int64),
        train_item_indices=np.asarray([0], dtype=np.int64),
        user_positive_items={0: {0}},
        item_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        user_id_to_index={"u1": 0},
        item_id_to_index={"A": 0, "B": 1, "C": 2, "D": 3},
        index_to_user_id={0: "u1"},
        index_to_item_id={0: "A", 1: "B", 2: "C", 3: "D"},
        candidates=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u1"],
                "candidate_item_id": ["A", "B", "C", "D"],
                "candidate_position": [1, 2, 3, 4],
                "is_ground_truth": [False, False, True, False],
            }
        ),
        ground_truth=pd.DataFrame({"user_id": ["u1"], "ground_truth_item_id": ["C"]}),
        num_train_interactions_before_limit=1,
    )


def _prepared_data_two_users() -> PreparedBPRData:
    return PreparedBPRData(
        dataset="ml1m",
        scope="diagnostic_subset",
        train_user_indices=np.asarray([0, 1], dtype=np.int64),
        train_item_indices=np.asarray([0, 1], dtype=np.int64),
        user_positive_items={0: {0}, 1: {1}},
        item_indices=np.asarray([0, 1], dtype=np.int64),
        user_id_to_index={"u1": 0, "u2": 1},
        item_id_to_index={"A": 0, "B": 1},
        index_to_user_id={0: "u1", 1: "u2"},
        index_to_item_id={0: "A", 1: "B"},
        candidates=pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2"],
                "candidate_item_id": ["A", "B", "A", "B"],
                "candidate_position": [1, 2, 1, 2],
                "is_ground_truth": [True, False, False, True],
            }
        ),
        ground_truth=pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "ground_truth_item_id": ["A", "B"],
            }
        ),
        num_train_interactions_before_limit=2,
    )


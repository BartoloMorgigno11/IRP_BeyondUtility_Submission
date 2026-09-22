from __future__ import annotations

from math import log2

import numpy as np
import pandas as pd
import pytest
import torch

from src.baselines.plan_d_bprmf import DeterministicNegativeSampler, select_device
from src.baselines.plan_d_gru4rec import (
    GRU4RecConfig,
    GRU4RecModel,
    GRU4RecTrainingResult,
    PreparedGRU4RecData,
    bpr_loss,
    build_metric_rows,
    build_user_sequences,
    last_non_padding_positions,
    pad_sequence,
    rank_gru4rec_candidates,
)


def test_gru4rec_forward_returns_score_shapes() -> None:
    model = GRU4RecModel(
        num_items=5,
        embedding_dim=4,
        hidden_dim=6,
        num_layers=1,
        dropout=0.0,
    )
    sequences = torch.tensor([[0, 0, 1, 2], [0, 3, 4, 5]], dtype=torch.long)
    positives = torch.tensor([3, 4], dtype=torch.long)
    negatives = torch.tensor([4, 1], dtype=torch.long)
    candidates = torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=torch.long)

    positive_scores, negative_scores = model(sequences, positives, negatives)
    candidate_scores = model.score_candidates(sequences, candidates)

    assert positive_scores.shape == (2,)
    assert negative_scores.shape == (2,)
    assert candidate_scores.shape == (2, 3)


def test_sequence_builder_uses_only_training_interactions_and_timestamp_order() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1"],
            "item_id": ["B", "T", "A", "C"],
            "split": ["train", "test", "train", "train"],
            "timestamp": [2, 3, 1, 4],
        }
    )

    sequences, count, note = build_user_sequences(
        interactions,
        user_id_to_index={"u1": 0},
        item_id_to_index={"A": 0, "B": 1, "C": 2, "T": 3},
        max_train_interactions=None,
    )

    assert count == 3
    assert sequences[0] == [0, 1, 2]
    assert "timestamp" in note


def test_sequence_builder_preserves_existing_order_without_timestamp() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["B", "A", "C"],
            "split": ["train", "train", "train"],
        }
    )

    sequences, _, note = build_user_sequences(
        interactions,
        user_id_to_index={"u1": 0},
        item_id_to_index={"A": 0, "B": 1, "C": 2},
        max_train_interactions=None,
    )

    assert sequences[0] == [1, 0, 2]
    assert "existing interaction row order" in note


def test_padding_and_last_non_padding_positions() -> None:
    assert pad_sequence([0, 2], 5).tolist() == [0, 0, 0, 1, 3]
    assert pad_sequence([0, 1, 2, 3], 3).tolist() == [2, 3, 4]

    positions = last_non_padding_positions(torch.tensor([[0, 0, 1, 2], [0, 3, 0, 0]], dtype=torch.long))

    assert positions.tolist() == [3, 1]


def test_gru4rec_bpr_loss_is_finite_on_tiny_batch() -> None:
    model = GRU4RecModel(
        num_items=4,
        embedding_dim=4,
        hidden_dim=4,
        num_layers=1,
        dropout=0.0,
    )
    sequences = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    positives = torch.tensor([3, 4], dtype=torch.long)
    negatives = torch.tensor([4, 1], dtype=torch.long)

    positive_scores, negative_scores = model(sequences, positives, negatives)
    loss = bpr_loss(positive_scores, negative_scores)

    assert torch.isfinite(loss)


def test_negative_sampler_never_samples_user_positives_and_is_repeatable() -> None:
    positives = {0: {0, 1}, 1: {2}}
    users = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    first = DeterministicNegativeSampler(
        item_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        user_positive_items=positives,
        seed=44,
    ).sample(users)
    second = DeterministicNegativeSampler(
        item_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        user_positive_items=positives,
        seed=44,
    ).sample(users)

    assert first.tolist() == second.tolist()
    for user_idx, item_idx in zip(users, first):
        assert int(item_idx) not in positives[int(user_idx)]


def test_candidate_ranking_uses_only_supplied_candidate_set() -> None:
    data = _prepared_data()
    config = GRU4RecConfig(
        dataset="ml1m",
        scope="diagnostic_subset",
        epochs=1,
        embedding_dim=4,
        hidden_dim=4,
        num_layers=1,
        dropout=0.0,
        max_seq_len=3,
        k=3,
    )
    model = _FixedCandidateScorer(scores=torch.tensor([[0.2, 0.9, 0.5, 0.7]], dtype=torch.float32))

    recommendations = rank_gru4rec_candidates(model, data, config=config, device=torch.device("cpu"))

    assert recommendations["recommended_item_id"].tolist() == ["B", "D", "C"]
    assert recommendations["rank"].tolist() == [1, 2, 3]
    assert set(recommendations["recommended_item_id"]).issubset({"A", "B", "C", "D"})


def test_metric_rows_compute_hr_and_ndcg_on_toy_recommendations() -> None:
    data = _prepared_data_two_users()
    recommendations = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "recommended_item_id": ["A", "B", "A", "B"],
            "rank": [1, 2, 1, 2],
        }
    )
    training = GRU4RecTrainingResult(
        model=GRU4RecModel(
            num_items=2,
            embedding_dim=4,
            hidden_dim=4,
            num_layers=1,
            dropout=0.0,
        ),
        epochs_completed=1,
        best_epoch=1,
        final_train_loss=0.5,
        best_train_loss=0.5,
        device="cpu",
    )
    metrics = build_metric_rows(
        recommendations,
        data,
        config=GRU4RecConfig(dataset="ml1m", scope="diagnostic_subset", epochs=1, embedding_dim=4, hidden_dim=4),
        training=training,
        runtime_seconds=1.0,
        config_hash="abc123",
        recommendations_path="results/raw_outputs/plan_d_ml1m_gru4rec_diagnostic_subset_recommendations.csv",
        timestamp="2026-01-01T00:00:00+00:00",
    )

    values = dict(zip(metrics["metric"], metrics["value"]))
    assert values["hit_rate"] == pytest.approx(1.0)
    assert values["ndcg"] == pytest.approx((1.0 + 1.0 / log2(3)) / 2)


def test_device_selection_falls_back_to_cpu_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    assert select_device("cuda").type == "cpu"


class _FixedCandidateScorer:
    def __init__(self, scores: torch.Tensor) -> None:
        self.scores = scores

    def eval(self) -> "_FixedCandidateScorer":
        return self

    def score_candidates(self, sequences: torch.Tensor, candidate_item_tokens: torch.Tensor) -> torch.Tensor:
        return self.scores[:, : candidate_item_tokens.shape[1]]


def _prepared_data() -> PreparedGRU4RecData:
    return PreparedGRU4RecData(
        dataset="ml1m",
        scope="diagnostic_subset",
        train_sequences=np.asarray([[0, 0, 1]], dtype=np.int64),
        train_user_indices=np.asarray([0], dtype=np.int64),
        train_positive_item_indices=np.asarray([1], dtype=np.int64),
        eval_sequences_by_user={"u1": np.asarray([0, 0, 1], dtype=np.int64)},
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
        num_train_interactions=1,
        ordering_note="test ordering",
    )


def _prepared_data_two_users() -> PreparedGRU4RecData:
    return PreparedGRU4RecData(
        dataset="ml1m",
        scope="diagnostic_subset",
        train_sequences=np.asarray([[0, 0, 1], [0, 0, 2]], dtype=np.int64),
        train_user_indices=np.asarray([0, 1], dtype=np.int64),
        train_positive_item_indices=np.asarray([1, 0], dtype=np.int64),
        eval_sequences_by_user={
            "u1": np.asarray([0, 0, 1], dtype=np.int64),
            "u2": np.asarray([0, 0, 2], dtype=np.int64),
        },
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
        num_train_interactions=2,
        ordering_note="test ordering",
    )

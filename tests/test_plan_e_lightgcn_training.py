from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.baselines.plan_d_lightgcn import LightGCNModel
from src.baselines.plan_e_lightgcn_adapter import GraphValidationError, build_graph_contract
from src.baselines.plan_e_lightgcn_training import (
    LightGCNPilotConfig,
    LightGCNPilotSafetyError,
    aggregate_metrics,
    build_negative_schedule,
    build_training_arrays,
    checkpoint_payload,
    embedding_sha256,
    graph_hashes,
    load_train_interactions,
    max_abs_history_diff,
    metric_lookup,
    propagate_final_embeddings,
    rows_sha256,
    score_candidate_rows,
    set_deterministic_execution,
    sparse_tensor_from_graph,
    stable_json_sha256,
    state_dict_sha256,
    train_one_run,
    validate_negative_samples,
    validate_project_sparse_backend,
    write_checkpoint_atomic,
)


def _config(device: str = "cpu", epochs: int = 2) -> LightGCNPilotConfig:
    return LightGCNPilotConfig(
        experiment_id="synthetic_lightgcn",
        model_id="lightgcn_synthetic",
        dataset="ML-1M",
        seed=17,
        embedding_dim=4,
        propagation_layers=1,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=4,
        epochs=epochs,
        device=device,
        evaluation_k=5,
        max_run_minutes=1,
    )


def _train_frame() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for item_id in ["i00", "i01", "i02", "i03", "i04"]:
        rows.append({"user_id": "u01", "item_id": item_id, "split": "train"})
    for item_id in ["i05", "i06", "i07", "i08", "i09", "i10", "i11"]:
        rows.append({"user_id": "u02", "item_id": item_id, "split": "train"})
    for item_id in ["i12", "i13", "i14", "i15", "i16", "i17", "i18", "i19"]:
        rows.append({"user_id": "u03", "item_id": item_id, "split": "train"})
    return pd.DataFrame(rows)


def _contract():
    return build_graph_contract(_train_frame())


def _candidate_items() -> list[str]:
    return ["i02", "i01", "i00"] + [f"i{index:02d}" for index in range(3, 20)]


def _pilot_rows(task_type: str = "ranking", matched: bool = True) -> list[dict[str, str]]:
    return [
        {
            "candidate_set_id": f"synthetic_{task_type}",
            "source_path": "synthetic.csv",
            "task_type": task_type,
            "user_id": "u01",
            "positive_item_id": "i01",
            "candidate_items": json.dumps(_candidate_items()),
            "eligible_plan_e2_matched": "true" if matched else "false",
        }
    ]


def _degree_maps(contract) -> tuple[dict[str, float], dict[str, float]]:
    degrees = contract.normalized_graph.degrees
    user_degrees = {
        raw_id: float(degrees[local_index])
        for raw_id, local_index in contract.user_mapping.raw_to_local.items()
    }
    item_offset = contract.user_mapping.count
    item_degrees = {
        raw_id: float(degrees[item_offset + local_index])
        for raw_id, local_index in contract.item_mapping.raw_to_local.items()
    }
    return user_degrees, item_degrees


def test_seed_application_produces_identical_initial_lightgcn_state() -> None:
    set_deterministic_execution(123)
    first = LightGCNModel(num_users=2, num_items=3, embedding_dim=4, num_layers=1)
    first_hash = state_dict_sha256(first.state_dict())

    set_deterministic_execution(123)
    second = LightGCNModel(num_users=2, num_items=3, embedding_dim=4, num_layers=1)

    assert first_hash == state_dict_sha256(second.state_dict())


def test_load_train_interactions_filters_train_and_preserves_string_ids(tmp_path: Path) -> None:
    path = tmp_path / "interactions.csv"
    pd.DataFrame(
        {
            "user_id": ["001", "002", "003"],
            "item_id": ["0007", "0008", "0009"],
            "split": ["train", "validation", "test"],
        }
    ).to_csv(path, index=False)

    train, summary = load_train_interactions(path)

    assert train["user_id"].tolist() == ["001"]
    assert train["item_id"].tolist() == ["0007"]
    assert summary["training_interaction_count"] == 1


def test_empty_train_interactions_fail(tmp_path: Path) -> None:
    path = tmp_path / "interactions.csv"
    pd.DataFrame({"user_id": ["u1"], "item_id": ["i1"], "split": ["test"]}).to_csv(path, index=False)

    with pytest.raises(GraphValidationError):
        load_train_interactions(path)


def test_graph_training_arrays_sparse_tensor_and_hashes_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "interactions.csv"
    _train_frame().to_csv(path, index=False)
    contract = _contract()
    arrays = build_training_arrays(contract)
    graph_hash_one = graph_hashes(contract, path)
    graph_hash_two = graph_hashes(contract, path)
    sparse = sparse_tensor_from_graph(contract.normalized_graph, device=torch.device("cpu"))
    probe = validate_project_sparse_backend(sparse)

    assert arrays.unique_train_users == 3
    assert arrays.unique_train_items == 20
    assert arrays.train_pair_count == 20
    assert graph_hash_one == graph_hash_two
    assert sparse.is_coalesced()
    assert probe["finite"] is True


def test_negative_schedule_is_seeded_safe_and_complete() -> None:
    arrays = build_training_arrays(_contract())
    first = build_negative_schedule(arrays=arrays, seed=5, epochs=2)
    second = build_negative_schedule(arrays=arrays, seed=5, epochs=2)

    assert first.evidence_sha256 == second.evidence_sha256
    assert first.checked_triple_count == arrays.train_pair_count * 2
    validate_negative_samples(
        arrays.user_indices[first.epoch_orders[0]],
        arrays.item_indices[first.epoch_orders[0]],
        first.epoch_negatives[0],
        item_universe=arrays.item_universe,
        user_positive_items=arrays.user_positive_items,
    )


def test_identical_cpu_training_runs_produce_identical_state_embeddings_and_losses() -> None:
    contract = _contract()
    arrays = build_training_arrays(contract)
    sparse = sparse_tensor_from_graph(contract.normalized_graph, device=torch.device("cpu"))
    schedule = build_negative_schedule(arrays=arrays, seed=17, epochs=2)

    first = train_one_run(
        config=_config(),
        arrays=arrays,
        normalized_adj=sparse,
        schedule=schedule,
        execution_role="primary",
    )
    second = train_one_run(
        config=_config(),
        arrays=arrays,
        normalized_adj=sparse,
        schedule=schedule,
        execution_role="same_seed_repeat",
    )

    assert first.initial_state_hash == second.initial_state_hash
    assert first.final_state_hash == second.final_state_hash
    assert first.final_embedding_hash == second.final_embedding_hash
    assert max_abs_history_diff(first.history, second.history) == pytest.approx(0.0)
    assert all(row["finite_loss"] for row in first.history)


def test_cuda_sparse_training_smoke_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    contract = _contract()
    arrays = build_training_arrays(contract)
    sparse = sparse_tensor_from_graph(contract.normalized_graph, device=torch.device("cuda"))
    schedule = build_negative_schedule(arrays=arrays, seed=19, epochs=1)
    validate_project_sparse_backend(sparse)

    result = train_one_run(
        config=_config(device="cuda", epochs=1),
        arrays=arrays,
        normalized_adj=sparse,
        schedule=schedule,
        execution_role="primary",
    )

    assert result.history[0]["finite_loss"] is True
    assert result.final_state_hash


def test_final_propagation_and_embedding_hash_are_stable() -> None:
    contract = _contract()
    sparse = sparse_tensor_from_graph(contract.normalized_graph, device=torch.device("cpu"))
    model = LightGCNModel(
        num_users=contract.user_mapping.count,
        num_items=contract.item_mapping.count,
        embedding_dim=2,
        num_layers=1,
    )
    with torch.no_grad():
        model.user_embedding.weight.fill_(0.5)
        model.item_embedding.weight.fill_(0.25)

    users_one, items_one = propagate_final_embeddings(model, sparse)
    users_two, items_two = propagate_final_embeddings(model, sparse)

    assert torch.isfinite(users_one).all()
    assert embedding_sha256(users_one, items_one) == embedding_sha256(users_two, items_two)


def test_scoring_preserves_candidate_membership_and_uses_deterministic_tie_breaking() -> None:
    contract = _contract()
    user_degrees, item_degrees = _degree_maps(contract)
    user_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32)
    item_embeddings = torch.zeros((contract.item_mapping.count, 2), dtype=torch.float32)

    rows = score_candidate_rows(
        _pilot_rows(),
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        user_index=contract.user_mapping.raw_to_local,
        item_index=contract.item_mapping.raw_to_local,
        user_degrees=user_degrees,
        item_degrees=item_degrees,
        config=_config(),
        training_run_id="synthetic_lightgcn_primary",
    )

    assert json.loads(rows[0]["candidate_items"]) == _candidate_items()
    assert json.loads(rows[0]["ranked_items"])[:3] == ["i00", "i01", "i02"]
    assert json.loads(rows[0]["candidate_scores"]) == [0.0] * 20
    assert rows[0]["positive_rank_zero_based"] == 1
    assert rows[0]["positive_in_top_5"] is True
    assert rows[0]["duplicate_ranked_item_count"] == 0


def test_scoring_rejects_unmapped_candidates() -> None:
    contract = _contract()
    user_degrees, item_degrees = _degree_maps(contract)
    bad_row = dict(_pilot_rows()[0])
    candidates = _candidate_items()
    candidates[-1] = "missing_item"
    bad_row["candidate_items"] = json.dumps(candidates)

    with pytest.raises(LightGCNPilotSafetyError):
        score_candidate_rows(
            [bad_row],
            user_embeddings=torch.zeros((contract.user_mapping.count, 2)),
            item_embeddings=torch.zeros((contract.item_mapping.count, 2)),
            user_index=contract.user_mapping.raw_to_local,
            item_index=contract.item_mapping.raw_to_local,
            user_degrees=user_degrees,
            item_degrees=item_degrees,
            config=_config(),
            training_run_id="synthetic_lightgcn_primary",
        )


def test_metric_aggregation_lookup_and_row_hashing_are_deterministic() -> None:
    contract = _contract()
    user_degrees, item_degrees = _degree_maps(contract)
    user_embeddings = torch.ones((contract.user_mapping.count, 2), dtype=torch.float32)
    item_embeddings = torch.zeros((contract.item_mapping.count, 2), dtype=torch.float32)
    rows = score_candidate_rows(
        _pilot_rows("ranking") + _pilot_rows("reranking"),
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        user_index=contract.user_mapping.raw_to_local,
        item_index=contract.item_mapping.raw_to_local,
        user_degrees=user_degrees,
        item_degrees=item_degrees,
        config=_config(),
        training_run_id="synthetic_lightgcn_primary",
    )

    metrics = aggregate_metrics(rows, config=_config())
    lookup = metric_lookup(metrics)

    assert len(metrics) == 4
    assert lookup[("synthetic_ranking", "plan_e2_lightgcn_strict_connected_full")]["hr_at_5"] == 1.0
    assert rows_sha256(rows, ["candidate_set_id", "ranked_items"]) == rows_sha256(
        rows, ["candidate_set_id", "ranked_items"]
    )
    assert stable_json_sha256({"b": [2, 1], "a": "x"}) == stable_json_sha256({"a": "x", "b": [2, 1]})


def test_checkpoint_atomic_write_and_path_restriction(tmp_path: Path) -> None:
    model = LightGCNModel(num_users=1, num_items=1, embedding_dim=2, num_layers=0)
    payload = checkpoint_payload(model=model, config={"seed": 17}, metadata={"role": "primary"})
    approved = tmp_path / "approved"
    target = approved / "model_state.pt"

    write_checkpoint_atomic(payload, target, approved_root=approved)

    assert target.exists()
    with pytest.raises(LightGCNPilotSafetyError):
        write_checkpoint_atomic(payload, target, approved_root=approved)
    with pytest.raises(LightGCNPilotSafetyError):
        write_checkpoint_atomic(payload, tmp_path / "outside.pt", approved_root=approved)

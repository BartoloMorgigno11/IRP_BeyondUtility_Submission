from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.baselines.plan_e_lightgcn_adapter import (
    PLAN_E2_CANDIDATE_COUNT,
    CandidateValidationError,
    DenseAdjacencyForbiddenError,
    GraphValidationError,
    MappingValidationError,
    ScoreAlignmentError,
    align_scores_to_candidate_ids,
    audit_plan_e2_candidate_row,
    build_directed_bipartite_edges,
    build_graph_contract,
    build_local_index,
    coalesce_edges,
    filter_train_interactions,
    hash_int_arrays,
    json_array,
    load_raw_to_internal_mapping,
    normalize_symmetric_edges,
    parse_candidate_items,
    prepare_candidate_node_payload,
    reject_dense_adjacency_request,
    safe_raw_id,
    stable_unique_train_pairs,
    synthetic_sparse_forward_probe,
    validate_candidate_order,
    validate_raw_to_internal_pairs,
)


def test_raw_ids_remain_strings_and_leading_zeros_are_preserved() -> None:
    assert safe_raw_id("00123") == "00123"
    assert safe_raw_id(123) == "123"
    assert json.loads(json_array(["001", "A02"])) == ["001", "A02"]


def test_malformed_ids_fail() -> None:
    with pytest.raises(MappingValidationError):
        safe_raw_id("  ")
    with pytest.raises(MappingValidationError):
        safe_raw_id("nan")


def test_only_split_train_is_retained_and_empty_train_fails() -> None:
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "item_id": ["i1", "i2", "i3"],
            "split": ["train", "validation", "test"],
        }
    )
    train = filter_train_interactions(frame)
    assert train["user_id"].tolist() == ["u1"]
    with pytest.raises(GraphValidationError):
        filter_train_interactions(frame.loc[frame["split"] != "train"])


def test_mapping_validation_and_datamap_loader(tmp_path) -> None:
    mapping = validate_raw_to_internal_pairs([("001", "0"), ("abc", 2)], entity_type="item")
    assert mapping.raw_to_internal == {"001": 0, "abc": 2}
    with pytest.raises(MappingValidationError):
        validate_raw_to_internal_pairs([("u1", 0), ("u1", 1)], entity_type="user")
    with pytest.raises(MappingValidationError):
        validate_raw_to_internal_pairs([("u1", 0), ("u2", 0)], entity_type="user")

    path = tmp_path / "datamaps.json"
    path.write_text(
        json.dumps(
            {
                "user2id": {"u01": "0", "u02": "1"},
                "id2user": {"0": "u01", "1": "u02"},
                "item2id": {"i01": "3"},
                "id2item": {"3": "i01"},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_raw_to_internal_mapping(path, entity_type="user")
    assert loaded.raw_to_internal["u01"] == 0


def test_deterministic_user_and_item_local_mappings() -> None:
    users = build_local_index(["10", "2", "001"], entity_type="user")
    items = build_local_index(["b", "a"], entity_type="item")
    assert list(users.raw_to_local) == ["001", "2", "10"]
    assert items.raw_to_local == {"a": 0, "b": 1}


def test_unique_binary_edge_construction_and_candidate_files_not_used() -> None:
    train = pd.DataFrame(
        {
            "user_id": ["u2", "u1", "u1"],
            "item_id": ["i2", "i1", "i1"],
            "split": ["train", "train", "train"],
        }
    )
    pairs = stable_unique_train_pairs(train)
    assert pairs == (("u1", "i1"), ("u2", "i2"))
    graph = build_graph_contract(train)
    assert graph.duplicate_user_item_pair_count == 1
    assert len(graph.train_pairs) == 2


def test_directed_bipartite_edges_use_item_offset_and_no_self_loops() -> None:
    edges = build_directed_bipartite_edges([0, 1], [0, 1], num_users=2, num_items=2)
    assert set(zip(edges.rows.tolist(), edges.cols.tolist())) == {(0, 2), (2, 0), (1, 3), (3, 1)}
    assert not np.any(edges.rows == edges.cols)
    with pytest.raises(GraphValidationError):
        build_directed_bipartite_edges([0], [2], num_users=1, num_items=2)


def test_repeated_edge_behavior_matches_plan_d_sparse_coalesce() -> None:
    from src.baselines.plan_d_lightgcn import build_normalized_bipartite_graph

    edges = build_directed_bipartite_edges([0, 0], [0, 0], num_users=1, num_items=1)
    normalized = normalize_symmetric_edges(edges)
    plan_d = build_normalized_bipartite_graph(
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        num_users=1,
        num_items=1,
    )
    observed = {
        tuple(index): float(value)
        for index, value in zip(plan_d.indices().cpu().numpy().T.tolist(), plan_d.values().cpu().numpy().tolist())
    }
    audited = {
        (int(row), int(col)): float(value)
        for row, col, value in zip(normalized.rows.tolist(), normalized.cols.tolist(), normalized.values.tolist())
    }
    assert set(audited) == set(observed) == {(0, 1), (1, 0)}
    assert audited[(0, 1)] == pytest.approx(observed[(0, 1)])
    assert audited[(1, 0)] == pytest.approx(observed[(1, 0)])
    assert observed[(0, 1)] == pytest.approx(1.0)
    assert observed[(1, 0)] == pytest.approx(1.0)


def test_degrees_symmetric_normalization_and_stable_hashes() -> None:
    edges = build_directed_bipartite_edges([0, 0, 1], [0, 1, 1], num_users=2, num_items=2)
    normalized = normalize_symmetric_edges(edges)
    assert normalized.degrees.tolist() == [2.0, 1.0, 1.0, 2.0]
    weights = {
        (int(row), int(col)): float(value)
        for row, col, value in zip(normalized.rows.tolist(), normalized.cols.tolist(), normalized.values.tolist())
    }
    assert weights[(0, 2)] == pytest.approx(1 / math.sqrt(2))
    assert weights[(0, 3)] == pytest.approx(0.5)
    assert weights[(2, 0)] == pytest.approx(1 / math.sqrt(2))
    assert all(math.isfinite(value) for value in weights.values())
    assert hash_int_arrays(normalized.rows, normalized.cols) == hash_int_arrays(normalized.rows, normalized.cols)


def test_zero_degree_and_out_of_range_nodes_fail() -> None:
    with pytest.raises(GraphValidationError):
        normalize_symmetric_edges(build_directed_bipartite_edges([], [], num_users=1, num_items=1))


def test_coalesce_is_deterministic() -> None:
    rows, cols, values = coalesce_edges(
        np.asarray([1, 0, 1, 0]),
        np.asarray([2, 1, 2, 1]),
        np.asarray([0.25, 0.5, 0.75, 0.5], dtype=np.float32),
    )
    assert rows.tolist() == [0, 1]
    assert cols.tolist() == [1, 2]
    assert values.tolist() == pytest.approx([1.0, 1.0])


def test_lightgcn_model_shape_layer_zero_and_layer_aggregation() -> None:
    import torch

    from src.baselines.plan_d_lightgcn import LightGCNModel, build_normalized_bipartite_graph

    graph = build_normalized_bipartite_graph(
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        num_users=1,
        num_items=1,
    )
    model = LightGCNModel(num_users=1, num_items=1, embedding_dim=2, num_layers=1)
    with torch.no_grad():
        model.user_embedding.weight[:] = torch.tensor([[2.0, 0.0]])
        model.item_embedding.weight[:] = torch.tensor([[4.0, 0.0]])
    users, items = model.final_embeddings(graph)
    assert tuple(users.shape) == (1, 2)
    assert tuple(items.shape) == (1, 2)
    assert users[0, 0].item() == pytest.approx(3.0)
    assert items[0, 0].item() == pytest.approx(3.0)

    layer_zero = LightGCNModel(num_users=1, num_items=1, embedding_dim=1, num_layers=0)
    with torch.no_grad():
        layer_zero.user_embedding.weight[:] = torch.tensor([[7.0]])
        layer_zero.item_embedding.weight[:] = torch.tensor([[9.0]])
    zero_users, zero_items = layer_zero.final_embeddings(graph)
    assert zero_users[0, 0].item() == pytest.approx(7.0)
    assert zero_items[0, 0].item() == pytest.approx(9.0)


def test_candidate_order_membership_and_payload_are_preserved() -> None:
    user_mapping = build_local_index(["u1"], entity_type="user")
    item_mapping = build_local_index([f"i{x:02d}" for x in range(20)], entity_type="item")
    candidates = [f"i{x:02d}" for x in range(19, -1, -1)]
    payload = prepare_candidate_node_payload(
        user_id="u1",
        candidate_item_ids=candidates,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )
    validate_candidate_order(candidates, payload)
    assert list(payload.raw_candidate_item_ids) == candidates
    assert payload.item_global_node_indices[0] == 20


def test_candidate_validation_rejects_unknowns_duplicates_and_bad_lengths() -> None:
    with pytest.raises(CandidateValidationError):
        parse_candidate_items(json.dumps(["i1", "i1"] + [f"i{x}" for x in range(2, 20)]))
    with pytest.raises(CandidateValidationError):
        parse_candidate_items(json.dumps(["i1"]))
    user_mapping = build_local_index(["u1"], entity_type="user")
    item_mapping = build_local_index([f"i{x:02d}" for x in range(20)], entity_type="item")
    with pytest.raises(CandidateValidationError):
        prepare_candidate_node_payload(
            user_id="u2",
            candidate_item_ids=[f"i{x:02d}" for x in range(20)],
            user_mapping=user_mapping,
            item_mapping=item_mapping,
        )


def test_score_vector_alignment_contract_rejects_non_finite_and_length_mismatch() -> None:
    aligned = align_scores_to_candidate_ids(["i2", "i1"], [0.2, 0.9])
    assert [row["candidate_item_id"] for row in aligned] == ["i2", "i1"]
    with pytest.raises(ScoreAlignmentError):
        align_scores_to_candidate_ids(["i1"], [0.1, 0.2])
    with pytest.raises(ScoreAlignmentError):
        align_scores_to_candidate_ids(["i1"], [math.inf])


def test_row_compatibility_statuses_are_explicit() -> None:
    row, user_source, item_source, graph_users, graph_items, user_local, item_local = _row_fixture()
    audited = audit_plan_e2_candidate_row(
        row,
        user_mapping=user_source,
        item_mapping=item_source,
        graph_users=graph_users,
        graph_items=graph_items,
        user_local_index=user_local,
        item_local_index=item_local,
        normalization_verified=True,
        eligible_full=True,
        eligible_matched=True,
    )
    assert audited["compatibility_status"] == "lightgcn_graph_compatible"
    assert audited["lightgcn_graph_compatible"]

    disconnected_candidate = dict(row)
    candidates = json.loads(disconnected_candidate["candidate_items"])
    candidates[-1] = "i99"
    disconnected_candidate["candidate_items"] = json.dumps(candidates)
    item_source = validate_raw_to_internal_pairs(
        [(item, idx) for idx, item in enumerate([f"i{x:02d}" for x in range(20)] + ["i99"])],
        entity_type="item",
    )
    audited_disconnected = audit_plan_e2_candidate_row(
        disconnected_candidate,
        user_mapping=user_source,
        item_mapping=item_source,
        graph_users=graph_users,
        graph_items=graph_items,
        user_local_index=user_local,
        item_local_index=item_local,
        normalization_verified=True,
        eligible_full=True,
        eligible_matched=False,
    )
    assert audited_disconnected["compatibility_status"] == "graph_disconnected_candidate"
    assert audited_disconnected["contains_isolated_embedding_risk"]
    assert json.loads(audited_disconnected["graph_disconnected_candidate_item_ids"]) == ["i99"]


def test_mapping_user_positive_and_normalization_failures_are_reported() -> None:
    row, user_source, item_source, graph_users, graph_items, user_local, item_local = _row_fixture()
    missing = dict(row)
    missing["user_id"] = "missing"
    audited = audit_plan_e2_candidate_row(
        missing,
        user_mapping=user_source,
        item_mapping=item_source,
        graph_users=graph_users,
        graph_items=graph_items,
        user_local_index=user_local,
        item_local_index=item_local,
        normalization_verified=True,
        eligible_full=True,
        eligible_matched=True,
    )
    assert audited["compatibility_status"] == "mapping_incompatible"

    disconnected_positive = dict(row)
    disconnected_positive["positive_item_id"] = "i99"
    item_source = validate_raw_to_internal_pairs(
        [(item, idx) for idx, item in enumerate([f"i{x:02d}" for x in range(20)] + ["i99"])],
        entity_type="item",
    )
    audited_positive = audit_plan_e2_candidate_row(
        disconnected_positive,
        user_mapping=user_source,
        item_mapping=item_source,
        graph_users=graph_users,
        graph_items=graph_items,
        user_local_index=user_local,
        item_local_index=item_local,
        normalization_verified=True,
        eligible_full=True,
        eligible_matched=True,
    )
    assert audited_positive["compatibility_status"] == "positive_graph_disconnected"

    unverified = audit_plan_e2_candidate_row(
        row,
        user_mapping=user_source,
        item_mapping=item_source,
        graph_users=graph_users,
        graph_items=graph_items,
        user_local_index=user_local,
        item_local_index=item_local,
        normalization_verified=False,
        eligible_full=True,
        eligible_matched=True,
    )
    assert unverified["compatibility_status"] == "normalization_unverified"


def test_synthetic_cpu_sparse_probe_is_forward_only() -> None:
    result = synthetic_sparse_forward_probe(device="cpu")
    assert result["passed"]
    assert result["user_shape"] == (2, 3)
    assert result["item_shape"] == (2, 3)


def test_synthetic_cuda_sparse_probe_when_available() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    result = synthetic_sparse_forward_probe(device="cuda")
    assert result["passed"]


def test_dense_project_adjacency_guard_rejects_project_sized_requests() -> None:
    reject_dense_adjacency_request(total_nodes=10, project_graph=False)
    with pytest.raises(DenseAdjacencyForbiddenError):
        reject_dense_adjacency_request(total_nodes=10, project_graph=True)
    with pytest.raises(DenseAdjacencyForbiddenError):
        reject_dense_adjacency_request(total_nodes=1001, project_graph=False, max_synthetic_nodes=1000)


def test_serialization_and_deterministic_ordering_helpers() -> None:
    values = json.loads(json_array(["002", "001"]))
    assert values == ["002", "001"]
    assert parse_candidate_items(json.dumps([f"i{x:02d}" for x in range(20)]))[0] == "i00"


def _row_fixture():
    candidate_items = [f"i{x:02d}" for x in range(20)]
    row = {
        "candidate_set_id": "plan_e_ml1m_1000user_ranking_manifest",
        "source_path": "synthetic.csv",
        "dataset": "ML-1M",
        "task_type": "ranking",
        "user_id": "u1",
        "positive_item_id": "i00",
        "candidate_items": json.dumps(candidate_items),
    }
    user_source = validate_raw_to_internal_pairs([("u1", 0), ("u2", 1)], entity_type="user")
    item_source = validate_raw_to_internal_pairs([(item, idx) for idx, item in enumerate(candidate_items)], entity_type="item")
    graph_users = {"u1"}
    graph_items = set(candidate_items)
    user_local = {"u1": 0}
    item_local = {item: idx for idx, item in enumerate(candidate_items)}
    return row, user_source, item_source, graph_users, graph_items, user_local, item_local

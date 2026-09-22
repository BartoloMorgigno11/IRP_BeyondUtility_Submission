from __future__ import annotations

import json

import pandas as pd

from src.extension.e0_baselines import rank_mostpop_candidates, rank_random_candidates
from src.extension.e0_candidates import build_primary_candidates, validate_primary_candidates
from src.extension.e0_history import build_history_manifest
from src.extension.e0_parser import ParsedUserOutput, evaluate_fail_strict_outputs, parse_extension_top10_response
from src.extension.e0_prompts import render_extension_prompt
from src.extension.e0_protocol import CANDIDATE_COUNT, PARSER_CONTRACT_ID


def _primary_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog = pd.DataFrame(
        {
            "item_id": [f"item_{idx}" for idx in range(1, 80)],
            "title": [f"Catalog Item {idx}" for idx in range(1, 80)],
        }
    )
    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "item_id": "item_1", "split": "train", "timestamp": 1},
            {"user_id": "u1", "item_id": "item_2", "split": "train", "timestamp": 2},
            {"user_id": "u1", "item_id": "item_3", "split": "test", "timestamp": 3},
            {"user_id": "u2", "item_id": "item_4", "split": "train", "timestamp": 1},
            {"user_id": "u2", "item_id": "item_5", "split": "train", "timestamp": 2},
            {"user_id": "u2", "item_id": "item_6", "split": "test", "timestamp": 3},
        ]
    )
    ground_truth = pd.DataFrame(
        [
            {"user_id": "u1", "ground_truth_item_id": "item_3", "dataset": "foodcom", "split": "test"},
            {"user_id": "u2", "ground_truth_item_id": "item_6", "dataset": "foodcom", "split": "test"},
        ]
    )
    return interactions, catalog, ground_truth


def test_primary_candidate_builder_invariants_hashes_and_determinism() -> None:
    interactions, catalog, ground_truth = _primary_frames()
    first = build_primary_candidates(interactions, catalog, ground_truth, seed=123, dataset="foodcom")
    second = build_primary_candidates(interactions, catalog, ground_truth, seed=123, dataset="foodcom")

    pd.testing.assert_frame_equal(first, second)
    validate_primary_candidates(first, train_interactions=interactions[interactions["split"] == "train"])

    assert len(first) == 2 * CANDIDATE_COUNT
    assert set(first["candidate_position"]) == set(range(1, 21))
    assert first.groupby("user_id").size().eq(20).all()
    assert first.groupby("user_id")["is_ground_truth"].sum().eq(1).all()
    assert first.groupby("user_id")["candidate_item_id"].nunique().eq(20).all()
    assert first.groupby("user_id")["candidate_list_hash"].nunique().eq(1).all()
    assert first.groupby("user_id")["unordered_candidate_set_hash"].nunique().eq(1).all()

    train_seen = {
        user_id: set(group["item_id"])
        for user_id, group in interactions[interactions["split"] == "train"].groupby("user_id")
    }
    for user_id, group in first.groupby("user_id"):
        negatives = set(group.loc[~group["is_ground_truth"], "candidate_item_id"])
        assert not negatives.intersection(train_seen[user_id])


def test_candidate_constrained_random_and_mostpop_rank_only_candidates() -> None:
    interactions, catalog, ground_truth = _primary_frames()
    candidates = build_primary_candidates(interactions, catalog, ground_truth, seed=55, dataset="foodcom")

    random_a = rank_random_candidates(candidates, dataset="foodcom", seed=99, run_id="r", top_n=10)
    random_b = rank_random_candidates(candidates, dataset="foodcom", seed=99, run_id="r", top_n=10)
    pd.testing.assert_frame_equal(random_a, random_b)

    candidate_pairs = set(zip(candidates["user_id"], candidates["candidate_item_id"]))
    assert all((user_id, item_id) in candidate_pairs for user_id, item_id in zip(random_a["user_id"], random_a["recommended_item_id"]))
    assert random_a.groupby("user_id")["recommended_item_id"].nunique().eq(10).all()

    popular_item = candidates.loc[candidates["user_id"].eq("u1"), "candidate_item_id"].iloc[0]
    train = pd.concat(
        [
            interactions,
            pd.DataFrame(
                [
                    {"user_id": "other1", "item_id": popular_item, "split": "train"},
                    {"user_id": "other2", "item_id": popular_item, "split": "train"},
                    {"user_id": "other3", "item_id": popular_item, "split": "train"},
                ]
            ),
        ],
        ignore_index=True,
    )
    mostpop = rank_mostpop_candidates(candidates, train, dataset="foodcom", run_id="m", top_n=10)
    first_u1 = mostpop.loc[mostpop["user_id"].eq("u1")].sort_values("rank").iloc[0]
    assert first_u1["recommended_item_id"] == popular_item
    assert all((user_id, item_id) in candidate_pairs for user_id, item_id in zip(mostpop["user_id"], mostpop["recommended_item_id"]))


def test_strict_top10_parser_accepts_valid_and_rejects_invalid_shapes() -> None:
    candidate_items = [f"item_{idx}" for idx in range(1, 21)]
    valid = parse_extension_top10_response(
        '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}',
        candidate_items=candidate_items,
    )
    assert valid.parse_valid
    assert valid.output_items == tuple(candidate_items[:10])

    invalid_cases = {
        "nine": '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9]}',
        "eleven": '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10,11]}',
        "duplicate": '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,9]}',
        "zero": '{"ranked_candidate_indices":[0,2,3,4,5,6,7,8,9,10]}',
        "twenty_one": '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,21]}',
        "strings": '{"ranked_candidate_indices":["1",2,3,4,5,6,7,8,9,10]}',
        "item_ids": '{"ranked_candidate_indices":["B000",2,3,4,5,6,7,8,9,10]}',
        "prose": 'Here: {"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}',
        "malformed": '{"ranked_candidate_indices":[1,2,3',
        "extra": '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10],"status":"ok"}',
    }
    for raw in invalid_cases.values():
        assert not parse_extension_top10_response(raw, candidate_items=candidate_items).parse_valid


def test_fail_strict_metrics_keep_invalid_users_in_denominator() -> None:
    candidate_items = [f"item_{idx}" for idx in range(1, 21)]
    valid = parse_extension_top10_response(
        '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}',
        candidate_items=candidate_items,
    )
    invalid = parse_extension_top10_response(
        '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9]}',
        candidate_items=candidate_items,
    )
    outputs = [
        ParsedUserOutput("u1", "synthetic", "qwen", "BASE", "raw", tuple(candidate_items), valid),
        ParsedUserOutput("u2", "synthetic", "qwen", "BASE", "raw", tuple(candidate_items), invalid),
    ]
    ground_truth = pd.DataFrame(
        [
            {"user_id": "u1", "ground_truth_item_id": "item_1"},
            {"user_id": "u2", "ground_truth_item_id": "item_1"},
        ]
    )
    metrics = evaluate_fail_strict_outputs(outputs, ground_truth, run_id="strict")
    assert metrics["user_count"] == 2
    assert metrics["hit_rate"] == 0.5
    assert metrics["ndcg"] == 0.5
    assert metrics["strict_valid_count"] == 1
    assert metrics["strict_invalid_count"] == 1


def _history_frames(user_count: int = 205) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = []
    candidates = []
    catalog_rows = []
    for user_idx in range(user_count):
        user_id = f"user_{user_idx:03d}"
        for hist_idx in range(1, 21):
            item_id = f"hist_{user_id}_{hist_idx:02d}"
            interactions.append({"user_id": user_id, "item_id": item_id, "split": "train", "timestamp": hist_idx})
            catalog_rows.append({"item_id": item_id, "title": f"History {user_id} {hist_idx:02d}"})
        for cand_idx in range(1, 21):
            item_id = f"cand_{user_id}_{cand_idx:02d}"
            candidates.append(
                {
                    "user_id": user_id,
                    "candidate_item_id": item_id,
                    "candidate_position": cand_idx,
                    "is_ground_truth": cand_idx == 7,
                    "candidate_item_title": f"Candidate {user_id} {cand_idx:02d}",
                }
            )
            catalog_rows.append({"item_id": item_id, "title": f"Candidate {user_id} {cand_idx:02d}"})
    return pd.DataFrame(interactions), pd.DataFrame(catalog_rows), pd.DataFrame(candidates)


def test_history_manifest_base_only_lengths_pairing_and_deterministic_200_user_selection() -> None:
    interactions, catalog, candidates = _history_frames()
    first, summary = build_history_manifest(
        dataset="synthetic",
        interactions=interactions,
        catalog=catalog,
        candidates=candidates,
        cohort_size=200,
        seed=20260824,
        models=("qwen2:7b-instruct",),
        lengths=(0, 1, 2, 5, 10, 20),
    )
    second, _ = build_history_manifest(
        dataset="synthetic",
        interactions=interactions,
        catalog=catalog,
        candidates=candidates,
        cohort_size=200,
        seed=20260824,
        models=("qwen2:7b-instruct",),
        lengths=(0, 1, 2, 5, 10, 20),
    )
    pd.testing.assert_frame_equal(first, second)

    assert summary.loc[0, "selected_users"] == 200
    assert len(first) == 200 * 6
    assert set(first["history_length"]) == {0, 1, 2, 5, 10, 20}
    assert first["prompt_strategy"].eq("BASE").all()
    assert first["parser_contract"].eq(PARSER_CONTRACT_ID).all()

    for user_id, group in first.groupby("user_id"):
        assert group["candidate_list_hash"].nunique() == 1
        assert group["positive_item_id"].nunique() == 1
        by_length = {int(row.history_length): json.loads(row.history_item_ids) for row in group.itertuples(index=False)}
        assert by_length[0] == []
        assert by_length[1] == [f"hist_{user_id}_20"]
        assert by_length[2] == [f"hist_{user_id}_19", f"hist_{user_id}_20"]
        assert len(by_length[20]) == 20


def test_l0_base_prompt_has_no_history_leakage() -> None:
    candidate_items = [{"candidate_position": idx, "title": f"Candidate Title {idx}"} for idx in range(1, 21)]
    prompt = render_extension_prompt(
        prompt_strategy="BASE",
        dataset_label="synthetic",
        candidate_items=candidate_items,
        history_items=[],
        history_length=0,
    )
    forbidden = ["hist_item_1", "History Title 1", "most recent visible interaction", "profile", "preference summary"]
    for token in forbidden:
        assert token not in prompt.user_message
    assert "(no visible user history)" in prompt.user_message
    assert "ranked_candidate_indices" in prompt.user_message

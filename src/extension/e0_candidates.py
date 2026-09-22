"""Candidate-set construction and validation for Extension Phase E0."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schemas import validate_ground_truth, validate_interactions
from src.extension.e0_protocol import (
    CANDIDATE_COUNT,
    EVIDENCE_CLASS_ADAPTED,
    EXPERIMENT_ID,
    NEGATIVE_COUNT,
    POSITION_DOMAIN,
    item_list_hash,
    stable_int_seed,
    unordered_item_set_hash,
)


CANDIDATE_COLUMNS = [
    "experiment_id",
    "dataset",
    "user_id",
    "candidate_position",
    "candidate_item_id",
    "candidate_item_title",
    "is_ground_truth",
    "positive_item_id",
    "ground_truth_id",
    "positive_position",
    "candidate_list_hash",
    "unordered_candidate_set_hash",
    "seed",
    "negative_sampling_policy",
    "ordering_policy",
    "evidence_class",
    "source_mode",
]


def build_primary_candidates(
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    seed: int,
    max_users: int | None = None,
    dataset: str = "foodcom",
    source_mode: str = "materialized",
) -> pd.DataFrame:
    """Build a deterministic 20-candidate primary cohort.

    Negatives are sampled from the catalog after excluding each user's train
    history and held-out positive.
    """

    interactions = _normalize_interactions(interactions)
    catalog = _normalize_catalog(catalog)
    ground_truth = _normalize_ground_truth(ground_truth)

    train = interactions.loc[interactions["split"].str.lower().eq("train")].copy()
    catalog_ids = sorted(catalog["item_id"].drop_duplicates().tolist())
    title_map = dict(zip(catalog["item_id"], catalog["title"]))
    train_seen = {
        str(user_id): set(group["item_id"].astype(str).tolist())
        for user_id, group in train.groupby("user_id", sort=False)
    }

    users = sorted(ground_truth["user_id"].astype(str).tolist())
    if max_users is not None:
        users = _deterministic_user_sample(users, max_users, seed=seed, label=f"{dataset}_primary")

    rows: list[dict[str, Any]] = []
    for user_id in users:
        user_gt = ground_truth.loc[ground_truth["user_id"].eq(user_id), "ground_truth_item_id"].astype(str).tolist()
        if len(user_gt) != 1:
            raise ValueError(f"user {user_id} must have exactly one ground-truth item")
        positive_item_id = user_gt[0]
        if positive_item_id not in title_map:
            raise ValueError(f"positive item {positive_item_id} for user {user_id} is missing from catalog")

        excluded = set(train_seen.get(user_id, set()))
        excluded.add(positive_item_id)
        negative_pool = [item_id for item_id in catalog_ids if item_id not in excluded]
        if len(negative_pool) < NEGATIVE_COUNT:
            raise ValueError(f"user {user_id} has only {len(negative_pool)} available negatives")

        neg_rng = random.Random(stable_int_seed(seed, "negatives", user_id))
        negatives = neg_rng.sample(negative_pool, NEGATIVE_COUNT)
        candidate_ids = [positive_item_id, *negatives]
        order_rng = random.Random(stable_int_seed(seed, "candidate_order", user_id))
        order_rng.shuffle(candidate_ids)

        list_hash = item_list_hash(candidate_ids)
        set_hash = unordered_item_set_hash(candidate_ids)
        positive_position = candidate_ids.index(positive_item_id) + 1
        for position, item_id in enumerate(candidate_ids, start=1):
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "dataset": dataset,
                    "user_id": user_id,
                    "candidate_position": position,
                    "candidate_item_id": item_id,
                    "candidate_item_title": title_map[item_id],
                    "is_ground_truth": item_id == positive_item_id,
                    "positive_item_id": positive_item_id,
                    "ground_truth_id": positive_item_id,
                    "positive_position": positive_position,
                    "candidate_list_hash": list_hash,
                    "unordered_candidate_set_hash": set_hash,
                    "seed": seed,
                    "negative_sampling_policy": "catalog_minus_train_history_and_positive_user_seeded_without_replacement",
                    "ordering_policy": "deterministic_user_seeded_shuffle_one_based_positions",
                    "evidence_class": EVIDENCE_CLASS_ADAPTED,
                    "source_mode": source_mode,
                }
            )

    result = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    validate_primary_candidates(result, train_interactions=train)
    return result


def normalize_candidate_manifest(
    candidates: pd.DataFrame,
    *,
    dataset: str,
    seed: int = 0,
    allow_reduce_to_20: bool = True,
) -> pd.DataFrame:
    """Normalize existing 20-candidate manifests to extension columns."""

    frame = candidates.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "candidate_item_id" not in frame.columns and "item_id" in frame.columns:
        frame["candidate_item_id"] = frame["item_id"]
    if "is_ground_truth" not in frame.columns and "is_positive" in frame.columns:
        frame["is_ground_truth"] = frame["is_positive"]
    if "candidate_item_title" not in frame.columns:
        frame["candidate_item_title"] = frame["candidate_item_id"].astype(str)
    required = ("user_id", "candidate_item_id", "candidate_position", "is_ground_truth")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"candidate manifest missing columns: {', '.join(missing)}")

    frame["dataset"] = dataset
    frame["user_id"] = frame["user_id"].astype(str)
    frame["candidate_item_id"] = frame["candidate_item_id"].astype(str)
    frame["candidate_item_title"] = frame["candidate_item_title"].astype(str)
    frame["candidate_position"] = pd.to_numeric(frame["candidate_position"], errors="raise").astype(int)
    frame["is_ground_truth"] = _truthy(frame["is_ground_truth"])
    frame = frame.sort_values(["user_id", "candidate_position"], kind="mergesort").reset_index(drop=True)

    rows: list[pd.DataFrame] = []
    for user_id, group in frame.groupby("user_id", sort=False):
        group = group.sort_values("candidate_position", kind="mergesort").copy()
        positive_ids = group.loc[group["is_ground_truth"], "candidate_item_id"].tolist()
        if len(positive_ids) != 1:
            continue
        if len(group) != CANDIDATE_COUNT:
            if not allow_reduce_to_20 or len(group) < CANDIDATE_COUNT:
                continue
            positives = group.loc[group["is_ground_truth"]]
            negatives = group.loc[~group["is_ground_truth"]].copy()
            if len(negatives) < NEGATIVE_COUNT:
                continue
            neg_indices = list(negatives.index)
            rng = random.Random(stable_int_seed(seed, "reduce_manifest_negatives", dataset, user_id))
            selected_negative_indices = rng.sample(neg_indices, NEGATIVE_COUNT)
            group = pd.concat([positives, negatives.loc[selected_negative_indices]], axis=0)
            order = list(group.index)
            order_rng = random.Random(stable_int_seed(seed, "reduce_manifest_order", dataset, user_id))
            order_rng.shuffle(order)
            group = group.loc[order].copy()
        item_ids = group["candidate_item_id"].tolist()
        if len(set(item_ids)) != CANDIDATE_COUNT:
            continue
        group = group.copy()
        group["candidate_position"] = range(1, CANDIDATE_COUNT + 1)
        group["positive_item_id"] = positive_ids[0]
        group["ground_truth_id"] = positive_ids[0]
        group["positive_position"] = int(group.loc[group["is_ground_truth"], "candidate_position"].iloc[0])
        group["candidate_list_hash"] = item_list_hash(item_ids)
        group["unordered_candidate_set_hash"] = unordered_item_set_hash(item_ids)
        rows.append(group)

    if not rows:
        return pd.DataFrame(columns=[*CANDIDATE_COLUMNS, "source_manifest"])

    normalized = pd.concat(rows, ignore_index=True)
    for column, value in {
        "experiment_id": EXPERIMENT_ID,
        "seed": "",
        "negative_sampling_policy": "preexisting_reconstructed_manifest",
        "ordering_policy": "preexisting_manifest_order_renumbered_one_based",
        "evidence_class": EVIDENCE_CLASS_ADAPTED,
        "source_mode": "existing_reconstructed_manifest",
    }.items():
        normalized[column] = value
    return normalized.loc[:, [column for column in CANDIDATE_COLUMNS if column in normalized.columns]]


def attach_titles(candidates: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Attach item titles from a catalog where a manifest only carries IDs."""

    frame = candidates.copy()
    catalog = _normalize_catalog(catalog)
    title_map = dict(zip(catalog["item_id"], catalog["title"]))
    frame["candidate_item_title"] = [
        title_map.get(str(item_id), str(item_id)) for item_id in frame["candidate_item_id"]
    ]
    return frame


def validate_primary_candidates(
    candidates: pd.DataFrame,
    *,
    train_interactions: pd.DataFrame | None = None,
) -> None:
    """Validate the extension 20-candidate invariant set."""

    frame = candidates.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = ("user_id", "candidate_item_id", "candidate_position", "is_ground_truth")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"candidate dataframe missing columns: {', '.join(missing)}")
    frame["user_id"] = frame["user_id"].astype(str)
    frame["candidate_item_id"] = frame["candidate_item_id"].astype(str)
    frame["candidate_position"] = pd.to_numeric(frame["candidate_position"], errors="raise").astype(int)
    frame["is_ground_truth"] = _truthy(frame["is_ground_truth"])

    seen_by_user: dict[str, set[str]] = {}
    if train_interactions is not None and not train_interactions.empty:
        train = _normalize_interactions(train_interactions)
        seen_by_user = {
            str(user_id): set(group["item_id"].astype(str).tolist())
            for user_id, group in train.groupby("user_id", sort=False)
        }

    for user_id, group in frame.groupby("user_id", sort=False):
        if len(group) != CANDIDATE_COUNT:
            raise ValueError(f"user {user_id} has {len(group)} candidates, expected {CANDIDATE_COUNT}")
        positions = sorted(group["candidate_position"].tolist())
        if positions != list(POSITION_DOMAIN):
            raise ValueError(f"user {user_id} candidate positions are not 1..20")
        if group["candidate_item_id"].nunique() != CANDIDATE_COUNT:
            raise ValueError(f"user {user_id} has duplicate candidate items")
        positives = group.loc[group["is_ground_truth"], "candidate_item_id"].tolist()
        if len(positives) != 1:
            raise ValueError(f"user {user_id} has {len(positives)} positives, expected 1")
        if seen_by_user:
            seen = seen_by_user.get(str(user_id), set())
            negatives = set(group.loc[~group["is_ground_truth"], "candidate_item_id"].tolist())
            overlap = seen.intersection(negatives)
            if overlap:
                raise ValueError(f"user {user_id} has seen-item negatives: {sorted(overlap)[:3]}")


def candidate_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    frame["is_ground_truth"] = _truthy(frame["is_ground_truth"])
    per_user = frame.groupby("user_id").size()
    positives = frame.groupby("user_id")["is_ground_truth"].sum()
    return pd.DataFrame(
        [
            {
                "rows": int(len(frame)),
                "users": int(per_user.size),
                "min_candidates_per_user": int(per_user.min()) if not per_user.empty else 0,
                "median_candidates_per_user": float(per_user.median()) if not per_user.empty else 0.0,
                "max_candidates_per_user": int(per_user.max()) if not per_user.empty else 0,
                "users_with_20_candidates": int((per_user == CANDIDATE_COUNT).sum()),
                "users_with_exactly_one_positive": int((positives == 1).sum()),
                "unique_candidate_items": int(frame["candidate_item_id"].nunique()),
            }
        ]
    )


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _normalize_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    frame = validate_interactions(interactions)
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    return frame


def _normalize_ground_truth(ground_truth: pd.DataFrame) -> pd.DataFrame:
    frame = validate_ground_truth(ground_truth)
    frame["user_id"] = frame["user_id"].astype(str)
    frame["ground_truth_item_id"] = frame["ground_truth_item_id"].astype(str)
    counts = frame.groupby("user_id").size()
    bad_users = counts.loc[counts != 1]
    if not bad_users.empty:
        raise ValueError("ground truth must have exactly one row per user")
    return frame


def _normalize_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    frame = catalog.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "item_id" not in frame.columns:
        raise ValueError("catalog must contain item_id")
    if "title" not in frame.columns:
        frame["title"] = frame["item_id"]
    frame["item_id"] = frame["item_id"].astype(str).str.strip()
    frame["title"] = frame["title"].astype(str).str.strip()
    frame = frame.loc[frame["item_id"].ne("") & frame["title"].ne("")]
    return frame.drop_duplicates("item_id", keep="first")


def _deterministic_user_sample(users: Iterable[str], count: int, *, seed: int, label: str) -> list[str]:
    ordered = sorted(str(user) for user in users)
    ordered = sorted(ordered, key=lambda user: (stable_int_seed(seed, label, user), user))
    return ordered[:count]


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})

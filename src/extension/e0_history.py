"""History/cold-start dry-run manifest builders for Extension E0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.extension.e0_candidates import attach_titles, normalize_candidate_manifest, validate_primary_candidates
from src.extension.e0_prompts import render_extension_prompt
from src.extension.e0_protocol import (
    CANDIDATE_COUNT,
    EVIDENCE_CLASS_ADAPTED,
    EXPERIMENT_ID,
    FUTURE_MODELS,
    HISTORY_LENGTHS,
    HISTORY_PROMPT_STRATEGY,
    METRIC_K,
    PARSER_CONTRACT_ID,
    REQUESTED_OUTPUT_LENGTH,
    item_list_hash,
    semantic_hash_for_row,
    stable_int_seed,
    stable_json,
)


HISTORY_MANIFEST_COLUMNS = [
    "experiment_id",
    "dataset",
    "user_id",
    "model_name",
    "prompt_strategy",
    "task",
    "history_length",
    "candidate_count",
    "candidate_position_convention",
    "candidate_list",
    "candidate_list_hash",
    "unordered_candidate_set_hash",
    "positive_item_id",
    "positive_position",
    "ground_truth_id",
    "history_item_ids",
    "history_hash",
    "requested_output_length",
    "metric_k",
    "parser_contract",
    "prompt_contract",
    "prompt_hash",
    "seed",
    "output_destination",
    "evidence_class",
    "semantic_hash",
]


def build_history_manifest(
    *,
    dataset: str,
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    candidates: pd.DataFrame,
    cohort_size: int,
    seed: int,
    models: Sequence[str] = FUTURE_MODELS,
    lengths: Sequence[int] = HISTORY_LENGTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a BASE-only paired history manifest without model calls."""

    train = _train_interactions(interactions)
    catalog_titles = _title_map(catalog)
    train_eligible_users = _train_eligible_users(train, min_history=max(lengths))
    selected_users: list[str] = []
    normalized_candidates = pd.DataFrame()
    candidate_multiplier = 1
    while len(selected_users) < cohort_size and candidate_multiplier <= 32:
        potential_users = _select_users(
            train_eligible_users,
            min(len(train_eligible_users), cohort_size * candidate_multiplier),
            seed=seed,
            dataset=dataset,
        )
        candidate_subset = _filter_candidate_users(candidates, potential_users)
        normalized_candidates = normalize_candidate_manifest(candidate_subset, dataset=dataset, seed=seed)
        normalized_candidates = attach_titles(normalized_candidates, catalog)
        validate_primary_candidates(normalized_candidates)
        eligible_users = _eligible_users(train_eligible_users, normalized_candidates)
        selected_users = _select_users(eligible_users, cohort_size, seed=seed, dataset=dataset)
        candidate_multiplier *= 2
    if len(selected_users) < cohort_size:
        raise ValueError(f"{dataset} has {len(selected_users)} eligible history users, expected {cohort_size}")

    candidate_groups = {
        str(user_id): group.sort_values("candidate_position", kind="mergesort").copy()
        for user_id, group in normalized_candidates.groupby("user_id", sort=False)
        if str(user_id) in set(selected_users)
    }
    history_groups = {
        str(user_id): group.copy()
        for user_id, group in train.groupby("user_id", sort=False)
        if str(user_id) in set(selected_users)
    }

    rows: list[dict[str, Any]] = []
    for user_id in selected_users:
        candidates_for_user = _candidate_payload(candidate_groups[user_id])
        candidate_list_hash = item_list_hash([item["item_id"] for item in candidates_for_user])
        unordered_hash = str(candidate_groups[user_id]["unordered_candidate_set_hash"].iloc[0])
        positive_item_id = str(candidate_groups[user_id]["positive_item_id"].iloc[0])
        positive_position = int(candidate_groups[user_id]["positive_position"].iloc[0])
        history = history_groups[user_id].sort_values("__history_order", kind="mergesort")
        for length in lengths:
            visible = history.tail(length) if length else history.iloc[0:0]
            history_items = _history_payload(visible, catalog_titles)
            history_ids = [item["item_id"] for item in history_items]
            history_hash = item_list_hash(history_ids)
            prompt = render_extension_prompt(
                prompt_strategy=HISTORY_PROMPT_STRATEGY,
                dataset_label=dataset,
                candidate_items=[{"candidate_position": item["position"], "title": item["title"]} for item in candidates_for_user],
                history_items=history_items,
                history_length=int(length),
            )
            for model_name in models:
                row = {
                    "experiment_id": EXPERIMENT_ID,
                    "dataset": dataset,
                    "user_id": user_id,
                    "model_name": model_name,
                    "prompt_strategy": HISTORY_PROMPT_STRATEGY,
                    "task": "ranking",
                    "history_length": int(length),
                    "candidate_count": CANDIDATE_COUNT,
                    "candidate_position_convention": "one_based_1_to_20",
                    "candidate_list": stable_json(candidates_for_user),
                    "candidate_list_hash": candidate_list_hash,
                    "unordered_candidate_set_hash": unordered_hash,
                    "positive_item_id": positive_item_id,
                    "positive_position": positive_position,
                    "ground_truth_id": positive_item_id,
                    "history_item_ids": stable_json(history_ids),
                    "history_hash": history_hash,
                    "requested_output_length": REQUESTED_OUTPUT_LENGTH,
                    "metric_k": METRIC_K,
                    "parser_contract": PARSER_CONTRACT_ID,
                    "prompt_contract": prompt.prompt_contract,
                    "prompt_hash": prompt.prompt_hash,
                    "seed": seed,
                    "output_destination": str(Path("results/extension/llm_outputs") / EXPERIMENT_ID / dataset / model_name.replace(":", "_") / f"{user_id}_L{length}.json"),
                    "evidence_class": EVIDENCE_CLASS_ADAPTED,
                }
                row["semantic_hash"] = semantic_hash_for_row(row, exclude=("semantic_hash",))
                rows.append(row)

    manifest = pd.DataFrame(rows, columns=HISTORY_MANIFEST_COLUMNS)
    summary = _history_summary(dataset, manifest, selected_users, train_eligible_users)
    return manifest, summary


def _train_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    frame = interactions.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    split = frame["split"].astype(str).str.lower() if "split" in frame.columns else pd.Series("train", index=frame.index)
    frame = frame.loc[split.eq("train")].copy()
    frame["__input_order"] = range(len(frame))
    if "timestamp" in frame.columns:
        frame["__timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame = frame.sort_values(["user_id", "__timestamp", "__input_order"], kind="mergesort")
    else:
        frame = frame.sort_values(["user_id", "__input_order"], kind="mergesort")
    frame["__history_order"] = frame.groupby("user_id", sort=False).cumcount() + 1
    return frame


def _title_map(catalog: pd.DataFrame) -> dict[str, str]:
    frame = catalog.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "title" not in frame.columns:
        frame["title"] = frame["item_id"]
    frame["item_id"] = frame["item_id"].astype(str)
    frame["title"] = frame["title"].astype(str)
    return dict(zip(frame["item_id"], frame["title"]))


def _train_eligible_users(train: pd.DataFrame, *, min_history: int) -> list[str]:
    counts = train.groupby("user_id").size()
    return sorted(str(user_id) for user_id, count in counts.items() if count >= min_history)


def _eligible_users(train_eligible_users: Sequence[str], candidates: pd.DataFrame) -> list[str]:
    candidate_users = set(candidates["user_id"].astype(str).unique())
    return sorted(str(user_id) for user_id in train_eligible_users if str(user_id) in candidate_users)


def _filter_candidate_users(candidates: pd.DataFrame, users: Sequence[str]) -> pd.DataFrame:
    frame = candidates.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    user_set = set(str(user) for user in users)
    return frame.loc[frame["user_id"].astype(str).isin(user_set)].copy()


def _select_users(users: Sequence[str], count: int, *, seed: int, dataset: str) -> list[str]:
    ordered = sorted(users, key=lambda user: (stable_int_seed(seed, "history_cohort", dataset, user), user))
    return ordered[:count]


def _candidate_payload(group: pd.DataFrame) -> list[dict[str, Any]]:
    payload = []
    for _, row in group.sort_values("candidate_position", kind="mergesort").iterrows():
        payload.append(
            {
                "position": int(row["candidate_position"]),
                "item_id": str(row["candidate_item_id"]),
                "title": str(row["candidate_item_title"]),
            }
        )
    return payload


def _history_payload(history: pd.DataFrame, title_map: Mapping[str, str]) -> list[dict[str, str]]:
    payload = []
    for _, row in history.iterrows():
        item_id = str(row["item_id"])
        payload.append({"item_id": item_id, "title": str(title_map.get(item_id, item_id))})
    return payload


def _history_summary(dataset: str, manifest: pd.DataFrame, selected_users: Sequence[str], eligible_users: Sequence[str]) -> pd.DataFrame:
    per_length = manifest.groupby("history_length")["user_id"].nunique().to_dict()
    candidate_hash_counts = manifest.groupby("user_id")["candidate_list_hash"].nunique()
    positive_counts = manifest.groupby("user_id")["positive_item_id"].nunique()
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "eligible_users_ge20": len(eligible_users),
                "selected_users": len(selected_users),
                "manifest_rows": len(manifest),
                "models": manifest["model_name"].nunique() if not manifest.empty else 0,
                "history_lengths": ";".join(str(length) for length in sorted(per_length)),
                "users_per_history_length": stable_json({str(k): int(v) for k, v in per_length.items()}),
                "users_with_constant_candidate_hash": int((candidate_hash_counts == 1).sum()),
                "users_with_constant_positive": int((positive_counts == 1).sum()),
            }
        ]
    )

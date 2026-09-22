"""Candidate-constrained offline baselines for the extension protocol."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.extension.e0_candidates import validate_primary_candidates
from src.extension.e0_protocol import REQUESTED_OUTPUT_LENGTH, stable_int_seed


RECOMMENDATION_COLUMNS = [
    "user_id",
    "model_name",
    "dataset",
    "rank",
    "recommended_item_id",
    "run_id",
    "candidate_position",
    "score",
    "train_popularity",
]


def rank_random_candidates(
    candidates: pd.DataFrame,
    *,
    dataset: str,
    seed: int,
    run_id: str,
    top_n: int = REQUESTED_OUTPUT_LENGTH,
) -> pd.DataFrame:
    """Return a deterministic random ranking over each user's candidates."""

    frame = _candidate_frame(candidates)
    rows: list[dict[str, Any]] = []
    for user_id, group in frame.groupby("user_id", sort=False):
        ranked = group.copy()
        ranked["score"] = [
            stable_int_seed(seed, "random_baseline", user_id, item_id) / float(16**16)
            for item_id in ranked["candidate_item_id"]
        ]
        ranked = ranked.sort_values(["score", "candidate_item_id"], ascending=[False, True], kind="mergesort")
        for rank, (_, item) in enumerate(ranked.head(top_n).iterrows(), start=1):
            rows.append(_row(user_id, "random_candidate", dataset, rank, item, run_id, score=float(item["score"])))
    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


def rank_mostpop_candidates(
    candidates: pd.DataFrame,
    train_interactions: pd.DataFrame,
    *,
    dataset: str,
    run_id: str,
    top_n: int = REQUESTED_OUTPUT_LENGTH,
) -> pd.DataFrame:
    """Rank only each user's candidates by global train-set popularity."""

    frame = _candidate_frame(candidates)
    train = train_interactions.copy()
    train.columns = [str(column).strip().lower() for column in train.columns]
    split = train["split"].astype(str).str.lower() if "split" in train.columns else pd.Series("train", index=train.index)
    train = train.loc[split.eq("train")].copy()
    train["item_id"] = train["item_id"].astype(str)
    popularity = train.groupby("item_id", sort=False).size().to_dict()

    rows: list[dict[str, Any]] = []
    for user_id, group in frame.groupby("user_id", sort=False):
        ranked = group.copy()
        ranked["train_popularity"] = [int(popularity.get(item_id, 0)) for item_id in ranked["candidate_item_id"]]
        ranked = ranked.sort_values(
            ["train_popularity", "candidate_item_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        for rank, (_, item) in enumerate(ranked.head(top_n).iterrows(), start=1):
            rows.append(
                _row(
                    user_id,
                    "mostpop_candidate",
                    dataset,
                    rank,
                    item,
                    run_id,
                    train_popularity=int(item["train_popularity"]),
                )
            )
    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


def _candidate_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    validate_primary_candidates(candidates)
    frame = candidates.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame["user_id"] = frame["user_id"].astype(str)
    frame["candidate_item_id"] = frame["candidate_item_id"].astype(str)
    frame["candidate_position"] = pd.to_numeric(frame["candidate_position"], errors="raise").astype(int)
    return frame.sort_values(["user_id", "candidate_position"], kind="mergesort").reset_index(drop=True)


def _row(
    user_id: Any,
    model_name: str,
    dataset: str,
    rank: int,
    item: pd.Series,
    run_id: str,
    *,
    score: float | None = None,
    train_popularity: int | None = None,
) -> dict[str, Any]:
    return {
        "user_id": str(user_id),
        "model_name": model_name,
        "dataset": dataset,
        "rank": int(rank),
        "recommended_item_id": str(item["candidate_item_id"]),
        "run_id": run_id,
        "candidate_position": int(item["candidate_position"]),
        "score": "" if score is None else score,
        "train_popularity": "" if train_popularity is None else int(train_popularity),
    }

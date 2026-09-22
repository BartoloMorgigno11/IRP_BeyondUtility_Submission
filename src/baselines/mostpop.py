"""Most-popular baseline recommendations.

Run from the repository root:

    python src/baselines/mostpop.py --config configs/beauty.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import read_table
from src.data.schemas import validate_interactions, validate_recommendations
from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path


LOGGER = logging.getLogger("mostpop")
MODEL_NAME = "mostpop"

RECOMMENDATION_COLUMNS = [
    "user_id",
    "model_name",
    "dataset",
    "rank",
    "recommended_item_id",
    "run_id",
    "train_popularity",
    "config_path",
    "interactions_path",
    "timestamp",
]

SUMMARY_COLUMNS = [
    "dataset",
    "model_name",
    "run_id",
    "config_path",
    "interactions_path",
    "top_n",
    "train_interactions",
    "test_interactions",
    "train_users",
    "test_users",
    "unique_train_items",
    "recommendations_written",
    "users_with_recommendations",
    "users_without_recommendations",
    "users_with_short_recommendations",
    "output_path",
    "timestamp",
]


class MostPopError(ValueError):
    """Raised when MostPop cannot run with the configured inputs."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate global MostPop recommendations from train interactions."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Project-relative YAML config path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of recommendations per test user. Defaults to max experiment.k_values.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id stored in output rows.",
    )
    return parser.parse_args(argv)


def run_mostpop(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    top_n: int | None = None,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Generate MostPop recommendations from configured train interactions."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_api_calls_disabled(config)

    dataset = str(config["dataset"]["name"])
    top_n = _resolve_top_n(config, top_n)
    run_id = run_id or f"{dataset}_{MODEL_NAME}_top{top_n}"
    timestamp = datetime.now(timezone.utc).isoformat()
    config_label = str(config_path)

    interactions_path = _resolve_interactions_path(config, root)
    interactions = validate_interactions(read_table(interactions_path))
    train, test = _split_train_test(interactions)

    popularity = compute_global_popularity(train)
    test_users = _ordered_unique(test["user_id"])
    seen_by_user = _seen_items_by_user(train)
    recommendations = recommend_mostpop(
        popularity,
        test_users,
        seen_by_user=seen_by_user,
        top_n=top_n,
        dataset=dataset,
        run_id=run_id,
        timestamp=timestamp,
        config_path=config_label,
        interactions_path=str(interactions_path.relative_to(root)),
    )

    raw_outputs_dir = _configured_output_dir(
        config,
        root,
        "raw_outputs_dir",
        "results/raw_outputs",
    )
    tables_dir = _configured_output_dir(config, root, "tables_dir", "results/tables")
    raw_outputs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    recommendations_path = raw_outputs_dir / f"{dataset}_mostpop_recommendations.csv"
    summary_path = tables_dir / f"{dataset}_mostpop_summary.csv"
    summary = _build_summary(
        dataset=dataset,
        run_id=run_id,
        config_path=config_label,
        interactions_path=str(interactions_path.relative_to(root)),
        top_n=top_n,
        train=train,
        test=test,
        popularity=popularity,
        recommendations=recommendations,
        output_path=recommendations_path.relative_to(root),
        timestamp=timestamp,
    )

    recommendations.to_csv(recommendations_path, index=False)
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Saved MostPop recommendations to %s", recommendations_path)
    LOGGER.info("Saved MostPop summary to %s", summary_path)
    return recommendations, summary, recommendations_path, summary_path


def compute_global_popularity(train_interactions: pd.DataFrame) -> pd.DataFrame:
    """Count global item popularity from train interactions only."""

    train = validate_interactions(train_interactions)
    if train.empty:
        raise MostPopError("No train interactions are available to compute popularity.")

    popularity = (
        train.groupby("item_id", sort=False)
        .size()
        .reset_index(name="train_popularity")
    )
    popularity["__item_sort"] = popularity["item_id"].astype(str)
    popularity = popularity.sort_values(
        ["train_popularity", "__item_sort"],
        ascending=[False, True],
        kind="mergesort",
    )
    return popularity.drop(columns=["__item_sort"]).reset_index(drop=True)


def recommend_mostpop(
    popularity: pd.DataFrame,
    test_users: list[Any],
    *,
    seen_by_user: dict[Any, set[Any]],
    top_n: int,
    dataset: str,
    run_id: str,
    timestamp: str,
    config_path: str,
    interactions_path: str,
) -> pd.DataFrame:
    """Recommend top popular unseen items for each test user."""

    if top_n < 1:
        raise MostPopError("top_n must be a positive integer.")
    if popularity.empty:
        raise MostPopError("Popularity table is empty.")

    rows: list[dict[str, Any]] = []
    for user_id in test_users:
        seen_items = seen_by_user.get(user_id, set())
        rank = 1
        for _, item in popularity.iterrows():
            item_id = item["item_id"]
            if item_id in seen_items:
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "model_name": MODEL_NAME,
                    "dataset": dataset,
                    "rank": rank,
                    "recommended_item_id": item_id,
                    "run_id": run_id,
                    "train_popularity": int(item["train_popularity"]),
                    "config_path": config_path,
                    "interactions_path": interactions_path,
                    "timestamp": timestamp,
                }
            )
            rank += 1
            if rank > top_n:
                break

    recommendations = pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)
    if not recommendations.empty:
        validate_recommendations(recommendations)
    return recommendations


def _resolve_interactions_path(config: dict[str, Any], project_root: Path) -> Path:
    configured = (
        config.get("paths", {}).get("interactions_path")
        or config.get("dataset", {}).get("interactions_path")
    )
    if configured is None:
        raise MostPopError(
            "Dataset interactions_path is not configured. Set paths.interactions_path "
            "to a project-relative CSV/TSV file with user_id, item_id, and split columns."
        )

    resolved = resolve_project_path(configured, project_root)
    if resolved is None or not resolved.is_file():
        raise MostPopError(f"Configured interactions_path does not exist: {configured}")
    return resolved


def _split_train_test(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = interactions["split"].astype("string").str.strip().str.lower()
    train = interactions.loc[split == "train"].copy()
    test = interactions.loc[split == "test"].copy()
    if train.empty:
        raise MostPopError("No train interactions found; MostPop must not use test data for popularity.")
    if test.empty:
        raise MostPopError("No test interactions found; no test users are available for recommendations.")
    return train, test


def _seen_items_by_user(train: pd.DataFrame) -> dict[Any, set[Any]]:
    return {
        user_id: set(group["item_id"].tolist())
        for user_id, group in train.groupby("user_id", sort=False)
    }


def _ordered_unique(series: pd.Series) -> list[Any]:
    return series.drop_duplicates().tolist()


def _resolve_top_n(config: dict[str, Any], top_n: int | None) -> int:
    if top_n is not None:
        if top_n < 1:
            raise MostPopError("top_n must be a positive integer.")
        return top_n

    k_values = config.get("experiment", {}).get("k_values", [])
    if not k_values:
        raise MostPopError("experiment.k_values must contain at least one cutoff.")
    return max(int(k) for k in k_values)


def _configured_output_dir(
    config: dict[str, Any],
    project_root: Path,
    key: str,
    fallback: str,
) -> Path:
    value = config.get("output", {}).get(key, fallback)
    resolved = resolve_project_path(value, project_root)
    return resolved if resolved is not None else project_root / fallback


def _build_summary(
    *,
    dataset: str,
    run_id: str,
    config_path: str,
    interactions_path: str,
    top_n: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    popularity: pd.DataFrame,
    recommendations: pd.DataFrame,
    output_path: Path,
    timestamp: str,
) -> pd.DataFrame:
    per_user_counts = (
        recommendations.groupby("user_id").size() if not recommendations.empty else pd.Series(dtype="int64")
    )
    test_users = _ordered_unique(test["user_id"])
    users_with_recommendations = int((per_user_counts > 0).sum())
    users_without_recommendations = len(test_users) - users_with_recommendations
    users_with_short_recommendations = sum(
        int(per_user_counts.get(user_id, 0)) < top_n for user_id in test_users
    )

    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model_name": MODEL_NAME,
                "run_id": run_id,
                "config_path": config_path,
                "interactions_path": interactions_path,
                "top_n": top_n,
                "train_interactions": int(len(train)),
                "test_interactions": int(len(test)),
                "train_users": int(train["user_id"].nunique()),
                "test_users": int(len(test_users)),
                "unique_train_items": int(len(popularity)),
                "recommendations_written": int(len(recommendations)),
                "users_with_recommendations": users_with_recommendations,
                "users_without_recommendations": int(users_without_recommendations),
                "users_with_short_recommendations": int(users_with_short_recommendations),
                "output_path": str(output_path),
                "timestamp": timestamp,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def _ensure_api_calls_disabled(config: dict[str, Any]) -> None:
    if config.get("api", {}).get("allow_api_calls"):
        raise ConfigError("api.allow_api_calls must be false; MostPop never calls APIs.")
    if config.get("experiment", {}).get("allow_api_calls"):
        raise ConfigError(
            "experiment.allow_api_calls must be false; MostPop never calls APIs."
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)

    try:
        recommendations, summary, recommendations_path, summary_path = run_mostpop(
            args.config,
            top_n=args.top_n,
            run_id=args.run_id,
        )
    except (ConfigError, MostPopError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2

    LOGGER.info("Generated %s MostPop recommendation rows.", len(recommendations))
    print(f"recommendations={recommendations_path}")
    print(f"summary={summary_path}")
    print(f"test_users={int(summary.loc[0, 'test_users'])}")
    print(f"recommendations_written={int(summary.loc[0, 'recommendations_written'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

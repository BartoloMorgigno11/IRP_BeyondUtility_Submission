"""Prepare canonical MovieLens-1M files for the Plan B reconstruction track.

Run from the repository root:

    python src/data/prepare_ml1m.py --config configs/ml1m.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import count_nonempty_lines
from src.data.schemas import validate_candidates, validate_ground_truth, validate_interactions
from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path


LOGGER = logging.getLogger("prepare_ml1m")

RATINGS_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
MOVIES_COLUMNS = ["item_id", "title", "genres"]
INTERACTIONS_COLUMNS = ["user_id", "item_id", "split", "timestamp", "rating"]
GROUND_TRUTH_COLUMNS = [
    "user_id",
    "ground_truth_item_id",
    "dataset",
    "split",
    "timestamp",
    "rating",
]
CANDIDATES_COLUMNS = [
    "user_id",
    "candidate_item_id",
    "candidate_position",
    "is_ground_truth",
    "candidate_item_title",
]
CATALOG_COLUMNS = ["item_id", "title", "genres", "normalized_text"]

DEFAULT_NEGATIVE_SAMPLES_PER_USER = 99
DEFAULT_MIN_INTERACTIONS = 2


@dataclass(frozen=True)
class PreparationResult:
    """Result object returned by the ML-1M preparation runner."""

    processed: bool
    summary: pd.DataFrame
    summary_path: Path
    output_paths: dict[str, Path]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare canonical MovieLens-1M files for Plan B."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Project-relative YAML config path.",
    )
    return parser.parse_args(argv)


def prepare_ml1m(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> PreparationResult:
    """Prepare ML-1M canonical CSVs or write a clear diagnostic summary.

    Required raw files are ``ratings.dat`` and ``movies.dat`` under the
    configured raw directory. ``users.dat`` is optional and is used only for a
    diagnostic count when present. Missing required raw files produce a
    diagnostic summary under ``results/tables/`` and do not create processed
    dataset files.
    """

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_api_calls_disabled(config)

    dataset = str(config["dataset"]["name"])
    seed = int(config["experiment"]["seed"])
    preprocessing = config.get("preprocessing", {})
    negative_samples_per_user = int(
        preprocessing.get(
            "negative_samples_per_user",
            DEFAULT_NEGATIVE_SAMPLES_PER_USER,
        )
    )
    min_interactions = int(
        preprocessing.get("min_interactions", DEFAULT_MIN_INTERACTIONS)
    )
    _validate_preprocessing_settings(
        negative_samples_per_user=negative_samples_per_user,
        min_interactions=min_interactions,
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    raw_dir = _configured_path(config, root, "raw_dir", f"data/raw/{dataset}")
    processed_dir = _configured_path(
        config,
        root,
        "processed_dir",
        f"data/processed/{dataset}",
    )
    tables_dir = _configured_output_dir(config, root, "tables_dir", "results/tables")
    summary_path = _configured_summary_path(config, root, dataset, tables_dir)
    output_paths = _canonical_output_paths(config, root, processed_dir)

    raw_paths = {
        "ratings": raw_dir / "ratings.dat",
        "movies": raw_dir / "movies.dat",
        "users": raw_dir / "users.dat",
    }
    missing_required = [
        _display_path(path, root)
        for key, path in raw_paths.items()
        if key in {"ratings", "movies"} and not path.is_file()
    ]
    if missing_required:
        summary = _summary_frame(
            dataset=dataset,
            status="missing_raw",
            message="Required MovieLens-1M raw file(s) are missing.",
            config_path=str(config_path),
            root=root,
            timestamp=timestamp,
            raw_paths=raw_paths,
            output_paths=output_paths,
            missing_files=missing_required,
            negative_samples_per_user=negative_samples_per_user,
            min_interactions=min_interactions,
            seed=seed,
            users_dat_present=raw_paths["users"].is_file(),
            users_dat_rows=(
                count_nonempty_lines(raw_paths["users"])
                if raw_paths["users"].is_file()
                else 0
            ),
        )
        _write_summary(summary, summary_path)
        LOGGER.warning("ML-1M preparation skipped; missing raw files: %s", ", ".join(missing_required))
        return PreparationResult(False, summary, summary_path, output_paths)

    ratings = read_ratings_dat(raw_paths["ratings"])
    movies = read_movies_dat(raw_paths["movies"])
    catalog = build_catalog(movies)
    users_dat_rows = (
        count_nonempty_lines(raw_paths["users"]) if raw_paths["users"].is_file() else 0
    )

    try:
        interactions, ground_truth, candidates = build_canonical_frames(
            ratings,
            catalog,
            dataset=dataset,
            seed=seed,
            negative_samples_per_user=negative_samples_per_user,
            min_interactions=min_interactions,
        )
    except Ml1mPreparationBlocked as exc:
        summary = _summary_frame(
            dataset=dataset,
            status=exc.status,
            message=str(exc),
            config_path=str(config_path),
            root=root,
            timestamp=timestamp,
            raw_paths=raw_paths,
            output_paths=output_paths,
            missing_files=[],
            negative_samples_per_user=negative_samples_per_user,
            min_interactions=min_interactions,
            seed=seed,
            raw_ratings=len(ratings),
            raw_users=ratings["user_id"].nunique(),
            raw_items=ratings["item_id"].nunique(),
            catalog_rows=len(catalog),
            users_dat_present=raw_paths["users"].is_file(),
            users_dat_rows=users_dat_rows,
        )
        _write_summary(summary, summary_path)
        LOGGER.warning("ML-1M preparation blocked: %s", exc)
        return PreparationResult(False, summary, summary_path, output_paths)

    processed_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(validate_interactions(interactions), output_paths["interactions"])
    _write_csv_atomic(validate_ground_truth(ground_truth), output_paths["ground_truth"])
    _write_csv_atomic(validate_candidates(candidates), output_paths["candidates"])
    _write_csv_atomic(catalog.loc[:, CATALOG_COLUMNS], output_paths["catalog"])

    summary = _summary_frame(
        dataset=dataset,
        status="completed",
        message="Prepared canonical MovieLens-1M Plan B files.",
        config_path=str(config_path),
        root=root,
        timestamp=timestamp,
        raw_paths=raw_paths,
        output_paths=output_paths,
        missing_files=[],
        negative_samples_per_user=negative_samples_per_user,
        min_interactions=min_interactions,
        seed=seed,
        raw_ratings=len(ratings),
        raw_users=ratings["user_id"].nunique(),
        raw_items=ratings["item_id"].nunique(),
        eligible_users=ground_truth["user_id"].nunique(),
        train_interactions=len(interactions),
        ground_truth_rows=len(ground_truth),
        candidate_rows=len(candidates),
        catalog_rows=len(catalog),
        users_dat_present=raw_paths["users"].is_file(),
        users_dat_rows=users_dat_rows,
    )
    _write_summary(summary, summary_path)
    LOGGER.info("Saved ML-1M interactions to %s", _display_path(output_paths["interactions"], root))
    LOGGER.info("Saved ML-1M ground truth to %s", _display_path(output_paths["ground_truth"], root))
    LOGGER.info("Saved ML-1M candidates to %s", _display_path(output_paths["candidates"], root))
    LOGGER.info("Saved ML-1M catalog to %s", _display_path(output_paths["catalog"], root))
    LOGGER.info("Saved ML-1M preparation summary to %s", _display_path(summary_path, root))
    return PreparationResult(True, summary, summary_path, output_paths)


class Ml1mPreparationBlocked(ValueError):
    """Raised internally when canonical files should not be written."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def read_ratings_dat(path: Path) -> pd.DataFrame:
    """Read MovieLens-1M ``ratings.dat`` with the ``::`` separator."""

    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=RATINGS_COLUMNS,
        encoding="latin-1",
        dtype={"user_id": "string", "item_id": "string"},
    )
    ratings["user_id"] = ratings["user_id"].astype("string").str.strip()
    ratings["item_id"] = ratings["item_id"].astype("string").str.strip()
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="raise")
    ratings["timestamp"] = pd.to_numeric(ratings["timestamp"], errors="raise").astype("int64")
    return ratings


def read_movies_dat(path: Path) -> pd.DataFrame:
    """Read MovieLens-1M ``movies.dat`` with the ``::`` separator."""

    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=MOVIES_COLUMNS,
        encoding="latin-1",
        dtype={"item_id": "string", "title": "string", "genres": "string"},
    )
    for column in MOVIES_COLUMNS:
        movies[column] = movies[column].astype("string").str.strip()
    return movies


def build_catalog(movies: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical ML-1M movie catalog."""

    catalog = movies.loc[:, MOVIES_COLUMNS].drop_duplicates("item_id").copy()
    catalog["normalized_text"] = (
        catalog["title"].fillna("")
        + " "
        + catalog["genres"].fillna("").str.replace("|", " ", regex=False)
    ).map(_normalize_text)
    return catalog.loc[:, CATALOG_COLUMNS]


def build_canonical_frames(
    ratings: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    dataset: str,
    seed: int,
    negative_samples_per_user: int,
    min_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build interactions, ground truth, and candidates from raw ratings."""

    sorted_ratings = ratings.copy()
    sorted_ratings["__input_order"] = np.arange(len(sorted_ratings))
    sorted_ratings = sorted_ratings.sort_values(
        ["user_id", "timestamp", "__input_order"],
        kind="mergesort",
    )

    user_counts = sorted_ratings.groupby("user_id", sort=False).size()
    eligible_users = user_counts.loc[user_counts >= min_interactions].index
    if len(eligible_users) == 0:
        raise Ml1mPreparationBlocked(
            "blocked_no_eligible_users",
            "No users have enough interactions for leave-one-out splitting.",
        )

    eligible = sorted_ratings.loc[
        sorted_ratings["user_id"].isin(eligible_users)
    ].copy()
    order_in_user = eligible.groupby("user_id", sort=False).cumcount()
    size_by_user = eligible.groupby("user_id", sort=False)["item_id"].transform("size")
    is_ground_truth = order_in_user == (size_by_user - 1)

    train = eligible.loc[~is_ground_truth].copy()
    heldout = eligible.loc[is_ground_truth].copy()

    interactions = train.loc[:, ["user_id", "item_id", "timestamp", "rating"]].copy()
    interactions.insert(2, "split", "train")
    interactions = interactions.loc[:, INTERACTIONS_COLUMNS]

    ground_truth = heldout.rename(columns={"item_id": "ground_truth_item_id"})
    ground_truth["dataset"] = dataset
    ground_truth["split"] = "test"
    ground_truth = ground_truth.loc[:, GROUND_TRUTH_COLUMNS]

    candidates = build_candidates(
        heldout,
        eligible,
        catalog,
        seed=seed,
        negative_samples_per_user=negative_samples_per_user,
    )
    return interactions, ground_truth, candidates


def build_candidates(
    heldout: pd.DataFrame,
    full_interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    seed: int,
    negative_samples_per_user: int,
) -> pd.DataFrame:
    """Build deterministic one-positive-plus-negative candidate sets."""

    all_items = _stable_sorted_values(catalog["item_id"].dropna().astype(str).unique())
    title_by_item = {
        str(row["item_id"]): row["title"]
        for _, row in catalog.loc[:, ["item_id", "title"]].iterrows()
    }
    history_by_user = {
        str(user_id): set(group["item_id"].astype(str).tolist())
        for user_id, group in full_interactions.groupby("user_id", sort=False)
    }

    rows: list[dict[str, Any]] = []
    for _, target in heldout.iterrows():
        user_id = str(target["user_id"])
        positive_item = str(target["item_id"])
        interacted_items = history_by_user.get(user_id, set())
        negative_pool = [item_id for item_id in all_items if item_id not in interacted_items]
        if len(negative_pool) < negative_samples_per_user:
            raise Ml1mPreparationBlocked(
                "blocked_insufficient_negative_pool",
                (
                    "At least one user does not have enough unseen catalog items "
                    "for the configured negative_samples_per_user."
                ),
            )

        rng = np.random.default_rng(_user_seed(seed, user_id))
        sampled_negatives = rng.choice(
            negative_pool,
            size=negative_samples_per_user,
            replace=False,
        ).astype(str).tolist()
        candidate_items = [
            {"item_id": positive_item, "is_ground_truth": True},
            *[
                {"item_id": negative_item, "is_ground_truth": False}
                for negative_item in sampled_negatives
            ],
        ]
        shuffled_indices = rng.permutation(len(candidate_items)).tolist()
        for position, index in enumerate(shuffled_indices, start=1):
            candidate = candidate_items[index]
            item_id = str(candidate["item_id"])
            rows.append(
                {
                    "user_id": user_id,
                    "candidate_item_id": item_id,
                    "candidate_position": position,
                    "is_ground_truth": bool(candidate["is_ground_truth"]),
                    "candidate_item_title": title_by_item.get(item_id, pd.NA),
                }
            )

    return pd.DataFrame(rows, columns=CANDIDATES_COLUMNS)


def _validate_preprocessing_settings(
    *,
    negative_samples_per_user: int,
    min_interactions: int,
) -> None:
    if negative_samples_per_user < 1:
        raise ConfigError("preprocessing.negative_samples_per_user must be positive")
    if min_interactions < 2:
        raise ConfigError("preprocessing.min_interactions must be at least 2")


def _configured_path(
    config: dict[str, Any],
    root: Path,
    key: str,
    fallback: str,
) -> Path:
    value = config.get("paths", {}).get(key) or config.get("dataset", {}).get(key) or fallback
    resolved = resolve_project_path(value, root)
    return resolved if resolved is not None else root / fallback


def _configured_output_dir(
    config: dict[str, Any],
    root: Path,
    key: str,
    fallback: str,
) -> Path:
    value = config.get("output", {}).get(key, fallback)
    resolved = resolve_project_path(value, root)
    return resolved if resolved is not None else root / fallback


def _configured_summary_path(
    config: dict[str, Any],
    root: Path,
    dataset: str,
    tables_dir: Path,
) -> Path:
    value = config.get("output", {}).get(
        "preparation_summary_path",
        f"results/tables/{dataset}_dataset_preparation_summary.csv",
    )
    resolved = resolve_project_path(value, root)
    return resolved if resolved is not None else tables_dir / f"{dataset}_dataset_preparation_summary.csv"


def _canonical_output_paths(
    config: dict[str, Any],
    root: Path,
    processed_dir: Path,
) -> dict[str, Path]:
    return {
        "interactions": _configured_path(
            config,
            root,
            "interactions_path",
            str(processed_dir.relative_to(root) / "interactions.csv"),
        ),
        "ground_truth": _configured_path(
            config,
            root,
            "ground_truth_path",
            str(processed_dir.relative_to(root) / "ground_truth.csv"),
        ),
        "candidates": _configured_path(
            config,
            root,
            "candidates_path",
            str(processed_dir.relative_to(root) / "candidates.csv"),
        ),
        "catalog": _configured_path(
            config,
            root,
            "catalog_path",
            str(processed_dir.relative_to(root) / "catalog.csv"),
        ),
    }


def _summary_frame(
    *,
    dataset: str,
    status: str,
    message: str,
    config_path: str,
    root: Path,
    timestamp: str,
    raw_paths: dict[str, Path],
    output_paths: dict[str, Path],
    missing_files: list[str],
    negative_samples_per_user: int,
    min_interactions: int,
    seed: int,
    raw_ratings: int = 0,
    raw_users: int = 0,
    raw_items: int = 0,
    eligible_users: int = 0,
    train_interactions: int = 0,
    ground_truth_rows: int = 0,
    candidate_rows: int = 0,
    catalog_rows: int = 0,
    users_dat_present: bool = False,
    users_dat_rows: int = 0,
) -> pd.DataFrame:
    row = {
        "dataset": dataset,
        "status": status,
        "message": message,
        "config_path": config_path,
        "timestamp": timestamp,
        "raw_ratings_path": _display_path(raw_paths["ratings"], root),
        "raw_movies_path": _display_path(raw_paths["movies"], root),
        "raw_users_path": _display_path(raw_paths["users"], root),
        "missing_files": ";".join(missing_files),
        "raw_ratings": int(raw_ratings),
        "raw_users": int(raw_users),
        "raw_items": int(raw_items),
        "eligible_users": int(eligible_users),
        "train_interactions": int(train_interactions),
        "ground_truth_rows": int(ground_truth_rows),
        "candidate_rows": int(candidate_rows),
        "catalog_rows": int(catalog_rows),
        "negative_samples_per_user": int(negative_samples_per_user),
        "min_interactions": int(min_interactions),
        "seed": int(seed),
        "users_dat_present": bool(users_dat_present),
        "users_dat_rows": int(users_dat_rows),
        "interactions_path": _display_path(output_paths["interactions"], root),
        "ground_truth_path": _display_path(output_paths["ground_truth"], root),
        "candidates_path": _display_path(output_paths["candidates"], root),
        "catalog_path": _display_path(output_paths["catalog"], root),
    }
    return pd.DataFrame([row])


def _write_summary(summary: pd.DataFrame, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(summary, summary_path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _stable_sorted_values(values: list[str] | np.ndarray) -> list[str]:
    return sorted((str(value) for value in values), key=_stable_id_sort_key)


def _stable_id_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _user_seed(seed: int, user_id: str) -> int:
    digest = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + int(seed)) % (2**32)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _ensure_api_calls_disabled(config: dict[str, Any]) -> None:
    if config.get("api", {}).get("allow_api_calls"):
        raise ConfigError("api.allow_api_calls must be false; ML-1M preparation never calls APIs.")
    if config.get("experiment", {}).get("allow_api_calls"):
        raise ConfigError(
            "experiment.allow_api_calls must be false; ML-1M preparation never calls APIs."
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    try:
        root = find_project_root()
        result = prepare_ml1m(args.config, project_root=root)
    except (ConfigError, ValueError, OSError, pd.errors.ParserError) as exc:
        LOGGER.error("%s", exc)
        return 2

    summary_row = result.summary.iloc[0]
    print(f"status={summary_row['status']}")
    print(f"processed={'true' if result.processed else 'false'}")
    print(f"summary={_display_path(result.summary_path, root)}")
    print(f"raw_ratings={int(summary_row['raw_ratings'])}")
    print(f"eligible_users={int(summary_row['eligible_users'])}")
    print(f"train_interactions={int(summary_row['train_interactions'])}")
    print(f"ground_truth_rows={int(summary_row['ground_truth_rows'])}")
    print(f"candidate_rows={int(summary_row['candidate_rows'])}")
    print(f"catalog_rows={int(summary_row['catalog_rows'])}")
    if str(summary_row.get("missing_files", "")).strip():
        print(f"missing_files={summary_row['missing_files']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

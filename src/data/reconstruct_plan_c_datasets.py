"""Reconstruct Plan C canonical datasets from manually staged public raw files.

Run from the repository root:

    python src/data/reconstruct_plan_c_datasets.py --config configs/plan_c_reconstruction.yaml

The generated files are reconstruction artifacts, not exact original paper
artifacts. This script never downloads data, calls APIs, trains models,
computes metrics, mutates staged raw files, or modifies ``original_repo/``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import logging
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ConfigError, find_project_root, load_yaml_config, resolve_project_path


LOGGER = logging.getLogger("reconstruct_plan_c_datasets")
NEGATIVE_ITEM_UNIVERSES = {"interaction_items", "catalog_items"}

INTERACTIONS_COLUMNS = [
    "user_id",
    "item_id",
    "split",
    "timestamp",
    "rating",
    "internal_user_id",
    "internal_item_id",
]
GROUND_TRUTH_COLUMNS = [
    "user_id",
    "ground_truth_item_id",
    "dataset",
    "split",
    "timestamp",
    "rating",
    "internal_user_id",
    "internal_ground_truth_item_id",
]
CANDIDATES_COLUMNS = [
    "user_id",
    "candidate_item_id",
    "candidate_position",
    "is_ground_truth",
    "candidate_item_title",
    "internal_user_id",
    "internal_candidate_item_id",
]
CATALOG_COLUMNS = [
    "item_id",
    "title",
    "categories",
    "description",
    "brand",
    "normalized_text",
    "internal_item_id",
]

SUMMARY_COLUMNS = [
    "dataset",
    "status",
    "diagnostic_subset",
    "max_users",
    "input_files_found",
    "users",
    "items",
    "raw_interactions",
    "eligible_users",
    "excluded_users",
    "train_interactions",
    "ground_truth_rows",
    "candidate_rows",
    "catalog_rows",
    "negative_samples_per_user",
    "candidate_item_universe",
    "candidate_universe_size",
    "catalog_size",
    "interaction_item_count",
    "users_with_insufficient_negatives",
    "split_type",
    "artifact_status",
    "exact_original_artifact_recovered",
    "reconstruction_status",
    "output_dir",
    "notes",
]
INVENTORY_COLUMNS = [
    "dataset",
    "artifact_name",
    "path",
    "written",
    "rows_or_lines",
    "artifact_status",
    "format_status",
    "notes",
]
BLOCKER_COLUMNS = [
    "dataset",
    "blocker_type",
    "missing_inputs",
    "affected_artifacts",
    "requires_author_contact",
    "notes",
]
STATS_COLUMNS = [
    "dataset",
    "diagnostic_subset",
    "max_users",
    "raw_users",
    "raw_items",
    "raw_interactions",
    "eligible_users",
    "excluded_users",
    "train_interactions",
    "ground_truth_rows",
    "candidate_rows",
    "catalog_rows",
    "min_interactions_per_user",
    "negative_samples_per_user",
    "candidate_item_universe",
    "candidate_universe_size",
    "catalog_size",
    "interaction_item_count",
    "users_with_insufficient_negatives",
    "split_type",
    "reconstruction_status",
]


@dataclass(frozen=True)
class ReconstructionSettings:
    root: Path
    config_path: Path
    staging_dir: Path
    output_base_dir: Path
    tables_dir: Path
    seed: int
    negatives_per_user: int
    negative_item_universe: str
    min_interactions: int
    max_users: int | None
    skip_paper_like_artifacts: bool
    datasets_enabled: list[str]
    dataset_configs: dict[str, dict[str, Any]]


@dataclass
class DatasetResult:
    dataset: str
    status: str
    summary: dict[str, Any]
    artifacts: list[dict[str, Any]]
    blockers: list[dict[str, Any]]
    stats: dict[str, Any]


class ReconstructionBlocked(ValueError):
    """Raised internally when a dataset cannot be reconstructed safely."""

    def __init__(
        self,
        status: str,
        message: str,
        *,
        missing_inputs: list[str] | None = None,
        affected_artifacts: list[str] | None = None,
        requires_author_contact: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.missing_inputs = missing_inputs or []
        self.affected_artifacts = affected_artifacts or []
        self.requires_author_contact = requires_author_contact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct Plan C canonical datasets from staged raw files."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to configs/plan_c_reconstruction.yaml.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of datasets to reconstruct.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help=(
            "Development-only diagnostic subset. Keeps the first N eligible users "
            "in deterministic order after loading interactions."
        ),
    )
    parser.add_argument(
        "--skip-paper-like-artifacts",
        action="store_true",
        help=(
            "Development-only option to skip datamaps.json, meta.json, "
            "sequential_data.txt, and negative_samples.txt."
        ),
    )
    return parser.parse_args(argv)


def reconstruct_plan_c_datasets(
    config_path: str | Path,
    *,
    datasets: list[str] | None = None,
    max_users: int | None = None,
    skip_paper_like_artifacts: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run Plan C raw-to-canonical reconstruction and write diagnostics."""

    settings = _load_settings(
        config_path,
        project_root=project_root,
        max_users=max_users,
        skip_paper_like_artifacts=skip_paper_like_artifacts,
    )
    selected = datasets or settings.datasets_enabled
    invalid = [dataset for dataset in selected if dataset not in settings.dataset_configs]
    if invalid:
        raise ConfigError(f"Unknown Plan C dataset(s): {', '.join(invalid)}")

    settings.output_base_dir.mkdir(parents=True, exist_ok=True)
    settings.tables_dir.mkdir(parents=True, exist_ok=True)

    results: list[DatasetResult] = []
    for dataset in selected:
        LOGGER.info("[%s] start dataset", dataset)
        dataset_config = settings.dataset_configs[dataset]
        try:
            if dataset in {"amazon_beauty", "amazon_sports"}:
                result = _reconstruct_amazon_dataset(settings, dataset, dataset_config)
            elif dataset == "ml1m":
                result = _reconstruct_ml1m(settings, dataset_config)
            elif dataset == "lastfm":
                result = _reconstruct_lastfm(settings, dataset_config)
            else:  # pragma: no cover - guarded above.
                raise ConfigError(f"Unhandled dataset: {dataset}")
        except ReconstructionBlocked as exc:
            LOGGER.warning("[%s] blocked: %s", dataset, exc)
            result = _blocked_result(settings, dataset, dataset_config, exc)
        results.append(result)
        LOGGER.info("[%s] finish dataset status=%s", dataset, result.status)

    LOGGER.info("write summaries")
    frames = _write_diagnostics(settings, results)
    return frames


def _load_settings(
    config_path: str | Path,
    *,
    project_root: str | Path | None,
    max_users: int | None,
    skip_paper_like_artifacts: bool,
) -> ReconstructionSettings:
    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    _ensure_no_api(config)
    plan = config.get("plan_c_reconstruction")
    if not isinstance(plan, dict):
        raise ConfigError("Missing plan_c_reconstruction section")
    staging = resolve_project_path(plan.get("input_staging_dir"), root)
    output_base = resolve_project_path(plan.get("output_base_dir"), root)
    tables_dir = resolve_project_path(config.get("output", {}).get("tables_dir"), root)
    if staging is None or output_base is None or tables_dir is None:
        raise ConfigError("Plan C reconstruction paths must not be null")
    dataset_configs = plan.get("datasets", {})
    if not isinstance(dataset_configs, dict):
        raise ConfigError("plan_c_reconstruction.datasets must be a mapping")
    if max_users is not None and max_users < 1:
        raise ConfigError("--max-users must be a positive integer when provided")
    negative_sampling = config.get("negative_sampling", {})
    plan_negative_sampling = plan.get("negative_sampling", {})
    negative_item_universe = str(
        negative_sampling.get(
            "item_universe",
            plan_negative_sampling.get("item_universe", "interaction_items"),
        )
    ).strip()
    if negative_item_universe not in NEGATIVE_ITEM_UNIVERSES:
        allowed = ", ".join(sorted(NEGATIVE_ITEM_UNIVERSES))
        raise ConfigError(f"negative_sampling.item_universe must be one of: {allowed}")
    return ReconstructionSettings(
        root=root,
        config_path=_resolve_input_path(Path(config_path), root),
        staging_dir=staging,
        output_base_dir=output_base,
        tables_dir=tables_dir,
        seed=int(plan.get("seed", config.get("experiment", {}).get("seed", 42))),
        negatives_per_user=int(plan.get("negatives_per_user", 99)),
        negative_item_universe=negative_item_universe,
        min_interactions=int(plan.get("min_interactions_per_user", 2)),
        max_users=max_users,
        skip_paper_like_artifacts=bool(skip_paper_like_artifacts),
        datasets_enabled=list(plan.get("datasets_enabled", [])),
        dataset_configs=dataset_configs,
    )


def _reconstruct_amazon_dataset(
    settings: ReconstructionSettings,
    dataset: str,
    dataset_config: dict[str, Any],
) -> DatasetResult:
    reviews_path = settings.staging_dir / str(dataset_config["reviews_file"])
    metadata_path = settings.staging_dir / str(dataset_config["metadata_file"])
    missing = [
        _display_path(path, settings.root)
        for path in [reviews_path, metadata_path]
        if not path.is_file()
    ]
    if missing:
        raise ReconstructionBlocked(
            "missing_raw",
            "Required Amazon raw file(s) are missing.",
            missing_inputs=missing,
            affected_artifacts=_canonical_artifact_names(include_split=True),
        )

    LOGGER.info("[%s] load reviews/interactions", dataset)
    reviews = _read_amazon_reviews(reviews_path)
    if reviews.empty:
        raise ReconstructionBlocked(
            "blocked_no_valid_reviews",
            "No valid Amazon review rows with required sequential fields were found.",
            affected_artifacts=_canonical_artifact_names(include_split=True),
        )
    raw_users = reviews["user_id"].nunique()
    raw_items = reviews["item_id"].nunique()
    LOGGER.info("[%s] load metadata/catalog", dataset)
    catalog = _read_amazon_metadata(metadata_path)
    LOGGER.info("[%s] build catalog", dataset)
    catalog = _ensure_catalog_contains_items(catalog, reviews["item_id"].unique())
    catalog = _assign_internal_items(catalog)
    item2internal = dict(zip(catalog["item_id"].astype(str), catalog["internal_item_id"].astype(str)))

    LOGGER.info("[%s] build leave-one-out split and candidates", dataset)
    split = _build_leave_one_out_split(
        reviews,
        catalog,
        dataset=dataset,
        seed=settings.seed,
        negatives_per_user=settings.negatives_per_user,
        negative_item_universe=settings.negative_item_universe,
        min_interactions=settings.min_interactions,
        max_users=settings.max_users,
    )
    LOGGER.info("[%s] write canonical CSV files and paper-like artifacts", dataset)
    _write_reconstructed_dataset(
        settings,
        dataset=dataset,
        dataset_config=dataset_config,
        split_type="chronological_leave_one_out",
        reconstruction_status="reconstructed_from_public_raw",
        exact_original_artifact_recovered="no",
        input_files=[reviews_path, metadata_path],
        interactions=split["interactions"],
        ground_truth=split["ground_truth"],
        catalog=catalog,
        user2internal=split["user2internal"],
        item2internal=item2internal,
        sequential_lines=split["sequential_lines"],
        negative_sample_lines=split["negative_sample_lines"],
        candidates_writer=split["candidates_writer"],
        candidate_item_universe=split["candidate_item_universe"],
    )
    output_dir = settings.output_base_dir / dataset
    artifacts = _artifact_inventory_for_written_dataset(
        settings,
        dataset,
        output_dir,
        include_split=True,
        counts={
            "interactions.csv": len(split["interactions"]),
            "ground_truth.csv": len(split["ground_truth"]),
            "candidates.csv": split["candidate_rows"],
            "catalog.csv": len(catalog),
            "sequential_data.txt": len(split["sequential_lines"]),
            "negative_samples.txt": len(split["negative_sample_lines"]),
        },
    )
    status = "diagnostic_subset_completed" if split["diagnostic_subset"] else "completed"
    reconstruction_status = (
        "diagnostic_subset_reconstructed_from_public_raw"
        if split["diagnostic_subset"]
        else "reconstructed_from_public_raw"
    )
    summary = _summary_row(
        settings,
        dataset=dataset,
        status=status,
        diagnostic_subset=split["diagnostic_subset"],
        input_files_found=True,
        users=len(split["user2internal"]),
        items=len(catalog),
        raw_interactions=len(reviews),
        eligible_users=len(split["user2internal"]),
        excluded_users=max(raw_users - len(split["user2internal"]), 0),
        train_interactions=len(split["interactions"]),
        ground_truth_rows=len(split["ground_truth"]),
        candidate_rows=split["candidate_rows"],
        catalog_rows=len(catalog),
        candidate_item_universe=split["candidate_item_universe"],
        candidate_universe_size=split["candidate_universe_size"],
        catalog_size=len(catalog),
        interaction_item_count=split["interaction_item_count"],
        users_with_insufficient_negatives=split["users_with_insufficient_negatives"],
        split_type="chronological_leave_one_out",
        artifact_status="written_reconstructed",
        exact_original_artifact_recovered="no",
        reconstruction_status=reconstruction_status,
        output_dir=output_dir,
        notes=(
            "Paper-like artifacts reconstructed from public Amazon raw data. "
            "Official Beauty/Sports processed Dropbox artifacts remain unavailable."
            + (" Diagnostic subset created with --max-users; not a final reconstruction." if split["diagnostic_subset"] else "")
        ),
    )
    stats = _stats_row_from_summary(summary, raw_users=raw_users, raw_items=raw_items)
    return DatasetResult(dataset, status, summary, artifacts, [], stats)


def _reconstruct_ml1m(
    settings: ReconstructionSettings,
    dataset_config: dict[str, Any],
) -> DatasetResult:
    dataset = "ml1m"
    archive_path = settings.staging_dir / str(dataset_config["archive_file"])
    if not archive_path.is_file():
        raise ReconstructionBlocked(
            "missing_raw",
            "MovieLens-1M ZIP is missing.",
            missing_inputs=[_display_path(archive_path, settings.root)],
            affected_artifacts=_canonical_artifact_names(include_split=True),
        )
    try:
        LOGGER.info("[%s] load archive members", dataset)
        with zipfile.ZipFile(archive_path, "r") as archive:
            names = archive.namelist()
            ratings_member = _find_zip_member(names, str(dataset_config["ratings_member"]))
            movies_member = _find_zip_member(names, str(dataset_config["movies_member"]))
            missing = []
            if ratings_member is None:
                missing.append(str(dataset_config["ratings_member"]))
            if movies_member is None:
                missing.append(str(dataset_config["movies_member"]))
            if missing:
                raise ReconstructionBlocked(
                    "missing_raw_member",
                    "MovieLens-1M ZIP is missing required member(s).",
                    missing_inputs=missing,
                    affected_artifacts=_canonical_artifact_names(include_split=True),
                )
            LOGGER.info("[%s] load ratings/interactions", dataset)
            ratings = _read_ml1m_ratings(archive, ratings_member)
            LOGGER.info("[%s] load metadata/catalog", dataset)
            movies = _read_ml1m_movies(archive, movies_member)
    except zipfile.BadZipFile as exc:
        raise ReconstructionBlocked(
            "invalid_archive",
            "MovieLens-1M ZIP could not be opened.",
            missing_inputs=[_display_path(archive_path, settings.root)],
            affected_artifacts=_canonical_artifact_names(include_split=True),
        ) from exc

    LOGGER.info("[%s] build catalog", dataset)
    catalog = _build_ml1m_catalog(movies)
    catalog = _ensure_catalog_contains_items(catalog, ratings["item_id"].unique())
    catalog = _assign_internal_items(catalog)
    item2internal = dict(zip(catalog["item_id"].astype(str), catalog["internal_item_id"].astype(str)))
    raw_users = ratings["user_id"].nunique()
    raw_items = ratings["item_id"].nunique()
    LOGGER.info("[%s] build leave-one-out split and candidates", dataset)
    split = _build_leave_one_out_split(
        ratings,
        catalog,
        dataset=dataset,
        seed=settings.seed,
        negatives_per_user=settings.negatives_per_user,
        negative_item_universe=settings.negative_item_universe,
        min_interactions=settings.min_interactions,
        max_users=settings.max_users,
    )
    LOGGER.info("[%s] write canonical CSV files and paper-like artifacts", dataset)
    _write_reconstructed_dataset(
        settings,
        dataset=dataset,
        dataset_config=dataset_config,
        split_type="chronological_leave_one_out",
        reconstruction_status="reconstructed_from_public_raw",
        exact_original_artifact_recovered="no",
        input_files=[archive_path],
        interactions=split["interactions"],
        ground_truth=split["ground_truth"],
        catalog=catalog,
        user2internal=split["user2internal"],
        item2internal=item2internal,
        sequential_lines=split["sequential_lines"],
        negative_sample_lines=split["negative_sample_lines"],
        candidates_writer=split["candidates_writer"],
        candidate_item_universe=split["candidate_item_universe"],
    )
    output_dir = settings.output_base_dir / dataset
    artifacts = _artifact_inventory_for_written_dataset(
        settings,
        dataset,
        output_dir,
        include_split=True,
        counts={
            "interactions.csv": len(split["interactions"]),
            "ground_truth.csv": len(split["ground_truth"]),
            "candidates.csv": split["candidate_rows"],
            "catalog.csv": len(catalog),
            "sequential_data.txt": len(split["sequential_lines"]),
            "negative_samples.txt": len(split["negative_sample_lines"]),
        },
    )
    status = "diagnostic_subset_completed" if split["diagnostic_subset"] else "completed"
    reconstruction_status = (
        "diagnostic_subset_reconstructed_from_public_raw"
        if split["diagnostic_subset"]
        else "reconstructed_from_public_raw"
    )
    summary = _summary_row(
        settings,
        dataset=dataset,
        status=status,
        diagnostic_subset=split["diagnostic_subset"],
        input_files_found=True,
        users=len(split["user2internal"]),
        items=len(catalog),
        raw_interactions=len(ratings),
        eligible_users=len(split["user2internal"]),
        excluded_users=max(raw_users - len(split["user2internal"]), 0),
        train_interactions=len(split["interactions"]),
        ground_truth_rows=len(split["ground_truth"]),
        candidate_rows=split["candidate_rows"],
        catalog_rows=len(catalog),
        candidate_item_universe=split["candidate_item_universe"],
        candidate_universe_size=split["candidate_universe_size"],
        catalog_size=len(catalog),
        interaction_item_count=split["interaction_item_count"],
        users_with_insufficient_negatives=split["users_with_insufficient_negatives"],
        split_type="chronological_leave_one_out",
        artifact_status="written_reconstructed",
        exact_original_artifact_recovered="no",
        reconstruction_status=reconstruction_status,
        output_dir=output_dir,
        notes=(
            "Plan C ML-1M reconstruction written separately from Plan B processed files."
            + (" Diagnostic subset created with --max-users; not a final reconstruction." if split["diagnostic_subset"] else "")
        ),
    )
    stats = _stats_row_from_summary(summary, raw_users=raw_users, raw_items=raw_items)
    return DatasetResult(dataset, status, summary, artifacts, [], stats)


def _reconstruct_lastfm(
    settings: ReconstructionSettings,
    dataset_config: dict[str, Any],
) -> DatasetResult:
    dataset = "lastfm"
    archive_path = settings.staging_dir / str(dataset_config["archive_file"])
    if not archive_path.is_file():
        raise ReconstructionBlocked(
            "missing_raw",
            "LastFM HetRec ZIP is missing.",
            missing_inputs=[_display_path(archive_path, settings.root)],
            affected_artifacts=["interactions.csv", "catalog.csv", "datamaps.json", "meta.json"],
        )
    try:
        LOGGER.info("[%s] load archive members", dataset)
        with zipfile.ZipFile(archive_path, "r") as archive:
            names = archive.namelist()
            user_artists_member = _find_zip_member(names, str(dataset_config["user_artists_member"]))
            artists_member = _find_zip_member(names, str(dataset_config["artists_member"]))
            missing = []
            if user_artists_member is None:
                missing.append(str(dataset_config["user_artists_member"]))
            if artists_member is None:
                missing.append(str(dataset_config["artists_member"]))
            if missing:
                raise ReconstructionBlocked(
                    "missing_raw_member",
                    "LastFM ZIP is missing required member(s).",
                    missing_inputs=missing,
                    affected_artifacts=["interactions.csv", "catalog.csv", "datamaps.json", "meta.json"],
                )
            LOGGER.info("[%s] load interactions", dataset)
            interactions = _read_lastfm_user_artists(archive, user_artists_member)
            LOGGER.info("[%s] load metadata/catalog", dataset)
            catalog = _read_lastfm_artists(archive, artists_member)
    except zipfile.BadZipFile as exc:
        raise ReconstructionBlocked(
            "invalid_archive",
            "LastFM ZIP could not be opened.",
            missing_inputs=[_display_path(archive_path, settings.root)],
            affected_artifacts=["interactions.csv", "catalog.csv", "datamaps.json", "meta.json"],
        ) from exc

    if interactions.empty or catalog.empty:
        raise ReconstructionBlocked(
            "blocked_no_valid_rows",
            "LastFM raw files did not contain valid interaction/catalog rows.",
            affected_artifacts=["interactions.csv", "catalog.csv", "datamaps.json", "meta.json"],
        )
    LOGGER.info("[%s] build catalog", dataset)
    catalog = _ensure_catalog_contains_items(catalog, interactions["item_id"].unique())
    catalog = _assign_internal_items(catalog)
    user2internal = _build_internal_map(interactions["user_id"].unique())
    item2internal = dict(zip(catalog["item_id"].astype(str), catalog["internal_item_id"].astype(str)))
    interactions = interactions.copy()
    interactions["split"] = "observed"
    interactions["timestamp"] = pd.NA
    interactions["internal_user_id"] = interactions["user_id"].astype(str).map(user2internal)
    interactions["internal_item_id"] = interactions["item_id"].astype(str).map(item2internal)
    interactions = interactions.loc[:, INTERACTIONS_COLUMNS]
    interactions = interactions.sort_values(
        ["internal_user_id", "internal_item_id"],
        key=lambda series: series.astype(str),
        kind="mergesort",
    )

    output_dir = settings.output_base_dir / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("[%s] write canonical CSV files", dataset)
    _write_csv(interactions, output_dir / "interactions.csv")
    _write_csv(catalog.loc[:, CATALOG_COLUMNS], output_dir / "catalog.csv")
    if settings.skip_paper_like_artifacts:
        LOGGER.info("[%s] skip paper-like artifacts by request", dataset)
    else:
        LOGGER.info("[%s] write paper-like artifacts", dataset)
        _write_datamaps(
            output_dir / "datamaps.json",
            dataset=dataset,
            user2internal=user2internal,
            item2internal=item2internal,
            reconstruction_status="extension_only",
            format_status="reconstructed_format",
            split_type="interactions_catalog_only_no_chronological_timestamp",
        )
        _write_meta_json(output_dir / "meta.json", catalog)
    _write_readme(
        output_dir / "README_RECONSTRUCTED.md",
        dataset=dataset,
        input_files=[archive_path],
        split_type="interactions_catalog_only_no_chronological_timestamp",
        reconstruction_status="extension_only",
        exact_original_artifact_recovered="no",
        limitations=[
            "user_artists.dat has interaction weights but no listening timestamp.",
            "No ground_truth.csv, candidates.csv, sequential_data.txt, or negative_samples.txt was written.",
            "This is an extension-only interaction/catalog reconstruction until an exact paper protocol is recovered.",
        ],
    )

    artifacts = _artifact_inventory_for_written_dataset(
        settings,
        dataset,
        output_dir,
        include_split=False,
        counts={
            "interactions.csv": len(interactions),
            "catalog.csv": len(catalog),
        },
    )
    blockers = [
        {
            "dataset": dataset,
            "blocker_type": "blocked_chronological_split",
            "missing_inputs": "listening timestamps in user_artists.dat",
            "affected_artifacts": "ground_truth.csv;candidates.csv;sequential_data.txt;negative_samples.txt",
            "requires_author_contact": False,
            "notes": "LastFM is kept extension_only; no non-chronological ground truth was fabricated.",
        }
    ]
    raw_users = interactions["user_id"].nunique()
    raw_items = interactions["item_id"].nunique()
    summary = _summary_row(
        settings,
        dataset=dataset,
        status="partial_extension_only",
        diagnostic_subset=False,
        input_files_found=True,
        users=raw_users,
        items=len(catalog),
        raw_interactions=len(interactions),
        eligible_users=0,
        excluded_users=0,
        train_interactions=len(interactions),
        ground_truth_rows=0,
        candidate_rows=0,
        catalog_rows=len(catalog),
        split_type="interactions_catalog_only_no_chronological_timestamp",
        artifact_status="partial_written_reconstructed",
        exact_original_artifact_recovered="no",
        reconstruction_status="extension_only",
        output_dir=output_dir,
        notes="LastFM interactions/catalog written only; no chronological leave-one-out split was created.",
    )
    stats = _stats_row_from_summary(summary, raw_users=raw_users, raw_items=raw_items)
    return DatasetResult(dataset, "partial_extension_only", summary, artifacts, blockers, stats)


def _read_amazon_reviews(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for input_order, line in enumerate(handle):
            record = _parse_record_line(line)
            if not record:
                continue
            if not all(_has_value(record.get(field)) for field in ["reviewerID", "asin", "overall", "unixReviewTime"]):
                continue
            rows.append(
                {
                    "user_id": str(record["reviewerID"]).strip(),
                    "item_id": str(record["asin"]).strip(),
                    "rating": float(record["overall"]),
                    "timestamp": int(record["unixReviewTime"]),
                    "__input_order": input_order,
                }
            )
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp", "__input_order"])


def _read_amazon_metadata(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = _parse_record_line(line)
            if not record or not _has_value(record.get("asin")):
                continue
            title = _clean_text(record.get("title", ""))
            categories = _stringify_categories(record.get("categories", ""))
            description = _stringify_description(record.get("description", ""))
            brand = _clean_text(record.get("brand", ""))
            rows.append(
                {
                    "item_id": str(record["asin"]).strip(),
                    "title": title,
                    "categories": categories,
                    "description": description,
                    "brand": brand,
                    "normalized_text": _normalize_text(" ".join([title, categories, description, brand])),
                }
            )
    catalog = pd.DataFrame(rows, columns=CATALOG_COLUMNS[:-1])
    if catalog.empty:
        return pd.DataFrame(columns=CATALOG_COLUMNS[:-1])
    catalog = catalog.drop_duplicates("item_id", keep="first")
    return catalog


def _read_ml1m_ratings(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with archive.open(member, "r") as raw:
        text = io.TextIOWrapper(raw, encoding="latin-1", errors="replace")
        for input_order, line in enumerate(text):
            parts = line.rstrip("\n\r").split("::")
            if len(parts) < 4:
                continue
            rows.append(
                {
                    "user_id": parts[0].strip(),
                    "item_id": parts[1].strip(),
                    "rating": float(parts[2]),
                    "timestamp": int(parts[3]),
                    "__input_order": input_order,
                }
            )
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp", "__input_order"])


def _read_ml1m_movies(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with archive.open(member, "r") as raw:
        text = io.TextIOWrapper(raw, encoding="latin-1", errors="replace")
        for line in text:
            parts = line.rstrip("\n\r").split("::")
            if len(parts) < 3:
                continue
            rows.append(
                {
                    "item_id": parts[0].strip(),
                    "title": parts[1].strip(),
                    "categories": parts[2].strip().replace("|", " | "),
                    "description": "",
                    "brand": "",
                    "normalized_text": _normalize_text(f"{parts[1]} {parts[2].replace('|', ' ')}"),
                }
            )
    return pd.DataFrame(rows, columns=CATALOG_COLUMNS[:-1])


def _build_ml1m_catalog(movies: pd.DataFrame) -> pd.DataFrame:
    if movies.empty:
        return pd.DataFrame(columns=CATALOG_COLUMNS[:-1])
    return movies.drop_duplicates("item_id", keep="first").copy()


def _read_lastfm_user_artists(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with archive.open(member, "r") as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        header = next(text, "")
        for line in text:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 3:
                continue
            rows.append(
                {
                    "user_id": parts[0].strip(),
                    "item_id": parts[1].strip(),
                    "rating": float(parts[2]),
                }
            )
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])


def _read_lastfm_artists(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with archive.open(member, "r") as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        header = next(text, "")
        for line in text:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 2:
                continue
            title = parts[1].strip()
            rows.append(
                {
                    "item_id": parts[0].strip(),
                    "title": title,
                    "categories": "artist",
                    "description": "",
                    "brand": "",
                    "normalized_text": _normalize_text(title),
                }
            )
    return pd.DataFrame(rows, columns=CATALOG_COLUMNS[:-1]).drop_duplicates("item_id", keep="first")


def _build_leave_one_out_split(
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    dataset: str,
    seed: int,
    negatives_per_user: int,
    negative_item_universe: str,
    min_interactions: int,
    max_users: int | None,
) -> dict[str, Any]:
    if interactions.empty:
        raise ReconstructionBlocked(
            "blocked_no_interactions",
            "No interactions are available for leave-one-out splitting.",
            affected_artifacts=_canonical_artifact_names(include_split=True),
        )
    sorted_df = interactions.copy()
    if "__input_order" not in sorted_df.columns:
        sorted_df["__input_order"] = np.arange(len(sorted_df))
    sorted_df["user_id"] = sorted_df["user_id"].astype(str)
    sorted_df["item_id"] = sorted_df["item_id"].astype(str)
    interaction_item_universe = _stable_sorted_values(sorted_df["item_id"].unique())
    LOGGER.info("[%s] sort interactions", dataset)
    sorted_df = sorted_df.sort_values(
        ["user_id", "timestamp", "item_id", "__input_order"],
        kind="mergesort",
    )
    counts = sorted_df.groupby("user_id", sort=False).size()
    eligible_users = _stable_sorted_values(counts.loc[counts >= min_interactions].index.astype(str).tolist())
    if not eligible_users:
        raise ReconstructionBlocked(
            "blocked_no_eligible_users",
            "No users have enough interactions for leave-one-out splitting.",
            affected_artifacts=_canonical_artifact_names(include_split=True),
        )
    diagnostic_subset = max_users is not None
    if max_users is not None:
        eligible_users = eligible_users[:max_users]
        LOGGER.info("[%s] diagnostic subset enabled: max_users=%s", dataset, max_users)
    eligible = sorted_df.loc[sorted_df["user_id"].isin(eligible_users)].copy()
    user2internal = _build_internal_map(eligible_users)
    item2internal = dict(zip(catalog["item_id"].astype(str), catalog["internal_item_id"].astype(str)))
    eligible["internal_user_id"] = eligible["user_id"].map(user2internal)
    eligible["internal_item_id"] = eligible["item_id"].map(item2internal)
    eligible = eligible.dropna(subset=["internal_item_id"])
    order_in_user = eligible.groupby("user_id", sort=False).cumcount()
    size_by_user = eligible.groupby("user_id", sort=False)["item_id"].transform("size")
    is_gt = order_in_user == (size_by_user - 1)
    train = eligible.loc[~is_gt].copy()
    heldout = eligible.loc[is_gt].copy()

    interactions_out = train.loc[
        :, ["user_id", "item_id", "timestamp", "rating", "internal_user_id", "internal_item_id"]
    ].copy()
    interactions_out.insert(2, "split", "train")
    interactions_out = interactions_out.loc[:, INTERACTIONS_COLUMNS]

    ground_truth = heldout.rename(columns={"item_id": "ground_truth_item_id"}).copy()
    ground_truth["dataset"] = dataset
    ground_truth["split"] = "test"
    ground_truth = ground_truth.rename(columns={"internal_item_id": "internal_ground_truth_item_id"})
    ground_truth = ground_truth.loc[:, GROUND_TRUTH_COLUMNS]

    LOGGER.info("[%s] build sequential_data lines", dataset)
    sequential_lines = _build_sequential_lines(eligible, user2internal, item2internal)
    LOGGER.info("[%s] generate candidates", dataset)
    candidates_state = _prepare_candidate_state(
        heldout,
        eligible,
        catalog,
        interaction_item_universe=interaction_item_universe,
        seed=seed,
        negatives_per_user=negatives_per_user,
        negative_item_universe=negative_item_universe,
    )
    return {
        "interactions": interactions_out,
        "ground_truth": ground_truth,
        "user2internal": user2internal,
        "sequential_lines": sequential_lines,
        "negative_sample_lines": candidates_state["negative_sample_lines"],
        "candidate_rows": candidates_state["candidate_rows"],
        "candidates_writer": candidates_state["writer"],
        "diagnostic_subset": diagnostic_subset,
        "candidate_item_universe": candidates_state["candidate_item_universe"],
        "candidate_universe_size": candidates_state["candidate_universe_size"],
        "catalog_size": candidates_state["catalog_size"],
        "interaction_item_count": candidates_state["interaction_item_count"],
        "users_with_insufficient_negatives": candidates_state["users_with_insufficient_negatives"],
    }


def _prepare_candidate_state(
    heldout: pd.DataFrame,
    full_interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    interaction_item_universe: list[str],
    seed: int,
    negatives_per_user: int,
    negative_item_universe: str,
) -> dict[str, Any]:
    catalog_items = _stable_sorted_values(catalog["item_id"].astype(str).unique())
    if negative_item_universe == "interaction_items":
        all_items = _stable_sorted_values(interaction_item_universe)
    elif negative_item_universe == "catalog_items":
        all_items = catalog_items
    else:
        raise ConfigError(f"Unsupported negative item universe: {negative_item_universe}")
    all_items_array = np.array(all_items, dtype=object)
    item2internal = dict(zip(catalog["item_id"].astype(str), catalog["internal_item_id"].astype(str)))
    title_by_item = dict(zip(catalog["item_id"].astype(str), catalog["title"].fillna("").astype(str)))
    history_by_user = {
        str(user_id): set(group["item_id"].astype(str).tolist())
        for user_id, group in full_interactions.groupby("user_id", sort=False)
    }
    heldout_ordered = heldout.sort_values(
        ["internal_user_id"],
        key=lambda series: series.astype(str).map(_stable_id_sort_tuple),
        kind="mergesort",
    )
    per_user: list[dict[str, Any]] = []
    negative_sample_lines: list[str] = []
    all_items_set = set(all_items)
    users_with_insufficient_negatives = 0
    for _, target in heldout_ordered.iterrows():
        user_id = str(target["user_id"])
        internal_user_id = str(target["internal_user_id"])
        positive_item = str(target["item_id"])
        history = history_by_user.get(user_id, set())
        available_negative_count = len(all_items_set) - len(history.intersection(all_items_set))
        if available_negative_count < negatives_per_user:
            users_with_insufficient_negatives += 1
            raise ReconstructionBlocked(
                "blocked_insufficient_negative_pool",
                (
                    "At least one user lacks enough unseen items for deterministic "
                    f"negative sampling from {negative_item_universe}."
                ),
                affected_artifacts=["candidates.csv", "negative_samples.txt"],
            )
        rng = np.random.default_rng(_user_seed(seed, user_id))
        negatives = _sample_negative_items(
            all_items_array,
            history,
            negatives_per_user,
            rng,
        )
        negative_sample_lines.append(
            " ".join([internal_user_id, *[item2internal[item] for item in negatives]])
        )
        candidate_items = [
            (positive_item, True),
            *[(negative, False) for negative in negatives],
        ]
        shuffled_indices = rng.permutation(len(candidate_items)).tolist()
        per_user.append(
            {
                "user_id": user_id,
                "internal_user_id": internal_user_id,
                "candidate_items": candidate_items,
                "shuffled_indices": shuffled_indices,
            }
        )

    def writer(path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=CANDIDATES_COLUMNS)
            csv_writer.writeheader()
            for entry in per_user:
                for position, index in enumerate(entry["shuffled_indices"], start=1):
                    item_id, is_positive = entry["candidate_items"][index]
                    csv_writer.writerow(
                        {
                            "user_id": entry["user_id"],
                            "candidate_item_id": item_id,
                            "candidate_position": position,
                            "is_ground_truth": bool(is_positive),
                            "candidate_item_title": title_by_item.get(item_id, ""),
                            "internal_user_id": entry["internal_user_id"],
                            "internal_candidate_item_id": item2internal[item_id],
                        }
                    )
                    row_count += 1
        return row_count

    return {
        "writer": writer,
        "candidate_rows": len(per_user) * (negatives_per_user + 1),
        "negative_sample_lines": negative_sample_lines,
        "candidate_item_universe": negative_item_universe,
        "candidate_universe_size": len(all_items),
        "catalog_size": len(catalog_items),
        "interaction_item_count": len(interaction_item_universe),
        "users_with_insufficient_negatives": users_with_insufficient_negatives,
    }


def _sample_negative_items(
    all_items: np.ndarray,
    history: set[str],
    negatives_per_user: int,
    rng: np.random.Generator,
) -> list[str]:
    """Sample deterministic negatives without scanning the full catalog per user."""

    selected: list[str] = []
    selected_set: set[str] = set()
    batch_size = max(128, negatives_per_user * 4)
    while len(selected) < negatives_per_user:
        candidate_indices = rng.integers(0, len(all_items), size=batch_size)
        for index in candidate_indices:
            item_id = str(all_items[int(index)])
            if item_id in history or item_id in selected_set:
                continue
            selected.append(item_id)
            selected_set.add(item_id)
            if len(selected) == negatives_per_user:
                break
    return selected


def _write_reconstructed_dataset(
    settings: ReconstructionSettings,
    *,
    dataset: str,
    dataset_config: dict[str, Any],
    split_type: str,
    reconstruction_status: str,
    exact_original_artifact_recovered: str,
    input_files: list[Path],
    interactions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    catalog: pd.DataFrame,
    user2internal: dict[str, str],
    item2internal: dict[str, str],
    sequential_lines: list[str],
    negative_sample_lines: list[str],
    candidates_writer: Any,
    candidate_item_universe: str,
) -> None:
    output_dir = settings.output_base_dir / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("[%s] write canonical CSV files", dataset)
    _write_csv(interactions, output_dir / "interactions.csv")
    _write_csv(ground_truth, output_dir / "ground_truth.csv")
    written_candidate_rows = candidates_writer(output_dir / "candidates.csv")
    if written_candidate_rows != len(ground_truth) * (settings.negatives_per_user + 1):
        raise ReconstructionBlocked(
            "blocked_candidate_write_mismatch",
            "Candidate row count did not match expected one-positive-plus-negatives structure.",
            affected_artifacts=["candidates.csv"],
    )
    _write_csv(catalog.loc[:, CATALOG_COLUMNS], output_dir / "catalog.csv")
    if settings.skip_paper_like_artifacts:
        LOGGER.info("[%s] skip paper-like artifacts by request", dataset)
    else:
        LOGGER.info("[%s] write paper-like artifacts", dataset)
        _write_datamaps(
            output_dir / "datamaps.json",
            dataset=dataset,
            user2internal=user2internal,
            item2internal=item2internal,
            reconstruction_status=reconstruction_status,
            format_status="reconstructed_format",
            split_type=split_type,
        )
        _write_meta_json(output_dir / "meta.json", catalog)
        _write_text_lines(output_dir / "sequential_data.txt", sequential_lines)
        _write_text_lines(output_dir / "negative_samples.txt", negative_sample_lines)
    _write_readme(
        output_dir / "README_RECONSTRUCTED.md",
        dataset=dataset,
        input_files=input_files,
        split_type=split_type,
        reconstruction_status=reconstruction_status,
        exact_original_artifact_recovered=exact_original_artifact_recovered,
        limitations=[
            "Generated from public raw files, not from the official paper processed artifacts.",
            (
                "Candidate negatives were sampled deterministically with the Plan C "
                f"config seed from `{candidate_item_universe}`."
            ),
            "Paper equivalence is not claimed without author artifacts or exact preprocessing evidence.",
        ],
    )


def _blocked_result(
    settings: ReconstructionSettings,
    dataset: str,
    dataset_config: dict[str, Any],
    exc: ReconstructionBlocked,
) -> DatasetResult:
    summary = _summary_row(
        settings,
        dataset=dataset,
        status=exc.status,
        diagnostic_subset=False,
        input_files_found=False,
        users=0,
        items=0,
        raw_interactions=0,
        eligible_users=0,
        excluded_users=0,
        train_interactions=0,
        ground_truth_rows=0,
        candidate_rows=0,
        catalog_rows=0,
        split_type=str(dataset_config.get("split_type", "")),
        artifact_status="not_written",
        exact_original_artifact_recovered="no",
        reconstruction_status="blocked_missing_raw" if "missing" in exc.status else "blocked_unclear_protocol",
        output_dir=settings.output_base_dir / dataset,
        notes=str(exc),
    )
    blocker = {
        "dataset": dataset,
        "blocker_type": exc.status,
        "missing_inputs": ";".join(exc.missing_inputs),
        "affected_artifacts": ";".join(exc.affected_artifacts),
        "requires_author_contact": bool(exc.requires_author_contact),
        "notes": str(exc),
    }
    stats = _stats_row_from_summary(summary, raw_users=0, raw_items=0)
    artifacts = [
        {
            "dataset": dataset,
            "artifact_name": artifact,
            "path": _display_path(settings.output_base_dir / dataset / artifact, settings.root),
            "written": False,
            "rows_or_lines": 0,
            "artifact_status": "not_written",
            "format_status": "not_applicable",
            "notes": str(exc),
        }
        for artifact in (exc.affected_artifacts or _canonical_artifact_names(include_split=True))
    ]
    return DatasetResult(dataset, exc.status, summary, artifacts, [blocker], stats)


def _write_diagnostics(
    settings: ReconstructionSettings,
    results: list[DatasetResult],
) -> dict[str, pd.DataFrame]:
    summary_df = pd.DataFrame([result.summary for result in results], columns=SUMMARY_COLUMNS)
    artifacts_df = pd.DataFrame(
        [row for result in results for row in result.artifacts],
        columns=INVENTORY_COLUMNS,
    )
    blockers_df = pd.DataFrame(
        [row for result in results for row in result.blockers],
        columns=BLOCKER_COLUMNS,
    )
    stats_df = pd.DataFrame([result.stats for result in results], columns=STATS_COLUMNS)
    _write_csv(summary_df, settings.tables_dir / "plan_c_reconstructed_dataset_summary.csv")
    _write_csv(artifacts_df, settings.tables_dir / "plan_c_reconstructed_artifact_inventory.csv")
    _write_csv(blockers_df, settings.tables_dir / "plan_c_reconstructed_dataset_blockers.csv")
    _write_csv(stats_df, settings.tables_dir / "plan_c_reconstructed_dataset_stats.csv")
    return {
        "summary": summary_df,
        "artifacts": artifacts_df,
        "blockers": blockers_df,
        "stats": stats_df,
    }


def _write_datamaps(
    path: Path,
    *,
    dataset: str,
    user2internal: dict[str, str],
    item2internal: dict[str, str],
    reconstruction_status: str,
    format_status: str,
    split_type: str,
) -> None:
    id2user = {internal: original for original, internal in user2internal.items()}
    id2item = {internal: original for original, internal in item2internal.items()}
    payload = {
        "user2id": user2internal,
        "item2id": item2internal,
        "id2user": id2user,
        "id2item": id2item,
        "attribute2id": {},
        "id2attribute": {},
        "attributeid2num": {},
        "reconstruction_metadata": {
            "dataset": dataset,
            "plan": "Plan C",
            "format_status": format_status,
            "reconstruction_status": reconstruction_status,
            "split_type": split_type,
            "exact_original_artifact_recovered": "no",
            "notes": "Paper-like reconstructed datamap; not the official original processed artifact.",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_meta_json(path: Path, catalog: pd.DataFrame) -> None:
    lines: list[str] = []
    ordered = catalog.sort_values(
        "internal_item_id",
        key=lambda series: series.astype(str).map(_stable_id_sort_tuple),
        kind="mergesort",
    )
    for _, row in ordered.iterrows():
        categories = str(row.get("categories", ""))
        meta_record = {
            "asin": str(row["item_id"]),
            "title": str(row.get("title", "")),
            "categories": [categories.split(" | ") if categories else []],
            "description": str(row.get("description", "")),
            "brand": str(row.get("brand", "")),
            "reconstructed_format": True,
        }
        lines.append(repr(meta_record))
    path.write_text(json.dumps("\n".join(lines) + "\n"), encoding="utf-8")


def _write_readme(
    path: Path,
    *,
    dataset: str,
    input_files: list[Path],
    split_type: str,
    reconstruction_status: str,
    exact_original_artifact_recovered: str,
    limitations: list[str],
) -> None:
    input_lines = "\n".join(f"- `{_display_path(file, PROJECT_ROOT)}`" for file in input_files)
    limitation_lines = "\n".join(f"- {limitation}" for limitation in limitations)
    content = f"""# Plan C Reconstructed Dataset: {dataset}

This directory was generated by `src/data/reconstruct_plan_c_datasets.py`.

## Status

- Reconstruction status: `{reconstruction_status}`
- Exact original artifact recovered: `{exact_original_artifact_recovered}`
- Split type: `{split_type}`

## Input Files

{input_lines}

## Limitations

{limitation_lines}

These files are reconstructed Plan C artifacts. They must not be reported as
the official processed artifacts from *Beyond Utility: Evaluating LLM as
Recommender* unless author-provided files and validation evidence are recovered
later.
"""
    path.write_text(content, encoding="utf-8")


def _artifact_inventory_for_written_dataset(
    settings: ReconstructionSettings,
    dataset: str,
    output_dir: Path,
    *,
    include_split: bool,
    counts: dict[str, int],
) -> list[dict[str, Any]]:
    artifact_names = ["interactions.csv", "catalog.csv", "datamaps.json", "meta.json", "README_RECONSTRUCTED.md"]
    if include_split:
        artifact_names.extend(["ground_truth.csv", "candidates.csv", "sequential_data.txt", "negative_samples.txt"])
    else:
        artifact_names.extend(["ground_truth.csv", "candidates.csv", "sequential_data.txt", "negative_samples.txt"])
    rows = []
    for artifact in artifact_names:
        path = output_dir / artifact
        written = path.exists()
        rows.append(
            {
                "dataset": dataset,
                "artifact_name": artifact,
                "path": _display_path(path, settings.root),
                "written": written,
                "rows_or_lines": int(counts.get(artifact, 1) if written else 0),
                "artifact_status": "written_reconstructed" if written else "not_written",
                "format_status": "reconstructed_format" if written else "not_applicable",
                "notes": _artifact_inventory_note(settings, artifact, written),
            }
        )
    return rows


def _artifact_inventory_note(settings: ReconstructionSettings, artifact: str, written: bool) -> str:
    if written:
        return "Generated Plan C reconstruction artifact; not an exact original paper file."
    if settings.skip_paper_like_artifacts and artifact in _paper_like_artifact_names():
        return "Skipped by --skip-paper-like-artifacts development option."
    return "Not written because this dataset lacks a safe chronological split in C5."


def _paper_like_artifact_names() -> set[str]:
    return {"datamaps.json", "meta.json", "sequential_data.txt", "negative_samples.txt"}


def _summary_row(
    settings: ReconstructionSettings,
    *,
    dataset: str,
    status: str,
    diagnostic_subset: bool,
    input_files_found: bool,
    users: int,
    items: int,
    raw_interactions: int,
    eligible_users: int,
    excluded_users: int,
    train_interactions: int,
    ground_truth_rows: int,
    candidate_rows: int,
    catalog_rows: int,
    split_type: str,
    artifact_status: str,
    exact_original_artifact_recovered: str,
    reconstruction_status: str,
    output_dir: Path,
    notes: str,
    candidate_item_universe: str = "",
    candidate_universe_size: int = 0,
    catalog_size: int = 0,
    interaction_item_count: int = 0,
    users_with_insufficient_negatives: int = 0,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "status": status,
        "diagnostic_subset": bool(diagnostic_subset),
        "max_users": int(settings.max_users) if settings.max_users is not None else "",
        "input_files_found": bool(input_files_found),
        "users": int(users),
        "items": int(items),
        "raw_interactions": int(raw_interactions),
        "eligible_users": int(eligible_users),
        "excluded_users": int(excluded_users),
        "train_interactions": int(train_interactions),
        "ground_truth_rows": int(ground_truth_rows),
        "candidate_rows": int(candidate_rows),
        "catalog_rows": int(catalog_rows),
        "negative_samples_per_user": int(settings.negatives_per_user),
        "candidate_item_universe": candidate_item_universe,
        "candidate_universe_size": int(candidate_universe_size),
        "catalog_size": int(catalog_size if catalog_size else catalog_rows),
        "interaction_item_count": int(interaction_item_count),
        "users_with_insufficient_negatives": int(users_with_insufficient_negatives),
        "split_type": split_type,
        "artifact_status": artifact_status,
        "exact_original_artifact_recovered": exact_original_artifact_recovered,
        "reconstruction_status": reconstruction_status,
        "output_dir": _display_path(output_dir, settings.root),
        "notes": notes,
    }


def _stats_row_from_summary(summary: dict[str, Any], *, raw_users: int, raw_items: int) -> dict[str, Any]:
    return {
        "dataset": summary["dataset"],
        "diagnostic_subset": bool(summary["diagnostic_subset"]),
        "max_users": summary["max_users"],
        "raw_users": int(raw_users),
        "raw_items": int(raw_items),
        "raw_interactions": int(summary["raw_interactions"]),
        "eligible_users": int(summary["eligible_users"]),
        "excluded_users": int(summary["excluded_users"]),
        "train_interactions": int(summary["train_interactions"]),
        "ground_truth_rows": int(summary["ground_truth_rows"]),
        "candidate_rows": int(summary["candidate_rows"]),
        "catalog_rows": int(summary["catalog_rows"]),
        "min_interactions_per_user": "",
        "negative_samples_per_user": int(summary["negative_samples_per_user"]),
        "candidate_item_universe": summary["candidate_item_universe"],
        "candidate_universe_size": int(summary["candidate_universe_size"]),
        "catalog_size": int(summary["catalog_size"]),
        "interaction_item_count": int(summary["interaction_item_count"]),
        "users_with_insufficient_negatives": int(summary["users_with_insufficient_negatives"]),
        "split_type": summary["split_type"],
        "reconstruction_status": summary["reconstruction_status"],
    }


def _ensure_catalog_contains_items(catalog: pd.DataFrame, item_ids: Iterable[Any]) -> pd.DataFrame:
    catalog = catalog.copy()
    if catalog.empty:
        catalog = pd.DataFrame(columns=CATALOG_COLUMNS[:-1])
    existing = set(catalog["item_id"].astype(str)) if "item_id" in catalog.columns else set()
    missing_rows = []
    for item_id in _stable_sorted_values(list(item_ids)):
        if item_id not in existing:
            missing_rows.append(
                {
                    "item_id": item_id,
                    "title": "",
                    "categories": "",
                    "description": "",
                    "brand": "",
                    "normalized_text": "",
                }
            )
    if missing_rows:
        catalog = pd.concat([catalog, pd.DataFrame(missing_rows)], ignore_index=True)
    for column in CATALOG_COLUMNS[:-1]:
        if column not in catalog.columns:
            catalog[column] = ""
    catalog["item_id"] = catalog["item_id"].astype(str)
    catalog = catalog.drop_duplicates("item_id", keep="first")
    catalog["normalized_text"] = catalog.apply(
        lambda row: _normalize_text(
            " ".join(
                [
                    str(row.get("title", "")),
                    str(row.get("categories", "")),
                    str(row.get("description", "")),
                    str(row.get("brand", "")),
                ]
            )
        ),
        axis=1,
    )
    return catalog.loc[:, CATALOG_COLUMNS[:-1]]


def _assign_internal_items(catalog: pd.DataFrame) -> pd.DataFrame:
    catalog = catalog.copy()
    catalog["item_id"] = catalog["item_id"].astype(str)
    catalog = catalog.sort_values(
        "item_id",
        key=lambda series: series.astype(str).map(_stable_id_sort_tuple),
        kind="mergesort",
    ).reset_index(drop=True)
    catalog["internal_item_id"] = [str(index + 1) for index in range(len(catalog))]
    return catalog.loc[:, CATALOG_COLUMNS]


def _build_internal_map(values: Iterable[Any]) -> dict[str, str]:
    return {
        value: str(index + 1)
        for index, value in enumerate(_stable_sorted_values(list(values)))
    }


def _build_sequential_lines(
    eligible: pd.DataFrame,
    user2internal: dict[str, str],
    item2internal: dict[str, str],
) -> list[str]:
    if eligible.empty:
        return []

    eligible_sorted = eligible.copy()
    eligible_sorted["__user_id_str"] = eligible_sorted["user_id"].astype(str)
    eligible_sorted["__item_id_str"] = eligible_sorted["item_id"].astype(str)
    eligible_sorted = eligible_sorted.sort_values(
        ["__user_id_str", "timestamp", "__item_id_str", "__input_order"],
        kind="mergesort",
    )
    items_by_user: dict[str, list[str]] = {}
    for user_id, group in eligible_sorted.groupby("__user_id_str", sort=False):
        items_by_user[str(user_id)] = [
            item2internal[str(item_id)]
            for item_id in group["__item_id_str"].tolist()
            if str(item_id) in item2internal
        ]

    ordered_users = sorted(user2internal.items(), key=lambda pair: _stable_id_sort_tuple(pair[1]))
    lines: list[str] = []
    for user_id, internal_user in ordered_users:
        item_ids = items_by_user.get(str(user_id), [])
        lines.append(" ".join([internal_user, *item_ids]))
    return lines


def _parse_record_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        value = json.loads(line)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        value = ast.literal_eval(line)
        return value if isinstance(value, dict) else None
    except (ValueError, SyntaxError):
        return None


def _find_zip_member(names: list[str], expected: str) -> str | None:
    expected_norm = expected.replace("\\", "/").lower()
    for name in names:
        if name.replace("\\", "/").lower() == expected_norm:
            return name
    expected_name = Path(expected).name.lower()
    for name in names:
        normalized = name.replace("\\", "/").lower()
        if normalized.endswith("/" + expected_name) or normalized == expected_name:
            return name
    return None


def _stable_sorted_values(values: Iterable[Any]) -> list[str]:
    return sorted((str(value) for value in values), key=_stable_id_sort_tuple)


def _stable_id_sort_tuple(value: Any) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _user_seed(seed: int, user_id: str) -> int:
    digest = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + int(seed)) % (2**32)


def _stringify_categories(value: Any) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return " | ".join(str(part) for part in value[0] if _has_value(part))
        return " | ".join(str(part) for part in value if _has_value(part))
    return _clean_text(value)


def _stringify_description(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_clean_text(part) for part in value if _has_value(part))
    return _clean_text(value)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _canonical_artifact_names(*, include_split: bool) -> list[str]:
    base = ["interactions.csv", "catalog.csv", "datamaps.json", "meta.json", "README_RECONSTRUCTED.md"]
    if include_split:
        base.extend(["ground_truth.csv", "candidates.csv", "sequential_data.txt", "negative_samples.txt"])
    return base


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _write_text_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _ensure_no_api(config: dict[str, Any]) -> None:
    if config.get("api", {}).get("allow_api_calls"):
        raise ConfigError("api.allow_api_calls must remain false for Plan C reconstruction.")
    if config.get("experiment", {}).get("allow_api_calls"):
        raise ConfigError("experiment.allow_api_calls must remain false for Plan C reconstruction.")


def _resolve_input_path(path: Path, root: Path) -> Path:
    if path.is_absolute():
        resolved = path.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ConfigError(f"Config path must be inside the project root: {path}") from exc
        return resolved
    return (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    try:
        frames = reconstruct_plan_c_datasets(
            args.config,
            datasets=args.datasets,
            max_users=args.max_users,
            skip_paper_like_artifacts=args.skip_paper_like_artifacts,
        )
    except (ConfigError, OSError, ValueError, pd.errors.ParserError) as exc:
        LOGGER.error("%s", exc)
        return 2
    summary = frames["summary"]
    print("Plan C reconstruction complete")
    for _, row in summary.iterrows():
        print(
            f"{row['dataset']}: status={row['status']} "
            f"diagnostic_subset={row['diagnostic_subset']} "
            f"train={row['train_interactions']} gt={row['ground_truth_rows']} "
            f"candidates={row['candidate_rows']} catalog={row['catalog_rows']} "
            f"reconstruction_status={row['reconstruction_status']}"
        )
    print("metrics_written=false")
    print("summary=results/tables/plan_c_reconstructed_dataset_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

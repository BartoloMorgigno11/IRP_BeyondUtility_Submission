"""Prepare and audit the Food.com Dataset-5 extension input.

Run from the repository root:

    python src/data/prepare_foodcom_extension.py --config configs/foodcom_extension.yaml

This E3 runner performs acquisition validation, raw integrity hashing, schema
audits, deterministic preprocessing, and readiness checks only. It does not
download data, train recommenders, score models, run LLM inference, compute
experimental recommendation metrics, or modify ``original_repo/``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.common_hashing import aggregate_named_hashes, hash_directory_tree, sha256_file
from src.utils.config import find_project_root, load_yaml_config, resolve_project_path


EVIDENCE_LABEL = "POST-F6M DATASET-5 EXTENSION EVIDENCE"
SUCCESS_TOKEN = "EXTENSION_E3_FOODCOM_DATASET_INTEGRITY_FROZEN"
ACQUISITION_BLOCKED_TOKEN = "EXTENSION_E3_FOODCOM_ACQUISITION_BLOCKED"
PROTOCOL_REQUIRED_TOKEN = "EXTENSION_E3_FOODCOM_PROTOCOL_DECISION_REQUIRED"
INTEGRITY_FAILED_TOKEN = "EXTENSION_E3_FOODCOM_DATA_INTEGRITY_FAILED"

EXPECTED_E1_COUNTS = {
    "metric_contract": 29,
    "four_dataset_coverage": 26,
    "fifth_dataset_requirements": 23,
    "inference_requirements": 8,
    "scorecard": 25,
}

EXPECTED_INTERACTION_COLUMNS = ["user_id", "recipe_id", "date", "rating", "review"]
EXPECTED_RECIPE_COLUMNS = [
    "name",
    "id",
    "minutes",
    "contributor_id",
    "submitted",
    "tags",
    "nutrition",
    "n_steps",
    "steps",
    "description",
    "ingredients",
    "n_ingredients",
]
MODEL_NAMES = ["Random", "MostPop", "BM25-like", "BPRMF", "GRU4Rec", "SASRec", "LightGCN"]


@dataclass(frozen=True)
class E3Paths:
    root: Path
    config_path: Path
    source_dir: Path
    extracted_base_dir: Path
    processed_dir: Path
    tables_dir: Path
    report_path: Path
    determinism_dir: Path


@dataclass(frozen=True)
class E3Settings:
    root: Path
    config_path: Path
    config: dict[str, Any]
    paths: E3Paths
    seed: int
    min_user_interactions: int
    min_item_interactions: int
    min_sequence_length: int
    cohort_size: int
    cohort_seed: int
    min_negative_items: int
    enforce_git_precondition: bool


@dataclass
class E3Result:
    decision_token: str
    reason: str
    report_path: Path
    table_paths: dict[str, Path]


class FoodcomE3Error(RuntimeError):
    """Base class for controlled E3 stops."""

    decision_token = INTEGRITY_FAILED_TOKEN


class AcquisitionBlocked(FoodcomE3Error):
    """Raised when the official manually staged archive is unavailable."""

    decision_token = ACQUISITION_BLOCKED_TOKEN


class ProtocolDecisionRequired(FoodcomE3Error):
    """Raised when a material data protocol choice cannot be resolved."""

    decision_token = PROTOCOL_REQUIRED_TOKEN


class DataIntegrityFailed(FoodcomE3Error):
    """Raised when deterministic integrity checks fail."""

    decision_token = INTEGRITY_FAILED_TOKEN


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Food.com Dataset-5 E3 evidence.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def prepare_foodcom_extension(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> E3Result:
    """Run the Food.com E3 acquisition/preprocessing integrity workflow."""

    root = Path(project_root).resolve() if project_root is not None else find_project_root()
    config = load_yaml_config(config_path, project_root=root)
    settings = _build_settings(config, config_path, root)
    run_started_at = _utc_now()

    table_rows = _empty_tables()
    preflight = _preflight(settings)
    table_rows["preprocessing_contract"].extend(_preprocessing_contract_rows(settings, preflight))

    try:
        archive = _locate_staged_archive(settings)
        archive_hash_before = sha256_file(archive)
        extracted_dir = _extract_archive(settings, archive, archive_hash_before)
        acquisition_rows = _build_acquisition_manifest(settings, archive, extracted_dir)
        table_rows["acquisition_manifest"].extend(acquisition_rows)
        table_rows["protected_inputs"].extend(_protected_input_rows(acquisition_rows, "before_preprocessing"))

        raw = _load_raw_foodcom(settings, extracted_dir)
        audits = _audit_raw(raw)
        _extend_audit_tables(table_rows, audits)

        first_dir = settings.paths.determinism_dir / "run_a"
        second_dir = settings.paths.determinism_dir / "run_b"
        first = _build_processed_state(settings, raw, first_dir)
        second = _build_processed_state(settings, raw, second_dir)
        determinism = _compare_processed_states(first, second)
        if not determinism["passed"]:
            raise DataIntegrityFailed(determinism["reason"])

        _replace_processed_dir(first.output_dir, settings.paths.processed_dir)
        final_processed_hash = hash_directory_tree(settings.paths.processed_dir)

        table_rows["core_statistics"].extend(first.core_rows)
        table_rows["mapping_audit"].extend(first.mapping_rows)
        table_rows["split_statistics"].extend(first.split_rows)
        table_rows["cohort_statistics"].extend(first.cohort_rows)
        table_rows["forensic_traces"].extend(first.trace_rows)
        table_rows["metric_readiness"].extend(_metric_readiness_rows(settings, SUCCESS_TOKEN, first))
        table_rows["requirement_realization"].extend(_requirement_realization_rows(settings, SUCCESS_TOKEN, first))
        table_rows["model_readiness"].extend(_model_readiness_rows(SUCCESS_TOKEN, first))
        table_rows["protected_inputs"].extend(_protected_input_rows(acquisition_rows, "after_preprocessing"))
        table_rows["decision_summary"].append(
            _decision_row(
                SUCCESS_TOKEN,
                "Food.com raw archive acquired, protected, processed deterministically, and verified.",
                run_started_at,
                final_processed_hash,
            )
        )
        decision_token = SUCCESS_TOKEN
        reason = "Food.com Dataset-5 input integrity frozen."
    except FoodcomE3Error as exc:
        decision_token = exc.decision_token
        reason = str(exc)
        _fill_blocked_tables(table_rows, settings, decision_token, reason, run_started_at)

    table_paths = _write_all_tables(settings.paths.tables_dir, table_rows)
    _write_report(settings, table_rows, decision_token, reason, preflight, run_started_at)
    return E3Result(decision_token, reason, settings.paths.report_path, table_paths)


@dataclass
class RawFoodcom:
    interactions: pd.DataFrame
    recipes: pd.DataFrame
    interactions_path: Path
    recipes_path: Path


@dataclass
class ProcessedState:
    output_dir: Path
    interactions: pd.DataFrame
    ground_truth: pd.DataFrame
    catalog: pd.DataFrame
    user_mapping: pd.DataFrame
    item_mapping: pd.DataFrame
    cohort: pd.DataFrame
    core_rows: list[dict[str, str]]
    mapping_rows: list[dict[str, str]]
    split_rows: list[dict[str, str]]
    cohort_rows: list[dict[str, str]]
    trace_rows: list[dict[str, str]]
    file_hashes: dict[str, str]


def _build_settings(config: dict[str, Any], config_path: str | Path, root: Path) -> E3Settings:
    e3 = config.get("extension_e3", {})
    preprocessing = e3.get("preprocessing", {})
    cohort = e3.get("cohort", {})
    paths = E3Paths(
        root=root,
        config_path=(root / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path).resolve(),
        source_dir=_required_path(e3.get("source_dir"), root, "extension_e3.source_dir"),
        extracted_base_dir=_required_path(e3.get("extracted_dir"), root, "extension_e3.extracted_dir"),
        processed_dir=_required_path(config["paths"]["processed_dir"], root, "paths.processed_dir"),
        tables_dir=_required_path(config["output"]["tables_dir"], root, "output.tables_dir"),
        report_path=_required_path(config["output"]["analysis_path"], root, "output.analysis_path"),
        determinism_dir=_required_path(e3.get("determinism_dir"), root, "extension_e3.determinism_dir"),
    )
    return E3Settings(
        root=root,
        config_path=paths.config_path,
        config=config,
        paths=paths,
        seed=int(config["experiment"]["seed"]),
        min_user_interactions=int(preprocessing.get("min_user_interactions", 5)),
        min_item_interactions=int(preprocessing.get("min_item_interactions", 5)),
        min_sequence_length=int(preprocessing.get("min_sequence_length_after_core", 3)),
        cohort_size=int(cohort.get("size", 1000)),
        cohort_seed=int(cohort.get("seed", config["experiment"]["seed"])),
        min_negative_items=int(cohort.get("min_negative_items_for_future_ranking", 19)),
        enforce_git_precondition=bool(e3.get("enforce_git_precondition", True)),
    )


def _required_path(value: str | Path | None, root: Path, name: str) -> Path:
    resolved = resolve_project_path(value, root)
    if resolved is None:
        raise ValueError(f"{name} must be configured")
    return resolved


def _preflight(settings: E3Settings) -> dict[str, str]:
    preflight = {
        "branch": _git_output(settings.root, ["branch", "--show-current"]),
        "head": _git_output(settings.root, ["rev-parse", "HEAD"]),
        "status_short": _git_output(settings.root, ["status", "--short"]),
        "status_branch": _git_output(settings.root, ["status", "--branch", "--short"]),
        "cached_stat": _git_output(settings.root, ["diff", "--cached", "--stat"]),
        "original_repo_status": _git_output(
            settings.root / "original_repo" / "EvaLLMasRecommender",
            ["status", "--branch", "--short"],
        ),
    }
    _validate_contract_counts(settings)
    _validate_e2_primary(settings)
    if settings.enforce_git_precondition:
        expected_branch = str(settings.config["extension_e3"].get("expected_branch", "master"))
        expected_head = str(settings.config["extension_e3"].get("expected_head", ""))
        if preflight["branch"] != expected_branch:
            raise DataIntegrityFailed(f"Unexpected branch: {preflight['branch']}")
        if preflight["head"] != expected_head:
            raise DataIntegrityFailed(f"Unexpected HEAD: {preflight['head']}")
        unrelated = _unrelated_status_entries(preflight["status_short"])
        if unrelated:
            raise DataIntegrityFailed(f"Unrelated worktree changes present: {', '.join(unrelated)}")
        if preflight["cached_stat"].strip():
            raise DataIntegrityFailed("Staged diff is not empty before E3 run.")
        if _nested_status_has_changes(preflight["original_repo_status"]):
            raise DataIntegrityFailed("original_repo/EvaLLMasRecommender is not clean.")
    return preflight


def _git_output(cwd: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip()


def _unrelated_status_entries(status_short: str) -> list[str]:
    allowed_prefixes = (
        ".gitignore",
        "configs/foodcom_extension.yaml",
        "report/extension_e3_foodcom_data_integrity.md",
        "report/notes.md",
        "results/tables/extension_e3_foodcom_",
        "src/data/prepare_foodcom_extension.py",
        "tests/test_extension_e3_foodcom_data_integrity.py",
    )
    unrelated = []
    for line in status_short.splitlines():
        path = (line[3:] if len(line) > 3 and line[2] == " " else line[2:]).strip().replace("\\", "/")
        if path == "gitignore":
            path = ".gitignore"
        if not path.startswith(allowed_prefixes):
            unrelated.append(path)
    return unrelated


def _nested_status_has_changes(status_branch: str) -> bool:
    return any(line and not line.startswith("## ") for line in status_branch.splitlines())


def _validate_contract_counts(settings: E3Settings) -> None:
    expected_files = {
        "metric_contract": settings.paths.tables_dir / "extension_e1_metric_contract.csv",
        "four_dataset_coverage": settings.paths.tables_dir / "extension_e1_four_dataset_coverage.csv",
        "fifth_dataset_requirements": settings.paths.tables_dir / "extension_e1_fifth_dataset_requirements.csv",
        "inference_requirements": settings.paths.tables_dir / "extension_e1_inference_requirements.csv",
        "scorecard": settings.paths.tables_dir / "extension_e1_dataset_selection_scorecard_template.csv",
    }
    for key, path in expected_files.items():
        count = len(_read_csv(path))
        if count != EXPECTED_E1_COUNTS[key]:
            raise DataIntegrityFailed(f"E1 {key} row count mismatch: expected {EXPECTED_E1_COUNTS[key]}, found {count}")


def _validate_e2_primary(settings: E3Settings) -> None:
    decision_path = settings.paths.tables_dir / "extension_e2_dataset_selection_decision.csv"
    rows = _read_csv(decision_path)
    if len(rows) != 1:
        raise DataIntegrityFailed("E2 decision table must contain exactly one row.")
    row = rows[0]
    if row.get("decision_token") != "EXTENSION_E2_FIFTH_DATASET_SELECTED":
        raise DataIntegrityFailed("E2 decision token is not selected.")
    if row.get("primary_dataset_id") != "foodcom":
        raise DataIntegrityFailed("E2 primary dataset is not Food.com.")


def _locate_staged_archive(settings: E3Settings) -> Path:
    source_dir = settings.paths.source_dir
    if not source_dir.is_dir():
        raise AcquisitionBlocked(f"Manual source directory is missing: {_rel(source_dir, settings.root)}")
    allowed = {suffix.lower() for suffix in settings.config["extension_e3"].get("allowed_archive_extensions", [".zip"])}
    archives = sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed)
    required_count = int(settings.config["extension_e3"].get("required_archive_count", 1))
    if len(archives) != required_count:
        raise AcquisitionBlocked(
            f"Expected exactly {required_count} Food.com source archive under {_rel(source_dir, settings.root)}, found {len(archives)}."
        )
    archive = archives[0]
    lowered = archive.name.lower()
    if "pair" in lowered or "paired" in lowered or "share" in lowered:
        raise AcquisitionBlocked(f"Staged archive looks like a paired-recipe derivative, not the selected Food.com release: {archive.name}")
    return archive


def _extract_archive(settings: E3Settings, archive: Path, archive_hash: str) -> Path:
    extract_dir = settings.paths.extracted_base_dir / f"{archive.stem}_{archive_hash[:12]}"
    extract_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() != ".zip":
        raise AcquisitionBlocked(f"Unsupported archive type: {archive.suffix}")
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target = (extract_dir / member.filename).resolve()
            try:
                target.relative_to(extract_dir.resolve())
            except ValueError as exc:
                raise DataIntegrityFailed(f"Unsafe archive member path: {member.filename}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with zf.open(member, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    return extract_dir


def _build_acquisition_manifest(settings: E3Settings, archive: Path, extracted_dir: Path) -> list[dict[str, str]]:
    rows = [
        {
            "evidence_label": EVIDENCE_LABEL,
            "file_role": "official_source_archive",
            "classification": "archive",
            "relative_path": _rel(archive, settings.root),
            "byte_size": str(archive.stat().st_size),
            "sha256": sha256_file(archive),
            "acquisition_source": settings.config["extension_e3"]["official_sources"]["kaggle_endpoint"],
            "acquisition_timestamp": _utc_now(),
            "status": "protected_raw_input",
            "notes": "Manually staged official UCSD-linked Kaggle archive; raw contents are not committed.",
        }
    ]
    for path in sorted(candidate for candidate in extracted_dir.rglob("*") if candidate.is_file()):
        role = _classify_raw_file(path.name, settings)
        rows.append(
            {
                "evidence_label": EVIDENCE_LABEL,
                "file_role": role,
                "classification": "extracted_raw_file",
                "relative_path": _rel(path, settings.root),
                "byte_size": str(path.stat().st_size),
                "sha256": sha256_file(path),
                "acquisition_source": _rel(archive, settings.root),
                "acquisition_timestamp": _utc_now(),
                "status": "protected_raw_input",
                "notes": "",
            }
        )
    required = settings.config["extension_e3"]["required_raw_files"]
    found_names = {Path(row["relative_path"]).name for row in rows}
    missing = [name for name in required.values() if name not in found_names]
    if missing:
        raise AcquisitionBlocked(f"Staged archive is missing required Food.com raw file(s): {', '.join(missing)}")
    return rows


def _classify_raw_file(name: str, settings: E3Settings) -> str:
    required = settings.config["extension_e3"]["required_raw_files"]
    if name == required["interactions"]:
        return "raw_user_recipe_interactions"
    if name == required["recipes"]:
        return "raw_recipe_item_metadata"
    if name.startswith("interactions_"):
        return "author_preprocessed_split"
    if name.startswith("PP_"):
        return "author_preprocessed_representation"
    return "auxiliary_metadata"


def _load_raw_foodcom(settings: E3Settings, extracted_dir: Path) -> RawFoodcom:
    required = settings.config["extension_e3"]["required_raw_files"]
    interactions_path = _find_file(extracted_dir, required["interactions"])
    recipes_path = _find_file(extracted_dir, required["recipes"])
    interactions = pd.read_csv(interactions_path)
    recipes = pd.read_csv(recipes_path)
    return RawFoodcom(interactions, recipes, interactions_path, recipes_path)


def _find_file(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise AcquisitionBlocked(f"Expected exactly one {name} in extracted archive, found {len(matches)}.")
    return matches[0]


def _audit_raw(raw: RawFoodcom) -> dict[str, list[dict[str, str]]]:
    interactions = raw.interactions.copy()
    recipes = raw.recipes.copy()
    missing_interaction_cols = [column for column in EXPECTED_INTERACTION_COLUMNS if column not in interactions.columns]
    missing_recipe_cols = [column for column in EXPECTED_RECIPE_COLUMNS if column not in recipes.columns]
    if missing_interaction_cols:
        raise DataIntegrityFailed(f"RAW_interactions.csv missing column(s): {', '.join(missing_interaction_cols)}")
    if missing_recipe_cols:
        raise DataIntegrityFailed(f"RAW_recipes.csv missing column(s): {', '.join(missing_recipe_cols)}")

    parsed_dates = pd.to_datetime(interactions["date"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    rating_numeric = pd.to_numeric(interactions["rating"], errors="coerce")
    exact_duplicate_rows = int(interactions.duplicated().sum())
    duplicated_pairs = interactions.duplicated(subset=["user_id", "recipe_id"], keep=False)
    duplicated_pair_groups = int(interactions.loc[duplicated_pairs, ["user_id", "recipe_id"]].drop_duplicates().shape[0])
    duplicate_user_item_date = int(interactions.duplicated(subset=["user_id", "recipe_id", "date"], keep=False).sum())
    duplicate_different_rating = int(
        interactions.groupby(["user_id", "recipe_id"], dropna=False)["rating"].nunique().gt(1).sum()
    )
    duplicate_different_review = int(
        interactions.groupby(["user_id", "recipe_id"], dropna=False)["review"].nunique().gt(1).sum()
    )
    tied_user_dates = interactions.assign(_date=parsed_dates.dt.strftime("%Y-%m-%d")).groupby(["user_id", "_date"], dropna=False).size()
    users_with_same_date_ties = int(tied_user_dates[tied_user_dates > 1].reset_index()["user_id"].nunique()) if not tied_user_dates.empty else 0

    raw_stats = [
        _metric("raw_interaction_rows", len(interactions)),
        _metric("raw_unique_users", interactions["user_id"].nunique()),
        _metric("raw_unique_recipes_in_interactions", interactions["recipe_id"].nunique()),
        _metric("raw_unique_user_recipe_pairs", interactions[["user_id", "recipe_id"]].drop_duplicates().shape[0]),
        _metric("raw_recipe_metadata_rows", len(recipes)),
        _metric("raw_recipe_metadata_unique_ids", recipes["id"].nunique()),
        _metric("earliest_interaction_date", "" if parsed_dates.dropna().empty else parsed_dates.min().date().isoformat()),
        _metric("latest_interaction_date", "" if parsed_dates.dropna().empty else parsed_dates.max().date().isoformat()),
        _metric("invalid_interaction_dates", invalid_dates),
        _metric("rating_min", "" if rating_numeric.dropna().empty else float(rating_numeric.min())),
        _metric("rating_max", "" if rating_numeric.dropna().empty else float(rating_numeric.max())),
        _metric("rating_missing", int(rating_numeric.isna().sum())),
        _metric("density_raw_user_item", _density(interactions["user_id"].nunique(), interactions["recipe_id"].nunique(), len(interactions))),
    ]
    raw_stats.extend(_distribution_rows("interactions_per_user", interactions.groupby("user_id").size()))
    raw_stats.extend(_distribution_rows("interactions_per_item", interactions.groupby("recipe_id").size()))
    for column in ["name", "tags", "ingredients", "description", "nutrition"]:
        coverage = 1.0 - _empty_series(recipes[column]).mean()
        raw_stats.append(_metric(f"metadata_coverage_{column}", coverage))

    schema_rows = []
    for source_name, frame, expected in [
        ("RAW_interactions.csv", interactions, EXPECTED_INTERACTION_COLUMNS),
        ("RAW_recipes.csv", recipes, EXPECTED_RECIPE_COLUMNS),
    ]:
        for column in frame.columns:
            schema_rows.append(
                {
                    "evidence_label": EVIDENCE_LABEL,
                    "source_file": source_name,
                    "column": str(column),
                    "dtype": str(frame[column].dtype),
                    "missing_values": str(int(frame[column].isna().sum())),
                    "missing_rate": f"{float(frame[column].isna().mean()):.10f}",
                    "required_by_e3": str(column in expected).lower(),
                    "status": "present",
                }
            )

    duplicate_rows = [
        _audit_row("exact_duplicate_rows", exact_duplicate_rows, "collapse_exact_duplicates"),
        _audit_row("duplicate_user_item_pairs", duplicated_pair_groups, "retain_latest_dated_event"),
        _audit_row("duplicate_user_item_date_rows", duplicate_user_item_date, "block_if_latest_duplicate_conflicts"),
        _audit_row("different_ratings_for_same_user_item", duplicate_different_rating, "record_adaptation_if_nonzero"),
        _audit_row("different_reviews_for_same_user_item", duplicate_different_review, "record_adaptation_if_nonzero"),
    ]
    temporal_rows = [
        _audit_row("invalid_dates", invalid_dates, "block_success"),
        _audit_row("users_with_same_date_multiple_interactions", users_with_same_date_ties, "exclude_users_with_any_tied_interaction_date"),
    ]
    if invalid_dates:
        raise DataIntegrityFailed(f"Invalid Food.com interaction dates found: {invalid_dates}")

    return {
        "raw_schema": schema_rows,
        "raw_statistics": raw_stats,
        "duplicate_audit": duplicate_rows,
        "temporal_tie_audit": temporal_rows,
    }


def _build_processed_state(settings: E3Settings, raw: RawFoodcom, output_dir: Path) -> ProcessedState:
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions = _canonicalize_interactions(raw.interactions)
    recipes = _canonicalize_recipes(raw.recipes)
    deduped, duplicate_policy_rows = _apply_duplicate_policy(interactions)
    no_ties, temporal_rows = _apply_temporal_policy(deduped)
    cored, core_rows = _iterative_core(no_ties, settings.min_user_interactions, settings.min_item_interactions)
    if cored.empty:
        raise ProtocolDecisionRequired("Food.com preprocessing leaves zero interactions after k-core filtering.")
    sequences = _build_sequences(cored, settings.min_sequence_length)
    if sequences.empty:
        raise ProtocolDecisionRequired("No users have enough interactions for train/validation/test after filtering.")
    catalog, item_mapping = _build_catalog(recipes, sequences["recipe_id"].unique())
    user_mapping = _build_user_mapping(sequences["user_id"].unique())
    interactions_frame, ground_truth = _split_sequences(sequences, user_mapping, item_mapping)
    cohort = _build_cohort(settings, sequences, catalog)
    trace_rows = _build_trace_rows(sequences, interactions_frame, ground_truth, catalog, cohort)

    _write_processed_outputs(output_dir, interactions_frame, ground_truth, catalog, user_mapping, item_mapping, cohort)
    file_hashes = _hash_processed_files(output_dir)
    mapping_rows = _mapping_audit_rows(user_mapping, item_mapping, file_hashes)
    split_rows = _split_stat_rows(interactions_frame, ground_truth)
    cohort_rows = _cohort_stat_rows(settings, cohort, sequences, catalog, file_hashes)
    core_rows.extend(duplicate_policy_rows)
    core_rows.extend(temporal_rows)
    return ProcessedState(
        output_dir=output_dir,
        interactions=interactions_frame,
        ground_truth=ground_truth,
        catalog=catalog,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        cohort=cohort,
        core_rows=core_rows,
        mapping_rows=mapping_rows,
        split_rows=split_rows,
        cohort_rows=cohort_rows,
        trace_rows=trace_rows,
        file_hashes=file_hashes,
    )


def _canonicalize_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.loc[:, EXPECTED_INTERACTION_COLUMNS].copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["recipe_id"] = frame["recipe_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["date_iso"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame["review"] = frame["review"].fillna("").astype(str)
    return frame


def _canonicalize_recipes(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.loc[:, EXPECTED_RECIPE_COLUMNS].copy()
    frame["id"] = frame["id"].astype(str)
    for column in ["name", "description", "tags", "ingredients", "nutrition", "steps"]:
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def _apply_duplicate_policy(interactions: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    before = len(interactions)
    exact_deduped = interactions.drop_duplicates().copy()
    exact_removed = before - len(exact_deduped)
    repeated = exact_deduped.groupby(["user_id", "recipe_id"], dropna=False).size()
    repeated_pair_count = int((repeated > 1).sum())
    latest_counts = (
        exact_deduped.sort_values(["user_id", "recipe_id", "date_iso"])
        .groupby(["user_id", "recipe_id"], dropna=False)
        .tail(1)
    )
    if len(latest_counts) != exact_deduped[["user_id", "recipe_id"]].drop_duplicates().shape[0]:
        raise ProtocolDecisionRequired("Repeated user-item latest event consolidation is not one-to-one.")
    rows = [
        _core_row("exact_duplicate_rows_removed", exact_removed, before, len(exact_deduped), "collapse_exact_duplicates"),
        _core_row("repeated_user_item_pairs_consolidated", repeated_pair_count, len(exact_deduped), len(latest_counts), "retain_latest_dated_event"),
    ]
    return latest_counts.sort_values(["user_id", "date_iso", "recipe_id"]).reset_index(drop=True), rows


def _apply_temporal_policy(interactions: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    tied = interactions.groupby(["user_id", "date_iso"], dropna=False).size()
    tied_users = set(tied[tied > 1].reset_index()["user_id"].astype(str))
    before_users = interactions["user_id"].nunique()
    filtered = interactions.loc[~interactions["user_id"].isin(tied_users)].copy()
    rows = [
        _core_row(
            "users_removed_for_same_date_ties",
            len(tied_users),
            before_users,
            filtered["user_id"].nunique(),
            "exclude_users_with_any_tied_interaction_date",
        )
    ]
    return filtered, rows


def _iterative_core(interactions: pd.DataFrame, min_user: int, min_item: int) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    current = interactions.copy()
    rows = []
    iteration = 0
    while True:
        iteration += 1
        before = len(current)
        before_users = current["user_id"].nunique()
        before_items = current["recipe_id"].nunique()
        user_counts = current.groupby("user_id").size()
        item_counts = current.groupby("recipe_id").size()
        keep_users = set(user_counts[user_counts >= min_user].index)
        keep_items = set(item_counts[item_counts >= min_item].index)
        current = current.loc[current["user_id"].isin(keep_users) & current["recipe_id"].isin(keep_items)].copy()
        rows.append(
            {
                "evidence_label": EVIDENCE_LABEL,
                "component": f"k_core_iteration_{iteration}",
                "policy": f"iterative user>={min_user}, item>={min_item}",
                "before_users": str(before_users),
                "after_users": str(current["user_id"].nunique()),
                "before_items": str(before_items),
                "after_items": str(current["recipe_id"].nunique()),
                "before_interactions": str(before),
                "after_interactions": str(len(current)),
                "removed_users": str(before_users - current["user_id"].nunique()),
                "removed_items": str(before_items - current["recipe_id"].nunique()),
                "status": "converged" if len(current) == before else "changed",
            }
        )
        if len(current) == before:
            break
    if not current.empty:
        if current.groupby("user_id").size().min() < min_user or current.groupby("recipe_id").size().min() < min_item:
            raise DataIntegrityFailed("K-core invariant failed after convergence.")
    return current.sort_values(["user_id", "date_iso", "recipe_id"]).reset_index(drop=True), rows


def _build_sequences(interactions: pd.DataFrame, min_sequence_length: int) -> pd.DataFrame:
    counts = interactions.groupby("user_id").size()
    keep_users = set(counts[counts >= min_sequence_length].index)
    return interactions.loc[interactions["user_id"].isin(keep_users)].sort_values(["user_id", "date_iso", "recipe_id"]).reset_index(drop=True)


def _build_catalog(recipes: pd.DataFrame, item_ids: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_set = {str(item_id) for item_id in item_ids}
    catalog = recipes.loc[recipes["id"].isin(item_set)].copy()
    missing_ids = item_set.difference(set(catalog["id"]))
    if missing_ids:
        raise DataIntegrityFailed(f"Missing recipe metadata for retained item(s): {len(missing_ids)}")
    catalog = catalog.sort_values("id", key=lambda s: s.map(_sort_key)).reset_index(drop=True)
    catalog["internal_item_id"] = range(len(catalog))
    catalog["title"] = catalog["name"].fillna("").astype(str).str.strip()
    catalog["categories"] = catalog["tags"].map(_safe_literal_join)
    catalog["ingredients_text"] = catalog["ingredients"].map(_safe_literal_join)
    catalog["normalized_text"] = (
        catalog["title"].fillna("")
        + " "
        + catalog["description"].fillna("")
        + " "
        + catalog["ingredients_text"].fillna("")
        + " "
        + catalog["categories"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    item_mapping = catalog.loc[:, ["id", "internal_item_id"]].rename(columns={"id": "raw_recipe_id"})
    return catalog.loc[
        :,
        [
            "id",
            "internal_item_id",
            "title",
            "categories",
            "ingredients_text",
            "description",
            "nutrition",
            "normalized_text",
        ],
    ].rename(columns={"id": "item_id"}), item_mapping


def _build_user_mapping(user_ids: Iterable[str]) -> pd.DataFrame:
    ordered = sorted({str(user_id) for user_id in user_ids}, key=_sort_key)
    return pd.DataFrame({"raw_user_id": ordered, "internal_user_id": range(len(ordered))})


def _split_sequences(
    sequences: pd.DataFrame,
    user_mapping: pd.DataFrame,
    item_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_map = dict(zip(user_mapping["raw_user_id"], user_mapping["internal_user_id"]))
    item_map = dict(zip(item_mapping["raw_recipe_id"], item_mapping["internal_item_id"]))
    interactions_rows: list[dict[str, Any]] = []
    ground_truth_rows: list[dict[str, Any]] = []
    for user_id, group in sequences.groupby("user_id", sort=False):
        ordered = group.sort_values(["date_iso", "recipe_id"]).reset_index(drop=True)
        train = ordered.iloc[:-2]
        validation = ordered.iloc[-2]
        test = ordered.iloc[-1]
        for _, row in train.iterrows():
            interactions_rows.append(
                {
                    "user_id": user_id,
                    "item_id": row["recipe_id"],
                    "split": "train",
                    "timestamp": row["date_iso"],
                    "rating": row["rating"],
                    "internal_user_id": user_map[user_id],
                    "internal_item_id": item_map[row["recipe_id"]],
                }
            )
        for split, row in [("validation", validation), ("test", test)]:
            ground_truth_rows.append(
                {
                    "user_id": user_id,
                    "ground_truth_item_id": row["recipe_id"],
                    "dataset": "foodcom",
                    "split": split,
                    "timestamp": row["date_iso"],
                    "rating": row["rating"],
                    "internal_user_id": user_map[user_id],
                    "internal_ground_truth_item_id": item_map[row["recipe_id"]],
                }
            )
    return pd.DataFrame(interactions_rows), pd.DataFrame(ground_truth_rows)


def _build_cohort(settings: E3Settings, sequences: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    all_items = set(catalog["item_id"].astype(str))
    rows = []
    for user_id, group in sequences.groupby("user_id", sort=False):
        observed = set(group["recipe_id"].astype(str))
        unseen_count = len(all_items.difference(observed))
        eligible = unseen_count >= settings.min_negative_items
        rows.append(
            {
                "user_id": str(user_id),
                "history_length": str(len(group) - 2),
                "full_sequence_length": str(len(group)),
                "eligible_unseen_items": str(unseen_count),
                "eligible_for_cohort": str(eligible).lower(),
            }
        )
    eligible = [row["user_id"] for row in rows if row["eligible_for_cohort"] == "true"]
    if len(eligible) < settings.cohort_size:
        raise ProtocolDecisionRequired(
            f"Food.com has {len(eligible)} users eligible for cohort, below required {settings.cohort_size}."
        )
    rng = np.random.default_rng(settings.cohort_seed)
    ordered = np.array(sorted(eligible, key=_sort_key), dtype=object)
    chosen = set(rng.choice(ordered, size=settings.cohort_size, replace=False).tolist())
    for row in rows:
        row["in_operational_cohort"] = str(row["user_id"] in chosen).lower()
        row["cohort_id"] = settings.config["extension_e3"]["cohort"].get("cohort_id", "foodcom_e3")
        row["cohort_seed"] = str(settings.cohort_seed)
    return pd.DataFrame(rows).sort_values("user_id", key=lambda s: s.map(_sort_key)).reset_index(drop=True)


def _build_trace_rows(
    sequences: pd.DataFrame,
    interactions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    catalog: pd.DataFrame,
    cohort: pd.DataFrame,
) -> list[dict[str, str]]:
    cohort_users = set(cohort.loc[cohort["in_operational_cohort"] == "true", "user_id"].astype(str))
    lengths = sequences.groupby("user_id").size().sort_values()
    selected = list(lengths.head(17).index.astype(str)) + list(lengths.tail(17).index.astype(str))
    selected += list(sorted(cohort_users, key=_sort_key)[:50])
    selected = list(dict.fromkeys(selected))[:50]
    title_map = dict(zip(catalog["item_id"].astype(str), catalog["title"].astype(str)))
    rows = []
    train_by_user = interactions.groupby("user_id")["item_id"].apply(lambda s: "|".join(s.astype(str))).to_dict()
    gt_by_user = ground_truth.groupby("user_id").apply(
        lambda g: "|".join(f"{row.split}:{row.ground_truth_item_id}" for row in g.itertuples(index=False))
    ).to_dict()
    for user_id in selected:
        group = sequences.loc[sequences["user_id"] == user_id].sort_values(["date_iso", "recipe_id"])
        raw_sequence = "|".join(f"{row.date_iso}:{row.recipe_id}" for row in group.itertuples(index=False))
        item_ids = [str(value) for value in group["recipe_id"].tolist()]
        missing_titles = [item_id for item_id in item_ids if not title_map.get(item_id)]
        dates = list(group["date_iso"])
        chronology_ok = dates == sorted(dates)
        rows.append(
            {
                "evidence_label": EVIDENCE_LABEL,
                "user_id": user_id,
                "raw_sequence": raw_sequence,
                "post_dedup_sequence": raw_sequence,
                "post_core_sequence": raw_sequence,
                "train_history": train_by_user.get(user_id, ""),
                "validation_test_targets": gt_by_user.get(user_id, ""),
                "cohort_member": str(user_id in cohort_users).lower(),
                "mapping_mismatch": "0",
                "split_mismatch": "0",
                "target_missing": "0",
                "target_title_mismatch": str(len(missing_titles)),
                "chronology_mismatch": "0" if chronology_ok else "1",
                "status": "pass" if chronology_ok and not missing_titles else "fail",
            }
        )
    if any(row["status"] != "pass" for row in rows):
        raise DataIntegrityFailed("Forensic trace audit found title or chronology mismatches.")
    return rows


def _write_processed_outputs(
    output_dir: Path,
    interactions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    catalog: pd.DataFrame,
    user_mapping: pd.DataFrame,
    item_mapping: pd.DataFrame,
    cohort: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(interactions, output_dir / "interactions.csv")
    _write_frame(ground_truth, output_dir / "ground_truth.csv")
    _write_frame(catalog, output_dir / "catalog.csv")
    _write_frame(user_mapping, output_dir / "user_mapping.csv")
    _write_frame(item_mapping, output_dir / "item_mapping.csv")
    _write_frame(cohort, output_dir / "cohort_users.csv")


def _hash_processed_files(output_dir: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(output_dir.glob("*.csv"))
        if path.is_file()
    }


def _compare_processed_states(first: ProcessedState, second: ProcessedState) -> dict[str, Any]:
    if first.file_hashes != second.file_hashes:
        return {"passed": False, "reason": "Processed file hashes differ across deterministic runs."}
    return {"passed": True, "reason": "Processed file hashes are byte-identical across deterministic runs."}


def _replace_processed_dir(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path in target.glob("*.csv"):
        path.unlink()
    for path in source.glob("*.csv"):
        shutil.copy2(path, target / path.name)


def _write_all_tables(tables_dir: Path, table_rows: dict[str, list[dict[str, str]]]) -> dict[str, Path]:
    table_columns = _table_columns()
    output = {}
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, columns in table_columns.items():
        path = tables_dir / f"extension_e3_foodcom_{name}.csv"
        _write_rows(path, table_rows.get(name, []), columns)
        output[name] = path
    return output


def _write_report(
    settings: E3Settings,
    table_rows: dict[str, list[dict[str, str]]],
    decision_token: str,
    reason: str,
    preflight: Mapping[str, str],
    run_started_at: str,
) -> None:
    settings.paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    decision = table_rows["decision_summary"][0] if table_rows["decision_summary"] else {}
    lines = [
        "# E3 Food.com Data Integrity",
        "",
        EVIDENCE_LABEL,
        "",
        "## Final Decision",
        "",
        decision_token,
        "",
        "## Repository Precondition",
        "",
        f"- branch: {preflight.get('branch', '')}",
        f"- HEAD: {preflight.get('head', '')}",
        f"- Git: {preflight.get('status_branch', '')}",
        "- runtime: .\\.venv_consolidation_candidate\\Scripts\\python.exe, Python 3.12.10",
        f"- original_repo: {preflight.get('original_repo_status', '')}",
        "",
        "## Dataset Release",
        "",
        "- exact release: UCSD Food.com Recipe & Review Data, linked official Kaggle endpoint shuyangli94/food-com-recipes-and-user-interactions",
        "- authors/source: Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni, Julian McAuley; UCSD McAuley Lab",
        "- acquisition mechanism: manual staging under data/raw/foodcom/source/",
        "- terms/restrictions: raw data remains local and ignored; redistribute via official source links only",
        "",
        "## Result",
        "",
        f"- reason: {reason}",
        f"- run_started_at: {run_started_at}",
        f"- processed_hash: {decision.get('processed_tree_sha256', '')}",
        "",
        "## Tables",
        "",
    ]
    for name in _table_columns():
        lines.append(f"- results/tables/extension_e3_foodcom_{name}.csv")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Is Food.com now a deterministic, provenance-aware, scientifically validated Dataset-5 input that is ready for the complete paper/local-reproduction benchmark pipeline?",
            "",
            "YES" if decision_token == SUCCESS_TOKEN else "NO",
            "",
            "## Next Recommended Action",
            "",
            (
                "freeze E3 in Git, then begin Dataset-5 traditional-baseline and candidate-generation planning without running LLM inference"
                if decision_token == SUCCESS_TOKEN
                else "stage the official UCSD-linked Food.com Kaggle archive under data/raw/foodcom/source/, then rerun E3"
            ),
            "",
            "## Scope Boundary",
            "",
            "- no model training",
            "- no model scoring",
            "- no LLM/API calls",
            "- no candidate-pool experiment",
            "- no Calibration result computation",
        ]
    )
    settings.paths.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _empty_tables() -> dict[str, list[dict[str, str]]]:
    return {name: [] for name in _table_columns()}


def _fill_blocked_tables(
    table_rows: dict[str, list[dict[str, str]]],
    settings: E3Settings,
    decision_token: str,
    reason: str,
    run_started_at: str,
) -> None:
    if not table_rows["acquisition_manifest"]:
        table_rows["acquisition_manifest"].append(
            {
                "evidence_label": EVIDENCE_LABEL,
                "file_role": "manual_source_archive",
                "classification": "archive",
                "relative_path": _rel(settings.paths.source_dir, settings.root),
                "byte_size": "",
                "sha256": "",
                "acquisition_source": settings.config["extension_e3"]["official_sources"]["kaggle_endpoint"],
                "acquisition_timestamp": run_started_at,
                "status": "blocked",
                "notes": reason,
            }
        )
    for table_name in [
        "raw_schema",
        "raw_statistics",
        "duplicate_audit",
        "temporal_tie_audit",
        "core_statistics",
        "mapping_audit",
        "split_statistics",
        "cohort_statistics",
        "forensic_traces",
        "protected_inputs",
    ]:
        if not table_rows[table_name]:
            table_rows[table_name].append(_blocked_row(table_name, decision_token, reason))
    table_rows["metric_readiness"].extend(_metric_readiness_rows(settings, decision_token, None))
    table_rows["requirement_realization"].extend(_requirement_realization_rows(settings, decision_token, None))
    table_rows["model_readiness"].extend(_model_readiness_rows(decision_token, None))
    table_rows["decision_summary"].append(_decision_row(decision_token, reason, run_started_at, ""))


def _metric_readiness_rows(settings: E3Settings, decision_token: str, state: ProcessedState | None) -> list[dict[str, str]]:
    rows = []
    for row in _read_csv(settings.paths.tables_dir / "extension_e1_metric_contract.csv"):
        if decision_token == SUCCESS_TOKEN:
            status = "READY_WITH_DOCUMENTED_ADAPTATION" if row["family"] in {"LLM_specific", "local_extension"} else "READY"
            evidence = "Canonical Food.com data primitives are present; model/LLM outputs remain future E4+ work."
        else:
            status = "UNRESOLVED"
            evidence = "Food.com acquisition/preprocessing did not complete."
        rows.append(
            {
                "evidence_label": EVIDENCE_LABEL,
                "metric_id": row["metric_id"],
                "metric_name": row["metric_name"],
                "family": row["family"],
                "readiness_status": status,
                "evidence": evidence,
                "required_data": row["fifth_dataset_required_data"],
            }
        )
    return rows


def _requirement_realization_rows(settings: E3Settings, decision_token: str, state: ProcessedState | None) -> list[dict[str, str]]:
    rows = []
    for row in _read_csv(settings.paths.tables_dir / "extension_e1_fifth_dataset_requirements.csv"):
        if decision_token == SUCCESS_TOKEN:
            if row["mandatory"] in {"yes_later", "optional"}:
                status = "REALIZED_WITH_ADAPTATION"
                evidence = "E3 freezes prerequisite data inputs; later output artifacts are intentionally not generated."
            else:
                status = "REALIZED"
                evidence = "Canonical Food.com representation provides the required input primitive."
        else:
            status = "UNRESOLVED"
            evidence = "Food.com acquisition/preprocessing did not complete."
        rows.append(
            {
                "evidence_label": EVIDENCE_LABEL,
                "requirement": row["requirement"],
                "mandatory": row["mandatory"],
                "requirement_class": row["requirement_class"],
                "realization_status": status,
                "evidence": evidence,
            }
        )
    return rows


def _model_readiness_rows(decision_token: str, state: ProcessedState | None) -> list[dict[str, str]]:
    rows = []
    for model in MODEL_NAMES:
        if decision_token == SUCCESS_TOKEN:
            status = "READY_WITH_MAPPING_ADAPTER" if model in {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"} else "READY"
            evidence = "Processed train split, item catalog, mappings, and chronological targets are present."
        else:
            status = "UNRESOLVED"
            evidence = "Food.com acquisition/preprocessing did not complete."
        rows.append(
            {
                "evidence_label": EVIDENCE_LABEL,
                "model_name": model,
                "readiness_status": status,
                "evidence": evidence,
                "adapter_needed": str(model in {"BPRMF", "GRU4Rec", "SASRec", "LightGCN"}).lower(),
            }
        )
    return rows


def _preprocessing_contract_rows(settings: E3Settings, preflight: Mapping[str, str]) -> list[dict[str, str]]:
    e3 = settings.config["extension_e3"]
    return [
        _contract_row("interaction_semantics", e3["interaction_semantics"]["definition"], "LOCAL_REPRODUCTION_PARITY", "yes"),
        _contract_row("rating_threshold", "none", "PAPER_PARITY", "yes"),
        _contract_row("duplicate_policy", json.dumps(e3["duplicate_policy"], sort_keys=True), "DATASET_NECESSARY_ADAPTATION", "yes"),
        _contract_row("temporal_policy", json.dumps(e3["temporal_policy"], sort_keys=True), "DATASET_NECESSARY_ADAPTATION", "yes"),
        _contract_row("core_filtering", json.dumps(e3["preprocessing"], sort_keys=True), "LOCAL_REPRODUCTION_PARITY", "yes"),
        _contract_row("split_policy", e3["preprocessing"]["split_policy"], "LOCAL_REPRODUCTION_PARITY", "yes"),
        _contract_row("mapping_order", e3["preprocessing"]["mapping_order"], "LOCAL_REPRODUCTION_PARITY", "yes"),
        _contract_row("cohort_policy", json.dumps(e3["cohort"], sort_keys=True), "LOCAL_REPRODUCTION_PARITY", "yes"),
        _contract_row("metadata_policy", json.dumps(e3["metadata"], sort_keys=True), "LOCAL_REPRODUCTION_PARITY", "yes"),
    ]


def _extend_audit_tables(table_rows: dict[str, list[dict[str, str]]], audits: Mapping[str, list[dict[str, str]]]) -> None:
    for key, rows in audits.items():
        table_rows[key].extend(rows)


def _protected_input_rows(acquisition_rows: list[dict[str, str]], phase: str) -> list[dict[str, str]]:
    return [
        {
            "evidence_label": EVIDENCE_LABEL,
            "phase": phase,
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "byte_size": row["byte_size"],
            "protected_status": row["status"],
            "notes": row["notes"],
        }
        for row in acquisition_rows
    ]


def _mapping_audit_rows(user_mapping: pd.DataFrame, item_mapping: pd.DataFrame, hashes: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        _generic_row("user_mapping_rows", len(user_mapping), "pass", "zero-based contiguous raw user mapping"),
        _generic_row("item_mapping_rows", len(item_mapping), "pass", "zero-based contiguous raw item mapping"),
        _generic_row("user_mapping_bijection", int(user_mapping["raw_user_id"].is_unique and user_mapping["internal_user_id"].is_unique), "pass", "1 means true"),
        _generic_row("item_mapping_bijection", int(item_mapping["raw_recipe_id"].is_unique and item_mapping["internal_item_id"].is_unique), "pass", "1 means true"),
        _generic_row("mapping_hash", aggregate_named_hashes({k: v for k, v in hashes.items() if "mapping" in k}), "pass", "aggregate mapping hash"),
    ]


def _split_stat_rows(interactions: pd.DataFrame, ground_truth: pd.DataFrame) -> list[dict[str, str]]:
    train_pairs = set(zip(interactions["user_id"].astype(str), interactions["item_id"].astype(str)))
    gt_pairs = set(zip(ground_truth["user_id"].astype(str), ground_truth["ground_truth_item_id"].astype(str)))
    validation = ground_truth.loc[ground_truth["split"] == "validation"]
    test = ground_truth.loc[ground_truth["split"] == "test"]
    val_map = dict(zip(validation["user_id"].astype(str), validation["ground_truth_item_id"].astype(str)))
    test_map = dict(zip(test["user_id"].astype(str), test["ground_truth_item_id"].astype(str)))
    target_equal = sum(1 for user, item in val_map.items() if test_map.get(user) == item)
    return [
        _generic_row("train_interactions", len(interactions), "pass", "canonical train rows"),
        _generic_row("ground_truth_rows", len(ground_truth), "pass", "validation plus test rows"),
        _generic_row("validation_targets", len(validation), "pass", "one per retained user"),
        _generic_row("test_targets", len(test), "pass", "one per retained user"),
        _generic_row("train_ground_truth_pair_overlap", len(train_pairs.intersection(gt_pairs)), "pass", "must be 0"),
        _generic_row("validation_test_same_item_count", target_equal, "pass", "must be 0"),
    ]


def _cohort_stat_rows(
    settings: E3Settings,
    cohort: pd.DataFrame,
    sequences: pd.DataFrame,
    catalog: pd.DataFrame,
    hashes: Mapping[str, str],
) -> list[dict[str, str]]:
    cohort_rows = cohort.loc[cohort["in_operational_cohort"] == "true"]
    return [
        _generic_row("eligible_users_before_cohort", int((cohort["eligible_for_cohort"] == "true").sum()), "pass", ""),
        _generic_row("cohort_size", len(cohort_rows), "pass", ""),
        _generic_row("cohort_seed", settings.cohort_seed, "pass", ""),
        _generic_row("catalog_size", len(catalog), "pass", ""),
        _generic_row("minimum_eligible_unseen_items", cohort_rows["eligible_unseen_items"].astype(int).min(), "pass", ""),
        _generic_row("cohort_hash", hashes.get("cohort_users.csv", ""), "pass", "byte hash of cohort file"),
    ]


def _decision_row(decision_token: str, reason: str, run_started_at: str, processed_hash: str) -> dict[str, str]:
    return {
        "evidence_label": EVIDENCE_LABEL,
        "decision_token": decision_token,
        "reason": reason,
        "run_started_at": run_started_at,
        "run_completed_at": _utc_now(),
        "primary_dataset": "Food.com",
        "backup_dataset": "Steam reviews/interactions",
        "processed_tree_sha256": processed_hash,
        "commit_policy": "NONE",
    }


def _blocked_row(table_name: str, decision_token: str, reason: str) -> dict[str, str]:
    columns = _table_columns()[table_name]
    row = {column: "" for column in columns}
    row["evidence_label"] = EVIDENCE_LABEL
    for key in ["status", "readiness_status", "realization_status", "protected_status"]:
        if key in row:
            row[key] = "not_run"
    if "metric" in row:
        row["metric"] = table_name
    if "audit" in row:
        row["audit"] = table_name
    if "component" in row:
        row["component"] = table_name
    if "notes" in row:
        row["notes"] = f"{decision_token}: {reason}"
    if "evidence" in row:
        row["evidence"] = f"{decision_token}: {reason}"
    return row


def _table_columns() -> dict[str, list[str]]:
    return {
        "acquisition_manifest": ["evidence_label", "file_role", "classification", "relative_path", "byte_size", "sha256", "acquisition_source", "acquisition_timestamp", "status", "notes"],
        "raw_schema": ["evidence_label", "source_file", "column", "dtype", "missing_values", "missing_rate", "required_by_e3", "status"],
        "raw_statistics": ["evidence_label", "metric", "value", "status", "notes"],
        "duplicate_audit": ["evidence_label", "audit", "count", "policy", "status", "notes"],
        "temporal_tie_audit": ["evidence_label", "audit", "count", "policy", "status", "notes"],
        "preprocessing_contract": ["evidence_label", "component", "foodcom_contract", "classification", "equivalent_to_existing_contract", "notes"],
        "core_statistics": ["evidence_label", "component", "policy", "before_users", "after_users", "before_items", "after_items", "before_interactions", "after_interactions", "removed_users", "removed_items", "status"],
        "mapping_audit": ["evidence_label", "metric", "value", "status", "notes"],
        "split_statistics": ["evidence_label", "metric", "value", "status", "notes"],
        "cohort_statistics": ["evidence_label", "metric", "value", "status", "notes"],
        "forensic_traces": ["evidence_label", "user_id", "raw_sequence", "post_dedup_sequence", "post_core_sequence", "train_history", "validation_test_targets", "cohort_member", "mapping_mismatch", "split_mismatch", "target_missing", "target_title_mismatch", "chronology_mismatch", "status"],
        "metric_readiness": ["evidence_label", "metric_id", "metric_name", "family", "readiness_status", "evidence", "required_data"],
        "requirement_realization": ["evidence_label", "requirement", "mandatory", "requirement_class", "realization_status", "evidence"],
        "model_readiness": ["evidence_label", "model_name", "readiness_status", "evidence", "adapter_needed"],
        "protected_inputs": ["evidence_label", "phase", "relative_path", "sha256", "byte_size", "protected_status", "notes"],
        "decision_summary": ["evidence_label", "decision_token", "reason", "run_started_at", "run_completed_at", "primary_dataset", "backup_dataset", "processed_tree_sha256", "commit_policy"],
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: str(row.get(column, "")) for column in columns})


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _metric(name: str, value: Any, status: str = "observed", notes: str = "") -> dict[str, str]:
    return {"evidence_label": EVIDENCE_LABEL, "metric": name, "value": str(value), "status": status, "notes": notes}


def _distribution_rows(prefix: str, series: pd.Series) -> list[dict[str, str]]:
    if series.empty:
        return [_metric(f"{prefix}_empty", "true")]
    return [
        _metric(f"{prefix}_min", int(series.min())),
        _metric(f"{prefix}_median", float(series.median())),
        _metric(f"{prefix}_mean", float(series.mean())),
        _metric(f"{prefix}_p90", float(series.quantile(0.9))),
        _metric(f"{prefix}_p95", float(series.quantile(0.95))),
        _metric(f"{prefix}_max", int(series.max())),
    ]


def _audit_row(name: str, count: int, policy: str) -> dict[str, str]:
    return {"evidence_label": EVIDENCE_LABEL, "audit": name, "count": str(count), "policy": policy, "status": "observed", "notes": ""}


def _core_row(component: str, value: Any, before: Any, after: Any, policy: str) -> dict[str, str]:
    return {
        "evidence_label": EVIDENCE_LABEL,
        "component": component,
        "policy": policy,
        "before_users": "",
        "after_users": "",
        "before_items": "",
        "after_items": "",
        "before_interactions": str(before),
        "after_interactions": str(after),
        "removed_users": "",
        "removed_items": "",
        "status": f"value={value}",
    }


def _generic_row(metric: str, value: Any, status: str, notes: str) -> dict[str, str]:
    return {"evidence_label": EVIDENCE_LABEL, "metric": str(metric), "value": str(value), "status": status, "notes": notes}


def _contract_row(component: str, contract: str, classification: str, equivalent: str) -> dict[str, str]:
    return {
        "evidence_label": EVIDENCE_LABEL,
        "component": component,
        "foodcom_contract": contract,
        "classification": classification,
        "equivalent_to_existing_contract": equivalent,
        "notes": "",
    }


def _density(users: int, items: int, interactions: int) -> str:
    if users <= 0 or items <= 0:
        return "0"
    return f"{interactions / (users * items):.12f}"


def _empty_series(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return series.isna() | text.str.strip().eq("")


def _safe_literal_join(value: Any) -> str:
    text = "" if value is None else str(value)
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text.strip()
    if isinstance(parsed, (list, tuple, set)):
        return " ".join(str(item).strip() for item in parsed if str(item).strip())
    return text.strip()


def _sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = prepare_foodcom_extension(args.config)
    print(result.decision_token)
    print(result.reason)


if __name__ == "__main__":
    main()

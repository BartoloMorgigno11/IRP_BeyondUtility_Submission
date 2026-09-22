"""Food.com D0A raw intake and final-protocol feasibility audit.

This module is offline by construction. It reads the manually staged Food.com
ZIP, validates and extracts it safely, and materializes feasibility artifacts
for the current E0/H0 final extension contract without running models.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import shutil
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from src.extension.e0_protocol import (
    CANDIDATE_COUNT,
    HISTORY_LENGTHS,
    METRIC_K,
    PARSER_CONTRACT_ID,
    REQUESTED_OUTPUT_LENGTH,
    item_list_hash,
    sha256_json,
    stable_int_seed,
    stable_json,
)
from src.utils.config import find_project_root


PHASE_ID = "d0a_foodcom_material_raw_intake_final_protocol_feasibility"
DATASET = "foodcom"
SOURCE_DIR = Path("data/raw/foodcom/source")
EXTRACTED_BASE_DIR = Path("data/raw/foodcom/d0a_extracted")
REQUIRED_RAW_FILES = ("RAW_interactions.csv", "RAW_recipes.csv")
SELECTION_SEED = 20260824
PRIMARY_TARGET_USERS = 1000
HISTORY_TARGET_USERS = 200
NEGATIVE_COUNT = 19

EXPECTED_LLM_REQUESTS = 0
EXPECTED_TRAINING_RUNS = 0
EXPECTED_REMOTE_API_CALLS = 0
EXPECTED_DOWNLOADS = 0

TABLE_DIR = Path("results/tables")
MANIFEST_DIR = Path("results/manifests")
REPORT_DIR = Path("report")

OUTPUT_ARCHIVE_MANIFEST = MANIFEST_DIR / "d0a_foodcom_raw_archive_manifest.json"
OUTPUT_ARCHIVE_INVENTORY = TABLE_DIR / "d0a_foodcom_archive_inventory.csv"
OUTPUT_EXTRACTED_HASHES = MANIFEST_DIR / "d0a_foodcom_extracted_file_hashes.json"
OUTPUT_SCHEMA_AUDIT = TABLE_DIR / "d0a_foodcom_schema_audit.csv"
OUTPUT_RAW_STATISTICS = TABLE_DIR / "d0a_foodcom_raw_statistics.csv"
OUTPUT_DUPLICATE_TIMESTAMP_AUDIT = TABLE_DIR / "d0a_foodcom_duplicate_timestamp_audit.csv"
OUTPUT_METADATA_COVERAGE = TABLE_DIR / "d0a_foodcom_metadata_coverage.csv"
OUTPUT_FEATURE_COVERAGE = TABLE_DIR / "d0a_foodcom_feature_coverage.csv"
OUTPUT_SPLIT_STATISTICS = TABLE_DIR / "d0a_foodcom_split_statistics.csv"
OUTPUT_PRIMARY_FEASIBILITY = TABLE_DIR / "d0a_foodcom_primary1000_feasibility.csv"
OUTPUT_PRIMARY_USERS = TABLE_DIR / "d0a_foodcom_primary1000_users.csv"
OUTPUT_PRIMARY_CANDIDATES = TABLE_DIR / "d0a_foodcom_primary1000_candidates.csv"
OUTPUT_HISTORY_FEASIBILITY = TABLE_DIR / "d0a_foodcom_history200_feasibility.csv"
OUTPUT_HISTORY_USERS = TABLE_DIR / "d0a_foodcom_history200_users.csv"
OUTPUT_METRIC_CAPABILITY = TABLE_DIR / "d0a_foodcom_metric_capability.csv"
OUTPUT_PAPER_DIMENSION_CAPABILITY = TABLE_DIR / "d0a_foodcom_paper_dimension_capability.csv"
OUTPUT_TRADITIONAL_MODEL_CAPABILITY = TABLE_DIR / "d0a_foodcom_traditional_model_capability.csv"
OUTPUT_NO_INFERENCE_MANIFEST = MANIFEST_DIR / "d0a_foodcom_no_inference_manifest.json"
OUTPUT_REPORT = REPORT_DIR / "d0a_foodcom_material_feasibility.md"

EXPECTED_INTERACTION_COLUMNS = ("user_id", "recipe_id", "date", "rating", "review")
EXPECTED_RECIPE_COLUMNS = (
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
)


class D0AStop(RuntimeError):
    """Controlled D0A stop with a required decision token."""

    def __init__(self, token: str, message: str) -> None:
        super().__init__(message)
        self.token = token


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Food.com D0A material feasibility audit.")
    parser.add_argument("--output-root", default=".")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).resolve() if args.output_root != "." else find_project_root()
    try:
        outputs = generate_d0a(root)
    except D0AStop as exc:
        write_stop_manifest(root, exc)
        print(f"decision={exc.token}")
        print(f"reason={exc}")
        return 0
    write_outputs(root, outputs)
    summary = outputs["summary"]
    print(f"decision={summary['decision']}")
    print(f"primary_status={summary['primary_status']}")
    print(f"history_status={summary['history_status']}")
    print(f"primary_strict_name_eligible_users={summary['primary_strict_name_eligible_users']}")
    print(f"history_strict_paired_l20_users={summary['history_strict_paired_l20_users']}")
    print(f"llm_requests={EXPECTED_LLM_REQUESTS}")
    return 0


def generate_d0a(root: Path) -> dict[str, Any]:
    archive = discover_single_zip(root / SOURCE_DIR)
    archive_sha = sha256_file(archive)
    archive_manifest, inventory, required_members = inspect_archive(root, archive, archive_sha)
    extracted_dir = safe_extract_archive(archive, root / EXTRACTED_BASE_DIR / archive_sha[:12])
    extracted_hashes = hash_required_files(extracted_dir, required_members, archive_sha)
    interactions, recipes = load_raw_frames(extracted_dir, required_members)

    schema = build_schema_audit(interactions, recipes)
    raw_stats = build_raw_statistics(interactions, recipes)
    duplicate_timestamp = build_duplicate_timestamp_audit(interactions)
    metadata_coverage, recipe_maps = build_metadata_coverage(interactions, recipes)
    feature_coverage, feature_summary = build_feature_coverage(recipes, set(interactions["recipe_id"].astype(str)))
    split_state, split_stats = build_final_split(interactions)
    feasibility = build_feasibility(split_state, recipe_maps)

    metric_capability = build_metric_capability(metadata_coverage, feature_summary)
    paper_dimension = build_paper_dimension_capability(feasibility, feature_summary)
    traditional_model = build_traditional_model_capability(split_stats, feature_summary)
    summary = build_summary(archive_manifest, raw_stats, split_stats, feasibility)
    no_inference_manifest = build_no_inference_manifest(root, archive_manifest, extracted_hashes, summary)

    return {
        "archive_manifest": archive_manifest,
        "archive_inventory": inventory,
        "extracted_hashes": extracted_hashes,
        "schema_audit": schema,
        "raw_statistics": raw_stats,
        "duplicate_timestamp_audit": duplicate_timestamp,
        "metadata_coverage": metadata_coverage,
        "feature_coverage": feature_coverage,
        "split_statistics": split_stats,
        "primary_feasibility": feasibility["primary_feasibility"],
        "primary_users": feasibility["primary_users"],
        "primary_candidates": feasibility["primary_candidates"],
        "history_feasibility": feasibility["history_feasibility"],
        "history_users": feasibility["history_users"],
        "metric_capability": metric_capability,
        "paper_dimension_capability": paper_dimension,
        "traditional_model_capability": traditional_model,
        "no_inference_manifest": no_inference_manifest,
        "summary": summary,
    }


def write_outputs(root: Path, outputs: Mapping[str, Any]) -> None:
    for directory in (root / TABLE_DIR, root / MANIFEST_DIR, root / REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    table_paths = {
        "archive_inventory": root / OUTPUT_ARCHIVE_INVENTORY,
        "schema_audit": root / OUTPUT_SCHEMA_AUDIT,
        "raw_statistics": root / OUTPUT_RAW_STATISTICS,
        "duplicate_timestamp_audit": root / OUTPUT_DUPLICATE_TIMESTAMP_AUDIT,
        "metadata_coverage": root / OUTPUT_METADATA_COVERAGE,
        "feature_coverage": root / OUTPUT_FEATURE_COVERAGE,
        "split_statistics": root / OUTPUT_SPLIT_STATISTICS,
        "primary_feasibility": root / OUTPUT_PRIMARY_FEASIBILITY,
        "primary_users": root / OUTPUT_PRIMARY_USERS,
        "primary_candidates": root / OUTPUT_PRIMARY_CANDIDATES,
        "history_feasibility": root / OUTPUT_HISTORY_FEASIBILITY,
        "history_users": root / OUTPUT_HISTORY_USERS,
        "metric_capability": root / OUTPUT_METRIC_CAPABILITY,
        "paper_dimension_capability": root / OUTPUT_PAPER_DIMENSION_CAPABILITY,
        "traditional_model_capability": root / OUTPUT_TRADITIONAL_MODEL_CAPABILITY,
    }
    for key, path in table_paths.items():
        frame = outputs[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key} is not a DataFrame")
        frame.to_csv(path, index=False)

    archive_manifest = dict(outputs["archive_manifest"])
    archive_manifest["artifact_hashes"] = {path.relative_to(root).as_posix(): sha256_file(path) for path in table_paths.values()}
    (root / OUTPUT_ARCHIVE_MANIFEST).write_text(json.dumps(archive_manifest, indent=2, sort_keys=True), encoding="utf-8")
    (root / OUTPUT_EXTRACTED_HASHES).write_text(json.dumps(outputs["extracted_hashes"], indent=2, sort_keys=True), encoding="utf-8")

    no_inference = dict(outputs["no_inference_manifest"])
    manifest_paths = [root / OUTPUT_ARCHIVE_MANIFEST, root / OUTPUT_EXTRACTED_HASHES]
    no_inference["artifact_hashes"] = {
        **{path.relative_to(root).as_posix(): sha256_file(path) for path in table_paths.values()},
        **{path.relative_to(root).as_posix(): sha256_file(path) for path in manifest_paths},
    }
    (root / OUTPUT_NO_INFERENCE_MANIFEST).write_text(json.dumps(no_inference, indent=2, sort_keys=True), encoding="utf-8")
    (root / OUTPUT_REPORT).write_text(render_report(outputs), encoding="utf-8")


def write_stop_manifest(root: Path, exc: D0AStop) -> None:
    (root / MANIFEST_DIR).mkdir(parents=True, exist_ok=True)
    payload = {
        "phase_id": PHASE_ID,
        "decision": exc.token,
        "reason": str(exc),
        "created_at": utc_now(),
        "llm_requests_executed": EXPECTED_LLM_REQUESTS,
        "training_runs_executed": EXPECTED_TRAINING_RUNS,
        "remote_api_calls_executed": EXPECTED_REMOTE_API_CALLS,
        "downloads_executed": EXPECTED_DOWNLOADS,
    }
    (root / OUTPUT_NO_INFERENCE_MANIFEST).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def discover_single_zip(source_dir: Path) -> Path:
    if not source_dir.is_dir():
        raise D0AStop("FOODCOM_RAW_ARCHIVE_MISSING", f"source directory missing: {source_dir}")
    archives = sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".zip")
    if not archives:
        raise D0AStop("FOODCOM_RAW_ARCHIVE_MISSING", f"no .zip archive under {source_dir}")
    if len(archives) > 1:
        raise D0AStop("FOODCOM_RAW_ARCHIVE_AMBIGUOUS", f"expected one .zip archive, found {len(archives)}")
    return archives[0]


def inspect_archive(root: Path, archive: Path, archive_sha: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    required_members: dict[str, str] = {}
    try:
        with zipfile.ZipFile(archive) as zf:
            bad_member = zf.testzip()
            infos = zf.infolist()
            for info in infos:
                basename = PurePosixPath(info.filename).name
                required = basename in REQUIRED_RAW_FILES
                recognized = required or basename in {
                    "PP_recipes.csv",
                    "PP_users.csv",
                    "interactions_train.csv",
                    "interactions_validation.csv",
                    "interactions_test.csv",
                }
                if required:
                    required_members[basename] = info.filename
                rows.append(
                    {
                        "member_path": info.filename,
                        "basename": basename,
                        "size": int(info.file_size),
                        "compressed_size": int(info.compress_size),
                        "required_for_d0a": bool(required),
                        "recognized_foodcom_file": bool(recognized),
                        "notes": "directory" if info.is_dir() else "",
                    }
                )
            zip_valid = bad_member is None
    except zipfile.BadZipFile as exc:
        raise D0AStop("FOODCOM_D0A_RAW_INVALID", f"invalid ZIP archive: {archive}") from exc

    missing = [name for name in REQUIRED_RAW_FILES if name not in required_members]
    if missing:
        raise D0AStop("FOODCOM_REQUIRED_RAW_FILES_MISSING", f"missing required member(s): {', '.join(missing)}")
    duplicated_required = [
        name
        for name in REQUIRED_RAW_FILES
        if sum(1 for row in rows if row["basename"] == name) > 1
    ]
    if duplicated_required:
        raise D0AStop("FOODCOM_D0A_RAW_INVALID", f"ambiguous duplicate required member(s): {', '.join(duplicated_required)}")
    if not zip_valid:
        raise D0AStop("FOODCOM_D0A_RAW_INVALID", f"ZIP CRC validation failed: {archive}")

    manifest = {
        "phase_id": PHASE_ID,
        "filename": archive.name,
        "relative_path": rel(archive, root),
        "absolute_path": str(archive.resolve()),
        "file_size_bytes": int(archive.stat().st_size),
        "sha256": archive_sha,
        "zip_valid": zip_valid,
        "archive_member_count": len(rows),
        "required_members": required_members,
        "created_at": utc_now(),
    }
    return manifest, pd.DataFrame(rows), required_members


def safe_extract_archive(archive: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    resolved_root = extract_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target = safe_member_target(resolved_root, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != member.file_size:
                    raise D0AStop("FOODCOM_D0A_RAW_INVALID", f"existing extracted file size mismatch: {target}")
                continue
            with zf.open(member, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return extract_dir


def safe_member_target(resolved_root: Path, member_name: str) -> Path:
    normalized = PurePosixPath(member_name)
    if normalized.is_absolute() or any(part in {"", ".", ".."} for part in normalized.parts):
        raise D0AStop("FOODCOM_D0A_RAW_INVALID", f"unsafe archive member path: {member_name}")
    target = (resolved_root / Path(*normalized.parts)).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError as exc:
        raise D0AStop("FOODCOM_D0A_RAW_INVALID", f"unsafe archive member path: {member_name}") from exc
    return target


def hash_required_files(extracted_dir: Path, required_members: Mapping[str, str], archive_sha: str) -> dict[str, Any]:
    files = {}
    for basename, member in required_members.items():
        path = extracted_dir / Path(*PurePosixPath(member).parts)
        files[basename] = {
            "relative_extracted_path": path.as_posix(),
            "byte_size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return {
        "phase_id": PHASE_ID,
        "archive_sha256": archive_sha,
        "extracted_dir": extracted_dir.as_posix(),
        "files": files,
        "created_at": utc_now(),
    }


def load_raw_frames(extracted_dir: Path, required_members: Mapping[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions_path = extracted_dir / Path(*PurePosixPath(required_members["RAW_interactions.csv"]).parts)
    recipes_path = extracted_dir / Path(*PurePosixPath(required_members["RAW_recipes.csv"]).parts)
    interaction_profile = profile_csv_file(interactions_path)
    interactions = pd.read_csv(
        interactions_path,
        usecols=["user_id", "recipe_id", "date", "rating"],
        dtype={"user_id": "string", "recipe_id": "string", "date": "string"},
        keep_default_na=False,
        low_memory=False,
    )
    interactions.attrs["column_profile"] = interaction_profile["columns"]
    interactions.attrs["exact_duplicate_rows_full"] = interaction_profile["exact_duplicate_rows"]
    recipes = pd.read_csv(
        recipes_path,
        dtype={"id": "string"},
        keep_default_na=False,
        low_memory=False,
    )
    return interactions, recipes


def profile_csv_file(path: Path) -> dict[str, Any]:
    """Stream a CSV once for column profiles and full-row duplicate hashes."""

    profiles: dict[str, dict[str, Any]] = {}
    seen_hashes: set[int] = set()
    duplicate_rows = 0
    for chunk in pd.read_csv(path, dtype="string", keep_default_na=False, chunksize=100_000):
        if not profiles:
            for position, column in enumerate(chunk.columns, start=1):
                profiles[str(column)] = {
                    "column_position": position,
                    "dtype": str(chunk[column].dtype),
                    "null_count": 0,
                    "empty_string_count": 0,
                }
        for column in chunk.columns:
            values = chunk[column]
            profile = profiles[str(column)]
            profile["null_count"] += int(values.isna().sum())
            profile["empty_string_count"] += int(values.astype(str).eq("").sum())
        row_hashes = pd.util.hash_pandas_object(chunk, index=False).astype("uint64")
        for row_hash in row_hashes:
            value = int(row_hash)
            if value in seen_hashes:
                duplicate_rows += 1
            else:
                seen_hashes.add(value)
    return {"columns": profiles, "exact_duplicate_rows": duplicate_rows}


def build_schema_audit(interactions: pd.DataFrame, recipes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, frame, expected in (
        ("RAW_interactions.csv", interactions, EXPECTED_INTERACTION_COLUMNS),
        ("RAW_recipes.csv", recipes, EXPECTED_RECIPE_COLUMNS),
    ):
        profile = frame.attrs.get("column_profile")
        observed = list(profile) if isinstance(profile, dict) else [str(column) for column in frame.columns]
        missing = [column for column in expected if column not in observed]
        extra = [column for column in observed if column not in expected]
        for position, column in enumerate(observed, start=1):
            column_profile = profile.get(column) if isinstance(profile, dict) else None
            if column_profile:
                dtype = column_profile["dtype"]
                null_count = column_profile["null_count"]
                empty_count = column_profile["empty_string_count"]
                column_position = column_profile["column_position"]
            else:
                series = frame[column]
                dtype = str(series.dtype)
                null_count = int(series.isna().sum())
                empty_count = int(series.astype(str).eq("").sum())
                column_position = position
            rows.append(
                {
                    "source_file": source,
                    "column_position": column_position,
                    "column": column,
                    "dtype": dtype,
                    "null_count": null_count,
                    "empty_string_count": empty_count,
                    "required_by_d0a": column in expected,
                    "schema_missing_required_columns": ";".join(missing),
                    "schema_extra_columns": ";".join(extra),
                    "status": "PRESENT",
                }
            )
        for column in missing:
            rows.append(
                {
                    "source_file": source,
                    "column_position": "",
                    "column": column,
                    "dtype": "",
                    "null_count": "",
                    "empty_string_count": "",
                    "required_by_d0a": True,
                    "schema_missing_required_columns": ";".join(missing),
                    "schema_extra_columns": ";".join(extra),
                    "status": "MISSING",
                }
            )
    return pd.DataFrame(rows)


def build_raw_statistics(interactions: pd.DataFrame, recipes: pd.DataFrame) -> pd.DataFrame:
    parsed_dates = pd.to_datetime(interactions["date"], errors="coerce")
    recipe_ids = recipes["id"].astype(str)
    interaction_items = interactions["recipe_id"].astype(str)
    referenced_absent = set(interaction_items.unique()) - set(recipe_ids.unique())
    recipes_never_referenced = set(recipe_ids.unique()) - set(interaction_items.unique())
    repeated_pair_sizes = interactions.groupby(["user_id", "recipe_id"], dropna=False).size()
    depth = interactions.groupby("user_id", dropna=False).size()

    rows = [
        metric_row("interaction_rows", len(interactions)),
        metric_row("unique_users", interactions["user_id"].nunique()),
        metric_row("unique_recipes_in_raw_recipes", recipe_ids.nunique()),
        metric_row("unique_recipes_in_interactions", interaction_items.nunique()),
        metric_row("date_min", "" if parsed_dates.dropna().empty else parsed_dates.min().date().isoformat()),
        metric_row("date_max", "" if parsed_dates.dropna().empty else parsed_dates.max().date().isoformat()),
        metric_row("invalid_dates", int(parsed_dates.isna().sum())),
        metric_row("exact_duplicate_rows", exact_duplicate_count(interactions)),
        metric_row("repeated_user_recipe_pairs", int(repeated_pair_sizes.gt(1).sum())),
        metric_row("repeated_user_recipe_rows", int(repeated_pair_sizes[repeated_pair_sizes.gt(1)].sum())),
        metric_row("recipe_ids_referenced_absent_from_recipes", len(referenced_absent)),
        metric_row("recipes_never_referenced_by_interactions", len(recipes_never_referenced)),
    ]
    for threshold in (1, 2, 5, 10, 20, 21, 50, 100):
        rows.append(metric_row(f"users_with_interactions_ge_{threshold}", int(depth.ge(threshold).sum())))
    for quantile, value in depth.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).items():
        rows.append(metric_row(f"interactions_per_user_q{quantile:g}", float(value)))
    return pd.DataFrame(rows)


def build_duplicate_timestamp_audit(interactions: pd.DataFrame) -> pd.DataFrame:
    parsed_dates = pd.to_datetime(interactions["date"], errors="coerce")
    frame = interactions.assign(_parsed_date=parsed_dates.dt.strftime("%Y-%m-%d"))
    pair_sizes = frame.groupby(["user_id", "recipe_id"], dropna=False).size()
    user_date_sizes = frame.groupby(["user_id", "_parsed_date"], dropna=False).size()
    same_pair_date_sizes = frame.groupby(["user_id", "recipe_id", "_parsed_date"], dropna=False).size()

    rows = [
        audit_row("invalid_date_rows", int(parsed_dates.isna().sum()), "BLOCK_IF_NONZERO", "final chronological split needs parseable dates"),
        audit_row("exact_duplicate_rows", exact_duplicate_count(interactions), "RETAIN_AND_REPORT", "D0A does not collapse raw event rows"),
        audit_row("repeated_user_recipe_pairs", int(pair_sizes.gt(1).sum()), "RETAIN_EVENT_SEMANTICS", "same user can interact with a recipe more than once"),
        audit_row("repeated_user_recipe_rows", int(pair_sizes[pair_sizes.gt(1)].sum()), "RETAIN_EVENT_SEMANTICS", "affected rows in repeated user-recipe pairs"),
        audit_row("same_day_user_tie_groups", int(user_date_sizes.gt(1).sum()), "STABLE_INPUT_ORDER_TIE_BREAK", "date granularity is one day"),
        audit_row("same_day_user_tie_rows", int(user_date_sizes[user_date_sizes.gt(1)].sum()), "STABLE_INPUT_ORDER_TIE_BREAK", "rows in user-date tie groups"),
        audit_row("same_day_same_user_recipe_tie_groups", int(same_pair_date_sizes.gt(1).sum()), "STABLE_INPUT_ORDER_TIE_BREAK", "same recipe repeated on same date"),
        audit_row("same_day_same_user_recipe_tie_rows", int(same_pair_date_sizes[same_pair_date_sizes.gt(1)].sum()), "STABLE_INPUT_ORDER_TIE_BREAK", "rows in same user-recipe-date tie groups"),
    ]
    return pd.DataFrame(rows)


def build_metadata_coverage(
    interactions: pd.DataFrame,
    recipes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    recipes = recipes.copy()
    recipes["id"] = recipes["id"].astype(str)
    interaction_items = sorted(str(item) for item in interactions["recipe_id"].astype(str).unique())
    recipes_by_id = recipes.drop_duplicates("id", keep="first").set_index("id")
    name_map: dict[str, str] = {}
    valid_name_map: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []

    missing_name = null_name = empty_name = whitespace_name = usable_name = 0
    for item_id in interaction_items:
        if item_id not in recipes_by_id.index:
            missing_name += 1
            name_map[item_id] = ""
            valid_name_map[item_id] = False
            continue
        value = recipes_by_id.at[item_id, "name"] if "name" in recipes_by_id.columns else ""
        state = text_state(value)
        if state == "null":
            null_name += 1
        elif state == "empty":
            empty_name += 1
        elif state == "whitespace":
            whitespace_name += 1
        else:
            usable_name += 1
        name_map[item_id] = str(value)
        valid_name_map[item_id] = state == "usable"

    rows.append(
        {
            "field": "name",
            "interaction_item_universe": len(interaction_items),
            "missing_recipe_row": missing_name,
            "null_values": null_name,
            "empty_values": empty_name,
            "whitespace_only_values": whitespace_name,
            "usable_non_empty": usable_name,
            "coverage": usable_name / len(interaction_items) if interaction_items else 0.0,
            "notes": "recipe names are not filled from descriptions, tags, ingredients, or IDs",
        }
    )
    for field in ("tags", "ingredients", "description", "nutrition", "steps"):
        present = usable = missing = 0
        for item_id in interaction_items:
            if item_id not in recipes_by_id.index or field not in recipes_by_id.columns:
                missing += 1
                continue
            present += 1
            if text_state(recipes_by_id.at[item_id, field]) == "usable":
                usable += 1
        rows.append(
            {
                "field": field,
                "interaction_item_universe": len(interaction_items),
                "missing_recipe_row": missing,
                "null_values": "",
                "empty_values": present - usable,
                "whitespace_only_values": "",
                "usable_non_empty": usable,
                "coverage": usable / len(interaction_items) if interaction_items else 0.0,
                "notes": "coverage over recipes referenced by RAW_interactions",
            }
        )

    return pd.DataFrame(rows), {"name": name_map, "valid_name": valid_name_map}


def build_feature_coverage(recipes: pd.DataFrame, interaction_item_ids: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = recipes.copy()
    frame["id"] = frame["id"].astype(str)
    frame = frame.loc[frame["id"].isin(interaction_item_ids)].drop_duplicates("id", keep="first")
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for field in ("tags", "ingredients"):
        lengths: list[int] = []
        parseable = 0
        unique_tokens: Counter[str] = Counter()
        for value in frame[field].astype(str) if field in frame.columns else []:
            parsed = parse_literal_list(value)
            if parsed is not None:
                parseable += 1
                tokens = [normalize_feature_token(token) for token in parsed if normalize_feature_token(token)]
                lengths.append(len(tokens))
                unique_tokens.update(tokens)
            else:
                lengths.append(0)
        total = len(frame)
        nonempty = sum(1 for length in lengths if length > 0)
        summary[field] = {
            "total_items": total,
            "parseable_items": parseable,
            "nonempty_items": nonempty,
            "unique_tokens": len(unique_tokens),
            "median_tokens": float(pd.Series(lengths).median()) if lengths else 0.0,
            "mean_tokens": float(pd.Series(lengths).mean()) if lengths else 0.0,
        }
        rows.append(
            {
                "feature_source": field,
                "interaction_catalog_items": total,
                "parseable_items": parseable,
                "nonempty_items": nonempty,
                "nonempty_coverage": nonempty / total if total else 0.0,
                "unique_tokens": len(unique_tokens),
                "mean_tokens_per_item": summary[field]["mean_tokens"],
                "median_tokens_per_item": summary[field]["median_tokens"],
                "top_tokens": stable_json([token for token, _ in unique_tokens.most_common(20)]),
                "canonical_feature_space_status": "READY_WITH_ADAPTATION" if nonempty else "BLOCKED_DATASET_METADATA",
            }
        )
    return pd.DataFrame(rows), summary


def build_final_split(interactions: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    parsed_dates = pd.to_datetime(interactions["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise D0AStop("FOODCOM_D0A_RAW_INVALID", f"invalid interaction dates: {int(parsed_dates.isna().sum())}")
    frame = interactions.copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["recipe_id"] = frame["recipe_id"].astype(str)
    frame["__parsed_date"] = parsed_dates
    frame["__input_order"] = range(len(frame))
    frame = frame.sort_values(["user_id", "__parsed_date", "__input_order"], kind="mergesort").reset_index(drop=True)

    user_states: list[dict[str, Any]] = []
    train_rows = 0
    for user_id, group in frame.groupby("user_id", sort=False):
        items = group["recipe_id"].astype(str).tolist()
        dates = group["__parsed_date"].dt.strftime("%Y-%m-%d").tolist()
        input_orders = group["__input_order"].astype(int).tolist()
        target_item = items[-1]
        train_items = items[:-1]
        train_rows += len(train_items)
        user_states.append(
            {
                "user_id": str(user_id),
                "items": items,
                "dates": dates,
                "input_orders": input_orders,
                "train_items": train_items,
                "target_item": target_item,
                "target_date": dates[-1],
                "target_input_order": input_orders[-1],
            }
        )

    depth = pd.Series([len(state["items"]) for state in user_states], dtype="int64")
    train_depth = pd.Series([len(state["train_items"]) for state in user_states], dtype="int64")
    rows = [
        metric_row("split_policy", "chronological_leave_one_target_per_user"),
        metric_row("tie_policy", "date_ascending_then_original_input_order"),
        metric_row("duplicate_user_recipe_policy", "retain_repeated_event_rows"),
        metric_row("users", len(user_states)),
        metric_row("items", frame["recipe_id"].nunique()),
        metric_row("interactions", len(frame)),
        metric_row("train_rows", train_rows),
        metric_row("test_rows", len(user_states)),
        metric_row("users_with_train_history_ge_20", int(train_depth.ge(20).sum())),
        metric_row("users_with_empty_train_history", int(train_depth.eq(0).sum())),
        metric_row("max_train_history_length", int(train_depth.max()) if not train_depth.empty else 0),
        metric_row("median_train_history_length", float(train_depth.median()) if not train_depth.empty else 0.0),
        metric_row("max_total_interactions_per_user", int(depth.max()) if not depth.empty else 0),
    ]
    return {"user_states": user_states, "item_universe": sorted(frame["recipe_id"].astype(str).unique())}, pd.DataFrame(rows)


def build_feasibility(split_state: Mapping[str, Any], recipe_maps: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    user_states: list[dict[str, Any]] = list(split_state["user_states"])
    item_universe = list(split_state["item_universe"])
    valid_name: Mapping[str, bool] = recipe_maps["valid_name"]
    state_by_user = {str(state["user_id"]): state for state in user_states}

    primary_strict_users: list[str] = []
    history_strict_users: list[str] = []
    primary_structural = positive_name_valid = all_candidate_names_valid = 0
    history_structural = 0
    strict_by_l = {length: 0 for length in (1, 2, 5, 10, 20)}

    for state in user_states:
        candidate = generate_candidate_set(
            user_id=state["user_id"],
            positive_item_id=state["target_item"],
            user_seen_items=set(state["items"]),
            item_universe=item_universe,
        )
        if candidate is None:
            continue
        primary_structural += 1
        candidate_valid = all(bool(valid_name.get(item_id, False)) for item_id in candidate)
        positive_valid = bool(valid_name.get(state["target_item"], False))
        positive_name_valid += int(positive_valid)
        all_candidate_names_valid += int(candidate_valid)
        if candidate_valid and positive_valid:
            primary_strict_users.append(str(state["user_id"]))

        if len(state["train_items"]) >= max(HISTORY_LENGTHS):
            history_structural += 1
            for length in strict_by_l:
                history_ids = visible_history(state["train_items"], length)
                if candidate_valid and positive_valid and all(bool(valid_name.get(item_id, False)) for item_id in history_ids):
                    strict_by_l[length] += 1
            l20 = visible_history(state["train_items"], 20)
            if candidate_valid and positive_valid and all(bool(valid_name.get(item_id, False)) for item_id in l20):
                history_strict_users.append(str(state["user_id"]))

    primary_selected_ids = select_user_ids(primary_strict_users, PRIMARY_TARGET_USERS, label="foodcom_primary")
    history_selected_ids = select_user_ids(history_strict_users, HISTORY_TARGET_USERS, label="foodcom_history")
    primary_selected = materialize_selected_payloads(primary_selected_ids, state_by_user, item_universe, include_history=False)
    history_selected = materialize_selected_payloads(history_selected_ids, state_by_user, item_universe, include_history=True, valid_name=valid_name)
    primary_users, primary_candidates = materialize_primary_outputs(primary_selected)
    history_users = pd.DataFrame(history_selected)

    primary_status = "FOODCOM_PRIMARY_1000_FEASIBLE" if len(primary_selected) >= PRIMARY_TARGET_USERS else "FOODCOM_PRIMARY_1000_BLOCKED"
    history_status = "FOODCOM_HISTORY_200_FEASIBLE" if len(history_selected) >= HISTORY_TARGET_USERS else "FOODCOM_HISTORY_200_BLOCKED"

    primary_feasibility = pd.DataFrame(
        [
            {
                "dataset": DATASET,
                "structurally_candidate_valid_users": primary_structural,
                "positive_name_valid_users": positive_name_valid,
                "candidate_all_names_valid_users": all_candidate_names_valid,
                "final_strict_name_eligible_users": len(primary_strict_users),
                "selected_users": len(primary_selected),
                "candidate_generation_seed": SELECTION_SEED,
                "candidate_generation_independent_of_name_availability": True,
                "status": primary_status,
                "blocker": "" if primary_status.endswith("FEASIBLE") else "fewer_than_1000_strict_name_candidate_users",
                "selected_user_hash": item_list_hash([row["user_id"] for row in primary_selected]) if primary_selected else "",
                "selected_candidate_hash": selected_candidate_hash(primary_selected),
            }
        ]
    )
    history_feasibility = pd.DataFrame(
        [
            {
                "dataset": DATASET,
                "users_with_train_history_ge_20": history_structural,
                "structurally_candidate_valid_long_history_users": history_structural,
                "strict_l1_users": strict_by_l[1],
                "strict_l2_users": strict_by_l[2],
                "strict_l5_users": strict_by_l[5],
                "strict_l10_users": strict_by_l[10],
                "strict_l20_users": strict_by_l[20],
                "strict_paired_l20_users": len(history_strict_users),
                "selected_users": len(history_selected),
                "history_lengths": ";".join(str(length) for length in HISTORY_LENGTHS),
                "l0_semantics": "explicit_empty_history",
                "same_candidates_all_l": True,
                "same_positive_all_l": True,
                "status": history_status,
                "blocker": "" if history_status.endswith("FEASIBLE") else "fewer_than_200_strict_l20_users",
                "selected_user_hash": item_list_hash([row["user_id"] for row in history_selected]) if history_selected else "",
                "selected_candidate_hash": selected_candidate_hash(history_selected),
            }
        ]
    )
    return {
        "primary_feasibility": primary_feasibility,
        "primary_users": primary_users,
        "primary_candidates": primary_candidates,
        "history_feasibility": history_feasibility,
        "history_users": history_users,
    }


def generate_candidate_set(
    *,
    user_id: str,
    positive_item_id: str,
    user_seen_items: set[str],
    item_universe: Sequence[str],
) -> list[str] | None:
    negative_exclusions = set(user_seen_items)
    negative_exclusions.add(str(positive_item_id))
    if len(item_universe) - len(negative_exclusions) < NEGATIVE_COUNT:
        return None
    rng = random.Random(stable_int_seed(SELECTION_SEED, "foodcom_candidate_negatives", user_id))
    negatives: list[str] = []
    selected = set()
    while len(negatives) < NEGATIVE_COUNT:
        item_id = str(item_universe[rng.randrange(len(item_universe))])
        if item_id in negative_exclusions or item_id in selected:
            continue
        selected.add(item_id)
        negatives.append(item_id)
    candidates = [str(positive_item_id), *negatives]
    order_rng = random.Random(stable_int_seed(SELECTION_SEED, "foodcom_candidate_order", user_id))
    order_rng.shuffle(candidates)
    return candidates


def visible_history(train_items: Sequence[str], length: int) -> list[str]:
    if length == 0:
        return []
    return [str(item) for item in train_items[-length:]]


def selected_payload(state: Mapping[str, Any], candidates: Sequence[str], candidate_hash: str) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "user_id": str(state["user_id"]),
        "train_history_count": len(state["train_items"]),
        "target_item_id": str(state["target_item"]),
        "target_date": str(state["target_date"]),
        "target_input_order": int(state["target_input_order"]),
        "candidate_item_ids": stable_json([str(item) for item in candidates]),
        "candidate_hash": candidate_hash,
        "unordered_candidate_set_hash": sha256_json(sorted(str(item) for item in candidates)),
        "positive_item_id": str(state["target_item"]),
        "positive_position": [str(item) for item in candidates].index(str(state["target_item"])) + 1,
        "candidate_count": len(candidates),
        "positive_count": 1,
        "negative_count": len(candidates) - 1,
    }


def select_user_ids(users: Sequence[str], count: int, *, label: str) -> list[str]:
    ordered = sorted((str(user) for user in users), key=lambda user: (stable_int_seed(SELECTION_SEED, label, user), user))
    return ordered[: min(count, len(ordered))]


def materialize_selected_payloads(
    user_ids: Sequence[str],
    state_by_user: Mapping[str, Mapping[str, Any]],
    item_universe: Sequence[str],
    *,
    include_history: bool,
    valid_name: Mapping[str, bool] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, user_id in enumerate(user_ids, start=1):
        state = state_by_user[str(user_id)]
        candidate = generate_candidate_set(
            user_id=str(user_id),
            positive_item_id=str(state["target_item"]),
            user_seen_items=set(state["items"]),
            item_universe=item_universe,
        )
        if candidate is None:
            raise ValueError(f"selected user lost candidate feasibility: {user_id}")
        payload = selected_payload(state, candidate, item_list_hash(candidate))
        payload["selection_rank"] = rank
        payload["selection_seed"] = SELECTION_SEED
        if include_history:
            if valid_name is None:
                raise ValueError("valid_name map is required for history payloads")
            for length in HISTORY_LENGTHS:
                history_ids = visible_history(state["train_items"], length)
                payload[f"history_l{length}_item_ids"] = stable_json(history_ids)
                payload[f"history_l{length}_hash"] = item_list_hash(history_ids)
                payload[f"history_l{length}_count"] = len(history_ids)
                payload[f"history_l{length}_names_valid"] = all(bool(valid_name.get(item_id, False)) for item_id in history_ids)
        rows.append(payload)
    return rows


def select_payloads(rows: Sequence[dict[str, Any]], count: int, *, label: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (stable_int_seed(SELECTION_SEED, label, row["user_id"]), row["user_id"]))
    selected = [dict(row) for row in ordered[: min(count, len(ordered))]]
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
        row["selection_seed"] = SELECTION_SEED
    return selected


def materialize_primary_outputs(selected: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_rows = [dict(row) for row in selected]
    candidate_rows: list[dict[str, Any]] = []
    for row in selected:
        candidates = json.loads(str(row["candidate_item_ids"]))
        for position, item_id in enumerate(candidates, start=1):
            candidate_rows.append(
                {
                    "dataset": DATASET,
                    "selection_rank": int(row["selection_rank"]),
                    "user_id": row["user_id"],
                    "candidate_position": position,
                    "candidate_item_id": str(item_id),
                    "is_ground_truth": str(item_id) == str(row["positive_item_id"]),
                    "positive_item_id": row["positive_item_id"],
                    "candidate_hash": row["candidate_hash"],
                }
            )
    return pd.DataFrame(user_rows), pd.DataFrame(candidate_rows)


def selected_candidate_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    return item_list_hash([item_id for row in rows for item_id in json.loads(str(row["candidate_item_ids"]))])


def build_metric_capability(metadata: pd.DataFrame, feature_summary: Mapping[str, Any]) -> pd.DataFrame:
    name_coverage = float(metadata.loc[metadata["field"].eq("name"), "coverage"].iloc[0])
    tags_ready = feature_summary.get("tags", {}).get("nonempty_items", 0) > 0
    rows = [
        capability("HR@5", "utility", "READY_DATA", "held-out target and 20-candidate ranking contract are materialized"),
        capability("NDCG@5", "utility", "READY_DATA", "held-out target and candidate order are materialized"),
        capability("MRR diagnostic", "utility", "READY_DATA", "ranked candidate outputs will be sufficient"),
        capability("SelfInformation@5", "novelty / popularity", "READY_DATA", "train item popularity is available after chronological split"),
        capability("Serendipity@5", "novelty / popularity", "READY_WITH_ADAPTATION", "requires predeclared recipe-feature or popularity baseline definition"),
        capability("ARP@5", "novelty / popularity", "READY_DATA", "train item popularity is available"),
        capability("APLT@5", "novelty / popularity", "READY_DATA", "long-tail threshold can be computed from train popularity"),
        capability("PopREO@5", "novelty / popularity", "BLOCKED_GLOBAL_PROTOCOL", "grouping policy is not recovered globally; not a Food.com metadata blocker"),
        capability("ItemCoverage@5", "diversity / coverage", "READY_DATA", "candidate and catalog item IDs are materialized"),
        capability("Catalog Coverage@5", "diversity / coverage", "READY_DATA", "recipe catalog is available"),
        capability("OIC@5", "diversity / coverage", "READY_DATA", "item coverage over observed recommendation outputs can be computed later"),
        capability("Shannon category entropy@5", "diversity / coverage", "READY_WITH_ADAPTATION" if tags_ready else "BLOCKED_DATASET_METADATA", "tags provide the canonical category feature space"),
        capability("ILD@5", "diversity / coverage", "READY_WITH_ADAPTATION" if tags_ready else "BLOCKED_DATASET_METADATA", "requires a declared Jaccard/cosine distance over tags or ingredients"),
        capability("Calibration@5", "calibration / fairness", "READY_WITH_ADAPTATION" if tags_ready else "BLOCKED_DATASET_METADATA", "requires user-profile feature distribution over tags or ingredients"),
        capability("Gini@5", "calibration / fairness", "READY_DATA", "exposure distribution over recommended item IDs is sufficient"),
        capability("DPD@5", "calibration / fairness", "BLOCKED_GLOBAL_PROTOCOL", "requires a global group definition before dataset-specific execution"),
        capability("Jain's Index@5", "calibration / fairness", "READY_DATA", "exposure distribution over recommended item IDs is sufficient"),
        capability("hallucination", "LLM behaviour", "READY_DATA" if name_coverage > 0 else "BLOCKED_DATASET_METADATA", "strict recipe-name candidate catalog is available"),
        capability("strict parse validity", "LLM behaviour", "READY_DATA", f"parser contract {PARSER_CONTRACT_ID} is global"),
        capability("duplicate rate", "LLM behaviour", "READY_DATA", "ranked candidate outputs can be checked for duplicates"),
        capability("invalid reason distribution", "LLM behaviour", "READY_DATA", "strict parser emits machine-readable invalid reasons"),
        capability("latency", "efficiency", "NOT_APPLICABLE", "execution telemetry, not raw dataset material capability"),
        capability("throughput", "efficiency", "NOT_APPLICABLE", "execution telemetry, not raw dataset material capability"),
        capability("runtime distributions", "efficiency", "NOT_APPLICABLE", "execution telemetry, not raw dataset material capability"),
        capability("tokens/cost where available", "efficiency", "NOT_APPLICABLE", "provider telemetry, not raw dataset material capability"),
    ]
    return pd.DataFrame(rows)


def build_paper_dimension_capability(feasibility: Mapping[str, pd.DataFrame], feature_summary: Mapping[str, Any]) -> pd.DataFrame:
    primary_ready = feasibility["primary_feasibility"]["status"].iloc[0] == "FOODCOM_PRIMARY_1000_FEASIBLE"
    history_ready = feasibility["history_feasibility"]["status"].iloc[0] == "FOODCOM_HISTORY_200_FEASIBLE"
    feature_ready = feature_summary.get("tags", {}).get("nonempty_items", 0) > 0
    rows = [
        paper_capability("BASE", "READY" if primary_ready else "BLOCKED_DATASET", "primary 1000 strict-name candidate gate"),
        paper_capability("LLMRank", "READY" if primary_ready else "BLOCKED_DATASET", "same candidate contract with LLMRank prompt family"),
        paper_capability("History L0/1/2/5/10/20", "READY" if history_ready else "BLOCKED_DATASET", "paired 200 strict L20 gate"),
        paper_capability("CandDif / candidate-position bias", "READY_WITH_ADAPTATION" if primary_ready else "BLOCKED_DATASET", "requires future order perturbation cells over frozen candidates"),
        paper_capability("hallucination", "READY" if primary_ready else "BLOCKED_DATASET", "candidate recipe names support strict candidate-set matching"),
        paper_capability("profile generation", "READY_WITH_ADAPTATION" if feature_ready else "BLOCKED_DATASET", "recipe history can be rendered from names/tags/ingredients"),
        paper_capability("profile-only recommendation", "READY_WITH_ADAPTATION" if primary_ready and feature_ready else "BLOCKED_DATASET", "profile prompt adaptation required"),
        paper_capability("profile + history recommendation", "READY_WITH_ADAPTATION" if history_ready and feature_ready else "BLOCKED_DATASET", "profile prompt plus visible history adaptation required"),
        paper_capability("ranking", "READY" if primary_ready else "BLOCKED_DATASET", "one held-out target and 20-candidate sets"),
        paper_capability("reranking", "READY" if primary_ready else "BLOCKED_DATASET", "candidate order is frozen"),
        paper_capability("candidate-order sensitivity", "READY_WITH_ADAPTATION" if primary_ready else "BLOCKED_DATASET", "requires future deterministic order variants"),
        paper_capability("candidate-pool sensitivity", "READY_WITH_ADAPTATION" if primary_ready else "BLOCKED_DATASET", "requires future deterministic pool-size variants"),
        paper_capability("K-S representativeness if globally recoverable", "BLOCKED_GLOBAL_PROTOCOL", "global K-S representativeness protocol is not recovered here"),
    ]
    return pd.DataFrame(rows)


def build_traditional_model_capability(split_stats: pd.DataFrame, feature_summary: Mapping[str, Any]) -> pd.DataFrame:
    train_rows = int(split_stats.loc[split_stats["metric"].eq("train_rows"), "value"].iloc[0])
    tags_ready = feature_summary.get("tags", {}).get("nonempty_items", 0) > 0
    rows = [
        model_capability("Random", "candidate sets", "20-candidate sets", "none", "READY_DATA" if train_rows >= 0 else "BLOCKED_DATASET", ""),
        model_capability("MostPop", "train interactions", "chronological train rows", "none", "READY_DATA" if train_rows > 0 else "BLOCKED_DATASET", ""),
        model_capability("BM25-like", "item text metadata", "name, description, tags, ingredients", "recipe text field concatenation", "READY_WITH_ADAPTATION" if tags_ready else "BLOCKED_DATASET_METADATA", ""),
        model_capability("BPRMF", "user-item train interactions", "chronological train rows with IDs", "map raw IDs to contiguous internals in future preprocessing", "READY_DATA", ""),
        model_capability("GRU4Rec", "ordered user sequences", "date plus input-order chronological sequences", "day-level tie policy documented", "READY_WITH_ADAPTATION", ""),
        model_capability("SASRec", "ordered user sequences", "date plus input-order chronological sequences", "day-level tie policy documented", "READY_WITH_ADAPTATION", ""),
        model_capability("LightGCN", "user-item train graph", "chronological train rows with IDs", "map raw IDs to contiguous internals in future preprocessing", "READY_DATA", ""),
    ]
    return pd.DataFrame(rows)


def build_summary(
    archive_manifest: Mapping[str, Any],
    raw_stats: pd.DataFrame,
    split_stats: pd.DataFrame,
    feasibility: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    primary = feasibility["primary_feasibility"].iloc[0]
    history = feasibility["history_feasibility"].iloc[0]
    primary_ready = primary["status"] == "FOODCOM_PRIMARY_1000_FEASIBLE"
    history_ready = history["status"] == "FOODCOM_HISTORY_200_FEASIBLE"
    decision = (
        "FOODCOM_D0A_FEASIBLE_WITH_ADAPTATIONS"
        if primary_ready and history_ready
        else "FOODCOM_D0A_DATASET_BLOCKED"
    )
    return {
        "phase_id": PHASE_ID,
        "decision": decision,
        "archive_sha256": archive_manifest["sha256"],
        "archive_file_size_bytes": archive_manifest["file_size_bytes"],
        "interaction_rows": scalar_metric(raw_stats, "interaction_rows"),
        "unique_users": scalar_metric(raw_stats, "unique_users"),
        "unique_recipes": scalar_metric(raw_stats, "unique_recipes_in_raw_recipes"),
        "users_with_train_history_ge_20": scalar_metric(split_stats, "users_with_train_history_ge_20"),
        "primary_status": primary["status"],
        "primary_strict_name_eligible_users": int(primary["final_strict_name_eligible_users"]),
        "history_status": history["status"],
        "history_strict_paired_l20_users": int(history["strict_paired_l20_users"]),
        "llm_requests_executed": EXPECTED_LLM_REQUESTS,
        "training_runs_executed": EXPECTED_TRAINING_RUNS,
        "remote_api_calls_executed": EXPECTED_REMOTE_API_CALLS,
        "downloads_executed": EXPECTED_DOWNLOADS,
    }


def build_no_inference_manifest(
    root: Path,
    archive_manifest: Mapping[str, Any],
    extracted_hashes: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(summary),
        "created_at": utc_now(),
        "source_commit": git_head(root),
        "raw_archive": {
            "relative_path": archive_manifest["relative_path"],
            "sha256": archive_manifest["sha256"],
            "file_size_bytes": archive_manifest["file_size_bytes"],
        },
        "required_extracted_files": extracted_hashes["files"],
    }


def render_report(outputs: Mapping[str, Any]) -> str:
    summary = outputs["summary"]
    archive = outputs["archive_inventory"]
    schema = outputs["schema_audit"]
    raw = outputs["raw_statistics"]
    primary = outputs["primary_feasibility"]
    history = outputs["history_feasibility"]
    metadata = outputs["metadata_coverage"]
    features = outputs["feature_coverage"]
    duplicate = outputs["duplicate_timestamp_audit"]
    split = outputs["split_statistics"]
    metrics = outputs["metric_capability"]
    paper = outputs["paper_dimension_capability"]
    models = outputs["traditional_model_capability"]
    return f"""# D0A Food.com Material Raw Intake and Final-Protocol Feasibility

Decision: `{summary['decision']}`

No LLM inference, recommender training, remote API calls, network downloads,
purges, commits, competing-dataset modifications, or `original_repo/`
modifications were performed.

## Raw Archive

- File size: `{summary['archive_file_size_bytes']}` bytes
- SHA256: `{summary['archive_sha256']}`

## Archive Contents

{archive.to_markdown(index=False)}

## Schema

{schema.to_markdown(index=False)}

## Raw Statistics

{raw.to_markdown(index=False)}

## Final Split

{split.to_markdown(index=False)}

## Timestamp and Duplicate Audit

{duplicate.to_markdown(index=False)}

## Name and Metadata Coverage

{metadata.to_markdown(index=False)}

## Feature Coverage

{features.to_markdown(index=False)}

## Primary 1000 Feasibility

{primary.to_markdown(index=False)}

## History 200 Feasibility

{history.to_markdown(index=False)}

## Metric Capability

{metrics.to_markdown(index=False)}

## Paper-Dimension Capability

{paper.to_markdown(index=False)}

## Traditional Model Capability

{models.to_markdown(index=False)}
"""


def text_state(value: Any) -> str:
    if pd.isna(value):
        return "null"
    raw = str(value)
    lowered = raw.strip().lower()
    if lowered in {"nan", "none", "null"}:
        return "null"
    if raw == "":
        return "empty"
    if raw.strip() == "":
        return "whitespace"
    return "usable"


def parse_literal_list(value: str) -> list[Any] | None:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def normalize_feature_token(value: Any) -> str:
    return str(value).strip().lower()


def metric_row(metric: str, value: Any) -> dict[str, Any]:
    return {"metric": metric, "value": value}


def exact_duplicate_count(interactions: pd.DataFrame) -> int:
    if "exact_duplicate_rows_full" in interactions.attrs:
        return int(interactions.attrs["exact_duplicate_rows_full"])
    return int(interactions.duplicated().sum())


def audit_row(metric: str, value: int, policy: str, notes: str) -> dict[str, Any]:
    return {"metric": metric, "value": value, "policy": policy, "notes": notes}


def capability(metric: str, family: str, status: str, evidence: str) -> dict[str, Any]:
    return {"metric": metric, "family": family, "status": status, "evidence": evidence}


def paper_capability(dimension: str, status: str, evidence: str) -> dict[str, Any]:
    return {"dimension": dimension, "status": status, "evidence": evidence}


def model_capability(
    model: str,
    required_inputs: str,
    available_inputs: str,
    adaptation_required: str,
    status: str,
    blocker: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "required_inputs": required_inputs,
        "available_inputs": available_inputs,
        "adaptation_required": adaptation_required,
        "status": status,
        "blocker": blocker,
    }


def scalar_metric(frame: pd.DataFrame, metric: str) -> Any:
    value = frame.loc[frame["metric"].eq(metric), "value"].iloc[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head(root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


if __name__ == "__main__":
    raise SystemExit(main())

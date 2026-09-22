"""Validate staged Plan C raw datasets for reconstruction readiness.

Run from the repository root:

    python src/audit/validate_plan_c_raw_datasets.py --staging-dir data/external/plan_c

This script samples raw files in place. It never downloads files, permanently
extracts archives, creates processed datasets, trains models, computes metrics,
or modifies ``original_repo/``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import io
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUMMARY_COLUMNS = [
    "dataset",
    "component",
    "source_path",
    "archive_member",
    "file_found",
    "member_found",
    "status",
    "sampled_rows",
    "parsed_rows",
    "parse_errors",
    "required_fields_present",
    "supports_reconstruction_role",
    "notes",
]

SCHEMA_COLUMNS = [
    "dataset",
    "component",
    "source_path",
    "archive_member",
    "field",
    "required",
    "present_in_sample_rows",
    "sampled_parsed_rows",
    "status",
    "notes",
]

STATS_COLUMNS = [
    "dataset",
    "component",
    "source_path",
    "archive_member",
    "sampled_rows",
    "parsed_rows",
    "parse_errors",
    "unique_users_sample",
    "unique_items_sample",
    "min_timestamp_sample",
    "max_timestamp_sample",
    "rating_distribution_sample",
    "metadata_records_with_title",
    "metadata_records_with_categories",
    "metadata_records_with_description",
    "notes",
]

READINESS_COLUMNS = [
    "dataset",
    "exact_original_artifact_recovered",
    "raw_reconstruction_input_available",
    "sequential_interactions_available",
    "item_metadata_available",
    "chronological_split_supported",
    "candidate_generation_possible",
    "expected_reconstruction_status",
    "notes",
]

AMAZON_REVIEW_FIELDS = ["reviewerID", "asin", "overall", "unixReviewTime"]
AMAZON_META_FIELDS = ["asin", "title", "categories", "description", "price", "brand"]
ML1M_RATINGS_FIELDS = ["user_id", "item_id", "rating", "timestamp"]
ML1M_MOVIES_FIELDS = ["item_id", "title", "genres"]
ML1M_USERS_FIELDS = ["user_id", "gender", "age", "occupation", "zip"]
LASTFM_USER_ARTISTS_FIELDS = ["userID", "artistID", "weight"]
LASTFM_ARTISTS_FIELDS = ["id", "name"]
LASTFM_TAGGED_FIELDS = ["userID", "artistID", "tagID", "day", "month", "year"]

CORE_PROCESSED_FILENAMES = {
    "sequential_data.txt",
    "negative_samples.txt",
    "datamaps.json",
    "meta.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate staged Plan C raw datasets without extracting archives."
    )
    parser.add_argument(
        "--staging-dir",
        required=True,
        type=Path,
        help="Directory containing manually staged Plan C raw artifacts.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Maximum data rows sampled from each raw file/member.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
        help="Directory for validation CSV outputs.",
    )
    return parser.parse_args(argv)


def validate_raw_datasets(
    staging_dir: str | Path,
    *,
    sample_size: int = 1000,
    project_root: str | Path | None = None,
    output_dir: str | Path = "results/tables",
) -> dict[str, pd.DataFrame]:
    """Validate staged raw datasets and write Plan C reconstruction-readiness tables."""

    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    staging = _resolve_path(Path(staging_dir), root)
    output = _resolve_path(Path(output_dir), root)
    sample_size = max(int(sample_size), 0)

    summary_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []

    for dataset, review_name, meta_name in [
        ("amazon_beauty", "reviews_Beauty_5.json.gz", "meta_Beauty.json.gz"),
        (
            "amazon_sports",
            "reviews_Sports_and_Outdoors_5.json.gz",
            "meta_Sports_and_Outdoors.json.gz",
        ),
    ]:
        _validate_amazon_reviews(
            staging / review_name,
            dataset=dataset,
            sample_size=sample_size,
            root=root,
            summary_rows=summary_rows,
            schema_rows=schema_rows,
            stats_rows=stats_rows,
        )
        _validate_amazon_metadata(
            staging / meta_name,
            dataset=dataset,
            sample_size=sample_size,
            root=root,
            summary_rows=summary_rows,
            schema_rows=schema_rows,
            stats_rows=stats_rows,
        )

    _validate_ml1m_zip(
        staging / "ml-1m.zip",
        sample_size=sample_size,
        root=root,
        summary_rows=summary_rows,
        schema_rows=schema_rows,
        stats_rows=stats_rows,
    )
    _validate_lastfm_zip(
        staging / "hetrec2011-lastfm-2k.zip",
        sample_size=sample_size,
        root=root,
        summary_rows=summary_rows,
        schema_rows=schema_rows,
        stats_rows=stats_rows,
    )

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    schema_df = pd.DataFrame(schema_rows, columns=SCHEMA_COLUMNS)
    stats_df = pd.DataFrame(stats_rows, columns=STATS_COLUMNS)
    readiness_df = pd.DataFrame(
        _build_readiness_rows(summary_df, staging),
        columns=READINESS_COLUMNS,
    )

    output.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output / "plan_c_raw_dataset_validation_summary.csv", index=False)
    schema_df.to_csv(output / "plan_c_raw_dataset_schema_details.csv", index=False)
    stats_df.to_csv(output / "plan_c_raw_dataset_sample_stats.csv", index=False)
    readiness_df.to_csv(
        output / "plan_c_reconstruction_readiness_summary.csv",
        index=False,
    )
    return {
        "summary": summary_df,
        "schema": schema_df,
        "stats": stats_df,
        "readiness": readiness_df,
    }


def _validate_amazon_reviews(
    path: Path,
    *,
    dataset: str,
    sample_size: int,
    root: Path,
    summary_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
) -> None:
    component = "amazon_reviews"
    records, sampled_rows, parse_errors, parser_notes = _sample_json_gz(path, sample_size)
    schema = _schema_counts(records, AMAZON_REVIEW_FIELDS)
    required_present = _required_present(schema, AMAZON_REVIEW_FIELDS, len(records))
    timestamps = _numeric_values(record.get("unixReviewTime") for record in records)
    ratings = _rating_distribution(record.get("overall") for record in records)
    status = "valid" if path.exists() and records and required_present else "missing_or_invalid"
    if not path.exists():
        status = "missing"

    summary_rows.append(
        _summary_row(
            dataset=dataset,
            component=component,
            source_path=_display_path(path, root),
            file_found=path.exists(),
            status=status,
            sampled_rows=sampled_rows,
            parsed_rows=len(records),
            parse_errors=parse_errors,
            required_fields_present=required_present,
            supports_reconstruction_role=bool(required_present and timestamps),
            notes=_join_notes(
                parser_notes,
                "Supports sequential interactions if reviewerID, asin, rating, and unixReviewTime are present."
                if required_present and timestamps
                else "Missing required interaction fields or timestamps in the sampled rows.",
            ),
        )
    )
    _extend_schema_rows(
        schema_rows,
        dataset=dataset,
        component=component,
        source_path=_display_path(path, root),
        archive_member="",
        fields=AMAZON_REVIEW_FIELDS,
        required_fields=AMAZON_REVIEW_FIELDS,
        schema=schema,
        parsed_rows=len(records),
    )
    stats_rows.append(
        _stats_row(
            dataset=dataset,
            component=component,
            source_path=_display_path(path, root),
            sampled_rows=sampled_rows,
            parsed_rows=len(records),
            parse_errors=parse_errors,
            unique_users=_unique_count(record.get("reviewerID") for record in records),
            unique_items=_unique_count(record.get("asin") for record in records),
            min_timestamp=min(timestamps) if timestamps else "",
            max_timestamp=max(timestamps) if timestamps else "",
            rating_distribution=ratings,
            notes="Amazon review sample; counts are sample diagnostics, not final dataset statistics.",
        )
    )


def _validate_amazon_metadata(
    path: Path,
    *,
    dataset: str,
    sample_size: int,
    root: Path,
    summary_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
) -> None:
    component = "amazon_metadata"
    records, sampled_rows, parse_errors, parser_notes = _sample_json_gz(path, sample_size)
    schema = _schema_counts(records, AMAZON_META_FIELDS)
    required_fields = ["asin"]
    asin_present = _required_present(schema, required_fields, len(records))
    has_catalog_text = bool(
        schema.get("title", 0) or schema.get("categories", 0) or schema.get("description", 0)
    )
    status = "valid" if path.exists() and records and asin_present and has_catalog_text else "missing_or_invalid"
    if not path.exists():
        status = "missing"

    summary_rows.append(
        _summary_row(
            dataset=dataset,
            component=component,
            source_path=_display_path(path, root),
            file_found=path.exists(),
            status=status,
            sampled_rows=sampled_rows,
            parsed_rows=len(records),
            parse_errors=parse_errors,
            required_fields_present=bool(asin_present and has_catalog_text),
            supports_reconstruction_role=bool(asin_present and has_catalog_text),
            notes=_join_notes(
                parser_notes,
                "Supports catalog reconstruction if asin and at least one text/metadata field are present."
                if asin_present and has_catalog_text
                else "Missing asin or usable catalog text fields in sampled rows.",
            ),
        )
    )
    _extend_schema_rows(
        schema_rows,
        dataset=dataset,
        component=component,
        source_path=_display_path(path, root),
        archive_member="",
        fields=AMAZON_META_FIELDS,
        required_fields=required_fields,
        schema=schema,
        parsed_rows=len(records),
    )
    stats_rows.append(
        _stats_row(
            dataset=dataset,
            component=component,
            source_path=_display_path(path, root),
            sampled_rows=sampled_rows,
            parsed_rows=len(records),
            parse_errors=parse_errors,
            unique_items=_unique_count(record.get("asin") for record in records),
            metadata_records_with_title=_present_count(record.get("title") for record in records),
            metadata_records_with_categories=_present_count(record.get("categories") for record in records),
            metadata_records_with_description=_present_count(record.get("description") for record in records),
            notes="Amazon metadata sample; coverage is sample-level only.",
        )
    )


def _validate_ml1m_zip(
    path: Path,
    *,
    sample_size: int,
    root: Path,
    summary_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
) -> None:
    dataset = "ml1m"
    expected = {
        "ratings": ("ml-1m/ratings.dat", ML1M_RATINGS_FIELDS, True),
        "movies": ("ml-1m/movies.dat", ML1M_MOVIES_FIELDS, True),
        "users": ("ml-1m/users.dat", ML1M_USERS_FIELDS, False),
    }
    if not path.exists():
        for component, (member, fields, required) in expected.items():
            _append_missing_zip_rows(
                summary_rows,
                schema_rows,
                stats_rows,
                dataset=dataset,
                component=f"ml1m_{component}",
                source_path=_display_path(path, root),
                archive_member=member,
                fields=fields,
                required=required,
                notes="MovieLens-1M ZIP is missing from staging.",
            )
        return

    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            for component, (expected_member, fields, required_component) in expected.items():
                member = _find_member(names, expected_member)
                if member is None:
                    _append_missing_zip_rows(
                        summary_rows,
                        schema_rows,
                        stats_rows,
                        dataset=dataset,
                        component=f"ml1m_{component}",
                        source_path=_display_path(path, root),
                        archive_member=expected_member,
                        fields=fields,
                        required=required_component,
                        notes="Expected MovieLens-1M member was not found in the ZIP.",
                    )
                    continue
                lines = _sample_zip_text_member(archive, member, sample_size)
                _validate_ml1m_member(
                    lines,
                    dataset=dataset,
                    component=f"ml1m_{component}",
                    source_path=_display_path(path, root),
                    archive_member=member,
                    fields=fields,
                    required_component=required_component,
                    summary_rows=summary_rows,
                    schema_rows=schema_rows,
                    stats_rows=stats_rows,
                )
    except zipfile.BadZipFile:
        for component, (member, fields, required) in expected.items():
            _append_missing_zip_rows(
                summary_rows,
                schema_rows,
                stats_rows,
                dataset=dataset,
                component=f"ml1m_{component}",
                source_path=_display_path(path, root),
                archive_member=member,
                fields=fields,
                required=required,
                notes="MovieLens-1M ZIP could not be opened.",
            )


def _validate_ml1m_member(
    lines: list[str],
    *,
    dataset: str,
    component: str,
    source_path: str,
    archive_member: str,
    fields: list[str],
    required_component: bool,
    summary_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
) -> None:
    parsed, parse_errors = _parse_delimited_lines(lines, "::", fields)
    schema = _schema_counts(parsed, fields)
    required_present = _required_present(schema, fields, len(parsed))
    timestamps = _numeric_values(record.get("timestamp") for record in parsed)
    status = "valid" if parsed and required_present else "missing_or_invalid"
    supports_role = bool(parsed and required_present)

    summary_rows.append(
        _summary_row(
            dataset=dataset,
            component=component,
            source_path=source_path,
            archive_member=archive_member,
            file_found=True,
            member_found=True,
            status=status,
            sampled_rows=len(lines),
            parsed_rows=len(parsed),
            parse_errors=parse_errors,
            required_fields_present=required_present,
            supports_reconstruction_role=supports_role or not required_component,
            notes="MovieLens-1M member parsed with '::' separator.",
        )
    )
    _extend_schema_rows(
        schema_rows,
        dataset=dataset,
        component=component,
        source_path=source_path,
        archive_member=archive_member,
        fields=fields,
        required_fields=fields if required_component else [],
        schema=schema,
        parsed_rows=len(parsed),
    )
    stats_rows.append(
        _stats_row(
            dataset=dataset,
            component=component,
            source_path=source_path,
            archive_member=archive_member,
            sampled_rows=len(lines),
            parsed_rows=len(parsed),
            parse_errors=parse_errors,
            unique_users=_unique_count(record.get("user_id") for record in parsed),
            unique_items=_unique_count(record.get("item_id") for record in parsed),
            min_timestamp=min(timestamps) if timestamps else "",
            max_timestamp=max(timestamps) if timestamps else "",
            rating_distribution=_rating_distribution(record.get("rating") for record in parsed),
            metadata_records_with_title=_present_count(record.get("title") for record in parsed),
            metadata_records_with_categories=_present_count(record.get("genres") for record in parsed),
            notes="MovieLens-1M sample; counts are sample diagnostics.",
        )
    )


def _validate_lastfm_zip(
    path: Path,
    *,
    sample_size: int,
    root: Path,
    summary_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
) -> None:
    dataset = "lastfm"
    expected = {
        "user_artists": ("user_artists.dat", LASTFM_USER_ARTISTS_FIELDS, True),
        "artists": ("artists.dat", LASTFM_ARTISTS_FIELDS, True),
        "user_taggedartists": ("user_taggedartists.dat", LASTFM_TAGGED_FIELDS, False),
    }
    if not path.exists():
        for component, (member, fields, required) in expected.items():
            _append_missing_zip_rows(
                summary_rows,
                schema_rows,
                stats_rows,
                dataset=dataset,
                component=f"lastfm_{component}",
                source_path=_display_path(path, root),
                archive_member=member,
                fields=fields,
                required=required,
                notes="LastFM HetRec ZIP is missing from staging.",
            )
        return

    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            for component, (expected_member, fields, required_component) in expected.items():
                member = _find_member(names, expected_member)
                if member is None:
                    _append_missing_zip_rows(
                        summary_rows,
                        schema_rows,
                        stats_rows,
                        dataset=dataset,
                        component=f"lastfm_{component}",
                        source_path=_display_path(path, root),
                        archive_member=expected_member,
                        fields=fields,
                        required=required_component,
                        notes="Expected LastFM member was not found in the ZIP.",
                    )
                    continue
                lines = _sample_zip_text_member(archive, member, sample_size, skip_header=True)
                _validate_lastfm_member(
                    lines,
                    dataset=dataset,
                    component=f"lastfm_{component}",
                    source_path=_display_path(path, root),
                    archive_member=member,
                    fields=fields,
                    required_component=required_component,
                    summary_rows=summary_rows,
                    schema_rows=schema_rows,
                    stats_rows=stats_rows,
                )
    except zipfile.BadZipFile:
        for component, (member, fields, required) in expected.items():
            _append_missing_zip_rows(
                summary_rows,
                schema_rows,
                stats_rows,
                dataset=dataset,
                component=f"lastfm_{component}",
                source_path=_display_path(path, root),
                archive_member=member,
                fields=fields,
                required=required,
                notes="LastFM HetRec ZIP could not be opened.",
            )


def _validate_lastfm_member(
    lines: list[str],
    *,
    dataset: str,
    component: str,
    source_path: str,
    archive_member: str,
    fields: list[str],
    required_component: bool,
    summary_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
) -> None:
    parsed, parse_errors = _parse_delimited_lines(lines, "\t", fields, minimum_fields=2)
    schema = _schema_counts(parsed, fields)
    required_fields = fields if required_component else []
    required_present = _required_present(schema, required_fields, len(parsed)) if required_fields else True
    timestamps = _lastfm_timestamp_values(parsed)
    status = "valid" if parsed and required_present else "missing_or_invalid"

    summary_rows.append(
        _summary_row(
            dataset=dataset,
            component=component,
            source_path=source_path,
            archive_member=archive_member,
            file_found=True,
            member_found=True,
            status=status,
            sampled_rows=len(lines),
            parsed_rows=len(parsed),
            parse_errors=parse_errors,
            required_fields_present=required_present,
            supports_reconstruction_role=bool(parsed and required_present),
            notes="LastFM HetRec member parsed with tab separator.",
        )
    )
    _extend_schema_rows(
        schema_rows,
        dataset=dataset,
        component=component,
        source_path=source_path,
        archive_member=archive_member,
        fields=fields,
        required_fields=required_fields,
        schema=schema,
        parsed_rows=len(parsed),
    )
    stats_rows.append(
        _stats_row(
            dataset=dataset,
            component=component,
            source_path=source_path,
            archive_member=archive_member,
            sampled_rows=len(lines),
            parsed_rows=len(parsed),
            parse_errors=parse_errors,
            unique_users=_unique_count(record.get("userID") for record in parsed),
            unique_items=_unique_count(
                record.get("artistID") or record.get("id") for record in parsed
            ),
            min_timestamp=min(timestamps) if timestamps else "",
            max_timestamp=max(timestamps) if timestamps else "",
            notes="LastFM sample; user_artists has weights but no listening timestamp.",
        )
    )


def _sample_json_gz(path: Path, sample_size: int) -> tuple[list[dict[str, Any]], int, int, str]:
    if not path.exists():
        return [], 0, 0, "File is missing."
    records: list[dict[str, Any]] = []
    sampled_rows = 0
    parse_errors = 0
    parsers: Counter[str] = Counter()
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if sampled_rows >= sample_size:
                break
            line = line.strip()
            if not line:
                continue
            sampled_rows += 1
            record, parser_name = _parse_record_line(line)
            if record is None:
                parse_errors += 1
                continue
            records.append(record)
            parsers[parser_name] += 1
    parser_note = "Parsers used: " + ";".join(f"{key}:{value}" for key, value in sorted(parsers.items()))
    return records, sampled_rows, parse_errors, parser_note


def _parse_record_line(line: str) -> tuple[dict[str, Any] | None, str]:
    try:
        value = json.loads(line)
        return (value, "json") if isinstance(value, dict) else (None, "json_non_dict")
    except json.JSONDecodeError:
        pass
    try:
        value = ast.literal_eval(line)
        return (value, "python_literal") if isinstance(value, dict) else (None, "literal_non_dict")
    except (ValueError, SyntaxError):
        return None, "unparsed"


def _sample_zip_text_member(
    archive: zipfile.ZipFile,
    member: str,
    sample_size: int,
    *,
    skip_header: bool = False,
) -> list[str]:
    lines: list[str] = []
    with archive.open(member, "r") as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        if skip_header:
            next(text, None)
        for line in text:
            if len(lines) >= sample_size:
                break
            line = line.rstrip("\n\r")
            if line:
                lines.append(line)
    return lines


def _parse_delimited_lines(
    lines: list[str],
    separator: str,
    fields: list[str],
    *,
    minimum_fields: int | None = None,
) -> tuple[list[dict[str, str]], int]:
    parsed: list[dict[str, str]] = []
    parse_errors = 0
    minimum = minimum_fields or len(fields)
    for line in lines:
        parts = line.split(separator)
        if len(parts) < minimum:
            parse_errors += 1
            continue
        record = {field: parts[index] for index, field in enumerate(fields) if index < len(parts)}
        parsed.append(record)
    return parsed, parse_errors


def _schema_counts(records: list[dict[str, Any]], fields: list[str]) -> dict[str, int]:
    return {
        field: sum(1 for record in records if _has_value(record.get(field)))
        for field in fields
    }


def _extend_schema_rows(
    schema_rows: list[dict[str, Any]],
    *,
    dataset: str,
    component: str,
    source_path: str,
    archive_member: str,
    fields: list[str],
    required_fields: list[str],
    schema: dict[str, int],
    parsed_rows: int,
) -> None:
    for field in fields:
        present = int(schema.get(field, 0))
        if parsed_rows == 0:
            status = "no_parsed_rows"
        elif present == parsed_rows:
            status = "present_all"
        elif present > 0:
            status = "present_partial"
        else:
            status = "missing"
        schema_rows.append(
            {
                "dataset": dataset,
                "component": component,
                "source_path": source_path,
                "archive_member": archive_member,
                "field": field,
                "required": field in required_fields,
                "present_in_sample_rows": present,
                "sampled_parsed_rows": parsed_rows,
                "status": status,
                "notes": "Field presence is sampled and does not validate full-file coverage.",
            }
        )


def _append_missing_zip_rows(
    summary_rows: list[dict[str, Any]],
    schema_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
    *,
    dataset: str,
    component: str,
    source_path: str,
    archive_member: str,
    fields: list[str],
    required: bool,
    notes: str,
) -> None:
    summary_rows.append(
        _summary_row(
            dataset=dataset,
            component=component,
            source_path=source_path,
            archive_member=archive_member,
            file_found=False,
            member_found=False,
            status="missing",
            notes=notes,
        )
    )
    _extend_schema_rows(
        schema_rows,
        dataset=dataset,
        component=component,
        source_path=source_path,
        archive_member=archive_member,
        fields=fields,
        required_fields=fields if required else [],
        schema={},
        parsed_rows=0,
    )
    stats_rows.append(
        _stats_row(
            dataset=dataset,
            component=component,
            source_path=source_path,
            archive_member=archive_member,
            notes=notes,
        )
    )


def _build_readiness_rows(summary_df: pd.DataFrame, staging: Path) -> list[dict[str, Any]]:
    exact_recovered = "yes" if _core_processed_artifacts_found(staging) else "no"
    rows: list[dict[str, Any]] = []
    for dataset in ["amazon_beauty", "amazon_sports", "ml1m", "lastfm"]:
        dataset_rows = summary_df[summary_df["dataset"] == dataset]
        raw_available = bool((dataset_rows["status"] == "valid").any())
        interactions = _component_supports(dataset_rows, _interaction_components(dataset))
        metadata = _component_supports(dataset_rows, _metadata_components(dataset))
        chronological = _chronological_status(dataset, dataset_rows)
        candidate_possible = "yes" if interactions == "yes" and metadata == "yes" else "no"
        status = _readiness_status(
            dataset=dataset,
            raw_available=raw_available,
            interactions=interactions,
            metadata=metadata,
            chronological=chronological,
        )
        rows.append(
            {
                "dataset": dataset,
                "exact_original_artifact_recovered": exact_recovered if dataset.startswith("amazon_") else "no",
                "raw_reconstruction_input_available": "yes" if raw_available else "no",
                "sequential_interactions_available": interactions,
                "item_metadata_available": metadata,
                "chronological_split_supported": chronological,
                "candidate_generation_possible": candidate_possible,
                "expected_reconstruction_status": status,
                "notes": _readiness_notes(dataset, status, chronological),
            }
        )
    return rows


def _component_supports(dataset_rows: pd.DataFrame, components: set[str]) -> str:
    if dataset_rows.empty:
        return "no"
    selected = dataset_rows[dataset_rows["component"].isin(components)]
    if selected.empty:
        return "no"
    return "yes" if bool(selected["supports_reconstruction_role"].astype(bool).all()) else "no"


def _interaction_components(dataset: str) -> set[str]:
    return {
        "amazon_beauty": {"amazon_reviews"},
        "amazon_sports": {"amazon_reviews"},
        "ml1m": {"ml1m_ratings"},
        "lastfm": {"lastfm_user_artists"},
    }[dataset]


def _metadata_components(dataset: str) -> set[str]:
    return {
        "amazon_beauty": {"amazon_metadata"},
        "amazon_sports": {"amazon_metadata"},
        "ml1m": {"ml1m_movies"},
        "lastfm": {"lastfm_artists"},
    }[dataset]


def _chronological_status(dataset: str, dataset_rows: pd.DataFrame) -> str:
    if dataset in {"amazon_beauty", "amazon_sports"}:
        rows = dataset_rows[dataset_rows["component"] == "amazon_reviews"]
        return "yes" if _summary_valid(rows) else "no"
    if dataset == "ml1m":
        rows = dataset_rows[dataset_rows["component"] == "ml1m_ratings"]
        return "yes" if _summary_valid(rows) else "no"
    tagged = dataset_rows[dataset_rows["component"] == "lastfm_user_taggedartists"]
    interactions = dataset_rows[dataset_rows["component"] == "lastfm_user_artists"]
    if _summary_valid(tagged) and _summary_valid(interactions):
        return "partial"
    return "no"


def _summary_valid(rows: pd.DataFrame) -> bool:
    return bool(not rows.empty and (rows["status"] == "valid").all())


def _readiness_status(
    *,
    dataset: str,
    raw_available: bool,
    interactions: str,
    metadata: str,
    chronological: str,
) -> str:
    if not raw_available:
        return "blocked_missing_raw"
    if interactions != "yes" or metadata != "yes":
        return "blocked_unclear_protocol"
    if dataset == "lastfm" and chronological != "yes":
        return "extension_only"
    return "reconstructible_from_public_raw"


def _readiness_notes(dataset: str, status: str, chronological: str) -> str:
    if dataset in {"amazon_beauty", "amazon_sports"}:
        return (
            "Raw Amazon 5-core files can support a controlled reconstruction, "
            "but exact Beauty/Sports paper reproduction remains blocked until "
            "author processed sequential_data, negative_samples, datamaps, and meta files are recovered."
        )
    if dataset == "ml1m":
        return (
            "MovieLens-1M raw archive supports reconstruction; paper-equivalent "
            "sampling, candidates, and protocol still require validation."
        )
    if dataset == "lastfm" and chronological == "partial":
        return (
            "LastFM raw files support interaction and metadata reconstruction, "
            "but user_artists has no listening timestamp; tag timestamps provide only partial chronological evidence."
        )
    if status == "extension_only":
        return "LastFM can support a non-chronological extension unless the paper protocol is clarified."
    return "Raw files are incomplete or protocol remains unclear."


def _core_processed_artifacts_found(staging: Path) -> bool:
    found: set[str] = set()
    if not staging.exists():
        return False
    for path in staging.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name in CORE_PROCESSED_FILENAMES:
            found.add(name)
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path, "r") as archive:
                    found.update(
                        Path(member).name.lower()
                        for member in archive.namelist()
                        if Path(member).name.lower() in CORE_PROCESSED_FILENAMES
                    )
            except zipfile.BadZipFile:
                continue
    return CORE_PROCESSED_FILENAMES.issubset(found)


def _summary_row(
    *,
    dataset: str,
    component: str,
    source_path: str,
    archive_member: str = "",
    file_found: bool = False,
    member_found: bool = False,
    status: str,
    sampled_rows: int = 0,
    parsed_rows: int = 0,
    parse_errors: int = 0,
    required_fields_present: bool = False,
    supports_reconstruction_role: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "component": component,
        "source_path": source_path,
        "archive_member": archive_member,
        "file_found": bool(file_found),
        "member_found": bool(member_found),
        "status": status,
        "sampled_rows": int(sampled_rows),
        "parsed_rows": int(parsed_rows),
        "parse_errors": int(parse_errors),
        "required_fields_present": bool(required_fields_present),
        "supports_reconstruction_role": bool(supports_reconstruction_role),
        "notes": notes,
    }


def _stats_row(
    *,
    dataset: str,
    component: str,
    source_path: str,
    archive_member: str = "",
    sampled_rows: int = 0,
    parsed_rows: int = 0,
    parse_errors: int = 0,
    unique_users: int | str = "",
    unique_items: int | str = "",
    min_timestamp: int | str = "",
    max_timestamp: int | str = "",
    rating_distribution: str = "",
    metadata_records_with_title: int | str = "",
    metadata_records_with_categories: int | str = "",
    metadata_records_with_description: int | str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "component": component,
        "source_path": source_path,
        "archive_member": archive_member,
        "sampled_rows": int(sampled_rows),
        "parsed_rows": int(parsed_rows),
        "parse_errors": int(parse_errors),
        "unique_users_sample": unique_users,
        "unique_items_sample": unique_items,
        "min_timestamp_sample": min_timestamp,
        "max_timestamp_sample": max_timestamp,
        "rating_distribution_sample": rating_distribution,
        "metadata_records_with_title": metadata_records_with_title,
        "metadata_records_with_categories": metadata_records_with_categories,
        "metadata_records_with_description": metadata_records_with_description,
        "notes": notes,
    }


def _required_present(schema: dict[str, int], fields: list[str], parsed_rows: int) -> bool:
    if parsed_rows <= 0:
        return False
    return all(schema.get(field, 0) == parsed_rows for field in fields)


def _find_member(names: list[str], expected_member: str) -> str | None:
    normalized_expected = expected_member.replace("\\", "/").lower()
    for name in names:
        if name.replace("\\", "/").lower() == normalized_expected:
            return name
    suffix = "/" + Path(expected_member).name.lower()
    for name in names:
        normalized = name.replace("\\", "/").lower()
        if normalized.endswith(suffix) or normalized == Path(expected_member).name.lower():
            return name
    return None


def _lastfm_timestamp_values(records: list[dict[str, Any]]) -> list[int]:
    values: list[int] = []
    for record in records:
        year = _to_int(record.get("year"))
        month = _to_int(record.get("month"))
        day = _to_int(record.get("day"))
        if year is not None and month is not None and day is not None:
            values.append(year * 10000 + month * 100 + day)
    return values


def _numeric_values(values: Iterable[Any]) -> list[int]:
    numeric: list[int] = []
    for value in values:
        parsed = _to_int(value)
        if parsed is not None:
            numeric.append(parsed)
    return numeric


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _rating_distribution(values: Iterable[Any]) -> str:
    counter: Counter[str] = Counter()
    for value in values:
        if value is None or value == "":
            continue
        counter[str(value)] += 1
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _unique_count(values: Iterable[Any]) -> int:
    return len({str(value) for value in values if _has_value(value)})


def _present_count(values: Iterable[Any]) -> int:
    return sum(1 for value in values if _has_value(value))


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _join_notes(*notes: str) -> str:
    return " ".join(note for note in notes if note)


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frames = validate_raw_datasets(
        args.staging_dir,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
    )
    readiness = frames["readiness"]
    counts = readiness["expected_reconstruction_status"].value_counts().to_dict()
    print(
        "Plan C raw dataset validation complete: "
        f"datasets={len(readiness)}, statuses={counts}"
    )
    print(f"Summary: {args.output_dir / 'plan_c_raw_dataset_validation_summary.csv'}")
    print(f"Readiness: {args.output_dir / 'plan_c_reconstruction_readiness_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

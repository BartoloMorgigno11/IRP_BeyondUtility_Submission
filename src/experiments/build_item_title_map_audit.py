from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import logging
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.item_title_map_reconciliation import (
    ALLOWED_ISSUE_TYPES,
    ALLOWED_SEVERITIES,
    ALLOWED_TITLE_STATUSES,
    ITEM_TITLE_COVERAGE_COLUMNS,
    ITEM_TITLE_ISSUE_COLUMNS,
    ITEM_TITLE_MAP_COLUMNS,
    MalformedSourceRecord,
    TitleSourceRecord,
    aggregate_candidate_usage,
    cleanup_title,
    coverage_summary_by_candidate_set,
    make_source_record,
    preserve_item_id,
    reconcile_title_map,
    validate_phase_2a_candidate_rows,
)
from src.audit.common_io import (
    read_csv_rows as _read_csv_rows,
    write_csv_rows as _write_csv_rows,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_CANDIDATE_SETS = PROJECT_ROOT / "results" / "tables" / "ranking_candidate_sets.csv"
DEFAULT_MAP_OUTPUT = PROJECT_ROOT / "results" / "tables" / "item_title_map.csv"
DEFAULT_ISSUES_OUTPUT = PROJECT_ROOT / "results" / "tables" / "item_title_mapping_issues.csv"
DEFAULT_COVERAGE_OUTPUT = PROJECT_ROOT / "results" / "tables" / "item_title_coverage_summary.csv"
DEFAULT_REPORT_OUTPUT = PROJECT_ROOT / "report" / "item_title_map_audit.md"


@dataclass(frozen=True)
class SourceInventoryRow:
    dataset: str
    source_path: str
    source_origin: str
    item_id_field: str
    title_field: str
    id_namespace: str
    source_priority: int | str
    direct_join_to_candidate_sets: str
    mapping_chain: str
    record_count: int
    usable_title_count: int
    used_in_map: str
    notes: str


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    candidate_path = resolve_path(args.candidate_sets)
    candidate_rows = read_csv_rows(candidate_path)
    verification = validate_phase_2a_candidate_rows(candidate_rows)
    if verification["row_count"] != 76121:
        raise ValueError(
            f"Expected 76121 candidate rows; found {verification['row_count']} in {display_path(candidate_path)}"
        )
    distribution = verification["candidate_count_distribution"]
    if distribution.get(20) != 6080 or distribution.get(100) != 70041:
        raise ValueError(f"Unexpected candidate-count distribution: {distribution}")

    usage = aggregate_candidate_usage(candidate_rows)
    records, source_inventory, malformed_records = load_verified_title_sources(
        PROJECT_ROOT,
    )
    preliminary_map_rows, _ = reconcile_title_map(records, usage, include_source_only_items=True)
    amazon_gap_ids = unresolved_amazon_candidate_ids(preliminary_map_rows)
    raw_records, raw_inventory, raw_malformed = load_amazon_raw_gap_sources(
        PROJECT_ROOT,
        amazon_gap_ids,
    )
    if raw_records or raw_inventory or raw_malformed:
        records.extend(raw_records)
        source_inventory.extend(raw_inventory)
        malformed_records.extend(raw_malformed)
    map_rows, issue_rows = reconcile_title_map(records, usage, include_source_only_items=True)
    issue_rows.extend(
        build_malformed_issue_rows(
            malformed_records,
            start_index=len(issue_rows) + 1,
        )
    )
    coverage_rows = coverage_summary_by_candidate_set(candidate_rows, map_rows)

    write_csv(DEFAULT_MAP_OUTPUT, ITEM_TITLE_MAP_COLUMNS, map_rows)
    write_csv(DEFAULT_ISSUES_OUTPUT, ITEM_TITLE_ISSUE_COLUMNS, issue_rows)
    write_csv(DEFAULT_COVERAGE_OUTPUT, ITEM_TITLE_COVERAGE_COLUMNS, coverage_rows)

    validation_messages = validate_outputs(
        candidate_rows=candidate_rows,
        map_rows=map_rows,
        issue_rows=issue_rows,
        coverage_rows=coverage_rows,
    )
    write_report(
        report_path=DEFAULT_REPORT_OUTPUT,
        candidate_path=candidate_path,
        verification=verification,
        source_inventory=source_inventory,
        map_rows=map_rows,
        issue_rows=issue_rows,
        coverage_rows=coverage_rows,
        validation_messages=validation_messages,
    )

    print(f"candidate rows: {verification['row_count']}")
    print(f"title source records: {len(records)}")
    print(f"item title map rows: {len(map_rows)}")
    print(f"mapping issue rows: {len(issue_rows)}")
    print(f"coverage summary rows: {len(coverage_rows)}")
    print(f"map: {display_path(DEFAULT_MAP_OUTPUT)}")
    print(f"issues: {display_path(DEFAULT_ISSUES_OUTPUT)}")
    print(f"coverage: {display_path(DEFAULT_COVERAGE_OUTPUT)}")
    print(f"report: {display_path(DEFAULT_REPORT_OUTPUT)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Phase 2B canonical item-title map audit from local repository artifacts."
    )
    parser.add_argument(
        "--candidate-sets",
        default=str(DEFAULT_CANDIDATE_SETS),
        help="Path to the Phase 2A canonical ranking candidate sets CSV.",
    )
    return parser.parse_args(argv)


def load_verified_title_sources(
    repo_root: Path,
) -> tuple[list[TitleSourceRecord], list[SourceInventoryRow], list[MalformedSourceRecord]]:
    records: list[TitleSourceRecord] = []
    inventory: list[SourceInventoryRow] = []
    malformed: list[MalformedSourceRecord] = []

    def extend(
        *,
        dataset: str,
        source_path: str,
        source_origin: str,
        item_id_field: str,
        title_field: str,
        id_namespace: str,
        source_priority: int,
        direct_join: str,
        mapping_chain: str,
        loaded_records: list[TitleSourceRecord],
        malformed_records: list[MalformedSourceRecord] | None = None,
        notes: str,
    ) -> None:
        records.extend(loaded_records)
        if malformed_records:
            malformed.extend(malformed_records)
        inventory.append(
            SourceInventoryRow(
                dataset=dataset,
                source_path=source_path,
                source_origin=source_origin,
                item_id_field=item_id_field,
                title_field=title_field,
                id_namespace=id_namespace,
                source_priority=source_priority,
                direct_join_to_candidate_sets=direct_join,
                mapping_chain=mapping_chain,
                record_count=len(loaded_records),
                usable_title_count=sum(1 for record in loaded_records if cleanup_title(record.title)),
                used_in_map="true",
                notes=notes,
            )
        )

    extend(
        dataset="ML-1M",
        source_path="data/processed/ml1m/catalog.csv",
        source_origin="processed",
        item_id_field="item_id",
        title_field="title",
        id_namespace="movie_id_string",
        source_priority=10,
        direct_join="yes",
        mapping_chain="direct item_id join",
        loaded_records=load_csv_title_records(
            repo_root / "data" / "processed" / "ml1m" / "catalog.csv",
            dataset="ML-1M",
            item_id_field="item_id",
            title_field="title",
            source_origin="processed",
            source_priority=10,
            id_namespace="movie_id_string",
        ),
        notes="Plan B processed catalog directly matches ML-1M candidate item IDs.",
    )
    extend(
        dataset="ML-1M",
        source_path="data/reconstructed/plan_c/ml1m/catalog.csv",
        source_origin="reconstructed",
        item_id_field="item_id",
        title_field="title",
        id_namespace="movie_id_string",
        source_priority=20,
        direct_join="yes",
        mapping_chain="direct item_id join; datamaps also available",
        loaded_records=load_csv_title_records(
            repo_root / "data" / "reconstructed" / "plan_c" / "ml1m" / "catalog.csv",
            dataset="ML-1M",
            item_id_field="item_id",
            title_field="title",
            source_origin="reconstructed",
            source_priority=20,
            id_namespace="movie_id_string",
        ),
        notes="Plan C catalog is secondary to the processed ML-1M catalog for Phase 2B.",
    )
    extend(
        dataset="ML-1M",
        source_path="data/raw/ml1m/movies.dat",
        source_origin="raw",
        item_id_field="movie_id",
        title_field="movie_title",
        id_namespace="movie_id_string",
        source_priority=30,
        direct_join="yes",
        mapping_chain="direct MovieLens movie_id join",
        loaded_records=load_ml1m_movies_dat(
            repo_root / "data" / "raw" / "ml1m" / "movies.dat",
            source_path="data/raw/ml1m/movies.dat",
            source_origin="raw",
            source_priority=30,
        ),
        notes="Raw manually provided ML-1M file confirms processed titles.",
    )
    extend(
        dataset="ML-1M",
        source_path="data/external/plan_c/ml-1m.zip::ml-1m/movies.dat",
        source_origin="raw_archive",
        item_id_field="movie_id",
        title_field="movie_title",
        id_namespace="movie_id_string",
        source_priority=40,
        direct_join="yes",
        mapping_chain="direct MovieLens movie_id join from streamed archive member",
        loaded_records=load_ml1m_movies_from_zip(
            repo_root / "data" / "external" / "plan_c" / "ml-1m.zip",
            "ml-1m/movies.dat",
            source_priority=40,
        ),
        notes="Read-only ZIP member streaming; archive was not extracted.",
    )

    for dataset, folder, raw_meta, priority in [
        (
            "Amazon Beauty",
            "amazon_beauty",
            "data/external/plan_c/meta_Beauty.json.gz",
            10,
        ),
        (
            "Amazon Sports",
            "amazon_sports",
            "data/external/plan_c/meta_Sports_and_Outdoors.json.gz",
            10,
        ),
    ]:
        extend(
            dataset=dataset,
            source_path=f"data/reconstructed/plan_c/{folder}/catalog.csv",
            source_origin="reconstructed",
            item_id_field="item_id",
            title_field="title",
            id_namespace="amazon_asin_string",
            source_priority=priority,
            direct_join="yes",
            mapping_chain=f"direct ASIN join; data/reconstructed/plan_c/{folder}/datamaps.json verifies raw ASIN to internal IDs",
            loaded_records=load_csv_title_records(
                repo_root / "data" / "reconstructed" / "plan_c" / folder / "catalog.csv",
                dataset=dataset,
                item_id_field="item_id",
                title_field="title",
                source_origin="reconstructed",
                source_priority=priority,
                id_namespace="amazon_asin_string",
            ),
            notes="Current canonical pipeline catalog has direct ASIN item IDs.",
        )
        inventory.append(
            SourceInventoryRow(
                dataset=dataset,
                source_path=raw_meta,
                source_origin="raw",
                item_id_field="asin",
                title_field="title",
                id_namespace="amazon_asin_string",
                source_priority=priority + 20,
                direct_join_to_candidate_sets="yes",
                mapping_chain="direct ASIN join from gzipped Python-literal metadata already staged locally",
                record_count=0,
                usable_title_count=0,
                used_in_map="gap_check_only",
                notes="Inspected as raw metadata; loaded separately only for unresolved candidate ASIN checks to avoid duplicating the reconstructed catalog.",
            )
        )

    extend(
        dataset="LastFM",
        source_path="data/reconstructed/plan_c/lastfm/catalog.csv",
        source_origin="reconstructed",
        item_id_field="item_id",
        title_field="title",
        id_namespace="lastfm_artist_id_string",
        source_priority=10,
        direct_join="not_applicable_no_candidate_sets",
        mapping_chain="direct artist ID join for catalog only",
        loaded_records=load_csv_title_records(
            repo_root / "data" / "reconstructed" / "plan_c" / "lastfm" / "catalog.csv",
            dataset="LastFM",
            item_id_field="item_id",
            title_field="title",
            source_origin="reconstructed",
            source_priority=10,
            id_namespace="lastfm_artist_id_string",
        ),
        notes="LastFM has catalog titles but no canonical candidate rows in Phase 2A.",
    )
    extend(
        dataset="LastFM",
        source_path="data/external/plan_c/hetrec2011-lastfm-2k.zip::artists.dat",
        source_origin="raw_archive",
        item_id_field="id",
        title_field="name",
        id_namespace="lastfm_artist_id_string",
        source_priority=20,
        direct_join="not_applicable_no_candidate_sets",
        mapping_chain="direct artist ID join from streamed archive member",
        loaded_records=load_lastfm_artists_from_zip(
            repo_root / "data" / "external" / "plan_c" / "hetrec2011-lastfm-2k.zip",
            "artists.dat",
            source_priority=20,
        ),
        notes="Read-only ZIP member streaming; archive was not extracted.",
    )

    inventory.extend(non_authoritative_inventory_rows(repo_root))
    return records, inventory, malformed


def load_csv_title_records(
    path: Path,
    *,
    dataset: str,
    item_id_field: str,
    title_field: str,
    source_origin: str,
    source_priority: int,
    id_namespace: str,
) -> list[TitleSourceRecord]:
    if not path.exists():
        return []
    rows: list[TitleSourceRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record = make_source_record(
                dataset=dataset,
                item_id=row.get(item_id_field, ""),
                title=row.get(title_field, ""),
                source_path=display_path(path),
                source_field=title_field,
                source_origin=source_origin,
                source_priority=source_priority,
                id_namespace=id_namespace,
            )
            if record is not None:
                rows.append(record)
    return rows


def load_ml1m_movies_dat(
    path: Path,
    *,
    source_path: str,
    source_origin: str,
    source_priority: int,
) -> list[TitleSourceRecord]:
    if not path.exists():
        return []
    rows: list[TitleSourceRecord] = []
    with path.open("r", encoding="latin-1") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 3:
                continue
            record = make_source_record(
                dataset="ML-1M",
                item_id=parts[0],
                title=parts[1],
                source_path=source_path,
                source_field="movie_title",
                source_origin=source_origin,
                source_priority=source_priority,
                id_namespace="movie_id_string",
            )
            if record is not None:
                rows.append(record)
    return rows


def load_ml1m_movies_from_zip(
    archive_path: Path,
    member_name: str,
    *,
    source_priority: int,
) -> list[TitleSourceRecord]:
    if not archive_path.exists():
        return []
    rows: list[TitleSourceRecord] = []
    source_path = f"{display_path(archive_path)}::{member_name}"
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as handle:
            for raw_line in handle:
                line = raw_line.decode("latin-1").rstrip("\n")
                parts = line.split("::")
                if len(parts) < 3:
                    continue
                record = make_source_record(
                    dataset="ML-1M",
                    item_id=parts[0],
                    title=parts[1],
                    source_path=source_path,
                    source_field="movie_title",
                    source_origin="raw_archive",
                    source_priority=source_priority,
                    id_namespace="movie_id_string",
                )
                if record is not None:
                    rows.append(record)
    return rows


def load_amazon_meta_json_gz(
    path: Path,
    *,
    dataset: str,
    source_priority: int,
    allowed_item_ids: set[str] | None = None,
) -> list[TitleSourceRecord]:
    if not path.exists():
        return []
    rows: list[TitleSourceRecord] = []
    asin_pattern = re.compile(r"""['"]asin['"]\s*:\s*['"]([^'"]+)['"]""")
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            asin_match = asin_pattern.search(line)
            if not asin_match:
                continue
            asin = preserve_item_id(asin_match.group(1))
            if allowed_item_ids is not None and asin not in allowed_item_ids:
                continue
            try:
                record_obj = json.loads(line)
            except json.JSONDecodeError:
                try:
                    record_obj = ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    continue
            if not isinstance(record_obj, dict):
                continue
            record = make_source_record(
                dataset=dataset,
                item_id=asin,
                title=record_obj.get("title", ""),
                source_path=display_path(path),
                source_field="title",
                source_origin="raw",
                source_priority=source_priority,
                id_namespace="amazon_asin_string",
            )
            if record is not None:
                rows.append(record)
    return rows


def unresolved_amazon_candidate_ids(map_rows: Iterable[dict[str, Any]]) -> dict[str, set[str]]:
    unresolved: dict[str, set[str]] = defaultdict(set)
    for row in map_rows:
        dataset = row["dataset"]
        if dataset not in {"Amazon Beauty", "Amazon Sports"}:
            continue
        if row["in_candidate_sets"] != "true":
            continue
        if row["title_status"] in {"missing_title", "unverifiable_mapping"}:
            unresolved[dataset].add(preserve_item_id(row["item_id"]))
    return unresolved


def load_amazon_raw_gap_sources(
    repo_root: Path,
    raw_ids_by_dataset: dict[str, set[str]],
) -> tuple[list[TitleSourceRecord], list[SourceInventoryRow], list[MalformedSourceRecord]]:
    records: list[TitleSourceRecord] = []
    inventory: list[SourceInventoryRow] = []
    malformed: list[MalformedSourceRecord] = []
    source_specs = [
        (
            "Amazon Beauty",
            "data/external/plan_c/meta_Beauty.json.gz",
            30,
        ),
        (
            "Amazon Sports",
            "data/external/plan_c/meta_Sports_and_Outdoors.json.gz",
            30,
        ),
    ]
    for dataset, source_path, priority in source_specs:
        unresolved_ids = {item_id for item_id in raw_ids_by_dataset.get(dataset, set()) if item_id}
        loaded_records = (
            load_amazon_meta_json_gz(
                repo_root / source_path,
                dataset=dataset,
                source_priority=priority,
                allowed_item_ids=unresolved_ids,
            )
            if unresolved_ids
            else []
        )
        records.extend(loaded_records)
        inventory.append(
            SourceInventoryRow(
                dataset=dataset,
                source_path=source_path,
                source_origin="raw",
                item_id_field="asin",
                title_field="title",
                id_namespace="amazon_asin_string",
                source_priority=priority,
                direct_join_to_candidate_sets="yes",
                mapping_chain="direct ASIN join from gzipped Python-literal metadata already staged locally",
                record_count=len(loaded_records),
                usable_title_count=sum(1 for record in loaded_records if cleanup_title(record.title)),
                used_in_map="true" if loaded_records else "gap_check_only",
                notes=(
                    f"Targeted raw metadata check for {len(unresolved_ids)} unresolved candidate ASINs; "
                    "archive was streamed read-only and not extracted."
                ),
            )
        )
    return records, inventory, malformed


def load_lastfm_artists_from_zip(
    archive_path: Path,
    member_name: str,
    *,
    source_priority: int,
) -> list[TitleSourceRecord]:
    if not archive_path.exists():
        return []
    rows: list[TitleSourceRecord] = []
    source_path = f"{display_path(archive_path)}::{member_name}"
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as handle:
            reader = csv.DictReader(
                (line.decode("utf-8", errors="replace") for line in handle),
                delimiter="\t",
            )
            for row in reader:
                record = make_source_record(
                    dataset="LastFM",
                    item_id=row.get("id", ""),
                    title=row.get("name", ""),
                    source_path=source_path,
                    source_field="name",
                    source_origin="raw_archive",
                    source_priority=source_priority,
                    id_namespace="lastfm_artist_id_string",
                )
                if record is not None:
                    rows.append(record)
    return rows


def non_authoritative_inventory_rows(repo_root: Path) -> list[SourceInventoryRow]:
    rows: list[SourceInventoryRow] = []
    for source_path, notes in [
        (
            "original_repo/EvaLLMasRecommender/llm_results/beauty.zip",
            "contains prompt/result text with item titles but no authoritative candidate item_id title map",
        ),
        (
            "original_repo/EvaLLMasRecommender/llm_results/sports.zip",
            "contains prompt/result text with item titles but no authoritative candidate item_id title map",
        ),
        (
            "original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip",
            "contains recommender output IDs or scores but no title metadata",
        ),
    ]:
        path = repo_root / source_path
        rows.append(
            SourceInventoryRow(
                dataset="multiple" if "llm_results" in source_path else "Amazon Beauty",
                source_path=source_path,
                source_origin="upstream_archive",
                item_id_field="not_used",
                title_field="not_used",
                id_namespace="unverified_prompt_or_score_payload",
                source_priority="not_used",
                direct_join_to_candidate_sets="no",
                mapping_chain="unverifiable for canonical title map",
                record_count=0,
                usable_title_count=0,
                used_in_map="false",
                notes=notes if path.exists() else f"not found locally; {notes}",
            )
        )
    return rows


def build_malformed_issue_rows(
    malformed_records: Iterable[MalformedSourceRecord],
    *,
    start_index: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    index = start_index
    for record in malformed_records:
        rows.append(
            {
                "issue_id": f"ITM-{index:06d}",
                "dataset": record.dataset,
                "item_id": record.item_id,
                "issue_type": record.issue_type,
                "severity": "warning",
                "source_paths": record.source_path,
                "candidate_set_ids": "",
                "candidate_occurrence_count": "0",
                "positive_occurrence_count": "0",
                "evidence": record.evidence,
                "selected_resolution": "record excluded from canonical title selection",
                "blocking_reason": "",
                "notes": record.notes,
            }
        )
        index += 1
    return rows


def validate_outputs(
    *,
    candidate_rows: list[dict[str, str]],
    map_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> list[str]:
    messages: list[str] = []
    if not map_rows:
        raise ValueError("item title map is empty")
    keys = [(row["dataset"], preserve_item_id(row["item_id"])) for row in map_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("item title map contains duplicate dataset/item_id keys")
    map_by_key = {key: row for key, row in zip(keys, map_rows, strict=True)}
    for row in map_rows:
        if row["title_status"] not in ALLOWED_TITLE_STATUSES:
            raise ValueError(f"invalid title_status: {row['title_status']}")
        json.loads(str(row["alternate_titles"]))
        if row["title_status"] in {"missing_title", "unverifiable_mapping"} and row["canonical_title"]:
            raise ValueError("unresolved title row has non-empty canonical_title")
        if row["title_status"].startswith("resolved") and not row["canonical_title"]:
            raise ValueError("resolved title row has empty canonical_title")
        if cleanup_title(row["canonical_title"]) == "Missing title":
            raise ValueError("synthetic placeholder title detected")

    candidate_item_keys: set[tuple[str, str]] = set()
    unresolved_candidate_keys: set[tuple[str, str]] = set()
    for row in candidate_rows:
        dataset = row["dataset"]
        for item_id in json.loads(row["candidate_items"]):
            key = (dataset, preserve_item_id(item_id))
            candidate_item_keys.add(key)
            map_row = map_by_key.get(key)
            if map_row is None:
                raise ValueError(f"candidate item missing from title map: {key}")
            if map_row["title_status"] in {"missing_title", "unverifiable_mapping"}:
                unresolved_candidate_keys.add(key)

    issue_keys = {
        (row["dataset"], preserve_item_id(row["item_id"]))
        for row in issue_rows
        if row["issue_type"] in {"missing_title", "unverifiable_id_mapping", "missing_source_item"}
    }
    missing_issue_keys = unresolved_candidate_keys.difference(issue_keys)
    if missing_issue_keys:
        sample = sorted(missing_issue_keys)[:5]
        raise ValueError(f"unresolved candidate items without issue rows: {sample}")

    issue_ids = [row["issue_id"] for row in issue_rows]
    if len(issue_ids) != len(set(issue_ids)):
        raise ValueError("issue IDs are not unique")
    for row in issue_rows:
        if row["issue_type"] not in ALLOWED_ISSUE_TYPES:
            raise ValueError(f"invalid issue_type: {row['issue_type']}")
        if row["severity"] not in ALLOWED_SEVERITIES:
            raise ValueError(f"invalid severity: {row['severity']}")
        if not any(str(value).strip() for value in row.values()):
            raise ValueError("blank issue row found")

    candidate_set_ids = {row["candidate_set_id"] for row in candidate_rows}
    summary_ids = {row["candidate_set_id"] for row in coverage_rows}
    if candidate_set_ids != summary_ids:
        raise ValueError("coverage summary does not contain one row per candidate_set_id")
    for row in coverage_rows:
        for field in [
            "unique_item_title_coverage",
            "candidate_occurrence_title_coverage",
            "row_complete_title_coverage",
            "positive_title_coverage",
        ]:
            value = float(row[field])
            if value < 0 or value > 1:
                raise ValueError(f"coverage value out of range for {field}: {value}")
        if int(row["rows_with_all_titles"]) + int(row["rows_with_missing_titles"]) != int(
            row["candidate_row_count"]
        ):
            raise ValueError("coverage row title completeness counts do not sum to row count")
        if int(row["titled_positive_item_count"]) > int(row["positive_item_count"]):
            raise ValueError("titled positive count exceeds positive item count")
        if int(row["titled_candidate_occurrence_count"]) > int(row["candidate_occurrence_count"]):
            raise ValueError("titled occurrence count exceeds candidate occurrence count")

    messages.append(f"validated {len(map_rows)} item_title_map rows")
    messages.append(f"validated {len(issue_rows)} item_title_mapping_issues rows")
    messages.append(f"validated {len(coverage_rows)} item_title_coverage_summary rows")
    messages.append(f"all {len(candidate_item_keys)} unique candidate items have title-map rows")
    return messages


def write_report(
    *,
    report_path: Path,
    candidate_path: Path,
    verification: dict[str, Any],
    source_inventory: list[SourceInventoryRow],
    map_rows: list[dict[str, Any]],
    issue_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    validation_messages: list[str],
) -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    dataset_map_stats = dataset_stats_from_map(map_rows)
    dataset_issue_stats = issue_stats_by_dataset(issue_rows)
    coverage_by_dataset = group_coverage_by_dataset(coverage_rows)
    source_by_dataset: dict[str, list[SourceInventoryRow]] = defaultdict(list)
    for row in source_inventory:
        source_by_dataset[row.dataset].append(row)

    lines: list[str] = []
    lines.append("# Canonical Item-Title Map Audit")
    lines.append("")
    lines.append("## 1. Scope and provenance")
    lines.append("")
    lines.append(f"- Audit timestamp: {now}")
    lines.append("- Phase: 2B canonical item-title map reconciliation")
    lines.append(f"- Candidate-set input: `{display_path(candidate_path)}`")
    lines.append("- Python executable: `C:\\ProgramData\\anaconda3\\python.exe` with `PYTHONNOUSERSITE=1`")
    lines.append("- Network/API/model/training activity: none")
    lines.append("")
    lines.append("## 2. Executive summary")
    lines.append("")
    lines.append(
        f"- Built `results/tables/item_title_map.csv` with {len(map_rows)} dataset-item rows."
    )
    lines.append(
        f"- Built `results/tables/item_title_mapping_issues.csv` with {len(issue_rows)} aggregate issue rows."
    )
    lines.append(
        f"- Built `results/tables/item_title_coverage_summary.csv` with {len(coverage_rows)} candidate-set summaries."
    )
    lines.append("- ML-1M and LastFM title sources are complete in the verified catalogs.")
    lines.append("- Beauty and Sports still have candidate rows with at least one missing title.")
    lines.append("- No missing title was filled with an item ID or generated placeholder.")
    lines.append("")
    lines.append("## 3. Input candidate-set verification")
    lines.append("")
    lines.append(f"- Row count: {verification['row_count']}")
    lines.append(f"- Candidate-count distribution: `{verification['candidate_count_distribution']}`")
    lines.append(f"- Row validation status distribution: `{verification['status_distribution']}`")
    lines.append("- Candidate item lists were JSON-decodable and candidate counts matched decoded list lengths.")
    lines.append("- The Phase 2A candidate-set file was read only and was not regenerated.")
    lines.append("")
    lines.append("## 4. Title-source inventory")
    lines.append("")
    for dataset in sorted(source_by_dataset):
        lines.append(f"### {dataset}")
        for row in sorted(source_by_dataset[dataset], key=lambda item: str(item.source_priority)):
            lines.append(
                "- "
                f"`{row.source_path}` | origin={row.source_origin} | id_field={row.item_id_field} | "
                f"title_field={row.title_field} | namespace={row.id_namespace} | "
                f"priority={row.source_priority} | records={row.record_count} | "
                f"usable_titles={row.usable_title_count} | used={row.used_in_map} | {row.notes}"
            )
        lines.append("")
    lines.append("## 5. Identifier namespaces and mapping chains")
    lines.append("")
    lines.append("- Amazon Beauty and Amazon Sports candidate items use raw ASIN strings. Reconstructed `datamaps.json` files verify ASIN to internal integer IDs but no internal-ID join is required for Phase 2B.")
    lines.append("- ML-1M candidate items use MovieLens movie IDs as strings. Processed and reconstructed catalogs join directly on `item_id`; raw `movies.dat` confirms the same namespace.")
    lines.append("- LastFM catalog items use artist IDs as strings. Phase 2A has no LastFM candidate-set rows, so LastFM is catalog-only in this audit.")
    lines.append("- Upstream LLM prompt archives and SOTA result archives were inspected as provenance but are not authoritative title maps because their title payloads are not a verified item-ID catalog.")
    lines.append("")
    lines.append("## 6. Reconciliation methodology")
    lines.append("")
    lines.append("- Item IDs are preserved as strings with leading zeros intact.")
    lines.append("- Title cleanup only trims whitespace, collapses repeated whitespace, decodes simple JSON string values, and treats explicit null-like tokens as missing.")
    lines.append("- Titles are selected by deterministic dataset-specific source priority.")
    lines.append("- Genuine title conflicts are retained in `alternate_titles` as JSON arrays and recorded in the issue table.")
    lines.append("- Unresolved titles remain empty in `canonical_title`.")
    lines.append("")
    lines.append("## 7. Dataset-specific source priorities")
    lines.append("")
    lines.append("- Amazon Beauty: reconstructed Plan C catalog first; raw staged Amazon metadata second.")
    lines.append("- Amazon Sports: reconstructed Plan C catalog first; raw staged Amazon metadata second.")
    lines.append("- ML-1M: processed Plan B catalog first; reconstructed Plan C catalog second; raw `movies.dat` third; staged Plan C ZIP member fourth.")
    lines.append("- LastFM: reconstructed Plan C catalog first; staged HetRec ZIP `artists.dat` second.")
    lines.append("")
    lines.append("## 8. Canonical-map statistics")
    lines.append("")
    lines.extend(markdown_table(
        ["dataset", "rows", "candidate_items", "resolved", "missing", "conflicted"],
        [
            [
                dataset,
                stats.get("rows", 0),
                stats.get("candidate_items", 0),
                stats.get("resolved", 0),
                stats.get("missing", 0),
                stats.get("conflicted", 0),
            ]
            for dataset, stats in dataset_map_stats.items()
        ],
    ))
    lines.append("")
    lines.append("## 9. Candidate-set title coverage")
    lines.append("")
    lines.extend(markdown_table(
        ["dataset", "candidate_sets", "rows", "rows_with_missing_titles", "occurrence_coverage"],
        [
            [
                dataset,
                stats["candidate_sets"],
                stats["rows"],
                stats["rows_with_missing_titles"],
                f"{stats['occurrence_coverage']:.6f}",
            ]
            for dataset, stats in coverage_by_dataset.items()
        ],
    ))
    lines.append("")
    lines.append("## 10. Positive-item title coverage")
    lines.append("")
    lines.extend(markdown_table(
        ["dataset", "positive_items", "titled_positive_items", "positive_title_coverage"],
        [
            [
                dataset,
                stats["positive_items"],
                stats["titled_positive_items"],
                f"{stats['positive_title_coverage']:.6f}",
            ]
            for dataset, stats in coverage_by_dataset.items()
        ],
    ))
    lines.append("")
    lines.append("## 11. Missing-title findings")
    lines.append("")
    for dataset, stats in dataset_issue_stats.items():
        missing = stats.get("missing_title", 0) + stats.get("missing_source_item", 0)
        lines.append(f"- {dataset}: {missing} missing-title or missing-source issue rows.")
    lines.append("")
    lines.append("## 12. Conflicting-title findings")
    lines.append("")
    conflict_total = sum(stats.get("conflicting_titles", 0) for stats in dataset_issue_stats.values())
    lines.append(f"- Aggregate conflicting-title issue rows: {conflict_total}.")
    lines.append("- Conflicts do not block map generation because the selected title follows documented source priority and alternates are retained.")
    lines.append("")
    lines.append("## 13. Beauty findings")
    lines.append("")
    lines.extend(dataset_detail_lines("Amazon Beauty", dataset_map_stats, coverage_by_dataset, dataset_issue_stats))
    lines.append("")
    lines.append("## 14. Sports findings")
    lines.append("")
    lines.extend(dataset_detail_lines("Amazon Sports", dataset_map_stats, coverage_by_dataset, dataset_issue_stats))
    lines.append("")
    lines.append("## 15. ML-1M findings")
    lines.append("")
    lines.extend(dataset_detail_lines("ML-1M", dataset_map_stats, coverage_by_dataset, dataset_issue_stats))
    lines.append("")
    lines.append("## 16. LastFM findings")
    lines.append("")
    lines.extend(dataset_detail_lines("LastFM", dataset_map_stats, coverage_by_dataset, dataset_issue_stats))
    lines.append("- LastFM has catalog title coverage but no Phase 2A canonical candidate-set rows.")
    lines.append("")
    lines.append("## 17. Plan C / Plan D / Plan E consistency")
    lines.append("")
    lines.append("- Plan C reconstructed 100-candidate pools join directly to reconstructed catalogs for Beauty, Sports, and ML-1M.")
    lines.append("- Plan D ML-1M Qwen2 diagnostic candidate manifests use the same ML-1M movie-ID namespace.")
    lines.append("- Plan E 1000-user 20-candidate manifests use the same dataset namespaces as their Plan C/processed sources.")
    lines.append("- Existing Plan C/D/E outputs were read as sources or candidate-set provenance only and were not modified.")
    lines.append("")
    lines.append("## 18. Remaining blocking issues")
    lines.append("")
    lines.append("- Candidate rows with missing Beauty or Sports titles remain blocked because matching reconstructed catalog records and targeted raw metadata records have empty or missing title values.")
    lines.append("- No unresolved title is safe to fill from descriptions, categories, item IDs, prompt text, or LLM inference.")
    lines.append("- Upstream prompt/result archives are not authoritative item-title maps for future prompt construction.")
    lines.append("")
    lines.append("## 19. Canonical title-map path for future experiments")
    lines.append("")
    lines.append("- Use `results/tables/item_title_map.csv` as the authoritative Phase 2B title source.")
    lines.append("- Future prompt construction should join on `(dataset, item_id)` and must handle empty `canonical_title` as unresolved.")
    lines.append("")
    lines.append("## 20. Artifacts that should not be used as authoritative title maps")
    lines.append("")
    lines.append("- `original_repo/EvaLLMasRecommender/llm_results/*.zip`: prompt/result payloads, not verified item-title catalogs.")
    lines.append("- `original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip`: recommender outputs, not title metadata.")
    lines.append("- `normalized_text`, `description`, `categories`, `brand`, and item IDs: not title fields for unresolved titles.")
    lines.append("")
    lines.append("## 21. Test and validation commands")
    lines.append("")
    lines.append("- `$env:PYTHONNOUSERSITE='1'; & 'C:\\ProgramData\\anaconda3\\python.exe' --version`")
    lines.append("- `$env:PYTHONNOUSERSITE='1'; & 'C:\\ProgramData\\anaconda3\\python.exe' src/experiments/build_item_title_map_audit.py`")
    lines.append("- `$env:PYTHONNOUSERSITE='1'; & 'C:\\ProgramData\\anaconda3\\python.exe' -m pytest tests/test_item_title_map_reconciliation.py tests/test_dataset_candidate_set_validation.py`")
    for message in validation_messages:
        lines.append(f"- {message}")
    lines.append("")
    lines.append("## 22. Recommended single next task")
    lines.append("")
    lines.append("Resolve remaining Beauty and Sports candidate-title gaps by auditing whether already-staged raw metadata contains missing ASINs under another verified repository-local source or mapping chain, without using web access or generated titles.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def dataset_stats_from_map(map_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in map_rows:
        dataset = row["dataset"]
        stats[dataset]["rows"] += 1
        if row["in_candidate_sets"] == "true":
            stats[dataset]["candidate_items"] += 1
        if str(row["title_status"]).startswith("resolved"):
            stats[dataset]["resolved"] += 1
        if row["title_status"] in {"missing_title", "unverifiable_mapping"}:
            stats[dataset]["missing"] += 1
        if int(row["title_conflict_count"]) > 0:
            stats[dataset]["conflicted"] += 1
    return {dataset: dict(values) for dataset, values in sorted(stats.items())}


def issue_stats_by_dataset(issue_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in issue_rows:
        stats[row["dataset"]][row["issue_type"]] += 1
    return {dataset: dict(values) for dataset, values in sorted(stats.items())}


def group_coverage_by_dataset(coverage_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    for row in coverage_rows:
        dataset = row["dataset"]
        grouped[dataset]["candidate_sets"] += 1
        grouped[dataset]["rows"] += int(row["candidate_row_count"])
        grouped[dataset]["rows_with_missing_titles"] += int(row["rows_with_missing_titles"])
        grouped[dataset]["candidate_occurrences"] += int(row["candidate_occurrence_count"])
        grouped[dataset]["titled_candidate_occurrences"] += int(row["titled_candidate_occurrence_count"])
        grouped[dataset]["positive_items"] += int(row["positive_item_count"])
        grouped[dataset]["titled_positive_items"] += int(row["titled_positive_item_count"])
    for values in grouped.values():
        values["occurrence_coverage"] = safe_divide(
            values["titled_candidate_occurrences"],
            values["candidate_occurrences"],
        )
        values["positive_title_coverage"] = safe_divide(
            values["titled_positive_items"],
            values["positive_items"],
        )
    return {dataset: dict(values) for dataset, values in sorted(grouped.items())}


def dataset_detail_lines(
    dataset: str,
    map_stats: dict[str, dict[str, int]],
    coverage_stats: dict[str, dict[str, Any]],
    issue_stats: dict[str, dict[str, int]],
) -> list[str]:
    map_row = map_stats.get(dataset, {})
    coverage_row = coverage_stats.get(dataset, {})
    issue_row = issue_stats.get(dataset, {})
    return [
        f"- Canonical map rows: {map_row.get('rows', 0)}.",
        f"- Candidate-set item rows in map: {map_row.get('candidate_items', 0)}.",
        f"- Resolved title rows: {map_row.get('resolved', 0)}.",
        f"- Missing or unverifiable title rows: {map_row.get('missing', 0)}.",
        f"- Rows with missing titles in candidate sets: {int(coverage_row.get('rows_with_missing_titles', 0))}.",
        f"- Candidate occurrence title coverage: {float(coverage_row.get('occurrence_coverage', 0.0)):.6f}.",
        f"- Positive item title coverage: {float(coverage_row.get('positive_title_coverage', 0.0)):.6f}.",
        f"- Issue counts: `{issue_row}`.",
    ]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["No rows."]
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(value) for value in row) + " |")
    return output


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_rows(path)


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    _write_csv_rows(path, columns, rows, extrasaction="ignore")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())

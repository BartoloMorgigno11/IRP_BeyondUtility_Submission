from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


ITEM_TITLE_MAP_COLUMNS = [
    "dataset",
    "item_id",
    "canonical_title",
    "title_status",
    "selected_source_path",
    "selected_source_field",
    "selected_source_origin",
    "selected_source_priority",
    "source_record_count",
    "non_empty_source_count",
    "title_conflict_count",
    "alternate_titles",
    "in_candidate_sets",
    "candidate_set_count",
    "candidate_occurrence_count",
    "positive_occurrence_count",
    "notes",
]

ITEM_TITLE_ISSUE_COLUMNS = [
    "issue_id",
    "dataset",
    "item_id",
    "issue_type",
    "severity",
    "source_paths",
    "candidate_set_ids",
    "candidate_occurrence_count",
    "positive_occurrence_count",
    "evidence",
    "selected_resolution",
    "blocking_reason",
    "notes",
]

ITEM_TITLE_COVERAGE_COLUMNS = [
    "dataset",
    "candidate_set_id",
    "source_plan",
    "source_path",
    "candidate_row_count",
    "unique_item_count",
    "titled_unique_item_count",
    "unique_item_title_coverage",
    "candidate_occurrence_count",
    "titled_candidate_occurrence_count",
    "candidate_occurrence_title_coverage",
    "rows_with_all_titles",
    "rows_with_missing_titles",
    "row_complete_title_coverage",
    "positive_item_count",
    "titled_positive_item_count",
    "positive_title_coverage",
    "notes",
]

ALLOWED_TITLE_STATUSES = {
    "resolved_single_source",
    "resolved_agreement",
    "resolved_by_priority",
    "missing_title",
    "unverifiable_mapping",
}

ALLOWED_ISSUE_TYPES = {
    "missing_title",
    "conflicting_titles",
    "unverifiable_id_mapping",
    "missing_source_item",
    "malformed_source_record",
    "duplicate_source_record",
    "incompatible_identifier_namespace",
}

ALLOWED_SEVERITIES = {"warning", "blocking"}

NULL_LIKE_TITLE_VALUES = {
    "",
    "null",
    "none",
    "nan",
    "n/a",
    "na",
    "<na>",
    "nil",
}

PHASE_2A_CANDIDATE_COLUMNS = [
    "candidate_set_id",
    "source_path",
    "source_plan",
    "dataset",
    "user_id",
    "positive_item_id",
    "candidate_items",
    "candidate_count",
    "negative_count",
    "stored_positive_position",
    "stored_position_base",
    "computed_positive_position_zero_based",
    "computed_positive_position_one_based",
    "positive_occurrence_count",
    "candidate_items_unique",
    "positive_in_candidates",
    "user_mapping_valid",
    "positive_item_mapping_valid",
    "all_candidate_items_mapping_valid",
    "all_candidate_titles_available",
    "ground_truth_match",
    "negative_history_exclusion_valid",
    "row_validation_status",
    "validation_issues",
]


@dataclass(frozen=True)
class TitleSourceRecord:
    dataset: str
    item_id: str
    title: str
    source_path: str
    source_field: str
    source_origin: str
    source_priority: int
    id_namespace: str = "raw"
    mapping_status: str = "verified"
    notes: str = ""


@dataclass
class CandidateUsage:
    dataset: str
    item_id: str
    candidate_set_ids: set[str]
    candidate_occurrence_count: int = 0
    positive_occurrence_count: int = 0


@dataclass(frozen=True)
class MalformedSourceRecord:
    dataset: str
    item_id: str
    source_path: str
    source_field: str
    issue_type: str
    evidence: str
    notes: str = ""


def preserve_item_id(value: Any) -> str:
    """Return an item identifier as a stripped string without numeric coercion."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def cleanup_title(value: Any) -> str:
    """Return a display title after safe cleanup only.

    Cleanup intentionally does not rewrite punctuation, years, variants,
    capitalization, or spelling. Null-like values become an empty string.
    """

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    stripped = text.strip()
    if (
        len(stripped) >= 2
        and stripped[0] == stripped[-1] == '"'
        and "\\n" not in stripped[:2]
    ):
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = stripped
        if isinstance(decoded, str):
            stripped = decoded.strip()
    cleaned = re.sub(r"\s+", " ", stripped).strip()
    if cleaned.casefold() in NULL_LIKE_TITLE_VALUES:
        return ""
    return cleaned


def comparison_normalize_title(value: Any) -> str:
    """Return the comparison-only normalized form for a title."""

    return cleanup_title(value)


def is_missing_title(value: Any) -> bool:
    """Return true when a title value is empty or an explicit null-like token."""

    return cleanup_title(value) == ""


def make_source_record(
    *,
    dataset: str,
    item_id: Any,
    title: Any,
    source_path: str,
    source_field: str,
    source_origin: str,
    source_priority: int,
    id_namespace: str = "raw",
    mapping_status: str = "verified",
    notes: str = "",
) -> TitleSourceRecord | None:
    """Create a source record when the item identifier is usable."""

    item_key = preserve_item_id(item_id)
    if not item_key:
        return None
    return TitleSourceRecord(
        dataset=str(dataset),
        item_id=item_key,
        title=cleanup_title(title),
        source_path=str(source_path),
        source_field=str(source_field),
        source_origin=str(source_origin),
        source_priority=int(source_priority),
        id_namespace=str(id_namespace),
        mapping_status=str(mapping_status),
        notes=str(notes),
    )


def validate_phase_2a_candidate_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the Phase 2A candidate-set rows without mutating them."""

    if not rows:
        raise ValueError("ranking_candidate_sets.csv is empty")
    actual_columns = list(rows[0].keys())
    if actual_columns != PHASE_2A_CANDIDATE_COLUMNS:
        raise ValueError(
            "ranking_candidate_sets.csv schema does not match Phase 2A canonical schema"
        )

    decoded_errors = 0
    count_mismatches = 0
    candidate_count_distribution: Counter[int] = Counter()
    status_distribution: Counter[str] = Counter()
    for row in rows:
        items = decode_candidate_items(row["candidate_items"])
        candidate_count = int(row["candidate_count"])
        if len(items) != candidate_count:
            count_mismatches += 1
        candidate_count_distribution[len(items)] += 1
        status_distribution[str(row["row_validation_status"])] += 1

    if decoded_errors:
        raise ValueError("candidate item lists contain invalid JSON")
    if count_mismatches:
        raise ValueError("candidate_count does not match decoded candidate_items")

    return {
        "row_count": len(rows),
        "candidate_count_distribution": dict(sorted(candidate_count_distribution.items())),
        "status_distribution": dict(status_distribution),
        "schema_columns": actual_columns,
    }


def decode_candidate_items(value: Any) -> list[str]:
    """Decode a JSON candidate item list preserving item IDs as strings."""

    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"candidate_items is not valid JSON: {exc}") from exc
    if not isinstance(decoded, list):
        raise ValueError("candidate_items must decode to a JSON list")
    return [preserve_item_id(item) for item in decoded]


def aggregate_candidate_usage(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], CandidateUsage]:
    """Aggregate candidate and positive-item occurrences by dataset and item ID."""

    usage: dict[tuple[str, str], CandidateUsage] = {}
    for row in rows:
        dataset = str(row["dataset"])
        candidate_set_id = str(row["candidate_set_id"])
        positive_item_id = preserve_item_id(row["positive_item_id"])
        candidate_items = decode_candidate_items(row["candidate_items"])
        for item_id in candidate_items:
            if not item_id:
                continue
            key = (dataset, item_id)
            entry = usage.setdefault(
                key,
                CandidateUsage(
                    dataset=dataset,
                    item_id=item_id,
                    candidate_set_ids=set(),
                ),
            )
            entry.candidate_set_ids.add(candidate_set_id)
            entry.candidate_occurrence_count += 1
        if positive_item_id:
            key = (dataset, positive_item_id)
            entry = usage.setdefault(
                key,
                CandidateUsage(
                    dataset=dataset,
                    item_id=positive_item_id,
                    candidate_set_ids=set(),
                ),
            )
            entry.positive_occurrence_count += 1
    return usage


def reconcile_title_map(
    records: Iterable[TitleSourceRecord],
    usage: Mapping[tuple[str, str], CandidateUsage],
    *,
    include_source_only_items: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build canonical item-title map rows and issue rows."""

    records_by_key: dict[tuple[str, str], list[TitleSourceRecord]] = defaultdict(list)
    duplicate_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    for record in records:
        key = (record.dataset, preserve_item_id(record.item_id))
        if not key[1]:
            continue
        records_by_key[key].append(record)
        duplicate_counts[
            (
                record.dataset,
                record.item_id,
                record.source_path,
                record.source_field,
                record.title,
            )
        ] += 1

    keys = set(usage)
    if include_source_only_items:
        keys.update(records_by_key)

    rows: list[dict[str, Any]] = []
    for dataset, item_id in sorted(keys, key=lambda key: (key[0], _sort_key(key[1]))):
        item_records = sorted(
            records_by_key.get((dataset, item_id), []),
            key=_record_sort_key,
        )
        item_usage = usage.get((dataset, item_id))
        rows.append(_reconcile_one_item(dataset, item_id, item_records, item_usage))

    issues = generate_issue_rows(rows, duplicate_counts)
    return rows, issues


def generate_issue_rows(
    map_rows: Iterable[Mapping[str, Any]],
    duplicate_counts: Counter[tuple[str, str, str, str, str]] | None = None,
    malformed_records: Iterable[MalformedSourceRecord] | None = None,
) -> list[dict[str, Any]]:
    """Generate aggregate issue rows for unresolved and conflicted map records."""

    issues: list[dict[str, Any]] = []
    issue_index = 1
    for row in map_rows:
        candidate_count = int(row["candidate_occurrence_count"])
        positive_count = int(row["positive_occurrence_count"])
        if row["title_status"] in {"missing_title", "unverifiable_mapping"} and candidate_count > 0:
            issue_type = (
                "unverifiable_id_mapping"
                if row["title_status"] == "unverifiable_mapping"
                else "missing_title"
            )
            issues.append(
                _issue_row(
                    issue_index,
                    row,
                    issue_type=issue_type,
                    severity="blocking",
                    evidence=(
                        "candidate item has no selected title"
                        if issue_type == "missing_title"
                        else "candidate item has only unverifiable title mappings"
                    ),
                    selected_resolution="canonical_title left empty",
                    blocking_reason="title unavailable from verified repository sources",
                    notes=row["notes"],
                )
            )
            issue_index += 1
            if int(row["source_record_count"]) == 0:
                issues.append(
                    _issue_row(
                        issue_index,
                        row,
                        issue_type="missing_source_item",
                        severity="blocking",
                        evidence="candidate item is absent from verified title sources",
                        selected_resolution="canonical_title left empty",
                        blocking_reason="item not present in available title catalogs",
                        notes=row["notes"],
                    )
                )
                issue_index += 1
        if int(row["title_conflict_count"]) > 0:
            issues.append(
                _issue_row(
                    issue_index,
                    row,
                    issue_type="conflicting_titles",
                    severity="warning",
                    evidence=f"{row['title_conflict_count']} alternate title group(s)",
                    selected_resolution="selected by deterministic source priority",
                    blocking_reason="",
                    notes="alternate_titles retains non-selected values",
                )
            )
            issue_index += 1

    if duplicate_counts:
        for (dataset, item_id, source_path, source_field, title), count in sorted(duplicate_counts.items()):
            if count <= 1:
                continue
            issues.append(
                {
                    "issue_id": f"ITM-{issue_index:06d}",
                    "dataset": dataset,
                    "item_id": item_id,
                    "issue_type": "duplicate_source_record",
                    "severity": "warning",
                    "source_paths": source_path,
                    "candidate_set_ids": "",
                    "candidate_occurrence_count": "0",
                    "positive_occurrence_count": "0",
                    "evidence": f"{count} identical records for {source_field}",
                    "selected_resolution": "duplicate records collapsed during title selection",
                    "blocking_reason": "",
                    "notes": f"title={title}",
                }
            )
            issue_index += 1

    if malformed_records:
        for malformed in malformed_records:
            issues.append(
                {
                    "issue_id": f"ITM-{issue_index:06d}",
                    "dataset": malformed.dataset,
                    "item_id": malformed.item_id,
                    "issue_type": malformed.issue_type,
                    "severity": "warning",
                    "source_paths": malformed.source_path,
                    "candidate_set_ids": "",
                    "candidate_occurrence_count": "0",
                    "positive_occurrence_count": "0",
                    "evidence": malformed.evidence,
                    "selected_resolution": "record excluded from canonical title selection",
                    "blocking_reason": "",
                    "notes": malformed.notes,
                }
            )
            issue_index += 1
    return issues


def coverage_summary_by_candidate_set(
    candidate_rows: Iterable[Mapping[str, Any]],
    map_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute title coverage for each canonical candidate-set ID."""

    title_by_key = {
        (str(row["dataset"]), preserve_item_id(row["item_id"])): cleanup_title(row["canonical_title"])
        for row in map_rows
    }
    grouped_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped_rows[str(row["candidate_set_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for candidate_set_id in sorted(grouped_rows):
        rows = grouped_rows[candidate_set_id]
        dataset = str(rows[0]["dataset"])
        source_plan = str(rows[0]["source_plan"])
        source_path = str(rows[0]["source_path"])
        unique_items: set[str] = set()
        titled_unique_items: set[str] = set()
        positive_items: set[str] = set()
        titled_positive_items: set[str] = set()
        candidate_occurrences = 0
        titled_candidate_occurrences = 0
        rows_with_all_titles = 0
        for row in rows:
            row_all_titled = True
            items = decode_candidate_items(row["candidate_items"])
            for item_id in items:
                key = (dataset, item_id)
                unique_items.add(item_id)
                candidate_occurrences += 1
                if title_by_key.get(key):
                    titled_unique_items.add(item_id)
                    titled_candidate_occurrences += 1
                else:
                    row_all_titled = False
            positive_item_id = preserve_item_id(row["positive_item_id"])
            if positive_item_id:
                positive_items.add(positive_item_id)
                if title_by_key.get((dataset, positive_item_id)):
                    titled_positive_items.add(positive_item_id)
            if row_all_titled:
                rows_with_all_titles += 1
        row_count = len(rows)
        rows_with_missing_titles = row_count - rows_with_all_titles
        summaries.append(
            {
                "dataset": dataset,
                "candidate_set_id": candidate_set_id,
                "source_plan": source_plan,
                "source_path": source_path,
                "candidate_row_count": row_count,
                "unique_item_count": len(unique_items),
                "titled_unique_item_count": len(titled_unique_items),
                "unique_item_title_coverage": _proportion(len(titled_unique_items), len(unique_items)),
                "candidate_occurrence_count": candidate_occurrences,
                "titled_candidate_occurrence_count": titled_candidate_occurrences,
                "candidate_occurrence_title_coverage": _proportion(
                    titled_candidate_occurrences, candidate_occurrences
                ),
                "rows_with_all_titles": rows_with_all_titles,
                "rows_with_missing_titles": rows_with_missing_titles,
                "row_complete_title_coverage": _proportion(rows_with_all_titles, row_count),
                "positive_item_count": len(positive_items),
                "titled_positive_item_count": len(titled_positive_items),
                "positive_title_coverage": _proportion(len(titled_positive_items), len(positive_items)),
                "notes": "computed from item_title_map.csv and ranking_candidate_sets.csv",
            }
        )
    return summaries


def alternate_titles_json(records: Iterable[Mapping[str, Any] | TitleSourceRecord]) -> str:
    """Serialize alternate title objects deterministically for CSV storage."""

    normalized: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, TitleSourceRecord):
            normalized.append(
                {
                    "title": record.title,
                    "source_path": record.source_path,
                    "source_field": record.source_field,
                    "source_origin": record.source_origin,
                    "source_priority": record.source_priority,
                }
            )
        else:
            normalized.append(dict(record))
    normalized.sort(
        key=lambda item: (
            str(item.get("title", "")),
            str(item.get("source_path", "")),
            str(item.get("source_field", "")),
        )
    )
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _reconcile_one_item(
    dataset: str,
    item_id: str,
    records: list[TitleSourceRecord],
    usage: CandidateUsage | None,
) -> dict[str, Any]:
    verified_records = [record for record in records if record.mapping_status == "verified"]
    non_empty_records = [record for record in verified_records if not is_missing_title(record.title)]
    all_non_empty_records = [record for record in records if not is_missing_title(record.title)]
    selected_record: TitleSourceRecord | None = None
    canonical_title = ""
    selected_title_norm = ""
    status = "missing_title"
    conflict_count = 0
    alternate_records: list[TitleSourceRecord] = []

    if non_empty_records:
        selected_record = sorted(non_empty_records, key=_record_sort_key)[0]
        canonical_title = selected_record.title
        selected_title_norm = comparison_normalize_title(canonical_title)
        grouped_titles: dict[str, list[TitleSourceRecord]] = defaultdict(list)
        for record in non_empty_records:
            grouped_titles[comparison_normalize_title(record.title)].append(record)
        if len(non_empty_records) == 1:
            status = "resolved_single_source"
        elif len(grouped_titles) == 1:
            status = "resolved_agreement"
        else:
            status = "resolved_by_priority"
            conflict_count = len(grouped_titles) - 1
            for norm_title, title_records in sorted(grouped_titles.items()):
                if norm_title == selected_title_norm:
                    continue
                alternate_records.append(sorted(title_records, key=_record_sort_key)[0])
    elif all_non_empty_records:
        status = "unverifiable_mapping"

    source_paths = sorted({record.source_path for record in records})
    candidate_set_ids = sorted(usage.candidate_set_ids) if usage else []
    notes = []
    if not records:
        notes.append("no source record for item")
    if conflict_count:
        notes.append("conflicting non-empty titles resolved by source priority")
    if status == "unverifiable_mapping":
        notes.append("only unverifiable mapping records were found")

    return {
        "dataset": dataset,
        "item_id": item_id,
        "canonical_title": canonical_title,
        "title_status": status,
        "selected_source_path": selected_record.source_path if selected_record else "",
        "selected_source_field": selected_record.source_field if selected_record else "",
        "selected_source_origin": selected_record.source_origin if selected_record else "",
        "selected_source_priority": selected_record.source_priority if selected_record else "",
        "source_record_count": len(records),
        "non_empty_source_count": len(non_empty_records),
        "title_conflict_count": conflict_count,
        "alternate_titles": alternate_titles_json(alternate_records),
        "in_candidate_sets": "true" if usage and usage.candidate_occurrence_count > 0 else "false",
        "candidate_set_count": len(candidate_set_ids),
        "candidate_occurrence_count": usage.candidate_occurrence_count if usage else 0,
        "positive_occurrence_count": usage.positive_occurrence_count if usage else 0,
        "notes": "; ".join(notes),
        "_candidate_set_ids": ";".join(candidate_set_ids),
        "_source_paths": ";".join(source_paths),
    }


def _issue_row(
    issue_index: int,
    map_row: Mapping[str, Any],
    *,
    issue_type: str,
    severity: str,
    evidence: str,
    selected_resolution: str,
    blocking_reason: str,
    notes: str,
) -> dict[str, Any]:
    if issue_type not in ALLOWED_ISSUE_TYPES:
        raise ValueError(f"unknown issue type: {issue_type}")
    if severity not in ALLOWED_SEVERITIES:
        raise ValueError(f"unknown issue severity: {severity}")
    return {
        "issue_id": f"ITM-{issue_index:06d}",
        "dataset": map_row["dataset"],
        "item_id": map_row["item_id"],
        "issue_type": issue_type,
        "severity": severity,
        "source_paths": map_row.get("_source_paths", map_row["selected_source_path"]),
        "candidate_set_ids": map_row.get("_candidate_set_ids", ""),
        "candidate_occurrence_count": map_row["candidate_occurrence_count"],
        "positive_occurrence_count": map_row["positive_occurrence_count"],
        "evidence": evidence,
        "selected_resolution": selected_resolution,
        "blocking_reason": blocking_reason,
        "notes": notes,
    }


def _record_sort_key(record: TitleSourceRecord) -> tuple[Any, ...]:
    return (
        record.source_priority,
        record.source_path,
        record.source_field,
        record.source_origin,
        comparison_normalize_title(record.title),
    )


def _sort_key(value: str) -> tuple[int, Any]:
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _proportion(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator

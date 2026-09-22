from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.audit.common_io import write_dataframe_csv as _write_dataframe_csv


DATASET_STATISTICS_COLUMNS = [
    "dataset",
    "artifact_status",
    "artifact_origin",
    "interaction_path",
    "user_mapping_path",
    "item_mapping_path",
    "title_mapping_path",
    "train_path",
    "validation_path",
    "test_path",
    "ground_truth_path",
    "negative_samples_path",
    "candidate_manifest_path",
    "num_users",
    "num_items",
    "num_interactions",
    "num_eligible_users",
    "num_evaluated_users",
    "filtering_protocol",
    "split_protocol",
    "candidate_protocol",
    "candidate_count",
    "title_mapping_coverage",
    "paper_alignment",
    "blocking_reason",
    "notes",
]

RANKING_CANDIDATE_SETS_COLUMNS = [
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

POSITIVE_POSITIONS_COLUMNS = [
    "candidate_set_id",
    "source_path",
    "source_plan",
    "dataset",
    "user_id",
    "positive_item_id",
    "candidate_count",
    "stored_positive_position",
    "stored_position_base",
    "computed_positive_position_zero_based",
    "computed_positive_position_one_based",
    "position_matches_stored",
    "position_validation_status",
    "notes",
]

TRUE_VALUES = {"true", "1", "yes", "y", "positive", "ground_truth"}
FALSE_VALUES = {"false", "0", "no", "n", "negative"}
STATUS_TRUE = "true"
STATUS_FALSE = "false"
STATUS_NOT_VERIFIABLE = "not_verifiable_from_current_artifacts"


class CandidateSetValidationError(ValueError):
    """Raised when a candidate artifact has an unrecoverable schema problem."""


@dataclass(frozen=True)
class DatasetContext:
    dataset: str
    interaction_path: Path | None = None
    ground_truth_path: Path | None = None
    catalog_path: Path | None = None
    datamaps_path: Path | None = None
    negative_samples_path: Path | None = None
    raw_paths: tuple[Path, ...] = ()
    artifact_origin: str = ""
    artifact_status: str = ""
    filtering_protocol: str = ""
    split_protocol: str = ""
    candidate_protocol: str = ""
    user_ids: frozenset[str] = frozenset()
    item_ids: frozenset[str] = frozenset()
    title_by_item: dict[str, str] | None = None
    ground_truth_by_user: dict[str, str] | None = None
    history_by_user: dict[str, frozenset[str]] | None = None
    num_users: int | str = ""
    num_items: int | str = ""
    num_interactions: int | str = ""
    num_eligible_users: int | str = ""
    blocking_reason: str = ""
    notes: str = ""


@dataclass(frozen=True)
class RowCandidateArtifact:
    candidate_set_id: str
    source_path: Path
    source_plan: str
    dataset: str
    context_key: str
    expected_candidate_count: int
    paper_alignment: str
    row_format: str
    user_column: str = "user_id"
    item_column: str = "candidate_item_id"
    positive_column: str = "is_ground_truth"
    position_column: str = "candidate_position"


@dataclass(frozen=True)
class ListCandidateArtifact:
    candidate_set_id_prefix: str
    source_path: Path
    source_plan: str
    dataset: str
    context_key: str
    expected_candidate_count: int
    paper_alignment: str
    row_format: str
    list_column: str = "candidate_ids"
    user_column: str = "user_id"
    positive_column: str = "positive_item_id"
    stored_position_column: str | None = None
    candidate_set_suffix_column: str | None = None


def parse_candidate_items(value: object) -> list[str]:
    """Parse a JSON or Python-list encoded candidate list while preserving ID strings."""

    if isinstance(value, list):
        parsed = value
    else:
        text = "" if value is None else str(value).strip()
        if text == "":
            raise CandidateSetValidationError("candidate list is empty")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError) as exc:
                raise CandidateSetValidationError("candidate list is not parseable as JSON or Python literal") from exc
    if not isinstance(parsed, list):
        raise CandidateSetValidationError("candidate list payload is not a list")
    return [normalise_id(item) for item in parsed]


def normalise_id(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_bool(value: object) -> bool:
    text = normalise_id(value).lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return False


def validate_candidate_row(
    *,
    candidate_set_id: str,
    source_path: str,
    source_plan: str,
    dataset: str,
    user_id: str,
    positive_item_id: str,
    candidate_items: list[str],
    stored_positive_position: str | int | None,
    expected_candidate_count: int | None,
    context: DatasetContext | None,
) -> dict[str, str | int]:
    """Validate one canonical candidate row and return machine-readable fields."""

    user_id = normalise_id(user_id)
    positive_item_id = normalise_id(positive_item_id)
    candidate_items = [normalise_id(item) for item in candidate_items]
    position = compute_positive_position(positive_item_id, candidate_items)
    stored = normalise_id(stored_positive_position)
    stored_base, position_matches = infer_stored_position_base(stored, position["zero_based"], position["one_based"])
    candidate_count = len(candidate_items)
    positive_occurrences = candidate_items.count(positive_item_id) if positive_item_id else 0
    candidate_items_unique = len(candidate_items) == len(set(candidate_items))
    positive_in_candidates = positive_occurrences > 0
    negative_count = candidate_count - positive_occurrences if positive_occurrences else candidate_count

    issues: list[str] = []
    warnings: list[str] = []

    if user_id == "":
        issues.append("empty_user_id")
    if positive_item_id == "":
        issues.append("empty_positive_item_id")
    if candidate_count == 0:
        issues.append("empty_candidate_list")
    if expected_candidate_count is not None and candidate_count != expected_candidate_count:
        issues.append(f"candidate_count_mismatch_expected_{expected_candidate_count}")
    if any(item == "" for item in candidate_items):
        issues.append("empty_candidate_id")
    if not candidate_items_unique:
        issues.append("duplicate_candidate_items")
    if positive_occurrences != 1:
        issues.append(f"positive_occurrence_count_{positive_occurrences}")
    if stored and position_matches == STATUS_FALSE:
        issues.append("stored_positive_position_mismatch")

    user_mapping_valid = _mapping_status(user_id, context.user_ids if context else frozenset())
    positive_mapping_valid = _mapping_status(positive_item_id, context.item_ids if context else frozenset())
    candidate_mapping_valid = _all_mapping_status(candidate_items, context.item_ids if context else frozenset())
    title_available = _title_status(candidate_items, context.title_by_item if context else None)
    ground_truth_match = _ground_truth_status(user_id, positive_item_id, context.ground_truth_by_user if context else None)
    history_valid = _history_status(user_id, positive_item_id, candidate_items, context.history_by_user if context else None)

    if user_mapping_valid == STATUS_FALSE:
        issues.append("user_mapping_missing")
    if positive_mapping_valid == STATUS_FALSE:
        issues.append("positive_item_mapping_missing")
    if candidate_mapping_valid == STATUS_FALSE:
        issues.append("candidate_item_mapping_missing")
    if title_available == STATUS_FALSE:
        warnings.append("candidate_title_mapping_incomplete")
    if ground_truth_match == STATUS_FALSE:
        issues.append("ground_truth_mismatch")
    if history_valid == STATUS_FALSE:
        issues.append("negative_history_leakage")

    if issues:
        row_status = "invalid"
    elif warnings:
        row_status = "valid_with_warnings"
    else:
        row_status = "valid"

    validation_issues = ";".join([*issues, *warnings]) if issues or warnings else "none"
    return {
        "candidate_set_id": candidate_set_id,
        "source_path": source_path,
        "source_plan": source_plan,
        "dataset": dataset,
        "user_id": user_id,
        "positive_item_id": positive_item_id,
        "candidate_items": json.dumps(candidate_items, ensure_ascii=True),
        "candidate_count": candidate_count,
        "negative_count": negative_count,
        "stored_positive_position": stored,
        "stored_position_base": stored_base,
        "computed_positive_position_zero_based": _position_text(position["zero_based"]),
        "computed_positive_position_one_based": _position_text(position["one_based"]),
        "positive_occurrence_count": positive_occurrences,
        "candidate_items_unique": _bool_text(candidate_items_unique),
        "positive_in_candidates": _bool_text(positive_in_candidates),
        "user_mapping_valid": user_mapping_valid,
        "positive_item_mapping_valid": positive_mapping_valid,
        "all_candidate_items_mapping_valid": candidate_mapping_valid,
        "all_candidate_titles_available": title_available,
        "ground_truth_match": ground_truth_match,
        "negative_history_exclusion_valid": history_valid,
        "row_validation_status": row_status,
        "validation_issues": validation_issues,
    }


def compute_positive_position(positive_item_id: str, candidate_items: list[str]) -> dict[str, int | None]:
    positive_item_id = normalise_id(positive_item_id)
    if positive_item_id == "":
        return {"zero_based": None, "one_based": None}
    try:
        zero_based = candidate_items.index(positive_item_id)
    except ValueError:
        return {"zero_based": None, "one_based": None}
    return {"zero_based": zero_based, "one_based": zero_based + 1}


def infer_stored_position_base(
    stored_positive_position: str | int | None,
    computed_zero_based: int | None,
    computed_one_based: int | None,
) -> tuple[str, str]:
    stored = normalise_id(stored_positive_position)
    if stored == "":
        return "not_available", STATUS_NOT_VERIFIABLE
    try:
        stored_int = int(float(stored))
    except ValueError:
        return "unknown", STATUS_FALSE
    if computed_zero_based is not None and stored_int == computed_zero_based:
        return "zero_based", STATUS_TRUE
    if computed_one_based is not None and stored_int == computed_one_based:
        return "one_based", STATUS_TRUE
    return "unknown", STATUS_FALSE


def canonicalize_row_candidate_artifact(
    artifact: RowCandidateArtifact,
    *,
    project_root: Path,
    context: DatasetContext | None,
) -> pd.DataFrame:
    """Canonicalize a row-per-candidate artifact into one row per user."""

    path = project_root / artifact.source_path
    columns = [artifact.user_column, artifact.item_column, artifact.positive_column]
    if artifact.position_column:
        columns.append(artifact.position_column)
    header = _header_columns(path)
    missing = [column for column in columns if column not in header]
    if missing:
        raise CandidateSetValidationError(f"{artifact.source_path}: missing columns {missing}")
    frame = pd.read_csv(path, usecols=list(dict.fromkeys(columns)), dtype=str).fillna("")
    frame["__input_order"] = range(len(frame))
    frame["__user"] = frame[artifact.user_column].map(normalise_id)
    frame["__item"] = frame[artifact.item_column].map(normalise_id)
    frame["__is_positive"] = frame[artifact.positive_column].map(parse_bool)
    if artifact.position_column:
        frame["__position"] = pd.to_numeric(frame[artifact.position_column], errors="coerce")
    else:
        frame["__position"] = pd.NA
    frame = frame.sort_values(["__user", "__position", "__input_order"], kind="mergesort", na_position="last")

    rows: list[dict[str, str | int]] = []
    for user_id, group in frame.groupby("__user", sort=False):
        candidate_items = group["__item"].tolist()
        positive_rows = group.loc[group["__is_positive"]]
        positive_item_id = normalise_id(positive_rows.iloc[0]["__item"]) if not positive_rows.empty else ""
        stored_position = (
            normalise_id(positive_rows.iloc[0][artifact.position_column])
            if not positive_rows.empty and artifact.position_column
            else ""
        )
        row = validate_candidate_row(
            candidate_set_id=artifact.candidate_set_id,
            source_path=artifact.source_path.as_posix(),
            source_plan=artifact.source_plan,
            dataset=artifact.dataset,
            user_id=user_id,
            positive_item_id=positive_item_id,
            candidate_items=candidate_items,
            stored_positive_position=stored_position,
            expected_candidate_count=artifact.expected_candidate_count,
            context=context,
        )
        declared_positive_count = int(group["__is_positive"].sum())
        if declared_positive_count != 1:
            row = _append_issue(row, f"declared_positive_count_{declared_positive_count}")
        rows.append(row)
    return pd.DataFrame(rows, columns=RANKING_CANDIDATE_SETS_COLUMNS)


def canonicalize_list_candidate_artifact(
    artifact: ListCandidateArtifact,
    *,
    project_root: Path,
    context: DatasetContext | None,
) -> pd.DataFrame:
    """Canonicalize a row-per-user artifact with an encoded candidate list."""

    path = project_root / artifact.source_path
    columns = [artifact.user_column, artifact.positive_column, artifact.list_column]
    if artifact.stored_position_column:
        columns.append(artifact.stored_position_column)
    if artifact.candidate_set_suffix_column:
        columns.append(artifact.candidate_set_suffix_column)
    header = _header_columns(path)
    missing = [column for column in columns if column not in header]
    if missing:
        raise CandidateSetValidationError(f"{artifact.source_path}: missing columns {missing}")
    frame = pd.read_csv(path, usecols=list(dict.fromkeys(columns)), dtype=str).fillna("")

    rows: list[dict[str, str | int]] = []
    for _, source_row in frame.iterrows():
        suffix = ""
        if artifact.candidate_set_suffix_column:
            suffix = normalise_id(source_row[artifact.candidate_set_suffix_column])
        candidate_set_id = artifact.candidate_set_id_prefix if suffix == "" else f"{artifact.candidate_set_id_prefix}_{_slug(suffix)}"
        try:
            candidate_items = parse_candidate_items(source_row[artifact.list_column])
        except CandidateSetValidationError as exc:
            raise CandidateSetValidationError(f"{artifact.source_path}: malformed candidate list for user {source_row.get(artifact.user_column, '')}: {exc}") from exc
        stored_position = (
            source_row[artifact.stored_position_column]
            if artifact.stored_position_column
            else ""
        )
        rows.append(
            validate_candidate_row(
                candidate_set_id=candidate_set_id,
                source_path=artifact.source_path.as_posix(),
                source_plan=artifact.source_plan,
                dataset=artifact.dataset,
                user_id=source_row[artifact.user_column],
                positive_item_id=source_row[artifact.positive_column],
                candidate_items=candidate_items,
                stored_positive_position=stored_position,
                expected_candidate_count=artifact.expected_candidate_count,
                context=context,
            )
        )
    return pd.DataFrame(rows, columns=RANKING_CANDIDATE_SETS_COLUMNS)


def derive_positive_positions(candidate_sets: pd.DataFrame) -> pd.DataFrame:
    """Derive the positive-position output directly from canonical candidate rows."""

    rows: list[dict[str, str | int]] = []
    for _, row in candidate_sets.iterrows():
        matches = _position_match_status(str(row["stored_position_base"]))
        status = "valid" if matches == STATUS_TRUE else (
            STATUS_NOT_VERIFIABLE if matches == STATUS_NOT_VERIFIABLE else "invalid"
        )
        notes = "derived_from_ranking_candidate_sets_csv"
        if row["stored_position_base"] == "not_available":
            notes = "no stored positive position in source artifact"
        rows.append(
            {
                "candidate_set_id": row["candidate_set_id"],
                "source_path": row["source_path"],
                "source_plan": row["source_plan"],
                "dataset": row["dataset"],
                "user_id": row["user_id"],
                "positive_item_id": row["positive_item_id"],
                "candidate_count": row["candidate_count"],
                "stored_positive_position": row["stored_positive_position"],
                "stored_position_base": row["stored_position_base"],
                "computed_positive_position_zero_based": row["computed_positive_position_zero_based"],
                "computed_positive_position_one_based": row["computed_positive_position_one_based"],
                "position_matches_stored": matches,
                "position_validation_status": status,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=POSITIVE_POSITIONS_COLUMNS)


def load_dataset_context(
    *,
    dataset: str,
    interaction_path: Path | None = None,
    ground_truth_path: Path | None = None,
    catalog_path: Path | None = None,
    datamaps_path: Path | None = None,
    negative_samples_path: Path | None = None,
    raw_paths: tuple[Path, ...] = (),
    artifact_origin: str = "",
    artifact_status: str = "",
    filtering_protocol: str = "",
    split_protocol: str = "",
    candidate_protocol: str = "",
    blocking_reason: str = "",
    notes: str = "",
) -> DatasetContext:
    """Load available mappings and statistics for a dataset artifact family."""

    user_ids: set[str] = set()
    item_ids: set[str] = set()
    title_by_item: dict[str, str] = {}
    ground_truth_by_user: dict[str, str] = {}
    history_by_user: dict[str, set[str]] = {}
    num_interactions: int | str = ""

    if datamaps_path and datamaps_path.exists():
        payload = json.loads(datamaps_path.read_text(encoding="utf-8"))
        user_ids.update(normalise_id(value) for value in payload.get("user2id", {}).keys())
        item_ids.update(normalise_id(value) for value in payload.get("item2id", {}).keys())

    if catalog_path and catalog_path.exists():
        catalog = pd.read_csv(catalog_path, dtype=str).fillna("")
        if "item_id" in catalog.columns:
            item_ids.update(catalog["item_id"].map(normalise_id).tolist())
            title_column = "title" if "title" in catalog.columns else ""
            if title_column:
                title_by_item.update(
                    {
                        normalise_id(row["item_id"]): normalise_id(row[title_column])
                        for _, row in catalog.loc[:, ["item_id", title_column]].iterrows()
                    }
                )

    if ground_truth_path and ground_truth_path.exists():
        gt = pd.read_csv(ground_truth_path, dtype=str).fillna("")
        user_col = "user_id" if "user_id" in gt.columns else ""
        item_col = "ground_truth_item_id" if "ground_truth_item_id" in gt.columns else ("item_id" if "item_id" in gt.columns else "")
        if user_col and item_col:
            for _, row in gt.loc[:, [user_col, item_col]].iterrows():
                user = normalise_id(row[user_col])
                item = normalise_id(row[item_col])
                if user and item:
                    ground_truth_by_user[user] = item
                    user_ids.add(user)
                    item_ids.add(item)

    if interaction_path and interaction_path.exists():
        interactions = pd.read_csv(interaction_path, dtype=str).fillna("")
        num_interactions = len(interactions)
        if {"user_id", "item_id"}.issubset(interactions.columns):
            users = interactions["user_id"].map(normalise_id)
            items = interactions["item_id"].map(normalise_id)
            user_ids.update(users.tolist())
            item_ids.update(items.tolist())
            if "split" in interactions.columns:
                train = interactions.loc[interactions["split"].astype(str).str.lower().eq("train")]
            else:
                train = interactions
            for user, group in train.groupby("user_id", sort=False):
                user_key = normalise_id(user)
                history_by_user[user_key] = set(group["item_id"].map(normalise_id).tolist())

    return DatasetContext(
        dataset=dataset,
        interaction_path=interaction_path,
        ground_truth_path=ground_truth_path,
        catalog_path=catalog_path,
        datamaps_path=datamaps_path,
        negative_samples_path=negative_samples_path,
        raw_paths=raw_paths,
        artifact_origin=artifact_origin,
        artifact_status=artifact_status,
        filtering_protocol=filtering_protocol,
        split_protocol=split_protocol,
        candidate_protocol=candidate_protocol,
        user_ids=frozenset(user_ids),
        item_ids=frozenset(item_ids),
        title_by_item=title_by_item,
        ground_truth_by_user=ground_truth_by_user,
        history_by_user={user: frozenset(items) for user, items in history_by_user.items()},
        num_users=len(user_ids) if user_ids else "",
        num_items=len(item_ids) if item_ids else "",
        num_interactions=num_interactions,
        num_eligible_users=len(ground_truth_by_user) if ground_truth_by_user else "",
        blocking_reason=blocking_reason,
        notes=notes,
    )


def title_coverage(candidate_sets: pd.DataFrame, candidate_set_id: str) -> str:
    subset = candidate_sets.loc[candidate_sets["candidate_set_id"] == candidate_set_id]
    if subset.empty:
        return ""
    total = int(subset["candidate_count"].astype(int).sum())
    if total == 0:
        return ""
    available = 0
    for _, row in subset.iterrows():
        items = parse_candidate_items(row["candidate_items"])
        if row["all_candidate_titles_available"] == STATUS_TRUE:
            available += len(items)
        elif row["all_candidate_titles_available"] == STATUS_NOT_VERIFIABLE:
            return STATUS_NOT_VERIFIABLE
        else:
            # Row-level output only records all-or-not-all coverage. Keep the
            # aggregate conservative when exact missing-title counts are not present.
            return "partial"
    return f"{available / total:.6f}"


def write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    _write_dataframe_csv(path, frame, columns=columns, lineterminator="\n")


def _mapping_status(value: str, mapping: frozenset[str]) -> str:
    if not mapping:
        return STATUS_NOT_VERIFIABLE
    return _bool_text(value in mapping)


def _all_mapping_status(values: list[str], mapping: frozenset[str]) -> str:
    if not mapping:
        return STATUS_NOT_VERIFIABLE
    return _bool_text(all(value in mapping for value in values))


def _title_status(values: list[str], title_by_item: dict[str, str] | None) -> str:
    if not title_by_item:
        return STATUS_NOT_VERIFIABLE
    return _bool_text(all(normalise_id(title_by_item.get(value, "")) != "" for value in values))


def _ground_truth_status(user_id: str, positive_item_id: str, ground_truth_by_user: dict[str, str] | None) -> str:
    if not ground_truth_by_user:
        return STATUS_NOT_VERIFIABLE
    expected = ground_truth_by_user.get(user_id)
    if expected is None:
        return STATUS_FALSE
    return _bool_text(expected == positive_item_id)


def _history_status(
    user_id: str,
    positive_item_id: str,
    candidate_items: list[str],
    history_by_user: dict[str, frozenset[str]] | None,
) -> str:
    if not history_by_user:
        return STATUS_NOT_VERIFIABLE
    history = history_by_user.get(user_id)
    if history is None:
        return STATUS_FALSE
    negatives = [item for item in candidate_items if item != positive_item_id]
    return _bool_text(all(item not in history for item in negatives))


def _append_issue(row: dict[str, str | int], issue: str) -> dict[str, str | int]:
    existing = str(row["validation_issues"])
    row = dict(row)
    row["validation_issues"] = issue if existing == "none" else f"{existing};{issue}"
    row["row_validation_status"] = "invalid"
    return row


def _position_text(value: int | None) -> str:
    return "" if value is None else str(value)


def _position_match_status(stored_position_base: str) -> str:
    if stored_position_base in {"zero_based", "one_based"}:
        return STATUS_TRUE
    if stored_position_base == "not_available":
        return STATUS_NOT_VERIFIABLE
    return STATUS_FALSE


def _header_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _slug(value: str) -> str:
    return "_".join(part for part in value.lower().replace("-", "_").split() if part).replace("/", "_")


def _bool_text(value: bool) -> str:
    return STATUS_TRUE if value else STATUS_FALSE


def sorted_unique(values: Iterable[Any]) -> list[str]:
    return sorted({normalise_id(value) for value in values if normalise_id(value)})

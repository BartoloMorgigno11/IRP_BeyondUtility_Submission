"""Plan E BPRMF candidate adapter utilities.

This module prepares and audits raw Plan E2 candidate rows for a future
BPRMF pilot. It deliberately does not train a model, score project data, or
import torch at module import time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PLAN_E2_CANDIDATE_COUNT = 20
PLAN_E2_CANDIDATE_SET_IDS = {
    "plan_e_beauty_1000user_ranking_manifest",
    "plan_e_beauty_1000user_reranking_manifest",
    "plan_e_sports_1000user_ranking_manifest",
    "plan_e_sports_1000user_reranking_manifest",
    "plan_e_ml1m_1000user_ranking_manifest",
    "plan_e_ml1m_1000user_reranking_manifest",
}


class BPRMFAdapterError(ValueError):
    """Base exception for adapter validation failures."""


class MappingValidationError(BPRMFAdapterError):
    """Raised when an explicit ID mapping is invalid or ambiguous."""


class CandidateValidationError(BPRMFAdapterError):
    """Raised when a candidate row cannot be safely adapted."""


class ScoreAlignmentError(BPRMFAdapterError):
    """Raised when future score vectors cannot be aligned to candidates."""


@dataclass(frozen=True)
class EntityMapping:
    """Validated dataset-scoped raw-to-internal mapping."""

    entity_type: str
    raw_to_internal: dict[str, int]
    internal_to_raw: dict[int, str]
    source_path: str
    mapping_direction: str = "raw_to_internal"

    @property
    def raw_id_count(self) -> int:
        return len(self.raw_to_internal)

    @property
    def internal_id_count(self) -> int:
        return len(self.internal_to_raw)


@dataclass(frozen=True)
class IndexSpaceAudit:
    """Summary of a mapping's explicit internal-ID space."""

    minimum_internal_id: int | None
    maximum_internal_id: int | None
    contiguous_index_space: bool
    missing_internal_index_count: int
    index_zero_semantics: str
    reserved_index_count: int


@dataclass(frozen=True)
class CandidateTensorPayload:
    """Future tensor payload with raw candidate order preserved."""

    user_id: str
    user_index: int
    raw_candidate_item_ids: tuple[str, ...]
    item_indices: tuple[int, ...]


def safe_raw_id(value: Any) -> str:
    """Return a repository ID as a stripped string without numeric coercion."""

    if value is None:
        raise MappingValidationError("raw ID is missing")
    text = str(value).strip()
    if not text:
        raise MappingValidationError("raw ID is empty")
    lowered = text.lower()
    if lowered in {"nan", "none", "null", "<na>"}:
        raise MappingValidationError(f"raw ID has null-like value: {text!r}")
    return text


def parse_internal_id(value: Any) -> int:
    """Parse a verified internal ID as an integer."""

    text = safe_raw_id(value)
    try:
        number = int(text)
    except ValueError as exc:
        raise MappingValidationError(f"internal ID is not an integer: {text!r}") from exc
    if number < 0:
        raise MappingValidationError(f"internal ID must be non-negative: {number}")
    return number


def validate_mapping_direction(mapping_name: str) -> str:
    """Validate an explicit datamap direction name."""

    if mapping_name in {"user2id", "item2id"}:
        return "raw_to_internal"
    if mapping_name in {"id2user", "id2item"}:
        return "internal_to_raw"
    raise MappingValidationError(f"ambiguous mapping direction: {mapping_name}")


def validate_raw_to_internal_pairs(
    pairs: Iterable[tuple[Any, Any]],
    *,
    entity_type: str,
    source_path: str = "",
) -> EntityMapping:
    """Build a validated raw-to-internal mapping from explicit pairs."""

    raw_to_internal: dict[str, int] = {}
    internal_to_raw: dict[int, str] = {}
    duplicate_raw = []
    duplicate_internal = []
    for raw_value, internal_value in pairs:
        raw_id = safe_raw_id(raw_value)
        internal_id = parse_internal_id(internal_value)
        if raw_id in raw_to_internal:
            duplicate_raw.append(raw_id)
        if internal_id in internal_to_raw:
            duplicate_internal.append(internal_id)
        raw_to_internal[raw_id] = internal_id
        internal_to_raw[internal_id] = raw_id
    if duplicate_raw:
        raise MappingValidationError(f"duplicate raw {entity_type} IDs: {sorted(set(duplicate_raw))[:5]}")
    if duplicate_internal:
        raise MappingValidationError(
            f"duplicate internal {entity_type} IDs: {sorted(set(duplicate_internal))[:5]}"
        )
    if not raw_to_internal:
        raise MappingValidationError(f"{entity_type} mapping is empty")
    return EntityMapping(
        entity_type=entity_type,
        raw_to_internal=raw_to_internal,
        internal_to_raw=internal_to_raw,
        source_path=source_path,
    )


def load_raw_to_internal_mapping(datamap_path: str | Path, *, entity_type: str) -> EntityMapping:
    """Load a raw-to-internal user or item mapping from a Plan C datamap."""

    path = Path(datamap_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "user2id" if entity_type == "user" else "item2id"
    inverse_key = "id2user" if entity_type == "user" else "id2item"
    if validate_mapping_direction(key) != "raw_to_internal":
        raise MappingValidationError(f"{key} is not a raw-to-internal mapping")
    if key not in payload or inverse_key not in payload:
        raise MappingValidationError(f"datamap lacks {key} or {inverse_key}: {path}")
    mapping = validate_raw_to_internal_pairs(
        payload[key].items(),
        entity_type=entity_type,
        source_path=str(path).replace("\\", "/"),
    )
    inverse = {parse_internal_id(k): safe_raw_id(v) for k, v in payload[inverse_key].items()}
    if mapping.internal_to_raw != inverse:
        raise MappingValidationError(f"{key} and {inverse_key} are not exact inverses: {path}")
    return mapping


def audit_index_space(mapping: EntityMapping) -> IndexSpaceAudit:
    """Audit contiguity and zero semantics for explicit internal IDs."""

    ids = sorted(mapping.internal_to_raw)
    if not ids:
        return IndexSpaceAudit(None, None, False, 0, "empty_mapping", 0)
    minimum = ids[0]
    maximum = ids[-1]
    observed = set(ids)
    expected = set(range(minimum, maximum + 1))
    missing = expected.difference(observed)
    zero_semantics = (
        "valid_datamap_entity; Plan D dense model index zero is also a valid entity"
        if 0 in observed
        else "absent_from_datamap; Plan D dense model index zero is valid after adapter remapping"
    )
    return IndexSpaceAudit(
        minimum_internal_id=minimum,
        maximum_internal_id=maximum,
        contiguous_index_space=not missing,
        missing_internal_index_count=len(missing),
        index_zero_semantics=zero_semantics,
        reserved_index_count=0,
    )


def read_csv_strings(path: str | Path) -> pd.DataFrame:
    """Read a CSV while preserving IDs as strings."""

    return pd.read_csv(path, dtype="string", keep_default_na=False)


def filter_train_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Return only rows whose split is exactly train after safe normalization."""

    required = {"user_id", "item_id", "split"}
    missing = required.difference(interactions.columns)
    if missing:
        raise CandidateValidationError(f"interactions missing columns: {sorted(missing)}")
    frame = interactions.copy()
    for column in ("user_id", "item_id", "split"):
        frame[column] = frame[column].astype("string").str.strip()
    split = frame["split"].str.lower()
    return frame.loc[split == "train"].copy()


def compute_train_seen_entities(train_interactions: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Return train-seen raw users and items from already-filtered train rows."""

    users = {safe_raw_id(value) for value in train_interactions["user_id"].tolist()}
    items = {safe_raw_id(value) for value in train_interactions["item_id"].tolist()}
    return users, items


def parse_candidate_items(value: Any, *, expected_count: int | None = PLAN_E2_CANDIDATE_COUNT) -> list[str]:
    """Decode a JSON candidate list and preserve raw item IDs as strings."""

    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise CandidateValidationError("candidate_items is not valid JSON") from exc
    if not isinstance(decoded, list):
        raise CandidateValidationError("candidate_items must decode to a list")
    candidates = [safe_raw_id(item) for item in decoded]
    if expected_count is not None and len(candidates) != expected_count:
        raise CandidateValidationError(f"expected {expected_count} candidates, found {len(candidates)}")
    if len(set(candidates)) != len(candidates):
        raise CandidateValidationError("candidate_items contains duplicate raw IDs")
    return candidates


def natural_id_sort_key(value: Any) -> tuple[int, int, str]:
    """Sort numeric-looking IDs numerically and other IDs lexicographically."""

    text = str(value)
    try:
        return (0, int(text), "")
    except ValueError:
        return (1, 0, text)


def build_existing_model_index(
    *,
    train_seen_ids: set[str],
    plan_e2_ids: set[str],
) -> dict[str, int]:
    """Build the zero-based dense raw-ID index used by Plan D-style models."""

    ordered = sorted(set(train_seen_ids).union(plan_e2_ids), key=natural_id_sort_key)
    return {raw_id: index for index, raw_id in enumerate(ordered)}


def ensure_indices_within_cardinality(indices: Iterable[int], *, cardinality: int, entity_type: str) -> None:
    """Validate that model indices fit an embedding cardinality."""

    if cardinality < 1:
        raise MappingValidationError(f"{entity_type} cardinality must be positive")
    bad = [index for index in indices if index < 0 or index >= cardinality]
    if bad:
        raise MappingValidationError(
            f"{entity_type} model indices out of range for cardinality {cardinality}: {bad[:5]}"
        )


def prepare_candidate_tensor_payload(
    *,
    user_id: Any,
    candidate_item_ids: Sequence[Any],
    user_model_index: Mapping[str, int],
    item_model_index: Mapping[str, int],
) -> CandidateTensorPayload:
    """Prepare future tensor indices while preserving raw candidate order."""

    raw_user_id = safe_raw_id(user_id)
    raw_candidates = tuple(safe_raw_id(item) for item in candidate_item_ids)
    if raw_user_id not in user_model_index:
        raise CandidateValidationError(f"user is not in model index: {raw_user_id}")
    missing_items = [item for item in raw_candidates if item not in item_model_index]
    if missing_items:
        raise CandidateValidationError(f"candidate item is not in model index: {missing_items[0]}")
    item_indices = tuple(int(item_model_index[item]) for item in raw_candidates)
    user_index = int(user_model_index[raw_user_id])
    ensure_indices_within_cardinality([user_index], cardinality=len(user_model_index), entity_type="user")
    ensure_indices_within_cardinality(item_indices, cardinality=len(item_model_index), entity_type="item")
    return CandidateTensorPayload(
        user_id=raw_user_id,
        user_index=user_index,
        raw_candidate_item_ids=raw_candidates,
        item_indices=item_indices,
    )


def audit_plan_e2_candidate_row(
    row: Mapping[str, Any],
    *,
    user_mapping: EntityMapping,
    item_mapping: EntityMapping,
    train_seen_users: set[str],
    train_seen_items: set[str],
    user_model_index: Mapping[str, int],
    item_model_index: Mapping[str, int],
    eligible_full: bool,
    eligible_matched: bool,
) -> dict[str, Any]:
    """Audit one Plan E2 candidate row for future BPRMF compatibility."""

    user_id = safe_raw_id(row["user_id"])
    positive_item_id = safe_raw_id(row["positive_item_id"])
    candidate_items = parse_candidate_items(row["candidate_items"])

    user_mapped = user_id in user_mapping.raw_to_internal
    positive_mapped = positive_item_id in item_mapping.raw_to_internal
    unmapped_candidates = [item for item in candidate_items if item not in item_mapping.raw_to_internal]
    train_unseen_candidates = [item for item in candidate_items if item not in train_seen_items]
    user_seen = user_id in train_seen_users
    positive_seen = positive_item_id in train_seen_items

    model_index_compatible = (
        user_mapped
        and positive_mapped
        and not unmapped_candidates
        and user_id in user_model_index
        and all(item in item_model_index for item in candidate_items)
    )
    if model_index_compatible:
        payload = prepare_candidate_tensor_payload(
            user_id=user_id,
            candidate_item_ids=candidate_items,
            user_model_index=user_model_index,
            item_model_index=item_model_index,
        )
        if tuple(candidate_items) != payload.raw_candidate_item_ids:
            raise CandidateValidationError("candidate order changed during tensor preparation")

    strict_train_known = model_index_compatible and user_seen and positive_seen and not train_unseen_candidates
    contains_untrained_risk = model_index_compatible and (
        not user_seen or not positive_seen or bool(train_unseen_candidates)
    )
    reasons = compatibility_reasons(
        user_mapped=user_mapped,
        user_seen=user_seen,
        positive_item_mapped=positive_mapped,
        positive_item_seen=positive_seen,
        unmapped_candidate_count=len(unmapped_candidates),
        train_unseen_candidate_count=len(train_unseen_candidates),
    )
    if not model_index_compatible:
        status = "mapping_incompatible"
    elif not user_seen:
        status = "user_not_seen_in_train"
    elif strict_train_known:
        status = "strict_train_known"
    else:
        status = "index_compatible_with_untrained_embedding_risk"

    return {
        "candidate_set_id": str(row["candidate_set_id"]),
        "source_path": str(row["source_path"]),
        "dataset": str(row["dataset"]),
        "task_type": str(row.get("task_type", "")),
        "user_id": user_id,
        "positive_item_id": positive_item_id,
        "candidate_count": len(candidate_items),
        "user_mapped": user_mapped,
        "user_seen_in_train": user_seen,
        "positive_item_mapped": positive_mapped,
        "positive_item_seen_in_train": positive_seen,
        "all_candidate_items_mapped": not unmapped_candidates,
        "all_candidate_items_seen_in_train": not train_unseen_candidates,
        "unmapped_candidate_item_count": len(unmapped_candidates),
        "unmapped_candidate_item_ids": json.dumps(unmapped_candidates, ensure_ascii=True),
        "train_unseen_candidate_item_count": len(train_unseen_candidates),
        "train_unseen_candidate_item_ids": json.dumps(train_unseen_candidates, ensure_ascii=True),
        "existing_model_index_compatible": model_index_compatible,
        "strict_train_known_compatible": strict_train_known,
        "contains_untrained_embedding_risk": contains_untrained_risk,
        "eligible_plan_e2_full": eligible_full,
        "eligible_plan_e2_matched": eligible_matched,
        "compatibility_status": status,
        "exclusion_reasons": ";".join(reasons) if reasons else "none",
        "notes": (
            "candidate_order_preserved; no project-data scoring"
            if model_index_compatible
            else "mapping incompatibility prevents future tensor preparation"
        ),
    }


def compatibility_reasons(
    *,
    user_mapped: bool,
    user_seen: bool,
    positive_item_mapped: bool,
    positive_item_seen: bool,
    unmapped_candidate_count: int,
    train_unseen_candidate_count: int,
) -> list[str]:
    """Return deterministic compatibility and strict-filter exclusion reasons."""

    reasons: list[str] = []
    if not user_mapped:
        reasons.append("user_unmapped")
    if user_mapped and not user_seen:
        reasons.append("user_not_seen_in_train")
    if not positive_item_mapped:
        reasons.append("positive_item_unmapped")
    if positive_item_mapped and not positive_item_seen:
        reasons.append("positive_item_not_seen_in_train")
    if unmapped_candidate_count:
        reasons.append("candidate_item_unmapped")
    if train_unseen_candidate_count:
        reasons.append("candidate_item_not_seen_in_train")
    return reasons


def align_scores_to_candidate_ids(candidate_item_ids: Sequence[Any], scores: Sequence[Any]) -> list[dict[str, Any]]:
    """Align future score vectors to raw candidate IDs without sorting."""

    raw_candidates = [safe_raw_id(item) for item in candidate_item_ids]
    if len(raw_candidates) != len(scores):
        raise ScoreAlignmentError("score length must equal candidate count")
    rows: list[dict[str, Any]] = []
    for position, (item_id, score_value) in enumerate(zip(raw_candidates, scores)):
        score = float(score_value)
        if not math.isfinite(score):
            raise ScoreAlignmentError("scores must be finite")
        rows.append(
            {
                "candidate_position_zero_based": position,
                "candidate_item_id": item_id,
                "score": score,
            }
        )
    return rows


def json_array(values: Iterable[Any]) -> str:
    """Serialize a deterministic JSON array for CSV output."""

    return json.dumps([safe_raw_id(value) for value in values], ensure_ascii=True)


def bool_text(value: bool) -> str:
    """Serialize booleans consistently for CSV output."""

    return "true" if bool(value) else "false"

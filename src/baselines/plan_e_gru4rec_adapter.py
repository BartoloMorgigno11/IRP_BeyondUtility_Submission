"""Plan E GRU4Rec sequence and candidate adapter utilities.

This module audits data contracts for a future GRU4Rec pilot. It does not
train a model, score project-data candidates, import torch at module import
time, or create recommendation outputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PLAN_E2_CANDIDATE_COUNT = 20
DEFAULT_MAX_SEQUENCE_LENGTH = 50
GRU4REC_PADDING_INDEX = 0
GRU4REC_REAL_ITEM_OFFSET = 1
PLAN_E2_CANDIDATE_SET_IDS = {
    "plan_e_beauty_1000user_ranking_manifest",
    "plan_e_beauty_1000user_reranking_manifest",
    "plan_e_sports_1000user_ranking_manifest",
    "plan_e_sports_1000user_reranking_manifest",
    "plan_e_ml1m_1000user_ranking_manifest",
    "plan_e_ml1m_1000user_reranking_manifest",
}


class GRU4RecAdapterError(ValueError):
    """Base exception for GRU4Rec adapter validation failures."""


class MappingValidationError(GRU4RecAdapterError):
    """Raised when an explicit ID mapping is invalid or ambiguous."""


class CandidateValidationError(GRU4RecAdapterError):
    """Raised when a candidate row cannot be safely adapted."""


class ChronologyValidationError(GRU4RecAdapterError):
    """Raised when chronological sequence order cannot be verified."""


class PaddingValidationError(GRU4RecAdapterError):
    """Raised when GRU4Rec padding or token semantics are invalid."""


class ScoreAlignmentError(GRU4RecAdapterError):
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
    """Summary of an explicit source mapping's internal-ID space."""

    minimum_internal_id: int | None
    maximum_internal_id: int | None
    contiguous_index_space: bool
    missing_internal_index_count: int
    index_zero_semantics: str


@dataclass(frozen=True)
class ChronologyAudit:
    """Chronology status for sorted training interactions."""

    timestamp_field: str
    timestamp_status: str
    chronology_source: str
    equal_timestamp_tie_rule: str
    chronology_ambiguous_user_count: int
    ambiguous_user_ids: frozenset[str]


@dataclass(frozen=True)
class CandidateTensorPayload:
    """Future GRU4Rec tensor payload with raw candidate order preserved."""

    user_id: str
    raw_history_item_ids: tuple[str, ...]
    raw_candidate_item_ids: tuple[str, ...]
    sequence_tokens: tuple[int, ...]
    candidate_item_tokens: tuple[int, ...]
    sequence_length: int


def safe_raw_id(value: Any) -> str:
    """Return a repository ID as a stripped string without numeric coercion."""

    if value is None:
        raise MappingValidationError("raw ID is missing")
    text = str(value).strip()
    if not text:
        raise MappingValidationError("raw ID is empty")
    if text.lower() in {"nan", "none", "null", "<na>"}:
        raise MappingValidationError(f"raw ID has null-like value: {text!r}")
    return text


def parse_internal_id(value: Any) -> int:
    """Parse a verified source internal ID as a non-negative integer."""

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
    duplicate_raw: list[str] = []
    duplicate_internal: list[int] = []
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
        source_path=str(source_path).replace("\\", "/"),
    )


def load_raw_to_internal_mapping(datamap_path: str | Path, *, entity_type: str) -> EntityMapping:
    """Load a raw-to-internal user or item mapping from a Plan C datamap."""

    path = Path(datamap_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "user2id" if entity_type == "user" else "item2id"
    inverse_key = "id2user" if entity_type == "user" else "id2item"
    if validate_mapping_direction(key) != "raw_to_internal":
        raise MappingValidationError(f"{key} is not raw-to-internal")
    if key not in payload or inverse_key not in payload:
        raise MappingValidationError(f"datamap lacks {key} or {inverse_key}: {path}")
    mapping = validate_raw_to_internal_pairs(
        payload[key].items(),
        entity_type=entity_type,
        source_path=str(path).replace("\\", "/"),
    )
    inverse = {parse_internal_id(key_value): safe_raw_id(raw_value) for key_value, raw_value in payload[inverse_key].items()}
    if mapping.internal_to_raw != inverse:
        raise MappingValidationError(f"{key} and {inverse_key} are not exact inverses: {path}")
    return mapping


def audit_index_space(mapping: EntityMapping) -> IndexSpaceAudit:
    """Audit contiguity and zero semantics for source internal IDs."""

    ids = sorted(mapping.internal_to_raw)
    if not ids:
        return IndexSpaceAudit(None, None, False, 0, "empty_mapping")
    minimum = ids[0]
    maximum = ids[-1]
    missing = set(range(minimum, maximum + 1)).difference(ids)
    zero_semantics = (
        "valid_source_entity; GRU4Rec model token zero remains reserved for padding after adapter remapping"
        if 0 in ids
        else "absent_from_source_mapping; GRU4Rec model token zero is reserved for padding"
    )
    return IndexSpaceAudit(
        minimum_internal_id=minimum,
        maximum_internal_id=maximum,
        contiguous_index_space=not missing,
        missing_internal_index_count=len(missing),
        index_zero_semantics=zero_semantics,
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


def sort_training_interactions(
    train_interactions: pd.DataFrame,
    *,
    timestamp_field: str = "timestamp",
    require_timestamp: bool = True,
) -> tuple[pd.DataFrame, ChronologyAudit]:
    """Sort train interactions by user, timestamp and original row-order tie key."""

    required = {"user_id", "item_id"}
    missing = required.difference(train_interactions.columns)
    if missing:
        raise ChronologyValidationError(f"training interactions missing columns: {sorted(missing)}")
    if timestamp_field not in train_interactions.columns:
        if require_timestamp:
            raise ChronologyValidationError(f"missing timestamp field: {timestamp_field}")
        frame = train_interactions.copy()
        frame["_input_order"] = range(len(frame))
        frame = frame.sort_values(["user_id", "_input_order"], kind="mergesort").drop(columns=["_input_order"])
        return frame.reset_index(drop=True), ChronologyAudit(
            timestamp_field=timestamp_field,
            timestamp_status="missing_not_required",
            chronology_source="repository_row_order",
            equal_timestamp_tie_rule="input_order_after_repository_csv_read",
            chronology_ambiguous_user_count=0,
            ambiguous_user_ids=frozenset(),
        )

    frame = train_interactions.copy()
    for column in ("user_id", "item_id", timestamp_field):
        frame[column] = frame[column].astype("string").str.strip()
    timestamp_text = frame[timestamp_field]
    missing_timestamp = timestamp_text.str.lower().isin({"", "nan", "none", "null", "<na>"})
    if require_timestamp and bool(missing_timestamp.any()):
        raise ChronologyValidationError("timestamp field contains missing values")
    numeric_timestamp = pd.to_numeric(timestamp_text, errors="coerce")
    invalid = numeric_timestamp.isna() & ~missing_timestamp
    if require_timestamp and bool(invalid.any()):
        raise ChronologyValidationError("timestamp field contains non-numeric values")
    if bool(invalid.any()):
        numeric_timestamp = numeric_timestamp.fillna(0)
    frame["_timestamp_numeric"] = numeric_timestamp
    frame["_input_order"] = range(len(frame))
    duplicate_mask = frame.duplicated(["user_id", "_timestamp_numeric"], keep=False)
    ambiguous_users = frozenset(safe_raw_id(value) for value in frame.loc[duplicate_mask, "user_id"].unique())
    sorted_frame = (
        frame.sort_values(["user_id", "_timestamp_numeric", "_input_order"], kind="mergesort")
        .drop(columns=["_timestamp_numeric", "_input_order"])
        .reset_index(drop=True)
    )
    return sorted_frame, ChronologyAudit(
        timestamp_field=timestamp_field,
        timestamp_status="present_numeric",
        chronology_source="timestamp_field_then_input_order_tie_break",
        equal_timestamp_tie_rule="input_order_after_repository_csv_read",
        chronology_ambiguous_user_count=len(ambiguous_users),
        ambiguous_user_ids=ambiguous_users,
    )


def build_training_histories(
    interactions: pd.DataFrame,
    *,
    timestamp_field: str = "timestamp",
    require_timestamp: bool = True,
) -> tuple[dict[str, tuple[str, ...]], ChronologyAudit, pd.DataFrame]:
    """Build per-user chronological training-only item histories."""

    train = filter_train_interactions(interactions)
    ordered, chronology = sort_training_interactions(
        train,
        timestamp_field=timestamp_field,
        require_timestamp=require_timestamp,
    )
    histories: dict[str, list[str]] = {}
    for row in ordered.itertuples(index=False):
        user_id = safe_raw_id(getattr(row, "user_id"))
        item_id = safe_raw_id(getattr(row, "item_id"))
        histories.setdefault(user_id, []).append(item_id)
    return {user_id: tuple(items) for user_id, items in histories.items()}, chronology, ordered


def make_prefix_target_pairs(histories: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    """Create GRU4Rec prefix to next-item training pairs for audit counts."""

    pairs: list[dict[str, Any]] = []
    for user_id in sorted(histories, key=natural_id_sort_key):
        sequence = [safe_raw_id(item) for item in histories[user_id]]
        for position in range(1, len(sequence)):
            pairs.append(
                {
                    "user_id": safe_raw_id(user_id),
                    "prefix_item_ids": tuple(sequence[:position]),
                    "target_item_id": sequence[position],
                    "prefix_length": position,
                }
            )
    return pairs


def apply_max_history_length(history_item_ids: Sequence[Any], *, max_sequence_length: int) -> tuple[str, ...]:
    """Keep the most recent items when a history exceeds the model length."""

    if max_sequence_length < 1:
        raise PaddingValidationError("max_sequence_length must be positive")
    history = tuple(safe_raw_id(item) for item in history_item_ids)
    return history[-max_sequence_length:]


def natural_id_sort_key(value: Any) -> tuple[int, int, str]:
    """Sort numeric-looking IDs numerically and other IDs lexicographically."""

    text = str(value)
    try:
        return (0, int(text), "")
    except ValueError:
        return (1, 0, text)


def build_gru4rec_model_index(train_seen_ids: Iterable[Any]) -> dict[str, int]:
    """Build the zero-based dense raw-ID index used before GRU4Rec token offset."""

    ordered = sorted({safe_raw_id(value) for value in train_seen_ids}, key=natural_id_sort_key)
    return {raw_id: index for index, raw_id in enumerate(ordered)}


def validate_padding_index_semantics(
    item_model_index: Mapping[str, int],
    *,
    padding_index: int = GRU4REC_PADDING_INDEX,
    real_item_offset: int = GRU4REC_REAL_ITEM_OFFSET,
) -> None:
    """Validate that real GRU4Rec item tokens cannot equal padding."""

    if padding_index != 0:
        raise PaddingValidationError("Plan D GRU4Rec uses padding index 0")
    if real_item_offset != 1:
        raise PaddingValidationError("Plan D GRU4Rec uses item index plus one for real tokens")
    tokens = [int(index) + real_item_offset for index in item_model_index.values()]
    if any(token == padding_index for token in tokens):
        raise PaddingValidationError("a real item token collides with padding index 0")
    if any(token < 1 for token in tokens):
        raise PaddingValidationError("real item tokens must be positive")


def ensure_gru4rec_tokens_within_cardinality(tokens: Iterable[int], *, model_cardinality: int) -> None:
    """Validate GRU4Rec tokens against embedding/output cardinality."""

    if model_cardinality < 1:
        raise PaddingValidationError("model_cardinality must be positive")
    bad = [int(token) for token in tokens if int(token) < 0 or int(token) >= model_cardinality]
    if bad:
        raise PaddingValidationError(
            f"GRU4Rec tokens out of range for cardinality {model_cardinality}: {bad[:5]}"
        )


def pad_history_left(
    history_item_ids: Sequence[Any],
    item_model_index: Mapping[str, int],
    *,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> tuple[int, ...]:
    """Left-pad raw item IDs as one-based GRU4Rec tokens."""

    clipped = apply_max_history_length(history_item_ids, max_sequence_length=max_sequence_length)
    missing = [item for item in clipped if item not in item_model_index]
    if missing:
        raise CandidateValidationError(f"history item is not in GRU4Rec model index: {missing[0]}")
    tokens = [int(item_model_index[item]) + GRU4REC_REAL_ITEM_OFFSET for item in clipped]
    output = [GRU4REC_PADDING_INDEX] * max_sequence_length
    if tokens:
        output[-len(tokens) :] = tokens
    return tuple(output)


def sequence_length_from_tokens(sequence_tokens: Sequence[int]) -> int:
    """Return the non-padding length for a GRU4Rec token sequence."""

    return sum(int(token) != GRU4REC_PADDING_INDEX for token in sequence_tokens)


def padding_mask_from_tokens(sequence_tokens: Sequence[int]) -> tuple[bool, ...]:
    """Return True for real item tokens and False for padding tokens."""

    return tuple(int(token) != GRU4REC_PADDING_INDEX for token in sequence_tokens)


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


def prepare_candidate_tensor_payload(
    *,
    user_id: Any,
    history_item_ids: Sequence[Any],
    candidate_item_ids: Sequence[Any],
    item_model_index: Mapping[str, int],
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    expected_candidate_count: int | None = PLAN_E2_CANDIDATE_COUNT,
) -> CandidateTensorPayload:
    """Prepare future GRU4Rec token tensors while preserving raw candidate order."""

    raw_user_id = safe_raw_id(user_id)
    raw_history = tuple(safe_raw_id(item) for item in history_item_ids)
    raw_candidates = tuple(safe_raw_id(item) for item in candidate_item_ids)
    if expected_candidate_count is not None and len(raw_candidates) != expected_candidate_count:
        raise CandidateValidationError(
            f"expected {expected_candidate_count} candidate items, found {len(raw_candidates)}"
        )
    missing = [item for item in raw_candidates if item not in item_model_index]
    if missing:
        raise CandidateValidationError(f"candidate item is not in GRU4Rec model index: {missing[0]}")
    sequence_tokens = pad_history_left(raw_history, item_model_index, max_sequence_length=max_sequence_length)
    candidate_tokens = tuple(int(item_model_index[item]) + GRU4REC_REAL_ITEM_OFFSET for item in raw_candidates)
    if any(token == GRU4REC_PADDING_INDEX for token in candidate_tokens):
        raise PaddingValidationError("candidate token collides with padding index 0")
    return CandidateTensorPayload(
        user_id=raw_user_id,
        raw_history_item_ids=raw_history,
        raw_candidate_item_ids=raw_candidates,
        sequence_tokens=sequence_tokens,
        candidate_item_tokens=candidate_tokens,
        sequence_length=sequence_length_from_tokens(sequence_tokens),
    )


def audit_plan_e2_gru4rec_candidate_row(
    row: Mapping[str, Any],
    *,
    user_mapping: EntityMapping,
    item_mapping: EntityMapping,
    train_histories: Mapping[str, Sequence[str]],
    train_seen_items: set[str],
    item_model_index: Mapping[str, int],
    eligible_full: bool,
    eligible_matched: bool,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    chronology_verified: bool = True,
    temporal_protocol_unverifiable: bool = False,
) -> dict[str, Any]:
    """Audit one Plan E2 row for future GRU4Rec sequential compatibility."""

    user_id = safe_raw_id(row["user_id"])
    positive_item_id = safe_raw_id(row["positive_item_id"])
    candidate_items = parse_candidate_items(row["candidate_items"])
    history = tuple(safe_raw_id(item) for item in train_histories.get(user_id, ()))
    raw_history_length = len(history)
    effective_history = apply_max_history_length(history, max_sequence_length=max_sequence_length)

    user_mapped = user_id in user_mapping.raw_to_internal
    positive_item_mapped = positive_item_id in item_mapping.raw_to_internal
    unmapped_candidates = [item for item in candidate_items if item not in item_mapping.raw_to_internal]
    train_unseen_candidates = [item for item in candidate_items if item not in train_seen_items]
    positive_seen = positive_item_id in train_seen_items
    history_items_mapped = all(item in item_mapping.raw_to_internal for item in history)
    history_items_in_index = all(item in item_model_index for item in history)
    history_available = raw_history_length >= 1
    training_pair_available = raw_history_length >= 2

    padding_index_compatible = False
    if history_available and history_items_mapped and history_items_in_index:
        try:
            validate_padding_index_semantics(item_model_index)
            _ = pad_history_left(effective_history, item_model_index, max_sequence_length=max_sequence_length)
            padding_index_compatible = True
        except (CandidateValidationError, PaddingValidationError):
            padding_index_compatible = False

    sequence_mapped = (
        user_mapped
        and positive_item_mapped
        and not unmapped_candidates
        and history_items_mapped
        and history_items_in_index
        and padding_index_compatible
    )
    strict_candidate_train_known = positive_seen and not train_unseen_candidates
    sequentially_compatible = (
        sequence_mapped
        and history_available
        and training_pair_available
        and strict_candidate_train_known
        and chronology_verified
        and not temporal_protocol_unverifiable
    )
    reasons = gru4rec_exclusion_reasons(
        user_mapped=user_mapped,
        positive_item_mapped=positive_item_mapped,
        unmapped_candidate_count=len(unmapped_candidates),
        history_available=history_available,
        training_pair_available=training_pair_available,
        positive_item_seen=positive_seen,
        train_unseen_candidate_count=len(train_unseen_candidates),
        padding_index_compatible=padding_index_compatible,
        chronology_verified=chronology_verified,
        temporal_protocol_unverifiable=temporal_protocol_unverifiable,
    )
    if sequentially_compatible:
        status = "sequentially_compatible"
    elif not user_mapped or not positive_item_mapped or unmapped_candidates:
        status = "mapping_incompatible"
    elif not history_available:
        status = "history_unavailable"
    elif not training_pair_available:
        status = "insufficient_training_history"
    elif not padding_index_compatible:
        status = "padding_or_index_incompatible"
    elif temporal_protocol_unverifiable or not chronology_verified:
        status = "temporal_protocol_unverifiable"
    elif not strict_candidate_train_known:
        status = "candidate_train_unseen"
    else:
        status = "validation_failed"

    return {
        "candidate_set_id": str(row["candidate_set_id"]),
        "source_path": str(row["source_path"]),
        "dataset": str(row["dataset"]),
        "task_type": str(row.get("task_type", "")),
        "user_id": user_id,
        "positive_item_id": positive_item_id,
        "candidate_count": len(candidate_items),
        "user_mapped": user_mapped,
        "history_available": history_available,
        "training_pair_available": training_pair_available,
        "raw_train_history_length": raw_history_length,
        "effective_history_length": len(effective_history),
        "history_truncated": raw_history_length > max_sequence_length,
        "positive_item_mapped": positive_item_mapped,
        "positive_item_seen_in_train": positive_seen,
        "all_candidate_items_mapped": not unmapped_candidates,
        "all_candidate_items_seen_in_train": not train_unseen_candidates,
        "unmapped_candidate_item_count": len(unmapped_candidates),
        "unmapped_candidate_item_ids": json_array(unmapped_candidates),
        "train_unseen_candidate_item_count": len(train_unseen_candidates),
        "train_unseen_candidate_item_ids": json_array(train_unseen_candidates),
        "padding_index_compatible": padding_index_compatible,
        "chronology_verified": chronology_verified,
        "temporal_protocol_unverifiable": temporal_protocol_unverifiable,
        "sequentially_compatible": sequentially_compatible,
        "eligible_plan_e2_full": eligible_full,
        "eligible_plan_e2_matched": eligible_matched,
        "compatibility_status": status,
        "exclusion_reasons": ";".join(reasons) if reasons else "none",
        "notes": "candidate_order_preserved; no project-data scoring",
    }


def gru4rec_exclusion_reasons(
    *,
    user_mapped: bool,
    positive_item_mapped: bool,
    unmapped_candidate_count: int,
    history_available: bool,
    training_pair_available: bool,
    positive_item_seen: bool,
    train_unseen_candidate_count: int,
    padding_index_compatible: bool,
    chronology_verified: bool,
    temporal_protocol_unverifiable: bool,
) -> list[str]:
    """Return deterministic sequential-compatibility exclusion reasons."""

    reasons: list[str] = []
    if not user_mapped:
        reasons.append("user_unmapped")
    if not positive_item_mapped:
        reasons.append("positive_item_unmapped")
    if unmapped_candidate_count:
        reasons.append("candidate_item_unmapped")
    if not history_available:
        reasons.append("history_unavailable")
    if history_available and not training_pair_available:
        reasons.append("insufficient_training_history")
    if positive_item_mapped and not positive_item_seen:
        reasons.append("positive_item_not_seen_in_train")
    if train_unseen_candidate_count:
        reasons.append("candidate_item_not_seen_in_train")
    if not padding_index_compatible:
        reasons.append("padding_or_index_incompatible")
    if not chronology_verified or temporal_protocol_unverifiable:
        reasons.append("temporal_protocol_unverifiable")
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


def history_bucket(length: int) -> str:
    """Return the deterministic report bucket for a training history length."""

    if length <= 0:
        return "rows_history_length_0"
    if length == 1:
        return "rows_history_length_1"
    if 2 <= length <= 4:
        return "rows_history_length_2_to_4"
    if 5 <= length <= 9:
        return "rows_history_length_5_to_9"
    if 10 <= length <= 19:
        return "rows_history_length_10_to_19"
    return "rows_history_length_20_or_more"


def json_array(values: Iterable[Any]) -> str:
    """Serialize a deterministic JSON array for CSV output."""

    return json.dumps([safe_raw_id(value) for value in values], ensure_ascii=True)


def bool_text(value: bool) -> str:
    """Serialize booleans consistently for CSV output."""

    return "true" if bool(value) else "false"

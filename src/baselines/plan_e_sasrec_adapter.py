"""Plan E SASRec adapter and transformer-sequence audit utilities.

This module validates data contracts for a future SASRec pilot. It does not
train a model, score project-data candidates, write outputs, or create
checkpoints.
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
SASREC_PADDING_INDEX = 0
SASREC_REAL_ITEM_OFFSET = 1
PLAN_E2_CANDIDATE_SET_IDS = {
    "plan_e_beauty_1000user_ranking_manifest",
    "plan_e_beauty_1000user_reranking_manifest",
    "plan_e_sports_1000user_ranking_manifest",
    "plan_e_sports_1000user_reranking_manifest",
    "plan_e_ml1m_1000user_ranking_manifest",
    "plan_e_ml1m_1000user_reranking_manifest",
}


class SASRecAdapterError(ValueError):
    """Base exception for Plan E SASRec adapter validation failures."""


class MappingValidationError(SASRecAdapterError):
    """Raised when an explicit ID mapping is invalid or ambiguous."""


class CandidateValidationError(SASRecAdapterError):
    """Raised when a candidate row cannot be safely adapted."""


class ChronologyValidationError(SASRecAdapterError):
    """Raised when chronological sequence order cannot be verified."""


class PaddingValidationError(SASRecAdapterError):
    """Raised when padding, token, or positional-index semantics are invalid."""


class MaskValidationError(SASRecAdapterError):
    """Raised when causal or padding masks do not match the audited contract."""


class TargetLeakageError(SASRecAdapterError):
    """Raised when a target appears in its own input prefix."""


class ScoreAlignmentError(SASRecAdapterError):
    """Raised when future score vectors cannot be aligned to candidates."""


@dataclass(frozen=True)
class EntityMapping:
    """Validated dataset-scoped raw-to-internal mapping."""

    entity_type: str
    raw_to_internal: dict[str, int]
    internal_to_raw: dict[int, str]
    source_path: str
    mapping_direction: str = "raw_to_internal"


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
    malformed_timestamp_count: int
    equal_timestamp_group_count: int
    affected_equal_timestamp_user_count: int
    chronology_ambiguous_user_count: int
    ambiguous_user_ids: tuple[str, ...]


@dataclass(frozen=True)
class SequenceAudit:
    """Train-only sequence summary for one dataset."""

    histories: dict[str, tuple[str, ...]]
    ordered_train: pd.DataFrame
    chronology: ChronologyAudit
    total_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    train_user_count: int
    train_item_count: int
    users_with_one_train_event: int
    users_with_two_or_more_train_events: int
    users_with_valid_objective: int
    training_example_count: int
    min_history_length: int
    max_history_length: int
    mean_history_length: float
    median_history_length: float
    histories_requiring_truncation: int
    duplicate_user_item_event_count: int
    repeated_item_event_count: int
    target_positions_generated: int
    padding_target_count: int
    target_leakage_count: int


@dataclass(frozen=True)
class CandidateTensorPayload:
    """Future SASRec tensor payload with candidate order preserved."""

    user_id: str
    raw_history_item_ids: tuple[str, ...]
    raw_candidate_item_ids: tuple[str, ...]
    sequence_tokens: tuple[int, ...]
    position_indices: tuple[int, ...]
    padding_key_mask: tuple[bool, ...]
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


def natural_id_sort_key(value: Any) -> tuple[int, int, str]:
    """Sort numeric-looking IDs numerically and other IDs lexicographically."""

    text = str(value)
    try:
        return (0, int(text), "")
    except ValueError:
        return (1, 0, text)


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
    inverse = {parse_internal_id(k): safe_raw_id(v) for k, v in payload[inverse_key].items()}
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
        "valid_source_entity; SASRec adapter still reserves token zero for padding"
        if 0 in ids
        else "absent_from_source_mapping; SASRec token zero is reserved for padding"
    )
    return IndexSpaceAudit(minimum, maximum, not missing, len(missing), zero_semantics)


def read_csv_strings(path: str | Path) -> pd.DataFrame:
    """Read a CSV while preserving IDs as strings."""

    return pd.read_csv(path, dtype="string", keep_default_na=False)


def freeze_source_row_index(interactions: pd.DataFrame) -> pd.DataFrame:
    """Add a stable source-row index before filtering or sorting."""

    frame = interactions.copy().reset_index(drop=True)
    frame["source_row_index"] = range(len(frame))
    return frame


def filter_train_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Return only rows whose split is exactly train after safe normalization."""

    required = {"user_id", "item_id", "split"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ChronologyValidationError(f"interactions missing columns: {sorted(missing)}")
    frame = freeze_source_row_index(interactions) if "source_row_index" not in interactions.columns else interactions.copy()
    for column in ("user_id", "item_id", "split"):
        frame[column] = frame[column].astype("string").str.strip()
    return frame.loc[frame["split"].str.lower() == "train"].copy()


def validate_numeric_timestamps(frame: pd.DataFrame, *, timestamp_field: str = "timestamp") -> pd.Series:
    """Return numeric timestamps or raise for missing/malformed values."""

    if timestamp_field not in frame.columns:
        raise ChronologyValidationError(f"missing timestamp field: {timestamp_field}")
    text = frame[timestamp_field].astype("string").str.strip()
    missing = text.str.lower().isin({"", "nan", "none", "null", "<na>"})
    if bool(missing.any()):
        raise ChronologyValidationError("timestamp field contains missing values")
    numeric = pd.to_numeric(text, errors="coerce")
    if bool(numeric.isna().any()):
        raise ChronologyValidationError("timestamp field contains non-numeric values")
    return numeric


def sort_chronological_train_interactions(
    train_interactions: pd.DataFrame,
    *,
    timestamp_field: str = "timestamp",
) -> tuple[pd.DataFrame, ChronologyAudit]:
    """Sort train interactions by user, timestamp, and frozen source row index."""

    required = {"user_id", "item_id", "source_row_index"}
    missing = required.difference(train_interactions.columns)
    if missing:
        raise ChronologyValidationError(f"training interactions missing columns: {sorted(missing)}")
    frame = train_interactions.copy()
    for column in ("user_id", "item_id"):
        frame[column] = frame[column].astype("string").str.strip()
    numeric = validate_numeric_timestamps(frame, timestamp_field=timestamp_field)
    frame["_timestamp_numeric"] = numeric
    equal_group_sizes = frame.groupby(["user_id", "_timestamp_numeric"], sort=False).size()
    equal_group_count = int((equal_group_sizes > 1).sum())
    equal_users = set()
    if equal_group_count:
        dup = frame.duplicated(["user_id", "_timestamp_numeric"], keep=False)
        equal_users = {safe_raw_id(value) for value in frame.loc[dup, "user_id"].unique()}
    sorted_frame = (
        frame.sort_values(["user_id", "_timestamp_numeric", "source_row_index"], kind="mergesort")
        .drop(columns=["_timestamp_numeric"])
        .reset_index(drop=True)
    )
    return sorted_frame, ChronologyAudit(
        timestamp_field=timestamp_field,
        timestamp_status="present_numeric",
        chronology_source="timestamp_field_then_frozen_source_row_index",
        equal_timestamp_tie_rule="frozen_persisted_source_row_order",
        malformed_timestamp_count=0,
        equal_timestamp_group_count=equal_group_count,
        affected_equal_timestamp_user_count=len(equal_users),
        chronology_ambiguous_user_count=len(equal_users),
        ambiguous_user_ids=tuple(sorted(equal_users, key=natural_id_sort_key)),
    )


def build_training_histories(
    interactions: pd.DataFrame,
    *,
    timestamp_field: str = "timestamp",
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> SequenceAudit:
    """Build per-user chronological training-only histories and objective counts."""

    frozen = freeze_source_row_index(interactions)
    split = frozen["split"].astype("string").str.strip().str.lower() if "split" in frozen.columns else pd.Series([], dtype="string")
    train = filter_train_interactions(frozen)
    ordered, chronology = sort_chronological_train_interactions(train, timestamp_field=timestamp_field)
    histories: dict[str, list[str]] = {}
    for row in ordered.itertuples(index=False):
        histories.setdefault(safe_raw_id(getattr(row, "user_id")), []).append(safe_raw_id(getattr(row, "item_id")))
    final_histories = {user: tuple(items) for user, items in histories.items()}
    lengths = [len(items) for items in final_histories.values()]
    duplicate_events = int(ordered.duplicated(["user_id", "item_id", "timestamp"]).sum()) if "timestamp" in ordered.columns else 0
    repeated = int(
        sum(max(0, len(items) - len(set(items))) for items in final_histories.values())
    )
    target_leakage = count_target_leakage_in_prefixes(final_histories)
    return SequenceAudit(
        histories=final_histories,
        ordered_train=ordered,
        chronology=chronology,
        total_rows=int(len(frozen)),
        train_rows=int((split == "train").sum()) if len(split) else int(len(train)),
        validation_rows=int(split.isin({"validation", "valid", "val"}).sum()) if len(split) else 0,
        test_rows=int((split == "test").sum()) if len(split) else 0,
        train_user_count=len(final_histories),
        train_item_count=len({item for items in final_histories.values() for item in items}),
        users_with_one_train_event=sum(1 for length in lengths if length == 1),
        users_with_two_or_more_train_events=sum(1 for length in lengths if length >= 2),
        users_with_valid_objective=sum(1 for length in lengths if length >= 2),
        training_example_count=sum(max(0, length - 1) for length in lengths),
        min_history_length=min(lengths) if lengths else 0,
        max_history_length=max(lengths) if lengths else 0,
        mean_history_length=float(sum(lengths) / len(lengths)) if lengths else 0.0,
        median_history_length=float(pd.Series(lengths).median()) if lengths else 0.0,
        histories_requiring_truncation=sum(1 for length in lengths if length > max_sequence_length),
        duplicate_user_item_event_count=duplicate_events,
        repeated_item_event_count=repeated,
        target_positions_generated=sum(max(0, length - 1) for length in lengths),
        padding_target_count=0,
        target_leakage_count=target_leakage,
    )


def count_target_leakage_in_prefixes(histories: Mapping[str, Sequence[str]]) -> int:
    """Count synthetic prefix-target construction errors where target is in its own prefix position."""

    leakage = 0
    for sequence in histories.values():
        normalized = [safe_raw_id(item) for item in sequence]
        for position in range(1, len(normalized)):
            if normalized[position] == "":
                leakage += 1
    return leakage


def make_sasrec_training_examples(
    histories: Mapping[str, Sequence[str]],
    *,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> list[dict[str, Any]]:
    """Create audited prefix -> next-item training examples in Plan D order."""

    rows: list[dict[str, Any]] = []
    for user_id in sorted(histories, key=natural_id_sort_key):
        sequence = [safe_raw_id(item) for item in histories[user_id]]
        for position in range(1, len(sequence)):
            prefix = tuple(sequence[:position])
            target = sequence[position]
            if not prefix:
                raise TargetLeakageError("SASRec target has empty prefix")
            rows.append(
                {
                    "user_id": safe_raw_id(user_id),
                    "prefix_item_ids": prefix,
                    "target_item_id": target,
                    "prefix_length": len(prefix),
                    "effective_prefix_length": min(len(prefix), max_sequence_length),
                    "input_contains_target_at_own_position": False,
                }
            )
    return rows


def apply_max_history_length(history_item_ids: Sequence[Any], *, max_sequence_length: int) -> tuple[str, ...]:
    """Keep the most recent items when a history exceeds the model length."""

    if max_sequence_length < 1:
        raise PaddingValidationError("max_sequence_length must be positive")
    history = tuple(safe_raw_id(item) for item in history_item_ids)
    return history[-max_sequence_length:]


def build_sasrec_token_index(train_seen_ids: Iterable[Any]) -> dict[str, int]:
    """Build the zero-based dense item index before the SASRec +1 token offset."""

    ordered = sorted({safe_raw_id(value) for value in train_seen_ids}, key=natural_id_sort_key)
    return {raw_id: index for index, raw_id in enumerate(ordered)}


def validate_padding_index_semantics(
    item_token_index: Mapping[str, int],
    *,
    padding_index: int = SASREC_PADDING_INDEX,
    real_item_offset: int = SASREC_REAL_ITEM_OFFSET,
) -> None:
    """Validate that real SASRec item tokens cannot equal padding."""

    if padding_index != 0:
        raise PaddingValidationError("Plan D SASRec uses padding index 0")
    if real_item_offset != 1:
        raise PaddingValidationError("Plan D SASRec uses item index plus one for real tokens")
    tokens = [int(index) + real_item_offset for index in item_token_index.values()]
    if any(token == padding_index for token in tokens):
        raise PaddingValidationError("a real item token collides with padding index 0")
    if any(token < 1 for token in tokens):
        raise PaddingValidationError("real item tokens must be positive")


def ensure_sasrec_tokens_within_cardinality(tokens: Iterable[int], *, embedding_cardinality: int) -> None:
    """Validate SASRec tokens against embedding/output cardinality."""

    if embedding_cardinality < 1:
        raise PaddingValidationError("embedding_cardinality must be positive")
    bad = [int(token) for token in tokens if int(token) < 0 or int(token) >= embedding_cardinality]
    if bad:
        raise PaddingValidationError(f"SASRec tokens out of range for cardinality {embedding_cardinality}: {bad[:5]}")


def pad_history_left(
    history_item_ids: Sequence[Any],
    item_token_index: Mapping[str, int],
    *,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> tuple[int, ...]:
    """Left-pad raw item IDs as one-based SASRec tokens."""

    clipped = apply_max_history_length(history_item_ids, max_sequence_length=max_sequence_length)
    missing = [item for item in clipped if item not in item_token_index]
    if missing:
        raise CandidateValidationError(f"history item is not in SASRec token index: {missing[0]}")
    tokens = [int(item_token_index[item]) + SASREC_REAL_ITEM_OFFSET for item in clipped]
    output = [SASREC_PADDING_INDEX] * max_sequence_length
    if tokens:
        output[-len(tokens) :] = tokens
    return tuple(output)


def position_indices_for_sequence(sequence_tokens: Sequence[int], *, max_sequence_length: int) -> tuple[int, ...]:
    """Return Plan D absolute position indices for the full padded tensor."""

    if len(sequence_tokens) > max_sequence_length:
        raise PaddingValidationError("sequence length exceeds configured max_sequence_length")
    positions = tuple(range(len(sequence_tokens)))
    if positions and (min(positions) < 0 or max(positions) >= max_sequence_length):
        raise PaddingValidationError("position index out of bounds")
    return positions


def sequence_length_from_tokens(sequence_tokens: Sequence[int]) -> int:
    """Return non-padding token count."""

    return sum(int(token) != SASREC_PADDING_INDEX for token in sequence_tokens)


def padding_key_mask_from_tokens(sequence_tokens: Sequence[int]) -> tuple[bool, ...]:
    """Return Plan D src_key_padding_mask semantics: True means blocked padding key."""

    return tuple(int(token) == SASREC_PADDING_INDEX for token in sequence_tokens)


def causal_attention_mask(seq_len: int) -> tuple[tuple[bool, ...], ...]:
    """Return a boolean upper-triangular mask where True blocks future positions."""

    if seq_len < 1:
        raise MaskValidationError("seq_len must be positive")
    return tuple(tuple(col > row for col in range(seq_len)) for row in range(seq_len))


def validate_causal_mask(mask: Sequence[Sequence[bool]]) -> None:
    """Validate Plan D causal-mask semantics."""

    if not mask:
        raise MaskValidationError("causal mask is empty")
    size = len(mask)
    if any(len(row) != size for row in mask):
        raise MaskValidationError("causal mask must be square")
    for row_index, row in enumerate(mask):
        for col_index, value in enumerate(row):
            expected = col_index > row_index
            if bool(value) != expected:
                raise MaskValidationError("causal mask does not block exactly future positions")


def validate_padding_mask(sequence_tokens: Sequence[int], padding_mask: Sequence[bool]) -> None:
    """Validate Plan D key-padding mask semantics."""

    if len(sequence_tokens) != len(padding_mask):
        raise MaskValidationError("padding mask length mismatch")
    expected = tuple(int(token) == SASREC_PADDING_INDEX for token in sequence_tokens)
    if tuple(bool(value) for value in padding_mask) != expected:
        raise MaskValidationError("padding mask must be true only for padding keys")


def final_non_padding_position(sequence_tokens: Sequence[int]) -> int:
    """Return the final non-padding tensor position used by Plan D SASRec."""

    real_positions = [index for index, token in enumerate(sequence_tokens) if int(token) != SASREC_PADDING_INDEX]
    if not real_positions:
        raise PaddingValidationError("fully padded SASRec rows are invalid")
    return max(real_positions)


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
    item_token_index: Mapping[str, int],
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    expected_candidate_count: int | None = PLAN_E2_CANDIDATE_COUNT,
) -> CandidateTensorPayload:
    """Prepare future SASRec tensors while preserving raw candidate order."""

    raw_user_id = safe_raw_id(user_id)
    raw_history = tuple(safe_raw_id(item) for item in history_item_ids)
    raw_candidates = tuple(safe_raw_id(item) for item in candidate_item_ids)
    if expected_candidate_count is not None and len(raw_candidates) != expected_candidate_count:
        raise CandidateValidationError(f"expected {expected_candidate_count} candidate items, found {len(raw_candidates)}")
    if len(set(raw_candidates)) != len(raw_candidates):
        raise CandidateValidationError("candidate items must be unique")
    missing = [item for item in raw_candidates if item not in item_token_index]
    if missing:
        raise CandidateValidationError(f"candidate item is not in SASRec token index: {missing[0]}")
    sequence_tokens = pad_history_left(raw_history, item_token_index, max_sequence_length=max_sequence_length)
    position_indices = position_indices_for_sequence(sequence_tokens, max_sequence_length=max_sequence_length)
    padding_mask = padding_key_mask_from_tokens(sequence_tokens)
    candidate_tokens = tuple(int(item_token_index[item]) + SASREC_REAL_ITEM_OFFSET for item in raw_candidates)
    if any(token == SASREC_PADDING_INDEX for token in candidate_tokens):
        raise PaddingValidationError("candidate token collides with padding index 0")
    return CandidateTensorPayload(
        user_id=raw_user_id,
        raw_history_item_ids=raw_history,
        raw_candidate_item_ids=raw_candidates,
        sequence_tokens=sequence_tokens,
        position_indices=position_indices,
        padding_key_mask=padding_mask,
        candidate_item_tokens=candidate_tokens,
        sequence_length=sequence_length_from_tokens(sequence_tokens),
    )


def validate_score_vector_alignment(candidate_item_ids: Sequence[Any], scores: Sequence[Any]) -> tuple[float, ...]:
    """Validate one finite score per candidate in original order."""

    candidates = tuple(safe_raw_id(item) for item in candidate_item_ids)
    if len(candidates) != len(scores):
        raise ScoreAlignmentError("candidate and score lengths differ")
    numeric_scores: list[float] = []
    for score in scores:
        value = float(score)
        if not math.isfinite(value):
            raise ScoreAlignmentError("candidate score is not finite")
        numeric_scores.append(value)
    return tuple(numeric_scores)


def audit_plan_e2_sasrec_candidate_row(
    row: Mapping[str, Any],
    *,
    user_mapping: EntityMapping,
    item_mapping: EntityMapping,
    train_histories: Mapping[str, Sequence[str]],
    train_seen_items: set[str],
    item_token_index: Mapping[str, int],
    eligible_full: bool,
    eligible_matched: bool,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    causal_mask_verified: bool = True,
    padding_mask_verified: bool = True,
    chronology_verified: bool = True,
    temporal_protocol_unverifiable: bool = False,
) -> dict[str, Any]:
    """Audit one Plan E2 row for future SASRec sequential compatibility."""

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
    history_available = raw_history_length >= 1
    training_objective_available = raw_history_length >= 2
    history_items_mapped = all(item in item_mapping.raw_to_internal for item in history)
    history_items_tokenized = all(item in item_token_index for item in effective_history)
    padding_index_compatible = True
    position_indices_valid = True
    target_leakage_detected = positive_item_id in set(history)
    try:
        validate_padding_index_semantics(item_token_index)
        if history_available and history_items_tokenized:
            payload = prepare_candidate_tensor_payload(
                user_id=user_id,
                history_item_ids=history,
                candidate_item_ids=[item for item in candidate_items if item in item_token_index],
                item_token_index=item_token_index,
                max_sequence_length=max_sequence_length,
                expected_candidate_count=None,
            )
            validate_padding_mask(payload.sequence_tokens, payload.padding_key_mask)
            final_non_padding_position(payload.sequence_tokens)
            position_indices_valid = max(payload.position_indices) < max_sequence_length
    except (SASRecAdapterError, ValueError):
        padding_index_compatible = False
        position_indices_valid = False
    all_candidate_items_mapped = not unmapped_candidates
    all_candidate_items_seen = not train_unseen_candidates
    sequence_mapped = (
        user_mapped
        and positive_item_mapped
        and all_candidate_items_mapped
        and history_items_mapped
    )
    compatible = (
        sequence_mapped
        and history_available
        and training_objective_available
        and positive_seen
        and all_candidate_items_seen
        and padding_index_compatible
        and position_indices_valid
        and causal_mask_verified
        and padding_mask_verified
        and chronology_verified
        and not temporal_protocol_unverifiable
        and not target_leakage_detected
    )
    reasons: list[str] = []
    if not sequence_mapped:
        reasons.append("mapping_incompatible")
    if not history_available:
        reasons.append("history_unavailable")
    if not training_objective_available:
        reasons.append("training_objective_unavailable")
    if not all_candidate_items_seen or not positive_seen:
        reasons.append("candidate_train_unseen")
    if not padding_index_compatible or not position_indices_valid:
        reasons.append("padding_or_position_incompatible")
    if not causal_mask_verified or not padding_mask_verified:
        reasons.append("causal_mask_unverified")
    if temporal_protocol_unverifiable or not chronology_verified:
        reasons.append("temporal_protocol_unverifiable")
    if target_leakage_detected:
        reasons.append("target_leakage_detected")
    status = compatibility_status_from_reasons(reasons)
    return {
        "candidate_set_id": safe_raw_id(row["candidate_set_id"]),
        "source_path": str(row["source_path"]).replace("\\", "/"),
        "dataset": safe_raw_id(row["dataset"]),
        "task_type": safe_raw_id(row.get("task_type", "ranking")),
        "user_id": user_id,
        "positive_item_id": positive_item_id,
        "candidate_count": len(candidate_items),
        "user_mapped": user_mapped,
        "history_available": history_available,
        "training_objective_available": training_objective_available,
        "raw_train_history_length": raw_history_length,
        "effective_history_length": len(effective_history),
        "history_truncated": raw_history_length > max_sequence_length,
        "positive_item_mapped": positive_item_mapped,
        "positive_item_seen_in_train": positive_seen,
        "all_candidate_items_mapped": all_candidate_items_mapped,
        "all_candidate_items_seen_in_train": all_candidate_items_seen,
        "unmapped_candidate_item_count": len(unmapped_candidates),
        "unmapped_candidate_item_ids": json_array(unmapped_candidates),
        "train_unseen_candidate_item_count": len(train_unseen_candidates),
        "train_unseen_candidate_item_ids": json_array(train_unseen_candidates),
        "padding_index_compatible": padding_index_compatible,
        "position_indices_valid": position_indices_valid,
        "causal_mask_verified": causal_mask_verified,
        "padding_mask_verified": padding_mask_verified,
        "chronology_verified": chronology_verified,
        "temporal_protocol_unverifiable": temporal_protocol_unverifiable,
        "target_leakage_detected": target_leakage_detected,
        "sasrec_sequentially_compatible": compatible,
        "eligible_plan_e2_full": eligible_full,
        "eligible_plan_e2_matched": eligible_matched,
        "compatibility_status": status,
        "exclusion_reasons": "none" if compatible else ";".join(reasons),
        "notes": "SASRec Plan E2 preflight only; no project-data scoring performed",
    }


def compatibility_status_from_reasons(reasons: Sequence[str]) -> str:
    """Return the primary compatibility status for ordered exclusion reasons."""

    if not reasons:
        return "sasrec_sequentially_compatible"
    priority = [
        "target_leakage_detected",
        "temporal_protocol_unverifiable",
        "causal_mask_unverified",
        "padding_or_position_incompatible",
        "mapping_incompatible",
        "history_unavailable",
        "training_objective_unavailable",
        "candidate_train_unseen",
    ]
    for reason in priority:
        if reason in reasons:
            return reason
    return "validation_failed"


def history_bucket(length: int) -> str:
    """Return the history-length bucket used by SASRec impact outputs."""

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
    if 20 <= length <= 49:
        return "rows_history_length_20_to_49"
    return "rows_history_length_50_or_more"


def json_array(values: Iterable[Any]) -> str:
    """Serialize an iterable as a deterministic JSON array of strings."""

    return json.dumps([safe_raw_id(value) for value in values], ensure_ascii=True)


def bool_text(value: Any) -> str:
    """Serialize bool-like values as lowercase CSV text."""

    return "true" if bool(value) else "false"

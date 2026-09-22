"""Plan E LightGCN adapter and graph preflight utilities.

This module audits data contracts for a future LightGCN pilot. It does not
train a model, score project-data candidates, write rankings, or create
checkpoints. Synthetic forward-only probes are provided for sparse execution
checks on toy graphs only.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
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


class LightGCNAdapterError(ValueError):
    """Base exception for Plan E LightGCN adapter validation failures."""


class MappingValidationError(LightGCNAdapterError):
    """Raised when an explicit ID mapping is invalid or ambiguous."""


class GraphValidationError(LightGCNAdapterError):
    """Raised when graph construction or normalization is invalid."""


class CandidateValidationError(LightGCNAdapterError):
    """Raised when a candidate row cannot be safely adapted."""


class ScoreAlignmentError(LightGCNAdapterError):
    """Raised when a future score vector cannot be aligned to candidates."""


class DenseAdjacencyForbiddenError(GraphValidationError):
    """Raised when code requests a dense project-sized adjacency matrix."""


@dataclass(frozen=True)
class EntityMapping:
    """Validated dataset-scoped raw-to-internal mapping."""

    entity_type: str
    raw_to_internal: dict[str, int]
    internal_to_raw: dict[int, str]
    source_path: str
    mapping_direction: str = "raw_to_internal"


@dataclass(frozen=True)
class LocalIndexMapping:
    """Dense LightGCN local index space for graph-connected raw IDs."""

    entity_type: str
    raw_to_local: dict[str, int]
    local_to_raw: dict[int, str]

    @property
    def count(self) -> int:
        return len(self.raw_to_local)


@dataclass(frozen=True)
class DirectedBipartiteEdges:
    """Symmetric directed edge representation for a bipartite LightGCN graph."""

    user_indices: np.ndarray
    item_indices: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    raw_values: np.ndarray
    num_users: int
    num_items: int

    @property
    def total_nodes(self) -> int:
        return int(self.num_users + self.num_items)


@dataclass(frozen=True)
class NormalizedSparseGraph:
    """Coalesced normalized sparse edge representation."""

    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    degrees: np.ndarray
    total_nodes: int


@dataclass(frozen=True)
class GraphContract:
    """Training-only graph contract for one dataset."""

    train_pairs: tuple[tuple[str, str], ...]
    user_mapping: LocalIndexMapping
    item_mapping: LocalIndexMapping
    directed_edges: DirectedBipartiteEdges
    normalized_graph: NormalizedSparseGraph
    duplicate_user_item_pair_count: int


@dataclass(frozen=True)
class CandidateNodePayload:
    """Future candidate node payload with raw order preserved."""

    user_id: str
    user_local_index: int
    user_global_node_index: int
    raw_candidate_item_ids: tuple[str, ...]
    item_local_indices: tuple[int, ...]
    item_global_node_indices: tuple[int, ...]


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

    text = str(value).strip()
    try:
        return (0, int(text), "")
    except ValueError:
        return (1, 0, text)


def read_csv_strings(path: str | Path) -> pd.DataFrame:
    """Read a CSV while preserving IDs as strings."""

    return pd.read_csv(path, dtype="string", keep_default_na=False)


def json_array(values: Sequence[Any]) -> str:
    """Serialize an ID array deterministically for CSV cells."""

    return json.dumps([str(value) for value in values], ensure_ascii=True, separators=(",", ":"))


def bool_text(value: Any) -> str:
    """Serialize booleans as lower-case CSV-friendly text."""

    return "true" if bool(value) else "false"


def bool_value(value: Any) -> bool:
    """Parse common repository boolean encodings."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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


def filter_train_interactions(interactions: pd.DataFrame, *, train_value: str = "train") -> pd.DataFrame:
    """Return only rows whose split exactly matches the verified train value."""

    required = {"user_id", "item_id", "split"}
    missing = required.difference(interactions.columns)
    if missing:
        raise GraphValidationError(f"interactions missing columns: {sorted(missing)}")
    frame = interactions.copy()
    for column in ("user_id", "item_id", "split"):
        frame[column] = frame[column].astype("string").str.strip()
    split = frame["split"].str.lower()
    train = frame.loc[split == train_value.lower()].copy()
    if train.empty:
        raise GraphValidationError(f"no split={train_value!r} interactions are available")
    return train


def stable_unique_train_pairs(train_interactions: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    """Return deterministic unique train user-item pairs.

    This mirrors the Plan D loader contract, which calls ``drop_duplicates`` on
    ``user_id,item_id`` before training arrays are passed into LightGCN.
    """

    required = {"user_id", "item_id"}
    missing = required.difference(train_interactions.columns)
    if missing:
        raise GraphValidationError(f"training interactions missing columns: {sorted(missing)}")
    pairs = {
        (safe_raw_id(row.user_id), safe_raw_id(row.item_id))
        for row in train_interactions[["user_id", "item_id"]].itertuples(index=False)
    }
    if not pairs:
        raise GraphValidationError("at least one unique training pair is required")
    return tuple(sorted(pairs, key=lambda pair: (natural_id_sort_key(pair[0]), natural_id_sort_key(pair[1]))))


def duplicate_user_item_pair_count(train_interactions: pd.DataFrame) -> int:
    """Count repeated user-item rows in already-filtered train interactions."""

    total = len(train_interactions)
    unique = len(stable_unique_train_pairs(train_interactions)) if total else 0
    return int(total - unique)


def build_local_index(raw_ids: Iterable[Any], *, entity_type: str) -> LocalIndexMapping:
    """Build a deterministic dense LightGCN local index mapping."""

    normalized = [safe_raw_id(value) for value in raw_ids]
    unique = sorted(set(normalized), key=natural_id_sort_key)
    if not unique:
        raise MappingValidationError(f"{entity_type} local index cannot be empty")
    raw_to_local = {raw_id: index for index, raw_id in enumerate(unique)}
    local_to_raw = {index: raw_id for raw_id, index in raw_to_local.items()}
    return LocalIndexMapping(entity_type=entity_type, raw_to_local=raw_to_local, local_to_raw=local_to_raw)


def build_directed_bipartite_edges(
    user_indices: Sequence[int],
    item_indices: Sequence[int],
    *,
    num_users: int,
    num_items: int,
) -> DirectedBipartiteEdges:
    """Build symmetric directed bipartite edges with item nodes offset by users."""

    users = np.asarray(user_indices, dtype=np.int64)
    items = np.asarray(item_indices, dtype=np.int64)
    if users.shape != items.shape:
        raise GraphValidationError("user_indices and item_indices must have the same shape")
    if users.size == 0:
        raise GraphValidationError("at least one training interaction is required")
    validate_node_indices(users, lower=0, upper=num_users, entity_type="user")
    validate_node_indices(items, lower=0, upper=num_items, entity_type="item")
    item_nodes = items + int(num_users)
    rows = np.concatenate([users, item_nodes]).astype(np.int64, copy=False)
    cols = np.concatenate([item_nodes, users]).astype(np.int64, copy=False)
    values = np.ones(rows.shape[0], dtype=np.float32)
    return DirectedBipartiteEdges(
        user_indices=users,
        item_indices=items,
        rows=rows,
        cols=cols,
        raw_values=values,
        num_users=int(num_users),
        num_items=int(num_items),
    )


def validate_node_indices(indices: np.ndarray, *, lower: int, upper: int, entity_type: str) -> None:
    """Validate that local indices are in ``[lower, upper)``."""

    if upper <= lower:
        raise GraphValidationError(f"{entity_type} index upper bound must exceed lower bound")
    if indices.size == 0:
        raise GraphValidationError(f"{entity_type} indices are empty")
    minimum = int(indices.min())
    maximum = int(indices.max())
    if minimum < lower or maximum >= upper:
        raise GraphValidationError(
            f"{entity_type} local indices out of range [{lower}, {upper}): min={minimum}, max={maximum}"
        )


def assert_valid_offset(num_users: int, num_items: int) -> None:
    """Validate the LightGCN user/item bipartite node offset contract."""

    if num_users <= 0 or num_items <= 0:
        raise GraphValidationError("num_users and num_items must be positive")
    if int(num_users) + int(num_items) <= int(num_users):
        raise GraphValidationError("invalid item-node offset")


def compute_degrees(rows: np.ndarray, values: np.ndarray, *, total_nodes: int) -> np.ndarray:
    """Compute out-degree after graph symmetrization."""

    if rows.size != values.size:
        raise GraphValidationError("rows and values must have the same length")
    if total_nodes <= 0:
        raise GraphValidationError("total_nodes must be positive")
    degree = np.bincount(rows.astype(np.int64), weights=values.astype(np.float64), minlength=total_nodes)
    return degree.astype(np.float64, copy=False)


def normalize_symmetric_edges(edges: DirectedBipartiteEdges) -> NormalizedSparseGraph:
    """Apply Plan D symmetric ``D^-1/2 A D^-1/2`` normalization and coalesce."""

    total_nodes = edges.total_nodes
    degree = compute_degrees(edges.rows, edges.raw_values, total_nodes=total_nodes)
    row_degree = degree[edges.rows]
    col_degree = degree[edges.cols]
    with np.errstate(divide="raise", invalid="raise"):
        try:
            values = np.power(row_degree, -0.5) * np.power(col_degree, -0.5)
        except FloatingPointError as exc:
            raise GraphValidationError("zero-degree edge endpoint encountered during normalization") from exc
    values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise GraphValidationError("normalized edge weights contain non-finite values")
    rows, cols, coalesced_values = coalesce_edges(edges.rows, edges.cols, values)
    verify_symmetric_edges(rows, cols)
    return NormalizedSparseGraph(
        rows=rows,
        cols=cols,
        values=coalesced_values,
        degrees=degree,
        total_nodes=total_nodes,
    )


def coalesce_edges(rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coalesce duplicate sparse COO entries by summing values deterministically."""

    if rows.shape != cols.shape or rows.shape != values.shape:
        raise GraphValidationError("rows, cols and values must have identical shapes")
    order = np.lexsort((cols, rows))
    sorted_rows = rows[order].astype(np.int64, copy=False)
    sorted_cols = cols[order].astype(np.int64, copy=False)
    sorted_values = values[order].astype(np.float32, copy=False)
    if sorted_rows.size == 0:
        raise GraphValidationError("cannot coalesce an empty edge set")
    keys_changed = np.ones(sorted_rows.size, dtype=bool)
    keys_changed[1:] = (sorted_rows[1:] != sorted_rows[:-1]) | (sorted_cols[1:] != sorted_cols[:-1])
    starts = np.flatnonzero(keys_changed)
    summed = np.add.reduceat(sorted_values, starts).astype(np.float32)
    return sorted_rows[starts], sorted_cols[starts], summed


def verify_symmetric_edges(rows: np.ndarray, cols: np.ndarray) -> None:
    """Verify that every directed edge has a reverse directed edge."""

    edge_set = set(zip(rows.astype(int).tolist(), cols.astype(int).tolist()))
    for row, col in edge_set:
        if (col, row) not in edge_set:
            raise GraphValidationError(f"edge ({row}, {col}) lacks a reverse edge")


def build_graph_contract(train_interactions: pd.DataFrame) -> GraphContract:
    """Build an audit-only LightGCN graph contract from train interactions."""

    pairs = stable_unique_train_pairs(train_interactions)
    duplicate_count = int(len(train_interactions) - len(pairs))
    user_mapping = build_local_index((user_id for user_id, _ in pairs), entity_type="user")
    item_mapping = build_local_index((item_id for _, item_id in pairs), entity_type="item")
    user_indices = np.asarray([user_mapping.raw_to_local[user_id] for user_id, _ in pairs], dtype=np.int64)
    item_indices = np.asarray([item_mapping.raw_to_local[item_id] for _, item_id in pairs], dtype=np.int64)
    edges = build_directed_bipartite_edges(
        user_indices,
        item_indices,
        num_users=user_mapping.count,
        num_items=item_mapping.count,
    )
    normalized = normalize_symmetric_edges(edges)
    return GraphContract(
        train_pairs=pairs,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        directed_edges=edges,
        normalized_graph=normalized,
        duplicate_user_item_pair_count=duplicate_count,
    )


def hash_raw_pairs(pairs: Sequence[tuple[str, str]]) -> str:
    """Hash sorted raw user-item pairs without persisting an edge-list file."""

    hasher = hashlib.sha256()
    for user_id, item_id in sorted(pairs, key=lambda pair: (natural_id_sort_key(pair[0]), natural_id_sort_key(pair[1]))):
        hasher.update(user_id.encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(item_id.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest().upper()


def hash_int_arrays(*arrays: np.ndarray) -> str:
    """Hash integer arrays in a stable byte order."""

    hasher = hashlib.sha256()
    for array in arrays:
        hasher.update(np.asarray(array, dtype="<i8").tobytes())
    return hasher.hexdigest().upper()


def hash_float_array(array: np.ndarray) -> str:
    """Hash float arrays in a stable byte order."""

    return hashlib.sha256(np.asarray(array, dtype="<f4").tobytes()).hexdigest().upper()


def hash_normalized_graph(graph: NormalizedSparseGraph) -> str:
    """Hash a coalesced normalized sparse graph deterministically."""

    hasher = hashlib.sha256()
    hasher.update(np.asarray(graph.rows, dtype="<i8").tobytes())
    hasher.update(np.asarray(graph.cols, dtype="<i8").tobytes())
    hasher.update(np.asarray(graph.values, dtype="<f4").tobytes())
    return hasher.hexdigest().upper()


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


def prepare_candidate_node_payload(
    *,
    user_id: Any,
    candidate_item_ids: Sequence[Any],
    user_mapping: LocalIndexMapping,
    item_mapping: LocalIndexMapping,
) -> CandidateNodePayload:
    """Prepare future LightGCN node indices without scoring project data."""

    raw_user_id = safe_raw_id(user_id)
    if raw_user_id not in user_mapping.raw_to_local:
        raise CandidateValidationError(f"user is not graph-connected: {raw_user_id}")
    raw_candidates = tuple(safe_raw_id(item) for item in candidate_item_ids)
    if len(raw_candidates) != PLAN_E2_CANDIDATE_COUNT:
        raise CandidateValidationError(f"expected {PLAN_E2_CANDIDATE_COUNT} candidates, found {len(raw_candidates)}")
    item_indices: list[int] = []
    item_nodes: list[int] = []
    for item_id in raw_candidates:
        if item_id not in item_mapping.raw_to_local:
            raise CandidateValidationError(f"candidate item is not graph-connected: {item_id}")
        local = item_mapping.raw_to_local[item_id]
        item_indices.append(local)
        item_nodes.append(user_mapping.count + local)
    return CandidateNodePayload(
        user_id=raw_user_id,
        user_local_index=user_mapping.raw_to_local[raw_user_id],
        user_global_node_index=user_mapping.raw_to_local[raw_user_id],
        raw_candidate_item_ids=raw_candidates,
        item_local_indices=tuple(item_indices),
        item_global_node_indices=tuple(item_nodes),
    )


def validate_candidate_order(original: Sequence[Any], payload: CandidateNodePayload) -> None:
    """Ensure candidate order is preserved exactly."""

    original_ids = tuple(safe_raw_id(item) for item in original)
    if original_ids != payload.raw_candidate_item_ids:
        raise CandidateValidationError("candidate order was changed")


def align_scores_to_candidate_ids(candidate_item_ids: Sequence[Any], scores: Sequence[float]) -> list[dict[str, Any]]:
    """Validate future score vectors and preserve candidate order."""

    candidates = [safe_raw_id(item) for item in candidate_item_ids]
    if len(candidates) != len(scores):
        raise ScoreAlignmentError("score vector length does not match candidate count")
    rows: list[dict[str, Any]] = []
    for position, (item_id, score) in enumerate(zip(candidates, scores)):
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ScoreAlignmentError("score vector contains a non-finite value")
        rows.append(
            {
                "candidate_item_id": item_id,
                "candidate_position_zero_based": position,
                "score": numeric_score,
            }
        )
    return rows


def audit_plan_e2_candidate_row(
    row: Mapping[str, Any],
    *,
    user_mapping: EntityMapping,
    item_mapping: EntityMapping,
    graph_users: set[str],
    graph_items: set[str],
    user_local_index: Mapping[str, int],
    item_local_index: Mapping[str, int],
    normalization_verified: bool,
    eligible_full: bool,
    eligible_matched: bool,
) -> dict[str, Any]:
    """Audit one Plan E2 candidate row for strict LightGCN graph compatibility."""

    user_id = safe_raw_id(row["user_id"])
    positive_item_id = safe_raw_id(row["positive_item_id"])
    candidate_items = parse_candidate_items(row["candidate_items"])
    user_mapped = user_id in user_mapping.raw_to_internal
    user_graph_connected = user_id in graph_users
    positive_item_mapped = positive_item_id in item_mapping.raw_to_internal
    positive_item_graph_connected = positive_item_id in graph_items
    unmapped_candidates = [item_id for item_id in candidate_items if item_id not in item_mapping.raw_to_internal]
    disconnected_candidates = [item_id for item_id in candidate_items if item_id in item_mapping.raw_to_internal and item_id not in graph_items]
    all_candidate_items_mapped = not unmapped_candidates
    all_candidate_items_graph_connected = not disconnected_candidates and all_candidate_items_mapped
    index_compatible = (
        user_graph_connected
        and positive_item_graph_connected
        and all_candidate_items_graph_connected
        and user_id in user_local_index
        and positive_item_id in item_local_index
        and all(item_id in item_local_index for item_id in candidate_items)
    )
    isolated_embedding_risk = (
        (user_mapped and not user_graph_connected)
        or (positive_item_mapped and not positive_item_graph_connected)
        or bool(disconnected_candidates)
    )
    graph_compatible = (
        user_mapped
        and user_graph_connected
        and positive_item_mapped
        and positive_item_graph_connected
        and all_candidate_items_mapped
        and all_candidate_items_graph_connected
        and index_compatible
        and normalization_verified
    )
    status, reasons = compatibility_status_and_reasons(
        user_mapped=user_mapped,
        user_graph_connected=user_graph_connected,
        positive_item_mapped=positive_item_mapped,
        positive_item_graph_connected=positive_item_graph_connected,
        all_candidate_items_mapped=all_candidate_items_mapped,
        all_candidate_items_graph_connected=all_candidate_items_graph_connected,
        existing_model_index_compatible=index_compatible,
        normalization_verified=normalization_verified,
    )
    return {
        "candidate_set_id": safe_raw_id(row["candidate_set_id"]),
        "source_path": str(row.get("source_path", "")).replace("\\", "/"),
        "dataset": safe_raw_id(row["dataset"]),
        "task_type": safe_raw_id(row.get("task_type", "")) if str(row.get("task_type", "")).strip() else "",
        "user_id": user_id,
        "positive_item_id": positive_item_id,
        "candidate_count": len(candidate_items),
        "user_mapped": user_mapped,
        "user_graph_connected": user_graph_connected,
        "positive_item_mapped": positive_item_mapped,
        "positive_item_graph_connected": positive_item_graph_connected,
        "all_candidate_items_mapped": all_candidate_items_mapped,
        "all_candidate_items_graph_connected": all_candidate_items_graph_connected,
        "unmapped_candidate_item_count": len(unmapped_candidates),
        "unmapped_candidate_item_ids": json_array(unmapped_candidates),
        "graph_disconnected_candidate_item_count": len(disconnected_candidates),
        "graph_disconnected_candidate_item_ids": json_array(disconnected_candidates),
        "all_candidate_items_train_seen": all_candidate_items_graph_connected,
        "existing_model_index_compatible": index_compatible,
        "contains_isolated_embedding_risk": isolated_embedding_risk,
        "lightgcn_graph_compatible": graph_compatible,
        "eligible_plan_e2_full": bool(eligible_full),
        "eligible_plan_e2_matched": bool(eligible_matched),
        "compatibility_status": status,
        "exclusion_reasons": ";".join(reasons) if reasons else "none",
        "notes": "LightGCN preflight only; no project-data scoring performed",
    }


def compatibility_status_and_reasons(
    *,
    user_mapped: bool,
    user_graph_connected: bool,
    positive_item_mapped: bool,
    positive_item_graph_connected: bool,
    all_candidate_items_mapped: bool,
    all_candidate_items_graph_connected: bool,
    existing_model_index_compatible: bool,
    normalization_verified: bool,
) -> tuple[str, list[str]]:
    """Return the strict compatibility status and deterministic reasons."""

    reasons: list[str] = []
    if not user_mapped:
        reasons.append("user_unmapped")
    if not positive_item_mapped:
        reasons.append("positive_item_unmapped")
    if not all_candidate_items_mapped:
        reasons.append("candidate_item_unmapped")
    if reasons:
        return "mapping_incompatible", reasons
    if not user_graph_connected:
        return "user_graph_disconnected", ["user_graph_disconnected"]
    if not positive_item_graph_connected:
        return "positive_graph_disconnected", ["positive_item_graph_disconnected"]
    if not all_candidate_items_graph_connected:
        return "graph_disconnected_candidate", ["candidate_item_graph_disconnected"]
    if not existing_model_index_compatible:
        return "index_incompatible", ["model_index_incompatible"]
    if not normalization_verified:
        return "normalization_unverified", ["normalization_unverified"]
    return "lightgcn_graph_compatible", []


def reject_dense_adjacency_request(*, total_nodes: int, project_graph: bool, max_synthetic_nodes: int = 1000) -> None:
    """Reject dense adjacency conversion for project-sized graphs."""

    if project_graph or total_nodes > max_synthetic_nodes:
        raise DenseAdjacencyForbiddenError("dense adjacency conversion is forbidden for project LightGCN graphs")


def estimate_sparse_memory_bytes(
    *,
    total_nodes: int,
    directed_edges: int,
    embedding_dim: int = 64,
    layer_count: int = 2,
) -> dict[str, int]:
    """Estimate persistent sparse tensors and embeddings without runtime claims."""

    edge_index_bytes = int(2 * directed_edges * np.dtype(np.int64).itemsize)
    edge_weight_bytes = int(directed_edges * np.dtype(np.float32).itemsize)
    embedding_bytes = int(total_nodes * embedding_dim * np.dtype(np.float32).itemsize)
    total_bytes = int(edge_index_bytes + edge_weight_bytes + embedding_bytes * (layer_count + 1))
    return {
        "estimated_edge_index_bytes": edge_index_bytes,
        "estimated_edge_weight_bytes": edge_weight_bytes,
        "estimated_embedding_bytes_per_layer": embedding_bytes,
        "estimated_total_graph_tensor_bytes": total_bytes,
    }


def synthetic_sparse_forward_probe(*, device: str = "cpu", deterministic: bool = True) -> dict[str, Any]:
    """Run a tiny synthetic LightGCN forward-only sparse probe.

    The probe uses toy IDs and a toy sparse graph, creates no optimizer, calls no
    backward pass, saves no checkpoint, and never touches repository data.
    """

    import torch

    from src.baselines.plan_d_lightgcn import LightGCNModel, build_normalized_bipartite_graph

    old_state = torch.are_deterministic_algorithms_enabled()
    warning_messages: list[str] = []
    try:
        if deterministic:
            torch.use_deterministic_algorithms(True)
        torch.manual_seed(1234)
        target_device = torch.device(device)
        graph = build_normalized_bipartite_graph(
            np.asarray([0, 0, 1], dtype=np.int64),
            np.asarray([0, 1, 1], dtype=np.int64),
            num_users=2,
            num_items=2,
            device=target_device,
        )
        model = LightGCNModel(num_users=2, num_items=2, embedding_dim=3, num_layers=2).to(target_device)
        model.eval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                user_embeddings, item_embeddings = model.final_embeddings(graph)
            warning_messages = [str(item.message) for item in caught]
        return {
            "passed": True,
            "device": device,
            "user_shape": tuple(user_embeddings.shape),
            "item_shape": tuple(item_embeddings.shape),
            "warnings": warning_messages,
            "error": "",
            "output": torch.cat([user_embeddings.flatten(), item_embeddings.flatten()]).detach().cpu().numpy(),
        }
    except Exception as exc:  # pragma: no cover - exercised when CUDA/backends fail.
        return {
            "passed": False,
            "device": device,
            "user_shape": (),
            "item_shape": (),
            "warnings": warning_messages,
            "error": f"{type(exc).__name__}: {exc}",
            "output": np.asarray([], dtype=np.float32),
        }
    finally:
        torch.use_deterministic_algorithms(old_state)


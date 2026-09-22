"""Canonical metric computation helpers for Phase 4B.

These helpers implement the formulas frozen by Phase 4A. They are deliberately
small and deterministic so the experiment runner can validate source contracts
before applying them to project ranking artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TOP_K = 5

COMPUTABLE_METRICS = (
    "hr_at_5",
    "ndcg_at_5",
    "mean_positive_rank_zero_based",
    "median_positive_rank_zero_based",
    "novelty_at_5",
    "catalog_coverage_at_5",
    "item_coverage_at_5",
    "long_tail_rate_at_5",
    "average_popularity_at_5",
    "duplicate_rate_at_5",
    "candidate_validity_rate",
)

BLOCKED_METRICS = ("popularity_bias_at_5", "intra_list_diversity_at_5")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_items(items: Iterable[Any]) -> str:
    return sha256_text(json_dumps([str(item) for item in sorted(items, key=str)]))


def contract_payload_hash(row: Mapping[str, Any], *, hash_column: str = "contract_sha256") -> str:
    payload = {str(key): str(value) for key, value in row.items() if str(key) != hash_column}
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def parse_json_array(value: Any) -> list[str]:
    decoded = json.loads(str(value))
    if not isinstance(decoded, list):
        raise ValueError("cell is not a JSON array")
    return [str(item) for item in decoded]


def parse_float_array(value: Any) -> list[float]:
    decoded = json.loads(str(value))
    if not isinstance(decoded, list):
        raise ValueError("score cell is not a JSON array")
    return [float(item) for item in decoded]


def finite_scores(scores: Sequence[float]) -> bool:
    return all(math.isfinite(float(score)) for score in scores)


def hr_from_rank(rank_zero_based: int, *, k: int = TOP_K) -> float:
    return 1.0 if int(rank_zero_based) < k else 0.0


def ndcg_from_rank(rank_zero_based: int, *, k: int = TOP_K) -> float:
    rank = int(rank_zero_based)
    if rank >= k:
        return 0.0
    return 1.0 / math.log2(rank + 2.0)


def self_information_bits(train_count: int, total_train_interactions: int) -> float:
    if train_count <= 0:
        raise ValueError("zero train count cannot be smoothed")
    if total_train_interactions <= 0:
        raise ValueError("total train interactions must be positive")
    probability = float(train_count) / float(total_train_interactions)
    if probability <= 0.0:
        raise ValueError("popularity probability must be positive")
    return -math.log2(probability)


def train_item_statistics(interactions: pd.DataFrame) -> tuple[Counter[str], Counter[str], int]:
    """Return train-only item counts, train unique-user counts, and total rows."""

    required = {"user_id", "item_id", "split"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions missing columns: {sorted(missing)}")
    train = interactions.loc[interactions["split"].astype(str).str.lower().eq("train")].copy()
    train["item_id"] = train["item_id"].astype(str)
    train["user_id"] = train["user_id"].astype(str)
    item_counts = Counter(train["item_id"].tolist())
    unique_users = train.drop_duplicates(["item_id", "user_id"]).groupby("item_id", sort=False).size()
    unique_user_counts = Counter({str(item): int(count) for item, count in unique_users.items()})
    total = int(sum(item_counts.values()))
    return item_counts, unique_user_counts, total


def descending_popularity_rank_items(counts: Mapping[str, int]) -> dict[str, int]:
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return {str(item_id): rank for rank, (item_id, _) in enumerate(ordered, start=1)}


def popularity_quantiles(counts: Mapping[str, int]) -> dict[str, float]:
    values = np.asarray(sorted([int(value) for value in counts.values()]), dtype=float)
    if len(values) == 0:
        return {}
    return {
        str(item_id): float(np.searchsorted(values, float(count), side="right") / len(values))
        for item_id, count in counts.items()
    }


def quantile_threshold(counts: Mapping[str, int], quantile: float = 0.8) -> float:
    positive = np.asarray([float(count) for count in counts.values() if int(count) > 0], dtype=float)
    if len(positive) == 0:
        raise ValueError("cannot compute long-tail threshold without positive counts")
    return float(np.quantile(positive, quantile))


def long_tail_member(train_count: int, threshold: float) -> bool:
    return int(train_count) > 0 and float(train_count) < float(threshold)


def duplicate_flags(items: Sequence[str]) -> list[bool]:
    seen: set[str] = set()
    flags: list[bool] = []
    for item in items:
        flags.append(item in seen)
        seen.add(item)
    return flags


def duplicate_rate(items: Sequence[str], *, k: int = TOP_K) -> float:
    top = list(items[:k])
    if len(top) != k:
        raise ValueError(f"expected {k} top items, found {len(top)}")
    return float(sum(duplicate_flags(top)) / k)


def candidate_validity_rate(
    top_items: Sequence[str],
    candidate_items: Iterable[str],
    train_seen_items: Iterable[str],
    *,
    k: int = TOP_K,
) -> float:
    top = list(top_items[:k])
    if len(top) != k:
        raise ValueError(f"expected {k} top items, found {len(top)}")
    candidates = {str(item) for item in candidate_items}
    train_seen = {str(item) for item in train_seen_items}
    valid_count = sum(1 for item in top if str(item) in candidates and str(item) in train_seen)
    return float(valid_count / k)


def row_metric_contributions(
    *,
    positive_rank_zero_based: int,
    top_items: Sequence[str],
    candidate_items: Sequence[str],
    train_counts: Mapping[str, int],
    unique_user_counts: Mapping[str, int],
    total_train_interactions: int,
    long_tail_threshold: float,
) -> dict[str, Any]:
    """Return row-level top-5 contributions under the Phase 4A contract."""

    if len(top_items) != TOP_K:
        raise ValueError(f"expected {TOP_K} top items")
    train_seen = set(train_counts)
    counts = [int(train_counts.get(str(item), 0)) for item in top_items]
    if any(count <= 0 for count in counts):
        raise ValueError("all top-5 items must have positive train popularity")
    unique_counts = [int(unique_user_counts.get(str(item), 0)) for item in top_items]
    info = [self_information_bits(count, total_train_interactions) for count in counts]
    tail = [long_tail_member(count, long_tail_threshold) for count in counts]
    duplicate = duplicate_flags([str(item) for item in top_items])
    validity = candidate_validity_rate(top_items, candidate_items, train_seen)
    return {
        "hr_at_5": hr_from_rank(positive_rank_zero_based),
        "ndcg_at_5": ndcg_from_rank(positive_rank_zero_based),
        "novelty_at_5": float(np.mean(np.asarray(info, dtype=float))),
        "long_tail_rate_at_5": float(np.mean(np.asarray(tail, dtype=float))),
        "average_popularity_at_5": float(np.mean(np.asarray(counts, dtype=float))),
        "duplicate_rate_at_5": float(np.mean(np.asarray(duplicate, dtype=float))),
        "candidate_validity_rate": validity,
        "top5_unique_item_count": len(set(str(item) for item in top_items)),
        "top5_valid_item_count": int(validity * TOP_K),
        "top5_train_popularity_counts": counts,
        "top5_train_unique_user_counts": unique_counts,
        "top5_self_information_bits": info,
        "top5_long_tail_membership": tail,
        "top5_duplicate_flags": duplicate,
    }


def coverage_ratio(numerator_items: Iterable[str], denominator_items: Iterable[str]) -> tuple[int, int, float]:
    numerator = {str(item) for item in numerator_items}
    denominator = {str(item) for item in denominator_items}
    if not denominator:
        raise ValueError("coverage denominator is empty")
    valid_numerator = numerator.intersection(denominator)
    return len(valid_numerator), len(denominator), float(len(valid_numerator) / len(denominator))


def mean_float(values: Sequence[Any]) -> float:
    if not values:
        raise ValueError("cannot aggregate empty values")
    return float(np.mean(np.asarray([float(value) for value in values], dtype=np.float64)))


def median_float(values: Sequence[Any]) -> float:
    if not values:
        raise ValueError("cannot aggregate empty values")
    return float(np.median(np.asarray([float(value) for value in values], dtype=np.float64)))


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        raise ValueError("non-finite float cannot be serialized")
    return f"{float(value):.17g}"


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def require_no_blocked_substitution(metric_id: str) -> None:
    if metric_id in BLOCKED_METRICS:
        raise ValueError(f"blocked metric cannot be computed: {metric_id}")


def require_known_metric(metric_id: str) -> None:
    if metric_id not in COMPUTABLE_METRICS and metric_id not in BLOCKED_METRICS:
        raise ValueError(f"unknown canonical metric: {metric_id}")

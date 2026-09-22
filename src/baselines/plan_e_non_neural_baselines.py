"""Deterministic non-neural baseline utilities for Plan E candidate sets.

The functions in this module are intentionally small and data-frame free so
they can be tested independently from repository-sized artifacts. They do not
train models, invoke neural libraries, or call external services.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.audit.common_io import write_csv_rows as _write_csv_rows


GLOBAL_RANDOM_SEED = 20260710
TOP_K = 5

RUN_MANIFEST_COLUMNS = [
    "run_id",
    "baseline_id",
    "baseline_family",
    "implementation_path",
    "configuration_path",
    "dataset",
    "task_type",
    "candidate_set_id",
    "source_plan",
    "candidate_count",
    "full_cohort_rows",
    "matched_cohort_rows",
    "training_data_path",
    "content_data_path",
    "global_seed",
    "seed_derivation",
    "score_definition",
    "tie_breaking_rule",
    "paper_alignment",
    "execution_status",
    "blocking_reason",
    "output_path",
    "notes",
]

RANKINGS_COLUMNS = [
    "run_id",
    "baseline_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "source_path",
    "user_id",
    "positive_item_id",
    "candidate_items",
    "candidate_scores",
    "ranked_items",
    "ranked_scores",
    "positive_rank_zero_based",
    "positive_rank_one_based",
    "positive_in_top_5",
    "ndcg_at_5",
    "ranking_valid",
    "duplicate_ranked_item_count",
    "missing_score_count",
    "eligible_full_cohort",
    "eligible_matched_cohort",
    "notes",
]

METRICS_COLUMNS = [
    "run_id",
    "baseline_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "cohort_id",
    "candidate_count",
    "evaluated_rows",
    "hr_at_5",
    "ndcg_at_5",
    "mean_positive_rank_zero_based",
    "median_positive_rank_zero_based",
    "valid_ranking_rate",
    "duplicate_ranked_item_rate",
    "missing_score_rate",
    "paper_alignment",
    "notes",
]

VALIDATION_COLUMNS = [
    "validation_id",
    "run_id",
    "baseline_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "check_name",
    "expected_value",
    "observed_value",
    "status",
    "severity",
    "evidence",
    "notes",
]

BLOCKED_COLUMNS = [
    "baseline_id",
    "baseline_family",
    "dataset",
    "task_type",
    "blocking_category",
    "blocking_reason",
    "required_dependency_or_artifact",
    "current_evidence",
    "recommended_future_action",
    "notes",
]


@dataclass(frozen=True)
class BaselineSpec:
    """Static definition for a baseline execution unit."""

    baseline_id: str
    baseline_family: str
    score_definition: str
    scoring_source_path: str
    training_source_path: str
    paper_alignment: str
    seed: str
    notes: str


@dataclass(frozen=True)
class ScoredCandidates:
    """Scores for one candidate row plus diagnostics."""

    candidate_scores: list[float]
    ranked_items: list[str]
    ranked_scores: list[float]
    missing_score_count: int


def stringify_id(value: Any) -> str:
    """Preserve item/user IDs as strings without integer coercion."""

    if value is None:
        return ""
    return str(value)


def parse_bool(value: Any) -> bool:
    """Parse common CSV boolean encodings conservatively."""

    if isinstance(value, bool):
        return value
    text = stringify_id(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def format_bool(value: bool) -> str:
    """Write deterministic lowercase booleans for CSV outputs."""

    return "true" if value else "false"


def decode_candidate_items(value: Any) -> list[str]:
    """Decode a candidate item JSON array while preserving string IDs."""

    if isinstance(value, list):
        return [stringify_id(item) for item in value]
    parsed = json.loads(stringify_id(value))
    if not isinstance(parsed, list):
        raise ValueError("candidate_items must decode to a JSON array")
    return [stringify_id(item) for item in parsed]


def json_list(values: Sequence[Any]) -> str:
    """Serialize lists deterministically for CSV cells."""

    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def stable_random_score(
    seed: int | str,
    dataset: str,
    task_type: str,
    candidate_set_id: str,
    user_id: str,
    item_id: str,
) -> float:
    """Return a reproducible SHA-256 based score independent of Python hash()."""

    key = "|".join(
        [
            stringify_id(seed),
            stringify_id(dataset),
            stringify_id(task_type),
            stringify_id(candidate_set_id),
            stringify_id(user_id),
            stringify_id(item_id),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    integer = int(digest[:16], 16)
    return integer / float(0xFFFFFFFFFFFFFFFF)


def compute_train_popularity(interaction_rows: Iterable[Mapping[str, Any]]) -> Counter[str]:
    """Count item popularity using only rows whose split is exactly train."""

    counts: Counter[str] = Counter()
    for row in interaction_rows:
        split = stringify_id(row.get("split", "")).strip().lower()
        if split != "train":
            continue
        item_id = stringify_id(row.get("item_id", ""))
        if item_id:
            counts[item_id] += 1
    return counts


def load_train_popularity(path: Path) -> tuple[Counter[str], int, int]:
    """Load train-only popularity counts from a reconstructed interaction CSV."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = compute_train_popularity(rows)
    train_rows = sum(1 for row in rows if stringify_id(row.get("split", "")).strip().lower() == "train")
    return counts, len(rows), train_rows


def rank_with_scores(
    candidate_items: Sequence[str],
    candidate_scores: Sequence[float],
) -> tuple[list[str], list[float]]:
    """Rank candidates by descending score, item ID, then original position."""

    if len(candidate_items) != len(candidate_scores):
        raise ValueError("candidate_items and candidate_scores must have the same length")
    indexed = [
        (stringify_id(item), float(score), position)
        for position, (item, score) in enumerate(zip(candidate_items, candidate_scores))
    ]
    ranked = sorted(indexed, key=lambda row: (-row[1], row[0], row[2]))
    return [item for item, _score, _position in ranked], [score for _item, score, _position in ranked]


def score_random_candidate_row(
    candidate_items: Sequence[str],
    seed: int | str,
    dataset: str,
    task_type: str,
    candidate_set_id: str,
    user_id: str,
) -> ScoredCandidates:
    """Score one row using seeded deterministic random scores."""

    scores = [
        stable_random_score(seed, dataset, task_type, candidate_set_id, user_id, item_id)
        for item_id in candidate_items
    ]
    ranked_items, ranked_scores = rank_with_scores(candidate_items, scores)
    return ScoredCandidates(scores, ranked_items, ranked_scores, missing_score_count=0)


def score_mostpop_candidate_row(
    candidate_items: Sequence[str],
    popularity_counts: Mapping[str, int],
) -> ScoredCandidates:
    """Score one row by train-only item frequency with zero for unseen candidates."""

    scores = [float(popularity_counts.get(stringify_id(item_id), 0)) for item_id in candidate_items]
    ranked_items, ranked_scores = rank_with_scores(candidate_items, scores)
    return ScoredCandidates(scores, ranked_items, ranked_scores, missing_score_count=0)


def score_lookup_candidate_row(
    candidate_items: Sequence[str],
    lookup_scores: Mapping[str, float],
    missing_score_value: float = 0.0,
) -> ScoredCandidates:
    """Score candidates from an explicit lookup, counting missing lookup entries."""

    scores: list[float] = []
    missing = 0
    for item_id in candidate_items:
        item_key = stringify_id(item_id)
        if item_key not in lookup_scores:
            missing += 1
            scores.append(float(missing_score_value))
        else:
            scores.append(float(lookup_scores[item_key]))
    ranked_items, ranked_scores = rank_with_scores(candidate_items, scores)
    return ScoredCandidates(scores, ranked_items, ranked_scores, missing_score_count=missing)


def positive_rank_zero_based(ranked_items: Sequence[str], positive_item_id: str) -> int | None:
    """Return the zero-based positive rank, or None if the positive is absent."""

    positive = stringify_id(positive_item_id)
    for index, item_id in enumerate(ranked_items):
        if stringify_id(item_id) == positive:
            return index
    return None


def hr_at_k(rank_zero_based: int | None, k: int = TOP_K) -> float:
    """Compute hit rate for one single-positive candidate row."""

    if rank_zero_based is None:
        return 0.0
    return 1.0 if rank_zero_based < k else 0.0


def ndcg_at_k(rank_zero_based: int | None, k: int = TOP_K) -> float:
    """Compute NDCG for one single-positive candidate row."""

    if rank_zero_based is None or rank_zero_based >= k:
        return 0.0
    rank_one_based = rank_zero_based + 1
    return 1.0 / math.log2(rank_one_based + 1)


def duplicate_count(values: Sequence[str]) -> int:
    """Return the number of duplicated positions in a ranked list."""

    return len(values) - len(set(values))


def finite_score_count(values: Sequence[float]) -> int:
    """Count finite score values."""

    return sum(1 for value in values if math.isfinite(float(value)))


def is_ranking_valid(
    candidate_items: Sequence[str],
    ranked_items: Sequence[str],
    ranked_scores: Sequence[float],
    positive_item_id: str,
) -> bool:
    """Validate structural ranking invariants for one candidate row."""

    if len(candidate_items) != len(ranked_items) or len(ranked_items) != len(ranked_scores):
        return False
    if len(set(ranked_items)) != len(ranked_items):
        return False
    if set(map(stringify_id, candidate_items)) != set(map(stringify_id, ranked_items)):
        return False
    if stringify_id(positive_item_id) not in set(map(stringify_id, ranked_items)):
        return False
    return finite_score_count(ranked_scores) == len(ranked_scores)


def make_ranking_output_row(
    *,
    run_id: str,
    baseline_id: str,
    dataset: str,
    task_type: str,
    candidate_set_id: str,
    source_path: str,
    user_id: str,
    positive_item_id: str,
    candidate_items: Sequence[str],
    scored: ScoredCandidates,
    eligible_full_cohort: bool,
    eligible_matched_cohort: bool,
    notes: str,
) -> dict[str, str]:
    """Create one rankings CSV row with JSON list fields."""

    rank_zero = positive_rank_zero_based(scored.ranked_items, positive_item_id)
    rank_one = "" if rank_zero is None else stringify_id(rank_zero + 1)
    valid = is_ranking_valid(candidate_items, scored.ranked_items, scored.ranked_scores, positive_item_id)
    dup_count = duplicate_count(scored.ranked_items)
    return {
        "run_id": run_id,
        "baseline_id": baseline_id,
        "dataset": dataset,
        "task_type": task_type,
        "candidate_set_id": candidate_set_id,
        "source_path": source_path,
        "user_id": stringify_id(user_id),
        "positive_item_id": stringify_id(positive_item_id),
        "candidate_items": json_list([stringify_id(item) for item in candidate_items]),
        "candidate_scores": json_list([round(float(score), 12) for score in scored.candidate_scores]),
        "ranked_items": json_list(scored.ranked_items),
        "ranked_scores": json_list([round(float(score), 12) for score in scored.ranked_scores]),
        "positive_rank_zero_based": "" if rank_zero is None else stringify_id(rank_zero),
        "positive_rank_one_based": rank_one,
        "positive_in_top_5": format_bool(hr_at_k(rank_zero) > 0),
        "ndcg_at_5": format_float(ndcg_at_k(rank_zero)),
        "ranking_valid": format_bool(valid),
        "duplicate_ranked_item_count": stringify_id(dup_count),
        "missing_score_count": stringify_id(scored.missing_score_count),
        "eligible_full_cohort": format_bool(eligible_full_cohort),
        "eligible_matched_cohort": format_bool(eligible_matched_cohort),
        "notes": notes,
    }


def aggregate_metrics(
    rankings: Sequence[Mapping[str, Any]],
    *,
    cohort_id: str,
    run_id: str,
    baseline_id: str,
    dataset: str,
    task_type: str,
    candidate_set_id: str,
    paper_alignment: str,
    notes: str,
) -> dict[str, str]:
    """Aggregate single-positive HR/NDCG and structural diagnostics."""

    if cohort_id == "plan_e2_matched_title_complete":
        selected = [row for row in rankings if parse_bool(row.get("eligible_matched_cohort"))]
    elif cohort_id == "plan_e2_full":
        selected = list(rankings)
    else:
        raise ValueError(f"Unknown cohort_id: {cohort_id}")

    ranks: list[int | None] = []
    for row in selected:
        rank_text = stringify_id(row.get("positive_rank_zero_based", ""))
        ranks.append(None if rank_text == "" else int(rank_text))

    evaluated = len(selected)
    if evaluated == 0:
        raise ValueError(f"No ranking rows selected for cohort_id={cohort_id}")
    candidate_counts = [len(decode_candidate_items(row["candidate_items"])) for row in selected]
    candidate_count_per_user = candidate_counts[0] if candidate_counts else 0
    total_positions = sum(candidate_counts)
    valid_count = sum(1 for row in selected if parse_bool(row.get("ranking_valid")))
    duplicate_total = sum(int(row["duplicate_ranked_item_count"]) for row in selected)
    missing_total = sum(int(row["missing_score_count"]) for row in selected)
    numeric_ranks = [rank for rank in ranks if rank is not None]

    hr = sum(hr_at_k(rank) for rank in ranks) / evaluated
    ndcg = sum(ndcg_at_k(rank) for rank in ranks) / evaluated
    mean_rank = sum(numeric_ranks) / len(numeric_ranks) if numeric_ranks else 0.0
    median_rank = float(statistics.median(numeric_ranks)) if numeric_ranks else 0.0
    valid_rate = valid_count / evaluated
    duplicate_rate = duplicate_total / total_positions if total_positions else 0.0
    missing_rate = missing_total / total_positions if total_positions else 0.0

    return {
        "run_id": run_id,
        "baseline_id": baseline_id,
        "dataset": dataset,
        "task_type": task_type,
        "candidate_set_id": candidate_set_id,
        "cohort_id": cohort_id,
        "candidate_count": stringify_id(candidate_count_per_user),
        "evaluated_rows": stringify_id(evaluated),
        "hr_at_5": format_float(hr),
        "ndcg_at_5": format_float(ndcg),
        "mean_positive_rank_zero_based": format_float(mean_rank),
        "median_positive_rank_zero_based": format_float(median_rank),
        "valid_ranking_rate": format_float(valid_rate),
        "duplicate_ranked_item_rate": format_float(duplicate_rate),
        "missing_score_rate": format_float(missing_rate),
        "paper_alignment": paper_alignment,
        "notes": notes,
    }


def format_float(value: float) -> str:
    """Format numeric outputs consistently without scientific notation churn."""

    return f"{float(value):.12g}"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a CSV as dictionaries preserving all values as strings."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a deterministic CSV with the requested header."""

    _write_csv_rows(path, fieldnames, rows, lineterminator="\n", value_transform=stringify_id)


def load_bm25_scores(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """Load prior Plan E3B item scores keyed by dataset, task, and user."""

    rows = load_csv_rows(path)
    lookup: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (
            stringify_id(row.get("dataset")),
            stringify_id(row.get("task")),
            stringify_id(row.get("user_id")),
        )
        item_id = stringify_id(row.get("item_id"))
        if not item_id:
            continue
        lookup[key][item_id] = float(row.get("bm25_score", "0") or 0.0)
    return dict(lookup)


def validate_bm25_variant_label(baseline_id: str, exactness_class: str) -> None:
    """Prevent reconstructed BM25-like scores from being labeled exact BM25."""

    exactness = stringify_id(exactness_class).lower()
    if baseline_id == "bm25_exact_official" and "reconstructed" in exactness:
        raise ValueError("Reconstructed BM25-like scores cannot be labeled exact official BM25")
    if baseline_id == "bm25_like_e3b" and "exact_official" in exactness:
        raise ValueError("Exact official BM25 evidence must not be labeled bm25_like_e3b")


def make_run_id(baseline_id: str, dataset: str, task_type: str) -> str:
    """Create a deterministic run identifier."""

    safe_dataset = stringify_id(dataset).lower().replace(" ", "_").replace("-", "_")
    safe_task = stringify_id(task_type).lower().replace(" ", "_")
    return f"phase3a_{baseline_id}_{safe_dataset}_{safe_task}"


def group_rows(
    rows: Iterable[Mapping[str, Any]],
    keys: Sequence[str],
) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    """Group rows by the requested string keys."""

    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(stringify_id(row.get(key, "")) for key in keys)].append(row)
    return dict(grouped)


def validation_row(
    *,
    validation_id: str,
    run_id: str,
    baseline_id: str,
    dataset: str,
    task_type: str,
    candidate_set_id: str,
    check_name: str,
    status: str,
    severity: str,
    observed_value: Any,
    expected_value: Any,
    evidence: str,
    notes: str,
) -> dict[str, str]:
    """Create one validation CSV row."""

    return {
        "validation_id": validation_id,
        "run_id": run_id,
        "baseline_id": baseline_id,
        "dataset": dataset,
        "task_type": task_type,
        "candidate_set_id": candidate_set_id,
        "check_name": check_name,
        "status": status,
        "severity": severity,
        "observed_value": stringify_id(observed_value),
        "expected_value": stringify_id(expected_value),
        "evidence": evidence,
        "notes": notes,
    }


def row_identity_for_random(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return the stable row identity used by the random baseline."""

    return (
        stringify_id(row.get("dataset")),
        stringify_id(row.get("task_type")),
        stringify_id(row.get("candidate_set_id")),
        stringify_id(row.get("user_id")),
    )

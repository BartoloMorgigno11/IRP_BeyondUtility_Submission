"""Offline RQ5 metrics with explicit fail-strict and canonical tie semantics."""
from __future__ import annotations
import json
import math
from collections.abc import Mapping, Sequence

def parse_ranking(text: str, candidates: Sequence[str], target: str, expected_count: int = 10, k: int = 5) -> dict:
    """Parse exact unique one-based candidate indices; invalid rankings have zero utility."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        value = None
    indices = value.get("ranked_candidate_indices") if isinstance(value, dict) else None
    valid = (isinstance(value, dict) and set(value) == {"ranked_candidate_indices"}
             and isinstance(indices, list) and len(indices) == expected_count
             and all(type(i) is int and 1 <= i <= len(candidates) for i in indices)
             and len(set(indices)) == expected_count)
    items = [str(candidates[i-1]) for i in indices] if valid else []
    rank = items.index(str(target))+1 if str(target) in items else 0
    hit = float(0 < rank <= k)
    return {"strict_valid": int(bool(valid)), "positive_rank": rank, "hr_at_5": hit,
            "ndcg_at_5": 1.0/math.log2(rank+1) if hit else 0.0, "top5": items[:k], "top10": items}

def mostpop(candidates: Sequence[str], frequencies: Mapping[str, int], k: int = 5) -> list[str]:
    """Rank by descending train frequency, then ascending canonical ID as a Unicode string.

    IDs are the exact catalog internal_item_id strings (no integer coercion).
    Candidate order, hash iteration order and NumPy sorting cannot break ties.
    Duplicated IDs are deduplicated before ranking; empty candidates return [].
    """
    return sorted(set(map(str, candidates)), key=lambda item: (-int(frequencies.get(item, 0)), item))[:k]

def beyond_utility(parsed: Mapping, candidates: Sequence[str], target: str,
                   frequencies: Mapping[str, int], long_tail: set[str], k: int = 5) -> dict:
    """APLT=unique top-K tail hits/K; serendipity=HR*(1-MostPop hit), invalid-zero.

    An unavailable popularity support is an error, never a silently fabricated zero.
    """
    if not frequencies:
        raise ValueError("Popularity evidence is unavailable")
    reference = mostpop(candidates, frequencies, k)
    reference_hit = float(str(target) in reference)
    valid = bool(parsed["strict_valid"])
    return {"aplt_at_5": len(set(parsed["top5"]) & long_tail)/k if valid else 0.0,
            "serendipity_at_5": float(parsed["hr_at_5"])*(1-reference_hit) if valid else 0.0,
            "mostpop_hit_at_5": reference_hit, "mostpop_top5_item_ids_json": json.dumps(reference, separators=(",", ":"))}

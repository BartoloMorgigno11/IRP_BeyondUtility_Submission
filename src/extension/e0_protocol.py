"""Protocol constants and stable hashing for Extension Phase E0.

This module is offline by construction: it contains no model client and no
Ollama/API entry point.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any


EXPERIMENT_ID = "extension_e0_protocol_freeze_v1"
TASK = "ranking"
PROMPT_BASE = "BASE"
PROMPT_LLMRANK = "LLMRANK"
PROMPT_STRATEGIES = (PROMPT_BASE, PROMPT_LLMRANK)
HISTORY_PROMPT_STRATEGY = PROMPT_BASE

PARSER_CONTRACT_ID = "llm_rank_strict_index_top10_v1"
PROMPT_CONTRACT_BASE_ID = "extension_base_strict_index_top10_v1"
PROMPT_CONTRACT_LLMRANK_ID = "extension_llmrank_strict_index_top10_v1"

CANDIDATE_COUNT = 20
POSITIVE_COUNT = 1
NEGATIVE_COUNT = 19
REQUESTED_OUTPUT_LENGTH = 10
METRIC_K = 5
POSITION_BASE = 1
POSITION_DOMAIN = tuple(range(1, CANDIDATE_COUNT + 1))
HISTORY_LENGTHS = (0, 1, 2, 5, 10, 20)

FUTURE_MODELS = (
    "qwen2:7b-instruct",
    "mistral:7b-instruct",
    "gemma2:9b",
)
PRIMARY_DATASETS = ("beauty", "sports", "foodcom")
HISTORY_DATASETS = ("beauty", "sports", "ml1m", "lastfm")
ACTIVE_EXTENSION_DATASETS = ("beauty", "sports", "ml1m", "lastfm", "foodcom")

EVIDENCE_CLASS_ADAPTED = "strict_index_adaptation_not_paper_exact"
GEMMA_STATUS = "gemma2:9b = REQUIRED_BEFORE_PILOT"


def stable_json(value: Any) -> str:
    """Return deterministic compact JSON for hashing and CSV manifests."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    """Return a lowercase SHA256 hex digest for text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-serializable value after deterministic serialization."""

    return sha256_text(stable_json(value))


def stable_int_seed(*parts: Any) -> int:
    """Derive a deterministic integer seed from semantic identifiers."""

    digest = sha256_text("|".join(str(part) for part in parts))
    return int(digest[:16], 16)


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def item_list_hash(item_ids: Iterable[Any]) -> str:
    return sha256_json([str(item_id) for item_id in item_ids])


def unordered_item_set_hash(item_ids: Iterable[Any]) -> str:
    return sha256_json(sorted(str(item_id) for item_id in item_ids))


def protocol_rows() -> list[dict[str, str]]:
    """Return frozen protocol rows for report and dry-run outputs."""

    rows = [
        {
            "contract_type": "primary_ranking",
            "contract_id": EXPERIMENT_ID,
            "task": TASK,
            "candidate_count": str(CANDIDATE_COUNT),
            "candidate_composition": "1_positive_19_negatives",
            "candidate_position_convention": "one_based_1_to_20",
            "requested_output_length": str(REQUESTED_OUTPUT_LENGTH),
            "parser_expected_output_length": str(REQUESTED_OUTPUT_LENGTH),
            "metric_k": str(METRIC_K),
            "prompt_families": ";".join(PROMPT_STRATEGIES),
            "parser_contract": PARSER_CONTRACT_ID,
            "malformed_response_policy": "fail_strict_invalid_scores_zero",
            "paper_alignment": EVIDENCE_CLASS_ADAPTED,
        },
        {
            "contract_type": "history_cold_start",
            "contract_id": f"{EXPERIMENT_ID}_history",
            "task": TASK,
            "candidate_count": str(CANDIDATE_COUNT),
            "candidate_composition": "same_candidate_set_across_lengths",
            "candidate_position_convention": "one_based_1_to_20",
            "requested_output_length": str(REQUESTED_OUTPUT_LENGTH),
            "parser_expected_output_length": str(REQUESTED_OUTPUT_LENGTH),
            "metric_k": str(METRIC_K),
            "prompt_families": HISTORY_PROMPT_STRATEGY,
            "parser_contract": PARSER_CONTRACT_ID,
            "malformed_response_policy": "fail_strict_invalid_scores_zero",
            "paper_alignment": EVIDENCE_CLASS_ADAPTED,
        },
    ]
    for row in rows:
        row["contract_sha256"] = sha256_json(row)
    return rows


def semantic_hash_for_row(row: Mapping[str, Any], *, exclude: Iterable[str] = ()) -> str:
    excluded = set(exclude)
    return sha256_json({str(k): row[k] for k in sorted(row) if str(k) not in excluded})

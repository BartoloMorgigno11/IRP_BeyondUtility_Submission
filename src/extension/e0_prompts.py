"""Prompt rendering for the strict-index extension protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.extension.e0_protocol import (
    CANDIDATE_COUNT,
    PROMPT_BASE,
    PROMPT_CONTRACT_BASE_ID,
    PROMPT_CONTRACT_LLMRANK_ID,
    PROMPT_LLMRANK,
    REQUESTED_OUTPUT_LENGTH,
    sha256_text,
)


SYSTEM_MESSAGE = (
    "You are a recommender-system evaluator. Return only valid JSON. "
    "Do not explain your reasoning."
)


@dataclass(frozen=True)
class PromptBundle:
    prompt_strategy: str
    prompt_contract: str
    system_message: str
    user_message: str
    prompt_hash: str


def render_extension_prompt(
    *,
    prompt_strategy: str,
    dataset_label: str,
    candidate_items: Sequence[Mapping[str, Any]],
    history_items: Sequence[Mapping[str, Any]],
    history_length: int,
) -> PromptBundle:
    """Render a BASE or LLMRank strict-index prompt without item IDs."""

    strategy = prompt_strategy.upper()
    if strategy not in {PROMPT_BASE, PROMPT_LLMRANK}:
        raise ValueError("prompt_strategy must be BASE or LLMRANK")
    if len(candidate_items) != CANDIDATE_COUNT:
        raise ValueError(f"expected {CANDIDATE_COUNT} candidates")

    history_render = _history_render(history_items)
    candidate_render = _candidate_render(candidate_items)
    output_instruction = (
        f"Return exactly {REQUESTED_OUTPUT_LENGTH} unique candidate positions as JSON: "
        '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,10]}'
    )

    if strategy == PROMPT_BASE:
        contract_id = PROMPT_CONTRACT_BASE_ID
        body = (
            f"Task: rank candidate items for the user's next interaction in {dataset_label}.\n"
            "Use only the visible train-history items and the candidate list.\n"
            "Candidate positions are 1-based indices from 1 to 20. Do not use item IDs or titles in the answer.\n"
            f"{output_instruction}\n\n"
            f"Visible history length: {history_length}\n"
            f"User history:\n{history_render}\n\n"
            f"Candidates:\n{candidate_render}"
        )
    else:
        contract_id = PROMPT_CONTRACT_LLMRANK_ID
        recency_sentence = _recency_sentence(history_items)
        body = (
            f"Task: rerank the candidate list for the user's next interaction in {dataset_label}.\n"
            "The candidate order below is the input order to evaluate and rerank.\n"
            f"{recency_sentence}\n"
            "Candidate positions are 1-based indices from 1 to 20. Do not use item IDs or titles in the answer.\n"
            f"{output_instruction}\n\n"
            f"Visible history length: {history_length}\n"
            f"User history:\n{history_render}\n\n"
            f"Candidates:\n{candidate_render}"
        )

    prompt_hash = sha256_text(SYSTEM_MESSAGE + "\n---\n" + body)
    return PromptBundle(strategy, contract_id, SYSTEM_MESSAGE, body, prompt_hash)


def _history_render(history_items: Sequence[Mapping[str, Any]]) -> str:
    if not history_items:
        return "(no visible user history)"
    lines = []
    for position, item in enumerate(history_items, start=1):
        title = str(item.get("title", "")).strip()
        if not title:
            raise ValueError("history item missing title")
        lines.append(f"{position}. {title}")
    return "\n".join(lines)


def _candidate_render(candidate_items: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for expected_position, item in enumerate(candidate_items, start=1):
        title = str(item.get("title") or item.get("candidate_item_title") or "").strip()
        position = int(item.get("candidate_position", expected_position))
        if position != expected_position:
            raise ValueError("candidate positions must be ordered 1..20")
        if not title:
            raise ValueError("candidate item missing title")
        lines.append(f"{position}. {title}")
    return "\n".join(lines)


def _recency_sentence(history_items: Sequence[Mapping[str, Any]]) -> str:
    if not history_items:
        return "No recency signal is available because no user history is visible."
    title = str(history_items[-1].get("title", "")).strip()
    if not title:
        raise ValueError("history item missing title")
    return f"Give extra attention to the most recent visible interaction: {title}."

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


SUPPORTED_PROMPT_STRATEGIES = ("openp5_base", "llmrank_recency", "llmsrec_icl_stub")


@dataclass(frozen=True)
class PromptItem:
    """Minimal item representation for LLM prompt construction."""

    item_id: str
    title: str
    metadata: str = ""


@dataclass(frozen=True)
class PromptTemplateSpec:
    prompt_strategy: str
    description: str
    uses_user_history: bool
    uses_candidate_list: bool
    uses_item_metadata: bool
    uses_in_context_examples: bool
    output_format: str
    intended_phase: str
    status: str
    notes: str


def prompt_template_registry_rows() -> list[dict[str, str]]:
    """Return the deterministic D5D prompt-template registry rows."""

    specs = [
        PromptTemplateSpec(
            prompt_strategy="openp5_base",
            description="Paper-inspired direct candidate ranking prompt using history and candidate metadata.",
            uses_user_history=True,
            uses_candidate_list=True,
            uses_item_metadata=True,
            uses_in_context_examples=False,
            output_format="strict_json",
            intended_phase="D5E_qwen2_ml1m_ranking_diagnostic",
            status="implemented",
            notes="first diagnostic template; asks model to rank only candidate IDs.",
        ),
        PromptTemplateSpec(
            prompt_strategy="llmrank_recency",
            description="Recency-focused candidate ranking variant inspired by LLMRank ordering sensitivity.",
            uses_user_history=True,
            uses_candidate_list=True,
            uses_item_metadata=True,
            uses_in_context_examples=False,
            output_format="strict_json",
            intended_phase="D5F_or_later_position_bias_and_ranking",
            status="implemented_or_stub",
            notes="recency-focused variant inspired by LLMRank; can emphasize recent history order.",
        ),
        PromptTemplateSpec(
            prompt_strategy="llmsrec_icl_stub",
            description="Documented in-context-learning placeholder for a future LLMSRec-style prompt.",
            uses_user_history=True,
            uses_candidate_list=True,
            uses_item_metadata=True,
            uses_in_context_examples=True,
            output_format="strict_json",
            intended_phase="optional_future",
            status="stub_documented",
            notes="documented placeholder for ICL; not used in D5E unless examples are implemented.",
        ),
    ]
    return [
        {
            "prompt_strategy": spec.prompt_strategy,
            "description": spec.description,
            "uses_user_history": _bool_text(spec.uses_user_history),
            "uses_candidate_list": _bool_text(spec.uses_candidate_list),
            "uses_item_metadata": _bool_text(spec.uses_item_metadata),
            "uses_in_context_examples": _bool_text(spec.uses_in_context_examples),
            "output_format": spec.output_format,
            "intended_phase": spec.intended_phase,
            "status": spec.status,
            "notes": spec.notes,
        }
        for spec in specs
    ]


def build_ranking_prompt(
    *,
    user_id: str,
    history_items: Iterable[Mapping[str, object] | PromptItem],
    candidate_items: Iterable[Mapping[str, object] | PromptItem],
    top_k: int,
    prompt_strategy: str,
) -> str:
    """Build a strict-JSON LLM ranking prompt from history and candidate items."""

    if prompt_strategy not in SUPPORTED_PROMPT_STRATEGIES:
        raise ValueError(f"Unsupported prompt_strategy: {prompt_strategy}")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    history = [_coerce_prompt_item(item) for item in history_items]
    candidates = [_coerce_prompt_item(item) for item in candidate_items]
    if not history:
        raise ValueError("history_items must not be empty")
    if not candidates:
        raise ValueError("candidate_items must not be empty")
    if top_k > len(candidates):
        raise ValueError("top_k cannot exceed candidate count")

    history_for_prompt = list(reversed(history)) if prompt_strategy == "llmrank_recency" else history
    history_heading = (
        "User history, most recent first:"
        if prompt_strategy == "llmrank_recency"
        else "User history in chronological order:"
    )
    strategy_note = _strategy_note(prompt_strategy)
    output_schema = _output_schema(top_k)

    return f"""You are evaluating candidate items for a recommender-system ranking task.
Use the user's prior interactions and the provided candidate list to rank items for user `{user_id}`.
{strategy_note}

{history_heading}
{_format_items(history_for_prompt)}

Candidate items:
{_format_items(candidates)}

Return only valid JSON.
Rank exactly {top_k} item IDs.
Use only candidate IDs from the provided candidate list.
Do not invent item IDs.
Do not include explanations.
If uncertain, still choose from the candidate IDs.

Required JSON schema:
{output_schema}
"""


def _strategy_note(prompt_strategy: str) -> str:
    if prompt_strategy == "openp5_base":
        return "Template strategy: openp5_base direct candidate ranking."
    if prompt_strategy == "llmrank_recency":
        return "Template strategy: llmrank_recency, with recent user history emphasized first."
    return (
        "Template strategy: llmsrec_icl_stub. In-context examples are documented for future work, "
        "but no examples are active in this D5D manifest."
    )


def _format_items(items: list[PromptItem]) -> str:
    lines: list[str] = []
    for item in items:
        metadata = f" | {item.metadata}" if item.metadata else ""
        title = item.title if item.title else item.item_id
        lines.append(f"- {item.item_id}: {title}{metadata}")
    return "\n".join(lines)


def _output_schema(top_k: int) -> str:
    placeholders = ", ".join(f'"item_id_{index}"' for index in range(1, top_k + 1))
    return "{\n" f'  "ranked_item_ids": [{placeholders}],\n' '  "status": "ok"\n}'


def _coerce_prompt_item(item: Mapping[str, object] | PromptItem) -> PromptItem:
    if isinstance(item, PromptItem):
        return item
    item_id = str(item.get("item_id", "")).strip()
    if not item_id:
        raise ValueError("Prompt item is missing item_id")
    title = str(item.get("title", item_id)).strip() or item_id
    metadata = str(item.get("metadata", "")).strip()
    return PromptItem(item_id=item_id, title=title, metadata=metadata)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"

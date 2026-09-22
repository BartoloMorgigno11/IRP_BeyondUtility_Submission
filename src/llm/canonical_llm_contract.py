"""Canonical LLM prompt, parser, and execution contract helpers for Phase 5A.

The module is intentionally offline. It defines deterministic contracts and
strict parsing utilities, but it does not call an LLM, an API, or Ollama.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


PROMPT_CONTRACT_COLUMNS = [
    "prompt_contract_id",
    "task_type",
    "prompt_stage",
    "system_message_template",
    "user_message_template",
    "history_source_policy",
    "history_order_policy",
    "history_length_policy",
    "history_item_representation",
    "candidate_representation",
    "candidate_numbering",
    "candidate_order_policy",
    "positive_label_hidden",
    "requested_output_count",
    "requested_output_representation",
    "explanation_allowed",
    "code_fence_allowed",
    "paper_source",
    "official_code_source",
    "local_source",
    "paper_alignment",
    "contract_status",
    "blocking_reason",
    "prompt_template_sha256",
    "contract_sha256",
    "notes",
]

PARSER_CONTRACT_COLUMNS = [
    "parser_contract_id",
    "task_type",
    "parser_mode",
    "expected_input_grammar",
    "expected_identifier_namespace",
    "expected_output_count",
    "requires_full_permutation",
    "allows_leading_prose",
    "allows_trailing_prose",
    "allows_code_fence",
    "allows_integer_strings",
    "index_base",
    "allows_duplicate_items",
    "allows_unknown_items",
    "allows_partial_output",
    "allows_fuzzy_title_matching",
    "normalization_policy",
    "invalid_output_policy",
    "candidate_membership_policy",
    "paper_alignment",
    "contract_status",
    "blocking_reason",
    "parser_schema_sha256",
    "notes",
]

MODEL_INVENTORY_COLUMNS = [
    "model_inventory_id",
    "model_family",
    "requested_model_identifier",
    "provider_or_backend",
    "open_or_closed",
    "paper_model",
    "local_model_tag",
    "local_manifest_or_digest",
    "local_availability_evidence",
    "previous_execution_evidence",
    "structured_output_evidence",
    "seed_support_evidence",
    "paper_alignment",
    "availability_status",
    "blocking_reason",
    "allowed_for_future_execution",
    "notes",
]

EXECUTION_PARAMETER_COLUMNS = [
    "execution_contract_id",
    "model_family",
    "task_type",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "max_output_tokens",
    "context_limit",
    "stop_sequences",
    "response_format",
    "streaming",
    "concurrency",
    "timeout_seconds",
    "infrastructure_retry_count",
    "scientific_retry_count",
    "request_order_policy",
    "raw_response_retention",
    "parameter_source",
    "paper_alignment",
    "contract_status",
    "blocking_reason",
    "contract_sha256",
    "notes",
]

NONDETERMINISM_COLUMNS = [
    "policy_id",
    "execution_scope",
    "model_family",
    "cohort_policy",
    "repetitions_per_row",
    "parameter_identity_required",
    "prompt_identity_required",
    "response_hash_required",
    "parse_validity_metric",
    "candidate_validity_metric",
    "ranked_list_agreement_metric",
    "top5_agreement_metric",
    "positive_rank_variance_metric",
    "utility_variance_metric",
    "result_independent",
    "paper_alignment",
    "recommendation_status",
    "blocking_reason",
    "notes",
]

POLICY_OPTION_COLUMNS = [
    "ambiguity_category",
    "policy_id",
    "policy_name",
    "description",
    "paper_evidence",
    "official_code_evidence",
    "local_evidence",
    "preserves_candidates",
    "preserves_prompt_identity",
    "uses_positive_label",
    "result_dependent",
    "changes_paper_definition",
    "comparability_impact",
    "methodological_risk",
    "allowed_for_future_execution",
    "recommendation_status",
    "rationale",
    "stop_condition",
    "notes",
]

ALLOWED_PROMPT_STATUSES = {"ready_adapted", "blocked_ambiguous", "diagnostic_only"}
ALLOWED_MODEL_STATUSES = {
    "paper_model_exact_identifier_verified",
    "paper_model_identifier_ambiguous",
    "local_model_available_and_tested",
    "local_model_available_not_tested",
    "conditional_remote_api",
    "blocked_retired_or_unavailable",
    "blocked_hardware",
    "blocked_missing_model",
    "adapted_extension_model",
    "not_applicable",
}

SYSTEM_MESSAGE_TEMPLATE = (
    "You are a recommender-system evaluator. Follow the user message exactly. "
    "Return only valid JSON. Do not explain your reasoning."
)

RANKING_USER_TEMPLATE = (
    "Task: rank candidate items for the user's next interaction.\n"
    "History is train-only and chronological.\n"
    "Candidate labels are 1-based indices. Do not use item IDs in the answer.\n"
    "Return exactly 5 unique candidate indices as JSON: "
    '{{"ranked_candidate_indices":[1,2,3,4,5]}}.\n\n'
    "User history:\n{history_render}\n\nCandidates:\n{candidate_render}"
)

RERANKING_USER_TEMPLATE = (
    "Task: rerank the supplied 20 candidate items from most to least likely.\n"
    "The input candidate order is the persisted initial ranking to be reranked.\n"
    "Candidate labels are 1-based indices. Do not use item IDs in the answer.\n"
    "Return all 20 unique candidate indices as JSON: "
    '{{"ranked_candidate_indices":[1,2,...,20]}}.\n\n'
    "User history:\n{history_render}\n\nInitial candidate order:\n{candidate_render}"
)


@dataclass(frozen=True)
class StrictParseResult:
    parse_valid: bool
    output_indices: tuple[int, ...]
    output_items: tuple[str, ...]
    duplicate_count: int
    unknown_item_count: int
    output_length: int
    status: str
    transformations: tuple[str, ...] = ()
    error: str = ""


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_items(items: Iterable[Any]) -> str:
    return sha256_text(json_dumps([str(item) for item in items]))


def contract_hash(row: Mapping[str, Any], excluded: Sequence[str] = ("contract_sha256",)) -> str:
    payload = {str(k): str(v) for k, v in row.items() if str(k) not in set(excluded)}
    return sha256_text(json_dumps(payload))


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def prompt_contract_rows() -> list[dict[str, str]]:
    rows = [
        {
            "prompt_contract_id": "llm_rank_index_json_top5_v1",
            "task_type": "ranking",
            "prompt_stage": "recommendation",
            "system_message_template": SYSTEM_MESSAGE_TEMPLATE,
            "user_message_template": RANKING_USER_TEMPLATE,
            "history_source_policy": "train_split_only_exclude_heldout_positive",
            "history_order_policy": "chronological_timestamp_then_source_order",
            "history_length_policy": "last_10_train_interactions_after_leakage_check",
            "history_item_representation": "numbered_title_only_from_item_title_map",
            "candidate_representation": "one_based_index_plus_title",
            "candidate_numbering": "one_based_1_to_20",
            "candidate_order_policy": "preserve_source_candidate_order",
            "positive_label_hidden": "true",
            "requested_output_count": "5",
            "requested_output_representation": "strict_json_one_based_candidate_indices",
            "explanation_allowed": "false",
            "code_fence_allowed": "false",
            "paper_source": "original_repo/eva/templates/template.txt requests top 10 titles",
            "official_code_source": "original_repo/eva/codes/llms.py system message requests python list of titles",
            "local_source": "src/llm/plan_d_prompt_templates.py uses strict_json item-id diagnostics",
            "paper_alignment": "adapted_project_policy",
            "contract_status": "ready_adapted",
            "blocking_reason": "",
            "notes": "Adapted strict index output for machine-checkable canonical execution.",
        },
        {
            "prompt_contract_id": "llm_rerank_index_json_full20_v1",
            "task_type": "reranking",
            "prompt_stage": "recommendation",
            "system_message_template": SYSTEM_MESSAGE_TEMPLATE,
            "user_message_template": RERANKING_USER_TEMPLATE,
            "history_source_policy": "train_split_only_exclude_heldout_positive",
            "history_order_policy": "chronological_timestamp_then_source_order",
            "history_length_policy": "last_10_train_interactions_after_leakage_check",
            "history_item_representation": "numbered_title_only_from_item_title_map",
            "candidate_representation": "one_based_index_plus_title",
            "candidate_numbering": "one_based_1_to_20",
            "candidate_order_policy": "preserve_source_initial_candidate_order",
            "positive_label_hidden": "true",
            "requested_output_count": "20",
            "requested_output_representation": "strict_json_full_permutation_one_based_candidate_indices",
            "explanation_allowed": "false",
            "code_fence_allowed": "false",
            "paper_source": "original_repo/eva/templates/llmrank.txt requests ranking all 20 candidates",
            "official_code_source": "original_repo/eva/codes/llms.py uses title-list output",
            "local_source": "Plan D reranking diagnostics use strict candidate IDs",
            "paper_alignment": "adapted_project_policy",
            "contract_status": "ready_adapted",
            "blocking_reason": "",
            "notes": "Full permutation preserves reranking semantics and allows top-5 evaluation downstream.",
        },
        {
            "prompt_contract_id": "llm_profile_generation_audit_only_v1",
            "task_type": "profile_generation",
            "prompt_stage": "profile_generation",
            "system_message_template": "",
            "user_message_template": "official template asks for a user profile from history",
            "history_source_policy": "train_split_only_not_execution_ready",
            "history_order_policy": "chronological",
            "history_length_policy": "blocked_until_profile_scope_approved",
            "history_item_representation": "titles_when_available",
            "candidate_representation": "not_applicable",
            "candidate_numbering": "not_applicable",
            "candidate_order_policy": "not_applicable",
            "positive_label_hidden": "true",
            "requested_output_count": "",
            "requested_output_representation": "free_text_profile",
            "explanation_allowed": "true",
            "code_fence_allowed": "false",
            "paper_source": "original_repo/eva/templates/template_user_profile_generate.txt",
            "official_code_source": "original_repo/eva/templates/template_profile*.txt",
            "local_source": "Plan D profile-related workflow evidence only",
            "paper_alignment": "blocked_ambiguous",
            "contract_status": "blocked_ambiguous",
            "blocking_reason": "profile prompt/model/reuse policy not frozen for canonical execution",
            "notes": "Included for audit only; no profile generation in Phase 5A.",
        },
    ]
    for row in rows:
        row["prompt_template_sha256"] = sha256_text(
            row["system_message_template"] + "\n---\n" + row["user_message_template"]
        )
        row["contract_sha256"] = contract_hash(row)
    return rows


def parser_contract_rows() -> list[dict[str, str]]:
    base_rows = [
        ("llm_rank_strict_primary_parser_v1", "ranking", "strict_primary_parser", "5", "false"),
        ("llm_rerank_strict_primary_parser_v1", "reranking", "strict_primary_parser", "20", "true"),
        ("llm_rank_diagnostic_salvage_parser_v1", "ranking", "diagnostic_salvage_parser", "5", "false"),
        ("llm_rerank_diagnostic_salvage_parser_v1", "reranking", "diagnostic_salvage_parser", "20", "true"),
    ]
    rows: list[dict[str, str]] = []
    for parser_id, task_type, mode, expected_count, full_permutation in base_rows:
        strict = mode == "strict_primary_parser"
        row = {
            "parser_contract_id": parser_id,
            "task_type": task_type,
            "parser_mode": mode,
            "expected_input_grammar": '{"ranked_candidate_indices":[int,...]}',
            "expected_identifier_namespace": "one_based_candidate_index",
            "expected_output_count": expected_count,
            "requires_full_permutation": full_permutation,
            "allows_leading_prose": bool_text(not strict),
            "allows_trailing_prose": bool_text(not strict),
            "allows_code_fence": bool_text(not strict),
            "allows_integer_strings": bool_text(not strict),
            "index_base": "1",
            "allows_duplicate_items": "false",
            "allows_unknown_items": "false",
            "allows_partial_output": "false",
            "allows_fuzzy_title_matching": "false",
            "normalization_policy": (
                "strip_code_fences_and_extract_first_json_object_for_diagnostic_only"
                if not strict
                else "strip_outer_whitespace_only"
            ),
            "invalid_output_policy": "mark_invalid_no_repair_no_metric_fabrication",
            "candidate_membership_policy": "all_indices_must_map_to_supplied_candidates",
            "paper_alignment": "adapted_project_policy",
            "contract_status": "ready_adapted",
            "blocking_reason": "",
            "notes": "Salvage parser is diagnostic only and never overwrites strict primary output."
            if not strict
            else "Canonical scientific parser.",
        }
        row["parser_schema_sha256"] = sha256_text(json_dumps({k: row[k] for k in row if k != "notes"}))
        rows.append(row)
    return rows


def render_history(history_items: Sequence[Mapping[str, str]], max_items: int = 10) -> str:
    selected = list(history_items)[-max_items:]
    if not selected:
        raise ValueError("history is empty")
    lines = []
    for index, item in enumerate(selected, start=1):
        title = str(item.get("title", "")).strip()
        item_id = str(item.get("item_id", "")).strip()
        if not item_id or not title:
            raise ValueError("history item missing id or title")
        lines.append(f"{index}. {title}")
    return "\n".join(lines)


def render_candidates(candidate_items: Sequence[Mapping[str, str]]) -> str:
    if len(candidate_items) != 20:
        raise ValueError("canonical LLM candidate prompts require exactly 20 candidates")
    lines = []
    for index, item in enumerate(candidate_items, start=1):
        item_id = str(item.get("item_id", "")).strip()
        title = str(item.get("title", "")).strip()
        if not item_id or not title:
            raise ValueError("candidate item missing id or title")
        lines.append(f"{index}. {title}")
    return "\n".join(lines)


def render_prompt(
    *,
    task_type: str,
    history_items: Sequence[Mapping[str, str]],
    candidate_items: Sequence[Mapping[str, str]],
) -> tuple[str, str, str, str]:
    history_render = render_history(history_items)
    candidate_render = render_candidates(candidate_items)
    contract = {row["task_type"]: row for row in prompt_contract_rows()}[task_type]
    user_message = contract["user_message_template"].format(
        history_render=history_render,
        candidate_render=candidate_render,
    )
    system_message = contract["system_message_template"]
    return system_message, user_message, history_render, candidate_render


def parse_strict_index_response(
    response_text: str,
    *,
    candidate_items: Sequence[str],
    expected_count: int,
    require_full_permutation: bool = False,
) -> StrictParseResult:
    text = str(response_text or "").strip()
    if not text:
        return _parse_failure("empty_output", "empty output")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return _parse_failure("parse_failed", str(exc))
    return _validate_index_payload(
        parsed,
        candidate_items=candidate_items,
        expected_count=expected_count,
        require_full_permutation=require_full_permutation,
        transformations=(),
    )


def parse_diagnostic_salvage_response(
    response_text: str,
    *,
    candidate_items: Sequence[str],
    expected_count: int,
    require_full_permutation: bool = False,
) -> StrictParseResult:
    text = str(response_text or "")
    transformations: list[str] = []
    cleaned = text.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
        transformations.append("removed_code_fence")
    if not cleaned.startswith("{") or not cleaned.endswith("}"):
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            cleaned = match.group(0)
            transformations.append("extracted_json_object")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return _parse_failure("parse_failed", str(exc), transformations=tuple(transformations))
    if isinstance(parsed, dict) and isinstance(parsed.get("ranked_candidate_indices"), list):
        converted = []
        changed = False
        for value in parsed["ranked_candidate_indices"]:
            if isinstance(value, str) and value.isdigit():
                converted.append(int(value))
                changed = True
            else:
                converted.append(value)
        if changed:
            parsed = dict(parsed)
            parsed["ranked_candidate_indices"] = converted
            transformations.append("converted_integer_strings")
    return _validate_index_payload(
        parsed,
        candidate_items=candidate_items,
        expected_count=expected_count,
        require_full_permutation=require_full_permutation,
        transformations=tuple(transformations),
    )


def _validate_index_payload(
    parsed: Any,
    *,
    candidate_items: Sequence[str],
    expected_count: int,
    require_full_permutation: bool,
    transformations: tuple[str, ...],
) -> StrictParseResult:
    if not isinstance(parsed, dict):
        return _parse_failure("invalid_schema", "top-level JSON must be an object", transformations=transformations)
    values = parsed.get("ranked_candidate_indices")
    if not isinstance(values, list):
        return _parse_failure("invalid_schema", "ranked_candidate_indices must be a list", transformations=transformations)
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return _parse_failure("invalid_schema", "all indices must be integers", transformations=transformations)
    output_length = len(values)
    if output_length != expected_count:
        return StrictParseResult(
            False,
            tuple(values),
            tuple(),
            duplicate_count=max(0, output_length - len(set(values))),
            unknown_item_count=sum(1 for value in values if value < 1 or value > len(candidate_items)),
            output_length=output_length,
            status="invalid_length",
            transformations=transformations,
            error=f"expected {expected_count} indices, observed {output_length}",
        )
    duplicate_count = output_length - len(set(values))
    unknown_count = sum(1 for value in values if value < 1 or value > len(candidate_items))
    if duplicate_count:
        status = "duplicate_indices"
    elif unknown_count:
        status = "unknown_indices"
    elif require_full_permutation and set(values) != set(range(1, len(candidate_items) + 1)):
        status = "missing_permutation_indices"
    else:
        status = "parsed_valid"
    valid = status == "parsed_valid"
    mapped = tuple(str(candidate_items[value - 1]) for value in values if 1 <= value <= len(candidate_items))
    return StrictParseResult(
        valid,
        tuple(values),
        mapped,
        duplicate_count,
        unknown_count,
        output_length,
        status,
        transformations=transformations,
        error="" if valid else status,
    )


def _parse_failure(status: str, error: str, transformations: tuple[str, ...] = ()) -> StrictParseResult:
    return StrictParseResult(False, tuple(), tuple(), 0, 0, 0, status, transformations, error)


def estimate_token_envelope(text: str) -> tuple[int, int, int, str]:
    character_count = len(text)
    whitespace_count = len(text.split())
    estimated_min = max(1, (character_count + 5) // 6)
    estimated_max = max(estimated_min, (character_count + 2) // 3)
    return character_count, whitespace_count, estimated_min, estimated_max


def request_count(
    rows: int,
    *,
    repetitions_per_row: int = 1,
    position_permutations_per_row: int = 1,
    history_variants_per_row: int = 1,
    profile_generation_requests_per_row: int = 0,
    recommendation_requests_per_row: int = 1,
) -> tuple[int, int]:
    if rows < 0:
        raise ValueError("rows must be non-negative")
    multipliers = [
        repetitions_per_row,
        position_permutations_per_row,
        history_variants_per_row,
        max(1, recommendation_requests_per_row),
    ]
    if any(int(value) < 1 for value in multipliers[:3]) or recommendation_requests_per_row < 0:
        raise ValueError("request multipliers must be non-negative and repetition/permutation/history must be positive")
    recommendation_requests = rows * repetitions_per_row * position_permutations_per_row * history_variants_per_row * recommendation_requests_per_row
    profile_requests = rows * profile_generation_requests_per_row
    return recommendation_requests, recommendation_requests + profile_requests


def validate_one_recommended_policy_per_category(rows: Sequence[Mapping[str, str]]) -> None:
    recommended: dict[str, int] = {}
    for row in rows:
        if str(row.get("recommendation_status", "")) == "recommended":
            category = str(row.get("ambiguity_category", ""))
            recommended[category] = recommended.get(category, 0) + 1
            if str(row.get("uses_positive_label", "")) == "true":
                raise ValueError("recommended policy may not use positive labels")
            if str(row.get("result_dependent", "")) == "true":
                raise ValueError("recommended policy may not be result-dependent")
    categories = {str(row.get("ambiguity_category", "")) for row in rows}
    missing = sorted(category for category in categories if recommended.get(category, 0) != 1)
    if missing:
        raise ValueError(f"categories without exactly one recommendation: {missing}")

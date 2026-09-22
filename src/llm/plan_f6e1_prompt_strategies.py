"""Offline prompt-strategy specifications for Phase F6E1.

The module contains deterministic metadata and pure rendering helpers for the
prompt-strategy recovery audit. It deliberately avoids LLM clients, network
libraries, subprocess execution, and import-time writes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.audit.common_paths import path_as_posix as _common_path_as_posix


FROZEN_TIMESTAMP_UTC = "2026-07-25T00:00:00Z"
EXPECTED_BRANCH = "plan-d-extensions"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"

REPORT_PATH = Path("report/phase_f6e1_prompt_strategy_recovery.md")
MANIFEST_PATH = Path("results/manifests/phase_f6e1_prompt_freeze.json")
OUTPUT_TABLES: dict[str, Path] = {
    "prompt_source_inventory": Path("results/tables/phase_f6e1_prompt_source_inventory.csv"),
    "prompt_strategy_spec": Path("results/tables/phase_f6e1_prompt_strategy_spec.csv"),
    "placeholder_contract": Path("results/tables/phase_f6e1_placeholder_contract.csv"),
    "parser_contract": Path("results/tables/phase_f6e1_parser_contract.csv"),
    "model_strategy_matrix": Path("results/tables/phase_f6e1_model_strategy_matrix.csv"),
    "missing_or_blocked": Path("results/tables/phase_f6e1_missing_or_blocked.csv"),
}

BASE_STRATEGY_ID = "paper_base_openp5_parametric_v1"
LLMRANK_STRATEGY_ID = "paper_llmrank_recency_parametric_v1"
LLMSREC_STRATEGY_ID = "paper_llmsrec_reconstructed_v1"
JSON_INDEX_STRATEGY_ID = "local_json_index_ablation_v1"
BLOCKED_FULL_LLMSREC_ID = "blocked_full_llmsrec_profile_state_v1"

PAPER_DATASETS = ["Amazon All Beauty", "Sports & Outdoors", "MovieLens-1M", "LastFM"]
LOCAL_EXECUTION_DATASETS = ["Amazon Beauty", "Amazon Sports", "ML-1M", "LastFM"]

PAPER_CORE_MODELS = [
    ("Qwen2 7B Instruct", "qwen-2-7b-instruct", "qwen2:7b-instruct"),
    ("Mistral 7B Instruct", "mistral-7b-instruct-v0.2", "mistral:7b-instruct"),
    ("Llama3-70B-Instruct", "llama-3-70b-instruct", ""),
]

CANONICAL_PLACEHOLDERS = [
    "user_id",
    "user_history",
    "chronological_history",
    "reverse_chronological_history",
    "recent_history",
    "long_term_history",
    "candidate_items",
    "candidate_indices",
    "item_ids",
    "item_titles",
    "item_metadata",
    "expected_list_length",
    "task",
    "dataset",
    "profile",
    "reasoning_context",
    "output_schema",
]

SOURCE_INVENTORY_COLUMNS = [
    "source_id",
    "relative_path",
    "file_type",
    "role",
    "linked_strategy",
    "literal_text_present",
    "parser_present",
    "placeholders",
    "output_format",
    "model_association",
    "dataset_association",
    "task_association",
    "provenance",
    "evidence_strength",
    "sha256",
    "preliminary_classification",
    "notes",
]

STRATEGY_SPEC_COLUMNS = [
    "strategy_id",
    "strategy_name",
    "classification",
    "provenance",
    "evidence_strength",
    "source_paths",
    "template_variant_count",
    "literal_template_text_available",
    "system_prompt_available",
    "placeholder_names",
    "input_history_contract",
    "candidate_contract",
    "output_contract",
    "parser_contract",
    "model_applicability",
    "dataset_applicability",
    "task_applicability",
    "multi_step_structure",
    "llm_calls_per_request",
    "local_execution_status",
    "paper_alignment",
    "template_sha256",
    "parser_sha256",
    "notes",
]

PLACEHOLDER_CONTRACT_COLUMNS = [
    "strategy_id",
    "placeholder_name",
    "source_name",
    "type",
    "required",
    "order",
    "serialization",
    "allowed_values",
    "fallback",
    "source_path",
    "notes",
]

PARSER_CONTRACT_COLUMNS = [
    "strategy_id",
    "parser_contract_id",
    "output_format",
    "identifier_namespace",
    "index_base",
    "list_format",
    "json_required",
    "free_text_allowed",
    "expected_output_count",
    "requires_full_permutation",
    "allows_duplicates",
    "allows_unknown_items",
    "allows_partial_output",
    "allows_reasoning",
    "allows_code_fence",
    "recoverable_errors",
    "nonrecoverable_errors",
    "original_parser_source",
    "local_parser_source",
    "strict_validity_definition",
    "parse_validity_definition",
    "candidate_hallucination_definition",
    "duplicate_ranking_definition",
    "parser_sha256",
    "classification",
    "notes",
]

MODEL_STRATEGY_COLUMNS = [
    "matrix_id",
    "model_family",
    "expected_model_identifier",
    "local_model_identifier",
    "strategy_id",
    "strategy_name",
    "status",
    "scope",
    "exactness_classification",
    "prompt_available",
    "parser_available",
    "llm_calls_per_request",
    "expected_context_size",
    "local_feasibility",
    "inference_status",
    "blocker",
    "notes",
]

MISSING_OR_BLOCKED_COLUMNS = [
    "block_id",
    "strategy_id",
    "evidence_gap",
    "severity",
    "blocked_scope",
    "status",
    "source_paths",
    "safe_next_action",
    "notes",
]


@dataclass(frozen=True)
class PromptRenderContext:
    dataset: str
    user_name: str
    user_history_titles: tuple[str, ...]
    candidate_titles: tuple[str, ...]
    user_profile: str = ""


class BlockedPromptStrategyError(RuntimeError):
    """Raised when a requested prompt strategy is intentionally blocked."""


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256_or_missing(root: Path, relative_path: Path) -> str:
    path = root / relative_path
    if not path.exists() or not path.is_file():
        return "missing"
    return sha256_file(path)


def stable_id(prefix: str, *parts: Any) -> str:
    digest = sha256_text("||".join(str(part) for part in parts))[:12]
    return f"{prefix}-{digest}"


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def rel(path: Path) -> str:
    """Compatibility wrapper retained for F6I2I path_normalization provenance."""

    return _common_path_as_posix(path)


def read_text(root: Path, relative_path: Path) -> str:
    return (root / relative_path).read_text(encoding="utf-8", errors="replace")


def read_template_lines(root: Path, relative_path: Path) -> tuple[str, ...]:
    text = read_text(root, relative_path)
    return tuple(line for line in text.splitlines() if line.strip())


def extract_placeholders(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", text))))


def source_combined_hash(root: Path, paths: Sequence[Path]) -> str:
    payload = {rel(path): file_sha256_or_missing(root, path) for path in sorted(paths, key=rel)}
    return sha256_text(stable_json(payload))


def _python_list(values: Sequence[str]) -> str:
    return repr([str(value) for value in values])


def _require_context(context: PromptRenderContext, *, min_history: int = 1, candidates: int | None = 20) -> None:
    if not context.dataset.strip():
        raise ValueError("dataset is required")
    if not context.user_name.strip():
        raise ValueError("user_name is required")
    if len(context.user_history_titles) < min_history:
        raise ValueError(f"at least {min_history} history items are required")
    if candidates is not None and len(context.candidate_titles) != candidates:
        raise ValueError(f"exactly {candidates} candidate titles are required")


def render_strategy_prompt(
    strategy_id: str,
    context: PromptRenderContext,
    *,
    root: Path | None = None,
    variant_index: int = 0,
) -> str:
    """Render a supported paper/local prompt deterministically without inference."""

    root = Path.cwd() if root is None else root
    if strategy_id == BLOCKED_FULL_LLMSREC_ID:
        raise BlockedPromptStrategyError("Full persistent LLMSRec profile-state execution is blocked by missing local evidence.")
    if strategy_id == JSON_INDEX_STRATEGY_ID:
        return render_json_index_prompt(context=context, task_type="ranking", top_k=5)["user_message"]

    if strategy_id == BASE_STRATEGY_ID:
        _require_context(context)
        template_path = Path("original_repo/EvaLLMasRecommender/eva/templates/template.txt")
        template = _select_template(root, template_path, variant_index)
        return _fill_template(
            template,
            {
                "dataset": context.dataset,
                "user_name": context.user_name,
                "user_history": _python_list(context.user_history_titles),
                "candidate_list": _python_list(context.candidate_titles),
            },
        )

    if strategy_id == LLMRANK_STRATEGY_ID:
        _require_context(context)
        template_path = Path("original_repo/EvaLLMasRecommender/eva/templates/llmrank_rf.txt")
        template = _select_template(root, template_path, 0)
        return _fill_template(
            template,
            {
                "user_history": _python_list(context.user_history_titles),
                "candidate_list": _python_list(context.candidate_titles),
                "last_item": _python_list(context.user_history_titles[-1:]),
            },
        )

    if strategy_id == LLMSREC_STRATEGY_ID:
        _require_context(context, min_history=2)
        template_path = Path("original_repo/EvaLLMasRecommender/eva/templates/template_in_context.txt")
        template = _select_template(root, template_path, variant_index)
        return _fill_template(
            template,
            {
                "dataset": context.dataset,
                "user_name": context.user_name,
                "user_history": _python_list(context.user_history_titles),
                "last_history": _python_list(context.user_history_titles[:-1]),
                "last_item": _python_list(context.user_history_titles[-1:]),
                "candidate_list": _python_list(context.candidate_titles),
            },
        )

    raise KeyError(f"unknown F6E1 prompt strategy: {strategy_id}")


def _select_template(root: Path, relative_path: Path, variant_index: int) -> str:
    lines = read_template_lines(root, relative_path)
    if not lines:
        raise ValueError(f"template source is empty: {relative_path}")
    if variant_index < 0 or variant_index >= len(lines):
        raise IndexError(f"variant_index out of range for {relative_path}: {variant_index}")
    return lines[variant_index]


def _fill_template(template: str, values: Mapping[str, str]) -> str:
    rendered = template
    missing = [name for name in extract_placeholders(template) if name not in values]
    if missing:
        raise ValueError(f"missing placeholder values: {missing}")
    for key in sorted(values):
        rendered = rendered.replace("{" + key + "}", values[key])
    return rendered


def render_json_index_prompt(
    *,
    context: PromptRenderContext,
    task_type: str,
    top_k: int,
) -> dict[str, str]:
    """Render the local JSON/index ablation prompt without contacting a model."""

    _require_context(context)
    if task_type not in {"ranking", "reranking"}:
        raise ValueError("task_type must be ranking or reranking")
    if task_type == "ranking" and top_k != 5:
        raise ValueError("ranking JSON/index contract is frozen at top_k=5")
    if task_type == "reranking" and top_k not in {10, 20}:
        raise ValueError("reranking JSON/index contract is frozen at top_k=10 or top_k=20")

    history_render = "\n".join(f"{idx}. {title}" for idx, title in enumerate(context.user_history_titles[-10:], start=1))
    candidate_render = "\n".join(f"{idx}. {title}" for idx, title in enumerate(context.candidate_titles, start=1))
    system = (
        "You are a recommender-system evaluator. Follow the user message exactly. "
        "Return only valid JSON. Do not explain your reasoning."
    )
    if task_type == "ranking":
        user = (
            "Task: rank candidate items for the user's next interaction.\n"
            "History is train-only and chronological.\n"
            "Candidate labels are 1-based indices. Do not use item IDs in the answer.\n"
            'Return exactly 5 unique candidate indices as JSON: {"ranked_candidate_indices":[1,2,3,4,5]}.\n\n'
            f"User history:\n{history_render}\n\nCandidates:\n{candidate_render}"
        )
    else:
        template_count = "10" if top_k == 10 else "20"
        user = (
            "Task: rerank the supplied 20 candidate items for the user's next interaction.\n"
            "The input candidate order is the persisted initial ranking to be reranked.\n"
            "Candidate labels are one-based indices from 1 to 20. Do not use item IDs or titles in the answer.\n"
            f"Return exactly {template_count} unique candidate indices, ordered from most to least likely, "
            "as one JSON object with exactly one key named ranked_candidate_indices.\n"
            f"The JSON value must be an array of {template_count} integers. Every integer must be between 1 and 20.\n"
            "Do not explain. Do not include Markdown or extra fields.\n\n"
            f"User history:\n{history_render}\n\nInitial candidate order:\n{candidate_render}"
        )
    return {"system_message": system, "user_message": user}


def validate_json_index_output(
    response_text: str,
    *,
    candidate_count: int,
    expected_count: int,
    require_full_permutation: bool = False,
) -> dict[str, Any]:
    """Validate the frozen JSON/index output contract."""

    try:
        parsed = json.loads(response_text.strip())
    except json.JSONDecodeError:
        return _parse_status(False, (), "json_parse_failed", 0, 0)
    if not isinstance(parsed, dict):
        return _parse_status(False, (), "invalid_schema", 0, 0)
    values = parsed.get("ranked_candidate_indices")
    if not isinstance(values, list) or not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return _parse_status(False, (), "invalid_schema", 0, 0)
    indices = tuple(int(value) for value in values)
    duplicate_count = len(indices) - len(set(indices))
    unknown_count = sum(1 for value in indices if value < 1 or value > candidate_count)
    if len(indices) != expected_count:
        return _parse_status(False, indices, "invalid_length", duplicate_count, unknown_count)
    if duplicate_count:
        return _parse_status(False, indices, "duplicate_indices", duplicate_count, unknown_count)
    if unknown_count:
        return _parse_status(False, indices, "unknown_indices", duplicate_count, unknown_count)
    if require_full_permutation and set(indices) != set(range(1, candidate_count + 1)):
        return _parse_status(False, indices, "missing_permutation_indices", duplicate_count, unknown_count)
    return _parse_status(True, indices, "parsed_valid", 0, 0)


def _parse_status(valid: bool, indices: Sequence[int], status: str, duplicate_count: int, unknown_count: int) -> dict[str, Any]:
    return {
        "strict_parse_valid": valid,
        "output_indices": tuple(indices),
        "output_length": len(indices),
        "duplicate_count": duplicate_count,
        "unknown_item_count": unknown_count,
        "status": status,
        "index_base": 1,
    }


def strategy_definitions(root: Path) -> list[dict[str, str]]:
    base_sources = [
        Path("original_repo/EvaLLMasRecommender/eva/templates/template.txt"),
        Path("original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb"),
        Path("original_repo/EvaLLMasRecommender/eva/codes/llms.py"),
        Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py"),
    ]
    llmrank_sources = [
        Path("original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt"),
        Path("original_repo/EvaLLMasRecommender/eva/templates/llmrank_rf.txt"),
        Path("original_repo/EvaLLMasRecommender/eva/codes/metrics.sh"),
        Path("original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb"),
        Path("original_repo/EvaLLMasRecommender/eva/codes/llms.py"),
        Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py"),
    ]
    llmsrec_sources = [
        Path("original_repo/EvaLLMasRecommender/eva/templates/template_in_context.txt"),
        Path("original_repo/EvaLLMasRecommender/eva/templates/template_user_profile_generate.txt"),
        Path("original_repo/EvaLLMasRecommender/eva/templates/template_profile.txt"),
        Path("original_repo/EvaLLMasRecommender/eva/templates/template_profile_history.txt"),
        Path("original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb"),
        Path("original_repo/EvaLLMasRecommender/eva/codes/zero_shot.py"),
        Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py"),
    ]
    json_sources = [
        Path("src/llm/canonical_llm_contract.py"),
        Path("src/llm/plan_d_prompt_templates.py"),
        Path("results/tables/canonical_llm_prompt_contract.csv"),
        Path("results/tables/canonical_llm_parser_contract.csv"),
        Path("results/tables/finalization_f6d3_prompt_contract.csv"),
        Path("results/tables/canonical_local_llm_reranking_top10_prompt_contract.csv"),
        Path("results/tables/canonical_local_llm_reranking_top10_parser_contract.csv"),
    ]
    return [
        {
            "strategy_id": BASE_STRATEGY_ID,
            "strategy_name": "OpenP5-derived base prompt",
            "classification": "exact_parametric",
            "provenance": "official_code_literal",
            "evidence_strength": "official_code_literal",
            "source_paths": ";".join(rel(path) for path in base_sources),
            "template_variant_count": str(len(read_template_lines(root, base_sources[0]))),
            "literal_template_text_available": "true",
            "system_prompt_available": "true",
            "placeholder_names": "candidate_list;dataset;user_history;user_name",
            "input_history_contract": "last his_len history titles; chronological unless explicit shuffle ablation; target item excluded",
            "candidate_contract": "20 title candidates; target plus negatives; shuffled unless candidate-order ablation",
            "output_contract": "python list of item titles; top 10 requested by system/user prompt",
            "parser_contract": "original title-list parser in llm_metrics.py; not JSON/index",
            "model_applicability": "all paper LLM families; local future execution only after runtime approval",
            "dataset_applicability": ";".join(PAPER_DATASETS),
            "task_applicability": "ranking",
            "multi_step_structure": "single recommendation prompt",
            "llm_calls_per_request": "1",
            "local_execution_status": "not_run_in_f6e1",
            "paper_alignment": "paper_prompt_source_available_but_exact_OpenP5_origin_not_proven",
            "template_sha256": source_combined_hash(root, [base_sources[0]]),
            "parser_sha256": source_combined_hash(root, [base_sources[2], base_sources[3]]),
            "notes": "Do not equate OpenP5-derived with exact OpenP5 without separate proof.",
        },
        {
            "strategy_id": LLMRANK_STRATEGY_ID,
            "strategy_name": "LLMRank recency-focused",
            "classification": "exact_parametric",
            "provenance": "official_code_literal",
            "evidence_strength": "official_code_literal",
            "source_paths": ";".join(rel(path) for path in llmrank_sources),
            "template_variant_count": "1 recency-focused plus 1 non-rf template",
            "literal_template_text_available": "true",
            "system_prompt_available": "true",
            "placeholder_names": "candidate_list;last_item;user_history",
            "input_history_contract": "history title list plus explicit last_item recency reminder",
            "candidate_contract": "20 title candidates; rank given candidates only",
            "output_contract": "rank candidate items; upstream system prompt requests python list of 10 titles",
            "parser_contract": "original title-list parser in llm_metrics.py; Llama branch expects numbered text",
            "model_applicability": "paper Llama3 plus planned Qwen2/Mistral cross-model extension",
            "dataset_applicability": "paper evidence local templates are dataset-generic; metrics.sh shows Beauty LLMRank runs",
            "task_applicability": "ranking",
            "multi_step_structure": "single recency-focused recommendation prompt",
            "llm_calls_per_request": "1",
            "local_execution_status": "not_run_in_f6e1",
            "paper_alignment": "paper_prompt_source_available; exact paper run protocol still unexecuted locally",
            "template_sha256": source_combined_hash(root, [llmrank_sources[0], llmrank_sources[1]]),
            "parser_sha256": source_combined_hash(root, [llmrank_sources[4], llmrank_sources[5]]),
            "notes": "llmrank_rf.txt contains a product/movie wording inconsistency; preserve it as source evidence.",
        },
        {
            "strategy_id": LLMSREC_STRATEGY_ID,
            "strategy_name": "LLMSRec in-context/profile strategy",
            "classification": "reconstructed",
            "provenance": "official_code_literal_plus_local_audit",
            "evidence_strength": "official_code_literal",
            "source_paths": ";".join(rel(path) for path in llmsrec_sources),
            "template_variant_count": str(len(read_template_lines(root, llmsrec_sources[0]))) + " in-context variants; profile templates present separately",
            "literal_template_text_available": "true",
            "system_prompt_available": "true",
            "placeholder_names": "candidate_list;dataset;last_history;last_item;user_history;user_name;user_profile",
            "input_history_contract": "in-context prompt uses last history excluding last_item plus last_item; profile pipeline uses generated user_profile",
            "candidate_contract": "20 title candidates inherited from zero-shot prompt construction",
            "output_contract": "python list of item titles for recommendation step; free-text profile for profile step",
            "parser_contract": "original title-list parser for recommendation; no local exact parser for profile quality",
            "model_applicability": "paper Llama3 LLMSRec core plus planned Qwen2/Mistral cross-model extension",
            "dataset_applicability": ";".join(PAPER_DATASETS),
            "task_applicability": "ranking;profile_generation",
            "multi_step_structure": "one-step in-context component plus optional profile-generation/profile-injection pipeline",
            "llm_calls_per_request": "1 or 2 depending on LLMSRec interpretation",
            "local_execution_status": "not_run_in_f6e1",
            "paper_alignment": "literal components available; full canonical LLMSRec orchestration not proven",
            "template_sha256": source_combined_hash(root, llmsrec_sources[:4]),
            "parser_sha256": source_combined_hash(root, [llmsrec_sources[4], llmsrec_sources[5], llmsrec_sources[6]]),
            "notes": "F6E1 freezes both the recoverable in-context prompt and the missing full pipeline evidence.",
        },
        {
            "strategy_id": JSON_INDEX_STRATEGY_ID,
            "strategy_name": "Local JSON/index prompt",
            "classification": "local_ablation",
            "provenance": "local_reproduction",
            "evidence_strength": "local_reproduction",
            "source_paths": ";".join(rel(path) for path in json_sources),
            "template_variant_count": "ranking top5; reranking full20; reranking top10",
            "literal_template_text_available": "true",
            "system_prompt_available": "true",
            "placeholder_names": "candidate_render;history_render",
            "input_history_contract": "train-only chronological title rendering; last 10 where applicable",
            "candidate_contract": "20 candidates represented as one-based index plus title; persisted order",
            "output_contract": "strict JSON object with ranked_candidate_indices",
            "parser_contract": "local strict JSON/index parser; one-based indices; no fuzzy title matching",
            "model_applicability": "Qwen2 local; Mistral local",
            "dataset_applicability": ";".join(LOCAL_EXECUTION_DATASETS),
            "task_applicability": "ranking;reranking",
            "multi_step_structure": "single request; parser deterministic",
            "llm_calls_per_request": "1",
            "local_execution_status": "executed_before_f6e1_but_not_reexecuted_here",
            "paper_alignment": "adapted_local_ablation_not_exact_paper_prompt",
            "template_sha256": source_combined_hash(root, [json_sources[0], json_sources[2], json_sources[5]]),
            "parser_sha256": source_combined_hash(root, [json_sources[0], json_sources[3], json_sources[6]]),
            "notes": "Preserved as local ablation; never promoted to exact paper prompt.",
        },
    ]


def build_prompt_source_inventory(root: Path) -> list[dict[str, str]]:
    sources = [
        ("template_base", "original_repo/EvaLLMasRecommender/eva/templates/template.txt", "txt", "base prompt template variants", BASE_STRATEGY_ID, True, False, "python list of item titles", "generic paper LLMs", "Beauty;Sports;ML-1M;LastFM", "ranking", "original_repo", "official_code_literal", "exact_parametric", "Five direct history/candidate ranking variants."),
        ("template_candidate_first", "original_repo/EvaLLMasRecommender/eva/templates/template_candidate_first.txt", "txt", "candidate-first ablation template", BASE_STRATEGY_ID, True, False, "python list of item titles", "generic paper LLMs", "generic", "ranking", "original_repo", "official_code_literal", "adapted", "Ordering ablation source, not primary paper-aligned matrix cell."),
        ("template_wohistory", "original_repo/EvaLLMasRecommender/eva/templates/template_wohistory.txt", "txt", "no-history ablation template", BASE_STRATEGY_ID, True, False, "python list of item titles", "generic paper LLMs", "generic", "ranking", "original_repo", "official_code_literal", "adapted", "No-history prompt source."),
        ("template_llmrank", "original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt", "txt", "LLMRank non-rf template", LLMRANK_STRATEGY_ID, True, False, "ranked candidate text/list", "generic paper LLMs", "generic", "ranking", "original_repo", "official_code_literal", "exact_parametric", "LLMRank base source without explicit last_item."),
        ("template_llmrank_rf", "original_repo/EvaLLMasRecommender/eva/templates/llmrank_rf.txt", "txt", "LLMRank recency-focused template", LLMRANK_STRATEGY_ID, True, False, "ranked candidate text/list", "generic paper LLMs", "generic", "ranking", "original_repo", "official_code_literal", "exact_parametric", "Contains last_item recency reminder."),
        ("template_in_context", "original_repo/EvaLLMasRecommender/eva/templates/template_in_context.txt", "txt", "LLMSRec in-context template", LLMSREC_STRATEGY_ID, True, False, "python list of item titles", "generic paper LLMs", "generic", "ranking", "original_repo", "official_code_literal", "reconstructed", "One-step in-context component with last_history and last_item."),
        ("template_user_profile_generate", "original_repo/EvaLLMasRecommender/eva/templates/template_user_profile_generate.txt", "txt", "profile generation template", LLMSREC_STRATEGY_ID, True, False, "free-text profile", "generic paper LLMs", "generic", "profile_generation", "original_repo", "official_code_literal", "reconstructed", "Profile generation component for LLMSRec-like pipeline."),
        ("template_profile", "original_repo/EvaLLMasRecommender/eva/templates/template_profile.txt", "txt", "profile-only recommendation template", LLMSREC_STRATEGY_ID, True, False, "python list of item titles", "generic paper LLMs", "generic", "ranking", "original_repo", "official_code_literal", "reconstructed", "Recommendation using generated profile without history."),
        ("template_profile_history", "original_repo/EvaLLMasRecommender/eva/templates/template_profile_history.txt", "txt", "profile plus history recommendation template", LLMSREC_STRATEGY_ID, True, False, "python list of item titles", "generic paper LLMs", "generic", "ranking", "original_repo", "official_code_literal", "reconstructed", "Recommendation using generated profile and history."),
        ("upstream_generate_prompt", "original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb", "ipynb", "prompt construction notebook", "all_paper_prompt_strategies", True, False, "prompt JSON with target/user/cand_list/template", "generic paper LLMs", "Beauty;Sports", "ranking;profile_generation", "original_repo", "official_code_literal", "reconstructed", "Defines his_len truncation, candidate list, template sampling and profile injection."),
        ("upstream_llms", "original_repo/EvaLLMasRecommender/eva/codes/llms.py", "py", "LLM system prompt and API wrapper", "all_paper_prompt_strategies", True, False, "python list of item titles", "gpt;glm in code", "generic", "ranking", "original_repo", "official_code_literal", "exact_parametric", "System message requests 10 title outputs; module itself calls APIs but is not imported by F6E1."),
        ("upstream_llm_metrics", "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py", "py", "original output parser and evaluator", "all_paper_prompt_strategies", False, True, "python title list or numbered Llama output", "gpt;glm;llama branch", "Beauty;Sports", "ranking", "original_repo", "official_code_literal", "exact_semantic_only", "Parses titles to item ids and validates length/compliance."),
        ("upstream_metrics_script", "original_repo/EvaLLMasRecommender/eva/codes/metrics.sh", "sh", "declared run/evaluation paths", "all_paper_prompt_strategies", False, False, "not_applicable", "gpt3.5/gpt4/glm paths", "Beauty;Sports", "ranking", "original_repo", "official_description_only", "exact_semantic_only", "Shows profile, rerank, llmrank and history sensitivity result paths."),
        ("local_canonical_contract", "src/llm/canonical_llm_contract.py", "py", "local JSON/index prompt and parser contract", JSON_INDEX_STRATEGY_ID, True, True, "strict JSON one-based candidate indices", "qwen2;mistral", "Beauty;Sports;ML-1M;LastFM", "ranking;reranking", "local_reproduction", "local_reproduction", "local_ablation", "Canonical local strict parser and prompt templates."),
        ("local_plan_d_templates", "src/llm/plan_d_prompt_templates.py", "py", "earlier local prompt registry", JSON_INDEX_STRATEGY_ID, True, False, "strict_json item ids", "qwen2;mistral planned", "ML-1M diagnostics", "ranking", "local_reproduction", "local_reproduction", "adapted", "Earlier local prompt helper with OpenP5/LLMRank/LLMSRec labels."),
        ("local_prompt_contract_csv", "results/tables/canonical_llm_prompt_contract.csv", "csv", "frozen local prompt contracts", JSON_INDEX_STRATEGY_ID, True, False, "strict JSON one-based candidate indices", "qwen2;mistral", "Beauty;Sports;ML-1M", "ranking;reranking", "local_reproduction", "local_reproduction", "local_ablation", "Phase 5A contract table."),
        ("local_parser_contract_csv", "results/tables/canonical_llm_parser_contract.csv", "csv", "frozen local parser contracts", JSON_INDEX_STRATEGY_ID, False, True, "strict JSON one-based candidate indices", "qwen2;mistral", "Beauty;Sports;ML-1M", "ranking;reranking", "local_reproduction", "local_reproduction", "local_ablation", "Strict and diagnostic parser contracts."),
        ("f6d3_prompt_contract", "results/tables/finalization_f6d3_prompt_contract.csv", "csv", "LastFM LLM preflight prompt contract", JSON_INDEX_STRATEGY_ID, True, True, "strict JSON one-based candidate indices", "qwen2;mistral", "LastFM", "ranking;reranking", "local_reproduction", "local_reproduction", "local_ablation", "Confirms no prompt format change in F6D3."),
        ("f6a_prompt_taxonomy", "results/tables/finalization_f6a_prompt_taxonomy.csv", "csv", "taxonomy evidence", "all_paper_prompt_strategies", False, False, "taxonomy only", "paper LLMs;qwen2;mistral", "paper datasets", "ranking;reranking", "local_report", "local_reproduction", "exact_semantic_only", "States JSON/index cannot be promoted to exact prompt equivalence."),
        ("f6b_model_prompt_matrix", "results/tables/finalization_f6b_model_prompt_matrix.csv", "csv", "model-prompt matrix evidence", "all_paper_prompt_strategies", False, False, "taxonomy only", "paper LLMs;qwen2;mistral", "paper datasets", "ranking;reranking", "local_report", "local_reproduction", "exact_semantic_only", "Prior matrix distinguishes exactness and blockers."),
        ("paper_docx_translation", "paper/traduzione_italiana_beyond_utility_llm_recommender.docx", "docx", "secondary paper translation", "all_paper_prompt_strategies", False, False, "paper description", "paper LLMs", "paper datasets", "ranking;reranking", "paper_translation_local", "official_description_only", "exact_semantic_only", "Secondary local source; not treated as literal template source."),
        ("paper_pdf", "paper/2411.00331v1.pdf", "pdf", "paper artifact", "all_paper_prompt_strategies", False, False, "paper description", "paper LLMs", "paper datasets", "ranking;reranking", "paper_local", "official_description_only", "exact_semantic_only", "Local paper artifact, used for scope not literal text extraction."),
        ("local_full_matrix_requests", "results/llm/canonical_local_full_matrix/requests.jsonl", "jsonl", "local rendered request evidence", JSON_INDEX_STRATEGY_ID, True, False, "strict JSON one-based candidate indices", "qwen2;mistral", "Beauty;Sports;ML-1M", "ranking;reranking", "raw_request", "inferred_from_outputs", "local_ablation", "Large request file; prompt text identity frozen by file hash."),
    ]
    rows: list[dict[str, str]] = []
    for key, path_text, file_type, role, strategy, literal, parser, output_format, model, dataset, task, provenance, strength, classification, notes in sources:
        path = Path(path_text)
        placeholders = ""
        if (root / path).exists() and path.suffix.lower() in {".txt", ".py", ".csv"}:
            placeholders = ";".join(extract_placeholders(read_text(root, path)))
        rows.append(
            {
                "source_id": stable_id("F6E1-SRC", key, path_text),
                "relative_path": path_text,
                "file_type": file_type,
                "role": role,
                "linked_strategy": strategy,
                "literal_text_present": bool_text(literal),
                "parser_present": bool_text(parser),
                "placeholders": placeholders,
                "output_format": output_format,
                "model_association": model,
                "dataset_association": dataset,
                "task_association": task,
                "provenance": provenance,
                "evidence_strength": strength,
                "sha256": file_sha256_or_missing(root, path),
                "preliminary_classification": classification,
                "notes": notes,
            }
        )
    return rows


def build_prompt_strategy_spec(root: Path) -> list[dict[str, str]]:
    return strategy_definitions(root)


def build_placeholder_contract(root: Path) -> list[dict[str, str]]:
    strategy_sources = {row["strategy_id"]: row for row in strategy_definitions(root)}
    rows: list[dict[str, str]] = []
    for strategy_id, spec in strategy_sources.items():
        for order, placeholder in enumerate(CANONICAL_PLACEHOLDERS, start=1):
            rows.append(_placeholder_row(strategy_id, spec, placeholder, order))
    return rows


def _placeholder_row(strategy_id: str, spec: Mapping[str, str], placeholder: str, order: int) -> dict[str, str]:
    source_map = {
        BASE_STRATEGY_ID: {
            "user_id": ("user_name", "string", "true", "source username mapping", "not_allowed", "official source uses user_name rather than raw user_id"),
            "user_history": ("user_history", "list[str]", "true", "python repr title list", "not_allowed", "chronological history after his_len truncation"),
            "chronological_history": ("user_history", "list[str]", "true", "python repr title list", "not_allowed", "default source order before shuffle ablation"),
            "candidate_items": ("candidate_list", "list[str]", "true", "python repr title list", "not_allowed", "candidate titles only"),
            "item_titles": ("candidate_list;user_history", "list[str]", "true", "python repr title list", "not_allowed", "titles are the primary identifier"),
            "expected_list_length": ("system prompt", "int", "true", "10", "10", "top 10 requested"),
            "dataset": ("dataset", "string", "true", "plain text", "not_allowed", "dataset display name"),
        },
        LLMRANK_STRATEGY_ID: {
            "user_history": ("user_history", "list[str]", "true", "python repr title list", "not_allowed", "history list in source prompt"),
            "chronological_history": ("user_history", "list[str]", "true", "python repr title list", "not_allowed", "ordered history in prompt text"),
            "recent_history": ("last_item", "list[str]", "true", "python repr single-item list", "not_allowed", "explicit recency reminder"),
            "candidate_items": ("candidate_list", "list[str]", "true", "python repr title list", "not_allowed", "20 candidates"),
            "item_titles": ("candidate_list;user_history;last_item", "list[str]", "true", "python repr title list", "not_allowed", "titles are the primary identifier"),
            "expected_list_length": ("system prompt", "int", "true", "10", "10", "system prompt requests top 10 despite LLMRank text ranking 20"),
            "reasoning_context": ("please think step by step", "string", "true", "literal instruction", "not_allowed", "source permits reasoning before system output constraint"),
        },
        LLMSREC_STRATEGY_ID: {
            "user_id": ("user_name", "string", "true", "source username mapping", "not_allowed", "official source uses user_name"),
            "user_history": ("user_history", "list[str]", "true", "python repr title list", "not_allowed", "full history list"),
            "chronological_history": ("user_history", "list[str]", "true", "python repr title list", "not_allowed", "default source order"),
            "recent_history": ("last_item", "list[str]", "true", "python repr single-item list", "not_allowed", "in-context target demonstration"),
            "long_term_history": ("last_history", "list[str]", "true", "python repr title list", "empty_list_when_length_one", "history excluding last_item"),
            "candidate_items": ("candidate_list", "list[str]", "true", "python repr title list", "not_allowed", "20 candidates"),
            "item_titles": ("candidate_list;user_history;last_item", "list[str]", "true", "python repr title list", "not_allowed", "titles are the primary identifier"),
            "expected_list_length": ("system prompt", "int", "true", "10", "10", "recommendation step expects top 10"),
            "dataset": ("dataset", "string", "true", "plain text", "not_allowed", "dataset display name"),
            "profile": ("user_profile", "string", "optional", "plain text profile", "blocked_without_generated_profile", "profile pipeline component"),
        },
        JSON_INDEX_STRATEGY_ID: {
            "user_history": ("history_render", "list[numbered title]", "true", "newline numbered titles", "not_allowed", "train-only chronological"),
            "chronological_history": ("history_render", "list[numbered title]", "true", "newline numbered titles", "not_allowed", "preserved render order"),
            "candidate_items": ("candidate_render", "list[numbered title]", "true", "newline numbered one-based candidates", "not_allowed", "20 candidates"),
            "candidate_indices": ("ranked_candidate_indices", "list[int]", "true", "JSON array one-based integers", "not_allowed", "output namespace"),
            "item_titles": ("candidate_render;history_render", "list[str]", "true", "numbered titles", "not_allowed", "titles are displayed but output uses indices"),
            "expected_list_length": ("requested_output_count", "int", "true", "5 ranking;10 or 20 reranking", "not_allowed", "contract-specific"),
            "task": ("task_type", "enum", "true", "ranking|reranking", "not_allowed", "local prompt branch"),
            "dataset": ("dataset field in manifest", "string", "true", "manifest metadata", "not_allowed", "not embedded in prompt text by default"),
            "output_schema": ("ranked_candidate_indices", "json_schema", "true", "JSON object exact key", "not_allowed", "strict local parser"),
        },
    }
    details = source_map.get(strategy_id, {}).get(placeholder)
    if details is None:
        source_name, value_type, required, serialization, fallback, notes = (
            "",
            "not_applicable",
            "false",
            "not_applicable",
            "not_applicable",
            "No compatible placeholder in this strategy; absence is explicit.",
        )
    else:
        source_name, value_type, required, serialization, fallback, notes = details
    return {
        "strategy_id": strategy_id,
        "placeholder_name": placeholder,
        "source_name": source_name,
        "type": value_type,
        "required": required,
        "order": str(order),
        "serialization": serialization,
        "allowed_values": _allowed_values_for_placeholder(placeholder),
        "fallback": fallback,
        "source_path": spec["source_paths"],
        "notes": notes,
    }


def _allowed_values_for_placeholder(placeholder: str) -> str:
    if placeholder == "task":
        return "ranking;reranking;profile_generation"
    if placeholder == "expected_list_length":
        return "5;10;20"
    if placeholder == "candidate_indices":
        return "one_based_1_to_candidate_count"
    return "source_defined"


def build_parser_contract(root: Path) -> list[dict[str, str]]:
    official_parser_hash = source_combined_hash(
        root,
        [
            Path("original_repo/EvaLLMasRecommender/eva/codes/llms.py"),
            Path("original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py"),
        ],
    )
    local_parser_hash = source_combined_hash(
        root,
        [
            Path("src/llm/canonical_llm_contract.py"),
            Path("results/tables/canonical_llm_parser_contract.csv"),
            Path("results/tables/canonical_local_llm_reranking_top10_parser_contract.csv"),
        ],
    )
    rows = [
        _official_parser_row(BASE_STRATEGY_ID, "official_title_list_top10_parser", official_parser_hash, "Base prompt title-list parser."),
        _official_parser_row(LLMRANK_STRATEGY_ID, "official_title_list_or_llama_numbered_parser", official_parser_hash, "LLMRank prompt text allows reasoning; upstream system/parser impose top-10 title parsing."),
        _official_parser_row(LLMSREC_STRATEGY_ID, "official_title_list_plus_free_text_profile_parser", official_parser_hash, "Recommendation step uses title-list parser; profile quality parser is not frozen."),
        _json_parser_row(JSON_INDEX_STRATEGY_ID, "llm_rank_strict_primary_parser_v1", "5", "false", local_parser_hash, "Local ranking strict parser."),
        _json_parser_row(JSON_INDEX_STRATEGY_ID, "llm_rerank_top10_strict_primary_parser_v1", "10", "false", local_parser_hash, "Local top-10 reranking strict parser."),
        _json_parser_row(JSON_INDEX_STRATEGY_ID, "llm_rerank_strict_primary_parser_v1", "20", "true", local_parser_hash, "Local full-20 reranking strict parser."),
    ]
    return rows


def _official_parser_row(strategy_id: str, parser_id: str, parser_hash: str, notes: str) -> dict[str, str]:
    return {
        "strategy_id": strategy_id,
        "parser_contract_id": parser_id,
        "output_format": "python_list_of_item_titles_or_numbered_title_lines",
        "identifier_namespace": "item_title_mapped_to_item_id",
        "index_base": "not_applicable",
        "list_format": "python_list_regex_for_chatgpt_glm;numbered_lines_for_llama",
        "json_required": "false",
        "free_text_allowed": "limited",
        "expected_output_count": "10",
        "requires_full_permutation": "false",
        "allows_duplicates": "not_explicitly_allowed",
        "allows_unknown_items": "false_after_title_mapping",
        "allows_partial_output": "false",
        "allows_reasoning": "not_in_final_output",
        "allows_code_fence": "false",
        "recoverable_errors": "length_over_10_truncated;length_under_10_padded_NA_in_original_metric_code",
        "nonrecoverable_errors": "missing_python_list;unmapped_title;invalid_llama_numbered_format",
        "original_parser_source": "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
        "local_parser_source": "",
        "strict_validity_definition": "one parseable top-10 title list that maps to known item ids",
        "parse_validity_definition": "parser accepts title list syntax and maps titles to ids",
        "candidate_hallucination_definition": "mapped item not in supplied candidate set or title cannot be mapped",
        "duplicate_ranking_definition": "same mapped item appears more than once",
        "parser_sha256": parser_hash,
        "classification": "official_code_literal",
        "notes": notes,
    }


def _json_parser_row(strategy_id: str, parser_id: str, expected_count: str, full: str, parser_hash: str, notes: str) -> dict[str, str]:
    return {
        "strategy_id": strategy_id,
        "parser_contract_id": parser_id,
        "output_format": "strict_json_ranked_candidate_indices",
        "identifier_namespace": "one_based_candidate_index",
        "index_base": "1",
        "list_format": "JSON object exact key ranked_candidate_indices",
        "json_required": "true",
        "free_text_allowed": "false",
        "expected_output_count": expected_count,
        "requires_full_permutation": full,
        "allows_duplicates": "false",
        "allows_unknown_items": "false",
        "allows_partial_output": "false",
        "allows_reasoning": "false",
        "allows_code_fence": "false",
        "recoverable_errors": "none_in_strict_primary_parser",
        "nonrecoverable_errors": "json_parse_failed;invalid_schema;invalid_length;duplicate_indices;unknown_indices;missing_permutation_indices",
        "original_parser_source": "",
        "local_parser_source": "src/llm/canonical_llm_contract.py;results/tables/canonical_llm_parser_contract.csv",
        "strict_validity_definition": "exact JSON object, correct length, unique in-range one-based indices, full permutation when required",
        "parse_validity_definition": "strict_parse_valid true only; diagnostic salvage is separate and non-authoritative",
        "candidate_hallucination_definition": "index outside supplied candidate range",
        "duplicate_ranking_definition": "repeated candidate index",
        "parser_sha256": parser_hash,
        "classification": "local_ablation",
        "notes": notes,
    }


def build_model_strategy_matrix(root: Path) -> list[dict[str, str]]:
    strategies = {row["strategy_id"]: row for row in strategy_definitions(root)}
    rows: list[dict[str, str]] = []

    def add(model_family: str, expected_id: str, local_id: str, strategy_id: str, scope: str, status: str, feasibility: str, blocker: str, notes: str) -> None:
        spec = strategies[strategy_id]
        rows.append(
            {
                "matrix_id": stable_id("F6E1-MX", model_family, expected_id, strategy_id, scope),
                "model_family": model_family,
                "expected_model_identifier": expected_id,
                "local_model_identifier": local_id,
                "strategy_id": strategy_id,
                "strategy_name": spec["strategy_name"],
                "status": status,
                "scope": scope,
                "exactness_classification": spec["classification"],
                "prompt_available": "true",
                "parser_available": "true" if strategy_id != LLMSREC_STRATEGY_ID else "partial",
                "llm_calls_per_request": spec["llm_calls_per_request"],
                "expected_context_size": _expected_context_size(strategy_id),
                "local_feasibility": feasibility,
                "inference_status": "not_run_in_f6e1",
                "blocker": blocker,
                "notes": notes,
            }
        )

    add("Qwen2 7B Instruct", "qwen-2-7b-instruct", "qwen2:7b-instruct", BASE_STRATEGY_ID, "paper_aligned_core", "defined_not_executed", "conditional_future_runtime", "exact model-version equivalence and no-inference F6E1 policy", "Paper family core cell; future run would not be exact unless version/protocol equivalence is proven.")
    add("Mistral 7B Instruct", "mistral-7b-instruct-v0.2", "mistral:7b-instruct", BASE_STRATEGY_ID, "paper_aligned_core", "defined_not_executed", "conditional_future_runtime", "exact model-version equivalence and no-inference F6E1 policy", "Paper family core cell; local tag is adapted.")
    add("Llama3-70B-Instruct", "llama-3-70b-instruct", "", BASE_STRATEGY_ID, "paper_aligned_core", "defined_blocked", "blocked_hardware_runtime", "Llama 3 70B local runtime not verified", "Do not declare locally executable without dedicated hardware/runtime gate.")
    add("Llama3-70B-Instruct", "llama-3-70b-instruct", "", LLMRANK_STRATEGY_ID, "paper_aligned_core", "defined_blocked", "blocked_hardware_runtime", "Llama 3 70B local runtime not verified", "Paper LLMRank core cell.")
    add("Llama3-70B-Instruct", "llama-3-70b-instruct", "", LLMSREC_STRATEGY_ID, "paper_aligned_core", "defined_blocked", "blocked_hardware_runtime_and_pipeline_evidence", "Llama 3 70B local runtime and full LLMSRec orchestration not verified", "Paper LLMSRec core cell with reconstructed prompt strategy.")
    add("Qwen2 7B Instruct", "qwen-2-7b-instruct", "qwen2:7b-instruct", LLMRANK_STRATEGY_ID, "cross_model_extension", "defined_not_executed", "conditional_future_runtime", "no-inference F6E1 policy", "Cross-model prompt extension, not paper-exact local result.")
    add("Qwen2 7B Instruct", "qwen-2-7b-instruct", "qwen2:7b-instruct", LLMSREC_STRATEGY_ID, "cross_model_extension", "defined_not_executed", "conditional_future_runtime", "full LLMSRec orchestration evidence and no-inference F6E1 policy", "Cross-model reconstructed strategy extension.")
    add("Mistral 7B Instruct", "mistral-7b-instruct-v0.2", "mistral:7b-instruct", LLMRANK_STRATEGY_ID, "cross_model_extension", "defined_not_executed", "conditional_future_runtime", "no-inference F6E1 policy", "Cross-model prompt extension, not paper-exact local result.")
    add("Mistral 7B Instruct", "mistral-7b-instruct-v0.2", "mistral:7b-instruct", LLMSREC_STRATEGY_ID, "cross_model_extension", "defined_not_executed", "conditional_future_runtime", "full LLMSRec orchestration evidence and no-inference F6E1 policy", "Cross-model reconstructed strategy extension.")
    add("Qwen2 7B Instruct", "qwen-2-7b-instruct", "qwen2:7b-instruct", JSON_INDEX_STRATEGY_ID, "local_ablation", "defined_existing_local_evidence", "already_executed_before_f6e1_not_reexecuted", "not a paper-exact prompt", "Frozen local JSON/index ablation evidence, including LastFM F6D4A.")
    add("Mistral 7B Instruct", "mistral-7b-instruct-v0.2", "mistral:7b-instruct", JSON_INDEX_STRATEGY_ID, "local_ablation", "defined_existing_local_evidence", "already_executed_before_f6e1_not_reexecuted", "not a paper-exact prompt", "Frozen local JSON/index ablation evidence, including LastFM F6D4A.")
    return rows


def _expected_context_size(strategy_id: str) -> str:
    if strategy_id == LLMSREC_STRATEGY_ID:
        return "history10_plus_candidate20_plus_in_context_or_profile_text"
    if strategy_id == LLMRANK_STRATEGY_ID:
        return "history10_plus_candidate20_plus_last_item"
    if strategy_id == JSON_INDEX_STRATEGY_ID:
        return "last10_history_titles_plus_20_numbered_candidates"
    return "history10_plus_candidate20"


def build_missing_or_blocked(root: Path) -> list[dict[str, str]]:
    rows = [
        (BASE_STRATEGY_ID, "Exact OpenP5 source identity is not proven by local files.", "medium", "exact_OpenP5_claim", "documented_block", "original_repo/EvaLLMasRecommender/eva/templates/template.txt", "Use 'OpenP5-derived' wording unless primary OpenP5 evidence is added.", "Template is exact-parametric for upstream code, not exact OpenP5 lineage."),
        (BASE_STRATEGY_ID, "Exact paper system prompt/model/candidate protocol is not locally executed in F6E1.", "high", "protocol_exact_execution", "blocked_no_inference", "original_repo/EvaLLMasRecommender/eva/codes/llms.py", "Freeze only; execute in later approved phase.", "F6E1 performs no requests."),
        (LLMRANK_STRATEGY_ID, "LLMRank recency-focused template uses product/movie wording inconsistency.", "medium", "literal_prompt_cleanup", "documented_do_not_edit", "original_repo/EvaLLMasRecommender/eva/templates/llmrank_rf.txt", "Preserve source literal; decide later whether execution uses exact literal or adapted correction.", "Do not silently repair template text."),
        (LLMRANK_STRATEGY_ID, "Paper-level LLMRank dataset/model application is not executed locally.", "high", "protocol_exact_execution", "blocked_no_inference", "results/tables/finalization_f6b_model_prompt_matrix.csv", "Use future runtime gate for execution.", "F6E1 matrix is formal only."),
        (LLMSREC_STRATEGY_ID, "Full LLMSRec orchestration cannot be proven as one canonical literal prompt.", "high", "full_LLMSRec_pipeline", "documented_block", "original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb;original_repo/EvaLLMasRecommender/eva/templates/template_in_context.txt;original_repo/EvaLLMasRecommender/eva/templates/template_profile*.txt", "Keep reconstructed status until exact pipeline evidence is added.", "Literal components exist but profile-state policy remains ambiguous."),
        (LLMSREC_STRATEGY_ID, "No exact parser/quality contract for generated user profile is frozen.", "medium", "profile_generation_validation", "documented_block", "original_repo/EvaLLMasRecommender/eva/templates/template_user_profile_generate.txt", "Treat profile generation as upstream component, not validated local result.", "Recommendation parser exists; profile parser does not."),
        (JSON_INDEX_STRATEGY_ID, "JSON/index prompt is not an exact paper prompt.", "critical", "paper_exact_prompt_equivalence", "intentionally_blocked", "src/llm/canonical_llm_contract.py;results/tables/finalization_f6d3_prompt_contract.csv", "Retain local_ablation classification.", "This is a deliberate reproducibility-control adaptation."),
        ("llama3_runtime", "Llama 3 70B local feasibility is not verified.", "high", "local_execution_feasibility", "blocked_hardware_runtime", "results/tables/finalization_f6b_model_prompt_matrix.csv", "Run a separate hardware/runtime preflight before any execution.", "F6E1 must not declare it runnable."),
        ("remote_api_models", "Closed API models remain blocked by no-network/no-API policy.", "high", "remote_model_execution", "blocked_external_dependency", "results/tables/finalization_f6b_model_prompt_matrix.csv", "Do not call remote APIs.", "Applies to GPT and Claude paper models."),
    ]
    return [
        {
            "block_id": stable_id("F6E1-BLOCK", strategy_id, gap),
            "strategy_id": strategy_id,
            "evidence_gap": gap,
            "severity": severity,
            "blocked_scope": scope,
            "status": status,
            "source_paths": source_paths,
            "safe_next_action": action,
            "notes": notes,
        }
        for strategy_id, gap, severity, scope, status, source_paths, action, notes in rows
    ]


def build_frames(root: Path) -> dict[str, list[dict[str, str]]]:
    return {
        "prompt_source_inventory": build_prompt_source_inventory(root),
        "prompt_strategy_spec": build_prompt_strategy_spec(root),
        "placeholder_contract": build_placeholder_contract(root),
        "parser_contract": build_parser_contract(root),
        "model_strategy_matrix": build_model_strategy_matrix(root),
        "missing_or_blocked": build_missing_or_blocked(root),
    }


def table_columns(table_name: str) -> list[str]:
    return {
        "prompt_source_inventory": SOURCE_INVENTORY_COLUMNS,
        "prompt_strategy_spec": STRATEGY_SPEC_COLUMNS,
        "placeholder_contract": PLACEHOLDER_CONTRACT_COLUMNS,
        "parser_contract": PARSER_CONTRACT_COLUMNS,
        "model_strategy_matrix": MODEL_STRATEGY_COLUMNS,
        "missing_or_blocked": MISSING_OR_BLOCKED_COLUMNS,
    }[table_name]


def rows_csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    from io import StringIO

    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: str(row.get(column, "")) for column in columns})
    return handle.getvalue()


def rows_sha256(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    return sha256_text(rows_csv_text(rows, columns))


def build_manifest(root: Path, frames: Mapping[str, list[dict[str, str]]]) -> dict[str, Any]:
    strategies = frames["prompt_strategy_spec"]
    placeholders = frames["placeholder_contract"]
    parsers = frames["parser_contract"]
    matrix = frames["model_strategy_matrix"]

    table_hashes = {
        table: rows_sha256(rows, table_columns(table))
        for table, rows in frames.items()
    }
    strategy_payloads = []
    for strategy in strategies:
        strategy_id = strategy["strategy_id"]
        placeholder_rows = [row for row in placeholders if row["strategy_id"] == strategy_id]
        parser_rows = [row for row in parsers if row["strategy_id"] == strategy_id]
        matrix_rows = [row for row in matrix if row["strategy_id"] == strategy_id]
        source_paths = [Path(path) for path in strategy["source_paths"].split(";") if path]
        strategy_payloads.append(
            {
                "strategy_id": strategy_id,
                "classification": strategy["classification"],
                "provenance": strategy["provenance"],
                "source_paths": [rel(path) for path in source_paths],
                "source_hashes": {rel(path): file_sha256_or_missing(root, path) for path in source_paths},
                "template_hash": strategy["template_sha256"],
                "parser_hash": strategy["parser_sha256"],
                "placeholder_contract_hash": rows_sha256(placeholder_rows, PLACEHOLDER_CONTRACT_COLUMNS),
                "multi_step_structure": strategy["multi_step_structure"],
                "model_applicability": strategy["model_applicability"],
                "task_applicability": strategy["task_applicability"],
                "dataset_applicability": strategy["dataset_applicability"],
                "matrix_cells": [row["matrix_id"] for row in matrix_rows],
            }
        )
    scientific_payload = {
        "phase": "F6E1",
        "strategy_freeze": strategy_payloads,
        "table_hashes": table_hashes,
    }
    scientific_hash = sha256_text(stable_json(scientific_payload))
    return {
        "phase": "F6E1",
        "decision": final_decision(frames),
        "frozen_timestamp_utc": FROZEN_TIMESTAMP_UTC,
        "timestamp_excluded_from_scientific_hash": True,
        "no_inference": True,
        "no_network": True,
        "no_ollama": True,
        "scientific_content_hash": scientific_hash,
        "table_hashes": table_hashes,
        "strategy_freeze": strategy_payloads,
    }


def final_decision(frames: Mapping[str, Sequence[Mapping[str, str]]]) -> str:
    classifications = {row["classification"] for row in frames["prompt_strategy_spec"]}
    required = {"exact_parametric", "reconstructed", "local_ablation"}
    matrix_rows = len(frames["model_strategy_matrix"])
    if required.issubset(classifications) and matrix_rows == 11:
        return "f6e1_prompt_strategy_freeze_complete_with_documented_blocks"
    return "f6e1_incomplete_missing_evidence"


def build_report(root: Path, frames: Mapping[str, list[dict[str, str]]], manifest: Mapping[str, Any]) -> str:
    strategies = frames["prompt_strategy_spec"]
    source_rows = frames["prompt_source_inventory"]
    matrix = frames["model_strategy_matrix"]
    blocks = frames["missing_or_blocked"]
    classifications = ", ".join(f"{row['strategy_name']}={row['classification']}" for row in strategies)
    lines = [
        "# F6E1 - Prompt Strategy Recovery, Formalization and Evidence Freeze",
        "",
        "## 1. Initial repository state",
        "Repository state is recorded by the final operator command log; F6E1 artifacts are generated without staging or commit.",
        "",
        "## 2. Local source discovery",
        f"Inventoried local prompt/parser sources: {len(source_rows)}.",
        "Discovery covers upstream templates, upstream prompt generation/evaluation code, local JSON/index contracts, F6A/F6B/F6D3 audit tables, local paper artifacts, and frozen local request evidence.",
        "",
        "## 3. Prompt source inventory",
        "Authoritative table: results/tables/phase_f6e1_prompt_source_inventory.csv.",
        "",
        "## 4. Base prompt recovery",
        f"Strategy {BASE_STRATEGY_ID} is classified as exact_parametric: official template variants are present in original_repo/EvaLLMasRecommender/eva/templates/template.txt.",
        "The output contract remains title-list based in upstream code; exact OpenP5 lineage is not asserted.",
        "",
        "## 5. LLMRank recovery",
        f"Strategy {LLMRANK_STRATEGY_ID} is classified as exact_parametric: llmrank.txt and llmrank_rf.txt are present locally.",
        "The recency-focused source includes last_item and is frozen literally, including its wording inconsistency.",
        "",
        "## 6. LLMSRec recovery",
        f"Strategy {LLMSREC_STRATEGY_ID} is classified as reconstructed.",
        "Literal components exist for in-context prompting and profile generation/injection, but full canonical orchestration is not proven as one exact local prompt.",
        "",
        "## 7. JSON/index local ablation audit",
        f"Strategy {JSON_INDEX_STRATEGY_ID} is classified as local_ablation.",
        "It uses strict JSON one-based candidate indices and is not promoted to an exact paper prompt.",
        "",
        "## 8. Placeholder contract",
        f"Placeholder rows frozen: {len(frames['placeholder_contract'])}.",
        "",
        "## 9. Parser contract",
        f"Parser rows frozen: {len(frames['parser_contract'])}. Strict validity is separate from hallucination/candidate validity.",
        "",
        "## 10. Model-strategy matrix",
        f"Matrix rows frozen: {len(matrix)}: paper-aligned core, cross-model extension, and local ablation cells.",
        "",
        "## 11. Scientific exactness classification",
        classifications,
        "",
        "## 12. Implementation files",
        "- src/llm/plan_f6e1_prompt_strategies.py",
        "- src/experiments/build_phase_f6e1_prompt_strategy_recovery.py",
        "- tests/test_plan_f6e1_prompt_strategies.py",
        "",
        "## 13. Tests added",
        "F6E1 tests check import purity, network/client absence, deterministic rendering, placeholder/parser contracts, blocked strategy behavior, raw hash invariance, and deterministic builder output.",
        "",
        "## 14. Targeted test results",
        "Recorded in final operator response after pytest execution.",
        "",
        "## 15. Full-suite result",
        "Recorded in final operator response after pytest execution.",
        "",
        "## 16. Determinism",
        f"Manifest scientific content hash: {manifest['scientific_content_hash']}. Timestamp is excluded from this hash.",
        "",
        "## 17. Protected-input verification",
        "Protected-input and raw hash checks are read-only and recorded in final operator response.",
        "",
        "## 18. Prompt freeze manifest",
        "Authoritative manifest: results/manifests/phase_f6e1_prompt_freeze.json.",
        "",
        "## 19. Missing or blocked evidence",
        f"Blocked/missing evidence rows frozen: {len(blocks)}.",
        "",
        "## 20. Final decision",
        f"{final_decision(frames)}.",
        "",
        "## 21. Final git state",
        "Recorded in final operator response; no staging or commit performed by F6E1.",
        "",
        "## 22. Files created or modified",
        "Only F6E1 implementation, test, report, CSV tables and manifest are part of this phase.",
        "",
        "## 23. Remaining limitations",
        "No inference was run; paper-exact model versions, exact LLMSRec orchestration, exact OpenP5 lineage, and Llama3 70B runtime feasibility remain blocked or conditional.",
        "",
        "## 24. Safe next action",
        "Use the frozen matrix to design a separate runtime preflight before any F6E2/F6E execution.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(root: Path, output_root: Path, frames: Mapping[str, list[dict[str, str]]], manifest: Mapping[str, Any], report: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for table_name, relative in OUTPUT_TABLES.items():
        path = output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rows_csv_text(frames[table_name], table_columns(table_name)), encoding="utf-8", newline="\n")
        hashes[rel(relative)] = sha256_file(path)

    manifest_path = output_root / MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    hashes[rel(MANIFEST_PATH)] = sha256_file(manifest_path)

    report_path = output_root / REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")
    hashes[rel(REPORT_PATH)] = sha256_file(report_path)
    return dict(sorted(hashes.items()))


def build_all(root: Path, output_root: Path | None = None) -> dict[str, Any]:
    output_root = root if output_root is None else output_root
    frames = build_frames(root)
    manifest = build_manifest(root, frames)
    report = build_report(root, frames, manifest)
    hashes = write_outputs(root, output_root, frames, manifest, report)
    return {
        "frames": frames,
        "manifest": manifest,
        "report": report,
        "hashes": hashes,
        "decision": final_decision(frames),
    }


def output_hashes(root: Path) -> dict[str, str]:
    paths = [REPORT_PATH, MANIFEST_PATH, *OUTPUT_TABLES.values()]
    return {rel(path): sha256_file(root / path) for path in paths if (root / path).exists()}


def protected_input_summary(root: Path, table: Path = Path("results/tables/finalization_f2_protected_input_hashes_after.csv")) -> dict[str, int]:
    checked = 0
    missing = 0
    changed = 0
    path = root / table
    if not path.exists():
        return {"checked": 0, "missing": 1, "changed": 0}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("exists") not in {"true", "True", "1"}:
                continue
            relative = Path(str(row.get("relative_path", "")).replace("\\", "/"))
            target = root / relative
            checked += 1
            if not target.exists() or not target.is_file():
                missing += 1
            elif sha256_file(target) != row.get("sha256"):
                changed += 1
    return {"checked": checked, "missing": missing, "changed": changed}

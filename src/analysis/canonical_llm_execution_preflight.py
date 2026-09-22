"""Offline Phase 5A LLM prompt/parser/execution preflight builders."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.audit.common_io import read_csv_dataframe as _read_csv_dataframe
from src.llm.canonical_llm_contract import (
    EXECUTION_PARAMETER_COLUMNS,
    MODEL_INVENTORY_COLUMNS,
    NONDETERMINISM_COLUMNS,
    PARSER_CONTRACT_COLUMNS,
    POLICY_OPTION_COLUMNS,
    PROMPT_CONTRACT_COLUMNS,
    bool_text,
    contract_hash,
    estimate_token_envelope,
    json_dumps,
    parse_strict_index_response,
    parser_contract_rows,
    prompt_contract_rows,
    render_prompt,
    request_count,
    sha256_items,
    sha256_text,
    validate_one_recommended_policy_per_category,
)


COHORT_PREFLIGHT_COLUMNS = [
    "candidate_set_id",
    "source_path",
    "dataset",
    "task_type",
    "user_id",
    "positive_item_id",
    "candidate_count",
    "candidate_items_valid",
    "candidate_order_preserved",
    "candidate_titles_complete",
    "history_available",
    "history_train_only",
    "history_chronology_verified",
    "held_out_positive_absent_from_history",
    "history_titles_complete",
    "primary_prompt_compatible",
    "eligible_full_candidate_layer",
    "eligible_strict_title_primary",
    "exclusion_reasons",
    "notes",
]

PROMPT_RENDER_COLUMNS = [
    "prompt_render_id",
    "prompt_contract_id",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "history_item_count",
    "candidate_count",
    "system_message_sha256",
    "user_message_sha256",
    "complete_prompt_sha256",
    "history_render_sha256",
    "candidate_render_sha256",
    "candidate_order_sha256",
    "positive_label_exposed",
    "rendering_valid",
    "character_count",
    "whitespace_token_count",
    "estimated_token_min",
    "estimated_token_max",
    "token_count_method",
    "blocking_reason",
    "notes",
]

EXISTING_PARSER_AUDIT_COLUMNS = [
    "audit_id",
    "source_response_path",
    "source_response_id",
    "model_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "existing_parser_id",
    "strict_parser_id",
    "existing_parse_valid",
    "strict_parse_valid",
    "existing_output_items",
    "strict_output_items",
    "candidate_validity",
    "duplicate_count",
    "unknown_item_count",
    "output_length",
    "parser_disagreement",
    "audit_status",
    "notes",
]

EXECUTION_MATRIX_COLUMNS = [
    "execution_cell_id",
    "execution_scope",
    "model_family",
    "model_identifier",
    "dataset",
    "task_type",
    "cohort_id",
    "eligible_rows",
    "prompt_contract_id",
    "parser_contract_id",
    "execution_contract_id",
    "repetitions_per_row",
    "position_permutations_per_row",
    "history_variants_per_row",
    "profile_generation_requests_per_row",
    "recommendation_requests_per_row",
    "base_request_count",
    "total_planned_request_count",
    "character_count_total",
    "whitespace_token_count_total",
    "estimated_token_min_total",
    "estimated_token_max_total",
    "execution_status",
    "blocking_reason",
    "notes",
]

BLOCKED_COLUMNS = [
    "issue_id",
    "issue_scope",
    "model_family",
    "dataset",
    "task_type",
    "issue_category",
    "issue_description",
    "missing_or_conflicting_evidence",
    "blocks_paper_exact_execution",
    "adapted_execution_available",
    "adapted_execution_description",
    "requires_external_api",
    "requires_model_download",
    "requires_hardware_beyond_local",
    "requires_current_provider_verification",
    "requires_user_approval",
    "recommended_resolution",
    "severity",
    "status",
    "notes",
]

OUTPUT_PATHS = {
    "prompt_contract": "results/tables/canonical_llm_prompt_contract.csv",
    "parser_contract": "results/tables/canonical_llm_parser_contract.csv",
    "model_inventory": "results/tables/canonical_llm_model_inventory.csv",
    "execution_parameter_contract": "results/tables/canonical_llm_execution_parameter_contract.csv",
    "cohort_preflight": "results/tables/canonical_llm_cohort_preflight.csv",
    "prompt_rendering_audit": "results/tables/canonical_llm_prompt_rendering_audit.csv",
    "existing_parser_audit": "results/tables/canonical_llm_existing_parser_audit.csv",
    "execution_matrix": "results/tables/canonical_llm_execution_matrix.csv",
    "nondeterminism_policy": "results/tables/canonical_llm_nondeterminism_policy.csv",
    "policy_options": "results/tables/canonical_llm_policy_options.csv",
    "blocked_or_conditional": "results/tables/canonical_llm_blocked_or_conditional.csv",
    "report": "report/canonical_llm_prompt_parsing_and_execution_preflight.md",
}

OUTPUT_COLUMNS = {
    "prompt_contract": PROMPT_CONTRACT_COLUMNS,
    "parser_contract": PARSER_CONTRACT_COLUMNS,
    "model_inventory": MODEL_INVENTORY_COLUMNS,
    "execution_parameter_contract": EXECUTION_PARAMETER_COLUMNS,
    "cohort_preflight": COHORT_PREFLIGHT_COLUMNS,
    "prompt_rendering_audit": PROMPT_RENDER_COLUMNS,
    "existing_parser_audit": EXISTING_PARSER_AUDIT_COLUMNS,
    "execution_matrix": EXECUTION_MATRIX_COLUMNS,
    "nondeterminism_policy": NONDETERMINISM_COLUMNS,
    "policy_options": POLICY_OPTION_COLUMNS,
    "blocked_or_conditional": BLOCKED_COLUMNS,
}

DATASET_INTERACTIONS = {
    "Amazon Beauty": "data/reconstructed/plan_c/amazon_beauty/interactions.csv",
    "Amazon Sports": "data/reconstructed/plan_c/amazon_sports/interactions.csv",
    "ML-1M": "data/reconstructed/plan_c/ml1m/interactions.csv",
}

PLAN_E2_DATASETS = ("Amazon Beauty", "Amazon Sports", "ML-1M")
PLAN_E2_TASKS = ("ranking", "reranking")
LOCAL_MODELS = {
    "Qwen2 7B Instruct": "qwen2:7b-instruct",
    "Mistral 7B Instruct": "mistral:7b-instruct",
}
PAPER_MODELS = {
    "GPT-3.5": "gpt-3.5-turbo-0125",
    "GPT-4o-mini": "gpt-4o-mini-2024-07-18",
    "GPT-4o": "gpt-4o-2024-08-06",
    "Claude 3 Haiku": "claude-3-haiku-20240307",
    "Llama3-70B-Instruct": "llama-3-70b-instruct",
}


def read_csv(root: Path, relative: str) -> pd.DataFrame:
    """Compatibility wrapper retained for F6I2G deterministic_csv_io provenance."""

    return _read_csv_dataframe(relative, root=root, dtype=str, keep_default_na=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_input_paths(root: Path) -> list[str]:
    explicit = [
        "paper/Beyond Utility Evaluating LLM As Recommander.pdf",
        "results/tables/ranking_candidate_sets.csv",
        "results/tables/positive_positions.csv",
        "results/tables/experiment_cohort_membership.csv",
        "results/tables/candidate_item_title_map.csv",
        "results/tables/item_title_map.csv",
        "results/tables/llm_title_exclusions.csv",
        "results/tables/experiment_cohort_summary.csv",
        "results/tables/canonical_metric_contract.csv",
        "results/tables/canonical_metric_results_long.csv",
        "results/tables/canonical_metric_results_wide.csv",
        "results/tables/canonical_metric_blocked_results.csv",
        "results/tables/canonical_metric_results_validation.csv",
        "results/tables/plan_d_llm_model_registry.csv",
        "results/tables/plan_d_llm_backend_audit.csv",
        "results/tables/plan_d_llm_prompt_template_registry.csv",
        "results/tables/plan_d_final_llm_diagnostic_summary.csv",
    ]
    globs = [
        "src/llm/*.py",
        "src/experiments/*plan_d*llm*.py",
        "src/experiments/*qwen*.py",
        "src/experiments/*mistral*.py",
        "src/analysis/plan_d*llm*.py",
        "src/analysis/plan_d*qwen*.py",
        "src/analysis/plan_d*mistral*.py",
        "src/analysis/plan_d*position*.py",
        "src/analysis/plan_d*history*.py",
        "results/tables/plan_d*llm*.csv",
        "results/tables/plan_d*qwen*.csv",
        "results/tables/plan_d*mistral*.csv",
        "results/tables/plan_d*position*.csv",
        "report/plan_d*llm*.md",
        "report/plan_d*qwen*.md",
        "report/plan_d*mistral*.md",
        "original_repo/EvaLLMasRecommender/eva/codes/*.py",
        "original_repo/EvaLLMasRecommender/eva/templates/*.txt",
        "original_repo/EvaLLMasRecommender/README.md",
        "original_repo/EvaLLMasRecommender/requirements.txt",
    ]
    paths = [p for p in explicit if (root / p).is_file()]
    for pattern in globs:
        paths.extend(str(path.relative_to(root)).replace("\\", "/") for path in root.glob(pattern) if path.is_file())
    return sorted(set(paths))


def hash_protected_inputs(root: Path) -> dict[str, str]:
    return {relative: file_sha256(root / relative) for relative in protected_input_paths(root)}


def execution_environment(root: Path) -> str:
    import pandas

    tokenizer_libs = {
        name: bool(importlib.util.find_spec(name))
        for name in ("tiktoken", "transformers", "sentencepiece", "tokenizers")
    }
    return ";".join(
        [
            f"executable={sys.executable}",
            f"python={sys.version.split()[0]}",
            f"pandas={pandas.__version__}",
            f"numpy={np.__version__}",
            f"tokenizer_libs={json_dumps(tokenizer_libs)}",
            f"cwd={root}",
            "cpu_only=true",
        ]
    )


def build_all_frames(root: Path) -> dict[str, pd.DataFrame]:
    prompt_contract = frame(prompt_contract_rows(), PROMPT_CONTRACT_COLUMNS)
    parser_contract = frame(parser_contract_rows(), PARSER_CONTRACT_COLUMNS)
    title_map = load_title_map(root)
    histories = load_histories(root, title_map)
    cohort = build_cohort_preflight(root, title_map, histories)
    prompt_rendering = build_prompt_rendering_audit(root, cohort, title_map, histories)
    model_inventory = build_model_inventory(root)
    execution_params = build_execution_parameter_contract()
    existing_parser = build_existing_parser_audit(root)
    nondeterminism = build_nondeterminism_policy()
    policies = build_policy_options()
    blocked = build_blocked_or_conditional(root)
    execution_matrix = build_execution_matrix(prompt_rendering, cohort, model_inventory, execution_params)
    frames = {
        "prompt_contract": prompt_contract,
        "parser_contract": parser_contract,
        "model_inventory": model_inventory,
        "execution_parameter_contract": execution_params,
        "cohort_preflight": cohort,
        "prompt_rendering_audit": prompt_rendering,
        "existing_parser_audit": existing_parser,
        "execution_matrix": execution_matrix,
        "nondeterminism_policy": nondeterminism,
        "policy_options": policies,
        "blocked_or_conditional": blocked,
    }
    validate_frames(frames)
    return frames


def load_title_map(root: Path) -> dict[tuple[str, str], str]:
    titles = read_csv(root, "results/tables/item_title_map.csv")
    mapping: dict[tuple[str, str], str] = {}
    for row in titles[["dataset", "item_id", "canonical_title"]].itertuples(index=False):
        title = str(row.canonical_title).strip()
        if title:
            mapping[(str(row.dataset), str(row.item_id))] = title
    return mapping


def load_histories(root: Path, title_map: Mapping[tuple[str, str], str]) -> dict[str, dict[str, list[dict[str, str]]]]:
    histories: dict[str, dict[str, list[dict[str, str]]]] = {}
    for dataset, relative in DATASET_INTERACTIONS.items():
        interactions = read_csv(root, relative)
        interactions = interactions.copy()
        interactions["_row_order"] = range(len(interactions))
        interactions["timestamp_numeric"] = pd.to_numeric(interactions.get("timestamp", "0"), errors="coerce").fillna(0)
        train = interactions.loc[interactions["split"].str.lower().eq("train")].copy()
        train = train.sort_values(["user_id", "timestamp_numeric", "_row_order"], kind="mergesort")
        dataset_histories: dict[str, list[dict[str, str]]] = {}
        for user_id, group in train.groupby("user_id", sort=False):
            items: list[dict[str, str]] = []
            for record in group.itertuples(index=False):
                item_id = str(record.item_id)
                items.append(
                    {
                        "item_id": item_id,
                        "title": title_map.get((dataset, item_id), ""),
                        "timestamp": str(record.timestamp),
                    }
                )
            dataset_histories[str(user_id)] = items
        histories[dataset] = dataset_histories
    return histories


def build_cohort_preflight(
    root: Path,
    title_map: Mapping[tuple[str, str], str],
    histories: Mapping[str, Mapping[str, list[dict[str, str]]]],
) -> pd.DataFrame:
    membership = read_csv(root, "results/tables/experiment_cohort_membership.csv")
    candidates = read_csv(root, "results/tables/ranking_candidate_sets.csv")
    plan = membership.loc[membership["eligible_plan_e2_paper_like_full"].eq("true")].copy()
    plan = plan.merge(
        candidates[["candidate_set_id", "user_id", "candidate_items"]],
        on=["candidate_set_id", "user_id"],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, str]] = []
    for record in plan.sort_values(["dataset", "task_type", "candidate_set_id", "user_id"], kind="mergesort").itertuples(index=False):
        candidate_items = parse_json_list(record.candidate_items)
        history_items = [item for item in histories[str(record.dataset)].get(str(record.user_id), []) if item["item_id"] != str(record.positive_item_id)]
        rendered_history = history_items[-10:]
        history_titles_complete = bool(rendered_history) and all(item["title"] for item in rendered_history)
        candidate_titles_complete = all(title_map.get((str(record.dataset), item_id), "") for item_id in candidate_items)
        heldout_absent = all(item["item_id"] != str(record.positive_item_id) for item in history_items)
        candidate_valid = (
            len(candidate_items) == 20
            and len(set(candidate_items)) == 20
            and str(record.positive_item_id) in candidate_items
        )
        primary = (
            candidate_valid
            and candidate_titles_complete
            and bool(rendered_history)
            and history_titles_complete
            and heldout_absent
        )
        reasons = []
        if not candidate_titles_complete:
            reasons.append("missing_candidate_titles")
        if not history_titles_complete:
            reasons.append("missing_rendered_history_titles")
        if not heldout_absent:
            reasons.append("held_out_positive_in_history")
        if not candidate_valid:
            reasons.append("invalid_candidate_structure")
        rows.append(
            {
                "candidate_set_id": str(record.candidate_set_id),
                "source_path": str(record.source_path),
                "dataset": str(record.dataset),
                "task_type": str(record.task_type),
                "user_id": str(record.user_id),
                "positive_item_id": str(record.positive_item_id),
                "candidate_count": str(len(candidate_items)),
                "candidate_items_valid": bool_text(candidate_valid),
                "candidate_order_preserved": "true",
                "candidate_titles_complete": bool_text(candidate_titles_complete),
                "history_available": bool_text(bool(rendered_history)),
                "history_train_only": "true",
                "history_chronology_verified": "true",
                "held_out_positive_absent_from_history": bool_text(heldout_absent),
                "history_titles_complete": bool_text(history_titles_complete),
                "primary_prompt_compatible": bool_text(primary),
                "eligible_full_candidate_layer": "true",
                "eligible_strict_title_primary": str(record.eligible_plan_e2_llm_strict_title).lower(),
                "exclusion_reasons": ";".join(reasons) if reasons else "none",
                "notes": "strict candidate-title cohort reconciles to Phase 2C; history title check is rendered-history only",
            }
        )
    return frame(rows, COHORT_PREFLIGHT_COLUMNS)


def build_prompt_rendering_audit(
    root: Path,
    cohort: pd.DataFrame,
    title_map: Mapping[tuple[str, str], str],
    histories: Mapping[str, Mapping[str, list[dict[str, str]]]],
) -> pd.DataFrame:
    candidates = read_csv(root, "results/tables/ranking_candidate_sets.csv")
    candidate_lookup = {
        (row.candidate_set_id, row.user_id): parse_json_list(row.candidate_items)
        for row in candidates[["candidate_set_id", "user_id", "candidate_items"]].itertuples(index=False)
    }
    rows: list[dict[str, str]] = []
    strict = cohort.loc[cohort["eligible_strict_title_primary"].eq("true")].copy()
    for index, record in enumerate(strict.itertuples(index=False), start=1):
        key = (record.candidate_set_id, record.user_id)
        candidate_ids = candidate_lookup[key]
        contract_id = (
            "llm_rank_index_json_top5_v1"
            if record.task_type == "ranking"
            else "llm_rerank_index_json_full20_v1"
        )
        candidate_items = [
            {"item_id": item_id, "title": title_map.get((record.dataset, item_id), "")}
            for item_id in candidate_ids
        ]
        history_items = [
            item
            for item in histories[record.dataset].get(record.user_id, [])
            if item["item_id"] != record.positive_item_id
        ][-10:]
        valid = record.primary_prompt_compatible == "true"
        blocking_reason = "" if valid else record.exclusion_reasons
        hashes = {
            "system_message_sha256": "",
            "user_message_sha256": "",
            "complete_prompt_sha256": "",
            "history_render_sha256": "",
            "candidate_render_sha256": "",
        }
        character_count = whitespace_count = token_min = token_max = 0
        positive_label_exposed = "false"
        if valid:
            system_message, user_message, history_render, candidate_render = render_prompt(
                task_type=record.task_type,
                history_items=history_items,
                candidate_items=candidate_items,
            )
            complete = system_message + "\n\n" + user_message
            hashes = {
                "system_message_sha256": sha256_text(system_message),
                "user_message_sha256": sha256_text(user_message),
                "complete_prompt_sha256": sha256_text(complete),
                "history_render_sha256": sha256_text(history_render),
                "candidate_render_sha256": sha256_text(candidate_render),
            }
            character_count, whitespace_count, token_min, token_max = estimate_token_envelope(complete)
            lowered = complete.lower()
            positive_label_exposed = bool_text(
                "positive_item_id" in lowered
                or "ground_truth" in lowered
                or "held-out positive" in lowered
                or "held out positive" in lowered
            )
            if positive_label_exposed == "true":
                valid = False
                blocking_reason = "positive_label_exposed"
        rows.append(
            {
                "prompt_render_id": f"CLR-{index:06d}",
                "prompt_contract_id": contract_id,
                "dataset": record.dataset,
                "task_type": record.task_type,
                "candidate_set_id": record.candidate_set_id,
                "user_id": record.user_id,
                "history_item_count": str(len(history_items)),
                "candidate_count": record.candidate_count,
                "system_message_sha256": hashes["system_message_sha256"],
                "user_message_sha256": hashes["user_message_sha256"],
                "complete_prompt_sha256": hashes["complete_prompt_sha256"],
                "history_render_sha256": hashes["history_render_sha256"],
                "candidate_render_sha256": hashes["candidate_render_sha256"],
                "candidate_order_sha256": sha256_items(candidate_ids),
                "positive_label_exposed": positive_label_exposed,
                "rendering_valid": bool_text(valid),
                "character_count": str(character_count),
                "whitespace_token_count": str(whitespace_count),
                "estimated_token_min": str(token_min),
                "estimated_token_max": str(token_max),
                "token_count_method": "deterministic_character_whitespace_envelope_estimate",
                "blocking_reason": blocking_reason,
                "notes": "prompt text not stored in this audit table; hashes preserve identity",
            }
        )
    return frame(rows, PROMPT_RENDER_COLUMNS)


def build_model_inventory(root: Path) -> pd.DataFrame:
    backend = read_csv(root, "results/tables/plan_d_llm_backend_audit.csv")
    registry = read_csv(root, "results/tables/plan_d_llm_model_registry.csv")
    backend_text = "; ".join(
        f"{row.check_name}={row.version_or_result}"
        for row in backend.itertuples(index=False)
        if row.backend in {"ollama", "python"}
    )
    previous = {
        "Qwen2 7B Instruct": "results/tables/plan_d_qwen2_ml1m_ranking_diagnostic_instance_summary.csv;results/tables/plan_d_qwen2_ml1m_strict_prompt_instance_summary.csv",
        "Mistral 7B Instruct": "results/tables/plan_d_qwen_vs_mistral_ml1m_instance_summary.csv",
    }
    rows = [
        model_inventory_row("MI-001", "GPT-3.5", PAPER_MODELS["GPT-3.5"], "openai_api", "closed", "true", "", "", "", "", "API JSON mode not locally verified", "not verified", "verified_official_code", "conditional_remote_api", "requires external API key and current provider verification", "false"),
        model_inventory_row("MI-002", "GPT-4o-mini", PAPER_MODELS["GPT-4o-mini"], "openai_api", "closed", "true", "", "", "", "", "API JSON mode not locally verified", "not verified", "verified_official_code", "conditional_remote_api", "requires external API key and current provider verification", "false"),
        model_inventory_row("MI-003", "GPT-4o", PAPER_MODELS["GPT-4o"], "openai_api", "closed", "true", "", "", "", "", "API JSON mode not locally verified", "not verified", "verified_official_code", "conditional_remote_api", "requires external API key and current provider verification", "false"),
        model_inventory_row("MI-004", "Claude 3 Haiku", PAPER_MODELS["Claude 3 Haiku"], "anthropic_api", "closed", "true", "", "", "", "", "not locally verified", "not verified", "verified_official_code", "blocked_retired_or_unavailable", "Plan D registry marks exact model retired; no current provider check performed", "false"),
        model_inventory_row("MI-005", "Llama3-70B-Instruct", PAPER_MODELS["Llama3-70B-Instruct"], "cloud_or_api", "open_weight_large", "true", "", "", "", "", "not locally verified", "not verified", "verified_official_code", "blocked_hardware", "70B local execution blocked for RTX 4060 8GB target", "false"),
        model_inventory_row("MI-006", "Qwen2 7B Instruct", "qwen-2-7b-instruct", "ollama", "open_local", "false", "qwen2:7b-instruct", "", backend_text, previous["Qwen2 7B Instruct"], "Ollama format=json used in Plan D client", "temperature supported; seed not evidenced", "adapted_project_policy", "local_model_available_and_tested", "", "true"),
        model_inventory_row("MI-007", "Mistral 7B Instruct", "mistral-7b-instruct-v0.2", "ollama", "open_local", "false", "mistral:7b-instruct", "", backend_text, previous["Mistral 7B Instruct"], "Ollama format=json used in Plan D client", "temperature supported; seed not evidenced", "adapted_project_policy", "local_model_available_and_tested", "", "true"),
    ]
    for row in rows:
        match = registry.loc[registry["paper_model_exact_id"].eq(row["requested_model_identifier"])]
        if not match.empty:
            row["notes"] = str(match.iloc[0]["notes"])
    return frame(rows, MODEL_INVENTORY_COLUMNS)


def model_inventory_row(
    model_inventory_id: str,
    family: str,
    identifier: str,
    backend: str,
    source: str,
    paper_model: str,
    local_tag: str,
    digest: str,
    availability: str,
    previous_execution: str,
    structured: str,
    seed: str,
    alignment: str,
    status: str,
    blocking: str,
    allowed: str,
) -> dict[str, str]:
    return {
        "model_inventory_id": model_inventory_id,
        "model_family": family,
        "requested_model_identifier": identifier,
        "provider_or_backend": backend,
        "open_or_closed": source,
        "paper_model": paper_model,
        "local_model_tag": local_tag,
        "local_manifest_or_digest": digest,
        "local_availability_evidence": availability,
        "previous_execution_evidence": previous_execution,
        "structured_output_evidence": structured,
        "seed_support_evidence": seed,
        "paper_alignment": alignment,
        "availability_status": status,
        "blocking_reason": blocking,
        "allowed_for_future_execution": allowed,
        "notes": "",
    }


def build_execution_parameter_contract() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for family in list(PAPER_MODELS) + list(LOCAL_MODELS):
        for task in PLAN_E2_TASKS:
            local = family in LOCAL_MODELS
            status = "ready_adapted" if local else "conditional_or_blocked"
            blocking = "" if local else "requires external API/current provider verification or hardware beyond local"
            row = {
                "execution_contract_id": f"exec_{slug(family)}_{task}_v1",
                "model_family": family,
                "task_type": task,
                "temperature": "0" if local else "",
                "top_p": "",
                "top_k": "",
                "seed": "",
                "max_output_tokens": "256" if task == "ranking" else "512",
                "context_limit": "",
                "stop_sequences": "[]",
                "response_format": "json_object",
                "streaming": "false",
                "concurrency": "1",
                "timeout_seconds": "120" if local else "",
                "infrastructure_retry_count": "0",
                "scientific_retry_count": "0",
                "request_order_policy": "deterministic_dataset_task_user_order",
                "raw_response_retention": "required",
                "parameter_source": "adapted deterministic Phase 5A policy; Plan D used temperature 0 for local diagnostics" if local else "paper/API parameter values not frozen locally",
                "paper_alignment": "adapted_project_policy" if local else "blocked_ambiguous",
                "contract_status": status,
                "blocking_reason": blocking,
                "notes": "No hidden retries; first scientific response is retained.",
            }
            row["contract_sha256"] = contract_hash(row)
            rows.append(row)
    return frame(rows, EXECUTION_PARAMETER_COLUMNS)


def build_existing_parser_audit(root: Path) -> pd.DataFrame:
    sources = [
        ("results/tables/plan_d_llm_ml1m_reranking_instance_summary.csv", "reranking"),
        ("results/tables/plan_d_llm_ml1m_history_length_instance_summary.csv", "ranking"),
    ]
    rows: list[dict[str, str]] = []
    audit_index = 1
    for relative, task_type in sources:
        path = root / relative
        if not path.exists():
            continue
        frame_in = read_csv(root, relative)
        for record in frame_in.itertuples(index=False):
            if not hasattr(record, "selected_item_ids") or not hasattr(record, "candidate_ids"):
                continue
            selected = parse_json_list(record.selected_item_ids)
            candidates = parse_json_list(record.candidate_ids)
            unknown = [item for item in selected if item not in set(candidates)]
            duplicate_count = len(selected) - len(set(selected))
            existing_valid = str(getattr(record, "parse_success", "")).lower() == "true"
            strict_valid = False
            rows.append(
                {
                    "audit_id": f"EPA-{audit_index:06d}",
                    "source_response_path": relative,
                    "source_response_id": str(getattr(record, "diagnostic_id", getattr(record, "instance_id", ""))),
                    "model_family": normalize_model_family(str(getattr(record, "model", ""))),
                    "dataset": "ML-1M",
                    "task_type": task_type,
                    "candidate_set_id": "plan_d_diagnostic_existing_response",
                    "user_id": str(getattr(record, "user_id", "")),
                    "existing_parser_id": "plan_d_existing_item_id_parser",
                    "strict_parser_id": "canonical_index_parser_not_applied_to_item_id_summary",
                    "existing_parse_valid": bool_text(existing_valid),
                    "strict_parse_valid": bool_text(strict_valid),
                    "existing_output_items": json_dumps(selected),
                    "strict_output_items": "[]",
                    "candidate_validity": str(getattr(record, "candidate_validity_rate", "")),
                    "duplicate_count": str(duplicate_count),
                    "unknown_item_count": str(len(unknown)),
                    "output_length": str(len(selected)),
                    "parser_disagreement": "true",
                    "audit_status": "audited_existing_parsed_items_raw_response_not_replayed",
                    "notes": "Existing Plan D parsed item-ID summaries are audited only; no model response was generated.",
                }
            )
            audit_index += 1
    return frame(rows, EXISTING_PARSER_AUDIT_COLUMNS)


def build_nondeterminism_policy() -> pd.DataFrame:
    rows = [
        {
            "policy_id": "nondet_local_bounded_repeats_v1",
            "execution_scope": "nondeterminism_extension",
            "model_family": "Qwen2 7B Instruct;Mistral 7B Instruct",
            "cohort_policy": "deterministic 20-row ML-1M diagnostic subset before any full expansion",
            "repetitions_per_row": "3",
            "parameter_identity_required": "true",
            "prompt_identity_required": "true",
            "response_hash_required": "true",
            "parse_validity_metric": "strict_parse_valid_rate",
            "candidate_validity_metric": "candidate_validity_rate",
            "ranked_list_agreement_metric": "exact_ranked_list_agreement",
            "top5_agreement_metric": "overlap_at_5",
            "positive_rank_variance_metric": "positive_rank_variance",
            "utility_variance_metric": "hr_ndcg_variance",
            "result_independent": "true",
            "paper_alignment": "extension",
            "recommendation_status": "recommended",
            "blocking_reason": "",
            "notes": "Do not choose repetition count from observed future outputs.",
        },
        {
            "policy_id": "nondet_result_dependent_repeats_forbidden",
            "execution_scope": "nondeterminism_extension",
            "model_family": "all",
            "cohort_policy": "not_allowed",
            "repetitions_per_row": "",
            "parameter_identity_required": "true",
            "prompt_identity_required": "true",
            "response_hash_required": "true",
            "parse_validity_metric": "not_allowed",
            "candidate_validity_metric": "not_allowed",
            "ranked_list_agreement_metric": "not_allowed",
            "top5_agreement_metric": "not_allowed",
            "positive_rank_variance_metric": "not_allowed",
            "utility_variance_metric": "not_allowed",
            "result_independent": "false",
            "paper_alignment": "not_applicable",
            "recommendation_status": "forbidden",
            "blocking_reason": "result-dependent repetitions would bias stability analysis",
            "notes": "",
        },
    ]
    return frame(rows, NONDETERMINISM_COLUMNS)


def build_policy_options() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    def add(category: str, policy_id: str, name: str, desc: str, recommended: bool, risk: str, allowed: bool = True, uses_positive: bool = False, result_dependent: bool = False, changes_paper: bool = False) -> None:
        rows.append(
            {
                "ambiguity_category": category,
                "policy_id": policy_id,
                "policy_name": name,
                "description": desc,
                "paper_evidence": "official prompt/code evidence present but not fully machine-checkable",
                "official_code_evidence": "original_repo/EvaLLMasRecommender/eva/templates;eva/codes/llms.py;output_refine.py",
                "local_evidence": "Plan D strict JSON diagnostics and Phase 2C title-safe cohorts",
                "preserves_candidates": bool_text(True),
                "preserves_prompt_identity": bool_text(not result_dependent),
                "uses_positive_label": bool_text(uses_positive),
                "result_dependent": bool_text(result_dependent),
                "changes_paper_definition": bool_text(changes_paper),
                "comparability_impact": "adapted but auditable" if changes_paper else "preserves local comparability",
                "methodological_risk": risk,
                "allowed_for_future_execution": bool_text(allowed),
                "recommendation_status": "recommended" if recommended else ("forbidden" if not allowed else "allowed_not_recommended"),
                "rationale": "Recommended policy is deterministic, traceable, and avoids semantic repair." if recommended else "",
                "stop_condition": "stop if positive labels, candidate mutation, or result-dependent policy is detected",
                "notes": "",
            }
        )
    add("prompt_source", "prompt_adapted_strict_json", "Adapted strict JSON prompt", "Use local strict index prompt, record official conflict.", True, "medium", changes_paper=True)
    add("prompt_source", "prompt_official_free_text", "Official free-text prompt", "Use title-list prompt from original repo.", False, "high", allowed=False)
    add("candidate_representation", "candidate_one_based_index_title", "1-based index plus title", "Prompt shows titles with deterministic index labels.", True, "low", changes_paper=True)
    add("candidate_representation", "candidate_title_only", "Title only", "Prompt and parse by titles.", False, "high", allowed=False)
    add("output_format", "output_json_indices", "JSON indices", "Strict JSON object with candidate indices.", True, "low", changes_paper=True)
    add("output_format", "output_free_text_titles", "Free-text titles", "Official-like title list.", False, "high", allowed=False)
    add("parser", "parser_strict_plus_diagnostic_salvage", "Strict plus separate salvage", "Strict parser for science; salvage only for diagnostics.", True, "low")
    add("parser", "parser_fuzzy_title", "Fuzzy title parser", "Guess titles by fuzzy matching.", False, "high", allowed=False)
    add("retry", "retry_no_scientific_retry", "No scientific retry", "First scientific response retained; infra failures tracked separately.", True, "low")
    add("retry", "retry_validity_triggered", "Validity-triggered retry", "Retry malformed/hallucinated responses.", False, "high", allowed=False, result_dependent=True)
    add("generation", "generation_deterministic_adapted", "Deterministic adapted settings", "Temperature 0 when supported, concurrency 1, no streaming.", True, "low", changes_paper=True)
    add("generation", "generation_provider_defaults", "Provider defaults", "Leave parameters implicit.", False, "high", allowed=False)
    add("cohort", "cohort_prompt_compatible_strict_title", "Prompt-compatible strict-title rows", "Use candidate-title strict rows that also render history titles.", True, "medium", changes_paper=True)
    add("cohort", "cohort_full_6000", "Full 6,000 rows", "Include rows with missing titles.", False, "high", allowed=False)
    add("nondeterminism", "nondet_fixed_bounded_repeats", "Fixed bounded repeats", "Use fixed repetitions set before execution.", True, "low")
    add("nondeterminism", "nondet_result_dependent", "Result-dependent repeats", "Repeat until stable or valid.", False, "high", allowed=False, result_dependent=True)
    add("position_bias", "position_deterministic_bounded", "Deterministic bounded permutations", "Separate diagnostic layer with fixed permutations.", True, "medium")
    add("position_bias", "position_random_uncontrolled", "Random uncontrolled permutations", "Generate uncontrolled order variants.", False, "high", allowed=False)
    validate_one_recommended_policy_per_category(rows)
    return frame(rows, POLICY_OPTION_COLUMNS)


def build_blocked_or_conditional(root: Path) -> pd.DataFrame:
    rows = [
        blocked("LBI-001", "model", "GPT-3.5", "", "", "conditional_remote_api", "Closed API model requires key and current provider verification.", "No API query in Phase 5A.", "true", "false", "", "true", "false", "false", "true", "true", "Keep blocked unless explicit API task is approved.", "blocking", "conditional"),
        blocked("LBI-002", "model", "GPT-4o-mini", "", "", "conditional_remote_api", "Closed API model requires key and current provider verification.", "No API query in Phase 5A.", "true", "false", "", "true", "false", "false", "true", "true", "Keep blocked unless explicit API task is approved.", "blocking", "conditional"),
        blocked("LBI-003", "model", "GPT-4o", "", "", "conditional_remote_api", "Closed API model requires key and current provider verification.", "No API query in Phase 5A.", "true", "false", "", "true", "false", "false", "true", "true", "Keep blocked unless explicit API task is approved.", "blocking", "conditional"),
        blocked("LBI-004", "model", "Claude 3 Haiku", "", "", "retired_or_unavailable", "Exact Claude 3 Haiku availability is not verified and prior registry marks retired.", "Requires provider verification.", "true", "false", "", "true", "false", "false", "true", "true", "Do not claim exact reproduction.", "blocking", "blocked"),
        blocked("LBI-005", "model", "Llama3-70B-Instruct", "", "", "hardware_blocked", "70B model exceeds local target hardware.", "Cloud/API would be required.", "true", "false", "", "false", "false", "true", "true", "true", "Do not run locally.", "blocking", "blocked"),
        blocked("LBI-006", "prompt", "", "", "", "official_prompt_conflict", "Official prompt expects top-10 title list; adapted contract uses JSON indices.", "Paper-exact parse grammar is ambiguous for local candidates.", "true", "true", "Adapted JSON-index execution available.", "false", "false", "false", "false", "false", "Use adapted strict contract and label as adapted.", "warning", "documented"),
        blocked("LBI-007", "cohort", "", "Amazon Beauty;Amazon Sports", "ranking;reranking", "missing_history_titles", "Some candidate-title-safe rows lack titles in rendered train history.", "Item-title map has missing history item titles.", "false", "true", "Exclude invalid rendered-history rows from primary prompt-compatible execution.", "false", "false", "false", "false", "false", "Resolve title gaps from repository evidence or retain exclusion.", "warning", "conditional"),
        blocked("LBI-008", "workflow", "", "", "", "profile_generation_ambiguous", "Profile prompt/model/reuse policy not frozen.", "Official templates exist but canonical profile workflow not reconstructed.", "true", "false", "", "false", "false", "false", "false", "true", "Separate future audit before profile generation.", "warning", "blocked"),
    ]
    return frame(rows, BLOCKED_COLUMNS)


def blocked(*values: str) -> dict[str, str]:
    row = {column: "" for column in BLOCKED_COLUMNS}
    for column, value in zip(BLOCKED_COLUMNS, values):
        row[column] = value
    return row


def build_execution_matrix(
    prompt_rendering: pd.DataFrame,
    cohort: pd.DataFrame,
    model_inventory: pd.DataFrame,
    execution_params: pd.DataFrame,
) -> pd.DataFrame:
    valid_render = prompt_rendering.loc[prompt_rendering["rendering_valid"].eq("true")].copy()
    summary = valid_render.groupby(["dataset", "task_type"], sort=True).agg(
        eligible_rows=("user_id", "size"),
        character_count_total=("character_count", lambda s: int(pd.to_numeric(s).sum())),
        whitespace_token_count_total=("whitespace_token_count", lambda s: int(pd.to_numeric(s).sum())),
        estimated_token_min_total=("estimated_token_min", lambda s: int(pd.to_numeric(s).sum())),
        estimated_token_max_total=("estimated_token_max", lambda s: int(pd.to_numeric(s).sum())),
    ).reset_index()
    prompt_contract = {"ranking": "llm_rank_index_json_top5_v1", "reranking": "llm_rerank_index_json_full20_v1"}
    parser_contract = {"ranking": "llm_rank_strict_primary_parser_v1", "reranking": "llm_rerank_strict_primary_parser_v1"}
    rows: list[dict[str, str]] = []
    index = 1
    for model, identifier in {**PAPER_MODELS, **LOCAL_MODELS}.items():
        for record in summary.itertuples(index=False):
            local = model in LOCAL_MODELS
            inventory_status = model_inventory.loc[model_inventory["model_family"].eq(model), "availability_status"].iloc[0]
            if local:
                status = "ready_contract_frozen_no_execution"
                blocking = ""
                scope = "adapted_local_execution"
            elif inventory_status == "blocked_hardware":
                status = "blocked_hardware"
                blocking = "requires hardware beyond local"
                scope = "canonical_paper_like_execution"
            else:
                status = "conditional_remote_api"
                blocking = "requires external API and current provider verification"
                scope = "canonical_paper_like_execution"
            rec_req, total_req = request_count(int(record.eligible_rows))
            rows.append(
                execution_row(
                    index,
                    scope,
                    model,
                    identifier,
                    record.dataset,
                    record.task_type,
                    "plan_e2_llm_primary_prompt_compatible",
                    int(record.eligible_rows),
                    prompt_contract[record.task_type],
                    parser_contract[record.task_type],
                    f"exec_{slug(model)}_{record.task_type}_v1",
                    1,
                    1,
                    1,
                    0,
                    1,
                    rec_req,
                    total_req,
                    int(record.character_count_total),
                    int(record.whitespace_token_count_total),
                    int(record.estimated_token_min_total),
                    int(record.estimated_token_max_total),
                    status,
                    blocking,
                    "request counts only; no request executed",
                )
            )
            index += 1
    ml1m_ranking = summary.loc[(summary["dataset"].eq("ML-1M")) & (summary["task_type"].eq("ranking"))].iloc[0]
    for model, identifier in LOCAL_MODELS.items():
        for scope, reps, perms, hist_vars, profile_reqs, rec_reqs, rows_used, status, notes in [
            ("nondeterminism_extension", 3, 1, 1, 0, 1, 20, "planned_extension_not_executed", "fixed 20-row diagnostic subset"),
            ("candidate_position_bias_diagnostic", 1, 3, 1, 0, 1, 20, "planned_diagnostic_not_executed", "three deterministic candidate-order variants"),
            ("history_length_diagnostic", 1, 1, 3, 0, 1, 20, "planned_diagnostic_not_executed", "short/medium/full history variants"),
            ("profile_generation_workflow", 1, 1, 1, 1, 1, 0, "blocked_ambiguous_profile_contract", "profile workflow requires separate approval"),
        ]:
            rec_req, total_req = request_count(
                rows_used,
                repetitions_per_row=reps,
                position_permutations_per_row=perms,
                history_variants_per_row=hist_vars,
                profile_generation_requests_per_row=profile_reqs,
                recommendation_requests_per_row=rec_reqs,
            )
            scale = rows_used / max(1, int(ml1m_ranking.eligible_rows))
            rows.append(
                execution_row(
                    index,
                    scope,
                    model,
                    identifier,
                    "ML-1M",
                    "ranking",
                    "deterministic_20_row_ml1m_diagnostic_subset",
                    rows_used,
                    prompt_contract["ranking"],
                    parser_contract["ranking"],
                    f"exec_{slug(model)}_ranking_v1",
                    reps,
                    perms,
                    hist_vars,
                    profile_reqs,
                    rec_reqs,
                    rec_req,
                    total_req,
                    int(int(ml1m_ranking.character_count_total) * scale),
                    int(int(ml1m_ranking.whitespace_token_count_total) * scale),
                    int(int(ml1m_ranking.estimated_token_min_total) * scale),
                    int(int(ml1m_ranking.estimated_token_max_total) * scale),
                    status,
                    "" if rows_used else "profile prompt/model/reuse contract ambiguous",
                    notes,
                )
            )
            index += 1
    return frame(rows, EXECUTION_MATRIX_COLUMNS)


def execution_row(index: int, *values: Any) -> dict[str, str]:
    names = EXECUTION_MATRIX_COLUMNS[1:]
    row = {"execution_cell_id": f"LEM-{index:04d}"}
    row.update({name: str(value) for name, value in zip(names, values, strict=True)})
    return row


def write_frames(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for key, data in frames.items():
        path = root / OUTPUT_PATHS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        data.loc[:, OUTPUT_COLUMNS[key]].to_csv(path, index=False, lineterminator="\n")


def validate_frames(frames: Mapping[str, pd.DataFrame]) -> None:
    for key, columns in OUTPUT_COLUMNS.items():
        if key not in frames:
            raise ValueError(f"missing frame {key}")
        if list(frames[key].columns) != columns:
            raise ValueError(f"{key} schema mismatch")
        if frames[key].empty:
            raise ValueError(f"{key} is empty")
    cohort = frames["cohort_preflight"]
    if len(cohort) != 6000:
        raise ValueError(f"cohort row count mismatch: {len(cohort)}")
    strict_count = int(cohort["eligible_strict_title_primary"].eq("true").sum())
    if strict_count != 5814:
        raise ValueError(f"strict-title count mismatch: {strict_count}")
    strict_by_dataset = cohort.loc[cohort["eligible_strict_title_primary"].eq("true")].groupby("dataset").size().to_dict()
    if strict_by_dataset != {"Amazon Beauty": 1988, "Amazon Sports": 1826, "ML-1M": 2000}:
        raise ValueError(f"strict-title dataset counts mismatch: {strict_by_dataset}")
    invalid_prompt = frames["prompt_rendering_audit"].loc[
        frames["prompt_rendering_audit"]["positive_label_exposed"].eq("true")
    ]
    if not invalid_prompt.empty:
        raise ValueError("positive label exposed in prompt rendering audit")
    recommended_nondet = frames["nondeterminism_policy"].loc[
        (frames["nondeterminism_policy"]["execution_scope"].eq("nondeterminism_extension"))
        & (frames["nondeterminism_policy"]["recommendation_status"].eq("recommended"))
    ]
    if len(recommended_nondet) != 1 or not recommended_nondet["result_independent"].eq("true").all():
        raise ValueError("nondeterminism policy must have one result-independent recommendation")
    validate_one_recommended_policy_per_category(frames["policy_options"].to_dict("records"))
    matrix = frames["execution_matrix"]
    for row in matrix.itertuples(index=False):
        expected_base, expected_total = request_count(
            int(row.eligible_rows),
            repetitions_per_row=int(row.repetitions_per_row),
            position_permutations_per_row=int(row.position_permutations_per_row),
            history_variants_per_row=int(row.history_variants_per_row),
            profile_generation_requests_per_row=int(row.profile_generation_requests_per_row),
            recommendation_requests_per_row=int(row.recommendation_requests_per_row),
        )
        if expected_base != int(row.base_request_count) or expected_total != int(row.total_planned_request_count):
            raise ValueError(f"request count mismatch for {row.execution_cell_id}")


def validate_report(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("# Canonical LLM Prompt, Parsing and Execution Contract Preflight"):
        raise ValueError("report title mismatch")
    sections = [line for line in text.splitlines() if line.startswith("## ")]
    if len(sections) != 60:
        raise ValueError(f"expected 60 report sections, found {len(sections)}")


def parse_json_list(value: Any) -> list[str]:
    decoded = json.loads(str(value))
    if not isinstance(decoded, list):
        raise ValueError("expected JSON list")
    return [str(item) for item in decoded]


def normalize_model_family(model: str) -> str:
    lowered = model.lower()
    if "qwen" in lowered:
        return "Qwen2 7B Instruct"
    if "mistral" in lowered:
        return "Mistral 7B Instruct"
    if "gpt" in lowered:
        return "GPT"
    return model or "unknown"


def slug(value: str) -> str:
    return re_safe(value).lower().strip("_")


def re_safe(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{column: str(row.get(column, "")) for column in columns} for row in rows],
        columns=list(columns),
    )

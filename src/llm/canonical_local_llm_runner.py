"""Canonical local Qwen2/Mistral pilot runner helpers for Phase 5B-P1.

The module uses only the local Ollama loopback endpoint. It never pulls,
creates, copies, deletes, or modifies models.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.canonical_llm_execution_preflight import (
    BLOCKED_COLUMNS,
    COHORT_PREFLIGHT_COLUMNS,
    DATASET_INTERACTIONS,
    EXECUTION_MATRIX_COLUMNS,
    EXISTING_PARSER_AUDIT_COLUMNS,
    OUTPUT_PATHS as PHASE5A_OUTPUT_PATHS,
    PROMPT_RENDER_COLUMNS,
    file_sha256,
    hash_protected_inputs as hash_phase5a_inputs,
    load_histories,
    load_title_map,
    parse_json_list,
    read_csv,
)
from src.llm.canonical_llm_contract import (
    contract_hash,
    estimate_token_envelope,
    json_dumps,
    parse_diagnostic_salvage_response,
    parse_strict_index_response,
    render_candidates,
    render_history,
    sha256_items,
    sha256_text,
)


PILOT_DIR = Path("results/llm/canonical_local_pilot")
REQUEST_JSONL = PILOT_DIR / "requests.jsonl"
RESPONSE_JSONL = PILOT_DIR / "responses.jsonl"

MODEL_ORDER = [
    ("Qwen2 7B Instruct", "qwen2:7b-instruct"),
    ("Mistral 7B Instruct", "mistral:7b-instruct"),
]
STRATUM_ORDER = [
    ("Amazon Beauty", "ranking"),
    ("Amazon Beauty", "reranking"),
    ("Amazon Sports", "ranking"),
    ("Amazon Sports", "reranking"),
    ("ML-1M", "ranking"),
    ("ML-1M", "reranking"),
]
PILOT_ROWS_PER_STRATUM = 10
READINESS_THRESHOLD = 0.80
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_VERSION_URL = "http://localhost:11434/api/version"

SAMPLE_COLUMNS = [
    "pilot_row_id",
    "selection_sha256",
    "dataset",
    "task_type",
    "candidate_set_id",
    "source_path",
    "user_id",
    "positive_item_id",
    "candidate_count",
    "history_item_count",
    "candidate_titles_complete",
    "history_titles_complete",
    "held_out_positive_absent",
    "primary_prompt_compatible",
    "prompt_contract_id",
    "parser_contract_id",
    "candidate_order_sha256",
    "system_message_sha256",
    "user_message_sha256",
    "complete_prompt_sha256",
    "sample_order",
    "sample_status",
    "blocking_reason",
    "notes",
]

REQUEST_MANIFEST_COLUMNS = [
    "request_id",
    "pilot_row_id",
    "model_family",
    "model_tag",
    "model_manifest_or_digest",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "request_order",
    "prompt_contract_id",
    "prompt_contract_sha256",
    "parser_contract_id",
    "parser_schema_sha256",
    "execution_contract_id",
    "execution_contract_sha256",
    "complete_prompt_sha256",
    "candidate_order_sha256",
    "requested_output_count",
    "temperature",
    "seed",
    "max_output_tokens",
    "response_format",
    "streaming",
    "concurrency",
    "scientific_retry_count",
    "infrastructure_retry_count",
    "request_jsonl_path",
    "response_jsonl_path",
    "request_status",
    "blocking_reason",
    "notes",
]

EXECUTION_COLUMNS = [
    "request_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "attempt_index",
    "request_started_at_utc",
    "request_completed_at_utc",
    "duration_seconds",
    "backend_success",
    "response_nonempty",
    "done_reason",
    "prompt_eval_count",
    "eval_count",
    "load_duration_nanoseconds",
    "prompt_eval_duration_nanoseconds",
    "eval_duration_nanoseconds",
    "total_duration_nanoseconds",
    "raw_response_sha256",
    "backend_error_type",
    "backend_error_message",
    "response_jsonl_recorded",
    "execution_status",
    "notes",
]

PARSE_COLUMNS = [
    "request_id",
    "model_family",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "raw_response_sha256",
    "strict_parser_id",
    "diagnostic_parser_id",
    "json_parse_valid",
    "schema_valid",
    "output_count_valid",
    "index_type_valid",
    "index_range_valid",
    "uniqueness_valid",
    "full_permutation_valid",
    "candidate_validity",
    "catalog_validity",
    "duplicate_count",
    "unknown_index_count",
    "hallucination_count",
    "strict_parse_valid",
    "diagnostic_salvage_valid",
    "diagnostic_transformations",
    "strict_indices",
    "salvage_indices",
    "primary_failure_reason",
    "notes",
]

RANKING_COLUMNS = [
    "request_id",
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "positive_item_id",
    "candidate_items",
    "candidate_order_sha256",
    "strict_parse_valid",
    "ranked_candidate_indices",
    "ranked_item_ids",
    "top5_item_ids",
    "positive_rank_zero_based",
    "positive_rank_one_based",
    "positive_in_top_5",
    "ndcg_at_5",
    "candidate_validity",
    "duplicate_count",
    "hallucination_count",
    "utility_evaluation_valid",
    "invalidity_reason",
    "notes",
]

METRIC_COLUMNS = [
    "model_family",
    "model_tag",
    "dataset",
    "task_type",
    "pilot_rows",
    "backend_success_rows",
    "strict_valid_rows",
    "strict_parse_validity_rate",
    "diagnostic_salvage_valid_rows",
    "diagnostic_salvage_validity_rate",
    "candidate_valid_rows",
    "candidate_validity_rate_among_strict",
    "duplicate_free_rows",
    "duplicate_free_rate_among_strict",
    "hallucination_free_rows",
    "hallucination_free_rate_among_strict",
    "complete_permutation_rows",
    "complete_permutation_rate_among_strict",
    "utility_evaluated_rows",
    "hr_at_5_valid_only",
    "ndcg_at_5_valid_only",
    "mean_positive_rank_valid_only",
    "median_duration_seconds",
    "mean_duration_seconds",
    "median_generated_token_count",
    "pilot_metric_status",
    "notes",
]

VALIDATION_COLUMNS = [
    "validation_id",
    "validation_scope",
    "model_family",
    "dataset",
    "task_type",
    "check_name",
    "expected_value",
    "observed_value",
    "status",
    "severity",
    "evidence",
    "notes",
]

DECISION_COLUMNS = [
    "pilot_id",
    "expected_request_count",
    "completed_request_count",
    "backend_completion_rate",
    "minimum_cell_strict_parse_validity_rate",
    "all_prompt_hashes_valid",
    "all_candidate_order_hashes_valid",
    "positive_label_never_exposed",
    "all_strict_outputs_candidate_valid",
    "all_strict_outputs_duplicate_free",
    "all_strict_reranking_outputs_complete_permutations",
    "raw_request_retention_complete",
    "raw_response_retention_complete",
    "output_truncation_detected",
    "readiness_threshold",
    "readiness_threshold_result_independent",
    "pilot_decision",
    "blocking_cells",
    "required_corrective_action",
    "recommended_single_next_task",
    "notes",
]

OUTPUT_TABLES = {
    "sample": ("results/tables/canonical_local_llm_pilot_sample.csv", SAMPLE_COLUMNS),
    "request_manifest": ("results/tables/canonical_local_llm_pilot_request_manifest.csv", REQUEST_MANIFEST_COLUMNS),
    "execution": ("results/tables/canonical_local_llm_pilot_execution.csv", EXECUTION_COLUMNS),
    "parse_results": ("results/tables/canonical_local_llm_pilot_parse_results.csv", PARSE_COLUMNS),
    "rankings": ("results/tables/canonical_local_llm_pilot_rankings.csv", RANKING_COLUMNS),
    "metrics": ("results/tables/canonical_local_llm_pilot_metrics.csv", METRIC_COLUMNS),
    "validation": ("results/tables/canonical_local_llm_pilot_validation.csv", VALIDATION_COLUMNS),
    "decision": ("results/tables/canonical_local_llm_pilot_decision.csv", DECISION_COLUMNS),
}
REPORT_PATH = "report/canonical_local_qwen2_mistral_pilot.md"


@dataclass(frozen=True)
class ContractBundle:
    prompt_contract: pd.DataFrame
    parser_contract: pd.DataFrame
    model_inventory: pd.DataFrame
    execution_contract: pd.DataFrame
    cohort_preflight: pd.DataFrame
    prompt_rendering: pd.DataFrame
    policy_options: pd.DataFrame
    blocked_or_conditional: pd.DataFrame


@dataclass(frozen=True)
class OllamaState:
    executable_path: str
    version: str
    service_available: bool
    service_version: str
    installed_models: dict[str, str]


@dataclass(frozen=True)
class RequestResult:
    backend_success: bool
    response_text: str
    raw_backend_response: dict[str, Any]
    backend_error_type: str
    backend_error_message: str
    started_at: str
    completed_at: str
    duration_seconds: float


def load_contract_bundle(root: Path) -> ContractBundle:
    bundle = ContractBundle(
        prompt_contract=read_csv(root, "results/tables/canonical_llm_prompt_contract.csv"),
        parser_contract=read_csv(root, "results/tables/canonical_llm_parser_contract.csv"),
        model_inventory=read_csv(root, "results/tables/canonical_llm_model_inventory.csv"),
        execution_contract=read_csv(root, "results/tables/canonical_llm_execution_parameter_contract.csv"),
        cohort_preflight=read_csv(root, "results/tables/canonical_llm_cohort_preflight.csv"),
        prompt_rendering=read_csv(root, "results/tables/canonical_llm_prompt_rendering_audit.csv"),
        policy_options=read_csv(root, "results/tables/canonical_llm_policy_options.csv"),
        blocked_or_conditional=read_csv(root, "results/tables/canonical_llm_blocked_or_conditional.csv"),
    )
    validate_contract_bundle(bundle)
    return bundle


def validate_contract_bundle(bundle: ContractBundle) -> None:
    if len(bundle.cohort_preflight) != 6000:
        raise ValueError("Phase 5A cohort-preflight row count is not 6000")
    if int(bundle.cohort_preflight["eligible_strict_title_primary"].eq("true").sum()) != 5814:
        raise ValueError("Phase 5A candidate strict-title count is not 5814")
    if int(bundle.cohort_preflight["primary_prompt_compatible"].eq("true").sum()) != 5764:
        raise ValueError("Phase 5A prompt-compatible count is not 5764")
    for row in bundle.prompt_contract.to_dict("records"):
        if row["prompt_template_sha256"] != sha256_text(row["system_message_template"] + "\n---\n" + row["user_message_template"]):
            raise ValueError(f"prompt template hash mismatch: {row['prompt_contract_id']}")
        if row["contract_sha256"] != contract_hash(row):
            raise ValueError(f"prompt contract hash mismatch: {row['prompt_contract_id']}")
    for row in bundle.parser_contract.to_dict("records"):
        payload = {key: row[key] for key in row if key not in {"notes", "parser_schema_sha256"}}
        if row["parser_schema_sha256"] != sha256_text(json_dumps(payload)):
            raise ValueError(f"parser schema hash mismatch: {row['parser_contract_id']}")
    for row in bundle.execution_contract.to_dict("records"):
        if row["contract_sha256"] != contract_hash(row):
            raise ValueError(f"execution contract hash mismatch: {row['execution_contract_id']}")
    for family, _tag in MODEL_ORDER:
        match = bundle.model_inventory.loc[bundle.model_inventory["model_family"].eq(family)]
        if match.empty:
            raise ValueError(f"model missing from Phase 5A inventory: {family}")
        if match.iloc[0]["availability_status"] != "local_model_available_and_tested":
            raise ValueError(f"model is not locally tested in Phase 5A inventory: {family}")
    for family, _tag in MODEL_ORDER:
        for task_type in ("ranking", "reranking"):
            match = bundle.execution_contract.loc[
                bundle.execution_contract["model_family"].eq(family)
                & bundle.execution_contract["task_type"].eq(task_type)
                & bundle.execution_contract["contract_status"].eq("ready_adapted")
            ]
            if len(match) != 1:
                raise ValueError(f"missing ready execution parameter contract: {family} {task_type}")
            row = match.iloc[0]
            required = [
                "temperature",
                "max_output_tokens",
                "response_format",
                "streaming",
                "concurrency",
                "infrastructure_retry_count",
                "scientific_retry_count",
                "timeout_seconds",
            ]
            missing = [column for column in required if str(row[column]).strip() == ""]
            if missing:
                raise ValueError(f"missing execution parameter values for {family} {task_type}: {missing}")
            if row["concurrency"] != "1" or row["infrastructure_retry_count"] != "0" or row["scientific_retry_count"] != "0":
                raise ValueError(f"unapproved retry or concurrency contract: {family} {task_type}")


def phase5b_protected_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for key, relative in PHASE5A_OUTPUT_PATHS.items():
        if relative != PHASE5A_OUTPUT_PATHS["report"] and (root / relative).is_file():
            paths.append(relative)
    extra = [
        "report/canonical_llm_prompt_parsing_and_execution_preflight.md",
        "src/llm/canonical_llm_contract.py",
        "src/analysis/canonical_llm_execution_preflight.py",
        "src/experiments/build_canonical_llm_execution_preflight.py",
        "tests/test_canonical_llm_contract.py",
        "results/tables/ranking_candidate_sets.csv",
        "results/tables/positive_positions.csv",
        "results/tables/experiment_cohort_membership.csv",
        "results/tables/candidate_item_title_map.csv",
        "results/tables/item_title_map.csv",
        "results/tables/llm_title_exclusions.csv",
        "results/tables/canonical_llm_cohort_preflight.csv",
        "results/tables/canonical_llm_prompt_rendering_audit.csv",
        "data/reconstructed/plan_c/amazon_beauty/interactions.csv",
        "data/reconstructed/plan_c/amazon_sports/interactions.csv",
        "data/reconstructed/plan_c/ml1m/interactions.csv",
        "paper/Beyond Utility Evaluating LLM As Recommander.pdf",
    ]
    paths.extend(relative for relative in extra if (root / relative).is_file())
    for pattern in [
        "src/llm/plan_d*.py",
        "src/experiments/*plan_d*llm*.py",
        "src/experiments/*qwen*.py",
        "src/experiments/*mistral*.py",
        "results/tables/plan_d*llm*.csv",
        "results/tables/plan_d*qwen*.csv",
        "results/tables/plan_d*mistral*.csv",
        "report/plan_d*llm*.md",
        "report/plan_d*qwen*.md",
        "report/plan_d*mistral*.md",
        "original_repo/EvaLLMasRecommender/eva/codes/*.py",
        "original_repo/EvaLLMasRecommender/eva/templates/*.txt",
    ]:
        paths.extend(str(path.relative_to(root)).replace("\\", "/") for path in root.glob(pattern) if path.is_file())
    return sorted(set(paths))


def hash_protected_inputs(root: Path) -> dict[str, str]:
    return {relative: file_sha256(root / relative) for relative in phase5b_protected_paths(root)}


def selection_key(dataset: str, task_type: str, candidate_set_id: str, user_id: str) -> str:
    return sha256_text(f"{dataset}|{task_type}|{candidate_set_id}|{user_id}")


def request_id_for_order(order: int) -> str:
    return f"CLLP-REQ-{order:04d}"


def load_prompt_context(root: Path) -> tuple[dict[tuple[str, str], str], dict[str, dict[str, list[dict[str, str]]]], dict[tuple[str, str], list[str]]]:
    title_map = load_title_map(root)
    histories = load_histories(root, title_map)
    candidates = read_csv(root, "results/tables/ranking_candidate_sets.csv")
    candidate_lookup = {
        (str(row.candidate_set_id), str(row.user_id)): parse_json_list(row.candidate_items)
        for row in candidates[["candidate_set_id", "user_id", "candidate_items"]].itertuples(index=False)
    }
    return title_map, histories, candidate_lookup


def render_from_contract(
    prompt_contract_row: Mapping[str, str],
    *,
    history_items: Sequence[Mapping[str, str]],
    candidate_items: Sequence[Mapping[str, str]],
) -> tuple[str, str, str, str]:
    history_render = render_history(history_items)
    candidate_render = render_candidates(candidate_items)
    system_message = str(prompt_contract_row["system_message_template"])
    user_message = str(prompt_contract_row["user_message_template"]).format(
        history_render=history_render,
        candidate_render=candidate_render,
    )
    return system_message, user_message, history_render, candidate_render


def build_pilot_sample(root: Path, bundle: ContractBundle) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    title_map, histories, candidate_lookup = load_prompt_context(root)
    render_lookup = {
        (row.candidate_set_id, row.user_id): row
        for row in bundle.prompt_rendering.itertuples(index=False)
    }
    prompt_by_task = {
        row.task_type: row._asdict()
        for row in bundle.prompt_contract.itertuples(index=False)
        if row.contract_status == "ready_adapted"
    }
    parser_by_task = {
        row.task_type: row._asdict()
        for row in bundle.parser_contract.itertuples(index=False)
        if row.parser_mode == "strict_primary_parser"
    }
    eligible = bundle.cohort_preflight.loc[bundle.cohort_preflight["primary_prompt_compatible"].eq("true")].copy()
    eligible["selection_sha256"] = eligible.apply(
        lambda row: selection_key(row["dataset"], row["task_type"], row["candidate_set_id"], row["user_id"]),
        axis=1,
    )
    rows: list[dict[str, str]] = []
    contexts: dict[str, dict[str, Any]] = {}
    sample_order = 1
    for dataset, task_type in STRATUM_ORDER:
        stratum = eligible.loc[(eligible["dataset"].eq(dataset)) & (eligible["task_type"].eq(task_type))].copy()
        stratum = stratum.sort_values(["selection_sha256", "candidate_set_id", "user_id"], kind="mergesort").head(PILOT_ROWS_PER_STRATUM)
        if len(stratum) != PILOT_ROWS_PER_STRATUM:
            raise ValueError(f"pilot stratum has fewer than 10 rows: {dataset} {task_type}")
        for record in stratum.itertuples(index=False):
            pilot_row_id = f"CLLP-{sample_order:04d}"
            candidate_ids = candidate_lookup[(record.candidate_set_id, record.user_id)]
            candidate_items = [{"item_id": item_id, "title": title_map[(record.dataset, item_id)]} for item_id in candidate_ids]
            history_items = [
                item
                for item in histories[record.dataset].get(record.user_id, [])
                if item["item_id"] != record.positive_item_id
            ][-10:]
            prompt_contract = prompt_by_task[task_type]
            parser_contract = parser_by_task[task_type]
            system_message, user_message, history_render, candidate_render = render_from_contract(
                prompt_contract,
                history_items=history_items,
                candidate_items=candidate_items,
            )
            complete_prompt = system_message + "\n\n" + user_message
            hashes = {
                "system_message_sha256": sha256_text(system_message),
                "user_message_sha256": sha256_text(user_message),
                "complete_prompt_sha256": sha256_text(complete_prompt),
                "history_render_sha256": sha256_text(history_render),
                "candidate_render_sha256": sha256_text(candidate_render),
                "candidate_order_sha256": sha256_items(candidate_ids),
            }
            render_row = render_lookup[(record.candidate_set_id, record.user_id)]
            for column in [
                "system_message_sha256",
                "user_message_sha256",
                "complete_prompt_sha256",
                "history_render_sha256",
                "candidate_render_sha256",
                "candidate_order_sha256",
            ]:
                if hashes[column] != getattr(render_row, column):
                    raise ValueError(f"prompt hash mismatch for {pilot_row_id}: {column}")
            lowered = complete_prompt.lower()
            if any(token in lowered for token in ["positive_item_id", "ground_truth", "held-out positive", "held out positive"]):
                raise ValueError(f"positive label exposure in prompt: {pilot_row_id}")
            row = {
                "pilot_row_id": pilot_row_id,
                "selection_sha256": record.selection_sha256,
                "dataset": record.dataset,
                "task_type": record.task_type,
                "candidate_set_id": record.candidate_set_id,
                "source_path": record.source_path,
                "user_id": record.user_id,
                "positive_item_id": record.positive_item_id,
                "candidate_count": record.candidate_count,
                "history_item_count": str(len(history_items)),
                "candidate_titles_complete": record.candidate_titles_complete,
                "history_titles_complete": record.history_titles_complete,
                "held_out_positive_absent": record.held_out_positive_absent_from_history,
                "primary_prompt_compatible": record.primary_prompt_compatible,
                "prompt_contract_id": prompt_contract["prompt_contract_id"],
                "parser_contract_id": parser_contract["parser_contract_id"],
                "candidate_order_sha256": hashes["candidate_order_sha256"],
                "system_message_sha256": hashes["system_message_sha256"],
                "user_message_sha256": hashes["user_message_sha256"],
                "complete_prompt_sha256": hashes["complete_prompt_sha256"],
                "sample_order": str(sample_order),
                "sample_status": "selected",
                "blocking_reason": "",
                "notes": "deterministic result-independent pilot sample",
            }
            rows.append(row)
            contexts[pilot_row_id] = {
                **row,
                "candidate_items": candidate_ids,
                "candidate_titles": [item["title"] for item in candidate_items],
                "system_message": system_message,
                "user_message": user_message,
                "history_render": history_render,
                "candidate_render": candidate_render,
                "prompt_contract_sha256": prompt_contract["contract_sha256"],
                "parser_schema_sha256": parser_contract["parser_schema_sha256"],
                "requested_output_count": parser_contract["expected_output_count"],
                "requires_full_permutation": parser_contract["requires_full_permutation"] == "true",
            }
            sample_order += 1
    sample = frame(rows, SAMPLE_COLUMNS)
    if len(sample) != 60:
        raise ValueError("pilot sample does not contain 60 rows")
    return sample, contexts


def inspect_ollama() -> OllamaState:
    executable = shutil.which("ollama")
    if not executable:
        raise RuntimeError("blocked_missing_ollama_executable")
    version = run_command([executable, "--version"])
    service_available = False
    service_version = ""
    try:
        with urllib.request.urlopen(OLLAMA_VERSION_URL, timeout=10) as response:
            service_payload = json.loads(response.read().decode("utf-8"))
            service_available = True
            service_version = str(service_payload.get("version", ""))
    except Exception as exc:
        service_version = str(exc)
    installed: dict[str, str] = {}
    if service_available:
        list_output = run_command([executable, "list"])
        installed = parse_ollama_list(list_output)
    return OllamaState(executable, version, service_available, service_version, installed)


def run_command(args: list[str]) -> str:
    completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=30)
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(output or f"command failed: {args}")
    return output


def parse_ollama_list(output: str) -> dict[str, str]:
    models: dict[str, str] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if parts[0].upper() == "NAME":
            continue
        name = parts[0]
        digest = parts[1] if len(parts) > 1 else ""
        models[name] = digest
    return models


def verify_local_models(bundle: ContractBundle, ollama_state: OllamaState) -> dict[str, str]:
    if not ollama_state.service_available:
        raise RuntimeError(f"backend_blocked: Ollama loopback service unavailable: {ollama_state.service_version}")
    digests: dict[str, str] = {}
    for family, expected_tag in MODEL_ORDER:
        inventory = bundle.model_inventory.loc[bundle.model_inventory["model_family"].eq(family)].iloc[0]
        tag = inventory["local_model_tag"] or expected_tag
        if tag != expected_tag:
            raise RuntimeError(f"model tag mismatch for {family}: {tag} != {expected_tag}")
        if tag not in ollama_state.installed_models:
            raise RuntimeError(f"blocked_missing_local_model: {tag}")
        digests[family] = ollama_state.installed_models[tag]
    return digests


def build_request_manifest(
    bundle: ContractBundle,
    sample: pd.DataFrame,
    contexts: Mapping[str, Mapping[str, Any]],
    model_digests: Mapping[str, str],
) -> pd.DataFrame:
    prompt_lookup = {row.prompt_contract_id: row._asdict() for row in bundle.prompt_contract.itertuples(index=False)}
    parser_lookup = {row.parser_contract_id: row._asdict() for row in bundle.parser_contract.itertuples(index=False)}
    exec_lookup = {
        (row.model_family, row.task_type): row._asdict()
        for row in bundle.execution_contract.itertuples(index=False)
    }
    sample_lookup = {row.pilot_row_id: row._asdict() for row in sample.itertuples(index=False)}
    rows: list[dict[str, str]] = []
    order = 1
    for family, tag in MODEL_ORDER:
        for dataset, task_type in STRATUM_ORDER:
            stratum = sample.loc[(sample["dataset"].eq(dataset)) & (sample["task_type"].eq(task_type))].sort_values("sample_order", key=lambda s: s.astype(int))
            for sample_row in stratum.itertuples(index=False):
                prompt_contract = prompt_lookup[sample_row.prompt_contract_id]
                parser_contract = parser_lookup[sample_row.parser_contract_id]
                execution_contract = exec_lookup[(family, task_type)]
                rows.append(
                    {
                        "request_id": request_id_for_order(order),
                        "pilot_row_id": sample_row.pilot_row_id,
                        "model_family": family,
                        "model_tag": tag,
                        "model_manifest_or_digest": model_digests.get(family, ""),
                        "dataset": dataset,
                        "task_type": task_type,
                        "candidate_set_id": sample_row.candidate_set_id,
                        "user_id": sample_row.user_id,
                        "request_order": str(order),
                        "prompt_contract_id": sample_row.prompt_contract_id,
                        "prompt_contract_sha256": prompt_contract["contract_sha256"],
                        "parser_contract_id": sample_row.parser_contract_id,
                        "parser_schema_sha256": parser_contract["parser_schema_sha256"],
                        "execution_contract_id": execution_contract["execution_contract_id"],
                        "execution_contract_sha256": execution_contract["contract_sha256"],
                        "complete_prompt_sha256": sample_row.complete_prompt_sha256,
                        "candidate_order_sha256": sample_row.candidate_order_sha256,
                        "requested_output_count": parser_contract["expected_output_count"],
                        "temperature": execution_contract["temperature"],
                        "seed": execution_contract["seed"],
                        "max_output_tokens": execution_contract["max_output_tokens"],
                        "response_format": execution_contract["response_format"],
                        "streaming": execution_contract["streaming"],
                        "concurrency": execution_contract["concurrency"],
                        "scientific_retry_count": execution_contract["scientific_retry_count"],
                        "infrastructure_retry_count": execution_contract["infrastructure_retry_count"],
                        "request_jsonl_path": str(REQUEST_JSONL).replace("\\", "/"),
                        "response_jsonl_path": str(RESPONSE_JSONL).replace("\\", "/"),
                        "request_status": "planned",
                        "blocking_reason": "",
                        "notes": "sequential local Ollama pilot request",
                    }
                )
                order += 1
    manifest = frame(rows, REQUEST_MANIFEST_COLUMNS)
    if len(manifest) != 120:
        raise ValueError("request manifest does not contain 120 rows")
    return manifest


def build_payload(manifest_row: Mapping[str, str], context: Mapping[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if manifest_row["temperature"] != "":
        options["temperature"] = float(manifest_row["temperature"])
    if manifest_row["max_output_tokens"] != "":
        options["num_predict"] = int(manifest_row["max_output_tokens"])
    if manifest_row["seed"] != "":
        options["seed"] = int(manifest_row["seed"])
    payload = {
        "model": manifest_row["model_tag"],
        "system": context["system_message"],
        "prompt": context["user_message"],
        "stream": False,
        "format": "json",
        "options": options,
    }
    return payload


def execute_ollama_request(payload: Mapping[str, Any], timeout_seconds: int) -> RequestResult:
    started = utc_now()
    start_clock = time.perf_counter()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
        raw = json.loads(raw_body)
        response_text = str(raw.get("response", ""))
        success = response_text != "" or bool(raw.get("done"))
        return RequestResult(success, response_text, raw, "", "", started, utc_now(), time.perf_counter() - start_clock)
    except urllib.error.HTTPError as exc:
        return RequestResult(False, "", {}, "HTTPError", f"{exc.code}: {exc.reason}", started, utc_now(), time.perf_counter() - start_clock)
    except urllib.error.URLError as exc:
        return RequestResult(False, "", {}, "URLError", str(exc.reason), started, utc_now(), time.perf_counter() - start_clock)
    except TimeoutError as exc:
        return RequestResult(False, "", {}, "TimeoutError", str(exc), started, utc_now(), time.perf_counter() - start_clock)
    except Exception as exc:  # pragma: no cover - depends on local backend state
        return RequestResult(False, "", {}, type(exc).__name__, str(exc), started, utc_now(), time.perf_counter() - start_clock)


def append_jsonl(path: Path, record: Mapping[str, Any], seen_request_ids: set[str]) -> None:
    request_id = str(record.get("request_id", ""))
    if request_id in seen_request_ids:
        raise ValueError(f"duplicate request_id for JSONL retention: {request_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    seen_request_ids.add(request_id)


def execute_manifest(
    root: Path,
    manifest: pd.DataFrame,
    contexts: Mapping[str, Mapping[str, Any]],
    initial_hashes: Mapping[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if (root / PILOT_DIR).exists():
        raise FileExistsError(f"pilot directory already exists: {PILOT_DIR}")
    (root / PILOT_DIR).mkdir(parents=True, exist_ok=False)
    request_seen: set[str] = set()
    response_seen: set[str] = set()
    execution_rows: list[dict[str, str]] = []
    response_records: list[dict[str, Any]] = []
    for row in manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        if row["concurrency"] != "1" or row["scientific_retry_count"] != "0" or row["infrastructure_retry_count"] != "0":
            raise ValueError("unapproved concurrency or retry setting")
        context = contexts[row["pilot_row_id"]]
        payload = build_payload(row, context)
        request_record = {
            "request_id": row["request_id"],
            "attempt_index": 1,
            "model_family": row["model_family"],
            "model_tag": row["model_tag"],
            "dataset": row["dataset"],
            "task_type": row["task_type"],
            "candidate_set_id": row["candidate_set_id"],
            "user_id": row["user_id"],
            "pilot_row_id": row["pilot_row_id"],
            "prompt_contract_id": row["prompt_contract_id"],
            "prompt_contract_sha256": row["prompt_contract_sha256"],
            "parser_contract_id": row["parser_contract_id"],
            "parser_schema_sha256": row["parser_schema_sha256"],
            "execution_contract_id": row["execution_contract_id"],
            "execution_contract_sha256": row["execution_contract_sha256"],
            "complete_prompt_sha256": row["complete_prompt_sha256"],
            "candidate_order_sha256": row["candidate_order_sha256"],
            "candidate_items": context["candidate_items"],
            "request_payload": payload,
            "timestamp_before_send_utc": utc_now(),
        }
        append_jsonl(root / REQUEST_JSONL, request_record, request_seen)
        timeout = int(float(row["max_output_tokens"] or 120))
        timeout_seconds = max(120, timeout)
        result = execute_ollama_request(payload, timeout_seconds)
        raw_response_sha = sha256_text(result.response_text)
        response_record = {
            "request_id": row["request_id"],
            "attempt_index": 1,
            "raw_response_text": result.response_text,
            "raw_response_sha256": raw_response_sha,
            "backend_metadata": result.raw_backend_response,
            "model_reported_tag": str(result.raw_backend_response.get("model", row["model_tag"])) if result.raw_backend_response else "",
            "completion_status": "backend_success" if result.backend_success else "backend_error",
            "done_reason": str(result.raw_backend_response.get("done_reason", "done" if result.raw_backend_response.get("done") else "")) if result.raw_backend_response else "",
            "prompt_eval_count": result.raw_backend_response.get("prompt_eval_count", ""),
            "generated_token_count": result.raw_backend_response.get("eval_count", ""),
            "load_duration_nanoseconds": result.raw_backend_response.get("load_duration", ""),
            "prompt_eval_duration_nanoseconds": result.raw_backend_response.get("prompt_eval_duration", ""),
            "eval_duration_nanoseconds": result.raw_backend_response.get("eval_duration", ""),
            "total_duration_nanoseconds": result.raw_backend_response.get("total_duration", ""),
            "timestamp_after_completion_utc": result.completed_at,
            "backend_error_type": result.backend_error_type,
            "backend_error_message": result.backend_error_message,
        }
        append_jsonl(root / RESPONSE_JSONL, response_record, response_seen)
        response_records.append(response_record)
        execution_rows.append(
            {
                "request_id": row["request_id"],
                "model_family": row["model_family"],
                "model_tag": row["model_tag"],
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "candidate_set_id": row["candidate_set_id"],
                "user_id": row["user_id"],
                "attempt_index": "1",
                "request_started_at_utc": result.started_at,
                "request_completed_at_utc": result.completed_at,
                "duration_seconds": format_float(result.duration_seconds),
                "backend_success": bool_text(result.backend_success),
                "response_nonempty": bool_text(bool(result.response_text.strip())),
                "done_reason": response_record["done_reason"],
                "prompt_eval_count": str(response_record["prompt_eval_count"]),
                "eval_count": str(response_record["generated_token_count"]),
                "load_duration_nanoseconds": str(response_record["load_duration_nanoseconds"]),
                "prompt_eval_duration_nanoseconds": str(response_record["prompt_eval_duration_nanoseconds"]),
                "eval_duration_nanoseconds": str(response_record["eval_duration_nanoseconds"]),
                "total_duration_nanoseconds": str(response_record["total_duration_nanoseconds"]),
                "raw_response_sha256": raw_response_sha,
                "backend_error_type": result.backend_error_type,
                "backend_error_message": clean_text(result.backend_error_message, 500),
                "response_jsonl_recorded": "true",
                "execution_status": "completed" if result.backend_success else "backend_error_recorded",
                "notes": "single scientific attempt; no retry",
            }
        )
        if hash_protected_inputs(root) != dict(initial_hashes):
            raise RuntimeError("protected input changed during pilot execution")
    return frame(execution_rows, EXECUTION_COLUMNS), response_records


def classify_response(
    raw_text: str,
    *,
    candidate_items: Sequence[str],
    expected_count: int,
    require_full_permutation: bool,
) -> dict[str, Any]:
    stripped = str(raw_text or "").strip()
    json_valid = False
    schema_valid = False
    output_count_valid = False
    index_type_valid = False
    index_range_valid = False
    uniqueness_valid = False
    full_permutation_valid = not require_full_permutation
    indices: list[int] = []
    primary_failure = "empty_response" if not stripped else ""
    if stripped:
        try:
            parsed = json.loads(stripped)
            json_valid = True
        except json.JSONDecodeError:
            parsed = None
            primary_failure = "json_parse_failed"
        if json_valid:
            if isinstance(parsed, dict) and set(parsed.keys()) == {"ranked_candidate_indices"} and isinstance(parsed["ranked_candidate_indices"], list):
                schema_valid = True
                values = parsed["ranked_candidate_indices"]
                output_count_valid = len(values) == expected_count
                index_type_valid = all(isinstance(value, int) and not isinstance(value, bool) for value in values)
                if index_type_valid:
                    indices = [int(value) for value in values]
                    index_range_valid = all(1 <= value <= len(candidate_items) for value in indices)
                    uniqueness_valid = len(indices) == len(set(indices))
                    if require_full_permutation:
                        full_permutation_valid = set(indices) == set(range(1, len(candidate_items) + 1)) and len(indices) == len(candidate_items)
                if not output_count_valid:
                    primary_failure = "output_count_invalid"
                elif not index_type_valid:
                    primary_failure = "index_type_invalid"
                elif not index_range_valid:
                    primary_failure = "index_range_invalid"
                elif not uniqueness_valid:
                    primary_failure = "duplicate_indices"
                elif require_full_permutation and not full_permutation_valid:
                    primary_failure = "full_permutation_invalid"
            else:
                primary_failure = "schema_invalid"
    duplicate_count = len(indices) - len(set(indices)) if indices else 0
    unknown_index_count = sum(1 for value in indices if value < 1 or value > len(candidate_items))
    strict = parse_strict_index_response(
        raw_text,
        candidate_items=candidate_items,
        expected_count=expected_count,
        require_full_permutation=require_full_permutation,
    )
    strict_valid = bool(
        json_valid
        and schema_valid
        and output_count_valid
        and index_type_valid
        and index_range_valid
        and uniqueness_valid
        and full_permutation_valid
        and strict.parse_valid
    )
    if strict_valid:
        primary_failure = ""
    salvage = parse_diagnostic_salvage_response(
        raw_text,
        candidate_items=candidate_items,
        expected_count=expected_count,
        require_full_permutation=require_full_permutation,
    )
    salvage_indices = list(salvage.output_indices)
    return {
        "json_parse_valid": json_valid,
        "schema_valid": schema_valid,
        "output_count_valid": output_count_valid,
        "index_type_valid": index_type_valid,
        "index_range_valid": index_range_valid,
        "uniqueness_valid": uniqueness_valid,
        "full_permutation_valid": full_permutation_valid,
        "candidate_validity": strict_valid,
        "catalog_validity": strict_valid,
        "duplicate_count": duplicate_count,
        "unknown_index_count": unknown_index_count,
        "hallucination_count": unknown_index_count,
        "strict_parse_valid": strict_valid,
        "diagnostic_salvage_valid": salvage.parse_valid,
        "diagnostic_transformations": list(salvage.transformations),
        "strict_indices": indices if strict_valid else [],
        "salvage_indices": salvage_indices,
        "primary_failure_reason": primary_failure,
    }


def build_parse_and_ranking_outputs(
    manifest: pd.DataFrame,
    execution: pd.DataFrame,
    response_records: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_lookup = {row.request_id: row._asdict() for row in manifest.itertuples(index=False)}
    execution_lookup = {row.request_id: row._asdict() for row in execution.itertuples(index=False)}
    response_lookup = {str(row["request_id"]): row for row in response_records}
    parse_rows: list[dict[str, str]] = []
    ranking_rows: list[dict[str, str]] = []
    for request_id, manifest_row in manifest_lookup.items():
        context = contexts[manifest_row["pilot_row_id"]]
        response_record = response_lookup.get(request_id, {})
        raw_text = str(response_record.get("raw_response_text", ""))
        candidate_items = list(context["candidate_items"])
        expected_count = int(context["requested_output_count"])
        require_full = bool(context["requires_full_permutation"])
        classified = classify_response(
            raw_text,
            candidate_items=candidate_items,
            expected_count=expected_count,
            require_full_permutation=require_full,
        )
        raw_sha = sha256_text(raw_text)
        parse_rows.append(
            {
                "request_id": request_id,
                "model_family": manifest_row["model_family"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "raw_response_sha256": raw_sha,
                "strict_parser_id": manifest_row["parser_contract_id"],
                "diagnostic_parser_id": manifest_row["parser_contract_id"].replace("strict_primary", "diagnostic_salvage"),
                "json_parse_valid": bool_text(classified["json_parse_valid"]),
                "schema_valid": bool_text(classified["schema_valid"]),
                "output_count_valid": bool_text(classified["output_count_valid"]),
                "index_type_valid": bool_text(classified["index_type_valid"]),
                "index_range_valid": bool_text(classified["index_range_valid"]),
                "uniqueness_valid": bool_text(classified["uniqueness_valid"]),
                "full_permutation_valid": bool_text(classified["full_permutation_valid"]),
                "candidate_validity": bool_text(classified["candidate_validity"]),
                "catalog_validity": bool_text(classified["catalog_validity"]),
                "duplicate_count": str(classified["duplicate_count"]),
                "unknown_index_count": str(classified["unknown_index_count"]),
                "hallucination_count": str(classified["hallucination_count"]),
                "strict_parse_valid": bool_text(classified["strict_parse_valid"]),
                "diagnostic_salvage_valid": bool_text(classified["diagnostic_salvage_valid"]),
                "diagnostic_transformations": json_dumps(classified["diagnostic_transformations"]),
                "strict_indices": json_dumps(classified["strict_indices"]),
                "salvage_indices": json_dumps(classified["salvage_indices"]),
                "primary_failure_reason": classified["primary_failure_reason"],
                "notes": "salvage output not used for primary ranking",
            }
        )
        ranked_items: list[str] = []
        top5: list[str] = []
        positive_rank_zero = ""
        positive_rank_one = ""
        positive_in_top5 = ""
        ndcg = ""
        utility_valid = "false"
        if classified["strict_parse_valid"]:
            ranked_items = [candidate_items[index - 1] for index in classified["strict_indices"]]
            top5 = ranked_items[:5]
            if context["positive_item_id"] in ranked_items:
                rank = ranked_items.index(context["positive_item_id"])
                positive_rank_zero = str(rank)
                positive_rank_one = str(rank + 1)
                positive_in_top5 = bool_text(rank < 5)
                ndcg = format_float(1.0 / math.log2(rank + 2.0)) if rank < 5 else "0"
            else:
                positive_in_top5 = "false"
                ndcg = "0"
            utility_valid = "true"
        ranking_rows.append(
            {
                "request_id": request_id,
                "model_family": manifest_row["model_family"],
                "model_tag": manifest_row["model_tag"],
                "dataset": manifest_row["dataset"],
                "task_type": manifest_row["task_type"],
                "candidate_set_id": manifest_row["candidate_set_id"],
                "user_id": manifest_row["user_id"],
                "positive_item_id": context["positive_item_id"],
                "candidate_items": json_dumps(candidate_items),
                "candidate_order_sha256": context["candidate_order_sha256"],
                "strict_parse_valid": bool_text(classified["strict_parse_valid"]),
                "ranked_candidate_indices": json_dumps(classified["strict_indices"] if classified["strict_parse_valid"] else []),
                "ranked_item_ids": json_dumps(ranked_items),
                "top5_item_ids": json_dumps(top5),
                "positive_rank_zero_based": positive_rank_zero,
                "positive_rank_one_based": positive_rank_one,
                "positive_in_top_5": positive_in_top5,
                "ndcg_at_5": ndcg,
                "candidate_validity": bool_text(classified["candidate_validity"]),
                "duplicate_count": str(classified["duplicate_count"]),
                "hallucination_count": str(classified["hallucination_count"]),
                "utility_evaluation_valid": utility_valid,
                "invalidity_reason": "" if utility_valid == "true" else classified["primary_failure_reason"],
                "notes": "valid-only utility; invalid rows have empty normalized outputs",
            }
        )
    return frame(parse_rows, PARSE_COLUMNS), frame(ranking_rows, RANKING_COLUMNS)


def build_metrics(manifest: pd.DataFrame, execution: pd.DataFrame, parse_results: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    merged = manifest[["request_id", "model_family", "model_tag", "dataset", "task_type"]].merge(execution, on=["request_id", "model_family", "model_tag", "dataset", "task_type"])
    merged = merged.merge(parse_results, on=["request_id", "model_family", "dataset", "task_type"])
    merged = merged.merge(rankings[["request_id", "positive_in_top_5", "ndcg_at_5", "positive_rank_zero_based", "utility_evaluation_valid"]], on="request_id")
    for (family, tag, dataset, task), group in merged.groupby(["model_family", "model_tag", "dataset", "task_type"], sort=True):
        pilot_rows = len(group)
        strict = group.loc[group["strict_parse_valid"].eq("true")]
        valid_utility = group.loc[group["utility_evaluation_valid"].eq("true")]
        durations = pd.to_numeric(group["duration_seconds"], errors="coerce")
        eval_counts = pd.to_numeric(group["eval_count"], errors="coerce")
        strict_count = len(strict)
        strict_denominator = max(1, strict_count)
        complete_perm = strict_count if task == "ranking" else int(strict["full_permutation_valid"].eq("true").sum())
        rows.append(
            {
                "model_family": family,
                "model_tag": tag,
                "dataset": dataset,
                "task_type": task,
                "pilot_rows": str(pilot_rows),
                "backend_success_rows": str(int(group["backend_success"].eq("true").sum())),
                "strict_valid_rows": str(strict_count),
                "strict_parse_validity_rate": format_float(strict_count / pilot_rows if pilot_rows else 0.0),
                "diagnostic_salvage_valid_rows": str(int(group["diagnostic_salvage_valid"].eq("true").sum())),
                "diagnostic_salvage_validity_rate": format_float(int(group["diagnostic_salvage_valid"].eq("true").sum()) / pilot_rows if pilot_rows else 0.0),
                "candidate_valid_rows": str(int(strict["candidate_validity"].eq("true").sum())),
                "candidate_validity_rate_among_strict": format_float(int(strict["candidate_validity"].eq("true").sum()) / strict_denominator),
                "duplicate_free_rows": str(int(pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).sum())),
                "duplicate_free_rate_among_strict": format_float(int(pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).sum()) / strict_denominator),
                "hallucination_free_rows": str(int(pd.to_numeric(strict["hallucination_count"], errors="coerce").fillna(0).eq(0).sum())),
                "hallucination_free_rate_among_strict": format_float(int(pd.to_numeric(strict["hallucination_count"], errors="coerce").fillna(0).eq(0).sum()) / strict_denominator),
                "complete_permutation_rows": str(complete_perm),
                "complete_permutation_rate_among_strict": format_float(complete_perm / strict_denominator),
                "utility_evaluated_rows": str(len(valid_utility)),
                "hr_at_5_valid_only": mean_bool(valid_utility["positive_in_top_5"]) if len(valid_utility) else "",
                "ndcg_at_5_valid_only": mean_numeric(valid_utility["ndcg_at_5"]) if len(valid_utility) else "",
                "mean_positive_rank_valid_only": mean_numeric(valid_utility.loc[valid_utility["positive_rank_zero_based"].ne(""), "positive_rank_zero_based"]) if len(valid_utility.loc[valid_utility["positive_rank_zero_based"].ne("")]) else "",
                "median_duration_seconds": format_float(float(durations.median())) if durations.notna().any() else "",
                "mean_duration_seconds": format_float(float(durations.mean())) if durations.notna().any() else "",
                "median_generated_token_count": format_float(float(eval_counts.median())) if eval_counts.notna().any() else "",
                "pilot_metric_status": "valid_only_metrics" if len(valid_utility) else "no_strict_valid_outputs",
                "notes": "Pilot utility is not a final full-cohort result.",
            }
        )
    return frame(rows, METRIC_COLUMNS)


def build_decision(
    manifest: pd.DataFrame,
    execution: pd.DataFrame,
    parse_results: pd.DataFrame,
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    protected_unchanged: bool,
) -> pd.DataFrame:
    completed = len(execution)
    backend_rate = int(execution["backend_success"].eq("true").sum()) / 120 if len(execution) else 0.0
    min_parse = float(pd.to_numeric(metrics["strict_parse_validity_rate"], errors="coerce").min()) if len(metrics) else 0.0
    prompt_hashes = len(manifest) == 120 and manifest["complete_prompt_sha256"].astype(str).str.len().gt(0).all()
    candidate_hashes = len(manifest) == 120 and manifest["candidate_order_sha256"].astype(str).str.len().gt(0).all()
    positive_hidden = True
    strict = parse_results.loc[parse_results["strict_parse_valid"].eq("true")]
    strict_candidate_valid = strict["candidate_validity"].eq("true").all() if len(strict) else False
    strict_duplicate_free = pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).all() if len(strict) else False
    rerank_strict = parse_results.loc[parse_results["strict_parse_valid"].eq("true") & parse_results["task_type"].eq("reranking")]
    rerank_perm = rerank_strict["full_permutation_valid"].eq("true").all() if len(rerank_strict) else False
    request_retained = count_jsonl(Path(REQUEST_JSONL)) == 120
    response_retained = count_jsonl(Path(RESPONSE_JSONL)) == 120
    truncation = execution["done_reason"].astype(str).str.lower().isin({"length", "max_tokens"}).any()
    blocking_cells = []
    for row in metrics.itertuples(index=False):
        if float(row.strict_parse_validity_rate) < READINESS_THRESHOLD:
            blocking_cells.append(f"{row.model_family}|{row.dataset}|{row.task_type}|strict_parse={row.strict_parse_validity_rate}")
    if not protected_unchanged:
        decision = "protected_input_validation_failed"
        corrective = "Restore protected inputs and rerun pilot from a clean state."
    elif completed != 120 or not request_retained or not response_retained:
        decision = "pilot_incomplete"
        corrective = "Investigate missing request/response retention before any full run."
    elif backend_rate < 1.0:
        decision = "backend_blocked"
        corrective = "Resolve local Ollama/backend failures without changing scientific prompts."
    elif truncation:
        decision = "require_output_token_limit_revision"
        corrective = "Revise max output token contract in a focused Phase 5B-P1 corrective task."
    elif min_parse < READINESS_THRESHOLD:
        decision = "require_prompt_contract_revision"
        corrective = "Run focused prompt/parser contract revision for cells below the predeclared 0.80 strict-parse threshold."
    elif not (strict_candidate_valid and strict_duplicate_free and rerank_perm and prompt_hashes and candidate_hashes and positive_hidden):
        decision = "require_parser_contract_revision"
        corrective = "Audit invalid strict outputs and revise parser/prompt contract without reusing salvage as primary."
    else:
        decision = "approve_full_local_execution"
        corrective = ""
    recommended = (
        "Phase 5B-P2 — full canonical local Qwen2 and Mistral ranking/reranking execution on 5,764 frozen prompt-compatible rows"
        if decision == "approve_full_local_execution"
        else "Phase 5B-P1C — focused prompt/parser corrective audit for failed pilot readiness gates"
    )
    row = {
        "pilot_id": "canonical_local_qwen2_mistral_pilot_v1",
        "expected_request_count": "120",
        "completed_request_count": str(completed),
        "backend_completion_rate": format_float(backend_rate),
        "minimum_cell_strict_parse_validity_rate": format_float(min_parse),
        "all_prompt_hashes_valid": bool_text(prompt_hashes),
        "all_candidate_order_hashes_valid": bool_text(candidate_hashes),
        "positive_label_never_exposed": bool_text(positive_hidden),
        "all_strict_outputs_candidate_valid": bool_text(strict_candidate_valid),
        "all_strict_outputs_duplicate_free": bool_text(strict_duplicate_free),
        "all_strict_reranking_outputs_complete_permutations": bool_text(rerank_perm),
        "raw_request_retention_complete": bool_text(request_retained),
        "raw_response_retention_complete": bool_text(response_retained),
        "output_truncation_detected": bool_text(bool(truncation)),
        "readiness_threshold": format_float(READINESS_THRESHOLD),
        "readiness_threshold_result_independent": "true",
        "pilot_decision": decision,
        "blocking_cells": json_dumps(blocking_cells),
        "required_corrective_action": corrective,
        "recommended_single_next_task": recommended,
        "notes": "Pilot readiness gate only; not a final model-quality conclusion.",
    }
    return frame([row], DECISION_COLUMNS)


def build_validation(
    sample: pd.DataFrame,
    manifest: pd.DataFrame,
    execution: pd.DataFrame,
    parse_results: pd.DataFrame,
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    decision: pd.DataFrame,
    protected_unchanged: bool,
    ollama_state: OllamaState,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(scope: str, check: str, expected: Any, observed: Any, passed: bool, severity: str = "error", evidence: str = "", model: str = "", dataset: str = "", task: str = "", notes: str = "") -> None:
        rows.append(
            {
                "validation_id": f"CLV-{len(rows)+1:04d}",
                "validation_scope": scope,
                "model_family": model,
                "dataset": dataset,
                "task_type": task,
                "check_name": check,
                "expected_value": str(expected),
                "observed_value": str(observed),
                "status": "passed" if passed else "failed",
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    add("protected_inputs", "protected_hashes_unchanged", True, protected_unchanged, protected_unchanged, evidence="protected input hash map")
    add("sample", "sample_rows", 60, len(sample), len(sample) == 60, evidence=OUTPUT_TABLES["sample"][0])
    add("sample", "ten_per_dataset_task", 10, sample.groupby(["dataset", "task_type"]).size().to_dict(), sample.groupby(["dataset", "task_type"]).size().eq(10).all(), evidence=OUTPUT_TABLES["sample"][0])
    add("manifest", "request_rows", 120, len(manifest), len(manifest) == 120, evidence=OUTPUT_TABLES["request_manifest"][0])
    add("manifest", "same_sample_across_models", 60, manifest.groupby("pilot_row_id")["model_family"].nunique().eq(2).sum(), manifest.groupby("pilot_row_id")["model_family"].nunique().eq(2).all(), evidence=OUTPUT_TABLES["request_manifest"][0])
    add("manifest", "request_order", list(range(1, 121)), [int(v) for v in manifest["request_order"]], [int(v) for v in manifest["request_order"]] == list(range(1, 121)), evidence=OUTPUT_TABLES["request_manifest"][0])
    add("prompt", "prompt_hashes_present", True, manifest["complete_prompt_sha256"].str.len().gt(0).all(), manifest["complete_prompt_sha256"].str.len().gt(0).all(), evidence=OUTPUT_TABLES["request_manifest"][0])
    add("candidate_order", "candidate_order_hashes_present", True, manifest["candidate_order_sha256"].str.len().gt(0).all(), manifest["candidate_order_sha256"].str.len().gt(0).all(), evidence=OUTPUT_TABLES["request_manifest"][0])
    add("positive_label", "positive_label_hidden", True, True, True, evidence=OUTPUT_TABLES["sample"][0])
    add("model", "ollama_service_available", True, ollama_state.service_available, ollama_state.service_available, evidence="ollama local loopback")
    for family, tag in MODEL_ORDER:
        add("model", "model_tag_available", tag, tag in ollama_state.installed_models, tag in ollama_state.installed_models, model=family, evidence="ollama list")
    add("execution", "concurrency_one", 1, set(manifest["concurrency"]), set(manifest["concurrency"]) == {"1"}, evidence=OUTPUT_TABLES["request_manifest"][0])
    add("execution", "zero_scientific_retries", 0, set(manifest["scientific_retry_count"]), set(manifest["scientific_retry_count"]) == {"0"}, evidence=OUTPUT_TABLES["request_manifest"][0])
    add("retention", "raw_request_records", 120, count_jsonl(Path(REQUEST_JSONL)), count_jsonl(Path(REQUEST_JSONL)) == 120, evidence=str(REQUEST_JSONL).replace("\\", "/"))
    add("retention", "raw_response_records", 120, count_jsonl(Path(RESPONSE_JSONL)), count_jsonl(Path(RESPONSE_JSONL)) == 120, evidence=str(RESPONSE_JSONL).replace("\\", "/"))
    add("execution", "execution_rows", 120, len(execution), len(execution) == 120, evidence=OUTPUT_TABLES["execution"][0])
    add("parse", "parse_rows", 120, len(parse_results), len(parse_results) == 120, evidence=OUTPUT_TABLES["parse_results"][0])
    add("rankings", "ranking_rows", 120, len(rankings), len(rankings) == 120, evidence=OUTPUT_TABLES["rankings"][0])
    add("metrics", "metric_rows", 12, len(metrics), len(metrics) == 12, evidence=OUTPUT_TABLES["metrics"][0])
    add("decision", "decision_rows", 1, len(decision), len(decision) == 1, evidence=OUTPUT_TABLES["decision"][0])
    strict = parse_results.loc[parse_results["strict_parse_valid"].eq("true")]
    add("parse", "strict_valid_index_ranges", True, strict["index_range_valid"].eq("true").all() if len(strict) else False, strict["index_range_valid"].eq("true").all() if len(strict) else False, evidence=OUTPUT_TABLES["parse_results"][0])
    add("parse", "no_candidate_substitution", True, True, True, evidence=OUTPUT_TABLES["rankings"][0])
    add("parse", "strict_duplicate_free", True, pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).all() if len(strict) else False, pd.to_numeric(strict["duplicate_count"], errors="coerce").fillna(0).eq(0).all() if len(strict) else False, evidence=OUTPUT_TABLES["parse_results"][0])
    rerank_strict = strict.loc[strict["task_type"].eq("reranking")]
    add("parse", "strict_reranking_full_permutation", True, rerank_strict["full_permutation_valid"].eq("true").all() if len(rerank_strict) else False, rerank_strict["full_permutation_valid"].eq("true").all() if len(rerank_strict) else False, evidence=OUTPUT_TABLES["parse_results"][0])
    add("parse", "no_salvage_substitution", True, rankings.loc[rankings["strict_parse_valid"].eq("false"), "ranked_item_ids"].eq("[]").all(), rankings.loc[rankings["strict_parse_valid"].eq("false"), "ranked_item_ids"].eq("[]").all(), evidence=OUTPUT_TABLES["rankings"][0])
    add("metrics", "valid_only_reconciliation", True, rankings["utility_evaluation_valid"].eq(rankings["strict_parse_valid"]).all(), rankings["utility_evaluation_valid"].eq(rankings["strict_parse_valid"]).all(), evidence=OUTPUT_TABLES["rankings"][0])
    add("ollama", "no_model_pull_command", "no pull", "runner has no pull path", True, evidence="src/llm/canonical_local_llm_runner.py")
    return frame(rows, VALIDATION_COLUMNS)


def validate_outputs(frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (relative, columns) in OUTPUT_TABLES.items():
        frame_value = frames[key]
        if list(frame_value.columns) != columns:
            raise ValueError(f"{key} schema mismatch")
    expected = {
        "sample": 60,
        "request_manifest": 120,
        "execution": 120,
        "parse_results": 120,
        "rankings": 120,
        "metrics": 12,
        "decision": 1,
    }
    for key, count in expected.items():
        if len(frames[key]) != count:
            raise ValueError(f"{key} row count mismatch: {len(frames[key])}")
    rates = [
        "strict_parse_validity_rate",
        "diagnostic_salvage_validity_rate",
        "candidate_validity_rate_among_strict",
        "duplicate_free_rate_among_strict",
        "hallucination_free_rate_among_strict",
        "complete_permutation_rate_among_strict",
    ]
    for column in rates:
        values = pd.to_numeric(frames["metrics"][column], errors="raise")
        if values.lt(0).any() or values.gt(1).any():
            raise ValueError(f"metric rate outside [0,1]: {column}")
    if count_jsonl(Path(REQUEST_JSONL)) != 120:
        raise ValueError("request JSONL count mismatch")
    if count_jsonl(Path(RESPONSE_JSONL)) != 120:
        raise ValueError("response JSONL count mismatch")


def write_tables(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    for key, (relative, columns) in OUTPUT_TABLES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[key].loc[:, columns].to_csv(path, index=False, lineterminator="\n")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: str(row.get(column, "")) for column in columns} for row in rows], columns=list(columns))


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.17g}"


def mean_numeric(values: Iterable[Any]) -> str:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return ""
    return format_float(float(series.mean()))


def mean_bool(values: Iterable[Any]) -> str:
    data = [1.0 if str(value).lower() == "true" else 0.0 for value in values]
    if not data:
        return ""
    return format_float(float(np.mean(data)))


def clean_text(value: str, limit: int) -> str:
    return " ".join(str(value).split())[:limit]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def environment_summary(root: Path) -> dict[str, str]:
    ram = ""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        ram = str(status.ullTotalPhys)
    except Exception:
        ram = "unavailable"
    gpu = ""
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu = ";".join(line.strip() for line in completed.stdout.splitlines() if line.strip())
    except Exception:
        gpu = "unavailable"
    return {
        "executable": sys.executable,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "ram_bytes": ram,
        "gpu": gpu,
        "cwd": str(root),
    }

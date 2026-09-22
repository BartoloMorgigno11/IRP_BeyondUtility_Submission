"""Phase 5B-P2 full local LLM matrix preflight.

This module is offline. It reconciles the frozen prompt-compatible cohort and
builds deterministic full-matrix request and chunk manifests, but it never
calls Ollama or any external service.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.audit import common_validation
from src.analysis.canonical_llm_execution_preflight import (
    COHORT_PREFLIGHT_COLUMNS,
    EXECUTION_PARAMETER_COLUMNS,
    PROMPT_RENDER_COLUMNS,
    file_sha256,
    load_histories,
    load_title_map,
    parse_json_list,
    read_csv,
)
from src.analysis.canonical_local_llm_top10_confirmation_preflight import (
    TOP10_PARSER_CONTRACT_COLUMNS,
    TOP10_PROMPT_CONTRACT_COLUMNS,
    TOP10_PARSER_ID,
    TOP10_PROMPT_ID,
    contract_hash as top10_contract_hash,
)
from src.llm.canonical_llm_contract import (
    PARSER_CONTRACT_COLUMNS,
    PROMPT_CONTRACT_COLUMNS,
    contract_hash,
    json_dumps,
    render_candidates,
    render_history,
    sha256_items,
    sha256_text,
)


PHASE_ID = "canonical_local_llm_full_matrix_v1"

DATASET_ORDER = ["Amazon Beauty", "Amazon Sports", "ML-1M"]
MODEL_ORDER = ["Qwen2 7B Instruct", "Mistral 7B Instruct"]
TASK_ORDER = ["ranking", "reranking"]
CHUNK_SIZE = 256
EXPECTED_FULL_CANDIDATE_ROWS = 6000
EXPECTED_STRICT_TITLE_ROWS = 5814
EXPECTED_HISTORY_TITLE_EXCLUSIONS = 50
EXPECTED_PROMPT_COMPATIBLE_ROWS = 5764
EXPECTED_TASK_ROWS = 2882
EXPECTED_REQUESTS = 11528
EXPECTED_CHUNKS = 46

RANKING_PROMPT_ID = "llm_rank_index_json_top5_v1"
RANKING_PARSER_ID = "llm_rank_strict_primary_parser_v1"
RERANKING_PROMPT_ID = TOP10_PROMPT_ID
RERANKING_PARSER_ID = TOP10_PARSER_ID

EXPECTED_DIGESTS = {
    "Qwen2 7B Instruct": {
        "tag": "qwen2:7b-instruct",
        "digest": "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755",
    },
    "Mistral 7B Instruct": {
        "tag": "mistral:7b-instruct",
        "digest": "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf",
    },
}

COHORT_PATH = Path("results/tables/canonical_llm_cohort_preflight.csv")
PROMPT_RENDERING_PATH = Path("results/tables/canonical_llm_prompt_rendering_audit.csv")
PHASE5A_PROMPT_CONTRACT_PATH = Path("results/tables/canonical_llm_prompt_contract.csv")
PHASE5A_PARSER_CONTRACT_PATH = Path("results/tables/canonical_llm_parser_contract.csv")
EXECUTION_CONTRACT_PATH = Path("results/tables/canonical_llm_execution_parameter_contract.csv")
MODEL_INVENTORY_PATH = Path("results/tables/canonical_llm_model_inventory.csv")
PHASE5A_EXECUTION_MATRIX_PATH = Path("results/tables/canonical_llm_execution_matrix.csv")
PHASE5A_PARAMETER_CONTRACT_PATH = EXECUTION_CONTRACT_PATH
TOP10_PROMPT_CONTRACT_PATH = Path("results/tables/canonical_local_llm_reranking_top10_prompt_contract.csv")
TOP10_PARSER_CONTRACT_PATH = Path("results/tables/canonical_local_llm_reranking_top10_parser_contract.csv")
TOP10_SOURCE_AUDIT_PATH = Path("results/tables/canonical_local_llm_reranking_top10_source_contract_audit.csv")
COUNT_RECONCILIATION_PATH = Path("results/tables/canonical_local_llm_prompt_compatible_count_reconciliation.csv")
P1F_DECISION_PATH = Path("results/tables/canonical_local_llm_final_reranking_protocol_decision.csv")
P1G_DECISION_PATH = Path("results/tables/canonical_local_llm_top10_confirmation_decision.csv")
P1G_REQUESTS_PATH = Path("results/llm/canonical_local_top10_confirmation_pilot/requests.jsonl")
P1G_RESPONSES_PATH = Path("results/llm/canonical_local_top10_confirmation_pilot/responses.jsonl")
P1G_REPORT_PATH = Path("report/canonical_local_llm_top10_reranking_confirmation_pilot.md")
RANKING_CANDIDATES_PATH = Path("results/tables/ranking_candidate_sets.csv")
POSITIVE_POSITIONS_PATH = Path("results/tables/positive_positions.csv")
COHORT_MEMBERSHIP_PATH = Path("results/tables/experiment_cohort_membership.csv")
ITEM_TITLE_MAP_PATH = Path("results/tables/item_title_map.csv")
ITEM_TITLE_COVERAGE_PATH = Path("results/tables/item_title_coverage_summary.csv")
TITLE_EXCLUSIONS_PATH = Path("results/tables/llm_title_exclusions.csv")
DATASET_STATISTICS_PATH = Path("results/tables/dataset_statistics.csv")
OLLAMA_RECOVERY_PATH = Path("results/tables/canonical_local_ollama_service_recovery.csv")
README_PATH = Path("README.md")
REQUIREMENTS_PATH = Path("requirements.txt")
ENVIRONMENT_PATH = Path("environment.yml")
PAPER_PATH = Path("paper/Beyond Utility Evaluating LLM As Recommander.pdf")

INTERACTION_PATHS = [
    Path("data/reconstructed/plan_c/amazon_beauty/interactions.csv"),
    Path("data/reconstructed/plan_c/amazon_sports/interactions.csv"),
    Path("data/reconstructed/plan_c/ml1m/interactions.csv"),
]

MANIFEST_PATH = Path("results/tables/canonical_local_llm_full_matrix_manifest.csv")
CHUNK_MANIFEST_PATH = Path("results/tables/canonical_local_llm_full_matrix_chunk_manifest.csv")
PREFLIGHT_VALIDATION_PATH = Path("results/tables/canonical_local_llm_full_matrix_preflight_validation.csv")
PREFLIGHT_DECISION_PATH = Path("results/tables/canonical_local_llm_full_matrix_preflight_decision.csv")

MANIFEST_COLUMNS = [
    "request_id",
    "request_order",
    "chunk_id",
    "model_family",
    "model_tag",
    "model_digest",
    "dataset",
    "task_type",
    "candidate_set_id",
    "user_id",
    "source_path",
    "prompt_contract_id",
    "prompt_contract_sha256",
    "parser_contract_id",
    "parser_contract_sha256",
    "execution_contract_id",
    "execution_contract_sha256",
    "candidate_order_sha256",
    "history_render_sha256",
    "candidate_count",
    "requested_output_count",
    "temperature",
    "seed",
    "max_output_tokens",
    "response_format",
    "streaming",
    "concurrency",
    "scientific_retry_count",
    "infrastructure_retry_count",
    "primary_prompt_compatible",
    "manifest_status",
    "blocking_reason",
    "notes",
]

CHUNK_MANIFEST_COLUMNS = [
    "chunk_id",
    "chunk_order",
    "start_request_order",
    "end_request_order",
    "planned_request_count",
    "chunk_status",
    "blocking_reason",
    "notes",
]

PREFLIGHT_VALIDATION_COLUMNS = [
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

PREFLIGHT_DECISION_COLUMNS = [
    "preflight_id",
    "full_candidate_rows",
    "candidate_strict_title_rows",
    "history_title_exclusion_rows",
    "prompt_compatible_rows",
    "prompt_compatible_ranking_rows",
    "prompt_compatible_reranking_rows",
    "full_manifest_rows",
    "chunk_rows",
    "expected_request_count",
    "manifest_sha256",
    "chunk_manifest_sha256",
    "protected_input_count",
    "ranking_prompt_contract_id",
    "reranking_prompt_contract_id",
    "ranking_parser_contract_id",
    "reranking_parser_contract_id",
    "preflight_decision",
    "blocking_reason",
    "recommended_single_next_task",
    "notes",
]


@dataclass(frozen=True)
class ContractBundle:
    ranking_prompt: dict[str, str]
    reranking_prompt: dict[str, str]
    ranking_parser: dict[str, str]
    reranking_parser: dict[str, str]
    execution_contracts: dict[tuple[str, str], dict[str, str]]


def frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: str(row.get(column, "")) for column in columns} for row in rows], columns=list(columns))


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def dataset_sort_key(value: str) -> int:
    return DATASET_ORDER.index(str(value))


def task_sort_key(value: str) -> int:
    return TASK_ORDER.index(str(value))


def chunk_id_for_order(order: int) -> str:
    return f"CLFULL-CHUNK-{((int(order) - 1) // CHUNK_SIZE) + 1:03d}"


def protected_input_paths(root: Path) -> list[str]:
    explicit = [
        PHASE5A_PROMPT_CONTRACT_PATH,
        PHASE5A_PARSER_CONTRACT_PATH,
        PHASE5A_PARAMETER_CONTRACT_PATH,
        COHORT_PATH,
        PROMPT_RENDERING_PATH,
        MODEL_INVENTORY_PATH,
        PHASE5A_EXECUTION_MATRIX_PATH,
        P1F_DECISION_PATH,
        COUNT_RECONCILIATION_PATH,
        TOP10_PROMPT_CONTRACT_PATH,
        TOP10_PARSER_CONTRACT_PATH,
        TOP10_SOURCE_AUDIT_PATH,
        Path("results/tables/canonical_local_llm_top10_confirmation_sample.csv"),
        Path("results/tables/canonical_local_llm_top10_confirmation_request_manifest.csv"),
        P1G_DECISION_PATH,
        P1G_REQUESTS_PATH,
        P1G_RESPONSES_PATH,
        P1G_REPORT_PATH,
        RANKING_CANDIDATES_PATH,
        POSITIVE_POSITIONS_PATH,
        COHORT_MEMBERSHIP_PATH,
        ITEM_TITLE_MAP_PATH,
        ITEM_TITLE_COVERAGE_PATH,
        TITLE_EXCLUSIONS_PATH,
        DATASET_STATISTICS_PATH,
        OLLAMA_RECOVERY_PATH,
        README_PATH,
        REQUIREMENTS_PATH,
        ENVIRONMENT_PATH,
        PAPER_PATH,
        *INTERACTION_PATHS,
    ]
    paths: list[Path] = [path for path in explicit if (root / path).exists()]
    original_root = root / "original_repo"
    if original_root.exists():
        paths.extend(path.relative_to(root) for path in original_root.rglob("*") if path.is_file())
    return sorted({path.as_posix() for path in paths})


def hash_protected_inputs(root: Path) -> dict[str, str]:
    return {relative: file_sha256(root / relative) for relative in protected_input_paths(root)}


def validate_phase5a_prompt(row: Mapping[str, str]) -> dict[str, str]:
    required = {
        "prompt_contract_id": RANKING_PROMPT_ID,
        "task_type": "ranking",
        "positive_label_hidden": "true",
        "requested_output_count": "5",
        "explanation_allowed": "false",
        "code_fence_allowed": "false",
    }
    for key, expected in required.items():
        if str(row.get(key, "")) != expected:
            raise ValueError(f"ranking prompt contract mismatch: {key}")
    if str(row.get("contract_sha256", "")) != contract_hash(row):
        raise ValueError("ranking prompt contract hash mismatch")
    return {str(k): str(v) for k, v in row.items()}


def validate_phase5a_parser(row: Mapping[str, str]) -> dict[str, str]:
    required = {
        "parser_contract_id": RANKING_PARSER_ID,
        "task_type": "ranking",
        "expected_output_count": "5",
        "requires_full_permutation": "false",
        "allows_leading_prose": "false",
        "allows_trailing_prose": "false",
        "allows_code_fence": "false",
        "allows_integer_strings": "false",
        "allows_duplicate_items": "false",
        "allows_unknown_items": "false",
        "allows_partial_output": "false",
        "allows_fuzzy_title_matching": "false",
    }
    for key, expected in required.items():
        if str(row.get(key, "")) != expected:
            raise ValueError(f"ranking parser contract mismatch: {key}")
    return {str(k): str(v) for k, v in row.items()}


def validate_top10_prompt(row: Mapping[str, str]) -> dict[str, str]:
    required = {
        "prompt_contract_id": RERANKING_PROMPT_ID,
        "task_type": "reranking",
        "positive_label_hidden": "true",
        "candidate_count": "20",
        "requested_output_count": "10",
        "requires_complete_permutation": "false",
        "requires_unique_indices": "true",
        "explanation_allowed": "false",
        "code_fence_allowed": "false",
        "model_specific_wording": "false",
    }
    for key, expected in required.items():
        if str(row.get(key, "")) != expected:
            raise ValueError(f"top10 prompt contract mismatch: {key}")
    if str(row.get("contract_sha256", "")) != top10_contract_hash(row, excluded=("contract_sha256",)):
        raise ValueError("top10 prompt contract hash mismatch")
    return {str(k): str(v) for k, v in row.items()}


def validate_top10_parser(row: Mapping[str, str]) -> dict[str, str]:
    required = {
        "parser_contract_id": RERANKING_PARSER_ID,
        "task_type": "reranking",
        "expected_output_count": "10",
        "requires_complete_permutation": "false",
        "allows_leading_prose": "false",
        "allows_trailing_prose": "false",
        "allows_code_fence": "false",
        "allows_integer_strings": "false",
        "allows_duplicate_items": "false",
        "allows_unknown_items": "false",
        "allows_partial_output": "false",
        "allows_fuzzy_title_matching": "false",
        "positive_label_used": "false",
    }
    for key, expected in required.items():
        if str(row.get(key, "")) != expected:
            raise ValueError(f"top10 parser contract mismatch: {key}")
    return {str(k): str(v) for k, v in row.items()}


def load_contract_bundle(root: Path) -> ContractBundle:
    prompts = read_csv(root, PHASE5A_PROMPT_CONTRACT_PATH.as_posix())
    if list(prompts.columns) != PROMPT_CONTRACT_COLUMNS:
        raise ValueError("Phase 5A prompt contract schema mismatch")
    parsers = read_csv(root, PHASE5A_PARSER_CONTRACT_PATH.as_posix())
    if list(parsers.columns) != PARSER_CONTRACT_COLUMNS:
        raise ValueError("Phase 5A parser contract schema mismatch")
    top10_prompt = read_csv(root, TOP10_PROMPT_CONTRACT_PATH.as_posix())
    if list(top10_prompt.columns) != TOP10_PROMPT_CONTRACT_COLUMNS:
        raise ValueError("top10 prompt contract schema mismatch")
    top10_parser = read_csv(root, TOP10_PARSER_CONTRACT_PATH.as_posix())
    if list(top10_parser.columns) != TOP10_PARSER_CONTRACT_COLUMNS:
        raise ValueError("top10 parser contract schema mismatch")
    execution = read_csv(root, EXECUTION_CONTRACT_PATH.as_posix())
    if list(execution.columns) != EXECUTION_PARAMETER_COLUMNS:
        raise ValueError("execution contract schema mismatch")
    exec_lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in execution.to_dict("records"):
        if row["model_family"] in MODEL_ORDER and row["task_type"] in TASK_ORDER:
            if row["contract_sha256"] != contract_hash(row):
                raise ValueError(f"execution contract hash mismatch: {row['execution_contract_id']}")
            if row["temperature"] != "0" or row["streaming"] != "false" or row["concurrency"] != "1":
                raise ValueError(f"execution parameter mismatch: {row['execution_contract_id']}")
            if row["scientific_retry_count"] != "0" or row["infrastructure_retry_count"] != "0":
                raise ValueError(f"execution retry mismatch: {row['execution_contract_id']}")
            exec_lookup[(row["model_family"], row["task_type"])] = {str(k): str(v) for k, v in row.items()}
    expected_execs = {(model, task) for model in MODEL_ORDER for task in TASK_ORDER}
    if set(exec_lookup) != expected_execs:
        raise ValueError("missing local execution contracts")
    ranking_prompt = validate_phase5a_prompt(prompts[prompts["prompt_contract_id"].eq(RANKING_PROMPT_ID)].iloc[0].to_dict())
    ranking_parser = validate_phase5a_parser(parsers[parsers["parser_contract_id"].eq(RANKING_PARSER_ID)].iloc[0].to_dict())
    return ContractBundle(
        ranking_prompt=ranking_prompt,
        reranking_prompt=validate_top10_prompt(top10_prompt.iloc[0].to_dict()),
        ranking_parser=ranking_parser,
        reranking_parser=validate_top10_parser(top10_parser.iloc[0].to_dict()),
        execution_contracts=exec_lookup,
    )


def render_from_contract(prompt_row: Mapping[str, str], history_items: Sequence[Mapping[str, str]], candidate_items: Sequence[Mapping[str, str]]) -> tuple[str, str, str, str]:
    history_render = render_history(history_items)
    candidate_render = render_candidates(candidate_items)
    system_message = str(prompt_row["system_message_template"])
    user_message = str(prompt_row["user_message_template"]).format(history_render=history_render, candidate_render=candidate_render)
    return system_message, user_message, history_render, candidate_render


def load_primary_cohort(root: Path) -> pd.DataFrame:
    cohort = read_csv(root, COHORT_PATH.as_posix())
    if list(cohort.columns) != COHORT_PREFLIGHT_COLUMNS:
        raise ValueError("cohort preflight schema mismatch")
    if len(cohort) != EXPECTED_FULL_CANDIDATE_ROWS:
        raise ValueError("full candidate row count mismatch")
    strict_title_count = int(cohort["eligible_strict_title_primary"].eq("true").sum())
    if strict_title_count != EXPECTED_STRICT_TITLE_ROWS:
        raise ValueError("candidate strict-title row count mismatch")
    primary = cohort[cohort["primary_prompt_compatible"].eq("true")].copy()
    if len(primary) != EXPECTED_PROMPT_COMPATIBLE_ROWS:
        raise ValueError("prompt-compatible row count mismatch")
    task_counts = primary.groupby("task_type").size().to_dict()
    if task_counts.get("ranking", 0) != EXPECTED_TASK_ROWS or task_counts.get("reranking", 0) != EXPECTED_TASK_ROWS:
        raise ValueError("task row count mismatch")
    exclusions = strict_title_count - len(primary)
    if exclusions != EXPECTED_HISTORY_TITLE_EXCLUSIONS:
        raise ValueError("history-title exclusion count mismatch")
    required_true = [
        "candidate_items_valid",
        "candidate_order_preserved",
        "candidate_titles_complete",
        "history_available",
        "history_train_only",
        "history_chronology_verified",
        "held_out_positive_absent_from_history",
        "history_titles_complete",
        "eligible_full_candidate_layer",
        "eligible_strict_title_primary",
    ]
    for column in required_true:
        if not primary[column].eq("true").all():
            raise ValueError(f"primary cohort contains false {column}")
    if not primary["candidate_count"].eq("20").all():
        raise ValueError("primary cohort candidate count mismatch")
    return primary


def build_contexts(root: Path, manifest: pd.DataFrame, cohort: pd.DataFrame, contracts: ContractBundle) -> dict[str, dict[str, Any]]:
    title_map = load_title_map(root)
    histories = load_histories(root, title_map)
    candidates = read_csv(root, RANKING_CANDIDATES_PATH.as_posix())
    candidate_lookup = {
        (str(row.candidate_set_id), str(row.user_id)): parse_json_list(row.candidate_items)
        for row in candidates[["candidate_set_id", "user_id", "candidate_items"]].itertuples(index=False)
    }
    cohort_lookup = {
        (str(row.dataset), str(row.task_type), str(row.candidate_set_id), str(row.user_id), str(row.source_path)): row._asdict()
        for row in cohort.itertuples(index=False)
    }
    prompt_rows = {"ranking": contracts.ranking_prompt, "reranking": contracts.reranking_prompt}
    contexts: dict[str, dict[str, Any]] = {}
    for row in manifest.sort_values("request_order", key=lambda s: s.astype(int)).to_dict("records"):
        cohort_key = (row["dataset"], row["task_type"], row["candidate_set_id"], row["user_id"], row["source_path"])
        if cohort_key not in cohort_lookup:
            raise ValueError(f"cohort row missing for manifest request: {row['request_id']}")
        cohort_row = cohort_lookup[cohort_key]
        candidate_key = (row["candidate_set_id"], row["user_id"])
        if candidate_key not in candidate_lookup:
            raise ValueError(f"candidate row missing: {row['request_id']}")
        candidate_ids = [str(item) for item in candidate_lookup[candidate_key]]
        if len(candidate_ids) != 20 or len(set(candidate_ids)) != 20:
            raise ValueError(f"candidate count/uniqueness mismatch: {row['request_id']}")
        if str(cohort_row["positive_item_id"]) not in candidate_ids:
            raise ValueError(f"positive missing from candidates: {row['request_id']}")
        candidate_items = [{"item_id": item_id, "title": title_map[(row["dataset"], item_id)]} for item_id in candidate_ids]
        history_items = [
            item
            for item in histories[row["dataset"]].get(row["user_id"], [])
            if item["item_id"] != str(cohort_row["positive_item_id"])
        ][-10:]
        system_message, user_message, history_render, candidate_render = render_from_contract(prompt_rows[row["task_type"]], history_items, candidate_items)
        candidate_hash = sha256_items(candidate_ids)
        history_hash = sha256_text(history_render)
        complete_hash = sha256_text(system_message + "\n\n" + user_message)
        if candidate_hash != row["candidate_order_sha256"]:
            raise ValueError(f"candidate-order hash mismatch: {row['request_id']}")
        if history_hash != row["history_render_sha256"]:
            raise ValueError(f"history-render hash mismatch: {row['request_id']}")
        contexts[row["request_id"]] = {
            "request_id": row["request_id"],
            "positive_item_id": str(cohort_row["positive_item_id"]),
            "candidate_items": candidate_ids,
            "candidate_titles": [item["title"] for item in candidate_items],
            "system_message": system_message,
            "user_message": user_message,
            "history_render": history_render,
            "candidate_render": candidate_render,
            "candidate_order_sha256": candidate_hash,
            "history_render_sha256": history_hash,
            "complete_prompt_sha256": complete_hash,
            "prompt_contract_sha256": row["prompt_contract_sha256"],
            "requested_output_count": row["requested_output_count"],
            "requires_full_permutation": False,
        }
    return contexts


def build_manifest(root: Path, cohort: pd.DataFrame, contracts: ContractBundle) -> pd.DataFrame:
    title_map = load_title_map(root)
    histories = load_histories(root, title_map)
    candidates = read_csv(root, RANKING_CANDIDATES_PATH.as_posix())
    candidate_lookup = {
        (str(row.candidate_set_id), str(row.user_id)): parse_json_list(row.candidate_items)
        for row in candidates[["candidate_set_id", "user_id", "candidate_items"]].itertuples(index=False)
    }
    prompt_lookup = {"ranking": contracts.ranking_prompt, "reranking": contracts.reranking_prompt}
    parser_lookup = {"ranking": contracts.ranking_parser, "reranking": contracts.reranking_parser}
    prompt_hash = {"ranking": contracts.ranking_prompt["contract_sha256"], "reranking": contracts.reranking_prompt["contract_sha256"]}
    parser_hash = {"ranking": contracts.ranking_parser["parser_schema_sha256"], "reranking": contracts.reranking_parser["parser_schema_sha256"]}
    requested_output = {"ranking": "5", "reranking": "10"}
    rows: list[dict[str, str]] = []
    request_order = 1
    for model in MODEL_ORDER:
        for task in TASK_ORDER:
            for dataset in DATASET_ORDER:
                subset = cohort[(cohort["dataset"].eq(dataset)) & (cohort["task_type"].eq(task))].copy()
                subset = subset.sort_values(["candidate_set_id", "user_id", "source_path"], kind="mergesort")
                for record in subset.itertuples(index=False):
                    candidate_ids = [str(item) for item in candidate_lookup[(str(record.candidate_set_id), str(record.user_id))]]
                    candidate_items = [{"item_id": item_id, "title": title_map[(dataset, item_id)]} for item_id in candidate_ids]
                    history_items = [
                        item
                        for item in histories[dataset].get(str(record.user_id), [])
                        if item["item_id"] != str(record.positive_item_id)
                    ][-10:]
                    system, user, history_render, _candidate_render = render_from_contract(prompt_lookup[task], history_items, candidate_items)
                    if not system or not user:
                        raise ValueError("prompt rendering failed")
                    execution = contracts.execution_contracts[(model, task)]
                    rows.append(
                        {
                            "request_id": f"CLFULL-REQ-{request_order:06d}",
                            "request_order": str(request_order),
                            "chunk_id": chunk_id_for_order(request_order),
                            "model_family": model,
                            "model_tag": EXPECTED_DIGESTS[model]["tag"],
                            "model_digest": EXPECTED_DIGESTS[model]["digest"],
                            "dataset": dataset,
                            "task_type": task,
                            "candidate_set_id": str(record.candidate_set_id),
                            "user_id": str(record.user_id),
                            "source_path": str(record.source_path),
                            "prompt_contract_id": prompt_lookup[task]["prompt_contract_id"],
                            "prompt_contract_sha256": prompt_hash[task],
                            "parser_contract_id": parser_lookup[task]["parser_contract_id"],
                            "parser_contract_sha256": parser_hash[task],
                            "execution_contract_id": execution["execution_contract_id"],
                            "execution_contract_sha256": execution["contract_sha256"],
                            "candidate_order_sha256": sha256_items(candidate_ids),
                            "history_render_sha256": sha256_text(history_render),
                            "candidate_count": "20",
                            "requested_output_count": requested_output[task],
                            "temperature": execution["temperature"],
                            "seed": execution["seed"],
                            "max_output_tokens": execution["max_output_tokens"],
                            "response_format": execution["response_format"],
                            "streaming": execution["streaming"],
                            "concurrency": execution["concurrency"],
                            "scientific_retry_count": execution["scientific_retry_count"],
                            "infrastructure_retry_count": execution["infrastructure_retry_count"],
                            "primary_prompt_compatible": "true",
                            "manifest_status": "planned_not_executed",
                            "blocking_reason": "",
                            "notes": "deterministic full-matrix manifest row; no inference in preflight",
                        }
                    )
                    request_order += 1
    manifest = frame(rows, MANIFEST_COLUMNS)
    validate_manifest(manifest)
    return manifest


def build_chunk_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for chunk_order, chunk_id in enumerate(sorted(manifest["chunk_id"].unique()), start=1):
        subset = manifest[manifest["chunk_id"].eq(chunk_id)]
        rows.append(
            {
                "chunk_id": chunk_id,
                "chunk_order": str(chunk_order),
                "start_request_order": str(int(subset["request_order"].astype(int).min())),
                "end_request_order": str(int(subset["request_order"].astype(int).max())),
                "planned_request_count": str(len(subset)),
                "chunk_status": "planned",
                "blocking_reason": "",
                "notes": "deterministic operational checkpoint; scientific order unchanged",
            }
        )
    chunks = frame(rows, CHUNK_MANIFEST_COLUMNS)
    validate_chunks(chunks)
    return chunks


def validate_manifest(manifest: pd.DataFrame) -> None:
    if list(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError("full manifest schema mismatch")
    if len(manifest) != EXPECTED_REQUESTS:
        raise ValueError("full manifest row count mismatch")
    if not manifest["request_id"].is_unique:
        raise ValueError("duplicate request_id")
    expected_ids = [f"CLFULL-REQ-{idx:06d}" for idx in range(1, EXPECTED_REQUESTS + 1)]
    if manifest["request_id"].tolist() != expected_ids:
        raise ValueError("request_id sequence mismatch")
    if manifest["request_order"].astype(int).tolist() != list(range(1, EXPECTED_REQUESTS + 1)):
        raise ValueError("request_order sequence mismatch")
    if not manifest["candidate_count"].eq("20").all():
        raise ValueError("candidate_count mismatch")
    if not manifest["primary_prompt_compatible"].eq("true").all():
        raise ValueError("manifest includes prompt-incompatible rows")
    if not manifest["temperature"].eq("0").all() or not manifest["streaming"].eq("false").all():
        raise ValueError("execution parameter mismatch")
    if not manifest["concurrency"].eq("1").all():
        raise ValueError("concurrency mismatch")
    if not manifest["scientific_retry_count"].eq("0").all() or not manifest["infrastructure_retry_count"].eq("0").all():
        raise ValueError("retry mismatch")
    for model in MODEL_ORDER:
        subset_model = manifest[manifest["model_family"].eq(model)]
        if len(subset_model) != EXPECTED_PROMPT_COMPATIBLE_ROWS:
            raise ValueError(f"model row count mismatch: {model}")
        if set(subset_model["model_tag"]) != {EXPECTED_DIGESTS[model]["tag"]}:
            raise ValueError(f"model tag mismatch: {model}")
        if set(subset_model["model_digest"]) != {EXPECTED_DIGESTS[model]["digest"]}:
            raise ValueError(f"model digest mismatch: {model}")
        for task in TASK_ORDER:
            subset_task = subset_model[subset_model["task_type"].eq(task)]
            if len(subset_task) != EXPECTED_TASK_ROWS:
                raise ValueError(f"model/task row count mismatch: {model} {task}")
    expected_counts = {
        ("Amazon Beauty", "ranking"): 990,
        ("Amazon Beauty", "reranking"): 990,
        ("Amazon Sports", "ranking"): 892,
        ("Amazon Sports", "reranking"): 892,
        ("ML-1M", "ranking"): 1000,
        ("ML-1M", "reranking"): 1000,
    }
    for model in MODEL_ORDER:
        for task in TASK_ORDER:
            for dataset in DATASET_ORDER:
                observed = len(manifest[(manifest["model_family"].eq(model)) & (manifest["task_type"].eq(task)) & (manifest["dataset"].eq(dataset))])
                if observed != expected_counts[(dataset, task)]:
                    raise ValueError(f"dataset cell count mismatch: {model} {task} {dataset}: {observed}")
    ranking = manifest[manifest["task_type"].eq("ranking")]
    reranking = manifest[manifest["task_type"].eq("reranking")]
    if set(ranking["prompt_contract_id"]) != {RANKING_PROMPT_ID} or set(ranking["parser_contract_id"]) != {RANKING_PARSER_ID}:
        raise ValueError("ranking contract assignment mismatch")
    if set(reranking["prompt_contract_id"]) != {RERANKING_PROMPT_ID} or set(reranking["parser_contract_id"]) != {RERANKING_PARSER_ID}:
        raise ValueError("reranking contract assignment mismatch")
    if not ranking["requested_output_count"].eq("5").all() or not reranking["requested_output_count"].eq("10").all():
        raise ValueError("requested output count mismatch")


def validate_chunks(chunks: pd.DataFrame) -> None:
    if list(chunks.columns) != CHUNK_MANIFEST_COLUMNS:
        raise ValueError("chunk manifest schema mismatch")
    if len(chunks) != EXPECTED_CHUNKS:
        raise ValueError("chunk count mismatch")
    expected_ids = [f"CLFULL-CHUNK-{idx:03d}" for idx in range(1, EXPECTED_CHUNKS + 1)]
    if chunks["chunk_id"].tolist() != expected_ids:
        raise ValueError("chunk_id sequence mismatch")
    sizes = chunks["planned_request_count"].astype(int).tolist()
    if sizes[:45] != [CHUNK_SIZE] * 45 or sizes[-1] != 8:
        raise ValueError("chunk size mismatch")


def validation_row(idx: int, scope: str, check: str, expected: Any, observed: Any, passed: bool, evidence: Path | str, notes: str = "", model: str = "ALL", dataset: str = "ALL", task: str = "ALL", severity: str = "blocking") -> dict[str, str]:
    return common_validation.make_validation_row(
        columns=PREFLIGHT_VALIDATION_COLUMNS,
        values={
            "validation_id": f"VAL-FULL-{idx:03d}",
            "validation_scope": scope,
            "model_family": model,
            "dataset": dataset,
            "task_type": task,
            "check_name": check,
            "expected_value": clean_cell(expected),
            "observed_value": clean_cell(observed),
            "status": common_validation.status_from_outcome(passed),
            "severity": severity,
            "evidence": str(evidence).replace("\\", "/"),
            "notes": notes,
        },
    )


def build_preflight_validation(root: Path, cohort: pd.DataFrame, manifest: pd.DataFrame, chunks: pd.DataFrame, protected_hashes: Mapping[str, str]) -> pd.DataFrame:
    all_cohort = read_csv(root, COHORT_PATH.as_posix())
    strict_title_count = int(all_cohort["eligible_strict_title_primary"].eq("true").sum())
    primary_counts = cohort.groupby("task_type").size().to_dict()
    checks = [
        ("cohort", "full_candidate_rows", EXPECTED_FULL_CANDIDATE_ROWS, len(all_cohort), len(all_cohort) == EXPECTED_FULL_CANDIDATE_ROWS, COHORT_PATH, ""),
        ("cohort", "candidate_strict_title_rows", EXPECTED_STRICT_TITLE_ROWS, strict_title_count, strict_title_count == EXPECTED_STRICT_TITLE_ROWS, COHORT_PATH, ""),
        ("cohort", "history_title_exclusions", EXPECTED_HISTORY_TITLE_EXCLUSIONS, strict_title_count - len(cohort), strict_title_count - len(cohort) == EXPECTED_HISTORY_TITLE_EXCLUSIONS, COHORT_PATH, ""),
        ("cohort", "prompt_compatible_rows", EXPECTED_PROMPT_COMPATIBLE_ROWS, len(cohort), len(cohort) == EXPECTED_PROMPT_COMPATIBLE_ROWS, COHORT_PATH, ""),
        ("cohort", "ranking_rows", EXPECTED_TASK_ROWS, primary_counts.get("ranking", 0), primary_counts.get("ranking", 0) == EXPECTED_TASK_ROWS, COHORT_PATH, ""),
        ("cohort", "reranking_rows", EXPECTED_TASK_ROWS, primary_counts.get("reranking", 0), primary_counts.get("reranking", 0) == EXPECTED_TASK_ROWS, COHORT_PATH, ""),
        ("manifest", "manifest_rows", EXPECTED_REQUESTS, len(manifest), len(manifest) == EXPECTED_REQUESTS, MANIFEST_PATH, ""),
        ("manifest", "qwen2_ranking_rows", EXPECTED_TASK_ROWS, len(manifest[(manifest["model_family"].eq("Qwen2 7B Instruct")) & (manifest["task_type"].eq("ranking"))]), True, MANIFEST_PATH, ""),
        ("manifest", "qwen2_reranking_rows", EXPECTED_TASK_ROWS, len(manifest[(manifest["model_family"].eq("Qwen2 7B Instruct")) & (manifest["task_type"].eq("reranking"))]), True, MANIFEST_PATH, ""),
        ("manifest", "mistral_ranking_rows", EXPECTED_TASK_ROWS, len(manifest[(manifest["model_family"].eq("Mistral 7B Instruct")) & (manifest["task_type"].eq("ranking"))]), True, MANIFEST_PATH, ""),
        ("manifest", "mistral_reranking_rows", EXPECTED_TASK_ROWS, len(manifest[(manifest["model_family"].eq("Mistral 7B Instruct")) & (manifest["task_type"].eq("reranking"))]), True, MANIFEST_PATH, ""),
        ("chunks", "chunk_rows", EXPECTED_CHUNKS, len(chunks), len(chunks) == EXPECTED_CHUNKS, CHUNK_MANIFEST_PATH, ""),
        ("chunks", "final_chunk_size", 8, chunks.iloc[-1]["planned_request_count"], chunks.iloc[-1]["planned_request_count"] == "8", CHUNK_MANIFEST_PATH, ""),
        ("contracts", "ranking_prompt", RANKING_PROMPT_ID, ",".join(sorted(manifest[manifest["task_type"].eq("ranking")]["prompt_contract_id"].unique())), set(manifest[manifest["task_type"].eq("ranking")]["prompt_contract_id"]) == {RANKING_PROMPT_ID}, MANIFEST_PATH, ""),
        ("contracts", "reranking_prompt", RERANKING_PROMPT_ID, ",".join(sorted(manifest[manifest["task_type"].eq("reranking")]["prompt_contract_id"].unique())), set(manifest[manifest["task_type"].eq("reranking")]["prompt_contract_id"]) == {RERANKING_PROMPT_ID}, MANIFEST_PATH, ""),
        ("protected", "protected_input_hashes_recorded", "nonzero", len(protected_hashes), len(protected_hashes) > 0, "hash_protected_inputs", ""),
    ]
    rows = [validation_row(idx, *check) for idx, check in enumerate(checks, start=1)]
    return frame(rows, PREFLIGHT_VALIDATION_COLUMNS)


def build_preflight_decision(root: Path, cohort: pd.DataFrame, manifest: pd.DataFrame, chunks: pd.DataFrame, validation: pd.DataFrame, protected_hashes: Mapping[str, str]) -> pd.DataFrame:
    failed = validation[~validation["status"].eq("pass")]
    decision = "approved_for_full_matrix_execution" if failed.empty else "blocked"
    row = {
        "preflight_id": PHASE_ID,
        "full_candidate_rows": str(EXPECTED_FULL_CANDIDATE_ROWS),
        "candidate_strict_title_rows": str(EXPECTED_STRICT_TITLE_ROWS),
        "history_title_exclusion_rows": str(EXPECTED_HISTORY_TITLE_EXCLUSIONS),
        "prompt_compatible_rows": str(len(cohort)),
        "prompt_compatible_ranking_rows": str(len(cohort[cohort["task_type"].eq("ranking")])),
        "prompt_compatible_reranking_rows": str(len(cohort[cohort["task_type"].eq("reranking")])),
        "full_manifest_rows": str(len(manifest)),
        "chunk_rows": str(len(chunks)),
        "expected_request_count": str(EXPECTED_REQUESTS),
        "manifest_sha256": file_sha256(root / MANIFEST_PATH) if (root / MANIFEST_PATH).exists() else "",
        "chunk_manifest_sha256": file_sha256(root / CHUNK_MANIFEST_PATH) if (root / CHUNK_MANIFEST_PATH).exists() else "",
        "protected_input_count": str(len(protected_hashes)),
        "ranking_prompt_contract_id": RANKING_PROMPT_ID,
        "reranking_prompt_contract_id": RERANKING_PROMPT_ID,
        "ranking_parser_contract_id": RANKING_PARSER_ID,
        "reranking_parser_contract_id": RERANKING_PARSER_ID,
        "preflight_decision": decision,
        "blocking_reason": ";".join(failed["check_name"].tolist()),
        "recommended_single_next_task": "Phase 5B-P2 - run the frozen full local matrix with --max-new-requests 256" if failed.empty else "Resolve full-matrix preflight validation failures without inference.",
        "notes": "offline deterministic preflight only; no Ollama request executed",
    }
    return frame([row], PREFLIGHT_DECISION_COLUMNS)


def build_preflight_frames(root: Path) -> dict[str, pd.DataFrame]:
    contracts = load_contract_bundle(root)
    cohort = load_primary_cohort(root)
    manifest = build_manifest(root, cohort, contracts)
    chunks = build_chunk_manifest(manifest)
    protected_hashes = hash_protected_inputs(root)
    validation = build_preflight_validation(root, cohort, manifest, chunks, protected_hashes)
    frames = {
        "manifest": manifest,
        "chunk_manifest": chunks,
        "preflight_validation": validation,
    }
    return frames


def write_preflight_frames(root: Path, frames: Mapping[str, pd.DataFrame]) -> None:
    outputs = {
        "manifest": (MANIFEST_PATH, MANIFEST_COLUMNS),
        "chunk_manifest": (CHUNK_MANIFEST_PATH, CHUNK_MANIFEST_COLUMNS),
        "preflight_validation": (PREFLIGHT_VALIDATION_PATH, PREFLIGHT_VALIDATION_COLUMNS),
    }
    for name, (path, columns) in outputs.items():
        out = root / path
        out.parent.mkdir(parents=True, exist_ok=True)
        data = frames[name].copy()
        if list(data.columns) != columns:
            raise ValueError(f"{name} schema mismatch")
        data.to_csv(out, index=False, lineterminator="\n")
    protected_hashes = hash_protected_inputs(root)
    cohort = load_primary_cohort(root)
    decision = build_preflight_decision(root, cohort, frames["manifest"], frames["chunk_manifest"], frames["preflight_validation"], protected_hashes)
    decision.to_csv(root / PREFLIGHT_DECISION_PATH, index=False, lineterminator="\n")


def validate_written_preflight(root: Path) -> dict[str, str]:
    expected = {
        MANIFEST_PATH: MANIFEST_COLUMNS,
        CHUNK_MANIFEST_PATH: CHUNK_MANIFEST_COLUMNS,
        PREFLIGHT_VALIDATION_PATH: PREFLIGHT_VALIDATION_COLUMNS,
        PREFLIGHT_DECISION_PATH: PREFLIGHT_DECISION_COLUMNS,
    }
    hashes: dict[str, str] = {}
    for path, columns in expected.items():
        data = pd.read_csv(root / path, dtype=str, keep_default_na=False)
        if list(data.columns) != columns:
            raise ValueError(f"{path} schema mismatch")
        if data.empty:
            raise ValueError(f"{path} is empty")
        hashes[path.as_posix()] = file_sha256(root / path)
    manifest = pd.read_csv(root / MANIFEST_PATH, dtype=str, keep_default_na=False)
    chunks = pd.read_csv(root / CHUNK_MANIFEST_PATH, dtype=str, keep_default_na=False)
    validate_manifest(manifest)
    validate_chunks(chunks)
    validation = pd.read_csv(root / PREFLIGHT_VALIDATION_PATH, dtype=str, keep_default_na=False)
    if not validation["status"].eq("pass").all():
        raise ValueError("preflight validation contains failures")
    decision = pd.read_csv(root / PREFLIGHT_DECISION_PATH, dtype=str, keep_default_na=False)
    if decision.iloc[0]["preflight_decision"] != "approved_for_full_matrix_execution":
        raise ValueError("full-matrix preflight not approved")
    return hashes

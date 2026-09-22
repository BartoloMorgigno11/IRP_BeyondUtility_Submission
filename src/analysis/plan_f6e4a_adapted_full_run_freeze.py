"""Phase F6E4A adapted full-run manifest and execution freeze.

This builder is strictly offline. It prepares the future F6E4B adapted
full-run workload from frozen local artifacts and never calls Ollama,
generation endpoints, model pulls, training, or inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import platform
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from src.analysis import plan_f6e3b_adapted_output_contract_pilot as f6e3b
from src.llm.plan_f6e1_prompt_strategies import BASE_STRATEGY_ID, LLMRANK_STRATEGY_ID, PromptRenderContext, render_strategy_prompt


PHASE_ID = "phase_f6e4a"
FROZEN_TIMESTAMP_UTC = "2026-07-26T00:00:00Z"
EXPECTED_HEAD = "02a20b8b509b8cf834a8766dd1cf1f751ae204bf"
EXPECTED_BRANCH = "plan-d-extensions"

QWEN2_TAG = "qwen2:7b-instruct"
MISTRAL_TAG = "mistral:7b-instruct"
MODEL_ORDER = [QWEN2_TAG, MISTRAL_TAG]
MODEL_FAMILY = {
    QWEN2_TAG: "Qwen2 7B Instruct",
    MISTRAL_TAG: "Mistral 7B Instruct",
}
EXPECTED_MODEL_DIGESTS = f6e3b.EXPECTED_TAG_DIGESTS
STRATEGY_ORDER = [BASE_STRATEGY_ID, LLMRANK_STRATEGY_ID]
STRATEGY_LABEL = {
    BASE_STRATEGY_ID: "base",
    LLMRANK_STRATEGY_ID: "llmrank",
}
DATASET_TASK_ORDER = [
    ("Amazon Beauty", "ranking"),
    ("Amazon Beauty", "reranking"),
    ("Amazon Sports", "ranking"),
    ("Amazon Sports", "reranking"),
    ("ML-1M", "ranking"),
    ("ML-1M", "reranking"),
    ("LastFM", "ranking"),
]
EXPECTED_SOURCE_CONTEXTS = 7623
EXPECTED_REQUESTS = 30492
SHARD_SIZE = 500

REPORT_PATH = Path("report/phase_f6e4a_adapted_full_run_freeze.md")
MANIFEST_PATH = Path("results/manifests/phase_f6e4a_full_run_freeze.json")
OUTPUT_TABLES = {
    "workload_eligibility": Path("results/tables/phase_f6e4a_workload_eligibility.csv"),
    "execution_matrix": Path("results/tables/phase_f6e4a_execution_matrix.csv"),
    "request_manifest": Path("results/tables/phase_f6e4a_request_manifest.csv"),
    "shard_manifest": Path("results/tables/phase_f6e4a_shard_manifest.csv"),
    "runtime_parameters": Path("results/tables/phase_f6e4a_runtime_parameters.csv"),
    "time_storage_budget": Path("results/tables/phase_f6e4a_time_storage_budget.csv"),
    "stop_conditions": Path("results/tables/phase_f6e4a_stop_conditions.csv"),
    "gate_plan": Path("results/tables/phase_f6e4a_gate_plan.csv"),
    "missing_or_blocked": Path("results/tables/phase_f6e4a_missing_or_blocked.csv"),
}

F6E1_FREEZE = Path("results/manifests/phase_f6e1_prompt_freeze.json")
F6E2_FREEZE = Path("results/manifests/phase_f6e2_preflight_freeze.json")
F6E3_FREEZE = Path("results/manifests/phase_f6e3_pilot_freeze.json")
F6E3A_FREEZE = Path("results/manifests/phase_f6e3a_parser_recovery_freeze.json")
F6E3B_FREEZE = Path("results/manifests/phase_f6e3b_pilot_freeze.json")
F6E1_PROMPT_MODULE = Path("src/llm/plan_f6e1_prompt_strategies.py")
F6E3B_MODULE = Path("src/analysis/plan_f6e3b_adapted_output_contract_pilot.py")
CANONICAL_FULL_MANIFEST = Path("results/tables/canonical_local_llm_full_matrix_manifest.csv")
CANONICAL_FULL_REQUESTS = Path("results/llm/canonical_local_full_matrix/requests.jsonl")
CANONICAL_COUNT_RECONCILIATION = Path("results/tables/canonical_local_llm_prompt_compatible_count_reconciliation.csv")
CANONICAL_PREFLIGHT_DECISION = Path("results/tables/canonical_local_llm_full_matrix_preflight_decision.csv")
F6D3_LASTFM_RANKING_MANIFEST = Path("results/tables/finalization_f6d3_lastfm_ranking_llm_manifest.csv")
F6D3_LASTFM_RERANKING_MANIFEST = Path("results/tables/finalization_f6d3_lastfm_reranking_llm_manifest.csv")
F6C_LASTFM_RERANKING_READINESS = Path("results/tables/finalization_f6c_lastfm_reranking_readiness.csv")
F6D3_GO_NO_GO = Path("results/tables/finalization_f6d3_llm_go_no_go.csv")
TITLE_MAP = Path("results/tables/candidate_item_title_map.csv")
F2_PROTECTED_VALIDATION = Path("results/tables/finalization_f2_protected_input_validation.csv")
F6D4A_PROTECTED_VERIFICATION = Path("results/tables/finalization_f6d4a_closure_protected_input_verification.csv")
F6D4A_INPUT_HASHES = Path("results/tables/finalization_f6d4a_input_hashes.csv")

F6D4A_RAW_HASHES = {
    "results/llm/finalization_f6d4a_lastfm_ranking_qwen2_raw.jsonl": "1384a47058ed8dd8c06fc52d6bc425020b1746b60ad11e4bde6153f948a493b4",
    "results/llm/finalization_f6d4a_lastfm_ranking_mistral_raw.jsonl": "7031f37d40092b4f318b5e03273772531ffda37cef0c38bfed32edfef935736a",
}
F6E3_RAW_HASHES = f6e3b.EXPECTED_F6E3_RAW_HASHES
F6E3B_RAW_HASHES = {
    "results/llm/phase_f6e3b_qwen2_adapted_indexed_pilot_raw.jsonl": "6d70eca3edaeee1174e066404ccc8b51acc5f1886aa75ce7b185fcc2d58a96bc",
    "results/llm/phase_f6e3b_mistral_adapted_indexed_pilot_raw.jsonl": "aad02404ba238f1edce0b7558ae9cbedd148a021799f6b7b67bdb2a1bcec77ad",
}
EXPECTED_F6E3B_SCIENTIFIC_HASH = "079a085837d1dade519d1e28e616a9ee2484450dd1c305c4e3261179b39889b1"
SELF_GENERATED_STATUS_PREFIXES = (
    "?? report/phase_f6e4a_",
    "?? results/manifests/phase_f6e4a_",
    "?? results/tables/phase_f6e4a_",
)

FUTURE_RAW_PATHS = {
    QWEN2_TAG: Path("results/llm/phase_f6e4b_qwen2_adapted_full_run_raw.jsonl"),
    MISTRAL_TAG: Path("results/llm/phase_f6e4b_mistral_adapted_full_run_raw.jsonl"),
}

PLAN_E_SOURCE_MANIFESTS = {
    ("Amazon Beauty", "ranking"): Path("results/tables/plan_e_beauty_1000user_ranking_manifest.csv"),
    ("Amazon Beauty", "reranking"): Path("results/tables/plan_e_beauty_1000user_reranking_manifest.csv"),
    ("Amazon Sports", "ranking"): Path("results/tables/plan_e_sports_1000user_ranking_manifest.csv"),
    ("Amazon Sports", "reranking"): Path("results/tables/plan_e_sports_1000user_reranking_manifest.csv"),
    ("ML-1M", "ranking"): Path("results/tables/plan_e_ml1m_1000user_ranking_manifest.csv"),
    ("ML-1M", "reranking"): Path("results/tables/plan_e_ml1m_1000user_reranking_manifest.csv"),
}

WORKLOAD_COLUMNS = [
    "workload_id",
    "dataset",
    "task",
    "source_kind",
    "source_manifest",
    "raw_candidate_or_manifest_rows",
    "full_logical_instances",
    "candidate_strict_title_rows",
    "history_title_complete_rows",
    "ready_source_contexts",
    "excluded_contexts",
    "candidate_count",
    "positive_policy",
    "title_availability",
    "candidate_order_hash_policy",
    "comparability",
    "status",
    "blocker",
    "notes",
]
EXECUTION_MATRIX_COLUMNS = [
    "matrix_id",
    "dataset",
    "task",
    "model",
    "model_family",
    "model_equivalence",
    "strategy_id",
    "adapted_strategy_id",
    "adapter_classification",
    "logical_instances",
    "expected_calls",
    "candidate_count",
    "prompt_token_estimate_mean",
    "expected_output_tokens",
    "estimated_duration_seconds",
    "estimated_raw_storage_bytes",
    "status",
    "blocker",
]
REQUEST_COLUMNS = [
    "request_id",
    "source_context_id",
    "source_request_id",
    "source_kind",
    "dataset",
    "task",
    "user_hash",
    "history_hash",
    "model",
    "model_family",
    "expected_model_digest",
    "model_equivalence",
    "core_strategy_id",
    "adapted_strategy_id",
    "adapter_classification",
    "scientific_role",
    "candidate_count",
    "candidate_order_hash",
    "candidate_item_ids",
    "positive_item_id",
    "positive_initial_position",
    "history_length",
    "history_length_stratum",
    "positive_position_stratum",
    "core_prompt_hash",
    "candidate_serialization_hash",
    "adapter_hash",
    "complete_prompt_hash",
    "prompt_character_count",
    "expected_schema",
    "runtime_parameters_hash",
    "runtime_parameters_json",
    "shard_id",
    "ordinal_position",
    "status",
    "notes",
]
SHARD_COLUMNS = [
    "shard_id",
    "model",
    "strategy_id",
    "adapted_strategy_id",
    "dataset",
    "task",
    "first_request_id",
    "last_request_id",
    "request_count",
    "estimated_runtime_seconds",
    "estimated_storage_bytes",
    "raw_output_path",
    "progress_path",
    "log_path",
    "checkpoint_policy",
    "resume_key",
    "status",
]
RUNTIME_COLUMNS = [
    "runtime_config_id",
    "strategy_id",
    "parameter_name",
    "value",
    "source",
    "classification",
    "config_hash",
    "notes",
]
BUDGET_COLUMNS = [
    "budget_id",
    "scope",
    "model",
    "strategy_id",
    "dataset",
    "task",
    "request_count",
    "estimate_kind",
    "seconds_per_request_mean",
    "seconds_per_request_p50",
    "seconds_per_request_p95",
    "best_duration_seconds",
    "expected_duration_seconds",
    "worst_duration_seconds",
    "raw_storage_bytes",
    "log_storage_bytes",
    "table_storage_bytes",
    "minimum_free_space_bytes",
    "provenance",
]
STOP_COLUMNS = [
    "condition_id",
    "condition_name",
    "trigger",
    "severity",
    "automatic_action",
    "resume_allowed",
    "retry_allowed",
    "evidence_to_record",
    "notes",
]
GATE_COLUMNS = [
    "gate_id",
    "gate_name",
    "phase_point",
    "expected",
    "status",
    "evidence",
    "blocking_decision_if_fail",
    "notes",
]
MISSING_COLUMNS = [
    "block_id",
    "category",
    "affected_scope",
    "status",
    "severity",
    "evidence",
    "safe_next_action",
    "notes",
]

TABLE_COLUMNS = {
    "workload_eligibility": WORKLOAD_COLUMNS,
    "execution_matrix": EXECUTION_MATRIX_COLUMNS,
    "request_manifest": REQUEST_COLUMNS,
    "shard_manifest": SHARD_COLUMNS,
    "runtime_parameters": RUNTIME_COLUMNS,
    "time_storage_budget": BUDGET_COLUMNS,
    "stop_conditions": STOP_COLUMNS,
    "gate_plan": GATE_COLUMNS,
    "missing_or_blocked": MISSING_COLUMNS,
}


@dataclass(frozen=True)
class SourceContext:
    source_context_id: str
    source_request_id: str
    source_kind: str
    dataset: str
    task: str
    user_id: str
    history_titles: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    candidate_titles: tuple[str, ...]
    positive_item_id: str
    positive_initial_position: str
    candidate_order_hash: str
    source_manifest: str
    source_order: int


@dataclass(frozen=True)
class BuildOutputs:
    tables: dict[str, list[dict[str, str]]]
    manifest: dict[str, Any]
    report: str


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


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    return f"{prefix}-{sha256_text('||'.join(str(part) for part in parts))[:length]}"


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def read_csv_rows(root: Path, relative_path: Path) -> list[dict[str, str]]:
    path = root / relative_path
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def csv_text(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def split_pipe(value: Any) -> list[str]:
    return [part for part in str(value or "").split("|") if part != ""]


def join_pipe(values: Sequence[Any]) -> str:
    return "|".join(str(value) for value in values)


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def command_output(args: Sequence[str], root: Path, timeout: int = 10) -> str:
    try:
        result = subprocess.run(list(args), cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:  # pragma: no cover - platform dependent
        return f"{type(exc).__name__}:{exc}"
    return (result.stdout + result.stderr).strip()


def filtered_self_generated_status(status_text: str) -> str:
    lines = [
        line
        for line in status_text.splitlines()
        if not any(line.startswith(prefix) for prefix in SELF_GENERATED_STATUS_PREFIXES)
    ]
    return "\n".join(lines)


def repository_state(root: Path) -> dict[str, str]:
    git_status_short = command_output(["git", "status", "--short"], root, timeout=20)
    return {
        "directory": str(root),
        "branch": command_output(["git", "branch", "--show-current"], root),
        "head": command_output(["git", "rev-parse", "HEAD"], root),
        "last_commit": command_output(["git", "log", "-1", "--oneline"], root),
        "git_status_short": filtered_self_generated_status(git_status_short),
        "git_status_self_generated_f6e4a_filtered": "yes",
        "git_diff_name_only": command_output(["git", "diff", "--name-only"], root, timeout=20),
        "git_diff_stat": command_output(["git", "diff", "--stat"], root, timeout=20),
        "git_diff_cached_stat": command_output(["git", "diff", "--cached", "--stat"], root, timeout=20),
        "staged_files": command_output(["git", "diff", "--name-only", "--cached"], root, timeout=20),
        "python_version": platform.python_version(),
        "python_executable": os.sys.executable,
        "os": platform.platform(),
        "timestamp_utc": FROZEN_TIMESTAMP_UTC,
    }


def verify_expected_hashes(root: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    expected.update(F6D4A_RAW_HASHES)
    expected.update(F6E3_RAW_HASHES)
    expected.update(F6E3B_RAW_HASHES)
    observed = {path: file_sha256_or_missing(root, Path(path)) for path in sorted(expected)}
    mismatches = [f"{path}:expected={expected[path]}:observed={observed[path]}" for path in sorted(expected) if observed[path] != expected[path]]

    f6e3b_freeze = root / F6E3B_FREEZE
    if not f6e3b_freeze.exists():
        mismatches.append(f"{F6E3B_FREEZE.as_posix()}:missing")
    else:
        content = json.loads(f6e3b_freeze.read_text(encoding="utf-8"))
        if content.get("scientific_content_hash") != EXPECTED_F6E3B_SCIENTIFIC_HASH:
            mismatches.append(
                f"{F6E3B_FREEZE.as_posix()}:expected_scientific={EXPECTED_F6E3B_SCIENTIFIC_HASH}:observed={content.get('scientific_content_hash')}"
            )
        if content.get("decision") != "f6e3b_adapted_output_contract_pilot_complete_gate_passed":
            mismatches.append(f"{F6E3B_FREEZE.as_posix()}:unexpected_decision={content.get('decision')}")
    if mismatches:
        raise RuntimeError("f6e4a_frozen_input_hash_mismatch:" + ";".join(mismatches))
    return observed


def extract_numbered_block(prompt: str, start_label: str, end_labels: Sequence[str] = ()) -> list[str]:
    if start_label not in prompt:
        return []
    text = prompt.split(start_label, 1)[1]
    end_positions = [text.find(label) for label in end_labels if label in text]
    if end_positions:
        text = text[: min(position for position in end_positions if position >= 0)]
    values: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", line)
        if match:
            values.append(html.unescape(match.group(2).strip()))
    return values


def prompt_titles(record: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    prompt = str(record.get("request_payload", {}).get("prompt", ""))
    history = tuple(extract_numbered_block(prompt, "User history:", ("Candidates:", "Initial candidate order:")))
    candidates = tuple(extract_numbered_block(prompt, "Candidates:"))
    if not candidates:
        candidates = tuple(extract_numbered_block(prompt, "Initial candidate order:"))
    return history, candidates


def source_manifest_groups(root: Path, relative_path: Path) -> dict[str, list[dict[str, str]]]:
    rows = read_csv_rows(root, relative_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["user_id"]].append(row)
    for group in grouped.values():
        group.sort(key=lambda value: int(float(value["candidate_position"])))
    return grouped


def load_request_payloads(root: Path, request_ids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    path = root / CANONICAL_FULL_REQUESTS
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(found) == len(request_ids):
                break
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = str(row.get("request_id", ""))
            if request_id in request_ids:
                found[request_id] = row
    return found


def build_canonical_source_contexts(root: Path) -> list[SourceContext]:
    rows = [
        row
        for row in read_csv_rows(root, CANONICAL_FULL_MANIFEST)
        if row.get("model_tag") == QWEN2_TAG
        and row.get("primary_prompt_compatible") == "true"
        and row.get("task_type") in {"ranking", "reranking"}
    ]
    rows.sort(key=lambda row: (DATASET_TASK_ORDER.index((row["dataset"], row["task_type"])), int(float(row["request_order"]))))
    payloads = load_request_payloads(root, {row["request_id"] for row in rows})
    grouped_cache = {path: source_manifest_groups(root, path) for path in sorted(set(Path(row["source_path"]) for row in rows), key=lambda value: value.as_posix())}
    contexts: list[SourceContext] = []
    for row in rows:
        payload = payloads.get(row["request_id"])
        if payload is None:
            raise RuntimeError(f"f6e4a_missing_canonical_request_payload:{row['request_id']}")
        history_titles, candidate_titles = prompt_titles(payload)
        source_path = Path(row["source_path"])
        user_rows = grouped_cache[source_path].get(row["user_id"], [])
        candidate_ids = tuple(candidate["item_id"] for candidate in user_rows)
        positives = [candidate for candidate in user_rows if parse_bool(candidate.get("is_positive"))]
        if len(candidate_ids) != 20 or len(candidate_titles) != 20 or len(positives) != 1 or not history_titles:
            raise RuntimeError(f"f6e4a_canonical_context_invalid:{row['request_id']}")
        candidate_hash = sha256_text(stable_json(candidate_ids))
        if candidate_hash != row["candidate_order_sha256"]:
            raise RuntimeError(f"f6e4a_candidate_order_hash_mismatch:{row['request_id']}")
        contexts.append(
            SourceContext(
                source_context_id=stable_id("F6E4A-SRC", row["request_id"]),
                source_request_id=row["request_id"],
                source_kind="canonical_full_matrix_qwen2_context",
                dataset=row["dataset"],
                task=row["task_type"],
                user_id=row["user_id"],
                history_titles=history_titles,
                candidate_ids=candidate_ids,
                candidate_titles=candidate_titles,
                positive_item_id=positives[0]["item_id"],
                positive_initial_position=positives[0]["candidate_position"],
                candidate_order_hash=candidate_hash,
                source_manifest=CANONICAL_FULL_MANIFEST.as_posix(),
                source_order=int(float(row["request_order"])),
            )
        )
    return contexts


def build_lastfm_source_contexts(root: Path) -> list[SourceContext]:
    rows = [
        row
        for row in read_csv_rows(root, F6D3_LASTFM_RANKING_MANIFEST)
        if row.get("model") == QWEN2_TAG and row.get("task") == "ranking"
    ]
    contexts: list[SourceContext] = []
    for order, row in enumerate(rows, start=1):
        candidate_ids = tuple(split_pipe(row.get("candidate_item_ids", "")))
        candidate_titles = tuple(split_pipe(row.get("candidate_item_names", "")))
        history_titles = tuple(split_pipe(row.get("history_item_names", "")))
        if len(candidate_ids) != 20 or len(candidate_titles) != 20 or not history_titles or row.get("positive_present") != "true":
            raise RuntimeError(f"f6e4a_lastfm_context_invalid:{row['request_id']}")
        contexts.append(
            SourceContext(
                source_context_id=stable_id("F6E4A-SRC", row["request_id"]),
                source_request_id=row["request_id"],
                source_kind="f6d3_lastfm_qwen2_ranking_context",
                dataset="LastFM",
                task="ranking",
                user_id=row["user_id"],
                history_titles=history_titles,
                candidate_ids=candidate_ids,
                candidate_titles=candidate_titles,
                positive_item_id=row["positive_item_id"],
                positive_initial_position=row["positive_candidate_position"],
                candidate_order_hash=sha256_text(stable_json(candidate_ids)),
                source_manifest=F6D3_LASTFM_RANKING_MANIFEST.as_posix(),
                source_order=order,
            )
        )
    return contexts


def build_source_contexts(root: Path) -> list[SourceContext]:
    contexts = build_canonical_source_contexts(root) + build_lastfm_source_contexts(root)
    contexts.sort(key=lambda context: (DATASET_TASK_ORDER.index((context.dataset, context.task)), context.source_order))
    if len(contexts) != EXPECTED_SOURCE_CONTEXTS:
        raise RuntimeError(f"f6e4a_source_context_count_invalid:{len(contexts)}")
    if len({context.source_context_id for context in contexts}) != len(contexts):
        raise RuntimeError("f6e4a_duplicate_source_context_id")
    return contexts


def history_stratum(length: int) -> str:
    if length <= 5:
        return "history_le_5"
    if length <= 10:
        return "history_6_10"
    return "history_gt_10"


def positive_stratum(position: str) -> str:
    if not position:
        return "positive_absent"
    value = int(float(position))
    if value <= 5:
        return "positive_top5"
    if value <= 10:
        return "positive_6_10"
    return "positive_11_20"


def runtime_parameters(strategy_id: str) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "num_predict": 256,
        "stream": False,
        "timeout_seconds": 120 if strategy_id == BASE_STRATEGY_ID else 150,
        "top_p": "",
        "top_k": "",
        "seed": "",
        "context_window": "",
        "stop_sequences": "",
        "keep_alive": "",
    }


def render_complete_prompt(root: Path, strategy_id: str, context: SourceContext) -> tuple[str, str, str]:
    render_context = PromptRenderContext(
        dataset=context.dataset,
        user_name=context.user_id,
        user_history_titles=context.history_titles,
        candidate_titles=context.candidate_titles,
    )
    core = render_strategy_prompt(strategy_id, render_context, root=root)
    adapter = f6e3b.output_adapter_text(context.candidate_titles)
    return core, adapter, core.rstrip() + adapter


def build_request_manifest(root: Path, contexts: Sequence[SourceContext], *, validate_expected: bool = True) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    ordinal = 0
    for model in MODEL_ORDER:
        for strategy_id in STRATEGY_ORDER:
            for dataset, task in DATASET_TASK_ORDER:
                for context in [ctx for ctx in contexts if ctx.dataset == dataset and ctx.task == task]:
                    ordinal += 1
                    core, adapter, prompt = render_complete_prompt(root, strategy_id, context)
                    runtime = runtime_parameters(strategy_id)
                    runtime_json = stable_json(runtime)
                    prompt_hash = sha256_text(prompt)
                    adapted_strategy_id = f"{STRATEGY_LABEL[strategy_id]}_adapted_indexed_output_contract_v1"
                    request_id = stable_id(
                        "F6E4A-REQ",
                        PHASE_ID,
                        context.source_context_id,
                        context.source_request_id,
                        context.dataset,
                        context.task,
                        model,
                        strategy_id,
                        context.candidate_order_hash,
                        prompt_hash,
                    )
                    rows.append(
                        {
                            "request_id": request_id,
                            "source_context_id": context.source_context_id,
                            "source_request_id": context.source_request_id,
                            "source_kind": context.source_kind,
                            "dataset": context.dataset,
                            "task": context.task,
                            "user_hash": sha256_text(f"{context.dataset}|{context.task}|{context.user_id}")[:16],
                            "history_hash": sha256_text(stable_json(context.history_titles)),
                            "model": model,
                            "model_family": MODEL_FAMILY[model],
                            "expected_model_digest": EXPECTED_MODEL_DIGESTS[model],
                            "model_equivalence": "exact_family_version_uncertain",
                            "core_strategy_id": strategy_id,
                            "adapted_strategy_id": adapted_strategy_id,
                            "adapter_classification": "adapted_indexed_output_contract",
                            "scientific_role": "adapted_full_run_freeze_pending_execution",
                            "candidate_count": str(len(context.candidate_ids)),
                            "candidate_order_hash": context.candidate_order_hash,
                            "candidate_item_ids": join_pipe(context.candidate_ids),
                            "positive_item_id": context.positive_item_id,
                            "positive_initial_position": context.positive_initial_position,
                            "history_length": str(len(context.history_titles)),
                            "history_length_stratum": history_stratum(len(context.history_titles)),
                            "positive_position_stratum": positive_stratum(context.positive_initial_position),
                            "core_prompt_hash": sha256_text(core),
                            "candidate_serialization_hash": sha256_text(stable_json(context.candidate_titles)),
                            "adapter_hash": sha256_text(adapter),
                            "complete_prompt_hash": prompt_hash,
                            "prompt_character_count": str(len(prompt)),
                            "expected_schema": '{"ranking":[1,2,3,4,5,6,7,8,9,10]}',
                            "runtime_parameters_hash": sha256_text(runtime_json),
                            "runtime_parameters_json": runtime_json,
                            "shard_id": "",
                            "ordinal_position": str(ordinal),
                            "status": "pending",
                            "notes": "Prompt text rendered from frozen source context at execution time; F6E4A stores hashes, not full prompts.",
                        }
                    )
    if validate_expected and len(rows) != EXPECTED_REQUESTS:
        raise RuntimeError(f"f6e4a_request_count_invalid:{len(rows)}")
    if len({row["request_id"] for row in rows}) != len(rows):
        raise RuntimeError("f6e4a_duplicate_request_id")
    return assign_shards(rows)


def assign_shards(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    shard_number = 0
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["core_strategy_id"], row["dataset"], row["task"])].append(row)
    for model in MODEL_ORDER:
        for strategy in STRATEGY_ORDER:
            for dataset, task in DATASET_TASK_ORDER:
                group = grouped.get((model, strategy, dataset, task), [])
                group.sort(key=lambda row: int(row["ordinal_position"]))
                for start in range(0, len(group), SHARD_SIZE):
                    shard_number += 1
                    shard_id = f"F6E4A-SHARD-{shard_number:04d}"
                    for row in group[start : start + SHARD_SIZE]:
                        row["shard_id"] = shard_id
    return rows


def build_workload_eligibility(root: Path, contexts: Sequence[SourceContext]) -> list[dict[str, str]]:
    reconciliation = {
        (row["dataset"], row["task_type"]): row
        for row in read_csv_rows(root, CANONICAL_COUNT_RECONCILIATION)
        if row.get("aggregation_level") == "dataset_task"
    }
    context_counts = Counter((context.dataset, context.task) for context in contexts)
    rows: list[dict[str, str]] = []
    for dataset, task in DATASET_TASK_ORDER:
        if dataset == "LastFM":
            ready = context_counts[(dataset, task)]
            rows.append(
                {
                    "workload_id": stable_id("F6E4A-WORK", dataset, task),
                    "dataset": dataset,
                    "task": task,
                    "source_kind": "f6d3_lastfm_qwen2_ranking_context",
                    "source_manifest": F6D3_LASTFM_RANKING_MANIFEST.as_posix(),
                    "raw_candidate_or_manifest_rows": str(len(read_csv_rows(root, F6D3_LASTFM_RANKING_MANIFEST))),
                    "full_logical_instances": "1859",
                    "candidate_strict_title_rows": "1859",
                    "history_title_complete_rows": "1859",
                    "ready_source_contexts": str(ready),
                    "excluded_contexts": "0",
                    "candidate_count": "20",
                    "positive_policy": "positive_present_per_request",
                    "title_availability": "candidate_and_history_titles_present",
                    "candidate_order_hash_policy": "sha256_stable_json_candidate_item_ids",
                    "comparability": "adapted_lastfm_ranking_not_paper_exact",
                    "status": "ready",
                    "blocker": "",
                    "notes": "Qwen2 rows used as source context; model dimension rematerialized in F6E4A.",
                }
            )
            continue
        recon = reconciliation[(dataset, task)]
        ready = context_counts[(dataset, task)]
        status = "ready" if ready == int(recon["authoritative_count"]) else "blocked"
        rows.append(
            {
                "workload_id": stable_id("F6E4A-WORK", dataset, task),
                "dataset": dataset,
                "task": task,
                "source_kind": "canonical_full_matrix_qwen2_context",
                "source_manifest": CANONICAL_FULL_MANIFEST.as_posix(),
                "raw_candidate_or_manifest_rows": recon["full_candidate_rows"],
                "full_logical_instances": recon["full_candidate_rows"],
                "candidate_strict_title_rows": recon["candidate_strict_title_rows"],
                "history_title_complete_rows": recon["history_title_complete_rows"],
                "ready_source_contexts": str(ready),
                "excluded_contexts": str(int(recon["full_candidate_rows"]) - ready),
                "candidate_count": "20",
                "positive_policy": "one_positive_per_prompt_compatible_context",
                "title_availability": "candidate_and_history_titles_complete",
                "candidate_order_hash_policy": "sha256_stable_json_candidate_item_ids_matches_canonical_manifest",
                "comparability": "adapted_indexed_local_full_run_not_paper_exact",
                "status": status,
                "blocker": "" if status == "ready" else "prompt_compatible_count_mismatch",
                "notes": "Authoritative count is prompt-compatible rows after candidate and history title exclusions.",
            }
        )
    rows.append(
        {
            "workload_id": stable_id("F6E4A-WORK", "LastFM", "reranking"),
            "dataset": "LastFM",
            "task": "reranking",
            "source_kind": "f6d3_lastfm_reranking_manifest",
            "source_manifest": F6D3_LASTFM_RERANKING_MANIFEST.as_posix(),
            "raw_candidate_or_manifest_rows": str(len(read_csv_rows(root, F6D3_LASTFM_RERANKING_MANIFEST))),
            "full_logical_instances": "1859",
            "candidate_strict_title_rows": "1859",
            "history_title_complete_rows": "1859",
            "ready_source_contexts": "0",
            "excluded_contexts": "1859",
            "candidate_count": "20",
            "positive_policy": "positive_absent_from_adapted_pool",
            "title_availability": "present_but_not_comparable",
            "candidate_order_hash_policy": "not_used",
            "comparability": "partial_non_comparable",
            "status": "blocked",
            "blocker": "positive_policy_partial_non_comparable",
            "notes": "Explicitly excluded by F6E4A; do not include diagnostic LastFM reranking in full run.",
        }
    )
    return rows


def estimate_record_bytes(root: Path, model: str) -> float:
    raw_path = root / {
        QWEN2_TAG: Path("results/llm/phase_f6e3b_qwen2_adapted_indexed_pilot_raw.jsonl"),
        MISTRAL_TAG: Path("results/llm/phase_f6e3b_mistral_adapted_indexed_pilot_raw.jsonl"),
    }[model]
    count = len(read_jsonl(raw_path))
    return raw_path.stat().st_size / count if count else 2048.0


def runtime_lookup(root: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (row["model"], row["strategy_id"], row["dataset"]): row
        for row in read_csv_rows(root, Path("results/tables/phase_f6e3b_runtime_summary.csv"))
    }


def build_execution_matrix(root: Path, request_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in request_rows:
        grouped[(row["dataset"], row["task"], row["model"], row["core_strategy_id"])].append(row)
    runtime = runtime_lookup(root)
    rows: list[dict[str, str]] = []
    for dataset, task in DATASET_TASK_ORDER:
        for model in MODEL_ORDER:
            for strategy in STRATEGY_ORDER:
                group = grouped[(dataset, task, model, strategy)]
                rt = runtime[(model, strategy, dataset)]
                seconds = float(rt["latency_mean"]) * len(group)
                bytes_est = estimate_record_bytes(root, model) * len(group)
                rows.append(
                    {
                        "matrix_id": stable_id("F6E4A-MATRIX", dataset, task, model, strategy),
                        "dataset": dataset,
                        "task": task,
                        "model": model,
                        "model_family": MODEL_FAMILY[model],
                        "model_equivalence": "exact_family_version_uncertain",
                        "strategy_id": strategy,
                        "adapted_strategy_id": f"{STRATEGY_LABEL[strategy]}_adapted_indexed_output_contract_v1",
                        "adapter_classification": "adapted_indexed_output_contract",
                        "logical_instances": str(len(group)),
                        "expected_calls": str(len(group)),
                        "candidate_count": "20",
                        "prompt_token_estimate_mean": rt["prompt_eval_count_mean"],
                        "expected_output_tokens": rt["eval_count_mean"],
                        "estimated_duration_seconds": fmt(seconds),
                        "estimated_raw_storage_bytes": str(int(bytes_est)),
                        "status": "ready" if group else "blocked",
                        "blocker": "" if group else "no_ready_requests",
                    }
                )
    return rows


def build_shard_manifest(root: Path, request_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in request_rows:
        grouped[row["shard_id"]].append(row)
    runtime = runtime_lookup(root)
    rows: list[dict[str, str]] = []
    for shard_id in sorted(grouped):
        group = sorted(grouped[shard_id], key=lambda row: int(row["ordinal_position"]))
        first = group[0]
        rt = runtime[(first["model"], first["core_strategy_id"], first["dataset"])]
        rows.append(
            {
                "shard_id": shard_id,
                "model": first["model"],
                "strategy_id": first["core_strategy_id"],
                "adapted_strategy_id": first["adapted_strategy_id"],
                "dataset": first["dataset"],
                "task": first["task"],
                "first_request_id": group[0]["request_id"],
                "last_request_id": group[-1]["request_id"],
                "request_count": str(len(group)),
                "estimated_runtime_seconds": fmt(float(rt["latency_mean"]) * len(group)),
                "estimated_storage_bytes": str(int(estimate_record_bytes(root, first["model"]) * len(group))),
                "raw_output_path": FUTURE_RAW_PATHS[first["model"]].as_posix(),
                "progress_path": f"results/tables/phase_f6e4b_progress_{first['model'].split(':')[0]}.csv",
                "log_path": f"results/logs/phase_f6e4b_{shard_id}.log",
                "checkpoint_policy": "flush_after_each_response_checkpoint_after_shard",
                "resume_key": "request_id",
                "status": "pending_future_execution",
            }
        )
    if any(int(row["request_count"]) > SHARD_SIZE for row in rows):
        raise RuntimeError("f6e4a_shard_size_exceeds_limit")
    if len(rows) != 64:
        raise RuntimeError(f"f6e4a_shard_count_invalid:{len(rows)}")
    return rows


def build_runtime_parameters() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for strategy in STRATEGY_ORDER:
        config = runtime_parameters(strategy)
        config_hash = sha256_text(stable_json(config))
        for key, value in config.items():
            rows.append(
                {
                    "runtime_config_id": stable_id("F6E4A-RUNTIME", strategy),
                    "strategy_id": strategy,
                    "parameter_name": key,
                    "value": json.dumps(value, ensure_ascii=True) if isinstance(value, bool) else str(value),
                    "source": "phase_f6e3b_request_manifest",
                    "classification": "adapted_runtime" if key in {"temperature", "num_predict", "stream", "timeout_seconds"} else "default_empty_unset",
                    "config_hash": config_hash,
                    "notes": "Frozen from F6E3B; not changed to improve pilot behavior.",
                }
            )
    return rows


def build_time_storage_budget(root: Path, request_rows: Sequence[Mapping[str, str]], shard_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    runtime = runtime_lookup(root)
    rows: list[dict[str, str]] = []
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in request_rows:
        grouped[(row["model"], row["core_strategy_id"], row["dataset"], row["task"])].append(row)
    for (model, strategy, dataset, task), group in sorted(grouped.items()):
        rt = runtime[(model, strategy, dataset)]
        mean_seconds = float(rt["latency_mean"])
        p50 = float(rt["latency_p50"])
        p95 = float(rt["latency_p95"])
        count = len(group)
        rows.append(
            {
                "budget_id": stable_id("F6E4A-BUDGET", model, strategy, dataset, task),
                "scope": "dataset_task_cell",
                "model": model,
                "strategy_id": strategy,
                "dataset": dataset,
                "task": task,
                "request_count": str(count),
                "estimate_kind": "observed_extrapolated" if task == "ranking" else "extrapolated_from_same_dataset_ranking_pilot",
                "seconds_per_request_mean": rt["latency_mean"],
                "seconds_per_request_p50": rt["latency_p50"],
                "seconds_per_request_p95": rt["latency_p95"],
                "best_duration_seconds": fmt(p50 * count),
                "expected_duration_seconds": fmt(mean_seconds * count),
                "worst_duration_seconds": fmt(p95 * count),
                "raw_storage_bytes": str(int(estimate_record_bytes(root, model) * count)),
                "log_storage_bytes": str(max(4096, count * 256)),
                "table_storage_bytes": str(max(4096, count * 512)),
                "minimum_free_space_bytes": str(max(1_000_000_000, int(estimate_record_bytes(root, model) * count * 10))),
                "provenance": "results/tables/phase_f6e3b_runtime_summary.csv",
            }
        )
    total_expected = sum(float(row["expected_duration_seconds"]) for row in rows)
    rows.append(
        {
            "budget_id": "F6E4A-BUDGET-TOTAL",
            "scope": "global_total",
            "model": "ALL",
            "strategy_id": "ALL",
            "dataset": "ALL",
            "task": "ALL",
            "request_count": str(len(request_rows)),
            "estimate_kind": "sum_of_dataset_task_cell_estimates",
            "seconds_per_request_mean": fmt(total_expected / len(request_rows)),
            "seconds_per_request_p50": "",
            "seconds_per_request_p95": "",
            "best_duration_seconds": fmt(sum(float(row["best_duration_seconds"]) for row in rows if row["best_duration_seconds"])),
            "expected_duration_seconds": fmt(total_expected),
            "worst_duration_seconds": fmt(sum(float(row["worst_duration_seconds"]) for row in rows if row["worst_duration_seconds"])),
            "raw_storage_bytes": str(sum(int(row["estimated_storage_bytes"]) for row in shard_rows)),
            "log_storage_bytes": str(sum(max(4096, int(row["request_count"]) * 256) for row in shard_rows)),
            "table_storage_bytes": str(max(10_000_000, len(request_rows) * 512)),
            "minimum_free_space_bytes": str(max(1_000_000_000, sum(int(row["estimated_storage_bytes"]) for row in shard_rows) * 10)),
            "provenance": "F6E3B pilot timing extrapolation",
        }
    )
    return rows


def build_stop_conditions() -> list[dict[str, str]]:
    specs = [
        ("ollama_unavailable", "local Ollama health or response endpoint unreachable before a shard", "critical", "stop_before_runtime_call", "yes", "no"),
        ("digest_mismatch", "observed model digest differs from frozen digest", "critical", "stop_before_runtime_call", "yes", "no"),
        ("model_unavailable", "required local tag missing", "critical", "stop_before_runtime_call", "yes", "no"),
        ("repeated_transport_failure", "three consecutive transport failures within a shard", "critical", "stop_shard", "yes", "no"),
        ("out_of_memory", "backend reports OOM or process exits with memory error", "critical", "stop_shard", "yes", "no"),
        ("context_overflow", "prompt rejected or truncated by context limit", "critical", "stop_shard", "yes", "no"),
        ("raw_write_failure", "append/flush/fsync fails", "critical", "stop_immediately", "yes", "no"),
        ("disk_insufficient", "free space below frozen minimum", "critical", "stop_before_shard", "yes", "no"),
        ("manifest_inconsistency", "request id, prompt hash or shard mismatch", "critical", "stop_before_runtime_call", "no", "no"),
        ("candidate_order_mismatch", "candidate_order_hash recomputation differs", "critical", "stop_before_runtime_call", "no", "no"),
        ("protected_input_change", "any frozen/protected input hash changes", "critical", "stop_before_runtime_call", "no", "no"),
        ("low_strict_valid_rate", "strict-valid rate below 70 percent after shard analysis", "high", "pause_for_review", "yes", "no"),
        ("systematic_non_json", "more than 20 percent JSON parse failures in a shard", "high", "pause_for_review", "yes", "no"),
    ]
    return [
        {
            "condition_id": stable_id("F6E4A-STOP", name),
            "condition_name": name,
            "trigger": trigger,
            "severity": severity,
            "automatic_action": action,
            "resume_allowed": resume,
            "retry_allowed": retry,
            "evidence_to_record": "raw_record_if_any;shard_id;request_id;exception;hash_state",
            "notes": "Retry must not mask systematic failure; transport retry requires a later documented phase.",
        }
        for name, trigger, severity, action, resume, retry in specs
    ]


def build_gate_plan() -> list[dict[str, str]]:
    specs = [
        ("pre_execution_freeze_gate", "before F6E4B", "all hashes match; no dirty protected input changes", "f6e4a_blocked_by_workload_inconsistency"),
        ("first_shard_qwen2_base", "after first Qwen2 Base shard", "transport ok; JSON parse not systematic failure", "f6e4a_blocked_by_runtime_budget"),
        ("qwen2_complete", "after all Qwen2 shards", "completed requests equal Qwen2 manifest requests", "f6e4a_blocked_by_runtime_budget"),
        ("first_shard_mistral_base", "after first Mistral Base shard", "transport ok; JSON parse not systematic failure", "f6e4a_blocked_by_runtime_budget"),
        ("mistral_complete", "after all Mistral shards", "completed requests equal Mistral manifest requests", "f6e4a_blocked_by_runtime_budget"),
        ("analysis_only_final", "after F6E4B execution", "derived outputs deterministic; raw append-only", "f6e4a_blocked_by_nondeterminism"),
    ]
    return [
        {
            "gate_id": stable_id("F6E4A-GATE", name),
            "gate_name": name,
            "phase_point": point,
            "expected": expected,
            "status": "planned_not_executed_in_f6e4a",
            "evidence": "future F6E4B runtime/progress artifacts",
            "blocking_decision_if_fail": decision,
            "notes": "F6E4A defines but does not execute this gate.",
        }
        for name, point, expected, decision in specs
    ]


def build_missing_or_blocked(workloads: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in workloads:
        if row["status"] != "ready":
            rows.append(
                {
                    "block_id": stable_id("F6E4A-BLOCK", row["dataset"], row["task"], row["blocker"]),
                    "category": "workload",
                    "affected_scope": f"{row['dataset']} {row['task']}",
                    "status": row["blocker"],
                    "severity": "high",
                    "evidence": row["source_manifest"],
                    "safe_next_action": "Keep excluded from F6E4B unless a separate phase resolves comparability.",
                    "notes": row["notes"],
                }
            )
        elif int(row["excluded_contexts"]):
            rows.append(
                {
                    "block_id": stable_id("F6E4A-EXCLUDE", row["dataset"], row["task"]),
                    "category": "partial_context_exclusion",
                    "affected_scope": f"{row['dataset']} {row['task']}",
                    "status": "excluded_not_prompt_compatible",
                    "severity": "medium",
                    "evidence": row["source_manifest"],
                    "safe_next_action": "Do not repair titles/history in F6E4A; use only ready contexts.",
                    "notes": f"excluded_contexts={row['excluded_contexts']}",
                }
            )
    rows.extend(
        [
            {
                "block_id": "F6E4A-BLOCK-LLMSREC",
                "category": "scope",
                "affected_scope": "LLMSRec",
                "status": "not_authorized",
                "severity": "info",
                "evidence": "F6E4A scope",
                "safe_next_action": "Design separate reconstructed-strategy phase if needed.",
                "notes": "LLMSRec remains excluded.",
            },
            {
                "block_id": "F6E4A-BLOCK-JSON-INDEX-STANDALONE",
                "category": "scope",
                "affected_scope": "JSON/index standalone",
                "status": "not_authorized",
                "severity": "info",
                "evidence": "F6E4A scope",
                "safe_next_action": "Keep as separate local ablation.",
                "notes": "F6E4A uses adapted Base/LLMRank only.",
            },
            {
                "block_id": "F6E4A-BLOCK-REMOTE-AND-LARGE-MODELS",
                "category": "scope",
                "affected_scope": "Llama3/GPT/Claude",
                "status": "not_authorized",
                "severity": "high",
                "evidence": "F6E4A scope",
                "safe_next_action": "Do not execute without explicit future phase.",
                "notes": "No model pull or remote API.",
            },
        ]
    )
    return rows


def protected_summary(root: Path) -> dict[str, int]:
    f2_rows = read_csv_rows(root, F2_PROTECTED_VALIDATION)
    f2_changed = sum(1 for row in f2_rows if row.get("status") != "unchanged")
    f6d4a_rows = read_csv_rows(root, F6D4A_PROTECTED_VERIFICATION)
    f6d4a_changed = sum(int(row.get("changed", "0") or 0) for row in f6d4a_rows)
    return {"f2_checked": len(f2_rows), "f2_changed": f2_changed, "f6d4a_groups_checked": len(f6d4a_rows), "f6d4a_changed": f6d4a_changed}


def input_hashes(root: Path) -> dict[str, str]:
    paths = [
        F6E1_FREEZE,
        F6E2_FREEZE,
        F6E3_FREEZE,
        F6E3A_FREEZE,
        F6E3B_FREEZE,
        F6E1_PROMPT_MODULE,
        F6E3B_MODULE,
        CANONICAL_FULL_MANIFEST,
        CANONICAL_FULL_REQUESTS,
        CANONICAL_COUNT_RECONCILIATION,
        CANONICAL_PREFLIGHT_DECISION,
        F6D3_LASTFM_RANKING_MANIFEST,
        F6D3_LASTFM_RERANKING_MANIFEST,
        F6C_LASTFM_RERANKING_READINESS,
        F6D3_GO_NO_GO,
        TITLE_MAP,
        F2_PROTECTED_VALIDATION,
        F6D4A_PROTECTED_VERIFICATION,
        F6D4A_INPUT_HASHES,
        Path("results/tables/phase_f6e3b_runtime_summary.csv"),
    ]
    paths.extend(PLAN_E_SOURCE_MANIFESTS.values())
    paths.extend(Path(path) for path in F6D4A_RAW_HASHES)
    paths.extend(Path(path) for path in F6E3_RAW_HASHES)
    paths.extend(Path(path) for path in F6E3B_RAW_HASHES)
    unique = sorted({path.as_posix() for path in paths})
    return {path: file_sha256_or_missing(root, Path(path)) for path in unique}


def table_hashes(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, str]:
    return {name: sha256_text(csv_text(rows, TABLE_COLUMNS[name])) for name, rows in sorted(tables.items())}


def final_decision(tables: Mapping[str, Sequence[Mapping[str, str]]], protected: Mapping[str, int]) -> str:
    ready = [row for row in tables["workload_eligibility"] if row["status"] == "ready"]
    requests = tables["request_manifest"]
    shards = tables["shard_manifest"]
    if protected["f2_changed"] or protected["f6d4a_changed"]:
        return "f6e4a_blocked_by_workload_inconsistency"
    if len(ready) != 7 or len(requests) != EXPECTED_REQUESTS or len(shards) != 64:
        return "f6e4a_blocked_by_workload_inconsistency"
    if any(int(row["request_count"]) > SHARD_SIZE for row in shards):
        return "f6e4a_blocked_by_workload_inconsistency"
    return "f6e4a_adapted_full_run_freeze_complete"


def build_manifest(root: Path, tables: Mapping[str, Sequence[Mapping[str, str]]], repo: Mapping[str, str], protected: Mapping[str, int]) -> dict[str, Any]:
    hashes = table_hashes(tables)
    inputs = input_hashes(root)
    decision = final_decision(tables, protected)
    content = {
        "phase": PHASE_ID,
        "decision": decision,
        "input_hashes": inputs,
        "table_hashes": hashes,
        "total_logical_requests": len(tables["request_manifest"]),
        "total_shards": len(tables["shard_manifest"]),
        "protected_summary": protected,
    }
    return {
        "phase": PHASE_ID,
        "decision": decision,
        "frozen_timestamp_utc": FROZEN_TIMESTAMP_UTC,
        "timestamp_excluded_from_scientific_hash": True,
        "repository_state": dict(repo),
        "input_hashes": inputs,
        "f6e3b_scientific_content_hash": EXPECTED_F6E3B_SCIENTIFIC_HASH,
        "model_digests": EXPECTED_MODEL_DIGESTS,
        "workload_hashes": {
            "workload_eligibility": hashes["workload_eligibility"],
            "execution_matrix": hashes["execution_matrix"],
        },
        "prompt_and_adapter_hashes": {
            "f6e1_prompt_module": inputs.get(F6E1_PROMPT_MODULE.as_posix(), "missing"),
            "f6e3b_adapter_module": inputs.get(F6E3B_MODULE.as_posix(), "missing"),
        },
        "request_manifest_hash": hashes["request_manifest"],
        "shard_manifest_hash": hashes["shard_manifest"],
        "runtime_configuration_hash": hashes["runtime_parameters"],
        "total_source_contexts": EXPECTED_SOURCE_CONTEXTS,
        "total_logical_requests": len(tables["request_manifest"]),
        "total_expected_calls": len(tables["request_manifest"]),
        "total_shards": len(tables["shard_manifest"]),
        "scientific_content_hash": sha256_text(stable_json(content)),
        "protected_summary": protected,
        "future_raw_paths": {model: path.as_posix() for model, path in FUTURE_RAW_PATHS.items()},
    }


def build_report(manifest: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, str]]]) -> str:
    matrix = tables["execution_matrix"]
    total_seconds = sum(float(row["estimated_duration_seconds"]) for row in matrix)
    sections = [
        ("Initial repository state", stable_json(manifest["repository_state"])),
        ("Frozen-input verification", f"Input hashes recorded: {len(manifest['input_hashes'])}; protected summary: {stable_json(manifest['protected_summary'])}."),
        ("Workload eligibility", f"Ready workloads: {sum(1 for row in tables['workload_eligibility'] if row['status'] == 'ready')}; source contexts: {manifest['total_source_contexts']}."),
        ("Excluded workloads", f"Blocked/partial rows: {len(tables['missing_or_blocked'])}. LastFM reranking is excluded."),
        ("Full-run matrix", f"Matrix cells: {len(matrix)} dataset/task/model/strategy rows."),
        ("Request counts", f"Total expected calls: {manifest['total_expected_calls']}."),
        ("Request manifest", f"Hash: {manifest['request_manifest_hash']}."),
        ("Shard design", f"Shards: {manifest['total_shards']}; max shard size {SHARD_SIZE}."),
        ("Runtime parameters", f"Runtime configuration hash: {manifest['runtime_configuration_hash']}."),
        ("Invalid-output policy", "No automatic retry for non-strict outputs; no truncation, completion, duplicate repair, or index substitution."),
        ("Raw/progress/resume contract", "Future raw files are model-separated, append-only, flushed after each response, and resumed by request_id."),
        ("Stop conditions", f"Stop conditions frozen: {len(tables['stop_conditions'])}."),
        ("Execution order", "Qwen2 Base, Qwen2 LLMRank, Mistral Base, Mistral LLMRank; dataset/task order is frozen in the request manifest."),
        ("Time and storage budget", f"Expected duration seconds: {fmt(total_seconds)}; estimates are extrapolated from F6E3B pilot timing."),
        ("Implementation files", "src/analysis/plan_f6e4a_adapted_full_run_freeze.py; src/experiments/build_phase_f6e4a_adapted_full_run_freeze.py."),
        ("Tests added", "tests/test_plan_f6e4a_adapted_full_run_freeze.py."),
        ("Targeted tests", "Recorded in final operator response after execution."),
        ("Full-suite result", "Recorded in final operator response after execution."),
        ("Determinism", "Builder uses frozen timestamps and deterministic table ordering; timestamp is excluded from scientific hash."),
        ("Protected-input verification", stable_json(manifest["protected_summary"])),
        ("Full-run freeze manifest", MANIFEST_PATH.as_posix()),
        ("Missing or blocked evidence", f"Rows: {len(tables['missing_or_blocked'])}."),
        ("Final decision", f"`{manifest['decision']}`."),
        ("Final git state", "Recorded in final operator response; no staging or commit is performed by F6E4A."),
        ("Files created or modified", "Only phase_f6e4a_* report, tables, manifest, source module, CLI and tests."),
        ("Remaining limitations", "No inference, no Ollama gate, and no full-run execution are performed in F6E4A."),
        ("Safe next action", "Request explicit F6E4B execution approval if the freeze remains complete."),
    ]
    lines = ["# Phase F6E4A - Adapted Full-Run Manifest, Workload And Execution Freeze", "", f"Decision: `{manifest['decision']}`.", ""]
    for index, (title, body) in enumerate(sections, start=1):
        lines.extend([f"## {index}. {title}", "", body, ""])
    return "\n".join(lines)


def build_all(root: Path) -> BuildOutputs:
    verify_expected_hashes(root)
    repo = repository_state(root)
    contexts = build_source_contexts(root)
    request_rows = build_request_manifest(root, contexts)
    workload_rows = build_workload_eligibility(root, contexts)
    shard_rows = build_shard_manifest(root, request_rows)
    tables = {
        "workload_eligibility": workload_rows,
        "execution_matrix": build_execution_matrix(root, request_rows),
        "request_manifest": request_rows,
        "shard_manifest": shard_rows,
        "runtime_parameters": build_runtime_parameters(),
        "time_storage_budget": build_time_storage_budget(root, request_rows, shard_rows),
        "stop_conditions": build_stop_conditions(),
        "gate_plan": build_gate_plan(),
        "missing_or_blocked": build_missing_or_blocked(workload_rows),
    }
    protected = protected_summary(root)
    manifest = build_manifest(root, tables, repo, protected)
    report = build_report(manifest, tables)
    return BuildOutputs(tables=tables, manifest=manifest, report=report)


def write_outputs(outputs: BuildOutputs, root: Path) -> dict[str, str]:
    written: dict[str, str] = {}
    for table_name, relative_path in OUTPUT_TABLES.items():
        write_csv(root / relative_path, outputs.tables[table_name], TABLE_COLUMNS[table_name])
        written[relative_path.as_posix()] = sha256_file(root / relative_path)
    (root / MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_PATH).write_text(json.dumps(outputs.manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    written[MANIFEST_PATH.as_posix()] = sha256_file(root / MANIFEST_PATH)
    (root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_PATH).write_text(outputs.report, encoding="utf-8")
    written[REPORT_PATH.as_posix()] = sha256_file(root / REPORT_PATH)
    return written


def run(root: Path) -> dict[str, str]:
    return write_outputs(build_all(root), root)


def output_hashes(root: Path) -> dict[str, str]:
    paths = [REPORT_PATH, MANIFEST_PATH, *OUTPUT_TABLES.values()]
    return {path.as_posix(): sha256_file(root / path) for path in paths if (root / path).exists()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase F6E4A adapted full-run freeze artifacts.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    written = run(args.root)
    print(json.dumps({"written": written}, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
